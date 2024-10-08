import Mathlib

namespace log_expression_l235_235925

theorem log_expression :
  (Real.log 2)^2 + Real.log 2 * Real.log 5 + Real.log 5 = 1 := by
  sorry

end log_expression_l235_235925


namespace max_value_expression_l235_235943

theorem max_value_expression (k : ℕ) (a b c : ℝ) (h : k > 0) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (habc : a + b + c = 3 * k) :
  a^(3 * k - 1) * b + b^(3 * k - 1) * c + c^(3 * k - 1) * a + k^2 * a^k * b^k * c^k ≤ (3 * k - 1)^(3 * k - 1) :=
sorry

end max_value_expression_l235_235943


namespace log_ordering_l235_235054

theorem log_ordering {x a b c : ℝ} (h1 : 1 < x) (h2 : x < 10) (ha : a = Real.log x^2) (hb : b = Real.log (Real.log x)) (hc : c = (Real.log x)^2) :
  a > c ∧ c > b :=
by
  sorry

end log_ordering_l235_235054


namespace foci_of_ellipse_l235_235733

-- Define the ellipsis
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 25) = 1

-- Prove the coordinates of foci of the ellipse
theorem foci_of_ellipse :
  ∃ c : ℝ, c = 3 ∧ ((0, c) ∈ {p : ℝ × ℝ | ellipse p.1 p.2} ∧ (0, -c) ∈ {p : ℝ × ℝ | ellipse p.1 p.2}) :=
by
  sorry

end foci_of_ellipse_l235_235733


namespace grace_is_14_l235_235429

def GraceAge (G F C E D : ℕ) : Prop :=
  G = F - 6 ∧ F = C + 2 ∧ E = C + 3 ∧ D = E - 4 ∧ D = 17

theorem grace_is_14 (G F C E D : ℕ) (h : GraceAge G F C E D) : G = 14 :=
by sorry

end grace_is_14_l235_235429


namespace sin_cos_unique_solution_l235_235451

theorem sin_cos_unique_solution (α : ℝ) (hα1 : 0 < α) (hα2 : α < (π / 2)) :
  ∃! x : ℝ, (Real.sin α) ^ x + (Real.cos α) ^ x = 1 :=
sorry

end sin_cos_unique_solution_l235_235451


namespace smallest_value_square_l235_235918

theorem smallest_value_square (z : ℂ) (hz : z.re > 0) (A : ℝ) :
  (A = 24 / 25) →
  abs ((Complex.abs z + 1 / Complex.abs z)^2 - (2 - 14 / 25)) = 0 :=
by
  sorry

end smallest_value_square_l235_235918


namespace perimeter_of_square_l235_235867

theorem perimeter_of_square
  (s : ℝ) -- s is the side length of the square
  (h_divided_rectangles : ∀ r, r ∈ {r : ℝ × ℝ | r = (s, s / 6)} → true) -- the square is divided into six congruent rectangles
  (h_perimeter_rect : 2 * (s + s / 6) = 42) -- the perimeter of each of these rectangles is 42 inches
  : 4 * s = 72 := 
sorry

end perimeter_of_square_l235_235867


namespace carla_total_time_l235_235971

def time_sharpening : ℝ := 15
def time_peeling : ℝ := 3 * time_sharpening
def time_chopping : ℝ := 0.5 * time_peeling
def time_breaks : ℝ := 2 * 5

def total_time : ℝ :=
  time_sharpening + time_peeling + time_chopping + time_breaks

theorem carla_total_time : total_time = 92.5 :=
by sorry

end carla_total_time_l235_235971


namespace arithmetic_seq_sum_l235_235335

theorem arithmetic_seq_sum (a_n : ℕ → ℝ) (h_arith_seq : ∃ d, ∀ n, a_n (n + 1) = a_n n + d)
    (h_sum : a_n 5 + a_n 8 = 24) : a_n 6 + a_n 7 = 24 := by
  sorry

end arithmetic_seq_sum_l235_235335


namespace driver_speed_l235_235389

theorem driver_speed (v t : ℝ) (h1 : t > 0) (h2 : v > 0) (h3 : v * t = (v + 37.5) * (3 / 8) * t) : v = 22.5 :=
by
  sorry

end driver_speed_l235_235389


namespace average_first_n_numbers_eq_10_l235_235718

theorem average_first_n_numbers_eq_10 (n : ℕ) 
  (h : (n * (n + 1)) / (2 * n) = 10) : n = 19 :=
  sorry

end average_first_n_numbers_eq_10_l235_235718


namespace cary_needs_six_weekends_l235_235651

theorem cary_needs_six_weekends
  (shoe_cost : ℕ)
  (saved : ℕ)
  (earn_per_lawn : ℕ)
  (lawns_per_weekend : ℕ)
  (additional_needed : ℕ := shoe_cost - saved)
  (earn_per_weekend : ℕ := earn_per_lawn * lawns_per_weekend)
  (weekends_needed : ℕ := additional_needed / earn_per_weekend) :
  shoe_cost = 120 ∧ saved = 30 ∧ earn_per_lawn = 5 ∧ lawns_per_weekend = 3 → weekends_needed = 6 := by 
  sorry

end cary_needs_six_weekends_l235_235651


namespace B_contribution_to_capital_l235_235238

theorem B_contribution_to_capital (A_capital : ℝ) (A_months : ℝ) (B_months : ℝ) (profit_ratio_A : ℝ) (profit_ratio_B : ℝ) (B_contribution : ℝ) :
  A_capital = 4500 →
  A_months = 12 →
  B_months = 5 →
  profit_ratio_A = 2 →
  profit_ratio_B = 3 →
  B_contribution = (4500 * 12 * 3) / (5 * 2) → 
  B_contribution = 16200 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end B_contribution_to_capital_l235_235238


namespace apple_juice_production_l235_235307

noncomputable def apple_usage 
  (total_apples : ℝ) 
  (mixed_percentage : ℝ) 
  (juice_percentage : ℝ) 
  (sold_fresh_percentage : ℝ) : ℝ := 
  let mixed_apples := total_apples * mixed_percentage / 100
  let remainder_apples := total_apples - mixed_apples
  let juice_apples := remainder_apples * juice_percentage / 100
  juice_apples

theorem apple_juice_production :
  apple_usage 6 20 60 40 = 2.9 := 
by
  sorry

end apple_juice_production_l235_235307


namespace find_x_correct_l235_235026

theorem find_x_correct (x : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * 180 = 360 * x - 480)
  (h2 : (x + 4) + x + (x - 4) = 3 * x)
  (h3 : 100 = (360 * x - 480) / (3 * x)) : 
  x = 8 := 
sorry

end find_x_correct_l235_235026


namespace trajectory_parabola_l235_235827

noncomputable def otimes (x1 x2 : ℝ) : ℝ := (x1 + x2)^2 - (x1 - x2)^2

theorem trajectory_parabola (x : ℝ) (h : 0 ≤ x) : 
  ∃ (y : ℝ), y^2 = 8 * x ∧ (∀ P : ℝ × ℝ, P = (x, y) → (P.snd^2 = 8 * P.fst)) :=
by
  sorry

end trajectory_parabola_l235_235827


namespace volume_inhaled_per_breath_is_correct_l235_235584

def breaths_per_minute : ℤ := 17
def volume_inhaled_24_hours : ℤ := 13600
def minutes_per_hour : ℤ := 60
def hours_per_day : ℤ := 24

def total_minutes_24_hours : ℤ := hours_per_day * minutes_per_hour
def total_breaths_24_hours : ℤ := total_minutes_24_hours * breaths_per_minute
def volume_per_breath := (volume_inhaled_24_hours : ℚ) / (total_breaths_24_hours : ℚ)

theorem volume_inhaled_per_breath_is_correct :
  volume_per_breath = 0.5556 := by
  sorry

end volume_inhaled_per_breath_is_correct_l235_235584


namespace floral_arrangement_carnations_percentage_l235_235979

theorem floral_arrangement_carnations_percentage :
  ∀ (F : ℕ),
  (1 / 4) * (7 / 10) * F + (2 / 3) * (3 / 10) * F = (29 / 40) * F :=
by
  sorry

end floral_arrangement_carnations_percentage_l235_235979


namespace find_c_and_general_formula_l235_235610

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

end find_c_and_general_formula_l235_235610


namespace scientific_notation_of_concentration_l235_235647

theorem scientific_notation_of_concentration :
  0.000042 = 4.2 * 10^(-5) :=
sorry

end scientific_notation_of_concentration_l235_235647


namespace replace_asterisks_l235_235917

theorem replace_asterisks (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end replace_asterisks_l235_235917


namespace min_value_fraction_subtraction_l235_235633

theorem min_value_fraction_subtraction
  (a b : ℝ)
  (ha : 0 < a ∧ a ≤ 3 / 4)
  (hb : 0 < b ∧ b ≤ 3 - a)
  (hineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → a * x + b - 3 ≤ 0) :
  ∃ a b, (0 < a ∧ a ≤ 3 / 4) ∧ (0 < b ∧ b ≤ 3 - a) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → a * x + b - 3 ≤ 0) ∧ (1 / a - b = 1) :=
by 
  sorry

end min_value_fraction_subtraction_l235_235633


namespace estimate_first_year_students_l235_235492

noncomputable def number_of_first_year_students (N : ℕ) : Prop :=
  let p1 := (N - 90) / N
  let p2 := (N - 100) / N
  let p_both := 1 - p1 * p2
  p_both = 20 / N → N = 450

theorem estimate_first_year_students : ∃ N : ℕ, number_of_first_year_students N :=
by
  use 450
  -- sorry added to skip the proof part
  sorry

end estimate_first_year_students_l235_235492


namespace polynomial_value_at_neg2_l235_235004

def polynomial (x : ℝ) : ℝ :=
  x^6 - 5 * x^5 + 6 * x^4 + x^2 + 0.3 * x + 2

theorem polynomial_value_at_neg2 :
  polynomial (-2) = 325.4 :=
by
  sorry

end polynomial_value_at_neg2_l235_235004


namespace cone_cannot_have_rectangular_cross_section_l235_235425

noncomputable def solid := Type

def is_cylinder (s : solid) : Prop := sorry
def is_cone (s : solid) : Prop := sorry
def is_rectangular_prism (s : solid) : Prop := sorry
def is_cube (s : solid) : Prop := sorry

def has_rectangular_cross_section (s : solid) : Prop := sorry

axiom cylinder_has_rectangular_cross_section (s : solid) : is_cylinder s → has_rectangular_cross_section s
axiom rectangular_prism_has_rectangular_cross_section (s : solid) : is_rectangular_prism s → has_rectangular_cross_section s
axiom cube_has_rectangular_cross_section (s : solid) : is_cube s → has_rectangular_cross_section s

theorem cone_cannot_have_rectangular_cross_section (s : solid) : is_cone s → ¬has_rectangular_cross_section s := 
sorry

end cone_cannot_have_rectangular_cross_section_l235_235425


namespace rectangle_area_l235_235112

-- Define the width and length of the rectangle
def w : ℚ := 20 / 3
def l : ℚ := 2 * w

-- Define the perimeter constraint
def perimeter_condition : Prop := 2 * (l + w) = 40

-- Define the area of the rectangle
def area : ℚ := l * w

-- The theorem to prove
theorem rectangle_area : perimeter_condition → area = 800 / 9 :=
by
  intro h
  have hw : w = 20 / 3 := rfl
  have hl : l = 2 * w := rfl
  have hp : 2 * (l + w) = 40 := h
  sorry

end rectangle_area_l235_235112


namespace combined_cost_increase_l235_235157

def original_bicycle_cost : ℝ := 200
def original_skates_cost : ℝ := 50
def bike_increase_percent : ℝ := 0.06
def skates_increase_percent : ℝ := 0.15

noncomputable def new_bicycle_cost : ℝ := original_bicycle_cost * (1 + bike_increase_percent)
noncomputable def new_skates_cost : ℝ := original_skates_cost * (1 + skates_increase_percent)
noncomputable def original_total_cost : ℝ := original_bicycle_cost + original_skates_cost
noncomputable def new_total_cost : ℝ := new_bicycle_cost + new_skates_cost
noncomputable def total_increase : ℝ := new_total_cost - original_total_cost
noncomputable def percent_increase : ℝ := (total_increase / original_total_cost) * 100

theorem combined_cost_increase : percent_increase = 7.8 := by
  sorry

end combined_cost_increase_l235_235157


namespace vehicles_count_l235_235269

theorem vehicles_count (T : ℕ) : 
    2 * T + 3 * (2 * T) + (T / 2) + T = 180 → 
    T = 19 ∧ 2 * T = 38 ∧ 3 * (2 * T) = 114 ∧ (T / 2) = 9 := 
by 
    intros h
    sorry

end vehicles_count_l235_235269


namespace quadratic_passing_point_l235_235900

theorem quadratic_passing_point :
  ∃ (m : ℝ), (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = 8 → x = 0) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = -10 → x = -1) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = m → x = 5) →
  m = 638 := by
  sorry

end quadratic_passing_point_l235_235900


namespace sum_first_40_terms_l235_235921

-- Defining the sequence a_n following the given conditions
noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 3
| n + 2 => a (n + 1) * a (n - 1)

-- Defining the sum of the first 40 terms of the sequence
noncomputable def S40 := (Finset.range 40).sum a

-- The theorem stating the desired property
theorem sum_first_40_terms : S40 = 60 :=
sorry

end sum_first_40_terms_l235_235921


namespace geometric_sequence_sum_ratio_l235_235738

noncomputable def a (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * q^n

-- Sum of the first 'n' terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a1 q : ℝ) 
  (h : 8 * (a 11 a1 q) = (a 14 a1 q)) :
  (S 4 a1 q) / (S 2 a1 q) = 5 :=
by
  sorry

end geometric_sequence_sum_ratio_l235_235738


namespace fraction_of_As_l235_235233

-- Define the conditions
def fraction_B (T : ℕ) := 1/4 * T
def fraction_C (T : ℕ) := 1/2 * T
def remaining_D : ℕ := 20
def total_students_approx : ℕ := 400

-- State the theorem
theorem fraction_of_As 
  (F : ℚ) : 
  ∀ T : ℕ, 
  T = F * T + fraction_B T + fraction_C T + remaining_D → 
  T = total_students_approx → 
  F = 1/5 :=
by
  intros
  sorry

end fraction_of_As_l235_235233


namespace sum_third_three_l235_235401

variables {a : ℕ → ℤ}

-- Define the properties of the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

-- Given conditions
axiom sum_first_three : a 1 + a 2 + a 3 = 9
axiom sum_second_three : a 4 + a 5 + a 6 = 27
axiom arithmetic_seq : is_arithmetic_sequence a

-- The proof goal
theorem sum_third_three : a 7 + a 8 + a 9 = 45 :=
by
  sorry  -- Proof is omitted here

end sum_third_three_l235_235401


namespace money_taken_l235_235276

def total_people : ℕ := 6
def cost_per_soda : ℝ := 0.5
def cost_per_pizza : ℝ := 1.0

theorem money_taken (total_people cost_per_soda cost_per_pizza : ℕ × ℝ × ℝ ) :
  total_people * cost_per_soda + total_people * cost_per_pizza = 9 := by
  sorry

end money_taken_l235_235276


namespace simplify_fraction_l235_235063

theorem simplify_fraction : (8 / (5 * 42) = 4 / 105) :=
by
    sorry

end simplify_fraction_l235_235063


namespace divisible_iff_condition_l235_235224

theorem divisible_iff_condition (a b : ℤ) : 
  (13 ∣ (2 * a + 3 * b)) ↔ (13 ∣ (2 * b - 3 * a)) :=
  sorry

end divisible_iff_condition_l235_235224


namespace find_x_l235_235809

-- Definitions of the conditions
def eq1 (x y z : ℕ) : Prop := x + y + z = 25
def eq2 (y z : ℕ) : Prop := y + z = 14

-- Statement of the mathematically equivalent proof problem
theorem find_x (x y z : ℕ) (h1 : eq1 x y z) (h2 : eq2 y z) : x = 11 :=
by {
  -- This is where the proof would go, but we can omit it for now:
  sorry
}

end find_x_l235_235809


namespace number_of_tangent_and_parallel_lines_l235_235830

theorem number_of_tangent_and_parallel_lines (p : ℝ × ℝ) (a : ℝ) (h : p = (2, 4)) (hp_on_parabola : (p.1)^2 = 8 * p.2) :
  ∃ l1 l2 : (ℝ × ℝ) → Prop, 
    (l1 (2, 4) ∧ l2 (2, 4)) ∧ 
    (∀ l, (l = l1 ∨ l = l2) ↔ (∃ q, q ≠ p ∧ q ∈ {p' | (p'.1)^2 = 8 * p'.2})) ∧ 
    (∀ p' ∈ {p' | (p'.1)^2 = 8 * p'.2}, (l1 p' ∨ l2 p') → False) :=
sorry

end number_of_tangent_and_parallel_lines_l235_235830


namespace probability_is_seven_fifteenths_l235_235797

-- Define the problem conditions
def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def choose_3_from_10 : ℕ := Nat.choose 10 3
def choose_3_red : ℕ := Nat.choose 5 3
def choose_3_green : ℕ := Nat.choose 3 3
def choose_2_red_1_green : ℕ := Nat.choose 5 2 * Nat.choose 3 1
def choose_2_green_1_red : ℕ := Nat.choose 3 2 * Nat.choose 5 1

-- Calculate favorable outcomes
def favorable_outcomes : ℕ :=
  choose_3_red + choose_3_green + choose_2_red_1_green + choose_2_green_1_red

-- Calculate the required probability
def probability_all_red_or_green : ℚ := favorable_outcomes / choose_3_from_10

-- Prove that probability_all_red_or_green is 7/15
theorem probability_is_seven_fifteenths :
  probability_all_red_or_green = 7 / 15 :=
by 
  -- Leaving the proof as a sorry for now
  sorry

end probability_is_seven_fifteenths_l235_235797


namespace anne_cleaning_time_l235_235851

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l235_235851


namespace translate_parabola_l235_235819

theorem translate_parabola :
  (∀ x : ℝ, (y : ℝ) = 6 * x^2 -> y = 6 * (x + 2)^2 + 3) :=
by
  sorry

end translate_parabola_l235_235819


namespace eval_expression_l235_235375

theorem eval_expression : 3^13 / 3^3 + 2^3 = 59057 := by
  sorry

end eval_expression_l235_235375


namespace train_crossing_platform_time_l235_235373

theorem train_crossing_platform_time (train_length : ℝ) (platform_length : ℝ) (time_cross_post : ℝ) :
  train_length = 300 → platform_length = 350 → time_cross_post = 18 → 
  (train_length + platform_length) / (train_length / time_cross_post) = 39 :=
by
  intros
  sorry

end train_crossing_platform_time_l235_235373


namespace shaded_area_l235_235923

theorem shaded_area (PQ : ℝ) (n_squares : ℕ) (d_intersect : ℝ)
  (h1 : PQ = 8) (h2 : n_squares = 20) (h3 : d_intersect = 8) : ∃ (A : ℝ), A = 160 := 
by {
  sorry
}

end shaded_area_l235_235923


namespace proposition_not_true_3_l235_235542

theorem proposition_not_true_3 (P : ℕ → Prop) (h1 : ∀ n, P n → P (n + 1)) (h2 : ¬ P 4) : ¬ P 3 :=
by
  sorry

end proposition_not_true_3_l235_235542


namespace butterfly_1023_distance_l235_235928

noncomputable def omega : Complex := Complex.exp (Complex.I * Real.pi / 4)

noncomputable def Q (n : ℕ) : Complex :=
  match n with
  | 0     => 0
  | k + 1 => Q k + (k + 1) * omega ^ k

noncomputable def butterfly_distance (n : ℕ) : ℝ := Complex.abs (Q n)

theorem butterfly_1023_distance : butterfly_distance 1023 = 511 * Real.sqrt (2 + Real.sqrt 2) :=
  sorry

end butterfly_1023_distance_l235_235928


namespace sin_2x_equals_neg_61_div_72_l235_235609

variable (x y : Real)
variable (h1 : Real.sin y = (3 / 2) * Real.sin x + (2 / 3) * Real.cos x)
variable (h2 : Real.cos y = (2 / 3) * Real.sin x + (3 / 2) * Real.cos x)

theorem sin_2x_equals_neg_61_div_72 : Real.sin (2 * x) = -61 / 72 :=
by
  -- Proof goes here
  sorry

end sin_2x_equals_neg_61_div_72_l235_235609


namespace remainder_division_123456789012_by_112_l235_235672

-- Define the conditions
def M : ℕ := 123456789012
def m7 : ℕ := M % 7
def m16 : ℕ := M % 16

-- State the proof problem
theorem remainder_division_123456789012_by_112 : M % 112 = 76 :=
by
  -- Conditions
  have h1 : m7 = 3 := by sorry
  have h2 : m16 = 12 := by sorry
  -- Conclusion
  sorry

end remainder_division_123456789012_by_112_l235_235672


namespace solution_set_system_of_inequalities_l235_235017

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l235_235017


namespace max_neg_p_l235_235250

theorem max_neg_p (p : ℤ) (h1 : p < 0) (h2 : ∃ k : ℤ, 2001 + p = k^2) : p ≤ -65 :=
by
  sorry

end max_neg_p_l235_235250


namespace integer_ratio_zero_l235_235193

theorem integer_ratio_zero
  (A B : ℤ)
  (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -1 → (A / (x - 3 : ℝ) + B / (x ^ 2 + 2 * x + 1) = (x ^ 3 - x ^ 2 + 3 * x + 1) / (x ^ 3 - x - 3))) :
  B / A = 0 :=
sorry

end integer_ratio_zero_l235_235193


namespace simplify_expression_l235_235245

theorem simplify_expression (x : ℝ) (hx : x ≠ 4):
  (x^2 - 4 * x) / (x^2 - 8 * x + 16) = x / (x - 4) :=
by sorry

end simplify_expression_l235_235245


namespace solve_for_x_l235_235826

-- Define the quadratic equation condition
def quadratic_eq (x : ℝ) : Prop := 3 * x^2 - 7 * x - 6 = 0

-- The main theorem to prove
theorem solve_for_x (x : ℝ) : x > 0 ∧ quadratic_eq x → x = 3 := by
  sorry

end solve_for_x_l235_235826


namespace M_lt_N_l235_235458

variables (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def N : ℝ := |a + b + c| + |2 * a - b|
def M : ℝ := |a - b + c| + |2 * a + b|

axiom h1 : f 1 < 0  -- a + b + c < 0
axiom h2 : f (-1) > 0  -- a - b + c > 0
axiom h3 : a > 0
axiom h4 : -b / (2 * a) > 1

theorem M_lt_N : M a b c < N a b c :=
by
  sorry

end M_lt_N_l235_235458


namespace length_of_room_l235_235164

theorem length_of_room (L : ℝ) (w : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) (room_area : ℝ) :
  w = 12 →
  veranda_width = 2 →
  veranda_area = 144 →
  (L + 2 * veranda_width) * (w + 2 * veranda_width) - L * w = veranda_area →
  L = 20 :=
by
  intro h_w
  intro h_veranda_width
  intro h_veranda_area
  intro h_area_eq
  sorry

end length_of_room_l235_235164


namespace rectangle_and_square_problems_l235_235286

theorem rectangle_and_square_problems :
  ∃ (length width : ℝ), 
    (length / width = 2) ∧ 
    (length * width = 50) ∧ 
    (length = 10) ∧
    (width = 5) ∧
    ∃ (side_length : ℝ), 
      (side_length ^ 2 = 50) ∧ 
      (side_length - width = 5 * (Real.sqrt 2 - 1)) := 
by
  sorry

end rectangle_and_square_problems_l235_235286


namespace swap_values_l235_235171

theorem swap_values : ∀ (a b : ℕ), a = 3 → b = 2 → 
  (∃ c : ℕ, c = b ∧ (b = a ∧ (a = c ∨ a = 2 ∧ b = 3))) :=
by
  sorry

end swap_values_l235_235171


namespace typist_speeds_l235_235648

noncomputable def num_pages : ℕ := 72
noncomputable def ratio : ℚ := 6 / 5
noncomputable def time_difference : ℚ := 1.5

theorem typist_speeds :
  ∃ (x y : ℚ), (x = 9.6 ∧ y = 8) ∧ 
                (num_pages / x - num_pages / y = time_difference) ∧
                (x / y = ratio) :=
by
  -- Let's skip the proof for now
  sorry

end typist_speeds_l235_235648


namespace range_of_m_l235_235422

theorem range_of_m (m y1 y2 k : ℝ) (h1 : y1 = -2 * (m - 2) ^ 2 + k) (h2 : y2 = -2 * (m - 1) ^ 2 + k) (h3 : y1 > y2) : m > 3 / 2 := 
sorry

end range_of_m_l235_235422


namespace sean_total_cost_l235_235794

noncomputable def total_cost (soda_cost soup_cost sandwich_cost : ℕ) (num_soda num_soup num_sandwich : ℕ) : ℕ :=
  num_soda * soda_cost + num_soup * soup_cost + num_sandwich * sandwich_cost

theorem sean_total_cost :
  let soda_cost := 1
  let soup_cost := 3 * soda_cost
  let sandwich_cost := 3 * soup_cost
  let num_soda := 3
  let num_soup := 2
  let num_sandwich := 1
  total_cost soda_cost soup_cost sandwich_cost num_soda num_soup num_sandwich = 18 :=
by
  sorry

end sean_total_cost_l235_235794


namespace intersection_point_for_m_l235_235132

variable (n : ℕ) (x_0 y_0 : ℕ)
variable (h₁ : n ≥ 2)
variable (h₂ : y_0 ^ 2 = n * x_0 - 1)
variable (h₃ : y_0 = x_0)

theorem intersection_point_for_m (m : ℕ) (hm : 0 < m) : ∃ k : ℕ, k ≥ 2 ∧ (y_0 ^ m = x_0 ^ m) ∧ (y_0 ^ m) ^ 2 = k * (x_0 ^ m) - 1 :=
by
  sorry

end intersection_point_for_m_l235_235132


namespace class_A_has_neater_scores_l235_235508

-- Definitions for the given problem conditions
def mean_Class_A : ℝ := 120
def mean_Class_B : ℝ := 120
def variance_Class_A : ℝ := 42
def variance_Class_B : ℝ := 56

-- The theorem statement to prove Class A has neater scores
theorem class_A_has_neater_scores : (variance_Class_A < variance_Class_B) := by
  sorry

end class_A_has_neater_scores_l235_235508


namespace rate_of_drawing_barbed_wire_is_correct_l235_235856

noncomputable def rate_of_drawing_barbed_wire (area cost: ℝ) (gate_width barbed_wire_extension: ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_barbed_wire := (perimeter - 2 * gate_width) + 4 * barbed_wire_extension
  cost / total_barbed_wire

theorem rate_of_drawing_barbed_wire_is_correct :
  rate_of_drawing_barbed_wire 3136 666 1 3 = 2.85 :=
by
  sorry

end rate_of_drawing_barbed_wire_is_correct_l235_235856


namespace solve_x_l235_235103

theorem solve_x :
  ∀ (x y z w : ℤ),
    x = y + 7 →
    y = z + 15 →
    z = w + 25 →
    w = 65 →
    x = 112 :=
by
  intros x y z w
  intros h1 h2 h3 h4
  sorry

end solve_x_l235_235103


namespace problem_l235_235659

theorem problem (a b c : ℝ) (h1 : ∀ (x : ℝ), x^2 + 3 * x - 1 = 0 → x^4 + a * x^2 + b * x + c = 0) :
  a + b + 4 * c + 100 = 93 := 
sorry

end problem_l235_235659


namespace sequence_a_5_l235_235503

theorem sequence_a_5 (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) (h2 : ∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) :
  a 5 = 48 := by
  -- The proof and implementations are omitted
  sorry

end sequence_a_5_l235_235503


namespace cube_root_inequality_l235_235585

theorem cube_root_inequality {a b : ℝ} (h : a > b) : (a^(1/3)) > (b^(1/3)) :=
sorry

end cube_root_inequality_l235_235585


namespace pie_difference_l235_235986

theorem pie_difference (s1 s3 : ℚ) (h1 : s1 = 7/8) (h3 : s3 = 3/4) :
  s1 - s3 = 1/8 :=
by
  sorry

end pie_difference_l235_235986


namespace elois_banana_bread_l235_235222

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end elois_banana_bread_l235_235222


namespace johns_total_weekly_gas_consumption_l235_235136

-- Definitions of conditions
def highway_mpg : ℝ := 30
def city_mpg : ℝ := 25
def work_miles_each_way : ℝ := 20
def work_days_per_week : ℝ := 5
def highway_miles_each_way : ℝ := 15
def city_miles_each_way : ℝ := 5
def leisure_highway_miles_per_week : ℝ := 30
def leisure_city_miles_per_week : ℝ := 10
def idling_gas_consumption_per_week : ℝ := 0.3

-- Proof problem
theorem johns_total_weekly_gas_consumption :
  let work_commute_miles_per_week := work_miles_each_way * 2 * work_days_per_week
  let highway_miles_work := highway_miles_each_way * 2 * work_days_per_week
  let city_miles_work := city_miles_each_way * 2 * work_days_per_week
  let total_highway_miles := highway_miles_work + leisure_highway_miles_per_week
  let total_city_miles := city_miles_work + leisure_city_miles_per_week
  let highway_gas_consumption := total_highway_miles / highway_mpg
  let city_gas_consumption := total_city_miles / city_mpg
  (highway_gas_consumption + city_gas_consumption + idling_gas_consumption_per_week) = 8.7 := by
  sorry

end johns_total_weekly_gas_consumption_l235_235136


namespace toy_cost_price_and_profit_l235_235612

-- Define the cost price of type A toy
def cost_A (x : ℝ) : ℝ := x

-- Define the cost price of type B toy
def cost_B (x : ℝ) : ℝ := 1.5 * x

-- Spending conditions
def spending_A (x : ℝ) (num_A : ℝ) : Prop := num_A = 1200 / x
def spending_B (x : ℝ) (num_B : ℝ) : Prop := num_B = 1500 / (1.5 * x)

-- Quantity difference condition
def quantity_difference (num_A num_B : ℝ) : Prop := num_A - num_B = 20

-- Selling prices
def selling_price_A : ℝ := 12
def selling_price_B : ℝ := 20

-- Total toys purchased condition
def total_toys (num_A num_B : ℝ) : Prop := num_A + num_B = 75

-- Profit condition
def profit_condition (num_A num_B cost_A cost_B : ℝ) : Prop :=
  (selling_price_A - cost_A) * num_A + (selling_price_B - cost_B) * num_B ≥ 300

theorem toy_cost_price_and_profit :
  ∃ (x : ℝ), 
  cost_A x = 10 ∧
  cost_B x = 15 ∧
  ∀ (num_A num_B : ℝ),
  spending_A x num_A →
  spending_B x num_B →
  quantity_difference num_A num_B →
  total_toys num_A num_B →
  profit_condition num_A num_B (cost_A x) (cost_B x) →
  num_A ≤ 25 :=
by
  sorry

end toy_cost_price_and_profit_l235_235612


namespace total_roses_in_a_week_l235_235075

theorem total_roses_in_a_week : 
  let day1 := 24 
  let day2 := day1 + 6
  let day3 := day2 + 6
  let day4 := day3 + 6
  let day5 := day4 + 6
  let day6 := day5 + 6
  let day7 := day6 + 6
  (day1 + day2 + day3 + day4 + day5 + day6 + day7) = 294 :=
by
  sorry

end total_roses_in_a_week_l235_235075


namespace exist_a_sequence_l235_235471

theorem exist_a_sequence (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ (a : Fin (n+1) → ℝ), (a 0 + a n = 0) ∧ (∀ i, |a i| ≤ 1) ∧ (∀ i : Fin n, |a i.succ - a i| = x i) :=
by
  sorry

end exist_a_sequence_l235_235471


namespace part1_part2_l235_235683

section
variables (x a m n : ℝ)
-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- a) Prove the solution of the inequality f(x) >= 4 + |x-3| - |x-1| given a=3.
theorem part1 (h_a : a = 3) :
  {x | f x a ≥ 4 + abs (x - 3) - abs (x - 1)} = {x | x ≤ 0} ∪ {x | x ≥ 4} :=
sorry

-- b) Prove that m + 2n >= 2 given f(x) <= 1 + |x-3| with solution set [1, 3] and 1/m + 1/(2n) = a
theorem part2 (h_sol : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + abs (x - 3)) 
  (h_a : 1 / m + 1 / (2 * n) = 2) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  m + 2 * n ≥ 2 :=
sorry
end

end part1_part2_l235_235683


namespace number_of_boys_in_first_group_l235_235169

-- Define the daily work ratios
variables (M B : ℝ) (h_ratio : M = 2 * B)

-- Define the number of boys in the first group
variable (x : ℝ)

-- Define the conditions provided by the problem
variables (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B))

-- State the theorem and include the correct answer
theorem number_of_boys_in_first_group (M B : ℝ) (h_ratio : M = 2 * B) (x : ℝ)
    (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B)) 
    : x = 16 := 
by 
    sorry

end number_of_boys_in_first_group_l235_235169


namespace sum_arithmetic_seq_l235_235747

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l235_235747


namespace john_less_than_david_by_4_l235_235635

/-
The conditions are:
1. Zachary did 51 push-ups.
2. David did 22 more push-ups than Zachary.
3. John did 69 push-ups.

We need to prove that John did 4 push-ups less than David.
-/

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := zachary_pushups + 22
def john_pushups : ℕ := 69

theorem john_less_than_david_by_4 :
  david_pushups - john_pushups = 4 :=
by
  -- Proof goes here.
  sorry

end john_less_than_david_by_4_l235_235635


namespace probability_all_white_balls_drawn_l235_235874

theorem probability_all_white_balls_drawn (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h_total : total_balls = 15) (h_white : white_balls = 7) (h_black : black_balls = 8) (h_drawn : drawn_balls = 7) :
  (Nat.choose 7 7 : ℚ) / (Nat.choose 15 7 : ℚ) = 1 / 6435 := by
sorry

end probability_all_white_balls_drawn_l235_235874


namespace exponent_property_l235_235840

theorem exponent_property (a x y : ℝ) (hx : a ^ x = 2) (hy : a ^ y = 3) : a ^ (x + y) = 6 := by
  sorry

end exponent_property_l235_235840


namespace scientific_notation_of_4370000_l235_235906

theorem scientific_notation_of_4370000 :
  4370000 = 4.37 * 10^6 :=
sorry

end scientific_notation_of_4370000_l235_235906


namespace eval_expr_x_eq_3_y_eq_4_l235_235977

theorem eval_expr_x_eq_3_y_eq_4 : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 5 * x^y + 6 * y^x + x * y = 801 := 
by 
  intros x y hx hy 
  rw [hx, hy]
  -- Proof omitted
  sorry

end eval_expr_x_eq_3_y_eq_4_l235_235977


namespace equidistant_point_on_x_axis_l235_235152

theorem equidistant_point_on_x_axis (x : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 0)) (hB : B = (3, 5)) :
  (Real.sqrt ((x - (-3))^2)) = (Real.sqrt ((x - 3)^2 + 25)) →
  x = 25 / 12 := 
by 
  sorry

end equidistant_point_on_x_axis_l235_235152


namespace r_s_t_u_bounds_l235_235615

theorem r_s_t_u_bounds (r s t u : ℝ) 
  (H1: 5 * r + 4 * s + 3 * t + 6 * u = 100)
  (H2: r ≥ s)
  (H3: s ≥ t)
  (H4: t ≥ u)
  (H5: u ≥ 0) :
  20 ≤ r + s + t + u ∧ r + s + t + u ≤ 25 := 
sorry

end r_s_t_u_bounds_l235_235615


namespace is_factor_l235_235251

-- Define the polynomial
def poly (x : ℝ) := x^4 + 4 * x^2 + 4

-- Define a candidate for being a factor
def factor_candidate (x : ℝ) := x^2 + 2

-- Proof problem: prove that factor_candidate is a factor of poly
theorem is_factor : ∀ x : ℝ, poly x = factor_candidate x * factor_candidate x := 
by
  intro x
  unfold poly factor_candidate
  sorry

end is_factor_l235_235251


namespace num_solutions_abcd_eq_2020_l235_235569

theorem num_solutions_abcd_eq_2020 :
  ∃ S : Finset (ℕ × ℕ × ℕ × ℕ), 
    (∀ (a b c d : ℕ), (a, b, c, d) ∈ S ↔ (a^2 + b^2) * (c^2 - d^2) = 2020 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧
    S.card = 6 :=
sorry

end num_solutions_abcd_eq_2020_l235_235569


namespace joan_total_spent_l235_235485

theorem joan_total_spent (cost_basketball cost_racing total_spent : ℝ) 
  (h1 : cost_basketball = 5.20) 
  (h2 : cost_racing = 4.23) 
  (h3 : total_spent = cost_basketball + cost_racing) : 
  total_spent = 9.43 := 
by 
  sorry

end joan_total_spent_l235_235485


namespace divisors_end_with_1_l235_235474

theorem divisors_end_with_1 (n : ℕ) (h : n > 0) :
  ∀ d : ℕ, d ∣ (10^(5^n) - 1) / 9 → d % 10 = 1 :=
sorry

end divisors_end_with_1_l235_235474


namespace fraction_simplification_l235_235150

theorem fraction_simplification :
  ( (5^1004)^4 - (5^1002)^4 ) / ( (5^1003)^4 - (5^1001)^4 ) = 25 := by
  sorry

end fraction_simplification_l235_235150


namespace determine_fraction_l235_235947

noncomputable def q (x : ℝ) : ℝ := (x + 3) * (x - 2)

noncomputable def p (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem determine_fraction (a b : ℝ) (h : a + b = 1 / 4) :
  (p a b (-1)) / (q (-1)) = (a - b) / 4 :=
by
  sorry

end determine_fraction_l235_235947


namespace infinite_primes_p_solutions_eq_p2_l235_235133

theorem infinite_primes_p_solutions_eq_p2 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ 
  (∃ (S : Finset (ZMod p × ZMod p × ZMod p)),
    S.card = p^2 ∧ ∀ (x y z : ZMod p), (3 * x^3 + 4 * y^4 + 5 * z^3 - y^4 * z = 0) ↔ (x, y, z) ∈ S) :=
sorry

end infinite_primes_p_solutions_eq_p2_l235_235133


namespace combination_10_5_l235_235134

theorem combination_10_5 :
  (Nat.choose 10 5) = 2520 :=
by
  sorry

end combination_10_5_l235_235134


namespace sqrt_of_product_eq_540_l235_235559

theorem sqrt_of_product_eq_540 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := 
by 
  sorry 

end sqrt_of_product_eq_540_l235_235559


namespace rows_seating_l235_235320

theorem rows_seating (x y : ℕ) (h : 7 * x + 6 * y = 52) : x = 4 :=
by
  sorry

end rows_seating_l235_235320


namespace probability_rolls_more_ones_than_eights_l235_235507

noncomputable def probability_more_ones_than_eights (n : ℕ) := 10246 / 32768

theorem probability_rolls_more_ones_than_eights :
  (probability_more_ones_than_eights 5) = 10246 / 32768 :=
by
  sorry

end probability_rolls_more_ones_than_eights_l235_235507


namespace least_possible_value_of_p_and_q_l235_235543

theorem least_possible_value_of_p_and_q 
  (p q : ℕ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 15 * (p + 1) = 29 * (q + 1)) : 
  p + q = 45 := 
sorry -- proof to be filled in

end least_possible_value_of_p_and_q_l235_235543


namespace geom_series_first_term_l235_235292

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l235_235292


namespace product_of_integers_l235_235014

theorem product_of_integers (x y : ℕ) (h_gcd : Nat.gcd x y = 10) (h_lcm : Nat.lcm x y = 60) : x * y = 600 := by
  sorry

end product_of_integers_l235_235014


namespace range_of_a_l235_235896

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|x-2| + |x+3| < a) → false) → a ≤ 5 :=
sorry

end range_of_a_l235_235896


namespace sequence_a8_value_l235_235760

theorem sequence_a8_value :
  ∃ a : ℕ → ℚ, a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) / a n = n / (n + 1)) ∧ a 8 = 1 / 8 :=
by
  -- To be proved
  sorry

end sequence_a8_value_l235_235760


namespace retailer_profit_percentage_l235_235144

theorem retailer_profit_percentage (items_sold : ℕ) (profit_per_item : ℝ) (discount_rate : ℝ)
  (discounted_items_needed : ℝ) (total_profit : ℝ) (item_cost : ℝ) :
  items_sold = 100 → 
  profit_per_item = 30 →
  discount_rate = 0.05 →
  discounted_items_needed = 156.86274509803923 →
  total_profit = 3000 →
  (discounted_items_needed * ((item_cost + profit_per_item) * (1 - discount_rate) - item_cost) = total_profit) →
  ((profit_per_item / item_cost) * 100 = 16) :=
by {
  sorry 
}

end retailer_profit_percentage_l235_235144


namespace transistors_2004_l235_235903

-- Definition of Moore's law specifying the initial amount and the doubling period
def moores_law (initial : ℕ) (years : ℕ) (doubling_period : ℕ) : ℕ :=
  initial * 2 ^ (years / doubling_period)

-- Condition: The number of transistors in 1992
def initial_1992 : ℕ := 2000000

-- Condition: The number of years between 1992 and 2004
def years_between : ℕ := 2004 - 1992

-- Condition: Doubling period every 2 years
def doubling_period : ℕ := 2

-- Goal: Prove the number of transistors in 2004 using the conditions above
theorem transistors_2004 : moores_law initial_1992 years_between doubling_period = 128000000 :=
by
  sorry

end transistors_2004_l235_235903


namespace budget_for_equipment_l235_235363

theorem budget_for_equipment 
    (transportation_p : ℝ := 20)
    (r_d_p : ℝ := 9)
    (utilities_p : ℝ := 5)
    (supplies_p : ℝ := 2)
    (salaries_degrees : ℝ := 216)
    (total_degrees : ℝ := 360)
    (total_budget : ℝ := 100)
    :
    (total_budget - (transportation_p + r_d_p + utilities_p + supplies_p +
    (salaries_degrees / total_degrees * total_budget))) = 4 := 
sorry

end budget_for_equipment_l235_235363


namespace probability_of_5_pieces_of_candy_l235_235413

-- Define the conditions
def total_eggs : ℕ := 100 -- Assume total number of eggs is 100 for simplicity
def blue_eggs : ℕ := 4 * total_eggs / 5
def purple_eggs : ℕ := total_eggs / 5
def blue_eggs_with_5_candies : ℕ := blue_eggs / 4
def purple_eggs_with_5_candies : ℕ := purple_eggs / 2
def total_eggs_with_5_candies : ℕ := blue_eggs_with_5_candies + purple_eggs_with_5_candies

-- The proof problem
theorem probability_of_5_pieces_of_candy : (total_eggs_with_5_candies : ℚ) / (total_eggs : ℚ) = 3 / 10 := 
by
  sorry

end probability_of_5_pieces_of_candy_l235_235413


namespace largest_cuts_9x9_l235_235712

theorem largest_cuts_9x9 (k : ℕ) (V E F : ℕ) (hV : V = 81) (hE : E = 4 * k) (hF : F = 1 + 2 * k)
  (hEuler : V - E + F ≥ 2) : k ≤ 21 :=
by
  sorry

end largest_cuts_9x9_l235_235712


namespace system_of_equations_solution_l235_235852

theorem system_of_equations_solution :
  ∃ (X Y: ℝ), 
    (X^2 * Y^2 + X * Y^2 + X^2 * Y + X * Y + X + Y + 3 = 0) ∧ 
    (X^2 * Y + X * Y + 1 = 0) ∧ 
    (X = -2) ∧ (Y = -1/2) :=
by
  sorry

end system_of_equations_solution_l235_235852


namespace bounded_fx_range_a_l235_235501

-- Part (1)
theorem bounded_fx :
  ∃ M > 0, ∀ x ∈ Set.Icc (-(1/2):ℝ) (1/2), abs (x / (x + 1)) ≤ M :=
by
  sorry

-- Part (2)
theorem range_a (a : ℝ) :
  (∀ x ≥ 0, abs (1 + a * (1/2)^x + (1/4)^x) ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by
  sorry

end bounded_fx_range_a_l235_235501


namespace find_b_l235_235755

noncomputable def ellipse_foci (a b : ℝ) (hb : b > 0) (hab : a > b) : Prop :=
∃ (F1 F2 P : ℝ×ℝ), 
    (∃ (h : a > b), (2 * b^2 + 9 = a^2)) ∧ 
    (dist P F1 + dist P F2 = 2 * a) ∧ 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (2 * 4 * (a^2 - b^2) = 36)

theorem find_b (a b : ℝ) (hb : b > 0) (hab : a > b) : 
    ellipse_foci a b hb hab → b = 3 :=
by
  sorry

end find_b_l235_235755


namespace salary_reduction_l235_235901

theorem salary_reduction (S : ℝ) (x : ℝ) 
  (H1 : S > 0) 
  (H2 : 1.25 * S * (1 - 0.01 * x) = 1.0625 * S) : 
  x = 15 := 
  sorry

end salary_reduction_l235_235901


namespace truck_distance_and_efficiency_l235_235443

theorem truck_distance_and_efficiency (m d g1 g2 : ℕ) (h1 : d = 300) (h2 : g1 = 10) (h3 : g2 = 15) :
  (d * (g2 / g1) = 450) ∧ (d / g1 = 30) :=
by
  sorry

end truck_distance_and_efficiency_l235_235443


namespace garbage_collection_l235_235641

theorem garbage_collection (Daliah Dewei Zane : ℝ) 
(h1 : Daliah = 17.5)
(h2 : Dewei = Daliah - 2)
(h3 : Zane = 4 * Dewei) :
Zane = 62 :=
sorry

end garbage_collection_l235_235641


namespace Chloe_final_points_l235_235823

-- Define the points scored (or lost) in each round
def round1_points : ℤ := 40
def round2_points : ℤ := 50
def round3_points : ℤ := 60
def round4_points : ℤ := 70
def round5_points : ℤ := -4
def round6_points : ℤ := 80
def round7_points : ℤ := -6

-- Statement to prove: Chloe's total points at the end of the game
theorem Chloe_final_points : 
  round1_points + round2_points + round3_points + round4_points + round5_points + round6_points + round7_points = 290 :=
by
  sorry

end Chloe_final_points_l235_235823


namespace max_value_8a_3b_5c_l235_235820

theorem max_value_8a_3b_5c (a b c : ℝ) (h_condition : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ (Real.sqrt 373) / 6 :=
by
  sorry

end max_value_8a_3b_5c_l235_235820


namespace calculate_expression_l235_235453

noncomputable def f (x : ℝ) : ℝ :=
  (x^3 + 5 * x^2 + 6 * x) / (x^3 - x^2 - 2 * x)

def num_holes (f : ℝ → ℝ) : ℕ := 1 -- hole at x = -2
def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := 2 -- vertical asymptotes at x = 0 and x = 1
def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := 0 -- no horizontal asymptote
def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := 1 -- oblique asymptote at y = x + 4

theorem calculate_expression : num_holes f + 2 * num_vertical_asymptotes f + 3 * num_horizontal_asymptotes f + 4 * num_oblique_asymptotes f = 9 :=
by
  -- Provide the proof here
  sorry

end calculate_expression_l235_235453


namespace missed_angle_l235_235729

theorem missed_angle (sum_calculated : ℕ) (missed_angle_target : ℕ) 
  (h1 : sum_calculated = 2843) 
  (h2 : missed_angle_target = 37) : 
  ∃ n : ℕ, (sum_calculated + missed_angle_target = n * 180) :=
by {
  sorry
}

end missed_angle_l235_235729


namespace fraction_subtraction_l235_235590

theorem fraction_subtraction : (18 : ℚ) / 45 - (3 : ℚ) / 8 = (1 : ℚ) / 40 := by
  sorry

end fraction_subtraction_l235_235590


namespace exists_monochromatic_triangle_in_K6_l235_235172

/-- In a complete graph with 6 vertices where each edge is colored either red or blue,
    there exists a set of 3 vertices such that the edges joining them are all the same color. -/
theorem exists_monochromatic_triangle_in_K6 (color : Fin 6 → Fin 6 → Prop)
  (h : ∀ {i j : Fin 6}, i ≠ j → (color i j ∨ ¬ color i j)) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  ((color i j ∧ color j k ∧ color k i) ∨ (¬ color i j ∧ ¬ color j k ∧ ¬ color k i)) :=
by
  sorry

end exists_monochromatic_triangle_in_K6_l235_235172


namespace value_of_a_plus_b_l235_235721

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 2) (h3 : a < b) : a + b = -3 := by
  -- Proof goes here
  sorry

end value_of_a_plus_b_l235_235721


namespace average_is_correct_l235_235431

theorem average_is_correct (x : ℝ) : 
  (2 * x + 12 + 3 * x + 3 + 5 * x - 8) / 3 = 3 * x + 2 → x = -1 :=
by
  sorry

end average_is_correct_l235_235431


namespace find_n_l235_235048

-- Define the original and new parabola conditions
def original_parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
noncomputable def new_parabola (x n : ℝ) : ℝ := (x - n + 2)^2 - 1

-- Define the conditions for points A and B lying on the new parabola
def point_A (n : ℝ) : Prop := ∃ y₁ : ℝ, new_parabola 2 n = y₁
def point_B (n : ℝ) : Prop := ∃ y₂ : ℝ, new_parabola 4 n = y₂

-- Define the condition that y1 > y2
def points_condition (n : ℝ) : Prop := ∃ y₁ y₂ : ℝ, new_parabola 2 n = y₁ ∧ new_parabola 4 n = y₂ ∧ y₁ > y₂

-- Prove that n = 6 is the necessary value given the conditions
theorem find_n : ∀ n, (0 < n) → point_A n ∧ point_B n ∧ points_condition n → n = 6 :=
  by
    sorry

end find_n_l235_235048


namespace mike_ride_distance_l235_235170

/-- 
Mike took a taxi to the airport and paid a starting amount plus $0.25 per mile. 
Annie took a different route to the airport and paid the same starting amount plus $5.00 in bridge toll fees plus $0.25 per mile. 
Each was charged exactly the same amount, and Annie's ride was 26 miles. 
Prove that Mike's ride was 46 miles given his starting amount was $2.50.
-/
theorem mike_ride_distance
  (S C A_miles : ℝ)                  -- S: starting amount, C: cost per mile, A_miles: Annie's ride distance
  (bridge_fee total_cost : ℝ)        -- bridge_fee: Annie's bridge toll fee, total_cost: total cost for both
  (M : ℝ)                            -- M: Mike's ride distance
  (hS : S = 2.5)
  (hC : C = 0.25)
  (hA_miles : A_miles = 26)
  (h_bridge_fee : bridge_fee = 5)
  (h_total_cost_equal : total_cost = S + bridge_fee + (C * A_miles))
  (h_total_cost_mike : total_cost = S + (C * M)) :
  M = 46 :=
by 
  sorry

end mike_ride_distance_l235_235170


namespace jason_cost_l235_235644

variable (full_page_cost_per_square_inch : ℝ := 6.50)
variable (half_page_cost_per_square_inch : ℝ := 8)
variable (quarter_page_cost_per_square_inch : ℝ := 10)

variable (full_page_area : ℝ := 9 * 12)
variable (half_page_area : ℝ := full_page_area / 2)
variable (quarter_page_area : ℝ := full_page_area / 4)

variable (half_page_ads : ℝ := 1)
variable (quarter_page_ads : ℝ := 4)

variable (total_ads : ℝ := half_page_ads + quarter_page_ads)
variable (bulk_discount : ℝ := if total_ads >= 4 then 0.10 else 0.0)

variable (half_page_cost : ℝ := half_page_area * half_page_cost_per_square_inch)
variable (quarter_page_cost : ℝ := quarter_page_ads * (quarter_page_area * quarter_page_cost_per_square_inch))

variable (total_cost_before_discount : ℝ := half_page_cost + quarter_page_cost)
variable (discount_amount : ℝ := total_cost_before_discount * bulk_discount)
variable (final_cost : ℝ := total_cost_before_discount - discount_amount)

theorem jason_cost :
  final_cost = 1360.80 := by
  sorry

end jason_cost_l235_235644


namespace abs_expression_equals_one_l235_235746

theorem abs_expression_equals_one : 
  abs (abs (-(abs (2 - 3)) + 2) - 2) = 1 := 
  sorry

end abs_expression_equals_one_l235_235746


namespace correlation_coefficient_correct_option_l235_235438

variable (r : ℝ)

-- Definitions of Conditions
def positive_correlation : Prop := r > 0 → ∀ x y : ℝ, x * y > 0
def range_r : Prop := -1 < r ∧ r < 1
def correlation_strength : Prop := |r| < 1 → (∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ δ < ε ∧ |r| < δ)

-- Theorem statement
theorem correlation_coefficient_correct_option :
  (positive_correlation r) ∧
  (range_r r) ∧
  (correlation_strength r) →
  (r ≠ 0 → |r| < 1) :=
by
  sorry

end correlation_coefficient_correct_option_l235_235438


namespace max_ages_within_two_std_dev_l235_235200

def average_age : ℕ := 30
def std_dev : ℕ := 12
def lower_limit : ℕ := average_age - 2 * std_dev
def upper_limit : ℕ := average_age + 2 * std_dev
def max_different_ages : ℕ := upper_limit - lower_limit + 1

theorem max_ages_within_two_std_dev
  (avg : ℕ) (std : ℕ) (h_avg : avg = average_age) (h_std : std = std_dev)
  : max_different_ages = 49 :=
by
  sorry

end max_ages_within_two_std_dev_l235_235200


namespace solidConstruction_l235_235381

-- Definitions
structure Solid where
  octagonal_faces : Nat
  hexagonal_faces : Nat
  square_faces : Nat

-- Conditions
def solidFromCube (S : Solid) : Prop :=
  S.octagonal_faces = 6 ∧ S.hexagonal_faces = 8 ∧ S.square_faces = 12

def circumscribedSphere (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

def solidFromOctahedron (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

-- Theorem statement
theorem solidConstruction {S : Solid} :
  solidFromCube S ∧ circumscribedSphere S → solidFromOctahedron S :=
by
  sorry

end solidConstruction_l235_235381


namespace no_intersection_abs_value_graphs_l235_235460

theorem no_intersection_abs_value_graphs : 
  ∀ (x : ℝ), ¬ (|3 * x + 6| = -|4 * x - 1|) :=
by
  intro x
  sorry

end no_intersection_abs_value_graphs_l235_235460


namespace partnership_total_annual_gain_l235_235564

theorem partnership_total_annual_gain 
  (x : ℝ) 
  (G : ℝ)
  (hA_investment : x * 12 = A_investment)
  (hB_investment : 2 * x * 6 = B_investment)
  (hC_investment : 3 * x * 4 = C_investment)
  (A_share : (A_investment / (A_investment + B_investment + C_investment)) * G = 6000) :
  G = 18000 := 
sorry

end partnership_total_annual_gain_l235_235564


namespace percentage_increase_from_boys_to_total_l235_235013

def DamesSchoolBoys : ℕ := 2000
def DamesSchoolGirls : ℕ := 5000
def TotalAttendance : ℕ := DamesSchoolBoys + DamesSchoolGirls
def PercentageIncrease (initial final : ℕ) : ℚ := ((final - initial) / initial) * 100

theorem percentage_increase_from_boys_to_total :
  PercentageIncrease DamesSchoolBoys TotalAttendance = 250 :=
by
  sorry

end percentage_increase_from_boys_to_total_l235_235013


namespace estimate_fish_population_l235_235881

theorem estimate_fish_population :
  ∀ (initial_tagged: ℕ) (august_sample: ℕ) (tagged_in_august: ℕ) (leaving_rate: ℝ) (new_rate: ℝ),
  initial_tagged = 50 →
  august_sample = 80 →
  tagged_in_august = 4 →
  leaving_rate = 0.30 →
  new_rate = 0.45 →
  ∃ (april_population : ℕ),
  april_population = 550 :=
by
  intros initial_tagged august_sample tagged_in_august leaving_rate new_rate
  intros h_initial_tagged h_august_sample h_tagged_in_august h_leaving_rate h_new_rate
  existsi 550
  sorry

end estimate_fish_population_l235_235881


namespace probability_10_products_expected_value_of_products_l235_235066

open ProbabilityTheory

/-- Probability calculations for worker assessment. -/
noncomputable def worker_assessment_probability (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p^9 * (10 - 9*p)

/-- Expected value of total products produced and debugged by Worker A -/
noncomputable def expected_products (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  20 - 10*p - 10*p^9 + 10*p^10

/-- Theorem 1: Prove that the probability that Worker A ends the assessment by producing only 10 products is p^9(10 - 9p). -/
theorem probability_10_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  worker_assessment_probability p h = p^9 * (10 - 9*p) := by
  sorry

/-- Theorem 2: Prove the expected value E(X) of the total number of products produced and debugged by Worker A is 20 - 10p - 10p^9 + 10p^{10}. -/
theorem expected_value_of_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  expected_products p h = 20 - 10*p - 10*p^9 + 10*p^10 := by
  sorry

end probability_10_products_expected_value_of_products_l235_235066


namespace length_AB_slope_one_OA_dot_OB_const_l235_235176

open Real

def parabola (x y : ℝ) : Prop := y * y = 4 * x
def line_through_focus (x y : ℝ) (k : ℝ) : Prop := x = k * y + 1
def line_slope_one (x y : ℝ) : Prop := y = x - 1

theorem length_AB_slope_one {x1 x2 y1 y2 : ℝ} (hA : parabola x1 y1) (hB : parabola x2 y2) 
  (hL : line_slope_one x1 y1) (hL' : line_slope_one x2 y2) : abs (x1 - x2) + abs (y1 - y2) = 8 := 
by
  sorry

theorem OA_dot_OB_const {x1 x2 y1 y2 : ℝ} {k : ℝ} (hA : parabola x1 y1)
  (hB : parabola x2 y2) (hL : line_through_focus x1 y1 k) (hL' : line_through_focus x2 y2 k) :
  x1 * x2 + y1 * y2 = -3 :=
by
  sorry

end length_AB_slope_one_OA_dot_OB_const_l235_235176


namespace range_of_m_l235_235027

theorem range_of_m (m x : ℝ) : 
  (2 / (x - 3) + (x + m) / (3 - x) = 2) 
  ∧ (x ≥ 0) →
  (m ≤ 8 ∧ m ≠ -1) :=
by 
  sorry

end range_of_m_l235_235027


namespace cost_price_of_table_l235_235024

theorem cost_price_of_table 
  (SP : ℝ) 
  (CP : ℝ) 
  (h1 : SP = 1.24 * CP) 
  (h2 : SP = 8215) :
  CP = 6625 :=
by
  sorry

end cost_price_of_table_l235_235024


namespace product_of_roots_l235_235020

noncomputable def f : ℝ → ℝ := sorry

theorem product_of_roots :
  (∀ x : ℝ, 4 * f (3 - x) - f x = 3 * x ^ 2 - 4 * x - 3) →
  (∃ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5) :=
sorry

end product_of_roots_l235_235020


namespace initial_markup_percentage_l235_235258

-- Conditions:
-- 1. Initial price of the coat is $76.
-- 2. Increasing the price by $4 results in a 100% markup.
-- 3. A 100% markup implies the selling price is double the wholesale price.

theorem initial_markup_percentage (W : ℝ) (h1 : W + (76 - W) = 76)
  (h2 : 2 * W = 76 + 4) : (36 / 40) * 100 = 90 :=
by
  -- Using the conditions directly from the problem, we need to prove the theorem statement.
  sorry

end initial_markup_percentage_l235_235258


namespace original_average_of_15_numbers_l235_235975

theorem original_average_of_15_numbers (A : ℝ) (h1 : 15 * A + 15 * 12 = 52 * 15) :
  A = 40 :=
sorry

end original_average_of_15_numbers_l235_235975


namespace number_of_pupils_l235_235783

theorem number_of_pupils (n : ℕ) (M : ℕ)
  (avg_all : 39 * n = M)
  (pupil_marks : 25 + 12 + 15 + 19 = 71)
  (new_avg : (M - 71) / (n - 4) = 44) :
  n = 21 := sorry

end number_of_pupils_l235_235783


namespace combined_rocket_height_l235_235035

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  sorry

end combined_rocket_height_l235_235035


namespace problem_l235_235547

namespace MathProof

-- Definitions of A, B, and conditions
def A (x : ℤ) : Set ℤ := {0, |x|}
def B : Set ℤ := {1, 0, -1}

-- Prove x = ± 1 when A ⊆ B, 
-- A ∪ B = { -1, 0, 1 }, 
-- and complement of A in B is { -1 }
theorem problem (x : ℤ) (hx : A x ⊆ B) : 
  (x = 1 ∨ x = -1) ∧ 
  (A x ∪ B = {-1, 0, 1}) ∧ 
  (B \ (A x) = {-1}) := 
sorry 

end MathProof

end problem_l235_235547


namespace firetruck_reachable_area_l235_235678

theorem firetruck_reachable_area :
  let speed_highway := 50
  let speed_prairie := 14
  let travel_time := 0.1
  let area := 16800 / 961
  ∀ (x r : ℝ),
    (x / speed_highway + r / speed_prairie = travel_time) →
    (0 ≤ x ∧ 0 ≤ r) →
    ∃ m n : ℕ, gcd m n = 1 ∧
    m = 16800 ∧ n = 961 ∧
    m + n = 16800 + 961 := by
  sorry

end firetruck_reachable_area_l235_235678


namespace min_value_a_over_b_l235_235178

theorem min_value_a_over_b (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 2 * Real.sqrt a + b = 1) : ∃ c, c = 0 := 
by
  -- We need to show that the minimum value of a / b is 0 
  sorry

end min_value_a_over_b_l235_235178


namespace lunch_cost_total_l235_235685

theorem lunch_cost_total (x y : ℝ) (h1 : y = 45) (h2 : x = (2 / 3) * y) : 
  x + y + y = 120 := by
  sorry

end lunch_cost_total_l235_235685


namespace girls_at_picnic_l235_235480

variables (g b : ℕ)

-- Conditions
axiom total_students : g + b = 1500
axiom students_at_picnic : (3/4) * g + (2/3) * b = 900

-- Goal: Prove number of girls who attended the picnic
theorem girls_at_picnic (hg : (3/4 : ℚ) * 1200 = 900) : (3/4 : ℚ) * 1200 = 900 :=
by sorry

end girls_at_picnic_l235_235480


namespace syllogistic_reasoning_problem_l235_235598

theorem syllogistic_reasoning_problem
  (H1 : ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I)
  (H2 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.re z = 2)
  (H3 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.im z = 3) :
  (¬ ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I) → "The conclusion is wrong due to the incorrect major premise" = "A" :=
sorry

end syllogistic_reasoning_problem_l235_235598


namespace min_pie_pieces_l235_235352

theorem min_pie_pieces (p : ℕ) : 
  (∀ (k : ℕ), (k = 5 ∨ k = 7) → ∃ (m : ℕ), p = k * m ∨ p = m * k) → p = 11 := 
sorry

end min_pie_pieces_l235_235352


namespace inequality_holds_l235_235621

theorem inequality_holds (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, (x^2 - m * x - 2) / (x^2 - 3 * x + 4) > -1) ↔ (-7 < m ∧ m < 1) :=
by
  sorry

end inequality_holds_l235_235621


namespace problem_l235_235978

theorem problem (n : ℕ) (p : ℕ) (a b c : ℤ)
  (hn : 0 < n)
  (hp : Nat.Prime p)
  (h_eq : a^n + p * b = b^n + p * c)
  (h_eq2 : b^n + p * c = c^n + p * a) :
  a = b ∧ b = c := 
sorry

end problem_l235_235978


namespace factor_expression_l235_235078

theorem factor_expression (x y a b : ℝ) : 
  ∃ f : ℝ, 3 * x * (a - b) - 9 * y * (b - a) = f * (x + 3 * y) ∧ f = 3 * (a - b) :=
by
  sorry

end factor_expression_l235_235078


namespace midpoint_trajectory_l235_235333

theorem midpoint_trajectory (x y : ℝ) (h : ∃ (xₚ yₚ : ℝ), yₚ = 2 * xₚ^2 + 1 ∧ y = 4 * (xₚ / 2) ^ 2) : y = 4 * x ^ 2 :=
sorry

end midpoint_trajectory_l235_235333


namespace correct_judgment_is_C_l235_235988

-- Definitions based on conditions
def three_points_determine_a_plane (p1 p2 p3 : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by three points
  sorry

def line_and_point_determine_a_plane (l : Line) (p : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by a line and a point not on the line
  sorry

def two_parallel_lines_and_intersecting_line_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Axiom 3 and its corollary stating that two parallel lines intersected by the same line are in the same plane
  sorry

def three_lines_intersect_pairwise_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Definition stating that three lines intersecting pairwise might be co-planar or not
  sorry

-- Statement of the problem in Lean
theorem correct_judgment_is_C :
    ¬ (three_points_determine_a_plane p1 p2 p3)
  ∧ ¬ (line_and_point_determine_a_plane l p)
  ∧ (two_parallel_lines_and_intersecting_line_same_plane l1 l2 l3)
  ∧ ¬ (three_lines_intersect_pairwise_same_plane l1 l2 l3) :=
  sorry

end correct_judgment_is_C_l235_235988


namespace joe_initial_tests_count_l235_235991

theorem joe_initial_tests_count (n S : ℕ) (h1 : S = 45 * n) (h2 : S - 30 = 50 * (n - 1)) : n = 4 := by
  sorry

end joe_initial_tests_count_l235_235991


namespace parabola_bisects_rectangle_l235_235752
open Real

theorem parabola_bisects_rectangle (a : ℝ) (h_pos : a > 0) : 
  ((a^3 + a) / 2 = (a^3 / 3 + a)) → a = sqrt 3 := by
  sorry

end parabola_bisects_rectangle_l235_235752


namespace sufficient_condition_for_quadratic_l235_235736

theorem sufficient_condition_for_quadratic (a : ℝ) : 
  (∃ (x : ℝ), (x > a) ∧ (x^2 - 5*x + 6 ≥ 0)) ∧ 
  (¬(∀ (x : ℝ), (x^2 - 5*x + 6 ≥ 0) → (x > a))) ↔ 
  a ≥ 3 :=
by
  sorry

end sufficient_condition_for_quadratic_l235_235736


namespace quotient_of_poly_div_l235_235165

theorem quotient_of_poly_div :
  (10 * X^4 - 5 * X^3 + 3 * X^2 + 11 * X - 6) / (5 * X^2 + 7) =
  2 * X^2 - X - (11 / 5) :=
sorry

end quotient_of_poly_div_l235_235165


namespace complement_inter_proof_l235_235591

open Set

variable (U : Set ℕ) (A B : Set ℕ)

def complement_inter (U A B : Set ℕ) : Set ℕ :=
  compl (A ∩ B)

theorem complement_inter_proof (hU : U = {1, 2, 3, 4, 5, 6, 7, 8} )
  (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) :
  complement_inter U A B = {1, 4, 5, 6, 7, 8} :=
by
  sorry

end complement_inter_proof_l235_235591


namespace min_value_expression_l235_235041

theorem min_value_expression (x y : ℝ) : (∃ z : ℝ, (forall x y : ℝ, z ≤ 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4) ∧ z = 3) :=
sorry

end min_value_expression_l235_235041


namespace gain_percent_is_100_l235_235053

variable {C S : ℝ}

-- Given conditions
axiom h1 : 50 * C = 25 * S
axiom h2 : S = 2 * C

-- Prove the gain percent is 100%
theorem gain_percent_is_100 (h1 : 50 * C = 25 * S) (h2 : S = 2 * C) : (S - C) / C * 100 = 100 :=
by
  sorry

end gain_percent_is_100_l235_235053


namespace largest_value_of_a_l235_235518

theorem largest_value_of_a : 
  ∃ (a : ℚ), (3 * a + 4) * (a - 2) = 9 * a ∧ ∀ b : ℚ, (3 * b + 4) * (b - 2) = 9 * b → b ≤ 4 :=
by
  sorry

end largest_value_of_a_l235_235518


namespace jack_initial_money_l235_235725

-- Define the cost of one pair of socks
def cost_pair_socks : ℝ := 9.50

-- Define the cost of soccer shoes
def cost_soccer_shoes : ℝ := 92

-- Define the additional money Jack needs
def additional_money_needed : ℝ := 71

-- Define the total cost of two pairs of socks and one pair of soccer shoes
def total_cost : ℝ := 2 * cost_pair_socks + cost_soccer_shoes

-- Theorem to prove Jack's initial money
theorem jack_initial_money : ∃ m : ℝ, total_cost - additional_money_needed = 40 :=
by
  sorry

end jack_initial_money_l235_235725


namespace find_m_values_l235_235927

-- Given function
def f (m x : ℝ) : ℝ := m * x^2 + 3 * m * x + m - 1

-- Theorem statement
theorem find_m_values (m : ℝ) :
  (∃ x y, f m x = 0 ∧ f m y = 0 ∧ (x = 0 ∨ y = 0)) →
  (m = 1 ∨ m = -(5/4)) :=
by sorry

end find_m_values_l235_235927


namespace part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l235_235010

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l235_235010


namespace local_maximum_no_global_maximum_equation_root_condition_l235_235849

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x^2 + 2*x - 3) * Real.exp x

theorem local_maximum_no_global_maximum : (∃ x0 : ℝ, f' x0 = 0 ∧ (∀ x < x0, f x < f x0) ∧ (∀ x > x0, f x < f x0))
∧ (f 1 = -2 * Real.exp 1) 
∧ (∀ x : ℝ, ∃ b : ℝ, f x = b ∧ b > 6 * Real.exp (-3) → ¬(f x = f 1))
:= sorry

theorem equation_root_condition (b : ℝ) : (∃ x1 x2 x3 : ℝ, f x1 = b ∧ f x2 = b ∧ f x3 = b ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) 
→ (0 < b ∧ b < 6 * Real.exp (-3))
:= sorry

end local_maximum_no_global_maximum_equation_root_condition_l235_235849


namespace product_of_two_numbers_l235_235649

theorem product_of_two_numbers (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h_sum : a + b = 210) (h_lcm : Nat.lcm a b = 1547) : a * b = 10829 :=
by
  sorry

end product_of_two_numbers_l235_235649


namespace non_neg_sum_of_squares_l235_235414

theorem non_neg_sum_of_squares (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (h : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
by
  sorry

end non_neg_sum_of_squares_l235_235414


namespace _l235_235231

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : triangle_inequality 1 1 1 := 
by {
  -- Prove using the triangle inequality theorem that the sides form a triangle.
  -- This part is left as an exercise to the reader.
  sorry
}

end _l235_235231


namespace problem_l235_235878

namespace arithmetic_sequence

def is_arithmetic_sequence (a : ℕ → ℚ) := ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem 
  (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 1 + a 7 + a 13 = 4) : a 2 + a 12 = 8 / 3 :=
sorry

end arithmetic_sequence

end problem_l235_235878


namespace brock_buys_7_cookies_l235_235436

variable (cookies_total : ℕ)
variable (sold_to_stone : ℕ)
variable (left_after_sale : ℕ)
variable (cookies_brock_buys : ℕ)
variable (cookies_katy_buys : ℕ)

theorem brock_buys_7_cookies
  (h1 : cookies_total = 5 * 12)
  (h2 : sold_to_stone = 2 * 12)
  (h3 : left_after_sale = 15)
  (h4 : cookies_total - sold_to_stone - (cookies_brock_buys + cookies_katy_buys) = left_after_sale)
  (h5 : cookies_katy_buys = 2 * cookies_brock_buys) :
  cookies_brock_buys = 7 :=
by
  -- Proof is skipped
  sorry

end brock_buys_7_cookies_l235_235436


namespace oranges_weigh_4_ounces_each_l235_235660

def apple_weight : ℕ := 4
def max_bag_capacity : ℕ := 49
def num_bags : ℕ := 3
def total_weight : ℕ := num_bags * max_bag_capacity
def total_apple_weight : ℕ := 84
def num_apples : ℕ := total_apple_weight / apple_weight
def num_oranges : ℕ := num_apples
def total_orange_weight : ℕ := total_apple_weight
def weight_per_orange : ℕ := total_orange_weight / num_oranges

theorem oranges_weigh_4_ounces_each :
  weight_per_orange = 4 := by
  sorry

end oranges_weigh_4_ounces_each_l235_235660


namespace ground_beef_total_cost_l235_235285

-- Define the conditions
def price_per_kg : ℝ := 5.00
def quantity_in_kg : ℝ := 12

-- The total cost calculation
def total_cost (price_per_kg quantity_in_kg : ℝ) : ℝ := price_per_kg * quantity_in_kg

-- Theorem statement
theorem ground_beef_total_cost :
  total_cost price_per_kg quantity_in_kg = 60.00 :=
sorry

end ground_beef_total_cost_l235_235285


namespace edward_money_l235_235255

theorem edward_money (initial_amount spent1 spent2 : ℕ) (h_initial : initial_amount = 34) (h_spent1 : spent1 = 9) (h_spent2 : spent2 = 8) :
  initial_amount - (spent1 + spent2) = 17 :=
by
  sorry

end edward_money_l235_235255


namespace min_toys_to_add_l235_235031

theorem min_toys_to_add (T x : ℕ) (h1 : T % 12 = 3) (h2 : T % 18 = 3) :
  ((T + x) % 7 = 0) → x = 4 :=
by
  sorry

end min_toys_to_add_l235_235031


namespace rectangle_area_is_432_l235_235116

-- Definition of conditions and problem in Lean 4
noncomputable def circle_radius : ℝ := 6
noncomputable def rectangle_ratio_length_width : ℝ := 3 / 1
noncomputable def calculate_rectangle_area (radius : ℝ) (ratio : ℝ) : ℝ :=
  let diameter := 2 * radius
  let width := diameter
  let length := ratio * width
  length * width

-- Lean statement to prove the area
theorem rectangle_area_is_432 : calculate_rectangle_area circle_radius rectangle_ratio_length_width = 432 := by
  sorry

end rectangle_area_is_432_l235_235116


namespace sqrt_six_greater_two_l235_235567

theorem sqrt_six_greater_two : Real.sqrt 6 > 2 :=
by
  sorry

end sqrt_six_greater_two_l235_235567


namespace graph_of_equation_l235_235391

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 :=
by sorry

end graph_of_equation_l235_235391


namespace always_meaningful_fraction_l235_235241

theorem always_meaningful_fraction {x : ℝ} : (∀ x, ∃ option : ℕ, 
  (option = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) ∨ 
  (option = 2 ∧ True) ∨ 
  (option = 3 ∧ x ≠ 0) ∨ 
  (option = 4 ∧ x ≠ 1)) → option = 2 :=
sorry

end always_meaningful_fraction_l235_235241


namespace line_passes_through_fixed_point_l235_235248

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) (h : a + b = 1) (h1 : 2 * a * x - b * y = 1) : x = 1/2 ∧ y = -1 :=
by 
  sorry

end line_passes_through_fixed_point_l235_235248


namespace maximum_value_expr_l235_235566

theorem maximum_value_expr :
  ∀ (a b c d : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1) →
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
by
  intros a b c d h
  sorry

end maximum_value_expr_l235_235566


namespace arithmetic_geometric_sequence_S30_l235_235688

variable (S : ℕ → ℝ)

theorem arithmetic_geometric_sequence_S30 :
  S 10 = 10 →
  S 20 = 30 →
  S 30 = 70 := by
  intros h1 h2
  -- proof steps go here
  sorry

end arithmetic_geometric_sequence_S30_l235_235688


namespace rebus_solution_l235_235367

theorem rebus_solution (A B C D : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : D ≠ 0) 
  (h5 : A ≠ B) (h6 : A ≠ C) (h7 : A ≠ D) (h8 : B ≠ C) (h9 : B ≠ D) (h10 : C ≠ D) :
  1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D) → 
  A = 2 ∧ B = 9 ∧ C = 1 ∧ D = 6 :=
by
  intros h
  sorry

end rebus_solution_l235_235367


namespace find_points_l235_235463

noncomputable def f (x y z : ℝ) : ℝ := (x^2 + y^2 + z^2) / (x + y + z)

theorem find_points :
  (∃ (x₀ y₀ z₀ : ℝ), 0 < x₀^2 + y₀^2 + z₀^2 ∧ x₀^2 + y₀^2 + z₀^2 < 1 / 1999 ∧
    1.999 < f x₀ y₀ z₀ ∧ f x₀ y₀ z₀ < 2) :=
  sorry

end find_points_l235_235463


namespace house_orderings_l235_235757

/-- Ralph walks past five houses each painted in a different color: 
orange, red, blue, yellow, and green.
Conditions:
1. Ralph passed the orange house before the red house.
2. Ralph passed the blue house before the yellow house.
3. The blue house was not next to the yellow house.
4. Ralph passed the green house before the red house and after the blue house.
Given these conditions, prove that there are exactly 3 valid orderings of the houses.
-/
theorem house_orderings : 
  ∃ (orderings : Finset (List String)), 
  orderings.card = 3 ∧
  (∀ (o : List String), 
   o ∈ orderings ↔ 
    ∃ (idx_o idx_r idx_b idx_y idx_g : ℕ), 
    o = ["orange", "red", "blue", "yellow", "green"] ∧
    idx_o < idx_r ∧ 
    idx_b < idx_y ∧ 
    (idx_b + 1 < idx_y ∨ idx_y + 1 < idx_b) ∧ 
    idx_b < idx_g ∧ idx_g < idx_r) := sorry

end house_orderings_l235_235757


namespace students_suggested_tomatoes_l235_235099

theorem students_suggested_tomatoes (students_total mashed_potatoes bacon tomatoes : ℕ) 
  (h_total : students_total = 826)
  (h_mashed_potatoes : mashed_potatoes = 324)
  (h_bacon : bacon = 374)
  (h_tomatoes : students_total = mashed_potatoes + bacon + tomatoes) :
  tomatoes = 128 :=
by {
  sorry
}

end students_suggested_tomatoes_l235_235099


namespace express_in_scientific_notation_l235_235939

theorem express_in_scientific_notation :
  (2370000 : ℝ) = 2.37 * 10^6 := 
by
  -- proof omitted
  sorry

end express_in_scientific_notation_l235_235939


namespace hall_area_l235_235220

theorem hall_area 
  (L W : ℝ)
  (h1 : W = 1/2 * L)
  (h2 : L - W = 10) : 
  L * W = 200 := 
sorry

end hall_area_l235_235220


namespace ten_pow_n_plus_one_divisible_by_eleven_l235_235332

theorem ten_pow_n_plus_one_divisible_by_eleven (n : ℕ) (h : n % 2 = 1) : 11 ∣ (10 ^ n + 1) :=
sorry

end ten_pow_n_plus_one_divisible_by_eleven_l235_235332


namespace min_value_of_f_min_value_achieved_min_value_f_l235_235889

noncomputable def f (x : ℝ) := x + 2 / (2 * x + 1) - 1

theorem min_value_of_f : ∀ x : ℝ, x > 0 → f x ≥ 1/2 := 
by sorry

theorem min_value_achieved : f (1/2) = 1/2 := 
by sorry

theorem min_value_f : ∃ x : ℝ, x > 0 ∧ f x = 1/2 := 
⟨1/2, by norm_num, by sorry⟩

end min_value_of_f_min_value_achieved_min_value_f_l235_235889


namespace measure_of_angle_A_l235_235012

theorem measure_of_angle_A {A B C : ℝ} (hC : C = 2 * B) (hB : B = 21) :
  A = 180 - B - C := 
by 
  sorry

end measure_of_angle_A_l235_235012


namespace problem_solution_l235_235090

theorem problem_solution (a0 a1 a2 a3 a4 a5 : ℝ) :
  (1 + 2*x)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 →
  a0 + a2 + a4 = 121 := 
sorry

end problem_solution_l235_235090


namespace good_number_sum_l235_235870

def is_good (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem good_number_sum (a : ℕ) (h1 : a > 6) (h2 : is_good a) :
  ∃ x y : ℕ, is_good x ∧ is_good y ∧ a * (a + 1) = x * (x + 1) + 3 * y * (y + 1) :=
sorry

end good_number_sum_l235_235870


namespace gcd_840_1764_l235_235588

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by 
  sorry

end gcd_840_1764_l235_235588


namespace find_A_n_find_d1_d2_zero_l235_235616

-- Defining the arithmetic sequences {a_n} and {b_n} with common differences d1 and d2 respectively
variables (a b : ℕ → ℤ)
variables (d1 d2 : ℤ)

-- Conditions on the sequences
axiom a_n_arith : ∀ n, a (n + 1) = a n + d1
axiom b_n_arith : ∀ n, b (n + 1) = b n + d2

-- Definitions of A_n and B_n
def A_n (n : ℕ) : ℤ := a n + b n
def B_n (n : ℕ) : ℤ := a n * b n

-- Given initial conditions
axiom A_1 : A_n a b 1 = 1
axiom A_2 : A_n a b 2 = 3

-- Prove that A_n = 2n - 1
theorem find_A_n : ∀ n, A_n a b n = 2 * n - 1 :=
by sorry

-- Condition that B_n is an arithmetic sequence
axiom B_n_arith : ∀ n, B_n a b (n + 1) - B_n a b n = B_n a b 1 - B_n a b 0

-- Prove that d1 * d2 = 0
theorem find_d1_d2_zero : d1 * d2 = 0 :=
by sorry

end find_A_n_find_d1_d2_zero_l235_235616


namespace expand_and_simplify_l235_235397

theorem expand_and_simplify :
  ∀ x : ℝ, (x^3 - 3*x + 3)*(x^2 + 3*x + 3) = x^5 + 3*x^4 - 6*x^2 + 9 := by sorry

end expand_and_simplify_l235_235397


namespace reflection_matrix_solution_l235_235452

variable (a b : ℚ)

def matrix_R : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b], ![-(3/4 : ℚ), (4/5 : ℚ)]]

theorem reflection_matrix_solution (h : matrix_R a b ^ 2 = 1) :
    (a, b) = (-4/5, -3/5) := sorry

end reflection_matrix_solution_l235_235452


namespace enclosed_area_of_curve_l235_235331

theorem enclosed_area_of_curve :
  let side_length := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let arc_length := Real.pi
  let arc_angle := Real.pi / 2
  let arc_radius := arc_length / arc_angle
  let sector_area := (arc_angle / (2 * Real.pi)) * Real.pi * arc_radius^2
  let total_sector_area := 12 * sector_area
  let enclosed_area := octagon_area + total_sector_area + 3 * Real.pi
  enclosed_area = 54 + 38.4 * Real.sqrt 2 + 3 * Real.pi :=
by
  -- We will use sorry to indicate the proof is omitted.
  sorry

end enclosed_area_of_curve_l235_235331


namespace divide_friends_among_teams_l235_235780

theorem divide_friends_among_teams :
  let friends_num := 8
  let teams_num := 4
  (teams_num ^ friends_num) = 65536 := by
  sorry

end divide_friends_among_teams_l235_235780


namespace no_integers_exist_l235_235770

theorem no_integers_exist :
  ¬ ∃ a b : ℤ, ∃ x y : ℤ, a^5 * b + 3 = x^3 ∧ a * b^5 + 3 = y^3 :=
by
  sorry

end no_integers_exist_l235_235770


namespace scientific_notation_l235_235428

theorem scientific_notation : 899000 = 8.99 * 10^5 := 
by {
  -- We start by recognizing that we need to express 899,000 in scientific notation.
  -- Placing the decimal point after the first non-zero digit yields 8.99.
  -- Count the number of places moved (5 places to the left).
  -- Thus, 899,000 in scientific notation is 8.99 * 10^5.
  sorry
}

end scientific_notation_l235_235428


namespace odd_function_domain_real_l235_235310

theorem odd_function_domain_real
  (a : ℤ)
  (h_condition : a = -1 ∨ a = 1 ∨ a = 3) :
  (∀ x : ℝ, ∃ y : ℝ, x ≠ 0 → y = x^a) →
  (∀ x : ℝ, x ≠ 0 → (x^a = (-x)^a)) →
  (a = 1 ∨ a = 3) :=
sorry

end odd_function_domain_real_l235_235310


namespace find_numerical_value_l235_235018

-- Define the conditions
variables {x y z : ℝ}
axiom h1 : 3 * x - 4 * y - 2 * z = 0
axiom h2 : x + 4 * y - 20 * z = 0
axiom h3 : z ≠ 0

-- State the goal
theorem find_numerical_value : (x^2 + 4 * x * y) / (y^2 + z^2) = 2.933 :=
by
  sorry

end find_numerical_value_l235_235018


namespace jill_runs_more_than_jack_l235_235808

noncomputable def streetWidth : ℝ := 15 -- Street width in feet
noncomputable def blockSide : ℝ := 300 -- Side length of the block in feet

noncomputable def jacksPerimeter : ℝ := 4 * blockSide -- Perimeter of Jack's running path
noncomputable def jillsPerimeter : ℝ := 4 * (blockSide + 2 * streetWidth) -- Perimeter of Jill's running path on the opposite side of the street

theorem jill_runs_more_than_jack :
  jillsPerimeter - jacksPerimeter = 120 :=
by
  sorry

end jill_runs_more_than_jack_l235_235808


namespace company_fund_initial_amount_l235_235905

theorem company_fund_initial_amount (n : ℕ) :
  (70 * n + 75 = 80 * n - 20) →
  (n = 9) →
  (80 * n - 20 = 700) :=
by
  intros h1 h2
  rw [h2] at h1
  linarith

end company_fund_initial_amount_l235_235905


namespace megans_candy_l235_235354

variable (M : ℕ)

theorem megans_candy (h1 : M * 3 + 10 = 25) : M = 5 :=
by sorry

end megans_candy_l235_235354


namespace gcd_abcd_dcba_l235_235107

-- Definitions based on the conditions
def abcd (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d
def dcba (a b c d : ℕ) : ℕ := 1000 * d + 100 * c + 10 * b + a
def consecutive_digits (a b c d : ℕ) : Prop := (b = a + 1) ∧ (c = a + 2) ∧ (d = a + 3)

-- Theorem statement
theorem gcd_abcd_dcba (a b c d : ℕ) (h : consecutive_digits a b c d) : 
  Nat.gcd (abcd a b c d + dcba a b c d) 1111 = 1111 :=
sorry

end gcd_abcd_dcba_l235_235107


namespace inequality_holds_for_all_x_l235_235033

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (x^2 + m * x - 1) / (2 * x^2 - 2 * x + 3) < 1) ↔ -6 < m ∧ m < 2 := 
sorry -- Proof to be provided

end inequality_holds_for_all_x_l235_235033


namespace cost_of_two_sandwiches_l235_235298

theorem cost_of_two_sandwiches (J S : ℝ) 
  (h1 : 5 * J = 10) 
  (h2 : S + J = 5) :
  2 * S = 6 := 
sorry

end cost_of_two_sandwiches_l235_235298


namespace melted_ice_cream_depth_l235_235897

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h : ℝ),
    r_sphere = 3 →
    r_cylinder = 10 →
    (4 / 3) * π * r_sphere^3 = 100 * π * h →
    h = 9 / 25 :=
  by
    intros r_sphere r_cylinder h
    intros hr_sphere hr_cylinder
    intros h_volume_eq
    sorry

end melted_ice_cream_depth_l235_235897


namespace lowest_score_jack_l235_235344

noncomputable def lowest_possible_score (mean : ℝ) (std_dev : ℝ) := 
  max ((1.28 * std_dev) + mean) (mean + 2 * std_dev)

theorem lowest_score_jack (mean : ℝ := 60) (std_dev : ℝ := 10) :
  lowest_possible_score mean std_dev = 73 := 
by
  -- We need to show that the minimum score Jack could get is 73 based on problem conditions
  sorry

end lowest_score_jack_l235_235344


namespace tangent_line_at_a1_one_zero_per_interval_l235_235891

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

theorem tangent_line_at_a1 (a : ℝ) (h : a = 1) : 
  (∃ (m b : ℝ), ∀ x, f a x = m * x + b ∧ m = 2 ∧ b = 0) :=
by
  sorry

theorem one_zero_per_interval (a : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a x = 0) ∧ (∃ x : ℝ, 0 < x ∧ f a x = 0) ↔ a < -1 :=
by
  sorry

end tangent_line_at_a1_one_zero_per_interval_l235_235891


namespace option_D_correct_l235_235385

theorem option_D_correct (x : ℝ) : 2 * x^2 * (3 * x)^2 = 18 * x^4 :=
by sorry

end option_D_correct_l235_235385


namespace exists_increasing_sequences_l235_235408

theorem exists_increasing_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, b n < b (n + 1)) ∧
  (∀ n : ℕ, a n * (a n + 1) ∣ b n ^ 2 + 1) :=
sorry

end exists_increasing_sequences_l235_235408


namespace sum_of_roots_eq_l235_235841

theorem sum_of_roots_eq (k : ℝ) : ∃ x1 x2 : ℝ, (2 * x1 ^ 2 - 3 * x1 + k = 7) ∧ (2 * x2 ^ 2 - 3 * x2 + k = 7) ∧ (x1 + x2 = 3 / 2) :=
by sorry

end sum_of_roots_eq_l235_235841


namespace width_of_first_tv_is_24_l235_235961

-- Define the conditions
def height_first_tv := 16
def cost_first_tv := 672
def width_new_tv := 48
def height_new_tv := 32
def cost_new_tv := 1152
def cost_per_sq_inch_diff := 1

-- Define the width of the first TV
def width_first_tv := 24

-- Define the areas
def area_first_tv (W : ℕ) := W * height_first_tv
def area_new_tv := width_new_tv * height_new_tv

-- Define the cost per square inch
def cost_per_sq_inch_first_tv (W : ℕ) := cost_first_tv / area_first_tv W
def cost_per_sq_inch_new_tv := cost_new_tv / area_new_tv

-- The proof statement
theorem width_of_first_tv_is_24 :
  cost_per_sq_inch_first_tv width_first_tv = cost_per_sq_inch_new_tv + cost_per_sq_inch_diff
  := by
    unfold cost_per_sq_inch_first_tv
    unfold area_first_tv
    unfold cost_per_sq_inch_new_tv
    unfold area_new_tv
    sorry -- proof to be filled in

end width_of_first_tv_is_24_l235_235961


namespace tan_alpha_frac_simplification_l235_235433

theorem tan_alpha_frac_simplification (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 4 / 3 :=
by sorry

end tan_alpha_frac_simplification_l235_235433


namespace print_shop_x_charge_l235_235308

theorem print_shop_x_charge :
  ∃ (x : ℝ), 60 * x + 90 = 60 * 2.75 ∧ x = 1.25 :=
by
  sorry

end print_shop_x_charge_l235_235308


namespace samantha_original_cans_l235_235680

theorem samantha_original_cans : 
  ∀ (cans_per_classroom : ℚ),
  (cans_per_classroom = (50 - 38) / 5) →
  (50 / cans_per_classroom) = 21 := 
by
  sorry

end samantha_original_cans_l235_235680


namespace compare_neg_fractions_l235_235073

theorem compare_neg_fractions : (-3 / 5) < (-1 / 3) := 
by {
  sorry
}

end compare_neg_fractions_l235_235073


namespace average_marks_l235_235737

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end average_marks_l235_235737


namespace length_AD_l235_235312

theorem length_AD (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 13) (h3 : x * (13 - x) = 36) : x = 4 ∨ x = 9 :=
by sorry

end length_AD_l235_235312


namespace sin_neg_270_eq_one_l235_235484

theorem sin_neg_270_eq_one : Real.sin (-(270 : ℝ) * (Real.pi / 180)) = 1 := by
  sorry

end sin_neg_270_eq_one_l235_235484


namespace ring_display_capacity_l235_235859

def necklace_capacity : ℕ := 12
def current_necklaces : ℕ := 5
def ring_capacity : ℕ := 18
def bracelet_capacity : ℕ := 15
def current_bracelets : ℕ := 8
def necklace_cost : ℕ := 4
def ring_cost : ℕ := 10
def bracelet_cost : ℕ := 5
def total_cost : ℕ := 183

theorem ring_display_capacity : ring_capacity + (total_cost - ((necklace_capacity - current_necklaces) * necklace_cost + (bracelet_capacity - current_bracelets) * bracelet_cost)) / ring_cost = 30 := by
  sorry

end ring_display_capacity_l235_235859


namespace cake_fractions_l235_235890

theorem cake_fractions (x y z : ℚ) 
  (h1 : x + y + z = 1)
  (h2 : 2 * z = x)
  (h3 : z = 1 / 2 * (y + 2 / 3 * x)) :
  x = 6 / 11 ∧ y = 2 / 11 ∧ z = 3 / 11 :=
sorry

end cake_fractions_l235_235890


namespace parabola_tangency_point_l235_235334

-- Definitions of the parabola equations
def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 20
def parabola2 (y : ℝ) : ℝ := y^2 + 36 * y + 380

-- The proof statement
theorem parabola_tangency_point : 
  ∃ (x y : ℝ), 
    parabola1 x = y ∧ parabola2 y = x ∧ x = -9 / 2 ∧ y = -35 / 2 :=
by
  sorry

end parabola_tangency_point_l235_235334


namespace cylinder_volume_l235_235042

theorem cylinder_volume (h : ℝ) (H1 : π * h ^ 2 = 4 * π) : (π * (h / 2) ^ 2 * h) = 2 * π :=
by
  sorry

end cylinder_volume_l235_235042


namespace angle_in_second_quadrant_l235_235727

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : (2 * Real.tan (α / 2)) / (1 - (Real.tan (α / 2))^2) < 0) : 
  ∃ q, q = 2 ∧ α ∈ {α | 0 < α ∧ α < π} :=
by
  sorry

end angle_in_second_quadrant_l235_235727


namespace sum_is_eighteen_or_twentyseven_l235_235704

theorem sum_is_eighteen_or_twentyseven :
  ∀ (A B C D E I J K L M : ℕ),
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ I ∧ A ≠ J ∧ A ≠ K ∧ A ≠ L ∧ A ≠ M ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ I ∧ B ≠ J ∧ B ≠ K ∧ B ≠ L ∧ B ≠ M ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ I ∧ C ≠ J ∧ C ≠ K ∧ C ≠ L ∧ C ≠ M ∧
  D ≠ E ∧ D ≠ I ∧ D ≠ J ∧ D ≠ K ∧ D ≠ L ∧ D ≠ M ∧
  E ≠ I ∧ E ≠ J ∧ E ≠ K ∧ E ≠ L ∧ E ≠ M ∧
  I ≠ J ∧ I ≠ K ∧ I ≠ L ∧ I ≠ M ∧
  J ≠ K ∧ J ≠ L ∧ J ≠ M ∧
  K ≠ L ∧ K ≠ M ∧
  L ≠ M ∧
  (0 < I) ∧ (0 < J) ∧ (0 < K) ∧ (0 < L) ∧ (0 < M) ∧
  A + B + C + D + E + I + J + K + L + M = 45 ∧
  (I + J + K + L + M) % 10 = 0 →
  A + B + C + D + E + (I + J + K + L + M) / 10 = 18 ∨
  A + B + C + D + E + (I + J + K + L + M) / 10 = 27 :=
by
  intros
  sorry

end sum_is_eighteen_or_twentyseven_l235_235704


namespace number_of_teachers_in_school_l235_235370

-- Definitions based on provided conditions
def number_of_girls : ℕ := 315
def number_of_boys : ℕ := 309
def total_number_of_people : ℕ := 1396

-- Proof goal: Number of teachers in the school
theorem number_of_teachers_in_school : 
  total_number_of_people - (number_of_girls + number_of_boys) = 772 :=
by
  sorry

end number_of_teachers_in_school_l235_235370


namespace bananas_distribution_l235_235235

noncomputable def total_bananas : ℝ := 550.5
noncomputable def lydia_bananas : ℝ := 80.25
noncomputable def dawn_bananas : ℝ := lydia_bananas + 93
noncomputable def emily_bananas : ℝ := 198
noncomputable def donna_bananas : ℝ := emily_bananas / 2

theorem bananas_distribution :
  dawn_bananas = 173.25 ∧
  lydia_bananas = 80.25 ∧
  donna_bananas = 99 ∧
  emily_bananas = 198 ∧
  dawn_bananas + lydia_bananas + donna_bananas + emily_bananas = total_bananas :=
by
  sorry

end bananas_distribution_l235_235235


namespace map_float_time_l235_235740

theorem map_float_time
  (t₀ t₁ : Nat) -- times representing 12:00 PM and 12:21 PM in minutes since midnight
  (v_w v_b : ℝ) -- constant speed of water current and boat in still water
  (h₀ : t₀ = 12 * 60) -- t₀ is 12:00 PM
  (h₁ : t₁ = 12 * 60 + 21) -- t₁ is 12:21 PM
  : t₁ - t₀ = 21 := 
  sorry

end map_float_time_l235_235740


namespace additive_inverse_of_half_l235_235418

theorem additive_inverse_of_half :
  - (1 / 2) = -1 / 2 :=
by
  sorry

end additive_inverse_of_half_l235_235418


namespace minimum_value_l235_235210

open Real

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : log (2^x) + log (8^y) = log 2) :
  ∃ (v : ℝ), v = 4 ∧ ∀ u, (∀ x y, x > 0 ∧ y > 0 → log (2^x) + log (8^y) = log 2 → x + 3*y = 1 → u = 4) := sorry

end minimum_value_l235_235210


namespace distinct_license_plates_l235_235715

theorem distinct_license_plates :
  let num_digits := 10
  let num_letters := 26
  let num_digit_positions := 5
  let num_letter_pairs := num_letters * num_letters
  let num_letter_positions := num_digit_positions + 1
  num_digits^num_digit_positions * num_letter_pairs * num_letter_positions = 40560000 := by
  sorry

end distinct_license_plates_l235_235715


namespace combination_8_5_l235_235059

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l235_235059


namespace triangle_existence_condition_l235_235420

theorem triangle_existence_condition 
  (a b f_c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : f_c > 0) : 
  (2 * a * b / (a + b)) > f_c :=
sorry

end triangle_existence_condition_l235_235420


namespace minimum_value_problem_l235_235861

theorem minimum_value_problem (a b c : ℝ) (hb : a > 0 ∧ b > 0 ∧ c > 0)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) : 
  ∃ x, (x = 47) ∧ (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ x :=
by
  sorry

end minimum_value_problem_l235_235861


namespace average_weight_calculation_l235_235778

noncomputable def new_average_weight (initial_people : ℕ) (initial_avg_weight : ℝ) 
                                     (new_person_weight : ℝ) (total_people : ℕ) : ℝ :=
  (initial_people * initial_avg_weight + new_person_weight) / total_people

theorem average_weight_calculation :
  new_average_weight 6 160 97 7 = 151 := by
  sorry

end average_weight_calculation_l235_235778


namespace equation_of_line_l235_235716

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  (dot_product / norm_sq * v.1, dot_product / norm_sq * v.2)

theorem equation_of_line (x y : ℝ) :
  projection (x, y) (7, 3) = (-7, -3) →
  y = -7/3 * x - 58/3 :=
by
  intro h
  sorry

end equation_of_line_l235_235716


namespace number_of_distinct_possible_values_for_c_l235_235347

variables {a b r s t : ℂ}
variables (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
variables (h_transform : ∀ z, (a * z + b - r) * (a * z + b - s) * (a * z + b - t) = (z - c * r) * (z - c * s) * (z - c * t))

theorem number_of_distinct_possible_values_for_c (h_nonzero : a ≠ 0) : 
  ∃ (n : ℕ), n = 4 := sorry

end number_of_distinct_possible_values_for_c_l235_235347


namespace sequence_sum_problem_l235_235759

theorem sequence_sum_problem (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * a n - n) :
  (2 / (a 1 * a 2) + 4 / (a 2 * a 3) + 8 / (a 3 * a 4) + 16 / (a 4 * a 5) : ℚ) = 30 / 31 := 
sorry

end sequence_sum_problem_l235_235759


namespace num_females_math_not_english_is_15_l235_235909

-- Define the conditions
def male_math := 120
def female_math := 80
def female_english := 120
def male_english := 80
def total_students := 260
def both_male := 75

def female_math_not_english : Nat :=
  female_math - (female_english + female_math - (total_students - (male_math + male_english - both_male)))

theorem num_females_math_not_english_is_15 :
  female_math_not_english = 15 :=
by
  -- This is where the proof will be, but for now, we use 'sorry' to skip it.
  sorry

end num_females_math_not_english_is_15_l235_235909


namespace geometric_sequence_first_term_l235_235111

theorem geometric_sequence_first_term (a b c : ℕ) (r : ℕ) (h1 : r = 2) (h2 : b = a * r)
  (h3 : c = b * r) (h4 : 32 = c * r) (h5 : 64 = 32 * r) :
  a = 4 :=
by sorry

end geometric_sequence_first_term_l235_235111


namespace knights_in_company_l235_235951

theorem knights_in_company :
  ∃ k : ℕ, (k = 0 ∨ k = 6) ∧ k ≤ 39 ∧
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 39) →
    (∃ i : ℕ, (1 ≤ i ∧ i ≤ 39) ∧ n * k = 1 + (i - 1) * k) →
    ∃ i : ℕ, ∃ nk : ℕ, (nk = i * k ∧ nk ≤ 39 ∧ (nk ∣ k → i = 1 + (i - 1))) :=
by
  sorry

end knights_in_company_l235_235951


namespace sum_squares_inequality_l235_235730

theorem sum_squares_inequality {a b c : ℝ} 
  (h1 : a > 0)
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
sorry

end sum_squares_inequality_l235_235730


namespace side_length_of_square_l235_235655

theorem side_length_of_square (length_rect width_rect : ℝ) (h_length : length_rect = 7) (h_width : width_rect = 5) :
  (∃ side_length : ℝ, 4 * side_length = 2 * (length_rect + width_rect) ∧ side_length = 6) :=
by
  use 6
  simp [h_length, h_width]
  sorry

end side_length_of_square_l235_235655


namespace math_problem_l235_235280

theorem math_problem
  (x y z : ℤ)
  (hz : z ≠ 0)
  (eq1 : 2 * x - 3 * y - z = 0)
  (eq2 : x + 3 * y - 14 * z = 0) :
  (x^2 - x * y) / (y^2 + 2 * z^2) = 10 / 11 := 
by 
  sorry

end math_problem_l235_235280


namespace bonnets_difference_thursday_monday_l235_235695

variable (Bm Bt Bf : ℕ)

-- Conditions
axiom monday_bonnets_made : Bm = 10
axiom tuesday_wednesday_bonnets_made : Bm + (2 * Bm) = 30
axiom bonnets_sent_to_orphanages : (Bm + Bt + (Bt - 5) + Bm + (2 * Bm)) / 5 = 11
axiom friday_bonnets_made : Bf = Bt - 5

theorem bonnets_difference_thursday_monday :
  Bt - Bm = 5 :=
sorry

end bonnets_difference_thursday_monday_l235_235695


namespace find_k_l235_235324

-- Assume three lines in the form of equations
def line1 (x y k : ℝ) := x + k * y = 0
def line2 (x y : ℝ) := 2 * x + 3 * y + 8 = 0
def line3 (x y : ℝ) := x - y - 1 = 0

-- Assume the intersection point exists
def intersection_point (x y : ℝ) := 
  line2 x y ∧ line3 x y

-- The main theorem statement
theorem find_k (k : ℝ) (x y : ℝ) (h : intersection_point x y) : 
  line1 x y k ↔ k = -1/2 := 
sorry

end find_k_l235_235324


namespace rectangular_prism_sum_l235_235682

theorem rectangular_prism_sum : 
  let edges := 12
  let vertices := 8
  let faces := 6
  edges + vertices + faces = 26 := by
sorry

end rectangular_prism_sum_l235_235682


namespace one_third_of_product_l235_235750

theorem one_third_of_product (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 4) : (1 / 3 : ℚ) * (a * b * c : ℕ) = 84 := by
  sorry

end one_third_of_product_l235_235750


namespace integer_solution_l235_235249

theorem integer_solution (x : ℤ) (h : x^2 < 3 * x) : x = 1 ∨ x = 2 :=
sorry

end integer_solution_l235_235249


namespace intersection_points_l235_235434

-- Define parameters: number of sides for each polygon
def n₆ := 6
def n₇ := 7
def n₈ := 8
def n₉ := 9

-- Condition: polygons are inscribed in the same circle, no shared vertices, no three sides intersect at a common point
def polygons_are_disjoint (n₁ n₂ : ℕ) (n₃ n₄ : ℕ) (n₅ : ℕ) : Prop :=
  true -- Assume this is a primitive condition encapsulating given constraints

-- Prove the number of intersection points is 80
theorem intersection_points : polygons_are_disjoint n₆ n₇ n₈ n₉ n₅ → 
  2 * (n₆ + n₇ + n₇ + n₈) + 2 * (n₇ + n₈) + 2 * n₉ = 80 :=
by  
  sorry

end intersection_points_l235_235434


namespace converse_implication_l235_235424

theorem converse_implication (a : ℝ) : (a^2 = 1 → a = 1) → (a = 1 → a^2 = 1) :=
sorry

end converse_implication_l235_235424


namespace smallest_n_for_terminating_decimal_l235_235387

theorem smallest_n_for_terminating_decimal : 
  ∃ n : ℕ, (0 < n) ∧ (∃ k m : ℕ, (n + 70 = 2 ^ k * 5 ^ m) ∧ k = 0 ∨ k = 1) ∧ n = 55 :=
by sorry

end smallest_n_for_terminating_decimal_l235_235387


namespace oranges_weight_is_10_l235_235579

def applesWeight (A : ℕ) : ℕ := A
def orangesWeight (A : ℕ) : ℕ := 5 * A
def totalWeight (A : ℕ) (O : ℕ) : ℕ := A + O
def totalCost (A : ℕ) (x : ℕ) (O : ℕ) (y : ℕ) : ℕ := A * x + O * y

theorem oranges_weight_is_10 (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := by
  sorry

end oranges_weight_is_10_l235_235579


namespace apex_angle_of_quadrilateral_pyramid_l235_235319

theorem apex_angle_of_quadrilateral_pyramid :
  ∃ (α : ℝ), α = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end apex_angle_of_quadrilateral_pyramid_l235_235319


namespace child_grandmother_ratio_l235_235009

def grandmother_weight (G D C : ℝ) : Prop :=
  G + D + C = 160

def daughter_child_weight (D C : ℝ) : Prop :=
  D + C = 60

def daughter_weight (D : ℝ) : Prop :=
  D = 40

theorem child_grandmother_ratio (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_child_weight D C) (h3 : daughter_weight D) :
  C / G = 1 / 5 :=
sorry

end child_grandmother_ratio_l235_235009


namespace unique_m_power_function_increasing_l235_235393

theorem unique_m_power_function_increasing : 
  ∃! (m : ℝ), (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m-1) > 0) ∧ (m^2 - m - 5 = 1) ∧ (m - 1 > 0) :=
by
  sorry

end unique_m_power_function_increasing_l235_235393


namespace time_to_cross_bridge_l235_235345

theorem time_to_cross_bridge 
  (speed_kmhr : ℕ) 
  (bridge_length_m : ℕ) 
  (h1 : speed_kmhr = 10)
  (h2 : bridge_length_m = 2500) :
  (bridge_length_m / (speed_kmhr * 1000 / 60) = 15) :=
by
  sorry

end time_to_cross_bridge_l235_235345


namespace number_of_perfect_numbers_l235_235047

-- Define the concept of a perfect number
def perfect_number (a b : ℕ) : ℕ := (a + b)^2

-- Define the proposition we want to prove
theorem number_of_perfect_numbers : ∃ n : ℕ, n = 15 ∧ 
  ∀ p, ∃ a b : ℕ, p = perfect_number a b ∧ p < 200 :=
sorry

end number_of_perfect_numbers_l235_235047


namespace area_of_square_eq_36_l235_235962

theorem area_of_square_eq_36 :
  ∃ (s q : ℝ), q = 6 ∧ s = 10 ∧ (∃ p : ℝ, p = 24 ∧ (p / 4) * (p / 4) = 36) := 
by
  sorry

end area_of_square_eq_36_l235_235962


namespace sam_wins_l235_235689

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l235_235689


namespace min_x_plus_y_l235_235093

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) :
  x + y ≥ 16 :=
sorry

end min_x_plus_y_l235_235093


namespace prob_both_selected_l235_235630

-- Define the probabilities of selection
def prob_selection_x : ℚ := 1 / 5
def prob_selection_y : ℚ := 2 / 3

-- Prove that the probability that both x and y are selected is 2 / 15
theorem prob_both_selected : prob_selection_x * prob_selection_y = 2 / 15 := 
by
  sorry

end prob_both_selected_l235_235630


namespace sequence_formula_l235_235207

theorem sequence_formula :
  ∀ (a : ℕ → ℕ),
  (a 1 = 11) ∧
  (a 2 = 102) ∧
  (a 3 = 1003) ∧
  (a 4 = 10004) →
  ∀ n, a n = 10^n + n := by
  sorry

end sequence_formula_l235_235207


namespace child_height_at_last_visit_l235_235924

-- Definitions for the problem
def h_current : ℝ := 41.5 -- current height in inches
def Δh : ℝ := 3 -- height growth in inches

-- The proof statement
theorem child_height_at_last_visit : h_current - Δh = 38.5 := by
  sorry

end child_height_at_last_visit_l235_235924


namespace determine_n_l235_235266

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem determine_n :
  (∃ n : ℕ, digit_sum (9 * (10^n - 1)) = 999 ∧ n = 111) :=
sorry

end determine_n_l235_235266


namespace winning_ticket_probability_l235_235782

theorem winning_ticket_probability (eligible_numbers : List ℕ) (length_eligible_numbers : eligible_numbers.length = 12)
(pick_6 : Π(t : List ℕ), List ℕ) (valid_ticket : List ℕ → Bool) (probability : ℚ) : 
(probability = (1 : ℚ) / (4 : ℚ)) :=
  sorry

end winning_ticket_probability_l235_235782


namespace square_tile_area_l235_235692

-- Definition and statement of the problem
theorem square_tile_area (side_length : ℝ) (h : side_length = 7) : 
  (side_length * side_length) = 49 :=
by
  sorry

end square_tile_area_l235_235692


namespace ratio_of_segments_l235_235691

theorem ratio_of_segments (a b x : ℝ) (h₁ : a = 9 * x) (h₂ : b = 99 * x) : b / a = 11 := by
  sorry

end ratio_of_segments_l235_235691


namespace best_sampling_method_l235_235357

/-- 
  Given a high school that wants to understand the psychological 
  pressure of students from three different grades, prove that 
  stratified sampling is the best method to use, assuming students
  from different grades may experience different levels of psychological
  pressure.
-/
theorem best_sampling_method
  (students_from_three_grades : Type)
  (survey_psychological_pressure : students_from_three_grades → ℝ)
  (potential_differences_by_grade : students_from_three_grades → ℝ → Prop):
  ∃ sampling_method, sampling_method = "stratified_sampling" :=
sorry

end best_sampling_method_l235_235357


namespace xiao_ming_water_usage_ge_8_l235_235039

def min_monthly_water_usage (x : ℝ) : Prop :=
  ∀ (c : ℝ), c ≥ 15 →
    (c = if x ≤ 5 then x * 1.8 else (5 * 1.8 + (x - 5) * 2)) →
      x ≥ 8

theorem xiao_ming_water_usage_ge_8 : ∃ x : ℝ, min_monthly_water_usage x :=
  sorry

end xiao_ming_water_usage_ge_8_l235_235039


namespace bottle_caps_total_l235_235573

def initial_bottle_caps := 51.0
def given_bottle_caps := 36.0

theorem bottle_caps_total : initial_bottle_caps + given_bottle_caps = 87.0 := by
  sorry

end bottle_caps_total_l235_235573


namespace range_of_a_l235_235478

open Set Real

noncomputable def f (x a : ℝ) := x ^ 2 + 2 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≥ 0) → a ≥ -8 :=
by
  intro h
  sorry

end range_of_a_l235_235478


namespace cos_of_angle_C_l235_235494

theorem cos_of_angle_C (A B C : ℝ)
  (h1 : Real.sin (π - A) = 3 / 5)
  (h2 : Real.tan (π + B) = 12 / 5)
  (h_cos_A : Real.cos A = 4 / 5) :
  Real.cos C = 16 / 65 :=
sorry

end cos_of_angle_C_l235_235494


namespace exists_integer_roots_l235_235623

theorem exists_integer_roots : 
  ∃ (a b c d e f : ℤ), ∃ r1 r2 r3 r4 r5 r6 : ℤ,
  (r1 + a) * (r2 ^ 2 + b * r2 + c) * (r3 ^ 3 + d * r3 ^ 2 + e * r3 + f) = 0 ∧
  (r4 + a) * (r5 ^ 2 + b * r5 + c) * (r6 ^ 3 + d * r6 ^ 2 + e * r6 + f) = 0 :=
  sorry

end exists_integer_roots_l235_235623


namespace sum_of_roots_l235_235528

theorem sum_of_roots {a b : Real} (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) : a + b = 4 :=
by
  sorry

end sum_of_roots_l235_235528


namespace trajectory_of_M_l235_235356

open Real

-- Define the endpoints A and B
variable {A B M : Real × Real}

-- Given conditions
def segment_length (A B : Real × Real) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25

def on_axes (A B : Real × Real) : Prop :=
  A.2 = 0 ∧ B.1 = 0

def point_m_relationship (A B M : Real × Real) : Prop :=
  let AM := (M.1 - A.1, M.2 - A.2)
  let MB := (M.1 - B.1, M.2 - B.2)
  AM.1 = (2 / 3) * MB.1 ∧ AM.2 = (2 / 3) * MB.2 ∧
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4

theorem trajectory_of_M (A B M : Real × Real)
  (h1 : segment_length A B)
  (h2 : on_axes A B)
  (h3 : point_m_relationship A B M) :
  (M.1^2 / 9) + (M.2^2 / 4) = 1 :=
sorry

end trajectory_of_M_l235_235356


namespace problem_statement_l235_235296

-- Definitions for the given conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- The main statement that needs to be proved
theorem problem_statement (f : ℝ → ℝ) (h_odd : odd_function f) (h_monotone : monotone_decreasing f) : f (-1) > f 3 :=
by 
  sorry

end problem_statement_l235_235296


namespace roger_has_more_candy_l235_235565

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

end roger_has_more_candy_l235_235565


namespace find_starting_number_of_range_l235_235535

theorem find_starting_number_of_range :
  ∃ n : ℕ, ∀ k : ℕ, k < 7 → (n + k * 9) ∣ 9 ∧ (n + k * 9) ≤ 97 ∧ (∀ m < k, (n + m * 9) < n + (m + 1) * 9) := 
sorry

end find_starting_number_of_range_l235_235535


namespace min_sum_distinct_positive_integers_l235_235696

theorem min_sum_distinct_positive_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (1 / a + 1 / b = k1 * (1 / c)) ∧ (1 / a + 1 / c = k2 * (1 / b)) ∧ (1 / b + 1 / c = k3 * (1 / a))) :
  a + b + c ≥ 11 :=
sorry

end min_sum_distinct_positive_integers_l235_235696


namespace greatest_integer_n_l235_235060

theorem greatest_integer_n (n : ℤ) : n^2 - 9 * n + 20 ≤ 0 → n ≤ 5 := sorry

end greatest_integer_n_l235_235060


namespace soccer_club_girls_l235_235303

theorem soccer_club_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : (1 / 3 : ℚ) * G + B = 18) : 
  G = 18 := 
  by sorry

end soccer_club_girls_l235_235303


namespace sin_double_angle_of_tan_pi_sub_alpha_eq_two_l235_235007

theorem sin_double_angle_of_tan_pi_sub_alpha_eq_two 
  (α : Real) 
  (h : Real.tan (Real.pi - α) = 2) : 
  Real.sin (2 * α) = -4 / 5 := 
  by sorry

end sin_double_angle_of_tan_pi_sub_alpha_eq_two_l235_235007


namespace root_in_interval_imp_range_m_l235_235614

theorem root_in_interval_imp_range_m (m : ℝ) (f : ℝ → ℝ) (h : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0) : 2 < m ∧ m < 4 :=
by
  have exists_x : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0 := h
  sorry

end root_in_interval_imp_range_m_l235_235614


namespace arithmetic_progression_K_l235_235629

theorem arithmetic_progression_K (K : ℕ) : 
  (∃ n : ℕ, K = 30 * n - 1) ↔ (K^K + 1) % 30 = 0 :=
sorry

end arithmetic_progression_K_l235_235629


namespace exists_x_eq_28_l235_235270

theorem exists_x_eq_28 : ∃ x : Int, 45 - (x - (37 - (15 - 16))) = 55 ↔ x = 28 := 
by
  sorry

end exists_x_eq_28_l235_235270


namespace people_in_gym_l235_235670

-- Define the initial number of people in the gym
def initial_people : ℕ := 16

-- Define the number of additional people entering the gym
def additional_people : ℕ := 5

-- Define the number of people leaving the gym
def people_leaving : ℕ := 2

-- Define the final number of people in the gym as per the conditions
def final_people (initial : ℕ) (additional : ℕ) (leaving : ℕ) : ℕ :=
  initial + additional - leaving

-- The theorem to prove
theorem people_in_gym : final_people initial_people additional_people people_leaving = 19 :=
  by
    sorry

end people_in_gym_l235_235670


namespace perfect_square_A_plus_B_plus1_l235_235459

-- Definitions based on conditions
def A (m : ℕ) : ℕ := (10^2*m - 1) / 9
def B (m : ℕ) : ℕ := 4 * (10^m - 1) / 9

-- Proof statement
theorem perfect_square_A_plus_B_plus1 (m : ℕ) : A m + B m + 1 = ((10^m + 2) / 3)^2 :=
by
  sorry

end perfect_square_A_plus_B_plus1_l235_235459


namespace pieces_eaten_first_night_l235_235972

def initial_candy_debby : ℕ := 32
def initial_candy_sister : ℕ := 42
def candy_after_first_night : ℕ := 39

theorem pieces_eaten_first_night :
  (initial_candy_debby + initial_candy_sister) - candy_after_first_night = 35 := by
  sorry

end pieces_eaten_first_night_l235_235972


namespace value_of_neg2_neg4_l235_235930

def operation (a b x y : ℤ) : ℤ := a * x - b * y

theorem value_of_neg2_neg4 (a b : ℤ) (h : operation a b 1 2 = 8) : operation a b (-2) (-4) = -16 := by
  sorry

end value_of_neg2_neg4_l235_235930


namespace equation_represents_circle_m_condition_l235_235552

theorem equation_represents_circle_m_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0) → m < 1/2 := 
by
  sorry

end equation_represents_circle_m_condition_l235_235552


namespace principal_amount_borrowed_l235_235572

theorem principal_amount_borrowed (SI R T : ℝ) (h_SI : SI = 2000) (h_R : R = 4) (h_T : T = 10) : 
    ∃ P, SI = (P * R * T) / 100 ∧ P = 5000 :=
by
    sorry

end principal_amount_borrowed_l235_235572


namespace increase_in_average_l235_235146

variable (A : ℝ)
variable (new_avg : ℝ := 44)
variable (score_12th_inning : ℝ := 55)
variable (total_runs_after_11 : ℝ := 11 * A)

theorem increase_in_average :
  ((total_runs_after_11 + score_12th_inning) / 12 - A = 1) :=
by
  sorry

end increase_in_average_l235_235146


namespace sqrt_of_sum_eq_l235_235821

noncomputable def cube_term : ℝ := 2 ^ 3
noncomputable def sum_cubes : ℝ := cube_term + cube_term + cube_term + cube_term
noncomputable def sqrt_sum : ℝ := Real.sqrt sum_cubes

theorem sqrt_of_sum_eq :
  sqrt_sum = 4 * Real.sqrt 2 :=
by
  sorry

end sqrt_of_sum_eq_l235_235821


namespace total_ways_is_13_l235_235766

-- Define the problem conditions
def num_bus_services : ℕ := 8
def num_train_services : ℕ := 3
def num_ferry_services : ℕ := 2

-- Define the total number of ways a person can travel from A to B
def total_ways : ℕ := num_bus_services + num_train_services + num_ferry_services

-- State the theorem that the total number of ways is 13
theorem total_ways_is_13 : total_ways = 13 :=
by
  -- Add a sorry placeholder for the proof
  sorry

end total_ways_is_13_l235_235766


namespace greatest_radius_l235_235179

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l235_235179


namespace value_of_y_l235_235650

theorem value_of_y (y: ℚ) (h: (2 / 5 - 1 / 7) = 14 / y): y = 490 / 9 :=
by
  sorry

end value_of_y_l235_235650


namespace problem_solution_l235_235495

noncomputable def ellipse_properties (F1 F2 : ℝ × ℝ) (sum_dists : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let a := sum_dists / 2 
  let c := (Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  (h, k, a, b)

theorem problem_solution :
  let F1 := (0, 1)
  let F2 := (6, 1)
  let sum_dists := 10
  let (h, k, a, b) := ellipse_properties F1 F2 sum_dists
  h + k + a + b = 13 :=
by
  -- assuming the proof here
  sorry

end problem_solution_l235_235495


namespace infinite_series_sum_l235_235675

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 1998)^n) = (3992004 / 3988009) :=
by sorry

end infinite_series_sum_l235_235675


namespace tan_add_pi_over_3_l235_235589

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l235_235589


namespace OH_squared_correct_l235_235056

noncomputable def OH_squared (O H : Point) (a b c R : ℝ) : ℝ :=
  9 * R^2 - (a^2 + b^2 + c^2)

theorem OH_squared_correct :
  ∀ (O H : Point) (a b c : ℝ) (R : ℝ),
    R = 7 →
    a^2 + b^2 + c^2 = 29 →
    OH_squared O H a b c R = 412 := by
  intros O H a b c R hR habc
  simp [OH_squared, hR, habc]
  sorry

end OH_squared_correct_l235_235056


namespace at_least_one_ge_one_l235_235847

theorem at_least_one_ge_one (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 2) : 
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ 1 :=
by 
  sorry

end at_least_one_ge_one_l235_235847


namespace total_fruits_on_display_l235_235869

-- Declare the variables and conditions as hypotheses
variables (apples oranges bananas : ℕ)
variables (h1 : apples = 2 * oranges)
variables (h2 : oranges = 2 * bananas)
variables (h3 : bananas = 5)

-- Define what we want to prove
theorem total_fruits_on_display : apples + oranges + bananas = 35 :=
by sorry

end total_fruits_on_display_l235_235869


namespace proposition_and_implication_l235_235646

theorem proposition_and_implication
  (m : ℝ)
  (h1 : 5/4 * (m^2 + m) > 0)
  (h2 : 1 + 9 - 4 * (5/4 * (m^2 + m)) > 0)
  (h3 : m + 3/2 ≥ 0)
  (h4 : m - 1/2 ≤ 0) :
  (-3/2 ≤ m ∧ m < -1) ∨ (0 < m ∧ m ≤ 1/2) :=
sorry

end proposition_and_implication_l235_235646


namespace expected_value_correct_prob_abs_diff_ge_1_correct_l235_235437

/-- Probability distribution for a single die roll -/
def prob_score (n : ℕ) : ℚ :=
  if n = 1 then 1/2 else if n = 2 then 1/3 else if n = 3 then 1/6 else 0

/-- Expected value based on the given probability distribution -/
def expected_value : ℚ := 
  (1 * prob_score 1) + (2 * prob_score 2) + (3 * prob_score 3)

/-- Proving the expected value calculation -/
theorem expected_value_correct : expected_value = 7/6 :=
  by sorry

/-- Calculate the probability of score difference being at least 1 between two players -/
def prob_abs_diff_ge_1 (x y : ℕ) : ℚ :=
  -- Implementation would involve detailed probability combinations that result in diff >= 1
  sorry

/-- Prove the probability of |x - y| being at least 1 -/
theorem prob_abs_diff_ge_1_correct : 
  ∀ (x y : ℕ), prob_abs_diff_ge_1 x y < 1 :=
  by sorry

end expected_value_correct_prob_abs_diff_ge_1_correct_l235_235437


namespace exists_m_for_division_l235_235904

theorem exists_m_for_division (n : ℕ) (h : 0 < n) : ∃ m : ℕ, n ∣ (2016 ^ m + m) := by
  sorry

end exists_m_for_division_l235_235904


namespace ratio_of_socks_l235_235272

-- Conditions:
variable (B : ℕ) (W : ℕ) (L : ℕ)
-- B = number of black socks
-- W = initial number of white socks
-- L = number of white socks lost

-- Setting given conditions:
axiom hB : B = 6
axiom hL : L = W / 2
axiom hCond : W / 2 = B + 6

-- Prove the ratio of white socks to black socks is 4:1
theorem ratio_of_socks : B = 6 → W / 2 = B + 6 → (W / 2) + (W / 2) = 24 → (B : ℚ) / (W : ℚ) = 1 / 4 :=
by intros hB hCond hW
   sorry

end ratio_of_socks_l235_235272


namespace base_extension_1_kilometer_l235_235450

-- Definition of the original triangle with hypotenuse length and inclination angle
def original_triangle (hypotenuse : ℝ) (angle : ℝ) : Prop :=
  hypotenuse = 1 ∧ angle = 20

-- Definition of the extension required for the new inclination angle
def extension_required (new_angle : ℝ) (extension : ℝ) : Prop :=
  new_angle = 10 ∧ extension = 1

-- The proof problem statement
theorem base_extension_1_kilometer :
  ∀ (hypotenuse : ℝ) (original_angle : ℝ) (new_angle : ℝ),
    original_triangle hypotenuse original_angle →
    new_angle = 10 →
    ∃ extension : ℝ, extension_required new_angle extension :=
by
  -- Sorry is a placeholder for the actual proof
  sorry

end base_extension_1_kilometer_l235_235450


namespace cosine_values_count_l235_235340

theorem cosine_values_count (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 360) (h3 : Real.cos x = -0.65) : 
  ∃ (n : ℕ), n = 2 := by
  sorry

end cosine_values_count_l235_235340


namespace largest_n_l235_235313

theorem largest_n (x y z n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12 → n ≤ 6 :=
by
  sorry

end largest_n_l235_235313


namespace crossing_time_indeterminate_l235_235262

-- Define the lengths of the two trains.
def train_A_length : Nat := 120
def train_B_length : Nat := 150

-- Define the crossing time of the two trains when moving in the same direction.
def crossing_time_together : Nat := 135

-- Define a theorem to state that without additional information, the crossing time for a 150-meter train cannot be determined.
theorem crossing_time_indeterminate 
    (V120 V150 : Nat) 
    (H : V150 - V120 = 2) : 
    ∃ t, t > 0 -> t < 150 / V150 -> False :=
by 
    -- The proof is not provided.
    sorry

end crossing_time_indeterminate_l235_235262


namespace sin_three_pi_over_two_l235_235946

theorem sin_three_pi_over_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end sin_three_pi_over_two_l235_235946


namespace find_piglets_l235_235044

theorem find_piglets (chickens piglets goats sick_animals : ℕ) 
  (h1 : chickens = 26) 
  (h2 : goats = 34) 
  (h3 : sick_animals = 50) 
  (h4 : (chickens + piglets + goats) / 2 = sick_animals) : piglets = 40 := 
by
  sorry

end find_piglets_l235_235044


namespace plates_remove_proof_l235_235553

noncomputable def total_weight_initial (plates: ℤ) (weight_per_plate: ℤ): ℤ :=
  plates * weight_per_plate

noncomputable def weight_limit (pounds: ℤ) (ounces_per_pound: ℤ): ℤ :=
  pounds * ounces_per_pound

noncomputable def plates_to_remove (initial_weight: ℤ) (limit: ℤ) (weight_per_plate: ℤ): ℤ :=
  (initial_weight - limit) / weight_per_plate

theorem plates_remove_proof :
  let pounds := 20
  let ounces_per_pound := 16
  let plates_initial := 38
  let weight_per_plate := 10
  let initial_weight := total_weight_initial plates_initial weight_per_plate
  let limit := weight_limit pounds ounces_per_pound
  plates_to_remove initial_weight limit weight_per_plate = 6 :=
by
  sorry

end plates_remove_proof_l235_235553


namespace find_f_neg1_l235_235853

-- Definitions based on conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable {b : ℝ} (f : ℝ → ℝ)

axiom odd_f : odd_function f
axiom f_form : ∀ x, 0 ≤ x → f x = 2^(x + 1) + 2 * x + b
axiom b_value : b = -2

theorem find_f_neg1 : f (-1) = -4 :=
sorry

end find_f_neg1_l235_235853


namespace basic_astrophysics_degrees_l235_235117

def budget_allocation : Nat := 100
def microphotonics_perc : Nat := 14
def home_electronics_perc : Nat := 19
def food_additives_perc : Nat := 10
def genetically_modified_perc : Nat := 24
def industrial_lubricants_perc : Nat := 8

def arc_of_sector (percentage : Nat) : Nat := percentage * 360 / budget_allocation

theorem basic_astrophysics_degrees :
  arc_of_sector (budget_allocation - (microphotonics_perc + home_electronics_perc + food_additives_perc + genetically_modified_perc + industrial_lubricants_perc)) = 90 :=
  by
  sorry

end basic_astrophysics_degrees_l235_235117


namespace part_one_part_two_l235_235913

section part_one
variables {x : ℝ}

def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

theorem part_one : ∀ x : ℝ, f x ≥ 3 ↔ (x ≤ 1 ∨ x ≥ 4) := by
  sorry
end part_one

section part_two
variables {a x : ℝ}

def g (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

theorem part_two : (∀ x ∈ (Set.Icc 1 2), g a x ≤ |x - 4|) → (a ∈ Set.Icc (-3) 0) := by
  sorry
end part_two

end part_one_part_two_l235_235913


namespace probability_sum_of_three_dice_is_9_l235_235748

def sum_of_three_dice_is_9 : Prop :=
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 9)

theorem probability_sum_of_three_dice_is_9 : 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 → a + b + c = 9 → sum_of_three_dice_is_9) ∧ 
  (1 / 216 = 25 / 216) := 
by
  sorry

end probability_sum_of_three_dice_is_9_l235_235748


namespace pencils_cost_proportion_l235_235515

/-- 
If a set of 15 pencils costs 9 dollars and the price of the set is directly 
proportional to the number of pencils it contains, then the cost of a set of 
35 pencils is 21 dollars.
--/
theorem pencils_cost_proportion :
  ∀ (p : ℕ), (∀ n : ℕ, n * 9 = p * 15) -> (35 * 9 = 21 * 15) :=
by
  intro p h1
  sorry

end pencils_cost_proportion_l235_235515


namespace sin_480_deg_l235_235261

theorem sin_480_deg : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_480_deg_l235_235261


namespace tan_theta_eq_neg_4_over_3_expression_eval_l235_235325

theorem tan_theta_eq_neg_4_over_3 (θ : ℝ) (h₁ : Real.sin θ = 4 / 5) (h₂ : Real.pi / 2 < θ ∧ θ < Real.pi) :
  Real.tan θ = -4 / 3 :=
sorry

theorem expression_eval (θ : ℝ) (h₁ : Real.sin θ = 4 / 5) (h₂ : Real.pi / 2 < θ ∧ θ < Real.pi) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (3 * Real.sin θ ^ 2 + Real.cos θ ^ 2) = 8 / 25 :=
sorry

end tan_theta_eq_neg_4_over_3_expression_eval_l235_235325


namespace problem_solution_l235_235516

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (x - 1)

theorem problem_solution (x : ℝ) : x ≥ 1 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 1)) = 2) ↔ (x = 13.25) :=
sorry

end problem_solution_l235_235516


namespace barneys_grocery_store_items_left_l235_235606

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

end barneys_grocery_store_items_left_l235_235606


namespace num_solutions_x_squared_minus_y_squared_eq_2001_l235_235394

theorem num_solutions_x_squared_minus_y_squared_eq_2001 
  (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2 - y^2 = 2001 ↔ (x, y) = (1001, 1000) ∨ (x, y) = (335, 332) := sorry

end num_solutions_x_squared_minus_y_squared_eq_2001_l235_235394


namespace number_of_students_in_all_events_l235_235079

variable (T A B : ℕ)

-- Defining given conditions
-- Total number of students in the class
def total_students : ℕ := 45
-- Number of students participating in the Soccer event
def soccer_students : ℕ := 39
-- Number of students participating in the Basketball event
def basketball_students : ℕ := 28

-- Main theorem to prove
theorem number_of_students_in_all_events
  (h_total : T = total_students)
  (h_soccer : A = soccer_students)
  (h_basketball : B = basketball_students) :
  ∃ x : ℕ, x = A + B - T := sorry

end number_of_students_in_all_events_l235_235079


namespace sin_C_value_area_of_triangle_l235_235049

open Real
open Classical

variable {A B C a b c : ℝ}

-- Given conditions
axiom h1 : b = sqrt 2
axiom h2 : c = 1
axiom h3 : cos B = 3 / 4

-- Proof statements
theorem sin_C_value : sin C = sqrt 14 / 8 := sorry

theorem area_of_triangle : 1 / 2 * b * c * sin (B + C) = sqrt 7 / 4 := sorry

end sin_C_value_area_of_triangle_l235_235049


namespace decryption_ease_comparison_l235_235625

def unique_letters_of_thermometer : Finset Char := {'т', 'е', 'р', 'м', 'о'}
def unique_letters_of_remont : Finset Char := {'р', 'е', 'м', 'о', 'н', 'т'}
def easier_to_decrypt : Prop :=
  unique_letters_of_remont.card > unique_letters_of_thermometer.card

theorem decryption_ease_comparison : easier_to_decrypt :=
by
  -- We need to prove that |unique_letters_of_remont| > |unique_letters_of_thermometer|
  sorry

end decryption_ease_comparison_l235_235625


namespace solve_m_l235_235290

theorem solve_m (m : ℝ) : 
  (m - 3) * x^2 - 3 * x + m^2 = 9 → m^2 - 9 = 0 → m = -3 :=
by
  sorry

end solve_m_l235_235290


namespace at_most_one_existence_l235_235029

theorem at_most_one_existence
  (p : ℕ) (hp : Nat.Prime p)
  (A B : Finset (Fin p))
  (h_non_empty_A : A.Nonempty) (h_non_empty_B : B.Nonempty)
  (h_union : A ∪ B = Finset.univ) (h_disjoint : A ∩ B = ∅) :
  ∃! a : Fin p, ¬ (∃ x y : Fin p, (x ∈ A ∧ y ∈ B ∧ x + y = a) ∨ (x + y = a + p)) :=
sorry

end at_most_one_existence_l235_235029


namespace probability_route_X_is_8_over_11_l235_235812

-- Definitions for the graph paths and probabilities
def routes_from_A_to_B (X Y : Nat) : Nat := 2 + 6 + 3

def routes_passing_through_X (X Y : Nat) : Nat := 2 + 6

def probability_passing_through_X (total_routes passing_routes : Nat) : Rat :=
  (passing_routes : Rat) / (total_routes : Rat)

theorem probability_route_X_is_8_over_11 :
  let total_routes := routes_from_A_to_B 2 3
  let passing_routes := routes_passing_through_X 2 3
  probability_passing_through_X total_routes passing_routes = 8 / 11 :=
by
  -- Assumes correct route calculations from the conditions and aims to prove the probability value
  sorry

end probability_route_X_is_8_over_11_l235_235812


namespace determine_q_l235_235776

theorem determine_q (q : ℕ) (h : 81^10 = 3^q) : q = 40 :=
by
  sorry

end determine_q_l235_235776


namespace helen_choc_chip_yesterday_l235_235226

variable (total_cookies morning_cookies : ℕ)

theorem helen_choc_chip_yesterday :
  total_cookies = 1081 →
  morning_cookies = 554 →
  total_cookies - morning_cookies = 527 := by
  sorry

end helen_choc_chip_yesterday_l235_235226


namespace salary_increase_after_three_years_l235_235109

-- Define the initial salary S and the raise percentage 12%
def initial_salary (S : ℝ) : ℝ := S
def raise_percentage : ℝ := 0.12

-- Define the salary after n raises
def salary_after_raises (S : ℝ) (n : ℕ) : ℝ :=
  S * (1 + raise_percentage)^n

-- Prove that the percentage increase after 3 years is 40.49%
theorem salary_increase_after_three_years (S : ℝ) :
  ((salary_after_raises S 3 - S) / S) * 100 = 40.49 :=
by sorry

end salary_increase_after_three_years_l235_235109


namespace difference_thursday_tuesday_l235_235526

-- Define the amounts given on each day
def amount_tuesday : ℕ := 8
def amount_wednesday : ℕ := 5 * amount_tuesday
def amount_thursday : ℕ := amount_wednesday + 9

-- Problem statement: prove that the difference between Thursday's and Tuesday's amount is $41
theorem difference_thursday_tuesday : amount_thursday - amount_tuesday = 41 := by
  sorry

end difference_thursday_tuesday_l235_235526


namespace possible_values_of_g_zero_l235_235366

variable {g : ℝ → ℝ}

theorem possible_values_of_g_zero (h : ∀ x : ℝ, g (2 * x) = g x ^ 2) : g 0 = 0 ∨ g 0 = 1 := 
sorry

end possible_values_of_g_zero_l235_235366


namespace solve_for_x_l235_235084

theorem solve_for_x (x : ℝ) (h : 3 * x + 1 = -(5 - 2 * x)) : x = -6 :=
by
  sorry

end solve_for_x_l235_235084


namespace max_min_diff_w_l235_235710

theorem max_min_diff_w (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 4) :
  let w := a^2 + a*b + b^2
  let w1 := max (0^2 + 0*b + b^2) (4^2 + 4*b + b^2)
  let w2 := (2-2)^2 + 12
  w1 - w2 = 4 :=
by
  -- skip the proof
  sorry

end max_min_diff_w_l235_235710


namespace gcd_6Pn_n_minus_2_l235_235457

-- Auxiliary definition to calculate the nth pentagonal number
def pentagonal (n : ℕ) : ℕ := n ^ 2

-- Statement of the theorem
theorem gcd_6Pn_n_minus_2 (n : ℕ) (hn : 0 < n) : 
  ∃ d, d = Int.gcd (6 * pentagonal n) (n - 2) ∧ d ≤ 24 ∧ (∀ k, Int.gcd (6 * pentagonal k) (k - 2) ≤ 24) :=
sorry

end gcd_6Pn_n_minus_2_l235_235457


namespace equation_of_trajectory_l235_235216

open Real

variable (P : ℝ → ℝ → Prop)
variable (C : ℝ → ℝ → Prop)
variable (L : ℝ → ℝ → Prop)

-- Definition of the fixed circle C
def fixed_circle (x y : ℝ) : Prop :=
  (x + 2) ^ 2 + y ^ 2 = 1

-- Definition of the fixed line L
def fixed_line (x y : ℝ) : Prop := 
  x = 1

noncomputable def moving_circle (P : ℝ → ℝ → Prop) (r : ℝ) : Prop :=
  ∃ x y : ℝ, P x y ∧ r > 0 ∧
  (∀ a b : ℝ, fixed_circle a b → ((x - a) ^ 2 + (y - b) ^ 2) = (r + 1) ^ 2) ∧
  (∀ a b : ℝ, fixed_line a b → (abs (x - a)) = (r + 1))

theorem equation_of_trajectory
  (P : ℝ → ℝ → Prop)
  (r : ℝ)
  (h : moving_circle P r) :
  ∀ x y : ℝ, P x y → y ^ 2 = -8 * x :=
by
  sorry

end equation_of_trajectory_l235_235216


namespace one_hundred_fifty_sixth_digit_is_five_l235_235403

def repeated_sequence := [0, 6, 0, 5, 1, 3]
def target_index := 156 - 1
def block_length := repeated_sequence.length

theorem one_hundred_fifty_sixth_digit_is_five :
  repeated_sequence[target_index % block_length] = 5 :=
by
  sorry

end one_hundred_fifty_sixth_digit_is_five_l235_235403


namespace larger_number_is_1671_l235_235069

variable (L S : ℕ)

noncomputable def problem_conditions :=
  L - S = 1395 ∧ L = 6 * S + 15

theorem larger_number_is_1671 (h : problem_conditions L S) : L = 1671 := by
  sorry

end larger_number_is_1671_l235_235069


namespace european_math_school_gathering_l235_235941

theorem european_math_school_gathering :
  ∃ n : ℕ, n < 400 ∧ n % 17 = 16 ∧ n % 19 = 12 ∧ n = 288 :=
by
  sorry

end european_math_school_gathering_l235_235941


namespace ratio_costs_equal_l235_235698

noncomputable def cost_first_8_years : ℝ := 10000 * 8
noncomputable def john_share_first_8_years : ℝ := cost_first_8_years / 2
noncomputable def university_tuition : ℝ := 250000
noncomputable def john_share_university : ℝ := university_tuition / 2
noncomputable def total_paid_by_john : ℝ := 265000
noncomputable def cost_between_8_and_18 : ℝ := total_paid_by_john - john_share_first_8_years - john_share_university
noncomputable def cost_per_year_8_to_18 : ℝ := cost_between_8_and_18 / 10
noncomputable def cost_per_year_first_8_years : ℝ := 10000

theorem ratio_costs_equal : cost_per_year_8_to_18 / cost_per_year_first_8_years = 1 := by
  sorry

end ratio_costs_equal_l235_235698


namespace midpoint_pentagon_inequality_l235_235575

noncomputable def pentagon_area_midpoints (T : ℝ) : ℝ := sorry

theorem midpoint_pentagon_inequality {T t : ℝ} 
  (h1 : t = pentagon_area_midpoints T)
  (h2 : 0 < T) : 
  (3/4) * T > t ∧ t > (1/2) * T :=
  sorry

end midpoint_pentagon_inequality_l235_235575


namespace age_of_B_is_23_l235_235021

-- Definitions of conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 87)
variable (h2 : A + C = 64)

-- Statement of the problem
theorem age_of_B_is_23 : B = 23 :=
by { sorry }

end age_of_B_is_23_l235_235021


namespace william_marbles_l235_235473

theorem william_marbles :
  let initial_marbles := 10
  let shared_marbles := 3
  (initial_marbles - shared_marbles) = 7 := 
by
  sorry

end william_marbles_l235_235473


namespace no_even_and_increasing_function_l235_235376

-- Definition of a function being even
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Definition of a function being increasing
def is_increasing_function (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≤ f y

-- Theorem stating the non-existence of a function that is both even and increasing
theorem no_even_and_increasing_function : ¬ ∃ f : ℝ → ℝ, is_even_function f ∧ is_increasing_function f :=
by
  sorry

end no_even_and_increasing_function_l235_235376


namespace value_of_Z_4_3_l235_235322

def Z (a b : ℤ) : ℤ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem value_of_Z_4_3 : Z 4 3 = 1 := by
  sorry

end value_of_Z_4_3_l235_235322


namespace coefficient_of_x2_in_expansion_l235_235723

def binomial_coefficient (n k : Nat) : Nat := Nat.choose k n

def binomial_term (a x : ℕ) (n r : ℕ) : ℕ :=
  a^(n-r) * binomial_coefficient n r * x^r

theorem coefficient_of_x2_in_expansion : 
  binomial_term 2 1 5 2 = 80 := by sorry

end coefficient_of_x2_in_expansion_l235_235723


namespace binom_divisible_by_4_l235_235523

theorem binom_divisible_by_4 (n : ℕ) : (n ≠ 0) ∧ (¬ (∃ k : ℕ, n = 2^k)) ↔ 4 ∣ n * (Nat.choose (2 * n) n) :=
by
  sorry

end binom_divisible_by_4_l235_235523


namespace pies_sold_each_day_l235_235199

theorem pies_sold_each_day (total_pies : ℕ) (days_in_week : ℕ) (h1 : total_pies = 56) (h2 : days_in_week = 7) :
  (total_pies / days_in_week = 8) :=
by
exact sorry

end pies_sold_each_day_l235_235199


namespace total_minutes_to_finish_album_l235_235983

variable (initial_songs : ℕ) (additional_songs : ℕ) (duration : ℕ)

theorem total_minutes_to_finish_album 
  (h1: initial_songs = 25) 
  (h2: additional_songs = 10) 
  (h3: duration = 3) :
  (initial_songs + additional_songs) * duration = 105 :=
sorry

end total_minutes_to_finish_album_l235_235983


namespace fourth_divisor_of_9600_l235_235260

theorem fourth_divisor_of_9600 (x : ℕ) (h1 : ∀ (d : ℕ), d = 15 ∨ d = 25 ∨ d = 40 → 9600 % d = 0) 
  (h2 : 9600 / Nat.lcm (Nat.lcm 15 25) 40 = x) : x = 16 := by
  sorry

end fourth_divisor_of_9600_l235_235260


namespace smallest_angle_range_l235_235390

theorem smallest_angle_range {A B C : ℝ} (hA : 0 < A) (hABC : A + B + C = 180) (horder : A ≤ B ∧ B ≤ C) :
  0 < A ∧ A ≤ 60 := by
  sorry

end smallest_angle_range_l235_235390


namespace carlos_goal_l235_235119

def july_books : ℕ := 28
def august_books : ℕ := 30
def june_books : ℕ := 42

theorem carlos_goal (goal : ℕ) :
  goal = june_books + july_books + august_books := by
  sorry

end carlos_goal_l235_235119


namespace check_correct_options_l235_235601

noncomputable def f (x a b: ℝ) := x^3 - a*x^2 + b*x + 1

theorem check_correct_options :
  (∀ (b: ℝ), b = 0 → ¬(∃ x: ℝ, 3 * x^2 - 2 * a * x = 0)) ∧
  (∀ (a: ℝ), a = 0 → (∀ x: ℝ, f x a b + f (-x) a b = 2)) ∧
  (∀ (a: ℝ), ∀ (b: ℝ), b = a^2 / 4 ∧ a > -4 → ∃ x1 x2 x3: ℝ, f x1 a b = 0 ∧ f x2 a b = 0 ∧ f x3 a b = 0) ∧
  (∀ (a: ℝ), ∀ (b: ℝ), (∀ x: ℝ, 3 * x^2 - 2 * a * x + b ≥ 0) → (a^2 ≤ 3*b)) := sorry

end check_correct_options_l235_235601


namespace problem_III_l235_235384

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / x

theorem problem_III
  (a x1 x2 : ℝ)
  (h_a : 0 < a ∧ a < 1)
  (h_roots : f x1 = a ∧ f x2 = a)
  (h_order : x1 < x2)
  (h_bounds : Real.exp (-1) < x1 ∧ x1 < 1 ∧ 1 < x2) :
  x2 - x1 > 1 / a - 1 :=
sorry

end problem_III_l235_235384


namespace ellipse_k_range_l235_235043

theorem ellipse_k_range
  (k : ℝ)
  (h1 : k - 4 > 0)
  (h2 : 10 - k > 0)
  (h3 : k - 4 > 10 - k) :
  7 < k ∧ k < 10 :=
sorry

end ellipse_k_range_l235_235043


namespace max_gcd_lcm_condition_l235_235244

theorem max_gcd_lcm_condition (a b c : ℕ) (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) : gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_condition_l235_235244


namespace rectangular_solid_volume_l235_235398

theorem rectangular_solid_volume 
  (a b c : ℝ) 
  (h1 : a * b = 18) 
  (h2 : b * c = 50) 
  (h3 : a * c = 45) : 
  a * b * c = 150 * Real.sqrt 3 := 
by 
  sorry

end rectangular_solid_volume_l235_235398


namespace right_side_longer_l235_235358

/-- The sum of the three sides of a triangle is 50. 
    The right side of the triangle is a certain length longer than the left side, which has a value of 12 cm. 
    The triangle base has a value of 24 cm. 
    Prove that the right side is 2 cm longer than the left side. -/
theorem right_side_longer (L R B : ℝ) (hL : L = 12) (hB : B = 24) (hSum : L + B + R = 50) : R = L + 2 :=
by
  sorry

end right_side_longer_l235_235358


namespace fraction_calculation_l235_235907

theorem fraction_calculation :
  ( (3 / 7 + 5 / 8 + 1 / 3) / (5 / 12 + 2 / 9) = 2097 / 966 ) :=
by
  sorry

end fraction_calculation_l235_235907


namespace smallest_possible_value_l235_235533

theorem smallest_possible_value (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) (h2 : n ≡ 2 [MOD 9]) (h3 : n ≡ 6 [MOD 7]) :
  n = 116 :=
by
  -- Proof omitted
  sorry

end smallest_possible_value_l235_235533


namespace remainder_div_x_minus_4_l235_235399

def f (x : ℕ) : ℕ := x^5 - 8 * x^4 + 16 * x^3 + 25 * x^2 - 50 * x + 24

theorem remainder_div_x_minus_4 : 
  (f 4) = 224 := 
by 
  -- Proof goes here
  sorry

end remainder_div_x_minus_4_l235_235399


namespace min_time_shoe_horses_l235_235973

variable (blacksmiths horses hooves_per_horse minutes_per_hoof : ℕ)
variable (total_time : ℕ)

theorem min_time_shoe_horses (h_blacksmiths : blacksmiths = 48) 
                            (h_horses : horses = 60)
                            (h_hooves_per_horse : hooves_per_horse = 4)
                            (h_minutes_per_hoof : minutes_per_hoof = 5)
                            (h_total_time : total_time = (horses * hooves_per_horse * minutes_per_hoof) / blacksmiths) :
                            total_time = 25 := 
by
  sorry

end min_time_shoe_horses_l235_235973


namespace max_writers_and_editors_l235_235407

theorem max_writers_and_editors (T W : ℕ) (E : ℕ) (x : ℕ) (hT : T = 100) (hW : W = 35) (hE : E > 38) (h_comb : W + E + x = T)
    (h_neither : T = W + E + x) : x = 26 := by
  sorry

end max_writers_and_editors_l235_235407


namespace impossible_digit_filling_l235_235254

theorem impossible_digit_filling (T : Fin 5 → Fin 8 → Fin 10) :
  (∀ d : Fin 10, (∃! r₁ r₂ r₃ r₄ : Fin 5, T r₁ = d ∧ T r₂ = d ∧ T r₃ = d ∧ T r₄ = d) ∧
                 (∃! c₁ c₂ c₃ c₄ : Fin 8, T c₁ = d ∧ T c₂ = d ∧ T c₃ = d ∧ T c₄ = d)) → False :=
by
  sorry

end impossible_digit_filling_l235_235254


namespace sheep_to_cow_water_ratio_l235_235749

-- Set up the initial conditions
def number_of_cows := 40
def water_per_cow_per_day := 80
def number_of_sheep := 10 * number_of_cows
def total_water_per_week := 78400

-- Calculate total water consumption of cows per week
def water_cows_per_week := number_of_cows * water_per_cow_per_day * 7

-- Calculate total water consumption of sheep per week
def water_sheep_per_week := total_water_per_week - water_cows_per_week

-- Calculate daily water consumption per sheep
def water_sheep_per_day := water_sheep_per_week / 7
def daily_water_per_sheep := water_sheep_per_day / number_of_sheep

-- Define the target ratio
def target_ratio := 1 / 4

-- Statement to prove
theorem sheep_to_cow_water_ratio :
  (daily_water_per_sheep / water_per_cow_per_day) = target_ratio :=
sorry

end sheep_to_cow_water_ratio_l235_235749


namespace ratio_H_over_G_l235_235416

theorem ratio_H_over_G (G H : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    (G : ℝ)/(x + 5) + (H : ℝ)/(x^2 - 4*x) = (x^2 - 2*x + 10) / (x^3 + x^2 - 20*x)) :
  H / G = 2 :=
  sorry

end ratio_H_over_G_l235_235416


namespace prob_yellow_is_3_over_5_required_red_balls_is_8_l235_235302

-- Defining the initial conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 4
def yellow_balls : ℕ := 6

-- Part 1: Prove the probability of drawing a yellow ball is 3/5
theorem prob_yellow_is_3_over_5 :
  (yellow_balls : ℚ) / (total_balls : ℚ) = 3 / 5 := sorry

-- Part 2: Prove that adding 8 red balls makes the probability of drawing a red ball 2/3
theorem required_red_balls_is_8 (x : ℕ) :
  (red_balls + x : ℚ) / (total_balls + x : ℚ) = 2 / 3 → x = 8 := sorry

end prob_yellow_is_3_over_5_required_red_balls_is_8_l235_235302


namespace ratio_of_roots_l235_235768

theorem ratio_of_roots (c : ℝ) :
  (∃ (x1 x2 : ℝ), 5 * x1^2 - 2 * x1 + c = 0 ∧ 5 * x2^2 - 2 * x2 + c = 0 ∧ x1 / x2 = -3 / 5) → c = -3 :=
by
  sorry

end ratio_of_roots_l235_235768


namespace fraction_ratio_l235_235784

theorem fraction_ratio (x y : ℕ) (h : (x / y : ℚ) / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 27 ∧ y = 35 :=
by 
  sorry

end fraction_ratio_l235_235784


namespace jessica_current_age_l235_235050

-- Define the conditions
def jessicaOlderThanClaire (jessica claire : ℕ) : Prop :=
  jessica = claire + 6

def claireAgeInTwoYears (claire : ℕ) : Prop :=
  claire + 2 = 20

-- State the theorem to prove
theorem jessica_current_age : ∃ jessica claire : ℕ, 
  jessicaOlderThanClaire jessica claire ∧ claireAgeInTwoYears claire ∧ jessica = 24 := 
sorry

end jessica_current_age_l235_235050


namespace cross_section_prism_in_sphere_l235_235448

noncomputable def cross_section_area 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) : ℝ :=
  a * Real.sqrt (4 * R^2 - a^2)

theorem cross_section_prism_in_sphere 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) :
  cross_section_area a R h1 h2 h3 = a * Real.sqrt (4 * R^2 - a^2) := 
  by
    sorry

end cross_section_prism_in_sphere_l235_235448


namespace mass_percentage_of_Cl_in_compound_l235_235102

theorem mass_percentage_of_Cl_in_compound (mass_percentage_Cl : ℝ) (h : mass_percentage_Cl = 92.11) : mass_percentage_Cl = 92.11 :=
sorry

end mass_percentage_of_Cl_in_compound_l235_235102


namespace arcade_ticket_problem_l235_235256

-- Define all the conditions given in the problem
def initial_tickets : Nat := 13
def used_tickets : Nat := 8
def more_tickets_for_clothes : Nat := 10
def tickets_for_toys : Nat := 8
def tickets_for_clothes := tickets_for_toys + more_tickets_for_clothes

-- The proof statement (goal)
theorem arcade_ticket_problem : tickets_for_clothes = 18 := by
  -- This is where the proof would go
  sorry

end arcade_ticket_problem_l235_235256


namespace opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l235_235415

theorem opposite_of_neg23_eq_23 : -(-23) = 23 := 
by sorry

theorem reciprocal_of_neg23_eq_neg_1_div_23 : (1 : ℚ) / (-23) = -(1 / 23 : ℚ) :=
by sorry

theorem abs_value_of_neg23_eq_23 : abs (-23) = 23 :=
by sorry

end opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l235_235415


namespace find_f_l235_235700

-- Define the conditions
def g (x : ℝ) : ℝ := 2 * x + 3
def f (x : ℝ) : ℝ := g (x + 2)

-- State the theorem
theorem find_f :
  ∀ x : ℝ, f x = 2 * x + 7 :=
by
  sorry

end find_f_l235_235700


namespace bens_old_car_cost_l235_235586

theorem bens_old_car_cost :
  ∃ (O N : ℕ), N = 2 * O ∧ O = 1800 ∧ N = 1800 + 2000 ∧ O = 1900 :=
by 
  sorry

end bens_old_car_cost_l235_235586


namespace sugar_needed_in_two_minutes_l235_235597

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

end sugar_needed_in_two_minutes_l235_235597


namespace vector_dot_product_l235_235829

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Vector addition and scalar multiplication
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematical statement to prove
theorem vector_dot_product : dot_product (vec_add (scalar_mul 2 vec_a) vec_b) vec_a = 6 :=
by
  -- Sorry is used to skip the proof; it's just a placeholder.
  sorry

end vector_dot_product_l235_235829


namespace inscribed_sphere_radius_l235_235908

theorem inscribed_sphere_radius 
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (R : ℝ) :
  (1 / 3) * R * (S1 + S2 + S3 + S4) = V ↔ R = (3 * V) / (S1 + S2 + S3 + S4) := 
by
  sorry

end inscribed_sphere_radius_l235_235908


namespace apply_f_2019_times_l235_235083

noncomputable def f (x : ℝ) : ℝ := (1 - x^3) ^ (-1/3 : ℝ)

theorem apply_f_2019_times (x : ℝ) (n : ℕ) (h : n = 2019) (hx : x = 2018) : 
  (f^[n]) x = 2018 :=
by
  sorry

end apply_f_2019_times_l235_235083


namespace motorcyclist_travel_time_l235_235186

-- Define the conditions and the proof goal:
theorem motorcyclist_travel_time :
  ∀ (z : ℝ) (t₁ t₂ t₃ : ℝ),
    t₂ = 60 →
    t₃ = 3240 →
    (t₃ - 5) / (z / 40 - z / t₁) = 10 →
    t₃ / (z / 40) = 10 + t₂ / (z / 60 - z / t₁) →
    t₁ = 80 :=
by
  intros z t₁ t₂ t₃ h1 h2 h3 h4
  sorry

end motorcyclist_travel_time_l235_235186


namespace gcf_lcm_360_210_l235_235070

theorem gcf_lcm_360_210 :
  let factorization_360 : ℕ × ℕ × ℕ × ℕ := (3, 2, 1, 0) -- Prime exponents for 2, 3, 5, 7
  let factorization_210 : ℕ × ℕ × ℕ × ℕ := (1, 1, 1, 1) -- Prime exponents for 2, 3, 5, 7
  gcd (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 30 ∧
  lcm (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 2520 :=
by {
  let factorization_360 := (3, 2, 1, 0)
  let factorization_210 := (1, 1, 1, 1)
  sorry
}

end gcf_lcm_360_210_l235_235070


namespace bigger_part_l235_235419

theorem bigger_part (x y : ℕ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) : y = 34 :=
sorry

end bigger_part_l235_235419


namespace gcd_18_30_is_6_l235_235879

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l235_235879


namespace prove_op_eq_l235_235801

-- Define the new operation ⊕
def op (x y : ℝ) := x^3 - 2*y + x

-- State that for any k, k ⊕ (k ⊕ k) = -k^3 + 3k
theorem prove_op_eq (k : ℝ) : op k (op k k) = -k^3 + 3*k :=
by 
  sorry

end prove_op_eq_l235_235801


namespace card_draw_probability_l235_235289

theorem card_draw_probability :
  (13 / 52) * (13 / 51) * (13 / 50) = 2197 / 132600 :=
by
  sorry

end card_draw_probability_l235_235289


namespace sum_due_is_42_l235_235844

-- Define the conditions
def BD : ℝ := 42
def TD : ℝ := 36

-- Statement to prove
theorem sum_due_is_42 (H1 : BD = 42) (H2 : TD = 36) : ∃ (FV : ℝ), FV = 42 := by
  -- Proof Placeholder
  sorry

end sum_due_is_42_l235_235844


namespace find_d_l235_235632

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end find_d_l235_235632


namespace sum_of_n_plus_k_l235_235793

theorem sum_of_n_plus_k (n k : ℕ) (h1 : 2 * (n - k) = 3 * (k + 1)) (h2 : 3 * (n - k - 1) = 4 * (k + 2)) : n + k = 47 := by
  sorry

end sum_of_n_plus_k_l235_235793


namespace small_slices_sold_l235_235790

theorem small_slices_sold (S L : ℕ) 
  (h1 : S + L = 5000) 
  (h2 : 150 * S + 250 * L = 1050000) : 
  S = 2000 :=
by
  sorry

end small_slices_sold_l235_235790


namespace leo_assignment_third_part_time_l235_235953

-- Define all the conditions as variables
def first_part_time : ℕ := 25
def first_break : ℕ := 10
def second_part_time : ℕ := 2 * first_part_time
def second_break : ℕ := 15
def total_time : ℕ := 150

-- The calculated total time of the first two parts and breaks
def time_spent_on_first_two_parts_and_breaks : ℕ :=
  first_part_time + first_break + second_part_time + second_break

-- The remaining time for the third part of the assignment
def third_part_time : ℕ :=
  total_time - time_spent_on_first_two_parts_and_breaks

-- The theorem to prove that the time Leo took to finish the third part is 50 minutes
theorem leo_assignment_third_part_time : third_part_time = 50 := by
  sorry

end leo_assignment_third_part_time_l235_235953


namespace range_of_a_l235_235814

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x - 1 else x ^ 2 + 1

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (3 / 2 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l235_235814


namespace length_DC_of_ABCD_l235_235857

open Real

structure Trapezoid (ABCD : Type) :=
  (AB DC : ℝ)
  (BC : ℝ := 0)
  (angleBCD angleCDA : ℝ)

noncomputable def given_trapezoid : Trapezoid ℝ :=
{ AB := 5,
  DC := 8 + sqrt 3, -- this is from the answer
  BC := 3 * sqrt 2,
  angleBCD := π / 4,   -- 45 degrees in radians
  angleCDA := π / 3 }  -- 60 degrees in radians

variable (ABCD : Trapezoid ℝ)

theorem length_DC_of_ABCD :
  ABCD.AB = 5 ∧
  ABCD.BC = 3 * sqrt 2 ∧
  ABCD.angleBCD = π / 4 ∧
  ABCD.angleCDA = π / 3 →
  ABCD.DC = 8 + sqrt 3 :=
sorry

end length_DC_of_ABCD_l235_235857


namespace luca_loss_years_l235_235886

variable (months_in_year : ℕ := 12)
variable (barbi_kg_per_month : ℚ := 1.5)
variable (luca_kg_per_year : ℚ := 9)
variable (luca_additional_kg : ℚ := 81)

theorem luca_loss_years (barbi_yearly_loss : ℚ :=
                          barbi_kg_per_month * months_in_year) :
  (81 + barbi_yearly_loss) / luca_kg_per_year = 11 := by
  let total_loss_by_luca := 81 + barbi_yearly_loss
  sorry

end luca_loss_years_l235_235886


namespace find_a_l235_235884

theorem find_a (a : ℚ) (h : ∃ r s : ℚ, (r*x + s)^2 = ax^2 + 18*x + 16) : a = 81 / 16 := 
by sorry 

end find_a_l235_235884


namespace redistribution_not_always_possible_l235_235952

theorem redistribution_not_always_possible (a b : ℕ) (h : a ≠ b) :
  ¬(∃ k : ℕ, a - k = b + k ∧ 0 ≤ k ∧ k ≤ a ∧ k ≤ b) ↔ (a + b) % 2 = 1 := 
by 
  sorry

end redistribution_not_always_possible_l235_235952


namespace number_that_divides_and_leaves_remainder_54_l235_235711

theorem number_that_divides_and_leaves_remainder_54 :
  ∃ n : ℕ, n > 0 ∧ (55 ^ 55 + 55) % n = 54 ∧ n = 56 :=
by
  sorry

end number_that_divides_and_leaves_remainder_54_l235_235711


namespace total_gas_cost_l235_235697

def gas_price_station_1 : ℝ := 3
def gas_price_station_2 : ℝ := 3.5
def gas_price_station_3 : ℝ := 4
def gas_price_station_4 : ℝ := 4.5
def tank_capacity : ℝ := 12

theorem total_gas_cost :
  let cost_station_1 := tank_capacity * gas_price_station_1
  let cost_station_2 := tank_capacity * gas_price_station_2
  let cost_station_3 := tank_capacity * gas_price_station_3
  let cost_station_4 := tank_capacity * gas_price_station_4
  cost_station_1 + cost_station_2 + cost_station_3 + cost_station_4 = 180 :=
by
  -- Proof is skipped
  sorry

end total_gas_cost_l235_235697


namespace polynomial_roots_identity_l235_235521

theorem polynomial_roots_identity {p q α β γ δ : ℝ} 
  (h1 : α^2 + p*α + 1 = 0)
  (h2 : β^2 + p*β + 1 = 0)
  (h3 : γ^2 + q*γ + 1 = 0)
  (h4 : δ^2 + q*δ + 1 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end polynomial_roots_identity_l235_235521


namespace april_roses_l235_235517

theorem april_roses (price_per_rose earnings number_of_roses_left : ℕ) 
  (h1 : price_per_rose = 7) 
  (h2 : earnings = 35) 
  (h3 : number_of_roses_left = 4) : 
  (earnings / price_per_rose + number_of_roses_left) = 9 :=
by
  sorry

end april_roses_l235_235517


namespace range_of_m_l235_235706

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4 * m - 5) * x^2 - 4 * (m - 1) * x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by
  sorry

end range_of_m_l235_235706


namespace sum_of_squares_not_divisible_by_13_l235_235003

theorem sum_of_squares_not_divisible_by_13
  (x y z : ℤ)
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_xz : Int.gcd x z = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_sum : (x + y + z) % 13 = 0)
  (h_prod : (x * y * z) % 13 = 0) :
  (x^2 + y^2 + z^2) % 13 ≠ 0 := by
  sorry

end sum_of_squares_not_divisible_by_13_l235_235003


namespace boat_problem_l235_235834

theorem boat_problem (x y : ℕ) (h : 12 * x + 5 * y = 99) :
  (x = 2 ∧ y = 15) ∨ (x = 7 ∧ y = 3) :=
sorry

end boat_problem_l235_235834


namespace vector_parallel_solution_l235_235799

theorem vector_parallel_solution (x : ℝ) :
  let a := (1, x)
  let b := (x - 1, 2)
  (a.1 * b.2 = a.2 * b.1) → (x = 2 ∨ x = -1) :=
by
  intros a b h
  let a := (1, x)
  let b := (x - 1, 2)
  sorry

end vector_parallel_solution_l235_235799


namespace cookies_baked_total_l235_235992

   -- Definitions based on the problem conditions
   def cookies_yesterday : ℕ := 435
   def cookies_this_morning : ℕ := 139

   -- The theorem we want to prove
   theorem cookies_baked_total : cookies_yesterday + cookies_this_morning = 574 :=
   by sorry
   
end cookies_baked_total_l235_235992


namespace felicity_gas_usage_l235_235548

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end felicity_gas_usage_l235_235548


namespace smallest_n_mod_l235_235227

theorem smallest_n_mod :
  ∃ n : ℕ, (23 * n ≡ 5678 [MOD 11]) ∧ (∀ m : ℕ, (23 * m ≡ 5678 [MOD 11]) → (0 < n) ∧ (n ≤ m)) :=
  by
  sorry

end smallest_n_mod_l235_235227


namespace set_listing_l235_235833

open Set

def A : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (x, y) ∧ x^2 = y + 1 ∧ |x| < 2}

theorem set_listing :
  A = {(-1, 0), (0, -1), (1, 0)} :=
by {
  sorry
}

end set_listing_l235_235833


namespace weight_of_mixture_correct_l235_235427

-- Defining the fractions of each component in the mixture
def sand_fraction : ℚ := 2 / 9
def water_fraction : ℚ := 5 / 18
def gravel_fraction : ℚ := 1 / 6
def cement_fraction : ℚ := 7 / 36
def limestone_fraction : ℚ := 1 - sand_fraction - water_fraction - gravel_fraction - cement_fraction

-- Given weight of limestone
def limestone_weight : ℚ := 12

-- Total weight of the mixture that we need to prove
def total_mixture_weight : ℚ := 86.4

-- Proof problem statement
theorem weight_of_mixture_correct : (limestone_fraction * total_mixture_weight = limestone_weight) :=
by
  have h_sand := sand_fraction
  have h_water := water_fraction
  have h_gravel := gravel_fraction
  have h_cement := cement_fraction
  have h_limestone := limestone_fraction
  have h_limestone_weight := limestone_weight
  have h_total_weight := total_mixture_weight
  sorry

end weight_of_mixture_correct_l235_235427


namespace complex_multiplication_l235_235259

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  ((a + b * i) * (c + d * i)) = (-6 + 33 * i) :=
by
  have a := 3
  have b := -4
  have c := -6
  have d := 3
  sorry

end complex_multiplication_l235_235259


namespace fraction_equivalence_1_algebraic_identity_l235_235582

/-- First Problem: Prove the equivalence of the fractions 171717/252525 and 17/25. -/
theorem fraction_equivalence_1 : 
  (171717 : ℚ) / 252525 = 17 / 25 := 
sorry

/-- Second Problem: Prove the equivalence of the algebraic expressions on both sides. -/
theorem algebraic_identity (a b : ℚ) : 
  2 * b^5 + (a^4 + a^3 * b + a^2 * b^2 + a * b^3 + b^4) * (a - b) = 
  (a^4 - a^3 * b + a^2 * b^2 - a * b^3 + b^4) * (a + b) := 
sorry

end fraction_equivalence_1_algebraic_identity_l235_235582


namespace gcd_65_130_l235_235380

theorem gcd_65_130 : Int.gcd 65 130 = 65 := by
  sorry

end gcd_65_130_l235_235380


namespace problem_I_l235_235511

def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_I {x : ℝ} : f (x + 3 / 2) ≥ 0 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end problem_I_l235_235511


namespace find_a_l235_235305

theorem find_a (a : ℝ) : 
  (a + 3)^2 = (a + 1)^2 + (a + 2)^2 → a = 2 := 
by
  intro h
  -- Proof should go here
  sorry

end find_a_l235_235305


namespace math_pattern_l235_235957

theorem math_pattern (n : ℕ) : (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by
  sorry

end math_pattern_l235_235957


namespace find_number_l235_235779

theorem find_number (x : ℝ) (h : 4 * (3 * x / 5 - 220) = 320) : x = 500 :=
sorry

end find_number_l235_235779


namespace smallest_rational_number_l235_235349

theorem smallest_rational_number : ∀ (a b c d : ℚ), (a = -3) → (b = -1) → (c = 0) → (d = 1) → (a < b ∧ a < c ∧ a < d) :=
by
  intros a b c d h₁ h₂ h₃ h₄
  have h₅ : a = -3 := h₁
  have h₆ : b = -1 := h₂
  have h₇ : c = 0 := h₃
  have h₈ : d = 1 := h₄
  sorry

end smallest_rational_number_l235_235349


namespace problem1_solution_problem2_solution_l235_235359

theorem problem1_solution (x : ℝ): 2 * x^2 + x - 3 = 0 → (x = 1 ∨ x = -3 / 2) :=
by
  intro h
  -- Proof skipped
  sorry

theorem problem2_solution (x : ℝ): (x - 3)^2 = 2 * x * (3 - x) → (x = 3 ∨ x = 1) :=
by
  intro h
  -- Proof skipped
  sorry

end problem1_solution_problem2_solution_l235_235359


namespace miles_to_drive_l235_235338

def total_miles : ℕ := 1200
def miles_driven : ℕ := 768
def miles_remaining : ℕ := total_miles - miles_driven

theorem miles_to_drive : miles_remaining = 432 := by
  -- Proof goes here, omitted as per instructions
  sorry

end miles_to_drive_l235_235338


namespace contrapositive_l235_235999

variable (k : ℝ)

theorem contrapositive (h : ¬∃ x : ℝ, x^2 - x - k = 0) : k ≤ 0 :=
sorry

end contrapositive_l235_235999


namespace age_problem_l235_235640

-- Definitions from conditions
variables (p q : ℕ) -- ages of p and q as natural numbers
variables (Y : ℕ) -- number of years ago p was half the age of q

-- Main statement
theorem age_problem :
  (p + q = 28) ∧ (p / q = 3 / 4) ∧ (p - Y = (q - Y) / 2) → Y = 8 :=
by
  sorry

end age_problem_l235_235640


namespace CandyGivenToJanetEmily_l235_235667

noncomputable def initial_candy : ℝ := 78.5
noncomputable def candy_left_after_janet : ℝ := 68.75
noncomputable def candy_given_to_emily : ℝ := 2.25

theorem CandyGivenToJanetEmily :
  initial_candy - candy_left_after_janet + candy_given_to_emily = 12 := 
by
  sorry

end CandyGivenToJanetEmily_l235_235667


namespace range_of_a_l235_235190

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧ 
  (∀ x > 1, f x = a^x) ∧ 
  (∀ x ≤ 1, f x = (4 - (a / 2)) * x + 2) → 
  4 ≤ a ∧ a < 8 :=
by
  sorry

end range_of_a_l235_235190


namespace beaver_hid_90_carrots_l235_235371

-- Defining the number of burrows and carrot condition homomorphic to the problem
def beaver_carrots (x : ℕ) := 5 * x
def rabbit_carrots (y : ℕ) := 7 * y

-- Stating the main theorem based on conditions derived from the problem
theorem beaver_hid_90_carrots (x y : ℕ) (h1 : beaver_carrots x = rabbit_carrots y) (h2 : y = x - 5) : 
  beaver_carrots x = 90 := 
by 
  sorry

end beaver_hid_90_carrots_l235_235371


namespace find_abc_and_sqrt_l235_235045

theorem find_abc_and_sqrt (a b c : ℤ) (h1 : 3 * a - 2 * b - 1 = 9) (h2 : a + 2 * b = -8) (h3 : c = Int.floor (2 + Real.sqrt 7)) :
  a = 2 ∧ b = -2 ∧ c = 4 ∧ (Real.sqrt (a - b + c) = 2 * Real.sqrt 2 ∨ Real.sqrt (a - b + c) = -2 * Real.sqrt 2) :=
by
  -- proof details go here
  sorry

end find_abc_and_sqrt_l235_235045


namespace cube_surface_area_proof_l235_235174

-- Conditions
def prism_volume : ℕ := 10 * 5 * 20
def cube_volume : ℕ := 1000
def edge_length_of_cube : ℕ := 10
def cube_surface_area (s : ℕ) : ℕ := 6 * s * s

-- Theorem Statement
theorem cube_surface_area_proof : cube_volume = prism_volume → cube_surface_area edge_length_of_cube = 600 := 
by
  intros h
  -- Proof goes here
  sorry

end cube_surface_area_proof_l235_235174


namespace flask_forces_l235_235284

theorem flask_forces (r : ℝ) (ρ g h_A h_B h_C V : ℝ) (A : ℝ) (FA FB FC : ℝ) (h1 : r = 2)
  (h2 : A = π * r^2)
  (h3 : V = A * h_A ∧ V = A * h_B ∧ V = A * h_C)
  (h4 : FC = ρ * g * h_C * A)
  (h5 : FA = ρ * g * h_A * A)
  (h6 : FB = ρ * g * h_B * A)
  (h7 : h_C > h_A ∧ h_A > h_B) : FC > FA ∧ FA > FB := 
sorry

end flask_forces_l235_235284


namespace nuts_to_raisins_ratio_l235_235326

/-- 
Given that Chris mixed 3 pounds of raisins with 4 pounds of nuts 
and the total cost of the raisins was 0.15789473684210525 of the total cost of the mixture, 
prove that the ratio of the cost of a pound of nuts to the cost of a pound of raisins is 4:1. 
-/
theorem nuts_to_raisins_ratio (R N : ℝ)
    (h1 : 3 * R = 0.15789473684210525 * (3 * R + 4 * N)) :
    N / R = 4 :=
sorry  -- proof skipped

end nuts_to_raisins_ratio_l235_235326


namespace intersection_lg_1_x_squared_zero_t_le_one_l235_235348

theorem intersection_lg_1_x_squared_zero_t_le_one  :
  let M := {x | 0 ≤ x ∧ x ≤ 2}
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_lg_1_x_squared_zero_t_le_one_l235_235348


namespace probability_all_yellow_l235_235417

-- Definitions and conditions
def total_apples : ℕ := 8
def red_apples : ℕ := 5
def yellow_apples : ℕ := 3
def chosen_apples : ℕ := 3

-- Theorem to prove
theorem probability_all_yellow :
  (yellow_apples.choose chosen_apples : ℚ) / (total_apples.choose chosen_apples) = 1 / 56 := sorry

end probability_all_yellow_l235_235417


namespace gold_bars_lost_l235_235145

-- Define the problem constants
def initial_bars : ℕ := 100
def friends : ℕ := 4
def bars_per_friend : ℕ := 20

-- Define the total distributed gold bars
def total_distributed : ℕ := friends * bars_per_friend

-- Define the number of lost gold bars
def lost_bars : ℕ := initial_bars - total_distributed

-- Theorem: Prove that the number of lost gold bars is 20
theorem gold_bars_lost : lost_bars = 20 := by
  sorry

end gold_bars_lost_l235_235145


namespace find_value_l235_235726

noncomputable def f : ℝ → ℝ := sorry

def tangent_line (x y : ℝ) : Prop := x + 2 * y + 1 = 0

def has_tangent_at (f : ℝ → ℝ) (x0 : ℝ) (L : ℝ → ℝ → Prop) : Prop :=
  L x0 (f x0)

theorem find_value (h : has_tangent_at f 2 tangent_line) :
  f 2 - 2 * (deriv f 2) = -1/2 :=
sorry

end find_value_l235_235726


namespace star_value_example_l235_235301

def star (a b c : ℤ) : ℤ := (a + b + c) ^ 2

theorem star_value_example : star 3 (-5) 2 = 0 :=
by
  sorry

end star_value_example_l235_235301


namespace total_points_needed_l235_235888

def num_students : ℕ := 25
def num_weeks : ℕ := 2
def vegetables_per_student_per_week : ℕ := 2
def points_per_vegetable : ℕ := 2

theorem total_points_needed : 
  (num_students * (vegetables_per_student_per_week * num_weeks) * points_per_vegetable) = 200 := by
  sorry

end total_points_needed_l235_235888


namespace product_xyz_l235_235191

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l235_235191


namespace consecutive_even_product_6digit_l235_235062

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end consecutive_even_product_6digit_l235_235062


namespace misha_second_round_score_l235_235217

def misha_score_first_round (darts : ℕ) (score_per_dart_min : ℕ) : ℕ := 
  darts * score_per_dart_min

def misha_score_second_round (score_first : ℕ) (multiplier : ℕ) : ℕ := 
  score_first * multiplier

def misha_score_third_round (score_second : ℕ) (multiplier : ℚ) : ℚ := 
  score_second * multiplier

theorem misha_second_round_score (darts : ℕ) (score_per_dart_min : ℕ) (multiplier_second : ℕ) (multiplier_third : ℚ) 
  (h_darts : darts = 8) (h_score_per_dart_min : score_per_dart_min = 3) (h_multiplier_second : multiplier_second = 2) (h_multiplier_third : multiplier_third = 1.5) :
  misha_score_second_round (misha_score_first_round darts score_per_dart_min) multiplier_second = 48 :=
by sorry

end misha_second_round_score_l235_235217


namespace shared_candy_equally_l235_235032

def Hugh_candy : ℕ := 8
def Tommy_candy : ℕ := 6
def Melany_candy : ℕ := 7
def total_people : ℕ := 3

theorem shared_candy_equally : 
  (Hugh_candy + Tommy_candy + Melany_candy) / total_people = 7 := 
by 
  sorry

end shared_candy_equally_l235_235032


namespace proof_problems_l235_235915

def otimes (a b : ℝ) : ℝ :=
  a * (1 - b)

theorem proof_problems :
  (otimes 2 (-2) = 6) ∧
  ¬ (∀ (a b : ℝ), otimes a b = otimes b a) ∧
  (∀ (a b : ℝ), a + b = 0 → otimes a a + otimes b b = 2 * a * b) ∧
  ¬ (∀ (a b : ℝ), otimes a b = 0 → a = 0) :=
by
  sorry
 
end proof_problems_l235_235915


namespace product_of_possible_b_values_l235_235596

theorem product_of_possible_b_values : 
  ∀ b : ℝ, 
    (abs (b - 2) = 2 * (4 - 1)) → 
    (b = 8 ∨ b = -4) → 
    (8 * (-4) = -32) := by
  sorry

end product_of_possible_b_values_l235_235596


namespace mrs_jackson_boxes_l235_235666

theorem mrs_jackson_boxes (decorations_per_box used_decorations given_decorations : ℤ) 
(h1 : decorations_per_box = 15)
(h2 : used_decorations = 35)
(h3 : given_decorations = 25) :
  (used_decorations + given_decorations) / decorations_per_box = 4 := 
by sorry

end mrs_jackson_boxes_l235_235666


namespace find_x_l235_235252

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : x * floor x = 50) : x = 7.142857 :=
by
  sorry

end find_x_l235_235252


namespace least_perimeter_of_triangle_l235_235402

-- Define the sides of the triangle
def side1 : ℕ := 40
def side2 : ℕ := 48

-- Given condition for the third side
def valid_third_side (x : ℕ) : Prop :=
  8 < x ∧ x < 88

-- The least possible perimeter given the conditions
def least_possible_perimeter : ℕ :=
  side1 + side2 + 9

theorem least_perimeter_of_triangle (x : ℕ) (h : valid_third_side x) (hx : x = 9) : least_possible_perimeter = 97 :=
by
  rw [least_possible_perimeter]
  exact rfl

end least_perimeter_of_triangle_l235_235402


namespace trigonometric_identity_l235_235206

theorem trigonometric_identity : 
  (Real.cos (15 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) - Real.cos (75 * Real.pi / 180) * Real.sin (105 * Real.pi / 180))
  = -1 / 2 :=
by
  sorry

end trigonometric_identity_l235_235206


namespace solution_set_inequality_l235_235769

theorem solution_set_inequality {a b : ℝ} 
  (h₁ : {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | ax^2 - bx + 2 < 0}) : a + b = -2 :=
by
  sorry

end solution_set_inequality_l235_235769


namespace difference_between_sums_l235_235346

-- Define the arithmetic sequence sums
def sum_seq (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Define sets A and B
def sumA : ℕ := sum_seq 10 75
def sumB : ℕ := sum_seq 76 125

-- State the problem
theorem difference_between_sums : sumB - sumA = 2220 :=
by
  -- The proof is omitted
  sorry

end difference_between_sums_l235_235346


namespace question_inequality_l235_235702

theorem question_inequality (x y z : ℝ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x ≥ max (3/4 * (x - y)^2) (max (3/4 * (y - z)^2) (3/4 * (z - x)^2)) := 
sorry

end question_inequality_l235_235702


namespace f_value_at_3_l235_235113

def f (x : ℝ) := 2 * (x + 1) + 1

theorem f_value_at_3 : f 3 = 9 :=
by sorry

end f_value_at_3_l235_235113


namespace telephone_call_duration_l235_235441

theorem telephone_call_duration (x : ℝ) :
  (0.60 + 0.06 * (x - 4) = 0.08 * x) → x = 18 :=
by
  sorry

end telephone_call_duration_l235_235441


namespace average_weight_increase_l235_235015

theorem average_weight_increase (A : ℝ) (hA : 8 * A + 20 = (80 : ℝ) + (8 * (A - (60 - 80) / 8))) :
  ((8 * A + 20) / 8) - A = (2.5 : ℝ) :=
by
  sorry

end average_weight_increase_l235_235015


namespace min_value_function_l235_235785

theorem min_value_function (x : ℝ) (h : 1 < x) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y ≥ 3) :=
sorry

end min_value_function_l235_235785


namespace mack_writing_time_tuesday_l235_235001

variable (minutes_per_page_mon : ℕ := 30)
variable (time_mon : ℕ := 60)
variable (pages_wed : ℕ := 5)
variable (total_pages : ℕ := 10)
variable (minutes_per_page_tue : ℕ := 15)

theorem mack_writing_time_tuesday :
  (time_mon / minutes_per_page_mon) + pages_wed + (3 * minutes_per_page_tue / minutes_per_page_tue) = total_pages →
  (3 * minutes_per_page_tue) = 45 := by
  intros h
  sorry

end mack_writing_time_tuesday_l235_235001


namespace Maddie_bought_two_white_packs_l235_235091

theorem Maddie_bought_two_white_packs 
  (W : ℕ)
  (total_cost : ℕ)
  (cost_per_shirt : ℕ)
  (white_pack_size : ℕ)
  (blue_pack_size : ℕ)
  (blue_packs : ℕ)
  (cost_per_white_pack : ℕ)
  (cost_per_blue_pack : ℕ) :
  total_cost = 66 ∧ cost_per_shirt = 3 ∧ white_pack_size = 5 ∧ blue_pack_size = 3 ∧ blue_packs = 4 ∧ cost_per_white_pack = white_pack_size * cost_per_shirt ∧ cost_per_blue_pack = blue_pack_size * cost_per_shirt ∧ 3 * (white_pack_size * W + blue_pack_size * blue_packs) = total_cost → W = 2 :=
by
  sorry

end Maddie_bought_two_white_packs_l235_235091


namespace no_partition_of_positive_integers_l235_235330

theorem no_partition_of_positive_integers :
  ∀ (A B C : Set ℕ), (∀ (x : ℕ), x ∈ A ∨ x ∈ B ∨ x ∈ C) →
  (∀ (x y : ℕ), x ∈ A ∧ y ∈ B → x^2 - x * y + y^2 ∈ C) →
  (∀ (x y : ℕ), x ∈ B ∧ y ∈ C → x^2 - x * y + y^2 ∈ A) →
  (∀ (x y : ℕ), x ∈ C ∧ y ∈ A → x^2 - x * y + y^2 ∈ B) →
  False := 
sorry

end no_partition_of_positive_integers_l235_235330


namespace difference_of_bases_l235_235788

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * (8^5) + 4 * (8^4) + 3 * (8^3) + 2 * (8^2) + 1 * (8^1) + 0 * (8^0)

def base5_to_base10 (n : ℕ) : ℕ :=
  4 * (5^4) + 3 * (5^3) + 2 * (5^2) + 1 * (5^1) + 0 * (5^0)

theorem difference_of_bases : 
  base8_to_base10 543210 - base5_to_base10 43210 = 177966 :=
by
  sorry

end difference_of_bases_l235_235788


namespace juniors_more_than_seniors_l235_235739

theorem juniors_more_than_seniors
  (j s : ℕ)
  (h1 : (1 / 3) * j = (2 / 3) * s)
  (h2 : j + s = 300) :
  j - s = 100 := 
sorry

end juniors_more_than_seniors_l235_235739


namespace average_speed_including_stoppages_l235_235963

/--
If the average speed of a bus excluding stoppages is 50 km/hr, and
the bus stops for 12 minutes per hour, then the average speed of the
bus including stoppages is 40 km/hr.
-/
theorem average_speed_including_stoppages
  (u : ℝ) (Δt : ℝ) (h₁ : u = 50) (h₂ : Δt = 12) : 
  (u * (60 - Δt) / 60) = 40 :=
by
  sorry

end average_speed_including_stoppages_l235_235963


namespace minimum_value_fraction_l235_235240

theorem minimum_value_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1) :
  (1 / a) + (4 / b) ≥ 9 :=
sorry

end minimum_value_fraction_l235_235240


namespace painting_price_after_5_years_l235_235938

variable (P : ℝ)
-- Conditions on price changes over the years
def year1_price (P : ℝ) := P * 1.30
def year2_price (P : ℝ) := year1_price P * 0.80
def year3_price (P : ℝ) := year2_price P * 1.25
def year4_price (P : ℝ) := year3_price P * 0.90
def year5_price (P : ℝ) := year4_price P * 1.15

theorem painting_price_after_5_years (P : ℝ) :
  year5_price P = 1.3455 * P := by
  sorry

end painting_price_after_5_years_l235_235938


namespace max_ab_min_reciprocal_sum_l235_235442

noncomputable section

-- Definitions for conditions
def is_positive_real (x : ℝ) : Prop := x > 0

def condition (a b : ℝ) : Prop := is_positive_real a ∧ is_positive_real b ∧ (a + 10 * b = 1)

-- Maximum value of ab
theorem max_ab (a b : ℝ) (h : condition a b) : a * b ≤ 1 / 40 :=
sorry

-- Minimum value of 1/a + 1/b
theorem min_reciprocal_sum (a b : ℝ) (h : condition a b) : 1 / a + 1 / b ≥ 11 + 2 * Real.sqrt 10 :=
sorry

end max_ab_min_reciprocal_sum_l235_235442


namespace cos_sum_zero_l235_235694

noncomputable def cos_sum : ℂ :=
  Real.cos (Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (7 * Real.pi / 15) + Real.cos (10 * Real.pi / 15)

theorem cos_sum_zero : cos_sum = 0 := by
  sorry

end cos_sum_zero_l235_235694


namespace trains_meet_1050_km_from_delhi_l235_235931

def distance_train_meet (t1_departure t2_departure : ℕ) (s1 s2 : ℕ) : ℕ :=
  let t_gap := t2_departure - t1_departure      -- Time difference between the departures in hours
  let d1 := s1 * t_gap                          -- Distance covered by the first train until the second train starts
  let relative_speed := s2 - s1                 -- Relative speed of the second train with respect to the first train
  d1 + s2 * (d1 / relative_speed)               -- Distance from Delhi where they meet

theorem trains_meet_1050_km_from_delhi :
  distance_train_meet 9 14 30 35 = 1050 := by
  -- Definitions based on the problem's conditions
  let t1 := 9          -- First train departs at 9 a.m.
  let t2 := 14         -- Second train departs at 2 p.m. (14:00 in 24-hour format)
  let s1 := 30         -- Speed of the first train in km/h
  let s2 := 35         -- Speed of the second train in km/h
  sorry -- proof to be filled in

end trains_meet_1050_km_from_delhi_l235_235931


namespace investment_rate_l235_235223

theorem investment_rate (total_investment : ℝ) (invest1 : ℝ) (rate1 : ℝ) (invest2 : ℝ) (rate2 : ℝ) (desired_income : ℝ) (remaining_investment : ℝ) (remaining_rate : ℝ) : 
( total_investment = 12000 ∧ invest1 = 5000 ∧ rate1 = 0.06 ∧ invest2 = 4000 ∧ rate2 = 0.035 ∧ desired_income = 700 ∧ remaining_investment = 3000 ) → remaining_rate = 0.0867 :=
by
  sorry

end investment_rate_l235_235223


namespace daily_rental_cost_l235_235085

theorem daily_rental_cost (rental_fee_per_day : ℝ) (mileage_rate : ℝ) (budget : ℝ) (max_miles : ℝ) 
  (h1 : mileage_rate = 0.20) 
  (h2 : budget = 88.0) 
  (h3 : max_miles = 190.0) :
  rental_fee_per_day = 50.0 := 
by
  sorry

end daily_rental_cost_l235_235085


namespace hyperbola_eccentricity_l235_235110

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end hyperbola_eccentricity_l235_235110


namespace vertical_shirts_count_l235_235237

-- Definitions from conditions
def total_people : ℕ := 40
def checkered_shirts : ℕ := 7
def horizontal_shirts := 4 * checkered_shirts

-- Proof goal
theorem vertical_shirts_count :
  ∃ vertical_shirts : ℕ, vertical_shirts = total_people - (checkered_shirts + horizontal_shirts) ∧ vertical_shirts = 5 :=
sorry

end vertical_shirts_count_l235_235237


namespace find_k_l235_235455

theorem find_k (k : ℕ) :
  (∑' n : ℕ, (5 + n * k) / 5 ^ n) = 12 → k = 90 :=
by
  sorry

end find_k_l235_235455


namespace smallest_n_for_square_and_cube_l235_235202

theorem smallest_n_for_square_and_cube (n : ℕ) 
  (h1 : ∃ m : ℕ, 3 * n = m^2) 
  (h2 : ∃ k : ℕ, 5 * n = k^3) : 
  n = 675 :=
  sorry

end smallest_n_for_square_and_cube_l235_235202


namespace triangle_side_length_mod_l235_235166

theorem triangle_side_length_mod {a d x : ℕ} 
  (h_equilateral : ∃ (a : ℕ), 3 * a = 1 + d + x)
  (h_triangle : ∀ {a d x : ℕ}, 1 + d > x ∧ 1 + x > d ∧ d + x > 1)
  : d % 3 = 1 :=
by
  sorry

end triangle_side_length_mod_l235_235166


namespace rate_of_interest_l235_235074

theorem rate_of_interest (SI P T R : ℝ) 
  (hSI : SI = 4016.25) 
  (hP : P = 6693.75) 
  (hT : T = 5) 
  (h : SI = (P * R * T) / 100) : 
  R = 12 :=
by 
  sorry

end rate_of_interest_l235_235074


namespace distance_to_x_axis_l235_235342

theorem distance_to_x_axis (x y : ℝ) (h : (x, y) = (3, -4)) : abs y = 4 := sorry

end distance_to_x_axis_l235_235342


namespace value_of_expression_l235_235984

open Polynomial

theorem value_of_expression (a b : ℚ) (h1 : (3 : ℚ) * a ^ 2 + 9 * a - 21 = 0) (h2 : (3 : ℚ) * b ^ 2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (2 * b - 2) = -4 :=
by sorry

end value_of_expression_l235_235984


namespace calculate_g_inv_l235_235599

noncomputable def g : ℤ → ℤ := sorry
noncomputable def g_inv : ℤ → ℤ := sorry

axiom g_inv_eq : ∀ x, g (g_inv x) = x

axiom cond1 : g (-1) = 2
axiom cond2 : g (0) = 3
axiom cond3 : g (1) = 6

theorem calculate_g_inv : 
  g_inv (g_inv 6 - g_inv 2) = -1 := 
by
  -- The proof goes here
  sorry

end calculate_g_inv_l235_235599


namespace min_n_constant_term_l235_235724

theorem min_n_constant_term (x : ℕ) (hx : x > 0) : 
  ∃ n : ℕ, 
  (∀ r : ℕ, (2 * n = 5 * r) → n ≥ 5) ∧ 
  (∃ r : ℕ, (2 * n = 5 * r) ∧ n = 5) := by
  sorry

end min_n_constant_term_l235_235724


namespace volume_of_sphere_eq_4_sqrt3_pi_l235_235618

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r ^ 3

theorem volume_of_sphere_eq_4_sqrt3_pi
  (r : ℝ) (h : 4 * Real.pi * r ^ 2 = 2 * Real.sqrt 3 * Real.pi * (2 * r)) :
  volume_of_sphere r = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end volume_of_sphere_eq_4_sqrt3_pi_l235_235618


namespace max_integer_in_form_3_x_3_sub_x_l235_235950

theorem max_integer_in_form_3_x_3_sub_x :
  ∃ x : ℝ, ∀ y : ℝ, y = 3^(x * (3 - x)) → ⌊y⌋ ≤ 11 := 
sorry

end max_integer_in_form_3_x_3_sub_x_l235_235950


namespace nonneg_int_values_of_fraction_condition_l235_235595

theorem nonneg_int_values_of_fraction_condition (n : ℕ) : (∃ k : ℤ, 30 * n + 2 = k * (12 * n + 1)) → n = 0 := by
  sorry

end nonneg_int_values_of_fraction_condition_l235_235595


namespace students_in_canteen_l235_235228

-- Definitions for conditions
def total_students : ℕ := 40
def absent_fraction : ℚ := 1 / 10
def classroom_fraction : ℚ := 3 / 4

-- Lean 4 statement
theorem students_in_canteen :
  let absent_students := (absent_fraction * total_students)
  let present_students := (total_students - absent_students)
  let classroom_students := (classroom_fraction * present_students)
  let canteen_students := (present_students - classroom_students)
  canteen_students = 9 := by
    sorry

end students_in_canteen_l235_235228


namespace housewife_spend_money_l235_235505

theorem housewife_spend_money (P M: ℝ) (h1: 0.75 * P = 30) (h2: M / (0.75 * P) - M / P = 5) : 
  M = 600 :=
by
  sorry

end housewife_spend_money_l235_235505


namespace min_value_3x_4y_l235_235412

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / x + 1 / y = 1) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end min_value_3x_4y_l235_235412


namespace inequality_one_solution_inequality_two_solution_l235_235798

-- The statement for the first inequality
theorem inequality_one_solution (x : ℝ) :
  |1 - ((2 * x - 1) / 3)| ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5 := sorry

-- The statement for the second inequality
theorem inequality_two_solution (x : ℝ) :
  (2 - x) * (x + 3) < 2 - x ↔ x < -2 ∨ x > 2 := sorry

end inequality_one_solution_inequality_two_solution_l235_235798


namespace problem_statement_l235_235772

-- Define the function
def f (x : ℝ) := -2 * x^2

-- We need to show that f is monotonically decreasing and even on (0, +∞)
theorem problem_statement : (∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x) ∧ (∀ x : ℝ, f (-x) = f x) := 
by {
  sorry -- proof goes here
}

end problem_statement_l235_235772


namespace max_a3_b3_c3_d3_l235_235657

-- Define that a, b, c, d are real numbers that satisfy the given conditions.
theorem max_a3_b3_c3_d3 (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 16)
  (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) :
  a^3 + b^3 + c^3 + d^3 ≤ 64 :=
sorry

end max_a3_b3_c3_d3_l235_235657


namespace simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l235_235898

variable (a b : ℤ)
def A : ℤ := b^2 - a^2 + 5 * a * b
def B : ℤ := 3 * a * b + 2 * b^2 - a^2

theorem simplify_2A_minus_B : 2 * A a b - B a b = -a^2 + 7 * a * b := by
  sorry

theorem evaluate_2A_minus_B_at_1_2 : 2 * A 1 2 - B 1 2 = 13 := by
  sorry

end simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l235_235898


namespace range_of_a_l235_235876

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- State the theorem that describes the condition and proves the answer
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 4 → x₂ < 4 → f a x₁ ≥ f a x₂) → a ≤ -3 :=
by
  -- The proof would go here; for now, we skip it
  sorry

end range_of_a_l235_235876


namespace center_of_circle_point_not_on_circle_l235_235873

-- Definitions and conditions
def circle_eq (x y : ℝ) := x^2 - 6 * x + y^2 + 2 * y - 11 = 0

-- The problem statement split into two separate theorems

-- Proving the center of the circle is (3, -1)
theorem center_of_circle : 
  ∃ h k : ℝ, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 21) ∧ (h, k) = (3, -1) := sorry

-- Proving the point (5, -1) does not lie on the circle
theorem point_not_on_circle : ¬ circle_eq 5 (-1) := sorry

end center_of_circle_point_not_on_circle_l235_235873


namespace no_integer_solutions_l235_235372

theorem no_integer_solutions (x y : ℤ) : x^3 + 3 ≠ 4 * y * (y + 1) :=
sorry

end no_integer_solutions_l235_235372


namespace xyz_value_l235_235665

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 20 / 3 :=
by
  sorry

end xyz_value_l235_235665


namespace find_a_b_l235_235653

theorem find_a_b (a b x y : ℝ) (h1 : x = 2) (h2 : y = 4) (h3 : a * x + b * y = 16) (h4 : b * x - a * y = -12) : a = 4 ∧ b = 2 := by
  sorry

end find_a_b_l235_235653


namespace find_percentage_l235_235141

theorem find_percentage (P : ℝ) (h : P / 100 * 3200 = 0.20 * 650 + 190) : P = 10 :=
by 
  sorry

end find_percentage_l235_235141


namespace proportional_function_ratio_l235_235297

-- Let k be a constant, and y = kx be a proportional function.
-- We know that f(1) = 3 and f(a) = b where b ≠ 0.
-- We want to prove that a / b = 1 / 3.

theorem proportional_function_ratio (a b k : ℝ) :
  (∀ x, x = 1 → k * x = 3) →
  (∀ x, x = a → k * x = b) →
  b ≠ 0 →
  a / b = 1 / 3 :=
by
  intros h1 h2 h3
  -- the proof will follow but is not required here
  sorry

end proportional_function_ratio_l235_235297


namespace problem1_problem2_l235_235605

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

end problem1_problem2_l235_235605


namespace ceil_floor_eq_zero_implies_sum_l235_235281

theorem ceil_floor_eq_zero_implies_sum (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ + ⌊x⌋ = 2 * x :=
by
  sorry

end ceil_floor_eq_zero_implies_sum_l235_235281


namespace jenny_spent_fraction_l235_235593

theorem jenny_spent_fraction
  (x : ℝ) -- The original amount of money Jenny had
  (h_half_x : 1/2 * x = 21) -- Half of the original amount is $21
  (h_left_money : x - 24 = 24) -- Jenny had $24 left after spending
  : (x - 24) / x = 3 / 7 := sorry

end jenny_spent_fraction_l235_235593


namespace minimum_value_of_function_l235_235990

theorem minimum_value_of_function :
  ∀ x : ℝ, (x > -2) → (x + (16 / (x + 2)) ≥ 6) :=
by
  intro x hx
  sorry

end minimum_value_of_function_l235_235990


namespace max_real_roots_among_polynomials_l235_235956

noncomputable def largest_total_real_roots (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : ℕ :=
  4  -- representing the largest total number of real roots

theorem max_real_roots_among_polynomials
  (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  largest_total_real_roots a b c h_a h_b h_c = 4 :=
sorry

end max_real_roots_among_polynomials_l235_235956


namespace unique_real_solution_l235_235321

-- Define the variables
variables (x y : ℝ)

-- State the condition
def equation (x y : ℝ) : Prop :=
  (2^(4*x + 2)) * (4^(2*x + 3)) = (8^(3*x + 4)) * y

-- State the theorem
theorem unique_real_solution (y : ℝ) (h_y : 0 < y) : ∃! x : ℝ, equation x y :=
sorry

end unique_real_solution_l235_235321


namespace two_digit_sequence_partition_property_l235_235600

theorem two_digit_sequence_partition_property :
  ∀ (A B : Set ℕ), (A ∪ B = {x | x < 100 ∧ x % 10 < 10}) →
  ∃ (C : Set ℕ), (C = A ∨ C = B) ∧ 
  ∃ (lst : List ℕ), (∀ (x : ℕ), x ∈ lst → x ∈ C) ∧ 
  (∀ (x y : ℕ), (x, y) ∈ lst.zip lst.tail → (y = x + 1 ∨ y = x + 10 ∨ y = x + 11)) :=
by
  intros A B partition_condition
  sorry

end two_digit_sequence_partition_property_l235_235600


namespace max_value_of_b_minus_a_l235_235120

theorem max_value_of_b_minus_a (a b : ℝ) (h₀ : a < 0)
  (h₁ : ∀ x : ℝ, a < x ∧ x < b → (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) :
  b - a ≤ 2017 :=
sorry

end max_value_of_b_minus_a_l235_235120


namespace not_possible_arrangement_l235_235355

theorem not_possible_arrangement : 
  ¬ ∃ (f : Fin 4026 → Fin 2014), 
    (∀ k : Fin 2014, ∃ i j : Fin 4026, i < j ∧ f i = k ∧ f j = k ∧ (j.val - i.val - 1) = k.val) :=
sorry

end not_possible_arrangement_l235_235355


namespace necessary_but_not_sufficient_for_parallel_lines_l235_235753

theorem necessary_but_not_sufficient_for_parallel_lines (m : ℝ) : 
  (m = -1/2 ∨ m = 0) ↔ (∀ x y : ℝ, (x + 2*m*y - 1 = 0 ∧ (3*m + 1)*x - m*y - 1 = 0) → false) :=
sorry

end necessary_but_not_sufficient_for_parallel_lines_l235_235753


namespace exists_composite_expression_l235_235948

-- Define what it means for a number to be composite
def is_composite (m : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = m

-- Main theorem statement
theorem exists_composite_expression :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → is_composite (n * 2^k + 1) :=
sorry

end exists_composite_expression_l235_235948


namespace quadratic_solution_l235_235092

theorem quadratic_solution (x : ℝ) : 
  x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by 
  sorry

end quadratic_solution_l235_235092


namespace fish_in_aquarium_l235_235094

theorem fish_in_aquarium (initial_fish : ℕ) (added_fish : ℕ) (h1 : initial_fish = 10) (h2 : added_fish = 3) : initial_fish + added_fish = 13 := by
  sorry

end fish_in_aquarium_l235_235094


namespace abs_neg_three_eq_three_l235_235211

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
sorry

end abs_neg_three_eq_three_l235_235211


namespace behavior_on_1_2_l235_235106

/-- Definition of an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

/-- Definition of being decreasing on an interval -/
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

/-- Definition of having a minimum value on an interval -/
def has_minimum_on (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → f x ≥ m

theorem behavior_on_1_2 
  {f : ℝ → ℝ} 
  (h_odd : is_odd_function f) 
  (h_dec : is_decreasing_on f (-2) (-1)) 
  (h_min : has_minimum_on f (-2) (-1) 3) :
  is_decreasing_on f 1 2 ∧ ∀ x, 1 ≤ x → x ≤ 2 → f x ≤ -3 := 
by 
  sorry

end behavior_on_1_2_l235_235106


namespace find_four_digit_numbers_l235_235558

noncomputable def four_digit_number_permutations_sum (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) : Prop :=
  6 * (x + y + z + t) * (1000 + 100 + 10 + 1) = 10 * (1111 * x)

theorem find_four_digit_numbers (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) :
  four_digit_number_permutations_sum x y z t distinct nonzero :=
  sorry

end find_four_digit_numbers_l235_235558


namespace total_students_l235_235754

-- Definitions based on problem conditions
def H := 36
def S := 32
def union_H_S := 59
def history_not_statistics := 27

-- The proof statement
theorem total_students : H + S - (H - history_not_statistics) = union_H_S :=
by sorry

end total_students_l235_235754


namespace set_inter_complement_l235_235816

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 3}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem set_inter_complement :
  B ∩ (U \ A) = {2} :=
by
  sorry

end set_inter_complement_l235_235816


namespace batsman_average_is_18_l235_235291
noncomputable def average_after_18_innings (score_18th: ℕ) (average_17th: ℕ) (innings: ℕ) : ℕ :=
  let total_runs_17 := average_17th * 17
  let total_runs_18 := total_runs_17 + score_18th
  total_runs_18 / innings

theorem batsman_average_is_18 {score_18th: ℕ} {average_17th: ℕ} {expected_average: ℕ} :
  score_18th = 1 → average_17th = 19 → expected_average = 18 →
  average_after_18_innings score_18th average_17th 18 = expected_average := by
  sorry

end batsman_average_is_18_l235_235291


namespace point_of_tangency_l235_235592

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem point_of_tangency (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) 
  (h_slope : ∃ x : ℝ, Real.exp x - 1 / Real.exp x = 3 / 2) :
  ∃ x : ℝ, x = Real.log 2 :=
by
  sorry

end point_of_tangency_l235_235592


namespace max_min_f_in_rectangle_l235_235130

def f (x y : ℝ) : ℝ := x^3 + y^3 + 6 * x * y

def in_rectangle (x y : ℝ) : Prop := 
  (-3 ≤ x ∧ x ≤ 1) ∧ (-3 ≤ y ∧ y ≤ 2)

theorem max_min_f_in_rectangle :
  ∃ (x_max y_max x_min y_min : ℝ),
    in_rectangle x_max y_max ∧ in_rectangle x_min y_min ∧
    (∀ x y, in_rectangle x y → f x y ≤ f x_max y_max) ∧
    (∀ x y, in_rectangle x y → f x_min y_min ≤ f x y) ∧
    f x_max y_max = 21 ∧ f x_min y_min = -55 :=
by
  sorry

end max_min_f_in_rectangle_l235_235130


namespace sum_of_solutions_of_fx_eq_0_l235_235638

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 7 * x + 10 else 3 * x - 15

theorem sum_of_solutions_of_fx_eq_0 :
  let x1 := -10 / 7
  let x2 := 5
  f x1 = 0 ∧ f x2 = 0 ∧ x1 ≤ 1 ∧ x2 > 1 → x1 + x2 = 25 / 7 :=
by
  sorry

end sum_of_solutions_of_fx_eq_0_l235_235638


namespace range_of_k_l235_235786

theorem range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) :=
by
  sorry

end range_of_k_l235_235786


namespace range_of_x_l235_235974

theorem range_of_x {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x)) 
  (h_mono_dec : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
  (h_f2 : f 2 = 0)
  (h_pos : ∀ x, f (x - 1) > 0) : 
  ∀ x, -1 < x ∧ x < 3 ↔ f (x - 1) > 0 :=
sorry

end range_of_x_l235_235974


namespace overall_profit_percentage_l235_235773

theorem overall_profit_percentage :
  let SP_A := 900
  let SP_B := 1200
  let SP_C := 1500
  let P_A := 300
  let P_B := 400
  let P_C := 500
  let CP_A := SP_A - P_A
  let CP_B := SP_B - P_B
  let CP_C := SP_C - P_C
  let TCP := CP_A + CP_B + CP_C
  let TSP := SP_A + SP_B + SP_C
  let TP := TSP - TCP
  let ProfitPercentage := (TP / TCP) * 100
  ProfitPercentage = 50 := by
  sorry

end overall_profit_percentage_l235_235773


namespace find_full_haired_dogs_l235_235538

-- Definitions of the given conditions
def minutes_per_short_haired_dog : Nat := 10
def short_haired_dogs : Nat := 6
def total_time_minutes : Nat := 4 * 60
def twice_as_long (n : Nat) : Nat := 2 * n

-- Define the problem
def full_haired_dogs : Nat :=
  let short_haired_total_time := short_haired_dogs * minutes_per_short_haired_dog
  let remaining_time := total_time_minutes - short_haired_total_time
  remaining_time / (twice_as_long minutes_per_short_haired_dog)

-- Theorem statement
theorem find_full_haired_dogs : 
  full_haired_dogs = 9 :=
by
  sorry

end find_full_haired_dogs_l235_235538


namespace quadratic_inequality_no_real_roots_l235_235709

theorem quadratic_inequality_no_real_roots (a b c : ℝ) (h : a ≠ 0) (h_Δ : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) :=
sorry

end quadratic_inequality_no_real_roots_l235_235709


namespace sum_of_numbers_l235_235382

variable (x y : ℝ)

def condition1 := 0.45 * x = 2700
def condition2 := y = 2 * x

theorem sum_of_numbers (h1 : condition1 x) (h2 : condition2 x y) : x + y = 18000 :=
by {
  sorry
}

end sum_of_numbers_l235_235382


namespace volume_to_surface_area_ratio_l235_235360

-- Definitions based on the conditions
def unit_cube_volume : ℕ := 1
def num_unit_cubes : ℕ := 7
def unit_cube_total_volume : ℕ := num_unit_cubes * unit_cube_volume

def surface_area_of_central_cube : ℕ := 0
def exposed_faces_per_surrounding_cube : ℕ := 5
def num_surrounding_cubes : ℕ := 6
def total_surface_area : ℕ := num_surrounding_cubes * exposed_faces_per_surrounding_cube

-- Mathematical proof statement
theorem volume_to_surface_area_ratio : 
  (unit_cube_total_volume : ℚ) / (total_surface_area : ℚ) = 7 / 30 :=
by sorry

end volume_to_surface_area_ratio_l235_235360


namespace sqrt_solution_range_l235_235690

theorem sqrt_solution_range : 
  7 < (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) ∧ (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) < 8 := 
by
  sorry

end sqrt_solution_range_l235_235690


namespace find_xy_l235_235916

theorem find_xy :
  ∃ (x y : ℝ), (x - 14)^2 + (y - 15)^2 + (x - y)^2 = 1/3 ∧ x = 14 + 1/3 ∧ y = 14 + 2/3 :=
by
  sorry

end find_xy_l235_235916


namespace weighted_average_fish_caught_l235_235684

-- Define the daily catches for each person
def AangCatches := [5, 7, 9]
def SokkaCatches := [8, 5, 6]
def TophCatches := [10, 12, 8]
def ZukoCatches := [6, 7, 10]

-- Define the group catches
def GroupCatches := AangCatches ++ SokkaCatches ++ TophCatches ++ ZukoCatches

-- Calculate the total number of fish caught by the group
def TotalFishCaught := List.sum GroupCatches

-- Calculate the total number of days fished by the group
def TotalDaysFished := 4 * 3

-- Calculate the weighted average
def WeightedAverage := TotalFishCaught.toFloat / TotalDaysFished.toFloat

-- Proof statement
theorem weighted_average_fish_caught :
  WeightedAverage = 7.75 := by
  sorry

end weighted_average_fish_caught_l235_235684


namespace proof_problem_l235_235196

def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem proof_problem (x : ℝ) :
  necessary_but_not_sufficient ((x+3)*(x-1) = 0) (x-1 = 0) :=
by
  sorry

end proof_problem_l235_235196


namespace sum_in_base4_l235_235677

def dec_to_base4 (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let rec convert (n : ℕ) (acc : ℕ) (power : ℕ) :=
    if n = 0 then acc
    else convert (n / 4) (acc + (n % 4) * power) (power * 10)
  convert n 0 1

theorem sum_in_base4 : dec_to_base4 (234 + 78) = 13020 :=
  sorry

end sum_in_base4_l235_235677


namespace complete_the_square_3x2_9x_20_l235_235229

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end complete_the_square_3x2_9x_20_l235_235229


namespace sum_remainder_zero_l235_235652

theorem sum_remainder_zero
  (a b c : ℕ)
  (h₁ : a % 53 = 31)
  (h₂ : b % 53 = 15)
  (h₃ : c % 53 = 7) :
  (a + b + c) % 53 = 0 :=
by
  sorry

end sum_remainder_zero_l235_235652


namespace find_rate_percent_l235_235140

theorem find_rate_percent (SI P T : ℝ) (h1 : SI = 160) (h2 : P = 800) (h3 : T = 5) : P * (4:ℝ) * T / 100 = SI :=
by
  sorry

end find_rate_percent_l235_235140


namespace isosceles_triangle_base_l235_235445

noncomputable def base_of_isosceles_triangle
  (height_to_base : ℝ)
  (height_to_side : ℝ)
  (is_isosceles : Bool) : ℝ :=
if is_isosceles then 7.5 else 0

theorem isosceles_triangle_base :
  base_of_isosceles_triangle 5 6 true = 7.5 :=
by
  -- The proof would go here, just placeholder for now
  sorry

end isosceles_triangle_base_l235_235445


namespace smallest_number_of_packs_l235_235803

theorem smallest_number_of_packs (n b w : ℕ) (Hn : n = 13) (Hb : b = 8) (Hw : w = 17) :
  Nat.lcm (Nat.lcm n b) w = 1768 :=
by
  sorry

end smallest_number_of_packs_l235_235803


namespace orchids_cut_l235_235488

-- defining the initial conditions
def initial_orchids : ℕ := 3
def final_orchids : ℕ := 7

-- the question: prove the number of orchids cut
theorem orchids_cut : final_orchids - initial_orchids = 4 := by
  sorry

end orchids_cut_l235_235488


namespace find_sum_of_cubes_l235_235802

noncomputable def roots (a b c : ℝ) : Prop :=
  5 * a^3 + 2014 * a + 4027 = 0 ∧ 
  5 * b^3 + 2014 * b + 4027 = 0 ∧ 
  5 * c^3 + 2014 * c + 4027 = 0

theorem find_sum_of_cubes (a b c : ℝ) (h : roots a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 2416.2 :=
sorry

end find_sum_of_cubes_l235_235802


namespace Janice_earnings_l235_235885

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end Janice_earnings_l235_235885


namespace time_to_finish_work_l235_235741

theorem time_to_finish_work (a b c : ℕ) (h1 : 1/a + 1/9 + 1/18 = 1/4) : a = 12 :=
by
  sorry

end time_to_finish_work_l235_235741


namespace gcd_expression_l235_235203

theorem gcd_expression (n : ℕ) (h : n > 2) : Nat.gcd (n^5 - 5 * n^3 + 4 * n) 120 = 120 :=
by
  sorry

end gcd_expression_l235_235203


namespace compute_Z_value_l235_235527

def operation_Z (c d : ℕ) : ℤ := c^2 - 3 * c * d + d^2

theorem compute_Z_value : operation_Z 4 3 = -11 := by
  sorry

end compute_Z_value_l235_235527


namespace min_value_of_abs_diff_l235_235311
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem min_value_of_abs_diff (x1 x2 x : ℝ) (h1 : f x1 ≤ f x) (h2: f x ≤ f x2) : |x1 - x2| = π := by
  sorry

end min_value_of_abs_diff_l235_235311


namespace coefficient_of_squared_term_l235_235774

theorem coefficient_of_squared_term (a b c : ℝ) (h_eq : 5 * a^2 + 14 * b + 5 = 0) :
  a = 5 :=
sorry

end coefficient_of_squared_term_l235_235774


namespace magic_8_ball_probability_l235_235561

theorem magic_8_ball_probability :
  let p_pos := 1 / 3
  let p_neg := 2 / 3
  let n := 6
  let k := 3
  (Nat.choose n k * (p_pos ^ k) * (p_neg ^ (n - k)) = 160 / 729) :=
by
  sorry

end magic_8_ball_probability_l235_235561


namespace holloway_soccer_team_l235_235491

theorem holloway_soccer_team (P M : Finset ℕ) (hP_union_M : (P ∪ M).card = 20) 
(hP : P.card = 12) (h_int : (P ∩ M).card = 6) : M.card = 14 := 
by
  sorry

end holloway_soccer_team_l235_235491


namespace final_price_of_jacket_l235_235154

noncomputable def original_price : ℝ := 240
noncomputable def initial_discount : ℝ := 0.6
noncomputable def additional_discount : ℝ := 0.25

theorem final_price_of_jacket :
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let final_price := price_after_initial_discount * (1 - additional_discount)
  final_price = 72 := 
by
  sorry

end final_price_of_jacket_l235_235154


namespace abc_value_l235_235509

theorem abc_value (a b c : ℝ) (h1 : ab = 30 * (4^(1/3))) (h2 : ac = 40 * (4^(1/3))) (h3 : bc = 24 * (4^(1/3))) :
  a * b * c = 120 :=
sorry

end abc_value_l235_235509


namespace inscribed_square_side_length_l235_235163

-- Define a right triangle
structure RightTriangle :=
  (PQ : ℝ)
  (QR : ℝ)
  (PR : ℝ)
  (is_right : PQ^2 + QR^2 = PR^2)

-- Define the triangle PQR
def trianglePQR : RightTriangle :=
  { PQ := 6, QR := 8, PR := 10, is_right := by norm_num }

-- Define the problem statement
theorem inscribed_square_side_length (t : ℝ) (h : RightTriangle) :
  t = 3 :=
  sorry

end inscribed_square_side_length_l235_235163


namespace rationalize_denominator_l235_235159

theorem rationalize_denominator :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) := by
  sorry

end rationalize_denominator_l235_235159


namespace number_of_distinct_d_l235_235837

noncomputable def calculateDistinctValuesOfD (u v w x : ℂ) (h_distinct : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x) : ℕ := 
by
  sorry

theorem number_of_distinct_d (u v w x : ℂ) (h : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x)
    (h_eqs : ∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
             (z - (d * u)) * (z - (d * v)) * (z - (d * w)) * (z - (d * x))) : 
    calculateDistinctValuesOfD u v w x h = 4 :=
by
  sorry

end number_of_distinct_d_l235_235837


namespace slowest_time_l235_235796

open Real

def time_lola (stories : ℕ) (run_time : ℝ) : ℝ := stories * run_time

def time_sam (stories_run stories_elevator : ℕ) (run_time elevate_time stop_time : ℝ) (wait_time : ℝ) : ℝ :=
  let run_part  := stories_run * run_time
  let wait_part := wait_time
  let elevator_part := stories_elevator * elevate_time + (stories_elevator - 1) * stop_time
  run_part + wait_part + elevator_part

def time_tara (stories : ℕ) (elevate_time stop_time : ℝ) : ℝ :=
  stories * elevate_time + (stories - 1) * stop_time

theorem slowest_time 
  (build_stories : ℕ) (lola_run_time sam_run_time elevate_time stop_time wait_time : ℝ)
  (h_build : build_stories = 50)
  (h_lola_run : lola_run_time = 12) (h_sam_run : sam_run_time = 15)
  (h_elevate : elevate_time = 10) (h_stop : stop_time = 4) (h_wait : wait_time = 20) :
  max (time_lola build_stories lola_run_time) 
    (max (time_sam 25 25 sam_run_time elevate_time stop_time wait_time) 
         (time_tara build_stories elevate_time stop_time)) = 741 := by
  sorry

end slowest_time_l235_235796


namespace smallest_prime_with_digit_sum_23_l235_235127

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Prime p ∧ digit_sum p = 23 ∧ ∀ q : ℕ, Prime q ∧ digit_sum q = 23 → p ≤ q :=
by
  sorry

end smallest_prime_with_digit_sum_23_l235_235127


namespace notebooks_problem_l235_235522

variable (a b c : ℕ)

theorem notebooks_problem (h1 : a + 6 = b + c) (h2 : b + 10 = a + c) : c = 8 :=
  sorry

end notebooks_problem_l235_235522


namespace cos_seven_pi_over_six_l235_235341

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l235_235341


namespace cost_of_50_roses_l235_235756

def cost_of_dozen_roses : ℝ := 24

def is_proportional (n : ℕ) (cost : ℝ) : Prop :=
  cost = (cost_of_dozen_roses / 12) * n

def has_discount (n : ℕ) : Prop :=
  n ≥ 45

theorem cost_of_50_roses :
  ∃ (cost : ℝ), is_proportional 50 cost ∧ has_discount 50 ∧ cost * 0.9 = 90 :=
by
  sorry

end cost_of_50_roses_l235_235756


namespace fraction_habitable_l235_235602

theorem fraction_habitable : (1 / 3) * (1 / 3) = 1 / 9 := 
by 
  sorry

end fraction_habitable_l235_235602


namespace cafeteria_ordered_red_apples_l235_235893

theorem cafeteria_ordered_red_apples
  (R : ℕ) 
  (h : (R + 17) - 10 = 32) : 
  R = 25 :=
sorry

end cafeteria_ordered_red_apples_l235_235893


namespace minimum_value_of_f_l235_235669

noncomputable def f (x : ℝ) : ℝ := x + 4 / (x - 1)

theorem minimum_value_of_f (x : ℝ) (hx : x > 1) : (∃ y : ℝ, f x = 5 ∧ ∀ y > 1, f y ≥ 5) :=
sorry

end minimum_value_of_f_l235_235669


namespace distance_between_first_and_last_stop_in_km_l235_235328

-- Define the total number of stops
def num_stops := 12

-- Define the distance between the third and sixth stops in meters
def dist_3_to_6 := 3300

-- The distance between consecutive stops is the same
def distance_between_first_and_last_stop : ℕ := (num_stops - 1) * (dist_3_to_6 / 3)

-- The distance in kilometers (1 kilometer = 1000 meters)
noncomputable def distance_km : ℝ := distance_between_first_and_last_stop / 1000

-- Statement to prove
theorem distance_between_first_and_last_stop_in_km : distance_km = 12.1 :=
by
  -- Theorem proof should go here
  sorry

end distance_between_first_and_last_stop_in_km_l235_235328


namespace cylinder_dimensions_l235_235294

theorem cylinder_dimensions (r_sphere : ℝ) (r_cylinder h d : ℝ)
  (h_d_eq : h = d) (r_sphere_val : r_sphere = 6) 
  (sphere_area_eq : 4 * Real.pi * r_sphere^2 = 2 * Real.pi * r_cylinder * h) :
  h = 12 ∧ d = 12 :=
by 
  sorry

end cylinder_dimensions_l235_235294


namespace equilateral_triangle_percentage_l235_235337

theorem equilateral_triangle_percentage (s : Real) :
  let area_square := s^2
  let area_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area := area_square + area_triangle
  area_triangle / total_area * 100 = (4 * Real.sqrt 3 - 3) / 13 * 100 := by
  sorry

end equilateral_triangle_percentage_l235_235337


namespace coordinates_of_point_M_l235_235504

theorem coordinates_of_point_M 
  (M : ℝ × ℝ) 
  (dist_x_axis : abs M.2 = 5) 
  (dist_y_axis : abs M.1 = 4) 
  (second_quadrant : M.1 < 0 ∧ M.2 > 0) : 
  M = (-4, 5) := 
sorry

end coordinates_of_point_M_l235_235504


namespace find_common_difference_l235_235996

variable {α : Type*} [LinearOrderedField α]

-- Define the properties of the arithmetic sequence
def arithmetic_sum (a1 d : α) (n : ℕ) : α := n * a1 + (n * (n - 1) * d) / 2

variables (a1 d : α) -- First term and common difference of the arithmetic sequence (to be found)
variable (S : ℕ → α) -- Sum of the first n terms of the arithmetic sequence

-- Conditions given in the problem
axiom sum_3_eq_6 : S 3 = 6
axiom term_3_eq_4 : a1 + 2 * d = 4

-- The question translated into a theorem statement that the common difference is 2
theorem find_common_difference : d = 2 :=
by
  sorry

end find_common_difference_l235_235996


namespace Alfred_spent_on_repairs_l235_235386

noncomputable def AlfredRepairCost (purchase_price selling_price gain_percent : ℚ) : ℚ :=
  let R := (selling_price - purchase_price * (1 + gain_percent)) / (1 + gain_percent)
  R

theorem Alfred_spent_on_repairs :
  AlfredRepairCost 4700 5800 0.017543859649122806 = 1000 := by
  sorry

end Alfred_spent_on_repairs_l235_235386


namespace minimize_expression_l235_235481

theorem minimize_expression (n : ℕ) (h : n > 0) : (n = 10) ↔ (∀ m : ℕ, m > 0 → (n / 2 + 50 / n: ℝ) ≤ (m / 2 + 50 / m: ℝ)) :=
sorry

end minimize_expression_l235_235481


namespace sequence_b_n_l235_235315

theorem sequence_b_n (b : ℕ → ℝ) (h₁ : b 1 = 2) (h₂ : ∀ n, (b (n + 1))^3 = 64 * (b n)^3) : 
    b 50 = 2 * 4^49 :=
sorry

end sequence_b_n_l235_235315


namespace original_plan_months_l235_235271

theorem original_plan_months (x : ℝ) (h : 1 / (x - 6) = 1.4 * (1 / x)) : x = 21 :=
by
  sorry

end original_plan_months_l235_235271


namespace solution_set_for_inequality_l235_235707

theorem solution_set_for_inequality : {x : ℝ | x ≠ 0 ∧ (x-1)/x ≤ 0} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end solution_set_for_inequality_l235_235707


namespace dividend_ratio_l235_235866

theorem dividend_ratio
  (expected_earnings_per_share : ℝ)
  (actual_earnings_per_share : ℝ)
  (dividend_per_share_increase : ℝ)
  (threshold_earnings_increase : ℝ)
  (shares_owned : ℕ)
  (h_expected_earnings : expected_earnings_per_share = 0.8)
  (h_actual_earnings : actual_earnings_per_share = 1.1)
  (h_dividend_increase : dividend_per_share_increase = 0.04)
  (h_threshold_increase : threshold_earnings_increase = 0.1)
  (h_shares_owned : shares_owned = 100)
  : (shares_owned * (expected_earnings_per_share + 
      (actual_earnings_per_share - expected_earnings_per_share) / threshold_earnings_increase * dividend_per_share_increase)) /
    (shares_owned * actual_earnings_per_share) = 46 / 55 :=
by
  sorry

end dividend_ratio_l235_235866


namespace melted_mixture_weight_l235_235122

/-- 
If the ratio of zinc to copper is 9:11 and 27 kg of zinc has been consumed, then the total weight of the melted mixture is 60 kg.
-/
theorem melted_mixture_weight (zinc_weight : ℕ) (ratio_zinc_to_copper : ℕ → ℕ → Prop)
  (h_ratio : ratio_zinc_to_copper 9 11) (h_zinc : zinc_weight = 27) :
  ∃ (total_weight : ℕ), total_weight = 60 :=
by
  sorry

end melted_mixture_weight_l235_235122


namespace sqrt_inequality_l235_235030

theorem sqrt_inequality : 2 * Real.sqrt 2 - Real.sqrt 7 < Real.sqrt 6 - Real.sqrt 5 := by sorry

end sqrt_inequality_l235_235030


namespace friend_spending_l235_235824

-- Definitions based on conditions
def total_spent (you friend : ℝ) : Prop := you + friend = 15
def friend_spent (you friend : ℝ) : Prop := friend = you + 1

-- Prove that the friend's spending equals $8 given the conditions
theorem friend_spending (you friend : ℝ) (htotal : total_spent you friend) (hfriend : friend_spent you friend) : friend = 8 :=
by
  sorry

end friend_spending_l235_235824


namespace at_least_two_equal_l235_235479

-- Define the problem
theorem at_least_two_equal (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x^2 / y) + (y^2 / z) + (z^2 / x) = (x^2 / z) + (y^2 / x) + (z^2 / y)) :
  x = y ∨ y = z ∨ z = x := 
by 
  sorry

end at_least_two_equal_l235_235479


namespace approx_values_relationship_l235_235568

theorem approx_values_relationship : 
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a = b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a > b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a < b) :=
by sorry

end approx_values_relationship_l235_235568


namespace Ariella_total_amount_l235_235708

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

end Ariella_total_amount_l235_235708


namespace eval_g_at_2_l235_235379

def g (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem eval_g_at_2 : g 2 = 3 :=
by {
  -- This is the place for proof steps, currently it is filled with sorry.
  sorry
}

end eval_g_at_2_l235_235379


namespace prime_factors_and_divisors_6440_l235_235204

theorem prime_factors_and_divisors_6440 :
  ∃ (a b c d : ℕ), 6440 = 2^a * 5^b * 7^c * 23^d ∧ a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧
  (a + 1) * (b + 1) * (c + 1) * (d + 1) = 32 :=
by 
  sorry

end prime_factors_and_divisors_6440_l235_235204


namespace eight_n_is_even_l235_235182

theorem eight_n_is_even (n : ℕ) (h : n = 7) : 8 * n = 56 :=
by {
  sorry
}

end eight_n_is_even_l235_235182


namespace table_relation_l235_235236

theorem table_relation (x y : ℕ) (hx : x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6) :
  (y = 3 ∧ x = 2) ∨ (y = 8 ∧ x = 3) ∨ (y = 15 ∧ x = 4) ∨ (y = 24 ∧ x = 5) ∨ (y = 35 ∧ x = 6) ↔ 
  y = x^2 - x + 2 :=
sorry

end table_relation_l235_235236


namespace find_value_of_x_squared_plus_one_over_x_squared_l235_235121

theorem find_value_of_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by 
  sorry

end find_value_of_x_squared_plus_one_over_x_squared_l235_235121


namespace toby_deleted_nine_bad_shots_l235_235894

theorem toby_deleted_nine_bad_shots 
  (x : ℕ)
  (h1 : 63 > x)
  (h2 : (63 - x) + 15 - 3 = 84)
  : x = 9 :=
by
  sorry

end toby_deleted_nine_bad_shots_l235_235894


namespace min_cosine_largest_angle_l235_235253

theorem min_cosine_largest_angle (a b c : ℕ → ℝ) 
  (triangle_inequality: ∀ i, a i ≤ b i ∧ b i ≤ c i)
  (pythagorean_inequality: ∀ i, (a i)^2 + (b i)^2 ≥ (c i)^2)
  (A : ℝ := ∑' i, a i)
  (B : ℝ := ∑' i, b i)
  (C : ℝ := ∑' i, c i) :
  (A^2 + B^2 - C^2) / (2 * A * B) ≥ 1 - (Real.sqrt 2) :=
sorry

end min_cosine_largest_angle_l235_235253


namespace find_value_of_a_l235_235854

def pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_value_of_a (a : ℝ) :
  pure_imaginary ((a^3 - a) + (a / (1 - a)) * Complex.I) ↔ a = -1 := 
sorry

end find_value_of_a_l235_235854


namespace intersection_of_sets_l235_235673

def setA : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x : ℝ | 2 < x }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_sets_l235_235673


namespace original_price_of_shirt_l235_235701

theorem original_price_of_shirt (P : ℝ) (h : 0.5625 * P = 18) : P = 32 := 
by 
sorry

end original_price_of_shirt_l235_235701


namespace solve_for_x_l235_235895

theorem solve_for_x (x : ℝ) (h : (8 - x)^2 = x^2) : x = 4 := 
by 
  sorry

end solve_for_x_l235_235895


namespace morse_code_sequences_l235_235243

theorem morse_code_sequences : 
  let number_of_sequences := 
        (2 ^ 1) + (2 ^ 2) + (2 ^ 3) + (2 ^ 4) + (2 ^ 5)
  number_of_sequences = 62 :=
by
  sorry

end morse_code_sequences_l235_235243


namespace total_dog_weight_l235_235483

theorem total_dog_weight (weight_evans_dog weight_ivans_dog : ℕ)
  (h₁ : weight_evans_dog = 63)
  (h₂ : weight_evans_dog = 7 * weight_ivans_dog) :
  weight_evans_dog + weight_ivans_dog = 72 :=
sorry

end total_dog_weight_l235_235483


namespace find_A_l235_235899

theorem find_A (A : ℕ) (B : ℕ) (h₀ : 0 ≤ B) (h₁ : B ≤ 999) :
  1000 * A + B = (A * (A + 1)) / 2 → A = 1999 := sorry

end find_A_l235_235899


namespace min_value_expression_l235_235449

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 3) :
  ∃ x : ℝ, (x = (a^2 + b^2 + 22) / (a + b)) ∧ (x = 8) :=
by
  sorry

end min_value_expression_l235_235449


namespace not_possible_to_create_3_piles_l235_235662

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l235_235662


namespace inequality_am_gm_l235_235247

theorem inequality_am_gm (a b c d : ℝ) (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by
  sorry

end inequality_am_gm_l235_235247


namespace isosceles_triangle_sin_vertex_angle_l235_235968

theorem isosceles_triangle_sin_vertex_angle (A : ℝ) (hA : 0 < A ∧ A < π / 2) 
  (hSinA : Real.sin A = 5 / 13) : 
  Real.sin (2 * A) = 120 / 169 :=
by 
  -- This placeholder indicates where the proof would go
  sorry

end isosceles_triangle_sin_vertex_angle_l235_235968


namespace value_of_a2_sub_b2_l235_235082

theorem value_of_a2_sub_b2 (a b : ℝ) (h1 : a + b = 6) (h2 : a - b = 2) : a^2 - b^2 = 12 :=
by
  sorry

end value_of_a2_sub_b2_l235_235082


namespace jake_car_washes_l235_235775

theorem jake_car_washes :
  ∀ (washes_per_bottle cost_per_bottle total_spent weekly_washes : ℕ),
  washes_per_bottle = 4 →
  cost_per_bottle = 4 →
  total_spent = 20 →
  weekly_washes = 1 →
  (total_spent / cost_per_bottle) * washes_per_bottle / weekly_washes = 20 :=
by
  intros washes_per_bottle cost_per_bottle total_spent weekly_washes
  sorry

end jake_car_washes_l235_235775


namespace add_base8_numbers_l235_235273

def fromBase8 (n : Nat) : Nat :=
  Nat.digits 8 n |> Nat.ofDigits 8

theorem add_base8_numbers : 
  fromBase8 356 + fromBase8 672 + fromBase8 145 = fromBase8 1477 :=
by
  sorry

end add_base8_numbers_l235_235273


namespace larger_sphere_radius_l235_235213

theorem larger_sphere_radius (r : ℝ) (π : ℝ) (h : r^3 = 2) :
  r = 2^(1/3) :=
by
  sorry

end larger_sphere_radius_l235_235213


namespace f_2023_pi_over_3_eq_4_l235_235642

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.cos x
| (n + 1), x => 4 / (2 - f n x)

theorem f_2023_pi_over_3_eq_4 : f 2023 (Real.pi / 3) = 4 := 
  sorry

end f_2023_pi_over_3_eq_4_l235_235642


namespace ratio_q_p_l235_235674

variable (p q : ℝ)
variable (hpq_pos : 0 < p ∧ 0 < q)
variable (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18)

theorem ratio_q_p (p q : ℝ) (hpq_pos : 0 < p ∧ 0 < q) 
    (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18) :
    q / p = (Real.sqrt 5 - 1) / 2 :=
  sorry

end ratio_q_p_l235_235674


namespace number_of_rhombuses_l235_235036

-- Definition: A grid with 25 small equilateral triangles arranged in a larger triangular pattern
def equilateral_grid (n : ℕ) : Prop :=
  n = 25

-- Theorem: Proving the number of rhombuses that can be formed from the grid
theorem number_of_rhombuses (n : ℕ) (h : equilateral_grid n) : ℕ :=
  30 

-- Main proof statement
example (n : ℕ) (h : equilateral_grid n) : number_of_rhombuses n h = 30 :=
by
  sorry

end number_of_rhombuses_l235_235036


namespace cylinder_curved_surface_area_l235_235620

theorem cylinder_curved_surface_area {r h : ℝ} (hr: r = 2) (hh: h = 5) :  2 * Real.pi * r * h = 20 * Real.pi :=
by
  rw [hr, hh]
  sorry

end cylinder_curved_surface_area_l235_235620


namespace original_proposition_converse_inverse_contrapositive_l235_235095

def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = n
def is_real (x : ℝ) : Prop := true

theorem original_proposition (x : ℝ) : is_integer x → is_real x := 
by sorry

theorem converse (x : ℝ) : ¬(is_real x → is_integer x) := 
by sorry

theorem inverse (x : ℝ) : ¬((¬ is_integer x) → (¬ is_real x)) := 
by sorry

theorem contrapositive (x : ℝ) : (¬ is_real x) → (¬ is_integer x) := 
by sorry

end original_proposition_converse_inverse_contrapositive_l235_235095


namespace gain_percentage_l235_235142

theorem gain_percentage (selling_price gain : ℝ) (h_selling : selling_price = 90) (h_gain : gain = 15) : 
  (gain / (selling_price - gain)) * 100 = 20 := 
by
  sorry

end gain_percentage_l235_235142


namespace cos_five_theta_l235_235022

theorem cos_five_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (5 * θ) = (125 * Real.sqrt 15 - 749) / 1024 := 
  sorry

end cos_five_theta_l235_235022


namespace simplify_expression_l235_235537

theorem simplify_expression : (625:ℝ)^(1/4) * (256:ℝ)^(1/2) = 80 := 
by 
  sorry

end simplify_expression_l235_235537


namespace Robin_total_distance_walked_l235_235580

-- Define the conditions
def distance_house_to_city_center := 500
def distance_walked_initially := 200

-- Define the proof problem
theorem Robin_total_distance_walked :
  distance_walked_initially * 2 + distance_house_to_city_center = 900 := by
  sorry

end Robin_total_distance_walked_l235_235580


namespace log_equation_l235_235096

theorem log_equation (x : ℝ) (h0 : x < 1) (h1 : (Real.log x / Real.log 10)^3 - 3 * (Real.log x / Real.log 10) = 243) :
  (Real.log x / Real.log 10)^4 - 4 * (Real.log x / Real.log 10) = 6597 :=
by
  sorry

end log_equation_l235_235096


namespace area_ratio_problem_l235_235410

theorem area_ratio_problem
  (A B C : ℝ) -- Areas of the corresponding regions
  (m n : ℕ)  -- Given ratios
  (PQR_is_right_triangle : true)  -- PQR is a right-angled triangle (placeholder condition)
  (RSTU_is_rectangle : true)  -- RSTU is a rectangle (placeholder condition)
  (ratio_A_B : A / B = m / 2)  -- Ratio condition 1
  (ratio_A_C : A / C = n / 1)  -- Ratio condition 2
  (PTS_sim_TQU_sim_PQR : true)  -- Similar triangles (placeholder condition)
  : n = 9 := 
sorry

end area_ratio_problem_l235_235410


namespace johns_profit_l235_235958

variable (numDucks : ℕ) (duckCost : ℕ) (duckWeight : ℕ) (sellPrice : ℕ)

def totalCost (numDucks duckCost : ℕ) : ℕ :=
  numDucks * duckCost

def totalWeight (numDucks duckWeight : ℕ) : ℕ :=
  numDucks * duckWeight

def totalRevenue (totalWeight sellPrice : ℕ) : ℕ :=
  totalWeight * sellPrice

def profit (totalRevenue totalCost : ℕ) : ℕ :=
  totalRevenue - totalCost

theorem johns_profit :
  totalCost 30 10 = 300 →
  totalWeight 30 4 = 120 →
  totalRevenue 120 5 = 600 →
  profit 600 300 = 300 :=
  by
    intros
    sorry

end johns_profit_l235_235958


namespace jackies_lotion_bottles_l235_235603

theorem jackies_lotion_bottles (L: ℕ) : 
  (10 + 10) + 6 * L + 12 = 50 → L = 3 :=
by
  sorry

end jackies_lotion_bottles_l235_235603


namespace kevin_expected_away_time_l235_235817

theorem kevin_expected_away_time
  (leak_rate : ℝ)
  (bucket_capacity : ℝ)
  (bucket_factor : ℝ)
  (leak_rate_eq : leak_rate = 1.5)
  (bucket_capacity_eq : bucket_capacity = 36)
  (bucket_factor_eq : bucket_factor = 2)
  : ((bucket_capacity / bucket_factor) / leak_rate) = 12 :=
by
  rw [bucket_capacity_eq, leak_rate_eq, bucket_factor_eq]
  sorry

end kevin_expected_away_time_l235_235817


namespace lead_atom_ratio_l235_235536

noncomputable def ratio_of_lead_atoms (average_weight : ℝ) 
  (weight_206 : ℕ) (weight_207 : ℕ) (weight_208 : ℕ) 
  (number_206 : ℕ) (number_207 : ℕ) (number_208 : ℕ) : Prop :=
  average_weight = 207.2 ∧ 
  weight_206 = 206 ∧ 
  weight_207 = 207 ∧ 
  weight_208 = 208 ∧ 
  number_208 = number_206 + number_207 →
  (number_206 : ℚ) / (number_207 : ℚ) = 3 / 2 ∧
  (number_208 : ℚ) / (number_207 : ℚ) = 5 / 2

theorem lead_atom_ratio : ratio_of_lead_atoms 207.2 206 207 208 3 2 5 :=
by sorry

end lead_atom_ratio_l235_235536


namespace fill_pool_time_l235_235822

theorem fill_pool_time (pool_volume : ℕ := 32000) 
                       (num_hoses : ℕ := 5) 
                       (flow_rate_per_hose : ℕ := 4) 
                       (operation_minutes : ℕ := 45) 
                       (maintenance_minutes : ℕ := 15) 
                       : ℕ :=
by
  -- Calculation steps will go here in the actual proof
  sorry

example : fill_pool_time = 47 := by
  -- Proof of the theorem fill_pool_time here
  sorry

end fill_pool_time_l235_235822


namespace average_salary_company_l235_235628

-- Define the conditions
def num_managers : Nat := 15
def num_associates : Nat := 75
def avg_salary_managers : ℤ := 90000
def avg_salary_associates : ℤ := 30000

-- Define the goal to prove
theorem average_salary_company : 
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates) = 40000 := by
  sorry

end average_salary_company_l235_235628


namespace chocolates_cost_l235_235965

-- Define the conditions given in the problem.
def boxes_needed (candies_total : ℕ) (candies_per_box : ℕ) : ℕ := 
    candies_total / candies_per_box

def total_cost_without_discount (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := 
    num_boxes * cost_per_box

def discount (total_cost : ℕ) : ℕ := 
    total_cost * 10 / 100

def final_cost (total_cost : ℕ) (discount : ℕ) : ℕ :=
    total_cost - discount

-- Theorem stating the total cost of buying 660 chocolate after discount is $138.60
theorem chocolates_cost (candies_total : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : 
     candies_total = 660 ∧ candies_per_box = 30 ∧ cost_per_box = 7 → 
     final_cost (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box) 
          (discount (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box)) = 13860 := 
by 
    intros h
    let ⟨h1, h2, h3⟩ := h 
    sorry 

end chocolates_cost_l235_235965


namespace jack_has_42_pounds_l235_235374

noncomputable def jack_pounds (P : ℕ) : Prop :=
  let euros := 11
  let yen := 3000
  let pounds_per_euro := 2
  let yen_per_pound := 100
  let total_yen := 9400
  let pounds_from_euros := euros * pounds_per_euro
  let pounds_from_yen := yen / yen_per_pound
  let total_pounds := P + pounds_from_euros + pounds_from_yen
  total_pounds * yen_per_pound = total_yen

theorem jack_has_42_pounds : jack_pounds 42 :=
  sorry

end jack_has_42_pounds_l235_235374


namespace power_function_decreasing_n_value_l235_235126

theorem power_function_decreasing_n_value (n : ℤ) (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < x → f x = (n^2 + 2 * n - 2) * x^(n^2 - 3 * n)) →
  (∀ x y : ℝ, 0 < x ∧ 0 < y → x < y → f y < f x) →
  n = 1 := 
by
  sorry

end power_function_decreasing_n_value_l235_235126


namespace range_of_a_l235_235263

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ a ≥ 9 :=
by
  sorry

end range_of_a_l235_235263


namespace ratio_of_areas_of_circles_l235_235232

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l235_235232


namespace jane_paid_five_l235_235482

noncomputable def cost_of_apple : ℝ := 0.75
noncomputable def change_received : ℝ := 4.25
noncomputable def amount_paid : ℝ := cost_of_apple + change_received

theorem jane_paid_five : amount_paid = 5.00 :=
by
  sorry

end jane_paid_five_l235_235482


namespace actual_books_bought_l235_235970

def initial_spending : ℕ := 180
def planned_books (x : ℕ) : Prop := initial_spending / x - initial_spending / (5 * x / 4) = 9

theorem actual_books_bought (x : ℕ) (hx : planned_books x) : (5 * x / 4) = 5 :=
by
  sorry

end actual_books_bought_l235_235970


namespace part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l235_235994

variable {x a : ℝ}

theorem part1_solution (h1 : a > 1 / 3) (h2 : (a * x - 1) / (x ^ 2 - 1) = 0) : x = 3 := by
  sorry

theorem part2_solution_1 (h1 : -1 < a) (h2 : a < 0) : {x | x < (1 / a) ∨ (-1 < x ∧ x < 1)} := by
  sorry

theorem part2_solution_2 (h1 : a = -1) : {x | x < 1 ∧ x ≠ -1} := by
  sorry

theorem part2_solution_3 (h1 : a < -1) : {x | x < -1 ∨ (1 / a < x ∧ x < 1)} := by
  sorry

end part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l235_235994


namespace TrishulPercentageLessThanRaghu_l235_235532

-- Define the variables and conditions
variables (R T V : ℝ)

-- Raghu's investment is Rs. 2200
def RaghuInvestment := (R : ℝ) = 2200

-- Vishal invested 10% more than Trishul
def VishalInvestment := (V : ℝ) = 1.10 * T

-- Total sum of investments is Rs. 6358
def TotalInvestment := R + T + V = 6358

-- Define the proof statement
theorem TrishulPercentageLessThanRaghu (R_is_2200 : RaghuInvestment R) 
    (V_is_10_percent_more : VishalInvestment V T) 
    (total_sum_is_6358 : TotalInvestment R T V) : 
  ((2200 - T) / 2200) * 100 = 10 :=
sorry

end TrishulPercentageLessThanRaghu_l235_235532


namespace minimum_minutes_for_planB_cheaper_l235_235664

-- Define the costs for Plan A and Plan B as functions of minutes
def planACost (x : Nat) : Nat := 1500 + 12 * x
def planBCost (x : Nat) : Nat := 3000 + 6 * x

-- Statement to prove
theorem minimum_minutes_for_planB_cheaper : 
  ∃ x : Nat, (planBCost x < planACost x) ∧ ∀ y : Nat, y < x → planBCost y ≥ planACost y :=
by
  sorry

end minimum_minutes_for_planB_cheaper_l235_235664


namespace three_lines_intersect_single_point_l235_235771

theorem three_lines_intersect_single_point (a : ℝ) :
  (∀ x y : ℝ, (x + 2*y + a) * (x^2 - y^2) = 0) ↔ a = 0 := by
  sorry

end three_lines_intersect_single_point_l235_235771


namespace johns_chore_homework_time_l235_235792

-- Definitions based on problem conditions
def cartoons_time : ℕ := 150  -- John's cartoon watching time in minutes
def chores_homework_per_10 : ℕ := 13  -- 13 minutes combined chores and homework per 10 minutes of cartoons
def cartoon_period : ℕ := 10  -- Per 10 minutes period

-- Theorem statement
theorem johns_chore_homework_time :
  cartoons_time / cartoon_period * chores_homework_per_10 = 195 :=
by sorry

end johns_chore_homework_time_l235_235792


namespace sedrich_more_jelly_beans_l235_235934

-- Define the given conditions
def napoleon_jelly_beans : ℕ := 17
def mikey_jelly_beans : ℕ := 19
def sedrich_jelly_beans (x : ℕ) : ℕ := napoleon_jelly_beans + x

-- Define the main theorem to be proved
theorem sedrich_more_jelly_beans (x : ℕ) :
  2 * (napoleon_jelly_beans + sedrich_jelly_beans x) = 4 * mikey_jelly_beans → x = 4 :=
by
  -- Proving the theorem
  sorry

end sedrich_more_jelly_beans_l235_235934


namespace solve_for_x_l235_235067

theorem solve_for_x (x : ℕ) : (1 : ℚ) / 2 = x / 8 → x = 4 := by
  sorry

end solve_for_x_l235_235067


namespace price_of_each_shirt_l235_235378

-- Defining the conditions
def total_pants_cost (pants_price : ℕ) (num_pants : ℕ) := num_pants * pants_price
def total_amount_spent (amount_given : ℕ) (change_received : ℕ) := amount_given - change_received
def total_shirts_cost (amount_spent : ℕ) (pants_cost : ℕ) := amount_spent - pants_cost
def price_per_shirt (shirts_total_cost : ℕ) (num_shirts : ℕ) := shirts_total_cost / num_shirts

-- The main statement
theorem price_of_each_shirt (pants_price num_pants amount_given change_received num_shirts : ℕ) :
  num_pants = 2 →
  pants_price = 54 →
  amount_given = 250 →
  change_received = 10 →
  num_shirts = 4 →
  price_per_shirt (total_shirts_cost (total_amount_spent amount_given change_received) 
                   (total_pants_cost pants_price num_pants)) num_shirts = 33
:= by
  sorry

end price_of_each_shirt_l235_235378


namespace intersection_points_zero_l235_235865

theorem intersection_points_zero (a b c: ℝ) (h1: b^2 = a * c) (h2: a * c > 0) : 
  ∀ x: ℝ, ¬ (a * x^2 + b * x + c = 0) := 
by 
  sorry

end intersection_points_zero_l235_235865


namespace sin_neg_1290_l235_235871

theorem sin_neg_1290 : Real.sin (-(1290 : ℝ) * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_neg_1290_l235_235871


namespace amount_of_solution_added_l235_235161

variable (x : ℝ)

-- Condition: The solution contains 90% alcohol
def solution_alcohol_amount (x : ℝ) : ℝ := 0.9 * x

-- Condition: Total volume of the new mixture after adding 16 liters of water
def total_volume (x : ℝ) : ℝ := x + 16

-- Condition: The percentage of alcohol in the new mixture is 54%
def new_mixture_alcohol_amount (x : ℝ) : ℝ := 0.54 * (total_volume x)

-- The proof goal: the amount of solution added is 24 liters
theorem amount_of_solution_added : new_mixture_alcohol_amount x = solution_alcohol_amount x → x = 24 :=
by
  sorry

end amount_of_solution_added_l235_235161


namespace total_number_of_guests_l235_235114

theorem total_number_of_guests (A C S : ℕ) (hA : A = 58) (hC : C = A - 35) (hS : S = 2 * C) : 
  A + C + S = 127 := 
by
  sorry

end total_number_of_guests_l235_235114


namespace zoe_total_songs_l235_235645

def initial_songs : ℕ := 15
def deleted_songs : ℕ := 8
def added_songs : ℕ := 50

theorem zoe_total_songs : initial_songs - deleted_songs + added_songs = 57 := by
  sorry

end zoe_total_songs_l235_235645


namespace problem_solution_l235_235762

theorem problem_solution (x y z : ℝ) (h1 : 2 * x - y - 2 * z - 6 = 0) (h2 : x^2 + y^2 + z^2 ≤ 4) :
  2 * x + y + z = 2 / 3 := 
by 
  sorry

end problem_solution_l235_235762


namespace problem_l235_235541

def otimes (x y : ℝ) : ℝ := x^3 + y - 2 * x

theorem problem (k : ℝ) : otimes k (otimes k k) = 2 * k^3 - 3 * k :=
by
  sorry

end problem_l235_235541


namespace total_earning_proof_l235_235556

noncomputable def total_earning (daily_wage_c : ℝ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) : ℝ :=
  let daily_wage_a := (ratio_a : ℝ) / (ratio_c : ℝ) * daily_wage_c
  let daily_wage_b := (ratio_b : ℝ) / (ratio_c : ℝ) * daily_wage_c
  (daily_wage_a * days_a) + (daily_wage_b * days_b) + (daily_wage_c * days_c)

theorem total_earning_proof : 
  total_earning 71.15384615384615 16 9 4 3 4 5 = 1480 := 
by 
  -- calculations here
  sorry

end total_earning_proof_l235_235556


namespace framing_feet_required_l235_235912

noncomputable def original_width := 5
noncomputable def original_height := 7
noncomputable def enlargement_factor := 4
noncomputable def border_width := 3
noncomputable def inches_per_foot := 12

theorem framing_feet_required :
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (final_width + final_height)
  let framing_feet := perimeter / inches_per_foot
  framing_feet = 10 :=
by
  sorry

end framing_feet_required_l235_235912


namespace min_packs_120_cans_l235_235143

theorem min_packs_120_cans (p8 p16 p32 : ℕ) (total_cans packs_needed : ℕ) :
  total_cans = 120 →
  p8 * 8 + p16 * 16 + p32 * 32 = total_cans →
  packs_needed = p8 + p16 + p32 →
  (∀ (q8 q16 q32 : ℕ), q8 * 8 + q16 * 16 + q32 * 32 = total_cans → q8 + q16 + q32 ≥ packs_needed) →
  packs_needed = 5 :=
by {
  sorry
}

end min_packs_120_cans_l235_235143


namespace complex_props_hold_l235_235777

theorem complex_props_hold (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ (a^2 = a*b → a = b) :=
by
  sorry

end complex_props_hold_l235_235777


namespace extra_chairs_added_l235_235679

theorem extra_chairs_added (rows cols total_chairs extra_chairs : ℕ) 
  (h1 : rows = 7) 
  (h2 : cols = 12) 
  (h3 : total_chairs = 95) 
  (h4 : extra_chairs = total_chairs - rows * cols) : 
  extra_chairs = 11 := by 
  sorry

end extra_chairs_added_l235_235679


namespace harry_total_cost_in_silver_l235_235151

def cost_of_spellbooks_in_gold := 5 * 5
def cost_of_potion_kits_in_silver := 3 * 20
def cost_of_owl_in_gold := 28
def gold_to_silver := 9

def cost_in_silver :=
  (cost_of_spellbooks_in_gold + cost_of_owl_in_gold) * gold_to_silver + cost_of_potion_kits_in_silver

theorem harry_total_cost_in_silver : cost_in_silver = 537 := by
  sorry

end harry_total_cost_in_silver_l235_235151


namespace find_a_l235_235587

def setA (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def setB : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) : setA a ⊆ setB ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l235_235587


namespace find_varphi_l235_235663

theorem find_varphi 
  (f g : ℝ → ℝ) 
  (x1 x2 varphi : ℝ) 
  (h_f : ∀ x, f x = 2 * Real.cos (2 * x)) 
  (h_g : ∀ x, g x = 2 * Real.cos (2 * x - 2 * varphi)) 
  (h_varphi_range : 0 < varphi ∧ varphi < π / 2) 
  (h_diff_cos : |f x1 - g x2| = 4) 
  (h_min_dist : |x1 - x2| = π / 6) 
: varphi = π / 3 := 
sorry

end find_varphi_l235_235663


namespace all_statements_false_l235_235468

theorem all_statements_false (r1 r2 : ℝ) (h1 : r1 ≠ r2) (h2 : r1 + r2 = 5) (h3 : r1 * r2 = 6) :
  ¬(|r1 + r2| > 6) ∧ ¬(3 < |r1 * r2| ∧ |r1 * r2| < 8) ∧ ¬(r1 < 0 ∧ r2 < 0) :=
by
  sorry

end all_statements_false_l235_235468


namespace min_combined_horses_and_ponies_l235_235987

theorem min_combined_horses_and_ponies : 
  ∀ (P : ℕ), 
  (∃ (P' : ℕ), 
    (P = P' ∧ (∃ (x : ℕ), x = 3 * P' / 10 ∧ x = 3 * P' / 16) ∧
     (∃ (y : ℕ), y = 5 * x / 8) ∧ 
      ∀ (H : ℕ), (H = 3 + P')) → 
  P + (3 + P) = 35) := 
sorry

end min_combined_horses_and_ponies_l235_235987


namespace hyperbola_eccentricity_l235_235643

-- Definitions of conditions
def asymptotes_of_hyperbola (a b x y : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  (b * x + a * y = 0) ∨ (b * x - a * y = 0)

def circle_tangent_to_asymptotes (x y a b : ℝ) : Prop :=
  ∀ x1 y1 : ℝ, 
  (x1, y1) = (0, 4) → 
  (Real.sqrt (b^2 + a^2) = 2 * a)

-- Main statement
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (h_asymptotes : ∀ (x y : ℝ), asymptotes_of_hyperbola a b x y h_a h_b) 
  (h_tangent : circle_tangent_to_asymptotes 0 4 a b) : 
  ∃ e : ℝ, e = 2 := 
sorry

end hyperbola_eccentricity_l235_235643


namespace gasoline_reduction_l235_235205

theorem gasoline_reduction
  (P Q : ℝ)
  (h1 : 0 < P)
  (h2 : 0 < Q)
  (price_increase_percent : ℝ := 0.25)
  (spending_increase_percent : ℝ := 0.05)
  (new_price : ℝ := P * (1 + price_increase_percent))
  (new_total_cost : ℝ := (P * Q) * (1 + spending_increase_percent)) :
  100 - (100 * (new_total_cost / new_price) / Q) = 16 :=
by
  sorry

end gasoline_reduction_l235_235205


namespace xy_equals_18_l235_235098

theorem xy_equals_18 (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 :=
by
  sorry

end xy_equals_18_l235_235098


namespace carpenter_job_duration_l235_235656

theorem carpenter_job_duration
  (total_estimate : ℤ)
  (carpenter_hourly_rate : ℤ)
  (assistant_hourly_rate : ℤ)
  (material_cost : ℤ)
  (H1 : total_estimate = 1500)
  (H2 : carpenter_hourly_rate = 35)
  (H3 : assistant_hourly_rate = 25)
  (H4 : material_cost = 720) :
  (total_estimate - material_cost) / (carpenter_hourly_rate + assistant_hourly_rate) = 13 :=
by
  sorry

end carpenter_job_duration_l235_235656


namespace count_correct_conclusions_l235_235554

structure Point where
  x : ℝ
  y : ℝ

def isDoublingPoint (P Q : Point) : Prop :=
  2 * (P.x + Q.x) = P.y + Q.y

def P1 : Point := {x := 2, y := 0}

def Q1 : Point := {x := 2, y := 8}
def Q2 : Point := {x := -3, y := -2}

def onLine (P : Point) : Prop :=
  P.y = P.x + 2

def onParabola (P : Point) : Prop :=
  P.y = P.x ^ 2 - 2 * P.x - 3

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

theorem count_correct_conclusions :
  (isDoublingPoint P1 Q1) ∧
  (isDoublingPoint P1 Q2) ∧
  (∃ A : Point, onLine A ∧ isDoublingPoint P1 A ∧ A = {x := -2, y := 0}) ∧
  (∃ B₁ B₂ : Point, onParabola B₁ ∧ onParabola B₂ ∧ isDoublingPoint P1 B₁ ∧ isDoublingPoint P1 B₂) ∧
  (∃ B : Point, isDoublingPoint P1 B ∧
   ∀ P : Point, isDoublingPoint P1 P → dist P1 P ≥ dist P1 B ∧
   dist P1 B = 8 * (5:ℝ)^(1/2) / 5) :=
by sorry

end count_correct_conclusions_l235_235554


namespace sum_of_possible_values_of_d_l235_235475

def base_digits (n : ℕ) (b : ℕ) : ℕ := 
  if n = 0 then 1 else Nat.log (n + 1) b

theorem sum_of_possible_values_of_d :
  let min_val_7 := 1 * 7^3
  let max_val_7 := 6 * 7^3 + 6 * 7^2 + 6 * 7^1 + 6 * 7^0
  let min_val_10 := 343
  let max_val_10 := 2400
  let d1 := base_digits min_val_10 3
  let d2 := base_digits max_val_10 3
  d1 + d2 = 13 := sorry

end sum_of_possible_values_of_d_l235_235475


namespace find_a_of_min_value_of_f_l235_235525

noncomputable def f (a x : ℝ) : ℝ := 4 * Real.sin (2 * x) + 3 * Real.cos (2 * x) + 2 * a * Real.sin x + 4 * a * Real.cos x

theorem find_a_of_min_value_of_f :
  (∃ a : ℝ, (∀ x : ℝ, f a x ≥ -6) ∧ (∃ x : ℝ, f a x = -6)) → (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by
  sorry

end find_a_of_min_value_of_f_l235_235525


namespace coefficient_of_x_in_expansion_l235_235118

noncomputable def binomial_expansion_term (r : ℕ) : ℤ :=
  (-1)^r * (2^(5-r)) * Nat.choose 5 r

theorem coefficient_of_x_in_expansion :
  binomial_expansion_term 3 = -40 := by
  sorry

end coefficient_of_x_in_expansion_l235_235118


namespace log2_15_eq_formula_l235_235949

theorem log2_15_eq_formula (a b : ℝ) (h1 : a = Real.log 6 / Real.log 3) (h2 : b = Real.log 20 / Real.log 5) :
  Real.log 15 / Real.log 2 = (2 * a + b - 3) / ((a - 1) * (b - 1)) :=
by
  sorry

end log2_15_eq_formula_l235_235949


namespace tan_sum_identity_l235_235910

noncomputable def tan_25 := Real.tan (Real.pi / 180 * 25)
noncomputable def tan_35 := Real.tan (Real.pi / 180 * 35)
noncomputable def sqrt_3 := Real.sqrt 3

theorem tan_sum_identity :
  tan_25 + tan_35 + sqrt_3 * tan_25 * tan_35 = 1 :=
by
  sorry

end tan_sum_identity_l235_235910


namespace parallelepiped_diagonal_inequality_l235_235300

theorem parallelepiped_diagonal_inequality 
  (a b c d : ℝ) 
  (h_d : d = Real.sqrt (a^2 + b^2 + c^2)) : 
  a^2 + b^2 + c^2 ≥ d^2 / 3 := 
by 
  sorry

end parallelepiped_diagonal_inequality_l235_235300


namespace hyperbola_asymptote_l235_235742

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 ∧ ∀ (x y : ℝ), (y = 3/5 * x ↔ y = 3 / 5 * x)) → a = 5 :=
by
  sorry

end hyperbola_asymptote_l235_235742


namespace minimum_value_exists_l235_235613

theorem minimum_value_exists (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_condition : x + 4 * y = 2) : 
  ∃ z : ℝ, z = (x + 40 * y + 4) / (3 * x * y) ∧ z ≥ 18 :=
by
  sorry

end minimum_value_exists_l235_235613


namespace eggs_left_after_capital_recovered_l235_235351

-- Conditions as definitions
def eggs_in_crate := 30
def crate_cost_dollars := 5
def price_per_egg_cents := 20

-- The amount of cents in a dollar
def cents_per_dollar := 100

-- Total cost in cents
def crate_cost_cents := crate_cost_dollars * cents_per_dollar

-- The number of eggs needed to recover the capital
def eggs_to_recover_capital := crate_cost_cents / price_per_egg_cents

-- The number of eggs left
def eggs_left := eggs_in_crate - eggs_to_recover_capital

-- The theorem stating the problem
theorem eggs_left_after_capital_recovered : eggs_left = 5 :=
by
  sorry

end eggs_left_after_capital_recovered_l235_235351


namespace find_first_number_l235_235858

theorem find_first_number (x y : ℝ) (h1 : x + y = 50) (h2 : 2 * (x - y) = 20) : x = 30 :=
by
  sorry

end find_first_number_l235_235858


namespace complement_intersection_l235_235365

-- Definitions
def A : Set ℝ := { x | x^2 + x - 6 < 0 }
def B : Set ℝ := { x | x > 1 }

-- Stating the problem
theorem complement_intersection (x : ℝ) : x ∈ (Aᶜ ∩ B) ↔ x ∈ Set.Ici 2 :=
by sorry

end complement_intersection_l235_235365


namespace smallest_fraction_of_land_l235_235461

noncomputable def smallest_share (n : ℕ) : ℚ :=
  if n = 150 then 1 / (2 * 3^49) else 0

theorem smallest_fraction_of_land :
  smallest_share 150 = 1 / (2 * 3^49) :=
sorry

end smallest_fraction_of_land_l235_235461


namespace power_function_evaluation_l235_235751

noncomputable def f (α : ℝ) (x : ℝ) := x ^ α

theorem power_function_evaluation (α : ℝ) (h : f α 8 = 2) : f α (-1/8) = -1/2 :=
by
  sorry

end power_function_evaluation_l235_235751


namespace trajectory_description_l235_235940

def trajectory_of_A (x y : ℝ) (m : ℝ) : Prop :=
  m * x^2 - y^2 = m ∧ y ≠ 0
  
theorem trajectory_description (x y m : ℝ) (h : m ≠ 0) :
  trajectory_of_A x y m →
    (m < -1 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m = -1 → (x^2 + y^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0))) ∧
    (-1 < m ∧ m < 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m > 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) :=
by
  intro h_trajectory
  sorry

end trajectory_description_l235_235940


namespace trig_expression_l235_235846

theorem trig_expression (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 3) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by sorry

end trig_expression_l235_235846


namespace find_common_ratio_of_geometric_sequence_l235_235189

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem find_common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : geometric_sequence a)
  (h_decreasing : ∀ n : ℕ, a n > a (n + 1))
  (h1 : a 1 * a 5 = 9)
  (h2 : a 2 + a 4 = 10) : 
  q = -1/3 :=
sorry

end find_common_ratio_of_geometric_sequence_l235_235189


namespace digit_pairs_for_divisibility_by_36_l235_235153

theorem digit_pairs_for_divisibility_by_36 (A B : ℕ) :
  (0 ≤ A) ∧ (A ≤ 9) ∧ (0 ≤ B) ∧ (B ≤ 9) ∧
  (∃ k4 k9 : ℕ, (10 * 5 + B = 4 * k4) ∧ (20 + A + B = 9 * k9)) ↔ 
  ((A = 5 ∧ B = 2) ∨ (A = 1 ∧ B = 6)) :=
by sorry

end digit_pairs_for_divisibility_by_36_l235_235153


namespace mike_arcade_ratio_l235_235184

theorem mike_arcade_ratio :
  ∀ (weekly_pay food_cost hourly_rate play_minutes : ℕ),
    weekly_pay = 100 →
    food_cost = 10 →
    hourly_rate = 8 →
    play_minutes = 300 →
    (food_cost + (play_minutes / 60) * hourly_rate) / weekly_pay = 1 / 2 := 
by
  intros weekly_pay food_cost hourly_rate play_minutes h1 h2 h3 h4
  sorry

end mike_arcade_ratio_l235_235184


namespace divide_by_3_result_l235_235476

-- Definitions
def n : ℕ := 4 * 12

theorem divide_by_3_result (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end divide_by_3_result_l235_235476


namespace singers_in_fifth_verse_l235_235183

theorem singers_in_fifth_verse (choir : ℕ) (absent : ℕ) (participating : ℕ) 
(half_first_verse : ℕ) (third_second_verse : ℕ) (quarter_third_verse : ℕ) 
(fifth_fourth_verse : ℕ) (late_singers : ℕ) :
  choir = 70 → 
  absent = 10 → 
  participating = choir - absent →
  half_first_verse = participating / 2 → 
  third_second_verse = (participating - half_first_verse) / 3 →
  quarter_third_verse = (participating - half_first_verse - third_second_verse) / 4 →
  fifth_fourth_verse = (participating - half_first_verse - third_second_verse - quarter_third_verse) / 5 →
  late_singers = 5 →
  participating = 60 :=
by sorry

end singers_in_fifth_verse_l235_235183


namespace ratio_noah_to_joe_l235_235465

def noah_age_after_10_years : ℕ := 22
def years_elapsed : ℕ := 10
def joe_age : ℕ := 6
def noah_age : ℕ := noah_age_after_10_years - years_elapsed

theorem ratio_noah_to_joe : noah_age / joe_age = 2 := by
  -- calculation omitted for brevity
  sorry

end ratio_noah_to_joe_l235_235465


namespace sum_of_coeffs_l235_235440

theorem sum_of_coeffs (A B C D : ℤ) (h₁ : A = 1) (h₂ : B = -1) (h₃ : C = -12) (h₄ : D = 3) :
  A + B + C + D = -9 := 
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end sum_of_coeffs_l235_235440


namespace solution_l235_235577

-- Define the conditions
variable (f : ℝ → ℝ)
variable (f_odd : ∀ x, f (-x) = -f x)
variable (f_periodic : ∀ x, f (x + 1) = f (1 - x))
variable (f_cubed : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x ^ 3)

-- Define the goal
theorem solution : f 2019 = -1 :=
by sorry

end solution_l235_235577


namespace integer_in_range_l235_235550

theorem integer_in_range (x : ℤ) 
  (h1 : 0 < x) 
  (h2 : x < 7)
  (h3 : 0 < x)
  (h4 : x < 15)
  (h5 : -1 < x)
  (h6 : x < 5)
  (h7 : 0 < x)
  (h8 : x < 3)
  (h9 : x + 2 < 4) : x = 1 := 
sorry

end integer_in_range_l235_235550


namespace sales_proof_valid_l235_235714

variables (T: ℝ) (Teq: T = 30)
noncomputable def check_sales_proof : Prop :=
  (6.4 * T + 228 = 420)

theorem sales_proof_valid (T : ℝ) (Teq: T = 30) : check_sales_proof T :=
  by
    rw [Teq]
    norm_num
    sorry

end sales_proof_valid_l235_235714


namespace N_intersect_M_complement_l235_235447

-- Definitions based on given conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def N : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def M_complement : Set ℝ := { x | x < -2 ∨ x > 3 }  -- complement of M in ℝ

-- Lean statement for the proof problem
theorem N_intersect_M_complement :
  N ∩ M_complement = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end N_intersect_M_complement_l235_235447


namespace jacob_total_bill_l235_235764

def base_cost : ℝ := 25
def included_hours : ℕ := 25
def cost_per_text : ℝ := 0.08
def cost_per_extra_minute : ℝ := 0.13
def jacob_texts : ℕ := 150
def jacob_hours : ℕ := 31

theorem jacob_total_bill : 
  let extra_minutes := (jacob_hours - included_hours) * 60
  let total_cost := base_cost + jacob_texts * cost_per_text + extra_minutes * cost_per_extra_minute
  total_cost = 83.80 := 
by 
  -- Placeholder for proof
  sorry

end jacob_total_bill_l235_235764


namespace find_d_l235_235654

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + 2 * c^2 + 4 = 2 * d + Real.sqrt (a^2 + b^2 + c - d)) :
  d = 1/2 :=
sorry

end find_d_l235_235654


namespace a_star_b_value_l235_235314

theorem a_star_b_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) (h3 : b = 8) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 3 / 8 := by
sorry

end a_star_b_value_l235_235314


namespace faulty_keys_l235_235411

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l235_235411


namespace train_length_l235_235439

theorem train_length (speed_km_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_hr = 72) (h_time : time_sec = 12) : 
  ∃ length_m : ℕ, length_m = 240 := 
by
  sorry

end train_length_l235_235439


namespace total_students_university_l235_235195

theorem total_students_university :
  ∀ (sample_size freshmen sophomores other_sample other_total total_students : ℕ),
  sample_size = 500 →
  freshmen = 200 →
  sophomores = 100 →
  other_sample = 200 →
  other_total = 3000 →
  total_students = (other_total * sample_size) / other_sample →
  total_students = 7500 :=
by
  intros sample_size freshmen sophomores other_sample other_total total_students
  sorry

end total_students_university_l235_235195


namespace find_r_value_l235_235499

theorem find_r_value (n : ℕ) (r s : ℕ) (h_s : s = 2^n - 1) (h_r : r = 3^s - s) (h_n : n = 3) : r = 2180 :=
by
  sorry

end find_r_value_l235_235499


namespace toys_produced_each_day_l235_235192

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_worked_per_week : ℕ)
  (same_number_toys_each_day : Prop) : 
  total_weekly_production = 4340 → days_worked_per_week = 2 → 
  same_number_toys_each_day →
  (total_weekly_production / days_worked_per_week = 2170) :=
by
  intros h_production h_days h_same_toys
  -- proof skipped
  sorry

end toys_produced_each_day_l235_235192


namespace distance_between_cyclists_l235_235462

def cyclist_distance (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * t

theorem distance_between_cyclists :
  cyclist_distance 10 25 1.4285714285714286 = 50 := by
  sorry

end distance_between_cyclists_l235_235462


namespace total_distance_covered_l235_235860

noncomputable def speed_train_a : ℚ := 80          -- Speed of Train A in kmph
noncomputable def speed_train_b : ℚ := 110         -- Speed of Train B in kmph
noncomputable def duration : ℚ := 15               -- Duration in minutes
noncomputable def conversion_factor : ℚ := 60      -- Conversion factor from hours to minutes

theorem total_distance_covered : 
    (speed_train_a / conversion_factor) * duration + 
    (speed_train_b / conversion_factor) * duration = 47.5 :=
by
  sorry

end total_distance_covered_l235_235860


namespace power_of_prime_implies_n_prime_l235_235362

theorem power_of_prime_implies_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Nat.Prime p) :
  3^n - 2^n = p^k → Nat.Prime n :=
by
  sorry

end power_of_prime_implies_n_prime_l235_235362


namespace find_a_and_b_l235_235530

noncomputable def f (x: ℝ) (b: ℝ): ℝ := x^2 + 5*x + b
noncomputable def g (x: ℝ) (b: ℝ): ℝ := 2*b*x + 3

theorem find_a_and_b (a b: ℝ):
  (∀ x: ℝ, f (g x b) b = a * x^2 + 30 * x + 24) →
  a = 900 / 121 ∧ b = 15 / 11 :=
by
  intro H
  -- Proof is omitted as requested
  sorry

end find_a_and_b_l235_235530


namespace magnification_proof_l235_235607

-- Define the conditions: actual diameter of the tissue and diameter of the magnified image
def actual_diameter := 0.0002
def magnified_diameter := 0.2

-- Define the magnification factor
def magnification_factor := magnified_diameter / actual_diameter

-- Prove that the magnification factor is 1000
theorem magnification_proof : magnification_factor = 1000 := by
  unfold magnification_factor
  unfold magnified_diameter
  unfold actual_diameter
  norm_num
  sorry

end magnification_proof_l235_235607


namespace inequality_solution_set_l235_235076

theorem inequality_solution_set (x : ℝ) : (|x - 1| + 2 * x > 4) ↔ (x > 3) := 
sorry

end inequality_solution_set_l235_235076


namespace neg_of_all_men_are_honest_l235_235377

variable {α : Type} (man honest : α → Prop)

theorem neg_of_all_men_are_honest :
  ¬ (∀ x, man x → honest x) ↔ ∃ x, man x ∧ ¬ honest x :=
by
  sorry

end neg_of_all_men_are_honest_l235_235377


namespace isosceles_triangle_area_l235_235175

theorem isosceles_triangle_area :
  ∀ (P Q R S : ℝ) (h1 : dist P Q = 26) (h2 : dist P R = 26) (h3 : dist Q R = 50),
  ∃ (area : ℝ), area = 25 * Real.sqrt 51 :=
by
  sorry

end isosceles_triangle_area_l235_235175


namespace sin_cos_equiv_l235_235470

theorem sin_cos_equiv (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) :
  Real.sin x + 5 * Real.cos x = -1/2 ∨ Real.sin x + 5 * Real.cos x = 17/13 := 
by
  sorry

end sin_cos_equiv_l235_235470


namespace necessarily_positive_l235_235835

theorem necessarily_positive (x y z : ℝ) (h1 : 0 < x ∧ x < 2) (h2 : -2 < y ∧ y < 0) (h3 : 0 < z ∧ z < 3) : 
  y + 2 * z > 0 := 
sorry

end necessarily_positive_l235_235835


namespace quadratic_completion_l235_235763

theorem quadratic_completion (x : ℝ) :
  (x^2 + 6 * x - 2) = ((x + 3)^2 - 11) := sorry

end quadratic_completion_l235_235763


namespace cos_2beta_correct_l235_235922

open Real

theorem cos_2beta_correct (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
    (h3 : tan α = 1 / 7) (h4 : cos (α + β) = 2 * sqrt 5 / 5) :
    cos (2 * β) = 4 / 5 := 
  sorry

end cos_2beta_correct_l235_235922


namespace cookie_sheet_perimeter_l235_235282

def width : ℕ := 10
def length : ℕ := 2

def perimeter (w l : ℕ) : ℕ := 2 * w + 2 * l

theorem cookie_sheet_perimeter : 
  perimeter width length = 24 := by
  sorry

end cookie_sheet_perimeter_l235_235282


namespace millions_place_correct_l235_235100

def number := 345000000
def hundred_millions_place := number / 100000000 % 10  -- 3
def ten_millions_place := number / 10000000 % 10  -- 4
def millions_place := number / 1000000 % 10  -- 5

theorem millions_place_correct : millions_place = 5 := 
by 
  -- Mathematical proof goes here
  sorry

end millions_place_correct_l235_235100


namespace pencils_and_notebooks_cost_l235_235464

theorem pencils_and_notebooks_cost
    (p n : ℝ)
    (h1 : 8 * p + 10 * n = 5.36)
    (h2 : 12 * (p - 0.05) + 5 * n = 4.05) :
    15 * (p - 0.05) + 12 * n = 7.01 := 
sorry

end pencils_and_notebooks_cost_l235_235464


namespace geometric_series_sum_l235_235574

theorem geometric_series_sum : 
    ∑' n : ℕ, (1 : ℝ) * (-1 / 2) ^ n = 2 / 3 :=
by
    sorry

end geometric_series_sum_l235_235574


namespace min_value_of_expression_l235_235128

theorem min_value_of_expression 
  (a b : ℝ) 
  (h : a > 0) 
  (h₀ : b > 0) 
  (h₁ : 2*a + b = 2) : 
  ∃ c : ℝ, c = (8*a + b) / (a*b) ∧ c = 9 :=
sorry

end min_value_of_expression_l235_235128


namespace fewer_ducks_than_chickens_and_geese_l235_235887

/-- There are 42 chickens and 48 ducks on the farm, and there are as many geese as there are chickens. 
Prove that there are 36 fewer ducks than the number of chickens and geese combined. -/
theorem fewer_ducks_than_chickens_and_geese (chickens ducks geese : ℕ)
  (h_chickens : chickens = 42)
  (h_ducks : ducks = 48)
  (h_geese : geese = chickens):
  ducks + 36 = chickens + geese :=
by
  sorry

end fewer_ducks_than_chickens_and_geese_l235_235887


namespace rectangular_diagonal_length_l235_235862

theorem rectangular_diagonal_length (x y z : ℝ) 
  (h_surface_area : 2 * (x * y + y * z + z * x) = 11)
  (h_edge_sum : x + y + z = 6) :
  Real.sqrt (x^2 + y^2 + z^2) = 5 := 
by
  sorry

end rectangular_diagonal_length_l235_235862


namespace symbols_in_P_l235_235498
-- Importing the necessary library

-- Define the context P and the operations
def context_P : Type := sorry

def mul_op (P : context_P) : String := "*"
def div_op (P : context_P) : String := "/"
def exp_op (P : context_P) : String := "∧"
def sqrt_op (P : context_P) : String := "SQR"
def abs_op (P : context_P) : String := "ABS"

-- Define what each symbol represents in the context of P
theorem symbols_in_P (P : context_P) :
  (mul_op P = "*") ∧
  (div_op P = "/") ∧
  (exp_op P = "∧") ∧
  (sqrt_op P = "SQR") ∧
  (abs_op P = "ABS") := 
sorry

end symbols_in_P_l235_235498


namespace existence_of_function_values_around_k_l235_235188

-- Define the function f(n, m) with the given properties
def is_valid_function (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f (n, m) = (f (n-1, m) + f (n+1, m) + f (n, m-1) + f (n, m+1)) / 4

-- Theorem to prove the existence of such a function
theorem existence_of_function :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f :=
sorry

-- Theorem to prove that for any k in ℤ, f(n, m) has values both greater and less than k
theorem values_around_k (k : ℤ) :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f ∧ (∃ n1 m1 n2 m2, f (n1, m1) > k ∧ f (n2, m2) < k) :=
sorry

end existence_of_function_values_around_k_l235_235188


namespace number_of_ways_to_choose_one_book_l235_235583

theorem number_of_ways_to_choose_one_book:
  let chinese_books := 10
  let english_books := 7
  let mathematics_books := 5
  chinese_books + english_books + mathematics_books = 22 := by
    -- The actual proof should go here.
    sorry

end number_of_ways_to_choose_one_book_l235_235583


namespace vector_addition_proof_l235_235353

def u : ℝ × ℝ × ℝ := (-3, 2, 5)
def v : ℝ × ℝ × ℝ := (4, -7, 1)
def result : ℝ × ℝ × ℝ := (-2, -3, 11)

theorem vector_addition_proof : (2 • u + v) = result := by
  sorry

end vector_addition_proof_l235_235353


namespace addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l235_235954

section OperationsAlwaysYieldOdd

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem addition_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a + b) :=
sorry

theorem subtraction_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a - b) :=
sorry

theorem squared_sum_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd ((a + b) * (a + b)) :=
sorry

end OperationsAlwaysYieldOdd

end addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l235_235954


namespace pentagon_angle_E_l235_235787

theorem pentagon_angle_E 
    (A B C D E : Type)
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
    (AB BC CD DE : ℝ)
    (angle_B angle_C angle_D : ℝ)
    (h1 : AB = BC)
    (h2 : BC = CD)
    (h3 : CD = DE)
    (h4 : angle_B = 96)
    (h5 : angle_C = 108)
    (h6 : angle_D = 108) :
    ∃ angle_E : ℝ, angle_E = 102 := 
by
  sorry

end pentagon_angle_E_l235_235787


namespace cricket_overs_played_initially_l235_235506

variables (x y : ℝ)

theorem cricket_overs_played_initially 
  (h1 : y = 3.2 * x)
  (h2 : 262 - y = 5.75 * 40) : 
  x = 10 := 
sorry

end cricket_overs_played_initially_l235_235506


namespace center_of_circle_l235_235368

theorem center_of_circle (x y : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 - 6 * y = 12) → ((x + 2)^2 + (y - 3)^2 = 25) :=
by
  sorry

end center_of_circle_l235_235368


namespace binary_arithmetic_l235_235731

theorem binary_arithmetic :
  (110010:ℕ) * (1100:ℕ) / (100:ℕ) / (10:ℕ) = 100100 :=
by sorry

end binary_arithmetic_l235_235731


namespace find_q_l235_235058

theorem find_q (q : ℤ) (h1 : lcm (lcm 12 16) (lcm 18 q) = 144) : q = 1 := sorry

end find_q_l235_235058


namespace sphere_radius_equals_three_l235_235246

noncomputable def radius_of_sphere : ℝ := 3

theorem sphere_radius_equals_three {R : ℝ} (h1 : 4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) : 
  R = radius_of_sphere :=
by
  sorry

end sphere_radius_equals_three_l235_235246


namespace right_obtuse_triangle_impossible_l235_235435

def triangle_interior_angles_sum (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def is_right_angle (α : ℝ) : Prop :=
  α = 90

def is_obtuse_angle (α : ℝ) : Prop :=
  α > 90

theorem right_obtuse_triangle_impossible (α β γ : ℝ) (h1 : triangle_interior_angles_sum α β γ) (h2 : is_right_angle α) (h3 : is_obtuse_angle β) : false :=
  sorry

end right_obtuse_triangle_impossible_l235_235435


namespace op_evaluation_l235_235466

-- Define the custom operation ⊕
def op (a b c : ℝ) : ℝ := b^2 - 3 * a * c

-- Statement of the theorem we want to prove
theorem op_evaluation : op 2 3 4 = -15 :=
by 
  -- This is a placeholder for the actual proof,
  -- which in a real scenario would involve computing the operation.
  sorry

end op_evaluation_l235_235466


namespace intersection_empty_l235_235806

noncomputable def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
noncomputable def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem intersection_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_empty_l235_235806


namespace amount_of_water_in_first_tank_l235_235581

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

end amount_of_water_in_first_tank_l235_235581


namespace total_numbers_l235_235277

theorem total_numbers (N : ℕ) (sum_total : ℝ) (avg_total : ℝ) (avg1 : ℝ) (avg2 : ℝ) (avg3 : ℝ) :
  avg_total = 6.40 → avg1 = 6.2 → avg2 = 6.1 → avg3 = 6.9 →
  sum_total = 2 * avg1 + 2 * avg2 + 2 * avg3 →
  N = sum_total / avg_total →
  N = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_numbers_l235_235277


namespace usual_time_proof_l235_235810

noncomputable 
def usual_time (P T : ℝ) := (P * T) / (100 - P)

theorem usual_time_proof (P T U : ℝ) (h1 : P > 0) (h2 : P < 100) (h3 : T > 0) (h4 : U = usual_time P T) : U = (P * T) / (100 - P) :=
by
    sorry

end usual_time_proof_l235_235810


namespace least_actual_square_area_l235_235969

theorem least_actual_square_area :
  let side_measured := 7
  let lower_bound := 6.5
  let actual_area := lower_bound * lower_bound
  actual_area = 42.25 :=
by
  sorry

end least_actual_square_area_l235_235969


namespace toy_cost_price_l235_235995

theorem toy_cost_price (x : ℝ) (h : 1.5 * x * 0.8 - x = 20) : x = 100 := 
sorry

end toy_cost_price_l235_235995


namespace determine_ratio_l235_235880

def p (x : ℝ) : ℝ := (x - 4) * (x + 3)
def q (x : ℝ) : ℝ := (x - 4) * (x + 3)

theorem determine_ratio : q 1 ≠ 0 ∧ p 1 / q 1 = 1 := by
  have hq : q 1 ≠ 0 := by
    simp [q]
    norm_num
  have hpq : p 1 / q 1 = 1 := by
    simp [p, q]
    norm_num
  exact ⟨hq, hpq⟩

end determine_ratio_l235_235880


namespace equal_number_of_frogs_after_6_months_l235_235687

theorem equal_number_of_frogs_after_6_months :
  ∃ n : ℕ, 
    n = 6 ∧ 
    (∀ Dn Qn : ℕ, 
      (Dn = 5^(n + 1) ∧ Qn = 3^(n + 5)) → 
      Dn = Qn) :=
by
  sorry

end equal_number_of_frogs_after_6_months_l235_235687


namespace sqrt_sum_bounds_l235_235046

theorem sqrt_sum_bounds (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
    4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2 - b)^2) + 
                   Real.sqrt (b^2 + (2 - c)^2) + 
                   Real.sqrt (c^2 + (2 - d)^2) + 
                   Real.sqrt (d^2 + (2 - a)^2) ∧
    Real.sqrt (a^2 + (2 - b)^2) + 
    Real.sqrt (b^2 + (2 - c)^2) + 
    Real.sqrt (c^2 + (2 - d)^2) + 
    Real.sqrt (d^2 + (2 - a)^2) ≤ 8 :=
sorry

end sqrt_sum_bounds_l235_235046


namespace roots_of_quadratic_identity_l235_235524

namespace RootProperties

theorem roots_of_quadratic_identity (a b : ℝ) 
(h1 : a^2 - 2*a - 1 = 0) 
(h2 : b^2 - 2*b - 1 = 0) 
(h3 : a ≠ b) 
: a^2 + b^2 = 6 := 
by sorry

end RootProperties

end roots_of_quadratic_identity_l235_235524


namespace ball_reaches_height_less_than_2_after_6_bounces_l235_235040

theorem ball_reaches_height_less_than_2_after_6_bounces :
  ∃ (k : ℕ), 16 * (2/3) ^ k < 2 ∧ ∀ (m : ℕ), m < k → 16 * (2/3) ^ m ≥ 2 :=
by
  sorry

end ball_reaches_height_less_than_2_after_6_bounces_l235_235040


namespace minimum_number_of_gloves_l235_235421

theorem minimum_number_of_gloves (participants : ℕ) (gloves_per_participant : ℕ) (total_participants : participants = 63) (each_participant_needs_2_gloves : gloves_per_participant = 2) : 
  participants * gloves_per_participant = 126 :=
by
  rcases participants, gloves_per_participant, total_participants, each_participant_needs_2_gloves
  -- sorry to skip the proof
  sorry

end minimum_number_of_gloves_l235_235421


namespace area_of_gray_region_l235_235005

def center_C : ℝ × ℝ := (4, 6)
def radius_C : ℝ := 6
def center_D : ℝ × ℝ := (14, 6)
def radius_D : ℝ := 6

theorem area_of_gray_region :
  let area_of_rectangle := (14 - 4) * 6
  let quarter_circle_area := (π * 6 ^ 2) / 4
  let area_to_subtract := 2 * quarter_circle_area
  area_of_rectangle - area_to_subtract = 60 - 18 * π := 
by {
  sorry
}

end area_of_gray_region_l235_235005


namespace algebra_expression_value_l235_235115

theorem algebra_expression_value (a b : ℝ) (h : a - 2 * b = -1) : 1 - 2 * a + 4 * b = 3 :=
by
  sorry

end algebra_expression_value_l235_235115


namespace jackson_earnings_l235_235087

def hourly_rate_usd : ℝ := 5
def hourly_rate_gbp : ℝ := 3
def hourly_rate_jpy : ℝ := 400

def hours_vacuuming : ℝ := 2
def sessions_vacuuming : ℝ := 2

def hours_washing_dishes : ℝ := 0.5
def hours_cleaning_bathroom := hours_washing_dishes * 3

def exchange_rate_gbp_to_usd : ℝ := 1.35
def exchange_rate_jpy_to_usd : ℝ := 0.009

def earnings_in_usd : ℝ := (hours_vacuuming * sessions_vacuuming * hourly_rate_usd)
def earnings_in_gbp : ℝ := (hours_washing_dishes * hourly_rate_gbp)
def earnings_in_jpy : ℝ := (hours_cleaning_bathroom * hourly_rate_jpy)

def converted_gbp_to_usd : ℝ := earnings_in_gbp * exchange_rate_gbp_to_usd
def converted_jpy_to_usd : ℝ := earnings_in_jpy * exchange_rate_jpy_to_usd

def total_earnings_usd : ℝ := earnings_in_usd + converted_gbp_to_usd + converted_jpy_to_usd

theorem jackson_earnings : total_earnings_usd = 27.425 := by
  sorry

end jackson_earnings_l235_235087


namespace max_reflections_l235_235477

theorem max_reflections (A B D : Point) (n : ℕ) (angle_CDA : ℝ) (incident_angle : ℕ → ℝ)
  (h1 : angle_CDA = 12)
  (h2 : ∀ k : ℕ, k ≤ n → incident_angle k = k * angle_CDA)
  (h3 : incident_angle n = 90) :
  n = 7 := 
sorry

end max_reflections_l235_235477


namespace find_y_l235_235828

theorem find_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 := by
  sorry

end find_y_l235_235828


namespace arithmetic_sequence_y_l235_235197

theorem arithmetic_sequence_y :
  let a := 3^3
  let c := 3^5
  let y := (a + c) / 2
  y = 135 :=
by
  let a := 27
  let c := 243
  let y := (a + c) / 2
  show y = 135
  sorry

end arithmetic_sequence_y_l235_235197


namespace anne_distance_l235_235267
  
theorem anne_distance (S T : ℕ) (H1 : S = 2) (H2 : T = 3) : S * T = 6 := by
  -- Given that speed S = 2 miles/hour and time T = 3 hours, we need to show the distance S * T = 6 miles.
  sorry

end anne_distance_l235_235267


namespace exponentiation_division_l235_235181

variable {a : ℝ} (h1 : (a^2)^3 = a^6) (h2 : a^6 / a^2 = a^4)

theorem exponentiation_division : (a^2)^3 / a^2 = a^4 := 
by 
  sorry

end exponentiation_division_l235_235181


namespace complex_number_properties_l235_235426

open Complex

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Given conditions in Lean: \( z \) satisfies \( z(2+i) = i^{10} \)
def satisfies_condition (z : ℂ) : Prop :=
  z * (2 + i) = i^10

-- Theorem stating the required proofs
theorem complex_number_properties (z : ℂ) (hc : satisfies_condition z) :
  Complex.abs z = Real.sqrt 5 / 5 ∧ 
  (z.re < 0 ∧ z.im > 0) := by
  -- Placeholders for the proof steps
  sorry

end complex_number_properties_l235_235426


namespace cost_of_paint_per_kg_l235_235108

/-- The cost of painting one square foot is Rs. 50. -/
theorem cost_of_paint_per_kg (side_length : ℝ) (cost_total : ℝ) (coverage_per_kg : ℝ) (total_surface_area : ℝ) (total_paint_needed : ℝ) (cost_per_kg : ℝ) 
  (h1 : side_length = 20)
  (h2 : cost_total = 6000)
  (h3 : coverage_per_kg = 20)
  (h4 : total_surface_area = 6 * side_length^2)
  (h5 : total_paint_needed = total_surface_area / coverage_per_kg)
  (h6 : cost_per_kg = cost_total / total_paint_needed) :
  cost_per_kg = 50 :=
sorry

end cost_of_paint_per_kg_l235_235108


namespace pages_removed_iff_original_pages_l235_235800

def booklet_sum (n r : ℕ) : ℕ :=
  (n * (2 * n + 1)) - (4 * r - 1)

theorem pages_removed_iff_original_pages (n r : ℕ) :
  booklet_sum n r = 963 ↔ (2 * n = 44 ∧ (2 * r - 1, 2 * r) = (13, 14)) :=
sorry

end pages_removed_iff_original_pages_l235_235800


namespace employed_population_percentage_l235_235578

noncomputable def percent_population_employed (total_population employed_males employed_females : ℝ) : ℝ :=
  employed_males + employed_females

theorem employed_population_percentage (population employed_males_percentage employed_females_percentage : ℝ) 
  (h1 : employed_males_percentage = 0.36 * population)
  (h2 : employed_females_percentage = 0.36 * population)
  (h3 : employed_females_percentage + employed_males_percentage = 0.50 * total_population)
  : total_population = 0.72 * population :=
by 
  sorry

end employed_population_percentage_l235_235578


namespace problem_l235_235052

noncomputable def f (x : ℝ) (a b : ℝ) := (b - 2^x) / (2^(x+1) + a)

theorem problem (a b k : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) →
  (f 0 a b = 0) → (f (-1) a b = -f 1 a b) → 
  a = 2 ∧ b = 1 ∧ 
  (∀ x y : ℝ, x < y → f x a b > f y a b) ∧ 
  (∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0 → k < 4 / 3) :=
by
  sorry

end problem_l235_235052


namespace janets_total_pockets_l235_235668

-- Define the total number of dresses
def totalDresses : ℕ := 36

-- Define the dresses with pockets
def dressesWithPockets : ℕ := totalDresses / 2

-- Define the dresses without pockets
def dressesWithoutPockets : ℕ := totalDresses - dressesWithPockets

-- Define the dresses with one hidden pocket
def dressesWithOneHiddenPocket : ℕ := (40 * dressesWithoutPockets) / 100

-- Define the dresses with 2 pockets
def dressesWithTwoPockets : ℕ := dressesWithPockets / 3

-- Define the dresses with 3 pockets
def dressesWithThreePockets : ℕ := dressesWithPockets / 4

-- Define the dresses with 4 pockets
def dressesWithFourPockets : ℕ := dressesWithPockets - dressesWithTwoPockets - dressesWithThreePockets

-- Calculate the total number of pockets
def totalPockets : ℕ := 
  2 * dressesWithTwoPockets + 
  3 * dressesWithThreePockets + 
  4 * dressesWithFourPockets + 
  dressesWithOneHiddenPocket

-- The theorem to prove the total number of pockets
theorem janets_total_pockets : totalPockets = 63 :=
  by
    -- Proof is omitted, use 'sorry'
    sorry

end janets_total_pockets_l235_235668


namespace coloring_even_conditional_l235_235158

-- Define the problem parameters and constraints
def number_of_colorings (n : Nat) (even_red : Bool) (even_yellow : Bool) : Nat :=
  sorry  -- This function would contain the detailed computational logic.

-- Define the main theorem statement
theorem coloring_even_conditional (n : ℕ) (h1 : n > 0) : ∃ C : Nat, number_of_colorings n true true = C := 
by
  sorry  -- The proof would go here.


end coloring_even_conditional_l235_235158


namespace largest_whole_number_l235_235011

theorem largest_whole_number :
  ∃ x : ℕ, 9 * x - 8 < 130 ∧ (∀ y : ℕ, 9 * y - 8 < 130 → y ≤ x) ∧ x = 15 :=
sorry

end largest_whole_number_l235_235011


namespace dogs_with_pointy_ears_l235_235604

theorem dogs_with_pointy_ears (total_dogs with_spots with_pointy_ears: ℕ) 
  (h1: with_spots = total_dogs / 2)
  (h2: total_dogs = 30) :
  with_pointy_ears = total_dogs / 5 :=
by
  sorry

end dogs_with_pointy_ears_l235_235604


namespace total_crayons_lost_or_given_away_l235_235804

def crayons_given_away : ℕ := 52
def crayons_lost : ℕ := 535

theorem total_crayons_lost_or_given_away :
  crayons_given_away + crayons_lost = 587 :=
by
  sorry

end total_crayons_lost_or_given_away_l235_235804


namespace sum_of_roots_is_k_over_5_l235_235125

noncomputable def sum_of_roots 
  (x1 x2 k d : ℝ) 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : ℝ :=
x1 + x2

theorem sum_of_roots_is_k_over_5 
  {x1 x2 k d : ℝ} 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : 
  sum_of_roots x1 x2 k d hx h1 h2 = k / 5 :=
sorry

end sum_of_roots_is_k_over_5_l235_235125


namespace mitch_total_scoops_l235_235911

theorem mitch_total_scoops :
  (3 : ℝ) / (1/3 : ℝ) + (2 : ℝ) / (1/3 : ℝ) = 15 :=
by
  sorry

end mitch_total_scoops_l235_235911


namespace log_sum_eq_two_l235_235489

theorem log_sum_eq_two (log6_3 log6_4 : ℝ) (H1 : Real.logb 6 3 = log6_3) (H2 : Real.logb 6 4 = log6_4) : 
  log6_3 + log6_4 = 2 := 
by 
  sorry

end log_sum_eq_two_l235_235489


namespace amanda_quizzes_l235_235571

theorem amanda_quizzes (n : ℕ) (h1 : n > 0) (h2 : 92 * n + 97 = 93 * 5) : n = 4 :=
by
  sorry

end amanda_quizzes_l235_235571


namespace bridge_length_correct_l235_235619

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 72
noncomputable def crossing_time : ℝ := 12.399008079353651

-- converting train speed from km/hr to m/s
noncomputable def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- total length the train covers to cross the bridge
noncomputable def total_length : ℝ := train_speed_m_per_s * crossing_time

-- length of the bridge
noncomputable def bridge_length : ℝ := total_length - train_length

theorem bridge_length_correct :
  bridge_length = 137.98 :=
by 
  sorry

end bridge_length_correct_l235_235619


namespace nonnegative_integers_existence_l235_235955

open Classical

theorem nonnegative_integers_existence (x y : ℕ) : 
  (∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d) ↔ (5 * x ≥ 7 * y) :=
by
  sorry

end nonnegative_integers_existence_l235_235955


namespace reading_days_l235_235637

theorem reading_days (total_pages pages_per_day_1 pages_per_day_2 : ℕ ) :
  total_pages = 525 →
  pages_per_day_1 = 25 →
  pages_per_day_2 = 21 →
  (total_pages / pages_per_day_1 = 21) ∧ (total_pages / pages_per_day_2 = 25) :=
by
  sorry

end reading_days_l235_235637


namespace trees_died_proof_l235_235088

def treesDied (original : Nat) (remaining : Nat) : Nat := original - remaining

theorem trees_died_proof : treesDied 20 4 = 16 := by
  -- Here we put the steps needed to prove the theorem, which is essentially 20 - 4 = 16.
  sorry

end trees_died_proof_l235_235088


namespace smallest_root_of_quadratic_l235_235855

theorem smallest_root_of_quadratic (y : ℝ) (h : 4 * y^2 - 7 * y + 3 = 0) : y = 3 / 4 :=
sorry

end smallest_root_of_quadratic_l235_235855


namespace spike_crickets_hunted_morning_l235_235306

def crickets_hunted_in_morning (C : ℕ) (total_daily_crickets : ℕ) : Prop :=
  4 * C = total_daily_crickets

theorem spike_crickets_hunted_morning (C : ℕ) (total_daily_crickets : ℕ) :
  total_daily_crickets = 20 → crickets_hunted_in_morning C total_daily_crickets → C = 5 :=
by
  intros h1 h2
  sorry

end spike_crickets_hunted_morning_l235_235306


namespace math_problem_l235_235728

variables {a b : ℝ}
open Real

theorem math_problem (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (a - 1) * (b - 1) = 1 ∧ 
  (∀ b : ℝ, (a = 2 * b → a + 4 * b = 9)) ∧ 
  (∀ b : ℝ, (b = 3 → (1 / a^2 + 2 / b^2) = 2 / 3)) :=
by
  sorry

end math_problem_l235_235728


namespace find_y_payment_l235_235624

-- Defining the conditions
def total_payment : ℝ := 700
def x_payment (y_payment : ℝ) : ℝ := 1.2 * y_payment

-- The theorem we want to prove
theorem find_y_payment (y_payment : ℝ) (h1 : y_payment + x_payment y_payment = total_payment) :
  y_payment = 318.18 := 
sorry

end find_y_payment_l235_235624


namespace final_price_wednesday_l235_235920

theorem final_price_wednesday :
  let coffee_price := 6
  let cheesecake_price := 10
  let sandwich_price := 8
  let coffee_discount := 0.25
  let cheesecake_discount_wednesday := 0.10
  let additional_discount := 3
  let sales_tax := 0.05
  let discounted_coffee_price := coffee_price - coffee_price * coffee_discount
  let discounted_cheesecake_price := cheesecake_price - cheesecake_price * cheesecake_discount_wednesday
  let total_price_before_additional_discount := discounted_coffee_price + discounted_cheesecake_price + sandwich_price
  let total_price_after_additional_discount := total_price_before_additional_discount - additional_discount
  let total_price_with_tax := total_price_after_additional_discount + total_price_after_additional_discount * sales_tax
  let final_price := total_price_with_tax.round
  final_price = 19.43 :=
by
  sorry

end final_price_wednesday_l235_235920


namespace polynomial_roots_l235_235299

theorem polynomial_roots (α : ℝ) : 
  (α^2 + α - 1 = 0) → (α^3 - 2 * α + 1 = 0) :=
by sorry

end polynomial_roots_l235_235299


namespace algebraic_expression_value_l235_235744

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) : 
  x^2 - 4 * y^2 = -4 :=
by
  sorry

end algebraic_expression_value_l235_235744


namespace election_votes_l235_235454

variable (V : ℝ)

theorem election_votes (h1 : 0.70 * V - 0.30 * V = 192) : V = 480 :=
by
  sorry

end election_votes_l235_235454


namespace ab_product_power_l235_235209

theorem ab_product_power (a b : ℤ) (n : ℕ) (h1 : (a * b)^n = 128 * 8) : n = 10 := by
  sorry

end ab_product_power_l235_235209


namespace sum_of_denominators_of_fractions_l235_235813

theorem sum_of_denominators_of_fractions {a b : ℕ} (ha : 3 * a / 5 * b + 2 * a / 9 * b + 4 * a / 15 * b = 28 / 45) (gcd_ab : Nat.gcd a b = 1) :
  5 * b + 9 * b + 15 * b = 203 := sorry

end sum_of_denominators_of_fractions_l235_235813


namespace distance_between_P_and_F2_l235_235539
open Real

theorem distance_between_P_and_F2 (x y c : ℝ) (h1 : c = sqrt 3)
    (h2 : x = -sqrt 3) (h3 : y = 1/2) : 
    sqrt ((sqrt 3 - x) ^ 2 + (0 - y) ^ 2) = 7 / 2 :=
by
  sorry

end distance_between_P_and_F2_l235_235539


namespace value_before_decrease_l235_235147

theorem value_before_decrease
  (current_value decrease : ℤ)
  (current_value_equals : current_value = 1460)
  (decrease_equals : decrease = 12) :
  current_value + decrease = 1472 :=
by
  -- We assume the proof to follow here.
  sorry

end value_before_decrease_l235_235147


namespace journal_sessions_per_week_l235_235201

/-- Given that each student writes 4 pages in each session and will write 72 journal pages in 6 weeks, prove that there are 3 journal-writing sessions per week.
--/
theorem journal_sessions_per_week (pages_per_session : ℕ) (total_pages : ℕ) (weeks : ℕ) (sessions_per_week : ℕ) :
  pages_per_session = 4 →
  total_pages = 72 →
  weeks = 6 →
  total_pages = pages_per_session * sessions_per_week * weeks →
  sessions_per_week = 3 :=
by
  intros h1 h2 h3 h4
  sorry

end journal_sessions_per_week_l235_235201


namespace algebraic_expression_value_l235_235493

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2) : a^2 - b^2 - 4*a = -4 := 
sorry

end algebraic_expression_value_l235_235493


namespace correct_answers_count_l235_235932

theorem correct_answers_count
  (c w : ℕ)
  (h1 : c + w = 150)
  (h2 : 4 * c - 2 * w = 420) :
  c = 120 := by
  sorry

end correct_answers_count_l235_235932


namespace frankie_candies_l235_235671

theorem frankie_candies (M D F : ℕ) (h1 : M = 92) (h2 : D = 18) (h3 : F = M - D) : F = 74 :=
by
  sorry

end frankie_candies_l235_235671


namespace projected_revenue_increase_is_20_percent_l235_235278

noncomputable def projected_percentage_increase_of_revenue (R : ℝ) (actual_revenue : ℝ) (projected_revenue : ℝ) : ℝ :=
  (projected_revenue / R - 1) * 100

theorem projected_revenue_increase_is_20_percent (R : ℝ) (actual_revenue : ℝ) :
  actual_revenue = R * 0.75 →
  actual_revenue = (R * (1 + 20 / 100)) * 0.625 →
  projected_percentage_increase_of_revenue R ((R * (1 + 20 / 100))) = 20 :=
by
  intros h1 h2
  sorry

end projected_revenue_increase_is_20_percent_l235_235278


namespace min_tables_42_l235_235162

def min_tables_needed (total_people : ℕ) (table_sizes : List ℕ) : ℕ :=
  sorry

theorem min_tables_42 :
  min_tables_needed 42 [4, 6, 8] = 6 :=
sorry

end min_tables_42_l235_235162


namespace difference_in_circumferences_l235_235395

def r_inner : ℝ := 25
def r_outer : ℝ := r_inner + 15

theorem difference_in_circumferences : 2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi := by
  sorry

end difference_in_circumferences_l235_235395


namespace triangle_side_count_l235_235942

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end triangle_side_count_l235_235942


namespace max_area_rect_l235_235257

noncomputable def maximize_area (l w : ℕ) : ℕ :=
  l * w

theorem max_area_rect (l w: ℕ) (hl_even : l % 2 = 0) (h_perim : 2*l + 2*w = 40) :
  maximize_area l w = 100 :=
by
  sorry 

end max_area_rect_l235_235257


namespace solve_equation_l235_235791

theorem solve_equation (x : ℕ) (h : x = 88320) : x + 1315 + 9211 - 1569 = 97277 :=
by sorry

end solve_equation_l235_235791


namespace solve_for_vee_l235_235848

theorem solve_for_vee (vee : ℝ) (h : 4 * vee ^ 2 = 144) : vee = 6 ∨ vee = -6 :=
by
  -- We state that this theorem should be true for all vee and given the condition h
  sorry

end solve_for_vee_l235_235848


namespace find_new_songs_l235_235982

-- Definitions for the conditions
def initial_songs : ℕ := 6
def deleted_songs : ℕ := 3
def final_songs : ℕ := 23

-- The number of new songs added
def new_songs_added : ℕ := 20

-- Statement of the proof problem
theorem find_new_songs (n d f x : ℕ) (h1 : n = initial_songs) (h2 : d = deleted_songs) (h3 : f = final_songs) : f = n - d + x → x = new_songs_added :=
by
  intros h4
  sorry

end find_new_songs_l235_235982


namespace runner_injury_point_l235_235937

theorem runner_injury_point
  (v d : ℝ)
  (h1 : 2 * (40 - d) / v = d / v + 11)
  (h2 : 2 * (40 - d) / v = 22) :
  d = 20 := 
by
  sorry

end runner_injury_point_l235_235937


namespace johns_final_push_time_l235_235400

-- Definitions and initial conditions.
def john_initial_distance_behind_steve : ℝ := 12
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.7
def john_final_distance_ahead_of_steve : ℝ := 2

-- The statement we want to prove:
theorem johns_final_push_time : ∃ t : ℝ, john_speed * t = steve_speed * t + john_initial_distance_behind_steve + john_final_distance_ahead_of_steve ∧ t = 28 := 
by 
  -- Adding blank proof body
  sorry

end johns_final_push_time_l235_235400


namespace area_of_parallelogram_l235_235268

theorem area_of_parallelogram (b h : ℕ) (hb : b = 60) (hh : h = 16) : b * h = 960 := by
  -- Here goes the proof
  sorry

end area_of_parallelogram_l235_235268


namespace perfect_square_trinomial_m_eq_l235_235327

theorem perfect_square_trinomial_m_eq (
    m y : ℝ) (h : ∃ k : ℝ, 4*y^2 - m*y + 25 = (2*y - k)^2) :
  m = 20 ∨ m = -20 :=
by
  sorry

end perfect_square_trinomial_m_eq_l235_235327


namespace theta_plus_2phi_l235_235361

theorem theta_plus_2phi (θ φ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hφ : 0 < φ ∧ φ < π / 2)
  (h_tan_θ : Real.tan θ = 1 / 7) (h_sin_φ : Real.sin φ = 1 / Real.sqrt 10) :
  θ + 2 * φ = π / 4 := 
sorry

end theta_plus_2phi_l235_235361


namespace strawb_eaten_by_friends_l235_235617

theorem strawb_eaten_by_friends (initial_strawberries remaining_strawberries eaten_strawberries : ℕ) : 
  initial_strawberries = 35 → 
  remaining_strawberries = 33 → 
  eaten_strawberries = initial_strawberries - remaining_strawberries → 
  eaten_strawberries = 2 := 
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end strawb_eaten_by_friends_l235_235617


namespace forgotten_angles_correct_l235_235832

theorem forgotten_angles_correct (n : ℕ) (h1 : (n - 2) * 180 = 2520) (h2 : 2345 + 175 = 2520) : 
  ∃ a b : ℕ, a + b = 175 :=
by
  sorry

end forgotten_angles_correct_l235_235832


namespace children_absent_l235_235745

theorem children_absent (A : ℕ) (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ) :
  total_children = 660 →
  bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children * bananas_per_child) = 1320 →
  ((total_children - A) * (bananas_per_child + extra_bananas_per_child)) = 1320 →
  A = 330 :=
by
  intros
  sorry

end children_absent_l235_235745


namespace jake_has_more_balloons_l235_235148

-- Defining the given conditions as parameters
def initial_balloons_allan : ℕ := 2
def initial_balloons_jake : ℕ := 6
def additional_balloons_allan : ℕ := 3

-- Calculate total balloons each person has
def total_balloons_allan : ℕ := initial_balloons_allan + additional_balloons_allan
def total_balloons_jake : ℕ := initial_balloons_jake

-- Formalize the statement to be proved
theorem jake_has_more_balloons :
  total_balloons_jake - total_balloons_allan = 1 :=
by
  -- Proof will be added here
  sorry

end jake_has_more_balloons_l235_235148


namespace simplify_expression_l235_235622

theorem simplify_expression (x : ℝ) : 5 * x + 2 * x + 7 * x = 14 * x :=
by
  sorry

end simplify_expression_l235_235622


namespace count_boys_correct_l235_235636

def total_vans : ℕ := 5
def students_per_van : ℕ := 28
def number_of_girls : ℕ := 80

theorem count_boys_correct : 
  (total_vans * students_per_van) - number_of_girls = 60 := 
by
  sorry

end count_boys_correct_l235_235636


namespace intersection_M_N_l235_235287

def M := { x : ℝ | |x| ≤ 1 }
def N := { x : ℝ | x^2 - x < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l235_235287


namespace length_of_common_chord_l235_235198

-- Problem conditions
variables (r : ℝ) (h : r = 15)

-- Statement to prove
theorem length_of_common_chord : 2 * (r / 2 * Real.sqrt 3) = 15 * Real.sqrt 3 :=
by
  sorry

end length_of_common_chord_l235_235198


namespace oldest_child_age_l235_235304

def arithmeticProgression (a d : ℕ) (n : ℕ) : ℕ := 
  a + (n - 1) * d

theorem oldest_child_age (a : ℕ) (d : ℕ) (n : ℕ) 
  (average : (arithmeticProgression a d 1 + arithmeticProgression a d 2 + arithmeticProgression a d 3 + arithmeticProgression a d 4 + arithmeticProgression a d 5) / 5 = 10)
  (distinct : ∀ i j, i ≠ j → arithmeticProgression a d i ≠ arithmeticProgression a d j)
  (constant_difference : d = 3) :
  arithmeticProgression a d 5 = 16 :=
by
  sorry

end oldest_child_age_l235_235304


namespace distribute_books_l235_235131

-- Definition of books and people
def num_books : Nat := 2
def num_people : Nat := 10

-- The main theorem statement that we need to prove.
theorem distribute_books : (num_people ^ num_books) = 100 :=
by
  -- Proof body
  sorry

end distribute_books_l235_235131


namespace profit_equations_l235_235364

-- Define the conditions
def total_workers : ℕ := 150
def fabric_per_worker_per_day : ℕ := 30
def clothing_per_worker_per_day : ℕ := 4
def fabric_needed_per_clothing : ℝ := 1.5
def profit_per_meter : ℝ := 2
def profit_per_clothing : ℝ := 25

-- Define the profit functions
def profit_clothing (x : ℕ) : ℝ := profit_per_clothing * clothing_per_worker_per_day * x
def profit_fabric (x : ℕ) : ℝ := profit_per_meter * (fabric_per_worker_per_day * (total_workers - x) - fabric_needed_per_clothing * clothing_per_worker_per_day * x)

-- Define the total profit function
def total_profit (x : ℕ) : ℝ := profit_clothing x + profit_fabric x

-- Prove the given statements
theorem profit_equations (x : ℕ) :
  profit_clothing x = 100 * x ∧
  profit_fabric x = 9000 - 72 * x ∧
  total_profit 100 = 11800 :=
by
  -- Proof omitted
  sorry

end profit_equations_l235_235364


namespace min_sum_of_positive_real_solution_l235_235839

theorem min_sum_of_positive_real_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y = 6 := 
by {
  sorry
}

end min_sum_of_positive_real_solution_l235_235839


namespace farmer_apples_after_giving_away_l235_235512

def initial_apples : ℕ := 127
def given_away_apples : ℕ := 88
def remaining_apples : ℕ := 127 - 88

theorem farmer_apples_after_giving_away : remaining_apples = 39 := by
  sorry

end farmer_apples_after_giving_away_l235_235512


namespace radius_of_surrounding_circles_is_correct_l235_235693

noncomputable def r : Real := 1 + Real.sqrt 2

theorem radius_of_surrounding_circles_is_correct (r: ℝ)
  (h₁: ∃c : ℝ, c = 2) -- central circle radius is 2
  (h₂: ∃far: ℝ, far = (1 + (Real.sqrt 2))) -- r is the solution as calculated
: 2 * r = 1 + Real.sqrt 2 :=
by
  sorry

end radius_of_surrounding_circles_is_correct_l235_235693


namespace total_cost_of_books_l235_235055

def book_cost (num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook : ℕ) : ℕ :=
  (num_mathbooks * cost_mathbook) + (num_artbooks * cost_artbook) + (num_sciencebooks * cost_sciencebook)

theorem total_cost_of_books :
  let num_mathbooks := 2
  let num_artbooks := 3
  let num_sciencebooks := 6
  let cost_mathbook := 3
  let cost_artbook := 2
  let cost_sciencebook := 3
  book_cost num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook = 30 :=
by
  sorry

end total_cost_of_books_l235_235055


namespace marvelous_class_student_count_l235_235215

theorem marvelous_class_student_count (g : ℕ) (jb : ℕ) (jg : ℕ) (j_total : ℕ) (jl : ℕ) (init_jb : ℕ) : 
  jb = g + 3 →  -- Number of boys
  jg = 2 * g + 1 →  -- Jelly beans received by each girl
  init_jb = 726 →  -- Initial jelly beans
  jl = 4 →  -- Leftover jelly beans
  j_total = init_jb - jl →  -- Jelly beans distributed
  (jb * jb + g * jg = j_total) → -- Total jelly beans distributed equation
  2 * g + 1 + g + jb = 31 := -- Total number of students
by
  sorry

end marvelous_class_student_count_l235_235215


namespace find_somus_age_l235_235531

def somus_current_age (S F : ℕ) := S = F / 3
def somus_age_7_years_ago (S F : ℕ) := (S - 7) = (F - 7) / 5

theorem find_somus_age (S F : ℕ) 
  (h1 : somus_current_age S F) 
  (h2 : somus_age_7_years_ago S F) : S = 14 :=
sorry

end find_somus_age_l235_235531


namespace sum_of_powers_modulo_l235_235735

theorem sum_of_powers_modulo (R : Finset ℕ) (S : ℕ) :
  (∀ n < 100, ∃ r, r ∈ R ∧ r = 3^n % 500) →
  S = R.sum id →
  (S % 500) = 0 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_powers_modulo_l235_235735


namespace number_solution_l235_235563

-- Statement based on identified conditions and answer
theorem number_solution (x : ℝ) (h : 0.10 * 0.30 * 0.50 * x = 90) : x = 6000 :=
by
  -- Skip the proof
  sorry

end number_solution_l235_235563


namespace bike_distance_difference_l235_235807

-- Defining constants for Alex's and Bella's rates and the time duration
def Alex_rate : ℕ := 12
def Bella_rate : ℕ := 10
def time : ℕ := 6

-- The goal is to prove the difference in distance is 12 miles
theorem bike_distance_difference : (Alex_rate * time) - (Bella_rate * time) = 12 := by
  sorry

end bike_distance_difference_l235_235807


namespace red_jellybeans_count_l235_235989

theorem red_jellybeans_count (total_jellybeans : ℕ)
  (blue_jellybeans : ℕ)
  (purple_jellybeans : ℕ)
  (orange_jellybeans : ℕ)
  (H1 : total_jellybeans = 200)
  (H2 : blue_jellybeans = 14)
  (H3 : purple_jellybeans = 26)
  (H4 : orange_jellybeans = 40) :
  total_jellybeans - (blue_jellybeans + purple_jellybeans + orange_jellybeans) = 120 :=
by sorry

end red_jellybeans_count_l235_235989


namespace seashell_count_l235_235283

theorem seashell_count (Sam Mary Lucy : Nat) (h1 : Sam = 18) (h2 : Mary = 47) (h3 : Lucy = 32) : 
  Sam + Mary + Lucy = 97 :=
by 
  sorry

end seashell_count_l235_235283


namespace geometric_sequence_sum_l235_235500

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 1 = 3)
    (h2 : a 4 = 24)
    (hn : ∀ n, a n = a 1 * q ^ (n - 1)) :
    (a 3 + a 4 + a 5 = 84) :=
by
  -- Proof will go here
  sorry

end geometric_sequence_sum_l235_235500


namespace cuboid_dimensions_sum_l235_235064

theorem cuboid_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 45) 
  (h2 : B * C = 80) 
  (h3 : C * A = 180) : 
  A + B + C = 145 / 9 :=
sorry

end cuboid_dimensions_sum_l235_235064


namespace trigonometric_identity_proof_l235_235101

variable (α : ℝ)

theorem trigonometric_identity_proof
  (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos (2 * α) + 2 * Real.sin (2 * α) = 1 :=
by
  sorry

end trigonometric_identity_proof_l235_235101


namespace sum_of_squares_l235_235767

theorem sum_of_squares (x : ℕ) (h : 2 * x = 14) : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862 := 
by 
  sorry

end sum_of_squares_l235_235767


namespace sum_not_complete_residue_system_l235_235551

theorem sum_not_complete_residue_system
  (n : ℕ) (hn : Even n)
  (a b : Fin n → Fin n)
  (ha : ∀ i : Fin n, ∃ j : Fin n, a j = i)
  (hb : ∀ i : Fin n, ∃ j : Fin n, b j = i) :
  ¬ (∀ k : Fin n, ∃ i : Fin n, a i + b i = k) :=
sorry

end sum_not_complete_residue_system_l235_235551


namespace complement_of_M_in_U_l235_235081

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U : (U \ M) = {2, 4, 6} :=
by
  sorry

end complement_of_M_in_U_l235_235081


namespace circle_representation_l235_235743

theorem circle_representation (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 + y^2 + x + 2*m*y + m = 0)) → m ≠ 1/2 :=
by
  sorry

end circle_representation_l235_235743


namespace bottle_caps_per_person_l235_235185

noncomputable def initial_caps : Nat := 150
noncomputable def rebecca_caps : Nat := 42
noncomputable def alex_caps : Nat := 2 * rebecca_caps
noncomputable def total_caps : Nat := initial_caps + rebecca_caps + alex_caps
noncomputable def number_of_people : Nat := 6

theorem bottle_caps_per_person : total_caps / number_of_people = 46 := by
  sorry

end bottle_caps_per_person_l235_235185


namespace complex_fraction_evaluation_l235_235309

open Complex

theorem complex_fraction_evaluation (c d : ℂ) (hz : c ≠ 0) (hz' : d ≠ 0) (h : c^2 + c * d + d^2 = 0) :
  (c^12 + d^12) / (c^3 + d^3)^4 = 1 / 8 := 
by sorry

end complex_fraction_evaluation_l235_235309


namespace remainder_when_divided_by_20_l235_235997

theorem remainder_when_divided_by_20 (n : ℕ) : (4 * 6^n + 5^(n-1)) % 20 = 9 := 
by
  sorry

end remainder_when_divided_by_20_l235_235997


namespace total_area_correct_l235_235872

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase : ℕ := 2
def num_extra_rooms_same_size : ℕ := 3
def num_double_size_rooms : ℕ := 1

def increased_length : ℕ := initial_length + increase
def increased_width : ℕ := initial_width + increase

def area_of_one_room (length width : ℕ) : ℕ := length * width

def number_of_rooms : ℕ := 1 + num_extra_rooms_same_size

def total_area_same_size_rooms : ℕ := number_of_rooms * area_of_one_room increased_length increased_width

def total_area_extra_size_room : ℕ := num_double_size_rooms * 2 * area_of_one_room increased_length increased_width

def total_area : ℕ := total_area_same_size_rooms + total_area_extra_size_room

theorem total_area_correct : total_area = 1800 := by
  -- Proof omitted
  sorry

end total_area_correct_l235_235872


namespace no_valid_pairs_l235_235432

theorem no_valid_pairs : ∀ (m n : ℕ), m ≥ n → m^2 - n^2 = 150 → false :=
by sorry

end no_valid_pairs_l235_235432


namespace surface_area_eighth_block_l235_235825

theorem surface_area_eighth_block {A B C D E F G H : ℕ} 
  (blockA : A = 148) 
  (blockB : B = 46) 
  (blockC : C = 72) 
  (blockD : D = 28) 
  (blockE : E = 88) 
  (blockF : F = 126) 
  (blockG : G = 58) 
  : H = 22 :=
by 
  sorry

end surface_area_eighth_block_l235_235825


namespace shaded_region_area_l235_235025

theorem shaded_region_area (ABCD: Type) (D B: Type) (AD CD: ℝ) 
  (h1: (AD = 5)) (h2: (CD = 12)):
  let radiusD := Real.sqrt (AD^2 + CD^2)
  let quarter_circle_area := Real.pi * radiusD^2 / 4
  let radiusC := CD / 2
  let semicircle_area := Real.pi * radiusC^2 / 2
  quarter_circle_area - semicircle_area = 97 * Real.pi / 4 :=
by sorry

end shaded_region_area_l235_235025


namespace intersection_of_sets_l235_235686

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3}) (hB : B = { x | x < 3 ∧ x ∈ Set.univ }) :
  A ∩ B = {0, 1, 2} :=
by
  sorry

end intersection_of_sets_l235_235686


namespace range_of_a_l235_235789

theorem range_of_a (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f x) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f (x - a) + f (x + a)) ↔ -1/2 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l235_235789


namespace max_red_socks_l235_235836

theorem max_red_socks (r b g t : ℕ) (h1 : t ≤ 2500) (h2 : r + b + g = t) 
  (h3 : (r * (r - 1) + b * (b - 1) + g * (g - 1)) = (2 / 3) * t * (t - 1)) : 
  r ≤ 1625 :=
by 
  sorry

end max_red_socks_l235_235836


namespace arc_length_of_sector_l235_235720

theorem arc_length_of_sector (r A l : ℝ) (h_r : r = 2) (h_A : A = π / 3) (h_area : A = 1 / 2 * r * l) : l = π / 3 :=
by
  rw [h_r, h_A] at h_area
  sorry

end arc_length_of_sector_l235_235720


namespace rationalize_denominator_correct_l235_235187

noncomputable def rationalize_denominator : Prop :=
  (50 + Real.sqrt 8) / (Real.sqrt 50 + Real.sqrt 8) = (50 * (Real.sqrt 50 - Real.sqrt 8) + 12) / 42

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l235_235187


namespace days_at_sister_l235_235383

def total_days_vacation : ℕ := 21
def days_plane : ℕ := 2
def days_grandparents : ℕ := 5
def days_train : ℕ := 1
def days_brother : ℕ := 5
def days_car_to_sister : ℕ := 1
def days_bus_to_sister : ℕ := 1
def extra_days_due_to_time_zones : ℕ := 1
def days_bus_back : ℕ := 1
def days_car_back : ℕ := 1

theorem days_at_sister : 
  total_days_vacation - (days_plane + days_grandparents + days_train + days_brother + days_car_to_sister + days_bus_to_sister + extra_days_due_to_time_zones + days_bus_back + days_car_back) = 3 :=
by
  sorry

end days_at_sister_l235_235383


namespace number_of_positive_integers_l235_235795

theorem number_of_positive_integers (n : ℕ) : 
  (0 < n ∧ n < 36 ∧ (∃ k : ℕ, n = k * (36 - k))) → 
  n = 18 ∨ n = 24 ∨ n = 30 ∨ n = 32 ∨ n = 34 ∨ n = 35 :=
sorry

end number_of_positive_integers_l235_235795


namespace math_problem_proof_l235_235914

variable {a : ℝ} (ha : a > 0)

theorem math_problem_proof : ((36 * a^9)^4 * (63 * a^9)^4 = a^(72)) :=
by sorry

end math_problem_proof_l235_235914


namespace clock_equiv_l235_235734

theorem clock_equiv (h : ℕ) (h_gt_6 : h > 6) : h ≡ h^2 [MOD 12] ∧ h ≡ h^3 [MOD 12] → h = 9 :=
by
  sorry

end clock_equiv_l235_235734


namespace gcd_7384_12873_l235_235219

theorem gcd_7384_12873 : Int.gcd 7384 12873 = 1 :=
by
  sorry

end gcd_7384_12873_l235_235219


namespace solve_for_m_l235_235168

theorem solve_for_m 
  (m : ℝ) 
  (h : (m - 1) * 2^2 + 3 * 2 - 5 * m + 4 = 0) : m = 6 :=
sorry

end solve_for_m_l235_235168


namespace food_sufficient_days_l235_235699

theorem food_sufficient_days (D : ℕ) (h1 : 1000 * D - 10000 = 800 * D) : D = 50 :=
sorry

end food_sufficient_days_l235_235699


namespace binom_n_n_minus_1_l235_235396

theorem binom_n_n_minus_1 (n : ℕ) (h : 0 < n) : (Nat.choose n (n-1)) = n :=
  sorry

end binom_n_n_minus_1_l235_235396


namespace area_of_given_sector_l235_235761

noncomputable def area_of_sector (alpha l : ℝ) : ℝ :=
  let r := l / alpha
  (1 / 2) * l * r

theorem area_of_given_sector :
  let alpha := Real.pi / 9
  let l := Real.pi / 3
  area_of_sector alpha l = Real.pi / 2 :=
by
  sorry

end area_of_given_sector_l235_235761


namespace xiaohong_height_l235_235316

theorem xiaohong_height 
  (father_height_cm : ℕ)
  (height_difference_dm : ℕ)
  (father_height : father_height_cm = 170)
  (height_difference : height_difference_dm = 4) :
  ∃ xiaohong_height_cm : ℕ, xiaohong_height_cm + height_difference_dm * 10 = father_height_cm :=
by
  use 130
  sorry

end xiaohong_height_l235_235316


namespace geom_seq_result_l235_235594

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

end geom_seq_result_l235_235594


namespace inequality_holds_for_real_numbers_l235_235456

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l235_235456


namespace xiaoMing_better_performance_l235_235544

-- Definitions based on conditions
def xiaoMing_scores : List Float := [90, 67, 90, 92, 96]
def xiaoLiang_scores : List Float := [87, 62, 90, 92, 92]

-- Definitions of average and variance calculation
def average (scores : List Float) : Float :=
  (scores.sum) / (scores.length.toFloat)

def variance (scores : List Float) : Float :=
  let avg := average scores
  (scores.map (λ x => (x - avg) ^ 2)).sum / (scores.length.toFloat)

-- Prove that Xiao Ming's performance is better than Xiao Liang's.
theorem xiaoMing_better_performance :
  average xiaoMing_scores > average xiaoLiang_scores ∧ variance xiaoMing_scores < variance xiaoLiang_scores :=
by
  sorry

end xiaoMing_better_performance_l235_235544


namespace sufficient_condition_for_inequality_l235_235002

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a > 4 :=
by 
  sorry

end sufficient_condition_for_inequality_l235_235002


namespace max_intersection_distance_l235_235089

theorem max_intersection_distance :
  let C1_x (α : ℝ) := 2 + 2 * Real.cos α
  let C1_y (α : ℝ) := 2 * Real.sin α
  let C2_x (β : ℝ) := 2 * Real.cos β
  let C2_y (β : ℝ) := 2 + 2 * Real.sin β
  let l1 (α : ℝ) := α
  let l2 (α : ℝ) := α - Real.pi / 6
  (0 < Real.pi / 2) →
  let OP (α : ℝ) := 4 * Real.cos α
  let OQ (α : ℝ) := 4 * Real.sin (α - Real.pi / 6)
  let pq_prod (α : ℝ) := OP α * OQ α
  ∀α, 0 < α ∧ α < Real.pi / 2 → pq_prod α ≤ 4 := by
  sorry

end max_intersection_distance_l235_235089


namespace find_base_number_l235_235502

theorem find_base_number (y : ℕ) (base : ℕ) (h : 9^y = base ^ 16) (hy : y = 8) : base = 3 :=
by
  -- We skip the proof steps and insert sorry here
  sorry

end find_base_number_l235_235502


namespace range_of_k_no_third_quadrant_l235_235945

theorem range_of_k_no_third_quadrant (k : ℝ) : ¬(∃ x : ℝ, ∃ y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x + 3) → k ≤ 0 := 
sorry

end range_of_k_no_third_quadrant_l235_235945


namespace find_integer_m_l235_235234

theorem find_integer_m 
  (m : ℤ) (h_pos : m > 0) 
  (h_intersect : ∃ (x y : ℤ), 17 * x + 7 * y = 1000 ∧ y = m * x + 2) : 
  m = 68 :=
by
  sorry

end find_integer_m_l235_235234


namespace triangle_sides_inequality_l235_235631

theorem triangle_sides_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
    (a/(b + c - a) + b/(c + a - b) + c/(a + b - c)) ≥ ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ∧
    ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ≥ 3 :=
by
  sorry

end triangle_sides_inequality_l235_235631


namespace remaining_surface_area_correct_l235_235705

open Real

-- Define the original cube and the corner cubes
def orig_cube : ℝ × ℝ × ℝ := (5, 5, 5)
def corner_cube : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define a function to compute the surface area of a cube given dimensions (a, b, c)
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)

-- Original surface area of the cube
def orig_surface_area : ℝ := surface_area 5 5 5

-- Total surface area of the remaining figure after removing 8 corner cubes
def remaining_surface_area : ℝ := 150  -- Calculated directly as 6 * 25

-- Theorem stating that the surface area of the remaining figure is 150 cm^2
theorem remaining_surface_area_correct :
  remaining_surface_area = 150 := sorry

end remaining_surface_area_correct_l235_235705


namespace cos_8_degree_l235_235006

theorem cos_8_degree (m : ℝ) (h : Real.sin (74 * Real.pi / 180) = m) :
  Real.cos (8 * Real.pi / 180) = Real.sqrt ((1 + m) / 2) :=
sorry

end cos_8_degree_l235_235006


namespace original_weight_of_potatoes_l235_235129

theorem original_weight_of_potatoes (W : ℝ) (h : W / (W / 2) = 36) : W = 648 := by
  sorry

end original_weight_of_potatoes_l235_235129


namespace base_r_correct_l235_235156

theorem base_r_correct (r : ℕ) :
  (5 * r ^ 2 + 6 * r) + (4 * r ^ 2 + 2 * r) = r ^ 3 + r ^ 2 → r = 8 := 
by 
  sorry

end base_r_correct_l235_235156


namespace seating_arrangements_l235_235212

theorem seating_arrangements (n : ℕ) (h_n : n = 6) (A B : Fin n) (h : A ≠ B) : 
  ∃ k : ℕ, k = 240 := 
by 
  sorry

end seating_arrangements_l235_235212


namespace closest_vector_l235_235608

open Real

def u (s : ℝ) : ℝ × ℝ × ℝ := (1 + 3 * s, -4 + 7 * s, 2 + 4 * s)
def b : ℝ × ℝ × ℝ := (5, 1, -3)
def direction : ℝ × ℝ × ℝ := (3, 7, 4)

theorem closest_vector (s : ℝ) :
  (u s - b) • direction = 0 ↔ s = 27 / 74 :=
sorry

end closest_vector_l235_235608


namespace probability_two_dice_same_number_l235_235214

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end probability_two_dice_same_number_l235_235214


namespace length_of_LO_l235_235576

theorem length_of_LO (MN LO : ℝ) (alt_O_MN alt_N_LO : ℝ) (h_MN : MN = 15) 
  (h_alt_O_MN : alt_O_MN = 9) (h_alt_N_LO : alt_N_LO = 7) : 
  LO = 19 + 2 / 7 :=
by
  -- Sorry means to skip the proof.
  sorry

end length_of_LO_l235_235576


namespace range_of_a_l235_235135

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 2 * x * Real.log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l235_235135


namespace distance_is_30_l235_235080

-- Define given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Define the distance from Mrs. Hilt's desk to the water fountain
def distance_to_water_fountain : ℕ := total_distance / trips

-- Prove the distance is 30 feet
theorem distance_is_30 : distance_to_water_fountain = 30 :=
by
  -- Utilizing the division defined in distance_to_water_fountain
  sorry

end distance_is_30_l235_235080


namespace geometric_progression_condition_l235_235514

noncomputable def condition_for_geometric_progression (a q : ℝ) (n p : ℤ) : Prop :=
  ∃ m : ℤ, a = q^m

theorem geometric_progression_condition (a q : ℝ) (n p k : ℤ) :
  condition_for_geometric_progression a q n p ↔ a * q^(n + p) = a * q^k :=
by
  sorry

end geometric_progression_condition_l235_235514


namespace least_n_divisibility_l235_235104

theorem least_n_divisibility :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ n → k ∣ (n - 1)^2) ∧ (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ ¬ k ∣ (n - 1)^2) ∧ n = 3 :=
by
  sorry

end least_n_divisibility_l235_235104


namespace probability_three_red_before_two_green_l235_235124

noncomputable def probability_red_chips_drawn_before_green (red_chips green_chips : ℕ) (total_chips : ℕ) : ℚ := sorry

theorem probability_three_red_before_two_green 
    (red_chips green_chips : ℕ) (total_chips : ℕ)
    (h_red : red_chips = 3) (h_green : green_chips = 2) 
    (h_total: total_chips = red_chips + green_chips) :
  probability_red_chips_drawn_before_green red_chips green_chips total_chips = 3 / 10 :=
  sorry

end probability_three_red_before_two_green_l235_235124


namespace boxes_left_for_Sonny_l235_235998

def initial_boxes : ℕ := 45
def boxes_given_to_brother : ℕ := 12
def boxes_given_to_sister : ℕ := 9
def boxes_given_to_cousin : ℕ := 7

def total_given_away : ℕ := boxes_given_to_brother + boxes_given_to_sister + boxes_given_to_cousin

def remaining_boxes : ℕ := initial_boxes - total_given_away

theorem boxes_left_for_Sonny : remaining_boxes = 17 := by
  sorry

end boxes_left_for_Sonny_l235_235998


namespace sequence_general_term_l235_235863

def recurrence_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n / (1 + a n)

theorem sequence_general_term :
  ∀ a : ℕ → ℚ, recurrence_sequence a → ∀ n : ℕ, n ≥ 1 → a n = 2 / (2 * n - 1) :=
by
  intro a h n hn
  sorry

end sequence_general_term_l235_235863


namespace billy_age_l235_235555

theorem billy_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end billy_age_l235_235555


namespace find_n_that_makes_vectors_collinear_l235_235295

theorem find_n_that_makes_vectors_collinear (n : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, n)) (h_collinear : ∃ k : ℝ, 2 • a - b = k • b) : n = 9 :=
sorry

end find_n_that_makes_vectors_collinear_l235_235295


namespace sphere_volume_in_cone_l235_235288

theorem sphere_volume_in_cone :
  let d := 24
  let theta := 90
  let r := 24 * (Real.sqrt 2 - 1)
  let V := (4 / 3) * Real.pi * r^3
  ∃ (R : ℝ), r = R ∧ V = (4 / 3) * Real.pi * R^3 := by
  sorry

end sphere_volume_in_cone_l235_235288


namespace minimize_fraction_l235_235065

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) ↔ (∀ m : ℕ, 0 < m → (n / 3 + 27 / n) ≤ (m / 3 + 27 / m)) :=
by
  sorry

end minimize_fraction_l235_235065


namespace num_sets_satisfying_union_l235_235497

theorem num_sets_satisfying_union : 
  ∃! (A : Set ℕ), ({1, 3} ∪ A = {1, 3, 5}) :=
by
  sorry

end num_sets_satisfying_union_l235_235497


namespace oranges_in_each_box_l235_235520

theorem oranges_in_each_box (total_oranges : ℝ) (boxes : ℝ) (h_total : total_oranges = 72) (h_boxes : boxes = 3.0) : total_oranges / boxes = 24 :=
by
  -- Begin proof
  sorry

end oranges_in_each_box_l235_235520


namespace area_RWP_l235_235781

-- Definitions
variables (X Y Z W P Q R : ℝ × ℝ)
variables (h₁ : (X.1 - Z.1) * (X.1 - Z.1) + (X.2 - Z.2) * (X.2 - Z.2) = 144)
variables (h₂ : P.1 = X.1 - 8 ∧ P.2 = X.2)
variables (h₃ : Q.1 = (Z.1 + P.1) / 2 ∧ Q.2 = (Z.2 + P.2) / 2)
variables (h₄ : R.1 = (Y.1 + P.1) / 2 ∧ R.2 = (Y.2 + P.2) / 2)
variables (h₅ : 1 / 2 * ((Z.1 - X.1) * (W.2 - X.2) - (Z.2 - X.2) * (W.1 - X.1)) = 72)
variables (h₆ : 1 / 2 * abs ((Q.1 - X.1) * (W.2 - X.2) - (Q.2 - X.2) * (W.1 - X.1)) = 20)

-- Theorem statement
theorem area_RWP : 
  1 / 2 * abs ((R.1 - W.1) * (P.2 - W.2) - (R.2 - W.2) * (P.1 - W.1)) = 12 :=
sorry

end area_RWP_l235_235781


namespace set_union_is_correct_l235_235405

noncomputable def M (a : ℝ) : Set ℝ := {3, 2^a}
noncomputable def N (a b : ℝ) : Set ℝ := {a, b}

variable (a b : ℝ)
variable (h₁ : M a ∩ N a b = {2})
variable (h₂ : ∃ a b, N a b = {1, 2} ∧ M a = {3, 2} ∧ M a ∪ N a b = {1, 2, 3})

theorem set_union_is_correct :
  M 1 ∪ N 1 2 = {1, 2, 3} :=
by
  sorry

end set_union_is_correct_l235_235405


namespace not_rented_two_bedroom_units_l235_235929

theorem not_rented_two_bedroom_units (total_units : ℕ)
  (units_rented_ratio : ℚ)
  (total_rented_units : ℕ)
  (one_bed_room_rented_ratio two_bed_room_rented_ratio three_bed_room_rented_ratio : ℚ)
  (one_bed_room_rented_count two_bed_room_rented_count three_bed_room_rented_count : ℕ)
  (x : ℕ) 
  (total_two_bed_room_units rented_two_bed_room_units : ℕ)
  (units_ratio_condition : 2*x + 3*x + 4*x = total_rented_units)
  (total_units_condition : total_units = 1200)
  (ratio_condition : units_rented_ratio = 7/12)
  (rented_units_condition : total_rented_units = (7/12) * total_units)
  (one_bed_condition : one_bed_room_rented_ratio = 2/5)
  (two_bed_condition : two_bed_room_rented_ratio = 1/2)
  (three_bed_condition : three_bed_room_rented_ratio = 3/8)
  (one_bed_count : one_bed_room_rented_count = 2 * x)
  (two_bed_count : two_bed_room_rented_count = 3 * x)
  (three_bed_count : three_bed_room_rented_count = 4 * x)
  (x_value : x = total_rented_units / 9)
  (total_two_bed_units_calc : total_two_bed_room_units = 2 * two_bed_room_rented_count)
  : total_two_bed_room_units - two_bed_room_rented_count = 231 :=
  by
  sorry

end not_rented_two_bedroom_units_l235_235929


namespace complement_union_A_B_in_U_l235_235875

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l235_235875


namespace sum_of_squares_of_diagonals_l235_235546

variable (OP R : ℝ)

theorem sum_of_squares_of_diagonals (AC BD : ℝ) :
  AC^2 + BD^2 = 8 * R^2 - 4 * OP^2 :=
sorry

end sum_of_squares_of_diagonals_l235_235546


namespace proof_of_p_and_not_q_l235_235838

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > 1

theorem proof_of_p_and_not_q : p ∧ ¬q :=
by {
  sorry
}

end proof_of_p_and_not_q_l235_235838


namespace number_in_interval_l235_235703

def number := 0.2012
def lower_bound := 0.2
def upper_bound := 0.25

theorem number_in_interval : lower_bound < number ∧ number < upper_bound :=
by
  sorry

end number_in_interval_l235_235703


namespace hannah_final_pay_l235_235409

theorem hannah_final_pay : (30 * 18) - (5 * 3) + (15 * 4) - (((30 * 18) - (5 * 3) + (15 * 4)) * 0.10 + ((30 * 18) - (5 * 3) + (15 * 4)) * 0.05) = 497.25 :=
by
  sorry

end hannah_final_pay_l235_235409


namespace number_of_triangles_l235_235496

theorem number_of_triangles (n : ℕ) (hn : 0 < n) :
  ∃ t, t = (n + 2) ^ 2 - 2 * (⌊ (n : ℝ) / 2 ⌋) / 4 :=
by
  sorry

end number_of_triangles_l235_235496


namespace product_value_l235_235902

noncomputable def product_of_integers (A B C D : ℕ) : ℕ :=
  A * B * C * D

theorem product_value :
  ∃ (A B C D : ℕ), A + B + C + D = 72 ∧ 
                    A + 2 = B - 2 ∧ 
                    A + 2 = C * 2 ∧ 
                    A + 2 = D / 2 ∧ 
                    product_of_integers A B C D = 64512 :=
by
  sorry

end product_value_l235_235902


namespace rhombus_area_l235_235008

theorem rhombus_area (side d1 d2 : ℝ) (h_side : side = 25) (h_d1 : d1 = 30) (h_diag : d2 = 40) :
  (d1 * d2) / 2 = 600 :=
by
  rw [h_d1, h_diag]
  norm_num

end rhombus_area_l235_235008


namespace minimum_value_2sqrt5_l235_235336

theorem minimum_value_2sqrt5 : ∀ x : ℝ, 
  ∃ m : ℝ, (∀ x : ℝ, m ≤ (x^2 + 10) / (Real.sqrt (x^2 + 5))) ∧ (m = 2 * Real.sqrt 5) := by
  sorry

end minimum_value_2sqrt5_l235_235336


namespace cube_dimension_l235_235981

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end cube_dimension_l235_235981


namespace unique_function_l235_235469

noncomputable def f : ℝ → ℝ := sorry

theorem unique_function 
  (h_f : ∀ x > 0, ∀ y > 0, f x * f y = 2 * f (x + y * f x)) : ∀ x > 0, f x = 2 :=
by
  sorry

end unique_function_l235_235469


namespace birds_in_tree_l235_235264

def initialBirds : Nat := 14
def additionalBirds : Nat := 21
def totalBirds := initialBirds + additionalBirds

theorem birds_in_tree : totalBirds = 35 := by
  sorry

end birds_in_tree_l235_235264


namespace score_after_7_hours_l235_235864

theorem score_after_7_hours (score : ℕ) (time : ℕ) : 
  (score / time = 90 / 5) → time = 7 → score = 126 :=
by
  sorry

end score_after_7_hours_l235_235864


namespace xy_yz_zx_over_x2_y2_z2_l235_235016

theorem xy_yz_zx_over_x2_y2_z2 (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h_sum : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end xy_yz_zx_over_x2_y2_z2_l235_235016


namespace common_ratio_geometric_sequence_l235_235529

theorem common_ratio_geometric_sequence (a b c d : ℤ) (h1 : a = 10) (h2 : b = -20) (h3 : c = 40) (h4 : d = -80) :
    b / a = -2 ∧ c = b * -2 ∧ d = c * -2 := by
  sorry

end common_ratio_geometric_sequence_l235_235529


namespace units_digit_7_pow_6_pow_5_l235_235000

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l235_235000


namespace sum_of_first_six_terms_of_geometric_series_l235_235627

-- Definitions for the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 6

-- Define the formula for the sum of the first n terms of a geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The equivalent Lean 4 statement
theorem sum_of_first_six_terms_of_geometric_series :
  geometric_series_sum a r n = 4095 / 12288 :=
by
  sorry

end sum_of_first_six_terms_of_geometric_series_l235_235627


namespace product_of_two_numbers_l235_235985

theorem product_of_two_numbers
  (x y : ℝ)
  (h_diff : x - y ≠ 0)
  (h1 : x + y = 5 * (x - y))
  (h2 : x * y = 15 * (x - y)) :
  x * y = 37.5 :=
by
  sorry

end product_of_two_numbers_l235_235985


namespace probability_at_least_one_correct_l235_235343

-- Define the probability of missing a single question
def prob_miss_one : ℚ := 3 / 4

-- Define the probability of missing all six questions
def prob_miss_six : ℚ := prob_miss_one ^ 6

-- Define the probability of getting at least one correct answer
def prob_at_least_one : ℚ := 1 - prob_miss_six

-- The problem statement
theorem probability_at_least_one_correct :
  prob_at_least_one = 3367 / 4096 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_one_correct_l235_235343


namespace unit_digit_25_pow_2010_sub_3_pow_2012_l235_235446

theorem unit_digit_25_pow_2010_sub_3_pow_2012 :
  (25^2010 - 3^2012) % 10 = 4 :=
by 
  sorry

end unit_digit_25_pow_2010_sub_3_pow_2012_l235_235446


namespace tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l235_235883

open Real

theorem tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence (α β γ : ℝ) 
  (h1 : α + β + γ = π)  -- Assuming α, β, γ are angles in a triangle
  (h2 : tan α + tan γ = 2 * tan β) :
  sin (2 * α) + sin (2 * γ) = 2 * sin (2 * β) :=
by
  sorry

end tangents_arithmetic_sequence_implies_sines_double_angles_arithmetic_sequence_l235_235883


namespace cubics_of_sum_and_product_l235_235534

theorem cubics_of_sum_and_product (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : 
  x^3 + y^3 = 640 :=
by
  sorry

end cubics_of_sum_and_product_l235_235534


namespace bobby_pays_correct_amount_l235_235430

noncomputable def bobby_total_cost : ℝ := 
  let mold_cost : ℝ := 250
  let material_original_cost : ℝ := 150
  let material_discount : ℝ := 0.20 * material_original_cost
  let material_cost : ℝ := material_original_cost - material_discount
  let hourly_rate_original : ℝ := 75
  let hourly_rate_increased : ℝ := hourly_rate_original + 10
  let work_hours : ℝ := 8
  let work_cost_original : ℝ := work_hours * hourly_rate_increased
  let work_cost_discount : ℝ := 0.80 * work_cost_original
  let cost_before_tax : ℝ := mold_cost + material_cost + work_cost_discount
  let tax : ℝ := 0.10 * cost_before_tax
  cost_before_tax + tax

theorem bobby_pays_correct_amount : bobby_total_cost = 1005.40 := sorry

end bobby_pays_correct_amount_l235_235430


namespace number_of_squares_centered_at_60_45_l235_235037

noncomputable def number_of_squares_centered_at (cx : ℕ) (cy : ℕ) : ℕ :=
  let aligned_with_axes := 45
  let not_aligned_with_axes := 2025
  aligned_with_axes + not_aligned_with_axes

theorem number_of_squares_centered_at_60_45 : number_of_squares_centered_at 60 45 = 2070 := 
  sorry

end number_of_squares_centered_at_60_45_l235_235037


namespace Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l235_235208

theorem Mishas_fathers_speed (d : ℝ) (t : ℝ) (V : ℝ) 
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) :
  V = 1 :=
by
  sorry

theorem Mishas_fathers_speed_in_kmh (d : ℝ) (t : ℝ) (V : ℝ) (V_kmh : ℝ)
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) 
  (h4 : V_kmh = V * 60):
  V_kmh = 60 :=
by
  sorry

end Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l235_235208


namespace Mitya_age_l235_235811

/--
Assume Mitya's current age is M and Shura's current age is S. If Mitya is 11 years older than Shura,
and when Mitya was as old as Shura is now, he was twice as old as Shura,
then prove that M = 27.5.
-/
theorem Mitya_age (S M : ℝ) (h1 : M = S + 11) (h2 : M - S = 2 * (S - (M - S))) : M = 27.5 :=
by
  sorry

end Mitya_age_l235_235811


namespace compute_difference_l235_235239

noncomputable def f (n : ℝ) : ℝ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem compute_difference (r : ℝ) : f r - f (r - 1) = r * (r + 1) * (r + 2) := by
  sorry

end compute_difference_l235_235239


namespace time_A_reaches_destination_l235_235139

theorem time_A_reaches_destination (x t : ℝ) (h_ratio : (4 * t) = 3 * (t + 0.5)) : (t + 0.5) = 2 :=
by {
  -- derived by algebraic manipulation
  sorry
}

end time_A_reaches_destination_l235_235139


namespace no_point_in_punctured_disk_l235_235350

theorem no_point_in_punctured_disk (A B C D E F G : ℝ) (hB2_4AC : B^2 - 4 * A * C < 0) :
  ∃ δ > 0, ∀ x y : ℝ, 0 < x^2 + y^2 → x^2 + y^2 < δ^2 → 
    ¬(A * x^2 + B * x * y + C * y^2 + D * x^3 + E * x^2 * y + F * x * y^2 + G * y^3 = 0) :=
sorry

end no_point_in_punctured_disk_l235_235350


namespace sum_of_coefficients_zero_l235_235933

theorem sum_of_coefficients_zero (A B C D E F : ℝ) :
  (∀ x : ℝ,
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by
  intro h
  -- Proof omitted
  sorry

end sum_of_coefficients_zero_l235_235933


namespace correct_result_l235_235097

-- Given condition
def mistaken_calculation (x : ℤ) : Prop :=
  x / 3 = 45

-- Proposition to prove the correct result
theorem correct_result (x : ℤ) (h : mistaken_calculation x) : 3 * x = 405 := by
  -- Here we can solve the proof later
  sorry

end correct_result_l235_235097


namespace females_count_l235_235105

-- Defining variables and constants
variables (P M F : ℕ)
-- The condition given the total population
def town_population := P = 600
-- The condition given the proportion of males
def proportion_of_males := M = P / 3
-- The condition determining the number of females
def number_of_females := F = P - M

-- The theorem stating the number of females is 400
theorem females_count (P M F : ℕ) (h1 : town_population P)
  (h2 : proportion_of_males P M) 
  (h3 : number_of_females P M F) : 
  F = 400 := 
sorry

end females_count_l235_235105


namespace prize_expectation_l235_235993

theorem prize_expectation :
  let total_people := 100
  let envelope_percentage := 0.4
  let grand_prize_prob := 0.1
  let second_prize_prob := 0.2
  let consolation_prize_prob := 0.3
  let people_with_envelopes := total_people * envelope_percentage
  let grand_prize_winners := people_with_envelopes * grand_prize_prob
  let second_prize_winners := people_with_envelopes * second_prize_prob
  let consolation_prize_winners := people_with_envelopes * consolation_prize_prob
  let empty_envelopes := people_with_envelopes - (grand_prize_winners + second_prize_winners + consolation_prize_winners)
  grand_prize_winners = 4 ∧
  second_prize_winners = 8 ∧
  consolation_prize_winners = 12 ∧
  empty_envelopes = 16 := by
  sorry

end prize_expectation_l235_235993


namespace min_passengers_to_fill_bench_l235_235137

theorem min_passengers_to_fill_bench (width_per_passenger : ℚ) (total_seat_width : ℚ) (num_seats : ℕ):
  width_per_passenger = 1/6 → total_seat_width = num_seats → num_seats = 6 → 3 ≥ (total_seat_width / width_per_passenger) :=
by
  intro h1 h2 h3
  sorry

end min_passengers_to_fill_bench_l235_235137


namespace max_children_tickets_l235_235274

theorem max_children_tickets 
  (total_budget : ℕ) (adult_ticket_cost : ℕ) 
  (child_ticket_cost_individual : ℕ) (child_ticket_cost_group : ℕ) (min_group_tickets : ℕ) 
  (remaining_budget : ℕ) :
  total_budget = 75 →
  adult_ticket_cost = 12 →
  child_ticket_cost_individual = 6 →
  child_ticket_cost_group = 4 →
  min_group_tickets = 5 →
  (remaining_budget = total_budget - adult_ticket_cost) →
  ∃ (n : ℕ), n = 15 ∧ n * child_ticket_cost_group ≤ remaining_budget :=
by
  intros h_total_budget h_adult_ticket_cost h_child_ticket_cost_individual h_child_ticket_cost_group h_min_group_tickets h_remaining_budget
  sorry

end max_children_tickets_l235_235274


namespace fraction_spent_on_candy_l235_235369

theorem fraction_spent_on_candy (initial_quarters : ℕ) (initial_cents remaining_cents cents_per_dollar : ℕ) (fraction_spent : ℝ) :
  initial_quarters = 14 ∧ remaining_cents = 300 ∧ initial_cents = initial_quarters * 25 ∧ cents_per_dollar = 100 →
  fraction_spent = (initial_cents - remaining_cents) / cents_per_dollar →
  fraction_spent = 1 / 2 :=
by
  intro h1 h2
  sorry

end fraction_spent_on_candy_l235_235369


namespace kris_fraction_l235_235805

-- Definitions based on problem conditions
def Trey (kris : ℕ) := 7 * kris
def Kristen := 12
def Trey_kristen_diff := 9
def Kris_fraction_to_Kristen (kris : ℕ) : ℚ := kris / Kristen

-- Theorem statement: Proving the required fraction
theorem kris_fraction (kris : ℕ) (h1 : Trey kris = Kristen + Trey_kristen_diff) : 
  Kris_fraction_to_Kristen kris = 1 / 4 :=
by
  sorry

end kris_fraction_l235_235805


namespace fraction_sum_eq_l235_235218

theorem fraction_sum_eq : (7 / 10 : ℚ) + (3 / 100) + (9 / 1000) = 0.739 := sorry

end fraction_sum_eq_l235_235218


namespace smallest_m_divisible_by_15_l235_235173

noncomputable def largest_prime_with_2023_digits : ℕ := sorry

theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ (largest_prime_with_2023_digits ^ 2 - m) % 15 = 0 ∧ m = 1 :=
  sorry

end smallest_m_divisible_by_15_l235_235173


namespace simplify_fraction_product_l235_235831

theorem simplify_fraction_product : 
  (21 / 28) * (14 / 33) * (99 / 42) = 1 := 
by 
  sorry

end simplify_fraction_product_l235_235831


namespace perpendicular_distance_H_to_plane_EFG_l235_235634

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def E : Point3D := ⟨5, 0, 0⟩
def F : Point3D := ⟨0, 3, 0⟩
def G : Point3D := ⟨0, 0, 4⟩
def H : Point3D := ⟨0, 0, 0⟩

def distancePointToPlane (H E F G : Point3D) : ℝ := sorry

theorem perpendicular_distance_H_to_plane_EFG :
  distancePointToPlane H E F G = 1.8 := sorry

end perpendicular_distance_H_to_plane_EFG_l235_235634


namespace complement_intersection_example_l235_235980

open Set

variable (U A B : Set ℕ)

def C_U (A : Set ℕ) (U : Set ℕ) : Set ℕ := U \ A

theorem complement_intersection_example 
  (hU : U = {0, 1, 2, 3})
  (hA : A = {0, 1})
  (hB : B = {1, 2, 3}) :
  (C_U A U) ∩ B = {2, 3} :=
by
  sorry

end complement_intersection_example_l235_235980


namespace total_pencils_given_out_l235_235467

theorem total_pencils_given_out (n p : ℕ) (h1 : n = 10) (h2 : p = 5) : n * p = 50 :=
by
  sorry

end total_pencils_given_out_l235_235467


namespace valid_rod_count_l235_235406

open Nat

theorem valid_rod_count :
  ∃ valid_rods : Finset ℕ,
    (∀ d ∈ valid_rods, 6 ≤ d ∧ d < 35 ∧ d ≠ 5 ∧ d ≠ 10 ∧ d ≠ 20) ∧ 
    valid_rods.card = 26 := sorry

end valid_rod_count_l235_235406


namespace functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l235_235057

-- Conditions for y1
def cost_price : ℕ := 60
def selling_price_first_10_days : ℕ := 80
def y1 : ℕ → ℕ := fun x => x * x - 8 * x + 56
def items_sold_day4 : ℕ := 40
def items_sold_day6 : ℕ := 44

-- Conditions for y2
def selling_price_post_10_days : ℕ := 100
def y2 : ℕ → ℕ := fun x => 2 * x + 8
def gross_profit_condition : ℕ := 1120

-- 1) Prove functional relationship of y1.
theorem functional_relationship_y1 (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) : 
  y1 x = x * x - 8 * x + 56 := 
by
  sorry

-- 2) Prove value of x for daily gross profit $1120 on any day within first 10 days.
theorem daily_gross_profit_1120_first_10_days (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) (gp : (selling_price_first_10_days - cost_price) * y1 x = gross_profit_condition) : 
  x = 8 := 
by
  sorry

-- 3) Prove total gross profit W and range for 26 < x ≤ 31.
theorem total_gross_profit_W (x : ℕ) (h : 26 < x ∧ x ≤ 31) : 
  (100 - (cost_price - 2 * (y2 x - 60))) * (y2 x) = 8 * x * x - 96 * x - 512 := 
by
  sorry

end functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l235_235057


namespace coffee_price_l235_235123

theorem coffee_price (qd : ℝ) (d : ℝ) (rp : ℝ) :
  qd = 4.5 ∧ d = 0.25 → rp = 12 :=
by 
  sorry

end coffee_price_l235_235123


namespace moving_circle_trajectory_l235_235275

-- Define the two given circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- The theorem statement
theorem moving_circle_trajectory :
  (∀ x y : ℝ, (exists r : ℝ, r > 0 ∧ ∃ M : ℝ × ℝ, 
  (C₁ M.1 M.2 ∧ ((M.1 - 4)^2 + M.2^2 = (13 - r)^2) ∧
  C₂ M.1 M.2 ∧ ((M.1 + 4)^2 + M.2^2 = (r + 3)^2)) ∧
  ((x = M.1) ∧ (y = M.2))) ↔ (x^2 / 64 + y^2 / 48 = 1)) := sorry

end moving_circle_trajectory_l235_235275


namespace rubber_boat_lost_time_l235_235061

theorem rubber_boat_lost_time (a b : ℝ) (x : ℝ) (h : (5 - x) * (a - b) + (6 - x) * b = a + b) : x = 4 :=
  sorry

end rubber_boat_lost_time_l235_235061


namespace det_transformed_matrix_l235_235818

variables {p q r s : ℝ} -- Defining the variables over the real numbers

-- Defining the first determinant condition as an axiom
axiom det_initial_matrix : (p * s - q * r) = 10

-- Stating the theorem to be proved
theorem det_transformed_matrix : 
  (p + 2 * r) * s - (q + 2 * s) * r = 10 :=
by
  sorry -- Placeholder for the actual proof

end det_transformed_matrix_l235_235818


namespace sqrt_9025_squared_l235_235868

-- Define the square root function and its properties
noncomputable def sqrt (x : ℕ) : ℕ := sorry

axiom sqrt_def (n : ℕ) (hn : 0 ≤ n) : (sqrt n) ^ 2 = n

-- Prove the specific case
theorem sqrt_9025_squared : (sqrt 9025) ^ 2 = 9025 :=
sorry

end sqrt_9025_squared_l235_235868


namespace subtract_rational_from_zero_yields_additive_inverse_l235_235877

theorem subtract_rational_from_zero_yields_additive_inverse (a : ℚ) : 0 - a = -a := by
  sorry

end subtract_rational_from_zero_yields_additive_inverse_l235_235877


namespace circle_areas_equal_l235_235936

theorem circle_areas_equal :
  let r1 := 15
  let d2 := 30
  let r2 := d2 / 2
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  A1 = A2 :=
by
  sorry

end circle_areas_equal_l235_235936


namespace find_z_l235_235329

-- Definitions from conditions
def x : ℕ := 22
def y : ℕ := 13
def total_boys_who_went_down_slide : ℕ := x + y
def ratio_slide_to_watch := 5 / 3

-- Statement we need to prove
theorem find_z : ∃ z : ℕ, (5 / 3 = total_boys_who_went_down_slide / z) ∧ z = 21 :=
by
  use 21
  sorry

end find_z_l235_235329


namespace cos_beta_l235_235843

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.cos α = 3/5)
  (h2 : Real.cos (α + β) = -5/13) : Real.cos β = 33/65 := 
sorry

end cos_beta_l235_235843


namespace find_m_given_a3_eq_40_l235_235765

theorem find_m_given_a3_eq_40 (m : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 - m * x) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_3 = 40 →
  m = -1 := 
by 
  sorry

end find_m_given_a3_eq_40_l235_235765


namespace raft_people_with_life_jackets_l235_235242

theorem raft_people_with_life_jackets (n m k : ℕ) (h1 : n = 21) (h2 : m = n - 7) (h3 : k = 8) :
  n - (k / (m / (n - m))) = 17 := 
by sorry

end raft_people_with_life_jackets_l235_235242


namespace no_arithmetic_sequence_without_square_gt1_l235_235230

theorem no_arithmetic_sequence_without_square_gt1 (a d : ℕ) (h_d : d ≠ 0) :
  ¬(∀ n : ℕ, ∃ k : ℕ, k > 0 ∧ k ∈ {a + n * d | n : ℕ} ∧ ∀ m : ℕ, m > 1 → m * m ∣ k → false) := sorry

end no_arithmetic_sequence_without_square_gt1_l235_235230


namespace interest_rate_l235_235570

theorem interest_rate (P1 P2 I T1 T2 total_amount : ℝ) (r : ℝ) :
  P1 = 10000 →
  P2 = 22000 →
  T1 = 2 →
  T2 = 3 →
  total_amount = 27160 →
  (I = P1 * r * T1 / 100 + P2 * r * T2 / 100) →
  P1 + P2 = 22000 →
  (P1 + I = total_amount) →
  r = 6 :=
by
  intros hP1 hP2 hT1 hT2 htotal_amount hI hP_total hP1_I_total
  -- Actual proof would go here
  sorry

end interest_rate_l235_235570


namespace max_volume_solid_l235_235964

-- Define volumes of individual cubes
def cube_volume (side: ℕ) : ℕ := side * side * side

-- Calculate the total number of cubes in the solid
def total_cubes (base_layer : ℕ) (second_layer : ℕ) : ℕ := base_layer + second_layer

-- Define the base layer and second layer cubes
def base_layer_cubes : ℕ := 4 * 4
def second_layer_cubes : ℕ := 2 * 2

-- Define the total volume of the solid
def total_volume (side_length : ℕ) (base_layer : ℕ) (second_layer : ℕ) : ℕ := 
  total_cubes base_layer second_layer * cube_volume side_length

theorem max_volume_solid :
  total_volume 3 base_layer_cubes second_layer_cubes = 540 := by
  sorry

end max_volume_solid_l235_235964


namespace speeding_tickets_l235_235540

theorem speeding_tickets (p1 p2 : ℝ)
  (h1 : p1 = 16.666666666666664)
  (h2 : p2 = 40) :
  (p1 * (100 - p2) / 100 = 10) :=
by sorry

end speeding_tickets_l235_235540


namespace mixture_percent_chemical_a_l235_235549

-- Defining the conditions
def solution_x : ℝ := 0.4
def solution_y : ℝ := 0.5
def percent_x_in_mixture : ℝ := 0.3
def percent_y_in_mixture : ℝ := 1.0 - percent_x_in_mixture

-- The goal is to prove that the mixture is 47% chemical a
theorem mixture_percent_chemical_a : (solution_x * percent_x_in_mixture + solution_y * percent_y_in_mixture) * 100 = 47 :=
by
  -- Calculation here
  sorry

end mixture_percent_chemical_a_l235_235549


namespace find_x_parallel_vectors_l235_235717

theorem find_x_parallel_vectors :
  ∀ x : ℝ, (∃ k : ℝ, (1, 2) = (k * (2 * x), k * (-3))) → x = -3 / 4 :=
by
  sorry

end find_x_parallel_vectors_l235_235717


namespace remaining_pie_after_carlos_and_maria_l235_235713

theorem remaining_pie_after_carlos_and_maria (C M R : ℝ) (hC : C = 0.60) (hM : M = 0.25 * (1 - C)) : R = 1 - C - M → R = 0.30 :=
by
  intro hR
  simp only [hC, hM] at hR
  sorry

end remaining_pie_after_carlos_and_maria_l235_235713


namespace football_defeat_points_l235_235967

theorem football_defeat_points (V D F : ℕ) (x : ℕ) :
    3 * V + D + x * F = 8 →
    27 + 6 * x = 32 →
    x = 0 :=
by
    intros h1 h2
    sorry

end football_defeat_points_l235_235967


namespace solve_inequality_l235_235388

open Real

theorem solve_inequality (x : ℝ) : (x ≠ 3) ∧ (x * (x + 1) / (x - 3) ^ 2 ≥ 9) ↔ (2.13696 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4.73804) :=
by
  sorry

end solve_inequality_l235_235388


namespace amoeba_doubling_time_l235_235318

theorem amoeba_doubling_time (H1 : ∀ t : ℕ, t = 60 → 2^(t / 3) = 2^20) :
  ∀ t : ℕ, 2 * 2^(t / 3) = 2^20 → t = 57 :=
by
  intro t
  intro H2
  sorry

end amoeba_doubling_time_l235_235318


namespace initial_card_count_l235_235639

theorem initial_card_count (r b : ℕ) (h₁ : (r : ℝ)/(r + b) = 1/4)
    (h₂ : (r : ℝ)/(r + (b + 6)) = 1/6) : r + b = 12 :=
by
  sorry

end initial_card_count_l235_235639


namespace speed_difference_l235_235038

theorem speed_difference (h_cyclist : 88 / 8 = 11) (h_car : 48 / 8 = 6) :
  (11 - 6 = 5) :=
by
  sorry

end speed_difference_l235_235038


namespace sum_of_ages_is_nineteen_l235_235472

-- Definitions representing the conditions
def Bella_age : ℕ := 5
def Brother_is_older : ℕ := 9
def Brother_age : ℕ := Bella_age + Brother_is_older
def Sum_of_ages : ℕ := Bella_age + Brother_age

-- Mathematical statement (theorem) to be proved
theorem sum_of_ages_is_nineteen : Sum_of_ages = 19 := by
  sorry

end sum_of_ages_is_nineteen_l235_235472


namespace shara_age_l235_235966

-- Definitions derived from conditions
variables (S : ℕ) (J : ℕ)

-- Jaymee's age is twice Shara's age plus 2
def jaymee_age_relation : Prop := J = 2 * S + 2

-- Jaymee's age is given as 22
def jaymee_age : Prop := J = 22

-- The proof problem to prove Shara's age equals 10
theorem shara_age (h1 : jaymee_age_relation S J) (h2 : jaymee_age J) : S = 10 :=
by 
  sorry

end shara_age_l235_235966


namespace inequality_proof_l235_235072

theorem inequality_proof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  (x * y / Real.sqrt (x * y + y * z) + y * z / Real.sqrt (y * z + z * x) + z * x / Real.sqrt (z * x + x * y)) 
  ≤ (Real.sqrt 2) / 2 :=
by
  sorry

end inequality_proof_l235_235072


namespace find_five_digit_number_l235_235279

theorem find_five_digit_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∃ rev_n : ℕ, rev_n = (n % 10) * 10000 + (n / 10 % 10) * 1000 + (n / 100 % 10) * 100 + (n / 1000 % 10) * 10 + (n / 10000) ∧ 9 * n = rev_n) ∧ n = 10989 :=
  sorry

end find_five_digit_number_l235_235279


namespace product_of_two_numbers_l235_235758

theorem product_of_two_numbers (x y : ℕ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 221) : x * y = 60 := sorry

end product_of_two_numbers_l235_235758


namespace complex_number_equation_l235_235510

theorem complex_number_equation
  (f : ℂ → ℂ)
  (z : ℂ)
  (h : f (i - z) = 2 * z - i) :
  (1 - i) * f (2 - i) = -1 + 7 * i := by
  sorry

end complex_number_equation_l235_235510


namespace find_y_l235_235167

theorem find_y (x y : ℝ) (h1 : x^2 = 2 * y - 6) (h2 : x = 7) : y = 55 / 2 :=
by
  sorry

end find_y_l235_235167


namespace time_to_pick_up_dog_l235_235562

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

end time_to_pick_up_dog_l235_235562


namespace jessica_final_balance_l235_235658

variable {original_balance current_balance final_balance withdrawal1 withdrawal2 deposit1 deposit2 : ℝ}

theorem jessica_final_balance:
  (2 / 5) * original_balance = 200 → 
  current_balance = original_balance - 200 → 
  withdrawal1 = (1 / 3) * current_balance → 
  current_balance - withdrawal1 = current_balance - (1 / 3 * current_balance) → 
  deposit1 = (1 / 5) * (current_balance - (1 / 3 * current_balance)) → 
  final_balance = (current_balance - (1 / 3 * current_balance)) + deposit1 → 
  deposit2 / 7 * 3 = final_balance - (current_balance - (1 / 3 * current_balance) + deposit1) → 
  (final_balance + deposit2) = 420 :=
sorry

end jessica_final_balance_l235_235658


namespace option_d_always_holds_l235_235661

theorem option_d_always_holds (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b := by
  sorry

end option_d_always_holds_l235_235661


namespace box_dimension_min_sum_l235_235513

theorem box_dimension_min_sum :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 := by
  sorry

end box_dimension_min_sum_l235_235513


namespace cone_base_circumference_l235_235160

theorem cone_base_circumference (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 240) :
  (2 / 3) * (2 * Real.pi * r) = 8 * Real.pi :=
by
  have circle_circumference : ℝ := 2 * Real.pi * r
  sorry

end cone_base_circumference_l235_235160


namespace solvable_eq_l235_235892

theorem solvable_eq (x : ℝ) :
    Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6 →
    (x = 2 ∨ x = -2) :=
by
  sorry

end solvable_eq_l235_235892


namespace gcd_A_C_gcd_B_C_l235_235519

def A : ℕ := 177^5 + 30621 * 173^3 - 173^5
def B : ℕ := 173^5 + 30621 * 177^3 - 177^5
def C : ℕ := 173^4 + 30621^2 + 177^4

theorem gcd_A_C : Nat.gcd A C = 30637 := sorry

theorem gcd_B_C : Nat.gcd B C = 30637 := sorry

end gcd_A_C_gcd_B_C_l235_235519


namespace decreasing_function_range_l235_235019

theorem decreasing_function_range {a : ℝ} (h1 : ∀ x y : ℝ, x < y → (1 - 2 * a)^x > (1 - 2 * a)^y) : 
    0 < a ∧ a < 1 / 2 :=
by
  sorry

end decreasing_function_range_l235_235019


namespace average_weight_l235_235404

theorem average_weight 
  (n₁ n₂ : ℕ) 
  (avg₁ avg₂ total_avg : ℚ) 
  (h₁ : n₁ = 24) 
  (h₂ : n₂ = 8)
  (h₃ : avg₁ = 50.25)
  (h₄ : avg₂ = 45.15)
  (h₅ : total_avg = 48.975) :
  ( (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = total_avg ) :=
sorry

end average_weight_l235_235404


namespace ratio_x_y_l235_235935

theorem ratio_x_y (x y : ℝ) (h : (1/x - 1/y) / (1/x + 1/y) = 2023) : (x + y) / (x - y) = -1 := 
by
  sorry

end ratio_x_y_l235_235935


namespace coordinates_equality_l235_235180

theorem coordinates_equality (a b : ℤ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : a + b = -1 :=
by 
  sorry

end coordinates_equality_l235_235180


namespace probability_point_not_above_x_axis_l235_235960

theorem probability_point_not_above_x_axis (A B C D : ℝ × ℝ) :
  A = (9, 4) →
  B = (3, -2) →
  C = (-3, -2) →
  D = (3, 4) →
  (1 / 2 : ℚ) = 1 / 2 := 
by 
  intros hA hB hC hD 
  sorry

end probability_point_not_above_x_axis_l235_235960


namespace greatest_xy_value_l235_235959

theorem greatest_xy_value :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 5 * y = 200 ∧ x * y = 285 :=
by 
  sorry

end greatest_xy_value_l235_235959


namespace ages_sum_l235_235138

theorem ages_sum (Beckett_age Olaf_age Shannen_age Jack_age : ℕ) 
  (h1 : Beckett_age = 12) 
  (h2 : Olaf_age = Beckett_age + 3) 
  (h3 : Shannen_age = Olaf_age - 2) 
  (h4 : Jack_age = 2 * Shannen_age + 5) : 
  Beckett_age + Olaf_age + Shannen_age + Jack_age = 71 := 
by
  sorry

end ages_sum_l235_235138


namespace jill_second_bus_time_l235_235842

-- Define constants representing the times
def wait_time_first_bus : ℕ := 12
def ride_time_first_bus : ℕ := 30

-- Define a function to calculate the total time for the first bus
def total_time_first_bus (wait : ℕ) (ride : ℕ) : ℕ :=
  wait + ride

-- Define a function to calculate the time for the second bus
def time_second_bus (total_first_bus_time : ℕ) : ℕ :=
  total_first_bus_time / 2

-- The theorem to prove
theorem jill_second_bus_time : 
  time_second_bus (total_time_first_bus wait_time_first_bus ride_time_first_bus) = 21 := by
  sorry

end jill_second_bus_time_l235_235842


namespace find_x_range_l235_235944

def tight_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, 0 < n → 1/2 ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

theorem find_x_range
  (a : ℕ → ℝ)
  (h_tight : tight_sequence a)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3 / 2)
  (h3 : ∃ x, a 3 = x)
  (h4 : a 4 = 4) :
  ∃ x, (2 : ℝ) ≤ x ∧ x ≤ (3 : ℝ) :=
sorry

end find_x_range_l235_235944


namespace quadratic_inequality_real_solutions_l235_235086

theorem quadratic_inequality_real_solutions (c : ℝ) (h1 : 0 < c) (h2 : c < 16) :
  ∃ x : ℝ, x^2 - 8*x + c < 0 :=
sorry

end quadratic_inequality_real_solutions_l235_235086


namespace question_1_question_2_question_3_l235_235926

variable (a b : ℝ)

-- (a * b)^n = a^n * b^n for natural numbers n
theorem question_1 (n : ℕ) : (a * b)^n = a^n * b^n := sorry

-- Calculate 2^5 * (-1/2)^5
theorem question_2 : 2^5 * (-1/2)^5 = -1 := sorry

-- Calculate (-0.125)^2022 * 2^2021 * 4^2020
theorem question_3 : (-0.125)^2022 * 2^2021 * 4^2020 = 1 / 32 := sorry

end question_1_question_2_question_3_l235_235926


namespace slices_of_bread_left_l235_235194

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end slices_of_bread_left_l235_235194


namespace number_of_valid_pairs_is_34_l235_235486

noncomputable def countValidPairs : Nat :=
  let primes : List Nat := [2, 3, 5, 7, 11, 13]
  let nonprimes : List Nat := [1, 4, 6, 8, 9, 10, 12, 14, 15]
  let countForN (n : Nat) : Nat :=
    match n with
    | 2 => Nat.choose 8 1
    | 3 => Nat.choose 7 2
    | 5 => Nat.choose 5 4
    | _ => 0
  primes.map countForN |>.sum

theorem number_of_valid_pairs_is_34 : countValidPairs = 34 :=
  sorry

end number_of_valid_pairs_is_34_l235_235486


namespace final_sum_after_50_passes_l235_235155

theorem final_sum_after_50_passes
  (particip: ℕ) 
  (num_passes: particip = 50) 
  (init_disp: ℕ → ℤ) 
  (initial_condition : init_disp 0 = 1 ∧ init_disp 1 = 0 ∧ init_disp 2 = -1)
  (operations: Π (i : ℕ), 
    (init_disp 0 = 1 →
    init_disp 1 = 0 →
    (i % 2 = 0 → init_disp 2 = -1) →
    (i % 2 = 1 → init_disp 2 = 1))
  )
  : init_disp 0 + init_disp 1 + init_disp 2 = 0 :=
by
  sorry

end final_sum_after_50_passes_l235_235155


namespace three_digit_even_sum_12_l235_235028

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l235_235028


namespace binom_2n_2_eq_n_2n_minus_1_l235_235392

theorem binom_2n_2_eq_n_2n_minus_1 (n : ℕ) (h : n > 0) : 
  (Nat.choose (2 * n) 2) = n * (2 * n - 1) := 
sorry

end binom_2n_2_eq_n_2n_minus_1_l235_235392


namespace binomial_coefficient_12_4_l235_235023

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

theorem binomial_coefficient_12_4 : binomial_coefficient 12 4 = 495 := by
  sorry

end binomial_coefficient_12_4_l235_235023


namespace water_added_16_l235_235487

theorem water_added_16 (W : ℝ) 
  (h1 : ∃ W, 24 * 0.90 = 0.54 * (24 + W)) : 
  W = 16 := 
by {
  sorry
}

end water_added_16_l235_235487


namespace hari_joined_after_5_months_l235_235423

noncomputable def praveen_investment := 3780 * 12
noncomputable def hari_investment (x : ℕ) := 9720 * (12 - x)

theorem hari_joined_after_5_months :
  ∃ (x : ℕ), (praveen_investment : ℝ) / (hari_investment x) = (2:ℝ) / 3 ∧ x = 5 :=
by {
  sorry
}

end hari_joined_after_5_months_l235_235423


namespace person2_speed_l235_235444

variables (v_1 : ℕ) (v_2 : ℕ)

def meet_time := 4
def catch_up_time := 16

def meet_equation : Prop := v_1 + v_2 = 22
def catch_up_equation : Prop := v_2 - v_1 = 4

theorem person2_speed :
  meet_equation v_1 v_2 → catch_up_equation v_1 v_2 →
  v_1 = 6 → v_2 = 10 :=
by
  intros h1 h2 h3
  sorry

end person2_speed_l235_235444


namespace Nicole_cards_l235_235676

variables (N : ℕ)

-- Conditions from step A
def Cindy_collected (N : ℕ) : ℕ := 2 * N
def Nicole_and_Cindy_combined (N : ℕ) : ℕ := N + Cindy_collected N
def Rex_collected (N : ℕ) : ℕ := (Nicole_and_Cindy_combined N) / 2
def Rex_cards_each (N : ℕ) : ℕ := Rex_collected N / 4

-- Question: How many cards did Nicole collect? Answer: N = 400
theorem Nicole_cards (N : ℕ) (h : Rex_cards_each N = 150) : N = 400 :=
sorry

end Nicole_cards_l235_235676


namespace sum_of_first_60_digits_l235_235177

-- Define the repeating sequence and the number of repetitions
def repeating_sequence : List ℕ := [0, 0, 0, 1]
def repetitions : ℕ := 15

-- Define the sum of first n elements of a repeating sequence
def sum_repeating_sequence (seq : List ℕ) (n : ℕ) : ℕ :=
  let len := seq.length
  let complete_cycles := n / len
  let remaining_digits := n % len
  let sum_complete_cycles := complete_cycles * seq.sum
  let sum_remaining_digits := (seq.take remaining_digits).sum
  sum_complete_cycles + sum_remaining_digits

-- Prove the specific case for 60 digits
theorem sum_of_first_60_digits : sum_repeating_sequence repeating_sequence 60 = 15 := 
by
  sorry

end sum_of_first_60_digits_l235_235177


namespace merchant_marked_price_l235_235557

variable (L C M S : ℝ)

theorem merchant_marked_price :
  (C = 0.8 * L) → (C = 0.8 * S) → (S = 0.8 * M) → (M = 1.25 * L) :=
by
  sorry

end merchant_marked_price_l235_235557


namespace value_of_y_l235_235626

theorem value_of_y 
  (x y : ℤ) 
  (h1 : x - y = 10) 
  (h2 : x + y = 8) 
  : y = -1 := by
  sorry

end value_of_y_l235_235626


namespace width_of_playground_is_250_l235_235071

noncomputable def total_area_km2 : ℝ := 0.6
def num_playgrounds : ℕ := 8
def length_of_playground_m : ℝ := 300

theorem width_of_playground_is_250 :
  let total_area_m2 := total_area_km2 * 1000000
  let area_of_one_playground := total_area_m2 / num_playgrounds
  let width_of_playground := area_of_one_playground / length_of_playground_m
  width_of_playground = 250 := by
  sorry

end width_of_playground_is_250_l235_235071


namespace increasing_function_range_l235_235545

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 ≤ a ∧ a < 3 :=
  sorry

end increasing_function_range_l235_235545


namespace wasting_water_notation_l235_235882

theorem wasting_water_notation (saving_wasting : ℕ → ℤ)
  (h_pos : saving_wasting 30 = 30) :
  saving_wasting 10 = -10 :=
by
  sorry

end wasting_water_notation_l235_235882


namespace tom_savings_by_having_insurance_l235_235034

noncomputable def insurance_cost_per_month : ℝ := 20
noncomputable def total_months : ℕ := 24
noncomputable def surgery_cost : ℝ := 5000
noncomputable def insurance_coverage_rate : ℝ := 0.80

theorem tom_savings_by_having_insurance :
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  savings = 3520 :=
by
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  sorry

end tom_savings_by_having_insurance_l235_235034


namespace a_cubed_value_l235_235845

theorem a_cubed_value (a b : ℝ) (k : ℝ) (h1 : a^3 * b^2 = k) (h2 : a = 5) (h3 : b = 2) : 
  ∃ (a : ℝ), (64 * a^3 = 500) → (a^3 = 125 / 16) :=
by
  sorry

end a_cubed_value_l235_235845


namespace bracelet_ratio_l235_235225

-- Definition of the conditions
def initial_bingley_bracelets : ℕ := 5
def kelly_bracelets_given : ℕ := 16 / 4
def total_bracelets_after_receiving := initial_bingley_bracelets + kelly_bracelets_given
def bingley_remaining_bracelets : ℕ := 6
def bingley_bracelets_given := total_bracelets_after_receiving - bingley_remaining_bracelets

-- Lean 4 Statement
theorem bracelet_ratio : bingley_bracelets_given * 3 = total_bracelets_after_receiving := by
  sorry

end bracelet_ratio_l235_235225


namespace expand_expression_l235_235339

theorem expand_expression (x y : ℝ) : 
  5 * (4 * x^2 + 3 * x * y - 4) = 20 * x^2 + 15 * x * y - 20 := 
by 
  sorry

end expand_expression_l235_235339


namespace champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l235_235490

-- Define the structure and relationship between teams in the tournament
structure Tournament (Team : Type) :=
  (competes : Team → Team → Prop) -- teams play against each other
  (no_ties : ∀ A B : Team, (competes A B ∧ ¬competes B A) ∨ (competes B A ∧ ¬competes A B)) -- no ties
  (superior : Team → Team → Prop) -- superiority relationship
  (superior_def : ∀ A B : Team, superior A B ↔ (competes A B ∧ ¬competes B A) ∨ (∃ C : Team, superior A C ∧ superior C B))

-- The main theorem based on the given questions
theorem champion_team_exists {Team : Type} (tournament : Tournament Team) :
  ∃ champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B) :=
  sorry

theorem unique_champion_wins_all {Team : Type} (tournament : Tournament Team)
  (h : ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B)) :
  ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B ∧ tournament.competes champion B ∧ ¬tournament.competes B champion) :=
  sorry

theorem not_exactly_two_champions {Team : Type} (tournament : Tournament Team) :
  ¬∃ A B : Team, A ≠ B ∧ (∀ C : Team, C ≠ A → tournament.superior A C) ∧ (∀ C : Team, C ≠ B → tournament.superior B C) :=
  sorry

end champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l235_235490


namespace roots_of_quadratic_l235_235681

theorem roots_of_quadratic (a b : ℝ) (h : a ≠ 0) (h1 : a + b = 0) :
  ∀ x, (a * x^2 + b * x = 0) → (x = 0 ∨ x = 1) := 
by
  sorry

end roots_of_quadratic_l235_235681


namespace total_wheels_correct_l235_235722

def total_wheels (bicycles cars motorcycles tricycles quads : ℕ) 
(missing_bicycle_wheels broken_car_wheels missing_motorcycle_wheels : ℕ) : ℕ :=
  let bicycles_wheels := (bicycles - missing_bicycle_wheels) * 2 + missing_bicycle_wheels
  let cars_wheels := (cars - broken_car_wheels) * 4 + broken_car_wheels * 3
  let motorcycles_wheels := (motorcycles - missing_motorcycle_wheels) * 2
  let tricycles_wheels := tricycles * 3
  let quads_wheels := quads * 4
  bicycles_wheels + cars_wheels + motorcycles_wheels + tricycles_wheels + quads_wheels

theorem total_wheels_correct : total_wheels 25 15 8 3 2 5 2 1 = 134 := 
  by sorry

end total_wheels_correct_l235_235722


namespace weight_ratio_l235_235919

noncomputable def weight_ratio_proof : Prop :=
  ∃ (R S : ℝ), 
  (R + S = 72) ∧ 
  (1.10 * R + 1.17 * S = 82.8) ∧ 
  (R / S = 1 / 2.5)

theorem weight_ratio : weight_ratio_proof := 
  by
    sorry

end weight_ratio_l235_235919


namespace linear_function_quadrant_l235_235815

theorem linear_function_quadrant (x y : ℝ) : 
  y = 2 * x - 3 → ¬ ((x < 0 ∧ y > 0)) := 
sorry

end linear_function_quadrant_l235_235815


namespace annual_decrease_rate_l235_235976

theorem annual_decrease_rate (r : ℝ) 
  (h1 : 15000 * (1 - r / 100)^2 = 9600) : 
  r = 20 := 
sorry

end annual_decrease_rate_l235_235976


namespace find_integer_triples_l235_235293

theorem find_integer_triples (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ 
  (x = 668 ∧ y = 668 ∧ z = 667) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 667 ∧ y = 668 ∧ z = 668) :=
by sorry

end find_integer_triples_l235_235293


namespace intersection_M_N_l235_235850

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 10 }
def N : Set ℝ := { x | x > 7 ∨ x < 1 }
def MN_intersection : Set ℝ := { x | (-1 ≤ x ∧ x < 1) ∨ (7 < x ∧ x ≤ 10) }

theorem intersection_M_N :
  M ∩ N = MN_intersection :=
by
  sorry

end intersection_M_N_l235_235850


namespace marble_158th_is_gray_l235_235732

def marble_color (n : ℕ) : String :=
  if (n % 12 < 5) then "gray"
  else if (n % 12 < 9) then "white"
  else "black"

theorem marble_158th_is_gray : marble_color 157 = "gray" := 
by
  sorry

end marble_158th_is_gray_l235_235732


namespace olivers_friend_gave_l235_235611

variable (initial_amount saved_amount spent_frisbee spent_puzzle final_amount : ℕ) 

theorem olivers_friend_gave (h1 : initial_amount = 9) 
                           (h2 : saved_amount = 5) 
                           (h3 : spent_frisbee = 4) 
                           (h4 : spent_puzzle = 3) 
                           (h5 : final_amount = 15) : 
                           final_amount - (initial_amount + saved_amount - (spent_frisbee + spent_puzzle)) = 8 := 
by 
  sorry

end olivers_friend_gave_l235_235611


namespace ratio_IM_IN_l235_235068

noncomputable def compute_ratio (IA IB IC ID : ℕ) (M N : ℕ) : ℚ :=
  (IA * IC : ℚ) / (IB * ID : ℚ)

theorem ratio_IM_IN (IA IB IC ID : ℕ) (hIA : IA = 12) (hIB : IB = 16) (hIC : IC = 14) (hID : ID = 11) :
  compute_ratio IA IB IC ID = 21 / 22 := by
  rw [hIA, hIB, hIC, hID]
  sorry

end ratio_IM_IN_l235_235068


namespace correct_ordering_of_powers_l235_235051

theorem correct_ordering_of_powers :
  (6 ^ 8) < (3 ^ 15) ∧ (3 ^ 15) < (8 ^ 10) :=
by
  -- Define the expressions for each power
  let a := (8 : ℕ) ^ 10
  let b := (3 : ℕ) ^ 15
  let c := (6 : ℕ) ^ 8
  
  -- To utilize the values directly in inequalities
  have h1 : (c < b) := sorry -- Proof that 6^8 < 3^15
  have h2 : (b < a) := sorry -- Proof that 3^15 < 8^10

  exact ⟨h1, h2⟩ -- Conjunction of h1 and h2 to show 6^8 < 3^15 < 8^10

end correct_ordering_of_powers_l235_235051


namespace parametric_plane_equation_l235_235317

-- Definitions to translate conditions
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ := (2 + 2 * s - t, 4 - 2 * s, 6 + s - 3 * t)

-- Theorem to prove the equivalence to plane equation
theorem parametric_plane_equation : 
  ∃ A B C D, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧ 
  (∀ s t x y z, parametric_plane s t = (x, y, z) → 6 * x - 5 * y - 2 * z + 20 = 0) := by
  sorry

end parametric_plane_equation_l235_235317


namespace sale_in_third_month_l235_235719

theorem sale_in_third_month (
  f1 f2 f4 f5 f6 average : ℕ
) (h1 : f1 = 7435) 
  (h2 : f2 = 7927) 
  (h4 : f4 = 8230) 
  (h5 : f5 = 7562) 
  (h6 : f6 = 5991) 
  (havg : average = 7500) :
  ∃ f3, f3 = 7855 ∧ f1 + f2 + f3 + f4 + f5 + f6 = average * 6 :=
by {
  sorry
}

end sale_in_third_month_l235_235719


namespace part1_part2_l235_235149

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.sin (x + Real.pi / 3) - 1

theorem part1 : f (5 * Real.pi / 6) = -2 := by
  sorry

variables {A : ℝ} (hA1 : A > 0) (hA2 : A ≤ Real.pi / 3) (hFA : f A = 8 / 5)

theorem part2 (h : A > 0 ∧ A ≤ Real.pi / 3 ∧ f A = 8 / 5) : f (A + Real.pi / 4) = 6 / 5 :=
by
  sorry

end part1_part2_l235_235149


namespace f_2a_eq_3_l235_235560

noncomputable def f (x : ℝ) : ℝ := 2^x + 1 / 2^x

theorem f_2a_eq_3 (a : ℝ) (h : f a = Real.sqrt 5) : f (2 * a) = 3 := by
  sorry

end f_2a_eq_3_l235_235560


namespace three_tenths_of_number_l235_235221

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 15) : (3/10) * x = 54 :=
by
  sorry

end three_tenths_of_number_l235_235221


namespace daisies_given_l235_235323

theorem daisies_given (S : ℕ) (h : (5 + S) / 2 = 7) : S = 9 := by
  sorry

end daisies_given_l235_235323


namespace smallest_possible_c_l235_235077

theorem smallest_possible_c 
  (a b c : ℕ) (hp : a > 0 ∧ b > 0 ∧ c > 0) 
  (hg : b^2 = a * c) 
  (ha : 2 * c = a + b) : 
  c = 2 :=
by
  sorry

end smallest_possible_c_l235_235077


namespace evaluate_expression_l235_235265

theorem evaluate_expression :
  ((gcd 54 42 |> lcm 36) * (gcd 78 66 |> gcd 90) + (lcm 108 72 |> gcd 66 |> gcd 84)) = 24624 := by
  sorry

end evaluate_expression_l235_235265
