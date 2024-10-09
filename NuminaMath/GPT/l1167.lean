import Mathlib

namespace complex_number_real_implies_m_is_5_l1167_116782

theorem complex_number_real_implies_m_is_5 (m : ℝ) (h : m^2 - 2 * m - 15 = 0) : m = 5 :=
  sorry

end complex_number_real_implies_m_is_5_l1167_116782


namespace dad_additional_money_l1167_116763

-- Define the conditions in Lean
def daily_savings : ℕ := 35
def days : ℕ := 7
def total_savings_before_doubling := daily_savings * days
def doubled_savings := 2 * total_savings_before_doubling
def total_amount_after_7_days : ℕ := 500

-- Define the theorem to prove
theorem dad_additional_money : (total_amount_after_7_days - doubled_savings) = 10 := by
  sorry

end dad_additional_money_l1167_116763


namespace problem_statement_l1167_116737

noncomputable def f (a x : ℝ) := a * (x ^ 2 + 1) + Real.log x

theorem problem_statement (a m : ℝ) (x : ℝ) 
  (h_a : -4 < a) (h_a' : a < -2) (h_x1 : 1 ≤ x) (h_x2 : x ≤ 3) :
  (m * a - f a x > a ^ 2) ↔ (m ≤ -2) :=
by
  sorry

end problem_statement_l1167_116737


namespace ratio_pages_l1167_116726

theorem ratio_pages (pages_Selena pages_Harry : ℕ) (h₁ : pages_Selena = 400) (h₂ : pages_Harry = 180) : 
  pages_Harry / pages_Selena = 9 / 20 := 
by
  -- proof goes here
  sorry

end ratio_pages_l1167_116726


namespace limit_at_2_l1167_116766

noncomputable def delta (ε : ℝ) : ℝ := ε / 3

theorem limit_at_2 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x : ℝ, (0 < |x - 2| ∧ |x - 2| < δ) → |(3 * x^2 - 5 * x - 2) / (x - 2) - 7| < ε :=
by
  let δ := delta ε
  have hδ : δ > 0 := by
    sorry
  use δ, hδ
  intros x hx
  sorry

end limit_at_2_l1167_116766


namespace solve_quadratics_l1167_116769

theorem solve_quadratics (p q u v : ℤ)
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q)
  (h2 : u ≠ 0 ∧ v ≠ 0 ∧ u ≠ v)
  (h3 : p + q = -u)
  (h4 : pq = -v)
  (h5 : u + v = -p)
  (h6 : uv = -q) :
  p = -1 ∧ q = 2 ∧ u = -1 ∧ v = 2 :=
by {
  sorry
}

end solve_quadratics_l1167_116769


namespace abs_neg_number_l1167_116704

theorem abs_neg_number : abs (-2023) = 2023 := sorry

end abs_neg_number_l1167_116704


namespace molecular_weight_1_mole_l1167_116741

-- Define the molecular weight of 3 moles
def molecular_weight_3_moles : ℕ := 222

-- Prove that the molecular weight of 1 mole is 74 given the molecular weight of 3 moles
theorem molecular_weight_1_mole (mw3 : ℕ) (h : mw3 = 222) : mw3 / 3 = 74 :=
by
  sorry

end molecular_weight_1_mole_l1167_116741


namespace inequality_proof_l1167_116789

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l1167_116789


namespace range_of_sum_l1167_116743

theorem range_of_sum (x y : ℝ) (h : 9 * x^2 + 16 * y^2 = 144) : 
  ∃ a b : ℝ, (x + y + 10 ≥ a) ∧ (x + y + 10 ≤ b) ∧ a = 5 ∧ b = 15 := 
sorry

end range_of_sum_l1167_116743


namespace johns_age_l1167_116717

-- Define variables for ages of John and Matt
variables (J M : ℕ)

-- Define the conditions based on the problem statement
def condition1 : Prop := M = 4 * J - 3
def condition2 : Prop := J + M = 52

-- The goal: prove that John is 11 years old
theorem johns_age (J M : ℕ) (h1 : condition1 J M) (h2 : condition2 J M) : J = 11 := by
  -- proof will go here
  sorry

end johns_age_l1167_116717


namespace a_range_iff_l1167_116793

theorem a_range_iff (a x : ℝ) (h1 : x < 3) (h2 : (a - 1) * x < a + 3) : 
  1 ≤ a ∧ a < 3 := 
by
  sorry

end a_range_iff_l1167_116793


namespace largest_composite_in_five_consecutive_ints_l1167_116705

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_of_five_composite_ints : ℕ :=
  36

theorem largest_composite_in_five_consecutive_ints (a b c d e : ℕ) :
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 ∧ 
  ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a = 32 ∧ b = 33 ∧ c = 34 ∧ d = 35 ∧ e = 36 →
  e = largest_of_five_composite_ints :=
by 
  sorry

end largest_composite_in_five_consecutive_ints_l1167_116705


namespace exists_a_b_l1167_116774

theorem exists_a_b (n : ℕ) (hn : 0 < n) : ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := by
  sorry

end exists_a_b_l1167_116774


namespace conversion_proofs_l1167_116784

-- Define the necessary constants for unit conversion
def cm_to_dm2 (cm2: ℚ) : ℚ := cm2 / 100
def m3_to_dm3 (m3: ℚ) : ℚ := m3 * 1000
def dm3_to_liters (dm3: ℚ) : ℚ := dm3
def liters_to_ml (liters: ℚ) : ℚ := liters * 1000

theorem conversion_proofs :
  (cm_to_dm2 628 = 6.28) ∧
  (m3_to_dm3 4.5 = 4500) ∧
  (dm3_to_liters 3.6 = 3.6) ∧
  (liters_to_ml 0.6 = 600) :=
by
  sorry

end conversion_proofs_l1167_116784


namespace sequence_periodicity_l1167_116708

theorem sequence_periodicity (a : ℕ → ℤ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 = 6) 
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n): 
  a 2015 = -6 := 
sorry

end sequence_periodicity_l1167_116708


namespace int_solutions_exist_for_x2_plus_15y2_eq_4n_l1167_116750

theorem int_solutions_exist_for_x2_plus_15y2_eq_4n (n : ℕ) (hn : n > 0) : 
  ∃ S : Finset (ℤ × ℤ), S.card ≥ n ∧ ∀ (xy : ℤ × ℤ), xy ∈ S → xy.1^2 + 15 * xy.2^2 = 4^n :=
by
  sorry

end int_solutions_exist_for_x2_plus_15y2_eq_4n_l1167_116750


namespace b_2016_value_l1167_116764

theorem b_2016_value : 
  ∃ (a b : ℕ → ℝ), 
    a 1 = 1 / 2 ∧ 
    (∀ n : ℕ, 0 < n → a n + b n = 1) ∧
    (∀ n : ℕ, 0 < n → b (n + 1) = b n / (1 - (a n)^2)) → 
    b 2016 = 2016 / 2017 :=
by
  sorry

end b_2016_value_l1167_116764


namespace remainder_of_product_divided_by_10_l1167_116791

theorem remainder_of_product_divided_by_10 :
  let a := 2457
  let b := 6273
  let c := 91409
  (a * b * c) % 10 = 9 :=
by
  sorry

end remainder_of_product_divided_by_10_l1167_116791


namespace students_remaining_after_fourth_stop_l1167_116719

variable (n : ℕ)
variable (frac : ℚ)

def initial_students := (64 : ℚ)
def fraction_remaining := (2/3 : ℚ)

theorem students_remaining_after_fourth_stop : 
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  after_fourth_stop = (1024 / 81) := 
by 
  sorry

end students_remaining_after_fourth_stop_l1167_116719


namespace share_difference_l1167_116773

variables {x : ℕ}

theorem share_difference (h1: 12 * x - 7 * x = 5000) : 7 * x - 3 * x = 4000 :=
by
  sorry

end share_difference_l1167_116773


namespace A_in_terms_of_B_l1167_116700

-- Definitions based on conditions
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^3
def g (B x : ℝ) : ℝ := B * x^2

-- Theorem statement
theorem A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) : A = 3 * B / 16 :=
by
  -- Proof omitted
  sorry

end A_in_terms_of_B_l1167_116700


namespace graphs_symmetric_respect_to_x_equals_1_l1167_116794

-- Define the function f
variable (f : ℝ → ℝ)

-- Define g(x) = f(x-1)
def g (x : ℝ) : ℝ := f (x - 1)

-- Define h(x) = f(1 - x)
def h (x : ℝ) : ℝ := f (1 - x)

-- The theorem that their graphs are symmetric with respect to the line x = 1
theorem graphs_symmetric_respect_to_x_equals_1 :
  ∀ x : ℝ, g f x = h f x ↔ f x = f (2 - x) :=
sorry

end graphs_symmetric_respect_to_x_equals_1_l1167_116794


namespace polynomial_divisibility_l1167_116796

theorem polynomial_divisibility (r s : ℝ) :
  (∀ x, 10 * x^4 - 15 * x^3 - 55 * x^2 + 85 * x - 51 = 10 * (x - r)^2 * (x - s)) →
  r = 3 / 2 ∧ s = -5 / 2 :=
by
  intros h
  sorry

end polynomial_divisibility_l1167_116796


namespace determine_a_l1167_116718

noncomputable def imaginary_unit : ℂ := Complex.I

def is_on_y_axis (z : ℂ) : Prop :=
  z.re = 0

theorem determine_a (a : ℝ) : 
  is_on_y_axis (⟨(a - 3 * imaginary_unit.re), -(a - 3 * imaginary_unit.im)⟩ / ⟨(1 - imaginary_unit.re), -(1 - imaginary_unit.im)⟩) → 
  a = -3 :=
sorry

end determine_a_l1167_116718


namespace train_length_correct_l1167_116723

noncomputable def speed_kmph : ℝ := 60
noncomputable def time_sec : ℝ := 6

-- Conversion factor from km/hr to m/s
noncomputable def conversion_factor := (1000 : ℝ) / 3600

-- Speed in m/s
noncomputable def speed_mps := speed_kmph * conversion_factor

-- Length of the train
noncomputable def train_length := speed_mps * time_sec

theorem train_length_correct :
  train_length = 100.02 :=
by
  sorry

end train_length_correct_l1167_116723


namespace boy_present_age_l1167_116727

theorem boy_present_age : ∃ x : ℕ, (x + 4 = 2 * (x - 6)) ∧ x = 16 := by
  sorry

end boy_present_age_l1167_116727


namespace num_true_statements_l1167_116786

theorem num_true_statements :
  (∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) ∧
  (∀ x y a, a ≠ 0 → (a^2 * x ≥ a^2 * y → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x ≥ y → x / a^2 ≥ y / a^2)) →
  ((∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) →
   (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y))) :=
sorry

end num_true_statements_l1167_116786


namespace negation_of_exists_abs_lt_one_l1167_116779

theorem negation_of_exists_abs_lt_one :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end negation_of_exists_abs_lt_one_l1167_116779


namespace square_distance_from_B_to_center_l1167_116759

-- Defining the conditions
structure Circle (α : Type _) :=
(center : α × α)
(radius2 : ℝ)

structure Point (α : Type _) :=
(x : α)
(y : α)

def is_right_angle (a b c : Point ℝ) : Prop :=
(b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) = 0

noncomputable def distance2 (p1 p2 : Point ℝ) : ℝ :=
(p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

theorem square_distance_from_B_to_center :
  ∀ (c : Circle ℝ) (A B C : Point ℝ), 
    c.radius2 = 65 →
    distance2 A B = 49 →
    distance2 B C = 9 →
    is_right_angle A B C →
    distance2 B {x:=0, y:=0} = 80 := 
by
  intros c A B C h_radius h_AB h_BC h_right_angle
  sorry

end square_distance_from_B_to_center_l1167_116759


namespace max_area_of_rectangle_with_perimeter_60_l1167_116756

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l1167_116756


namespace interest_calculated_years_l1167_116738

variable (P T : ℝ)

-- Given conditions
def principal_sum_positive : Prop := P > 0
def simple_interest_condition : Prop := (P * 5 * T) / 100 = P / 5

-- Theorem statement
theorem interest_calculated_years (h1 : principal_sum_positive P) (h2 : simple_interest_condition P T) : T = 4 :=
  sorry

end interest_calculated_years_l1167_116738


namespace smallest_value_of_expression_l1167_116778

noncomputable def f (x : ℝ) : ℝ := x^4 + 14*x^3 + 52*x^2 + 56*x + 16

theorem smallest_value_of_expression :
  ∀ z : Fin 4 → ℝ, (∀ i, f (z i) = 0) → 
  ∃ (a b c d : Fin 4), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ d ≠ b ∧ a ≠ c ∧ 
  |(z a * z b) + (z c * z d)| = 8 :=
by
  sorry

end smallest_value_of_expression_l1167_116778


namespace Faye_total_pencils_l1167_116752

def pencils_per_row : ℕ := 8
def number_of_rows : ℕ := 4
def total_pencils : ℕ := pencils_per_row * number_of_rows

theorem Faye_total_pencils : total_pencils = 32 := by
  sorry

end Faye_total_pencils_l1167_116752


namespace red_side_probability_l1167_116790

theorem red_side_probability
  (num_cards : ℕ)
  (num_black_black : ℕ)
  (num_black_red : ℕ)
  (num_red_red : ℕ)
  (num_red_sides_total : ℕ)
  (num_red_sides_with_red_other_side : ℕ) :
  num_cards = 8 →
  num_black_black = 4 →
  num_black_red = 2 →
  num_red_red = 2 →
  num_red_sides_total = (num_red_red * 2 + num_black_red) →
  num_red_sides_with_red_other_side = (num_red_red * 2) →
  (num_red_sides_with_red_other_side / num_red_sides_total : ℝ) = 2 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end red_side_probability_l1167_116790


namespace part1_solution_set_part2_range_of_a_l1167_116748

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l1167_116748


namespace equilibrium_stability_l1167_116721

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x - 2)

theorem equilibrium_stability (x : ℝ) :
  (x = 0 → HasDerivAt f (-1) 0 ∧ (-1 < 0)) ∧
  (x = Real.log 2 → HasDerivAt f (2 * Real.log 2) (Real.log 2) ∧ (2 * Real.log 2 > 0)) :=
by
  sorry

end equilibrium_stability_l1167_116721


namespace exponentiation_correct_l1167_116711

theorem exponentiation_correct (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 :=
sorry

end exponentiation_correct_l1167_116711


namespace ternary_to_decimal_l1167_116797

theorem ternary_to_decimal (k : ℕ) (hk : k > 0) : (1 * 3^3 + k * 3^1 + 2 = 35) → k = 2 :=
by
  sorry

end ternary_to_decimal_l1167_116797


namespace farm_field_area_l1167_116765

theorem farm_field_area
  (planned_daily_plough : ℕ)
  (actual_daily_plough : ℕ)
  (extra_days : ℕ)
  (remaining_area : ℕ)
  (total_days_hectares : ℕ → ℕ) :
  planned_daily_plough = 260 →
  actual_daily_plough = 85 →
  extra_days = 2 →
  remaining_area = 40 →
  total_days_hectares (total_days_hectares (1 + 2) * 85 + 40) = 312 :=
by
  sorry

end farm_field_area_l1167_116765


namespace men_in_first_group_l1167_116740

theorem men_in_first_group (M : ℕ) (h1 : ∀ W, W = M * 30) (h2 : ∀ W, W = 10 * 36) : 
  M = 12 :=
by
  sorry

end men_in_first_group_l1167_116740


namespace neg_p_eq_exist_l1167_116751

theorem neg_p_eq_exist:
  (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2 * a * b) ↔ ∃ a b : ℝ, a^2 + b^2 < 2 * a * b := by
  sorry

end neg_p_eq_exist_l1167_116751


namespace math_problem_l1167_116747

theorem math_problem : 2^5 + (5^2 / 5^1) - 3^3 = 10 :=
by
  sorry

end math_problem_l1167_116747


namespace range_of_x_l1167_116709

theorem range_of_x :
  (∀ t : ℝ, |t - 3| + |2 * t + 1| ≥ |2 * x - 1| + |x + 2|) →
  (-1/2 ≤ x ∧ x ≤ 5/6) :=
by
  intro h 
  sorry

end range_of_x_l1167_116709


namespace inequality_solution_l1167_116733

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + (a - 1) < 0) ↔ a < -1/3 :=
by
  sorry

end inequality_solution_l1167_116733


namespace total_female_students_l1167_116799

def total_students : ℕ := 1600
def sample_size : ℕ := 200
def fewer_girls : ℕ := 10

theorem total_female_students (x : ℕ) (sampled_girls sampled_boys : ℕ) (h_total_sample : sampled_girls + sampled_boys = sample_size)
                             (h_fewer_girls : sampled_girls + fewer_girls = sampled_boys) :
  sampled_girls * 8 = 760 :=
by
  sorry

end total_female_students_l1167_116799


namespace right_triangle_side_length_l1167_116798

theorem right_triangle_side_length (a c b : ℕ) (h₁ : a = 6) (h₂ : c = 10) (h₃ : c * c = a * a + b * b) : b = 8 :=
by {
  sorry
}

end right_triangle_side_length_l1167_116798


namespace probability_same_color_of_two_12_sided_dice_l1167_116761

-- Define the conditions
def sides := 12
def red_sides := 3
def blue_sides := 5
def green_sides := 3
def golden_sides := 1

-- Calculate the probabilities for each color being rolled
def pr_both_red := (red_sides / sides) ^ 2
def pr_both_blue := (blue_sides / sides) ^ 2
def pr_both_green := (green_sides / sides) ^ 2
def pr_both_golden := (golden_sides / sides) ^ 2

-- Total probability calculation
def total_probability_same_color := pr_both_red + pr_both_blue + pr_both_green + pr_both_golden

theorem probability_same_color_of_two_12_sided_dice :
  total_probability_same_color = 11 / 36 := by
  sorry

end probability_same_color_of_two_12_sided_dice_l1167_116761


namespace mean_first_set_l1167_116785

noncomputable def mean (s : List ℚ) : ℚ := s.sum / s.length

theorem mean_first_set (x : ℚ) (h : mean [128, 255, 511, 1023, x] = 423) :
  mean [28, x, 42, 78, 104] = 90 :=
sorry

end mean_first_set_l1167_116785


namespace beads_initial_state_repeats_l1167_116735

-- Define the setup of beads on a circular wire
structure BeadConfig (n : ℕ) :=
(beads : Fin n → ℝ)  -- Each bead's position indexed by a finite set, ℝ denotes angular position

-- Define the instantaneous collision swapping function
def swap (n : ℕ) (i j : Fin n) (config : BeadConfig n) : BeadConfig n :=
⟨fun k => if k = i then config.beads j else if k = j then config.beads i else config.beads k⟩

-- Define what it means for a configuration to return to its initial state
def returns_to_initial (n : ℕ) (initial : BeadConfig n) (t : ℝ) : Prop :=
  ∃ (config : BeadConfig n), (∀ k, config.beads k = initial.beads k) ∧ (config = initial)

-- Specification of the problem
theorem beads_initial_state_repeats (n : ℕ) (initial : BeadConfig n) (ω : Fin n → ℝ) :
  (∀ k, ω k > 0) →  -- condition that all beads have positive angular speed, either clockwise or counterclockwise
  ∃ t : ℝ, t > 0 ∧ returns_to_initial n initial t := 
by
  sorry

end beads_initial_state_repeats_l1167_116735


namespace find_y_l1167_116724

theorem find_y {x y : ℤ} (h1 : x - y = 12) (h2 : x + y = 6) : y = -3 := 
by
  sorry

end find_y_l1167_116724


namespace value_of_t_l1167_116788

theorem value_of_t (k : ℤ) (t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 68) : t = 20 :=
by
  sorry

end value_of_t_l1167_116788


namespace problem_statement_l1167_116702

noncomputable def c := 3 + Real.sqrt 21
noncomputable def d := 3 - Real.sqrt 21

theorem problem_statement : 
  (c + 2 * d) = 9 - Real.sqrt 21 :=
by
  sorry

end problem_statement_l1167_116702


namespace minimum_value_of_a_l1167_116722

theorem minimum_value_of_a :
  ∀ (x : ℝ), (2 * x + 2 / (x - 1) ≥ 7) ↔ (3 ≤ x) :=
sorry

end minimum_value_of_a_l1167_116722


namespace false_converse_implication_l1167_116792

theorem false_converse_implication : ∃ x : ℝ, (0 < x) ∧ (x - 3 ≤ 0) := by
  sorry

end false_converse_implication_l1167_116792


namespace group_C_forms_triangle_l1167_116749

theorem group_C_forms_triangle :
  ∀ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) ↔ ((a, b, c) = (2, 3, 4)) :=
by
  -- we'll prove the forward and backward directions separately
  sorry

end group_C_forms_triangle_l1167_116749


namespace simplify_expression_l1167_116716

-- Define the original expression and the simplified version
def original_expr (x y : ℤ) : ℤ := 7 * x + 3 - 2 * x + 15 + y
def simplified_expr (x y : ℤ) : ℤ := 5 * x + y + 18

-- The equivalence to be proved
theorem simplify_expression (x y : ℤ) : original_expr x y = simplified_expr x y :=
by sorry

end simplify_expression_l1167_116716


namespace probability_of_other_note_being_counterfeit_l1167_116770

def total_notes := 20
def counterfeit_notes := 5

-- Binomial coefficient (n choose k)
noncomputable def binom (n k : ℕ) : ℚ := n.choose k

-- Probability of event A: both notes are counterfeit
noncomputable def P_A : ℚ :=
  binom counterfeit_notes 2 / binom total_notes 2

-- Probability of event B: at least one note is counterfeit
noncomputable def P_B : ℚ :=
  (binom counterfeit_notes 2 + binom counterfeit_notes 1 * binom (total_notes - counterfeit_notes) 1) / binom total_notes 2

-- Conditional probability P(A|B)
noncomputable def P_A_given_B : ℚ :=
  P_A / P_B

theorem probability_of_other_note_being_counterfeit :
  P_A_given_B = 2/17 :=
by
  sorry

end probability_of_other_note_being_counterfeit_l1167_116770


namespace sum_mod_20_l1167_116757

theorem sum_mod_20 : 
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 20 = 15 :=
by 
  -- The proof goes here
  sorry

end sum_mod_20_l1167_116757


namespace region_midpoint_area_equilateral_triangle_52_36_l1167_116780

noncomputable def equilateral_triangle (A B C: ℝ × ℝ) : Prop :=
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

def midpoint_region_area (a b c : ℝ × ℝ) : ℝ := sorry

theorem region_midpoint_area_equilateral_triangle_52_36 (A B C: ℝ × ℝ) (h: equilateral_triangle A B C) :
  let m := (midpoint_region_area A B C)
  100 * m = 52.36 :=
sorry

end region_midpoint_area_equilateral_triangle_52_36_l1167_116780


namespace compute_expression_l1167_116777

theorem compute_expression : (5 + 9)^2 + Real.sqrt (5^2 + 9^2) = 196 + Real.sqrt 106 := 
by sorry

end compute_expression_l1167_116777


namespace length_BC_l1167_116703

theorem length_BC (AB AC AM : ℝ)
  (hAB : AB = 5)
  (hAC : AC = 7)
  (hAM : AM = 4)
  (M_midpoint_of_BC : ∃ (BM MC : ℝ), BM = MC ∧ ∀ (BC: ℝ), BC = BM + MC) :
  ∃ (BC : ℝ), BC = 2 * Real.sqrt 21 := by
  sorry

end length_BC_l1167_116703


namespace alex_new_salary_in_may_l1167_116706

def initial_salary : ℝ := 50000
def february_increase (s : ℝ) : ℝ := s * 1.10
def april_bonus (s : ℝ) : ℝ := s + 2000
def may_pay_cut (s : ℝ) : ℝ := s * 0.95

theorem alex_new_salary_in_may : may_pay_cut (april_bonus (february_increase initial_salary)) = 54150 :=
by
  sorry

end alex_new_salary_in_may_l1167_116706


namespace equation_one_solution_equation_two_solution_l1167_116728

theorem equation_one_solution (x : ℝ) (h : 7 * x - 20 = 2 * (3 - 3 * x)) : x = 2 :=
by {
  sorry
}

theorem equation_two_solution (x : ℝ) (h : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1) : x = -1 :=
by {
  sorry
}

end equation_one_solution_equation_two_solution_l1167_116728


namespace least_number_to_add_for_divisibility_by_11_l1167_116776

theorem least_number_to_add_for_divisibility_by_11 : ∃ k : ℕ, 11002 + k ≡ 0 [MOD 11] ∧ k = 9 := by
  sorry

end least_number_to_add_for_divisibility_by_11_l1167_116776


namespace root_of_equation_imp_expression_eq_one_l1167_116715

variable (m : ℝ)

theorem root_of_equation_imp_expression_eq_one
  (h : m^2 - m - 1 = 0) : m^2 - m = 1 :=
  sorry

end root_of_equation_imp_expression_eq_one_l1167_116715


namespace find_the_number_l1167_116745

noncomputable def special_expression (x : ℝ) : ℝ :=
  9 - 8 / x * 5 + 10

theorem find_the_number (x : ℝ) (h : special_expression x = 13.285714285714286) : x = 7 := by
  sorry

end find_the_number_l1167_116745


namespace problem_condition_implies_statement_l1167_116736

variable {a b c : ℝ}

theorem problem_condition_implies_statement :
  a^3 + a * b + a * c < 0 → b^5 - 4 * a * c > 0 :=
by
  intros h
  sorry

end problem_condition_implies_statement_l1167_116736


namespace sum_a1_a4_l1167_116772

variables (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := n^2 + n + 1

-- Define the individual terms of the sequence
def term_seq (n : ℕ) : ℕ :=
if n = 1 then sum_seq 1 else sum_seq n - sum_seq (n - 1)

-- Prove that the sum of the first and fourth terms equals 11
theorem sum_a1_a4 : 
  (term_seq 1) + (term_seq 4) = 11 :=
by
  -- to be completed with proof steps
  sorry

end sum_a1_a4_l1167_116772


namespace lowry_earnings_l1167_116731

def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def small_bonsai_sold : ℕ := 3
def big_bonsai_sold : ℕ := 5

def total_earnings (small_cost : ℕ) (big_cost : ℕ) (small_sold : ℕ) (big_sold : ℕ) : ℕ :=
  small_cost * small_sold + big_cost * big_sold

theorem lowry_earnings :
  total_earnings small_bonsai_cost big_bonsai_cost small_bonsai_sold big_bonsai_sold = 190 := 
by
  sorry

end lowry_earnings_l1167_116731


namespace factorization_of_polynomial_l1167_116701

theorem factorization_of_polynomial (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end factorization_of_polynomial_l1167_116701


namespace power_function_decreasing_m_eq_2_l1167_116746

theorem power_function_decreasing_m_eq_2 (x : ℝ) (m : ℝ) (hx : 0 < x) 
  (h_decreasing : ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → 
                    (m^2 - m - 1) * x₁^(-m+1) > (m^2 - m - 1) * x₂^(-m+1))
  (coeff_positive : m^2 - m - 1 > 0)
  (expo_condition : -m + 1 < 0) : 
  m = 2 :=
by
  sorry

end power_function_decreasing_m_eq_2_l1167_116746


namespace sufficient_but_not_necessary_condition_l1167_116720

-- Define the conditions as predicates
def p (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6 * x + 9 - m^2 ≤ 0

-- Range for m where p is sufficient but not necessary for q
def m_range (m : ℝ) : Prop := m ≤ -4 ∨ m ≥ 4

-- The main goal to be proven
theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x, p x → q x m) ∧ ¬(∀ x, q x m → p x) ↔ m_range m :=
sorry

end sufficient_but_not_necessary_condition_l1167_116720


namespace set_difference_example_l1167_116712

-- Define P and Q based on the given conditions
def P : Set ℝ := {x | 0 < x ∧ x < 2}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the theorem: P - Q equals to the set {x | 0 < x ≤ 1}
theorem set_difference_example : P \ Q = {x | 0 < x ∧ x ≤ 1} := 
  by
  sorry

end set_difference_example_l1167_116712


namespace solve_for_x_l1167_116710

theorem solve_for_x (x : ℝ) (h : 3 * x - 7 = 2 * x + 5) : x = 12 :=
sorry

end solve_for_x_l1167_116710


namespace point_in_first_quadrant_l1167_116744

/-- In the Cartesian coordinate system, if a point P has x-coordinate 2 and y-coordinate 4, it lies in the first quadrant. -/
theorem point_in_first_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = 4) : 
  x > 0 ∧ y > 0 → 
  (x, y).1 = 2 ∧ (x, y).2 = 4 → 
  (x > 0 ∧ y > 0) := 
by
  intros
  sorry

end point_in_first_quadrant_l1167_116744


namespace max_square_plots_l1167_116771

theorem max_square_plots (length width available_fencing : ℕ) 
(h : length = 30 ∧ width = 60 ∧ available_fencing = 2500) : 
  ∃ n : ℕ, n = 72 ∧ ∀ s : ℕ, ((30 * (60 / s - 1)) + (60 * (30 / s - 1)) ≤ 2500) → ((30 / s) * (60 / s) = n) := by
  sorry

end max_square_plots_l1167_116771


namespace find_A_l1167_116707

theorem find_A (d q r A : ℕ) (h1 : d = 7) (h2 : q = 5) (h3 : r = 3) (h4 : A = d * q + r) : A = 38 := 
by 
  { sorry }

end find_A_l1167_116707


namespace pigeons_percentage_l1167_116767

theorem pigeons_percentage (total_birds pigeons sparrows crows doves non_sparrows : ℕ)
  (h_total : total_birds = 100)
  (h_pigeons : pigeons = 40)
  (h_sparrows : sparrows = 20)
  (h_crows : crows = 15)
  (h_doves : doves = 25)
  (h_non_sparrows : non_sparrows = total_birds - sparrows) :
  (pigeons / non_sparrows : ℚ) * 100 = 50 :=
sorry

end pigeons_percentage_l1167_116767


namespace sum_x_y_z_w_l1167_116714

-- Define the conditions in Lean
variables {x y z w : ℤ}
axiom h1 : x - y + z = 7
axiom h2 : y - z + w = 8
axiom h3 : z - w + x = 4
axiom h4 : w - x + y = 3

-- Prove the result
theorem sum_x_y_z_w : x + y + z + w = 22 := by
  sorry

end sum_x_y_z_w_l1167_116714


namespace bike_travel_distance_l1167_116758

def avg_speed : ℝ := 3  -- average speed in m/s
def time : ℝ := 7       -- time in seconds

theorem bike_travel_distance : avg_speed * time = 21 := by
  sorry

end bike_travel_distance_l1167_116758


namespace exist_three_sum_eq_third_l1167_116739

theorem exist_three_sum_eq_third
  (A : Finset ℕ)
  (h_card : A.card = 52)
  (h_cond : ∀ (a : ℕ), a ∈ A → a ≤ 100) :
  ∃ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y = z :=
sorry

end exist_three_sum_eq_third_l1167_116739


namespace max_jars_in_crate_l1167_116753

-- Define the conditions given in the problem
def side_length_cardboard_box := 20 -- in cm
def jars_per_box := 8
def crate_width := 80 -- in cm
def crate_length := 120 -- in cm
def crate_height := 60 -- in cm
def volume_box := side_length_cardboard_box ^ 3
def volume_crate := crate_width * crate_length * crate_height
def boxes_per_crate := volume_crate / volume_box
def max_jars_per_crate := boxes_per_crate * jars_per_box

-- Statement that needs to be proved
theorem max_jars_in_crate : max_jars_per_crate = 576 := sorry

end max_jars_in_crate_l1167_116753


namespace no_solution_natural_p_q_r_l1167_116768

theorem no_solution_natural_p_q_r :
  ¬ ∃ (p q r : ℕ), 2^p + 5^q = 19^r := sorry

end no_solution_natural_p_q_r_l1167_116768


namespace min_copy_paste_actions_l1167_116783

theorem min_copy_paste_actions :
  ∀ (n : ℕ), (n ≥ 10) ∧ (n ≤ n) → (2^n ≥ 1001) :=
by sorry

end min_copy_paste_actions_l1167_116783


namespace cricket_bat_cost_price_l1167_116755

theorem cricket_bat_cost_price (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 222) : CP_A = 148 := 
by
  sorry

end cricket_bat_cost_price_l1167_116755


namespace simplify_abs_expr_l1167_116742

theorem simplify_abs_expr : |(-4 ^ 2 + 6)| = 10 := by
  sorry

end simplify_abs_expr_l1167_116742


namespace max_xy_value_l1167_116795

theorem max_xy_value (x y : ℝ) (h : x^2 + y^2 + 3 * x * y = 2015) : xy <= 403 :=
sorry

end max_xy_value_l1167_116795


namespace sum_of_squares_l1167_116730

theorem sum_of_squares :
  ∃ p q r s t u : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧ 
    (p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210) :=
sorry

end sum_of_squares_l1167_116730


namespace ratio_of_two_numbers_l1167_116729

theorem ratio_of_two_numbers (A B : ℕ) (x y : ℕ) (h1 : lcm A B = 60) (h2 : A + B = 50) (h3 : A / B = x / y) (hx : x = 3) (hy : y = 2) : x = 3 ∧ y = 2 := 
by
  -- Conditions provided in the problem
  sorry

end ratio_of_two_numbers_l1167_116729


namespace water_lilies_half_pond_l1167_116787

theorem water_lilies_half_pond (growth_rate : ℕ → ℕ) (start_day : ℕ) (full_covered_day : ℕ) 
  (h_growth : ∀ n, growth_rate (n + 1) = 2 * growth_rate n) 
  (h_start : growth_rate start_day = 1) 
  (h_full_covered : growth_rate full_covered_day = 2^(full_covered_day - start_day)) : 
  growth_rate (full_covered_day - 1) = 2^(full_covered_day - start_day - 1) :=
by
  sorry

end water_lilies_half_pond_l1167_116787


namespace find_g_l1167_116754

theorem find_g (x : ℝ) (g : ℝ → ℝ) :
  2 * x^5 - 4 * x^3 + 3 * x^2 + g x = 7 * x^4 - 5 * x^3 + x^2 - 9 * x + 2 →
  g x = -2 * x^5 + 7 * x^4 - x^3 - 2 * x^2 - 9 * x + 2 :=
by
  intro h
  sorry

end find_g_l1167_116754


namespace compute_expression_l1167_116713

theorem compute_expression :
  23 ^ 12 / 23 ^ 5 + 5 = 148035894 :=
  sorry

end compute_expression_l1167_116713


namespace field_dimension_solution_l1167_116781

theorem field_dimension_solution (m : ℝ) (h₁ : (3 * m + 10) * (m - 5) = 72) : m = 7 :=
sorry

end field_dimension_solution_l1167_116781


namespace distance_from_B_to_center_is_74_l1167_116725

noncomputable def circle_radius := 10
noncomputable def B_distance (a b : ℝ) := a^2 + b^2

theorem distance_from_B_to_center_is_74 
  (a b : ℝ)
  (hA : a^2 + (b + 6)^2 = 100)
  (hC : (a + 4)^2 + b^2 = 100) :
  B_distance a b = 74 :=
sorry

end distance_from_B_to_center_is_74_l1167_116725


namespace solve_integers_l1167_116762

theorem solve_integers (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^(2 * y) + (x + 1)^(2 * y) = (x + 2)^(2 * y) → (x = 3 ∧ y = 1) :=
by
  sorry

end solve_integers_l1167_116762


namespace scalene_triangle_angles_l1167_116734

theorem scalene_triangle_angles (x y z : ℝ) (h1 : x + y + z = 180) (h2 : x ≠ y ∧ y ≠ z ∧ x ≠ z)
(h3 : x = 36 ∨ y = 36 ∨ z = 36) (h4 : x = 2 * y ∨ y = 2 * x ∨ z = 2 * x ∨ x = 2 * z ∨ y = 2 * z ∨ z = 2 * y) :
(x = 36 ∧ y = 48 ∧ z = 96) ∨ (x = 18 ∧ y = 36 ∧ z = 126) ∨ (x = 36 ∧ z = 48 ∧ y = 96) ∨ (y = 18 ∧ x = 36 ∧ z = 126) :=
sorry

end scalene_triangle_angles_l1167_116734


namespace intersection_point_of_lines_l1167_116775

theorem intersection_point_of_lines : ∃ (x y : ℝ), x + y = 5 ∧ x - y = 1 ∧ x = 3 ∧ y = 2 :=
by
  sorry

end intersection_point_of_lines_l1167_116775


namespace measure_angle_C_l1167_116732

noncomputable def triangle_angles_sum (a b c : ℝ) : Prop :=
  a + b + c = 180

noncomputable def angle_B_eq_twice_angle_C (b c : ℝ) : Prop :=
  b = 2 * c

noncomputable def angle_A_eq_40 : ℝ := 40

theorem measure_angle_C :
  ∀ (B C : ℝ), triangle_angles_sum angle_A_eq_40 B C → angle_B_eq_twice_angle_C B C → C = 140 / 3 :=
by
  intros B C h1 h2
  sorry

end measure_angle_C_l1167_116732


namespace thabo_paperback_diff_l1167_116760

variable (total_books : ℕ) (H_books : ℕ) (P_books : ℕ) (F_books : ℕ)

def thabo_books_conditions :=
  total_books = 160 ∧
  H_books = 25 ∧
  P_books > H_books ∧
  F_books = 2 * P_books ∧
  total_books = F_books + P_books + H_books 

theorem thabo_paperback_diff :
  thabo_books_conditions total_books H_books P_books F_books → 
  (P_books - H_books) = 20 :=
by
  sorry

end thabo_paperback_diff_l1167_116760
