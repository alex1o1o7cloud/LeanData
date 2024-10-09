import Mathlib

namespace watch_all_episodes_in_67_weeks_l1547_154740

def total_episodes : ℕ := 201
def episodes_per_week : ℕ := 1 + 2

theorem watch_all_episodes_in_67_weeks :
  total_episodes / episodes_per_week = 67 := by 
  sorry

end watch_all_episodes_in_67_weeks_l1547_154740


namespace election_votes_l1547_154756

theorem election_votes (P : ℕ) (M : ℕ) (V : ℕ) (hP : P = 60) (hM : M = 1300) :
  V = 6500 :=
by
  sorry

end election_votes_l1547_154756


namespace factory_processing_time_eq_l1547_154758

variable (x : ℝ) (initial_rate : ℝ := x)
variable (parts : ℝ := 500)
variable (first_stage_parts : ℝ := 100)
variable (remaining_parts : ℝ := parts - first_stage_parts)
variable (total_days : ℝ := 6)
variable (new_rate : ℝ := 2 * initial_rate)

theorem factory_processing_time_eq (h : x > 0) : (first_stage_parts / initial_rate) + (remaining_parts / new_rate) = total_days := 
by
  sorry

end factory_processing_time_eq_l1547_154758


namespace stratified_sampling_freshman_l1547_154799

def total_students : ℕ := 1800 + 1500 + 1200
def sample_size : ℕ := 150
def freshman_students : ℕ := 1200

/-- if a sample of 150 students is drawn using stratified sampling, 40 students should be drawn from the freshman year -/
theorem stratified_sampling_freshman :
  (freshman_students * sample_size) / total_students = 40 :=
by
  sorry

end stratified_sampling_freshman_l1547_154799


namespace f_prime_neg1_l1547_154703

def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

def f' (a b c x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem f_prime_neg1 (a b c : ℝ) (h : f' a b c 1 = 2) : f' a b c (-1) = -2 :=
by
  sorry

end f_prime_neg1_l1547_154703


namespace simplify_expression_l1547_154721

theorem simplify_expression (α : ℝ) :
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) - (2 * Real.cos (2 * α)^2 - 1)) /
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) + (2 * Real.cos (2 * α)^2 - 1)) = Real.tan (2 * α) :=
by
  sorry

end simplify_expression_l1547_154721


namespace hall_length_l1547_154711

theorem hall_length
  (breadth : ℝ) (stone_length_dm stone_width_dm : ℝ) (num_stones : ℕ) (L : ℝ)
  (h_breadth : breadth = 15)
  (h_stone_length : stone_length_dm = 6)
  (h_stone_width : stone_width_dm = 5)
  (h_num_stones : num_stones = 1800)
  (h_length : L = 36) :
  let stone_length := stone_length_dm / 10
  let stone_width := stone_width_dm / 10
  let stone_area := stone_length * stone_width
  let total_area := num_stones * stone_area
  total_area / breadth = L :=
by {
  sorry
}

end hall_length_l1547_154711


namespace num_isosceles_triangles_with_perimeter_30_l1547_154747

theorem num_isosceles_triangles_with_perimeter_30 : 
  (∃ (s : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ s → 2 * a + b = 30 ∧ (a ≥ b) ∧ b ≠ 0 ∧ a + a > b ∧ a + b > a ∧ b + a > a) 
    ∧ s.card = 7) :=
by {
  sorry
}

end num_isosceles_triangles_with_perimeter_30_l1547_154747


namespace sector_area_given_angle_radius_sector_max_area_perimeter_l1547_154752

open Real

theorem sector_area_given_angle_radius :
  ∀ (α : ℝ) (R : ℝ), α = 60 * (π / 180) ∧ R = 10 →
  (α / 360 * 2 * π * R) = 10 * π / 3 ∧ 
  (α * π * R^2 / 360) = 50 * π / 3 :=
by
  intros α R h
  rcases h with ⟨hα, hR⟩
  sorry

theorem sector_max_area_perimeter :
  ∀ (r α: ℝ), (2 * r + r * α) = 8 →
  α = 2 →
  r = 2 :=
by
  intros r α h ha
  sorry

end sector_area_given_angle_radius_sector_max_area_perimeter_l1547_154752


namespace fraction_subtraction_l1547_154792

theorem fraction_subtraction (x : ℚ) : x - (1/5 : ℚ) = (3/5 : ℚ) → x = (4/5 : ℚ) :=
by
  sorry

end fraction_subtraction_l1547_154792


namespace function_periodicity_l1547_154750

theorem function_periodicity (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + f x = 0)
  (h2 : ∀ x, f (x + 1) = f (1 - x)) (h3 : f 1 = 5) : f 2015 = -5 :=
sorry

end function_periodicity_l1547_154750


namespace find_sixth_number_l1547_154795

theorem find_sixth_number (avg_all : ℝ) (avg_first6 : ℝ) (avg_last6 : ℝ) (total_avg : avg_all = 10.7) (first6_avg: avg_first6 = 10.5) (last6_avg: avg_last6 = 11.4) : 
  let S1 := 6 * avg_first6
  let S2 := 6 * avg_last6
  let total_sum := 11 * avg_all
  let X := total_sum - (S1 - X + S2 - X)
  X = 13.7 :=
by 
  sorry

end find_sixth_number_l1547_154795


namespace different_answers_due_to_different_cuts_l1547_154737

noncomputable def problem_89914 (bub : Type) (cut : bub → (bub × bub)) (is_log_cut : bub → Prop) (is_halved_log : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_log_cut b) → is_halved_log (cut b)

noncomputable def problem_89915 (bub : Type) (cut : bub → (bub × bub)) (is_sector_cut : bub → Prop) (is_sectors : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_sector_cut b) → is_sectors (cut b)

theorem different_answers_due_to_different_cuts
  (bub : Type)
  (cut : bub → (bub × bub))
  (is_log_cut : bub → Prop)
  (is_halved_log : bub × bub → Prop)
  (is_sector_cut : bub → Prop)
  (is_sectors : bub × bub → Prop) :
  problem_89914 bub cut is_log_cut is_halved_log ∧ problem_89915 bub cut is_sector_cut is_sectors →
  ∃ b : bub, (is_log_cut b ∧ ¬ is_sector_cut b) ∨ (¬ is_log_cut b ∧ is_sector_cut b) := sorry

end different_answers_due_to_different_cuts_l1547_154737


namespace eq_radicals_same_type_l1547_154778

theorem eq_radicals_same_type (a b : ℕ) (h1 : a - 1 = 2) (h2 : 3 * b - 1 = 7 - b) : a + b = 5 :=
by
  sorry

end eq_radicals_same_type_l1547_154778


namespace water_formed_l1547_154728

theorem water_formed (n_HCl : ℕ) (n_CaCO3: ℕ) (n_H2O: ℕ) 
  (balance_eqn: ∀ (n : ℕ), 
    (2 * n_CaCO3) ≤ n_HCl ∧
    n_H2O = n_CaCO3 ):
  n_HCl = 4 ∧ n_CaCO3 = 2 → n_H2O = 2 :=
by
  intros h0
  obtain ⟨h1, h2⟩ := h0
  sorry

end water_formed_l1547_154728


namespace probability_heads_9_or_more_12_flips_l1547_154729

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l1547_154729


namespace sweetsies_remainder_l1547_154701

-- Each definition used in Lean 4 statement should be directly from the conditions in a)
def number_of_sweetsies_in_one_bag (m : ℕ): Prop :=
  m % 8 = 5

theorem sweetsies_remainder (m : ℕ) (h : number_of_sweetsies_in_one_bag m) : 
  (4 * m) % 8 = 4 := by
  -- Proof will be provided here.
  sorry

end sweetsies_remainder_l1547_154701


namespace population_of_seventh_village_l1547_154704

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 980]

def average_population : ℕ := 1000

theorem population_of_seventh_village 
  (h1 : List.length village_populations = 6)
  (h2 : 1000 * 7 = 7000)
  (h3 : village_populations.sum = 5751) : 
  7000 - village_populations.sum = 1249 := 
by {
  -- h1 ensures there's exactly 6 villages in the list
  -- h2 calculates the total population of 7 villages assuming the average population
  -- h3 calculates the sum of populations in the given list of 6 villages
  -- our goal is to show that 7000 - village_populations.sum = 1249
  -- this will be simplified in the proof
  sorry
}

end population_of_seventh_village_l1547_154704


namespace smallest_d_for_range_of_g_l1547_154788

theorem smallest_d_for_range_of_g :
  ∃ d, (∀ x : ℝ, x^2 + 4 * x + d = 3) → d = 7 := by
  sorry

end smallest_d_for_range_of_g_l1547_154788


namespace increasing_on_real_iff_a_range_l1547_154739

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a / x

theorem increasing_on_real_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -3 ≤ a ∧ a ≤ -2 := 
by
  sorry

end increasing_on_real_iff_a_range_l1547_154739


namespace range_of_m_l1547_154766

variable {α : Type*} [LinearOrder α]

def increasing (f : α → α) : Prop :=
  ∀ ⦃x y : α⦄, x < y → f x < f y

theorem range_of_m 
  (f : ℝ → ℝ) 
  (h_inc : increasing f) 
  (h_cond : ∀ m : ℝ, f (m + 3) ≤ f 5) : 
  {m : ℝ | f (m + 3) ≤ f 5} = {m : ℝ | m ≤ 2} := 
sorry

end range_of_m_l1547_154766


namespace part1_part2_l1547_154784

noncomputable def f (a x : ℝ) : ℝ := (a / x) - Real.log x

theorem part1 (a : ℝ) (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2) 
(hf : f a x1 = -3) (hf2 : f a x2 = -3) : a ∈ Set.Ioo (-Real.exp 2) 0 :=
sorry

theorem part2 (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2)
(hfa : f (-2) x1 = -3) (hfb : f (-2) x2 = -3) : x1 + x2 > 4 :=
sorry

end part1_part2_l1547_154784


namespace simplify_sqrt_mul_l1547_154781

theorem simplify_sqrt_mul : (Real.sqrt 5 * Real.sqrt (4 / 5) = 2) :=
by
  sorry

end simplify_sqrt_mul_l1547_154781


namespace range_of_m_l1547_154718

variable (f : Real → Real)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom decreasing_function : ∀ x y, x < y → -1 < x ∧ y < 1 → f x > f y
axiom domain : ∀ x, -1 < x ∧ x < 1 → true

-- The statement to be proved
theorem range_of_m (m : Real) : 
  f (1 - m) + f (1 - m^2) < 0 → 0 < m → m < 1 :=
by
  sorry

end range_of_m_l1547_154718


namespace fraction_surface_area_red_l1547_154796

theorem fraction_surface_area_red :
  ∀ (num_unit_cubes : ℕ) (side_length_large_cube : ℕ) (total_surface_area_painted : ℕ) (total_surface_area_unit_cubes : ℕ),
    num_unit_cubes = 8 →
    side_length_large_cube = 2 →
    total_surface_area_painted = 6 * (side_length_large_cube ^ 2) →
    total_surface_area_unit_cubes = num_unit_cubes * 6 →
    (total_surface_area_painted : ℝ) / total_surface_area_unit_cubes = 1 / 2 :=
by
  intros num_unit_cubes side_length_large_cube total_surface_area_painted total_surface_area_unit_cubes
  sorry

end fraction_surface_area_red_l1547_154796


namespace cost_price_perc_of_selling_price_l1547_154765

theorem cost_price_perc_of_selling_price
  (SP : ℝ) (CP : ℝ) (P : ℝ)
  (h1 : P = SP - CP)
  (h2 : P = (4.166666666666666 / 100) * SP) :
  CP = SP * 0.9583333333333334 :=
by
  sorry

end cost_price_perc_of_selling_price_l1547_154765


namespace find_number_l1547_154773

theorem find_number (n : ℝ) (h : (1/2) * n + 5 = 11) : n = 12 :=
by
  sorry

end find_number_l1547_154773


namespace money_raised_is_correct_l1547_154774

noncomputable def cost_per_dozen : ℚ := 2.40
noncomputable def selling_price_per_donut : ℚ := 1
noncomputable def dozens : ℕ := 10

theorem money_raised_is_correct :
  (dozens * 12 * selling_price_per_donut - dozens * cost_per_dozen) = 96 := by
sorry

end money_raised_is_correct_l1547_154774


namespace dan_marbles_l1547_154735

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  original_marbles = 128 →
  given_marbles = 32 →
  remaining_marbles = original_marbles - given_marbles →
  remaining_marbles = 96 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dan_marbles_l1547_154735


namespace max_non_managers_depA_l1547_154775

theorem max_non_managers_depA (mA : ℕ) (nA : ℕ) (sA : ℕ) (gA : ℕ) (totalA : ℕ) :
  mA = 9 ∧ (8 * nA > 37 * mA) ∧ (sA = 2 * gA) ∧ (nA = sA + gA) ∧ (mA + nA ≤ 250) →
  nA = 39 :=
by
  sorry

end max_non_managers_depA_l1547_154775


namespace maximum_height_l1547_154746

-- Define the quadratic function h(t)
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 60

-- Define our proof problem
theorem maximum_height : ∃ t : ℝ, h t = 140 :=
by
  let t := -80 / (2 * -20)
  use t
  sorry

end maximum_height_l1547_154746


namespace Series_value_l1547_154753

theorem Series_value :
  (∑' n : ℕ, (2^n) / (7^(2^n) + 1)) = 1 / 6 :=
sorry

end Series_value_l1547_154753


namespace total_cost_correct_l1547_154791

-- Conditions given in the problem.
def net_profit : ℝ := 44
def gross_revenue : ℝ := 47
def lemonades_sold : ℝ := 50
def babysitting_income : ℝ := 31

def cost_per_lemon : ℝ := 0.20
def cost_per_sugar : ℝ := 0.15
def cost_per_ice : ℝ := 0.05

def one_time_cost_sunhat : ℝ := 10

-- Definition of variable cost per lemonade.
def variable_cost_per_lemonade : ℝ := cost_per_lemon + cost_per_sugar + cost_per_ice

-- Definition of total variable cost for all lemonades sold.
def total_variable_cost : ℝ := lemonades_sold * variable_cost_per_lemonade

-- Final total cost to operate the lemonade stand.
def total_cost : ℝ := total_variable_cost + one_time_cost_sunhat

-- The proof statement that total cost is equal to $30.
theorem total_cost_correct : total_cost = 30 := by
  sorry

end total_cost_correct_l1547_154791


namespace minimum_omega_l1547_154700

theorem minimum_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω * (3 * π / 4) - ω * (π / 4) = k * π) → ω = 2 :=
by
  sorry

end minimum_omega_l1547_154700


namespace find_width_of_floor_l1547_154712

variable (w : ℝ) -- width of the floor

theorem find_width_of_floor (h1 : w - 4 > 0) (h2 : 10 - 4 > 0) 
                            (area_rug : (10 - 4) * (w - 4) = 24) : w = 8 :=
by
  sorry

end find_width_of_floor_l1547_154712


namespace greatest_possible_perimeter_l1547_154733

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l1547_154733


namespace even_integer_operations_l1547_154727

theorem even_integer_operations (k : ℤ) (a : ℤ) (h : a = 2 * k) :
  (a * 5) % 2 = 0 ∧ (a ^ 2) % 2 = 0 ∧ (a ^ 3) % 2 = 0 :=
by
  sorry

end even_integer_operations_l1547_154727


namespace find_ordered_pair_l1547_154780

theorem find_ordered_pair (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) := by
  sorry

end find_ordered_pair_l1547_154780


namespace xy_uv_zero_l1547_154787

theorem xy_uv_zero (x y u v : ℝ) (h1 : x^2 + y^2 = 1) (h2 : u^2 + v^2 = 1) (h3 : x * u + y * v = 0) : x * y + u * v = 0 :=
by
  sorry

end xy_uv_zero_l1547_154787


namespace fruit_ratio_l1547_154734

variable (A P B : ℕ)
variable (n : ℕ)

theorem fruit_ratio (h1 : A = 4) (h2 : P = n * A) (h3 : A + P + B = 21) (h4 : B = 5) : P / A = 3 := by
  sorry

end fruit_ratio_l1547_154734


namespace remainder_of_899830_divided_by_16_is_6_l1547_154751

theorem remainder_of_899830_divided_by_16_is_6 :
  ∃ k : ℕ, 899830 = 16 * k + 6 :=
by
  sorry

end remainder_of_899830_divided_by_16_is_6_l1547_154751


namespace total_skips_l1547_154764

-- Definitions of the given conditions
def BobsSkipsPerRock := 12
def JimsSkipsPerRock := 15
def NumberOfRocks := 10

-- Statement of the theorem to be proved
theorem total_skips :
  (BobsSkipsPerRock * NumberOfRocks) + (JimsSkipsPerRock * NumberOfRocks) = 270 :=
by
  sorry

end total_skips_l1547_154764


namespace two_digit_number_representation_l1547_154786

def tens_digit := ℕ
def units_digit := ℕ

theorem two_digit_number_representation (b a : ℕ) : 
  (∀ (b a : ℕ), 10 * b + a = 10 * b + a) := sorry

end two_digit_number_representation_l1547_154786


namespace find_x_and_y_l1547_154768

variable {x y : ℝ}

-- Given condition
def angleDCE : ℝ := 58

-- Proof statements
theorem find_x_and_y : x = 180 - angleDCE ∧ y = 180 - angleDCE := by
  sorry

end find_x_and_y_l1547_154768


namespace ratio_of_areas_l1547_154702

theorem ratio_of_areas 
  (lenA : ℕ) (brdA : ℕ) (lenB : ℕ) (brdB : ℕ)
  (h_lenA : lenA = 48) 
  (h_brdA : brdA = 30)
  (h_lenB : lenB = 60) 
  (h_brdB : brdB = 35) :
  (lenA * brdA : ℚ) / (lenB * brdB) = 24 / 35 :=
by
  sorry

end ratio_of_areas_l1547_154702


namespace general_term_arithmetic_sum_terms_geometric_l1547_154722

section ArithmeticSequence

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Conditions for Part 1
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  S 5 - S 2 = 195 ∧ d = -2 ∧
  ∀ n, S n = n * (a 1 + (n - 1) * (d / 2))

-- Prove the general term formula for the sequence {a_n}
theorem general_term_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) 
    (h : sum_arithmetic_sequence S a d) : 
    ∀ n, a n = -2 * n + 73 :=
sorry

end ArithmeticSequence


section GeometricSequence

variables {b : ℕ → ℝ} {n : ℕ} {T : ℕ → ℝ} {a : ℕ → ℝ}

-- Conditions for Part 2
def sum_geometric_sequence (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = 13 ∧ b 2 = 65 ∧ a 4 = 65

-- Prove the sum of the first n terms for the sequence {b_n}
theorem sum_terms_geometric (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ)
    (h : sum_geometric_sequence b T a) : 
    ∀ n, T n = 13 * (5^n - 1) / 4 :=
sorry

end GeometricSequence

end general_term_arithmetic_sum_terms_geometric_l1547_154722


namespace construct_using_five_twos_l1547_154797

theorem construct_using_five_twos :
  (∃ (a b c d e f : ℕ), (22 * (a / b)) / c = 11 ∧
                        (22 / d) + (e / f) = 12 ∧
                        (22 + g + h) / i = 13 ∧
                        (2 * 2 * 2 * 2 - j) = 14 ∧
                        (22 / k) + (2 * 2) = 15) := by
  sorry

end construct_using_five_twos_l1547_154797


namespace range_of_b_in_acute_triangle_l1547_154719

variable {a b c : ℝ}

theorem range_of_b_in_acute_triangle (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_acute : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2))
  (h_arith_seq : ∃ d : ℝ, 0 ≤ d ∧ a = b - d ∧ c = b + d)
  (h_sum_squares : a^2 + b^2 + c^2 = 21) :
  (2 * Real.sqrt 42) / 5 < b ∧ b ≤ Real.sqrt 7 :=
sorry

end range_of_b_in_acute_triangle_l1547_154719


namespace hectares_per_day_initial_l1547_154741

variable (x : ℝ) -- x is the number of hectares one tractor ploughs initially per day

-- Condition 1: A field can be ploughed by 6 tractors in 4 days.
def total_area_initial := 6 * x * 4

-- Condition 2: 6 tractors plough together a certain number of hectares per day, denoted as x hectares/day.
-- This is incorporated in the variable declaration of x.

-- Condition 3: If 2 tractors are moved to another field, the remaining 4 tractors can plough the same field in 5 days.
-- Condition 4: One of the 4 tractors ploughs 144 hectares a day when 4 tractors plough the field in 5 days.
def total_area_with_4_tractors := 4 * 144 * 5

-- The statement that equates the two total area expressions.
theorem hectares_per_day_initial : total_area_initial x = total_area_with_4_tractors := by
  sorry

end hectares_per_day_initial_l1547_154741


namespace find_c_l1547_154705

theorem find_c (x c : ℤ) (h1 : 3 * x + 9 = 0) (h2 : c * x - 5 = -11) : c = 2 := by
  have x_eq : x = -3 := by
    linarith
  subst x_eq
  have c_eq : c = 2 := by
    linarith
  exact c_eq

end find_c_l1547_154705


namespace boys_love_marbles_l1547_154742

def total_marbles : ℕ := 26
def marbles_per_boy : ℕ := 2
def num_boys_love_marbles : ℕ := total_marbles / marbles_per_boy

theorem boys_love_marbles : num_boys_love_marbles = 13 := by
  rfl

end boys_love_marbles_l1547_154742


namespace estimated_total_score_l1547_154761

noncomputable def regression_score (x : ℝ) : ℝ := 7.3 * x - 96.9

theorem estimated_total_score (x : ℝ) (h : x = 95) : regression_score x = 596 :=
by
  rw [h]
  -- skipping the actual calculation steps
  sorry

end estimated_total_score_l1547_154761


namespace minimum_value_of_z_l1547_154782

theorem minimum_value_of_z
  (x y : ℝ)
  (h1 : 3 * x + y - 6 ≥ 0)
  (h2 : x - y - 2 ≤ 0)
  (h3 : y - 3 ≤ 0) :
  ∃ z, z = 4 * x + y ∧ z = 7 :=
sorry

end minimum_value_of_z_l1547_154782


namespace no_real_solutions_l1547_154794

theorem no_real_solutions (x : ℝ) : 
  x^(Real.log x / Real.log 2) ≠ x^4 / 256 :=
by
  sorry

end no_real_solutions_l1547_154794


namespace suma_work_rate_l1547_154731

theorem suma_work_rate (W : ℕ) : 
  (∀ W, (W / 6) + (W / S) = W / 4) → S = 24 :=
by
  intro h
  -- detailed proof would actually go here
  sorry

end suma_work_rate_l1547_154731


namespace CD_is_b_minus_a_minus_c_l1547_154749

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (a b c : V)

def AB : V := a
def AD : V := b
def BC : V := c

theorem CD_is_b_minus_a_minus_c (h1 : A + a = B) (h2 : A + b = D) (h3 : B + c = C) :
  D - C = b - a - c :=
by sorry

end CD_is_b_minus_a_minus_c_l1547_154749


namespace cubic_expression_solution_l1547_154744

theorem cubic_expression_solution (r s : ℝ) (h₁ : 3 * r^2 - 4 * r - 7 = 0) (h₂ : 3 * s^2 - 4 * s - 7 = 0) :
  (3 * r^3 - 3 * s^3) / (r - s) = 37 / 3 :=
sorry

end cubic_expression_solution_l1547_154744


namespace quadratic_has_two_distinct_roots_l1547_154757

theorem quadratic_has_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2*x₁ + k = 0) ∧ (x₂^2 - 2*x₂ + k = 0))
  ↔ k < 1 :=
by sorry

end quadratic_has_two_distinct_roots_l1547_154757


namespace fraction_color_films_l1547_154776

variables {x y : ℕ} (h₁ : y ≠ 0) (h₂ : x ≠ 0)

theorem fraction_color_films (h₃ : 30 * x > 0) (h₄ : 6 * y > 0) :
  (6 * y : ℚ) / ((3 * y / 10) + 6 * y) = 20 / 21 := by
  sorry

end fraction_color_films_l1547_154776


namespace usual_time_to_school_l1547_154785

theorem usual_time_to_school (S T t : ℝ) (h : 1.2 * S * (T - t) = S * T) : T = 6 * t :=
by
  sorry

end usual_time_to_school_l1547_154785


namespace f_f_f_three_l1547_154723

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 1

theorem f_f_f_three : f (f (f 3)) = 43 :=
by
  -- Introduction of definitions and further necessary steps here are skipped
  sorry

end f_f_f_three_l1547_154723


namespace value_of_f_three_l1547_154789

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * Real.cos x - x

theorem value_of_f_three (a b : ℝ) (h : f a b (-3) = 7) : f a b 3 = 1 :=
by
  sorry

end value_of_f_three_l1547_154789


namespace initial_violet_balloons_l1547_154759

-- Let's define the given conditions
def red_balloons : ℕ := 4
def violet_balloons_lost : ℕ := 3
def violet_balloons_now : ℕ := 4

-- Define the statement to prove
theorem initial_violet_balloons :
  (violet_balloons_now + violet_balloons_lost) = 7 :=
by
  sorry

end initial_violet_balloons_l1547_154759


namespace star_24_75_l1547_154755

noncomputable def star (a b : ℝ) : ℝ := sorry 

-- Conditions
axiom star_one_one : star 1 1 = 2
axiom star_ab_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : star (a * b) b = a * (star b b)
axiom star_a_one (a : ℝ) (h : 0 < a) : star a 1 = 2 * a

-- Theorem to prove
theorem star_24_75 : star 24 75 = 1800 := 
by 
  sorry

end star_24_75_l1547_154755


namespace prob_of_selecting_blue_ball_l1547_154760

noncomputable def prob_select_ball :=
  let prob_X := 1 / 3
  let prob_Y := 1 / 3
  let prob_Z := 1 / 3
  let prob_blue_X := 7 / 10
  let prob_blue_Y := 1 / 2
  let prob_blue_Z := 2 / 5
  prob_X * prob_blue_X + prob_Y * prob_blue_Y + prob_Z * prob_blue_Z

theorem prob_of_selecting_blue_ball :
  prob_select_ball = 8 / 15 :=
by
  -- Provide the proof here
  sorry

end prob_of_selecting_blue_ball_l1547_154760


namespace cupcakes_difference_l1547_154714

theorem cupcakes_difference (h : ℕ) (betty_rate : ℕ) (dora_rate : ℕ) (betty_break : ℕ) 
  (cupcakes_difference : ℕ) 
  (H₁ : betty_rate = 10) 
  (H₂ : dora_rate = 8) 
  (H₃ : betty_break = 2) 
  (H₄ : cupcakes_difference = 10) : 
  8 * h - 10 * (h - 2) = 10 → h = 5 :=
by
  intro H
  sorry

end cupcakes_difference_l1547_154714


namespace count_harmonic_vals_l1547_154777

def floor (x : ℝ) : ℤ := sorry -- or use Mathlib function
def frac (x : ℝ) : ℝ := x - (floor x)

def is_harmonic_progression (a b c : ℝ) : Prop := 
  (1 / a) = (2 / b) - (1 / c)

theorem count_harmonic_vals :
  (∃ x, is_harmonic_progression x (floor x) (frac x)) ∧
  (∃! x1 x2, is_harmonic_progression x1 (floor x1) (frac x1) ∧
               is_harmonic_progression x2 (floor x2) (frac x2)) ∧
  x1 ≠ x2 :=
  sorry

end count_harmonic_vals_l1547_154777


namespace arithmetic_sequence_k_l1547_154706

theorem arithmetic_sequence_k :
  ∀ (a : ℕ → ℤ) (d : ℤ) (k : ℕ),
  d ≠ 0 →
  (∀ n : ℕ, a n = a 0 + n * d) →
  a 0 = 0 →
  a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 →
  k = 22 :=
by
  intros a d k hdnz h_arith h_a1_zero h_ak_sum
  sorry

end arithmetic_sequence_k_l1547_154706


namespace no_solutions_system_l1547_154793

theorem no_solutions_system :
  ∀ (x y : ℝ), 
  (x^3 + x + y + 1 = 0) →
  (y * x^2 + x + y = 0) →
  (y^2 + y - x^2 + 1 = 0) →
  false :=
by
  intro x y h1 h2 h3
  -- Proof goes here
  sorry

end no_solutions_system_l1547_154793


namespace points_satisfying_clubsuit_l1547_154772

def clubsuit (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem points_satisfying_clubsuit (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x + y = 0) :=
by
  sorry

end points_satisfying_clubsuit_l1547_154772


namespace find_k_l1547_154771

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + 2 * k = 0 ∧ x = 1) : k = 1 / 2 :=
by {
  sorry 
}

end find_k_l1547_154771


namespace solve_for_x_l1547_154724

theorem solve_for_x (x y z : ℝ) 
  (h1 : x * y + 3 * x + 2 * y = 12) 
  (h2 : y * z + 5 * y + 3 * z = 15) 
  (h3 : x * z + 5 * x + 4 * z = 40) :
  x = 4 :=
by
  sorry

end solve_for_x_l1547_154724


namespace prism_properties_sum_l1547_154725

/-- Prove that the sum of the number of edges, corners, and faces of a rectangular box (prism) with dimensions 2 by 3 by 4 is 26. -/
theorem prism_properties_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := 
by
  -- Provided conditions and definitions
  let edges := 12
  let corners := 8
  let faces := 6
  -- Summing up these values
  exact rfl

end prism_properties_sum_l1547_154725


namespace x_plus_y_l1547_154748

variables {e1 e2 : ℝ → ℝ → Prop} -- Represents the vectors as properties of reals
variables {x y : ℝ} -- Real numbers x and y

-- Assuming non-collinearity of e1 and e2 (This means e1 and e2 are independent)
axiom non_collinear : e1 ≠ e2 

-- Given condition translated into Lean
axiom main_equation : (3 * x - 4 * y = 6) ∧ (2 * x - 3 * y = 3)

-- Prove that x + y = 9
theorem x_plus_y : x + y = 9 := 
by
  sorry -- Proof will be provided here

end x_plus_y_l1547_154748


namespace intersection_points_l1547_154726

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := -f x
noncomputable def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points :
  let a := 2
  let b := 1
  10 * a + b = 21 :=
by
  sorry

end intersection_points_l1547_154726


namespace merchant_profit_percentage_l1547_154730

theorem merchant_profit_percentage (C S : ℝ) (h : 24 * C = 16 * S) : ((S - C) / C) * 100 = 50 := by
  -- Adding "by" to denote beginning of proof section
  sorry  -- Proof is skipped

end merchant_profit_percentage_l1547_154730


namespace largest_possible_d_l1547_154716

theorem largest_possible_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 := 
sorry

end largest_possible_d_l1547_154716


namespace smallest_number_ending_in_6_moved_front_gives_4_times_l1547_154754

theorem smallest_number_ending_in_6_moved_front_gives_4_times (x m n : ℕ) 
  (h1 : n = 10 * x + 6)
  (h2 : 6 * 10^m + x = 4 * n) :
  n = 1538466 :=
by
  sorry

end smallest_number_ending_in_6_moved_front_gives_4_times_l1547_154754


namespace minimum_value_is_correct_l1547_154783

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024

theorem minimum_value_is_correct (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (minimum_value x y) ≥ -2050208 := 
sorry

end minimum_value_is_correct_l1547_154783


namespace triangles_form_even_square_l1547_154707

-- Given conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def triangle_area (b h : ℕ) : ℚ :=
  (b * h) / 2

-- Statement of the problem
theorem triangles_form_even_square (n : ℕ) :
  (∀ t : Fin n, is_right_triangle 3 4 5 ∧ triangle_area 3 4 = 6) →
  (∃ a : ℕ, a^2 = 6 * n) →
  Even n :=
by
  sorry

end triangles_form_even_square_l1547_154707


namespace coin_probability_l1547_154738

theorem coin_probability (p : ℝ) (h1 : p < 1/2) (h2 : (Nat.choose 6 3) * p^3 * (1-p)^3 = 1/20) : p = 1/400 := sorry

end coin_probability_l1547_154738


namespace range_of_a_l1547_154779

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l1547_154779


namespace matrix_mult_7_l1547_154732

theorem matrix_mult_7 (M : Matrix (Fin 3) (Fin 3) ℝ) (v : Fin 3 → ℝ) : 
  (∀ v, M.mulVec v = (7 : ℝ) • v) ↔ M = 7 • 1 :=
by
  sorry

end matrix_mult_7_l1547_154732


namespace parabola_directrix_standard_eq_l1547_154763

theorem parabola_directrix_standard_eq (y : ℝ) (x : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ ∀ (P : {P // P ≠ x ∨ P ≠ y}), 
  (y + 1) = p) → x^2 = 4 * y :=
sorry

end parabola_directrix_standard_eq_l1547_154763


namespace cupcakes_frosted_in_10_minutes_l1547_154770

theorem cupcakes_frosted_in_10_minutes (r1 r2 time : ℝ) (cagney_rate lacey_rate : r1 = 1 / 15 ∧ r2 = 1 / 25)
  (time_in_seconds : time = 600) :
  (1 / ((1 / r1) + (1 / r2)) * time) = 64 := by
  sorry

end cupcakes_frosted_in_10_minutes_l1547_154770


namespace area_of_largest_medallion_is_314_l1547_154720

noncomputable def largest_medallion_area_in_square (side: ℝ) (π: ℝ) : ℝ :=
  let diameter := side
  let radius := diameter / 2
  let area := π * radius^2
  area

theorem area_of_largest_medallion_is_314 :
  largest_medallion_area_in_square 20 3.14 = 314 := 
  sorry

end area_of_largest_medallion_is_314_l1547_154720


namespace find_functional_solution_l1547_154710

theorem find_functional_solution (c : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) :
  ∀ x : ℝ, f x = x ^ 3 + c * x := by
  sorry

end find_functional_solution_l1547_154710


namespace price_of_sundae_l1547_154708

theorem price_of_sundae (total_ice_cream_bars : ℕ) (total_sundae_price : ℝ)
                        (total_price : ℝ) (price_per_ice_cream_bar : ℝ) (num_ice_cream_bars : ℕ) (num_sundaes : ℕ)
                        (h1 : total_ice_cream_bars = num_ice_cream_bars)
                        (h2 : total_price = 200)
                        (h3 : price_per_ice_cream_bar = 0.40)
                        (h4 : num_ice_cream_bars = 200)
                        (h5 : num_sundaes = 200)
                        (h6 : total_ice_cream_bars * price_per_ice_cream_bar + total_sundae_price = total_price) :
  total_sundae_price / num_sundaes = 0.60 :=
sorry

end price_of_sundae_l1547_154708


namespace find_m_l1547_154762

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -13.6 :=
by sorry

end find_m_l1547_154762


namespace necessary_but_not_sufficient_l1547_154790

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (x > 1) ∨ (x ≤ -1) := 
by 
  sorry

end necessary_but_not_sufficient_l1547_154790


namespace total_spent_is_64_l1547_154767

def deck_price : ℕ := 8
def victors_decks : ℕ := 6
def friends_decks : ℕ := 2

def victors_spending : ℕ := victors_decks * deck_price
def friends_spending : ℕ := friends_decks * deck_price
def total_spending : ℕ := victors_spending + friends_spending

theorem total_spent_is_64 : total_spending = 64 := by
  sorry

end total_spent_is_64_l1547_154767


namespace alcohol_to_water_ratio_l1547_154709

theorem alcohol_to_water_ratio (V p q : ℝ) (hV : V > 0) (hp : p > 0) (hq : q > 0) :
  let alcohol_first_jar := (p / (p + 1)) * V
  let water_first_jar   := (1 / (p + 1)) * V
  let alcohol_second_jar := (2 * q / (q + 1)) * V
  let water_second_jar   := (2 / (q + 1)) * V
  let total_alcohol := alcohol_first_jar + alcohol_second_jar
  let total_water := water_first_jar + water_second_jar
  (total_alcohol / total_water) = ((p * (q + 1) + 2 * p + 2 * q) / (q + 1 + 2 * p + 2)) :=
by
  sorry

end alcohol_to_water_ratio_l1547_154709


namespace book_arrangements_l1547_154769

theorem book_arrangements (n : ℕ) (b1 b2 b3 b4 b5 : ℕ) (h_b123 : b1 < b2 ∧ b2 < b3):
  n = 20 := sorry

end book_arrangements_l1547_154769


namespace closed_under_all_operations_l1547_154717

structure sqrt2_num where
  re : ℚ
  im : ℚ

namespace sqrt2_num

def add (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re + y.re, x.im + y.im⟩

def subtract (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re - y.re, x.im - y.im⟩

def multiply (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re * y.re + 2 * x.im * y.im, x.re * y.im + x.im * y.re⟩

def divide (x y : sqrt2_num) : sqrt2_num :=
  let denom := y.re^2 - 2 * y.im^2
  ⟨(x.re * y.re - 2 * x.im * y.im) / denom, (x.im * y.re - x.re * y.im) / denom⟩

theorem closed_under_all_operations (a b c d : ℚ) :
  ∃ (e f : ℚ), 
    add ⟨a, b⟩ ⟨c, d⟩ = ⟨e, f⟩ ∧ 
    ∃ (g h : ℚ), 
    subtract ⟨a, b⟩ ⟨c, d⟩ = ⟨g, h⟩ ∧ 
    ∃ (i j : ℚ), 
    multiply ⟨a, b⟩ ⟨c, d⟩ = ⟨i, j⟩ ∧ 
    ∃ (k l : ℚ), 
    divide ⟨a, b⟩ ⟨c, d⟩ = ⟨k, l⟩ := by
  sorry

end sqrt2_num

end closed_under_all_operations_l1547_154717


namespace find_integer_cube_sum_l1547_154713

-- Define the problem in Lean
theorem find_integer_cube_sum : ∃ n : ℤ, n^3 = (n-1)^3 + (n-2)^3 + (n-3)^3 := by
  use 6
  sorry

end find_integer_cube_sum_l1547_154713


namespace gum_cost_700_eq_660_cents_l1547_154743

-- defining the cost function
def gum_cost (n : ℕ) : ℝ :=
  if n ≤ 500 then n * 0.01
  else 5 + (n - 500) * 0.008

-- proving the specific case for 700 pieces of gum
theorem gum_cost_700_eq_660_cents : gum_cost 700 = 6.60 := by
  sorry

end gum_cost_700_eq_660_cents_l1547_154743


namespace tetrahedrons_from_triangular_prism_l1547_154736

theorem tetrahedrons_from_triangular_prism : 
  let n := 6
  let choose4 := Nat.choose n 4
  let coplanar_cases := 3
  choose4 - coplanar_cases = 12 := by
  sorry

end tetrahedrons_from_triangular_prism_l1547_154736


namespace identify_clothing_l1547_154745

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l1547_154745


namespace truck_travel_distance_l1547_154715

variable (d1 d2 g1 g2 : ℝ)
variable (rate : ℝ)

-- Define the conditions
axiom condition1 : d1 = 300
axiom condition2 : g1 = 10
axiom condition3 : rate = d1 / g1
axiom condition4 : g2 = 15

-- Define the goal
theorem truck_travel_distance : d2 = rate * g2 := by
  -- axiom assumption placeholder
  exact sorry

end truck_travel_distance_l1547_154715


namespace find_y_l1547_154798

theorem find_y (y z : ℕ) (h1 : 50 = y * 10) (h2 : 300 = 50 * z) : y = 5 :=
by
  sorry

end find_y_l1547_154798
