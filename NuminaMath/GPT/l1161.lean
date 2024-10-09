import Mathlib

namespace temp_below_zero_negative_l1161_116169

theorem temp_below_zero_negative (temp_below_zero : ℤ) : temp_below_zero = -3 ↔ temp_below_zero < 0 := by
  sorry

end temp_below_zero_negative_l1161_116169


namespace find_sum_invested_l1161_116195

theorem find_sum_invested (P : ℝ) 
  (SI_1: ℝ) (SI_2: ℝ)
  (h1 : SI_1 = P * (15 / 100) * 2)
  (h2 : SI_2 = P * (12 / 100) * 2)
  (h3 : SI_1 - SI_2 = 900) :
  P = 15000 := by
sorry

end find_sum_invested_l1161_116195


namespace triangle_area_rational_l1161_116167

theorem triangle_area_rational
  (x1 y1 x2 y2 x3 y3 : ℤ)
  (h : y1 = y2) :
  ∃ (k : ℚ), 
    k = abs ((x2 - x1) * y3) / 2 := sorry

end triangle_area_rational_l1161_116167


namespace schools_participating_l1161_116102

noncomputable def num_schools (students_per_school : ℕ) (total_students : ℕ) : ℕ :=
  total_students / students_per_school

theorem schools_participating (students_per_school : ℕ) (beth_rank : ℕ) 
  (carla_rank : ℕ) (highest_on_team : ℕ) (n : ℕ) :
  students_per_school = 4 ∧ beth_rank = 46 ∧ carla_rank = 79 ∧
  (∀ i, i ≤ 46 → highest_on_team = 40) → 
  num_schools students_per_school ((2 * highest_on_team) - 1) = 19 := 
by
  intros h
  sorry

end schools_participating_l1161_116102


namespace fifty_times_reciprocal_of_eight_times_number_three_l1161_116107

theorem fifty_times_reciprocal_of_eight_times_number_three (x : ℚ) 
  (h : 8 * x = 3) : 50 * (1 / x) = 133 + 1 / 3 :=
sorry

end fifty_times_reciprocal_of_eight_times_number_three_l1161_116107


namespace MrSami_sold_20_shares_of_stock_x_l1161_116186

theorem MrSami_sold_20_shares_of_stock_x
    (shares_v : ℕ := 68)
    (shares_w : ℕ := 112)
    (shares_x : ℕ := 56)
    (shares_y : ℕ := 94)
    (shares_z : ℕ := 45)
    (additional_shares_y : ℕ := 23)
    (increase_in_range : ℕ := 14)
    : (shares_x - (shares_y + additional_shares_y - ((shares_w - shares_z + increase_in_range) - shares_y - additional_shares_y)) = 20) :=
by
  sorry

end MrSami_sold_20_shares_of_stock_x_l1161_116186


namespace boat_speed_in_still_water_l1161_116168

-- Definitions and conditions
def Vs : ℕ := 5  -- Speed of the stream in km/hr
def distance : ℕ := 135  -- Distance traveled in km
def time : ℕ := 5  -- Time in hours

-- Statement to prove
theorem boat_speed_in_still_water : 
  ((distance = (Vb + Vs) * time) -> Vb = 22) :=
by
  sorry

end boat_speed_in_still_water_l1161_116168


namespace positive_reals_power_equality_l1161_116139

open Real

theorem positive_reals_power_equality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : a < 1) : a = b := 
  by
  sorry

end positive_reals_power_equality_l1161_116139


namespace number_of_crystals_in_container_l1161_116154

-- Define the dimensions of the energy crystal
def length_crystal := 30
def width_crystal := 25
def height_crystal := 5

-- Define the dimensions of the cubic container
def side_container := 27

-- Volume of the cubic container
def volume_container := side_container ^ 3

-- Volume of the energy crystal
def volume_crystal := length_crystal * width_crystal * height_crystal

-- Proof statement
theorem number_of_crystals_in_container :
  volume_container / volume_crystal ≥ 5 :=
sorry

end number_of_crystals_in_container_l1161_116154


namespace total_sum_lent_l1161_116157

theorem total_sum_lent (x : ℚ) (second_part : ℚ) (total_sum : ℚ) (h : second_part = 1688) 
  (h_interest : x * 3/100 * 8 = second_part * 5/100 * 3) : total_sum = 2743 :=
by
  sorry

end total_sum_lent_l1161_116157


namespace find_value_of_m_l1161_116193

theorem find_value_of_m (m : ℤ) (x : ℤ) (h : (x - 3 ≠ 0) ∧ (x = 3)) : 
  ((x - 1) / (x - 3) = m / (x - 3)) → m = 2 :=
by
  sorry

end find_value_of_m_l1161_116193


namespace find_functions_l1161_116111

noncomputable def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (p q : ℝ), p ≠ q → (f q - f p) / (q - p) * 0 + f p - (f q - f p) / (q - p) * p = p * q

theorem find_functions (f : ℝ → ℝ) (c : ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = x * (c + x)) :=
by
  intros
  sorry

end find_functions_l1161_116111


namespace goldfish_below_surface_l1161_116164

theorem goldfish_below_surface (Toby_counts_at_surface : ℕ) (percentage_at_surface : ℝ) (total_goldfish : ℕ) (below_surface : ℕ) :
    (Toby_counts_at_surface = 15 ∧ percentage_at_surface = 0.25 ∧ Toby_counts_at_surface = percentage_at_surface * total_goldfish ∧ below_surface = total_goldfish - Toby_counts_at_surface) →
    below_surface = 45 :=
by
  sorry

end goldfish_below_surface_l1161_116164


namespace ratio_of_m_l1161_116133

theorem ratio_of_m (a b m m1 m2 : ℚ) 
  (h1 : a^2 - 2*a + (3/m) = 0)
  (h2 : a + b = 2 - 2/m)
  (h3 : a * b = 3/m)
  (h4 : (a/b) + (b/a) = 3/2) 
  (h5 : 8 * m^2 - 31 * m + 8 = 0)
  (h6 : m1 + m2 = 31/8)
  (h7 : m1 * m2 = 1) :
  (m1/m2) + (m2/m1) = 833/64 :=
sorry

end ratio_of_m_l1161_116133


namespace intersection_complement_N_l1161_116159

def is_universal_set (R : Set ℝ) : Prop := ∀ x : ℝ, x ∈ R

def is_complement (U S C : Set ℝ) : Prop := 
  ∀ x : ℝ, x ∈ C ↔ x ∈ U ∧ x ∉ S

theorem intersection_complement_N 
  (U M N C : Set ℝ)
  (h_universal : is_universal_set U)
  (hM : M = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (hN : N = {x : ℝ | x < 1})
  (h_compl : is_complement U M C) :
  (C ∩ N) = {x : ℝ | x < -2} := 
by 
  sorry

end intersection_complement_N_l1161_116159


namespace no_solution_for_inequalities_l1161_116122

theorem no_solution_for_inequalities (x : ℝ) : ¬ ((6 * x - 2 < (x + 2) ^ 2) ∧ ((x + 2) ^ 2 < 9 * x - 5)) :=
by sorry

end no_solution_for_inequalities_l1161_116122


namespace alcohol_solution_l1161_116137

/-- 
A 40-liter solution of alcohol and water is 5 percent alcohol. If 3.5 liters of alcohol and 6.5 liters of water are added to this solution, 
what percent of the solution produced is alcohol? 
-/
theorem alcohol_solution (original_volume : ℝ) (original_percent_alcohol : ℝ)
                        (added_alcohol : ℝ) (added_water : ℝ) :
  original_volume = 40 →
  original_percent_alcohol = 5 →
  added_alcohol = 3.5 →
  added_water = 6.5 →
  (100 * (original_volume * original_percent_alcohol / 100 + added_alcohol) / (original_volume + added_alcohol + added_water)) = 11 := 
by 
  intros h1 h2 h3 h4
  sorry

end alcohol_solution_l1161_116137


namespace area_of_overlap_l1161_116177

def area_of_square_1 : ℝ := 1
def area_of_square_2 : ℝ := 4
def area_of_square_3 : ℝ := 9
def area_of_square_4 : ℝ := 16
def total_area_of_rectangle : ℝ := 27.5
def unshaded_area : ℝ := 1.5

def total_area_of_squares : ℝ := area_of_square_1 + area_of_square_2 + area_of_square_3 + area_of_square_4
def total_area_covered_by_squares : ℝ := total_area_of_rectangle - unshaded_area

theorem area_of_overlap :
  total_area_of_squares - total_area_covered_by_squares = 4 := 
sorry

end area_of_overlap_l1161_116177


namespace proof_problem_l1161_116189

noncomputable def calc_a_star_b (a b : ℤ) : ℚ :=
1 / (a:ℚ) + 1 / (b:ℚ)

theorem proof_problem (a b : ℤ) (h1 : a + b = 10) (h2 : a * b = 24) :
  calc_a_star_b a b = 5 / 12 ∧ (a * b > a + b) := by
  sorry

end proof_problem_l1161_116189


namespace stephanie_fewer_forks_l1161_116198

noncomputable def fewer_forks := 
  (60 - 44) / 4

theorem stephanie_fewer_forks : fewer_forks = 4 := by
  sorry

end stephanie_fewer_forks_l1161_116198


namespace fraction_integer_condition_special_integers_l1161_116145

theorem fraction_integer_condition (p : ℕ) (h : (p + 2) % (p + 1) = 0) : p = 2 :=
by
  sorry

theorem special_integers (N : ℕ) (h1 : ∀ q : ℕ, N = 2 ^ p * 3 ^ q ∧ (2 * p + 1) * (2 * q + 1) = 3 * (p + 1) * (q + 1)) : 
  N = 144 ∨ N = 324 :=
by
  sorry

end fraction_integer_condition_special_integers_l1161_116145


namespace arithmetic_sequence_sum_l1161_116138

theorem arithmetic_sequence_sum (S : ℕ → ℕ)
  (h₁ : S 3 = 9)
  (h₂ : S 6 = 36) :
  S 9 - S 6 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l1161_116138


namespace cost_of_each_item_number_of_purchasing_plans_l1161_116152

-- Question 1: Cost of each item
theorem cost_of_each_item : 
  ∃ (x y : ℕ), 
    (10 * x + 5 * y = 2000) ∧ 
    (5 * x + 3 * y = 1050) ∧ 
    (x = 150) ∧ 
    (y = 100) :=
by
    sorry

-- Question 2: Number of different purchasing plans
theorem number_of_purchasing_plans : 
  (∀ (a b : ℕ), 
    (150 * a + 100 * b = 4000) → 
    (a ≥ 12) → 
    (b ≥ 12) → 
    (4 = 4)) :=
by
    sorry

end cost_of_each_item_number_of_purchasing_plans_l1161_116152


namespace percent_of_g_is_h_l1161_116160

variable (a b c d e f g h : ℝ)

-- Conditions
def cond1a : f = 0.60 * a := sorry
def cond1b : f = 0.45 * b := sorry
def cond2a : g = 0.70 * b := sorry
def cond2b : g = 0.30 * c := sorry
def cond3a : h = 0.80 * c := sorry
def cond3b : h = 0.10 * f := sorry
def cond4a : c = 0.30 * a := sorry
def cond4b : c = 0.25 * b := sorry
def cond5a : d = 0.40 * a := sorry
def cond5b : d = 0.35 * b := sorry
def cond6a : e = 0.50 * b := sorry
def cond6b : e = 0.20 * c := sorry

-- Theorem to prove
theorem percent_of_g_is_h (h_percent_g : ℝ) 
  (h_formula : h = h_percent_g * g) : 
  h = 0.285714 * g :=
by
  sorry

end percent_of_g_is_h_l1161_116160


namespace quadratic_min_value_unique_l1161_116146

theorem quadratic_min_value_unique {a b c : ℝ} (h : a > 0) :
  (∀ x : ℝ, 3 * x^2 - 8 * x + 7 ≥ 3 * (4 / 3)^2 - 8 * (4 / 3) + 7) → 
  ∃ x : ℝ, x = 4 / 3 :=
by
  sorry

end quadratic_min_value_unique_l1161_116146


namespace Tyler_scissors_count_l1161_116106

variable (S : ℕ)

def Tyler_initial_money : ℕ := 100
def cost_per_scissors : ℕ := 5
def number_of_erasers : ℕ := 10
def cost_per_eraser : ℕ := 4
def Tyler_remaining_money : ℕ := 20

theorem Tyler_scissors_count :
  Tyler_initial_money - (cost_per_scissors * S + number_of_erasers * cost_per_eraser) = Tyler_remaining_money →
  S = 8 :=
by
  sorry

end Tyler_scissors_count_l1161_116106


namespace a_le_neg4_l1161_116174

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

noncomputable def h (a x : ℝ) : ℝ := f x - g a x

-- Theorem
theorem a_le_neg4 (a : ℝ) : 
  (∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 > 0 → x2 > 0 → (h a x1 - h a x2) / (x1 - x2) > 2) →
  a ≤ -4 :=
by
  sorry

end a_le_neg4_l1161_116174


namespace min_value_expression_l1161_116143

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  ∃ c, ∀ x y, 0 < x → 0 < y → x + y = 1 → c = 9 ∧ ((1 / x) + (4 / y)) ≥ 9 := 
sorry

end min_value_expression_l1161_116143


namespace solve_for_x_l1161_116179

theorem solve_for_x (x : ℝ) (h1 : x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 := sorry

end solve_for_x_l1161_116179


namespace solve_for_x_l1161_116110

theorem solve_for_x : ∃ x : ℝ, 3 * x - 48.2 = 0.25 * (4 * x + 56.8) → x = 31.2 :=
by sorry

end solve_for_x_l1161_116110


namespace M_subset_P_l1161_116109

def M := {x : ℕ | ∃ a : ℕ, 0 < a ∧ x = a^2 + 1}
def P := {y : ℕ | ∃ b : ℕ, 0 < b ∧ y = b^2 - 4*b + 5}

theorem M_subset_P : M ⊂ P :=
by
  sorry

end M_subset_P_l1161_116109


namespace major_airlines_wifi_l1161_116142

-- Definitions based on conditions
def percentage (x : ℝ) := 0 ≤ x ∧ x ≤ 100

variables (W S B : ℝ)

-- Assume the conditions
axiom H1 : S = 70
axiom H2 : B = 45
axiom H3 : B ≤ S

-- The final proof problem that W = 45
theorem major_airlines_wifi : W = B :=
by
  sorry

end major_airlines_wifi_l1161_116142


namespace circle_center_sum_l1161_116149

theorem circle_center_sum (x y : ℝ) (h : (x - 2)^2 + (y + 1)^2 = 15) : x + y = 1 :=
sorry

end circle_center_sum_l1161_116149


namespace jeans_price_increase_l1161_116188

theorem jeans_price_increase 
  (C : ℝ) 
  (R : ℝ) 
  (F : ℝ) 
  (H1 : R = 1.40 * C)
  (H2 : F = 1.82 * C) 
  : (F - C) / C * 100 = 82 := 
sorry

end jeans_price_increase_l1161_116188


namespace part_I_part_II_l1161_116112

namespace VectorProblems

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem part_I (m : ℝ) :
  let u := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
  let v := (4 * m + vector_b.1, m + vector_b.2)
  dot_product u v > 0 →
  m ≠ 4 / 7 →
  m > -1 / 2 :=
sorry

theorem part_II (k : ℝ) :
  let u := (vector_a.1 + 4 * k, vector_a.2 + k)
  let v := (2 * vector_b.1 - vector_a.1, 2 * vector_b.2 - vector_a.2)
  dot_product u v = 0 →
  k = -11 / 18 :=
sorry

end VectorProblems

end part_I_part_II_l1161_116112


namespace books_left_over_after_repacking_l1161_116197

theorem books_left_over_after_repacking :
  ((1335 * 39) % 40) = 25 :=
sorry

end books_left_over_after_repacking_l1161_116197


namespace polynomial_complete_square_l1161_116156

theorem polynomial_complete_square :
  ∃ a h k : ℝ, (∀ x : ℝ, 4 * x^2 - 12 * x + 1 = a * (x - h)^2 + k) ∧ a + h + k = -2.5 := by
  sorry

end polynomial_complete_square_l1161_116156


namespace solve_congruence_l1161_116190

theorem solve_congruence :
  ∃ a m : ℕ, m ≥ 2 ∧ a < m ∧ a + m = 27 ∧ (10 * x + 3 ≡ 7 [MOD 15]) → x ≡ 12 [MOD 15] := 
by
  sorry

end solve_congruence_l1161_116190


namespace handshake_problem_l1161_116147

noncomputable def number_of_handshakes (n : ℕ) : ℕ :=
  n.choose 2

theorem handshake_problem : number_of_handshakes 25 = 300 := 
  by
  sorry

end handshake_problem_l1161_116147


namespace annulus_area_of_tangent_segments_l1161_116128

theorem annulus_area_of_tangent_segments (r : ℝ) (l : ℝ) (region_area : ℝ) 
  (h_rad : r = 3) (h_len : l = 6) : region_area = 9 * Real.pi :=
sorry

end annulus_area_of_tangent_segments_l1161_116128


namespace luke_number_of_rounds_l1161_116132

variable (points_per_round total_points : ℕ)

theorem luke_number_of_rounds 
  (h1 : points_per_round = 3)
  (h2 : total_points = 78) : 
  total_points / points_per_round = 26 := 
by 
  sorry

end luke_number_of_rounds_l1161_116132


namespace complex_div_imaginary_unit_eq_l1161_116166

theorem complex_div_imaginary_unit_eq :
  (∀ i : ℂ, i^2 = -1 → (1 / (1 + i)) = ((1 - i) / 2)) :=
by
  intro i
  intro hi
  /- The proof will be inserted here -/
  sorry

end complex_div_imaginary_unit_eq_l1161_116166


namespace lucien_balls_count_l1161_116161

theorem lucien_balls_count (lucca_balls : ℕ) (lucca_percent_basketballs : ℝ) (lucien_percent_basketballs : ℝ) (total_basketballs : ℕ)
  (h1 : lucca_balls = 100)
  (h2 : lucca_percent_basketballs = 0.10)
  (h3 : lucien_percent_basketballs = 0.20)
  (h4 : total_basketballs = 50) :
  ∃ lucien_balls : ℕ, lucien_balls = 200 :=
by
  sorry

end lucien_balls_count_l1161_116161


namespace equation_nth_position_l1161_116121

theorem equation_nth_position (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 :=
by
  sorry

end equation_nth_position_l1161_116121


namespace cannot_achieve_80_cents_l1161_116181

def is_possible_value (n : ℕ) : Prop :=
  ∃ (n_nickels n_dimes n_quarters n_half_dollars : ℕ), 
    n_nickels + n_dimes + n_quarters + n_half_dollars = 5 ∧
    5 * n_nickels + 10 * n_dimes + 25 * n_quarters + 50 * n_half_dollars = n

theorem cannot_achieve_80_cents : ¬ is_possible_value 80 :=
by sorry

end cannot_achieve_80_cents_l1161_116181


namespace coin_loading_impossible_l1161_116127

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l1161_116127


namespace algebraic_expression_value_l1161_116178

-- Define the premises as a Lean statement
theorem algebraic_expression_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a * (b + c) + b * (a + c) + c * (a + b) = -1 :=
sorry

end algebraic_expression_value_l1161_116178


namespace eval_expression_l1161_116115

theorem eval_expression (a : ℕ) (h : a = 2) : 
  8^3 + 4 * a * 8^2 + 6 * a^2 * 8 + a^3 = 1224 := 
by
  rw [h]
  sorry

end eval_expression_l1161_116115


namespace curve_intersects_every_plane_l1161_116151

theorem curve_intersects_every_plane (A B C D : ℝ) (h : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0) :
  ∃ t : ℝ, A * t + B * t^3 + C * t^5 + D = 0 :=
by
  sorry

end curve_intersects_every_plane_l1161_116151


namespace compound_interest_correct_l1161_116180

-- define the problem conditions
def P : ℝ := 3000
def r : ℝ := 0.07
def n : ℕ := 25

-- the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- state the theorem we want to prove
theorem compound_interest_correct :
  compound_interest P r n = 16281 := 
by
  sorry

end compound_interest_correct_l1161_116180


namespace password_probability_l1161_116162

theorem password_probability :
  let even_digits := [0, 2, 4, 6, 8]
  let vowels := ['A', 'E', 'I', 'O', 'U']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  (even_digits.length / 10) * (vowels.length / 26) * (non_zero_digits.length / 10) = 9 / 52 :=
by
  sorry

end password_probability_l1161_116162


namespace vertex_not_neg2_2_l1161_116113

theorem vertex_not_neg2_2 (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : a * 1^2 + b * 1 + c = 0)
  (hsymm : ∀ x y, y = a * x^2 + b * x + c → y = a * (4 - x)^2 + b * (4 - x) + c) :
  ¬ ((-b) / (2 * a) = -2 ∧ a * (-2)^2 + b * (-2) + c = 2) :=
by
  sorry

end vertex_not_neg2_2_l1161_116113


namespace line_intersects_circle_l1161_116163

noncomputable def line_eqn (a : ℝ) (x y : ℝ) : ℝ := a * x - y - a + 3
noncomputable def circle_eqn (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x - 2 * y - 4

-- Given the line l passes through M(1, 3)
def passes_through_M (a : ℝ) : Prop := line_eqn a 1 3 = 0

-- Given M(1, 3) is inside the circle
def M_inside_circle : Prop := circle_eqn 1 3 < 0

-- To prove the line intersects the circle
theorem line_intersects_circle (a : ℝ) (h1 : passes_through_M a) (h2 : M_inside_circle) : 
  ∃ p : ℝ × ℝ, line_eqn a p.1 p.2 = 0 ∧ circle_eqn p.1 p.2 = 0 :=
sorry

end line_intersects_circle_l1161_116163


namespace frog_eats_per_day_l1161_116129

-- Definition of the constants
def flies_morning : ℕ := 5
def flies_afternoon : ℕ := 6
def escaped_flies : ℕ := 1
def weekly_required_flies : ℕ := 14
def days_in_week : ℕ := 7

-- Prove that the frog eats 2 flies per day
theorem frog_eats_per_day : (flies_morning + flies_afternoon - escaped_flies) * days_in_week + 4 = 14 → (14 / days_in_week = 2) :=
by
  sorry

end frog_eats_per_day_l1161_116129


namespace total_frogs_in_ponds_l1161_116136

def pondA_frogs := 32
def pondB_frogs := pondA_frogs / 2

theorem total_frogs_in_ponds : pondA_frogs + pondB_frogs = 48 := by
  sorry

end total_frogs_in_ponds_l1161_116136


namespace minimize_distance_l1161_116124

noncomputable def f : ℝ → ℝ := λ x => x ^ 2
noncomputable def g : ℝ → ℝ := λ x => Real.log x
noncomputable def y : ℝ → ℝ := λ x => f x - g x

theorem minimize_distance (t : ℝ) (ht : t = Real.sqrt 2 / 2) :
  ∀ x > 0, y x ≥ y (Real.sqrt 2 / 2) := sorry

end minimize_distance_l1161_116124


namespace tenth_term_of_sequence_l1161_116119

variable (a : ℕ → ℚ) (n : ℕ)

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem tenth_term_of_sequence :
  let a₁ := (5 : ℚ)
  let r := (5 / 3 : ℚ)
  geometric_sequence a₁ r 10 = (9765625 / 19683 : ℚ) :=
by
  sorry

end tenth_term_of_sequence_l1161_116119


namespace sequence_term_expression_l1161_116199

theorem sequence_term_expression (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = 3^n + 1) :
  (a 1 = 4) ∧ (∀ n, n ≥ 2 → a n = 2 * 3^(n-1)) :=
by
  sorry

end sequence_term_expression_l1161_116199


namespace probability_x_lt_2y_in_rectangle_l1161_116183

-- Define the rectangle and the conditions
def in_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 3

-- Define the condition x < 2y
def condition_x_lt_2y (x y : ℝ) : Prop :=
  x < 2 * y

-- Define the probability calculation
theorem probability_x_lt_2y_in_rectangle :
  let rectangle_area := (4:ℝ) * 3
  let triangle_area := (1:ℝ) / 2 * 4 * 2
  let probability := triangle_area / rectangle_area
  probability = 1 / 3 :=
by
  sorry

end probability_x_lt_2y_in_rectangle_l1161_116183


namespace find_x_value_l1161_116182

def solve_for_x (a b x : ℝ) (rectangle_perimeter triangle_height equated_areas : Prop) :=
  rectangle_perimeter -> triangle_height -> equated_areas -> x = 20 / 3

-- Definitions of the conditions
def rectangle_perimeter (a b : ℝ) : Prop := 2 * (a + b) = 60
def triangle_height : Prop := 60 > 0
def equated_areas (a b x : ℝ) : Prop := a * b = 30 * x

theorem find_x_value :
  ∃ a b x : ℝ, solve_for_x a b x (rectangle_perimeter a b) triangle_height (equated_areas a b x) :=
  sorry

end find_x_value_l1161_116182


namespace initial_amount_l1161_116155

theorem initial_amount (M : ℝ) (h1 : M * 2 - 50 > 0) (h2 : (M * 2 - 50) * 2 - 60 > 0) 
(h3 : ((M * 2 - 50) * 2 - 60) * 2 - 70 > 0) 
(h4 : (((M * 2 - 50) * 2 - 60) * 2 - 70) * 2 - 80 = 0) : M = 53.75 := 
sorry

end initial_amount_l1161_116155


namespace solve_equation_l1161_116165

theorem solve_equation : ∃ x : ℝ, 2 * x + 1 = 0 ∧ x = -1 / 2 := by
  sorry

end solve_equation_l1161_116165


namespace tree_height_at_2_years_l1161_116176

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end tree_height_at_2_years_l1161_116176


namespace fraction_simplification_l1161_116131

theorem fraction_simplification : (3 : ℚ) / (2 - (3 / 4)) = 12 / 5 := by
  sorry

end fraction_simplification_l1161_116131


namespace ratio_of_a_over_5_to_b_over_4_l1161_116101

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) : (a/5) / (b/4) = 1 :=
sorry

end ratio_of_a_over_5_to_b_over_4_l1161_116101


namespace average_of_five_numbers_l1161_116173

noncomputable def average_of_two (x1 x2 : ℝ) := (x1 + x2) / 2
noncomputable def average_of_three (x3 x4 x5 : ℝ) := (x3 + x4 + x5) / 3
noncomputable def average_of_five (x1 x2 x3 x4 x5 : ℝ) := (x1 + x2 + x3 + x4 + x5) / 5

theorem average_of_five_numbers (x1 x2 x3 x4 x5 : ℝ)
    (h1 : average_of_two x1 x2 = 12)
    (h2 : average_of_three x3 x4 x5 = 7) :
    average_of_five x1 x2 x3 x4 x5 = 9 := by
  sorry

end average_of_five_numbers_l1161_116173


namespace train_distance_900_l1161_116120

theorem train_distance_900 (x t : ℝ) (H1 : x = 50 * t) (H2 : x - 100 = 40 * t) : 
  x + (x - 100) = 900 :=
by
  sorry

end train_distance_900_l1161_116120


namespace right_triangle_set_l1161_116175

theorem right_triangle_set:
  (1^2 + 2^2 = (Real.sqrt 5)^2) ∧
  ¬ (6^2 + 8^2 = 9^2) ∧
  ¬ ((Real.sqrt 3)^2 + (Real.sqrt 2)^2 = 5^2) ∧
  ¬ ((3^2)^2 + (4^2)^2 = (5^2)^2)  :=
by
  sorry

end right_triangle_set_l1161_116175


namespace intersection_points_range_l1161_116117

theorem intersection_points_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ a = x₁^3 - 3 * x₁ ∧
  a = x₂^3 - 3 * x₂ ∧ a = x₃^3 - 3 * x₃) ↔ (-2 < a ∧ a < 2) :=
sorry

end intersection_points_range_l1161_116117


namespace inequality_solution_l1161_116184

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x ∈ (Set.Ioi 0 ∩ Set.Iic (1/2)) ∪ (Set.Ioi 1.5 ∩ Set.Iio 2)) 
  ↔ ( (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ) := by
  sorry

end inequality_solution_l1161_116184


namespace negation_of_forall_x_geq_1_l1161_116118

theorem negation_of_forall_x_geq_1 :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 1)) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by
  sorry

end negation_of_forall_x_geq_1_l1161_116118


namespace sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l1161_116103

noncomputable def calculate_time (distance1 distance2 speed1 speed2 : ℕ) : ℕ := 
  (distance1 / speed1) + (distance2 / speed2)

noncomputable def total_time_per_lap := calculate_time 200 100 4 6

theorem sofia_total_time_for_5_laps : total_time_per_lap * 5 = 335 := 
  by sorry

def converted_time (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / 60, total_seconds % 60)

theorem sofia_total_time_in_minutes_and_seconds :
  converted_time (total_time_per_lap * 5) = (5, 35) :=
  by sorry

end sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l1161_116103


namespace marbles_problem_l1161_116185

theorem marbles_problem (n : ℕ) :
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0) → 
  n - 10 = 830 :=
sorry

end marbles_problem_l1161_116185


namespace f_sum_lt_zero_l1161_116172

theorem f_sum_lt_zero {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) (h_monotone : ∀ x y, x < y → f y < f x)
  (α β γ : ℝ) (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
sorry

end f_sum_lt_zero_l1161_116172


namespace unit_vector_norm_equal_l1161_116123

variables (a b : EuclideanSpace ℝ (Fin 2)) -- assuming 2D Euclidean space for simplicity

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 1

theorem unit_vector_norm_equal {a b : EuclideanSpace ℝ (Fin 2)}
  (ha : is_unit_vector a) (hb : is_unit_vector b) : ‖a‖ = ‖b‖ :=
by 
  sorry

end unit_vector_norm_equal_l1161_116123


namespace f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l1161_116104

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 2^x + a

theorem f_monotone_on_0_to_2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2 :=
sorry

theorem find_range_a_part2 : (∀ x1 : ℝ, x1 ∈ (Set.Icc (1/2) 1) → 
  ∃ x2 : ℝ, x2 ∈ (Set.Icc 2 3) ∧ f x1 ≥ g x2 a) → a ≤ 1 :=
sorry

theorem find_range_a_part3 : (∃ x : ℝ, x ∈ (Set.Icc 0 2) ∧ f x ≤ g x a) → a ≥ 0 :=
sorry

end f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l1161_116104


namespace ZacharysBusRideLength_l1161_116125

theorem ZacharysBusRideLength (vince_ride zach_ride : ℝ) 
  (h1 : vince_ride = 0.625) 
  (h2 : vince_ride = zach_ride + 0.125) : 
  zach_ride = 0.500 := 
by
  sorry

end ZacharysBusRideLength_l1161_116125


namespace quadratic_to_vertex_form_addition_l1161_116126

theorem quadratic_to_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intro h_eq
  sorry

end quadratic_to_vertex_form_addition_l1161_116126


namespace candy_per_smaller_bag_l1161_116130

-- Define the variables and parameters
def george_candy : ℕ := 648
def friends : ℕ := 3
def total_people : ℕ := friends + 1
def smaller_bags : ℕ := 8

-- Define the theorem
theorem candy_per_smaller_bag : (george_candy / total_people) / smaller_bags = 20 :=
by
  -- Assume the proof steps, not required to actually complete
  sorry

end candy_per_smaller_bag_l1161_116130


namespace adults_riding_bicycles_l1161_116100

theorem adults_riding_bicycles (A : ℕ) (H1 : 15 * 3 + 2 * A = 57) : A = 6 :=
by
  sorry

end adults_riding_bicycles_l1161_116100


namespace minimum_basketballs_sold_l1161_116153

theorem minimum_basketballs_sold :
  ∃ (F B K : ℕ), F + B + K = 180 ∧ 3 * F + 5 * B + 10 * K = 800 ∧ F > B ∧ B > K ∧ K = 2 :=
by
  sorry

end minimum_basketballs_sold_l1161_116153


namespace necessary_but_not_sufficient_l1161_116196

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (a > b - 1) ∧ ¬(a > b - 1 → a > b) :=
sorry

end necessary_but_not_sufficient_l1161_116196


namespace infinite_n_exists_l1161_116187

theorem infinite_n_exists (p : ℕ) (hp : Nat.Prime p) (hp_gt_7 : 7 < p) :
  ∃ᶠ n in at_top, (n ≡ 1 [MOD 2016]) ∧ (p ∣ 2^n + n) :=
sorry

end infinite_n_exists_l1161_116187


namespace simplest_quadratic_radical_problem_l1161_116158

/-- The simplest quadratic radical -/
def simplest_quadratic_radical (r : ℝ) : Prop :=
  ((∀ a b : ℝ, r = a * b → b = 1 ∧ a = r) ∧ (∀ a b : ℝ, r ≠ a / b))

theorem simplest_quadratic_radical_problem :
  (simplest_quadratic_radical (Real.sqrt 6)) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt 8)) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt (1/3))) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt 4)) :=
by
  sorry

end simplest_quadratic_radical_problem_l1161_116158


namespace sampling_correct_l1161_116144

def systematic_sampling (total_students : Nat) (num_selected : Nat) (interval : Nat) (start : Nat) : List Nat :=
  (List.range num_selected).map (λ i => start + i * interval)

theorem sampling_correct :
  systematic_sampling 60 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end sampling_correct_l1161_116144


namespace find_largest_m_l1161_116194

theorem find_largest_m (m : ℤ) : (m^2 - 11 * m + 24 < 0) → m ≤ 7 := sorry

end find_largest_m_l1161_116194


namespace sin_half_angle_correct_l1161_116150

noncomputable def sin_half_angle (theta : ℝ) (h1 : Real.sin theta = 3 / 5) (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) : ℝ :=
  -3 * Real.sqrt 10 / 10

theorem sin_half_angle_correct (theta : ℝ) (h1 : Real.sin theta = 3 / 5) (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  sin_half_angle theta h1 h2 = Real.sin (theta / 2) :=
by
  sorry

end sin_half_angle_correct_l1161_116150


namespace ratio_G_to_C_is_1_1_l1161_116191

variable (R C G : ℕ)

-- Given conditions
def Rover_has_46_spots : Prop := R = 46
def Cisco_has_half_R_minus_5 : Prop := C = R / 2 - 5
def Granger_Cisco_combined_108 : Prop := G + C = 108
def Granger_Cisco_equal : Prop := G = C

-- Theorem stating the final answer to the problem
theorem ratio_G_to_C_is_1_1 (h1 : Rover_has_46_spots R) 
                            (h2 : Cisco_has_half_R_minus_5 C R) 
                            (h3 : Granger_Cisco_combined_108 G C) 
                            (h4 : Granger_Cisco_equal G C) : 
                            G / C = 1 := by
  sorry

end ratio_G_to_C_is_1_1_l1161_116191


namespace largest_percentage_drop_l1161_116135

theorem largest_percentage_drop (jan feb mar apr may jun : ℤ) 
  (h_jan : jan = -10)
  (h_feb : feb = 5)
  (h_mar : mar = -15)
  (h_apr : apr = 10)
  (h_may : may = -30)
  (h_jun : jun = 0) :
  may = -30 ∧ ∀ month, month ≠ may → month ≥ -30 :=
by
  sorry

end largest_percentage_drop_l1161_116135


namespace geometric_series_first_term_l1161_116170

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l1161_116170


namespace geometric_sequence_relation_l1161_116108

variables {a : ℕ → ℝ} {q : ℝ}
variables {m n p : ℕ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def are_in_geometric_sequence (a : ℕ → ℝ) (m n p : ℕ) : Prop :=
  a n ^ 2 = a m * a p

-- Theorem
theorem geometric_sequence_relation (h_geom : is_geometric_sequence a q) (h_order : are_in_geometric_sequence a m n p) (hq_ne_one : q ≠ 1) :
  2 * n = m + p :=
sorry

end geometric_sequence_relation_l1161_116108


namespace remainder_equiv_l1161_116171

theorem remainder_equiv (x : ℤ) (h : ∃ k : ℤ, x = 95 * k + 31) : ∃ m : ℤ, x = 19 * m + 12 := 
sorry

end remainder_equiv_l1161_116171


namespace average_marks_l1161_116148

theorem average_marks (avg1 avg2 : ℝ) (n1 n2 : ℕ) 
  (h_avg1 : avg1 = 40) 
  (h_avg2 : avg2 = 60) 
  (h_n1 : n1 = 25) 
  (h_n2 : n2 = 30) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 50.91 := 
by
  sorry

end average_marks_l1161_116148


namespace find_b_l1161_116116

theorem find_b (g : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∃ x, g x ≠ 0) 
               (h3 : a > 0) (h4 : a ≠ 1) (h5 : ∀ x, (1 / (a ^ x - 1) - 1 / b) * g x = (1 / (a ^ (-x) - 1) - 1 / b) * g (-x)) :
    b = -2 :=
sorry

end find_b_l1161_116116


namespace sum_of_areas_lt_side_length_square_l1161_116192

variable (n : ℕ) (a : ℝ)
variable (S : Fin n → ℝ) (d : Fin n → ℝ)

-- Conditions
axiom areas_le_one : ∀ i, S i ≤ 1
axiom sum_d_le_a : (Finset.univ).sum d ≤ a
axiom areas_less_than_diameters : ∀ i, S i < d i

-- Theorem Statement
theorem sum_of_areas_lt_side_length_square :
  ((Finset.univ : Finset (Fin n)).sum S) < a :=
sorry

end sum_of_areas_lt_side_length_square_l1161_116192


namespace binomial_9_3_l1161_116105

theorem binomial_9_3 : (Nat.choose 9 3) = 84 := by
  sorry

end binomial_9_3_l1161_116105


namespace ratio_of_times_l1161_116114

theorem ratio_of_times (D S : ℝ) (hD : D = 27) (hS : S / 2 = D / 2 + 13.5) :
  D / S = 1 / 2 :=
by
  -- the proof will go here
  sorry

end ratio_of_times_l1161_116114


namespace trig_expression_value_l1161_116141

theorem trig_expression_value (x : ℝ) (h : Real.tan (3 * Real.pi - x) = 2) : 
  (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 :=
by 
  sorry

end trig_expression_value_l1161_116141


namespace cubes_sum_l1161_116140

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 1) (h2 : ab + ac + bc = -4) (h3 : abc = -6) :
  a^3 + b^3 + c^3 = -5 :=
by
  sorry

end cubes_sum_l1161_116140


namespace quadratic_inequality_real_solutions_l1161_116134

-- Definitions and conditions
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main statement
theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∀ x : ℝ, x^2 - 8 * x + c < 0) ↔ (c < 16) :=
by 
  sorry

end quadratic_inequality_real_solutions_l1161_116134
