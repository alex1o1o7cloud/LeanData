import Mathlib

namespace NUMINAMATH_GPT_verify_b_c_sum_ten_l311_31172

theorem verify_b_c_sum_ten (a b c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hb : 1 ≤ b ∧ b < 10) (hc : 1 ≤ c ∧ c < 10) 
    (h_eq : (10 * b + a) * (10 * c + a) = 100 * b * c + 100 * a + a ^ 2) : b + c = 10 :=
by
  sorry

end NUMINAMATH_GPT_verify_b_c_sum_ten_l311_31172


namespace NUMINAMATH_GPT_second_player_wins_l311_31155

noncomputable def is_winning_position (n : ℕ) : Prop :=
  n % 4 = 0

theorem second_player_wins (n : ℕ) (h : n = 100) :
  ∃ f : ℕ → ℕ, (∀ k, 0 < k → k ≤ n → (k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 5) → is_winning_position (n - k)) ∧ is_winning_position n := 
sorry

end NUMINAMATH_GPT_second_player_wins_l311_31155


namespace NUMINAMATH_GPT_proportion_solution_l311_31168

theorem proportion_solution (x: ℕ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end NUMINAMATH_GPT_proportion_solution_l311_31168


namespace NUMINAMATH_GPT_closest_number_to_fraction_l311_31175

theorem closest_number_to_fraction (x : ℝ) : 
  (abs (x - 2000) < abs (x - 1500)) ∧ 
  (abs (x - 2000) < abs (x - 2500)) ∧ 
  (abs (x - 2000) < abs (x - 3000)) ∧ 
  (abs (x - 2000) < abs (x - 3500)) :=
by
  let x := 504 / 0.252
  sorry

end NUMINAMATH_GPT_closest_number_to_fraction_l311_31175


namespace NUMINAMATH_GPT_problem1_problem2_l311_31185

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then (1/2)^x - 2 else (x - 2) * (|x| - 1)

theorem problem1 : f (f (-2)) = 0 := by 
  sorry

theorem problem2 (x : ℝ) (h : f x ≥ 2) : x ≥ 3 ∨ x = 0 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l311_31185


namespace NUMINAMATH_GPT_convex_power_function_l311_31193

theorem convex_power_function (n : ℕ) (h : 0 < n) : 
  (∀ x : ℝ, 0 < x → 0 ≤ (↑n * (↑n - 1) * x ^ (↑n - 2))) ↔ (n = 1 ∨ ∃ k : ℕ, n = 2 * k) :=
by
  sorry

end NUMINAMATH_GPT_convex_power_function_l311_31193


namespace NUMINAMATH_GPT_complex_div_eq_l311_31116

theorem complex_div_eq (z1 z2 : ℂ) (h1 : z1 = 3 - i) (h2 : z2 = 2 + i) :
  z1 / z2 = 1 - i := by
  sorry

end NUMINAMATH_GPT_complex_div_eq_l311_31116


namespace NUMINAMATH_GPT_number_of_new_players_l311_31139

variable (returning_players : ℕ)
variable (groups : ℕ)
variable (players_per_group : ℕ)

theorem number_of_new_players
  (h1 : returning_players = 6)
  (h2 : groups = 9)
  (h3 : players_per_group = 6) :
  (groups * players_per_group - returning_players = 48) := 
sorry

end NUMINAMATH_GPT_number_of_new_players_l311_31139


namespace NUMINAMATH_GPT_nods_per_kilometer_l311_31174

theorem nods_per_kilometer
  (p q r s t u : ℕ)
  (h1 : p * q = q * p)
  (h2 : r * s = s * r)
  (h3 : t * u = u * t) : 
  (1 : ℕ) = qts/pru :=
by
  sorry

end NUMINAMATH_GPT_nods_per_kilometer_l311_31174


namespace NUMINAMATH_GPT_probability_not_e_after_n_spins_l311_31118

theorem probability_not_e_after_n_spins
    (S : Type)
    (e b c d : S)
    (p_e : ℝ)
    (p_b : ℝ)
    (p_c : ℝ)
    (p_d : ℝ) :
    (p_e = 0.25) →
    (p_b = 0.25) →
    (p_c = 0.25) →
    (p_d = 0.25) →
    (1 - p_e)^2 = 0.5625 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_e_after_n_spins_l311_31118


namespace NUMINAMATH_GPT_geometric_concepts_cases_l311_31135

theorem geometric_concepts_cases :
  (∃ x y, x = "rectangle" ∧ y = "rhombus") ∧ 
  (∃ x y z, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "acute_triangle") ∧ 
  (∃ x y z u, x = "parallelogram" ∧ y = "rectangle" ∧ z = "square" ∧ u = "acute_angled_rhombus") ∧ 
  (∃ x y z u t, x = "polygon" ∧ y = "triangle" ∧ z = "isosceles_triangle" ∧ u = "equilateral_triangle" ∧ t = "right_triangle") ∧ 
  (∃ x y z u, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "obtuse_triangle" ∧ u = "scalene_triangle") :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_concepts_cases_l311_31135


namespace NUMINAMATH_GPT_Chang_solution_A_amount_l311_31109

def solution_alcohol_content (A B : ℝ) (x : ℝ) : ℝ :=
  0.16 * x + 0.10 * (x + 500)

theorem Chang_solution_A_amount (x : ℝ) :
  solution_alcohol_content 0.16 0.10 x = 76 → x = 100 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Chang_solution_A_amount_l311_31109


namespace NUMINAMATH_GPT_evaluate_expression_l311_31171

variable (x y z : ℤ)

theorem evaluate_expression :
  x = 3 → y = 2 → z = 4 → 3 * x - 4 * y + 5 * z = 21 :=
by
  intros hx hy hz
  rw [hx, hy, hz]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l311_31171


namespace NUMINAMATH_GPT_domain_ln_x_minus_1_l311_31167

def domain_of_log_function (x : ℝ) : Prop := x > 1

theorem domain_ln_x_minus_1 (x : ℝ) : domain_of_log_function x ↔ x > 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_ln_x_minus_1_l311_31167


namespace NUMINAMATH_GPT_gum_ratio_correct_l311_31179

variable (y : ℝ)
variable (cherry_pieces : ℝ := 30)
variable (grape_pieces : ℝ := 40)
variable (pieces_per_pack : ℝ := y)

theorem gum_ratio_correct:
  ((cherry_pieces - 2 * pieces_per_pack) / grape_pieces = cherry_pieces / (grape_pieces + 4 * pieces_per_pack)) ↔ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_gum_ratio_correct_l311_31179


namespace NUMINAMATH_GPT_dirocks_rectangular_fence_count_l311_31130

/-- Dirock's backyard problem -/
def grid_side : ℕ := 32

def rock_placement (i j : ℕ) : Prop := (i % 3 = 0) ∧ (j % 3 = 0)

noncomputable def dirocks_rectangular_fence_ways : ℕ :=
  sorry

theorem dirocks_rectangular_fence_count : dirocks_rectangular_fence_ways = 1920 :=
sorry

end NUMINAMATH_GPT_dirocks_rectangular_fence_count_l311_31130


namespace NUMINAMATH_GPT_solve_for_x_l311_31166

theorem solve_for_x (x : ℝ) (h : (15 - 2 + (x / 1)) / 2 * 8 = 77) : x = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l311_31166


namespace NUMINAMATH_GPT_selection_ways_l311_31154

-- Step a): Define the conditions
def number_of_boys := 26
def number_of_girls := 24

-- Step c): State the problem
theorem selection_ways :
  number_of_boys + number_of_girls = 50 := by
  sorry

end NUMINAMATH_GPT_selection_ways_l311_31154


namespace NUMINAMATH_GPT_find_point_B_coordinates_l311_31150

theorem find_point_B_coordinates (a : ℝ) : 
  (∀ (x y : ℝ), x^2 - 4*x + y^2 = 0 → (x - a)^2 + y^2 = 4 * ((x - 1)^2 + y^2)) →
  a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_point_B_coordinates_l311_31150


namespace NUMINAMATH_GPT_boy_actual_height_is_236_l311_31134

def actual_height (n : ℕ) (incorrect_avg correct_avg wrong_height : ℕ) : ℕ :=
  let incorrect_total := n * incorrect_avg
  let correct_total := n * correct_avg
  let diff := incorrect_total - correct_total
  wrong_height + diff

theorem boy_actual_height_is_236 :
  ∀ (n incorrect_avg correct_avg wrong_height actual_height : ℕ),
  n = 35 → 
  incorrect_avg = 183 → 
  correct_avg = 181 → 
  wrong_height = 166 → 
  actual_height = wrong_height + (n * incorrect_avg - n * correct_avg) →
  actual_height = 236 :=
by
  intros n incorrect_avg correct_avg wrong_height actual_height hn hic hg hw ha
  rw [hn, hic, hg, hw] at ha
  -- At this point, we would normally proceed to prove the statement.
  -- However, as per the requirements, we just include "sorry" to skip the proof.
  sorry

end NUMINAMATH_GPT_boy_actual_height_is_236_l311_31134


namespace NUMINAMATH_GPT_mid_point_between_fractions_l311_31182

theorem mid_point_between_fractions : (1 / 12 + 1 / 20) / 2 = 1 / 15 := by
  sorry

end NUMINAMATH_GPT_mid_point_between_fractions_l311_31182


namespace NUMINAMATH_GPT_vector_addition_l311_31164

def a : ℝ × ℝ := (5, -3)
def b : ℝ × ℝ := (-6, 4)

theorem vector_addition : a + b = (-1, 1) := by
  rw [a, b]
  sorry

end NUMINAMATH_GPT_vector_addition_l311_31164


namespace NUMINAMATH_GPT_remainder_of_sum_of_consecutive_days_l311_31114

theorem remainder_of_sum_of_consecutive_days :
  (100045 + 100046 + 100047 + 100048 + 100049 + 100050 + 100051 + 100052) % 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_consecutive_days_l311_31114


namespace NUMINAMATH_GPT_basketball_team_starters_l311_31173

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_team_starters :
  choose 4 2 * choose 14 4 = 6006 := by
  sorry

end NUMINAMATH_GPT_basketball_team_starters_l311_31173


namespace NUMINAMATH_GPT_more_student_tickets_l311_31184

-- Definitions of given conditions
def student_ticket_price : ℕ := 6
def nonstudent_ticket_price : ℕ := 9
def total_sales : ℕ := 10500
def total_tickets : ℕ := 1700

-- Definitions of the variables for student and nonstudent tickets
variables (S N : ℕ)

-- Lean statement of the problem
theorem more_student_tickets (h1 : student_ticket_price * S + nonstudent_ticket_price * N = total_sales)
                            (h2 : S + N = total_tickets) : S - N = 1500 :=
by
  sorry

end NUMINAMATH_GPT_more_student_tickets_l311_31184


namespace NUMINAMATH_GPT_question_l311_31122

section

variable (x : ℝ)
variable (p q : Prop)

-- Define proposition p: ∀ x in [0,1], e^x ≥ 1
def Proposition_p : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → Real.exp x ≥ 1

-- Define proposition q: ∃ x in ℝ such that x^2 + x + 1 < 0
def Proposition_q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- The problem to prove: p ∨ q
theorem question (p q : Prop) (hp : Proposition_p) (hq : ¬ Proposition_q) : p ∨ q := by
  sorry

end

end NUMINAMATH_GPT_question_l311_31122


namespace NUMINAMATH_GPT_trapezoid_median_l311_31131

theorem trapezoid_median {BC AD : ℝ} (h AC CD : ℝ) (h_nonneg : h = 2) (AC_eq_CD : AC = 4) (BC_eq_0 : BC = 0) 
: (AD = 4 * Real.sqrt 3) → (median = 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_trapezoid_median_l311_31131


namespace NUMINAMATH_GPT_binomial_expansion_value_l311_31101

theorem binomial_expansion_value : 
  105^3 - 3 * 105^2 + 3 * 105 - 1 = 1124864 := by
  sorry

end NUMINAMATH_GPT_binomial_expansion_value_l311_31101


namespace NUMINAMATH_GPT_max_marks_l311_31140

theorem max_marks (M : ℝ) (pass_percent : ℝ) (obtained_marks : ℝ) (failed_by : ℝ) (pass_marks : ℝ) 
  (h1 : pass_percent = 0.40) 
  (h2 : obtained_marks = 150) 
  (h3 : failed_by = 50) 
  (h4 : pass_marks = 200) 
  (h5 : pass_marks = obtained_marks + failed_by) 
  : M = 500 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_max_marks_l311_31140


namespace NUMINAMATH_GPT_xyz_value_l311_31178

theorem xyz_value
  (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + y * z + z * x) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 14 / 3 :=
  sorry

end NUMINAMATH_GPT_xyz_value_l311_31178


namespace NUMINAMATH_GPT_milk_total_correct_l311_31104

def chocolate_milk : Nat := 2
def strawberry_milk : Nat := 15
def regular_milk : Nat := 3
def total_milk : Nat := chocolate_milk + strawberry_milk + regular_milk

theorem milk_total_correct : total_milk = 20 := by
  sorry

end NUMINAMATH_GPT_milk_total_correct_l311_31104


namespace NUMINAMATH_GPT_segment_ratios_correct_l311_31108

noncomputable def compute_segment_ratios : (ℕ × ℕ) :=
  let ratio := 20 / 340;
  let gcd := Nat.gcd 1 17;
  if (ratio = 1 / 17) ∧ (gcd = 1) then (1, 17) else (0, 0) 

theorem segment_ratios_correct : 
  compute_segment_ratios = (1, 17) := 
by
  sorry

end NUMINAMATH_GPT_segment_ratios_correct_l311_31108


namespace NUMINAMATH_GPT_num_distinct_intersections_l311_31129

def linear_eq1 (x y : ℝ) := x + 2 * y - 10
def linear_eq2 (x y : ℝ) := x - 4 * y + 8
def linear_eq3 (x y : ℝ) := 2 * x - y - 1
def linear_eq4 (x y : ℝ) := 5 * x + 3 * y - 15

theorem num_distinct_intersections (n : ℕ) :
  (∀ x y : ℝ, linear_eq1 x y = 0 ∨ linear_eq2 x y = 0) ∧ 
  (∀ x y : ℝ, linear_eq3 x y = 0 ∨ linear_eq4 x y = 0) →
  n = 3 :=
  sorry

end NUMINAMATH_GPT_num_distinct_intersections_l311_31129


namespace NUMINAMATH_GPT_side_lengths_of_triangle_l311_31192

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

theorem side_lengths_of_triangle (m : ℝ) (a b c : ℝ) 
  (h1 : f m a > 0) 
  (h2 : f m b > 0) 
  (h3 : f m c > 0) 
  (h4 : f m a + f m b > f m c)
  (h5 : f m a + f m c > f m b)
  (h6 : f m b + f m c > f m a) :
  m ∈ Set.Ioo (7/5 : ℝ) 5 :=
sorry

end NUMINAMATH_GPT_side_lengths_of_triangle_l311_31192


namespace NUMINAMATH_GPT_range_of_x_l311_31144

theorem range_of_x (θ : ℝ) (h0 : 0 < θ) (h1 : θ < Real.pi / 2) (h2 : ∀ θ, (0 < θ) → (θ < Real.pi / 2) → (1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2 ≥ abs (2 * x - 1))) :
  -4 ≤ x ∧ x ≤ 5 := sorry

end NUMINAMATH_GPT_range_of_x_l311_31144


namespace NUMINAMATH_GPT_four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l311_31183

-- Number of four-digit numbers greater than 3999 such that the product of the middle two digits > 12 is 4260
theorem four_digit_numbers_greater_3999_with_middle_product_exceeding_12
  {d1 d2 d3 d4 : ℕ}
  (h1 : 4 ≤ d1 ∧ d1 ≤ 9)
  (h2 : 0 ≤ d4 ∧ d4 ≤ 9)
  (h3 : 1 ≤ d2 ∧ d2 ≤ 9)
  (h4 : 1 ≤ d3 ∧ d3 ≤ 9)
  (h5 : d2 * d3 > 12) :
  (6 * 71 * 10 = 4260) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l311_31183


namespace NUMINAMATH_GPT_problem_I_problem_II_l311_31136

-- Problem (I)
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) : 
  (f (x + 8) ≥ 10 - f x) ↔ (x ≤ -10 ∨ x ≥ 0) :=
sorry

-- Problem (II)
theorem problem_II (x y : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) 
(h_abs_x : |x| > 1) (h_abs_y : |y| < 1) :
  f y < |x| * f (y / x^2) :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_l311_31136


namespace NUMINAMATH_GPT_rbcmul_div7_div89_l311_31119

theorem rbcmul_div7_div89 {r b c : ℕ} (h : (523000 + 100 * r + 10 * b + c) % 7 = 0 ∧ (523000 + 100 * r + 10 * b + c) % 89 = 0) :
  r * b * c = 36 :=
by
  sorry

end NUMINAMATH_GPT_rbcmul_div7_div89_l311_31119


namespace NUMINAMATH_GPT_ethanol_total_amount_l311_31180

-- Definitions based on Conditions
def total_tank_capacity : ℕ := 214
def fuel_A_volume : ℕ := 106
def fuel_B_volume : ℕ := total_tank_capacity - fuel_A_volume
def ethanol_in_fuel_A : ℚ := 0.12
def ethanol_in_fuel_B : ℚ := 0.16

-- Theorem Statement
theorem ethanol_total_amount :
  (fuel_A_volume * ethanol_in_fuel_A + fuel_B_volume * ethanol_in_fuel_B) = 30 := 
sorry

end NUMINAMATH_GPT_ethanol_total_amount_l311_31180


namespace NUMINAMATH_GPT_number_of_distinct_configurations_l311_31146

-- Define the conditions
def numConfigurations (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 else n + 1

-- Theorem statement
theorem number_of_distinct_configurations (n : ℕ) : 
  numConfigurations n = if n % 2 = 1 then 2 else n + 1 :=
by
  sorry -- Proof intentionally left out

end NUMINAMATH_GPT_number_of_distinct_configurations_l311_31146


namespace NUMINAMATH_GPT_train_speed_is_correct_l311_31191

-- Definitions for conditions
def train_length : ℝ := 150  -- length of the train in meters
def time_to_cross_pole : ℝ := 3  -- time to cross the pole in seconds

-- Proof statement
theorem train_speed_is_correct : (train_length / time_to_cross_pole) = 50 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_correct_l311_31191


namespace NUMINAMATH_GPT_opposite_of_one_l311_31170

theorem opposite_of_one (a : ℤ) (h : a = -1) : a = -1 := 
by 
  exact h

end NUMINAMATH_GPT_opposite_of_one_l311_31170


namespace NUMINAMATH_GPT_grandson_age_l311_31132

variable (G F : ℕ)

-- Define the conditions given in the problem
def condition1 := F = 6 * G
def condition2 := (F + 4) + (G + 4) = 78

-- The theorem to prove
theorem grandson_age : condition1 G F → condition2 G F → G = 10 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_grandson_age_l311_31132


namespace NUMINAMATH_GPT_total_eyes_l311_31169

def boys := 23
def girls := 18
def cats := 10
def spiders := 5

def boy_eyes := 2
def girl_eyes := 2
def cat_eyes := 2
def spider_eyes := 8

theorem total_eyes : (boys * boy_eyes) + (girls * girl_eyes) + (cats * cat_eyes) + (spiders * spider_eyes) = 142 := by
  sorry

end NUMINAMATH_GPT_total_eyes_l311_31169


namespace NUMINAMATH_GPT_lion_to_leopard_ratio_l311_31194

variable (L P E : ℕ)

axiom lion_count : L = 200
axiom total_population : L + P + E = 450
axiom elephants_relation : E = (1 / 2 : ℚ) * (L + P)

theorem lion_to_leopard_ratio : L / P = 2 :=
by
  sorry

end NUMINAMATH_GPT_lion_to_leopard_ratio_l311_31194


namespace NUMINAMATH_GPT_razorback_shop_tshirts_l311_31198

theorem razorback_shop_tshirts (T : ℕ) (h : 215 * T = 4300) : T = 20 :=
by sorry

end NUMINAMATH_GPT_razorback_shop_tshirts_l311_31198


namespace NUMINAMATH_GPT_solution_set_of_inequality_l311_31142

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x - 2 else if x < 0 then -(x - 2) else 0

theorem solution_set_of_inequality :
  {x : ℝ | f x < 1 / 2} =
  {x : ℝ | (0 ≤ x ∧ x < 5 / 2) ∨ x < -3 / 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l311_31142


namespace NUMINAMATH_GPT_songs_listened_l311_31138

theorem songs_listened (x y : ℕ) 
  (h1 : y = 9) 
  (h2 : y = 2 * (Nat.sqrt x) - 5) 
  : y + x = 58 := 
  sorry

end NUMINAMATH_GPT_songs_listened_l311_31138


namespace NUMINAMATH_GPT_average_age_l311_31107

theorem average_age (avg_age_students : ℝ) (num_students : ℕ) (avg_age_teachers : ℝ) (num_teachers : ℕ) :
  avg_age_students = 13 → 
  num_students = 40 → 
  avg_age_teachers = 42 → 
  num_teachers = 60 → 
  (num_students * avg_age_students + num_teachers * avg_age_teachers) / (num_students + num_teachers) = 30.4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_age_l311_31107


namespace NUMINAMATH_GPT_complementary_three_card_sets_l311_31153

-- Definitions for the problem conditions
inductive Shape | circle | square | triangle | star
inductive Color | red | blue | green | yellow
inductive Shade | light | medium | dark | very_dark

-- Definition of a Card as a combination of shape, color, shade
structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)

-- Definition of a set being complementary
def is_complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape = c2.shape ∧ c2.shape = c3.shape) ∨ (c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape)) ∧
  ((c1.color = c2.color ∧ c2.color = c3.color) ∨ (c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color)) ∧
  ((c1.shade = c2.shade ∧ c2.shade = c3.shade) ∨ (c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade))

-- Definition of the problem statement
def complementary_three_card_sets_count : Nat :=
  360

-- The theorem to be proved
theorem complementary_three_card_sets : ∃ (n : Nat), n = complementary_three_card_sets_count :=
  by
    use 360
    sorry

end NUMINAMATH_GPT_complementary_three_card_sets_l311_31153


namespace NUMINAMATH_GPT_relationship_between_M_and_N_l311_31195
   
   variable (x : ℝ)
   def M := 2*x^2 - 12*x + 15
   def N := x^2 - 8*x + 11
   
   theorem relationship_between_M_and_N : M x ≥ N x :=
   by
     sorry
   
end NUMINAMATH_GPT_relationship_between_M_and_N_l311_31195


namespace NUMINAMATH_GPT_product_of_numbers_l311_31128

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
sorry

end NUMINAMATH_GPT_product_of_numbers_l311_31128


namespace NUMINAMATH_GPT_size_of_angle_B_length_of_side_b_and_area_l311_31196

-- Given problem conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h1 : a < b) (h2 : b < c) (h3 : a / Real.sin A = 2 * b / Real.sqrt 3)

-- Prove that B = π / 3
theorem size_of_angle_B : B = Real.pi / 3 := 
sorry

-- Additional conditions for part (2)
variables (h4 : a = 2) (h5 : c = 3) (h6 : Real.cos B = 1 / 2)

-- Prove b = √7 and the area of triangle ABC
theorem length_of_side_b_and_area :
  b = Real.sqrt 7 ∧ 1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_size_of_angle_B_length_of_side_b_and_area_l311_31196


namespace NUMINAMATH_GPT_equivalent_annual_rate_approx_l311_31190

noncomputable def annual_rate : ℝ := 0.045
noncomputable def days_in_year : ℝ := 365
noncomputable def daily_rate : ℝ := annual_rate / days_in_year
noncomputable def equivalent_annual_rate : ℝ := (1 + daily_rate) ^ days_in_year - 1

theorem equivalent_annual_rate_approx :
  abs (equivalent_annual_rate - 0.0459) < 0.0001 :=
by sorry

end NUMINAMATH_GPT_equivalent_annual_rate_approx_l311_31190


namespace NUMINAMATH_GPT_part_one_part_two_l311_31177

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem part_one :
  ∀ x m : ℕ, f x ≤ -m^2 + 6 * m → 1 ≤ m ∧ m ≤ 5 := 
by
  sorry

theorem part_two (a b c : ℝ) (h : 3 * a + 4 * b + 5 * c = 1) :
  (a^2 + b^2 + c^2) ≥ (1 / 50) :=
by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l311_31177


namespace NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l311_31163

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h : |x - 1/2| + (2*y + 1)^2 = 0) : 
  x^2 + y^2 = 1/2 :=
sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l311_31163


namespace NUMINAMATH_GPT_smallest_integer_mod_conditions_l311_31157

theorem smallest_integer_mod_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 3) ∧ (x % 3 = 2) ∧ (∀ y : ℕ, (y % 4 = 3) ∧ (y % 3 = 2) → x ≤ y) ∧ x = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_mod_conditions_l311_31157


namespace NUMINAMATH_GPT_total_cups_sold_l311_31112

theorem total_cups_sold (plastic_cups : ℕ) (ceramic_cups : ℕ) (total_sold : ℕ) :
  plastic_cups = 284 ∧ ceramic_cups = 284 → total_sold = 568 :=
by
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_total_cups_sold_l311_31112


namespace NUMINAMATH_GPT_simplify_expr_l311_31147

theorem simplify_expr : 
  (576:ℝ)^(1/4) * (216:ℝ)^(1/2) = 72 := 
by 
  have h1 : 576 = (2^4 * 36 : ℝ) := by norm_num
  have h2 : 36 = (6^2 : ℝ) := by norm_num
  have h3 : 216 = (6^3 : ℝ) := by norm_num
  sorry

end NUMINAMATH_GPT_simplify_expr_l311_31147


namespace NUMINAMATH_GPT_base_conversion_l311_31103

theorem base_conversion (k : ℕ) (h : 26 = 3*k + 2) : k = 8 := 
by 
  sorry

end NUMINAMATH_GPT_base_conversion_l311_31103


namespace NUMINAMATH_GPT_cnc_processing_time_l311_31102

theorem cnc_processing_time :
  (∃ (hours: ℕ), 3 * (960 / hours) = 960 / 3) → 1 * (400 / 5) = 400 / 1 :=
by
  sorry

end NUMINAMATH_GPT_cnc_processing_time_l311_31102


namespace NUMINAMATH_GPT_percent_of_a_is_4b_l311_31162

variables (a b : ℝ)
theorem percent_of_a_is_4b (h : a = 2 * b) : 4 * b / a = 2 :=
by 
  sorry

end NUMINAMATH_GPT_percent_of_a_is_4b_l311_31162


namespace NUMINAMATH_GPT_new_average_score_l311_31145

theorem new_average_score (average_initial : ℝ) (total_practices : ℕ) (highest_score lowest_score : ℝ) :
  average_initial = 87 → 
  total_practices = 10 → 
  highest_score = 95 → 
  lowest_score = 55 → 
  ((average_initial * total_practices - highest_score - lowest_score) / (total_practices - 2)) = 90 :=
by
  intros h_avg h_total h_high h_low
  sorry

end NUMINAMATH_GPT_new_average_score_l311_31145


namespace NUMINAMATH_GPT_hard_candy_food_colouring_l311_31111

theorem hard_candy_food_colouring :
  (∀ lollipop_colour hard_candy_count total_food_colouring lollipop_count hard_candy_food_total_per_lollipop,
    lollipop_colour = 5 →
    lollipop_count = 100 →
    hard_candy_count = 5 →
    total_food_colouring = 600 →
    hard_candy_food_total_per_lollipop = lollipop_colour * lollipop_count →
    total_food_colouring - hard_candy_food_total_per_lollipop = hard_candy_count * hard_candy_food_total_per_candy →
    hard_candy_food_total_per_candy = 20) :=
by
  sorry

end NUMINAMATH_GPT_hard_candy_food_colouring_l311_31111


namespace NUMINAMATH_GPT_taxi_fare_l311_31176

theorem taxi_fare (x : ℝ) (h : x > 3) : 
  let starting_price := 6
  let additional_fare_per_km := 1.4
  let fare := starting_price + additional_fare_per_km * (x - 3)
  fare = 1.4 * x + 1.8 :=
by
  sorry

end NUMINAMATH_GPT_taxi_fare_l311_31176


namespace NUMINAMATH_GPT_b_divisible_by_a_l311_31188

theorem b_divisible_by_a (a b c : ℕ) (ha : a > 1) (hbc : b > c ∧ c > 1) (hdiv : (abc + 1) % (ab - b + 1) = 0) : a ∣ b :=
  sorry

end NUMINAMATH_GPT_b_divisible_by_a_l311_31188


namespace NUMINAMATH_GPT_dog_speed_correct_l311_31161

-- Definitions of the conditions
def football_field_length_yards : ℕ := 200
def total_football_fields : ℕ := 6
def yards_to_feet_conversion : ℕ := 3
def time_to_fetch_minutes : ℕ := 9

-- The goal is to find the dog's speed in feet per minute
def dog_speed_feet_per_minute : ℕ :=
  (total_football_fields * football_field_length_yards * yards_to_feet_conversion) / time_to_fetch_minutes

-- Statement for the proof
theorem dog_speed_correct : dog_speed_feet_per_minute = 400 := by
  sorry

end NUMINAMATH_GPT_dog_speed_correct_l311_31161


namespace NUMINAMATH_GPT_smallest_sum_l311_31133

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end NUMINAMATH_GPT_smallest_sum_l311_31133


namespace NUMINAMATH_GPT_heart_then_club_probability_l311_31110

theorem heart_then_club_probability :
  (13 / 52) * (13 / 51) = 13 / 204 := by
  sorry

end NUMINAMATH_GPT_heart_then_club_probability_l311_31110


namespace NUMINAMATH_GPT_races_needed_to_declare_winner_l311_31115

noncomputable def total_sprinters : ℕ := 275
noncomputable def sprinters_per_race : ℕ := 7
noncomputable def sprinters_advance : ℕ := 2
noncomputable def sprinters_eliminated : ℕ := 5

theorem races_needed_to_declare_winner :
  (total_sprinters - 1 + sprinters_eliminated) / sprinters_eliminated = 59 :=
by
  sorry

end NUMINAMATH_GPT_races_needed_to_declare_winner_l311_31115


namespace NUMINAMATH_GPT_probability_relationship_l311_31187

def total_outcomes : ℕ := 36

def P1 : ℚ := 1 / total_outcomes
def P2 : ℚ := 2 / total_outcomes
def P3 : ℚ := 3 / total_outcomes

theorem probability_relationship :
  P1 < P2 ∧ P2 < P3 :=
by
  sorry

end NUMINAMATH_GPT_probability_relationship_l311_31187


namespace NUMINAMATH_GPT_cottage_cost_per_hour_l311_31158

-- Define the conditions
def jack_payment : ℝ := 20
def jill_payment : ℝ := 20
def total_payment : ℝ := jack_payment + jill_payment
def rental_duration : ℝ := 8

-- Define the theorem to be proved
theorem cottage_cost_per_hour : (total_payment / rental_duration) = 5 := by
  sorry

end NUMINAMATH_GPT_cottage_cost_per_hour_l311_31158


namespace NUMINAMATH_GPT_solution_set_of_inequality_l311_31156

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x | 3 * a < x ∧ x < -a} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l311_31156


namespace NUMINAMATH_GPT_max_value_of_f_l311_31106

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + Real.sin (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, f x = 1 + Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_max_value_of_f_l311_31106


namespace NUMINAMATH_GPT_smallest_n_divisibility_problem_l311_31123

theorem smallest_n_divisibility_problem :
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ n + 2 → n^3 - n ≠ 0 → (n^3 - n) % k = 0) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → k ∣ n^3 - n) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → ¬ k ∣ n^3 - n) ∧
    (∀ (m : ℕ), m > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ m + 2 → m^3 - m ≠ 0 → (m^3 - m) % k = 0) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → k ∣ m^3 - m) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → ¬ k ∣ m^3 - m) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_n_divisibility_problem_l311_31123


namespace NUMINAMATH_GPT_invalid_transformation_of_equation_l311_31125

theorem invalid_transformation_of_equation (x y m : ℝ) (h : x = y) :
  (m = 0 → (x = y → x / m = y / m)) = false :=
by
  sorry

end NUMINAMATH_GPT_invalid_transformation_of_equation_l311_31125


namespace NUMINAMATH_GPT_sequence_value_2023_l311_31181

theorem sequence_value_2023 (a : ℕ → ℕ) (h₁ : a 1 = 3)
  (h₂ : ∀ m n : ℕ, a (m + n) = a m + a n) : a 2023 = 6069 := by
  sorry

end NUMINAMATH_GPT_sequence_value_2023_l311_31181


namespace NUMINAMATH_GPT_problem_solution_l311_31141

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem problem_solution (x1 x2 : ℝ) 
  (hx1 : x1 ∈ Set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (hx2 : x2 ∈ Set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (h : f x1 < f x2) : x1^2 > x2^2 := 
sorry

end NUMINAMATH_GPT_problem_solution_l311_31141


namespace NUMINAMATH_GPT_find_chemistry_marks_l311_31199

theorem find_chemistry_marks
  (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (chemistry_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 → biology_marks = 81 → average_marks = 85 →
  chemistry_marks = 425 - (english_marks + math_marks + physics_marks + biology_marks) →
  chemistry_marks = 87 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  have total_marks := 425 - (86 + 89 + 82 + 81)
  norm_num at total_marks
  exact h6

end NUMINAMATH_GPT_find_chemistry_marks_l311_31199


namespace NUMINAMATH_GPT_expand_expression_l311_31127

theorem expand_expression (x : ℝ) : (x + 3) * (2 * x ^ 2 - x + 4) = 2 * x ^ 3 + 5 * x ^ 2 + x + 12 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l311_31127


namespace NUMINAMATH_GPT_exists_unique_circle_l311_31160

structure Circle := (center : ℝ × ℝ) (radius : ℝ)

def diametrically_opposite_points (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (cx, cy) := C.center
  let (px, py) := P
  (px - cx) ^ 2 + (py - cy) ^ 2 = (C.radius ^ 2)

def intersects_at_diametrically_opposite_points (K A : Circle) : Prop :=
  ∃ P₁ P₂ : ℝ × ℝ, diametrically_opposite_points A P₁ ∧ diametrically_opposite_points A P₂ ∧
  P₁ ≠ P₂ ∧ diametrically_opposite_points K P₁ ∧ diametrically_opposite_points K P₂

theorem exists_unique_circle (A B C : Circle) :
  ∃! K : Circle, intersects_at_diametrically_opposite_points K A ∧
  intersects_at_diametrically_opposite_points K B ∧
  intersects_at_diametrically_opposite_points K C := sorry

end NUMINAMATH_GPT_exists_unique_circle_l311_31160


namespace NUMINAMATH_GPT_intersection_eq_l311_31121

noncomputable def A : Set ℝ := { x | x < 2 }
noncomputable def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l311_31121


namespace NUMINAMATH_GPT_find_a_n_l311_31100

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : ∀ n, a n > 0)
  (h₂ : ∀ n, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_n_l311_31100


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l311_31149

-- Define the sets A and B
def A (x : ℝ) : Prop := x > 2
def B (x : ℝ) : Prop := x > 1

-- Prove that B (necessary condition x > 1) does not suffice for A (x > 2)
theorem necessary_but_not_sufficient (x : ℝ) (h : B x) : A x ∨ ¬A x :=
by
  -- B x is a necessary condition for A x
  have h1 : x > 1 := h
  -- A x is not necessarily implied by B x
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l311_31149


namespace NUMINAMATH_GPT_solve_problem_l311_31151

-- Declare the variables n and m
variables (n m : ℤ)

-- State the theorem with given conditions and prove that 2n + m = 36
theorem solve_problem
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 :=
sorry

end NUMINAMATH_GPT_solve_problem_l311_31151


namespace NUMINAMATH_GPT_value_of_m_l311_31165

theorem value_of_m
  (m : ℝ)
  (a : ℝ × ℝ := (-1, 3))
  (b : ℝ × ℝ := (m, m - 2))
  (collinear : a.1 * b.2 = a.2 * b.1) :
  m = 1 / 2 :=
sorry

end NUMINAMATH_GPT_value_of_m_l311_31165


namespace NUMINAMATH_GPT_sum_remainder_of_consecutive_odds_l311_31105

theorem sum_remainder_of_consecutive_odds :
  (11075 + 11077 + 11079 + 11081 + 11083 + 11085 + 11087) % 14 = 7 :=
by
  -- Adding the proof here
  sorry

end NUMINAMATH_GPT_sum_remainder_of_consecutive_odds_l311_31105


namespace NUMINAMATH_GPT_shorten_ellipse_parametric_form_l311_31124

theorem shorten_ellipse_parametric_form :
  ∀ (θ : ℝ), 
  ∃ (x' y' : ℝ),
    x' = 4 * Real.cos θ ∧ y' = 2 * Real.sin θ ∧
    (∃ (x y : ℝ),
      x' = 2 * x ∧ y' = y ∧
      x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ) :=
by
  sorry

end NUMINAMATH_GPT_shorten_ellipse_parametric_form_l311_31124


namespace NUMINAMATH_GPT_largest_side_of_enclosure_l311_31126

theorem largest_side_of_enclosure (l w : ℕ) (h1 : 2 * l + 2 * w = 180) (h2 : l * w = 1800) : max l w = 60 := 
by 
  sorry

end NUMINAMATH_GPT_largest_side_of_enclosure_l311_31126


namespace NUMINAMATH_GPT_octagon_perimeter_l311_31113

def side_length_meters : ℝ := 2.3
def number_of_sides : ℕ := 8
def meter_to_cm (meters : ℝ) : ℝ := meters * 100

def perimeter_cm (side_length_meters : ℝ) (number_of_sides : ℕ) : ℝ :=
  meter_to_cm side_length_meters * number_of_sides

theorem octagon_perimeter :
  perimeter_cm side_length_meters number_of_sides = 1840 :=
by
  sorry

end NUMINAMATH_GPT_octagon_perimeter_l311_31113


namespace NUMINAMATH_GPT_maximum_value_sum_l311_31189

theorem maximum_value_sum (a b c d : ℕ) (h1 : a + c = 1000) (h2 : b + d = 500) :
  ∃ a b c d, a + c = 1000 ∧ b + d = 500 ∧ (a = 1 ∧ c = 999 ∧ b = 499 ∧ d = 1) ∧ 
  ((a : ℝ) / b + (c : ℝ) / d = (1 / 499) + 999) := 
  sorry

end NUMINAMATH_GPT_maximum_value_sum_l311_31189


namespace NUMINAMATH_GPT_fraction_addition_l311_31148

theorem fraction_addition : (3 / 4 : ℚ) + (5 / 6) = 19 / 12 :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_l311_31148


namespace NUMINAMATH_GPT_eggs_per_chicken_per_day_l311_31120

theorem eggs_per_chicken_per_day (E c d : ℕ) (hE : E = 36) (hc : c = 4) (hd : d = 3) :
  (E / d) / c = 3 := by
  sorry

end NUMINAMATH_GPT_eggs_per_chicken_per_day_l311_31120


namespace NUMINAMATH_GPT_factorize_poly_l311_31117

-- Statement of the problem
theorem factorize_poly (x : ℝ) : x^2 - 3 * x = x * (x - 3) :=
sorry

end NUMINAMATH_GPT_factorize_poly_l311_31117


namespace NUMINAMATH_GPT_find_number_satisfy_equation_l311_31143

theorem find_number_satisfy_equation (x : ℝ) :
  9 - x / 7 * 5 + 10 = 13.285714285714286 ↔ x = -20 := sorry

end NUMINAMATH_GPT_find_number_satisfy_equation_l311_31143


namespace NUMINAMATH_GPT_quadratic_polynomial_AT_BT_l311_31152

theorem quadratic_polynomial_AT_BT (p s : ℝ) :
  ∃ (AT BT : ℝ), (AT + BT = p + 3) ∧ (AT * BT = s^2) ∧ (∀ (x : ℝ), (x^2 - (p+3) * x + s^2) = (x - AT) * (x - BT)) := 
sorry

end NUMINAMATH_GPT_quadratic_polynomial_AT_BT_l311_31152


namespace NUMINAMATH_GPT_silver_cube_price_l311_31159

theorem silver_cube_price
  (price_2inch_cube : ℝ := 300) (side_length_2inch : ℝ := 2) (side_length_4inch : ℝ := 4) : 
  price_4inch_cube = 2400 := 
by 
  sorry

end NUMINAMATH_GPT_silver_cube_price_l311_31159


namespace NUMINAMATH_GPT_find_value_of_x_l311_31137

theorem find_value_of_x (w : ℕ) (x y z : ℕ) (h₁ : x = y / 3) (h₂ : y = z / 6) (h₃ : z = 2 * w) (hw : w = 45) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_x_l311_31137


namespace NUMINAMATH_GPT_find_x_l311_31186

theorem find_x (x : ℕ) (h1 : 8^x = 2^9) (h2 : 8 = 2^3) : x = 3 := by
  sorry

end NUMINAMATH_GPT_find_x_l311_31186


namespace NUMINAMATH_GPT_max_notebooks_with_budget_l311_31197

/-- Define the prices and quantities of notebooks -/
def notebook_price : ℕ := 2
def four_pack_price : ℕ := 6
def seven_pack_price : ℕ := 9
def max_budget : ℕ := 15

def total_notebooks (single_packs four_packs seven_packs : ℕ) : ℕ :=
  single_packs + 4 * four_packs + 7 * seven_packs

theorem max_notebooks_with_budget : 
  ∃ (single_packs four_packs seven_packs : ℕ), 
    notebook_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs ≤ max_budget ∧ 
    booklet_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs + total_notebooks single_packs four_packs seven_packs = 11 := 
by
  sorry

end NUMINAMATH_GPT_max_notebooks_with_budget_l311_31197
