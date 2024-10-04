import Mathlib

namespace arithmetic_seq_a8_l14_14835

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h1 : a 5 = 10)
  (h2 : a 1 + a 2 + a 3 = 3) :
  a 8 = 19 := sorry

end arithmetic_seq_a8_l14_14835


namespace smallest_positive_debt_resolvable_l14_14071

theorem smallest_positive_debt_resolvable :
  ∃ p g : ℤ, 280 * p + 200 * g = 40 ∧
  ∀ k : ℤ, k > 0 → (∃ p g : ℤ, 280 * p + 200 * g = k) → 40 ≤ k :=
by
  sorry

end smallest_positive_debt_resolvable_l14_14071


namespace plane_equation_through_point_and_line_l14_14904

theorem plane_equation_through_point_and_line :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1 ∧
  ∀ (x y z : ℝ),
    (A * x + B * y + C * z + D = 0 ↔ 
    (∃ (t : ℝ), x = -3 * t - 1 ∧ y = 2 * t + 3 ∧ z = t - 2) ∨ 
    (x = 0 ∧ y = 7 ∧ z = -7)) :=
by
  -- sorry, implementing proofs is not required.
  sorry

end plane_equation_through_point_and_line_l14_14904


namespace find_m_n_l14_14313

theorem find_m_n :
  ∃ m n : ℕ, m! + 12 = n^2 ∧ (m, n) = (4, 6) :=
by
  use 4, 6
  split
  · calc (4! + 12) = 24 + 12 : by rw Nat.factorial_succ
                 ... = 36     : by norm_num
                 ... = 6^2    : by norm_num
  · sorry

end find_m_n_l14_14313


namespace part1_part2_l14_14432

-- Proof for part 1
theorem part1 (x : ℤ) : (x - 1 ∣ x - 3 ↔ (x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3)) :=
by sorry

-- Proof for part 2
theorem part2 (x : ℤ) : (x + 2 ∣ x^2 + 3 ↔ (x = -9 ∨ x = -3 ∨ x = -1 ∨ x = 5)) :=
by sorry

end part1_part2_l14_14432


namespace range_of_a_l14_14169

theorem range_of_a (a : ℝ) :
  (∀ x, (x^2 - x ≤ 0 → 2^(1 - x) + a ≤ 0)) ↔ (a ≤ -2) := by
  sorry

end range_of_a_l14_14169


namespace radii_ratio_interval_l14_14361

-- Define the trapezoid structure
structure Trapezoid :=
  (A B C D : Point)
  (base : segment C D)
  (parallelogram : parallelogram A B C D)

-- Define conditions
axiom trapezoid_data : Trapezoid
axiom E_on_BC : E ∈ line (B, C)
axiom E_cyclic_ACD : cyclic [A, C, D, E] -- E, A, C, D lie on the same circle
axiom ABCD_cyclic_BCA : cyclic [A, B, C] -- A, B, C lie on the same circle and tangent to CD

-- Lengths definitions
noncomputable def length_AB : ℝ := 12
noncomputable def ratio_BE_EC : ℝ := 4 / 5

-- Theorem to be proved
theorem radii_ratio_interval (AB : line × ℝ := 12)
  (BE_EC_ratio : (line → ℝ := 4 / 5)
  (BC len: ℝ) (AB : line * ℝ) : 
  let first_circle_radius : ℝ := calculate_radius_first_circle C E
  let second_circle_radius : ℝ := calculate_radius_second_circle A B :=
  \frac{first_circle_radius}{second_circle_radius} ∈ \left( \frac{2}{3}, \frac{4}{3} \right) := sorry

end radii_ratio_interval_l14_14361


namespace find_x_l14_14679

theorem find_x (a b x: ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : x = 16 * a^(3 / 2) := by
  sorry

end find_x_l14_14679


namespace nonneg_integer_solutions_otimes_l14_14305

noncomputable def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

theorem nonneg_integer_solutions_otimes :
  {x : ℕ | otimes 2 x ≥ 3} = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_otimes_l14_14305


namespace abs_inequality_equiv_l14_14908

theorem abs_inequality_equiv (x : ℝ) : 1 ≤ |x - 2| ∧ |x - 2| ≤ 7 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9) :=
by
  sorry

end abs_inequality_equiv_l14_14908


namespace time_to_cross_bridge_l14_14489

noncomputable def train_length := 300  -- in meters
noncomputable def train_speed_kmph := 72  -- in km/h
noncomputable def bridge_length := 1500  -- in meters

-- Define the conversion from km/h to m/s
noncomputable def train_speed_mps := (train_speed_kmph * 1000) / 3600  -- in m/s

-- Define the total distance to be traveled
noncomputable def total_distance := train_length + bridge_length  -- in meters

-- Define the time to cross the bridge
noncomputable def time_to_cross := total_distance / train_speed_mps  -- in seconds

theorem time_to_cross_bridge : time_to_cross = 90 := by
  -- skipping the proof
  sorry

end time_to_cross_bridge_l14_14489


namespace crayons_slightly_used_l14_14851

theorem crayons_slightly_used (total_crayons : ℕ) (new_fraction : ℚ) (broken_fraction : ℚ) 
  (htotal : total_crayons = 120) (hnew : new_fraction = 1 / 3) (hbroken : broken_fraction = 20 / 100) :
  let new_crayons := total_crayons * new_fraction
  let broken_crayons := total_crayons * broken_fraction
  let slightly_used_crayons := total_crayons - new_crayons - broken_crayons
  slightly_used_crayons = 56 := 
by
  -- This is where the proof would go
  sorry

end crayons_slightly_used_l14_14851


namespace tom_average_score_increase_l14_14712

def initial_scores : List ℕ := [72, 78, 81]
def fourth_exam_score : ℕ := 90

theorem tom_average_score_increase :
  let initial_avg := (initial_scores.sum : ℚ) / (initial_scores.length : ℚ)
  let total_score_after_fourth := initial_scores.sum + fourth_exam_score
  let new_avg := (total_score_after_fourth : ℚ) / (initial_scores.length + 1 : ℚ)
  new_avg - initial_avg = 3.25 := by 
  -- Proof goes here
  sorry

end tom_average_score_increase_l14_14712


namespace star_polygon_edges_congruent_l14_14021

theorem star_polygon_edges_congruent
  (n : ℕ)
  (α β : ℝ)
  (h1 : ∀ i j : ℕ, i ≠ j → (n = 133))
  (h2 : α = (5 / 14) * β)
  (h3 : n * (α + β) = 360) :
n = 133 :=
by sorry

end star_polygon_edges_congruent_l14_14021


namespace boxes_needed_l14_14826

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end boxes_needed_l14_14826


namespace solution_l14_14214

noncomputable def problem (a b c : ℝ) : Prop :=
  (Polynomial.eval a (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval b (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval c (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)

theorem solution (a b c : ℝ) (h : problem a b c) : 
  (∃ abc : ℝ, abc = a * b * c ∧ abc = 10) →
  (a + b + c = 15) ∧ (a * b + b * c + c * a = 25) →
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
sorry

end solution_l14_14214


namespace cookout_ratio_l14_14192

theorem cookout_ratio (K_2004 K_2005 : ℕ) (h1 : K_2004 = 60) (h2 : (2 / 3) * K_2005 = 20) :
  K_2005 / K_2004 = 1 / 2 :=
by sorry

end cookout_ratio_l14_14192


namespace area_of_rectangle_l14_14575

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l14_14575


namespace sum_zero_implies_inequality_l14_14801

variable {a b c d : ℝ}

theorem sum_zero_implies_inequality
  (h : a + b + c + d = 0) :
  5 * (a * b + b * c + c * d) + 8 * (a * c + a * d + b * d) ≤ 0 := 
sorry

end sum_zero_implies_inequality_l14_14801


namespace sequence_count_646634_l14_14222

theorem sequence_count_646634 :
  let S := { p : ℤ × ℤ | 0 ≤ p.1 ∧ p.1 ≤ 11 ∧ 0 ≤ p.2 ∧ p.2 ≤ 9 } in
  ∃ (n : ℕ) (seq : list (ℤ × ℤ)), 
    (0 < n) ∧ 
    (s0 = (0,0)) ∧ 
    (s1 = (1,0)) ∧ 
    (∀ i, 2 ≤ i ∧ i ≤ n → 
      seq.nth i = (rotate seq.nth (i-2) seq.nth (i-1))) ∧ 
    (seq.nodup) -> seq.length = 646634 :=
by sorry

end sequence_count_646634_l14_14222


namespace fifth_equation_correct_l14_14809

def fifth_equation (x : ℕ) := x= 1^3 + 2^3 + 3^3 + 4^3 + 5^3 = 15^2

theorem fifth_equation_correct : fifth_equation (225) :=
by
  unfold fifth_equation
  sorry

end fifth_equation_correct_l14_14809


namespace range_of_m_l14_14356

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_m (h1 : ∀ x : ℝ, f (-x) = f x)
                   (h2 : ∀ a b : ℝ, a ≠ b → a ≤ 0 → b ≤ 0 → (f a - f b) / (a - b) < 0)
                   (h3 : f (m + 1) < f 2) : 
  ∃ m : ℝ, -3 < m ∧ m < 1 :=
sorry

end range_of_m_l14_14356


namespace range_b_intersects_ellipse_l14_14319

open Real

noncomputable def line_intersects_ellipse (b : ℝ) : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < π → ∃ x y : ℝ, x = 2 * cos θ ∧ y = 4 * sin θ ∧ y = x + b

theorem range_b_intersects_ellipse :
  ∀ b : ℝ, line_intersects_ellipse b ↔ b ∈ Set.Icc (-2 : ℝ) (2 * sqrt 5) :=
by
  sorry

end range_b_intersects_ellipse_l14_14319


namespace h_of_k_neg_3_l14_14929

def h (x : ℝ) : ℝ := 4 - real.sqrt x

def k (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem h_of_k_neg_3 : h (k (-3)) = 4 - 3 * real.sqrt 2 :=
by
  sorry

end h_of_k_neg_3_l14_14929


namespace nice_set_l14_14208

def nice (P : Set (ℤ × ℤ)) : Prop :=
  ∀ (a b c d : ℤ), (a, b) ∈ P ∧ (c, d) ∈ P → (b, a) ∈ P ∧ (a + c, b - d) ∈ P

def is_solution (p q : ℤ) : Prop :=
  Int.gcd p q = 1 ∧ p % 2 ≠ q % 2

theorem nice_set (p q : ℤ) (P : Set (ℤ × ℤ)) :
  nice P → (p, q) ∈ P → is_solution p q → P = Set.univ := 
  sorry

end nice_set_l14_14208


namespace angle_sum_solution_l14_14559

theorem angle_sum_solution
  (x : ℝ)
  (h : 3 * x + 140 = 360) :
  x = 220 / 3 :=
by
  sorry

end angle_sum_solution_l14_14559


namespace inequality_pow_l14_14327

variable {n : ℕ}

theorem inequality_pow (hn : n > 0) : 
  (3:ℝ) / 2 ≤ (1 + (1:ℝ) / (2 * n)) ^ n ∧ (1 + (1:ℝ) / (2 * n)) ^ n < 2 := 
sorry

end inequality_pow_l14_14327


namespace student_B_incorrect_l14_14001

-- Define the quadratic function and the non-zero condition on 'a'
def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x - 6

-- Conditions stated by the students
def student_A_condition (a b : ℝ) : Prop := -b / (2 * a) = 1
def student_B_condition (a b : ℝ) : Prop := quadratic a b 3 = -6
def student_C_condition (a b : ℝ) : Prop := (4 * a * (-6) - b^2) / (4 * a) = -8
def student_D_condition (a b : ℝ) : Prop := quadratic a b 3 = 0

-- The proof problem: Student B's conclusion is incorrect
theorem student_B_incorrect : 
  ∀ (a b : ℝ), 
  a ≠ 0 → 
  student_A_condition a b ∧ 
  student_C_condition a b ∧ 
  student_D_condition a b → 
  ¬ student_B_condition a b :=
by 
  -- problem converted to Lean problem format 
  -- based on the conditions provided
  sorry

end student_B_incorrect_l14_14001


namespace derivative_at_x1_is_12_l14_14988

theorem derivative_at_x1_is_12 : 
  (deriv (fun x : ℝ => (2 * x + 1) ^ 2) 1) = 12 :=
by
  sorry

end derivative_at_x1_is_12_l14_14988


namespace calculate_sum_calculate_product_l14_14149

theorem calculate_sum : 13 + (-7) + (-6) = 0 :=
by sorry

theorem calculate_product : (-8) * (-4 / 3) * (-0.125) * (5 / 4) = -5 / 3 :=
by sorry

end calculate_sum_calculate_product_l14_14149


namespace simplest_fraction_l14_14267

theorem simplest_fraction (x y : ℝ) (h1 : 2 * x ≠ 0) (h2 : x + y ≠ 0) :
  let A := (2 * x) / (4 * x^2)
  let B := (x^2 + y^2) / (x + y)
  let C := (x^2 + 2 * x + 1) / (x + 1)
  let D := (x^2 - 4) / (x + 2)
  B = (x^2 + y^2) / (x + y) ∧
  A ≠ (2 * x) / (4 * x^2) ∧
  C ≠ (x^2 + 2 * x + 1) / (x + 1) ∧
  D ≠ (x^2 - 4) / (x + 2) := sorry

end simplest_fraction_l14_14267


namespace verify_sum_of_new_rates_proof_l14_14706

-- Given conditions and initial setup
variable (k : ℕ)
variable (h_initial : ℕ := 5 * k) -- Hanhan's initial hourly rate
variable (x_initial : ℕ := 4 * k) -- Xixi's initial hourly rate
variable (increment : ℕ := 20)    -- Increment in hourly rates

-- New rates after increment
variable (h_new : ℕ := h_initial + increment) -- Hanhan's new hourly rate
variable (x_new : ℕ := x_initial + increment) -- Xixi's new hourly rate

-- Given ratios
variable (initial_ratio : h_initial / x_initial = 5 / 4) 
variable (new_ratio : h_new / x_new = 6 / 5)

-- Target sum of the new hourly rates
def sum_of_new_rates_proof : Prop :=
  h_new + x_new = 220

theorem verify_sum_of_new_rates_proof : sum_of_new_rates_proof k :=
by
  sorry

end verify_sum_of_new_rates_proof_l14_14706


namespace rectangular_field_area_l14_14586

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l14_14586


namespace intersection_range_l14_14498

theorem intersection_range (k : ℝ) :
  (∃ x y : ℝ, y = k * x + k + 2 ∧ y = -2 * x + 4 ∧ x > 0 ∧ y > 0) ↔ -2/3 < k ∧ k < 2 :=
by
  sorry

end intersection_range_l14_14498


namespace volume_of_smaller_cube_l14_14736

noncomputable def volume_of_larger_cube : ℝ := 343
noncomputable def number_of_smaller_cubes : ℝ := 343
noncomputable def surface_area_difference : ℝ := 1764

theorem volume_of_smaller_cube (v_lc : ℝ) (n_sc : ℝ) (sa_diff : ℝ) :
  v_lc = volume_of_larger_cube →
  n_sc = number_of_smaller_cubes →
  sa_diff = surface_area_difference →
  ∃ (v_sc : ℝ), v_sc = 1 :=
by sorry

end volume_of_smaller_cube_l14_14736


namespace area_ratio_of_squares_l14_14995

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 16 * b) : a ^ 2 = 16 * b ^ 2 := by
  sorry

end area_ratio_of_squares_l14_14995


namespace find_S6_l14_14998

def arithmetic_sum (n : ℕ) : ℝ := sorry
def S_3 := 6
def S_9 := 27

theorem find_S6 : ∃ S_6 : ℝ, S_6 = 15 ∧ 
                              S_6 - S_3 = (6 + (S_9 - S_6)) / 2 :=
sorry

end find_S6_l14_14998


namespace value_of_expression_l14_14264

theorem value_of_expression : ((25 + 8)^2 - (8^2 + 25^2) = 400) :=
by 
  sorry

end value_of_expression_l14_14264


namespace new_ratio_cooks_waiters_l14_14451

theorem new_ratio_cooks_waiters
  (initial_ratio : ℕ → ℕ → Prop)
  (cooks waiters : ℕ) :
  initial_ratio 9 24 → 
  12 + waiters = 36 →
  initial_ratio 3 8 →
  9 * 4 = 36 :=
by
  intros h1 h2 h3
  sorry

end new_ratio_cooks_waiters_l14_14451


namespace derivative_at_pi_over_4_l14_14775

-- Define the function f and its derivative
def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

-- State the theorem we want to prove
theorem derivative_at_pi_over_4 : f' (π / 4) = 0 :=
by 
  -- This is the placeholder for the proof
  sorry

end derivative_at_pi_over_4_l14_14775


namespace speed_of_train_in_km_per_hr_l14_14134

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l14_14134


namespace solve_equation_l14_14404

theorem solve_equation (x : ℝ) (h1 : x + 2 ≠ 0) (h2 : 3 - x ≠ 0) :
  (3 * x - 5) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -15 / 2 :=
by
  sorry

end solve_equation_l14_14404


namespace smallest_floor_sum_l14_14185

theorem smallest_floor_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (⌊(a + b + d) / c⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + a + d) / b⌋) = 9 :=
sorry

end smallest_floor_sum_l14_14185


namespace sin_cos_eq_sqrt2_l14_14628

theorem sin_cos_eq_sqrt2 (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 2 * Real.pi) (h2 : Real.sin x - Real.cos x = Real.sqrt 2) :
  x = (3 * Real.pi) / 4 :=
sorry

end sin_cos_eq_sqrt2_l14_14628


namespace slightly_used_crayons_correct_l14_14850

def total_crayons : ℕ := 120
def new_crayons : ℕ := total_crayons / 3
def broken_crayons : ℕ := (total_crayons * 20) / 100
def slightly_used_crayons : ℕ := total_crayons - new_crayons - broken_crayons

theorem slightly_used_crayons_correct : slightly_used_crayons = 56 := sorry

end slightly_used_crayons_correct_l14_14850


namespace basketball_games_played_l14_14433

theorem basketball_games_played (G : ℕ) (H1 : 35 ≤ G) (H2 : 25 ≥ 0) (H3 : 64 = 100 * (48 / (G + 25))):
  G = 50 :=
sorry

end basketball_games_played_l14_14433


namespace brad_reads_26_pages_per_day_l14_14650

-- Define conditions
def greg_daily_reading : ℕ := 18
def brad_extra_pages : ℕ := 8

-- Define Brad's daily reading
def brad_daily_reading : ℕ := greg_daily_reading + brad_extra_pages

-- The theorem to be proven
theorem brad_reads_26_pages_per_day : brad_daily_reading = 26 := by
  sorry

end brad_reads_26_pages_per_day_l14_14650


namespace positive_numbers_inequality_l14_14230

theorem positive_numbers_inequality
  (x y z : ℝ)
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x * y + y * z + z * x = 6) :
  (1 / (2 * Real.sqrt 2 + x^2 * (y + z)) + 
   1 / (2 * Real.sqrt 2 + y^2 * (x + z)) + 
   1 / (2 * Real.sqrt 2 + z^2 * (x + y))) <= 
  (1 / (x * y * z)) :=
by
  sorry

end positive_numbers_inequality_l14_14230


namespace proof_problem_l14_14407

variable {a b c : ℝ}

-- Condition: a < 0
variable (ha : a < 0)
-- Condition: b > 0
variable (hb : b > 0)
-- Condition: c > 0
variable (hc : c > 0)
-- Condition: a < b < c
variable (hab : a < b) (hbc : b < c)

-- Proof statement
theorem proof_problem :
  (ab * b < b * c) ∧
  (a * c < b * c) ∧
  (a + c < b + c) ∧
  (c / a < 1) :=
  by
    sorry

end proof_problem_l14_14407


namespace find_k_value_l14_14648

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + 3 * x + 7
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 3 * x^3 - k * x^2 + 4

theorem find_k_value : (f 5 - g 5 k = 45) → k = 27 / 25 :=
by
  intro h
  sorry

end find_k_value_l14_14648


namespace domain_of_g_eq_7_infty_l14_14897

noncomputable def domain_function (x : ℝ) : Prop := (2 * x + 1 ≥ 0) ∧ (x - 7 > 0)

theorem domain_of_g_eq_7_infty : 
  (∀ x : ℝ, domain_function x ↔ x > 7) :=
by 
  -- We declare the structure of our proof problem here.
  -- The detailed proof steps would follow.
  sorry

end domain_of_g_eq_7_infty_l14_14897


namespace parallelogram_height_l14_14469

theorem parallelogram_height
  (area : ℝ)
  (base : ℝ)
  (h_area : area = 375)
  (h_base : base = 25) :
  (area / base) = 15 :=
by
  sorry

end parallelogram_height_l14_14469


namespace brownies_left_is_zero_l14_14072

-- Definitions of the conditions
def total_brownies : ℝ := 24
def tina_lunch : ℝ := 1.5 * 5
def tina_dinner : ℝ := 0.5 * 5
def tina_total : ℝ := tina_lunch + tina_dinner
def husband_total : ℝ := 0.75 * 5
def guests_total : ℝ := 2.5 * 2
def daughter_total : ℝ := 2 * 3

-- Formulate the proof statement
theorem brownies_left_is_zero :
    total_brownies - (tina_total + husband_total + guests_total + daughter_total) = 0 := by
  sorry

end brownies_left_is_zero_l14_14072


namespace largest_n_unique_k_l14_14855

theorem largest_n_unique_k (n k : ℕ) :
  (frac9_17_lt_frac n (n + k) ∧ frac n (n + k) lt frac8_15) 
  ∧ (∀ (n1 k1 : ℕ), frac9_17_lt_frac n1 (n1 + k1) ∧ frac n1 (n1 + k1) lt frac8_15 
  → (n1 ≤ 136 
  ∧ ((n1 = 136) → (k1 = unique_k))))
  :=
sorry

def frac9_17_lt_frac (a b : ℕ) : Prop := 
  (9:ℚ) / 17 < (a : ℚ) / b

def frac (a b : ℕ) : ℚ :=
  (a : ℚ) / b

def frac8_15 := 
  (8:ℚ) / 15

def unique_k : ℕ :=
  119

end largest_n_unique_k_l14_14855


namespace minimum_cost_to_buy_additional_sheets_l14_14719

def total_sheets : ℕ := 98
def students : ℕ := 12
def cost_per_sheet : ℕ := 450

theorem minimum_cost_to_buy_additional_sheets : 
  (students * (1 + total_sheets / students) - total_sheets) * cost_per_sheet = 4500 :=
by {
  sorry
}

end minimum_cost_to_buy_additional_sheets_l14_14719


namespace tim_minus_tom_l14_14063

def sales_tax_rate : ℝ := 0.07
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def city_tax_rate : ℝ := 0.05

noncomputable def tim_total : ℝ :=
  let price_with_tax := original_price * (1 + sales_tax_rate)
  price_with_tax * (1 - discount_rate)

noncomputable def tom_total : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_sales_tax := discounted_price * (1 + sales_tax_rate)
  price_with_sales_tax * (1 + city_tax_rate)

theorem tim_minus_tom : tim_total - tom_total = -4.82 := 
by sorry

end tim_minus_tom_l14_14063


namespace limit_sum_odd_terms_l14_14785

-- Given conditions
def geom_seq_sum (n : ℕ) : ℝ := (1 / 2) ^ n - 1

-- Definition of the sequence terms
noncomputable def a (n : ℕ) : ℝ := 
  if n = 0 then geom_seq_sum 1
  else geom_seq_sum (n + 1) - geom_seq_sum n

-- Summing terms at odd indices
noncomputable def sum_odd_terms (n : ℕ) : ℝ :=
  finset.sum (finset.range n) (λ i, a (2 * i + 1))

-- Statement of the proof to be shown
theorem limit_sum_odd_terms :
  tendsto (λ n, sum_odd_terms n) at_top (𝓝 (-2 / 3)) :=
sorry

end limit_sum_odd_terms_l14_14785


namespace second_man_start_time_l14_14881

theorem second_man_start_time (P Q : Type) (departure_time_P departure_time_Q meeting_time arrival_time_P arrival_time_Q : ℕ) 
(distance speed : ℝ) (first_man_speed second_man_speed : ℕ → ℝ)
(h1 : departure_time_P = 6) 
(h2 : arrival_time_Q = 10) 
(h3 : arrival_time_P = 12) 
(h4 : meeting_time = 9) 
(h5 : ∀ t, 0 ≤ t ∧ t ≤ 4 → first_man_speed t = distance / 4)
(h6 : ∀ t, second_man_speed t = distance / 4)
(h7 : ∀ t, second_man_speed t * (meeting_time - t) = (3 * distance / 4))
: departure_time_Q = departure_time_P :=
by 
  sorry

end second_man_start_time_l14_14881


namespace total_arrangements_l14_14566

-- Question: 
-- Given 6 teachers and 4 schools with specific constraints, 
-- prove that the number of different ways to arrange the teachers is 240.

def teachers : List Char := ['A', 'B', 'C', 'D', 'E', 'F']

def schools : List Nat := [1, 2, 3, 4]

def B_and_D_in_same_school (assignment: Char → Nat) : Prop :=
  assignment 'B' = assignment 'D'

def each_school_has_at_least_one_teacher (assignment: Char → Nat) : Prop :=
  ∀ s ∈ schools, ∃ t ∈ teachers, assignment t = s

noncomputable def num_arrangements : Nat := sorry -- This would actually involve complex combinatorial calculations

theorem total_arrangements : num_arrangements = 240 :=
  sorry

end total_arrangements_l14_14566


namespace max_piece_length_l14_14893

theorem max_piece_length (L1 L2 L3 L4 : ℕ) (hL1 : L1 = 48) (hL2 : L2 = 72) (hL3 : L3 = 120) (hL4 : L4 = 144) 
  (h_min_pieces : ∀ L k, L = 48 ∨ L = 72 ∨ L = 120 ∨ L = 144 → k > 0 → L / k ≥ 5) : 
  ∃ k, k = 8 ∧ ∀ L, (L = L1 ∨ L = L2 ∨ L = L3 ∨ L = L4) → L % k = 0 :=
by
  sorry

end max_piece_length_l14_14893


namespace green_turtles_1066_l14_14297

def number_of_turtles (G H : ℕ) : Prop :=
  H = 2 * G ∧ G + H = 3200

theorem green_turtles_1066 : ∃ G : ℕ, number_of_turtles G (2 * G) ∧ G = 1066 :=
by
  sorry

end green_turtles_1066_l14_14297


namespace negation_of_p_l14_14224

-- Define the proposition p
def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 > 0

-- State the theorem: the negation of proposition p
theorem negation_of_p : ¬ proposition_p ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by 
  sorry

end negation_of_p_l14_14224


namespace sum_1_to_50_l14_14413

-- Given conditions: initial values, and the loop increments
def initial_index : ℕ := 1
def initial_sum : ℕ := 0
def loop_condition (i : ℕ) : Prop := i ≤ 50

-- Increment step for index and running total in loop
def increment_index (i : ℕ) : ℕ := i + 1
def increment_sum (S : ℕ) (i : ℕ) : ℕ := S + i

-- Expected sum output for the given range
def sum_up_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the sum of integers from 1 to 50
theorem sum_1_to_50 : sum_up_to_n 50 = 1275 := by
  sorry

end sum_1_to_50_l14_14413


namespace greatest_third_side_l14_14083

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l14_14083


namespace like_terms_sum_l14_14117

theorem like_terms_sum (m n : ℤ) (h_x : 1 = m - 2) (h_y : 2 = n + 3) : m + n = 2 :=
by
  sorry

end like_terms_sum_l14_14117


namespace jony_speed_l14_14563

theorem jony_speed :
  let start_block := 10
  let end_block := 90
  let turn_around_block := 70
  let block_length := 40 -- meters
  let start_time := 0 -- 07:00 in minutes from the start of his walk
  let end_time := 40 -- 07:40 in minutes from the start of his walk
  let total_blocks_walked := (end_block - start_block) + (end_block - turn_around_block)
  let total_distance := total_blocks_walked * block_length
  let total_time := end_time - start_time
  total_distance / total_time = 100 :=
by
  sorry

end jony_speed_l14_14563


namespace intersection_M_N_l14_14348

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l14_14348


namespace find_n_tangent_l14_14163

theorem find_n_tangent (n : ℤ) (h1 : -180 < n) (h2 : n < 180) (h3 : ∃ k : ℤ, 210 = n + 180 * k) : n = 30 :=
by
  -- Proof steps would go here
  sorry

end find_n_tangent_l14_14163


namespace angle_B_l14_14943

-- Define the conditions
variables {A B C : ℝ} (a b c : ℝ)
variable (h : a^2 + c^2 = b^2 + ac)

-- State the theorem
theorem angle_B (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  B = π / 3 :=
sorry

end angle_B_l14_14943


namespace sara_received_quarters_correct_l14_14233

-- Define the initial number of quarters Sara had
def sara_initial_quarters : ℕ := 21

-- Define the total number of quarters Sara has now
def sara_total_quarters : ℕ := 70

-- Define the number of quarters Sara received from her dad
def sara_received_quarters : ℕ := 49

-- State that the number of quarters Sara received can be deduced by the difference
theorem sara_received_quarters_correct :
  sara_total_quarters = sara_initial_quarters + sara_received_quarters :=
by simp [sara_initial_quarters, sara_total_quarters, sara_received_quarters]

end sara_received_quarters_correct_l14_14233


namespace sum_of_digits_8_pow_2003_l14_14111

noncomputable def units_digit (n : ℕ) : ℕ :=
n % 10

noncomputable def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

noncomputable def sum_of_tens_and_units_digits (n : ℕ) : ℕ :=
units_digit n + tens_digit n

theorem sum_of_digits_8_pow_2003 :
  sum_of_tens_and_units_digits (8 ^ 2003) = 2 :=
by
  sorry

end sum_of_digits_8_pow_2003_l14_14111


namespace sonya_falls_6_l14_14819

def number_of_falls_steven : ℕ := 3
def number_of_falls_stephanie : ℕ := number_of_falls_steven + 13
def number_of_falls_sonya : ℕ := (number_of_falls_stephanie / 2) - 2

theorem sonya_falls_6 : number_of_falls_sonya = 6 := 
by
  -- The actual proof is to be filled in here
  sorry

end sonya_falls_6_l14_14819


namespace cube_expansion_l14_14116

variable {a b : ℝ}

theorem cube_expansion (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 :=
  sorry

end cube_expansion_l14_14116


namespace linear_system_solution_l14_14977

/-- Given a system of three linear equations:
      x + y + z = 1
      a x + b y + c z = h
      a² x + b² y + c² z = h²
    Prove that the solution x, y, z is given by:
    x = (h - b)(h - c) / (a - b)(a - c)
    y = (h - a)(h - c) / (b - a)(b - c)
    z = (h - a)(h - b) / (c - a)(c - b) -/
theorem linear_system_solution (a b c h : ℝ) (x y z : ℝ) :
  x + y + z = 1 →
  a * x + b * y + c * z = h →
  a^2 * x + b^2 * y + c^2 * z = h^2 →
  x = (h - b) * (h - c) / ((a - b) * (a - c)) ∧
  y = (h - a) * (h - c) / ((b - a) * (b - c)) ∧
  z = (h - a) * (h - b) / ((c - a) * (c - b)) :=
by
  intros
  sorry

end linear_system_solution_l14_14977


namespace email_scam_check_l14_14426

-- Define the condition for receiving an email about winning a car
def received_email (info: String) : Prop :=
  info = "You received an email informing you that you have won a car. You are asked to provide your mobile phone number for contact and to transfer 150 rubles to a bank card to cover the postage fee for sending the invitation letter."

-- Define what indicates a scam
def is_scam (info: String) : Prop :=
  info = "Request for mobile number already known to the sender and an upfront payment."

-- Proving that the information in the email implies it is a scam
theorem email_scam_check (info: String) (h1: received_email info) : is_scam info :=
by
  sorry

end email_scam_check_l14_14426


namespace intersection_points_l14_14301

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 12

theorem intersection_points :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ ¬(x = x ∧ y = y) → 0 = 1 :=
sorry

end intersection_points_l14_14301


namespace sum_of_first_n_natural_numbers_single_digit_l14_14367

theorem sum_of_first_n_natural_numbers_single_digit (n : ℕ) :
  (∃ a : ℕ, a ≤ 9 ∧ (a ≠ 0) ∧ 37 * (3 * a) = n * (n + 1) / 2) ↔ (n = 36) :=
by
  sorry

end sum_of_first_n_natural_numbers_single_digit_l14_14367


namespace inequality_solution_l14_14697

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l14_14697


namespace determine_q_l14_14749

theorem determine_q (q : ℕ) (h : 81^10 = 3^q) : q = 40 :=
by
  sorry

end determine_q_l14_14749


namespace roots_imply_value_l14_14212

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end roots_imply_value_l14_14212


namespace gold_silver_weight_problem_l14_14872

theorem gold_silver_weight_problem (x y : ℕ) (h1 : 9 * x = 11 * y) (h2 : (10 * y + x) - (8 * x + y) = 13) :
  9 * x = 11 * y ∧ (10 * y + x) - (8 * x + y) = 13 :=
by
  refine ⟨h1, h2⟩

end gold_silver_weight_problem_l14_14872


namespace quadratic_inequality_solution_l14_14471

theorem quadratic_inequality_solution (x : ℝ) : 16 ≤ x ∧ x ≤ 20 → x^2 - 36 * x + 323 ≤ 3 :=
by
  sorry

end quadratic_inequality_solution_l14_14471


namespace smallest_x_for_multiple_of_625_l14_14263

theorem smallest_x_for_multiple_of_625 (x : ℕ) (hx_pos : 0 < x) : (500 * x) % 625 = 0 → x = 5 :=
by
  sorry

end smallest_x_for_multiple_of_625_l14_14263


namespace second_grade_survey_count_l14_14875

theorem second_grade_survey_count :
  ∀ (total_students first_ratio second_ratio third_ratio total_surveyed : ℕ),
  total_students = 1500 →
  first_ratio = 4 →
  second_ratio = 5 →
  third_ratio = 6 →
  total_surveyed = 150 →
  second_ratio * total_surveyed / (first_ratio + second_ratio + third_ratio) = 50 :=
by 
  intros total_students first_ratio second_ratio third_ratio total_surveyed
  sorry

end second_grade_survey_count_l14_14875


namespace MN_length_correct_l14_14281

open Real

noncomputable def MN_segment_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
  sqrt (a * b)

theorem MN_length_correct (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (MN : ℝ), MN = MN_segment_length a b h1 h2 :=
by
  use sqrt (a * b)
  exact rfl

end MN_length_correct_l14_14281


namespace painting_price_after_new_discount_l14_14234

namespace PaintingPrice

-- Define the original price and the price Sarah paid
def original_price (x : ℕ) : Prop := x / 5 = 15

-- Define the new discounted price
def new_discounted_price (y x : ℕ) : Prop := y = x * 2 / 3

-- Theorem to prove the final price considering both conditions
theorem painting_price_after_new_discount (x y : ℕ) 
  (h1 : original_price x)
  (h2 : new_discounted_price y x) : y = 50 :=
by
  sorry

end PaintingPrice

end painting_price_after_new_discount_l14_14234


namespace greatest_third_side_l14_14085

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l14_14085


namespace second_flower_shop_groups_l14_14805

theorem second_flower_shop_groups (n : ℕ) (h1 : n ≠ 0) (h2 : n ≠ 9) (h3 : Nat.lcm 9 n = 171) : n = 19 := 
by
  sorry

end second_flower_shop_groups_l14_14805


namespace angle_A_is_60_degrees_l14_14029

theorem angle_A_is_60_degrees
  (a b c : ℝ) (A : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : 0 < A) (h3 : A < 180) : 
  A = 60 := 
  sorry

end angle_A_is_60_degrees_l14_14029


namespace gravitational_force_at_384000km_l14_14059

theorem gravitational_force_at_384000km
  (d1 d2 : ℝ)
  (f1 f2 : ℝ)
  (k : ℝ)
  (h1 : d1 = 6400)
  (h2 : d2 = 384000)
  (h3 : f1 = 800)
  (h4 : f1 * d1^2 = k)
  (h5 : f2 * d2^2 = k) :
  f2 = 2 / 9 :=
by
  sorry

end gravitational_force_at_384000km_l14_14059


namespace t_bounds_f_bounds_l14_14004

noncomputable def t (x : ℝ) : ℝ := 3^x

noncomputable def f (x : ℝ) : ℝ := 9^x - 2 * 3^x + 4

theorem t_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (1/3 ≤ t x ∧ t x ≤ 9) :=
sorry

theorem f_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (3 ≤ f x ∧ f x ≤ 67) :=
sorry

end t_bounds_f_bounds_l14_14004


namespace arithmetic_sequence_common_difference_l14_14195

theorem arithmetic_sequence_common_difference :
  let a := 5
  let a_n := 50
  let S_n := 330
  exists (d n : ℤ), (a + (n - 1) * d = a_n) ∧ (n * (a + a_n) / 2 = S_n) ∧ (d = 45 / 11) :=
by
  let a := 5
  let a_n := 50
  let S_n := 330
  use 45 / 11, 12
  sorry

end arithmetic_sequence_common_difference_l14_14195


namespace inequality_implies_l14_14932

theorem inequality_implies:
  ∀ (x y : ℝ), (x > y) → (2 * x - 1 > 2 * y - 1) :=
by
  intro x y hxy
  sorry

end inequality_implies_l14_14932


namespace non_neg_int_solutions_l14_14306

def operation (a b : ℝ) : ℝ := a * (a - b) + 1

theorem non_neg_int_solutions (x : ℕ) :
  2 * (2 - x) + 1 ≥ 3 ↔ x = 0 ∨ x = 1 := by
  sorry

end non_neg_int_solutions_l14_14306


namespace range_of_a_l14_14804

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * x - 1) / (x - 1) < 0 ↔ x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) → 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  intro h
  sorry

end range_of_a_l14_14804


namespace product_of_positive_integer_solutions_l14_14166

theorem product_of_positive_integer_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (n^2 - 47 * n + 660 = p) → False :=
by
  -- Placeholder for proof, based on the problem conditions.
  sorry

end product_of_positive_integer_solutions_l14_14166


namespace prove_f_2013_l14_14171

-- Defining the function f that satisfies the given conditions
variable (f : ℕ → ℕ)

-- Conditions provided in the problem
axiom cond1 : ∀ n, f (f n) + f n = 2 * n + 3
axiom cond2 : f 0 = 1
axiom cond3 : f 2014 = 2015

-- The statement to be proven
theorem prove_f_2013 : f 2013 = 2014 := sorry

end prove_f_2013_l14_14171


namespace bernoulli_inequality_l14_14007

theorem bernoulli_inequality (n : ℕ) (h : 1 ≤ n) (x : ℝ) (h1 : x > -1) : (1 + x) ^ n ≥ 1 + n * x := 
sorry

end bernoulli_inequality_l14_14007


namespace inequality_solution_l14_14699

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l14_14699


namespace find_a_l14_14913

noncomputable def geometric_sequence_solution (a : ℝ) : Prop :=
  (a + 1) ^ 2 = (1 / (a - 1)) * (a ^ 2 - 1)

theorem find_a (a : ℝ) : geometric_sequence_solution a → a = 0 :=
by
  intro h
  sorry

end find_a_l14_14913


namespace deficit_percentage_l14_14502

variable (A B : ℝ) -- Actual lengths of the sides of the rectangle
variable (x : ℝ) -- Percentage in deficit
variable (measuredA := A * 1.06) -- One side measured 6% in excess
variable (errorPercent := 0.7) -- Error percent in area
variable (measuredB := B * (1 - x / 100)) -- Other side measured x% in deficit
variable (actualArea := A * B) -- Actual area of the rectangle
variable (calculatedArea := (A * 1.06) * (B * (1 - x / 100))) -- Calculated area with measurement errors
variable (correctArea := actualArea * (1 + errorPercent / 100)) -- Correct area considering the error

theorem deficit_percentage : 
  calculatedArea = correctArea → 
  x = 5 :=
by
  sorry

end deficit_percentage_l14_14502


namespace find_triplets_of_real_numbers_l14_14467

theorem find_triplets_of_real_numbers (x y z : ℝ) :
  (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z) ∧ 
  (3 * x^2 + 2 * y^2 + z^2 = 240) → 
  (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) := 
sorry

end find_triplets_of_real_numbers_l14_14467


namespace fraction_multiplication_l14_14747

-- Define the problem as a theorem in Lean
theorem fraction_multiplication
  (a b x : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) (ha : a ≠ 0): 
  (3 * a * b / x) * (2 * x^2 / (9 * a * b^2)) = (2 * x) / (3 * b) := 
by
  sorry

end fraction_multiplication_l14_14747


namespace length_of_BC_l14_14676

open Triangle Real

theorem length_of_BC (A B C I D E F : Point) (h_iso : is_isosceles_triangle A B C)
  (h_incenter : is_incenter I)
  (h_AI : dist A I = 3)
  (h_inradius : dist I D = 2)
  (h_touching : touches_incircle I A D B C) :
  dist B C = 4 * sqrt 5 :=
sorry

end length_of_BC_l14_14676


namespace question_1_solution_question_2_solution_l14_14483

def f (m x : ℝ) := m*x^2 - (m^2 + 1)*x + m

theorem question_1_solution (x : ℝ) :
  (f 2 x ≤ 0) ↔ (1 / 2 ≤ x ∧ x ≤ 2) :=
sorry

theorem question_2_solution (x m : ℝ) :
  (m > 0) → 
  ((0 < m ∧ m < 1 → f m x > 0 ↔ x < m ∨ x > 1 / m) ∧
  (m = 1 → f m x > 0 ↔ x ≠ 1) ∧
  (m > 1 → f m x > 0 ↔ x < 1 / m ∨ x > m)) :=
sorry

end question_1_solution_question_2_solution_l14_14483


namespace stratified_sampling_correct_l14_14885

-- Define the total number of employees
def total_employees : ℕ := 100

-- Define the number of employees in each age group
def under_30 : ℕ := 20
def between_30_and_40 : ℕ := 60
def over_40 : ℕ := 20

-- Define the number of people to be drawn
def total_drawn : ℕ := 20

-- Function to calculate number of people to be drawn from each group
def stratified_draw (group_size : ℕ) (total_size : ℕ) (drawn : ℕ) : ℕ :=
  (group_size * drawn) / total_size

-- The proof problem statement
theorem stratified_sampling_correct :
  stratified_draw under_30 total_employees total_drawn = 4 ∧
  stratified_draw between_30_and_40 total_employees total_drawn = 12 ∧
  stratified_draw over_40 total_employees total_drawn = 4 := by
  sorry

end stratified_sampling_correct_l14_14885


namespace trailing_zeros_a6_l14_14920

theorem trailing_zeros_a6:
  (∃ a : ℕ+ → ℚ, 
    a 1 = 3 / 2 ∧ 
    (∀ n : ℕ+, a (n + 1) = (1 / 2) * (a n + (1 / a n))) ∧
    (∃ k, 10^k ≤ a 6 ∧ a 6 < 10^(k + 1))) →
  (∃ m, m = 22) :=
sorry

end trailing_zeros_a6_l14_14920


namespace mooncake_packaging_problem_l14_14551

theorem mooncake_packaging_problem
  (x y : ℕ)
  (L : ℕ := 9)
  (S : ℕ := 4)
  (M : ℕ := 35)
  (h1 : L = 9)
  (h2 : S = 4)
  (h3 : M = 35) :
  9 * x + 4 * y = 35 ∧ x + y = 5 := 
by
  sorry

end mooncake_packaging_problem_l14_14551


namespace total_expenditure_of_7_people_l14_14276

theorem total_expenditure_of_7_people :
  ∃ A : ℝ, 
    (6 * 11 + (A + 6) = 7 * A) ∧
    (6 * 11 = 66) ∧
    (∃ total : ℝ, total = 6 * 11 + (A + 6) ∧ total = 84) :=
by 
  sorry

end total_expenditure_of_7_people_l14_14276


namespace zionsDadX_l14_14270

section ZionProblem

-- Define the conditions
variables (Z : ℕ) (D : ℕ) (X : ℕ)

-- Zion's current age
def ZionAge : Prop := Z = 8

-- Zion's dad's age in terms of Zion's age and X
def DadsAge : Prop := D = 4 * Z + X

-- Zion's dad's age in 10 years compared to Zion's age in 10 years
def AgeInTenYears : Prop := D + 10 = (Z + 10) + 27

-- The theorem statement to be proved
theorem zionsDadX :
  ZionAge Z →  
  DadsAge Z D X →  
  AgeInTenYears Z D →  
  X = 3 := 
sorry

end ZionProblem

end zionsDadX_l14_14270


namespace solve_equation_l14_14535

theorem solve_equation :
  ∀ (x : ℚ), x ≠ 1 → (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by
  intro x hx
  split
  { intro h
    sorry }
  { intro h
    rw [h]
    norm_num }

end solve_equation_l14_14535


namespace sum_of_three_numbers_l14_14499

theorem sum_of_three_numbers (a b c : ℝ) (h₁ : a + b = 31) (h₂ : b + c = 48) (h₃ : c + a = 59) :
  a + b + c = 69 :=
by
  sorry

end sum_of_three_numbers_l14_14499


namespace count_multiples_of_12_l14_14184

theorem count_multiples_of_12 (a b : ℤ) (h1 : a = 5) (h2 : b = 145) :
  ∃ n : ℕ, (12 * n + 12 ≤ b) ∧ (12 * n + 12 > a) ∧ n = 12 :=
by
  sorry

end count_multiples_of_12_l14_14184


namespace ion_electronic_structure_l14_14543

theorem ion_electronic_structure (R M Z n m X : ℤ) (h1 : R + X = M - n) (h2 : M - n = Z - m) (h3 : n > m) : M > Z ∧ Z > R := 
by 
  sorry

end ion_electronic_structure_l14_14543


namespace probability_of_rolling_8_l14_14979

theorem probability_of_rolling_8 :
  let num_favorable := 5
  let num_total := 36
  let probability := (5 : ℚ) / 36
  probability =
    (num_favorable : ℚ) / num_total :=
by
  sorry

end probability_of_rolling_8_l14_14979


namespace roots_of_equation_l14_14996

theorem roots_of_equation (x : ℝ) : 3 * x * (x - 1) = 2 * (x - 1) → (x = 1 ∨ x = 2 / 3) :=
by 
  intros h
  sorry

end roots_of_equation_l14_14996


namespace number_of_M_partitions_l14_14391

-- Definitions for A and M
def A : Finset ℕ := Finset.range 2002 |>.image (λ x => x + 1)
def M : Set ℕ := {1001, 2003, 3005}

-- Definition of M-free set
def MFreeSet (B : Finset ℕ) : Prop :=
  ∀ {m n : ℕ}, m ∈ B → n ∈ B → m + n ∉ M

-- Definition of M-partition
def MPartition (A1 A2 : Finset ℕ) : Prop :=
  A1 ∪ A2 = A ∧ A1 ∩ A2 = ∅ ∧ MFreeSet A1 ∧ MFreeSet A2

-- The theorem to prove
theorem number_of_M_partitions :
  ∃ n : ℕ, n = 2 ^ 501 :=
sorry

end number_of_M_partitions_l14_14391


namespace orcs_carry_swords_l14_14539

theorem orcs_carry_swords:
  (let total_swords := 1200 in
   let squads := 10 in
   let orcs_per_squad := 8 in
   let total_orcs := squads * orcs_per_squad in
   let swords_per_orc := total_swords / total_orcs in
   swords_per_orc = 15) :=
by
  sorry

end orcs_carry_swords_l14_14539


namespace combinations_of_painting_options_l14_14806

theorem combinations_of_painting_options : 
  let colors := 6
  let methods := 3
  let finishes := 2
  colors * methods * finishes = 36 := by
  sorry

end combinations_of_painting_options_l14_14806


namespace vertex_of_parabola_is_max_and_correct_l14_14460

theorem vertex_of_parabola_is_max_and_correct (x y : ℝ) (h : y = -3 * x^2 + 6 * x + 1) :
  (x, y) = (1, 4) ∧ ∃ ε > 0, ∀ z : ℝ, abs (z - x) < ε → y ≥ -3 * z^2 + 6 * z + 1 :=
by
  sorry

end vertex_of_parabola_is_max_and_correct_l14_14460


namespace greatest_third_side_l14_14104

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l14_14104


namespace smallest_tree_height_correct_l14_14846

-- Defining the conditions
def TallestTreeHeight : ℕ := 108
def MiddleTreeHeight (tallest : ℕ) : ℕ := (tallest / 2) - 6
def SmallestTreeHeight (middle : ℕ) : ℕ := middle / 4

-- Proof statement
theorem smallest_tree_height_correct :
  SmallestTreeHeight (MiddleTreeHeight TallestTreeHeight) = 12 :=
by
  -- Here we would put the proof, but we are skipping it with sorry.
  sorry

end smallest_tree_height_correct_l14_14846


namespace cannot_sum_to_nine_l14_14303

def sum_pairs (a b c d : ℕ) : List ℕ :=
  [a + b, c + d, a + c, b + d, a + d, b + c]

theorem cannot_sum_to_nine :
  ∀ (a b c d : ℕ), a ≠ 5 ∧ b ≠ 6 ∧ c ≠ 5 ∧ d ≠ 6 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b ≠ 11 ∧ a + c ≠ 11 ∧ a + d ≠ 11 ∧ b + c ≠ 11 ∧ b + d ≠ 11 ∧ c + d ≠ 11 →
  ¬9 ∈ sum_pairs a b c d :=
by
  intros a b c d h
  sorry

end cannot_sum_to_nine_l14_14303


namespace volume_of_box_ground_area_of_box_l14_14604

-- Given conditions
variable (l w h : ℕ)
variable (hl : l = 20)
variable (hw : w = 15)
variable (hh : h = 5)

-- Define volume and ground area
def volume (l w h : ℕ) : ℕ := l * w * h
def ground_area (l w : ℕ) : ℕ := l * w

-- Theorem to prove the correct volume
theorem volume_of_box : volume l w h = 1500 := by
  rw [hl, hw, hh]
  sorry

-- Theorem to prove the correct ground area
theorem ground_area_of_box : ground_area l w = 300 := by
  rw [hl, hw]
  sorry

end volume_of_box_ground_area_of_box_l14_14604


namespace velocity_at_3_velocity_at_4_l14_14741

-- Define the distance as a function of time
def s (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Define the velocity as the derivative of the distance
noncomputable def v (t : ℝ) : ℝ := deriv s t

theorem velocity_at_3 : v 3 = 20 :=
by
  sorry

theorem velocity_at_4 : v 4 = 26 :=
by
  sorry

end velocity_at_3_velocity_at_4_l14_14741


namespace minimum_money_lost_l14_14969

-- Define the conditions and setup the problem

def check_amount : ℕ := 1270
def T_used (F : ℕ) : Σ' T, (T = F + 1 ∨ T = F - 1) :=
sorry

def money_used (T F : ℕ) : ℕ := 10 * T + 50 * F

def total_bills_used (T F : ℕ) : Prop := T + F = 15

theorem minimum_money_lost : (∃ T F, (T = F + 1 ∨ T = F - 1) ∧ T + F = 15 ∧ (check_amount - (10 * T + 50 * F) = 800)) :=
sorry

end minimum_money_lost_l14_14969


namespace correct_weight_misread_l14_14986

theorem correct_weight_misread : 
  ∀ (x : ℝ) (n : ℝ) (avg1 : ℝ) (avg2 : ℝ) (misread : ℝ),
  n = 20 → avg1 = 58.4 → avg2 = 59 → misread = 56 → 
  (n * avg2 - n * avg1 + misread) = x → 
  x = 68 :=
by
  intros x n avg1 avg2 misread
  intros h1 h2 h3 h4 h5
  sorry

end correct_weight_misread_l14_14986


namespace supermarket_selection_expected_value_l14_14379

noncomputable def small_supermarkets := 72
noncomputable def medium_supermarkets := 24
noncomputable def large_supermarkets := 12
noncomputable def total_supermarkets := small_supermarkets + medium_supermarkets + large_supermarkets
noncomputable def selected_supermarkets := 9

-- Problem (I)
noncomputable def small_selected := (small_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def medium_selected := (medium_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def large_selected := (large_supermarkets * selected_supermarkets) / total_supermarkets

theorem supermarket_selection :
  small_selected = 6 ∧ medium_selected = 2 ∧ large_selected = 1 :=
sorry

-- Problem (II)
noncomputable def further_analysis := 3
noncomputable def prob_small := small_selected / selected_supermarkets
noncomputable def E_X := prob_small * further_analysis

theorem expected_value :
  E_X = 2 :=
sorry

end supermarket_selection_expected_value_l14_14379


namespace carson_gardening_time_l14_14150

-- Definitions of the problem conditions
def lines_to_mow : ℕ := 40
def minutes_per_line : ℕ := 2
def rows_of_flowers : ℕ := 8
def flowers_per_row : ℕ := 7
def minutes_per_flower : ℚ := 0.5

-- Total time calculation for the proof 
theorem carson_gardening_time : 
  (lines_to_mow * minutes_per_line) + (rows_of_flowers * flowers_per_row * minutes_per_flower) = 108 := 
by 
  sorry

end carson_gardening_time_l14_14150


namespace piastres_in_6th_purse_l14_14506

theorem piastres_in_6th_purse (x : ℕ) (sum : ℕ := 10) (total : ℕ := 150)
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) = 150)
  (h2 : x * 2 ≥ x + 9)
  (n : ℕ := 5):
  x + n = 15 :=
  sorry

end piastres_in_6th_purse_l14_14506


namespace overlapping_area_fraction_l14_14573

variable (Y X : ℝ)
variable (hY : 0 < Y)
variable (hX : X = (1 / 8) * (2 * Y - X))

theorem overlapping_area_fraction : X = (2 / 9) * Y :=
by
  -- We define the conditions and relationships stated in the problem
  -- Prove the theorem accordingly
  sorry

end overlapping_area_fraction_l14_14573


namespace cookies_on_first_plate_l14_14549

theorem cookies_on_first_plate :
  ∃ a1 a2 a3 a4 a5 a6 : ℤ, 
  a2 = 7 ∧ 
  a3 = 10 ∧
  a4 = 14 ∧
  a5 = 19 ∧
  a6 = 25 ∧
  a2 = a1 + 2 ∧ 
  a3 = a2 + 3 ∧ 
  a4 = a3 + 4 ∧ 
  a5 = a4 + 5 ∧ 
  a6 = a5 + 6 ∧ 
  a1 = 5 :=
sorry

end cookies_on_first_plate_l14_14549


namespace train_speed_is_72_km_per_hr_l14_14129

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l14_14129


namespace area_of_triangle_l14_14968

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 8 = 1

def foci_distance (F1 F2 : ℝ × ℝ) : Prop := (F1.1, F1.2) = (-3, 0) ∧ (F2.1, F2.2) = (3, 0)

def point_on_hyperbola (x y : ℝ) : Prop := hyperbola x y

def distance_ratios (P F1 F2 : ℝ × ℝ) : Prop := 
  let PF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2
  let PF2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  PF1 / PF2 = 3 / 4

theorem area_of_triangle {P F1 F2 : ℝ × ℝ} 
  (H1 : foci_distance F1 F2)
  (H2 : point_on_hyperbola P.1 P.2)
  (H3 : distance_ratios P F1 F2) :
  let area := 1 / 2 * (6:ℝ) * (8:ℝ) * Real.sqrt 5
  area = 8 * Real.sqrt 5 := 
sorry

end area_of_triangle_l14_14968


namespace largest_multiple_of_18_with_digits_9_or_0_l14_14542

theorem largest_multiple_of_18_with_digits_9_or_0 :
  ∃ (n : ℕ), (n = 9990) ∧ (n % 18 = 0) ∧ (∀ d ∈ (n.digits 10), d = 9 ∨ d = 0) ∧ (n / 18 = 555) :=
by
  sorry

end largest_multiple_of_18_with_digits_9_or_0_l14_14542


namespace cover_rectangle_with_polyomino_l14_14309

-- Defining the conditions under which the m x n rectangle can be covered by the given polyomino
theorem cover_rectangle_with_polyomino (m n : ℕ) :
  (6 ∣ (m * n)) →
  (m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5) →
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) →
  ((3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ (12 ∣ (m * n))) :=
sorry

end cover_rectangle_with_polyomino_l14_14309


namespace find_k_values_l14_14627

/-- 
Prove that the values of k such that the positive difference between the 
roots of 3x^2 + 5x + k = 0 equals the sum of the squares of the roots 
are exactly (70 + 10sqrt(33))/8 and (70 - 10sqrt(33))/8.
-/
theorem find_k_values (k : ℝ) :
  (∀ (a b : ℝ), (3 * a^2 + 5 * a + k = 0 ∧ 3 * b^2 + 5 * b + k = 0 ∧ |a - b| = a^2 + b^2))
  ↔ (k = (70 + 10 * Real.sqrt 33) / 8 ∨ k = (70 - 10 * Real.sqrt 33) / 8) :=
sorry

end find_k_values_l14_14627


namespace product_expression_evaluation_l14_14148

theorem product_expression_evaluation :
  (1 + 2 / 1) * (1 + 2 / 2) * (1 + 2 / 3) * (1 + 2 / 4) * (1 + 2 / 5) * (1 + 2 / 6) - 1 = 25 / 3 :=
by
  sorry

end product_expression_evaluation_l14_14148


namespace dot_product_of_ab_ac_l14_14948

def vec_dot (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem dot_product_of_ab_ac :
  vec_dot (1, -2) (2, -2) = 6 := by
  sorry

end dot_product_of_ab_ac_l14_14948


namespace mean_age_correct_l14_14540

def children_ages : List ℕ := [6, 6, 9, 12]

def number_of_children : ℕ := 4

def sum_of_ages (ages : List ℕ) : ℕ := ages.sum

def mean_age (ages : List ℕ) (num_children : ℕ) : ℚ :=
  sum_of_ages ages / num_children

theorem mean_age_correct :
  mean_age children_ages number_of_children = 8.25 := by
  sorry

end mean_age_correct_l14_14540


namespace quadratic_root_range_quadratic_product_of_roots_l14_14061

-- Problem (1): Prove the range of m.
theorem quadratic_root_range (m : ℝ) :
  (∀ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 → x1 ≠ x2) ↔ m < 1 := 
sorry

-- Problem (2): Prove the existence of m such that x1 * x2 = 0.
theorem quadratic_product_of_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 ∧ x1 * x2 = 0) ↔ m = -1 := 
sorry

end quadratic_root_range_quadratic_product_of_roots_l14_14061


namespace no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l14_14431

-- Problem 1: There is no value of m that makes the lines parallel.
theorem no_solution_for_parallel_lines (m : ℝ) :
  ¬ ∃ m, (2 * m^2 + m - 3) / (m^2 - m) = 1 := sorry

-- Problem 2: The values of a that make the lines perpendicular.
theorem values_of_a_for_perpendicular_lines (a : ℝ) :
  (a = 1 ∨ a = -3) ↔ (a * (a - 1) + (1 - a) * (2 * a + 3) = 0) := sorry

end no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l14_14431


namespace range_of_a_l14_14938

theorem range_of_a (a : ℝ) :
  (∃ x : ℤ, 2 * (x : ℝ) - 1 > 3 ∧ x ≤ a) ∧ (∀ x : ℤ, 2 * (x : ℝ) - 1 > 3 → x ≤ a) → 5 ≤ a ∧ a < 6 :=
by
  sorry

end range_of_a_l14_14938


namespace EF_squared_correct_l14_14703

-- Define the problem setup and the proof goal.
theorem EF_squared_correct :
  ∀ (A B C D E F : Type)
  (side : ℝ)
  (h1 : side = 10)
  (BE DF AE CF : ℝ)
  (h2 : BE = 7)
  (h3 : DF = 7)
  (h4 : AE = 15)
  (h5 : CF = 15)
  (EF_squared : ℝ),
  EF_squared = 548 :=
by
  sorry

end EF_squared_correct_l14_14703


namespace lloyd_hourly_rate_l14_14516

variable (R : ℝ)  -- Lloyd's regular hourly rate

-- Conditions
def lloyd_works_regular_hours_per_day : Prop := R > 0
def lloyd_earns_excess_rate : Prop := 1.5 * R > 0
def lloyd_worked_hours : Prop := 10.5 > 7.5
def lloyd_earned_amount : Prop := 7.5 * R + 3 * 1.5 * R = 66

-- Theorem statement
theorem lloyd_hourly_rate (hr_pos : lloyd_works_regular_hours_per_day R)
                           (excess_rate : lloyd_earns_excess_rate R)
                           (worked_hours : lloyd_worked_hours)
                           (earned_amount : lloyd_earned_amount R) : 
    R = 5.5 :=
by sorry

end lloyd_hourly_rate_l14_14516


namespace rainy_days_l14_14386

theorem rainy_days :
  ∃ (A : Finset ℕ) (H : A.card = 5), 
  ( ∃ (B : Finset (Finset ℕ)), 
      (∀ b ∈ B, b.card = 3 ∧ (∃ c : ℕ, c ∈ b ∧ c + 1 ∈ b ∧ c + 2 ∈ b)) ∧ 
      B.card = 9 ) := 
sorry

end rainy_days_l14_14386


namespace seq_formula_l14_14221

def S (n : ℕ) (a : ℕ → ℤ) : ℤ := 2 * a n + 1

theorem seq_formula (a : ℕ → ℤ) (S_n : ℕ → ℤ)
  (hS : ∀ n, S_n n = S n a) :
  a = fun n => -2^(n-1) := by
  sorry

end seq_formula_l14_14221


namespace older_grandchild_pancakes_eaten_l14_14364

theorem older_grandchild_pancakes_eaten (initial_pancakes : ℕ) (remaining_pancakes : ℕ)
  (younger_eat_per_cycle : ℕ) (older_eat_per_cycle : ℕ) (bake_per_cycle : ℕ)
  (n : ℕ) 
  (h_initial : initial_pancakes = 19)
  (h_remaining : remaining_pancakes = 11)
  (h_younger_eat : younger_eat_per_cycle = 1)
  (h_older_eat : older_eat_per_cycle = 3)
  (h_bake : bake_per_cycle = 2)
  (h_reduction : initial_pancakes - remaining_pancakes = n * (younger_eat_per_cycle + older_eat_per_cycle - bake_per_cycle)) :
  older_eat_per_cycle * n = 12 :=
begin
  sorry
end

end older_grandchild_pancakes_eaten_l14_14364


namespace cube_volume_l14_14065

theorem cube_volume (s : ℝ) (hs : 12 * s = 96) : s^3 = 512 := by
  have s_eq : s = 8 := by
    linarith
  rw s_eq
  norm_num

end cube_volume_l14_14065


namespace infinitely_many_positive_integers_in_sequence_l14_14249

open Function

theorem infinitely_many_positive_integers_in_sequence
  (a : ℕ → ℕ) 
  (ha1 : a 1 = 0) 
  (hrec : ∀ n : ℕ, n ≥ 1 → (n + 1) ^ 3 * a (n + 1) = 2 * n ^ 2 * (2 * n + 1) * a n + 2 * (3 * n + 1)) 
  (hbinom : ∀ p : ℕ, Nat.Prime p → p ^ 2 ∣ Nat.choose (2 * p) p - 2) :
  ∃ infinitely_many n, a n ∈ ℕ ∧ a n > 0 := by
  sorry

end infinitely_many_positive_integers_in_sequence_l14_14249


namespace incorrect_relationship_f_pi4_f_pi_l14_14393

open Real

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_derivative_exists : ∀ x : ℝ, DifferentiableAt ℝ f x
axiom f_derivative_lt_sin2x : ∀ x : ℝ, 0 < x → deriv f x < (sin x) ^ 2
axiom f_symmetric_property : ∀ x : ℝ, f (-x) + f x = 2 * (sin x) ^ 2

theorem incorrect_relationship_f_pi4_f_pi : ¬ (f (π / 4) < f π) :=
by sorry

end incorrect_relationship_f_pi4_f_pi_l14_14393


namespace proof_l_shaped_area_l14_14127

-- Define the overall rectangle dimensions
def overall_length : ℕ := 10
def overall_width : ℕ := 7

-- Define the dimensions of the removed rectangle
def removed_length : ℕ := overall_length - 3
def removed_width : ℕ := overall_width - 2

-- Calculate the areas
def overall_area : ℕ := overall_length * overall_width
def removed_area : ℕ := removed_length * removed_width
def l_shaped_area : ℕ := overall_area - removed_area

-- The theorem to be proved
theorem proof_l_shaped_area : l_shaped_area = 35 := by
  sorry

end proof_l_shaped_area_l14_14127


namespace gain_percentage_calculation_l14_14606

theorem gain_percentage_calculation 
  (C S : ℝ)
  (h1 : 30 * S = 40 * C) :
  (10 * S / (30 * C)) * 100 = 44.44 :=
by
  sorry

end gain_percentage_calculation_l14_14606


namespace find_large_number_l14_14868

theorem find_large_number (L S : ℤ)
  (h1 : L - S = 2415)
  (h2 : L = 21 * S + 15) : 
  L = 2535 := 
sorry

end find_large_number_l14_14868


namespace problem1_problem2_l14_14772

noncomputable def f (x m : ℝ) := x * log x - (1 / 2) * m * x^2 - x

-- Problem 1: Prove that if f is decreasing on (0, +∞), then m >= 1/e
theorem problem1 (m : ℝ) (h_decreasing : ∀ x > 0, (deriv (λ x, f x m)) x ≤ 0) : m ≥ 1 / real.exp 1 :=
sorry

-- Problem 2: Prove that if f has two extreme points on (0, +∞), then ln x_1 + ln x_2 > 2
theorem problem2 (m x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 < x2)
  (h_extreme : deriv (λ x, f x m) x1 = 0 ∧ deriv (λ x, f x m) x2 = 0) : log x1 + log x2 > 2 :=
sorry

end problem1_problem2_l14_14772


namespace rad_times_trivia_eq_10000_l14_14238

theorem rad_times_trivia_eq_10000 
  (h a r v d m i t : ℝ)
  (H1 : h * a * r * v * a * r * d = 100)
  (H2 : m * i * t = 100)
  (H3 : h * m * m * t = 100) :
  (r * a * d) * (t * r * i * v * i * a) = 10000 := 
  sorry

end rad_times_trivia_eq_10000_l14_14238


namespace distance_AD_btw_41_and_42_l14_14688

noncomputable def distance_between (x y : ℝ × ℝ) : ℝ :=
  Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

theorem distance_AD_btw_41_and_42 :
  let A := (0, 0)
  let B := (15, 0)
  let C := (15, 5 * Real.sqrt 3)
  let D := (15, 5 * Real.sqrt 3 + 30)

  41 < distance_between A D ∧ distance_between A D < 42 :=
by
  sorry

end distance_AD_btw_41_and_42_l14_14688


namespace sum_sqrt_inequality_l14_14638

theorem sum_sqrt_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (3 / 2) * (a + b + c) ≥ (Real.sqrt (a^2 + b * c) + Real.sqrt (b^2 + c * a) + Real.sqrt (c^2 + a * b)) :=
by
  sorry

end sum_sqrt_inequality_l14_14638


namespace range_of_a_l14_14731

def f (x a : ℝ) := |x - 2| + |x + a|

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 3) → a ≤ -5 ∨ a ≥ 1 :=
  sorry

end range_of_a_l14_14731


namespace slope_l3_is_5_over_6_l14_14226

noncomputable theory

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, -3)
def B : ℝ × ℝ := (2, 2)
def C (x : ℝ) : ℝ × ℝ := (x, 2)

-- Define the lines l₁, l₂, and the constraint for l₃ passing through A and C
def line_l1 (p : ℝ × ℝ) : Prop := 4 * p.1 - 3 * p.2 = 2
def line_l2 (p : ℝ × ℝ) : Prop := p.2 = 2

-- Define the area of triangle ABC
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Define the slope function of a line given two points
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- The proof problem statement
theorem slope_l3_is_5_over_6 : ∃ x : ℝ, 
  (line_l1 B) ∧ (line_l1 A) ∧ (line_l2 B) ∧ (line_l2 (C x)) ∧
  (area_of_triangle A B (C x) = 5) ∧ (slope A (C x) = 5 / 6) :=
sorry

end slope_l3_is_5_over_6_l14_14226


namespace marius_scored_3_more_than_darius_l14_14894

theorem marius_scored_3_more_than_darius 
  (D M T : ℕ) 
  (h1 : D = 10) 
  (h2 : T = D + 5) 
  (h3 : M + D + T = 38) : 
  M = D + 3 := 
by
  sorry

end marius_scored_3_more_than_darius_l14_14894


namespace value_of_k_l14_14373

theorem value_of_k (k : ℤ) : 
  (∃ a b : ℤ, x^2 + k * x + 81 = a^2 * x^2 + 2 * a * b * x + b^2) → (k = 18 ∨ k = -18) :=
by
  sorry

end value_of_k_l14_14373


namespace problem_statement_l14_14458

-- Definition of the function f with the given condition
def satisfies_condition (f : ℝ → ℝ) := ∀ (α β : ℝ), f (α + β) - (f α + f β) = 2008

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) := ∀ (x : ℝ), f (-x) = -f x

-- Main statement to prove in Lean
theorem problem_statement (f : ℝ → ℝ) (h : satisfies_condition f) : is_odd (fun x => f x + 2008) :=
sorry

end problem_statement_l14_14458


namespace inscribed_regular_polygon_sides_l14_14026

theorem inscribed_regular_polygon_sides (n : ℕ) (h_central_angle : 360 / n = 72) : n = 5 :=
by
  sorry

end inscribed_regular_polygon_sides_l14_14026


namespace inequality_solution_l14_14702

theorem inequality_solution
  (x : ℝ) :
  x ∉ {2, 3, 4, 5, 6, 7} →
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  (x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) :=
by
  -- Proof goes here
  sorry

end inequality_solution_l14_14702


namespace meryll_questions_l14_14684

/--
Meryll wants to write a total of 35 multiple-choice questions and 15 problem-solving questions. 
She has written \(\frac{2}{5}\) of the multiple-choice questions and \(\frac{1}{3}\) of the problem-solving questions.
We need to prove that she needs to write 31 more questions in total.
-/
theorem meryll_questions : (35 - (2 / 5) * 35) + (15 - (1 / 3) * 15) = 31 := by
  sorry

end meryll_questions_l14_14684


namespace composite_has_at_least_three_factors_l14_14122

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ n ∧ b ∣ n ∧ c ∣ n :=
sorry

end composite_has_at_least_three_factors_l14_14122


namespace handshake_count_l14_14452

-- Define the number of team members, referees, and the total number of handshakes
def num_team_members := 7
def num_referees := 3
def num_coaches := 2

-- Calculate the handshakes
def team_handshakes := num_team_members * num_team_members
def player_refhandshakes := (2 * num_team_members) * num_referees
def coach_handshakes := num_coaches * (2 * num_team_members + num_referees)

-- The total number of handshakes
def total_handshakes := team_handshakes + player_refhandshakes + coach_handshakes

-- The proof statement
theorem handshake_count : total_handshakes = 125 := 
by
  -- Placeholder for proof
  sorry

end handshake_count_l14_14452


namespace first_runner_meets_conditions_l14_14853

noncomputable def first_runner_time := 11

theorem first_runner_meets_conditions (T : ℕ) (second_runner_time third_runner_time : ℕ) (meet_time : ℕ)
  (h1 : second_runner_time = 4)
  (h2 : third_runner_time = 11 / 2)
  (h3 : meet_time = 44)
  (h4 : meet_time % T = 0)
  (h5 : meet_time % second_runner_time = 0)
  (h6 : meet_time % third_runner_time = 0) : 
  T = first_runner_time :=
by
  sorry

end first_runner_meets_conditions_l14_14853


namespace math_equivalence_l14_14497

theorem math_equivalence (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) (hbc : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c := 
by 
  sorry

end math_equivalence_l14_14497


namespace probability_of_event_A_l14_14421

def probability_event_A : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes

-- Statement of the theorem
theorem probability_of_event_A :
  probability_event_A = 1 / 6 :=
by
  -- This is where the proof would go, replaced with sorry for now.
  sorry

end probability_of_event_A_l14_14421


namespace minimal_withdrawals_proof_l14_14005

-- Defining the conditions
def red_marbles : ℕ := 200
def blue_marbles : ℕ := 300
def green_marbles : ℕ := 400

def max_red_withdrawal_per_time : ℕ := 1
def max_blue_withdrawal_per_time : ℕ := 2
def max_total_withdrawal_per_time : ℕ := 5

-- The target minimal number of withdrawals
def minimal_withdrawals : ℕ := 200

-- Lean statement of the proof problem
theorem minimal_withdrawals_proof :
  ∃ (w : ℕ), w = minimal_withdrawals ∧ 
    (∀ n, n ≤ w →
      (n = 200 ∧ 
       (∀ r b g, r ≤ max_red_withdrawal_per_time ∧ b ≤ max_blue_withdrawal_per_time ∧ (r + b + g) ≤ max_total_withdrawal_per_time))) :=
sorry

end minimal_withdrawals_proof_l14_14005


namespace exists_nat_number_reduce_by_57_l14_14750

theorem exists_nat_number_reduce_by_57 :
  ∃ (N : ℕ), ∃ (k : ℕ) (a x : ℕ),
    N = 10^k * a + x ∧
    10^k * a + x = 57 * x ∧
    N = 7125 :=
sorry

end exists_nat_number_reduce_by_57_l14_14750


namespace rectangular_field_area_l14_14596

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l14_14596


namespace car_sales_total_l14_14436

theorem car_sales_total (a b c : ℕ) (h1 : a = 14) (h2 : b = 16) (h3 : c = 27):
  a + b + c = 57 :=
by
  repeat {rwa [h1, h2, h3]}
  sorry

end car_sales_total_l14_14436


namespace number_of_mappings_n_elements_l14_14958

theorem number_of_mappings_n_elements
  (A : Type) [Fintype A] [DecidableEq A] (n : ℕ) (h : 3 ≤ n) (f : A → A)
  (H1 : ∀ x : A, ∃ c : A, ∀ (i : ℕ), i ≥ n - 2 → f^[i] x = c)
  (H2 : ∃ x₁ x₂ : A, f^[n] x₁ ≠ f^[n] x₂) :
  ∃ m : ℕ, m = (2 * n - 5) * (n.factorial) / 2 :=
sorry

end number_of_mappings_n_elements_l14_14958


namespace candy_remaining_l14_14031

def initial_candy : ℝ := 1012.5
def talitha_took : ℝ := 283.7
def solomon_took : ℝ := 398.2
def maya_took : ℝ := 197.6

theorem candy_remaining : initial_candy - (talitha_took + solomon_took + maya_took) = 133 := 
by
  sorry

end candy_remaining_l14_14031


namespace geometric_sequence_divisible_l14_14680

theorem geometric_sequence_divisible (a1 a2 : ℝ) (h1 : a1 = 5 / 8) (h2 : a2 = 25) :
  ∃ n : ℕ, n = 7 ∧ (40^(n-1) * (5/8)) % 10^7 = 0 :=
by
  sorry

end geometric_sequence_divisible_l14_14680


namespace largest_angle_in_convex_pentagon_l14_14840

theorem largest_angle_in_convex_pentagon (x : ℕ) (h : (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 540) : 
  x + 2 = 110 :=
by
  sorry

end largest_angle_in_convex_pentagon_l14_14840


namespace inequality_solution_l14_14695

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l14_14695


namespace find_X_l14_14557

theorem find_X (X : ℝ) (h : 0.80 * X - 0.35 * 300 = 31) : X = 170 :=
by
  sorry

end find_X_l14_14557


namespace power_multiplication_eq_neg4_l14_14147

theorem power_multiplication_eq_neg4 :
  (-0.25) ^ 11 * (-4) ^ 12 = -4 := 
  sorry

end power_multiplication_eq_neg4_l14_14147


namespace log_product_eq_one_l14_14612

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem log_product_eq_one :
  log_base 2 3 * log_base 9 4 = 1 := 
by {
  sorry
}

end log_product_eq_one_l14_14612


namespace min_buses_needed_l14_14445

theorem min_buses_needed (n : ℕ) (h1 : 45 * n ≥ 500) (h2 : n ≥ 2) : n = 12 :=
sorry

end min_buses_needed_l14_14445


namespace find_c_l14_14640

theorem find_c (c : ℝ) (h : ∃ β : ℝ, (5 + β = -c) ∧ (5 * β = 45)) : c = -14 := 
  sorry

end find_c_l14_14640


namespace compare_logs_l14_14218

noncomputable def e := Real.exp 1
noncomputable def log_base_10 (x : Real) := Real.log x / Real.log 10

theorem compare_logs (x : Real) (hx : e < x ∧ x < 10) :
  let a := Real.log (Real.log x)
  let b := log_base_10 (log_base_10 x)
  let c := Real.log (log_base_10 x)
  let d := log_base_10 (Real.log x)
  c < b ∧ b < d ∧ d < a := 
sorry

end compare_logs_l14_14218


namespace men_build_walls_l14_14496

-- Define the variables
variables (a b d y : ℕ)

-- Define the work rate based on given conditions
def rate := d / (a * b)

-- Theorem to prove that y equals (a * a) / d given the conditions
theorem men_build_walls (h : a * b * y = a * a * d / a) : 
  y = a * a / d :=
by sorry

end men_build_walls_l14_14496


namespace pizza_store_total_sales_l14_14518

theorem pizza_store_total_sales (pepperoni bacon cheese : ℕ) (h1 : pepperoni = 2) (h2 : bacon = 6) (h3 : cheese = 6) :
  pepperoni + bacon + cheese = 14 :=
by sorry

end pizza_store_total_sales_l14_14518


namespace abs_diff_eq_five_l14_14035

theorem abs_diff_eq_five (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 7) : |a - b| = 5 :=
by
  sorry

end abs_diff_eq_five_l14_14035


namespace average_bracelets_per_day_l14_14076

theorem average_bracelets_per_day
  (cost_of_bike : ℕ)
  (price_per_bracelet : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (h1 : cost_of_bike = 112)
  (h2 : price_per_bracelet = 1)
  (h3 : weeks = 2)
  (h4 : days_per_week = 7) :
  (cost_of_bike / price_per_bracelet) / (weeks * days_per_week) = 8 :=
by
  sorry

end average_bracelets_per_day_l14_14076


namespace remainder_of_E_div_88_l14_14156

-- Define the given expression E and the binomial coefficient 
noncomputable def E : ℤ :=
  1 - 90 * Nat.choose 10 1 + 90 ^ 2 * Nat.choose 10 2 - 90 ^ 3 * Nat.choose 10 3 + 
  90 ^ 4 * Nat.choose 10 4 - 90 ^ 5 * Nat.choose 10 5 + 90 ^ 6 * Nat.choose 10 6 - 
  90 ^ 7 * Nat.choose 10 7 + 90 ^ 8 * Nat.choose 10 8 - 90 ^ 9 * Nat.choose 10 9 + 
  90 ^ 10 * Nat.choose 10 10

-- The theorem that we need to prove
theorem remainder_of_E_div_88 : E % 88 = 1 := by
  sorry

end remainder_of_E_div_88_l14_14156


namespace remainder_four_times_plus_six_l14_14189

theorem remainder_four_times_plus_six (n : ℤ) (h : n % 5 = 3) : (4 * n + 6) % 5 = 3 :=
by
  sorry

end remainder_four_times_plus_six_l14_14189


namespace Q_ratio_eq_one_l14_14240

noncomputable def g (x : ℂ) : ℂ := x^2007 - 2 * x^2006 + 2

theorem Q_ratio_eq_one (Q : ℂ → ℂ) (s : ℕ → ℂ) (h_root : ∀ j : ℕ, j < 2007 → g (s j) = 0) 
  (h_Q : ∀ j : ℕ, j < 2007 → Q (s j + (1 / s j)) = 0) :
  Q 1 / Q (-1) = 1 := by
  sorry

end Q_ratio_eq_one_l14_14240


namespace intersection_M_N_l14_14344

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l14_14344


namespace zhijie_suanjing_l14_14950

theorem zhijie_suanjing :
  ∃ (x y: ℕ), x + y = 100 ∧ 3 * x + y / 3 = 100 :=
by
  sorry

end zhijie_suanjing_l14_14950


namespace arrangement_same_side_of_C_l14_14069

-- Define the number of arrangements of 6 people
noncomputable def arrangements (n : ℕ) : ℕ :=
  nat.factorial n

-- Define the number of arrangements where A and B are on the same side of C
theorem arrangement_same_side_of_C :
  arrangements 6 * 2 / 3 = 480 := by
  sorry

end arrangement_same_side_of_C_l14_14069


namespace intersection_M_N_l14_14347

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l14_14347


namespace perpendicular_lines_eq_l14_14175

def dot_product (v1 v2 : Vector ℝ 3) : ℝ :=
  v1.head * v2.head + v1.tail.head * v2.tail.head + v1.tail.tail.head * v2.tail.tail.head

theorem perpendicular_lines_eq (m : ℝ) 
  (a : Vector ℝ 3 := ⟨[1, m, -1]⟩) 
  (b : Vector ℝ 3 := ⟨[-2, 1, 1]⟩) 
  (h : dot_product a b = 0) : 
  m = 3 :=
by {
  sorry
}

end perpendicular_lines_eq_l14_14175


namespace minor_premise_l14_14457

variables (A B C : Prop)

theorem minor_premise (hA : A) (hB : B) (hC : C) : B := 
by
  exact hB

end minor_premise_l14_14457


namespace num_of_nickels_l14_14380

theorem num_of_nickels (x : ℕ) (hx_eq_dimes : ∀ n, n = x → n = x) (hx_eq_quarters : ∀ n, n = x → n = 2 * x) (total_value : 5 * x + 10 * x + 50 * x = 1950) : x = 30 :=
sorry

end num_of_nickels_l14_14380


namespace ratio_when_volume_maximized_l14_14865

-- Definitions based on conditions
def cylinder_perimeter := 24

-- Definition of properties derived from maximizing the volume
def max_volume_height := 4

def max_volume_circumference := 12 - max_volume_height

-- The ratio of the circumference of the cylinder's base to its height when the volume is maximized
def max_volume_ratio := max_volume_circumference / max_volume_height

-- The theorem to be proved
theorem ratio_when_volume_maximized :
  max_volume_ratio = 2 :=
by sorry

end ratio_when_volume_maximized_l14_14865


namespace total_number_recruits_l14_14415

theorem total_number_recruits 
  (x y z : ℕ)
  (h1 : x = 50)
  (h2 : y = 100)
  (h3 : z = 170)
  (h4 : x = 4 * (y - 50) ∨ y = 4 * (z - 170) ∨ x = 4 * (z - 170)) : 
  171 + (z - 170) = 211 :=
by
  sorry

end total_number_recruits_l14_14415


namespace smaller_angle_at_7_15_l14_14611

theorem smaller_angle_at_7_15 
  (hour_hand_rate : ℕ → ℝ)
  (minute_hand_rate : ℕ → ℝ)
  (hour_time : ℕ)
  (minute_time : ℕ)
  (top_pos : ℝ)
  (smaller_angle : ℝ) 
  (h1 : hour_hand_rate hour_time + (minute_time/60) * hour_hand_rate hour_time = 217.5)
  (h2 : minute_hand_rate minute_time = 90.0)
  (h3 : |217.5 - 90.0| = smaller_angle) :
  smaller_angle = 127.5 :=
by
  sorry

end smaller_angle_at_7_15_l14_14611


namespace train_speed_is_72_km_per_hr_l14_14130

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l14_14130


namespace hands_in_class_not_including_peters_l14_14070

def total_students : ℕ := 11
def hands_per_student : ℕ := 2
def peter_hands : ℕ := 2

theorem hands_in_class_not_including_peters :  (total_students * hands_per_student) - peter_hands = 20 :=
by
  sorry

end hands_in_class_not_including_peters_l14_14070


namespace masks_purchased_in_first_batch_l14_14074

theorem masks_purchased_in_first_batch
    (cost_first_batch cost_second_batch : ℝ)
    (quantity_ratio : ℝ)
    (unit_price_difference : ℝ)
    (h1 : cost_first_batch = 1600)
    (h2 : cost_second_batch = 6000)
    (h3 : quantity_ratio = 3)
    (h4 : unit_price_difference = 2) :
    ∃ x : ℝ, (cost_first_batch / x) + unit_price_difference = (cost_second_batch / (quantity_ratio * x)) ∧ x = 200 :=
by {
    sorry
}

end masks_purchased_in_first_batch_l14_14074


namespace force_for_wrenches_l14_14836

open Real

theorem force_for_wrenches (F : ℝ) (k : ℝ) :
  (F * 12 = 3600) → 
  (k = 3600) →
  (3600 / 8 = 450) →
  (3600 / 18 = 200) →
  true :=
by
  intro hF hk h8 h18
  trivial

end force_for_wrenches_l14_14836


namespace relationship_abc_l14_14515

noncomputable def a (x : ℝ) : ℝ := Real.log x
noncomputable def b (x : ℝ) : ℝ := Real.exp (Real.log x)
noncomputable def c (x : ℝ) : ℝ := Real.exp (Real.log (1 / x))

theorem relationship_abc (x : ℝ) (h : (1 / Real.exp 1) < x ∧ x < 1) : a x < b x ∧ b x < c x :=
by
  have ha : a x = Real.log x := rfl
  have hb : b x = Real.exp (Real.log x) := rfl
  have hc : c x = Real.exp (Real.log (1 / x)) := rfl
  sorry

end relationship_abc_l14_14515


namespace inequality_one_solution_inequality_two_solution_l14_14055

-- The statement for the first inequality
theorem inequality_one_solution (x : ℝ) :
  |1 - ((2 * x - 1) / 3)| ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5 := sorry

-- The statement for the second inequality
theorem inequality_two_solution (x : ℝ) :
  (2 - x) * (x + 3) < 2 - x ↔ x < -2 ∨ x > 2 := sorry

end inequality_one_solution_inequality_two_solution_l14_14055


namespace card_probability_multiple_l14_14290

def is_multiple_of (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

def count_multiples (n k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def inclusion_exclusion (a b c : ℕ) (n : ℕ) : ℕ :=
  (count_multiples n a) + (count_multiples n b) + (count_multiples n c) - 
  (count_multiples n (Nat.lcm a b)) - (count_multiples n (Nat.lcm a c)) - 
  (count_multiples n (Nat.lcm b c)) + 
  count_multiples n (Nat.lcm a (Nat.lcm b c))

theorem card_probability_multiple (n : ℕ) 
  (a b c : ℕ) (hne : n ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (inclusion_exclusion a b c n) / n = 47 / 100 := by
  sorry

end card_probability_multiple_l14_14290


namespace colten_chickens_l14_14525

theorem colten_chickens (x : ℕ) (Quentin Skylar Colten : ℕ) 
  (h1 : Quentin + Skylar + Colten = 383)
  (h2 : Quentin = 25 + 2 * Skylar)
  (h3 : Skylar = 3 * Colten - 4) : 
  Colten = 37 := 
  sorry

end colten_chickens_l14_14525


namespace route_comparison_l14_14791

-- Definitions
def distance (P Z C : Type) : Type := ℝ

variables {P Z C : Type} -- P: Park, Z: Zoo, C: Circus
variables (x y C : ℝ)     -- x: direct distance from Park to Zoo, y: direct distance from Circus to Zoo, C: total circumference

-- Conditions
axiom h1 : x + 3 * x = C -- distance from Park to Zoo via Circus is three times longer than not via Circus
axiom h2 : y = (C - x) / 2 -- distance from Circus to Zoo directly is y
axiom h3 : 2 * y = C - x -- distance from Circus to Zoo via Park is twice as short as not via Park

-- Proof statement
theorem route_comparison (P Z C : Type) (x y C : ℝ) (h1 : x + 3 * x = C) (h2 : y = (C - x) / 2) (h3 : 2 * y = C - x) :
  let direct_route := x
  let via_zoo_route := 3 * x - x
  via_zoo_route = 11 * direct_route := 
sorry

end route_comparison_l14_14791


namespace benzene_molecular_weight_l14_14715

theorem benzene_molecular_weight (w: ℝ) (h: 4 * w = 312) : w = 78 :=
by
  sorry

end benzene_molecular_weight_l14_14715


namespace solution_l14_14475

-- Define the conditions
variable (f : ℝ → ℝ)
variable (f_odd : ∀ x, f (-x) = -f x)
variable (f_periodic : ∀ x, f (x + 1) = f (1 - x))
variable (f_cubed : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x ^ 3)

-- Define the goal
theorem solution : f 2019 = -1 :=
by sorry

end solution_l14_14475


namespace lcm_smallest_value_l14_14657

/-- The smallest possible value of lcm(k, l) for positive 5-digit integers k and l such that gcd(k, l) = 5 is 20010000. -/
theorem lcm_smallest_value (k l : ℕ) (h1 : 10000 ≤ k ∧ k < 100000) (h2 : 10000 ≤ l ∧ l < 100000) (h3 : Nat.gcd k l = 5) : Nat.lcm k l = 20010000 :=
sorry

end lcm_smallest_value_l14_14657


namespace greatest_possible_third_side_l14_14080

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l14_14080


namespace area_of_rectangle_l14_14574

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l14_14574


namespace tax_free_value_is_500_l14_14198

-- Definitions of the given conditions
def total_value : ℝ := 730
def paid_tax : ℝ := 18.40
def tax_rate : ℝ := 0.08

-- Definition of the excess value
def excess_value (E : ℝ) := tax_rate * E = paid_tax

-- Definition of the tax-free threshold value
def tax_free_limit (V : ℝ) := total_value - (paid_tax / tax_rate) = V

-- The theorem to be proven
theorem tax_free_value_is_500 : 
  ∃ V : ℝ, (total_value - (paid_tax / tax_rate) = V) ∧ V = 500 :=
  by
    sorry -- Proof to be completed

end tax_free_value_is_500_l14_14198


namespace tail_count_likelihood_draw_and_rainy_l14_14898

def coin_tosses : ℕ := 25
def heads_count : ℕ := 11
def draws_when_heads : ℕ := 7
def rainy_when_tails : ℕ := 4

theorem tail_count :
  coin_tosses - heads_count = 14 :=
sorry

theorem likelihood_draw_and_rainy :
  0 = 0 :=
sorry

end tail_count_likelihood_draw_and_rainy_l14_14898


namespace find_b_l14_14493

theorem find_b 
  (a b c d : ℚ) 
  (h1 : a = 2 * b + c) 
  (h2 : b = 2 * c + d) 
  (h3 : 2 * c = d + a - 1) 
  (h4 : d = a - c) : 
  b = 2 / 9 :=
by
  -- Proof is omitted (the proof steps would be inserted here)
  sorry

end find_b_l14_14493


namespace range_of_b_l14_14940

theorem range_of_b 
  (b : ℝ)
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁ + b = 3 - real.sqrt (4 * x₁ - x₁^2)) ∧ 
    (x₂ + b = 3 - real.sqrt (4 * x₂ - x₂^2))) :
  b ∈ set.Ioo ((1 - real.sqrt 29) / 2) ((1 + real.sqrt 29) / 2) :=
by
  sorry

end range_of_b_l14_14940


namespace remaining_puppies_l14_14974

def initial_puppies : Nat := 8
def given_away_puppies : Nat := 4

theorem remaining_puppies : initial_puppies - given_away_puppies = 4 := 
by 
  sorry

end remaining_puppies_l14_14974


namespace find_positive_number_l14_14867
-- Prove the positive number x that satisfies the condition is 8
theorem find_positive_number (x : ℝ) (hx : 0 < x) :
    x + 8 = 128 * (1 / x) → x = 8 :=
by
  intro h
  sorry

end find_positive_number_l14_14867


namespace find_f2_plus_g2_l14_14479

-- Functions f and g are defined
variable (f g : ℝ → ℝ)

-- Conditions based on the problem
def even_function : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function : Prop := ∀ x : ℝ, g (-x) = g x
def function_equation : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

-- Lean Theorem Statement
theorem find_f2_plus_g2 (h1 : even_function f) (h2 : odd_function g) (h3 : function_equation f g) :
  f 2 + g 2 = -4 :=
by
  sorry

end find_f2_plus_g2_l14_14479


namespace numerical_form_463001_l14_14722

theorem numerical_form_463001 : 463001 = 463001 := by
  rfl

end numerical_form_463001_l14_14722


namespace sum_of_three_squares_l14_14291

-- Using the given conditions to define the problem.
variable (square triangle : ℝ)

-- Conditions
axiom h1 : square + triangle + 2 * square + triangle = 34
axiom h2 : triangle + square + triangle + 3 * square = 40

-- Statement to prove
theorem sum_of_three_squares : square + square + square = 66 / 7 :=
by
  sorry

end sum_of_three_squares_l14_14291


namespace negative_number_zero_exponent_l14_14848

theorem negative_number_zero_exponent (a : ℤ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end negative_number_zero_exponent_l14_14848


namespace third_side_triangle_max_l14_14097

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l14_14097


namespace result_when_decreased_by_5_and_divided_by_7_l14_14191

theorem result_when_decreased_by_5_and_divided_by_7 (x y : ℤ)
  (h1 : (x - 5) / 7 = y)
  (h2 : (x - 6) / 8 = 6) :
  y = 7 :=
by
  sorry

end result_when_decreased_by_5_and_divided_by_7_l14_14191


namespace intersection_M_N_l14_14346

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l14_14346


namespace three_sport_players_l14_14668

def total_members := 50
def B := 22
def T := 28
def Ba := 18
def BT := 10
def BBa := 8
def TBa := 12
def N := 4
def All := 8

theorem three_sport_players : B + T + Ba - (BT + BBa + TBa) + All = total_members - N :=
by
suffices h : 22 + 28 + 18 - (10 + 8 + 12) + 8 = 50 - 4
exact h
-- The detailed proof is left as an exercise
sorry

end three_sport_players_l14_14668


namespace emma_harry_weight_l14_14945

theorem emma_harry_weight (e f g h : ℕ) 
  (h1 : e + f = 280) 
  (h2 : f + g = 260) 
  (h3 : g + h = 290) : 
  e + h = 310 := 
sorry

end emma_harry_weight_l14_14945


namespace calculate_purple_pants_l14_14056

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def plaid_shirts : ℕ := 3
def non_plaid_non_purple_items : ℕ := 21

theorem calculate_purple_pants : total_pants - (non_plaid_non_purple_items - (total_shirts - plaid_shirts)) = 5 :=
by 
  sorry

end calculate_purple_pants_l14_14056


namespace parity_of_expression_l14_14672

theorem parity_of_expression
  (a b c : ℕ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_c_pos : c > 0) :
  ((3^a + (b + 2)^2 * c) % 2 = 1 ↔ c % 2 = 0) ∧ 
  ((3^a + (b + 2)^2 * c) % 2 = 0 ↔ c % 2 = 1) :=
by sorry

end parity_of_expression_l14_14672


namespace div_by_11_l14_14505

theorem div_by_11 (x y : ℤ) (k : ℤ) (h : 14 * x + 13 * y = 11 * k) : 11 ∣ (19 * x + 9 * y) :=
by
  sorry

end div_by_11_l14_14505


namespace value_of_a2_l14_14797

theorem value_of_a2 
  (a1 a2 a3 : ℝ)
  (h_seq : ∃ d : ℝ, (-8) = -8 + d * 0 ∧ a1 = -8 + d * 1 ∧ 
                     a2 = -8 + d * 2 ∧ a3 = -8 + d * 3 ∧ 
                     10 = -8 + d * 4) :
  a2 = 1 :=
by {
  sorry
}

end value_of_a2_l14_14797


namespace replace_star_l14_14720

theorem replace_star (x : ℕ) : 2 * 18 * 14 = 6 * x * 7 → x = 12 :=
sorry

end replace_star_l14_14720


namespace circumcenter_locus_of_triangle_MBN_l14_14608

-- Define the problem structure
variable {A B C P M N O G: Point} -- Declare the variables as abstract points in the plane

-- Conditions of the problem
axiom hABC : EquilateralTriangle A B C
axiom hP : (InteriorPoint P A B C) ∧ (Angle A P C = 120)
axiom hM : LineIntersect A B (Ray P C) M
axiom hN : LineIntersect B C (Ray P A) N

-- Definition of the problem statement
theorem circumcenter_locus_of_triangle_MBN:
  locus_of_circumcenter_triangle M B N = perp_bisector_segment B G
  sorry

end circumcenter_locus_of_triangle_MBN_l14_14608


namespace ted_cookies_eaten_l14_14634

def cookies_per_tray : ℕ := 12
def trays_per_day : ℕ := 2
def days_baking : ℕ := 6
def cookies_per_day : ℕ := trays_per_day * cookies_per_tray
def total_cookies_baked : ℕ := days_baking * cookies_per_day
def cookies_eaten_by_frank : ℕ := days_baking
def cookies_before_ted : ℕ := total_cookies_baked - cookies_eaten_by_frank
def cookies_left_after_ted : ℕ := 134

theorem ted_cookies_eaten : cookies_before_ted - cookies_left_after_ted = 4 := by
  sorry

end ted_cookies_eaten_l14_14634


namespace root_expr_value_eq_175_div_11_l14_14215

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end root_expr_value_eq_175_div_11_l14_14215


namespace parabola_translation_l14_14886

theorem parabola_translation :
  ∀ (x : ℝ),
  (∃ x' y' : ℝ, x' = x - 1 ∧ y' = 2 * x' ^ 2 - 3 ∧ y = y' + 3) →
  (y = 2 * x ^ 2) :=
by
  sorry

end parabola_translation_l14_14886


namespace range_of_m_l14_14474

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := x^2 - x + 1

-- Define the interval
def interval (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 2

-- Prove the range of m
theorem range_of_m (m : ℝ) : (∀ x : ℝ, interval x → f x > 2 * x + m) ↔ m < - 5 / 4 :=
by
  -- This is the theorem statement, hence the proof starts here
  sorry

end range_of_m_l14_14474


namespace tangent_line_at_M_l14_14011

noncomputable def isOnCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def M : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem tangent_line_at_M (hM : isOnCircle (M.1) (M.2)) : (∀ x y, M.1 = x ∨ M.2 = y → x + y = Real.sqrt 2) :=
by
  sorry

end tangent_line_at_M_l14_14011


namespace necessary_but_not_sufficient_l14_14637

variables (α β : Plane) (m : Line)

-- Define what it means for planes and lines to be perpendicular
def plane_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry

-- The main theorem to be established
theorem necessary_but_not_sufficient :
  (plane_perpendicular α β) → (line_perpendicular_plane m β) ∧ ¬ ((plane_perpendicular α β) ↔ (line_perpendicular_plane m β)) :=
sorry

end necessary_but_not_sufficient_l14_14637


namespace marks_difference_l14_14824

theorem marks_difference (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 48) 
  (h2 : (A + B + C + D) / 4 = 47) 
  (h3 : E > D) 
  (h4 : (B + C + D + E) / 4 = 48) 
  (h5 : A = 43) : 
  E - D = 3 := 
sorry

end marks_difference_l14_14824


namespace number_when_added_by_5_is_30_l14_14237

theorem number_when_added_by_5_is_30 (x: ℕ) (h: x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end number_when_added_by_5_is_30_l14_14237


namespace zero_of_f_l14_14918

noncomputable def f (x : ℝ) : ℝ := (|Real.log x - Real.log 2|) - (1 / 3) ^ x

theorem zero_of_f :
  ∃ x1 x2 : ℝ, x1 < x2 ∧ (f x1 = 0) ∧ (f x2 = 0) ∧
  (1 < x1 ∧ x1 < 2) ∧ (2 < x2) := 
sorry

end zero_of_f_l14_14918


namespace prob_not_answered_after_three_rings_l14_14545

def prob_first_ring_answered := 0.1
def prob_second_ring_answered := 0.25
def prob_third_ring_answered := 0.45

theorem prob_not_answered_after_three_rings : 
  1 - prob_first_ring_answered - prob_second_ring_answered - prob_third_ring_answered = 0.2 :=
by
  sorry

end prob_not_answered_after_three_rings_l14_14545


namespace find_t_l14_14519

theorem find_t (t : ℕ) : 
  t > 3 ∧ (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) → t = 6 := 
by
  intro h
  have h1 : t > 3 := h.1
  have h2 : (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) := h.2
  sorry

end find_t_l14_14519


namespace rectangle_area_l14_14580

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l14_14580


namespace largest_difference_rounding_l14_14609

variable (A B : ℝ)
variable (estimate_A estimate_B : ℝ)
variable (within_A within_B : ℝ)
variable (diff : ℝ)

axiom est_A : estimate_A = 55000
axiom est_B : estimate_B = 58000
axiom cond_A : within_A = 0.15
axiom cond_B : within_B = 0.10

axiom bounds_A : 46750 ≤ A ∧ A ≤ 63250
axiom bounds_B : 52727 ≤ B ∧ B ≤ 64444

noncomputable def max_possible_difference : ℝ :=
  max (abs (B - A)) (abs (A - B))

theorem largest_difference_rounding :
  max_possible_difference A B = 18000 :=
by
  sorry

end largest_difference_rounding_l14_14609


namespace smallest_n_containing_375_consecutively_l14_14057

theorem smallest_n_containing_375_consecutively :
  ∃ (m n : ℕ), m < n ∧ Nat.gcd m n = 1 ∧ (n = 8) ∧ (∀ (d : ℕ), d < 1000 →
  ∃ (k : ℕ), k * d % n = m ∧ (d / 100) % 10 = 3 ∧ (d / 10) % 10 = 7 ∧ d % 10 = 5) :=
sorry

end smallest_n_containing_375_consecutively_l14_14057


namespace earl_up_second_time_l14_14900

def earl_floors (n top start up1 down up2 dist : ℕ) : Prop :=
  start + up1 - down + up2 = top - dist

theorem earl_up_second_time 
  (start up1 down top dist : ℕ) 
  (h_start : start = 1) 
  (h_up1 : up1 = 5) 
  (h_down : down = 2) 
  (h_top : top = 20) 
  (h_dist : dist = 9) : 
  ∃ up2, earl_floors n top start up1 down up2 dist ∧ up2 = 7 :=
by
  use 7
  sorry

end earl_up_second_time_l14_14900


namespace range_of_a_l14_14172

noncomputable def A (a : ℝ) : Set ℝ := { x | 3 + a ≤ x ∧ x ≤ 4 + 3 * a }
noncomputable def B : Set ℝ := { x | -4 ≤ x ∧ x < 5 }

theorem range_of_a (a : ℝ) : A a ⊆ B ↔ -1/2 ≤ a ∧ a < 1/3 :=
  sorry

end range_of_a_l14_14172


namespace derivative_at_pi_over_4_l14_14774

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_over_4 : (deriv f) (Real.pi / 4) = 0 := 
by
  sorry

end derivative_at_pi_over_4_l14_14774


namespace inequality_solution_l14_14978

theorem inequality_solution :
  ∀ x : ℝ, (x - 3) / (x^2 + 4 * x + 10) ≥ 0 ↔ x ≥ 3 :=
by
  sorry

end inequality_solution_l14_14978


namespace express_y_in_terms_of_x_l14_14763

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 4) : y = 4 - 5 * x :=
by
  /- Proof to be filled in here. -/
  sorry

end express_y_in_terms_of_x_l14_14763


namespace boxes_needed_to_pack_all_muffins_l14_14833

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end boxes_needed_to_pack_all_muffins_l14_14833


namespace comedies_in_terms_of_a_l14_14271

variable (T a : ℝ)
variables (Comedies Dramas Action : ℝ)
axiom Condition1 : Comedies = 0.64 * T
axiom Condition2 : Dramas = 5 * a
axiom Condition3 : Action = a
axiom Condition4 : Comedies + Dramas + Action = T

theorem comedies_in_terms_of_a : Comedies = 10.67 * a :=
by sorry

end comedies_in_terms_of_a_l14_14271


namespace greatest_third_side_l14_14103

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l14_14103


namespace correct_mean_of_values_l14_14837

variable (n : ℕ) (mu_incorrect : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mu_correct : ℝ)

theorem correct_mean_of_values
  (h1 : n = 30)
  (h2 : mu_incorrect = 150)
  (h3 : incorrect_value = 135)
  (h4 : correct_value = 165)
  : mu_correct = 151 :=
by
  let S_incorrect := mu_incorrect * n
  let S_correct := S_incorrect - incorrect_value + correct_value
  let mu_correct := S_correct / n
  sorry

end correct_mean_of_values_l14_14837


namespace pow_mod_eq_l14_14109

theorem pow_mod_eq :
  11 ^ 2023 % 5 = 1 :=
by
  sorry

end pow_mod_eq_l14_14109


namespace book_arrangements_l14_14442

theorem book_arrangements (total_books : ℕ) (at_least_in_library : ℕ) (at_least_checked_out : ℕ) 
  (h_total : total_books = 10) (h_at_least_in : at_least_in_library = 2) 
  (h_at_least_out : at_least_checked_out = 3) : 
  ∃ arrangements : ℕ, arrangements = 6 :=
by
  sorry

end book_arrangements_l14_14442


namespace tom_roses_per_day_l14_14553

-- Define variables and conditions
def total_roses := 168
def days_in_week := 7
def dozen := 12

-- Theorem to prove
theorem tom_roses_per_day : (total_roses / dozen) / days_in_week = 2 :=
by
  -- The actual proof would go here, using the sorry placeholder
  sorry

end tom_roses_per_day_l14_14553


namespace range_g_l14_14159

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + 2 * Real.arcsin x

theorem range_g : 
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -Real.pi / 2 ≤ g x ∧ g x ≤ 3 * Real.pi / 2) := 
by {
  sorry
}

end range_g_l14_14159


namespace horizontal_distance_travel_l14_14607

noncomputable def radius : ℝ := 2
noncomputable def angle_degrees : ℝ := 30
noncomputable def angle_radians : ℝ := angle_degrees * (Real.pi / 180)
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def cos_theta : ℝ := Real.cos angle_radians
noncomputable def horizontal_distance (r : ℝ) (θ : ℝ) : ℝ := (circumference r) * (Real.cos θ)

theorem horizontal_distance_travel (r : ℝ) (θ : ℝ) (h_radius : r = 2) (h_angle : θ = angle_radians) :
  horizontal_distance r θ = 2 * Real.pi * Real.sqrt 3 := 
by
  sorry

end horizontal_distance_travel_l14_14607


namespace ceiling_fraction_evaluation_l14_14621

theorem ceiling_fraction_evaluation :
  (Int.ceil ((19 : ℚ) / 8 - Int.ceil ((45 : ℚ) / 19)) / Int.ceil ((45 : ℚ) / 8 + Int.ceil ((8 * 19 : ℚ) / 45))) = 0 :=
by
  sorry

end ceiling_fraction_evaluation_l14_14621


namespace fuchsia_to_mauve_l14_14450

def fuchsia_to_mauve_amount (F : ℝ) : Prop :=
  let blue_in_fuchsia := (3 / 8) * F
  let red_in_fuchsia := (5 / 8) * F
  blue_in_fuchsia + 14 = 2 * red_in_fuchsia

theorem fuchsia_to_mauve (F : ℝ) (h : fuchsia_to_mauve_amount F) : F = 16 :=
by
  sorry

end fuchsia_to_mauve_l14_14450


namespace bus_passenger_count_l14_14434

-- Definition of the function f representing the number of passengers per trip
def passengers (n : ℕ) : ℕ :=
  120 - 2 * n

-- The total number of trips is 18 (from 9 AM to 5:30 PM inclusive)
def total_trips : ℕ := 18

-- Sum of passengers over all trips
def total_passengers : ℕ :=
  List.sum (List.map passengers (List.range total_trips))

-- Problem statement
theorem bus_passenger_count :
  total_passengers = 1854 :=
sorry

end bus_passenger_count_l14_14434


namespace s_of_4_l14_14678

noncomputable def t (x : ℚ) : ℚ := 5 * x - 14
noncomputable def s (y : ℚ) : ℚ := 
  let x := (y + 14) / 5
  x^2 + 5 * x - 4

theorem s_of_4 : s (4) = 674 / 25 := by
  sorry

end s_of_4_l14_14678


namespace root_at_neg_x0_l14_14186

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom x0_root : ∃ x0, f x0 = Real.exp x0

-- Theorem
theorem root_at_neg_x0 : 
  (∃ x0, (f (-x0) * Real.exp (-x0) + 1 = 0))
  → (∃ x0, (f x0 * Real.exp x0 + 1 = 0)) := 
sorry

end root_at_neg_x0_l14_14186


namespace cody_marbles_l14_14299

theorem cody_marbles (M : ℕ) (h1 : M / 3 + 5 + 7 = M) : M = 18 :=
by
  have h2 : 3 * M / 3 + 3 * 5 + 3 * 7 = 3 * M := by sorry
  have h3 : 3 * M / 3 = M := by sorry
  have h4 : 3 * 7 = 21 := by sorry
  have h5 : M + 15 + 21 = 3 * M := by sorry
  have h6 : M = 18 := by sorry
  exact h6

end cody_marbles_l14_14299


namespace arithmetic_sequence_n_2005_l14_14641

/-- Define an arithmetic sequence with first term a₁ = 1 and common difference d = 3. -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + (n - 1) * 3

/-- Statement of the proof problem. -/
theorem arithmetic_sequence_n_2005 : 
  ∃ n : ℕ, arithmetic_sequence n = 2005 ∧ n = 669 := 
sorry

end arithmetic_sequence_n_2005_l14_14641


namespace common_ratio_of_geometric_series_l14_14903

theorem common_ratio_of_geometric_series :
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  r = -1 / 2 :=
by
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  have : r = -1 / 2 := sorry
  exact this

end common_ratio_of_geometric_series_l14_14903


namespace find_set_B_l14_14921

def A : Set ℕ := {1, 2}
def B : Set (Set ℕ) := { x | x ⊆ A }

theorem find_set_B : B = { ∅, {1}, {2}, {1, 2} } :=
by
  sorry

end find_set_B_l14_14921


namespace angles_symmetric_about_y_axis_l14_14644

theorem angles_symmetric_about_y_axis (α β : ℝ) (k : ℤ) (h : β = (2 * ↑k + 1) * Real.pi - α) : 
  α + β = (2 * ↑k + 1) * Real.pi :=
sorry

end angles_symmetric_about_y_axis_l14_14644


namespace probability_different_plants_l14_14841

theorem probability_different_plants :
  let plants := 4
  let total_combinations := plants * plants
  let favorable_combinations := total_combinations - plants
  (favorable_combinations : ℚ) / total_combinations = 3 / 4 :=
by
  sorry

end probability_different_plants_l14_14841


namespace rectangular_field_area_l14_14585

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l14_14585


namespace range_of_m_l14_14199

theorem range_of_m (m : ℝ) (P : ℝ × ℝ) (h : P = (m + 3, m - 5)) (quadrant4 : P.1 > 0 ∧ P.2 < 0) : -3 < m ∧ m < 5 :=
by
  sorry

end range_of_m_l14_14199


namespace non_neg_int_solutions_l14_14307

def operation (a b : ℝ) : ℝ := a * (a - b) + 1

theorem non_neg_int_solutions (x : ℕ) :
  2 * (2 - x) + 1 ≥ 3 ↔ x = 0 ∨ x = 1 := by
  sorry

end non_neg_int_solutions_l14_14307


namespace simplify_expr_l14_14816

variable (x : ℝ)

theorem simplify_expr : (2 * x^2 + 5 * x - 7) - (x^2 + 9 * x - 3) = x^2 - 4 * x - 4 :=
by
  sorry

end simplify_expr_l14_14816


namespace product_mod_17_eq_zero_l14_14110

theorem product_mod_17_eq_zero :
    (2001 * 2002 * 2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end product_mod_17_eq_zero_l14_14110


namespace areas_of_shared_parts_l14_14027

-- Define the areas of the non-overlapping parts
def area_non_overlap_1 : ℝ := 68
def area_non_overlap_2 : ℝ := 110
def area_non_overlap_3 : ℝ := 87

-- Define the total area of each circle
def total_area : ℝ := area_non_overlap_2 + area_non_overlap_3 - area_non_overlap_1

-- Define the areas of the shared parts A and B
def area_shared_A : ℝ := total_area - area_non_overlap_2
def area_shared_B : ℝ := total_area - area_non_overlap_3

-- Prove the areas of the shared parts
theorem areas_of_shared_parts :
  area_shared_A = 19 ∧ area_shared_B = 42 :=
by
  sorry

end areas_of_shared_parts_l14_14027


namespace immigration_per_year_l14_14247

-- Definitions based on the initial conditions
def initial_population : ℕ := 100000
def birth_rate : ℕ := 60 -- this represents 60%
def duration_years : ℕ := 10
def emigration_per_year : ℕ := 2000
def final_population : ℕ := 165000

-- Theorem statement: The number of people that immigrated per year
theorem immigration_per_year (immigration_per_year : ℕ) :
  immigration_per_year = 2500 :=
  sorry

end immigration_per_year_l14_14247


namespace combination_8_choose_2_l14_14294

theorem combination_8_choose_2 : Nat.choose 8 2 = 28 := sorry

end combination_8_choose_2_l14_14294


namespace inverse_proportion_quadrants_l14_14486

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ y = k / x) →
  (∀ x : ℝ, x ≠ 0 → ( (x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0) ) ) :=
by
  sorry

end inverse_proportion_quadrants_l14_14486


namespace infinite_series_sum_l14_14454

theorem infinite_series_sum :
  ∑' n : ℕ, (n + 1) * (1 / 1950)^n = 3802500 / 3802601 :=
by
  sorry

end infinite_series_sum_l14_14454


namespace find_x_minus_y_l14_14372

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 :=
by
  have h3 : x^2 - y^2 = (x + y) * (x - y) := by sorry
  have h4 : (x + y) * (x - y) = 8 * (x - y) := by sorry
  have h5 : 16 = 8 * (x - y) := by sorry
  have h6 : 16 = 8 * (x - y) := by sorry
  have h7 : x - y = 2 := by sorry
  exact h7

end find_x_minus_y_l14_14372


namespace combined_rocket_height_l14_14205

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  sorry

end combined_rocket_height_l14_14205


namespace log_expression_l14_14002

variable (a : ℝ) (log3 : ℝ → ℝ)
axiom h_a : a = log3 2
axiom log3_8_eq : log3 8 = 3 * log3 2
axiom log3_6_eq : log3 6 = log3 2 + 1

theorem log_expression (log_def : log3 8 - 2 * log3 6 = a - 2) :
  log3 8 - 2 * log3 6 = a - 2 := by
  sorry

end log_expression_l14_14002


namespace walkway_time_stopped_l14_14572

noncomputable def effective_speed_with_walkway (v_p v_w : ℝ) : ℝ := v_p + v_w
noncomputable def effective_speed_against_walkway (v_p v_w : ℝ) : ℝ := v_p - v_w

theorem walkway_time_stopped (v_p v_w : ℝ) (h1 : effective_speed_with_walkway v_p v_w = 2)
                            (h2 : effective_speed_against_walkway v_p v_w = 2 / 3) :
    (200 / v_p) = 150 :=
by sorry

end walkway_time_stopped_l14_14572


namespace greatest_possible_third_side_l14_14081

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l14_14081


namespace green_block_weight_l14_14207

theorem green_block_weight (y g : ℝ) (h1 : y = 0.6) (h2 : y = g + 0.2) : g = 0.4 :=
by
  sorry

end green_block_weight_l14_14207


namespace intersection_of_M_and_N_l14_14358

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

-- The theorem to be proved
theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_of_M_and_N_l14_14358


namespace uniform_heights_l14_14980

theorem uniform_heights (varA varB : ℝ) (hA : varA = 0.56) (hB : varB = 2.1) : varA < varB := by
  rw [hA, hB]
  exact (by norm_num)

end uniform_heights_l14_14980


namespace paint_more_expensive_than_wallpaper_l14_14236

variable (x y z : ℝ)
variable (h : 4 * x + 4 * y = 7 * x + 2 * y + z)

theorem paint_more_expensive_than_wallpaper : y > x :=
by
  sorry

end paint_more_expensive_than_wallpaper_l14_14236


namespace point_on_xaxis_y_coord_zero_l14_14010

theorem point_on_xaxis_y_coord_zero (m : ℝ) (h : (3, m).snd = 0) : m = 0 :=
by 
  -- proof goes here
  sorry

end point_on_xaxis_y_coord_zero_l14_14010


namespace quadratic_factored_b_l14_14062

theorem quadratic_factored_b (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 30 = (m * x + n) * (p * x + q) ∧ m * p = 15 ∧ n * q = 30 ∧ m * q + n * p = b) ↔ b = 43 :=
by {
  sorry
}

end quadratic_factored_b_l14_14062


namespace alpha_in_third_quadrant_l14_14926

theorem alpha_in_third_quadrant (k : ℤ) (α : ℝ) :
  (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60 → 180 < α ∧ α < 240 :=
  sorry

end alpha_in_third_quadrant_l14_14926


namespace gcd_228_1995_l14_14756

theorem gcd_228_1995 : Int.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l14_14756


namespace find_ratio_l14_14476

variable {x y z : ℝ}

theorem find_ratio
  (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (2 * x + y - z) / (3 * x - 2 * y + z) = 5 / 6 := by
  sorry

end find_ratio_l14_14476


namespace adoption_cost_l14_14504

theorem adoption_cost :
  let cost_cat := 50
  let cost_adult_dog := 100
  let cost_puppy := 150
  let num_cats := 2
  let num_adult_dogs := 3
  let num_puppies := 2
  (num_cats * cost_cat + num_adult_dogs * cost_adult_dog + num_puppies * cost_puppy) = 700 :=
by
  sorry

end adoption_cost_l14_14504


namespace oranges_to_juice_l14_14729

theorem oranges_to_juice (oranges: ℕ) (juice: ℕ) (h: oranges = 18 ∧ juice = 27): 
  ∃ x, (juice / oranges) = (9 / x) ∧ x = 6 :=
by
  sorry

end oranges_to_juice_l14_14729


namespace quadratic_two_distinct_real_roots_l14_14843

theorem quadratic_two_distinct_real_roots:
  ∃ (α β : ℝ), α ≠ β ∧ (∀ x : ℝ, x * (x - 2) = x - 2 ↔ x = α ∨ x = β) :=
by
  sorry

end quadratic_two_distinct_real_roots_l14_14843


namespace aero_flight_tees_per_package_l14_14453

theorem aero_flight_tees_per_package {A : ℕ} :
  (∀ (num_people : ℕ), num_people = 4 → 20 * num_people ≤ A * 28 + 2 * 12) →
  A * 28 ≥ 56 →
  A = 2 :=
by
  intros h1 h2
  sorry

end aero_flight_tees_per_package_l14_14453


namespace imaginary_part_of_z_l14_14376

open Complex

theorem imaginary_part_of_z :
  ∃ z: ℂ, (3 - 4 * I) * z = abs (4 + 3 * I) ∧ z.im = 4 / 5 :=
by
  sorry

end imaginary_part_of_z_l14_14376


namespace inequality_solution_l14_14696

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l14_14696


namespace greatest_integer_third_side_l14_14088

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l14_14088


namespace find_y_l14_14170

theorem find_y (a b c x : ℝ) (p q r y: ℝ) (hx : x ≠ 1) 
  (h₁ : (Real.log a) / p = Real.log x) 
  (h₂ : (Real.log b) / q = Real.log x) 
  (h₃ : (Real.log c) / r = Real.log x)
  (h₄ : (b^3) / (a^2 * c) = x^y) : 
  y = 3 * q - 2 * p - r := 
by {
  sorry
}

end find_y_l14_14170


namespace expedition_ratios_l14_14670

theorem expedition_ratios (F : ℕ) (S : ℕ) (L : ℕ) (R : ℕ) 
  (h1 : F = 3) 
  (h2 : S = F + 2) 
  (h3 : F + S + L = 18) 
  (h4 : L = R * S) : 
  R = 2 := 
sorry

end expedition_ratios_l14_14670


namespace perimeter_ABFCDE_l14_14153

theorem perimeter_ABFCDE 
  (ABCD_perimeter : ℝ)
  (ABCD : ℝ)
  (triangle_BFC : ℝ -> ℝ)
  (translate_BFC : ℝ -> ℝ)
  (ABFCDE : ℝ -> ℝ -> ℝ)
  (h1 : ABCD_perimeter = 40)
  (h2 : ABCD = ABCD_perimeter / 4)
  (h3 : triangle_BFC ABCD = 10 * Real.sqrt 2)
  (h4 : translate_BFC (10 * Real.sqrt 2) = 10 * Real.sqrt 2)
  (h5 : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2)
  : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2 := 
by 
  sorry

end perimeter_ABFCDE_l14_14153


namespace intersection_is_correct_l14_14006

noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def B : Set ℝ := {x | x^2 - 5 * x - 6 < 0}

theorem intersection_is_correct : A ∩ B = {x | -1 < x ∧ x < 2} := 
by { sorry }

end intersection_is_correct_l14_14006


namespace passed_candidates_l14_14983

theorem passed_candidates (P F : ℕ) (h1 : P + F = 120) (h2 : 39 * P + 15 * F = 35 * 120) : P = 100 :=
by
  sorry

end passed_candidates_l14_14983


namespace interval_contains_integer_l14_14468

theorem interval_contains_integer (a : ℝ) : 
  (∃ n : ℤ, (3 * a < n) ∧ (n < 5 * a - 2)) ↔ (1.2 < a ∧ a < 4 / 3) ∨ (7 / 5 < a) :=
by sorry

end interval_contains_integer_l14_14468


namespace greatest_possible_third_side_l14_14078

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l14_14078


namespace cistern_filling_time_l14_14871

theorem cistern_filling_time :
  let rate_P := (1 : ℚ) / 12
  let rate_Q := (1 : ℚ) / 15
  let combined_rate := rate_P + rate_Q
  let time_combined := 6
  let filled_after_combined := combined_rate * time_combined
  let remaining_after_combined := 1 - filled_after_combined
  let time_Q := remaining_after_combined / rate_Q
  time_Q = 1.5 := sorry

end cistern_filling_time_l14_14871


namespace isosceles_trapezoid_ratio_l14_14296

theorem isosceles_trapezoid_ratio (a b d_E d_G : ℝ) (h1 : a > b)
  (h2 : (1/2) * b * d_G = 3) (h3 : (1/2) * a * d_E = 7)
  (h4 : (1/2) * (a + b) * (d_E + d_G) = 24) :
  (a / b) = 7 / 3 :=
sorry

end isosceles_trapezoid_ratio_l14_14296


namespace set_intersection_l14_14351

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l14_14351


namespace sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l14_14054

theorem sqrt_99_eq_9801 : 99^2 = 9801 := by
  sorry

theorem expr_2000_1999_2001_eq_1 : 2000^2 - 1999 * 2001 = 1 := by
  sorry

end sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l14_14054


namespace boxes_needed_l14_14825

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end boxes_needed_l14_14825


namespace lateral_surface_area_of_frustum_l14_14473

theorem lateral_surface_area_of_frustum (slant_height : ℝ) (ratio : ℕ × ℕ) (central_angle_deg : ℝ)
  (h_slant_height : slant_height = 10) 
  (h_ratio : ratio = (2, 5)) 
  (h_central_angle_deg : central_angle_deg = 216) : 
  ∃ (area : ℝ), area = (252 * Real.pi / 5) := 
by 
  sorry

end lateral_surface_area_of_frustum_l14_14473


namespace angle_in_fourth_quadrant_l14_14654

variable (α : ℝ)

def is_in_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_in_fourth_quadrant (θ : ℝ) : Prop := 270 < θ ∧ θ < 360

theorem angle_in_fourth_quadrant (h : is_in_first_quadrant α) : is_in_fourth_quadrant (360 - α) := sorry

end angle_in_fourth_quadrant_l14_14654


namespace sum_first_20_odds_is_400_l14_14717

-- Define the n-th odd positive integer
def odd_integer (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd positive integers as a function
def sum_first_n_odds (n : ℕ) : ℕ := (n * (2 * n + 1)) / 2

-- Theorem statement: sum of the first 20 odd positive integers is 400
theorem sum_first_20_odds_is_400 : sum_first_n_odds 20 = 400 := 
  sorry

end sum_first_20_odds_is_400_l14_14717


namespace finite_pos_int_set_condition_l14_14312

theorem finite_pos_int_set_condition (X : Finset ℕ) 
  (hX : ∀ a ∈ X, 0 < a) 
  (h2 : 2 ≤ X.card) 
  (hcond : ∀ {a b : ℕ}, a ∈ X → b ∈ X → a > b → b^2 / (a - b) ∈ X) :
  ∃ a : ℕ, X = {a, 2 * a} :=
by
  sorry

end finite_pos_int_set_condition_l14_14312


namespace g_at_minus_six_l14_14963

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end g_at_minus_six_l14_14963


namespace greatest_third_side_l14_14105

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l14_14105


namespace airplane_distance_difference_l14_14888

variable (a : ℝ)

theorem airplane_distance_difference :
  let wind_speed := 20
  (4 * a) - (3 * (a - wind_speed)) = a + 60 := by
  sorry

end airplane_distance_difference_l14_14888


namespace intersection_M_N_l14_14333

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14333


namespace expand_expression_l14_14901

variable (y : ℤ)

theorem expand_expression : 12 * (3 * y - 4) = 36 * y - 48 := 
by
  sorry

end expand_expression_l14_14901


namespace triangle_area_45_45_90_l14_14744

/--
A right triangle has one angle of 45 degrees, and its hypotenuse measures 10√2 inches.
Prove that the area of the triangle is 50 square inches.
-/
theorem triangle_area_45_45_90 {x : ℝ} (h1 : 0 < x) (h2 : x * Real.sqrt 2 = 10 * Real.sqrt 2) : 
  (1 / 2) * x * x = 50 :=
sorry

end triangle_area_45_45_90_l14_14744


namespace sum_of_products_eq_131_l14_14417

theorem sum_of_products_eq_131 (a b c : ℝ) 
    (h1 : a^2 + b^2 + c^2 = 222)
    (h2 : a + b + c = 22) : 
    a * b + b * c + c * a = 131 :=
by
  sorry

end sum_of_products_eq_131_l14_14417


namespace infections_first_wave_l14_14412

theorem infections_first_wave (x : ℕ)
  (h1 : 4 * x * 14 = 21000) : x = 375 :=
  sorry

end infections_first_wave_l14_14412


namespace n_squared_divisible_by_144_l14_14375

-- Definitions based on the conditions
variables (n k : ℕ)
def is_positive (n : ℕ) : Prop := n > 0
def largest_divisor_of_n_is_twelve (n : ℕ) : Prop := ∃ k, n = 12 * k
def divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k

theorem n_squared_divisible_by_144
  (h1 : is_positive n)
  (h2 : largest_divisor_of_n_is_twelve n) :
  divisible_by (n * n) 144 :=
sorry

end n_squared_divisible_by_144_l14_14375


namespace sandys_average_price_l14_14053

noncomputable def average_price_per_book (priceA : ℝ) (discountA : ℝ) (booksA : ℕ) (priceB : ℝ) (discountB : ℝ) (booksB : ℕ) (conversion_rate : ℝ) : ℝ :=
  let costA := priceA / (1 - discountA)
  let priceB_in_usd := priceB / conversion_rate
  let costB := priceB_in_usd / (1 - discountB)
  let total_cost := costA + costB
  let total_books := booksA + booksB
  total_cost / total_books

theorem sandys_average_price :
  average_price_per_book 1380 0.15 65 900 0.10 55 0.85 = 23.33 :=
by
  sorry

end sandys_average_price_l14_14053


namespace relationship_between_T_and_S_l14_14911

variable (a b : ℝ)

def T : ℝ := a + 2 * b
def S : ℝ := a + b^2 + 1

theorem relationship_between_T_and_S : T a b ≤ S a b := by
  sorry

end relationship_between_T_and_S_l14_14911


namespace cost_of_each_croissant_l14_14403

theorem cost_of_each_croissant 
  (quiches_price : ℝ) (num_quiches : ℕ) (each_quiche_cost : ℝ)
  (buttermilk_biscuits_price : ℝ) (num_biscuits : ℕ) (each_biscuit_cost : ℝ)
  (total_cost_with_discount : ℝ) (discount_rate : ℝ)
  (num_croissants : ℕ) (croissant_price : ℝ) :
  quiches_price = num_quiches * each_quiche_cost →
  each_quiche_cost = 15 →
  num_quiches = 2 →
  buttermilk_biscuits_price = num_biscuits * each_biscuit_cost →
  each_biscuit_cost = 2 →
  num_biscuits = 6 →
  discount_rate = 0.10 →
  (quiches_price + buttermilk_biscuits_price + (num_croissants * croissant_price)) * (1 - discount_rate) = total_cost_with_discount →
  total_cost_with_discount = 54 →
  num_croissants = 6 →
  croissant_price = 3 :=
sorry

end cost_of_each_croissant_l14_14403


namespace necessary_but_not_sufficient_condition_l14_14934

theorem necessary_but_not_sufficient_condition (x y : ℝ) : 
  ((x > 1) ∨ (y > 2)) → (x + y > 3) ∧ ¬((x > 1) ∨ (y > 2) ↔ (x + y > 3)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l14_14934


namespace angle_x_value_l14_14503

theorem angle_x_value 
  (AB CD : Prop) -- AB and CD are straight lines
  (angle_AXB angle_AXZ angle_BXY angle_CYX : ℝ) -- Given angles in the problem
  (h1 : AB) (h2 : CD)
  (h3 : angle_AXB = 180)
  (h4 : angle_AXZ = 60)
  (h5 : angle_BXY = 50)
  (h6 : angle_CYX = 120) : 
  ∃ x : ℝ, x = 50 := by
sorry

end angle_x_value_l14_14503


namespace intersection_M_N_l14_14340

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l14_14340


namespace k_is_perfect_square_l14_14272

theorem k_is_perfect_square (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (k : ℕ)
  (h_k : k = (m + n)^2 / (4 * m * (m - n)^2 + 4)) 
  (h_int_k : k * (4 * m * (m - n)^2 + 4) = (m + n)^2) :
  ∃ x : ℕ, k = x^2 := 
sorry

end k_is_perfect_square_l14_14272


namespace discount_on_item_l14_14402

noncomputable def discount_percentage : ℝ := 20
variable (total_cart_value original_price final_amount : ℝ)
variable (coupon_discount : ℝ)

axiom cart_value : total_cart_value = 54
axiom item_price : original_price = 20
axiom coupon : coupon_discount = 0.10
axiom final_price : final_amount = 45

theorem discount_on_item :
  ∃ x : ℝ, (total_cart_value - (x / 100) * original_price) * (1 - coupon_discount) = final_amount ∧ x = discount_percentage :=
by
  have eq1 := cart_value
  have eq2 := item_price
  have eq3 := coupon
  have eq4 := final_price
  sorry

end discount_on_item_l14_14402


namespace intersection_M_N_l14_14341

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l14_14341


namespace inequality_solution_set_l14_14547

theorem inequality_solution_set : 
  { x : ℝ | (x + 1) / (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry 

end inequality_solution_set_l14_14547


namespace speed_of_train_in_km_per_hr_l14_14136

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l14_14136


namespace rectangle_area_l14_14581

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l14_14581


namespace find_k_l14_14937

theorem find_k (k : ℤ) (h1 : ∃(a b c : ℤ), a = (36 + k) ∧ b = (300 + k) ∧ c = (596 + k) ∧ (∃ d, 
  (a = d^2) ∧ (b = (d + 1)^2) ∧ (c = (d + 2)^2)) ) : k = 925 := by
  sorry

end find_k_l14_14937


namespace mikes_original_speed_l14_14045

variable (x : ℕ) -- x is the original typing speed of Mike

-- Condition: After the accident, Mike's typing speed is 20 words per minute less
def currentSpeed : ℕ := x - 20

-- Condition: It takes Mike 18 minutes to type 810 words at his reduced speed
def typingTimeCondition : Prop := 18 * currentSpeed x = 810

-- Proof goal: Prove that Mike's original typing speed is 65 words per minute
theorem mikes_original_speed (h : typingTimeCondition x) : x = 65 := 
sorry

end mikes_original_speed_l14_14045


namespace jerry_liters_of_mustard_oil_l14_14034

-- Definitions
def cost_per_liter_mustard_oil : ℕ := 13
def cost_per_pound_penne_pasta : ℕ := 4
def cost_per_pound_pasta_sauce : ℕ := 5
def total_money_jerry_had : ℕ := 50
def money_left_with_jerry : ℕ := 7
def pounds_of_penne_pasta : ℕ := 3
def pounds_of_pasta_sauce : ℕ := 1

-- Our goal is to calculate how many liters of mustard oil Jerry bought
theorem jerry_liters_of_mustard_oil : ℕ :=
  let cost_of_penne_pasta := pounds_of_penne_pasta * cost_per_pound_penne_pasta
  let cost_of_pasta_sauce := pounds_of_pasta_sauce * cost_per_pound_pasta_sauce
  let total_spent := total_money_jerry_had - money_left_with_jerry
  let spent_on_pasta_and_sauce := cost_of_penne_pasta + cost_of_pasta_sauce
  let spent_on_mustard_oil := total_spent - spent_on_pasta_and_sauce
  spent_on_mustard_oil / cost_per_liter_mustard_oil

example : jerry_liters_of_mustard_oil = 2 := by
  unfold jerry_liters_of_mustard_oil
  simp
  sorry

end jerry_liters_of_mustard_oil_l14_14034


namespace find_g_neg_6_l14_14959

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end find_g_neg_6_l14_14959


namespace factorization_correct_l14_14892

theorem factorization_correct (x : ℝ) :
  16 * x ^ 2 + 8 * x - 24 = 8 * (2 * x ^ 2 + x - 3) ∧ (2 * x ^ 2 + x - 3) = (2 * x + 3) * (x - 1) :=
by
  sorry

end factorization_correct_l14_14892


namespace Misha_earnings_needed_l14_14686

-- Define the conditions and the goal in Lean 4
def Misha_current_dollars : ℕ := 34
def Misha_target_dollars : ℕ := 47

theorem Misha_earnings_needed : Misha_target_dollars - Misha_current_dollars = 13 := by
  sorry

end Misha_earnings_needed_l14_14686


namespace find_a_purely_imaginary_z1_z2_l14_14768

noncomputable def z1 (a : ℝ) : ℂ := ⟨a^2 - 3, a + 5⟩
noncomputable def z2 (a : ℝ) : ℂ := ⟨a - 1, a^2 + 2 * a - 1⟩

theorem find_a_purely_imaginary_z1_z2 (a : ℝ)
    (h_imaginary : ∃ b : ℝ, z2 a - z1 a = ⟨0, b⟩) : 
    a = -1 :=
sorry

end find_a_purely_imaginary_z1_z2_l14_14768


namespace derivative_at_pi_over_six_l14_14484

-- Define the function f(x) = cos(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- State the theorem: the derivative of f at π/6 is -1/2
theorem derivative_at_pi_over_six : deriv f (Real.pi / 6) = -1 / 2 :=
by sorry

end derivative_at_pi_over_six_l14_14484


namespace greatest_third_side_l14_14084

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l14_14084


namespace Wolfgang_marble_count_l14_14711

theorem Wolfgang_marble_count
  (W L M : ℝ)
  (hL : L = 5/4 * W)
  (hM : M = 2/3 * (W + L))
  (hTotal : W + L + M = 60) :
  W = 16 :=
by {
  sorry
}

end Wolfgang_marble_count_l14_14711


namespace square_flag_side_length_side_length_of_square_flags_is_4_l14_14895

theorem square_flag_side_length 
  (total_fabric : ℕ)
  (fabric_left : ℕ)
  (num_square_flags : ℕ)
  (num_wide_flags : ℕ)
  (num_tall_flags : ℕ)
  (wide_flag_length : ℕ)
  (wide_flag_width : ℕ)
  (tall_flag_length : ℕ)
  (tall_flag_width : ℕ)
  (fabric_used_on_wide_and_tall_flags : ℕ)
  (fabric_used_on_all_flags : ℕ)
  (fabric_used_on_square_flags : ℕ)
  (square_flag_area : ℕ)
  (side_length : ℕ) : Prop :=
  total_fabric = 1000 ∧
  fabric_left = 294 ∧
  num_square_flags = 16 ∧
  num_wide_flags = 20 ∧
  num_tall_flags = 10 ∧
  wide_flag_length = 5 ∧
  wide_flag_width = 3 ∧
  tall_flag_length = 5 ∧
  tall_flag_width = 3 ∧
  fabric_used_on_wide_and_tall_flags = (num_wide_flags + num_tall_flags) * (wide_flag_length * wide_flag_width) ∧
  fabric_used_on_all_flags = total_fabric - fabric_left ∧
  fabric_used_on_square_flags = fabric_used_on_all_flags - fabric_used_on_wide_and_tall_flags ∧
  square_flag_area = fabric_used_on_square_flags / num_square_flags ∧
  side_length = Int.sqrt square_flag_area ∧
  side_length = 4

theorem side_length_of_square_flags_is_4 : 
  square_flag_side_length 1000 294 16 20 10 5 3 5 3 450 706 256 16 4 :=
  by
    sorry

end square_flag_side_length_side_length_of_square_flags_is_4_l14_14895


namespace slightly_used_crayons_correct_l14_14849

def total_crayons : ℕ := 120
def new_crayons : ℕ := total_crayons / 3
def broken_crayons : ℕ := (total_crayons * 20) / 100
def slightly_used_crayons : ℕ := total_crayons - new_crayons - broken_crayons

theorem slightly_used_crayons_correct : slightly_used_crayons = 56 := sorry

end slightly_used_crayons_correct_l14_14849


namespace coprime_divisors_property_l14_14626

theorem coprime_divisors_property (n : ℕ) 
  (h : ∀ a b : ℕ, a ∣ n → b ∣ n → gcd a b = 1 → (a + b - 1) ∣ n) : 
  (∃ k : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n = p ^ k) ∨ (n = 12) :=
sorry

end coprime_divisors_property_l14_14626


namespace min_value_of_expression_l14_14480

theorem min_value_of_expression {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (1 / a) + (2 / b) >= 8 :=
by
  sorry

end min_value_of_expression_l14_14480


namespace train_speed_is_72_km_per_hr_l14_14132

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l14_14132


namespace count_integers_between_cubes_l14_14652

theorem count_integers_between_cubes (a b : ℝ) (h1 : a = 10.5) (h2 : b = 10.6) : 
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  (last_integer - first_integer + 1) = 33 :=
by
  -- Definitions for clarity
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  
  -- Skipping the proof
  sorry

end count_integers_between_cubes_l14_14652


namespace midpoint_C_is_either_l14_14058

def A : ℝ := -7
def dist_AB : ℝ := 5

theorem midpoint_C_is_either (C : ℝ) (h : C = (A + (A + dist_AB / 2)) / 2 ∨ C = (A + (A - dist_AB / 2)) / 2) : 
  C = -9 / 2 ∨ C = -19 / 2 := 
sorry

end midpoint_C_is_either_l14_14058


namespace parabola_through_P_l14_14064

-- Define the point P
def P : ℝ × ℝ := (4, -2)

-- Define a condition function for equations y^2 = a*x
def satisfies_y_eq_ax (a : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ y^2 = a * x

-- Define a condition function for equations x^2 = b*y
def satisfies_x_eq_by (b : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ x^2 = b * y

-- Lean's theorem statement
theorem parabola_through_P : satisfies_y_eq_ax 1 ∨ satisfies_x_eq_by (-8) :=
sorry

end parabola_through_P_l14_14064


namespace max_identifiable_cards_2013_l14_14229

-- Define the number of cards
def num_cards : ℕ := 2013

-- Define the function that determines the maximum t for which the numbers can be found
def max_identifiable_cards (cards : ℕ) (select : ℕ) : ℕ :=
  if (cards = 2013) ∧ (select = 10) then 1986 else 0

-- The theorem to prove the property
theorem max_identifiable_cards_2013 :
  max_identifiable_cards 2013 10 = 1986 :=
sorry

end max_identifiable_cards_2013_l14_14229


namespace stratified_sampling_second_grade_survey_l14_14876

theorem stratified_sampling_second_grade_survey :
  let total_students := 1500
  let ratio_first := 4
  let ratio_second := 5
  let ratio_third := 6
  let survey_total := 150
  let total_ratio_parts := ratio_first + ratio_second + ratio_third
  let fraction_second := ratio_second.toReal / total_ratio_parts.toReal
  survey_total * fraction_second = 50 := 
by
  sorry

end stratified_sampling_second_grade_survey_l14_14876


namespace highest_value_of_a_l14_14630

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def highest_a : Nat :=
  7

theorem highest_value_of_a (a : Nat) 
  (last_three_digits := a * 100 + 53)
  (number := 4 * 10^8 + 3 * 10^7 + 7 * 10^6 + 5 * 10^5 + 2 * 10^4 + a * 10^3 + 5 * 10^2 + 3 * 10^1 + 9) :
  (∃ a, last_three_digits % 8 = 0 ∧ sum_of_digits number % 9 = 0 ∧ number % 12 = 0 ∧ a <= 9) → a = highest_a :=
by
  intros
  sorry

end highest_value_of_a_l14_14630


namespace area_of_triangle_pqr_l14_14949

noncomputable def area_of_triangle (P Q R : ℝ) : ℝ :=
  let PQ := P + Q
  let PR := P + R
  let QR := Q + R
  if PQ^2 = PR^2 + QR^2 then
    1 / 2 * PR * QR
  else
    0

theorem area_of_triangle_pqr : 
  area_of_triangle 3 2 1 = 6 :=
by
  simp [area_of_triangle]
  sorry

end area_of_triangle_pqr_l14_14949


namespace pq_plus_sum_eq_20_l14_14931

theorem pq_plus_sum_eq_20 
  (p q : ℕ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (hpl : p < 30) 
  (hql : q < 30) 
  (heq : p + q + p * q = 119) : 
  p + q = 20 :=
sorry

end pq_plus_sum_eq_20_l14_14931


namespace quadratic_has_only_positive_roots_l14_14660

theorem quadratic_has_only_positive_roots (m : ℝ) :
  (∀ (x : ℝ), x^2 + (m + 2) * x + (m + 5) = 0 → x > 0) →
  -5 < m ∧ m ≤ -4 :=
by 
  -- added sorry to skip the proof.
  sorry

end quadratic_has_only_positive_roots_l14_14660


namespace prove_statement_II_l14_14202

variable (digit : ℕ)

def statement_I : Prop := (digit = 2)
def statement_II : Prop := (digit ≠ 3)
def statement_III : Prop := (digit = 5)
def statement_IV : Prop := (digit ≠ 6)

/- The main proposition that three statements are true and one is false. -/
def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop :=
  (s1 ∧ s2 ∧ s3 ∧ ¬s4) ∨ (s1 ∧ s2 ∧ ¬s3 ∧ s4) ∨ 
  (s1 ∧ ¬s2 ∧ s3 ∧ s4) ∨ (¬s1 ∧ s2 ∧ s3 ∧ s4)

theorem prove_statement_II : 
  (three_true_one_false (statement_I digit) (statement_II digit) (statement_III digit) (statement_IV digit)) → 
  statement_II digit :=
sorry

end prove_statement_II_l14_14202


namespace solve_for_x_l14_14371

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = (9 / (x / 3))^2) : x = 23.43 :=
by {
  sorry
}

end solve_for_x_l14_14371


namespace age_ratio_l14_14546

theorem age_ratio (R D : ℕ) (h1 : D = 15) (h2 : R + 6 = 26) : R / D = 4 / 3 := by
  sorry

end age_ratio_l14_14546


namespace bride_groom_couples_sum_l14_14878

def wedding_reception (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) : Prop :=
  total_guests - friends = couples_guests

theorem bride_groom_couples_sum (B G : ℕ) (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) 
  (h1 : total_guests = 180) (h2 : friends = 100) (h3 : wedding_reception total_guests friends couples_guests) 
  (h4 : couples_guests = 80) : B + G = 40 := 
  by
  sorry

end bride_groom_couples_sum_l14_14878


namespace limit_proof_l14_14753

open Real

-- Define the conditions
axiom sin_6x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |sin (6 * x) / (6 * x) - 1| < ε
axiom arctg_2x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |arctan (2 * x) / (2 * x) - 1| < ε

-- State the limit proof problem
theorem limit_proof :
  (∃ ε > 0, ∀ x : ℝ, |x| < ε → x ≠ 0 →
  |(x * sin (6 * x)) / (arctan (2 * x)) ^ 2 - (3 / 2)| < ε) :=
sorry

end limit_proof_l14_14753


namespace parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l14_14488

section vector

variables {k : ℝ}
def a : ℝ × ℝ := (6, 2)
def b : ℝ × ℝ := (-2, k)

-- Parallel condition
theorem parallel_vectors : 
  (∀ c : ℝ, (6, 2) = -2 * (c * k, c)) → k = -2 / 3 :=
by 
  sorry

-- Perpendicular condition
theorem perpendicular_vectors : 
  6 * (-2) + 2 * k = 0 → k = 6 :=
by 
  sorry

-- Obtuse angle condition
theorem obtuse_angle_vectors : 
  6 * (-2) + 2 * k < 0 ∧ k ≠ -2 / 3 → k < 6 ∧ k ≠ -2 / 3 :=
by 
  sorry

end vector

end parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l14_14488


namespace set_intersection_l14_14349

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l14_14349


namespace non_gray_squares_count_l14_14439

-- Define the dimensions of the grid strip
def width : ℕ := 5
def length : ℕ := 250

-- Define the repeating pattern dimensions and color distribution
def pattern_columns : ℕ := 4
def pattern_non_gray_squares : ℕ := 13
def pattern_total_squares : ℕ := width * pattern_columns

-- Define the number of complete patterns in the grid strip
def complete_patterns : ℕ := length / pattern_columns

-- Define the number of additional columns and additional non-gray squares
def additional_columns : ℕ := length % pattern_columns
def additional_non_gray_squares : ℕ := 6

-- Calculate the total non-gray squares
def total_non_gray_squares : ℕ := complete_patterns * pattern_non_gray_squares + additional_non_gray_squares

theorem non_gray_squares_count : total_non_gray_squares = 812 := by
  sorry

end non_gray_squares_count_l14_14439


namespace flat_rate_65_l14_14440

noncomputable def flat_rate_first_night (f n : ℝ) : Prop := 
  (f + 4 * n = 245) ∧ (f + 9 * n = 470)

theorem flat_rate_65 :
  ∃ (f n : ℝ), flat_rate_first_night f n ∧ f = 65 := 
by
  sorry

end flat_rate_65_l14_14440


namespace rectangular_field_area_l14_14587

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l14_14587


namespace y_increase_by_20_l14_14666

-- Define the conditions
def relationship (Δx Δy : ℕ) : Prop :=
  Δy = (11 * Δx) / 5

-- The proof problem statement
theorem y_increase_by_20 : relationship 5 11 → relationship 20 44 :=
by
  intros h
  sorry

end y_increase_by_20_l14_14666


namespace probability_of_winning_prize_l14_14619

def total_balls : ℕ := 10
def winning_balls : Finset ℕ := {6, 7, 8, 9, 10}

theorem probability_of_winning_prize : 
  ((winning_balls.card : ℚ) / (total_balls : ℚ)) = 1 / 2 := sorry

end probability_of_winning_prize_l14_14619


namespace min_abs_ab_l14_14802

theorem min_abs_ab (a b : ℤ) (h : 1009 * a + 2 * b = 1) : ∃ k : ℤ, |a * b| = 504 :=
by
  sorry

end min_abs_ab_l14_14802


namespace colten_chickens_l14_14527

variable (Colten Skylar Quentin : ℕ)

def chicken_problem_conditions :=
  (Skylar = 3 * Colten - 4) ∧
  (Quentin = 6 * Skylar + 17) ∧
  (Colten + Skylar + Quentin = 383)

theorem colten_chickens (h : chicken_problem_conditions Colten Skylar Quentin) : Colten = 37 :=
sorry

end colten_chickens_l14_14527


namespace sonya_falls_count_l14_14818

/-- The number of times Sonya fell down given the conditions. -/
theorem sonya_falls_count : 
  let steven_falls := 3 in
  let stephanie_falls := steven_falls + 13 in
  let sonya_falls := (stephanie_falls / 2) - 2 in
  sonya_falls = 6 := 
by
  sorry

end sonya_falls_count_l14_14818


namespace vec_op_not_comm_l14_14459

open Real

-- Define the operation ⊙
def vec_op (a b: ℝ × ℝ) : ℝ :=
  (a.1 * b.2) - (a.2 * b.1)

-- Define a predicate to check if two vectors are collinear
def collinear (a b: ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Define the proof theorem
theorem vec_op_not_comm (a b: ℝ × ℝ) : vec_op a b ≠ vec_op b a :=
by
  -- The contents of the proof will go here. Insert 'sorry' to skip.
  sorry

end vec_op_not_comm_l14_14459


namespace mean_of_y_and_18_is_neg1_l14_14410

theorem mean_of_y_and_18_is_neg1 (y : ℤ) : 
  ((4 + 6 + 10 + 14) / 4) = ((y + 18) / 2) → y = -1 := 
by 
  -- Placeholder for the proof
  sorry

end mean_of_y_and_18_is_neg1_l14_14410


namespace minimum_x_value_l14_14570

theorem minimum_x_value
  (sales_jan_may june_sales x : ℝ)
  (h_sales_jan_may : sales_jan_may = 38.6)
  (h_june_sales : june_sales = 5)
  (h_total_sales_condition : sales_jan_may + june_sales + 2 * june_sales * (1 + x / 100) + 2 * june_sales * (1 + x / 100)^2 ≥ 70) :
  x = 20 := by
  sorry

end minimum_x_value_l14_14570


namespace ducks_cows_problem_l14_14667

theorem ducks_cows_problem (D C : ℕ) (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end ducks_cows_problem_l14_14667


namespace largest_sum_of_base8_digits_l14_14927

theorem largest_sum_of_base8_digits (a b c y : ℕ) (h1 : a < 8) (h2 : b < 8) (h3 : c < 8) (h4 : 0 < y ∧ y ≤ 16) (h5 : (a * 64 + b * 8 + c) * y = 512) :
  a + b + c ≤ 5 :=
sorry

end largest_sum_of_base8_digits_l14_14927


namespace expression_equals_16_l14_14933

open Real

theorem expression_equals_16 (x : ℝ) :
  (x + 1) ^ 2 + 2 * (x + 1) * (3 - x) + (3 - x) ^ 2 = 16 :=
sorry

end expression_equals_16_l14_14933


namespace gear_B_turns_l14_14562

theorem gear_B_turns (teeth_A teeth_B turns_A: ℕ) (h₁: teeth_A = 6) (h₂: teeth_B = 8) (h₃: turns_A = 12) :
(turn_A * teeth_A) / teeth_B = 9 :=
by  sorry

end gear_B_turns_l14_14562


namespace convert_neg_300_deg_to_rad_l14_14615

theorem convert_neg_300_deg_to_rad :
  -300 * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end convert_neg_300_deg_to_rad_l14_14615


namespace question_inequality_l14_14663

theorem question_inequality (m : ℝ) :
  (∀ x : ℝ, ¬ (m * x ^ 2 - m * x - 1 ≥ 0)) ↔ (-4 < m ∧ m ≤ 0) :=
sorry

end question_inequality_l14_14663


namespace statement2_true_l14_14283

def digit : ℕ := sorry

def statement1 : Prop := digit = 2
def statement2 : Prop := digit ≠ 3
def statement3 : Prop := digit = 5
def statement4 : Prop := digit ≠ 6

def condition : Prop := (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (¬ statement1 ∨ ¬ statement2 ∨ ¬ statement3 ∨ ¬ statement4)

theorem statement2_true (h : condition) : statement2 :=
sorry

end statement2_true_l14_14283


namespace find_roots_range_l14_14796

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem find_roots_range 
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hx : -1 < -1/2 ∧ -1/2 < 0 ∧ 0 < 1/2 ∧ 1/2 < 1 ∧ 1 < 3/2 ∧ 3/2 < 2 ∧ 2 < 5/2 ∧ 5/2 < 3)
  (hy : ∀ {x : ℝ}, x = -1 → quadratic_function a b c x = -2 ∧
                   x = -1/2 → quadratic_function a b c x = -1/4 ∧
                   x = 0 → quadratic_function a b c x = 1 ∧
                   x = 1/2 → quadratic_function a b c x = 7/4 ∧
                   x = 1 → quadratic_function a b c x = 2 ∧
                   x = 3/2 → quadratic_function a b c x = 7/4 ∧
                   x = 2 → quadratic_function a b c x = 1 ∧
                   x = 5/2 → quadratic_function a b c x = -1/4 ∧
                   x = 3 → quadratic_function a b c x = -2) :
  ∃ x1 x2 : ℝ, -1/2 < x1 ∧ x1 < 0 ∧ 2 < x2 ∧ x2 < 5/2 ∧ quadratic_function a b c x1 = 0 ∧ quadratic_function a b c x2 = 0 :=
by sorry

end find_roots_range_l14_14796


namespace distance_after_rest_l14_14425

-- Define the conditions
def distance_before_rest := 0.75
def total_distance := 1.0

-- State the theorem
theorem distance_after_rest :
  total_distance - distance_before_rest = 0.25 :=
by sorry

end distance_after_rest_l14_14425


namespace age_difference_l14_14944

-- defining the conditions
variable (A B : ℕ)
variable (h1 : B = 35)
variable (h2 : A + 10 = 2 * (B - 10))

-- the proof statement
theorem age_difference : A - B = 5 :=
by
  sorry

end age_difference_l14_14944


namespace joan_kittens_total_l14_14387

-- Definition of the initial conditions
def joan_original_kittens : ℕ := 8
def neighbor_original_kittens : ℕ := 6
def joan_gave_away : ℕ := 2
def neighbor_gave_away : ℕ := 4
def joan_adopted_from_neighbor : ℕ := 3

-- The final number of kittens Joan has
def joan_final_kittens : ℕ :=
  let joan_remaining := joan_original_kittens - joan_gave_away
  let neighbor_remaining := neighbor_original_kittens - neighbor_gave_away
  let adopted := min joan_adopted_from_neighbor neighbor_remaining
  joan_remaining + adopted

theorem joan_kittens_total : joan_final_kittens = 8 := 
by 
  -- Lean proof would go here, but adding sorry for now
  sorry

end joan_kittens_total_l14_14387


namespace h_k_minus3_eq_l14_14928

def h (x : ℝ) : ℝ := 4 - Real.sqrt x
def k (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem h_k_minus3_eq : h (k (-3)) = 4 - 3 * Real.sqrt 2 := 
by 
  sorry

end h_k_minus3_eq_l14_14928


namespace cookies_remaining_in_jar_l14_14041

-- Definition of the conditions
variable (initial_cookies : Nat)

def cookies_taken_by_Lou_Senior := 3 + 1
def cookies_taken_by_Louie_Junior := 7
def total_cookies_taken := cookies_taken_by_Lou_Senior + cookies_taken_by_Louie_Junior

-- Debra's assumption and the proof goal
theorem cookies_remaining_in_jar (half_cookies_removed : total_cookies_taken = initial_cookies / 2) : 
  initial_cookies - total_cookies_taken = 11 := by
  sorry

end cookies_remaining_in_jar_l14_14041


namespace quotient_of_N_div_3_l14_14248

-- Define the number N
def N : ℕ := 7 * 12 + 4

-- Statement we need to prove
theorem quotient_of_N_div_3 : N / 3 = 29 :=
by
  sorry

end quotient_of_N_div_3_l14_14248


namespace mod_exp_value_l14_14536

theorem mod_exp_value (m : ℕ) (h1: 0 ≤ m) (h2: m < 9) (h3: 14^4 ≡ m [MOD 9]) : m = 5 :=
by
  sorry

end mod_exp_value_l14_14536


namespace cost_of_fencing_per_meter_l14_14997

theorem cost_of_fencing_per_meter (x : ℝ) (length width : ℝ) (area : ℝ) (total_cost : ℝ) :
  length = 3 * x ∧ width = 2 * x ∧ area = 3750 ∧ area = length * width ∧ total_cost = 125 →
  (total_cost / (2 * (length + width)) = 0.5) :=
by
  sorry

end cost_of_fencing_per_meter_l14_14997


namespace min_C2_D2_at_36_l14_14675

noncomputable def min_value_C2_D2 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : ℝ :=
  let C := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
  let D := (Real.sqrt (x + 1) + Real.sqrt (y + 2) + Real.sqrt (z + 3))
  C^2 - D^2

theorem min_C2_D2_at_36 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : 
  min_value_C2_D2 x y z hx hy hz = 36 :=
sorry

end min_C2_D2_at_36_l14_14675


namespace fifth_graders_more_than_eighth_graders_l14_14538

theorem fifth_graders_more_than_eighth_graders 
  (cost : ℕ) 
  (h_cost : cost > 0) 
  (h_div_234 : 234 % cost = 0) 
  (h_div_312 : 312 % cost = 0) 
  (h_40_fifth_graders : 40 > 0) : 
  (312 / cost) - (234 / cost) = 6 := 
by 
  sorry

end fifth_graders_more_than_eighth_graders_l14_14538


namespace boxes_produced_by_machine_A_in_10_minutes_l14_14864

-- Define the variables and constants involved
variables {A : ℕ} -- number of boxes machine A produces in 10 minutes

-- Define the condition that machine B produces 4*A boxes in 10 minutes
def boxes_produced_by_machine_B_in_10_minutes := 4 * A

-- Define the combined production working together for 20 minutes
def combined_production_in_20_minutes := 10 * A

-- Statement to prove that machine A produces A boxes in 10 minutes
theorem boxes_produced_by_machine_A_in_10_minutes :
  ∀ (boxes_produced_by_machine_B_in_10_minutes : ℕ) (combined_production_in_20_minutes : ℕ),
    boxes_produced_by_machine_B_in_10_minutes = 4 * A →
    combined_production_in_20_minutes = 10 * A →
    A = A :=
by
  intros _ _ hB hC
  sorry

end boxes_produced_by_machine_A_in_10_minutes_l14_14864


namespace mica_should_have_28_26_euros_l14_14044

namespace GroceryShopping

def pasta_cost : ℝ := 3 * 1.70
def ground_beef_cost : ℝ := 0.5 * 8.20
def pasta_sauce_base_cost : ℝ := 3 * 2.30
def pasta_sauce_discount : ℝ := pasta_sauce_base_cost * 0.10
def pasta_sauce_discounted_cost : ℝ := pasta_sauce_base_cost - pasta_sauce_discount
def quesadillas_cost : ℝ := 11.50

def total_cost_before_vat : ℝ :=
  pasta_cost + ground_beef_cost + pasta_sauce_discounted_cost + quesadillas_cost

def vat : ℝ := total_cost_before_vat * 0.05

def total_cost_including_vat : ℝ := total_cost_before_vat + vat

theorem mica_should_have_28_26_euros :
  total_cost_including_vat = 28.26 := by
  -- This is the statement without the proof. 
  sorry

end GroceryShopping

end mica_should_have_28_26_euros_l14_14044


namespace projection_vector_ratio_l14_14409

open Matrix

theorem projection_vector_ratio :
  let M := ![
    ![(3 : ℚ) / 17, -8 / 17],
    ![-8 / 17, 15 / 17]
  ]
  ∃ (x y : ℚ), M.mul_vec ![x, y] = ![x, y] → y / x = 7 / 4 :=
by
  intro M
  sorry

end projection_vector_ratio_l14_14409


namespace count_4_letter_words_with_E_l14_14183

theorem count_4_letter_words_with_E :
  let letters := {'A', 'B', 'C', 'D', 'E'}
  let total_4_letter_words := (letters.card) ^ 4
  let words_without_E := (letters.erase 'E').card ^ 4
  total_4_letter_words - words_without_E = 369 := by
  sorry

end count_4_letter_words_with_E_l14_14183


namespace subset_to_union_eq_l14_14477

open Set

variable {α : Type*} (A B : Set α)

theorem subset_to_union_eq (h : A ∩ B = A) : A ∪ B = B :=
by
  sorry

end subset_to_union_eq_l14_14477


namespace jenna_remaining_money_l14_14669

theorem jenna_remaining_money (m c : ℝ) (h : (1 / 4) * m = (1 / 2) * c) : (m - c) / m = 1 / 2 :=
by
  sorry

end jenna_remaining_money_l14_14669


namespace part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l14_14882

-- Definitions for times needed by copiers A and B
def time_A : ℕ := 90
def time_B : ℕ := 60

-- (1) Combined time for both copiers
theorem part1_combined_time : 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 36 = 1 := 
by sorry

-- (2) Time left for copier A alone
theorem part2_copier_A_insufficient (mins_combined : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → time_left = 13 → 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + time_left / (time_A : ℝ) ≠ 1 := 
by sorry

-- (3) Combined time with B after repair is sufficient
theorem part3_combined_after_repair (mins_combined : ℕ) (mins_repair_B : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → mins_repair_B = 9 → time_left = 13 →
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + 9 / (time_A : ℝ) + 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 2.4 = 1 := 
by sorry

end part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l14_14882


namespace females_in_band_not_orchestra_l14_14052

/-- The band at Pythagoras High School has 120 female members. -/
def females_in_band : ℕ := 120

/-- The orchestra at Pythagoras High School has 70 female members. -/
def females_in_orchestra : ℕ := 70

/-- There are 45 females who are members of both the band and the orchestra. -/
def females_in_both : ℕ := 45

/-- The combined total number of students involved in either the band or orchestra or both is 250. -/
def total_students : ℕ := 250

/-- The number of females in the band who are NOT in the orchestra. -/
def females_in_band_only : ℕ := females_in_band - females_in_both

theorem females_in_band_not_orchestra : females_in_band_only = 75 := by
  sorry

end females_in_band_not_orchestra_l14_14052


namespace arctan_sum_l14_14495

theorem arctan_sum (a b : ℝ) (h1 : a = 1/3) (h2 : (a + 1) * (b + 1) = 3) : 
  Real.arctan a + Real.arctan b = Real.arctan (19 / 7) :=
by
  sorry

end arctan_sum_l14_14495


namespace one_positive_real_solution_l14_14616

noncomputable def f (x : ℝ) : ℝ := x^4 + 5 * x^3 + 10 * x^2 + 2023 * x - 2021

theorem one_positive_real_solution : 
  ∃! x : ℝ, 0 < x ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end one_positive_real_solution_l14_14616


namespace variance_of_dataset_l14_14769

noncomputable def dataset : List ℝ := [3, 6, 9, 8, 4]

noncomputable def mean (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => y + acc) 0) / (x.length)

noncomputable def variance (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => (y - mean x)^2 + acc) 0) / (x.length)

theorem variance_of_dataset :
  variance dataset = 26 / 5 :=
by
  sorry

end variance_of_dataset_l14_14769


namespace location_determined_l14_14266

def determine_location(p : String) : Prop :=
  p = "Longitude 118°E, Latitude 40°N"

axiom row_2_in_cinema : ¬determine_location "Row 2 in a cinema"
axiom daqiao_south_road_nanjing : ¬determine_location "Daqiao South Road in Nanjing"
axiom thirty_degrees_northeast : ¬determine_location "30° northeast"
axiom longitude_latitude : determine_location "Longitude 118°E, Latitude 40°N"

theorem location_determined : determine_location "Longitude 118°E, Latitude 40°N" :=
longitude_latitude

end location_determined_l14_14266


namespace find_x_squared_plus_y_squared_l14_14494

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 75) : x^2 + y^2 = 3205 / 121 :=
by
  sorry

end find_x_squared_plus_y_squared_l14_14494


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_l14_14258

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4 : 
  Nat.minFac (11^5 - 11^4) = 2 := 
by sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_l14_14258


namespace perfect_square_conditions_l14_14073

theorem perfect_square_conditions (x y k : ℝ) :
  (∃ a : ℝ, x^2 + k * x * y + 81 * y^2 = a^2) ↔ (k = 18 ∨ k = -18) :=
sorry

end perfect_square_conditions_l14_14073


namespace pow_mod_l14_14108

theorem pow_mod (a n m : ℕ) (h : a % m = 1) : (a ^ n) % m = 1 := by
  sorry

example : (11 ^ 2023) % 5 = 1 := by
  apply pow_mod 11 2023 5
  norm_num
  have : 11 % 5 = 1 := by norm_num
  exact this

end pow_mod_l14_14108


namespace coaching_fee_correct_l14_14814

noncomputable def total_coaching_fee : ℝ :=
  let daily_fee : ℝ := 39
  let discount_threshold : ℝ := 50
  let discount_rate : ℝ := 0.10
  let total_days : ℝ := 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 3 -- non-leap year days count up to Nov 3
  let discount_days : ℝ := total_days - discount_threshold
  let discounted_fee : ℝ := daily_fee * (1 - discount_rate)
  let fee_before_discount : ℝ := discount_threshold * daily_fee
  let fee_after_discount : ℝ := discount_days * discounted_fee
  fee_before_discount + fee_after_discount

theorem coaching_fee_correct :
  total_coaching_fee = 10967.7 := by
  sorry

end coaching_fee_correct_l14_14814


namespace number_of_passed_candidates_l14_14984

theorem number_of_passed_candidates
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 4200) : P = 100 :=
sorry

end number_of_passed_candidates_l14_14984


namespace theoretical_yield_H2SO4_l14_14618

-- Define the theoretical yield calculation problem in terms of moles of reactions and products
theorem theoretical_yield_H2SO4 
  (moles_SO3 : ℝ) (moles_H2O : ℝ) 
  (reaction : moles_SO3 + moles_H2O = 2.0 + 1.5) 
  (limiting_reactant_H2O : moles_H2O = 1.5) : 
  1.5 = moles_H2O * 1 :=
  sorry

end theoretical_yield_H2SO4_l14_14618


namespace part_a_part_b_l14_14068

-- Definitions for maximum factor increases
def f (n : ℕ) (a : ℕ) : ℚ := sorry
def t (n : ℕ) (a : ℕ) : ℚ := sorry

-- Part (a): Prove the factor increase for exactly 1 blue cube in 100 boxes
theorem part_a : f 100 1 = 2^100 / 100 := sorry

-- Part (b): Prove the factor increase for some integer \( k \) blue cubes in 100 boxes, \( 1 < k \leq 100 \)
theorem part_b (k : ℕ) (hk : 1 < k ∧ k ≤ 100) : t 100 k = 2^100 / (2^100 - k - 1) := sorry

end part_a_part_b_l14_14068


namespace nonneg_integer_solutions_otimes_l14_14304

noncomputable def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

theorem nonneg_integer_solutions_otimes :
  {x : ℕ | otimes 2 x ≥ 3} = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_otimes_l14_14304


namespace prove_inequality_l14_14862

theorem prove_inequality (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 :=
  sorry

end prove_inequality_l14_14862


namespace ab_fraction_inequality_l14_14689

theorem ab_fraction_inequality (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) :
  (a * b * (1 - a) * (1 - b)) / ((1 - a * b) ^ 2) < 1 / 4 :=
by
  sorry

end ab_fraction_inequality_l14_14689


namespace ratio_fenced_region_l14_14743

theorem ratio_fenced_region (L W : ℝ) (k : ℝ) 
  (area_eq : L * W = 200)
  (fence_eq : 2 * W + L = 40)
  (mult_eq : L = k * W) :
  k = 2 :=
by
  sorry

end ratio_fenced_region_l14_14743


namespace all_lines_can_be_paired_perpendicular_l14_14798

noncomputable def can_pair_perpendicular_lines : Prop := 
  ∀ (L1 L2 : ℝ), 
    L1 ≠ L2 → 
      ∃ (m : ℝ), 
        (m * L1 = -1/L2 ∨ L1 = 0 ∧ L2 ≠ 0 ∨ L2 = 0 ∧ L1 ≠ 0)

theorem all_lines_can_be_paired_perpendicular : can_pair_perpendicular_lines :=
sorry

end all_lines_can_be_paired_perpendicular_l14_14798


namespace base7_to_base10_321_is_162_l14_14883

-- Define the conversion process from a base-7 number to base-10
def convert_base7_to_base10 (n: ℕ) : ℕ :=
  3 * 7^2 + 2 * 7^1 + 1 * 7^0

theorem base7_to_base10_321_is_162 :
  convert_base7_to_base10 321 = 162 :=
by
  sorry

end base7_to_base10_321_is_162_l14_14883


namespace book_page_count_l14_14723

theorem book_page_count (pages_per_night : ℝ) (nights : ℝ) : pages_per_night = 120.0 → nights = 10.0 → pages_per_night * nights = 1200.0 :=
by
  sorry

end book_page_count_l14_14723


namespace total_buttons_needed_l14_14531

def shirts_sewn_on_monday := 4
def shirts_sewn_on_tuesday := 3
def shirts_sewn_on_wednesday := 2
def buttons_per_shirt := 5

theorem total_buttons_needed : 
  (shirts_sewn_on_monday + shirts_sewn_on_tuesday + shirts_sewn_on_wednesday) * buttons_per_shirt = 45 :=
by 
  sorry

end total_buttons_needed_l14_14531


namespace det_S_l14_14510

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0], ![0, 1]]

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![ ![real.sqrt 2 / 2, -real.sqrt 2 / 2],
     ![real.sqrt 2 / 2, real.sqrt 2 / 2] ]

noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ := A ⬝ B

theorem det_S : S.det = 2 :=
sorry

end det_S_l14_14510


namespace matrix_determinant_l14_14455

theorem matrix_determinant (x : ℝ) :
  Matrix.det ![![x, x + 2], ![3, 2 * x]] = 2 * x^2 - 3 * x - 6 :=
by
  sorry

end matrix_determinant_l14_14455


namespace taller_tree_height_l14_14550

-- Definitions and Variables
variables (h : ℝ)

-- Conditions as Definitions
def top_difference_condition := (h - 20) / h = 5 / 7

-- Proof Statement
theorem taller_tree_height (h : ℝ) (H : top_difference_condition h) : h = 70 := 
by {
  sorry
}

end taller_tree_height_l14_14550


namespace greatest_possible_third_side_l14_14079

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l14_14079


namespace standard_deviation_is_one_l14_14789

noncomputable def standard_deviation (μ : ℝ) (σ : ℝ) : Prop :=
  ∀ x : ℝ, (0.68 * μ ≤ x ∧ x ≤ 1.32 * μ) → σ = 1

theorem standard_deviation_is_one (a : ℝ) (σ : ℝ) :
  (0.68 * a ≤ a + σ ∧ a + σ ≤ 1.32 * a) → σ = 1 :=
by
  -- Proof omitted.
  sorry

end standard_deviation_is_one_l14_14789


namespace parallel_vectors_perpendicular_vectors_l14_14511

open Real

variables (e1 e2 : ℝ × ℝ) 
variables (a b : ℝ × ℝ) (λ : ℝ)

-- Conditions: e1 and e2 are mutually perpendicular unit vectors
def perpendicular_unit_vectors (e1 e2 : ℝ × ℝ) : Prop :=
  (e1.1 ^ 2 + e1.2 ^ 2 = 1) ∧ (e2.1 ^ 2 + e2.2 ^ 2 = 1) ∧ (e1.1 * e2.1 + e1.2 * e2.2 = 0)

-- Definitions of vectors a and b
def vector_a (e1 e2 : ℝ × ℝ) : ℝ × ℝ := 
  (-2 * e1.1 - e2.1, -2 * e1.2 - e2.2)

def vector_b (e1 e2 : ℝ × ℝ) (λ : ℝ) : ℝ × ℝ := 
  (e1.1 - λ * e2.1, e1.2 - λ * e2.2)

-- Proof problem 1: If a parallel b, then λ = -1/2
theorem parallel_vectors (h : perpendicular_unit_vectors e1 e2) 
  (h_parallel : vector_a e1 e2 = -2 * vector_b e1 e2 (-1/2)) : λ = -1/2 := sorry

-- Proof problem 2: If a perpendicular b, then λ = 2
theorem perpendicular_vectors (h : perpendicular_unit_vectors e1 e2) 
  (h_perpendicular : (vector_a e1 e2).fst * (vector_b e1 e2 λ).fst + 
                    (vector_a e1 e2).snd * (vector_b e1 e2 λ).snd = 0) : λ = 2 := sorry

end parallel_vectors_perpendicular_vectors_l14_14511


namespace closest_multiple_of_18_2021_l14_14860

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def closest_multiple_of (n k : ℕ) : ℕ :=
if (n % k) * 2 < k then n - (n % k) else n + (k - n % k)

theorem closest_multiple_of_18_2021 :
  closest_multiple_of 2021 18 = 2016 := by
    sorry

end closest_multiple_of_18_2021_l14_14860


namespace complement_of_A_l14_14019

variables (U : Set ℝ) (A : Set ℝ)
def universal_set : Prop := U = Set.univ
def range_of_function : Prop := A = {x : ℝ | 0 ≤ x}

theorem complement_of_A (hU : universal_set U) (hA : range_of_function A) : 
  U \ A = {x : ℝ | x < 0} :=
by 
  sorry

end complement_of_A_l14_14019


namespace rectangle_area_is_243_square_meters_l14_14602

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l14_14602


namespace intersection_M_N_l14_14337

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14337


namespace largest_n_unique_k_l14_14857

theorem largest_n_unique_k (n : ℕ) (h : ∃ k : ℕ, (9 / 17 : ℚ) < n / (n + k) ∧ n / (n + k) < (8 / 15 : ℚ) ∧ ∀ k' : ℕ, ((9 / 17 : ℚ) < n / (n + k') ∧ n / (n + k') < (8 / 15 : ℚ)) → k' = k) : n = 72 :=
sorry

end largest_n_unique_k_l14_14857


namespace days_C_alone_l14_14730

theorem days_C_alone (r_A r_B r_C : ℝ) (h1 : r_A + r_B = 1 / 3) (h2 : r_B + r_C = 1 / 6) (h3 : r_A + r_C = 5 / 18) : 
  1 / r_C = 18 := 
  sorry

end days_C_alone_l14_14730


namespace evaluate_expression_l14_14112

theorem evaluate_expression :
  2 ^ (0 ^ (1 ^ 9)) + ((2 ^ 0) ^ 1) ^ 9 = 2 := 
sorry

end evaluate_expression_l14_14112


namespace surface_area_increase_l14_14437

def cube_dimensions : ℝ × ℝ × ℝ := (10, 10, 10)

def number_of_cuts := 3

def initial_surface_area (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  6 * (length * width)

def increase_in_surface_area (cuts : ℕ) (length : ℝ) (width : ℝ) : ℝ :=
  cuts * 2 * (length * width)

theorem surface_area_increase : 
  initial_surface_area 10 10 10 + increase_in_surface_area 3 10 10 = 
  initial_surface_area 10 10 10 + 600 :=
by
  sorry

end surface_area_increase_l14_14437


namespace find_solutions_l14_14760

theorem find_solutions (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 48 →
  (x = 1.2 ∨ x = -81.2) :=
by sorry

end find_solutions_l14_14760


namespace intersection_complement_eq_l14_14225

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}
def U : Set ℝ := Set.univ  -- Universal set U is the set of all real numbers

theorem intersection_complement_eq : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_eq_l14_14225


namespace average_salary_of_technicians_l14_14792

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (average_salary_all : ℕ)
  (average_salary_non_technicians : ℕ)
  (num_technicians : ℕ)
  (num_non_technicians : ℕ)
  (h1 : total_workers = 21)
  (h2 : average_salary_all = 8000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : num_non_technicians = 14) :
  (average_salary_all * total_workers - average_salary_non_technicians * num_non_technicians) / num_technicians = 12000 :=
by
  sorry

end average_salary_of_technicians_l14_14792


namespace intersection_M_N_l14_14342

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l14_14342


namespace rectangle_area_l14_14582

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l14_14582


namespace greatest_third_side_l14_14082

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l14_14082


namespace total_snow_volume_l14_14842

theorem total_snow_volume (length width initial_depth additional_depth: ℝ) 
  (h_length : length = 30) 
  (h_width : width = 3) 
  (h_initial_depth : initial_depth = 3 / 4) 
  (h_additional_depth : additional_depth = 1 / 4) : 
  (length * width * initial_depth) + (length * width * additional_depth) = 90 := 
by
  -- proof steps would go here
  sorry

end total_snow_volume_l14_14842


namespace crayons_slightly_used_l14_14852

theorem crayons_slightly_used (total_crayons : ℕ) (new_fraction : ℚ) (broken_fraction : ℚ) 
  (htotal : total_crayons = 120) (hnew : new_fraction = 1 / 3) (hbroken : broken_fraction = 20 / 100) :
  let new_crayons := total_crayons * new_fraction
  let broken_crayons := total_crayons * broken_fraction
  let slightly_used_crayons := total_crayons - new_crayons - broken_crayons
  slightly_used_crayons = 56 := 
by
  -- This is where the proof would go
  sorry

end crayons_slightly_used_l14_14852


namespace area_increase_300_percent_l14_14981

noncomputable def percentage_increase_of_area (d : ℝ) : ℝ :=
  let d' := 2 * d
  let r := d / 2
  let r' := d' / 2
  let A := Real.pi * r^2
  let A' := Real.pi * (r')^2
  100 * (A' - A) / A

theorem area_increase_300_percent (d : ℝ) : percentage_increase_of_area d = 300 :=
by
  sorry

end area_increase_300_percent_l14_14981


namespace least_possible_value_of_smallest_integer_l14_14241

theorem least_possible_value_of_smallest_integer {A B C D : ℤ} 
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_mean: (A + B + C + D) / 4 = 68)
  (h_largest: D = 90) :
  A ≥ 5 := 
sorry

end least_possible_value_of_smallest_integer_l14_14241


namespace max_groups_eq_one_l14_14812

-- Defining the conditions 
def eggs : ℕ := 16
def marbles : ℕ := 3
def rubber_bands : ℕ := 5

-- The theorem statement
theorem max_groups_eq_one
  (h1 : eggs = 16)
  (h2 : marbles = 3)
  (h3 : rubber_bands = 5) :
  ∀ g : ℕ, (g ≤ eggs ∧ g ≤ marbles ∧ g ≤ rubber_bands) →
  (eggs % g = 0) ∧ (marbles % g = 0) ∧ (rubber_bands % g = 0) →
  g = 1 :=
by
  sorry

end max_groups_eq_one_l14_14812


namespace complex_inverse_l14_14930

noncomputable def complex_expression (i : ℂ) (h_i : i ^ 2 = -1) : ℂ :=
  (3 * i - 3 * (1 / i))⁻¹

theorem complex_inverse (i : ℂ) (h_i : i^2 = -1) :
  complex_expression i h_i = -i / 6 :=
by
  -- the proof part is omitted
  sorry

end complex_inverse_l14_14930


namespace boxes_needed_l14_14827

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end boxes_needed_l14_14827


namespace marble_draw_probability_l14_14874

theorem marble_draw_probability :
  let total_marbles := 12
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 3

  let p_red_first := (red_marbles / total_marbles : ℚ)
  let p_white_second := (white_marbles / (total_marbles - 1) : ℚ)
  let p_blue_third := (blue_marbles / (total_marbles - 2) : ℚ)
  
  p_red_first * p_white_second * p_blue_third = (1/22 : ℚ) :=
by
  sorry

end marble_draw_probability_l14_14874


namespace find_f_of_16_l14_14009

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem find_f_of_16 : (∃ a : ℝ, f 2 a = Real.sqrt 2) → f 16 (1/2) = 4 :=
by
  intro h
  sorry

end find_f_of_16_l14_14009


namespace initial_number_of_students_l14_14541

theorem initial_number_of_students (W : ℝ) (n : ℕ) (new_student_weight avg_weight1 avg_weight2 : ℝ)
  (h1 : avg_weight1 = 15)
  (h2 : new_student_weight = 13)
  (h3 : avg_weight2 = 14.9)
  (h4 : W = n * avg_weight1)
  (h5 : W + new_student_weight = (n + 1) * avg_weight2) : n = 19 := 
by
  sorry

end initial_number_of_students_l14_14541


namespace determinant_computation_l14_14912

variable (x y z w : ℝ)
variable (det : ℝ)
variable (H : x * w - y * z = 7)

theorem determinant_computation : 
  (x + z) * w - (y + 2 * w) * z = 7 - w * z := by
  sorry

end determinant_computation_l14_14912


namespace greatest_possible_third_side_l14_14077

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l14_14077


namespace base7_digit_sum_l14_14370

theorem base7_digit_sum (A B C : ℕ) (hA : 1 ≤ A ∧ A < 7) (hB : 1 ≤ B ∧ B < 7) 
  (hC : 1 ≤ C ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eq : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) : 
  B + C = 6 := 
sorry

end base7_digit_sum_l14_14370


namespace no_real_ordered_triples_l14_14631

theorem no_real_ordered_triples (x y z : ℝ) (h1 : x + y = 3) (h2 : xy - z^2 = 4) : false :=
sorry

end no_real_ordered_triples_l14_14631


namespace triangle_min_perimeter_l14_14448

-- Definitions of points A, B, and C and the conditions specified in the problem.
def pointA : ℝ × ℝ := (3, 2)
def pointB (t : ℝ) : ℝ × ℝ := (t, t)
def pointC (c : ℝ) : ℝ × ℝ := (c, 0)

-- Main theorem which states that the minimum perimeter of triangle ABC is sqrt(26).
theorem triangle_min_perimeter : 
  ∃ (B C : ℝ × ℝ), B = pointB (B.1) ∧ C = pointC (C.1) ∧ 
  ∀ (B' C' : ℝ × ℝ), B' = pointB (B'.1) ∧ C' = pointC (C'.1) →
  (dist pointA B + dist B C + dist C pointA ≥ dist (2, 3) (3, -2)) :=
by 
  sorry

end triangle_min_perimeter_l14_14448


namespace triangle_side_length_l14_14733

theorem triangle_side_length 
  (r : ℝ)                    -- radius of the inscribed circle
  (h_cos_ABC : ℝ)            -- cosine of angle ABC
  (h_midline : Bool)         -- the circle touches the midline parallel to AC
  (h_r : r = 1)              -- given radius is 1
  (h_cos : h_cos_ABC = 0.8)  -- given cos(ABC) = 0.8
  (h_touch : h_midline = true)  -- given that circle touches the midline
  : AC = 3 := 
sorry

end triangle_side_length_l14_14733


namespace pi_irrational_l14_14861

theorem pi_irrational :
  ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (π = a / b) :=
by
  sorry

end pi_irrational_l14_14861


namespace num_integers_between_l14_14651

-- Define the constants
def a : ℝ := 10
def b₁ : ℝ := 0.5
def b₂ : ℝ := 0.6

-- Define the cubes
def x : ℝ := (a + b₁)^3
def y : ℝ := (a + b₂)^3

-- Define the function to count the integers within the interval
def count_integers_between (x y : ℝ) : ℕ :=
  let min_int := Int.ceil x
  let max_int := Int.floor y
  (max_int - min_int + 1).toNat

-- The statement to prove
theorem num_integers_between : count_integers_between x y = 33 := by
  sorry

end num_integers_between_l14_14651


namespace emails_in_morning_and_evening_l14_14033

def morning_emails : ℕ := 3
def afternoon_emails : ℕ := 4
def evening_emails : ℕ := 8

theorem emails_in_morning_and_evening : morning_emails + evening_emails = 11 :=
by
  sorry

end emails_in_morning_and_evening_l14_14033


namespace kevin_ends_with_604_cards_l14_14508

theorem kevin_ends_with_604_cards : 
  ∀ (initial_cards found_cards : ℕ), initial_cards = 65 → found_cards = 539 → initial_cards + found_cards = 604 :=
by
  intros initial_cards found_cards h_initial h_found
  sorry

end kevin_ends_with_604_cards_l14_14508


namespace ticket_cost_difference_l14_14808

theorem ticket_cost_difference
  (num_adults : ℕ) (num_children : ℕ)
  (cost_adult_ticket : ℕ) (cost_child_ticket : ℕ)
  (h1 : num_adults = 9)
  (h2 : num_children = 7)
  (h3 : cost_adult_ticket = 11)
  (h4 : cost_child_ticket = 7) :
  num_adults * cost_adult_ticket - num_children * cost_child_ticket = 50 := 
by
  sorry

end ticket_cost_difference_l14_14808


namespace conference_duration_is_960_l14_14567

-- The problem statement definition
def conference_sessions_duration_in_minutes (day1_hours : ℕ) (day1_minutes : ℕ) (day2_hours : ℕ) (day2_minutes : ℕ) : ℕ :=
  (day1_hours * 60 + day1_minutes) + (day2_hours * 60 + day2_minutes)

-- The theorem we want to prove given the above conditions
theorem conference_duration_is_960 :
  conference_sessions_duration_in_minutes 7 15 8 45 = 960 :=
by 
  -- The proof is omitted
  sorry

end conference_duration_is_960_l14_14567


namespace pancakes_eaten_by_older_is_12_l14_14365

/-- Pancake problem conditions -/
def initial_pancakes : ℕ := 19
def final_pancakes : ℕ := 11
def younger_eats_per_cycle : ℕ := 1
def older_eats_per_cycle : ℕ := 3
def grandma_bakes_per_cycle : ℕ := 2
def net_reduction_per_cycle := younger_eats_per_cycle + older_eats_per_cycle - grandma_bakes_per_cycle
def total_pancakes_eaten_by_older (cycles : ℕ) := older_eats_per_cycle * cycles

/-- Calculate the cycles based on net reduction -/
def cycles : ℕ := (initial_pancakes - final_pancakes) / net_reduction_per_cycle

/-- Prove the number of pancakes the older grandchild eats is 12 based on given conditions --/
theorem pancakes_eaten_by_older_is_12 : total_pancakes_eaten_by_older cycles = 12 := by
  sorry

end pancakes_eaten_by_older_is_12_l14_14365


namespace greatest_integer_third_side_l14_14087

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l14_14087


namespace imaginary_part_of_z_l14_14993

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_of_z : Complex.im z = -2 := by
  sorry

end imaginary_part_of_z_l14_14993


namespace length_of_chord_l14_14782

theorem length_of_chord
    (center : ℝ × ℝ) 
    (radius : ℝ) 
    (line : ℝ × ℝ × ℝ) 
    (circle_eq : (x : ℝ) → (y : ℝ) → ((x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2))
    (line_eq : (x : ℝ) → (y : ℝ) → (line.1 * x + line.2 * y + line.3 = 0)) :
    2 * radius * (if h : radius ≠ 0 then (1 - (1 / 2) * ((|line.1 * center.1 + line.2 * center.2 + line.3| / (real.sqrt (line.1 ^ 2 + line.2 ^ 2))) / radius) ^ 2) else 0).sqrt = 2 * real.sqrt 2 :=
by
    sorry

-- Definitions and conditions
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 2
def line : ℝ × ℝ × ℝ := (1, 1, 1)

noncomputable def circle_eq : (ℝ → ℝ → Prop) :=
    λ x y, (x - center.1) ^ 2 + y ^ 2 = 4

noncomputable def line_eq : (ℝ → ℝ → Prop) :=
    λ x y, x + y + 1 = 0

-- Applying the theorem
#eval (length_of_chord center radius line circle_eq line_eq)

end length_of_chord_l14_14782


namespace original_workers_l14_14568

theorem original_workers (x y : ℝ) (h : x = (65 / 100) * y) : y = (20 / 13) * x :=
by sorry

end original_workers_l14_14568


namespace difference_of_roots_l14_14314

theorem difference_of_roots : 
  let a := 6 + 3 * Real.sqrt 5
  let b := 3 + Real.sqrt 5
  let c := 1
  ∃ x1 x2 : ℝ, (a * x1^2 - b * x1 + c = 0) ∧ (a * x2^2 - b * x2 + c = 0) ∧ x1 ≠ x2 
  ∧ x1 > x2 ∧ (x1 - x2) = (Real.sqrt 6 - Real.sqrt 5) / 3 := 
sorry

end difference_of_roots_l14_14314


namespace kimberly_store_visits_l14_14957

def peanuts_per_visit : ℕ := 7
def total_peanuts : ℕ := 21

def visits : ℕ := total_peanuts / peanuts_per_visit

theorem kimberly_store_visits : visits = 3 :=
by
  sorry

end kimberly_store_visits_l14_14957


namespace find_exp_l14_14003

noncomputable def a : ℝ := sorry
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom a_m_eq_six : a ^ m = 6
axiom a_n_eq_six : a ^ n = 6

theorem find_exp : a ^ (2 * m - n) = 6 :=
by
  sorry

end find_exp_l14_14003


namespace area_of_rectangle_l14_14578

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l14_14578


namespace inequality_proof_l14_14645

theorem inequality_proof (a b : ℝ) (h_a : a > 0) (h_b : 3 + b = a) : 
  3 / b + 1 / a >= 3 :=
sorry

end inequality_proof_l14_14645


namespace find_unique_digit_sets_l14_14377

theorem find_unique_digit_sets (a b c : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
 (h4 : 22 * (a + b + c) = 462) :
  (a = 4 ∧ b = 8 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 9 ∧ c = 8) ∨ 
  (a = 8 ∧ b = 4 ∧ c = 9) ∨
  (a = 8 ∧ b = 9 ∧ c = 4) ∨ 
  (a = 9 ∧ b = 4 ∧ c = 8) ∨ 
  (a = 9 ∧ b = 8 ∧ c = 4) ∨
  (a = 5 ∧ b = 7 ∧ c = 9) ∨ 
  (a = 5 ∧ b = 9 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 5 ∧ c = 9) ∨
  (a = 7 ∧ b = 9 ∧ c = 5) ∨ 
  (a = 9 ∧ b = 5 ∧ c = 7) ∨ 
  (a = 9 ∧ b = 7 ∧ c = 5) ∨
  (a = 6 ∧ b = 7 ∧ c = 8) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 6 ∧ c = 8) ∨
  (a = 7 ∧ b = 8 ∧ c = 6) ∨ 
  (a = 8 ∧ b = 6 ∧ c = 7) ∨ 
  (a = 8 ∧ b = 7 ∧ c = 6) :=
sorry

end find_unique_digit_sets_l14_14377


namespace divisor_is_13_l14_14492

theorem divisor_is_13 (N D : ℕ) (h1 : N = 32) (h2 : (N - 6) / D = 2) : D = 13 := by
  sorry

end divisor_is_13_l14_14492


namespace remainder_M_divided_by_1000_l14_14708

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

end remainder_M_divided_by_1000_l14_14708


namespace T_bisects_broken_line_l14_14509

def midpoint_arc {α : Type*} [LinearOrderedField α] (A B C : α) : α := (A + B + C) / 2
def projection_perpendicular {α : Type*} [LinearOrderedField α] (F A B C : α) : α := sorry -- Define perpendicular projection T

theorem T_bisects_broken_line {α : Type*} [LinearOrderedField α]
  (A B C : α) (F := midpoint_arc A B C) (T := projection_perpendicular F A B C) :
  T = (A + B + C) / 2 :=
sorry

end T_bisects_broken_line_l14_14509


namespace no_integer_triplets_satisfying_eq_l14_14465

theorem no_integer_triplets_satisfying_eq (x y z : ℤ) : 3 * x^2 + 7 * y^2 ≠ z^4 := 
by {
  sorry
}

end no_integer_triplets_satisfying_eq_l14_14465


namespace solve_equation_l14_14405

noncomputable def is_solution (x : ℝ) : Prop :=
  (x / (2 * Real.sqrt 2) + (5 * Real.sqrt 2) / 2) * Real.sqrt (x^3 - 64 * x + 200) = x^2 + 6 * x - 40

noncomputable def conditions (x : ℝ) : Prop :=
  (x^3 - 64 * x + 200) ≥ 0 ∧ x ≥ 4

theorem solve_equation :
  (∀ x, is_solution x → conditions x) = (x = 6 ∨ x = 1 + Real.sqrt 13) :=
by sorry

end solve_equation_l14_14405


namespace third_side_triangle_max_l14_14098

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l14_14098


namespace intersection_M_N_l14_14339

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l14_14339


namespace sum_distinct_x2_y2_z2_l14_14408

/-
Given positive integers x, y, and z such that
x + y + z = 30 and gcd(x, y) + gcd(y, z) + gcd(z, x) = 10,
prove that the sum of all possible distinct values of x^2 + y^2 + z^2 is 404.
-/
theorem sum_distinct_x2_y2_z2 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 30) 
  (h_gcd : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10) : 
  x^2 + y^2 + z^2 = 404 :=
sorry

end sum_distinct_x2_y2_z2_l14_14408


namespace train_times_valid_l14_14422

-- Define the parameters and conditions
def trainA_usual_time : ℝ := 180 -- minutes
def trainB_travel_time : ℝ := 810 -- minutes

theorem train_times_valid (t : ℝ) (T_B : ℝ) 
  (cond1 : (7 / 6) * t = t + 30)
  (cond2 : T_B = 4.5 * t) : 
  t = trainA_usual_time ∧ T_B = trainB_travel_time :=
by
  sorry

end train_times_valid_l14_14422


namespace cos_double_plus_cos_l14_14008

theorem cos_double_plus_cos (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 3) :
  Real.cos (2 * α) + Real.cos α = -4 / 9 :=
by
  sorry

end cos_double_plus_cos_l14_14008


namespace saved_percent_correct_l14_14435

noncomputable def price_kit : ℝ := 144.20
noncomputable def price1 : ℝ := 21.75
noncomputable def price2 : ℝ := 18.60
noncomputable def price3 : ℝ := 23.80
noncomputable def price4 : ℝ := 29.35

noncomputable def total_price_individual : ℝ := 2 * price1 + 2 * price2 + price3 + 2 * price4
noncomputable def amount_saved : ℝ := total_price_individual - price_kit
noncomputable def percent_saved : ℝ := 100 * (amount_saved / total_price_individual)

theorem saved_percent_correct : percent_saved = 11.64 := by
  sorry

end saved_percent_correct_l14_14435


namespace ratio_c_to_d_l14_14659

theorem ratio_c_to_d (a b c d : ℚ) 
  (h1 : a / b = 3 / 4) 
  (h2 : b / c = 7 / 9) 
  (h3 : a / d = 0.4166666666666667) : 
  c / d = 5 / 7 := 
by
  -- Proof not needed
  sorry

end ratio_c_to_d_l14_14659


namespace living_room_curtain_length_l14_14610

theorem living_room_curtain_length :
  let length_bolt := 16
  let width_bolt := 12
  let area_bolt := length_bolt * width_bolt
  let area_left := 160
  let area_cut := area_bolt - area_left
  let length_bedroom := 2
  let width_bedroom := 4
  let area_bedroom := length_bedroom * width_bedroom
  let area_living_room := area_cut - area_bedroom
  let width_living_room := 4
  area_living_room / width_living_room = 6 :=
by
  sorry

end living_room_curtain_length_l14_14610


namespace algorithm_outputs_min_value_l14_14746

theorem algorithm_outputs_min_value (a b c d : ℕ) :
  let m := a;
  let m := if b < m then b else m;
  let m := if c < m then c else m;
  let m := if d < m then d else m;
  m = min (min (min a b) c) d :=
by
  sorry

end algorithm_outputs_min_value_l14_14746


namespace inverse_proportion_increasing_implication_l14_14909

theorem inverse_proportion_increasing_implication (m x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2, x1 > 0 → x2 > 0 → x1 < x2 → (m + 3) / x1 < (m + 3) / x2) : m < -3 :=
by
  sorry

end inverse_proportion_increasing_implication_l14_14909


namespace find_values_l14_14556

theorem find_values (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : a = 2 * b + 5) (h3 : Nat.Prime (a + 7 * b)) : (a = 9 ∧ b = 2) ∨ (a = 17 ∧ b = 6) :=
sorry

end find_values_l14_14556


namespace relationship_between_abc_l14_14217

noncomputable def a := (4 / 5) ^ (1 / 2)
noncomputable def b := (5 / 4) ^ (1 / 5)
noncomputable def c := (3 / 4) ^ (3 / 4)

theorem relationship_between_abc : c < a ∧ a < b := by
  sorry

end relationship_between_abc_l14_14217


namespace range_of_area_of_acute_triangle_ABC_l14_14793

theorem range_of_area_of_acute_triangle_ABC
  (A B C : ℝ)
  (h_angle_A : A = π / 6)
  (h_BC : B = 1)
  (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :
  let S_ABC := (1 / 2) * B * (cos C + sqrt 3 * sin C) * sin A in
  S_ABC ∈ set.Ioc (sqrt 3 / 4) (1 / 2 + sqrt 3 / 4) := 
sorry

end range_of_area_of_acute_triangle_ABC_l14_14793


namespace num_4digit_special_integers_l14_14776

noncomputable def count_valid_4digit_integers : ℕ :=
  let first_two_options := 3 * 3 -- options for the first two digits
  let valid_last_two_pairs := 4 -- (6,9), (7,8), (8,7), (9,6)
  first_two_options * valid_last_two_pairs

theorem num_4digit_special_integers : count_valid_4digit_integers = 36 :=
by
  sorry

end num_4digit_special_integers_l14_14776


namespace linear_equation_must_be_neg2_l14_14780

theorem linear_equation_must_be_neg2 {m : ℝ} (h1 : |m| - 1 = 1) (h2 : m ≠ 2) : m = -2 :=
sorry

end linear_equation_must_be_neg2_l14_14780


namespace comparison_of_a_b_c_l14_14642

theorem comparison_of_a_b_c (a b c : ℝ) (h_a : a = Real.log 2) (h_b : b = 5^(-1/2 : ℝ)) (h_c : c = Real.sin (Real.pi / 6)) : 
  b < c ∧ c < a :=
by
  sorry

end comparison_of_a_b_c_l14_14642


namespace people_remaining_on_bus_l14_14971

theorem people_remaining_on_bus
  (students_left : ℕ) (students_right : ℕ) (students_back : ℕ)
  (students_aisle : ℕ) (teachers : ℕ) (bus_driver : ℕ) 
  (students_off1 : ℕ) (teachers_off1 : ℕ)
  (students_off2 : ℕ) (teachers_off2 : ℕ)
  (students_off3 : ℕ) :
  students_left = 42 ∧ students_right = 38 ∧ students_back = 5 ∧
  students_aisle = 15 ∧ teachers = 2 ∧ bus_driver = 1 ∧
  students_off1 = 14 ∧ teachers_off1 = 1 ∧
  students_off2 = 18 ∧ teachers_off2 = 1 ∧
  students_off3 = 5 →
  (students_left + students_right + students_back + students_aisle + teachers + bus_driver) -
  (students_off1 + teachers_off1 + students_off2 + teachers_off2 + students_off3) = 64 :=
by {
  sorry
}

end people_remaining_on_bus_l14_14971


namespace rectangle_area_is_243_square_meters_l14_14603

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l14_14603


namespace model_to_statue_ratio_l14_14994

theorem model_to_statue_ratio (h_statue : ℝ) (h_model : ℝ) (h_statue_eq : h_statue = 60) (h_model_eq : h_model = 4) :
  (h_statue / h_model) = 15 := by
  sorry

end model_to_statue_ratio_l14_14994


namespace cos_diff_of_symmetric_sines_l14_14024

theorem cos_diff_of_symmetric_sines (a β : Real) (h1 : Real.sin a = 1 / 3) 
  (h2 : Real.sin β = 1 / 3) (h3 : Real.cos a = -Real.cos β) : 
  Real.cos (a - β) = -7 / 9 := by
  sorry

end cos_diff_of_symmetric_sines_l14_14024


namespace unique_solution_for_a_half_l14_14917

noncomputable def unique_solution_a (a : ℝ) (h_pos : 0 < a) : Prop :=
∀ x : ℝ, 2 * a * x = x^2 - 2 * a * (Real.log x) → x = 1

theorem unique_solution_for_a_half : unique_solution_a (1 / 2) (by norm_num : 0 < (1 / 2)) :=
sorry

end unique_solution_for_a_half_l14_14917


namespace part_a_part_b_l14_14430

noncomputable theory

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (ξ ζ : Ω → ℝ) 
-- conditions for part (a)
variable [h1 : Independent ξ ζ]
variable [h2 : IdentDistrib ξ ζ]
variable [h3 : HasExpectation ξ]

theorem part_a :
  ∀ (ω : Ω), 
  E[ξ | ξ + ζ] = E[ζ | ξ + ζ] :=  
begin
  sorry
end

-- conditions for part (b)
variable [h4 : HasFiniteExpectation (ξ^2)]
variable [h5 : HasFiniteExpectation (ζ^2)]
variable [h6 : IdentDistrib ξ (-ξ)]

theorem part_b :
  ∀ (ω : Ω), 
  E[(ξ + ζ)^2 | ξ^2 + ζ^2] = ξ^2 + ζ^2 := 
begin
  sorry
end

end part_a_part_b_l14_14430


namespace parabola_tangent_angle_l14_14632

noncomputable def tangent_slope_angle : Real :=
  let x := (1 / 2 : ℝ)
  let y := x^2
  let slope := (deriv (fun x => x^2)) x
  Real.arctan slope

theorem parabola_tangent_angle :
  tangent_slope_angle = Real.pi / 4 :=
by
sorry

end parabola_tangent_angle_l14_14632


namespace quadratic_has_real_roots_l14_14941

theorem quadratic_has_real_roots (k : ℝ) :
  (∃ (x : ℝ), (k-2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ (3 / 2) ∧ k ≠ 2) :=
by
  sorry

end quadratic_has_real_roots_l14_14941


namespace mailman_total_pieces_l14_14124

def piecesOfMailFirstHouse := 6 + 5 + 3 + 4 + 2
def piecesOfMailSecondHouse := 4 + 7 + 2 + 5 + 3
def piecesOfMailThirdHouse := 8 + 3 + 4 + 6 + 1

def totalPiecesOfMail := piecesOfMailFirstHouse + piecesOfMailSecondHouse + piecesOfMailThirdHouse

theorem mailman_total_pieces : totalPiecesOfMail = 63 := by
  sorry

end mailman_total_pieces_l14_14124


namespace bethany_age_l14_14252

theorem bethany_age : ∀ (B S R : ℕ),
  (B - 3 = 2 * (S - 3)) →
  (B - 3 = R - 3 + 4) →
  (S + 5 = 16) →
  (R + 5 = 21) →
  B = 19 :=
by
  intros B S R h1 h2 h3 h4
  sorry

end bethany_age_l14_14252


namespace probability_log_value_l14_14919

noncomputable def f (x : ℝ) := Real.log x / Real.log 2 - 1

theorem probability_log_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ 10) :
  (4 / 9 : ℝ) = 
    ((8 - 4) / (10 - 1) : ℝ) := by
  sorry

end probability_log_value_l14_14919


namespace probability_C_l14_14119

-- Definitions of probabilities
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Main proof statement
theorem probability_C :
  ∀ P_C : ℚ, P_A + P_B + P_C + P_D = 1 → P_C = 1 / 4 :=
by
  intro P_C h
  sorry

end probability_C_l14_14119


namespace three_digit_numbers_condition_l14_14411

theorem three_digit_numbers_condition (a b c : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c = 2 * ((10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b)))
  ↔ (100 * a + 10 * b + c = 132 ∨ 100 * a + 10 * b + c = 264 ∨ 100 * a + 10 * b + c = 396) :=
by
  sorry

end three_digit_numbers_condition_l14_14411


namespace slope_of_tangent_at_minus_1_l14_14018

theorem slope_of_tangent_at_minus_1
  (c : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (x - 2) * (x^2 + c))
  (h_extremum : deriv f 1 = 0) :
  deriv f (-1) = 8 :=
by
  sorry

end slope_of_tangent_at_minus_1_l14_14018


namespace ratio_Nicolai_to_Charliz_l14_14922

-- Definitions based on conditions
def Haylee_guppies := 3 * 12
def Jose_guppies := Haylee_guppies / 2
def Charliz_guppies := Jose_guppies / 3
def Total_guppies := 84
def Nicolai_guppies := Total_guppies - (Haylee_guppies + Jose_guppies + Charliz_guppies)

-- Proof statement
theorem ratio_Nicolai_to_Charliz : Nicolai_guppies / Charliz_guppies = 4 := by
  sorry

end ratio_Nicolai_to_Charliz_l14_14922


namespace cloth_sales_worth_l14_14144

/--
An agent gets a commission of 2.5% on the sales of cloth. If on a certain day, he gets Rs. 15 as commission, 
proves that the worth of the cloth sold through him on that day is Rs. 600.
-/
theorem cloth_sales_worth (commission : ℝ) (rate : ℝ) (total_sales : ℝ) 
  (h_commission : commission = 15) (h_rate : rate = 2.5) (h_commission_formula : commission = (rate / 100) * total_sales) : 
  total_sales = 600 := 
by
  sorry

end cloth_sales_worth_l14_14144


namespace speed_of_train_in_km_per_hr_l14_14133

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l14_14133


namespace rectangular_field_area_l14_14594

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l14_14594


namespace lino_shells_total_l14_14227

def picked_up_shells : Float := 324.0
def put_back_shells : Float := 292.0

theorem lino_shells_total : picked_up_shells - put_back_shells = 32.0 :=
by
  sorry

end lino_shells_total_l14_14227


namespace quadratic_always_positive_l14_14155

theorem quadratic_always_positive (k : ℝ) :
  ∀ x : ℝ, x^2 - (k - 4) * x + k - 7 > 0 :=
sorry

end quadratic_always_positive_l14_14155


namespace jessica_total_cost_l14_14955

-- Define the costs
def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

-- Define the total cost
def total_cost : ℝ := cost_cat_toy + cost_cage

-- State the theorem
theorem jessica_total_cost : total_cost = 21.95 := by
  sorry

end jessica_total_cost_l14_14955


namespace intersection_A_B_l14_14647

variable (A : Set ℤ) (B : Set ℤ)

-- Define the set A and B
def set_A : Set ℤ := {0, 1, 2}
def set_B : Set ℤ := {x | 1 < x ∧ x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {2} :=
by
  sorry

end intersection_A_B_l14_14647


namespace not_equivalent_expression_l14_14721

/--
Let A, B, C, D be expressions defined as follows:
A := 3 * (x + 2)
B := (-9 * x - 18) / -3
C := (1/3) * (3 * x) + (2/3) * 9
D := (1/3) * (9 * x + 18)

Prove that only C is not equivalent to 3 * x + 6.
-/
theorem not_equivalent_expression (x : ℝ) :
  let A := 3 * (x + 2)
  let B := (-9 * x - 18) / -3
  let C := (1/3) * (3 * x) + (2/3) * 9
  let D := (1/3) * (9 * x + 18)
  C ≠ 3 * x + 6 :=
by
  intros A B C D
  sorry

end not_equivalent_expression_l14_14721


namespace third_side_triangle_max_l14_14100

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l14_14100


namespace problem_l14_14028

-- Condition that defines s and t
def s : ℤ := 4
def t : ℤ := 3

theorem problem (s t : ℤ) (h_s : s = 4) (h_t : t = 3) : s - 2 * t = -2 := by
  sorry

end problem_l14_14028


namespace cube_dot_path_length_l14_14438

noncomputable def length_of_path_traced_by_dot_in_terms_of_d (d : ℝ) : Prop :=
  ∃ c, c = 2 * real.sqrt 2 ∧ d = 2 * real.sqrt 2

theorem cube_dot_path_length :
  ∀ (cube_length : ℝ) (dot_position : (ℝ × ℝ)) (condition1 : cube_length = 2)
  (condition2 : dot_position = (1, 1)) (d : ℝ),
    length_of_path_traced_by_dot_in_terms_of_d d :=
by sorry

end cube_dot_path_length_l14_14438


namespace range_of_a_l14_14758

open Real 

noncomputable def trigonometric_inequality (θ a : ℝ) : Prop :=
  sin (2 * θ) - (2 * sqrt 2 + sqrt 2 * a) * sin (θ + π / 4) - 2 * sqrt 2 / cos (θ - π / 4) > -3 - 2 * a

theorem range_of_a (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → trigonometric_inequality θ a) ↔ (a > 3) :=
sorry

end range_of_a_l14_14758


namespace find_triangle_sides_l14_14925

theorem find_triangle_sides (a : Fin 7 → ℝ) (h : ∀ i, 1 < a i ∧ a i < 13) : 
  ∃ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 7 ∧ 
           a i + a j > a k ∧ 
           a j + a k > a i ∧ 
           a k + a i > a j :=
sorry

end find_triangle_sides_l14_14925


namespace square_area_from_conditions_l14_14060

theorem square_area_from_conditions :
  ∀ (r s l b : ℝ), 
  l = r / 4 →
  r = s →
  l * b = 35 →
  b = 5 →
  s^2 = 784 := 
by 
  intros r s l b h1 h2 h3 h4
  sorry

end square_area_from_conditions_l14_14060


namespace train_speed_l14_14138

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l14_14138


namespace textbook_cost_l14_14635

theorem textbook_cost 
  (credits : ℕ) 
  (cost_per_credit : ℕ) 
  (facility_fee : ℕ) 
  (total_cost : ℕ) 
  (num_textbooks : ℕ) 
  (total_spent : ℕ) 
  (h1 : credits = 14) 
  (h2 : cost_per_credit = 450) 
  (h3 : facility_fee = 200) 
  (h4 : total_spent = 7100) 
  (h5 : num_textbooks = 5) :
  (total_cost - (credits * cost_per_credit + facility_fee)) / num_textbooks = 120 :=
by
  sorry

end textbook_cost_l14_14635


namespace factor_problem_l14_14016

theorem factor_problem (x y m : ℝ) (h : (1 - 2 * x + y) ∣ (4 * x * y - 4 * x^2 - y^2 - m)) :
  m = -1 :=
by
  sorry

end factor_problem_l14_14016


namespace quadratic_double_root_eq1_quadratic_double_root_eq2_l14_14781

theorem quadratic_double_root_eq1 :
  (∃ r : ℝ , ∃ s : ℝ, (r ≠ s) ∧ (
  (1 : ℝ) * r^2 + (-3 : ℝ) * r + (2 : ℝ) = 0 ∧
  (1 : ℝ) * s^2 + (-3 : ℝ) * s + (2 : ℝ) = 0 ∧
  (r = 2 * s ∨ s = 2 * r) 
  )) := 
  sorry

theorem quadratic_double_root_eq2 :
  (∃ a b : ℝ, a ≠ 0 ∧
  ((∃ r : ℝ, (-b / a = 2 + r) ∧ (-6 / a = 2 * r)) ∨ 
  ((-b / a = 2 + 1) ∧ (-6 / a = 2 * 1))) ∧ 
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9))) :=
  sorry

end quadratic_double_root_eq1_quadratic_double_root_eq2_l14_14781


namespace YaoMing_stride_impossible_l14_14292

-- Defining the conditions as Lean definitions.
def XiaoMing_14_years_old (current_year : ℕ) : Prop := current_year = 14
def sum_of_triangle_angles (angles : ℕ) : Prop := angles = 180
def CCTV5_broadcasting_basketball_game : Prop := ∃ t : ℕ, true -- Random event placeholder
def YaoMing_stride (stride_length : ℕ) : Prop := stride_length = 10

-- The main statement: Prove that Yao Ming cannot step 10 meters in one stride.
theorem YaoMing_stride_impossible (h1: ∃ y : ℕ, XiaoMing_14_years_old y) 
                                  (h2: ∃ a : ℕ, sum_of_triangle_angles a) 
                                  (h3: CCTV5_broadcasting_basketball_game) 
: ¬ ∃ s : ℕ, YaoMing_stride s := sorry

end YaoMing_stride_impossible_l14_14292


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2_l14_14259

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2 :
  nat.min_fac (11^5 - 11^4) = 2 :=
by
  sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2_l14_14259


namespace negation_of_universal_l14_14173

theorem negation_of_universal (P : Prop) :
  (¬ (∀ x : ℝ, x > 0 → x^3 > 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^3 ≤ 0) :=
by sorry

end negation_of_universal_l14_14173


namespace solve_equation_l14_14534

theorem solve_equation (x : ℚ) (h : x ≠ 1) : (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by 
  sorry

end solve_equation_l14_14534


namespace last_digit_of_7_to_the_7_l14_14315

theorem last_digit_of_7_to_the_7 :
  (7 ^ 7) % 10 = 3 :=
by
  sorry

end last_digit_of_7_to_the_7_l14_14315


namespace rectangular_field_area_l14_14588

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l14_14588


namespace moles_of_naoh_combined_number_of_moles_of_naoh_combined_l14_14164

-- Define the reaction equation and given conditions
def reaction_equation := "2 NaOH + Cl₂ → NaClO + NaCl + H₂O"

-- Given conditions
def moles_chlorine : ℕ := 2
def moles_water_produced : ℕ := 2
def moles_naoh_needed_for_one_mole_water : ℕ := 2

-- Stoichiometric relationship from the reaction equation
def moles_naoh_per_mole_water : ℕ := 2

-- Theorem to prove the number of moles of NaOH combined
theorem moles_of_naoh_combined (moles_water_produced : ℕ)
  (moles_naoh_per_mole_water : ℕ) : ℕ :=
  moles_water_produced * moles_naoh_per_mole_water

-- Statement of the theorem
theorem number_of_moles_of_naoh_combined : moles_of_naoh_combined 2 2 = 4 :=
by sorry

end moles_of_naoh_combined_number_of_moles_of_naoh_combined_l14_14164


namespace inequality_solution_l14_14701

theorem inequality_solution
  (x : ℝ) :
  x ∉ {2, 3, 4, 5, 6, 7} →
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  (x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) :=
by
  -- Proof goes here
  sorry

end inequality_solution_l14_14701


namespace convex_polygon_with_tiles_l14_14114

variable (n : ℕ)

def canFormConvexPolygon (n : ℕ) : Prop :=
  3 ≤ n ∧ n ≤ 12

theorem convex_polygon_with_tiles (n : ℕ) 
  (square_internal_angle : ℕ := 90) 
  (equilateral_triangle_internal_angle : ℕ := 60)
  (external_angle_step : ℕ := 30)
  (total_external_angle : ℕ := 360) :
  canFormConvexPolygon n :=
by 
  sorry

end convex_polygon_with_tiles_l14_14114


namespace final_score_is_89_l14_14732

def final_score (s_e s_l s_b : ℝ) (p_e p_l p_b : ℝ) : ℝ :=
  s_e * p_e + s_l * p_l + s_b * p_b

theorem final_score_is_89 :
  final_score 95 92 80 0.4 0.25 0.35 = 89 := 
by
  sorry

end final_score_is_89_l14_14732


namespace water_level_drop_l14_14664

theorem water_level_drop :
  (∀ x : ℝ, x > 0 → (x = 4) → (x > 0 → x = 4)) →
  ∃ y : ℝ, y < 0 ∧ (y = -1) :=
by
  sorry

end water_level_drop_l14_14664


namespace intersect_xz_plane_at_point_l14_14757

-- Define points and vectors in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the points A and B
def A : Point3D := ⟨2, -1, 3⟩
def B : Point3D := ⟨6, 7, -2⟩

-- Define the direction vector as the difference between points A and B
def direction_vector (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

-- Function to parameterize the line given a point and direction vector
def parametric_line (P : Point3D) (v : Point3D) (t : ℝ) : Point3D :=
  ⟨P.x + t * v.x, P.y + t * v.y, P.z + t * v.z⟩

-- Define the xz-plane intersection condition (y coordinate should be 0)
def intersects_xz_plane (P : Point3D) (v : Point3D) (t : ℝ) : Prop :=
  (parametric_line P v t).y = 0

-- Define the intersection point as a Point3D
def intersection_point : Point3D := ⟨2.5, 0, 2.375⟩

-- Statement to prove the intersection
theorem intersect_xz_plane_at_point : 
  ∃ t : ℝ, intersects_xz_plane A (direction_vector A B) t ∧ parametric_line A (direction_vector A B) t = intersection_point :=
by
  sorry

end intersect_xz_plane_at_point_l14_14757


namespace piastres_in_6th_purse_l14_14507

-- We define the amounts and constraints
variables (x : ℕ) -- Number of piastres in the first purse
variables (piastres : ℕ) -- Total number of piastres
variables (purse6 : ℕ) -- Number of piastres in the 6th purse

-- Given conditions as variables
axiom total_piastres : 150 = piastres
axiom num_purses : 10
axiom increasing_sequence : ∀ i j : ℕ, i < j → (x + i) < (x + j)
axiom first_last_condition : x ≥ (x + 9) / 2

-- Prove the number of piastres in the 6th purse
theorem piastres_in_6th_purse : purse6 = 16 :=
by
  -- placeholder for proof
  sorry

end piastres_in_6th_purse_l14_14507


namespace intersection_M_N_l14_14336

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14336


namespace total_weight_of_8_moles_of_BaCl2_l14_14714

-- Define atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular weight of BaCl2
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

-- Define the number of moles
def moles : ℝ := 8

-- Define the total weight calculation
def total_weight : ℝ := molecular_weight_BaCl2 * moles

-- The theorem to prove
theorem total_weight_of_8_moles_of_BaCl2 : total_weight = 1665.84 :=
by sorry

end total_weight_of_8_moles_of_BaCl2_l14_14714


namespace medium_bed_rows_l14_14363

theorem medium_bed_rows (large_top_beds : ℕ) (large_bed_rows : ℕ) (large_bed_seeds_per_row : ℕ) 
                         (medium_beds : ℕ) (medium_bed_seeds_per_row : ℕ) (total_seeds : ℕ) :
    large_top_beds = 2 ∧ large_bed_rows = 4 ∧ large_bed_seeds_per_row = 25 ∧
    medium_beds = 2 ∧ medium_bed_seeds_per_row = 20 ∧ total_seeds = 320 →
    ((total_seeds - (large_top_beds * large_bed_rows * large_bed_seeds_per_row)) / medium_bed_seeds_per_row) = 6 :=
by
  intro conditions
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := conditions
  sorry

end medium_bed_rows_l14_14363


namespace find_number_l14_14869

theorem find_number (n : ℝ) : (1 / 2) * n + 6 = 11 → n = 10 := by
  sorry

end find_number_l14_14869


namespace time_to_cross_stationary_train_l14_14447

theorem time_to_cross_stationary_train (t_pole : ℝ) (speed_train : ℝ) (length_stationary_train : ℝ) 
  (t_pole_eq : t_pole = 5) (speed_train_eq : speed_train = 64.8) (length_stationary_train_eq : length_stationary_train = 360) :
  (t_pole * speed_train + length_stationary_train) / speed_train = 10.56 := 
by
  rw [t_pole_eq, speed_train_eq, length_stationary_train_eq]
  norm_num
  sorry

end time_to_cross_stationary_train_l14_14447


namespace students_answered_both_correct_l14_14822

theorem students_answered_both_correct (total_students : ℕ)
  (answered_sets_correctly : ℕ) (answered_functions_correctly : ℕ)
  (both_wrong : ℕ) (total : total_students = 50)
  (sets_correct : answered_sets_correctly = 40)
  (functions_correct : answered_functions_correctly = 31)
  (wrong_both : both_wrong = 4) :
  (40 + 31 - (total_students - 4) + both_wrong = 50) → total_students - (40 + 31 - (total_students - 4)) = 29 :=
by
  sorry

end students_answered_both_correct_l14_14822


namespace root_expr_value_eq_175_div_11_l14_14216

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end root_expr_value_eq_175_div_11_l14_14216


namespace initial_pens_l14_14521

-- Conditions as definitions
def initial_books := 108
def books_after_sale := 66
def books_sold := 42
def pens_after_sale := 59

-- Theorem statement proving the initial number of pens
theorem initial_pens:
  initial_books - books_after_sale = books_sold →
  ∃ (P : ℕ), P - pens_sold = pens_after_sale ∧ (P = 101) :=
by
  sorry

end initial_pens_l14_14521


namespace total_social_media_hours_in_a_week_l14_14751

variable (daily_social_media_hours : ℕ) (days_in_week : ℕ)

theorem total_social_media_hours_in_a_week
(h1 : daily_social_media_hours = 3)
(h2 : days_in_week = 7) :
daily_social_media_hours * days_in_week = 21 := by
  sorry

end total_social_media_hours_in_a_week_l14_14751


namespace train_speed_l14_14140

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l14_14140


namespace find_s_l14_14512

noncomputable def s_value (m : ℝ) : ℝ := m + 16.25

theorem find_s (a b m s : ℝ)
  (h1 : a + b = m) (h2 : a * b = 4) :
  s = s_value m :=
by
  sorry

end find_s_l14_14512


namespace base_7_to_base_10_conversion_l14_14734

theorem base_7_to_base_10_conversion :
  (6 * 7^2 + 5 * 7^1 + 3 * 7^0) = 332 :=
by sorry

end base_7_to_base_10_conversion_l14_14734


namespace shape_area_is_36_l14_14384

def side_length : ℝ := 3
def num_squares : ℕ := 4
def area_square : ℝ := side_length ^ 2
def total_area : ℝ := num_squares * area_square

theorem shape_area_is_36 :
  total_area = 36 := by
  sorry

end shape_area_is_36_l14_14384


namespace find_min_value_l14_14326

theorem find_min_value (a x y : ℝ) (h : y = -x^2 + 3 * Real.log x) : ∃ x, ∃ y, (a - x)^2 + (a + 2 - y)^2 = 8 :=
by
  sorry

end find_min_value_l14_14326


namespace arrange_pencils_l14_14385

-- Definition to express the concept of pencil touching
def pencil_touches (a b : Type) : Prop := sorry

-- Assume we have six pencils represented as 6 distinct variables.
variables (A B C D E F : Type)

-- Main theorem statement
theorem arrange_pencils :
  ∃ (A B C D E F : Type), (pencil_touches A B) ∧ (pencil_touches A C) ∧ 
  (pencil_touches A D) ∧ (pencil_touches A E) ∧ (pencil_touches A F) ∧ 
  (pencil_touches B C) ∧ (pencil_touches B D) ∧ (pencil_touches B E) ∧ 
  (pencil_touches B F) ∧ (pencil_touches C D) ∧ (pencil_touches C E) ∧ 
  (pencil_touches C F) ∧ (pencil_touches D E) ∧ (pencil_touches D F) ∧ 
  (pencil_touches E F) :=
sorry

end arrange_pencils_l14_14385


namespace tan_alpha_add_pi_over_3_l14_14765

theorem tan_alpha_add_pi_over_3 (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5) 
  (h2 : Real.tan (β - π / 3) = 1 / 4) : 
  Real.tan (α + π / 3) = 7 / 23 := 
by
  sorry

end tan_alpha_add_pi_over_3_l14_14765


namespace sum_of_f_values_l14_14967

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f_values :
  (∀ x : ℝ, f x + f (-x) = 0) →
  (∀ x : ℝ, f x = f (x + 2)) →
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = 2^x - 1) →
  f (1/2) + f 1 + f (3/2) + f 2 + f (5/2) = Real.sqrt 2 - 1 :=
by
  intros h1 h2 h3
  sorry

end sum_of_f_values_l14_14967


namespace at_least_one_zero_l14_14524

theorem at_least_one_zero (a b : ℝ) : (¬ (a ≠ 0 ∧ b ≠ 0)) → (a = 0 ∨ b = 0) := by
  intro h
  have h' : ¬ ((a ≠ 0) ∧ (b ≠ 0)) := h
  sorry

end at_least_one_zero_l14_14524


namespace terminating_decimal_representation_l14_14311

-- Definitions derived from conditions
def given_fraction : ℚ := 53 / (2^2 * 5^3)

-- The theorem we aim to state that expresses the question and correct answer
theorem terminating_decimal_representation : given_fraction = 0.106 :=
by
  sorry  -- proof goes here

end terminating_decimal_representation_l14_14311


namespace time_to_reach_ship_l14_14286

-- Conditions in Lean 4
def rate : ℕ := 22
def depth : ℕ := 7260

-- The theorem that we want to prove
theorem time_to_reach_ship : depth / rate = 330 := by
  sorry

end time_to_reach_ship_l14_14286


namespace total_applicants_is_40_l14_14046

def total_applicants (PS GPA_high Not_PS_GPA_low both : ℕ) : ℕ :=
  let PS_or_GPA_high := PS + GPA_high - both 
  PS_or_GPA_high + Not_PS_GPA_low

theorem total_applicants_is_40 :
  total_applicants 15 20 10 5 = 40 :=
by
  sorry

end total_applicants_is_40_l14_14046


namespace quadratic_condition_l14_14779

theorem quadratic_condition (m : ℝ) (h1 : m^2 - 2 = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
by
  sorry

end quadratic_condition_l14_14779


namespace total_distance_trip_l14_14287

-- Defining conditions
def time_paved := 2 -- hours
def time_dirt := 3 -- hours
def speed_dirt := 32 -- mph
def speed_paved := speed_dirt + 20 -- mph

-- Defining distances
def distance_dirt := speed_dirt * time_dirt -- miles
def distance_paved := speed_paved * time_paved -- miles

-- Proving total distance
theorem total_distance_trip : distance_dirt + distance_paved = 200 := by
  sorry

end total_distance_trip_l14_14287


namespace evaluate_expression_at_three_l14_14622

-- Define the evaluation of the expression (x^x)^(x^x) at x=3
theorem evaluate_expression_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end evaluate_expression_at_three_l14_14622


namespace recurring_decimal_of_division_l14_14624

theorem recurring_decimal_of_division (a b : ℤ) (h1 : a = 60) (h2 : b = 55) : (a : ℝ) / (b : ℝ) = 1.09090909090909090909090909090909 :=
by
  -- Import the necessary definitions and facts
  sorry

end recurring_decimal_of_division_l14_14624


namespace tax_computation_l14_14193

def income : ℕ := 56000
def first_portion_income : ℕ := 40000
def first_portion_rate : ℝ := 0.12
def remaining_income : ℕ := income - first_portion_income
def remaining_rate : ℝ := 0.20
def expected_tax : ℝ := 8000

theorem tax_computation :
  (first_portion_rate * first_portion_income) +
  (remaining_rate * remaining_income) = expected_tax := by
  sorry

end tax_computation_l14_14193


namespace range_of_a_l14_14392

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - 2 * x ^ 2

theorem range_of_a (a : ℝ) :
  (∀ x0 : ℝ, 0 < x0 ∧ x0 < 1 →
  (0 < (deriv (fun x => f a x - x)) x0)) →
  a > (4 / Real.exp (3 / 4)) :=
by
  intro h
  sorry

end range_of_a_l14_14392


namespace first_number_in_set_l14_14242

theorem first_number_in_set (x : ℝ)
  (h : (x + 40 + 60) / 3 = (10 + 80 + 15) / 3 + 5) :
  x = 20 := by
  sorry

end first_number_in_set_l14_14242


namespace courtyard_width_l14_14552

theorem courtyard_width 
  (length_of_courtyard : ℝ) 
  (num_paving_stones : ℕ) 
  (length_of_stone width_of_stone : ℝ) 
  (total_area_stone : ℝ) 
  (W : ℝ) : 
  length_of_courtyard = 40 →
  num_paving_stones = 132 →
  length_of_stone = 2.5 →
  width_of_stone = 2 →
  total_area_stone = 660 →
  40 * W = 660 →
  W = 16.5 :=
by
  intros
  sorry

end courtyard_width_l14_14552


namespace chocolates_difference_l14_14973

/-!
We are given that:
- Robert ate 10 chocolates
- Nickel ate 5 chocolates

We need to prove that Robert ate 5 more chocolates than Nickel.
-/

def robert_chocolates := 10
def nickel_chocolates := 5

theorem chocolates_difference : robert_chocolates - nickel_chocolates = 5 :=
by
  -- Proof is omitted as per instructions
  sorry

end chocolates_difference_l14_14973


namespace max_value_of_f_product_of_zeros_l14_14360

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b
 
theorem max_value_of_f (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) : f (1 / a) a b = -Real.log a - 1 + b :=
by
  sorry

theorem product_of_zeros (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) (hx_ne : x1 ≠ x2) : x1 * x2 < 1 / (a * a) :=
by
  sorry

end max_value_of_f_product_of_zeros_l14_14360


namespace remaining_length_after_cut_l14_14126

/- Definitions -/
def original_length (a b : ℕ) : ℕ := 5 * a + 4 * b
def rectangle_perimeter (a b : ℕ) : ℕ := 2 * (a + b)
def remaining_length (a b : ℕ) : ℕ := original_length a b - rectangle_perimeter a b

/- Theorem statement -/
theorem remaining_length_after_cut (a b : ℕ) : remaining_length a b = 3 * a + 2 * b := 
by 
  sorry

end remaining_length_after_cut_l14_14126


namespace birds_after_changes_are_235_l14_14740

-- Define initial conditions for the problem
def initial_cages : Nat := 15
def parrots_per_cage : Nat := 3
def parakeets_per_cage : Nat := 8
def canaries_per_cage : Nat := 5
def parrots_sold : Nat := 5
def canaries_sold : Nat := 2
def parakeets_added : Nat := 2


-- Define the function to count total birds after the changes
def total_birds_after_changes (initial_cages parrots_per_cage parakeets_per_cage canaries_per_cage parrots_sold canaries_sold parakeets_added : Nat) : Nat :=
  let initial_parrots := initial_cages * parrots_per_cage
  let initial_parakeets := initial_cages * parakeets_per_cage
  let initial_canaries := initial_cages * canaries_per_cage
  
  let final_parrots := initial_parrots - parrots_sold
  let final_parakeets := initial_parakeets + parakeets_added
  let final_canaries := initial_canaries - canaries_sold
  
  final_parrots + final_parakeets + final_canaries

-- Prove that the total number of birds is 235
theorem birds_after_changes_are_235 : total_birds_after_changes 15 3 8 5 5 2 2 = 235 :=
  by 
    -- Proof is omitted as per the instructions
    sorry

end birds_after_changes_are_235_l14_14740


namespace find_time_when_velocity_is_one_l14_14120

-- Define the equation of motion
def equation_of_motion (t : ℝ) : ℝ := 7 * t^2 + 8

-- Define the velocity function as the derivative of the equation of motion
def velocity (t : ℝ) : ℝ := by
  let s := equation_of_motion t
  exact 14 * t  -- Since we calculated the derivative above

-- Statement of the problem to be proved
theorem find_time_when_velocity_is_one : 
  (velocity (1 / 14)) = 1 :=
by
  -- Placeholder for the proof
  sorry

end find_time_when_velocity_is_one_l14_14120


namespace measure_of_each_interior_angle_l14_14839

theorem measure_of_each_interior_angle (n : ℕ) (hn : 3 ≤ n) : 
  ∃ angle : ℝ, angle = (n - 2) * 180 / n :=
by
  sorry

end measure_of_each_interior_angle_l14_14839


namespace intersection_M_N_l14_14329

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14329


namespace probability_queen_in_center_after_2004_moves_l14_14194

def initial_probability (n : ℕ) : ℚ :=
if n = 0 then 1
else if n = 1 then 0
else if n % 2 = 0 then (1 : ℚ) / 2^(n / 2)
else (1 - (1 : ℚ) / 2^((n - 1) / 2)) / 2

theorem probability_queen_in_center_after_2004_moves :
  initial_probability 2004 = 1 / 3 + 1 / (3 * 2^2003) :=
sorry

end probability_queen_in_center_after_2004_moves_l14_14194


namespace find_divisor_l14_14726

-- Define the conditions
def dividend : ℕ := 22
def quotient : ℕ := 7
def remainder : ℕ := 1

-- The divisor is what we need to find
def divisor : ℕ := 3

-- The proof problem: proving that the given conditions imply the divisor is 3
theorem find_divisor :
  ∃ d : ℕ, dividend = d * quotient + remainder ∧ d = divisor :=
by
  use 3
  -- Replace actual proof with sorry for now
  sorry

end find_divisor_l14_14726


namespace sequence_b_n_l14_14308

theorem sequence_b_n (b : ℕ → ℝ) (h₁ : b 1 = 2) (h₂ : ∀ n, (b (n + 1))^3 = 64 * (b n)^3) : 
    b 50 = 2 * 4^49 :=
sorry

end sequence_b_n_l14_14308


namespace sum_of_edges_proof_l14_14418

noncomputable def sum_of_edges (a r : ℝ) : ℝ :=
  let l1 := a / r
  let l2 := a
  let l3 := a * r
  4 * (l1 + l2 + l3)

theorem sum_of_edges_proof : 
  ∀ (a r : ℝ), 
  (a > 0 ∧ r > 0 ∧ (a / r) * a * (a * r) = 512 ∧ 2 * ((a^2 / r) + a^2 + a^2 * r) = 384) → sum_of_edges a r = 96 :=
by
  intros a r h
  -- We skip the proof here with sorry
  sorry

end sum_of_edges_proof_l14_14418


namespace inequality_solution_l14_14759

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := abs ((7 - 2 * x) / 4) < 3

-- Define the correct answer as an interval
def correct_interval (x : ℝ) : Prop := -2.5 < x ∧ x < 9.5

-- The theorem states that the inequality condition and correct interval are equivalent
theorem inequality_solution (x : ℝ) : inequality_condition x ↔ correct_interval x := 
sorry

end inequality_solution_l14_14759


namespace initial_oranges_l14_14128

theorem initial_oranges (X : ℕ) : 
  (X - 9 + 38 = 60) → X = 31 :=
sorry

end initial_oranges_l14_14128


namespace beret_count_l14_14953

/-- James can make a beret from 3 spools of yarn. 
    He has 12 spools of red yarn, 15 spools of black yarn, and 6 spools of blue yarn.
    Prove that he can make 11 berets in total. -/
theorem beret_count (red_yarn : ℕ) (black_yarn : ℕ) (blue_yarn : ℕ) (spools_per_beret : ℕ) 
  (total_yarn : ℕ) (num_berets : ℕ) (h1 : red_yarn = 12) (h2 : black_yarn = 15) (h3 : blue_yarn = 6)
  (h4 : spools_per_beret = 3) (h5 : total_yarn = red_yarn + black_yarn + blue_yarn) 
  (h6 : num_berets = total_yarn / spools_per_beret) : 
  num_berets = 11 :=
by sorry

end beret_count_l14_14953


namespace roots_sum_roots_product_algebraic_expression_l14_14643

theorem roots_sum (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 + x2 = 1 :=
sorry

theorem roots_product (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 * x2 = -1 :=
sorry

theorem algebraic_expression (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1^2 + x2^2 = 3 :=
sorry

end roots_sum_roots_product_algebraic_expression_l14_14643


namespace circle_convex_polygons_count_l14_14752

theorem circle_convex_polygons_count : 
  let total_subsets := (2^15 - 1) - (15 + 105 + 455 + 255)
  let final_count := total_subsets - 500
  final_count = 31437 :=
by
  sorry

end circle_convex_polygons_count_l14_14752


namespace fitted_bowling_ball_volume_correct_l14_14277

noncomputable def volume_of_fitted_bowling_ball : ℝ :=
  let ball_radius := 12
  let ball_volume := (4/3) * Real.pi * ball_radius^3
  let hole1_radius := 1
  let hole1_volume := Real.pi * hole1_radius^2 * 6
  let hole2_radius := 1.25
  let hole2_volume := Real.pi * hole2_radius^2 * 6
  let hole3_radius := 2
  let hole3_volume := Real.pi * hole3_radius^2 * 6
  ball_volume - (hole1_volume + hole2_volume + hole3_volume)

theorem fitted_bowling_ball_volume_correct :
  volume_of_fitted_bowling_ball = 2264.625 * Real.pi := by
  -- proof would go here
  sorry

end fitted_bowling_ball_volume_correct_l14_14277


namespace max_diff_y_l14_14935

theorem max_diff_y (x y z : ℕ) (h₁ : 4 < x) (h₂ : x < z) (h₃ : z < y) (h₄ : y < 10) (h₅ : y - x = 5) : y = 9 :=
sorry

end max_diff_y_l14_14935


namespace percentage_less_than_l14_14942

variable (x y : ℝ)
variable (H : y = 1.4 * x)

theorem percentage_less_than :
  ((y - x) / y) * 100 = 28.57 := by
  sorry

end percentage_less_than_l14_14942


namespace polygon_sides_l14_14482

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l14_14482


namespace arithmetic_sequence_n_is_17_l14_14795

theorem arithmetic_sequence_n_is_17
  (a : ℕ → ℤ)  -- An arithmetic sequence a_n
  (h1 : a 1 = 5)  -- First term is 5
  (h5 : a 5 = -3)  -- Fifth term is -3
  (hn : a n = -27) : n = 17 := sorry

end arithmetic_sequence_n_is_17_l14_14795


namespace percentage_volume_occupied_is_100_l14_14444

-- Define the dimensions of the box and cube
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 2

-- Define the volumes
def box_volume : ℕ := box_length * box_width * box_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the number of cubes that fit in each dimension
def cubes_along_length : ℕ := box_length / cube_side
def cubes_along_width : ℕ := box_width / cube_side
def cubes_along_height : ℕ := box_height / cube_side

-- Define the total number of cubes and the volume they occupy
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height
def volume_occupied_by_cubes : ℕ := total_cubes * cube_volume

-- Define the percentage of the box volume occupied by the cubes
def percentage_volume_occupied : ℕ := (volume_occupied_by_cubes * 100) / box_volume

-- Statement to prove
theorem percentage_volume_occupied_is_100 : percentage_volume_occupied = 100 := by
  sorry

end percentage_volume_occupied_is_100_l14_14444


namespace inscribed_circle_radius_of_triangle_l14_14317

theorem inscribed_circle_radius_of_triangle (a b c : ℕ)
  (h₁ : a = 50) (h₂ : b = 120) (h₃ : c = 130) :
  ∃ r : ℕ, r = 20 :=
by sorry

end inscribed_circle_radius_of_triangle_l14_14317


namespace boxes_needed_l14_14828

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end boxes_needed_l14_14828


namespace factorization_theorem_l14_14168

-- Define the polynomial p(x, y)
def p (x y k : ℝ) : ℝ := x^2 - 2*x*y + k*y^2 + 3*x - 5*y + 2

-- Define the condition for factorization into two linear factors
def can_be_factored (x y m n : ℝ) : Prop :=
  (p x y (m * n)) = ((x + m * y + 1) * (x + n * y + 2))

-- The main theorem proving that k = -3 is the value for factorizability
theorem factorization_theorem (k : ℝ) : (∃ m n : ℝ, can_be_factored x y m n) ↔ k = -3 := by sorry

end factorization_theorem_l14_14168


namespace find_average_of_xyz_l14_14040

variable (x y z k : ℝ)

def system_of_equations : Prop :=
  (2 * x + y - z = 26) ∧
  (x + 2 * y + z = 10) ∧
  (x - y + z = k)

theorem find_average_of_xyz (h : system_of_equations x y z k) : 
  (x + y + z) / 3 = (36 + k) / 6 :=
by sorry

end find_average_of_xyz_l14_14040


namespace tangent_line_with_smallest_slope_l14_14906

-- Define the given curve
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the given curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the equation of the tangent line with the smallest slope
def tangent_line (x y : ℝ) : Prop := 3 * x - y = 11

-- Prove that the equation of the tangent line with the smallest slope on the curve is 3x - y - 11 = 0
theorem tangent_line_with_smallest_slope :
  ∃ x y : ℝ, curve x = y ∧ curve_derivative x = 3 ∧ tangent_line x y :=
by
  sorry

end tangent_line_with_smallest_slope_l14_14906


namespace non_prime_in_sequence_l14_14235

theorem non_prime_in_sequence : ∃ n : ℕ, ¬Prime (41 + n * (n - 1)) :=
by {
  use 41,
  sorry
}

end non_prime_in_sequence_l14_14235


namespace triangle_third_side_l14_14093

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l14_14093


namespace cats_to_dogs_ratio_l14_14414

theorem cats_to_dogs_ratio
    (cats dogs : ℕ)
    (ratio : cats / dogs = 3 / 4)
    (num_cats : cats = 18) :
    dogs = 24 :=
by
    sorry

end cats_to_dogs_ratio_l14_14414


namespace total_pages_read_l14_14520

theorem total_pages_read (days : ℕ)
  (deshaun_books deshaun_pages_per_book lilly_percent ben_extra eva_factor sam_pages_per_day : ℕ)
  (lilly_percent_correct : lilly_percent = 75)
  (ben_extra_correct : ben_extra = 25)
  (eva_factor_correct : eva_factor = 2)
  (total_break_days : days = 80)
  (deshaun_books_correct : deshaun_books = 60)
  (deshaun_pages_per_book_correct : deshaun_pages_per_book = 320)
  (sam_pages_per_day_correct : sam_pages_per_day = 150) :
  deshaun_books * deshaun_pages_per_book +
  (lilly_percent * deshaun_books * deshaun_pages_per_book / 100) +
  (deshaun_books * (100 + ben_extra) / 100) * 280 +
  (eva_factor * (deshaun_books * (100 + ben_extra) / 100 * 280)) +
  (sam_pages_per_day * days) = 108450 := 
sorry

end total_pages_read_l14_14520


namespace distinct_real_numbers_sum_l14_14219

theorem distinct_real_numbers_sum:
  ∀ (p q r s : ℝ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    (r + s = 12 * p) →
    (r * s = -13 * q) →
    (p + q = 12 * r) →
    (p * q = -13 * s) →
    p + q + r + s = 2028 :=
by
  intros p q r s h_distinct h1 h2 h3 h4
  sorry

end distinct_real_numbers_sum_l14_14219


namespace total_marbles_l14_14020

-- Definitions based on given conditions
def ratio_white := 2
def ratio_purple := 3
def ratio_red := 5
def ratio_blue := 4
def ratio_green := 6
def blue_marbles := 24

-- Definition of sum of ratio parts
def sum_of_ratio_parts := ratio_white + ratio_purple + ratio_red + ratio_blue + ratio_green

-- Definition of ratio of blue marbles to total
def ratio_blue_to_total := ratio_blue / sum_of_ratio_parts

-- Proof goal: total number of marbles
theorem total_marbles : blue_marbles / ratio_blue_to_total = 120 := by
  sorry

end total_marbles_l14_14020


namespace find_sum_due_l14_14705

variable (BD TD FV : ℝ)

-- given conditions
def condition_1 : Prop := BD = 80
def condition_2 : Prop := TD = 70
def condition_3 : Prop := BD = TD + (TD * BD / FV)

-- goal statement
theorem find_sum_due (h1 : condition_1 BD) (h2 : condition_2 TD) (h3 : condition_3 BD TD FV) : FV = 560 :=
by
  sorry

end find_sum_due_l14_14705


namespace set_intersection_l14_14353

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l14_14353


namespace complex_equation_solution_l14_14614

open Complex

theorem complex_equation_solution (x y : ℝ) :
  ((-5 + 2 * I) * (x: ℂ) - (3 - 4 * I) * (y: ℂ) = 2 - I) ↔ 
  (x = -5 / 14 ∧ y = -1 / 14) :=
by
  sorry

end complex_equation_solution_l14_14614


namespace perimeter_of_equilateral_triangle_l14_14443

theorem perimeter_of_equilateral_triangle : 
  ∃ (L : ℝ → ℝ), L 0 = 0 ∧ (∃ x1, L x1 = (1 + real.sqrt 3 / 3 * x1)) ∧ 
  (∃ y1, y1 = 1 + real.sqrt 3 / 3) ∧ 
  (∃ y2, y2 = -real.sqrt 3 / 3) ∧ 
  (∀ x, L x = -real.sqrt 3 / 3 * x) ∧ 
  (∀ x1 x2 y1 y2, x1 = x2 ∧ x1 = 1 ∧ 
  L x1 = y2 ∧ y1 = y2 + 1 + real.sqrt 3 / 3 ∧ 
  3 * (1 + 2 * real.sqrt 3 / 3) = 3 + 2 * real.sqrt 3) :=
begin
  sorry
end

end perimeter_of_equilateral_triangle_l14_14443


namespace area_of_paper_l14_14285

-- Define the variables and conditions
variable (L W : ℝ)
variable (h1 : 2 * L + 4 * W = 34)
variable (h2 : 4 * L + 2 * W = 38)

-- Statement to prove
theorem area_of_paper : L * W = 35 := 
by
  sorry

end area_of_paper_l14_14285


namespace triangle_third_side_l14_14096

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l14_14096


namespace binary_equals_octal_l14_14461

-- Define the binary number 1001101 in decimal
def binary_1001101_decimal : ℕ := 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the octal number 115 in decimal
def octal_115_decimal : ℕ := 1 * 8^2 + 1 * 8^1 + 5 * 8^0

-- Theorem statement
theorem binary_equals_octal :
  binary_1001101_decimal = octal_115_decimal :=
sorry

end binary_equals_octal_l14_14461


namespace calc_angle_CAB_l14_14245

theorem calc_angle_CAB (α β γ ε : ℝ) (hα : α = 79) (hβ : β = 63) (hγ : γ = 131) (hε : ε = 123.5) : 
  ∃ φ : ℝ, φ = 24 + 52 / 60 :=
by
  sorry

end calc_angle_CAB_l14_14245


namespace car_mpg_in_city_l14_14565

theorem car_mpg_in_city:
  ∃ (h c T : ℝ), 
    (420 = h * T) ∧ 
    (336 = c * T) ∧ 
    (c = h - 6) ∧ 
    (c = 24) :=
by
  sorry

end car_mpg_in_city_l14_14565


namespace shorts_cost_l14_14427

theorem shorts_cost :
  let football_cost := 3.75
  let shoes_cost := 11.85
  let zachary_money := 10
  let additional_needed := 8
  ∃ S, football_cost + shoes_cost + S = zachary_money + additional_needed ∧ S = 2.40 :=
by
  sorry

end shorts_cost_l14_14427


namespace linear_function_quadrants_l14_14282

theorem linear_function_quadrants (k : ℝ) :
  (∀ x y : ℝ, y = (k + 1) * x + k - 2 → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) ↔ (-1 < k ∧ k < 2) := 
sorry

end linear_function_quadrants_l14_14282


namespace solve_system_l14_14406

theorem solve_system :
  ∃ x y : ℝ, (x^2 - 9 * y^2 = 0 ∧ 2 * x - 3 * y = 6) ∧ (x = 6 ∧ y = 2) ∨ (x = 2 ∧ y = -2 / 3) :=
by
  -- The proof will go here
  sorry

end solve_system_l14_14406


namespace hire_applicant_A_l14_14877

-- Define the test scores for applicants A and B
def education_A := 7
def experience_A := 8
def attitude_A := 9

def education_B := 10
def experience_B := 7
def attitude_B := 8

-- Define the weights for the test items
def weight_education := 1 / 6
def weight_experience := 2 / 6
def weight_attitude := 3 / 6

-- Define the final scores
def final_score_A := education_A * weight_education + experience_A * weight_experience + attitude_A * weight_attitude
def final_score_B := education_B * weight_education + experience_B * weight_experience + attitude_B * weight_attitude

-- Prove that Applicant A is hired because their final score is higher
theorem hire_applicant_A : final_score_A > final_score_B :=
by sorry

end hire_applicant_A_l14_14877


namespace find_possible_values_l14_14665
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def satisfies_conditions (a bc de fg : ℕ) : Prop :=
  (a % 2 = 0) ∧ (is_prime bc) ∧ (de % 5 = 0) ∧ (fg % 3 = 0) ∧
  (fg - de = de - bc) ∧ (de - bc = bc - a)

theorem find_possible_values :
  ∃ (debc1 debc2 : ℕ),
    (satisfies_conditions 6 (debc1 % 100) ((debc1 / 100) % 100) ((debc1 / 10000) % 100)) ∧
    (satisfies_conditions 6 (debc2 % 100) ((debc2 / 100) % 100) ((debc2 / 10000) % 100)) ∧
    (debc1 = 2013 ∨ debc1 = 4023) ∧
    (debc2 = 2013 ∨ debc2 = 4023) :=
  sorry

end find_possible_values_l14_14665


namespace find_naturals_for_divisibility_l14_14466

theorem find_naturals_for_divisibility (n : ℕ) (h1 : 3 * n ≠ 1) :
  (∃ k : ℤ, 7 * n + 5 = k * (3 * n - 1)) ↔ n = 1 ∨ n = 4 := 
by
  sorry

end find_naturals_for_divisibility_l14_14466


namespace find_length_of_polaroid_l14_14244

theorem find_length_of_polaroid 
  (C : ℝ) (W : ℝ) (L : ℝ)
  (hC : C = 40) (hW : W = 8) 
  (hFormula : C = 2 * (L + W)) : 
  L = 12 :=
by
  sorry

end find_length_of_polaroid_l14_14244


namespace prove_function_domain_l14_14834

def function_domain := {x : ℝ | (x + 4 ≥ 0 ∧ x ≠ 0)}

theorem prove_function_domain :
  function_domain = {x : ℝ | x ∈ (Set.Icc (-4:ℝ) 0).diff ({0}:Set ℝ) ∪ (Set.Ioi 0)} :=
by
  sorry

end prove_function_domain_l14_14834


namespace rectangular_to_polar_coordinates_l14_14151

theorem rectangular_to_polar_coordinates :
  ∀ (x y : ℝ) (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ x = 2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 →
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) →
  (x, y) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) →
  (r, θ) = (4, Real.pi / 4) :=
by
  intros x y r θ h1 h2 h3
  sorry

end rectangular_to_polar_coordinates_l14_14151


namespace ascending_order_l14_14325

theorem ascending_order (a b : ℝ) (ha : a < 0) (hb1 : -1 < b) (hb2 : b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end ascending_order_l14_14325


namespace license_plates_count_l14_14569

noncomputable def num_license_plates : Nat :=
  let num_w := 26 * 26      -- number of combinations for w
  let num_w_orders := 2     -- two possible orders for w
  let num_digits := 10 ^ 5  -- number of combinations for 5 digits
  let num_positions := 6    -- number of valid positions for w
  2 * num_positions * num_digits * num_w

theorem license_plates_count : num_license_plates = 809280000 := by
  sorry

end license_plates_count_l14_14569


namespace expression_positive_l14_14188

theorem expression_positive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 5 * a ^ 2 - 6 * a * b + 5 * b ^ 2 > 0 :=
by
  sorry

end expression_positive_l14_14188


namespace piglet_straws_l14_14420

theorem piglet_straws (total_straws : ℕ) (straws_adult_pigs_ratio : ℚ) (straws_piglets_ratio : ℚ) (number_piglets : ℕ) :
  total_straws = 300 →
  straws_adult_pigs_ratio = 3/5 →
  straws_piglets_ratio = 1/3 →
  number_piglets = 20 →
  (total_straws * straws_piglets_ratio) / number_piglets = 5 := 
by
  intros
  sorry

end piglet_straws_l14_14420


namespace energy_conservation_l14_14228

-- Define the conditions
variables (m : ℝ) (v_train v_ball : ℝ)
-- The speed of the train and the ball, converted to m/s
variables (v := 60 * 1000 / 3600) -- 60 km/h in m/s
variables (E_initial : ℝ := 0.5 * m * (v ^ 2))

-- Kinetic energy of the ball when thrown in the same direction
variables (E_same_direction : ℝ := 0.5 * m * (2 * v)^2)

-- Kinetic energy of the ball when thrown in the opposite direction
variables (E_opposite_direction : ℝ := 0.5 * m * (0)^2)

-- Prove energy conservation
theorem energy_conservation : 
  (E_same_direction - E_initial) + (E_opposite_direction - E_initial) = 0 :=
sorry

end energy_conservation_l14_14228


namespace roots_imply_value_l14_14211

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end roots_imply_value_l14_14211


namespace greatest_third_side_l14_14102

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l14_14102


namespace carrie_pants_l14_14298

theorem carrie_pants (P : ℕ) (shirts := 4) (pants := P) (jackets := 2)
  (shirt_cost := 8) (pant_cost := 18) (jacket_cost := 60)
  (total_cost := shirts * shirt_cost + jackets * jacket_cost + pants * pant_cost)
  (total_cost_half := 94) :
  total_cost = 188 → total_cost_half = 94 → total_cost = 2 * total_cost_half → P = 2 :=
by
  intros h_total h_half h_relation
  sorry

end carrie_pants_l14_14298


namespace smallest_digit_to_correct_l14_14704

def incorrect_sum : ℕ := 2104
def correct_sum : ℕ := 738 + 625 + 841
def difference : ℕ := correct_sum - incorrect_sum

theorem smallest_digit_to_correct (d : ℕ) (h : difference = 100) :
  d = 6 := 
sorry

end smallest_digit_to_correct_l14_14704


namespace inequality_solution_l14_14700

theorem inequality_solution
  (x : ℝ) :
  x ∉ {2, 3, 4, 5, 6, 7} →
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  (x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) :=
by
  -- Proof goes here
  sorry

end inequality_solution_l14_14700


namespace tangent_line_min_slope_l14_14905

theorem tangent_line_min_slope {x y : ℝ} (hx : y = x^3 + 3 * x^2 + 6 * x - 10) :
  ∃ x : ℝ, ∃ y : ℝ, ∃ m : ℝ, ∃ b : ℝ, (m = 3) ∧ (y = m * (x + 1) - 14) ∧ (3 * x - y - 11 = 0).
proof
  sorry

end tangent_line_min_slope_l14_14905


namespace parallel_lines_not_coincident_l14_14015

theorem parallel_lines_not_coincident (x y : ℝ) (m : ℝ) :
  (∀ y, x + (1 + m) * y = 2 - m ∧ ∀ y, m * x + 2 * y + 8 = 0) → (m =1) := 
sorry

end parallel_lines_not_coincident_l14_14015


namespace sum_of_integers_remainders_l14_14424

theorem sum_of_integers_remainders (a b c : ℕ) :
  (a % 15 = 11) →
  (b % 15 = 13) →
  (c % 15 = 14) →
  ((a + b + c) % 15 = 8) ∧ ((a + b + c) % 10 = 8) :=
by
  sorry

end sum_of_integers_remainders_l14_14424


namespace michelle_will_have_four_crayons_l14_14517

def michelle_crayons (m j : ℕ) : ℕ := m + j

theorem michelle_will_have_four_crayons (H₁ : michelle_crayons 2 2 = 4) : michelle_crayons 2 2 = 4 :=
by
  sorry

end michelle_will_have_four_crayons_l14_14517


namespace g_at_minus_six_l14_14962

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end g_at_minus_six_l14_14962


namespace ratio_of_cream_l14_14388

theorem ratio_of_cream (coffee_init : ℕ) (joe_coffee_drunk : ℕ) (cream_added : ℕ) (joann_total_drunk : ℕ) 
  (joann_coffee_init : ℕ := coffee_init)
  (joe_coffee_init : ℕ := coffee_init) (joann_cream_init : ℕ := cream_added)
  (joe_cream_init : ℕ := cream_added)
  (joann_drunk_cream_ratio : ℚ := joann_cream_init / (joann_coffee_init + joann_cream_init)) :
  (joe_cream_init / (joann_cream_init - joann_total_drunk * (joann_drunk_cream_ratio))) = 
  (6 / 5) := 
by
  sorry

end ratio_of_cream_l14_14388


namespace determine_f_when_alpha_l14_14355

noncomputable def solves_functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
∀ (x y : ℝ), 0 < x → 0 < y → f (f x + y) = α * x + 1 / (f (1 / y))

theorem determine_f_when_alpha (α : ℝ) (f : ℝ → ℝ) :
  (α = 1 → ∀ x, 0 < x → f x = x) ∧ (α ≠ 1 → ∀ f, ¬ solves_functional_equation f α) := by
  sorry

end determine_f_when_alpha_l14_14355


namespace projection_of_vector_a_on_b_l14_14649

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / norm_b

theorem projection_of_vector_a_on_b
  (a b : ℝ × ℝ) 
  (ha : Real.sqrt (a.1^2 + a.2^2) = 1)
  (hb : Real.sqrt (b.1^2 + b.2^2) = 2)
  (theta : ℝ)
  (h_theta : theta = Real.pi * (5/6)) -- 150 degrees in radians
  (h_cos_theta : Real.cos theta = -(Real.sqrt 3 / 2)) :
  vector_projection a b = -Real.sqrt 3 / 2 := 
by
  sorry

end projection_of_vector_a_on_b_l14_14649


namespace ab_greater_than_a_plus_b_l14_14048

variable {a b : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (h : a - b = a / b)

theorem ab_greater_than_a_plus_b : a * b > a + b :=
sorry

end ab_greater_than_a_plus_b_l14_14048


namespace intersection_M_N_l14_14331

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14331


namespace star_intersections_l14_14232

theorem star_intersections (n k : ℕ) (h_coprime : Nat.gcd n k = 1) (h_n_ge_5 : 5 ≤ n) (h_k_lt_n_div_2 : k < n / 2) :
    k = 25 → n = 2018 → n * (k - 1) = 48432 := by
  intros
  sorry

end star_intersections_l14_14232


namespace find_f_of_2_l14_14764

variable (f : ℝ → ℝ)

def functional_equation_condition :=
  ∀ x : ℝ, f (f (f x)) + 3 * f (f x) + 9 * f x + 27 * x = 0

theorem find_f_of_2
  (h : functional_equation_condition f) :
  f (f (f (f 2))) = 162 :=
sorry

end find_f_of_2_l14_14764


namespace colten_chickens_l14_14529

variable (C Q S : ℕ)

-- Conditions
def condition1 : Prop := Q + S + C = 383
def condition2 : Prop := Q = 2 * S + 25
def condition3 : Prop := S = 3 * C - 4

-- Theorem to prove
theorem colten_chickens : condition1 C Q S ∧ condition2 C Q S ∧ condition3 C Q S → C = 37 := by
  sorry

end colten_chickens_l14_14529


namespace trapezoid_combined_area_correct_l14_14300

noncomputable def combined_trapezoid_area_proof : Prop :=
  let EF : ℝ := 60
  let GH : ℝ := 40
  let altitude_EF_GH : ℝ := 18
  let trapezoid_EFGH_area : ℝ := (1 / 2) * (EF + GH) * altitude_EF_GH

  let IJ : ℝ := 30
  let KL : ℝ := 25
  let altitude_IJ_KL : ℝ := 10
  let trapezoid_IJKL_area : ℝ := (1 / 2) * (IJ + KL) * altitude_IJ_KL

  let combined_area : ℝ := trapezoid_EFGH_area + trapezoid_IJKL_area

  combined_area = 1175

theorem trapezoid_combined_area_correct : combined_trapezoid_area_proof := by
  sorry

end trapezoid_combined_area_correct_l14_14300


namespace find_x_when_y_is_6_l14_14429

-- Condition for inverse variation
def inverse_var (k y : ℝ) (x : ℝ) : Prop := x = k / y^2

-- Given values
def given_value_x : ℝ := 1
def given_value_y : ℝ := 2
def new_value_y : ℝ := 6

-- The theorem to prove
theorem find_x_when_y_is_6 :
  ∃ k, inverse_var k given_value_y given_value_x → inverse_var k new_value_y (1/9) :=
by
  sorry

end find_x_when_y_is_6_l14_14429


namespace area_of_rectangle_l14_14576

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l14_14576


namespace combined_future_value_l14_14295

noncomputable def future_value (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem combined_future_value :
  let A1 := future_value 3000 0.05 3
  let A2 := future_value 5000 0.06 4
  let A3 := future_value 7000 0.07 5
  A1 + A2 + A3 = 19603.119 :=
by
  sorry

end combined_future_value_l14_14295


namespace eagles_per_section_l14_14146

theorem eagles_per_section (total_eagles sections : ℕ) (h1 : total_eagles = 18) (h2 : sections = 3) :
  total_eagles / sections = 6 := by
  sorry

end eagles_per_section_l14_14146


namespace toothpicks_at_20th_stage_l14_14710

def toothpicks_in_stage (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

theorem toothpicks_at_20th_stage : toothpicks_in_stage 20 = 61 :=
by 
  sorry

end toothpicks_at_20th_stage_l14_14710


namespace circle_radius_squared_l14_14788

-- Let r be the radius of the circle.
-- Let AB and CD be chords of the circle with lengths 10 and 7 respectively.
-- Let the extensions of AB and CD intersect at a point P outside the circle.
-- Let ∠APD be 60 degrees.
-- Let BP be 8.

theorem circle_radius_squared
  (r : ℝ)       -- radius of the circle
  (AB : ℝ)     -- length of chord AB
  (CD : ℝ)     -- length of chord CD
  (APD : ℝ)    -- angle APD
  (BP : ℝ)     -- length of segment BP
  (hAB : AB = 10)
  (hCD : CD = 7)
  (hAPD : APD = 60)
  (hBP : BP = 8)
  : r^2 = 73 := 
  sorry

end circle_radius_squared_l14_14788


namespace problem_l14_14220

noncomputable def nums : Type := { p q r s : ℝ // p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s }

theorem problem (n : nums) :
  let p := n.1
      q := n.2.1
      r := n.2.2.1
      s := n.2.2.2.1
  in (r + s = 12 * p) → (r * s = -13 * q) → (p + q = 12 * r) → (p * q = -13 * s) → p + q + r + s = 2028 :=
by
  intros
  sorry

end problem_l14_14220


namespace probability_of_head_l14_14946

def events : Type := {e // e = "H" ∨ e = "T"}

def equallyLikely (e : events) : Prop :=
  e = ⟨"H", Or.inl rfl⟩ ∨ e = ⟨"T", Or.inr rfl⟩

def totalOutcomes := 2

def probOfHead : ℚ := 1 / totalOutcomes

theorem probability_of_head : probOfHead = 1 / 2 :=
by
  sorry

end probability_of_head_l14_14946


namespace correct_answer_l14_14936

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + m * x - 1

theorem correct_answer (m : ℝ) : 
  (∀ x₁ x₂, 1 < x₁ → 1 < x₂ → (f x₁ m - f x₂ m) / (x₁ - x₂) > 0) → m ≥ -4 :=
by
  sorry

end correct_answer_l14_14936


namespace find_y_positive_monotone_l14_14167

noncomputable def y (y : ℝ) : Prop :=
  0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 ∧ y = 12

theorem find_y_positive_monotone : ∃ y : ℝ, 0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 := by
  sorry

end find_y_positive_monotone_l14_14167


namespace smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l14_14030

theorem smallest_number_after_operations_n_111 :
  ∀ (n : ℕ), n = 111 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 111 →
       (f l) = 0)) :=
by 
  sorry

theorem smallest_number_after_operations_n_110 :
  ∀ (n : ℕ), n = 110 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 110 →
       (f l) = 1)) :=
by 
  sorry

end smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l14_14030


namespace meryll_questions_l14_14685

theorem meryll_questions :
  let total_mc := 35
  let total_ps := 15
  let written_mc := (2/5 : ℝ) * total_mc
  let written_ps := (1/3 : ℝ) * total_ps
  let remaining_mc := total_mc - written_mc
  let remaining_ps := total_ps - written_ps
  remaining_mc + remaining_ps = 31 :=
by
  let total_mc := 35
  let total_ps := 15
  let written_mc := (2/5 : ℝ) * total_mc
  let written_ps := (1/3 : ℝ) * total_ps
  let remaining_mc := total_mc - written_mc
  let remaining_ps := total_ps - written_ps
  have h1 : remaining_mc = 21 := by sorry
  have h2 : remaining_ps = 10 := by sorry
  show remaining_mc + remaining_ps = 31 from by sorry

end meryll_questions_l14_14685


namespace probability_of_drawing_red_ball_l14_14197

/-- Define the colors of the balls in the bag -/
def yellow_balls : ℕ := 2
def red_balls : ℕ := 3
def white_balls : ℕ := 5

/-- Define the total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + red_balls + white_balls

/-- Define the probability of drawing exactly one red ball -/
def probability_of_red_ball : ℚ := red_balls / total_balls

/-- The main theorem to prove the given problem -/
theorem probability_of_drawing_red_ball :
  probability_of_red_ball = 3 / 10 :=
by
  -- Calculation steps would go here, but are omitted
  sorry

end probability_of_drawing_red_ball_l14_14197


namespace mass_of_sodium_acetate_formed_l14_14617

-- Define the reaction conditions and stoichiometry
def initial_moles_acetic_acid : ℝ := 3
def initial_moles_sodium_hydroxide : ℝ := 4
def initial_reaction_moles_acetic_acid_with_sodium_carbonate : ℝ := 2
def initial_reaction_moles_sodium_carbonate : ℝ := 1
def product_moles_sodium_acetate_from_step1 : ℝ := 2
def remaining_moles_acetic_acid : ℝ := initial_moles_acetic_acid - initial_reaction_moles_acetic_acid_with_sodium_carbonate
def product_moles_sodium_acetate_from_step2 : ℝ := remaining_moles_acetic_acid
def total_moles_sodium_acetate : ℝ := product_moles_sodium_acetate_from_step1 + product_moles_sodium_acetate_from_step2
def molar_mass_sodium_acetate : ℝ := 82.04

-- Translate to the equivalent proof problem
theorem mass_of_sodium_acetate_formed :
  total_moles_sodium_acetate * molar_mass_sodium_acetate = 246.12 :=
by
  -- The detailed proof steps would go here
  sorry

end mass_of_sodium_acetate_formed_l14_14617


namespace num_unpainted_cubes_l14_14879

theorem num_unpainted_cubes (n : ℕ) (h1 : n ^ 3 = 125) : (n - 2) ^ 3 = 27 :=
by
  sorry

end num_unpainted_cubes_l14_14879


namespace train_speed_is_72_km_per_hr_l14_14131

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l14_14131


namespace no_solutions_exists_unique_l14_14755

def is_solution (a b c x y z : ℤ) : Prop :=
  2 * x - b * y + z = 2 * b ∧
  a * x + 5 * y - c * z = a

def no_solutions_for (a b c : ℤ) : Prop :=
  ∀ x y z : ℤ, ¬ is_solution a b c x y z

theorem no_solutions_exists_unique (a b c : ℤ) :
  (a = -2 ∧ b = 5 ∧ c = 1) ∨
  (a = 2 ∧ b = -5 ∧ c = -1) ∨
  (a = 10 ∧ b = -1 ∧ c = -5) ↔
  no_solutions_for a b c := 
sorry

end no_solutions_exists_unique_l14_14755


namespace base_n_divisible_by_13_l14_14470

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 7 + 3 * n + 5 * n^2 + 6 * n^3 + 3 * n^4 + 5 * n^5

-- The main theorem stating the result
theorem base_n_divisible_by_13 : 
  (∃ ns : Finset ℕ, ns.card = 16 ∧ ∀ n ∈ ns, 3 ≤ n ∧ n ≤ 200 ∧ f n % 13 = 0) :=
sorry

end base_n_divisible_by_13_l14_14470


namespace minimum_value_f_l14_14456

noncomputable def f (x : ℝ) : ℝ := max (3 - x) (x^2 - 4 * x + 3)

theorem minimum_value_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ (∀ ε > 0, ∃ x : ℝ, x ≥ 0 ∧ f x < m + ε) ∧ m = 0 := 
sorry

end minimum_value_f_l14_14456


namespace rectangle_area_l14_14579

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l14_14579


namespace third_side_triangle_max_l14_14099

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l14_14099


namespace jim_catches_up_to_cara_l14_14115

noncomputable def time_to_catch_up (jim_speed: ℝ) (cara_speed: ℝ) (initial_time: ℝ) (stretch_time: ℝ) : ℝ :=
  let initial_distance_jim := jim_speed * initial_time
  let initial_distance_cara := cara_speed * initial_time
  let added_distance_cara := cara_speed * stretch_time
  let distance_gap := added_distance_cara
  let relative_speed := jim_speed - cara_speed
  distance_gap / relative_speed

theorem jim_catches_up_to_cara :
  time_to_catch_up 6 5 (30/60) (18/60) * 60 = 90 :=
by
  sorry

end jim_catches_up_to_cara_l14_14115


namespace no_pos_int_lt_2000_7_times_digits_sum_l14_14923

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_pos_int_lt_2000_7_times_digits_sum :
  ∀ n : ℕ, n < 2000 → n = 7 * sum_of_digits n → False :=
by
  intros n h1 h2
  sorry

end no_pos_int_lt_2000_7_times_digits_sum_l14_14923


namespace counting_numbers_dividing_48_with_remainder_7_l14_14490

theorem counting_numbers_dividing_48_with_remainder_7 :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n ∈ S, n > 7 ∧ 48 % n = 0 :=
by
  sorry

end counting_numbers_dividing_48_with_remainder_7_l14_14490


namespace intersection_M_N_l14_14330

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14330


namespace derivative_at_pi_div_4_l14_14773

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_div_4 :
  (deriv f) (Real.pi / 4) = 0 :=
sorry

end derivative_at_pi_div_4_l14_14773


namespace log2_bounds_sum_l14_14154

theorem log2_bounds_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : (a : ℝ) < Real.log 50 / Real.log 2) (h4 : Real.log 50 / Real.log 2 < (b : ℝ)) :
  a + b = 11 :=
sorry

end log2_bounds_sum_l14_14154


namespace range_of_a_l14_14845

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ a < -4 ∨ a > 4 :=
by
  sorry

end range_of_a_l14_14845


namespace most_reasonable_sample_l14_14279

-- Define what it means to be a reasonable sample
def is_reasonable_sample (sample : String) : Prop :=
  sample = "D"

-- Define the conditions for each sample
def sample_A := "A"
def sample_B := "B"
def sample_C := "C"
def sample_D := "D"

-- Define the problem statement
theorem most_reasonable_sample :
  is_reasonable_sample sample_D :=
sorry

end most_reasonable_sample_l14_14279


namespace range_of_a_l14_14992

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- State the main theorem
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f_prime a x1 = 0 ∧ f_prime a x2 = 0) →
  a < 0 :=
by
  sorry

end range_of_a_l14_14992


namespace inequality_solution_l14_14698

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l14_14698


namespace relation_y1_y2_y3_l14_14399

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -x^2 + 2*x + c

variables (x1 x2 x3 : ℝ)
variables (y1 y2 y3 c : ℝ)
variables (P1 : x1 = -1)
variables (P2 : x2 = 3)
variables (P3 : x3 = 5)
variables (H1 : y1 = quadratic_function x1 c)
variables (H2 : y2 = quadratic_function x2 c)
variables (H3 : y3 = quadratic_function x3 c)

theorem relation_y1_y2_y3 (c : ℝ) :
  (y1 = y2) ∧ (y1 > y3) :=
sor_問題ry

end relation_y1_y2_y3_l14_14399


namespace matrix_not_invertible_values_l14_14762

noncomputable theory

open Matrix

-- Define the matrix
def myMatrix (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![a + d, b, c], 
    ![b, c + d, a], 
    ![c, a, b + d]]

variables {a b c d : ℝ}

-- Prove the given statement:
theorem matrix_not_invertible_values :
  det (myMatrix a b c d) = 0 → 
  (d = - a - b - c ∨ a = b ∧ b = c ∨ a + b + c = 0 ∨ someOtherSpecialCase) → -- conditions for simplicity
  (∃ v, v = (a / (b + c) + b / (a + c) + c / (a + b)) ∧ (v = -3 ∨ v = 3 / 2)) := 
sorry

end matrix_not_invertible_values_l14_14762


namespace rectangle_area_is_243_square_meters_l14_14601

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l14_14601


namespace expression_is_integer_iff_divisible_l14_14803

theorem expression_is_integer_iff_divisible (k n : ℤ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ m : ℤ, n = m * (k + 2) ↔ (∃ C : ℤ, (3 * n - 4 * k + 2) / (k + 2) * C = (3 * n - 4 * k + 2) / (k + 2)) :=
sorry

end expression_is_integer_iff_divisible_l14_14803


namespace y_intercept_of_line_l14_14713

def equation (x y : ℝ) : Prop := 3 * x - 5 * y = 10

theorem y_intercept_of_line : equation 0 (-2) :=
by
  sorry

end y_intercept_of_line_l14_14713


namespace S_sum_l14_14223

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2)
  else (n + 1) / 2

theorem S_sum :
  S 19 + S 37 + S 52 = 3 :=
by
  sorry

end S_sum_l14_14223


namespace add_same_sign_abs_l14_14268

theorem add_same_sign_abs (a b : ℤ) : 
  (∀ a b : ℤ, (a ≥ 0 ∧ b ≥ 0) → (|a + b| = |a| + |b| ∧ a + b ≥ 0)) ∧ 
  (∀ a b : ℤ, (a < 0 ∧ b < 0) → (|a + b| = |a| + |b| ∧ a + b < 0)) :=
by
  intro a b
  sorry

end add_same_sign_abs_l14_14268


namespace pow_mod_79_l14_14858

theorem pow_mod_79 (a : ℕ) (h : a = 7) : a^79 % 11 = 6 := by
  sorry

end pow_mod_79_l14_14858


namespace question_1_question_2_l14_14487

def curve_is_ellipse (m : ℝ) : Prop :=
  (3 - m > 0) ∧ (m - 1 > 0) ∧ (3 - m > m - 1)

def domain_is_R (m : ℝ) : Prop :=
  m^2 < (9 / 4)

theorem question_1 (m : ℝ) :
  curve_is_ellipse m → 1 < m ∧ m < 2 :=
sorry

theorem question_2 (m : ℝ) :
  (curve_is_ellipse m ∧ domain_is_R m) → 1 < m ∧ m < (3 / 2) :=
sorry

end question_1_question_2_l14_14487


namespace seats_shortage_l14_14275

-- Definitions of the conditions
def children := 52
def adults := 29
def seniors := 15
def pets := 3
def total_seats := 95

-- Theorem statement to prove the number of people and pets without seats
theorem seats_shortage : children + adults + seniors + pets - total_seats = 4 :=
by
  sorry

end seats_shortage_l14_14275


namespace rectangular_field_area_l14_14598

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l14_14598


namespace find_f_neg2_l14_14038

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Define the conditions and statement to be proved
theorem find_f_neg2 (a b : ℝ) (h1 : f a b 2 = 9) : f a b (-2) = 13 :=
by
  -- Conditions lead to the conclusion to be proved
  sorry

end find_f_neg2_l14_14038


namespace problem1_problem2_l14_14915

noncomputable def f : ℝ → ℝ := -- we assume f is noncomputable since we know its explicit form in the desired interval
sorry

axiom periodic_f (x : ℝ) : f (x + 5) = f x
axiom odd_f {x : ℝ} (h : -1 ≤ x ∧ x ≤ 1) : f (-x) = -f x
axiom quadratic_f {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * (x - 2) ^ 2 - 5
axiom minimum_f : f 2 = -5

theorem problem1 : f 1 + f 4 = 0 :=
by
  sorry

theorem problem2 {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * x ^ 2 - 8 * x + 3 :=
by
  sorry

end problem1_problem2_l14_14915


namespace number_of_intersections_l14_14924

def line_eq (x y : ℝ) : Prop := 4 * x + 9 * y = 12
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

theorem number_of_intersections : 
  ∃ (p1 p2 : ℝ × ℝ), 
  (line_eq p1.1 p1.2 ∧ circle_eq p1.1 p1.2) ∧ 
  (line_eq p2.1 p2.2 ∧ circle_eq p2.1 p2.2) ∧ 
  p1 ≠ p2 ∧ 
  ∀ p : ℝ × ℝ, 
    (line_eq p.1 p.2 ∧ circle_eq p.1 p.2) → (p = p1 ∨ p = p2) :=
sorry

end number_of_intersections_l14_14924


namespace tan_beta_identity_l14_14655

theorem tan_beta_identity (α β : ℝ) (h1 : Real.tan α = 1/3) (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 1/7 :=
sorry

end tan_beta_identity_l14_14655


namespace reflect_center_of_circle_l14_14987

def reflect_point (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

theorem reflect_center_of_circle :
  reflect_point (3, -7) = (7, -3) :=
by
  sorry

end reflect_center_of_circle_l14_14987


namespace rectangular_field_area_l14_14595

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l14_14595


namespace combined_average_score_l14_14807

theorem combined_average_score (M E : ℕ) (m e : ℕ) (h1 : M = 82) (h2 : E = 68) (h3 : m = 5 * e / 7) :
  ((m * M) + (e * E)) / (m + e) = 72 :=
by
  -- Placeholder for the proof
  sorry

end combined_average_score_l14_14807


namespace third_side_triangle_max_l14_14101

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end third_side_triangle_max_l14_14101


namespace intersection_complement_l14_14180

open Set

variable {α : Type*}
noncomputable def A : Set ℝ := {x | x^2 ≥ 1}
noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_complement :
  A ∩ (compl B) = (Iic (-1)) ∪ (Ioi 2) := by
sorry

end intersection_complement_l14_14180


namespace solve_eq_log_base_l14_14693

theorem solve_eq_log_base (x : ℝ) : (9 : ℝ)^(x+8) = (10 : ℝ)^x → x = Real.logb (10 / 9) ((9 : ℝ)^8) := by
  intro h
  sorry

end solve_eq_log_base_l14_14693


namespace part_one_part_two_l14_14646

def f (x : ℝ) : ℝ := |x| + |x - 1|

theorem part_one (m : ℝ) (h : ∀ x, f x ≥ |m - 1|) : m ≤ 2 := by
  sorry

theorem part_two (a b : ℝ) (M : ℝ) (ha : 0 < a) (hb : 0 < b) (hM : a^2 + b^2 = M) (hM_value : M = 2) : a + b ≥ 2 * a * b := by
  sorry

end part_one_part_two_l14_14646


namespace greatest_third_side_l14_14106

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l14_14106


namespace no_adjacent_numbers_differ_by_10_or_multiple_10_l14_14143

theorem no_adjacent_numbers_differ_by_10_or_multiple_10 :
  ¬ ∃ (f : Fin 25 → Fin 25),
    (∀ n : Fin 25, f (n + 1) - f n = 10 ∨ (f (n + 1) - f n) % 10 = 0) :=
by
  sorry

end no_adjacent_numbers_differ_by_10_or_multiple_10_l14_14143


namespace total_num_problems_eq_30_l14_14125

-- Define the conditions
def test_points : ℕ := 100
def points_per_3_point_problem : ℕ := 3
def points_per_4_point_problem : ℕ := 4
def num_4_point_problems : ℕ := 10

-- Define the number of 3-point problems
def num_3_point_problems : ℕ :=
  (test_points - num_4_point_problems * points_per_4_point_problem) / points_per_3_point_problem

-- Prove the total number of problems is 30
theorem total_num_problems_eq_30 :
  num_3_point_problems + num_4_point_problems = 30 := 
sorry

end total_num_problems_eq_30_l14_14125


namespace inequality_holds_for_all_real_l14_14691

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 6 * x + 8 ≥ -(x + 4) * (x + 6) :=
  sorry

end inequality_holds_for_all_real_l14_14691


namespace cos_of_angle_B_l14_14661

theorem cos_of_angle_B (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : 6 * Real.sin A = 4 * Real.sin B) (h3 : 4 * Real.sin B = 3 * Real.sin C) : 
  Real.cos B = Real.sqrt 7 / 4 :=
by
  sorry

end cos_of_angle_B_l14_14661


namespace airplane_travel_difference_correct_l14_14889

-- Define airplane's speed without wind
def airplane_speed_without_wind : ℕ := a

-- Define wind speed
def wind_speed : ℕ := 20

-- Define time without wind
def time_without_wind : ℕ := 4

-- Define time against wind
def time_against_wind : ℕ := 3

-- Define distance covered without wind
def distance_without_wind : ℕ := airplane_speed_without_wind * time_without_wind

-- Define effective speed against wind
def effective_speed_against_wind : ℕ := airplane_speed_without_wind - wind_speed

-- Define distance covered against wind
def distance_against_wind : ℕ := effective_speed_against_wind * time_against_wind

-- Define the difference in distances
def distance_difference : ℕ := distance_without_wind - distance_against_wind

-- The theorem we wish to prove
theorem airplane_travel_difference_correct (a : ℕ) :
  distance_difference = a + 60 :=
by
sorry

end airplane_travel_difference_correct_l14_14889


namespace smallest_zarks_l14_14025

theorem smallest_zarks (n : ℕ) : (n^2 > 15 * n) → (n ≥ 16) := sorry

end smallest_zarks_l14_14025


namespace frog_final_position_probability_l14_14880

noncomputable def frog_position_probability (positions : ℕ → ℝ^2) : ℚ :=
  sorry -- Define position sequence and probability distribution accurately

theorem frog_final_position_probability :
  frog_position_probability (λ n, (1 : ℝ^2)) = 1/8 :=
sorry -- Prove that the probability of the final position being within a 1-meter radius after 4 jumps is 1/8

end frog_final_position_probability_l14_14880


namespace possible_six_digit_numbers_divisible_by_3_l14_14203

theorem possible_six_digit_numbers_divisible_by_3 (missing_digit_condition : ∀ k : Nat, (8 + 5 + 5 + 2 + 2 + k) % 3 = 0) : 
  ∃ count : Nat, count = 13 := by
  sorry

end possible_six_digit_numbers_divisible_by_3_l14_14203


namespace functional_equation_solution_l14_14625

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x) + f (f y) = 2 * y + f (x - y)) ↔ (∀ x : ℝ, f x = x) := by
  sorry

end functional_equation_solution_l14_14625


namespace parallel_lines_l14_14190

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + a + 3 = 0) ∧ (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) 
  → a = -2 :=
sorry

end parallel_lines_l14_14190


namespace area_of_park_l14_14416

theorem area_of_park (x : ℕ) (rate_per_meter : ℝ) (total_cost : ℝ)
  (ratio_len_wid : ℕ × ℕ)
  (h_ratio : ratio_len_wid = (3, 2))
  (h_cost : total_cost = 140)
  (unit_rate : rate_per_meter = 0.50)
  (h_perimeter : 10 * x * rate_per_meter = total_cost) :
  6 * x^2 = 4704 :=
by
  sorry

end area_of_park_l14_14416


namespace quadratic_condition_l14_14400

variables {c y1 y2 y3 : ℝ}

/-- Points P1(-1, y1), P2(3, y2), P3(5, y3) are all on the graph of the quadratic function y = -x^2 + 2x + c. --/
def points_on_parabola (y1 y2 y3 c : ℝ) : Prop :=
  y1 = -(-1)^2 + 2*(-1) + c ∧
  y2 = -(3)^2 + 2*(3) + c ∧
  y3 = -(5)^2 + 2*(5) + c

/-- The quadratic function y = -x^2 + 2x + c has an axis of symmetry at x = 1 and opens downwards. --/
theorem quadratic_condition (h : points_on_parabola y1 y2 y3 c) : 
  y1 = y2 ∧ y2 > y3 :=
sorry

end quadratic_condition_l14_14400


namespace no_nontrivial_integer_solutions_l14_14972

theorem no_nontrivial_integer_solutions (x y z : ℤ) : x^3 + 2*y^3 + 4*z^3 - 6*x*y*z = 0 -> x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_nontrivial_integer_solutions_l14_14972


namespace like_terms_m_n_sum_l14_14639

theorem like_terms_m_n_sum :
  ∃ (m n : ℕ), (2 : ℤ) * x ^ (3 * n) * y ^ (m + 4) = (-3 : ℤ) * x ^ 9 * y ^ (2 * n) ∧ m + n = 5 :=
by 
  sorry

end like_terms_m_n_sum_l14_14639


namespace find_a_l14_14916

def line1 (a : ℝ) (P : ℝ × ℝ) : Prop := 2 * P.1 - a * P.2 - 1 = 0

def line2 (P : ℝ × ℝ) : Prop := P.1 + 2 * P.2 = 0

theorem find_a (a : ℝ) :
  (∀ P : ℝ × ℝ, line1 a P ∧ line2 P) → a = 1 := by
  sorry

end find_a_l14_14916


namespace find_surcharge_l14_14787

-- The property tax in 1996 is increased by 6% over the 1995 tax.
def increased_tax (T_1995 : ℝ) : ℝ := T_1995 * 1.06

-- Petersons' property tax for the year 1995 is $1800.
def T_1995 : ℝ := 1800

-- The Petersons' 1996 tax totals $2108.
def T_1996 : ℝ := 2108

-- Additional surcharge for a special project.
def surcharge (T_1996 : ℝ) (increased_tax : ℝ) : ℝ := T_1996 - increased_tax

theorem find_surcharge : surcharge T_1996 (increased_tax T_1995) = 200 := by
  sorry

end find_surcharge_l14_14787


namespace fred_seashells_now_l14_14472

def seashells_initial := 47
def seashells_given := 25

theorem fred_seashells_now : seashells_initial - seashells_given = 22 := 
by 
  sorry

end fred_seashells_now_l14_14472


namespace decreasing_geometric_sequence_l14_14039

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q ^ n

theorem decreasing_geometric_sequence (a₁ q : ℝ) (aₙ : ℕ → ℝ) (hₙ : ∀ n, aₙ n = geometric_sequence a₁ q n) 
  (h_condition : 0 < q ∧ q < 1) : ¬(0 < q ∧ q < 1 ↔ ∀ n, aₙ n > aₙ (n + 1)) :=
sorry

end decreasing_geometric_sequence_l14_14039


namespace f_lg_equality_l14_14012

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_lg_equality : f (Real.log 2) + f (Real.log (1 / 2)) = 2 := sorry

end f_lg_equality_l14_14012


namespace total_legs_arms_tentacles_correct_l14_14811

-- Define the counts of different animals
def num_horses : Nat := 2
def num_dogs : Nat := 5
def num_cats : Nat := 7
def num_turtles : Nat := 3
def num_goat : Nat := 1
def num_snakes : Nat := 4
def num_spiders : Nat := 2
def num_birds : Nat := 3
def num_starfish : Nat := 1
def num_octopus : Nat := 1
def num_three_legged_dogs : Nat := 1

-- Define the legs, arms, and tentacles for each type of animal
def legs_per_horse : Nat := 4
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def legs_per_turtle : Nat := 4
def legs_per_goat : Nat := 4
def legs_per_snake : Nat := 0
def legs_per_spider : Nat := 8
def legs_per_bird : Nat := 2
def arms_per_starfish : Nat := 5
def tentacles_per_octopus : Nat := 6
def legs_per_three_legged_dog : Nat := 3

-- Define the total number of legs, arms, and tentacles
def total_legs_arms_tentacles : Nat := 
  (num_horses * legs_per_horse) + 
  (num_dogs * legs_per_dog) + 
  (num_cats * legs_per_cat) + 
  (num_turtles * legs_per_turtle) + 
  (num_goat * legs_per_goat) + 
  (num_snakes * legs_per_snake) + 
  (num_spiders * legs_per_spider) + 
  (num_birds * legs_per_bird) + 
  (num_starfish * arms_per_starfish) + 
  (num_octopus * tentacles_per_octopus) + 
  (num_three_legged_dogs * legs_per_three_legged_dog)

-- The theorem to prove
theorem total_legs_arms_tentacles_correct :
  total_legs_arms_tentacles = 108 := by
  -- Proof goes here
  sorry

end total_legs_arms_tentacles_correct_l14_14811


namespace has_exactly_one_solution_l14_14320

theorem has_exactly_one_solution (a : ℝ) : 
  (∀ x : ℝ, 5^(x^2 + 2 * a * x + a^2) = a * x^2 + 2 * a^2 * x + a^3 + a^2 - 6 * a + 6) ↔ (a = 1) :=
sorry

end has_exactly_one_solution_l14_14320


namespace situps_difference_l14_14390

def ken_situps : ℕ := 20
def nathan_situps : ℕ := 2 * ken_situps
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2
def emma_situps : ℕ := bob_situps / 3

theorem situps_difference : 
  (nathan_situps + bob_situps + emma_situps) - ken_situps = 60 := by
  sorry

end situps_difference_l14_14390


namespace ratio_of_shaded_area_l14_14145

-- Define the problem in terms of ratios of areas
open Real

theorem ratio_of_shaded_area (ABCD E F G H : ℝ) (h : ∀ (E F G H ∈ ABCD), 
  (E = 1/3 * BA) ∧ (F = 2/3 * CB) ∧ (G = 2/3 * DC) ∧ (H = 1/3 * AD)) :
  ∃ (shaded_area ABCD_area : ℝ), shaded_area / ABCD_area = 5 / 9 := sorry

end ratio_of_shaded_area_l14_14145


namespace rectangular_field_area_l14_14589

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l14_14589


namespace find_days_l14_14656

variables (a d e k m : ℕ) (y : ℕ)

-- Assumptions based on the problem
def workers_efficiency_condition : Prop := 
  (a * e * (d * k) / (a * e)) = d

-- Conclusion we aim to prove
def target_days_condition : Prop :=
  y = (a * a) / (d * k * m)

theorem find_days (h : workers_efficiency_condition a d e k) : target_days_condition a d k m y :=
  sorry

end find_days_l14_14656


namespace chocolates_per_student_class_7B_l14_14310

theorem chocolates_per_student_class_7B :
  (∃ (x : ℕ), 9 * x < 288 ∧ 10 * x > 300 ∧ x = 31) :=
by
  use 31
  -- proof steps omitted here
  sorry

end chocolates_per_student_class_7B_l14_14310


namespace books_per_shelf_l14_14288

def initial_coloring_books : ℕ := 86
def sold_coloring_books : ℕ := 37
def shelves : ℕ := 7

theorem books_per_shelf : (initial_coloring_books - sold_coloring_books) / shelves = 7 := by
  sorry

end books_per_shelf_l14_14288


namespace sum_first_20_odds_is_400_l14_14716

-- Define the sequence of the first 20 positive odd integers
def sequence (n : ℕ) : ℕ := 1 + 2 * n

-- Define the sum of the first 'n' terms of an arithmetic sequence
def sum_arithmetic_sequence (a l n : ℕ) : ℕ := ((a + l) * n) / 2

-- Define the sum of the first 20 positive odd integers
def sum_first_20_odds : ℕ := sum_arithmetic_sequence 1 39 20

-- Claim that the sum of the first 20 positive odd integers is 400
theorem sum_first_20_odds_is_400 : sum_first_20_odds = 400 :=
by
  -- Proof omitted
  sorry

end sum_first_20_odds_is_400_l14_14716


namespace reciprocal_inequality_pos_reciprocal_inequality_neg_l14_14187

theorem reciprocal_inequality_pos {a b : ℝ} (h : a < b) (ha : 0 < a) : (1 / a) > (1 / b) :=
sorry

theorem reciprocal_inequality_neg {a b : ℝ} (h : a < b) (hb : b < 0) : (1 / a) < (1 / b) :=
sorry

end reciprocal_inequality_pos_reciprocal_inequality_neg_l14_14187


namespace suff_not_nec_l14_14478

variables (a b : ℝ)
def P := (a = 1) ∧ (b = 1)
def Q := (a + b = 2)

theorem suff_not_nec : P a b → Q a b ∧ ¬ (Q a b → P a b) :=
by
  sorry

end suff_not_nec_l14_14478


namespace lcm_of_product_of_mutually_prime_l14_14423

theorem lcm_of_product_of_mutually_prime (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.lcm a b = a * b :=
by
  sorry

end lcm_of_product_of_mutually_prime_l14_14423


namespace inflection_point_on_3x_l14_14178

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 * Real.sin x - Real.cos x
noncomputable def f' (x : ℝ) : ℝ := 3 + 4 * Real.cos x + Real.sin x
noncomputable def f'' (x : ℝ) : ℝ := -4 * Real.sin x + Real.cos x

theorem inflection_point_on_3x {x0 : ℝ} (h : f'' x0 = 0) : (f x0) = 3 * x0 := by
  sorry

end inflection_point_on_3x_l14_14178


namespace least_prime_factor_of_expression_l14_14260

theorem least_prime_factor_of_expression : 
  ∀ (p : ℕ), p.prime → (p ∣ (11 ^ 5 - 11 ^ 4)) → (p = 2) :=
sorry

end least_prime_factor_of_expression_l14_14260


namespace polynomial_solution_l14_14786

theorem polynomial_solution (x : ℝ) (h : (2 * x - 1) ^ 2 = 9) : x = 2 ∨ x = -1 :=
by
  sorry

end polynomial_solution_l14_14786


namespace solve_x2_y2_eq_3z2_in_integers_l14_14533

theorem solve_x2_y2_eq_3z2_in_integers (x y z : ℤ) : x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end solve_x2_y2_eq_3z2_in_integers_l14_14533


namespace am_gm_inequality_l14_14513

theorem am_gm_inequality {a1 a2 a3 : ℝ} (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) :
  (a1 * a2 / a3) + (a2 * a3 / a1) + (a3 * a1 / a2) ≥ a1 + a2 + a3 := 
by 
  sorry

end am_gm_inequality_l14_14513


namespace tan_domain_l14_14990

open Real

theorem tan_domain (k : ℤ) (x : ℝ) :
  (∀ k : ℤ, x ≠ (k * π / 2) + (3 * π / 8)) ↔ 
  (∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2) := sorry

end tan_domain_l14_14990


namespace geometric_sequence_a3_l14_14500

theorem geometric_sequence_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 1)
  (h5 : a 5 = 4)
  (geo_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 3 = 2 :=
by
  sorry

end geometric_sequence_a3_l14_14500


namespace total_books_count_l14_14107

theorem total_books_count (total_cost : ℕ) (math_book_cost : ℕ) (history_book_cost : ℕ) 
    (math_books_count : ℕ) (history_books_count : ℕ) (total_books : ℕ) :
    total_cost = 390 ∧ math_book_cost = 4 ∧ history_book_cost = 5 ∧ 
    math_books_count = 10 ∧ total_books = math_books_count + history_books_count ∧ 
    total_cost = (math_book_cost * math_books_count) + (history_book_cost * history_books_count) →
    total_books = 80 := by
  sorry

end total_books_count_l14_14107


namespace line_equation_l14_14991

theorem line_equation (P : ℝ × ℝ) (slope : ℝ) (hP : P = (-2, 0)) (hSlope : slope = 3) :
    ∃ (a b : ℝ), ∀ x y : ℝ, y = a * x + b ↔ P.1 = -2 ∧ P.2 = 0 ∧ slope = 3 ∧ y = 3 * x + 6 :=
by
  sorry

end line_equation_l14_14991


namespace largest_number_is_l14_14999

-- Define the conditions stated in the problem
def sum_of_three_numbers_is_100 (a b c : ℝ) : Prop :=
  a + b + c = 100

def two_larger_numbers_differ_by_8 (b c : ℝ) : Prop :=
  c - b = 8

def two_smaller_numbers_differ_by_5 (a b : ℝ) : Prop :=
  b - a = 5

-- Define the hypothesis
def problem_conditions (a b c : ℝ) : Prop :=
  sum_of_three_numbers_is_100 a b c ∧
  two_larger_numbers_differ_by_8 b c ∧
  two_smaller_numbers_differ_by_5 a b

-- Define the proof problem
theorem largest_number_is (a b c : ℝ) (h : problem_conditions a b c) : 
  c = 121 / 3 :=
sorry

end largest_number_is_l14_14999


namespace garden_width_l14_14261

theorem garden_width (L W : ℕ) 
  (area_playground : 192 = 16 * 12)
  (area_garden : 192 = L * W)
  (perimeter_garden : 64 = 2 * L + 2 * W) :
  W = 12 :=
by
  sorry

end garden_width_l14_14261


namespace solve_floor_equation_l14_14902

theorem solve_floor_equation (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 20) : 5 ≤ x ∧ x < 5.25 := by
  sorry

end solve_floor_equation_l14_14902


namespace inequality_solution_set_system_of_inequalities_solution_set_l14_14844

theorem inequality_solution_set (x : ℝ) (h : 3 * x - 5 > 5 * x + 3) : x < -4 :=
by sorry

theorem system_of_inequalities_solution_set (x : ℤ) 
  (h₁ : x - 1 ≥ 1 - x) 
  (h₂ : x + 8 > 4 * x - 1) : x = 1 ∨ x = 2 :=
by sorry

end inequality_solution_set_system_of_inequalities_solution_set_l14_14844


namespace length_of_chord_l14_14783

theorem length_of_chord (x y : ℝ) 
  (h1 : (x - 1)^2 + y^2 = 4) 
  (h2 : x + y + 1 = 0) 
  : ∃ (l : ℝ), l = 2 * Real.sqrt 2 := by
  sorry

end length_of_chord_l14_14783


namespace tegwen_family_total_children_l14_14821

variable (Tegwen : Type)

-- Variables representing the number of girls and boys
variable (g b : ℕ)

-- Conditions from the problem
variable (h1 : b = g - 1)
variable (h2 : g = (3/2:ℚ) * (b - 1))

-- Proposition that the total number of children is 11
theorem tegwen_family_total_children : g + b = 11 := by
  sorry

end tegwen_family_total_children_l14_14821


namespace sum_incorrect_correct_l14_14274

theorem sum_incorrect_correct (x : ℕ) (h : x + 9 = 39) :
  ((x - 5 + 14) + (x * 5 + 14)) = 203 :=
sorry

end sum_incorrect_correct_l14_14274


namespace candy_bar_multiple_l14_14815

theorem candy_bar_multiple (s m x : ℕ) (h1 : s = m * x + 6) (h2 : x = 24) (h3 : s = 78) : m = 3 :=
by
  sorry

end candy_bar_multiple_l14_14815


namespace integer_count_between_l14_14653

theorem integer_count_between (a b : ℝ) (ha : a = (10.5)^3) (hb : b = (10.6)^3) :
  (b.floor - a.ceil + 1 = 33) :=
by
  have h1 : a = 1157.625 := by rw [ha]; norm_num
  have h2 : b = 1191.016 := by rw [hb]; norm_num
  sorry

end integer_count_between_l14_14653


namespace negation_of_exists_l14_14707

theorem negation_of_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end negation_of_exists_l14_14707


namespace repeating_decimal_sum_l14_14162

theorem repeating_decimal_sum :
  let x := (1 : ℚ) / 3
  let y := (7 : ℚ) / 33
  x + y = 6 / 11 :=
  by
  sorry

end repeating_decimal_sum_l14_14162


namespace rectangular_field_area_l14_14593

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l14_14593


namespace speed_of_train_in_km_per_hr_l14_14135

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l14_14135


namespace greatest_power_divides_factorial_l14_14328

open Nat

noncomputable def greatest_power_dividing_factorial (p : ℕ) (n : ℕ) [hp : Fact p.Prime] : ℕ :=
  ∑ i in Finset.range (n+1), n / p^i

theorem greatest_power_divides_factorial (p : ℕ) (n : ℕ) [hp : Fact p.Prime] :
  ∃ k : ℕ, (p^k ∣ n.factorial) ∧
  (∀ j : ℕ, (p^j ∣ n.factorial) → j ≤ k) :=
begin
  use greatest_power_dividing_factorial p n,
  sorry
end

end greatest_power_divides_factorial_l14_14328


namespace roots_of_equations_l14_14771

theorem roots_of_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + 4 * a * x - 4 * a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a - 1) * x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2 * a * x - 2 * a = 0) ↔ 
  a ≤ -3 / 2 ∨ a ≥ -1 :=
sorry

end roots_of_equations_l14_14771


namespace mowing_time_approximately_correct_l14_14687

noncomputable def timeToMowLawn 
  (length width : ℝ) -- dimensions of the lawn in feet
  (swath overlap : ℝ) -- swath width and overlap in inches
  (speed : ℝ) : ℝ :=  -- walking speed in feet per hour
  (length * (width / ((swath - overlap) / 12))) / speed

theorem mowing_time_approximately_correct
  (h_length : ∀ (length : ℝ), length = 100)
  (h_width : ∀ (width : ℝ), width = 120)
  (h_swath : ∀ (swath : ℝ), swath = 30)
  (h_overlap : ∀ (overlap : ℝ), overlap = 6)
  (h_speed : ∀ (speed : ℝ), speed = 4500) :
  abs (timeToMowLawn 100 120 30 6 4500 - 1.33) < 0.01 := -- assert the answer is approximately 1.33 with a tolerance
by
  intros
  have length := h_length 100
  have width := h_width 120
  have swath := h_swath 30
  have overlap := h_overlap 6
  have speed := h_speed 4500
  rw [length, width, swath, overlap, speed]
  simp [timeToMowLawn]
  sorry

end mowing_time_approximately_correct_l14_14687


namespace maximum_M_value_l14_14181

theorem maximum_M_value (x y z u M : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < u)
  (h5 : x - 2 * y = z - 2 * u) (h6 : 2 * y * z = u * x) (h7 : z ≥ y) 
  : ∃ M, M ≤ z / y ∧ M ≤ 6 + 4 * Real.sqrt 2 :=
sorry

end maximum_M_value_l14_14181


namespace monkey_hop_distance_l14_14739

theorem monkey_hop_distance
    (total_height : ℕ)
    (slip_back : ℕ)
    (hours : ℕ)
    (reach_time : ℕ)
    (hop : ℕ)
    (H1 : total_height = 19)
    (H2 : slip_back = 2)
    (H3 : hours = 17)
    (H4 : reach_time = 16 * (hop - slip_back) + hop)
    (H5 : total_height = reach_time) :
    hop = 3 := by
  sorry

end monkey_hop_distance_l14_14739


namespace cow_count_l14_14738

theorem cow_count
  (initial_cows : ℕ) (cows_died : ℕ) (cows_sold : ℕ)
  (increase_cows : ℕ) (gift_cows : ℕ) (final_cows : ℕ) (bought_cows : ℕ) :
  initial_cows = 39 ∧ cows_died = 25 ∧ cows_sold = 6 ∧
  increase_cows = 24 ∧ gift_cows = 8 ∧ final_cows = 83 →
  bought_cows = 43 :=
by
  sorry

end cow_count_l14_14738


namespace B_cycling_speed_l14_14141

variable (A_speed B_distance B_time B_speed : ℕ)
variable (t1 : ℕ := 7)
variable (d_total : ℕ := 140)
variable (B_catch_time : ℕ := 7)

theorem B_cycling_speed :
  A_speed = 10 → 
  d_total = 140 →
  B_catch_time = 7 → 
  B_speed = 20 :=
by
  sorry

end B_cycling_speed_l14_14141


namespace train_speed_l14_14139

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l14_14139


namespace acute_angled_triangle_count_l14_14887

def num_vertices := 8

def total_triangles := Nat.choose num_vertices 3

def right_angled_triangles := 8 * 6

def acute_angled_triangles := total_triangles - right_angled_triangles

theorem acute_angled_triangle_count : acute_angled_triangles = 8 :=
by
  sorry

end acute_angled_triangle_count_l14_14887


namespace greatest_integer_third_side_l14_14091

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l14_14091


namespace positive_real_solution_unique_l14_14491

theorem positive_real_solution_unique :
  (∃! x : ℝ, 0 < x ∧ x^12 + 5 * x^11 - 3 * x^10 + 2000 * x^9 - 1500 * x^8 = 0) :=
sorry

end positive_real_solution_unique_l14_14491


namespace sin_neg_pi_div_two_l14_14754

theorem sin_neg_pi_div_two : Real.sin (-π / 2) = -1 := by
  -- Define the necessary conditions
  let π_in_deg : ℝ := 180 -- π radians equals 180 degrees
  have sin_neg_angle : ∀ θ : ℝ, Real.sin (-θ) = -Real.sin θ := sorry -- sin(-θ) = -sin(θ) for any θ
  have sin_90_deg : Real.sin (π_in_deg / 2) = 1 := sorry -- sin(90 degrees) = 1

  -- The main statement to prove
  sorry

end sin_neg_pi_div_two_l14_14754


namespace sum_of_possible_values_g1_non_const_poly_l14_14799

noncomputable def g (x : ℝ) : ℝ := 6051 * x -- this will be the assumption based on our proof 
-- but ideally, we derive it instead of defining directly for the automated proof.

theorem sum_of_possible_values_g1_non_const_poly (g : ℝ → ℝ) (h : ¬ ∀ x : ℝ, g x = 0) :
  (∀ x : ℝ, x ≠ 0 → g(x - 1) + g(x) + g(x + 1) = (g(x))^2 / (2021 * x)) →
  g 1 = 6051 :=
begin
  sorry
end

end sum_of_possible_values_g1_non_const_poly_l14_14799


namespace correct_statement_l14_14863

-- Conditions as definitions
def deductive_reasoning (p q r : Prop) : Prop :=
  (p → q) → (q → r) → (p → r)

def correctness_of_conclusion := true  -- Indicates statement is defined to be correct

def pattern_of_reasoning (p q r : Prop) : Prop :=
  deductive_reasoning p q r

-- Statement to prove
theorem correct_statement (p q r : Prop) :
  pattern_of_reasoning p q r = deductive_reasoning p q r :=
by sorry

end correct_statement_l14_14863


namespace gcd_sum_equality_l14_14000

theorem gcd_sum_equality (n : ℕ) : 
  (Nat.gcd 6 n + Nat.gcd 8 (2 * n) = 10) ↔ 
  (∃ t : ℤ, n = 12 * t + 4 ∨ n = 12 * t + 6 ∨ n = 12 * t + 8) :=
by
  sorry

end gcd_sum_equality_l14_14000


namespace set_intersection_l14_14352

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l14_14352


namespace linear_system_solution_l14_14014

theorem linear_system_solution (x y m : ℝ) (h1 : x + 2 * y = m) (h2 : 2 * x - 3 * y = 4) (h3 : x + y = 7) : 
  m = 9 :=
sorry

end linear_system_solution_l14_14014


namespace four_letter_words_with_E_count_l14_14182

open Finset

/-- Number of 4-letter words from alphabet {A, B, C, D, E} with at least one E --/
theorem four_letter_words_with_E_count :
  let alphabet := {A, B, C, D, E}
      total_words := (Finset.card alphabet) ^ 4,
      words_without_E := (Finset.card (alphabet \ {'E'})) ^ 4,
      words_with_at_least_one_E := total_words - words_without_E in
  words_with_at_least_one_E = 369 :=
by
  let alphabet := {A, B, C, D, E}
  let total_words := (Finset.card alphabet) ^ 4
  let words_without_E := (Finset.card (alphabet \ {'E'})) ^ 4
  let words_with_at_least_one_E := total_words - words_without_E
  have h_total_words : total_words = 625 := by sorry
  have h_words_without_E : words_without_E = 256 := by sorry
  have h : words_with_at_least_one_E = 369 := by sorry
  exact h

end four_letter_words_with_E_count_l14_14182


namespace eddie_weekly_earnings_l14_14157

theorem eddie_weekly_earnings :
  let mon_hours := 2.5
  let tue_hours := 7 / 6
  let wed_hours := 7 / 4
  let sat_hours := 3 / 4
  let weekday_rate := 4
  let saturday_rate := 6
  let mon_earnings := mon_hours * weekday_rate
  let tue_earnings := tue_hours * weekday_rate
  let wed_earnings := wed_hours * weekday_rate
  let sat_earnings := sat_hours * saturday_rate
  let total_earnings := mon_earnings + tue_earnings + wed_earnings + sat_earnings
  total_earnings = 26.17 := by
  simp only
  norm_num
  sorry

end eddie_weekly_earnings_l14_14157


namespace solve_equation_1_solve_equation_2_l14_14976

theorem solve_equation_1 (x : ℚ) : 1 - (1 / (x - 5)) = (x / (x + 5)) → x = 15 / 2 := 
by
  sorry

theorem solve_equation_2 (x : ℚ) : (3 / (x - 1)) - (2 / (x + 1)) = (1 / (x^2 - 1)) → x = -4 := 
by
  sorry

end solve_equation_1_solve_equation_2_l14_14976


namespace coloring_connected_circles_diff_colors_l14_14989

def num_ways_to_color_five_circles : ℕ :=
  36

theorem coloring_connected_circles_diff_colors (A B C D E : Type) (colors : Fin 3) 
  (connected : (A → B → C → D → E → Prop)) : num_ways_to_color_five_circles = 36 :=
by sorry

end coloring_connected_circles_diff_colors_l14_14989


namespace school_minimum_payment_l14_14847

noncomputable def individual_ticket_price : ℝ := 6
noncomputable def group_ticket_price : ℝ := 40
noncomputable def discount : ℝ := 0.9
noncomputable def students : ℕ := 1258

-- Define the minimum amount the school should pay
noncomputable def minimum_amount := 4536

theorem school_minimum_payment :
  (students / 10 : ℝ) * group_ticket_price * discount + 
  (students % 10) * individual_ticket_price * discount = minimum_amount := sorry

end school_minimum_payment_l14_14847


namespace greatest_integer_third_side_l14_14090

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l14_14090


namespace number_is_100_l14_14398

theorem number_is_100 (n : ℕ) 
  (hquot : n / 11 = 9) 
  (hrem : n % 11 = 1) : 
  n = 100 := 
by 
  sorry

end number_is_100_l14_14398


namespace unique_two_scoop_sundaes_l14_14293

open Nat

theorem unique_two_scoop_sundaes (n : ℕ) (h : n = 8) : (nat.choose n 2) = 28 :=
by 
  rw h 
  simp 
  sorry

end unique_two_scoop_sundaes_l14_14293


namespace shelley_weight_l14_14560

theorem shelley_weight (p s r : ℕ) (h1 : p + s = 151) (h2 : s + r = 132) (h3 : p + r = 115) : s = 84 := 
  sorry

end shelley_weight_l14_14560


namespace original_price_l14_14142

theorem original_price (P : ℝ) (h : P * 0.5 = 1200) : P = 2400 := 
by
  sorry

end original_price_l14_14142


namespace proposition_B_l14_14231

-- Definitions of the conditions
def line (α : Type) := α
def plane (α : Type) := α
def is_within {α : Type} (a : line α) (p : plane α) : Prop := sorry
def is_perpendicular {α : Type} (a : line α) (p : plane α) : Prop := sorry
def planes_are_perpendicular {α : Type} (p₁ p₂ : plane α) : Prop := sorry
def is_prism (poly : Type) : Prop := sorry

-- Propositions
def p {α : Type} (a : line α) (α₁ α₂ : plane α) : Prop :=
  is_within a α₁ ∧ is_perpendicular a α₂ → planes_are_perpendicular α₁ α₂

def q (poly : Type) : Prop := 
  (∃ (face1 face2 : poly), face1 ≠ face2 ∧ sorry) ∧ sorry

-- Proposition B
theorem proposition_B {α : Type} (a : line α) (α₁ α₂ : plane α) (poly : Type) :
  (p a α₁ α₂) ∧ ¬(q poly) :=
by {
  -- Skipping proof
  sorry
}

end proposition_B_l14_14231


namespace closest_distance_l14_14269

theorem closest_distance (x y z : ℕ)
  (h1 : x + y = 10)
  (h2 : y + z = 13)
  (h3 : z + x = 11) :
  min x (min y z) = 4 :=
by
  -- Here you would provide the proof steps in Lean, but for the statement itself, we leave it as sorry.
  sorry

end closest_distance_l14_14269


namespace root_condition_l14_14179

noncomputable def f (a : ℝ) (x : ℝ) := a * x^3 - 3 * x^2 + 1

theorem root_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 0 ∧ ∀ x ≠ x₀, f a x ≠ 0 ∧ x₀ < 0) → a > 2 :=
sorry

end root_condition_l14_14179


namespace difference_between_scores_l14_14382

variable (H F : ℕ)
variable (h_hajar_score : H = 24)
variable (h_sum_scores : H + F = 69)
variable (h_farah_higher : F > H)

theorem difference_between_scores : F - H = 21 := by
  sorry

end difference_between_scores_l14_14382


namespace average_birds_seen_l14_14396

def MarcusBirds : Nat := 7
def HumphreyBirds : Nat := 11
def DarrelBirds : Nat := 9
def IsabellaBirds : Nat := 15

def totalBirds : Nat := MarcusBirds + HumphreyBirds + DarrelBirds + IsabellaBirds
def numberOfIndividuals : Nat := 4

theorem average_birds_seen : (totalBirds / numberOfIndividuals : Real) = 10.5 := 
by
  -- Proof skipped
  sorry

end average_birds_seen_l14_14396


namespace complete_the_square_l14_14256

theorem complete_the_square (x : ℝ) (h : x^2 + 7 * x - 5 = 0) : (x + 7 / 2) ^ 2 = 69 / 4 :=
sorry

end complete_the_square_l14_14256


namespace solve_for_y_l14_14692

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 5 * (y / y ^ (3 / 4)) = 2 + y ^ (1 / 4)) : y = 16 / 81 :=
by
  sorry

end solve_for_y_l14_14692


namespace trapezoid_perimeter_l14_14253

theorem trapezoid_perimeter (AB CD BC DA : ℝ) (BCD_angle : ℝ)
  (h1 : AB = 60) (h2 : CD = 40) (h3 : BC = DA) (h4 : BCD_angle = 120) :
  AB + BC + CD + DA = 220 := 
sorry

end trapezoid_perimeter_l14_14253


namespace in_range_p_1_to_100_l14_14037

def p (m n : ℤ) : ℤ :=
  2 * m^2 - 6 * m * n + 5 * n^2

-- Predicate that asserts k is in the range of p
def in_range_p (k : ℤ) : Prop :=
  ∃ m n : ℤ, p m n = k

-- Lean statement for the theorem
theorem in_range_p_1_to_100 :
  {k : ℕ | 1 ≤ k ∧ k ≤ 100 ∧ in_range_p k} = 
  {1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100} :=
  by
    sorry

end in_range_p_1_to_100_l14_14037


namespace kenya_more_peanuts_l14_14671

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- The proof problem: Prove that Kenya has 48 more peanuts than Jose
theorem kenya_more_peanuts : Kenya_peanuts - Jose_peanuts = 48 :=
by
  -- The proof will go here
  sorry

end kenya_more_peanuts_l14_14671


namespace rectangular_field_area_l14_14592

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l14_14592


namespace cube_sum_identity_l14_14777

theorem cube_sum_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 ∨ r^3 + 1/r^3 = -2 * Real.sqrt 5 := by
  sorry

end cube_sum_identity_l14_14777


namespace find_g_neg_6_l14_14960

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end find_g_neg_6_l14_14960


namespace trigonometric_identity_l14_14623

theorem trigonometric_identity :
  (1 / Real.cos (80 * (Real.pi / 180)) - Real.sqrt 3 / Real.sin (80 * (Real.pi / 180)) = 4) :=
by
  sorry

end trigonometric_identity_l14_14623


namespace find_number_to_be_multiplied_l14_14561

def correct_multiplier := 43
def incorrect_multiplier := 34
def difference := 1224

theorem find_number_to_be_multiplied (x : ℕ) : correct_multiplier * x - incorrect_multiplier * x = difference → x = 136 :=
by
  sorry

end find_number_to_be_multiplied_l14_14561


namespace g_at_minus_six_l14_14964

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end g_at_minus_six_l14_14964


namespace find_number_of_cats_l14_14970

theorem find_number_of_cats (dogs ferrets cats total_shoes shoes_per_animal : ℕ) 
  (h_dogs : dogs = 3)
  (h_ferrets : ferrets = 1)
  (h_total_shoes : total_shoes = 24)
  (h_shoes_per_animal : shoes_per_animal = 4) :
  cats = (total_shoes - (dogs + ferrets) * shoes_per_animal) / shoes_per_animal := by
  sorry

end find_number_of_cats_l14_14970


namespace fraction_lt_sqrt2_bound_l14_14677

theorem fraction_lt_sqrt2_bound (m n : ℕ) (h : (m : ℝ) / n < Real.sqrt 2) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * (n * n))) :=
sorry

end fraction_lt_sqrt2_bound_l14_14677


namespace total_salary_after_strict_manager_l14_14381

-- Definitions based on conditions
def total_initial_salary (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  500 * x + (Finset.sum (Finset.range y) s) = 10000

def kind_manager_total (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  1500 * x + (Finset.sum (Finset.range y) s) + 1000 * y = 24000

def strict_manager_total (x y : ℕ) : ℕ :=
  500 * (x + y)

-- Lean statement to prove the required
theorem total_salary_after_strict_manager (x y : ℕ) (s : ℕ → ℕ) 
  (h_total_initial : total_initial_salary x y s) (h_kind_manager : kind_manager_total x y s) :
  strict_manager_total x y = 7000 := by
  sorry

end total_salary_after_strict_manager_l14_14381


namespace toadon_population_percentage_l14_14419

theorem toadon_population_percentage {pop_total G L T : ℕ}
    (h_total : pop_total = 80000)
    (h_gordonia : G = pop_total / 2)
    (h_lakebright : L = 16000)
    (h_total_population : pop_total = G + T + L) :
    (T * 100 / G) = 60 :=
by sorry

end toadon_population_percentage_l14_14419


namespace carrie_weekly_earning_l14_14613

-- Definitions and conditions
def iphone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weeks_needed : ℕ := 7

-- Calculate the required weekly earning
def weekly_earning : ℕ := (iphone_cost - trade_in_value) / weeks_needed

-- Problem statement: Prove that Carrie makes $80 per week babysitting
theorem carrie_weekly_earning :
  weekly_earning = 80 := by
  sorry

end carrie_weekly_earning_l14_14613


namespace contrapositive_l14_14265

variables (p q : Prop)

theorem contrapositive (hpq : p → q) : ¬ q → ¬ p :=
by sorry

end contrapositive_l14_14265


namespace ryan_chinese_learning_hours_l14_14160

theorem ryan_chinese_learning_hours
    (hours_per_day : ℕ) 
    (days : ℕ) 
    (h1 : hours_per_day = 4) 
    (h2 : days = 6) : 
    hours_per_day * days = 24 := 
by 
    sorry

end ryan_chinese_learning_hours_l14_14160


namespace initial_lives_l14_14204

theorem initial_lives (x : ℕ) (h1 : x - 23 + 46 = 70) : x = 47 := 
by 
  sorry

end initial_lives_l14_14204


namespace teal_sold_pumpkin_pies_l14_14501

def pies_sold 
  (pumpkin_pie_slices : ℕ) (pumpkin_pie_price : ℕ) 
  (custard_pie_slices : ℕ) (custard_pie_price : ℕ) 
  (custard_pies_sold : ℕ) (total_revenue : ℕ) : ℕ :=
  total_revenue / (pumpkin_pie_slices * pumpkin_pie_price)

theorem teal_sold_pumpkin_pies : 
  pies_sold 8 5 6 6 5 340 = 4 := 
by 
  sorry

end teal_sold_pumpkin_pies_l14_14501


namespace cylinder_volume_l14_14555

theorem cylinder_volume (V1 V2 : ℝ) (π : ℝ) (r1 r3 h2 h5 : ℝ)
  (h_radii_ratio : r3 = 3 * r1)
  (h_heights_ratio : h5 = 5 / 2 * h2)
  (h_first_volume : V1 = π * r1^2 * h2)
  (h_V1_value : V1 = 40) :
  V2 = 900 :=
by sorry

end cylinder_volume_l14_14555


namespace mark_total_votes_l14_14022

-- Definitions for the problem conditions
def first_area_registered_voters : ℕ := 100000
def first_area_undecided_percentage : ℕ := 5
def first_area_mark_votes_percentage : ℕ := 70

def remaining_area_increase_percentage : ℕ := 20
def remaining_area_undecided_percentage : ℕ := 7
def multiplier_for_remaining_area_votes : ℕ := 2

-- The Lean statement
theorem mark_total_votes : 
  let first_area_undecided_voters := first_area_registered_voters * first_area_undecided_percentage / 100
  let first_area_votes_cast := first_area_registered_voters - first_area_undecided_voters
  let first_area_mark_votes := first_area_votes_cast * first_area_mark_votes_percentage / 100

  let remaining_area_registered_voters := first_area_registered_voters * (1 + remaining_area_increase_percentage / 100)
  let remaining_area_undecided_voters := remaining_area_registered_voters * remaining_area_undecided_percentage / 100
  let remaining_area_votes_cast := remaining_area_registered_voters - remaining_area_undecided_voters
  let remaining_area_mark_votes := first_area_mark_votes * multiplier_for_remaining_area_votes

  let total_mark_votes := first_area_mark_votes + remaining_area_mark_votes
  total_mark_votes = 199500 := 
by
  -- We skipped the proof (it's not required as per instructions)
  sorry

end mark_total_votes_l14_14022


namespace calculate_group_A_B_C_and_total_is_correct_l14_14899

def groupA_1week : Int := 175000
def groupA_2week : Int := 107000
def groupA_3week : Int := 35000
def groupB_1week : Int := 100000
def groupB_2week : Int := 70350
def groupB_3week : Int := 19500
def groupC_1week : Int := 45000
def groupC_2week : Int := 87419
def groupC_3week : Int := 14425
def kids_staying_home : Int := 590796
def kids_outside_county : Int := 22

def total_kids_in_A := groupA_1week + groupA_2week + groupA_3week
def total_kids_in_B := groupB_1week + groupB_2week + groupB_3week
def total_kids_in_C := groupC_1week + groupC_2week + groupC_3week
def total_kids_in_camp := total_kids_in_A + total_kids_in_B + total_kids_in_C
def total_kids := total_kids_in_camp + kids_staying_home + kids_outside_county

theorem calculate_group_A_B_C_and_total_is_correct :
  total_kids_in_A = 317000 ∧
  total_kids_in_B = 189850 ∧
  total_kids_in_C = 146844 ∧
  total_kids = 1244512 := by
  sorry

end calculate_group_A_B_C_and_total_is_correct_l14_14899


namespace problems_per_page_is_five_l14_14952

-- Let M and R be the number of problems on each math and reading page respectively
variables (M R : ℕ)

-- Conditions given in problem
def two_math_pages := 2 * M
def four_reading_pages := 4 * R
def total_problems := two_math_pages + four_reading_pages

-- Assume the number of problems per page is the same for both math and reading as P
variable (P : ℕ)
def problems_per_page_equal := (2 * P) + (4 * P) = 30

theorem problems_per_page_is_five :
  (2 * P) + (4 * P) = 30 → P = 5 :=
by
  intro h
  sorry

end problems_per_page_is_five_l14_14952


namespace largest_n_unique_k_l14_14854

theorem largest_n_unique_k (n k : ℕ) :
  (frac9_17_lt_frac n (n + k) ∧ frac n (n + k) lt frac8_15) 
  ∧ (∀ (n1 k1 : ℕ), frac9_17_lt_frac n1 (n1 + k1) ∧ frac n1 (n1 + k1) lt frac8_15 
  → (n1 ≤ 136 
  ∧ ((n1 = 136) → (k1 = unique_k))))
  :=
sorry

def frac9_17_lt_frac (a b : ℕ) : Prop := 
  (9:ℚ) / 17 < (a : ℚ) / b

def frac (a b : ℕ) : ℚ :=
  (a : ℚ) / b

def frac8_15 := 
  (8:ℚ) / 15

def unique_k : ℕ :=
  119

end largest_n_unique_k_l14_14854


namespace rectangle_area_l14_14583

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l14_14583


namespace rectangular_field_area_l14_14597

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l14_14597


namespace tenth_day_of_month_is_monday_l14_14537

def total_run_minutes_in_month (hours : ℕ) : ℕ := hours * 60

def run_minutes_per_week (runs_per_week : ℕ) (minutes_per_run : ℕ) : ℕ := 
  runs_per_week * minutes_per_run

def weeks_in_month (total_minutes : ℕ) (minutes_per_week : ℕ) : ℕ := 
  total_minutes / minutes_per_week

def identify_day_of_week (first_day : ℕ) (target_day : ℕ) : ℕ := 
  (first_day + target_day - 1) % 7

theorem tenth_day_of_month_is_monday :
  let hours := 5
  let runs_per_week := 3
  let minutes_per_run := 20
  let first_day := 6 -- Assuming 0=Sunday, ..., 6=Saturday
  let target_day := 10
  total_run_minutes_in_month hours = 300 ∧
  run_minutes_per_week runs_per_week minutes_per_run = 60 ∧
  weeks_in_month 300 60 = 5 ∧
  identify_day_of_week first_day target_day = 1 := -- 1 represents Monday
sorry

end tenth_day_of_month_is_monday_l14_14537


namespace point_in_fourth_quadrant_l14_14239

theorem point_in_fourth_quadrant (θ : ℝ) (h : -1 < Real.cos θ ∧ Real.cos θ < 0) :
    ∃ (x y : ℝ), x = Real.sin (Real.cos θ) ∧ y = Real.cos (Real.cos θ) ∧ x < 0 ∧ y > 0 :=
by
  sorry

end point_in_fourth_quadrant_l14_14239


namespace variance_of_y_eq_4_l14_14359

theorem variance_of_y_eq_4 (x : Fin 2017 → ℝ)
  (hxvar : (∑ i, (x i - (∑ i, x i) / 2017) ^ 2) / 2017 = 4) :
  let y (i : Fin 2017) := x i - 1 in
  (∑ i, (y i - (∑ i, y i) / 2017) ^ 2) / 2017 = 4 := by
  sorry

end variance_of_y_eq_4_l14_14359


namespace large_bottle_water_amount_l14_14951

noncomputable def sport_drink_water_amount (C V : ℝ) (prop_e : ℝ) : ℝ :=
  let F := C / 4
  let W := (C * 15)
  W

theorem large_bottle_water_amount (C V : ℝ) (prop_e : ℝ) (hc : C = 7) (hprop_e : prop_e = 0.05) : sport_drink_water_amount C V prop_e = 105 := by
  sorry

end large_bottle_water_amount_l14_14951


namespace length_of_bridge_l14_14544

-- Define the conditions
def train_length : ℕ := 130 -- length of the train in meters
def train_speed : ℕ := 45  -- speed of the train in km/hr
def crossing_time : ℕ := 30  -- time to cross the bridge in seconds

-- Prove that the length of the bridge is 245 meters
theorem length_of_bridge : 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 245 := 
by
  sorry

end length_of_bridge_l14_14544


namespace value_of_x_minus_y_l14_14662

theorem value_of_x_minus_y (x y : ℝ) (h1 : x = -(-3)) (h2 : |y| = 5) (h3 : x * y < 0) : x - y = 8 := 
sorry

end value_of_x_minus_y_l14_14662


namespace solve_for_x_l14_14975

theorem solve_for_x 
  (x : ℝ) 
  (h : (2/7) * (1/4) * x = 8) : 
  x = 112 :=
sorry

end solve_for_x_l14_14975


namespace sqrt_ab_eq_18_l14_14800

noncomputable def a := Real.log 9 / Real.log 4
noncomputable def b := 108 * (Real.log 8 / Real.log 3)

theorem sqrt_ab_eq_18 : Real.sqrt (a * b) = 18 := by
  sorry

end sqrt_ab_eq_18_l14_14800


namespace first_set_broken_percent_l14_14323

-- Defining some constants
def firstSetTotal : ℕ := 50
def secondSetTotal : ℕ := 60
def secondSetBrokenPercent : ℕ := 20
def totalBrokenMarbles : ℕ := 17

-- Define the function that calculates broken marbles from percentage
def brokenMarbles (percent marbles : ℕ) : ℕ := (percent * marbles) / 100

-- Theorem statement
theorem first_set_broken_percent :
  ∃ (x : ℕ), brokenMarbles x firstSetTotal + brokenMarbles secondSetBrokenPercent secondSetTotal = totalBrokenMarbles ∧ x = 10 :=
by
  sorry

end first_set_broken_percent_l14_14323


namespace lcm_of_numbers_l14_14870

-- Define the conditions given in the problem
def ratio (a b : ℕ) : Prop := 7 * b = 13 * a
def hcf_23 (a b : ℕ) : Prop := Nat.gcd a b = 23

-- Main statement to prove
theorem lcm_of_numbers (a b : ℕ) (h_ratio : ratio a b) (h_hcf : hcf_23 a b) : Nat.lcm a b = 2093 := by
  sorry

end lcm_of_numbers_l14_14870


namespace range_of_a_l14_14013

-- Given definition of the function
def f (x a : ℝ) := abs (x - a)

-- Statement of the problem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < -1 → x₂ < -1 → f x₁ a ≤ f x₂ a) → a ≥ -1 :=
by
  sorry

end range_of_a_l14_14013


namespace cab_drivers_income_on_third_day_l14_14278

theorem cab_drivers_income_on_third_day
  (day1 day2 day4 day5 avg_income n_days : ℝ)
  (h_day1 : day1 = 600)
  (h_day2 : day2 = 250)
  (h_day4 : day4 = 400)
  (h_day5 : day5 = 800)
  (h_avg_income : avg_income = 500)
  (h_n_days : n_days = 5) :
  ∃ day3 : ℝ, (day1 + day2 + day3 + day4 + day5) / n_days = avg_income ∧ day3 = 450 :=
by
  sorry

end cab_drivers_income_on_third_day_l14_14278


namespace boxes_needed_l14_14829

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end boxes_needed_l14_14829


namespace triangle_third_side_l14_14095

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l14_14095


namespace sum_of_decimals_is_one_l14_14813

-- Define digits for each decimal place
def digit_a : ℕ := 2
def digit_b : ℕ := 3
def digit_c : ℕ := 2
def digit_d : ℕ := 2

-- Define the decimal numbers with these digits
def decimal1 : Rat := (digit_b * 10 + digit_a) / 100
def decimal2 : Rat := (digit_d * 10 + digit_c) / 100
def decimal3 : Rat := (2 * 10 + 2) / 100
def decimal4 : Rat := (2 * 10 + 3) / 100

-- The main theorem that states the sum of these decimals is 1
theorem sum_of_decimals_is_one : decimal1 + decimal2 + decimal3 + decimal4 = 1 := by
  sorry

end sum_of_decimals_is_one_l14_14813


namespace max_statements_true_l14_14036

theorem max_statements_true : ∃ x : ℝ, 
  (0 < x^2 ∧ x^2 < 1 ∨ x^2 > 1) ∧ 
  (-1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1) ∧ 
  (0 < (x - x^3) ∧ (x - x^3) < 1) :=
  sorry

end max_statements_true_l14_14036


namespace min_box_coeff_l14_14369

theorem min_box_coeff (a b c d : ℤ) (h_ac : a * c = 40) (h_bd : b * d = 40) : 
  ∃ (min_val : ℤ), min_val = 89 ∧ (a * d + b * c) ≥ min_val :=
sorry

end min_box_coeff_l14_14369


namespace profit_function_definition_maximum_profit_at_100_l14_14947

noncomputable def revenue (x : ℝ) : ℝ := 700 * x
noncomputable def cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 
    10 * x^2 + 100 * x + 250 
  else 
    701 * x + 10000 / x - 9450 + 250

noncomputable def profit (x : ℝ) : ℝ := revenue x - cost x

theorem profit_function_definition :
  ∀ x : ℝ, 0 < x → (profit x = 
    if x < 40 then 
      -10 * x^2 + 600 * x - 250 
    else 
      -(x + 10000 / x) + 9200) := sorry

theorem maximum_profit_at_100 :
  ∃ x_max : ℝ, x_max = 100 ∧ (∀ x : ℝ, 0 < x → profit x ≤ profit x_max)
  := sorry

end profit_function_definition_maximum_profit_at_100_l14_14947


namespace jason_seashells_l14_14954

theorem jason_seashells (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) 
(h1 : initial_seashells = 49) (h2 : given_seashells = 13) :
remaining_seashells = initial_seashells - given_seashells := by
  sorry

end jason_seashells_l14_14954


namespace winning_ticket_probability_l14_14049

theorem winning_ticket_probability (eligible_numbers : List ℕ) (length_eligible_numbers : eligible_numbers.length = 12)
(pick_6 : Π(t : List ℕ), List ℕ) (valid_ticket : List ℕ → Bool) (probability : ℚ) : 
(probability = (1 : ℚ) / (4 : ℚ)) :=
  sorry

end winning_ticket_probability_l14_14049


namespace find_point_P_l14_14177

-- Define the function
def f (x : ℝ) := x^4 - 2 * x

-- Define the derivative of the function
def f' (x : ℝ) := 4 * x^3 - 2

theorem find_point_P :
  ∃ (P : ℝ × ℝ), (f' P.1 = 2) ∧ (f P.1 = P.2) ∧ (P = (1, -1)) :=
by
  -- here would go the actual proof
  sorry

end find_point_P_l14_14177


namespace quadratic_root_a_value_l14_14449

theorem quadratic_root_a_value (a k : ℝ) (h1 : k = 65) (h2 : a * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : a = 2 :=
by
  sorry

end quadratic_root_a_value_l14_14449


namespace shooter_variance_l14_14605

def scores : List ℝ := [9.7, 9.9, 10.1, 10.2, 10.1] -- Defining the scores

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length -- Calculating the mean

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length -- Defining the variance

theorem shooter_variance :
  variance scores = 0.032 :=
by
  sorry -- Proof to be provided later

end shooter_variance_l14_14605


namespace intersection_M_N_l14_14338

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14338


namespace orthogonal_pairs_zero_l14_14165

open Matrix

theorem orthogonal_pairs_zero : 
  ¬ ∃ (a d : ℝ), (fun M : Matrix (Fin 2) (Fin 2) ℝ => 
    (Mᵀ ⬝ M = (1 : Matrix (Fin 2) (Fin 2) ℝ)) ∧ 
    M = ![![a, 4], ![-9, d]]) :=
by 
  intro h 
  rcases h with ⟨a, d, orthogonal, matrix_def⟩
  rw matrix_def at orthogonal
  have eq1 : a * a + 16 = 1 := by sorry
  have eq2 : 81 + d * d = 1 := by sorry
  have eq3 : -9 * a + 4 * d = 0 := by sorry
  have h1 : ¬ ∃ a : ℝ, a * a = -15 := by
    intro h
    rcases h with ⟨a, eq⟩
    linarith
  contradiction

end orthogonal_pairs_zero_l14_14165


namespace div_binomial_expansion_l14_14051

theorem div_binomial_expansion
  (a n b : Nat)
  (hb : a^n ∣ b) :
  a^(n+1) ∣ (a+1)^b - 1 := by
  sorry

end div_binomial_expansion_l14_14051


namespace probability_between_652_760_l14_14383

noncomputable def binomial_prob (n : ℕ) (p : ℝ) : ℝ :=
  let μ := n * p
  let σ := Math.sqrt (n * p * (1 - p))
  let α := (652 - μ) / σ
  let β := (760 - μ) / σ
  toReal ((Gaussian.cdf β - Gaussian.cdf α))

theorem probability_between_652_760 (h₁: 1000 = 1000) (h₂: 0.7 = 0.7) :
  binomial_prob 1000 0.7 ≈ 0.999 :=
sorry

end probability_between_652_760_l14_14383


namespace smallest_number_of_roses_to_buy_l14_14682

-- Definitions representing the conditions
def group_size1 : ℕ := 9
def group_size2 : ℕ := 19

-- Statement representing the problem and solution
theorem smallest_number_of_roses_to_buy : Nat.lcm group_size1 group_size2 = 171 := 
by 
  sorry

end smallest_number_of_roses_to_buy_l14_14682


namespace ant_probability_after_6_minutes_l14_14890

open Probability

def is_valid_position (x y : ℤ) : Prop := -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

def valid_moves : List (ℤ × ℤ) :=
  [(1, 0), (-1, 0), (0, 1), (0, -1)]

def ant_move (pos : ℤ × ℤ) (move : ℤ × ℤ) : ℤ × ℤ :=
  (pos.1 + move.1, pos.2 + move.2)

noncomputable def probability_ant_at_C :
  ℚ := 20 * (1 / 4096) -- Combines 𝜀_perm and sequence prob

theorem ant_probability_after_6_minutes : 
  probability_ant_at_C = 5 / 1024 := 
by
  sorry

end ant_probability_after_6_minutes_l14_14890


namespace unit_vector_perpendicular_l14_14636

theorem unit_vector_perpendicular (x y : ℝ)
  (h1 : 4 * x + 2 * y = 0) 
  (h2 : x^2 + y^2 = 1) :
  (x = (Real.sqrt 5) / 5 ∧ y = -(2 * (Real.sqrt 5) / 5)) ∨ 
  (x = -(Real.sqrt 5) / 5 ∧ y = 2 * (Real.sqrt 5) / 5) :=
sorry

end unit_vector_perpendicular_l14_14636


namespace reflected_line_eq_l14_14737

noncomputable def point_symmetric_reflection :=
  ∃ (A : ℝ × ℝ) (B : ℝ × ℝ) (A' : ℝ × ℝ),
  A = (-1 / 2, 0) ∧ B = (0, 1) ∧ A' = (1 / 2, 0) ∧ 
  ∀ (x y : ℝ), 2 * x + y = 1 ↔
  (y - 1) / (0 - 1) = x / (1 / 2 - 0)

theorem reflected_line_eq :
  point_symmetric_reflection :=
sorry

end reflected_line_eq_l14_14737


namespace johns_meeting_distance_l14_14389

theorem johns_meeting_distance (d t: ℝ) 
    (h1 : d = 40 * (t + 1.5))
    (h2 : d - 40 = 60 * (t - 2)) :
    d = 420 :=
by sorry

end johns_meeting_distance_l14_14389


namespace inequality_solution_l14_14694

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l14_14694


namespace max_third_altitude_l14_14254

theorem max_third_altitude (h1 h2 : ℕ) (h1_eq : h1 = 6) (h2_eq : h2 = 18) (triangle_scalene : true)
: (exists h3 : ℕ, (∀ h3_alt > h3, h3_alt > 8)) := 
sorry

end max_third_altitude_l14_14254


namespace pieces_given_l14_14075

def pieces_initially := 38
def pieces_now := 54

theorem pieces_given : pieces_now - pieces_initially = 16 := by
  sorry

end pieces_given_l14_14075


namespace investmentAmounts_l14_14047

variable (totalInvestment : ℝ) (bonds stocks mutualFunds : ℝ)

-- Given conditions
def conditions := 
  totalInvestment = 210000 ∧
  stocks = 2 * bonds ∧
  mutualFunds = 4 * stocks ∧
  bonds + stocks + mutualFunds = totalInvestment

-- Prove the investments
theorem investmentAmounts (h : conditions totalInvestment bonds stocks mutualFunds) :
  bonds = 19090.91 ∧ stocks = 38181.82 ∧ mutualFunds = 152727.27 :=
sorry

end investmentAmounts_l14_14047


namespace f_value_2009_l14_14761

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_2009
    (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
    (h2 : f 0 ≠ 0) :
    f 2009 = 1 :=
sorry

end f_value_2009_l14_14761


namespace hawks_win_at_least_4_l14_14823

noncomputable def hawks_win_probability : ℚ :=
  let p_win := 0.8 in
  let p_lose := 1 - p_win in
  let prob_4_wins := (5.choose 4) * (p_win ^ 4) * (p_lose) in
  let prob_5_wins := (p_win ^ 5) in
  prob_4_wins + prob_5_wins

theorem hawks_win_at_least_4 : hawks_win_probability = 73728 / 100000 :=
sorry

end hawks_win_at_least_4_l14_14823


namespace fraction_inequality_l14_14690

theorem fraction_inequality (a : ℝ) (h : a ≠ 2) : (1 / (a^2 - 4 * a + 4) > 2 / (a^3 - 8)) :=
by sorry

end fraction_inequality_l14_14690


namespace determine_c_plus_d_l14_14820

theorem determine_c_plus_d (x : ℝ) (c d : ℤ) (h1 : x^2 + 5*x + (5/x) + (1/(x^2)) = 35) (h2 : x = c + Real.sqrt d) : c + d = 5 :=
sorry

end determine_c_plus_d_l14_14820


namespace cyclic_sum_inequality_l14_14514

theorem cyclic_sum_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 :=
by
  -- TODO: Provide proof here
  sorry

end cyclic_sum_inequality_l14_14514


namespace avg_annual_growth_rate_optimal_room_price_l14_14289

-- Problem 1: Average Annual Growth Rate
theorem avg_annual_growth_rate (visitors_2021 visitors_2023 : ℝ) (years : ℕ) (visitors_2021_pos : 0 < visitors_2021) :
  visitors_2023 > visitors_2021 → visitors_2023 / visitors_2021 = 2.25 → 
  ∃ x : ℝ, (1 + x)^2 = 2.25 ∧ x = 0.5 :=
by sorry

-- Problem 2: Optimal Room Price for Desired Profit
theorem optimal_room_price (rooms : ℕ) (base_price cost_per_room desired_profit : ℝ)
  (rooms_pos : 0 < rooms) :
  base_price = 180 → cost_per_room = 20 → desired_profit = 9450 → 
  ∃ y : ℝ, (y - cost_per_room) * (rooms - (y - base_price) / 10) = desired_profit ∧ y = 230 :=
by sorry

end avg_annual_growth_rate_optimal_room_price_l14_14289


namespace smallest_x_for_multiple_l14_14262

theorem smallest_x_for_multiple 
  (x : ℕ) (h₁ : ∀ m : ℕ, 450 * x = 800 * m) 
  (h₂ : ∀ y : ℕ, (∀ m : ℕ, 450 * y = 800 * m) → x ≤ y) : 
  x = 16 := 
sorry

end smallest_x_for_multiple_l14_14262


namespace exp_gt_pow_l14_14401

theorem exp_gt_pow (x : ℝ) (h_pos : 0 < x) (h_ne : x ≠ Real.exp 1) : Real.exp x > x ^ Real.exp 1 := by
  sorry

end exp_gt_pow_l14_14401


namespace percentage_owning_cats_percentage_owning_birds_l14_14790

def total_students : ℕ := 500
def students_owning_cats : ℕ := 80
def students_owning_birds : ℕ := 120

theorem percentage_owning_cats : students_owning_cats * 100 / total_students = 16 := 
by 
  sorry

theorem percentage_owning_birds : students_owning_birds * 100 / total_students = 24 := 
by 
  sorry

end percentage_owning_cats_percentage_owning_birds_l14_14790


namespace initial_people_count_25_l14_14394

-- Definition of the initial number of people (X) and the condition
def initial_people (X : ℕ) : Prop := X - 8 + 13 = 30

-- The theorem stating that the initial number of people is 25
theorem initial_people_count_25 : ∃ (X : ℕ), initial_people X ∧ X = 25 :=
by
  -- We add sorry here to skip the actual proof
  sorry

end initial_people_count_25_l14_14394


namespace average_weight_increase_l14_14243

theorem average_weight_increase (A : ℝ) (X : ℝ) (h : (8 * A - 65 + 93) / 8 = A + X) :
  X = 3.5 :=
sorry

end average_weight_increase_l14_14243


namespace intersection_M_N_l14_14332

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14332


namespace average_xyz_l14_14658

theorem average_xyz (x y z : ℝ) (h1 : x = 3) (h2 : y = 2 * x) (h3 : z = 3 * y) : 
  (x + y + z) / 3 = 9 :=
by
  sorry

end average_xyz_l14_14658


namespace platform_length_proof_l14_14118

noncomputable def train_length : ℝ := 480

noncomputable def speed_kmph : ℝ := 55

noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

noncomputable def crossing_time : ℝ := 71.99424046076314

noncomputable def total_distance_covered : ℝ := speed_mps * crossing_time

noncomputable def platform_length : ℝ := total_distance_covered - train_length

theorem platform_length_proof : platform_length = 620 := by
  sorry

end platform_length_proof_l14_14118


namespace trajectory_is_parabola_l14_14481

theorem trajectory_is_parabola
  (P : ℝ × ℝ) : 
  (dist P (0, P.2 + 1) < dist P (0, 2)) -> 
  (P.1^2 = 8 * (P.2 + 2)) :=
by
  sorry

end trajectory_is_parabola_l14_14481


namespace boxes_needed_to_pack_all_muffins_l14_14832

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end boxes_needed_to_pack_all_muffins_l14_14832


namespace total_time_in_cocoons_l14_14251

theorem total_time_in_cocoons (CA CB CC: ℝ) 
    (h1: 4 * CA = 90)
    (h2: 4 * CB = 120)
    (h3: 4 * CC = 150) 
    : CA + CB + CC = 90 := 
by
  -- To be proved
  sorry

end total_time_in_cocoons_l14_14251


namespace depth_of_first_hole_l14_14873

theorem depth_of_first_hole (n1 t1 n2 t2 : ℕ) (D : ℝ) (r : ℝ) 
  (h1 : n1 = 45) (h2 : t1 = 8) (h3 : n2 = 90) (h4 : t2 = 6) 
  (h5 : r = 1 / 12) (h6 : D = n1 * t1 * r) (h7 : n2 * t2 * r = 45) : 
  D = 30 := 
by 
  sorry

end depth_of_first_hole_l14_14873


namespace product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l14_14558

theorem product_two_smallest_one_digit_primes_and_largest_three_digit_prime :
  2 * 3 * 997 = 5982 :=
by
  sorry

end product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l14_14558


namespace river_bend_students_more_than_pets_l14_14620

theorem river_bend_students_more_than_pets 
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (hamsters_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ := students_per_classroom * number_of_classrooms)
  (total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms)
  (total_hamsters : ℕ := hamsters_per_classroom * number_of_classrooms)
  (total_pets : ℕ := total_rabbits + total_hamsters) :
  students_per_classroom = 24 ∧ rabbits_per_classroom = 2 ∧ hamsters_per_classroom = 3 ∧ number_of_classrooms = 5 →
  total_students - total_pets = 95 :=
by
  sorry

end river_bend_students_more_than_pets_l14_14620


namespace total_playing_time_scenarios_l14_14023

theorem total_playing_time_scenarios :
  (∑ (x y : ℕ) in (finset.Icc (0 : ℕ) 33).product (finset.Icc (0 : ℕ) 20),
    (if 7 * x + 13 * y = 270 then (nat.choose (x + 3) 3 * nat.choose (y + 2) 2) else 0)) = 42244 :=
by sorry

end total_playing_time_scenarios_l14_14023


namespace cannot_sum_to_nine_l14_14302

def sum_pairs (a b c d : ℕ) : List ℕ :=
  [a + b, c + d, a + c, b + d, a + d, b + c]

theorem cannot_sum_to_nine :
  ∀ (a b c d : ℕ), a ≠ 5 ∧ b ≠ 6 ∧ c ≠ 5 ∧ d ≠ 6 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b ≠ 11 ∧ a + c ≠ 11 ∧ a + d ≠ 11 ∧ b + c ≠ 11 ∧ b + d ≠ 11 ∧ c + d ≠ 11 →
  ¬9 ∈ sum_pairs a b c d :=
by
  intros a b c d h
  sorry

end cannot_sum_to_nine_l14_14302


namespace train_speed_l14_14137

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l14_14137


namespace set_intersection_l14_14350

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l14_14350


namespace cubic_roots_solve_l14_14209

-- Let a, b, c be roots of the equation x^3 - 15x^2 + 25x - 10 = 0
variables {a b c : ℝ}
def eq1 := a + b + c = 15
def eq2 := a * b + b * c + c * a = 25
def eq3 := a * b * c = 10

theorem cubic_roots_solve :
  eq1 → eq2 → eq3 → 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
by
  intros,
  sorry

end cubic_roots_solve_l14_14209


namespace rational_expression_is_rational_l14_14050

theorem rational_expression_is_rational (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ r : ℚ, 
    r = Real.sqrt ((1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2)) :=
sorry

end rational_expression_is_rational_l14_14050


namespace maximum_sum_minimum_difference_l14_14939

-- Definitions based on problem conditions
def is_least_common_multiple (m n lcm: ℕ) : Prop := Nat.lcm m n = lcm
def is_greatest_common_divisor (m n gcd: ℕ) : Prop := Nat.gcd m n = gcd

-- The target theorem to prove
theorem maximum_sum_minimum_difference (x y: ℕ) (h_lcm: is_least_common_multiple x y 2010) (h_gcd: is_greatest_common_divisor x y 2) :
  (x + y = 2012 ∧ x - y = 104 ∨ y - x = 104) :=
by
  sorry

end maximum_sum_minimum_difference_l14_14939


namespace expand_product_l14_14161

theorem expand_product (x : ℝ): (x + 4) * (x - 5 + 2) = x^2 + x - 12 :=
by 
  sorry

end expand_product_l14_14161


namespace product_closest_to_l14_14891

def is_closest_to (n target : ℝ) (options : List ℝ) : Prop :=
  ∀ o ∈ options, |n - target| ≤ |n - o|

theorem product_closest_to : is_closest_to ((2.5) * (50.5 + 0.25)) 127 [120, 125, 127, 130, 140] :=
by
  sorry

end product_closest_to_l14_14891


namespace prob_exactly_four_twos_l14_14158

-- Define the probability of success (rolling a 2)
def p (k : ℕ) (n : ℕ) : ℚ := (choose n k) * ((1/6)^k) * ((5/6)^(n-k))

-- Define the specific instance for the problem
def probability_ex4_dice_show_2 : ℚ := p 4 8

-- The main assertion
theorem prob_exactly_four_twos : probability_ex4_dice_show_2 ≈ 0.026 :=
by
  sorry -- Proof not provided, just the statement.

end prob_exactly_four_twos_l14_14158


namespace find_angle_D_l14_14766

noncomputable def measure.angle_A := 80
noncomputable def measure.angle_B := 30
noncomputable def measure.angle_C := 20

def sum_angles_pentagon (A B C : ℕ) := 540 - (A + B + C)

theorem find_angle_D
  (A B C E F : ℕ)
  (hA : A = measure.angle_A)
  (hB : B = measure.angle_B)
  (hC : C = measure.angle_C)
  (h_sum_pentagon : A + B + C + D + E + F = 540)
  (h_triangle : D + E + F = 180) :
  D = 130 :=
by
  sorry

end find_angle_D_l14_14766


namespace distance_point_to_vertical_line_l14_14246

/-- The distance from a point to a vertical line equals the absolute difference in the x-coordinates. -/
theorem distance_point_to_vertical_line (x1 y1 x2 : ℝ) (h_line : x2 = -2) (h_point : (x1, y1) = (1, 2)) :
  abs (x1 - x2) = 3 :=
by
  -- Place proof here
  sorry

end distance_point_to_vertical_line_l14_14246


namespace how_many_ducks_did_john_buy_l14_14206

def cost_price_per_duck : ℕ := 10
def weight_per_duck : ℕ := 4
def selling_price_per_pound : ℕ := 5
def profit : ℕ := 300

theorem how_many_ducks_did_john_buy (D : ℕ) (h : 10 * D - 10 * D + 10 * D = profit) : D = 30 :=
by 
  sorry

end how_many_ducks_did_john_buy_l14_14206


namespace interest_rate_A_l14_14280

-- Given conditions
variables (Principal : ℝ := 4000)
variables (interestRate_C : ℝ := 11.5 / 100)
variables (gain_B : ℝ := 180)
variables (time : ℝ := 3)
variables (interest_from_C : ℝ := Principal * interestRate_C * time)
variables (interest_to_A : ℝ := interest_from_C - gain_B)

-- The proof goal
theorem interest_rate_A (R : ℝ) : 
  1200 = Principal * (R / 100) * time → 
  R = 10 :=
by
  sorry

end interest_rate_A_l14_14280


namespace expression_for_neg_x_l14_14176

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem expression_for_neg_x (f : ℝ → ℝ) (h_odd : odd_function f) (h_nonneg : ∀ (x : ℝ), 0 ≤ x → f x = x^2 - 2 * x) :
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2 * x :=
by 
  intros x hx 
  have hx_pos : -x > 0 := by linarith 
  have h_fx_neg : f (-x) = -f x := h_odd x
  rw [h_nonneg (-x) (by linarith)] at h_fx_neg
  linarith

end expression_for_neg_x_l14_14176


namespace solution_pairs_count_l14_14907

theorem solution_pairs_count : 
  ∃ (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s → 5 * p.1 + 7 * p.2 = 708) ∧ s.card = 20 :=
sorry

end solution_pairs_count_l14_14907


namespace equation_proof_l14_14273

theorem equation_proof :
  (40 + 5 * 12) / (180 / 3^2) + Real.sqrt 49 = 12 := 
by 
  sorry

end equation_proof_l14_14273


namespace no_solution_for_k_eq_4_l14_14718

theorem no_solution_for_k_eq_4 (x k : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : (k = 4) → ¬ ((x - 3) * (x - 8) = (x - k) * (x - 4)) :=
by
  sorry

end no_solution_for_k_eq_4_l14_14718


namespace intersection_M_N_l14_14335

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14335


namespace hyperbola_eccentricity_l14_14255

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  let e := (1 + (b^2) / (a^2)).sqrt
  e

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a + b = 5)
  (h2 : a * b = 6)
  (h3 : a > b) :
  eccentricity a b = Real.sqrt 13 / 3 :=
sorry

end hyperbola_eccentricity_l14_14255


namespace profit_eqn_65_to_75_maximize_profit_with_discount_l14_14121

-- Definitions for the conditions
def total_pieces (x y : ℕ) : Prop := x + y = 100

def total_cost (x y : ℕ) : Prop := 80 * x + 60 * y ≤ 7500

def min_pieces_A (x : ℕ) : Prop := x ≥ 65

def profit_without_discount (x : ℕ) : ℕ := 10 * x + 3000

def profit_with_discount (x a : ℕ) (h1 : 0 < a) (h2 : a < 20): ℕ := (10 - a) * x + 3000

-- Proof statement
theorem profit_eqn_65_to_75 (x: ℕ) (h1: total_pieces x (100 - x)) (h2: total_cost x (100 - x)) (h3: min_pieces_A x) :
  65 ≤ x ∧ x ≤ 75 → profit_without_discount x = 10 * x + 3000 :=
by
  sorry

theorem maximize_profit_with_discount (x a : ℕ) (h1 : total_pieces x (100 - x)) (h2 : total_cost x (100 - x)) (h3 : min_pieces_A x) (h4 : 0 < a) (h5 : a < 20) :
  if a < 10 then x = 75 ∧ profit_with_discount 75 a h4 h5 = (10 - a) * 75 + 3000
  else if a = 10 then 65 ≤ x ∧ x ≤ 75 ∧ profit_with_discount x a h4 h5 = 3000
  else x = 65 ∧ profit_with_discount 65 a h4 h5 = (10 - a) * 65 + 3000 :=
by
  sorry

end profit_eqn_65_to_75_maximize_profit_with_discount_l14_14121


namespace contradiction_assumption_l14_14113

-- Proposition P: "Among a, b, c, d, at least one is negative"
def P (a b c d : ℝ) : Prop :=
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0

-- Correct assumption when using contradiction: all are non-negative
def notP (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof problem statement: assuming notP leads to contradiction to prove P
theorem contradiction_assumption (a b c d : ℝ) (h : ¬ P a b c d) : notP a b c d :=
by
  sorry

end contradiction_assumption_l14_14113


namespace one_and_one_third_of_x_is_36_l14_14810

theorem one_and_one_third_of_x_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := 
sorry

end one_and_one_third_of_x_is_36_l14_14810


namespace total_people_who_eat_vegetarian_l14_14428

def people_who_eat_only_vegetarian := 16
def people_who_eat_both_vegetarian_and_non_vegetarian := 12

-- We want to prove that the total number of people who eat vegetarian is 28
theorem total_people_who_eat_vegetarian : 
  people_who_eat_only_vegetarian + people_who_eat_both_vegetarian_and_non_vegetarian = 28 :=
by 
  sorry

end total_people_who_eat_vegetarian_l14_14428


namespace chord_length_cut_by_line_l14_14316

theorem chord_length_cut_by_line {x y : ℝ} (h_line : y = 3 * x) (h_circle : (x + 1) ^ 2 + (y - 2) ^ 2 = 25) :
  ∃ x1 x2 y1 y2, 
    (y1 = 3 * x1) ∧ (y2 = 3 * x2) ∧ 
    ((x1 + 1) ^ 2 + (y1 - 2) ^ 2 = 25) ∧ ((x2 + 1) ^ 2 + (y2 - 2) ^ 2 = 25) ∧ 
    (real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 3 * real.sqrt 10) := sorry

end chord_length_cut_by_line_l14_14316


namespace petya_numbers_board_l14_14522

theorem petya_numbers_board (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k → k < n → (∀ d : ℕ, 4 ∣ 10 ^ d → ¬(4 ∣ k))) 
  (h3 : ∀ k : ℕ, 0 ≤ k → k < n→ (∀ d : ℕ, 7 ∣ 10 ^ d → ¬(7 ∣ (k + n - 1)))) : 
  ∃ x : ℕ, (x = 2021) := 
by
  sorry

end petya_numbers_board_l14_14522


namespace value_of_n_l14_14673

theorem value_of_n (n : ℕ) (h1 : 0 < n) (h2 : n < Real.sqrt 65) (h3 : Real.sqrt 65 < n + 1) : n = 8 := 
sorry

end value_of_n_l14_14673


namespace mikails_age_l14_14397

-- Define the conditions
def dollars_per_year_old : ℕ := 5
def total_dollars_given : ℕ := 45

-- Main theorem statement
theorem mikails_age (age : ℕ) : (age * dollars_per_year_old = total_dollars_given) → age = 9 :=
by
  sorry

end mikails_age_l14_14397


namespace hyperbola_eccentricity_l14_14357

theorem hyperbola_eccentricity 
  (a b : ℝ) (h1 : 2 * (1 : ℝ) + 1 = 0) (h2 : 0 < a) (h3 : 0 < b) 
  (h4 : b = 2 * a) : 
  (∃ e : ℝ, e = (Real.sqrt 5)) 
:= 
  sorry

end hyperbola_eccentricity_l14_14357


namespace rectangular_field_area_l14_14590

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l14_14590


namespace max_value_of_expression_l14_14767

theorem max_value_of_expression 
  (x y : ℝ)
  (h : x^2 + y^2 = 20 * x + 9 * y + 9) :
  ∃ x y : ℝ, 4 * x + 3 * y = 83 := sorry

end max_value_of_expression_l14_14767


namespace evaluate_x_l14_14463

variable {R : Type*} [LinearOrderedField R]

theorem evaluate_x (m n k x : R) (hm : m ≠ 0) (hn : n ≠ 0) (h : m ≠ n) (h_eq : (x + m)^2 - (x + n)^2 = k * (m - n)^2) :
  x = ((k - 1) * (m + n) - 2 * k * n) / 2 :=
by
  sorry

end evaluate_x_l14_14463


namespace tailor_cut_difference_l14_14446

def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

theorem tailor_cut_difference : skirt_cut - pants_cut = 0.25 :=
by
  sorry

end tailor_cut_difference_l14_14446


namespace cubic_root_sum_cubed_l14_14965

theorem cubic_root_sum_cubed
  (p q r : ℂ)
  (h1 : 3 * p^3 - 9 * p^2 + 27 * p - 6 = 0)
  (h2 : 3 * q^3 - 9 * q^2 + 27 * q - 6 = 0)
  (h3 : 3 * r^3 - 9 * r^2 + 27 * r - 6 = 0)
  (hpq : p ≠ q)
  (hqr : q ≠ r)
  (hrp : r ≠ p) :
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 585 := 
  sorry

end cubic_root_sum_cubed_l14_14965


namespace centroids_and_orthocenters_loci_are_rays_l14_14735

variables (S : Point) (T : TrihedralAngle) (P : FamilyOfParallelPlanes)

-- Definition of the trihedral angle with vertex S
def is_trihedral_angle_with_vertex (T : TrihedralAngle) (S : Point) : Prop :=
  T.vertex = S

-- Definition of centroids' locus as a ray from S.
def locus_of_centroids_is_ray_from (P : FamilyOfParallelPlanes) (T : TrihedralAngle) (S : Point) : Prop :=
  ∀ (t : Triangle), (t ∈ (P ∩ T.faces)) → (∃ (r : Ray), r.origin = S ∧ t.centroid ∈ r)

-- Definition of orthocenters' locus as a ray from S.
def locus_of_orthocenters_is_ray_from (P : FamilyOfParallelPlanes) (T : TrihedralAngle) (S : Point) : Prop :=
  ∀ (t : Triangle), (t ∈ (P ∩ T.faces)) → (∃ (r : Ray), r.origin = S ∧ t.orthocenter ∈ r)

-- Theorem statement combining both centroids and orthocenters loci properties.
theorem centroids_and_orthocenters_loci_are_rays (S : Point) (T : TrihedralAngle) (P : FamilyOfParallelPlanes)
  (hT : is_trihedral_angle_with_vertex T S) :
  locus_of_centroids_is_ray_from P T S ∧ locus_of_orthocenters_is_ray_from P T S :=
sorry

end centroids_and_orthocenters_loci_are_rays_l14_14735


namespace perpendicular_k_value_exists_l14_14362

open Real EuclideanSpace

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b : ℝ × ℝ := (3, 2)

theorem perpendicular_k_value_exists : ∃ k : ℝ, (vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) ∧ k = 5 / 4 := by
  sorry

end perpendicular_k_value_exists_l14_14362


namespace symmetric_line_equation_l14_14374

theorem symmetric_line_equation (l : ℝ × ℝ → Prop)
  (h1 : ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0)
  (h2 : ∀ p : ℝ × ℝ, l p ↔ p = (0, 2) ∨ p = ⟨-3, 2⟩) :
  ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0 :=
by
  sorry

end symmetric_line_equation_l14_14374


namespace cylinder_ellipse_major_axis_l14_14284

-- Given a right circular cylinder of radius 2
-- and a plane intersecting it forming an ellipse
-- with the major axis being 50% longer than the minor axis,
-- prove that the length of the major axis is 6.

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ) (major minor : ℝ),
    r = 2 → major = 1.5 * minor → minor = 2 * r → major = 6 :=
by
  -- Proof step to be filled by the prover.
  sorry

end cylinder_ellipse_major_axis_l14_14284


namespace fifth_hexagon_dots_l14_14322

-- Definitions as per conditions
def dots_in_nth_layer (n : ℕ) : ℕ := 6 * (n + 2)

-- Function to calculate the total number of dots in the nth hexagon
def total_dots_in_hexagon (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + dots_in_nth_layer k) (dots_in_nth_layer 0)

-- The proof problem statement
theorem fifth_hexagon_dots : total_dots_in_hexagon 5 = 150 := sorry

end fifth_hexagon_dots_l14_14322


namespace area_of_rectangle_l14_14577

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l14_14577


namespace cube_volume_l14_14066

theorem cube_volume (s : ℝ) (h : 12 * s = 96) : s^3 = 512 :=
by
  sorry

end cube_volume_l14_14066


namespace colten_chickens_l14_14526

theorem colten_chickens (x : ℕ) (Quentin Skylar Colten : ℕ) 
  (h1 : Quentin + Skylar + Colten = 383)
  (h2 : Quentin = 25 + 2 * Skylar)
  (h3 : Skylar = 3 * Colten - 4) : 
  Colten = 37 := 
  sorry

end colten_chickens_l14_14526


namespace kim_gum_distribution_l14_14956

theorem kim_gum_distribution (cousins : ℕ) (total_gum : ℕ) 
  (h1 : cousins = 4) (h2 : total_gum = 20) : 
  total_gum / cousins = 5 :=
by
  sorry

end kim_gum_distribution_l14_14956


namespace greatest_third_side_l14_14086

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l14_14086


namespace fraction_is_three_eights_l14_14571

-- The given number
def number := 48

-- The fraction 'x' by which the number exceeds by 30
noncomputable def fraction (x : ℝ) : Prop :=
number = number * x + 30

-- Our goal is to prove that the fraction is 3/8
theorem fraction_is_three_eights : fraction (3 / 8) :=
by
  -- We reduced the goal proof to a simpler form for illustration, you can solve it rigorously
  sorry

end fraction_is_three_eights_l14_14571


namespace rectangle_area_is_243_square_meters_l14_14599

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l14_14599


namespace value_range_abs_function_l14_14067

theorem value_range_abs_function : 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 9 → 1 ≤ (abs (x - 3) + 1) ∧ (abs (x - 3) + 1) ≤ 7 :=
by
  intro x hx
  sorry

end value_range_abs_function_l14_14067


namespace boxes_needed_to_pack_all_muffins_l14_14831

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end boxes_needed_to_pack_all_muffins_l14_14831


namespace train_time_36kmph_200m_l14_14725

/-- How many seconds will a train 200 meters long running at the rate of 36 kmph take to pass a certain telegraph post? -/
def time_to_pass_post (length_of_train : ℕ) (speed_kmph : ℕ) : ℕ :=
  length_of_train * 3600 / (speed_kmph * 1000)

theorem train_time_36kmph_200m : time_to_pass_post 200 36 = 20 := by
  sorry

end train_time_36kmph_200m_l14_14725


namespace triangle_third_side_l14_14092

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l14_14092


namespace boat_distance_against_stream_l14_14794

-- Definitions from Step a)
def speed_boat_still_water : ℝ := 15  -- speed of the boat in still water in km/hr
def distance_downstream : ℝ := 21  -- distance traveled downstream in one hour in km
def time_hours : ℝ := 1  -- time in hours

-- Translation of the described problem proof
theorem boat_distance_against_stream :
  ∃ (v_s : ℝ), (speed_boat_still_water + v_s = distance_downstream / time_hours) → 
               (15 - v_s = 9) :=
by
  sorry

end boat_distance_against_stream_l14_14794


namespace new_person_weight_l14_14564

theorem new_person_weight (W : ℝ) (N : ℝ)
  (h1 : ∀ avg_increase : ℝ, avg_increase = 2.5 → N = 55) 
  (h2 : ∀ original_weight : ℝ, original_weight = 35) 
  : N = 55 := 
by 
  sorry

end new_person_weight_l14_14564


namespace determine_M_l14_14462

theorem determine_M (M : ℕ) (h : 12 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2) : M = 36 :=
by
  sorry

end determine_M_l14_14462


namespace cubic_roots_solve_l14_14210

-- Let a, b, c be roots of the equation x^3 - 15x^2 + 25x - 10 = 0
variables {a b c : ℝ}
def eq1 := a + b + c = 15
def eq2 := a * b + b * c + c * a = 25
def eq3 := a * b * c = 10

theorem cubic_roots_solve :
  eq1 → eq2 → eq3 → 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
by
  intros,
  sorry

end cubic_roots_solve_l14_14210


namespace complex_expression_value_l14_14745

theorem complex_expression_value :
  ((6^2 - 4^2) + 2)^3 / 2 = 5324 :=
by
  sorry

end complex_expression_value_l14_14745


namespace amy_lily_tie_l14_14196

noncomputable def tie_probability : ℚ :=
    let amy_win := (2 / 5 : ℚ)
    let lily_win := (1 / 4 : ℚ)
    let total_win := amy_win + lily_win
    1 - total_win

theorem amy_lily_tie (h1 : (2 / 5 : ℚ) = 2 / 5) 
                     (h2 : (1 / 4 : ℚ) = 1 / 4)
                     (h3 : (2 / 5 : ℚ) ≥ 2 * (1 / 4 : ℚ) ∨ (1 / 4 : ℚ) ≥ 2 * (2 / 5 : ℚ)) :
    tie_probability = 7 / 20 :=
by
  sorry

end amy_lily_tie_l14_14196


namespace find_BG_l14_14554

-- Define given lengths and the required proof
def BC : ℝ := 5
def BF : ℝ := 12

theorem find_BG : BG = 13 := by
  -- Formal proof would go here
  sorry

end find_BG_l14_14554


namespace rectangular_field_area_l14_14591

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l14_14591


namespace maxwell_walking_speed_l14_14042

open Real

theorem maxwell_walking_speed (v : ℝ) : 
  (∀ (v : ℝ), (4 * v + 6 * 3 = 34)) → v = 4 :=
by
  intros
  have h1 : 4 * v + 18 = 34 := by sorry
  have h2 : 4 * v = 16 := by sorry
  have h3 : v = 4 := by sorry
  exact h3

end maxwell_walking_speed_l14_14042


namespace valid_p_interval_l14_14748

theorem valid_p_interval :
  ∀ p, (∀ q, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 0 ≤ p ∧ p < 4 :=
sorry

end valid_p_interval_l14_14748


namespace cordelia_bleaching_l14_14152

noncomputable def bleaching_time (B : ℝ) : Prop :=
  B + 4 * B + B / 3 = 10

theorem cordelia_bleaching : ∃ B : ℝ, bleaching_time B ∧ B = 1.875 :=
by {
  sorry
}

end cordelia_bleaching_l14_14152


namespace intersection_M_N_l14_14343

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l14_14343


namespace total_money_spent_on_clothing_l14_14683

theorem total_money_spent_on_clothing (cost_shirt cost_jacket : ℝ)
  (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) :
  cost_shirt + cost_jacket = 25.31 :=
sorry

end total_money_spent_on_clothing_l14_14683


namespace number_of_games_in_season_l14_14884

-- Define the number of teams and divisions
def num_teams := 20
def num_divisions := 4
def teams_per_division := 5

-- Define the games played within and between divisions
def intra_division_games_per_team := 12  -- 4 teams * 3 games each
def inter_division_games_per_team := 15  -- (20 - 5) teams * 1 game each

-- Define the total number of games played by each team
def total_games_per_team := intra_division_games_per_team + inter_division_games_per_team

-- Define the total number of games played (double-counting needs to be halved)
def total_games (num_teams : ℕ) (total_games_per_team : ℕ) : ℕ :=
  (num_teams * total_games_per_team) / 2

-- The theorem to be proven
theorem number_of_games_in_season :
  total_games num_teams total_games_per_team = 270 :=
by
  sorry

end number_of_games_in_season_l14_14884


namespace number_of_passed_candidates_l14_14985

theorem number_of_passed_candidates
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 4200) : P = 100 :=
sorry

end number_of_passed_candidates_l14_14985


namespace value_of_b_l14_14257

theorem value_of_b (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : b - a = 1) : b = 1 := 
by 
  sorry

end value_of_b_l14_14257


namespace ratio_of_time_spent_l14_14378

theorem ratio_of_time_spent {total_minutes type_a_minutes type_b_minutes : ℝ}
  (h1 : total_minutes = 180)
  (h2 : type_a_minutes = 32.73)
  (h3 : type_b_minutes = total_minutes - type_a_minutes) :
  type_a_minutes / type_a_minutes = 1 ∧ type_b_minutes / type_a_minutes = 4.5 := by
  sorry

end ratio_of_time_spent_l14_14378


namespace range_of_m_l14_14485
noncomputable def f (x : ℝ) : ℝ := ((x - 1) / (x + 1))^2

noncomputable def f_inv (x : ℝ) : ℝ := (1 + Real.sqrt x) / (1 - Real.sqrt x)

theorem range_of_m {x : ℝ} (m : ℝ) (h1 : 1 / 16 ≤ x) (h2 : x ≤ 1 / 4) 
  (h3 : ∀ (x : ℝ), (1 - Real.sqrt x) * f_inv x > m * (m - Real.sqrt x)): 
  -1 < m ∧ m < 5 / 4 :=
sorry

end range_of_m_l14_14485


namespace quadratic_equation_roots_l14_14464

theorem quadratic_equation_roots {x y : ℝ}
  (h1 : x + y = 10)
  (h2 : |x - y| = 4)
  (h3 : x * y = 21) : (x - 7) * (x - 3) = 0 ∨ (x - 3) * (x - 7) = 0 :=
by
  sorry

end quadratic_equation_roots_l14_14464


namespace sum_k1_k2_k3_l14_14017

theorem sum_k1_k2_k3 :
  ∀ (k1 k2 k3 t1 t2 t3 : ℝ),
  t1 = 105 →
  t2 = 80 →
  t3 = 45 →
  t1 = (5 / 9) * (k1 - 32) →
  t2 = (5 / 9) * (k2 - 32) →
  t3 = (5 / 9) * (k3 - 32) →
  k1 + k2 + k3 = 510 :=
by
  intros k1 k2 k3 t1 t2 t3 ht1 ht2 ht3 ht1k1 ht2k2 ht3k3
  sorry

end sum_k1_k2_k3_l14_14017


namespace rectangular_field_area_l14_14584

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l14_14584


namespace intersection_M_N_l14_14334

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l14_14334


namespace colten_chickens_l14_14530

variable (C Q S : ℕ)

-- Conditions
def condition1 : Prop := Q + S + C = 383
def condition2 : Prop := Q = 2 * S + 25
def condition3 : Prop := S = 3 * C - 4

-- Theorem to prove
theorem colten_chickens : condition1 C Q S ∧ condition2 C Q S ∧ condition3 C Q S → C = 37 := by
  sorry

end colten_chickens_l14_14530


namespace find_x_in_PetyaSequence_l14_14523

noncomputable def PetyaSequence (n : ℕ) : Prop :=
n ≥ 2 ∧ 
(∀ k, (0 ≤ k ∧ k < n → ∀ d, d ≠ 4 ∧ (to_list (k / 10 : ℕ).digit.to_string).get 1 d ≠ some '4')) ∧
(∀ d, (to_list ((n - 1) / 10 : ℕ).digit.to_string).get 1 d ≠ some '7) ∧
(∃ a b : ℕ, Prime a ∧ Prime b ∧ a ≠ b ∧ b = a + 4 ∧ (10 ∣ (((a + b) / 2) - 5)) ∧
 ∃ x : ℕ, x = a * b ∧ x % 100 = 21 ∧ x = 2021)

theorem find_x_in_PetyaSequence (n : ℕ) (h : PetyaSequence n) : ∃ x : ℕ, x = 2021 := by
  sorry

end find_x_in_PetyaSequence_l14_14523


namespace find_a_value_l14_14629

theorem find_a_value (a x y : ℝ) :
  (|y| + |y - x| ≤ a - |x - 1| ∧ (y - 4) * (y + 3) ≥ (4 - x) * (3 + x)) → a = 7 :=
by
  sorry

end find_a_value_l14_14629


namespace quadratic_roots_sign_l14_14674

theorem quadratic_roots_sign (p q : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x * y = q ∧ x + y = -p) ↔ q < 0 :=
sorry

end quadratic_roots_sign_l14_14674


namespace unique_solution_of_inequality_l14_14896

open Real

theorem unique_solution_of_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 + 2 * b * x + 2 * b| ≤ 1) ↔ b = 1 := 
by exact sorry

end unique_solution_of_inequality_l14_14896


namespace num_words_at_least_one_vowel_l14_14366

-- Definitions based on conditions.
def letters : List Char := ['A', 'B', 'E', 'G', 'H']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'G', 'H']

-- The main statement posing the question and answer.
theorem num_words_at_least_one_vowel :
  let total_words := (letters.length) ^ 5
  let consonant_words := (consonants.length) ^ 5
  let result := total_words - consonant_words
  result = 2882 :=
by {
  let total_words := 5 ^ 5
  let consonant_words := 3 ^ 5
  let result := total_words - consonant_words
  have : result = 2882 := by sorry
  exact this
}

end num_words_at_least_one_vowel_l14_14366


namespace colten_chickens_l14_14528

variable (Colten Skylar Quentin : ℕ)

def chicken_problem_conditions :=
  (Skylar = 3 * Colten - 4) ∧
  (Quentin = 6 * Skylar + 17) ∧
  (Colten + Skylar + Quentin = 383)

theorem colten_chickens (h : chicken_problem_conditions Colten Skylar Quentin) : Colten = 37 :=
sorry

end colten_chickens_l14_14528


namespace find_some_number_l14_14784

def op (x w : ℕ) := (2^x) / (2^w)

theorem find_some_number (n : ℕ) (hn : 0 < n) : (op (op 4 n) n) = 4 → n = 2 :=
by
  sorry

end find_some_number_l14_14784


namespace proof_of_inequality_proof_of_coprime_l14_14123

-- Define the probability that ab + 2c >= abc
def probabilityOfInequality : ℚ :=
  -- Total outcomes: 6^3 = 216
  -- Favorable outcomes: 58
  29 / 108

-- Define the probability that ab + 2c and 2abc are coprime
def probabilityOfCoprime : ℚ :=
  -- Total outcomes: 216
  -- Favorable outcomes: 39
  13 / 72

-- Prove the probability of the inequality
theorem proof_of_inequality :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6) →
  P(ab + 2c ≥ abc) = probabilityOfInequality :=
begin
  sorry
end

-- Prove the probability of being coprime
theorem proof_of_coprime :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6) →
  P(gcd(ab + 2c, 2abc) = 1) = probabilityOfCoprime :=
begin
  sorry
end

end proof_of_inequality_proof_of_coprime_l14_14123


namespace mike_washed_cars_l14_14032

theorem mike_washed_cars 
    (total_work_time : ℕ := 4 * 60) 
    (wash_time : ℕ := 10)
    (oil_change_time : ℕ := 15) 
    (tire_change_time : ℕ := 30) 
    (num_oil_changes : ℕ := 6) 
    (num_tire_changes : ℕ := 2) 
    (remaining_time : ℕ := total_work_time - (num_oil_changes * oil_change_time + num_tire_changes * tire_change_time))
    (num_cars_washed : ℕ := remaining_time / wash_time) :
    num_cars_washed = 9 := by
  sorry

end mike_washed_cars_l14_14032


namespace value_of_expression_l14_14324

theorem value_of_expression (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : 
  (x - 2 * y + 3 * z) / (x + y + z) = 8 / 9 := 
  sorry

end value_of_expression_l14_14324


namespace triangle_third_side_l14_14094

noncomputable def greatest_valid_side (a b : ℕ) : ℕ :=
  Nat.floor_real ((a + b : ℕ) - 1 : ℕ_real)

theorem triangle_third_side (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
    greatest_valid_side a b = 14 := by
  sorry

end triangle_third_side_l14_14094


namespace probability_co_captains_l14_14910

theorem probability_co_captains :
  let teams := [{size := 6, co_captains := 3}, {size := 8, co_captains := 3},
                {size := 9, co_captains := 3}, {size := 11, co_captains := 3}] in
  let prob (team : {size : ℕ, co_captains : ℕ}) :=
    (team.co_captains * (team.co_captains - 1)) / (team.size * (team.size - 1)) in
  let weighted_sum :=
    (1 / 4 : ℚ) * ((prob teams[0]) + (prob teams[1]) + (prob teams[2]) + (prob teams[3])) in
  weighted_sum = (1115 / 18480 : ℚ) :=
begin
  sorry
end

end probability_co_captains_l14_14910


namespace find_number_of_girls_l14_14321

-- Definitions
variables (B G : ℕ)
variables (total children_holding_boys_hand children_holding_girls_hand : ℕ)
variables (children_counted_twice : ℕ)

-- Conditions
axiom cond1 : B + G = 40
axiom cond2 : children_holding_boys_hand = 22
axiom cond3 : children_holding_girls_hand = 30
axiom cond4 : total = 40

-- Goal
theorem find_number_of_girls (h : children_counted_twice = children_holding_boys_hand + children_holding_girls_hand - total) :
  G = 24 :=
sorry

end find_number_of_girls_l14_14321


namespace greatest_integer_third_side_l14_14089

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l14_14089


namespace cube_splitting_odd_numbers_l14_14633

theorem cube_splitting_odd_numbers (m : ℕ) (h1 : m > 1) (h2 : ∃ k, 2 * k + 1 = 333) : m = 18 :=
sorry

end cube_splitting_odd_numbers_l14_14633


namespace intersection_A_B_l14_14174

-- Define the conditions of set A and B using the given inequalities and constraints
def set_A : Set ℤ := {x | -2 < x ∧ x < 3}
def set_B : Set ℤ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the proof problem translating conditions and question to Lean
theorem intersection_A_B : (set_A ∩ set_B) = {0, 1, 2} := by
  sorry

end intersection_A_B_l14_14174


namespace cannot_reach_target_l14_14200

def initial_price : ℕ := 1
def annual_increment : ℕ := 1
def tripling_year (n : ℕ) : ℕ := 3 * n
def total_years : ℕ := 99
def target_price : ℕ := 152
def incremental_years : ℕ := 98

noncomputable def final_price (x : ℕ) : ℕ := 
  initial_price + incremental_years * annual_increment + tripling_year x - annual_increment

theorem cannot_reach_target (p : ℕ) (h : p = final_price p) : p ≠ target_price :=
sorry

end cannot_reach_target_l14_14200


namespace rectangle_area_is_243_square_meters_l14_14600

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l14_14600


namespace smaller_solution_quadratic_equation_l14_14859

theorem smaller_solution_quadratic_equation :
  (∀ x : ℝ, x^2 + 7 * x - 30 = 0 → x = -10 ∨ x = 3) → -10 = min (-10) 3 :=
by
  sorry

end smaller_solution_quadratic_equation_l14_14859


namespace quadratic_function_solution_l14_14778

theorem quadratic_function_solution (m : ℝ) :
  (m^2 - 2 = 2) ∧ (m + 2 ≠ 0) → m = 2 :=
by
  intro h
  cases h with h1 h2
  have h3 : m^2 = 4 := by linarith
  have h4 : m = 2 ∨ m = -2 := by nlinarith
  cases h4
  · exact h4
  · contradiction

end quadratic_function_solution_l14_14778


namespace area_of_BEIH_l14_14724

def calculate_area_of_quadrilateral (A B C D E F I H : (ℝ × ℝ)) : ℝ := 
  sorry

theorem area_of_BEIH : 
  let A := (0, 3)
  let B := (0, 0)
  let C := (3, 0)
  let D := (3, 3)
  let E := (0, 1.5)
  let F := (1, 0)
  let I := (3 / 5, 9 / 5)
  let H := (3 / 4, 3 / 4)
  calculate_area_of_quadrilateral A B C D E F I H = 27 / 40 :=
sorry

end area_of_BEIH_l14_14724


namespace intersection_M_N_l14_14345

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l14_14345


namespace find_g_neg_6_l14_14961

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end find_g_neg_6_l14_14961


namespace one_million_div_one_fourth_l14_14368

theorem one_million_div_one_fourth : (1000000 : ℝ) / (1 / 4) = 4000000 := by
  sorry

end one_million_div_one_fourth_l14_14368


namespace non_zero_number_is_9_l14_14727

theorem non_zero_number_is_9 (x : ℝ) (hx : x ≠ 0) (h : (x + x^2) / 2 = 5 * x) : x = 9 :=
sorry

end non_zero_number_is_9_l14_14727


namespace largest_n_unique_k_l14_14856

theorem largest_n_unique_k (n : ℕ) (h : ∃ k : ℕ, (9 / 17 : ℚ) < n / (n + k) ∧ n / (n + k) < (8 / 15 : ℚ) ∧ ∀ k' : ℕ, ((9 / 17 : ℚ) < n / (n + k') ∧ n / (n + k') < (8 / 15 : ℚ)) → k' = k) : n = 72 :=
sorry

end largest_n_unique_k_l14_14856


namespace probability_calculation_l14_14318

open Classical

def probability_odd_sum_given_even_product :=
  let num_even := 4  -- even numbers: 2, 4, 6, 8
  let num_odd := 4   -- odd numbers: 1, 3, 5, 7
  let total_outcomes := 8^5
  let prob_all_odd := (num_odd / 8)^5
  let prob_even_product := 1 - prob_all_odd

  let ways_one_odd := 5 * num_odd * num_even^4
  let ways_three_odd := Nat.choose 5 3 * num_odd^3 * num_even^2
  let ways_five_odd := num_odd^5

  let favorable_outcomes := ways_one_odd + ways_three_odd + ways_five_odd
  let total_even_product_outcomes := total_outcomes * prob_even_product

  favorable_outcomes / total_even_product_outcomes

theorem probability_calculation :
  probability_odd_sum_given_even_product = rational_result := sorry

end probability_calculation_l14_14318


namespace grades_with_fewer_students_l14_14250

-- Definitions of the involved quantities
variables (G1 G2 G5 G1_2 : ℕ)
variables (Set_X : ℕ)

-- Conditions given in the problem
theorem grades_with_fewer_students (h1: G1_2 = Set_X + 30) (h2: G5 = G1 - 30) :
  exists Set_X, G1_2 - Set_X = 30 :=
by 
  sorry

end grades_with_fewer_students_l14_14250


namespace second_integer_is_64_l14_14548

theorem second_integer_is_64
  (n : ℤ)
  (h1 : (n - 2) + (n + 2) = 128) :
  n = 64 := 
  sorry

end second_integer_is_64_l14_14548


namespace total_buttons_needed_l14_14532

def shirtsMonday : ℕ := 4
def shirtsTuesday : ℕ := 3
def shirtsWednesday : ℕ := 2
def buttonsPerShirt : ℕ := 5

theorem total_buttons_needed : (shirtsMonday + shirtsTuesday + shirtsWednesday) * buttonsPerShirt = 45 :=
by
  have shirtsTotal : ℕ := shirtsMonday + shirtsTuesday + shirtsWednesday
  have buttonsTotal : ℕ := shirtsTotal * buttonsPerShirt
  have h1 := rfl
  rw [← h1, add_assoc, ← add_assoc shirtsTuesday shirtsWednesday, add_comm shirtsThursday shirtsWednesday, add_assoc shirtsTuesday shirtsWednesday shirtsMonday] at h1
  rw [← h1, mul_add, ← add_mul] at h1
  sorry

end total_buttons_needed_l14_14532


namespace average_score_of_class_l14_14866

-- Definitions based on the conditions
def class_size : ℕ := 20
def group1_size : ℕ := 10
def group2_size : ℕ := 10
def group1_avg_score : ℕ := 80
def group2_avg_score : ℕ := 60

-- Average score of the whole class
theorem average_score_of_class : 
  (group1_size * group1_avg_score + group2_size * group2_avg_score) / class_size = 70 := 
by sorry

end average_score_of_class_l14_14866


namespace rectangle_minimal_area_l14_14742

theorem rectangle_minimal_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (l + w) = 120) : l * w = 675 :=
by
  -- Proof will go here
  sorry

end rectangle_minimal_area_l14_14742


namespace change_in_expression_l14_14770

theorem change_in_expression (x a : ℝ) (ha : 0 < a) :
  (x^3 - 3*x + 1) + (3*a*x^2 + 3*a^2*x + a^3 - 3*a) = (x + a)^3 - 3*(x + a) + 1 ∧
  (x^3 - 3*x + 1) + (-3*a*x^2 + 3*a^2*x - a^3 + 3*a) = (x - a)^3 - 3*(x - a) + 1 :=
by sorry

end change_in_expression_l14_14770


namespace sam_gave_plums_l14_14043

variable (initial_plums : ℝ) (total_plums : ℝ) (plums_given : ℝ)

theorem sam_gave_plums (h1 : initial_plums = 7.0) (h2 : total_plums = 10.0) (h3 : total_plums = initial_plums + plums_given) :
  plums_given = 3 := 
by
  sorry

end sam_gave_plums_l14_14043


namespace passed_candidates_l14_14982

theorem passed_candidates (P F : ℕ) (h1 : P + F = 120) (h2 : 39 * P + 15 * F = 35 * 120) : P = 100 :=
by
  sorry

end passed_candidates_l14_14982


namespace boxes_needed_l14_14830

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end boxes_needed_l14_14830


namespace area_triangle_DEF_l14_14201

noncomputable def triangleDEF (DE EF DF : ℝ) (angleDEF : ℝ) : ℝ :=
  if angleDEF = 60 ∧ DF = 3 ∧ EF = 6 / Real.sqrt 3 then
    1 / 2 * DE * EF * Real.sin (Real.pi / 3)
  else
    0

theorem area_triangle_DEF :
  triangleDEF (Real.sqrt 3) (6 / Real.sqrt 3) 3 60 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end area_triangle_DEF_l14_14201


namespace inequality_arith_geo_mean_l14_14354

theorem inequality_arith_geo_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
by
  sorry

end inequality_arith_geo_mean_l14_14354


namespace average_birds_seen_l14_14395

theorem average_birds_seen (marcus birds: ℕ) (humphrey birds: ℕ) (darrel birds: ℕ) (isabella birds: ℕ) :
  marcus = 7 ∧ humphrey = 11 ∧ darrel = 9 ∧ isabella = 15 →
  (marcus + humphrey + darrel + isabella) / 4 = 10.5 :=
by
  intros h
  rcases h with ⟨h_marcus, ⟨h_humphrey, ⟨h_darrel, h_isabella⟩⟩⟩
  simp [h_marcus, h_humphrey, h_darrel, h_isabella]
  norm_num
  sorry

end average_birds_seen_l14_14395


namespace solution_l14_14213

noncomputable def problem (a b c : ℝ) : Prop :=
  (Polynomial.eval a (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval b (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval c (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)

theorem solution (a b c : ℝ) (h : problem a b c) : 
  (∃ abc : ℝ, abc = a * b * c ∧ abc = 10) →
  (a + b + c = 15) ∧ (a * b + b * c + c * a = 25) →
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
sorry

end solution_l14_14213


namespace solution_l14_14681

theorem solution (A B C : ℚ) (h1 : A + B = 10) (h2 : 2 * A = 3 * B + 5) (h3 : A * B * C = 120) :
  A = 7 ∧ B = 3 ∧ C = 40 / 7 := by
  sorry

end solution_l14_14681


namespace sum_x_coords_Q3_is_132_l14_14966

noncomputable def sum_x_coords_Q3 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) : ℝ :=
  sum_x1 -- given sum_x1 is the sum of x-coordinates of Q1 i.e., 132

theorem sum_x_coords_Q3_is_132 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) (h: sum_x1 = 132) :
  sum_x_coords_Q3 x_coords sum_x1 = 132 :=
by
  sorry

end sum_x_coords_Q3_is_132_l14_14966


namespace solve_equation_in_nat_l14_14817

theorem solve_equation_in_nat {x y : ℕ} :
  (x - 1) / (1 + (x - 1) * y) + (y - 1) / (2 * y - 1) = x / (x + 1) →
  x = 2 ∧ y = 2 :=
by
  sorry

end solve_equation_in_nat_l14_14817


namespace sum_of_three_numbers_l14_14838

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a <= 10) (h2 : 10 <= c)
  (h3 : (a + 10 + c) / 3 = a + 8)
  (h4 : (a + 10 + c) / 3 = c - 20) :
  a + 10 + c = 66 :=
by
  sorry

end sum_of_three_numbers_l14_14838


namespace min_product_log_condition_l14_14914

theorem min_product_log_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : Real.log a / Real.log 2 * Real.log b / Real.log 2 = 1) : 4 ≤ a * b :=
by
  sorry

end min_product_log_condition_l14_14914


namespace sequence_to_geometric_l14_14728

variable (a : ℕ → ℝ)

def seq_geom (a : ℕ → ℝ) : Prop :=
∀ m n, a (m + n) = a m * a n

def condition (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 2) = a n * a (n + 1)

theorem sequence_to_geometric (a1 a2 : ℝ) (h1 : a 1 = a1) (h2 : a 2 = a2) (h : ∀ n, a (n + 2) = a n * a (n + 1)) :
  a1 = 1 → a2 = 1 → seq_geom a :=
by
  intros ha1 ha2
  have h_seq : ∀ n, a n = 1 := sorry
  intros m n
  sorry

end sequence_to_geometric_l14_14728


namespace find_number_of_partners_l14_14441

noncomputable def law_firm_partners (P A : ℕ) : Prop :=
  (P / A = 3 / 97) ∧ (P / (A + 130) = 1 / 58)

theorem find_number_of_partners (P A : ℕ) (h : law_firm_partners P A) : P = 5 :=
  sorry

end find_number_of_partners_l14_14441


namespace ping_pong_balls_sold_l14_14709

theorem ping_pong_balls_sold (total_baseballs initial_baseballs initial_pingpong total_baseballs_sold total_balls_left : ℕ)
  (h1 : total_baseballs = 2754)
  (h2 : initial_pingpong = 1938)
  (h3 : total_baseballs_sold = 1095)
  (h4 : total_balls_left = 3021) :
  initial_pingpong - (total_balls_left - (total_baseballs - total_baseballs_sold)) = 576 :=
by sorry

end ping_pong_balls_sold_l14_14709
