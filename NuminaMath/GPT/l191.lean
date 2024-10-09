import Mathlib

namespace value_of_expression_l191_19105

-- Define the variables and conditions
variables (x y : ℝ)
axiom h1 : x + 2 * y = 4
axiom h2 : x * y = -8

-- Define the statement to be proven
theorem value_of_expression : x^2 + 4 * y^2 = 48 := 
by
  sorry

end value_of_expression_l191_19105


namespace evaluate_complex_modulus_l191_19101

namespace ComplexProblem

open Complex

theorem evaluate_complex_modulus : 
  abs ((1 / 2 : ℂ) - (3 / 8) * Complex.I) = 5 / 8 :=
by
  sorry

end ComplexProblem

end evaluate_complex_modulus_l191_19101


namespace fraction_evaluation_l191_19161

theorem fraction_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/5) = 5/8 := by
  sorry

end fraction_evaluation_l191_19161


namespace fraction_identity_l191_19116

theorem fraction_identity (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 := 
by 
  sorry

end fraction_identity_l191_19116


namespace gcd_360_150_l191_19162

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcd_360_150_l191_19162


namespace aziz_age_l191_19124

-- Definitions of the conditions
def year_moved : ℕ := 1982
def years_before_birth : ℕ := 3
def current_year : ℕ := 2021

-- Prove the main statement
theorem aziz_age : current_year - (year_moved + years_before_birth) = 36 :=
by
  sorry

end aziz_age_l191_19124


namespace average_of_second_pair_l191_19112

theorem average_of_second_pair (S : ℝ) (S1 : ℝ) (S3 : ℝ) (S2 : ℝ) (avg : ℝ) :
  (S / 6 = 3.95) →
  (S1 / 2 = 3.8) →
  (S3 / 2 = 4.200000000000001) →
  (S = S1 + S2 + S3) →
  (avg = S2 / 2) →
  avg = 3.85 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end average_of_second_pair_l191_19112


namespace fraction_to_terminating_decimal_l191_19178

theorem fraction_to_terminating_decimal :
  (45 / (2^2 * 5^3) : ℚ) = 0.09 :=
by sorry

end fraction_to_terminating_decimal_l191_19178


namespace sally_earnings_in_dozens_l191_19113

theorem sally_earnings_in_dozens (earnings_per_house : ℕ) (houses_cleaned : ℕ) (dozens_of_dollars : ℕ) : 
  earnings_per_house = 25 ∧ houses_cleaned = 96 → dozens_of_dollars = 200 := 
by
  intros h
  sorry

end sally_earnings_in_dozens_l191_19113


namespace similar_right_triangles_l191_19164

theorem similar_right_triangles (x c : ℕ) 
  (h1 : 12 * 6 = 9 * x) 
  (h2 : c^2 = x^2 + 6^2) :
  x = 8 ∧ c = 10 :=
by
  sorry

end similar_right_triangles_l191_19164


namespace operation_result_l191_19150

-- Define x and the operations
def x : ℕ := 40

-- Define the operation sequence
def operation (y : ℕ) : ℕ :=
  let step1 := y / 4
  let step2 := step1 * 5
  let step3 := step2 + 10
  let step4 := step3 - 12
  step4

-- The statement we need to prove
theorem operation_result : operation x = 48 := by
  sorry

end operation_result_l191_19150


namespace arriving_late_l191_19183

-- Definitions from conditions
def usual_time : ℕ := 24
def slower_factor : ℚ := 3 / 4

-- Derived from conditions
def slower_time : ℚ := usual_time * (4 / 3)

-- To be proven
theorem arriving_late : slower_time - usual_time = 8 := by
  sorry

end arriving_late_l191_19183


namespace largest_w_l191_19134

variable {x y z w : ℝ}

def x_value (x y z w : ℝ) := 
  x + 3 = y - 1 ∧ x + 3 = z + 5 ∧ x + 3 = w - 4

theorem largest_w (h : x_value x y z w) : 
  max x (max y (max z w)) = w := 
sorry

end largest_w_l191_19134


namespace combined_salaries_l191_19140
-- Import the required libraries

-- Define the salaries and conditions
def salary_c := 14000
def avg_salary_five := 8600
def num_individuals := 5
def total_salary := avg_salary_five * num_individuals

-- Define what we need to prove
theorem combined_salaries : total_salary - salary_c = 29000 :=
by
  -- The theorem statement
  sorry

end combined_salaries_l191_19140


namespace arithmetic_sequence_difference_l191_19198

noncomputable def arithmetic_difference (d: ℚ) (b₁: ℚ) : Prop :=
  (50 * b₁ + ((50 * 49) / 2) * d = 150) ∧
  (50 * (b₁ + 50 * d) + ((50 * 149) / 2) * d = 250)

theorem arithmetic_sequence_difference {d b₁ : ℚ} (h : arithmetic_difference d b₁) :
  (b₁ + d) - b₁ = (200 / 1295) :=
by
  sorry

end arithmetic_sequence_difference_l191_19198


namespace lcm_Anthony_Bethany_Casey_Dana_l191_19149

theorem lcm_Anthony_Bethany_Casey_Dana : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 10) = 120 := 
by
  sorry

end lcm_Anthony_Bethany_Casey_Dana_l191_19149


namespace integer_roots_of_quadratic_l191_19192

theorem integer_roots_of_quadratic (b : ℤ) :
  (∃ x : ℤ, x^2 + 4 * x + b = 0) ↔ b = -12 ∨ b = -5 ∨ b = 3 ∨ b = 4 :=
sorry

end integer_roots_of_quadratic_l191_19192


namespace percentage_loss_l191_19195

variable (CP SP : ℝ) (Loss : ℝ := CP - SP) (Percentage_of_Loss : ℝ := (Loss / CP) * 100)

theorem percentage_loss (h1: CP = 1600) (h2: SP = 1440) : Percentage_of_Loss = 10 := by
  sorry

end percentage_loss_l191_19195


namespace find_x_value_l191_19121

theorem find_x_value (x : ℚ) (h1 : 9 * x ^ 2 + 8 * x - 1 = 0) (h2 : 27 * x ^ 2 + 65 * x - 8 = 0) : x = 1 / 9 :=
sorry

end find_x_value_l191_19121


namespace solve_for_m_l191_19125

theorem solve_for_m (m x : ℝ) (h1 : 3 * m - 2 * x = 6) (h2 : x = 3) : m = 4 := by
  sorry

end solve_for_m_l191_19125


namespace joyce_gave_apples_l191_19139

theorem joyce_gave_apples : 
  ∀ (initial_apples final_apples given_apples : ℕ), (initial_apples = 75) ∧ (final_apples = 23) → (given_apples = initial_apples - final_apples) → (given_apples = 52) :=
by
  intros
  sorry

end joyce_gave_apples_l191_19139


namespace complement_event_l191_19146

-- Definitions based on conditions
variables (shoot1 shoot2 : Prop) -- shoots the target on the first and second attempt

-- Definition based on the question and answer
def hits_at_least_once : Prop := shoot1 ∨ shoot2
def misses_both_times : Prop := ¬shoot1 ∧ ¬shoot2

-- Theorem statement based on the mathematical translation
theorem complement_event :
  misses_both_times shoot1 shoot2 = ¬hits_at_least_once shoot1 shoot2 :=
by sorry

end complement_event_l191_19146


namespace denominator_exceeds_numerator_by_263_l191_19160

def G : ℚ := 736 / 999

theorem denominator_exceeds_numerator_by_263 : 999 - 736 = 263 := by
  -- Since 736 / 999 is the simplest form already, we simply state the obvious difference
  rfl

end denominator_exceeds_numerator_by_263_l191_19160


namespace find_max_min_find_angle_C_l191_19106

open Real

noncomputable def f (x : ℝ) : ℝ :=
  12 * sin (x + π / 6) * cos x - 3

theorem find_max_min (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 4) :
  let fx := f x 
  (∀ a, a = abs (fx - 6)) -> (∀ b, b = abs (fx - 3)) -> fx = 6 ∨ fx = 3 := sorry

theorem find_angle_C (AC BC CD : ℝ) (hAC : AC = 6) (hBC : BC = 3) (hCD : CD = 2 * sqrt 2) :
  ∃ C : ℝ, C = π / 2 := sorry

end find_max_min_find_angle_C_l191_19106


namespace hyperbola_sufficient_but_not_necessary_asymptote_l191_19197

-- Define the equation of the hyperbola and the related asymptotes
def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptote_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

-- Stating the theorem that expresses the sufficiency but not necessity
theorem hyperbola_sufficient_but_not_necessary_asymptote (a b : ℝ) :
  (∃ x y, hyperbola_eq a b x y) → (∀ x y, asymptote_eq a b x y) ∧ ¬ (∀ x y, (asymptote_eq a b x y) → (hyperbola_eq a b x y)) := 
sorry

end hyperbola_sufficient_but_not_necessary_asymptote_l191_19197


namespace peanuts_in_box_after_addition_l191_19163

theorem peanuts_in_box_after_addition : 4 + 12 = 16 := by
  sorry

end peanuts_in_box_after_addition_l191_19163


namespace find_M_for_same_asymptotes_l191_19120

theorem find_M_for_same_asymptotes :
  ∃ M : ℝ, ∀ x y : ℝ,
    (x^2 / 16 - y^2 / 25 = 1) →
    (y^2 / 50 - x^2 / M = 1) →
    (∀ x : ℝ, ∃ k : ℝ, y = k * x ↔ k = 5 / 4) →
    M = 32 :=
by
  sorry

end find_M_for_same_asymptotes_l191_19120


namespace wheat_bread_served_l191_19174

noncomputable def total_bread_served : ℝ := 0.6
noncomputable def white_bread_served : ℝ := 0.4

theorem wheat_bread_served : total_bread_served - white_bread_served = 0.2 :=
by
  sorry

end wheat_bread_served_l191_19174


namespace geometric_sequence_sum_l191_19196

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q a2005 a2006 : ℝ), 
    (∀ n, a (n + 1) = a n * q) ∧
    q > 1 ∧
    a2005 + a2006 = 2 ∧ 
    a2005 * a2006 = 3 / 4 ∧ 
    a (2005) = a2005 ∧ 
    a (2006) = a2006 → 
    a (2007) + a (2008) = 18 := 
by
  sorry

end geometric_sequence_sum_l191_19196


namespace measure_of_angle_F_l191_19142

theorem measure_of_angle_F (D E F : ℝ) (h₁ : D = 85) (h₂ : E = 4 * F + 15) (h₃ : D + E + F = 180) : 
  F = 16 :=
by
  sorry

end measure_of_angle_F_l191_19142


namespace polynomial_value_at_neg3_l191_19185

def polynomial (a b c x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 7

theorem polynomial_value_at_neg3 (a b c : ℝ) (h : polynomial a b c 3 = 65) :
  polynomial a b c (-3) = -79 := 
sorry

end polynomial_value_at_neg3_l191_19185


namespace most_accurate_method_is_independence_test_l191_19123

-- Definitions and assumptions
inductive Methods
| contingency_table
| independence_test
| stacked_bar_chart
| others

def related_or_independent_method : Methods := Methods.independence_test

-- Proof statement
theorem most_accurate_method_is_independence_test :
  related_or_independent_method = Methods.independence_test :=
sorry

end most_accurate_method_is_independence_test_l191_19123


namespace circles_intersect_l191_19115

-- Definition of the first circle
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

-- Definition of the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 8 = 0

-- Proving that the circles defined by C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
by sorry

end circles_intersect_l191_19115


namespace sam_gave_plums_l191_19154

variable (initial_plums : ℝ) (total_plums : ℝ) (plums_given : ℝ)

theorem sam_gave_plums (h1 : initial_plums = 7.0) (h2 : total_plums = 10.0) (h3 : total_plums = initial_plums + plums_given) :
  plums_given = 3 := 
by
  sorry

end sam_gave_plums_l191_19154


namespace abhay_speed_l191_19152

-- Definitions of the problem's conditions
def condition1 (A S : ℝ) : Prop := 42 / A = 42 / S + 2
def condition2 (A S : ℝ) : Prop := 42 / (2 * A) = 42 / S - 1

-- Define Abhay and Sameer's speeds and declare the main theorem
theorem abhay_speed (A S : ℝ) (h1 : condition1 A S) (h2 : condition2 A S) : A = 10.5 :=
by
  sorry

end abhay_speed_l191_19152


namespace correct_option_d_l191_19156

-- Define the conditions as separate lemmas
lemma option_a_incorrect : ¬ (Real.sqrt 18 + Real.sqrt 2 = 2 * Real.sqrt 5) :=
sorry 

lemma option_b_incorrect : ¬ (Real.sqrt 18 - Real.sqrt 2 = 4) :=
sorry

lemma option_c_incorrect : ¬ (Real.sqrt 18 * Real.sqrt 2 = 36) :=
sorry

-- Define the statement to prove
theorem correct_option_d : Real.sqrt 18 / Real.sqrt 2 = 3 :=
by
  sorry

end correct_option_d_l191_19156


namespace sides_equal_max_diagonal_at_most_two_l191_19159

variable {n : ℕ}
variable (P : Polygon n)
variable (is_convex : P.IsConvex)
variable (max_diagonal : ℝ)
variable (sides_equal_max_diagonal : list ℝ)
variable (length_sides_equal_max_diagonal : sides_equal_max_diagonal.length)

-- Here we assume the basic conditions given in the problem:
-- 1. The polygon P is convex.
-- 2. The number of sides equal to the longest diagonal are stored in sides_equal_max_diagonal.

theorem sides_equal_max_diagonal_at_most_two :
  is_convex → length_sides_equal_max_diagonal ≤ 2 :=
by
  sorry

end sides_equal_max_diagonal_at_most_two_l191_19159


namespace min_omega_value_l191_19189

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega_value (ω : ℝ) (φ : ℝ) (h_ω_pos : ω > 0)
  (h_even : ∀ x : ℝ, f ω φ x = f ω φ (-x))
  (h_symmetry : f ω φ 1 = 0 ∧ ∀ x : ℝ, f ω φ (1 + x) = - f ω φ (1 - x)) :
  ω = Real.pi / 2 :=
by
  sorry

end min_omega_value_l191_19189


namespace Jake_width_proof_l191_19172

-- Define the dimensions of Sara's birdhouse in feet
def Sara_width_feet := 1
def Sara_height_feet := 2
def Sara_depth_feet := 2

-- Convert the dimensions to inches
def Sara_width_inch := Sara_width_feet * 12
def Sara_height_inch := Sara_height_feet * 12
def Sara_depth_inch := Sara_depth_feet * 12

-- Calculate Sara's birdhouse volume
def Sara_volume := Sara_width_inch * Sara_height_inch * Sara_depth_inch

-- Define the dimensions of Jake's birdhouse in inches
def Jake_height_inch := 20
def Jake_depth_inch := 18
def Jake_volume (Jake_width_inch : ℝ) := Jake_width_inch * Jake_height_inch * Jake_depth_inch

-- Difference in volume
def volume_difference := 1152

-- Prove the width of Jake's birdhouse
theorem Jake_width_proof : ∃ (W : ℝ), Jake_volume W - Sara_volume = volume_difference ∧ W = 22.4 := by
  sorry

end Jake_width_proof_l191_19172


namespace find_first_number_l191_19186

theorem find_first_number
  (avg1 : (20 + 40 + 60) / 3 = 40)
  (avg2 : 40 - 4 = (x + 70 + 28) / 3)
  (sum_eq : x + 70 + 28 = 108) :
  x = 10 :=
by
  sorry

end find_first_number_l191_19186


namespace inequality_not_true_l191_19168

theorem inequality_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬ (a > 0) :=
sorry

end inequality_not_true_l191_19168


namespace initial_books_from_library_l191_19138

-- Definitions of the problem conditions
def booksGivenAway : ℝ := 23.0
def booksLeft : ℝ := 31.0

-- Statement of the problem, proving that the initial number of books
def initialBooks (x : ℝ) : Prop :=
  x = booksGivenAway + booksLeft

-- Main theorem
theorem initial_books_from_library : initialBooks 54.0 :=
by
  -- Proof pending
  sorry

end initial_books_from_library_l191_19138


namespace smallest_n_divisible_l191_19166

theorem smallest_n_divisible (n : ℕ) : 
  (450 ∣ n^3) ∧ (2560 ∣ n^4) ↔ n = 60 :=
by {
  sorry
}

end smallest_n_divisible_l191_19166


namespace invertible_my_matrix_l191_19180

def my_matrix : Matrix (Fin 2) (Fin 2) ℚ := ![![4, 5], ![-2, 9]]

noncomputable def inverse_of_my_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.det my_matrix • Matrix.adjugate my_matrix

theorem invertible_my_matrix :
  inverse_of_my_matrix = (1 / 46 : ℚ) • ![![9, -5], ![2, 4]] :=
by
  sorry

end invertible_my_matrix_l191_19180


namespace coffee_shop_ratio_l191_19119

theorem coffee_shop_ratio (morning_usage afternoon_multiplier weekly_usage days_per_week : ℕ) (r : ℕ) 
  (h_morning : morning_usage = 3)
  (h_afternoon : afternoon_multiplier = 3)
  (h_weekly : weekly_usage = 126)
  (h_days : days_per_week = 7):
  weekly_usage = days_per_week * (morning_usage + afternoon_multiplier * morning_usage + r * morning_usage) →
  r = 2 :=
by
  intros h_eq
  sorry

end coffee_shop_ratio_l191_19119


namespace remainder_when_divided_l191_19104

theorem remainder_when_divided (P D Q R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D * D') = R + R' * D :=
by
  sorry

end remainder_when_divided_l191_19104


namespace percent_game_of_thrones_altered_l191_19187

def votes_game_of_thrones : ℕ := 10
def votes_twilight : ℕ := 12
def votes_art_of_deal : ℕ := 20

def altered_votes_art_of_deal : ℕ := votes_art_of_deal - (votes_art_of_deal * 80 / 100)
def altered_votes_twilight : ℕ := votes_twilight / 2
def total_altered_votes : ℕ := altered_votes_art_of_deal + altered_votes_twilight + votes_game_of_thrones

theorem percent_game_of_thrones_altered :
  ((votes_game_of_thrones * 100) / total_altered_votes) = 50 := by
  sorry

end percent_game_of_thrones_altered_l191_19187


namespace probability_hits_10_ring_l191_19109

-- Definitions based on conditions
def total_shots : ℕ := 10
def hits_10_ring : ℕ := 2

-- Theorem stating the question and answer equivalence.
theorem probability_hits_10_ring : (hits_10_ring : ℚ) / total_shots = 0.2 := by
  -- We are skipping the proof with 'sorry'
  sorry

end probability_hits_10_ring_l191_19109


namespace probability_x_lt_2y_l191_19145

noncomputable def rectangle := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 }

noncomputable def region_of_interest := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 < 2 * p.2 }

noncomputable def area_rectangle := 6 * 2

noncomputable def area_trapezoid := (1 / 2) * (4 + 6) * 2

theorem probability_x_lt_2y : (area_trapezoid / area_rectangle) = 5 / 6 :=
by
  -- skip the proof
  sorry

end probability_x_lt_2y_l191_19145


namespace cow_spots_total_l191_19148

theorem cow_spots_total
  (left_spots : ℕ) (right_spots : ℕ)
  (left_spots_eq : left_spots = 16)
  (right_spots_eq : right_spots = 3 * left_spots + 7) :
  left_spots + right_spots = 71 :=
by
  sorry

end cow_spots_total_l191_19148


namespace trigonometric_identity_x1_trigonometric_identity_x2_l191_19102

noncomputable def x1 (n : ℤ) : ℝ := (2 * n + 1) * (Real.pi / 4)
noncomputable def x2 (k : ℤ) : ℝ := ((-1)^(k + 1)) * (Real.pi / 8) + k * (Real.pi / 2)

theorem trigonometric_identity_x1 (n : ℤ) : 
  (Real.cos (4 * x1 n) * Real.cos (Real.pi + 2 * x1 n) - 
   Real.sin (2 * x1 n) * Real.cos (Real.pi / 2 - 4 * x1 n)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x1 n) := 
by
  sorry

theorem trigonometric_identity_x2 (k : ℤ) : 
  (Real.cos (4 * x2 k) * Real.cos (Real.pi + 2 * x2 k) - 
   Real.sin (2 * x2 k) * Real.cos (Real.pi / 2 - 4 * x2 k)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x2 k) := 
by
  sorry

end trigonometric_identity_x1_trigonometric_identity_x2_l191_19102


namespace geometric_difference_l191_19131

def is_geometric_sequence (n : ℕ) : Prop :=
∃ (a b c : ℤ), n = a * 100 + b * 10 + c ∧
a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
(b^2 = a * c) ∧
(b % 2 = 1)

theorem geometric_difference :
  ∃ (n1 n2 : ℕ), is_geometric_sequence n1 ∧ is_geometric_sequence n2 ∧
  n2 > n1 ∧
  n2 - n1 = 220 :=
sorry

end geometric_difference_l191_19131


namespace max_non_real_roots_l191_19184

theorem max_non_real_roots (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (A B : ℕ → ℕ) (h_turns : ∀ i < 3 * n, A i + B i = 1),
    (∀ i, (A i + B (i + 1)) % 3 = 0) →
    ∃ k, ∀ m, ∃ j < n, j % 2 = 1 → j + m * 2 ≤ 2 * k + j - m)
  → (∃ k, k = (n + 1) / 2) :=
sorry

end max_non_real_roots_l191_19184


namespace dot_product_computation_l191_19170

open Real

variables (a b : ℝ) (θ : ℝ)

noncomputable def dot_product (u v : ℝ) : ℝ :=
  u * v * cos θ

noncomputable def magnitude (v : ℝ) : ℝ :=
  abs v

theorem dot_product_computation (a b : ℝ) (h1 : θ = 120) (h2 : magnitude a = 4) (h3 : magnitude b = 4) :
  dot_product b (3 * a + b) = -8 :=
by
  sorry

end dot_product_computation_l191_19170


namespace cube_edge_length_l191_19135

-- Define the edge length 'a'
variable (a : ℝ)

-- Given conditions: 6a^2 = 24
theorem cube_edge_length (h : 6 * a^2 = 24) : a = 2 :=
by {
  -- The actual proof would go here, but we use sorry to skip it as per instructions.
  sorry
}

end cube_edge_length_l191_19135


namespace c_plus_d_l191_19132

theorem c_plus_d (a b c d : ℝ) (h1 : a + b = 11) (h2 : b + c = 9) (h3 : a + d = 5) :
  c + d = 3 + b :=
by
  sorry

end c_plus_d_l191_19132


namespace tan_150_eq_neg_one_over_sqrt_three_l191_19127

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l191_19127


namespace negation_universal_statement_l191_19130

theorem negation_universal_statement :
  (¬ (∀ x : ℝ, |x| ≥ 0)) ↔ (∃ x : ℝ, |x| < 0) :=
by sorry

end negation_universal_statement_l191_19130


namespace x_intercept_perpendicular_line_l191_19129

theorem x_intercept_perpendicular_line 
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 12)
  (h2 : y = - (3 / 4) * x + 4)
  : x = 16 / 3 := 
sorry

end x_intercept_perpendicular_line_l191_19129


namespace linear_function_above_x_axis_l191_19175

theorem linear_function_above_x_axis (a : ℝ) :
  (-1 < a ∧ a < 2 ∧ a ≠ 0) ↔
  (∀ x, -2 ≤ x ∧ x ≤ 1 → ax + a + 2 > 0) :=
sorry

end linear_function_above_x_axis_l191_19175


namespace lucas_age_correct_l191_19110

variable (Noah_age : ℕ) (Mia_age : ℕ) (Lucas_age : ℕ)

-- Conditions
axiom h1 : Noah_age = 12
axiom h2 : Mia_age = Noah_age + 5
axiom h3 : Lucas_age = Mia_age - 6

-- Goal
theorem lucas_age_correct : Lucas_age = 11 := by
  sorry

end lucas_age_correct_l191_19110


namespace warriors_games_won_l191_19108

open Set

-- Define the variables for the number of games each team won
variables (games_L games_H games_W games_F games_R : ℕ)

-- Define the set of possible game scores
def game_scores : Set ℕ := {19, 23, 28, 32, 36}

-- Define the conditions as assumptions
axiom h1 : games_L > games_H
axiom h2 : games_W > games_F
axiom h3 : games_W < games_R
axiom h4 : games_F > 18
axiom h5 : ∃ min_games ∈ game_scores, min_games > games_H ∧ min_games < 20

-- Prove the main statement
theorem warriors_games_won : games_W = 32 :=
sorry

end warriors_games_won_l191_19108


namespace sum_of_a_b_c_l191_19171

theorem sum_of_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc1 : a * b + c = 47) (habc2 : b * c + a = 47) (habc3 : a * c + b = 47) : a + b + c = 48 := 
sorry

end sum_of_a_b_c_l191_19171


namespace inequality_proof_l191_19137

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 = 1000 / 9 :=
sorry

end inequality_proof_l191_19137


namespace combined_salaries_l191_19151

variable {A B C E : ℝ}
variable (D : ℝ := 7000)
variable (average_salary : ℝ := 8400)
variable (n : ℕ := 5)

theorem combined_salaries (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ E) 
  (h4 : B ≠ C) (h5 : B ≠ E) (h6 : C ≠ E)
  (h7 : average_salary = (A + B + C + D + E) / n) :
  A + B + C + E = 35000 :=
by
  sorry

end combined_salaries_l191_19151


namespace smallest_sphere_radius_l191_19143

noncomputable def radius_smallest_sphere : ℝ := 2 * Real.sqrt 3 + 2

theorem smallest_sphere_radius (r : ℝ) (h : r = 2) : radius_smallest_sphere = 2 * Real.sqrt 3 + 2 := by
  sorry

end smallest_sphere_radius_l191_19143


namespace determine_a_if_fx_odd_l191_19114

theorem determine_a_if_fx_odd (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = 2^x + a * 2^(-x)) (h2 : ∀ x, f (-x) = -f x) : a = -1 :=
by
  sorry

end determine_a_if_fx_odd_l191_19114


namespace bianca_marathon_total_miles_l191_19158

theorem bianca_marathon_total_miles : 8 + 4 = 12 :=
by
  sorry

end bianca_marathon_total_miles_l191_19158


namespace triangle_has_at_most_one_obtuse_angle_l191_19117

-- Definitions
def Triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def Obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

def Two_obtuse_angles (α β γ : ℝ) : Prop :=
  Obtuse_angle α ∧ Obtuse_angle β

-- Theorem Statement
theorem triangle_has_at_most_one_obtuse_angle (α β γ : ℝ) (h_triangle : Triangle α β γ) :
  ¬ Two_obtuse_angles α β γ := 
sorry

end triangle_has_at_most_one_obtuse_angle_l191_19117


namespace find_target_number_l191_19100

theorem find_target_number : ∃ n ≥ 0, (∀ k < 5, ∃ m, 0 ≤ m ∧ m ≤ n ∧ m % 11 = 3 ∧ m = 3 + k * 11) ∧ n = 47 :=
by
  sorry

end find_target_number_l191_19100


namespace sum_of_three_pairwise_rel_prime_integers_l191_19118

theorem sum_of_three_pairwise_rel_prime_integers (a b c : ℕ)
  (h1: 1 < a) (h2: 1 < b) (h3: 1 < c)
  (prod: a * b * c = 216000)
  (rel_prime_ab : Nat.gcd a b = 1)
  (rel_prime_ac : Nat.gcd a c = 1)
  (rel_prime_bc : Nat.gcd b c = 1) : 
  a + b + c = 184 := 
sorry

end sum_of_three_pairwise_rel_prime_integers_l191_19118


namespace solution_f_derivative_l191_19194

noncomputable def f (x : ℝ) := Real.sqrt x

theorem solution_f_derivative :
  (deriv f 1) = 1 / 2 :=
by
  -- This is where the proof would go, but for now, we just state sorry.
  sorry

end solution_f_derivative_l191_19194


namespace number_of_real_solutions_l191_19167

theorem number_of_real_solutions (floor : ℝ → ℤ) 
  (h_floor : ∀ x, floor x = ⌊x⌋)
  (h_eq : ∀ x, 9 * x^2 - 45 * floor (x^2 - 1) + 94 = 0) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_real_solutions_l191_19167


namespace problem_statement_l191_19169

theorem problem_statement (x y : ℕ) (h1 : x = 3) (h2 :y = 5) :
  (x^5 + 2*y^2 - 15) / 7 = 39 + 5 / 7 := 
by 
  sorry

end problem_statement_l191_19169


namespace isosceles_triangles_possible_l191_19199

theorem isosceles_triangles_possible :
  ∃ (sticks : List ℕ), (sticks = [1, 1, 2, 2, 3, 3] ∧ 
    ∀ (a b c : ℕ), a ∈ sticks → b ∈ sticks → c ∈ sticks → 
    ((a + b > c ∧ b + c > a ∧ c + a > b) → a = b ∨ b = c ∨ c = a)) :=
sorry

end isosceles_triangles_possible_l191_19199


namespace question1_question2_l191_19181

variable (a : ℤ)
def point_P : (ℤ × ℤ) := (2*a - 2, a + 5)

-- Part 1: If point P lies on the x-axis, its coordinates are (-12, 0).
theorem question1 (h1 : a + 5 = 0) : point_P a = (-12, 0) :=
sorry

-- Part 2: If point P lies in the second quadrant and the distance from point P to the x-axis is equal to the distance from point P to the y-axis,
-- the value of a^2023 + 2023 is 2022.
theorem question2 (h2 : 2*a - 2 < 0) (h3 : -(2*a - 2) = a + 5) : a ^ 2023 + 2023 = 2022 :=
sorry

end question1_question2_l191_19181


namespace geometric_sequence_third_term_l191_19176

theorem geometric_sequence_third_term (q : ℝ) (b1 : ℝ) (h1 : abs q < 1)
    (h2 : b1 / (1 - q) = 8 / 5) (h3 : b1 * q = -1 / 2) :
    b1 * q^2 / 2 = 1 / 8 := by
  sorry

end geometric_sequence_third_term_l191_19176


namespace correct_calculation_l191_19133

theorem correct_calculation (x : ℤ) (h : x + 54 = 78) : x + 45 = 69 :=
by
  sorry

end correct_calculation_l191_19133


namespace no_possible_values_for_n_l191_19122

theorem no_possible_values_for_n (n a : ℤ) (h : n > 1) (d : ℤ := 3) (Sn : ℤ := 180) :
  ∃ n > 1, ∃ k : ℤ, a = k^2 ∧ Sn = n / 2 * (2 * a + (n - 1) * d) :=
sorry

end no_possible_values_for_n_l191_19122


namespace num_women_in_luxury_suite_l191_19153

theorem num_women_in_luxury_suite (total_passengers : ℕ) (pct_women : ℕ) (pct_women_luxury : ℕ)
  (h_total_passengers : total_passengers = 300)
  (h_pct_women : pct_women = 50)
  (h_pct_women_luxury : pct_women_luxury = 15) :
  (total_passengers * pct_women / 100) * pct_women_luxury / 100 = 23 := 
by
  sorry

end num_women_in_luxury_suite_l191_19153


namespace relationship_roots_geometric_progression_l191_19182

theorem relationship_roots_geometric_progression 
  (x y z p q r : ℝ)
  (h1 : x^2 ≠ y^2 ∧ y^2 ≠ z^2 ∧ x^2 ≠ z^2) -- Distinct non-zero numbers
  (h2 : y^2 = x^2 * r)
  (h3 : z^2 = y^2 * r)
  (h4 : x + y + z = p)
  (h5 : x * y + y * z + z * x = q)
  (h6 : x * y * z = r) : r^2 = 1 := sorry

end relationship_roots_geometric_progression_l191_19182


namespace value_of_expression_l191_19126

variable {a : ℕ → ℤ}
variable {a₁ a₄ a₁₀ a₁₆ a₁₉ : ℤ}
variable {d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + d * n

-- Given conditions
axiom h₀ : arithmetic_sequence a a₁ d
axiom h₁ : a₁ + a₄ + a₁₀ + a₁₆ + a₁₉ = 150

-- Prove the required statement
theorem value_of_expression :
  a 20 - a 26 + a 16 = 30 :=
sorry

end value_of_expression_l191_19126


namespace conical_pile_volume_l191_19141

noncomputable def volume_of_cone (d : ℝ) (h : ℝ) : ℝ :=
  (Real.pi * (d / 2) ^ 2 * h) / 3

theorem conical_pile_volume :
  let diameter := 10
  let height := 0.60 * diameter
  volume_of_cone diameter height = 50 * Real.pi :=
by
  sorry

end conical_pile_volume_l191_19141


namespace sticks_needed_for_4x4_square_largest_square_with_100_sticks_l191_19193

-- Problem a)
def sticks_needed_for_square (n: ℕ) : ℕ := 2 * n * (n + 1)

theorem sticks_needed_for_4x4_square : sticks_needed_for_square 4 = 40 :=
by
  sorry

-- Problem b)
def max_square_side_length (total_sticks : ℕ) : ℕ × ℕ :=
  let n := Nat.sqrt (total_sticks / 2)
  if 2*n*(n+1) <= total_sticks then (n, total_sticks - 2*n*(n+1)) else (n-1, total_sticks - 2*(n-1)*n)

theorem largest_square_with_100_sticks : max_square_side_length 100 = (6, 16) :=
by
  sorry

end sticks_needed_for_4x4_square_largest_square_with_100_sticks_l191_19193


namespace left_handed_and_like_scifi_count_l191_19191

-- Definitions based on the problem conditions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def like_scifi_members : ℕ := 18
def right_handed_not_like_scifi : ℕ := 4

-- Main proof statement
theorem left_handed_and_like_scifi_count :
  ∃ x : ℕ, (left_handed_members - x) + (like_scifi_members - x) + x + right_handed_not_like_scifi = total_members ∧ x = 4 :=
by
  use 4
  sorry

end left_handed_and_like_scifi_count_l191_19191


namespace cost_of_five_trip_ticket_l191_19128

-- Variables for the costs of the tickets
variables (x y z : ℕ)

-- Conditions from the problem
def condition1 : Prop := 5 * x > y
def condition2 : Prop := 4 * y > z
def condition3 : Prop := z + 3 * y = 33
def condition4 : Prop := 20 + 3 * 5 = 35

-- The theorem to prove
theorem cost_of_five_trip_ticket (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z y) (h4 : condition4) : y = 5 := 
by
  sorry

end cost_of_five_trip_ticket_l191_19128


namespace area_bounded_by_parabola_and_x_axis_l191_19157

/-- Define the parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x - x^2

/-- The function for the x-axis -/
def x_axis : ℝ := 0

/-- Prove that the area bounded by the parabola and x-axis between x = 0 and x = 2 is 4/3 -/
theorem area_bounded_by_parabola_and_x_axis : 
  (∫ x in (0 : ℝ)..(2 : ℝ), parabola x) = 4 / 3 := by
    sorry

end area_bounded_by_parabola_and_x_axis_l191_19157


namespace amy_tickets_initial_l191_19147

theorem amy_tickets_initial (x : ℕ) (h1 : x + 21 = 54) : x = 33 :=
by sorry

end amy_tickets_initial_l191_19147


namespace lowest_possible_price_l191_19136

theorem lowest_possible_price
  (MSRP : ℝ)
  (D1 : ℝ)
  (D2 : ℝ)
  (P_final : ℝ)
  (h1 : MSRP = 45.00)
  (h2 : 0.10 ≤ D1 ∧ D1 ≤ 0.30)
  (h3 : D2 = 0.20) :
  P_final = 25.20 :=
by
  sorry

end lowest_possible_price_l191_19136


namespace solve_for_y_l191_19190

theorem solve_for_y (y : ℝ) : (10 - y) ^ 2 = 4 * y ^ 2 → y = 10 / 3 ∨ y = -10 :=
by
  intro h
  -- The proof steps would go here, but we include sorry to allow for compilation.
  sorry

end solve_for_y_l191_19190


namespace joan_apples_l191_19144

theorem joan_apples (initial_apples : ℕ) (given_to_melanie : ℕ) (given_to_sarah : ℕ) : 
  initial_apples = 43 ∧ given_to_melanie = 27 ∧ given_to_sarah = 11 → (initial_apples - given_to_melanie - given_to_sarah) = 5 := 
by
  sorry

end joan_apples_l191_19144


namespace f_1984_can_be_any_real_l191_19103

noncomputable def f : ℤ → ℝ := sorry

axiom f_condition : ∀ (x y : ℤ), f (x - y^2) = f x + (y^2 - 2 * x) * f y

theorem f_1984_can_be_any_real
    (a : ℝ)
    (h : f 1 = a) : f 1984 = 1984^2 * a := sorry

end f_1984_can_be_any_real_l191_19103


namespace sum_distances_saham_and_mother_l191_19173

theorem sum_distances_saham_and_mother :
  let saham_distance := 2.6
  let mother_distance := 5.98
  saham_distance + mother_distance = 8.58 :=
by
  sorry

end sum_distances_saham_and_mother_l191_19173


namespace knights_rearrangement_impossible_l191_19188

theorem knights_rearrangement_impossible :
  ∀ (b : ℕ → ℕ → Prop), (b 0 0 = true) ∧ (b 0 2 = true) ∧ (b 2 0 = true) ∧ (b 2 2 = true) ∧
  (b 0 0 = b 0 2) ∧ (b 2 0 ≠ b 2 2) → ¬(∃ (b' : ℕ → ℕ → Prop), 
  (b' 0 0 ≠ b 0 0) ∧ (b' 0 2 ≠ b 0 2) ∧ (b' 2 0 ≠ b 2 0) ∧ (b' 2 2 ≠ b 2 2) ∧ 
  (b' 0 0 ≠ b' 0 2) ∧ (b' 2 0 ≠ b' 2 2)) :=
by { sorry }

end knights_rearrangement_impossible_l191_19188


namespace simplified_expression_l191_19165

theorem simplified_expression :
  (0.2 * 0.4 - 0.3 / 0.5) + (0.6 * 0.8 + 0.1 / 0.2) - 0.9 * (0.3 - 0.2 * 0.4) = 0.262 :=
by
  sorry

end simplified_expression_l191_19165


namespace isosceles_largest_angle_eq_60_l191_19107

theorem isosceles_largest_angle_eq_60 :
  ∀ (A B C : ℝ), (
    -- Condition: A triangle is isosceles with two equal angles of 60 degrees.
    ∀ (x y : ℝ), A = x ∧ B = x ∧ C = y ∧ x = 60 →
    -- Prove that
    max A (max B C) = 60 ) :=
by
  intros A B C h
  -- Sorry denotes skipping the proof.
  sorry

end isosceles_largest_angle_eq_60_l191_19107


namespace possible_values_of_a_l191_19111

theorem possible_values_of_a (a b c : ℝ) (h1 : a + b + c = 2005) (h2 : (a - 1 = a ∨ a - 1 = b ∨ a - 1 = c) ∧ (b + 1 = a ∨ b + 1 = b ∨ b + 1 = c) ∧ (c ^ 2 = a ∨ c ^ 2 = b ∨ c ^ 2 = c)) :
  a = 1003 ∨ a = 1002.5 :=
sorry

end possible_values_of_a_l191_19111


namespace octagon_area_in_square_l191_19177

/--
An octagon is inscribed in a square such that each vertex of the octagon cuts off a corner
triangle from the square. Each triangle has legs equal to one-fourth of the square's side.
If the perimeter of the square is 160 centimeters, what is the area of the octagon?
-/
theorem octagon_area_in_square
  (side_of_square : ℝ)
  (h1 : 4 * (side_of_square / 4) = side_of_square)
  (h2 : 8 * (side_of_square / 4) = side_of_square)
  (perimeter_of_square : ℝ)
  (h3 : perimeter_of_square = 160)
  (area_of_square : ℝ)
  (h4 : area_of_square = side_of_square^2)
  : ∃ (area_of_octagon : ℝ), area_of_octagon = 1400 := by
  sorry

end octagon_area_in_square_l191_19177


namespace probability_blue_is_4_over_13_l191_19179

def num_red : ℕ := 5
def num_green : ℕ := 6
def num_yellow : ℕ := 7
def num_blue : ℕ := 8
def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue

def probability_blue : ℚ := num_blue / total_jelly_beans

theorem probability_blue_is_4_over_13
  (h_num_red : num_red = 5)
  (h_num_green : num_green = 6)
  (h_num_yellow : num_yellow = 7)
  (h_num_blue : num_blue = 8) :
  probability_blue = 4 / 13 :=
by
  sorry

end probability_blue_is_4_over_13_l191_19179


namespace price_second_oil_per_litre_is_correct_l191_19155

-- Definitions based on conditions
def price_first_oil_per_litre := 54
def volume_first_oil := 10
def volume_second_oil := 5
def mixture_rate_per_litre := 58
def total_volume := volume_first_oil + volume_second_oil
def total_cost_mixture := total_volume * mixture_rate_per_litre
def total_cost_first_oil := volume_first_oil * price_first_oil_per_litre

-- The statement to prove
theorem price_second_oil_per_litre_is_correct (x : ℕ) (h : total_cost_first_oil + (volume_second_oil * x) = total_cost_mixture) : x = 66 :=
by
  sorry

end price_second_oil_per_litre_is_correct_l191_19155
