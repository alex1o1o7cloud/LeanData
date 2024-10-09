import Mathlib

namespace arithmetic_sequence_sum_abs_values_l1611_161172

theorem arithmetic_sequence_sum_abs_values (n : ℕ) (a : ℕ → ℤ)
  (h₁ : a 1 = 13)
  (h₂ : ∀ k, a (k + 1) = a k + (-4)) :
  T_n = if n ≤ 4 then 15 * n - 2 * n^2 else 2 * n^2 - 15 * n + 56 :=
by sorry

end arithmetic_sequence_sum_abs_values_l1611_161172


namespace total_voters_l1611_161194

-- Definitions
def number_of_voters_first_hour (x : ℕ) := x
def percentage_october_22 (x : ℕ) := 35 * x / 100
def percentage_october_29 (x : ℕ) := 65 * x / 100
def additional_voters_october_22 := 80
def final_percentage_october_29 (total_votes : ℕ) := 45 * total_votes / 100

-- Statement
theorem total_voters (x : ℕ) (h1 : percentage_october_22 x + additional_voters_october_22 = 35 * (x + additional_voters_october_22) / 100)
                      (h2 : percentage_october_29 x = 65 * x / 100)
                      (h3 : final_percentage_october_29 (x + additional_voters_october_22) = 45 * (x + additional_voters_october_22) / 100):
  x + additional_voters_october_22 = 260 := 
sorry

end total_voters_l1611_161194


namespace set_intersection_l1611_161155

-- Define set A
def A := {x : ℝ | x^2 - 4 * x < 0}

-- Define set B
def B := {x : ℤ | -2 < x ∧ x ≤ 2}

-- Define the intersection of A and B in ℝ
def A_inter_B := {x : ℝ | (x ∈ A) ∧ (∃ (z : ℤ), (x = z) ∧ (z ∈ B))}

-- Proof statement
theorem set_intersection : A_inter_B = {1, 2} :=
by sorry

end set_intersection_l1611_161155


namespace max_sum_of_squares_l1611_161100

theorem max_sum_of_squares :
  ∃ m n : ℕ, (m ∈ Finset.range 101) ∧ (n ∈ Finset.range 101) ∧ ((n^2 - n * m - m^2)^2 = 1) ∧ (m^2 + n^2 = 10946) :=
by
  sorry

end max_sum_of_squares_l1611_161100


namespace taylor_pets_count_l1611_161123

noncomputable def totalPetsTaylorFriends (T : ℕ) (x1 : ℕ) (x2 : ℕ) : ℕ :=
  T + 3 * x1 + 2 * x2

theorem taylor_pets_count (T : ℕ) (x1 x2 : ℕ) (h1 : x1 = 2 * T) (h2 : x2 = 2) (h3 : totalPetsTaylorFriends T x1 x2 = 32) :
  T = 4 :=
by
  sorry

end taylor_pets_count_l1611_161123


namespace paige_mp3_player_songs_l1611_161113

/--
Paige had 11 songs on her mp3 player.
She deleted 9 old songs.
She added 8 new songs.

We are to prove:
- The final number of songs on her mp3 player is 10.
-/
theorem paige_mp3_player_songs (initial_songs deleted_songs added_songs final_songs : ℕ)
  (h₁ : initial_songs = 11)
  (h₂ : deleted_songs = 9)
  (h₃ : added_songs = 8) :
  final_songs = initial_songs - deleted_songs + added_songs :=
by
  sorry

end paige_mp3_player_songs_l1611_161113


namespace sum_and_count_l1611_161114

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count (x y : ℕ) (hx : x = sum_of_integers 30 50) (hy : y = count_even_integers 30 50) :
  x + y = 851 :=
by
  -- proof goes here
  sorry

end sum_and_count_l1611_161114


namespace function_periodic_l1611_161193

open Real

def periodic (f : ℝ → ℝ) := ∃ T > 0, ∀ x, f (x + T) = f x

theorem function_periodic (a : ℚ) (b d c : ℝ) (f : ℝ → ℝ) 
    (hf : ∀ x : ℝ, f (x + ↑a + b) - f (x + b) = c * (x + 2 * ↑a + ⌊x⌋ - 2 * ⌊x + ↑a⌋ - ⌊b⌋) + d) : 
    periodic f :=
sorry

end function_periodic_l1611_161193


namespace max_m_value_inequality_abc_for_sum_l1611_161178

-- Define the mathematical conditions and the proof problem.

theorem max_m_value (x m : ℝ) (h1 : |x - 2| - |x + 3| ≥ |m + 1|) :
  m ≤ 4 :=
sorry

theorem inequality_abc_for_sum (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_eq_M : a + 2 * b + c = 4) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 1 :=
sorry

end max_m_value_inequality_abc_for_sum_l1611_161178


namespace marco_paint_fraction_l1611_161107

theorem marco_paint_fraction (W : ℝ) (M : ℝ) (minutes_paint : ℝ) (fraction_paint : ℝ) :
  M = 60 ∧ W = 1 ∧ minutes_paint = 12 ∧ fraction_paint = 1/5 → 
  (minutes_paint / M) * W = fraction_paint := 
by
  sorry

end marco_paint_fraction_l1611_161107


namespace problem_solution_l1611_161168

theorem problem_solution (a d e : ℕ) (ha : 0 < a ∧ a < 10) (hd : 0 < d ∧ d < 10) (he : 0 < e ∧ e < 10) :
  ((10 * a + d) * (10 * a + e) = 100 * a ^ 2 + 110 * a + d * e) ↔ (d + e = 11) := by
  sorry

end problem_solution_l1611_161168


namespace no_perfect_squares_l1611_161152

theorem no_perfect_squares (x y z t : ℕ) (h1 : xy - zt = k) (h2 : x + y = k) (h3 : z + t = k) :
  ¬ (∃ m n : ℕ, x * y = m^2 ∧ z * t = n^2) := by
  sorry

end no_perfect_squares_l1611_161152


namespace numbers_of_form_xy9z_div_by_132_l1611_161174

theorem numbers_of_form_xy9z_div_by_132 (x y z : ℕ) :
  let N := 1000 * x + 100 * y + 90 + z
  (N % 4 = 0) ∧ ((x + y + 9 + z) % 3 = 0) ∧ ((x + 9 - y - z) % 11 = 0) ↔ 
  (N = 3696) ∨ (N = 4092) ∨ (N = 6996) ∨ (N = 7392) :=
by
  intros
  let N := 1000 * x + 100 * y + 90 + z
  sorry

end numbers_of_form_xy9z_div_by_132_l1611_161174


namespace mean_equality_l1611_161164

theorem mean_equality (z : ℚ) :
  (8 + 12 + 24) / 3 = (16 + z) / 2 ↔ z = 40 / 3 :=
by
  sorry

end mean_equality_l1611_161164


namespace greatest_m_value_l1611_161102

theorem greatest_m_value (x y m : ℝ) 
  (h₁: x^2 + y^2 = 1)
  (h₂ : |x^3 - y^3| + |x - y| = m^3) : 
  m ≤ 2^(1/3) :=
sorry

end greatest_m_value_l1611_161102


namespace simplify_expression_l1611_161181

theorem simplify_expression :
  (4 + 5) * (4 ^ 2 + 5 ^ 2) * (4 ^ 4 + 5 ^ 4) * (4 ^ 8 + 5 ^ 8) * (4 ^ 16 + 5 ^ 16) * (4 ^ 32 + 5 ^ 32) * (4 ^ 64 + 5 ^ 64) = 5 ^ 128 - 4 ^ 128 :=
by sorry

end simplify_expression_l1611_161181


namespace education_expenses_l1611_161177

theorem education_expenses (rent milk groceries petrol miscellaneous savings total_salary education : ℝ) 
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_petrol : petrol = 2000)
  (h_miscellaneous : miscellaneous = 6100)
  (h_savings : savings = 2400)
  (h_saving_percentage : savings = 0.10 * total_salary)
  (h_total_salary : total_salary = savings / 0.10)
  (h_total_expenses : total_salary - savings = rent + milk + groceries + petrol + miscellaneous + education) :
  education = 2500 :=
by
  sorry

end education_expenses_l1611_161177


namespace interval_length_l1611_161117

theorem interval_length (c d : ℝ) (h : (d - 5) / 3 - (c - 5) / 3 = 15) : d - c = 45 :=
sorry

end interval_length_l1611_161117


namespace vertex_set_is_parabola_l1611_161199

variables (a c k : ℝ) (ha : a > 0) (hc : c > 0) (hk : k ≠ 0)

theorem vertex_set_is_parabola :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) :=
sorry

end vertex_set_is_parabola_l1611_161199


namespace max_correct_answers_l1611_161122

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 30) (h2 : 4 * c - w = 85) : c ≤ 23 :=
  sorry

end max_correct_answers_l1611_161122


namespace center_square_is_15_l1611_161101

noncomputable def center_square_value : ℤ :=
  let d1 := (15 - 3) / 2
  let d3 := (33 - 9) / 2
  let middle_first_row := 3 + d1
  let middle_last_row := 9 + d3
  let d2 := (middle_last_row - middle_first_row) / 2
  middle_first_row + d2

theorem center_square_is_15 : center_square_value = 15 := by
  sorry

end center_square_is_15_l1611_161101


namespace scalene_triangle_cannot_be_divided_into_two_congruent_triangles_l1611_161111

-- Definitions and Conditions
structure Triangle :=
(a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)

-- Statement of the problem
theorem scalene_triangle_cannot_be_divided_into_two_congruent_triangles (T : Triangle) :
  ¬(∃ (D : ℝ) (ABD ACD : Triangle), ABD.a = ACD.a ∧ ABD.b = ACD.b ∧ ABD.c = ACD.c) :=
sorry

end scalene_triangle_cannot_be_divided_into_two_congruent_triangles_l1611_161111


namespace min_2a_plus_3b_l1611_161154

theorem min_2a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_parallel : (a * (b - 3) - 2 * b = 0)) :
  (2 * a + 3 * b) = 25 :=
by
  -- proof goes here
  sorry

end min_2a_plus_3b_l1611_161154


namespace marble_weights_total_l1611_161137

theorem marble_weights_total:
  0.33 + 0.33 + 0.08 + 0.25 + 0.02 + 0.12 + 0.15 = 1.28 :=
by {
  sorry
}

end marble_weights_total_l1611_161137


namespace partial_fraction_product_l1611_161115

theorem partial_fraction_product : 
  (∃ A B C : ℚ, 
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 5 → 
      (x^2 - 21) / ((x - 3) * (x + 3) * (x - 5)) = A / (x - 3) + B / (x + 3) + C / (x - 5))
      ∧ (A * B * C = -1/16)) := 
    sorry

end partial_fraction_product_l1611_161115


namespace range_of_m_l1611_161118

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem range_of_m (m : ℝ) : (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) ↔ m ≤ -1/3 :=
by
  sorry

end range_of_m_l1611_161118


namespace domain_of_sqrt_tan_minus_one_l1611_161156

open Real
open Set

def domain_sqrt_tan_minus_one : Set ℝ := 
  ⋃ k : ℤ, Ico (π/4 + k * π) (π/2 + k * π)

theorem domain_of_sqrt_tan_minus_one :
  {x : ℝ | ∃ y : ℝ, y = sqrt (tan x - 1)} = domain_sqrt_tan_minus_one :=
sorry

end domain_of_sqrt_tan_minus_one_l1611_161156


namespace smaller_number_is_five_l1611_161189

theorem smaller_number_is_five (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end smaller_number_is_five_l1611_161189


namespace correct_average_marks_l1611_161135

theorem correct_average_marks 
  (n : ℕ) (wrong_avg : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (h1 : n = 10)
  (h2 : wrong_avg = 100)
  (h3 : wrong_mark = 90)
  (h4 : correct_mark = 10) :
  (n * wrong_avg - wrong_mark + correct_mark) / n = 92 :=
by
  sorry

end correct_average_marks_l1611_161135


namespace find_angle_complement_supplement_l1611_161159

theorem find_angle_complement_supplement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end find_angle_complement_supplement_l1611_161159


namespace combined_weight_l1611_161198

-- Define the main proof problem
theorem combined_weight (student_weight : ℝ) (sister_weight : ℝ) :
  (student_weight - 5 = 2 * sister_weight) ∧ (student_weight = 79) → (student_weight + sister_weight = 116) :=
by
  sorry

end combined_weight_l1611_161198


namespace num_integers_between_700_and_900_with_sum_of_digits_18_l1611_161157

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem num_integers_between_700_and_900_with_sum_of_digits_18 : 
  ∃ k, k = 17 ∧ ∀ n, 700 ≤ n ∧ n ≤ 900 ∧ sum_of_digits n = 18 ↔ (1 ≤ k) := 
sorry

end num_integers_between_700_and_900_with_sum_of_digits_18_l1611_161157


namespace sum_possible_x_l1611_161191

noncomputable def sum_of_x (x : ℝ) : ℝ :=
  let lst : List ℝ := [1, 2, 5, 2, 3, 2, x]
  let mean := (1 + 2 + 5 + 2 + 3 + 2 + x) / 7
  let median := 2
  let mode := 2
  if lst = List.reverse lst ∧ mean ≠ mode then
    mean
  else 
    0

theorem sum_possible_x : sum_of_x 1 + sum_of_x 5 = 6 :=
by 
  sorry

end sum_possible_x_l1611_161191


namespace common_ratio_geometric_series_l1611_161130

theorem common_ratio_geometric_series (a r S : ℝ) (h₁ : S = a / (1 - r))
  (h₂ : r ≠ 1)
  (h₃ : r^4 * S = S / 81) :
  r = 1/3 :=
by 
  sorry

end common_ratio_geometric_series_l1611_161130


namespace arithmetic_sequence_fourth_term_l1611_161142

theorem arithmetic_sequence_fourth_term (b d : ℝ) (h : 2 * b + 2 * d = 10) : b + d = 5 :=
by
  sorry

end arithmetic_sequence_fourth_term_l1611_161142


namespace calculation_correct_l1611_161108

theorem calculation_correct : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end calculation_correct_l1611_161108


namespace total_pencils_l1611_161173

-- Define the initial conditions
def initial_pencils : ℕ := 41
def added_pencils : ℕ := 30

-- Define the statement to be proven
theorem total_pencils :
  initial_pencils + added_pencils = 71 :=
by
  sorry

end total_pencils_l1611_161173


namespace solve_for_x_l1611_161151

theorem solve_for_x (x : ℚ) (h : 1/4 + 7/x = 13/x + 1/9) : x = 216/5 :=
by
  sorry

end solve_for_x_l1611_161151


namespace inequality_proof_l1611_161190

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) : a < 2 * b - b^2 / a := 
by
  -- mathematical proof goes here
  sorry

end inequality_proof_l1611_161190


namespace solution_of_system_l1611_161141

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l1611_161141


namespace crates_of_mangoes_sold_l1611_161165

def total_crates_sold := 50
def crates_grapes_sold := 13
def crates_passion_fruits_sold := 17

theorem crates_of_mangoes_sold : 
  (total_crates_sold - (crates_grapes_sold + crates_passion_fruits_sold) = 20) :=
by 
  sorry

end crates_of_mangoes_sold_l1611_161165


namespace sum_of_sides_eq_13_or_15_l1611_161134

noncomputable def squares_side_lengths (b d : ℕ) : Prop :=
  15^2 = b^2 + 10^2 + d^2

theorem sum_of_sides_eq_13_or_15 :
  ∃ b d : ℕ, squares_side_lengths b d ∧ (b + d = 13 ∨ b + d = 15) :=
sorry

end sum_of_sides_eq_13_or_15_l1611_161134


namespace initial_percentage_correct_l1611_161140

noncomputable def percentInitiallyFull (initialWater: ℕ) (waterAdded: ℕ) (fractionFull: ℚ) (capacity: ℕ) : ℚ :=
  (initialWater : ℚ) / (capacity : ℚ) * 100

theorem initial_percentage_correct (initialWater waterAdded capacity: ℕ) (fractionFull: ℚ) :
  waterAdded = 14 →
  fractionFull = 3/4 →
  capacity = 40 →
  initialWater + waterAdded = fractionFull * capacity →
  percentInitiallyFull initialWater waterAdded fractionFull capacity = 40 :=
by
  intros h1 h2 h3 h4
  unfold percentInitiallyFull
  sorry

end initial_percentage_correct_l1611_161140


namespace prove_bounds_l1611_161144

variable (a b : ℝ)

-- Conditions
def condition1 : Prop := 6 * a - b = 45
def condition2 : Prop := 4 * a + b > 60

-- Proof problem statement
theorem prove_bounds (h1 : condition1 a b) (h2 : condition2 a b) : a > 10.5 ∧ b > 18 :=
sorry

end prove_bounds_l1611_161144


namespace problem_statement_l1611_161125

variable {x : ℝ}
noncomputable def A : ℝ := 39
noncomputable def B : ℝ := -5

theorem problem_statement (h : ∀ x ≠ 3, (A / (x - 3) + B * (x + 2)) = (-5 * x ^ 2 + 18 * x + 30) / (x - 3)) : A + B = 34 := 
sorry

end problem_statement_l1611_161125


namespace total_cost_of_barbed_wire_l1611_161182

noncomputable def cost_of_barbed_wire : ℝ :=
  let area : ℝ := 3136
  let side_length : ℝ := Real.sqrt area
  let perimeter_without_gates : ℝ := 4 * side_length - 2 * 1
  let rate_per_meter : ℝ := 1.10
  perimeter_without_gates * rate_per_meter

theorem total_cost_of_barbed_wire :
  cost_of_barbed_wire = 244.20 :=
sorry

end total_cost_of_barbed_wire_l1611_161182


namespace jessies_weight_loss_l1611_161105

-- Definitions based on the given conditions
def initial_weight : ℝ := 74
def weight_loss_rate_even_days : ℝ := 0.2 + 0.15
def weight_loss_rate_odd_days : ℝ := 0.3
def total_exercise_days : ℕ := 25
def even_days : ℕ := (total_exercise_days - 1) / 2
def odd_days : ℕ := even_days + 1

-- The goal is to prove the total weight loss is 8.1 kg
theorem jessies_weight_loss : 
  (even_days * weight_loss_rate_even_days + odd_days * weight_loss_rate_odd_days) = 8.1 := 
by
  sorry

end jessies_weight_loss_l1611_161105


namespace inscribed_circle_radius_l1611_161185

variable (AB AC BC s K r : ℝ)
variable (AB_eq AC_eq BC_eq : AB = AC ∧ AC = 8 ∧ BC = 7)
variable (s_eq : s = (AB + AC + BC) / 2)
variable (K_eq : K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)))
variable (r_eq : r * s = K)

/-- Prove that the radius of the inscribed circle is 23.75 / 11.5 given the conditions of the triangle --/
theorem inscribed_circle_radius :
  AB = 8 → AC = 8 → BC = 7 → 
  s = (AB + AC + BC) / 2 → 
  K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) →
  r * s = K →
  r = (23.75 / 11.5) :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end inscribed_circle_radius_l1611_161185


namespace CarrieSpent_l1611_161104

variable (CostPerShirt NumberOfShirts : ℝ)

def TotalCost (CostPerShirt NumberOfShirts : ℝ) : ℝ :=
  CostPerShirt * NumberOfShirts

theorem CarrieSpent {CostPerShirt NumberOfShirts : ℝ} 
  (h1 : CostPerShirt = 9.95) 
  (h2 : NumberOfShirts = 20) : 
  TotalCost CostPerShirt NumberOfShirts = 199.00 :=
by
  sorry

end CarrieSpent_l1611_161104


namespace slope_range_l1611_161166

variables (x y k : ℝ)

theorem slope_range :
  (2 ≤ x ∧ x ≤ 3) ∧ (y = -2 * x + 8) ∧ (k = -3 * y / (2 * x)) →
  -3 ≤ k ∧ k ≤ -1 :=
by
  sorry

end slope_range_l1611_161166


namespace angle_C_is_100_l1611_161187

-- Define the initial measures in the equilateral triangle
def initial_angle (A B C : ℕ) (h_equilateral : A = B ∧ B = C ∧ C = 60) : ℕ := C

-- Definition to capture the increase in angle C
def increased_angle (C : ℕ) : ℕ := C + 40

-- Now, we need to state the theorem assuming the given conditions
theorem angle_C_is_100
  (A B C : ℕ)
  (h_equilateral : A = 60 ∧ B = 60 ∧ C = 60)
  (h_increase : C = 60 + 40)
  : C = 100 := 
sorry

end angle_C_is_100_l1611_161187


namespace max_quadratic_in_interval_l1611_161128

-- Define the quadratic function
noncomputable def quadratic_fun (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the closed interval
def interval (a b : ℝ) (x : ℝ) : Prop := a ≤ x ∧ x ≤ b

-- Define the maximum value property
def is_max_value (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, interval a b x → f x ≤ max_val

-- State the problem in Lean 4
theorem max_quadratic_in_interval : 
  is_max_value quadratic_fun (-5) 3 36 := 
sorry

end max_quadratic_in_interval_l1611_161128


namespace solution_is_x_l1611_161175

def find_x (x : ℝ) : Prop :=
  64 * (x + 1)^3 - 27 = 0

theorem solution_is_x : ∃ x : ℝ, find_x x ∧ x = -1 / 4 :=
by
  sorry

end solution_is_x_l1611_161175


namespace train_length_l1611_161195

theorem train_length (L : ℝ) : (L + 200) / 15 = (L + 300) / 20 → L = 100 :=
by
  intro h
  -- Skipping the proof steps
  sorry

end train_length_l1611_161195


namespace probability_ratio_l1611_161119

-- Defining the total number of cards and each number's frequency
def total_cards := 60
def each_number_frequency := 4
def distinct_numbers := 15

-- Defining probability p' and q'
def p' := (15: ℕ) * (Nat.choose 4 4) / (Nat.choose 60 4)
def q' := 210 * (Nat.choose 4 3) * (Nat.choose 4 1) / (Nat.choose 60 4)

-- Prove the value of q'/p'
theorem probability_ratio : (q' / p') = 224 := by
  sorry

end probability_ratio_l1611_161119


namespace packet_weight_l1611_161197

theorem packet_weight :
  ∀ (num_packets : ℕ) (total_weight_kg : ℕ), 
  num_packets = 20 → total_weight_kg = 2 →
  (total_weight_kg * 1000) / num_packets = 100 := by
  intro num_packets total_weight_kg h1 h2
  sorry

end packet_weight_l1611_161197


namespace parabola_origin_l1611_161196

theorem parabola_origin (x y c : ℝ) (h : y = x^2 - 2 * x + c - 4) (h0 : (0, 0) = (x, y)) : c = 4 :=
by
  sorry

end parabola_origin_l1611_161196


namespace solve_for_x_l1611_161127

theorem solve_for_x (x : ℚ) (h₁ : (7 * x + 2) / (x - 4) = -6 / (x - 4)) (h₂ : x ≠ 4) :
  x = -8 / 7 := 
  sorry

end solve_for_x_l1611_161127


namespace ratio_of_volumes_l1611_161163

-- Definitions based on given conditions
def V1 : ℝ := sorry -- Volume of the first vessel
def V2 : ℝ := sorry -- Volume of the second vessel

-- Given condition
def condition : Prop := (3 / 4) * V1 = (5 / 8) * V2

-- The theorem to prove the ratio V1 / V2 is 5 / 6
theorem ratio_of_volumes (h : condition) : V1 / V2 = 5 / 6 :=
sorry

end ratio_of_volumes_l1611_161163


namespace train_speed_l1611_161180

theorem train_speed (lt_train : ℝ) (lt_bridge : ℝ) (time_cross : ℝ) (total_speed_kmph : ℝ) :
  lt_train = 150 ∧ lt_bridge = 225 ∧ time_cross = 30 ∧ total_speed_kmph = (375 / 30) * 3.6 → 
  total_speed_kmph = 45 := 
by
  sorry

end train_speed_l1611_161180


namespace volume_of_inequality_region_l1611_161103

-- Define the inequality condition as a predicate
def region (x y z : ℝ) : Prop :=
  |4 * x - 20| + |3 * y + 9| + |z - 2| ≤ 6

-- Define the volume calculation for the region
def volume_of_region := 36

-- The proof statement
theorem volume_of_inequality_region : 
  (∃ x y z : ℝ, region x y z) → volume_of_region = 36 :=
by
  sorry

end volume_of_inequality_region_l1611_161103


namespace problem_statement_l1611_161158

variable (x1 x2 x3 x4 x5 x6 x7 : ℝ)

theorem problem_statement
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 20)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 145) :
  16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 380 :=
sorry

end problem_statement_l1611_161158


namespace probability_intersection_l1611_161179

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l1611_161179


namespace percent_of_a_is_4b_l1611_161162

variable (a b : ℝ)

theorem percent_of_a_is_4b (hab : a = 1.8 * b) :
  (4 * b / a) * 100 = 222.22 := by
  sorry

end percent_of_a_is_4b_l1611_161162


namespace minimize_time_theta_l1611_161184

theorem minimize_time_theta (α θ : ℝ) (h1 : 0 < α) (h2 : α < 90) (h3 : θ = α / 2) : 
  θ = α / 2 :=
by
  sorry

end minimize_time_theta_l1611_161184


namespace mustard_found_at_second_table_l1611_161176

variables (total_mustard first_table third_table second_table : ℝ)

def mustard_found (total_mustard first_table third_table : ℝ) := total_mustard - (first_table + third_table)

theorem mustard_found_at_second_table
    (h_total : total_mustard = 0.88)
    (h_first : first_table = 0.25)
    (h_third : third_table = 0.38) :
    mustard_found total_mustard first_table third_table = 0.25 :=
by
    rw [mustard_found, h_total, h_first, h_third]
    simp
    sorry

end mustard_found_at_second_table_l1611_161176


namespace sample_group_b_correct_l1611_161171

noncomputable def stratified_sample_group_b (total_cities: ℕ) (group_b_cities: ℕ) (sample_size: ℕ) : ℕ :=
  (sample_size * group_b_cities) / total_cities

theorem sample_group_b_correct : stratified_sample_group_b 36 12 12 = 4 := by
  sorry

end sample_group_b_correct_l1611_161171


namespace tan_alpha_value_complex_expression_value_l1611_161112

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 :=
sorry

theorem complex_expression_value 
(α : ℝ) 
(h1 : Real.tan (π / 4 + α) = 1 / 2) 
(h2 : Real.tan α = -1 / 3) : 
Real.sin (2 * α + 2 * π) - (Real.sin (π / 2 - α))^2 / 
(1 - Real.cos (π - 2 * α) + (Real.sin α)^2) = -15 / 19 :=
sorry

end tan_alpha_value_complex_expression_value_l1611_161112


namespace triangle_inequality_iff_inequality_l1611_161153

theorem triangle_inequality_iff_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) ↔ (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  sorry

end triangle_inequality_iff_inequality_l1611_161153


namespace sum_of_vars_l1611_161121

theorem sum_of_vars 
  (x y z : ℝ) 
  (h1 : x + y = 4) 
  (h2 : y + z = 6) 
  (h3 : z + x = 8) : 
  x + y + z = 9 := 
by 
  sorry

end sum_of_vars_l1611_161121


namespace total_books_gwen_has_l1611_161136

-- Definitions based on conditions in part a
def mystery_shelves : ℕ := 5
def picture_shelves : ℕ := 3
def books_per_shelf : ℕ := 4

-- Problem statement in Lean 4
theorem total_books_gwen_has : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 32 := by
  -- This is where the proof would go, but we include sorry to skip for now
  sorry

end total_books_gwen_has_l1611_161136


namespace hyperbola_parabola_intersection_l1611_161120

open Real

theorem hyperbola_parabola_intersection :
  let A := (4, 4)
  let B := (4, -4)
  |dist A B| = 8 :=
by
  let hyperbola_asymptote (x y: ℝ) := x^2 - y^2 = 1
  let parabola_equation (x y : ℝ) := y^2 = 4 * x
  sorry

end hyperbola_parabola_intersection_l1611_161120


namespace arithmetic_sequence_ratio_l1611_161183

open Nat

noncomputable def S (n : ℕ) : ℝ := n^2
noncomputable def T (n : ℕ) : ℝ := n * (2 * n + 3)

theorem arithmetic_sequence_ratio 
  (h : ∀ n : ℕ, (2 * n + 3) * S n = n * T n) : 
  (S 5 - S 4) / (T 6 - T 5) = 9 / 25 := by
  sorry

end arithmetic_sequence_ratio_l1611_161183


namespace yogurt_combinations_l1611_161167

theorem yogurt_combinations (flavors toppings : ℕ) (h_flavors : flavors = 5) (h_toppings : toppings = 7) :
  (flavors * Nat.choose toppings 3) = 175 := by
  sorry

end yogurt_combinations_l1611_161167


namespace fenced_area_l1611_161145

theorem fenced_area (L W : ℝ) (square_side triangle_leg : ℝ) :
  L = 20 ∧ W = 18 ∧ square_side = 4 ∧ triangle_leg = 3 →
  (L * W - square_side^2 - (1 / 2) * triangle_leg^2 = 339.5) := by
  intros h
  rcases h with ⟨hL, hW, hs, ht⟩
  rw [hL, hW, hs, ht]
  simp
  sorry

end fenced_area_l1611_161145


namespace study_tour_buses_l1611_161186

variable (x : ℕ) (num_people : ℕ)

def seats_A := 45
def seats_B := 60
def extra_people := 30
def fewer_B := 6

theorem study_tour_buses (h : seats_A * x + extra_people = seats_B * (x - fewer_B)) : 
  x = 26 ∧ (seats_A * 26 + extra_people = 1200) := 
  sorry

end study_tour_buses_l1611_161186


namespace complementary_event_equivalence_l1611_161138

-- Define the event E: hitting the target at least once in two shots.
-- Event E complementary: missing the target both times.

def eventE := "hitting the target at least once"
def complementaryEvent := "missing the target both times"

theorem complementary_event_equivalence :
  (complementaryEvent = "missing the target both times") ↔ (eventE = "hitting the target at least once") :=
by
  sorry

end complementary_event_equivalence_l1611_161138


namespace solution_set_ineq_l1611_161126

theorem solution_set_ineq (x : ℝ) : 
  (x - 1) / (2 * x + 3) > 1 ↔ -4 < x ∧ x < -3 / 2 :=
by
  sorry

end solution_set_ineq_l1611_161126


namespace circle_center_sum_l1611_161132

theorem circle_center_sum (x y : ℝ) (h : (x - 5)^2 + (y - 2)^2 = 38) : x + y = 7 := 
  sorry

end circle_center_sum_l1611_161132


namespace rectangle_area_l1611_161139

namespace RectangleAreaProof

theorem rectangle_area (SqrArea : ℝ) (SqrSide : ℝ) (RectWidth : ℝ) (RectLength : ℝ) (RectArea : ℝ) :
  SqrArea = 36 →
  SqrSide = Real.sqrt SqrArea →
  RectWidth = SqrSide →
  RectLength = 3 * RectWidth →
  RectArea = RectWidth * RectLength →
  RectArea = 108 := by
  sorry

end RectangleAreaProof

end rectangle_area_l1611_161139


namespace sum_local_values_of_digits_l1611_161129

theorem sum_local_values_of_digits :
  let d2 := 2000
  let d3 := 300
  let d4 := 40
  let d5 := 5
  d2 + d3 + d4 + d5 = 2345 :=
by
  sorry

end sum_local_values_of_digits_l1611_161129


namespace equation_has_exactly_one_solution_l1611_161192

theorem equation_has_exactly_one_solution (m : ℝ) : 
  (m ∈ { -1 } ∪ Set.Ioo (-1/2 : ℝ) (1/0) ) ↔ ∃ (x : ℝ), 2 * Real.sqrt (1 - m * (x + 2)) = x + 4 :=
sorry

end equation_has_exactly_one_solution_l1611_161192


namespace babjis_height_less_by_20_percent_l1611_161147

variable (B A : ℝ) (h : A = 1.25 * B)

theorem babjis_height_less_by_20_percent : ((A - B) / A) * 100 = 20 := by
  sorry

end babjis_height_less_by_20_percent_l1611_161147


namespace combined_soldiers_correct_l1611_161116

-- Define the parameters for the problem
def interval : ℕ := 5
def wall_length : ℕ := 7300
def soldiers_per_tower : ℕ := 2

-- Calculate the number of towers and the total number of soldiers
def num_towers : ℕ := wall_length / interval
def combined_soldiers : ℕ := num_towers * soldiers_per_tower

-- Prove that the combined number of soldiers is as expected
theorem combined_soldiers_correct : combined_soldiers = 2920 := 
by
  sorry

end combined_soldiers_correct_l1611_161116


namespace intersection_of_A_and_B_l1611_161169

open Set

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 9} :=
by
  sorry

end intersection_of_A_and_B_l1611_161169


namespace sum_first_6_is_correct_l1611_161124

namespace ProofProblem

def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ ∀ n : ℕ, n ≥ 2 → a (n - 1) = 2 * a n

def sum_first_6 (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_first_6_is_correct (a : ℕ → ℚ) (h : sequence a) :
  sum_first_6 a = 63 / 32 :=
sorry

end ProofProblem

end sum_first_6_is_correct_l1611_161124


namespace profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l1611_161148

-- Define the selling price and cost price
def cost_price : ℝ := 60
def sales_price (x : ℝ) := x

-- 1. Prove the profit per piece
def profit_per_piece (x : ℝ) : ℝ := sales_price x - cost_price

theorem profit_per_piece_correct (x : ℝ) : profit_per_piece x = x - 60 :=
by 
  -- it follows directly from the definition of profit_per_piece
  sorry

-- 2. Define the linear function relationship between monthly sales volume and selling price
def sales_volume (x : ℝ) : ℝ := -2 * x + 400

theorem sales_volume_correct (x : ℝ) : sales_volume x = -2 * x + 400 :=
by 
  -- it follows directly from the definition of sales_volume
  sorry

-- 3. Define the monthly profit and prove the maximized profit
def monthly_profit (x : ℝ) : ℝ := profit_per_piece x * sales_volume x

theorem maximum_monthly_profit (x : ℝ) : 
  monthly_profit x = -2 * x^2 + 520 * x - 24000 :=
by 
  -- it follows directly from the definition of monthly_profit
  sorry

theorem optimum_selling_price_is_130 : ∃ (x : ℝ), (monthly_profit x = 9800) ∧ (x = 130) :=
by
  -- solve this using the properties of quadratic functions
  sorry

end profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l1611_161148


namespace determine_numbers_l1611_161110

theorem determine_numbers (a b c : ℕ) (h₁ : a + b + c = 15) 
  (h₂ : (1 / (a : ℝ)) + (1 / (b : ℝ)) + (1 / (c : ℝ)) = 71 / 105) : 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 5) ∨ (a = 5 ∧ b = 3 ∧ c = 7) ∨ 
  (a = 5 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 5) ∨ (a = 7 ∧ b = 5 ∧ c = 3) :=
sorry

end determine_numbers_l1611_161110


namespace determine_treasures_possible_l1611_161188

structure Subject :=
  (is_knight : Prop)
  (is_liar : Prop)
  (is_normal : Prop)

def island_has_treasures : Prop := sorry

def can_determine_treasures (A B C : Subject) (at_most_one_normal : Bool) : Prop :=
  if at_most_one_normal then
    ∃ (question : (Subject → Prop)),
      (∀ response1, ∃ (question2 : (Subject → Prop)),
        (∀ response2, island_has_treasures ↔ (response1 ∧ response2)))
  else
    false

theorem determine_treasures_possible (A B C : Subject) (at_most_one_normal : Bool) :
  at_most_one_normal = true → can_determine_treasures A B C at_most_one_normal :=
by
  intro h
  sorry

end determine_treasures_possible_l1611_161188


namespace speed_last_segment_l1611_161146

-- Definitions corresponding to conditions
def drove_total_distance : ℝ := 150
def total_time_minutes : ℝ := 120
def time_first_segment_minutes : ℝ := 40
def speed_first_segment_mph : ℝ := 70
def speed_second_segment_mph : ℝ := 75

-- The statement of the problem
theorem speed_last_segment :
  let total_distance : ℝ := drove_total_distance
  let total_time : ℝ := total_time_minutes / 60
  let time_first_segment : ℝ := time_first_segment_minutes / 60
  let time_second_segment : ℝ := time_first_segment
  let time_last_segment : ℝ := time_first_segment
  let distance_first_segment : ℝ := speed_first_segment_mph * time_first_segment
  let distance_second_segment : ℝ := speed_second_segment_mph * time_second_segment
  let distance_two_segments : ℝ := distance_first_segment + distance_second_segment
  let distance_last_segment : ℝ := total_distance - distance_two_segments
  let speed_last_segment := distance_last_segment / time_last_segment
  speed_last_segment = 80 := 
  sorry

end speed_last_segment_l1611_161146


namespace max_integer_value_of_f_l1611_161161

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 9 * x + 13) / (3 * x^2 + 9 * x + 5)

theorem max_integer_value_of_f : ∀ x : ℝ, ∃ n : ℤ, f x ≤ n ∧ n = 2 :=
by 
  sorry

end max_integer_value_of_f_l1611_161161


namespace angle_A_area_of_triangle_l1611_161131

open Real

theorem angle_A (a : ℝ) (A B C : ℝ) 
  (h_a : a = 2 * sqrt 3)
  (h_condition1 : 4 * cos A ^ 2 + 4 * cos B * cos C + 1 = 4 * sin B * sin C) :
  A = π / 3 := 
sorry

theorem area_of_triangle (a b c A : ℝ) 
  (h_A : A = π / 3)
  (h_a : a = 2 * sqrt 3)
  (h_b : b = 3 * c) :
  (1 / 2) * b * c * sin A = 9 * sqrt 3 / 7 := 
sorry

end angle_A_area_of_triangle_l1611_161131


namespace simplify_expr_l1611_161160

variable (a b : ℝ)

theorem simplify_expr (h : a + b ≠ 0) : 
  a - b + 2 * b^2 / (a + b) = (a^2 + b^2) / (a + b) :=
sorry

end simplify_expr_l1611_161160


namespace f_g_2_eq_neg_19_l1611_161106

def f (x : ℝ) : ℝ := 5 - 4 * x

def g (x : ℝ) : ℝ := x^2 + 2

theorem f_g_2_eq_neg_19 : f (g 2) = -19 := 
by
  -- The proof is omitted
  sorry

end f_g_2_eq_neg_19_l1611_161106


namespace tangent_line_circle_l1611_161109

theorem tangent_line_circle (r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = r^2 → x + y = 2 * r) ↔ r = 2 + Real.sqrt 2 :=
by
  sorry

end tangent_line_circle_l1611_161109


namespace no_four_primes_exist_l1611_161143

theorem no_four_primes_exist (a b c d : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b)
  (hc : Nat.Prime c) (hd : Nat.Prime d) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : (1 / a : ℚ) + (1 / d) = (1 / b) + (1 / c)) : False := sorry

end no_four_primes_exist_l1611_161143


namespace value_of_a_minus_b_l1611_161133

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a + b > 0) :
  a - b = 4 ∨ a - b = 8 :=
  sorry

end value_of_a_minus_b_l1611_161133


namespace division_quotient_is_correct_l1611_161170

noncomputable def polynomial_division_quotient : Polynomial ℚ :=
  Polynomial.div (Polynomial.C 8 * Polynomial.X ^ 3 + 
                  Polynomial.C 16 * Polynomial.X ^ 2 + 
                  Polynomial.C (-7) * Polynomial.X + 
                  Polynomial.C 4) 
                 (Polynomial.C 2 * Polynomial.X + Polynomial.C 5)

theorem division_quotient_is_correct :
  polynomial_division_quotient =
    Polynomial.C 4 * Polynomial.X ^ 2 +
    Polynomial.C (-2) * Polynomial.X +
    Polynomial.C (3 / 2) :=
by
  sorry

end division_quotient_is_correct_l1611_161170


namespace students_growth_rate_l1611_161149

theorem students_growth_rate (x : ℝ) 
  (h_total : 728 = 200 + 200 * (1+x) + 200 * (1+x)^2) : 
  200 + 200 * (1+x) + 200*(1+x)^2 = 728 := 
  by
  sorry

end students_growth_rate_l1611_161149


namespace problem_l1611_161150

namespace MathProof

variable {p a b : ℕ}

theorem problem (h1 : Nat.Prime p) (h2 : p % 2 = 1) (h3 : a > 0) (h4 : b > 0) (h5 : (p + 1)^a - p^b = 1) : a = 1 ∧ b = 1 := 
sorry

end MathProof

end problem_l1611_161150
