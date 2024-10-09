import Mathlib

namespace range_x_y_l551_55111

variable (x y : ℝ)

theorem range_x_y (hx : 60 < x ∧ x < 84) (hy : 28 < y ∧ y < 33) : 
  27 < x - y ∧ x - y < 56 :=
sorry

end range_x_y_l551_55111


namespace infinitely_many_lovely_no_lovely_square_gt_1_l551_55130

def lovely (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin k → ℕ),
    n = (List.ofFn d).prod ∧
    ∀ i, (d i)^2 ∣ n + (d i)

theorem infinitely_many_lovely : ∀ N : ℕ, ∃ n > N, lovely n :=
  sorry

theorem no_lovely_square_gt_1 : ∀ n : ℕ, n > 1 → lovely n → ¬∃ m, n = m^2 :=
  sorry

end infinitely_many_lovely_no_lovely_square_gt_1_l551_55130


namespace susan_change_sum_susan_possible_sums_l551_55110

theorem susan_change_sum
  (change : ℕ)
  (h_lt_100 : change < 100)
  (h_nickels : ∃ k : ℕ, change = 5 * k + 2)
  (h_quarters : ∃ m : ℕ, change = 25 * m + 5) :
  change = 30 ∨ change = 55 ∨ change = 80 :=
sorry

theorem susan_possible_sums :
  30 + 55 + 80 = 165 :=
by norm_num

end susan_change_sum_susan_possible_sums_l551_55110


namespace find_k_and_other_root_l551_55182

def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem find_k_and_other_root (k β : ℝ) (h1 : quadratic_eq 4 k 2 (-0.5)) (h2 : 4 * (-0.5) ^ 2 + k * (-0.5) + 2 = 0) : 
  k = 6 ∧ β = -1 ∧ quadratic_eq 4 k 2 β := 
by 
  sorry

end find_k_and_other_root_l551_55182


namespace parabola_coefficients_l551_55186

theorem parabola_coefficients (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (y = (x + 2)^2 + 5) ∧ y = 9 ↔ x = 0) →
  (a, b, c) = (1, 4, 9) :=
by
  intros h
  sorry

end parabola_coefficients_l551_55186


namespace find_alpha_l551_55103

theorem find_alpha (α β : ℝ) (h1 : Real.arctan α = 1/2) (h2 : Real.arctan (α - β) = 1/3)
  (h3 : 0 < α ∧ α < π/2) (h4 : 0 < β ∧ β < π/2) : α = π/4 := by
  sorry

end find_alpha_l551_55103


namespace factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l551_55144

-- Statement for question 1
theorem factorize_m4_minus_5m_plus_4 (m : ℤ) : 
  (m ^ 4 - 5 * m + 4) = (m ^ 4 - 5 * m + 4) := sorry

-- Statement for question 2
theorem factorize_x3_plus_2x2_plus_4x_plus_3 (x : ℝ) :
  (x ^ 3 + 2 * x ^ 2 + 4 * x + 3) = (x + 1) * (x ^ 2 + x + 3) := sorry

-- Statement for question 3
theorem factorize_x5_minus_1 (x : ℝ) :
  (x ^ 5 - 1) = (x - 1) * (x ^ 4 + x ^ 3 + x ^ 2 + x + 1) := sorry

end factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l551_55144


namespace optimal_washing_effect_l551_55170

noncomputable def optimal_laundry_addition (x y : ℝ) : Prop :=
  (5 + 0.02 * 2 + x + y = 20) ∧
  (0.02 * 2 + x = (20 - 5) * 0.004)

theorem optimal_washing_effect :
  ∃ x y : ℝ, optimal_laundry_addition x y ∧ x = 0.02 ∧ y = 14.94 :=
by
  sorry

end optimal_washing_effect_l551_55170


namespace minimize_sum_of_squares_l551_55101

theorem minimize_sum_of_squares (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 16) :
  a^2 + b^2 + c^2 ≥ 86 :=
sorry

end minimize_sum_of_squares_l551_55101


namespace option_B_proof_option_C_proof_l551_55149

-- Definitions and sequences
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Statement of the problem

theorem option_B_proof (A B : ℝ) :
  (∀ n : ℕ, S n = A * (n : ℝ)^2 + B * n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d := 
sorry

theorem option_C_proof :
  (∀ n : ℕ, S n = 1 - (-1)^n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end option_B_proof_option_C_proof_l551_55149


namespace range_of_a_l551_55127

noncomputable def exists_unique_y (a : ℝ) (x : ℝ) : Prop :=
∃! (y : ℝ), y ∈ Set.Icc (-1) 1 ∧ x + y^2 * Real.exp y = a

theorem range_of_a (e : ℝ) (H_e : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 0 1, exists_unique_y a x) →
  a ∈ Set.Ioc (1 + 1/e) e :=
by
  sorry

end range_of_a_l551_55127


namespace exists_C_a_n1_minus_a_n_l551_55184

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| 2 => 8
| (n+1) => a (n - 1) + (4 / n) * a n

theorem exists_C (C : ℕ) (hC : C = 2) : ∃ C > 0, ∀ n > 0, a n ≤ C * n^2 := by
  use 2
  sorry

theorem a_n1_minus_a_n (n : ℕ) (h : n > 0) : a (n + 1) - a n ≤ 4 * n + 3 := by
  sorry

end exists_C_a_n1_minus_a_n_l551_55184


namespace InfinitePairsExist_l551_55178

theorem InfinitePairsExist (a b : ℕ) : (∀ n : ℕ, ∃ a b : ℕ, a ∣ b^2 + 1 ∧ b ∣ a^2 + 1) :=
sorry

end InfinitePairsExist_l551_55178


namespace find_S_l551_55193

theorem find_S (x y : ℝ) (h : x + y = 4) : 
  ∃ S, (∀ x y, x + y = 4 → 3*x^2 + y^2 = 12) → S = 6 := 
by 
  sorry

end find_S_l551_55193


namespace total_stuffed_animals_l551_55157

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end total_stuffed_animals_l551_55157


namespace total_staff_correct_l551_55122

noncomputable def total_staff_weekdays_weekends : ℕ := 84

theorem total_staff_correct :
  let chefs_weekdays := 16
  let waiters_weekdays := 16
  let busboys_weekdays := 10
  let hostesses_weekdays := 5
  let additional_chefs_weekends := 5
  let additional_hostesses_weekends := 2
  
  let chefs_leave := chefs_weekdays * 25 / 100
  let waiters_leave := waiters_weekdays * 20 / 100
  let busboys_leave := busboys_weekdays * 30 / 100
  let hostesses_leave := hostesses_weekdays * 15 / 100
  
  let chefs_left_weekdays := chefs_weekdays - chefs_leave
  let waiters_left_weekdays := waiters_weekdays - Nat.floor waiters_leave
  let busboys_left_weekdays := busboys_weekdays - busboys_leave
  let hostesses_left_weekdays := hostesses_weekdays - Nat.ceil hostesses_leave

  let total_staff_weekdays := chefs_left_weekdays + waiters_left_weekdays + busboys_left_weekdays + hostesses_left_weekdays

  let chefs_weekends := chefs_weekdays + additional_chefs_weekends
  let waiters_weekends := waiters_left_weekdays
  let busboys_weekends := busboys_left_weekdays
  let hostesses_weekends := hostesses_weekdays + additional_hostesses_weekends
  
  let total_staff_weekends := chefs_weekends + waiters_weekends + busboys_weekends + hostesses_weekends

  total_staff_weekdays + total_staff_weekends = total_staff_weekdays_weekends
:= by
  sorry

end total_staff_correct_l551_55122


namespace one_twentieth_of_eighty_l551_55187

/--
Given the conditions, to prove that \(\frac{1}{20}\) of 80 is equal to 4.
-/
theorem one_twentieth_of_eighty : (80 : ℚ) * (1 / 20) = 4 :=
by
  sorry

end one_twentieth_of_eighty_l551_55187


namespace inequality_proof_l551_55192

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) >= (2 / 3) * (a^2 + b^2 + c^2) := 
sorry

end inequality_proof_l551_55192


namespace domain_of_f_l551_55148

noncomputable def f (x : ℝ) : ℝ := (x^3 - 125) / (x + 5)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≠ -5} := 
by
  sorry

end domain_of_f_l551_55148


namespace Clinton_belts_l551_55138

variable {Shoes Belts Hats : ℕ}

theorem Clinton_belts :
  (Shoes = 14) → (Shoes = 2 * Belts) → Belts = 7 :=
by
  sorry

end Clinton_belts_l551_55138


namespace ellipse_major_minor_axis_condition_l551_55140

theorem ellipse_major_minor_axis_condition (h1 : ∀ x y : ℝ, x^2 + m * y^2 = 1) 
                                          (h2 : ∀ a b : ℝ, a = 2 * b) :
  m = 1 / 4 :=
sorry

end ellipse_major_minor_axis_condition_l551_55140


namespace fencing_required_l551_55191

theorem fencing_required (L W : ℕ) (hL : L = 40) (hA : 40 * W = 680) : 2 * W + L = 74 :=
by sorry

end fencing_required_l551_55191


namespace final_apples_count_l551_55194

-- Definitions from the problem conditions
def initialApples : ℕ := 150
def soldToJill (initial : ℕ) : ℕ := initial * 30 / 100
def remainingAfterJill (initial : ℕ) := initial - soldToJill initial
def soldToJune (remaining : ℕ) : ℕ := remaining * 20 / 100
def remainingAfterJune (remaining : ℕ) := remaining - soldToJune remaining
def givenToFriend (current : ℕ) : ℕ := current - 2
def soldAfterFriend (current : ℕ) : ℕ := current * 10 / 100
def remainingAfterAll (current : ℕ) := current - soldAfterFriend current

theorem final_apples_count : remainingAfterAll (givenToFriend (remainingAfterJune (remainingAfterJill initialApples))) = 74 :=
by
  sorry

end final_apples_count_l551_55194


namespace equilateral_triangle_side_length_l551_55105

noncomputable def side_length_of_triangle (PQ PR PS : ℕ) : ℝ := 
  let s := 8 * Real.sqrt 3
  s

theorem equilateral_triangle_side_length (PQ PR PS : ℕ) (P_inside_triangle : true) 
  (Q_foot : true) (R_foot : true) (S_foot : true)
  (hPQ : PQ = 2) (hPR : PR = 4) (hPS : PS = 6) : 
  side_length_of_triangle PQ PR PS = 8 * Real.sqrt 3 := 
sorry

end equilateral_triangle_side_length_l551_55105


namespace number_difference_l551_55117

theorem number_difference 
  (a1 a2 a3 : ℝ)
  (h1 : a1 = 2 * a2)
  (h2 : a1 = 3 * a3)
  (h3 : (a1 + a2 + a3) / 3 = 88) : 
  a1 - a3 = 96 :=
sorry

end number_difference_l551_55117


namespace element_in_set_l551_55124

theorem element_in_set (A : Set ℕ) (h : A = {1, 2}) : 1 ∈ A := 
by 
  rw[h]
  simp

end element_in_set_l551_55124


namespace find_angle_A_l551_55106

theorem find_angle_A (a b : ℝ) (sin_B : ℝ) (ha : a = 3) (hb : b = 4) (hsinB : sin_B = 2/3) :
  ∃ A : ℝ, A = π / 6 :=
by
  sorry

end find_angle_A_l551_55106


namespace side_length_of_square_l551_55160

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l551_55160


namespace sum_of_numbers_equal_16_l551_55121

theorem sum_of_numbers_equal_16 
  (a b c : ℕ) 
  (h1 : a * b = a * c - 1 ∨ a * b = b * c - 1 ∨ a * c = b * c - 1) 
  (h2 : a * b = a * c + 49 ∨ a * b = b * c + 49 ∨ a * c = b * c + 49) :
  a + b + c = 16 :=
sorry

end sum_of_numbers_equal_16_l551_55121


namespace angles_identity_l551_55159
open Real

theorem angles_identity (α β : ℝ) (hα : 0 < α ∧ α < (π / 2)) (hβ : 0 < β ∧ β < (π / 2))
  (h1 : 3 * (sin α)^2 + 2 * (sin β)^2 = 1)
  (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end angles_identity_l551_55159


namespace money_total_l551_55165

theorem money_total (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 350) (h3 : C = 100) : A + B + C = 450 :=
by {
  sorry
}

end money_total_l551_55165


namespace baron_munchausen_is_telling_truth_l551_55168

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_10_digit (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def not_divisible_by_10 (n : ℕ) : Prop :=
  ¬(n % 10 = 0)

theorem baron_munchausen_is_telling_truth :
  ∃ a b : ℕ, a ≠ b ∧ is_10_digit a ∧ is_10_digit b ∧ not_divisible_by_10 a ∧ not_divisible_by_10 b ∧
  (a - digit_sum (a^2) = b - digit_sum (b^2)) := sorry

end baron_munchausen_is_telling_truth_l551_55168


namespace appropriate_chart_for_milk_powder_l551_55141

-- Define the chart requirements and the correctness condition
def ChartType := String
def pie : ChartType := "pie"
def line : ChartType := "line"
def bar : ChartType := "bar"

-- The condition we need for our proof
def representsPercentagesWell (chart: ChartType) : Prop :=
  chart = pie

-- The main theorem statement
theorem appropriate_chart_for_milk_powder : representsPercentagesWell pie :=
by
  sorry

end appropriate_chart_for_milk_powder_l551_55141


namespace voting_proposal_l551_55145

theorem voting_proposal :
  ∀ (T Votes_against Votes_in_favor More_votes_in_favor : ℕ),
    T = 290 →
    Votes_against = (40 * T) / 100 →
    Votes_in_favor = T - Votes_against →
    More_votes_in_favor = Votes_in_favor - Votes_against →
    More_votes_in_favor = 58 :=
by sorry

end voting_proposal_l551_55145


namespace coins_left_l551_55153

-- Define the initial number of coins from each source
def piggy_bank_coins : ℕ := 15
def brother_coins : ℕ := 13
def father_coins : ℕ := 8

-- Define the number of coins given to Laura
def given_to_laura_coins : ℕ := 21

-- Define the total initial coins collected by Kylie
def total_initial_coins : ℕ := piggy_bank_coins + brother_coins + father_coins

-- Lean statement to prove
theorem coins_left : total_initial_coins - given_to_laura_coins = 15 :=
by
  sorry

end coins_left_l551_55153


namespace smallest_part_division_l551_55133

theorem smallest_part_division (S : ℚ) (P1 P2 P3 : ℚ) (total : ℚ) :
  (P1, P2, P3) = (1, 2, 3) →
  total = 64 →
  S = total / (P1 + P2 + P3) →
  S = 10 + 2/3 :=
by
  sorry

end smallest_part_division_l551_55133


namespace chess_tournament_games_l551_55119

theorem chess_tournament_games (P : ℕ) (TotalGames : ℕ) (hP : P = 21) (hTotalGames : TotalGames = 210) : 
  ∃ G : ℕ, G = 20 ∧ TotalGames = (P * (P - 1)) / 2 :=
by
  sorry

end chess_tournament_games_l551_55119


namespace quadratic_inequality_solution_set_l551_55169

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a < 0)
  (h2 : -1 + 2 = b / a) (h3 : -1 * 2 = c / a) :
  (b = a) ∧ (c = -2 * a) :=
by
  sorry

end quadratic_inequality_solution_set_l551_55169


namespace plane_equidistant_from_B_and_C_l551_55167

-- Define points B and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def B : Point3D := { x := 4, y := 1, z := 0 }
def C : Point3D := { x := 2, y := 0, z := 3 }

-- Define the predicate for a plane equation
def plane_eq (a b c d : ℝ) (P : Point3D) : Prop :=
  a * P.x + b * P.y + c * P.z + d = 0

-- The problem statement
theorem plane_equidistant_from_B_and_C :
  ∃ D : ℝ, plane_eq (-2) (-1) 3 D { x := B.x, y := B.y, z := B.z } ∧
            plane_eq (-2) (-1) 3 D { x := C.x, y := C.y, z := C.z } :=
sorry

end plane_equidistant_from_B_and_C_l551_55167


namespace supplement_of_complement_of_65_l551_55156

def complement (angle : ℝ) : ℝ := 90 - angle
def supplement (angle : ℝ) : ℝ := 180 - angle

theorem supplement_of_complement_of_65 : supplement (complement 65) = 155 :=
by
  -- provide the proof steps here
  sorry

end supplement_of_complement_of_65_l551_55156


namespace crafts_sold_l551_55179

theorem crafts_sold (x : ℕ) 
  (h1 : ∃ (n : ℕ), 12 * n = x * 12)
  (h2 : x * 12 + 7 - 18 = 25):
  x = 3 :=
by
  sorry

end crafts_sold_l551_55179


namespace evaluate_expression_l551_55126

theorem evaluate_expression (x c : ℕ) (h1 : x = 3) (h2 : c = 2) : 
  ((x^2 + c)^2 - (x^2 - c)^2) = 72 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l551_55126


namespace evaluate_trig_expression_l551_55104

theorem evaluate_trig_expression (α : ℝ) (h : Real.tan α = -4/3) : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1 / 7 :=
by
  sorry

end evaluate_trig_expression_l551_55104


namespace smallest_integer_inequality_l551_55180

theorem smallest_integer_inequality (x y z : ℝ) : 
  (x^3 + y^3 + z^3)^2 ≤ 3 * (x^6 + y^6 + z^6) ∧ 
  (∃ n : ℤ, (0 < n ∧ n < 3) → ∀ x y z : ℝ, ¬(x^3 + y^3 + z^3)^2 ≤ n * (x^6 + y^6 + z^6)) :=
by
  sorry

end smallest_integer_inequality_l551_55180


namespace vase_net_gain_l551_55199

theorem vase_net_gain 
  (selling_price : ℝ)
  (V1_cost : ℝ)
  (V2_cost : ℝ)
  (hyp1 : selling_price = 2.50)
  (hyp2 : 1.25 * V1_cost = selling_price)
  (hyp3 : 0.85 * V2_cost = selling_price) :
  (selling_price + selling_price) - (V1_cost + V2_cost) = 0.06 := 
by 
  sorry

end vase_net_gain_l551_55199


namespace rectangle_perimeter_is_22_l551_55132

-- Definition of sides of the triangle DEF
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Helper function to compute the area of a right triangle
def triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Ensure the triangle is a right triangle and calculate its area
def area_of_triangle : ℕ :=
  if (side1 * side1 + side2 * side2 = hypotenuse * hypotenuse) then
    triangle_area side1 side2
  else
    0

-- Definition of rectangle's width and equation to find its perimeter
def width : ℕ := 5
def rectangle_length : ℕ := area_of_triangle / width
def perimeter_of_rectangle : ℕ := 2 * (width + rectangle_length)

theorem rectangle_perimeter_is_22 : perimeter_of_rectangle = 22 :=
by
  -- Proof content goes here
  sorry

end rectangle_perimeter_is_22_l551_55132


namespace parallel_lines_m_eq_one_l551_55166

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y + 8 = 0 ∧ (m + 1) * x + y + (m - 2) = 0 → m = 1) :=
by
  intro x y h
  let L1_slope := -2 / m
  let L2_slope := -(m + 1)
  have h_slope : L1_slope = L2_slope := sorry
  have m_positive : m = 1 := sorry
  exact m_positive

end parallel_lines_m_eq_one_l551_55166


namespace symmetry_axis_is_neg_pi_over_12_l551_55155

noncomputable def symmetry_axis_of_sine_function : Prop :=
  ∃ k : ℤ, ∀ x : ℝ, (3 * x + 3 * Real.pi / 4 = Real.pi / 2 + k * Real.pi) ↔ (x = - Real.pi / 12 + k * Real.pi / 3)

theorem symmetry_axis_is_neg_pi_over_12 : symmetry_axis_of_sine_function := sorry

end symmetry_axis_is_neg_pi_over_12_l551_55155


namespace variance_is_4_l551_55100

variable {datapoints : List ℝ}

noncomputable def variance (datapoints : List ℝ) : ℝ :=
  let n := datapoints.length
  let mean := (datapoints.sum / n : ℝ)
  (1 / n : ℝ) * ((datapoints.map (λ x => x ^ 2)).sum - n * mean ^ 2)

theorem variance_is_4 :
  (datapoints.length = 20)
  → ((datapoints.map (λ x => x ^ 2)).sum = 800)
  → (datapoints.sum / 20 = 6)
  → variance datapoints = 4 := by
  intros length_cond sum_squares_cond mean_cond
  sorry

end variance_is_4_l551_55100


namespace tennis_tournament_rounds_l551_55181

/-- Defining the constants and conditions stated in the problem -/
def first_round_games : ℕ := 8
def second_round_games : ℕ := 4
def third_round_games : ℕ := 2
def finals_games : ℕ := 1
def cans_per_game : ℕ := 5
def balls_per_can : ℕ := 3
def total_balls_used : ℕ := 225

/-- Theorem stating the number of rounds in the tennis tournament -/
theorem tennis_tournament_rounds : 
  first_round_games + second_round_games + third_round_games + finals_games = 15 ∧
  15 * cans_per_game = 75 ∧
  75 * balls_per_can = total_balls_used →
  4 = 4 :=
by sorry

end tennis_tournament_rounds_l551_55181


namespace imaginary_part_of_z_l551_55195

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 1 - 3 * I) : z.im = -2 := by
  sorry

end imaginary_part_of_z_l551_55195


namespace find_number_l551_55108

theorem find_number (x : ℝ) : 
  ( ((x - 1.9) * 1.5 + 32) / 2.5 = 20 ) → x = 13.9 :=
by
  sorry

end find_number_l551_55108


namespace trig_identity_l551_55174

noncomputable def sin_40 := Real.sin (40 * Real.pi / 180)
noncomputable def tan_10 := Real.tan (10 * Real.pi / 180)
noncomputable def sqrt_3 := Real.sqrt 3

theorem trig_identity : sin_40 * (tan_10 - sqrt_3) = -1 := by
  sorry

end trig_identity_l551_55174


namespace min_value_of_f_l551_55134

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x - 3 / x

theorem min_value_of_f : ∃ x < 0, ∀ y : ℝ, y = f x → y ≥ 1 + 2 * Real.sqrt 6 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end min_value_of_f_l551_55134


namespace package_weights_l551_55112

theorem package_weights (a b c : ℕ) 
  (h1 : a + b = 108) 
  (h2 : b + c = 132) 
  (h3 : c + a = 138) 
  (h4 : a ≥ 40) 
  (h5 : b ≥ 40) 
  (h6 : c ≥ 40) : 
  a + b + c = 189 :=
sorry

end package_weights_l551_55112


namespace clock_hand_swap_times_l551_55136

noncomputable def time_between_2_and_3 : ℚ := (2 * 143 + 370) / 143
noncomputable def time_between_6_and_7 : ℚ := (6 * 143 + 84) / 143

theorem clock_hand_swap_times :
  time_between_2_and_3 = 2 + 31 * 7 / 143 ∧
  time_between_6_and_7 = 6 + 12 * 84 / 143 :=
by
  -- Math proof will go here
  sorry

end clock_hand_swap_times_l551_55136


namespace second_set_parallel_lines_l551_55107

theorem second_set_parallel_lines (n : ℕ) :
  (5 * (n - 1)) = 280 → n = 71 :=
by
  intros h
  sorry

end second_set_parallel_lines_l551_55107


namespace caleb_double_burgers_count_l551_55109

theorem caleb_double_burgers_count
    (S D : ℕ)
    (cost_single cost_double total_hamburgers total_cost : ℝ)
    (h1 : cost_single = 1.00)
    (h2 : cost_double = 1.50)
    (h3 : total_hamburgers = 50)
    (h4 : total_cost = 66.50)
    (h5 : S + D = total_hamburgers)
    (h6 : cost_single * S + cost_double * D = total_cost) :
    D = 33 := 
sorry

end caleb_double_burgers_count_l551_55109


namespace remainder_when_divided_by_2_l551_55129

-- Define the main parameters
def n : ℕ := sorry  -- n is a positive integer
def k : ℤ := sorry  -- Provided for modular arithmetic context

-- Conditions
axiom h1 : n > 0  -- n is a positive integer
axiom h2 : (n + 1) % 6 = 4  -- When n + 1 is divided by 6, the remainder is 4

-- The theorem statement
theorem remainder_when_divided_by_2 : n % 2 = 1 :=
by
  sorry

end remainder_when_divided_by_2_l551_55129


namespace find_abc_l551_55163

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a * (b + c) = 156)
  (h2 : b * (c + a) = 168)
  (h3 : c * (a + b) = 180) : 
  abc_value a b c = 762 :=
sorry

end find_abc_l551_55163


namespace evaluate_ceiling_sum_l551_55171

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l551_55171


namespace cos_arcsin_l551_55183

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l551_55183


namespace pentagon_area_l551_55135

theorem pentagon_area (a b c d e : ℝ)
  (ht_base ht_height : ℝ)
  (trap_base1 trap_base2 trap_height : ℝ)
  (side_a : a = 17)
  (side_b : b = 22)
  (side_c : c = 30)
  (side_d : d = 26)
  (side_e : e = 22)
  (rt_height : ht_height = 17)
  (rt_base : ht_base = 22)
  (trap_base1_eq : trap_base1 = 26)
  (trap_base2_eq : trap_base2 = 30)
  (trap_height_eq : trap_height = 22)
  : 1/2 * ht_base * ht_height + 1/2 * (trap_base1 + trap_base2) * trap_height = 803 :=
by sorry

end pentagon_area_l551_55135


namespace gcd_21_eq_7_count_l551_55161

theorem gcd_21_eq_7_count : Nat.card {n : Fin 200 // Nat.gcd 21 n = 7} = 19 := 
by
  sorry

end gcd_21_eq_7_count_l551_55161


namespace inequality_proof_l551_55175

theorem inequality_proof (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := 
by
  sorry

end inequality_proof_l551_55175


namespace wage_percent_change_l551_55123

-- Definitions based on given conditions
def initial_wage (W : ℝ) := W
def first_decrease (W : ℝ) := 0.60 * W
def first_increase (W : ℝ) := 0.78 * W
def second_decrease (W : ℝ) := 0.624 * W
def second_increase (W : ℝ) := 0.6864 * W

-- Lean theorem statement to prove overall percent change
theorem wage_percent_change : ∀ (W : ℝ), 
  ((second_increase (second_decrease (first_increase (first_decrease W))) - initial_wage W) / initial_wage W) * 100 = -31.36 :=
by sorry

end wage_percent_change_l551_55123


namespace find_z_l551_55172

theorem find_z (x : ℕ) (z : ℚ) (h1 : x = 103)
               (h2 : x^3 * z - 3 * x^2 * z + 2 * x * z = 208170) 
               : z = 5 / 265 := 
by 
  sorry

end find_z_l551_55172


namespace number_of_multiples_of_six_ending_in_four_and_less_than_800_l551_55102

-- Definitions from conditions
def is_multiple_of_six (n : ℕ) : Prop := n % 6 = 0
def ends_with_four (n : ℕ) : Prop := n % 10 = 4
def less_than_800 (n : ℕ) : Prop := n < 800

-- Theorem to prove
theorem number_of_multiples_of_six_ending_in_four_and_less_than_800 :
  ∃ k : ℕ, k = 26 ∧ ∀ n : ℕ, (is_multiple_of_six n ∧ ends_with_four n ∧ less_than_800 n) → n = 24 + 60 * k ∨ n = 54 + 60 * k :=
sorry

end number_of_multiples_of_six_ending_in_four_and_less_than_800_l551_55102


namespace necessary_but_not_sufficient_l551_55176

def simple_prop (p q : Prop) :=
  (¬ (p ∧ q)) → (¬ (p ∨ q))

theorem necessary_but_not_sufficient (p q : Prop) (h : simple_prop p q) :
  ((¬ (p ∧ q)) → (¬ (p ∨ q))) ∧ ¬ ((¬ (p ∨ q)) → (¬ (p ∧ q))) := by
sorry

end necessary_but_not_sufficient_l551_55176


namespace find_red_cards_l551_55139

-- We use noncomputable here as we are dealing with real numbers in a theoretical proof context.
noncomputable def red_cards (r b : ℕ) (_initial_prob : r / (r + b) = 1 / 5) 
                            (_added_prob : r / (r + b + 6) = 1 / 7) : ℕ := 
r

theorem find_red_cards 
  {r b : ℕ}
  (h1 : r / (r + b) = 1 / 5)
  (h2 : r / (r + b + 6) = 1 / 7) : 
  red_cards r b h1 h2 = 3 :=
sorry  -- Proof not required

end find_red_cards_l551_55139


namespace price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l551_55125

def cost_price : ℝ := 40
def initial_price : ℝ := 60
def initial_sales_volume : ℕ := 300
def sales_decrease_rate (x : ℕ) : ℕ := 10 * x
def sales_increase_rate (a : ℕ) : ℕ := 20 * a

noncomputable def price_increase_proft_relation (x : ℕ) : ℝ :=
  -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000

theorem price_increase_profit_relation_proof (x : ℕ) (h : 0 ≤ x ∧ x ≤ 30) :
  price_increase_proft_relation x = -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000 := sorry

noncomputable def price_decrease_profit_relation (a : ℕ) : ℝ :=
  -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000

theorem price_decrease_profit_relation_proof (a : ℕ) :
  price_decrease_profit_relation a = -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000 := sorry

theorem max_profit_price_increase :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧ price_increase_proft_relation x = 6250 := sorry

end price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l551_55125


namespace simplify_expression_l551_55198

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  2 * (1 - (2 * (1 - (1 + (2 * (1 - x)))))) = 8 * x - 10 := 
by sorry

end simplify_expression_l551_55198


namespace find_x_such_that_l551_55190

theorem find_x_such_that {x : ℝ} (h : ⌈x⌉ * x + 15 = 210) : x = 195 / 14 :=
by
  sorry

end find_x_such_that_l551_55190


namespace x_intercept_of_line_l551_55152

theorem x_intercept_of_line (x y : ℚ) (h : 4 * x + 6 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by
  sorry

end x_intercept_of_line_l551_55152


namespace incorrect_statement_A_l551_55137

-- We need to prove that statement (A) is incorrect given the provided conditions.

theorem incorrect_statement_A :
  ¬(∀ (a b : ℝ), a > b → ∀ (c : ℝ), c < 0 → a * c > b * c ∧ a / c > b / c) := 
sorry

end incorrect_statement_A_l551_55137


namespace complement_intersection_l551_55185

def A : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}
def B : Set ℝ := {x | x > 7}

theorem complement_intersection :
  (Set.univ \ A) ∩ B = {x | x > 7} :=
by
  sorry

end complement_intersection_l551_55185


namespace compute_fraction_pow_mul_l551_55188

theorem compute_fraction_pow_mul :
  8 * (2 / 3)^4 = 128 / 81 :=
by 
  sorry

end compute_fraction_pow_mul_l551_55188


namespace find_number_l551_55173

theorem find_number (x n : ℝ) (h1 : x > 0) (h2 : x / 50 + x / n = 0.06 * x) : n = 25 :=
by
  sorry

end find_number_l551_55173


namespace jame_annual_earnings_difference_l551_55142

-- Define conditions
def new_hourly_wage := 20
def new_hours_per_week := 40
def old_hourly_wage := 16
def old_hours_per_week := 25
def weeks_per_year := 52

-- Define annual earnings calculations
def annual_earnings_old (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

def annual_earnings_new (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

-- Problem statement to prove
theorem jame_annual_earnings_difference :
  annual_earnings_new new_hourly_wage new_hours_per_week weeks_per_year -
  annual_earnings_old old_hourly_wage old_hours_per_week weeks_per_year = 20800 := by
  sorry

end jame_annual_earnings_difference_l551_55142


namespace derivative_at_1_l551_55196

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_1 : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end derivative_at_1_l551_55196


namespace greatest_int_value_not_satisfy_condition_l551_55154

/--
For the inequality 8 - 6x > 26, the greatest integer value 
of x that satisfies this is -4.
-/
theorem greatest_int_value (x : ℤ) : 8 - 6 * x > 26 → x ≤ -4 :=
by sorry

theorem not_satisfy_condition (x : ℤ) : x > -4 → ¬ (8 - 6 * x > 26) :=
by sorry

end greatest_int_value_not_satisfy_condition_l551_55154


namespace seeds_in_bucket_A_l551_55177

theorem seeds_in_bucket_A (A B C : ℕ) (h_total : A + B + C = 100) (h_B : B = 30) (h_C : C = 30) : A = 40 :=
by
  sorry

end seeds_in_bucket_A_l551_55177


namespace angle_line_plane_l551_55120

theorem angle_line_plane {l α : Type} (θ : ℝ) (h : θ = 150) : 
  ∃ φ : ℝ, φ = 60 := 
by
  -- This part would require the actual proof.
  sorry

end angle_line_plane_l551_55120


namespace remainder_of_x_mod_10_l551_55143

def x : ℕ := 2007 ^ 2008

theorem remainder_of_x_mod_10 : x % 10 = 1 := by
  sorry

end remainder_of_x_mod_10_l551_55143


namespace lemonade_calories_l551_55146

theorem lemonade_calories 
    (lime_juice_weight : ℕ)
    (lime_juice_calories_per_grams : ℕ)
    (sugar_weight : ℕ)
    (sugar_calories_per_grams : ℕ)
    (water_weight : ℕ)
    (water_calories_per_grams : ℕ)
    (mint_weight : ℕ)
    (mint_calories_per_grams : ℕ)
    :
    lime_juice_weight = 150 →
    lime_juice_calories_per_grams = 30 →
    sugar_weight = 200 →
    sugar_calories_per_grams = 390 →
    water_weight = 500 →
    water_calories_per_grams = 0 →
    mint_weight = 50 →
    mint_calories_per_grams = 7 →
    (300 * ((150 * 30 + 200 * 390 + 500 * 0 + 50 * 7) / 900) = 276) :=
by
  sorry

end lemonade_calories_l551_55146


namespace probability_same_length_l551_55116

/-- Defining the set of all sides and diagonals of a regular hexagon. -/
def T : Finset ℚ := sorry

/-- There are exactly 6 sides in the set T. -/
def sides_count : ℕ := 6

/-- There are exactly 9 diagonals in the set T. -/
def diagonals_count : ℕ := 9

/-- The total number of segments in the set T. -/
def total_segments : ℕ := sides_count + diagonals_count

theorem probability_same_length :
  let prob_side := (6 : ℚ) / total_segments * (5 / (total_segments - 1))
  let prob_diagonal := (9 : ℚ) / total_segments * (4 / (total_segments - 1))
  prob_side + prob_diagonal = 17 / 35 := 
by
  admit

end probability_same_length_l551_55116


namespace chloe_candies_l551_55115

-- Definitions for the conditions
def lindaCandies : ℕ := 34
def totalCandies : ℕ := 62

-- The statement to prove
theorem chloe_candies :
  (totalCandies - lindaCandies) = 28 :=
by
  -- Proof would go here
  sorry

end chloe_candies_l551_55115


namespace triangle_side_length_l551_55128

-- Definitions based on problem conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

variables (AC BC AD AB CD : ℝ)

-- Conditions from the problem
axiom h1 : BC = 2 * AC
axiom h2 : AD = (1 / 3) * AB

-- Theorem statement to be proved
theorem triangle_side_length (h1 : BC = 2 * AC) (h2 : AD = (1 / 3) * AB) : CD = 2 * AD :=
sorry

end triangle_side_length_l551_55128


namespace problem_inequality_l551_55118

variable {a b : ℝ}

theorem problem_inequality 
  (h_a_nonzero : a ≠ 0) 
  (h_b_nonzero : b ≠ 0)
  (h_a_gt_b : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) := 
by 
  sorry

end problem_inequality_l551_55118


namespace actual_height_of_boy_l551_55158

variable (wrong_height : ℕ) (boys : ℕ) (wrong_avg correct_avg : ℕ)
variable (x : ℕ)

-- Given conditions
def conditions 
:= boys = 35 ∧
   wrong_height = 166 ∧
   wrong_avg = 185 ∧
   correct_avg = 183

-- Question: Proving the actual height
theorem actual_height_of_boy (h : conditions boys wrong_height wrong_avg correct_avg) : 
  x = wrong_height + (boys * wrong_avg - boys * correct_avg) := 
  sorry

end actual_height_of_boy_l551_55158


namespace find_fraction_increase_l551_55147

noncomputable def present_value : ℝ := 64000
noncomputable def value_after_two_years : ℝ := 87111.11111111112

theorem find_fraction_increase (f : ℝ) :
  64000 * (1 + f) ^ 2 = 87111.11111111112 → f = 0.1666666666666667 := 
by
  intro h
  -- proof steps here
  sorry

end find_fraction_increase_l551_55147


namespace max_area_rectangular_playground_l551_55150

theorem max_area_rectangular_playground (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 360) 
  (h_length : l ≥ 90) 
  (h_width : w ≥ 50) : 
  (l * w) ≤ 8100 :=
by
  sorry

end max_area_rectangular_playground_l551_55150


namespace not_symmetric_about_point_l551_55164

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) + Real.log (4 - x)

theorem not_symmetric_about_point : ¬ (∀ h : ℝ, f (1 + h) = f (1 - h)) :=
by
  sorry

end not_symmetric_about_point_l551_55164


namespace distance_probability_at_least_sqrt2_over_2_l551_55113

noncomputable def prob_dist_at_least : ℝ := 
  let T := ((0,0), (1,0), (0,1))
  -- Assumes conditions incorporated through identifying two random points within the triangle T.
  let area_T : ℝ := 0.5
  let valid_area : ℝ := 0.5 - (Real.pi * (Real.sqrt 2 / 2)^2 / 8 + ((Real.sqrt 2 / 2)^2 / 2) / 2)
  valid_area / area_T

theorem distance_probability_at_least_sqrt2_over_2 :
  prob_dist_at_least = (4 - π) / 8 :=
by
  sorry

end distance_probability_at_least_sqrt2_over_2_l551_55113


namespace military_unit_soldiers_l551_55131

theorem military_unit_soldiers:
  ∃ (x N : ℕ), 
      (N = x * (x + 5)) ∧
      (N = 5 * (x + 845)) ∧
      N = 4550 :=
by
  sorry

end military_unit_soldiers_l551_55131


namespace quadrant_of_point_C_l551_55197

theorem quadrant_of_point_C
  (a b : ℝ)
  (h1 : -(a-2) = -1)
  (h2 : b+5 = 3) :
  a = 3 ∧ b = -2 ∧ 0 < a ∧ b < 0 :=
by {
  sorry
}

end quadrant_of_point_C_l551_55197


namespace sum_a5_a6_a7_l551_55114

def S (n : ℕ) : ℕ :=
  n^2 + 2 * n + 5

theorem sum_a5_a6_a7 : S 7 - S 4 = 39 :=
  by sorry

end sum_a5_a6_a7_l551_55114


namespace axis_of_symmetry_sine_function_l551_55189

theorem axis_of_symmetry_sine_function :
  ∃ k : ℤ, x = k * (π / 2) := sorry

end axis_of_symmetry_sine_function_l551_55189


namespace ali_spending_ratio_l551_55151

theorem ali_spending_ratio
  (initial_amount : ℝ := 480)
  (remaining_amount : ℝ := 160)
  (F : ℝ)
  (H1 : (initial_amount - F - (1/3) * (initial_amount - F) = remaining_amount))
  : (F / initial_amount) = 1 / 2 :=
by
  sorry

end ali_spending_ratio_l551_55151


namespace ball_bounces_less_than_two_meters_l551_55162

theorem ball_bounces_less_than_two_meters : ∀ k : ℕ, 500 * (1/3 : ℝ)^k < 2 → k ≥ 6 := by
  sorry

end ball_bounces_less_than_two_meters_l551_55162
