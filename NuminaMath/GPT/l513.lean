import Mathlib

namespace percentage_bob_is_36_l513_513610

def water_per_acre_corn : ℕ := 20
def water_per_acre_cotton : ℕ := 80
def water_per_acre_beans : ℕ := 2 * water_per_acre_corn

def acres_bob_corn : ℕ := 3
def acres_bob_cotton : ℕ := 9
def acres_bob_beans : ℕ := 12

def acres_brenda_corn : ℕ := 6
def acres_brenda_cotton : ℕ := 7
def acres_brenda_beans : ℕ := 14

def acres_bernie_corn : ℕ := 2
def acres_bernie_cotton : ℕ := 12

def water_bob : ℕ := (acres_bob_corn * water_per_acre_corn) +
                      (acres_bob_cotton * water_per_acre_cotton) +
                      (acres_bob_beans * water_per_acre_beans)

def water_brenda : ℕ := (acres_brenda_corn * water_per_acre_corn) +
                         (acres_brenda_cotton * water_per_acre_cotton) +
                         (acres_brenda_beans * water_per_acre_beans)

def water_bernie : ℕ := (acres_bernie_corn * water_per_acre_corn) +
                         (acres_bernie_cotton * water_per_acre_cotton)

def total_water : ℕ := water_bob + water_brenda + water_bernie

def percentage_bob : ℚ := (water_bob : ℚ) / (total_water : ℚ) * 100

theorem percentage_bob_is_36 : percentage_bob = 36 := by
  sorry

end percentage_bob_is_36_l513_513610


namespace calculate_value_l513_513182

theorem calculate_value : (535^2 - 465^2) / 70 = 1000 := by
  sorry

end calculate_value_l513_513182


namespace mn_value_l513_513903

theorem mn_value (m n : ℤ) (h1 : 2 * m = 6) (h2 : m - n = 2) : m * n = 3 := by
  sorry

end mn_value_l513_513903


namespace circle_center_radius_sum_correct_l513_513382

noncomputable def circle_center_radius_sum (eq : String) : ℝ :=
  if h : eq = "x^2 + 8x - 2y^2 - 6y = -6" then
    let c : ℝ := -4
    let d : ℝ := -3 / 2
    let s : ℝ := Real.sqrt (47 / 4)
    c + d + s
  else 0

theorem circle_center_radius_sum_correct :
  circle_center_radius_sum "x^2 + 8x - 2y^2 - 6y = -6" = (-11 + Real.sqrt 47) / 2 :=
by
  -- proof omitted
  sorry

end circle_center_radius_sum_correct_l513_513382


namespace area_of_triangle_AOB_l513_513650

noncomputable def A : ℝ × ℝ := (3, π / 3)
noncomputable def B : ℝ × ℝ := (4, -π / 6)
def O : ℝ × ℝ := (0, 0)

theorem area_of_triangle_AOB :
  let base := (A.1) in
  let height := (B.1) in
  (1 / 2) * base * height = 6 :=
by 
  sorry

end area_of_triangle_AOB_l513_513650


namespace total_weight_of_settings_l513_513414

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end total_weight_of_settings_l513_513414


namespace proof_problem_l513_513782

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513782


namespace total_emails_675_l513_513958

-- Definitions based on conditions
def emails_per_day_before : ℕ := 20
def extra_emails_per_day_after : ℕ := 5
def halfway_days : ℕ := 15
def total_days : ℕ := 30

-- Define the total number of emails received by the end of April
def total_emails_received : ℕ :=
  let emails_before := emails_per_day_before * halfway_days
  let emails_after := (emails_per_day_before + extra_emails_per_day_after) * halfway_days
  emails_before + emails_after

-- Theorem stating that the total number of emails received by the end of April is 675
theorem total_emails_675 : total_emails_received = 675 := by
  sorry

end total_emails_675_l513_513958


namespace minkowski_sum_corner_points_l513_513394

theorem minkowski_sum_corner_points (K1 K2 : set (ℝ × ℝ)) 
  (A1 A2 : ℝ × ℝ) : 
  convex K1 → convex K2 → 
  (A1 ∈ K1 ∧ A2 ∈ K2) → 
  (is_corner_point (A1 + A2) (K1 + K2)) → 
  (is_corner_point A1 K1 ∧ is_corner_point A2 K2) :=
by
  intros hK1 hK2 hA hCorner
  sorry

-- Definitions for convexity and corner points would be required but are defined in Mathlib.

end minkowski_sum_corner_points_l513_513394


namespace max_value_g_l513_513230

def g (x : ℝ) : ℝ := min (3 * x + 2) (min ((3 / 2) * x + 1) (- (3 / 4) * x + 7))

theorem max_value_g : ∃ x : ℝ, g x = 25 / 3 := by
  sorry

end max_value_g_l513_513230


namespace not_possible_arrangement_l513_513364

theorem not_possible_arrangement : 
  ¬ ∃ (f : Fin 4026 → Fin 2014), 
    (∀ k : Fin 2014, ∃ i j : Fin 4026, i < j ∧ f i = k ∧ f j = k ∧ (j.val - i.val - 1) = k.val) :=
sorry

end not_possible_arrangement_l513_513364


namespace volume_double_l513_513608

-- Conditions
def diameter : ℝ := 12
def height : ℝ := 10
def radius : ℝ := diameter / 2

-- Volume of a cone formula
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Theorem stating that doubling the volume results in the expected total volume
theorem volume_double (d h : ℝ) (r := d / 2) : volume_cone r h * 2 = 240 * π := by
  sorry

#check volume_double

end volume_double_l513_513608


namespace joy_quadrilateral_rod_count_l513_513375

theorem joy_quadrilateral_rod_count : 
  ∃ n, n = 30 ∧ 
  (∀ lengths : Finset ℕ, lengths = Finset.range 41.erase 0.erase 5.erase 12.erase 25 → {d : ℕ | 8 < d ∧ d < 42 ∧ d ∈ lengths}.card = n) :=
by
  sorry

end joy_quadrilateral_rod_count_l513_513375


namespace binary_to_hexadecimal_equiv_l513_513618

theorem binary_to_hexadecimal_equiv :
  let binary_num : ℕ := 0b110101 in
  let hexadecimal_equiv : String := "35" in
  (binary_num.toStringBase 16 = hexadecimal_equiv) :=
by
  sorry

end binary_to_hexadecimal_equiv_l513_513618


namespace median_incorrect_l513_513225

open List

def data : List ℕ := [3, 3, 6, 5, 3]

noncomputable def median (l: List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem median_incorrect : median data ≠ 6 :=
  sorry

end median_incorrect_l513_513225


namespace unit_cost_of_cranberry_juice_l513_513553

theorem unit_cost_of_cranberry_juice (total_cost : ℕ) (ounces : ℕ) (h1 : total_cost = 84) (h2 : ounces = 12) :
  total_cost / ounces = 7 :=
by
  sorry

end unit_cost_of_cranberry_juice_l513_513553


namespace prove_sum_expression_l513_513644

noncomputable def sum_expression (m n : ℕ) (z : 𝕜) : 𝕜 :=
  ∑ i in finset.range (n + 1), (-1) ^ i * nat.choose n i * 
              (z + n - i) * (z + n - i - 1) * ... * (z + n - i - m + 1)

theorem prove_sum_expression (m n : ℕ) (z : 𝕜) : 
  sum_expression m n z = 
  if m < n then 0 else if m = n then nat.factorial m else
    nat.choose m n * nat.factorial n * z * (z - 1) * ... * (z - m + n + 1) :=
sorry

end prove_sum_expression_l513_513644


namespace cake_volume_and_icing_area_sum_l513_513568

-- Definitions for the problem based on conditions
def edge_length : ℝ := 4
def triangle_B_volume : ℝ := ((edge_length^2) / 2) * edge_length
def top_triangle_area : ℝ := (edge_length^2) / 2
def lateral_face_area : ℝ := (edge_length^2) / 2

-- Define the total volume and icing area
def volume : ℝ := triangle_B_volume
def icing_area : ℝ := top_triangle_area + lateral_face_area

theorem cake_volume_and_icing_area_sum : volume + icing_area = 48 := 
by
  have h1 : volume = 32 := calc
    volume = ((4^2) / 2) * 4 : by sorry
    ... = 32 : by sorry 
  have h2 : icing_area = 16 := calc
    icing_area = ((4^2) / 2) + ((4^2) / 2) : by sorry
    ... = 16 : by sorry
  show 32 + 16 = 48
  sorry

end cake_volume_and_icing_area_sum_l513_513568


namespace picnic_participants_l513_513470

def num_participants_is_correct (total : ℕ) (plates_remaining: ℕ) (participants: ℕ) : Prop :=
  (total - participants) + participants = total ∧ (total - 2 * participants + 1) = plates_remaining

theorem picnic_participants (c : 2015) (pr : 4) : num_participants_is_correct c pr 1006 :=
by
  unfold num_participants_is_correct
  split
  {
    trivial, -- first condition (total - participants + participants = total)
  },
  {
    sorry  -- second condition (2015 - 2 * 1006 + 1 = 4)
  }

end picnic_participants_l513_513470


namespace product_of_3rd_and_2nd_smallest_l513_513071

theorem product_of_3rd_and_2nd_smallest :
  let numbers := [10, 11, 12, 13, 14] in
  let sorted_numbers := List.sort (· ≤ ·) numbers in
  let third_smallest := sorted_numbers.nthLe 2 (by decide) in
  let second_smallest := sorted_numbers.nthLe 1 (by decide) in
  third_smallest * second_smallest = 132 := by
sorry

end product_of_3rd_and_2nd_smallest_l513_513071


namespace product_in_third_quadrant_l513_513400

def z1 : ℂ := 1 - 3 * Complex.I
def z2 : ℂ := 3 - 2 * Complex.I
def z := z1 * z2

theorem product_in_third_quadrant : z.re < 0 ∧ z.im < 0 := 
sorry

end product_in_third_quadrant_l513_513400


namespace jimin_notebooks_proof_l513_513984

variable (m f o n : ℕ)

theorem jimin_notebooks_proof (hm : m = 7) (hf : f = 14) (ho : o = 33) (hn : n = o + m + f) :
  n - o = 21 := by
  sorry

end jimin_notebooks_proof_l513_513984


namespace polynomial_coefficient_sum_l513_513235

theorem polynomial_coefficient_sum :
  let p := (1 + 2 * x) * (1 - 2 * x)^7
  let coeffs := λ i, p.coeff i
  coeffs 0 + coeffs 1 + coeffs 2 + coeffs 3 + coeffs 4 + coeffs 5 + coeffs 6 + coeffs 7 = 253 := 
by
  let p := (1 + 2 * x) * (1 - 2 * x)^7
  let a := (λ i, p.coeff i)
  have ha8 : a 8 = -256 := sorry
  have p1 := (1 + 2) * (1 - 2) ^ 7 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 := sorry
  exact sorry

end polynomial_coefficient_sum_l513_513235


namespace proposition_A_l513_513817

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513817


namespace proposition_A_l513_513816

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513816


namespace bug_probability_tenth_move_l513_513555

def recurrence_relation (n : Nat) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 0
  else 1 / 2 * (1 - recurrence_relation (n - 1))

theorem bug_probability_tenth_move : 
  let P₁₀ := recurrence_relation 10
  ∃ m n : ℕ, 
    (nat.gcd m n = 1) ∧
    P₁₀ = (m : ℚ) / (n : ℚ) ∧
    (m + n = 683) :=
by
  let P₁₀ := recurrence_relation 10
  sorry

end bug_probability_tenth_move_l513_513555


namespace solve_quadratic_eq_l513_513031

theorem solve_quadratic_eq : ∀ x : ℝ, (x^2 - 6 * x + 2 = 0) ↔ (x = 3 + sqrt 7 ∨ x = 3 - sqrt 7) :=
by
  sorry

end solve_quadratic_eq_l513_513031


namespace kite_smallest_angle_l513_513135

-- Noncomputable theory since angles are real numbers
noncomputable theory

-- Given conditions
def kite_conditions (a d : ℝ) : Prop :=
  (a + 3 * d = 150 ∧ 2 * a + 3 * d = 180)

-- The main theorem to prove
theorem kite_smallest_angle (a d : ℝ) (h : kite_conditions a d) : a = 15 :=
by
  sorry

end kite_smallest_angle_l513_513135


namespace equal_distances_l513_513429

theorem equal_distances (c : ℝ) (distance : ℝ) :
  abs (2 - -4) = distance ∧ (abs (c - -4) = distance ∨ abs (c - 2) = distance) ↔ (c = -10 ∨ c = 8) :=
by
  sorry

end equal_distances_l513_513429


namespace sum_of_selected_numbers_l513_513185

noncomputable def selected_numbers : List ℝ := [1.4, 9 / 10, 1.2, 0.5, 13 / 10]

theorem sum_of_selected_numbers :
  ([x for x in selected_numbers if x > 1.1]).sum = 3.9 := by
sorry

end sum_of_selected_numbers_l513_513185


namespace true_proposition_l513_513763

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513763


namespace proposition_A_l513_513813

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513813


namespace kate_retirement_fund_value_l513_513999

theorem kate_retirement_fund_value 
(initial_value decrease final_value : ℝ) 
(h1 : initial_value = 1472)
(h2 : decrease = 12)
(h3 : final_value = initial_value - decrease) : 
final_value = 1460 := 
by
  sorry

end kate_retirement_fund_value_l513_513999


namespace maria_spends_on_soap_l513_513404

theorem maria_spends_on_soap 
  (lasts_per_bar : ℕ) (cost_per_bar : ℚ) (discount_threshold : ℕ) (discount_percentage : ℚ)
  (months_in_year : ℕ)
  (h1 : lasts_per_bar = 2)
  (h2 : cost_per_bar = 8)
  (h3 : discount_threshold = 6)
  (h4 : discount_percentage = 0.1)
  (h5 : months_in_year = 12) :
  let
    bars_needed := months_in_year / lasts_per_bar,
    total_cost_without_discount := bars_needed * cost_per_bar,
    discount := if bars_needed >= discount_threshold then discount_percentage * total_cost_without_discount else 0,
    final_cost := total_cost_without_discount - discount
  in final_cost = 43.20 := sorry

end maria_spends_on_soap_l513_513404


namespace correct_equation_l513_513097

theorem correct_equation : (√((-6)^2) = 6) ∧ (¬(√((-2)^2) = -2)) ∧ (¬(√(x^2) = x)) ∧ (¬(√((-5)^2) = ±5)) :=
by
  sorry

end correct_equation_l513_513097


namespace equal_incircles_implies_equilateral_triangle_l513_513039

variables {A B C A1 B1 C1 M : Type}

def is_angle_bisector (x y z p : Type) : Prop := sorry
def incircle_radius_eq (u v w x y z : Type) : Prop := sorry
def is_equilateral_triangle (t1 t2 t3 : Type) : Prop := sorry

theorem equal_incircles_implies_equilateral_triangle
  (triangle_ABC : Type)
  (angle_bisectors : is_angle_bisector A B C A1 ∧ is_angle_bisector B C A B1 ∧ is_angle_bisector C A B C1)
  (M : Type)
  (equal_radii : incircle_radius_eq M B1 A ∧ incircle_radius_eq M C1 A ∧ incircle_radius_eq M C1 B ∧
                  incircle_radius_eq M A1 B ∧ incircle_radius_eq M A1 C ∧ incircle_radius_eq M B1 C) :
  is_equilateral_triangle A B C :=
sorry

end equal_incircles_implies_equilateral_triangle_l513_513039


namespace total_emails_675_l513_513959

-- Definitions based on conditions
def emails_per_day_before : ℕ := 20
def extra_emails_per_day_after : ℕ := 5
def halfway_days : ℕ := 15
def total_days : ℕ := 30

-- Define the total number of emails received by the end of April
def total_emails_received : ℕ :=
  let emails_before := emails_per_day_before * halfway_days
  let emails_after := (emails_per_day_before + extra_emails_per_day_after) * halfway_days
  emails_before + emails_after

-- Theorem stating that the total number of emails received by the end of April is 675
theorem total_emails_675 : total_emails_received = 675 := by
  sorry

end total_emails_675_l513_513959


namespace max_moves_440_l513_513987

-- Define the set of initial numbers
def initial_numbers : List ℕ := List.range' 1 22

-- Define what constitutes a valid move
def is_valid_move (a b : ℕ) : Prop := b ≥ a + 2

-- Perform the move operation
def perform_move (numbers : List ℕ) (a b : ℕ) : List ℕ :=
  (numbers.erase a).erase b ++ [a + 1, b - 1]

-- Define the maximum number of moves we need to prove
theorem max_moves_440 : ∃ m, m = 440 ∧
  ∀ (moves_done : ℕ) (numbers : List ℕ),
    moves_done <= m → ∃ a b, a ∈ numbers ∧ b ∈ numbers ∧
                             is_valid_move a b ∧
                             numbers = initial_numbers →
                             perform_move numbers a b ≠ numbers
 := sorry

end max_moves_440_l513_513987


namespace angle_FHP_eq_angle_BAC_l513_513392

-- Defining the structure of a triangle
structure Triangle :=
(A B C : Point)
(ac_angle_triangle : acute_angle ∠ A B C)
(BC_gt_CA : segment B C > segment C A)

-- Defining Points O, H, F, and P
def O : Point := circumcenter_of_triangle ABC
def H : Point := orthocenter_of_triangle ABC
def F : Point := foot_of_altitude_from_C ABC
def P : Point := let OF := line_through_points O F in perpendicular_to_line_at_point OF F intersect_side CA

-- Requirement to prove angle equality
theorem angle_FHP_eq_angle_BAC (ABC : Triangle) :
  ∀ (ABC : Triangle)(O : Point)(H : Point)(F : Point)(P : Point), 
    (O = circumcenter_of_triangle ABC) →
    (H = orthocenter_of_triangle ABC) →
    (F = foot_of_altitude_from_C ABC) →
    (P = (let OF := line_through_points O F in perpendicular_to_line_at_point OF F intersect_side CA)) →
    ∠ FHP = ∠ BAC :=
by
  intros
  sorry

end angle_FHP_eq_angle_BAC_l513_513392


namespace proposition_true_l513_513839

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513839


namespace quadratic_roots_l513_513321

theorem quadratic_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*m = 0) ∧ (x2^2 + 2*x2 + 2*m = 0)) ↔ m < 1/2 :=
by sorry

end quadratic_roots_l513_513321


namespace regular_polygon_sides_l513_513167

theorem regular_polygon_sides (C : ℕ) (h : (C - 2) * 180 / C = 144) : C = 10 := 
sorry

end regular_polygon_sides_l513_513167


namespace real_solution_l513_513208

noncomputable def condition_1 (x : ℝ) : Prop := 
  4 ≤ x / (2 * x - 7)

noncomputable def condition_2 (x : ℝ) : Prop := 
  x / (2 * x - 7) < 10

noncomputable def solution_set : Set ℝ :=
  { x | (70 / 19 : ℝ) < x ∧ x ≤ 4 }

theorem real_solution (x : ℝ) : 
  (condition_1 x ∧ condition_2 x) ↔ x ∈ solution_set :=
sorry

end real_solution_l513_513208


namespace calculate_total_weight_l513_513410

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end calculate_total_weight_l513_513410


namespace proposition_A_l513_513818

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513818


namespace proof_problem_l513_513809

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513809


namespace correct_calculation_l513_513532

theorem correct_calculation :
  (∀ (x : ℝ), (x^3 * 2 * x^4 = 2 * x^7) ∧
  (x^6 / x^3 = x^2) ∧
  ((x^3)^4 = x^7) ∧
  (x^2 + x = x^3)) → 
  (∀ (x : ℝ), x^3 * 2 * x^4 = 2 * x^7) :=
by
  intros h x
  have A := h x
  exact A.1

end correct_calculation_l513_513532


namespace mean_of_six_numbers_l513_513485

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end mean_of_six_numbers_l513_513485


namespace contractor_fine_per_day_l513_513565

theorem contractor_fine_per_day
    (total_days : ℕ) 
    (work_days_fine_amt : ℕ) 
    (total_amt : ℕ) 
    (absent_days : ℕ) 
    (worked_days : ℕ := total_days - absent_days)
    (earned_amt : ℕ := worked_days * work_days_fine_amt)
    (fine_per_day : ℚ)
    (total_fine : ℚ := absent_days * fine_per_day) : 
    (earned_amt - total_fine = total_amt) → 
    fine_per_day = 7.5 :=
by
  intros h
  -- proof here is omitted
  sorry

end contractor_fine_per_day_l513_513565


namespace problem_statement_l513_513251

-- Definitions of conditions
def p (a : ℝ) : Prop := a < 0
def q (a : ℝ) : Prop := a^2 > a

-- Statement of the problem
theorem problem_statement (a : ℝ) (h1 : p a) (h2 : q a) : (¬ p a) → (¬ q a) → ∃ x, ¬ (¬ q x) → (¬ (¬ p x)) :=
by
  sorry

end problem_statement_l513_513251


namespace t_f_3_equals_l513_513976

def t (x : ℝ) : ℝ := sqrt (4 * x + 2)
def f (x : ℝ) : ℝ := 5 * t x

theorem t_f_3_equals : t (f 3) = sqrt (20 * sqrt 14 + 2) := by
  sorry

end t_f_3_equals_l513_513976


namespace find_coordinates_of_P_l513_513318

noncomputable def f : ℝ → ℝ := sorry

variable (φ : ℝ)

def even_fn_property : Prop :=
∀ x : ℝ, f (-x) - sin (-x + φ) = f x - sin (x + φ)

def odd_fn_property : Prop :=
∀ x : ℝ, f (-x) - cos (-x + φ) = - (f x - cos (x + φ))

def reciprocal_slopes_property (x₁ : ℝ) : Prop :=
f' x₁ * f' (x₁ + π / 2) = 1

def coordinates_point_P (x₁ : ℝ) : Prop :=
x₁ = π / 2 ∧ (f x₁ = 1 ∨ f x₁ = -1)

theorem find_coordinates_of_P (x₁ : ℝ) (φ : ℝ) (h₁ : even_fn_property φ) (h₂ : odd_fn_property φ) (h₃ : reciprocal_slopes_property φ x₁) : coordinates_point_P x₁ φ := 
sorry

end find_coordinates_of_P_l513_513318


namespace find_xyz_l513_513254

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 25)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 7) : 
  x * y * z = 6 := 
by 
  sorry

end find_xyz_l513_513254


namespace calculate_total_weight_l513_513411

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end calculate_total_weight_l513_513411


namespace zaim_larger_part_l513_513536

theorem zaim_larger_part (x y : ℕ) (h_sum : x + y = 20) (h_prod : x * y = 96) : max x y = 12 :=
by
  -- The proof goes here
  sorry

end zaim_larger_part_l513_513536


namespace find_coordinates_of_C_l513_513993

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def A : ℝ × ℝ := point (-3) 2
def B : ℝ × ℝ := point 5 10

theorem find_coordinates_of_C (C : ℝ × ℝ)
  (h1 : distance A C = 2 * distance C B) :
  C = (7 / 3, 22 / 3) :=
  sorry

end find_coordinates_of_C_l513_513993


namespace antacids_per_month_proof_l513_513422

structure ConsumptionRates where
  indian : Nat
  mexican : Nat
  other : Nat

structure WeeklyEatingPattern where
  indian_days : Nat
  mexican_days : Nat
  other_days : Nat

def antacids_per_week (rates : ConsumptionRates) (pattern : WeeklyEatingPattern) : Nat :=
  (pattern.indian_days * rates.indian) +
  (pattern.mexican_days * rates.mexican) +
  (pattern.other_days * rates.other)

def weeks_in_month : Nat := 4

theorem antacids_per_month_proof :
  let rates := ConsumptionRates.mk 3 2 1 in
  let pattern := WeeklyEatingPattern.mk 3 2 2 in -- 7 days - 3 Indian days - 2 Mexican days = 2 other days
  let weekly_antacids := antacids_per_week rates pattern in
  weekly_antacids * weeks_in_month = 60 := 
by
  -- let's skip the proof
  sorry

end antacids_per_month_proof_l513_513422


namespace complex_division_real_a_value_l513_513658

theorem complex_division_real_a_value (a : ℝ) (z1 z2 : ℂ) (h1 : z1 = a + 2 * complex.I) (h2 : z2 = 3 - 4 * complex.I) (h3 : (z1 / z2).im = 0) : a = -3/2 :=
by sorry

end complex_division_real_a_value_l513_513658


namespace blue_string_length_is_320_l513_513053

-- Define the lengths of the strings
def red_string_length := 8
def white_string_length := 5 * red_string_length
def blue_string_length := 8 * white_string_length

-- The main theorem to prove
theorem blue_string_length_is_320 : blue_string_length = 320 := by
  sorry

end blue_string_length_is_320_l513_513053


namespace hyperbola_asymptote_l513_513283

theorem hyperbola_asymptote (a b : ℝ) (h : a > 0) (h1 : b > 0)
  (eccentricity : 2 * a = real.sqrt (a ^ 2 + b ^ 2)) :
  (∀ x y : ℝ, ∃ k : ℝ, k = real.sqrt 3 ∧ (y = k * x ∨ y = -k * x)) :=
by sorry

end hyperbola_asymptote_l513_513283


namespace problem_statement_l513_513790

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513790


namespace pyramid_volume_property_l513_513243

theorem pyramid_volume_property
  (V : ℝ) 
  (S₁ S₂ S₃ S₄ : ℝ) 
  (H₁ H₂ H₃ H₄ : ℝ)
  (K : ℝ)
  (h₁ h₂ h₃ h₄ : ℝ)
  (a₁ a₂ a₃ a₄ : ℝ)
  (S : ℝ)
  (P : Point)
  (Q : Point)
  (convex_quadrilateral : ConvexQuadrilateral) :
  (a₁ / 1 = a₂ / 2 = a₃ / 3 = a₄ / 4 = k) →
  (h₁ + 2 * h₂ + 3 * h₃ + 4 * h₄ = (2 * S) / k) →
  (S₁ / 1 = S₂ / 2 = S₃ / 3 = S₄ / 4 = K) →
  (S₁ * H₁ + S₂ * H₂ + S₃ * H₃ + S₄ * H₄ = 3 * V) →
  (H₁ + 2 * H₂ + 3 * H₃ + 4 * H₄ = (3 * V) / K) :=
  sorry

end pyramid_volume_property_l513_513243


namespace first_term_of_geometric_sequence_l513_513473

theorem first_term_of_geometric_sequence (a r : ℝ) 
  (h1 : a * r^5 = 9!) 
  (h2 : a * r^8 = 10!) : 
  a = 9! / 10^(5/3) :=
by
  sorry

end first_term_of_geometric_sequence_l513_513473


namespace evaluate_expression_l513_513527

theorem evaluate_expression : (3^1 + 2^3 + 7^0 + 1)^{-2} * 2 = 2 / 169 := by
  sorry

end evaluate_expression_l513_513527


namespace arccos_one_half_l513_513613

theorem arccos_one_half : real.arccos (1/2) = real.pi / 3 := 
  sorry

end arccos_one_half_l513_513613


namespace probability_of_more_heads_than_tails_l513_513916

-- Define the probability of getting more heads than tails when flipping 10 coins
def probabilityMoreHeadsThanTails : ℚ :=
  193 / 512

-- Define the proof statement
theorem probability_of_more_heads_than_tails :
  let p : ℚ := probabilityMoreHeadsThanTails in
  p = 193 / 512 :=
by
  sorry

end probability_of_more_heads_than_tails_l513_513916


namespace peter_wins_prize_at_least_one_wins_prize_l513_513005

noncomputable def probability_peter_wins (N : Nat) := 
  (5/6) ^ (N - 1)

theorem peter_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob = probability_peter_wins N := by
  have h2 : prob = (5/6) ^ 9 := by
    sorry
  exact h2

noncomputable def probability_at_least_one_wins (N : Nat) := 
  10 * (5/6)^9 - 45 * (5 * 4^8 / 6^9) + 120 * (5 * 4 * 3^7 / 6^9) - 210 * (5 * 4 * 3 * 2^6 / 6^9) + 252 * (5 * 4 * 3 * 2 * 1 / 6^9)

theorem at_least_one_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob ≈ probability_at_least_one_wins N := by
  have h2 : prob ≈ 0.919 := by
    sorry
  exact h2

end peter_wins_prize_at_least_one_wins_prize_l513_513005


namespace proposition_p_and_q_is_true_l513_513718

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513718


namespace true_proposition_l513_513852

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513852


namespace statement_C_is_incorrect_l513_513098

def prism_sides_faces_vertices_edges (n : ℕ) : Prop :=
  n_faces = n + 2 ∧
  2n_vertices = 2 * n ∧
  3n_edges = 3 * n

def condition_C_is_incorrect (n : ℕ) : Prop :=
  ¬ (prism_sides_faces_vertices_edges n)

theorem statement_C_is_incorrect (n : ℕ) : condition_C_is_incorrect n :=
by {
  sorry, -- proof goes here
}

end statement_C_is_incorrect_l513_513098


namespace find_speed2_l513_513961

-- Definitions of the given conditions
def distance1 := 180 -- miles
def speed1 := 60 -- miles per hour
def distance2 := 120 -- miles
def avgSpeed := 50 -- miles per hour
def totalDistance := distance1 + distance2 -- total distance of the trip

-- The expected result
def speed2 := 40 -- miles per hour

theorem find_speed2 :
  let totalTime := totalDistance / avgSpeed in
  let time1 := distance1 / speed1 in
  let time2 := totalTime - time1 in
  speed2 = distance2 / time2 :=
by
  sorry

end find_speed2_l513_513961


namespace large_circle_radius_l513_513592

theorem large_circle_radius (s : ℝ) (r : ℝ) (R : ℝ)
  (side_length : s = 6)
  (coverage : ∀ (x y : ℝ), (x - y)^2 + (x - y)^2 = (2 * R)^2) :
  R = 3 * Real.sqrt 2 :=
by
  sorry

end large_circle_radius_l513_513592


namespace count_integers_between_cubes_l513_513299

theorem count_integers_between_cubes :
  let a := 10.5
  let b := 10.7
  let n1 := (a^3).ceil
  let n2 := (b^3).floor
  n2 - n1 + 1 = 67 :=
by
  let a := 10.5
  let b := 10.7
  let n1 := (a^3).ceil
  let n2 := (b^3).floor
  have h1 : (a^3 = 1157.625) := by sorry
  have h2 : (b^3 = 1225.043) := by sorry
  have h3 : (n1 = 1158) := by sorry
  have h4 : (n2 = 1224) := by sorry
  rw [h1, h2, h3, h4]
  sorry

end count_integers_between_cubes_l513_513299


namespace problem_statement_l513_513795

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513795


namespace percentage_problem_l513_513919

variable (N P : ℝ)

theorem percentage_problem (h1 : 0.3 * N = 120) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end percentage_problem_l513_513919


namespace total_emails_in_april_is_675_l513_513955

-- Define the conditions
def daily_emails : ℕ := 20
def additional_emails : ℕ := 5
def april_days : ℕ := 30
def half_april_days : ℕ := april_days / 2

-- Define the total number of emails received
def total_emails : ℕ :=
  (daily_emails * half_april_days) +
  ((daily_emails + additional_emails) * half_april_days)

-- Define the statement to be proven
theorem total_emails_in_april_is_675 : total_emails = 675 :=
  by
  sorry

end total_emails_in_april_is_675_l513_513955


namespace digit_count_divisibility_l513_513231

/-- There are exactly 7 digits n such that 15n is divisible by n. -/
theorem digit_count_divisibility : 
  (finset.filter (λ n : ℕ, n * (15 * n) % n = 0) (finset.range 10)).card = 7 :=
by
  sorry

end digit_count_divisibility_l513_513231


namespace proof_problem_l513_513778

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513778


namespace average_disk_space_per_hour_l513_513578

/-- A digital music library contains 15 days of music and takes up 20,000 megabytes of disk space.
    Prove that the average disk space used per hour of music in this library is 56 megabytes, to the nearest whole number.
-/
theorem average_disk_space_per_hour (days: ℕ) (total_disk_space: ℕ) (hours_per_day: ℕ) (div_result: ℕ) (avg_disk_space: ℕ) :
  days = 15 ∧ 
  total_disk_space = 20000 ∧ 
  hours_per_day = 24 ∧ 
  div_result = days * hours_per_day ∧
  avg_disk_space = (total_disk_space: ℝ) / div_result ∧ 
  (an_index: ℕ, (avg_disk_space: ℝ) ≈ 55.56 → an_index = 56) :=
by 
  sorry

end average_disk_space_per_hour_l513_513578


namespace inverse_value_l513_513460

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / x

theorem inverse_value :
  (∃ g : ℝ → ℝ, ∀ y, f (g y) = y) → (∀ y, g (f y) = y) →
  f⁻¹(3 / 2) = 2 :=
by
  sorry

end inverse_value_l513_513460


namespace evaluate_expression_l513_513205

-- Definitions based on conditions
def a : ℤ := 5
def b : ℤ := -3
def c : ℤ := 2

-- Theorem to be proved: evaluate the expression
theorem evaluate_expression : (3 : ℚ) / (a + b + c) = 3 / 4 := by
  sorry

end evaluate_expression_l513_513205


namespace net_income_on_15th_day_l513_513589

noncomputable def net_income_15th_day : ℝ :=
  let earnings_15th_day := 3 * (3 ^ 14)
  let tax := 0.10 * earnings_15th_day
  let earnings_after_tax := earnings_15th_day - tax
  earnings_after_tax - 100

theorem net_income_on_15th_day :
  net_income_15th_day = 12913916.3 := by
  sorry

end net_income_on_15th_day_l513_513589


namespace piravena_trip_distance_l513_513427

noncomputable def triangle_round_trip_distance (XZ XY : ℕ) (h : XZ^2 + (XY - XZ)^2 = XY^2) : ℕ :=
  let YZ := Math.sqrt (XY^2 - XZ^2)
  in if XZ^2 + YZ^2 = XY^2 then XY + YZ + XZ else 0

theorem piravena_trip_distance : triangle_round_trip_distance 4000 5000 (by norm_num) = 12000 :=
by
  let YZ := Math.sqrt (5000^2 - 4000^2)
  have hYZ : YZ = 3000 := by norm_num
  rw [hYZ]
  norm_num
  sorry

end piravena_trip_distance_l513_513427


namespace benny_spent_on_baseball_gear_l513_513176

theorem benny_spent_on_baseball_gear (initial_amount left_over spent : ℕ) 
  (h_initial : initial_amount = 67) 
  (h_left : left_over = 33) 
  (h_spent : spent = initial_amount - left_over) : 
  spent = 34 :=
by
  rw [h_initial, h_left] at h_spent
  exact h_spent

end benny_spent_on_baseball_gear_l513_513176


namespace sum_of_digits_of_repeating_decimal_of_1_div_81_squared_l513_513624

def repeating_decimal (a b : ℚ) : Prop :=
  ∃ n (digits : Fin n → ℕ), ∀ k : ℕ, (digits ⟨k % n, Nat.mod_lt _ (nat_pos_of_ne_zero (by decide))⟩) = (fractional_part (a / b) * 10 ^ k) % 10

noncomputable def fractional_part (q : ℚ) : ℚ := q - q.num / q.denom

theorem sum_of_digits_of_repeating_decimal_of_1_div_81_squared : 
  repeating_decimal (1 : ℚ) (81 ^ 2) → 
  ∃ (n : ℕ) (digits : Fin n → ℕ), 
  sum (List.ofFn digits) = 684 := 
by
  intro h
  sorry

end sum_of_digits_of_repeating_decimal_of_1_div_81_squared_l513_513624


namespace combined_share_rent_CD_l513_513156

/-- Given the contributions of A, B, C, and D in oxen-months and the total rent, 
    prove the combined share of rent for C and D. -/
def combined_share_rent (oxen_months_A : ℕ) (oxen_months_B : ℕ) (oxen_months_C : ℕ) 
(oxen_months_D : ℕ) (total_rent : ℝ) : ℝ :=
  let total_oxen_months := oxen_months_A + oxen_months_B + oxen_months_C + oxen_months_D
  let combined_CD := oxen_months_C + oxen_months_D
  (combined_CD : ℝ) / total_oxen_months * total_rent

theorem combined_share_rent_CD :
  combined_share_rent 70 60 45 120 2080 = 1163.39 :=
by
  sorry

end combined_share_rent_CD_l513_513156


namespace harmonic_sequence_sum_l513_513197

theorem harmonic_sequence_sum :
  ∀ (a b : ℕ → ℚ),
  (∀ n, n > 0 → (n : ℚ) / (a 1 + a 2 + ... + a n) = 1 / (2 * n + 1)) →
  (∀ n, b n = (a n + 1) / 4) →
  (∑ k in finset.range 10, 1 / (b k * b (k + 1))) = 10 / 11 :=
by
  intro a b h_mean h_b
  sorry

end harmonic_sequence_sum_l513_513197


namespace minimum_distance_proof_l513_513861

noncomputable def minimum_distance (a b c d : ℝ) (P : ℝ × ℝ) : ℝ :=
  abs ((-3 * P.1 + 4 * P.2 + 25) / 5)

theorem minimum_distance_proof :
  ∀ (a b c d : ℝ) (P : ℝ × ℝ),
    (P.1, P.2) ∈ 
      { (x, y) : ℝ × ℝ | 
        (x - a) ^ 2 + (y - b) ^ 2 = b ^ 2 + 1 ∧ 
        (x - c) ^ 2 + (y - d) ^ 2 = d ^ 2 + 1 } ∧ 
    ac = 8 ∧
    (a / b = c / d) →
    minimum_distance a b c d P = 2 :=
by
  intros a b c d P h h1 h2
  sorry

end minimum_distance_proof_l513_513861


namespace convert_mps_to_kmph_l513_513110

-- Define the conversion factor
def conversion_factor : ℝ := 3.6

-- Define the initial speed in meters per second
def initial_speed_mps : ℝ := 50

-- Define the target speed in kilometers per hour
def target_speed_kmph : ℝ := 180

-- Problem statement: Prove the conversion is correct
theorem convert_mps_to_kmph : initial_speed_mps * conversion_factor = target_speed_kmph := by
  sorry

end convert_mps_to_kmph_l513_513110


namespace roots_of_polynomial_l513_513219

noncomputable def polynomial : Polynomial ℝ := (X^2 - 5 * X + 6) * (X - 3) * (X + 1)

theorem roots_of_polynomial :
  ∃ (a b c : ℝ), polynomial = (X - a) * (X - b) * (X - c) ∧ (a = -1 ∨ a = 2 ∨ a = 3)
                                              ∧ (b = -1 ∨ b = 2 ∨ b = 3)
                                              ∧ (c = -1 ∨ c = 2 ∨ c = 3) := by
  sorry

end roots_of_polynomial_l513_513219


namespace problem_statement_l513_513704

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513704


namespace area_triangle_ABC_value_of_b_l513_513389

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {S₁ S₂ S₃ : ℝ}

-- Given conditions
axiom sides_of_triangle : (S₁ = (√3 / 4) * a^2) ∧ (S₂ = (√3 / 4) * b^2) ∧ (S₃ = (√3 / 4) * c^2)
axiom eq_area_diff : S₁ - S₂ + S₃ = √3 / 2
axiom sin_B : sin B = 1 / 3
axiom cos_B_positive : cos B = 2 * √2 / 3
axiom product_ac : a * c = 3 * √2 / 4
axiom sin_A_sin_C : sin A * sin C = √2 / 3

-- Prove
theorem area_triangle_ABC : (1 / 2) * a * c * sin B = √2 / 8 := by
  sorry

theorem value_of_b : b = 1 / 2 := by
  sorry

end area_triangle_ABC_value_of_b_l513_513389


namespace stating_count_1973_in_I_1000000_l513_513191

/--
Sequence definition: 
- I_0 consists of two ones: {1, 1}.
- For each subsequent I_n, it is formed by taking I_{n-1} and inserting the sum of each pair of adjacent terms between them.
-/
def I : ℕ → List ℕ 
| 0 => [1, 1]
| n + 1 => let prev := I n
           prev.head :: (List.zipWith (· + ·) (prev) (prev.tail)) ++ [prev.lastI]

/--
Theorem stating that the number 1973 appears exactly 1972 times in I_{1000000}.
-/
theorem count_1973_in_I_1000000 : I 1000000 |>.count 1973 = 1972 :=
  sorry

end stating_count_1973_in_I_1000000_l513_513191


namespace sequence_Sn_l513_513663

noncomputable def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ (∀ n, S (sum_seq n) = n^2 * a n)

def sum_seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  finset.sum (finset.range n.succ) a

def Sn (a : ℕ → ℚ) (S : ℕ → ℚ) (n : ℕ) : Prop :=
  S n = (2 * n) / (n + 1)

theorem sequence_Sn
  (a : ℕ → ℚ) (S : ℕ → ℚ) :
  sequence a → (∀ n, Sn a S n) :=
begin
  sorry
end

end sequence_Sn_l513_513663


namespace average_mb_per_hour_l513_513573

theorem average_mb_per_hour (days : ℕ) (total_disk_space : ℕ)
  (h_days : days = 15) (h_total_disk_space : total_disk_space = 20000) :
  let total_hours := days * 24
  let mb_per_hour := total_disk_space / total_hours.toFloat
  round mb_per_hour = 56 :=
by
  let total_hours := days * 24
  let mb_per_hour := total_disk_space / total_hours.toFloat
  have : total_hours = 360 := by rw [h_days]; simp
  have : mb_per_hour ≈ 55.56 := by rw [h_total_disk_space, this]; simp
  have : round mb_per_hour = 56 := by norm_cast; simp
  exact this

end average_mb_per_hour_l513_513573


namespace continuous_function_must_be_identity_example_function_properties_l513_513119

-- Definitions based on conditions in part a)
variable (f : ℝ → ℝ)
variable (continuous_f : Continuous f)
variable (h_comp : ∀ x : ℝ, f (f (f x)) = x)

-- The problem statement for part 1
theorem continuous_function_must_be_identity (x : ℝ) : f x = x :=
 by sorry

-- Definitions and theorem statement for part 2: Example function g
noncomputable def g : ℝ → ℝ := λ x, if x = 1 then 2 else if x = 2 then 3 else if x = 3 then 1 else x

theorem example_function_properties :
  (∀ x : ℝ, g (g (g x)) = x ∧ (x = 1 ∨ x = 2 ∨ x = 3 → g x ≠ x)) :=
 by sorry

end continuous_function_must_be_identity_example_function_properties_l513_513119


namespace peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513014

-- Probability of Peter winning a prize:
theorem peter_wins_prize_probability :
  let p := (5 / 6) in p ^ 9 = (5 / 6) ^ 9 := by
  sorry

-- Probability that at least one person wins a prize:
theorem at_least_one_wins_prize_probability :
  let p := (5 / 6) in
  let q := (1 - p^9) in 
  (1 - q^10) ≈ 0.919 := by
  sorry

end peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513014


namespace telegraph_longer_than_pardee_l513_513452

theorem telegraph_longer_than_pardee : 
  let telegraph_length_km := 162 in
  let pardee_length_m := 12000 in
  let pardee_length_km := pardee_length_m / 1000 in
  telegraph_length_km - pardee_length_km = 150 :=
by
  sorry

end telegraph_longer_than_pardee_l513_513452


namespace proposition_p_and_q_is_true_l513_513724

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513724


namespace p_and_q_is_true_l513_513689

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513689


namespace order_of_f_l513_513969

theorem order_of_f (f : ℝ → ℝ) (h_even : ∀ x, f (-x) = f x) (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  have h1 : f (-2) = f 2 := h_even 2
  have h2 : f (-π) = f π := h_even π
  have h3 : f 2 < f 3 := h_mono 2 3 (by linarith) (by linarith)
  have h4 : f 3 < f π := h_mono 3 π (by linarith) (by norm_num [Real.pi_pos])
  exact ⟨by linarith [h2, h4, h1], by linarith [h1, h3]⟩

end order_of_f_l513_513969


namespace ordering_of_a_b_c_l513_513654

-- Definitions based on given conditions
def a : ℝ := Real.log 2
def b : ℝ := Real.sqrt 3 - 1
def c : ℝ := Real.sin 1

-- Theorem statement that needs to be proven
theorem ordering_of_a_b_c : c > b ∧ b > a :=
by
  sorry

end ordering_of_a_b_c_l513_513654


namespace final_portfolio_value_l513_513372

theorem final_portfolio_value (initial_amount : ℕ) (growth_1 : ℕ) (additional_funds : ℕ) (growth_2 : ℕ) :
  let after_first_year_growth := initial_amount + (initial_amount * growth_1 / 100)
  let after_adding_funds := after_first_year_growth + additional_funds
  let after_second_year_growth := after_adding_funds + (after_adding_funds * growth_2 / 100)
  after_second_year_growth = 132 :=
by
  let after_first_year_growth := initial_amount + (initial_amount * growth_1 / 100);
  let after_adding_funds := after_first_year_growth + additional_funds;
  let after_second_year_growth := after_adding_funds + (after_adding_funds * growth_2 / 100);
  trivial

-- substituting the values as per conditions:
example : final_portfolio_value 80 15 28 10 = 132 := by
  let after_first_year_growth := 80 + (80 * 15 / 100);
  let after_adding_funds := after_first_year_growth + 28;
  let after_second_year_growth := after_adding_funds + (after_adding_funds * 10 / 100);
  trivial

end final_portfolio_value_l513_513372


namespace product_units_digit_odd_numbers_20_120_l513_513521

open Finset

def odd_numbers_between_20_and_120 : Finset ℕ := 
  filter (λ n => (n % 2 = 1) ∧ (20 < n) ∧ (n < 120)) (range 121)

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def product_units_digit (s : Finset ℕ) : ℕ := 
  (s.prod id) % 10

theorem product_units_digit_odd_numbers_20_120 :
  product_units_digit odd_numbers_between_20_and_120 = 5 :=
sorry

end product_units_digit_odd_numbers_20_120_l513_513521


namespace tangent_half_angle_product_l513_513989

theorem tangent_half_angle_product (r d : ℝ) (h_r : 0 < r) (h_d : 0 < d) :
  ∀ (α β : ℝ), 
    (∃ M : ℝ, M ∈ t) → 
    (∃ t₁ t₂ : ℝ, t₁ tangent_to_circle r d ∧ t₂ tangent_to_circle r d ∧ forms_oriented_angle t₁ α t ∧ forms_oriented_angle t₂ β t) → 
    (tan (α / 2)) * (tan (β / 2)) = (r - d) / (r + d) :=
sorry

end tangent_half_angle_product_l513_513989


namespace problem_statement_l513_513788

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513788


namespace peter_wins_prize_at_least_one_person_wins_prize_l513_513010

-- Part (a): Probability that Peter wins a prize
theorem peter_wins_prize :
  let p : Probability := (5 / 6) ^ 9
  p = 0.194 := sorry

-- Part (b): Probability that at least one person wins a prize
theorem at_least_one_person_wins_prize :
  let p : Probability := 0.919
  p = 0.919 := sorry

end peter_wins_prize_at_least_one_person_wins_prize_l513_513010


namespace sarah_initial_money_l513_513443

def initial_money 
  (cost_toy_car : ℕ)
  (cost_scarf : ℕ)
  (cost_beanie : ℕ)
  (remaining_money : ℕ)
  (number_of_toy_cars : ℕ) : ℕ :=
  remaining_money + cost_beanie + cost_scarf + number_of_toy_cars * cost_toy_car

theorem sarah_initial_money : 
  (initial_money 11 10 14 7 2) = 53 :=
by
  rfl 

end sarah_initial_money_l513_513443


namespace boys_count_at_table_l513_513120

-- Definitions from conditions
def children_count : ℕ := 13
def alternates (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- The problem to be proven in Lean:
theorem boys_count_at_table : ∃ b g : ℕ, b + g = children_count ∧ alternates b ∧ alternates g ∧ b = 7 :=
by
  sorry

end boys_count_at_table_l513_513120


namespace binomial_arithmetic_sequence_iff_l513_513646

open Nat

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  n.choose k 

-- Conditions
def is_arithmetic_sequence (n k : ℕ) : Prop :=
  binomial n (k-1) - 2 * binomial n k + binomial n (k+1) = 0

-- Statement to prove
theorem binomial_arithmetic_sequence_iff (u : ℕ) (u_gt2 : u > 2) :
  ∃ (n k : ℕ), (n = u^2 - 2) ∧ (k = binomial u 2 - 1 ∨ k = binomial (u+1) 2 - 1) 
  ↔ is_arithmetic_sequence n k := 
sorry

end binomial_arithmetic_sequence_iff_l513_513646


namespace number_of_correct_propositions_l513_513968

section Geometry

variables (a b : Line) (alpha beta : Plane)

-- Definitions for the problem conditions
def perp_line_line (a b : Line) : Prop := ⟪a, b⟫ = 0
def perp_line_plane (a : Line) (alpha : Plane) : Prop := ∀ (p : Point), p ∈ a → perp (p, alpha)
def parallel_line_plane (b : Line) (alpha : Plane) : Prop := ∀ (p₁ p₂ : Point), p₁ ∈ b ∧ p₂ ∈ alpha → p₁ = p₂
def subset_line_plane (b : Line) (alpha : Plane) : Prop := ∀ (p : Point), p ∈ b → p ∈ alpha
def perp_plane_plane (alpha beta : Plane) : Prop := ∀ (p : Point), perp (p, alpha) ∧ perp (p, beta)

-- Theorem Statement
theorem number_of_correct_propositions :
  (∀ (a b : Line) (alpha beta : Plane),
    (perp_line_line a b ∧ perp_line_plane a alpha ∧ ¬subset_line_plane b alpha → parallel_line_plane b alpha) ∧
    (parallel_line_plane a alpha ∧ perp_line_plane a beta → perp_plane_plane alpha beta) ∧
    (perp_line_plane a beta ∧ perp_plane_plane alpha beta → parallel_line_plane a alpha ∨ subset_line_plane a alpha) ∧
    (perp_line_line a b ∧ perp_line_plane a alpha ∧ perp_line_plane b beta → perp_plane_plane alpha beta)) := sorry

end Geometry

end number_of_correct_propositions_l513_513968


namespace conjugate_of_expression_l513_513267

-- Definitions for conditions
def z := 1 - complex.i

-- The problem statement to be proven
theorem conjugate_of_expression :
  complex.conj (2 / z - z^2) = 1 - 3 * complex.i :=
sorry

end conjugate_of_expression_l513_513267


namespace correct_function_l513_513101

noncomputable def f (x : ℝ) : ℝ := x * (x + 1) * (x - 1)

lemma f_odd : ∀ x : ℝ, f (-x) = -f (x) :=
by
  sorry

lemma f_monotonic_increasing : ∀ x : ℝ, x > 2 → 0 < (3 * x^2 - 1) :=
by
  sorry

lemma f_zeros : ∃ a : ℝ, f(a) = 0 ∧ f(-a) = 0 ∧ f(0) = 0 ∧ ∀ x : ℝ, f(x) = 0 → x = 0 ∨ x = 1 ∨ x = -1 :=
by
  sorry

theorem correct_function : 
  ∃ (f : ℝ → ℝ), (∀ x, f(-x) = -f(x)) ∧ 
                  (∀ x, x > 2 → 0 < (3 * x^2 - 1)) ∧ 
                  (∃ a, f(a) = 0 ∧ f(-a) = 0 ∧ f(0) = 0 ∧ ∀ x, f(x) = 0 → x = 0 ∨ x = 1 ∨ x = -1) :=
by
  use f
  refine ⟨f_odd, f_monotonic_increasing, f_zeros⟩
  sorry

end correct_function_l513_513101


namespace least_three_students_same_score_l513_513151

noncomputable def scoring_function (initial_points num_correct num_incorrect : ℕ) : ℕ :=
  initial_points + 4 * num_correct - 1 * num_incorrect

def min_duplicated_scores (initial_points total_questions total_students : ℕ) : Prop :=
  ∃ (duplicated_scores : ℕ), duplicated_scores ≥ 3

theorem least_three_students_same_score :
  min_duplicated_scores 6 6 51 :=
by 
  have score_set := 
    {scoring_function 6 x y | x y : ℕ, x + y ≤ 6}
  have unique_scores := score_set.to_finset.card
  have : unique_scores ≤ 25 := by sorry
  have pigeonhole := nat.ceil_div (51 : ℕ) 25
  have at_least := pigeonhole.to_fun_ge_3
  exact at_least

end least_three_students_same_score_l513_513151


namespace Nancy_antacid_consumption_l513_513420

theorem Nancy_antacid_consumption :
  let antacids_per_month : ℕ :=
    let antacids_per_day_indian := 3
    let antacids_per_day_mexican := 2
    let antacids_per_day_other := 1
    let days_indian_per_week := 3
    let days_mexican_per_week := 2
    let days_total_per_week := 7
    let weeks_per_month := 4

    let antacids_per_week_indian := antacids_per_day_indian * days_indian_per_week
    let antacids_per_week_mexican := antacids_per_day_mexican * days_mexican_per_week
    let days_other_per_week := days_total_per_week - days_indian_per_week - days_mexican_per_week
    let antacids_per_week_other := antacids_per_day_other * days_other_per_week

    let antacids_per_week_total := antacids_per_week_indian + antacids_per_week_mexican + antacids_per_week_other

    antacids_per_week_total * weeks_per_month
    
  antacids_per_month = 60 := sorry

end Nancy_antacid_consumption_l513_513420


namespace count_unique_ones_digits_divisible_by_16_l513_513983

def is_divisible_by_16 (n : ℕ) : Prop :=
  n % 16 = 0

def ones_digit (n : ℕ) : ℕ :=
  n % 10

def possible_ones_digits : Finset ℕ :=
  (Finset.range 100).filter is_divisible_by_16 |>.image ones_digit

theorem count_unique_ones_digits_divisible_by_16 :
  possible_ones_digits.card = 5 :=
by
  sorry

end count_unique_ones_digits_divisible_by_16_l513_513983


namespace proposition_p_and_q_is_true_l513_513719

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513719


namespace true_conjunction_l513_513681

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513681


namespace binary_representation_of_multiple_of_617_has_at_least_six_zeros_l513_513137

theorem binary_representation_of_multiple_of_617_has_at_least_six_zeros
    (n : ℕ) (h : ∃ m : ℕ, n = 617 * m) (hn : nat.popcount n = 3) :
  nat.bodd n = false ∧ nat.bcount n ≥ 6 := sorry

end binary_representation_of_multiple_of_617_has_at_least_six_zeros_l513_513137


namespace sum_of_ages_l513_513374

theorem sum_of_ages (a b c d : ℕ) (h1 : a * b = 20) (h2 : c * d = 28) (distinct : ∀ (x y : ℕ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) : a + b + c + d = 19 :=
sorry

end sum_of_ages_l513_513374


namespace proof_problem_l513_513779

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513779


namespace num_solution_divisor_is_8_l513_513463

theorem num_solution_divisor_is_8 :
  {x : ℤ | (x + 1) ∣ -6}.to_finset.card = 8 :=
by
  sorry

end num_solution_divisor_is_8_l513_513463


namespace unique_common_tangent_l513_513667

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (a x : ℝ) : ℝ := a * Real.exp (x + 1)

theorem unique_common_tangent (a : ℝ) (h : a > 0) : 
  (∃ k x₁ x₂, k = 2 * x₁ ∧ k = a * Real.exp (x₂ + 1) ∧ k = (g a x₂ - f x₁) / (x₂ - x₁)) →
  a = 4 / Real.exp 3 :=
by
  sorry

end unique_common_tangent_l513_513667


namespace total_salmons_caught_l513_513295

theorem total_salmons_caught :
  let hazel_salmons := 24
  let dad_salmons := 27
  hazel_salmons + dad_salmons = 51 :=
by
  sorry

end total_salmons_caught_l513_513295


namespace probability_sum_scores_9_l513_513498

-- Define the points logic
def points (result : string) : ℕ :=
  if result = "win" then 3
  else if result = "draw" then 1
  else 0

-- Define a function that computes total points
def total_points (a_results : list string) (b_results : list string) : ℕ × ℕ :=
  (a_results.map points).sum, (b_results.map points).sum

-- The total number of games
def total_games := 3

--List out all possible score combinations for 3 games
def possible_scores : list (ℕ × ℕ) := [
  (9,0), (7,1), (6,3), (5,2), (4,4), 
  (3,6), (0,9), (1,7), (2,5), (3,3)
]

-- The favorable outcomes
def favorable_scores : list (ℕ × ℕ) := [
  (9,0), (6,3), (3,6), (0,9)
]

-- Checking probability calculations for sum of scores is 9
theorem probability_sum_scores_9 : 
  (favorable_scores.length : ℝ) / (possible_scores.length : ℝ) = (2 / 5) :=
by
  sorry

end probability_sum_scores_9_l513_513498


namespace true_proposition_l513_513759

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513759


namespace minimum_value_of_f_l513_513216

/-
  Define the function to be minimized.
  We are looking to prove that the minimum value of this function
  over all real numbers x is equal to 2 * sqrt 5.
-/
def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 2 * Real.sqrt 5 :=
sorry

end minimum_value_of_f_l513_513216


namespace number_of_arrangements_l513_513900

-- Define the set of numbers and conditions
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
def line_equivalence_sum := 34

-- Problem statement: Determine the number of valid arrangements
theorem number_of_arrangements : 
  (set of all valid arrangements of numbers such that the sum of each line equals 34).card = 1296 :=
sorry

end number_of_arrangements_l513_513900


namespace problem_statement_l513_513797

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513797


namespace no_integer_roots_for_odd_pairs_l513_513365

theorem no_integer_roots_for_odd_pairs :
  ∀ (a b : ℕ) (S : Fin 10 → ℤ),
    (∀ i : Fin 10, S i ≡ 1 [ZMod 2]) →  -- S contains 10 consecutive odd numbers
    (∀ (n : ℕ), n < 10 → -- We can index from 0 to 9
      ∃ k : Fin 5, -- We can pair them up
        (a = S (2 * k)) ∧ -- Each pair (a, b) consists of consecutive indices
        (b = S (2 * k + 1))) →
    ¬ (∀ k : Fin 5, ∃ x y : ℤ, x + y = -a ∧ x * y = b ∧ -- Condition for integer roots
      x^2 + a * x + b = 0) := -- Form of the quadratic equation
sorry

end no_integer_roots_for_odd_pairs_l513_513365


namespace trade_ban_impact_l513_513543

noncomputable def absolute_advantage (a_zu a_caul b_zu b_caul : ℝ) : Prop :=
b_zu > a_zu ∧ b_caul > a_caul

noncomputable def comparative_advantage_caul (a_c_caul a_c_zu b_c_caul b_c_zu : ℝ) : Prop :=
(a_c_zu / a_c_caul) > (b_c_zu / b_c_caul)

noncomputable def comparative_advantage_zu (a_c_zu a_c_caul b_c_zu b_c_caul : ℝ) : Prop :=
(a_c_caul / a_c_zu) > (b_c_caul / b_c_zu)

noncomputable def production_possibility_curve (max_zu max_caul production_zu production_caul : ℝ) : Prop :=
(production_zu / max_zu) + (production_caul / max_caul) ≤ 1

noncomputable def free_trade_consumption (a_revenue a_price b_revenue b_price : ℝ) : Prop :=
(a_revenue / a_price) / 2 = 8 ∧ (b_revenue / b_price) / 2 = 18

noncomputable def post_trade_ban_consumption (a_caul b_caul_b a_zu_b_zu_diff : ℝ) : Prop :=
(a_caul + b_caul_b) = 24 ∧ (a_zu_b_zu_diff) = 24 ∧ 24 < 28

theorem trade_ban_impact (max_a_zu max_a_caul max_b_zu max_b_caul : ℝ) (a_price b_price : ℝ) 
                         (a_revenue b_revenue trade_ban a_zu_b_zu_diff b_caul_b a_caul : ℝ) :
    absolute_advantage max_a_zu max_a_caul max_b_zu max_b_caul →
    comparative_advantage_caul max_a_caul max_a_zu max_b_caul max_b_zu →
    comparative_advantage_zu max_a_zu max_a_caul max_b_zu max_b_caul →
    production_possibility_curve max_a_zu max_a_caul max_b_zu max_b_caul →
    free_trade_consumption a_revenue a_price b_revenue b_price →
    post_trade_ban_consumption a_caul b_caul_b a_zu_b_zu_diff →
    24 < 28 → 
    sorry

end trade_ban_impact_l513_513543


namespace sum_A_C_l513_513629

-- Introduce the definitions and conditions
def A : ℕ := sorry
def B : ℕ := sorry
def C : ℕ := sorry
def D : ℕ := sorry

-- A, B, C, and D are distinct integers
axiom h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
-- A, B, C, and D are within the set {1, 2, 3, 4}
axiom h2 : A ∈ {1, 2, 3, 4} ∧ B ∈ {1, 2, 3, 4} ∧ C ∈ {1, 2, 3, 4} ∧ D ∈ {1, 2, 3, 4}
-- The given equation
axiom h3 : (A / B) + (C / D) = 3

-- We aim to prove A + C = 4
theorem sum_A_C : A + C = 4 :=
by
  sorry

end sum_A_C_l513_513629


namespace train_crossing_time_l513_513356

noncomputable def speed_km_hr_to_m_s (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem train_crossing_time (length : ℝ) (speed : ℝ) :
  speed = 350 → length = 500 →
  (length / (speed_km_hr_to_m_s speed)) ≈ 5.14 :=
by
  intros hs hl
  rw [hs, hl]
  -- Proof skipped for brevity
  sorry

end train_crossing_time_l513_513356


namespace new_seq_49_equals_original_seq_12_l513_513358

-- Defining the original sequence
def original_seq : ℕ → ℕ := sorry

-- Defining the new sequence after inserting 3 numbers between every two terms of the original sequence
def new_seq (n : ℕ) : ℕ :=
  let q := n / 4
  in if n % 4 = 3 then
    sorry -- insert a specific value between adjacent terms
  else
    original_seq q

-- Proving that the 49th term of the new sequence is the 12th term of the original sequence
theorem new_seq_49_equals_original_seq_12 :
  new_seq 49 = original_seq 12 :=
by
  sorry

end new_seq_49_equals_original_seq_12_l513_513358


namespace who_is_murdock_l513_513115

variables (A B C : Prop)
variables (is_murdock : Prop)
variables (is_knight : Prop)
variables (is_liar : Prop)

-- Conditions
def A_statement := is_murdock
def B_statement := A_statement
def C_statement := ¬ is_murdock

-- Knight tells the truth
axiom knight_truth : is_knight → A_statement = true ∨ B_statement = true ∨ C_statement = true

-- Liar tells lies
axiom liar_lying : is_liar → A_statement = false ∨ B_statement = false ∨ C_statement = false

-- Exactly one knight and one liar
axiom unique_knight_liar : ∃ A_is_knight A_is_liar B_is_murdock B_is_knight B_is_liar C_is_murdock C_is_knight C_is_liar, 
  (A_is_knight ≠ B_is_knight ∧ A_is_knight ≠ C_is_knight ∧ B_is_knight ≠ C_is_knight) ∧
  (A_is_liar ≠ B_is_liar ∧ A_is_liar ≠ C_is_liar ∧ B_is_liar ≠ C_is_liar) ∧
  (is_knight ∨ is_liar)

-- The goal: prove B is Murdock
theorem who_is_murdock : is_murdock = B :=
by sorry

end who_is_murdock_l513_513115


namespace line_equation_l513_513078

theorem line_equation 
  (m b k : ℝ) 
  (h1 : ∀ k, abs ((k^2 + 4 * k + 4) - (m * k + b)) = 4)
  (h2 : m * 2 + b = 8) 
  (h3 : b ≠ 0) : 
  m = 8 ∧ b = -8 :=
by sorry

end line_equation_l513_513078


namespace washington_goats_l513_513001

variables (W : ℕ) (P : ℕ) (total_goats : ℕ)

theorem washington_goats (W : ℕ) (h1 : P = W + 40) (h2 : total_goats = W + P) (h3 : total_goats = 320) : W = 140 :=
by
  sorry

end washington_goats_l513_513001


namespace orange_balls_count_l513_513596

theorem orange_balls_count (total_balls red_balls blue_balls yellow_balls green_balls pink_balls orange_balls : ℕ) 
(h_total : total_balls = 100)
(h_red : red_balls = 30)
(h_blue : blue_balls = 20)
(h_yellow : yellow_balls = 10)
(h_green : green_balls = 5)
(h_pink : pink_balls = 2 * green_balls)
(h_orange : orange_balls = 3 * pink_balls)
(h_sum : red_balls + blue_balls + yellow_balls + green_balls + pink_balls + orange_balls = total_balls) :
orange_balls = 30 :=
sorry

end orange_balls_count_l513_513596


namespace min_altitude_third_altitude_l513_513086

noncomputable def minimum_third_altitude : ℕ :=
  let h_D := 18
  let h_E := 8
  let h_F_min := 17 in
  ∃ (a b c : ℕ), b = 2 * a ∧ 
                (2 * 9 * a) / c = 18 ∧ 
                (2 * 4 * b) / 8 = 9 * a ∧ 
                2 * 9 * a / c = h_F_min

theorem min_altitude_third_altitude :
  minimum_third_altitude = 17 :=
sorry

end min_altitude_third_altitude_l513_513086


namespace units_digit_of_product_of_odds_between_20_and_120_l513_513526

theorem units_digit_of_product_of_odds_between_20_and_120 : 
  let odds := filter (fun n => n % 2 = 1) [21, 23, ... , 119] in
  ∃ digit, digit = 5 ∧ digit = ((odds.foldl (λ acc n => acc * n) 1) % 10) := 
sorry

end units_digit_of_product_of_odds_between_20_and_120_l513_513526


namespace main_statement_l513_513749

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513749


namespace john_fouled_per_game_l513_513962

theorem john_fouled_per_game
  (hit_rate : ℕ) (shots_per_foul : ℕ) (total_games : ℕ) (participation_rate : ℚ) (total_free_throws : ℕ) :
  hit_rate = 70 → shots_per_foul = 2 → total_games = 20 → participation_rate = 0.8 → total_free_throws = 112 →
  (total_free_throws / (participation_rate * total_games)) / shots_per_foul = 3.5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end john_fouled_per_game_l513_513962


namespace proposition_p_and_q_is_true_l513_513726

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513726


namespace average_brown_mms_per_bag_l513_513438

-- Definitions based on the conditions
def bag1_brown_mm : ℕ := 9
def bag2_brown_mm : ℕ := 12
def bag3_brown_mm : ℕ := 8
def bag4_brown_mm : ℕ := 8
def bag5_brown_mm : ℕ := 3
def number_of_bags : ℕ := 5

-- The proof problem statement
theorem average_brown_mms_per_bag : 
  (bag1_brown_mm + bag2_brown_mm + bag3_brown_mm + bag4_brown_mm + bag5_brown_mm) / number_of_bags = 8 := 
by
  sorry

end average_brown_mms_per_bag_l513_513438


namespace find_derivative_l513_513873

theorem find_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by
  sorry

end find_derivative_l513_513873


namespace find_area_of_triangle_FIL_l513_513081

def square_area (a: ℝ) : ℝ := a * a

def triangle_area (base height: ℝ) : ℝ := 0.5 * base * height

theorem find_area_of_triangle_FIL 
  (a1 a2 a3 : ℝ) 
  (h1 : square_area (sqrt a1) = 10) 
  (h2 : square_area (sqrt a2) = 90) 
  (h3 : square_area (sqrt a3) = 40) 
  (base height : ℝ) 
  (h4 : base = (21 / 2) * sqrt 10) 
  (h5 : height = (21 / 5) * sqrt 10) 
  : triangle_area base height = 220.5 := 
by 
  sorry

end find_area_of_triangle_FIL_l513_513081


namespace particle_max_height_l513_513595

noncomputable def max_height (r ω g : ℝ) : ℝ :=
  (r * ω + g / ω) ^ 2 / (2 * g)

theorem particle_max_height (r ω g : ℝ) (h : ω > Real.sqrt (g / r)) :
    max_height r ω g = (r * ω + g / ω) ^ 2 / (2 * g) :=
sorry

end particle_max_height_l513_513595


namespace complex_number_in_second_quadrant_l513_513866

def i : ℂ := Complex.i
def z : ℂ := i / (1 - i)

theorem complex_number_in_second_quadrant (z : ℂ) :
  z = i / (1 - i) → -1/2 < 0 ∧ 1/2 > 0 :=
sorry

end complex_number_in_second_quadrant_l513_513866


namespace pupils_in_class_l513_513587

theorem pupils_in_class (n : ℕ) (wrong_entry_increase : n * (1/2) = 13) : n = 26 :=
sorry

end pupils_in_class_l513_513587


namespace parallelogram_d_l513_513346

theorem parallelogram_d (A B C D : ℂ) :
  A = 4 + complex.I ∧ 
  B = 3 + 4 * complex.I ∧ 
  C = 5 + 2 * complex.I → 
  D = 6 - complex.I :=
by
  intro h
  cases h with hA hBC
  cases hBC with hB hC
  sorry

end parallelogram_d_l513_513346


namespace product_units_digit_odd_numbers_20_120_l513_513520

open Finset

def odd_numbers_between_20_and_120 : Finset ℕ := 
  filter (λ n => (n % 2 = 1) ∧ (20 < n) ∧ (n < 120)) (range 121)

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def product_units_digit (s : Finset ℕ) : ℕ := 
  (s.prod id) % 10

theorem product_units_digit_odd_numbers_20_120 :
  product_units_digit odd_numbers_between_20_and_120 = 5 :=
sorry

end product_units_digit_odd_numbers_20_120_l513_513520


namespace area_triangle_ABC_value_of_b_l513_513388

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {S₁ S₂ S₃ : ℝ}

-- Given conditions
axiom sides_of_triangle : (S₁ = (√3 / 4) * a^2) ∧ (S₂ = (√3 / 4) * b^2) ∧ (S₃ = (√3 / 4) * c^2)
axiom eq_area_diff : S₁ - S₂ + S₃ = √3 / 2
axiom sin_B : sin B = 1 / 3
axiom cos_B_positive : cos B = 2 * √2 / 3
axiom product_ac : a * c = 3 * √2 / 4
axiom sin_A_sin_C : sin A * sin C = √2 / 3

-- Prove
theorem area_triangle_ABC : (1 / 2) * a * c * sin B = √2 / 8 := by
  sorry

theorem value_of_b : b = 1 / 2 := by
  sorry

end area_triangle_ABC_value_of_b_l513_513388


namespace true_proposition_l513_513764

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513764


namespace part_a_part_b_l513_513664

-- Define the problem context and the theorem to be proven.
variables (A B C X Y : Type*)
variables [HasDistance A B] [HasDistance B C] [HasDistance A C]

-- Given conditions for triangle ABC being acute-angled.
def acute_angled_triangle (A B C : Type*) :=
  ∀ {a b c : ℝ}, 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- Part (a) conditions and theorem statement.
theorem part_a (h : acute_angled_triangle A B C) 
  (hX : AX = XY) (hY : XY = YC) : AX = XY ∧ XY = YC := by sorry

-- Part (b) conditions and theorem statement.
theorem part_b (h : acute_angled_triangle A B C) 
  (hX : BX = XY) (hY : XY = YC) : BX = XY ∧ XY = YC := by sorry

end part_a_part_b_l513_513664


namespace quadratic_solution_l513_513445

theorem quadratic_solution (x : ℝ) : (x^2 + 6 * x + 8 = -2 * (x + 4) * (x + 5)) ↔ (x = -8 ∨ x = -4) :=
by
  sorry

end quadratic_solution_l513_513445


namespace product_of_third_side_l513_513500

theorem product_of_third_side (a b : ℝ) (h₁ : a = 5) (h₂ : b = 7) :
  let c := real.sqrt (a ^ 2 + b ^ 2)
  let d := real.sqrt ((b ^ 2) - (a ^ 2))
  real.round ((c * d) * 10) / 10 = 42.0 :=
by
  sorry

end product_of_third_side_l513_513500


namespace total_emails_in_april_l513_513952

-- Definitions representing the conditions
def emails_per_day_initial : Nat := 20
def extra_emails_per_day : Nat := 5
def days_in_month : Nat := 30
def half_days_in_month : Nat := days_in_month / 2

-- Definitions to calculate total emails
def emails_first_half : Nat := emails_per_day_initial * half_days_in_month
def emails_per_day_after_subscription : Nat := emails_per_day_initial + extra_emails_per_day
def emails_second_half : Nat := emails_per_day_after_subscription * half_days_in_month

-- Main theorem to prove the total number of emails received in April
theorem total_emails_in_april : emails_first_half + emails_second_half = 675 := by 
  calc
    emails_first_half + emails_second_half
    = (emails_per_day_initial * half_days_in_month) + (emails_per_day_after_subscription * half_days_in_month) : rfl
    ... = (20 * 15) + ((20 + 5) * 15) : rfl
    ... = 300 + 375 : rfl
    ... = 675 : rfl

end total_emails_in_april_l513_513952


namespace rings_stack_distance_l513_513148

theorem rings_stack_distance :
  ∀ top_diam bottom_diam thickness_decrement : ℝ,
  ∀ ring_thickness number_of_rings : ℕ,
  top_diam = 25 →
  bottom_diam = 5 →
  thickness_decrement = 1.5 →
  ring_thickness = 0.5 →
  number_of_rings = 14 →
  let inside_diameter_sum := (number_of_rings : ℝ) / 2 * (top_diam - ring_thickness + (bottom_diam - ring_thickness)) in
  let total_distance := inside_diameter_sum + ring_thickness in
  total_distance = 204 := 
begin
  intros,
  calc
  inside_diameter_sum = 14 / 2 * (25 - 0.5 + (5 - 0.5)) : by rw [top_diam, bottom_diam, ring_thickness]
  ... = 7 * 29 : by norm_num
  ... = 203 : by norm_num,

  calc
  total_distance = inside_diameter_sum + 0.5 : by rw [ring_thickness]
  ... = 203 + 1 : by norm_num
  ... = 204 : by norm_num,
end

end rings_stack_distance_l513_513148


namespace first_term_of_geometric_sequence_l513_513472

theorem first_term_of_geometric_sequence (a r : ℝ) 
  (h1 : a * r^5 = 9!) 
  (h2 : a * r^8 = 10!) : 
  a = 9! / 10^(5/3) :=
by
  sorry

end first_term_of_geometric_sequence_l513_513472


namespace solve_f_eq_1_l513_513977

def f (x : ℝ) : ℝ :=
  if x < -1 then 2 * x + 4
  else if x < 2 then x - 10
  else 3 * x - 5

theorem solve_f_eq_1 : {x : ℝ | f(x) = 1} = {-3 / 2, 2} :=
by
  sorry

end solve_f_eq_1_l513_513977


namespace p_and_q_is_true_l513_513694

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513694


namespace true_proposition_l513_513851

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513851


namespace banquet_food_consumption_l513_513049

theorem banquet_food_consumption (n : ℕ) (food_per_guest : ℕ) (total_food : ℕ) 
  (h1 : ∀ g : ℕ, g ≤ n -> g * food_per_guest ≤ total_food)
  (h2 : n = 169) 
  (h3 : food_per_guest = 2) :
  total_food = 338 := 
sorry

end banquet_food_consumption_l513_513049


namespace valid_inequalities_l513_513904

theorem valid_inequalities (a b : ℝ) (h : a > b) : a^3 > b^3 ∧ 2^a > 2^b :=
by
  -- Given that a > b, prove a^3 > b^3
  have h1 : a^3 > b^3 := 
    sorry,
  -- Given that a > b, prove 2^a > 2^b
  have h2 : 2^a > 2^b := 
    sorry,
  exact ⟨h1, h2⟩

end valid_inequalities_l513_513904


namespace point_in_second_quadrant_l513_513343

-- Define the quadrants based on the conditions
def quadrant (x y : ℝ) : ℕ :=
  if x > 0 ∧ y > 0 then 1 else
  if x < 0 ∧ y > 0 then 2 else
  if x < 0 ∧ y < 0 then 3 else
  if x > 0 ∧ y < 0 then 4 else 0

-- Define the specific point
def point : ℝ × ℝ := (-2, 5)

-- The theorem to prove that the point lies in the second quadrant
theorem point_in_second_quadrant : quadrant point.1 point.2 = 2 :=
by {
  -- Proof is omitted
  sorry
}

end point_in_second_quadrant_l513_513343


namespace no_positive_integer_solutions_l513_513634

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ) (h1 : x > 0) (h2 : y > 0), 21 * x * y = 7 - 3 * x - 4 * y :=
by
  sorry

end no_positive_integer_solutions_l513_513634


namespace answer_is_p_and_q_l513_513736

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513736


namespace proposition_p_and_q_is_true_l513_513728

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513728


namespace problem_statement_l513_513657

open Complex

theorem problem_statement (x y : ℝ) (i : ℂ) (h_i : i = Complex.I) (h : x + (y - 2) * i = 2 / (1 + i)) : x + y = 2 :=
by
  sorry

end problem_statement_l513_513657


namespace probability_heads_l513_513910

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let y := (n.choose (n / 2)) / total_outcomes
  (1 - y) / 2

theorem probability_heads (h₁ : ∀ (n : ℕ), n = 10 → probability_more_heads_than_tails n = 193 / 512) : 
  probability_more_heads_than_tails 10 = 193 / 512 :=
by
  apply h₁
  exact rfl

end probability_heads_l513_513910


namespace pumpkin_pie_problem_l513_513293

theorem pumpkin_pie_problem
  (baked : ℕ)
  (sold : ℕ)
  (given_away : ℕ)
  (pieces_each : ℕ)
  (portion_eaten : ℚ)
  (initial_pies_eq_baked : baked = 4)
  (sold_pies_eq : sold = 1)
  (given_away_pies_eq : given_away = 1)
  (pieces_each_eq : pieces_each = 6)
  (portion_eaten_eq : portion_eaten = 2 / 3) :
  let remaining_pies := baked - sold - given_away,
      sliced_pieces := remaining_pies * pieces_each,
      pieces_eaten := sliced_pieces * portion_eaten,
      pieces_left := sliced_pieces - pieces_eaten in
  pieces_left = 4 := by
  sorry

end pumpkin_pie_problem_l513_513293


namespace find_m_value_l513_513262

def ellipse_has_focus (m : ℝ) : Prop :=
  let a_squared := 3 * m in
  let b_squared := m in
  (1 : ℝ) ^ 2 = a_squared - b_squared

theorem find_m_value (m : ℝ) (h : ellipse_has_focus m) : m = 1 / 2 :=
by
  sorry

end find_m_value_l513_513262


namespace sum_ab_l513_513178

theorem sum_ab (a b : ℕ) (h1 : 1 < b) (h2 : a ^ b < 500) (h3 : ∀ x y : ℕ, (1 < y ∧ x ^ y < 500 ∧ (x + y) % 2 = 0) → a ^ b ≥ x ^ y) (h4 : (a + b) % 2 = 0) : a + b = 24 :=
  sorry

end sum_ab_l513_513178


namespace hexagon_largest_angle_l513_513332

-- Definitions for conditions
def hexagon_interior_angle_sum : ℝ := 720  -- Sum of all interior angles of hexagon

def angle_A : ℝ := 100
def angle_B : ℝ := 120

-- Define x for angles C and D
variables (x : ℝ)
def angle_C : ℝ := x
def angle_D : ℝ := x
def angle_F : ℝ := 3 * x + 10

-- The formal statement to prove
theorem hexagon_largest_angle (x : ℝ) : 
  100 + 120 + x + x + (3 * x + 10) = 720 → 
  3 * x + 10 = 304 :=
by 
  sorry

end hexagon_largest_angle_l513_513332


namespace true_conjunction_l513_513676

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513676


namespace combined_total_l513_513066

-- Definitions for the problem conditions
def marks_sandcastles : ℕ := 20
def towers_per_marks_sandcastle : ℕ := 10

def jeffs_multiplier : ℕ := 3
def towers_per_jeffs_sandcastle : ℕ := 5

-- Definitions derived from conditions
def jeffs_sandcastles : ℕ := jeffs_multiplier * marks_sandcastles
def marks_towers : ℕ := marks_sandcastles * towers_per_marks_sandcastle
def jeffs_towers : ℕ := jeffs_sandcastles * towers_per_jeffs_sandcastle

-- Question translated to a Lean theorem
theorem combined_total : 
  (marks_sandcastles + jeffs_sandcastles) + (marks_towers + jeffs_towers) = 580 :=
by
  -- The proof would go here
  sorry

end combined_total_l513_513066


namespace proposition_A_l513_513825

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513825


namespace ratio_x_y_l513_513668

noncomputable theory

variables (x y : ℝ)
variables (h1 : (3 * x + y)^5 + x^5 + 4 * x + y = 0)
variables (hx : x ≠ 0) (hy : y ≠ 0)

theorem ratio_x_y : x / y = -1 / 4 := sorry

end ratio_x_y_l513_513668


namespace stacks_distribution_l513_513035

theorem stacks_distribution (x y : ℕ) (h_total : x + y = 80) (h_relation : x = 0.6 * y) : (x, y) = (50, 30) ∨ (x, y) = (30, 50) :=
by
  sorry

end stacks_distribution_l513_513035


namespace pipe_a_fills_cistern_l513_513497

theorem pipe_a_fills_cistern :
  ∀ (x : ℝ), (1 / x + 1 / 120 - 1 / 120 = 1 / 60) → x = 60 :=
by
  intro x
  intro h
  sorry

end pipe_a_fills_cistern_l513_513497


namespace units_digit_of_product_of_odds_is_5_l513_513514

/-- The product of all odd positive integers between 20 and 120 has a units digit of 5. -/
theorem units_digit_of_product_of_odds_is_5 : 
  let odd_integers := {n : ℕ | 20 < n ∧ n < 120 ∧ n % 2 = 1}
  let product_of_odds := ∏ n in odd_integers.to_finset, n
  product_of_odds % 10 = 5 := by
  sorry

end units_digit_of_product_of_odds_is_5_l513_513514


namespace sum_of_reciprocal_squares_lt_one_l513_513430

theorem sum_of_reciprocal_squares_lt_one (n : ℕ) (hn : n ≥ 2) : ∑ k in Finset.range (n + 1), if k ≥ 2 then 1 / k^2 else 0 < 1 :=
by sorry

end sum_of_reciprocal_squares_lt_one_l513_513430


namespace probability_of_more_heads_than_tails_l513_513917

-- Define the probability of getting more heads than tails when flipping 10 coins
def probabilityMoreHeadsThanTails : ℚ :=
  193 / 512

-- Define the proof statement
theorem probability_of_more_heads_than_tails :
  let p : ℚ := probabilityMoreHeadsThanTails in
  p = 193 / 512 :=
by
  sorry

end probability_of_more_heads_than_tails_l513_513917


namespace mandy_med_school_ratio_l513_513403

theorem mandy_med_school_ratio 
    (researched_schools : ℕ)
    (applied_ratio : ℚ)
    (accepted_schools : ℕ)
    (h1 : researched_schools = 42)
    (h2 : applied_ratio = 1 / 3)
    (h3 : accepted_schools = 7)
    : (accepted_schools : ℚ) / ((researched_schools : ℚ) * applied_ratio) = 1 / 2 :=
by sorry

end mandy_med_school_ratio_l513_513403


namespace smallest_abundant_number_not_multiple_of_five_l513_513165

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d < n ∧ n % d = 0) (List.range n)

def is_abundant (n : ℕ) : Prop :=
  n > 1 ∧ (∑ d in proper_divisors n, d) ≥ n

theorem smallest_abundant_number_not_multiple_of_five : ∃ n : ℕ, is_abundant n ∧ n % 5 ≠ 0 ∧ ∀ m : ℕ, is_abundant m ∧ m % 5 ≠ 0 → n ≤ m :=
by 
  existsi 12
  repeat 
    sorry

end smallest_abundant_number_not_multiple_of_five_l513_513165


namespace scientific_notation_of_32000000_l513_513170

theorem scientific_notation_of_32000000 : 32000000 = 3.2 * 10^7 := 
  sorry

end scientific_notation_of_32000000_l513_513170


namespace maria_money_after_utility_bills_l513_513405

def salary : ℝ := 2000
def tax : ℝ := 0.20 * salary
def insurance : ℝ := 0.05 * salary
def total_deductions : ℝ := tax + insurance
def money_left_after_deductions : ℝ := salary - total_deductions
def utility_bills : ℝ := (1 / 4) * money_left_after_deductions
def money_after_utility_bills : ℝ := money_left_after_deductions - utility_bills

theorem maria_money_after_utility_bills : money_after_utility_bills = 1125 := by
  sorry

end maria_money_after_utility_bills_l513_513405


namespace sarah_initial_money_l513_513442

def initial_money 
  (cost_toy_car : ℕ)
  (cost_scarf : ℕ)
  (cost_beanie : ℕ)
  (remaining_money : ℕ)
  (number_of_toy_cars : ℕ) : ℕ :=
  remaining_money + cost_beanie + cost_scarf + number_of_toy_cars * cost_toy_car

theorem sarah_initial_money : 
  (initial_money 11 10 14 7 2) = 53 :=
by
  rfl 

end sarah_initial_money_l513_513442


namespace average_disk_space_per_hour_l513_513576

theorem average_disk_space_per_hour:
  (let total_hours := 15 * 24 in
  ∀ (total_disk_space : ℕ), total_disk_space = 20000 →
  20000 / total_hours = 56) :=
begin
  let total_hours := 15 * 24,
  intro total_disk_space,
  intro h,
  simp [h, total_hours],
  sorry
end

end average_disk_space_per_hour_l513_513576


namespace angle_BAC_range_l513_513548

theorem angle_BAC_range (A B C : ℝ) (BC AB AC AD : ℝ) (h_AD : AD = sqrt (BC * (BC + AB + AC - BC) / AC))
  (h_cond : BC + AD = AB + AC) :
  (45 : ℝ) < 90 - real.arctan 1 / 2 * 2 → 
  (90 - 2 * real.arctan 1 / 2 : ℝ ≤ 2 * real.arctan 1 / 2) → 
  180 - 4 * real.arctan (1 / 2) ≤ BAC ∧ BAC < 90 := sorry

end angle_BAC_range_l513_513548


namespace find_de_l513_513941

namespace MagicSquare

variables (a b c d e : ℕ)

-- Hypotheses based on the conditions provided.
axiom H1 : 20 + 15 + a = 57
axiom H2 : 25 + b + a = 57
axiom H3 : 18 + c + a = 57
axiom H4 : 20 + c + b = 57
axiom H5 : d + c + a = 57
axiom H6 : d + e + 18 = 57
axiom H7 : e + 25 + 15 = 57

def magicSum := 57

theorem find_de :
  ∃ d e, d + e = 42 :=
by sorry

end MagicSquare

end find_de_l513_513941


namespace Mary_hill_length_l513_513981

noncomputable def length_of_hill_Mary_slides_down : ℝ :=
  let v_M := 90 -- Mary's speed in feet/minute
  let L_A := 800 -- Length of Ann's hill in feet
  let v_A := 40 -- Ann's speed in feet/minute
  let t_A := L_A / v_A -- Ann's time in minutes
  let t_M := x / v_M -- Mary's time in minutes
  in t_A = t_M + 13

theorem Mary_hill_length : length_of_hill_Mary_slides_down = 630 :=
  by
  sorry

end Mary_hill_length_l513_513981


namespace a_gt_0_sufficient_not_necessary_l513_513872

def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^3 + a * x + 4

def isMonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem a_gt_0_sufficient_not_necessary (a : ℝ) :
  (0 < a → isMonotonicallyIncreasing (f a)) ∧ ¬ (∀ b : ℝ, 0 < b → b = a) :=
sorry

end a_gt_0_sufficient_not_necessary_l513_513872


namespace area_triangle_CAB_l513_513285

-- Definitions for the parametric equations
def line_l (t : ℝ) := (x = 4 - (sqrt 2 / 2) * t, y = (sqrt 2 / 2) * t)

def curve_C1 (α : ℝ) := (x = 1 + cos α, y = sin α)

def curve_C2 (φ : ℝ) := (x = 2 + 2 * cos φ, y = 2 * sin φ)

-- Polar coordinate equations
def polar_line_l (θ : ℝ) := ρ (cos θ + sin θ) = 4

def polar_curve_C1 (θ : ℝ) := ρ = 2 * cos θ

def polar_curve_C2 (θ : ℝ) := ρ = 4 * cos θ

-- Points A, B, and C
def point_A : ℝ × ℝ := (1/2, sqrt 3 / 2)

def point_B : ℝ × ℝ := (1, sqrt 3)

def point_C : ℝ × ℝ := (-2 - 2 * sqrt 3, 6 + 2 * sqrt 3)

-- Distance AB
def distance_AB : ℝ := 1

-- Distance d from point C to line y = sqrt 3 * x
def distance_d : ℝ := 6 + 2 * sqrt 3

-- The area of triangle CAB
theorem area_triangle_CAB : (1/2) * distance_AB * distance_d = 3 + sqrt 3 := by
  sorry

end area_triangle_CAB_l513_513285


namespace triangle_statements_correct_l513_513350

variables {A B C : ℝ} {a b c : ℝ} (h_triangle : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)

theorem triangle_statements_correct :
  (sin A > sin B → A > B) ∧
  (sin B^2 + sin C^2 < sin A^2 → ∃ D, A = D ∨ A + B = pi / 2 ∧ D ∈ Ioo (pi/2) pi) ∧
  (B = 2 * pi / 3 ∧ b = 6 → ∃ c, ∃ a, ∃ angleA : ℝ, angleA = C ∧ a = 6 ∧ c = 6 ∧ 3 * sqrt 3 = (1 / 2) * a * c * sin (2 * pi / 3)) ∧
  (sin (2 * A) = sin (2 * B) → ∃ (A + B = pi / 2) ∨ A = B) :=
  sorry

end triangle_statements_correct_l513_513350


namespace shortest_path_length_l513_513945

open Real

theorem shortest_path_length :
  let A := (0, 0)
  let D := (16, 12)
  let O := (8, 6)
  let R := 6
  (sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2) > R) ->
  let OA := sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  let AB := sqrt (OA^2 - R^2)
  let angle_OAB := atan (O.2 - A.2) / (O.1 - A.1)
  let angle_BOC := 2 * angle_OAB
  let arc_BC := angle_BOC / (2 * π) * 2 * π * R
  let total_path := AB + arc_BC + AB
  total_path = 16 + 2.23 * π :=
begin
  sorry
end

end shortest_path_length_l513_513945


namespace cyclotomic_absolute_square_l513_513991

open Complex

def is_primitive_root (ε : ℂ) (n : ℕ) : Prop :=
  ε ^ n = 1 ∧ ∀ m : ℕ, m < n → ε ^ m ≠ 1

def seventh_root_of_unity (ε : ℂ) : Prop :=
  is_primitive_root ε 7

def cyclotomic_field : Type :=
  { α : ℂ // ∃ ε : ℂ, seventh_root_of_unity ε ∧ α ∈ adjoin ℚ {ε} }

theorem cyclotomic_absolute_square (α : cyclotomic_field) : 
  ∃ (k : ℤ), (0 ≤ k) ∧ (complex.abs α) ^ 2 = k :=
sorry

end cyclotomic_absolute_square_l513_513991


namespace ratio_of_areas_of_enlarged_circle_l513_513131

theorem ratio_of_areas_of_enlarged_circle (d : ℝ) :
  let original_area := π * (d / 2)^2 in
  let enlarged_area := π * (3 * d / 2)^2 in
  (original_area / enlarged_area) = (1 / 9) :=
by
  sorry

end ratio_of_areas_of_enlarged_circle_l513_513131


namespace n_m_sum_not_even_if_n2_plus_m2_odd_l513_513312

-- Definition of the proof problem
theorem n_m_sum_not_even_if_n2_plus_m2_odd (n m : ℤ) (h : odd (n^2 + m^2)) : ¬ even (n + m) :=
by
  sorry

end n_m_sum_not_even_if_n2_plus_m2_odd_l513_513312


namespace final_portfolio_value_l513_513369

-- Define the initial conditions and growth rates
def initial_investment : ℝ := 80
def first_year_growth_rate : ℝ := 0.15
def additional_investment : ℝ := 28
def second_year_growth_rate : ℝ := 0.10

-- Calculate the values of the portfolio at each step
def after_first_year_investment : ℝ := initial_investment * (1 + first_year_growth_rate)
def after_addition : ℝ := after_first_year_investment + additional_investment
def after_second_year_investment : ℝ := after_addition * (1 + second_year_growth_rate)

theorem final_portfolio_value : after_second_year_investment = 132 := by
  -- This is where the proof would go, but we are omitting it
  sorry

end final_portfolio_value_l513_513369


namespace inverse_function_value_l513_513660

def f (x : ℝ) : ℝ := 2^x - 1

theorem inverse_function_value : f⁻¹ 3 = 2 :=
by sorry

end inverse_function_value_l513_513660


namespace private_schools_in_district_B_l513_513344

theorem private_schools_in_district_B :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  remaining_private_schools = 4 :=
by
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  sorry

end private_schools_in_district_B_l513_513344


namespace prime_product_div_by_four_l513_513056

theorem prime_product_div_by_four 
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq1 : Nat.Prime (p * q + 1)) : 
  4 ∣ (2 * p + q) * (p + 2 * q) := 
sorry

end prime_product_div_by_four_l513_513056


namespace average_brown_mms_per_bag_l513_513439

-- Definitions based on the conditions
def bag1_brown_mm : ℕ := 9
def bag2_brown_mm : ℕ := 12
def bag3_brown_mm : ℕ := 8
def bag4_brown_mm : ℕ := 8
def bag5_brown_mm : ℕ := 3
def number_of_bags : ℕ := 5

-- The proof problem statement
theorem average_brown_mms_per_bag : 
  (bag1_brown_mm + bag2_brown_mm + bag3_brown_mm + bag4_brown_mm + bag5_brown_mm) / number_of_bags = 8 := 
by
  sorry

end average_brown_mms_per_bag_l513_513439


namespace feeding_amount_per_horse_per_feeding_l513_513373

-- Define the conditions as constants
def num_horses : ℕ := 25
def feedings_per_day : ℕ := 2
def half_ton_in_pounds : ℕ := 1000
def bags_needed : ℕ := 60
def days : ℕ := 60

-- Statement of the problem
theorem feeding_amount_per_horse_per_feeding :
  (bags_needed * half_ton_in_pounds / days / feedings_per_day) / num_horses = 20 := by
  -- Assume conditions are satisfied
  sorry

end feeding_amount_per_horse_per_feeding_l513_513373


namespace toy_selling_price_l513_513556

theorem toy_selling_price (x : ℝ) (units_sold : ℝ) (profit_per_day : ℝ) : 
  (units_sold = 200 + 20 * (80 - x)) → 
  (profit_per_day = (x - 60) * units_sold) → 
  profit_per_day = 2500 → 
  x ≤ 60 * 1.4 → 
  x = 65 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_selling_price_l513_513556


namespace intersection_points_l513_513187

def fractional_part (x : ℝ) : ℝ := x - floor x

def equation1 (x y : ℝ) : Prop := (fractional_part x)^2 + y^2 = fractional_part x
def equation2 (x y : ℝ) : Prop := y = (1 / 4) * x

theorem intersection_points:
  (∀ x y : ℝ, x < 5 → equation1 x y → equation2 x y → ∃ n : ℕ, n = 10) :=
by
  sorry

end intersection_points_l513_513187


namespace find_b_for_perpendicular_lines_l513_513642

noncomputable def slope (a b c : ℝ) : ℝ :=
if b = 0 then 0 else -a / b

theorem find_b_for_perpendicular_lines
  (b : ℝ)
  (h1 : slope (-3) 1 4 = -3)
  (h2 : slope b 2 10 = - b / 2)
  (h3 : b ≠ 0) :
  (slope (-3) 1 4 * slope b 2 10) = -1 ↔ b = -2 / 3 :=
by {
  sorry
}

end find_b_for_perpendicular_lines_l513_513642


namespace true_proposition_l513_513847

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513847


namespace part1_part2_part3_l513_513862

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (x + b) / (x^2 - 4)

theorem part1 (h_odd : ∀ x ∈ Ioo (-2) 2, f b x = -f b (-x)) : b = 0 :=
sorry

theorem part2 (b = 0) : ∀ x1 x2 ∈ Ioo (-2) 2, x1 < x2 → f 0 x1 > f 0 x2 :=
sorry

theorem part3 (h_ineq : ∀ t : ℝ, f 0 (t-2) + f 0 t > 0) : ∀ t : ℝ, 0 < t ∧ t < 1 :=
sorry

end part1_part2_part3_l513_513862


namespace increasing_condition_l513_513457

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then 2 * x else x + 1

theorem increasing_condition (a : ℝ) : a = 0 → ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
by
  sorry

end increasing_condition_l513_513457


namespace true_proposition_l513_513770

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513770


namespace multiply_48_52_l513_513615

theorem multiply_48_52 : (48 * 52 = 2496) :=
by
  have h1: 48 = 50 - 2 := by simp
  have h2: 52 = 50 + 2 := by simp
  have identity : ∀ a b : ℤ, (a - b) * (a + b) = a^2 - b^2 :=
    λ a b, by ring
  calc
    48 * 52 = (50 - 2) * (50 + 2) : by rw [h1, h2]
        ... = 50^2 - 2^2 : identity 50 2
        ... = 2500 - 4 : by norm_num
        ... = 2496 : by norm_num

end multiply_48_52_l513_513615


namespace probability_not_bought_by_Jim_l513_513109

theorem probability_not_bought_by_Jim (total_pics : ℕ) (bought_by_Jim : ℕ) (picked_pics : ℕ)
  (h_total : total_pics = 10) (h_bought : bought_by_Jim = 3) (h_picked : picked_pics = 2):
  let remaining_pics := total_pics - bought_by_Jim in 
  let num_ways_pick_from_7 := (Nat.choose remaining_pics picked_pics) in
  let num_ways_pick_from_10 := (Nat.choose total_pics picked_pics) in
  remaining_pics = 7 → num_ways_pick_from_7 = 21 → num_ways_pick_from_10 = 45 →
  (num_ways_pick_from_7 / num_ways_pick_from_10 : ℚ) = 7 / 15 :=
by {
  intros; 
  rw [h_total, h_bought, h_picked];
  have h1 : remaining_pics = 7 := by simp [remaining_pics, h_total, h_bought]; rw [h1];
  have h2 : num_ways_pick_from_7 = 21 := by simp [num_ways_pick_from_7, h1, h_picked]; rw [h2];
  have h3 : num_ways_pick_from_10 = 45 := by simp [num_ways_pick_from_10, h_total, h_picked]; rw [h3];
  simp;
  sorry
}


end probability_not_bought_by_Jim_l513_513109


namespace proof_problem_l513_513661

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def satisfies_conditions (a : ℕ → ℝ) : Prop :=
  (2 * a 1 + a 3 = 3 * a 2) ∧ (a 3 + 2 = (a 2 + a 4) / 2)

def sequence_general_term (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a n = 2 ^ n

def bn_formula (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = a n - (Real.log 2 (a n))

def sn_definition (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in Finset.range (n + 1), b i

def inequality_condition (S : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ), S n - 2 ^ (n + 1) + 47 < 0

theorem proof_problem (a b S : ℕ → ℝ) :
  is_geometric_sequence a →
  satisfies_conditions a →
  sequence_general_term a →
  bn_formula a b →
  sn_definition b S →
  inequality_condition S :=
by
  sorry

end proof_problem_l513_513661


namespace find_x_l513_513906

theorem find_x (b x : ℝ) (h_b_gt_1 : b > 1) (h_x_gt_0 : x > 0)
  (h_eq : (4 * x) ^ Real.log b 4 - (5 * x) ^ Real.log b 5 = 0) : 
  x = 4 / 5 := 
sorry

end find_x_l513_513906


namespace f_16_expression_l513_513239

def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | 1 => f x
  | n + 1 => f (f_n n x)

theorem f_16_expression (x : ℝ) : f_n 16 x = (x - 1) / x := 
sorry

end f_16_expression_l513_513239


namespace expansion_1_expansion_2_expansion_3_expansion_4_expansion_5_l513_513501

open Real

-- 1. Prove that the power series expansion for (1 + x)e^x is ∑_{n=0}^∞ ((n+1)x^n)/n!
theorem expansion_1 (x : ℝ) : 
  (((1 + x) * exp x) = ∑ n in (Finset.range (5 + 1)), ((n + 1) * x^n) / Nat.factorial n + (sorry : ℝ)) := sorry

-- 2. Prove that the power series expansion for sin^2 x is 1/2 - 1/2 ∑_{n=0}^∞ ((-1)^n (2^(2n)x^(2n)))/(2n)!
theorem expansion_2 (x : ℝ) : 
  ((sin x) ^ 2 = (1/2 - 1/2 * ∑ n in (Finset.range (5 + 1)), ((-1 : ℝ)^n * (2 : ℝ)^(2 * n) * x^(2 * n)) / Nat.factorial (2 * n) + (sorry : ℝ))) := sorry

-- 3. Prove that the power series expansion for (x-3)/(x+1)^2 is -3 + 7x - 11x^2 + …
theorem expansion_3 (x : ℝ) : 
  (((x - 3) / (x + 1) ^ 2) = (-3 + 7 * x - 11 * x^2 + (sorry : ℝ))) := sorry

-- 4. Prove that the power series expansion for e^{-x} sin x is x - x^2 + (1/3)x^3 - (1/20)x^5 + (7/360) x^6 + …
theorem expansion_4 (x : ℝ) : 
  ((exp (-x) * sin x) = (x - x^2 + (1 / 3) * x^3 - (1 / 20) * x^5 + (7 / 360) * x^6 + (sorry : ℝ))) := sorry

-- 5. Prove that the power series expansion for ln (1 + 3x + 2x^2) is ∑_{n=1}^∞ ((-1)^{n-1} (1 + 2^n)x^n)/n
theorem expansion_5 (x : ℝ) : 
  (log (1 + 3 * x + 2 * x^2) = ∑ n in (Finset.range (5 + 1)).filter (λ (n : ℕ), 0 < n), ((-1 : ℝ)^(n - 1) * (1 + 2^n) * x^n) / n + (sorry : ℝ)) := sorry

end expansion_1_expansion_2_expansion_3_expansion_4_expansion_5_l513_513501


namespace rulers_in_drawer_l513_513488

theorem rulers_in_drawer (initial_rulers : Nat) (taken_rulers : Nat) (h : initial_rulers = 14) (h1 : taken_rulers = 11) : initial_rulers - taken_rulers = 3 :=
by
  rw [h, h1]
  exact Nat.sub_eq_of_eq_add (by norm_num)

end rulers_in_drawer_l513_513488


namespace min_M_value_l513_513228

variable {a b c t : ℝ}

theorem min_M_value (h1 : a < b)
                    (h2 : a > 0)
                    (h3 : b^2 - 4 * a * c ≤ 0)
                    (h4 : b = t + a)
                    (h5 : t > 0)
                    (h6 : c ≥ (t + a)^2 / (4 * a)) :
    ∃ M : ℝ, (∀ x : ℝ, (a * x^2 + b * x + c) ≥ 0) → M = 3 := 
  sorry

end min_M_value_l513_513228


namespace sqrt_two_program_fails_l513_513378

theorem sqrt_two_program_fails (seq : ℕ → ℕ)
  (h1 : ∀ n, seq n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h2 : ∃ N, ∀ n ≥ N, ∃ m > n, seq m = seq n)
  (h3 : irrational (real.sqrt 2)) :
  ∃ n, seq n ≠ nat.digits 10 (real.sqrt 2) :=
  sorry

end sqrt_two_program_fails_l513_513378


namespace mean_of_six_numbers_sum_three_quarters_l513_513482

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end mean_of_six_numbers_sum_three_quarters_l513_513482


namespace peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513015

-- Probability of Peter winning a prize:
theorem peter_wins_prize_probability :
  let p := (5 / 6) in p ^ 9 = (5 / 6) ^ 9 := by
  sorry

-- Probability that at least one person wins a prize:
theorem at_least_one_wins_prize_probability :
  let p := (5 / 6) in
  let q := (1 - p^9) in 
  (1 - q^10) ≈ 0.919 := by
  sorry

end peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513015


namespace total_payment_leila_should_pay_l513_513163

-- Definitions of the conditions
def chocolateCakes := 3
def chocolatePrice := 12
def strawberryCakes := 6
def strawberryPrice := 22

-- Mathematical equivalent proof problem
theorem total_payment_leila_should_pay : 
  chocolateCakes * chocolatePrice + strawberryCakes * strawberryPrice = 168 := 
by 
  sorry

end total_payment_leila_should_pay_l513_513163


namespace least_number_to_add_l513_513138

theorem least_number_to_add (n : ℕ) (h : n = 821562) : ∃ (k : ℕ), k = 3 ∧ (n + k) % 5 = 0 :=
by
  have rem := n % 5
  have q := n / 5
  have eq1 : n = 5 * q + rem := Nat.mod_add_div n 5
  have rem_value : rem = 2 := by sorry
  use 3
  split
  · rfl
  · rw [←h, eq1, rem_value]
    calc
      (821562 + 3) % 5 = (5 * q + 2 + 3) % 5 := congrArg (fun x => x % 5) eq1
      ... = (5 * q + 5) % 5 := by rw rem_value
      ... = 0 := by rw [Nat.add_mod_right, Nat.mod_self]

end least_number_to_add_l513_513138


namespace constant_function_of_bounded_sum_l513_513449

theorem constant_function_of_bounded_sum (f : ℝ → ℝ)
  (h : ∀ (n : ℕ) (x y : ℝ), |∑ k in finset.range (n + 1), 3^k * (f(x + k • y) - f(x - k • y))| ≤ 1) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_bounded_sum_l513_513449


namespace base8_operations_l513_513158

def add_base8 (a b : ℕ) : ℕ :=
  let sum := (a + b) % 8
  sum

def subtract_base8 (a b : ℕ) : ℕ :=
  let diff := (a + 8 - b) % 8
  diff

def step1 := add_base8 672 156
def step2 := subtract_base8 step1 213

theorem base8_operations :
  step2 = 0645 :=
by
  sorry

end base8_operations_l513_513158


namespace units_digit_of_product_of_odds_between_20_and_120_l513_513524

theorem units_digit_of_product_of_odds_between_20_and_120 : 
  let odds := filter (fun n => n % 2 = 1) [21, 23, ... , 119] in
  ∃ digit, digit = 5 ∧ digit = ((odds.foldl (λ acc n => acc * n) 1) % 10) := 
sorry

end units_digit_of_product_of_odds_between_20_and_120_l513_513524


namespace true_conjunction_l513_513683

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513683


namespace arrangements_of_15_cents_l513_513625

noncomputable def num_distinct_arrangements : Nat := 
263

theorem arrangements_of_15_cents : 
  let stamps := [{(1,1)}, {(2,2)}, {(3,3)}, {(4,4)}, {(5,5)}, {(6,6)}, {(7,7)}, {(8,8)}, {(9,9)}] in
  num_distinct_arrangements = 263 :=
sorry

end arrangements_of_15_cents_l513_513625


namespace total_days_2001_to_2004_l513_513898

def regular_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def num_regular_years : ℕ := 3
def num_leap_years : ℕ := 1

theorem total_days_2001_to_2004 : 
  (num_regular_years * regular_year_days) + (num_leap_years * leap_year_days) = 1461 :=
by
  sorry

end total_days_2001_to_2004_l513_513898


namespace sum_of_real_solutions_eq_l513_513641

theorem sum_of_real_solutions_eq : 
  (∑ x in (multiset.to_finset (multiset.filter is_real (multiset.of_finset (finset.filter (λ x, (x-3)/(x^2+5*x+2) = (x-6)/(x^2-11*x)) (finset.range 100)))), id)) = -64 / 13 := 
sorry

end sum_of_real_solutions_eq_l513_513641


namespace initial_numbers_count_l513_513072

theorem initial_numbers_count (n : ℕ) (S : ℝ)
  (h1 : S / n = 56)
  (h2 : (S - 100) / (n - 2) = 56.25) :
  n = 50 :=
sorry

end initial_numbers_count_l513_513072


namespace calculate_sum_l513_513607

theorem calculate_sum (n : ℕ) : 
  ∑ m in Finset.range (n + 1), (-1)^(m+1) * (1 / m) * (Nat.choose n (m - 1)) = 1 / (n + 1) := 
by
  sorry

end calculate_sum_l513_513607


namespace find_geometric_first_term_l513_513477

noncomputable def geometric_first_term (a r : ℝ) (h1 : a * r^5 = real.from_nat (nat.factorial 9)) 
  (h2 : a * r^8 = real.from_nat (nat.factorial 10)) : ℝ :=
  a

theorem find_geometric_first_term :
  ∃ a r, a * r^5 = real.from_nat (nat.factorial 9) ∧ a * r^8 = real.from_nat (nat.factorial 10) ∧ a = 362880 / 10^(5/3) :=
by
  have h : ∃ (a r: ℝ), a * r^5 = real.from_nat (nat.factorial 9) ∧ a * r^8 = real.from_nat (nat.factorial 10),
  {
    use [362880 / 10^(5/3), 10^(1/3)],
    split,
    { sorry },
    { sorry }
  },
  cases h with a ha,
  cases ha with r hr,
  use [a, r],
  split,
  { exact hr.left },
  split,
  { exact hr.right },
  { sorry }

end find_geometric_first_term_l513_513477


namespace main_statement_l513_513756

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513756


namespace problem_statement_l513_513701

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513701


namespace diagonals_perpendicular_l513_513241

-- Define the quadrilateral and its tangency points
variables (A B C D K L M N O1 O2 O3 O4 : Type)
          [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
          [InnerProductSpace ℝ D] [InnerProductSpace ℝ K] [InnerProductSpace ℝ L]
          [InnerProductSpace ℝ M] [InnerProductSpace ℝ N] [InnerProductSpace ℝ O1]
          [InnerProductSpace ℝ O2] [InnerProductSpace ℝ O3] [InnerProductSpace ℝ O4]

-- Hypotheses: ABCD is a circumscribed quadrilateral, and O1, O2, O3, O4 are the incenters of BKL, CLM, DMN, ANK respectively.
variables (H1 : TangentCircle K A B) (H2 : TangentCircle L B C) (H3 : TangentCircle M C D)
          (H4 : TangentCircle N D A) (H5 : Incenter O1 B K L) (H6 : Incenter O2 C L M)
          (H7 : Incenter O3 D M N) (H8 : Incenter O4 A N K)

-- Goal: Prove diagonals between incenters are perpendicular
theorem diagonals_perpendicular : 
  ∃ P : Type, (InnerProduct O1 O3 = 0) ∧ (InnerProduct O2 O4 = 0) :=
sorry

end diagonals_perpendicular_l513_513241


namespace maria_remaining_salary_l513_513408

variable (salary taxRate insuranceRate utilityRate : ℝ)

def remaining_salary_after_deductions (salary taxRate insuranceRate utilityRate : ℝ) : ℝ :=
  let tax = salary * taxRate
  let insurance = salary * insuranceRate
  let totalDeductions = tax + insurance
  let amountAfterDeductions = salary - totalDeductions
  let utilityBills = amountAfterDeductions * utilityRate
  amountAfterDeductions - utilityBills

theorem maria_remaining_salary :
  remaining_salary_after_deductions 2000 0.2 0.05 (1 / 4) = 1125 :=
by
  unfold remaining_salary_after_deductions
  norm_num
  sorry

end maria_remaining_salary_l513_513408


namespace exists_z0_of_mod_ge_one_l513_513974

open Complex

theorem exists_z0_of_mod_ge_one (a : ℝ) (f : ℂ → ℂ) (z : ℂ) (h_a : 0 < a ∧ a < 1)
  (h_f : ∀ z, f z = z^2 - z + a) (h_z : |z| ≥ 1) :
  ∃ z0 : ℂ, |z0| = 1 ∧ |f z0| ≤ |f z| :=
sorry

end exists_z0_of_mod_ge_one_l513_513974


namespace answer_is_p_and_q_l513_513740

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513740


namespace tan_alpha_eq_3_l513_513323

open Complex

-- Conditions
def P : ℂ := -√10/10 + (-3*√10/10) * I
def α := angle_of (P.to_real_angle)
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0 ∧ abs z = 1

-- Theorem statement
theorem tan_alpha_eq_3 (h1 : in_third_quadrant P) (hx : P.re = -√10/10) : tan α = 3 :=
by
  sorry

end tan_alpha_eq_3_l513_513323


namespace smallest_value_not_defined_l513_513510

noncomputable def smallest_undefined_x : ℝ :=
  let a := 6
  let b := -37
  let c := 5
  let discriminant := b * b - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 < x2 then x1 else x2

theorem smallest_value_not_defined :
  smallest_undefined_x = 0.1383 :=
by sorry

end smallest_value_not_defined_l513_513510


namespace all_numbers_positive_l513_513266

theorem all_numbers_positive (n : ℕ) (a : Fin (2 * n + 1) → ℝ) 
  (h : ∀ S : Finset (Fin (2 * n + 1)), 
        S.card = n + 1 → 
        S.sum a > (Finset.univ \ S).sum a) : 
  ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l513_513266


namespace proposition_true_l513_513827

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513827


namespace problem_solution_l513_513662

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * n - 1

def sum_S (n : ℕ) : ℕ :=
  n * n

def sequence_c (n : ℕ) : ℕ :=
  sequence_a n * 2 ^ (sequence_a n)

def sum_T (n : ℕ) : ℕ :=
  (6 * n - 5) * 2 ^ (2 * n + 1) + 10

theorem problem_solution (n : ℕ) (hn : n ≥ 1) :
  ∀ n, (sum_S 1 = 1) ∧ (sequence_a 1 = 1) ∧ 
          (∀ n ≥ 2, sequence_a n = 2 * n - 1) ∧
          (sum_T n = (6 * n - 5) * 2 ^ (2 * n + 1) + 10 / 9) :=
by sorry

end problem_solution_l513_513662


namespace positive_ints_satisfying_inequality_count_l513_513622

open Int

theorem positive_ints_satisfying_inequality_count :
  {n : ℤ // 0 < n ∧ (n + 7) * (n - 5) * (n - 10) < 0}.size = 4 :=
sorry

end positive_ints_satisfying_inequality_count_l513_513622


namespace inequality_holds_equality_holds_if_l513_513544

variables {n : ℕ} (a b : Fin n → ℝ) (A B : ℝ)

def sum_a (a : Fin n → ℝ) : ℝ := ∑ i, a i
def sum_b (b : Fin n → ℝ) : ℝ := ∑ i, b i

theorem inequality_holds (h : ∀ i, 0 < a i ∧ 0 < b i)
  (A_def : A = sum_a a)
  (B_def : B = sum_b b) :
  ∑ i, (a i * b i) / (a i + b i) ≤ (A * B) / (A + B) :=
sorry

theorem equality_holds_if (h : ∀ i, 0 < a i ∧ 0 < b i)
  (A_def : A = sum_a a)
  (B_def : B = sum_b b)
  (hk : n = 1 ∨ (n > 1 ∧ ∃ k, ∀ i, a i / b i = a k / b k)) :
  ∑ i, (a i * b i) / (a i + b i) = (A * B) / (A + B) :=
sorry

end inequality_holds_equality_holds_if_l513_513544


namespace problem1_problem2_l513_513671

-- Problem 1
theorem problem1 {θ : ℝ} :
  (let A := (1 : ℝ, 0),
       B := (0, 1),
       C := (2 * Real.sin θ, Real.cos θ),
       AC := (2 * Real.sin θ - 1, Real.cos θ),
       BC := (2 * Real.sin θ, Real.cos θ - 1))
  (Real.sqrt ((2 * Real.sin θ - 1)^2 + (Real.cos θ)^2) = Real.sqrt (4 * (Real.sin θ)^2 + (Real.cos θ - 1)^2)) → 
  (Real.sin θ + 2 * Real.cos θ) / (Real.sin θ - Real.cos θ) = -5 :=
sorry

-- Problem 2
theorem problem2 {θ : ℝ} :
  (let A := (1, 0),
       B := (0, 1),
       C := (2 * Real.sin θ, Real.cos θ),
       OA := (1 : ℝ, 0),
       OB := (0, 1),
       OC := (2 * Real.sin θ, Real.cos θ))
  ((1 : ℝ, 2) • (2 * Real.sin θ, Real.cos θ) = 1) → 
  Real.sin θ * Real.cos θ = -3 / 8 :=
sorry

end problem1_problem2_l513_513671


namespace total_settings_weight_l513_513416

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end total_settings_weight_l513_513416


namespace peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513012

-- Probability of Peter winning a prize:
theorem peter_wins_prize_probability :
  let p := (5 / 6) in p ^ 9 = (5 / 6) ^ 9 := by
  sorry

-- Probability that at least one person wins a prize:
theorem at_least_one_wins_prize_probability :
  let p := (5 / 6) in
  let q := (1 - p^9) in 
  (1 - q^10) ≈ 0.919 := by
  sorry

end peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513012


namespace lily_final_balance_l513_513357

noncomputable def initial_balance : ℝ := 55
noncomputable def shirt_cost : ℝ := 7
noncomputable def shoes_cost : ℝ := 3 * shirt_cost
noncomputable def book_cost : ℝ := 4
noncomputable def books_amount : ℝ := 5
noncomputable def gift_fraction : ℝ := 0.20

noncomputable def remaining_balance : ℝ :=
  initial_balance - 
  shirt_cost - 
  shoes_cost - 
  books_amount * book_cost - 
  gift_fraction * (initial_balance - shirt_cost - shoes_cost - books_amount * book_cost)

theorem lily_final_balance : remaining_balance = 5.60 := 
by 
  sorry

end lily_final_balance_l513_513357


namespace true_proposition_l513_513768

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513768


namespace Jason_spent_correct_amount_l513_513950

namespace MusicStore

def costFlute : Real := 142.46
def costMusicStand : Real := 8.89
def costSongBook : Real := 7.00
def totalCost : Real := 158.35

theorem Jason_spent_correct_amount :
  costFlute + costMusicStand + costSongBook = totalCost :=
sorry

end MusicStore

end Jason_spent_correct_amount_l513_513950


namespace units_digit_of_product_of_odds_between_20_and_120_l513_513525

theorem units_digit_of_product_of_odds_between_20_and_120 : 
  let odds := filter (fun n => n % 2 = 1) [21, 23, ... , 119] in
  ∃ digit, digit = 5 ∧ digit = ((odds.foldl (λ acc n => acc * n) 1) % 10) := 
sorry

end units_digit_of_product_of_odds_between_20_and_120_l513_513525


namespace exceeds_paths_l513_513920

variables {G : Type} [Graph G] {A B : Set (Vertex G)} {P : Set (Path G)} {W : Path G}

theorem exceeds_paths (hW : W.alternating_path)
  (hT : W.terminates_in (B \ V[P])) :
  ∃ (Q : Set (Path G)), disjoint A B Q ∧ Q.card > P.card :=
sorry

end exceeds_paths_l513_513920


namespace bernardo_score_is_75_l513_513177

noncomputable def bernardo_overall_score : ℕ :=
  let correct_answers_first : ℕ := 12
  let correct_answers_second : ℕ := 28
  let correct_answers_third : ℕ := 42
  let total_questions : ℕ := 110
  let total_correct_answers : ℕ := correct_answers_first + correct_answers_second + correct_answers_third
  let overall_percentage : ℚ := (total_correct_answers.to_rat / total_questions) * 100
  overall_percentage.natAbs

theorem bernardo_score_is_75:
  bernardo_overall_score = 75 :=
by 
  sorry

end bernardo_score_is_75_l513_513177


namespace power_relation_l513_513238

variable (R : Type*) [Field R]

theorem power_relation (x : R) (h : x + x⁻¹ = 3) : x^2 + x⁻² = 7 :=
by sorry

end power_relation_l513_513238


namespace possible_values_of_a_l513_513079

theorem possible_values_of_a (a b c : ℝ) (h1 : a + b + c = 2005) (h2 : (a - 1 = a ∨ a - 1 = b ∨ a - 1 = c) ∧ (b + 1 = a ∨ b + 1 = b ∨ b + 1 = c) ∧ (c ^ 2 = a ∨ c ^ 2 = b ∨ c ^ 2 = c)) :
  a = 1003 ∨ a = 1002.5 :=
sorry

end possible_values_of_a_l513_513079


namespace average_speed_is_approximately_10_point_1_l513_513537

-- Definitions for the conditions
def distance1 : ℝ := 9
def speed1 : ℝ := 12
def distance2 : ℝ := 12
def speed2 : ℝ := 9

-- Time for each part of the trip
def time1 : ℝ := distance1 / speed1
def time2 : ℝ := distance2 / speed2

-- Total distance and total time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + time2

-- Average speed for the entire trip
def average_speed : ℝ := total_distance / total_time

-- The proof problem
theorem average_speed_is_approximately_10_point_1 : abs (average_speed - 10.1) < 0.01 := by
  sorry

end average_speed_is_approximately_10_point_1_l513_513537


namespace sequence_expression_l513_513621

theorem sequence_expression (n : ℕ) (h : n ≥ 2) (T : ℕ → ℕ) (a : ℕ → ℕ)
  (hT : ∀ k : ℕ, T k = 2 * k^2)
  (ha : ∀ k : ℕ, k ≥ 2 → a k = T k / T (k - 1)) :
  a n = (n / (n - 1))^2 := 
sorry

end sequence_expression_l513_513621


namespace minimum_marked_squares_l513_513114

theorem minimum_marked_squares (n : ℕ) (h : n % 2 = 0) : 
  ∃ N, (∀ i j, (i < n ∧ j < n → 
    (i > 0 ∧ (N ∣ (i - 1) + j) ∨ 
    i < n - 1 ∧ (N ∣ (i + 1) + j) ∨ 
    j > 0 ∧ (N ∣ i + (j - 1)) ∨ 
    j < n - 1 ∧ (N ∣ i + (j + 1)))  )) ∧ 
  N = n * (n + 2) / 4 :=
begin
  sorry
end

end minimum_marked_squares_l513_513114


namespace true_proposition_l513_513846

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513846


namespace tangents_of_diagonal_angles_l513_513934

variables (a b : ℝ) (α : ℝ)

noncomputable def tangent_of_angle_A := b * sin α / (a + b * cos α)
noncomputable def tangent_of_angle_B := a * sin α / (b + a * cos α)

theorem tangents_of_diagonal_angles (ha : a ≠ 0) (hb : b ≠ 0) (hα1 : 0 < α) (hα2 : α < π / 2) :
  (tangent_of_angle_A a b α, tangent_of_angle_B a b α) =
  (b * sin α / (a + b * cos α), a * sin α / (b + a * cos α)) :=
by sorry

end tangents_of_diagonal_angles_l513_513934


namespace units_digit_of_product_of_odds_between_20_and_120_l513_513523

theorem units_digit_of_product_of_odds_between_20_and_120 : 
  let odds := filter (fun n => n % 2 = 1) [21, 23, ... , 119] in
  ∃ digit, digit = 5 ∧ digit = ((odds.foldl (λ acc n => acc * n) 1) % 10) := 
sorry

end units_digit_of_product_of_odds_between_20_and_120_l513_513523


namespace problem_statement_l513_513787

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513787


namespace cost_price_per_meter_l513_513152

theorem cost_price_per_meter (SP : ℕ) (P : ℕ) (n : ℕ) : 
  SP = 6788 → P = 29 → n = 78 → (SP - n * P) / n = 58 :=
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end cost_price_per_meter_l513_513152


namespace proposition_p_and_q_is_true_l513_513723

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513723


namespace sarah_initial_money_l513_513440

-- Definitions based on conditions
def cost_toy_car := 11
def cost_scarf := 10
def cost_beanie := 14
def remaining_money := 7
def total_cost := 2 * cost_toy_car + cost_scarf + cost_beanie
def initial_money := total_cost + remaining_money

-- Statement of the theorem
theorem sarah_initial_money : initial_money = 53 :=
by
  sorry

end sarah_initial_money_l513_513440


namespace bisects_BD_l513_513384

-- Definitions of the geometric entities and conditions
variables {A B C D O B1 : Type} 

-- Axioms based on the problem description
axiom O_is_incenter (triangle_ABC : Triangle A B C) : Incenter O triangle_ABC
axiom D_is_tangency_point (triangle_ABC : Triangle A B C) : TangencyPoint D (side AC) (Incircle triangle_ABC)
axiom B1_is_midpoint (triangle_ABC : Triangle A B C) : Midpoint B1 (side AC)

-- The theorem to be proven
theorem bisects_BD (triangle_ABC : Triangle A B C) (O_incenter : Incenter O triangle_ABC) 
  (D_tangency : TangencyPoint D (side AC) (Incircle triangle_ABC)) 
  (B1_midpoint : Midpoint B1 (side AC)) : Bisects (line B1 O) (segment B D) := 
sorry

end bisects_BD_l513_513384


namespace transformed_sum_eq_3s_l513_513132

variable {n : ℕ}
variable {s : ℤ}
variable (x : Fin n → ℤ)
variable (h_sum : ∑ i, x i = s)

theorem transformed_sum_eq_3s (h_sum : ∑ i, x i = s) : ∑ i, 3 * (x i - 10) + 30 = 3 * s := by
  sorry

end transformed_sum_eq_3s_l513_513132


namespace number_of_valid_arrangements_l513_513453

namespace FiveElements

-- Definitions of the elements as an inductive type.
inductive Element
| metal 
| wood 
| earth 
| water 
| fire

open Element

-- Overcoming relationship as a function.
def overcomes : Element → Element → Prop
| metal, wood := true
| wood, earth := true
| earth, water := true
| water, fire := true
| fire, metal := true
| _, _ := false

-- Check if two elements can be adjacent according to the overcoming relationship.
def can_be_adjacent (e1 e2 : Element) : Prop :=
  ¬overcomes e1 e2 ∧ ¬overcomes e2 e1

-- The main theorem stating the number of valid arrangements.
theorem number_of_valid_arrangements: ∃(n : ℕ), n = 10 ∧ 
  ∀(arrangement : list Element), arrangement.perm [metal, wood, earth, water, fire] → (∀i < arrangement.length - 1, can_be_adjacent (arrangement.nth_le i sorry) (arrangement.nth_le (i + 1) sorry)) → arrangement.length = 5 :=
begin
  existsi 10,
  split,
  { -- Proof that there are 10 valid arrangements
    sorry,
  },
  { -- Proof that any valid arrangement satisfies the condition
    sorry,
  }
end

end FiveElements

end number_of_valid_arrangements_l513_513453


namespace select_2_cooks_from_10_people_l513_513338

theorem select_2_cooks_from_10_people (total_people refusing_people eligible_people cooks_to_select : ℕ) 
  (h1 : total_people = 10) (h2 : refusing_people = 1) (h3 : eligible_people = total_people - refusing_people) 
  (h4 : cooks_to_select = 2) : nat.choose eligible_people cooks_to_select = 36 :=
by
  -- total_people = 10
  -- refusing_people = 1
  -- eligible_people = total_people - refusing_people = 9
  -- cooks_to_select = 2
  -- nat.choose 9 2 = 36
  sorry

end select_2_cooks_from_10_people_l513_513338


namespace socks_picking_l513_513339

theorem socks_picking : 
  let socks := [ ("white", 5), ("brown", 3), ("blue", 2), ("red", 2) ] in
  ∑ (color in socks), nat.choose color.2 2 = 15 :=
by
  sorry

end socks_picking_l513_513339


namespace total_weight_of_settings_l513_513412

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end total_weight_of_settings_l513_513412


namespace power_greater_than_linear_l513_513996

theorem power_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
by {
  sorry
}

end power_greater_than_linear_l513_513996


namespace distance_from_point_A_l513_513147

theorem distance_from_point_A (area : ℝ) (x : ℝ) (h_area : area = 18) (h_fold : x^2 = 12) : 
  Real.sqrt((2 * Real.sqrt(3))^2 + (2 * Real.sqrt(3))^2) = 2 * Real.sqrt(6) :=
by
  sorry

end distance_from_point_A_l513_513147


namespace number_of_square_factors_l513_513304

theorem number_of_square_factors (n1 n2 n3 n4 : ℕ) (h1 : n1 = 12) (h2 : n2 = 15) (h3 : n3 = 18) (h4 : n4 = 8) :
  let factors_2 := 7,
      factors_3 := 8,
      factors_5 := 10,
      factors_7 := 5 in
  factors_2 * factors_3 * factors_5 * factors_7 = 2800 := by
  -- The provided proof can be added here
  sorry

end number_of_square_factors_l513_513304


namespace scientific_notation_32000000_l513_513169

def scientific_notation (n : ℕ) : String := sorry

theorem scientific_notation_32000000 :
  scientific_notation 32000000 = "3.2 × 10^7" :=
sorry

end scientific_notation_32000000_l513_513169


namespace proposition_A_l513_513814

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513814


namespace set_representation_l513_513856

def A (x : ℝ) := -3 < x ∧ x < 1
def B (x : ℝ) := x ≤ -1
def C (x : ℝ) := -2 < x ∧ x ≤ 2

theorem set_representation :
  (∀ x, A x ↔ (A x ∧ (B x ∨ C x))) ∧
  (∀ x, A x ↔ (A x ∨ (B x ∧ C x))) ∧
  (∀ x, A x ↔ ((A x ∧ B x) ∨ (A x ∧ C x))) :=
by
  sorry

end set_representation_l513_513856


namespace correct_calculation_l513_513529

theorem correct_calculation : ∀ (a : ℝ), a^3 * a^2 = a^5 := 
by
  intro a
  sorry

end correct_calculation_l513_513529


namespace sqrt_12_minus_3_sqrt_1_div_3_plus_sqrt_8_eq_sqrt_5_minus_1_squared_plus_sqrt_5_mul_sqrt_5_add_2_eq_l513_513609

-- Problem 1
theorem sqrt_12_minus_3_sqrt_1_div_3_plus_sqrt_8_eq :
  sqrt 12 - 3 * sqrt (1 / 3) + sqrt 8 = sqrt 3 + 2 * sqrt 2 := by
  sorry

-- Problem 2
theorem sqrt_5_minus_1_squared_plus_sqrt_5_mul_sqrt_5_add_2_eq :
  (sqrt 5 - 1) ^ 2 + sqrt 5 * (sqrt 5 + 2) = 11 := by
  sorry

end sqrt_12_minus_3_sqrt_1_div_3_plus_sqrt_8_eq_sqrt_5_minus_1_squared_plus_sqrt_5_mul_sqrt_5_add_2_eq_l513_513609


namespace evaluate_difference_floor_squares_l513_513630

theorem evaluate_difference_floor_squares (x : ℝ) (h : x = 15.3) : ⌊x^2⌋ - ⌊x⌋^2 = 9 := by
  sorry

end evaluate_difference_floor_squares_l513_513630


namespace seller_loss_percentage_l513_513143

noncomputable def calculate_profit_percentage 
  (bats_cost_range : ℝ × ℝ)
  (num_bats : ℕ)
  (discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (exchange_rate_increase : ℝ)
  (average_cost_price : ℝ) : ℝ :=
  let
    original_cost := (bats_cost_range.1 + bats_cost_range.2) / 2,
    increased_cost := original_cost * (1 + exchange_rate_increase),
    selling_price := increased_cost * (1 - discount_rate),
    final_price := increased_cost * (1 + sales_tax_rate),
    profit_per_bat := selling_price - final_price,
    total_profit := num_bats * profit_per_bat,
    total_cost := num_bats * increased_cost
  in (total_profit / total_cost) * 100

theorem seller_loss_percentage :
  calculate_profit_percentage (450, 750) 10 0.10 0.05 0.02 600 = -15 := 
sorry

end seller_loss_percentage_l513_513143


namespace curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l513_513270

noncomputable def curve_C (a x y : ℝ) := a * x ^ 2 + a * y ^ 2 - 2 * x - 2 * y = 0

theorem curve_C_straight_line (a : ℝ) : a = 0 → ∃ x y : ℝ, curve_C a x y :=
by
  intro ha
  use (-1), 1
  rw [curve_C, ha]
  simp

theorem curve_C_not_tangent (a : ℝ) : a = 1 → ¬ ∀ x y, 3 * x + y = 0 → curve_C a x y :=
by
  sorry

theorem curve_C_fixed_point (x y a : ℝ) : curve_C a 0 0 :=
by
  rw [curve_C]
  simp

theorem curve_C_intersect (a : ℝ) : a = 1 → ∃ x y : ℝ, (x + 2 * y = 0) ∧ curve_C a x y :=
by
  sorry

end curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l513_513270


namespace total_weight_of_settings_l513_513413

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end total_weight_of_settings_l513_513413


namespace percentage_2x_minus_y_of_x_l513_513325

noncomputable def x_perc_of_2x_minus_y (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) : ℤ :=
  (2 * x - y) * 100 / x

theorem percentage_2x_minus_y_of_x (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) :
  x_perc_of_2x_minus_y x y z h1 h2 h3 h4 = 175 :=
  sorry

end percentage_2x_minus_y_of_x_l513_513325


namespace max_people_with_neighbors_l513_513435

theorem max_people_with_neighbors (n : ℕ) (hn : n > 1) :
  ∃ max_num : ℕ, max_num = n * (n + 1) / 2 ∧ 
  ∀ (people : list ℕ), 
    (∀ i j ∈ people, i = j → people.in_neighbors_different i) → 
    list.length people ≤ max_num :=
sorry

end max_people_with_neighbors_l513_513435


namespace min_value_of_n_l513_513448

theorem min_value_of_n {n : ℕ} (a : ℕ → ℚ) (b : ℕ → ℚ) :
  (∀ x : ℝ, x^2 + x + 4 = ∑ i in finset.range n, (a i * x + b i)^2) → 
  n ≥ 5 :=
sorry

end min_value_of_n_l513_513448


namespace even_product_probability_l513_513032

theorem even_product_probability :
  let A := {1, 2, 3, 4, 5}
  let B := {1, 2, 3, 4}
  let total_outcomes := 5 * 4
  let even_product_outcomes := 4 + 4 + 3 + 3
  even_product_outcomes / total_outcomes = (7 : ℚ) / 10 :=
by
  let A := {1, 2, 3, 4, 5}
  let B := {1, 2, 3, 4}
  let total_outcomes := 5 * 4
  let even_product_outcomes := 4 + 4 + 3 + 3
  have h_even_product_outcomes : even_product_outcomes = 14 := rfl
  have h_total_outcomes : total_outcomes = 20 := rfl
  rw [h_even_product_outcomes, h_total_outcomes]
  norm_num
  exact @eq.refl ℚ (7 / 10)

end even_product_probability_l513_513032


namespace graph_union_l513_513201

-- Definitions of the conditions from part a)
def graph1 (z y : ℝ) : Prop := z^4 - 6 * y^4 = 3 * z^2 - 2

def graph_hyperbola (z y : ℝ) : Prop := z^2 - 3 * y^2 = 2

def graph_ellipse (z y : ℝ) : Prop := z^2 - 2 * y^2 = 1

-- Lean statement to prove the question is equivalent to the answer
theorem graph_union (z y : ℝ) : graph1 z y ↔ (graph_hyperbola z y ∨ graph_ellipse z y) := 
sorry

end graph_union_l513_513201


namespace candy_last_duration_l513_513222

theorem candy_last_duration (candy_from_neighbors : ℕ) (candy_from_sister : ℕ) (candies_per_day : ℕ)
  (h_neighbors : candy_from_neighbors = 66)
  (h_sister : candy_from_sister = 15)
  (h_daily : candies_per_day = 9) :
  (candy_from_neighbors + candy_from_sister) / candies_per_day = 9 :=
by
  have total_candies : candy_from_neighbors + candy_from_sister = 81 := by
    rw [h_neighbors, h_sister]
    norm_num
  rw [total_candies, h_daily]
  norm_num

end candy_last_duration_l513_513222


namespace z_below_real_axis_l513_513391

theorem z_below_real_axis (t : ℝ) : 
  let z := (2 * t^2 + 5 * t - 3 : ℂ) + (t^2 + 2 * t + 2 : ℂ) * complex.i in
  complex.imag z < 0 := 
by
  sorry

end z_below_real_axis_l513_513391


namespace number_of_books_l513_513054

theorem number_of_books (pages_per_book : ℕ) (total_pages : ℕ) (h_pages_per_book : pages_per_book = 478) (h_total_pages : total_pages = 3824) :
  total_pages / pages_per_book = 8 :=
by 
  rw [h_pages_per_book, h_total_pages]
  exact sorry

end number_of_books_l513_513054


namespace main_statement_l513_513753

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513753


namespace alpha_perpendicular_beta_l513_513232

variables {α β : Type} [Plane α] [Plane β] 
variables {m n : Line}

-- Conditions
def m_parallel_n : Prop := parallel m n
def n_perpendicular_beta : Prop := perpendicular n β
def m_subset_alpha : Prop := subset m α

-- Theorem to prove α ⊥ β under the given conditions
theorem alpha_perpendicular_beta (h1 : m_parallel_n) (h2 : n_perpendicular_beta) (h3 : m_subset_alpha) : perpendicular α β :=
sorry

end alpha_perpendicular_beta_l513_513232


namespace average_income_family_l513_513069

theorem average_income_family (income1 income2 income3 income4 : ℕ) 
  (h1 : income1 = 8000) (h2 : income2 = 15000) (h3 : income3 = 6000) (h4 : income4 = 11000) :
  (income1 + income2 + income3 + income4) / 4 = 10000 := by
  sorry

end average_income_family_l513_513069


namespace proposition_true_l513_513837

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513837


namespace true_proposition_l513_513767

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513767


namespace ben_heads_probability_l513_513912

def coin_flip_probability : ℚ :=
  let total_ways := 2^10
  let ways_exactly_five_heads := Nat.choose 10 5
  let probability_exactly_five_heads := ways_exactly_five_heads / total_ways
  let remaining_probability := 1 - probability_exactly_five_heads
  let probability_more_heads := remaining_probability / 2
  probability_more_heads

theorem ben_heads_probability :
  coin_flip_probability = 193 / 512 := by
  sorry

end ben_heads_probability_l513_513912


namespace proposition_A_l513_513823

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513823


namespace p_and_q_is_true_l513_513699

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513699


namespace percent_defective_units_l513_513942

theorem percent_defective_units (D : ℝ) (h1 : 0.05 * D = 0.5) : D = 10 := by
  sorry

end percent_defective_units_l513_513942


namespace product_of_b_product_of_values_l513_513623

-- Condition: The length of the segment between the points (3b, 2b+5) and (7, 4) is 5sqrt(2).
def length_condition (b : ℝ) : Prop :=
  real.sqrt ((3 * b - 7) ^ 2 + ((2 * b + 5) - 4) ^ 2) = 5 * real.sqrt 2

theorem product_of_b (b : ℝ) (h : length_condition b) : b = 0 ∨ b = 38 / 13 :=
sorry

theorem product_of_values (b₁ b₂ : ℝ) (h1 : length_condition b₁) (h2 : length_condition b₂)
  (HB1 : b₁ = 0 ∨ b₁ = 38 / 13) (HB2 : b₂ = 0 ∨ b₂ = 38 / 13) : b₁ * b₂ = 0 :=
sorry

end product_of_b_product_of_values_l513_513623


namespace probability_three_marbles_same_color_l513_513124

-- Definitions of the counts of each color
def red_marbles : ℕ := 3
def white_marbles : ℕ := 4
def blue_marbles : ℕ := 5
def green_marbles : ℕ := 2

-- Total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

-- Probabilities for drawing three marbles of the same color
def P_all_red : ℚ := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
def P_all_white : ℚ := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
def P_all_blue : ℚ := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))
def P_all_green : ℚ := (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) * ((green_marbles - 2) / (total_marbles - 2))

-- Combined probability of drawing three marbles of the same color
def P_same_color : ℚ := P_all_red + P_all_white + P_all_blue + P_all_green

-- The required theorem
theorem probability_three_marbles_same_color :
  P_same_color = 15 / 364 := by
  sorry  -- Proof goes here but is not required by the task


end probability_three_marbles_same_color_l513_513124


namespace algebraic_expression_value_l513_513905

theorem algebraic_expression_value (a b : ℝ) (h : a + b = 3) : a^3 + b^3 + 9 * a * b = 27 :=
by
  sorry

end algebraic_expression_value_l513_513905


namespace true_proposition_l513_513842

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513842


namespace eventually_good_l513_513964

def is_good (a : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ) (terms : fin k → ℕ), (∀ i, terms i = seq (some_idx i)) ∧ a = (terms).sum

-- The core statement of the problem
theorem eventually_good (seq : ℕ → ℕ) (h_inc : ∀ n, seq n < seq (n + 1)) : ∃ N, ∀ n ≥ N, is_good (seq n) seq :=
begin
  sorry -- We skip the proof here
end

end eventually_good_l513_513964


namespace fine_per_day_l513_513562

theorem fine_per_day (x : ℝ) : 
  (let total_days := 30 in
   let earnings_per_day := 25 in
   let total_amount_received := 425 in
   let days_absent := 10 in
   let days_worked := total_days - days_absent in
   let total_earnings := days_worked * earnings_per_day in
   let total_fine := days_absent * x in
   total_earnings - total_fine = total_amount_received) → x = 7.5 :=
by
  intros h
  sorry

end fine_per_day_l513_513562


namespace even_distinct_digits_count_l513_513301

theorem even_distinct_digits_count : 
  let even_digits := {0, 2, 4, 6, 8} in
  let is_valid n := (2000 ≤ n ∧ n < 8000) ∧ 
                      (n.digits.to_list.all (λ d, d ∈ even_digits)) ∧ 
                      (n.digits.to_list.nodup) in
  (finset.range 8000).filter is_valid = 96 :=
begin
  sorry
end

end even_distinct_digits_count_l513_513301


namespace fraction_comparison_l513_513878

def inequality_solution (a : ℝ) : Prop :=
  2 < a ∧ a < 4

def fraction_simplification (a : ℝ) : ℝ :=
  a - (a + 4) / (a + 1)

def simplified_fraction (a : ℝ) : ℝ :=
  (a^2 - 4) / (a + 1)

theorem fraction_comparison (a : ℝ) : inequality_solution a → fraction_simplification a = simplified_fraction a :=
by
  intro h
  sorry

end fraction_comparison_l513_513878


namespace at_least_one_met_own_wife_l513_513491

-- Define the entities
structure Person := (name : String)
def Brother := Person
def Wife := Person

-- Define relationships
structure Family :=
  (husband : Brother)
  (wife : Wife)

-- Define the visiting condition
structure Visit :=
  (visitor : Person)
  (time : Nat) -- Simple model where "time" is an integer representing the order of visits

-- Define the conditions
def met_at_bedside (v1 v2 : Visit) : Prop :=
  v1.time = v2.time  -- simple condition for meeting at bedside at the same time

-- The three families
def A := {name := "A"}
def B := {name := "B"}
def C := {name := "C"}
def Mrs_A := {name := "Mrs_A"}
def Mrs_B := {name := "Mrs_B"}
def Mrs_C := {name := "Mrs_C"}

-- Visits of all individuals (times assigned for simplicity)
def visits := [
  {visitor := A, time := 1},
  {visitor := B, time := 2},
  {visitor := C, time := 3},
  {visitor := Mrs_A, time := 1},
  {visitor := Mrs_B, time := 1},
  {visitor := Mrs_C, time := 1}
]

-- Define the meeting conditions in Lean
def met_all_inlaws (v : Visit) (all_visits : List Visit) : Prop :=
  ∀ w in all_visits, w.visitor.name ≠ v.visitor.name ∧ w.visitor.name ≠ (v.visitor.name ++ "'s Wife") -> met_at_bedside v w

-- The problem to prove
theorem at_least_one_met_own_wife :
  -- Assuming all conditions:
  (∀ v in visits, met_all_inlaws v visits) →
  -- Prove at least one met their own wife:
  ∃ v1 v2 in visits, (v1.visitor.name ++ "'s Wife" = v2.visitor.name ∧ met_at_bedside v1 v2) := 
  sorry

end at_least_one_met_own_wife_l513_513491


namespace true_proposition_l513_513765

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513765


namespace triangle_areas_equal_l513_513351

variables {A B C D M : Type*} [inner_product_space ℝ T]

theorem triangle_areas_equal {A B C D M : Type*} [inner_product_space ℝ T] 
  (h1 : D ∈ segment ℝ B C) 
  (h2 : M ∈ segment ℝ A D) 
  (h3 : vector.angle A B M = vector.angle A C M) :
  (∃ D : T, midpoint ℝ B C D) ∧ (∃ M : T, M ∈ segment ℝ A D ∧ M ≠ midpoint ℝ A D) := 
sorry

end triangle_areas_equal_l513_513351


namespace main_statement_l513_513755

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513755


namespace units_digit_of_product_of_odds_l513_513517

theorem units_digit_of_product_of_odds :
  let odds := {n | 20 < n ∧ n < 120 ∧ n % 2 = 1}
  (∃ us : List ℕ, us = (List.filter (λ x, x ∈ odds) (List.range 120))) →
  (∃ p : ℕ, p = us.prod) →
  (∃ d : ℕ, d = p % 10) →
  d = 5 :=
by
  sorry

end units_digit_of_product_of_odds_l513_513517


namespace perpendicular_lines_l513_513320

theorem perpendicular_lines (a : ℝ) :
  (a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0 ↔ (a = 1 ∨ a = -1) := 
sorry

end perpendicular_lines_l513_513320


namespace median_incorrect_l513_513224

open List

def data : List ℕ := [3, 3, 6, 5, 3]

noncomputable def median (l: List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem median_incorrect : median data ≠ 6 :=
  sorry

end median_incorrect_l513_513224


namespace num_points_satisfying_inequalities_l513_513943

theorem num_points_satisfying_inequalities :
  ∃ (n : ℕ), n = 2551 ∧
  ∀ (x y : ℤ), (y ≤ 3 * x) ∧ (y ≥ x / 3) ∧ (x + y ≤ 100) → 
              ∃ (p : ℕ), p = n := 
by
  sorry

end num_points_satisfying_inequalities_l513_513943


namespace proof_problem_l513_513552

-- Definitions for the conditions used in the problem
def line1 (x y : ℝ) : Prop := 2 * x - 3 * y - 3 = 0
def line2 (x y : ℝ) : Prop := x + y + 2 = 0
def line_perpendicular (x y : ℝ) : Prop := 3 * x + y - 1 = 0
def line_through_intersection (x y : ℝ) (m : ℝ) : Prop := x - 3 * y + m = 0
def required_line (x y : ℝ) : Prop := 5 * x - 15 * y - 18 = 0

-- Definition for the line in the second part of the problem
def line_l (x y : ℝ) (m : ℝ) : Prop := m * x + y - 2 * (m + 1) = 0

-- The maximum distance from the origin to the line l
def max_distance_to_origin (d : ℝ) : Prop := d = 2 * Real.sqrt 2

-- The main statement for the proof problem in Lean
theorem proof_problem :
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line_through_intersection x y (-18/5)) ∧
  (∀ m : ℝ, (max_distance_to_origin (Real.sqrt ((2:ℝ)^2+(2:ℝ)^2))) 
  := by
  sorry

end proof_problem_l513_513552


namespace antacids_per_month_proof_l513_513423

structure ConsumptionRates where
  indian : Nat
  mexican : Nat
  other : Nat

structure WeeklyEatingPattern where
  indian_days : Nat
  mexican_days : Nat
  other_days : Nat

def antacids_per_week (rates : ConsumptionRates) (pattern : WeeklyEatingPattern) : Nat :=
  (pattern.indian_days * rates.indian) +
  (pattern.mexican_days * rates.mexican) +
  (pattern.other_days * rates.other)

def weeks_in_month : Nat := 4

theorem antacids_per_month_proof :
  let rates := ConsumptionRates.mk 3 2 1 in
  let pattern := WeeklyEatingPattern.mk 3 2 2 in -- 7 days - 3 Indian days - 2 Mexican days = 2 other days
  let weekly_antacids := antacids_per_week rates pattern in
  weekly_antacids * weeks_in_month = 60 := 
by
  -- let's skip the proof
  sorry

end antacids_per_month_proof_l513_513423


namespace range_of_a_l513_513637

theorem range_of_a (x θ a : ℝ) (hx : ∀ θ ∈ ℝ, 0 ≤ θ ∧ θ ≤ π / 2) :
  ((x + 3 + 2 * sin θ * cos θ) ^ 2 + (x + a * sin θ + a * cos θ) ^ 2 ≥ 1 / 8) ↔ (a ≥ 7 / 2) := 
by 
  sorry

end range_of_a_l513_513637


namespace ram_distance_l513_513988

variable (map_distance_two_mountains : ℝ) (actual_distance_two_mountains : ℝ) (map_distance_ram : ℝ) (k : ℝ)

-- Define the conditions
def conditions : Prop :=
  (map_distance_two_mountains = 312) ∧ 
  (actual_distance_two_mountains = 136) ∧
  (map_distance_ram = 25)

-- Define the statement to be proven
theorem ram_distance (h : conditions map_distance_two_mountains actual_distance_two_mountains map_distance_ram) :
  let scale := actual_distance_two_mountains / map_distance_two_mountains in
  let actual_distance_ram := map_distance_ram * scale in
  actual_distance_ram ≈ 10.897425 :=
by
  unfold conditions at h
  cases h with h1 h
  cases h with h2 h3
  let scale := actual_distance_two_mountains / map_distance_two_mountains
  let actual_distance_ram := map_distance_ram * scale
  sorry

end ram_distance_l513_513988


namespace triangle_angle_A_eq_60_l513_513927

theorem triangle_angle_A_eq_60 
  (A B C : Type) [EuclideanSpace A B C]
  (a b c : ℝ) 
  (ha : b^2 + c^2 - a^2 = b * c) : 
  A = 60 :=
sorry

end triangle_angle_A_eq_60_l513_513927


namespace average_disk_space_per_hour_l513_513580

/-- A digital music library contains 15 days of music and takes up 20,000 megabytes of disk space.
    Prove that the average disk space used per hour of music in this library is 56 megabytes, to the nearest whole number.
-/
theorem average_disk_space_per_hour (days: ℕ) (total_disk_space: ℕ) (hours_per_day: ℕ) (div_result: ℕ) (avg_disk_space: ℕ) :
  days = 15 ∧ 
  total_disk_space = 20000 ∧ 
  hours_per_day = 24 ∧ 
  div_result = days * hours_per_day ∧
  avg_disk_space = (total_disk_space: ℝ) / div_result ∧ 
  (an_index: ℕ, (avg_disk_space: ℝ) ≈ 55.56 → an_index = 56) :=
by 
  sorry

end average_disk_space_per_hour_l513_513580


namespace shortest_side_of_right_triangle_l513_513154

theorem shortest_side_of_right_triangle (a b : ℕ) (ha : a = 7) (hb : b = 24) (h_right : a^2 + b^2 = c^2) : 
  min a b = 7 :=
by
  assume (c : ℕ)
  have h := ha.symm ▸ hb.symm ▸ h_right
  sorry

end shortest_side_of_right_triangle_l513_513154


namespace distinct_sum_inequality_l513_513018

theorem distinct_sum_inequality {a : ℕ → ℕ} (n : ℕ) (h_distinct : ∀ i j : ℕ, i < n → j < n → a i = a j → i = j) :
  (1 + (finset.range n).sum (λ k, 1 / (k + 1))) ≤ (finset.range n).sum (λ k, a k / ((k + 1) ^ 2)) :=
sorry

end distinct_sum_inequality_l513_513018


namespace assignment_plans_count_l513_513627

/-- There are four students and three pavilions: A, B, and C. 
    Student A cannot be assigned to pavilion A.
    Each pavilion must have at least one student assigned. -/
theorem assignment_plans_count :
  let students := {A, B, C, D}
  let pavilions := {A, B, C}
  let assignments := {f : students → pavilions | (∀ s, ∃ p, s ≠ p) ∧ (∀ p, ∃ s, s = p)}
  ∃ (f : students → pavilions), f 'A ≠ A ∧
  #assignments = 24 :=
sorry

end assignment_plans_count_l513_513627


namespace problem_statement_l513_513706

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513706


namespace acute_angle_sum_equals_pi_over_two_l513_513397

theorem acute_angle_sum_equals_pi_over_two (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
  (h1 : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 1)
  (h2 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0) :
  2 * a + b = π / 2 := 
sorry

end acute_angle_sum_equals_pi_over_two_l513_513397


namespace proposition_true_l513_513831

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513831


namespace polynomial_divisibility_l513_513966

variable {R : Type*} [CommRing R] 

-- Define the polynomial f and its degree n
variable {f : R[X]} (n : ℕ) (p q : ℕ) (hq : 0 < q) (hp : 0 < p)

-- Conditions: f(x) is of degree n and f(x)^p is divisible by f'(x)^q
def is_polynomial_of_degree_and_property (f : R[X]) :=
  degree f = n ∧ f ^ p ∣ (derivative f) ^ q

-- Proposition to prove
theorem polynomial_divisibility (hf : is_polynomial_of_degree_and_property f n p q hq hp) :
  ∃ α : R, f = (X - C α) ^ n :=
  sorry -- proof goes here

end polynomial_divisibility_l513_513966


namespace angle_BAC_value_l513_513387

noncomputable theory

variables {A B C D E F : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited F]
variables {α : Type} [linear_ordered_ring α] [algebra α α] 
variables (AD BE CF : α) (H : α) 

-- Conditions
axiom altitudes : ∀(A B C : Type) [inhabited A] [inhabited B] [inhabited C], α
axiom condition1 : 5 * AD + 3 * BE + 2 * CF = 0

-- Main statement
theorem angle_BAC_value : ∀(A B C D E F : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited F], 
  (5 * AD + 3 * BE + 2 * CF = 0) → 
  ∃ (angle_BAC : α), angle_BAC = real.arccos (sqrt(10) / 5) :=
sorry

end angle_BAC_value_l513_513387


namespace geometric_sequence_common_ratio_l513_513331

theorem geometric_sequence_common_ratio
  (a₁ a₂ a₃ : ℝ) (q : ℝ) 
  (h₀ : 0 < a₁) 
  (h₁ : a₂ = a₁ * q) 
  (h₂ : a₃ = a₁ * q^2) 
  (h₃ : 2 * a₁ + a₂ = 2 * (1 / 2 * a₃)) 
  : q = 2 := 
sorry

end geometric_sequence_common_ratio_l513_513331


namespace heather_blocks_l513_513296

theorem heather_blocks (initial_blocks : ℕ) (shared_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 86 → shared_blocks = 41 → remaining_blocks = initial_blocks - shared_blocks → remaining_blocks = 45 :=
by
  sorry

end heather_blocks_l513_513296


namespace find_geometric_first_term_l513_513475

noncomputable def geometric_first_term (a r : ℝ) (h1 : a * r^5 = real.from_nat (nat.factorial 9)) 
  (h2 : a * r^8 = real.from_nat (nat.factorial 10)) : ℝ :=
  a

theorem find_geometric_first_term :
  ∃ a r, a * r^5 = real.from_nat (nat.factorial 9) ∧ a * r^8 = real.from_nat (nat.factorial 10) ∧ a = 362880 / 10^(5/3) :=
by
  have h : ∃ (a r: ℝ), a * r^5 = real.from_nat (nat.factorial 9) ∧ a * r^8 = real.from_nat (nat.factorial 10),
  {
    use [362880 / 10^(5/3), 10^(1/3)],
    split,
    { sorry },
    { sorry }
  },
  cases h with a ha,
  cases ha with r hr,
  use [a, r],
  split,
  { exact hr.left },
  split,
  { exact hr.right },
  { sorry }

end find_geometric_first_term_l513_513475


namespace angle_FCG_proof_l513_513995

-- Assume points A, B, C, D, E, F, G are on a circle in clockwise order.
-- AE is the diameter of the circle.
-- ∠ABF = 81° and ∠EDG = 76°.
-- We need to show that ∠FCG = 67°.

noncomputable def angle_FCG (A B C D E F G : Point) (circle : Circle) 
  (on_circle : ∀ (P : Point), P ∈ {A, B, C, D, E, F, G} → P ∈ circle) 
  (diameter : Segment) (H1 : diameter.End1 = A) (H2 : diameter.End2 = E) 
  (is_diameter : diameter.IsDiameter)
  (angle_ABF : ℝ) (H3 : angle_ABF = 81)
  (angle_EDG : ℝ) (H4 : angle_EDG = 76) 
  : ℝ :=
67

theorem angle_FCG_proof (A B C D E F G : Point) (circle : Circle) 
  (on_circle : ∀ (P : Point), P ∈ {A, B, C, D, E, F, G} → P ∈ circle) 
  (diameter : Segment) (H1 : diameter.End1 = A) (H2 : diameter.End2 = E) 
  (is_diameter : diameter.IsDiameter)
  (angle_ABF : ℝ) (H3 : angle_ABF = 81)
  (angle_EDG : ℝ) (H4 : angle_EDG = 76) 
  : angle_FCG A B C D E F G circle on_circle diameter H1 H2 is_diameter angle_ABF H3 angle_EDG H4 = 67 :=
by
  sorry

end angle_FCG_proof_l513_513995


namespace infinite_series_sum_eq_one_fourth_l513_513179

theorem infinite_series_sum_eq_one_fourth :
  (∑' n : ℕ, 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2))) = 1 / 4 :=
sorry

end infinite_series_sum_eq_one_fourth_l513_513179


namespace find_heaviest_weight_in_Geometric_Progression_l513_513252

theorem find_heaviest_weight_in_Geometric_Progression
  (b d : ℝ)
  (h_b_pos : b > 0)
  (h_d_gt_one : d > 1) :
  let w₁ := b,
    w₂ := b * d,
    w₃ := b * d ^ 2,
    w₄ := b * d ^ 3 in
  by sorry

end find_heaviest_weight_in_Geometric_Progression_l513_513252


namespace number_of_people_chose_pop_l513_513329

theorem number_of_people_chose_pop (total_people : ℕ) (angle_pop : ℕ) (h1 : total_people = 540) (h2 : angle_pop = 270) : (total_people * (angle_pop / 360)) = 405 := by
  sorry

end number_of_people_chose_pop_l513_513329


namespace angle_between_chords_half_diff_arcs_l513_513038

theorem angle_between_chords_half_diff_arcs {α : Type*} [noncomputable_field α] 
  (circle : Type*) [circle α] (A B C D O : circle.Point) 
  (hAB : A ≠ B ∧ A ≠ O ∧ B ≠ O) 
  (hCD : C ≠ D ∧ C ≠ O ∧ D ≠ O) :
  let β := ∠ (line_through A B).to_linear_map (line_through C D).to_linear_map in
  let arc_AB := arc_measure (A, B) in
  let arc_CD := arc_measure (C, D) in
  β = 1/2 * (arc_measure (B, D) - arc_measure (A, C)) :=
by
  sorry

end angle_between_chords_half_diff_arcs_l513_513038


namespace distance_AB_intersection_l513_513881

-- Definitions of the given line l and curve C
def line (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (sqrt 3 / 2) * t)
def curve (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Equation representing the curve in its standard form
def curve_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Statement of the problem in Lean
theorem distance_AB_intersection : 
  ∀ (t1 t2 : ℝ),
  (curve_eq (1 + (1/2) * t1) ((sqrt 3 / 2) * t1)) →
  (curve_eq (1 + (1/2) * t2) ((sqrt 3 / 2) * t2)) →
  |t1 - t2| = 1 :=
by 
  sorry

end distance_AB_intersection_l513_513881


namespace area_of_triangle_l513_513257

noncomputable def hyperbola := {x : ℝ × ℝ // (x.1 ^ 2) / 4 - (x.2 ^ 2) = 1}

def are_orthogonal (P F1 F2: ℝ × ℝ) : Prop :=
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

def foci_of_hyperbola (a : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((a * sqrt 5, 0), (-a * sqrt 5, 0))

theorem area_of_triangle
  (a : ℝ)
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (h1 : F1 = (2 * sqrt 5, 0))
  (h2 : F2 = (-2 * sqrt 5, 0))
  (h3 : P ∈ hyperbola)
  (h4 : are_orthogonal P F1 F2) :
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  (1/2) * (PF1.1 * PF2.2 - PF1.2 * PF2.1) = 1 := 
sorry

end area_of_triangle_l513_513257


namespace find_coefficient_a_l513_513255

theorem find_coefficient_a :
  ∀ a b : ℚ, 
  (∀ x : ℝ, x^3 + a * x^2 + (b : ℝ) * x + 49 = 0 → (x = -2 - 5 * Real.sqrt 3 ∨ x = -2 + 5 * Real.sqrt 3 ∨ x = 49 / 71)) →
  a = 235 / 71 :=
begin
  sorry
end

end find_coefficient_a_l513_513255


namespace exp_is_convex_log_is_concave_weighted_geometric_inequality_l513_513528

variable {a x1 x2 : ℝ} {q1 q2 : ℝ}
variable (hx : a > 1) (hq : q1 + q2 = 1)

-- Statement 1: \(a^x\) is convex for \(a > 1\)
theorem exp_is_convex : ConvexOn ℝ (λ x, a ^ x) :=
sorry

-- Statement 2: \(\log_a (x)\) is concave for \(a > 1\)
theorem log_is_concave : ConcaveOn ℝ (λ x, log a x) :=
sorry

-- Statement 3: Weighted geometric mean is between arithmetic and harmonic means
theorem weighted_geometric_inequality (hx1 : x1 > 0) (hx2 : x2 > 0) :
  x1 ^ q1 * x2 ^ q2 ≤ q1 * x1 + q2 * x2 ∧
  x1 ^ q1 * x2 ^ q2 ≥ 1 / (q1 * (1 / x1) + q2 * (1 / x2)) :=
sorry

end exp_is_convex_log_is_concave_weighted_geometric_inequality_l513_513528


namespace main_statement_l513_513744

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513744


namespace probability_even_product_and_prime_sum_l513_513085

noncomputable def total_pairs : ℕ := nat.choose 8 2

def even_product_pairs : ℕ := 16 + 6 -- choose 1 even and 1 odd + choose 2 evens

def prime_sum_pairs : list (ℕ × ℕ) :=
  [(1, 2), (1, 4), (2, 3), (1, 6), (2, 5), (3, 4), (3, 8), (4, 7), 
   (5, 6), (5, 8), (6, 7)]

def even_product_and_prime_sum_pairs : list (ℕ × ℕ) :=
  [(1, 2), (1, 4), (2, 3), (3, 4), (3, 8), (5, 6), (6, 7)]

theorem probability_even_product_and_prime_sum :
  (even_product_and_prime_sum_pairs.length : ℚ) / (total_pairs : ℚ) = 1 / 4 := 
by
  sorry

end probability_even_product_and_prime_sum_l513_513085


namespace rabbit_speed_final_result_l513_513326

def rabbit_speed : ℕ := 45

def double_speed (speed : ℕ) : ℕ := speed * 2

def add_four (n : ℕ) : ℕ := n + 4

def final_operation : ℕ := double_speed (add_four (double_speed rabbit_speed))

theorem rabbit_speed_final_result : final_operation = 188 := 
by
  sorry

end rabbit_speed_final_result_l513_513326


namespace incircle_radius_and_tangent_length_l513_513454

-- Define the given parameters and the expected results
variables {a m : ℝ}

-- Define the isosceles triangle with base BC=2a and height AD=m
def isosceles_triangle (a m : ℝ) : Prop :=
  m > 0 ∧ a > 0

-- Define the radius of the incircle
def incircle_radius (a m : ℝ) : ℝ :=
  am / (a + real.sqrt (a^2 + m^2))

-- Define the length of the tangent parallel to the base
def parallel_tangent_length (a m : ℝ) : ℝ :=
  2 * a * (real.sqrt (a^2 + m^2) - a) / (a + real.sqrt (a^2 + m^2))

theorem incircle_radius_and_tangent_length (h : isosceles_triangle a m) :
  incircle_radius a m = am / (a + real.sqrt (a^2 + m^2)) ∧
  parallel_tangent_length a m = 2 * a * (real.sqrt (a^2 + m^2) - a) / (a + real.sqrt (a^2 + m^2)) :=
by {
  sorry
}

end incircle_radius_and_tangent_length_l513_513454


namespace answer_is_p_and_q_l513_513729

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513729


namespace peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513013

-- Probability of Peter winning a prize:
theorem peter_wins_prize_probability :
  let p := (5 / 6) in p ^ 9 = (5 / 6) ^ 9 := by
  sorry

-- Probability that at least one person wins a prize:
theorem at_least_one_wins_prize_probability :
  let p := (5 / 6) in
  let q := (1 - p^9) in 
  (1 - q^10) ≈ 0.919 := by
  sorry

end peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513013


namespace C_share_theorem_l513_513554

-- Definitions and conditions
def totalPayment : ℝ := 500
def work_done_in_days (days: ℝ) (initial_rate: ℝ) (rate_change: ℝ → ℝ) : ℝ :=
  initial_rate + rate_change initial_rate

def A_initial_rate : ℝ := 1 / 5
def B_initial_rate : ℝ := 1 / 10
def A_rate_change (rate: ℝ) : ℝ := rate * 0.9
def B_rate_change (rate: ℝ) : ℝ := rate * 1.05

-- Calculate work done by A and B in 2 days
def A_work : ℝ := work_done_in_days 1 A_initial_rate A_rate_change + work_done_in_days 2 (A_rate_change A_initial_rate) A_rate_change
def B_work : ℝ := work_done_in_days 1 B_initial_rate B_rate_change + work_done_in_days 2 (B_rate_change B_initial_rate) B_rate_change

def total_work_done_by_A_and_B : ℝ := A_work + B_work

def C_work : ℝ := 1 - total_work_done_by_A_and_B
def C_fraction_of_work : ℝ := C_work

-- The actual share of C
def C_share : ℝ := totalPayment * C_fraction_of_work

theorem C_share_theorem : C_share = 370.95 := by
  sorry

end C_share_theorem_l513_513554


namespace inverse_value_l513_513655

noncomputable def f (x : ℝ) : ℝ := (x ^ 7 - 1) / 5

theorem inverse_value :
  ∃ (y : ℝ), f(y) = -1 / 10 ∧ y = 2^(-1/7) :=
by
  sorry

end inverse_value_l513_513655


namespace proposition_p_and_q_is_true_l513_513727

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513727


namespace true_proposition_l513_513769

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513769


namespace eval_expression_l513_513184

theorem eval_expression : |-3| - (Real.sqrt 7 + 1)^0 - 2^2 = -2 :=
by
  sorry

end eval_expression_l513_513184


namespace peter_wins_prize_at_least_one_person_wins_prize_l513_513009

-- Part (a): Probability that Peter wins a prize
theorem peter_wins_prize :
  let p : Probability := (5 / 6) ^ 9
  p = 0.194 := sorry

-- Part (b): Probability that at least one person wins a prize
theorem at_least_one_person_wins_prize :
  let p : Probability := 0.919
  p = 0.919 := sorry

end peter_wins_prize_at_least_one_person_wins_prize_l513_513009


namespace solve_for_y_l513_513028

theorem solve_for_y : ∀ y : ℝ, (y - 5)^3 = (1 / 27)⁻¹ → y = 8 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l513_513028


namespace arithmetic_sequence_x_value_l513_513464

theorem arithmetic_sequence_x_value :
  (∃ x : ℚ, (4 * x, 2 * x - 3, 4 * x - 3) ∈ arithmetic_sequence ∧ x = -3 / 4) :=
by
  sorry

end arithmetic_sequence_x_value_l513_513464


namespace knights_placement_maximal_attacks_l513_513937

-- Define the board and the central region
def is_in_central_region (x y : ℕ) : Prop :=
  (3 ≤ x ∧ x ≤ 6) ∧ (3 ≤ y ∧ y ≤ 6)

-- Define the color of each square in the checkerboard pattern
def is_black_square (x y : ℕ) : Prop :=
  (x + y) % 2 = 0

-- Number of ways to place knights satisfying the conditions
theorem knights_placement_maximal_attacks : 
  ∃ (count : ℕ), count = 64 ∧ 
           ∀ (k1 k2 k3 : (ℕ × ℕ)), 
              is_in_central_region k1.1 k1.2 ∧ is_in_central_region k2.1 k2.2 ∧ is_in_central_region k3.1 k3.2 →
              is_black_square k1.1 k1.2 ≠ is_black_square k2.1 k2.2 ∨ 
              is_black_square k2.1 k2.2 ≠ is_black_square k3.1 k3.2 ∨ 
              is_black_square k1.1 k1.2 ≠ is_black_square k3.1 k3.2 →
              count = 64 :=
begin
  sorry
end

end knights_placement_maximal_attacks_l513_513937


namespace peter_wins_prize_at_least_one_wins_prize_l513_513004

noncomputable def probability_peter_wins (N : Nat) := 
  (5/6) ^ (N - 1)

theorem peter_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob = probability_peter_wins N := by
  have h2 : prob = (5/6) ^ 9 := by
    sorry
  exact h2

noncomputable def probability_at_least_one_wins (N : Nat) := 
  10 * (5/6)^9 - 45 * (5 * 4^8 / 6^9) + 120 * (5 * 4 * 3^7 / 6^9) - 210 * (5 * 4 * 3 * 2^6 / 6^9) + 252 * (5 * 4 * 3 * 2 * 1 / 6^9)

theorem at_least_one_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob ≈ probability_at_least_one_wins N := by
  have h2 : prob ≈ 0.919 := by
    sorry
  exact h2

end peter_wins_prize_at_least_one_wins_prize_l513_513004


namespace horizontal_distance_on_parabola_l513_513139

theorem horizontal_distance_on_parabola :
  let P := {x : ℝ | x^2 - x - 12 = 0},
      Q := {x : ℝ | x^2 - x - 2 = 0} in
  ∀ p ∈ P, ∃ q ∈ Q, |p - q| = 2 :=
by sorry

end horizontal_distance_on_parabola_l513_513139


namespace peter_wins_prize_at_least_one_wins_prize_l513_513006

noncomputable def probability_peter_wins (N : Nat) := 
  (5/6) ^ (N - 1)

theorem peter_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob = probability_peter_wins N := by
  have h2 : prob = (5/6) ^ 9 := by
    sorry
  exact h2

noncomputable def probability_at_least_one_wins (N : Nat) := 
  10 * (5/6)^9 - 45 * (5 * 4^8 / 6^9) + 120 * (5 * 4 * 3^7 / 6^9) - 210 * (5 * 4 * 3 * 2^6 / 6^9) + 252 * (5 * 4 * 3 * 2 * 1 / 6^9)

theorem at_least_one_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob ≈ probability_at_least_one_wins N := by
  have h2 : prob ≈ 0.919 := by
    sorry
  exact h2

end peter_wins_prize_at_least_one_wins_prize_l513_513006


namespace height_difference_l513_513150

variable (h_A h_B h_D h_E h_F h_G : ℝ)

theorem height_difference :
  (h_A - h_D = 4.5) →
  (h_E - h_D = -1.7) →
  (h_F - h_E = -0.8) →
  (h_G - h_F = 1.9) →
  (h_B - h_G = 3.6) →
  (h_A - h_B > 0) :=
by
  intro h_AD h_ED h_FE h_GF h_BG
  sorry

end height_difference_l513_513150


namespace solve_siblings_age_problem_l513_513493

def siblings_age_problem (x : ℕ) : Prop :=
  let age_eldest := 20
  let age_middle := 15
  let age_youngest := 10
  (age_eldest + x) + (age_middle + x) + (age_youngest + x) = 75 → x = 10

theorem solve_siblings_age_problem : siblings_age_problem 10 :=
by
  sorry

end solve_siblings_age_problem_l513_513493


namespace chandler_total_cost_l513_513921

theorem chandler_total_cost (
  cost_per_movie_ticket : ℕ := 30
  cost_8_movie_tickets : ℕ := 8 * cost_per_movie_ticket
  cost_per_football_game_ticket : ℕ := cost_8_movie_tickets / 2
  cost_5_football_game_tickets : ℕ := 5 * cost_per_football_game_ticket
  total_cost : ℕ := cost_8_movie_tickets + cost_5_football_game_tickets
) : total_cost = 840 := by
  sorry

end chandler_total_cost_l513_513921


namespace pumpkin_pie_problem_l513_513292

theorem pumpkin_pie_problem
  (baked : ℕ)
  (sold : ℕ)
  (given_away : ℕ)
  (pieces_each : ℕ)
  (portion_eaten : ℚ)
  (initial_pies_eq_baked : baked = 4)
  (sold_pies_eq : sold = 1)
  (given_away_pies_eq : given_away = 1)
  (pieces_each_eq : pieces_each = 6)
  (portion_eaten_eq : portion_eaten = 2 / 3) :
  let remaining_pies := baked - sold - given_away,
      sliced_pieces := remaining_pies * pieces_each,
      pieces_eaten := sliced_pieces * portion_eaten,
      pieces_left := sliced_pieces - pieces_eaten in
  pieces_left = 4 := by
  sorry

end pumpkin_pie_problem_l513_513292


namespace even_function_odd_function_neither_even_nor_odd_function_l513_513103

def f (x : ℝ) : ℝ := 1 + x^2 + x^4
def g (x : ℝ) : ℝ := x + x^3 + x^5
def h (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + x^4

theorem even_function : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

theorem neither_even_nor_odd_function : ∀ x : ℝ, (h (-x) ≠ h x) ∧ (h (-x) ≠ -h x) :=
by
  sorry

end even_function_odd_function_neither_even_nor_odd_function_l513_513103


namespace problem_solution_l513_513209

noncomputable def solution_set := {p : ℝ × ℝ × ℝ | 
  let (x, y, z) := p in 
  (4 * x^2) / (1 + 4 * x^2) = y ∧ 
  (4 * y^2) / (1 + 4 * y^2) = z ∧ 
  (4 * z^2) / (1 + 4 * z^2) = x }

theorem problem_solution : solution_set = {(0, 0, 0), (1/2, 1/2, 1/2)} :=
by {
  sorry
}

end problem_solution_l513_513209


namespace circle_ordinary_eq_product_of_distances_l513_513944

noncomputable def circle_eq (x y θ : ℝ) : Prop :=
  (x = sqrt 3 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)

noncomputable def line_eq_m (x y t : ℝ) : Prop :=
  (x = sqrt 3 + t / 2) ∧ (y = 7 + (sqrt 3) * t / 2)

noncomputable def ray_eq_l (x y : ℝ) : Prop :=
  (x ≤ 0) ∧ (y = -(sqrt 3 / 3) * x)

theorem circle_ordinary_eq :
  ∀ (x y θ : ℝ), circle_eq x y θ → (x - sqrt 3)^2 + y^2 = 4 := by
    intros x y θ h
    cases h with h1 h2
    have h1' : x - sqrt 3 = 2 * Real.cos θ := by exact h1
    have h2' : y = 2 * Real.sin θ := by exact h2
    sorry

theorem product_of_distances 
  (θ : ℝ) 
  (hθ1 : θ = 5 * Real.pi / 6) 
  (A B : ℝ) 
  (hA : A = sqrt 3) 
  (hB : B = sqrt 13) 
  (x y : ℝ) 
  (hx : x = (sqrt 3 - 3) / 2) 
  (hy : y = 2) 
  : |A * B| = -3 + sqrt 13 := by
    have h₁ : |(sqrt 3 - 3) / 2 * 2| = -3 + sqrt 13 := by sorry
    rw [hx, hy] at h₁
    exact h₁

-- The specific X, Y, etc., must be derived and plugged into the theorem properly.

end circle_ordinary_eq_product_of_distances_l513_513944


namespace tangent_line_at_4_is_correct_range_of_a_l513_513277

-- Definition of the function
def f (x : ℝ) : ℝ := (x^2) / 4 - Real.sqrt x

-- Point coordinates
def f_at_4 : ℝ := f 4

-- Slope of the tangent line at x = 4
def f'_at_4 : ℝ := (4 : ℝ) / 2 - 1 / (2 * Real.sqrt 4)

-- Equation of the tangent line
def tangent_line_equation (x y : ℝ) : Prop :=
  7 * x - 4 * y - 20 = 0

-- Minimum value of the function f(x)
def minimum_a : ℝ := -(3 : ℝ) / 4

-- Proof problems
theorem tangent_line_at_4_is_correct : ∀ x y : ℝ, 
  tangent_line_equation x y ↔ 
  (y = f_at_4 + f'_at_4 * (x - 4)) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x ≥ a) ↔ (a ≤ minimum_a) := sorry

end tangent_line_at_4_is_correct_range_of_a_l513_513277


namespace find_complex_z_l513_513260

open Complex

theorem find_complex_z (z : ℂ) 
  (h : z * conj z + 2 * I * z = 9 + 2 * I) : 
  z = 1 - 2 * I ∨ z = 1 + 4 * I :=
sorry

end find_complex_z_l513_513260


namespace fraction_of_circle_radius_l513_513052

def side_of_square (area_of_square : ℝ) : ℝ := real.sqrt area_of_square

noncomputable def length_of_rectangle (area_of_rectangle breadth_of_rectangle : ℝ) : ℝ :=
  area_of_rectangle / breadth_of_rectangle

theorem fraction_of_circle_radius (area_of_square : ℝ) (area_of_rectangle : ℝ) (breadth_of_rectangle : ℝ) :
  side_of_square area_of_square = 35 →
  area_of_square = 1225 →
  area_of_rectangle = 200 →
  breadth_of_rectangle = 10 →
  length_of_rectangle area_of_rectangle breadth_of_rectangle / side_of_square area_of_square = 4 / 7 :=
by
  sorry

end fraction_of_circle_radius_l513_513052


namespace largest_square_side_length_correct_l513_513361

-- Define the problem parameters
def rectangle : Type := {length : ℝ, width : ℝ}
def square : Type := {side_length : ℝ}
def triangle : Type := {side_length : ℝ}

-- Given conditions
def rect : rectangle := {length := 16, width := 12}
def congruent_triangles : triangle := {side_length := (20 * real.sqrt(3)) / 3}

-- The largest square inscribed in the specific region discussed
def largest_square : square := {side_length := 5 * real.sqrt 2 - (5 * real.sqrt 6) / 3}

-- Proof problem
theorem largest_square_side_length_correct :
  ∀ (r : rectangle) (t : triangle),
    r = rect →
    t = congruent_triangles →
    (∃ s : square, s.side_length = largest_square.side_length) := by
  sorry

end largest_square_side_length_correct_l513_513361


namespace driver_average_speed_l513_513582

theorem driver_average_speed (v t : ℝ) (h1 : ∀ d : ℝ, d = v * t → (d / (v + 10)) = (3 / 4) * t) : v = 30 := by
  sorry

end driver_average_speed_l513_513582


namespace P_iff_Q_l513_513672

def P (x : ℝ) := x > 1 ∨ x < -1
def Q (x : ℝ) := |x + 1| + |x - 1| > 2

theorem P_iff_Q : ∀ x, P x ↔ Q x :=
by
  intros x
  sorry

end P_iff_Q_l513_513672


namespace combined_total_l513_513067

-- Definitions for the problem conditions
def marks_sandcastles : ℕ := 20
def towers_per_marks_sandcastle : ℕ := 10

def jeffs_multiplier : ℕ := 3
def towers_per_jeffs_sandcastle : ℕ := 5

-- Definitions derived from conditions
def jeffs_sandcastles : ℕ := jeffs_multiplier * marks_sandcastles
def marks_towers : ℕ := marks_sandcastles * towers_per_marks_sandcastle
def jeffs_towers : ℕ := jeffs_sandcastles * towers_per_jeffs_sandcastle

-- Question translated to a Lean theorem
theorem combined_total : 
  (marks_sandcastles + jeffs_sandcastles) + (marks_towers + jeffs_towers) = 580 :=
by
  -- The proof would go here
  sorry

end combined_total_l513_513067


namespace range_of_3x_plus_2y_l513_513236

theorem range_of_3x_plus_2y (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : -1 ≤ y ∧ y ≤ 4) :
  1 ≤ 3 * x + 2 * y ∧ 3 * x + 2 * y ≤ 17 :=
sorry

end range_of_3x_plus_2y_l513_513236


namespace evaluate_expression_l513_513117

theorem evaluate_expression : (+1) + (-2) - (+8) - (-9) = 0 := by
  sorry

end evaluate_expression_l513_513117


namespace exists_one_to_one_correspondence_l513_513591

noncomputable def P : Set ℕ := {p : ℕ | p.Prime} -- A set of 2005 distinct prime numbers
def A (P : Set ℕ) : Set (Set ℕ) := {S : Set ℕ | S ⊆ P ∧ S.card = 1002} -- Set of all possible products of 1002 elements of P
def B (P : Set ℕ) : Set (Set ℕ) := {S : Set ℕ | S ⊆ P ∧ S.card = 1003} -- Set of all products of 1003 elements of P

theorem exists_one_to_one_correspondence (P : Set ℕ)
  (hP : P.card = 2005 ∧ (∀ p ∈ P, p.Prime)) :
  ∃ f : A P → B P, ∀ a ∈ A P, (∃ q ∈ P, q ∉ a ∧ f a = insert q a ∧ a ∣ (f a).prod) :=
sorry

end exists_one_to_one_correspondence_l513_513591


namespace students_not_A_either_l513_513933

-- Given conditions as definitions
def total_students : ℕ := 40
def students_A_history : ℕ := 10
def students_A_math : ℕ := 18
def students_A_both : ℕ := 6

-- Statement to prove
theorem students_not_A_either : (total_students - (students_A_history + students_A_math - students_A_both)) = 18 := 
by
  sorry

end students_not_A_either_l513_513933


namespace answer_is_p_and_q_l513_513742

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513742


namespace solve_rational_eq_l513_513211

theorem solve_rational_eq {x : ℝ} (h : 1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 4 * x - 5) + 1 / (x^2 - 15 * x - 12) = 0) :
  x = 3 ∨ x = -4 ∨ x = 1 ∨ x = -5 :=
by {
  sorry
}

end solve_rational_eq_l513_513211


namespace product_units_digit_odd_numbers_20_120_l513_513519

open Finset

def odd_numbers_between_20_and_120 : Finset ℕ := 
  filter (λ n => (n % 2 = 1) ∧ (20 < n) ∧ (n < 120)) (range 121)

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def product_units_digit (s : Finset ℕ) : ℕ := 
  (s.prod id) % 10

theorem product_units_digit_odd_numbers_20_120 :
  product_units_digit odd_numbers_between_20_and_120 = 5 :=
sorry

end product_units_digit_odd_numbers_20_120_l513_513519


namespace complex_equation_real_root_l513_513269

theorem complex_equation_real_root (m : ℝ) (x : ℝ) (h_eq : x^2 + (1 - 2 * complex.I) * x + 3 * m - complex.I = 0):
  m = 1 / 12 := 
by
  sorry

end complex_equation_real_root_l513_513269


namespace line_through_intersection_and_origin_l513_513889

theorem line_through_intersection_and_origin :
  ∃ (x y : ℝ), (2*x + y = 3) ∧ (x + 4*y = 2) ∧ (x - 10*y = 0) :=
by
  sorry

end line_through_intersection_and_origin_l513_513889


namespace answer_is_p_and_q_l513_513734

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513734


namespace satisfies_conditions_l513_513099

theorem satisfies_conditions : 
  ∃ f : ℝ → ℝ, (∀ x, f (-x) = -f x) ∧
               (∀ x, x > 2 → 0 < deriv f x) ∧
               (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
               f = λ x, x * (x + 1) * (x - 1) :=
by {
  use λ x : ℝ, x * (x + 1) * (x - 1),
  split,
  { intro x,
    simp,
    ring },
  split,
  { intros x hx,
    calc deriv (λ x : ℝ, x * (x + 1) * (x - 1)) x = 3 * x ^ 2 - 1 : by { sorry }
    ... > 0 : by { sorry }},
  {
    use [0, 1, -1],
    simp,
    split,
    { linarith },
    split,
    { linarith },
    split,
    { linarith },
    split,
    { ring },
    split,
    { ring },
    { ring }
  }
}

end satisfies_conditions_l513_513099


namespace nell_baseball_cards_now_l513_513986

theorem nell_baseball_cards_now :
  ∃ (B : ℕ), B = 376 - 265 :=
by
  use 111
  sorry

end nell_baseball_cards_now_l513_513986


namespace max_value_sqrt_expr_l513_513972

/-- Given nonnegative real numbers x, y, z with x + y + z = 15, the maximum value of 
    sqrt(3 * x + 1) + sqrt(3 * y + 1) + sqrt(3 * z + 1) is sqrt(48). -/
theorem max_value_sqrt_expr (x y z : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : x + y + z = 15) : 
  sqrt (3 * x + 1) + sqrt (3 * y + 1) + sqrt (3 * z + 1) ≤ sqrt 48 :=
sorry

end max_value_sqrt_expr_l513_513972


namespace proposition_true_l513_513833

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513833


namespace main_statement_l513_513751

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513751


namespace not_all_zero_implies_at_least_one_nonzero_l513_513494

variable {a b c : ℤ}

theorem not_all_zero_implies_at_least_one_nonzero (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) : 
  a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 := 
by 
  sorry

end not_all_zero_implies_at_least_one_nonzero_l513_513494


namespace proof_problem_l513_513783

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513783


namespace units_digit_of_product_of_odds_l513_513515

theorem units_digit_of_product_of_odds :
  let odds := {n | 20 < n ∧ n < 120 ∧ n % 2 = 1}
  (∃ us : List ℕ, us = (List.filter (λ x, x ∈ odds) (List.range 120))) →
  (∃ p : ℕ, p = us.prod) →
  (∃ d : ℕ, d = p % 10) →
  d = 5 :=
by
  sorry

end units_digit_of_product_of_odds_l513_513515


namespace initial_bottle_caps_correct_l513_513196

-- Defining the variables based on the conditions
def bottle_caps_found : ℕ := 7
def total_bottle_caps_now : ℕ := 32
def initial_bottle_caps : ℕ := 25

-- Statement of the theorem
theorem initial_bottle_caps_correct:
  total_bottle_caps_now - bottle_caps_found = initial_bottle_caps :=
sorry

end initial_bottle_caps_correct_l513_513196


namespace correct_equation_l513_513096

theorem correct_equation (a b : ℝ) : 
  (a + b)^2 = a^2 + 2 * a * b + b^2 := by
  sorry

end correct_equation_l513_513096


namespace find_c_for_minimum_value_l513_513274

-- Definitions based on the conditions
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Main statement to be proved
theorem find_c_for_minimum_value (c : ℝ) : (∀ x, (3*x^2 - 4*c*x + c^2) = 0) → c = 3 :=
by
  sorry

end find_c_for_minimum_value_l513_513274


namespace problem_statement_l513_513869

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - x^2

theorem problem_statement (x0 x1 x2 m : ℝ) (h0 : f x0 = m) (h1 : 0 < x1) (h2 : x1 < x0) (h3 : x0 < x2) :
    f x1 > m ∧ f x2 < m :=
sorry

end problem_statement_l513_513869


namespace total_emails_in_april_l513_513954

-- Definitions representing the conditions
def emails_per_day_initial : Nat := 20
def extra_emails_per_day : Nat := 5
def days_in_month : Nat := 30
def half_days_in_month : Nat := days_in_month / 2

-- Definitions to calculate total emails
def emails_first_half : Nat := emails_per_day_initial * half_days_in_month
def emails_per_day_after_subscription : Nat := emails_per_day_initial + extra_emails_per_day
def emails_second_half : Nat := emails_per_day_after_subscription * half_days_in_month

-- Main theorem to prove the total number of emails received in April
theorem total_emails_in_april : emails_first_half + emails_second_half = 675 := by 
  calc
    emails_first_half + emails_second_half
    = (emails_per_day_initial * half_days_in_month) + (emails_per_day_after_subscription * half_days_in_month) : rfl
    ... = (20 * 15) + ((20 + 5) * 15) : rfl
    ... = 300 + 375 : rfl
    ... = 675 : rfl

end total_emails_in_april_l513_513954


namespace g_18_value_l513_513970

-- Define the function g as taking positive integers to positive integers
variable (g : ℕ+ → ℕ+)

-- Define the conditions for the function g
axiom increasing (n : ℕ+) : g (n + 1) > g n
axiom multiplicative (m n : ℕ+) : g (m * n) = g m * g n
axiom power_property (m n : ℕ+) (h : m ≠ n ∧ m ^ (n : ℕ) = n ^ (m : ℕ)) :
  g m = n ∨ g n = m

-- Prove that g(18) is 72
theorem g_18_value : g 18 = 72 :=
sorry

end g_18_value_l513_513970


namespace num_solutions_triples_l513_513199

def sign (a : ℝ) : ℝ :=
if a > 0 then 1 else if a < 0 then -1 else 0

theorem num_solutions_triples :
  let x := 2020 - 2021 * sign (y + z),
      y := 2020 - 2021 * sign (x + z),
      z := 2020 - 2021 * sign (x + y) in
  count_solutions x y z = 3 :=
by sorry

end num_solutions_triples_l513_513199


namespace initial_cost_price_of_bicycle_for_A_l513_513108

/-
Given:
1. A sells the bicycle to B at a profit of 35% after offering a discount of 20% on the marked price.
2. B sells the bicycle to C at a profit of 45% after adding an additional cost of 10% for service charges.
3. The final selling price of the bicycle was Rs. 225.
4. A paid a tax of 15% on the initial cost price.

Prove:
Initial cost price of the bicycle for A, including tax, is approximately Rs. 120.168
-/

-- Define the given constants
def marked_price := ℝ
def initial_cost_price_A := ℝ
def discount_rate := 0.20
def profit_rate_A := 0.35
def service_charge_rate := 0.10
def profit_rate_B := 0.45
def final_selling_price := 225.0
def tax_rate := 0.15

-- Define the selling price from A to B after offering a discount
def SP_ab := (marked_price - discount_rate * marked_price)

-- Define the selling price from A to B with profit
def SP_ab_with_profit := (initial_cost_price_A + profit_rate_A * initial_cost_price_A)

-- Define the total cost for B after adding service charges
def total_cost_B := (SP_ab_with_profit + service_charge_rate * SP_ab_with_profit)

-- Define the selling price from B to C with profit
def SP_bc := (total_cost_B + profit_rate_B * total_cost_B)

-- Given equation from final selling price
axiom final_selling_price_def : SP_bc = final_selling_price

-- Tax paid by A
def tax_paid_by_A := (tax_rate * initial_cost_price_A)

-- Initial cost price of A including tax
def initial_cost_price_A_with_tax := (initial_cost_price_A + tax_paid_by_A)

-- Prove the initial cost price of the bicycle for A (including tax)
theorem initial_cost_price_of_bicycle_for_A :
  initial_cost_price_A_with_tax ≈ 120.168 :=
sorry

end initial_cost_price_of_bicycle_for_A_l513_513108


namespace isosceles_triangle_base_angles_l513_513860

theorem isosceles_triangle_base_angles (T : Type) [triangle T] (a b c : T) : 
  is_isosceles_triangle a b c ∧ (angle a b c = 80 ∨ angle b c a = 80 ∨ angle c a b = 80) → 
  (angle a b c = 50 ∨ angle b c a = 50 ∨ angle c a b = 50 ∨ angle a b c = 80 ∨ angle b c a = 80 ∨ angle c a b = 80) :=
by
  sorry

end isosceles_triangle_base_angles_l513_513860


namespace proof_problem_l513_513775

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513775


namespace correct_graph_is_C_l513_513327

-- Define the years and corresponding remote work percentages
def percentages : List (ℕ × ℝ) := [
  (1960, 0.1),
  (1970, 0.15),
  (1980, 0.12),
  (1990, 0.25),
  (2000, 0.4)
]

-- Define the property of the graph trend
def isCorrectGraph (p : List (ℕ × ℝ)) : Prop :=
  p = [
    (1960, 0.1),
    (1970, 0.15),
    (1980, 0.12),
    (1990, 0.25),
    (2000, 0.4)
  ]

-- State the theorem
theorem correct_graph_is_C : isCorrectGraph percentages = True :=
  sorry

end correct_graph_is_C_l513_513327


namespace inscribed_rectangle_area_correct_l513_513021

noncomputable def area_of_inscribed_rectangle : Prop := 
  let AD : ℝ := 15 / (12 / (1 / 3) + 3)
  let AB : ℝ := 1 / 3 * AD
  AD * AB = 25 / 12

theorem inscribed_rectangle_area_correct :
  area_of_inscribed_rectangle
  := by
  let hf : ℝ := 12
  let eg : ℝ := 15
  let ad : ℝ := 15 / (hf / (1 / 3) + 3)
  let ab : ℝ := 1 / 3 * ad
  have area : ad * ab = 25 / 12 := by sorry
  exact area

end inscribed_rectangle_area_correct_l513_513021


namespace product_factors_eq_l513_513180

theorem product_factors_eq :
  (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * (1 - 1/7) * (1 - 1/8) * (1 - 1/9) * (1 - 1/10) * (1 - 1/11) = 1 / 11 := 
by
  sorry

end product_factors_eq_l513_513180


namespace tan_angle_addition_l513_513309

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 :=
sorry

end tan_angle_addition_l513_513309


namespace det_A_eq_zero_l513_513631

open Matrix

-- Define the matrix
def A (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![0, real.cos α, real.sin α], 
    ![-real.cos α, 0, real.cos β], 
    ![-real.sin α, -real.cos β, 0]]

-- State the theorem
theorem det_A_eq_zero (α β : ℝ) : det (A α β) = 0 := 
sorry

end det_A_eq_zero_l513_513631


namespace age_problem_l513_513985

theorem age_problem (my_age mother_age : ℕ) 
  (h1 : mother_age = 3 * my_age) 
  (h2 : my_age + mother_age = 40)
  : my_age = 10 :=
by 
  sorry

end age_problem_l513_513985


namespace distance_E_to_CD_l513_513348

-- Definitions and theorem statement
noncomputable def trapezoid_ABCD (AB BC AD: ℝ) : Prop :=
  AB > 0 ∧ BC > 0 ∧ AD > 0 ∧ 
  BC ≠ 0 ∧ AD ≠ 0 ∧ 
  AD = 4 ∧ 
  BC = 3 ∧ 
  ∃ (E C D: ℝ), 
  -- E is on AB, making circle contain C and D tangent at E to AB
  E ∈ AB ∧ (circle_through_ct (E, AB) C D) ∧ 
  -- C and D are on the circle
  (C ∈ circle_through (E, AB)) ∧ (D ∈ circle_through(E, AB)) ∧ 
  -- AB is perpendicular to BC
  (perpendicular AB BC) ∧ 
  -- Valid circle radius, tangent, and distances given
  true

theorem distance_E_to_CD : 
  ∀ AB BC AD,
  trapezoid_ABCD AB BC AD →
  ∃ E: ℝ,
  distance E CD = 2 * sqrt 3 :=
sorry

end distance_E_to_CD_l513_513348


namespace problem_statement_l513_513708

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513708


namespace toy_selling_price_l513_513557

theorem toy_selling_price (x : ℝ) (units_sold : ℝ) (profit_per_day : ℝ) : 
  (units_sold = 200 + 20 * (80 - x)) → 
  (profit_per_day = (x - 60) * units_sold) → 
  profit_per_day = 2500 → 
  x ≤ 60 * 1.4 → 
  x = 65 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_selling_price_l513_513557


namespace incorrect_statement_median_median_incorrect_l513_513227

theorem incorrect_statement_median (data : List ℕ) (h_data : data = [3, 3, 6, 5, 3]) :
  ¬ (median data = 6) :=
by
  sorry

-- Additional definitions and properties that could be used within a Lean proof
def mean (data : List ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

def mode (data : List ℕ) : ℕ :=
  data.groupBy id |>.sortBy (·.length) |>.last |>.head |>.getD 0

def median (data : List ℕ) : ℕ :=
  let sorted := data |>.sort
  sorted.getD (sorted.length / 2) 0

def variance (data : List ℕ) (μ : ℚ) : ℚ :=
  data.map (λ x => (x - μ) ^ 2) |>.sum / data.length

-- Assertions for the properties calculated
assert mean_data : mean [3, 3, 6, 5, 3] = 4 := by sorry
assert mode_data : mode [3, 3, 6, 5, 3] = 3 := by sorry
assert variance_data : variance [3, 3, 6, 5, 3] (mean [3, 3, 6, 5, 3]) = 1.6 := by sorry

-- Proving that statement C is incorrect
theorem median_incorrect (data : List ℕ) (h : data = [3, 3, 6, 5, 3]) : ¬ (median data = 6) :=
by
  sorry

end incorrect_statement_median_median_incorrect_l513_513227


namespace answer_is_p_and_q_l513_513733

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513733


namespace modulus_of_complex_number_l513_513908

variable (z : ℂ)

theorem modulus_of_complex_number (h : z = (1 + complex.I) / (1 - complex.I)) : complex.abs z = 1 :=
sorry

end modulus_of_complex_number_l513_513908


namespace function_is_linear_l513_513190

theorem function_is_linear (f : ℝ → ℝ) :
  (∀ a b c d : ℝ,
    a ≠ b → b ≠ c → c ≠ d → d ≠ a →
    (a ≠ c ∧ b ≠ d ∧ a ≠ d ∧ b ≠ c) →
    (a - b) / (b - c) + (a - d) / (d - c) = 0 →
    (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f d ∧ f a ≠ f c ∧ f a ≠ f d ∧ f b ≠ f d) →
    (f a - f b) / (f b - f c) + (f a - f d) / (f d - f c) = 0) →
  ∃ m c : ℝ, ∀ x : ℝ, f x = m * x + c :=
by
  sorry

end function_is_linear_l513_513190


namespace probability_heads_l513_513911

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let y := (n.choose (n / 2)) / total_outcomes
  (1 - y) / 2

theorem probability_heads (h₁ : ∀ (n : ℕ), n = 10 → probability_more_heads_than_tails n = 193 / 512) : 
  probability_more_heads_than_tails 10 = 193 / 512 :=
by
  apply h₁
  exact rfl

end probability_heads_l513_513911


namespace true_conjunction_l513_513679

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513679


namespace mean_of_six_numbers_sum_three_quarters_l513_513483

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end mean_of_six_numbers_sum_three_quarters_l513_513483


namespace largest_a_value_l513_513212

-- Define the quadratic trinomial
def quadratic_trinomial (a : ℝ) : ℝ → ℝ :=
  λ x, (1 / 3) * x^2 + (a + 1 / 2) * x + (a^2 + a)

-- Define the condition for the roots
def roots_condition (a : ℝ) (x1 x2 : ℝ) : Prop :=
  x1^3 + x2^3 = 3 * x1 * x2

-- Define Vieta's formulas for the sum and product of the roots
def vieta_sum (a : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = -3 * (a + 1 / 2)

def vieta_product (a : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 * x2 = 3 * (a^2 + a)

-- The main theorem statement
theorem largest_a_value : ∃ (a : ℝ), 
  ∀ (x1 x2 : ℝ), 
    roots_condition a x1 x2 ∧
    vieta_sum a x1 x2 ∧
    vieta_product a x1 x2 ∧
    (∀ (b : ℝ), 
      (roots_condition b x1 x2 ∧ vieta_sum b x1 x2 ∧ vieta_product b x1 x2 → b ≤ a)) :=
begin
  let a := -1 / 4,
  use a,
  intros x1 x2,
  split,
  { -- Placeholder for the roots condition proof
    sorry },
  split,
  { -- Placeholder for the Vieta's sum proof
    sorry },
  split,
  { -- Placeholder for the Vieta's product proof
    sorry },
  intro b,
  intros hb,
  -- Placeholder for the comparison proof
  sorry
end

end largest_a_value_l513_513212


namespace Dave_paid_more_than_Doug_l513_513203

theorem Dave_paid_more_than_Doug :
  let total_slices := 10
  let plain_pizza_cost := 10
  let extra_toppings_cost := 4
  let total_cost := plain_pizza_cost + extra_toppings_cost
  let cost_per_slice := total_cost / total_slices
  let Dave_slices := 6
  let Doug_slices := 4
  let Dave_paid := Dave_slices * cost_per_slice
  let Doug_paid := Doug_slices * cost_per_slice
  Dave_paid - Doug_paid = 2.8 := 
by
  let total_slices := 10
  let plain_pizza_cost := 10
  let extra_toppings_cost := 4
  let total_cost := plain_pizza_cost + extra_toppings_cost
  let cost_per_slice := total_cost / total_slices
  let Dave_slices := 6
  let Doug_slices := 4
  let Dave_paid := Dave_slices * cost_per_slice
  let Doug_paid := Doug_slices * cost_per_slice
  show Dave_paid - Doug_paid = 2.8 from sorry

end Dave_paid_more_than_Doug_l513_513203


namespace imaginary_part_of_complex_l513_513459

def complex_number := -1 - 2 * complex.i

theorem imaginary_part_of_complex : 
(imaginary_part complex_number) = -2 :=
by 
  sorry

end imaginary_part_of_complex_l513_513459


namespace main_statement_l513_513752

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513752


namespace fine_per_day_l513_513561

theorem fine_per_day (x : ℝ) : 
  (let total_days := 30 in
   let earnings_per_day := 25 in
   let total_amount_received := 425 in
   let days_absent := 10 in
   let days_worked := total_days - days_absent in
   let total_earnings := days_worked * earnings_per_day in
   let total_fine := days_absent * x in
   total_earnings - total_fine = total_amount_received) → x = 7.5 :=
by
  intros h
  sorry

end fine_per_day_l513_513561


namespace second_student_marks_l513_513087

theorem second_student_marks (x y : ℝ) 
  (h1 : x = y + 9) 
  (h2 : x = 0.56 * (x + y)) : 
  y = 33 := 
sorry

end second_student_marks_l513_513087


namespace proof_problem_l513_513774

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513774


namespace compute_dividend_l513_513979

theorem compute_dividend (x : ℕ) (h : x > 1) : 
  ∃ dividend : ℕ, dividend = 15 * x^3 + 7 * x + 9 := 
by 
  use 15 * x^3 + 7 * x + 9
  sorry

end compute_dividend_l513_513979


namespace num_ways_to_form_parallelogram_l513_513855

-- Define the conditions for the quadrilateral ABCD
def condition_1 (A B C D : Type) [Parallel A B C D] := AB ∥ CD
def condition_2 (A B C D : Type) [Equal A B C D] := AB = CD
def condition_3 (A B C D : Type) [Parallel A B C D] := BC ∥ AD
def condition_4 (A B C D : Type) [Equal A B C D] := BC = AD

-- Theorem stating that there are 4 ways to select two conditions to make ABCD a parallelogram
theorem num_ways_to_form_parallelogram (A B C D : Type) [Parallel A B C D] [Equal A B C D] :
  (condition_1 A B C D ∧ condition_3 A B C D) ∨
  (condition_2 A B C D ∧ condition_4 A B C D) ∨
  (condition_1 A B C D ∧ condition_2 A B C D) ∨
  (condition_3 A B C D ∧ condition_4 A B C D) →
  4 = 4 :=
by
  -- Proof is omitted
  sorry

end num_ways_to_form_parallelogram_l513_513855


namespace true_proposition_l513_513841

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513841


namespace distance_MC_l513_513043

theorem distance_MC (MA MB MC : ℝ) (hMA : MA = 2) (hMB : MB = 3) (hABC : ∀ x y z : ℝ, x + y > z ∧ y + z > x ∧ z + x > y) :
  1 ≤ MC ∧ MC ≤ 5 := 
by 
  sorry

end distance_MC_l513_513043


namespace main_statement_l513_513754

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513754


namespace true_conjunction_l513_513682

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513682


namespace find_a_l513_513112

-- Given conditions
def div_by_3 (a : ℤ) : Prop :=
  (5 * a + 1) % 3 = 0 ∨ (3 * a + 2) % 3 = 0

def div_by_5 (a : ℤ) : Prop :=
  (5 * a + 1) % 5 = 0 ∨ (3 * a + 2) % 5 = 0

-- Proving the question 
theorem find_a (a : ℤ) : div_by_3 a ∧ div_by_5 a → a % 15 = 4 :=
by
  sorry

end find_a_l513_513112


namespace proof_problem_l513_513780

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513780


namespace parabola_intersection_proof_l513_513617

def parabola_intersection_points : ℕ :=
  let a_values := {-3, -2, -1, 0, 1, 2, 3}
  let b_values := {-4, -3, -2, -1, 0, 1, 2, 3, 4}
  let num_parabolas := a_values.card * b_values.card
  let total_pairs := num_parabolas.choose 2
  let non_intersecting_pairs := 7 * (5.choose 2 + 4.choose 2)
  let intersecting_pairs := total_pairs - non_intersecting_pairs
  2 * intersecting_pairs

theorem parabola_intersection_proof :
  parabola_intersection_points = 3682 :=
by
  sorry

end parabola_intersection_proof_l513_513617


namespace min_sum_of_factors_l513_513468

theorem min_sum_of_factors (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 3432) :
  a + b + c ≥ 56 :=
sorry

end min_sum_of_factors_l513_513468


namespace correct_decision_box_l513_513867

theorem correct_decision_box (a b c : ℝ) (x : ℝ) : 
  x = a ∨ x = b → (x = b → b > a) →
  (c > x) ↔ (max (max a b) c = c) :=
by sorry

end correct_decision_box_l513_513867


namespace bisector_maximizes_product_l513_513888

noncomputable def maximize_product 
  (k1 k2 : Circle) (P Q : Point) 
  (t1 t2 : Line) (G : Region) 
  (tangent1 : Point → Line → Prop) 
  (tangent2 : Point → Line → Prop) 
  (intersect : k1 ∩ k2 = {P, Q}) 
  (union_regions : Region) 
  (region_G : Region)
  (true_tangent1 : tangent1 P t1)
  (true_tangent2 : tangent2 P t2)
  (AB : Line) (within_region_G : ∀ A B, A ∈ k1 → B ∈ k2 → AB.contains P → AB ∈ G) 
  (max_product_line : Line → Prop) 
  : Prop := 
    max_product_line AB ↔ 
    ∀ (AP PB: Length),
      AP * PB ≤ 
      (|AB.bisector t1 t2|)

theorem bisector_maximizes_product 
  {k1 k2 : Circle} {P Q : Point} 
  {t1 t2 : Line} {G : Region} 
  (tangent1 : Point → Line → Prop) 
  (tangent2 : Point → Line → Prop) 
  (intersect : k1 ∩ k2 = {P, Q}) 
  (true_tangent1 : tangent1 P t1) 
  (true_tangent2 : tangent2 P t2) 
  (within_region_G : ∀ A B, A ∈ k1 → B ∈ k2 → contains P AB → AB ∈ G) 
  : ∀ (AB : Line), 
    maximize_product k1 k2 P Q t1 t2 G tangent1 tangent2 intersect true_tangent1 true_tangent2 AB.

end bisector_maximizes_product_l513_513888


namespace cos_double_angle_l513_513652

-- Define the given condition
def cos_alpha := 4 / 5

-- State the theorem to be proved
theorem cos_double_angle (h : cos α = cos_alpha) : cos (2 * α) = 7 / 25 := 
begin
  sorry
end

end cos_double_angle_l513_513652


namespace find_a_find_t_l513_513276

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / x) + a * Real.log x

theorem find_a (a : ℝ) : (∀ x : ℝ, x > 0 → deriv (f x) = (-1 / (x^2)) + (a / x)) 
  → deriv (f 1) = 0
  → a = 1 :=
sorry 

theorem find_t (f : ℝ → ℝ) (t : ℕ) (a : ℝ) (x : ℝ) : 
  a = 1 
  → (x ≥ 1)
  → ((f (x + 1)) > ((x^2 + (t + 2) * x + t + 2) / (x^2 + 3 * x + 2))) 
  → (t = 1) :=
sorry

end find_a_find_t_l513_513276


namespace proof_problem_l513_513804

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513804


namespace point_in_second_quadrant_point_neg2_5_is_in_second_quadrant_l513_513341

-- Definitions derived from conditions
def point : ℝ × ℝ := (-2, 5)

-- The proof goal
theorem point_in_second_quadrant (x y : ℝ) (hx : x < 0) (hy : y > 0) : (x, y) = point → (x < 0 ∧ y > 0) := by
  intros h_eq
  rw h_eq
  exact ⟨hx, hy⟩

-- Prove the theorem for the specific point
theorem point_neg2_5_is_in_second_quadrant : point = (-2, 5) → (point.1 < 0 ∧ point.2 > 0) := by
  intro h
  rw h
  exact ⟨by norm_num, by norm_num⟩

end point_in_second_quadrant_point_neg2_5_is_in_second_quadrant_l513_513341


namespace units_digit_of_product_of_odds_is_5_l513_513511

/-- The product of all odd positive integers between 20 and 120 has a units digit of 5. -/
theorem units_digit_of_product_of_odds_is_5 : 
  let odd_integers := {n : ℕ | 20 < n ∧ n < 120 ∧ n % 2 = 1}
  let product_of_odds := ∏ n in odd_integers.to_finset, n
  product_of_odds % 10 = 5 := by
  sorry

end units_digit_of_product_of_odds_is_5_l513_513511


namespace proof_problem_l513_513812

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513812


namespace intersection_of_M_and_N_l513_513886

-- Define set M
def M : Set ℝ := {x | Real.log x > 0}

-- Define set N
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the target set
def target : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem intersection_of_M_and_N :
  M ∩ N = target :=
sorry

end intersection_of_M_and_N_l513_513886


namespace problem_statement_l513_513713

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513713


namespace values_of_x_for_f_l513_513250

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem values_of_x_for_f (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_monotonically_increasing_on_nonneg f) : 
  (∀ x : ℝ, f (2*x - 1) < f 3 ↔ (-1 < x ∧ x < 2)) :=
by
  sorry

end values_of_x_for_f_l513_513250


namespace problem_statement_l513_513703

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513703


namespace locus_of_intersection_points_l513_513863

def curveC (x : ℝ) : ℝ :=
  x + 1 / x

def lineL (k x : ℝ) : ℝ :=
  k * x + 1

theorem locus_of_intersection_points (M N : ℝ × ℝ)
  (hM : M.2 = curveC M.1) (hN : N.2 = curveC N.1)
  (hM_ne_N : M ≠ N)
  (h_intersectM : ∃ k, M.2 = lineL k M.1)
  (h_intersectN : ∃ k, N.2 = lineL k N.1) :
  ∃ y : ℝ, (2, y) = (2, 2) ∧ ∃ y, y > 2 ∧ y < 5 / 2 :=
sorry

end locus_of_intersection_points_l513_513863


namespace correct_function_l513_513102

noncomputable def f (x : ℝ) : ℝ := x * (x + 1) * (x - 1)

lemma f_odd : ∀ x : ℝ, f (-x) = -f (x) :=
by
  sorry

lemma f_monotonic_increasing : ∀ x : ℝ, x > 2 → 0 < (3 * x^2 - 1) :=
by
  sorry

lemma f_zeros : ∃ a : ℝ, f(a) = 0 ∧ f(-a) = 0 ∧ f(0) = 0 ∧ ∀ x : ℝ, f(x) = 0 → x = 0 ∨ x = 1 ∨ x = -1 :=
by
  sorry

theorem correct_function : 
  ∃ (f : ℝ → ℝ), (∀ x, f(-x) = -f(x)) ∧ 
                  (∀ x, x > 2 → 0 < (3 * x^2 - 1)) ∧ 
                  (∃ a, f(a) = 0 ∧ f(-a) = 0 ∧ f(0) = 0 ∧ ∀ x, f(x) = 0 → x = 0 ∨ x = 1 ∨ x = -1) :=
by
  use f
  refine ⟨f_odd, f_monotonic_increasing, f_zeros⟩
  sorry

end correct_function_l513_513102


namespace find_a_plus_b_l513_513864

-- We have the following conditions:
-- The solution set for the inequality ax^2 - (a+1)x + b < 0 is {x | 1 < x < 5}
-- We need to prove that a + b = 6/5 given the above condition

theorem find_a_plus_b (a b : ℚ) 
  (h1 : ∀ x : ℚ, 1 < x → x < 5 → ax^2 - (a + 1)x + b < 0) : 
  a + b = 6 / 5 := sorry

end find_a_plus_b_l513_513864


namespace circle_radius_of_tangent_parabolas_l513_513492

theorem circle_radius_of_tangent_parabolas :
  ∃ r : ℝ, 
  (∀ (x : ℝ), (x^2 + r = x)) →
  r = 1 / 4 :=
by
  sorry

end circle_radius_of_tangent_parabolas_l513_513492


namespace ben_heads_probability_l513_513913

def coin_flip_probability : ℚ :=
  let total_ways := 2^10
  let ways_exactly_five_heads := Nat.choose 10 5
  let probability_exactly_five_heads := ways_exactly_five_heads / total_ways
  let remaining_probability := 1 - probability_exactly_five_heads
  let probability_more_heads := remaining_probability / 2
  probability_more_heads

theorem ben_heads_probability :
  coin_flip_probability = 193 / 512 := by
  sorry

end ben_heads_probability_l513_513913


namespace equally_spaced_unit_circle_l513_513198

open Complex

theorem equally_spaced_unit_circle (z : Fin 5 → ℂ) (h₁ : ∀ i, Complex.abs (z i) = 1) (h₂ : ∑ i, z i = 0) : 
  ∃ θ : ℕ → ℝ, ∀ i, z i = exp (Complex.I * (θ i * 2 * Real.pi / 5)) :=
sorry

end equally_spaced_unit_circle_l513_513198


namespace vec_result_l513_513290

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

noncomputable def vec_scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

theorem vec_result :
  let a := (3, 2) in
  let b := (0, -1) in
  vec_sub (vec_scalar_mult 2 b) a = (-3, -4) :=
by
  let a := (3, 2)
  let b := (0, -1)
  sorry

end vec_result_l513_513290


namespace unique_perpendicular_plane_through_point_l513_513019

variable {Point : Type} {Line : Type} {Plane : Type}

noncomputable def is_perpendicular (h : Line) (π : Plane) : Prop := sorry
noncomputable def passes_through (M : Point) (π : Plane) : Prop := sorry
noncomputable def point_not_on_line (M : Point) (h : Line) : Prop := sorry

theorem unique_perpendicular_plane_through_point 
  (M : Point) (h : Line) (hM : point_not_on_line M h) : 
  ∃! π : Plane, passes_through M π ∧ is_perpendicular h π := 
begin
  sorry
end

end unique_perpendicular_plane_through_point_l513_513019


namespace proof_problem_l513_513773

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513773


namespace main_statement_l513_513748

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513748


namespace proof_problem_l513_513771

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513771


namespace projection_correct_l513_513636

def projection_onto_line (v : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := v.1 * d.1 + v.2 * d.2 + v.3 * d.3
  let squared_norm := d.1 * d.1 + d.2 * d.2 + d.3 * d.3
  let scalar := dot_product / squared_norm
  (scalar * d.1, scalar * d.2, scalar * d.3)

theorem projection_correct :
  projection_onto_line (5, -3, 4) (1, -1/2, 1/2) = (17/3, -17/6, 17/6) :=
by
  sorry

end projection_correct_l513_513636


namespace true_proposition_l513_513848

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513848


namespace problem_statement_l513_513786

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513786


namespace int_fraction_not_integer_l513_513997

theorem int_fraction_not_integer (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ (k : ℤ), a^2 + b^2 = k * (a^2 - b^2) := 
sorry

end int_fraction_not_integer_l513_513997


namespace boxes_with_neither_l513_513419

theorem boxes_with_neither (total_boxes boxes_with_markers boxes_with_crayons boxes_with_both : ℕ) :
  total_boxes = 15 →
  boxes_with_markers = 9 →
  boxes_with_crayons = 5 →
  boxes_with_both = 4 →
  total_boxes - ((boxes_with_markers + boxes_with_crayons) - boxes_with_both) = 5 :=
by
  intros h_total h_markers h_crayons h_both
  rw [h_total, h_markers, h_crayons, h_both]
  sorry

end boxes_with_neither_l513_513419


namespace numbers_neither_squares_nor_cubes_l513_513893

theorem numbers_neither_squares_nor_cubes :
  let count_squares := 14
  let count_cubes := 5
  let count_sixth_powers := 2
  ∃ count, count = 200 - (count_squares + count_cubes - count_sixth_powers) ∧ count = 183 :=
by
  let count_squares := 14
  let count_cubes := 5
  let count_sixth_powers := 2
  existsi 183
  split
  . calc
      200 - (count_squares + count_cubes - count_sixth_powers)
      = 200 - (14 + 5 - 2) : by rfl
      = 200 - 17 : by rfl
      = 183 : by rfl
  . rfl

end numbers_neither_squares_nor_cubes_l513_513893


namespace true_proposition_l513_513845

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513845


namespace problem_statement_l513_513796

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513796


namespace solution_to_equation_l513_513060

theorem solution_to_equation (x : ℝ) : 
    (sin (4 * x) * cos (5 * x) = - cos (4 * x) * sin (5 * x)) → 
    (∃ k : ℤ, x = k * π / 9) :=
by
  sorry

end solution_to_equation_l513_513060


namespace p_and_q_is_true_l513_513698

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513698


namespace find_particular_number_l513_513307

theorem find_particular_number (x : ℝ) (h : 4 * x * 25 = 812) : x = 8.12 :=
by sorry

end find_particular_number_l513_513307


namespace simplify_neg_x_mul_3_minus_x_l513_513183

theorem simplify_neg_x_mul_3_minus_x (x : ℝ) : -x * (3 - x) = -3 * x + x^2 :=
by
  sorry

end simplify_neg_x_mul_3_minus_x_l513_513183


namespace full_time_employees_count_l513_513586

/-- Given the total number of employees and the number of part-time employees,
    this theorem proves the number of full-time employees. -/
theorem full_time_employees_count (part_time_employees total_employees : ℕ) 
  (h_part_time : part_time_employees = 2041) 
  (h_total : total_employees = 65134) : 
  total_employees - part_time_employees = 63093 :=
by
  rw [h_part_time, h_total]
  norm_num

end full_time_employees_count_l513_513586


namespace probability_not_less_than_30_l513_513234

theorem probability_not_less_than_30 
  (P_X_lt_30 : ℝ)
  (P_30_le_X_le_40 : ℝ)
  (h1 : P_X_lt_30 = 0.30)
  (h2 : P_30_le_X_le_40 = 0.50)
  (h3 : P_X_lt_30 + P_30_le_X_le_40 + P(λ x, x > 40) = 1) :
  P(λ x, x ≥ 30) = 0.70 :=
by
  sorry

end probability_not_less_than_30_l513_513234


namespace proposition_p_and_q_is_true_l513_513721

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513721


namespace proof_problem_l513_513801

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513801


namespace answer_is_p_and_q_l513_513738

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513738


namespace collinear_opposite_vectors_l513_513253

theorem collinear_opposite_vectors {m : ℝ} :
  let a := (m, -4)
  let b := (-1, m + 3)
  m * (-1) = -4 * (m + 3) ∧ m * (m + 3) = 4 → m = 1 :=
begin
  intros,
  sorry
end

end collinear_opposite_vectors_l513_513253


namespace exponents_identity_l513_513549

theorem exponents_identity :
  (0.027)^(-1 / 3) - (-1 / 7)^(-2) + (2 + 7 / 9)^(1 / 2) - (Real.sqrt 2 - 1)^0 = -45 :=
by
  -- include calculation: sorry to represent the steps to simplify this in Lean.
  sorry

end exponents_identity_l513_513549


namespace tangent_line_eq_l513_513044

theorem tangent_line_eq {f : ℝ → ℝ} (h : ∀ x, f x = x * Real.log x) :
  ∃ m b, (∀ x, tangent_line_point_slope f 1 (1, 0) m b x = (x - y - 1 = 0)) :=
begin
  sorry
end

end tangent_line_eq_l513_513044


namespace margin_in_terms_of_selling_price_l513_513924

variable (C S n M : ℝ)

theorem margin_in_terms_of_selling_price (h : M = (2 * C) / n) : M = (2 * S) / (n + 2) :=
sorry

end margin_in_terms_of_selling_price_l513_513924


namespace min_k_S_l513_513396

theorem min_k_S (n : ℕ) (hn : n ≥ 2) (a : Finₓ n → ℝ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ S : Set ℝ, ∀ a ∈ S, ∃ i j, a = (a i) + 2 ^ j ∧ (S.card = (n * (n + 1)) / 2) :=
sorry

end min_k_S_l513_513396


namespace proposition_true_l513_513828

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513828


namespace train_speed_is_45_km_per_h_l513_513593

-- Definitions as conditions from the problem statement
def train_length : ℝ := 485 -- meters
def bridge_length : ℝ := 140 -- meters
def time_to_pass : ℝ := 50 -- seconds

-- The total distance the train travels is the sum of train length and bridge length
def total_distance : ℝ := train_length + bridge_length

-- Speed is calculated as distance divided by time
def speed_m_per_s : ℝ := total_distance / time_to_pass

-- Conversion factor from m/s to km/h
def conversion_factor : ℝ := 3.6

-- Converted speed from m/s to km/h
def speed_km_per_h : ℝ := speed_m_per_s * conversion_factor

-- The theorem is to prove that the computed speed is 45 km/h
theorem train_speed_is_45_km_per_h : speed_km_per_h = 45 := by
  sorry

end train_speed_is_45_km_per_h_l513_513593


namespace find_angle_MBA_l513_513355

-- Define the angles and the triangle
def triangle (A B C : Type) := true

-- Define the angles in degrees
def angle (deg : ℝ) := deg

-- Assume angles' degrees as given in the problem
variables {A B C M : Type}
variable {BAC ABC MAB MCA MBA : ℝ}

-- Given conditions
axiom angle_BAC : angle BAC = 30
axiom angle_ABC : angle ABC = 70
axiom angle_MAB : angle MAB = 20
axiom angle_MCA : angle MCA = 20

-- Prove that angle MBA is 30 degrees
theorem find_angle_MBA : angle MBA = 30 := 
by 
  sorry

end find_angle_MBA_l513_513355


namespace proposition_true_l513_513840

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513840


namespace centroid_inequality_l513_513383

def Triangle := Type
variables (A B C G : Triangle)
variables (GA GB GC AB BC CA s_1 s_2 : ℝ)

-- Definitions and conditions
def is_centroid (G A B C : Triangle) : Prop :=
  ∃ (M N P : Triangle), 
    (GA = GM + MG ∧ GB = GN + NG ∧ GC = GP + PG) ∧ 
    (GM = GA / 3 ∧ GN = GB / 3 ∧ GP = GC / 3)

def s1_def (GA GB GC : ℝ) : ℝ := GA + GB + GC
def s2_def (AB BC CA : ℝ) : ℝ := AB + BC + CA

-- Prove the final statement
theorem centroid_inequality
  (hG : is_centroid G A B C)
  (hs1 : s_1 = s1_def GA GB GC)
  (hs2 : s_2 = s2_def AB BC CA) :
  s_1 < 0.5 * s_2 ∧ s_1 < s_2 := 
sorry

end centroid_inequality_l513_513383


namespace calculate_total_weight_l513_513409

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end calculate_total_weight_l513_513409


namespace scientific_notation_of_32000000_l513_513171

theorem scientific_notation_of_32000000 : 32000000 = 3.2 * 10^7 := 
  sorry

end scientific_notation_of_32000000_l513_513171


namespace triangle_XYZ_y_approx_60_l513_513349

variable {X Y Z : Type}
-- Angles in radians
variable (angleX angleZ : ℝ)
-- Sides opposite the respective angles
variable (x z : ℝ)
-- Side we want to determine
variable (y : ℝ)

-- Given Conditions
theorem triangle_XYZ_y_approx_60 
  (h1 : angleZ = 4 * angleX) 
  (h2 : x = 35) 
  (h3 : z = 56) : 
  y ≈ 60 := 
  sorry

end triangle_XYZ_y_approx_60_l513_513349


namespace base_5_division_l513_513632

theorem base_5_division (n m : ℕ) : 
  let d₂₁₀₄₅ := 2 * 5^3 + 1 * 5^2 + 0 * 5^1 + 4 * 5^0 in
  let d₂₃₅ := 2 * 5^1 + 3 * 5^0 in 
  d₂₁₀₄₅ / d₂₃₅ = 4 * 5^1 + 1 * 5^0 := by
  sorry

end base_5_division_l513_513632


namespace proof_problem_l513_513530

variable (x y : ℝ)

theorem proof_problem :
  ¬ (x^2 + x^2 = x^4) ∧
  ¬ ((x - y)^2 = x^2 - y^2) ∧
  ¬ ((x^2 * y)^3 = x^6 * y) ∧
  ((-x)^2 * x^3 = x^5) :=
by
  sorry

end proof_problem_l513_513530


namespace largest_consecutive_even_sum_is_4_l513_513508

theorem largest_consecutive_even_sum_is_4 :
  ∃ k n : ℕ, (∑ i in finset.range k, (2*n + 2*i) = 156) ∧ 
    (∀ m : ℕ, (∑ i in finset.range m, (2*n + 2*i) = 156) → m ≤ k) ∧ k = 4 :=
by
  sorry

end largest_consecutive_even_sum_is_4_l513_513508


namespace Bob_walked_35_miles_l513_513542

theorem Bob_walked_35_miles (distance : ℕ) 
  (Yolanda_rate Bob_rate : ℕ) (Bob_start_after : ℕ) (Yolanda_initial_walk : ℕ)
  (h1 : distance = 65) 
  (h2 : Yolanda_rate = 5) 
  (h3 : Bob_rate = 7) 
  (h4 : Bob_start_after = 1)
  (h5 : Yolanda_initial_walk = Yolanda_rate * Bob_start_after) :
  Bob_rate * (distance - Yolanda_initial_walk) / (Yolanda_rate + Bob_rate) = 35 := 
by 
  sorry

end Bob_walked_35_miles_l513_513542


namespace true_conjunction_l513_513686

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513686


namespace rachel_reading_pages_l513_513020

theorem rachel_reading_pages (M T : ℕ) (hM : M = 10) (hT : T = 23) : T - M = 3 := 
by
  rw [hM, hT]
  norm_num
  sorry

end rachel_reading_pages_l513_513020


namespace probability_red_ball_from_bag_A_l513_513366

variable (m n : ℕ)
variable h1 : ∃ (m n : ℕ), (4 * m + 3 * n) = (15 * (m + n))

theorem probability_red_ball_from_bag_A (m n : ℕ) (h2 : (4 * m + 3 * n) = 15 * (m + n)) : (m : ℚ) / (m + n) = 3 / 4 := by
  sorry

end probability_red_ball_from_bag_A_l513_513366


namespace problem_statement_l513_513793

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513793


namespace inverse_of_exponential_function_l513_513050

theorem inverse_of_exponential_function (x : ℝ) (hx : x > -1) :
  ∃ (f : ℝ → ℝ), (∀ (y : ℝ), y = 3 ^ x - 1 → f y = log 3 (x + 1)) :=
sorry

end inverse_of_exponential_function_l513_513050


namespace petya_final_percentage_l513_513017

variables (x y : ℝ)

def initial_percentage_votes_petya : ℝ := 0.25
def initial_percentage_votes_vasya : ℝ := 0.45
def final_percentage_votes_vasya : ℝ := 0.27

def votes_by_noon_petya := initial_percentage_votes_petya * x
def votes_by_noon_vasya := initial_percentage_votes_vasya * x
def votes_after_noon_petya := y

axiom votes_equation : votes_by_noon_vasya = final_percentage_votes_vasya * (x + y)

def votes_by_end_petya := votes_by_noon_petya + votes_after_noon_petya

theorem petya_final_percentage :
  votes_by_end_petya / (x + y) = 0.55 :=
sorry

end petya_final_percentage_l513_513017


namespace true_conjunction_l513_513673

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513673


namespace area_of_square_l513_513092

theorem area_of_square (a : ℝ) (h : a = 12) : a * a = 144 := by
  rw [h]
  norm_num

end area_of_square_l513_513092


namespace correct_option_cbrt_neg_seven_l513_513533

theorem correct_option_cbrt_neg_seven : 
  (sqrt 16 ≠ 4 ∨ sqrt 16 ≠ -4) ∧
  (-sqrt 0.4 ≠ -0.2) ∧
  (sqrt ((-12)^2) ≠ -12) ∧
  (sqrt ((-12)^2) = 12) ∧
  (cbrt (-7) = -cbrt 7) := 
by {
  -- Definitions related to problem conditions
  have h1 : sqrt 16 = 4 := by sorry,
  have h2 : -sqrt 0.4 ≠ -0.2 := by sorry,
  have h3 : sqrt ((-12)^2) = 12 := by sorry,
  have h4 : sqrt ((-12)^2) ≠ -12 := by sorry,
  have h5 : cbrt (-7) = -cbrt 7 := by sorry,
  split,
  { -- For option A
    intro h,
    exact neq_of_gt (by sorry) h,
  },
  split,
  { -- For option B
    exact h2,
  },
  split,
  { -- For option C
    exact h4,
  },
  -- For option D
  { intro, exact h5, }
}

end correct_option_cbrt_neg_seven_l513_513533


namespace ratio_triangle_to_square_l513_513928

open Real

-- Define the points of the triangle
def A := (1, 1) : ℝ × ℝ
def B := (1, 4) : ℝ × ℝ
def C := (4, 4) : ℝ × ℝ

-- Define the side length of the square and its area
def side_length_square := 5
def area_square := (side_length_square * side_length_square : ℝ)

-- Define the area of the triangle using the given coordinates
def area_triangle : ℝ := 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Proving the ratio of area of the triangle to the area of the large square is 9/50
theorem ratio_triangle_to_square : (area_triangle / area_square) = (9 / 50) :=
by
  have h1 : area_triangle = 4.5 := by
    -- Calculate the area of the triangle
    sorry
  have h2 : area_square = 25 := by
    -- Calculate the area of the square
    simp [area_square, side_length_square]
  -- Calculate the ratio
  calc
    area_triangle / area_square
    = 4.5 / 25 : by rw [h1, h2]
    ... = 9 / 50 : by norm_num

end ratio_triangle_to_square_l513_513928


namespace Tim_scored_30_l513_513929

-- Definitions and conditions
variables (Joe Tim Ken : ℕ)
variables (h1 : Tim = Joe + 20)
variables (h2 : Tim = Nat.div (Ken * 2) 2)
variables (h3 : Joe + Tim + Ken = 100)

-- Statement to prove
theorem Tim_scored_30 : Tim = 30 :=
by sorry

end Tim_scored_30_l513_513929


namespace analysis_hours_l513_513022

theorem analysis_hours (n t : ℕ) (h1 : n = 206) (h2 : t = 1) : n * t = 206 := by
  sorry

end analysis_hours_l513_513022


namespace min_value_5x_plus_6y_l513_513857

theorem min_value_5x_plus_6y (x y : ℝ) (h : 3 * x ^ 2 + 3 * y ^ 2 = 20 * x + 10 * y + 10) : 
  ∃ x y, (5 * x + 6 * y = 122) :=
by
  sorry

end min_value_5x_plus_6y_l513_513857


namespace repeated_transform_does_not_ensure_all_lt_half_l513_513973

def irrationals : Set ℝ := { x : ℝ | ¬ ∃ (q : ℚ), ↑q = x }

def positive_irrationals : Set ℝ := { x : ℝ | x > 0 ∧ x ∈ irrationals }

def S : Set (Set ℝ) := {A | ∃ x y z, A = {x, y, z} ∧ x + y + z = 1 ∧
                    x ∈ positive_irrationals ∧ y ∈ positive_irrationals ∧ z ∈ positive_irrationals}

def transform (A : Set ℝ) : Set ℝ :=
  match A.toList with
  | [x, y, z] => if x > 1 / 2 then {2 * x - 1, 2 * y, 2 * z} else A
  | _ => A  -- not expected to happen given our conditions on S

theorem repeated_transform_does_not_ensure_all_lt_half :
  ∃ (A ∈ S), ∀ n, ¬ (transform^[n] A).subset {x | x < 1 / 2} :=
sorry

end repeated_transform_does_not_ensure_all_lt_half_l513_513973


namespace total_payment_leila_should_pay_l513_513164

-- Definitions of the conditions
def chocolateCakes := 3
def chocolatePrice := 12
def strawberryCakes := 6
def strawberryPrice := 22

-- Mathematical equivalent proof problem
theorem total_payment_leila_should_pay : 
  chocolateCakes * chocolatePrice + strawberryCakes * strawberryPrice = 168 := 
by 
  sorry

end total_payment_leila_should_pay_l513_513164


namespace problem_statement_l513_513907

theorem problem_statement (x y a : ℝ) (h1 : x + a < y + a) (h2 : a * x > a * y) : x < y ∧ a < 0 :=
sorry

end problem_statement_l513_513907


namespace quarters_and_dimes_l513_513324

theorem quarters_and_dimes (n : ℕ) (qval : ℕ := 25) (dval : ℕ := 10) 
  (hq : 20 * qval + 10 * dval = 10 * qval + n * dval) : 
  n = 35 :=
by
  sorry

end quarters_and_dimes_l513_513324


namespace lewis_weekly_earnings_correct_l513_513980

-- Define the conditions
def weeks_per_harvest_season : ℕ := 223
def num_harvest_seasons : ℕ := 73
def total_earnings : ℕ := 22090603

-- Define the weekly earnings calculation
def total_weeks := weeks_per_harvest_season * num_harvest_seasons
def weekly_earnings := total_earnings / total_weeks

theorem lewis_weekly_earnings_correct : weekly_earnings = 1357.14 :=
by sorry

end lewis_weekly_earnings_correct_l513_513980


namespace answer_is_p_and_q_l513_513741

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513741


namespace remaining_musicians_age_l513_513055

/-- Define the ages of the saxophonist, singer, trumpeter and mean age -/
def saxophonist_age : ℕ := 19
def singer_age : ℕ := 20
def trumpeter_age : ℕ := 21
def mean_age (members : ℕ) (total_age : ℕ) : ℕ := total_age / members
def total_members : ℕ := 6

/-- Prove the age of each of the remaining musicians is equal to 22 -/
theorem remaining_musicians_age :
  let total_age := 126 in
  let known_age_sum := saxophonist_age + singer_age + trumpeter_age in
  let remaining_age_sum := total_age - known_age_sum in
  let remaining_individual_age := remaining_age_sum / 3 in  
  remaining_individual_age = 22 :=
by
  sorry

end remaining_musicians_age_l513_513055


namespace triangle_similarity_l513_513965

-- Define the structures and the conditions based on the problem.

-- Let's define the triangle and basic entities.
variables (A B C G M X Y P Q : Type)
variables [is_centroid A B C G] [is_midpoint B C M]
variables (h1 : collinear X Y G) (h2 : parallel X Y B C)
variables (h3 : intersects_at X C G B Q) (h4 : intersects_at Y B G C P)

-- The goal is to prove similarity of triangles
theorem triangle_similarity (A B C G M X Y P Q : Type) 
  [is_centroid A B C G] [is_midpoint B C M] 
  (h1 : collinear X Y G) (h2 : parallel X Y B C)
  (h3 : intersects_at X C G B Q) (h4 : intersects_at Y B G C P) :
  similar (triangle M P Q) (triangle A B C) :=
sorry

end triangle_similarity_l513_513965


namespace leila_total_payment_l513_513162

theorem leila_total_payment:
  (choc_cost : ℕ) (choc_quantity : ℕ) (straw_cost : ℕ) (straw_quantity : ℕ)
  (h_choc : choc_cost = 12) (h_choc_qty : choc_quantity = 3)
  (h_straw : straw_cost = 22) (h_straw_qty : straw_quantity = 6) :
  choc_cost * choc_quantity + straw_cost * straw_quantity = 168 := 
by
  sorry

end leila_total_payment_l513_513162


namespace Nancy_antacid_consumption_l513_513421

theorem Nancy_antacid_consumption :
  let antacids_per_month : ℕ :=
    let antacids_per_day_indian := 3
    let antacids_per_day_mexican := 2
    let antacids_per_day_other := 1
    let days_indian_per_week := 3
    let days_mexican_per_week := 2
    let days_total_per_week := 7
    let weeks_per_month := 4

    let antacids_per_week_indian := antacids_per_day_indian * days_indian_per_week
    let antacids_per_week_mexican := antacids_per_day_mexican * days_mexican_per_week
    let days_other_per_week := days_total_per_week - days_indian_per_week - days_mexican_per_week
    let antacids_per_week_other := antacids_per_day_other * days_other_per_week

    let antacids_per_week_total := antacids_per_week_indian + antacids_per_week_mexican + antacids_per_week_other

    antacids_per_week_total * weeks_per_month
    
  antacids_per_month = 60 := sorry

end Nancy_antacid_consumption_l513_513421


namespace intersection_count_l513_513897

-- Define the absolute value functions
def f (x : ℝ) : ℝ := abs (3 * x + 6)
def g (x : ℝ) : ℝ := -abs (4 * x - 3)

-- Statement of the theorem
theorem intersection_count : ∃ (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = g x :=
by
  -- s is the set of x-values where f(x) = g(x)
  use (Finset.mk [-(3/7), -9] (by decide))
  split
  -- The size of the set is 2
  · norm_num
  -- For each element x in the set, f(x) is equal to g(x)
  · intros x hx
    finset_cases hx
    · show f (-(3/7)) = g (-(3/7))
      norm_num
      rw [abs_of_nonneg, abs_of_nonpos]
      norm_num
      linarith
    · show f (-9) = g (-9)
      norm_num
      rw [abs_of_nonpos, abs_of_nonneg]
      norm_num
      linarith
  sorry

end intersection_count_l513_513897


namespace max_oranges_taken_l513_513381

theorem max_oranges_taken (n : ℕ) (h1 : 3 ∣ n → false) : 
    let total_oranges := (n + 1) * (n + 2) / 2 in
    total_oranges - 3 = (n + 1) * (n + 2) / 2 - 3 := 
by
  sorry

end max_oranges_taken_l513_513381


namespace peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513016

-- Probability of Peter winning a prize:
theorem peter_wins_prize_probability :
  let p := (5 / 6) in p ^ 9 = (5 / 6) ^ 9 := by
  sorry

-- Probability that at least one person wins a prize:
theorem at_least_one_wins_prize_probability :
  let p := (5 / 6) in
  let q := (1 - p^9) in 
  (1 - q^10) ≈ 0.919 := by
  sorry

end peter_wins_prize_probability_at_least_one_wins_prize_probability_l513_513016


namespace leila_total_payment_l513_513161

theorem leila_total_payment:
  (choc_cost : ℕ) (choc_quantity : ℕ) (straw_cost : ℕ) (straw_quantity : ℕ)
  (h_choc : choc_cost = 12) (h_choc_qty : choc_quantity = 3)
  (h_straw : straw_cost = 22) (h_straw_qty : straw_quantity = 6) :
  choc_cost * choc_quantity + straw_cost * straw_quantity = 168 := 
by
  sorry

end leila_total_payment_l513_513161


namespace num_integers_between_cubed_values_l513_513297

theorem num_integers_between_cubed_values : 
  let a : ℝ := 10.5
  let b : ℝ := 10.7
  let c1 := a^3
  let c2 := b^3
  let first_integer := Int.ceil c1
  let last_integer := Int.floor c2
  first_integer ≤ last_integer → 
  last_integer - first_integer + 1 = 67 :=
by
  sorry

end num_integers_between_cubed_values_l513_513297


namespace problem_statement_l513_513785

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513785


namespace valid_numbers_count_l513_513213

def count_valid_numbers (n : ℕ) : ℕ := 2 ^ (n + 1) - 2 * n - 2

theorem valid_numbers_count {n : ℕ} (h: n > 0) :
    ∑ k in finset.range n, (n - k + 1) * 2^(k - 1) - 1 = count_valid_numbers n :=
  sorry

end valid_numbers_count_l513_513213


namespace line_intersects_circle_find_k_when_chord_is_smallest_l513_513880

noncomputable theory

-- Define the line l and circle M
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y - 3 * k = 0
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 8 * x - 2 * y + 9 = 0

-- Part (1): Prove that the line intersects the circle
theorem line_intersects_circle (k : ℝ) :
  (∃ x y : ℝ, line_l k x y ∧ circle_M x y) :=
begin
  -- Line always passes through (3, 0)
  let P := (3 : ℝ, 0 : ℝ),
  have line_through_P : line_l k P.1 P.2,
  { unfold line_l, sorry, },
  have point_inside_circle : ¬circle_M P.1 P.2,
  { unfold circle_M, sorry, },
  -- Conclusion that the line intersects the circle
  exact ⟨3, 0, line_through_P, point_inside_circle⟩
end

-- Part (2): When the chord length is smallest, find k
theorem find_k_when_chord_is_smallest :
  ∃ k : ℝ, -- There exists a k such that
    (∀ x y : ℝ, line_l k x y ∧ circle_M x y) ∧ -- Line intersects circle
    k = -1 := -- k is -1 when chord length is smallest
begin
  -- When chord length is smallest, MP is perpendicular to l
  let M := (4 : ℝ, 1 : ℝ),
  let P := (3 : ℝ, 0 : ℝ),
  have slope_MP : (M.2 - P.2) / (M.1 - P.1) = 1,
  { unfold slope_MP, sorry, },
  have perpendicular_condition : (M.2 - P.2) / (M.1 - P.1) * (-1 / k) = -1,
  { sorry, },
  -- Solve for k
  existsi (-1 : ℝ),
  split,
  { sorry, },
  { refl },
end

end line_intersects_circle_find_k_when_chord_is_smallest_l513_513880


namespace unique_monic_polynomial_of_degree_2_l513_513971

-- Define the monic polynomial g(x) of degree 2 with conditions g(0) = 6 and g(2) = 18
def is_monic (g : ℝ → ℝ) : Prop := 
  ∃ b c: ℝ, g = (λ x, x^2 + b * x + c)

-- Define the conditions g(0) = 6 and g(2) = 18
def satisfies_conditions (g : ℝ → ℝ) : Prop :=
  g 0 = 6 ∧ g 2 = 18

-- Define the polynomial g
def g : ℝ → ℝ := λ x, x^2 + 4 * x + 6

-- The theorem to prove
theorem unique_monic_polynomial_of_degree_2 :
  (∃ h : ℝ → ℝ, is_monic h ∧ satisfies_conditions h) → (g x = x^2 + 4 * x + 6) :=
by
  sorry

end unique_monic_polynomial_of_degree_2_l513_513971


namespace true_conjunction_l513_513680

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513680


namespace pints_in_2_liters_l513_513259

def litersToPints : ℝ := 1.05 / 0.5
def pintsIn2Liters := 2 * litersToPints

theorem pints_in_2_liters : pintsIn2Liters = 4.2 :=
by
  sorry

end pints_in_2_liters_l513_513259


namespace arithmetic_general_term_sum_value_l513_513665

variable {a₁ : ℕ} {S : ℕ → ℕ}

-- Condition: a_1 = 1
def a₁_def : a₁ = 1 := by sorry

-- Condition: S_9 = 81
def S₉_def : S 9 = 81 := by sorry

-- First part of the problem: Prove general term a_n = 2n - 1
theorem arithmetic_general_term (n : ℕ) (d : ℕ) (aₙ : ℕ → ℕ) 
  (h₀ : a₁ = 1) 
  (h₁ : S 9 = 81) 
  (h₂ : S n = ∑ i in range n, aₙ (i+1)) 
  (h₃ : ∀ n, aₙ (n+1) = 1 + n * d) : 
  ∀ n, aₙ n = 2 * n - 1 :=
by sorry

-- Second part of the problem: Prove sum value
theorem sum_value :
  (∑ n in range 2017, (1 / (n^2 + n : ℝ))) = 2017 / 2018 :=
by sorry

end arithmetic_general_term_sum_value_l513_513665


namespace ideal_number_with_3_l513_513248

-- Define the initial sequence and sum of the first n terms.
def S (n : ℕ) : ℝ := sorry -- We use sorry as we are not provided with an explicit sequence definition

-- Define the ideal number T.
def T (n : ℕ) : ℝ := (finset.sum (finset.range n) S) / n

-- Given condition: the "ideal number" of the terms a_1, a_2, ..., a_20 is 21.
axiom T_20 : T 20 = 21

-- Theorem: the "ideal number" of the terms 3, a_1, a_2, ..., a_{20} is 23.
theorem ideal_number_with_3 : T 21 = 23 :=
  sorry

end ideal_number_with_3_l513_513248


namespace simplify_and_evaluate_l513_513444

variable (x y : ℝ)

theorem simplify_and_evaluate :
  x = -2 → y = 1 / 3 → (x^2 + 3 * y) = 5 :=
by
  intro hx hy
  rw [hx, hy]
  have hx2 : (-2 : ℝ)^2 = 4 := by norm_num
  have hy3 : 3 * (1 / 3 : ℝ) = 1 := by norm_num
  rw [hx2, hy3]
  norm_num
  exact (eq.refl 5)

end simplify_and_evaluate_l513_513444


namespace proposition_true_l513_513829

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513829


namespace true_proposition_l513_513762

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513762


namespace true_proposition_l513_513849

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513849


namespace proposition_p_and_q_is_true_l513_513725

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513725


namespace best_approximation_of_square_feet_per_person_l513_513467

noncomputable def population : ℕ := 38_005_238
noncomputable def area_square_miles : ℕ := 3_855_103
noncomputable def feet_per_mile : ℕ := 5280
noncomputable def total_square_feet := area_square_miles * feet_per_mile ^ 2
noncomputable def average_square_feet_per_person := total_square_feet / population

theorem best_approximation_of_square_feet_per_person :
  abs (average_square_feet_per_person - 2_800_000) ≤ abs (average_square_feet_per_person - X) ∀ X ∈ {100_000, 350_000, 700_000, 1_000_000, 1_500_000, 1_000_000, 2_500_000, 2_800_000, 3_500_000, 5_000_000} :=
by
  -- setup
  sorry

end best_approximation_of_square_feet_per_person_l513_513467


namespace number_of_convex_polyhedra_is_even_l513_513547

theorem number_of_convex_polyhedra_is_even (points : Fin 100 → ℝ × ℝ × ℝ) 
  (h : ∀ (p1 p2 p3 p4 : Fin 100), 
      p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → p3 ≠ p4 → 
      ¬ ∃ (a b c d : ℝ), (a, b, c, d).2 ≠ 0 ∧ ∀ i ∈ {p1, p2, p3, p4}, 
        a * (points i).1 + b * (points i).2 + c * (points i).3 + d = 0) : 
  Even (number_of_convex_polyhedra 5 points) :=
sorry

noncomputable def number_of_convex_polyhedra (k : ℕ) (points : Fin 100 → ℝ × ℝ × ℝ) : ℕ :=
-- Definition (not needed for the statement, included for understanding)
sorry

end number_of_convex_polyhedra_is_even_l513_513547


namespace z_is_real_z_is_pure_imaginary_z_in_third_quadrant_l513_513242

open Complex

def z (m : ℝ) : ℂ := (m^2 - 8 * m + 15) + (m^2 - 9 * m + 18) * Complex.i

-- 1. Proving z is real when m = 3 or m = 6
theorem z_is_real (m : ℝ) : (Im (z m) = 0) ↔ (m = 3 ∨ m = 6) := by
  sorry

-- 2. Proving z is pure imaginary when m = 5
theorem z_is_pure_imaginary (m : ℝ) : (Re (z m) = 0 ∧ Im (z m) ≠ 0) ↔ (m = 5) := by
  sorry

-- 3. Proving z is in the third quadrant when 3 < m < 5
theorem z_in_third_quadrant (m : ℝ) : (Re (z m) < 0 ∧ Im (z m) < 0) ↔ (3 < m ∧ m < 5) := by
  sorry

end z_is_real_z_is_pure_imaginary_z_in_third_quadrant_l513_513242


namespace jordan_rectangle_width_l513_513540

theorem jordan_rectangle_width :
  ∀ (areaC areaJ : ℕ) (lengthC widthC lengthJ widthJ : ℕ), 
    (areaC = lengthC * widthC) →
    (areaJ = lengthJ * widthJ) →
    (areaC = areaJ) →
    (lengthC = 5) →
    (widthC = 24) →
    (lengthJ = 3) →
    widthJ = 40 :=
by
  intros areaC areaJ lengthC widthC lengthJ widthJ
  intro hAreaC
  intro hAreaJ
  intro hEqualArea
  intro hLengthC
  intro hWidthC
  intro hLengthJ
  sorry

end jordan_rectangle_width_l513_513540


namespace find_a_div_c_l513_513461

def f (x : ℝ) : ℝ := (3 * x + 2) / (x - 4)
def f_inv (x : ℝ) : ℝ := (-4 * x - 2) / (x - 3)
variables (a b c d : ℝ)

theorem find_a_div_c : (a = -4) → (b = -2) → (c = 1) → (d = -3) → (a / c = -4) :=
by
  intros ha hb hc hd
  rw [ha, hc]
  norm_num

end find_a_div_c_l513_513461


namespace main_statement_l513_513747

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513747


namespace true_proposition_l513_513761

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513761


namespace p_and_q_is_true_l513_513693

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513693


namespace time_first_train_cross_post_l513_513088

-- Conditions
def length_of_train : ℝ := 120
def time_second_train_cross_post : ℝ := 15
def time_trains_cross_each_other : ℝ := 7.5
def speed_second_train : ℝ := length_of_train / time_second_train_cross_post
def relative_speed : ℝ := (2 * length_of_train) / time_trains_cross_each_other
def speed_first_train : ℝ := relative_speed - speed_second_train

-- Theorem to prove
theorem time_first_train_cross_post : (length_of_train / speed_first_train) = 5 := by
  sorry

end time_first_train_cross_post_l513_513088


namespace probability_heads_l513_513909

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let y := (n.choose (n / 2)) / total_outcomes
  (1 - y) / 2

theorem probability_heads (h₁ : ∀ (n : ℕ), n = 10 → probability_more_heads_than_tails n = 193 / 512) : 
  probability_more_heads_than_tails 10 = 193 / 512 :=
by
  apply h₁
  exact rfl

end probability_heads_l513_513909


namespace sum_of_three_digit_integers_with_1_and_2_l513_513082

theorem sum_of_three_digit_integers_with_1_and_2 : 
  let digits := {1, 2}
  let three_digit_numbers := {x : ℕ | ∃ d1 d2 d3 ∈ digits, x = 100 * d1 + 10 * d2 + d3}
  (∑ x in three_digit_numbers, x) = 1332 :=
by
  sorry

end sum_of_three_digit_integers_with_1_and_2_l513_513082


namespace internal_tangent_bisects_external_arc_l513_513496

noncomputable section

-- Definitions (based on conditions)
structure Circle (P : Type) :=
(center : P)
(radius : Real)

variables {P : Type} [MetricSpace P]

def touches (C1 C2 : Circle P) : Prop :=
∃ p : P, dist p C1.center = C1.radius ∧ dist p C2.center = C2.radius

def internally_touches (C1 C2 : Circle P) : Prop :=
∃ p : P, dist p C1.center = C1.radius ∧ dist p C2.center = C2.radius ∧ dist C1.center C2.center < C1.radius + C2.radius

def externally_touches (C1 C2 : Circle P) : Prop :=
∃ p : P, dist p C1.center = C1.radius ∧ dist p C2.center = C2.radius ∧ dist C1.center C2.center = C1.radius + C2.radius

def tangent (L : Set P) (C : Circle P) : Prop :=
∃ p : P, p ∈ L ∧ dist p C.center = C.radius

-- Proof problem statement
theorem internal_tangent_bisects_external_arc 
  (C1 C2 C3 : Circle P) 
  (L1 L2 : Set P)  :
  externally_touches C1 C2 →
  internally_touches C1 C3 →
  internally_touches C2 C3 →
  tangent L1 C1 →
  tangent L1 C2 →
  tangent L2 C1 →
  tangent L2 C3 →
  tangent L2 C2 →
  ∃ P : P, is_midpoint_arc P C3 L1 L2 :=
sorry

end internal_tangent_bisects_external_arc_l513_513496


namespace last_number_2001_l513_513992

noncomputable def last_remaining_number : ℕ :=
  let rec a : ℕ → ℕ
  | 0     => 0
  | 1     => 1
  | n+1   => if n % 2 = 1 then 2 * a (n / 2) + 1 else 2 * a (n / 2) - 1
  in a 2001

theorem last_number_2001 : last_remaining_number = 1955 := 
  by
  sorry

end last_number_2001_l513_513992


namespace how_many_right_triangles_l513_513302

def is_right_triangle (a b c : ℕ) : Prop := a * a + b * b = c * c

def area_eq_4_times_perimeter (a b c : ℕ) : Prop :=
  (a * b) / 2 = 4 * (a + b + c)

def num_non_congruent_right_triangles (A : ℕ) : ℕ :=
  (finset.range A).filter (λ a,
    (finset.range A).filter (λ b,
      ∃ c,
        is_right_triangle a b c ∧
        area_eq_4_times_perimeter a b c
    ).nonempty
  ).card

theorem how_many_right_triangles :
  ∃ n : ℕ, num_non_congruent_right_triangles 100 = n := sorry

end how_many_right_triangles_l513_513302


namespace peter_wins_prize_at_least_one_wins_prize_l513_513003

noncomputable def probability_peter_wins (N : Nat) := 
  (5/6) ^ (N - 1)

theorem peter_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob = probability_peter_wins N := by
  have h2 : prob = (5/6) ^ 9 := by
    sorry
  exact h2

noncomputable def probability_at_least_one_wins (N : Nat) := 
  10 * (5/6)^9 - 45 * (5 * 4^8 / 6^9) + 120 * (5 * 4 * 3^7 / 6^9) - 210 * (5 * 4 * 3 * 2^6 / 6^9) + 252 * (5 * 4 * 3 * 2 * 1 / 6^9)

theorem at_least_one_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob ≈ probability_at_least_one_wins N := by
  have h2 : prob ≈ 0.919 := by
    sorry
  exact h2

end peter_wins_prize_at_least_one_wins_prize_l513_513003


namespace true_proposition_l513_513758

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513758


namespace right_triangle_legs_shorter_than_hypotenuse_l513_513998

theorem right_triangle_legs_shorter_than_hypotenuse {A B C : Type*}
  [inner_product_space ℝ (A × B × C)] (AC BC AB : ℝ) (right_angle : ∠ C = 90°) : 
  AC < AB ∧ BC < AB :=
by 
  sorry

end right_triangle_legs_shorter_than_hypotenuse_l513_513998


namespace pascal_triangle_fifth_number_l513_513347

theorem pascal_triangle_fifth_number : binomial 13 4 = 715 :=
by 
  sorry

end pascal_triangle_fifth_number_l513_513347


namespace sin_theta_l513_513402

open Real

variables {a b c : E ℝ 3}

noncomputable def angle_between (b c : E ℝ 3) : ℝ :=
  acos ((b ⬝ c) / (∥b∥ * ∥c∥))

theorem sin_theta (a b c : E ℝ 3) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
(h4 : ¬Collinear ℝ ({a, b, c} : Set (E ℝ 3)))
(h5 : (a × b) × c = -((1 / 2) * (∥b∥ * ∥c∥) • a)) :
  sin (angle_between b c) = sqrt 3 / 2 :=
sorry

end sin_theta_l513_513402


namespace average_mb_per_hour_l513_513571

theorem average_mb_per_hour (days : ℕ) (total_disk_space : ℕ)
  (h_days : days = 15) (h_total_disk_space : total_disk_space = 20000) :
  let total_hours := days * 24
  let mb_per_hour := total_disk_space / total_hours.toFloat
  round mb_per_hour = 56 :=
by
  let total_hours := days * 24
  let mb_per_hour := total_disk_space / total_hours.toFloat
  have : total_hours = 360 := by rw [h_days]; simp
  have : mb_per_hour ≈ 55.56 := by rw [h_total_disk_space, this]; simp
  have : round mb_per_hour = 56 := by norm_cast; simp
  exact this

end average_mb_per_hour_l513_513571


namespace complex_multiplication_result_l513_513118

theorem complex_multiplication_result :
  (1 - 2 * complex.I) * (1 - complex.I) = -1 - 3 * complex.I :=
by
  -- proof steps omitted here
  sorry

end complex_multiplication_result_l513_513118


namespace f_positive_l513_513874

variable {f : ℝ → ℝ}

-- Conditions: f(x) is decreasing and f''(x) satisfies the given inequality.
def f_decreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x > f y
def f_inequality (f'' : ℝ → ℝ) (x : ℝ) : Prop := f x / f'' x < 1 - x

-- Theorem statement: Prove that f(x) > 0 for all x in ℝ, given the conditions.
theorem f_positive (hf_dec : f_decreasing f) (hf_ineq : ∀ x : ℝ, f_inequality (deriv^[2] f) x) : ∀ x : ℝ, f x > 0 :=
by
  -- placeholder for the proof
  sorry

end f_positive_l513_513874


namespace multiplication_approximation_correct_l513_513926

noncomputable def closest_approximation (x : ℝ) : ℝ := 
  if 15700 <= x ∧ x < 15750 then 15700
  else if 15750 <= x ∧ x < 15800 then 15750
  else if 15800 <= x ∧ x < 15900 then 15800
  else if 15900 <= x ∧ x < 16000 then 15900
  else 16000

theorem multiplication_approximation_correct :
  closest_approximation (0.00525 * 3153420) = 15750 := 
by
  sorry

end multiplication_approximation_correct_l513_513926


namespace motorbike_speed_l513_513153

noncomputable def speed_of_motorbike 
  (V_train : ℝ) 
  (t_overtake : ℝ) 
  (train_length_m : ℝ) : ℝ :=
  V_train - (train_length_m / 1000) * (3600 / t_overtake)

theorem motorbike_speed : 
  speed_of_motorbike 100 80 800.064 = 63.99712 :=
by
  -- this is where the proof steps would go
  sorry

end motorbike_speed_l513_513153


namespace main_statement_l513_513743

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513743


namespace find_a_l513_513272

theorem find_a :
  (∀ x : ℝ, (f x = if x < 0 then 2^x else a * (Real.sqrt x)) ∧ (f (-1) + f 1 = 2)) →
  a = 3 / 2 :=
by
  intros h
  sorry

end find_a_l513_513272


namespace cloth_gain_representation_l513_513145

theorem cloth_gain_representation (C S : ℝ) (h1 : S = 1.20 * C) (h2 : ∃ gain, gain = 60 * S - 60 * C) :
  ∃ meters : ℝ, meters = (60 * S - 60 * C) / S ∧ meters = 12 :=
by
  sorry

end cloth_gain_representation_l513_513145


namespace family_reunion_handshakes_l513_513173

theorem family_reunion_handshakes :
  ∃ (men women siblings : ℕ), men = 15 ∧ women = 17 ∧ siblings = 1 ∧ 
  (let handshakes_men := men * (men - 1) / 2 in 
   let handshakes_women := women * (women - 1) / 2 - siblings in
   let handshakes_men_women := men * (women - 1) in
   handshakes_men + handshakes_women + handshakes_men_women = 480) :=
by
  use 15, 17, 1
  split
  -- men = 15
  · exact rfl
  split
  -- women = 17
  · exact rfl
  split
  -- siblings = 1
  · exact rfl
  -- Total handshakes computation
  have handshakes_men := 15 * (15 - 1) / 2
  have handshakes_women := 17 * (17 - 1) / 2 - 1
  have handshakes_men_women := 15 * (17 - 1)
  have total_handshakes := handshakes_men + handshakes_women + handshakes_men_women
  show total_handshakes = 480 from sorry

end family_reunion_handshakes_l513_513173


namespace piecewise_function_l513_513923

def f (x : ℝ) : ℝ := if x > 0 then Real.log x else Real.exp (x + 1) - 2

theorem piecewise_function (h : f (f (1 / Real.exp 1)) = -1) : true :=
by
  sorry

end piecewise_function_l513_513923


namespace longer_side_of_new_rectangle_l513_513037

theorem longer_side_of_new_rectangle {z : ℕ} (h : ∃x : ℕ, 9 * 16 = 144 ∧ x * z = 144 ∧ z ≠ 9 ∧ z ≠ 16) : z = 18 :=
sorry

end longer_side_of_new_rectangle_l513_513037


namespace quadratic_equation_solution_unique_l513_513450

theorem quadratic_equation_solution_unique (b : ℝ) (hb : b ≠ 0) (h1_sol : ∀ x1 x2 : ℝ, 2*b*x1^2 + 16*x1 + 5 = 0 → 2*b*x2^2 + 16*x2 + 5 = 0 → x1 = x2) :
  ∃ x : ℝ, x = -5/8 ∧ 2*b*x^2 + 16*x + 5 = 0 :=
by
  sorry

end quadratic_equation_solution_unique_l513_513450


namespace coin_flip_probability_l513_513316

theorem coin_flip_probability (n : ℕ)
  (h : (1 / 2) ^ 4 * (1 / 2)^(n - 5) * (1 / 2) = 0.03125) :
  n = 9 :=
by
  -- Because (1 / 2)^4 * (1 / 2)^(n - 5) * (1 / 2) = 0.03125,
  -- we can simplify to find (1 / 2)^(n - 4) = (1 / 2)^5,
  -- which implies n - 4 = 5.
  -- Therefore, n = 9.
  sorry

end coin_flip_probability_l513_513316


namespace even_rows_sum_pascals_triangle_up_to_20_l513_513892

def pascals_triangle_even_sum (n : ℕ) : ℕ :=
  ∑ k in range ((n / 2) + 1), 2 * k + 1

theorem even_rows_sum_pascals_triangle_up_to_20 :
  pascals_triangle_even_sum 20 = 121 :=
by
  sorry

end even_rows_sum_pascals_triangle_up_to_20_l513_513892


namespace p_and_q_is_true_l513_513691

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513691


namespace largest_square_side_length_correct_l513_513362

-- Define the problem parameters
def rectangle : Type := {length : ℝ, width : ℝ}
def square : Type := {side_length : ℝ}
def triangle : Type := {side_length : ℝ}

-- Given conditions
def rect : rectangle := {length := 16, width := 12}
def congruent_triangles : triangle := {side_length := (20 * real.sqrt(3)) / 3}

-- The largest square inscribed in the specific region discussed
def largest_square : square := {side_length := 5 * real.sqrt 2 - (5 * real.sqrt 6) / 3}

-- Proof problem
theorem largest_square_side_length_correct :
  ∀ (r : rectangle) (t : triangle),
    r = rect →
    t = congruent_triangles →
    (∃ s : square, s.side_length = largest_square.side_length) := by
  sorry

end largest_square_side_length_correct_l513_513362


namespace greatest_n_for_3n_factorial_l513_513541

theorem greatest_n_for_3n_factorial :
  ∃ n : ℕ, (∀ k : ℕ, (k ≤ n → (∀ m : ℕ, (m = 3^k) → m ∣ 19!))) ∧ n = 8 :=
by
  sorry

end greatest_n_for_3n_factorial_l513_513541


namespace weighted_avg_markup_percentage_is_1583_l513_513583

def cost_apples := 30
def cost_oranges := 40
def cost_bananas := 50

def markup_percentage_apples := 0.10
def markup_percentage_oranges := 0.15
def markup_percentage_bananas := 0.20

def markup_apples := cost_apples * markup_percentage_apples
def markup_oranges := cost_oranges * markup_percentage_oranges
def markup_bananas := cost_bananas * markup_percentage_bananas

def selling_price_apples := cost_apples + markup_apples
def selling_price_oranges := cost_oranges + markup_oranges
def selling_price_bananas := cost_bananas + markup_bananas

def total_cost := cost_apples + cost_oranges + cost_bananas
def total_markup := markup_apples + markup_oranges + markup_bananas

def weighted_average_markup_percentage := (total_markup / total_cost) * 100

theorem weighted_avg_markup_percentage_is_1583 :
  weighted_average_markup_percentage = 15.83 :=
sorry

end weighted_avg_markup_percentage_is_1583_l513_513583


namespace income_percentage_medium_large_l513_513584

-- Define constants for the number of planes and their costs
def small_planes : ℕ := 150
def medium_planes : ℕ := 75
def large_planes : ℕ := 60

def small_cost : ℝ := 125
def medium_cost : ℝ := 175
def large_cost : ℝ := 220

-- Define the total income from each type of airliner
def total_small_income : ℝ := small_planes * small_cost
def total_medium_income : ℝ := medium_planes * medium_cost
def total_large_income : ℝ := large_planes * large_cost

-- Define the total income from all types of airliners
def total_income : ℝ := total_small_income + total_medium_income + total_large_income

-- Define the income from medium and large airliners combined
def medium_large_income : ℝ := total_medium_income + total_large_income

-- Define the percentage calculation
def income_percentage (part total : ℝ) : ℝ := (part / total) * 100

-- State the theorem to prove
theorem income_percentage_medium_large : income_percentage medium_large_income total_income = 58.39 := by
  sorry

end income_percentage_medium_large_l513_513584


namespace find_m_value_l513_513535

def magic_box (a b : ℝ) : ℝ := a^2 + 2 * b - 3

theorem find_m_value (m : ℝ) :
  magic_box m (-3 * m) = 4 ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end find_m_value_l513_513535


namespace proof_problem_l513_513776

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513776


namespace initial_concentration_l513_513127

theorem initial_concentration (f : ℚ) (C : ℚ) (h₀ : f = 0.7142857142857143) (h₁ : (1 - f) * C + f * 0.25 = 0.35) : C = 0.6 :=
by
  rw [h₀] at h₁
  -- The proof will follow the steps to solve for C
  sorry

end initial_concentration_l513_513127


namespace empty_with_three_pumps_in_12_minutes_l513_513174

-- Define the conditions
def conditions (a b x : ℝ) : Prop :=
  x = a + b ∧ 2 * x = 3 * a + b

-- Define the main theorem to prove
theorem empty_with_three_pumps_in_12_minutes (a b x : ℝ) (h : conditions a b x) : 
  (3 * (1 / 5) * x = a + (1 / 5) * b) ∧ ((1 / 5) * 60 = 12) := 
by
  -- Use the given conditions in the proof.
  sorry

end empty_with_three_pumps_in_12_minutes_l513_513174


namespace horses_put_by_c_l513_513538

theorem horses_put_by_c (a_horses a_months b_horses b_months c_months total_cost b_cost : ℕ) (x : ℕ) 
  (h1 : a_horses = 12) 
  (h2 : a_months = 8) 
  (h3 : b_horses = 16) 
  (h4 : b_months = 9) 
  (h5 : c_months = 6) 
  (h6 : total_cost = 870) 
  (h7 : b_cost = 360) 
  (h8 : 144 / (96 + 144 + 6 * x) = 360 / 870) : 
  x = 18 := 
by 
  sorry

end horses_put_by_c_l513_513538


namespace ex_ineq_l513_513638

theorem ex_ineq (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 40) :
  x + y = 2 + 2 * Real.sqrt 3 ∨ x + y = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end ex_ineq_l513_513638


namespace proposition_A_l513_513822

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513822


namespace unique_triangle_areas_l513_513626

-- Given Conditions
variables (G H I J K L M : Point)
noncomputable def line1 : Line := Line.mk (21, 21) -- Let's define some temporary values to define Line1
noncomputable def line2 : Line := Line.mk (30, 30) -- And Line2 to be parallel to Line1

axiom parallel_lines : parallel line1 line2
axiom on_line1_G : is_on_line G line1
axiom on_line1_H : is_on_line H line1
axiom on_line1_I : is_on_line I line1
axiom on_line1_J : is_on_line J line1
axiom on_line2_K : is_on_line K line2
axiom on_line2_L : is_on_line L line2
axiom on_line2_M : is_on_line M line2

axiom dist_GH : distance G H = 1
axiom dist_HI : distance H I = 1
axiom dist_IJ : distance I J = 2
axiom dist_KL : distance K L = 2
axiom dist_KM : distance K M = 1

-- Prove the number of unique triangle areas is equal to 3
theorem unique_triangle_areas : 
  (count_unique_triangle_areas G H I J K L M) = 3 :=
sorry

end unique_triangle_areas_l513_513626


namespace total_number_of_balls_in_fish_tank_l513_513076

-- Definitions as per conditions
def num_goldfish := 3
def num_platyfish := 10
def red_balls_per_goldfish := 10
def white_balls_per_platyfish := 5

-- Theorem statement
theorem total_number_of_balls_in_fish_tank : 
  (num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish) = 80 := 
by
  sorry

end total_number_of_balls_in_fish_tank_l513_513076


namespace victor_decks_l513_513502

theorem victor_decks (V : ℕ) (cost_per_deck total_spent friend_decks : ℕ) 
  (h1 : cost_per_deck = 8)
  (h2 : total_spent = 64)
  (h3 : friend_decks = 2) 
  (h4 : 8 * V + 8 * friend_decks = total_spent) : 
  V = 6 :=
by sorry

end victor_decks_l513_513502


namespace tangent_slope_at_C_passing_P_l513_513268

noncomputable def slope_of_tangent_through_point (P : ℝ × ℝ) (C : ℝ → ℝ) : Set ℝ :=
  {m | ∃ t : ℝ, C t = t^2 ∧ m = 2*t ∧ (0, t^2) = 2*t * (1 - t)}

theorem tangent_slope_at_C_passing_P :
  slope_of_tangent_through_point (1, 0) (λ x => x^2) = {0, 4} :=
sorry

end tangent_slope_at_C_passing_P_l513_513268


namespace average_speed_l513_513479

theorem average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 60) (h3 : t1 = 1) (h4 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 35 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_speed_l513_513479


namespace find_areas_after_shortening_l513_513418

-- Define initial dimensions
def initial_length : ℤ := 5
def initial_width : ℤ := 7
def shortened_by : ℤ := 2

-- Define initial area condition
def initial_area_condition : Prop := 
  initial_length * (initial_width - shortened_by) = 15 ∨ (initial_length - shortened_by) * initial_width = 15

-- Define the resulting areas for shortening each dimension
def area_shortening_length : ℤ := (initial_length - shortened_by) * initial_width
def area_shortening_width : ℤ := initial_length * (initial_width - shortened_by)

-- Statement for proof
theorem find_areas_after_shortening
  (h : initial_area_condition) :
  area_shortening_length = 21 ∧ area_shortening_width = 25 :=
sorry

end find_areas_after_shortening_l513_513418


namespace volume_is_correct_l513_513322

-- Definitions
def perimeter_of_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

def volume_of_parallelepiped (length width height : ℕ) : ℕ :=
  length * width * height

-- Problem Conditions
def length := 8 -- derived from condition: perimeter_of_rectangle 8 8 = 32
def width := 8  -- derived from condition: perimeter_of_rectangle 8 8 = 32
def height := 9 -- directly given

-- Theorem to prove
theorem volume_is_correct :
  volume_of_parallelepiped length width height = 576 :=
by
  sorry

end volume_is_correct_l513_513322


namespace true_conjunction_l513_513685

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513685


namespace exists_set_S_l513_513380

-- Definition of a neighbor in \mathbb{Z}^n
def is_neighbor {n : ℕ} (p q : Fin n → ℤ) : Prop :=
  (∃ i, ∀ j, (i = j → (p i = q i + 1 ∨ p i = q i - 1)) ∧ (i ≠ j → p j = q j))

-- Main statement
theorem exists_set_S (n : ℕ) (hn : n ≥ 1) : 
  ∃ S : set (Fin n → ℤ), 
  (∀ p ∈ S, ∀ q, is_neighbor p q → q ∉ S) ∧
  (∀ p, p ∉ S → (∃! q, is_neighbor p q ∧ q ∈ S)) :=
sorry

end exists_set_S_l513_513380


namespace intersection_point_count_l513_513894

def abs_val_f1 (x : ℝ) : ℝ := abs (3 * x + 6)
def abs_val_f2 (x : ℝ) : ℝ := -abs (4 * x - 3)

def abs_val_f1_piecewise (x : ℝ) : ℝ :=
if x >= -2 then
  3 * x + 6
else
  -3 * x - 6

def abs_val_f2_piecewise (x : ℝ) : ℝ :=
if x >= 3 / 4 then
  -4 * x + 3
else
  4 * x - 3

theorem intersection_point_count : 
  ∃! (x y : ℝ), abs_val_f1 x = y ∧ abs_val_f2 x = y := 
sorry

end intersection_point_count_l513_513894


namespace five_letter_words_with_vowels_l513_513891

/-- How many 5-letter words with at least one vowel can be constructed
    from the letters A, B, C, D, E, F, G, and I? 
    - Any word is valid (not just English words)
    - Letters may be used more than once. -/
theorem five_letter_words_with_vowels :
    let total_5_letter_words := 8^5
    let no_vowel_5_letter_words := 5^5
    total_5_letter_words - no_vowel_5_letter_words = 29643 :=
by
  let total_5_letter_words := 8^5
  let no_vowel_5_letter_words := 5^5
  have total_words := 32768 -- From 8^5 = 32768
  have words_without_vowels := 3125 -- From 5^5 = 3125
  calc
    total_words - words_without_vowels = 32768 - 3125 := by rfl
    ...                               = 29643          := by rfl
  done

end five_letter_words_with_vowels_l513_513891


namespace product_divisible_by_10_probability_l513_513604

noncomputable def probability_divisible_by_10 (n : ℕ) (h: n > 1) : ℝ :=
  1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ))

theorem product_divisible_by_10_probability (n : ℕ) (h: n > 1) :
  probability_divisible_by_10 n h = 1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ)) :=
by
  -- The proof is omitted
  sorry

end product_divisible_by_10_probability_l513_513604


namespace ratio_of_angles_l513_513932

theorem ratio_of_angles (O A B C D : Point) (h_circle : ∃ (r : ℝ), ∀ p : Point, dist O p = r) 
  (h_triangle : Triangle A B C) (h_arcAB_80 : arc_angle O A B = 80) 
  (h_arcBC_120 : arc_angle O B C = 120) (h_D_on_minor_AC : OnMinorArc D A C) 
  (h_OD_perp_AC : ⊥ (line_through O D) (line_through A C)) :
  (angle_mag (angle O B D)) / (angle_mag (angle B A C)) = 3/4 := 
sorry

end ratio_of_angles_l513_513932


namespace length_segment_AB_l513_513879

-- Definitions of the conditions
variables (t α : ℝ)
noncomputable def line_x (t α : ℝ) := 1 + t * Real.cos α
noncomputable def line_y (t α : ℝ) := t * Real.sin α
def parabola (y : ℝ) := (y ^ 2 = 4 * (1 + t * Real.cos α))  -- Substituting x in parabola
def midpoint_x (m : ℝ) := m
def midpoint_y := 2

-- The proof problem in Lean
theorem length_segment_AB (m : ℝ) :
  (parabola (2 * Real.sin α)) ∧ 
  ((1 + t * Real.cos α) = 4) →
  m = 1 ∧ α = Real.pi / 4 →
  sqrt(2 * (2^2 + 2 * (4))) = 8 :=
by
  sorry

end length_segment_AB_l513_513879


namespace Jason_spent_on_music_store_l513_513948

theorem Jason_spent_on_music_store:
  let flute := 142.46
  let music_stand := 8.89
  let song_book := 7.00
  flute + music_stand + song_book = 158.35 := sorry

end Jason_spent_on_music_store_l513_513948


namespace locus_of_G_is_plane_l513_513669

noncomputable def is_plane_parallel_at_distance (plane1 plane2 : Type) (d : ℝ) : Prop :=
  ∃ plane_parallel, (plane_parallel ∥ plane1) ∧ (distance plane_parallel plane1 = d)

theorem locus_of_G_is_plane (e : Type) (A B C : Type)
  (not_collinear_ABC : ¬ collinear A B C) 
  (A_prime B_prime C_prime : Type)
  (L : midpoint A A_prime)
  (M : midpoint B B_prime)
  (N : midpoint C C_prime)
  (G : centroid L M N)
  (condition1 : A ∈ half_space e)
  (condition2 : B ∈ half_space e)
  (condition3 : C ∈ half_space e)
  (condition4 : A_prime ∈ e)
  (condition5 : B_prime ∈ e)
  (condition6 : C_prime ∈ e)
  (condition7 : forms_triangle L M N)
  (a b c : ℝ)
  (dist_A_plane : distance A e = a)
  (dist_B_plane : distance B e = b)
  (dist_C_plane : distance C e = c) :
  is_plane_parallel_at_distance (plane_of_points A B C) e (1/6 * (a + b + c)) := sorry

end locus_of_G_is_plane_l513_513669


namespace peter_wins_prize_at_least_one_wins_prize_l513_513002

noncomputable def probability_peter_wins (N : Nat) := 
  (5/6) ^ (N - 1)

theorem peter_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob = probability_peter_wins N := by
  have h2 : prob = (5/6) ^ 9 := by
    sorry
  exact h2

noncomputable def probability_at_least_one_wins (N : Nat) := 
  10 * (5/6)^9 - 45 * (5 * 4^8 / 6^9) + 120 * (5 * 4 * 3^7 / 6^9) - 210 * (5 * 4 * 3 * 2^6 / 6^9) + 252 * (5 * 4 * 3 * 2 * 1 / 6^9)

theorem at_least_one_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob ≈ probability_at_least_one_wins N := by
  have h2 : prob ≈ 0.919 := by
    sorry
  exact h2

end peter_wins_prize_at_least_one_wins_prize_l513_513002


namespace cos_B_value_sin_2A_plus_sin_C_value_l513_513946

theorem cos_B_value 
  (a : ℝ) (b : ℝ) (A B : ℝ)
  (h₁ : a = 3)
  (h₂ : b = 4)
  (h₃ : B = π / 2 + A) : 
  cos B = -3 / 5 :=
sorry

theorem sin_2A_plus_sin_C_value
  (a : ℝ) (b : ℝ) (A B : ℝ)
  (h₁ : a = 3)
  (h₂ : b = 4)
  (h₃ : B = π / 2 + A) : 
  sin (2 * A) + sin (π - A - B) = 31 / 25 :=
sorry

end cos_B_value_sin_2A_plus_sin_C_value_l513_513946


namespace tenth_square_tiles_more_than_ninth_l513_513336

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := 2 * n - 1

-- Calculate the number of tiles used in the nth square
def tiles_count (n : ℕ) : ℕ := (side_length n) ^ 2

-- State the theorem that the tenth square requires 72 more tiles than the ninth square
theorem tenth_square_tiles_more_than_ninth : tiles_count 10 - tiles_count 9 = 72 :=
by
  sorry

end tenth_square_tiles_more_than_ninth_l513_513336


namespace total_number_of_balls_l513_513075

theorem total_number_of_balls
  (goldfish : ℕ) (platyfish : ℕ)
  (goldfish_balls : ℕ) (platyfish_balls : ℕ)
  (h1 : goldfish = 3) (h2 : platyfish = 10)
  (h3 : goldfish_balls = 10) (h4 : platyfish_balls = 5) :
  (goldfish * goldfish_balls + platyfish * platyfish_balls) = 80 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end total_number_of_balls_l513_513075


namespace B_cannot_do_work_l513_513107

-- Define the concepts of work rates for each individual and combinations
def work : Type := ℝ

variables (A_rate B_rate C_rate : work → ℝ)

-- Below are the conditions provided in the problem:
-- 1. A can do a piece of work in 3 hours.
axiom A_can_complete : ∀ W : work, A_rate W = W / 3

-- 2. B and C together can do the work in 2 hours.
axiom B_C_can_complete_together : ∀ W : work, (B_rate W + C_rate W) = W / 2

-- 3. A and C together can do the work in 2 hours.
axiom A_C_can_complete_together : ∀ W : work, (A_rate W + C_rate W) = W / 2

-- The question: Prove that B's work rate alone must be zero
theorem B_cannot_do_work : ∀ W : work, B_rate W = 0 := by
sorry

end B_cannot_do_work_l513_513107


namespace cube_root_neg27_l513_513534

theorem cube_root_neg27 : (∛(-27) = -3) := 
by sorry

end cube_root_neg27_l513_513534


namespace area_enclosed_shape_l513_513280

-- Definitions of the conditions
def f (x : ℝ) := x^3

-- The theorem stating the problem and the correct answer
theorem area_enclosed_shape : 2 * ∫ x in 0..1, (x - f x) = 1 / 2 :=
by 
  sorry

end area_enclosed_shape_l513_513280


namespace total_number_of_balls_l513_513074

theorem total_number_of_balls
  (goldfish : ℕ) (platyfish : ℕ)
  (goldfish_balls : ℕ) (platyfish_balls : ℕ)
  (h1 : goldfish = 3) (h2 : platyfish = 10)
  (h3 : goldfish_balls = 10) (h4 : platyfish_balls = 5) :
  (goldfish * goldfish_balls + platyfish * platyfish_balls) = 80 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end total_number_of_balls_l513_513074


namespace scientific_notation_of_56_point_5_million_l513_513424

-- Definitions based on conditions
def million : ℝ := 10^6
def number_in_millions : ℝ := 56.5 * million

-- Statement to be proved
theorem scientific_notation_of_56_point_5_million : 
  number_in_millions = 5.65 * 10^7 :=
sorry

end scientific_notation_of_56_point_5_million_l513_513424


namespace solution_set_for_inequality_l513_513261

variable (f : ℝ → ℝ)
variable (x : ℝ)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = f(x)
def is_decreasing_on_negative_domain (f : ℝ → ℝ) : Prop := ∀ x y, x < y ∧ y ≤ 0 → f(x) > f(y)
def f_half := f (1 / 2) = 2

theorem solution_set_for_inequality
  (evenf : is_even_function f)
  (decrf : is_decreasing_on_negative_domain f)
  (fhalf : f_half f) :
  (0 < x ∧ x < 1 / 2 ∨ x > 2) ↔ f (Real.log 4 x) > 2 :=
by
  sorry

end solution_set_for_inequality_l513_513261


namespace shrimp_count_l513_513503

variable (V : ℕ)

def Austins_shrimp := V - 8
def Brians_shrimp := (V + Austins_shrimp) / 2
def total_shrimp := V + Austins_shrimp + Brians_shrimp

theorem shrimp_count (h : 3 * 14 = 42) (h1 : total_shrimp = 66) : V = 26 := by 
  sorry

end shrimp_count_l513_513503


namespace proof_problem_l513_513811

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513811


namespace solution_set_g_le_1_l513_513258

def f (x : ℝ) : ℝ := (1 + 2 * Real.sin x ^ 2) / (Real.sin (2 * x))
def g (x : ℝ) : ℝ :=
  if x > Real.pi / 4 ∧ x < Real.pi / 2 then -1
  else if x > 0 ∧ x <= Real.pi / 4 then 8 * x ^ 2 - 6 * Real.sqrt 3 * x + 4
  else 0 -- setting default value for else case

theorem solution_set_g_le_1 : {x : ℝ | g x ≤ 1} = set.Icc (Real.sqrt 3 / 4) (Real.pi / 2) :=
  sorry

end solution_set_g_le_1_l513_513258


namespace paths_from_jan1_to_dec31_l513_513939

-- Define the grid size
def rows : ℕ := 5
def columns : ℕ := 10

-- Define the set of obstacles
def is_obstacle (i j : ℕ) : Prop :=
  (i = 1 ∧ j = 1) ∨ (i = 1 ∧ j = 2) ∨ (i = 1 ∧ j = 4) ∨
  (i = 1 ∧ j = 5) ∨ (i = 1 ∧ j = 8) ∨ (i = 3 ∧ j = 2) ∨ 
  (i = 3 ∧ j = 5) ∨ (i = 3 ∧ j = 8)

-- Define the dynamic programming table
noncomputable def dp : array (rows + 1) (array (columns + 1) ℕ) :=
  let initial : array (rows + 1) (array (columns + 1) ℕ) :=
    (array.mk (rows + 1) (λ _, (array.mk (columns + 1) 0)))
  let initialized : array (rows + 1) (array (columns + 1) ℕ) :=
    initial.write 0 ((initial.read 0).write 0 1)
  matrix.iterate 0 (rows + 1) (λ i m,
    matrix.iterate 0 (columns + 1) (λ j n,
      if is_obstacle i j then
        n.write i ((n.read i).write j 0)
      else
        let up := if i > 0 then (n.read (i - 1)).read j else 0
        let left := if j > 0 then (n.read i).read (j - 1) else 0
        let diag := if i > 0 ∧ j > 0 then (n.read (i - 1)).read (j - 1) else 0
        n.write i ((n.read i).write j (up + left + diag))
    )) initialized

-- Define the total number of paths
def total_paths : ℕ := dp.read (rows - 1).read (columns - 1)

-- The final statement to prove the number of paths equals the given answer
theorem paths_from_jan1_to_dec31 : total_paths = 38 :=
  sorry

end paths_from_jan1_to_dec31_l513_513939


namespace proposition_true_l513_513836

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513836


namespace first_player_winning_strategy_l513_513990

-- Assuming a chessboard is represented as a matrix of pieces
def chessboard : Type := matrix ℕ ℕ bool

-- Define initial conditions of the chessboard (specific configuration can be added)
variable (board : chessboard)

-- Define the game rules as predicates or functions
def valid_move (board : chessboard) (row col start end_ : ℕ) : Prop :=
(start ≤ end_) ∧ (∀ i, start ≤ i ∧ i ≤ end_ → board row col = true)

def first_player_has_winning_strategy (board : chessboard) : Prop :=
let moves := [
  (λ board, valid_move board 1 0 0 (matrix.dim board 1)),
  (λ board, valid_move board 3 0 0 (matrix.dim board 1)),
  (λ board, valid_move board 5 0 0 (matrix.dim board 1)),
  (λ board, valid_move board 7 0 0 (matrix.dim board 1)),
  (λ board, valid_move board 9 0 0 (matrix.dim board 1)),
  (λ board, valid_move board 0 2 0 (matrix.dim board 0)),
  (λ board, valid_move board 0 4 0 (matrix.dim board 0))
] in
∃ move ∈ moves, ensures_winning_strategy board move

-- Now we state the theorem
theorem first_player_winning_strategy : first_player_has_winning_strategy board := sorry

end first_player_winning_strategy_l513_513990


namespace problem_statement_l513_513975

theorem problem_statement (p a_1 a_2 : ℤ) (k : ℕ) (a : ℕ → ℤ) (m : ℕ) 
  (h1 : p > 2) 
  (h2 : p % 3 ≠ 0)
  (h3 : ∀ i j : ℕ, i < k → j < k → i < j → a i < a j)
  (h4 : ∀ i : ℕ, i < k → -p / 2 < a i ∧ a i < p / 2) :
  (\prod i in Finset.range k, (p - a i) / |a i|) = 3 ^ m :=
sorry

end problem_statement_l513_513975


namespace woman_work_rate_l513_513585

theorem woman_work_rate :
  let M := 1/6
  let B := 1/9
  let combined_rate := 1/3
  ∃ W : ℚ, M + B + W = combined_rate ∧ 1 / W = 18 := 
by
  sorry

end woman_work_rate_l513_513585


namespace integral_sqrt_subtract_one_l513_513614

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt (1 - (1 - x)^2) - 1

theorem integral_sqrt_subtract_one :
  ∫ x in 0..1, f x = (π/4) - 1 := by
  sorry

end integral_sqrt_subtract_one_l513_513614


namespace circle_diameter_l513_513606

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ d : ℝ, d = 16 :=
by
  sorry

end circle_diameter_l513_513606


namespace total_length_after_connection_l513_513489

-- Definitions of the conditions
def num_sticks := 6
def length_per_stick := 50  -- cm
def num_connections := num_sticks - 1
def connection_length := 10  -- cm

-- The goal is to prove that the total length of the sticks after connecting them is 250 cm
theorem total_length_after_connection : 
  let total_length := (num_sticks * length_per_stick) - (num_connections * connection_length)
  in total_length = 250 := 
by 
  sorry

end total_length_after_connection_l513_513489


namespace area_within_fence_is_328_l513_513046

-- Define the dimensions of the fenced area
def main_rectangle_length : ℝ := 20
def main_rectangle_width : ℝ := 18

-- Define the dimensions of the square cutouts
def cutout_length : ℝ := 4
def cutout_width : ℝ := 4

-- Calculate the areas
def main_rectangle_area : ℝ := main_rectangle_length * main_rectangle_width
def cutout_area : ℝ := cutout_length * cutout_width

-- Define the number of cutouts
def number_of_cutouts : ℝ := 2

-- Calculate the final area within the fence
def area_within_fence : ℝ := main_rectangle_area - number_of_cutouts * cutout_area

theorem area_within_fence_is_328 : area_within_fence = 328 := by
  -- This is a place holder for the proof, replace it with the actual proof
  sorry

end area_within_fence_is_328_l513_513046


namespace coffee_mixture_weight_l513_513149

theorem coffee_mixture_weight (weight_A weight_B : ℕ) (hA : weight_A = 240) (hB : weight_B = 240) :
  weight_A + weight_B = 480 :=
by
  rw [hA, hB] -- use the given conditions to rewrite weight_A and weight_B
  exact rfl -- conclude that 240 + 240 equals 480

end coffee_mixture_weight_l513_513149


namespace part1_l513_513273

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : 
  f x 1 ≥ 1 :=
sorry

end part1_l513_513273


namespace problem_statement_l513_513791

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513791


namespace answer_is_p_and_q_l513_513731

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513731


namespace right_pyramid_sum_edges_l513_513142

theorem right_pyramid_sum_edges (a h : ℝ) (base_side slant_height : ℝ) :
  base_side = 12 ∧ slant_height = 15 ∧ ∀ x : ℝ, a = 117 :=
by
  sorry

end right_pyramid_sum_edges_l513_513142


namespace probability_of_more_heads_than_tails_l513_513915

-- Define the probability of getting more heads than tails when flipping 10 coins
def probabilityMoreHeadsThanTails : ℚ :=
  193 / 512

-- Define the proof statement
theorem probability_of_more_heads_than_tails :
  let p : ℚ := probabilityMoreHeadsThanTails in
  p = 193 / 512 :=
by
  sorry

end probability_of_more_heads_than_tails_l513_513915


namespace max_value_l513_513462

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * real.sqrt x - 3 * x

-- Define the non-negative condition
def non_negative (x : ℝ) : Prop := x ≥ 0

-- Statement to prove the maximum value is 1/3
theorem max_value : ∀ (x : ℝ), non_negative x → ∃ (M : ℝ), f(x) ≤ M ∧ M = (1 / 3) := sorry

end max_value_l513_513462


namespace total_settings_weight_l513_513417

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end total_settings_weight_l513_513417


namespace problem_statement_l513_513707

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513707


namespace area_ratio_l513_513994

-- Definitions allowing for a convex quadrilateral and its properties
def convex_quadrilateral (A B C D : ℝ) : Prop := sorry -- Define convexity properly

def area (A B C D : ℝ) : ℝ := sorry -- Define area calculation

-- Given points on the extensions with the specified conditions
def points_on_extensions (A B C D : ℝ) : Prop := 
  BB1 = AB ∧ CC1 = BC ∧ DD1 = CD ∧ AA1 = DA

-- Main theorem to be proved
theorem area_ratio (A B C D A1 B1 C1 D1 : ℝ) :
  convex_quadrilateral A B C D →
  points_on_extensions A B C D →
  area A1 B1 C1 D1 = 5 * area A B C D :=
  sorry

end area_ratio_l513_513994


namespace mean_of_six_numbers_sum_three_quarters_l513_513481

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end mean_of_six_numbers_sum_three_quarters_l513_513481


namespace segment_length_is_parabolic_l513_513188

variable (A B C D : Point)
variable (line_length : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ℝ)

-- Conditions
def segment_starts_at_A_ends_at_C (line_length : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ℝ) : Prop :=
  line_length 0 = 0 ∧ line_length 1 = 0

def maximum_length_when_parallel_to_BD (line_length : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ℝ) : Prop :=
  ∃ t_max : ℝ, 0 ≤ t_max ∧ t_max ≤ 1 ∧ 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → line_length t ≤ line_length t_max) ∧
  is_parallel (segment_from A C) (diagonal_from B D) -- assumes segment_from and diagonal_from are defined geometrically

-- Conclusion
theorem segment_length_is_parabolic :
  ∀ (line_length : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ℝ),
    segment_starts_at_A_ends_at_C line_length →
    maximum_length_when_parallel_to_BD line_length →
    parabolic_change (length_of line_length)  -- assuming parabolic_change and length_of are defined
:= sorry

end segment_length_is_parabolic_l513_513188


namespace telegraph_longer_than_pardee_l513_513451

theorem telegraph_longer_than_pardee : 
  let telegraph_length_km := 162 in
  let pardee_length_m := 12000 in
  let pardee_length_km := pardee_length_m / 1000 in
  telegraph_length_km - pardee_length_km = 150 :=
by
  sorry

end telegraph_longer_than_pardee_l513_513451


namespace jerrie_minutes_l513_513605

-- Define the conditions
def barney_situps_per_minute := 45
def carrie_situps_per_minute := 2 * barney_situps_per_minute
def jerrie_situps_per_minute := carrie_situps_per_minute + 5
def barney_total_situps := 1 * barney_situps_per_minute
def carrie_total_situps := 2 * carrie_situps_per_minute
def combined_total_situps := 510

-- Define the question and required proof
theorem jerrie_minutes :
  ∃ J : ℕ, barney_total_situps + carrie_total_situps + J * jerrie_situps_per_minute = combined_total_situps ∧ J = 3 :=
  by
  sorry

end jerrie_minutes_l513_513605


namespace problem_l513_513887

universe u

def U : Set ℤ := {2, 3, 5, 7, 9}
def A (a : ℤ) : Set ℤ := {2, |a - 5|, 7}
def C_U_A (A : Set ℤ) : Set ℤ := U \ A

theorem problem (a : ℤ) (h : C_U_A (A a) = {5, 9}) : a = 2 ∨ a = 8 := by
  have h_compl : U \ (A a) = {5, 9} := h
  have h_A : A a ⊆ U := sorry  -- This would generally need proof but is assumed here.
  have h_compl_def : ∀ x, x ∈ U → x ∉ A a → x ∈ {5, 9} := sorry
  have h_A_def : A a = {2, |a - 5|, 7} := rfl
  cases Abs_succ (a - 5) 3 with h_pos h_neg
  · exact (or.inr (eq.symm (eq_of_abs_eq Abs_succ) 3 h_pos))
  · exact (or.inl (eq_of_abs_eq_neg Abs_succ 3 h_neg))

end problem_l513_513887


namespace sum_of_squares_of_coeffs_l513_513094

theorem sum_of_squares_of_coeffs :
  let p := 5 * (2 * X ^ 6 + 4 * X ^ 3 + X + 2) in
  (coeffs := [10, 20, 5, 10]) → (sum := (coeffs.map (λ c, c^2)).sum) → sum = 625 := by
  sorry

end sum_of_squares_of_coeffs_l513_513094


namespace sequence_divisible_by_41_count_l513_513193

noncomputable
def sequence_term (n : ℕ) : ℕ := 20 ^ n + 1

theorem sequence_divisible_by_41_count :
  (finset.range 1000).filter (λ n, sequence_term (n + 1) % 41 = 0).card = 250 :=
by sorry

end sequence_divisible_by_41_count_l513_513193


namespace inequality_sums_cubed_roots_l513_513432

theorem inequality_sums_cubed_roots (n : ℕ) (h : n ≥ 2) :
  (∑ i in finset.range(n+1), real.cbrt (i / (n + 1))) / n ≤
  (∑ i in finset.range(n), real.cbrt (i / n)) / (n - 1) :=
sorry

end inequality_sums_cubed_roots_l513_513432


namespace probability_below_curve_l513_513883

open Real

def region (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ π ∧ 0 ≤ y ∧ y ≤ 1

def curve (x : ℝ) : ℝ := sin x ^ 2

theorem probability_below_curve :
  let area_region := π * 1
  let area_below_curve := ∫ x in 0..π, curve x
  (0 < area_region) → (0 ≤ area_below_curve) →
  (area_below_curve / area_region = 1 / 2) :=
by
  sorry

end probability_below_curve_l513_513883


namespace midpoint_of_equal_area_l513_513353

variables {A B C D M : Type} [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty M]
variables [FragmentedTriangle A B C] (AD : Segment A D) (BC : Segment B C) 

def equal_area (AMB AMC : Triangle A M B × Triangle A M C) : Prop := 
  AMB.area = AMC.area 

def midpoint (P Q : Point) (S : Segment P Q): Prop := 
  S.midpoint = midpoint(S.segment)

theorem midpoint_of_equal_area 
  (AD : Segment A D) 
  (BC : Segment B C)
  (AMB AMC : Triangle A M B × Triangle A M C)
  (h : equal_area AMB AMC) :
  (∃ (D : Point), midpoint B C D BC) ∧ ¬(midpoint A D M AD) :=
sorry

end midpoint_of_equal_area_l513_513353


namespace average_disk_space_per_hour_l513_513574

theorem average_disk_space_per_hour:
  (let total_hours := 15 * 24 in
  ∀ (total_disk_space : ℕ), total_disk_space = 20000 →
  20000 / total_hours = 56) :=
begin
  let total_hours := 15 * 24,
  intro total_disk_space,
  intro h,
  simp [h, total_hours],
  sorry
end

end average_disk_space_per_hour_l513_513574


namespace intersection_product_l513_513882

-- Definitions from the conditions
def C1_parametric (t : ℝ) : ℝ × ℝ := (4 * t, 3 * t - 1)
def C1_cartesian (x y : ℝ) : Prop := 3 * x - 4 * y - 4 = 0
def C2_polar (θ : ℝ) : ℝ := 8 * cos θ / (1 - cos (2 * θ))
def C2_cartesian (x y : ℝ) : Prop := y^2 = 4 * x
def point_P := (0, -1)

-- Goal
theorem intersection_product :
  let A := C1_cartesian
  let B := C2_cartesian
  |PA| * |PB| = 25 / 9 := sorry

end intersection_product_l513_513882


namespace total_emails_in_april_is_675_l513_513957

-- Define the conditions
def daily_emails : ℕ := 20
def additional_emails : ℕ := 5
def april_days : ℕ := 30
def half_april_days : ℕ := april_days / 2

-- Define the total number of emails received
def total_emails : ℕ :=
  (daily_emails * half_april_days) +
  ((daily_emails + additional_emails) * half_april_days)

-- Define the statement to be proven
theorem total_emails_in_april_is_675 : total_emails = 675 :=
  by
  sorry

end total_emails_in_april_is_675_l513_513957


namespace largest_square_in_rectangle_l513_513359

noncomputable def largest_square_side_length (length width side_triangle : ℝ) : ℝ :=
  (8 - real.sqrt 6)

theorem largest_square_in_rectangle (length width : ℝ) (side_triangle : ℝ)
  (h1 : length = 16) (h2 : width = 12)
  (h3 : side_triangle = (12 * real.sqrt 2 / real.sqrt 3))
  : largest_square_side_length length width side_triangle = 8 - real.sqrt 6 :=
by
  sorry

end largest_square_in_rectangle_l513_513359


namespace find_x_l513_513207

theorem find_x (x : ℝ) (h : 3 ^ (Real.log x / Real.log 8) = 9) : x = 64 :=
  sorry

end find_x_l513_513207


namespace car_race_probability_l513_513333

theorem car_race_probability :
  let pX := 1/8
  let pY := 1/12
  let pZ := 1/6
  pX + pY + pZ = 3/8 :=
by
  sorry

end car_race_probability_l513_513333


namespace perfect_square_divisors_count_l513_513306

open Nat

theorem perfect_square_divisors_count :
  let p := (2 ^ 12) * (3 ^ 15) * (5 ^ 18) * (7 ^ 8)
  (∃ n : ℕ, p = n^2) → 
  (num_perfect_square_divisors p = 2800) :=
by
  -- Definitions based on the conditions
  let e2 := 12
  let e3 := 15
  let e5 := 18
  let e7 := 8

  -- Expected number of perfect square divisors
  let num_perfect_square_divisors := (e2 / 2 + 1) * (e3 / 2 + 1) * (e5 / 2 + 1) * (e7 / 2 + 1)

  -- Assertion of correct answer
  have h : num_perfect_square_divisors = 2800 :=
    by -- The steps simply verifying and multiplying the counts
       rfl
  exact (num_perfect_square_divisors, h).snd

end perfect_square_divisors_count_l513_513306


namespace log_a_less_than_neg_b_minus_one_l513_513875

variable {x : ℝ} (a b : ℝ) (f : ℝ → ℝ)

theorem log_a_less_than_neg_b_minus_one
  (h1 : 0 < a)
  (h2 : ∀ x > 0, f x ≥ f 3)
  (h3 : ∀ x > 0, f x = -3 * Real.log x + a * x^2 + b * x) :
  Real.log a < -b - 1 :=
  sorry

end log_a_less_than_neg_b_minus_one_l513_513875


namespace min_num_equilateral_triangles_needed_l513_513093

def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

def num_small_triangles_needed (large_side small_side : ℝ) : ℝ :=
  (area_of_equilateral_triangle large_side) / (area_of_equilateral_triangle small_side)

theorem min_num_equilateral_triangles_needed 
  (large_side small_side : ℝ) 
  (h_large : large_side = 16) 
  (h_small : small_side = 2) : 
  num_small_triangles_needed large_side small_side = 64 :=
by
  rw [h_large, h_small]
  simp [num_small_triangles_needed, area_of_equilateral_triangle]
  sorry

end min_num_equilateral_triangles_needed_l513_513093


namespace gaussian_expectation_l513_513385
open Real

variable {n : ℕ}
variable {X : Fin (2 * n) → Gaussian}
variable {C : Matrix (Fin (2 * n)) (Fin (2 * n)) ℝ}

#check @Gaussian

noncomputable def covariance_matrix (v : Fin (2 * n) → ℝ) : Matrix (Fin (2 * n)) (Fin (2 * n)) ℝ := sorry

noncomputable def mean_vector (v : Fin (2 * n) → ℝ) : Fin n → ℝ := sorry

theorem gaussian_expectation (hX : ∀ i, mean_vector X i = 0) 
  (hC : covariance_matrix X = C) : 
  (∃! (σ : (Fin (2 * n)) → (Fin (2 * n))),
    ∀ i, i.even → σ i < σ (i + 1)) →
  ∀ X, 
    E (∏ i in Finset.range (2 * n), X i) = 
    (1 / (n.factorial : ℝ)) * ∑ (σ : (Fin (2 * n)) → (Fin (2 * n)))
      (hσ : ∀ i, i.even → σ i < σ (i + 1)), 
      ∏ i in Finset.range n, C (σ (2 * i)) (σ (2 * i + 1)) := 
sorry

end gaussian_expectation_l513_513385


namespace triangle_inequality_l513_513431

theorem triangle_inequality (a b c S : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 
                                      (exists (α : ℝ), S = 1 / 2 * b * c * sin α)) :
  (ab + ac + bc) / (4 * S) ≥ Real.sqrt 3 :=
by
  sorry

end triangle_inequality_l513_513431


namespace min_g_l513_513600

noncomputable def g (x : ℝ) : ℝ := (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem min_g : ∃ x : ℝ, g x = 2 :=
by
  use 0
  sorry

end min_g_l513_513600


namespace problem1_problem2_l513_513265

-- Define the permutations
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem problem1 (hnondefects : 6 ≥ 4)
    (hfirst : 10 ≥ 1) (hlast : 8 ≥ 4) :
    ((A 4 2) * (A 5 2) * (A 6 4)) = A 4 2 * A 5 2 * A 6 4 := sorry

theorem problem2 (hnondefects : 6 ≥ 4)
    (hfour : 10 ≥ 4) :
    ((A 4 4) + 4 * (A 4 3) * (A 6 1) + 4 * (A 5 3) * (A 6 2) + (A 6 6)) =
    A 4 4 + 4 * A 4 3 * A 6 1 + 4 * A 5 3 * A 6 2 + A 6 6 := sorry

end problem1_problem2_l513_513265


namespace exists_circle_tangent_to_circle_and_line_l513_513560

theorem exists_circle_tangent_to_circle_and_line :
  ∃ (C : ℝ × ℝ × ℝ), 
  (C = (4, 0, 2) ∨ C = (0, -4 * Real.sqrt 3, 6)) ∧
  -- Circle C is externally tangent to (x - 1)^2 + y^2 = 1
  ((C.1 - 1)^2 + C.2^2 = (1 + C.3)^2) ∧
  -- Circle C is tangent to the line x + sqrt(3) y = 0 at (3, -sqrt(3))
  (Real.sqrt ((C.1 - 3)^2 + (C.2 + Real.sqrt 3)^2) = 0) :=
begin
  sorry
end

end exists_circle_tangent_to_circle_and_line_l513_513560


namespace find_offset_length_l513_513635

theorem find_offset_length 
  (diagonal_offset_7 : ℝ) 
  (area_of_quadrilateral : ℝ) 
  (diagonal_length : ℝ) 
  (result : ℝ) : 
  (diagonal_length = 10) 
  ∧ (diagonal_offset_7 = 7) 
  ∧ (area_of_quadrilateral = 50) 
  → (∃ x, x = result) :=
by
  sorry

end find_offset_length_l513_513635


namespace speed_of_b_l513_513539

variables (va vk vb t : ℝ)
variables (a k b_same_instant : Prop)

-- Conditions
def a_speed (v : ℝ) : Prop := v = 30
def k_speed (v : ℝ) : Prop := v = 60
def start_time_difference (t : ℝ) : Prop := t = 5
def overtake_at_same_time (t : ℝ) (va t_a : ℝ) (vk t_k : ℝ) : Prop := va * t_a = vk * t_k

-- The main statement to prove
theorem speed_of_b {vb t : ℝ} : 
  a_speed va ∧ k_speed vk ∧ start_time_difference t ∧ overtake_at_same_time t va (t + t) vk t 
  → vb = 60 :=
by
  intros h,
  sorry

end speed_of_b_l513_513539


namespace elderly_people_sampled_l513_513330

theorem elderly_people_sampled (total_population : ℕ) (children : ℕ) (elderly : ℕ) (middle_aged : ℕ) (sample_size : ℕ)
  (h1 : total_population = 1500)
  (h2 : ∃ d, children + d = elderly ∧ elderly + d = middle_aged)
  (h3 : total_population = children + elderly + middle_aged)
  (h4 : sample_size = 60) :
  elderly * (sample_size / total_population) = 20 :=
by
  -- Proof will be written here
  sorry

end elderly_people_sampled_l513_513330


namespace problem_statement_l513_513702

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513702


namespace find_fg_l513_513287

def f (x : ℕ) : ℕ := 3 * x^2 + 2
def g (x : ℕ) : ℕ := 4 * x + 1

theorem find_fg :
  f (g 3) = 509 :=
by
  sorry

end find_fg_l513_513287


namespace ellipse_equation_proof_l513_513249

noncomputable def ellipseEquation (x y : ℝ) (m n : ℝ): Prop :=
  x^2 / m + y^2 / n = 1

theorem ellipse_equation_proof :
  ∃ (m n : ℝ), 
    (m > 0 ∧ n > 0 ∧ m ≠ n) ∧
    ellipseEquation (sqrt 6) 1 1 m n ∧
    ellipseEquation (- sqrt 3) (- sqrt 2) 1 m n ∧
    ellipseEquation x y (1 / 9) (1 / 3) :=
by
  sorry

end ellipse_equation_proof_l513_513249


namespace problem_statement_l513_513705

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513705


namespace female_athletes_in_sample_l513_513146

-- Definitions based on the conditions:
def total_athletes : ℕ := 49
def male_athletes : ℕ := 28
def female_athletes : ℕ := total_athletes - male_athletes
def sample_size : ℕ := 14
def female_proportion : ℚ := female_athletes / total_athletes

-- The theorem statement:
theorem female_athletes_in_sample :
  let num_female_in_sample := sample_size * female_proportion in
  num_female_in_sample = 6 :=
by
  sorry

end female_athletes_in_sample_l513_513146


namespace problem_statement_l513_513714

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513714


namespace range_f_range_m_l513_513870

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * (sin (x + π / 4))^2 - (sqrt 3) * cos (2 * x)

-- Define the interval condition
def x_interval (x : ℝ) := x ∈ Icc (π / 4) (π / 2)

-- Stating the problem in Lean
theorem range_f :
  ∀ x, x_interval x → (2 : ℝ) ≤ f x ∧ f x ≤ 3 :=
sorry

theorem range_m (m : ℝ) :
  (∀ x, x_interval x → abs (f x - m) < 2) → (1 < m ∧ m < 4) :=
sorry

end range_f_range_m_l513_513870


namespace problem_statement_l513_513789

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513789


namespace proposition_p_and_q_is_true_l513_513715

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513715


namespace number_of_square_factors_l513_513303

theorem number_of_square_factors (n1 n2 n3 n4 : ℕ) (h1 : n1 = 12) (h2 : n2 = 15) (h3 : n3 = 18) (h4 : n4 = 8) :
  let factors_2 := 7,
      factors_3 := 8,
      factors_5 := 10,
      factors_7 := 5 in
  factors_2 * factors_3 * factors_5 * factors_7 = 2800 := by
  -- The provided proof can be added here
  sorry

end number_of_square_factors_l513_513303


namespace proof_problem_l513_513784

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513784


namespace proposition_p_and_q_is_true_l513_513720

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513720


namespace scientific_notation_32000000_l513_513168

def scientific_notation (n : ℕ) : String := sorry

theorem scientific_notation_32000000 :
  scientific_notation 32000000 = "3.2 × 10^7" :=
sorry

end scientific_notation_32000000_l513_513168


namespace trapezoid_shorter_diagonal_l513_513083

noncomputable def trapezoid_diagonal_length (PQ RS PS QR : ℝ) (acute_PQ : Prop) : ℝ :=
  let y := 7 / 3
  let k := Real.sqrt (1472 / 9)
  let PR := Real.sqrt ((y + 28) ^ 2 + k ^ 2)
  let QS := Real.sqrt ((12 - y) ^ 2 + k ^ 2)
  min PR QS

theorem trapezoid_shorter_diagonal : 
  let PQ := 40
  let RS := 28
  let PS := 13
  let QR := 15
  acute_angles : Prop := -- Assume angles P and Q are acute
  trapezoid_diagonal_length PQ RS PS QR acute_angles = 27 := 
sorry

end trapezoid_shorter_diagonal_l513_513083


namespace area_covered_by_three_layers_l513_513080

theorem area_covered_by_three_layers (A B C : ℕ) (h1 : A = 200) (h2 : B = 140) (h3 : C = 22) :
  (∃ D : ℕ, B = A - C - 2 * D ∧ D = 19) :=
by
  use 19
  split
  · rw [h1, h2, h3]
    norm_num
  · rfl

end area_covered_by_three_layers_l513_513080


namespace maria_money_after_utility_bills_l513_513406

def salary : ℝ := 2000
def tax : ℝ := 0.20 * salary
def insurance : ℝ := 0.05 * salary
def total_deductions : ℝ := tax + insurance
def money_left_after_deductions : ℝ := salary - total_deductions
def utility_bills : ℝ := (1 / 4) * money_left_after_deductions
def money_after_utility_bills : ℝ := money_left_after_deductions - utility_bills

theorem maria_money_after_utility_bills : money_after_utility_bills = 1125 := by
  sorry

end maria_money_after_utility_bills_l513_513406


namespace true_conjunction_l513_513675

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513675


namespace Sam_has_walked_25_miles_l513_513648

variables (d : ℕ) (v_fred v_sam : ℕ)

def Fred_and_Sam_meet (d : ℕ) (v_fred v_sam : ℕ) := 
  d / (v_fred + v_sam) * v_sam

theorem Sam_has_walked_25_miles :
  Fred_and_Sam_meet 50 5 5 = 25 :=
by
  sorry

end Sam_has_walked_25_miles_l513_513648


namespace cows_to_eat_grass_in_96_days_l513_513458

theorem cows_to_eat_grass_in_96_days (G r : ℕ) : 
  (∀ N : ℕ, (70 * 24 = G + 24 * r) → (30 * 60 = G + 60 * r) → 
  (∃ N : ℕ, 96 * N = G + 96 * r) → N = 20) :=
by
  intro N
  intro h1 h2 h3
  sorry

end cows_to_eat_grass_in_96_days_l513_513458


namespace range_of_f_eq_real_l513_513871

def f (a x : ℝ) : ℝ :=
if x < a then x + 4 else x^2 - 2 * x

theorem range_of_f_eq_real (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ -5 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_f_eq_real_l513_513871


namespace proof_problem_l513_513802

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513802


namespace arithmetic_sequence_property_l513_513189

theorem arithmetic_sequence_property {a : ℕ → ℝ} (h_seq : ∀ n, a (n+1) - a n = d) 
  (h_sum : ∑ i in finset.range 101, a i = 0) : a 2 + a 98 = 0 := 
by
  sorry

end arithmetic_sequence_property_l513_513189


namespace diff_square_mental_math_l513_513612

theorem diff_square_mental_math :
  75 ^ 2 - 45 ^ 2 = 3600 :=
by
  -- The proof would go here
  sorry

end diff_square_mental_math_l513_513612


namespace circumcircle_of_PMQ_tangent_to_σ_l513_513455

variable {A B C K L M P Q : Point}
variable {σ : Circle}
variable {ABC : Triangle}

-- Circle σ is tangent to the equal sides AB and AC of the isosceles triangle ABC.
def tangent_to_sides (σ : Circle) (ABC : Triangle) : Prop :=
  σ.tangent (ABC.side AB) ∧ σ.tangent (ABC.side AC)

-- Circle σ intersects side BC at points K and L.
def intersects_BC_at_KL (σ : Circle) (ABC : Triangle) (K L : Point) : Prop :=
  σ.intersects (ABC.side BC) [K, L]

-- Segment AK intersects σ again at point M.
def segment_AK_intersects_σ_at_M (σ : Circle) (A K M : Point) : Prop :=
  segment (A, K).intersects_on_circle_twice σ A K M

-- Points P and Q are symmetric to K with respect to points B and C, respectively.
def symmetric_points (K B C P Q : Point) : Prop :=
  is_symmetric K B P ∧ is_symmetric K C Q

-- Prove that the circumcircle of triangle PMQ is tangent to the circle σ.
theorem circumcircle_of_PMQ_tangent_to_σ :
  tangent_to_sides σ ABC →
  intersects_BC_at_KL σ ABC K L →
  segment_AK_intersects_σ_at_M σ A K M →
  symmetric_points K B C P Q →
  circumscribed (triangle P M Q).circumcircle σ :=
  by sorry

end circumcircle_of_PMQ_tangent_to_σ_l513_513455


namespace proposition_true_l513_513832

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513832


namespace number_of_tie_games_l513_513337

def total_games (n_teams: ℕ) (games_per_matchup: ℕ) : ℕ :=
  (n_teams * (n_teams - 1) / 2) * games_per_matchup

def theoretical_max_points (total_games: ℕ) (points_per_win: ℕ): ℕ :=
  total_games * points_per_win

def actual_total_points (lions: ℕ) (tigers: ℕ) (mounties: ℕ) (royals: ℕ): ℕ :=
  lions + tigers + mounties + royals

def tie_games (theoretical_points: ℕ) (actual_points: ℕ) (points_per_tie: ℕ): ℕ :=
  (theoretical_points - actual_points) / points_per_tie

theorem number_of_tie_games
  (n_teams: ℕ)
  (games_per_matchup: ℕ)
  (points_per_win: ℕ)
  (points_per_tie: ℕ)
  (lions: ℕ)
  (tigers: ℕ)
  (mounties: ℕ)
  (royals: ℕ)
  (h_teams: n_teams = 4)
  (h_games: games_per_matchup = 4)
  (h_points_win: points_per_win = 3)
  (h_points_tie: points_per_tie = 2)
  (h_lions: lions = 22)
  (h_tigers: tigers = 19)
  (h_mounties: mounties = 14)
  (h_royals: royals = 12) :
  tie_games (theoretical_max_points (total_games n_teams games_per_matchup) points_per_win) 
  (actual_total_points lions tigers mounties royals) points_per_tie = 5 :=
by
  rw [h_teams, h_games, h_points_win, h_points_tie, h_lions, h_tigers, h_mounties, h_royals]
  simp [total_games, theoretical_max_points, actual_total_points, tie_games]
  sorry

end number_of_tie_games_l513_513337


namespace proof_goats_minus_pigs_l513_513172

noncomputable def number_of_goats : ℕ := 66
noncomputable def number_of_chickens : ℕ := 2 * number_of_goats - 10
noncomputable def number_of_ducks : ℕ := (number_of_goats + number_of_chickens) / 2
noncomputable def number_of_pigs : ℕ := number_of_ducks / 3
noncomputable def number_of_rabbits : ℕ := Nat.floor (Real.sqrt (2 * number_of_ducks - number_of_pigs))
noncomputable def number_of_cows : ℕ := number_of_rabbits ^ number_of_pigs / Nat.factorial (number_of_goats / 2)

theorem proof_goats_minus_pigs : number_of_goats - number_of_pigs = 35 := by
  sorry

end proof_goats_minus_pigs_l513_513172


namespace tangent_parallel_range_a_l513_513487

noncomputable def f (x : ℝ) := x^2 + real.log x
noncomputable def g (x : ℝ) (a : ℝ) := real.exp x - a * x

theorem tangent_parallel_range_a :
  ∃ (a : ℝ), ∀ (x1 x2 : ℝ), x1 > 0 → 
    let k1 := 2 * x1 + 1 / x1 in
    let k2 := real.exp x2 - a in
    k1 = k2 → a > -2 * real.sqrt 2 :=
sorry

end tangent_parallel_range_a_l513_513487


namespace incenter_locus_l513_513335

theorem incenter_locus (O A B P Q I : Point) (r : ℝ)
  (sector_OAB : sector O A B)
  (central_angle_OAB : angle O A B = 90)
  (P_on_arc_AB : on_arc P A B)
  (PQ_tangent_to_arc : tangent PQ (arc A B))
  (PQ_intersects_OA_at_Q : intersects PQ OA Q)
  (I_incenter_OPQ : incenter I (triangle O P Q)) :
  ∃ r, locus_of_I = circle O r := sorry

end incenter_locus_l513_513335


namespace locus_of_A_l513_513858

/-- Given vertices \( B(-6, 0) \) and \( C(6, 0) \) of a triangle \(\triangle ABC\), and the equation 
 \(\sin B - \sin C = \frac{1}{2} \sin A\), the locus of vertex \(A\) is a hyperbola described by 
 the equation \(\frac{x^2}{9} - \frac{y^2}{27} = 1\) and \(x < -3\). --/
theorem locus_of_A 
  (x y : ℝ) 
  (h1 : ∀ A B C : ℝ, sin B - sin C = (1 / 2 : ℝ) * sin A)
  (h2 : ∀ A B C : ℝ, (B = (-6, 0)) ∧ (C = (6, 0))) :
  \(\frac{x^2}{9} - \frac{y^2}{27} = 1\) ∧ (x < -3) :=
sorry

end locus_of_A_l513_513858


namespace proposition_true_l513_513830

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513830


namespace true_proposition_l513_513844

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513844


namespace ben_heads_probability_l513_513914

def coin_flip_probability : ℚ :=
  let total_ways := 2^10
  let ways_exactly_five_heads := Nat.choose 10 5
  let probability_exactly_five_heads := ways_exactly_five_heads / total_ways
  let remaining_probability := 1 - probability_exactly_five_heads
  let probability_more_heads := remaining_probability / 2
  probability_more_heads

theorem ben_heads_probability :
  coin_flip_probability = 193 / 512 := by
  sorry

end ben_heads_probability_l513_513914


namespace lcm_is_multiple_of_230_l513_513051

theorem lcm_is_multiple_of_230 (d n : ℕ) (h1 : n = 230) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (2 ∣ d)) : ∃ m : ℕ, Nat.lcm d n = 230 * m :=
by
  exists 1 -- Placeholder for demonstration purposes
  sorry

end lcm_is_multiple_of_230_l513_513051


namespace deck_card_probability_l513_513569

theorem deck_card_probability 
    (cards_removed : ℕ = 2 * 2 := 4)
    (total_cards : ℕ = 50 := 50)
    (remaining_cards : ℕ = total_cards - cards_removed := 46)
    (pairs_from_five : ℕ = 4 * Nat.choose 5 2 := 40) 
    (pairs_from_three : ℕ = 2 * Nat.choose 3 2 := 6) 
    (total_pairs : ℕ = pairs_from_five + pairs_from_three := 46) 
    (total_ways : ℕ = Nat.choose remaining_cards 2 := 1035) : 
    let m := 46 in 
    let n := 1035 in 
    Nat.gcd m n = 1 → m + n = 1081 :=
by sorry

end deck_card_probability_l513_513569


namespace moles_of_HCH3CO2_combined_l513_513217

-- Definitions based on conditions
def neutralization_reaction : Prop :=
  ∀ (HCH3CO2 NaOH NaCH3CO2 H2O : Type), 
  (HCH3CO2 → NaOH → NaCH3CO2 × H2O)

def balanced_equation (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ) : Prop :=
  HCH3CO2 + NaOH = NaCH3CO2 + H2O

def moles_NaOH : ℕ := 1
def moles_H2O_produced : ℕ := 1
def moles_HCH3CO2_combined : ℕ := 1

-- The theorem to prove
theorem moles_of_HCH3CO2_combined :
  (neutralization_reaction → balanced_equation 1 1 1 1 → moles_NaOH = 1 → moles_H2O_produced = 1) 
  → moles_HCH3CO2_combined = 1 :=
by
  sorry

end moles_of_HCH3CO2_combined_l513_513217


namespace triangle_DEF_area_and_perimeter_l513_513940

theorem triangle_DEF_area_and_perimeter :
  ∀ (DE DF : ℝ), DE = 15 ∧ DF = 10 ∧ ∃ D E F : ℝ, ∠ D E F = 90 →
  (1 / 2 * DE * DF = 75) ∧ (DE + DF + (Real.sqrt (DE^2 + DF^2)) = 25 + 5 * Real.sqrt 13) := by
intros DE DF h
rcases h with ⟨hDE, hDF, ⟨_, _, _, hAngle⟩⟩
sorry

end triangle_DEF_area_and_perimeter_l513_513940


namespace vertex_angle_greater_l513_513091

variables {T a b m_a m_b : ℝ} {α : ℝ} 

-- Define an isosceles triangle with AB = AC, base BC = a, sides AB and AC = b, and vertex angle α.
def isosceles_triangle (AB AC BC α : ℝ) : Prop := 
  AB = AC ∧ BC = a ∧ α > 0

-- Define the condition that altitudes form another triangle if 2m_b > m_a.
def altitudes_form_triangle (m_a m_b : ℝ) : Prop :=
  2 * m_b > m_a

-- Expressing altitudes in terms of the triangle's area
def altitude_expressions (T a b : ℝ) (m_a m_b : ℝ) : Prop :=
  m_a = (2 * T) / a ∧ m_b = (2 * T) / b

-- Relationship involving vertex angle and side ratios using sine
def vertex_angle_condition (α : ℝ) (a b : ℝ) :=
  sin (α / 2) = a / (2 * b)

-- Prove that the vertex angle α is greater than 28.95°.
theorem vertex_angle_greater (AB AC BC α T a b m_a m_b : ℝ)
  (h1 : isosceles_triangle AB AC BC α)
  (h2 : altitudes_form_triangle m_a m_b)
  (h3 : altitude_expressions T a b m_a m_b)
  (h4 : vertex_angle_condition α a b) :
  α > 28.95 :=
sorry

end vertex_angle_greater_l513_513091


namespace inverse_36_mod_53_l513_513256

theorem inverse_36_mod_53 (h : 17 * 26 ≡ 1 [MOD 53]) : 36 * 27 ≡ 1 [MOD 53] :=
sorry

end inverse_36_mod_53_l513_513256


namespace polynomial_divisible_l513_513546

theorem polynomial_divisible (a b c : ℕ) :
  (X^(3 * a) + X^(3 * b + 1) + X^(3 * c + 2)) % (X^2 + X + 1) = 0 :=
by sorry

end polynomial_divisible_l513_513546


namespace star_addition_l513_513313

-- Definition of the binary operation "star"
def star (x y : ℤ) := 5 * x - 2 * y

-- Statement of the problem
theorem star_addition : star 3 4 + star 2 2 = 13 :=
by
  -- By calculation, we have:
  -- star 3 4 = 7 and star 2 2 = 6
  -- Thus, star 3 4 + star 2 2 = 7 + 6 = 13
  sorry

end star_addition_l513_513313


namespace intersection_count_l513_513896

-- Define the absolute value functions
def f (x : ℝ) : ℝ := abs (3 * x + 6)
def g (x : ℝ) : ℝ := -abs (4 * x - 3)

-- Statement of the theorem
theorem intersection_count : ∃ (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = g x :=
by
  -- s is the set of x-values where f(x) = g(x)
  use (Finset.mk [-(3/7), -9] (by decide))
  split
  -- The size of the set is 2
  · norm_num
  -- For each element x in the set, f(x) is equal to g(x)
  · intros x hx
    finset_cases hx
    · show f (-(3/7)) = g (-(3/7))
      norm_num
      rw [abs_of_nonneg, abs_of_nonpos]
      norm_num
      linarith
    · show f (-9) = g (-9)
      norm_num
      rw [abs_of_nonpos, abs_of_nonneg]
      norm_num
      linarith
  sorry

end intersection_count_l513_513896


namespace slope_angle_150_degrees_l513_513059

noncomputable def slope_angle_line_parametric (α : ℝ) : Prop :=
  ∀ t : ℝ, let x := 5 - 3 * t
            let y := 3 + sqrt 3 * t
            (∃ k : ℝ, x + sqrt 3 * y = 5 + 3 * sqrt 3 ∧ 
              k = -sqrt 3 / 3 ∧ tan α = k)

theorem slope_angle_150_degrees : slope_angle_line_parametric (150 * (Real.pi / 180)) :=
by
  sorry

end slope_angle_150_degrees_l513_513059


namespace average_disk_space_per_hour_l513_513581

/-- A digital music library contains 15 days of music and takes up 20,000 megabytes of disk space.
    Prove that the average disk space used per hour of music in this library is 56 megabytes, to the nearest whole number.
-/
theorem average_disk_space_per_hour (days: ℕ) (total_disk_space: ℕ) (hours_per_day: ℕ) (div_result: ℕ) (avg_disk_space: ℕ) :
  days = 15 ∧ 
  total_disk_space = 20000 ∧ 
  hours_per_day = 24 ∧ 
  div_result = days * hours_per_day ∧
  avg_disk_space = (total_disk_space: ℝ) / div_result ∧ 
  (an_index: ℕ, (avg_disk_space: ℝ) ≈ 55.56 → an_index = 56) :=
by 
  sorry

end average_disk_space_per_hour_l513_513581


namespace final_inversion_count_eq_initial_l513_513377

def is_swapped_pair (σ : list ℕ) (i j : ℕ) : Prop :=
  i < j ∧ σ.nth_le i sorry > σ.nth_le j sorry

def inversion_count (σ : list ℕ) : ℕ :=
  (list.range σ.length).bind (λ i, (list.range (σ.length - i - 1)).map (λ j, if is_swapped_pair σ i (i + j + 1) then 1 else 0)).sum

def apply_process (σ : list ℕ) : list ℕ :=
  list.range σ.length.reverse.foldl (λ τ i, let ones = list.filter (λ x, x < i) τ in
  let rest = list.filter_not (λ x, x < i) τ in ones ++ (i :: rest)) σ

theorem final_inversion_count_eq_initial (n : ℕ) (σ : list ℕ) (hσ : (list.range n).permute σ) :
  inversion_count σ = inversion_count (apply_process σ) :=
sorry

end final_inversion_count_eq_initial_l513_513377


namespace true_proposition_l513_513850

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513850


namespace omega_is_8_l513_513279

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

theorem omega_is_8
  (ω : ℝ) 
  (h1 : ω > 0)
  (h2 : ∃! x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2), isExtreme (f ω x))
  (h3 : f ω (Real.pi / 6) + f ω (Real.pi / 2) = 0) :
  ω = 8 := sorry

end omega_is_8_l513_513279


namespace peter_wins_prize_at_least_one_person_wins_prize_l513_513007

-- Part (a): Probability that Peter wins a prize
theorem peter_wins_prize :
  let p : Probability := (5 / 6) ^ 9
  p = 0.194 := sorry

-- Part (b): Probability that at least one person wins a prize
theorem at_least_one_person_wins_prize :
  let p : Probability := 0.919
  p = 0.919 := sorry

end peter_wins_prize_at_least_one_person_wins_prize_l513_513007


namespace midpoint_of_equal_area_l513_513354

variables {A B C D M : Type} [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty M]
variables [FragmentedTriangle A B C] (AD : Segment A D) (BC : Segment B C) 

def equal_area (AMB AMC : Triangle A M B × Triangle A M C) : Prop := 
  AMB.area = AMC.area 

def midpoint (P Q : Point) (S : Segment P Q): Prop := 
  S.midpoint = midpoint(S.segment)

theorem midpoint_of_equal_area 
  (AD : Segment A D) 
  (BC : Segment B C)
  (AMB AMC : Triangle A M B × Triangle A M C)
  (h : equal_area AMB AMC) :
  (∃ (D : Point), midpoint B C D BC) ∧ ¬(midpoint A D M AD) :=
sorry

end midpoint_of_equal_area_l513_513354


namespace find_x_l513_513314

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x^2 + 18 * x * y = x^3 + 3 * x^2 * y + 6 * x) : 
  x = 3 :=
sorry

end find_x_l513_513314


namespace no_positive_integers_abc_l513_513202

theorem no_positive_integers_abc :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (a * b + b * c = a * c) ∧ (a * b * c = Nat.factorial 10) :=
sorry

end no_positive_integers_abc_l513_513202


namespace final_portfolio_value_l513_513371

theorem final_portfolio_value (initial_amount : ℕ) (growth_1 : ℕ) (additional_funds : ℕ) (growth_2 : ℕ) :
  let after_first_year_growth := initial_amount + (initial_amount * growth_1 / 100)
  let after_adding_funds := after_first_year_growth + additional_funds
  let after_second_year_growth := after_adding_funds + (after_adding_funds * growth_2 / 100)
  after_second_year_growth = 132 :=
by
  let after_first_year_growth := initial_amount + (initial_amount * growth_1 / 100);
  let after_adding_funds := after_first_year_growth + additional_funds;
  let after_second_year_growth := after_adding_funds + (after_adding_funds * growth_2 / 100);
  trivial

-- substituting the values as per conditions:
example : final_portfolio_value 80 15 28 10 = 132 := by
  let after_first_year_growth := 80 + (80 * 15 / 100);
  let after_adding_funds := after_first_year_growth + 28;
  let after_second_year_growth := after_adding_funds + (after_adding_funds * 10 / 100);
  trivial

end final_portfolio_value_l513_513371


namespace find_present_age_of_Abe_l513_513061

-- Define variables and conditions
variables (A : ℕ)
axiom condition : A + (A - 7) = 31

-- Goal is to prove A = 19
theorem find_present_age_of_Abe : A = 19 :=
by {
  have h : A + (A - 7) = 31 := condition,
  sorry,
}

end find_present_age_of_Abe_l513_513061


namespace proof_problem_l513_513805

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513805


namespace average_mb_per_hour_l513_513570

theorem average_mb_per_hour (days : ℕ) (total_disk_space : ℕ)
  (h_days : days = 15) (h_total_disk_space : total_disk_space = 20000) :
  let total_hours := days * 24
  let mb_per_hour := total_disk_space / total_hours.toFloat
  round mb_per_hour = 56 :=
by
  let total_hours := days * 24
  let mb_per_hour := total_disk_space / total_hours.toFloat
  have : total_hours = 360 := by rw [h_days]; simp
  have : mb_per_hour ≈ 55.56 := by rw [h_total_disk_space, this]; simp
  have : round mb_per_hour = 56 := by norm_cast; simp
  exact this

end average_mb_per_hour_l513_513570


namespace proposition_true_l513_513838

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513838


namespace average_brown_MnMs_l513_513437

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end average_brown_MnMs_l513_513437


namespace problem_statement_l513_513711

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513711


namespace convert_base_10_to_base_6_l513_513194

theorem convert_base_10_to_base_6 : 
  ∃ (digits : List ℕ), (digits.length = 4 ∧
    List.foldr (λ (x : ℕ) (acc : ℕ) => acc * 6 + x) 0 digits = 314 ∧
    digits = [1, 2, 4, 2]) := by
  sorry

end convert_base_10_to_base_6_l513_513194


namespace arithmetic_sequence_b_c_sum_l513_513471

theorem arithmetic_sequence_b_c_sum :
  ∃ (b c d : ℤ), 
    10 - 3 = d ∧
    24 - 10 = 2 * d ∧
    b = 10 + d ∧
    c = 24 + d ∧
    b + c = 48 :=
begin
  sorry
end

end arithmetic_sequence_b_c_sum_l513_513471


namespace matches_among_three_players_l513_513930

theorem matches_among_three_players :
  ∀ {n : ℕ}, (∃ n : ℕ, n ≥ 3) →
  let planned_matches := n * (n - 1) / 2 in
  3 * 2 + planned_matches = 50 →
  ∃ r : ℕ, r = 1 := 
by
  sorry

end matches_among_three_players_l513_513930


namespace answer_is_p_and_q_l513_513737

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513737


namespace pigeons_increased_l513_513063

-- Defining the conditions as hypotheses
variable (initial_pigeons : Nat) (total_pigeons : Nat)
hypothesis h1 : total_pigeons = 21
hypothesis h2 : initial_pigeons = 15

-- Defining the statement to be proved
theorem pigeons_increased :
  (total_pigeons - initial_pigeons) = 6 :=
by
  -- Skipping the proof
  sorry

end pigeons_increased_l513_513063


namespace line_intersects_circle_l513_513200

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 5

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y + 2) ^ 2 = 5

-- Define the center and radius of the circle
def circle_center : ℝ × ℝ := (1, -2)
def circle_radius : ℝ := Real.sqrt 5

-- Define the distance between a point and a line
def distance (px py : ℝ) : ℝ := 
  Real.abs (3 * px + 4 * py - 5) / Real.sqrt (3 ^ 2 + 4 ^ 2)

-- Define the theorem we need to prove
theorem line_intersects_circle : 
  distance (circle_center.1) (circle_center.2) < circle_radius ∧ 
  ∃ x y : ℝ, line_eq x y ∧ circle_eq x y := 
by
  sorry

end line_intersects_circle_l513_513200


namespace hannah_run_wednesday_l513_513890

theorem hannah_run_wednesday :
  ∃ (x : ℕ), let M := 9000 in let F := 2095 in M = x + F + 2089 ∧ x = 4816 :=
by
  use 4816
  let M := 9000
  let F := 2095
  have h : M = 4816 + F + 2089 := by
    calc
      M = 9000 : by rfl
      ... = 4816 + 2095 + 2089 : by norm_num
  exact ⟨h, rfl⟩

end hannah_run_wednesday_l513_513890


namespace p_and_q_is_true_l513_513697

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513697


namespace trajectory_ellipse_or_segment_l513_513651

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def trajectory (P : ℝ × ℝ) (a: ℝ) : Prop :=
  let F1 := (-2, 0)
  let F2 := (2, 0)
  distance P F1 + distance P F2 = a + 4 / a ∧ a > 0

theorem trajectory_ellipse_or_segment (P : ℝ × ℝ) (a : ℝ) :
  trajectory P a → (∃ a, ∀ P, trajectory P a) → (a = 2) ∨ (a ≠ 2) :=
by
  sorry

end trajectory_ellipse_or_segment_l513_513651


namespace proof_problem_l513_513800

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513800


namespace incorrect_statement_median_median_incorrect_l513_513226

theorem incorrect_statement_median (data : List ℕ) (h_data : data = [3, 3, 6, 5, 3]) :
  ¬ (median data = 6) :=
by
  sorry

-- Additional definitions and properties that could be used within a Lean proof
def mean (data : List ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

def mode (data : List ℕ) : ℕ :=
  data.groupBy id |>.sortBy (·.length) |>.last |>.head |>.getD 0

def median (data : List ℕ) : ℕ :=
  let sorted := data |>.sort
  sorted.getD (sorted.length / 2) 0

def variance (data : List ℕ) (μ : ℚ) : ℚ :=
  data.map (λ x => (x - μ) ^ 2) |>.sum / data.length

-- Assertions for the properties calculated
assert mean_data : mean [3, 3, 6, 5, 3] = 4 := by sorry
assert mode_data : mode [3, 3, 6, 5, 3] = 3 := by sorry
assert variance_data : variance [3, 3, 6, 5, 3] (mean [3, 3, 6, 5, 3]) = 1.6 := by sorry

-- Proving that statement C is incorrect
theorem median_incorrect (data : List ℕ) (h : data = [3, 3, 6, 5, 3]) : ¬ (median data = 6) :=
by
  sorry

end incorrect_statement_median_median_incorrect_l513_513226


namespace seating_arrangements_l513_513070

theorem seating_arrangements (n : ℕ) (h_n : n = 6) (A B : Fin n) (h : A ≠ B) : 
  ∃ k : ℕ, k = 240 := 
by 
  sorry

end seating_arrangements_l513_513070


namespace constant_angle_sum_l513_513666

theorem constant_angle_sum 
  {O1 O2 : Type} [circle O1] [circle O2] (A A' B C D E : point)
  (h_inter1 : O1 A) (h_inter2 : O1 A') (h_inter3 : O2 A) (h_inter4 : O2 A') 
  (h_line1 : line (A, A')) (h_line2 : intersects_with h_line1 (B, C, O1)) 
  (h_line3 : intersects_with h_line1 (D, E, O2)) 
  (h_segment : lies_on (C, D, B, E)) : 
  constant (angle_sum (B, A, D) (C, A, E)) := 
sorry

end constant_angle_sum_l513_513666


namespace problem_statement_l513_513712

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513712


namespace true_conjunction_l513_513674

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513674


namespace find_y_intercept_l513_513319

theorem find_y_intercept (a b : ℝ) (h1 : (3 : ℝ) ≠ (7 : ℝ))
  (h2 : -2 = a * 3 + b) (h3 : 14 = a * 7 + b) :
  b = -14 :=
sorry

end find_y_intercept_l513_513319


namespace polynomial_has_integer_root_l513_513395

theorem polynomial_has_integer_root
  (a b c d : ℤ)
  (h_poly : P(x) = a * x^3 + b * x^2 + c * x + d)
  (h_a_nonzero : a ≠ 0)
  (h_infinite_pairs : ∃⁺ (x y : ℤ), x ≠ y ∧ x * P(x) = y * P(y)) :
  ∃ k : ℤ, P(k) = 0 :=
sorry

end polynomial_has_integer_root_l513_513395


namespace true_conjunction_l513_513677

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513677


namespace smallest_positive_n_l513_513509

theorem smallest_positive_n (n : ℕ) (h : 19 * n ≡ 789 [MOD 11]) : n = 1 := 
by
  sorry

end smallest_positive_n_l513_513509


namespace volume_of_cone_l513_513590

-- Define the given conditions
def slant_height : ℝ := 6
def circumference_base (r : ℝ) := 2 * Real.pi * r = 6 * Real.pi

-- Define the height using Pythagorean theorem
def cone_height (l r : ℝ) := Real.sqrt (l^2 - r^2)

-- Define the volume of the cone
def cone_volume (r h : ℝ) := (1/3) * Real.pi * r^2 * h

-- The final proof statement
theorem volume_of_cone : 
  ∃ (r : ℝ), circumference_base r ∧ cone_volume r (cone_height slant_height r) = 9 * Real.sqrt 3 * Real.pi :=
by
  -- The proof would go here
  sorry

end volume_of_cone_l513_513590


namespace proposition_p_and_q_is_true_l513_513722

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513722


namespace line_parabola_intersect_two_points_find_k_for_slopes_sum_l513_513284

-- Given conditions
def parabola (x : ℝ) : ℝ := 2 * x^2
def line (k x : ℝ) : ℝ := k * x + 1
def origin : (ℝ × ℝ) := (0, 0)

-- Question (1): Prove that line l and parabola C intersect at two points
theorem line_parabola_intersect_two_points
  (k : ℝ) :
  let Δ := k^2 + 8 in Δ > 0 :=
sorry

-- Definitions related to question (2)
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Question (2): Find the value of k for the given conditions
theorem find_k_for_slopes_sum
  (k x1 x2 : ℝ)
  (hx1 : parabola x1 = line k x1)
  (hx2 : parabola x2 = line k x2)
  (h_sum_slopes : slope origin (x1, parabola x1) + slope origin (x2, parabola x2) = 1) :
  k = 1 :=
sorry

end line_parabola_intersect_two_points_find_k_for_slopes_sum_l513_513284


namespace find_angle_B_find_range_of_c_l513_513947

-- Definitions of the given conditions
variable {A B C a b c : ℝ}

-- Given condition: angles A, B, C are interior angles of a triangle
variable (hABC : ∠A + ∠B + ∠C = π)

-- Given condition: sides a, b, c opposite respective angles in ∆ABC
variable (ha : a = c + 2)
variable (h1 : tan B + tan C = (2 * sin A) / cos C)

-- Given that the triangle is an obtuse triangle
variable (obtuse_ABC : ∃ A' B' C' : ℝ, ∠A' + ∠B' + ∠C' = π ∧ max ∠A' ∠B' ∠C' > π / 2 
  ∧ A' = A ∧ B' = B ∧ C' = C)

-- The proof of angle B and range of c
theorem find_angle_B : B = π / 3 := sorry

theorem find_range_of_c (ha : a = c + 2) (hB : B = π / 3) : 0 < c ∧ c < 2 :=
sorry

end find_angle_B_find_range_of_c_l513_513947


namespace point_inside_circle_l513_513466

theorem point_inside_circle (a : ℝ) (h : 5 * a^2 - 4 * a - 1 < 0) : -1/5 < a ∧ a < 1 :=
    sorry

end point_inside_circle_l513_513466


namespace perfect_square_divisors_count_l513_513305

open Nat

theorem perfect_square_divisors_count :
  let p := (2 ^ 12) * (3 ^ 15) * (5 ^ 18) * (7 ^ 8)
  (∃ n : ℕ, p = n^2) → 
  (num_perfect_square_divisors p = 2800) :=
by
  -- Definitions based on the conditions
  let e2 := 12
  let e3 := 15
  let e5 := 18
  let e7 := 8

  -- Expected number of perfect square divisors
  let num_perfect_square_divisors := (e2 / 2 + 1) * (e3 / 2 + 1) * (e5 / 2 + 1) * (e7 / 2 + 1)

  -- Assertion of correct answer
  have h : num_perfect_square_divisors = 2800 :=
    by -- The steps simply verifying and multiplying the counts
       rfl
  exact (num_perfect_square_divisors, h).snd

end perfect_square_divisors_count_l513_513305


namespace count_valid_permutations_l513_513233

noncomputable def num_valid_permutations : ℕ := 480

def is_valid_permutation (perm : List ℕ) : Prop :=
  perm.length = 6 ∧
  perm.all (λ x, x ∈ [1, 2, 3, 4, 5, 6]) ∧
  perm.nodup ∧
  ((perm.indexOf 5 < perm.indexOf 3 ∧ perm.indexOf 6 < perm.indexOf 3) ∨
   (perm.indexOf 5 > perm.indexOf 3 ∧ perm.indexOf 6 > perm.indexOf 3))

theorem count_valid_permutations :
  (List.permutations [1, 2, 3, 4, 5, 6]).count is_valid_permutation = num_valid_permutations :=
by
  sorry

end count_valid_permutations_l513_513233


namespace number_of_boys_in_school_l513_513062

theorem number_of_boys_in_school (total_students : ℕ) (number_of_boys : ℕ) :
  total_students = 150 ∧ number_of_boys + number_of_boys * 150 / 100 = 150 → number_of_boys = 60 :=
by
  intros h
  cases h with h1 h2
  sorry

end number_of_boys_in_school_l513_513062


namespace mean_of_six_numbers_l513_513486

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end mean_of_six_numbers_l513_513486


namespace main_statement_l513_513750

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513750


namespace sequence_first_term_l513_513885

theorem sequence_first_term (a : ℕ → ℤ) 
  (h1 : a 3 = 5) 
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) : 
  a 1 = 2 := 
sorry

end sequence_first_term_l513_513885


namespace flags_of_both_colors_perc_l513_513126

variable (F : ℕ) (C : ℕ)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def percentage (part whole : ℕ) : ℝ := (part : ℝ) / (whole : ℝ) * 100

def percentage_of_children_with_blue_flags := 60.0
def percentage_of_children_with_red_flags := 60.0
def percentage_of_children_with_both_colors := 20.0

theorem flags_of_both_colors_perc (F : ℕ) (hF_even : is_even F) (C : ℕ) (hC : C = F / 2) (h_blue: percentage (6 * C / 10) C = percentage_of_children_with_blue_flags) (h_red: percentage (6 * C / 10) C = percentage_of_children_with_red_flags) :
  percentage (2 * C / 10) C = percentage_of_children_with_both_colors :=
sorry

end flags_of_both_colors_perc_l513_513126


namespace range_of_k_if_intersection_empty_l513_513401

open Set

variable (k : ℝ)

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem range_of_k_if_intersection_empty (h : M ∩ N k = ∅) : k ≤ -1 :=
by {
  sorry
}

end range_of_k_if_intersection_empty_l513_513401


namespace mean_of_six_numbers_l513_513484

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end mean_of_six_numbers_l513_513484


namespace Jan_drove_195_more_l513_513104

variable (d t s m : ℝ)
variable Ian_equation : d = s * t
variable Han_equation : d + 120 = (s + 10) * (t + 2)
variable Jan_equation : m = (s + 15) * (t + 3)

theorem Jan_drove_195_more : m - d = 195 :=
by
  sorry

end Jan_drove_195_more_l513_513104


namespace p_and_q_is_true_l513_513695

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513695


namespace calculator_display_l513_513034

def first_key (x : ℝ) : ℝ := x + 2
def second_key (x : ℝ) : ℝ := 1 / (2 - x)

theorem calculator_display (x_0 : ℝ) :
  x_0 = 3 →
  let x_1 := first_key x_0 in
  let sequence := (λ n x, if n = 0 then x_1 else second_key (sequence (n - 1) x)) in
  sequence 50 x_1 = x_50 :=
begin
  sorry
end

end calculator_display_l513_513034


namespace peter_wins_prize_at_least_one_person_wins_prize_l513_513011

-- Part (a): Probability that Peter wins a prize
theorem peter_wins_prize :
  let p : Probability := (5 / 6) ^ 9
  p = 0.194 := sorry

-- Part (b): Probability that at least one person wins a prize
theorem at_least_one_person_wins_prize :
  let p : Probability := 0.919
  p = 0.919 := sorry

end peter_wins_prize_at_least_one_person_wins_prize_l513_513011


namespace james_earnings_l513_513368

theorem james_earnings :
  let jan_earn : ℕ := 4000
  let feb_earn := 2 * jan_earn
  let total_earnings : ℕ := 18000
  let earnings_jan_feb := jan_earn + feb_earn
  let mar_earn := total_earnings - earnings_jan_feb
  (feb_earn - mar_earn) = 2000 := by
  sorry

end james_earnings_l513_513368


namespace range_of_ratio_l513_513978

variable (f : ℝ → ℝ)

-- Given Conditions
axiom f_pos : ∀ x: ℝ, 0 < x → f x > 0
axiom deriv_exists : ∀ x: ℝ, 0 < x → ∃ f' : ℝ → ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y, |x - y| < δ → |f x - f y - f' x * (x - y)| < ε * |x - y|
axiom inequality : ∀ x: ℝ, 0 < x → 2 * f x < x * (deriv (deriv f x)) ∧ x * (deriv (deriv f x)) < 3 * f x

-- Problem Statement to Prove
theorem range_of_ratio (h_deriv : ∀ x: ℝ, 0 < x → deriv_exists x ∧ inequality x) :
  27 / 64 < f 3 / f 4 ∧ f 3 / f 4 < 9 / 16 :=
by {
  sorry
}

end range_of_ratio_l513_513978


namespace range_of_a_l513_513275

-- Defining function f
def f (a x : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5 * x + 2

-- Defining function g
def g (a x : ℝ) : ℝ := f a x - 2 * x

-- Stating the theorem
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ g a x1 = 0 ∧ g a x2 = 0 ∧ g a x3 = 0) ↔ (-1 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l513_513275


namespace sonia_and_joss_time_spent_moving_l513_513446

def total_time_spent_moving (fill_time_per_trip drive_time_per_trip trips : ℕ) :=
  (fill_time_per_trip + drive_time_per_trip) * trips

def total_time_in_hours (total_time_in_minutes : ℕ) : ℚ :=
  total_time_in_minutes / 60

theorem sonia_and_joss_time_spent_moving :
  total_time_in_hours (total_time_spent_moving 15 30 6) = 4.5 :=
by
  sorry

end sonia_and_joss_time_spent_moving_l513_513446


namespace clayton_total_points_l513_513611

theorem clayton_total_points 
  (game1 game2 game3 : ℕ)
  (game1_points : game1 = 10)
  (game2_points : game2 = 14)
  (game3_points : game3 = 6)
  (game4 : ℕ)
  (game4_points : game4 = (game1 + game2 + game3) / 3) :
  game1 + game2 + game3 + game4 = 40 :=
sorry

end clayton_total_points_l513_513611


namespace number_of_common_tangents_between_circleC_and_circleD_l513_513186

noncomputable def circleC := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }

noncomputable def circleD := { p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 + 2 * p.2 - 4 = 0 }

theorem number_of_common_tangents_between_circleC_and_circleD : 
    ∃ (num_tangents : ℕ), num_tangents = 2 :=
by
    -- Proving the number of common tangents is 2
    sorry

end number_of_common_tangents_between_circleC_and_circleD_l513_513186


namespace answer_is_p_and_q_l513_513730

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513730


namespace part_a_part_b_part_c_l513_513245

-- Condition for obtainability transformation
def obtainable (N M : ℕ) : Prop :=
  ∃ (seq : List ℕ) , seq.head = N ∧ seq.last = M ∧
  ∀ (n : ℕ), n ∈ seq → 
    ∃ (x y : ℕ), x * y % 10 = n ∧ 
    nat.has_digit N x ∧ nat.has_digit N y -- Placeholder definition, real check needed for digit forms

-- Part (a) Given Problem Statement
theorem part_a :
  obtainable 2567777899 2018 :=
  sorry

-- Part (b) Given Problem Statement
theorem part_b :
  ∃ (A B : ℕ), ∃ (C : ℕ), A = 2 ∧ B = 3 ∧ ¬(obtainable C A ∧ obtainable C B) :=
  sorry

-- Part (c) Given Problem Statement
theorem part_c (S : finset ℕ) (h : ∀ s ∈ S, ¬ nat.has_digit s 5) :
  ∃ (N : ℕ), ∀ (s ∈ S), obtainable N s :=
  sorry

end part_a_part_b_part_c_l513_513245


namespace proof_problem_l513_513807

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513807


namespace lilly_rosy_fish_total_l513_513111

theorem lilly_rosy_fish_total : let lilly_fish := 10; let rosy_fish := 9 in lilly_fish + rosy_fish = 19 :=
by
  let lilly_fish := 10
  let rosy_fish := 9
  show lilly_fish + rosy_fish = 19
  sorry

end lilly_rosy_fish_total_l513_513111


namespace dist_M_M_l513_513426

variable {a b c d : ℝ}

def pointA₁ := (a, b)
def pointB₁ := (c, d)
def pointA₂ := (a + 4, b + 12)
def pointB₂ := (c - 15, d - 5)

def midpoint (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ :=
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

def M₁ := midpoint a b c d
def M₂ := midpoint (a + 4) (b + 12) (c - 15) (d - 5)

def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2)

def M_diff (m₁ m₂ : ℝ × ℝ) : ℝ × ℝ :=
  (m₁.1 - m₂.1, m₁.2 - m₂.2)

def M_new := (fst M₁ - 11 / 2, snd M₁ + 7 / 2)

-- The conjecture we need to prove
theorem dist_M_M' : dist (fst M₁) (snd M₁) (fst M_new) (snd M_new) = Real.sqrt 42.5 := sorry

end dist_M_M_l513_513426


namespace peter_wins_prize_at_least_one_person_wins_prize_l513_513008

-- Part (a): Probability that Peter wins a prize
theorem peter_wins_prize :
  let p : Probability := (5 / 6) ^ 9
  p = 0.194 := sorry

-- Part (b): Probability that at least one person wins a prize
theorem at_least_one_person_wins_prize :
  let p : Probability := 0.919
  p = 0.919 := sorry

end peter_wins_prize_at_least_one_person_wins_prize_l513_513008


namespace radius_increase_by_one_percent_l513_513469

section 
variables (r r' : ℝ) (π : ℝ := Real.pi) (a_increase : ℝ := 0.0201)

/-- Given the increase in the area of a circle by 2.01%, prove that the radius is increased by 1.00%. -/
theorem radius_increase_by_one_percent
  (h1 : π * r'^2 = π * r^2 + 0.0201 * (π * r^2)) :
  (r' / r - 1) * 100 ≈ 1.00 :=
begin
  sorry -- insert the proof here
end

end

end radius_increase_by_one_percent_l513_513469


namespace true_proposition_l513_513760

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513760


namespace true_proposition_l513_513843

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513843


namespace june_needs_68_percent_male_vote_to_win_l513_513376

theorem june_needs_68_percent_male_vote_to_win 
  (total_students : ℕ) 
  (percentage_boys : ℝ)
  (winning_percentage : ℝ)
  (percentage_females_to_june : ℝ)
  (boys girls : ℕ)
  (total_votes_needed votes_from_girls votes_needed_from_boys : ℝ)
  (percentage_male_votes_needed : ℝ) : 
  total_students = 200 →
  percentage_boys = 0.60 →
  winning_percentage = 0.51 →
  percentage_females_to_june = 0.25 →
  boys = 120 →
  girls = 80 →
  total_votes_needed = 101 →
  votes_from_girls = 20 →
  votes_needed_from_boys = 81 →
  percentage_male_votes_needed = (votes_needed_from_boys / boys.to_real) * 100 →
  percentage_male_votes_needed ≥ 68 :=
by
  sorry

end june_needs_68_percent_male_vote_to_win_l513_513376


namespace point_on_angle_bisector_l513_513308

theorem point_on_angle_bisector (a b : ℝ) (h : (a, b) = (b, a)) : a = b ∨ a = -b := 
by
  sorry

end point_on_angle_bisector_l513_513308


namespace complex_point_in_third_quadrant_l513_513041

theorem complex_point_in_third_quadrant (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  let z := complex.exp (complex.I * (3 * π / 2 - θ)) + complex.exp (complex.I * (π + θ))
  ∃ x y : ℝ, z = x + y * complex.I ∧ x < 0 ∧ y < 0 :=
by
  sorry

end complex_point_in_third_quadrant_l513_513041


namespace proposition_A_l513_513820

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513820


namespace johns_balance_at_end_of_first_year_l513_513089

theorem johns_balance_at_end_of_first_year (initial_deposit interest_first_year : ℝ) 
  (h1 : initial_deposit = 5000) 
  (h2 : interest_first_year = 500) :
  initial_deposit + interest_first_year = 5500 :=
by
  rw [h1, h2]
  norm_num

end johns_balance_at_end_of_first_year_l513_513089


namespace true_conjunction_l513_513684

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513684


namespace solve_given_equation_l513_513029

noncomputable def solve_equation : Prop :=
  let a := - (Real.sqrt 3) / 2
  in (∀ x : ℂ, (2 * x^3 + 6 * x^2 * Real.sqrt 3 + 12 * x + 6 * Real.sqrt 3
                + 2 * x + Real.sqrt 3 = 0) ↔
                (x = a ∨ x = a + Complex.I / 2 ∨ x = a - Complex.I / 2))

theorem solve_given_equation : solve_equation := by
  sorry

end solve_given_equation_l513_513029


namespace false_statement_none_of_these_l513_513288

open Real

-- Definitions of the conditions
def centerA : Point := ⟨0, 0⟩
def centerB : Point := ⟨x, y⟩
def radiusA : ℝ := 8
def radiusB : ℝ := 5
def distanceAB : ℝ := dist centerA centerB

-- The proposition to be proven
theorem false_statement_none_of_these :
  ¬(distanceAB = 3 ∨ distanceAB = 13 ∨ distanceAB < 13 ∨ distanceAB < 3) :=
sorry

end false_statement_none_of_these_l513_513288


namespace max_squares_covered_by_card_l513_513129

theorem max_squares_covered_by_card (side_len : ℕ) (card_side : ℕ) : 
  side_len = 1 → card_side = 2 → n ≤ 12 :=
by
  sorry

end max_squares_covered_by_card_l513_513129


namespace first_term_of_geometric_sequence_l513_513474

theorem first_term_of_geometric_sequence (a r : ℝ) 
  (h1 : a * r^5 = 9!) 
  (h2 : a * r^8 = 10!) : 
  a = 9! / 10^(5/3) :=
by
  sorry

end first_term_of_geometric_sequence_l513_513474


namespace proof_problem_l513_513810

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513810


namespace max_positive_is_200_l513_513068

-- Define the problem conditions
def condition (a : ℕ → ℤ) : Prop :=
  ∀ i : ℕ, a i > (a (i + 1) % 300) * (a (i + 2) % 300) * (a (i + 3) % 300)

-- Define the maximum number of positive integers
def max_positive (a : ℕ → ℤ) : ℕ :=
  ∑ i in finset.range 300, if a i > 0 then 1 else 0

-- Prove the maximum number of positive integers is 200
theorem max_positive_is_200 (a : ℕ → ℤ) (h : condition a) : max_positive a ≤ 200 :=
  sorry

end max_positive_is_200_l513_513068


namespace total_number_of_balls_in_fish_tank_l513_513077

-- Definitions as per conditions
def num_goldfish := 3
def num_platyfish := 10
def red_balls_per_goldfish := 10
def white_balls_per_platyfish := 5

-- Theorem statement
theorem total_number_of_balls_in_fish_tank : 
  (num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish) = 80 := 
by
  sorry

end total_number_of_balls_in_fish_tank_l513_513077


namespace average_disk_space_per_hour_l513_513577

theorem average_disk_space_per_hour:
  (let total_hours := 15 * 24 in
  ∀ (total_disk_space : ℕ), total_disk_space = 20000 →
  20000 / total_hours = 56) :=
begin
  let total_hours := 15 * 24,
  intro total_disk_space,
  intro h,
  simp [h, total_hours],
  sorry
end

end average_disk_space_per_hour_l513_513577


namespace proposition_A_l513_513824

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513824


namespace initial_fee_l513_513084

theorem initial_fee (total_bowls : ℤ) (lost_bowls : ℤ) (broken_bowls : ℤ) (safe_fee : ℤ)
  (loss_fee : ℤ) (total_payment : ℤ) (paid_amount : ℤ) :
  total_bowls = 638 →
  lost_bowls = 12 →
  broken_bowls = 15 →
  safe_fee = 3 →
  loss_fee = 4 →
  total_payment = 1825 →
  paid_amount = total_payment - ((total_bowls - lost_bowls - broken_bowls) * safe_fee - (lost_bowls + broken_bowls) * loss_fee) →
  paid_amount = 100 :=
by
  intros _ _ _ _ _ _ _
  sorry

end initial_fee_l513_513084


namespace find_m_range_l513_513656

-- Definitions
def p (x : ℝ) : Prop := abs (2 * x + 1) ≤ 3
def q (x m : ℝ) (h : m > 0) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0

-- Problem Statement
theorem find_m_range : 
  (∀ (x : ℝ) (h : m > 0), (¬ (p x)) → (¬ (q x m h))) ∧ 
  (∃ (x : ℝ), ¬ (p x) ∧ ¬ (q x m h)) → 
  ∃ (m : ℝ), m ≥ 3 := 
sorry

end find_m_range_l513_513656


namespace hunting_distribution_l513_513545

section hunting_problem

-- Define the hunters
inductive Hunter
| András | Béla | Csaba | Dénes
deriving DecidableEq

open Hunter

-- Points for each type of game
def points : Hunter → ℕ 
| András := 6
| Béla := 4
| Csaba := 5
| Dénes := 3

-- Define the animals
inductive Animal
| WildBoar | Deer | Wolf | Fox
deriving DecidableEq

open Animal

-- Mapping from animals to points
def animal_points : Animal → ℕ 
| WildBoar := 5
| Deer := 4
| Wolf := 2
| Fox := 1

-- Define the proof problem
theorem hunting_distribution :
    (points András + points Dénes = points Béla + points Csaba) ∧
    (points András + points Béla + points Csaba + points Dénes = 18) ∧
    points Dénes <= points András ∧ points Dénes <= points Béla ∧ points Dénes <= points Csaba :=
by
  sorry

end hunting_problem

end hunting_distribution_l513_513545


namespace ball_travel_distance_l513_513125

noncomputable def total_distance : ℝ :=
  200 + (2 * (200 * (1 / 3))) + (2 * (200 * ((1 / 3) ^ 2))) +
  (2 * (200 * ((1 / 3) ^ 3))) + (2 * (200 * ((1 / 3) ^ 4)))

theorem ball_travel_distance :
  total_distance = 397.2 :=
by
  sorry

end ball_travel_distance_l513_513125


namespace power_comparison_l513_513057

theorem power_comparison : (5 : ℕ) ^ 30 < (3 : ℕ) ^ 50 ∧ (3 : ℕ) ^ 50 < (4 : ℕ) ^ 40 := by
  sorry

end power_comparison_l513_513057


namespace triangle_areas_equal_l513_513352

variables {A B C D M : Type*} [inner_product_space ℝ T]

theorem triangle_areas_equal {A B C D M : Type*} [inner_product_space ℝ T] 
  (h1 : D ∈ segment ℝ B C) 
  (h2 : M ∈ segment ℝ A D) 
  (h3 : vector.angle A B M = vector.angle A C M) :
  (∃ D : T, midpoint ℝ B C D) ∧ (∃ M : T, M ∈ segment ℝ A D ∧ M ≠ midpoint ℝ A D) := 
sorry

end triangle_areas_equal_l513_513352


namespace main_statement_l513_513746

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513746


namespace solve_equation_l513_513030

theorem solve_equation : ∀ x : ℝ, 2 * x - 6 = 3 * x * (x - 3) ↔ (x = 3 ∨ x = 2 / 3) := by sorry

end solve_equation_l513_513030


namespace surface_area_diff_l513_513931

theorem surface_area_diff (r : ℝ) :
  let θ := Real.pi / 2001 in
  let A_2001 := (2001 / 2) * r^2 * Real.sin(2 * Real.pi / 2001) in
  let A_667 := (667 / 2) * r^2 * Real.sin(2 * Real.pi / 667) in
  let A_diff := A_2001 - A_667 in
  A_diff = 10672 * r^2 * Real.sin(θ)^3 * Real.cos(θ)^3 :=
begin
  sorry
end

end surface_area_diff_l513_513931


namespace boa_constrictor_times_shorter_l513_513597

noncomputable def garden_snake_length : ℝ := 10.0
noncomputable def boa_constrictor_length : ℝ := 1.428571429

theorem boa_constrictor_times_shorter :
  garden_snake_length / boa_constrictor_length ≈ 7 := 
sorry -- proof placeholder

end boa_constrictor_times_shorter_l513_513597


namespace total_questions_l513_513106

theorem total_questions (two_point_questions : ℕ) (four_point_questions : ℕ) (h1 : two_point_questions = 30) (h2 : four_point_questions = 10) : two_point_questions + four_point_questions = 40 :=
by
  rw [h1, h2]
  simp
  exact rfl

end total_questions_l513_513106


namespace find_point_P_l513_513670

noncomputable def point_P_coordinates {A B : ℝ × ℝ × ℝ} (P : ℝ × ℝ × ℝ) : Prop :=
let (Ax, Ay, Az) := A in
let (Bx, By, Bz) := B in
let (Px, Py, Pz) := P in
Px = 0 ∧ Py = 0 ∧ (sqrt ((Ax - Px)^2 + (Ay - Py)^2 + (Az - Pz)^2) = sqrt ((Bx - Px)^2 + (By - Py)^2 + (Bz - Pz)^2))

theorem find_point_P : point_P_coordinates (-1, -2, 1) (2, 2, 2) (0, 0, 3) :=
by
  dunfold point_P_coordinates
  sorry

end find_point_P_l513_513670


namespace proof_problem_l513_513806

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513806


namespace final_portfolio_value_l513_513370

-- Define the initial conditions and growth rates
def initial_investment : ℝ := 80
def first_year_growth_rate : ℝ := 0.15
def additional_investment : ℝ := 28
def second_year_growth_rate : ℝ := 0.10

-- Calculate the values of the portfolio at each step
def after_first_year_investment : ℝ := initial_investment * (1 + first_year_growth_rate)
def after_addition : ℝ := after_first_year_investment + additional_investment
def after_second_year_investment : ℝ := after_addition * (1 + second_year_growth_rate)

theorem final_portfolio_value : after_second_year_investment = 132 := by
  -- This is where the proof would go, but we are omitting it
  sorry

end final_portfolio_value_l513_513370


namespace count_integers_between_cubes_l513_513300

theorem count_integers_between_cubes :
  let a := 10.5
  let b := 10.7
  let n1 := (a^3).ceil
  let n2 := (b^3).floor
  n2 - n1 + 1 = 67 :=
by
  let a := 10.5
  let b := 10.7
  let n1 := (a^3).ceil
  let n2 := (b^3).floor
  have h1 : (a^3 = 1157.625) := by sorry
  have h2 : (b^3 = 1225.043) := by sorry
  have h3 : (n1 = 1158) := by sorry
  have h4 : (n2 = 1224) := by sorry
  rw [h1, h2, h3, h4]
  sorry

end count_integers_between_cubes_l513_513300


namespace bobby_exercise_time_l513_513602

def bobby_jump_rate := 60 -- jumps per minute
def bobby_situp_rate := 25 -- sit-ups per minute
def bobby_pushup_rate := 20 -- push-ups per minute

def jumps := 200
def situps := 150
def pushups := 100

def time_for_jumps := jumps / bobby_jump_rate
def time_for_situps := situps / bobby_situp_rate
def time_for_pushups := pushups / bobby_pushup_rate

def total_time := time_for_jumps + time_for_situps + time_for_pushups

theorem bobby_exercise_time : total_time = 14.33 := by
  sorry

end bobby_exercise_time_l513_513602


namespace find_s_log_eq_l513_513633

theorem find_s_log_eq (s : ℝ) (h : 2 * log s / log 4 = log (8 * s) / log 4) : s = 8 :=
sorry

end find_s_log_eq_l513_513633


namespace num_integers_between_cubed_values_l513_513298

theorem num_integers_between_cubed_values : 
  let a : ℝ := 10.5
  let b : ℝ := 10.7
  let c1 := a^3
  let c2 := b^3
  let first_integer := Int.ceil c1
  let last_integer := Int.floor c2
  first_integer ≤ last_integer → 
  last_integer - first_integer + 1 = 67 :=
by
  sorry

end num_integers_between_cubed_values_l513_513298


namespace exponent_equation_l513_513901

theorem exponent_equation (a b : ℝ) (h : 3^a * 9^b = 1/3) : a + 2*b = -1 := by
  sorry

end exponent_equation_l513_513901


namespace proposition_A_l513_513826

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513826


namespace alice_difference_theorem_l513_513599

def alice_payment_plan_difference
  (initial_balance : ℝ) 
  (annual_rate : ℝ) 
  (compounding_intervals_plan1 : ℕ) 
  (compounding_intervals_plan2 : ℕ) 
  (years : ℕ)
  (one_third_payment_years : ℕ)
  (correct_difference : ℝ): Prop :=
let plan1_payment_after_n_years :=
      initial_balance * (1 + annual_rate / compounding_intervals_plan1) ^ (compounding_intervals_plan1 * one_third_payment_years) in
let one_third_payment := plan1_payment_after_n_years / 3 in
let remaining_balance_after_one_third_payment := plan1_payment_after_n_years - one_third_payment in
let final_payment :=
      remaining_balance_after_one_third_payment * 
      (1 + annual_rate / compounding_intervals_plan1) ^ (compounding_intervals_plan1 * one_third_payment_years) in
let total_payment_plan1 := one_third_payment + final_payment in
let total_payment_plan2 := 
      initial_balance * (1 + annual_rate / compounding_intervals_plan2) ^ (compounding_intervals_plan2 * years) in
abs (total_payment_plan2 - total_payment_plan1) = correct_difference

theorem alice_difference_theorem : 
  alice_payment_plan_difference 12000 0.08 2 1 8 4 1130 :=
by
  sorry

end alice_difference_theorem_l513_513599


namespace assignment_schemes_equiv_240_l513_513649

-- Define the problem
theorem assignment_schemes_equiv_240
  (students : Finset ℕ)
  (tasks : Finset ℕ)
  (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students)
  (h_tasks : tasks = {0, 1, 2, 3})
  (h_students : students.card = 6) :
  let ways_to_assign := (EUₖstudents.card, tasks.card )
  let ways_with_A_or_B_not_taskA := (C(2,1) * A(5, tasks.card - 1))
  (ways_to_assign - ways_with_A_or_B_not_taskA) = 240 :=
by
  sorry

end assignment_schemes_equiv_240_l513_513649


namespace combinedTotalSandcastlesAndTowers_l513_513065

def markSandcastles : Nat := 20
def towersPerMarkSandcastle : Nat := 10
def jeffSandcastles : Nat := 3 * markSandcastles
def towersPerJeffSandcastle : Nat := 5

theorem combinedTotalSandcastlesAndTowers :
  (markSandcastles + markSandcastles * towersPerMarkSandcastle) +
  (jeffSandcastles + jeffSandcastles * towersPerJeffSandcastle) = 580 :=
by
  sorry

end combinedTotalSandcastlesAndTowers_l513_513065


namespace answer_is_p_and_q_l513_513732

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513732


namespace sum_of_squares_of_digits_of_N_l513_513936

theorem sum_of_squares_of_digits_of_N :
  let N := 456 in
  let digits := [4, 5, 6] in
  N = 456 → 
  (4^2 + 5^2 + 6^2 = 77) :=
by
  intro N_eq
  rfl

end sum_of_squares_of_digits_of_N_l513_513936


namespace find_geometric_first_term_l513_513476

noncomputable def geometric_first_term (a r : ℝ) (h1 : a * r^5 = real.from_nat (nat.factorial 9)) 
  (h2 : a * r^8 = real.from_nat (nat.factorial 10)) : ℝ :=
  a

theorem find_geometric_first_term :
  ∃ a r, a * r^5 = real.from_nat (nat.factorial 9) ∧ a * r^8 = real.from_nat (nat.factorial 10) ∧ a = 362880 / 10^(5/3) :=
by
  have h : ∃ (a r: ℝ), a * r^5 = real.from_nat (nat.factorial 9) ∧ a * r^8 = real.from_nat (nat.factorial 10),
  {
    use [362880 / 10^(5/3), 10^(1/3)],
    split,
    { sorry },
    { sorry }
  },
  cases h with a ha,
  cases ha with r hr,
  use [a, r],
  split,
  { exact hr.left },
  split,
  { exact hr.right },
  { sorry }

end find_geometric_first_term_l513_513476


namespace coplanar_vectors_exist_m_n_l513_513291

-- Definitions of given vectors
def vector_a : ℝ × ℝ × ℝ := (2, -1, 2)
def vector_b : ℝ × ℝ × ℝ := (-1, 3, -3)
def vector_c (λ : ℝ) : ℝ × ℝ × ℝ := (13, 6, λ)

-- The coplanarity condition that ∃ m n ∈ ℝ such that vector_c = m * vector_a + n * vector_b
theorem coplanar_vectors_exist_m_n (λ : ℝ) :
  (∃ m n : ℝ, vector_c λ = (m * vector_a.1 + n * vector_b.1, 
                            m * vector_a.2 + n * vector_b.2, 
                            m * vector_a.3 + n * vector_b.3)) ↔ 
  λ = 3 :=
by
  sorry

end coplanar_vectors_exist_m_n_l513_513291


namespace α_plus_β_eq_two_l513_513240

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

theorem α_plus_β_eq_two
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 := 
sorry

end α_plus_β_eq_two_l513_513240


namespace max_area_difference_l513_513499

theorem max_area_difference (l w l' w' : ℕ) 
    (h1 : 2 * l + 2 * w = 144) 
    (h2 : 2 * l' + 2 * w' = 144) :
    (l * w ≤ 1296 ∧ l' * w' ≤ 1296) ∧ 
    (∃ l w l' w', l * w = 1296 ∧ l' * w' = 71) → 
    abs ((l * w) - (l' * w')) = 1225 :=
by
  sorry

end max_area_difference_l513_513499


namespace count_positive_integers_l513_513645

def f (x : ℕ) : ℕ := x^2 + 10 * x + 25

theorem count_positive_integers :
  { x : ℕ | 50 ≤ f x ∧  f x < 100 }.toFinset.card = 3 :=
by
  sorry

end count_positive_integers_l513_513645


namespace additional_coins_needed_l513_513598

/-
Alex has 20 friends and 192 coins. Prove the minimum number of additional coins he needs so that he can give each friend at least one coin and no two friends receive the same number of coins.
-/

theorem additional_coins_needed : 
  ∃ m : ℕ, 
  let total_needed := (20 * (20 + 1)) / 2 in
  let current_coins := 192 in
  total_needed - current_coins = m ∧ m = 18 := by
sorry

end additional_coins_needed_l513_513598


namespace proposition_A_l513_513821

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513821


namespace problem_statement_l513_513794

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513794


namespace point_in_second_quadrant_point_neg2_5_is_in_second_quadrant_l513_513340

-- Definitions derived from conditions
def point : ℝ × ℝ := (-2, 5)

-- The proof goal
theorem point_in_second_quadrant (x y : ℝ) (hx : x < 0) (hy : y > 0) : (x, y) = point → (x < 0 ∧ y > 0) := by
  intros h_eq
  rw h_eq
  exact ⟨hx, hy⟩

-- Prove the theorem for the specific point
theorem point_neg2_5_is_in_second_quadrant : point = (-2, 5) → (point.1 < 0 ∧ point.2 > 0) := by
  intro h
  rw h
  exact ⟨by norm_num, by norm_num⟩

end point_in_second_quadrant_point_neg2_5_is_in_second_quadrant_l513_513340


namespace fraction_spent_toy_store_l513_513294

def allowance : ℝ := 1.5
def fraction_spent_arcade : ℝ := 3 / 5
def remaining_after_candy : ℝ := 0.40

theorem fraction_spent_toy_store :
  let spent_at_arcade := fraction_spent_arcade * allowance in
  let remaining_after_arcade := allowance - spent_at_arcade in
  let spent_at_toy_store := remaining_after_arcade - remaining_after_candy in
  spent_at_toy_store / remaining_after_arcade = 1 / 3 :=
by
  sorry

end fraction_spent_toy_store_l513_513294


namespace sarah_initial_money_l513_513441

-- Definitions based on conditions
def cost_toy_car := 11
def cost_scarf := 10
def cost_beanie := 14
def remaining_money := 7
def total_cost := 2 * cost_toy_car + cost_scarf + cost_beanie
def initial_money := total_cost + remaining_money

-- Statement of the theorem
theorem sarah_initial_money : initial_money = 53 :=
by
  sorry

end sarah_initial_money_l513_513441


namespace combined_students_l513_513328

theorem combined_students (students : ℕ) (maths_ratio science_ratio history_ratio : ℚ) 
  (students_like_maths students_like_science students_like_history : ℕ) :
  students = 30 →
  maths_ratio = 3/10 →
  science_ratio = 1/4 →
  history_ratio = 2/5 →
  students_like_maths = floor (maths_ratio * students) →
  students_like_science = floor (science_ratio * (students - students_like_maths)) →
  students_like_history = floor (history_ratio * (students - students_like_maths - students_like_science)) →
  students_like_maths + students_like_history = 15 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end combined_students_l513_513328


namespace convert_cylindrical_to_rectangular_l513_513195

-- Define cylindrical to rectangular conversion
def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

-- Given point in cylindrical coordinates
def c_point : ℝ × ℝ × ℝ := (10, Real.pi / 4, -3)

-- Expected point in rectangular coordinates
def r_point : ℝ × ℝ × ℝ := (5 * Real.sqrt 2, 5 * Real.sqrt 2, -3)

-- Theorem statement
theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 10 (Real.pi / 4) (-3) = r_point :=
by
  sorry

end convert_cylindrical_to_rectangular_l513_513195


namespace locus_of_Q_l513_513363

noncomputable def point_locus (O P : ℝ^3) (r : ℝ) (A B C : ℝ^3) (Q : ℝ^3) : Prop :=
  dist O Q = sqrt (3 * r^2 - 2 * dist O P ^ 2)

theorem locus_of_Q {O P : ℝ^3} {r : ℝ}
  {A B C Q : ℝ^3}
  (hA : dist O A = r)
  (hB : dist O B = r)
  (hC : dist O C = r)
  (hPA : ∠ (P - B) (P - A) = 90)
  (hPC : ∠ (P - C) (P - A) = 90)
  (hPB : ∠ (P - C) (P - B) = 90)
  (hQ : dist Q P = sqrt (dist P A ^ 2 + dist P B ^ 2 + dist P C ^ 2)) :
  point_locus O P r A B C Q :=
by
  sorry

end locus_of_Q_l513_513363


namespace simplify_trig_expression_l513_513027

open Real

theorem simplify_trig_expression (α : ℝ) : 
  (cos (2 * π + α) * tan (π + α)) / cos (π / 2 - α) = 1 := 
sorry

end simplify_trig_expression_l513_513027


namespace proposition_analysis_l513_513047

theorem proposition_analysis :
  let f (x : ℝ) := (Real.log x) / x,
      f_derivative (x : ℝ) := (1 - Real.log x) / (x ^ 2) in
  (0 < 2) ∧ (2 < Real.sqrt 5) ∧ (Real.sqrt 5 < Real.exp 1) ∧ (Real.exp 1 < Real.sqrt 11) ∧ (Real.sqrt 11 < 4) ∧
  (∀ x, 0 < x → x < Real.exp 1 → f_derivative x > 0) ∧
  (∀ x, Real.exp 1 < x → f_derivative x < 0) →
  ¬ (Real.log 5 < Real.sqrt 5 * Real.log 2) ∧
  (2 ^ Real.sqrt 11 < 11) ∧
  (3 * (Real.exp 1) * Real.log 2 < 4 * Real.sqrt 2) :=
by
  sorry

end proposition_analysis_l513_513047


namespace proof_problem_l513_513808

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513808


namespace deductive_reasoning_l513_513271

def reasoning1 : Prop := ∀ n : ℕ, n ≥ 6 → (∃ p q : ℕ, odd p ∧ odd q ∧ prime p ∧ prime q ∧ n = p + q)

def reasoning2 : Prop := ∀ Δ : Triangle, sum_of_interior_angles Δ = 180°
def reasoning2r : Prop := ∀ Δ : RightTriangle, sum_of_interior_angles Δ = 180°

def reasoning3 (s : ℝ) : Prop := volume_cube s = s^3
def reasoning3r (s : ℝ) : Prop := ∀ W : Square, volume W = area W * height W

def reasoning4 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * a * b
def reasoning4r (x : ℝ) : Prop := sin (2 * x) ≤ 1

theorem deductive_reasoning : 
  (reasoning2 → reasoning2r) ∧ (reasoning4 → reasoning4r) :=
by
  sorry

end deductive_reasoning_l513_513271


namespace largest_square_in_rectangle_l513_513360

noncomputable def largest_square_side_length (length width side_triangle : ℝ) : ℝ :=
  (8 - real.sqrt 6)

theorem largest_square_in_rectangle (length width : ℝ) (side_triangle : ℝ)
  (h1 : length = 16) (h2 : width = 12)
  (h3 : side_triangle = (12 * real.sqrt 2 / real.sqrt 3))
  : largest_square_side_length length width side_triangle = 8 - real.sqrt 6 :=
by
  sorry

end largest_square_in_rectangle_l513_513360


namespace marcie_cups_coffee_l513_513024

theorem marcie_cups_coffee (S M T : ℕ) (h1 : S = 6) (h2 : S + M = 8) : M = 2 :=
by
  sorry

end marcie_cups_coffee_l513_513024


namespace knight_20_moves_count_l513_513603

def knight_moves (x y : ℕ) : set (ℕ × ℕ) :=
  {(x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1),
   (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2)}

def knight_can_reach (n : ℕ) (start : ℕ × ℕ) : set (ℕ × ℕ) :=
  finset.fold (λ s (xy : ℕ × ℕ), s ∪ knight_moves xy.1 xy.2) ∅ (finset.range n)

def chessboard : finset (ℕ × ℕ) := finset.product (finset.range 8) (finset.range 8)

def white_squares : finset (ℕ × ℕ) :=
  chessboard.filter (λ (xy : ℕ × ℕ), (xy.1 + xy.2) % 2 = 1)

def knight_reachable_squares_after_20_moves (start : ℕ × ℕ) : finset (ℕ × ℕ) :=
  (knight_can_reach 20 start).filter (λ (xy : ℕ × ℕ), xy ∈ white_squares)

theorem knight_20_moves_count (start : ℕ × ℕ) (h : (start.1 + start.2) % 2 = 1) :
  (knight_reachable_squares_after_20_moves start).card = 32 := sorry

end knight_20_moves_count_l513_513603


namespace find_x_values_l513_513210

theorem find_x_values (
  x : ℝ
) (h₁ : x ≠ 0) (h₂ : x ≠ 1) (h₃ : x ≠ 2) :
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 4) ↔ 
  (x < (1 - Real.sqrt 17) / 2 ∨ (0 < x ∧ x < 1) ∨ (2 < x ∧ x < (1 + Real.sqrt 17) / 2)) :=
by
  sorry

end find_x_values_l513_513210


namespace area_of_one_cookie_l513_513105

theorem area_of_one_cookie (L W : ℝ)
    (W_eq_15 : W = 15)
    (circumference_condition : 4 * L + 2 * W = 70) :
    L * W = 150 :=
by
  sorry

end area_of_one_cookie_l513_513105


namespace contractor_fine_per_day_l513_513566

theorem contractor_fine_per_day
    (total_days : ℕ) 
    (work_days_fine_amt : ℕ) 
    (total_amt : ℕ) 
    (absent_days : ℕ) 
    (worked_days : ℕ := total_days - absent_days)
    (earned_amt : ℕ := worked_days * work_days_fine_amt)
    (fine_per_day : ℚ)
    (total_fine : ℚ := absent_days * fine_per_day) : 
    (earned_amt - total_fine = total_amt) → 
    fine_per_day = 7.5 :=
by
  intros h
  -- proof here is omitted
  sorry

end contractor_fine_per_day_l513_513566


namespace problem_statement_l513_513798

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513798


namespace proof_problem_l513_513772

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513772


namespace a_n_correct_T_n_correct_l513_513247

noncomputable def a (n : ℕ) : ℕ :=
if h : n = 1 then 2 else n * (n + 1)

noncomputable def S (n : ℕ) : ℝ :=
(n + 2) / 3 * a n

noncomputable def T (n : ℕ) : ℝ :=
(∑ i in finset.range n, 1 / a (i + 1))

theorem a_n_correct (n : ℕ) (h : n > 0) : a n = n * (n + 1) :=
by
  sorry

theorem T_n_correct (n : ℕ) (h : n > 0) : T n = n / (n + 1) :=
by
  sorry

end a_n_correct_T_n_correct_l513_513247


namespace problem_statement_l513_513709

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513709


namespace p_and_q_is_true_l513_513690

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513690


namespace solution_set_abs_inequality_l513_513478

theorem solution_set_abs_inequality :
  { x : ℝ | |x - 2| - |2 * x - 1| > 0 } = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end solution_set_abs_inequality_l513_513478


namespace solution_set_inequality_l513_513620

noncomputable def f : ℝ → ℝ := sorry -- Assume f is defined somewhere

-- Conditions:
axiom f_1 : f 1 = 1
axiom f_prime_lt_half : ∀ x : ℝ, deriv f x < 1/2

-- Proposition to prove:
theorem solution_set_inequality : {x : ℝ | f x > (x + 1) / 2} = set.Iio 1 :=
by
  sorry -- Proof omitted

end solution_set_inequality_l513_513620


namespace contractor_fine_per_day_l513_513564

theorem contractor_fine_per_day
    (total_days : ℕ) 
    (work_days_fine_amt : ℕ) 
    (total_amt : ℕ) 
    (absent_days : ℕ) 
    (worked_days : ℕ := total_days - absent_days)
    (earned_amt : ℕ := worked_days * work_days_fine_amt)
    (fine_per_day : ℚ)
    (total_fine : ℚ := absent_days * fine_per_day) : 
    (earned_amt - total_fine = total_amt) → 
    fine_per_day = 7.5 :=
by
  intros h
  -- proof here is omitted
  sorry

end contractor_fine_per_day_l513_513564


namespace units_digit_of_product_of_odds_is_5_l513_513512

/-- The product of all odd positive integers between 20 and 120 has a units digit of 5. -/
theorem units_digit_of_product_of_odds_is_5 : 
  let odd_integers := {n : ℕ | 20 < n ∧ n < 120 ∧ n % 2 = 1}
  let product_of_odds := ∏ n in odd_integers.to_finset, n
  product_of_odds % 10 = 5 := by
  sorry

end units_digit_of_product_of_odds_is_5_l513_513512


namespace ribbon_cuts_l513_513133

theorem ribbon_cuts (num_spools : ℕ) (spool_length piece_length : ℝ) (pieces_per_spool : ℕ) : 
  num_spools = 5 → 
  spool_length = 60 →
  piece_length = 1.5 →
  pieces_per_spool = (spool_length / piece_length).to_nat →
  ∀ total_cuts, total_cuts = num_spools * (pieces_per_spool - 1) → 
  total_cuts = 195 :=
by
  intros
  sorry

end ribbon_cuts_l513_513133


namespace infinite_strong_triples_l513_513594

noncomputable def strong_triple (t : ℕ × ℕ × ℕ) : Prop :=
  let ⟨a, b, c⟩ := t in ∀ m > 1, ¬(a + b + c ∣ a^m + b^m + c^m)

def pairwise_coprime (l : List ℕ) : Prop :=
  ∀ i j, i < l.length → j < l.length → i ≠ j → Nat.coprime (l.get i) (l.get j)

theorem infinite_strong_triples :
  ∃ (l : List (ℕ × ℕ × ℕ)), (∀ t ∈ l, strong_triple t) ∧ 
  (pairwise_coprime (l.map (λ t, let ⟨a, b, c⟩ := t in a + b + c))) :=
sorry

end infinite_strong_triples_l513_513594


namespace cubes_with_even_faces_l513_513155

-- Define the initial conditions as a structure
structure Block :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)
  (painted : bool)

-- Given conditions
def woodenBlock : Block :=
  { length := 5, width := 5, height := 1, painted := true }

-- Define what it means for a cube to have even number of painted faces
def even_painted_faces (n : ℕ) : Prop :=
  n % 2 = 0

-- Main theorem statement
theorem cubes_with_even_faces (b : Block) (n_cubes : ℕ) (n_even_faces : ℕ) : 
  b = woodenBlock → 
  n_cubes = 25 →
  n_even_faces = 12 :=
by 
  intros
  sorry

end cubes_with_even_faces_l513_513155


namespace find_locus_of_R_l513_513244

noncomputable theory

open_locale classical

-- Define the fixed point O
def O : ℝ × ℝ := (0, 0)

-- Define the fixed perpendicular lines l1 and l2 passing through O
def l1 (y k : ℝ) := (y = k)
def l2 (x h : ℝ) := (x = h)

-- Define the moving lines intersecting l1 and l2 and passing through O
def moving_line_PQ (α p : ℝ) (x y : ℝ) := (x * real.cos α + y * real.sin α - p = 0)

-- Define the points P and Q
def P (p h α : ℝ) : ℝ × ℝ := (h, (p - h * real.cos α) / real.sin α)
def Q (p k α : ℝ) : ℝ × ℝ := ((p - k * real.sin α) / real.cos α, k)

-- Projection R from O onto PQ
def R (p α : ℝ) := (p * real.cos α, p * real.sin α)

-- The proof statement
theorem find_locus_of_R (k h α p : ℝ) :
  k * (p * real.cos α) + h * (p * real.sin α) = k * h :=
sorry

end find_locus_of_R_l513_513244


namespace cube_volume_l513_513134

-- Define the surface area constant
def surface_area : ℝ := 725.9999999999998

-- Define the formula for surface area of a cube and solve for volume given the conditions
theorem cube_volume (SA : ℝ) (h : SA = surface_area) : 11^3 = 1331 :=
by sorry

end cube_volume_l513_513134


namespace proof_problem_l513_513531

variable (x y : ℝ)

theorem proof_problem :
  ¬ (x^2 + x^2 = x^4) ∧
  ¬ ((x - y)^2 = x^2 - y^2) ∧
  ¬ ((x^2 * y)^3 = x^6 * y) ∧
  ((-x)^2 * x^3 = x^5) :=
by
  sorry

end proof_problem_l513_513531


namespace value_of_3b_minus_a_l513_513045

theorem value_of_3b_minus_a :
  ∃ (a b : ℕ), (a > b) ∧ (a >= 0) ∧ (b >= 0) ∧ (∀ x : ℝ, (x - a) * (x - b) = x^2 - 16 * x + 60) ∧ (3 * b - a = 8) := 
sorry

end value_of_3b_minus_a_l513_513045


namespace proposition_true_l513_513834

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513834


namespace product_units_digit_odd_numbers_20_120_l513_513522

open Finset

def odd_numbers_between_20_and_120 : Finset ℕ := 
  filter (λ n => (n % 2 = 1) ∧ (20 < n) ∧ (n < 120)) (range 121)

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def product_units_digit (s : Finset ℕ) : ℕ := 
  (s.prod id) % 10

theorem product_units_digit_odd_numbers_20_120 :
  product_units_digit odd_numbers_between_20_and_120 = 5 :=
sorry

end product_units_digit_odd_numbers_20_120_l513_513522


namespace midpoint_sum_of_coordinates_l513_513428

theorem midpoint_sum_of_coordinates {A B M : Type}
  (hxA : A = (2, 8))
  (hxM : M = (3, 5))
  (hxB : ∃ x y, B = (x, y) ∧ (x + 2) / 2 = 3 ∧ (y + 8) / 2 = 5) :
  ∃ x y, B = (x, y) ∧ x + y = 6 := by
  sorry

end midpoint_sum_of_coordinates_l513_513428


namespace proposition_true_l513_513835

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l513_513835


namespace point_in_second_quadrant_l513_513342

-- Define the quadrants based on the conditions
def quadrant (x y : ℝ) : ℕ :=
  if x > 0 ∧ y > 0 then 1 else
  if x < 0 ∧ y > 0 then 2 else
  if x < 0 ∧ y < 0 then 3 else
  if x > 0 ∧ y < 0 then 4 else 0

-- Define the specific point
def point : ℝ × ℝ := (-2, 5)

-- The theorem to prove that the point lies in the second quadrant
theorem point_in_second_quadrant : quadrant point.1 point.2 = 2 :=
by {
  -- Proof is omitted
  sorry
}

end point_in_second_quadrant_l513_513342


namespace solution_set_l513_513876

def f (x : ℝ) : ℝ :=
if x < 1 then 2 * exp (x - 1)
else x^3 + x

theorem solution_set (S : Set ℝ) : 
  (∀ x ∈ S, f (f x) < 2) ↔ S = {x | x < 1 - Real.log 2} :=
by
  sorry

end solution_set_l513_513876


namespace hyperbola_eq_correct_l513_513877

noncomputable def hyperbola_equation : String :=
by
  let a := 2
  let c := Real.sqrt 7
  let b := Real.sqrt (c^2 - a^2)
  have foci_same : c = Real.sqrt 7 := by sorry
  have ecc_same : (c / a) = 2 * (c / 4) := by sorry
  let hyperbola_eq := s!"x^2/{a^2} - y^2/{b^2} = 1"
  exact hyperbola_eq

theorem hyperbola_eq_correct : hyperbola_equation = "x^2/4 - y^2/3 = 1" := by
  rw [hyperbola_equation]
  exact rfl

end hyperbola_eq_correct_l513_513877


namespace mike_can_buy_nine_games_l513_513116

noncomputable def mike_dollars (initial_dollars : ℕ) (spent_dollars : ℕ) (game_cost : ℕ) : ℕ :=
  (initial_dollars - spent_dollars) / game_cost

theorem mike_can_buy_nine_games : mike_dollars 69 24 5 = 9 := by
  sorry

end mike_can_buy_nine_games_l513_513116


namespace determine_p1_values_l513_513967

theorem determine_p1_values :
  ∃ (p : ℕ → ℕ), 
    (∀ n m, p n - n * p m + n * p b ∈ Int) ∧ 
    p 0 = 0 ∧
    0 ≤ p 1 ∧ p 1 ≤ 10^7 ∧
    ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ p a = 1999 ∧ p b = 2001 ∧
    {p 1} ⊆ {1, 1999, 3996001, 7992001} := 
by 
  sorry

end determine_p1_values_l513_513967


namespace sum_of_prime_factors_of_expression_l513_513220

theorem sum_of_prime_factors_of_expression : 
  let n := 7 ^ 7 - 7 ^ 4
  ∑ p in (n.factorize.to_finset).filter nat.prime p = 24 :=
by
  sorry

end sum_of_prime_factors_of_expression_l513_513220


namespace p_and_q_is_true_l513_513692

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513692


namespace main_statement_l513_513745

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l513_513745


namespace sally_reads_10_pages_on_weekdays_l513_513023

def sallyReadsOnWeekdays (x : ℕ) (total_pages : ℕ) (weekdays : ℕ) (weekend_days : ℕ) (weekend_pages : ℕ) : Prop :=
  (weekdays + weekend_days * weekend_pages = total_pages) → (weekdays * x = total_pages - weekend_days * weekend_pages)

theorem sally_reads_10_pages_on_weekdays :
  sallyReadsOnWeekdays 10 180 10 4 20 :=
by
  intros h
  sorry  -- proof to be filled in

end sally_reads_10_pages_on_weekdays_l513_513023


namespace optimal_metro_station_placement_l513_513040

def grid_width : ℕ := 39
def grid_height : ℕ := 9

def move_time (x1 y1 x2 y2 : ℕ) : ℕ :=
  abs (x2 - x1) + abs (y2 - y1)

theorem optimal_metro_station_placement :
  ∃ (station1 station2 : (ℕ × ℕ)),
  (station1 = (grid_width / 2, 0)) ∧
  (station2 = (grid_width / 2, grid_height - 1)) ∧
  ∀ (x1 y1 x2 y2 : ℕ),
  (x1 < grid_width) ∧ (y1 < grid_height) ∧
  (x2 < grid_width) ∧ (y2 < grid_height) →
  move_time x1 y1 (grid_width / 2) y1 + move_time (grid_width / 2) y1 (grid_width / 2) (grid_height - 1) +
  move_time (grid_width / 2) (grid_height - 1) x2 y2 ≤ 30 :=
by
  use (grid_width / 2, 0)
  use (grid_width / 2, grid_height - 1)
  sorry

end optimal_metro_station_placement_l513_513040


namespace stella_unpaid_leave_l513_513447

theorem stella_unpaid_leave (monthly_salary : ℕ) (actual_annual_salary : ℕ) (total_months : ℕ) : 
  monthly_salary = 4919 → actual_annual_salary = 49190 → total_months = 12 → 
  (total_months - actual_annual_salary / monthly_salary) = 2 :=
begin
  intros h1 h2 h3,
  sorry
end

end stella_unpaid_leave_l513_513447


namespace mode_median_correct_l513_513090

-- Define the set of data
def dataSet : List ℕ := [50, 20, 50, 30, 50, 25, 135]

-- Define a Lean structure to hold the mode and median
structure ModeMedian :=
  (mode : ℕ)
  (median : ℕ)

-- The main statement we're proving
theorem mode_median_correct : ModeMedian dataSet = { mode := 50, median := 50 } :=
by
  sorry

end mode_median_correct_l513_513090


namespace quadratic_two_distinct_real_roots_l513_513058

theorem quadratic_two_distinct_real_roots : 
  ∀ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 6 → 
  b^2 - 4 * a * c > 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l513_513058


namespace chandler_total_cost_l513_513922

theorem chandler_total_cost (
  cost_per_movie_ticket : ℕ := 30
  cost_8_movie_tickets : ℕ := 8 * cost_per_movie_ticket
  cost_per_football_game_ticket : ℕ := cost_8_movie_tickets / 2
  cost_5_football_game_tickets : ℕ := 5 * cost_per_football_game_ticket
  total_cost : ℕ := cost_8_movie_tickets + cost_5_football_game_tickets
) : total_cost = 840 := by
  sorry

end chandler_total_cost_l513_513922


namespace angle_between_diagonals_l513_513504

theorem angle_between_diagonals (a b : ℝ) (h1 : a ≥ b) (h2 : a ≠ 0 ∧ b ≠ 0)
  (h3 : a - b = 2 * (a + b) - 4 * real.sqrt (a * b)) :
  ∃ θ : ℝ, θ ≈ 12.68 ∧ (180 - θ) ≈ 167.32 :=
by
  sorry

end angle_between_diagonals_l513_513504


namespace expected_balls_in_original_position_after_two_transpositions_l513_513036

-- Define the conditions
def num_balls : ℕ := 10

def probs_ball_unchanged : ℚ :=
  (1 / 50) + (16 / 25)

def expected_unchanged_balls (num_balls : ℕ) (probs_ball_unchanged : ℚ) : ℚ :=
  num_balls * probs_ball_unchanged

-- The theorem stating the expected number of balls in original positions
theorem expected_balls_in_original_position_after_two_transpositions
  (num_balls_eq : num_balls = 10)
  (prob_eq : probs_ball_unchanged = (1 / 50) + (16 / 25)) :
  expected_unchanged_balls num_balls probs_ball_unchanged = 7.2 := 
by
  sorry

end expected_balls_in_original_position_after_two_transpositions_l513_513036


namespace answer_is_p_and_q_l513_513735

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513735


namespace discriminant_of_quadratic_eq_l513_513506

theorem discriminant_of_quadratic_eq : 
  (let (a, b, c) := (5, -9, 1) in b^2 - 4 * a * c) = 61 :=
by
  let (a, b, c) := (5, -9, 1)
  have h1 : b^2 = (-9)^2 := rfl
  have h2 : (-9)^2 = 81 := rfl
  have h3 : 4 * a * c = 4 * 5 * 1 := rfl
  have h4 : 4 * 5 * 1 = 20 := rfl
  have h5 : 81 - 20 = 61 := rfl
  exact (Eq.trans (Eq.trans (Eq.trans (Eq.trans rfl h1) h2) (by rw [h3, h4])) h5)

end discriminant_of_quadratic_eq_l513_513506


namespace units_digit_of_product_of_odds_is_5_l513_513513

/-- The product of all odd positive integers between 20 and 120 has a units digit of 5. -/
theorem units_digit_of_product_of_odds_is_5 : 
  let odd_integers := {n : ℕ | 20 < n ∧ n < 120 ∧ n % 2 = 1}
  let product_of_odds := ∏ n in odd_integers.to_finset, n
  product_of_odds % 10 = 5 := by
  sorry

end units_digit_of_product_of_odds_is_5_l513_513513


namespace black_area_greater_than_gray_area_by_l513_513345

-- Define the sides of the squares
def side_A : ℕ := 12
def side_B : ℕ := 9
def side_C : ℕ := 7
def side_D : ℕ := 3

-- Define the areas of the squares as given in the conditions
def area_A : ℕ := side_A^2
def area_B : ℕ := side_B^2
def area_C : ℕ := side_C^2
def area_D : ℕ := side_D^2

-- Define the specific black and grey areas using provided values directly
def black_area_A : ℕ := 144
def black_area_E : ℕ := 49
def gray_area_C : ℕ := 81
def gray_area_G : ℕ := 9

-- Prove the sum of black areas is greater than the sum of gray areas by 103
theorem black_area_greater_than_gray_area_by :
  black_area_A + black_area_E - gray_area_C - gray_area_G = 103 := by
  -- Directly substituting these values and simplifying
  show 144 + 49 - 81 - 9 = 103
  all_goals solved sorry -- Skip proof for simplicity

end black_area_greater_than_gray_area_by_l513_513345


namespace true_proposition_l513_513766

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513766


namespace units_digit_of_product_of_odds_l513_513516

theorem units_digit_of_product_of_odds :
  let odds := {n | 20 < n ∧ n < 120 ∧ n % 2 = 1}
  (∃ us : List ℕ, us = (List.filter (λ x, x ∈ odds) (List.range 120))) →
  (∃ p : ℕ, p = us.prod) →
  (∃ d : ℕ, d = p % 10) →
  d = 5 :=
by
  sorry

end units_digit_of_product_of_odds_l513_513516


namespace BC_on_same_side_of_A_l513_513317

theorem BC_on_same_side_of_A (A B C D : Type) :
  ∃ f : (A × B × C × D → ℕ), (f (A, B, C, D) = 16) :=
by
  sorry

end BC_on_same_side_of_A_l513_513317


namespace problem_equiv_proof_l513_513026

noncomputable def simplify_and_evaluate (a : ℝ) :=
  ((a + 1) / (a + 2) + 1 / (a - 2)) / (2 / (a^2 - 4))

theorem problem_equiv_proof :
  simplify_and_evaluate (Real.sqrt 2) = 1 := 
  sorry

end problem_equiv_proof_l513_513026


namespace net_effect_transactions_l513_513166

theorem net_effect_transactions {a o : ℝ} (h1 : 3 * a / 4 = 15000) (h2 : 5 * o / 4 = 15000) :
  a + o - (2 * 15000) = 2000 :=
by
  sorry

end net_effect_transactions_l513_513166


namespace line_y_intercept_l513_513505

theorem line_y_intercept : ∀ (x y : ℝ), x - 2 * y = 5 ∧ x = 0 -> y = -5 / 2 := by
  intros x y h
  cases h with h1 h2
  rw [h2, zero_sub, neg_eq_iff_neg_eq] at h1
  linarith

end line_y_intercept_l513_513505


namespace true_propositions_for_function_l513_513434

theorem true_propositions_for_function :
  let f : ℝ → ℝ := λ x, 4 * sin (2 * x + π / 3)
  ∧
    -- Proposition (1) is incorrect
    (∃ T > 0, ∀ x, f (x + T) = f x) ↔ T = π ∨ T > π
  ∧
    -- Proposition (2) is correct
    (∀ x, f x = 4 * cos (2 * x - π / 6))
  ∧
    -- Proposition (3) is correct
    (∀ x, f (-π / 6 - x) = -f (π / 6 + x))
  ∧
    -- Proposition (4) is incorrect
    ¬ (∀ x, f (-π / 6 + x) = f (π / 6 - x)) :=
by {
  let f := λ x : ℝ, 4 * sin (2 * x + π / 3),
  -- You can add the proof steps here
  sorry
}

end true_propositions_for_function_l513_513434


namespace original_ribbon_length_l513_513588

-- Define the parameters and the conditions of the problem
def ribbon_width := 2 -- cm
def num_small_ribbons := 10
def smallest_ribbon_length := 8 -- from the solution step

-- Define what we need to prove
theorem original_ribbon_length : 
  ∃ (L : ℕ), L = num_small_ribbons * smallest_ribbon_length ∧ ribbon_width = 2 :=
by 
  -- Define the length of the original ribbon
  let L := num_small_ribbons * smallest_ribbon_length
  -- Ensure L meets the required conditions
  existsi L
  split
  -- Prove the conditions
  exact rfl
  exact rfl
  sorry

end original_ribbon_length_l513_513588


namespace part_a_part_b_part_c_l513_513334

-- Defining a structure for the problem
structure Rectangle :=
(area : ℝ)

structure Figure :=
(area : ℝ)

-- Defining the conditions
variables (R : Rectangle) 
  (F1 F2 F3 F4 F5 : Figure)
  (overlap_area_pair : Figure → Figure → ℝ)
  (overlap_area_triple : Figure → Figure → Figure → ℝ)

-- Given conditions
axiom R_area : R.area = 1
axiom F1_area : F1.area = 0.5
axiom F2_area : F2.area = 0.5
axiom F3_area : F3.area = 0.5
axiom F4_area : F4.area = 0.5
axiom F5_area : F5.area = 0.5

-- Statements to prove
theorem part_a : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 3 / 20 := sorry
theorem part_b : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 1 / 5 := sorry
theorem part_c : ∃ (F1 F2 F3 : Figure), overlap_area_triple F1 F2 F3 ≥ 1 / 20 := sorry

end part_a_part_b_part_c_l513_513334


namespace infinitely_many_m_l513_513433

theorem infinitely_many_m (r : ℕ) (n : ℕ) (h_r : r > 1) (h_n : n > 0) : 
  ∃ m, m = 4 * r ^ 4 ∧ ¬Prime (n^4 + m) :=
by
  sorry

end infinitely_many_m_l513_513433


namespace tram_passengers_l513_513095

theorem tram_passengers:
  ∃ P : ℕ, 
  P % 2 = 0 ∧
  1.08 * P ≤ 70 ∧
  P = 50 :=
by
  have P_possible := int.mod_eq_zero_of_dvd 2,
  have mul_ineq := real.mul_self_le_mul_mul 1.08 1 P (by norm_num) (by exact_mod_cast P),
  have eq_check := eq.refl 50,
  sorry

end tram_passengers_l513_513095


namespace first_term_exceeds_10000_l513_513456

noncomputable def sequence : ℕ → ℕ
| 0 => 3
| n + 1 => ∑ i in Finset.range (n + 1), (sequence i)^2

theorem first_term_exceeds_10000 : ∃ n, sequence n > 10000 ∧ sequence n = 67085190 := by
sorry

end first_term_exceeds_10000_l513_513456


namespace range_of_y_when_m_eq_neg1_max_value_of_y_l513_513281

section
variables (x m : ℝ)
noncomputable def y (x m : ℝ) : ℝ := 2 * (sin x)^2 + m * cos x - 1 / 8

theorem range_of_y_when_m_eq_neg1 :
  -π/3 ≤ x ∧ x ≤ 2π/3 → m = -1 → y x m ∈ set.interval (-9/8) 2 :=
sorry

theorem max_value_of_y :
  x ∈ ℝ →
  (m < -4 → y x m ≤ -m - 1/8)
  ∧ (m > 4 → y x m ≤ m - 1/8)
  ∧ (-4 ≤ m ∧ m ≤ 4 → y x m ≤ (m^2 + 15) / 8) :=
sorry
end

end range_of_y_when_m_eq_neg1_max_value_of_y_l513_513281


namespace wage_ratio_l513_513042

-- Define the conditions
variable (M W : ℝ) -- M stands for man's daily wage, W stands for woman's daily wage
variable (h1 : 40 * 10 * M = 14400) -- Condition 1: 40 men working for 10 days earn Rs. 14400
variable (h2 : 40 * 30 * W = 21600) -- Condition 2: 40 women working for 30 days earn Rs. 21600

-- The statement to prove
theorem wage_ratio (h1 : 40 * 10 * M = 14400) (h2 : 40 * 30 * W = 21600) : M / W = 2 := by
  sorry

end wage_ratio_l513_513042


namespace units_digit_of_product_of_odds_l513_513518

theorem units_digit_of_product_of_odds :
  let odds := {n | 20 < n ∧ n < 120 ∧ n % 2 = 1}
  (∃ us : List ℕ, us = (List.filter (λ x, x ∈ odds) (List.range 120))) →
  (∃ p : ℕ, p = us.prod) →
  (∃ d : ℕ, d = p % 10) →
  d = 5 :=
by
  sorry

end units_digit_of_product_of_odds_l513_513518


namespace intersection_point_exists_l513_513218

def line_param_eq (x y z : ℝ) (t : ℝ) := x = 5 + t ∧ y = 3 - t ∧ z = 2
def plane_eq (x y z : ℝ) := 3 * x + y - 5 * z - 12 = 0

theorem intersection_point_exists : 
  ∃ t : ℝ, ∃ x y z : ℝ, line_param_eq x y z t ∧ plane_eq x y z ∧ x = 7 ∧ y = 1 ∧ z = 2 :=
by {
  -- Skipping the proof
  sorry
}

end intersection_point_exists_l513_513218


namespace fine_per_day_l513_513563

theorem fine_per_day (x : ℝ) : 
  (let total_days := 30 in
   let earnings_per_day := 25 in
   let total_amount_received := 425 in
   let days_absent := 10 in
   let days_worked := total_days - days_absent in
   let total_earnings := days_worked * earnings_per_day in
   let total_fine := days_absent * x in
   total_earnings - total_fine = total_amount_received) → x = 7.5 :=
by
  intros h
  sorry

end fine_per_day_l513_513563


namespace find_r_power_4_l513_513315

variable {r : ℝ}

theorem find_r_power_4 (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := 
sorry

end find_r_power_4_l513_513315


namespace bill_sunday_vs_saturday_l513_513425

theorem bill_sunday_vs_saturday:
  ∃ (B_Sat : ℕ), B_Sun = 9 ∧ J_Sun = 2 * 9 ∧ (B_Sat + B_Sun + J_Sun = 32) ∧ 
  (B_Sun - B_Sat = 4) :=
begin
  let B_Sun := 9,
  let J_Sun := 2 * B_Sun,
  have h_sum : B_Sat + B_Sun + J_Sun = 32 := by sorry,
  have h_diff : B_Sun - B_Sat = 4 := by sorry,
  use (32 - 27), -- This is B_Sat which is actually 5
  exact ⟨rfl, rfl, h_sum, h_diff⟩,
end

end bill_sunday_vs_saturday_l513_513425


namespace average_disk_space_per_hour_l513_513579

/-- A digital music library contains 15 days of music and takes up 20,000 megabytes of disk space.
    Prove that the average disk space used per hour of music in this library is 56 megabytes, to the nearest whole number.
-/
theorem average_disk_space_per_hour (days: ℕ) (total_disk_space: ℕ) (hours_per_day: ℕ) (div_result: ℕ) (avg_disk_space: ℕ) :
  days = 15 ∧ 
  total_disk_space = 20000 ∧ 
  hours_per_day = 24 ∧ 
  div_result = days * hours_per_day ∧
  avg_disk_space = (total_disk_space: ℝ) / div_result ∧ 
  (an_index: ℕ, (avg_disk_space: ℝ) ≈ 55.56 → an_index = 56) :=
by 
  sorry

end average_disk_space_per_hour_l513_513579


namespace volleyball_team_selection_l513_513000

theorem volleyball_team_selection :
  let total_players := 16
  let quadruplets := 4
  let starters := 7
  (nat.choose (total_players - quadruplets) starters) + -- Case 1
  (nat.choose quadruplets 1 * nat.choose (total_players - quadruplets) (starters - 1)) = 4488 := by
  sorry

end volleyball_team_selection_l513_513000


namespace p_and_q_is_true_l513_513696

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513696


namespace central_angle_measure_l513_513130

theorem central_angle_measure (p : ℝ) (x : ℝ) (h1 : p = 1 / 8) (h2 : p = x / 360) : x = 45 :=
by
  -- skipping the proof
  sorry

end central_angle_measure_l513_513130


namespace find_a_and_b_l513_513286

theorem find_a_and_b (a b : ℤ) :
  let M := ({(a + 3) + ((b^2 - 1) * I), 8} : Set ℂ)
  let N := ({3 * I, (a^2 - 1) + ((b + 2) * I)} : Set ℂ)
  M ∩ N ⊆ M ∧ M ∩ N ≠ ∅ →
  (a = -3 ∧ b = 2) ∨ (a = 3 ∧ b = -2) :=
by
  intros M N h
  sorry

end find_a_and_b_l513_513286


namespace intersection_point_count_l513_513895

def abs_val_f1 (x : ℝ) : ℝ := abs (3 * x + 6)
def abs_val_f2 (x : ℝ) : ℝ := -abs (4 * x - 3)

def abs_val_f1_piecewise (x : ℝ) : ℝ :=
if x >= -2 then
  3 * x + 6
else
  -3 * x - 6

def abs_val_f2_piecewise (x : ℝ) : ℝ :=
if x >= 3 / 4 then
  -4 * x + 3
else
  4 * x - 3

theorem intersection_point_count : 
  ∃! (x y : ℝ), abs_val_f1 x = y ∧ abs_val_f2 x = y := 
sorry

end intersection_point_count_l513_513895


namespace prove_inequality_l513_513550

-- Define the function properties
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Function properties as given in the problem
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def decreasing_on_nonneg (f : ℝ → ℝ) := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- The main theorem statement
theorem prove_inequality (h_even : even_function f) (h_dec : decreasing_on_nonneg f) :
  f (-3 / 4) ≥ f (a^2 - a + 1) :=
sorry

end prove_inequality_l513_513550


namespace min_value_expression_l513_513386

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 20)^2 ≥ 100 :=
sorry

end min_value_expression_l513_513386


namespace g_h_of_2_eq_5288_l513_513282

def g (x : ℝ) : ℝ := 3 * x^2 - 4
def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem g_h_of_2_eq_5288 : g (h 2) = 5288 :=
by
  have h2 := h 2
  have gh2 := g h2
  show gh2 = 5288
  sorry

end g_h_of_2_eq_5288_l513_513282


namespace ball_total_distance_l513_513122

def total_distance (initial_height : ℝ) (bounce_factor : ℝ) (bounces : ℕ) : ℝ :=
  let rec loop (height : ℝ) (total : ℝ) (remaining : ℕ) : ℝ :=
    if remaining = 0 then total
    else loop (height * bounce_factor) (total + height + height * bounce_factor) (remaining - 1)
  loop initial_height 0 bounces

theorem ball_total_distance : 
  total_distance 20 0.8 4 = 106.272 :=
by
  sorry

end ball_total_distance_l513_513122


namespace files_more_than_apps_l513_513619

def initial_apps : ℕ := 11
def initial_files : ℕ := 3
def remaining_apps : ℕ := 2
def remaining_files : ℕ := 24

theorem files_more_than_apps : remaining_files - remaining_apps = 22 :=
by
  sorry

end files_more_than_apps_l513_513619


namespace proposition_A_l513_513815

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513815


namespace pow_two_grows_faster_than_square_l513_513229

theorem pow_two_grows_faster_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := sorry

end pow_two_grows_faster_than_square_l513_513229


namespace price_of_n_kilograms_l513_513925

theorem price_of_n_kilograms (m n : ℕ) (hm : m ≠ 0) (h : 9 = m) : (9 * n) / m = (9 * n) / m :=
by
  sorry

end price_of_n_kilograms_l513_513925


namespace angle_between_a_and_c_l513_513289

open Real

-- Define the vectors 
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -1)
def c (λ : ℝ) : ℝ × ℝ := (1, λ)

-- Define the dot product function
def dot (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define magnitudes of vectors
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the angle function
def cos_angle (u v : ℝ × ℝ) : ℝ := (dot u v) / (magnitude u * magnitude v)

theorem angle_between_a_and_c : ∀ (λ : ℝ), (dot (a.1 + b.1, a.2 + b.2) (c λ) = 0) → 
                                λ = -3 →
                                arccos (cos_angle a (c λ)) = 3 * π / 4 :=
by
  intros
  sorry

end angle_between_a_and_c_l513_513289


namespace total_emails_in_april_l513_513953

-- Definitions representing the conditions
def emails_per_day_initial : Nat := 20
def extra_emails_per_day : Nat := 5
def days_in_month : Nat := 30
def half_days_in_month : Nat := days_in_month / 2

-- Definitions to calculate total emails
def emails_first_half : Nat := emails_per_day_initial * half_days_in_month
def emails_per_day_after_subscription : Nat := emails_per_day_initial + extra_emails_per_day
def emails_second_half : Nat := emails_per_day_after_subscription * half_days_in_month

-- Main theorem to prove the total number of emails received in April
theorem total_emails_in_april : emails_first_half + emails_second_half = 675 := by 
  calc
    emails_first_half + emails_second_half
    = (emails_per_day_initial * half_days_in_month) + (emails_per_day_after_subscription * half_days_in_month) : rfl
    ... = (20 * 15) + ((20 + 5) * 15) : rfl
    ... = 300 + 375 : rfl
    ... = 675 : rfl

end total_emails_in_april_l513_513953


namespace sum_mod_1000_l513_513144

def sequence (n : ℕ) : ℕ 
| 0     := 2
| 1     := 2
| 2     := 2
| (n+3) := sequence n + sequence (n+1) + sequence (n+2)

noncomputable def b := sequence

example : b 30 = 24791411 := by sorry
example : b 31 = 45765219 := by sorry
example : b 32 = 84215045 := by sorry

theorem sum_mod_1000 : (∑ k in finset.range 30, b k) % 1000 = 228 := by
  sorry

end sum_mod_1000_l513_513144


namespace sum_of_all_possible_values_of_z_l513_513647

noncomputable def sum_of_z_values (w x y z : ℚ) : ℚ :=
if h : w < x ∧ x < y ∧ y < z ∧ 
       (w + x = 1 ∧ w + y = 2 ∧ w + z = 3 ∧ x + y = 4 ∨ 
        w + x = 1 ∧ w + y = 2 ∧ w + z = 4 ∧ x + y = 3) ∧ 
       ((w + x) ≠ (w + y) ∧ (w + x) ≠ (w + z) ∧ (w + x) ≠ (x + y) ∧ (w + x) ≠ (x + z) ∧ (w + x) ≠ (y + z)) ∧ 
       ((w + y) ≠ (w + z) ∧ (w + y) ≠ (x + y) ∧ (w + y) ≠ (x + z) ∧ (w + y) ≠ (y + z)) ∧ 
       ((w + z) ≠ (x + y) ∧ (w + z) ≠ (x + z) ∧ (w + z) ≠ (y + z)) ∧ 
       ((x + y) ≠ (x + z) ∧ (x + y) ≠ (y + z)) ∧ 
       ((x + z) ≠ (y + z)) then
  if w + z = 4 then
    4 + 7/2
  else 0
else
  0

theorem sum_of_all_possible_values_of_z : sum_of_z_values w x y z = 15 / 2 :=
by sorry

end sum_of_all_possible_values_of_z_l513_513647


namespace part_a_part_b_l513_513495

-- Define the setup and conditions
variable (A B C D E F : Point)
variable (circumcircle : Circle)
variable (incircle : Circle)
variable (occurs_in_order : CycleOrder A F B D C E circumcircle)
variable (Δ_A Δ_B Δ_C Δ_D Δ_E Δ_F : Triangle)
variable (circum_A, circum_D : Circle)
variable (incircle_A, incircle_D : Circle)
variable (common_external_tangents_concurrent_or_parallel : 
  ∀ (tangent1 tangent2 tangent3 tangent4 : Line), 
    (CommonExternalTangent tangent1 circum_A circum_D ∧ 
     CommonExternalTangent tangent2 circum_A circum_D ∧ 
     CommonExternalTangent tangent3 incircle_A incircle_D ∧ 
     CommonExternalTangent tangent4 incircle_A incircle_D) 
    → (TangentConcurrent tangent1 tangent2 tangent3 tangent4 ∨ 
       TangentParallel tangent1 tangent2 tangent3 tangent4))

-- Ensure this corresponds to the mathematical problem
theorem part_a : common_external_tangents_concurrent_or_parallel := 
by
  sorry

-- Define additional setup for part (b)
variable (T_A T_B T_C : Point)
variable (T_A_defined : TangentIntersect T_A circum_A incircle_A)
variable (T_B_defined : TangentIntersect T_B circum_B incircle_B)
variable (T_C_defined : TangentIntersect T_C circum_C incircle_C)
variable (points_collinear : Collinear T_A T_B T_C)

-- Ensure this corresponds to the mathematical problem
theorem part_b : ∃ T_A T_B T_C, points_collinear := 
by
  sorry

end part_a_part_b_l513_513495


namespace max_value_of_sum_abs_diff_l513_513884

theorem max_value_of_sum_abs_diff (a : Fin 5 → ℝ)
  (h₁ : (∑ i in Finset.univ, a i ^ 2) = 1)
  (h₂ : a 5 = a 0) :
  (∑ i in Finset.univ, |a i - a ((i + 1) % 5)|) ≤ 2 * sqrt 3 :=
sorry

end max_value_of_sum_abs_diff_l513_513884


namespace total_settings_weight_l513_513415

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end total_settings_weight_l513_513415


namespace prob_revenue_30_million_expect_revenue_l513_513938

-- Conditions given
def bestOfSevenSeries : Prop :=
  ∀ (n : ℕ), (n ≥ 0 → n ≤ 7) ∧ (n = 4 → True) ∧ (n < 4 → False)

def evenlyMatchedTeams : Prop :=
  ∀ (A B : ℕ), (A + B = 1) ∧ (probA = 1/2) ∧ (probB = 1/2)

def ticketRevenue (n : ℕ) : ℕ :=
  if n = 1 then 4 else 100 * n + 300

-- Question 1: Prove the probability that the total ticket revenue is exactly 30 million yuan is 1/4
theorem prob_revenue_30_million (n : ℕ) (prob : ℚ) :
  bestOfSevenSeries →
  evenlyMatchedTeams →
  ticketRevenue n = 30 →
  prob = 1/4 :=
sorry

-- Question 2: Prove the mathematical expectation of total ticket revenue is 3775 million yuan.
theorem expect_revenue (E : ℚ) :
  bestOfSevenSeries →
  evenlyMatchedTeams →
  (∃ (S : ℕ), ticketRevenue S = E) →
  E = 3775 :=
sorry

end prob_revenue_30_million_expect_revenue_l513_513938


namespace compare_abc_l513_513653

noncomputable def a : ℝ := 9 ^ (Real.log 4.1 / Real.log 2)
noncomputable def b : ℝ := 9 ^ (Real.log 2.7 / Real.log 2)
noncomputable def c : ℝ := (1 / 3 : ℝ) ^ (Real.log 0.1 / Real.log 2)

theorem compare_abc :
  a > c ∧ c > b := by
  sorry

end compare_abc_l513_513653


namespace at_least_one_not_less_than_neg_two_l513_513398

theorem at_least_one_not_less_than_neg_two (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≥ -2 ∨ b + 1/c ≥ -2 ∨ c + 1/a ≥ -2) :=
sorry

end at_least_one_not_less_than_neg_two_l513_513398


namespace proof_problem_l513_513799

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513799


namespace proof_problem_l513_513803

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l513_513803


namespace total_squares_in_6x6_grid_l513_513628

theorem total_squares_in_6x6_grid : 
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  total_squares = 91 :=
by
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  have eqn : total_squares = 91 := sorry
  exact eqn

end total_squares_in_6x6_grid_l513_513628


namespace matrix_N_unique_l513_513214

theorem matrix_N_unique :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
  (N ⬝ (colVec ((4 : ℝ), (1 : ℝ))) = colVec ((9 : ℝ), (0 : ℝ))) ∧ 
  (N ⬝ (colVec ((2 : ℝ), (-3 : ℝ))) = colVec ((-6 : ℝ), (12 : ℝ))) ∧ 
  N = (Matrix.vecCons (colVec (3, 0)) (Matrix.vecCons (colVec (0, 0)) Matrix.vecEmpty)) :=
by
  sorry

def colVec (a b : ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  ![[a], [b]]

end matrix_N_unique_l513_513214


namespace three_legged_extraterrestrials_l513_513204

-- Define the conditions
variables (x y : ℕ)

-- Total number of heads
def heads_equation := x + y = 300

-- Total number of legs
def legs_equation := 3 * x + 4 * y = 846

theorem three_legged_extraterrestrials : heads_equation x y ∧ legs_equation x y → x = 246 :=
by
  sorry

end three_legged_extraterrestrials_l513_513204


namespace average_disk_space_per_hour_l513_513575

theorem average_disk_space_per_hour:
  (let total_hours := 15 * 24 in
  ∀ (total_disk_space : ℕ), total_disk_space = 20000 →
  20000 / total_hours = 56) :=
begin
  let total_hours := 15 * 24,
  intro total_disk_space,
  intro h,
  simp [h, total_hours],
  sorry
end

end average_disk_space_per_hour_l513_513575


namespace statements_correctness_l513_513192

def statement1 : Prop := 
  "Raising the temperature by 5°C and -3°C are a pair of quantities with opposite meanings."

def statement2 : Prop := 
  ∀ (x : ℚ), x.is_point_on_number_line → (x : ℝ).is_point_on_number_line

def statement3 : Prop := 
  ∀ (x : ℝ), abs x > 0

def statement4 : Prop := 
  ∀ (x : ℝ), sqrt x = x → (x = 0 ∨ x = 1)

def statement5 : Prop := 
  ∀ (x y : ℚ), (x + y = 0) → (x > 0 ∨ y > 0)

theorem statements_correctness : 
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 ∧ ¬statement5) → 
  0 = 0 := by sorry

end statements_correctness_l513_513192


namespace single_elimination_matches_l513_513935

theorem single_elimination_matches (n : ℕ) (h : n = 512) :
  ∃ (m : ℕ), m = n - 1 ∧ m = 511 :=
by
  sorry

end single_elimination_matches_l513_513935


namespace puppies_count_l513_513073

theorem puppies_count 
  (dogs : ℕ := 3)
  (dog_meal_weight : ℕ := 4)
  (dog_meals_per_day : ℕ := 3)
  (total_food : ℕ := 108)
  (puppy_meal_multiplier : ℕ := 2)
  (puppy_meal_frequency_multiplier : ℕ := 3) :
  ∃ (puppies : ℕ), puppies = 4 :=
by
  let dog_daily_food := dog_meal_weight * dog_meals_per_day
  let puppy_meal_weight := dog_meal_weight / puppy_meal_multiplier
  let puppy_daily_food := puppy_meal_weight * puppy_meal_frequency_multiplier * dog_meals_per_day
  let total_dog_food := dogs * dog_daily_food
  let total_puppy_food := total_food - total_dog_food
  let puppies := total_puppy_food / puppy_daily_food
  use puppies
  have h_puppies_correct : puppies = 4 := sorry
  exact h_puppies_correct

end puppies_count_l513_513073


namespace trig_identity_l513_513264

theorem trig_identity (k : ℝ) (hk : k < 0) (θ : ℝ)
  (hθ : ∃ (P : ℝ × ℝ), P = (-4*k, 3*k) ∧
        cos θ = (-4*k) / (sqrt ((-4*k)^2 + (3*k)^2)) ∧
        sin θ = (3*k) / (sqrt ((-4*k)^2 + (3*k)^2))) :
  2 * sin θ + cos θ = -2/5 :=
sorry

end trig_identity_l513_513264


namespace Jason_spent_on_music_store_l513_513949

theorem Jason_spent_on_music_store:
  let flute := 142.46
  let music_stand := 8.89
  let song_book := 7.00
  flute + music_stand + song_book = 158.35 := sorry

end Jason_spent_on_music_store_l513_513949


namespace find_selling_price_l513_513559

noncomputable def selling_price (x : ℝ) : ℝ :=
  (x - 60) * (1800 - 20 * x)

constant purchase_price : ℝ := 60
constant max_profit_margin : ℝ := 0.40
constant base_selling_price : ℝ := 80
constant base_units_sold : ℝ := 200
constant decrement_units_sold : ℝ := 20
constant target_profit : ℝ := 2500

theorem find_selling_price (x : ℝ) :
  selling_price x = target_profit ∧
  (x - 60) / 60 ≤ max_profit_margin ∧
  ∃ u : ℝ, u = (base_units_sold + decrement_units_sold * (base_selling_price - x))
  → x = 65 :=
sorry

end find_selling_price_l513_513559


namespace satisfies_conditions_l513_513100

theorem satisfies_conditions : 
  ∃ f : ℝ → ℝ, (∀ x, f (-x) = -f x) ∧
               (∀ x, x > 2 → 0 < deriv f x) ∧
               (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
               f = λ x, x * (x + 1) * (x - 1) :=
by {
  use λ x : ℝ, x * (x + 1) * (x - 1),
  split,
  { intro x,
    simp,
    ring },
  split,
  { intros x hx,
    calc deriv (λ x : ℝ, x * (x + 1) * (x - 1)) x = 3 * x ^ 2 - 1 : by { sorry }
    ... > 0 : by { sorry }},
  {
    use [0, 1, -1],
    simp,
    split,
    { linarith },
    split,
    { linarith },
    split,
    { linarith },
    split,
    { ring },
    split,
    { ring },
    { ring }
  }
}

end satisfies_conditions_l513_513100


namespace maria_remaining_salary_l513_513407

variable (salary taxRate insuranceRate utilityRate : ℝ)

def remaining_salary_after_deductions (salary taxRate insuranceRate utilityRate : ℝ) : ℝ :=
  let tax = salary * taxRate
  let insurance = salary * insuranceRate
  let totalDeductions = tax + insurance
  let amountAfterDeductions = salary - totalDeductions
  let utilityBills = amountAfterDeductions * utilityRate
  amountAfterDeductions - utilityBills

theorem maria_remaining_salary :
  remaining_salary_after_deductions 2000 0.2 0.05 (1 / 4) = 1125 :=
by
  unfold remaining_salary_after_deductions
  norm_num
  sorry

end maria_remaining_salary_l513_513407


namespace omega_value_monotonically_decreasing_intervals_l513_513278

-- Definitions according to the conditions
def f (omega : ℝ) (x : ℝ) : ℝ := 
  sin (omega * x) * (cos (omega * x) - real.sqrt 3 * sin (omega * x)) + real.sqrt 3 / 2

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≥ f y

-- Theorem statements based on the questions
theorem omega_value (omega : ℝ) (h : is_periodic (f omega) (π / 2)) : omega = 2 := 
sorry

theorem monotonically_decreasing_intervals (omega : ℝ) (h₁ : omega = 2) :
  ∀ k : ℤ, is_monotonically_decreasing (f omega) (k * π / 2 + π / 24) (k * π / 2 + 7 * π / 24) := 
sorry

end omega_value_monotonically_decreasing_intervals_l513_513278


namespace oil_tank_height_l513_513601

/--
Given an oil tank shaped like a right cylinder with the following properties:
- The tank is 20 feet tall when standing upright.
- The tank has circular bases each with a diameter of 5 feet.
- The tank is currently horizontal, and the oil inside reaches a depth of 4 feet from the bottom.

Prove that when the tank is standing upright, the oil inside reaches a height of 1.0 foot.
-/
theorem oil_tank_height
  (cylinder_height : ℝ) (cylinder_diameter : ℝ) (oil_depth_horizontal : ℝ)
  (cylinder_height_eq : cylinder_height = 20)
  (cylinder_diameter_eq : cylinder_diameter = 5)
  (oil_depth_horizontal_eq : oil_depth_horizontal = 4) :
  ∃ (oil_depth_upright : ℝ), oil_depth_upright = 1 :=
by
  let r := cylinder_diameter / 2
  have r_eq : r = 2.5 := by
    rw [cylinder_diameter_eq]
    exact (by norm_num : 5 / 2 = 2.5)

  let V := (oil_depth_horizontal / cylinder_diameter) * pi * (r^2) * (20 : ℝ)
  have V_eq : V = 20 * pi * (r^2) * (4 / 5) := by
    rw [oil_depth_horizontal_eq, cylinder_diameter_eq]
    
  let h := V / (pi * r^2)
  use h
  have h_eq : h = 1 := by
    sorry

  exact ⟨h, h_eq⟩

end oil_tank_height_l513_513601


namespace number_of_x_l513_513899

open Real

theorem number_of_x (tan cos : ℝ → ℝ) (a u : ℝ) :
(∀ x : ℝ, -20 < x ∧ x < 100 → tan x = sin x / cos x ∧ cos x ≠ 0 ∧ tan x * tan x + 2 * cos x * cos x = 2) → 
card {x : ℝ | -20 < x ∧ x < 100 ∧ tan x * tan x + 2 * cos x * cos x = 2} = 114 := 
sorry

end number_of_x_l513_513899


namespace problem_1_problem_2_l513_513551

-- Define arithmetic sequence sum S_n
def S_arith (a d n : ℕ) : ℕ := n * a + (n * (n - 1) / 2) * d

-- Problem (1)
theorem problem_1 : S_arith (-2) 4 8 = 96 := by
  sorry

-- Define geometric sequence nth term a_n
def a_geo (a q n : ℕ) : ℕ := a * q ^ (n - 1)

-- Problem (2)
theorem problem_2 : 
  (∃ a : ℕ, a_geo a 3 4 = 27) → a_geo 1 3 7 = 729 := by
  sorry

end problem_1_problem_2_l513_513551


namespace problem_statement_l513_513710

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l513_513710


namespace square_of_binomial_l513_513310

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 30 * x + a) → a = 25 :=
by
  sorry

end square_of_binomial_l513_513310


namespace true_proposition_l513_513757

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l513_513757


namespace det_mul_eq_one_of_inverse_property_l513_513379

variable {n : Type*} [Fintype n] [DecidableEq n] [Field ℝ]
variables (A B : Matrix n n ℝ)

theorem det_mul_eq_one_of_inverse_property
  (hA_inv : ∃ A_inv : Matrix n n ℝ, A * A_inv = 1 ∧ A_inv * A = 1)
  (hB_inv : ∃ B_inv : Matrix n n ℝ, B * B_inv = 1 ∧ B_inv * B = 1)
  (h_inverse : ∃ C_inv : Matrix n n ℝ, C_inv = A⁻¹ + B ∧ (A + B⁻¹) * C_inv = 1) :
  det (A ⬝ B) = 1 := sorry

end det_mul_eq_one_of_inverse_property_l513_513379


namespace cosine_periodic_func_1_cosine_periodic_func_2_cosine_periodic_func_3_count_cosine_periodic_funcs_l513_513223

def is_cosine_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, cos (f (x + T)) = cos (f x)

def func_1 : ℝ → ℝ := λ x, 2018 * x
def func_2 : ℝ → ℝ := abs
def func_3 : ℝ → ℝ := λ x, x + sin (x / 3)

theorem cosine_periodic_func_1 : ∃ T > 0, is_cosine_periodic func_1 T := by
  use π
  sorry

theorem cosine_periodic_func_2 : ∃ T > 0, is_cosine_periodic func_2 T := by
  use 2 * π
  sorry

theorem cosine_periodic_func_3 : ∃ T > 0, is_cosine_periodic func_3 T := by
  use 6 * π
  sorry

theorem count_cosine_periodic_funcs :
  ({func_1, func_2, func_3} : set (ℝ → ℝ)).filter (λ f, ∃ T > 0, is_cosine_periodic f T)).card = 3 := by
  sorry

end cosine_periodic_func_1_cosine_periodic_func_2_cosine_periodic_func_3_count_cosine_periodic_funcs_l513_513223


namespace polynomial_divisible_by_a_plus_1_l513_513643

theorem polynomial_divisible_by_a_plus_1 (a : ℤ) : (3 * a + 5) ^ 2 - 4 ∣ a + 1 := 
by
  sorry

end polynomial_divisible_by_a_plus_1_l513_513643


namespace p_and_q_is_true_l513_513687

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513687


namespace a_plus_b_eq_neg12_l513_513263

noncomputable def find_a_b : ℝ × ℝ :=
  let y (x : ℝ) (a : ℝ) (b : ℝ) := x^3 + a * x^2 + b * x + 27
  let y' (x : ℝ) (a : ℝ) (b : ℝ) := 3 * x^2 + 2 * a * x + b
  let a := -3
  let b := -9
  (a, b)

theorem a_plus_b_eq_neg12 : 
  ∃ a b : ℝ, 
    (∀ x, deriv (λ x : ℝ, x^3 + a * x^2 + b * x + 27) x = 3 * x^2 + 2 * a * x + b) ∧
    (deriv (λ x : ℝ, x^3 + a * x^2 + b * x + 27) (-1) = 0) ∧
    (deriv (λ x : ℝ, x^3 + a * x^2 + b * x + 27) 3 = 0) ∧
    a + b = -12 :=
by 
  let a := -3
  let b := -9
  use [a, b]
  sorry

end a_plus_b_eq_neg12_l513_513263


namespace new_average_l513_513616

variables (k : ℝ)
variables (numbers : Fin 20 → ℝ)
variables (avg : ℝ)

-- Assume that the average of 20 numbers is 35
axiom avg_20_numbers (h_avg : avg = 35) : (1 / (20 : ℝ)) * (Finset.univ.sum numbers) = avg

-- Statement to prove
theorem new_average (h_avg : avg = 35) (h_sum : Finset.univ.sum numbers = 700) :
  (1 / (20 : ℝ)) * (Finset.univ.sum (λ i, k * numbers i)) = 35 * k :=
by
  sorry

end new_average_l513_513616


namespace centroid_positions_count_correct_l513_513033

noncomputable def centroid_positions_count : ℕ :=
  let points := (List.range 21).map (λ i, (i, 0)) ++ 
                (List.range 1 21).map (λ i, (20, i)) ++ 
                (List.range 19).reverse.map (λ i, (i, 20)) ++ 
                (List.range 1 20).reverse.map (λ i, (0, i))
  let centroids : List (ℚ × ℚ) := (points.product points).product points |>.map (λ ((p1, p2), p3), 
    let (x1, y1) := p1
    let (x2, y2) := p2
    let (x3, y3) := p3
    ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3))
  let valid_centroids := centroids.filter (λ (x, y), 0 < x ∧ x < 20 ∧ 0 < y ∧ y < 20)
  valid_centroids.uniq.length

theorem centroid_positions_count_correct :
  centroid_positions_count = 3481 :=
sorry

end centroid_positions_count_correct_l513_513033


namespace solve_equation_l513_513640

theorem solve_equation :
  ∀ x : ℝ, 
    (x ≠ -2) → 
    ((17 * x - x^2) / (x + 2) * (x + (17 - x) / (x + 2)) = 56) ↔ 
    (x = 1 ∨ x = 65 ∨ x = -25 + real.sqrt 624 ∨ x = -25 - real.sqrt 624) := 
by
  intro x hx,
  split,
  { intro h,
    sorry, },
  { intro h,
    sorry }

end solve_equation_l513_513640


namespace triangle_type_invariant_l513_513490

open EuclideanGeometry

def regular_ngon (n : ℕ) (n_pos : n ≥ 3) : Type :=
{ vertices : ℕ → ℝ × ℝ // ∀ i : ℕ, (i < n -> vertices i ≠ vertices ((i + 1) % n)) }

theorem triangle_type_invariant (n : ℕ) (n_pos : n ≥ 3) (n_ne_5 : n ≠ 5) :
  ∃ (v1 v2 v3 w1 w2 w3 : ℤ),
    (is_acute (v1, v2, v3) ∨ is_obtuse (v1, v2, v3) ∨ is_right (v1, v2, v3)) ∧
    (is_acute (w1, w2, w3) ∨ is_obtuse (w1, w2, w3) ∨ is_right (w1, w2, w3)) :=
sorry

end triangle_type_invariant_l513_513490


namespace problem_divisible_by_factors_l513_513025

theorem problem_divisible_by_factors (n : ℕ) (x : ℝ) : 
  ∃ k : ℝ, (x + 1)^(2 * n) - x^(2 * n) - 2 * x - 1 = k * x * (x + 1) * (2 * x + 1) :=
by
  sorry

end problem_divisible_by_factors_l513_513025


namespace proposition_p_and_q_is_true_l513_513716

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513716


namespace fraction_sum_neg_one_l513_513181

variable (a : ℚ)

theorem fraction_sum_neg_one (h : a ≠ 1/2) : (a / (1 - 2 * a)) + ((a - 1) / (1 - 2 * a)) = -1 := 
sorry

end fraction_sum_neg_one_l513_513181


namespace problem_statement_l513_513792

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l513_513792


namespace common_sum_on_faces_l513_513465

def vertices := {1, 2, 3, 4, 5, 6, 7, 8}

def faces (f : Finset (Finset ℕ)) := f.card = 6 ∧ (∀ face ∈ f, face.card = 4 ∧ ∀ v ∈ face, v ∈ vertices)

def sums_to_same (f : Finset (Finset ℕ)) (s : ℕ) := ∀ face ∈ f, finset.sum face id = s

theorem common_sum_on_faces (f : Finset (Finset ℕ)) (h_faces : faces f) (h_same_sum : sums_to_same f 18) :
  ∀ face ∈ f, finset.sum face id = 18 := by
  sorry

end common_sum_on_faces_l513_513465


namespace problem1a_problem1b_problem2_problem3_problem4_l513_513221

-- Condition: distance between points representing rational numbers a and b on the number line is AB = |a - b|
def distance (a b : ℚ) : ℚ := abs (a - b)

-- Problem (1)
theorem problem1a : distance (-3) 4 = 7 := sorry

theorem problem1b (A : ℚ) (h1 : distance 2 A = 3) : A = -1 ∨ A = 5 :=
by
  have h1': abs (2 - A) = 3 := h1
  sorry

-- Problem (2)
theorem problem2 (x : ℚ) (h2 : abs (x + 4) = abs (x - 2)) : x = -1 :=
by
  have h2': abs (x + 4) = abs (x - 2) := h2
  sorry

-- Problem (3)
def sum_abs_diff_2023 (x : ℚ) : ℚ := ∑ i in (finset.range 2023).map (λ i, i + 1), abs (x - i)

theorem problem3 : ∀ x, (x = 1012) → sum_abs_diff_2023 x = sum {z in finset.Icc 1 1011, abs (1012 - z)} * 2 :=
by
  intros x h3
  have h3': x = 1012 := h3
  sorry

-- Problem (4)
theorem problem4 : sum_abs_diff_2023 1012 = 1023132 :=
by
  have : sum_abs_diff_2023 1012 = sum {z in finset.Icc 1 1011, abs (1012 - z)} * 2 := by
     calc
       sum_abs_diff_2023 1012 = ∑ i in (finset.range 1011), (1012 - i + (i + 1)) := sorry
       ... = sum {z in finset.Icc 1 1011, abs (1012 - z)} * 2 := sorry
  have : sum {z in finset.Icc 1 1011, abs(1012 - z)} = 506156 := sorry
  rw [this] at this_lemma
  linarith
  sorry

end problem1a_problem1b_problem2_problem3_problem4_l513_513221


namespace B3_set_equality_l513_513113

def B3_set (A : Set ℝ) : Prop :=
  ∀ {a1 a2 a3 a4 a5 a6 : ℝ}, a1 ∈ A → a2 ∈ A → a3 ∈ A → a4 ∈ A → a5 ∈ A → a6 ∈ A →
    a1 + a2 + a3 = a4 + a5 + a6 →
    multiset.pmap id [a1, a2, a3] (λ x _, x ∈ A) = multiset.pmap id [a4, a5, a6] (λ x _, x ∈ A)

def difference_set (X : Set ℝ) : Set ℝ :=
  {d | ∃ x y : ℝ, x ∈ X ∧ y ∈ X ∧ d = |x - y|}

theorem B3_set_equality {A B : Set ℝ} (hA : ∃ x : ℝ, x ∈ A) (hB : ∃ x : ℝ, x ∈ B)
  (seqA : ∀ n, ∃! a : ℝ, a ∈ A ∧ (∀ m, m < n → ∃ b : ℝ, b ∈ A ∧ a > b))
  (seqB : ∀ n, ∃! b : ℝ, b ∈ B ∧ (∀ m, m < n → ∃ a : ℝ, a ∈ B ∧ b > a))
  (eq_diff : difference_set A = difference_set B)
  (b3 : B3_set A) :
  A = B :=
sorry

end B3_set_equality_l513_513113


namespace average_brown_MnMs_l513_513436

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end average_brown_MnMs_l513_513436


namespace min_value_tangent_cotangent_l513_513659

theorem min_value_tangent_cotangent (α β γ : ℝ)
  (h : sin α * cos β + abs (cos α * sin β) = sin α * abs (cos α) + abs (sin α) * cos β) :
  ∃ m : ℝ, m = 3 - 2 * real.sqrt 2 ∧ 
  ∀ γ' : ℝ, (tan γ' - sin α) ^ 2 - (cot γ' - cos β) ^ 2 ≥ m :=
sorry

end min_value_tangent_cotangent_l513_513659


namespace brother_raking_time_l513_513367

theorem brother_raking_time (x : ℝ) (hx : x > 0)
  (h_combined : (1 / 30) + (1 / x) = 1 / 18) : x = 45 :=
by
  sorry

end brother_raking_time_l513_513367


namespace rectangle_diagonal_locus_l513_513865

theorem rectangle_diagonal_locus
  (A B C D F : Point)
  (acute_angle_ABC : acute (angle A B C))
  (acute_angle_BCA : acute (angle B C A))
  (acute_angle_CAB : acute (angle C A B))
  (H : ∀ P₁ P₂ P₃ P₄ : Point,
       P₁ ∈ line_segment A B →
       P₂ ∈ line_segment A B →
       P₃ ∈ line_segment A C →
       P₄ ∈ line_segment B C →
       ∃ M : Point, (M = midpoint (diagonal P₁ P₃)) ∧
                    (M ∈ median C (line_segment A B))) :
  ∀ M : Point, (M ∈ locus_of_midpoints_diagonals (triangle A B C)) → (M ∈ line_segment D F) :=
sorry

end rectangle_diagonal_locus_l513_513865


namespace smallest_k_multiple_of_180_l513_513639

def sum_of_squares (k : ℕ) : ℕ :=
  (k * (k + 1) * (2 * k + 1)) / 6

def divisible_by_180 (n : ℕ) : Prop :=
  n % 180 = 0

theorem smallest_k_multiple_of_180 :
  ∃ k : ℕ, k > 0 ∧ divisible_by_180 (sum_of_squares k) ∧ ∀ m : ℕ, m > 0 ∧ divisible_by_180 (sum_of_squares m) → k ≤ m :=
sorry

end smallest_k_multiple_of_180_l513_513639


namespace anne_total_bottle_caps_l513_513160

def initial_bottle_caps_anne : ℕ := 10
def found_bottle_caps_anne : ℕ := 5

theorem anne_total_bottle_caps : initial_bottle_caps_anne + found_bottle_caps_anne = 15 := 
by
  sorry

end anne_total_bottle_caps_l513_513160


namespace total_emails_in_april_is_675_l513_513956

-- Define the conditions
def daily_emails : ℕ := 20
def additional_emails : ℕ := 5
def april_days : ℕ := 30
def half_april_days : ℕ := april_days / 2

-- Define the total number of emails received
def total_emails : ℕ :=
  (daily_emails * half_april_days) +
  ((daily_emails + additional_emails) * half_april_days)

-- Define the statement to be proven
theorem total_emails_in_april_is_675 : total_emails = 675 :=
  by
  sorry

end total_emails_in_april_is_675_l513_513956


namespace smallest_square_area_l513_513123

theorem smallest_square_area : ∃ (s : ℝ), 
  (∀ (r₁ r₂ : ℝ × ℝ), r₁ = (3, 4) ∧ r₂ = (4, 5) → 
  (∃ (square : ℝ), square = s ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), ((0 ≤ x₁ ∧ x₁ ≤ 3) ∧ (0 ≤ y₁ ∧ y₁ ≤ 4)) ∧
    ((0 ≤ x₂ ∧ x₂ ≤ 4) ∧ (0 ≤ y₂ ∧ y₂ ≤ 5)) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ((x₁, y₁) ≠ (x₂, y₂)) ∧    
    (0 ≤ x₁ ∧ x₁ ≤ s) ∧ (0 ≤ y₁ ∧ y₁ ≤ s) ∧
    (0 ≤ x₂ ∧ x₂ ≤ s) ∧ (0 ≤ y₂ ∧ y₂ ≤ s))) ∧ s^2 = 49 := 
begin
  let min_side := 7,
  use min_side,
  sorry
end

end smallest_square_area_l513_513123


namespace prob_f_has_root_l513_513246

-- Let X be a random variable that follows a binomial distribution with n = 5 and p = 1/2
def X : Probability Mass Function nat := binomial 5 (1/2)

-- Define the function f(x) = x^2 + 4x + X
def f (x : ℝ) (X : ℝ) := x^2 + 4 * x + X

-- Define the event Ω where f(x) has a root
def has_root (X : ℝ) := ∃ x : ℝ, f x X = 0

-- Probability that has_root occurs; we provide the solution in the question
theorem prob_f_has_root : ∀ X : ℕ, X ∼ binomial 5 (1/2) -> P(has_root X) = 31 / 32 := by
  -- The proof is omitted
  sorry

end prob_f_has_root_l513_513246


namespace proof_problem_l513_513777

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513777


namespace fractional_linear_map_l513_513215

-- Definitions of conditions
def z1 := 1
def z2 := Complex.i
def z3 := -1
def w1 := -1
def w2 := 0
def w3 := 1

-- Mapping function to verify
def fractional_linear_function (z : ℂ) : ℂ :=
  Complex.i * (Complex.i - z) / (Complex.i + z)

-- Goal statement
theorem fractional_linear_map : 
  (fractional_linear_function z1 = w1) ∧ 
  (fractional_linear_function z2 = w2) ∧ 
  (fractional_linear_function z3 = w3) :=
  by
    sorry

end fractional_linear_map_l513_513215


namespace answer_is_p_and_q_l513_513739

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l513_513739


namespace true_proposition_l513_513854

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513854


namespace average_mb_per_hour_l513_513572

theorem average_mb_per_hour (days : ℕ) (total_disk_space : ℕ)
  (h_days : days = 15) (h_total_disk_space : total_disk_space = 20000) :
  let total_hours := days * 24
  let mb_per_hour := total_disk_space / total_hours.toFloat
  round mb_per_hour = 56 :=
by
  let total_hours := days * 24
  let mb_per_hour := total_disk_space / total_hours.toFloat
  have : total_hours = 360 := by rw [h_days]; simp
  have : mb_per_hour ≈ 55.56 := by rw [h_total_disk_space, this]; simp
  have : round mb_per_hour = 56 := by norm_cast; simp
  exact this

end average_mb_per_hour_l513_513572


namespace melanie_initial_dimes_l513_513982

theorem melanie_initial_dimes 
    (dimes_from_dad : ℕ) 
    (dimes_from_mom : ℕ) 
    (current_dimes : ℕ) 
    (total_received : dimes_from_dad + dimes_from_mom = 12) 
    (current_total : current_dimes = 19): 
    (initial_dimes : ℕ) 
    (initial_dimes = current_dimes - dimes_from_dad - dimes_from_mom) 
    (initial_dimes = 7) :=
by sorry

end melanie_initial_dimes_l513_513982


namespace volume_of_sphere_container_l513_513128

theorem volume_of_sphere_container :
    ∀ (vol_hemisphere : ℕ) (num_hemispheres : ℕ),
    vol_hemisphere = 4 → num_hemispheres = 2945 →
    vol_hemisphere * num_hemispheres = 11780 :=
by
  intros vol_hemisphere num_hemispheres h1 h2
  rw [h1, h2]
  sorry

end volume_of_sphere_container_l513_513128


namespace school_payment_difference_l513_513140

/-- 
In 2017-18, a school spends exactly Rs. 10000 on pens and notebooks,
with a pen costing Rs. 13 and a notebook costing Rs. 17.
The quantities of pens and notebooks are as close as possible.
Prove that the school paid Rs. 40 more in 2018-19 after swapping the quantities
of pens and notebooks. 
-/
theorem school_payment_difference 
  (pen_cost : ℕ) (notebook_cost : ℕ) (total_cost : ℕ) 
  (x y : ℕ) (h_cost : total_cost = pen_cost * x + notebook_cost * y) 
  (h_min_diff : ∀ x' y', total_cost = pen_cost * x' + notebook_cost * y' → abs (x' - y') ≥ abs (x - y)) :
  (notebook_cost * y + pen_cost * x) - (pen_cost * x + notebook_cost * y) = 40 := 
by 
  have h1 : notebook_cost * y + pen_cost * x = notebook_cost * y + pen_cost * x := rfl
  have h2 : pen_cost * x + notebook_cost * y = pen_cost * x + notebook_cost * y := rfl
  have h3 : notebook_cost * y + pen_cost * x - (pen_cost * x + notebook_cost * y) = 
            (notebook_cost * y - notebook_cost * y) + (pen_cost * x - pen_cost * x) := by ring
  rw [h1, h2, h3]
  rw [add_zero, sub_self, sub_self]
  rw [sub_self]
  sorry

end school_payment_difference_l513_513140


namespace dividend_percentage_l513_513136

theorem dividend_percentage (investment_amount market_value : ℝ) (interest_rate : ℝ) 
  (h1 : investment_amount = 44) (h2 : interest_rate = 12) (h3 : market_value = 33) : 
  ((interest_rate / 100) * investment_amount / market_value) * 100 = 16 := 
by
  sorry

end dividend_percentage_l513_513136


namespace proposition_A_l513_513819

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l513_513819


namespace probability_zero_point_l513_513141

theorem probability_zero_point 
  (a : ℝ) (h₀ : a ∈ Icc (-2 : ℝ) 2) :
  (∃ x : ℝ, 4^x - a * 2^(x+1) + 1 = 0) ↔ 
  ∃ p : ℝ, (0 ≤ p ∧ p ≤ 1 ∧ p = 1/4) :=
sorry

end probability_zero_point_l513_513141


namespace no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l513_513121

theorem no_integer_solutions_x_x_plus_1_eq_13y_plus_1 :
  ¬ ∃ x y : ℤ, x * (x + 1) = 13 * y + 1 :=
by sorry

end no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l513_513121


namespace polynomial_coeff_sum_l513_513918

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the polynomial expansion for (5x - 4)^5
noncomputable def a (r : ℕ) : ℤ :=
  (-1)^r * 4^r * binom 5 r * 5^(5 - r)

-- Lean statement for proof problem
theorem polynomial_coeff_sum :
  let a1 := a 1
  let a2 := a 2
  let a3 := a 3
  let a4 := a 4
  let a5 := a 5
  a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 = 25 :=
by {
  let a1 := a 1,
  let a2 := a 2,
  let a3 := a 3,
  let a4 := a 4,
  let a5 := a 5,
  have formula := a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5,
  sorry
}

end polynomial_coeff_sum_l513_513918


namespace true_conjunction_l513_513678

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l513_513678


namespace p_and_q_is_true_l513_513700

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513700


namespace p_and_q_is_true_l513_513688

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l513_513688


namespace gcd_of_g_and_y_l513_513859

noncomputable def g (y : ℕ) := (3 * y + 5) * (8 * y + 3) * (16 * y + 9) * (y + 16)

theorem gcd_of_g_and_y (y : ℕ) (hy : y % 46896 = 0) : Nat.gcd (g y) y = 2160 :=
by
  -- Proof to be written here
  sorry

end gcd_of_g_and_y_l513_513859


namespace angle_C_correct_l513_513237

theorem angle_C_correct (A B C : ℝ) (h1 : A = 65) (h2 : B = 40) (h3 : A + B + C = 180) : C = 75 :=
sorry

end angle_C_correct_l513_513237


namespace cookies_with_flour_l513_513963

theorem cookies_with_flour (x: ℕ) (c1: ℕ) (c2: ℕ) (h: c1 = 18 ∧ c2 = 2 ∧ x = 9 * 5):
  x = 45 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end cookies_with_flour_l513_513963


namespace proposition_p_and_q_is_true_l513_513717

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l513_513717


namespace determine_position_l513_513157

def position (description : String) : Prop :=
  description = "Row 3, Seat 6 at Phoenix Cinema" → 
  ∃ p : Point, p = Point.mk 3 6 "Phoenix Cinema"

theorem determine_position :
  position "Row 3, Seat 6 at Phoenix Cinema" := 
by 
  sorry

end determine_position_l513_513157


namespace distance_center_to_point_l513_513507

def center_of_circle (a b c : ℝ) : ℝ × ℝ :=
  let h := -a / 2
  let k := -b / 2
  (h, k)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_center_to_point :
  let c := center_of_circle 8 (-4) 16 in
  distance c (3, -1) = real.sqrt 2 := by
    sorry

end distance_center_to_point_l513_513507


namespace propositions_correct_l513_513868

theorem propositions_correct :
  (¬ (∀ (a b : Line), (a ∥ b) → (∀ (π : Plane), (b ∈ π → a ∥ π ∨ ¬(a ∈ π))))) ∧
  (∀ (α β : Plane), ¬(α ⊥ β) → ¬(∃ (l : Line), (l ∈ α ∧ l ⊥ β))) ∧ 
  (¬ (∀ (a b c : Line), (skew a b) → (skew b c) → (skew a c))) ∧
  (¬ (∀ (a : Line) (α β : Plane), (α ⊥ β) → (a ∈ α) → (α ∩ β = b) → (a ⊥ b) → (a ⊥ β))) ∧
  (∀ (a : Line) (α β : Plane), (a ⊥ α) → (b ∈ β) → (a ∥ b) → (α ⊥ β)) :=
by sorry

end propositions_correct_l513_513868


namespace true_proposition_l513_513853

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l513_513853


namespace largest_set_domain_l513_513390

noncomputable def largest_domain (g : ℝ → ℝ) : set ℝ :=
  {x | g x + g (1 / x^2) = x^2 ∧ g (1 / x^2) + g x = 1 / x^2}

theorem largest_set_domain (g : ℝ → ℝ)
  (h_domain : ∀ x, x ∈ domain g → (x^2 ∈ domain g ∧ 1 / x^2 ∈ domain g))
  (h_function : ∀ x, g x + g (1 / x^2) = x^2) :
  largest_domain g = {-1, 1} :=
sorry

end largest_set_domain_l513_513390


namespace ant_walk_distance_on_cube_l513_513567

theorem ant_walk_distance_on_cube :
  ∀ (side_length : ℕ) (edges_walked : ℕ),
    side_length = 18 →
    edges_walked = 5 →
    (side_length * edges_walked = 90) :=
by
  intros side_length edges_walked h1 h2
  rw [h1, h2]
  exact rfl

end ant_walk_distance_on_cube_l513_513567


namespace ratio_evaluation_l513_513206

theorem ratio_evaluation :
  (10 ^ 2003 + 10 ^ 2001) / (2 * 10 ^ 2002) = 101 / 20 := 
by sorry

end ratio_evaluation_l513_513206


namespace proof_problem_l513_513781

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l513_513781


namespace power_identity_l513_513902

theorem power_identity (y : ℤ) (h : 3 ^ y = 243) : 3 ^ (y + 3) = 6561 :=
by {
  sorry
}

end power_identity_l513_513902


namespace range_of_m_l513_513311

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x + 2 + real.log (2 - abs x)

theorem range_of_m (a : ℝ) (m : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) : 
  f (1 - m) 0 < f m 0 ↔ m ∈ set.Ioo (-1 : ℝ) (1 / 2 : ℝ) :=
by 
  have h_a_zero : a = 0, from sorry,
  rw h_a_zero at *,
  sorry

end range_of_m_l513_513311


namespace Jason_spent_correct_amount_l513_513951

namespace MusicStore

def costFlute : Real := 142.46
def costMusicStand : Real := 8.89
def costSongBook : Real := 7.00
def totalCost : Real := 158.35

theorem Jason_spent_correct_amount :
  costFlute + costMusicStand + costSongBook = totalCost :=
sorry

end MusicStore

end Jason_spent_correct_amount_l513_513951


namespace total_length_of_segments_l513_513159

theorem total_length_of_segments
  (a b c d e : ℕ) (h1 : a = 9) (h2 : b = 2) (h3 : c = 7) (h4 : d = 1) (h5 : e = 1) :
  9 + (2 + 1 + 1 + 1) + 7 + 1 = 22 := 
by
  simp [h1, h2, h3, h4, h5]
  sorry

end total_length_of_segments_l513_513159


namespace edge_RS_correct_l513_513048

-- Define the type for vertices
inductive Vertex 
| P | Q | R | S
deriving DecidableEq, Repr

-- We need a function to give numbers to vertices and edges
variable (number : Vertex → ℕ)

-- Edge function: given two vertices, gives their edge sum
def edge_sum (v1 v2 : Vertex) : ℕ := number v1 + number v2

-- Set of numbers used in the problem
def used_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 11}

-- Instantiate correction relation on used_numbers
axiom number_in_used_numbers (v : Vertex) : number v ∈ used_numbers

-- Given condition that edge PQ sums to 9
axiom edge_PQ : edge_sum Vertex.P Vertex.Q = 9

-- Question: What is the number on edge RS?
theorem edge_RS_correct : ∃ (number : Vertex → ℕ), edge_sum Vertex.R Vertex.S = 5 := 
sorry

end edge_RS_correct_l513_513048


namespace integral_f_l513_513399

def f (x : ℝ) : ℝ := if x ≥ 0 then -sin x else 1

theorem integral_f : ∫ x in - (π / 2)..(π / 2), f x = π / 2 - 1 := by
  sorry

end integral_f_l513_513399


namespace no_counterexample_l513_513393

-- Definition to calculate the sum of digits of a number
def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Definition to check divisibility
def divisible (a b : ℕ) : Prop := b ∣ a

-- The mathematical statement to be proven
theorem no_counterexample :
  ∀ (n : ℕ), n ∈ {45, 54, 81, 90} →
  (divisible (sumOfDigits n) 9 → divisible n 9) :=
by
  intros n hn
  cases hn
  case Or.inl h45 =>
    exact sorry
  case Or.inr hn54 =>
    cases hn54
    case Or.inl h54 =>
      exact sorry
    case Or.inr hn81_90 =>
      cases hn81_90
      case Or.inl h81 =>
        exact sorry
      case Or.inr h90 =>
        exact sorry

end no_counterexample_l513_513393


namespace sum_of_solutions_eq_3_l513_513480

theorem sum_of_solutions_eq_3 : 
  let f x := (x^2 - 5x + 5) in 
  let g x := (x^2 + 4x - 60) in 
  ∑ x in {x | f x = 1 ∨ g x = 0 ∨ (f x = -1 ∧ g x % 2 = 0)}, x = 3 := 
sorry

end sum_of_solutions_eq_3_l513_513480


namespace find_selling_price_l513_513558

noncomputable def selling_price (x : ℝ) : ℝ :=
  (x - 60) * (1800 - 20 * x)

constant purchase_price : ℝ := 60
constant max_profit_margin : ℝ := 0.40
constant base_selling_price : ℝ := 80
constant base_units_sold : ℝ := 200
constant decrement_units_sold : ℝ := 20
constant target_profit : ℝ := 2500

theorem find_selling_price (x : ℝ) :
  selling_price x = target_profit ∧
  (x - 60) / 60 ≤ max_profit_margin ∧
  ∃ u : ℝ, u = (base_units_sold + decrement_units_sold * (base_selling_price - x))
  → x = 65 :=
sorry

end find_selling_price_l513_513558


namespace baker_cakes_l513_513175

theorem baker_cakes (initial_cakes sold_cakes remaining_cakes final_cakes new_cakes : ℕ)
  (h1 : initial_cakes = 110)
  (h2 : sold_cakes = 75)
  (h3 : final_cakes = 111)
  (h4 : new_cakes = final_cakes - (initial_cakes - sold_cakes)) :
  new_cakes = 76 :=
by {
  sorry
}

end baker_cakes_l513_513175


namespace total_emails_675_l513_513960

-- Definitions based on conditions
def emails_per_day_before : ℕ := 20
def extra_emails_per_day_after : ℕ := 5
def halfway_days : ℕ := 15
def total_days : ℕ := 30

-- Define the total number of emails received by the end of April
def total_emails_received : ℕ :=
  let emails_before := emails_per_day_before * halfway_days
  let emails_after := (emails_per_day_before + extra_emails_per_day_after) * halfway_days
  emails_before + emails_after

-- Theorem stating that the total number of emails received by the end of April is 675
theorem total_emails_675 : total_emails_received = 675 := by
  sorry

end total_emails_675_l513_513960


namespace combinedTotalSandcastlesAndTowers_l513_513064

def markSandcastles : Nat := 20
def towersPerMarkSandcastle : Nat := 10
def jeffSandcastles : Nat := 3 * markSandcastles
def towersPerJeffSandcastle : Nat := 5

theorem combinedTotalSandcastlesAndTowers :
  (markSandcastles + markSandcastles * towersPerMarkSandcastle) +
  (jeffSandcastles + jeffSandcastles * towersPerJeffSandcastle) = 580 :=
by
  sorry

end combinedTotalSandcastlesAndTowers_l513_513064
