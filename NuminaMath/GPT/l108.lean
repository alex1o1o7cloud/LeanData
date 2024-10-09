import Mathlib

namespace cone_height_from_sphere_l108_10858

theorem cone_height_from_sphere (d_sphere d_base : ℝ) (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) 
  (h₁ : d_sphere = 6) 
  (h₂ : d_base = 12)
  (h₃ : V_sphere = 36 * Real.pi)
  (h₄ : V_cone = (1/3) * Real.pi * (d_base / 2)^2 * h) 
  (h₅ : V_sphere = V_cone) :
  h = 3 := by
  sorry

end cone_height_from_sphere_l108_10858


namespace find_arith_seq_sum_l108_10809

noncomputable def arith_seq_sum : ℕ → ℕ → ℕ
| 0, d => 2
| (n+1), d => arith_seq_sum n d + d

theorem find_arith_seq_sum :
  ∃ d : ℕ, 
    arith_seq_sum 1 d + arith_seq_sum 2 d = 13 ∧
    arith_seq_sum 3 d + arith_seq_sum 4 d + arith_seq_sum 5 d = 42 :=
by
  sorry

end find_arith_seq_sum_l108_10809


namespace denominator_of_second_fraction_l108_10881

theorem denominator_of_second_fraction (y x : ℝ) (h_cond : y > 0) (h_eq : (y / 20) + (3 * y / x) = 0.35 * y) : x = 10 :=
sorry

end denominator_of_second_fraction_l108_10881


namespace limo_cost_is_correct_l108_10852

def prom_tickets_cost : ℕ := 2 * 100
def dinner_cost : ℕ := 120
def dinner_tip : ℕ := (30 * dinner_cost) / 100
def total_cost_before_limo : ℕ := prom_tickets_cost + dinner_cost + dinner_tip
def total_cost : ℕ := 836
def limo_hours : ℕ := 6
def limo_total_cost : ℕ := total_cost - total_cost_before_limo
def limo_cost_per_hour : ℕ := limo_total_cost / limo_hours

theorem limo_cost_is_correct : limo_cost_per_hour = 80 := 
by
  sorry

end limo_cost_is_correct_l108_10852


namespace parallel_vectors_implies_value_of_x_l108_10804

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the condition for parallel vectors
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (u.1 = k * v.1) ∧ (u.2 = k * v.2)

-- The proof statement
theorem parallel_vectors_implies_value_of_x : ∀ (x : ℝ), parallel a (b x) → x = 6 :=
by
  intro x
  intro h
  sorry

end parallel_vectors_implies_value_of_x_l108_10804


namespace total_goals_l108_10826

theorem total_goals (B M : ℕ) (hB : B = 4) (hM : M = 3 * B) : B + M = 16 := by
  sorry

end total_goals_l108_10826


namespace min_unit_cubes_intersect_all_l108_10842

theorem min_unit_cubes_intersect_all (n : ℕ) : 
  let A_n := if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2
  A_n = if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2 :=
sorry

end min_unit_cubes_intersect_all_l108_10842


namespace jeffs_mean_l108_10831

-- Define Jeff's scores as a list or array
def jeffsScores : List ℚ := [86, 94, 87, 96, 92, 89]

-- Prove that the arithmetic mean of Jeff's scores is 544 / 6
theorem jeffs_mean : (jeffsScores.sum / jeffsScores.length) = (544 / 6) := by
  sorry

end jeffs_mean_l108_10831


namespace math_problem_l108_10872

variable (a b c : ℝ)

theorem math_problem 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ Real.sqrt 3 := 
by
  sorry

end math_problem_l108_10872


namespace max_intersections_between_quadrilateral_and_pentagon_l108_10891

-- Definitions based on the conditions
def quadrilateral_sides : ℕ := 4
def pentagon_sides : ℕ := 5

-- Theorem statement based on the problem
theorem max_intersections_between_quadrilateral_and_pentagon 
  (qm_sides : ℕ := quadrilateral_sides) 
  (pm_sides : ℕ := pentagon_sides) : 
  (∀ (n : ℕ), n = qm_sides →
    ∀ (m : ℕ), m = pm_sides →
      ∀ (intersection_points : ℕ), 
        intersection_points = (n * m) →
        intersection_points = 20) :=
sorry

end max_intersections_between_quadrilateral_and_pentagon_l108_10891


namespace moles_of_water_produced_l108_10899

theorem moles_of_water_produced (H₃PO₄ NaOH NaH₂PO₄ H₂O : ℝ) (h₁ : H₃PO₄ = 3) (h₂ : NaOH = 3) (h₃ : NaH₂PO₄ = 3) (h₄ : NaH₂PO₄ / H₂O = 1) : H₂O = 3 :=
by
  sorry

end moles_of_water_produced_l108_10899


namespace simplify_complex_subtraction_l108_10819

-- Definition of the nested expression
def complex_subtraction (x : ℝ) : ℝ :=
  1 - (2 - (3 - (4 - (5 - (6 - x)))))

-- Statement of the theorem to be proven
theorem simplify_complex_subtraction (x : ℝ) : complex_subtraction x = x - 3 :=
by {
  -- This proof will need to be filled in to verify the statement
  sorry
}

end simplify_complex_subtraction_l108_10819


namespace rectangular_prism_width_l108_10818

variables (w : ℝ)

theorem rectangular_prism_width (h : ℝ) (l : ℝ) (d : ℝ) (hyp_l : l = 5) (hyp_h : h = 7) (hyp_d : d = 15) :
  w = Real.sqrt 151 :=
by 
  -- Proof goes here
  sorry

end rectangular_prism_width_l108_10818


namespace jakes_present_weight_l108_10897

theorem jakes_present_weight:
  ∃ J S : ℕ, J - 15 = 2 * S ∧ J + S = 132 ∧ J = 93 :=
by
  sorry

end jakes_present_weight_l108_10897


namespace jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l108_10828

/-- Conditions:
1. The grandmother decorates five gingerbread cookies for every cycle.
2. Little Mary decorates three gingerbread cookies for every cycle.
3. Little John decorates two gingerbread cookies for every cycle.
4. All three together decorated five trays, with each tray holding twelve gingerbread cookies.
5. Little John also sorted the gingerbread cookies onto trays twelve at a time and carried them to the pantry.
6. The grandmother decorates one gingerbread cookie in four minutes.
-/

def decorated_cookies_per_cycle := 10
def total_trays := 5
def cookies_per_tray := 12
def total_cookies := total_trays * cookies_per_tray
def babicka_cookies_per_cycle := 5
def marenka_cookies_per_cycle := 3
def jenicek_cookies_per_cycle := 2
def babicka_time_per_cookie := 4

theorem jenicek_decorated_cookies :
  (total_cookies - (total_cookies / decorated_cookies_per_cycle * marenka_cookies_per_cycle + total_cookies / decorated_cookies_per_cycle * babicka_cookies_per_cycle)) = 4 :=
sorry

theorem total_time_for_work :
  (total_cookies / decorated_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 140 :=
sorry

theorem jenicek_decorating_time :
  (4 / jenicek_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 40 :=
sorry

end jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l108_10828


namespace perp_line_eq_l108_10856

theorem perp_line_eq (x y : ℝ) (h1 : (x, y) = (1, 1)) (h2 : y = 2 * x) :
  ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 :=
by 
  sorry

end perp_line_eq_l108_10856


namespace sum_of_digits_of_largest_n_l108_10832

def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_single_digit_prime (p : ℕ) : Prop := is_prime p ∧ p < 10

noncomputable def required_n (d e : ℕ) : ℕ := d * e * (d^2 + 10 * e)

def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n 
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_largest_n : 
  ∃ (d e : ℕ), 
    is_single_digit_prime d ∧ is_single_digit_prime e ∧ 
    is_prime (d^2 + 10 * e) ∧ 
    (∀ d' e' : ℕ, is_single_digit_prime d' ∧ is_single_digit_prime e' ∧ is_prime (d'^2 + 10 * e') → required_n d e ≥ required_n d' e') ∧ 
    sum_of_digits (required_n d e) = 9 :=
sorry

end sum_of_digits_of_largest_n_l108_10832


namespace reciprocal_of_5_over_7_l108_10807

theorem reciprocal_of_5_over_7 : (5 / 7 : ℚ) * (7 / 5) = 1 := by
  sorry

end reciprocal_of_5_over_7_l108_10807


namespace min_number_of_each_coin_l108_10802

def total_cost : ℝ := 1.30 + 0.75 + 0.50 + 0.45

def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50

def min_coins :=
  ∃ (n q d h : ℕ), 
  (n ≥ 1) ∧ (q ≥ 1) ∧ (d ≥ 1) ∧ (h ≥ 1) ∧ 
  ((n * nickel_value) + (q * quarter_value) + (d * dime_value) + (h * half_dollar_value) = total_cost)

theorem min_number_of_each_coin :
  min_coins ↔ (5 * half_dollar_value + 1 * quarter_value + 2 * dime_value + 1 * nickel_value = total_cost) :=
by sorry

end min_number_of_each_coin_l108_10802


namespace evaluate_polynomial_at_minus_two_l108_10857

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

theorem evaluate_polynomial_at_minus_two :
  P (-2) = -18 :=
by
  sorry

end evaluate_polynomial_at_minus_two_l108_10857


namespace peter_age_fraction_l108_10855

theorem peter_age_fraction 
  (harriet_age : ℕ) 
  (mother_age : ℕ) 
  (peter_age_plus_four : ℕ) 
  (harriet_age_plus_four : ℕ) 
  (harriet_age_current : harriet_age = 13)
  (mother_age_current : mother_age = 60)
  (peter_age_condition : peter_age_plus_four = 2 * harriet_age_plus_four)
  (harriet_four_years : harriet_age_plus_four = harriet_age + 4)
  (peter_four_years : ∀ P : ℕ, peter_age_plus_four = P + 4)
: ∃ P : ℕ, P = 30 ∧ P = mother_age / 2 := 
sorry

end peter_age_fraction_l108_10855


namespace pascal_triangle_row51_sum_l108_10811

theorem pascal_triangle_row51_sum : (Nat.choose 51 4) + (Nat.choose 51 6) = 18249360 :=
by
  sorry

end pascal_triangle_row51_sum_l108_10811


namespace correct_choice_d_l108_10836

def is_quadrant_angle (alpha : ℝ) (k : ℤ) : Prop :=
  2 * k * Real.pi - Real.pi / 2 < alpha ∧ alpha < 2 * k * Real.pi

theorem correct_choice_d (alpha : ℝ) (k : ℤ) :
  is_quadrant_angle alpha k ↔ (2 * k * Real.pi - Real.pi / 2 < alpha ∧ alpha < 2 * k * Real.pi) := by
sorry

end correct_choice_d_l108_10836


namespace rhombus_area_l108_10882

theorem rhombus_area (d₁ d₂ : ℕ) (h₁ : d₁ = 6) (h₂ : d₂ = 8) : 
  (1 / 2 : ℝ) * d₁ * d₂ = 24 := 
by
  sorry

end rhombus_area_l108_10882


namespace problem_l108_10815

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4)) ^ 6 + (Real.cos (x / 4)) ^ 6

theorem problem : (derivative^[2008] f 0) = 3 / 8 := by sorry

end problem_l108_10815


namespace hyperbola_eccentricity_a_l108_10860

theorem hyperbola_eccentricity_a (a : ℝ) (ha : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1) ∧ (∃ (e : ℝ), e = 2 ∧ e = Real.sqrt (a^2 + 3) / a) → a = 1 :=
by
  sorry

end hyperbola_eccentricity_a_l108_10860


namespace complement_A_complement_U_range_of_a_empty_intersection_l108_10889

open Set Real

noncomputable def complement_A_in_U := { x : ℝ | ¬ (x < -1 ∨ x > 3) }

theorem complement_A_complement_U
  {A : Set ℝ} (hA : A = {x | x^2 - 2 * x - 3 > 0}) :
  (complement_A_in_U = (Icc (-1) 3)) :=
by sorry

theorem range_of_a_empty_intersection
  {B : Set ℝ} {a : ℝ}
  (hB : B = {x | abs (x - a) > 3})
  (h_empty : (Icc (-1) 3) ∩ B = ∅) :
  (0 ≤ a ∧ a ≤ 2) :=
by sorry

end complement_A_complement_U_range_of_a_empty_intersection_l108_10889


namespace number_of_female_only_child_students_l108_10875

def students : Finset ℕ := Finset.range 21 -- Set of students with attendance numbers from 1 to 20

def female_students : Finset ℕ := {1, 3, 4, 6, 7, 10, 11, 13, 16, 17, 18, 20}

def only_child_students : Finset ℕ := {1, 4, 5, 8, 11, 14, 17, 20}

def common_students : Finset ℕ := female_students ∩ only_child_students

theorem number_of_female_only_child_students :
  common_students.card = 5 :=
by
  sorry

end number_of_female_only_child_students_l108_10875


namespace mark_buttons_l108_10837

theorem mark_buttons (initial_buttons : ℕ) (shane_buttons : ℕ) (sam_buttons : ℕ) :
  initial_buttons = 14 →
  shane_buttons = 3 * initial_buttons →
  sam_buttons = (initial_buttons + shane_buttons) / 2 →
  final_buttons = (initial_buttons + shane_buttons) - sam_buttons →
  final_buttons = 28 :=
by
  sorry

end mark_buttons_l108_10837


namespace max_value_of_expression_l108_10830

theorem max_value_of_expression (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 3) : 2 * x + 3 * y ≤ Real.sqrt 31 :=
sorry

end max_value_of_expression_l108_10830


namespace length_diff_width_8m_l108_10890

variables (L W : ℝ)

theorem length_diff_width_8m (h1: W = (1/2) * L) (h2: L * W = 128) : L - W = 8 :=
by sorry

end length_diff_width_8m_l108_10890


namespace baker_cakes_remaining_l108_10896

def InitialCakes : ℕ := 48
def SoldCakes : ℕ := 44
def RemainingCakes (initial sold : ℕ) : ℕ := initial - sold

theorem baker_cakes_remaining : RemainingCakes InitialCakes SoldCakes = 4 := 
by {
  -- placeholder for the proof
  sorry
}

end baker_cakes_remaining_l108_10896


namespace minimum_elapsed_time_l108_10835

theorem minimum_elapsed_time : 
  let initial_time := 45  -- in minutes
  let final_time := 3 * 60 + 30  -- 3 hours 30 minutes in minutes
  let elapsed_time := final_time - initial_time
  elapsed_time = 2 * 60 + 45 :=
by
  sorry

end minimum_elapsed_time_l108_10835


namespace find_function_value_at_2_l108_10848

variables {f : ℕ → ℕ}

theorem find_function_value_at_2 (H : ∀ x : ℕ, Nat.succ (Nat.succ x * Nat.succ x + f x) = 12) : f 2 = 4 :=
by
  sorry

end find_function_value_at_2_l108_10848


namespace dr_jones_remaining_salary_l108_10816

noncomputable def remaining_salary (salary rent food utilities insurances taxes transport emergency loan retirement : ℝ) : ℝ :=
  salary - (rent + food + utilities + insurances + taxes + transport + emergency + loan + retirement)

theorem dr_jones_remaining_salary :
  remaining_salary 6000 640 385 (1/4 * 6000) (1/5 * 6000) (0.10 * 6000) (0.03 * 6000) (0.02 * 6000) 300 (0.05 * 6000) = 1275 :=
by
  sorry

end dr_jones_remaining_salary_l108_10816


namespace smaller_root_of_equation_l108_10874

theorem smaller_root_of_equation :
  ∀ x : ℚ, (x - 7 / 8)^2 + (x - 1/4) * (x - 7 / 8) = 0 → x = 9 / 16 :=
by
  intro x
  intro h
  sorry

end smaller_root_of_equation_l108_10874


namespace find_fifth_number_l108_10801

-- Define the sets and their conditions
def first_set : List ℕ := [28, 70, 88, 104]
def second_set : List ℕ := [50, 62, 97, 124]

-- Define the means
def mean_first_set (x : ℕ) (y : ℕ) : ℚ := (28 + x + 70 + 88 + y) / 5
def mean_second_set (x : ℕ) : ℚ := (50 + 62 + 97 + 124 + x) / 5

-- Conditions given in the problem
axiom mean_first_set_condition (x y : ℕ) : mean_first_set x y = 67
axiom mean_second_set_condition (x : ℕ) : mean_second_set x = 75.6

-- Lean 4 theorem statement to prove the fifth number in the first set is 104 given above conditions
theorem find_fifth_number : ∃ x y, mean_first_set x y = 67 ∧ mean_second_set x = 75.6 ∧ y = 104 := by
  sorry

end find_fifth_number_l108_10801


namespace arrasta_um_proof_l108_10805

variable (n : ℕ)

def arrasta_um_possible_moves (n : ℕ) : ℕ :=
  6 * n - 8

theorem arrasta_um_proof (n : ℕ) (h : n ≥ 2) : arrasta_um_possible_moves n =
6 * n - 8 := by
  sorry

end arrasta_um_proof_l108_10805


namespace sin_zero_degrees_l108_10824

theorem sin_zero_degrees : Real.sin 0 = 0 := 
by {
  -- The proof is added here (as requested no proof is required, hence using sorry)
  sorry
}

end sin_zero_degrees_l108_10824


namespace brown_rabbit_hop_distance_l108_10863

theorem brown_rabbit_hop_distance
  (w : ℕ) (b : ℕ) (t : ℕ)
  (h1 : w = 15)
  (h2 : t = 135)
  (hop_distance_in_5_minutes : w * 5 + b * 5 = t) : 
  b = 12 :=
by
  sorry

end brown_rabbit_hop_distance_l108_10863


namespace distinct_ways_to_distribute_l108_10851

theorem distinct_ways_to_distribute :
  ∃ (n : ℕ), n = 7 ∧ ∀ (balls : ℕ) (boxes : ℕ)
    (indistinguishable_balls : Prop := true) 
    (indistinguishable_boxes : Prop := true), 
    balls = 6 → boxes = 3 → 
    indistinguishable_balls → 
    indistinguishable_boxes → 
    n = 7 :=
by
  sorry

end distinct_ways_to_distribute_l108_10851


namespace coordinates_of_M_l108_10894

-- Let M be a point in the 2D Cartesian plane
variable {x y : ℝ}

-- Definition of the conditions
def distance_from_x_axis (y : ℝ) : Prop := abs y = 1
def distance_from_y_axis (x : ℝ) : Prop := abs x = 2

-- Theorem to prove
theorem coordinates_of_M (hx : distance_from_y_axis x) (hy : distance_from_x_axis y) :
  (x = 2 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
sorry

end coordinates_of_M_l108_10894


namespace identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l108_10849

-- Proving the identification of the counterfeit coin among 13 coins in 3 weighings
theorem identify_counterfeit_13_coins (coins : Fin 13 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0) :=
sorry

-- Proving counterfeit coin weight determination with an additional genuine coin using 3 weighings
theorem identify_and_determine_weight_14_coins (coins : Fin 14 → Real) (genuine : Real) (is_counterfeit : ∃! i, coins i ≠ genuine) :
  ∃ method_exists : Prop, 
    (method_exists ∧ ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ (i : Fin 14), coins i ≠ genuine) :=
sorry

-- Proving the impossibility of identifying counterfeit coin among 14 coins using 3 weighings
theorem impossible_with_14_coins (coins : Fin 14 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ¬ (∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0)) :=
sorry

end identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l108_10849


namespace area_of_circle_l108_10864

theorem area_of_circle (C : ℝ) (hC : C = 30 * Real.pi) : ∃ k : ℝ, (Real.pi * (C / (2 * Real.pi))^2 = k * Real.pi) ∧ k = 225 :=
by
  sorry

end area_of_circle_l108_10864


namespace haley_deleted_pictures_l108_10854

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (remaining_pictures : ℕ) (deleted_pictures : ℕ)

theorem haley_deleted_pictures :
  zoo_pictures = 50 → museum_pictures = 8 → remaining_pictures = 20 →
  deleted_pictures = zoo_pictures + museum_pictures - remaining_pictures →
  deleted_pictures = 38 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end haley_deleted_pictures_l108_10854


namespace board_partition_possible_l108_10841

-- Definition of natural numbers m and n greater than 15
variables (m n : ℕ)
-- m > 15
def m_greater_than_15 := m > 15
-- n > 15
def n_greater_than_15 := n > 15

-- Definition of m and n divisibility conditions
def divisible_by_4_or_5 (x : ℕ) : Prop :=
  x % 4 = 0 ∨ x % 5 = 0

def partition_possible (m n : ℕ) : Prop :=
  (m % 4 = 0 ∧ n % 5 = 0) ∨ (m % 5 = 0 ∧ n % 4 = 0)

-- The final statement of Lean
theorem board_partition_possible :
  m_greater_than_15 m → n_greater_than_15 n → partition_possible m n :=
by
  intro h_m h_n
  sorry

end board_partition_possible_l108_10841


namespace scientific_notation_103M_l108_10829

theorem scientific_notation_103M : 103000000 = 1.03 * 10^8 := sorry

end scientific_notation_103M_l108_10829


namespace traveler_drank_32_ounces_l108_10859

-- Definition of the given condition
def total_gallons : ℕ := 2
def ounces_per_gallon : ℕ := 128
def total_ounces := total_gallons * ounces_per_gallon
def camel_multiple : ℕ := 7
def traveler_ounces (T : ℕ) := T
def camel_ounces (T : ℕ) := camel_multiple * T
def total_drunk (T : ℕ) := traveler_ounces T + camel_ounces T

-- Theorem to prove
theorem traveler_drank_32_ounces :
  ∃ T : ℕ, total_drunk T = total_ounces ∧ T = 32 :=
by 
  sorry

end traveler_drank_32_ounces_l108_10859


namespace chess_group_unique_pairings_l108_10838

theorem chess_group_unique_pairings:
  ∀ (players games : ℕ), players = 50 → games = 1225 →
  (∃ (games_per_pair : ℕ), games_per_pair = 1 ∧ (∀ p: ℕ, p < players → (players - 1) * games_per_pair = games)) :=
by
  sorry

end chess_group_unique_pairings_l108_10838


namespace incorrect_statement_D_l108_10803

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2

theorem incorrect_statement_D :
  (∃ T : ℝ, ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, f (π / 2 + x) = f (π / 2 - x)) ∧
  (f (π / 2 + π / 4) = 0) ∧ ¬(∀ x : ℝ, (π / 2 < x ∧ x < π) → f x < f (x - 0.1)) := by
  sorry

end incorrect_statement_D_l108_10803


namespace circumcircle_radius_of_right_triangle_l108_10817

theorem circumcircle_radius_of_right_triangle (a b c : ℝ) (h1: a = 8) (h2: b = 6) (h3: c = 10) (h4: a^2 + b^2 = c^2) : (c / 2) = 5 := 
by
  sorry

end circumcircle_radius_of_right_triangle_l108_10817


namespace group_friends_opponents_l108_10898

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l108_10898


namespace calculate_expression_l108_10876

theorem calculate_expression : 
  (3 * 7.5 * (6 + 4) / 2.5) = 90 := 
by
  sorry

end calculate_expression_l108_10876


namespace range_of_m_l108_10845

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m

theorem range_of_m (m : ℝ) : 
  satisfies_inequality m ↔ (m ≥ 4 ∨ m ≤ -1) :=
by
  sorry

end range_of_m_l108_10845


namespace roots_of_quadratic_solve_inequality_l108_10827

theorem roots_of_quadratic (a b : ℝ) (h1 : ∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :
  a = 1 ∧ b = 2 :=
by
  sorry

theorem solve_inequality (c : ℝ) :
  let a := 1
  let b := 2
  ∀ x : ℝ, a * x^2 - (a * c + b) * x + b * x < 0 ↔
    (c > 0 → (0 < x ∧ x < c)) ∧
    (c = 0 → false) ∧
    (c < 0 → (c < x ∧ x < 0)) :=
by
  sorry

end roots_of_quadratic_solve_inequality_l108_10827


namespace sally_quarters_l108_10825

theorem sally_quarters (initial_quarters spent_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 760) 
  (h2 : spent_quarters = 418) 
  (calc_final : final_quarters = initial_quarters - spent_quarters) : 
  final_quarters = 342 := 
by 
  rw [h1, h2] at calc_final 
  exact calc_final

end sally_quarters_l108_10825


namespace find_circles_tangent_to_axes_l108_10877

def tangent_to_axes_and_passes_through (R : ℝ) (P : ℝ × ℝ) :=
  let center := (R, R)
  (P.1 - R) ^ 2 + (P.2 - R) ^ 2 = R ^ 2

theorem find_circles_tangent_to_axes (x y : ℝ) :
  (tangent_to_axes_and_passes_through 1 (2, 1) ∧ tangent_to_axes_and_passes_through 1 (x, y)) ∨
  (tangent_to_axes_and_passes_through 5 (2, 1) ∧ tangent_to_axes_and_passes_through 5 (x, y)) :=
by {
  sorry
}

end find_circles_tangent_to_axes_l108_10877


namespace lowest_point_on_graph_l108_10808

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2 * x + 2) / (x + 1)

theorem lowest_point_on_graph : ∃ (x y : ℝ), x = 0 ∧ y = 2 ∧ ∀ z > -1, f z ≥ y ∧ f x = y := by
  sorry

end lowest_point_on_graph_l108_10808


namespace largest_mersenne_prime_lt_1000_l108_10861

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def is_mersenne_prime (n : ℕ) : Prop :=
  is_prime n ∧ ∃ p : ℕ, is_prime p ∧ n = 2^p - 1

theorem largest_mersenne_prime_lt_1000 : ∃ (n : ℕ), is_mersenne_prime n ∧ n < 1000 ∧ ∀ (m : ℕ), is_mersenne_prime m ∧ m < 1000 → m ≤ n :=
by
  -- Proof goes here
  sorry

end largest_mersenne_prime_lt_1000_l108_10861


namespace coin_and_die_probability_l108_10823

-- Probability of a coin showing heads
def P_heads : ℚ := 2 / 3

-- Probability of a die showing 5
def P_die_5 : ℚ := 1 / 6

-- Probability of both events happening together
def P_heads_and_die_5 : ℚ := P_heads * P_die_5

-- Theorem statement: Proving the calculated probability equals the expected value
theorem coin_and_die_probability : P_heads_and_die_5 = 1 / 9 := by
  -- The detailed proof is omitted here.
  sorry

end coin_and_die_probability_l108_10823


namespace jellybeans_left_l108_10844

theorem jellybeans_left :
  let initial_jellybeans := 500
  let total_kindergarten := 10
  let total_firstgrade := 10
  let total_secondgrade := 10
  let sick_kindergarten := 2
  let sick_secondgrade := 3
  let jellybeans_sick_kindergarten := 5
  let jellybeans_sick_secondgrade := 10
  let jellybeans_remaining_kindergarten := 3
  let jellybeans_firstgrade := 5
  let jellybeans_secondgrade_per_firstgrade := 5 / 2 * total_firstgrade
  let consumed_by_sick := sick_kindergarten * jellybeans_sick_kindergarten + sick_secondgrade * jellybeans_sick_secondgrade
  let remaining_kindergarten := total_kindergarten - sick_kindergarten
  let consumed_by_remaining := remaining_kindergarten * jellybeans_remaining_kindergarten + total_firstgrade * jellybeans_firstgrade + total_secondgrade * jellybeans_secondgrade_per_firstgrade
  let total_consumed := consumed_by_sick + consumed_by_remaining
  initial_jellybeans - total_consumed = 176 := by 
  sorry

end jellybeans_left_l108_10844


namespace find_inverse_l108_10893

theorem find_inverse :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 3 * x ^ 3 + 9) → (f⁻¹ 90 = 3) :=
by
  intros f h
  sorry

end find_inverse_l108_10893


namespace tom_shirts_total_cost_l108_10814

theorem tom_shirts_total_cost 
  (num_tshirts_per_fandom : ℕ)
  (num_fandoms : ℕ)
  (cost_per_shirt : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (total_shirts : ℕ := num_tshirts_per_fandom * num_fandoms)
  (discount_per_shirt : ℚ := (cost_per_shirt : ℚ) * discount_rate)
  (cost_per_shirt_after_discount : ℚ := (cost_per_shirt : ℚ) - discount_per_shirt)
  (total_cost_before_tax : ℚ := (total_shirts * cost_per_shirt_after_discount))
  (tax_added : ℚ := total_cost_before_tax * tax_rate)
  (total_amount_paid : ℚ := total_cost_before_tax + tax_added)
  (h1 : num_tshirts_per_fandom = 5)
  (h2 : num_fandoms = 4)
  (h3 : cost_per_shirt = 15) 
  (h4 : discount_rate = 0.2)
  (h5 : tax_rate = 0.1)
  : total_amount_paid = 264 := 
by 
  sorry

end tom_shirts_total_cost_l108_10814


namespace number_of_figures_l108_10834

theorem number_of_figures (num_squares num_rectangles : ℕ) 
  (h1 : 8 * 8 / 4 = num_squares + num_rectangles) 
  (h2 : 2 * 54 + 4 * 8 = 8 * num_squares + 10 * num_rectangles) :
  num_squares = 10 ∧ num_rectangles = 6 :=
sorry

end number_of_figures_l108_10834


namespace min_bottles_needed_l108_10812

theorem min_bottles_needed (num_people : ℕ) (exchange_rate : ℕ) (bottles_needed_per_person : ℕ) (total_bottles_purchased : ℕ):
  num_people = 27 → exchange_rate = 3 → bottles_needed_per_person = 1 → total_bottles_purchased = 18 → 
  ∀ n, n = num_people → (n / bottles_needed_per_person) = 27 ∧ (num_people * 2 / 3) = 18 :=
by
  intros
  sorry

end min_bottles_needed_l108_10812


namespace price_increase_count_l108_10806

-- Conditions
def original_price (P : ℝ) : ℝ := P
def increase_factor : ℝ := 1.15
def final_factor : ℝ := 1.3225

-- The theorem that states the number of times the price increased
theorem price_increase_count (n : ℕ) :
  increase_factor ^ n = final_factor → n = 2 :=
by
  sorry

end price_increase_count_l108_10806


namespace shaded_region_correct_l108_10843

def side_length_ABCD : ℝ := 8
def side_length_BEFG : ℝ := 6

def area_square (side_length : ℝ) : ℝ := side_length ^ 2

def area_ABCD : ℝ := area_square side_length_ABCD
def area_BEFG : ℝ := area_square side_length_BEFG

def shaded_region_area : ℝ :=
  area_ABCD + area_BEFG - 32

theorem shaded_region_correct :
  shaded_region_area = 32 :=
by
  -- Proof omitted, but placeholders match problem conditions and answer
  sorry

end shaded_region_correct_l108_10843


namespace car_travel_distance_l108_10839

noncomputable def distance_in_miles (b t : ℝ) : ℝ :=
  (25 * b) / (1320 * t)

theorem car_travel_distance (b t : ℝ) : 
  let distance_in_feet := (b / 3) * (300 / t)
  let distance_in_miles' := distance_in_feet / 5280
  distance_in_miles' = distance_in_miles b t := 
by
  sorry

end car_travel_distance_l108_10839


namespace split_into_similar_heaps_l108_10883

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l108_10883


namespace Rogers_expense_fraction_l108_10867

variables (B m s p : ℝ)

theorem Rogers_expense_fraction (h1 : m = 0.25 * (B - s))
                              (h2 : s = 0.10 * (B - m))
                              (h3 : p = 0.10 * (m + s)) :
  m + s + p = 0.34 * B :=
by
  sorry

end Rogers_expense_fraction_l108_10867


namespace lcm_is_600_l108_10865

def lcm_of_24_30_40_50_60 : ℕ :=
  Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 (Nat.lcm 50 60)))

theorem lcm_is_600 : lcm_of_24_30_40_50_60 = 600 := by
  sorry

end lcm_is_600_l108_10865


namespace compute_f_six_l108_10868

def f (x : Int) : Int :=
  if x ≥ 0 then -x^2 - 1 else x + 10

theorem compute_f_six (x : Int) : f (f (f (f (f (f 1))))) = -35 :=
by
  sorry

end compute_f_six_l108_10868


namespace joe_new_average_score_after_dropping_lowest_l108_10884

theorem joe_new_average_score_after_dropping_lowest 
  (initial_average : ℕ)
  (lowest_score : ℕ)
  (num_tests : ℕ)
  (new_num_tests : ℕ)
  (total_points : ℕ)
  (new_total_points : ℕ)
  (new_average : ℕ) :
  initial_average = 70 →
  lowest_score = 55 →
  num_tests = 4 →
  new_num_tests = 3 →
  total_points = num_tests * initial_average →
  new_total_points = total_points - lowest_score →
  new_average = new_total_points / new_num_tests →
  new_average = 75 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end joe_new_average_score_after_dropping_lowest_l108_10884


namespace CauchySchwarz_l108_10885

theorem CauchySchwarz' (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 := by
  sorry

end CauchySchwarz_l108_10885


namespace area_of_shaded_region_l108_10892

-- Define the vertices of the larger square
def large_square_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 40), (0, 40)]

-- Define the vertices of the polygon forming the shaded area
def shaded_polygon_vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (40, 30), (40, 40), (10, 40), (0, 10)]

-- Provide the area of the larger square for reference
def large_square_area : ℝ := 1600

-- Provide the area of the triangles subtracted
def triangles_area : ℝ := 450

-- The main theorem stating the problem:
theorem area_of_shaded_region :
  let shaded_area := large_square_area - triangles_area
  shaded_area = 1150 :=
by
  sorry

end area_of_shaded_region_l108_10892


namespace smallest_positive_multiple_of_37_l108_10840

theorem smallest_positive_multiple_of_37 (a : ℕ) (h1 : 37 * a ≡ 3 [MOD 101]) (h2 : ∀ b : ℕ, 0 < b ∧ (37 * b ≡ 3 [MOD 101]) → a ≤ b) : 37 * a = 1628 :=
sorry

end smallest_positive_multiple_of_37_l108_10840


namespace simplify_tan_alpha_l108_10850

noncomputable def f (α : ℝ) : ℝ :=
(Real.sin (Real.pi / 2 + α) + Real.sin (-Real.pi - α)) /
  (3 * Real.cos (2 * Real.pi - α) + Real.cos (3 * Real.pi / 2 - α))

theorem simplify_tan_alpha (α : ℝ) (h : Real.tan α = 1) : f α = 1 := by
  sorry

end simplify_tan_alpha_l108_10850


namespace arithmetic_sequence_monotone_l108_10888

theorem arithmetic_sequence_monotone (a : ℕ → ℝ) (d : ℝ) (h_arithmetic : ∀ n, a (n + 1) - a n = d) :
  (a 2 > a 1) ↔ (∀ n, a (n + 1) > a n) :=
by 
  sorry

end arithmetic_sequence_monotone_l108_10888


namespace negation_of_p_is_neg_p_l108_10870

-- Define the proposition p
def p : Prop := ∃ m : ℝ, m > 0 ∧ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- Define the negation of p
def neg_p : Prop := ∀ m : ℝ, m > 0 → ¬ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- The theorem statement
theorem negation_of_p_is_neg_p : (¬ p) = neg_p := 
by
  sorry

end negation_of_p_is_neg_p_l108_10870


namespace intersection_A_B_l108_10820

def A : Set ℝ := { x | Real.log x > 0 }

def B : Set ℝ := { x | Real.exp x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | 1 < x ∧ x < Real.log 3 / Real.log 2 } :=
sorry

end intersection_A_B_l108_10820


namespace handshakes_at_gathering_l108_10866

-- Define the number of couples
def couples := 6

-- Define the total number of people
def total_people := 2 * couples

-- Each person shakes hands with 10 others (excluding their spouse)
def handshakes_per_person := 10

-- Total handshakes counted with pairs counted twice
def total_handshakes := total_people * handshakes_per_person / 2

-- The theorem to prove the number of handshakes
theorem handshakes_at_gathering : total_handshakes = 60 :=
by
  sorry

end handshakes_at_gathering_l108_10866


namespace square_vertex_distance_l108_10810

noncomputable def inner_square_perimeter : ℝ := 24
noncomputable def outer_square_perimeter : ℝ := 32
noncomputable def greatest_distance : ℝ := 7 * Real.sqrt 2

theorem square_vertex_distance :
  let inner_side := inner_square_perimeter / 4
  let outer_side := outer_square_perimeter / 4
  let inner_diagonal := Real.sqrt (inner_side ^ 2 + inner_side ^ 2)
  let outer_diagonal := Real.sqrt (outer_side ^ 2 + outer_side ^ 2)
  let distance := (inner_diagonal / 2) + (outer_diagonal / 2)
  distance = greatest_distance :=
by
  sorry

end square_vertex_distance_l108_10810


namespace upper_seat_ticket_price_l108_10821

variable (U : ℝ) 

-- Conditions
def lower_seat_price : ℝ := 30
def total_tickets_sold : ℝ := 80
def total_revenue : ℝ := 2100
def lower_tickets_sold : ℝ := 50

theorem upper_seat_ticket_price :
  (lower_seat_price * lower_tickets_sold + (total_tickets_sold - lower_tickets_sold) * U = total_revenue) →
  U = 20 := by
  sorry

end upper_seat_ticket_price_l108_10821


namespace Eunji_has_most_marbles_l108_10886

-- Declare constants for each person's marbles
def Minyoung_marbles : ℕ := 4
def Yujeong_marbles : ℕ := 2
def Eunji_marbles : ℕ := Minyoung_marbles + 1

-- Theorem: Eunji has the most marbles
theorem Eunji_has_most_marbles :
  Eunji_marbles > Minyoung_marbles ∧ Eunji_marbles > Yujeong_marbles :=
by
  sorry

end Eunji_has_most_marbles_l108_10886


namespace parallel_vectors_x_value_l108_10871

/-
Given that \(\overrightarrow{a} = (1,2)\) and \(\overrightarrow{b} = (2x, -3)\) are parallel vectors, prove that \(x = -\frac{3}{4}\).
-/
theorem parallel_vectors_x_value (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (2 * x, -3)) 
  (h_parallel : (a.1 * b.2) - (a.2 * b.1) = 0) : 
  x = -3 / 4 := by
  sorry

end parallel_vectors_x_value_l108_10871


namespace common_root_l108_10879

theorem common_root (a b x : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eq1 : x^2 + a * x + b = 0) (eq2 : x^3 + b * x + a = 0) : x = -1 :=
by
  sorry

end common_root_l108_10879


namespace siblings_water_intake_l108_10887

theorem siblings_water_intake 
  (theo_daily : ℕ := 8) 
  (mason_daily : ℕ := 7) 
  (roxy_daily : ℕ := 9) 
  (days_in_week : ℕ := 7) 
  : (theo_daily + mason_daily + roxy_daily) * days_in_week = 168 := 
by 
  sorry

end siblings_water_intake_l108_10887


namespace volume_of_bottle_l108_10873

theorem volume_of_bottle (r h : ℝ) (π : ℝ) (h₀ : π > 0)
  (h₁ : r^2 * h + (4 / 3) * r^3 = 625) :
  π * (r^2 * h + (4 / 3) * r^3) = 625 * π :=
by sorry

end volume_of_bottle_l108_10873


namespace cylinder_radius_l108_10878

open Real

theorem cylinder_radius (r : ℝ) 
  (h₁ : ∀(V₁ : ℝ), V₁ = π * (r + 4)^2 * 3)
  (h₂ : ∀(V₂ : ℝ), V₂ = π * r^2 * 9)
  (h₃ : ∀(V₁ V₂ : ℝ), V₁ = V₂) :
  r = 2 + 2 * sqrt 3 :=
by
  sorry

end cylinder_radius_l108_10878


namespace polynomial_at_3_l108_10869

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_at_3 : f 3 = 1641 := 
by
  -- proof would go here
  sorry

end polynomial_at_3_l108_10869


namespace set_intersection_l108_10800

theorem set_intersection :
  let A := {x : ℝ | 0 < x}
  let B := {x : ℝ | -1 ≤ x ∧ x < 3}
  A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := 
by
  sorry

end set_intersection_l108_10800


namespace percentage_increase_in_expenditure_l108_10813

/-- Given conditions:
- The price of sugar increased by 32%
- The family's original monthly sugar consumption was 30 kg
- The family's new monthly sugar consumption is 25 kg
- The family's expenditure on sugar increased by 10%

Prove that the percentage increase in the family's expenditure on sugar is 10%. -/
theorem percentage_increase_in_expenditure (P : ℝ) :
  let initial_consumption := 30
  let new_consumption := 25
  let price_increase := 0.32
  let original_price := P
  let new_price := (1 + price_increase) * original_price
  let original_expenditure := initial_consumption * original_price
  let new_expenditure := new_consumption * new_price
  let expenditure_increase := new_expenditure - original_expenditure
  let percentage_increase := (expenditure_increase / original_expenditure) * 100
  percentage_increase = 10 := sorry

end percentage_increase_in_expenditure_l108_10813


namespace sum_of_a_b_l108_10822

theorem sum_of_a_b (a b : ℝ) (h1 : a > b) (h2 : |a| = 9) (h3 : b^2 = 4) : a + b = 11 ∨ a + b = 7 := 
sorry

end sum_of_a_b_l108_10822


namespace Vanya_Journey_Five_times_Anya_Journey_l108_10853

theorem Vanya_Journey_Five_times_Anya_Journey (a_start a_end v_start v_end : ℕ)
  (h1 : a_start = 1) (h2 : a_end = 2) (h3 : v_start = 1) (h4 : v_end = 6) :
  (v_end - v_start) = 5 * (a_end - a_start) :=
  sorry

end Vanya_Journey_Five_times_Anya_Journey_l108_10853


namespace problem_statement_l108_10833

theorem problem_statement (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3 * y^2) / 7 = 75 / 7 :=
by 
  -- proof goes here
  sorry

end problem_statement_l108_10833


namespace man_age_year_l108_10847

theorem man_age_year (x : ℕ) (h1 : x^2 = 1892) (h2 : 1850 ≤ x ∧ x ≤ 1900) :
  (x = 44) → (1892 = 1936) := by
sorry

end man_age_year_l108_10847


namespace quadratic_nonnegative_l108_10895

theorem quadratic_nonnegative (x y : ℝ) : x^2 + x * y + y^2 ≥ 0 :=
by sorry

end quadratic_nonnegative_l108_10895


namespace geometric_sequence_sum_l108_10880

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 := 
sorry

end geometric_sequence_sum_l108_10880


namespace fred_fewer_games_l108_10846

/-- Fred went to 36 basketball games last year -/
def games_last_year : ℕ := 36

/-- Fred went to 25 basketball games this year -/
def games_this_year : ℕ := 25

/-- Prove that Fred went to 11 fewer games this year compared to last year -/
theorem fred_fewer_games : games_last_year - games_this_year = 11 := by
  sorry

end fred_fewer_games_l108_10846


namespace area_triangle_AMB_l108_10862

def parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

def point_A : ℝ × ℝ := (0, parabola 0)

def rotated_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 2

def point_B : ℝ × ℝ := (0, rotated_parabola 0)

def vertex_M : ℝ × ℝ := (-1, 2)

def area_of_triangle (A B M : ℝ × ℝ) : ℝ :=
  0.5 * (A.2 - M.2) * (M.1 - B.1)

theorem area_triangle_AMB : area_of_triangle point_A point_B vertex_M = 1 :=
  sorry

end area_triangle_AMB_l108_10862
