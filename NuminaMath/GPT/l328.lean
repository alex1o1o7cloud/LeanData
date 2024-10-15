import Mathlib

namespace NUMINAMATH_GPT_ratio_values_l328_32871

theorem ratio_values (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) 
  (h₀ : (x + y) / z = (y + z) / x) (h₀' : (y + z) / x = (z + x) / y) :
  ∃ a : ℝ, a = -1 ∨ a = 8 :=
sorry

end NUMINAMATH_GPT_ratio_values_l328_32871


namespace NUMINAMATH_GPT_f_of_2_l328_32811

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end NUMINAMATH_GPT_f_of_2_l328_32811


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_product_square_l328_32810

theorem right_triangle_hypotenuse_product_square (A₁ A₂ : ℝ) (a₁ b₁ a₂ b₂ : ℝ) 
(h₁ : a₁ * b₁ / 2 = A₁) (h₂ : a₂ * b₂ / 2 = A₂) 
(h₃ : A₁ = 2) (h₄ : A₂ = 3) 
(h₅ : a₁ = a₂) (h₆ : b₂ = 2 * b₁) : 
(a₁ ^ 2 + b₁ ^ 2) * (a₂ ^ 2 + b₂ ^ 2) = 325 := 
by sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_product_square_l328_32810


namespace NUMINAMATH_GPT_outdoor_section_length_l328_32877

theorem outdoor_section_length (W : ℝ) (A : ℝ) (hW : W = 4) (hA : A = 24) : ∃ L : ℝ, A = W * L ∧ L = 6 := 
by
  use 6
  sorry

end NUMINAMATH_GPT_outdoor_section_length_l328_32877


namespace NUMINAMATH_GPT_low_degree_polys_condition_l328_32861

theorem low_degree_polys_condition :
  ∃ (f : Polynomial ℤ), ∃ (g : Polynomial ℤ), ∃ (h : Polynomial ℤ),
    (f = Polynomial.X ^ 3 + Polynomial.X ^ 2 + Polynomial.X + 1 ∨
          f = Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2 * Polynomial.X + 2 ∨
          f = 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 2 * Polynomial.X + 1 ∨
          f = 2 * Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + Polynomial.X + 2) ∧
          f ^ 4 + 2 * f + 2 = (Polynomial.X ^ 4 + 2 * Polynomial.X ^ 2 + 2) * g + 3 * h := 
sorry

end NUMINAMATH_GPT_low_degree_polys_condition_l328_32861


namespace NUMINAMATH_GPT_bags_on_wednesday_l328_32842

theorem bags_on_wednesday (charge_per_bag money_per_bag monday_bags tuesday_bags total_money wednesday_bags : ℕ)
  (h_charge : charge_per_bag = 4)
  (h_money_per_dollar : money_per_bag = charge_per_bag)
  (h_monday : monday_bags = 5)
  (h_tuesday : tuesday_bags = 3)
  (h_total : total_money = 68) :
  wednesday_bags = (total_money - (monday_bags + tuesday_bags) * money_per_bag) / charge_per_bag :=
by
  sorry

end NUMINAMATH_GPT_bags_on_wednesday_l328_32842


namespace NUMINAMATH_GPT_no_solution_range_of_a_l328_32834

theorem no_solution_range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) → a ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_range_of_a_l328_32834


namespace NUMINAMATH_GPT_sum_of_first_five_multiples_of_15_l328_32851

theorem sum_of_first_five_multiples_of_15 : (15 + 30 + 45 + 60 + 75) = 225 :=
by sorry

end NUMINAMATH_GPT_sum_of_first_five_multiples_of_15_l328_32851


namespace NUMINAMATH_GPT_tangent_sufficient_but_not_necessary_condition_l328_32868

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let line := fun (x y : ℝ) => x + y - m = 0
  let circle := fun (x y : ℝ) => (x - 1) ^ 2 + (y - 1) ^ 2 = 2
  ∃ (x y: ℝ), line x y ∧ circle x y -- A line and circle are tangent if they intersect exactly at one point

theorem tangent_sufficient_but_not_necessary_condition (m : ℝ) :
  (tangent_condition m) ↔ (m = 0 ∨ m = 4) := by
  sorry

end NUMINAMATH_GPT_tangent_sufficient_but_not_necessary_condition_l328_32868


namespace NUMINAMATH_GPT_exists_a_b_l328_32893

theorem exists_a_b (r : Fin 5 → ℝ) : ∃ (i j : Fin 5), i ≠ j ∧ 0 ≤ (r i - r j) / (1 + r i * r j) ∧ (r i - r j) / (1 + r i * r j) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_a_b_l328_32893


namespace NUMINAMATH_GPT_max_groups_l328_32819

def eggs : ℕ := 20
def marbles : ℕ := 6
def eggs_per_group : ℕ := 5
def marbles_per_group : ℕ := 2

def groups_of_eggs := eggs / eggs_per_group
def groups_of_marbles := marbles / marbles_per_group

theorem max_groups (h1 : eggs = 20) (h2 : marbles = 6) 
                    (h3 : eggs_per_group = 5) (h4 : marbles_per_group = 2) : 
                    min (groups_of_eggs) (groups_of_marbles) = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_groups_l328_32819


namespace NUMINAMATH_GPT_rowing_upstream_speed_l328_32870

-- Define the speed of the man in still water
def V_m : ℝ := 45

-- Define the speed of the man rowing downstream
def V_downstream : ℝ := 65

-- Define the speed of the stream
def V_s : ℝ := V_downstream - V_m

-- Define the speed of the man rowing upstream
def V_upstream : ℝ := V_m - V_s

-- Prove that the speed of the man rowing upstream is 25 kmph
theorem rowing_upstream_speed :
  V_upstream = 25 := by
  sorry

end NUMINAMATH_GPT_rowing_upstream_speed_l328_32870


namespace NUMINAMATH_GPT_total_surface_area_of_resulting_structure_l328_32802

-- Definitions for the conditions
def bigCube := 12 * 12 * 12
def smallCube := 2 * 2 * 2
def totalSmallCubes := 64
def removedCubes := 7
def remainingCubes := totalSmallCubes - removedCubes
def surfaceAreaPerSmallCube := 24
def extraExposedSurfaceArea := 6
def effectiveSurfaceAreaPerSmallCube := surfaceAreaPerSmallCube + extraExposedSurfaceArea

-- Definition and the main statement of the proof problem.
def totalSurfaceArea := remainingCubes * effectiveSurfaceAreaPerSmallCube

theorem total_surface_area_of_resulting_structure : totalSurfaceArea = 1710 :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_resulting_structure_l328_32802


namespace NUMINAMATH_GPT_initial_butterfat_percentage_l328_32895

theorem initial_butterfat_percentage (P : ℝ) :
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  initial_butterfat - removed_butterfat = desired_butterfat
→ P = 4 :=
by
  intros
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  sorry

end NUMINAMATH_GPT_initial_butterfat_percentage_l328_32895


namespace NUMINAMATH_GPT_minimum_value_of_f_l328_32843

-- Define the function
def f (a b x : ℝ) := x^2 + (a + 2) * x + b

-- Condition that ensures the graph is symmetric about x = 1
def symmetric_about_x1 (a : ℝ) : Prop := a + 2 = -2

-- Minimum value of the function f(x) in terms of the constant c
theorem minimum_value_of_f (a b : ℝ) (h : symmetric_about_x1 a) : ∃ c : ℝ, ∀ x : ℝ, f a b x ≥ c :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_f_l328_32843


namespace NUMINAMATH_GPT_find_x_for_opposite_directions_l328_32814

-- Define the vectors and the opposite direction condition
def vector_a (x : ℝ) : ℝ × ℝ := (1, -x)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -16)

-- Define the condition that vectors are in opposite directions
def opp_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (-k) • b

-- The main theorem statement
theorem find_x_for_opposite_directions : ∃ x : ℝ, opp_directions (vector_a x) (vector_b x) ∧ x = -5 := 
sorry

end NUMINAMATH_GPT_find_x_for_opposite_directions_l328_32814


namespace NUMINAMATH_GPT_sixty_percent_of_40_greater_than_four_fifths_of_25_l328_32865

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  (60 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25 = 4 := by
  sorry

end NUMINAMATH_GPT_sixty_percent_of_40_greater_than_four_fifths_of_25_l328_32865


namespace NUMINAMATH_GPT_area_ratio_S_T_l328_32874

open Set

def T : Set (ℝ × ℝ × ℝ) := {p | let (x, y, z) := p; x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 1}

def supports (p q : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  let (a, b, c) := q
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

def S : Set (ℝ × ℝ × ℝ) := {p ∈ T | supports p (1/4, 1/4, 1/2)}

theorem area_ratio_S_T : ∃ k : ℝ, k = 3 / 4 ∧
  ∃ (area_T area_S : ℝ), area_T ≠ 0 ∧ (area_S / area_T = k) := sorry

end NUMINAMATH_GPT_area_ratio_S_T_l328_32874


namespace NUMINAMATH_GPT_remainder_3001_3005_mod_23_l328_32812

theorem remainder_3001_3005_mod_23 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_3001_3005_mod_23_l328_32812


namespace NUMINAMATH_GPT_find_4digit_number_l328_32805

theorem find_4digit_number (a b c d n n' : ℕ) :
  n = 1000 * a + 100 * b + 10 * c + d →
  n' = 1000 * d + 100 * c + 10 * b + a →
  n = n' - 7182 →
  n = 1909 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_4digit_number_l328_32805


namespace NUMINAMATH_GPT_total_oranges_is_correct_l328_32884

/-- Define the number of boxes and the number of oranges per box -/
def boxes : ℕ := 7
def oranges_per_box : ℕ := 6

/-- Prove that the total number of oranges is 42 -/
theorem total_oranges_is_correct : boxes * oranges_per_box = 42 := 
by 
  sorry

end NUMINAMATH_GPT_total_oranges_is_correct_l328_32884


namespace NUMINAMATH_GPT_calories_per_person_l328_32888

-- Definitions based on the conditions from a)
def oranges : ℕ := 5
def pieces_per_orange : ℕ := 8
def people : ℕ := 4
def calories_per_orange : ℝ := 80

-- Theorem based on the equivalent proof problem
theorem calories_per_person : 
    ((oranges * pieces_per_orange) / people) / pieces_per_orange * calories_per_orange = 100 := 
by
  sorry

end NUMINAMATH_GPT_calories_per_person_l328_32888


namespace NUMINAMATH_GPT_length_of_largest_square_l328_32849

-- Define the conditions of the problem
def side_length_of_shaded_square : ℕ := 10
def side_length_of_largest_square : ℕ := 24

-- The statement to prove
theorem length_of_largest_square (x : ℕ) (h1 : x = side_length_of_shaded_square) : 
  4 * x = side_length_of_largest_square :=
  by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_length_of_largest_square_l328_32849


namespace NUMINAMATH_GPT_least_n_value_l328_32821

open Nat

theorem least_n_value (n : ℕ) (h : 1 / (n * (n + 1)) < 1 / 15) : n = 4 :=
sorry

end NUMINAMATH_GPT_least_n_value_l328_32821


namespace NUMINAMATH_GPT_smallest_n_mod_l328_32806

theorem smallest_n_mod : ∃ n : ℕ, 5 * n ≡ 2024 [MOD 26] ∧ n > 0 ∧ ∀ m : ℕ, (5 * m ≡ 2024 [MOD 26] ∧ m > 0) → n ≤ m :=
  sorry

end NUMINAMATH_GPT_smallest_n_mod_l328_32806


namespace NUMINAMATH_GPT_set_inclusion_l328_32896

-- Definitions based on given conditions
def setA (x : ℝ) : Prop := 0 < x ∧ x < 2
def setB (x : ℝ) : Prop := x > 0

-- Statement of the proof problem
theorem set_inclusion : ∀ x, setA x → setB x :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_set_inclusion_l328_32896


namespace NUMINAMATH_GPT_find_constants_l328_32824

open BigOperators

theorem find_constants (a b c : ℕ) :
  (∀ n : ℕ, n > 0 → (∑ k in Finset.range n, k.succ * (k.succ + 1) ^ 2) = (n * (n + 1) * (a * n^2 + b * n + c)) / 12) →
  (a = 3 ∧ b = 11 ∧ c = 10) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l328_32824


namespace NUMINAMATH_GPT_QED_mul_eq_neg_25I_l328_32846

namespace ComplexMultiplication

open Complex

def Q : ℂ := 3 + 4 * Complex.I
def E : ℂ := -Complex.I
def D : ℂ := 3 - 4 * Complex.I

theorem QED_mul_eq_neg_25I : Q * E * D = -25 * Complex.I :=
by
  sorry

end ComplexMultiplication

end NUMINAMATH_GPT_QED_mul_eq_neg_25I_l328_32846


namespace NUMINAMATH_GPT_fliers_sent_afternoon_fraction_l328_32876

-- Definitions of given conditions
def total_fliers : ℕ := 2000
def fliers_morning_fraction : ℚ := 1 / 10
def remaining_fliers_next_day : ℕ := 1350

-- Helper definitions based on conditions
def fliers_sent_morning := total_fliers * fliers_morning_fraction
def fliers_after_morning := total_fliers - fliers_sent_morning
def fliers_sent_afternoon := fliers_after_morning - remaining_fliers_next_day

-- Theorem stating the required proof
theorem fliers_sent_afternoon_fraction :
  fliers_sent_afternoon / fliers_after_morning = 1 / 4 :=
sorry

end NUMINAMATH_GPT_fliers_sent_afternoon_fraction_l328_32876


namespace NUMINAMATH_GPT_sin_inequality_solution_set_l328_32885

theorem sin_inequality_solution_set : 
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x < - Real.sqrt 3 / 2} =
  {x : ℝ | (4 * Real.pi / 3) < x ∧ x < (5 * Real.pi / 3)} := by
  sorry

end NUMINAMATH_GPT_sin_inequality_solution_set_l328_32885


namespace NUMINAMATH_GPT_solve_ab_eq_l328_32886

theorem solve_ab_eq (a b : ℕ) (h : a^b + a + b = b^a) : a = 5 ∧ b = 2 :=
sorry

end NUMINAMATH_GPT_solve_ab_eq_l328_32886


namespace NUMINAMATH_GPT_speed_of_first_part_l328_32898

theorem speed_of_first_part (v : ℝ) (h1 : v > 0)
  (h_total_distance : 50 = 25 + 25)
  (h_average_speed : 44 = 50 / ((25 / v) + (25 / 33))) :
  v = 66 :=
by sorry

end NUMINAMATH_GPT_speed_of_first_part_l328_32898


namespace NUMINAMATH_GPT_factor_expression_l328_32899

variable (y : ℝ)

theorem factor_expression : 
  6*y*(y + 2) + 15*(y + 2) + 12 = 3*(2*y + 5)*(y + 2) :=
sorry

end NUMINAMATH_GPT_factor_expression_l328_32899


namespace NUMINAMATH_GPT_westward_measurement_l328_32822

def east_mov (d : ℕ) : ℤ := - (d : ℤ)

def west_mov (d : ℕ) : ℤ := d

theorem westward_measurement :
  east_mov 50 = -50 →
  west_mov 60 = 60 :=
by
  intro h
  exact rfl

end NUMINAMATH_GPT_westward_measurement_l328_32822


namespace NUMINAMATH_GPT_intersection_of_sets_l328_32839

def set_M := { y : ℝ | y ≥ 0 }
def set_N := { y : ℝ | ∃ x : ℝ, y = -x^2 + 1 }

theorem intersection_of_sets : set_M ∩ set_N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l328_32839


namespace NUMINAMATH_GPT_phi_value_l328_32817

theorem phi_value (phi : ℝ) (h : 0 < phi ∧ phi < π) 
  (hf : ∀ x : ℝ, 3 * Real.sin (2 * abs x - π / 3 + phi) = 3 * Real.sin (2 * x - π / 3 + phi)) 
  : φ = 5 * π / 6 :=
by 
  sorry

end NUMINAMATH_GPT_phi_value_l328_32817


namespace NUMINAMATH_GPT_more_pairs_B_than_A_l328_32879

theorem more_pairs_B_than_A :
    let pairs_per_box := 20
    let boxes_A := 8
    let pairs_A := boxes_A * pairs_per_box
    let pairs_B := 5 * pairs_A
    let more_pairs := pairs_B - pairs_A
    more_pairs = 640
:= by
    sorry

end NUMINAMATH_GPT_more_pairs_B_than_A_l328_32879


namespace NUMINAMATH_GPT_golden_section_AC_length_l328_32804

namespace GoldenSection

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def AC_length (AB : ℝ) : ℝ :=
  let φ := golden_ratio
  AB / φ

theorem golden_section_AC_length (AB : ℝ) (C_gold : Prop) (hAB : AB = 2) (A_gt_B : AC_length AB > AB - AC_length AB) :
  AC_length AB = Real.sqrt 5 - 1 :=
  sorry

end GoldenSection

end NUMINAMATH_GPT_golden_section_AC_length_l328_32804


namespace NUMINAMATH_GPT_find_overlap_length_l328_32818

-- Definitions of the given conditions
def total_length_of_segments := 98 -- cm
def edge_to_edge_distance := 83 -- cm
def number_of_overlaps := 6

-- Theorem stating the value of x in centimeters
theorem find_overlap_length (x : ℝ) 
  (h1 : total_length_of_segments = 98) 
  (h2 : edge_to_edge_distance = 83) 
  (h3 : number_of_overlaps = 6) 
  (h4 : total_length_of_segments = edge_to_edge_distance + number_of_overlaps * x) : 
  x = 2.5 :=
  sorry

end NUMINAMATH_GPT_find_overlap_length_l328_32818


namespace NUMINAMATH_GPT_vikas_rank_among_boys_l328_32859

def vikas_rank_overall := 9
def tanvi_rank_overall := 17
def girls_between := 2
def vikas_rank_top_boys := 4
def vikas_rank_bottom_overall := 18

theorem vikas_rank_among_boys (vikas_rank_overall tanvi_rank_overall girls_between vikas_rank_top_boys vikas_rank_bottom_overall : ℕ) :
  vikas_rank_top_boys = 4 := by
  sorry

end NUMINAMATH_GPT_vikas_rank_among_boys_l328_32859


namespace NUMINAMATH_GPT_discount_percentage_l328_32881

variable {P P_b P_s : ℝ}
variable {D : ℝ}

theorem discount_percentage (P_s_eq_bought : P_s = 1.60 * P_b)
  (P_s_eq_original : P_s = 1.52 * P)
  (P_b_eq_discount : P_b = P * (1 - D)) :
  D = 0.05 := by
sorry

end NUMINAMATH_GPT_discount_percentage_l328_32881


namespace NUMINAMATH_GPT_intersection_property_l328_32860

def universal_set : Set ℝ := Set.univ

def M : Set ℝ := {-1, 1, 2, 4}

def N : Set ℝ := {x : ℝ | x > 2}

theorem intersection_property : (M ∩ N) = {4} := by
  sorry

end NUMINAMATH_GPT_intersection_property_l328_32860


namespace NUMINAMATH_GPT_rebecca_marbles_l328_32866

theorem rebecca_marbles (M : ℕ) (h1 : 20 = M + 14) : M = 6 :=
by
  sorry

end NUMINAMATH_GPT_rebecca_marbles_l328_32866


namespace NUMINAMATH_GPT_smallest_sum_of_two_perfect_squares_l328_32853

theorem smallest_sum_of_two_perfect_squares (x y : ℕ) (h : x^2 - y^2 = 143) :
  x + y = 13 ∧ x - y = 11 → x^2 + y^2 = 145 :=
by
  -- Add this placeholder "sorry" to skip the proof, as required.
  sorry

end NUMINAMATH_GPT_smallest_sum_of_two_perfect_squares_l328_32853


namespace NUMINAMATH_GPT_solve_complex_addition_l328_32856

def complex_addition_problem : Prop :=
  let B := Complex.mk 3 (-2)
  let Q := Complex.mk (-5) 1
  let R := Complex.mk 1 (-2)
  let T := Complex.mk 4 3
  B - Q + R + T = Complex.mk 13 (-2)

theorem solve_complex_addition : complex_addition_problem := by
  sorry

end NUMINAMATH_GPT_solve_complex_addition_l328_32856


namespace NUMINAMATH_GPT_number_of_possible_lengths_of_diagonal_l328_32831

theorem number_of_possible_lengths_of_diagonal :
  ∃ n : ℕ, n = 13 ∧
  (∀ y : ℕ, (5 ≤ y ∧ y ≤ 17) ↔ (y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨
   y = 10 ∨ y = 11 ∨ y = 12 ∨ y = 13 ∨ y = 14 ∨ y = 15 ∨ y = 16 ∨ y = 17)) :=
by
  exists 13
  sorry

end NUMINAMATH_GPT_number_of_possible_lengths_of_diagonal_l328_32831


namespace NUMINAMATH_GPT_problem_min_value_l328_32801

noncomputable def min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : ℝ :=
  1 / x^2 + 1 / y^2 + 1 / (x * y)

theorem problem_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  min_value x y hx hy hxy = 3 := 
sorry

end NUMINAMATH_GPT_problem_min_value_l328_32801


namespace NUMINAMATH_GPT_borrowed_amount_correct_l328_32808

noncomputable def principal_amount (I: ℚ) (r1 r2 r3 r4 t1 t2 t3 t4: ℚ): ℚ :=
  I / (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4)

def interest_rate_1 := (6.5 / 100 : ℚ)
def interest_rate_2 := (9.5 / 100 : ℚ)
def interest_rate_3 := (11 / 100 : ℚ)
def interest_rate_4 := (14.5 / 100 : ℚ)

def time_period_1 := (2.5 : ℚ)
def time_period_2 := (3.75 : ℚ)
def time_period_3 := (1.5 : ℚ)
def time_period_4 := (4.25 : ℚ)

def total_interest := (14500 : ℚ)

def expected_principal := (11153.846153846154 : ℚ)

theorem borrowed_amount_correct :
  principal_amount total_interest interest_rate_1 interest_rate_2 interest_rate_3 interest_rate_4 time_period_1 time_period_2 time_period_3 time_period_4 = expected_principal :=
by
  sorry

end NUMINAMATH_GPT_borrowed_amount_correct_l328_32808


namespace NUMINAMATH_GPT_recurring_decimal_addition_l328_32809

noncomputable def recurring_decimal_sum : ℚ :=
  (23 / 99) + (14 / 999) + (6 / 9999)

theorem recurring_decimal_addition :
  recurring_decimal_sum = 2469 / 9999 :=
sorry

end NUMINAMATH_GPT_recurring_decimal_addition_l328_32809


namespace NUMINAMATH_GPT_avg_calculation_l328_32800

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem avg_calculation :
  avg4 (avg2 1 2) (avg2 3 1) (avg2 2 0) (avg2 1 1) = 11 / 8 := by
  sorry

end NUMINAMATH_GPT_avg_calculation_l328_32800


namespace NUMINAMATH_GPT_mt_product_l328_32816

noncomputable def g (x : ℝ) : ℝ := sorry

theorem mt_product
  (hg : ∀ (x y : ℝ), g (g x + y) = g x + g (g y + g (-x)) - x) : 
  ∃ m t : ℝ, m = 1 ∧ t = -5 ∧ m * t = -5 := 
by
  sorry

end NUMINAMATH_GPT_mt_product_l328_32816


namespace NUMINAMATH_GPT_cost_of_TOP_book_l328_32887

theorem cost_of_TOP_book (T : ℝ) (h1 : T = 8)
  (abc_cost : ℝ := 23)
  (top_books_sold : ℝ := 13)
  (abc_books_sold : ℝ := 4)
  (earnings_difference : ℝ := 12)
  (h2 : top_books_sold * T - abc_books_sold * abc_cost = earnings_difference) :
  T = 8 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_TOP_book_l328_32887


namespace NUMINAMATH_GPT_total_number_of_students_l328_32838

theorem total_number_of_students (T G : ℕ) (h1 : 50 + G = T) (h2 : G = 50 * T / 100) : T = 100 :=
  sorry

end NUMINAMATH_GPT_total_number_of_students_l328_32838


namespace NUMINAMATH_GPT_total_bread_amt_l328_32857

-- Define the conditions
variables (bread_dinner bread_lunch bread_breakfast total_bread : ℕ)
axiom bread_dinner_amt : bread_dinner = 240
axiom dinner_lunch_ratio : bread_dinner = 8 * bread_lunch
axiom dinner_breakfast_ratio : bread_dinner = 6 * bread_breakfast

-- The proof statement
theorem total_bread_amt : total_bread = bread_dinner + bread_lunch + bread_breakfast → total_bread = 310 :=
by
  -- Use the axioms and the given conditions to derive the statement
  sorry

end NUMINAMATH_GPT_total_bread_amt_l328_32857


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l328_32883

noncomputable def first_term : ℕ := 5
noncomputable def last_term : ℕ := 50
noncomputable def sum_terms : ℕ := 275

theorem common_difference_arithmetic_sequence :
  ∃ d n, (last_term = first_term + (n - 1) * d) ∧ (sum_terms = n * (first_term + last_term) / 2) ∧ d = 5 :=
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l328_32883


namespace NUMINAMATH_GPT_log_cut_piece_weight_l328_32875

-- Defining the conditions

def log_length : ℕ := 20
def half_log_length : ℕ := log_length / 2
def weight_per_foot : ℕ := 150

-- The main theorem stating the problem
theorem log_cut_piece_weight : (half_log_length * weight_per_foot) = 1500 := 
by 
  sorry

end NUMINAMATH_GPT_log_cut_piece_weight_l328_32875


namespace NUMINAMATH_GPT_find_base_b_l328_32845

theorem find_base_b (b : ℕ) (h : (3 * b + 4) ^ 2 = b ^ 3 + 2 * b ^ 2 + 9 * b + 6) : b = 10 :=
sorry

end NUMINAMATH_GPT_find_base_b_l328_32845


namespace NUMINAMATH_GPT_volume_of_regular_triangular_pyramid_l328_32878

noncomputable def pyramid_volume (R φ : ℝ) : ℝ :=
  (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)

theorem volume_of_regular_triangular_pyramid (R φ : ℝ) 
  (cond1 : R > 0)
  (cond2: 0 < φ ∧ φ < π) :
  ∃ V, V = pyramid_volume R φ := by
    use (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)
    sorry

end NUMINAMATH_GPT_volume_of_regular_triangular_pyramid_l328_32878


namespace NUMINAMATH_GPT_areas_of_triangles_l328_32882

-- Define the condition that the gcd of a, b, and c is 1
def gcd_one (a b c : ℤ) : Prop := Int.gcd (Int.gcd a b) c = 1

-- Define the set of possible areas for triangles in E
def f_E : Set ℝ :=
  { area | ∃ (a b c : ℤ), gcd_one a b c ∧ area = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) }

theorem areas_of_triangles : 
  f_E = { area | ∃ (a b c : ℤ), gcd_one a b c ∧ area = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) } :=
by {
  sorry
}

end NUMINAMATH_GPT_areas_of_triangles_l328_32882


namespace NUMINAMATH_GPT_johns_total_spent_l328_32833

def total_spent (num_tshirts: Nat) (price_per_tshirt: Nat) (price_pants: Nat): Nat :=
  (num_tshirts * price_per_tshirt) + price_pants

theorem johns_total_spent : total_spent 3 20 50 = 110 := by
  sorry

end NUMINAMATH_GPT_johns_total_spent_l328_32833


namespace NUMINAMATH_GPT_boys_assigned_l328_32891

theorem boys_assigned (B G : ℕ) (h1 : B + G = 18) (h2 : B = G - 2) : B = 8 :=
sorry

end NUMINAMATH_GPT_boys_assigned_l328_32891


namespace NUMINAMATH_GPT_fraction_of_larger_part_l328_32847

theorem fraction_of_larger_part (x y : ℝ) (f : ℝ) (h1 : x = 50) (h2 : x + y = 66) (h3 : f * x = 0.625 * y + 10) : f = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_larger_part_l328_32847


namespace NUMINAMATH_GPT_lines_intersection_l328_32848

theorem lines_intersection :
  ∃ (x y : ℝ), 
    (x - y = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersection_l328_32848


namespace NUMINAMATH_GPT_problem_l328_32852

theorem problem (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  x + (x ^ 3 / y ^ 2) + (y ^ 3 / x ^ 2) + y = 440 := by
  sorry

end NUMINAMATH_GPT_problem_l328_32852


namespace NUMINAMATH_GPT_other_continents_passengers_l328_32867

def passengers_from_other_continents (T N_A E A As : ℕ) : ℕ := T - (N_A + E + A + As)

theorem other_continents_passengers :
  passengers_from_other_continents 108 (108 / 12) (108 / 4) (108 / 9) (108 / 6) = 42 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_other_continents_passengers_l328_32867


namespace NUMINAMATH_GPT_number_of_multiples_in_range_l328_32862

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end NUMINAMATH_GPT_number_of_multiples_in_range_l328_32862


namespace NUMINAMATH_GPT_tangent_line_y_intercept_l328_32837

def circle1Center: ℝ × ℝ := (3, 0)
def circle1Radius: ℝ := 3
def circle2Center: ℝ × ℝ := (7, 0)
def circle2Radius: ℝ := 2

theorem tangent_line_y_intercept
    (tangent_line: ℝ × ℝ -> ℝ) 
    (P : tangent_line (circle1Center.1, circle1Center.2 + circle1Radius) = 0) -- Tangent condition for Circle 1
    (Q : tangent_line (circle2Center.1, circle2Center.2 + circle2Radius) = 0) -- Tangent condition for Circle 2
    :
    tangent_line (0, 4.5) = 0 := 
sorry

end NUMINAMATH_GPT_tangent_line_y_intercept_l328_32837


namespace NUMINAMATH_GPT_value_of_expression_l328_32841

theorem value_of_expression (x1 x2 : ℝ) 
  (h1 : x1 ^ 2 - 3 * x1 - 4 = 0) 
  (h2 : x2 ^ 2 - 3 * x2 - 4 = 0)
  (h3 : x1 + x2 = 3) 
  (h4 : x1 * x2 = -4) : 
  x1 ^ 2 - 4 * x1 - x2 + 2 * x1 * x2 = -7 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l328_32841


namespace NUMINAMATH_GPT_fill_box_with_L_blocks_l328_32889

theorem fill_box_with_L_blocks (m n k : ℕ) 
  (hm : m > 1) (hn : n > 1) (hk : k > 1) (hk_div3 : k % 3 = 0) : 
  ∃ (fill : ℕ → ℕ → ℕ → Prop), fill m n k → True := 
by
  sorry

end NUMINAMATH_GPT_fill_box_with_L_blocks_l328_32889


namespace NUMINAMATH_GPT_excircle_inequality_l328_32828

variables {a b c : ℝ} -- The sides of the triangle

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2 -- Definition of semiperimeter

noncomputable def excircle_distance (p a : ℝ) : ℝ := p - a -- Distance from vertices to tangency points

theorem excircle_inequality (a b c : ℝ) (p : ℝ) 
    (h1 : p = semiperimeter a b c) : 
    (excircle_distance p a) + (excircle_distance p b) > p := 
by
    -- Placeholder for proof
    sorry

end NUMINAMATH_GPT_excircle_inequality_l328_32828


namespace NUMINAMATH_GPT_find_c_l328_32897

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 8)) : c = 17 / 3 := 
by
  -- Add the necessary assumptions and let Lean verify these assumptions.
  have b_eq : 3 * b = 8 := sorry
  have b_val : b = 8 / 3 := sorry
  have h_coeff : c = b + 3 := sorry
  exact h_coeff.trans (by rw [b_val]; norm_num)

end NUMINAMATH_GPT_find_c_l328_32897


namespace NUMINAMATH_GPT_case1_BL_case2_BL_l328_32890

variable (AD BD BL AL : ℝ)

theorem case1_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 3)
  (h₃ : AB = 6 * Real.sqrt 13)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 2 * AL)
  : BL = 16 * Real.sqrt 3 - 12 := by
  sorry

theorem case2_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 6)
  (h₃ : AB = 30)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 4 * AL)
  : BL = (16 * Real.sqrt 6 - 6) / 5 := by
  sorry

end NUMINAMATH_GPT_case1_BL_case2_BL_l328_32890


namespace NUMINAMATH_GPT_domain_shift_l328_32873

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the domain of f
def domain_f : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- State the problem in Lean
theorem domain_shift (hf : ∀ x, f x ∈ domain_f) : 
    { x | 1 ≤ x ∧ x ≤ 2 } = { x | ∃ y, y ∈ domain_f ∧ x = y + 1 } :=
by
  sorry

end NUMINAMATH_GPT_domain_shift_l328_32873


namespace NUMINAMATH_GPT_sum_of_odd_coefficients_l328_32854

theorem sum_of_odd_coefficients (a : ℝ) (h : (a + 1) * 16 = 32) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_odd_coefficients_l328_32854


namespace NUMINAMATH_GPT_find_PB_l328_32844

noncomputable def PA : ℝ := 5
noncomputable def PT (AB : ℝ) : ℝ := 2 * (AB - PA) + 1
noncomputable def PB (AB : ℝ) : ℝ := PA + AB

theorem find_PB (AB : ℝ) (AB_condition : AB = PB AB - PA) :
  PB AB = (81 + Real.sqrt 5117) / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_PB_l328_32844


namespace NUMINAMATH_GPT_number_of_ways_to_draw_balls_l328_32827

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_draw_balls_l328_32827


namespace NUMINAMATH_GPT_find_common_difference_l328_32825

def is_arithmetic_sequence (a : (ℕ → ℝ)) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def is_arithmetic_sequence_with_sum (a : (ℕ → ℝ)) (S : (ℕ → ℝ)) (d : ℝ) : Prop :=
  S 0 = a 0 ∧
  ∀ n, S (n + 1) = S n + a (n + 1) ∧
        ∀ n, (S (n + 1) / a (n + 1) - S n / a n) = d

theorem find_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a →
  is_arithmetic_sequence_with_sum a S d →
  (d = 1 ∨ d = 1 / 2) :=
sorry

end NUMINAMATH_GPT_find_common_difference_l328_32825


namespace NUMINAMATH_GPT_Thabo_books_problem_l328_32850

theorem Thabo_books_problem 
  (P F : ℕ)
  (H1 : 180 = F + P + 30)
  (H2 : F = 2 * P)
  (H3 : P > 30) :
  P - 30 = 20 := 
sorry

end NUMINAMATH_GPT_Thabo_books_problem_l328_32850


namespace NUMINAMATH_GPT_PlatformC_location_l328_32894

noncomputable def PlatformA : ℝ := 9
noncomputable def PlatformB : ℝ := 1 / 3
noncomputable def PlatformC : ℝ := 7
noncomputable def AB := PlatformA - PlatformB
noncomputable def AC := PlatformA - PlatformC

theorem PlatformC_location :
  AB = (13 / 3) * AC → PlatformC = 7 :=
by
  intro h
  simp [AB, AC, PlatformA, PlatformB, PlatformC] at h
  sorry

end NUMINAMATH_GPT_PlatformC_location_l328_32894


namespace NUMINAMATH_GPT_ordered_pair_count_l328_32869

theorem ordered_pair_count :
  (∃ (bc : ℕ × ℕ), bc.1 > 0 ∧ bc.2 > 0 ∧ bc.1 ^ 4 - 4 * bc.2 ≤ 0 ∧ bc.2 ^ 4 - 4 * bc.1 ≤ 0) ∧
  ∀ (bc1 bc2 : ℕ × ℕ),
    bc1 ≠ bc2 →
    bc1.1 > 0 ∧ bc1.2 > 0 ∧ bc1.1 ^ 4 - 4 * bc1.2 ≤ 0 ∧ bc1.2 ^ 4 - 4 * bc1.1 ≤ 0 →
    bc2.1 > 0 ∧ bc2.2 > 0 ∧ bc2.1 ^ 4 - 4 * bc2.2 ≤ 0 ∧ bc2.2 ^ 4 - 4 * bc2.1 ≤ 0 →
    false
:=
sorry

end NUMINAMATH_GPT_ordered_pair_count_l328_32869


namespace NUMINAMATH_GPT_stable_table_configurations_l328_32829

noncomputable def numberOfStableConfigurations (n : ℕ) : ℕ :=
  1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)

theorem stable_table_configurations (n : ℕ) (hn : 0 < n) :
  numberOfStableConfigurations n = 
    (1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)) :=
by
  sorry

end NUMINAMATH_GPT_stable_table_configurations_l328_32829


namespace NUMINAMATH_GPT_series_solution_l328_32863

theorem series_solution (r : ℝ) (h : (r^3 - r^2 + (1 / 4) * r - 1 = 0) ∧ r > 0) :
  (∑' (n : ℕ), (n + 1) * r^(3 * (n + 1))) = 16 * r :=
by
  sorry

end NUMINAMATH_GPT_series_solution_l328_32863


namespace NUMINAMATH_GPT_coffee_break_l328_32807

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end NUMINAMATH_GPT_coffee_break_l328_32807


namespace NUMINAMATH_GPT_reciprocal_fraction_addition_l328_32823

theorem reciprocal_fraction_addition (a b c : ℝ) (h : a ≠ b) :
  (a + c) / (b + c) = b / a ↔ c = - (a + b) := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_fraction_addition_l328_32823


namespace NUMINAMATH_GPT_positive_difference_l328_32836

theorem positive_difference (y : ℤ) (h : (46 + y) / 2 = 52) : |y - 46| = 12 := by
  sorry

end NUMINAMATH_GPT_positive_difference_l328_32836


namespace NUMINAMATH_GPT_moles_of_HCl_combined_eq_one_l328_32864

-- Defining the chemical species involved in the reaction
def NaHCO3 : Type := Nat
def HCl : Type := Nat
def NaCl : Type := Nat
def H2O : Type := Nat
def CO2 : Type := Nat

-- Defining the balanced chemical equation as a condition
def reaction (n_NaHCO3 n_HCl n_NaCl n_H2O n_CO2 : Nat) : Prop :=
  n_NaHCO3 + n_HCl = n_NaCl + n_H2O + n_CO2

-- Given conditions
def one_mole_of_NaHCO3 : Nat := 1
def one_mole_of_NaCl_produced : Nat := 1

-- Proof problem
theorem moles_of_HCl_combined_eq_one :
  ∃ (n_HCl : Nat), reaction one_mole_of_NaHCO3 n_HCl one_mole_of_NaCl_produced 1 1 ∧ n_HCl = 1 := 
by
  sorry

end NUMINAMATH_GPT_moles_of_HCl_combined_eq_one_l328_32864


namespace NUMINAMATH_GPT_ratio_students_l328_32855

theorem ratio_students
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (h_finley : finley_students = 24)
  (h_johnson : johnson_students = 22)
  : (johnson_students : ℚ) / ((finley_students / 2 : ℕ) : ℚ) = 11 / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_students_l328_32855


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l328_32830

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 1 / 4 > 0)) = ∃ x : ℝ, x^2 - x + 1 / 4 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l328_32830


namespace NUMINAMATH_GPT_cos_five_pi_over_four_l328_32835

theorem cos_five_pi_over_four : Real.cos (5 * Real.pi / 4) = -1 / Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_five_pi_over_four_l328_32835


namespace NUMINAMATH_GPT_find_discount_percentage_l328_32872

noncomputable def discount_percentage (P B S : ℝ) (H1 : B = P * (1 - D / 100)) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : ℝ :=
D

theorem find_discount_percentage (P B S : ℝ) (H1 : B = P * (1 - (60 / 100))) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : 
  discount_percentage P B S H1 H2 H3 = 60 := sorry

end NUMINAMATH_GPT_find_discount_percentage_l328_32872


namespace NUMINAMATH_GPT_pool_capacity_percentage_l328_32880

theorem pool_capacity_percentage
  (rate : ℕ := 60) -- cubic feet per minute
  (time : ℕ := 800) -- minutes
  (width : ℕ := 60) -- feet
  (length : ℕ := 100) -- feet
  (depth : ℕ := 10) -- feet
  : (rate * time * 100) / (width * length * depth) = 8 := by
{
  sorry
}

end NUMINAMATH_GPT_pool_capacity_percentage_l328_32880


namespace NUMINAMATH_GPT_tangents_quadrilateral_cyclic_l328_32840

variables {A B C D K L O1 O2 : Point}
variable (r : ℝ)
variable (AB_cut_circles : ∀ {A B : Point} {O1 O2 : Point}, is_intersect AB O1 O2)
variable (parallel_AB_O1O2 : is_parallel AB O1O2)
variable (tangents_formed_quadrilateral : is_quadrilateral C D K L)
variable (quadrilateral_contains_circles : contains C D K L O1 O2)

theorem tangents_quadrilateral_cyclic
  (h1: AB_cut_circles)
  (h2: parallel_AB_O1O2) 
  (h3: tangents_formed_quadrilateral)
  (h4: quadrilateral_contains_circles)
  : ∃ O : Circle, is_inscribed O C D K L :=
sorry

end NUMINAMATH_GPT_tangents_quadrilateral_cyclic_l328_32840


namespace NUMINAMATH_GPT_average_score_in_5_matches_l328_32813

theorem average_score_in_5_matches 
  (avg1 avg2 : ℕ)
  (total_matches1 total_matches2 : ℕ)
  (h1 : avg1 = 27) 
  (h2 : avg2 = 32)
  (h3 : total_matches1 = 2) 
  (h4 : total_matches2 = 3) 
  : 
  (avg1 * total_matches1 + avg2 * total_matches2) / (total_matches1 + total_matches2) = 30 :=
by 
  sorry

end NUMINAMATH_GPT_average_score_in_5_matches_l328_32813


namespace NUMINAMATH_GPT_remainder_698_div_D_l328_32803

-- Defining the conditions
variables (D k1 k2 k3 R : ℤ)

-- Given conditions
axiom condition1 : 242 = k1 * D + 4
axiom condition2 : 940 = k3 * D + 7
axiom condition3 : 698 = k2 * D + R

-- The theorem to prove the remainder 
theorem remainder_698_div_D : R = 3 :=
by
  -- Here you would provide the logical deduction steps
  sorry

end NUMINAMATH_GPT_remainder_698_div_D_l328_32803


namespace NUMINAMATH_GPT_constant_term_in_expansion_l328_32826

noncomputable def P (x : ℕ) : ℕ := x^4 + 2 * x + 7
noncomputable def Q (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 + 10

theorem constant_term_in_expansion :
  (P 0) * (Q 0) = 70 := 
sorry

end NUMINAMATH_GPT_constant_term_in_expansion_l328_32826


namespace NUMINAMATH_GPT_tenth_pirate_receives_exactly_1296_coins_l328_32832

noncomputable def pirate_coins (n : ℕ) : ℕ :=
  if n = 0 then 0
  else Nat.factorial 9 / 11^9 * 11^(10 - n)

theorem tenth_pirate_receives_exactly_1296_coins :
  pirate_coins 10 = 1296 :=
sorry

end NUMINAMATH_GPT_tenth_pirate_receives_exactly_1296_coins_l328_32832


namespace NUMINAMATH_GPT_point_of_tangency_l328_32892

theorem point_of_tangency : 
    ∃ (m n : ℝ), 
    (∀ x : ℝ, x ≠ 0 → n = 1 / m ∧ (-1 / m^2) = (n - 2) / (m - 0)) ∧ 
    m = 1 ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_point_of_tangency_l328_32892


namespace NUMINAMATH_GPT_C_investment_is_20000_l328_32820

-- Definitions of investments and profits
def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def total_profit : ℕ := 86400
def C_share_of_profit : ℕ := 36000

-- The proof problem statement
theorem C_investment_is_20000 (X : ℕ) (hA : A_investment = 12000) (hB : B_investment = 16000)
  (h_total_profit : total_profit = 86400) (h_C_share_of_profit : C_share_of_profit = 36000) :
  X = 20000 :=
sorry

end NUMINAMATH_GPT_C_investment_is_20000_l328_32820


namespace NUMINAMATH_GPT_no_real_solutions_for_equation_l328_32858

theorem no_real_solutions_for_equation:
  ∀ x : ℝ, (3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1) →
  (¬(∃ x : ℝ, 3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1)) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_for_equation_l328_32858


namespace NUMINAMATH_GPT_minimize_g_function_l328_32815

noncomputable def g (x : ℝ) : ℝ := (9 * x^2 + 18 * x + 29) / (8 * (2 + x))

theorem minimize_g_function : ∀ x : ℝ, x ≥ -1 → g x = 29 / 8 :=
sorry

end NUMINAMATH_GPT_minimize_g_function_l328_32815
