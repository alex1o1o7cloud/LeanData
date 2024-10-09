import Mathlib

namespace OM_geq_ON_l1558_155881

variables {A B C D E F G H P Q M N O : Type*}

-- Definitions for geometrical concepts
def is_intersection_of_diagonals (M : Type*) (A B C D : Type*) : Prop :=
-- M is the intersection of the diagonals AC and BD
sorry

def is_intersection_of_midlines (N : Type*) (A B C D : Type*) : Prop :=
-- N is the intersection of the midlines connecting the midpoints of opposite sides
sorry

def is_center_of_circumscribed_circle (O : Type*) (A B C D : Type*) : Prop :=
-- O is the center of the circumscribed circle around quadrilateral ABCD
sorry

-- Proof problem
theorem OM_geq_ON (A B C D M N O : Type*) 
  (hm : is_intersection_of_diagonals M A B C D)
  (hn : is_intersection_of_midlines N A B C D)
  (ho : is_center_of_circumscribed_circle O A B C D) : 
  ∃ (OM ON : ℝ), OM ≥ ON :=
sorry

end OM_geq_ON_l1558_155881


namespace incorrect_ratio_implies_l1558_155886

variable {a b c d : ℝ} (h : a * d = b * c) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

theorem incorrect_ratio_implies :
  ¬ (c / b = a / d) :=
sorry

end incorrect_ratio_implies_l1558_155886


namespace balls_balance_l1558_155882

theorem balls_balance (G Y W B : ℕ) (h1 : G = 2 * B) (h2 : Y = 5 * B / 2) (h3 : W = 3 * B / 2) :
  5 * G + 3 * Y + 3 * W = 22 * B :=
by
  sorry

end balls_balance_l1558_155882


namespace line_passes_through_fixed_point_l1558_155861

theorem line_passes_through_fixed_point (m : ℝ) :
  (m-1) * 9 + (2*m-1) * (-4) = m - 5 :=
by
  sorry

end line_passes_through_fixed_point_l1558_155861


namespace quadratic_polynomial_half_coefficient_l1558_155818

theorem quadratic_polynomial_half_coefficient :
  ∃ b c : ℚ, ∀ x : ℤ, ∃ k : ℤ, (1/2 : ℚ) * (x^2 : ℚ) + b * (x : ℚ) + c = (k : ℚ) :=
by
  sorry

end quadratic_polynomial_half_coefficient_l1558_155818


namespace students_play_both_football_and_tennis_l1558_155880

theorem students_play_both_football_and_tennis 
  (T : ℕ) (F : ℕ) (L : ℕ) (N : ℕ) (B : ℕ)
  (hT : T = 38) (hF : F = 26) (hL : L = 20) (hN : N = 9) :
  B = F + L - (T - N) → B = 17 :=
by 
  intros h
  rw [hT, hF, hL, hN] at h
  exact h

end students_play_both_football_and_tennis_l1558_155880


namespace quadratic_real_roots_condition_l1558_155846

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) ↔ (a ≥ 1 ∧ a ≠ 5) :=
by
  sorry

end quadratic_real_roots_condition_l1558_155846


namespace smallest_number_is_21_5_l1558_155863

-- Definitions of the numbers in their respective bases
def num1 := 3 * 4^0 + 3 * 4^1
def num2 := 0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3
def num3 := 2 * 3^0 + 2 * 3^1 + 1 * 3^2
def num4 := 1 * 5^0 + 2 * 5^1

-- Statement asserting that num4 is the smallest number
theorem smallest_number_is_21_5 : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 := by
  sorry

end smallest_number_is_21_5_l1558_155863


namespace rectangle_width_l1558_155810

-- Conditions
def length (w : Real) : Real := 4 * w
def area (w : Real) : Real := w * length w

-- Theorem stating that the width of the rectangle is 5 inches if the area is 100 square inches
theorem rectangle_width (h : area w = 100) : w = 5 :=
sorry

end rectangle_width_l1558_155810


namespace find_a_l1558_155832

theorem find_a (a : ℝ) :
  (∃ x : ℝ, (a + 1) * x^2 - x + a^2 - 2*a - 2 = 0 ∧ x = 1) → a = 2 :=
by
  sorry

end find_a_l1558_155832


namespace max_tulips_l1558_155851

theorem max_tulips (y r : ℕ) (h1 : (y + r) % 2 = 1) (h2 : r = y + 1 ∨ y = r + 1) (h3 : 50 * y + 31 * r ≤ 600) : y + r = 15 :=
by
  sorry

end max_tulips_l1558_155851


namespace number_of_tables_l1558_155878

-- Define the conditions
def seats_per_table : ℕ := 8
def total_seating_capacity : ℕ := 32

-- Define the main statement using the conditions
theorem number_of_tables : total_seating_capacity / seats_per_table = 4 := by
  sorry

end number_of_tables_l1558_155878


namespace product_evaluation_l1558_155868

theorem product_evaluation : 
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end product_evaluation_l1558_155868


namespace triangle_construction_condition_l1558_155817

variable (varrho_a varrho_b m_c : ℝ)

theorem triangle_construction_condition :
  (∃ (triangle : Type) (ABC : triangle)
    (r_a : triangle → ℝ)
    (r_b : triangle → ℝ)
    (h_from_C : triangle → ℝ),
      r_a ABC = varrho_a ∧
      r_b ABC = varrho_b ∧
      h_from_C ABC = m_c)
  ↔ 
  (1 / m_c = 1 / 2 * (1 / varrho_a + 1 / varrho_b)) :=
sorry

end triangle_construction_condition_l1558_155817


namespace price_of_peas_l1558_155864

theorem price_of_peas
  (P : ℝ) -- price of peas per kg in rupees
  (price_soybeans : ℝ) (price_mixture : ℝ)
  (ratio_peas_soybeans : ℝ) :
  price_soybeans = 25 →
  price_mixture = 19 →
  ratio_peas_soybeans = 2 →
  P = 16 :=
by
  intros h_price_soybeans h_price_mixture h_ratio
  sorry

end price_of_peas_l1558_155864


namespace original_cost_of_car_l1558_155844

theorem original_cost_of_car (C : ℝ)
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (h1 : repairs_cost = 14000)
  (h2 : selling_price = 72900)
  (h3 : profit_percent = 17.580645161290324)
  (h4 : profit_percent = ((selling_price - (C + repairs_cost)) / C) * 100) :
  C = 50075 := 
sorry

end original_cost_of_car_l1558_155844


namespace range_of_k_for_ellipse_l1558_155866

theorem range_of_k_for_ellipse (k : ℝ) :
  (4 - k > 0) ∧ (k - 1 > 0) ∧ (4 - k ≠ k - 1) ↔ (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  sorry

end range_of_k_for_ellipse_l1558_155866


namespace expression_equals_66069_l1558_155840

-- Definitions based on the conditions
def numerator : Nat := 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10
def denominator : Nat := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
def expression : Rat := numerator / denominator

-- The main theorem to be proven
theorem expression_equals_66069 : expression = 66069 := by
  sorry

end expression_equals_66069_l1558_155840


namespace rival_awards_l1558_155848

theorem rival_awards (S J R : ℕ) (h1 : J = 3 * S) (h2 : S = 4) (h3 : R = 2 * J) : R = 24 := 
by sorry

end rival_awards_l1558_155848


namespace parallel_lines_condition_l1558_155802

theorem parallel_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * m * x + y + 6 = 0 → (m - 3) * x - y + 7 = 0) → m = 1 :=
by
  sorry

end parallel_lines_condition_l1558_155802


namespace minimum_triangle_perimeter_l1558_155845

def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem minimum_triangle_perimeter (l m n : ℕ) (h1 : l > m) (h2 : m > n)
  (h3 : fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4)) 
  (h4 : fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)) :
   l + m + n = 3003 := 
sorry

end minimum_triangle_perimeter_l1558_155845


namespace find_f_of_2_l1558_155884

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^3 + x^2 else 0

theorem find_f_of_2 :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, x < 0 → f x = x^3 + x^2) → f 2 = 4 :=
by
  intros h_odd h_def_neg
  sorry

end find_f_of_2_l1558_155884


namespace lionel_initial_boxes_crackers_l1558_155856

/--
Lionel went to the grocery store and bought some boxes of Graham crackers and 15 packets of Oreos. 
To make an Oreo cheesecake, Lionel needs 2 boxes of Graham crackers and 3 packets of Oreos. 
After making the maximum number of Oreo cheesecakes he can with the ingredients he bought, 
he had 4 boxes of Graham crackers left over. 

The number of boxes of Graham crackers Lionel initially bought is 14.
-/
theorem lionel_initial_boxes_crackers (G : ℕ) (h1 : G - 4 = 10) : G = 14 := 
by sorry

end lionel_initial_boxes_crackers_l1558_155856


namespace no_good_polygon_in_division_of_equilateral_l1558_155870

def is_equilateral_polygon (P : List Point) : Prop :=
  -- Definition of equilateral polygon
  sorry

def is_good_polygon (P : List Point) : Prop :=
  -- Definition of good polygon (having a pair of parallel sides)
  sorry

def is_divided_by_non_intersecting_diagonals (P : List Point) (polygons : List (List Point)) : Prop :=
  -- Definition for dividing by non-intersecting diagonals into several polygons
  sorry

def have_same_odd_sides (polygons : List (List Point)) : Prop :=
  -- Definition for all polygons having the same odd number of sides
  sorry

theorem no_good_polygon_in_division_of_equilateral (P : List Point) (polygons : List (List Point)) :
  is_equilateral_polygon P →
  is_divided_by_non_intersecting_diagonals P polygons →
  have_same_odd_sides polygons →
  ¬ ∃ gp ∈ polygons, is_good_polygon gp :=
by
  intro h_eq h_div h_odd
  intro h_good
  -- Proof goes here
  sorry

end no_good_polygon_in_division_of_equilateral_l1558_155870


namespace tangent_to_parabola_l1558_155872

theorem tangent_to_parabola {k : ℝ} : 
  (∀ x y : ℝ, (4 * x + 3 * y + k = 0) ↔ (y ^ 2 = 16 * x)) → k = 9 :=
by
  sorry

end tangent_to_parabola_l1558_155872


namespace equation_holds_if_a_eq_neg_b_c_l1558_155897

-- Define the conditions and equation
variables {a b c : ℝ} (h1 : a ≠ 0) (h2 : a + b ≠ 0)

-- Statement to be proved
theorem equation_holds_if_a_eq_neg_b_c : 
  (a = -(b + c)) ↔ (a + b + c) / a = (b + c) / (a + b) := 
sorry

end equation_holds_if_a_eq_neg_b_c_l1558_155897


namespace roots_cube_reciprocal_eqn_l1558_155860

variable (a b c r s : ℝ)

def quadratic_eqn (r s : ℝ) : Prop :=
  3 * a * r ^ 2 + 5 * b * r + 7 * c = 0 ∧ 
  3 * a * s ^ 2 + 5 * b * s + 7 * c = 0

theorem roots_cube_reciprocal_eqn (h : quadratic_eqn a b c r s) :
  (1 / r^3 + 1 / s^3) = (-5 * b * (25 * b ^ 2 - 63 * c) / (343 * c^3)) :=
sorry

end roots_cube_reciprocal_eqn_l1558_155860


namespace estimated_fish_in_pond_l1558_155885

theorem estimated_fish_in_pond :
  ∀ (number_marked_first_catch total_second_catch number_marked_second_catch : ℕ),
    number_marked_first_catch = 100 →
    total_second_catch = 108 →
    number_marked_second_catch = 9 →
    ∃ est_total_fish : ℕ, (number_marked_second_catch / total_second_catch : ℝ) = (number_marked_first_catch / est_total_fish : ℝ) ∧ est_total_fish = 1200 := 
by
  intros number_marked_first_catch total_second_catch number_marked_second_catch
  sorry

end estimated_fish_in_pond_l1558_155885


namespace investment_amount_l1558_155875

noncomputable def PV (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r) ^ n

theorem investment_amount (FV : ℝ) (r : ℝ) (n : ℕ) (PV : ℝ) : FV = 1000000 ∧ r = 0.08 ∧ n = 20 → PV = 1000000 / (1 + 0.08)^20 :=
by
  intros
  sorry

end investment_amount_l1558_155875


namespace quadrilateral_diagonals_inequality_l1558_155836

theorem quadrilateral_diagonals_inequality (a b c d e f : ℝ) :
  e^2 + f^2 ≤ b^2 + d^2 + 2 * a * c :=
by
  sorry

end quadrilateral_diagonals_inequality_l1558_155836


namespace window_width_is_28_l1558_155876

noncomputable def window_width (y : ℝ) : ℝ :=
  12 * y + 4

theorem window_width_is_28 : ∃ (y : ℝ), window_width y = 28 :=
by
  -- The proof goes here
  sorry

end window_width_is_28_l1558_155876


namespace rectangle_length_l1558_155852

theorem rectangle_length (side_of_square : ℕ) (width_of_rectangle : ℕ) (same_wire_length : ℕ) 
(side_eq : side_of_square = 12) (width_eq : width_of_rectangle = 6) 
(square_perimeter : same_wire_length = 4 * side_of_square) :
  ∃ (length_of_rectangle : ℕ), 2 * (length_of_rectangle + width_of_rectangle) = same_wire_length ∧ length_of_rectangle = 18 :=
by
  sorry

end rectangle_length_l1558_155852


namespace solve_system_of_inequalities_l1558_155839

theorem solve_system_of_inequalities (x y : ℤ) :
  (2 * x - y > 3 ∧ 3 - 2 * x + y > 0) ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := 
by { sorry }

end solve_system_of_inequalities_l1558_155839


namespace number_of_posts_needed_l1558_155806

-- Define the conditions
def length_of_field : ℕ := 80
def width_of_field : ℕ := 60
def distance_between_posts : ℕ := 10

-- Statement to prove the number of posts needed to completely fence the field
theorem number_of_posts_needed : 
  (2 * (length_of_field / distance_between_posts + 1) + 
   2 * (width_of_field / distance_between_posts + 1) - 
   4) = 28 := 
by
  -- Skipping the proof for this theorem
  sorry

end number_of_posts_needed_l1558_155806


namespace anna_interest_l1558_155841

noncomputable def interest_earned (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n - P

theorem anna_interest : interest_earned 2000 0.08 5 = 938.66 := by
  sorry

end anna_interest_l1558_155841


namespace pentagon_PTRSQ_area_proof_l1558_155869

-- Define the geometric setup and properties
def quadrilateral_PQRS_is_square (P Q R S T : Type) : Prop :=
  -- Here, we will skip the precise geometric construction and assume the properties directly.
  sorry

def segment_PT_perpendicular_to_TR (P T R : Type) : Prop :=
  sorry

def PT_eq_5 (PT : ℝ) : Prop :=
  PT = 5

def TR_eq_12 (TR : ℝ) : Prop :=
  TR = 12

def area_PTRSQ (area : ℝ) : Prop :=
  area = 139

theorem pentagon_PTRSQ_area_proof
  (P Q R S T : Type)
  (PQRS_is_square : quadrilateral_PQRS_is_square P Q R S T)
  (PT_perpendicular_TR : segment_PT_perpendicular_to_TR P T R)
  (PT_length : PT_eq_5 5)
  (TR_length : TR_eq_12 12)
  : area_PTRSQ 139 :=
  sorry

end pentagon_PTRSQ_area_proof_l1558_155869


namespace people_in_group_l1558_155877

theorem people_in_group (n : ℕ) 
  (h1 : ∀ (new_weight old_weight : ℕ), old_weight = 70 → new_weight = 110 → (70 * n + (new_weight - old_weight) = 70 * n + 4 * n)) :
  n = 10 :=
sorry

end people_in_group_l1558_155877


namespace complex_number_identity_l1558_155822

theorem complex_number_identity (a b : ℝ) (i : ℂ) (h : (a + i) * (1 + i) = b * i) : a + b * i = 1 + 2 * i := 
by
  sorry

end complex_number_identity_l1558_155822


namespace square_area_from_hexagon_l1558_155831

theorem square_area_from_hexagon (hex_side length square_side : ℝ) (h1 : hex_side = 4) (h2 : length = 6 * hex_side)
  (h3 : square_side = length / 4) : square_side ^ 2 = 36 :=
by 
  sorry

end square_area_from_hexagon_l1558_155831


namespace algebraic_expression_value_l1558_155843

/-- Given \( x^2 - 5x - 2006 = 0 \), prove that the expression \(\frac{(x-2)^3 - (x-1)^2 + 1}{x-2}\) is equal to 2010. -/
theorem algebraic_expression_value (x : ℝ) (h: x^2 - 5 * x - 2006 = 0) :
  ( (x - 2)^3 - (x - 1)^2 + 1 ) / (x - 2) = 2010 :=
by
  sorry

end algebraic_expression_value_l1558_155843


namespace value_makes_expression_undefined_l1558_155847

theorem value_makes_expression_undefined (a : ℝ) : 
    (a^2 - 9 * a + 20 = 0) ↔ (a = 4 ∨ a = 5) :=
by
  sorry

end value_makes_expression_undefined_l1558_155847


namespace pyramid_surface_area_l1558_155828

noncomputable def total_surface_area (a : ℝ) : ℝ :=
  a^2 * (6 + 3 * Real.sqrt 3 + Real.sqrt 7) / 2

theorem pyramid_surface_area (a : ℝ) :
  let hexagon_base_area := 3 * a^2 * Real.sqrt 3 / 2
  let triangle_area_1 := a^2 / 2
  let triangle_area_2 := a^2
  let triangle_area_3 := a^2 * Real.sqrt 7 / 4
  let lateral_area := 2 * (triangle_area_1 + triangle_area_2 + triangle_area_3)
  total_surface_area a = hexagon_base_area + lateral_area := 
sorry

end pyramid_surface_area_l1558_155828


namespace correct_statement_d_l1558_155805

theorem correct_statement_d (x : ℝ) : 2 * (x + 1) = x + 7 → x = 5 :=
by
  sorry

end correct_statement_d_l1558_155805


namespace g_neg_one_l1558_155862

variables {F : Type*} [Field F]

def odd_function (f : F → F) := ∀ x, f (-x) = -f x

variables (f : F → F) (g : F → F)

-- Given conditions
lemma given_conditions :
  (∀ x, f (-x) + (-x)^2 = -(f x + x^2)) ∧
  f 1 = 1 ∧
  (∀ x, g x = f x + 2) :=
sorry

-- Prove that g(-1) = -1
theorem g_neg_one :
  g (-1) = -1 :=
sorry

end g_neg_one_l1558_155862


namespace find_divisor_value_l1558_155835

theorem find_divisor_value (x : ℝ) (h : 63 / x = 63 - 42) : x = 3 :=
by
  sorry

end find_divisor_value_l1558_155835


namespace minimum_degree_q_l1558_155893

variable (p q r : Polynomial ℝ)

theorem minimum_degree_q (h1 : 2 * p + 5 * q = r)
                        (hp : p.degree = 7)
                        (hr : r.degree = 10) :
  q.degree = 10 :=
sorry

end minimum_degree_q_l1558_155893


namespace sum_of_coordinates_D_l1558_155814

structure Point where
  x : ℝ
  y : ℝ

def is_midpoint (M C D : Point) : Prop :=
  M = ⟨(C.x + D.x) / 2, (C.y + D.y) / 2⟩

def sum_of_coordinates (P : Point) : ℝ :=
  P.x + P.y

theorem sum_of_coordinates_D :
  ∀ (C M : Point), C = ⟨1/2, 3/2⟩ → M = ⟨2, 5⟩ →
  ∃ D : Point, is_midpoint M C D ∧ sum_of_coordinates D = 12 :=
by
  intros C M hC hM
  sorry

end sum_of_coordinates_D_l1558_155814


namespace num_brownies_correct_l1558_155807

-- Define the conditions (pan dimensions and brownie piece dimensions)
def pan_width : ℕ := 24
def pan_length : ℕ := 15
def piece_width : ℕ := 3
def piece_length : ℕ := 2

-- Define the area calculations for the pan and each piece
def pan_area : ℕ := pan_width * pan_length
def piece_area : ℕ := piece_width * piece_length

-- Define the problem statement to prove the number of brownies
def number_of_brownies : ℕ := pan_area / piece_area

-- The statement we need to prove
theorem num_brownies_correct : number_of_brownies = 60 :=
by
  sorry

end num_brownies_correct_l1558_155807


namespace michael_brought_5000_rubber_bands_l1558_155803

noncomputable def totalRubberBands
  (small_band_count : ℕ) (large_band_count : ℕ)
  (small_ball_count : ℕ := 22) (large_ball_count : ℕ := 13)
  (rubber_bands_per_small : ℕ := 50) (rubber_bands_per_large : ℕ := 300) 
: ℕ :=
small_ball_count * rubber_bands_per_small + large_ball_count * rubber_bands_per_large

theorem michael_brought_5000_rubber_bands :
  totalRubberBands 22 13 = 5000 := by
  sorry

end michael_brought_5000_rubber_bands_l1558_155803


namespace parallel_vectors_sum_l1558_155849

variable (x y : ℝ)
variable (k : ℝ)

theorem parallel_vectors_sum :
  (k * 3 = 2) ∧ (k * x = 4) ∧ (k * y = 5) → x + y = 27 / 2 :=
by
  sorry

end parallel_vectors_sum_l1558_155849


namespace derivative_at_1_of_f_l1558_155896

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1_of_f :
  (deriv f 1) = 2 * Real.log 2 - 3 :=
sorry

end derivative_at_1_of_f_l1558_155896


namespace isaac_ribbon_length_l1558_155834

variable (part_length : ℝ) (total_length : ℝ := part_length * 6) (unused_length : ℝ := part_length * 2)

theorem isaac_ribbon_length
  (total_parts : ℕ := 6)
  (used_parts : ℕ := 4)
  (not_used_parts : ℕ := total_parts - used_parts)
  (not_used_length : Real := 10)
  (equal_parts : total_length / total_parts = part_length) :
  total_length = 30 := by
  sorry

end isaac_ribbon_length_l1558_155834


namespace cost_to_replace_and_install_l1558_155895

theorem cost_to_replace_and_install (s l : ℕ) 
  (h1 : l = 3 * s) (h2 : 2 * s + 2 * l = 640) 
  (cost_per_foot : ℕ) (cost_per_gate : ℕ) (installation_cost_per_gate : ℕ) 
  (h3 : cost_per_foot = 5) (h4 : cost_per_gate = 150) (h5 : installation_cost_per_gate = 75) : 
  (s * cost_per_foot + 2 * (cost_per_gate + installation_cost_per_gate)) = 850 := 
by 
  sorry

end cost_to_replace_and_install_l1558_155895


namespace page_shoes_count_l1558_155804

theorem page_shoes_count (p_i : ℕ) (d : ℝ) (b : ℕ) (h1 : p_i = 120) (h2 : d = 0.45) (h3 : b = 15) : 
  (p_i - (d * p_i)) + b = 81 :=
by
  sorry

end page_shoes_count_l1558_155804


namespace length_sum_l1558_155887

theorem length_sum : 
  let m := 1 -- Meter as base unit
  let cm := 0.01 -- 1 cm in meters
  let mm := 0.001 -- 1 mm in meters
  2 * m + 3 * cm + 5 * mm = 2.035 * m :=
by sorry

end length_sum_l1558_155887


namespace option_B_is_incorrect_l1558_155873

-- Define the set A
def A := { x : ℤ | x ^ 2 - 4 = 0 }

-- Statement to prove that -2 is an element of A
theorem option_B_is_incorrect : -2 ∈ A :=
sorry

end option_B_is_incorrect_l1558_155873


namespace revenue_from_full_price_tickets_l1558_155826

theorem revenue_from_full_price_tickets (f h p : ℕ) (h1 : f + h = 160) (h2 : f * p + h * (p / 2) = 2400) : f * p = 1600 :=
by
  sorry

end revenue_from_full_price_tickets_l1558_155826


namespace no_nonzero_solutions_l1558_155874

theorem no_nonzero_solutions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + x = y^2 - y) ∧ (y^2 + y = z^2 - z) ∧ (z^2 + z = x^2 - x) → false :=
by
  sorry

end no_nonzero_solutions_l1558_155874


namespace range_of_omega_l1558_155853

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f ω x = 0 → 
      (∃ x₁ x₂, x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 
        0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧ f ω x₁ = 0 ∧ f ω x₂ = 0)) ↔ 2 ≤ ω ∧ ω < 4 :=
sorry

end range_of_omega_l1558_155853


namespace common_ratio_of_geometric_sequence_l1558_155899

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def is_geometric_sequence (x y z : ℝ) (q : ℝ) : Prop :=
  y^2 = x * z

theorem common_ratio_of_geometric_sequence 
    (a_n : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a_n) 
    (a1 a3 a5 : ℝ)
    (h1 : a1 = a_n 1 + 1) 
    (h3 : a3 = a_n 3 + 3) 
    (h5 : a5 = a_n 5 + 5) 
    (h_geom : is_geometric_sequence a1 a3 a5 1) : 
  1 = 1 :=
by
  sorry

end common_ratio_of_geometric_sequence_l1558_155899


namespace crescents_area_eq_rectangle_area_l1558_155850

noncomputable def rectangle_area (a b : ℝ) : ℝ := 4 * a * b

noncomputable def semicircle_area (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

noncomputable def circumscribed_circle_area (a b : ℝ) : ℝ :=
  Real.pi * (a^2 + b^2)

noncomputable def combined_area (a b : ℝ) : ℝ :=
  rectangle_area a b + 2 * (semicircle_area a) + 2 * (semicircle_area b)

theorem crescents_area_eq_rectangle_area (a b : ℝ) : 
  combined_area a b - circumscribed_circle_area a b = rectangle_area a b :=
by
  unfold combined_area
  unfold circumscribed_circle_area
  unfold rectangle_area
  unfold semicircle_area
  sorry

end crescents_area_eq_rectangle_area_l1558_155850


namespace remainder_5n_div_3_l1558_155825

theorem remainder_5n_div_3 (n : ℤ) (h : n % 3 = 2) : (5 * n) % 3 = 1 := by
  sorry

end remainder_5n_div_3_l1558_155825


namespace find_m_l1558_155867

-- Defining vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b : ℝ × ℝ := (1, -1)

-- Proving that if b is perpendicular to (a + 2b), then m = 6
theorem find_m (m : ℝ) :
  let a_vec := a m
  let b_vec := b
  let sum_vec := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (b_vec.1 * sum_vec.1 + b_vec.2 * sum_vec.2 = 0) → m = 6 :=
by
  intros a_vec b_vec sum_vec perp_cond
  sorry

end find_m_l1558_155867


namespace time_to_school_building_l1558_155833

theorem time_to_school_building 
  (total_time : ℕ := 30) 
  (time_to_gate : ℕ := 15) 
  (time_to_room : ℕ := 9)
  (remaining_time := total_time - time_to_gate - time_to_room) : 
  remaining_time = 6 :=
by
  sorry

end time_to_school_building_l1558_155833


namespace number_of_female_students_school_l1558_155823

theorem number_of_female_students_school (T S G_s B_s B G : ℕ) (h1 : T = 1600)
    (h2 : S = 200) (h3 : G_s = B_s - 10) (h4 : G_s + B_s = 200) (h5 : B_s = 105) (h6 : G_s = 95) (h7 : B + G = 1600) : 
    G = 760 :=
by
  sorry

end number_of_female_students_school_l1558_155823


namespace arithmetic_sequence_a3_value_l1558_155889

theorem arithmetic_sequence_a3_value {a : ℕ → ℕ}
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
sorry

end arithmetic_sequence_a3_value_l1558_155889


namespace num_triangles_in_circle_l1558_155858

noncomputable def num_triangles (n : ℕ) : ℕ :=
  n.choose 3

theorem num_triangles_in_circle (n : ℕ) :
  num_triangles n = n.choose 3 :=
by
  sorry

end num_triangles_in_circle_l1558_155858


namespace number_of_valid_numbers_l1558_155809

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def four_digit_number_conditions : Prop :=
  (∀ N : ℕ, 7000 ≤ N ∧ N < 9000 → 
    (N % 5 = 0) →
    (∃ a b c d : ℕ, 
      N = 1000 * a + 100 * b + 10 * c + d ∧
      (a = 7 ∨ a = 8) ∧
      (d = 0 ∨ d = 5) ∧
      3 ≤ b ∧ is_prime b ∧ b < c ∧ c ≤ 7))

theorem number_of_valid_numbers : four_digit_number_conditions → 
  (∃ n : ℕ, n = 24) :=
  sorry

end number_of_valid_numbers_l1558_155809


namespace delphine_chocolates_l1558_155888

theorem delphine_chocolates (x : ℕ) 
  (h1 : ∃ n, n = (2 * x - 3)) 
  (h2 : ∃ m, m = (x - 2))
  (h3 : ∃ p, p = (x - 3))
  (total_eq : x + (2 * x - 3) + (x - 2) + (x - 3) + 12 = 24) : 
  x = 4 := 
sorry

end delphine_chocolates_l1558_155888


namespace largest_n_with_integer_solutions_l1558_155883

theorem largest_n_with_integer_solutions : ∃ n, ∀ x y1 y2 y3 y4, 
 ( ((x + 1)^2 + y1^2) = ((x + 2)^2 + y2^2) ∧  ((x + 2)^2 + y2^2) = ((x + 3)^2 + y3^2) ∧ 
  ((x + 3)^2 + y3^2) = ((x + 4)^2 + y4^2)) → (n = 3) := sorry

end largest_n_with_integer_solutions_l1558_155883


namespace crayons_erasers_difference_l1558_155838

theorem crayons_erasers_difference
  (initial_erasers : ℕ) (initial_crayons : ℕ) (final_crayons : ℕ)
  (no_eraser_lost : initial_erasers = 457)
  (initial_crayons_condition : initial_crayons = 617)
  (final_crayons_condition : final_crayons = 523) :
  final_crayons - initial_erasers = 66 :=
by
  -- These would be assumptions in the proof; be aware that 'sorry' is used to skip the proof details.
  sorry

end crayons_erasers_difference_l1558_155838


namespace buffy_whiskers_l1558_155821

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end buffy_whiskers_l1558_155821


namespace markup_percentage_l1558_155865

theorem markup_percentage (PP SP SaleP : ℝ) (M : ℝ) (hPP : PP = 60) (h1 : SP = 60 + M * SP)
  (h2 : SaleP = SP * 0.8) (h3 : 4 = SaleP - PP) : M = 0.25 :=
by 
  sorry

end markup_percentage_l1558_155865


namespace add_to_fraction_l1558_155819

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l1558_155819


namespace evaluate_exponential_operations_l1558_155812

theorem evaluate_exponential_operations (a : ℝ) :
  (2 * a^2 - a^2 ≠ 2) ∧
  (a^2 * a^4 = a^6) ∧
  ((a^2)^3 ≠ a^5) ∧
  (a^6 / a^2 ≠ a^3) := by
  sorry

end evaluate_exponential_operations_l1558_155812


namespace inverse_of_3_mod_199_l1558_155830

theorem inverse_of_3_mod_199 : (3 * 133) % 199 = 1 :=
by
  sorry

end inverse_of_3_mod_199_l1558_155830


namespace unique_common_tangent_l1558_155813

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (a x : ℝ) : ℝ := a * Real.exp (x + 1)

theorem unique_common_tangent (a : ℝ) (h : a > 0) : 
  (∃ k x₁ x₂, k = 2 * x₁ ∧ k = a * Real.exp (x₂ + 1) ∧ k = (g a x₂ - f x₁) / (x₂ - x₁)) →
  a = 4 / Real.exp 3 :=
by
  sorry

end unique_common_tangent_l1558_155813


namespace ellipse_parameters_l1558_155820

theorem ellipse_parameters 
  (x y : ℝ)
  (h : 2 * x^2 + y^2 + 42 = 8 * x + 36 * y) :
  ∃ (h k : ℝ) (a b : ℝ), 
    (h = 2) ∧ (k = 18) ∧ (a = Real.sqrt 290) ∧ (b = Real.sqrt 145) ∧ 
    ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1 :=
sorry

end ellipse_parameters_l1558_155820


namespace particle_speed_correct_l1558_155879

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 9)

noncomputable def particle_speed : ℝ :=
  Real.sqrt (3 ^ 2 + 5 ^ 2)

theorem particle_speed_correct : particle_speed = Real.sqrt 34 := by
  sorry

end particle_speed_correct_l1558_155879


namespace pounds_added_l1558_155854

-- Definitions based on conditions
def initial_weight : ℝ := 5
def weight_increase_percent : ℝ := 1.5  -- 150% increase
def final_weight : ℝ := 28

-- Statement to prove
theorem pounds_added (w_initial w_final w_percent_added : ℝ) (h_initial: w_initial = 5) (h_final: w_final = 28)
(h_percent: w_percent_added = 1.5) :
  w_final - w_initial = 23 := 
by
  sorry

end pounds_added_l1558_155854


namespace attendance_calculation_l1558_155816

theorem attendance_calculation (total_students : ℕ) (attendance_rate : ℚ)
  (h1 : total_students = 120)
  (h2 : attendance_rate = 0.95) :
  total_students * attendance_rate = 114 := 
  sorry

end attendance_calculation_l1558_155816


namespace min_nS_n_l1558_155894

open Function

noncomputable def a (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

noncomputable def S (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := n * a_1 + d * n * (n - 1) / 2

theorem min_nS_n (d : ℤ) (h_a7 : ∃ a_1 : ℤ, a 7 a_1 d = 5)
  (h_S5 : ∃ a_1 : ℤ, S 5 a_1 d = -55) :
  ∃ n : ℕ, n > 0 ∧ n * S n a_1 d = -343 :=
by
  sorry

end min_nS_n_l1558_155894


namespace family_can_purchase_furniture_in_april_l1558_155857

noncomputable def monthly_income : ℤ := 150000
noncomputable def monthly_expenses : ℤ := 115000
noncomputable def initial_savings : ℤ := 45000
noncomputable def furniture_cost : ℤ := 127000

theorem family_can_purchase_furniture_in_april : 
  ∃ (months : ℕ), months = 3 ∧ 
  (initial_savings + months * (monthly_income - monthly_expenses) >= furniture_cost) :=
by
  -- proof will be written here
  sorry

end family_can_purchase_furniture_in_april_l1558_155857


namespace greatest_sum_l1558_155898

theorem greatest_sum {x y : ℤ} (h₁ : x^2 + y^2 = 49) : x + y ≤ 9 :=
sorry

end greatest_sum_l1558_155898


namespace tangent_line_eq_extreme_values_interval_l1558_155827

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x + 2

theorem tangent_line_eq (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  9 * 1 + (f 1 a b) = 0 :=
sorry

theorem extreme_values_interval (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  ∃ (min_val max_val : ℝ), 
    min_val = -14 ∧ f 2 a b = min_val ∧
    max_val = 18 ∧ f (-2) a b = max_val ∧
    ∀ x, (x ∈ Set.Icc (-3 : ℝ) 3 → f x a b ≥ min_val ∧ f x a b ≤ max_val) :=
sorry

end tangent_line_eq_extreme_values_interval_l1558_155827


namespace minimum_voters_needed_l1558_155801

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end minimum_voters_needed_l1558_155801


namespace rectangle_perimeter_l1558_155855

variable (x : ℝ) (y : ℝ)

-- Definitions based on conditions
def area_of_rectangle : Prop := x * (x + 5) = 500
def side_length_relation : Prop := y = x + 5

-- The theorem we want to prove
theorem rectangle_perimeter (h_area : area_of_rectangle x) (h_side_length : side_length_relation x y) : 2 * (x + y) = 90 := by
  sorry

end rectangle_perimeter_l1558_155855


namespace richmond_tickets_l1558_155837

theorem richmond_tickets (total_tickets : ℕ) (second_half_tickets : ℕ) (first_half_tickets : ℕ) :
  total_tickets = 9570 →
  second_half_tickets = 5703 →
  first_half_tickets = total_tickets - second_half_tickets →
  first_half_tickets = 3867 := by
  sorry

end richmond_tickets_l1558_155837


namespace no_such_integers_exist_l1558_155815

theorem no_such_integers_exist (x y z : ℤ) (hx : x ≠ 0) :
  ¬ (2 * x ^ 4 + 2 * x ^ 2 * y ^ 2 + y ^ 4 = z ^ 2) :=
by
  sorry

end no_such_integers_exist_l1558_155815


namespace g_at_3_l1558_155829

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_at_3 : g 3 = 79 := 
by 
  -- proof placeholder
  sorry

end g_at_3_l1558_155829


namespace minimum_value_expression_l1558_155891

variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 4)

theorem minimum_value_expression : (a + 3 * b) * (2 * b + 3 * c) * (a * c + 2) = 192 := by
  sorry

end minimum_value_expression_l1558_155891


namespace handshakes_count_l1558_155800

def women := 6
def teams := 3
def shakes_per_woman := 4
def total_handshakes := (6 * 4) / 2

theorem handshakes_count : total_handshakes = 12 := by
  -- We provide this theorem directly.
  rfl

end handshakes_count_l1558_155800


namespace proportion_not_necessarily_correct_l1558_155811

theorem proportion_not_necessarily_correct
  (a b c d : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : c ≠ 0)
  (h₄ : d ≠ 0)
  (h₅ : a * d = b * c) :
  ¬ ((a + 1) / b = (c + 1) / d) :=
by 
  sorry

end proportion_not_necessarily_correct_l1558_155811


namespace range_of_m_l1558_155892

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = 3) (h3 : x + y > 0) : m > -4 := by
  sorry

end range_of_m_l1558_155892


namespace sum_first_five_terms_eq_15_l1558_155871

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d 

variable (a : ℕ → ℝ) (h_arith_seq : is_arithmetic_sequence a) (h_a3 : a 3 = 3)

theorem sum_first_five_terms_eq_15 : (a 1 + a 2 + a 3 + a 4 + a 5 = 15) :=
sorry

end sum_first_five_terms_eq_15_l1558_155871


namespace amplitude_five_phase_shift_minus_pi_over_4_l1558_155808

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos (x + (Real.pi / 4))

theorem amplitude_five : ∀ x : ℝ, 5 * Real.cos (x + (Real.pi / 4)) = f x :=
by
  sorry

theorem phase_shift_minus_pi_over_4 : ∀ x : ℝ, f x = 5 * Real.cos (x + (Real.pi / 4)) :=
by
  sorry

end amplitude_five_phase_shift_minus_pi_over_4_l1558_155808


namespace complex_number_in_second_quadrant_l1558_155859

theorem complex_number_in_second_quadrant 
  (a b : ℝ) 
  (h : ¬ (a ≥ 0 ∨ b ≤ 0)) : 
  (a < 0 ∧ b > 0) :=
sorry

end complex_number_in_second_quadrant_l1558_155859


namespace determine_p_and_q_l1558_155824

noncomputable def find_p_and_q (a : ℝ) (p q : ℝ) : Prop :=
  (∀ x : ℝ, x = 1 ∨ x = -1 → (x^4 + p * x^2 + q * x + a^2 = 0))

theorem determine_p_and_q (a p q : ℝ) (h : find_p_and_q a p q) : p = -(a^2 + 1) ∧ q = 0 :=
by
  -- The proof would go here.
  sorry

end determine_p_and_q_l1558_155824


namespace solve_quadratic_l1558_155842

theorem solve_quadratic (y : ℝ) :
  3 * y * (y - 1) = 2 * (y - 1) → y = 2 / 3 ∨ y = 1 :=
by
  sorry

end solve_quadratic_l1558_155842


namespace triangle_angle_measure_l1558_155890

/-- Proving the measure of angle x in a defined triangle -/
theorem triangle_angle_measure (A B C x : ℝ) (hA : A = 85) (hB : B = 35) (hC : C = 30) : x = 150 :=
by
  sorry

end triangle_angle_measure_l1558_155890
