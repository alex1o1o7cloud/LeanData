import Mathlib

namespace dogs_wearing_tags_l663_663853

theorem dogs_wearing_tags (total_dogs flea_collared both_tags_and_collared neither :
                           ℕ) 
                           (h_total_eq : total_dogs = 80)
                           (h_flea_collared_eq : flea_collared = 40)
                           (h_both_tags_and_collared_eq : both_tags_and_collared = 6)
                           (h_neither_eq : neither = 1) :
                           ∃ (T : ℕ), T = 45 := 
by {
  let dogs_with_at_least_one := total_dogs - neither,
  have h_at_least_one_eq : dogs_with_at_least_one = 79 := by 
  { rw [h_total_eq, h_neither_eq], simp },
  let T := 79 - flea_collared + both_tags_and_collared,
  use T,
  rw [← h_at_least_one_eq, h_flea_collared_eq, h_both_tags_and_collared_eq],
  simp,
  norm_num,
}

end dogs_wearing_tags_l663_663853


namespace kayak_rental_cost_l663_663023

-- Definitions of given conditions
def cost_per_canoe : ℕ := 9
def total_revenue : ℕ := 432
def ratio_canoes_to_kayaks : ℕ × ℕ := (4, 3)
def extra_canoes : ℕ := 6

-- Define the cost of a kayak rental per day as K
def cost_of_kayak_rental (K : ℕ) : Prop := 
  ∃ (x : ℕ), 
    (x > 0) ∧ 
    (let num_kayaks := x in
     let num_canoes := x + extra_canoes in
     let canoes_revenue := num_canoes * cost_per_canoe in
     let kayaks_revenue := num_kayaks * K in
     canoes_revenue + kayaks_revenue = total_revenue) ∧
    (4 * num_kayaks = 3 * num_canoes)

-- The theorem we need to prove
theorem kayak_rental_cost : cost_of_kayak_rental 12 := by
  sorry

end kayak_rental_cost_l663_663023


namespace shaded_area_l663_663047

theorem shaded_area (r_small r_large : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) (h1 : r_small = 2)
                    (h2 : r_large = 3) (h3 : dist A B = 2 * r_small) 
                    (h4 : dist (C.1, C.2) A = r_small)
                    (h5 : dist (C.1, C.2) B = r_small)
                    (h6 : C.1 = (A.1 + B.1) / 2)
                    (h7 : C.2 = (A.2 + B.2) / 2) :
  let area := (0.82 * Real.pi - 5.4) in area = 0.82 * Real.pi - 5.4 :=
by
  sorry

end shaded_area_l663_663047


namespace angle_between_AD_BC_l663_663578

noncomputable def AB_perpendicular_BC : Prop := ∀ (A B C : ℝ^3), AB ⊥ BC
noncomputable def BC_perpendicular_CD : Prop := ∀ (B C D : ℝ^3), BC ⊥ CD
noncomputable def AB_length : ℝ := 2 * Real.sqrt 3
noncomputable def BC_length : ℝ := 2 * Real.sqrt 3
noncomputable def CD_length : ℝ := 2 * Real.sqrt 3
noncomputable def volume_tetrahedron : ℝ := 6

theorem angle_between_AD_BC :
  ∀ (A B C D : ℝ^3),
    AB_perpendicular_BC → 
    BC_perpendicular_CD → 
    (dist A B = AB_length) → 
    (dist B C = BC_length) → 
    (dist C D = CD_length) → 
    (volume (tetrahedron A B C D) = volume_tetrahedron) →
    (∃ θ, angle (line_through A D) (line_through B C) = 60 ∨ angle (line_through A D) (line_through B C) = 45) :=
by
  sorry

end angle_between_AD_BC_l663_663578


namespace num_noncongruent_triangles_l663_663310

-- Define the points and their properties
structure TrianglePoints (α : Type) :=
(A B C P Q R : α)
(isosceles_right_triangle : ∃ (s : ℝ), s > 0 ∧ dist A B = s ∧ dist A C = s ∧ dist B C = s * √2)
(midpoint_AB : P = (A + B) / 2)
(midpoint_BC : Q = (B + C) / 2)
(midpoint_CA : R = (C + A) / 2)

-- Define the problem statement
theorem num_noncongruent_triangles {α : Type} [metric_space α] [normed_group α] [normed_space ℝ α]
  (points : TrianglePoints α) : 
  ∃ n : ℕ, n = 3 := 
sorry

end num_noncongruent_triangles_l663_663310


namespace shaded_l_shaped_area_l663_663875

def square (side : ℕ) : ℕ := side * side
def rectangle (length width : ℕ) : ℕ := length * width

theorem shaded_l_shaped_area :
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  areaABCD - total_area_small_shapes = 20 :=
by
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  have h : areaABCD - total_area_small_shapes = 20 := sorry
  exact h

end shaded_l_shaped_area_l663_663875


namespace D_72_eq_93_l663_663691

def D (n : ℕ) : ℕ :=
-- The function definition of D would go here, but we leave it abstract for now.
sorry

theorem D_72_eq_93 : D 72 = 93 :=
sorry

end D_72_eq_93_l663_663691


namespace range_of_a_l663_663258

noncomputable
def parabola_intersects_line_segment (a : ℝ) : Prop := 
  ∃ x1 x2 : ℝ,
    (x1 ≠ x2) ∧
    (x1 ≥ -1 ∧ x1 ≤ 1) ∧
    (x2 ≥ -1 ∧ x2 ≤ 1) ∧
    (ax1^2 - x1 + 1 = (1 / 2) * x1 + 1 / 2) ∧
    (ax2^2 - x2 + 1 = (1 / 2) * x2 + 1 / 2)

theorem range_of_a (a : ℝ) (h : a ≠ 0) : 
  parabola_intersects_line_segment a ↔ (1 ≤ a ∧ a < 9 / 8) :=
sorry

end range_of_a_l663_663258


namespace triangle_area_l663_663131

def areaTriangle : ℝ := 15 / 2

theorem triangle_area (A B C : ℝ × ℝ) (hA : A = (5, -2)) (hB : B = (0, 3)) (hC : C = (3, -3)) :
  let v := ((5 - 3), (-2 - (-3)))
  let w := ((0 - 3), (3 - (-3)))
  abs (v.1 * w.2 - v.2 * w.1) / 2 = areaTriangle :=
by
  let v := ((5 - 3), (-2 - (-3)))
  let w := ((0 - 3), (3 - (-3)))
  have hv : v = (2, 1) by sorry
  have hw : w = (-3, 6) by sorry
  have det : abs (v.1 * w.2 - v.2 * w.1) = 15 by sorry
  show abs (v.1 * w.2 - v.2 * w.1) / 2 = 15 / 2 from sorry

#check @triangle_area

end triangle_area_l663_663131


namespace non_existent_triangles_l663_663789

-- Definitions for the types of triangles
def equilateral_triangle (T : Type) [triangle T] : Prop :=
  ∃ a b c : T, (a = b ∧ b = c ∧ c = a ∧ a ≠ b)

def right_triangle (T : Type) [triangle T] : Prop :=
  ∃ A B C : angle T, A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

def scalene_triangle (T : Type) [triangle T] : Prop :=
  ∃ a b c : T, (a ≠ b ∧ b ≠ c ∧ c ≠ a)

theorem non_existent_triangles (T : Type) [triangle T] :
  ¬ (scalene_triangle T ∧ equilateral_triangle T) ∧ ¬ (equilateral_triangle T ∧ right_triangle T) :=
by
  sorry

end non_existent_triangles_l663_663789


namespace enclosed_triangle_area_le_one_quarter_l663_663118

noncomputable def shape_area (S : ℝ) : Prop :=
  ∃ (S1 S2 S3 S4 S5 S6 S7 : ℝ), 
  (S1 + S2 + S3 + S4 + S5 + S6 + S7 = S) ∧
  (S1 + S2 + S3 + S4 = S / 2) ∧
  (S1 + S6 + S2 + S7 = S / 2) ∧
  (S3 + S2 + S7 = S / 2) ∧
  (S5 = S / 2) ∧
  (S6 = S / 2) ∧
  (S7 = S / 2)

theorem enclosed_triangle_area_le_one_quarter (S : ℝ) (h : shape_area S) : 
  ∃ S1 : ℝ, S1 ≤ S / 4 :=
begin
  sorry
end

end enclosed_triangle_area_le_one_quarter_l663_663118


namespace afec_projection_sine_curve_l663_663161

/-- Given a thin, rectangular plate ABCD where BC = 2AB.
The midpoints of sides AD and BC are E and F, respectively.
The segments AF and EC are made visible on the plate.
When the plate is bent into the shape of a cylindrical surface such that AD and BC coincide,
E and F merge, forming a helix with two turns.
Prove that the projection of the line AFEC onto a plane parallel to the cylinder's axis
is a sine curve. Characterize the planes onto which the projection of AFEC
has a peak that is not an endpoint image occurring at 45 degrees. -/
theorem afec_projection_sine_curve (ABCD : Type*) [rectangular_plate ABCD]
  (BC_eq_2AB : BC = 2 * AB)
  (E : midpoint (side AD)) (F : midpoint (side BC))
  (bent_to_cylinder : bent_into_cylinder ABCD AD BC)
  (helix_with_2_turns : helix_turns E F 2) :
  ∀ (plane_parallel_to_axis : Type*), plane_parallel plane_parallel_to_axis cylinder_axis →
  projection_of_AFEC_parallel_to_axis_is_sine_curve AFEC plane_parallel_to_axis ∧
  characterization_of_AFEC_projection_peaks plane_parallel_to_axis = 45 := sorry

end afec_projection_sine_curve_l663_663161


namespace root_in_interval_l663_663028

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem root_in_interval :
  (f 2 < 0) →
  (f 3 > 0) →
  ∃ c ∈ Ioo 2 3, f c = 0 :=
by
  sorry

end root_in_interval_l663_663028


namespace inequality_selection_l663_663269

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  1/a + 4/b ≥ 9/(a + b) :=
sorry

end inequality_selection_l663_663269


namespace equation_elliptic_and_canonical_form_l663_663515

-- Defining the necessary conditions and setup
def a11 := 1
def a12 := 1
def a22 := 2

def is_elliptic (a11 a12 a22 : ℝ) : Prop :=
  a12^2 - a11 * a22 < 0

def canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) : Prop :=
  let ξ := y - x
  let η := x
  let u_ξξ := u_xx -- Assuming u_xx represents u_ξξ after change of vars
  let u_ξη := u_xy
  let u_ηη := u_yy
  let u_ξ := u_x -- Assuming u_x represents u_ξ after change of vars
  let u_η := u_y
  u_ξξ + u_ηη = -2 * u_η + u + η + (ξ + η)^2

theorem equation_elliptic_and_canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) :
  is_elliptic a11 a12 a22 ∧
  canonical_form u_xx u_xy u_yy u_x u_y u x y :=
by
  sorry -- Proof to be completed

end equation_elliptic_and_canonical_form_l663_663515


namespace scientific_notation_of_billion_l663_663072

-- Define the initial values and conditions
def billion := 10^9
def value := 1.14 * billion

-- The proposition we need to prove
theorem scientific_notation_of_billion : value = 1.14 * 10^9 := 
sorry

end scientific_notation_of_billion_l663_663072


namespace sum_of_coefficients_l663_663285

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) 
  (h : (1 + 2*x)^7 = a + a₁*(1 - x) + a₂*(1 - x)^2 + a₃*(1 - x)^3 + a₄*(1 - x)^4 + a₅*(1 - x)^5 + a₆*(1 - x)^6 + a₇*(1 - x)^7) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 1 :=
by 
  sorry

end sum_of_coefficients_l663_663285


namespace additional_treetags_l663_663659

noncomputable def initial_numerals : Finset ℕ := {1, 2, 3, 4}
noncomputable def initial_letters : Finset Char := {'A', 'E', 'I'}
noncomputable def initial_symbols : Finset Char := {'!', '@', '#', '$'}
noncomputable def added_numeral : Finset ℕ := {5}
noncomputable def added_symbols : Finset Char := {'&'}

theorem additional_treetags : 
  let initial_treetags := initial_numerals.card * initial_letters.card * initial_symbols.card
  let new_numerals := initial_numerals ∪ added_numeral
  let new_symbols := initial_symbols ∪ added_symbols
  let new_treetags := new_numerals.card * initial_letters.card * new_symbols.card
  new_treetags - initial_treetags = 27 := 
by 
  sorry

end additional_treetags_l663_663659


namespace angle_B_in_triangle_l663_663677

noncomputable def find_angle_B (a b : ℝ) (A : ℝ) : ℝ :=
  let sin_A := real.sin (real.pi * A / 180) in
  let sin_B := b * sin_A / a in
  real.arcsin (sin_B) * 180 / real.pi

theorem angle_B_in_triangle
  (a b : ℝ)
  (A : ℝ)
  (ha : a = 2)
  (hb : b = real.sqrt 2)
  (hA : A = 45) :
  find_angle_B a b A = 30 :=
by
  rw [ha, hb, hA]
  sorry

end angle_B_in_triangle_l663_663677


namespace max_value_of_x_plus_inv_x_l663_663363

theorem max_value_of_x_plus_inv_x (x : ℝ) (y : ℕ → ℝ) (n : ℕ) (sum_y : ℝ) (sum_inv_y : ℝ) (sum_cub_y : ℝ) :
  n = 1000 →
  (∀ i, y i > 0) →
  sum (finset.range n) (λ i, y i) = sum_y →
  sum_y + x = 1002 →
  sum (finset.range n) (λ i, (y i)⁻¹) = sum_inv_y →
  sum_inv_y + x⁻¹ = 1002 →
  sum (finset.range n) (λ i, (y i)^3) = sum_cub_y →
  sum_cub_y + x^3 = 1002 →
  x + x⁻¹ ≤ 4 :=
by sorry

end max_value_of_x_plus_inv_x_l663_663363


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663234

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663234


namespace cost_of_materials_for_cars_l663_663814

theorem cost_of_materials_for_cars :
  ∃ (C : ℕ), (C = 100) ∧
    (let revenue_cars := 4 * 50 in
     let profit_cars := revenue_cars - C in
     let revenue_motorcycles := 8 * 50 in
     let profit_motorcycles := revenue_motorcycles - 250 in
     profit_motorcycles = profit_cars + 50) :=
begin
  use 100,
  split,
  { refl },
  { 
    let revenue_cars := 4 * 50,
    let profit_cars := revenue_cars - 100,
    let revenue_motorcycles := 8 * 50,
    let profit_motorcycles := revenue_motorcycles - 250,
    have h1 : profit_motorcycles = 150 := by norm_num,
    have h2 : profit_cars = 100 := by norm_num,
    rw h2,
    simp [h1]
  }
end

end cost_of_materials_for_cars_l663_663814


namespace james_missing_legos_l663_663683

theorem james_missing_legos  (h1 : 500 > 0) (h2 : 500 % 2 = 0) (h3 : 245 < 500)  :
  let total_legos := 500
  let used_legos := total_legos / 2
  let leftover_legos := total_legos - used_legos
  let legos_in_box := 245
  leftover_legos - legos_in_box = 5 := by
{
  sorry
}

end james_missing_legos_l663_663683


namespace odd_function_condition_l663_663347

noncomputable def f (x a b : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) ↔ (a = 0 ∧ b = 0) := 
by
  sorry

end odd_function_condition_l663_663347


namespace variance_transformation_l663_663637

variable (X : ℕ → ℕ → Prop)

def binomial_distribution (n : ℕ) (p : ℝ) : Prop :=
  X n p = B(n, p)  -- Assuming B stands for the binomial distribution

theorem variance_transformation :
  binomial_distribution 10 0.8 →
  D(X) = 10 * 0.8 * 0.2 →
  D(2 * X + 1) = 6.4 := by
  sorry

end variance_transformation_l663_663637


namespace post_height_l663_663838

-- Conditions
def spiral_path (circuit_per_rise rise_distance : ℝ) := ∀ (total_distance circ_circumference height : ℝ), 
  circuit_per_rise = total_distance / circ_circumference ∧ 
  height = circuit_per_rise * rise_distance

-- Given conditions
def cylinder_post : Prop := 
  ∀ (total_distance circ_circumference rise_distance : ℝ), 
    spiral_path (total_distance / circ_circumference) rise_distance ∧ 
    circ_circumference = 3 ∧ 
    rise_distance = 4 ∧ 
    total_distance = 12

-- Proof problem: Post height
theorem post_height : cylinder_post → ∃ height : ℝ, height = 16 := 
by sorry

end post_height_l663_663838


namespace division_remainder_l663_663714

theorem division_remainder (dividend divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 15) 
  (h_quotient : quotient = 9) 
  (h_dividend_eq : dividend = 136) 
  (h_eq : dividend = (divisor * quotient) + remainder) : 
  remainder = 1 :=
by
  sorry

end division_remainder_l663_663714


namespace fourth_person_height_l663_663405

theorem fourth_person_height (H : ℝ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 77) : 
  H + 10 = 83 :=
sorry

end fourth_person_height_l663_663405


namespace number_of_attendants_writing_with_both_l663_663096

-- Definitions for each of the conditions
def attendants_using_pencil : ℕ := 25
def attendants_using_pen : ℕ := 15
def attendants_using_only_one : ℕ := 20

-- Theorem that states the mathematically equivalent proof problem
theorem number_of_attendants_writing_with_both 
  (p : ℕ := attendants_using_pencil)
  (e : ℕ := attendants_using_pen)
  (o : ℕ := attendants_using_only_one) : 
  ∃ x, (p - x) + (e - x) = o ∧ x = 10 :=
by
  sorry

end number_of_attendants_writing_with_both_l663_663096


namespace arrange_in_order_l663_663799

noncomputable def x1 : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def x2 : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def x3 : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))
noncomputable def x4 : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))

theorem arrange_in_order : 
  x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 := 
by 
  sorry

end arrange_in_order_l663_663799


namespace payment_difference_l663_663300

noncomputable def plan1_total_payment (P : ℝ) (r : ℝ) (t1 t2 : ℕ) : ℝ :=
  let A1 := P * (1 + r / 2)^(2 * t1)
  let payment1 := A1 / 3
  let remaining := A1 - payment1
  let A2 := remaining * (1 + r / 2)^(2 * t2)
  payment1 + A2

noncomputable def plan2_total_payment (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t

theorem payment_difference
  (P : ℝ := 12000) (r : ℝ := 0.08) (t1 : ℕ := 5) (t2 : ℕ := 10) (t : ℕ := 15) :
  abs (plan2_total_payment P r t - plan1_total_payment P r t1 t2) ≈ 150 :=
sorry

end payment_difference_l663_663300


namespace lowest_price_correct_l663_663396

noncomputable def lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components : ℕ) : ℕ :=
(cost_per_component + shipping_cost_per_unit) * number_of_components + fixed_costs

theorem lowest_price_correct :
  lowest_price 80 5 16500 150 / 150 = 195 :=
by
  sorry

end lowest_price_correct_l663_663396


namespace product_of_roots_of_Q_is_neg18_l663_663696

-- Let u be the cube root of 2.
def u : ℚ := real.root (2 : ℝ) (3 : ℕ)

-- Define the polynomial Q(x) with the root being (u + 2)
def Q (x : ℝ) : ℝ := x^3 + 6*x^2 + 12*x + 18

-- Statement to prove
theorem product_of_roots_of_Q_is_neg18 : ∀ roots : List ℝ, 
  (∀ r ∈ roots, Q r = 0) → (roots.product = -18) :=
sorry

end product_of_roots_of_Q_is_neg18_l663_663696


namespace trapezium_area_l663_663881

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 14) (hh : h = 18) : 
  (1 / 2) * (a + b) * h = 342 := 
by
  -- Conditions
  rw [ha, hb, hh]
  -- Area calculation
  calc
    (1 / 2) * (a + b) * h = (1 / 2) * (24 + 14) * 18 : by rw [ha, hb, hh]
    ... = (1 / 2) * 38 * 18 : by norm_num
    ... = 19 * 18 : by norm_num
    ... = 342 : by norm_num

end trapezium_area_l663_663881


namespace parabola_focus_at_centroid_l663_663165

theorem parabola_focus_at_centroid (A B C : ℝ × ℝ) (a : ℝ) 
  (hA : A = (-1, 2))
  (hB : B = (3, 4))
  (hC : C = (4, -6))
  (h_focus : (a/4, 0) = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  a = 8 :=
by
  sorry

end parabola_focus_at_centroid_l663_663165


namespace area_diff_circle_square_l663_663740

theorem area_diff_circle_square (d_square d_circle : ℝ) (h1 : d_square = 10) (h2 : d_circle = 10) :
  let s := d_square / Real.sqrt 2,
      area_square := s^2,
      r := d_circle / 2,
      area_circle := Real.pi * r^2,
      area_diff := area_circle - area_square in
  Real.floor (area_diff * 10) / 10 = 28.5 :=
by
  sorry

end area_diff_circle_square_l663_663740


namespace minimum_k_good_line_set_l663_663918

-- Definitions of a Colombian point set and a good line set
def ColombianPointSet (S : Set (ℝ × ℝ)) : Prop :=
  (∀ P1 P2 P3 ∈ S, no_triangle P1 P2 P3) ∧ (∃ r b, r = 2013 ∧ b = 2014 ∧ 
    (∀ x ∈ S, color x ∈ {red, blue}))

def good_line_set (S : Set (ℝ × ℝ)) (k : ℕ) (lines : Set (set ℝ)) : Prop :=
  (∀ l ∈ lines, ∀ p ∈ S, ¬(p ∈ l)) ∧
  (∀ region, region ∈ (divide_plane lines S) → 
    ∀ a b ∈ region, color a = color b)

-- Main statement
theorem minimum_k_good_line_set : ∀ (S : Set (ℝ × ℝ)),
  ColombianPointSet S → (∃ k lines, k ≤ 2013 ∧ good_line_set S k lines) :=
sorry

end minimum_k_good_line_set_l663_663918


namespace regular_octagon_vector_sum_l663_663662

-- Definitions of vectors in a regular octagon
variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Given definitions based on the problem statement
def a (i : ℕ) : α := sorry  -- Vector A_i A_{i+1}
def b (i : ℕ) : α := sorry  -- Vector OA_i

-- The proof goal
theorem regular_octagon_vector_sum : b 2 + b 5 + b 7 + a 2 + a 5 = b 6 :=
sorry

end regular_octagon_vector_sum_l663_663662


namespace asbestos_tile_covering_l663_663366

theorem asbestos_tile_covering (n : ℕ) (h : n > 0) : 
  let single_tile_width := 60 
  let overlap_width := 10 
  let covered_width := (single_tile_width - overlap_width) * n + overlap_width 
  in covered_width = 50 * n + 10 := 
by 
  sorry

end asbestos_tile_covering_l663_663366


namespace job_completion_time_l663_663369

noncomputable def time_to_complete_job : ℚ :=
  let rate_R := 1 / 5
  let rate_B := 1 / 7
  let rate_C := 1 / 9
  let total_rate := 4 * rate_R + 3 * rate_B + 2 * rate_C
  1 / total_rate

theorem job_completion_time :
  time_to_complete_job ≈ 0.6897 := by
  sorry

end job_completion_time_l663_663369


namespace minimum_area_ellipse_contains_circles_l663_663450

-- Define the ellipse equation and the two circle equations
def ellipse (a b x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def circle1 (x y : ℝ) := (x - 1)^2 + y^2 = 1
def circle2 (x y : ℝ) := (x + 1)^2 + y^2 = 1

-- The goal is to prove the minimum possible area of the ellipse containing the two circles
theorem minimum_area_ellipse_contains_circles :
  ∃ a b : ℝ, (∀ x y : ℝ, circle1 x y → ellipse a b x y) ∧
              (∀ x y : ℝ, circle2 x y → ellipse a b x y) ∧
              a * b * π = (3 * real.sqrt 3) / 2 * π :=
by sorry

end minimum_area_ellipse_contains_circles_l663_663450


namespace product_of_20_random_digits_ends_in_0_probability_l663_663569

theorem product_of_20_random_digits_ends_in_0_probability :
  let prob_at_least_one_0 := 1 - (9 / 10) ^ 20,
      prob_even_digit := 1 - (5 / 9) ^ 20,
      prob_5 := 1 - (8 / 9) ^ 19
  in 
    prob_at_least_one_0 + ( (9 / 10) ^ 20 * prob_even_digit * prob_5 ) ≈ 0.988 :=
by sorry

end product_of_20_random_digits_ends_in_0_probability_l663_663569


namespace find_a_monotonic_intervals_l663_663939

-- Definition of the function f
def f (x a : ℝ) : ℝ := (1/3) * x^3 - a * x^2 - 3 * x

-- Statement: when tangent at (1, f(1, a)) is parallel to y = -4x + 1, a = 1
theorem find_a (a : ℝ) (h : (deriv (λ x, f x a)) 1 = -4) : a = 1 := sorry

-- Function f with a = 1
def f_a1 (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3 * x

-- Derivative of f_a1
def f_a1' (x : ℝ) : ℝ := deriv f_a1 x

-- Statement: monotonic intervals for f_a1
theorem monotonic_intervals : 
  (∀ x, f_a1' x > 0 ↔ x < -1 ∨ x > 3) ∧ 
  (∀ x, f_a1' x < 0 ↔ -1 < x ∧ x < 3) := 
sorry

end find_a_monotonic_intervals_l663_663939


namespace divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l663_663349

theorem divisor_probability_of_25_factorial_is_odd_and_multiple_of_5 :
  let prime_factors_25 := 2^22 * 3^10 * 5^6 * 7^3 * 11^2 * 13^1 * 17^1 * 19^1 * 23^1
  let total_divisors := (22+1) * (10+1) * (6+1) * (3+1) * (2+1) * (1+1) * (1+1) * (1+1)
  let odd_and_multiple_of_5_divisors := (6+1) * (3+1) * (2+1) * (1+1) * (1+1)
  (odd_and_multiple_of_5_divisors / total_divisors : ℚ) = 7 / 23 := 
sorry

end divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l663_663349


namespace log_expression_calculation_l663_663500

theorem log_expression_calculation :
  2 * log 5 10 + log 5 0.25 = 2 := by
  -- assume properties of logarithms and definition of 0.25
  sorry

end log_expression_calculation_l663_663500


namespace equation_of_circle_l663_663811

theorem equation_of_circle :
  ∃ (a : ℝ), a < 0 ∧ (∀ (x y : ℝ), (x + 2 * y = 0) → (x + 5)^2 + y^2 = 5) :=
by
  sorry

end equation_of_circle_l663_663811


namespace greatest_prime_factor_of_sum_l663_663250

def product_of_evens_up_to (x : ℕ) : ℕ :=
  if x % 2 = 0 then -- x is even
    List.prod (List.filter (λ k => k % 2 = 0) (List.range' 2 (x + 1)))
  else -- x is odd
    List.prod (List.filter (λ k => k % 2 = 0) (List.range' 2 x))

def greatest_prime_factor (n : ℕ) : ℕ :=
  (List.filter Nat.Prime (List.range' 2 (n + 1))).filter (λ p => n % p = 0)).get_last

theorem greatest_prime_factor_of_sum {24} {26} : greatest_prime_factor (product_of_evens_up_to 26 + product_of_evens_up_to 24) = 23 := 
  sorry

end greatest_prime_factor_of_sum_l663_663250


namespace expression_one_expression_two_l663_663855

-- Define the expressions to be proved.
theorem expression_one : (3.6 - 0.8) * (1.8 + 2.05) = 10.78 :=
by sorry

theorem expression_two : (34.28 / 2) - (16.2 / 4) = 13.09 :=
by sorry

end expression_one_expression_two_l663_663855


namespace equivalence_of_equations_l663_663719

theorem equivalence_of_equations (a b c : ℝ) :
  (frac (a * (b - c)) (b + c) + frac (b * (c - a)) (c + a) + frac (c * (a - b)) (a + b) = 0) ↔ 
  (frac (a^2 * (b - c)) (b + c) + frac (b^2 * (c - a)) (c + a) + frac (c^2 * (a - b)) (a + b) = 0) :=
sorry

end equivalence_of_equations_l663_663719


namespace integer_solutions_set_l663_663125

theorem integer_solutions_set :
  {x : ℤ | 2 * x + 4 > 0 ∧ 1 + x ≥ 2 * x - 1} = {-1, 0, 1, 2} :=
by {
  sorry
}

end integer_solutions_set_l663_663125


namespace sum_of_consecutive_integers_of_sqrt3_l663_663208

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l663_663208


namespace moles_H2SO4_formed_l663_663889

-- Definitions based on conditions in problem
def SO2 : Type := ℕ
def H2O2 : Type := ℕ
def H2SO4 : Type := ℕ

-- 1 mole of SO2
def initial_SO2 : SO2 := 1

-- 1 mole of H2O2
def initial_H2O2 : H2O2 := 1

-- Reaction between SO2 and H2O2 to form H2SO4
def reaction (so2 : SO2) (h2o2 : H2O2) : H2SO4 :=
  so2 -- Given the reaction is in a 1:1:1 ratio.

-- Prove that given the initial moles of SO2 and H2O2, 1 mole of H2SO4 is formed
theorem moles_H2SO4_formed : reaction initial_SO2 initial_H2O2 = 1 :=
  by
  sorry

end moles_H2SO4_formed_l663_663889


namespace all_a_equal_l663_663644

theorem all_a_equal (n : ℕ) (a : ℕ → ℂ) (c : ℕ → ℂ) :
  (∀ (x : ℂ), (x ^ n - n • a 1 * x ^ (n - 1) + c n ^ 2 * (a 2) ^ 2 * x ^ (n - 2) +
  ∑ i in fin n, (-1) ^ i * c n ^ i * (a i) ^ i * x ^ (n - i) +
  (-1) ^ n * (a n) ^ n = 0)) →
  (∀ i j, a i = a j) :=
by
  sorry

end all_a_equal_l663_663644


namespace convert_to_rectangular_form_l663_663109

theorem convert_to_rectangular_form :
  2 * exp (13 * real.pi * complex.I / 6) = (sqrt 3 : ℂ) + (1 : ℂ) * complex.I :=
by sorry

end convert_to_rectangular_form_l663_663109


namespace ellipse_line_slope_condition_l663_663744

theorem ellipse_line_slope_condition 
  (m n x y : ℝ)
  (h1 : ∀ A B : ℝ × ℝ, ((∃ x1 y1, A = (x1, y1) ∧ mx1^2 + ny1^2 = 1 ∧ x1 + y1 = 1) ∧ 
                        (∃ x2 y2, B = (x2, y2) ∧ mx2^2 + ny2^2 = 1 ∧ x2 + y2 = 1)) → 
                       (let mid := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in 
                        (mid.2 / mid.1) = sqrt(2) / 2)) :
  m / n = sqrt(2) / 2 := by
  sorry

end ellipse_line_slope_condition_l663_663744


namespace circle_with_AB_as_diameter_l663_663584

structure Point where
  x : ℝ
  y : ℝ

def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

def distance (A B : Point) : ℝ :=
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

def circle_eqn (center : Point) (radius : ℝ) (P : Point) : Prop :=
  (P.x - center.x)^2 + (P.y - center.y)^2 = radius^2

theorem circle_with_AB_as_diameter :
  let A := { x := -3, y := -5 } in
  let B := { x := 5, y := 1 } in
  circle_eqn (midpoint A B) (distance A B / 2) { x := 1, y := -2 } :=
  by
    let A := { x := -3, y := -5 }
    let B := { x := 5, y := 1 }
    let M := midpoint A B
    let d := distance A B
    show circle_eqn M (d / 2) { x := 1, y := -2 }
    -- Placeholder for proof
    sorry

end circle_with_AB_as_diameter_l663_663584


namespace sum_G_l663_663897

noncomputable def G(n : ℕ) : ℕ :=
  if (n % 2) = 0 then n + 1 else n

theorem sum_G : (∑ n in Finset.range 1002 \ Finset.range 1, G (n + 2)) = 502753 := 
by
  sorry

end sum_G_l663_663897


namespace igor_number_is_5_l663_663444

def initial_players : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def command (players : List ℕ) := players.filter (λ n, (λ neighbors := neighbors.any (λ m, n ≥ m)) (neighbors players n))

theorem igor_number_is_5 :
  ∃ (igor_number : ℕ),
    (igor_number ∈ initial_players) ∧
    ∀ (remaining_players_after_igor : List ℕ),
      remaining_players_after_igor = (command (command (command (command initial_players)))) \{ igor_number } ∧
      remaining_players_after_igor.length = 3 →
      igor_number = 5 :=
  by
    sorry

end igor_number_is_5_l663_663444


namespace find_a7_l663_663341

noncomputable def a_n (n : ℕ) : ℂ := sorry -- Define the geometric sequence

axiom geom_seq (a_n : ℕ → ℂ) (q : ℂ) (a_1 : ℂ) :
  (∀ n, a_n = a_1 * q ^ n) ∧
  (a_1 + a_1 * q = 3) ∧
  (a_1 * q + a_1 * q ^ 2 = 6)

theorem find_a7 : (∃ a_n : ℕ → ℂ, ∃ q a_1 : ℂ, 
  (∀ n, a_n = a_1 * q ^ n) ∧
  (a_1 + a_1 * q = 3) ∧
  (a_1 * q + a_1 * q ^ 2 = 6)) → 
  a_n 6 = 64 := -- Prove that the 7th term a_7 = a_n 6 (0-indexed in Lean) is 64
by sorry

end find_a7_l663_663341


namespace problem1_problem2_l663_663185

-- Define the piecewise function f
def f : ℝ → ℝ := λ x, if x ≤ 0 then -x + 3 else 4 * x

-- Problem 1: Prove that f(f(-1)) = 16
theorem problem1 : f (f (-1)) = 16 := 
by
  sorry

-- Problem 2: Prove the range of x₀ if f(x₀) > 2
theorem problem2 : {x₀ : ℝ | f x₀ > 2} = {x | x ≤ 0} ∪ {x | x > 1 / 2} :=
by
  sorry

end problem1_problem2_l663_663185


namespace area_difference_correct_l663_663737

-- Definitions for given conditions
def square_diagonal : ℝ := 10
def circle_diameter : ℝ := 10
def pi_approx : ℝ := 3.14159

-- Symbolic calculations
def square_side (d : ℝ) : ℝ := (d^2 / 2).sqrt
def square_area (s : ℝ) : ℝ := s^2
def circle_radius (d : ℝ) : ℝ := d / 2
def circle_area (r : ℝ) (π : ℝ) : ℝ := π * r^2
def area_difference (circle_area : ℝ) (square_area : ℝ) : ℝ := circle_area - square_area

theorem area_difference_correct:
  area_difference (circle_area (circle_radius circle_diameter) pi_approx) (square_area (square_side square_diagonal)) = 28.5 :=
by sorry

end area_difference_correct_l663_663737


namespace consecutive_sum_l663_663216

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l663_663216


namespace product_ends_in_0_l663_663574

/-- Given a set of 20 random digits, the probability that the product of these digits ends in 
    0 is approximately 0.988 -/
theorem product_ends_in_0 (s : Finset ℕ) (h: s.card = 20) :
  (∑ k in Finset.range 10, if k = 0 then 1 else if ∃ n ∈ s, n = 2 ∧ ∃ m ∈ s, m = 5 then 1 else 0) / 10 ^ 20 ≈ 0.988 := 
sorry

end product_ends_in_0_l663_663574


namespace range_of_m_l663_663890

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (m+1)*x^2 - m*x + m - 1 ≥ 0) ↔ m ≥ (2*Real.sqrt 3)/3 := by
  sorry

end range_of_m_l663_663890


namespace find_a_b_monotonicity_l663_663561

-- Definition of the function f(x) and conditions
noncomputable def f (x : ℝ) (a b k : ℝ) : ℝ := a * x ^ 2 + b * x + k

-- Statement for the first question: finding a and b
theorem find_a_b (a b k : ℝ) (h : k > 0) (extremum : f 0 a b k = f 1 a b k)
  (perpendicular : 2 * a = -1 / 2) : a = -1 / 4 ∧ b = 0 := sorry

-- Definition of the function g(x)
noncomputable def g (x : ℝ) (a b k : ℝ) : ℝ := exp x / f x a b k

-- Statement for the second question: monotonicity of g(x)
theorem monotonicity (a b k : ℝ) (h : k > 0) (ha : a = -1 / 4) (hb : b = 0) :
  (∀ x, x < 8 → derivative (g x a b k) > 0) ∧
  (∀ x, x > 8 → derivative (g x a b k) < 0) := sorry

end find_a_b_monotonicity_l663_663561


namespace max_area_of_triangle_l663_663580

variables (a b c : ℝ) (m : ℝ)
variables (x y : ℝ) (P A B : ℝ × ℝ)
variables (h_eccentricity_hyperbola : ℝ := sqrt 2)
variables (h_eccentricity_ellipse : ℝ := 1 / h_eccentricity_hyperbola)

noncomputable def ellipse_equation : Prop :=
  y^2 / a^2 + x^2 / b^2 = 1

def is_point_on_ellipse (p : ℝ × ℝ) : Prop :=
  ellipse_equation p.1 p.2

def line_equation (x : ℝ) : ℝ :=
  sqrt 2 * x + m

def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.fst * (p2.snd - p3.snd) + 
               p2.fst * (p3.snd - p1.snd) + 
               p3.fst * (p1.snd - p2.snd))

def problem_conditions : Prop :=
  ellipse_equation ∧
  is_point_on_ellipse (1, sqrt 2) ∧
  2 * a = 4 ∧
  e = h_eccentricity_ellipse ∧
  a > b ∧
  b > 0 ∧
  e * a = sqrt 2

theorem max_area_of_triangle : 
  problem_conditions →
  ∃ (S : ℝ), S = sqrt 2 :=
sorry

end max_area_of_triangle_l663_663580


namespace sum_digits_after_time_addition_l663_663270

/-- 
   Let current_time be 3:25:15 PM and added_time be 137 hours, 59 minutes, 59 seconds. 
   Prove that the sum of the digits of the time shown on a 12-hour digital clock
   after adding added_time to current_time is 21.
-/
theorem sum_digits_after_time_addition : 
  let current_hour := 3 
  let current_min := 25 
  let current_sec := 15
  let added_hours := 137 
  let added_min := 59 
  let added_sec := 59 
  let new_sec := (current_sec + added_sec) % 60
  let extra_min := (current_sec + added_sec) / 60
  let new_min := (current_min + added_min + extra_min) % 60
  let extra_hour := (current_min + added_min + extra_min) / 60
  let new_hour := ((current_hour + added_hours + extra_hour) % 12) 
  let digits_sum := new_hour + (new_min / 10) + (new_min % 10) + (new_sec / 10) + (new_sec % 10)
  in digits_sum = 21 :=
by
  -- The steps of the proof would be carried out here
  sorry

end sum_digits_after_time_addition_l663_663270


namespace consecutive_sum_l663_663215

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l663_663215


namespace consecutive_integers_sum_l663_663239

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l663_663239


namespace parallelogram_area_20_l663_663777

-- Definition of the points of the parallelogram
def point1 := (0, 0) : ℝ×ℝ
def point2 := (4, 0) : ℝ×ℝ
def point3 := (3, 5) : ℝ×ℝ
def point4 := (7, 5) : ℝ×ℝ

-- Definition of a function to calculate the area of the parallelogram given its points
def parallelogram_area (A B C D : ℝ×ℝ) : ℝ := (B.1 - A.1) * (C.2 - A.2)

-- Theorem stating that the area of the given parallelogram is 20 square units
theorem parallelogram_area_20 : parallelogram_area point1 point2 point3 point4 = 20 :=
by
  -- Skipping the actual proof for this step
  sorry

end parallelogram_area_20_l663_663777


namespace probability_non_defective_second_draw_l663_663076

theorem probability_non_defective_second_draw 
  (total_products : ℕ)
  (defective_products : ℕ)
  (first_draw_defective : Bool)
  (second_draw_non_defective_probability : ℚ) : 
  total_products = 100 → 
  defective_products = 3 → 
  first_draw_defective = true → 
  second_draw_non_defective_probability = 97 / 99 :=
by
  intros h_total h_defective h_first_draw
  subst h_total
  subst h_defective
  subst h_first_draw
  sorry

end probability_non_defective_second_draw_l663_663076


namespace order_of_a_b_c_l663_663288

theorem order_of_a_b_c :
  let a := Real.log 2 / Real.log 3
  let b := 2^(1/3 : ℝ)
  let c := 3^(1/3 : ℝ)
  c > b ∧ b > a :=
by
  let a := Real.log 2 / Real.log 3
  let b := 2^(1/3 : ℝ)
  let c := 3^(1/3 : ℝ)
  have ha : 0 < a ∧ a < 1 := ⟨Real.log_pos_iff.mpr (by norm_num), (Real.log_lt_log_iff (by norm_num) (by norm_num)).mpr (by norm_num)⟩
  have hb : 1 < b ∧ b < 2 := ⟨Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num) (by norm_num), Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num) (by norm_num)⟩
  have hc : 1 < c ∧ c < 2 := ⟨Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num) (by norm_num), Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num) (by norm_num)⟩
  have hbc : b < c := Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num) (by norm_num)
  exact ⟨hbc, ha.2⟩

end order_of_a_b_c_l663_663288


namespace gardening_project_cost_l663_663496

def cost_rose_bushes : ℕ := 20 * 150 - (0.05 * 3000).toInt
def cost_fertilizer : ℕ := 20 * 25 - (0.10 * 500).toInt
def free_fruit_trees : ℕ := 10 / 3
def cost_fruit_trees : ℕ := (10 - free_fruit_trees) * 75
def cost_ornamental_shrubs : ℕ := 5 * 50
def gardener_hours : ℕ := 6 + 5 + 4 + 7
def cost_gardener : ℕ := gardener_hours * 30
def cost_soil : ℕ := 100 * 5
def tool_rental_tiller : ℕ := 3 * 40
def tool_rental_wheelbarrow : ℕ := 3 * 10
def cost_tool_rental : ℕ := tool_rental_tiller + tool_rental_wheelbarrow
def total_cost : ℕ := cost_rose_bushes + cost_fertilizer + cost_fruit_trees + cost_ornamental_shrubs + cost_gardener + cost_soil + cost_tool_rental

theorem gardening_project_cost : total_cost = 6385 := by
  sorry

end gardening_project_cost_l663_663496


namespace volume_of_pyramid_klmn_l663_663437

noncomputable def pyramidVolume (R KN ML : ℝ) (N K L M : Point) := 1

theorem volume_of_pyramid_klmn {R KN ML : ℝ} {N K L M : Point}
  (h1 : R = 3 / 2) 
  (h2 : dist N K = 3 * sqrt 5 / 2) 
  (h3 : dist L M = 2) 
  (h4 : tangent K L R N)
  (h5 : tangent K M R N) : 
  pyramidVolume R (dist N K) (dist L M) N K L M = 1 := 
sorry

end volume_of_pyramid_klmn_l663_663437


namespace ellipse_foci_distance_l663_663090

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end ellipse_foci_distance_l663_663090


namespace problem_conditions_l663_663933

-- Geometric sequence conditions
def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Arithmetic sequence conditions
def arithmetic_seq (b : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, b (n + 1) = b n + d

-- Definition of T_n
def sum_arithmetic_seq (b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n, T n = ∑ i in finset.range n, b (i + 1)

-- Proof statements
theorem problem_conditions (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (q : ℝ) (d : ℝ) (n : ℕ) :
  geometric_seq a q ∧ arithmetic_seq b d ∧ sum_arithmetic_seq b T ∧ a 1 + 2 * a 2 = 3 * a 3 →
  (q = 1 ∨ q = -1/3) ∧
  (q = 1 → ∀ n, n ≥ 2 → T n > b n) ∧
  (q = -1/3 → (n > 14 → T n < b n) ∧ (n = 14 → T n = b n) ∧ (2 ≤ n ∧ n < 14 → T n > b n)) :=
sorry

end problem_conditions_l663_663933


namespace last_digit_7_powers_l663_663435

theorem last_digit_7_powers :
  (∃ n : ℕ, (∀ k < 4004, k.mod 2002 == n))
  := sorry

end last_digit_7_powers_l663_663435


namespace find_sixty_percent_of_x_l663_663959

variable x : ℝ

theorem find_sixty_percent_of_x
  (h : 0.40 * x = 160) :
  0.60 * x = 240 :=
sorry

end find_sixty_percent_of_x_l663_663959


namespace paint_cubes_l663_663267

def cube_faces := 6
def small_cubes := 8
def black_faces := 8
def total_faces := cube_faces * small_cubes

theorem paint_cubes : 
  ∃ (ways : Nat), ways = 10 ∧ 
  (∀ (paint_distribution : List Nat), 
    paint_distribution.sum = black_faces ∧ 
    (∀ cube in paint_distribution, cube_faces ≥ cube) → 
    List.length (List.duplicate_count paint_distribution) = small_cubes) :=
begin
  sorry
end

end paint_cubes_l663_663267


namespace CorrectPropositions_l663_663610

-- Definitions for the conditions
def PropositionA := ∀ (u v w : Vector ℝ 3), ¬(¬ linearly_independent ℝ ![u, v, w] ∧ ¬ coplanar ℝ ![u, v, w])
def PropositionB (a b : Vector ℝ 3) := (a = c * b) → (∀ (c : ℝ), ¬ linearly_independent ℝ ![a, b, c, Vector.zeros (3 - 2)])
def PropositionC (a b c : Vector ℝ 3) := basis ℝ ![a, b, c] → basis ℝ ![c, a + b, a - b]
def PropositionD (A B M N : Point ℝ 3) := ¬ basis ℝ ![B - A, B - M, B - N] → coplanar ℝ ![A, B, M, N]

-- The theorem statement
theorem CorrectPropositions :
  ¬ PropositionA ∧ PropositionB ∧ PropositionC ∧ PropositionD :=
by
  sorry

end CorrectPropositions_l663_663610


namespace max_knights_seated_next_to_two_knights_l663_663490

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l663_663490


namespace proof_problem_l663_663000

-- Lean 4 statement representation
theorem proof_problem
  (vertex_origin : ∀ C : ℝ × ℝ → Prop, C (0, 0))
  (focus_x_axis : ∀ C : ℝ × ℝ → Prop, ∃ f : ℝ, C (f, 0))
  (line_l_intersect_C : ∀ C : ℝ × ℝ → Prop, ∃ P Q : ℝ × ℝ, C P ∧ C Q ∧ P.1 = 1 ∧ Q.1 = 1 ∧ P.2 ≠ Q.2 ∧ ∀ O : ℝ × ℝ, O = (0, 0) → (P.1 * Q.1 + P.2 * Q.2 = 0))
  (circle_M : ∃ M : ℝ × ℝ, M = (2, 0) ∧ ∀ l : ℝ → Prop, l 1 → ∃ r : ℝ, r = 1 ∧ (∀ (x y : ℝ), (x - 2)^2 + y^2 = r^2))
  (A1 A2 A3 : ℝ × ℝ)
  (A1_A2_tangent : ∀ l : ℝ × ℝ → Prop, ∀ C M : ℝ × ℝ → Prop, l A1 → l A2 → C A1 → C A2 → M (2, 0) → ( ∀ A1 A2 : ℝ × ℝ, ∃ k : ℝ, l k) )
  (A1_A3_tangent : ∀ l : ℝ × ℝ → Prop, ∀ C M : ℝ × ℝ → Prop, l A1 → l A3 → C A1 → C A3 → M (2, 0) → ( ∀ A1 A3 : ℝ × ℝ, ∃ k : ℝ, l k) )
  (points_on_C : ∀ C : ℝ × ℝ → Prop, C A1 ∧ C A2 ∧ C A3) :
  (∃ C : ℝ × ℝ → Prop, (∀ x y : ℝ, C (x, y) ↔ y^2 = x)) ∧
  (∃ M : ℝ × ℝ → Prop, (∀ x y : ℝ, M (x, y) ↔ (x - 2)^2 + y^2 = 1)) ∧
  (∀ A2 A3 : ℝ × ℝ, tangent (A2, A3, 2, 0, 1)) := by
  sorry

end proof_problem_l663_663000


namespace difference_smallest_three_digit_largest_two_digit_l663_663380

theorem difference_smallest_three_digit_largest_two_digit :
  ∀ (a b : ℕ), (a = 100) ∧ (b = 99) → (a - b) = 1 :=
by
  intros a b h
  cases h with ha hb
  rw [ha, hb]
  exact Nat.sub_self_add 99 1 sorry

end difference_smallest_three_digit_largest_two_digit_l663_663380


namespace unique_zero_of_derivative_f_greater_than_constant_l663_663184

/-- Part 1: Prove that the derivative f'(x) has exactly one zero given 0 < a < 2 for f(x) = e^x - ln(2x + a) -/
theorem unique_zero_of_derivative (a : ℝ) (h : 0 < a ∧ a < 2) :
  ∃! x : ℝ, (x > -a / 2) ∧ (HasDerivAt (fun x => Real.exp x - Real.log (2 * x + a)) (Real.exp x - (2 / (2 * x + a))) x)
sorry

/-- Part 2: Prove that f(x) > (3/2 - ln 2) for all x in the domain when a = 1 -/
theorem f_greater_than_constant (a : ℝ) (hx : a = 1) :
  ∀ x : ℝ, (x > -1 / 2) → (Real.exp x - Real.log (2 * x + 1) > 3 / 2 - Real.log 2)
sorry

end unique_zero_of_derivative_f_greater_than_constant_l663_663184


namespace updated_mean_corrected_dataset_l663_663345

theorem updated_mean_corrected_dataset :
  let n := 50
  let mean := 200
  let decreased_amount := 34
  let missing_observations := [150, 190, 210]
  let extra_observation := 250

  let original_sum := n * mean
  let corrected_sum_decrement := original_sum - (n * decreased_amount)
  let corrected_sum_missing := corrected_sum_decrement + (missing_observations.sum)
  let final_corrected_sum  := corrected_sum_missing - extra_observation
  let corrected_n := n - 1 + missing_observations.length
  let updated_mean := final_corrected_sum / corrected_n

  updated_mean = 165.38 :=
by {
  let n := 50
  let mean := 200
  let decreased_amount := 34
  let missing_observations := [150, 190, 210]
  let extra_observation := 250

  let original_sum := n * mean
  let corrected_sum_decrement := original_sum - (n * decreased_amount)
  let corrected_sum_missing := corrected_sum_decrement + (missing_observations.sum)
  let final_corrected_sum  := corrected_sum_missing - extra_observation
  let corrected_n := n - 1 + missing_observations.length
  let my_updated_mean := final_corrected_sum / corrected_n

  have h1 : my_updated_mean = 165.38, from sorry

  exact h1
}
 
end updated_mean_corrected_dataset_l663_663345


namespace ones_digit_sum_of_powers_l663_663381

theorem ones_digit_sum_of_powers :
  (1^1002 + 2^1002 + 3^1002 + ... + 1002^1002) % 10 = 3 :=
sorry

end ones_digit_sum_of_powers_l663_663381


namespace eccentricity_range_l663_663914

variables (a b c e : ℝ)
variables (a_pos : 0 < a) (b_pos : 0 < b)
variables (H : ∀ (x y : ℝ), x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)
variables (min_value_range : -3/4 * c^2 ≤ - (c - a) * (c + a) ∧ - (c - a) * (c + a) ≤ -1/2 * c^2)

theorem eccentricity_range (h : ∀ P : ℝ × ℝ, ∃ PF1 PF2 : ℝ × ℝ, PF1 ⋅ PF2) :
  sqrt 2 ≤ e ∧ e ≤ 2 :=
sorry

end eccentricity_range_l663_663914


namespace problem_l663_663928

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f(x)

noncomputable def is_even (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = g(x)

theorem problem (f g : ℝ → ℝ) (h₀ : is_odd f) (h₁ : is_even g)
  (h₂ : ∀ x, g x ≠ 0) :
  (is_odd (λ x, f(x) * g(x))) ∧ (is_odd (λ x, f(x) / g(x))) :=
by
  sorry

end problem_l663_663928


namespace inequality_holds_iff_x_in_interval_l663_663511

theorem inequality_holds_iff_x_in_interval (x : ℝ) :
  (∀ n : ℕ, 0 < n → (1 + x)^n ≤ 1 + (2^n - 1) * x) ↔ (0 ≤ x ∧ x ≤ 1) :=
sorry

end inequality_holds_iff_x_in_interval_l663_663511


namespace max_knights_between_knights_l663_663473

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l663_663473


namespace limit_problem_l663_663105

open Real Topology

noncomputable def limit_function := λ x : ℝ, (5 - 4 / (cos x)) ^ (1 / (sin (3 * x))^2)

theorem limit_problem : tendsto limit_function (nhds 0) (𝓝 (exp (-2 / 9))) :=
sorry

end limit_problem_l663_663105


namespace problem_lean_l663_663973

noncomputable def a : ℕ+ → ℝ := sorry

theorem problem_lean :
  a 11 = 1 / 52 ∧ (∀ n : ℕ+, 1 / a (n + 1) - 1 / a n = 5) → a 1 = 1 / 2 :=
by
  sorry

end problem_lean_l663_663973


namespace find_f2_l663_663907

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := 
by
  sorry

end find_f2_l663_663907


namespace tesses_ride_is_longer_l663_663715

noncomputable def tesses_total_distance : ℝ := 0.75 + 0.85 + 1.15
noncomputable def oscars_total_distance : ℝ := 0.25 + 1.35

theorem tesses_ride_is_longer :
  (tesses_total_distance - oscars_total_distance) = 1.15 := by
  sorry

end tesses_ride_is_longer_l663_663715


namespace books_per_bookshelf_l663_663099

theorem books_per_bookshelf (total_books : ℕ) (bookshelves : ℕ) (h_total : total_books = 504) (h_shelves : bookshelves = 9) :
  total_books / bookshelves = 56 :=
by
  rw [h_total, h_shelves]
  norm_num
  -- sorry

end books_per_bookshelf_l663_663099


namespace ants_meet_at_QS_l663_663017

theorem ants_meet_at_QS (P Q R S : Type)
  (dist_PQ : Nat)
  (dist_QR : Nat)
  (dist_PR : Nat)
  (ants_meet : 2 * (dist_PQ + (5 : Nat)) = dist_PQ + dist_QR + dist_PR)
  (perimeter : dist_PQ + dist_QR + dist_PR = 24)
  (distance_each_ant_crawls : (dist_PQ + 5) = 12) :
  5 = 5 :=
by
  sorry

end ants_meet_at_QS_l663_663017


namespace distance_third_day_l663_663263

theorem distance_third_day (total_distance : ℝ) (days : ℕ) (first_day_factor : ℝ) (halve_factor : ℝ) (third_day_distance : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ first_day_factor = 4 ∧ halve_factor = 0.5 →
  third_day_distance = 48 := sorry

end distance_third_day_l663_663263


namespace knights_max_seated_between_knights_l663_663465

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l663_663465


namespace proof_numbers_exist_l663_663873

noncomputable def exists_numbers : Prop :=
  ∃ a b c : ℕ, a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
  (a * b % (a + 2012) = 0) ∧
  (a * c % (a + 2012) = 0) ∧
  (b * c % (b + 2012) = 0) ∧
  (a * b * c % (b + 2012) = 0) ∧
  (a * b * c % (c + 2012) = 0)

theorem proof_numbers_exist : exists_numbers :=
  sorry

end proof_numbers_exist_l663_663873


namespace distinct_ordered_pairs_sum_of_reciprocals_eq_one_sixth_l663_663629

theorem distinct_ordered_pairs_sum_of_reciprocals_eq_one_sixth :
  {p : ℕ × ℕ | (1 : ℚ) / (p.1 : ℚ) + (1 : ℚ) / (p.2 : ℚ) = 1 / 6}.to_finset.card = 9 :=
sorry

end distinct_ordered_pairs_sum_of_reciprocals_eq_one_sixth_l663_663629


namespace small_triangles_required_l663_663059

def area_equilateral_triangle (s : ℝ) : ℝ :=
  (math.sqrt 3 / 4) * s^2

theorem small_triangles_required (side_large : ℝ) (side_small : ℝ) (h_large : side_large = 12) (h_small : side_small = 1) :
  let area_large := area_equilateral_triangle side_large,
      area_small := area_equilateral_triangle side_small in
  (area_large / area_small) = 144 :=
by
  -- Definitions only, proof skipped.
  sorry

end small_triangles_required_l663_663059


namespace value_added_to_numerator_l663_663350

theorem value_added_to_numerator (x : ℤ) : 
  let numerator := 63 in
  let sum := 349 in
  numerator + x = sum → x = 286 :=
by
  intros
  exact sorry

end value_added_to_numerator_l663_663350


namespace karen_wins_in_race_l663_663797

theorem karen_wins_in_race (w : ℝ) (h1 : w / 45 > 1 / 15) 
    (h2 : 60 * (w / 45 - 1 / 15) = w + 4) : 
    w = 8 / 3 := 
sorry

end karen_wins_in_race_l663_663797


namespace ratio_area_PQM_PQN_l663_663421

-- Define the relevant geometric setup
variables (A B C M N P Q : Point)
variable (xyz_circle : Circle)
variable (triangle_ABC : Triangle)
variable (xyz_bisector : Line)

-- The conditions in the Lean 4 formalization
def conditions : Prop :=
  xyz_circle.inscribed_in triangle_ABC ∧
  xyz_circle.touches_side AC M ∧
  xyz_circle.touches_side BC N ∧
  xyz_circle.intersects_bisector BD P Q ∧
  triangle_ABC.angle_A = π / 4 ∧
  triangle_ABC.angle_B = π / 3

-- The theorem we want to prove
theorem ratio_area_PQM_PQN :
  conditions → 
  area_ratio ΔPQM ΔPQN = (sqrt 3 - 1) / sqrt 6 :=
begin
  sorry,
end

end ratio_area_PQM_PQN_l663_663421


namespace wanda_walks_total_distance_l663_663021

theorem wanda_walks_total_distance :
  let miles_per_trip := 0.5
  let trips_per_day := 2
  let days_per_week := 5
  let weeks := 4
  let total_distance := 2 * days_per_week * weeks * miles_per_trip
  total_distance = 40 :=
begin
  sorry -- Proof is not required
end

end wanda_walks_total_distance_l663_663021


namespace leftover_coins_value_l663_663063

theorem leftover_coins_value
  (quarters_per_roll dimes_per_roll : ℕ)
  (james_quarters james_dimes : ℕ)
  (lindsay_quarters lindsay_dimes : ℕ)
  (value_per_quarter value_per_dime : ℝ) :
  quarters_per_roll = 40 →
  dimes_per_roll = 50 →
  james_quarters = 83 →
  james_dimes = 159 →
  lindsay_quarters = 129 →
  lindsay_dimes = 266 →
  value_per_quarter = 0.25 →
  value_per_dime = 0.10 →
  let total_quarters := james_quarters + lindsay_quarters in
  let total_dimes := james_dimes + lindsay_dimes in
  let leftover_quarters := total_quarters % quarters_per_roll in
  let leftover_dimes := total_dimes % dimes_per_roll in
  let value_leftover_quarters := leftover_quarters * value_per_quarter in
  let value_leftover_dimes := leftover_dimes * value_per_dime in
  value_leftover_quarters + value_leftover_dimes = 5.50 :=
by
  intros
  sorry

end leftover_coins_value_l663_663063


namespace neutralization_reaction_l663_663139

/-- When combining 2 moles of CH3COOH and 2 moles of NaOH, 2 moles of H2O are formed
    given the balanced chemical reaction CH3COOH + NaOH → CH3COONa + H2O 
    with a molar ratio of 1:1:1 (CH3COOH:NaOH:H2O). -/
theorem neutralization_reaction
  (mCH3COOH : ℕ) (mNaOH : ℕ) :
  (mCH3COOH = 2) → (mNaOH = 2) → (mCH3COOH = mNaOH) →
  ∃ mH2O : ℕ, mH2O = 2 :=
by intros; existsi 2; sorry

end neutralization_reaction_l663_663139


namespace minimum_number_of_positive_numbers_l663_663974

open Set

def sum_of_any_10_less_than_remaining_11 (a : Finₓ 21 → ℝ) : Prop :=
  ∀ (S : Finset (Finₓ 21)), S.card = 10 → S.sum a < (range 21).filter (λ i, i ∉ S).sum a

theorem minimum_number_of_positive_numbers (a : Finₓ 21 → ℝ) (h : sum_of_any_10_less_than_remaining_11 a) : 
  ∃ (P : Finset (Finₓ 21)), P.card = 21 ∧ (∀ i ∈ P, 0 < a i) :=
begin
  sorry
end

end minimum_number_of_positive_numbers_l663_663974


namespace circle_arc_length_l663_663265

theorem circle_arc_length (O T Q : Point × Point × Point)
  (angle_TOQ_eq : angle O T Q = 45)
  (radius_OT_eq : distance O T = 15) :
  arc_length O T Q = 7.5 * π :=
by
  -- Proof can be added here
  sorry

end circle_arc_length_l663_663265


namespace imaginary_part_of_complex_expr_l663_663134

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number problem
noncomputable
def complex_expr : ℂ := (2 + i) / i

-- State the theorem
theorem imaginary_part_of_complex_expr :
  Complex.imag (complex_expr * i) = -2 := by
  sorry

end imaginary_part_of_complex_expr_l663_663134


namespace simplify_complex_expression_l663_663725

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) + 4 * i * (7 - 3 * i) = 40 + 14 * i :=
by
  sorry

end simplify_complex_expression_l663_663725


namespace smallest_n_divisible_by_2013_l663_663395

def is_even (n : ℕ) : Prop := n % 2 = 0

def consecutive_even_product (n : ℕ) : ℕ :=
  ∏ k in (finset.range (n // 2 + 1)).map (multiset.map ((*) 2)), k

theorem smallest_n_divisible_by_2013 :
  ∃ n : ℕ, is_even n ∧ 2013 ∣ consecutive_even_product n ∧ (∀ m : ℕ, m < n → is_even m → ¬ (2013 ∣ consecutive_even_product m)) :=
sorry

end smallest_n_divisible_by_2013_l663_663395


namespace smallest_Norwegian_l663_663061

def is_Norwegian (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a * b * c ∧ a + b + c = 2022

theorem smallest_Norwegian :
  ∀ n : ℕ, is_Norwegian n → 1344 ≤ n := by
  sorry

end smallest_Norwegian_l663_663061


namespace domain_translation_l663_663966

theorem domain_translation (f : ℝ → ℝ) (h : ∀ x, x ∈ Icc (-1 : ℝ) 0 → x ∈ dom f) :
  ∀ x, x ∈ Icc (-2 : ℝ) (-1) → x + 1 ∈ dom f :=
by
  intro x h
  sorry

end domain_translation_l663_663966


namespace fewest_presses_to_return_to_81_l663_663638

-- Define the reciprocal function f
def f (x : ℝ) : ℝ := 1 / x

-- Theorem to prove the fewest number of presses required
theorem fewest_presses_to_return_to_81 : (∃ n : ℕ, n = 2 ∧ (f ∘ f) 81 = 81) :=
by
  have h1 : f 81 = 1 / 81,
  { sorry },
  have h2 : f (1 / 81) = 81,
  { sorry },
  existsi 2,
  split,
  { refl },
  { exact h2 }

end fewest_presses_to_return_to_81_l663_663638


namespace probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five_l663_663576

noncomputable def probability_of_product_ending_in_0 : ℝ :=
  let p_no_0 := (9 / 10 : ℝ) ^ 20
  let p_at_least_one_0 := 1 - p_no_0
  let p_no_even := (5 / 9 : ℝ) ^ 20
  let p_at_least_one_even := 1 - p_no_even
  let p_no_5_in_19 := (8 / 9 : ℝ) ^ 19
  let p_at_least_one_5 := 1 - p_no_5_in_19
  p_at_least_one_0 + p_no_0 * p_at_least_one_even * p_at_least_one_5

theorem probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five :
  abs (probability_of_product_ending_in_0 - 0.9865) < 0.0001 :=
begin
  -- proofs go here, sorry is used to skip the proof details
  sorry
end

end probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five_l663_663576


namespace angle_between_vectors_l663_663622

open Real

variables (a b : ℝ^2) -- Assuming 2D vectors for simplicity

-- Given conditions
def magnitude_a : ℝ := 4
def magnitude_b : ℝ := 3
def dot_product_condition : ℝ := (a + 2 • b) ⬝ (a - b) = 4

-- Question rewritten as a proof statement.
theorem angle_between_vectors (h1 : ∥a∥ = magnitude_a) (h2 : ∥b∥ = magnitude_b) (h3 : (a + 2 • b) ⬝ (a - b) = dot_product_condition) :
  ∃ θ : ℝ, θ = 60 ∧ cos θ = (a ⬝ b) / (∥a∥ * ∥b∥) :=
begin
  sorry
end

end angle_between_vectors_l663_663622


namespace volume_in_cubic_yards_l663_663430

theorem volume_in_cubic_yards (volume_cubic_feet : ℕ) (cubic_feet_in_yard : ℕ) (h1 : volume_cubic_feet = 216) (h2 : cubic_feet_in_yard = 27) :
  volume_cubic_feet / cubic_feet_in_yard = 8 :=
by
  rw [h1, h2]
  norm_num

end volume_in_cubic_yards_l663_663430


namespace geometric_sequence_sum_l663_663981

theorem geometric_sequence_sum (a r : ℝ) (h1 : a + a*r + a*r^2 = 13) 
    (h2 : a * (1 - r^8) / (1 - r) = 1093) : 
    a * (1 + r + r^2 + r^3 + r^4) = 13 * ((1 + (sqrt 333 / 2))) := 
sorry

end geometric_sequence_sum_l663_663981


namespace sector_area_l663_663964

noncomputable def radius_of_sector (l α : ℝ) : ℝ := l / α

noncomputable def area_of_sector (r l : ℝ) : ℝ := (1 / 2) * r * l

theorem sector_area {α l S : ℝ} (hα : α = 2) (hl : l = 3 * Real.pi) (hS : S = 9 * Real.pi ^ 2 / 4) :
  area_of_sector (radius_of_sector l α) l = S := 
by 
  rw [hα, hl, hS]
  rw [radius_of_sector, area_of_sector]
  sorry

end sector_area_l663_663964


namespace common_point_of_lines_l663_663870

theorem common_point_of_lines (a b c : ℝ) (h : b = a + a / 2 ∧ c = a + 2 * (a / 2)) :
  ∀ x, (∃ y, (a * x + b * y = c)) -> (0, 4 / 3) do
by
  sorry

end common_point_of_lines_l663_663870


namespace granger_bought_4_loaves_of_bread_l663_663626

-- Define the prices of items
def price_of_spam : Nat := 3
def price_of_pb : Nat := 5
def price_of_bread : Nat := 2

-- Define the quantities bought by Granger
def qty_spam : Nat := 12
def qty_pb : Nat := 3
def total_amount_paid : Nat := 59

-- The problem statement in Lean: Prove the number of loaves of bread bought
theorem granger_bought_4_loaves_of_bread :
  (qty_spam * price_of_spam) + (qty_pb * price_of_pb) + (4 * price_of_bread) = total_amount_paid :=
sorry

end granger_bought_4_loaves_of_bread_l663_663626


namespace sum_of_coefficients_excluding_x3_in_binomial_expansion_l663_663357

theorem sum_of_coefficients_excluding_x3_in_binomial_expansion :
  let f := fun (x : ℕ) => ∑ i in Finset.range (x+1), (Nat.choose x i)
  let total_sum := f 5
  let coefficient_of_x3 := Nat.choose 5 3
  let sum_excluding_x3 := total_sum - coefficient_of_x3
  sum_excluding_x3 = 22 := by
  sorry

end sum_of_coefficients_excluding_x3_in_binomial_expansion_l663_663357


namespace correct_calculation_l663_663029

theorem correct_calculation :
  (∀ a : ℝ, a^3 + a^2 ≠ a^5) ∧
  (∀ a : ℝ, a^3 / a^2 = a) ∧
  (∀ a : ℝ, 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ a : ℝ, (a - 2)^2 ≠ a^2 - 4) :=
by
  sorry

end correct_calculation_l663_663029


namespace min_lambda_inequality_l663_663172

theorem min_lambda_inequality (x y z : ℝ) (h_pos: x > 0 ∧ y > 0 ∧ z > 0) (h_sum: x + y + z = 1) :
  ∃ λ : ℝ, λ = 5 ∧ ∀ λ, (λ ≥ 5) → (λ * (x * y + y * z + z * x) ≥ 3 * (λ + 1) * x * y * z + 1) :=
sorry

end min_lambda_inequality_l663_663172


namespace courier_cost_formula_l663_663426

def cost (P : ℕ) : ℕ :=
if P = 0 then 0 else max 50 (30 + 7 * (P - 1))

theorem courier_cost_formula (P : ℕ) : cost P = 
  if P = 0 then 0 else max 50 (30 + 7 * (P - 1)) :=
by
  sorry

end courier_cost_formula_l663_663426


namespace range_of_a_l663_663244

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > -1 / 2 ∧ x < 0 → log (a^2 - 1) (2 * x + 1) > 0) ↔ (a > 1 ∧ a < sqrt 2) ∨ (a < -1 ∧ a > - sqrt 2) :=
by {
  sorry
}

end range_of_a_l663_663244


namespace president_vice_secretary_choice_l663_663988

theorem president_vice_secretary_choice (n : ℕ) (h : n = 6) :
  (∀ a b c : fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (n * (n - 1) * (n - 2) = 120) := 
sorry

end president_vice_secretary_choice_l663_663988


namespace hyperbola_eccentricity_l663_663864

/--
Consider the hyperbola C: x^2 / a^2 - y^2 / b^2 = 1 (a > 0, b > 0)
with left and right foci F1 and F2, respectively.
On the right branch of the hyperbola C, there exists a point P
such that the inscribed circle of triangle PF1F2 has a radius of a.
The center of the circle is denoted as M, the centroid of triangle PF1F2 is G,
and MG is parallel to F1F2. Prove that the eccentricity of the hyperbola C is 2.
-/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
    (F1 F2 P : ℝ × ℝ) (H1 : ∀ x y, (x, y) ∈ {F1, F2, P} → x ≠ 0)
    (radius_PF1F2_inscribed_circle : inscribed_circle_radius P F1 F2 = a)
    (G M : ℝ × ℝ)
    (centroid_PF1F2 : is_centroid G P F1 F2)
    (center_inscribed_circle : is_center M P F1 F2)
    (parallel_MG_F1F2 : parallel (G.1 - M.1, G.2 - M.2) (F1.1 - F2.1, F1.2 - F2.2)) :
    eccentricity (a : ℝ) (b : ℝ) = 2 :=
sorry

end hyperbola_eccentricity_l663_663864


namespace part_a_part_b_l663_663305

-- Define the natural numbers m and n
variable (m n : Nat)

-- Condition: m * n is divisible by m + n
def divisible_condition : Prop :=
  ∃ (k : Nat), m * n = k * (m + n)

-- Define prime number
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ d : Nat, d ∣ p → d = 1 ∨ d = p

-- Define n as the product of two distinct primes
def is_product_of_two_distinct_primes (n : Nat) : Prop :=
  ∃ (p₁ p₂ : Nat), is_prime p₁ ∧ is_prime p₂ ∧ p₁ ≠ p₂ ∧ n = p₁ * p₂

-- Problem (a): Prove that m is divisible by n when n is a prime number and m * n is divisible by m + n
theorem part_a (prime_n : is_prime n) (h : divisible_condition m n) : n ∣ m := sorry

-- Problem (b): Prove that m is not necessarily divisible by n when n is a product of two distinct prime numbers
theorem part_b (prod_of_primes_n : is_product_of_two_distinct_primes n) (h : divisible_condition m n) :
  ¬ (n ∣ m) := sorry

end part_a_part_b_l663_663305


namespace pine_seedlings_in_sample_l663_663817

-- Define the total number of seedlings and the number of pine seedlings
def total_seedlings : ℕ := 30000
def pine_seedlings : ℕ := 4000

-- Define the sample size
def sample_size : ℕ := 150

-- Define the probability of selection in stratified sampling
def probability_of_selection : ℝ := (sample_size : ℝ) / (total_seedlings : ℝ)

-- Prove the number of pine seedlings in the sample
theorem pine_seedlings_in_sample : 
  (probability_of_selection * (pine_seedlings : ℝ)) = 20 :=
by
  -- Here we would provide the proof, but for now we use sorry.
  sorry

end pine_seedlings_in_sample_l663_663817


namespace strictly_decreasing_interval_l663_663186

noncomputable def f (x : ℝ) : ℝ :=
  (real.sqrt 3 - real.tan x) * real.cos x ^ 2

theorem strictly_decreasing_interval :
  ∀ x₁ x₂ ∈ set.Icc (11 * real.pi / 12) real.pi, x₁ < x₂ → f x₁ > f x₂ :=
by
  sorry

end strictly_decreasing_interval_l663_663186


namespace cos_C_area_ABC_l663_663679

theorem cos_C (B C : ℝ) (hB : B = π / 6) (h1 : 6 * cos B * cos C - 1 = 3 * cos (B - C)) : 
  cos C = (2 * real.sqrt 2 + real.sqrt 3) / 6 :=
by
  sorry

theorem area_ABC (a b c AD : ℝ) (hA : a = 4) (hC : c = 3) (hAD : AD = 8 * real.sqrt 3 / 7)
  (hBC_bisector : true) (h_cos_C : cos C = (2 * real.sqrt 2 + real.sqrt 3) / 6) : 
  let S := (1/2) * c * a * sin ((2 * real.sqrt 2) / 3) in
  S = 4 * real.sqrt 2 :=
by
  sorry

end cos_C_area_ABC_l663_663679


namespace find_q_l663_663801

section
  variable (d : ℕ) (q : ℚ)

  def arithmetic_seq (n : ℕ) : ℕ := n * d
  def geometric_seq (n : ℕ) : ℚ := d^2 * q^(n-1)

  def a1 := arithmetic_seq d 1
  def a2 := arithmetic_seq d 2
  def a3 := arithmetic_seq d 3

  def b1 := geometric_seq d q 1
  def b2 := geometric_seq d q 2
  def b3 := geometric_seq d q 3

  theorem find_q 
    (cond_a1 : a1 = d)
    (cond_b1 : b1 = d^2)
    (cond_pos_int : (a1^2 + a2^2 + a3^2) / (b1 + b2 + b3)).den = 1) :
    q = 1/2 := 
  sorry
end

end find_q_l663_663801


namespace S_5_value_l663_663168
noncomputable theory

open Nat

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (∑ i in range n, 1 / (2 ^ i) / a i.succ = 2 * n)

def a_n (n : ℕ) : ℝ := 1 / 2^n

def S_n (n : ℕ) : ℝ := ∑ i in range n, a_n (i + 1)

theorem S_5_value :
  sequence_condition a_n → S_n 5 = 31 / 32 :=
by
  intro h
  rw [S_n, finset.sum_range_succ, finset.sum_range_succ, finset.sum_range_succ, finset.sum_range_succ, finset.sum_range_succ, a_n]
  norm_num
  sorry

end S_5_value_l663_663168


namespace max_knights_between_knights_l663_663456

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l663_663456


namespace value_of_m_l663_663040

theorem value_of_m (m : ℕ) : (5^m = 5 * 25^2 * 125^3) → m = 14 :=
by
  sorry

end value_of_m_l663_663040


namespace cover_distance_in_given_time_l663_663440

noncomputable def time_to_cover_mile (width feet : ℝ) (speed : ℝ) := 
  let radius := width / 2
  let semicircle_length := real.pi * radius
  let full_circle_length := 2 * real.pi * radius
  let pattern_length := semicircle_length + full_circle_length
  let mile_in_feet := 5280
  let total_distance := mile_in_feet
  let total_time := total_distance / (pattern_length * (total_distance / pattern_length)) / (speed * 5280)
  total_time

theorem cover_distance_in_given_time : time_to_cover_mile 50 5280 6 = 1 / 6 := 
  by sorry

end cover_distance_in_given_time_l663_663440


namespace represent_same_function_l663_663786

noncomputable def f1 (x : ℝ) : ℝ := (x^3 + x) / (x^2 + 1)
def f2 (x : ℝ) : ℝ := x

theorem represent_same_function : ∀ x : ℝ, f1 x = f2 x := 
by
  sorry

end represent_same_function_l663_663786


namespace complex_proof_l663_663553

open Complex

theorem complex_proof (z : ℂ) (i_un : ℂ := complex.I) (h₁ : i_un / z = 1 + i_un) : z - conj z = i_un := by
  sorry

end complex_proof_l663_663553


namespace units_digit_of_23_mul_51_squared_l663_663543

theorem units_digit_of_23_mul_51_squared : 
  ∀ n m : ℕ, (n % 10 = 3) ∧ ((m^2 % 10) = 1) → (n * m^2 % 10) = 3 :=
by
  intros n m h
  sorry

end units_digit_of_23_mul_51_squared_l663_663543


namespace sum_four_digit_integers_l663_663382

theorem sum_four_digit_integers : 
  (∑ k in finset.range (5000 - 1000 + 1), k + 1000) = 12003000 :=
by
  sorry

end sum_four_digit_integers_l663_663382


namespace Rajesh_monthly_salary_l663_663721

variable (S : ℝ) -- Rajesh's monthly salary

def food_expense : ℝ := 0.40 * S
def medicine_expense : ℝ := 0.20 * S
def remaining_amount : ℝ := S - (food_expense + medicine_expense)
def savings : ℝ := 0.60 * remaining_amount

theorem Rajesh_monthly_salary (h : savings = 4320) : S = 18000 :=
by
  have h1 : remaining_amount = 0.40 * S := sorry
  have h2 : savings = 0.60 * (0.40 * S) := sorry
  have h3 : 0.24 * S = 4320 := sorry
  have h4 : S = 4320 / 0.24 := sorry
  exact sorry

end Rajesh_monthly_salary_l663_663721


namespace first_point_x_coord_l663_663675

theorem first_point_x_coord (m n : ℝ) (h1 : m = 2n + 5)
  (h2 : m + 4 = 2 * (n + 2) + 5) : m = 2n + 5 :=
by
  sorry

end first_point_x_coord_l663_663675


namespace product_ends_in_0_l663_663572

/-- Given a set of 20 random digits, the probability that the product of these digits ends in 
    0 is approximately 0.988 -/
theorem product_ends_in_0 (s : Finset ℕ) (h: s.card = 20) :
  (∑ k in Finset.range 10, if k = 0 then 1 else if ∃ n ∈ s, n = 2 ∧ ∃ m ∈ s, m = 5 then 1 else 0) / 10 ^ 20 ≈ 0.988 := 
sorry

end product_ends_in_0_l663_663572


namespace max_knights_adjacent_to_two_other_knights_l663_663486

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l663_663486


namespace smallest_period_value_of_a_l663_663702

noncomputable def f (x a : ℝ) : ℝ :=
  sqrt 3 * sin x * cos x + cos x ^ 2 + a

theorem smallest_period :
  ∀ a : ℝ, let f := f x a in
    (∀ x : ℝ, f (x + π) = f x) ∧
    ∀ k : ℤ,
      (∀ x : ℝ, (π / 6 + ↑k * π) ≤ x ∧ x ≤ (2 * π / 3 + ↑k * π) → (-2 * π / 3 + ↑k * π) ≤ x ∧ x ≤ (-π / 6 + ↑k * π)) :=
by
  sorry

theorem value_of_a (a : ℝ) :
  let f := f x a in
    ((-π / 6 ≤ x ∧ x ≤ π / 3) ∧
      ∀ x : ℝ, f (x - π / 6) + f (x + π / 3) = 3 / 2) →
    a = 0 :=
by
  sorry

end smallest_period_value_of_a_l663_663702


namespace max_value_of_function_for_x_lt_0_l663_663872

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem max_value_of_function_for_x_lt_0 :
  ∀ x : ℝ, x < 0 → f x ≤ -4 ∧ (∃ y : ℝ, f y = -4 ∧ y < 0) := sorry

end max_value_of_function_for_x_lt_0_l663_663872


namespace find_certain_number_l663_663958

theorem find_certain_number (x : ℝ) (h : ((x^4) * 3.456789)^10 = 10^20) : x = 10 :=
sorry

end find_certain_number_l663_663958


namespace fifth_term_of_seq_l663_663730

-- Define the sequence
def seq (n : ℕ) : ℚ :=
  if even n then 
    -((n + 1) / (n + 2) : ℚ) 
  else 
    ((n + 1) / (n + 2) : ℚ)

theorem fifth_term_of_seq :
  seq 4 = 10 / 11 :=
by sorry

end fifth_term_of_seq_l663_663730


namespace same_parity_as_cos_l663_663846

-- Define the functions involved
def cos_x (x : ℝ) : ℝ := Real.cos x
def tan_x (x : ℝ) : ℝ := Real.tan x
def abs_sin_x (x : ℝ) : ℝ := abs (Real.sin x)
def sin_x (x : ℝ) : ℝ := Real.sin x
def neg_sin_x (x : ℝ) : ℝ := - (Real.sin x)

-- The property for even functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- The theorem to prove
theorem same_parity_as_cos : is_even abs_sin_x :=
by
  unfold is_even abs_sin_x Real.sin abs
  intro x
  rw [Real.sin_neg, abs_neg]
  sorry

end same_parity_as_cos_l663_663846


namespace Keith_total_spent_l663_663687

variables (spent_speakers : ℝ) (spent_cd_player : ℝ) (spent_tires : ℝ)
variables (wanted_cds : ℝ)

-- Stating the conditions
def Keith_spent := spent_speakers = 136.01 ∧ spent_cd_player = 139.38 ∧ spent_tires = 112.46 ∧ wanted_cds = 6.16

-- Prove the total amount spent
theorem Keith_total_spent (h : Keith_spent spent_speakers spent_cd_player spent_tires wanted_cds) : 
  spent_speakers + spent_cd_player + spent_tires = 387.85 :=
by
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  rw [h1, h3, h5],
  norm_num,
  sorry

end Keith_total_spent_l663_663687


namespace sum_arithmetic_series_1000_to_5000_l663_663384

theorem sum_arithmetic_series_1000_to_5000 :
  ∑ k in finset.range (5001 - 1000), (1000 + k) = 12003000 :=
by
  sorry

end sum_arithmetic_series_1000_to_5000_l663_663384


namespace integral_equality_derivative_equality_l663_663360

noncomputable def integral_value : ℝ :=
  ∫ x in -3..3, (sqrt (9 - x^2) - x^3)

noncomputable def derivative_value (a : ℝ) (x0 : ℝ) (h : a > 0) : Prop :=
  let f := λ x, a^2 / x
  in deriv f x0 = -4

theorem integral_equality :
  integral_value = 9 * Real.pi / 2 :=
sorry

theorem derivative_equality (a : ℝ) (h : a > 0) (x0 : ℝ) :
  (derivative_value a x0 h) → (x0 = a / 2 ∨ x0 = -a / 2) :=
sorry

end integral_equality_derivative_equality_l663_663360


namespace proof_problem_l663_663295

open Complex

-- Given conditions
def condition1 (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, f (f z) = (z * conj(z) - z - conj(z))^2

def condition2 (f : ℂ → ℂ) : Prop :=
  f 1 = 0

-- Theorem to prove |f(i) - 1| = 1
theorem proof_problem (f : ℂ → ℂ) (h1 : condition1 f) (h2 : condition2 f) : 
  |f(Complex.i) - 1| = 1 :=
sorry

end proof_problem_l663_663295


namespace city_daily_revenue_min_val_l663_663445

open Real

noncomputable def f (t : ℕ) : ℝ := 4 + 1 / t
noncomputable def g (t : ℕ) : ℝ := 120 - abs (t - 20)

def W (t : ℕ) : ℝ :=
  if t > 0 ∧ t ≤ 20 then 401 + 4 * t + 100 / t
  else if t > 20 ∧ t ≤ 30 then 559 + 140 / t - 4 * t
  else 0

theorem city_daily_revenue_min_val :
  (∀ t, 1 ≤ t ∧ t ≤ 30 → W t = (f t) * (g t)) ∧ (∀ t, 1 ≤ t ∧ t ≤ 30 → W t ≥ 441) := 
  by
    sorry

end city_daily_revenue_min_val_l663_663445


namespace fully_filled_boxes_l663_663275

-- Define the number of cards of each type
def numMagicCards := 33
def numRareCards := 28
def numCommonCards := 33

-- Define box capacities
def smallBoxMagicCapacity := 5
def smallBoxRareCapacity := 5
def smallBoxCommonCapacity := 6

def largeBoxMagicCapacity := 10
def largeBoxRareCapacity := 10
def largeBoxCommonCapacity := 15

-- Define the Lean proof problem
theorem fully_filled_boxes :
  let numSmallMagicBoxes := numMagicCards / smallBoxMagicCapacity,
      numSmallRareBoxes := numRareCards / smallBoxRareCapacity,
      numSmallCommonBoxes := numCommonCards / smallBoxCommonCapacity,
      numLargeMagicBoxes := (numMagicCards % smallBoxMagicCapacity) / largeBoxMagicCapacity,
      numLargeRareBoxes := (numRareCards % smallBoxRareCapacity) / largeBoxRareCapacity,
      numLargeCommonBoxes := (numCommonCards % smallBoxCommonCapacity) / largeBoxCommonCapacity in
  numSmallMagicBoxes + numSmallRareBoxes + numSmallCommonBoxes +
  numLargeMagicBoxes + numLargeRareBoxes + numLargeCommonBoxes = 16 := by
  sorry

end fully_filled_boxes_l663_663275


namespace functional_equation_solution_l663_663133

noncomputable def f : ℝ → ℝ :=
  λ x, 1 / 2 - x

theorem functional_equation_solution :
  ∀ x y : ℝ, f (x - f y) = 1 - x - y :=
by
  intro x y
  have h : f (x - f y) = f (x - (1 / 2 - y)), from rfl
  rw [h]
  calc
    f (x - (1 / 2 - y)) = 1 / 2 - (x - (1 / 2 - y)) : rfl
                     ... = 1 / 2 - x + 1 / 2 - y : by ring
                     ... = 1 - x - y : by ring

end functional_equation_solution_l663_663133


namespace dartboard_distribution_count_l663_663098

-- Definition of the problem in Lean 4
def count_dartboard_distributions : ℕ :=
  -- We directly use the identified correct answer
  5

theorem dartboard_distribution_count :
  count_dartboard_distributions = 5 :=
sorry

end dartboard_distribution_count_l663_663098


namespace assembly_shortest_time_l663_663731

-- Define the times taken for each assembly path
def time_ACD : ℕ := 3 + 4
def time_EDF : ℕ := 4 + 2

-- State the theorem for the shortest time required to assemble the product
theorem assembly_shortest_time : max time_ACD time_EDF + 4 = 13 :=
by {
  -- Introduction of the given conditions and simplified value calculation
  sorry
}

end assembly_shortest_time_l663_663731


namespace triangle_XYZ_length_XY_l663_663978

theorem triangle_XYZ_length_XY
  (XYZ : Type*) [Triangle XYZ]
  (angleX : XYZ.angle = 90)
  (tanZ : XYZ.tangent = 3)
  (lengthYZ : XYZ.length = 150) :
  XYZ.lengthXY = 45 * sqrt 10 := by
  sorry

end triangle_XYZ_length_XY_l663_663978


namespace decimal_fraction_to_percentage_l663_663419

theorem decimal_fraction_to_percentage (d : ℝ) (h : d = 0.03) : d * 100 = 3 := by
  sorry

end decimal_fraction_to_percentage_l663_663419


namespace part_I_part_II_l663_663613

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^2 + a * x - b * log x

theorem part_I :
  let a := 0
  let b := 1
  let f := f x a b
  exists x: ℝ, x = real.sqrt(2)/2 ∧ ∀ x > 0, f x ≥ f (real.sqrt(2)/2) ∧
  ∀ x, x > real.sqrt(2)/2 → f x > f (real.sqrt(2)/2) :=
sorry

theorem part_II :
  let a_vals := {-6, -5, -4, -3}
  let b_vals := {-2, -3, -4}
  (∑ a in a_vals, ∑ b in b_vals, if a^2 + 8 * b > 0 ∧ -a / 2 > 0 ∧ -b / 2 > 0 then 1 else 0) /
  (a_vals.card * b_vals.card) = 5 / 12 :=
sorry

end part_I_part_II_l663_663613


namespace second_player_always_wins_l663_663162

theorem second_player_always_wins (n : ℕ) (h1 : n > 1) :
  ∃ (strategy_for_second : (fin n) → (fin n)), 
  ∀ (strategy_for_first : (fin n) → (fin n)), 
  let 
    -- translate strategies into marked points
    first_player_marks := set_of (strategy_for_first),
    second_player_marks := set_of (strategy_for_second),
    -- define the endpoints of arcs for both players
    first_player_arcs := {arc ∈ circle_arcs | ∃ x y ∈ first_player_marks, arc = [x, y] ∧ arc_not_marked_between x y},
    second_player_arcs := {arc ∈ circle_arcs | ∃ x y ∈ second_player_marks, arc = [x, y] ∧ arc_not_marked_between x y},
    max_first_arc := max_length (first_player_arcs),
    max_second_arc := max_length (second_player_arcs)
  in
  max_second_arc > max_first_arc.

end second_player_always_wins_l663_663162


namespace dubblefud_red_balls_l663_663400

theorem dubblefud_red_balls (R B : ℕ) 
  (h1 : 2 ^ R * 4 ^ B * 5 ^ B = 16000)
  (h2 : B = G) : R = 6 :=
by
  -- Skipping the actual proof
  sorry

end dubblefud_red_balls_l663_663400


namespace max_value_of_M_l663_663592

theorem max_value_of_M {x y z w : ℝ} (h : x + y + z + w = 1) :
  (xw + 2yw + 3xy + 3zw + 4xz + 5yz ≤ 3/2) := by
  sorry

end max_value_of_M_l663_663592


namespace two_point_form_eq_l663_663906

theorem two_point_form_eq : 
  ∀ x y, (∃ A B : ℝ × ℝ, A = (1, 2) ∧ B = (-1, 1)) → 
         (∃ (x y: ℝ), (x - 1)/(-2) = (y - 2)/(-1)) := 
by
  -- Placeholder for the proof
  intro x y
  rintro ⟨A, B, hA, hB⟩
  sorry

end two_point_form_eq_l663_663906


namespace consecutive_sum_l663_663212

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l663_663212


namespace calculate_f_diff_l663_663867

noncomputable def f : ℝ → ℝ :=
  sorry

theorem calculate_f_diff :
  (∀ x, f(x-1) = f(4-x)) ∧
  (∀ x, 0 < x ∧ x < 3/2 → f(x) = x) ∧
  (∀ x, f(-x) = -f(x)) →
  f(2012) - f(2010) = 1 :=
by
  sorry

end calculate_f_diff_l663_663867


namespace correct_statements_ABCD_l663_663407

-- Given lengths of sides
def AB : ℕ := 1
def BC : ℕ := 9
def CD : ℕ := 8
def DA : ℕ := 6

-- Proof Problem in Lean 4 statement form
theorem correct_statements_ABCD :
    (¬ ∃ AC BD : ℝ, (AC ^ 2 + BD ^ 2 = 1 ^ 2 + 9 ^ 2 + 8 ^ 2 + 6 ^ 2 ∧ AC = √100) ∧ ¬(angle_ADC_ge_90 6 8)) ∧
    ¬(isosceles_triangle 9 8 (√(BD^2))) :=
sorry

-- Helper definitions that might be utilized
def angle_ADC_ge_90 : ℕ → ℕ → Prop := 
λ DA CD, AC ≥ 10

def isosceles_triangle : ℕ → ℕ → Prop := 
λ BC CD BD, (BC = CD) ∨ (BC = BD) ∨ (CD = BD)

end correct_statements_ABCD_l663_663407


namespace magnitude_of_complex_root_unique_l663_663330

theorem magnitude_of_complex_root_unique (z : ℂ) 
  (h : z^2 - 6 * z + 25 = 0) : ∃! m : ℝ, m = complex.abs z := sorry

end magnitude_of_complex_root_unique_l663_663330


namespace problem1_problem2_l663_663159

-- Part 1: Prove that sequence {ln(a_n - 1)} is geometric and find the general formula for {a_n}
theorem problem1 (a : ℕ → ℝ) (h1 : a 1 = 3) (h_rec : ∀ n, a (n + 1) = a n ^ 2 - 2 * a n + 2) :
  (∃ r : ℝ, ∀ n, ∃ k : ℕ, (Real.log (a n - 1)) = (Real.log (2) * r ^ k)) 
  ∧ (∀ n, a n = 2 ^ 2 ^ (n - 1) + 1) := sorry

-- Part 2: Prove that the sum of the first n terms of {b_n} where b_n = 1/a_n + 1/(a_n - 2) is less than 2
theorem problem2 (a b : ℕ → ℝ) (h_a1 : a 1 = 3) 
    (h_arec : ∀ n, a (n + 1) = a n ^ 2 - 2 * a n + 2)
    (h_b : ∀ n, b n = 1/a n + 1/(a n - 2)) :
  (∀ n, ∑ i in Finset.range n, b (i + 1) < 2) := sorry

end problem1_problem2_l663_663159


namespace measure_angle_OBC_is_40_l663_663688

open Real

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry

-- Define the setup of the problem including the points, the triangles, and the given conditions
variables (A B C O P : Point) (ω : Circle)

-- Assume the conditions of the problem
axiom h1 : circumcenter A B C = O
axiom h2 : collinear B O P
axiom h3 : collinear A C P
axiom h4 : circumcircle A O P = ω
axiom h5 : dist O B = dist A P
axiom h6 : measure_arc_not_containing A O P ω = 40

-- Define the problem to prove that the angle ∠OBC is 40 degrees given the conditions
theorem measure_angle_OBC_is_40 : ∠ O B C = 40 := by
  sorry

end measure_angle_OBC_is_40_l663_663688


namespace evaluate_f_four_thirds_l663_663614

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then Real.sin (Real.pi * x) else f (x - 1)

theorem evaluate_f_four_thirds : f (4 / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end evaluate_f_four_thirds_l663_663614


namespace imaginary_part_conjugate_z_l663_663182

-- Given condition
def z : ℂ := (2 * complex.I) / (1 + complex.I)

-- Proof that the imaginary part of the conjugate of z is -1
theorem imaginary_part_conjugate_z : complex.im (conj z) = -1 :=
sorry

end imaginary_part_conjugate_z_l663_663182


namespace lines_perpendicular_l663_663647

-- Define the lines and point
def line1 (a : ℝ) : ℝ × ℝ → Prop :=
  λ point, 2 * point.1 - a * point.2 - 1 = 0

def line2 : ℝ × ℝ → Prop :=
  λ point, point.1 + 2 * point.2 = 0

def point_11 : ℝ × ℝ := (1, 1)

-- Define the slopes
def slope (l : (ℝ × ℝ) → Prop) : ℝ → Prop :=
  ∀ m, (∀ p1 p2 : ℝ × ℝ, l p1 → l p2 → p1.1 ≠ p2.1 → 
  m = (p2.2 - p1.2) / (p2.1 - p1.1))

def slope_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Statement to prove
theorem lines_perpendicular : 
  line1 (1) point_11 → 
  (slope (line1 1) 2) → 
  (slope line2 (-1 / 2)) → 
  slope_perpendicular 2 (-1 / 2) :=
by
  sorry

end lines_perpendicular_l663_663647


namespace cistern_fill_time_l663_663049

def fill_rate (capacity : ℝ) := capacity / 12
def empty_rate (capacity : ℝ) := capacity / 18

theorem cistern_fill_time (C : ℝ) (h : C > 0) :
  let net_rate := (fill_rate C) - (empty_rate C) in
  (C / net_rate) = 36 := by
  sorry

end cistern_fill_time_l663_663049


namespace red_bus_driver_can_see_18_feet_l663_663751

-- Definitions according to the conditions in a)
def length_red_bus : ℝ := 48
def length_orange_car : ℝ := length_red_bus / 4
def length_yellow_bus : ℝ := length_orange_car * 3.5
def length_green_truck : ℝ := length_orange_car * 2
def visible_length_yellow_bus : ℝ := length_yellow_bus - length_green_truck

-- Theorem stating the desired property
theorem red_bus_driver_can_see_18_feet : visible_length_yellow_bus = 18 := by
  sorry

end red_bus_driver_can_see_18_feet_l663_663751


namespace third_pipe_empty_time_l663_663376

theorem third_pipe_empty_time (x : ℝ) :
  (1 / 60 : ℝ) + (1 / 120) - (1 / x) = (1 / 60) →
  x = 120 :=
by
  intros h
  sorry

end third_pipe_empty_time_l663_663376


namespace bouncy_balls_total_l663_663068

theorem bouncy_balls_total (T : ℕ) (h : 0.75 * T = 90) : T = 120 := by
  sorry

end bouncy_balls_total_l663_663068


namespace value_of_f_eval_at_pi_over_12_l663_663187

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem value_of_f_eval_at_pi_over_12 : f (Real.pi / 12) = (Real.sqrt 6) / 2 :=
by
  sorry

end value_of_f_eval_at_pi_over_12_l663_663187


namespace simplify_expression_l663_663123

theorem simplify_expression :
  (sin (real.pi * 38 / 180) * sin (real.pi * 38 / 180) + 
   cos (real.pi * 38 / 180) * sin (real.pi * 52 / 180) - 
   tan (real.pi * 15 / 180) ^ 2) / 
  (3 * tan (real.pi * 15 / 180)) = 
  (2 + real.sqrt 3) / 9 :=
by
  sorry

end simplify_expression_l663_663123


namespace min_ratio_of_five_points_l663_663149

noncomputable def min_ratio (S : set (ℝ × ℝ)) (h : S.card = 5) (h' : ∀ P Q R : (ℝ × ℝ), P ∈ S → Q ∈ S → R ∈ S → P ≠ Q → Q ≠ R → R ≠ P → ¬ collinear ({P, Q, R} : set (ℝ × ℝ))) : ℝ :=
  M(S) / m(S)

theorem min_ratio_of_five_points (S : set (ℝ × ℝ)) (h : S.card = 5) (h' : ∀ P Q R : (ℝ × ℝ), P ∈ S → Q ∈ S → R ∈ S → P ≠ Q → Q ≠ R → R ≠ P → ¬ collinear ({P, Q, R} : set (ℝ × ℝ))) :
  min_ratio S h h' = (1 + Real.sqrt 5) / 2 :=
sorry

end min_ratio_of_five_points_l663_663149


namespace conditional_probability_is_half_l663_663765

variables (people : Finset ℕ) (universities : Finset ℕ)
variables (event_A : Set (Finset (ℕ × ℕ))) (event_B : Set (Finset (ℕ × ℕ)))

-- Assuming the conditions of the problem
def three_people := people.card = 3
def three_universities := universities.card = 3
def event_A_def := event_A = {f | ∀ (a b c : ℕ) (u1 u2 u3 : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f = {(a, u1), (b, u2), (c, u3)} ∧ u1 ≠ u2 ∧ u2 ≠ u3 ∧ u1 ≠ u3}
def event_B_def := event_B = {f | ∀ (a b c : ℕ) (u1 u2 u3 : ℕ), f = {(a, u1), (b, u2), (c, u3)} ∧ u1 ≠ u2 ∧ u1 ≠ u3}

-- The theorem statement
theorem conditional_probability_is_half (h_three_people : three_people people)
    (h_three_universities : three_universities universities)
    (h_event_A : event_A_def event_A)
    (h_event_B : event_B_def event_B) :
    (measure_theory.conditional_prob event_A event_B) = 1 / 2 := sorry

end conditional_probability_is_half_l663_663765


namespace smallest_other_divisor_of_40_l663_663519

theorem smallest_other_divisor_of_40 (n : ℕ) (h₁ : n > 1) (h₂ : 40 % n = 0) (h₃ : n ≠ 8) :
  (∀ m : ℕ, m > 1 → 40 % m = 0 → m ≠ 8 → n ≤ m) → n = 5 :=
by 
  sorry

end smallest_other_divisor_of_40_l663_663519


namespace piecewise_function_solution_l663_663701

def f (x : ℝ) : ℝ :=
if x ≤ 0 then -x else x^2

theorem piecewise_function_solution (a : ℝ) : f a = 4 → (a = -4 ∨ a = 2) :=
by
  sorry

end piecewise_function_solution_l663_663701


namespace prob_5_10_l663_663204

def X (μ σ : ℝ) : (ℝ → ℝ) := sorry -- Placeholder for Normal distribution

axiom prob_mu_sigma (μ σ : ℝ) : ∀ (X : ℝ → ℝ), 
  (∀ (μ σ : ℝ), P(μ - σ < X ≤ μ + σ) = 0.6826 ∧ 
              P(μ - 2σ < X ≤ μ + 2σ) = 0.9544)

theorem prob_5_10 : P(5 < (X 0 5) ≤ 10) = 0.1359 :=
by 
  sorry

end prob_5_10_l663_663204


namespace choose_officers_from_six_l663_663991

/--
In how many ways can a President, a Vice-President, and a Secretary be chosen from a group of 6 people 
(assuming that all positions must be held by different individuals)?
-/
theorem choose_officers_from_six : (6 * 5 * 4 = 120) := 
by sorry

end choose_officers_from_six_l663_663991


namespace no_real_a_b_l663_663697

noncomputable def SetA (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ n : ℤ, p.1 = n ∧ p.2 = n * a + b}

noncomputable def SetB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}

noncomputable def SetC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 144}

theorem no_real_a_b :
  ¬ ∃ (a b : ℝ), (∃ p ∈ SetA a b, p ∈ SetB) ∧ (a, b) ∈ SetC :=
by
    sorry

end no_real_a_b_l663_663697


namespace max_height_of_pyramid_l663_663565

noncomputable def side_length : ℝ := 2 * Real.sqrt 3

noncomputable def height (a : ℝ) : ℝ := Real.sqrt (side_length^2 - (Real.sqrt 2 * a / 2)^2)

noncomputable def volume (a : ℝ) : ℝ := (1 / 3) * a^2 * height a

theorem max_height_of_pyramid : 
  ∀ a : ℝ, 
  let y := 12 * a^4 - (1 / 2) * a^6,
  (∀ a : ℝ, deriv y a = 0 → a = 0 ∨ a = 4) →
  height 4 = 2 := 
by
  sorry

end max_height_of_pyramid_l663_663565


namespace volume_of_water_cylinder_l663_663051

theorem volume_of_water_cylinder :
  let r := 5
  let h := 10
  let depth := 3
  let θ := Real.arccos (3 / 5)
  let sector_area := (2 * θ) / (2 * Real.pi) * Real.pi * r^2
  let triangle_area := r * (2 * r * Real.sin θ)
  let water_surface_area := sector_area - triangle_area
  let volume := h * water_surface_area
  volume = 232.6 * Real.pi - 160 :=
by
  sorry

end volume_of_water_cylinder_l663_663051


namespace fourth_vertex_not_in_third_quadrant_l663_663583

-- Define the points A, B, and C
def A : Point := ⟨2, 0⟩
def B : Point := ⟨-1/2, 0⟩
def C : Point := ⟨0, 1⟩

-- Define a predicate to check if a point is in the third quadrant
def is_in_third_quadrant (p : Point) : Prop :=
  p.1 < 0 ∧ p.2 < 0

-- The theorem stating the fourth vertex D cannot be in the third quadrant
theorem fourth_vertex_not_in_third_quadrant
  (D : Point)
  (hD1 : D = (B + (C - A))) ∨ (D = (A + (C - B))) ∨ (D = (C + (B - A))) :
  ¬ is_in_third_quadrant D :=
sorry

end fourth_vertex_not_in_third_quadrant_l663_663583


namespace players_odd_sum_probability_l663_663728

def tiles_numbered : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def players : fin 3 := sorry

-- Define the conditions of the problem
def game_conditions (tiles : set ℕ) (players : fin 3) :=
  (∀ p : fin 3, ∃ tiles_p : set ℕ, tiles_p ⊆ tiles ∧ tiles_p.card = 4 
    ∧ ((tiles_p.filter odd).card = 1 ∨ (tiles_p.filter odd).card = 3))

-- The probability of all three players having odd sums should translate to m+n
theorem players_odd_sum_probability (h : game_conditions tiles_numbered players) : ∃ m n : ℕ, (m + n = 25) ∧ (nat.gcd m n) = 1 ∧ (m = 4) ∧ (n = 21) :=
  sorry

end players_odd_sum_probability_l663_663728


namespace infinite_series_sum_eq_half_l663_663124

theorem infinite_series_sum_eq_half :
  (∑' n : ℕ, (if n ≠ 0 then (n^3 + n^2 - n) / (n + 3)! else 0)) = 1 / 2 := sorry

end infinite_series_sum_eq_half_l663_663124


namespace factorial_sum_count_l663_663951

theorem factorial_sum_count : 
  let factorials := [1, 1, 2, 6, 24, 120] in
  let distinct_factorial_sums := {n | ∃ (s: finset ℕ), s ⊆ finset.range 6 ∧ n = s.sum (λ i, factorials.get? i).get_or_else 0 } in
  let positive_factorial_sums := distinct_factorial_sums.filter (λ n, n > 0) in
  positive_factorial_sums.card = 39 := sorry

end factorial_sum_count_l663_663951


namespace monotonic_decreasing_intervals_l663_663601

noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x, x^α

theorem monotonic_decreasing_intervals (α : ℝ) (h : power_function α 2 = 1/2) :
  α = -1 →
  ∀ x : ℝ, (x < 0 ∨ x > 0) → (deriv (power_function α) x) < 0 :=
begin
  sorry
end

end monotonic_decreasing_intervals_l663_663601


namespace least_integer_value_l663_663538

-- Define the condition and then prove the statement
theorem least_integer_value (x : ℤ) (h : 3 * |x| - 2 > 13) : x = -6 :=
by
  sorry

end least_integer_value_l663_663538


namespace deposit_amount_correct_l663_663850

noncomputable def deposit_amount (initial_amount : ℝ) : ℝ :=
  let first_step := 0.30 * initial_amount
  let second_step := 0.25 * first_step
  0.20 * second_step

theorem deposit_amount_correct :
  deposit_amount 50000 = 750 :=
by
  sorry

end deposit_amount_correct_l663_663850


namespace televisions_selection_ways_l663_663151

noncomputable def combination (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

theorem televisions_selection_ways :
  let TypeA := 4
  let TypeB := 5
  let choosen := 3
  (∃ (n m : ℕ), n + m = choosen ∧ 1 ≤ n ∧ n ≤ TypeA ∧ 1 ≤ m ∧ m ≤ TypeB ∧
    combination TypeA n * combination TypeB m = 70) :=
by
  sorry

end televisions_selection_ways_l663_663151


namespace relationship_among_f_values_l663_663603

-- Definitions of p and q as roots of given equations
def is_root (f : ℝ → ℝ) (r : ℝ) : Prop := f(r) = 0

def f₁ (x : ℝ) : ℝ := 2^x + x + 2
def f₂ (x : ℝ) : ℝ := log x / log 2 + x + 2

-- Assume p and q are roots
axiom p : ℝ
axiom q : ℝ
axiom hp : is_root f₁ p
axiom hq : is_root f₂ q

-- Helper fact: p + q = -2
lemma pq_sum : p + q = -2 := sorry

-- Definition of function f
def f (x : ℝ) : ℝ := (x + p) * (x + q) + 2

-- The proof goal
theorem relationship_among_f_values : f 2 = f 0 ∧ f 0 < f 3 :=
by
  -- Given p + q = -2, rewrite f(x)
  have h_f : f x = x^2 - 2*x + p*q + 2 := sorry
  -- Calculate f(2), f(0), and f(3)
  have h_f2 : f 2 = 4 - 4 + pq + 2 := sorry
  have h_f0 : f 0 = pq + 2 := sorry
  have h_f3 : f 3 = 9 - 6 + pq + 2 := sorry
  -- Prove relationships
  exact sorry

end relationship_among_f_values_l663_663603


namespace yogurt_combinations_l663_663841

theorem yogurt_combinations (flavors toppings : ℕ) (h_flavors : flavors = 5) (h_toppings : toppings = 7) :
  (flavors * Nat.choose toppings 3) = 175 := by
  sorry

end yogurt_combinations_l663_663841


namespace stock_investment_decrease_l663_663439

theorem stock_investment_decrease :
  ∀ (x : ℝ),
  (1.30 * x) * 1.10 = 1.43 * x →
  ∃ d : ℝ, 1.43 * x * (1 - d) = x ∧ d = 0.3007 :=
begin
  intros x h1,
  use (1 - 1/1.43),
  split,
  { refine h1 },
  { norm_num }
end

end stock_investment_decrease_l663_663439


namespace infinite_bad_integers_l663_663166

theorem infinite_bad_integers (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ᶠ n in at_top, (¬(n^b + 1) ∣ (a^n + 1)) :=
by
  sorry

end infinite_bad_integers_l663_663166


namespace largest_n_for_factorial_condition_l663_663635

theorem largest_n_for_factorial_condition (n : ℕ) (a : ℕ) (h : n! = (n + a - 4)! / a!) : n = 3 :=
begin
  sorry
end

end largest_n_for_factorial_condition_l663_663635


namespace max_knights_between_knights_l663_663457

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l663_663457


namespace max_total_cut_length_l663_663414

/-- Given a 30 x 30 square board that is divided into 225 parts of equal area, 
prove that the maximum possible total length of the cuts is 1065. -/
theorem max_total_cut_length (A : ℝ) (hA : A = 900) (n : ℕ) (hn : n = 225) 
  (area_each_piece : ℝ) (h1 : area_each_piece = A / n) 
  (perimeter_each_piece : ℝ) (h2 : perimeter_each_piece = 10) : 
  (scenario : Prop) (h3 : scenario = ((225 * 10 - 4 * 30) / 2 = 1065)) : 
  1065 = (225 * 10 - 4 * 30) / 2 := 
by 
  rw [h3]
  sorry

end max_total_cut_length_l663_663414


namespace train_pass_station_time_l663_663397

-- Define the lengths of the train and station
def length_train : ℕ := 250
def length_station : ℕ := 200

-- Define the speed of the train in km/hour
def speed_kmh : ℕ := 36

-- Convert the speed to meters per second
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Calculate the total distance the train needs to cover
def total_distance : ℕ := length_train + length_station

-- Define the expected time to pass the station
def expected_time : ℕ := 45

-- State the theorem that needs to be proven
theorem train_pass_station_time :
  total_distance / speed_mps = expected_time := by
  sorry

end train_pass_station_time_l663_663397


namespace limestone_amount_l663_663806

theorem limestone_amount (L S : ℝ) (h1 : L + S = 100) (h2 : 3 * L + 5 * S = 425) : L = 37.5 :=
by
  -- Proof will go here
  sorry

end limestone_amount_l663_663806


namespace average_weight_of_16_boys_is_50_25_l663_663758

theorem average_weight_of_16_boys_is_50_25
  (W : ℝ)
  (h1 : 8 * 45.15 = 361.2)
  (h2 : 24 * 48.55 = 1165.2)
  (h3 : 16 * W + 361.2 = 1165.2) :
  W = 50.25 :=
sorry

end average_weight_of_16_boys_is_50_25_l663_663758


namespace triangle_shape_l663_663680

theorem triangle_shape
  (A B C : ℝ) -- Internal angles of triangle ABC
  (a b c : ℝ) -- Sides opposite to angles A, B, and C respectively
  (h1 : a * (Real.cos A) * (Real.cos B) + b * (Real.cos A) * (Real.cos A) = a * (Real.cos A)) :
  (A = Real.pi / 2) ∨ (A = C) :=
sorry

end triangle_shape_l663_663680


namespace area_of_sector_42_degrees_l663_663034

def radians (θ: ℝ) : ℝ := θ * (Real.pi / 180)
def area_of_sector (r: ℝ) (θ: ℝ) : ℝ := (θ / 360) * Real.pi * r^2

theorem area_of_sector_42_degrees :
  area_of_sector 12 42 = 16.8 * Real.pi :=
by
  sorry

end area_of_sector_42_degrees_l663_663034


namespace smaller_circle_tangent_circumference_l663_663985

theorem smaller_circle_tangent_circumference 
  (A B C : Type) 
  [metric_space A] [has_dist A] [metric_space B] [has_dist B] [metric_space C] [has_dist C]
  (triangle_ABC : triangle A B C)
  (center_arc_AC : B)
  (center_arc_BC : A)
  (arc_BC_length : real)
  (angle_α_at_C : real)
  (angle_C_is_90_deg : angle_α_at_C = π / 2)
  (arc_BC_16_units : arc_BC_length = 16) :
  ∃ (r2 : real), 2 * π * r2 = (32 * π - 256) / 5 := 
sorry

end smaller_circle_tangent_circumference_l663_663985


namespace extremum_at_one_g_monotonicity_intervals_value_range_combined_theorem_l663_663191

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x + a^2 / x
def g (x : ℝ) : ℝ := -x - log (-x)

-- The conditions and values for part 1
theorem extremum_at_one (a : ℝ) (h : f a 1 = x + (a^2 / x)) : x = 1 where f' a 1 = 0 :=
sorry

-- The monotonicity intervals of g(x)
theorem g_monotonicity_intervals (x : ℝ) : differentiable_on ℝ g {interval (-3) (-2)} :=
sorry

-- Given the specific range for part 2
theorem value_range (x1 x2 : ℝ) (a : ℝ) (h1 : x1 ∈ set.Icc 1 2) (h2 : x2 ∈ set.Icc -3 -2) 
  (range_a : -2 < a ∧ a < -1 - (1/2) * log 2) : 
  f a x1 ≥ g x2 :=
sorry

-- Combining preceding results
theorem combined_theorem (x1 x2 : ℝ) (a : ℝ) 
  (h1 : x1 ∈ set.Icc 1 2) (h2 : x2 ∈ set.Icc -3 -2)
  (cond1 : -2 < a) (cond2 : a < -1 - (1/2) * log 2)
  (h_extremum : f a 1 = x + (a^2 / x) ) -- extremum at x = 1
  (h_monotonicity : differentiable_on ℝ g {interval (-3) (-2)})
  : f a x1 ≥ g x2 :=
sorry

end extremum_at_one_g_monotonicity_intervals_value_range_combined_theorem_l663_663191


namespace max_knights_between_knights_l663_663459

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l663_663459


namespace class_president_is_wang_liang_l663_663661

open Classical

-- Definitions of roles and predictions
inductive Role
| president
| life_delegate
| study_delegate

def predictions (student : string) : Role → Prop :=
  match student with
  | "A" => λ r, r = Role.president ∧ Role.president = ZhangQiang ∨ r = Role.life_delegate ∧ Role.life_delegate = LiMing
  | "B" => λ r, r = Role.president ∧ Role.president = WangLiang ∨ r = Role.life_delegate ∧ Role.life_delegate = ZhangQiang
  | "C" => λ r, r = Role.president ∧ Role.president = LiMing ∨ r = Role.study_delegate ∧ Role.study_delegate = ZhangQiang
  | _   => λ _, False

-- Given students correctly guessed half of their predictions
def half_correct (student : string) : Prop :=
  match student with
  | "A" => (predictions "A" Role.president ∧ ¬ predictions "A" Role.life_delegate) ∨ (¬ predictions "A" Role.president ∧ predictions "A" Role.life_delegate)
  | "B" => (predictions "B" Role.president ∧ ¬ predictions "B" Role.life_delegate) ∨ (¬ predictions "B" Role.president ∧ predictions "B" Role.life_delegate)
  | "C" => (predictions "C" Role.president ∧ ¬ predictions "C" Role.study_delegate) ∨ (¬ predictions "C" Role.president ∧ predictions "C" Role.study_delegate)
  | _   => False

-- Main theorem: the announced class president is Wang Liang
theorem class_president_is_wang_liang
  (H1 : ZhangQiang ∈ class_council) (H2 : LiMing ∈ class_council) (H3 : WangLiang ∈ class_council)
  (H4 : half_correct "A") (H5 : half_correct "B") (H6 : half_correct "C") :
  Role.president = WangLiang :=
sorry

end class_president_is_wang_liang_l663_663661


namespace ratio_a_b_l663_663181

-- Let f(x) = a * sin x + b * cos x
def f (a b x : ℝ) := a * Real.sin x + b * Real.cos x

-- Given conditions
variables (a b : ℝ)
axiom max_value : f a b (π / 3) = 4
axiom max_eq_four : ∀ x, f a b x ≤ 4

-- Prove the value of a / b
theorem ratio_a_b : a / b = Real.sqrt 3 := sorry

end ratio_a_b_l663_663181


namespace cos_C_value_l663_663621

noncomputable def cos_C (A B C O I : Type) [triangle A B C] [circumcenter A O] [incenter A I]
  (h_angle_B : angle B = 45 * π / 180) (h_parallel : parallel OI BC) : ℝ :=
1 - real.sqrt 2 / 2

theorem cos_C_value (A B C O I : Type) [triangle A B C] [circumcenter A O] [incenter A I]
  (h_angle_B : angle B = 45 * π / 180) (h_parallel : parallel OI BC) : cos_C A B C O I h_angle_B h_parallel = 1 - real.sqrt 2 / 2 := 
sorry

end cos_C_value_l663_663621


namespace arithmetic_progression_subset_gt_1850_l663_663703

theorem arithmetic_progression_subset_gt_1850
  (n : ℕ)
  (h : ∀ (S : Finset ℕ), S.card = n → S ⊆ Finset.range 1989 → ∃ (T : Finset ℕ), T.card = 29 ∧ ∃ d : ℕ, T = Finset.range' d 29) : 
  n > 1850 := 
sorry

end arithmetic_progression_subset_gt_1850_l663_663703


namespace find_x_l663_663894

theorem find_x (x : ℝ) : 9999 * x = 724787425 ↔ x = 72487.5 := 
sorry

end find_x_l663_663894


namespace committee_size_l663_663551

theorem committee_size {B G : Nat} (hb : B = 5) (hg : G = 6) 
  (committees : Nat) (hcommittees : committees = 150) 
  (choose_boys : Nat := Nat.choose B 2) 
  (choose_girls : Nat := Nat.choose G 2) 
  (hchoose_boys : choose_boys = 10) 
  (hchoose_girls : choose_girls = 15) 
  (hcommittees_eq : committees = choose_boys * choose_girls) 
  (n : Nat) (h : n = 2 + 2) : n = 4 := 
by
  rw [h]
  rfl

end committee_size_l663_663551


namespace max_minus_min_value_f_on_1_3_is_9_over_4_l663_663929

def f (x : ℝ) : ℝ :=
  if x < 0 then x ^ 2 + 3 * x + 2
  else -(x ^ 2 + 3 * x + 2)

theorem max_minus_min_value_f_on_1_3_is_9_over_4 :
  let m := Real.sup (Set.image f (Set.Icc 1 3)),
      n := Real.inf (Set.image f (Set.Icc 1 3))
  in m - n = 9 / 4 := by
  sorry

end max_minus_min_value_f_on_1_3_is_9_over_4_l663_663929


namespace max_days_rainbow_lives_l663_663033

theorem max_days_rainbow_lives (n : ℕ) : 
  (∀ (days : ℕ → ℕ), 
    (∀ i : ℕ, i < days -> days i < n) → 
    (∀ i : ℕ, (i > 0) → days i ≠ days (i - 1)) → 
    (∀ i j k l : ℕ, i < j < k < l → days i = days k ∧ days j = days l → days i ≠ days j) → 
  days <= 2 * n - 1) :=
sorry

end max_days_rainbow_lives_l663_663033


namespace max_knights_adjacent_to_two_other_knights_l663_663481

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l663_663481


namespace product_seq_fraction_l663_663856

theorem product_seq_fraction :
  (∏ k in finset.range 667, (3 * k + 2 - 1) / (3 * k + 2)) = (1 / 1002) := 
sorry

end product_seq_fraction_l663_663856


namespace parabola_focus_is_centroid_l663_663586

-- Define the given points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -6)

-- Calculate the centroid of triangle ABC
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define the focus of the parabola and the centroid of the triangle
def focus_of_parabola (a : ℝ) : ℝ × ℝ := (a / 4, 0)
def centroid_ABC := centroid A B C

-- Statement of the proof problem
theorem parabola_focus_is_centroid :
  focus_of_parabola 8 = centroid_ABC := 
by
  sorry

end parabola_focus_is_centroid_l663_663586


namespace find_constant_a_l663_663112

-- Define the expression
def expr (x y : ℝ) : ℝ :=
  log x / (8 * log y) *
  (3 * log y) / (7 * log x) *
  (4 * log x) / (6 * log y) *
  (6 * log y) / (4 * log x) *
  (7 * log x) / (3 * log y)

-- Define the goal as a theorem
theorem find_constant_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ 1) (hyy : y ≠ 1) :
  expr x y = (1 / 2) * log x / log y :=
sorry

end find_constant_a_l663_663112


namespace limit_my_function_l663_663035

noncomputable def my_function (x : ℝ) : ℝ :=
(2 - real.exp(x^2))^(1 / real.log(1 + real.tan(π * x / 3)^2))

theorem limit_my_function : 
  tendsto (λ x : ℝ, (2 - real.exp(x^2))^(1 / real.log(1 + real.tan(π * x / 3)^2))) (𝓝 0) (𝓝 (real.exp (-9 / π^2))) :=
sorry

end limit_my_function_l663_663035


namespace bus_journey_distance_l663_663416

theorem bus_journey_distance (x : ℝ) (h1 : 0 ≤ x)
  (h2 : 0 ≤ 250 - x)
  (h3 : x / 40 + (250 - x) / 60 = 5.2) :
  x = 124 :=
sorry

end bus_journey_distance_l663_663416


namespace eight_boy_walk_min_distance_l663_663120

theorem eight_boy_walk_min_distance :
  (let radius := 50 in
   let num_boys := 8 in
   let angles := [90, 135] in
   let distance angle := 2 * radius * Real.sin (Real.ofInt angle / 2 * Real.pi / 180) in
   let distances := angles.map distance in
   let total_distance_per_boy := distances.sum * 2 in
   num_boys * total_distance_per_boy =
   800 * Real.sqrt 2 + 800 * Real.sqrt (2 + Real.sqrt 2)) := 
by
  sorry

end eight_boy_walk_min_distance_l663_663120


namespace ellipse_foci_distance_l663_663089

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end ellipse_foci_distance_l663_663089


namespace solve_digits_A_B_l663_663871

theorem solve_digits_A_B :
    ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 
    (A * (10 * A + B) = 100 * B + 10 * A + A) ∧ A = 8 ∧ B = 6 :=
by
  sorry

end solve_digits_A_B_l663_663871


namespace max_knights_between_other_knights_l663_663478

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l663_663478


namespace chocolate_type_probability_l663_663095

noncomputable def probability_same_type_chocolate (steps : ℕ) : ℝ :=
  -- Symmetric Binomial Distribution Property
  1/2 * (1 + (1/3 : ℝ) ^ steps)

theorem chocolate_type_probability : 
  probability_same_type_chocolate 100 = 1/2 * (1 + (1/3 : ℝ) ^ 100) := by
  sorry

end chocolate_type_probability_l663_663095


namespace cos_alpha_plus_pi_over_four_l663_663552

variable (α : ℝ)

noncomputable def cos_alpha (h1 : cos α = 12 / 13) (h2 : α ∈ (3 * Real.pi / 2, 2 * Real.pi)) : Real :=
  cos (α + Real.pi / 4)

theorem cos_alpha_plus_pi_over_four (h1 : cos α = 12 / 13) (h2 : α ∈ (3 * Real.pi / 2, 2 * Real.pi)) :
  cos (α + Real.pi / 4) = 17 * Real.sqrt 2 / 26 :=
sorry

end cos_alpha_plus_pi_over_four_l663_663552


namespace sum_of_fifth_powers_52070424_l663_663895

noncomputable def sum_of_fifth_powers (n : ℤ) : ℤ :=
  (n-1)^5 + n^5 + (n+1)^5

theorem sum_of_fifth_powers_52070424 :
  ∃ (n : ℤ), (n-1)^2 + n^2 + (n+1)^2 = 2450 ∧ sum_of_fifth_powers n = 52070424 :=
by
  sorry

end sum_of_fifth_powers_52070424_l663_663895


namespace complete_the_square_l663_663027

theorem complete_the_square (x : ℝ) : (x^2 + 2 * x - 1 = 0) -> ((x + 1)^2 = 2) :=
by
  intro h
  sorry

end complete_the_square_l663_663027


namespace min_value_x_y_l663_663594

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : 
  x + y ≥ 20 :=
sorry

end min_value_x_y_l663_663594


namespace frequency_third_group_l663_663671

-- Definitions based on conditions:
variables (S : ℕ → ℝ) (m : ℕ)
variables (h1 : m ≥ 3)
variables (h2 : S 1 + S 2 + S 3 = 1 / 4 * (S 4 + ∑ i in finset.range (m-3), S (i+4)))
variables (h3 : ∑ i in finset.range m, S (i + 1) = 1)
variables (h4 : S 1 = 1 / 20)
variables (h5 : S 1 + S 2 + S 3 = 3 * S 2)
variables (sample_size : ℕ := 120)

-- Goal: The frequency of the third group.
theorem frequency_third_group : 120 * S 3 = 10 :=
by
  sorry

end frequency_third_group_l663_663671


namespace number_of_pairs_satisfying_equation_l663_663141

theorem number_of_pairs_satisfying_equation :
  ∃ n : ℕ, n = 4998 ∧ (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → (x, y) ≠ (0, 0)) ∧
  (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → ((x + 6 * y) = (3 * 5) ^ a ∧ (x + y) = (3 ^ (50 - a) * 5 ^ (50 - b)) ∨
        (x + 6 * y) = -(3 * 5) ^ a ∧ (x + y) = -(3 ^ (50 - a) * 5 ^ (50 - b)) → (a + b = 50))) :=
sorry

end number_of_pairs_satisfying_equation_l663_663141


namespace divisor_is_163_l663_663032

theorem divisor_is_163 (D Q R d : ℕ) (hD : D = 12401) (hQ : Q = 76) (hR : R = 13) (h : D = d * Q + R) : d = 163 :=
by
  have h1 : 12401 = d * 76 + 13 := by rw [hQ, hR]; exact h
  have h2 : 12388 = d * 76 := by linarith
  have h3 : d = 163 := by linarith [h2, mul_comm]
  exact h3

end divisor_is_163_l663_663032


namespace symmetry_condition_l663_663343

theorem symmetry_condition (p q r s t u : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (yx_eq : ∀ x y, y = (p * x ^ 2 + q * x + r) / (s * x ^ 2 + t * x + u) ↔ x = (p * y ^ 2 + q * y + r) / (s * y ^ 2 + t * y + u)) :
  p = s ∧ q = t ∧ r = u :=
sorry

end symmetry_condition_l663_663343


namespace total_bugs_eaten_l663_663672

-- Define the conditions
def gecko_eats : ℕ := 12
def lizard_eats : ℕ := gecko_eats / 2
def frog_eats : ℕ := lizard_eats * 3
def toad_eats : ℕ := frog_eats + (frog_eats / 2)

-- Define the proof
theorem total_bugs_eaten : gecko_eats + lizard_eats + frog_eats + toad_eats = 63 :=
by
  sorry

end total_bugs_eaten_l663_663672


namespace repeating_decimal_equals_fraction_l663_663885

theorem repeating_decimal_equals_fraction : 
  let a := 58 / 100
  let r := 1 / 100
  let S := a / (1 - r)
  S = (58 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_equals_fraction_l663_663885


namespace choose_officers_from_six_l663_663992

/--
In how many ways can a President, a Vice-President, and a Secretary be chosen from a group of 6 people 
(assuming that all positions must be held by different individuals)?
-/
theorem choose_officers_from_six : (6 * 5 * 4 = 120) := 
by sorry

end choose_officers_from_six_l663_663992


namespace contrapositive_of_x_squared_gt_1_l663_663788

theorem contrapositive_of_x_squared_gt_1 (x : ℝ) (h : x ≤ 1) : x^2 ≤ 1 :=
sorry

end contrapositive_of_x_squared_gt_1_l663_663788


namespace categorize_numbers_l663_663529

def is_positive_integer (n : ℤ) : Prop := n > 0
def is_negative_integer (n : ℤ) : Prop := n < 0

def is_positive_fraction (r : ℚ) : Prop := (r > 0) ∧ (r.den ≠ 1)
def is_negative_fraction (r : ℚ) : Prop := (r < 0) ∧ (r.den ≠ 1)

noncomputable def problem_numbers : list ℚ := [-5, 1.2, -4.1666666666666665, 7, -5.4]
noncomputable def positive_integer : ℤ := 7
noncomputable def negative_integer : ℤ := -5
noncomputable def positive_fraction : ℚ := 1.2
noncomputable def negative_fractions : list ℚ := [-5.4, -4.1666666666666665]

theorem categorize_numbers :
  ∃ (pi : ℤ) (ni : ℤ) (pf : ℚ) (nf : list ℚ),
    pi = 7 ∧
    ni = -5 ∧
    pf = 1.2 ∧
    nf = [-5.4, -4.1666666666666665] ∧
    (is_positive_integer pi) ∧
    (is_negative_integer ni) ∧
    (is_positive_fraction pf) ∧
    (∀ n ∈ nf, is_negative_fraction n) :=
begin
  existsi positive_integer,
  existsi negative_integer,
  existsi positive_fraction,
  existsi negative_fractions,
  repeat { split },
  all_goals { try { trivial } },
  { norm_num_lemmas },
  { norm_num_lemmas },
  { split,
    { norm_num_lemmas },
    { norm_num_lemmas } },
  { intros,
    cases a,
    { norm_num_lemmas,
      split,
      { norm_num },
      { norm_num } },
    { cases a,
      { norm_num_lemmas,
        split,
        { norm_num },
        { norm_num } },
      { exfalso,
        simp at a } } }
end

end categorize_numbers_l663_663529


namespace arithmetic_sequence_property_l663_663670

variable {a : ℕ → ℕ}

-- Given condition in the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d c : ℕ, ∀ n : ℕ, a n = c + n * d

def condition (a : ℕ → ℕ) : Prop := a 4 + a 8 = 16

-- Problem statement
theorem arithmetic_sequence_property (a : ℕ → ℕ)
  (h_arith_seq : arithmetic_sequence a)
  (h_condition : condition a) :
  a 2 + a 6 + a 10 = 24 :=
sorry

end arithmetic_sequence_property_l663_663670


namespace amount_spent_on_berries_l663_663304

def amount_spent_on_apples : ℝ := 14.33
def amount_spent_on_peaches : ℝ := 9.31
def total_amount_spent : ℝ := 34.72

theorem amount_spent_on_berries : total_amount_spent - (amount_spent_on_apples + amount_spent_on_peaches) = 11.08 := by
  calc
  total_amount_spent - (amount_spent_on_apples + amount_spent_on_peaches) = 34.72 - (14.33 + 9.31) := by sorry
  ... = 11.08 : by sorry

end amount_spent_on_berries_l663_663304


namespace parallel_vectors_l663_663558

variable (x : ℝ)
def v_a : ℝ × ℝ := (2, x)
def v_b : ℝ × ℝ := (4, -1)

theorem parallel_vectors (h : v_a x ∥ v_b) : x = -1 / 2 := by
  sorry

end parallel_vectors_l663_663558


namespace weight_of_new_person_l663_663403

theorem weight_of_new_person (average_increase : ℕ) (group_size : ℕ) (old_weight : ℕ) : 
  new_weight = old_weight + group_size * average_increase :=
by
  -- Definitions from the conditions
  let average_increase := 3
  let group_size := 8
  let old_weight := 65
  -- Expected outcome based on solution
  have : new_weight = 89 := by sorry
  -- Final theorem
  exact this

end weight_of_new_person_l663_663403


namespace probability_of_drawing_two_pairs_and_one_different_l663_663200

noncomputable def total_ways_to_draw_socks := Nat.choose 12 5

noncomputable def favorable_scenarios := (Nat.choose 3 2 * Nat.choose 2 1) * 
                                         (Nat.choose 3 2 * Nat.choose 3 2 * Nat.choose 1 1)

noncomputable def probability_of_event := (favorable_scenarios : ℚ) / total_ways_to_draw_socks

theorem probability_of_drawing_two_pairs_and_one_different :
  probability_of_event = 3 / 44 :=
by
  -- Definitions and given data
  let total_ways := Nat.choose 12 5
  let favorable := (Nat.choose 3 2 * Nat.choose 2 1) * 
                   (Nat.choose 3 2 * Nat.choose 3 2 * Nat.choose 1 1)

  -- Calculating the probability
  have prob := (favorable : ℚ) / total_ways
  show prob = 3 / 44 from
  sorry

end probability_of_drawing_two_pairs_and_one_different_l663_663200


namespace consecutive_integers_sum_l663_663237

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l663_663237


namespace find_g_g_half_l663_663557

def g (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem find_g_g_half : g (g (1 / 2)) = 1 / 2 := by
  sorry

end find_g_g_half_l663_663557


namespace jerry_total_cost_l663_663008

-- Definition of the costs and quantities
def cost_color : ℕ := 32
def cost_bw : ℕ := 27
def num_color : ℕ := 3
def num_bw : ℕ := 1

-- Definition of the total cost
def total_cost : ℕ := (cost_color * num_color) + (cost_bw * num_bw)

-- The theorem that needs to be proved
theorem jerry_total_cost : total_cost = 123 :=
by
  sorry

end jerry_total_cost_l663_663008


namespace speed_of_faster_train_l663_663406

theorem speed_of_faster_train 
  (length_train : ℝ)
  (cross_time_seconds : ℝ)
  (speed_ratio : ℝ)
  (length_train = 100) 
  (cross_time_seconds = 8)
  (speed_ratio = 2)
  : 
  let slower_speed := 25 / 3 in 
  let faster_speed_m_per_s := 2 * slower_speed in
  let faster_speed_km_per_hr := faster_speed_m_per_s * 3.6 in
  faster_speed_km_per_hr = 60 
:= sorry

end speed_of_faster_train_l663_663406


namespace projection_a_projection_b_l663_663164

variables {Point : Type} [plane : Plane Point]

structure Prism (A B C D A₁ B₁ C₁ D₁ : Point) :=
(base_trapezoid : is_trapezoid A B C D)
(parallel_AD_BC : is_parallel AD BC)
(ad_eq_2bc : AD.length = 2 * BC.length)

def divides (C₁ C P : Point) (r : ℝ) : Prop :=
  ∃ k : ℝ, k = r ∧ P = k • C₁ + (1 - k) • C

axiom divides_C₁C : divides C₁ C P (1/3)

theorem projection_a {A B C D A₁ B₁ C₁ D₁ P : Point}
  (h : Prism A B C D A₁ B₁ C₁ D₁) (h_div : divides C₁ C P (1/3)) :
  project_section A₁ D₁ C B = A₁ B := sorry

theorem projection_b {A D₁ P : Point} :
  project_triangle A D₁ P = A D₁ := sorry

end projection_a_projection_b_l663_663164


namespace compute_DE_l663_663281

noncomputable def triangle := ℕ -- For simplicity, an Artificial Placeholder

variables {a b c : triangle} {A B C: ℝ} -- Vertices
variables {D E P Q : ℝ} -- Points
variables {omega : ℝ} -- Circumcircle (ω)
variables (circumcircle_property : ω = omega) -- Placeholder Circumcircle Condition
variables (altitude_D : D = some (altitude from B to AC)) -- Altitude conditions as placeholders
variables (altitude_E : E = some (altitude from C to AB))
variables {PD QE AP BC : ℝ}
variables (condition1 : PD = 3)
variables (condition2 : QE = 2)
variables (parallel_condition : AP ∥ BC)

theorem compute_DE (h : AP ∥ BC) (PD_EQ_3 : PD = 3) (QE_EQ_2 : QE = 2) : DE = sqrt(23) :=
sorry

end compute_DE_l663_663281


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663229

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663229


namespace distance_between_foci_of_ellipse_l663_663091

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end distance_between_foci_of_ellipse_l663_663091


namespace common_sum_of_matrix_l663_663332

theorem common_sum_of_matrix : 
  ∃ m : Matrix (Fin 6) (Fin 6) ℤ, 
    (∀ i : Fin 6, (∑ j, m i j) = 7) ∧ 
    (∀ j : Fin 6, (∑ i, m i j) = 7) ∧ 
    ((∑ k, m k k) = 7) ∧ 
    ((∑ k, m k (⟨5 - k, sorry⟩)) = 7) ∧ 
    (∀ i j, m i j ∈ Finset.Icc (-12 : ℤ) 15) :=
by
  sorry

end common_sum_of_matrix_l663_663332


namespace polynomial_integral_factorization_l663_663877

theorem polynomial_integral_factorization (k : ℤ) :
  (∃ (K : ℕ → polynomial ℤ), ∀ n : ℕ, n ≥ 3 → (K n = polynomial.X^(n+1) + C (↑k) * polynomial.X^n - C 870 * polynomial.X^2 + C 1945 * polynomial.X + C 1995) 
  ∧ (∃ (A B : polynomial ℤ), K n = A * B)) ↔ k = -3071 ∨ k = -821 ∨ k = 821 := 
sorry

end polynomial_integral_factorization_l663_663877


namespace fraction_simplification_l663_663497

theorem fraction_simplification :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16) /
  (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by
  sorry

end fraction_simplification_l663_663497


namespace knights_max_seated_between_knights_l663_663460

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l663_663460


namespace quadratic_equation_even_coefficient_l663_663774

-- Define the predicate for a rational root
def has_rational_root (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), (q ≠ 0) ∧ (p.gcd q = 1) ∧ (a * p^2 + b * p * q + c * q^2 = 0)

-- Define the predicate for at least one being even
def at_least_one_even (a b c : ℤ) : Prop :=
  (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0)

theorem quadratic_equation_even_coefficient 
  (a b c : ℤ) (h_non_zero : a ≠ 0) (h_rational_root : has_rational_root a b c) :
  at_least_one_even a b c :=
sorry

end quadratic_equation_even_coefficient_l663_663774


namespace area_triangle_LOM_l663_663664

theorem area_triangle_LOM
  (ABC : Triangle)
  (α β γ : ℝ)
  (L O M : Point)
  (A B C : Point)
  (h1 : α = β - γ)
  (h2 : β = 2 * γ)
  (h3 : α + β + γ = 180)
  (h4 : ABC.area = 32)
  (h5 : angle_bisector A ∩ circumcircle ABC = L)
  (h6 : angle_bisector B ∩ circumcircle ABC = O)
  (h7 : angle_bisector C ∩ circumcircle ABC = M) :
  area Triangle.mk L O M = 44 :=
sorry

end area_triangle_LOM_l663_663664


namespace choose_officers_from_six_l663_663993

/--
In how many ways can a President, a Vice-President, and a Secretary be chosen from a group of 6 people 
(assuming that all positions must be held by different individuals)?
-/
theorem choose_officers_from_six : (6 * 5 * 4 = 120) := 
by sorry

end choose_officers_from_six_l663_663993


namespace correct_inferences_l663_663953

/--
If "p or q" is true only if "¬r" is true, then:
1. p or q implies ¬r
2. ¬r implies p
3. r implies ¬(p or q)
4. ¬p and ¬q implies r
The number of correct inferences is 1.
-/
theorem correct_inferences (p q r : Prop) (h : (p ∨ q) → ¬r) :
  ((¬r → (p ∨ q)) ∧ ¬((p ∨ q) → ¬r) ∧ ¬(r → ¬(p ∨ q)) ∧ (¬p ∧ ¬q → r)) ↔ 1 := sorry

end correct_inferences_l663_663953


namespace power_function_monotonic_increasing_iff_l663_663972

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x ≤ f y

theorem power_function_monotonic_increasing_iff (m : ℝ) :
  ((m^2 - 2*m - 2 > 0) ∧ (m^2 - 4*m + 1 > 0)) → m = -1 :=
by 
  intro h
  let h1 : (m^2 - 2*m - 2 > 0) := h.1
  let h2 : (m^2 - 4*m + 1 > 0) := h.2
  sorry

end power_function_monotonic_increasing_iff_l663_663972


namespace max_score_per_student_l663_663686

theorem max_score_per_student (score_tests : ℕ → ℕ) (avg_score_tests_lt_8 : ℕ) (combined_score_two_tests : ℕ) : (∀ i, 1 ≤ i ∧ i ≤ 8 → score_tests i ≤ 100) ∧ avg_score_tests_lt_8 = 70 ∧ combined_score_two_tests = 290 →
  ∃ max_score : ℕ, max_score = 145 := 
by
  sorry

end max_score_per_student_l663_663686


namespace probability_of_playing_A_or_B_l663_663597

theorem probability_of_playing_A_or_B (songs : Finset ℕ) (h : songs.card = 5) :
  let (total_pairs : ℕ) := (Finset.card (Finset.image2 (λ x y, (x, y)) songs songs)) / 2,
      (favorable_pairs : ℕ) := 7
  in total_pairs = 10 → ((favorable_pairs : ℚ) / (total_pairs : ℚ)) = 7 / 10 :=
sorry

end probability_of_playing_A_or_B_l663_663597


namespace range_of_theta_div_4_l663_663173

noncomputable def theta_third_quadrant (k : ℤ) (θ : ℝ) : Prop :=
  (2 * k * Real.pi + Real.pi < θ) ∧ (θ < 2 * k * Real.pi + 3 * Real.pi / 2)

noncomputable def sin_lt_cos (θ : ℝ) : Prop :=
  Real.sin (θ / 4) < Real.cos (θ / 4)

theorem range_of_theta_div_4 (k : ℤ) (θ : ℝ) :
  theta_third_quadrant k θ →
  sin_lt_cos θ →
  (2 * k * Real.pi + 5 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 11 * Real.pi / 8) ∨
  (2 * k * Real.pi + 7 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 15 * Real.pi / 8) := 
  by
    sorry

end range_of_theta_div_4_l663_663173


namespace total_cartridge_cost_l663_663010

theorem total_cartridge_cost:
  ∀ (bw_cartridge_cost color_cartridge_cost bw_quantity color_quantity : ℕ),
  bw_cartridge_cost = 27 →
  color_cartridge_cost = 32 →
  bw_quantity = 1 →
  color_quantity = 3 →
  bw_quantity * bw_cartridge_cost + color_quantity * color_cartridge_cost = 123 :=
begin
  intros bw_cartridge_cost color_cartridge_cost bw_quantity color_quantity,
  intros h_bw_cost h_color_cost h_bw_qty h_color_qty,
  rw [h_bw_cost, h_color_cost, h_bw_qty, h_color_qty],
  norm_num,
end

end total_cartridge_cost_l663_663010


namespace solve_for_x_l663_663320

theorem solve_for_x : ∃ x : ℤ, 24 - 5 = 3 + x ∧ x = 16 :=
by
  sorry

end solve_for_x_l663_663320


namespace exp_ineq_of_r_gt_one_l663_663645

theorem exp_ineq_of_r_gt_one {x r : ℝ} (hx : x > 0) (hr : r > 1) : (1 + x)^r > 1 + r * x :=
by
  sorry

end exp_ineq_of_r_gt_one_l663_663645


namespace ae_is_half_l663_663326

theorem ae_is_half 
    (x : ℝ) 
    (h₁ : ∀ E F : ℝ, AE = x ∧ CF = 1 / 2)
    (h₂ : fold_along_DE_DF_onto_BD : ∀ D E F G : ℝ, sides_AD_and_CD_coincide_and_lie_on_BD (AD CD BD)) :
  x = 1 / 2 :=
by
  sorry

end ae_is_half_l663_663326


namespace new_oranges_added_l663_663839
-- Import the necessary library

-- Define the constants and conditions
def initial_oranges : ℕ := 5
def thrown_away : ℕ := 2
def total_oranges_now : ℕ := 31

-- Define new_oranges as the variable we want to prove
def new_oranges (x : ℕ) : Prop := x = 28

-- The theorem to prove how many new oranges were added
theorem new_oranges_added :
  ∃ (x : ℕ), new_oranges x ∧ total_oranges_now = initial_oranges - thrown_away + x :=
by
  sorry

end new_oranges_added_l663_663839


namespace parallel_and_through_point_l663_663747

-- Defining the given line
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Defining the target line passing through the point (0, 4)
def line2 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Define the point (0, 4)
def point : ℝ × ℝ := (0, 4)

-- Prove that line2 passes through the point (0, 4) and is parallel to line1
theorem parallel_and_through_point (x y : ℝ) 
  (h1 : line1 x y) 
  : line2 (point.fst) (point.snd) := by
  sorry

end parallel_and_through_point_l663_663747


namespace peter_age_l663_663364

theorem peter_age (P Q : ℕ) (h1 : Q - P = P / 2) (h2 : P + Q = 35) : Q = 21 :=
  sorry

end peter_age_l663_663364


namespace tickets_left_l663_663037

-- Definitions for the conditions given in the problem
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- The main proof statement to verify
theorem tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by
  sorry

end tickets_left_l663_663037


namespace probability_53_sundays_in_leap_year_l663_663143

-- Define the conditions
def num_days_in_leap_year : ℕ := 366
def num_weeks_in_leap_year : ℕ := 52
def extra_days_in_leap_year : ℕ := 2
def num_combinations : ℕ := 7
def num_sunday_combinations : ℕ := 2

-- Define the problem statement
theorem probability_53_sundays_in_leap_year (hdays : num_days_in_leap_year = 52 * 7 + extra_days_in_leap_year) :
  (num_sunday_combinations / num_combinations : ℚ) = 2 / 7 :=
by
  sorry

end probability_53_sundays_in_leap_year_l663_663143


namespace midpoint_correct_parallel_line_correct_l663_663585

variable (A B : ℝ × ℝ)
variable (l : ℝ → ℝ → Prop)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 )

def line_eq (a b c : ℝ) : ℝ → ℝ → Prop :=
  λ x y, a * x + b * y + c = 0

def parallel_line_eq (a b c d e f: ℝ) : ℝ → ℝ → Prop :=
  (λ x y, a * x + b * y + c = 0) ∧ (λ x y, d * x + e * y + f = 0)

theorem midpoint_correct :
  A = (-1, 1) →
  B = (1, 3) →
  midpoint A B = (0, 2) := 
by
  sorry

theorem parallel_line_correct :
  l = line_eq 1 2 3 →
  B = (1, 3) →
  ∃ f, (∀ x y, l x y ↔ x + 2 * y + 3 = 0) ∧ (∀ x y, line_eq 1 2 f x y ↔ x + 2 * y - 7 = 0) :=
by
  sorry

end midpoint_correct_parallel_line_correct_l663_663585


namespace sum_of_consecutive_integers_l663_663223

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l663_663223


namespace frac_calc_l663_663857

theorem frac_calc : (2 / 9) * (5 / 11) + 1 / 3 = 43 / 99 :=
by sorry

end frac_calc_l663_663857


namespace inequality_solution_l663_663512

theorem inequality_solution (x: ℝ) (h1: x ≠ -1) (h2: x ≠ 0) :
  (x-2)/(x+1) + (x-3)/(3*x) ≥ 2 ↔ x ∈ Set.Iic (-3) ∪ Set.Icc (-1) (-1/2) :=
by
  sorry

end inequality_solution_l663_663512


namespace problem_1_problem_2_l663_663615

-- First part of the problem about the zeros of h(x)
def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x + 1 

def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1 - Real.log x

def h (k : ℝ) (x : ℝ) : ℝ :=
if x < 1 then f x else g k x

theorem problem_1 (k : ℝ) (hk : k < 0) :
  (k < -1 → ∃! x : ℝ, h k x = 0) ∧ 
  (-1 ≤ k ∧ k < 0 → ∃ x1 y1 : ℝ, x1 ≠ y1 ∧ h k x1 = 0 ∧ h k y1 = 0) := 
sorry

-- Second part of the problem about the range of a
def P (a : ℝ) : ℝ × ℝ := (a, -4)

def tangent_eq (t : ℝ) (a : ℝ) : ℝ := 
  let slope := 6 * t ^ 2 - 6 * t in
  slope * a - 6 * t ^ 2 * t + 6 * t * a - 4 - (2 * t ^ 3 - 3 * t + 1)

def H (t : ℝ) (a : ℝ) : ℝ := 4 * t ^ 3 - 3 * t ^ 2 - 6 * t ^ 2 * a + 6 * t * a - 5

theorem problem_2 (a : ℝ) :
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ H t1 a = 0 ∧ H t2 a = 0 ∧ H t3 a = 0) ↔ (a > 7 / 2 ∨ a < -1) :=
sorry

end problem_1_problem_2_l663_663615


namespace convex_quadrilaterals_fixed_point_l663_663284

theorem convex_quadrilaterals_fixed_point 
  (n : ℤ) 
  (h_n : n ≥ 16)
  (G : set (ℤ × ℤ)) 
  (h_G : G = {p | p.1 ∈ finset.range (n + 1) ∧ p.2 ∈ finset.range (n + 1)})
  (A : set (ℤ × ℤ)) 
  (h_A : A ⊆ G ∧ A.card ≥ 4 * n * int.of_nat (int.sqrt n)) :
  ∃ Q : finset (finset (ℤ × ℤ)), Q.card ≥ n^2 ∧ 
    (∀ q ∈ Q, q.card = 4 ∧ 
     ∃ P : ℤ × ℤ, ∀ a b ∈ q, a + b = 2 * P) :=
begin
  sorry
end

end convex_quadrilaterals_fixed_point_l663_663284


namespace consecutive_sum_l663_663214

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l663_663214


namespace not_or_false_implies_or_true_l663_663246

variable (p q : Prop)

theorem not_or_false_implies_or_true (h : ¬(p ∨ q) = False) : p ∨ q :=
by
  sorry

end not_or_false_implies_or_true_l663_663246


namespace max_knights_adjacent_to_two_other_knights_l663_663483

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l663_663483


namespace abs_eq_implies_y_eq_half_l663_663386

theorem abs_eq_implies_y_eq_half (y : ℝ) (h : |y - 3| = |y + 2|) : y = 1 / 2 :=
by 
  sorry

end abs_eq_implies_y_eq_half_l663_663386


namespace coefficient_x3_in_expansion_l663_663535

theorem coefficient_x3_in_expansion :
  let f := (λ x : ℝ, (sqrt x - 2 / x + 1)^7) in
  coefficient_of_x3 f = 7 :=
sorry

end coefficient_x3_in_expansion_l663_663535


namespace degree_d_l663_663107

def polynomial (α : Type*) [comm_ring α] := add_comm_group.polynomial α
open_locale polynomial

variable {α : Type*} [comm_ring α]

theorem degree_d (f d q r : polynomial α) (h_f : f.degree = 15)
  (h_q : q.degree = 7) (h_r : r = polynomial.C 4 * polynomial.X ^ 4 +
    polynomial.C 5 * polynomial.X ^ 3 - polynomial.C 2 * polynomial.X ^ 2 +
    polynomial.C 3 * polynomial.X + polynomial.C 20):
  (f = d * q + r) → d.degree = 8 :=
by {
  sorry
}

end degree_d_l663_663107


namespace anthony_transactions_more_percentage_l663_663306

def transactions (Mabel Anthony Cal Jade : ℕ) : Prop := 
  Mabel = 90 ∧ 
  Jade = 84 ∧ 
  Jade = Cal + 18 ∧ 
  Cal = (2 * Anthony) / 3 ∧ 
  Anthony = Mabel + (Mabel * 10 / 100)

theorem anthony_transactions_more_percentage (Mabel Anthony Cal Jade : ℕ) 
    (h : transactions Mabel Anthony Cal Jade) : 
  (Anthony = Mabel + (Mabel * 10 / 100)) :=
by 
  sorry

end anthony_transactions_more_percentage_l663_663306


namespace arithmetic_prog_contains_sixth_power_l663_663724

theorem arithmetic_prog_contains_sixth_power {α : Type*} [linear_ordered_field α]
  (a d : α) (h : d ≠ 0) (n : ℕ) : 
  (∀ k: ℕ, ∃ x: α, x^2 = a + k * d ∧ ∃ y: α, y^3 = a + k * d) → 
  ∃ z: α, ∃ k: ℕ, z^6 = a + k * d :=
by
  sorry

end arithmetic_prog_contains_sixth_power_l663_663724


namespace max_knights_seated_next_to_two_knights_l663_663488

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l663_663488


namespace borrowed_sheets_10_l663_663708

-- Definitions and conditions
def size_of_notebook := 60
def sheets_of_notebook := 30
def remaining_pages_mean := 25

-- Borrowing sheets definition
def borrowed_sheets (total_pages : ℕ) (total_sheets : ℕ) (mean_remaining : ℕ) : ℕ :=
  let sum_pages := (total_pages * (total_pages + 1)) / 2 -- sum of pages 1 to 60
  let n := 2 in -- pages per sheet
  let num_sheets := total_pages / n in 
  let k := (total_pages - mean_remaining * (total_sheets - 2 * 50)) / (4 * mean_remaining)
  k.to_nat

theorem borrowed_sheets_10 :
  borrowed_sheets size_of_notebook sheets_of_notebook remaining_pages_mean = 10 :=
by sorry

end borrowed_sheets_10_l663_663708


namespace race_track_cost_l663_663119

def toy_car_cost : ℝ := 0.95
def num_toy_cars : ℕ := 4
def total_money : ℝ := 17.80
def money_left : ℝ := 8.00

theorem race_track_cost :
  total_money - num_toy_cars * toy_car_cost - money_left = 6.00 :=
by
  sorry

end race_track_cost_l663_663119


namespace sum_of_consecutive_integers_l663_663218

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l663_663218


namespace patch_area_difference_l663_663075

theorem patch_area_difference :
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  area_difference = 100 := 
by
  -- Definitions
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  -- Proof (intentionally left as sorry)
  -- Lean should be able to use the initial definitions to verify the theorem statement.
  sorry

end patch_area_difference_l663_663075


namespace four_p_squared_plus_one_l663_663432

theorem four_p_squared_plus_one (p : ℕ) (hp_prime : prime p) (hp_ne_three : p ≠ 3) : 
  ∃ a b c : ℕ, 4 * p ^ 2 + 1 = a ^ 2 + b ^ 2 + c ^ 2 :=
  sorry

end four_p_squared_plus_one_l663_663432


namespace water_drunk_last_mile_l663_663627

theorem water_drunk_last_mile 
(full_canteen : ℕ) (remaining_water : ℕ) (total_miles : ℕ)
(total_time : ℕ) (leak_rate : ℕ) (first_6mile_rate : ℕ → ℕ) 
(h1 : full_canteen = 11) (h2 : remaining_water = 2) 
(h3 : total_miles = 7) (h4 : total_time = 3)
(h5 : leak_rate = 1) 
(h6 : ∀ x, first_6mile_rate x = 1/2) :

(drunk_last_mile : ℕ) :=
    drunk_last_mile = 3 :=
by
    sorry

end water_drunk_last_mile_l663_663627


namespace common_chord_bisects_AC_l663_663587

open_locale classical
noncomputable theory

/-- Points A, B, C, D are collinear and satisfy BC = 2AB and CD = AC -/
variables {A B C D : ℝ} [linear_order ℝ]
variable (h1 : B = A + B)
variable (h2 : C = B + B)
variable (h3 : D = C + D)

/-- The common chord of any circle passing through points A, C and B, D bisects segment AC -/
theorem common_chord_bisects_AC 
  (A B C D : ℝ) 
  (h1 : B = 2 * A)
  (h2 : C - B = A)
  (h3 : D - C = A) :
  is_midpoint (midpoint A C) :=
begin
  sorry
end

end common_chord_bisects_AC_l663_663587


namespace area_of_figure_l663_663868

open Real

theorem area_of_figure : 
  (∀ (x y : ℝ), abs (9 + 8 * y - x^2 - y^2) + abs (8 * y) = 16 * y + 9 - x^2 - y^2 → (y ≥ 0) ∧ (9 + 8 * y - x^2 - y^2 ≥ 0)) →
  let area := 25 * π - 25 * arcsin 0.6 + 12
in area = 25 * π - 25 * arcsin 0.6 + 12 :=
by {
  sorry
}

end area_of_figure_l663_663868


namespace min_value_of_quadratic_l663_663653

theorem min_value_of_quadratic (m : ℝ) (x : ℝ) (hx1 : 3 ≤ x) (hx2 : x < 4) (h : x^2 - 4 * x ≥ m) : 
  m ≤ -3 :=
sorry

end min_value_of_quadratic_l663_663653


namespace radius_of_circle_l663_663514

theorem radius_of_circle :
  ∀ (x y : ℝ), (2 * x^2 + 2 * y^2 - 10 = 2 * x + 4 * y) →
  (∃ r : ℝ, r = real.sqrt (13 / 2)) :=
by
  sorry

end radius_of_circle_l663_663514


namespace sock_ratio_l663_663331

theorem sock_ratio (g r y : ℕ) (h1 : g = 6) (h2 : 3ary + 6y + 15 = 1.8 * (18y + ry + 15)) : g / r = 6 / 23 :=
by {
  sorry,
}

end sock_ratio_l663_663331


namespace new_value_of_y_l663_663729

-- Definitions based on the conditions
def rectangle_length : ℝ := 20
def rectangle_width : ℝ := 10
def rectangle_area : ℝ := rectangle_length * rectangle_width

def square_side_length : ℝ := Math.sqrt rectangle_area
def y : ℝ := square_side_length / 4

-- Theorem to prove that y = 2.5 * sqrt(2)
theorem new_value_of_y :
  y = 2.5 * Real.sqrt 2 :=
begin
  sorry -- Proof is omitted
end

end new_value_of_y_l663_663729


namespace admission_price_for_children_l663_663843

theorem admission_price_for_children 
  (admission_price_adult : ℕ)
  (total_persons : ℕ)
  (total_amount_dollars : ℕ)
  (children_attended : ℕ)
  (admission_price_children : ℕ)
  (h1 : admission_price_adult = 60)
  (h2 : total_persons = 280)
  (h3 : total_amount_dollars = 140)
  (h4 : children_attended = 80)
  (h5 : (total_persons - children_attended) * admission_price_adult + children_attended * admission_price_children = total_amount_dollars * 100)
  : admission_price_children = 25 := 
by 
  sorry

end admission_price_for_children_l663_663843


namespace maximum_triangles_in_right_angle_triangle_l663_663307

-- Definition of grid size and right-angled triangle on graph paper
def grid_size : Nat := 7

-- Definition of the vertices of the right-angled triangle
def vertices : List (Nat × Nat) := [(0,0), (grid_size,0), (0,grid_size)]

-- Total number of unique triangles that can be identified
theorem maximum_triangles_in_right_angle_triangle (grid_size : Nat) (vertices : List (Nat × Nat)) : 
  Nat :=
  if vertices = [(0,0), (grid_size,0), (0,grid_size)] then 28 else 0

end maximum_triangles_in_right_angle_triangle_l663_663307


namespace total_balloons_l663_663015

-- Define variables for the number of balloons Tom and Sara have
variables (tom_balloons : Nat) (sara_balloons : Nat)

-- Given conditions
def tom_has_9_balloons : tom_balloons = 9 := sorry
def sara_has_8_balloons : sara_balloons = 8 := sorry

-- Proof statement
theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by 
  rw [tom_has_9_balloons, sara_has_8_balloons]
  sorry

end total_balloons_l663_663015


namespace inequality_abc_l663_663555

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (b * c / a) + (a * c / b) + (a * b / c) ≥ a + b + c := 
  sorry

end inequality_abc_l663_663555


namespace evaluate_rs_minus_pq_l663_663640

theorem evaluate_rs_minus_pq :
  let α β : ℝ
  (h1 : sin α + sin β = 3)
  (h2 : sin α * sin β = 1)
  (h3 : cos α + cos β = 2)
  (h4 : cos α * cos β = 1) :
  let p := sin α + sin β
  let q := sin α * sin β
  let r := cos α + cos β
  let s := cos α * cos β
  rs - pq = -1 :=
sorry

end evaluate_rs_minus_pq_l663_663640


namespace volume_of_pyramid_l663_663336

theorem volume_of_pyramid (r α β : ℝ) (h_r_pos : 0 < r) (h_α_acute : 0 < α ∧ α < π/2) (h_β_acute : 0 < β ∧ β < π/2) :
  let V := (r^3 * sqrt 2 * tan β * cot (π/4 - α/2) * cot (α/2)) / (12 * sin (π/4 - α/2) * sin (α/2)) in
  volume r α β = V :=
begin
  sorry
end

-- Define the function to calculate volume for pyramid
noncomputable def volume (r α β : ℝ) : ℝ :=
  (r^3 * sqrt 2 * Real.tan β * Real.cot (π/4 - α/2) * Real.cot (α/2)) / (12 * Real.sin (π/4 - α/2) * Real.sin (α/2))

end volume_of_pyramid_l663_663336


namespace applicants_hired_probability_same_l663_663362

-- Definition of the problem setup
def n_job_openings (n : ℕ) : Prop := n ≥ 3
def qualified (i j : ℕ) : Prop := i ≥ j
def hired_highest_ranking (i j : ℕ) (jobs : Finset ℕ): Prop :=
  j ∈ jobs ∧ qualified i j ∧ (∀ j' ∈ jobs, j' < j → qualified i j')

-- Theorem statement to prove the main result
theorem applicants_hired_probability_same (n : ℕ) (jobs : Finset ℕ) (applicants : Finset ℕ) :
  (n_job_openings n) →
  (∀ i ∈ applicants, ∃ j ∈ jobs, qualified i j) →
  (∀ i ∈ applicants, ∃! j ∈ jobs, hired_highest_ranking i j jobs) →
  (∀ A : List ℕ, Permutation A (List.range n)) →
  ∃ p : ℚ, 
    (Prob (λ A, ∃ j ∈ jobs, A.last = n ∧ hired_highest_ranking n j jobs) = p) ∧ 
    (Prob (λ A, ∃ j ∈ jobs, A.last = n-1 ∧ hired_highest_ranking (n-1) j jobs) = p) := sorry

end applicants_hired_probability_same_l663_663362


namespace sqrt2_sqrt3_sqrt5_not_geometric_sequence_l663_663717

theorem sqrt2_sqrt3_sqrt5_not_geometric_sequence :
  ¬ (∃ r, sqrt(3:ℝ) = sqrt(2:ℝ) * r ∧ sqrt(5:ℝ) = sqrt(3:ℝ) * r) :=
sorry

end sqrt2_sqrt3_sqrt5_not_geometric_sequence_l663_663717


namespace maximize_x3y5_l663_663289

noncomputable theory

open Real

theorem maximize_x3y5 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 20) :
  (x, y) = (7.5, 12.5) → x^3 * y^5 = max (fun p => let (x', y') := p in x' > 0 ∧ y' > 0 ∧ x' + y' = 20 → x'^3 * y'^5) :=
begin
  sorry
end

end maximize_x3y5_l663_663289


namespace complement_A_in_U_l663_663196

/-- Problem conditions -/
def is_universal_set (x : ℕ) : Prop := (x - 6) * (x + 1) ≤ 0
def A : Set ℕ := {1, 2, 4}
def U : Set ℕ := { x | is_universal_set x }

/-- Proof statement -/
theorem complement_A_in_U : (U \ A) = {3, 5, 6} :=
by
  sorry  -- replacement for the proof

end complement_A_in_U_l663_663196


namespace no_roots_ge_two_l663_663800

theorem no_roots_ge_two (x : ℝ) (h : x ≥ 2) : 4 * x^3 - 5 * x^2 - 6 * x + 3 ≠ 0 := by
  sorry

end no_roots_ge_two_l663_663800


namespace max_knights_between_knights_l663_663469

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l663_663469


namespace solve_for_x_l663_663321

theorem solve_for_x (x : ℝ) (h : 3^(x-2) = 9^(x+2)) : x = -6 :=
sorry

end solve_for_x_l663_663321


namespace distance_third_day_l663_663264

theorem distance_third_day (total_distance : ℝ) (days : ℕ) (first_day_factor : ℝ) (halve_factor : ℝ) (third_day_distance : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ first_day_factor = 4 ∧ halve_factor = 0.5 →
  third_day_distance = 48 := sorry

end distance_third_day_l663_663264


namespace sphere_to_cube_volume_ratio_l663_663066

noncomputable def volume_ratio (s : ℝ) : ℝ :=
  let r := s / 4
  let V_s := (4/3:ℝ) * Real.pi * r^3 
  let V_c := s^3
  V_s / V_c

theorem sphere_to_cube_volume_ratio (s : ℝ) (h : s > 0) : volume_ratio s = Real.pi / 48 := by
  sorry

end sphere_to_cube_volume_ratio_l663_663066


namespace sufficient_but_not_necessary_condition_l663_663642

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h₀ : b > a) (h₁ : a > 0) :
  (1 / (a ^ 2) > 1 / (b ^ 2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l663_663642


namespace five_distinct_coprime_and_composite_sum_l663_663128

def a_i (i : ℕ) : ℕ := i * Nat.factorial 5 + 1

theorem five_distinct_coprime_and_composite_sum :
  ∃ (a1 a2 a3 a4 a5 : ℕ), 
    (a1 = a_i 1) ∧ (a2 = a_i 2) ∧ (a3 = a_i 3) ∧ (a4 = a_i 4) ∧ (a5 = a_i 5) ∧
    (∀ (x y : ℕ), x ≠ y → (x = a1 ∨ x = a2 ∨ x = a3 ∨ x = a4 ∨ x = a5) → (y = a1 ∨ y = a2 ∨ y = a3 ∨ y = a4 ∨ y = a5) → Nat.gcd x y = 1) ∧
    (∀ (S : Finset ℕ), 1 < S.card → (∀ x ∈ S, x = a1 ∨ x = a2 ∨ x = a3 ∨ x = a4 ∨ x = a5) → (∃ d : ℕ, d > 1 ∧ d ∣ S.sum)) :=
by
  use a_i 1, a_i 2, a_i 3, a_i 4, a_i 5
  split
  repeat {split},
  all_goals {
    sorry
  }

end five_distinct_coprime_and_composite_sum_l663_663128


namespace sum_first_10_terms_zero_l663_663693

noncomputable theory

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + n * d

-- Given conditions
variables (a₁ d : ℝ)
variable (h_d_nonzero : d ≠ 0)
variable (h_cond : (arithmetic_seq a₁ d 3)^2 + (arithmetic_seq a₁ d 4)^2 = (arithmetic_seq a₁ d 5)^2 + (arithmetic_seq a₁ d 6)^2)

-- We want to prove that the sum of the first 10 terms is 0
theorem sum_first_10_terms_zero (a₁ d : ℝ) (h_d_nonzero : d ≠ 0)
  (h_cond : (arithmetic_seq a₁ d 3)^2 + (arithmetic_seq a₁ d 4)^2 = (arithmetic_seq a₁ d 5)^2 + (arithmetic_seq a₁ d 6)^2) :
  (∑ i in finset.range 10, arithmetic_seq a₁ d i) = 0 :=
sorry

end sum_first_10_terms_zero_l663_663693


namespace find_m_l663_663299

theorem find_m (θ₁ θ₂ : ℝ) (l : ℝ → ℝ) (m : ℕ) 
  (hθ₁ : θ₁ = Real.pi / 100) 
  (hθ₂ : θ₂ = Real.pi / 75)
  (hl : ∀ x, l x = x / 4) 
  (R : ((ℝ → ℝ) → (ℝ → ℝ)))
  (H_R : ∀ l, R l = (sorry : ℝ → ℝ)) 
  (R_n : ℕ → (ℝ → ℝ) → (ℝ → ℝ)) 
  (H_R1 : R_n 1 l = R l) 
  (H_Rn : ∀ n, R_n (n + 1) l = R (R_n n l)) :
  m = 1500 :=
sorry

end find_m_l663_663299


namespace length_of_AB_in_parallelogram_l663_663256

theorem length_of_AB_in_parallelogram 
  (ABCD : parallelogram)
  (P : point)
  (hP_on_BC : P ∈ line_segment B C)
  (BP : ℝ)
  (CP : ℝ)
  (tan_APD : ℝ)
  (hBP : BP = 20)
  (hCP : CP = 10)
  (h_tan_APD : tan_APD = 4) :
  AB = 18.375 :=
by sorry

end length_of_AB_in_parallelogram_l663_663256


namespace proof_l663_663591

variable (α : ℝ)

def p : Prop := (cos (2 * (α - (π / 5))) = cos (2 * α - (2 * π / 5)))
def q : Prop := tan α = 2 → (cos α ^ 2 - 2 * sin α ^ 2) / (sin (2 * α)) = -7 / 4

theorem proof : ¬p ∧ q :=
by
  sorry

end proof_l663_663591


namespace mean_is_4_greater_than_median_l663_663896

-- Define the set of integers
def integers (x : ℕ) : list ℕ := [x, x + 2, x + 4, x + 7, x + 27]

-- Calculate the mean of the set
def mean (x : ℕ) : ℕ := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5

-- Calculate the median of the set
def median (x : ℕ) : ℕ := x + 4

-- The main theorem to prove
theorem mean_is_4_greater_than_median (x : ℕ) : mean x = median x + 4 := by
  sorry

end mean_is_4_greater_than_median_l663_663896


namespace number_of_distinct_elements_in_set_l663_663155

noncomputable def f (n : ℕ) : ℂ := complex.I^n - complex.I^(-n)

theorem number_of_distinct_elements_in_set : 
  (set.count (set.image f {n : ℕ | n > 0})).card = 3 := 
sorry

end number_of_distinct_elements_in_set_l663_663155


namespace sum_of_consecutive_integers_of_sqrt3_l663_663207

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l663_663207


namespace inequality_proof_l663_663559

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (2 * a^2) / (1 + a + a * b)^2 + (2 * b^2) / (1 + b + b * c)^2 + (2 * c^2) / (1 + c + c * a)^2 +
  9 / ((1 + a + a * b) * (1 + b + b * c) * (1 + c + c * a)) ≥ 1 :=
by {
  sorry -- The proof goes here
}

end inequality_proof_l663_663559


namespace price_of_each_sundae_l663_663805

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℝ)
  (price_per_ice_cream_bar : ℝ)
  (total_cost_for_sundaes : ℝ) :
  num_ice_cream_bars = 225 →
  num_sundaes = 125 →
  total_price = 200 →
  price_per_ice_cream_bar = 0.60 →
  total_cost_for_sundaes = total_price - (num_ice_cream_bars * price_per_ice_cream_bar) →
  (total_cost_for_sundaes / num_sundaes) = 0.52 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_of_each_sundae_l663_663805


namespace count_integers_with_digit_sum_9_l663_663202

theorem count_integers_with_digit_sum_9 : 
  set.count {n : ℕ | 10 ≤ n ∧ n ≤ 999 ∧ (n.digits 10).sum = 9} = 54 := 
by sorry

end count_integers_with_digit_sum_9_l663_663202


namespace sqrt_eq_seventy_two_l663_663533

theorem sqrt_eq_seventy_two (x : ℝ) (h : sqrt (9 + 3 * x) = 15) : x = 72 :=
sorry

end sqrt_eq_seventy_two_l663_663533


namespace apothem_inequality_apothem_inequality_false_for_greater_r_l663_663698

theorem apothem_inequality (n : ℕ) (r : ℝ) (hn : n ≥ 3) (hr : r > 0) : 
  (n+1) * (r * real.cos (real.pi / (n + 1))) - n * (r * real.cos (real.pi / n)) > r :=
sorry

theorem apothem_inequality_false_for_greater_r (n : ℕ) (r k : ℝ) (hn : n ≥ 3) (hr : r > 0) (hk : k > r) : 
  ¬ ((n+1) * (r * real.cos (real.pi / (n + 1))) - n * (r * real.cos (real.pi / n)) > k) :=
sorry

end apothem_inequality_apothem_inequality_false_for_greater_r_l663_663698


namespace exists_coprime_among_consecutive_l663_663413

theorem exists_coprime_among_consecutive (n : ℕ) :
  ∃ m ∈ (set.Icc n (n + 8)), ∀ x ∈ (set.Icc n (n + 8)), m ≠ x → Nat.gcd m x = 1 :=
by
  sorry

end exists_coprime_among_consecutive_l663_663413


namespace odd_function_has_specific_a_l663_663650

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def f (a : ℝ) : ℝ → ℝ := λ x, x / ((2*x + 1) * (x - a))

theorem odd_function_has_specific_a (a : ℝ) (h : is_odd_function (f a)) : a = 1 / 2 :=
by
  sorry

end odd_function_has_specific_a_l663_663650


namespace actors_per_group_l663_663984

theorem actors_per_group (actors_per_hour : ℕ) (show_time_per_actor : ℕ) (total_show_time : ℕ)
  (h1 : show_time_per_actor = 15) (h2 : actors_per_hour = 20) (h3 : total_show_time = 60) :
  actors_per_hour * show_time_per_actor / total_show_time = 5 :=
by sorry

end actors_per_group_l663_663984


namespace solve_for_a_l663_663247

theorem solve_for_a (a : ℚ) (h : 2 * a - 3 = 5 - a) : a = 8 / 3 :=
by
  sorry

end solve_for_a_l663_663247


namespace mat_mul_eq_l663_663504

variable {α : Type*} [CommRing α]

def matA : Matrix (Fin 3) (Fin 3) α :=
  ![![2, 3, -1], ![1, 5, -2], ![0, 6, 2]]

def matB : Matrix (Fin 3) (Fin 3) α :=
  ![![3, -1, 0], ![2, 1, -4], ![5, 0, 1]]

def matC : Matrix (Fin 3) (Fin 3) α :=
  ![![7, 1, -13], ![3, 4, -22], ![22, 6, -22]]

theorem mat_mul_eq :
  matA.mul matB = matC :=
by
  sorry

end mat_mul_eq_l663_663504


namespace cylinder_has_two_views_same_l663_663077

-- Define the views for each geometric shape.
def views (shape : String) : List String :=
  match shape with
  | "Cube"             => ["same", "same", "same"]
  | "Sphere"           => ["same", "same", "same"]
  | "Triangular Prism" => ["rectangle1", "rectangle2", "triangle"]
  | "Cylinder"         => ["rectangle", "rectangle", "circle"]
  | _                  => []

-- Define a check for if exactly two views are the same among the three views.
def exactly_two_views_same (views : List String) : Bool :=
  views = ["same", "same", _] ∨ views = [_, "same", "same"] ∨ views = ["same", _, "same"]

-- Prove that Cylinder is the only shape with exactly two views the same.
theorem cylinder_has_two_views_same :
  ∃ shape, (shape = "Cube" ∨ shape = "Sphere" ∨ shape = "Triangular Prism" ∨ shape = "Cylinder")
    ∧ exactly_two_views_same (views shape) = true ↔ shape = "Cylinder" := by
  sorry

end cylinder_has_two_views_same_l663_663077


namespace max_knights_between_other_knights_l663_663480

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l663_663480


namespace steakmaker_exists_l663_663409

theorem steakmaker_exists (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_steakmaker : 1 + 2 ^ m = n ^ 2) : 
(m = 3 ∧ n = 3 ∧ (m * n = 9)) :=
begin
  sorry
end

end steakmaker_exists_l663_663409


namespace equal_area_division_l663_663520

theorem equal_area_division (d : ℝ) : 
  (∃ x y, 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 
   (x = d ∨ x = 4) ∧ (y = 4 ∨ y = 0) ∧ 
   (2 : ℝ) * (4 - d) = 4) ↔ d = 2 :=
by
  sorry

end equal_area_division_l663_663520


namespace max_knights_adjacent_to_two_other_knights_l663_663487

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l663_663487


namespace find_a_l663_663938

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 0 then 1 - x else a ^ x

theorem find_a (a : ℝ) : f a 1 = f a (-1) → a = 2 :=
by
  intro h
  have h1 : f a 1 = a := by
    simp [f]
    split_ifs
    { exfalso; linarith }
    { rfl }
  have h2 : f a (-1) = 2 := by
    simp [f]
    split_ifs
    { norm_num }
    { exfalso; linarith }
  rw [h1, h2] at h
  exact eq_of_eq h


end find_a_l663_663938


namespace quadratic_coefficient_b_l663_663451

theorem quadratic_coefficient_b :
  ∃ (b : ℝ), (∀ (x : ℝ), 2 * x^2 + b * x - 119 = 0 → x = 7 ∨ x = -17 / 2) → b = 3 :=
begin
  use 3,
  intro h,
  sorry
end

end quadratic_coefficient_b_l663_663451


namespace Frank_initial_savings_l663_663901

theorem Frank_initial_savings 
  (cost_per_toy : Nat)
  (number_of_toys : Nat)
  (allowance : Nat)
  (total_cost : Nat)
  (initial_savings : Nat)
  (h1 : cost_per_toy = 8)
  (h2 : number_of_tys = 5)
  (h3 : allowance = 37)
  (h4 : total_cost = number_of_toys * cost_per_toy)
  (h5 : initial_savings + allowance = total_cost)
  : initial_savings = 3 := 
by
  sorry

end Frank_initial_savings_l663_663901


namespace ellen_painted_17_lilies_l663_663521

theorem ellen_painted_17_lilies :
  (∃ n : ℕ, n * 5 + 10 * 7 + 6 * 3 + 20 * 2 = 213) → 
    ∃ n : ℕ, n = 17 := 
by sorry

end ellen_painted_17_lilies_l663_663521


namespace bridge_length_l663_663442

theorem bridge_length (T : Type) [LinearOrder T] [HasAdd T] [HasMul T] 
  (train_length : T) (train_time_to_cross_lamp_post : T) 
  (total_time_to_cross_bridge : T) (speed : T) (bridge_length : T) :
  train_length = 400 →
  train_time_to_cross_lamp_post = 15 →
  total_time_to_cross_bridge = 45 →
  speed = train_length / train_time_to_cross_lamp_post →
  (bridge_length = speed * total_time_to_cross_bridge - train_length) ∧
  bridge_length = 800 :=
begin
  intros h1 h2 h3 h4,
  have h5 : speed = 80 / 3 := by {
    rw [h4, h1, h2],
    exact eq.refl _,
  },
  split,
  { sorry },
  { sorry },
end

end bridge_length_l663_663442


namespace weight_of_replaced_person_l663_663335

variable (averageWeightIncrease : ℝ)
variable (newPersonWeight : ℝ)

theorem weight_of_replaced_person (averageWeightIncrease = 8.5) (newPersonWeight = 129) : 
  let increaseInTotalWeight := 4 * averageWeightIncrease in
  let weightOld := newPersonWeight - increaseInTotalWeight in 
  weightOld = 95 := 
by
  sorry

end weight_of_replaced_person_l663_663335


namespace tangent_parallel_l663_663248

def curve (x : ℝ) : ℝ := x^3 + x - 2
def tangent_slope (x : ℝ) : ℝ := Deriv 1 (λ x, x^3 + x - 2)

theorem tangent_parallel (P : ℝ × ℝ) 
  (h_tangent : (∃ (x : ℝ), curve x = P.2 ∧ P.1 = x ∧ tangent_slope x = 4)) : 
  (P = (1, 0) ∨ P = (-1, -4)) :=
by
  sorry  

end tangent_parallel_l663_663248


namespace butterfat_calculation_correct_l663_663375

noncomputable def butterfat_percentage_of_milk_added : ℕ := by 
  let butterfat_p1 := 0.30 * 8
  let gallons_p2 := 8
  let total_gallons := 8 + gallons_p2
  let total_butterfat := 0.20 * total_gallons
  let x := (total_butterfat - butterfat_p1) / (0.01 * gallons_p2)
  exact x

theorem butterfat_calculation_correct : butterfat_percentage_of_milk_added = 10 := by
  sorry

end butterfat_calculation_correct_l663_663375


namespace points_concyclic_l663_663412

variables (A B C D L K O : Point)

-- Definitions of the given conditions
def is_parallelogram (ABCD : Rhombus) := 
  parallelogram ABCD

def angle_acute (A B D : Point) := 
  angle A B D < π / 2

def is_angle_bisector (A B D L K : Point) :=
  is_bisector (angle A B D) L K

def circumcenter (L C K : Point) (O : Point) :=
  is_circumcenter O L C K

-- The statement to prove
theorem points_concyclic 
  (h₁ : is_parallelogram A B C D)
  (h₂ : distance A B > distance A D)
  (h₃ : angle_acute A B D)
  (h₄ : is_angle_bisector A B D L K)
  (h₅ : circumcenter L C K O):
  concyclic B C O D := 
sorry

end points_concyclic_l663_663412


namespace solve_equation1_solve_equation2_l663_663325

def equation1 (x : ℝ) := (x - 1) ^ 2 = 4
def equation2 (x : ℝ) := 2 * x ^ 3 = -16

theorem solve_equation1 (x : ℝ) (h : equation1 x) : x = 3 ∨ x = -1 := 
sorry

theorem solve_equation2 (x : ℝ) (h : equation2 x) : x = -2 := 
sorry

end solve_equation1_solve_equation2_l663_663325


namespace area_perimeter_inequality_l663_663393

-- Defining the conditions for convex polygons and their areas and perimeters
variable {X Y : Type} -- Types representing convex polygons X and Y
variable [ConvexPolygon X] [ConvexPolygon Y] -- X and Y are convex polygons
variable (contained : X ⊆ Y) -- X is contained within Y

-- Defining areas and perimeters
variable (S : X → ℝ) (S' : Y → ℝ) -- Area functions for X and Y respectively
variable (P : X → ℝ) (P' : Y → ℝ) -- Perimeter functions for X and Y respectively

-- Proving the desired property
theorem area_perimeter_inequality
    (A_X : ℝ := S X)
    (A_Y : ℝ := S' Y)
    (P_X : ℝ := P X)
    (P_Y : ℝ := P' Y) :
    A_X / P_X < 2 * (A_Y / P_Y) :=
by
  -- The proof will go here
  sorry

end area_perimeter_inequality_l663_663393


namespace plane_through_points_l663_663884

noncomputable def plane_equation : ℝ × ℝ × ℝ × ℝ := (2, 3, -4, 9)

theorem plane_through_points :
  ∃ (A B C D : ℝ), 
    -- Plane equation in the form Ax + By + Cz + D = 0
    plane_equation = (A, B, C, D) ∧
    -- Points (2, -3, 1), (6, -3, 3), (4, -5, 2) lie on the plane
    (A * 2 + B * -3 + C * 1 + D = 0) ∧
    (A * 6 + B * -3 + C * 3 + D = 0) ∧
    (A * 4 + B * -5 + C * 2 + D = 0) ∧
    -- Additional constraints: A > 0 and gcd(|A|, |B|, |C|, |D|) = 1
    (A > 0) ∧ (Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) (Int.natAbs D))) = 1) :=
begin
  use [2, 3, -4, 9],
  split, exact rfl,
  split, norm_num,
  split, norm_num,
  split, norm_num,
  split,
  exact zero_lt_two,
  norm_num,
end

end plane_through_points_l663_663884


namespace sequence_S_n_a_n_l663_663924

noncomputable def sequence_S (n : ℕ) : ℝ := -1 / (n : ℝ)

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then -1 else 1 / ((n : ℝ) * (n - 1))

theorem sequence_S_n_a_n (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = -1 →
  (∀ n, (a (n + 1)) / (S (n + 1)) = S n) →
  S n = sequence_S n ∧ a n = sequence_a n :=
by
  intros h1 h2
  sorry

end sequence_S_n_a_n_l663_663924


namespace product_ends_in_0_l663_663573

/-- Given a set of 20 random digits, the probability that the product of these digits ends in 
    0 is approximately 0.988 -/
theorem product_ends_in_0 (s : Finset ℕ) (h: s.card = 20) :
  (∑ k in Finset.range 10, if k = 0 then 1 else if ∃ n ∈ s, n = 2 ∧ ∃ m ∈ s, m = 5 then 1 else 0) / 10 ^ 20 ≈ 0.988 := 
sorry

end product_ends_in_0_l663_663573


namespace isosceles_triangle_b_value_l663_663772

theorem isosceles_triangle_b_value (a b : ℝ) :
  (∃ (s₁ s₂ s₃ : ℝ), s₁ = 5 ∧ s₂ = 5 ∧ s₃ = 8 ∧ (s₁ + s₂ + s₃ = 18) ∧ 
   let A := real.sqrt (9 * (9 - s₁) * (9 - s₂) * (9 - s₃)) in A = 12) ∧
  (∃ (x y z : ℝ), x = a ∧ y = a ∧ z = b ∧ (2 * x + z = 18) ∧ 
   let B := (1/2) * z * real.sqrt (x^2 - (z/2)^2) in B = 12) →
  abs (b - 4) < 1 :=
by 
  sorry

end isosceles_triangle_b_value_l663_663772


namespace simplify_expression_l663_663524

theorem simplify_expression : 
  2 * (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8)) = 3 / 4 :=
by
  sorry

end simplify_expression_l663_663524


namespace total_amount_earned_l663_663794

-- Definitions of the conditions.
def work_done_per_day (days : ℕ) : ℚ := 1 / days

def total_work_done_per_day : ℚ :=
  work_done_per_day 6 + work_done_per_day 8 + work_done_per_day 12

def b_share : ℚ := work_done_per_day 8

def total_amount (b_earnings : ℚ) : ℚ := b_earnings * (total_work_done_per_day / b_share)

-- Main theorem stating that the total amount earned is $1170 if b's share is $390.
theorem total_amount_earned (h_b : b_share * 390 = 390) : total_amount 390 = 1170 := by sorry

end total_amount_earned_l663_663794


namespace odd_function_solution_l663_663582

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 1 then x^2 - 1
else if h : -1 < x ∧ x < 0 then -(x^2) + 1
else 0

theorem odd_function_solution (x₀ : ℝ) (h_odd : ∀ x, f(-x) = -f(x)) 
    (h_defined : ∀ x, 0 < x ∧ x < 1 → f(x) = x^2 - 1) :
    f(x₀) = 1/2 → 
    x₀ = - (Real.sqrt 2) / 2 :=
sorry

end odd_function_solution_l663_663582


namespace hydrogen_moles_formed_l663_663140

open Function

-- Define types for the substances involved in the reaction
structure Substance :=
  (name : String)
  (moles : ℕ)

-- Define the reaction
def reaction (NaH H2O NaOH H2 : Substance) : Prop :=
  NaH.moles = H2O.moles ∧ NaOH.moles = H2.moles

-- Given conditions
def NaH_initial : Substance := ⟨"NaH", 2⟩
def H2O_initial : Substance := ⟨"H2O", 2⟩
def NaOH_final : Substance := ⟨"NaOH", 2⟩
def H2_final : Substance := ⟨"H2", 2⟩

-- Problem statement in Lean
theorem hydrogen_moles_formed :
  reaction NaH_initial H2O_initial NaOH_final H2_final → H2_final.moles = 2 :=
by
  -- Skip proof
  sorry

end hydrogen_moles_formed_l663_663140


namespace quadratic_distinct_real_roots_l663_663245

open Real

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x ^ 2 - 2 * x - 1 = 0 ∧ k * y ^ 2 - 2 * y - 1 = 0) ↔ k > -1 ∧ k ≠ 0 :=
by
  sorry

end quadratic_distinct_real_roots_l663_663245


namespace max_knights_adjacent_to_two_other_knights_l663_663484

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l663_663484


namespace house_sequence_count_l663_663315

theorem house_sequence_count : 
  let houses := ["orange", "red", "blue", "yellow", "green"]
  ∃ (seq : list string), 
    seq.permutations houses ∧
    (list.index_of "orange" seq < list.index_of "red" seq) ∧
    (list.index_of "blue" seq < list.index_of "yellow" seq) ∧
    (list.index_of "yellow" seq ≠ list.index_of "blue" seq + 1) ∧
    (list.index_of "yellow" seq ≠ list.index_of "blue" seq - 1) ∧
    (list.index_of "green" seq < list.index_of "red" seq) ∧
    seq.length = houses.length ∧
    (seq.permutations.filter (λ s, 
      (list.index_of "orange" s < list.index_of "red" s) ∧
      (list.index_of "blue" s < list.index_of "yellow" s) ∧
      (list.index_of "yellow" s ≠ list.index_of "blue" s + 1) ∧
      (list.index_of "yellow" s ≠ list.index_of "blue" s - 1) ∧
      (list.index_of "green" s < list.index_of "red" s)).length = 9) := sorry

end house_sequence_count_l663_663315


namespace partition_with_pink_cells_l663_663253

-- Representation of an n x n grid.
def grid (n : ℕ) := (fin n) × (fin n)

-- Define what a rectangle is in the grid.
structure rectangle (n : ℕ) :=
(top_left bottom_right : grid n)
(horiz_ineq : top_left.1 ≤ bottom_right.1)
(vert_ineq : top_left.2 ≤ bottom_right.2)

-- Define a function to check if a rectangle contains a pink cell.
def contains_pink (n : ℕ) (rect : rectangle n) (is_pink : grid n → Prop) : Prop :=
∃ cell : grid n, is_pink cell ∧
  rect.top_left.1 ≤ cell.1 ∧ cell.1 ≤ rect.bottom_right.1 ∧
  rect.top_left.2 ≤ cell.2 ∧ cell.2 ≤ rect.bottom_right.2

-- Main theorem: For any n x n grid with at least one pink cell, you can partition
-- the grid into rectangles such that each rectangle contains exactly one pink cell.
theorem partition_with_pink_cells (n : ℕ) (is_pink : grid n → Prop) (h : ∃ cell : grid n, is_pink cell) :
  ∃ (rects : list (rectangle n)), (∀ rect ∈ rects, contains_pink n rect is_pink) ∧
  (∀ rect1 rect2 ∈ rects, rect1 ≠ rect2 → disjoint rect1 rect2) ∧
  (⋃ (rect : rectangle n) (h : rect ∈ rects), grid_points_in rect = { cell : grid n | is_pink cell }) :=
sorry

-- Auxiliary predicate to check if two rectangles are disjoint.
def disjoint {n : ℕ} (rect1 rect2 : rectangle n) : Prop :=
rect1.bottom_right.1 < rect2.top_left.1 ∨
rect2.bottom_right.1 < rect1.top_left.1 ∨
rect1.bottom_right.2 < rect2.top_left.2 ∨
rect2.bottom_right.2 < rect1.top_left.2

-- Function to get all grid points in a rectangle.
def grid_points_in {n : ℕ} (rect : rectangle n) : finset (grid n) :=
(fin_range rect.top_left.1 rect.bottom_right.1) ×ˢ (fin_range rect.top_left.2 rect.bottom_right.2)

-- Helper to generate a finset of fin n within a range
def fin_range (a b : fin n) : finset (fin n) :=
(finset.Icc a b)

end partition_with_pink_cells_l663_663253


namespace inequality_solution_l663_663652

theorem inequality_solution (a : ℝ) (h : a^2 > 2 * a - 1) : a ≠ 1 := 
sorry

end inequality_solution_l663_663652


namespace quadratic_expression_inequality_l663_663880

theorem quadratic_expression_inequality (k : ℝ) : 
  (k^2 - 4*k - 12 < 0) → k ∈ set.Ioo (-2 : ℝ) (6 : ℝ) :=
by {
  intros h,
  sorry
}


end quadratic_expression_inequality_l663_663880


namespace area_diff_circle_square_l663_663738

theorem area_diff_circle_square (d_square d_circle : ℝ) (h1 : d_square = 10) (h2 : d_circle = 10) :
  let s := d_square / Real.sqrt 2,
      area_square := s^2,
      r := d_circle / 2,
      area_circle := Real.pi * r^2,
      area_diff := area_circle - area_square in
  Real.floor (area_diff * 10) / 10 = 28.5 :=
by
  sorry

end area_diff_circle_square_l663_663738


namespace prank_combinations_l663_663766

theorem prank_combinations :
  let monday := 1
  let tuesday := 4
  let wednesday := 7
  let thursday := 5
  let friday := 1
  (monday * tuesday * wednesday * thursday * friday) = 140 :=
by
  sorry

end prank_combinations_l663_663766


namespace sum_of_consecutive_integers_l663_663217

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l663_663217


namespace vans_for_field_trip_l663_663038

def total_people (students adults : ℕ) : ℕ :=
  students + adults

def vans_needed (total van_capacity : ℕ) : ℕ :=
  (total + van_capacity - 1) / van_capacity  -- to ensure we round up for any remainder

theorem vans_for_field_trip : 
  ∀ (students adults van_capacity : ℕ), 
  students = 12 → adults = 3 → van_capacity = 5 → 
  vans_needed (total_people students adults) van_capacity = 3 := 
by 
  intros students adults van_capacity h1 h2 h3 
  simp [total_people, vans_needed, h1, h2, h3]
  sorry

end vans_for_field_trip_l663_663038


namespace repeating_decimal_addition_subtraction_l663_663527

theorem repeating_decimal_addition_subtraction :
  (let x y z : ℚ := 0.\overline{5}, 0.\overline{1}, 0.\overline{3} in x + y - z) = (1 / 3) := by
sorry

end repeating_decimal_addition_subtraction_l663_663527


namespace tina_first_hour_coins_l663_663374

variable (X : ℕ)

theorem tina_first_hour_coins :
  let first_hour_coins := X
  let second_third_hour_coins := 30 + 30
  let fourth_hour_coins := 40
  let fifth_hour_removed_coins := 20
  let total_coins := first_hour_coins + second_third_hour_coins + fourth_hour_coins - fifth_hour_removed_coins
  total_coins = 100 → X = 20 :=
by
  intro h
  sorry

end tina_first_hour_coins_l663_663374


namespace min_value_fraction_l663_663749

variable (a m n : ℝ)
variable (a_pos : a > 0) (a_ne_one : a ≠ 1) (m_pos : m > 0) (n_pos : n > 0)

noncomputable def log_fn := λ x : ℝ, -1 + Real.log (x+3) / Real.log a

theorem min_value_fraction :
  log_fn a (-2) = -1 → 
  (∀ x y, x = -2 ∧ y = -1 → m * x + n * y + 1 = 0) → 
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ (2 * m + n = 1) ∧ (∀ (m_pos : m > 0) (n_pos : n > 0), 4 + Real.sqrt ((n / m) * (4 * m / n)) ≥ 8)
  :=
  sorry -- Proof will be added here

end min_value_fraction_l663_663749


namespace complex_sum_power_l663_663694

noncomputable def z : ℂ := sorry

theorem complex_sum_power (hz : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 :=
sorry

end complex_sum_power_l663_663694


namespace equation_of_parallel_tangent_line_through_P_l663_663746

theorem equation_of_parallel_tangent_line_through_P:
  let C : ℝ → ℝ := λ x => 3 * x^2 - 4 * x + 2
  let P : ℝ × ℝ := (-1, 2)
  let tangent_slope_at : ℝ → ℝ := λ x => (deriv C x)
  let M : ℝ × ℝ := (1, C 1)
  let tangent_line_eq : ℝ × ℝ → (ℝ → ℝ) → Prop := λ (A : ℝ × ℝ) (f : ℝ → ℝ), 
    ∀ x: ℝ, x * (2) + f P.1 == P.2

  in tangent_line_eq P (λ x: ℝ, 2 * x + 4)

end equation_of_parallel_tangent_line_through_P_l663_663746


namespace base_5_twenty_fifth_number_l663_663983

theorem base_5_twenty_fifth_number : 
    (to_base 5 25 = [1, 0, 0]) :=
by
  sorry

end base_5_twenty_fifth_number_l663_663983


namespace community_theater_ticket_sales_l663_663007

theorem community_theater_ticket_sales (A C : ℕ) 
  (h1 : 12 * A + 4 * C = 840) 
  (h2 : A + C = 130) :
  A = 40 :=
sorry

end community_theater_ticket_sales_l663_663007


namespace intersection_M_N_l663_663947

-- Definitions
def M : Set ℝ := { x | x^2 - 3x - 4 < 0 }
def N : Set ℝ := { x | -5 ≤ x ∧ x ≤ 0 }

-- Theorem Statement
theorem intersection_M_N : M ∩ N = { x | -1 < x ∧ x ≤ 0 } :=
by
  -- Proof omitted
  sorry

end intersection_M_N_l663_663947


namespace incorrect_statements_l663_663153

def f (ω x : ℝ) : ℝ := (√ 3 / 2) * sin (ω * x) - cos (ω * x / 2) ^ 2 + 1 / 2

theorem incorrect_statements (ω : ℝ) (hω : ω > 0) :
  ¬ (∀ x, 0 < x ∧ x < π → (0 < ω ∧ ω ≤ 1 / 6) = false) ∧
  ¬ (∀ x, (abs (f ω x)) = ( ∀ ω, ω = 2) = false) :=
sorry

end incorrect_statements_l663_663153


namespace red_triangle_or_blue_quadrilateral_l663_663073

open SimpleGraph

theorem red_triangle_or_blue_quadrilateral 
  (G : SimpleGraph (Fin 9)) [CompleteGraph G] 
  {c : Sym2 (Fin 9) → Prop} 
  (hc : ∀ e, c e ∨ (¬ c e)) :
  (∃ (a b c : Fin 9), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ c ⟦a, b⟧ ∧ c ⟦b, c⟧ ∧ c ⟦a, c⟧) ∨
  (∃ (d e f g : Fin 9),
    d ≠ e ∧ e ≠ f ∧ f ≠ g ∧ g ≠ d ∧ d ≠ f ∧ e ≠ g ∧
    ¬ c ⟦d, e⟧ ∧ ¬ c ⟦d, f⟧ ∧ ¬ c ⟦d, g⟧ ∧
    ¬ c ⟦e, f⟧ ∧ ¬ c ⟦e, g⟧ ∧ ¬ c ⟦f, g⟧) :=
sorry

end red_triangle_or_blue_quadrilateral_l663_663073


namespace max_knights_between_knights_l663_663472

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l663_663472


namespace part1_part2_part3_l663_663612

-- Given
def f (a : ℝ) (x : ℝ) := a^x + a^(-x)
axiom a_gt_one (a : ℝ) : a > 1
axiom f_one_eq_three (a : ℝ) : f a 1 = 3

-- To Prove
theorem part1 (a : ℝ) [a_gt_one a] [f_one_eq_three a] : f a 2 = 7 := sorry

theorem part2 (a : ℝ) [a_gt_one a] : ∀ x1 x2 : ℝ, x1 > x2 ∧ (x2 ≥ 0) → f a x1 > f a x2 := sorry

theorem part3 (a : ℝ) (m : ℝ) [a_gt_one a] [f_one_eq_three a] : 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 
  let y := f a (2 * x) - m * f a x in
  ∃ y_min : ℝ, (m ≤ 4 ∧ y_min = 2-2*m) ∨ (4 < m ∧ m < 6 ∧ y_min = -m^2/4 - 2) ∨ (m ≥ 6 ∧ y_min = 7-3*m) := sorry

end part1_part2_part3_l663_663612


namespace inequality_proof_l663_663589

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)

theorem inequality_proof :
  (a^2 / (c * (b + c)) + b^2 / (a * (c + a)) + c^2 / (b * (a + b))) >= 3 / 2 :=
by
  sorry

end inequality_proof_l663_663589


namespace fruit_ratio_calc_l663_663666

noncomputable def initial_pineapples := 200
noncomputable def initial_apples := 300
noncomputable def initial_oranges := 100

noncomputable def sold_pineapples := 56
noncomputable def sold_apples := 128
noncomputable def sold_oranges := 22

noncomputable def spoil_rate_pineapples := 0.10
noncomputable def spoil_rate_apples := 0.15
noncomputable def spoil_rate_oranges := 0.05

noncomputable def remaining_pineapples : ℕ := initial_pineapples - sold_pineapples
noncomputable def remaining_apples : ℕ := initial_apples - sold_apples
noncomputable def remaining_oranges : ℕ := initial_oranges - sold_oranges

noncomputable def spoiled_pineapples : ℕ := (remain_pineapples * spoil_rate_pineapples).round
noncomputable def spoiled_apples : ℕ := (remain_apples * spoil_rate_apples).round
noncomputable def spoiled_oranges : ℕ := (remain_oranges * spoil_rate_oranges).round

noncomputable def fresh_pineapples : ℕ := remain_pineapples - spoiled_pineapples
noncomputable def fresh_apples : ℕ := remain_apples - spoiled_apples
noncomputable def fresh_oranges : ℕ := remain_oranges - spoiled_oranges

theorem fruit_ratio_calc :
  fresh_pineapples = 130 ∧ fresh_apples = 146 ∧ fresh_oranges = 74 := 
by
  sorry

end fruit_ratio_calc_l663_663666


namespace cannot_be_cylinder_l663_663975

-- Define the geometric body type
inductive GeometricBody
| sphere
| triangular_pyramid
| cube
| cylinder

open GeometricBody

-- Define the conditions as functions
def length_match : GeometricBody → Prop
| sphere := true
| triangular_pyramid := true
| cube := true
| cylinder := false

def height_match : GeometricBody → Prop
| sphere := true
| triangular_pyramid := true
| cube := true
| cylinder := false

def width_match : GeometricBody → Prop
| sphere := true
| triangular_pyramid := true
| cube := true
| cylinder := false

-- Define the theorem
theorem cannot_be_cylinder (G : GeometricBody) :
  length_match G = true ∧ height_match G = true ∧ width_match G = true → G ≠ cylinder :=
by
  intros h
  cases h with lm hm
  cases hm with hm wm
  cases G
  · sorry
  · sorry
  · sorry
  · contradiction


end cannot_be_cylinder_l663_663975


namespace crayon_division_l663_663902

theorem crayon_division (total_crayons : ℕ) (crayons_each : ℕ) (Fred Benny Jason : ℕ) 
  (h_total : total_crayons = 24) (h_each : crayons_each = 8) 
  (h_division : Fred = crayons_each ∧ Benny = crayons_each ∧ Jason = crayons_each) : 
  Fred + Benny + Jason = total_crayons :=
by
  sorry

end crayon_division_l663_663902


namespace complex_magnitude_l663_663560

noncomputable def z : ℂ := -1 + complex.I

theorem complex_magnitude :
  (H : complex.I - z = 1 + 2 * complex.I) → complex.abs z = real.sqrt 2 :=
begin
  intro H,
  sorry
end

end complex_magnitude_l663_663560


namespace fuel_consumption_rate_l663_663836

theorem fuel_consumption_rate (fuel_left time_left r: ℝ) 
    (h_fuel: fuel_left = 6.3333) 
    (h_time: time_left = 0.6667) 
    (h_rate: r = fuel_left / time_left) : r = 9.5 := 
by
    sorry

end fuel_consumption_rate_l663_663836


namespace trigonometric_identity_l663_663106

theorem trigonometric_identity :
  (1 - 1/real.cos (30 * real.pi / 180)) *
  (1 + 1/real.sin (60 * real.pi / 180)) *
  (1 - 1/real.sin (30 * real.pi / 180)) *
  (1 + 1/real.cos (60 * real.pi / 180)) = 1 := 
sorry

end trigonometric_identity_l663_663106


namespace variance_of_data_set_l663_663436

-- Define the set of data
def data_set := [7, 8, 10, 8, 9, 6]

-- Average of the data set
def avg (l : List ℝ) := l.sum / l.length

-- Squared differences from the average
def squared_diffs (l : List ℝ) := (l.map (λ x => (x - avg l) ^ 2))

-- Variance of the data set
def variance (l : List ℝ) := (squared_diffs l).sum / l.length

-- Statement to prove
theorem variance_of_data_set : variance data_set = 5 / 3 :=
by
  sorry

end variance_of_data_set_l663_663436


namespace find_a_plus_b_l663_663170

theorem find_a_plus_b (a b : ℝ)
  (hz1 : z1 = (sqrt 3 / 2) * a + (a + 1) * complex.I)
  (hz2 : z2 = -3 * sqrt 3 * b + (b + 2) * complex.I)
  (h_real : a - b = 1)
  (h_img : (sqrt 3 / 2) * a + 3 * sqrt 3 * b = 4 * sqrt 3) :
  a + b = 29 / 15 :=
  sorry

end find_a_plus_b_l663_663170


namespace parabola_equation_and_slopes_constant_l663_663944

namespace ParabolaProof

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def line (m : ℝ) (x y : ℝ) := x = m * y + 3
def slopes (A B C : ℝ × ℝ) (k1 k2 : ℝ) := 
  k1 = (A.2) / (A.1 + 3) ∧ k2 = (B.2) / (B.1 + 3)
def dot_product (A B : ℝ × ℝ) := A.1 * B.1 + A.2 * B.2

theorem parabola_equation_and_slopes_constant
  (p : ℝ)
  (hp : p = 1/2)
  (A B : ℝ × ℝ)
  (hA : parabola p A.1 A.2)
  (hB : parabola p B.1 B.2)
  (hlineA : line m A.1 A.2)
  (hlineB : line m B.1 B.2)
  (hO : dot_product A B = 6)
  (C : ℝ × ℝ)
  (hC : C = (-3, 0))
  (k1 k2 : ℝ)
  (hk : slopes A B C k1 k2) :
  parabola p A.1 A.2 ∧
  parabola p B.1 B.2 ∧
  (frac 1 (k1^2) + frac 1 (k2^2) - 2 * m^2) = 24 := 
sorry

end ParabolaProof

end parabola_equation_and_slopes_constant_l663_663944


namespace possible_distances_AG_l663_663750

theorem possible_distances_AG (A B V G : ℝ) (AB VG : ℝ) (x AG : ℝ) :
  (AB = 600) →
  (VG = 600) →
  (AG = 3 * x) →
  (AG = 900 ∨ AG = 1800) :=
by
  intros h1 h2 h3
  sorry

end possible_distances_AG_l663_663750


namespace four_convex_figures_intersect_l663_663922

variable {α : Type*} [MetricSpace α]
variable Φ0 Φ1 Φ2 Φ3 : Set α

theorem four_convex_figures_intersect 
  (h0 : Convex ℝ Φ0)
  (h1 : Convex ℝ Φ1)
  (h2 : Convex ℝ Φ2)
  (h3 : Convex ℝ Φ3)
  (h012 : (Φ1 ∩ Φ2 ∩ Φ3).Nonempty)
  (h013 : (Φ0 ∩ Φ2 ∩ Φ3).Nonempty)
  (h023 : (Φ0 ∩ Φ1 ∩ Φ3).Nonempty)
  (h123 : (Φ0 ∩ Φ1 ∩ Φ2).Nonempty) :
    (Φ0 ∩ Φ1 ∩ Φ2 ∩ Φ3).Nonempty := 
  sorry

end four_convex_figures_intersect_l663_663922


namespace min_possible_value_l663_663137

theorem min_possible_value (x : ℝ) : ∃ y : ℝ, (y = (x^2 + 6x + 2)^2) ∧ y = 0 :=
sorry

end min_possible_value_l663_663137


namespace papaya_cost_is_one_l663_663016

theorem papaya_cost_is_one (lemons_cost : ℕ) (mangos_cost : ℕ) (total_fruits : ℕ) (total_cost_paid : ℕ) :
    (lemons_cost = 2) → (mangos_cost = 4) → (total_fruits = 12) → (total_cost_paid = 21) → 
    let discounts := total_fruits / 4
    let lemons_bought := 6
    let mangos_bought := 2
    let papayas_bought := 4
    let total_discount := discounts
    let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
    total_cost_before_discount - total_discount = total_cost_paid → 
    P = 1 := 
by 
  intros h1 h2 h3 h4 
  let discounts := total_fruits / 4
  let lemons_bought := 6
  let mangos_bought := 2
  let papayas_bought := 4
  let total_discount := discounts
  let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
  sorry

end papaya_cost_is_one_l663_663016


namespace average_score_l663_663402

theorem average_score (avg1 avg2 : ℕ) (matches1 matches2 : ℕ) (h_avg1 : avg1 = 60) (h_matches1 : matches1 = 10) (h_avg2 : avg2 = 70) (h_matches2 : matches2 = 15) : 
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2) = 66 :=
by
  sorry

end average_score_l663_663402


namespace find_b_l663_663657

theorem find_b (a c S : ℝ) (h₁ : a = 5) (h₂ : c = 2) (h₃ : S = 4) : 
  b = Real.sqrt 17 ∨ b = Real.sqrt 41 := by
  sorry

end find_b_l663_663657


namespace sum_of_consecutive_integers_l663_663219

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l663_663219


namespace evaluate_expression_l663_663122

theorem evaluate_expression : 
  (√((16^6 + 8^8) / (16^3 + 8^9)) = 1 / 2) :=
by
  -- Define the transformations for 16 and 8
  have h16 : (16 : ℝ) = 2^4, by norm_num,
  have h8 : (8 : ℝ) = 2^3, by norm_num,
  -- Use these definitions in the expression and assert the expected result
  sorry

end evaluate_expression_l663_663122


namespace max_knights_between_knights_l663_663467

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l663_663467


namespace mirasol_account_balance_l663_663707

def initial_balance : ℕ := 50
def coffee_beans_cost : ℕ := 10
def tumbler_cost : ℕ := 30
def coffee_filter_cost : ℕ := 5
def tumbler_refund : ℕ := 20

def final_balance : ℕ :=
  initial_balance
  - (coffee_beans_cost + tumbler_cost + coffee_filter_cost)
  + tumbler_refund

theorem mirasol_account_balance :
  final_balance = 25 :=
by
  unfold final_balance initial_balance coffee_beans_cost tumbler_cost coffee_filter_cost tumbler_refund
  rw [Nat.add_assoc, Nat.add_comm 30, Nat.add_assoc 10, Nat.add_comm 10, Nat.add_assoc, Nat.add_comm 50]
  simp
  sorry

end mirasol_account_balance_l663_663707


namespace video_files_initial_l663_663448

-- Definitions derived from problem conditions
def m : Nat := 4  -- number of music files
def d : Nat := 23 -- number of deleted files
def r : Nat := 2  -- number of remaining files

-- Proof that Amy initially had 21 video files
theorem video_files_initial : 
  let f := d + r in       -- total initial files
  let v := f - m in       -- initial video files
  v = 21 := 
by
  let f := d + r
  let v := f - m
  sorry

end video_files_initial_l663_663448


namespace max_knights_between_other_knights_l663_663479

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l663_663479


namespace original_inhabitants_7200_l663_663844

noncomputable def original_inhabitants (X : ℝ) : Prop :=
  let initial_decrease := 0.9 * X
  let final_decrease := 0.75 * initial_decrease
  final_decrease = 4860

theorem original_inhabitants_7200 : ∃ X : ℝ, original_inhabitants X ∧ X = 7200 := by
  use 7200
  unfold original_inhabitants
  simp
  sorry

end original_inhabitants_7200_l663_663844


namespace cos_theta_plus_phi_l663_663955

theorem cos_theta_plus_phi (θ φ : ℝ) 
  (h1 : complex.exp (complex.I * θ) = (4 / 5) + (3 / 5) * complex.I)
  (h2 : complex.exp (complex.I * φ) = (-5 / 13) + (12 / 13) * complex.I) :
  real.cos (θ + φ) = -1 / 13 :=
by
  sorry

end cos_theta_plus_phi_l663_663955


namespace option_D_correct_l663_663030

theorem option_D_correct (a : ℝ) :
  3 * a ^ 2 - a ≠ 2 * a ∧
  a - (1 - 2 * a) ≠ a - 1 ∧
  -5 * (1 - a ^ 2) ≠ -5 - 5 * a ^ 2 ∧
  a ^ 3 + 7 * a ^ 3 - 5 * a ^ 3 = 3 * a ^ 3 :=
by
  sorry

end option_D_correct_l663_663030


namespace answer_part_a_answer_part_b_l663_663398

noncomputable def part_a : Prop :=
  ∃ (pockets : Finset (Finset ℕ)), (pockets.card = 3) ∧ 
  ((pockets.subsets.fst.val ∪ pockets.subsets.snd.val).sum = 70 ∧ 
   (pockets.subsets.snd.val ∪ pockets.subsets.snd.snd.val).sum = 70 ∧ 
   (pockets.subsets.snd.snd.val ∪ pockets.subsets.snd.snd.snd.val).sum = 70 ∧ 
   ((pockets.subsets.fst.val ∪ pockets.subsets.snd.val ∪ pockets.subsets.snd.snd.val ∪ pockets.subsets.snd.snd.snd.val).sum = 210)

noncomputable def part_b : Prop :=
  ∀ (pockets : Finset (Finset ℕ)), (pockets.card = 3) →
  ¬((pockets.subsets.fst.val ∪ pockets.subsets.snd.val).sum = 103 ∧ 
  (pockets.subsets.snd.val ∪ pockets.subsets.snd.snd.val).sum = 103 ∧ 
  (pockets.subsets.snd.snd.val ∪ pockets.subsets.snd.snd.snd.val).sum = 103 ∧ 
  ((pockets.subsets.fst.val ∪ pockets.subsets.snd.val ∪ pockets.subsets.snd.snd.val ∪ pockets.subsets.snd.snd.snd.val).sum = 310))

theorem answer_part_a : part_a :=
sorry

theorem answer_part_b : part_b :=
sorry

end answer_part_a_answer_part_b_l663_663398


namespace area_difference_l663_663742

theorem area_difference (d_square d_circle : ℝ) (diag_eq : d_square = 10) (diam_eq : d_circle = 10) : 
  let s := d_square / real.sqrt 2 in
  let area_square := s * s in
  let r := d_circle / 2 in
  let area_circle := real.pi * r * r in
  (area_circle - area_square) = 28.5 := by
s := d_square / real.sqrt 2
area_square := s * s
r := d_circle / 2
area_circle := real.pi * r * r
suffices : area_circle - area_square ≈ 28.5
by sorry

end area_difference_l663_663742


namespace cost_of_24_pounds_l663_663309

def cost_of_oranges (rate : ℕ) (weight : ℕ) : ℕ :=
  (rate * weight) / 8

theorem cost_of_24_pounds (h : cost_of_oranges 6 8 = 6) : cost_of_oranges 6 24 = 18 :=
by
  have ratio : 24 / 8 = 3 := rfl
  rw [cost_of_oranges, ratio, Nat.mul_div_cancel 18] 
  sorry

end cost_of_24_pounds_l663_663309


namespace solve_equation_l663_663323

theorem solve_equation (x a b : ℝ) (h : x^2 - 6*x + 11 = 27) (sol_a : a = 8) (sol_b : b = -2) :
  3 * a - 2 * b = 28 :=
by
  sorry

end solve_equation_l663_663323


namespace Anya_pancakes_l663_663452

theorem Anya_pancakes :
  ∀ (x : ℕ), x > 0 →
  let flipped := x * (2/3 : ℝ) in
  let non_burnt := flipped * 0.6 in
  let not_dropped := non_burnt * 0.8 in
  (not_dropped / x) * 100 = 32 :=
begin
  intro x,
  by_cases hx : x > 0,
  { -- We assume x > 0, hence the conditions hold for some positive number of pancakes.
    intro H,
    let flipped := x * (2/3 : ℝ),
    let non_burnt := flipped * 0.6,
    let not_dropped := non_burnt * 0.8,
    have h_flipped : flipped = x * (2/3 : ℝ), by refl,
    have h_non_burnt : non_burnt = flipped * 0.6, by refl,
    have h_not_dropped : not_dropped = non_burnt * 0.8, by refl,
    have h1 : flipped = x * (2/3 : ℝ) := by assumption,
    have h2 : non_burnt = x * (2/3 : ℝ) * 0.6, from calc
      non_burnt = (x * (2/3 : ℝ)) * 0.6 : by assumption
               ... = x * (2/3 : ℝ * 0.6) : by ring,
    have h3 : not_dropped = x * (2/3 : ℝ * 0.6) * 0.8, from calc
      not_dropped = (x * ((2/3 : ℝ) * 0.6)) * 0.8 : by assumption
                 ... = x * ((2/3 * 0.6) * 0.8) : by ring,
    have h4 : (not_dropped / x) * 100 = ((x * (2/3 * 0.6 * 0.8)) / x) * 100, by ring,
    have h5 : ((x * (2/3 * 0.6 * 0.8)) / x) = (2/3 * 0.6 * 0.8), by rw div_mul_cancel,
    rw [h4, h5],
    norm_num },
  { intro H, contradiction } -- x > 0 is not true
end.

end Anya_pancakes_l663_663452


namespace find_a10_l663_663604

noncomputable def ladder_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), (a (n + 3))^2 = a n * a (n + 6)

theorem find_a10 {a : ℕ → ℝ} (h1 : ladder_geometric_sequence a) 
(h2 : a 1 = 1) 
(h3 : a 4 = 2) : a 10 = 8 :=
sorry

end find_a10_l663_663604


namespace common_tangents_l663_663138

noncomputable theory

open Real

def circle_center (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

def O1 := circle_center 2 (-3) 2
def O2 := circle_center (-1) 1 3

theorem common_tangents : ∃ n : ℕ, n = 3 :=
by
  let d := sqrt ((2 + 1)^2 + (-3 - 1)^2)
  have h₀ : d = 5, sorry
  have h₁ : 2 + 3 = 5, sorry
  have h₂ : d = 2 + 3, from h₀.trans h₁.symm
  have h₃ : O1 ∩ O2 = ∅, sorry
  exact ⟨3, rfl⟩

end common_tangents_l663_663138


namespace represent_same_function_l663_663787

noncomputable def f1 (x : ℝ) : ℝ := (x^3 + x) / (x^2 + 1)
def f2 (x : ℝ) : ℝ := x

theorem represent_same_function : ∀ x : ℝ, f1 x = f2 x := 
by
  sorry

end represent_same_function_l663_663787


namespace product_of_roots_eq_25_l663_663542

theorem product_of_roots_eq_25 (t : ℝ) (h : t^2 - 10 * t + 25 = 0) : t * t = 25 :=
sorry

end product_of_roots_eq_25_l663_663542


namespace metallic_sheet_width_l663_663827

theorem metallic_sheet_width (length : ℝ) (side : ℝ) (volume : ℝ) (width : ℝ) :
  length = 48 → side = 8 → volume = 5120 → volume = (length - 2 * side) * (width - 2 * side) * side → width = 36 :=
by
  intro h_length h_side h_volume h_eq
  have h1 : length - 2 * side = 32 := by sorry
  have h2 : side = 8 := h_side
  have h3 : h_volume = (32) * (width - 16) * 8 := by sorry
  have h4 : width - 16 = 20 := by sorry
  show width = 36 from by sorry

end metallic_sheet_width_l663_663827


namespace value_of_n_l663_663254

-- Define required conditions
variables (n : ℕ) (f : ℕ → ℕ → ℕ)

-- Conditions
axiom cond1 : n > 7
axiom cond2 : ∀ m k : ℕ, f m k = 2^(n - m) * Nat.choose m k

-- Given condition
axiom after_seventh_round : f 7 5 = 42

-- Theorem to prove
theorem value_of_n : n = 8 :=
by
  -- Proof goes here
  sorry

end value_of_n_l663_663254


namespace area_bounded_curve_l663_663498

theorem area_bounded_curve : 
  (∫ x in (0:ℝ)..(2*sqrt 2), (x^2) * sqrt (8 - x^2)) = 4 * real.pi := by
  sorry

end area_bounded_curve_l663_663498


namespace identify_compound_l663_663346

def molecular_weight_of_element (symbol : String) : ℝ :=
  if symbol = "N" then 14.01 else -- Atomic weight of Nitrogen
  if symbol = "H" then 1.008 else  -- Atomic weight of Hydrogen
  0 -- For simplicity, assume unknowns have 0 weight.

def molecular_weight (compound : List (String × ℕ)) : ℝ :=
  compound.foldr (λ (pair : String × ℕ) acc, acc + (molecular_weight_of_element pair.fst) * pair.snd) 0

def is_ammonia (compound : List (String × ℕ)) : Prop :=
  compound = [("N", 1), ("H", 3)]

theorem identify_compound (compound : List (String × ℕ)) (hw : molecular_weight compound = 18) : is_ammonia compound :=
by
  sorry

end identify_compound_l663_663346


namespace collinear_c1_c2_l663_663798

def vec3 := (ℝ × ℝ × ℝ)

def a : vec3 := (8, 3, -1)
def b : vec3 := (4, 1, 3)

def c1 : vec3 := (2 * 8 - 4, 2 * 3 - 1, 2 * (-1) - 3) -- (12, 5, -5)
def c2 : vec3 := (2 * 4 - 4 * 8, 2 * 1 - 4 * 3, 2 * 3 - 4 * (-1)) -- (-24, -10, 10)

theorem collinear_c1_c2 : ∃ γ : ℝ, c1 = (γ * -24, γ * -10, γ * 10) :=
  sorry

end collinear_c1_c2_l663_663798


namespace max_knights_adjacent_to_two_other_knights_l663_663485

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l663_663485


namespace roy_sports_hours_l663_663317

theorem roy_sports_hours (total_days : ℕ) (missed_days : ℕ) (total_sports_hours : ℕ) (remaining_days : ℕ)
  (active_days_in_week : ∀ (total_days : ℕ) (missed_days : ℕ), remaining_days = total_days - missed_days)
  (sports_hours_per_day : ∀ (total_sports_hours : ℕ) (day_count : ℕ), day_count ≠ 0 → sports_hours_per_day = total_sports_hours / day_count) :
  total_days = 5 → missed_days = 2 → total_sports_hours = 6 → sports_hours_per_day = 2 :=
by
  intros h1 h2 h3
  have h4 : remaining_days = 5 - 2, from active_days_in_week 5 2
  have h5 : remaining_days = 3, by simp [h4]
  rw h5 at h3
  rw ←sports_hours_per_day 6 3 (by norm_num)
  norm_num


end roy_sports_hours_l663_663317


namespace distance_between_foci_of_ellipse_l663_663085

theorem distance_between_foci_of_ellipse :
  ∀ (ellipse: ℝ × ℝ → Prop),
    (∀ x y, ellipse (x, y) ↔ (x - 5)^2 / 25 + (y - 2)^2 / 4 = 1) →
    ∃ c : ℝ, 2 * c = 2 * Real.sqrt (25 - 4) :=
by
  intro ellipse h
  use Real.sqrt (25 - 4)
  sorry

end distance_between_foci_of_ellipse_l663_663085


namespace binomial_mod_500_remainder_l663_663861

theorem binomial_mod_500_remainder :
  let S := ∑ k in Finset.range(2024),
             if k % 3 = 0 then Nat.choose 2023 k else 0
  in S % 500 = 137 := by
    sorry

end binomial_mod_500_remainder_l663_663861


namespace car_speed_without_red_light_l663_663851

theorem car_speed_without_red_light (v : ℝ) :
  (∃ k : ℕ+, v = 10 / k) ↔ 
  ∀ (dist : ℝ) (green_duration red_duration total_cycle : ℝ),
    dist = 1500 ∧ green_duration = 90 ∧ red_duration = 60 ∧ total_cycle = 150 →
    v * total_cycle = dist / (green_duration + red_duration) := 
by
  sorry

end car_speed_without_red_light_l663_663851


namespace new_mean_is_70_l663_663278

-- Definitions from conditions
variable (numbers : Fin 25 → ℝ)
variable (avg : ℝ) (h_avg : avg = 40)
variable (n : ℕ) (h_n : n = 25)

-- The operations described
def transform (x : ℝ) := (x - 5) * 2

-- Problem statement: Given the original average and operations, prove the new mean is 70
theorem new_mean_is_70 (sum_org : ℝ) (h_sum : sum numbers = sum_org) (h_sum_org : sum_org = 1000) : 
    (sum (transform ∘ numbers) / n) = 70 := 
    sorry

end new_mean_is_70_l663_663278


namespace license_plate_count_l663_663201

-- Define the conditions as constants
def even_digit_count : Nat := 5
def consonant_count : Nat := 20
def vowel_count : Nat := 6

-- Define the problem as a theorem to prove
theorem license_plate_count : even_digit_count * consonant_count * vowel_count * consonant_count = 12000 := 
by
  -- The proof is not required, so we leave it as sorry
  sorry

end license_plate_count_l663_663201


namespace S15_eq_l663_663913

-- Definitions in terms of the geometric sequence and given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions given in the problem
axiom geom_seq (n : ℕ) : S n = (a 0) * (1 - (a 1) ^ n) / (1 - (a 1))
axiom S5_eq : S 5 = 10
axiom S10_eq : S 10 = 50

-- The problem statement to prove
theorem S15_eq : S 15 = 210 :=
by sorry

end S15_eq_l663_663913


namespace percentage_donated_l663_663831

def income : ℝ := 1200000
def children_percentage : ℝ := 0.20
def wife_percentage : ℝ := 0.30
def remaining : ℝ := income - (children_percentage * 3 * income + wife_percentage * income)
def left_amount : ℝ := 60000
def donated : ℝ := remaining - left_amount

theorem percentage_donated : (donated / remaining) * 100 = 50 := by
  sorry

end percentage_donated_l663_663831


namespace rectangle_area_l663_663796

theorem rectangle_area (l w : ℝ) (h₁ : (2 * l + 2 * w) = 46) (h₂ : (l^2 + w^2) = 289) : l * w = 120 :=
by
  sorry

end rectangle_area_l663_663796


namespace cannot_fold_patternD_to_cube_l663_663549

def patternA : Prop :=
  -- 5 squares arranged in a cross shape
  let squares := 5
  let shape  := "cross"
  squares = 5 ∧ shape = "cross"

def patternB : Prop :=
  -- 4 squares in a straight line
  let squares := 4
  let shape  := "line"
  squares = 4 ∧ shape = "line"

def patternC : Prop :=
  -- 3 squares in an L shape, and 2 squares attached to one end of the L making a T shape
  let squares := 5
  let shape  := "T"
  squares = 5 ∧ shape = "T"

def patternD : Prop :=
  -- 6 squares in a "+" shape with one extra square
  let squares := 7
  let shape  := "plus"
  squares = 7 ∧ shape = "plus"

theorem cannot_fold_patternD_to_cube :
  patternD → ¬ (patternA ∨ patternB ∨ patternC) :=
by
  sorry

end cannot_fold_patternD_to_cube_l663_663549


namespace complex_expression_result_l663_663935

-- Let \( Z = -2 + i \)
noncomputable def Z : ℂ := -2 + complex.I

-- Let \(\overline{Z}\) be the conjugate of Z
noncomputable def Z_conj : ℂ := conj Z

-- Define the expression to be proven
noncomputable def expr : ℂ := (Z * Z_conj) / (Z - Z_conj)

-- State the theorem that expr = -\(\frac{5}{2}i\)
theorem complex_expression_result : expr = -5 / 2 * complex.I := 
by sorry

end complex_expression_result_l663_663935


namespace part1_part2_part3_l663_663996

variable (a b c d S A B C D : ℝ)

-- The given conditions
def cond1 : Prop := a + c = b + d
def cond2 : Prop := A + C = B + D
def cond3 : Prop := S^2 = a * b * c * d

-- The statements to prove
theorem part1 (h1 : cond1 a b c d) (h2 : cond2 A B C D) : cond3 a b c d S := sorry
theorem part2 (h1 : cond1 a b c d) (h3 : cond3 a b c d S) : cond2 A B C D := sorry
theorem part3 (h2 : cond2 A B C D) : cond3 a b c d S := sorry

end part1_part2_part3_l663_663996


namespace triangle_intersect_sum_l663_663667

theorem triangle_intersect_sum (P Q R S T U : ℝ × ℝ) :
  P = (0, 8) →
  Q = (0, 0) →
  R = (10, 0) →
  S = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  T = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  ∃ U : ℝ × ℝ, 
    (U.1 = (P.1 + ((T.2 - P.2) / (T.1 - P.1)) * (U.1 - P.1)) ∧
     U.2 = (R.2 + ((S.2 - R.2) / (S.1 - R.1)) * (U.1 - R.1))) ∧
    (U.1 + U.2) = 6 :=
by
  sorry

end triangle_intersect_sum_l663_663667


namespace invalid_general_term_l663_663946

def is_general_term (u : ℕ → ℝ) : Prop :=
∀ n : ℕ, u n = if n % 2 = 0 then 1 else 0

theorem invalid_general_term :
  ¬ is_general_term (λ n, (1 / 2) * (1 + (-1:ℚ) ^ n)) :=
by {
  intro h,
  have h1 := h 1,
  simp at h1,
  have : (-1:ℚ) ^ 1 = -1
    by simp,
  rw [this, mul_add, mul_one, add_neg_self, mul_zero] at h1,
  have h2 : (1:ℚ) / 2 ≠ 1 by norm_num,
  contradiction,
  sorry
}

end invalid_general_term_l663_663946


namespace sum_of_coefficients_l663_663146

theorem sum_of_coefficients :
  let P := λ (x : ℝ), -3 * (x^7 - 2 * x^6 + x^4 - 3 * x^2 + 6) + 6 * (x^3 - 4 * x + 1) - 2 * (x^5 - 5 * x + 7)
  in P 1 = -27 :=
by
  sorry

end sum_of_coefficients_l663_663146


namespace distance_between_foci_of_ellipse_l663_663093

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end distance_between_foci_of_ellipse_l663_663093


namespace thirty_knocks_eq_eighty_knicks_l663_663643

-- Definitions based on the given conditions
def knicks := ℝ
def knacks := ℝ
def knocks := ℝ

-- Given conditions
constant k_to_kck : knicks = 3 / 10 * knacks
constant kc_to_knk : knacks = 5 / 4 * knocks

-- Theorem statement
theorem thirty_knocks_eq_eighty_knicks (k_to_kck : knicks = 3 / 10 * knacks) (kc_to_knk : knacks = 5 / 4 * knocks) : 
  30 * knocks = 80 * knicks :=
sorry

end thirty_knocks_eq_eighty_knicks_l663_663643


namespace percent_increase_is_correct_l663_663849

-- Define the initial side length and the growth factor
def initial_side_length : ℝ := 4
def growth_factor : ℝ := 1.25

-- Define the function to compute the side length of the nth triangle
def side_length (n : ℕ) : ℝ :=
  initial_side_length * (growth_factor ^ n)

-- Define the function to compute the perimeter of the nth triangle
def perimeter (n : ℕ) : ℝ :=
  3 * side_length n

-- Define the percentage increase in perimeter from the first to the fifth triangle
def percent_increase : ℝ :=
  (perimeter 4 - perimeter 0) / perimeter 0 * 100

-- State the theorem that the percent increase in perimeter from the first to the fifth triangle is approximately 144.1%
theorem percent_increase_is_correct :
  abs (percent_increase - 144.1) < 0.1 := by
  sorry

end percent_increase_is_correct_l663_663849


namespace percentage_pine_cones_on_roof_l663_663845

theorem percentage_pine_cones_on_roof 
  (num_trees : Nat) 
  (pine_cones_per_tree : Nat) 
  (pine_cone_weight_oz : Nat) 
  (total_pine_cone_weight_on_roof_oz : Nat) 
  : num_trees = 8 ∧ pine_cones_per_tree = 200 ∧ pine_cone_weight_oz = 4 ∧ total_pine_cone_weight_on_roof_oz = 1920 →
    (total_pine_cone_weight_on_roof_oz / pine_cone_weight_oz) / (num_trees * pine_cones_per_tree) * 100 = 30 := 
by
  sorry

end percentage_pine_cones_on_roof_l663_663845


namespace total_spokes_in_garage_l663_663854

def bicycle1_front_spokes : ℕ := 16
def bicycle1_back_spokes : ℕ := 18
def bicycle2_front_spokes : ℕ := 20
def bicycle2_back_spokes : ℕ := 22
def bicycle3_front_spokes : ℕ := 24
def bicycle3_back_spokes : ℕ := 26
def bicycle4_front_spokes : ℕ := 28
def bicycle4_back_spokes : ℕ := 30
def tricycle_front_spokes : ℕ := 32
def tricycle_middle_spokes : ℕ := 34
def tricycle_back_spokes : ℕ := 36

theorem total_spokes_in_garage :
  bicycle1_front_spokes + bicycle1_back_spokes +
  bicycle2_front_spokes + bicycle2_back_spokes +
  bicycle3_front_spokes + bicycle3_back_spokes +
  bicycle4_front_spokes + bicycle4_back_spokes +
  tricycle_front_spokes + tricycle_middle_spokes + tricycle_back_spokes = 286 :=
by
  sorry

end total_spokes_in_garage_l663_663854


namespace angle_sine_equivalence_l663_663678

theorem angle_sine_equivalence {A B C : ℝ} (h_triangle : A + B + C = 180) : 
  (A = B) ↔ (real.sin A = real.sin B) :=
sorry

end angle_sine_equivalence_l663_663678


namespace concurrent_lines_l663_663422

open Classical

variables (A B C D1 D2 E1 E2 F1 F2 L M N : Point)

-- Definitions required to spell out the conditions.
-- The points D1, D2, E1, E2, F1, F2 are points of intersection of a circle with the sides of the triangle ABC.
axiom circle_intersects_triangle (A B C D1 D2 E1 E2 F1 F2 : Point) : 
  Circle → (D1 ∈ side BC ∧ D2 ∈ side BC ∧ 
            E1 ∈ side CA ∧ E2 ∈ side CA ∧ 
            F1 ∈ side AB ∧ F2 ∈ side AB)

-- The points L, M, N are intersections of line segments.
axiom intersections (D1 D2 E1 E2 F1 F2 L M N : Point) :
  (L ∈ line D1E1 ∧ L ∈ line D2F2) ∧
  (M ∈ line E1F1 ∧ M ∈ line E2D2) ∧
  (N ∈ line F1D1 ∧ N ∈ line F2E2)

-- The main statement to prove the concurrency of lines.
noncomputable def are_concurrent (A B C L M N : Point) : Prop :=
  ∃ (O : Point), O ∈ line AL ∧ O ∈ line BM ∧ O ∈ line CN

theorem concurrent_lines 
  (A B C D1 D2 E1 E2 F1 F2 L M N : Point)
  (h1 : circle_intersects_triangle A B C D1 D2 E1 E2 F1 F2)
  (h2 : intersections D1 D2 E1 E2 F1 F2 L M N) :
  are_concurrent A B C L M N :=
  sorry

end concurrent_lines_l663_663422


namespace calc_305_squared_minus_295_squared_l663_663103

theorem calc_305_squared_minus_295_squared :
  305^2 - 295^2 = 6000 := 
  by
    sorry

end calc_305_squared_minus_295_squared_l663_663103


namespace sum_four_digit_integers_l663_663383

theorem sum_four_digit_integers : 
  (∑ k in finset.range (5000 - 1000 + 1), k + 1000) = 12003000 :=
by
  sorry

end sum_four_digit_integers_l663_663383


namespace log_sum_geometric_sequence_l663_663358

noncomputable def geometric_sequence_log_sum (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_a1_a8 : a 1 * a 8 = 9) : ℝ :=
  ∑ i in finset.range 8, real.log (a (i + 1)) / real.log 3

theorem log_sum_geometric_sequence (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_a1_a8 : a 1 * a 8 = 9) :
  geometric_sequence_log_sum a h_pos h_a1_a8 = 8 :=
sorry

end log_sum_geometric_sequence_l663_663358


namespace rectangle_midpoints_sum_l663_663433

theorem rectangle_midpoints_sum (A B C D M N O P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 3))
  (hD : D = (0, 3))
  (hM : M = (2, 0))
  (hN : N = (4, 1.5))
  (hO : O = (2, 3))
  (hP : P = (0, 1.5)) :
  (Real.sqrt ((2 - 0) ^ 2 + (0 - 0) ^ 2) + 
  Real.sqrt ((4 - 0) ^ 2 + (1.5 - 0) ^ 2) + 
  Real.sqrt ((2 - 0) ^ 2 + (3 - 0) ^ 2) + 
  Real.sqrt ((0 - 0) ^ 2 + (1.5 - 0) ^ 2)) = 11.38 :=
by
  sorry

end rectangle_midpoints_sum_l663_663433


namespace hyperbola_equation_line_equation_AB_l663_663193

-- Condition definitions
variable (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
variable (e : ℝ := sqrt 3)
variable (dist_left_focus_asymptote : ℝ := sqrt 2)
variable (c : ℝ := a * e)

-- Definitions of the hyperbola and necessary components
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def circle_passing_origin (A B : ℝ × ℝ) : Prop := 
  let ⟨x1, y1⟩ := A in let ⟨x2, y2⟩ := B in
  (x1^2 + y1^2) = (x2^2 + y2^2)

-- Theorem statement for part (1)
theorem hyperbola_equation :
  (b^2 = 2 * a^2) →
  ∃ a b : ℝ, 
  ∀ x y : ℝ, hyperbola x y ↔ (x^2 / 1) - (y^2 / 2) = 1 := sorry

-- Theorem statement for part (2)
theorem line_equation_AB :
  ∀ A B : ℝ × ℝ, 
  (A.1 = 2 ∧ A.2 = 0) →
  intersection_points A B (hyperbola) →
  circle_passing_origin A B →
  ∃ m : ℝ, 
  (m = 1 ∨ m = -1) ∧
  ∀ x y : ℝ, (y = m * x - 2) : sorry

end hyperbola_equation_line_equation_AB_l663_663193


namespace fermat_numbers_pairwise_coprime_l663_663378

theorem fermat_numbers_pairwise_coprime :
  ∀ i j : ℕ, i ≠ j → Nat.gcd (2 ^ (2 ^ i) + 1) (2 ^ (2 ^ j) + 1) = 1 :=
sorry

end fermat_numbers_pairwise_coprime_l663_663378


namespace max_knights_adjacent_to_two_other_knights_l663_663482

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l663_663482


namespace chloe_winter_clothing_l663_663859

theorem chloe_winter_clothing (boxes scarves mittens per_box total : ℕ) 
  (h_boxes : boxes = 4)
  (h_scarves : scarves = 2)
  (h_mittens : mittens = 6)
  (h_per_box : per_box = scarves + mittens)
  (h_total : total = boxes * per_box) : 
  total = 32 := by
  rw [h_boxes, h_scarves, h_mittens, h_per_box] at h_total
  exact h_total

#eval chloe_winter_clothing 4 2 6 8 32 rfl rfl rfl rfl rfl

end chloe_winter_clothing_l663_663859


namespace consecutive_integers_sum_l663_663236

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l663_663236


namespace intersection_is_23_l663_663963

open Set

def setA : Set ℤ := {1, 2, 3, 4}
def setB : Set ℤ := {x | 2 ≤ x ∧ x ≤ 3}

theorem intersection_is_23 : setA ∩ setB = {2, 3} := 
by 
  sorry

end intersection_is_23_l663_663963


namespace nine_point_five_minutes_in_seconds_l663_663952

-- Define the number of seconds in one minute
def seconds_per_minute : ℝ := 60

-- Define the function to convert minutes to seconds
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * seconds_per_minute

-- Define the theorem to prove
theorem nine_point_five_minutes_in_seconds : minutes_to_seconds 9.5 = 570 :=
by
  sorry

end nine_point_five_minutes_in_seconds_l663_663952


namespace bushes_needed_for_circular_patio_l663_663648

noncomputable def number_of_bushes_needed (radius : ℝ) (spacing : ℝ) : ℝ :=
  (2 * Real.pi * radius) / spacing

theorem bushes_needed_for_circular_patio :
  number_of_bushes_needed 15 0.5 ≈ 188 :=
by
  have bush_count : ℝ := number_of_bushes_needed 15 0.5
  have approx_pi : Real.pi ≈ 3.14159 := by sorry -- approximation of π
  have correct_bush_count : bush_count ≈ 60 * 3.14159 := by sorry -- substitution and multiplication
  have final_count : bush_count ≈ 188 := by -- rounding
    sorry
  exact final_count

end bushes_needed_for_circular_patio_l663_663648


namespace area_of_ellipse_l663_663130

theorem area_of_ellipse : 
  ∀ (x y : ℝ), 4 * x^2 + 12 * x + 9 * y^2 + 27 * y + 36 = 0 → 
  (area : ℝ) (h : area = π * (3 / 2) * 1), 
  area = 3 * π / 2 := 
by
  sorry

end area_of_ellipse_l663_663130


namespace find_cos_C_l663_663126

theorem find_cos_C (A B C : Type) [metric_space A] [measurable_space B] [normed_group B] [char_zero C] 
  (h_right_triangle : metric_space.is_right_triangle (A, B, C))
  (h_AB_15 : metric_space.dist A B = 15)
  (h_AC_10 : metric_space.dist A C = 10) :
  real.cos C = 3 * real.sqrt 325 / 65 :=
sorry

end find_cos_C_l663_663126


namespace num_fixed_point_functions_eq_nineteen_l663_663911

/-- Given \( f: A_{3} \rightarrow A_{3} \) where \( A_{3} \) is the set with elements {1, 2, 3}, 
prove that the number of functions \( f \) such that \( f^{(3)} = f \) is 19. -/
def countFixedPointFunctions (f : Fin 3 → Fin 3) (hf : ∀ x, (f^(3) x) = f x) : Nat := 19

-- We need to prove that the number of such functions is exactly 19.
theorem num_fixed_point_functions_eq_nineteen : countFixedPointFunctions = 19 := 
sorry

end num_fixed_point_functions_eq_nineteen_l663_663911


namespace find_matrix_N_l663_663888

def mul_matrix (m1 m2 : Matrix (Fin 2) (Fin 2) ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  λ i j, ∑ k, m1 i k * m2 k j

theorem find_matrix_N :
  ∀ (a b c d : ℚ),
    mul_matrix !![![1, 0], ![0, 4]] !![![a, b], ![c, d]] = !![![a, b], ![4 * c, 4 * d]] :=
by sorry

end find_matrix_N_l663_663888


namespace even_divisors_of_8_factorial_l663_663630

theorem even_divisors_of_8_factorial :
  ∀ n : ℕ, n = 8 → (factorial n = 2^7 * 3^2 * 5 * 7) → (∃ d : ℕ, d = 84) :=
by
  intros n hn hfactorial
  have h8 : 8 = n := hn
  existsi 84
  sorry -- proof will be constructed here

end even_divisors_of_8_factorial_l663_663630


namespace area_difference_correct_l663_663736

-- Definitions for given conditions
def square_diagonal : ℝ := 10
def circle_diameter : ℝ := 10
def pi_approx : ℝ := 3.14159

-- Symbolic calculations
def square_side (d : ℝ) : ℝ := (d^2 / 2).sqrt
def square_area (s : ℝ) : ℝ := s^2
def circle_radius (d : ℝ) : ℝ := d / 2
def circle_area (r : ℝ) (π : ℝ) : ℝ := π * r^2
def area_difference (circle_area : ℝ) (square_area : ℝ) : ℝ := circle_area - square_area

theorem area_difference_correct:
  area_difference (circle_area (circle_radius circle_diameter) pi_approx) (square_area (square_side square_diagonal)) = 28.5 :=
by sorry

end area_difference_correct_l663_663736


namespace prove_value_of_expression_l663_663427

def cube_painted_red_and_cut :=
  (a b c d : ℕ) ×
  (a = 1) ×
  (b = 6) ×
  (c = 12) ×
  (d = 8)

theorem prove_value_of_expression (h : cube_painted_red_and_cut) :
  h.fst.1 - h.fst.2 - h.fst.3 + h.fst.4 = -9 :=
by
  obtain ⟨a, b, c, d, ha, hb, hc, hd⟩ := h
  -- Proof steps are skipped
  sorry

end prove_value_of_expression_l663_663427


namespace kitten_positions_minimal_distance_l663_663762

theorem kitten_positions_minimal_distance :
  ∃ P : Set (ℝ × ℝ), 
    P = {(-3*Real.sqrt 3, 3), (3, 3*Real.sqrt 3), (3*Real.sqrt 3, -3), (-3, -3*Real.sqrt 3)} ∧
    ∀ (x y : ℝ), 
      (∃ t : ℝ, t ≥ 0 ∧
       (x = -6 * Real.cos t ∧ y = -6 * Real.sin t) ∧
       let α := Real.pi in
       let β := Real.pi / 3 in
       let ω_kitten := (α - β) / 4 in
       let ω_puppy := 2 * ω_kitten in
       ∃ (θ : ℝ), 
         θ = t ∧
         (2 * (x, y) = (2 * Real.cos (ω_puppy * θ) - 6 * Real.cos (ω_kitten * θ), 2 * Real.sin (ω_puppy * θ) - 6 * Real.sin (ω_kitten * θ)) ∧
          (x, y) ∈ P)) :=
begin
  sorry
end

end kitten_positions_minimal_distance_l663_663762


namespace distance_walked_on_third_day_l663_663261

theorem distance_walked_on_third_day:
  ∃ x : ℝ, 
    4 * x + 2 * x + x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 378 ∧
    x = 48 := 
by
  sorry

end distance_walked_on_third_day_l663_663261


namespace range_of_a_l663_663609

theorem range_of_a (a : ℝ) :
  (∃! x, 1 < x ∧ x < 3 ∧ log (x - 1) + log (3 - x) = log (x - a)) ↔ (3 / 4 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_l663_663609


namespace q1_geom_seq_and_general_term_q2_exists_n_such_that_a_n_greater_1_l663_663917

theorem q1_geom_seq_and_general_term (λ : ℝ) (hλ : λ > 0) :
  (∀ n : ℕ+, ∃ a : ℕ+ → ℝ, (∀ n, a n > 0) ∧ a 1 = 1/2 ∧
    (∀ n, a (n + 1) = a n + λ * a n^2) ∧ λ = 1/(a (n + 1))
    → ∃ r : ℝ, r = (1 + sqrt 5) / 2 ∧ 
      (∀ n, a (n + 1) / a n = r) ∧
      (∀ n, a n = 1/2 * r^(n - 1))) :=
sorry

theorem q2_exists_n_such_that_a_n_greater_1 (λ : ℝ) (hλ : λ = 1 / 2016) :
  (∃ (n : ℕ+), ∀ a : ℕ+ → ℝ, (∀ n, a n > 0) ∧ a 1 = 1/2 ∧
    (∀ n, a (n + 1) = a n + λ * a n^2) ∧
    a n > 1 ∧ n = 2018) :=
sorry

end q1_geom_seq_and_general_term_q2_exists_n_such_that_a_n_greater_1_l663_663917


namespace old_model_car_consumption_l663_663829

noncomputable def proof_old_model_consumption (x : ℝ) : Prop :=
  let new_model_consumption := x - 2 in
  let old_model_distance_per_liter := 100 / x in
  let new_model_distance_per_liter := 100 / new_model_consumption in
  new_model_distance_per_liter = old_model_distance_per_liter + 4.2

theorem old_model_car_consumption : ∃ x : ℝ, proof_old_model_consumption x ∧ x = 7.97 :=
sorry

end old_model_car_consumption_l663_663829


namespace frosting_cupcakes_l663_663101

noncomputable def time_in_seconds (minutes : Nat) : Nat := minutes * 60

def frosting_rate (seconds_per_cupcake : ℝ) : ℝ := 1 / seconds_per_cupcake

theorem frosting_cupcakes 
  (cagney_rate lacey_rate : ℝ)
  (work_time break_time total_time : ℕ)
  (h_cagney : cagney_rate = frosting_rate 25)
  (h_lacey : lacey_rate = frosting_rate 35)
  (h_work_time : work_time = time_in_seconds 6)
  (h_break_time : break_time = time_in_seconds 1)
  (h_total_time : total_time = work_time + break_time) :
  let combined_rate := cagney_rate + lacey_rate
  let frosting_by_both := combined_rate * work_time
  let frosting_by_cagney_alone := cagney_rate * break_time
  frosting_by_both + frosting_by_cagney_alone = 26 := by
{
  unfold frosting_rate,
  rw [h_cagney, h_lacey, h_work_time, h_break_time, h_total_time],
  have h_combined_rate : combined_rate = 1 / 25 + 1 / 35 := by
  {
    rw [h_cagney, h_lacey],
  },
  have h_total_cupcakes : frosting_by_both + frosting_by_cagney_alone = (1 / 25 + 1 / 35) * 360 + (1 / 25) * 60 := by
  {
    rw [h_combined_rate],
    norm_num,
  },
  exact h_total_cupcakes,
}

end frosting_cupcakes_l663_663101


namespace number_of_interesting_subsets_l663_663961

open Finset

-- Define the set S and the predicate "interesting"
def S : Finset ℕ := range 31 \ {0}

def is_interesting (s : Finset ℕ) : Prop :=
  s.card = 3 ∧ 8 ∣ s.prod id

-- Define the main statement
theorem number_of_interesting_subsets :
  (S.subsets.filter is_interesting).card = 1223 :=
sorry -- Proof not required

end number_of_interesting_subsets_l663_663961


namespace solve_equation_l663_663324

theorem solve_equation (x : ℝ) (h : 
  ∑ k in Finset.range 9, 1 / ((x + k : ℝ) * (x + k + 1)) +
  1 / (x + 10) = 2 / 5) : x = 3 / 2 :=
  sorry

end solve_equation_l663_663324


namespace smallest_FGJ_l663_663608

-- Definitions of the conditions in Lean 4
def unique_digits (n : ℕ) : Prop := 
  -- Convert the number to a list of digits and check uniqueness
  (n.digits 10).nodup

def no_digit_is_nine (n : ℕ) : Prop := 
  ∀ d ∈ n.digits 10, d ≠ 9

def no_leading_zero (n : ℕ) : Prop :=
  ∀ m, (n = m * 10 ^ ((n.digits 10).length - 1) + n % 10 ^ ((n.digits 10).length - 1)) → m ≠ 0

def satisfies_cryptarithm (AB BC DE FGJ : ℕ) : Prop :=
  AB + BC + DE = FGJ

-- Problem statement to be proven in Lean 4
theorem smallest_FGJ (AB BC DE FGJ : ℕ) 
  (h1 : unique_digits AB) (h2 : unique_digits BC) (h3 : unique_digits DE) (h4 : unique_digits FGJ)
  (h5 : no_digit_is_nine AB) (h6 : no_digit_is_nine BC) (h7 : no_digit_is_nine DE) (h8 : no_digit_is_nine FGJ)
  (h9 : no_leading_zero AB) (h10 : no_leading_zero BC) (h11 : no_leading_zero DE) (h12 : no_leading_zero FGJ)
  (h : satisfies_cryptarithm AB BC DE FGJ) : 
  FGJ = 108 :=
begin
  sorry,
end

end smallest_FGJ_l663_663608


namespace find_P_l663_663339

theorem find_P 
  (digits : Finset ℕ) 
  (h_digits : digits = {1, 2, 3, 4, 5, 6}) 
  (P Q R S T U : ℕ)
  (h_unique : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits ∧ U ∈ digits ∧ 
              P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
              Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
              R ≠ S ∧ R ≠ T ∧ R ≠ U ∧ 
              S ≠ T ∧ S ≠ U ∧ 
              T ≠ U) 
  (h_div5 : (100 * P + 10 * Q + R) % 5 = 0)
  (h_div3 : (100 * Q + 10 * R + S) % 3 = 0)
  (h_div2 : (100 * R + 10 * S + T) % 2 = 0) :
  P = 2 :=
sorry

end find_P_l663_663339


namespace area_difference_correct_l663_663735

-- Definitions for given conditions
def square_diagonal : ℝ := 10
def circle_diameter : ℝ := 10
def pi_approx : ℝ := 3.14159

-- Symbolic calculations
def square_side (d : ℝ) : ℝ := (d^2 / 2).sqrt
def square_area (s : ℝ) : ℝ := s^2
def circle_radius (d : ℝ) : ℝ := d / 2
def circle_area (r : ℝ) (π : ℝ) : ℝ := π * r^2
def area_difference (circle_area : ℝ) (square_area : ℝ) : ℝ := circle_area - square_area

theorem area_difference_correct:
  area_difference (circle_area (circle_radius circle_diameter) pi_approx) (square_area (square_side square_diagonal)) = 28.5 :=
by sorry

end area_difference_correct_l663_663735


namespace relationship_y1_y2_y3_l663_663968

open Classical
noncomputable theory

-- Definitions of the quadratic function and points
def quadratic (x : ℝ) (p q : ℝ) : ℝ := -5 * x^2 + p * x + q

variables {a b p q y1 y2 y3 : ℝ}

-- Points on the graph
def A := (a, b)
def B := (0, y1)
def C := (4 - a, b)
def D := (1, y2)
def E := (4, y3)

-- Lean 4 statement to represent the proof problem
theorem relationship_y1_y2_y3 :
  (quadratic 0 p q = y1) ∧
  (quadratic 1 p q = y2) ∧
  (quadratic 4 p q = y3) ∧
  (quadratic a p q = b) ∧
  (quadratic (4 - a) p q = b) →
  y1 = y3 ∧ y1 < y2 :=
by {
  assume h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h_rest,
  cases h_rest with ha hc,
  sorry
}

end relationship_y1_y2_y3_l663_663968


namespace largest_minus_second_largest_is_one_l663_663108

open Finset

variable (S : Finset ℕ)
variable (hS : S = {10, 11, 12, 13})

theorem largest_minus_second_largest_is_one :
  S.max' (by sorry) - S.erase (S.max' (by sorry)).max' (by sorry) = 1 :=
by
  rw hS
  sorry

end largest_minus_second_largest_is_one_l663_663108


namespace sum_of_consecutive_integers_l663_663224

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l663_663224


namespace E_union_F_eq_univ_l663_663619

-- Define the given conditions
def E : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def F (a : ℝ) : Set ℝ := { x | x - 5 < a }
def I : Set ℝ := Set.univ
axiom a_gt_6 : ∃ a : ℝ, a > 6 ∧ 11 ∈ F a

-- State the theorem
theorem E_union_F_eq_univ (a : ℝ) (h₁ : a > 6) (h₂ : 11 ∈ F a) : E ∪ F a = I := by
  sorry

end E_union_F_eq_univ_l663_663619


namespace max_knights_between_knights_l663_663453

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l663_663453


namespace chess_winning_theorem_l663_663431

noncomputable def winning_strategy : Prop :=
  let O := (0, 0)   -- The center of the chessboard
  let first_player_wins := ∀ (initial_position : ℕ × ℕ), -- piece placed initially at some square
    ∃ (strategy : ℕ → ℕ × ℕ), -- Strategy function determining next position based on previous moves
      (∀ n, -- For all moves n
        (∥ strategy n - O ∥ > ∥ strategy (n - 1) - O ∥) ∧ -- The distance condition
        ((∥ strategy (n - 1) - O ∥ * k < ∥ strategy n - O ∥ * k) ∧ k > 1) -- Distance increases strictly for k > 1
      ) → 
        (∀ nmoves, -- Number of moves made
          first_player_wins_with_strategy (initial_position strategy nmoves)  -- outcome
        )
  first_player_wins

theorem chess_winning_theorem : winning_strategy :=
  sorry

end chess_winning_theorem_l663_663431


namespace plane_fuel_consumption_rate_l663_663834

/-- A plane has 6.3333 gallons of fuel left and can continue to fly for 0.6667 hours.
    Prove that the rate of fuel consumption per hour is approximately 9.5 gallons per hour. -/
theorem plane_fuel_consumption_rate :
  let fuel_left := 6.3333
  let time_left_to_fly := 0.6667
  let rate_of_fuel_consumption_per_hour := fuel_left / time_left_to_fly
  abs (rate_of_fuel_consumption_per_hour - 9.5) < 0.01 :=
by {
  let fuel_left := 6.3333
  let time_left_to_fly := 0.6667
  let rate_of_fuel_consumption_per_hour := fuel_left / time_left_to_fly
  show abs (rate_of_fuel_consumption_per_hour - 9.5) < 0.01,
  apply sorry
}

end plane_fuel_consumption_rate_l663_663834


namespace simplify_expression_l663_663726

theorem simplify_expression (p q x : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : x > 0) (h₃ : x ≠ 1) :
  (x^(3 / p) - x^(3 / q)) / ((x^(1 / p) + x^(1 / q))^2 - 2 * x^(1 / q) * (x^(1 / q) + x^(1 / p)))
  + x^(1 / p) / (x^((q - p) / (p * q)) + 1) = x^(1 / p) + x^(1 / q) := 
sorry

end simplify_expression_l663_663726


namespace option_c_correct_l663_663903

-- Define the conditions
def quantity_exceeds_two (products : Type) (qualified : set products) (defective : set products) : Prop :=
  2 < qualified.card ∧ 2 < defective.card

def at_least_one_defective (selected : Finset products) (defective : set products) : Prop :=
  ∃ x ∈ selected, x ∈ defective

def all_qualified (selected : Finset products) (qualified : set products) : Prop :=
  ∀ x ∈ selected, x ∈ qualified

-- Define mutually exclusive and complementary events
def mutually_exclusive (event1 event2 : Prop) : Prop :=
  event1 → ¬ event2

def complementary (event1 event2 : Prop) : Prop :=
  mutually_exclusive event1 event2 ∧ (event1 ∨ event2)

-- Given the problem's conditions and definitions, state the final assertion
theorem option_c_correct
  (products : Type)
  (qualified defective : set products)
  (selected : Finset products)
  (h : quantity_exceeds_two products qualified defective)
  (h_selected : selected.card = 2) :
  complementary (at_least_one_defective selected defective) (all_qualified selected qualified) :=
sorry

end option_c_correct_l663_663903


namespace min_y_value_l663_663541

theorem min_y_value (x : ℝ) : 
  (∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ 12 ∧ (y = 12 ↔ x = -1)) :=
sorry

end min_y_value_l663_663541


namespace fill_3x3_grid_with_conditions_l663_663632

theorem fill_3x3_grid_with_conditions :
  ∃ (arr : array3x3 ℕ), 
  (∀ i : fin 3, ∑ j, arr i j = 7) ∧
  (∀ j : fin 3, ∑ i, arr i j = 7) ∧
  (∀ i : fin 3, ∀ j₁ j₂ : fin 3, j₁ ≠ j₂ → arr i j₁ ≠ arr i j₂ ∧ arr i j₁ ≠ 0) ∧
  (∀ j : fin 3, ∀ i₁ i₂ : fin 3, i₁ ≠ i₂ → arr i₁ j ≠ arr i₂ j ∧ arr i₁ j ≠ 0) :=
  sorry

end fill_3x3_grid_with_conditions_l663_663632


namespace find_remainders_l663_663296

theorem find_remainders (x : ℤ) :
  (x ≡ 25 [MOD 35]) → (x ≡ 31 [MOD 42]) → 
  ((x ≡ 10 [MOD 15]) ∧ (x ≡ 13 [MOD 18])) :=
by
  intro h1 h2
  have a : x % 15 = 10 := sorry
  have b : x % 18 = 13 := sorry
  exact ⟨a,b⟩

end find_remainders_l663_663296


namespace distance_between_foci_of_ellipse_l663_663092

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end distance_between_foci_of_ellipse_l663_663092


namespace probability_two_people_between_l663_663784

theorem probability_two_people_between (total_people : ℕ) (favorable_arrangements : ℕ) (total_arrangements : ℕ) :
  total_people = 6 ∧ favorable_arrangements = 144 ∧ total_arrangements = 720 →
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 5 :=
by
  intros h
  -- We substitute the given conditions
  have ht : total_people = 6 := h.1
  have hf : favorable_arrangements = 144 := h.2.1
  have ha : total_arrangements = 720 := h.2.2
  -- We need to calculate the probability considering the favorable and total arrangements
  sorry

end probability_two_people_between_l663_663784


namespace x_equals_y_l663_663503

-- Conditions
def x := 2 * 20212021 * 1011 * 202320232023
def y := 43 * 47 * 20232023 * 202220222022

-- Proof statement
theorem x_equals_y : x = y := sorry

end x_equals_y_l663_663503


namespace count_n_for_roots_as_consecutive_integers_l663_663950

theorem count_n_for_roots_as_consecutive_integers :
  let n_candidates := { n : ℕ | ∃ (r : ℕ), n = 2 * r + 1 ∧ r < 75 } in
  let m_divisible_by_4 := { n ∈ n_candidates | ∃ (r : ℕ), n = 2 * r + 1 ∧ m = r * (r + 1) ∧ 4 ∣ m } in
  ∃ count : ℕ, count = 37 ∧ count = m_divisible_by_4.card :=
by
  sorry

end count_n_for_roots_as_consecutive_integers_l663_663950


namespace validate_statements_l663_663567

open Nat

variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom a_seq : ∀ n : ℕ, (∑ k in finset.range (n + 1), (2 * k + 1) * a (k + 1)) = 2 * (n + 1)
axiom b_defn : ∀ n, b n = a n / (2 * n + 1)
axiom S_defn : ∀ n, S n = ∑ k in finset.range n, b (k + 1)

theorem validate_statements :
  (a 1 = 2) ∧
  (∀ n, n > 0 → a n ≠ 2 / (2 * n + 1)) ∧
  (∀ n, S n = 2 * n / (2 * n + 1)) ∧
  (∀ n m, n < m → a n > a m) :=
by
  sorry

end validate_statements_l663_663567


namespace max_knights_between_other_knights_l663_663475

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l663_663475


namespace parabola_symmetric_point_l663_663915

theorem parabola_symmetric_point (p : ℝ) (hp : 0 < p) : 
  ∃ x_0 y_0 : ℝ, y_0^2 = 2 * p * x_0 ∧
  y_0 = -real.sqrt 3 * (x_0 - 5) ∧
  x_0 = 3 :=
begin
  sorry
end

end parabola_symmetric_point_l663_663915


namespace possible_integer_roots_l663_663338

theorem possible_integer_roots 
  (a b c d e : ℤ)
  (h_consecutive : ∃ s r : ℤ, b = a + s ∧ c = a + 2 * s ∧ d = a + 3 * s ∧ e = a + 4 * s)
  (h_poly : ∀ x : ℤ, (x ^ 5 + a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e = 0 → ∃ k, k ≤ 5 ∧ x = k)) :
  ∃ m : ℕ, m ∈ ({0, 1, 2, 3, 5} : set ℕ) :=
sorry

end possible_integer_roots_l663_663338


namespace shoe_more_than_shirt_l663_663319

def shirt_cost := 7
def num_shirts := 2
def shoe_cost : ℕ -- We'll solve for this
def bag_cost := (2 * num_shirts * shirt_cost + shoe_cost) / 2
def total_cost := (2 * num_shirts * shirt_cost + shoe_cost + bag_cost)

theorem shoe_more_than_shirt
  (H_total_cost : total_cost = 36) :
  shoe_cost - shirt_cost = 3 :=
by
  -- Formalize the given information and conditions
  let h_shirts := 2 * shirt_cost
  let h_bag := (h_shirts + shoe_cost) / 2
  let h_total := h_shirts + shoe_cost + h_bag
  have : h_total = 36 := by exact H_total_cost
  sorry

end shoe_more_than_shirt_l663_663319


namespace integer_values_of_a_l663_663899

theorem integer_values_of_a :
  {a : ℤ // ∀ x : ℤ, x * x + a * x + a * a = 0 → ∃ x₁ x₂ : ℤ, x₁ * x₁ + a * x₁ + a * a = 0 ∧ x₂ * x₂ + a * x₂ + a * a = 0 ∧ x₁ = x₂} = {0} := sorry

end integer_values_of_a_l663_663899


namespace find_value_of_a_l663_663916

theorem find_value_of_a (α : ℝ) (a : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (-4, a))
  (h : (Real.sin α) * (Real.cos α) = (√3) / 4) :
  (a = -4 * √3 ∨ a = -4 * √3 / 3) :=
by
  sorry

end find_value_of_a_l663_663916


namespace total_volume_of_five_cubes_l663_663783

-- Definition for volume of a cube function
def volume_of_cube (edge_length : ℝ) : ℝ :=
  edge_length ^ 3

-- Conditions
def edge_length : ℝ := 5
def number_of_cubes : ℝ := 5

-- Proof statement
theorem total_volume_of_five_cubes : 
  volume_of_cube edge_length * number_of_cubes = 625 := 
by
  sorry

end total_volume_of_five_cubes_l663_663783


namespace real_roots_for_all_K_l663_663548

theorem real_roots_for_all_K (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x-1) * (x-2) + 2 * x :=
sorry

end real_roots_for_all_K_l663_663548


namespace product_of_a_values_l663_663752

theorem product_of_a_values : ∀ a : ℝ,
  dist (3 * a, 2 * a - 3) (5, 1) = Real.sqrt 34 →
  (3 * a * 13 - 23)^2 - (3 * sqrt 73 ^2) * (3 * a * 13 - 23)^2 - 3 * sqrt 73 = -722 / 169 :=
by sorry

end product_of_a_values_l663_663752


namespace bottles_per_day_l663_663508

theorem bottles_per_day (b d : ℕ) (h1 : b = 8066) (h2 : d = 74) : b / d = 109 :=
by {
  sorry
}

end bottles_per_day_l663_663508


namespace sum_of_consecutive_integers_l663_663225

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l663_663225


namespace non_positive_sequence_l663_663283

theorem non_positive_sequence
  (N : ℕ)
  (a : ℕ → ℝ)
  (h₀ : a 0 = 0)
  (hN : a N = 0)
  (h_rec : ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2) :
  ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 := sorry

end non_positive_sequence_l663_663283


namespace number_of_ways_to_choose_books_l663_663633

def num_books := 15
def books_to_choose := 3

theorem number_of_ways_to_choose_books : Nat.choose num_books books_to_choose = 455 := by
  sorry

end number_of_ways_to_choose_books_l663_663633


namespace frog_reaches_top_l663_663053

theorem frog_reaches_top (x : ℕ) (h1 : ∀ d ≤ x - 1, 3 * d + 5 ≥ 50) : x = 16 := by
  sorry

end frog_reaches_top_l663_663053


namespace shift_cos_graph_l663_663768

theorem shift_cos_graph :
  ∀ (x : ℝ), cos (2 * (x - (π / 8))) = cos (2 * x - π / 4) → cos (2 * (x - π / 8) + 2 * (π / 8)) = cos (2 * x) :=
by
  -- Define the problem setup
  assume x h1,
  -- Apply the given condition
  have h2 : 2 * (x - π / 8) + 2 * (π / 8) = 2 * x,
  { sorry },
  -- Conclude the proof using provided steps
  rw h2,
  rw h1

end shift_cos_graph_l663_663768


namespace percentage_gain_is_correct_l663_663830

noncomputable def num_bowls_purchased : ℕ := 115
noncomputable def cost_per_bowl : ℝ := 18
noncomputable def num_bowls_sold : ℕ := 104
noncomputable def selling_price_per_bowl : ℝ := 20

theorem percentage_gain_is_correct :
  let total_cost := num_bowls_purchased * cost_per_bowl
  let total_selling_price := num_bowls_sold * selling_price_per_bowl
  let gain := total_selling_price - total_cost
  let percentage_gain := (gain / total_cost) * 100
  percentage_gain ≈ 0.483 := 
by 
  sorry

end percentage_gain_is_correct_l663_663830


namespace find_T_2023_floor_l663_663160

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := 3 * a (n+1) - 2 * a n + 1

-- Define the sum T_n
def T : ℕ → ℚ
| 0     := 0
| (n+1) := T n + 8 / (a (n + 1))

-- The proof goal: finding the floor of T_2023
theorem find_T_2023_floor : floor (T 2023) = 14 :=
sorry

end find_T_2023_floor_l663_663160


namespace ratio_geometric_sequence_of_arithmetic_l663_663919

variable {d : ℤ}
variable {a : ℕ → ℤ}

-- definition of an arithmetic sequence with common difference d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- definition of a geometric sequence for a_5, a_9, a_{15}
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 9 * a 9 = a 5 * a 15

theorem ratio_geometric_sequence_of_arithmetic
  (h_arith : arithmetic_sequence a d) (h_nonzero : d ≠ 0) (h_geom : geometric_sequence a) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end ratio_geometric_sequence_of_arithmetic_l663_663919


namespace vectors_collinear_if_magnitude_diff_eq_sum_l663_663198

variable {V : Type} [AddCommGroup V] [VectorSpace ℝ V]
variable (a b : V)

theorem vectors_collinear_if_magnitude_diff_eq_sum (h : ∥a - b∥ = ∥a∥ + ∥b∥) : 
  ∃ (λ : ℝ), a = λ • b := 
sorry

end vectors_collinear_if_magnitude_diff_eq_sum_l663_663198


namespace number_of_solutions_l663_663142

open Real

-- Define the sign function
def sign (a : ℝ) : ℝ := 
  if a > 0 then 1 
  else if a = 0 then 0 
  else -1

-- Define conditions
def conditions (x y z : ℝ) : Prop :=
  x = 2020 - 2021 * (sign (2 * y + 2 * z)) ∧
  y = 2020 - 2021 * (sign (2 * x + 2 * z)) ∧
  z = 2020 - 2021 * (sign (2 * x + 2 * y))

-- State the theorem
theorem number_of_solutions : 
  (∃ x y z : ℝ, conditions x y z) ∧ 
  (∀ x y z : ℝ, conditions x y z → 
    (x = 4041 ∧ y = -1 ∧ z = -1) ∨ 
    (x = -1 ∧ y = 4041 ∧ z = -1) ∨ 
    (x = -1 ∧ y = -1 ∧ z = 4041)) ∧
  (∀ x' y' z' : ℝ, conditions x' y' z' → 
    ∃ x y z : ℝ, conditions x y z ∧ 
    ({x, y, z} = {x', y', z'})) :=
by
  sorry

end number_of_solutions_l663_663142


namespace percentage_gain_on_transaction_l663_663815

noncomputable
def percentageGain (cost_per_sheep : ℝ) : ℝ :=
let total_cost := 800 * cost_per_sheep in
let price_per_sheep_sold := 16 / 15 * cost_per_sheep in
let revenue_first_750 := total_cost in
let revenue_remaining_50 := 50 * price_per_sheep_sold in
let total_revenue := revenue_first_750 + revenue_remaining_50 in
let profit := total_revenue - total_cost in
(profit / total_cost) * 100

theorem percentage_gain_on_transaction :
  ∀ (cost_per_sheep : ℝ), percentageGain cost_per_sheep = 6.67 := by
  intros
  let cost_per_sheep_cost := cost_per_sheep
  let total_cost := 800 * cost_per_sheep_cost
  let price_per_sheep_sold := (16 / 15) * cost_per_sheep_cost
  let revenue_first_750 := total_cost
  let revenue_remaining_50 := 50 * price_per_sheep_sold
  let total_revenue := revenue_first_750 + revenue_remaining_50
  let profit := total_revenue - total_cost
  have h1 : profit = 53.33 * cost_per_sheep_cost := sorry
  have h2 : total_cost = 800 * cost_per_sheep_cost := sorry
  have percentage_gain := (profit / total_cost) * 100
  have h3 : percentage_gain = 6.67 := sorry
  exact h3

end percentage_gain_on_transaction_l663_663815


namespace necessary_condition_lg_l663_663910

theorem necessary_condition_lg (x : ℝ) : ¬(x > -1) → ¬(10^1 > x + 1) := by {
    sorry
}

end necessary_condition_lg_l663_663910


namespace solve_for_f_zero_l663_663054

noncomputable def f : ℝ → ℝ := sorry

theorem solve_for_f_zero :
  (∀ x : ℝ, (2 - x) * f(x) - 2 * f(3 - x) = -x^3 + 5 * x - 18) →
  f(0) = 7 :=
by
  intros h
  sorry

end solve_for_f_zero_l663_663054


namespace largest_constant_C_l663_663135

theorem largest_constant_C :
  ∃ C : ℝ, 
    (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z - 1)) 
      ∧ (∀ D : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ D * (x + y + z - 1)) → C ≥ D)
    ∧ C = (2 + 2 * Real.sqrt 7) / 3 :=
sorry

end largest_constant_C_l663_663135


namespace g_at_4_l663_663329

-- Define the function f
def f (x : ℝ) := 4 / (3 - x)

-- Define the inverse function f⁻¹
def f_inv (x : ℝ) := (3 * x - 4) / x

-- Define the function g
def g (x : ℝ) := 2 * (1 / (f_inv x)) + 8

-- State the lean theorem
theorem g_at_4 : g 4 = 9 :=
by
  -- The exact proof is skipped
  sorry

end g_at_4_l663_663329


namespace colin_speed_l663_663104

noncomputable def B : Real := 1
noncomputable def T : Real := 2 * B
noncomputable def Br : Real := (1/3) * T
noncomputable def C : Real := 6 * Br

theorem colin_speed : C = 4 := by
  sorry

end colin_speed_l663_663104


namespace minimize_error_l663_663014
noncomputable def exact_product (a b : ℝ) : ℝ := a * b

def approximations_a : List ℝ := [4.2, 4.3]
def approximations_b : List ℝ := [11.5, 11.6]

def errors (exact : ℝ) (approx : List (ℝ × ℝ)) : List ℝ :=
  approx.map (λ (xy : ℝ × ℝ), abs(exact - (xy.1 * xy.2)))

theorem minimize_error :
  let exact := exact_product 4.27 11.56 in
  let approx_pairs := [(4.2, 11.5), (4.2, 11.6), (4.3, 11.5), (4.3, 11.6)] in
  let errs := errors exact approx_pairs in 
  (minBy errs id) = abs (exact - (4.3 * 11.5)) := sorry

end minimize_error_l663_663014


namespace range_of_a_l663_663293

theorem range_of_a (a : ℝ) : 
  (∀ (x1 : ℝ), ∃ (x2 : ℝ), |x1| = Real.log (a * x2^2 - 4 * x2 + 1)) → (0 ≤ a) :=
by
  sorry

end range_of_a_l663_663293


namespace angle_between_vectors_l663_663655

theorem angle_between_vectors (θ : ℝ) (h1 : 0 ≤ θ ∧ θ ≤ π) (h2 : Real.sin θ = √2 / 2) : θ = π / 4 ∨ θ = 3 * π / 4 :=
sorry

end angle_between_vectors_l663_663655


namespace john_total_expenditure_l663_663684

-- Definitions based on provided conditions
def cost_tshirt := 20
def num_tshirts := 3
def cost_pants := 50
def num_pants := 2
def cost_jacket := 80
def discount_jacket := 0.25
def cost_hat := 15
def cost_shoes := 60
def discount_shoes := 0.10

-- Main theorem statement
theorem john_total_expenditure :
  (num_tshirts * cost_tshirt) +
  (num_pants * cost_pants) +
  (cost_jacket * (1 - discount_jacket)) +
  cost_hat +
  (cost_shoes * (1 - discount_shoes)) = 289 := 
begin 
  sorry 
end

end john_total_expenditure_l663_663684


namespace max_knights_between_knights_l663_663454

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l663_663454


namespace calculate_longest_side_l663_663755

noncomputable def length_longest_side_triangle : ℝ :=
  let V := 98 * 35 * 52
  let P := Real.cbrt V
  let a_x := Real.sqrt 2
  let b_x := 2 / Real.pi
  let c_x := 3 * Real.sqrt 3
  let x := P / (a_x + b_x + c_x)
  let a := a_x * x
  let b := b_x * x
  let c := c_x * x
  c

theorem calculate_longest_side : length_longest_side_triangle ≈ 40.25 := 
by
  sorry

end calculate_longest_side_l663_663755


namespace problem_true_propositions_l663_663847

-- Definitions
def is_square (q : ℕ) : Prop := q = 4
def is_trapezoid (q : ℕ) : Prop := q ≠ 4
def is_parallelogram (q : ℕ) : Prop := q = 2

-- Propositions
def prop_negation (p : Prop) : Prop := ¬ p
def prop_contrapositive (p q : Prop) : Prop := ¬ q → ¬ p
def prop_inverse (p q : Prop) : Prop := p → q

-- True propositions
theorem problem_true_propositions (a b c : ℕ) (h1 : ¬ (is_square 4)) (h2 : ¬ (is_parallelogram 3)) (h3 : ¬ (a * c^2 > b * c^2 → a > b)) : 
    (prop_negation (is_square 4) ∧ prop_contrapositive (is_trapezoid 3) (is_parallelogram 3)) ∧ ¬ prop_inverse (a * c^2 > b * c^2) (a > b) := 
by
    sorry

end problem_true_propositions_l663_663847


namespace rods_and_connectors_total_pieces_l663_663067

theorem rods_and_connectors_total_pieces (n : ℕ) (n = 10) :
  let rods := 3 * (n * (n + 1)) / 2 + 2 * 3 * n,
      connectors := (n + 1) * (n + 2) / 2 + 2 * 3 * n,
      total_pieces := rods + connectors
  in 
  total_pieces = 351 :=
by
  sorry

end rods_and_connectors_total_pieces_l663_663067


namespace train_speed_on_time_l663_663388

theorem train_speed_on_time :
  ∃ (v : ℝ), 
  (∀ (d : ℝ) (t : ℝ),
    d = 133.33 ∧ 
    80 * (t + 1/3) = d ∧ 
    v * t = d) → 
  v = 100 :=
by
  sorry

end train_speed_on_time_l663_663388


namespace inequality_solution_l663_663878

theorem inequality_solution (x : ℝ) :
  (2 / (x^2 + 2*x + 1) + 4 / (x^2 + 8*x + 7) > 3/2) ↔
  (x < -7 ∨ (-7 < x ∧ x < -1) ∨ (-1 < x)) :=
by sorry

end inequality_solution_l663_663878


namespace sum_of_squares_eq_ten_l663_663154

noncomputable def x1 : ℝ := Real.sqrt 3 - Real.sqrt 2
noncomputable def x2 : ℝ := Real.sqrt 3 + Real.sqrt 2

theorem sum_of_squares_eq_ten : x1^2 + x2^2 = 10 := 
by
  sorry

end sum_of_squares_eq_ten_l663_663154


namespace inequalities_hold_l663_663117

theorem inequalities_hold (a b c x y z : ℝ) (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧
  (x * y * z ≤ a * b * c) :=
by
  sorry

end inequalities_hold_l663_663117


namespace initial_amount_is_1179_6_l663_663848

-- Definitions based on the conditions
def amount_after_5_years (P R : ℝ) := P * (1 + (R * 5) / 100)
def amount_after_increased_interest (P R : ℝ) := P * (1 + ((R + 2) * 5) / 100)

theorem initial_amount_is_1179_6 (R : ℝ) :
  let P := 1179.6 in
  amount_after_5_years P R = 670 ∧ amount_after_increased_interest P R = 552.04 :=
by {
  sorry
}

end initial_amount_is_1179_6_l663_663848


namespace add_sub_decimals_l663_663074

theorem add_sub_decimals :
  (0.513 + 0.0067 - 0.048 = 0.4717) :=
by
  sorry

end add_sub_decimals_l663_663074


namespace how_far_did_the_man_swim_downstream_l663_663824

variable (V_m : ℝ) (time : ℝ) (distance_upstream : ℝ)
variable (V_downstream : ℝ) (distance_downstream_to_prove : ℝ)

def proof_problem : Prop :=
  V_m = 12 ∧
  time = 3 ∧
  distance_upstream = 18 ∧ 
  distance_downstream_to_prove = V_downstream * time

theorem how_far_did_the_man_swim_downstream 
  (V_m_pos : 0 < V_m)
  (time_pos : 0 < time) 
  (distance_upstream_pos : 0 < distance_upstream)
  (V_upstream : ℝ)
  (V_s : ℝ) 
  (V_s_eq : V_s = V_m - (distance_upstream / time)) :
  proof_problem V_m time distance_upstream ((V_m + V_s) * time) := by
  sorry

end how_far_did_the_man_swim_downstream_l663_663824


namespace max_a5_a6_l663_663600

theorem max_a5_a6 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) - a n = d^2) 
  (h2 : 0 < a 1) 
  (h3 : (∑ i in finset.range 10, a (i + 1)) = 30) :
  (∃ (a5 a6 : ℝ), a5 + a6 = 6 ∧ a5 * a6 = 9) :=
sorry

end max_a5_a6_l663_663600


namespace area_difference_l663_663741

theorem area_difference (d_square d_circle : ℝ) (diag_eq : d_square = 10) (diam_eq : d_circle = 10) : 
  let s := d_square / real.sqrt 2 in
  let area_square := s * s in
  let r := d_circle / 2 in
  let area_circle := real.pi * r * r in
  (area_circle - area_square) = 28.5 := by
s := d_square / real.sqrt 2
area_square := s * s
r := d_circle / 2
area_circle := real.pi * r * r
suffices : area_circle - area_square ≈ 28.5
by sorry

end area_difference_l663_663741


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663230

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663230


namespace width_of_metallic_sheet_is_36_l663_663825

-- Given conditions
def length_of_metallic_sheet : ℕ := 48
def side_length_of_cutoff_square : ℕ := 8
def volume_of_box : ℕ := 5120

-- Proof statement
theorem width_of_metallic_sheet_is_36 :
  ∀ (w : ℕ), w - 2 * side_length_of_cutoff_square = 36 - 16 →  length_of_metallic_sheet - 2* side_length_of_cutoff_square = 32  →  5120 = 256 * (w - 16)  := sorry

end width_of_metallic_sheet_is_36_l663_663825


namespace part1_solution_set_part2_no_real_x_l663_663188

-- Condition and problem definitions
def f (x a : ℝ) : ℝ := a^2 * x^2 + 2 * a * x - a^2 + 1

theorem part1_solution_set :
  (∀ x : ℝ, f x 2 ≤ 0 ↔ -3 / 2 ≤ x ∧ x ≤ 1 / 2) := sorry

theorem part2_no_real_x :
  ¬ ∃ x : ℝ, ∀ a : ℝ, -2 ≤ a ∧ a ≤ 2 → f x a ≥ 0 := sorry

end part1_solution_set_part2_no_real_x_l663_663188


namespace sum_of_consecutive_integers_l663_663227

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l663_663227


namespace stone_137_is_5_l663_663121

-- Define the counting function that represents the described counting mechanism
def stone_position (n : ℕ) : ℕ :=
  let m := n % 20 in
  if m = 0 then 1
  else if m ≤ 11 then m
  else 21 - m

-- Define the target stone for the 137th count
def target_stone := stone_position 137

theorem stone_137_is_5 : target_stone = 5 := by
  sorry

end stone_137_is_5_l663_663121


namespace no_positive_integer_divisible_l663_663779

theorem no_positive_integer_divisible (n : ℕ) (h : n > 1) :
  ∀ m : ℕ, m ∈ (set_of (λ k, ∃ (d : ℕ), (d > 0 ∧ d ≤ n) ∧ k / d = 2)) -> false :=
by
  sorry

end no_positive_integer_divisible_l663_663779


namespace exists_circumcircle_containing_polygon_l663_663157

variables {n : ℕ} (A : ℕ → ℝ × ℝ)
def convex_polygon (A : ℕ → ℝ × ℝ) : Prop := 
  ∀ i j k, (0 < i ∧ i < j ∧ j < k ∧ k ≤ n) → 
  let (xi, yi) := A i in
  let (xj, yj) := A j in
  let (xk, yk) := A k in
  (yj - yi) * (xk - xi) - (xj - xi) * (yk - yi) >= 0

theorem exists_circumcircle_containing_polygon :
  convex_polygon A →
  ∃ (i : ℕ), 0 ≤ i ∧ i + 2 < n ∧ 
  let circumcircle := λ (A1 A2 A3 : ℝ × ℝ), sorry in -- Definition of circumcircle omitted for brevity
  ∀ (m : ℕ), 1 ≤ m ∧ m ≤ n → ∃ (x, y : ℝ), 
    let (x_m, y_m) := A m in
    (x - x_m)^2 + (y - y_m)^2 ≤ circumcircle (A i) (A (i+1)) (A (i+2)) in
  true :=
begin
  intro h_convex,
  sorry -- Proof omitted.
end

end exists_circumcircle_containing_polygon_l663_663157


namespace area_diff_circle_square_l663_663739

theorem area_diff_circle_square (d_square d_circle : ℝ) (h1 : d_square = 10) (h2 : d_circle = 10) :
  let s := d_square / Real.sqrt 2,
      area_square := s^2,
      r := d_circle / 2,
      area_circle := Real.pi * r^2,
      area_diff := area_circle - area_square in
  Real.floor (area_diff * 10) / 10 = 28.5 :=
by
  sorry

end area_diff_circle_square_l663_663739


namespace athletes_derangement_l663_663544

theorem athletes_derangement: ∃ (d : ℕ), d = 44 ∧ ∀ (f : Fin 5 → Fin 5), (∀ i, f i ≠ i) ↔ (f ∈ Equiv.perm_derangements (Fin 5)) :=
by sorry

end athletes_derangement_l663_663544


namespace total_collection_l663_663429

theorem total_collection (num_members : ℕ) (contribution_per_member : ℕ) (h1 : num_members = 99) (h2 : contribution_per_member = 99) : 
  (num_members * contribution_per_member) / 100 = 98.01 := by
  sorry

end total_collection_l663_663429


namespace apex_angle_of_identical_cones_l663_663763

theorem apex_angle_of_identical_cones {α : ℝ} :
  (∀ (C1 C2 C3 C4 : Type) [cone C1] [cone C2] [cone C3] [cone C4],
    -- Conditions
    (cone.apex_angle C1 = cone.apex_angle C2) ∧
    (cone.apex_angle C3 = π / 4) ∧
    (cone.apex_angle C4 = 3 * π / 4) →
    cone.touches_externally C1 C2 ∧
    cone.touches_externally C1 C3 ∧
    cone.touches_externally C2 C3 ∧
    cone.touches_internally C4 C1 ∧
    cone.touches_internally C4 C2 ∧
    cone.touches_internally C4 C3 →
    -- Conclusion
    cone.apex_angle C1 = 2 * arctan (2 / 3))) :=
sorry

end apex_angle_of_identical_cones_l663_663763


namespace Tom_time_to_complete_wall_after_one_hour_l663_663271

noncomputable def avery_rate : ℝ := 1 / 2
noncomputable def tom_rate : ℝ := 1 / 4
noncomputable def combined_rate : ℝ := avery_rate + tom_rate
noncomputable def wall_built_in_first_hour : ℝ := combined_rate * 1
noncomputable def remaining_wall : ℝ := 1 - wall_built_in_first_hour 
noncomputable def tom_time_to_complete_remaining_wall : ℝ := remaining_wall / tom_rate

theorem Tom_time_to_complete_wall_after_one_hour : 
  tom_time_to_complete_remaining_wall = 1 :=
by
  sorry

end Tom_time_to_complete_wall_after_one_hour_l663_663271


namespace tangent_line_to_circle_l663_663342

theorem tangent_line_to_circle (a : ℝ) :
  (∃ k : ℝ, k = a ∧ (∀ x y : ℝ, y = x + 4 → (x - k)^2 + (y - 3)^2 = 8)) ↔ (a = 3 ∨ a = -5) := by
  sorry

end tangent_line_to_circle_l663_663342


namespace digit_correction_product_l663_663333

-- Definitions based on the given conditions
def original_sum : ℕ := 935467 + 716820
def displayed_sum : ℕ := 1419327

-- Noncomputable definition to specify the proof problem
noncomputable def digit_correction (d e : ℕ) : Prop :=
  (d = 5) ∧ (e = 9) ∧ (original_sum = displayed_sum + (e - d) * 10 ^ 2) -- Positioning for changing a digit

-- The main proof problem statement
theorem digit_correction_product : ∃ d e, digit_correction d e ∧ d * e = 45 := 
by {
  use 5,
  use 9,
  sorry
}

end digit_correction_product_l663_663333


namespace min_disks_needed_l663_663019

-- Definitions based on conditions:
def num_files : Nat := 40
def disk_capacity : Float := 2.0
def size_file_0_9 : Float := 0.9
def num_files_0_9 : Nat := 5
def size_file_0_75 : Float := 0.75
def num_files_0_75 : Nat := 15
def size_file_0_5 : Float := 0.5
def num_files_0_5 : Nat :=
  num_files - num_files_0_9 - num_files_0_75

-- Theorem statement:
theorem min_disks_needed :
  (num_files_0_9 * size_file_0_9 + num_files_0_75 * size_file_0_75 + num_files_0_5 * size_file_0_5) <= num_files * disk_capacity → 
  ∃ min_disks : Nat, min_disks = 18 :=
sorry

end min_disks_needed_l663_663019


namespace probability_not_red_is_two_thirds_l663_663058

-- Given conditions as definitions
def number_of_orange_marbles : ℕ := 4
def number_of_purple_marbles : ℕ := 7
def number_of_red_marbles : ℕ := 8
def number_of_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_red_marbles + 
  number_of_yellow_marbles

def number_of_non_red_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_yellow_marbles

-- Define the probability
def probability_not_red : ℚ :=
  number_of_non_red_marbles / total_marbles

-- The theorem that states the probability of not picking a red marble is 2/3
theorem probability_not_red_is_two_thirds :
  probability_not_red = 2 / 3 :=
by
  sorry

end probability_not_red_is_two_thirds_l663_663058


namespace max_knights_between_other_knights_l663_663477

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l663_663477


namespace find_length_of_wood_l663_663832

-- Definitions based on given conditions
def Area := 24  -- square feet
def Width := 6  -- feet

-- The mathematical proof problem turned into Lean 4 statement
theorem find_length_of_wood (h : Area = 24) (hw : Width = 6) : (Length : ℕ) ∈ {l | l = Area / Width ∧ l = 4} :=
by {
  sorry
}

end find_length_of_wood_l663_663832


namespace pencil_distribution_l663_663370

-- Definition of the problem conditions
def distribution_ways (n k : ℕ) (f : ℕ → ℕ) : Prop :=
  (∀ i, i < k → f i > 0) ∧ (∑ i in finset.range k, f i = n)

-- Prove that there are 10 ways to distribute 6 pencils among 3 friends
theorem pencil_distribution : ∃ (f : fin 3 → ℕ), distribution_ways 6 3 f ∧ (finset.univ.image f).card = 10 :=
sorry

end pencil_distribution_l663_663370


namespace max_knights_seated_next_to_two_knights_l663_663492

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l663_663492


namespace choose_president_vp_committee_l663_663994

theorem choose_president_vp_committee (n : ℕ) (h1 : n = 10) :
  (∃ p v : Fin n, p ≠ v ∧ ∃ (committee : Finset (Fin n)), 
  committee.card = 3 ∧ ¬(p ∈ committee) ∧ ¬(v ∈ committee)) → 
  (10 * 9 * Nat.choose 8 3 = 5040) :=
by
  intro hyp
  rw h1
  sorry

end choose_president_vp_committee_l663_663994


namespace acme_vowel_soup_sequences_l663_663842

-- Define the vowels and their frequencies
def vowels : List (Char × ℕ) := [('A', 6), ('E', 6), ('I', 6), ('O', 4), ('U', 4)]

-- Noncomputable definition to calculate the total number of sequences
noncomputable def number_of_sequences : ℕ :=
  let single_vowel_choices := 6 + 6 + 6 + 4 + 4
  single_vowel_choices^5

-- Theorem stating the number of five-letter sequences
theorem acme_vowel_soup_sequences : number_of_sequences = 11881376 := by
  sorry

end acme_vowel_soup_sequences_l663_663842


namespace diagonals_intersect_at_one_point_in_regular_18gon_l663_663408

theorem diagonals_intersect_at_one_point_in_regular_18gon
  (a b c d e f : ℕ)
  (k p m q n r : ℤ)
  (ha : a < 18) (hb : b < 18) (hc : c < 18) (hd : d < 18) (he : e < 18) (hf : f < 18)
  (hk : k = (a : ℤ) - (b : ℤ)) (hp : p = (b : ℤ) - (c : ℤ))
  (hm : m = (c : ℤ) - (d : ℤ)) (hq : q = (d : ℤ) - (e : ℤ))
  (hn : n = (e : ℤ) - (f : ℤ)) (hr : r = (f : ℤ) - (a : ℤ))
  (h1 : {k, m, n} = {p, q, r} ∨
        ({k, m, n} = {1, 2, 7} ∧ {p, q, r} = {1, 3, 4}) ∨
        ({k, m, n} = {1, 2, 8} ∧ {p, q, r} = {2, 2, 3})) :
    ∃ P, P ∈ (line_through (A a) (A d) ∩ line_through (A b) (A e) ∩ line_through (A c) (A f)) :=
sorry

end diagonals_intersect_at_one_point_in_regular_18gon_l663_663408


namespace machine_A_production_rate_l663_663706

theorem machine_A_production_rate :
  ∀ (A B T_A T_B : ℝ),
    500 = A * T_A →
    500 = B * T_B →
    B = 1.25 * A →
    T_A = T_B + 15 →
    A = 100 / 15 :=
by
  intros A B T_A T_B hA hB hRate hTime
  sorry

end machine_A_production_rate_l663_663706


namespace fourth_number_unit_digit_l663_663359

def unit_digit (n : ℕ) : ℕ := n % 10

theorem fourth_number_unit_digit (a b c d : ℕ) (h₁ : a = 7858) (h₂: b = 1086) (h₃ : c = 4582) (h₄ : unit_digit (a * b * c * d) = 8) :
  unit_digit d = 4 :=
sorry

end fourth_number_unit_digit_l663_663359


namespace median_salary_of_employees_l663_663660

-- Definitions based on the conditions provided in the problem
def employee_salaries : List ℕ := [2000, 2200, 2200, 2200, 2200, 2400, 2400, 2400, 2600, 2600]

-- The theorem stating the median
theorem median_salary_of_employees : List.median employee_salaries = 2300 :=
by
  -- Definitions and proof would go here
  sorry

end median_salary_of_employees_l663_663660


namespace sum_of_consecutive_integers_of_sqrt3_l663_663210

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l663_663210


namespace laps_remaining_l663_663274

theorem laps_remaining 
  (total_laps : ℕ)
  (saturday_laps : ℕ)
  (sunday_morning_laps : ℕ)
  (total_laps_eq : total_laps = 98)
  (saturday_laps_eq : saturday_laps = 27)
  (sunday_morning_laps_eq : sunday_morning_laps = 15) :
  total_laps - saturday_laps - sunday_morning_laps = 56 :=
by
  rw [total_laps_eq, saturday_laps_eq, sunday_morning_laps_eq]
  norm_num

end laps_remaining_l663_663274


namespace find_a_and_union_l663_663194

open Set

theorem find_a_and_union (a : ℤ) (A B : Set ℤ) (hA : A = {2, 3, a ^ 2 + 4 * a + 2})
    (hB : B = {0, 7, 2 - a, a ^ 2 + 4 * a - 2}) (h_inter : A ∩ B = {3, 7}) :
    a = 1 ∧ A ∪ B = {0, 1, 2, 3, 7} := by
  -- Proof to be provided
  sorry

end find_a_and_union_l663_663194


namespace rhombus_ABCD_EF_range_l663_663852

open Real -- For dealing with real numbers

noncomputable def rhombus_ABCD_range_of_EF (E F : ℝ × ℝ) (x y : ℝ) : Prop :=
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 2)
  let D := (0, 2)
  let AE := dist A E
  let CF := dist C F
  let EF := dist E F
  let BD := dist B D
  2 ≤ AE + CF → BD = 2 → (B.1 - D.1) ^ 2 + (B.2 - D.2) ^ 2 = 4 →
  A.1 + CF = 2 → x = y → (sqrt 3 ≤ EF ∧ EF ≤ 2)

theorem rhombus_ABCD_EF_range {E F : ℝ × ℝ} (h1 : 2 ≤ dist (0, 0) E + dist (2, 2) F)
(h2 : dist (2, 0) (0, 2) = 2)
(h3 : (2 - 0) ^ 2 + (0 - 2) ^ 2 = 4)
(h4 : (0) + dist (2, 2) F = 2)
(h5 : ?x = ?y) : rhombus_ABCD_range_of_EF E F ?x ?y := by
  -- We need to prove that the length of the segment EF falls within the range sqrt(3) and 2
  sorry

end rhombus_ABCD_EF_range_l663_663852


namespace A1B1_B1C1_C1A1_geq_A1B_B1C_C1A_l663_663689

-- Define the equilateral triangle and a point inside it
variable (A B C P A_1 B_1 C_1 : Point)
variable (h_eq_triangle : EquilateralTriangle A B C)
variable (h_P_interior : PointInInterior P (Triangle A B C))
variable (h_A1 : LineSegmentIntersection (AP P) (Side BC))
variable (h_B1 : LineSegmentIntersection (BP P) (Side CA))
variable (h_C1 : LineSegmentIntersection (CP P) (Side AB))

-- Define the main theorem for the problem
theorem A1B1_B1C1_C1A1_geq_A1B_B1C_C1A :
  A1B1 * B1C1 * C1A1 ≥ A1B * B1C * C1A :=
  sorry

end A1B1_B1C1_C1A1_geq_A1B_B1C_C1A_l663_663689


namespace cos_angle_BCG_zero_l663_663803

-- Define variables and conditions
variables (A B C D E F G H : Type) [cube : Cube A B C D E F G H]

-- Define the points B, C, and G and properties
variables [adjBC : Adjacent B C] [aboveG : Above C G]

-- The theorem to prove
theorem cos_angle_BCG_zero (a : ℝ) :
  ∀ {B C G : Type}, (BC = a) ∧ (CG = a) ∧ (angle B C G = 90) → cos (angle B C G) = 0 :=
by
  sorry

end cos_angle_BCG_zero_l663_663803


namespace lcm_of_numbers_is_correct_lcm_of_polynomials_is_correct_l663_663114

-- Definition of the given natural numbers
def n1 := 4199
def n2 := 4641
def n3 := 5083

-- Definition of the given polynomials
def f (x : ℝ) : ℝ := 300 * x^4 + 425 * x^3 + 138 * x^2 - 17 * x - 6
def g (x : ℝ) : ℝ := 225 * x^4 - 109 * x^3 + 4

-- The LCM of the three numbers
def lcm_numbers : ℕ := 2028117

-- The LCM of the two polynomials
def lcm_polynomials (x : ℝ) : ℝ := (225 * x^4 - 109 * x^3 + 4) * (4 * x + 3)

theorem lcm_of_numbers_is_correct : Nat.lcm (Nat.lcm n1 n2) n3 = lcm_numbers :=
by 
  sorry

theorem lcm_of_polynomials_is_correct : 
  (λ x, f x * g x / Polynomial.gcd (Polynomial.of_real f) (Polynomial.of_real g)) = lcm_polynomials :=
by
  sorry

end lcm_of_numbers_is_correct_lcm_of_polynomials_is_correct_l663_663114


namespace minimum_value_of_n_l663_663243

theorem minimum_value_of_n : ∃ (n : ℕ), n ≥ 2 ∧ 
  (∃ (a : Fin n → ℤ), (∑ i, a i = 1990) ∧ (∏ i, a i = 1990)) ∧ n = 5 :=
by {
  sorry -- Proof is omitted
}

end minimum_value_of_n_l663_663243


namespace select_gloves_l663_663203

theorem select_gloves : 
  (∃ f : {Set (Fin 12) // f.card = 4} → Prop, 
  (∀ f, (∃ (p1 ∈ f) (p2 ∈ f), p1 ≠ p2 ∧ p1.1 / 2 = p2.1 / 2) ∧ 
  ∃! pairAmong p, 1 ≤ p.card ∧ p.card < 4) ∧ 
  (f.count (λ i, (i / 2)) = 6)
  → (f.count (λ i, (i.card)) = 240)) :=
sorry

end select_gloves_l663_663203


namespace area_enclosed_by_curve_l663_663733

-- Define the side length of the regular octagon
def side_length_octagon : ℝ := 3

-- Define the number of arcs
def num_arcs : ℕ := 12

-- Define the length of each arc
def arc_length : ℝ := π

-- Define the total area calculation
theorem area_enclosed_by_curve :
  let radius := arc_length / π,
  let area_half_circle_sector := (1 / 2) * π * radius^2,
  let total_area_sectors := num_arcs * area_half_circle_sector / 2,
  let area_octagon := 2 * (1 + Real.sqrt 2) * side_length_octagon^2,
  let total_area := area_octagon + 3 * area_half_circle_sector
  in total_area = 54 + 54 * Real.sqrt 2 + 1.5 * π :=
by
  -- sorry will be replaced by the actual proof
  sorry

end area_enclosed_by_curve_l663_663733


namespace remi_water_intake_l663_663316

def bottle_capacity := 20
def daily_refills := 3
def num_days := 7
def spill1 := 5
def spill2 := 8

def daily_intake := daily_refills * bottle_capacity
def total_intake_without_spill := daily_intake * num_days
def total_spill := spill1 + spill2
def total_intake_with_spill := total_intake_without_spill - total_spill

theorem remi_water_intake : total_intake_with_spill = 407 := 
by
  -- Provide proof here
  sorry

end remi_water_intake_l663_663316


namespace coefficient_of_x_in_expansion_l663_663266

theorem coefficient_of_x_in_expansion :
  (∑ k in range (4 + 1), (binomial 4 k) * (sqrt x ^ (4 - k)) * (-1) ^ k) *
  (x^2 - 2 * x + 1) = 4 :=
by
  sorry

end coefficient_of_x_in_expansion_l663_663266


namespace min_covering_circles_min_radius_required_l663_663775

-- Declare constants and definitions
constant unit_circle : Type
constant congruent_circles : Type
constant radius : congruent_circles → ℝ

-- Define the covering condition
def covers (c1 c2 : congruent_circles) (u : unit_circle) : Prop :=
-- details of coverage logic are omitted
sorry

-- We state the problem conditions and what needs to be proved.

constant unit_circle_instance : unit_circle

-- State that we need at least 3 congruent circles to cover the unit circle
theorem min_covering_circles (u : unit_circle) :
  ∃ (circles : List congruent_circles), circles.length = 3 ∧ (∀ c ∈ circles, radius c = 1) ∧ ∀ x ∈ {u}, ∃ c ∈ circles, covers c unit_circle_instance := 
begin
  sorry
end

-- Determine the minimum radius of the covering circles given there are 3 circles required to cover.
theorem min_radius_required (u : unit_circle) :
  ∃ (r : ℝ) (circles : List congruent_circles), circles.length = 3 ∧ (∀ c ∈ circles, radius c = r) ∧ r = real.sqrt 3 / 2 ∧ ∀ x ∈ {u}, ∃ c ∈ circles, covers c unit_circle_instance := 
begin
  sorry
end

end min_covering_circles_min_radius_required_l663_663775


namespace quadratic_equivalence_statement_l663_663945

noncomputable def quadratic_in_cos (a b c x : ℝ) : Prop := 
  a * (Real.cos x)^2 + b * Real.cos x + c = 0

noncomputable def transform_to_cos2x (a b c : ℝ) : Prop := 
  (4*a^2) * (Real.cos (2*a))^2 + (2*a^2 + 4*a*c - 2*b^2) * Real.cos (2*a) + a^2 + 4*a*c - 2*b^2 + 4*c^2 = 0

theorem quadratic_equivalence_statement (a b c : ℝ) (h : quadratic_in_cos 4 2 (-1) a) :
  transform_to_cos2x 16 12 (-4) :=
sorry

end quadratic_equivalence_statement_l663_663945


namespace find_range_of_a_l663_663189

noncomputable def quadratic_function_increasing (a : ℝ) : Prop := 
  ∀ x y : ℝ, (x ≤ y) → (x ≤ -1) → (y ≤ -1) → (a * x^2 + (a^3 - a) * x + 1 ≤ a * y^2 + (a^3 - a) * y + 1)

theorem find_range_of_a
  (h : quadratic_function_increasing a) : 
  -real.sqrt 3 ≤ a ∧ a < 0 :=
sorry

end find_range_of_a_l663_663189


namespace test_question_total_l663_663392

theorem test_question_total
  (total_points : ℕ)
  (points_2q : ℕ)
  (points_4q : ℕ)
  (num_2q : ℕ)
  (num_4q : ℕ)
  (H1 : total_points = 100)
  (H2 : points_2q = 2)
  (H3 : points_4q = 4)
  (H4 : num_2q = 30)
  (H5 : total_points = num_2q * points_2q + num_4q * points_4q) :
  num_2q + num_4q = 40 := 
sorry

end test_question_total_l663_663392


namespace min_x0_l663_663651

noncomputable def f (ω x : ℝ) : ℝ := Real.sin(ω * x + Real.pi / 6)

theorem min_x0 (ω : ℝ) (x0 : ℝ) (hω : ω > 0)
  (h_symmetry : ∀ x, f ω x = f ω (x + π))
  (h_central_symmetry : f ω x0 = 0 ∧ x0 > 0) :
  x0 = 5 * Real.pi / 12 :=
sorry

end min_x0_l663_663651


namespace sum_of_consecutive_integers_l663_663228

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l663_663228


namespace magnitude_of_difference_between_roots_l663_663925

variable (α β m : ℝ)

theorem magnitude_of_difference_between_roots
    (hαβ_root : ∀ x, x^2 - 2 * m * x + m^2 - 4 = 0 → (x = α ∨ x = β)) :
    |α - β| = 4 := by
  sorry

end magnitude_of_difference_between_roots_l663_663925


namespace sine_product_identity_l663_663718

theorem sine_product_identity (n : ℕ) :
  2^n * (∏ k in Finset.range n.succ, Real.sin ((k : ℝ) * Real.pi / (2 * n.succ + 1))) = Real.sqrt (2 * n + 1) := 
sorry

end sine_product_identity_l663_663718


namespace slope_and_inclination_l663_663176

def Point : Type := ℝ × ℝ
def line_slope (A B : Point) : ℝ := (B.2 - A.2) / (B.1 - A.1)
def angle_of_inclination (k : ℝ) : ℝ := arctan k

theorem slope_and_inclination (A B : Point) (k : ℝ) (α : ℝ) 
  (hA : A = (1, 2)) 
  (hB : B = (4, 2 + real.sqrt 3))
  (hk : k = line_slope A B)
  (hα : α = angle_of_inclination k) :
  k = real.sqrt 3 / 3 ∧ α = real.pi / 6 := by
  sorry

end slope_and_inclination_l663_663176


namespace problem_statement_eq_l663_663150

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_statement_eq :
  dollar ((x + y) ^ 2) ((y + x) ^ 2) = 0 := by
  sorry

end problem_statement_eq_l663_663150


namespace abs_sqrt_simplify_l663_663957

theorem abs_sqrt_simplify (x : ℝ) (h : x < -2) : 
  abs (x - real.sqrt ((x + 2) ^ 2)) = -2 * x - 2 :=
sorry

end abs_sqrt_simplify_l663_663957


namespace consecutive_sum_l663_663213

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l663_663213


namespace apples_in_bowl_l663_663045

theorem apples_in_bowl (A : ℕ) (h : 0.60 * (A + 8) = A) : A = 12 :=
by
  sorry

end apples_in_bowl_l663_663045


namespace metallic_sheet_width_l663_663828

theorem metallic_sheet_width (length : ℝ) (side : ℝ) (volume : ℝ) (width : ℝ) :
  length = 48 → side = 8 → volume = 5120 → volume = (length - 2 * side) * (width - 2 * side) * side → width = 36 :=
by
  intro h_length h_side h_volume h_eq
  have h1 : length - 2 * side = 32 := by sorry
  have h2 : side = 8 := h_side
  have h3 : h_volume = (32) * (width - 16) * 8 := by sorry
  have h4 : width - 16 = 20 := by sorry
  show width = 36 from by sorry

end metallic_sheet_width_l663_663828


namespace lorry_speed_in_kmph_l663_663060

/-- 
  A lorry travels a known distance in a given time.
  Calculate the speed in kmph.
  Conditions:
  - length_of_lorry: The length of the lorry in meters (200 m).
  - time_taken: The time taken to cross the bridge in seconds (17.998560115190784 s).
  - length_of_bridge: The length of the bridge in meters (200 m).
-/
theorem lorry_speed_in_kmph (length_of_lorry length_of_bridge : ℝ) (time_taken : ℝ)
  (length_of_lorry_eq : length_of_lorry = 200)
  (length_of_bridge_eq : length_of_bridge = 200)
  (time_taken_eq : time_taken = 17.998560115190784) :
  let speed_in_kmph := ((length_of_lorry + length_of_bridge) / time_taken) * 3.6 in
  abs (speed_in_kmph - 80) < 1 :=
by
  sorry

end lorry_speed_in_kmph_l663_663060


namespace width_of_metallic_sheet_is_36_l663_663826

-- Given conditions
def length_of_metallic_sheet : ℕ := 48
def side_length_of_cutoff_square : ℕ := 8
def volume_of_box : ℕ := 5120

-- Proof statement
theorem width_of_metallic_sheet_is_36 :
  ∀ (w : ℕ), w - 2 * side_length_of_cutoff_square = 36 - 16 →  length_of_metallic_sheet - 2* side_length_of_cutoff_square = 32  →  5120 = 256 * (w - 16)  := sorry

end width_of_metallic_sheet_is_36_l663_663826


namespace solve_equation_l663_663322

-- Define the problem as a proposition in Lean
theorem solve_equation : ∃ x : ℤ, (∑ k in Finset.range 10, (x+k)*(x+k+1) = ∑ k in Finset.range (10-1), k*(k+1)) ∧ (x = 0 ∨ x = -10) :=
begin
  sorry
end

end solve_equation_l663_663322


namespace square_of_sum_possible_l663_663004

theorem square_of_sum_possible (a b c : ℝ) : 
  ∃ (cells : set ℝ), 
    (∀ x y ∈ cells, x ≠ y → (∃ z, z ∉ cells ∧ cells = insert z (cells \ {x, y}) ∧ z = x + y))
    ∧ (∀ x y z ∈ cells, x ≠ y ∧ y ≠ z ∧ x ≠ z → (∃ w, w ∉ cells ∧ cells = insert w (cells \ {x, y, z}) ∧ w = x * y + z^2))
    ∧ ((a + b + c)^2 ∈ cells) :=
sorry

end square_of_sum_possible_l663_663004


namespace total_bugs_eaten_l663_663673

-- Define the conditions
def gecko_eats : ℕ := 12
def lizard_eats : ℕ := gecko_eats / 2
def frog_eats : ℕ := lizard_eats * 3
def toad_eats : ℕ := frog_eats + (frog_eats / 2)

-- Define the proof
theorem total_bugs_eaten : gecko_eats + lizard_eats + frog_eats + toad_eats = 63 :=
by
  sorry

end total_bugs_eaten_l663_663673


namespace tangent_at_1_tangent_passing_through_B_l663_663942

def f (x : ℝ) : ℝ := x^3 + 2

theorem tangent_at_1 : ∃ (a b c : ℝ), 3 * 1 - 1 * 3 = b ∧  b = c ∧ (∀ x y : ℝ, (3x - y = 0)) :=
by
  sorry

theorem tangent_passing_through_B : ∃ (a b c : ℝ), 3 * 0 - 4 = c + b ∧ (0,4) = 1 ∧ (∀ x y : ℝ,  3x - y + 4 = 0 ) :=
by
  sorry

end tangent_at_1_tangent_passing_through_B_l663_663942


namespace sample_size_eq_n_l663_663411

def models_ratio := (2, 3, 5)
def model_A_units := 16
def n : ℕ := 80

theorem sample_size_eq_n :
  let (a, b, c) := models_ratio in
  let total := model_A_units + model_A_units * b / a + model_A_units * c / a in
  total = n :=
by { sorry }

end sample_size_eq_n_l663_663411


namespace john_finishes_fourth_task_at_12_18_PM_l663_663685

theorem john_finishes_fourth_task_at_12_18_PM :
  let start_time := 8 * 60 + 45 -- Start time in minutes from midnight
  let third_task_time := 11 * 60 + 25 -- End time of the third task in minutes from midnight
  let total_time_three_tasks := third_task_time - start_time -- Total time in minutes to complete three tasks
  let time_per_task := total_time_three_tasks / 3 -- Time per task in minutes
  let fourth_task_end_time := third_task_time + time_per_task -- End time of the fourth task in minutes from midnight
  fourth_task_end_time = 12 * 60 + 18 := -- Expected end time in minutes from midnight
  sorry

end john_finishes_fourth_task_at_12_18_PM_l663_663685


namespace total_votes_l663_663711

noncomputable def total_votes_proof : Prop :=
  ∃ T A : ℝ, 
    A = 0.40 * T ∧ 
    T = A + (A + 70) ∧ 
    T = 350

theorem total_votes : total_votes_proof :=
sorry

end total_votes_l663_663711


namespace water_remaining_8_pourings_l663_663812

-- Define the fraction sequence and calculate the remaining water after each pouring
def water_after_pouring (n : ℕ) : ℚ :=
  match n with
  | 0     => 1
  | 1     => 3 / 4
  | 2     => 1 / 4
  | 3     => 1 / 8
  | _+3 => let reset := if (n % 4 = 0) then 1 / 2 else (1 - (Float.ofNat (n % 4) * 1 / 10)) in
           water_after_pouring (n-1) * reset

theorem water_remaining_8_pourings : water_after_pouring 8 = 1 / 5 :=
by 
  /- Proof steps will be filled here; the current statement is set to ensure build success -/
  sorry

end water_remaining_8_pourings_l663_663812


namespace output_when_t_eq_5_l663_663936

def C : ℝ → ℝ
| t => if t ≤ 3 then 0.2 else 0.2 + 0.1 * (t - 3)

theorem output_when_t_eq_5 : C 5 = 0.4 := by
  sorry

end output_when_t_eq_5_l663_663936


namespace cos_alpha_l663_663905

theorem cos_alpha (α β : ℝ) (h1 : 0 < β) (h2 : β < α) (h3 : α < π / 3) (h4 : α - β = π / 6) (h5 : cos (2 * α + 2 * β) = -1 / 2) : 
  cos α = sqrt 2 / 2 :=
by
  sorry

end cos_alpha_l663_663905


namespace batting_average_drop_l663_663334

theorem batting_average_drop 
    (avg : ℕ)
    (innings : ℕ)
    (high : ℕ)
    (high_low_diff : ℕ)
    (low : ℕ)
    (total_runs : ℕ)
    (new_avg : ℕ)

    (h1 : avg = 50)
    (h2 : innings = 40)
    (h3 : high = 174)
    (h4 : high = low + 172)
    (h5 : total_runs = avg * innings)
    (h6 : new_avg = (total_runs - high - low) / (innings - 2)) :

  avg - new_avg = 2 :=
by
  sorry

end batting_average_drop_l663_663334


namespace avg_of_four_consecutive_l663_663764

theorem avg_of_four_consecutive (x : ℤ) (y : ℤ) (h : y = (x + (x + 2) + (x + 4)) / 3) :
  ((y + 1) + (y + 2) + (y + 3) + (y + 4)) / 4 = x + 4.5 := by
  sorry

end avg_of_four_consecutive_l663_663764


namespace sin_A_in_right_triangle_l663_663997

noncomputable def triangle_abc_bc : ℝ := real.sqrt (24^2 + 7^2)

theorem sin_A_in_right_triangle (A B C : Type) [triangle ABC] [right_triangle ABC (angle A B C)] (AB AC : ℝ) (h1 : AB = 7) (h2 : AC = 24) :
  sin (angle A B C) = 7 / real.sqrt (24^2 + 7^) := by 
sorry

end sin_A_in_right_triangle_l663_663997


namespace total_number_of_animals_l663_663757

-- Definitions based on conditions
def number_of_females : ℕ := 35
def males_outnumber_females_by : ℕ := 7
def number_of_males : ℕ := number_of_females + males_outnumber_females_by

-- Theorem to prove the total number of animals
theorem total_number_of_animals :
  number_of_females + number_of_males = 77 := by
  sorry

end total_number_of_animals_l663_663757


namespace ratio_of_powers_l663_663241

theorem ratio_of_powers (a x : ℝ) (h : a^(2 * x) = Real.sqrt 2 - 1) : (a^(3 * x) + a^(-3 * x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end ratio_of_powers_l663_663241


namespace radius_of_base_of_cone_l663_663934

-- Define the surface area of the cone
def surface_area_cone (r l : ℝ) : ℝ := π * r^2 + π * r * l

-- Define the relationship between the slant height and the base radius
def slant_height_relation (r l : ℝ) : Prop := l = 2 * r

-- Define the given conditions
def given_conditions (r l : ℝ) : Prop :=
  surface_area_cone r l = 12 * π ∧ slant_height_relation r l

-- State the theorem
theorem radius_of_base_of_cone (r l : ℝ) (hc : given_conditions r l) : r = 2 :=
by
  sorry

end radius_of_base_of_cone_l663_663934


namespace exists_circumcircle_containing_polygon_l663_663156

variables {n : ℕ} (A : ℕ → ℝ × ℝ)
def convex_polygon (A : ℕ → ℝ × ℝ) : Prop := 
  ∀ i j k, (0 < i ∧ i < j ∧ j < k ∧ k ≤ n) → 
  let (xi, yi) := A i in
  let (xj, yj) := A j in
  let (xk, yk) := A k in
  (yj - yi) * (xk - xi) - (xj - xi) * (yk - yi) >= 0

theorem exists_circumcircle_containing_polygon :
  convex_polygon A →
  ∃ (i : ℕ), 0 ≤ i ∧ i + 2 < n ∧ 
  let circumcircle := λ (A1 A2 A3 : ℝ × ℝ), sorry in -- Definition of circumcircle omitted for brevity
  ∀ (m : ℕ), 1 ≤ m ∧ m ≤ n → ∃ (x, y : ℝ), 
    let (x_m, y_m) := A m in
    (x - x_m)^2 + (y - y_m)^2 ≤ circumcircle (A i) (A (i+1)) (A (i+2)) in
  true :=
begin
  intro h_convex,
  sorry -- Proof omitted.
end

end exists_circumcircle_containing_polygon_l663_663156


namespace probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five_l663_663577

noncomputable def probability_of_product_ending_in_0 : ℝ :=
  let p_no_0 := (9 / 10 : ℝ) ^ 20
  let p_at_least_one_0 := 1 - p_no_0
  let p_no_even := (5 / 9 : ℝ) ^ 20
  let p_at_least_one_even := 1 - p_no_even
  let p_no_5_in_19 := (8 / 9 : ℝ) ^ 19
  let p_at_least_one_5 := 1 - p_no_5_in_19
  p_at_least_one_0 + p_no_0 * p_at_least_one_even * p_at_least_one_5

theorem probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five :
  abs (probability_of_product_ending_in_0 - 0.9865) < 0.0001 :=
begin
  -- proofs go here, sorry is used to skip the proof details
  sorry
end

end probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five_l663_663577


namespace polly_to_sandy_ratio_l663_663344

variable {W P S : ℝ}
variable (h1 : S = (5/2) * W) (h2 : P = 2 * W)

theorem polly_to_sandy_ratio : P = (4/5) * S := by
  sorry

end polly_to_sandy_ratio_l663_663344


namespace ellipse_with_foci_on_x_axis_l663_663967

theorem ellipse_with_foci_on_x_axis (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a - 5) + (y^2) / 2 = 1 →  
   (∃ cx cy : ℝ, ∀ x', cx - x' = a - 5 ∧ cy = 2)) → 
  a > 7 :=
by sorry

end ellipse_with_foci_on_x_axis_l663_663967


namespace problem_graph_empty_l663_663513

open Real

theorem problem_graph_empty : ∀ x y : ℝ, ¬ (x^2 + 3 * y^2 - 4 * x - 12 * y + 28 = 0) :=
by
  intro x y
  -- Apply the contradiction argument based on the conditions given
  sorry


end problem_graph_empty_l663_663513


namespace vector_statements_l663_663447

-- Definitions based on the conditions
variables {α : Type} [AddCommGroup α] [Module ℝ α]

-- Statements to be proven
def vector1 (a b : α) : Prop := ∃ (h : ℝ), (a = h • b ∧ h ≠ 0)
def vector2 (a b : α) : Prop := ∃ (h : ℝ), (b = h • a ∧ h ≠ 0)

-- Formalizing the solution:
theorem vector_statements :
  let a b : α in
  (∀ a b : α, (∃ u, u * a = b) → (a = b)) ∧
  (∀ a b : α, (a = b) → (vector1 a b)).  
by {
  intros a b,
  split,
  intros a b h,
  obtain ⟨u, hu⟩ := h,
  exact hu,
  intros a b hab,
  rw hab,
  exact ⟨1, by simp⟩
}
sorry

end vector_statements_l663_663447


namespace point_B_in_intersection_l663_663179

noncomputable def point : Type := (ℝ × ℝ)

-- Definitions from conditions in a)
def O1 : point := sorry
def O2 : point := sorry

def line (p : point) : Prop := p.2 = 1 - p.1
def point_A : point := (-7, 9)

def circle (center : point) (radius : ℝ) : set point :=
  {q : point | (q.1 - center.1)^2 + (q.2 - center.2)^2 = radius^2 }

def intersect (c1 c2 : set point) : set point :=
  {q : point | q ∈ c1 ∧ q ∈ c2}

-- The circles centered at O1 and O2 with some radius intersect at points A and B
def circle1 : set point := circle O1 (sqrtd (O1.1 - point_A.1)^2 + (O1.2 - point_A.2)^2)
def circle2 : set point := circle O2 (sqrtd (O2.1 - point_A.1)^2 + (O2.2 - point_A.2)^2)
def intersection_points : set point := intersect circle1 circle2

-- Point B is the reflection of point A across the line y = 1 - x
def point_B : point := (-8, 8)

-- The final proposition to prove B is in the intersection points
theorem point_B_in_intersection : point_B ∈ intersection_points :=
sorry

end point_B_in_intersection_l663_663179


namespace supplierB_stats_l663_663807

noncomputable def supplierB_data : List ℝ :=
  [72, 75, 72, 75, 78, 77, 73, 75, 76, 77, 71, 78, 79, 72, 75]

def mean (l : List ℝ) : ℝ := l.sum / l.length

def mode (l : List ℝ) : ℝ :=
  l.foldl (λ m x, if l.count x > l.count m then x else m) l.head!

def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x, (x - μ) ^ 2)).sum / l.length

theorem supplierB_stats :
  mean supplierB_data = 75 ∧
  mode supplierB_data = 75 ∧
  variance supplierB_data = 6 :=
by
  sorry

end supplierB_stats_l663_663807


namespace trig_identity_l663_663502

theorem trig_identity :
  (Real.tan (30 * Real.pi / 180) * Real.cos (60 * Real.pi / 180) + Real.tan (45 * Real.pi / 180) * Real.cos (30 * Real.pi / 180)) = (2 * Real.sqrt 3) / 3 :=
by
  -- Proof is omitted
  sorry

end trig_identity_l663_663502


namespace total_laps_jogged_l663_663280

-- Defining the conditions
def jogged_PE_class : ℝ := 1.12
def jogged_track_practice : ℝ := 2.12

-- Statement to prove
theorem total_laps_jogged : jogged_PE_class + jogged_track_practice = 3.24 := by
  -- Proof would go here
  sorry

end total_laps_jogged_l663_663280


namespace no_pairs_exist_l663_663532

theorem no_pairs_exist (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : (1/a + 1/b = 2/(a+b)) → False :=
by
  sorry

end no_pairs_exist_l663_663532


namespace minimum_shots_to_destroy_tank_l663_663816

theorem minimum_shots_to_destroy_tank :
  ∃ (n : ℕ), ∀ (field : fin 41 × fin 41) (tank_move : (fin 41 × fin 41) → (fin 41 × fin 41) → Prop),
    (∀ (pos : fin 41 × fin 41), ∃ (tank : fin 41 × fin 41), (tank_move pos tank → tank ≠ pos)) →
    (∀ (pos₁ pos₂ : fin 41 × fin 41), ∃ (shots : ℕ), shots ≤ n ∧ 
      ∀ (tank_initial : fin 41 × fin 41) (moves : fin 41 × fin 41 → fin 41 × fin 41),
        tank_move tank_initial moves →
        (moves tank_initial) = pos₁ ∨ (moves (moves tank_initial)) = pos₂) :=
  exists.intro 2521 sorry

end minimum_shots_to_destroy_tank_l663_663816


namespace determine_a_range_l663_663941

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem determine_a_range (e : ℝ) (he : e = Real.exp 1) :
  ∃ a_range : Set ℝ, a_range = Set.Icc 1 (e + 1 / e) :=
by 
  sorry

end determine_a_range_l663_663941


namespace cubic_identity_l663_663242

theorem cubic_identity (x : ℝ) (hx : x + 1/x = -5) : x^3 + 1/x^3 = -110 := by
  sorry

end cubic_identity_l663_663242


namespace find_f_zero_l663_663956

def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d, ∀ x, f(x) = x^4 + a * x^3 + b * x^2 + c * x + d

theorem find_f_zero (f : ℝ → ℝ)
  (hf_monic : is_monic_quartic f)
  (hf1 : f (-2) = -4)
  (hf2 : f (1) = -1)
  (hf3 : f (3) = -9)
  (hf4 : f (5) = -25) :
  f (0) = -30 :=
sorry

end find_f_zero_l663_663956


namespace sequence_a_forth_value_l663_663654

theorem sequence_a_forth_value :
  let a : ℕ → ℕ := λ n, Nat.recOn n 1 (λ n a_n, 3 * a_n + 1)
  in a 4 = 40 :=
by
  sorry

end sequence_a_forth_value_l663_663654


namespace Joey_age_digit_sum_l663_663277

structure Ages :=
  (joey_age : ℕ)
  (chloe_age : ℕ)
  (zoe_age : ℕ)

def is_multiple (a b : ℕ) : Prop :=
  ∃ k, a = k * b

def sum_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem Joey_age_digit_sum
  (C J Z : ℕ)
  (h1 : J = C + 1)
  (h2 : Z = 1)
  (h3 : ∃ n, C + n = (n + 1) * m)
  (m : ℕ) (hm : m = 9)
  (h4 : C - 1 = 36) :
  sum_digits (J + 37) = 12 :=
by
  sorry

end Joey_age_digit_sum_l663_663277


namespace infinite_k_lcm_gt_ck_l663_663290

theorem infinite_k_lcm_gt_ck 
  (a : ℕ → ℕ) 
  (distinct_pos : ∀ n m : ℕ, n ≠ m → a n ≠ a m) 
  (pos : ∀ n, 0 < a n) 
  (c : ℝ) 
  (c_pos : 0 < c) 
  (c_lt : c < 1.5) : 
  ∃ᶠ k in at_top, (Nat.lcm (a k) (a (k + 1)) : ℝ) > c * k :=
sorry

end infinite_k_lcm_gt_ck_l663_663290


namespace president_vice_secretary_choice_l663_663989

theorem president_vice_secretary_choice (n : ℕ) (h : n = 6) :
  (∀ a b c : fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (n * (n - 1) * (n - 2) = 120) := 
sorry

end president_vice_secretary_choice_l663_663989


namespace ages_correct_l663_663791

def current_ages (A B : ℝ) : Prop := 
  (A = 2 * (2 * B - A)) ∧ 
  (3 * A - B = 130)

def A : ℝ := 520 / 9  -- 57 7/9
def B : ℝ := 1560 / 36  -- 43 1/3

theorem ages_correct : current_ages A B :=
by
  sorry

end ages_correct_l663_663791


namespace water_charging_standard_l663_663022

theorem water_charging_standard
  (x y : ℝ)
  (h1 : 10 * x + 5 * y = 35)
  (h2 : 10 * x + 8 * y = 44) : 
  x = 2 ∧ y = 3 :=
by
  sorry

end water_charging_standard_l663_663022


namespace sum_of_consecutive_integers_of_sqrt3_l663_663206

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l663_663206


namespace capacity_of_can_is_eight_l663_663979

noncomputable def capacity_of_can : ℕ :=
  let M := 1 in -- derived from 10 = 10M above
  let W := 5 * M in
  let V_initial := M + W in
  let V_total := V_initial + 2 in
  V_total

theorem capacity_of_can_is_eight (h1 : ∀ M W, M / W = 1 / 5) (h2 : ∀ M, (M + 2) / (5 * M) = 3 / 5) : capacity_of_can = 8 := by
  sorry

end capacity_of_can_is_eight_l663_663979


namespace binomial_ratio_l663_663908

open Nat

theorem binomial_ratio (n k : ℕ) (hnk : n > k)
    (h_ratio : (binom n (k-1)) / (binom n k) = 1 ∧ 
               (binom n k) / (binom n (k+1)) = 2) : n + k = 3 :=
by
  sorry

end binomial_ratio_l663_663908


namespace point_on_line_l663_663516

theorem point_on_line (x : ℝ) : 
  (∃ x, (x, 7) lies_on line_through (0,4) (-6,1)) → x = 6 := 
by
  sorry

end point_on_line_l663_663516


namespace consecutive_integers_sum_l663_663235

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l663_663235


namespace cannot_form_right_angled_triangle_l663_663080

theorem cannot_form_right_angled_triangle (a b c : ℝ) (h : {a, b, c} = {2, 2, 3}) : ¬(a^2 + b^2 = c^2) :=
by
  sorry

end cannot_form_right_angled_triangle_l663_663080


namespace count_fair_numbers_less_than_100_l663_663545

-- Define the number of divisors of n that are perfect squares
def num_perfect_square_divisors (n : ℕ) : ℕ := 
  ∏ p in n.factorization.support, (n.factorization p / 2 + 1)

-- Define the number of divisors of n that are perfect cubes
def num_perfect_cube_divisors (n : ℕ) : ℕ := 
  ∏ p in n.factorization.support, (n.factorization p / 3 + 1)

-- Define a number being fair based on the given conditions
def is_fair (n : ℕ) : Prop := 
  let s := num_perfect_square_divisors n 
  let c := num_perfect_cube_divisors n
  s = c ∧ s > 1

-- Prove there are exactly 7 fair numbers less than 100
theorem count_fair_numbers_less_than_100 : 
  (Finset.range 100).filter is_fair = 7 := 
by
  -- We use "by" to set up the context for the proof that will follow.
  sorry

end count_fair_numbers_less_than_100_l663_663545


namespace sum_of_consecutive_integers_l663_663221

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l663_663221


namespace positive_values_x_l663_663628

theorem positive_values_x (x : ℕ) : 
  (100 ≤ x ∧ x < 1000) ∧ 
  (100 ≤ 2 * x ∧ 2 * x < 1000) ∧ 
  (3 * x ≥ 1000) ↔ 166 :=
by
  sorry

end positive_values_x_l663_663628


namespace domain_of_F_F_is_odd_positive_diff_a_gt_1_positive_diff_a_lt_1_l663_663192

noncomputable def f (a x : ℝ) := log a (3 * x + 1)
noncomputable def g (a x : ℝ) := log a (1 - 3 * x)
noncomputable def F (a x : ℝ) := f a x - g a x

variables (a x : ℝ) (h : a > 0) (h1 : a ≠ 1)

-- 1. Domain of F(x)
theorem domain_of_F : (-1/3 < x ∧ x < 1/3) ↔ (3*x + 1 > 0 ∧ 1 - 3*x > 0) :=
sorry

-- 2. Parity of F(x)
theorem F_is_odd : F a x = - F a (-x) :=
sorry

-- 3. f(x) - g(x) > 0 solutions
theorem positive_diff_a_gt_1 : a > 1 → (0 < x ∧ x < 1/3) ↔ F a x > 0 :=
sorry

theorem positive_diff_a_lt_1 : (0 < a ∧ a < 1) → (-1/3 < x ∧ x < 0) ↔ F a x > 0 :=
sorry

end domain_of_F_F_is_odd_positive_diff_a_gt_1_positive_diff_a_lt_1_l663_663192


namespace gauss_company_percent_five_years_or_more_l663_663001

def num_employees_less_1_year (x : ℕ) : ℕ := 5 * x
def num_employees_1_to_2_years (x : ℕ) : ℕ := 5 * x
def num_employees_2_to_3_years (x : ℕ) : ℕ := 8 * x
def num_employees_3_to_4_years (x : ℕ) : ℕ := 3 * x
def num_employees_4_to_5_years (x : ℕ) : ℕ := 2 * x
def num_employees_5_to_6_years (x : ℕ) : ℕ := 2 * x
def num_employees_6_to_7_years (x : ℕ) : ℕ := 2 * x
def num_employees_7_to_8_years (x : ℕ) : ℕ := x
def num_employees_8_to_9_years (x : ℕ) : ℕ := x
def num_employees_9_to_10_years (x : ℕ) : ℕ := x

def total_employees (x : ℕ) : ℕ :=
  num_employees_less_1_year x +
  num_employees_1_to_2_years x +
  num_employees_2_to_3_years x +
  num_employees_3_to_4_years x +
  num_employees_4_to_5_years x +
  num_employees_5_to_6_years x +
  num_employees_6_to_7_years x +
  num_employees_7_to_8_years x +
  num_employees_8_to_9_years x +
  num_employees_9_to_10_years x

def employees_with_5_years_or_more (x : ℕ) : ℕ :=
  num_employees_5_to_6_years x +
  num_employees_6_to_7_years x +
  num_employees_7_to_8_years x +
  num_employees_8_to_9_years x +
  num_employees_9_to_10_years x

theorem gauss_company_percent_five_years_or_more (x : ℕ) :
  (employees_with_5_years_or_more x : ℝ) / (total_employees x : ℝ) * 100 = 30 :=
by
  sorry

end gauss_company_percent_five_years_or_more_l663_663001


namespace sixth_cube_is_green_l663_663770

def color :=
  | blue
  | yellow
  | red
  | green

def valid_arrangement (cubes : List color) : Prop :=
  cubes.length = 12 ∧
  (cubes.count color.blue = 3) ∧
  (cubes.count color.yellow = 2) ∧
  (cubes.count color.red = 3) ∧
  (cubes.count color.green = 4) ∧
  (cubes.head = some color.yellow ∨ cubes.head = some color.red ∨
    cubes.last = some color.yellow ∨ cubes.last = some color.red) ∧
  (∃ start, start < 10 ∧ (cubes.slice start (start + 3)).all (λ c, c = color.red)) ∧
  (∃ start, start < 9 ∧ (cubes.slice start (start + 4)).all (λ c, c = color.green)) ∧
  (cubes.get! 9 = color.blue)

theorem sixth_cube_is_green (cubes : List color) (h : valid_arrangement cubes) : 
  cubes.get! 5 = color.green :=
sorry

end sixth_cube_is_green_l663_663770


namespace sequence_proof_l663_663065

noncomputable def Sn (b : ℕ → ℚ) (n : ℕ) : ℚ := ∑ i in Finset.range n, b i

noncomputable def bn (S : ℕ → ℚ) (n : ℕ) : ℚ := 2 - 2 * S n

def arith_seq_an (n : ℕ) : ℕ := 3 * n - 1 -- assuming based on calculation from solution

def cn (a b: ℕ → ℚ) (n : ℕ): ℚ := a n * b n

noncomputable def Tn (c: ℕ → ℚ) (n : ℕ) : ℚ := ∑ i in Finset.range n, c i

theorem sequence_proof :
  (∀ n, S n = Sn b n) →
  (∀ n, b n = bn S n) →
  (a 5 = 14) →
  (a 7 = 20) →
  (∀ n, b n = 2 * (1 / 3)^n) ∧
  (Tn c n = 7 / 2 - 1 / (2 * 3^(n - 2)) - (3 * n - 1) / 3^n) :=
by
  intro hS hb ha5 ha7
  sorry

end sequence_proof_l663_663065


namespace problem_1_problem_2_l663_663927

variable {c : ℝ}

def p (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → c ^ x₁ > c ^ x₂

def q (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, (1 / 2) < x₁ ∧ x₁ < x₂ → (x₁ ^ 2 - 2 * c * x₁ + 1) < (x₂ ^ 2 - 2 * c * x₂ + 1)

theorem problem_1 (hc : 0 < c) (hcn1 : c ≠ 1) (hp : p c) (hnq_false : ¬ ¬ q c) : 0 < c ∧ c ≤ 1 / 2 :=
by
  sorry

theorem problem_2 (hc : 0 < c) (hcn1 : c ≠ 1) (hpq_false : ¬ (p c ∧ q c)) (hp_or_q : p c ∨ q c) : 1 / 2 < c ∧ c < 1 :=
by
  sorry

end problem_1_problem_2_l663_663927


namespace sum_of_solutions_eq_l663_663354

theorem sum_of_solutions_eq:
  let a := (2:ℝ)
  let b := (-13:ℝ)
  let c := (-30:ℝ)
  let quadratic_eq := (∀ x, 2 * x^2 - 8 * x - 10 = 5 * x + 20)
  in -b / a = 13 / 2 :=
by 
  sorry

end sum_of_solutions_eq_l663_663354


namespace general_term_of_sequence_smallest_n_for_sum_ineq_l663_663158

-- Problem Part I: Prove the general term formula of the arithmetic sequence.
theorem general_term_of_sequence (d : ℝ) (h_d_nonzero : d ≠ 0) (a : ℕ → ℝ) (h_a1 : a 1 = 2)
    (h_arith_seq : ∀ n, a (n + 1) = a n + d) (h_geom_seq : a 1 * a 5 = (a 2)^2) :
  ∃ (f : ℕ → ℝ), (∀ n, a n = f n) ∧ (f = λ n, 4 * n - 2) :=
by
  sorry

-- Problem Part II: Prove the smallest positive integer n such that Sn > 60n + 800.
theorem smallest_n_for_sum_ineq (a : ℕ → ℝ) (S : ℕ → ℝ) (h_a1 : a 1 = 2)
    (h_arith_seq : ∀ n, a (n + 1) = a n + 4) (h_S : ∀ n, S n = n * (a 1 + a n) / 2) :
  ∃ n : ℕ, n > 40 ∧ S n > 60 * n + 800 :=
by
  sorry

end general_term_of_sequence_smallest_n_for_sum_ineq_l663_663158


namespace transport_cost_B_condition_l663_663365

-- Define the parameters for coal from Mine A
def calories_per_gram_A := 4
def price_per_ton_A := 20
def transport_cost_A := 8

-- Define the parameters for coal from Mine B
def calories_per_gram_B := 6
def price_per_ton_B := 24

-- Define the total cost for transporting one ton from Mine A to city N
def total_cost_A := price_per_ton_A + transport_cost_A

-- Define the question as a Lean theorem
theorem transport_cost_B_condition : 
  ∀ (transport_cost_B : ℝ), 
  (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) → 
  transport_cost_B = 18 :=
by
  intros transport_cost_B h
  have h_eq : (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) := h
  sorry

end transport_cost_B_condition_l663_663365


namespace triangle_tangent_segments_sum_one_l663_663298

variable (a b c a_1 b_1 c_1 : ℝ)

noncomputable def is_tangent_segment (tri_side parallel_side tangent_segment : ℝ) : Prop :=
∃ Δ : Type, ∃ incircle : Δ → ℝ, -- Existence of an incircle
  parallel_to_side : Δ → (ℝ → ℝ) → Prop, -- Parallelism to the given side
  tangent_lines : Δ → (ℝ → ℝ) → Prop, -- Tangency to the incircle
  (parallel_to_side Δ (λ x, x = parallel_side)) ∧ (tangent_lines Δ (λ x, x = tangent_segment))

theorem triangle_tangent_segments_sum_one 
  (h1 : is_tangent_segment a a_1)
  (h2 : is_tangent_segment b b_1)
  (h3 : is_tangent_segment c c_1) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 → 
  (a_1 / a) + (b_1 / b) + (c_1 / c) = 1 := 
by
  sorry

end triangle_tangent_segments_sum_one_l663_663298


namespace A_salary_l663_663404

theorem A_salary (x y : ℝ) (h1 : x + y = 7000) (h2 : 0.05 * x = 0.15 * y) : x = 5250 :=
by
  sorry

end A_salary_l663_663404


namespace sequence_product_l663_663754

theorem sequence_product : 
  (∀ n : ℕ, 0 < n → 
    ∑ i in finset.range n, 3 ^ (i - 1) * a i = n / 3) → 
  (∏ i in finset.range 10, a i) = (1 / 3) ^ 55 := 
by
  sorry

end sequence_product_l663_663754


namespace solve_tangent_circle_l663_663883

noncomputable def equation_of_tangent_circle : Prop :=
  ∃ (a b : ℝ), 
    (∀ x y : ℝ, (x + 3)^2 + (y - 6)^2 = 20) ∧ 
    (x^2 + y^2 = 5) ∧ 
    (x = -1 ∧ y = 2) ∧ 
    (sqrt ((-1 - a)^2 + (2 - b)^2) = 2 * sqrt 5) ∧ 
    (b / a = b - 2 / (a + 1))

theorem solve_tangent_circle : equation_of_tangent_circle :=
  sorry

end solve_tangent_circle_l663_663883


namespace all_nonnegative_or_nonpositive_l663_663720

theorem all_nonnegative_or_nonpositive (n : ℕ) (c : ℕ → ℝ) (h1 : n ≥ 2) 
  (h2 : (n - 1) * (∑ i in Finset.range n, c i ^ 2) = (∑ i in Finset.range n, c i) ^ 2) :
  (∀ i, 0 ≤ c i) ∨ (∀ i, c i ≤ 0) := 
sorry

end all_nonnegative_or_nonpositive_l663_663720


namespace probability_neither_red_nor_purple_l663_663394

theorem probability_neither_red_nor_purple : 
  let total_balls := 60
  let white_balls := 22
  let green_balls := 10
  let yellow_balls := 7
  let red_balls := 15
  let purple_balls := 6
  (white_balls + green_balls + yellow_balls) / total_balls = 0.65 :=
by
  -- Definitions
  let total_balls := 60
  let white_balls := 22
  let green_balls := 10
  let yellow_balls := 7
  let red_balls := 15
  let purple_balls := 6

  -- Assertion of equality
  have h : (white_balls + green_balls + yellow_balls) / total_balls = 0.65
  sorry

end probability_neither_red_nor_purple_l663_663394


namespace sum_of_internal_angles_lt_360_l663_663268

-- Definition of spatial quadrilateral
structure SpatialQuadrilateral (A B C D : Type) :=
  (not_coplanar : ¬∃ P : C, A ⊆ P ∧ B ⊆ P ∧ C ⊆ P ∧ D ⊆ P)

-- Definition of internal angles of spatial quadrilateral
def internal_angles (A B C D : Type) : Prop :=
  θ = ∠ABC + ∠BCD + ∠CDA + ∠DAB

-- Theorem statement
theorem sum_of_internal_angles_lt_360 (A B C D : Type) (quad : SpatialQuadrilateral A B C D) :
  ∃ θ, internal_angles A B C D ∧ θ < 360 :=
sorry

end sum_of_internal_angles_lt_360_l663_663268


namespace nf_n_lt_harmonic_series_l663_663556

def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem nf_n_lt_harmonic_series (n : ℕ) (h1 : n ≥ 2) :
  n * f n < 2 + ∑ i in Finset.range (n - 1), (1 / (i + 1)) := 
sorry

end nf_n_lt_harmonic_series_l663_663556


namespace find_f_values_l663_663294

noncomputable def f : ℕ → ℕ := sorry

-- Function is strictly increasing on natural numbers (positive integers in context)
axiom f_increasing : ∀ {a b : ℕ}, a < b → f(a) < f(b)
-- Given functional equation
axiom f_nested : ∀ (k : ℕ), f(f(k)) = 3 * k

-- Main theorem
theorem find_f_values : f(1) + f(9) + f(10) = 39 :=
sorry

end find_f_values_l663_663294


namespace volume_of_polyhedron_l663_663904

-- Definitions derived from the conditions
def isVertexOfCube (v : ℝ × ℝ × ℝ) (a : ℝ) : Prop :=
  ∃ (i j k : ℤ), (i = 0 ∨ i = 1) ∧ (j = 0 ∨ j = 1) ∧ (k = 0 ∨ k = 1) ∧
  (v = (i * a, j * a, k * a))

def isBaseArea (B : ℝ) (a : ℝ) : Prop :=
  B = a ^ 2

def isHeight (h : ℝ) (a : ℝ) : Prop :=
  h = a

def volumeOfPyramid (a : ℝ) (B : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * B * h

theorem volume_of_polyhedron :
  ∃ (a : ℝ), a = 1 ∧
  ∃ (B : ℝ), isBaseArea B a ∧
  ∃ (h : ℝ), isHeight h a ∧
  volumeOfPyramid a B h = 10 := by
  sorry

end volume_of_polyhedron_l663_663904


namespace find_vertex_angle_l663_663005

-- Define the conditions of the problem
variables (A : Type) [point A] 
variables (V : Type) [vertex A V]
variables (cone1 cone2 cone3 cone4 : Type) [cone cone1] [cone cone2] [cone cone3] [cone cone4]
variables (vertex_angle_cone3 : angle) (vertex_angle_cone4 : angle)

-- Define specific angles from the conditions
def vertex_angle_cone3 := (π / 3)
def vertex_angle_cone4 := (5 * π / 6)

-- Define what we need to prove
theorem find_vertex_angle (2α : angle) :
  (∃ α : angle, 2α = 2 * arctan (sqrt 3 - 1)) := 
sorry

end find_vertex_angle_l663_663005


namespace find_line_equation_l663_663177

noncomputable def line_equation : real × real × real := sorry

theorem find_line_equation (m : real) :
  (∃ (m : real), (3, 4, m) = line_equation ∧
    let a := -m / 4 in
    let b := -m / 3 in
    (1 / 2) * |a| * |b| = 24) :=
begin
  sorry
end

end find_line_equation_l663_663177


namespace original_price_l663_663823

theorem original_price (sale_price gain_percent : ℕ) (h_sale : sale_price = 130) (h_gain : gain_percent = 30) : 
    ∃ P : ℕ, (P * (1 + gain_percent / 100)) = sale_price := 
by
  use 100
  rw [h_sale, h_gain]
  norm_num
  sorry

end original_price_l663_663823


namespace max_knights_seated_next_to_two_knights_l663_663493

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l663_663493


namespace circle_diameter_l663_663391

theorem circle_diameter (r : ℕ) (h : r = 7) : 2 * r = 14 :=
by
  rw [h]
  norm_num

end circle_diameter_l663_663391


namespace percentage_spent_l663_663379

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h_initial : initial_amount = 1200) 
  (h_remaining : remaining_amount = 840) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 :=
by
  sorry

end percentage_spent_l663_663379


namespace distance_between_foci_of_ellipse_l663_663084

theorem distance_between_foci_of_ellipse :
  ∀ (ellipse: ℝ × ℝ → Prop),
    (∀ x y, ellipse (x, y) ↔ (x - 5)^2 / 25 + (y - 2)^2 / 4 = 1) →
    ∃ c : ℝ, 2 * c = 2 * Real.sqrt (25 - 4) :=
by
  intro ellipse h
  use Real.sqrt (25 - 4)
  sorry

end distance_between_foci_of_ellipse_l663_663084


namespace product_of_20_random_digits_ends_in_0_probability_l663_663571

theorem product_of_20_random_digits_ends_in_0_probability :
  let prob_at_least_one_0 := 1 - (9 / 10) ^ 20,
      prob_even_digit := 1 - (5 / 9) ^ 20,
      prob_5 := 1 - (8 / 9) ^ 19
  in 
    prob_at_least_one_0 + ( (9 / 10) ^ 20 * prob_even_digit * prob_5 ) ≈ 0.988 :=
by sorry

end product_of_20_random_digits_ends_in_0_probability_l663_663571


namespace num_memorable_numbers_l663_663510

def is_memorable (d : Fin 10 → Fin 10) : Prop :=
  (d 0 = d 1 ∧ d 1 = d 2 ∧ d 2 = d 3 ∧ d 3 = d 4) ∨
  (d 1 = d 2 ∧ d 2 = d 3 ∧ d 3 = d 4 ∧ d 4 = d 5) ∨
  (d 2 = d 3 ∧ d 3 = d 4 ∧ d 4 = d 5 ∧ d 5 = d 6)

theorem num_memorable_numbers : 
  (Fin 10 → Fin 10) → Justis_memorable dmemorable d_contrad d_count
  28000 sorry колбу

end num_memorable_numbers_l663_663510


namespace sine_function_parameters_l663_663097

theorem sine_function_parameters : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ x, a * sin(b * x + c) ≤ 3) ∧ 
  (∃ x, a * sin(b * x + c) = 3) ∧ 
  (2 * π / b = 2 * π) ∧ 
  (a * sin(b * (π / 6) + c) = 3) := 
by
  use 3, 1, π / 3
  sorry

end sine_function_parameters_l663_663097


namespace sum_series_eq_11_over_12_l663_663528

theorem sum_series_eq_11_over_12 :
  (∑ n in Finset.range 9, 1 / ((n + 1) * (n + 2)) + 1 / (10 * 12)) = 11 / 12 :=
by
  sorry

end sum_series_eq_11_over_12_l663_663528


namespace value_preserving_interval_x_squared_value_preserving_interval_x_squared_plus_m_l663_663163

theorem value_preserving_interval_x_squared :
  ∀ (a b : ℝ), 0 ≤ a → 0 ≤ b → a < b → 
  (∀ (x : ℝ), a ≤ x → x ≤ b → a ≤ x^2 → x^2 ≤ b) → 
  (∀ (y : ℝ), ∃ (x : ℝ), a ≤ x → x ≤ b → y = x^2) → 
  [a, b] = [0, 1] :=
begin
  sorry
end

theorem value_preserving_interval_x_squared_plus_m (m : ℝ) (h : m ≠ 0) :
  ∀ (a b : ℝ), a < b → 
  (∀ (x : ℝ), a ≤ x → x ≤ b → a ≤ x^2 + m → x^2 + m ≤ b) → 
  (∀ (y : ℝ), ∃ (x : ℝ), a ≤ x → x ≤ b → y = x^2 + m) → 
  (m ∈ (Set.Ico (-1 : ℝ) (-3 / 4 : ℝ) ∪ Set.Ioo 0 (1 / 4))) :=
begin
  sorry
end

end value_preserving_interval_x_squared_value_preserving_interval_x_squared_plus_m_l663_663163


namespace tan_17pi_over_4_l663_663876

theorem tan_17pi_over_4 : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_17pi_over_4_l663_663876


namespace cos_alpha_theorem_l663_663639

noncomputable def cosAlphaProof : Prop :=
  ∃ α : ℝ, (sin (real.pi - α) = 1 / 3 ∧ (real.pi / 2 ≤ α ∧ α ≤ real.pi)) → cos α = - (2 * real.sqrt 2) / 3

theorem cos_alpha_theorem : cosAlphaProof :=
sorry

end cos_alpha_theorem_l663_663639


namespace line_through_circle_center_parallel_l663_663132

theorem line_through_circle_center_parallel (x y : ℝ) :
  let circle_eq := x^2 + y^2 - 2*x + 2*y = 0
  let line_parallel := (2 : ℝ)*x - y = 0
  let center := (1, -1)                    -- Center of the circle
  2*x - y - 3 = 0 :=
begin
  sorry
end

end line_through_circle_center_parallel_l663_663132


namespace tan_theta_plus_45_l663_663623

theorem tan_theta_plus_45 (θ : ℝ) (h : (2:ℝ, 1) ∥ (Real.sin θ, Real.cos θ)) :
  Real.tan (θ + Real.pi / 4) = -3 :=
sorry

end tan_theta_plus_45_l663_663623


namespace correct_propositions_l663_663078

-- Proposition 1
def P1 (x y : ℝ) : Prop :=
  (x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4

-- Proposition 2
def P2 : Prop :=
  (∀ n : ℤ, (∃ k : ℤ, n = 2 * k) → (∃ k : ℤ, ¬(n = 2 * k + 1)))

-- Proposition 3 (Original)
def P3 (a x : ℝ) : Prop :=
  |a| ≤ 1 → ¬((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)

-- Proposition 3 (Converse)
def P3_converse (a x : ℝ) : Prop :=
  ¬((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) → |a| ≤ 1

-- Proposition 4
def P4 (m : ℝ) : Prop :=
  ∀ x y : ℝ, (m = 1 ↔ ((m + 2) * x + m * y + 1 = 0) → ((m - 2) * x + (m + 2) * y - 3 = 0) → x = 0 ∧ y = 0)

theorem correct_propositions :
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) :=
by
  sorry

end correct_propositions_l663_663078


namespace any_nat_in_frac_l663_663389

theorem any_nat_in_frac (n : ℕ) : ∃ x y : ℕ, y ≠ 0 ∧ x^2 = y^3 * n := by
  sorry

end any_nat_in_frac_l663_663389


namespace sarah_sleep_hours_l663_663318

-- Given conditions
def sleep_score_constant := 60 * 6 = 360
def average_score (s1 s2 : ℕ) := (s1 + s2) / 2

-- Prove how many hours Sarah needs to sleep before the second exam
theorem sarah_sleep_hours 
    (constant : sleep_score_constant) -- condition 1 
    (sleep1 : 6) -- condition 2
    (score1 : 60) -- condition 2
    (avg : average_score 60 s2 = 85): -- condition 3
    h = 360 / 110 := 
sorry

end sarah_sleep_hours_l663_663318


namespace binary_to_decimal_l663_663987

-- Define the function to compute the decimal value of a binary number.
def decimal_value (b : string) : ℕ :=
  b.foldr (λ c acc => acc * 2 + if c = '1' then 1 else 0) 0

-- The binary number given in the problem.
def binary_number := "10011"

-- We prove that the decimal value of the binary number "10011" is 19.
theorem binary_to_decimal :
  decimal_value binary_number = 19 :=
by
  sorry

end binary_to_decimal_l663_663987


namespace sum_of_digits_is_15_l663_663658

theorem sum_of_digits_is_15
  (A B C D E : ℕ) 
  (h_distinct: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h_digits: A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
  (h_divisible_by_9: (A * 10000 + B * 1000 + C * 100 + D * 10 + E) % 9 = 0) 
  : A + B + C + D + E = 15 := 
sorry

end sum_of_digits_is_15_l663_663658


namespace hyperbola_eccentricity_l663_663932

namespace HyperbolaEccentricity

-- Given conditions
def is_hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def asymptotic_angle (angle : ℝ) : Prop :=
  angle = Real.pi / 6

-- The primary question translated into a proof problem
theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : is_hyperbola a b b)
  (h2 : asymptotic_angle (Real.pi / 6))
  (h3 : b = a * Real.sqrt(3) / 3) :
  e = 2 * Real.sqrt(3) / 3 :=
sorry

end HyperbolaEccentricity

end hyperbola_eccentricity_l663_663932


namespace area_difference_l663_663743

theorem area_difference (d_square d_circle : ℝ) (diag_eq : d_square = 10) (diam_eq : d_circle = 10) : 
  let s := d_square / real.sqrt 2 in
  let area_square := s * s in
  let r := d_circle / 2 in
  let area_circle := real.pi * r * r in
  (area_circle - area_square) = 28.5 := by
s := d_square / real.sqrt 2
area_square := s * s
r := d_circle / 2
area_circle := real.pi * r * r
suffices : area_circle - area_square ≈ 28.5
by sorry

end area_difference_l663_663743


namespace consecutive_integers_sum_l663_663240

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l663_663240


namespace sum_of_eligible_primes_is_223_l663_663892
open Nat
  
def is_prime (n : ℕ) : Prop := Nat.Prime n

def reversed_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λacc d => acc * 10 + d) 0
  
def eligible_prime (n : ℕ) : Prop :=
  is_prime n ∧ n >= 20 ∧ n < 100 ∧
  let r := reversed_digits n
  is_prime r ∧ (r % 10 = 1 ∨ r % 10 = 7)

def sum_eligible_primes : ℕ :=
  (filter eligible_prime (List.range 100)).sum
  
theorem sum_of_eligible_primes_is_223 : sum_eligible_primes = 223 := by
  -- Proof would go here
  sorry

end sum_of_eligible_primes_is_223_l663_663892


namespace circumradius_eq_incenter_circumcenter_distance_l663_663999

-- Define acute-angled triangle conditions
structure Triangle (α : Type) [LinearOrderedField α] :=
  (a b c : α)
  (acute : ∀ x ∈ [a, b, c], x ∠ x < π/2) -- Input that all angles are acute
  (side_ineq : a < b ∧ b < c)
  (pt_D : α) (pt_E : α)
  (BD_eq_BC_CE : b = c ∧ c = pt_E)

-- Define the circumradius of a triangle
noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * (sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4))

-- Define the incenter and circumcenter distance
noncomputable def incenter_circumcenter_distance (a b c : ℝ) : ℝ :=
  sqrt ((circumradius a b c) ^ 2 - 2 * (a * b * c) / (a + b + c))

-- The statement to be proved in Lean
theorem circumradius_eq_incenter_circumcenter_distance {α : Type} [LinearOrderedField α]
  {a b c : α} {triangle : Triangle α} :
  ∃ (R' : ℝ), 
  R' = incenter_circumcenter_distance a b c 
  :=
sorry

end circumradius_eq_incenter_circumcenter_distance_l663_663999


namespace consecutive_integers_sum_l663_663238

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l663_663238


namespace problem_part1_problem_part2_l663_663562

noncomputable def g (x : Real) : Real := Real.cos x
noncomputable def f (x : Real) : Real := 2 * Real.sin x

theorem problem_part1 (α β m : Real) (hα : 0 ≤ α ∧ α < 2 * Real.pi) (hβ : 0 ≤ β ∧ β < 2 * Real.pi)
  (h : f(α) + g(α) = m ∧ f(β) + g(β) = m) (hαβ : α ≠ β) :
  (-Real.sqrt 5) < m ∧ m < Real.sqrt 5 :=
sorry

theorem problem_part2 (α β m : Real) (hα : 0 ≤ α ∧ α < 2 * Real.pi) (hβ : 0 ≤ β ∧ β < 2 * Real.pi)
  (h : f(α) + g(α) = m ∧ f(β) + g(β) = m) (hαβ : α ≠ β) :
  Real.cos (α - β) = (2 * m^2) / 5 - 1 :=
sorry

end problem_part1_problem_part2_l663_663562


namespace point_on_y_axis_l663_663259

theorem point_on_y_axis (x y : ℝ) (h : x = 0 ∧ y = -1) : y = -1 := by
  -- Using the conditions directly
  cases h with
  | intro hx hy =>
    -- The proof would typically follow, but we include sorry to complete the statement
    sorry

end point_on_y_axis_l663_663259


namespace parallel_planes_l663_663606

-- Given α, β, and γ are three different planes
variables (α β γ : Plane)

-- Condition 1: α is parallel to β
axiom h1 : Parallel α β

-- Condition 2: β is parallel to γ
axiom h2 : Parallel β γ

-- We need to prove that α is parallel to γ
theorem parallel_planes : Parallel α γ :=
by { sorry }

end parallel_planes_l663_663606


namespace volume_of_cube_B_is_64_l663_663387

-- Define the volume of a cube in terms of its side length
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Define the surface area of a cube in terms of its side length
def surface_area_of_cube (s : ℝ) : ℝ := 6 * s^2

-- Given conditions
def V_A := 8
def A_A := surface_area_of_cube (real.cbrt V_A)
def A_B := 4 * A_A

theorem volume_of_cube_B_is_64 : 
  ∃ s_B : ℝ, 
    surface_area_of_cube s_B = A_B ∧ volume_of_cube s_B = 64 :=
by 
  sorry

end volume_of_cube_B_is_64_l663_663387


namespace find_initial_value_summing_to_6_l663_663822

def machine (N : ℕ) : ℕ :=
  if N % 2 = 1 then 5 * N + 3
  else if N % 3 = 0 then N / 3
  else N + 1

theorem find_initial_value_summing_to_6 :
  machine (machine (machine (machine (machine 6)))) = 1 ∧ (∑ N in {6}, N) = 6 := by
  sorry

end find_initial_value_summing_to_6_l663_663822


namespace number_of_edges_R_l663_663425

/-
  Given:
  1. A convex polyhedron Q has n vertices and 150 edges.
  2. Each vertex V_k of Q is cut by a unique plane P_k.
  3. These planes P_k only cut edges meeting at V_k and do not intersect inside or on the surface of Q.
  4. After cutting the polyhedron, n pyramids and a new polyhedron R are formed.

  Prove:
  The number of edges of the new polyhedron R is 450.
-/

open_locale classical
noncomputable theory

def polyhedron (vertices : ℕ) (edges : ℕ) := 
  { Q : Type | vertices = n ∧ edges = 150 }

def unique_planes (V : ℕ → Type) := 
  { P : Type | ∀ k, cuts_edges_meeting_at P V_k }

def no_intersection (P : Type) (Q : Type) := 
  ¬ ∃ x ∈ P, x ∈ Q

theorem number_of_edges_R (n : ℕ) (Q : Type) 
  [polyhedron n 150 Q] (P : ℕ → Type) 
  [unique_planes P] [no_intersection P Q] : 
  edges_of (new_polyhedron R) = 450 :=
sorry

end number_of_edges_R_l663_663425


namespace range_of_m_l663_663590

-- Definitions based on the conditions
def p (m : ℝ) : Prop := 4 - 4 * m > 0
def q (m : ℝ) : Prop := m + 2 > 0

-- Problem statement in Lean 4
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ≤ -2 ∨ m ≥ 1 := by
  sorry

end range_of_m_l663_663590


namespace evaluate_expression_l663_663523

theorem evaluate_expression :
  (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 :=
by
  sorry

end evaluate_expression_l663_663523


namespace value_of_n_l663_663255

-- Define required conditions
variables (n : ℕ) (f : ℕ → ℕ → ℕ)

-- Conditions
axiom cond1 : n > 7
axiom cond2 : ∀ m k : ℕ, f m k = 2^(n - m) * Nat.choose m k

-- Given condition
axiom after_seventh_round : f 7 5 = 42

-- Theorem to prove
theorem value_of_n : n = 8 :=
by
  -- Proof goes here
  sorry

end value_of_n_l663_663255


namespace max_integer_le_x0_zero_of_ln_x_plus_2x_minus_6_l663_663249

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem max_integer_le_x0_zero_of_ln_x_plus_2x_minus_6 :
  ∃ (x0 : ℝ), f x0 = 0 ∧ ∀ k : ℤ, k ≤ x0 → k ≤ 2 :=
begin
  sorry
end

end max_integer_le_x0_zero_of_ln_x_plus_2x_minus_6_l663_663249


namespace pool_capacity_is_800_l663_663003

-- Definitions for the given problem conditions
def fill_time_all_valves : ℝ := 36
def fill_time_first_valve : ℝ := 180
def fill_time_second_valve : ℝ := 240
def third_valve_more_than_first : ℝ := 30
def third_valve_more_than_second : ℝ := 10
def leak_rate : ℝ := 20

-- Function definition for the capacity of the pool
def capacity (W : ℝ) : Prop :=
  let V1 := W / fill_time_first_valve
  let V2 := W / fill_time_second_valve
  let V3 := (W / fill_time_first_valve) + third_valve_more_than_first
  let effective_rate := V1 + V2 + V3 - leak_rate
  (W / fill_time_all_valves) = effective_rate

-- Proof statement that the capacity of the pool is 800 cubic meters
theorem pool_capacity_is_800 : capacity 800 :=
by
  -- Proof is omitted
  sorry

end pool_capacity_is_800_l663_663003


namespace intersection_exists_implies_m_range_l663_663175

theorem intersection_exists_implies_m_range (m : ℝ) :
  (∀ (b : ℝ), ∃ (x y : ℝ), x^2 + y^2 - x - y = 0 ∧ y = m (x - m + b)) →
  (0 ≤ m ∧ m ≤ 1) := by
sorry

end intersection_exists_implies_m_range_l663_663175


namespace logarithm_inequality_l663_663554

noncomputable def log_base (a b : ℝ) : ℝ := log b / log a

theorem logarithm_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ 1) (h4 : b ≠ 1) (h5 : log_base a b > 1) : 
  (b - 1) * (b - a) > 0 :=
by
  sorry

end logarithm_inequality_l663_663554


namespace election_votes_l663_663986

theorem election_votes (total_votes : ℕ) (invalid_percentage : ℝ) (votes_percentage_A : ℝ) :
  total_votes = 560000 → invalid_percentage = 0.15 → votes_percentage_A = 0.75 →
  let valid_votes := (1 - invalid_percentage) * total_votes in
  let votes_for_A := votes_percentage_A * valid_votes in
  votes_for_A = 357000 := 
by 
  intros h_total_votes h_invalid_percentage h_votes_percentage_A;
  have h_valid_votes : valid_votes = 0.85 * 560000 := by 
    rw [h_total_votes, h_invalid_percentage];
    norm_num;
  have h_votes_for_A : votes_for_A = 0.75 * 476000 := by 
    rw [h_valid_votes, h_votes_percentage_A];
    norm_num;
  rw h_votes_for_A;
  norm_num;
  sorry

end election_votes_l663_663986


namespace parabola_circle_intersection_distance_sum_l663_663287

theorem parabola_circle_intersection_distance_sum :
  let focus := (0, 1/4),
      p1 := (15, 225),
      p2 := (4, 16),
      p3 := (-3, 9),
      p4 := (-16, 256),
      dist := λ (p q : ℝ × ℝ), (sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) in
  dist focus p1 + dist focus p2 + dist focus p3 + dist focus p4 ≈ 705.8567 := 
sorry

end parabola_circle_intersection_distance_sum_l663_663287


namespace problem1_problem2_l663_663625

-- Assume x and y are positive numbers
variables (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

-- Prove that x^3 + y^3 >= x^2*y + y^2*x
theorem problem1 : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
by sorry

-- Prove that m ≤ 2 given the additional condition
variables (m : ℝ)
theorem problem2 (cond : (x/y^2 + y/x^2) ≥ m/2 * (1/x + 1/y)) : m ≤ 2 :=
by sorry

end problem1_problem2_l663_663625


namespace geometric_locus_of_C_l663_663712

open EuclideanGeometry

variables {S : Sphere} (A B : S)

def constant_angle (γ : ℝ) (C : S) : Prop :=
  let ⟨α, β⟩ := (angle A C B, angle B C A) in
  α + β - (angle A B C) = γ

theorem geometric_locus_of_C (γ : ℝ) :
  ∃ C₁ C₂ : S, constant_angle γ C₁ ∧ constant_angle γ C₂ ∧
  symmetric_with_respect_to_plane (plane A B) C₁ C₂ :=
sorry

end geometric_locus_of_C_l663_663712


namespace unique_function_identity_l663_663129

theorem unique_function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f(x + 1) = f(x) + 1) 
  (h2 : ∀ x : ℝ, f(x^2) = f(x)^2) : 
  ∀ x : ℝ, f(x) = x :=
by
  sorry

end unique_function_identity_l663_663129


namespace red_balls_l663_663352

theorem red_balls (w r : ℕ) (h1 : w = 12) (h2 : w * 3 = r * 4) : r = 9 :=
sorry

end red_balls_l663_663352


namespace calculate_liquids_l663_663013

def water_ratio := 60 -- mL of water for every 400 mL of flour
def milk_ratio := 80 -- mL of milk for every 400 mL of flour
def flour_ratio := 400 -- mL of flour in one portion

def flour_quantity := 1200 -- mL of flour available

def number_of_portions := flour_quantity / flour_ratio

def total_water := number_of_portions * water_ratio
def total_milk := number_of_portions * milk_ratio

theorem calculate_liquids :
  total_water = 180 ∧ total_milk = 240 :=
by
  -- Proof will be filled in here. Skipping with sorry for now.
  sorry

end calculate_liquids_l663_663013


namespace chord_length_l663_663810

theorem chord_length {r : ℝ} (h : r = 10) 
    (bisector : ∀ O A M C D : ℝ, O = 0 ∧ A = 10 ∧ M = 5 ∧ ∃ D, ∃ C, CD bisects OA perpendicularly) : 
    let CD := 2 * (sorry : ℝ) in 
    CD = 10 * Real.sqrt 3 := sorry

end chord_length_l663_663810


namespace most_likely_units_digit_is_zero_l663_663518

def picks : list ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def J (n : ℕ) : Prop := n ∈ picks
def K (n : ℕ) : Prop := n ∈ picks

def units_digit (n : ℕ) : ℕ := n % 10

theorem most_likely_units_digit_is_zero :
  ∃ (n : ℕ), (∀ (m : ℕ), (units_digit (n + m) = 0) → (J n) ∧ (K m)) ∧
    (∀ (d : ℕ),
      (∀ (m : ℕ), (units_digit (n + m) = d) → (J n) ∧ (K m)) →
      d ≠ 0 →
      frequency (units_digit (n + m) = 0) > frequency (units_digit (n + m) = d)) :=
sorry

end most_likely_units_digit_is_zero_l663_663518


namespace area_triangle_LOM_l663_663665

theorem area_triangle_LOM
  (ABC : Triangle)
  (α β γ : ℝ)
  (L O M : Point)
  (A B C : Point)
  (h1 : α = β - γ)
  (h2 : β = 2 * γ)
  (h3 : α + β + γ = 180)
  (h4 : ABC.area = 32)
  (h5 : angle_bisector A ∩ circumcircle ABC = L)
  (h6 : angle_bisector B ∩ circumcircle ABC = O)
  (h7 : angle_bisector C ∩ circumcircle ABC = M) :
  area Triangle.mk L O M = 44 :=
sorry

end area_triangle_LOM_l663_663665


namespace total_gems_in_chest_l663_663443

theorem total_gems_in_chest (diamonds rubies : ℕ) 
  (h_diamonds : diamonds = 45)
  (h_rubies : rubies = 5110) : 
  diamonds + rubies = 5155 := 
by 
  sorry

end total_gems_in_chest_l663_663443


namespace back_wheel_revolutions_is_900_l663_663713

/-- Define the radii of the wheels -/
def front_wheel_radius : ℝ := 3
def back_wheel_radius : ℝ := 0.5

/-- Define the number of revolutions of the front wheel -/
def front_wheel_revolutions : ℝ := 150

/-- Calculate the circumference of the wheels -/
def front_wheel_circumference := 2 * Real.pi * front_wheel_radius
def back_wheel_circumference := 2 * Real.pi * back_wheel_radius

/-- Calculate the distance traveled by the front wheel -/
def distance_front_wheel := front_wheel_revolutions * front_wheel_circumference

/-- The number of revolutions the back wheel makes -/
def back_wheel_revolutions := distance_front_wheel / back_wheel_circumference

/-- The theorem stating that the back wheel makes 900 revolutions -/
theorem back_wheel_revolutions_is_900 :
  back_wheel_revolutions = 900 :=
by
  sorry

end back_wheel_revolutions_is_900_l663_663713


namespace candy_difference_l663_663327

theorem candy_difference 
  (total_candies : ℕ)
  (strawberry_candies : ℕ)
  (total_eq : total_candies = 821)
  (strawberry_eq : strawberry_candies = 267) : 
  (total_candies - strawberry_candies - strawberry_candies = 287) :=
by
  sorry

end candy_difference_l663_663327


namespace set_of_a_l663_663649

theorem set_of_a (a : ℝ) :
  (∃ x : ℝ, a * x ^ 2 + a * x + 1 = 0) → -- Set A contains elements
  (a ≠ 0 ∧ a ^ 2 - 4 * a = 0) →           -- Conditions a ≠ 0 and Δ = 0
  a = 4 := 
sorry

end set_of_a_l663_663649


namespace not_proper_subset_of_itself_l663_663031

-- Define the sets
def A := {1}
def B := {2, 3}
def C := (∅ : Set ℕ)
def D := {1, 2, 3}
def U := {1, 2, 3}

-- Proof statement
theorem not_proper_subset_of_itself : ¬ D ⊂ U :=
by {
  -- Since D is equal to U, D is not a proper subset of U
  sorry
}

end not_proper_subset_of_itself_l663_663031


namespace smallest_positive_angle_l663_663026

theorem smallest_positive_angle : 
  (∃ x : ℝ, 0 < x ∧ 2^(Real.sin x)^2 * 4^(Real.cos x)^2 * 2^(Real.tan x) = 8 ∧ x = Real.pi / 3) :=
by
  sorry

end smallest_positive_angle_l663_663026


namespace proof_problem_l663_663178

variables {a b : ℝ^3} (theta : ℝ)
variables (angle_ab : ℝ) (magnitude_a : ℝ) (magnitude_b : ℝ)
variables (dot_ab : ℝ ) (magnitude_a_minus_b : ℝ) (theta_a_minus_b_b : ℝ)

def conditions :=
  angle_ab = 30 ∧ 
  magnitude_a = sqrt 3 ∧ 
  magnitude_b = 2

theorem proof_problem :
  conditions angle_ab magnitude_a magnitude_b →
  dot_ab = 3 ∧
  magnitude_a_minus_b = 1 ∧
  theta_a_minus_b_b = 120 :=
begin
  -- proof goes here
  sorry,
end

end proof_problem_l663_663178


namespace circle_intersections_l663_663804

/-- 
Given 4n points arranged alternately around a circle, colored in an alternating yellow and blue pattern,
such that points of the same color are connected by line segments and at most two segments meet at any point inside the circle,
there are at least n points of intersection between yellow and blue segments.
-/
theorem circle_intersections (n : ℕ) 
    (alternating_points : (fin 4n) → Prop)
    (alternate_coloring : ∀ i, alternating_points i ↔ (i % 2 = 0))
    (pair_yellow : ∀ (i : fin n), alternating_points (2*i)) 
    (pair_blue : ∀ (i : fin n), ¬alternating_points (2*i+1))
    (at_most_two_segments_meet : ∀ (i j : fin (4*n)), i ≠ j → (pair_yellow i ∧ pair_blue j → intersect_segments i j))
    : ∃ k (intersect_YB : (pair_yellow k ∧ pair_blue k)), k ≥ n := 
by 
  sorry

end circle_intersections_l663_663804


namespace toothpicks_structure_l663_663424

theorem toothpicks_structure (n m : ℕ) (hn : n = 100) (hm : m = 99) : 
  let T1 := (n * (n + 1)) / 2,
      T2 := (m * (m + 1)) / 2,
      T := T1 + T2,
      P := 3 * T,
      B := 3 * (n + m + 1),
      P_adjusted := (P - B * 2) / 2 + B * 2
  in P_adjusted = 15596 :=
by
  have hT1 : T1 = 5050 := by sorry
  have hT2 : T2 = 4950 := by sorry
  have hT : T = 10000 := by sorry
  have hP : P = 30000 := by sorry
  have hB : B = 298 := by sorry
  have hPadjusted : P_adjusted = 15596 := by sorry
  sorry

end toothpicks_structure_l663_663424


namespace percentage_of_goals_by_two_players_l663_663377

-- Definitions from conditions
def total_goals_league := 300
def goals_per_player := 30
def number_of_players := 2

-- Mathematically equivalent proof problem
theorem percentage_of_goals_by_two_players :
  let combined_goals := number_of_players * goals_per_player
  let percentage := (combined_goals / total_goals_league : ℝ) * 100 
  percentage = 20 :=
by
  sorry

end percentage_of_goals_by_two_players_l663_663377


namespace line_does_not_pass_through_fourth_quadrant_l663_663252

-- Definitions based on conditions
def line (x : ℝ) : ℝ := 2 * x + 1
def slope : ℝ := 2
def y_intercept : ℝ := 1

-- Statement of the problem
theorem line_does_not_pass_through_fourth_quadrant : 
  ∀ x y : ℝ, y = line x → ¬(x > 0 ∧ y < 0) :=
by
  sorry

end line_does_not_pass_through_fourth_quadrant_l663_663252


namespace f_zero_f_odd_f_decreasing_f_inequality_solution_l663_663509

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (x y : ℝ) : f(x + y) = f(x) + f(y)
axiom f_pos (x : ℝ) (hx : 0 < x) : f(x) < 0
axiom f_one : f(1) = -2

theorem f_zero : f(0) = 0 := sorry

theorem f_odd (x : ℝ) : f(-x) = -f(x) := sorry

theorem f_decreasing (x₁ x₂ : ℝ) (h : x₁ < x₂) : f(x₁) > f(x₂) := sorry

theorem f_inequality_solution (x : ℝ) : -1 ≤ x ∧ x ≤ 4 ↔ f(x^2 - 2*x) - f(x) ≥ -8 := sorry

end f_zero_f_odd_f_decreasing_f_inequality_solution_l663_663509


namespace triangle_AC_length_l663_663676

/-- In a triangle ABC, given AB = 7, BC = 10, and the median AM = 5,
prove that the length of side AC is sqrt(51). -/
theorem triangle_AC_length
  (A B C M : Point)
  (hAB : dist A B = 7)
  (hBC : dist B C = 10)
  (hAM : dist A M = 5)
  (hMedian : is_median A B C M) :
  dist A C = sqrt 51 := 
sorry

end triangle_AC_length_l663_663676


namespace smallest_possible_range_l663_663064

noncomputable def smallest_range (xs : List ℝ) : ℝ :=
  xs.maximum' - xs.minimum'

theorem smallest_possible_range :
  ∀ (x1 x2 x3 x4 x5 : ℝ),
    (x1 + x2 + x3 + x4 + x5) / 5 = 15 →
    (x3 = 18) →
    x1 ≤ x2 →
    x2 ≤ x3 →
    x3 ≤ x4 →
    x4 ≤ x5 →
  smallest_range [x1, x2, x3, x4, x5] = 8.5 :=
by
  intros x1 x2 x3 x4 x5 h_mean h_median h1 h2 h3 h4
  sorry

end smallest_possible_range_l663_663064


namespace max_knights_between_knights_l663_663455

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l663_663455


namespace sequence_sum_first_10_l663_663624

noncomputable def sequence (n : ℕ) : ℕ 
| 0       => 5
| (n + 1) => 5

theorem sequence_sum_first_10 : (∑ k in Finset.range 10, sequence k) = 50 := 
by
  let v n := (sequence (n + 1) - sequence n / 2, sequence (n + 1)^2 / (2 * sequence n))
  let mu := (3, 3)
  have parallel : ∀ n, v n = mu := by sorry
  have sequence_n : ∀ n, sequence n = 5 := by sorry
  calc 
    (∑ k in Finset.range 10, sequence k) 
    = ∑ k in Finset.range 10, 5      : by simp [sequence_n]
    ... = 10 * 5                     : by simp
    ... = 50                         : by simp

end sequence_sum_first_10_l663_663624


namespace area_triangle_QDA_l663_663869

-- Define the points
def Q : ℝ × ℝ := (0, 15)
def A (q : ℝ) : ℝ × ℝ := (q, 15)
def D (p : ℝ) : ℝ × ℝ := (0, p)

-- Define the conditions
variable (q : ℝ) (p : ℝ)
variable (hq : q > 0) (hp : p < 15)

-- Theorem stating the area of the triangle QDA in terms of q and p
theorem area_triangle_QDA : 
  1 / 2 * q * (15 - p) = 1 / 2 * q * (15 - p) :=
by sorry

end area_triangle_QDA_l663_663869


namespace red_balls_count_l663_663044

theorem red_balls_count (y : ℕ) (p_yellow : ℚ) (h1 : y = 10)
  (h2 : p_yellow = 5/8) (total_balls_le : ∀ r : ℕ, y + r ≤ 32) :
  ∃ r : ℕ, 10 + r > 0 ∧ p_yellow = 10 / (10 + r) ∧ r = 6 :=
by
  sorry

end red_balls_count_l663_663044


namespace fraction_income_spent_on_rent_l663_663428

theorem fraction_income_spent_on_rent
  (hourly_wage : ℕ)
  (work_hours_per_week : ℕ)
  (weeks_in_month : ℕ)
  (food_expense : ℕ)
  (tax_expense : ℕ)
  (remaining_income : ℕ) :
  hourly_wage = 30 →
  work_hours_per_week = 48 →
  weeks_in_month = 4 →
  food_expense = 500 →
  tax_expense = 1000 →
  remaining_income = 2340 →
  ((hourly_wage * work_hours_per_week * weeks_in_month - remaining_income - (food_expense + tax_expense)) / (hourly_wage * work_hours_per_week * weeks_in_month) = 1/3) :=
by
  intros h_wage h_hours h_weeks h_food h_taxes h_remaining
  sorry

end fraction_income_spent_on_rent_l663_663428


namespace geom_mean_between_2_and_8_l663_663886

theorem geom_mean_between_2_and_8 (b : ℝ) (h : b^2 = 16) : b = 4 ∨ b = -4 :=
by
  sorry

end geom_mean_between_2_and_8_l663_663886


namespace find_x_l663_663636

theorem find_x : 
  (∑ n in Finset.range 2980, n * (2980 - n)) = 2979 * 1490 * 993 :=
by
  sorry

end find_x_l663_663636


namespace unique_x_condition_l663_663531

theorem unique_x_condition (x : ℝ) : 
  (1 ≤ x ∧ x < 2) ∧ (∀ n : ℕ, 0 < n → (⌊2^n * x⌋ % 4 = 1 ∨ ⌊2^n * x⌋ % 4 = 2)) ↔ x = 4/3 := 
by 
  sorry

end unique_x_condition_l663_663531


namespace max_of_function_l663_663136

-- Using broader imports can potentially bring in necessary libraries, so let's use Mathlib

noncomputable def max_value_in_domain : ℝ :=
  sup ((λ (p : ℝ × ℝ), p.1 * p.2 / (p.1^2 + p.2^2)) '' { x | 1/4 ≤ x.1 ∧ x.1 ≤ 3/5 ∧ 2/7 ≤ x.2 ∧ x.2 ≤ 1/2 } : set ℝ)

theorem max_of_function :
  max_value_in_domain = 2 / 5 :=
sorry

end max_of_function_l663_663136


namespace probability_of_drawing_jingyuetan_ticket_l663_663371

-- Definitions from the problem
def num_jingyuetan_tickets : ℕ := 3
def num_changying_tickets : ℕ := 2
def total_tickets : ℕ := num_jingyuetan_tickets + num_changying_tickets
def num_envelopes : ℕ := total_tickets

-- Probability calculation
def probability_jingyuetan : ℚ := (num_jingyuetan_tickets : ℚ) / (num_envelopes : ℚ)

-- Theorem statement
theorem probability_of_drawing_jingyuetan_ticket : probability_jingyuetan = 3 / 5 :=
by
  sorry

end probability_of_drawing_jingyuetan_ticket_l663_663371


namespace part_a_part_b_l663_663898

def sum_digits (n : ℕ) : ℕ :=
  (2 + 0 + 0 + 5) * n -- sum of digits of "2005" repeated n times

def a (n : ℕ) : ℕ :=
  (sum_digits n) % 10 -- last digit of the sum

theorem part_a (n : ℕ) : a n = 0 ↔ n % 10 = 0 :=
by
  sorry

theorem part_b : (Finset.range 2005).sum (λ k, a (k + 1)) = 9025 :=
by
  sorry

end part_a_part_b_l663_663898


namespace brad_books_this_month_l663_663790

-- Define the number of books William read last month
def william_books_last_month : ℕ := 6

-- Define the number of books Brad read last month
def brad_books_last_month : ℕ := 3 * william_books_last_month

-- Define the number of books Brad read this month as a variable
variable (B : ℕ)

-- Define the total number of books William read over the two months
def total_william_books (B : ℕ) : ℕ := william_books_last_month + 2 * B

-- Define the total number of books Brad read over the two months
def total_brad_books (B : ℕ) : ℕ := brad_books_last_month + B

-- State the condition that William read 4 more books than Brad
def william_read_more_books_condition (B : ℕ) : Prop := total_william_books B = total_brad_books B + 4

-- State the theorem to be proven
theorem brad_books_this_month (B : ℕ) : william_read_more_books_condition B → B = 16 :=
by
  sorry

end brad_books_this_month_l663_663790


namespace probability_closer_to_B_l663_663069

noncomputable def triangle_ABC := (A : (ℝ × ℝ) := (0, 0), B : (ℝ × ℝ) := (12, 0), C : (ℝ × ℝ) := (8, 10))

def inside_triangle (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  ∃ (λ₁ λ₂ λ₃ : ℝ), λ₁ ≥ 0 ∧ λ₂ ≥ 0 ∧ λ₃ ≥ 0 ∧ λ₁ + λ₂ + λ₃ = 1 ∧
  (x = λ₁ * 0 + λ₂ * 12 + λ₃ * 8) ∧ (y = λ₁ * 0 + λ₂ * 0 + λ₃ * 10)

def distance (P Q : ℝ × ℝ) :=
  let (x1, y1) := P
  let (x2, y2) := Q
  (x1 - x2)^2 + (y1 - y2)^2

def closer_to_B (P : ℝ × ℝ) :=
  distance P (12, 0) < distance P (8, 10) ∧ distance P (12, 0) < distance P (0, 0)

theorem probability_closer_to_B :
  let total_area := (12 * 10) / 2
  let closer_area := 109 / 5
  let probability := closer_area / total_area
  probability = 109 / 300
:= sorry

end probability_closer_to_B_l663_663069


namespace greatest_integer_2pi_minus_5_l663_663893

theorem greatest_integer_2pi_minus_5 : 
  let x := 2 * Real.pi - 5
  in ⌊x⌋ = 1 := 
by 
  have h1 : Real.pi ≈ 3.14 := sorry
  have h2 : 2 * Real.pi ≈ 6.28 := sorry
  have h3 : 1 < 2 * Real.pi - 5 := sorry
  have h4 : 2 * Real.pi - 5 < 2 := sorry
  show ⌊2 * Real.pi - 5⌋ = 1 from sorry

end greatest_integer_2pi_minus_5_l663_663893


namespace Manoj_borrowed_years_l663_663301

def principal_Anwar : ℝ := 3900
def rate_Anwar : ℝ := 6 / 100
def principal_Ramu : ℝ := 5655
def rate_Ramu : ℝ := 9 / 100
def gain : ℝ := 824.85

theorem Manoj_borrowed_years :
  ∃ t : ℝ, 
  principal_Anwar * rate_Anwar * t + gain = principal_Ramu * rate_Ramu * t := 
begin
  use 3,
  sorry
end

end Manoj_borrowed_years_l663_663301


namespace ratio_of_r_l663_663401

theorem ratio_of_r
  (total : ℕ) (r_amount : ℕ) (pq_amount : ℕ)
  (h_total : total = 7000 )
  (h_r_amount : r_amount = 2800 )
  (h_pq_amount : pq_amount = total - r_amount) :
  (r_amount / Nat.gcd r_amount pq_amount, pq_amount / Nat.gcd r_amount pq_amount) = (2, 3) :=
by
  sorry

end ratio_of_r_l663_663401


namespace ratio_of_areas_of_triangles_l663_663769

noncomputable def area_ratio_triangle_GHI_JKL
  (a_GHI b_GHI c_GHI : ℕ) (a_JKL b_JKL c_JKL : ℕ) 
  (alt_ratio_GHI : ℕ × ℕ) (alt_ratio_JKL : ℕ × ℕ) : ℚ :=
  let area_GHI := (a_GHI * b_GHI) / 2
  let area_JKL := (a_JKL * b_JKL) / 2
  area_GHI / area_JKL

theorem ratio_of_areas_of_triangles :
  let GHI_sides := (7, 24, 25)
  let JKL_sides := (9, 40, 41)
  area_ratio_triangle_GHI_JKL 7 24 25 9 40 41 (2, 3) (4, 5) = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l663_663769


namespace eagles_total_games_l663_663495

theorem eagles_total_games (y x : ℕ) (hyp1 : x = 0.4 * y) (hyp2 : x + 9 = 0.55 * (y + 10)) : y + 10 = 33 :=
by
  sorry

end eagles_total_games_l663_663495


namespace sum_prime_factors_2310_l663_663782

def prime_factors (n : ℕ) : List ℕ := [2, 3, 5, 7, 11]

def sum_greater_than_five (factors : List ℕ) (threshold : ℕ) : ℕ :=
  factors.filter (λ p, p > threshold) |>.sum

theorem sum_prime_factors_2310 :
  sum_greater_than_five (prime_factors 2310) 5 = 18 := 
by
  -- The list of prime factors of 2310 is [2, 3, 5, 7, 11]
  -- We need to consider only those greater than 5
  -- They are 7 and 11
  -- The sum of 7 and 11 is 18
  sorry

end sum_prime_factors_2310_l663_663782


namespace Joey_age_digit_sum_l663_663276

structure Ages :=
  (joey_age : ℕ)
  (chloe_age : ℕ)
  (zoe_age : ℕ)

def is_multiple (a b : ℕ) : Prop :=
  ∃ k, a = k * b

def sum_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem Joey_age_digit_sum
  (C J Z : ℕ)
  (h1 : J = C + 1)
  (h2 : Z = 1)
  (h3 : ∃ n, C + n = (n + 1) * m)
  (m : ℕ) (hm : m = 9)
  (h4 : C - 1 = 36) :
  sum_digits (J + 37) = 12 :=
by
  sorry

end Joey_age_digit_sum_l663_663276


namespace value_of_a_l663_663588

noncomputable def point_P (m n : ℝ) : ℝ × ℝ := (m / 5, n / 4)

theorem value_of_a (m n a : ℝ) (hmn : m > 0) (hnn : n > 0) (h : m + n = 1) 
  (hP : point_P m n ∈ (λ x : ℝ, (x, x ^ a)) '' set.univ) : a = 1 / 2 :=
by sorry

end value_of_a_l663_663588


namespace finite_solutions_l663_663314

noncomputable def f : ℕ → ℝ := sorry -- Placeholder for the actual function definition

axiom fx_pos (x : ℕ) : f(x) > 0
axiom fx_limit_zero : ∀ ε > 0, ∃ N : ℕ, ∀ x ≥ N, |f(x)| < ε

theorem finite_solutions (m n p : ℕ) :
  f(m) + f(n) + f(p) = 1 → (∃ S : set (ℕ × ℕ × ℕ), S.finite ∧ ∀ (m n p : ℕ), (m, n, p) ∈ S) :=
by
  sorry

end finite_solutions_l663_663314


namespace find_square_of_radius_l663_663046

-- Definitions of lengths as constants
constants (E F G H R S : ℝ)
constants (ER RF GS SH : ℝ)

-- Assign the given lengths to their respective constants
def ER := 25
def RF := 31
def GS := 41
def SH := 29

-- Definition of radius r
constant r : ℝ

-- Hypothesis converting the problem setup into formal conditions
hypothesis h1 : tangent_to_circle E F R -- placeholder for the actual tangent condition
hypothesis h2 : tangent_to_circle G H S -- placeholder for the actual tangent condition
hypothesis h3 : segment_length E R = ER
hypothesis h4 : segment_length R F = RF
hypothesis h5 : segment_length G S = GS
hypothesis h6 : segment_length S H = SH

-- The mathematically equivalent proof problem in Lean
theorem find_square_of_radius : (∃ (r : ℝ), r^2 = 803) :=
sorry

end find_square_of_radius_l663_663046


namespace correct_option_is_d_l663_663390

def isCorrectOption : Prop :=
  ∅ ⊆ Set_of (λ x : ℤ, True)

theorem correct_option_is_d : isCorrectOption :=
by
  sorry

end correct_option_is_d_l663_663390


namespace max_knights_between_other_knights_l663_663476

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l663_663476


namespace total_string_length_l663_663682

-- Define the conditions
def table_top_area : ℝ := 176 -- table top area in square inches
def pi_approx : ℝ := 22 / 7 -- approximate value of π
def extra_length : ℝ := 3 -- additional length of string in inches

-- Prove that the total length of string Jack needs to buy is equal to the expected value
theorem total_string_length :
  let r := Real.sqrt(56) in
  let circumference := 2 * pi_approx * r in
  let total_string := circumference + extra_length in
  total_string = (88 / 7) * Real.sqrt(14) + 3 := 
by
  sorry

end total_string_length_l663_663682


namespace pizza_cost_increase_l663_663833

theorem pizza_cost_increase
  (r1 r2 : ℝ) 
  (A1 A2 : ℝ)
  (h1 : r1 = 5)
  (h2 : r2 = 3)
  (h3 : A1 = π * r1 ^ 2)
  (h4 : A2 = π * r2 ^ 2)
  (h5 : ∆A = A1 - A2)
  (h6 : PercentageIncreaseInArea = ∆A / A2 * 100)
  (TotalPercentageIncreaseInCost : ℝ)
  (h7 : TotalPercentageIncreaseInCost = PercentageIncreaseInArea * 1.20):
  TotalPercentageIncreaseInCost ≈ 213 :=
by
  sorry

end pizza_cost_increase_l663_663833


namespace max_houses_mezhdugrad_l663_663251

def a_k (k : ℕ) : ℕ :=
  match k with
  | 0 => 0
  | ℕ.succ k' => 1 + 2 * a_k k'

theorem max_houses_mezhdugrad : a_k 9 = 511 :=
  by sorry

end max_houses_mezhdugrad_l663_663251


namespace min_attempts_sufficient_a_l663_663399

theorem min_attempts_sufficient_a (n : ℕ) (h : n > 2)
  (good_batteries bad_batteries : ℕ)
  (h1 : good_batteries = n + 1)
  (h2 : bad_batteries = n)
  (total_batteries := 2 * n + 1) :
  (∃ attempts, attempts = n + 1) := sorry

end min_attempts_sufficient_a_l663_663399


namespace quadrilateral_AD_length_l663_663995

noncomputable def length_of_AD (A B C D : EuclideanGeometry.Point) 
  (h_AB : A.distance B = 4) 
  (h_BC : B.distance C = 7) 
  (h_CD : C.distance D = 13) 
  (h_B_right : B.angle 90) 
  (h_C_60 : C.angle 60) : ℝ :=
  sqrt(130)

theorem quadrilateral_AD_length (A B C D : EuclideanGeometry.Point) 
  (h_AB : A.distance B = 4) 
  (h_BC : B.distance C = 7) 
  (h_CD : C.distance D = 13) 
  (h_B_right : B.angle 90) 
  (h_C_60 : C.angle 60) : 
  A.distance D = length_of_AD A B C D h_AB h_BC h_CD h_B_right h_C_60 := 
sorry

end quadrilateral_AD_length_l663_663995


namespace remove_edges_preserve_connectedness_l663_663002

noncomputable theory

open Mathlib.GraphTheory.SimpleGraph

variables {V : Type} [Fintype V]

def chromatic_number (G : SimpleGraph V) : ℕ := G.chromaticNumber

def remove_edges (G : SimpleGraph V) (n: ℕ) : Prop :=
∃ (F : Finset (Sym2 V)), F.card = n ∧ G.deleteEdges F

theorem remove_edges_preserve_connectedness (G : SimpleGraph V) (n : ℕ) 
  (h_conn : G.Connected) (h_chi : chromatic_number G > n) (h_finite : Fintype.card V < ∞) : 
  remove_edges G (n * (n - 1) / 2) ∧ G.Connected :=
sorry

end remove_edges_preserve_connectedness_l663_663002


namespace books_sold_on_thursday_l663_663279

def initial_stock : ℕ := 900
def sold_monday : ℕ := 75
def sold_tuesday : ℕ := 50
def sold_wednesday : ℕ := 64
def sold_friday : ℕ := 135
def unsold_percentage : ℝ := 55.333333333333336 / 100

-- Define the number of unsold books
def unsold_books : ℕ := (initial_stock * (unsold_percentage * 100) / 100).toInt

-- Sum of books sold from Monday to Wednesday and on Friday
def sold_mon_wed_fri : ℕ := sold_monday + sold_tuesday + sold_wednesday + sold_friday

-- Define the proof statement
theorem books_sold_on_thursday : (initial_stock - (sold_mon_wed_fri + unsold_books) = 78) :=
by {
  -- Details of the calculations done here assuming the steps would go into the proof
  sorry
}

end books_sold_on_thursday_l663_663279


namespace expected_value_neg_ξ_l663_663980

noncomputable def ξ : ℕ → ℚ := λ x, if x = 5 then (1 / 4) else 0

theorem expected_value_neg_ξ :
  E(-ξ) = - (5 * (1 / 4)) := by
  -- placeholder for the proof
  sorry

end expected_value_neg_ξ_l663_663980


namespace count_valid_pairs_l663_663111

theorem count_valid_pairs : 
  {n : ℕ | ∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 14}.card = 5 := 
by sorry

end count_valid_pairs_l663_663111


namespace odd_sum_probability_l663_663272

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def prob_A_odd := 1 / 3
def prob_A_even := 2 / 3

def prob_B_odd := 3 / 5
def prob_B_even := 2 / 5

def prob_C_odd_given_B_odd := 2 / 3
def prob_C_even_given_B_odd := 1 / 3

def prob_C_even_given_B_even := 1

theorem odd_sum_probability : 
  let prob_B_C_both_odd := prob_B_odd * prob_C_odd_given_B_odd in
  let prob_B_C_both_even := prob_B_even * prob_C_even_given_B_even in
  prob_A_odd * (prob_B_C_both_odd + prob_B_C_both_even) = 4 / 15 := 
by
  let prob_B_C_both_odd := prob_B_odd * prob_C_odd_given_B_odd
  let prob_B_C_both_even := prob_B_even * prob_C_even_given_B_even
  exact sorry

end odd_sum_probability_l663_663272


namespace total_precious_stones_is_305_l663_663962

theorem total_precious_stones_is_305 :
  let agate := 25
  let olivine := agate + 5
  let sapphire := 2 * olivine
  let diamond := olivine + 11
  let amethyst := sapphire + diamond
  let ruby := diamond + 7
  agate + olivine + sapphire + diamond + amethyst + ruby = 305 :=
by
  sorry

end total_precious_stones_is_305_l663_663962


namespace geometric_sequence_fifth_term_proof_l663_663820

-- Define the properties of the geometric sequence based on the provided conditions
def first_term : ℕ := 4
def fourth_term_eq : ℕ := 324
def common_ratio := classical.some (⟨3, by rfl⟩)

-- Formalizing the problem statement in Lean
theorem geometric_sequence_fifth_term_proof 
  (a : ℕ) 
  (h1 : a = first_term) 
  (r : ℕ) 
  (h2 : ∃ x : ℕ, x^3 = (fourth_term_eq / first_term) ∧ (a * x^3 = fourth_term_eq)) : 
  a * r^4 = 324 := 
sorry

end geometric_sequence_fifth_term_proof_l663_663820


namespace find_b_days_l663_663793

theorem find_b_days 
  (a_days b_days c_days : ℕ)
  (a_wage b_wage c_wage : ℕ)
  (total_earnings : ℕ)
  (ratio_3_4_5 : a_wage * 5 = b_wage * 4 ∧ b_wage * 5 = c_wage * 4 ∧ a_wage * 5 = c_wage * 3)
  (c_wage_val : c_wage = 110)
  (a_days_val : a_days = 6)
  (c_days_val : c_days = 4) 
  (total_earnings_val : total_earnings = 1628)
  (earnings_eq : a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings) :
  b_days = 9 := by
  sorry

end find_b_days_l663_663793


namespace min_x_y_l663_663595

theorem min_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 2) :
  x + y ≥ 9 / 2 := 
by 
  sorry

end min_x_y_l663_663595


namespace distance_between_foci_of_ellipse_l663_663094

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end distance_between_foci_of_ellipse_l663_663094


namespace sum_of_consecutive_integers_of_sqrt3_l663_663209

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l663_663209


namespace sqrt_neg_ge_a_plus_sqrt_neg_two_l663_663312

theorem sqrt_neg_ge_a_plus_sqrt_neg_two (a : ℝ) (ha : a > 0) : -real.sqrt a ≥ a + real.sqrt (-2) :=
sorry

end sqrt_neg_ge_a_plus_sqrt_neg_two_l663_663312


namespace ellipse_eq_line_condition_l663_663921

noncomputable def ellipse_standard_eq (a b : ℝ) : Prop :=
  (a > b) ∧ (a > 0) ∧ (b > 0) ∧ ( ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 )

theorem ellipse_eq :
  let c := 1 in let a := 2 in let b := sqrt (a^2 - c^2)
  in a = 2 ∧ b = sqrt 3 ∧ ellipse_standard_eq a b := 
by
  sorry

noncomputable def line_intersects_ellipse (k m : ℝ) :=
  ∃ (x y : ℝ), (y = k * x + m) ∧ ((x^2 / 4) + (y^2 / 3) = 1)

theorem line_condition :
  ∀ (k m : ℝ), line_intersects_ellipse k m →
  abs (abs (2 * (k * x2 + m) * x2) = 12) ↔ m^2 ≥ 12 / 7 ∧ (abs m ≥ 2 * (sqrt 21 / 7)) :=
by
  sorry

end ellipse_eq_line_condition_l663_663921


namespace president_vice_secretary_choice_l663_663990

theorem president_vice_secretary_choice (n : ℕ) (h : n = 6) :
  (∀ a b c : fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (n * (n - 1) * (n - 2) = 120) := 
sorry

end president_vice_secretary_choice_l663_663990


namespace fuel_consumption_rate_l663_663837

theorem fuel_consumption_rate (fuel_left time_left r: ℝ) 
    (h_fuel: fuel_left = 6.3333) 
    (h_time: time_left = 0.6667) 
    (h_rate: r = fuel_left / time_left) : r = 9.5 := 
by
    sorry

end fuel_consumption_rate_l663_663837


namespace pencil_cost_proof_l663_663302

-- Define the cost of the pencil
def pencil_cost : ℝ := 8

-- Define the cost of the pen in terms of the pencil
def pen_cost (P : ℝ) : ℝ := P / 2

-- Define the total cost condition
def total_cost_condition (P : ℝ) : Prop := P + pen_cost P = 12

-- Prove that the pencil cost is indeed $8 given the conditions
theorem pencil_cost_proof : total_cost_condition pencil_cost :=
by
  unfold total_cost_condition
  unfold pen_cost
  rw [pencil_cost]
  norm_num
  sorry

end pencil_cost_proof_l663_663302


namespace find_c_l663_663646

theorem find_c (c : ℝ) (h : ∃ (f : ℝ → ℝ), (f = λ x => c * x^3 + 23 * x^2 - 5 * c * x + 55) ∧ f (-5) = 0) : c = 6.3 := 
by {
  sorry
}

end find_c_l663_663646


namespace base_digits_equality_l663_663410

theorem base_digits_equality (b : ℕ) (h_condition : b^5 ≤ 200 ∧ 200 < b^6) : b = 2 := 
by {
  sorry -- proof not required as per the instructions
}

end base_digits_equality_l663_663410


namespace volume_of_revolved_triangle_is_correct_l663_663669

noncomputable section

-- Define the properties of the right-angled triangle
structure RightAngledTriangle where
  AB BC AC : ℝ
  is_right_angled : AB^2 + BC^2 = AC^2

-- Define the given right-angled triangle
def triangleABC : RightAngledTriangle :=
{ AB := 3, BC := 4, AC := 5,
  is_right_angled := by linarith }

-- Define the volume of the solid generated by revolving the triangle around AB
def volumeOfRevolvedTriangle (t : RightAngledTriangle) : ℝ :=
  (1 / 3) * Real.pi * (t.BC ^ 2) * t.AB

-- Theorem statement
theorem volume_of_revolved_triangle_is_correct :
  volumeOfRevolvedTriangle triangleABC = 16 * Real.pi :=
by sorry

end volume_of_revolved_triangle_is_correct_l663_663669


namespace ellipse_foci_distance_l663_663088

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end ellipse_foci_distance_l663_663088


namespace initial_population_first_village_equals_l663_663070

-- Definitions of the conditions
def initial_population_second_village : ℕ := 42000
def decrease_first_village_per_year : ℕ := 1200
def increase_second_village_per_year : ℕ := 800
def years : ℕ := 13

-- Proposition we want to prove
/-- The initial population of the first village such that both villages have the same population after 13 years. -/
theorem initial_population_first_village_equals :
  ∃ (P : ℕ), (P - decrease_first_village_per_year * years) = (initial_population_second_village + increase_second_village_per_year * years) 
  := sorry

end initial_population_first_village_equals_l663_663070


namespace max_knights_seated_next_to_two_knights_l663_663491

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l663_663491


namespace assembly_line_average_output_l663_663081

theorem assembly_line_average_output :
  (60 / 90) + (60 / 60) = (5 / 3) →
  60 + 60 = 120 →
  120 / (5 / 3) = 72 :=
by
  intros h1 h2
  -- Proof follows, but we will end with 'sorry' to indicate further proof steps need to be done.
  sorry

end assembly_line_average_output_l663_663081


namespace lucas_candy_last_monday_l663_663526

theorem lucas_candy_last_monday 
    (S : ℕ) 
    (H1 : ∀ n, Lucas makes 4 pieces of chocolate candy for each student on Monday)
    (H2 : 3 of Lucas' students will not be coming to class this upcoming Monday)
    (H3 : Lucas will make 28 pieces of chocolate candy this upcoming Monday):
    4 * S = 40 :=
by
  have H4 : 4 * (S - 3) = 28 from sorry
  have H5 : 4 * S - 12 = 28 from sorry
  have H6 : 4 * S = 40 from sorry
  sorry

end lucas_candy_last_monday_l663_663526


namespace range_of_m_value_of_m_l663_663190

-- Definitions and conditions
def quadratic_function (m x : ℝ) : ℝ := (m + 6) * x^2 + 2 * (m - 1) * x + (m + 1)

def always_has_root (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_function m x = 0

def distinct_roots (m : ℝ) : Prop :=
  let Δ := 4 * (m - 1)^2 - 4 * (m + 6) * (m + 1) in
  Δ > 0

def reciprocal_sum (m : ℝ) : Prop :=
  let x1 := (-2 * (m - 1) + real.sqrt (4 * (m - 1)^2 - 4 * (m + 6) * (m + 1))) / (2 * (m + 6))
  let x2 := (-2 * (m - 1) - real.sqrt (4 * (m - 1)^2 - 4 * (m + 6) * (m + 1))) / (2 * (m + 6))
  (1 / x1 + 1 / x2) = -4
  
-- Theorem that matches the conditions to results
theorem range_of_m :
  ∀ (m : ℝ), always_has_root m ↔ m ≤ -5 / 9 := sorry

theorem value_of_m :
  ∀ (m : ℝ), distinct_roots m ∧ reciprocal_sum m → m = -3 := sorry

end range_of_m_value_of_m_l663_663190


namespace distance_from_origin_to_point_on_parabola_l663_663180

theorem distance_from_origin_to_point_on_parabola
  (y x : ℝ)
  (focus : ℝ × ℝ := (4, 0))
  (on_parabola : y^2 = 8 * x)
  (distance_to_focus : Real.sqrt ((x - 4)^2 + y^2) = 4) :
  Real.sqrt (x^2 + y^2) = 2 * Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_point_on_parabola_l663_663180


namespace number_of_valid_pairs_l663_663546

open Finset

-- Definition of the finite set
def finiteSet := {1, 2, 3, 4, 5}

-- Definition of the condition on subsets A and B
def valid_pair (A B : Finset ℕ) : Prop :=
  |A| * |B| = |A ∩ B| * |A ∪ B|

-- Main statement to prove
theorem number_of_valid_pairs : 
  let pairs := { (A, B) | A ∈ powerset finiteSet ∧ B ∈ powerset finiteSet ∧ valid_pair A B } in
  ∑ (A, B) in pairs, 1 = 454 :=
by 
  sorry

end number_of_valid_pairs_l663_663546


namespace domain_of_f_l663_663537

open Set

def f (x : ℝ) : ℝ := (x^4 + 3 * x^2 - 8) / ((x - 2) * |x + 3|)

theorem domain_of_f : {x : ℝ | x ≠ 2 ∧ x ≠ -3} = (-∞, -3) ∪ (-3, 2) ∪ (2, ∞) :=
by
  sorry

end domain_of_f_l663_663537


namespace triangle_area_l663_663656

noncomputable def area_triangle (A B C : ℝ) (b c : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem triangle_area
  (A B C : ℝ) (b : ℝ) 
  (hA : A = π / 4)
  (h0 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) :
  ∃ c : ℝ, area_triangle A B C b c = 2 :=
by
  sorry

end triangle_area_l663_663656


namespace complex_power_expr_l663_663127

theorem complex_power_expr (i : ℂ) (h : i * i = -1) :
  (⟨(1 + i) / real.sqrt 2⟩ : ℂ) ^ 100 = -1 :=
sorry

end complex_power_expr_l663_663127


namespace range_of_a_l663_663042

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - log x - a ≥ 0) ∧
  (∃ x : ℝ, x^2 + 2 * a * x - 8 - 6 * a = 0) →
  a ∈ Set.Iic (-4) ∪ Set.Icc (-2) 1 :=
by
  sorry

end range_of_a_l663_663042


namespace range_of_m_l663_663909

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0)
  (h_equation : (2 / x) + (1 / y) = 1 / 3)
  (h_inequality : x + 2 * y > m^2 - 2 * m) : 
  -4 < m ∧ m < 6 := 
sorry

end range_of_m_l663_663909


namespace arrangement_count_l663_663767

-- Define the conditions
def num_male_students : ℕ := 3
def num_female_students : ℕ := 2
def num_teachers : ℕ := 1

-- The condition that no males or females are adjacent
def non_adjacent (l : List ℕ) : Prop := ∀ (i : ℕ), i < l.length - 1 → l[i] ≠ l[i+1]

-- Define the list representing the arrangement: 1 for male, 2 for female, 3 for teacher
def arrange_students (seq : List ℕ) : Prop :=
  seq.count 1 = num_male_students ∧
  seq.count 2 = num_female_students ∧
  seq.count 3 = num_teachers ∧
  non_adjacent seq

-- The theorem to prove
theorem arrangement_count : ∃ seq : List ℕ, arrange_students seq ∧ seq.count = 120 := sorry

end arrangement_count_l663_663767


namespace solve_system_of_eq_l663_663727

noncomputable def system_of_eq (x y z : ℝ) : Prop :=
  y = x^3 * (3 - 2 * x) ∧
  z = y^3 * (3 - 2 * y) ∧
  x = z^3 * (3 - 2 * z)

theorem solve_system_of_eq (x y z : ℝ) :
  system_of_eq x y z →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end solve_system_of_eq_l663_663727


namespace max_knights_seated_next_to_two_knights_l663_663494

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l663_663494


namespace constant_term_expansion_sum_of_coefficients_expansion_l663_663734

noncomputable def general_term (n r : ℕ) (x : ℂ) : ℂ := 
  (nat.choose n r) * (2 : ℂ)^(n-r) * (- (1 : ℂ) / x)^r * x^(n-2*r)

theorem constant_term_expansion : 
  general_term 4 2 1 = 24 :=
by
  sorry

theorem sum_of_coefficients_expansion : 
  (2 : ℂ - 1)^4 = 1 :=
by
  sorry

end constant_term_expansion_sum_of_coefficients_expansion_l663_663734


namespace distance_between_foci_of_ellipse_l663_663083

theorem distance_between_foci_of_ellipse :
  ∀ (ellipse: ℝ × ℝ → Prop),
    (∀ x y, ellipse (x, y) ↔ (x - 5)^2 / 25 + (y - 2)^2 / 4 = 1) →
    ∃ c : ℝ, 2 * c = 2 * Real.sqrt (25 - 4) :=
by
  intro ellipse h
  use Real.sqrt (25 - 4)
  sorry

end distance_between_foci_of_ellipse_l663_663083


namespace find_lending_rate_l663_663062
-- Import necessary libraries

-- Define the borrowed amount, borrowing rate, time period, and gain per year as constants
def borrowed_amount : ℝ := 4000
def borrowing_rate : ℝ := 4
def time_period : ℕ := 2
def gain_per_year : ℝ := 80

-- Define the question as calculating the lending rate
def lending_rate (earnings lending_rate: ℝ) : Prop :=
  (borrowed_amount * lending_rate * (time_period:ℝ) / 100 = earnings)

-- Define the intermediate calculations
def interest_paid_for_borrowing : ℝ :=
  borrowed_amount * borrowing_rate * (time_period:ℝ) / 100

def total_gain : ℝ :=
  gain_per_year * (time_period:ℝ)

def total_earnings_from_lending : ℝ :=
  interest_paid_for_borrowing + total_gain

-- The theorem states that the lending rate is 6% given the defined conditions.
theorem find_lending_rate : lending_rate total_earnings_from_lending 6 :=
  by
    sorry

end find_lending_rate_l663_663062


namespace find_pointB_l663_663998

variables (xA yA xB yB : ℝ) (d : ℝ)
def pointA := (2, 1 : ℝ)
def lineAB_parallel_x_axis := yB = yA
def AB_length := abs (xB - xA) = d

theorem find_pointB (h1 : pointA = (2, 1)) (h2 : lineAB_parallel_x_axis) (h3 : AB_length := 4) :
  (xB = 6 ∧ yB = 1) ∨ (xB = -2 ∧ yB = 1) :=
sorry

end find_pointB_l663_663998


namespace vishal_investment_more_than_trishul_l663_663020

theorem vishal_investment_more_than_trishul :
  ∀ (V T R : ℝ), R = 2000 → T = R - 0.10 * R → V + T + R = 5780 → (V - T) / T * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end vishal_investment_more_than_trishul_l663_663020


namespace negation_universal_prop_l663_663348

theorem negation_universal_prop :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_universal_prop_l663_663348


namespace sum_of_factors_of_144_eq_403_l663_663781

theorem sum_of_factors_of_144_eq_403 : (∑ d in (Finset.filter (λ d, 144 % d = 0) (Finset.range 145)), d) = 403 := 
sorry

end sum_of_factors_of_144_eq_403_l663_663781


namespace slowest_pump_time_l663_663372

noncomputable theory

-- Define the rates and times
def rates_ratio : ℚ := 2 / 9 -- since 2+3+4 = 9 and we focus on the slowest rate (2)
def time_together : ℚ := 6 

theorem slowest_pump_time :
  (time_together / rates_ratio) = 27 := 
by
  sorry

end slowest_pump_time_l663_663372


namespace lifespan_represents_sample_l663_663012

-- Definitions
def survey_population := 2500
def provinces_and_cities := 11

-- Theorem stating that the lifespan of the urban residents surveyed represents a sample
theorem lifespan_represents_sample
  (number_of_residents : ℕ) (num_provinces : ℕ) 
  (h₁ : number_of_residents = survey_population)
  (h₂ : num_provinces = provinces_and_cities) :
  "Sample" = "Sample" :=
by 
  -- Proof skipped
  sorry

end lifespan_represents_sample_l663_663012


namespace largest_sampled_number_l663_663550

theorem largest_sampled_number (N : ℕ) (a₁ a₂ : ℕ) (k : ℕ) (H_N : N = 1500)
  (H_a₁ : a₁ = 18) (H_a₂ : a₂ = 68) (H_k : k = a₂ - a₁) :
  ∃ m, m ≤ N ∧ (m % k = 18 % k) ∧ ∀ n, (n % k = 18 % k) → n ≤ N → n ≤ m :=
by {
  -- sorry
  sorry
}

end largest_sampled_number_l663_663550


namespace quadratic_function_expression_quadratic_function_range_l663_663563

variable {α : Type*} [LinearOrderedField α]

/-- Given a quadratic function f(x) that satisfies the conditions:
  - f(0) = 1,
  - f(x+1) - f(x) = 2x,
  prove that the explicit expression for f(x) is x^2 - x + 1.
 -/
theorem quadratic_function_expression (f : α → α)
  (h1 : f 0 = 1)
  (h2 : ∀ x, f (x + 1) - f x = 2 * x) :
  ∀ x, f x = x^2 - x + 1 :=
by
  sorry

/-- Given the function f(x) = x^2 - x + 1, 
  prove that the range of f(x) when x ∈ [-1, 1] is [3/4, 3].
 -/
theorem quadratic_function_range :
  ∀ x ∈ (set.Icc (-1 : α) 1), 
  (3 / 4 : α) ≤ (x^2 - x + 1) ∧ (x^2 - x + 1) ≤ 3 :=
by
  sorry

end quadratic_function_expression_quadratic_function_range_l663_663563


namespace division_of_floats_l663_663780

theorem division_of_floats : 4.036 / 0.04 = 100.9 :=
by
  sorry

end division_of_floats_l663_663780


namespace maintain_constant_chromosomes_l663_663361

-- Definitions
def meiosis_reduces_chromosomes (original_chromosomes : ℕ) : ℕ := original_chromosomes / 2

def fertilization_restores_chromosomes (half_chromosomes : ℕ) : ℕ := half_chromosomes * 2

-- The proof problem
theorem maintain_constant_chromosomes (original_chromosomes : ℕ) (somatic_chromosomes : ℕ) :
  meiosis_reduces_chromosomes original_chromosomes = somatic_chromosomes / 2 ∧
  fertilization_restores_chromosomes (meiosis_reduces_chromosomes original_chromosomes) = somatic_chromosomes :=
sorry

end maintain_constant_chromosomes_l663_663361


namespace sum_of_first_15_terms_l663_663579

-- Define the arithmetic sequence
def arith_seq (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arith_seq (a₁ d : ℝ) (n : ℕ) := (n / 2) * (2 * a₁ + (n - 1) * d)

-- Given conditions
axiom sum_condition {a₁ d : ℝ} : arith_seq a₁ d 7 + arith_seq a₁ d 9 = 10

-- Goal
theorem sum_of_first_15_terms (a₁ d : ℝ) (h : arith_seq a₁ d 7 + arith_seq a₁ d 9 = 10) :
  sum_arith_seq a₁ d 15 = 75 :=
by {
  -- Proof goes here
  sorry
}

end sum_of_first_15_terms_l663_663579


namespace smallest_element_in_T_l663_663692

def is_valid_T (T : Finset ℕ) : Prop :=
  T.card = 8 ∧ (∀ a b ∈ T, a < b → ¬(b % a = 0 ∨ b - 3 = a))

theorem smallest_element_in_T :
  ∃ (T : Finset ℕ), (∀ x ∈ T, x ∈ (Finset.range 20).filter (λ n, n ≠ 0)) ∧ is_valid_T T ∧ T.min' ⟨4, by decide⟩ = 4 :=
by
  sorry

end smallest_element_in_T_l663_663692


namespace general_term_sqrt2_constant_term_l663_663564

noncomputable def frac_part (x : ℝ) : ℝ :=
  x - x.floor

def seq_a (a : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0     => 0  -- invalid case since sequence is defined for natural numbers starting from 1
  | (n+1) => if n = 0 then frac_part a else
             if (seq_a a n) = 0 then 0 else frac_part (1 / (seq_a a n))

/-- General term for a = sqrt 2 in the sequence is sqrt(2) - 1 --/
theorem general_term_sqrt2 : 
  (∀ n : ℕ, seq_a (√2) n = √2 - 1) :=
sorry

/-- If a > 1/2, constant term a_n = a implies a = (sqrt(5) - 1) / 2 --/
theorem constant_term (a : ℝ) (h1 : a > 1/2) (h2 : ∀ n : ℕ, 1 ≤ n → seq_a a n = a) :
  a = (√5 - 1) / 2 :=
sorry

end general_term_sqrt2_constant_term_l663_663564


namespace tan_theta_minus_pi_over_4_l663_663174

theorem tan_theta_minus_pi_over_4 (θ : ℝ) (h1 : θ ∈ Icc (3 * π / 2) (2 * π))
  (h2 : Real.sin (θ + π / 4) = 5 / 13) : Real.tan (θ - π / 4) = -12 / 5 := 
by
  sorry

end tan_theta_minus_pi_over_4_l663_663174


namespace min_third_side_of_right_triangle_l663_663773

theorem min_third_side_of_right_triangle (a b : ℕ) (h1 : a = 4) (h2 : b = 5) :
  ∃ c : ℕ, (min c (4 + 5 - 3) - (4 - 3)) = 3 :=
sorry

end min_third_side_of_right_triangle_l663_663773


namespace knights_max_seated_between_knights_l663_663466

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l663_663466


namespace g_of_7_l663_663819

def g (x : ℝ) := (x^2 - 10 * x + 37) / 4

theorem g_of_7 : g 7 = 4 := by
  -- Here we can simply refer the proof by steps logically since we are only constructing the statement now.
  sorry

end g_of_7_l663_663819


namespace perpendicular_line_slope_l663_663971

theorem perpendicular_line_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x - 2 * y + 5 = 0 → x = 2 * y - 5)
  (h2 : ∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = - (2 / m) * x + 6 / m)
  (h3 : (1 / 2 : ℝ) * - (2 / m) = -1) : m = 1 :=
sorry

end perpendicular_line_slope_l663_663971


namespace part1_part2_l663_663611

-- Definitions
def f (x a : ℝ) : ℝ := (sqrt x - log x - a) / x
def g (x : ℝ) : ℝ := sqrt x - log x

-- Problem statements as Lean theorems
theorem part1 (h : ∀ x : ℝ, x > 0 → (f x a) ≤ 0) : a ≤ 3 - 4 * log 2 :=
sorry

theorem part2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : (1 / (2 * sqrt x1)) - (1 / x1) = (1 / (2 * sqrt x2)) - (1 / x2)) :
    x1 * x2 > 256 ∧ g x1 + g x2 > 8 - 8 * log 2 :=
sorry

end part1_part2_l663_663611


namespace arithmetic_sequence_a3_a5_arithmetic_sequence_S7_l663_663920

/-- An arithmetic sequence a_n with first term a_1 = 1 and seventh term a_7 = 19. -/
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d

/-- Given conditions for the arithmetic sequence -/
def a₁ : ℝ := 1
def a₇ : ℝ := 19

/-- Define the sum of the first n terms for the sequence -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 0 + a (n - 1))) / 2

/-- Prove that for the arithmetic sequence, a_3 + a_5 = 20 given a_1 = 1 and a_7 = 19 -/
theorem arithmetic_sequence_a3_a5 (a : ℕ → ℝ) (h_a : is_arithmetic_sequence a) (h1 : a 0 = a₁) (h7 : a 6 = a₇) : 
  a 2 + a 4 = 20 :=
sorry

/-- Prove that the sum of the first 7 terms in the arithmetic sequence is 70 -/
theorem arithmetic_sequence_S7 (a : ℕ → ℝ) (h_a : is_arithmetic_sequence a) (h1 : a 0 = a₁) (h7 : a 6 = a₇) : 
  S 7 a = 70 :=
sorry

end arithmetic_sequence_a3_a5_arithmetic_sequence_S7_l663_663920


namespace log_expression_calculation_l663_663501

theorem log_expression_calculation :
  2 * log 5 10 + log 5 0.25 = 2 := by
  -- assume properties of logarithms and definition of 0.25
  sorry

end log_expression_calculation_l663_663501


namespace max_knights_seated_next_to_two_knights_l663_663489

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l663_663489


namespace cost_per_day_additional_weeks_l663_663732

theorem cost_per_day_additional_weeks (cost_first_week_per_day : ℝ) (days_first_week : ℕ) (total_days : ℕ) (total_cost : ℝ) : 
  ∃ (cost_additional_per_day : ℝ), 
  cost_first_week_per_day = 18 ∧ days_first_week = 7 ∧ total_days = 23 ∧ total_cost = 334 ∧
  (let cost_first_week := days_first_week * cost_first_week_per_day,
       additional_days := total_days - days_first_week,
       additional_cost := total_cost - cost_first_week
   in cost_additional_per_day = additional_cost / additional_days ∧ cost_additional_per_day = 13) :=
begin
  sorry
end

end cost_per_day_additional_weeks_l663_663732


namespace regular_octahedron_vertices_count_l663_663631

def regular_octahedron_faces := 8
def regular_octahedron_edges := 12
def regular_octahedron_faces_shape := "equilateral triangle"
def regular_octahedron_vertices_meet := 4

theorem regular_octahedron_vertices_count :
  ∀ (F E V : ℕ),
    F = regular_octahedron_faces →
    E = regular_octahedron_edges →
    (∀ (v : ℕ), v = regular_octahedron_vertices_meet) →
    V = 6 :=
by
  intros F E V hF hE hV
  sorry

end regular_octahedron_vertices_count_l663_663631


namespace solve_for_m_l663_663965

theorem solve_for_m (m : ℝ) (z : ℂ) (h : z = (m^2 - 2*m - 15) * complex.I) (hz_real : z.im = 0) : m = 5 ∨ m = -3 :=
  sorry

end solve_for_m_l663_663965


namespace sale_in_third_month_l663_663055

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (average_sales : ℕ)
  (h1 : sale1 = 5420)
  (h2 : sale2 = 5660)
  (h4 : sale4 = 6350)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 6470)
  (h_avg : average_sales = 6100) : 
  ∃ sale3, sale1 + sale2 + sale3 + sale4 + sale5 + sale6 = average_sales * 6 ∧ sale3 = 6200 :=
by
  sorry

end sale_in_third_month_l663_663055


namespace conic_sections_proof_l663_663145

noncomputable def ellipse_equation_λ : Prop :=
  ∃ λ, λ = 3 ∧ (λ = 3 ∧ ∀ x y, (x, y) = (-3, 2) → (x^2 / (9 + λ) + y^2 / (4 + λ) = 1) → (x^2 / 12 + y^2 / 7 = 1))

noncomputable def parabola_equation_x_axis : Prop :=
  ∃ p, p = - (1 / Real.sqrt 2) ∧ (p = - (1 / Real.sqrt 2) ∧ ∀ x y, (x, y) = (-4, -4 * Real.sqrt 2) → (x^2 = 4 * p * y) → (x^2 = -2 * Real.sqrt 2 * y))

noncomputable def parabola_equation_y_axis : Prop :=
  ∃ p', p' = -2 ∧ (p' = -2 ∧ ∀ x y, (x, y) = (-4, -4 * Real.sqrt 2) → (y^2 = 4 * p' * x) → (y^2 = -8 * x))

theorem conic_sections_proof : ellipse_equation_λ ∧ parabola_equation_x_axis ∧ parabola_equation_y_axis :=
  by
  split
  · unfold ellipse_equation_λ
    sorry
  split
  · unfold parabola_equation_x_axis
    sorry
  · unfold parabola_equation_y_axis
    sorry

end conic_sections_proof_l663_663145


namespace slope_intercept_form_l663_663505

-- Definitions of vectors and the line equation
def line_eq (x y : ℝ) : Prop :=
  let v1 := (-3, 4 : ℝ × ℝ)
  let v2 := (x - 2, y + 6 : ℝ × ℝ)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem statement that given the line equation, the equation in slope-intercept form is y = (3/4)x - 7.5
theorem slope_intercept_form (x y : ℝ) :
  line_eq x y → y = (3 / 4) * x - 7.5 :=
by
  intros h
  sorry

end slope_intercept_form_l663_663505


namespace f_3_add_f_10_l663_663598

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom f_1 : f 1 = 4

theorem f_3_add_f_10 : f 3 + f 10 = 4 :=
by sorry

end f_3_add_f_10_l663_663598


namespace gcd_polynomial_997_l663_663169

theorem gcd_polynomial_997 (b : ℤ) (h : ∃ k : ℤ, b = 997 * k ∧ k % 2 = 1) :
  Int.gcd (3 * b ^ 2 + 17 * b + 31) (b + 7) = 1 := by
  sorry

end gcd_polynomial_997_l663_663169


namespace largest_rhombus_diagonal_in_circle_l663_663969

theorem largest_rhombus_diagonal_in_circle (r : ℝ) (h : r = 10) : (2 * r = 20) :=
by
  sorry

end largest_rhombus_diagonal_in_circle_l663_663969


namespace arithmetic_contains_geometric_l663_663311

theorem arithmetic_contains_geometric (a d : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) : 
  ∃ b q : ℕ, (b = a) ∧ (q = 1 + d) ∧ (∀ n : ℕ, ∃ k : ℕ, a * (1 + d)^n = a + k * d) :=
by
  sorry

end arithmetic_contains_geometric_l663_663311


namespace sin_alpha_plus_beta_l663_663930

variable (α β : ℝ)
variable (h1 : α ∈ (0, π/2))
variable (h2 : β ∈ (0, π/2))
variable (h_cos_α : Real.cos α = 12 / 13)
variable (h_cos_2α_β : Real.cos (2 * α + β) = 3 / 5)

theorem sin_alpha_plus_beta :
  Real.sin (α + β) = 33 / 65 :=
by
  sorry

end sin_alpha_plus_beta_l663_663930


namespace min_length_AB_eq_4sqrt2_l663_663167

theorem min_length_AB_eq_4sqrt2 :
  (∀ (E F : ℝ × ℝ), 
   (x - 1)^2 + (y - 2)^2 = 4 → 
   (∀ P : ℝ × ℝ, P = ((E.1 + F.1) / 2, (E.2 + F.2) / 2) →
   (x - 1)^2 + (y - 2)^2 = 2 →
   (∀ (A B : ℝ × ℝ), 
   (line: x - y - 3 = 0) →
    (angle APB ≥ π/2) →
    (distance_from_center_to_line (1, 2) (x - y - 3 = 0) = 2√2) →
    ∃ |AB| = 4√2))) :=
begin
  sorry
end

end min_length_AB_eq_4sqrt2_l663_663167


namespace incorrect_statement_B_l663_663183

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x - 1)^3 - a * x - b + 2

-- Condition for statement B
axiom eqn_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1

-- The theorem to prove:
theorem incorrect_statement_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1 := by
  exact eqn_B a b

end incorrect_statement_B_l663_663183


namespace probability_at_least_one_correct_l663_663705

open Classical

theorem probability_at_least_one_correct :
  let prob_miss := 5 / 6,
      prob_miss_all := (prob_miss ^ 5 : ℚ),
      prob_at_least_one := 1 - prob_miss_all
  in prob_at_least_one = 4651 / 7776 := by
  let prob_miss := 5 / 6 : ℚ
  let prob_miss_all := (prob_miss ^ 5 : ℚ)
  let prob_at_least_one := 1 - prob_miss_all
  show prob_at_least_one = 4651 / 7776
  sorry

end probability_at_least_one_correct_l663_663705


namespace triangle_area_l663_663534

theorem triangle_area : 
  let L1 (x y : ℝ) := y - 2 * x + 3 = 0
  let L2 (x y : ℝ) := 2 * y + x - 10 = 0
  let intercept_y1 := -3
  let intercept_y2 := 5
  let base := intercept_y2 - intercept_y1
  let intersection_x := (16:ℝ) / 5
  let height := intersection_x
  in (1 / 2) * base * height = 64 / 5 :=
by 
  let L1 (x y : ℝ) := y - 2 * x + 3 = 0
  let L2 (x y : ℝ) := 2 * y + x - 10 = 0
  let intercept_y1 := -3
  let intercept_y2 := 5
  let base := intercept_y2 - intercept_y1
  let intersection_x := (16:ℝ) / 5
  let height := intersection_x
  show (1 / 2) * base * height = 64 / 5
  sorry

end triangle_area_l663_663534


namespace intersection_with_complement_range_of_a_l663_663620

noncomputable theory

open Set

-- Definitions from the problem
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | x * (3 - x) > 0}
def M (a : ℝ) : Set ℝ := {x | 2 * x - a < 0}

-- Question (1)
theorem intersection_with_complement : A ∩ (compl B) = {x : ℝ | -1 < x ∧ x ≤ 0} :=
sorry

-- Question (2)
theorem range_of_a {a : ℝ} : (A ∪ B) ⊆ M a → 6 ≤ a :=
sorry

end intersection_with_complement_range_of_a_l663_663620


namespace consecutive_sum_l663_663211

theorem consecutive_sum (a b : ℤ) (h1 : a + 1 = b) (h2 : (a : ℝ) < real.sqrt 3) (h3 : real.sqrt 3 < (b : ℝ)) : a + b = 3 := 
sorry

end consecutive_sum_l663_663211


namespace distance_A_moves_l663_663434

-- Define the initial conditions and parameters of the rectangular sheet.
def width := 1  -- Width of the sheet in cm
def length := 12  -- Length of the sheet in cm
def mid_length := length / 2  -- Midpoint of the length in cm

-- Calculate the distance A moves using Pythagorean theorem.
def distance_move (a : ℝ) (b : ℝ) := Real.sqrt (a^2 + b^2)

theorem distance_A_moves :
  distance_move width mid_length = Real.sqrt 37 :=
sorry    -- Proof omitted

end distance_A_moves_l663_663434


namespace max_knights_between_knights_l663_663470

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l663_663470


namespace shape_of_phi_eq_c_is_cone_l663_663148

-- Given conditions
def spherical_coordinates (r θ φ : ℝ) : Type :=
  ⟨r, θ, φ⟩

variable (c : ℝ)

def phi_eq_c (r θ φ : ℝ) : Prop :=
  φ = c

-- The problem statement
theorem shape_of_phi_eq_c_is_cone (r θ φ : ℝ) (h : phi_eq_c c r θ φ) : 
  (∃ k : ℝ, r = k * sin c ∧ θ = k * cos c) :=
sorry

end shape_of_phi_eq_c_is_cone_l663_663148


namespace each_cinema_has_empty_showtime_l663_663722

/-- 
There are seven students attending 7 different cinemas, 
each with eight different showtimes in a day. 
Each showtime sees 6 students together, 1 goes alone to any other cinema. 
By the end of the day, each student visits every cinema exactly once.
-/
def seven_students_visiting_seven_cinemas : Prop := 
  ∀ (showtimes: Fin 8),
  ∃ (no_students_time : Fin 8), 
  ∀ (students: Fin 7),
  ∀ (cinemas: Fin 7),
  (students ∈ cinemas → cinema_visits students cinemas = 1) →
  (∃ (no_students_time : Fin 8), 
    ∀ (students: Fin 7),
    not (attendance students no_students_time))

-- Prove that at each cinema, there's at least one showtime without any student
theorem each_cinema_has_empty_showtime 
  (condition : seven_students_visiting_seven_cinemas) :
  ∀ (cinema: Fin 7), 
  ∃ (showtime: Fin 8), 
  (∀ (student: Fin 7), ¬attendance student showtime) :=
sorry

end each_cinema_has_empty_showtime_l663_663722


namespace derivative_of_f_l663_663536

noncomputable def f (x : ℝ) : ℝ :=
  cos (2 * x - 1) + 1 / (x^2)

noncomputable def f' (x : ℝ) : ℝ :=
  -2 * sin (2 * x - 1) - 2 / (x^3)

theorem derivative_of_f (x : ℝ) : HasDerivAt f (f' x) x :=
by
  sorry

end derivative_of_f_l663_663536


namespace matrix_equivalent_l663_663802

def M := λ (i j : Fin 4), 
  if (i, j) = (0, 0) then 0 else
  if (i, j) = (0, 1) then 7 else
  if (i, j) = (0, 2) then 9 else
  if (i, j) = (0, 3) then 14 else
  if (i, j) = (1, 0) then 11 else
  if (i, j) = (1, 1) then 12 else
  if (i, j) = (1, 2) then 2 else
  if (i, j) = (1, 3) then 5 else
  if (i, j) = (2, 0) then 6 else
  if (i, j) = (2, 1) then 1 else
  if (i, j) = (2, 2) then 15 else
  if (i, j) = (2, 3) then 8 else
  if (i, j) = (3, 0) then 13 else
  if (i, j) = (3, 1) then 10 else
  if (i, j) = (3, 2) then 4 else
  if (i, j) = (3, 3) then 3 else
  0

theorem matrix_equivalent : 
  ∃ M : Fin 4 → Fin 4 → ℕ,
  (∀ i k, i ∈ Fin 4 ∧ k ∈ ({5, 6, 7} : Finset ℕ) → M i k = M i (k - 4)) ∧
  (∀ i, ∑ j in Finset.range 4, M (i % 4) ((j + i) % 4) = ∑ j in Finset.range 4, M (i + j) 0) ∧
  (∑ j in Finset.range 4, M 0 j = ∑ j in Finset.range 4, M 0 (j + 1)) ∧
  (∀ (i : Fin 4), M i 0 + M i 1 + M i 2 + M i 3 = 30) ∧
  (∀ (i : Fin 4), M 0 i + M 1 i + M 2 i + M 3 i = 30) ∧
  M 0 0 = 0 ∧ M 0 1 = 7 ∧ M 1 0 = 11 ∧ M 1 2 = 2 ∧ M 2 2 = 15 ∧
  M = (λ i j, if (i, j) = (0, 0) then 0 else
              if (i, j) = (0, 1) then 7 else
              if (i, j) = (0, 2) then 9 else
              if (i, j) = (0, 3) then 14 else
              if (i, j) = (1, 0) then 11 else
              if (i, j) = (1, 1) then 12 else
              if (i, j) = (1, 2) then 2 else
              if (i, j) = (1, 3) then 5 else
              if (i, j) = (2, 0) then 6 else
              if (i, j) = (2, 1) then 1 else
              if (i, j) = (2, 2) then 15 else
              if (i, j) = (2, 3) then 8 else
              if (i, j) = (3, 0) then 13 else
              if (i, j) = (3, 1) then 10 else
              if (i, j) = (3, 2) then 4 else
              if (i, j) = (3, 3) then 3 else
              0) := by
  sorry

end matrix_equivalent_l663_663802


namespace horner_eval_at_minus_point_two_l663_663018

def f (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_eval_at_minus_point_two :
  f (-0.2) = 0.81873 :=
by 
  sorry

end horner_eval_at_minus_point_two_l663_663018


namespace find_angle_D_l663_663982

variables (A B C D F : ℝ)

axiom angle_sum_triangle (a b c : ℝ) : a + b + c = 180
axiom equal_angles (x y : ℝ) : x = y

theorem find_angle_D (h1 : A + B = 180) (h2 : B = 90) (h3 : C = D) (h4 : F = 50) :
  D = 40 :=
by
  have hA : A = 90 :=
    calc
      A + 90 = 180 : by exact h1
      A = 90 : by linarith,
  have hC : 90 + C + 50 = 180 :=
    calc
      A + C + F = 180 : by apply angle_sum_triangle
      90 + C + 50 = 180 : by rw [←hA, ←h4],
  have hC_value : C = 40 := by linarith,
  exact calc
    D = C : by exact h3
    D = 40 : by rw hC_value

end find_angle_D_l663_663982


namespace remainder_of_3_pow_100_mod_7_is_4_l663_663025

theorem remainder_of_3_pow_100_mod_7_is_4
  (h1 : 3^1 ≡ 3 [MOD 7])
  (h2 : 3^2 ≡ 2 [MOD 7])
  (h3 : 3^3 ≡ 6 [MOD 7])
  (h4 : 3^4 ≡ 4 [MOD 7])
  (h5 : 3^5 ≡ 5 [MOD 7])
  (h6 : 3^6 ≡ 1 [MOD 7]) :
  3^100 ≡ 4 [MOD 7] :=
by
  sorry

end remainder_of_3_pow_100_mod_7_is_4_l663_663025


namespace convert_to_rectangular_form_l663_663507

theorem convert_to_rectangular_form :
  2 * Real.sqrt 3 * Complex.exp (13 * Real.pi * Complex.I / 6) = 3 + Complex.I * Real.sqrt 3 :=
by
  sorry

end convert_to_rectangular_form_l663_663507


namespace probability_not_sit_next_to_each_other_l663_663303

noncomputable def total_ways_to_choose_two_chairs_excluding_broken : ℕ := 28

noncomputable def unfavorable_outcomes : ℕ := 6

theorem probability_not_sit_next_to_each_other :
  (1 - (unfavorable_outcomes / total_ways_to_choose_two_chairs_excluding_broken) = (11 / 14)) :=
by sorry

end probability_not_sit_next_to_each_other_l663_663303


namespace jerry_total_cost_l663_663009

-- Definition of the costs and quantities
def cost_color : ℕ := 32
def cost_bw : ℕ := 27
def num_color : ℕ := 3
def num_bw : ℕ := 1

-- Definition of the total cost
def total_cost : ℕ := (cost_color * num_color) + (cost_bw * num_bw)

-- The theorem that needs to be proved
theorem jerry_total_cost : total_cost = 123 :=
by
  sorry

end jerry_total_cost_l663_663009


namespace count_valid_x_l663_663351

theorem count_valid_x :
  (∃ (x : ℕ), x ≤ 60 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 300 → Nat.coprime (7 * x + k) (k + 1)) ↔ 3 :=
by
  sorry

end count_valid_x_l663_663351


namespace fixed_costs_16699_50_l663_663050

noncomputable def fixed_monthly_costs (production_cost shipping_cost units_sold price_per_unit : ℝ) : ℝ :=
  let total_variable_cost := (production_cost + shipping_cost) * units_sold
  let total_revenue := price_per_unit * units_sold
  total_revenue - total_variable_cost

theorem fixed_costs_16699_50 :
  fixed_monthly_costs 80 7 150 198.33 = 16699.5 :=
by
  sorry

end fixed_costs_16699_50_l663_663050


namespace find_a4_b4_c4_l663_663596

-- Given conditions
variables {a b c : ℝ}

def condition1 := (a^2 - b^2) + c^2 = 8
def condition2 := a * b * c = 2

-- Prove the main statement given the conditions
theorem find_a4_b4_c4 (hc1 : condition1) (hc2 : condition2) : a^4 + b^4 + c^4 = 70 := 
by 
  sorry

end find_a4_b4_c4_l663_663596


namespace max_M_l663_663195

noncomputable def conditions (x y z u : ℝ) : Prop :=
  (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) ∧ (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < u) ∧ (z ≥ y)

theorem max_M (x y z u : ℝ) : conditions x y z u → ∃ M : ℝ, M = 6 + 4 * Real.sqrt 2 ∧ M ≤ z / y :=
by {
  sorry
}

end max_M_l663_663195


namespace distinct_digit_sums_l663_663260

theorem distinct_digit_sums (A B C E D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ E ∧ A ≠ D ∧ B ≠ C ∧ B ≠ E ∧ B ≠ D ∧ C ≠ E ∧ C ≠ D ∧ E ≠ D)
 (h_ab : A + B = D) (h_ab_lt_10 : A + B < 10) (h_ce : C + E = D) :
  ∃ (x : ℕ), x = 8 := 
sorry

end distinct_digit_sums_l663_663260


namespace inequality_condition_l663_663617

theorem inequality_condition (a : ℝ) (h : a ∈ set.Ioo (3 - 2 * real.sqrt 2) (3 + 2 * real.sqrt 2)) : 
  ∃ (a1 a2 : ℝ), a1 = 3 - 2 * real.sqrt 2 ∧ a2 = 3 + 2 * real.sqrt 2 ∧ a1 + a2 = 6 :=
by
  use [3 - 2 * real.sqrt 2, 3 + 2 * real.sqrt 2]
  split
  { exact rfl }
  split
  { exact rfl }
  { norm_num }
  sorry

end inequality_condition_l663_663617


namespace finite_decimal_representation_nat_numbers_l663_663785

theorem finite_decimal_representation_nat_numbers (n : ℕ) : 
  (∀ k : ℕ, k < n → (∃ u v : ℕ, (k + 1 = 2^u ∨ k + 1 = 5^v) ∨ (k - 1 = 2^u ∨ k -1  = 5^v))) ↔ 
  (n = 2 ∨ n = 3 ∨ n = 6) :=
by sorry

end finite_decimal_representation_nat_numbers_l663_663785


namespace fabric_sales_fraction_l663_663813

def total_sales := 36
def stationery_sales := 15
def jewelry_sales := total_sales / 4
def fabric_sales := total_sales - jewelry_sales - stationery_sales

theorem fabric_sales_fraction:
  (fabric_sales : ℝ) / total_sales = 1 / 3 :=
by
  sorry

end fabric_sales_fraction_l663_663813


namespace vector_c_equals_combination_l663_663976

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)
def vector_c : ℝ × ℝ := (-2, 4)

theorem vector_c_equals_combination : vector_c = (vector_a.1 - 3 * vector_b.1, vector_a.2 - 3 * vector_b.2) :=
sorry

end vector_c_equals_combination_l663_663976


namespace knights_max_seated_between_knights_l663_663464

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l663_663464


namespace find_x_l663_663291

-- Introducing the main theorem
theorem find_x (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (x : ℝ) (h_x : 0 < x) : 
  let r := (4 * a) ^ (4 * b)
  let y := x ^ 2
  r = a ^ b * y → 
  x = 16 ^ b * a ^ (1.5 * b) :=
by
  sorry

end find_x_l663_663291


namespace pi_over_12_is_15_deg_thirteen_pi_over_6_is_390_deg_minus_five_pi_over_12_is_minus_75_deg_thirty_six_deg_is_pi_over_5_rad_minus_105_deg_is_minus_seven_pi_over_12_rad_l663_663866

-- Lean 4 Statements

-- π equals 180 degrees
def pi_equiv_180_deg : real := real.pi = 180

-- 1. Prove that π/12 equals 15 degrees
theorem pi_over_12_is_15_deg (h : pi_equiv_180_deg) : (real.pi / 12) * (180 / real.pi) = 15 := by
  sorry

-- 2. Prove that 13π/6 equals 390 degrees
theorem thirteen_pi_over_6_is_390_deg (h : pi_equiv_180_deg) : (13 * real.pi / 6) * (180 / real.pi) = 390 := by
  sorry

-- 3. Prove that -5π/12 equals -75 degrees
theorem minus_five_pi_over_12_is_minus_75_deg (h : pi_equiv_180_deg) : (-5 * real.pi / 12) * (180 / real.pi) = -75 := by
  sorry

-- 4. Prove that 36 degrees equals π/5 radians
theorem thirty_six_deg_is_pi_over_5_rad (h : pi_equiv_180_deg) : (36 * (real.pi / 180)) = real.pi / 5 := by
  sorry

-- 5. Prove that -105 degrees equals -7π/12 radians
theorem minus_105_deg_is_minus_seven_pi_over_12_rad (h : pi_equiv_180_deg) : (-105 * (real.pi / 180)) = -7 * real.pi / 12 := by
  sorry

end pi_over_12_is_15_deg_thirteen_pi_over_6_is_390_deg_minus_five_pi_over_12_is_minus_75_deg_thirty_six_deg_is_pi_over_5_rad_minus_105_deg_is_minus_seven_pi_over_12_rad_l663_663866


namespace find_element_in_H2O_with_given_mass_percentage_l663_663539

-- Definitions of molar masses
def molar_mass_H : ℝ := 1.01
def molar_mass_O : ℝ := 16.00
def molar_mass_H2_PER_O : ℕ := 2

-- Total molar mass of H2O
def molar_mass_H2O : ℝ := 2 * molar_mass_H + molar_mass_O

-- Given mass percentage
def given_mass_percentage : ℝ := 88.89

-- Calculation of mass percentage of oxygen
def mass_percentage_O : ℝ := (molar_mass_O / molar_mass_H2O) * 100

-- The proof problem statement
theorem find_element_in_H2O_with_given_mass_percentage : mass_percentage_O ≈ given_mass_percentage → "The element is Oxygen (O)" :=
by
  sorry

end find_element_in_H2O_with_given_mass_percentage_l663_663539


namespace plane_fuel_consumption_rate_l663_663835

/-- A plane has 6.3333 gallons of fuel left and can continue to fly for 0.6667 hours.
    Prove that the rate of fuel consumption per hour is approximately 9.5 gallons per hour. -/
theorem plane_fuel_consumption_rate :
  let fuel_left := 6.3333
  let time_left_to_fly := 0.6667
  let rate_of_fuel_consumption_per_hour := fuel_left / time_left_to_fly
  abs (rate_of_fuel_consumption_per_hour - 9.5) < 0.01 :=
by {
  let fuel_left := 6.3333
  let time_left_to_fly := 0.6667
  let rate_of_fuel_consumption_per_hour := fuel_left / time_left_to_fly
  show abs (rate_of_fuel_consumption_per_hour - 9.5) < 0.01,
  apply sorry
}

end plane_fuel_consumption_rate_l663_663835


namespace find_range_of_k_l663_663940

noncomputable def f (x k : ℝ) : ℝ := |x^2 - 1| + x^2 + k * x

theorem find_range_of_k :
  (∀ x : ℝ, 0 < x → 0 ≤ f x k) → (-1 ≤ k) :=
by
  sorry

end find_range_of_k_l663_663940


namespace circle_through_fixed_point_l663_663943

open Real

noncomputable def hyperbola_eq (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

def A1 : (ℝ × ℝ) := (2, 0)
def A2 : (ℝ × ℝ) := (-2, 0)

theorem circle_through_fixed_point :
  ∀ P : ℝ × ℝ, hyperbola_eq P.1 P.2 → P ≠ A1 → P ≠ A2 → 
  ∀ M1 M2 : ℝ × ℝ, (M1.1 = 1 ∧ ∃ λ : ℝ, M1.2 = λ * (P.2 - 0) / (P.1 - 2 + 2) ∧ (λ + P.1 - 2) = 1) →
             (M2.1 = 1 ∧ ∃ μ : ℝ, M2.2 = μ * (P.2 - 0) / (P.1 + 2 - 2) ∧ (μ + P.1 + 2) = 1) →
  ∃ fixed_point : ℝ × ℝ, fixed_point = (3 / 2, 0) :=
sorry

end circle_through_fixed_point_l663_663943


namespace gifts_and_charitable_causes_amount_l663_663874

-- Define Jill's net monthly salary
def net_monthly_salary : ℝ := 3400

-- Discretionary income is one fifth of her net monthly salary
def discretionary_income : ℝ := net_monthly_salary / 5

-- Define allocations from discretionary income
def vacation_fund : ℝ := 0.30 * discretionary_income
def savings : ℝ := 0.20 * discretionary_income
def eating_out_socializing : ℝ := 0.35 * discretionary_income

-- Calculate the percentage remaining for gifts and charitable causes
def remaining_percentage : ℝ := 1 - (0.30 + 0.20 + 0.35)

-- Calculate the amount for gifts and charitable causes
def gifts_and_charitable_causes : ℝ := remaining_percentage * discretionary_income

-- Prove the amount used for gifts and charitable causes is $102
theorem gifts_and_charitable_causes_amount:
  gifts_and_charitable_causes = 102 := by
  sorry

end gifts_and_charitable_causes_amount_l663_663874


namespace knights_max_seated_between_knights_l663_663463

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l663_663463


namespace postage_cost_l663_663308

theorem postage_cost (w : ℚ) (base_rate additional_rate : ℚ) (h_w : w = 4.5) (h_base_rate : base_rate = 30) (h_additional_rate : additional_rate = 22) :
    let total_cost := base_rate + additional_rate * (w - 1).ceil in
    total_cost = 118 :=
by
  sorry

end postage_cost_l663_663308


namespace triangle_area_l663_663776

def base := 4
def height := 6

theorem triangle_area : (base * height) / 2 = 12 := by
  sorry

end triangle_area_l663_663776


namespace tap_emptying_time_l663_663048

theorem tap_emptying_time
  (F : ℝ := 1 / 3)
  (T_combined : ℝ := 7.5):
  ∃ x : ℝ, x = 5 ∧ (F - (1 / x) = 1 / T_combined) := 
sorry

end tap_emptying_time_l663_663048


namespace find_b_l663_663753

-- Define the polynomial P
def P (x a b : ℝ) : ℝ := x^4 + x^3 - x^2 + a * x + b

-- Define the condition that P is a square of some other polynomial Q
def isSquare (P : ℝ → ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ x, P x = (x^2 + p*x + q)^2

theorem find_b (a b : ℝ) (h : isSquare (P x a b)) : b = 25 / 64 :=
sorry

end find_b_l663_663753


namespace max_squares_8_11_max_squares_8_12_l663_663152

/-- A helper function to define the largest squares that can be cut from given dimensions -/
def max_squares (m n : ℕ) : ℕ :=
  let side_length := min m n
  let remaining_area := (max (m - side_length) 0) * n + m * max (n - side_length) 0
  1 + (remaining_area / (side_length * side_length))

theorem max_squares_8_11 : max_squares 8 11 = 5 :=
  by simp [max_squares]

theorem max_squares_8_12 : max_squares 8 12 = 4 :=
  by simp [max_squares]

end max_squares_8_11_max_squares_8_12_l663_663152


namespace area_of_quadrilateral_BFDE_l663_663566

-- Define the rhombus ABCD with the given diagonals
variables (d1 d2 : ℝ)
variables (A B C D E F : Type)
include d1 d2

-- Define the conditions of the problem
def is_rhombus (A B C D : Type) : Prop :=  -- Add necessary geometric conditions for a rhombus here.
sorry

def diagonals_of_rhombus_are (d1 d2 : ℝ) : Prop := 
d1 = 3 ∧ d2 = 4

def altitudes_drawn (B E F : Type) : Prop := -- Add necessary geometric conditions for altitudes here.
sorry

-- Main theorem to be proved
theorem area_of_quadrilateral_BFDE 
  (ht_rhombus : is_rhombus A B C D)
  (ht_diag : diagonals_of_rhombus_are d1 d2)
  (ht_altitudes : altitudes_drawn B E F) :
  quadrilateral_area B F D E = 4.32 := 
sorry

end area_of_quadrilateral_BFDE_l663_663566


namespace min_value_of_reciprocal_sums_l663_663771

theorem min_value_of_reciprocal_sums (a b : ℝ) (hab : a * b ≠ 0) 
  (h1 : ∃ x y : ℝ, x^2 + y^2 + 2 * a * x + a^2 - 4 = 0) 
  (h2 : ∃ x y : ℝ, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0)
  (h3 : ∃ c1 r1 c2 r2 : ℝ, (c1 + a)^2 + r1^2 = 4 ∧ c2^2 + (r2 - 2 * b)^2 = 1 ∧ sqrt(a^2 + 4*b^2) = 3) :
  (1 / (a^2) + 1 / (b^2)) = 1 :=
by 
  sorry

end min_value_of_reciprocal_sums_l663_663771


namespace sum_of_consecutive_integers_l663_663226

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end sum_of_consecutive_integers_l663_663226


namespace range_of_f_l663_663937

noncomputable def f (x : ℝ) := -3 * Real.cos x + 1

theorem range_of_f : set.range f = set.Icc (-2 : ℝ) (4 : ℝ) :=
  sorry

end range_of_f_l663_663937


namespace distance_walked_on_third_day_l663_663262

theorem distance_walked_on_third_day:
  ∃ x : ℝ, 
    4 * x + 2 * x + x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 378 ∧
    x = 48 := 
by
  sorry

end distance_walked_on_third_day_l663_663262


namespace distance_between_foci_of_ellipse_l663_663086

theorem distance_between_foci_of_ellipse :
  ∀ (ellipse: ℝ × ℝ → Prop),
    (∀ x y, ellipse (x, y) ↔ (x - 5)^2 / 25 + (y - 2)^2 / 4 = 1) →
    ∃ c : ℝ, 2 * c = 2 * Real.sqrt (25 - 4) :=
by
  intro ellipse h
  use Real.sqrt (25 - 4)
  sorry

end distance_between_foci_of_ellipse_l663_663086


namespace find_x_l663_663795

theorem find_x (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end find_x_l663_663795


namespace solve_cubic_root_eqn_l663_663879

theorem solve_cubic_root_eqn:
  ∃ x : ℝ, x = -208 / 27 ∧ (∛(5 - x) = 7 / 3) :=
begin
  sorry
end

end solve_cubic_root_eqn_l663_663879


namespace unique_sum_of_two_three_digit_numbers_l663_663681

theorem unique_sum_of_two_three_digit_numbers:
  ∃ a b: ℕ, (100 ≤ a ∧ a < 1000) ∧ (100 ≤ b ∧ b < 1000) ∧
  (∀ d, d ∈ (Nat.digits 10 a) → d ∉ (Nat.digits 10 b)) ∧
  -- a has tens digit 9
  ((a / 10) % 10 = 9) ∧ 
  -- b has units digit 0
  (b % 10 = 0) ∧ 
  -- c = a + b, c is a three-digit number
  let c := a + b in (100 ≤ c ∧ c < 1000) ∧ 
  -- c's hundreds digit is between 5 and 7
  (5 ≤ (c / 100) ∧ (c / 100) ≤ 7) ∧ 
  -- c's tens digit is odd
  (((c / 10) % 10) % 2 = 1) ∧
  -- c's units digit is 1
  (c % 10 = 1) ∧ 
  -- Prove the valid pairs:
  ((a = 240 ∧ b = 391) ∨ (a = 260 ∧ b = 491) ∨ (a = 460 ∧ b = 291) ∨ (a = 480 ∧ b = 291))
:=
  sorry

end unique_sum_of_two_three_digit_numbers_l663_663681


namespace bottles_of_regular_soda_l663_663056

theorem bottles_of_regular_soda (R : ℕ) : 
  let apples := 36 
  let diet_soda := 54
  let total_bottles := apples + 98 
  R + diet_soda = total_bottles → R = 80 :=
by
  sorry

end bottles_of_regular_soda_l663_663056


namespace exists_line_in_alpha_parallel_to_gamma_l663_663197

variables (α β γ : Plane) (a : Line)

-- Conditions
axiom beta_perp_gamma : ∀ β γ : Plane, β ⊥ γ
axiom alpha_inter_gamma_not_perp : ∀ α γ : Plane, ∃ l : Line, l ∈ α ∧ l ∈ γ ∧ ¬(α ⊥ γ)
axiom a_in_alpha : a ∈ α

-- Proof statement
theorem exists_line_in_alpha_parallel_to_gamma :
  (∃ a : Line, a ∈ α ∧ a ∥ γ) :=
sorry

end exists_line_in_alpha_parallel_to_gamma_l663_663197


namespace ellipse_foci_distance_l663_663087

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end ellipse_foci_distance_l663_663087


namespace shelves_needed_is_five_l663_663441

-- Definitions for the conditions
def initial_bears : Nat := 15
def additional_bears : Nat := 45
def bears_per_shelf : Nat := 12

-- Adding the number of bears received to the initial stock
def total_bears : Nat := initial_bears + additional_bears

-- Calculating the number of shelves used
def shelves_used : Nat := total_bears / bears_per_shelf

-- Statement to prove
theorem shelves_needed_is_five : shelves_used = 5 :=
by
  -- Insert specific step only if necessary, otherwise use sorry
  sorry

end shelves_needed_is_five_l663_663441


namespace trajectory_of_Q_is_circle_l663_663581

theorem trajectory_of_Q_is_circle {F1 F2 P Q : Type*}
    (ellipse : ellipse_with_foci F1 F2 → Prop)
    (on_ellipse : P ∈ ellipse)
    (extension_condition : |F1P| → |PQ| = |PF2|) :
    trajectory Q = circle := 
  sorry

end trajectory_of_Q_is_circle_l663_663581


namespace vector_equation_median_bd_l663_663931

variable (A B C P D : Type) [InnerProductSpace ℝ A]
variables (x y : ℝ)
variables (a b c p d : A)

-- Given conditions
def median_condition (a b c d : A) : Prop :=
  d = (b + c) / 2

def point_on_median (b d p : A) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ p = (1 - t) • b + t • d

def vector_equation (a b c p : A) (x y : ℝ) : Prop :=
  p = x • (b - a) + y • (c - a)

-- The proof problem statement
theorem vector_equation_median_bd (a b c p d : A)
  (h_median : median_condition a b c d)
  (h_point : point_on_median b d p)
  (h_vector_eq : vector_equation a b c p x y) :
  x + 2 * y = 1 ∧ 0 < x ∧ x < 1 :=
by
  sorry

end vector_equation_median_bd_l663_663931


namespace second_camp_students_selected_l663_663115

theorem second_camp_students_selected : 
  let total_students := 100 in 
  let first_camp_range := (1, 15) in 
  let second_camp_range := (16, 55) in 
  let third_camp_range := (56, 100) in 
  let initial_student := 3 in 
  let step := 5 in 
  count_students_in_range(total_students, first_camp_range, second_camp_range, third_camp_range, initial_student, step) = 8 :=
by
  sorry

def count_students_in_range (total_students : ℕ) 
  (first_camp_range second_camp_range third_camp_range: (ℕ × ℕ)) 
  (initial_student step: ℕ) : ℕ := 
  let (second_start, second_end) := second_camp_range in
  let mut count := 0 in
  let mut student_num := initial_student in
  while student_num ≤ total_students do
    if second_start ≤ student_num ∧ student_num ≤ second_end then
      count := count + 1
    student_num := student_num + step
  count

end second_camp_students_selected_l663_663115


namespace cone_diameter_base_l663_663605

theorem cone_diameter_base 
  (r l : ℝ) 
  (h_semicircle : l = 2 * r) 
  (h_surface_area : π * r ^ 2 + π * r * l = 3 * π) 
  : 2 * r = 2 :=
by
  sorry

end cone_diameter_base_l663_663605


namespace initial_HNO3_percentage_is_correct_l663_663809

def initial_percentage_of_HNO3 (P : ℚ) : Prop :=
  let initial_volume := 60
  let added_volume := 18
  let final_volume := 78
  let final_percentage := 50
  (P / 100) * initial_volume + added_volume = (final_percentage / 100) * final_volume

theorem initial_HNO3_percentage_is_correct :
  initial_percentage_of_HNO3 35 :=
by
  sorry

end initial_HNO3_percentage_is_correct_l663_663809


namespace consecutive_even_sum_ways_l663_663668

theorem consecutive_even_sum_ways :
  let n : ℕ := 195
  let sum_even_sequence : ℕ → ℕ → ℕ :=
    λ (n k : ℕ), 2 * (n + (n + 1) + (n + 2) + ... + (n + k - 1))
  ∃ ways : ℕ, ways = 6 ∧ (∀ k : ℕ, ∃ n : ℕ, sum_even_sequence n k = 390) :=
sorry

end consecutive_even_sum_ways_l663_663668


namespace quadratic_real_roots_prob_classical_correct_quadratic_real_roots_prob_geometric_correct_l663_663865

noncomputable def quadratic_real_roots_prob_classical (A : ℕ → ℕ → Prop) : ℚ :=
let events := {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)} in
have favorable_events := {(a, b) ∈ events | A a b},
(favorable_events.card / events.card : ℚ)

theorem quadratic_real_roots_prob_classical_correct :
  quadratic_real_roots_prob_classical (λ a b => a ≥ b) = 3/4 :=
sorry

noncomputable def quadratic_real_roots_prob_geometric (A : ℝ → ℝ → Prop) : ℚ :=
let volume_total := 6 in  -- Area of [0,3] x [0,2] region
let volume_event := 5 in  -- Area of region satisfying the condition a ≥ b
(volume_event / volume_total : ℚ)

theorem quadratic_real_roots_prob_geometric_correct :
  quadratic_real_roots_prob_geometric (λ a b => a ≥ b) = 2/3 :=
sorry

end quadratic_real_roots_prob_classical_correct_quadratic_real_roots_prob_geometric_correct_l663_663865


namespace part1_range_of_m_part2_max_area_l663_663700

open Real

noncomputable def C₁ (a : ℝ) := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 = 1}
noncomputable def C₂ (m : ℝ) := {p : ℝ × ℝ | p.2^2 = 2 * (p.1 + m)}

theorem part1_range_of_m (a : ℝ) (m : ℝ) :
    (a > 0) → ((∃ p ∈ C₁ a, p ∈ C₂ m ∧ p.2 > 0) →
    ((0 < a < 1) → m = (a^2 + 1) / 2 ∨ -a < m ∧ m ≤ a) ∧
    (a ≥ 1 → -a < m ∧ m < a)) :=
  by intro ha hexists_range; sorry

noncomputable def triangle_area (a x y : ℝ) := 1 / 2 * abs (a * y)

theorem part2_max_area (a : ℝ) :
    (0 < a) → (a < 1 / 2) → 
    (∃ (x_P y_P : ℝ), 
      ((x_P = -a^2 + a*sqrt(a^2 + 1 - 2*m) ∧ y_P = sqrt(2 * (x_P + m))) ∧
      (x_P, y_P).2^2 = 2 * (-a^2 + a * sqrt(a^2 + 1 - 2*m) + m)) →
      triangle_area a _ _ ≤
    if a ≤ 1/3 then 1/2 * a * sqrt(1 - a^2)
    else a * sqrt(a - a^2)) :=
  by intros ha ha_lt_half hP; sorry

end part1_range_of_m_part2_max_area_l663_663700


namespace product_prs_l663_663954

theorem product_prs : ∃ p r s : ℕ, 4^p + 4^3 = 272 ∧ 3^r + 27 = 81 ∧ 2^s + 7^2 = 1024 ∧ p * r * s = 160 :=
by {
  use 4, 4, 10,
  split,
  { exact rfl }, -- proof of the first condition
  split,
  { exact rfl }, -- proof of the second condition
  split,
  { exact rfl }, -- proof of the third condition
  { exact rfl }  -- proof of the product
}

end product_prs_l663_663954


namespace symmetric_circle_eq_l663_663340

theorem symmetric_circle_eq {x y : ℝ} :
  (∃ x y : ℝ, (x+2)^2 + (y-1)^2 = 5) →
  (x - 1)^2 + (y + 2)^2 = 5 :=
sorry

end symmetric_circle_eq_l663_663340


namespace less_sum_mult_l663_663641

theorem less_sum_mult {a b : ℝ} (h1 : a < 1) (h2 : b > 1) : a * b < a + b :=
sorry

end less_sum_mult_l663_663641


namespace knights_max_seated_between_knights_l663_663461

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l663_663461


namespace find_smallest_angle_l663_663891

theorem find_smallest_angle (x : ℝ) (h1 : Real.tan (2 * x) + Real.tan (3 * x) = 1) :
  x = 9 * Real.pi / 180 :=
by
  sorry

end find_smallest_angle_l663_663891


namespace probability_of_successful_rolls_l663_663449

def odd (n : ℕ) : Prop := n % 2 = 1
def multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

theorem probability_of_successful_rolls :
  let total_outcomes := 8 * 12 in
  let successful_8_sided := { n | n ∈ {1, 3, 5, 7} } in
  let successful_12_sided := { n | n ∈ {3, 6, 9, 12} } in
  let successful_outcomes := 4 * 4 in
  successful_outcomes / total_outcomes = 1 / 6 :=
by
  sorry

end probability_of_successful_rolls_l663_663449


namespace money_bounds_l663_663900

   theorem money_bounds (a b : ℝ) (h₁ : 4 * a + 2 * b > 110) (h₂ : 2 * a + 3 * b = 105) : a > 15 ∧ b < 25 :=
   by
     sorry
   
end money_bounds_l663_663900


namespace min_ab_value_l663_663602

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + 9 * b + 7) : a * b ≥ 49 :=
sorry

end min_ab_value_l663_663602


namespace no_solution_f_eq_zero_l663_663292

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then -x^2 - 2 else x / 3 + 2

theorem no_solution_f_eq_zero : ∀ x : ℝ, f x ≠ 0 := by {
  intro x,
  cases le_or_gt x 2 with h1 h2,
  { -- Case: x ≤ 2
    calc f x = -x^2 - 2 : by rw [if_pos h1]
         ... ≠ 0 : sorry },
  { -- Case: x > 2
    calc f x = x / 3 + 2 : by rw [if_neg h2]
         ... ≠ 0 : sorry }
}

end no_solution_f_eq_zero_l663_663292


namespace solution_set_of_inequality_l663_663756

theorem solution_set_of_inequality :
  { x : ℝ | 2 / (x - 1) ≥ 1 } = { x : ℝ | 1 < x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l663_663756


namespace red_balls_l663_663353

theorem red_balls (w r : ℕ) (h1 : w = 12) (h2 : w * 3 = r * 4) : r = 9 :=
sorry

end red_balls_l663_663353


namespace a_seq_correct_l663_663568

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 -- a_0 is not defined in the problem, but we put it to 0 to have a base case
  else if n = 1 then 1
  else 2 * 3^(n-2)

theorem a_seq_correct (n : ℕ) : 
  a_seq n = 
  if n = 1 then 1 
  else if n ≥ 2 then 2 * 3^(n-2) 
  else 0 :=
begin
  sorry
end

end a_seq_correct_l663_663568


namespace inequality_transformation_l663_663041

variable {a b : ℝ}

theorem inequality_transformation (h : a < b) : -a / 3 > -b / 3 :=
  sorry

end inequality_transformation_l663_663041


namespace parallel_line_a_value_line_through_A_equal_intercepts_l663_663948

theorem parallel_line_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, (a^2 - 1) * x + a * y - 1 = 0 ↔ 3 * x + 2 * y - 1 = 0) →
  a = -1/2 := by
  sorry

theorem line_through_A_equal_intercepts :
  ∃ l : AffineLine ℝ,
    l.contains ⟨-1, 2⟩ ∧ (∀ x y : ℝ, l.equation = (√2*x + √2*y = 2*√2 - 1) ∨
                                      l.equation = (√5*x + y = -2 * √5)) := by
  sorry

end parallel_line_a_value_line_through_A_equal_intercepts_l663_663948


namespace num_valid_sets_l663_663297

theorem num_valid_sets : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let count_valid_sets (S : Set ℕ) := ∑ (A : Set ℕ) in Finset.powerset (Finset.range (12 + 1)), 
                                   if A.card = 3 ∧ ∃ a1 a2 a3, a1 ∈ A ∧ a2 ∈ A ∧ a3 ∈ A ∧ a1 < a2 ∧ a2 < a3 ∧ a3 - a2 ≤ 5 then 1 else 0 
  in count_valid_sets S = 185 :=
by sorry

end num_valid_sets_l663_663297


namespace proof_problem_l663_663036

-- Define assumptions
variable (τ_n_x : ℕ → ℕ) (S : ℕ → ℤ) (A B x p q : ℤ) (ξ₁ : ℤ)
variable (E : (ℕ → ℤ) → ℤ) (D : (ℕ → ℤ) → ℤ → ℕ → ℕ)
variable [IsStoppingTime τ_n_x]
variable (h_τ_n_x : ∀ n x, τ_n_x n x = Nat.min' {l | 0 ≤ l ∧ l ≤ n ∧ (S l + x = A ∨ S l + x = B)} n)
variable (h_S : ∀ l, 0 ≤ l ∧ l ≤ n → A - x < S l ∧ S l < B - x)
variable (h_t_stop: ∀ n x, τ_n_x n x represents a stopping moment for a random walk bounded by {A - x, B - x} starting at zero)
variable (h_E_S : ∀ τ_n', E (S τ_n') = (p - q) * E τ_n')

theorem proof_problem :
  ∀ (τ_n_x : ℕ → ℕ) (S : ℕ → ℤ) (A B x p q : ℤ) (ξ₁ : ℤ) (E : (ℕ → ℤ) → ℤ) (D : (ℕ → ℤ) → ℤ → ℕ → ℕ)
  [IsStoppingTime τ_n_x],
  (∀ n x, τ_n_x n x = Nat.min' {l | 0 ≤ l ∧ l ≤ n ∧ (S l + x = A ∨ S l + x = B)} n) →
  (∀ l, 0 ≤ l ∧ l ≤ n → A - x < S l ∧ S l < B - x) →
  (∀ n x, τ_n_x n x represents a stopping moment for a random walk bounded by {A - x, B - x} starting at zero) →
  (∀ τ_n', E (S τ_n') = (p - q) * E τ_n') →
  (E S (τ_n_x) + x = x + (p - q) * E (τ_n_x)) ∧ 
  (E ((S (τ_n_x) - τ_n_x * E ξ₁)^2) = D ξ₁ * E τ_n_x + x^2) := by 
  sorry

end proof_problem_l663_663036


namespace rectangle_area_l663_663356

theorem rectangle_area (L W P : ℝ) (hL : L = 13) (hP : P = 50) (hP_eq : P = 2 * L + 2 * W) :
  L * W = 156 :=
by
  have hL_val : L = 13 := hL
  have hP_val : P = 50 := hP
  have h_perimeter : P = 2 * L + 2 * W := hP_eq
  sorry

end rectangle_area_l663_663356


namespace right_to_left_handed_ratio_l663_663418

def team_ratio (R L : ℕ) (R_absent L_absent : ℕ) : Prop :=
  (R_absent = (1 / 3 : ℝ) * R) ∧ 
  (L_absent = (1 / 3 : ℝ) * L) ∧ 
  ((R_absent / L_absent) ≈ 0.2)

theorem right_to_left_handed_ratio (R L : ℕ) (R_absent L_absent : ℕ)
  (h1 : R_absent = (1 / 3 : ℝ) * R)
  (h2 : L_absent = (1 / 3 : ℝ) * L)
  (h3 : (R_absent / L_absent) ≈ 0.2) : (R / L ≈ 0.2) :=
by 
  unfold team_ratio at *
  have hR_L : R / L = R_absent / L_absent, from sorry
  exact h3.trans hR_L.symm

end right_to_left_handed_ratio_l663_663418


namespace distance_AB_eq_l663_663257

open Real

def curve_cartesian_eq (α : ℝ) : Prop :=
  let x := cos α
  let y := sqrt 3 * sin α
  x^2 + y^2 / 3 = 1

def line_cartesian_eq (ρ θ : ℝ) : Prop :=
  let cart_eq := sqrt 2 / 2 * sqrt (ρ^2) * cos (θ + π / 4) = -1
  cart_eq = (x - y + 2 = 0)

theorem distance_AB_eq (x1 y1 x2 y2 : ℝ) (hA : x1 = -sqrt 2 / 2 ∧ y1 = -sqrt 2 / 2 - 1)
  (hB : x2 = sqrt 2 ∧ y2 = sqrt 2 - 1) : 
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 3 * sqrt 2 / 2 :=
sorry

end distance_AB_eq_l663_663257


namespace chocolate_more_expensive_l663_663043

variables (C P : ℝ)
theorem chocolate_more_expensive (h : 7 * C > 8 * P) : 8 * C > 9 * P :=
sorry

end chocolate_more_expensive_l663_663043


namespace closed_path_even_length_l663_663423

def is_closed_path (steps : List Char) : Bool :=
  let net_vertical := steps.count 'U' - steps.count 'D'
  let net_horizontal := steps.count 'R' - steps.count 'L'
  net_vertical = 0 ∧ net_horizontal = 0

def move_length (steps : List Char) : Nat :=
  steps.length

theorem closed_path_even_length (steps : List Char) :
  is_closed_path steps = true → move_length steps % 2 = 0 :=
by
  -- Conditions extracted as definitions
  intros h
  -- The proof will handle showing that the length of the closed path is even
  sorry

end closed_path_even_length_l663_663423


namespace max_knights_between_other_knights_l663_663474

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l663_663474


namespace sqrt_pow_six_eq_729_l663_663530

theorem sqrt_pow_six_eq_729 :
  (\sqrt{((\sqrt 3) ^ 4)} ^ 6) = 729 :=
sorry

end sqrt_pow_six_eq_729_l663_663530


namespace significant_digits_of_square_side_l663_663438

noncomputable def side_length (A : ℝ) : ℝ := real.sqrt A
def significant_digits (A : ℝ) : ℕ := 
  match to_string A with
  | s => s.filter (λ c, c ≠ '0' ∧ c ≠ '.').length -- Counting significant digits

theorem significant_digits_of_square_side 
  (A P : ℝ) 
  (h1 : A = 1.44) 
  (h2 : P = 5) 
  (h3 : A = (side_length A) * (side_length A)) 
  (h4 : P = 4 * (side_length A)) 
  : significant_digits (side_length A) = 2 :=
sorry

end significant_digits_of_square_side_l663_663438


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663233

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663233


namespace sum_of_pairwise_products_leq_one_fourth_l663_663313

theorem sum_of_pairwise_products_leq_one_fourth
  {n : ℕ} (hn : n ≥ 4) (a : Fin n → ℝ)
  (h_nonneg : ∀ i, 0 ≤ a i)
  (h_sum : ∑ i, a i = 1) : 
  (∑ i, a i * a ((i + 1) % n)) ≤ 1 / 4 :=
sorry

end sum_of_pairwise_products_leq_one_fourth_l663_663313


namespace distinguishable_colorings_tetrahedron_l663_663517

noncomputable def tetrahedronColorings : ℕ :=
  40

theorem distinguishable_colorings_tetrahedron :
  let colors := 4 in
  let faces := 4 in
  let permutations := 12 in  -- in a regular tetrahedron, there are 12 unique rotations.
  tetrahedronColorings = 40 :=
by
  sorry

end distinguishable_colorings_tetrahedron_l663_663517


namespace A_sufficient_not_necessary_for_B_l663_663699

noncomputable def propositionA (x : ℝ) : Prop :=
0 < x ∧ x < 5

noncomputable def propositionB (x : ℝ) : Prop :=
|x - 2| < 3

theorem A_sufficient_not_necessary_for_B :
  (∀ x, propositionA x → propositionB x) ∧ ¬(∀ x, propositionB x → propositionA x) :=
by
  sorry

end A_sufficient_not_necessary_for_B_l663_663699


namespace probability_of_first_greater_second_l663_663052

def is_fair_die (n : ℕ) : Prop := n = 8

def probability_first_greater_second (die_faces : ℕ) : ℚ :=
let total_outcomes := die_faces * die_faces in
let favorable_outcomes := (die_faces * (die_faces - 1)) / 2 in
favorable_outcomes / total_outcomes

theorem probability_of_first_greater_second
  (n : ℕ) (h_fair : is_fair_die n) :
  probability_first_greater_second n = 7 / 16 :=
by
  subst h_fair
  simp [probability_first_greater_second]
  sorry

end probability_of_first_greater_second_l663_663052


namespace product_of_20_random_digits_ends_in_0_probability_l663_663570

theorem product_of_20_random_digits_ends_in_0_probability :
  let prob_at_least_one_0 := 1 - (9 / 10) ^ 20,
      prob_even_digit := 1 - (5 / 9) ^ 20,
      prob_5 := 1 - (8 / 9) ^ 19
  in 
    prob_at_least_one_0 + ( (9 / 10) ^ 20 * prob_even_digit * prob_5 ) ≈ 0.988 :=
by sorry

end product_of_20_random_digits_ends_in_0_probability_l663_663570


namespace tiffany_found_bags_l663_663373

theorem tiffany_found_bags (initial_bags : ℕ) (total_bags : ℕ) (found_bags : ℕ) :
  initial_bags = 4 ∧ total_bags = 12 ∧ total_bags = initial_bags + found_bags → found_bags = 8 :=
by
  sorry

end tiffany_found_bags_l663_663373


namespace max_knights_between_knights_l663_663468

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l663_663468


namespace number_of_arithmetic_sequences_is_three_l663_663079

-- Definitions for each sequence
def seq1 (n : ℕ) : ℤ := n + 4

def seq2 : ℕ → ℤ
| 0 => 3
| 1 => 0
| 2 => -3
| _ => sorry  -- Skipped the incorrect sequence definition for simplicity

def seq3 (n : ℕ) : ℤ := 0

def seq4 (n : ℕ) : ℚ := (n + 1) / 10

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (seq : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, seq (n + 1) - seq n = d

-- The theorem to prove
theorem number_of_arithmetic_sequences_is_three :
  [seq1, seq3, seq4].count is_arithmetic_sequence = 3 :=
by simp

end number_of_arithmetic_sequences_is_three_l663_663079


namespace largest_6_digit_number_div_by_41_l663_663778

-- Define the condition of being a 6-digit number
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

-- Define the condition of being divisible by 41
def divisible_by_41 (n : ℕ) : Prop := n % 41 = 0

-- The problem statement asserting the largest 6-digit number divisible by 41
theorem largest_6_digit_number_div_by_41 : ∃ n, is_six_digit n ∧ divisible_by_41 n ∧ ∀ m, is_six_digit m ∧ divisible_by_41 m → m ≤ n :=
  ∃ n, is_six_digit n ∧ divisible_by_41 n ∧ ∀ m, is_six_digit m ∧ divisible_by_41 m → m ≤ 999990 :=
sorry

end largest_6_digit_number_div_by_41_l663_663778


namespace quadratic_parabola_equation_l663_663618

theorem quadratic_parabola_equation :
  ∃ (a b c : ℝ), 
    (∀ x y, y = 3 * x^2 - 6 * x + 5 → (x - 1)*(x - 1) = (x - 1)^2) ∧ -- Original vertex condition and standard form
    (∀ x y, y = -x - 2 → a = 2) ∧ -- Given intersection point condition
    (∀ x y, y = -3 * (x - 1)^2 + 2 → y = -3 * (x - 1)^2 + b ∧ y = -4) → -- Vertex unchanged and direction reversed
    (a, b, c) = (-3, 6, -4) := -- Resulting equation coefficients
sorry

end quadratic_parabola_equation_l663_663618


namespace geometric_seq_relation_l663_663593

theorem geometric_seq_relation (a : ℕ → ℝ) (q : ℝ) (a_pos : ∀ n, 0 < a n) (q_ne_one : q ≠ 1) :
  a 1 + a 4 > a 2 + a 3 :=
by
  -- Definitions and conditions
  assume h_geom : ∀ n, a (n + 1) = a n * q,
  -- The steps are assumed correct through the value relationship and properties of q and a.
  sorry

end geometric_seq_relation_l663_663593


namespace log_identity_solution_l663_663960

theorem log_identity_solution (P : ℝ) (h : logBase 2 (logBase 4 P) = logBase 4 (logBase 2 P)) (h_ne : P ≠ 1) : P = 16 :=
by
  sorry

end log_identity_solution_l663_663960


namespace probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five_l663_663575

noncomputable def probability_of_product_ending_in_0 : ℝ :=
  let p_no_0 := (9 / 10 : ℝ) ^ 20
  let p_at_least_one_0 := 1 - p_no_0
  let p_no_even := (5 / 9 : ℝ) ^ 20
  let p_at_least_one_even := 1 - p_no_even
  let p_no_5_in_19 := (8 / 9 : ℝ) ^ 19
  let p_at_least_one_5 := 1 - p_no_5_in_19
  p_at_least_one_0 + p_no_0 * p_at_least_one_even * p_at_least_one_5

theorem probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five :
  abs (probability_of_product_ending_in_0 - 0.9865) < 0.0001 :=
begin
  -- proofs go here, sorry is used to skip the proof details
  sorry
end

end probability_product_ends_in_0_approx_zero_pt_nine_eight_six_five_l663_663575


namespace cos_zero_degree_l663_663860

-- Let θ be the angle in radians, with θ = 0 for 0 degrees
def θ : ℝ := 0

-- Define the point on the unit circle
def unit_circle_point : ℝ × ℝ := (1, 0)

-- Definition of rotation of the point (x, y) by an angle θ counterclockwise
def rotate (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  (p.fst * Real.cos θ - p.snd * Real.sin θ, p.fst * Real.sin θ + p.snd * Real.cos θ)

theorem cos_zero_degree :
  rotate unit_circle_point θ = (1, 0) → (Real.cos 0 = 1) :=
by
  intro h
  have h_rotation : rotate unit_circle_point θ = (1, 0) := h
  show Real.cos 0 = 1
  sorry

end cos_zero_degree_l663_663860


namespace XYZ_total_length_is_correct_l663_663863

structure segments where
  vertical : ℕ
  horizontal : ℕ
  slanted : ℕ

def length_of_segments (s : segments) : ℝ :=
  (s.vertical + s.horizontal) * 2 + s.slanted * (2 * Real.sqrt 2)

def XYZ_segments : segments :=
  { vertical := 2,
    horizontal := 2,
    slanted := 5 }

theorem XYZ_total_length_is_correct :
  length_of_segments XYZ_segments = 8 + 10 * Real.sqrt 2 :=
by
  sorry

end XYZ_total_length_is_correct_l663_663863


namespace sum_arithmetic_series_1000_to_5000_l663_663385

theorem sum_arithmetic_series_1000_to_5000 :
  ∑ k in finset.range (5001 - 1000), (1000 + k) = 12003000 :=
by
  sorry

end sum_arithmetic_series_1000_to_5000_l663_663385


namespace cyclic_quadrilateral_JMIT_l663_663282

theorem cyclic_quadrilateral_JMIT
  (a b c : ℂ)
  (I J M N T : ℂ)
  (hI : I = -(a*b + b*c + c*a))
  (hJ : J = a*b - b*c + c*a)
  (hM : M = (b^2 + c^2) / 2)
  (hN : N = b*c)
  (hT : T = 2*a^2 - b*c) :
  ∃ (k : ℝ), k = ((M - I) * (T - J)) / ((J - I) * (T - M)) :=
by
  sorry

end cyclic_quadrilateral_JMIT_l663_663282


namespace greatest_integer_1000y_l663_663071

theorem greatest_integer_1000y 
  (edge_length : ℝ := 2)
  (y : ℝ)
  (shadow_area_excluding_cube : ℝ := 112)
  (total_shadow_area : ℝ := shadow_area_excluding_cube + edge_length^2)
  (shadow_side_length : ℝ := real.sqrt total_shadow_area)
  (cube_height : ℝ := edge_length)
  (similarity_factor : ℝ := (shadow_side_length - cube_height) / cube_height)
  (y_value : ℝ := cube_height * similarity_factor) :
  ¬1000 * (y) < 1000 * 8.77 + 1 :=
by
  have h : y = 8.77, sorry
  have : 1000 * y = 8770, sorry
  have : 1000 * y < 8770 + 1, sorry
  exact this

end greatest_integer_1000y_l663_663071


namespace floor_sqrt_200_l663_663522

theorem floor_sqrt_200 : 
  let a := 200 
  let sqrt_a := Real.sqrt a 
  (196 < a ∧ a < 225) → 
  (14 < sqrt_a ∧ sqrt_a < 15) → 
  Int.floor sqrt_a = 14 := 
by {
  intros,
  sorry
}

end floor_sqrt_200_l663_663522


namespace overall_progress_in_meters_l663_663704

-- Definitions for the given conditions
def x := -15 -- yards
def y := 20 -- yards
def z := 10 -- yards
def w := 25 -- yards
def v := 5 -- yards

-- Initial conversion factor
def yard_to_meter := 0.9144

-- Expression for overall progress in yards
def overall_progress_yards := x + y - z + w + (0.5 * y) - v

-- Expression for overall progress in meters
def overall_progress_meters := overall_progress_yards * yard_to_meter

-- Statement to prove
theorem overall_progress_in_meters : overall_progress_meters = 22.86 :=
by
  -- Using the definitions above, we can now compute the resultant expression
  sorry

end overall_progress_in_meters_l663_663704


namespace mr_green_expects_expected_potatoes_yield_l663_663709

theorem mr_green_expects_expected_potatoes_yield :
  ∀ (length_steps width_steps: ℕ) (step_length yield_per_sqft: ℝ),
  length_steps = 18 →
  width_steps = 25 →
  step_length = 2.5 →
  yield_per_sqft = 0.75 →
  (length_steps * step_length) * (width_steps * step_length) * yield_per_sqft = 2109.375 :=
by
  intros length_steps width_steps step_length yield_per_sqft
  intros h_length_steps h_width_steps h_step_length h_yield_per_sqft
  rw [h_length_steps, h_width_steps, h_step_length, h_yield_per_sqft]
  sorry

end mr_green_expects_expected_potatoes_yield_l663_663709


namespace line_circle_slope_l663_663970

theorem line_circle_slope (m n : ℝ) (h : n ≠ 0) :
  (∃ k : ℝ, k = m / n ∧
    ((abs (3 * m + 2 * n - m - n)) / (real.sqrt (m ^ 2 + n ^ 2)) = 1)) →
  m / n = 0 ∨ m / n = 4 / 3 :=
by
  intro h1
  sorry

end line_circle_slope_l663_663970


namespace sum_of_consecutive_integers_of_sqrt3_l663_663205

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l663_663205


namespace determine_x_l663_663328

theorem determine_x (A B C : ℝ) (x : ℝ) (h1 : C > B) (h2 : B > A) (h3 : A > 0)
  (h4 : A = B - (x / 100) * B) (h5 : C = A + 2 * B) :
  x = 100 * ((B - A) / B) :=
sorry

end determine_x_l663_663328


namespace number_of_games_l663_663082

variable (shelves : ℝ) (games_per_shelf : ℝ)

theorem number_of_games (h1 : shelves ≈ 2) (h2 : games_per_shelf = 84) : shelves * games_per_shelf ≈ 168 :=
  sorry

end number_of_games_l663_663082


namespace range_of_expression_l663_663748

theorem range_of_expression (a : ℝ) : (∃ a : ℝ, a + 1 ≥ 0 ∧ a - 2 ≠ 0) → (a ≥ -1 ∧ a ≠ 2) := 
by sorry

end range_of_expression_l663_663748


namespace find_multiplier_l663_663171

theorem find_multiplier (x : ℕ) (h1 : 268 * x = 19832) (h2 : 2.68 * 0.74 = 1.9832) : x = 74 :=
sorry

end find_multiplier_l663_663171


namespace hyperbola_eccentricity_l663_663616

variables {t : ℝ}

theorem hyperbola_eccentricity (t = 2) (focal_dist : ℝ) (eccentricity : ℝ) :
  (x^2 - 2 * y^2 = 6) ∧ (2 * c = 6) → eccentricity = sqrt 6 / 2 :=
by
  sorry

end hyperbola_eccentricity_l663_663616


namespace sum_of_factorials_last_two_digits_l663_663499

theorem sum_of_factorials_last_two_digits :
  (Finset.sum (Finset.range' 3 11) (λ n, if n % 10 = 3 then (nat.factorial (n + 10 * (n - 3) / 10)) else 0)) % 100 = 6 :=
by sorry

end sum_of_factorials_last_two_digits_l663_663499


namespace ice_cream_flavors_l663_663634

theorem ice_cream_flavors (scoops flavors : ℕ) (h_scoops : scoops = 5) (h_flavors : flavors = 3) : 
  Nat.choose (scoops + flavors - 1) (flavors - 1) = 21 :=
by
  rw [h_scoops, h_flavors]
  -- Here, the actual computation and proof would typically go, but for the purposes 
  -- of this task as per the provided instruction, we leave it to sorry.
  sorry

end ice_cream_flavors_l663_663634


namespace cos_arithmetic_sequence_l663_663926

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem cos_arithmetic_sequence (a : ℕ → ℝ) (h_seq : arithmetic_sequence a) (h_sum : a 1 + a 5 + a 9 = 8 * Real.pi) :
  Real.cos (a 3 + a 7) = -1 / 2 :=
sorry

end cos_arithmetic_sequence_l663_663926


namespace math_problem_l663_663862

-- Define necessary elements in the proof
def zero_exponent (x : ℝ) (h : x ≠ 0) : (x ^ 0) = 1 := by
  exact Real.rpow_zero _

def abs_val (x : ℝ) : ℝ := abs x

def neg_exp (x : ℝ) (h : x ≠ 0) : (x ^ (-1)) = 1 / x := by
  exact Real.rpow_neg_one _

def cos_30 := Real.cos (Real.pi / 6)

theorem math_problem :
  (1 - Real.pi) ^ 0 - abs (3 - 2 * Real.sqrt 3) + (-⅓) ^ (-1) + 4 * Real.cos (Real.pi / 6) 
  = -1.464 + 2 * Real.sqrt 3 := by
  sorry

end math_problem_l663_663862


namespace honey_calculation_l663_663415

noncomputable def honey_production (A_w : ℝ) (A_s : ℝ) (A_k : ℝ) 
                                  (B_w : ℝ) (B_s : ℝ) (B_k : ℝ) 
                                  (C_w : ℝ) (C_s : ℝ) (C_k : ℝ) 
                                  (h_w : ℝ) 
                                  (h_s : ℝ) : ℝ := 
  let total_s := A_k * A_s + B_k * B_s + C_k * C_s in 
  total_s / h_s

theorem honey_calculation :
  honey_production 0.6 0.4 2 0.5 0.5 3 0.4 0.6 1 0.3 0.7 ≈ 4.14 := 
by sorry

end honey_calculation_l663_663415


namespace minimum_value_PM_minus_PF1_l663_663286

noncomputable def ellipse_equation (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 16) = 1

def focal_point_1 := (-3, 0 : ℝ × ℝ)
def focal_point_2 := (3, 0 : ℝ × ℝ)
def point_M := (6, 4 : ℝ × ℝ)

theorem minimum_value_PM_minus_PF1 :
  ∃ P : ℝ × ℝ, ellipse_equation P.1 P.2 ∧ 
  ∀ Q : ℝ × ℝ, ellipse_equation Q.1 Q.2 → |((Q.1 - point_M.1)^2 + (Q.2 - point_M.2)^2)^(1/2) - ((Q.1 - focal_point_1.1)^2 + (Q.2 - focal_point_1.2)^2)^(1/2)| ≥ -5 :=
sorry

end minimum_value_PM_minus_PF1_l663_663286


namespace dave_initial_video_games_l663_663110

theorem dave_initial_video_games (non_working_games working_game_price total_earnings : ℕ) 
  (h1 : non_working_games = 2) 
  (h2 : working_game_price = 4) 
  (h3 : total_earnings = 32) : 
  non_working_games + total_earnings / working_game_price = 10 := 
by 
  sorry

end dave_initial_video_games_l663_663110


namespace unique_n_for_given_divisors_l663_663690

theorem unique_n_for_given_divisors :
  ∃! (n : ℕ), 
    ∀ (k : ℕ) (d : ℕ → ℕ), 
      k ≥ 22 ∧ 
      d 1 = 1 ∧ d k = n ∧ 
      (∀ i j, i < j → d i < d j) ∧ 
      (d 7) ^ 2 + (d 10) ^ 2 = (n / d 22) ^ 2 →
      n = 2^3 * 3 * 5 * 17 :=
sorry

end unique_n_for_given_divisors_l663_663690


namespace stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l663_663506

-- Definition of the speeds
def speed_excluding_stoppages_A : ℕ := 60
def speed_including_stoppages_A : ℕ := 48
def speed_excluding_stoppages_B : ℕ := 75
def speed_including_stoppages_B : ℕ := 60
def speed_excluding_stoppages_C : ℕ := 90
def speed_including_stoppages_C : ℕ := 72

-- Theorem to prove the stopped time per hour for each bus
theorem stopped_time_per_hour_A : (speed_excluding_stoppages_A - speed_including_stoppages_A) * 60 / speed_excluding_stoppages_A = 12 := sorry

theorem stopped_time_per_hour_B : (speed_excluding_stoppages_B - speed_including_stoppages_B) * 60 / speed_excluding_stoppages_B = 12 := sorry

theorem stopped_time_per_hour_C : (speed_excluding_stoppages_C - speed_including_stoppages_C) * 60 / speed_excluding_stoppages_C = 12 := sorry

end stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l663_663506


namespace knights_max_seated_between_knights_l663_663462

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l663_663462


namespace triangle_equilateral_of_ha_ha_eq_equilateral_triangle_of_equal_squares_l663_663792

-- Part (a)
theorem triangle_equilateral_of_ha_ha_eq (a b c h_a h_b h_c : ℝ)
  (H : a + h_a = b + h_b ∧ b + h_b = c + h_c) :
  (∃ (ABC : Triangle), ABC.isEquilateral) := 
by sorry

-- Part (b)
theorem equilateral_triangle_of_equal_squares (a b c h_a h_b h_c : ℝ)
  (S : ℝ)
  (H_sq1 : ∃ (square1 : Square), square1.side_length = 2 * S / (a + h_a))
  (H_sq2 : ∃ (square2 : Square), square2.side_length = 2 * S / (b + h_b))
  (H_sq3 : ∃ (square3 : Square), square3.side_length = 2 * S / (c + h_c))
  (H_eq : (2 * S / (a + h_a) = 2 * S / (b + h_b))
  ∧ (2 * S / (b + h_b) = 2 * S / (c + h_c))) :
  (∃ (ABC : Triangle), ABC.isEquilateral) := 
by sorry

end triangle_equilateral_of_ha_ha_eq_equilateral_triangle_of_equal_squares_l663_663792


namespace symmetric_circle_equation_l663_663745

theorem symmetric_circle_equation (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 5 → (x + 1)^2 + (y - 2)^2 = 5 :=
by
  sorry

end symmetric_circle_equation_l663_663745


namespace ratio_as_percent_l663_663808

def first_part : ℕ := 4
def second_part : ℕ := 20

theorem ratio_as_percent : (first_part.toRat / second_part.toRat) * 100 = 20 := by
  sorry

end ratio_as_percent_l663_663808


namespace smallest_nine_integer_circle_l663_663710

theorem smallest_nine_integer_circle (n : ℕ) 
  (a : ℕ → ℕ) 
  (distinct : ∀ i j, 0 ≤ i < 9 → 0 ≤ j < 9 → i ≠ j → a i ≠ a j)
  (non_adj_mult : ∀ i j, 0 ≤ i < 9 → 0 ≤ j < 9 → (i-j) % 9 ≠ 1 ∧ (j-i) % 9 ≠ 1 → n ∣ (a i * a j))
  (adj_not_mult : ∀ i, 0 ≤ i < 9 → ¬ n ∣ (a i * a ((i+1) % 9))) : 
  n = 485100 :=
sorry

end smallest_nine_integer_circle_l663_663710


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663232

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663232


namespace iron_conducts_electricity_l663_663039

-- Define the predicates
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
noncomputable def Iron : Type := sorry
  
theorem iron_conducts_electricity (h1 : ∀ x, Metal x → ConductsElectricity x)
  (h2 : Metal Iron) : ConductsElectricity Iron :=
by
  sorry

end iron_conducts_electricity_l663_663039


namespace watermelon_percentage_l663_663818

theorem watermelon_percentage (total_drink : ℕ)
  (orange_percentage : ℕ)
  (grape_juice : ℕ)
  (watermelon_amount : ℕ)
  (W : ℕ) :
  total_drink = 300 →
  orange_percentage = 25 →
  grape_juice = 105 →
  watermelon_amount = total_drink - (orange_percentage * total_drink) / 100 - grape_juice →
  W = (watermelon_amount * 100) / total_drink →
  W = 40 :=
sorry

end watermelon_percentage_l663_663818


namespace lock_combination_l663_663723

def valid_combination (T I D E b : ℕ) : Prop :=
  (T > 0) ∧ (I > 0) ∧ (D > 0) ∧ (E > 0) ∧
  (T ≠ I) ∧ (T ≠ D) ∧ (T ≠ E) ∧ (I ≠ D) ∧ (I ≠ E) ∧ (D ≠ E) ∧
  (T * b^3 + I * b^2 + D * b + E) + 
  (E * b^3 + D * b^2 + I * b + T) + 
  (T * b^3 + I * b^2 + D * b + E) = 
  (D * b^3 + I * b^2 + E * b + T)

theorem lock_combination : ∃ (T I D E b : ℕ), valid_combination T I D E b ∧ (T * 100 + I * 10 + D = 984) :=
sorry

end lock_combination_l663_663723


namespace minimize_total_cost_l663_663420

noncomputable def fuelCost (x : ℝ) : ℝ :=
  (2 / 3) * x^2

noncomputable def otherCosts (x : ℝ) : ℝ :=
  864

noncomputable def totalCost (x : ℝ) : ℝ :=
  (fuelCost x) * (100 / x) + (otherCosts x) * (100 / x)

theorem minimize_total_cost :
  ∃ x ∈ Ioo (0 : ℝ) 48, x = 36 ∧ totalCost x = 4800 :=
by
  -- proof omitted
  sorry

end minimize_total_cost_l663_663420


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663231

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l663_663231


namespace perimeter_of_regular_polygon_l663_663116

-- Definitions based on conditions
def side_length : ℝ := 7
def exterior_angle : ℝ := 90
def num_sides : ℝ := 360 / exterior_angle

-- The hypothesis directly from the given conditions
def is_regular_polygon : Prop := num_sides = 4

-- Definition of the perimeter given the number of sides and side length
def perimeter := num_sides * side_length

-- Lean 4 statement of the proof problem
theorem perimeter_of_regular_polygon
  (h1 : side_length = 7)
  (h2 : exterior_angle = 90)
  (h3 : is_regular_polygon)
  : perimeter = 28 := by
  sorry

end perimeter_of_regular_polygon_l663_663116


namespace sum_of_consecutive_integers_l663_663220

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l663_663220


namespace annulus_contains_at_least_10_points_l663_663761

theorem annulus_contains_at_least_10_points :
  ∀ (points : Finset (ℝ × ℝ)), 
    points.card = 650 → 
    ∃ (x : ℝ × ℝ), 
      (∃ (r1 r2 : ℝ), r1 = 2 ∧ r2 = 3 ∧ 
        (Finset.filter 
          (λ p, r1^2 ≤ (p.1 - x.1)^2 + (p.2 - x.2)^2 ∧ (p.1 - x.1)^2 + (p.2 - x.2)^2 ≤ r2^2) 
          points).card ≥ 10) := 
by {
  sorry
}

end annulus_contains_at_least_10_points_l663_663761


namespace books_per_bookshelf_l663_663100

theorem books_per_bookshelf (total_books bookshelves : ℕ) (h_total_books : total_books = 34) (h_bookshelves : bookshelves = 2) : total_books / bookshelves = 17 :=
by
  sorry

end books_per_bookshelf_l663_663100


namespace max_sqrt_sum_l663_663912

theorem max_sqrt_sum (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hxy : x + y = 8) :
  abs (Real.sqrt (x - 1 / y) + Real.sqrt (y - 1 / x)) ≤ Real.sqrt 15 :=
sorry

end max_sqrt_sum_l663_663912


namespace num_4_digit_odd_numbers_l663_663760

theorem num_4_digit_odd_numbers :
  let cards := [1, 1, 2, 3, 4, 5]
  in (number_of_4_digit_odd_numbers cards) = 126 :=
by
  sorry

end num_4_digit_odd_numbers_l663_663760


namespace laps_remaining_l663_663273

theorem laps_remaining 
  (total_laps : ℕ)
  (saturday_laps : ℕ)
  (sunday_morning_laps : ℕ)
  (total_laps_eq : total_laps = 98)
  (saturday_laps_eq : saturday_laps = 27)
  (sunday_morning_laps_eq : sunday_morning_laps = 15) :
  total_laps - saturday_laps - sunday_morning_laps = 56 :=
by
  rw [total_laps_eq, saturday_laps_eq, sunday_morning_laps_eq]
  norm_num

end laps_remaining_l663_663273


namespace Q_is_midpoint_of_AB_l663_663695

noncomputable def midpoint_values (a b : ℝ) : Prop :=
  (a + b) / 2 = 13 ∧ a / 2 = 4

theorem Q_is_midpoint_of_AB :
  midpoint_values 8 18 :=
by
  unfold midpoint_values
  split
  norm_num
  norm_num
  sorry

end Q_is_midpoint_of_AB_l663_663695


namespace hostel_initial_plan_l663_663057

variable (x : ℕ) -- representing the initial number of days

-- Define the conditions
def provisions_for_250_men (x : ℕ) : ℕ := 250 * x
def provisions_for_200_men_45_days : ℕ := 200 * 45

-- Prove the statement
theorem hostel_initial_plan (x : ℕ) (h : provisions_for_250_men x = provisions_for_200_men_45_days) :
  x = 36 :=
by
  sorry

end hostel_initial_plan_l663_663057


namespace sum_of_two_numbers_l663_663355

theorem sum_of_two_numbers (x y : ℕ) (h1 : 3 * x = 180) (h2 : 4 * x = y) : x + y = 420 := by
  sorry

end sum_of_two_numbers_l663_663355


namespace samantha_final_score_l663_663663

theorem samantha_final_score (correct incorrect unanswered : ℕ) (correct_pts incorrect_pts : ℤ) :
  correct = 15 → incorrect = 5 → unanswered = 5 → correct_pts = 2 → incorrect_pts = -1 →
  (correct * correct_pts + incorrect * incorrect_pts = 25) :=
by
  intros h_correct h_incorrect h_unanswered h_correct_pts h_incorrect_pts
  rw [h_correct, h_incorrect, h_unanswered, h_correct_pts, h_incorrect_pts]
  sorry

end samantha_final_score_l663_663663


namespace area_of_shaded_quadrilateral_l663_663840

-- The problem setup
variables 
  (triangle : Type) [Nonempty triangle]
  (area : triangle → ℝ)
  (EFA FAB FBD CEDF : triangle)
  (h_EFA : area EFA = 5)
  (h_FAB : area FAB = 9)
  (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF)

-- The goal to prove
theorem area_of_shaded_quadrilateral (EFA FAB FBD CEDF : triangle) 
  (h_EFA : area EFA = 5) (h_FAB : area FAB = 9) (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF) : 
  area CEDF = 45 :=
by
  sorry

end area_of_shaded_quadrilateral_l663_663840


namespace percentage_decrease_of_y_compared_to_z_l663_663977

theorem percentage_decrease_of_y_compared_to_z (x y z : ℝ)
  (h1 : x = 1.20 * y)
  (h2 : x = 0.60 * z) :
  (y = 0.50 * z) → (1 - (y / z)) * 100 = 50 :=
by
  sorry

end percentage_decrease_of_y_compared_to_z_l663_663977


namespace expression_eval_l663_663525

variable (a b c : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem expression_eval : 
  ( (1/a + 1/b + 1/c) * (ab + bc + ca)⁻² * (a + b + c) * ( (1/ab + 1/bc + 1/ca) )⁻¹ = (1/(ab + bc + ca) ^ 2) ) :=
by
  sorry

end expression_eval_l663_663525


namespace expression_positive_intervals_l663_663113
open Real

theorem expression_positive_intervals (x : ℝ) :
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := by
  sorry

end expression_positive_intervals_l663_663113


namespace distinct_zeros_abs_minus_one_l663_663446

theorem distinct_zeros_abs_minus_one : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (|x₁| - 1 = 0) ∧ (|x₂| - 1 = 0) := 
by
  sorry

end distinct_zeros_abs_minus_one_l663_663446


namespace sum_of_consecutive_integers_l663_663222

theorem sum_of_consecutive_integers (a b : ℤ) (h1 : a + 1 = b) (h2 : a < real.sqrt 3) (h3 : real.sqrt 3 < b) : a + b = 3 :=
sorry

end sum_of_consecutive_integers_l663_663222


namespace distance_between_intersections_l663_663821

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 16) - (y^2 / 9) = 1

noncomputable def parabola_eq (x y : ℝ) : Prop :=
  x = (y^2 / 10) + 2.5

theorem distance_between_intersections :
  ∃ y1 y2 : ℝ, hyperbola_eq ((y1^2 / 10) + 2.5) y1 ∧ hyperbola_eq ((y2^2 / 10) + 2.5) y2 ∧
  y1 ≠ y2 ∧ dist (0, y1) (0, y2) = 4 * Real.sqrt 218 / 15 :=
begin
  sorry
end

end distance_between_intersections_l663_663821


namespace lines_parallel_lines_perpendicular_l663_663923

-- Conditions for the lines
def l1 (m : ℝ) : (ℝ × ℝ) → Prop := λ p, p.1 + m * p.2 + 6 = 0
def l2 (m : ℝ) : (ℝ × ℝ) → Prop := λ p, (m-2) * p.1 + 3 * m * p.2 + 2 * m = 0

-- Proving parallelism
theorem lines_parallel (m : ℝ) : 
  (∀ p : ℝ × ℝ, l1 m p → l2 m p) ↔ (m = 0 ∨ m = 5) := sorry

-- Proving perpendicularity
theorem lines_perpendicular (m : ℝ) :
  (∃ p : ℝ × ℝ, l1 m p ∧ l2 m p) ↔ (m = -1 ∨ m = 2/3) := sorry

end lines_parallel_lines_perpendicular_l663_663923


namespace club_president_vice_president_l663_663716

-- Definitions of the given conditions
def boys_count : ℕ := 18
def girls_count : ℕ := 12

-- Calculation for the number of ways
def boys_ways : ℕ := boys_count * (boys_count - 1)
def girls_ways : ℕ := girls_count * (girls_count - 1)
def total_ways : ℕ := boys_ways + girls_ways

-- Lean proof statement asserting the total number of ways
theorem club_president_vice_president :
  total_ways = 438 :=
by
  rw [boys_ways, girls_ways, total_ways]
  sorry

end club_president_vice_president_l663_663716


namespace conj_of_complex_number_l663_663607

theorem conj_of_complex_number : 
  let i := Complex.I in
  let z := Complex.abs ((Real.sqrt 3 - i) * i) - i^5 in
  Complex.conj z = 2 + i := by
  sorry

end conj_of_complex_number_l663_663607


namespace minimum_distance_a_is_five_l663_663599

theorem minimum_distance_a_is_five
  (P_on_curve : ∀ (x : ℝ), x ∈ Icc (-1 : ℝ) (real.sqrt 3) → ∃ (y : ℝ), y = real.sqrt (x^2 + 1) ∧ (x, y) = P)
  (Q_on_circle : ∀ (x y : ℝ), x^2 + (y - a)^2 = 3/4 → (x, y) = Q)
  (h : ∃ (dist : ℝ), dist = 3/2 * real.sqrt 3 ∧ ∀ (P Q : ℝ × ℝ), dist = real.sqrt ((fst P - fst Q)^2 + (snd P - snd Q)^2)) :
  a = 5 :=
sorry

end minimum_distance_a_is_five_l663_663599


namespace candidate_marks_l663_663417

theorem candidate_marks 
  (m : ℝ) 
  (p : ℝ) 
  (f : ℝ) 
  (h₀ : m = 153.84615384615384) 
  (h₁ : p = 0.52) 
  (h₂ : f = 35) :
  let passing_marks := p * m in
  let x := passing_marks - f in
  x = 45 := 
by
  simp [h₀, h₁, h₂, passing_marks, x]
  norm_num
  sorry

end candidate_marks_l663_663417


namespace proof_point_D_proof_length_AB_l663_663674

-- Define the right triangle with given properties
structure right_triangle (A B C : Type) :=
  (angle_B_90 : ∠B = 90)
  (length_AC : AC = 100)
  (slope_AC : slope AC = 4 / 3)

-- Define the point D which is equidistant from A, B, and C
def point_D_is_circumcenter_of_right_triangle {A B C : Type} [right_triangle A B C] : Prop :=
  let D := midpoint A C in
  (D = (30, 40))

-- Define the length of AB in the right triangle
def length_AB_of_right_triangle {A B C : Type} [right_triangle A B C] : Prop :=
  (AB = 60)

-- Theorems to prove
theorem proof_point_D {A B C : Type} [tri : right_triangle A B C] : 
  point_D_is_circumcenter_of_right_triangle := by
  sorry

theorem proof_length_AB {A B C : Type} [tri : right_triangle A B C] : 
  length_AB_of_right_triangle := by
  sorry

end proof_point_D_proof_length_AB_l663_663674


namespace problem_statement_l663_663547

theorem problem_statement (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 1) (x : ℝ) :
    (m * x^2 - 2 * x - m ≥ 2) ↔ (x ≤ -1) := sorry

end problem_statement_l663_663547


namespace jessica_cut_r_l663_663367

variable (r_i r_t r_c : ℕ)

theorem jessica_cut_r : r_i = 7 → r_g = 59 → r_t = 20 → r_c = r_t - r_i → r_c = 13 :=
by
  intros h_i h_g h_t h_c
  have h1 : r_i = 7 := h_i
  have h2 : r_t = 20 := h_t
  have h3 : r_c = r_t - r_i := h_c
  have h_correct : r_c = 13
  · sorry
  exact h_correct

end jessica_cut_r_l663_663367


namespace find_c_l663_663199

namespace VectorProof

structure Vector2D (α : Type) :=
  (x : α)
  (y : α)

open Vector2D

def parallel {α : Type} [Field α] (v1 v2 : Vector2D α) : Prop :=
  ∃ λ : α, v2.x = λ * v1.x ∧ v2.y = λ * v1.y

def orthogonal {α : Type} [Field α] (v1 v2 : Vector2D α) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

def c := Vector2D.mk (-7/9 : ℚ) (-7/3 : ℚ)
def a := Vector2D.mk (1 : ℚ) (2 : ℚ)
def b := Vector2D.mk (2 : ℚ) (-3 : ℚ)
def a_plus_b := Vector2D.mk (3 : ℚ) (-1 : ℚ)

theorem find_c : 
  parallel {α := ℚ} (Vector2D.mk (c.x + a.x) (c.y + a.y)) b ∧ 
  orthogonal c a_plus_b := 
sorry

end VectorProof

end find_c_l663_663199


namespace wire_length_square_field_l663_663887

theorem wire_length_square_field : 
  ∀ (A : ℝ) (n : ℕ), A = 69696 → n = 15 → (∃ L, L = 15840) :=
by 
  intros A n hA hn 
  use 15840
  sorry

end wire_length_square_field_l663_663887


namespace coefficient_x4_expansion_l663_663337

theorem coefficient_x4_expansion (x : ℝ) : 
  (1 + real.sqrt x)^10.coefficient_of_x4 = 45 := 
sorry

end coefficient_x4_expansion_l663_663337


namespace division_remainder_l663_663144

noncomputable def polynomial_1 : Polynomial ℝ := X^4 - 2*X^2 + 3
noncomputable def polynomial_2 : Polynomial ℝ := X^2 - 4*X + 7
noncomputable def remainder : Polynomial ℝ := 28*X - 46

theorem division_remainder : polynomial_1 % polynomial_2 = remainder :=
by sorry

end division_remainder_l663_663144


namespace max_knights_between_knights_l663_663458

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l663_663458


namespace number_of_children_l663_663759

-- Definitions of given conditions
def total_passengers := 170
def men := 90
def women := men / 2
def adults := men + women
def children := total_passengers - adults

-- Theorem statement
theorem number_of_children : children = 35 :=
by
  sorry

end number_of_children_l663_663759


namespace original_number_contains_digit_5_or_greater_l663_663368

-- Definitions of the conditions
def no_zeros (n : ℕ) : Prop := ∀ d ∈ (to_digits 10 n), d ≠ 0

def rearranged_digits_sum_to_ones (orig : ℕ) (rearr1 rearr2 rearr3 : ℕ) : Prop :=
  let sum := orig + rearr1 + rearr2 + rearr3 in
  all_digits_are_ones sum

-- The property that we need to prove
theorem original_number_contains_digit_5_or_greater
  (orig rearr1 rearr2 rearr3 : ℕ)
  (hz : no_zeros orig)
  (hs : rearranged_digits_sum_to_ones orig rearr1 rearr2 rearr3) :
  ∃ d ∈ (to_digits 10 orig), d ≥ 5 := 
sorry

end original_number_contains_digit_5_or_greater_l663_663368


namespace max_knights_between_knights_l663_663471

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l663_663471


namespace find_max_value_l663_663540

theorem find_max_value : ∀ (x : ℝ), x ∈ (Set.Icc (-5 : ℝ) 13) → f x = x - 5 → f 13 = 8 :=
by
  intros x hx hf
  sorry

end find_max_value_l663_663540


namespace chef_pillsbury_flour_l663_663858

theorem chef_pillsbury_flour (x : ℕ) (h : 7 / 2 = 28 / x) : x = 8 := sorry

end chef_pillsbury_flour_l663_663858


namespace combined_area_of_three_rugs_l663_663006

theorem combined_area_of_three_rugs (floor_area : ℕ) (two_layers_area : ℕ) (three_layers_area : ℕ) (combined_area : ℕ) :
  floor_area = 140 →
  two_layers_area = 22 →
  three_layers_area = 19 →
  combined_area = 99 + 2 * two_layers_area + 3 * three_layers_area →
  combined_area = 200 := by
  intros ha hb hc hd
  rw [ha, hb, hc] at hd
  exact hd

end combined_area_of_three_rugs_l663_663006


namespace number_of_solutions_l663_663949

theorem number_of_solutions : 
  (∃ n : ℕ, n = (set_of (λ (p : ℤ × ℤ), abs p.1 + abs p.2 < 1000)).to_finite.to_finset.card) 
  → n = 1998001 :=
sorry

end number_of_solutions_l663_663949


namespace area_of_smallest_square_that_encloses_circle_l663_663024

def radius : ℕ := 5

def diameter (r : ℕ) : ℕ := 2 * r

def side_length (d : ℕ) : ℕ := d

def area_of_square (s : ℕ) : ℕ := s * s

theorem area_of_smallest_square_that_encloses_circle :
  area_of_square (side_length (diameter radius)) = 100 := by
  sorry

end area_of_smallest_square_that_encloses_circle_l663_663024


namespace rectangle_sides_l663_663147

def side_length_square : ℝ := 18
def num_rectangles : ℕ := 5

variable (a b : ℝ)
variables (h1 : 2 * (a + b) = side_length_square) (h2 : 3 * a = side_length_square)

theorem rectangle_sides : a = 6 ∧ b = 3 :=
by {
  sorry
}

end rectangle_sides_l663_663147


namespace area_of_given_triangle_is_25_l663_663882

noncomputable def area_of_triangle {R : Type*} [linear_ordered_field R] (A B C : euclidean_space R (fin 2))
  (x1 y1 x2 y2 x3 y3 : R)
  (A_def : A = ![x1, y1])
  (B_def : B = ![x2, y2])
  (C_def : C = ![x3, y3]) : R := do
    let v := C - A
    let w := C - B
    let area_par := abs (v 0 * w 1 - v 1 * w 0)
    area_par / 2

theorem area_of_given_triangle_is_25 :
  area_of_triangle ![5, -3] ![0, 2] ![-1, -7] 5 (-3) 0 2 (-1) (-7) rfl rfl rfl = 25 := by
  sorry

end area_of_given_triangle_is_25_l663_663882


namespace multiply_powers_same_base_l663_663102

theorem multiply_powers_same_base (a : ℝ) : a^3 * a = a^4 :=
by
  sorry

end multiply_powers_same_base_l663_663102


namespace total_cartridge_cost_l663_663011

theorem total_cartridge_cost:
  ∀ (bw_cartridge_cost color_cartridge_cost bw_quantity color_quantity : ℕ),
  bw_cartridge_cost = 27 →
  color_cartridge_cost = 32 →
  bw_quantity = 1 →
  color_quantity = 3 →
  bw_quantity * bw_cartridge_cost + color_quantity * color_cartridge_cost = 123 :=
begin
  intros bw_cartridge_cost color_cartridge_cost bw_quantity color_quantity,
  intros h_bw_cost h_color_cost h_bw_qty h_color_qty,
  rw [h_bw_cost, h_color_cost, h_bw_qty, h_color_qty],
  norm_num,
end

end total_cartridge_cost_l663_663011
