import Mathlib

namespace point_in_quadrant_IV_l109_109040

/-- In triangle ABC where angle A is obtuse, point P (tan B, cos A) is located in Quadrant IV. -/
theorem point_in_quadrant_IV
  (A B : ℝ)
  (hA_obtuse : π / 2 < A ∧ A < π)
  (h_sum_angles : A + B < π)
  (h_tan_B : tan B > 0)
  (h_cos_A : cos A < 0)
  : (tan B > 0 ∧ cos A < 0) := by
  sorry

end point_in_quadrant_IV_l109_109040


namespace max_area_equilateral_triangle_in_rectangle_l109_109217

theorem max_area_equilateral_triangle_in_rectangle (PQ PR : ℕ) (hPQ : PQ = 14) (hPR : PR = 13) : 
  ∃ a b c : ℕ, (a * Real.sqrt b - c) = (maximum_area PQ PR) ∧ ¬ ∃ p : ℕ, p * p ∣ b ∧ p > 1 ∧ a + b + c = 140 := 
sorry

end max_area_equilateral_triangle_in_rectangle_l109_109217


namespace find_B_and_b_l109_109386

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π ∧ 
(∀ (A B : ℝ), cos A / cos B = (2 * c - a) / b) 

theorem find_B_and_b (a b c : ℝ) (h1 : a + c = 3 * real.sqrt 3)
  (h2 : 1/2 * a * c * real.sin (π / 3) = 3 * real.sqrt 3 / 2)
  (h3 : triangle_ABC A B C a b c)
  (h4 : ∀ (A B : ℝ), cos A / cos B = (2 * c - a) / b) :
  B = π / 3 ∧ b = 3 :=
sorry

end find_B_and_b_l109_109386


namespace josephine_food_cost_l109_109046

variable (total_bill medication remaining_after_medication overnight_stays ambulance_ride food : ℕ)

-- Conditions
def condition_1 : total_bill = 5000 := sorry
def condition_2 : medication = (50 * total_bill) / 100 := sorry
def condition_3 : remaining_after_medication = total_bill - medication := sorry
def condition_4 : overnight_stays = (25 * remaining_after_medication) / 100 := sorry
def condition_5 : ambulance_ride = 1700 := sorry
def condition_6 : food = total_bill - medication - overnight_stays - ambulance_ride := sorry

theorem josephine_food_cost : food = 175 :=
by
  rw [condition_6, condition_5, condition_4, condition_3, condition_2, condition_1]
  sorry

end josephine_food_cost_l109_109046


namespace exists_zero_in_interval_l109_109345

noncomputable def f (x : ℝ) : ℝ := x^2 - 2^x

theorem exists_zero_in_interval : ∃ c ∈ set.Ioc (-1 : ℝ) (0 : ℝ), f c = 0 :=
begin
  have cont_f : continuous f := continuous.pow 2 (continuous_id) -- continuous of x^2 part
    .sub (continuous_rpow_const (2 : ℝ)), -- continuous of 2^x part
  have h1 : 0 < f (-1) := by norm_num,
  have h2 : f 0 < 0 := by norm_num,
  -- Using Intermediate Value Theorem
  exact intermediate_value_Ioc 0 (by norm_num) cont_f h1 h2,
end

end exists_zero_in_interval_l109_109345


namespace sin_pi_minus_alpha_l109_109680

noncomputable def tan_two_alpha (α : ℝ) := Real.tan (2 * α) = -4 / 3
noncomputable def is_acute_angle (α : ℝ) := 0 < α ∧ α < π / 2

theorem sin_pi_minus_alpha {α : ℝ} (h1 : tan_two_alpha α) (h2 : is_acute_angle α) :
  Real.sin (π - α) = (2 * Real.sqrt 5) / 5 :=
sorry

end sin_pi_minus_alpha_l109_109680


namespace tetrahedron_perpendicular_distances_inequalities_l109_109935

section Tetrahedron

variables {R : Type*} [LinearOrderedField R]

variables {S_A S_B S_C S_D V d_A d_B d_C d_D h_A h_B h_C h_D : R}

/-- Given areas and perpendicular distances of a tetrahedron, prove inequalities involving these parameters. -/
theorem tetrahedron_perpendicular_distances_inequalities 
  (h1 : S_A * d_A + S_B * d_B + S_C * d_C + S_D * d_D = 3 * V) : 
  (min h_A (min h_B (min h_C h_D)) ≤ d_A + d_B + d_C + d_D) ∧ 
  (d_A + d_B + d_C + d_D ≤ max h_A (max h_B (max h_C h_D))) ∧ 
  (d_A * d_B * d_C * d_D ≤ 81 * V ^ 4 / (256 * S_A * S_B * S_C * S_D)) :=
sorry

end Tetrahedron

end tetrahedron_perpendicular_distances_inequalities_l109_109935


namespace sin_function_monotone_l109_109920

def f (x : Real) : Real := sin (2 * x + π / 6)
def interval (t : Real) (h : 0 < t ∧ t < π / 6) : Set Real := {x | -t < x ∧ x < t}

theorem sin_function_monotone (t : Real) (h : 0 < t ∧ t < π / 6) : 
  ∀ x ∈ interval t h, is_monotone_on (f) (interval t h) :=
sorry

end sin_function_monotone_l109_109920


namespace ellipse_properties_l109_109325

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) 
  (ecc : a * 2 / 3 = b * sqrt 2)
  (area_triangle : 1 / 2 * b * (2 * sqrt (a^2 - b^2)) = 5 * sqrt 2 / 3) :
  (∃ x y : ℝ, x^2 / 5 + y^2 / (5 / 3) = 1) ∧
  (∀ k : ℝ, let A B : ℝ × ℝ := if h : ∃ A B : ℝ, (1 + 3 * k^2) * A + 6 * k^2 * B + 3 * k^2 - 5 = 0  
                                 then sorry 
                              else sorry,
  let M : ℝ × ℝ := (-7 / 3, 0) in
  ((M.1 + A.1) * (M.2 + B.2) = 4 / 9)) :=
sorry

end ellipse_properties_l109_109325


namespace pie_slices_total_l109_109219

theorem pie_slices_total :
  let apple_pie_lunch := 3
  let apple_pie_dinner := 1
  let apple_pie_yesterday := 8
  let blueberry_pie_lunch := 2
  let blueberry_pie_dinner := 2
  let blueberry_pie_yesterday := 8
  let cherry_pie_lunch := 2
  let cherry_pie_dinner := 1
  let pumpkin_pie_dinner := 1
  apple_pie_lunch + apple_pie_dinner + apple_pie_yesterday = 12 ∧
  blueberry_pie_lunch + blueberry_pie_dinner + blueberry_pie_yesterday = 12 ∧
  cherry_pie_lunch + cherry_pie_dinner = 3 ∧
  pumpkin_pie_dinner = 1 :=
by {
  let apple_pie_lunch := 3
  let apple_pie_dinner := 1
  let apple_pie_yesterday := 8
  let blueberry_pie_lunch := 2
  let blueberry_pie_dinner := 2
  let blueberry_pie_yesterday := 8
  let cherry_pie_lunch := 2
  let cherry_pie_dinner := 1
  let pumpkin_pie_dinner := 1
  have h1 : apple_pie_lunch + apple_pie_dinner + apple_pie_yesterday = 12 := by norm_num,
  have h2 : blueberry_pie_lunch + blueberry_pie_dinner + blueberry_pie_yesterday = 12 := by norm_num,
  have h3 : cherry_pie_lunch + cherry_pie_dinner = 3 := by norm_num,
  have h4 : pumpkin_pie_dinner = 1 := by norm_num,
  exact ⟨h1, h2, h3, h4⟩
}

end pie_slices_total_l109_109219


namespace subset_relation_l109_109366

theorem subset_relation (M N : set ℕ) (hM : M = {1, 2, 3, 4, 5}) (hN : N = {1, 4}) : N ⊆ M :=
by {
  sorry
}

end subset_relation_l109_109366


namespace possible_values_of_a_l109_109697

open BigOperators

noncomputable def valid_values (a : ℝ) (x : Fin 6 → ℝ) : Prop :=
  (∑ i in Finset.range 1 6, (i : ℝ) * x i = a) ∧
  (∑ i in Finset.range 1 6, (i : ℝ)^3 * x i = a^2) ∧
  (∑ i in Finset.range 1 6, (i : ℝ)^5 * x i = a^3)

theorem possible_values_of_a (a : ℝ) (x : Fin 6 → ℝ) :
  valid_values a x → a ∈ ({0, 1, 4, 9, 16, 25} : Set ℝ) :=
sorry

end possible_values_of_a_l109_109697


namespace find_area_of_tangency_quadrilateral_l109_109585

def radius : ℝ := 1
def trapezoid_area : ℝ := 5
def quadrilateral_area : ℝ := 1.6

-- Define the conditions: isosceles trapezoid circumscribed around a circle
def is_isosceles_trapezoid_circumscribed (r : ℝ) (area : ℝ) : Prop :=
  ∃ (ABCD : ℝ × ℝ × ℝ × ℝ), 
    -- Here, properties of the trapezoid ABCD would be expressed appropriately
    -- E.g., sides and angles fitting the definition of isosceles trapezoid circumscribed around the circle
    -- with given radius and area.

-- Define the question as a proof problem
theorem find_area_of_tangency_quadrilateral
  (h : is_isosceles_trapezoid_circumscribed radius trapezoid_area) :
  quadrilateral_area = 1.6 := 
  sorry

end find_area_of_tangency_quadrilateral_l109_109585


namespace degree_g_l109_109467

noncomputable def degree (p : Polynomial ℚ) : ℕ := p.degree.getOrElse 0

theorem degree_g {a b c d : ℚ} (g : Polynomial ℚ)
  (h : Polynomial ℚ)
  (g_deg : ℕ)
  (h_deg : ℕ)
  (f_deg : ℕ)
  (h_def : h = (Polynomial.C a) * g^3 + (Polynomial.C b) * g^2 + (Polynomial.C c) * g + Polynomial.C d + g)
  (h_deg_eq : degree h = 6)
  (f_deg_eq : f_deg = 3) :
  degree g = 2 := 
  sorry

end degree_g_l109_109467


namespace bisection_method_interval_characteristic_l109_109899

theorem bisection_method_interval_characteristic {a b : ℝ} (f : ℝ → ℝ) : ∃ (a b : ℝ), a < b ∧ f(a) * f(b) < 0 := sorry

end bisection_method_interval_characteristic_l109_109899


namespace lucy_picked_more_l109_109442

variable (Mary Peter Lucy : ℕ)
variable (Mary_amt Peter_amt Lucy_amt : ℕ)

-- Conditions
def mary_amount : Mary_amt = 12 := sorry
def twice_as_peter : Mary_amt = 2 * Peter_amt := sorry
def total_picked : Mary_amt + Peter_amt + Lucy_amt = 26 := sorry

-- Statement to Prove
theorem lucy_picked_more (h1: Mary_amt = 12) (h2: Mary_amt = 2 * Peter_amt) (h3: Mary_amt + Peter_amt + Lucy_amt = 26) :
  Lucy_amt - Peter_amt = 2 := 
sorry

end lucy_picked_more_l109_109442


namespace bernardo_larger_than_silvia_l109_109241

theorem bernardo_larger_than_silvia :
  let bernardo_candidates := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
      silvia_candidates := {1, 2, 3, 4, 5, 6, 7}
      bernardo_possibilities := choose 10 3
      silvia_possibilities := choose 7 3
      favorable_cases := 25 in
  (favorable_cases : ℚ) / (bernardo_possibilities : ℚ) = 5 / 12 :=
sorry

end bernardo_larger_than_silvia_l109_109241


namespace value_of_a2022_l109_109033

noncomputable def sequence_a : ℕ → ℚ
| 0       := -(1/4 : ℚ)  -- Lean uses zero-based indexing, so a₀ corresponds to a₁.
| (n + 1) := 1 - 1 / sequence_a n

theorem value_of_a2022 : sequence_a 2021 = 4 / 5 := sorry

end value_of_a2022_l109_109033


namespace mode_median_of_data_set_l109_109858

def data_set : list ℕ := [6, 8, 3, 6, 4, 6, 5]

def mode (l : list ℕ) : ℕ :=
l.foldr (λ x (y : option ℕ) => 
  match y with
  | some y' => if l.count x > l.count y' then some x else some y'
  | none => some x
  ) none

def median (l : list ℕ) : ℕ :=
let sorted_l := l.qsort (≤) in
sorted_l.nth (sorted_l.length / 2) |>.get_or_else 0

theorem mode_median_of_data_set :
  mode data_set = 6 ∧ median data_set = 6 := by
sorry

end mode_median_of_data_set_l109_109858


namespace find_n_l109_109275

theorem find_n (n : ℕ) (h : n ≥ 2) :
  let x_n := ∑ i in Finset.range n, i * (i + 1)
  let y_n := ∑ i in Finset.range n, i^2
  4 * x_n + 2 * y_n = 20 * n^2 + 13 * n - 33 ↔ n = 11 := 
by
  let x_n := ∑ i in Finset.range n, i * (i + 1)
  let y_n := ∑ i in Finset.range n, i^2
  sorry

end find_n_l109_109275


namespace largest_four_digit_number_divisible_by_5_l109_109514

theorem largest_four_digit_number_divisible_by_5 : ∃ n : ℕ, n < 10000 ∧ n % 5 = 0 ∧ ∀ m : ℕ, m < 10000 ∧ m % 5 = 0 → m ≤ n :=
begin
  use 9995,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m h1 h2,
    have : m ≤ 9999 := h1,
    nlinarith,
    sorry
  }
end

end largest_four_digit_number_divisible_by_5_l109_109514


namespace radian_measure_15_deg_l109_109914

theorem radian_measure_15_deg (Degrees : ℝ) (Radians : ℝ) :
  Degrees = 15 → Radians = (Real.pi / 180) * Degrees → Radians = Real.pi / 12 := by
  intros h1 h2
  rw [h1] at h2
  rw [← h2]
  norm_num
  have : (Real.pi / 180) * 15 = Real.pi / 12 := by
    field_simp
    rw [mul_comm]
  rw [this]
  rfl
  sorry -- proof would go here 

end radian_measure_15_deg_l109_109914


namespace pizza_toppings_l109_109571

theorem pizza_toppings (n : ℕ) (h : n = 7) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 63 :=
by
  rw h
  simp
  sorry -- Placeholder to complete the proof

end pizza_toppings_l109_109571


namespace zinc_to_copper_ratio_l109_109926

noncomputable def ratio_zinc_copper (total_weight zinc_weight : ℝ) : ℕ × ℕ :=
  let copper_weight := total_weight - zinc_weight
  let g := Int.gcd (Int.ofReal (10 * zinc_weight)) (Int.ofReal (10 * copper_weight))
  ((Int.ofReal (10 * zinc_weight) / g).natAbs, (Int.ofReal (10 * copper_weight) / g).natAbs)

theorem zinc_to_copper_ratio (total_weight : ℝ) (zinc_weight : ℝ) (h1 : total_weight = 70) (h2 : zinc_weight = 31.5) :
  ratio_zinc_copper total_weight zinc_weight = (9, 11) :=
by
  rw [h1, h2]
  sorry

end zinc_to_copper_ratio_l109_109926


namespace max_tan_X_in_triangle_l109_109412

theorem max_tan_X_in_triangle (X Y Z : Type) (XY YZ : ℝ)
  (h1 : XY = 26) (h2 : YZ = 18) (h3 : ∀ (X Y Z : Type), Triangle X Y Z ) :
  ∃ (tanX : ℝ), tanX = (9 * Real.sqrt 22) / 44 :=
by sorry

end max_tan_X_in_triangle_l109_109412


namespace total_students_in_class_l109_109736

theorem total_students_in_class
  (S : ℕ)
  (H1 : 5/8 * S = S - 60)
  (H2 : 60 = 3/8 * S) :
  S = 160 :=
by
  sorry

end total_students_in_class_l109_109736


namespace find_possible_values_l109_109675

noncomputable def abs_sign (x : ℝ) : ℝ := x / |x|

theorem find_possible_values (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) :
  ∃ (v : ℝ), v ∈ {-5, -2, 1, 4, 5} ∧ 
  v = abs_sign a + abs_sign b + abs_sign c + abs_sign d + abs_sign (a * b * c * d) :=
by
  sorry

end find_possible_values_l109_109675


namespace hyperbola_slope_range_l109_109872

theorem hyperbola_slope_range (x y : ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ) (slope : ℝ) : 
  (x^2 - y^2 = 1) → 
  F = (-sqrt(2), 0) → 
  P = (u, v) ∧ (v < 0) ∧ (u^2 - v^2 = 1) → 
  (slope = (v - 0) / (u - (-sqrt(2)))) → 
  slope ∈ (-∞, 0) ∪ (1, ∞) :=
sorry

end hyperbola_slope_range_l109_109872


namespace binary_101_to_decimal_is_5_l109_109611

-- Define a function to convert binary to decimal
def binary_to_decimal (b : list ℕ) : ℕ :=
  b.reverse.enum_from 0
    .foldr (λ ⟨i, n⟩ acc, acc + n * (2 ^ i)) 0

-- Now, we state the proof problem
theorem binary_101_to_decimal_is_5 : binary_to_decimal [1, 0, 1] = 5 := by
  sorry

end binary_101_to_decimal_is_5_l109_109611


namespace at_least_one_high_degree_l109_109048

noncomputable def degree (p : polynomial ℝ) : ℕ := polynomial.natDegree p

-- Conditions
variables {f g : polynomial ℝ} {n : ℕ}

-- Main theorem statement
theorem at_least_one_high_degree (h₁ : n ≥ 3) 
  (h₂ : ∀ k ∈ finset.range n, is_vertex (f.eval (k+1), g.eval (k+1)) 
  (regular_polygon_vertices n)) : 
  max (degree f) (degree g) ≥ n - 1 := 
sorry

-- Placeholder for the definition of is_vertex and regular_polygon_vertices
def is_vertex (p : ℝ × ℝ) (s : set (ℝ × ℝ)) : Prop := sorry
def regular_polygon_vertices (n : ℕ) : set (ℝ × ℝ) := sorry

end at_least_one_high_degree_l109_109048


namespace meeting_point_l109_109451

theorem meeting_point :
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  (Paul_start.1 + Lisa_start.1) / 2 = -2 ∧ (Paul_start.2 + Lisa_start.2) / 2 = 3 :=
by
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  have x_coord : (Paul_start.1 + Lisa_start.1) / 2 = -2 := sorry
  have y_coord : (Paul_start.2 + Lisa_start.2) / 2 = 3 := sorry
  exact ⟨x_coord, y_coord⟩

end meeting_point_l109_109451


namespace impossible_disjoint_rational_irrational_unions_l109_109944

-- Define the geometrical structures involved
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Segment :=
  (A : Point)
  (B : Point)

-- Definition of the letter T at a point on the x-axis
structure LetterT :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

-- Define the condition to erect a letter T
def erect_T_at (x : ℝ) : LetterT :=
  let A := {x := x, y := 0}
  let B := {x := x, y := 1}  -- arbitrary unit height for the vertical segment
  let C := {x := x - 0.5, y := 0.5}
  let D := {x := x + 0.5, y := 0.5}
  { A := A, B := B, C := C, D := D }

-- The main conjecture
theorem impossible_disjoint_rational_irrational_unions :
  ¬ (∀ x ∈ ℝ, erect_T_at (x : ℚ) ∩ erect_T_at (x ∈ ℝ \ ℚ) = ∅) :=
begin
  sorry
end

end impossible_disjoint_rational_irrational_unions_l109_109944


namespace order_of_expressions_l109_109199

variable (a : ℝ)

theorem order_of_expressions (h : a > 1) : log (0.2) a < 0.2 * a ∧ 0.2 * a < a ^ 0.2 :=
by
sorry

end order_of_expressions_l109_109199


namespace part1_part2_part3_l109_109483

noncomputable def sequence_a : ℕ → ℝ
| 0       := 1/2
| (n + 1) := sequence_a n / (1 + sequence_a n)^2

noncomputable def sequence_b (n : ℕ) : ℝ := 1 / sequence_a n

theorem part1 (n : ℕ) (hn : 1 < n) : sequence_b n > 2 * n := sorry

theorem part2 : tendsto (λ n, (1 : ℝ) / n * ∑ i in finset.range n, sequence_a (i + 1)) at_top (𝓝 0) := sorry

theorem part3 : tendsto (λ n, n * sequence_a n) at_top (𝓝 (1 / 2)) := sorry

end part1_part2_part3_l109_109483


namespace domain_of_ln_over_x_minus_3_l109_109473

noncomputable def domain_of_function (x : ℝ) : Set ℝ :=
  {x | x > -1 ∧ x ≠ 3}

theorem domain_of_ln_over_x_minus_3 :
  (domain_of_function = {x | (x > -1 ∧ x < 3) ∨ (x > 3)}) :=
by
  sorry

end domain_of_ln_over_x_minus_3_l109_109473


namespace mass_percentage_Al_in_Al2S3_proof_l109_109636

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_S : ℝ := 32.06
noncomputable def moles_Al_in_Al2S3 : ℕ := 2
noncomputable def moles_S_in_Al2S3 : ℕ := 3

def total_mass_Al : ℝ :=
  (moles_Al_in_Al2S3 : ℝ) * molar_mass_Al

def total_mass_S : ℝ :=
  (moles_S_in_Al2S3 : ℝ) * molar_mass_S

def molar_mass_Al2S3 : ℝ :=
  total_mass_Al + total_mass_S

def mass_percentage_Al_in_Al2S3 : ℝ :=
  (total_mass_Al / molar_mass_Al2S3) * 100

theorem mass_percentage_Al_in_Al2S3_proof :
  mass_percentage_Al_in_Al2S3 = 35.94 :=
by
  have h1 : total_mass_Al = 2 * 26.98 := rfl
  have h2 : total_mass_S = 3 * 32.06 := rfl
  have h3 : molar_mass_Al2S3 = (2 * 26.98) + (3 * 32.06) := by
    simp [total_mass_Al, total_mass_S]
  have h4 : mass_percentage_Al_in_Al2S3 = ((2 * 26.98) / ((2 * 26.98) + (3 * 32.06))) * 100 := by
    simp [total_mass_Al, molar_mass_Al2S3]
  have : ((2 * 26.98) / ((2 * 26.98) + (3 * 32.06))) * 100 = 35.94 := by
    norm_num
  exact this

end mass_percentage_Al_in_Al2S3_proof_l109_109636


namespace integer_root_of_polynomial_l109_109254

theorem integer_root_of_polynomial :
  ∀ (d e : ℚ), (∃ x : ℝ, x^3 + (d : ℝ) * x + (e : ℝ) = 0 ∧ x = 2 - real.sqrt 5) →
  ∃ (r : ℤ), 
  (∃ (z : ℝ), z^3 + (d : ℝ) * z + (e : ℝ) = 0 ∧ z = 2 + real.sqrt 5) 
  ∧ (2 - real.sqrt 5) + (2 + real.sqrt 5) + (r : ℝ) = 0 ∧ r = -4 :=
by
  intros
  sorry

end integer_root_of_polynomial_l109_109254


namespace new_population_difference_l109_109222

def population_eagles : ℕ := 150
def population_falcons : ℕ := 200
def population_hawks : ℕ := 320
def population_owls : ℕ := 270
def increase_rate : ℕ := 10

theorem new_population_difference :
  let least_populous := min population_eagles (min population_falcons (min population_hawks population_owls))
  let most_populous := max population_eagles (max population_falcons (max population_hawks population_owls))
  let increased_least_populous := least_populous + least_populous * increase_rate / 100
  most_populous - increased_least_populous = 155 :=
by
  sorry

end new_population_difference_l109_109222


namespace stratified_sampling_BA3_count_l109_109206

-- Defining the problem parameters
def num_Om_BA1 : ℕ := 60
def num_Om_BA2 : ℕ := 20
def num_Om_BA3 : ℕ := 40
def total_sample_size : ℕ := 30

-- Proving using stratified sampling
theorem stratified_sampling_BA3_count : 
  (total_sample_size * num_Om_BA3 / (num_Om_BA1 + num_Om_BA2 + num_Om_BA3)) = 10 :=
by
  -- Since Lean doesn't handle reals and integers simplistically,
  -- we need to translate the division and multiplication properly.
  sorry

end stratified_sampling_BA3_count_l109_109206


namespace b_1008_equals_2017_l109_109340

-- Define the sequence a_n and its properties
def seq_a (n : ℕ) : ℕ → ℝ := λ n, ite (n > 0) (4 ^ (n - 1) * 8 : ℝ) 0

-- Define the sequence b_n as the log base 2 of a_n
def seq_b (n : ℕ) : ℕ → ℝ := λ n, log (seq_a n n) / log 2

-- State the theorem that b_{1008} = 2017
theorem b_1008_equals_2017 : seq_b 1008 1008 = 2017 := by
  sorry

end b_1008_equals_2017_l109_109340


namespace sequence_max_length_l109_109628

theorem sequence_max_length (x : ℕ) :
  (2000 - 2 * x > 0) ∧ (3 * x - 2000 > 0) ∧ (4000 - 5 * x > 0) ∧ 
  (8 * x - 6000 > 0) ∧ (10000 - 13 * x > 0) ∧ (21 * x - 16000 > 0) → x = 762 :=
by
  sorry

end sequence_max_length_l109_109628


namespace problem_solution_l109_109602

noncomputable def a (n : ℕ) : ℕ := ((10^n) - 1) / 9

def repeat_digit_1 (n : ℕ) : ℕ := a n

def repeat_digit_4 (n : ℕ) : ℕ := 4 * repeat_digit_1 (2 * n)

def repeat_digit_1_n_plus_1 (n : ℕ) : ℕ := 10 * repeat_digit_1 n + 1

def repeat_digit_6 (n : ℕ) : ℕ := 6 * repeat_digit_1 n

def final_expression (n : ℕ) : ℕ := 
  (repeat_digit_4 n / (2 * n)) + repeat_digit_1_n_plus_1 n - repeat_digit_6 n

theorem problem_solution (n : ℕ) :
  sqrt (final_expression n) = (6 * repeat_digit_1 n) + 1 := by
  sorry

end problem_solution_l109_109602


namespace sphere_surface_area_l109_109889

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (S : ℝ)
  (hV : V = 36 * π)
  (hvol : V = (4 / 3) * π * r^3) :
  S = 4 * π * r^2 :=
by
  sorry

end sphere_surface_area_l109_109889


namespace cindy_marbles_l109_109835

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end cindy_marbles_l109_109835


namespace root_rational_poly_l109_109667

theorem root_rational_poly {c d : ℚ} (h : (3 + Real.sqrt 5) ∈ (λ x, x^3 + c * x^2 + d * x + 15).roots) : 
  d = -18.5 :=
sorry

end root_rational_poly_l109_109667


namespace johns_total_earning_this_year_l109_109044

-- Define the conditions given in the problem
def last_year_salary : ℝ := 100_000
def last_year_bonus : ℝ := 10_000
def current_year_salary : ℝ := 200_000
def bonus_percentage : ℝ := last_year_bonus / last_year_salary

-- The statement we aim to prove
theorem johns_total_earning_this_year :
  current_year_salary + current_year_salary * bonus_percentage = 220_000 :=
by
  sorry

end johns_total_earning_this_year_l109_109044


namespace verify_other_root_l109_109351

variable {a b c x : ℝ}

-- Given conditions
axiom distinct_non_zero_constants : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

axiom root_two : a * 2^2 - (a + b + c) * 2 + (b + c) = 0

-- Function under test
noncomputable def other_root (a b c : ℝ) : ℝ :=
  (b + c - a) / a

-- The goal statement
theorem verify_other_root :
  ∀ (a b c : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) → (a * 2^2 - (a + b + c) * 2 + (b + c) = 0) → 
  (∀ x, (a * x^2 - (a + b + c) * x + (b + c) = 0) → (x = 2 ∨ x = (b + c - a) / a)) :=
by
  intros a b c h1 h2 x h3
  sorry

end verify_other_root_l109_109351


namespace log_base_3_of_243_l109_109268

theorem log_base_3_of_243 : ∃ x : ℤ, (∃ y : ℤ, y = 243 ∧ 3^5 = y) ∧ 3^x = 243 ∧ log 3 243 = 5 := by
  -- Introducing the given conditions
  let y := 243
  have h1: 3^5 = y := by sorry -- Known condition in the problem
  -- Prove the main statement
  use 5
  split
  . use y
    exact ⟨rfl, h1⟩
  . split
    . exact h1
    . sorry

end log_base_3_of_243_l109_109268


namespace min_a_value_l109_109860

noncomputable def choose (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
else 0

def favorable_outcomes (a : ℕ) : ℕ :=
(choose (a - 1) 2) + (choose (43 - a) 2)

def p (a : ℕ) : ℚ :=
(favorable_outcomes a : ℚ) / 1225

def p_min_value (a : ℕ) : Prop :=
p(a) ≥ 1/2

theorem min_a_value (a : ℕ) (ha: a = 8) : p_min_value a := 
by 
  rw [ha]
  sorry

end min_a_value_l109_109860


namespace systematic_sample_interval_l109_109947

theorem systematic_sample_interval (
  num_workers : ℕ := 840
  num_sampled : ℕ := 42
  sample_interval : ℕ := 20
  start_number : ℕ := 21
  interval_start : ℕ := 421
  interval_end : ℕ := 720
) : (interval_end - interval_start + 1) / sample_interval = 15 :=
by
  -- Definitions and conditions can be skipped as per instruction
  sorry

end systematic_sample_interval_l109_109947


namespace first_term_of_geometric_series_l109_109974

theorem first_term_of_geometric_series (r a S : ℝ) (h_r : r = 1 / 4) (h_S : S = 40) 
  (h_geometric_sum : S = a / (1 - r)) : a = 30 :=
by
  -- The proof would go here, but we place a sorry to skip the proof.
  sorry

end first_term_of_geometric_series_l109_109974


namespace triangle_is_isosceles_l109_109729

structure Triangle (α : Type) :=
(A B C : α)

structure PointOnExtension (α : Type) :=
(D E : α)

variables {P : PointOnExtension ℝ}
variables {T : Triangle ℝ}
variables {k : ℝ}

def are_equal_length (BD CE : ℝ) : Prop :=
BD = CE

def angle_relation (P : PointOnExtension ℝ) (angleAEP angleADP anglePED anglePDE k: ℝ) : Prop :=
angleAEP - angleADP = k^2 * (anglePED - anglePDE)

theorem triangle_is_isosceles (T : Triangle ℝ) (D E : ℝ)
  (h1 : are_equal_length D E)
  (h2 : ∃ k, ∀ P, angle_relation P T.A T.B T.C D E k) : 
  T.AB = T.AC :=
by sorry

end triangle_is_isosceles_l109_109729


namespace problem1_problem2_l109_109246

-- Definitions for Problem 1
def sin_45_eq : Real := Real.sqrt 2 / 2
def tan_60_eq : Real := Real.sqrt 3
def tan_30_eq : Real := Real.sqrt 3 / 3
def cos_60_eq : Real := 1 / 2

theorem problem1 :
  2 * (sin_45_eq ^ 2) + tan_60_eq * tan_30_eq - cos_60_eq = 3 / 2 := by
  sorry

-- Definitions for Problem 2
def sqrt_12_eq : Real := 2 * Real.sqrt 3
def cos_30_eq : Real := Real.sqrt 3 / 2
def three_minus_pi_power_zero : Real := 1
def abs_1_minus_sqrt_3 : Real := Real.abs (Real.sqrt 3 - 1)

theorem problem2 :
  sqrt_12_eq - 2 * cos_30_eq + three_minus_pi_power_zero + abs_1_minus_sqrt_3 = 2 * Real.sqrt 3 := by
  sorry

end problem1_problem2_l109_109246


namespace necessary_but_not_sufficient_l109_109719

theorem necessary_but_not_sufficient (A B : Prop) (h : A → B) : ¬ (B → A) :=
sorry

end necessary_but_not_sufficient_l109_109719


namespace count_valid_n_l109_109293

theorem count_valid_n :
  {n : ℕ | (∀ t : ℝ, (Complex.sin t - Complex.I * Complex.cos t) ^ n = Complex.sin (n * t) - Complex.I * Complex.cos (n * t)) ∧ n ≤ 500}.to_finset.card = 125 := by
sorry

end count_valid_n_l109_109293


namespace exists_finite_set_with_neighbors_l109_109460

-- The main statement: proving there exists a finite set A
theorem exists_finite_set_with_neighbors :
  ∃ (A : Set (ℝ × ℝ)), Set.Finite A ∧
  ∀ (X ∈ A), ∃ (Ys : Fin 1993 → (ℝ × ℝ)),
  (∀ i, Ys i ∈ A ∧ dist X (Ys i) = 1) :=
sorry

end exists_finite_set_with_neighbors_l109_109460


namespace min_equilateral_triangles_l109_109912

theorem min_equilateral_triangles (s : ℝ) (S : ℝ) :
  s = 1 → S = 15 → 
  225 = (S / s) ^ 2 :=
by
  intros hs hS
  rw [hs, hS]
  simp
  sorry

end min_equilateral_triangles_l109_109912


namespace remainder_problem_l109_109165

theorem remainder_problem :
  ((98 * 103 + 7) % 12) = 1 :=
by
  sorry

end remainder_problem_l109_109165


namespace avg_speed_first_5_hours_is_30_l109_109556

-- Given conditions as definitions
def avgSpeedFirst5Hours (S : ℝ) (first5Hours : ℕ) : ℝ := S
def avgSpeedAdditionalHours : ℝ := 42
def avgSpeedEntireTrip : ℝ := 38
def totalTripHours : ℕ := 15
def remainingHours : ℕ := totalTripHours - 5
def totalDistanceCalculated : ℝ := avgSpeedEntireTrip * totalTripHours

-- Prove that the average speed for the first 5 hours is 30 miles per hour
theorem avg_speed_first_5_hours_is_30 (S : ℝ) :
  5.S + remainingHours * avgSpeedAdditionalHours = totalDistanceCalculated → S = 30 :=
by
  sorry

end avg_speed_first_5_hours_is_30_l109_109556


namespace tan_difference_l109_109335

theorem tan_difference (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h₁ : Real.sin α = 3 / 5) (h₂ : Real.cos β = 12 / 13) : 
    Real.tan (α - β) = 16 / 63 := 
by
  sorry

end tan_difference_l109_109335


namespace min_value_of_2a_minus_ab_l109_109083

theorem min_value_of_2a_minus_ab :
  ∃ (a : ℕ) (b : ℕ), (0 < a ∧ a < 10) ∧ (0 < b ∧ b < 10) ∧ 2 * a - a * b = -63 :=
sorry

end min_value_of_2a_minus_ab_l109_109083


namespace equation_has_root_implies_a_eq_neg6_l109_109299

theorem equation_has_root_implies_a_eq_neg6 
  (a x : ℝ) 
  (h1 : (x - 2)/(x + 4) = a/(x + 4))
  (h2 : x = -4) : 
  a = -6 := 
begin 
  sorry 
end

end equation_has_root_implies_a_eq_neg6_l109_109299


namespace taylor_series_expansion_l109_109635

theorem taylor_series_expansion (y : ℝ → ℝ) (y' : ℝ → ℝ)
  (h₀ : y 0 = 1)
  (h₁ : y' 0 = 0)
  (h₂ : ∀ x, (deriv^[2] y x) = real.exp (x * y x)) :
  ∃ c₀ c₂ c₃ c₄ : ℝ,
  (y 0 = c₀) ∧
  (y' 0 = 0) ∧
  (∀ x, y x = c₀ + c₂*x^2 + c₃*x^3 + c₄*x^4 + O(x^5)) := by
  sorry

end taylor_series_expansion_l109_109635


namespace determine_angle_AQC_l109_109906

-- Define the basic setup of the problem
structure Circle (α : Type u) :=
(center : α)
(radius : ℝ)

variables {α : Type u} [metric_space α] [normed_group α] [normed_space ℝ α]

def is_perpendicular (a b : α) : Prop := pseudoangle a b = real.pi / 2

def circle_tangent_point (O P Q : α) (c1 c2 : Circle α) : Prop :=
  dist O Q = c1.radius ∧ dist P Q = c2.radius

-- Define points and circles
variables (O A B Q C D : α)
variable (c : Circle α)
variable (smaller_circle : Circle α)

-- The given conditions in the problem
variables (h1 : is_perpendicular (A - O) (B - O))
variables (h2 : circle_tangent_point O Q Q c smaller_circle)
variables (h3 : circle_tangent_point A O C c smaller_circle)
variables (h4 : circle_tangent_point B O D c smaller_circle)

-- Lean statement for the required proof
theorem determine_angle_AQC :
  ∃ θ : ℝ, θ = 45 ∨ θ = 90 :=
sorry

end determine_angle_AQC_l109_109906


namespace heaviest_person_is_Vanya_l109_109978

variables (A D T V M : ℕ)

-- conditions
def condition1 : Prop := A + D = 82
def condition2 : Prop := D + T = 74
def condition3 : Prop := T + V = 75
def condition4 : Prop := V + M = 65
def condition5 : Prop := M + A = 62

theorem heaviest_person_is_Vanya (h1 : condition1 A D) (h2 : condition2 D T) (h3 : condition3 T V) (h4 : condition4 V M) (h5 : condition5 M A) :
  V = 43 :=
sorry

end heaviest_person_is_Vanya_l109_109978


namespace seating_arrangement_l109_109624

theorem seating_arrangement (x y : ℕ) (h1 : 9 * x + 7 * y = 61) : x = 6 :=
by 
  sorry

end seating_arrangement_l109_109624


namespace projection_magnitude_l109_109338

variables {a b : ℝ} {theta : ℝ}

def len_a : ℝ := 3
def len_b : ℝ := 1
def angle_ab : ℝ := π / 6

theorem projection_magnitude : len_b * Real.cos angle_ab = Real.sqrt 3 / 2 :=
by
  sorry

end projection_magnitude_l109_109338


namespace length_YW_l109_109752

-- Definitions of the sides of the triangle
def XY := 6
def YZ := 8
def XZ := 10

-- The total perimeter of triangle XYZ
def perimeter : ℕ := XY + YZ + XZ

-- Each ant travels half the perimeter
def halfPerimeter : ℕ := perimeter / 2

-- Distance one ant travels from X to W through Y
def distanceXtoW : ℕ := XY + 6

-- Prove that the distance segment YW is 6
theorem length_YW : distanceXtoW = halfPerimeter := by sorry

end length_YW_l109_109752


namespace ratio_of_areas_l109_109008

theorem ratio_of_areas (s2 : ℝ) (h : 0 < s2) : 
  ((s2 * sqrt 2) ^ 2) / (s2 ^ 2) = 2 := 
by
  -- Proof not required, so we use sorry to skip it
  sorry

end ratio_of_areas_l109_109008


namespace positive_distinct_solutions_conditons_l109_109440

-- Definitions corresponding to the conditions in the problem
variables {x y z a b : ℝ}

-- The statement articulates the condition
theorem positive_distinct_solutions_conditons (h1 : x + y + z = a) (h2 : x^2 + y^2 + z^2 = b^2) (h3 : xy = z^2) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x ≠ y) (h8 : y ≠ z) (h9 : x ≠ z) : 
  b^2 ≥ a^2 / 2 :=
sorry

end positive_distinct_solutions_conditons_l109_109440


namespace positive_result_l109_109924

theorem positive_result (A B C D : ℤ) (hA : A = 0 * (-2019) ^ 2018)
  (hB : B = (-3) ^ 2) (hC : C = -2 / (-3) ^ 4) (hD : D = (-2) ^ 3) :
  B > 0 :=
by {
  have hA_eval : A = 0, by rw [hA, mul_zero],
  have hB_eval : B = 9, by rw [hB, pow_two, mul_neg_one_pow_two.symm],
  have hC_eval : C = -2 / 81, by rw [hC, pow_four, neg_div_eq_mul_inv],
  have hD_eval : D = -8, by rw [hD, neg_pow],
  rw hB_eval,
  norm_num,
}

end positive_result_l109_109924


namespace exist_c_l109_109865

noncomputable theory

variable {f g : ℝ → ℝ}

-- Conditions
axiom h1 : ∫ x in 0..1, (f x) ^ 2 = 1
axiom h2 : ∫ x in 0..1, (g x) ^ 2 = 1

theorem exist_c (h1 : ∫ x in 0..1, (f x) ^ 2 = 1) (h2 : ∫ x in 0..1, (g x) ^ 2 = 1) :
  ∃ c ∈ (set.Icc 0 1), f c + g c ≤ 2 := 
sorry

end exist_c_l109_109865


namespace convex_quad_sum_greater_diff_l109_109084

theorem convex_quad_sum_greater_diff (α β γ δ : ℝ) 
    (h_sum : α + β + γ + δ = 360) 
    (h_convex : α < 180 ∧ β < 180 ∧ γ < 180 ∧ δ < 180) :
    ∀ (x y z w : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) → (y = α ∨ y = β ∨ y = γ ∨ y = δ) → 
                     (z = α ∨ z = β ∨ z = γ ∨ z = δ) → (w = α ∨ w = β ∨ w = γ ∨ w = δ) 
                     → x + y > |z - w| := 
by
  sorry

end convex_quad_sum_greater_diff_l109_109084


namespace max_abs_diff_l109_109662

noncomputable theory
open_locale real

-- Define the conditions
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1
def C2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 9
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

-- Define the points M, N on circles C1, C2 and P on the x-axis.
variables (M N P : ℝ × ℝ)
variables (hM : C1 M.1 M.2) (hN : C2 N.1 N.2) (hP : on_x_axis P)

-- Prove the maximum value of |PN| - |PM| is 9.
theorem max_abs_diff : 
  ∃ M N P, C1 M.1 M.2 ∧ C2 N.1 N.2 ∧ on_x_axis P ∧ (dist P N - dist P M) = 9 :=
sorry

end max_abs_diff_l109_109662


namespace sum_solutions_eq_26_l109_109768

theorem sum_solutions_eq_26:
  (∃ (n : ℕ) (solutions: Fin n → (ℝ × ℝ)),
    (∀ i, let (x, y) := solutions i in |x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|)
    ∧ (Finset.univ.sum (λ i, let (x, y) := solutions i in x + y) = 26))
:= sorry

end sum_solutions_eq_26_l109_109768


namespace average_score_median_score_percentage_excellent_students_l109_109032

def scores : list ℕ := [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 
                        59, 59, 59, 59, 59, 
                        58, 58, 58, 58, 58, 58, 58, 
                        57, 57, 57, 57, 57, 
                        56, 56, 
                        54]

def total_students : ℕ := 30

def excellent_threshold : ℕ := 59

-- Average score calculation
theorem average_score : (60 * 10 + 59 * 5 + 58 * 7 + 57 * 5 + 56 * 2 + 54 * 1) / total_students = 58.4 :=
sorry

-- Median score calculation
theorem median_score : 
  let sorted_scores := list.sort (≤) scores in
  (list.nth sorted_scores 14).iget = 58 ∧ (list.nth sorted_scores 15).iget = 59 ∧
  (list.nth sorted_scores 14).iget + (list.nth sorted_scores 15).iget = 117 / 2 = 58.5 :=
sorry

-- Percentage of students with excellent grades
theorem percentage_excellent_students :
  let excellent_students := (list.count (λ x, x >= 59) scores) in
  (excellent_students * 100) / total_students = 50 :=
sorry

end average_score_median_score_percentage_excellent_students_l109_109032


namespace average_burning_time_probability_l109_109450

variables (N : ℕ) (n : ℕ) (Xbar : ℝ) (Xtilde : ℝ) (var: ℝ)

def sample_properties 
  (hN : N = 5000) 
  (hn : n = 300) 
  (hXtilde : Xtilde = 1450) 
  (hvar : var = 40000) : Prop :=
  let sem := real.sqrt (var / n * (1 - n / N)) in
  P (1410 < Xbar ∧ Xbar < 1490) = 0.99964

theorem average_burning_time_probability (hN : N = 5000) 
  (hn : n = 300) 
  (hXtilde : Xtilde = 1450) 
  (hvar : var = 40000) :
  sample_properties N n Xbar Xtilde var :=
sorry

end average_burning_time_probability_l109_109450


namespace number_of_siblings_l109_109092

theorem number_of_siblings : ∃ (S : ℕ), 1_000_000 / 2 - (1_000_000 / 2 / S) = 375_000 ∧ 1_000_000 / 2 = (S + 1) * (1_000_000 / 2 / (S + 1)) :=
by
  sorry

end number_of_siblings_l109_109092


namespace no_integer_solution_for_equation_l109_109463

theorem no_integer_solution_for_equation :
    ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = x * y * z - 1 :=
by
  sorry

end no_integer_solution_for_equation_l109_109463


namespace widgets_per_week_l109_109758

theorem widgets_per_week 
  (widgets_per_hour : ℕ) 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (h1 : widgets_per_hour = 20) 
  (h2 : hours_per_day = 8) 
  (h3 : days_per_week = 5) :
  widgets_per_hour * hours_per_day * days_per_week = 800 :=
by
  rw [h1, h2, h3]
  exact rfl

end widgets_per_week_l109_109758


namespace number_of_exchanges_l109_109183

theorem number_of_exchanges (n : ℕ) (hz_initial : ℕ) (hl_initial : ℕ) 
  (hz_decrease : ℕ) (hl_decrease : ℕ) (k : ℕ) :
  hz_initial = 200 →
  hl_initial = 20 →
  hz_decrease = 6 →
  hl_decrease = 1 →
  k = 11 →
  (hz_initial - n * hz_decrease) = k * (hl_initial - n * hl_decrease) →
  n = 4 := 
sorry

end number_of_exchanges_l109_109183


namespace problem1_problem2_l109_109988

-- Problem 1
theorem problem1 :
  sqrt 48 - sqrt 27 + sqrt (1/3) = (4 * sqrt 3) / 3 :=
by
  sorry

-- Problem 2
theorem problem2 :
  (sqrt 5 - sqrt 2) * (sqrt 5 + sqrt 2) - (sqrt 3 - 1)^2 = 2 * sqrt 3 - 1 :=
by
  sorry

end problem1_problem2_l109_109988


namespace angle_between_p_q_l109_109814

open Real

variables (p q r : ℝ^3)

-- Conditions: p, q, r are unit vectors
axiom hp : ∥p∥ = 1
axiom hq : ∥q∥ = 1
axiom hr : ∥r∥ = 1

-- Condition: p + 2q + sqrt(2) r = 0
axiom h : p + 2 • q + sqrt 2 • r = 0

-- Constant to represent the desired angle
noncomputable def desired_angle := real.arccos (-3 / 4)

-- Proof statement
theorem angle_between_p_q : 
  (real.arccos ((p.dot q) / (∥p∥ * ∥q∥))) = desired_angle := 
sorry

end angle_between_p_q_l109_109814


namespace number_of_yellow_marbles_l109_109402

theorem number_of_yellow_marbles (Y : ℕ) (h : Y / (7 + 11 + Y) = 1 / 4) : Y = 6 :=
by
  -- Proof to be filled in
  sorry

end number_of_yellow_marbles_l109_109402


namespace number_of_frog_sequences_l109_109952

def frog_move_sequences : ℕ → ℕ
| 0 := 1
| n := if n >= 3 then frog_move_sequences (n - 3) else 0 +
       if n >= 13 then frog_move_sequences (n - 13) else 0

theorem number_of_frog_sequences (n : ℕ) :
  frog_move_sequences 39 = 169 :=
sorry

end number_of_frog_sequences_l109_109952


namespace volume_calculation_l109_109995

-- Define the dimensions of the rectangular parallelepiped
def length : ℝ := 2
def width  : ℝ := 3
def height : ℝ := 6

-- Define the radius of the quarter-cylinders and eighth-spheres
def radius : ℝ := 1

-- Total volume calculation
def total_volume : ℝ := 
  let V_box := length * width * height
  let V_out := 2 * (radius * length * width + radius * length * height + radius * width * height)
  let V_cyl := 22 * real.pi
  let V_sph := (4 / 3) * real.pi
  V_box + V_out + V_cyl + V_sph

-- Expected correct total volume
def correct_volume : ℝ := (324 + 70 * real.pi) / 3

-- Proof problem statement
theorem volume_calculation : total_volume = correct_volume :=
by
  -- The proof would go here, but we skip it with sorry.
  sorry

end volume_calculation_l109_109995


namespace largest_among_options_l109_109529

theorem largest_among_options :
  let A := real.sqrt (real.cbrt 56)
  let B := real.sqrt (real.cbrt 3584)
  let C := real.sqrt (real.cbrt 2744)
  let D := real.cbrt (real.sqrt 448)
  let E := real.cbrt (real.sqrt 392)
  B > A ∧ B > C ∧ B > D ∧ B > E := 
  begin
    sorry
  end

end largest_among_options_l109_109529


namespace largest_four_digit_divisible_by_5_l109_109511

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

theorem largest_four_digit_divisible_by_5 : 
  ∃ n : ℕ, n ≤ 9999 ∧ is_divisible_by_5 n ∧ ∀ m : ℕ, m ≤ 9999 → is_divisible_by_5 m → m ≤ n :=
begin
  use 9995,
  split,
  { linarith },
  split,
  { right, norm_num },
  { intros m hm hdiv,
    by_cases h : m = 9995,
    { rw h },
    have : m % 10 ≠ 0 ∧ m % 10 ≠ 5,
    { intro h,
      cases h,
      { rw [h, nat.sub_self, zero_mod] at hdiv, linarith },
      { norm_num at h } },
    have : m < 9995 := sorry,
    linarith }
end

end largest_four_digit_divisible_by_5_l109_109511


namespace gcf_45_75_90_l109_109162

-- Definitions as conditions
def number1 : Nat := 45
def number2 : Nat := 75
def number3 : Nat := 90

def factors_45 : Nat × Nat := (3, 2) -- represents 3^2 * 5^1 {prime factor 3, prime factor 5}
def factors_75 : Nat × Nat := (5, 1) -- represents 3^1 * 5^2 {prime factor 3, prime factor 5}
def factors_90 : Nat × Nat := (3, 2) -- represents 2^1 * 3^2 * 5^1 {prime factor 3, prime factor 5}

-- Theorems to be proved
theorem gcf_45_75_90 : Nat.gcd (Nat.gcd number1 number2) number3 = 15 :=
by {
  -- This is here as placeholder for actual proof
  sorry
}

end gcf_45_75_90_l109_109162


namespace trig_identity_l109_109599

theorem trig_identity :
  (Real.cos (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (48 * Real.pi / 180) * Real.sin (18 * Real.pi / 180)) = 1 / 2 := 
by sorry

end trig_identity_l109_109599


namespace inequality_a_c_b_l109_109615

-- Define the main conditions
variables (f : ℝ → ℝ)
hypothesis (h_odd : ∀ x, f (-x) = -f x)
hypothesis (h_diff : ∀ x < 0, f x + x * (deriv f x) < 0)

noncomputable def a := 3 * f 3
noncomputable def b := (log π 3) * f (log π 3)
noncomputable def c := -2 * f (-2)

theorem inequality_a_c_b : a f > c f > b f :=
sorry

end inequality_a_c_b_l109_109615


namespace game_cost_l109_109530

theorem game_cost
    (total_earnings : ℕ)
    (expenses : ℕ)
    (games_bought : ℕ)
    (remaining_money := total_earnings - expenses)
    (cost_per_game := remaining_money / games_bought)
    (h1 : total_earnings = 104)
    (h2 : expenses = 41)
    (h3 : games_bought = 7) :
    cost_per_game = 9 := by
  sorry

end game_cost_l109_109530


namespace domain_of_f_l109_109866

def f (x : ℝ) : ℝ := sqrt (x - 1) + log (x + 1)

theorem domain_of_f : {x : ℝ | 1 ≤ x} = {x : ℝ | 0 ≤ x - 1} ∩ {x : ℝ | -1 < x} :=
by
  sorry

end domain_of_f_l109_109866


namespace intersection_points_squared_distance_l109_109862

open Real

theorem intersection_points_squared_distance :
  let center1 := (3 : ℝ, -2 : ℝ)
  let radius1 := 5 : ℝ
  let center2 := (3 : ℝ, 4 : ℝ)
  let radius2 := sqrt 17 : ℝ
  ∀ (A B : ℝ × ℝ), 
    (x - 3)^2 + (y + 2)^2 = 25
    ∧ (x - 3)^2 + (y - 4)^2 = 17 →
    (AB.1 - center1.1) = 2 * sqrt 26 / 3 →
    (AB.2 - center1.2) = 0 →
    (AB)^2 = 416 / 9 :=
sorry

end intersection_points_squared_distance_l109_109862


namespace monotonicity_of_f_value_set_of_m_l109_109701

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 + Real.log x + 1

theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 0 → ∀ x1 x2, 0 < x1 → x1 < x2 → f a x1 < f a x2) ∧
    (a < 0 → (∀ x1 x2, 0 < x1 → x1 < sqrt (-1 / (2 * a)) → x1 < x2 → f a x1 < f a x2) ∧
            (∀ x1 x2, sqrt (-1 / (2 * a)) < x1 → x1 < x2 → f a x1 > f a x2)) := sorry

theorem value_set_of_m (a m : ℝ) (h_a : -2 < a ∧ a < -1) :
    (∀ x ∈ set.Icc 1 2, m * a - f a x > a ^ 2) → m ≤ -3 / 2 := sorry

end monotonicity_of_f_value_set_of_m_l109_109701


namespace find_possible_values_l109_109674

noncomputable def abs_sign (x : ℝ) : ℝ := x / |x|

theorem find_possible_values (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) :
  ∃ (v : ℝ), v ∈ {-5, -2, 1, 4, 5} ∧ 
  v = abs_sign a + abs_sign b + abs_sign c + abs_sign d + abs_sign (a * b * c * d) :=
by
  sorry

end find_possible_values_l109_109674


namespace diesel_fuel_cost_l109_109546

def cost_per_liter (total_cost : ℝ) (num_liters : ℝ) : ℝ := total_cost / num_liters

def full_tank_cost (cost_per_l : ℝ) (tank_capacity : ℝ) : ℝ := cost_per_l * tank_capacity

theorem diesel_fuel_cost (total_cost : ℝ) (num_liters : ℝ) (tank_capacity : ℝ) :
  total_cost = 18 → num_liters = 36 → tank_capacity = 64 → full_tank_cost (cost_per_liter total_cost num_liters) tank_capacity = 32 :=
by
  intros h_total h_num h_tank
  rw [h_total, h_num, h_tank]
  norm_num
  sorry -- Full proof can be completed with detailed steps.

end diesel_fuel_cost_l109_109546


namespace find_number_l109_109213

theorem find_number (x : ℤ) (h : x = 5 * (x - 4)) : x = 5 :=
by {
  sorry
}

end find_number_l109_109213


namespace otimes_calculation_l109_109124

def otimes (a b : ℝ) : ℝ := (a^3) / (b^2)

theorem otimes_calculation : 
  ((otimes (otimes 2 4) (otimes 1 3)) - (otimes 2 (otimes 4 3))) = (1215 / 512) := 
by 
  sorry

end otimes_calculation_l109_109124


namespace mod_remainder_l109_109809

theorem mod_remainder (b : ℤ) (h : b ≡ ((2⁻¹ * 2 + 3⁻¹ * 3 + 5⁻¹ * 5)⁻¹ : ℤ) [MOD 13]) : 
  b ≡ 6 [MOD 13] := 
  sorry

end mod_remainder_l109_109809


namespace colbert_planks_needed_to_buy_l109_109250

variables (total_planks : ℕ) (planks_from_storage : ℕ) 
          (planks_from_parents : ℕ) (planks_from_friends : ℕ)

def planks_needed_from_store := 
  total_planks - (planks_from_storage + planks_from_parents + planks_from_friends)

theorem colbert_planks_needed_to_buy : 
  total_planks = 200 → planks_from_storage = total_planks / 4 → 
  planks_from_parents = total_planks / 2 → planks_from_friends = 20 → 
  planks_needed_from_store total_planks planks_from_storage planks_from_parents planks_from_friends = 30 :=
by
  -- proof steps here
  sorry

end colbert_planks_needed_to_buy_l109_109250


namespace count_three_digit_integers_mod_8_l109_109363

theorem count_three_digit_integers_mod_8 : 
  (∃ n, ∀ x, (100 ≤ x ∧ x < 1000 ∧ x % 8 = 3) ↔ x = 8 * n + 3) ∧
  (n | 13 ≤ n ∧ n ≤ 124) ∧
  (112 = 124 - 13 + 1) :=
by
  sorry

end count_three_digit_integers_mod_8_l109_109363


namespace minimum_area_triangle_AJ1J2_minimum_l109_109437

-- Conditions
variables (A B C Y J1 J2 : Point)
terms
  (AB AC BC : ℝ) (J1_J2_incenters : J1 = incenter (triangle A B Y) ∧ J2 = incenter (triangle A C Y))

  -- Given side lengths
  (sqrt_length: AB = 13 ∧ AC = 15 ∧ BC = 14)
  
  -- Proof Problem
theorem minimum_area_triangle_AJ1J2_minimum 
  (h: ∀ Y on interior (line_segment B C), Y ∈ interior (line_segment B C))
  (incenter_1: J1 = incenter (triangle A B Y))
  (incenter_2: J2 = incenter (triangle A C Y))
  : area (triangle A J1 J2) = 3.375 := 
sorry

end minimum_area_triangle_AJ1J2_minimum_l109_109437


namespace sufficient_but_not_necessary_condition_l109_109198

theorem sufficient_but_not_necessary_condition (h1 : 1^2 - 1 = 0) (h2 : ∀ x, x^2 - 1 = 0 → (x = 1 ∨ x = -1)) :
  (∀ x, x = 1 → x^2 - 1 = 0) ∧ ¬ (∀ x, x^2 - 1 = 0 → x = 1) := by
  sorry

end sufficient_but_not_necessary_condition_l109_109198


namespace rats_meet_on_fourth_day_l109_109399

theorem rats_meet_on_fourth_day :
  ∃ n : ℕ, 
    (n > 0) → 
    let larger_rat_burrow := (finset.range n).sum (λ x, 2^x)
    let smaller_rat_burrow := (finset.range n).sum (λ x, (1 / 2)^x)
    (larger_rat_burrow + smaller_rat_burrow = 10) ∧ (n = 4) :=
by
  sorry

end rats_meet_on_fourth_day_l109_109399


namespace max_expr_value_l109_109435

theorem max_expr_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x^2 + y^2 + z^2 = 1) :
  3 * x * y * real.sqrt 7 + 9 * y * z ≤ (1 / 2) * real.sqrt 88 :=
sorry

end max_expr_value_l109_109435


namespace speed_for_remaining_distance_l109_109983

theorem speed_for_remaining_distance (D T : ℝ) (h1 : 70 = (2 * D / 3) / (T / 3)) :
  let S2 := (D / 3) / (2 * T / 3) in
  S2 = 17.5 :=
by
  -- proof can be filled in here
  sorry

end speed_for_remaining_distance_l109_109983


namespace cosine_inequality_l109_109066

theorem cosine_inequality
  (x y z : ℝ)
  (hx : 0 < x ∧ x < π / 2)
  (hy : 0 < y ∧ y < π / 2)
  (hz : 0 < z ∧ z < π / 2) :
  (x * Real.cos x + y * Real.cos y + z * Real.cos z) / (x + y + z) ≤
  (Real.cos x + Real.cos y + Real.cos z) / 3 := sorry

end cosine_inequality_l109_109066


namespace largest_sphere_radius_on_torus_l109_109225

theorem largest_sphere_radius_on_torus :
  ∀ r : ℝ, 16 + (r - 1)^2 = (r + 2)^2 → r = 13 / 6 :=
by
  intro r
  intro h
  sorry

end largest_sphere_radius_on_torus_l109_109225


namespace interval_exists_solution_f_eq_e_l109_109334

-- Define the function f and its properties
variables {f : ℝ → ℝ}

-- Conditions:
-- f is monotonically increasing
axiom f_monotone : ∀ x y : ℝ, x < y → f x ≤ f y

-- ∀ x ∈ (0, +∞), f(f(x) - ln(x)) = e + 1
axiom f_property : ∀ x : ℝ, 0 < x → f(f(x) - log x) = Real.exp 1 + 1

-- Goal: Determine the interval where the solution to f(x) - f''(x) = e exists.
theorem interval_exists_solution_f_eq_e : ∀ x : ℝ, 1 < x ∧ x < 2 → f x - deriv 2 f x = Real.exp 1 :=
sorry

end interval_exists_solution_f_eq_e_l109_109334


namespace sin_cos_identity_count_sin_cos_identity_l109_109291

theorem sin_cos_identity (n : ℕ) (h₁ : n ≤ 500) :
  (∀ t : ℝ, (sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)) ↔
  (∃ k : ℕ, n = 4 * k + 1) :=
sorry

theorem count_sin_cos_identity :
  ∃ m : ℕ, m = 125 ∧ ∀ n : ℕ, n ≤ 500 → (∀ t : ℝ, (sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)) ↔
  n = 4 * (n / 4) + 1 :=
sorry

end sin_cos_identity_count_sin_cos_identity_l109_109291


namespace monotone_increasing_condition_l109_109923

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotone_increasing_condition (t : ℝ) (h₁ : 0 < t) (h₂ : t < Real.pi / 6) :
  ∀ x y, x ∈ Ioo (-t) t → y ∈ Ioo (-t) t → x ≤ y → f x ≤ f y :=
sorry

end monotone_increasing_condition_l109_109923


namespace rhombus_area_l109_109846

-- Definitions from conditions
def is_rhombus (A B C D : Type) :=
  sorry -- Assuming there is some definition of rhombus

def perimeter (A B C D : Type) : ℕ :=
  sorry -- Assuming there is some definition for perimeter

def diagonal_AC_length (A B C D : Type) : ℕ :=
  sorry -- Assuming there is some definition for diagonal length AC

-- Theorem statement
theorem rhombus_area {A B C D : Type} (h_rhombus: is_rhombus A B C D) (h_perimeter: perimeter A B C D = 52) 
  (h_diagonal_AC: diagonal_AC_length A B C D = 24): 
  area A B C D = 120 :=
sorry

end rhombus_area_l109_109846


namespace four_times_remaining_marbles_l109_109836

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end four_times_remaining_marbles_l109_109836


namespace probability_at_least_one_head_l109_109071

theorem probability_at_least_one_head : 
  (1 - (1 / 2) * (1 / 2) * (1 / 2) = 7 / 8) :=
by
  sorry

end probability_at_least_one_head_l109_109071


namespace determine_y_l109_109067

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0)
variable (hxy : x = 2 + (1 / y))
variable (hyx : y = 2 + (2 / x))

theorem determine_y (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 2 + (1 / y)) (hyx : y = 2 + (2 / x)) :
  y = (5 + Real.sqrt 41) / 4 ∨ y = (5 - Real.sqrt 41) / 4 := 
sorry

end determine_y_l109_109067


namespace factorable_quadratic_even_b_l109_109114

theorem factorable_quadratic_even_b :
  ∃ c d e f : ℤ, c * e = 15 ∧ d * f = 45 ∧ 15x^2 + (c * f + d * e) * x + 45 = (c * x + d) * (e * x + f) → ∃ k : ℤ, b = 2 * k :=
sorry

end factorable_quadratic_even_b_l109_109114


namespace possible_values_expression_l109_109673

theorem possible_values_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (val : ℤ), val ∈ {5, 2, 1, -2, -3} ∧ 
  val = (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (a * b * c * d / |a * b * c * d|) :=
begin
  sorry
end

end possible_values_expression_l109_109673


namespace collinearity_A_P_Q_l109_109584

-- Definition of the triangle and excircles
variables {A B C C1 B1 A1 B2 C2 A2 P Q : Type}

-- Conditions for the problem setup
def is_excircle_triangle (A B C C1 B1 A1 B2 C2 A2 : Type) : Prop :=
  -- Conditions to define excircles and touch points
  -- Assume we have a way to define excircles and touching points by introducing appropriate axiom or definitions
  sorry

def intersection_points (A1 B1 A2 B2 A1 C1 A2 C2 P Q : Type) : Prop :=
  -- Conditions to define intersections at points P and Q
  sorry

-- The problem statement: Given the above conditions, prove collinearity
theorem collinearity_A_P_Q 
  (h_excircle : is_excircle_triangle A B C C1 B1 A1 B2 C2 A2)
  (h_intersec : intersection_points A1 B1 A2 B2 A1 C1 A2 C2 P Q) :
  are_collinear A P Q :=
sorry

end collinearity_A_P_Q_l109_109584


namespace limit_dist_cauchy_l109_109761

noncomputable def U : ℝ → ℝ := sorry -- A definition for the uniformly distributed random variable on [0,1]

def S_n (n : ℕ) : ℝ :=
  2 * (∑ k in range (1, n+1), sin (2 * k * U * π))

theorem limit_dist_cauchy :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, dist (S_n n) (cauchy_dist) < ε := sorry

end limit_dist_cauchy_l109_109761


namespace range_f_l109_109168

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem range_f : ∀ y, (∃ x : ℝ, f x = y) ↔ y ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end range_f_l109_109168


namespace sum_of_solutions_eq_46_l109_109772

theorem sum_of_solutions_eq_46 (x y : ℤ) (sols : List (ℤ × ℤ)) :
  (∀ (xi yi : ℤ), (xi, yi) ∈ sols →
    (|xi - 4| = |yi - 10| ∧ |xi - 10| = 3 * |yi - 4|)) →
  (sols = [(10, 4), (5, -1), (10, 4), (-5, 19)]) →
  List.sum (sols.map (λ p, p.1 + p.2)) = 46 :=
by
  intro h1 h2
  rw [h2]
  dsimp
  norm_num

end sum_of_solutions_eq_46_l109_109772


namespace double_addition_arithmetic_correct_l109_109035

/-
  We need to define: 
  - Variables for each letter, ensuring each represents a unique digit.
  - Conditions ensuring that different letters map to different digits.
  - A proof goal that checks the arithmetic result of the given equations.
-/

variable (N A G Y F O K I L E : ℕ)
variable (digit_unique : ∀ x y : ℕ, (x ≠ y) → (N ≠ A ∧ N ≠ G ∧ N ≠ Y ∧ N ≠ F ∧ N ≠ O ∧ N ≠ K ∧ N ≠ I ∧ N ≠ L ∧ N ≠ E ∧ ...) :=
  -- continue for A, G, Y, F, O, K, I, L, E ensuring all pairs are unique
sorry -- Placeholder for all uniqueness constraints

-- Setting specific constraints based on the solution
variable (F_nonzero : F ≠ 0)
variable (F_value : F = 1)
variable (O_value : O = 0)
variable (N_value : N = 9)
variable (K_value : K = 8)
variable (Y_value : Y = 7)
variable (A_value : A = 5)
variable (G_value : G = 6)
variable (I_value : I = 2)
variable (L_value : L = 3)
variable (E_value : E = 4)

-- Goal: Verify the arithmetic equivalence
theorem double_addition_arithmetic_correct :
  N * 1000 + A * 100 + G * 10 + Y +
  F * 1000 + O * 100 + K * 10 + A =
  F * 10000 + O * 1000 + G * 100 + A * 10 + I + 
  (E * 10 + L) + 
  F * 10000 + O * 1000 + G * 100 + N * 10 + A :=
by {
  -- Numerical checks
  let lhs := 9567 + 1085
  let rhs := 10652 + 43
  exact lhs = rhs,
}

end double_addition_arithmetic_correct_l109_109035


namespace determine_m_l109_109378

noncomputable def root_and_modulus (z : ℂ) (m : ℝ) : Prop := 
  (z^2 - 2*z + m = 0) ∧ (|conj z| = √2)

theorem determine_m {z : ℂ} {m : ℝ} (h : root_and_modulus z m) : m = 2 :=
by sorry

end determine_m_l109_109378


namespace train_length_correct_l109_109227

def train_speed_kmh : ℝ := 90
def bus_speed_kmh : ℝ := 60
def passing_time_sec : ℝ := 5.279577633789296

def relative_speed_kmh := train_speed_kmh + bus_speed_kmh
def relative_speed_ms := relative_speed_kmh * (5 / 18)

theorem train_length_correct : 
  (relative_speed_ms * passing_time_sec) = 41.663147 := 
by 
  sorry

end train_length_correct_l109_109227


namespace AE_bisects_DB_l109_109436

/-- 
Prove that if \( ABC \) is an isosceles triangle with \( B \) as the vertex where the 
two equal sides meet, the tangents to the circumcircle \( \Gamma \) of \( ABC \) at points 
\( A \) and \( B \) intersect at point \( D \), and \( E \) is the second point of intersection
of line \( DC \) with \( \Gamma \), then line \( AE \) bisects segment \( DB \). 
--/
theorem AE_bisects_DB
  {A B C D E : Point}
  (h_isosceles : isosceles_triangle ABC B)
  (h_tangent : tangents_meet_at A B Γ D)
  (h_intersect : second_inter_point DC Γ E) :
  bisects AE DB :=
  sorry

end AE_bisects_DB_l109_109436


namespace BE_eq_BC_l109_109971

-- Define the geometric entities and their relationships
variables {A B C D E O1 O2 : Point}
variables {k1 k2 : Circle}

-- Assume the initial conditions based on the problem statement
-- 1. C is an arbitrary point on the semicircle with diameter AB
axiom semicircle_def : Semicircle k1 A B
axiom point_on_semicircle : OnSemicircle C k1

-- 2. D is the projection of C onto the diameter AB
axiom D_projection : Projection C AB D

-- 3. A circle is inscribed into the shape bordered by the arc AC, and segments CD and DA
axiom inscribed_circle : InscribedCircle k2 (Arc AC) CD DA

-- 4. The inscribed circle touches the segment AD at point E
axiom inscribed_circle_touch : Touches k2 AD E

-- The goal is to show that BE equals BC
theorem BE_eq_BC : SegmentLength B E = SegmentLength B C :=
begin
  sorry
end

end BE_eq_BC_l109_109971


namespace necessary_condition_not_sufficient_condition_l109_109481

variable (a b : ℝ)
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0
def proposition_p (a : ℝ) : Prop := a = 0

theorem necessary_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : isPureImaginary z → proposition_p a := sorry

theorem not_sufficient_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : proposition_p a → ¬isPureImaginary z := sorry

end necessary_condition_not_sufficient_condition_l109_109481


namespace length_of_chord_on_circle_l109_109348

noncomputable def parametric_line : ℝ → ℝ × ℝ :=
λ t, (4 * t - 1, 3 * t)

noncomputable def circle_eq (x y : ℝ) : Prop :=
x^2 + y^2 - 4 * y = 0

theorem length_of_chord_on_circle : 
  ∀ (l : ℝ → ℝ × ℝ) (C : ℝ → ℝ → Prop), 
  (∀ t, l t = (4 * t - 1, 3 * t)) →
  (∀ x y, C x y ↔ x^2 + y^2 - 4 * y = 0) →
  let chord_length := 
    if h : ∃ t1 t2, 25 * t1^2 - 20 * t1 + 1 = 0 ∧ 25 * t2^2 - 20 * t2 + 1 = 0 then
      5 * (abs (classical.some h - classical.some (Exists.intro (classical.some h).snd _)))
    else 0
  in chord_length = 2 * √3 :=
by { 
  intros l C hl hC chord_length,
  sorry
}

end length_of_chord_on_circle_l109_109348


namespace sequence_a_2017_minus_2016_l109_109439

noncomputable def sequence_a : ℕ → ℤ 
| 1 := 20
| 2 := 17
| n := 3 * sequence_a (n-1) - 2 * sequence_a (n-2)

theorem sequence_a_2017_minus_2016 : 
  (sequence_a 2017 - sequence_a 2016) = -3 * 2^2015 := 
sorry

end sequence_a_2017_minus_2016_l109_109439


namespace percent_university_diploma_no_job_choice_l109_109396

theorem percent_university_diploma_no_job_choice
    (total_people : ℕ)
    (P1 : 10 * total_people / 100 = total_people / 10)
    (P2 : 20 * total_people / 100 = total_people / 5)
    (P3 : 30 * total_people / 100 = 3 * total_people / 10) :
  25 = (20 * total_people / (80 * total_people / 100)) :=
by
  sorry

end percent_university_diploma_no_job_choice_l109_109396


namespace sum_solutions_eq_26_l109_109767

theorem sum_solutions_eq_26:
  (∃ (n : ℕ) (solutions: Fin n → (ℝ × ℝ)),
    (∀ i, let (x, y) := solutions i in |x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|)
    ∧ (Finset.univ.sum (λ i, let (x, y) := solutions i in x + y) = 26))
:= sorry

end sum_solutions_eq_26_l109_109767


namespace sum_of_distinct_terms_if_and_only_if_l109_109799

theorem sum_of_distinct_terms_if_and_only_if
  (a : ℕ → ℕ)
  (h_decreasing : ∀ k, a k > a (k + 1))
  (S : ℕ → ℕ)
  (h_S : ∀ k, S k = (∑ i in finset.range (k - 1), a i))
  (h_S1 : S 1 = 0) :
  (∀ n : ℕ, ∃ (l : list ℕ), (∀ (i : ℕ), i ∈ l → ∃ k, i = a k) ∧  l.sum = n) ↔ (∀ k, a k ≤ S k + 1) :=
begin
  sorry,
end

end sum_of_distinct_terms_if_and_only_if_l109_109799


namespace largest_four_digit_divisible_by_5_l109_109513

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

theorem largest_four_digit_divisible_by_5 : 
  ∃ n : ℕ, n ≤ 9999 ∧ is_divisible_by_5 n ∧ ∀ m : ℕ, m ≤ 9999 → is_divisible_by_5 m → m ≤ n :=
begin
  use 9995,
  split,
  { linarith },
  split,
  { right, norm_num },
  { intros m hm hdiv,
    by_cases h : m = 9995,
    { rw h },
    have : m % 10 ≠ 0 ∧ m % 10 ≠ 5,
    { intro h,
      cases h,
      { rw [h, nat.sub_self, zero_mod] at hdiv, linarith },
      { norm_num at h } },
    have : m < 9995 := sorry,
    linarith }
end

end largest_four_digit_divisible_by_5_l109_109513


namespace standard_deviation_upper_bound_l109_109888

theorem standard_deviation_upper_bound (Mean StdDev : ℝ) (h : Mean = 54) (h2 : 54 - 3 * StdDev > 47) : StdDev < 2.33 :=
by
  sorry

end standard_deviation_upper_bound_l109_109888


namespace lines_perpendicular_to_same_plane_are_parallel_l109_109329

-- Definitions used directly from the conditions in a)
variables {α β γ : Plane}
variables {m n : Line}

-- Lean 4 statement of the equivalent proof problem
theorem lines_perpendicular_to_same_plane_are_parallel
  (h1 : m ⊥ α)
  (h2 : n ⊥ α) :
  m ∥ n :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l109_109329


namespace lunks_needed_for_20_apples_l109_109374

-- Based on the conditions provided:
def lunks_for_kunks : ℕ := 4
def kunks_for_lunks : ℕ := 2
def kunks_for_apples : ℕ := 3
def apples_for_kunks : ℕ := 5

-- Number of lunks needed to purchase 20 apples
def required_lunks (apples : ℕ) : ℕ :=
  let kunks := (kunks_for_apples * apples) / apples_for_kunks
  in (lunks_for_kunks * kunks) / kunks_for_lunks

-- Given the problem statement, we assert:
theorem lunks_needed_for_20_apples : required_lunks 20 = 24 :=
sorry

end lunks_needed_for_20_apples_l109_109374


namespace compare_abc_l109_109370

theorem compare_abc (a b c : ℝ) (h1 : a = -3 * real.sqrt (3^2)) (h2 : b = -| -real.sqrt 2 |) (h3 : c = -real.cbrt ((-2 : ℝ)^3)) : c > b ∧ b > a := 
by
  sorry

end compare_abc_l109_109370


namespace sum_of_solutions_l109_109766

theorem sum_of_solutions :
  ∀ (x y : ℤ), (|x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|) →
  ({ (2, 8), (4, 10), (10, 4), (10, 4) }).sum (λ p, p.1 + p.2) = 52 :=
by
  sorry

end sum_of_solutions_l109_109766


namespace unit_disk_inequality_l109_109050

noncomputable def maxMod (f : ℂ → ℂ) (r : ℝ) := ⨆ z : {z // complex.abs z = r}, complex.abs (f (z : ℂ))

theorem unit_disk_inequality (D : set ℂ) (hD : is_open D)
    (h_unit_disk : ∀ z, complex.abs z ≤ 1 → z ∈ D)
    (f : ℂ → ℂ) (hf : ∀ z ∈ D, differentiable_at ℂ f z)
    (p : polynomial ℂ) (hp_monic : p.monic) :
  complex.abs (f 0) ≤ maxMod (λ z, f z * polynomial.eval z p) 1 :=
sorry

end unit_disk_inequality_l109_109050


namespace negation_of_exists_is_forall_l109_109876

theorem negation_of_exists_is_forall :
  (¬ ∃ x : ℝ, x^3 + 1 = 0) ↔ ∀ x : ℝ, x^3 + 1 ≠ 0 :=
by 
  sorry

end negation_of_exists_is_forall_l109_109876


namespace negation_of_proposition_p_is_false_l109_109688

variable (p : Prop)

theorem negation_of_proposition_p_is_false
  (h : ¬p) : ¬(¬p) :=
by
  sorry

end negation_of_proposition_p_is_false_l109_109688


namespace sum_two_smallest_prime_factors_l109_109524

theorem sum_two_smallest_prime_factors (n : ℕ) (h : n = 462) : 
  (2 + 3) = 5 := 
by {
  sorry
}

end sum_two_smallest_prime_factors_l109_109524


namespace andrew_age_proof_l109_109238

def andrew_age_problem : Prop :=
  ∃ (a g : ℚ), g = 15 * a ∧ g - a = 60 ∧ a = 30 / 7

theorem andrew_age_proof : andrew_age_problem :=
by
  sorry

end andrew_age_proof_l109_109238


namespace sum_divisible_by_n_l109_109792

theorem sum_divisible_by_n (a : ℕ → ℕ) (n : ℕ) (hpos : 0 < n) :
  ∃ k l : ℕ, 0 ≤ k ∧ k < l ∧ l ≤ n ∧ (∑ i in (finset.Icc (k + 1) l), a i) % n = 0 :=
sorry

end sum_divisible_by_n_l109_109792


namespace decompose_vectors_of_triangle_l109_109427

namespace Geometry

variables {V : Type} [AddCommGroup V] [VectorSpace ℝ V]

def is_centroid (O A B C : V) : Prop :=
  let M1 := (B + C) / 2 in
  O = (A + B + C) / 3 ∧ 
  ((A - O) = 2/3 * (A - M1))

theorem decompose_vectors_of_triangle
  {A B C O : V}
  {a b : V}
  (h_centroid : is_centroid O A B C)
  (h_AO : A - O = a)
  (h_AC : A - C = -b) :
  (A - B = 3 * a - b) ∧ 
  (B - C = 2 * b - 3 * a) :=
sorry

end Geometry

end decompose_vectors_of_triangle_l109_109427


namespace speed_in_still_water_l109_109190

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 35

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 :=
by
  sorry

end speed_in_still_water_l109_109190


namespace A_equals_4_of_rounded_to_tens_9430_l109_109170

variable (A B : ℕ)

theorem A_equals_4_of_rounded_to_tens_9430
  (h1 : 9430 = 9000 + 100 * A + 10 * 3 + B)
  (h2 : B < 5)
  (h3 : 0 ≤ A ∧ A ≤ 9)
  (h4 : 0 ≤ B ∧ B ≤ 9) :
  A = 4 :=
by
  sorry

end A_equals_4_of_rounded_to_tens_9430_l109_109170


namespace brown_ball_weight_l109_109087

def total_weight : ℝ := 9.12
def weight_blue : ℝ := 6
def weight_brown : ℝ := 3.12

theorem brown_ball_weight : total_weight - weight_blue = weight_brown :=
by 
  sorry

end brown_ball_weight_l109_109087


namespace prove_related_points_count_l109_109682

noncomputable def related_points_count : ℕ :=
  let G_curve := λ x : ℝ, Real.log (x + 1)
  let M_curve := λ x : ℝ, 1 / x
  let midpoint (A B : ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let A := (1, 0)
  let satisfies_condition (B : ℝ × ℝ) :=
    B.2 = G_curve B.1 ∧ midpoint A B ∈ set_of (λ p, p.2 = M_curve p.1)
  {n : ℕ // ∃ B : ℝ × ℝ, satisfies_condition B} := 1

theorem prove_related_points_count :
  related_points_count = 1 :=
sorry

end prove_related_points_count_l109_109682


namespace triangle_third_side_length_l109_109394

/-- In a triangle with sides a = 5 and b = 12 and the angle θ = 150 degrees, the length of the third side c is sqrt(169 + 60sqrt(3)). -/
theorem triangle_third_side_length {a b : ℝ} {θ : ℝ} (h_a : a = 5) (h_b : b = 12) (h_θ : θ = 150) :
  let c := real.sqrt (a^2 + b^2 - 2 * a * b * real.cos (θ * real.pi / 180)) in
  c = real.sqrt (169 + 60 * real.sqrt 3) :=
by
  sorry

end triangle_third_side_length_l109_109394


namespace initial_number_of_students_l109_109106

/-- 
Theorem: If the average mark of the students of a class in an exam is 90, and 2 students whose average mark is 45 are excluded, resulting in the average mark of the remaining students being 95, then the initial number of students is 20.
-/
theorem initial_number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = N * 90)
  (h2 : (T - 90) / (N - 2) = 95) : 
  N = 20 :=
sorry

end initial_number_of_students_l109_109106


namespace number_of_females_l109_109013

variable (M F C malt coke : ℕ)

-- Define the conditions as hypotheses
hypothesis condition1 : M = 10
hypothesis condition2 : 2 * coke = malt
hypothesis condition3 : 6 + 8 = malt
hypothesis condition4 : M = 6
hypothesis condition5 : F = 8 + 7

-- State the theorem to prove the number of females in the group
theorem number_of_females : F = 15 :=
by
  have h1 : M + 6 + 8 = 10 := condition1
  have h2 : malt = 14 := condition3
  have h3 : coke = 7 := by sorry
  -- From condition2 and previous steps conclude that F = 15
  exact condition5

end number_of_females_l109_109013


namespace min_value_frac_inverse_l109_109308

-- Define the function f(x) as given in the problem
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x <= 1 then 1 - Real.log x
  else -1 + Real.log x

-- State the lemma to be proven in Lean 4
theorem min_value_frac_inverse (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a > 1) (h₄ : b ≤ 1)
    (hf : f a = f b) : (1 / a + 1 / b) = 1 + 1 / Real.exp 2 := 
by
  sorry

end min_value_frac_inverse_l109_109308


namespace probability_plane_contains_points_inside_octahedron_l109_109218

noncomputable def enhanced_octahedron_probability : ℚ :=
  let total_vertices := 18
  let total_ways := Nat.choose total_vertices 3
  let faces := 8
  let triangles_per_face := 4
  let unfavorable_ways := faces * triangles_per_face
  total_ways - unfavorable_ways

theorem probability_plane_contains_points_inside_octahedron :
  enhanced_octahedron_probability / (816 : ℚ) = 49 / 51 :=
sorry

end probability_plane_contains_points_inside_octahedron_l109_109218


namespace michael_wants_to_buy_more_packs_l109_109072

theorem michael_wants_to_buy_more_packs
  (initial_packs : ℕ)
  (cost_per_pack : ℝ)
  (total_value_after_purchase : ℝ)
  (value_of_current_packs : ℝ := initial_packs * cost_per_pack)
  (additional_value_needed : ℝ := total_value_after_purchase - value_of_current_packs)
  (packs_to_buy : ℝ := additional_value_needed / cost_per_pack)
  (answer : ℕ := 2) :
  initial_packs = 4 → cost_per_pack = 2.5 → total_value_after_purchase = 15 → packs_to_buy = answer :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end michael_wants_to_buy_more_packs_l109_109072


namespace total_weight_is_319_l109_109465

variable (Jim_weight : ℕ)
variable (Steve_weight : ℕ)
variable (Stan_weight : ℕ)

axiom Jim_weight_is_110 : Jim_weight = 110
axiom Steve_is_8_lbs_lighter_than_Jim : Steve_weight = Jim_weight - 8
axiom Stan_is_5_lbs_heavier_than_Steve : Stan_weight = Steve_weight + 5

theorem total_weight_is_319 :
  Jim_weight + Steve_weight + Stan_weight = 319 :=
by
  rw [Jim_weight_is_110, Steve_is_8_lbs_lighter_than_Jim]
  simp [Stan_is_5_lbs_heavier_than_Steve]
  sorry -- Proving the arithmetic, to be completed.

end total_weight_is_319_l109_109465


namespace distance_to_place_is_24_l109_109569

-- Definitions of the problem's conditions
def rowing_speed_still_water := 10    -- kmph
def current_velocity := 2             -- kmph
def round_trip_time := 5              -- hours

-- Effective speeds
def effective_speed_with_current := rowing_speed_still_water + current_velocity
def effective_speed_against_current := rowing_speed_still_water - current_velocity

-- Define the unknown distance D
variable (D : ℕ)

-- Define the times for each leg of the trip
def time_with_current := D / effective_speed_with_current
def time_against_current := D / effective_speed_against_current

-- The final theorem stating the round trip distance
theorem distance_to_place_is_24 :
  time_with_current + time_against_current = round_trip_time → D = 24 :=
by sorry

end distance_to_place_is_24_l109_109569


namespace pq_bisects_ac_l109_109590

theorem pq_bisects_ac (A B C D M N P Q : Point) (hM : midpoint A B M) 
    (hN : midpoint C D N) (hP : midpoint B D P) (hQ : midpoint M N Q) 
    (hAC : A = (0, a)) (hBC : B = (0, 0)) (hCD : C = (0, c)) (hDC : D = (d, 0)) :
    bisects PQ AC := 
sorry

end pq_bisects_ac_l109_109590


namespace sam_initial_watermelons_l109_109091

theorem sam_initial_watermelons (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  -- proof steps would go here
  sorry

end sam_initial_watermelons_l109_109091


namespace expand_binomial_square_l109_109601

variables (x : ℝ)

theorem expand_binomial_square (x : ℝ) : (2 - x) ^ 2 = 4 - 4 * x + x ^ 2 := 
sorry

end expand_binomial_square_l109_109601


namespace smallest_a_l109_109104

-- Definitions based on conditions
def parabola (a : ℝ) (b : ℝ) (c : ℝ) (p : ℝ × ℝ) : Prop :=
  p = (1/2, -1/2) ∧
  ∃ x : ℝ, a > 0 ∧ y = a*x^2 + b*x + c

-- Condition 4 as a definition
def integer_condition (a b c : ℝ) : Prop :=
  ∃ n : ℤ, a + 2*b + 3*c = n

theorem smallest_a (a b c : ℝ) :
  parabola a b c (1/2, -1/2) →
  integer_condition a b c →
  a > 0 →
  a = 2 :=
sorry

end smallest_a_l109_109104


namespace largest_four_digit_number_divisible_by_5_l109_109515

theorem largest_four_digit_number_divisible_by_5 : ∃ n : ℕ, n < 10000 ∧ n % 5 = 0 ∧ ∀ m : ℕ, m < 10000 ∧ m % 5 = 0 → m ≤ n :=
begin
  use 9995,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m h1 h2,
    have : m ≤ 9999 := h1,
    nlinarith,
    sorry
  }
end

end largest_four_digit_number_divisible_by_5_l109_109515


namespace cindy_marbles_problem_l109_109832

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end cindy_marbles_problem_l109_109832


namespace solve_problem_l109_109069

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def perp_bisector_intersection :=
  let A := (24 : ℝ, 7 : ℝ)
  let B := (3 : ℝ, 4 : ℝ)
  let C := midpoint A B
  (2 * C.1 - 4 * C.2 = 5)

theorem solve_problem : perp_bisector_intersection :=
by
  sorry

end solve_problem_l109_109069


namespace EF_CD_eq_AC_BD_l109_109195

variable {α : Type} [LinearOrderedField α]

-- Declare the points
variables {O P A B C D E F : α}

-- Define the circle and relevant points on the circle
variable (ω : Circle O α) 

-- Conditions
variable (chord_AB : ¬Collinear {...})

/- Define P as a point on the minor arc AB -/
variable (P_on_minor_arc_AB : IsOnMinorArc P A B ω)

/- Define points E and F such that AE = EF = FB -/
variable (segments_split : (Distance A E = Distance E F) ∧ (Distance E F = Distance F B))

-- Define the fact that PE and PF intersect ω at points C and D, respectively
variable (PE_intersect_C : (OnLine P E C) ∧ (C ∈ ω))
variable (PF_intersect_D : (OnLine P F D) ∧ (D ∈ ω))

-- The theorem statement to be proved
theorem EF_CD_eq_AC_BD 
  (chord_AB : ¬Collinear {A, B, O})
  (P_on_minor_arc_AB : IsOnMinorArc P A B ω)
  (segments_split : (Distance A E = Distance E F) ∧ (Distance E F = Distance F B))
  (PE_intersect_C : (OnLine P E C) ∧ (C ∈ ω))
  (PF_intersect_D : (OnLine P F D) ∧ (D ∈ ω))
  :  Distance E F * Distance C D = Distance A C * Distance B D :=
sorry

end EF_CD_eq_AC_BD_l109_109195


namespace maze_paths_l109_109406

/-- Define the types of movements allowed in the maze. -/
inductive MovementDirection
  | right
  | down
  | up

/-- Define the properties of the junction points in the maze. -/
def Junction := {motive : MovementDirection // motive == MovementDirection.right ∨ motive == MovementDirection.down ∨ motive == MovementDirection.up}

/-- Function that calculates the number of paths from a given junction to the exit. -/
def num_paths_from_junction (junction : Junction) : ℕ :=
  if h : junction ∈ {x // x.motive = MovementDirection.right ∨ x.motive = MovementDirection.down}
  then 8 -- Always 8 paths from \( \times \)
  else 0 -- Meaningless since no such other junctions are evaluated

/-- The total number of paths through the maze from entry to exit is 16. -/
theorem maze_paths : num_paths_from_junction ⟨MovementDirection.right, by simp [MovementDirection.right]⟩ +
                       num_paths_from_junction ⟨MovementDirection.down, by simp [MovementDirection.down]⟩ = 16 := by
  sorry

end maze_paths_l109_109406


namespace sum_of_solutions_l109_109765

theorem sum_of_solutions :
  ∀ (x y : ℤ), (|x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|) →
  ({ (2, 8), (4, 10), (10, 4), (10, 4) }).sum (λ p, p.1 + p.2) = 52 :=
by
  sorry

end sum_of_solutions_l109_109765


namespace value_of_y_l109_109918

theorem value_of_y (y : ℝ) (h : (3 * y - 9) / 3 = 18) : y = 21 :=
sorry

end value_of_y_l109_109918


namespace problem_statement_l109_109588

def satisfies_condition (seq : List ℕ) : Prop :=
  ∀ i j : ℕ, (i < j ∧ j - i > 100) → |(seq.getD i 0) - (seq.getD j 0)| ≤ 100

noncomputable def special_sequence : List ℕ := 
  [102, 103, 104, ..., 199, 201, 1, 2, 3, ..., 99, 202, 200, 198, ..., 106, 104, 101]

theorem problem_statement : satisfies_condition special_sequence :=
  sorry

end problem_statement_l109_109588


namespace num_squares_sharing_two_vertices_l109_109428

-- Define the isosceles triangle and condition AB = AC
structure IsoscelesTriangle (A B C : Type) :=
  (AB AC : ℝ)
  (h_iso : AB = AC)

-- Define the problem statement in Lean
theorem num_squares_sharing_two_vertices 
  (A B C : Type) 
  (iso_tri : IsoscelesTriangle A B C) 
  (planeABC : ∀ P Q R : Type, P ≠ Q ∧ Q ≠ R ∧ P ≠ R) :
  ∃ n : ℕ, n = 4 := sorry

end num_squares_sharing_two_vertices_l109_109428


namespace base_conversion_correct_l109_109272

theorem base_conversion_correct :
  (let n1 := 2 * 8^2 + 5 * 8^1 + 4 * 8^0,
       d1 := 1 * 2^1 + 1 * 2^0,
       n2 := 1 * 5^2 + 4 * 5^1 + 4 * 5^0,
       d2 := 3 * 4^1 + 2 * 4^0 in
   (n1 / d1 + n2 / d2 : ℝ) = 57.4) :=
by
  sorry

end base_conversion_correct_l109_109272


namespace possible_values_expression_l109_109672

theorem possible_values_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (val : ℤ), val ∈ {5, 2, 1, -2, -3} ∧ 
  val = (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (a * b * c * d / |a * b * c * d|) :=
begin
  sorry
end

end possible_values_expression_l109_109672


namespace natural_number_representable_l109_109801

def strictly_decreasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) < a n

def sum_of_first_k_minus_one (a : ℕ → ℕ) (k : ℕ) : ℕ :=
(if k = 1 then 0 else (Nat.range (k - 1)).sum a)

theorem natural_number_representable
  (a : ℕ → ℕ)
  (h_decreasing : strictly_decreasing_seq a) :
  (∀ n : ℕ, ∃ (s : set ℕ), s.finite ∧ (∀ i ∈ s, i < n) ∧ (s.sum id = n)) ↔ (∀ k : ℕ, a k ≤ sum_of_first_k_minus_one a k + 1) :=
sorry

end natural_number_representable_l109_109801


namespace triangle_angle_contradiction_l109_109910

theorem triangle_angle_contradiction (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : A + B + C = 180) :
  A > 60 → B > 60 → C > 60 → false :=
by
  sorry

end triangle_angle_contradiction_l109_109910


namespace boys_meet_42_times_l109_109502

-- Definitions based on conditions
def track_length := 2 * 180 + 2 * 120  -- 600 ft
def speed_boy1 := 6  -- ft/s
def speed_boy2 := 8  -- ft/s
def relative_speed := speed_boy1 + speed_boy2  -- 14 ft/s
def meeting_time := track_length / relative_speed  -- Time to meet at point A again

-- Statement to prove the number of times they meet excluding the start and finish
theorem boys_meet_42_times : 
  (⟨boy1, boy2 : track_length, speed_boy1, speed_boy2, relative_speed, meeting_time⟩ :
  ∀ (track_length = 600) (speed_boy1 = 6) (speed_boy2 = 8) (relative_speed = 14) (meeting_time = track_length / relative_speed), 
  (meeting_time * relative_speed / track_length - 1 = 42)) :=
  sorry

end boys_meet_42_times_l109_109502


namespace perpendicular_and_square_condition_l109_109790

section geometry

variables {A B C G : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space G]

-- Assume we have triangle ABC and G its centroid
variable (triangle_ABC : triangle A B C)
variable (is_centroid_G : centroid G triangle_ABC)

-- The condition we need to prove
theorem perpendicular_and_square_condition :
  (perpendicular AG BG) ↔ (BC^2 + AC^2 = 5 * AB^2) := 
sorry

end geometry

end perpendicular_and_square_condition_l109_109790


namespace basketball_opponents_score_l109_109945

theorem basketball_opponents_score 
  (games : Fin 12 → ℕ)
  (score_range : ∀ i, 1 ≤ games i ∧ games i ≤ 12)
  (lost_by_two : {i // i < 12} → Prop)
  (lost_count : ∃ s : Finset (Fin 12), s.card = 6 ∧ ∀ i ∈ s, lost_by_two i)
  (lost_opponents_scores : ∀ i ∈ lost, ∃ j, j = games i + 2)
  (won_count : ∃ s : Finset (Fin 12), s.card = 6 ∧ ∀ i ∈ s, ¬ lost_by_two i)
  (won_opponents_scores : ∀ i ∈ won, ∃ j, 3 * j = games i)
  (won_non_neg : ∀ i ∈ won, ∃ j, j ≥ 0)
  : (∑ i in (lost ∪ won), (games i + (if lost_by_two i then 2 else games i / 3))) = 50 := 
sorry

end basketball_opponents_score_l109_109945


namespace problem_statement_l109_109025

variables {t x y ρ θ : ℝ}
def l_parametric : Prop := (x = -1 + t ∧ y = 1 + t)
def C_equation : Prop := (x - 2)^2 + (y - 1)^2 = 5
def P_polar_coordinates : Prop := true   -- Polar coordinates of point P are given and used in the triangle area calculation
def general_equation_l : Prop := y = x + 2
def polar_equation_C : Prop := ρ = 4 * cos θ + 2 * sin θ
def general_equation_l_prime : Prop := y = x
def area_triangle_PAB : ℝ := 6

theorem problem_statement 
(hl : l_parametric)
(hC : C_equation)
(hP : P_polar_coordinates)
(hgeneral_l : general_equation_l)
(hpolar_C : polar_equation_C)
(hgeneral_l_prime : general_equation_l_prime) :
  area_triangle_PAB = 6 :=
sorry

end problem_statement_l109_109025


namespace product_is_58_l109_109152

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def p := 2
def q := 29

-- Conditions based on the problem
axiom prime_p : is_prime p
axiom prime_q : is_prime q
axiom sum_eq_31 : p + q = 31

-- Theorem to be proven
theorem product_is_58 : p * q = 58 :=
by
  sorry

end product_is_58_l109_109152


namespace complex_with_smallest_argument_l109_109579

theorem complex_with_smallest_argument
  (p : ℂ)
  (h : abs (p - (0 + 25 * complex.I)) ≤ 15) :
  p = 12 + 16 * complex.I :=
sorry

end complex_with_smallest_argument_l109_109579


namespace johns_raise_percentage_increase_l109_109535

theorem johns_raise_percentage_increase (original_amount new_amount : ℝ) (h_original : original_amount = 60) (h_new : new_amount = 70) :
  ((new_amount - original_amount) / original_amount) * 100 = 16.67 := 
  sorry

end johns_raise_percentage_increase_l109_109535


namespace YooSeung_has_108_marbles_l109_109185

def YoungSoo_marble_count : ℕ := 21
def HanSol_marble_count : ℕ := YoungSoo_marble_count + 15
def YooSeung_marble_count : ℕ := 3 * HanSol_marble_count
def total_marble_count : ℕ := YoungSoo_marble_count + HanSol_marble_count + YooSeung_marble_count

theorem YooSeung_has_108_marbles 
  (h1 : YooSeung_marble_count = 3 * (YoungSoo_marble_count + 15))
  (h2 : HanSol_marble_count = YoungSoo_marble_count + 15)
  (h3 : total_marble_count = 165) :
  YooSeung_marble_count = 108 :=
by sorry

end YooSeung_has_108_marbles_l109_109185


namespace total_people_in_building_l109_109236

theorem total_people_in_building :
  ∀ (floors full_floors half_capacity_floors apartments_per_floor people_per_apartment : ℕ),
  floors = 12 →
  full_floors = floors / 2 →
  half_capacity_floors = floors - full_floors →
  apartments_per_floor = 10 →
  people_per_apartment = 4 →
  let people_in_full_floor := apartments_per_floor * people_per_apartment in
  let people_in_half_floor := people_in_full_floor / 2 in
  let total_people := full_floors * people_in_full_floor + half_capacity_floors * people_in_half_floor in
  total_people = 360 :=
by
  intros
  sorry

end total_people_in_building_l109_109236


namespace train_crossing_time_l109_109551

theorem train_crossing_time
  (length_of_train : ℝ)
  (speed_in_kmh : ℝ)
  (speed_in_mps : ℝ)
  (conversion_factor : ℝ)
  (time : ℝ)
  (h1 : length_of_train = 160)
  (h2 : speed_in_kmh = 36)
  (h3 : conversion_factor = 1 / 3.6)
  (h4 : speed_in_mps = speed_in_kmh * conversion_factor)
  (h5 : time = length_of_train / speed_in_mps) : time = 16 :=
by
  sorry

end train_crossing_time_l109_109551


namespace geometric_proof_l109_109962

noncomputable def sphere_touches_plane_at_point (CE ABC : set Point) (C : Point) : Prop :=
  touches (Sphere (diameter CE)) ABC C

noncomputable def tangent_to_sphere (AD : Line) (sphere : Sphere) : Prop :=
  is_tangent AD sphere

noncomputable def point_on_line (B : Point) (DE : Line) : Prop :=
  lies_on B DE

noncomputable def proof_problem (A B C : Point) (AD DE : Line) (CE ABC : set Point) (sphere : Sphere) (h1 : sphere_touches_plane_at_point CE ABC C)
(h2 : tangent_to_sphere AD sphere) (h3 : point_on_line B DE) : Prop :=
  distance A C = distance A B

theorem geometric_proof (A B C : Point) (AD DE : Line) (CE ABC : set Point) (sphere : Sphere)
  (h1 : sphere_touches_plane_at_point CE ABC C)
  (h2 : tangent_to_sphere AD sphere)
  (h3 : point_on_line B DE) :
  proof_problem A B C AD DE CE ABC sphere h1 h2 h3 :=
sorry

end geometric_proof_l109_109962


namespace exists_A_for_sqrt_d_l109_109842

def is_not_perfect_square (d : ℕ) : Prop := ∀ m : ℕ, m * m ≠ d

def s (d n : ℕ) : ℕ := 
  -- count number of 1's in the first n digits of binary representation of √d
  sorry 

theorem exists_A_for_sqrt_d (d : ℕ) (h : is_not_perfect_square d) :
  ∃ A : ℕ, ∀ n ≥ A, s d n > Int.sqrt (2 * n) - 2 :=
sorry

end exists_A_for_sqrt_d_l109_109842


namespace inequality_x_add_inv_x_ge_two_l109_109094

theorem inequality_x_add_inv_x_ge_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
  sorry

end inequality_x_add_inv_x_ge_two_l109_109094


namespace monkeys_bananas_minimum_l109_109898

theorem monkeys_bananas_minimum (b1 b2 b3 : ℕ) (x y z : ℕ) : 
  (x = 2 * y) ∧ (z = (2 * y) / 3) ∧ 
  (x = (2 * b1) / 3 + (b2 / 3) + (5 * b3) / 12) ∧ 
  (y = (b1 / 6) + (b2 / 3) + (5 * b3) / 12) ∧ 
  (z = (b1 / 6) + (b2 / 3) + (b3 / 6)) →
  b1 = 324 ∧ b2 = 162 ∧ b3 = 72 ∧ (b1 + b2 + b3 = 558) :=
sorry

end monkeys_bananas_minimum_l109_109898


namespace sum_of_solutions_l109_109764

theorem sum_of_solutions :
  ∀ (x y : ℤ), (|x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|) →
  ({ (2, 8), (4, 10), (10, 4), (10, 4) }).sum (λ p, p.1 + p.2) = 52 :=
by
  sorry

end sum_of_solutions_l109_109764


namespace trigonometric_function_monotonicity_l109_109698

noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem trigonometric_function_monotonicity :
  (∀ k : ℤ, is_monotonically_decreasing (λ x, sin(2 * x + π / 4))
    (k * π + π / 8) (k * π + 5 * π / 8)) :=
begin
  sorry
end

end trigonometric_function_monotonicity_l109_109698


namespace gcf_120_180_300_l109_109510

theorem gcf_120_180_300 : Nat.gcd (Nat.gcd 120 180) 300 = 60 := 
by eval_gcd 120 180 300

end gcf_120_180_300_l109_109510


namespace correct_coefficient_3x2_l109_109925

theorem correct_coefficient_3x2 :
  ∀ (x : ℝ), coeff(3 * x^2) = 3 :=
by
  intro x
  sorry

end correct_coefficient_3x2_l109_109925


namespace book_arrangement_not_adjacent_l109_109491

theorem book_arrangement_not_adjacent (C M : Type) [Fintype C] [Fintype M] 
  (hC : Fintype.card C = 2) (hM : Fintype.card M = 2) : 
  ∃ n, n = 8 ∧ books_not_adjacent (C ⊕ M) n := 
sorry

def books_not_adjacent (books : Type) (n : ℕ) : Prop :=
  -- Definition of the condition that no two books of the same subject are adjacent
  sorry

end book_arrangement_not_adjacent_l109_109491


namespace largest_garden_is_candace_and_difference_is_100_l109_109229

-- Define the dimensions of the gardens
def area_alice : Nat := 30 * 50
def area_bob : Nat := 35 * 45
def area_candace : Nat := 40 * 40

-- The proof goal
theorem largest_garden_is_candace_and_difference_is_100 :
  area_candace > area_alice ∧ area_candace > area_bob ∧ area_candace - area_alice = 100 := by
    sorry

end largest_garden_is_candace_and_difference_is_100_l109_109229


namespace sufficient_but_not_necessary_condition_for_monotonicity_l109_109484

theorem sufficient_but_not_necessary_condition_for_monotonicity
  (a : ℕ → ℝ)
  (h_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = a n ^ 2)
  (h_initial : a 1 = 2) :
  (∀ n : ℕ, n > 0 → a n > a 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_monotonicity_l109_109484


namespace count_valley_numbers_l109_109160

def is_valley_number (n : Nat) : Prop :=
  let digits := (n / 100, (n / 10) % 10, n % 10)
  match digits with
  | (x, y, z) => (x > y ∧ y < z) ∨ (x > y ∧ y = z) ∨ (x > y ∧ z = 0)

theorem count_valley_numbers : 
  (Finset.filter is_valley_number (Finset.range 1000)).card = 156 :=
sorry

end count_valley_numbers_l109_109160


namespace find_C_income_l109_109875

-- Definitions for the incomes
def A_m := 470400 / 12
def B_m := A_m / (5 / 2)
def C_m := B_m / 1.12

-- Theorem: Prove that C's monthly income is 14,000 given the conditions
theorem find_C_income : C_m = 14000 := by
  -- Definitions based on the conditions
  let A_m := 470400 / 12
  let B_m := A_m * (2 / 5)
  let C_m := B_m / 1.12
  
  -- sorry is used to skip the actual proof steps, which are not required
  sorry

end find_C_income_l109_109875


namespace profit_calculation_l109_109902

noncomputable def total_profit_tom_and_jose 
  (t_invest: ℕ) (j_invest: ℕ) (j_months: ℕ) (j_share: ℕ) : ℕ :=
let t_total_invest := t_invest * 12 in
let j_total_invest := j_invest * j_months in
let ratio := t_total_invest / 90000 + j_total_invest / 90000 in
let part_value := j_share / (j_total_invest / 90000) in
part_value * ratio

theorem profit_calculation
  (t_invest: ℕ := 30000) 
  (j_invest: ℕ := 45000) 
  (j_months: ℕ := 10) 
  (j_share: ℕ := 40000) :
  total_profit_tom_and_jose t_invest j_invest j_months j_share = 72000 :=
by
  -- Proof goes here
  sorry

end profit_calculation_l109_109902


namespace tangential_quadrilateral_tangent_lengths_l109_109782

variable {A B C D : Type} -- vertices of the quadrilateral
variables {u v : ℝ} -- lengths of tangents from vertices A and C respectively
variables {AB CD : ℝ} -- lengths of sides AB and CD respectively

-- Let $ABCD$ be a tangential quadrilateral with $AB \parallel CD$
def is_tangential_quadrilateral (A B C D : Type) : Prop :=
  -- Definitions relating to the inscribed circle can be introduced here
  sorry

-- Main theorem statement
theorem tangential_quadrilateral_tangent_lengths 
  (hAB_parallel_CD : AB ∥ CD)
  (h_tangential_quadrilateral : is_tangential_quadrilateral A B C D)
  (hu : u = (tangent_length_from A to inscribed_circle))
  (hv : v = (tangent_length_from C to inscribed_circle)) : 
  u * CD = v * AB :=
sorry

end tangential_quadrilateral_tangent_lengths_l109_109782


namespace other_acute_angle_l109_109017

axiom right_angle : Real := 90
axiom sum_of_angles_triangle : Real := 180

theorem other_acute_angle (a : Real) (h : a = 27) : (sum_of_angles_triangle - right_angle - a = 63) :=
by
  rw [h]
  norm_num
  sorry

end other_acute_angle_l109_109017


namespace geometric_sum_eight_terms_l109_109472

theorem geometric_sum_eight_terms (a_1 : ℕ) (S_4 : ℕ) (r : ℕ) (S_8 : ℕ) 
    (h1 : r = 2) (h2 : S_4 = a_1 * (1 + r + r^2 + r^3)) (h3 : S_4 = 30) :
    S_8 = a_1 * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) → S_8 = 510 := 
by sorry

end geometric_sum_eight_terms_l109_109472


namespace fettuccine_to_tortellini_ratio_l109_109943

-- Definitions based on the problem conditions
def total_students := 800
def preferred_spaghetti := 320
def preferred_fettuccine := 200
def preferred_tortellini := 160
def preferred_penne := 120

-- Theorem to prove that the ratio is 5/4
theorem fettuccine_to_tortellini_ratio :
  (preferred_fettuccine : ℚ) / (preferred_tortellini : ℚ) = 5 / 4 :=
sorry

end fettuccine_to_tortellini_ratio_l109_109943


namespace log_base_example_l109_109270

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ :=
if b > 0 ∧ b ≠ 1 ∧ x > 0 then real.log x / real.log b else 0

theorem log_base_example : log_base 3 243 = 5 :=
by
  sorry

end log_base_example_l109_109270


namespace sin_product_eq_l109_109681

open Real

-- Define the conditions as hypotheses
variables {α β : ℝ}
hypothesis hα : α ∈ Ioo (-π / 2) (π / 2)
hypothesis hβ : β ∈ Ioo (-π / 2) (π / 2)
hypothesis h_arith_seq : α + π / 2 = 2 * β
hypothesis h_cos_beta : cos β = sqrt 6 / 3

-- Goal to prove
theorem sin_product_eq :
  sin α * sin β = -sqrt 3 / 9 :=
by
  sorry

end sin_product_eq_l109_109681


namespace rectangle_length_to_width_ratio_l109_109654

theorem rectangle_length_to_width_ratio :
  ∀ (s : ℝ), 
  let large_square_side := 3 * s in
  let small_square_side := s in
  let rectangle_width := s in
  let rectangle_length := (large_square_side / 2) in
  rectangle_length / rectangle_width = 1.5 := 
by
  assume s,
  let large_square_side := 3 * s,
  let small_square_side := s,
  let rectangle_width := s,
  let rectangle_length := (large_square_side / 2),
  show rectangle_length / rectangle_width = 1.5,
  sorry

end rectangle_length_to_width_ratio_l109_109654


namespace remainder_ab_div_48_is_15_l109_109532

noncomputable def remainder_ab_div_48 (a b : ℕ) (ha : a % 8 = 3) (hb : b % 6 = 5) : ℕ :=
  (a * b) % 48

theorem remainder_ab_div_48_is_15 {a b : ℕ} (ha : a % 8 = 3) (hb : b % 6 = 5) : remainder_ab_div_48 a b ha hb = 15 :=
  sorry

end remainder_ab_div_48_is_15_l109_109532


namespace natural_sum_representation_l109_109795

-- Definitions
def is_strictly_decreasing (a : ℕ → ℕ) := ∀ n, a (n + 1) < a n
def S (a : ℕ → ℕ) (k : ℕ) : ℕ := if k = 1 then 0 else finset.sum (finset.range (k - 1)) a

-- Theorem statement
theorem natural_sum_representation (a : ℕ → ℕ) 
  (h_dec : is_strictly_decreasing a) :
  (∀ (k : ℕ), a k ≤ S a k + 1) ↔ 
  ∀ n, ∃ s : finset ℕ, s.to_finset.sum a = n :=
sorry

end natural_sum_representation_l109_109795


namespace students_neither_play_l109_109734

theorem students_neither_play (total_students football_players tennis_players both_players neither_players : ℕ)
  (h1 : total_students = 40)
  (h2 : football_players = 26)
  (h3 : tennis_players = 20)
  (h4 : both_players = 17)
  (h5 : neither_players = total_students - (football_players + tennis_players - both_players)) :
  neither_players = 11 :=
by
  sorry

end students_neither_play_l109_109734


namespace train_a_speed_54_l109_109504

noncomputable def speed_of_train_A (length_A length_B : ℕ) (speed_B : ℕ) (time_to_cross : ℕ) : ℕ :=
  let total_distance := length_A + length_B
  let relative_speed := total_distance / time_to_cross
  let relative_speed_km_per_hr := relative_speed * 36 / 10
  let speed_A := relative_speed_km_per_hr - speed_B
  speed_A

theorem train_a_speed_54 
  (length_A length_B : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (h_length_A : length_A = 150)
  (h_length_B : length_B = 150)
  (h_speed_B : speed_B = 36)
  (h_time_to_cross : time_to_cross = 12) :
  speed_of_train_A length_A length_B speed_B time_to_cross = 54 := by
  sorry

end train_a_speed_54_l109_109504


namespace milk_needed_6_cookies_3_3_pints_l109_109144

def gallon_to_quarts (g : ℚ) : ℚ := g * 4
def quarts_to_pints (q : ℚ) : ℚ := q * 2
def cookies_to_pints (p : ℚ) (c : ℚ) (n : ℚ) : ℚ := (p / c) * n
def measurement_error (p : ℚ) : ℚ := p * 1.1

theorem milk_needed_6_cookies_3_3_pints :
  (measurement_error (cookies_to_pints (quarts_to_pints (gallon_to_quarts 1.5)) 24 6) = 3.3) :=
by
  sorry

end milk_needed_6_cookies_3_3_pints_l109_109144


namespace max_voltage_on_capacitor_l109_109741

-- Conditions
variable (C : ℝ) (L1 L2 : ℝ) (I_max : ℝ) (U_max : ℝ)

-- Given values
def C_value := 1e-6 -- capacitance in farads
def L1_value := 4e-3 -- inductance in henrys
def L2_value := 2e-3 -- inductance in henrys
def I_max_value := 10e-3 -- current in amperes
def U_max_expected := 516e-3 -- expected maximum voltage in volts

-- Given conditions
axiom capacitor_is_1uF : C = C_value
axiom inductorL1_is_4mH : L1 = L1_value
axiom inductorL2_is_2mH : L2 = L2_value
axiom current_is_10mA : I_max = I_max_value

-- Theorem to be proven
theorem max_voltage_on_capacitor :
  U_max = U_max_expected := by
  sorry

end max_voltage_on_capacitor_l109_109741


namespace cyclic_quadrilateral_intersecting_circles_l109_109422

theorem cyclic_quadrilateral_intersecting_circles 
  (O : Type) [metric_space O] [has_dist O]
  (R : ℝ)
  (A B C D A1 A2 B1 B2 C1 C2 D1 D2 : O)
  (c : O → ℝ → Prop)
  (C_A C_B C_C C_D : O → ℝ → Prop)
  (h_inscribed : c O R → cyclic_quadrilateral A B C D)
  (h_center_CA : ∀ P, C_A A P → dist A P = R)
  (h_center_CB : ∀ P, C_B B P → dist B P = R)
  (h_center_CC : ∀ P, C_C C P → dist C P = R)
  (h_center_CD : ∀ P, C_D D P → dist D P = R)
  (h_intersect_CA : C_A A1 A2)
  (h_intersect_CB : C_B B1 B2)
  (h_intersect_CC : C_C C1 C2)
  (h_intersect_CD : C_D D1 D2)
  : cyclic_quadrilateral A1 A2 B1 B2 C1 C2 D1 D2 :=
by sorry

end cyclic_quadrilateral_intersecting_circles_l109_109422


namespace inequality_proof_l109_109438

open Real

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l109_109438


namespace expectation_converges_to_zero_l109_109791

noncomputable theory

variables {α : Type*} [measurable_space α] {μ : measure α}

-- Conditions
variable (ξ : α → ℝ)
variable (ξn : ℕ → α → ℝ)
variable (n : ℕ)

-- Assumptions
axiom a1 : ∀ᵐ ω ∂μ, antitone (λ n, ξn n ω)
axiom a2 : ∀ n, integrable (ξn n) μ
axiom a3 : ∃ C, ∀ n, C ≤ ∫ x, ξn n x ∂μ

-- The statement to prove
theorem expectation_converges_to_zero : tendsto (λ n, ∫ x, |ξn n x - ξ x| ∂μ) at_top (𝓝 0) :=
sorry

end expectation_converges_to_zero_l109_109791


namespace number_of_solutions_l109_109434

theorem number_of_solutions (p : ℕ) (hp : prime p) (h : 2 < p) : 
  (set_of (λ a, ∃ c k, c^2 = p * k + a)).finite.card = (p + 1) / 2 :=
sorry

end number_of_solutions_l109_109434


namespace nat_sum_representation_l109_109805

noncomputable def S (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  if k = 1 then 0 else (Finset.range (k - 1)).sum a

theorem nat_sum_representation (a : ℕ → ℕ) (h_seq : ∀ n m, n < m → a n > a m) :
  (∀ n, ∃ U : Finset ℕ, U.sum a = n ∧ U.pairwise (≠)) ↔
  (∀ k, a k ≤ S a k + 1) := 
sorry

end nat_sum_representation_l109_109805


namespace angle_equality_l109_109034

variables {A B C P K D E F : Type} 
variables (circ_O : ∀ x, x ∈ {A, B, C} → Point x) 
variables (circ_Γ : ∀ x, x ∈ {K, P, C, D, E} → Point x)
variables [InscribedInCircle A B C circ_O]
variables [OnArc P B C circ_O]
variables [OnSegment K A P]
variables [Bisects BK (Angle A B C)]
variables [OnCircle {K, P, C} circ_Γ]
variables [IntersectsAt D circ_Γ A C]
variables [IntersectsAgain E circ_Γ B D]
variables [ExtendedIntersectsAt F circ_Γ P E AB]

theorem angle_equality : ∀ (A B C P K D E F : Point), 
  inscribed_in_circle A B C circ_O ∧
  on_arc P B C circ_O ∧
  on_segment K A P ∧
  bisects BK (angle ABC) ∧
  passes_through circ_Γ K P D C ∧
  intersects_side D circ_Γ A C ∧
  intersects_again E circ_Γ B D ∧
  extended_intersects F circ_Γ P E AB →
  angle_ABC = 2 * angle_FCB :=
  sorry

end angle_equality_l109_109034


namespace part_I_part_II_l109_109656

noncomputable def a_seq (n : ℕ) : ℝ 
  | 0     => 1
  | (n+1) => a_seq n ^ 2 + 2 * a_seq n

def log2_seq (n : ℕ) : ℝ := Real.log2 (a_seq n + 1)

def b_seq (n : ℕ) : ℝ := n * log2_seq n

def S_n (n : ℕ) : ℝ := (Finset.range (n + 1)).sum b_seq

theorem part_I (n : ℕ) : 
  ∃ r : ℝ, (∀ m : ℕ, log2_seq (m + 1) = r * log2_seq m) ∧ (r = 0.5) :=
sorry

theorem part_II (n : ℕ) : 
  1 ≤ S_n n ∧ S_n n < 4 :=
sorry

end part_I_part_II_l109_109656


namespace quadratic_relationship_l109_109316

theorem quadratic_relationship (a b c : ℝ) (h1 : a ≠ 0):
  (∀ x : ℝ, a^2 * x^2 + b^2 * x + c^2 = 0) →
  (∀ x : ℝ, a * x^2 + b * x + c = 0) →
  (let m n : ℝ := (roots_of_equation_1 h1) in
   let α β : ℝ := (roots_of_equation_2 h1) in
   (m + n = α^2 + β^2)) →
  b^2 = a * c :=
by
  sorry

noncomputable def roots_of_equation_1 (a b c : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  let Δ := b^2 * b^2 - 4 * a^2 * c^2 in
  ((-b^2 + Real.sqrt Δ) / (2 * a^2), (-b^2 - Real.sqrt Δ) / (2 * a^2))

noncomputable def roots_of_equation_2 (a b c : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  let Δ := b * b - 4 * a * c in
  ((-b + Real.sqrt Δ) / (2 * a), (-b - Real.sqrt Δ) / (2 * a))

end quadratic_relationship_l109_109316


namespace math_problem_proof_l109_109248

noncomputable def problem_expr : ℚ :=
  ((11 + 1/9) - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / (36/10) / (2 + 6/25)

theorem math_problem_proof : problem_expr = 20 / 9 := by
  sorry

end math_problem_proof_l109_109248


namespace tan_angle_ABC_l109_109747

theorem tan_angle_ABC
  {A B C : Type} [metric_space A]
  (dist_AB : dist A B = 30)
  (dist_BC : dist B C = 34)
  (right_angle_BAC : ∀ (angle_BAC : A), angle_BAC = 90) :
  tan A = 8 / 15 :=
sorry

end tan_angle_ABC_l109_109747


namespace photo_album_pages_l109_109232

theorem photo_album_pages 
  (total_photos : ℕ)
  (photos_first_10_pages : ℕ)
  (photos_next_10_pages : ℕ)
  (photos_remaining_per_page : ℕ) 
  (photos_remaining_pages : ℕ)
  (total_photos = 100) 
  (photos_first_10_pages = 10 * 3) 
  (photos_next_10_pages = 10 * 4) 
  (photos_remaining_per_page = 3) :
  (10 * 3 + 10 * 4 + (100 - (10 * 3 + 10 * 4)) / 3 = 30) := 
by
  sorry

end photo_album_pages_l109_109232


namespace max_min_values_f_l109_109639

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem max_min_values_f :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 2) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ Real.sqrt 3) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = Real.sqrt 3) :=
by
  sorry

end max_min_values_f_l109_109639


namespace percentage_by_which_x_more_than_y_l109_109009

theorem percentage_by_which_x_more_than_y
    (x y z : ℝ)
    (h1 : y = 1.20 * z)
    (h2 : z = 150)
    (h3 : x + y + z = 555) :
    ((x - y) / y) * 100 = 25 :=
by
  sorry

end percentage_by_which_x_more_than_y_l109_109009


namespace inverse_proportion_indeterminate_l109_109341

theorem inverse_proportion_indeterminate (k : ℝ) (x1 x2 y1 y2 : ℝ) (h1 : x1 < x2)
  (h2 : y1 = k / x1) (h3 : y2 = k / x2) : 
  (y1 > 0 ∧ y2 > 0) ∨ (y1 < 0 ∧ y2 < 0) ∨ (y1 * y2 < 0) → false :=
sorry

end inverse_proportion_indeterminate_l109_109341


namespace radius_of_sphere_is_six_l109_109955

-- Variables and conditions
variable (r : ℝ)
variable (h_meter_stick : ℝ := 1.5)
variable (shadow_meter_stick : ℝ := 3)
variable (shadow_sphere : ℝ := 12)

-- Assumptions based on the problem statement
def tan_theta := h_meter_stick / shadow_meter_stick

-- Translate the condition from the problem
def sphere_tan_condition := r / shadow_sphere = tan_theta

-- The final proof goal
theorem radius_of_sphere_is_six (h_meter_stick : ℝ := 1.5)
                                 (shadow_meter_stick : ℝ := 3)
                                 (shadow_sphere : ℝ := 12)
                                 (h : sphere_tan_condition r) : r = 6 :=
by
  sorry

end radius_of_sphere_is_six_l109_109955


namespace total_drums_hit_l109_109419

/-- 
Given the conditions of the problem, Juanita hits 4500 drums in total. 
-/
theorem total_drums_hit (entry_fee cost_per_drum_hit earnings_per_drum_hit_beyond_200_double
                         net_loss: ℝ) 
                         (first_200_drums hits_after_200: ℕ) :
  entry_fee = 10 → 
  cost_per_drum_hit = 0.02 →
  earnings_per_drum_hit_beyond_200_double = 0.025 →
  net_loss = -7.5 →
  hits_after_200 = 4300 →
  first_200_drums = 200 →
  (-net_loss = entry_fee + (first_200_drums * cost_per_drum_hit) +
   (hits_after_200 * (earnings_per_drum_hit_beyond_200_double - cost_per_drum_hit))) →
  first_200_drums + hits_after_200 = 4500 :=
by
  intro h_entry_fee h_cost_per_drum_hit h_earnings_per_drum_hit_beyond_200_double h_net_loss h_hits_after_200
       h_first_200_drums h_loss_equation
  sorry

end total_drums_hit_l109_109419


namespace inverse_g_of_neg_92_l109_109703

noncomputable def g (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 1

theorem inverse_g_of_neg_92 : g (-3) = -92 :=
by 
-- This would be the proof but we are skipping it as requested
sorry

end inverse_g_of_neg_92_l109_109703


namespace borrowing_period_l109_109568

theorem borrowing_period 
  (principal : ℕ) (rate_1 : ℕ) (rate_2 : ℕ) (gain : ℕ)
  (h1 : principal = 5000)
  (h2 : rate_1 = 4)
  (h3 : rate_2 = 8)
  (h4 : gain = 200)
  : ∃ n : ℕ, n = 1 :=
by
  sorry

end borrowing_period_l109_109568


namespace odd_a_all_white_or_black_even_a_not_all_same_l109_109193

-- Define the problem: Black-White inversion operations on a 2011-gon.

def invert_colors (vertices_colored : List Bool) (start : ℕ) (a : ℕ) (n : ℕ) : List Bool :=
  vertices_colored.mapIdx (λ idx color =>
    if start ≤ idx ∧ idx < start + a then
      !color
    else
      color)

theorem odd_a_all_white_or_black (a : ℕ) (h1 : a < 2011) (h2 : a % 2 = 1) : 
  ∃ (f : List Bool → List Bool), 
    (∀ v : List Bool, f v = List.repeat true 2011) ∨ (∀ v : List Bool, f v = List.repeat false 2011) := 
sorry

theorem even_a_not_all_same (a : ℕ) (h1 : a < 2011) (h2 : a % 2 = 0) : 
  ¬ ∃ (f : List Bool → List Bool), 
    (∀ v : List Bool, f v = List.repeat true 2011) ∧ (∀ v : List Bool, f v = List.repeat false 2011) :=
sorry

end odd_a_all_white_or_black_even_a_not_all_same_l109_109193


namespace diesel_fuel_cost_l109_109545

def cost_per_liter (total_cost : ℝ) (num_liters : ℝ) : ℝ := total_cost / num_liters

def full_tank_cost (cost_per_l : ℝ) (tank_capacity : ℝ) : ℝ := cost_per_l * tank_capacity

theorem diesel_fuel_cost (total_cost : ℝ) (num_liters : ℝ) (tank_capacity : ℝ) :
  total_cost = 18 → num_liters = 36 → tank_capacity = 64 → full_tank_cost (cost_per_liter total_cost num_liters) tank_capacity = 32 :=
by
  intros h_total h_num h_tank
  rw [h_total, h_num, h_tank]
  norm_num
  sorry -- Full proof can be completed with detailed steps.

end diesel_fuel_cost_l109_109545


namespace compute_a_inv3_b_sq_l109_109431

theorem compute_a_inv3_b_sq (a b : ℚ) (h1 : a = 4/7) (h2 : b = 5/6) : a^(-3) * b^(2) = 8575 / 2304 := by
  -- definitions and conditions used in the problem
  rw [h1, h2]
  sorry

end compute_a_inv3_b_sq_l109_109431


namespace solve_quadratic_eq_l109_109464

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2 * x = 1) : x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end solve_quadratic_eq_l109_109464


namespace log_base_example_l109_109269

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ :=
if b > 0 ∧ b ≠ 1 ∧ x > 0 then real.log x / real.log b else 0

theorem log_base_example : log_base 3 243 = 5 :=
by
  sorry

end log_base_example_l109_109269


namespace cos_sum_nonneg_l109_109854

theorem cos_sum_nonneg {α β γ : ℝ} 
  (hα : α ∈ Ico (0 : ℝ) (π / 2)) 
  (hβ : β ∈ Ico (0 : ℝ) (π / 2)) 
  (hγ : γ ∈ Ico (0 : ℝ) (π / 2)) 
  (h_sum_tan : Real.tan α + Real.tan β + Real.tan γ ≤ 3) :
  Real.cos (2 * α) + Real.cos (2 * β) + Real.cos (2 * γ) ≥ 0 :=
by
  sorry

end cos_sum_nonneg_l109_109854


namespace transformed_data_stats_l109_109724

theorem transformed_data_stats (a : ℕ → ℝ) (n : ℕ)
  (h_average : (∑ i in finset.range n, a i) / n = 10)
  (h_variance : (∑ i in finset.range n, (a i - 10) ^ 2) / n = 4) :
  ((∑ i in finset.range n, (3 * a i - 2)) / n = 28) ∧
  ((∑ i in finset.range n, ((3 * a i - 2) - 28) ^ 2) / n = 36) :=
by
  -- Proof to be filled in here
  sorry

end transformed_data_stats_l109_109724


namespace sum_of_b_equals_30_l109_109614

def b : ℕ → ℚ
| 1     := 2
| 2     := 3
| (n+3) := 1/2 * b(n+2) + 1/3 * b(n+1)

def S : ℚ := ∑' n : ℕ, b (n+1)

theorem sum_of_b_equals_30 : S = 30 := 
by {
    sorry
}

end sum_of_b_equals_30_l109_109614


namespace black_marbles_remaining_l109_109458

-- Given conditions
def original_black_marbles : ℕ := 792
def marbles_taken_by_fred : ℕ := 233

-- Prove that Sara has 559 black marbles left
theorem black_marbles_remaining : original_black_marbles - marbles_taken_by_fred = 559 :=
by
  rw [original_black_marbles, marbles_taken_by_fred]
  exact Nat.sub_eq_of_eq_add sorry

end black_marbles_remaining_l109_109458


namespace average_mark_of_all_three_boys_is_432_l109_109740

noncomputable def max_score : ℝ := 900
noncomputable def get_score (percent : ℝ) : ℝ := (percent / 100) * max_score

noncomputable def amar_score : ℝ := get_score 64
noncomputable def bhavan_score : ℝ := get_score 36
noncomputable def chetan_score : ℝ := get_score 44

noncomputable def total_score : ℝ := amar_score + bhavan_score + chetan_score
noncomputable def average_score : ℝ := total_score / 3

theorem average_mark_of_all_three_boys_is_432 : average_score = 432 := 
by
  sorry

end average_mark_of_all_three_boys_is_432_l109_109740


namespace hexagon_angle_sum_l109_109447

theorem hexagon_angle_sum (A B C D E F : Type) :
  angle A + angle B + angle C + angle D + angle E + angle F = 720 :=
sorry

end hexagon_angle_sum_l109_109447


namespace value_of_a2017_l109_109318

noncomputable def a : ℕ → ℝ 
| 1 := 1
| 2 := 2
| n := if h : n ≥ 3 then a (n - 1) / a (n - 2) else 0

theorem value_of_a2017 : a 2017 = 1 :=
by 
  -- Proof goes here
  sorry

end value_of_a2017_l109_109318


namespace eight_by_eight_checkerboard_not_tileable_with_3x1_triminos_eight_by_eight_with_top_left_missing_tileable_with_3x1_triminos_l109_109416

theorem eight_by_eight_checkerboard_not_tileable_with_3x1_triminos : ¬ ∃ t : set (ℕ × ℕ), (∀ p ∈ t, p.1 < 8 ∧ p.2 < 8) ∧ (∀ (x y : ℕ), ((x, y)∈ t) → ((x+1, y)∈ t) ∨ ((x, y+1)∈ t) ∨ ((x+2, y) ∈ t)) ∧ (t.card = 64) :=
sorry

theorem eight_by_eight_with_top_left_missing_tileable_with_3x1_triminos : ∃ t : set (ℕ × ℕ), (∀ p ∈ t, p.1 < 8 ∧ p.2 < 8) ∧ ¬((0, 0)∈ t) ∧ (∀ (x y : ℕ), ((x, y)∈ t) → ((x+1, y)∈ t) ∨ ((x, y+1)∈ t) ∨ ((x+2, y) ∈ t)) ∧ (t.card = 63) :=
sorry

end eight_by_eight_checkerboard_not_tileable_with_3x1_triminos_eight_by_eight_with_top_left_missing_tileable_with_3x1_triminos_l109_109416


namespace KrystianaChargesForSecondFloorRooms_Theorem_l109_109047

noncomputable def KrystianaChargesForSecondFloorRooms (X : ℝ) : Prop :=
  let costFirstFloor := 3 * 15
  let costThirdFloor := 3 * (2 * 15)
  let totalEarnings := costFirstFloor + 3 * X + costThirdFloor
  totalEarnings = 165 → X = 10

-- This is the statement only. The proof is not included.
theorem KrystianaChargesForSecondFloorRooms_Theorem : KrystianaChargesForSecondFloorRooms 10 :=
sorry

end KrystianaChargesForSecondFloorRooms_Theorem_l109_109047


namespace problem_1_problem_2_l109_109038

def triangle (α β γ BC A : ℝ) (tri_ABC : α + β + γ = π) : Prop := true

theorem problem_1 :
  ∀ (BC : ℝ) (B A : ℝ) (tri_ABC : triangle A B (π - A - B) BC (π - B - (π - A - B))) (AB : ℝ),
  B = π / 3 → AC = (3 * Real.sin (π / 3)) / (Real.sin (π / 4)) → AC = 3 * Real.sqrt 6 / 2 :=
by sorry

theorem problem_2 :
  ∀ (BC AB A_BC_area : ℝ) (tri_ABC_area: triangle A B (π - A - B) BC (π - B - (π - A - B))) (BC_Area_eq: BC * AB * Real.sin (π / 3) / 2 = 3 * Real.sqrt 3),
  B = π / 3 → A_BC_area = 3 * Real.sqrt 3 → AB = 4 → 
  AC = Real.sqrt (AB^2 + BC^2 - 2 * AB * BC * Real.cos (π / 3)) → AC = Real.sqrt 13 :=
by sorry

end problem_1_problem_2_l109_109038


namespace problem_divisible_by_480_l109_109722

theorem problem_divisible_by_480 (a : ℕ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) : ∃ k : ℕ, a * (a^2 - 1) * (a^2 - 4) = 480 * k :=
by
  sorry

end problem_divisible_by_480_l109_109722


namespace find_integer_n_l109_109279

theorem find_integer_n : ∃ n, 5 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ n = 9 :=   
by 
  -- The proof will be written here.
  sorry

end find_integer_n_l109_109279


namespace bug_max_path_length_l109_109203

/-- The dimensions of the rectangular prism. --/
def length : ℕ := 3
def width : ℕ := 4
def height : ℕ := 5

/-- The bug's path must visit every corner and return to the starting point.
We need to prove the length of this path within the given prism is maximized
as per the mentioned probable calculations in the solution. --/
theorem bug_max_path_length : 
  let space_diagonal := Real.sqrt (length^2 + width^2 + height^2)
  let face_diagonal_1 := Real.sqrt (length^2 + width^2)
  let face_diagonal_2 := Real.sqrt (width^2 + height^2)
  let face_diagonal_3 := Real.sqrt (length^2 + height^2) in
  MaximumPossibleLength length width height = 
  4 * space_diagonal + 2 * face_diagonal_2 + 5 := 
sorry

end bug_max_path_length_l109_109203


namespace zeta_distance_midway_major_axis_l109_109840

-- Define constants for perigee and apogee distances
def perigee : ℝ := 3
def apogee : ℝ := 15

-- Define a parameter for the major axis length
def major_axis_length : ℝ := perigee + apogee

-- Define the distance from the center to either end of the major axis
def semi_major_axis_length : ℝ := major_axis_length / 2

theorem zeta_distance_midway_major_axis :
  ∀ (P A : ℝ), P = perigee → A = apogee →
  major_axis_length = P + A →
  semi_major_axis_length = major_axis_length / 2 →
  semi_major_axis_length = 9 := 
by
  intros P A hP hA hMajorAxis hSemiMajorAxis
  sorry

end zeta_distance_midway_major_axis_l109_109840


namespace domain_of_g_l109_109618

def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g :
  {x : ℝ | 1 < log 6 x ∧ 1 < log 5 (log 6 x) ∧ 1 < log 4 (log 5 (log 6 x))} =
  {x : ℝ | x > 6^625} :=
by 
  sorry

end domain_of_g_l109_109618


namespace probability_at_least_one_white_l109_109493

-- Conditions
def total_balls : ℕ := 5
def probability_two_white : ℚ := 3 / 10

-- Binomial coefficient
def binom (n k : ℕ) : ℕ :=
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_at_least_one_white :
  let x := 3 in  -- number of white balls, determined from the given condition
  let y := total_balls - x in  -- number of black balls
  let prob_all_black := (binom y 2) / (binom total_balls 2) in
  1 - prob_all_black = 9 / 10 :=
by
  sorry  -- proof required

end probability_at_least_one_white_l109_109493


namespace process_always_terminates_average_steps_l109_109139

-- Definitions and assumptions 
def coin_flip_process (n : ℕ) : ℕ → ℕ :=
  sorry -- We need definitions to represent the state and steps of the process.

def binary_value (s : list ℕ) : ℕ :=
  s.foldr (λ x acc, x + 2 * acc) 0

-- Problem statement in Lean 4 to prove the process always terminates
theorem process_always_terminates (n : ℕ) : 
  ∀ initial_state : list ℕ, initial_state.length = n → 
  ∃ k, binary_value (coin_flip_process n k) = 0 :=
sorry

-- Problem statement in Lean 4 to calculate the average number of steps
theorem average_steps (n : ℕ) :
  ∑ k in finset.range (n + 1), (1 : ℚ) / (k + 1) = 
  ∑ i in finset.range n, 1 / (i + 1) :=
sorry

end process_always_terminates_average_steps_l109_109139


namespace leadership_ways_l109_109562

open Fintype

/--
In a village with 16 members, prove that the total number
of ways to choose a leadership consisting of 1 mayor,
3 deputy mayors, each with 3 council members, is 154828800.
-/
theorem leadership_ways (n : ℕ) (hn : n = 16) :
  (∃ m d1 d2 d3 c1 c2 c3 : Finₓ n,
    ∀ (h : m ≠ d1) (h : m ≠ d2) (h : m ≠ d3) (h : d1 ≠ d2)
    (h : d1 ≠ d3) (h : d2 ≠ d3),
      16 * 15 * 14 * 13 * choose 12 3 * choose 9 3 * choose 6 3 = 154828800) :=
by -- proof to be added
  sorry

end leadership_ways_l109_109562


namespace natural_sum_representation_l109_109794

-- Definitions
def is_strictly_decreasing (a : ℕ → ℕ) := ∀ n, a (n + 1) < a n
def S (a : ℕ → ℕ) (k : ℕ) : ℕ := if k = 1 then 0 else finset.sum (finset.range (k - 1)) a

-- Theorem statement
theorem natural_sum_representation (a : ℕ → ℕ) 
  (h_dec : is_strictly_decreasing a) :
  (∀ (k : ℕ), a k ≤ S a k + 1) ↔ 
  ∀ n, ∃ s : finset ℕ, s.to_finset.sum a = n :=
sorry

end natural_sum_representation_l109_109794


namespace triangle_ratio_l109_109414

theorem triangle_ratio (A B C P X Y Z : Type)
  [Triangle : ∀ {A B C : Type}, AC > AB]
  [P_bisector : ∃ P, P is_intersection_of_perp_bisector_and_internal_bisector A]
  [PX_perp_AB : ∃ X, PX ⊥ AB_at_ext_point X]
  [PY_perp_AC : ∃ Y, PY ⊥ AC_at_point Y]
  [Z_intersection : ∃ Z, Z is_intersection_of_XY_and_BC] :
  BZ / ZC = 1 :=
sorry

end triangle_ratio_l109_109414


namespace Two_Parallel_Lines_Set_l109_109337

open Real

variable (a π : Set (ℝ × ℝ × ℝ)) (d : ℝ)

-- Condition: line a is parallel to plane π
def Line_Parallel_To_Plane (a π : Set (ℝ × ℝ × ℝ)) : Prop := 
  ∃ v : ℝ × ℝ × ℝ, ∀ p ∈ a, ∃ t : ℝ, ∀ q ∈ π, q = (p.1 + t * v.1, p.2 + t * v.2, p.3 + t * v.3)

-- Condition: distance from line a to plane π is d
def Distance_From_Line_To_Plane (a π : Set (ℝ × ℝ × ℝ)) (d : ℝ) : Prop :=
  ∀ p ∈ a, ∃ q ∈ π, sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2) = d

-- Question: Set of points at distance d from line a and plane π is two parallel lines
theorem Two_Parallel_Lines_Set (a π : Set (ℝ × ℝ × ℝ)) (d : ℝ) 
  (h1 : Line_Parallel_To_Plane a π) 
  (h2 : Distance_From_Line_To_Plane a π d) : 
  ∃ l1 l2 : Set (ℝ × ℝ × ℝ), l1 ∪ l2 = { p : ℝ × ℝ × ℝ | 
    ∀ q ∈ a, sqrt((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2) = d ∧
    ∃ r ∈ π, sqrt((p.1 - r.1)^2 + (p.2 - r.2)^2 + (p.3 - r.3)^2) = d } ∧
  ∀ p q ∈ l1, ∃ v : ℝ × ℝ × ℝ, q = (p.1 + v.1, p.2 + v.2, p.3 + v.3) ∧
  ∀ p q ∈ l2, ∃ v : ℝ × ℝ × ℝ, q = (p.1 + v.1, p.2 + v.2, p.3 + v.3) :=
sorry

end Two_Parallel_Lines_Set_l109_109337


namespace total_cups_of_liquid_drunk_l109_109626

-- Definitions for the problem conditions
def elijah_pints : ℝ := 8.5
def emilio_pints : ℝ := 9.5
def cups_per_pint : ℝ := 2
def elijah_cups : ℝ := elijah_pints * cups_per_pint
def emilio_cups : ℝ := emilio_pints * cups_per_pint
def total_cups : ℝ := elijah_cups + emilio_cups

-- Theorem to prove the required equality
theorem total_cups_of_liquid_drunk : total_cups = 36 :=
by
  sorry

end total_cups_of_liquid_drunk_l109_109626


namespace cars_distance_l109_109903

-- Define the initial conditions
def initial_distance : ℝ := 105
def first_car_distance : ℝ := 50
def second_car_distance : ℝ := 35

-- Define the mathematical proposition to be proved
theorem cars_distance (d_initial d_first d_second : ℝ) 
  (h_initial : d_initial = 105) 
  (h_first : d_first = 50) 
  (h_second : d_second = 35) : 
  d_initial - (d_first + d_second) = 20 :=
by
  rw [h_initial, h_first, h_second]
  norm_num
  exact eq.refl 20

end cars_distance_l109_109903


namespace sin_65pi_over_6_l109_109993

theorem sin_65pi_over_6 : sin (65 * real.pi / 6) = 1 / 2 := 
by sorry

end sin_65pi_over_6_l109_109993


namespace option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l109_109919

theorem option_A_correct : 
  -5 * (-4) * (-2) * (-2) = 80 := 
by
  calc
    -5 * (-4) * (-2) * (-2)
        = 5 * 4 * (-2) * (-2) : by linarith
    ... = -5 * 4 * 2 * (-2) : by linarith
    ... = 5 * 4 * 2 * 2 : by linarith
    ... = 80 : by linarith

theorem option_B_incorrect : 
  (-12) * ((1/3) - (1/4) - 1) ≠ -4 + 3 + 1 :=
by
  calc
    (-12) * ((1/3) - (1/4) - 1)
        = (-12) * ((4/12) - (3/12) - (12/12)) : by linarith
    ... = (-12) * (-11/12) : by linarith
    ... = 11 : by linarith
    ... ≠ 0 : by linarith

theorem option_C_incorrect : 
  (-9) * 5 * (-4) * 0 ≠ 9 * 5 * 4 :=
by
  calc
    (-9) * 5 * (-4) * 0
        = 0 : by linarith
    ... ≠ 180 : by linarith

theorem option_D_incorrect :
  -2 * 5 - 2 * (-1) - (-2) * 2 ≠ -2 * (5 + 1 - 2) :=
by
  calc
    -2 * 5 - 2 * (-1) - (-2) * 2 
        = -10 + 2 + 4 : by linarith
    ... = -4 : by linarith
    ... ≠ -8 : by linarith

end option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l109_109919


namespace coefficient_x10_in_expansion_is_179_l109_109108

def poly : Polynomial ℤ := (X + 2)^10 * (X^2 - 1)

theorem coefficient_x10_in_expansion_is_179 : coeff poly 10 = 179 :=
by
  sorry

end coefficient_x10_in_expansion_is_179_l109_109108


namespace number_of_subsets_of_P_l109_109709

def M : Set ℕ := {0, 2, 4}
def P : Set ℕ := {x | ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ x = a * b}

theorem number_of_subsets_of_P : P.toFinset.card = 4 → 2 ^ 4 = 16 := by
  sorry

end number_of_subsets_of_P_l109_109709


namespace solution1_solution2_l109_109245

noncomputable def prob1_expr : ℝ :=
  (2 + 7/9)^(1/2) + 0.1^(-2) + (2 + 10/27)^(1/3) - 3 * Real.pi^0

noncomputable def prob2_expr : ℝ :=
  2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5

theorem solution1 : prob1_expr = 100 := by
  sorry

theorem solution2 : prob2_expr = 2 := by 
  sorry

end solution1_solution2_l109_109245


namespace distance_from_center_to_line_l109_109349

-- Define the parametric equations of the line l
def line_l_x (t : ℝ) : ℝ := (sqrt 2) / 2 * t
def line_l_y (t : ℝ) : ℝ := 1 + (sqrt 2) / 2 * t

-- Define the parametric equations of the circle C
def circle_C_x (θ : ℝ) : ℝ := cos θ + 2
def circle_C_y (θ : ℝ) : ℝ := sin θ

-- Define the center of the circle C
def center_C := (2 : ℝ, 0 : ℝ)

-- Define the standard form of the line l from its parametric equations
def line_l_standard (x y : ℝ) : Prop := y = x + 1

-- Function to calculate the distance from a point to a line in Ax + By + C = 0 form
def point_to_line_distance (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / sqrt (A^2 + B^2)

-- Main statement of the theorem
theorem distance_from_center_to_line :
  point_to_line_distance 2 0 (-1) 1 (-1) = (3 * sqrt 2) / 2 :=
by
  sorry

end distance_from_center_to_line_l109_109349


namespace cos_beta_value_l109_109333

open Real

theorem cos_beta_value (α β : ℝ) (h1 : sin α = sqrt 5 / 5) (h2 : sin (α - β) = - sqrt 10 / 10) (h3 : 0 < α ∧ α < π / 2) (h4 : 0 < β ∧ β < π / 2) : cos β = sqrt 2 / 2 :=
by
sorry

end cos_beta_value_l109_109333


namespace number_exceeds_by_35_l109_109533

theorem number_exceeds_by_35 (x : ℤ) (h : x = (3 / 8 : ℚ) * x + 35) : x = 56 :=
by
  sorry

end number_exceeds_by_35_l109_109533


namespace count_valid_arrangements_l109_109302

def valid_grid (grid : matrix (fin 4) (fin 4) (option char)) : Prop :=
  ∀ i j, ∃ row : fin 4, ∃ col : fin 4,
    grid row col = some 'A' ∧ grid i j ≠ some 'A'

-- Fixing 'A' at the second cell of the first row
def initial_placement : matrix (fin 4) (fin 4) (option char) :=
  λ _ _, none | (0,1) := some 'A'

def four_by_four_grid_constraints (grid : matrix (fin 4) (fin 4) (option char)) : Prop :=
  (∀ i, multiset.card (finset.univ.filter (λ j, grid i j = some 'A')) = 1) ∧
  (∀ j, multiset.card (finset.univ.filter (λ i, grid i j = some 'A')) = 1) ∧
  (∀ i, multiset.card (finset.univ.filter (λ j, grid i j = some 'B')) = 1) ∧
  (∀ j, multiset.card (finset.univ.filter (λ i, grid i j = some 'B')) = 1) ∧
  (∀ i, multiset.card (finset.univ.filter (λ j, grid i j = some 'C')) = 1) ∧
  (∀ j, multiset.card (finset.univ.filter (λ i, grid i j = some 'C')) = 1) ∧
  (∀ i, multiset.card (finset.univ.filter (λ j, grid i j = none)) = 1) ∧
  (∀ j, multiset.card (finset.univ.filter (λ i, grid i j = none)) = 1)

noncomputable def number_of_valid_arrangements (initial : matrix (fin 4) (fin 4) (option char)) : ℕ :=
  sorry  -- Here we skip the implementation of counting valid arrangements given the constraints

theorem count_valid_arrangements : number_of_valid_arrangements initial_placement = 4 :=
  sorry  -- This denotes the statement we want to prove

end count_valid_arrangements_l109_109302


namespace volume_tetrahedron_OABC_l109_109323

noncomputable def volume_of_tetrahedron (O : Point) (A B C : Point) (r : ℝ) : ℝ := 
  if 
    (distance O A = r) ∧ 
    (distance O B = r) ∧ 
    (distance O C = r) ∧ 
    (distance A B = 12 * sqrt 3) ∧ 
    (distance A C = 12) ∧ 
    (distance B C = 12) 
  then 
    60 * sqrt 3 
  else 
    0

-- The statement we need to prove:
theorem volume_tetrahedron_OABC :
  ∀ (O A B C : Point),
    (distance O A = 13) →
    (distance O B = 13) →
    (distance O C = 13) →
    (distance A B = 12 * sqrt 3) →
    (distance A C = 12) →
    (distance B C = 12) →
    volume_of_tetrahedron O A B C 13 = 60 * sqrt 3 :=
sorry

end volume_tetrahedron_OABC_l109_109323


namespace arithmetic_mean_of_given_numbers_l109_109161

def numbers : List ℝ := [-5, 3.5, 12, 20]

def sum_of_numbers (nums : List ℝ) : ℝ :=
  nums.foldr (· + ·) 0

def arithmetic_mean (nums : List ℝ) : ℝ :=
  sum_of_numbers nums / nums.length

theorem arithmetic_mean_of_given_numbers :
  arithmetic_mean numbers = 7.625 :=
by
  sorry

end arithmetic_mean_of_given_numbers_l109_109161


namespace length_AK_independent_of_D_on_BC_l109_109542

-- Given a triangle ABC
variables {A B C D K : Type*}

-- Assume A, B, and C are points, and D is a point on BC
variables [IsTriangle A B C] [IsPointOnLine D B C]

-- Assume K is a point on AD where AD intersects the common external tangent of the incircles of ABD and ACD.
axiom common_external_tangent : ∀ (A B C D : Type*), ∃ (K : Type*), IsPointOnLine K A D ∧ CommonExternalTangent (InCircle A B D) (InCircle A C D) K

-- Define the length function
noncomputable def length (x y : Type*) := sorry

-- Define the result: AK is independent of D and always equal to (AB + AC - BC) / 2
theorem length_AK_independent_of_D_on_BC : 
  ∀ (D : Type*) (hD : IsPointOnLine D B C), 
  AK_length (length K A) = (length A B + length A C - length B C) / 2 :=
sorry

end length_AK_independent_of_D_on_BC_l109_109542


namespace blue_balls_in_box_l109_109202

theorem blue_balls_in_box (total_balls : ℕ) (p_two_blue : ℚ) (b : ℕ) 
  (h1 : total_balls = 12) (h2 : p_two_blue = 1/22) 
  (h3 : (↑b / 12) * (↑(b-1) / 11) = p_two_blue) : b = 3 :=
by {
  sorry
}

end blue_balls_in_box_l109_109202


namespace slope_of_AB_l109_109346

theorem slope_of_AB (k : ℝ) (y1 y2 x1 x2 : ℝ) 
  (hP : (1, Real.sqrt 2) ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1})
  (hPA_eq : ∀ x, (x, y1) ∈ {p : ℝ × ℝ | p.2 = k * p.1 - k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hPB_eq : ∀ x, (x, y2) ∈ {p : ℝ × ℝ | p.2 = -k * p.1 + k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hx1 : y1 = k * x1 - k + Real.sqrt 2) 
  (hx2 : y2 = -k * x2 + k + Real.sqrt 2) :
  ((y2 - y1) / (x2 - x1)) = -2 - 2 * Real.sqrt 2 :=
by
  sorry

end slope_of_AB_l109_109346


namespace round_trip_time_l109_109110

variables {distance_downstream distance_upstream : ℝ} 
          {speed_boat_still speed_current : ℝ}

-- Define the conditions as Lean variables
def current_speed := 4 -- stream speed in kmph
def boat_speed := 8 -- boat speed in kmph in still water
def distance := 6 -- distance in km (one way)

-- Define the effective speeds for downstream and upstream
def downstream_speed := boat_speed + current_speed
def upstream_speed := boat_speed - current_speed

-- Define the times taken for downstream and upstream journeys
def time_downstream := distance / downstream_speed
def time_upstream := distance / upstream_speed

-- Prove the total time for the round trip is 2 hours
theorem round_trip_time : (time_downstream + time_upstream) = 2 :=
by
  unfold time_downstream time_upstream downstream_speed upstream_speed
  have h_downstream_speed : downstream_speed = 12 := by norm_num [downstream_speed]
  have h_upstream_speed : upstream_speed = 4 := by norm_num [upstream_speed]
  rw [h_downstream_speed, h_upstream_speed]
  norm_num
  sorry

end round_trip_time_l109_109110


namespace arithmetic_sequence_sixth_term_l109_109488

variables (a d : ℤ)

theorem arithmetic_sequence_sixth_term :
  a + (a + d) + (a + 2 * d) = 12 →
  a + 3 * d = 0 →
  a + 5 * d = -4 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sixth_term_l109_109488


namespace number_of_students_only_in_science_l109_109240

variable (T S D : ℕ)
variable (students : Set ℕ)
variable (science drama : Set ℕ)

theorem number_of_students_only_in_science
  (H1 : T = 88)
  (H2 : S = 75)
  (H3 : D = 65)
  (H4 : students = { x | x ∈ science ∪ drama })
  (H5 : ∀ x, x ∈ students) :
  75 - (75 + 65 - 88) = 23 :=
by
  have H6 : |S ∪ D| := T - |S ∩ D|, sorry

end number_of_students_only_in_science_l109_109240


namespace geometric_sequence_common_ratio_l109_109029

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℤ) 
  (q : ℤ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 1 / 4) : 
  q = 1 / 2 :=
  sorry

end geometric_sequence_common_ratio_l109_109029


namespace perpendiculars_from_incircle_centers_intersect_at_single_point_l109_109145

theorem perpendiculars_from_incircle_centers_intersect_at_single_point 
  (A B C P : Point)
  (h_equilateral : equilateral_triangle A B C) :
  ∃ (Q : Point), 
    (is_perpendicular (incircle_center P B C) Q B C) ∧
    (is_perpendicular (incircle_center P C A) Q C A) ∧
    (is_perpendicular (incircle_center P A B) Q A B) := 
sorry

end perpendiculars_from_incircle_centers_intersect_at_single_point_l109_109145


namespace expression_equality_l109_109247

theorem expression_equality:
  (Real.sqrt (1 / 4) - Real.sqrt3 (1 / 8) + Real.sqrt 81 + Real.abs (Real.sqrt 2 - 3) = 12 - Real.sqrt 2) :=
by
  sorry

end expression_equality_l109_109247


namespace smallest_possible_value_of_N_l109_109961

-- Define the dimensions of the block
variables (l m n : ℕ) 

-- Define the condition that the product of dimensions minus one is 143
def hidden_cubes_count (l m n : ℕ) : Prop := (l - 1) * (m - 1) * (n - 1) = 143

-- Define the total number of cubes in the outer block
def total_cubes (l m n : ℕ) : ℕ := l * m * n

-- The final proof statement
theorem smallest_possible_value_of_N : 
  ∃ (l m n : ℕ), hidden_cubes_count l m n → N = total_cubes l m n → N = 336 :=
sorry

end smallest_possible_value_of_N_l109_109961


namespace negation_of_proposition_l109_109003

open Real

theorem negation_of_proposition : (¬ (∀ x : ℝ, cos x ≤ 1)) ↔ (∃ x : ℝ, cos x > 1) := by
  sorry

end negation_of_proposition_l109_109003


namespace sixty_th_number_is_8_l109_109609

theorem sixty_th_number_is_8 :
  (∃ n k : ℕ, 1 ≤ n ∧ k = 60 ∧ (∑ i in range n.succ, 4 * i^2) + 4 * n^2 = k) → 8 := sorry

end sixty_th_number_is_8_l109_109609


namespace hamza_bucket_problem_l109_109355

theorem hamza_bucket_problem :
  let bucket_sizes := (3, 5, 6)
  let initial_full_bucket := 5
  let transfer_to_small_bucket := min initial_full_bucket 3
  let remaining_in_full_bucket := initial_full_bucket - transfer_to_small_bucket
  let transferred_to_large_bucket := remaining_in_full_bucket
  let total_large_bucket_capacity := 6
  total_large_bucket_capacity - transferred_to_large_bucket = 4 := by
  let bucket_sizes := (3, 5, 6)
  let initial_full_bucket := 5
  let transfer_to_small_bucket := min initial_full_bucket 3
  let remaining_in_full_bucket := initial_full_bucket - transfer_to_small_bucket
  let transferred_to_large_bucket := remaining_in_full_bucket
  let total_large_bucket_capacity := 6
  show total_large_bucket_capacity - transferred_to_large_bucket = 4 from sorry

end hamza_bucket_problem_l109_109355


namespace correct_propositions_count_l109_109582

-- Definitions of propositions according to given conditions
def prop1 (A B : α) : Prop := AB ⊂ α
def prop2 (A : α) (β : Type) : Prop := ∃ m : Line, (A ∈ m) ∧ (∀ x : β, x ∈ α → x ∈ β → x ∈ m)
def prop3 : Prop := ∀ A B C : Point, ∃! P : Plane, (A ∈ P) ∧ (B ∈ P) ∧ (C ∈ P)
def prop4 (a b c : Vector) : Prop := (a ⊥ b) ∧ (c ⊥ b) → (a ∥ c)

-- The main statement which states that exactly two of the propositions are true
theorem correct_propositions_count : 
  (^prop1 (A : α) (B : α) ∧ prop2 (A : α) (β : Type) ∧ ¬ prop3 ∧ ¬ prop4 (a : Vector) (b : Vector) (c : Vector)) = 2 := 
sorry

end correct_propositions_count_l109_109582


namespace largest_possible_value_of_n_l109_109123

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def largest_product : ℕ :=
  705

theorem largest_possible_value_of_n :
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧
  is_prime x ∧ is_prime y ∧
  is_prime (10 * y - x) ∧
  largest_product = x * y * (10 * y - x) :=
by
  sorry

end largest_possible_value_of_n_l109_109123


namespace line_through_intersection_and_origin_l109_109867

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := 2023 * x - 2022 * y - 1 = 0
def line2 (x y : ℝ) : Prop := 2022 * x + 2023 * y + 1 = 0

-- Define the line passing through the origin
def line_pass_origin (x y : ℝ) : Prop := 4045 * x + y = 0

-- Define the intersection point of the two lines
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the theorem stating the desired property
theorem line_through_intersection_and_origin (x y : ℝ)
    (h1 : intersection x y)
    (h2 : x = 0 ∧ y = 0) :
    line_pass_origin x y :=
by
    sorry

end line_through_intersection_and_origin_l109_109867


namespace price_per_glass_first_day_l109_109446

theorem price_per_glass_first_day (O P2 P1: ℝ) (H1 : O > 0) (H2 : P2 = 0.2) (H3 : 2 * O * P1 = 3 * O * P2) : P1 = 0.3 :=
by
  sorry

end price_per_glass_first_day_l109_109446


namespace minimum_a_l109_109705

variable (a x : ℝ)

theorem minimum_a {a : ℤ} (h₁ : x < a) (h₂ : 2 * x + 3 > 0) (h₃ : ∃ l : list ℤ, l.length ≥ 4 ∧ ∀ x ∈ l, -1 < x ∧ x < a) : a ≥ 3 :=
by sorry

end minimum_a_l109_109705


namespace club_truncator_more_wins_than_losses_l109_109991

noncomputable def club_truncator_probability : ℚ :=
  let total_games := 8
  let win_prob := 1/3
  let lose_prob := 1/3
  let tie_prob := 1/3
  -- The probability given by the solution
  let final_probability := 2741 / 6561
  final_probability

theorem club_truncator_more_wins_than_losses :
  club_truncator_probability = 2741 / 6561 :=
sorry

end club_truncator_more_wins_than_losses_l109_109991


namespace solution_l109_109433

def g₁ (x : ℝ) : ℝ := (1 / 2) - (4 / (4 * x + 2))

def g : ℕ → ℝ → ℝ
| 0, x       => x
| (n + 1), x => g₁ (g n x)

theorem solution (x : ℝ) :
  g 1003 x = x - 4 ↔ x = 11 / 2 :=
by
  sorry

end solution_l109_109433


namespace identify_counterfeit_coin_l109_109577

theorem identify_counterfeit_coin (
    n : ℕ, h : ∃ k, n = 26 ∧ k < n ∧ ∀ i < n, (∀ j, i ≠ j → k ≠ j)
) :
    ∃ (steps : ℕ), steps = 3 ∧
        (∀ (s : set ℕ), s ⊆ finset.range 26 → ∃! (c : ℕ), c ∈ s ∧
            (∀ x : ℕ, x ∈ s → balance (c, x) = lighter)) := 
sorry

end identify_counterfeit_coin_l109_109577


namespace simplify_expression_l109_109851

theorem simplify_expression :
  64^(1/4) - 144^(1/4) = 2 * Real.sqrt 2 - 2 * Real.sqrt 3 := 
by
  sorry

end simplify_expression_l109_109851


namespace product_of_three_consecutive_integers_is_square_l109_109616

theorem product_of_three_consecutive_integers_is_square (x : ℤ) : 
  ∃ n : ℤ, x * (x + 1) * (x + 2) = n^2 → x = 0 ∨ x = -1 ∨ x = -2 :=
by
  sorry

end product_of_three_consecutive_integers_is_square_l109_109616


namespace equation_of_pq_l109_109367

-- Define the circle x^2 + y^2 = 9
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the midpoint of PQ as (1, 2)
def midpoint (x y : ℝ) : Prop := x = 1 ∧ y = 2

-- Define the condition that PQ is a chord and the midpoint of PQ is (1, 2)
def is_chord (x y : ℝ) (L : ℝ → ℝ) : Prop := circle_eq x y ∧ midpoint x y ∧ L(x) = - (1 / 2) * x + (5 / 2)

-- State the main theorem to prove
theorem equation_of_pq (x y : ℝ) (L : ℝ → ℝ) (h : is_chord x y L) : L = λ x, - (1 / 2) * x + (5 / 2) :=
sorry

end equation_of_pq_l109_109367


namespace eval_expression_l109_109911

theorem eval_expression : 6 + 15 / 3 - 4^2 + 1 = -4 := by
  sorry

end eval_expression_l109_109911


namespace find_coordinates_of_N_l109_109686

def point := ℝ × ℝ

noncomputable def M : point := (2, -4)
noncomputable def lengthMN : ℝ := 5
noncomputable def is_parallel_to_x_axis (M N : point) : Prop := M.snd = N.snd

theorem find_coordinates_of_N (N : point) (h1 : is_parallel_to_x_axis M N) (h2 : (N.fst - M.fst).abs = lengthMN) : 
    N = (-3, -4) ∨ N = (7, -4) := 
by {
    sorry
}

end find_coordinates_of_N_l109_109686


namespace sum_in_base_5_correct_l109_109985

-- Definitions
def decSum := 122 + 78 -- Calculate the sum in decimal

-- Conversion functions
def to_base_5(n: ℕ) : List ℕ :=
  let rec convert (num : ℕ) (base : ℕ) (acc : List ℕ) :=
    if num < base then num :: acc
    else convert (num / base) base ((num % base) :: acc)
  convert n 5 []

noncomputable def base_5_122 := to_base_5 122
noncomputable def base_5_78 := to_base_5 78
noncomputable def base_5_200 := to_base_5 200

-- Check Result
theorem sum_in_base_5_correct : to_base_5 decSum = [1, 3, 0, 0] := by
  sorry

end sum_in_base_5_correct_l109_109985


namespace trajectory_of_P_l109_109342

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x = 15

structure Point :=
  (x : ℝ)
  (y : ℝ)

def moving_point_on_circle (M : Point) : Prop :=
  equation_of_circle M.x M.y

def N_fixed_point : Point :=
  { x := 1, y := 0 }

theorem trajectory_of_P (P : Point) : 
  (∃ M : Point, moving_point_on_circle M ∧ 
    (perpendicular_bisector_intersects P M N_fixed_point ∧ 
     intersects_CM P M)) →
  (P.x^2 / 4 + P.y^2 / 3 = 1) :=
sorry

-- Auxiliary Definitions: Assume these definitions exist in the library or can be defined.
def perpendicular_bisector_intersects (P M N : Point) : Prop := sorry
def intersects_CM (P M : Point) : Prop := sorry

end trajectory_of_P_l109_109342


namespace calculate_expression_l109_109244

theorem calculate_expression:
  500 * 4020 * 0.0402 * 20 = 1616064000 := by
  sorry

end calculate_expression_l109_109244


namespace xy_condition_l109_109720

variable (x y : ℝ) -- This depends on the problem context specifying real numbers.

theorem xy_condition (h : x ≠ 0 ∧ y ≠ 0) : (x + y = 0 ↔ y / x + x / y = -2) :=
  sorry

end xy_condition_l109_109720


namespace option_c_is_always_odd_l109_109371

theorem option_c_is_always_odd (n : ℤ) : ∃ (q : ℤ), n^2 + n + 5 = 2*q + 1 := by
  sorry

end option_c_is_always_odd_l109_109371


namespace roots_relationship_l109_109728

theorem roots_relationship (x1 x2 : ℝ) :
  is_root (λ x : ℝ, 3 * x^2 - 5 * x - 7) x1 →
  is_root (λ x : ℝ, 3 * x^2 - 5 * x - 7) x2 →
  (x1 + x2 = 5 / 3) ∧ (x1 * x2 = -7 / 3) :=
by
  intros h1 h2
  sorry

end roots_relationship_l109_109728


namespace angle_bisector_exists_l109_109076

theorem angle_bisector_exists (l1 l2 : Line) (vertex : Point) (P : Plane) :
  vertex ∉ P ∧ ∃ x : Plane, P = x → (x ∩ l1) ≠ ∅ ∧ (x ∩ l2) ≠ ∅ →
  ∃ (O : Point), O ∈ P →
  ∃ (k : ℝ) (h : k > 0),
  let Q := homothety_transform l1 l2 vertex O k in
  let bisector := angle_bisector Q.1 Q.2 in
  bisector_part_exists (inverse_homothety_transform bisector) :=
sorry

end angle_bisector_exists_l109_109076


namespace time_taken_l109_109154

def time_to_pass (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := length1 + length2
  total_distance / relative_speed

noncomputable def length1 : ℝ := 300
noncomputable def speed1 : ℝ := 40
noncomputable def length2 : ℝ := 100
noncomputable def speed2 : ℝ := 15

theorem time_taken :
  time_to_pass length1 length2 speed1 speed2 = 26.18 :=
by
  -- Just stating the theorem, proof is a task for a later step.
  sorry

end time_taken_l109_109154


namespace smallest_n_divisible_by_57_l109_109670

-- Definitions for the conditions
def is_positive_integer (n : ℕ) : Prop :=
  n > 0

def divisible_by_57 (f : ℕ → ℕ) (n : ℕ) : Prop :=
  57 ∣ f n

-- Proof of the problem statement
theorem smallest_n_divisible_by_57 :
  ∃ (n : ℕ), is_positive_integer n ∧ divisible_by_57 (λ n, 7 ^ n + 2 * n) n ∧
  ∀ (m : ℕ), is_positive_integer m ∧ divisible_by_57 (λ n, 7 ^ n + 2 * n) m → n ≤ m :=
begin
  sorry
end

end smallest_n_divisible_by_57_l109_109670


namespace percentage_spent_on_repairs_l109_109586

def cost := 5500
def repairs := 500
def profit := 1100
def profit_percentage := 0.20

theorem percentage_spent_on_repairs : (repairs / cost) * 100 = 9.09 := by
  sorry

end percentage_spent_on_repairs_l109_109586


namespace Luca_milk_water_needed_l109_109900

def LucaMilk (flour : ℕ) : ℕ := (flour / 250) * 50
def LucaWater (flour : ℕ) : ℕ := (flour / 250) * 30

theorem Luca_milk_water_needed (flour : ℕ) (h : flour = 1250) : LucaMilk flour = 250 ∧ LucaWater flour = 150 := by
  rw [h]
  sorry

end Luca_milk_water_needed_l109_109900


namespace expression_for_m_l109_109647

-- Definitions from the conditions
variable {a m n d : ℝ}
variable (log_a_m : real.log a m = c - 2 * real.log a n)
variable (c_def : c = real.log a (a^d))

-- Goal: Prove that m = a^d / n^2 given the conditions
theorem expression_for_m (h1 : log_a_m) (h2 : c_def) :
  m = a^d / n^2 :=
sorry

end expression_for_m_l109_109647


namespace unique_3_coloring_edges_count_l109_109085

open GraphTheory

variable (G : SimpleGraph V) [Fintype V]
variable (n : ℕ) (h : 3 ≤ n) (unique_3_coloring : ∀ (c₁ c₂ : V → ℕ), (∀ v, c₁ v < 3 ∧ c₂ v < 3) ∧ G.ProperColoring c₁ ∧ G.ProperColoring c₂ → c₁ = c₂)

theorem unique_3_coloring_edges_count :
    (Fintype.card V ≥ 3 ∧ unique_3_coloring) → G.edgeCount ≥ 2 * Fintype.card V - 3 :=
sorry

end unique_3_coloring_edges_count_l109_109085


namespace solution_set_of_inequality_l109_109886

-- Definitions for the problem
def inequality (x : ℝ) : Prop := (1 + x) * (2 - x) * (3 + x^2) > 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l109_109886


namespace work_modes_growth_l109_109731

theorem work_modes_growth (p1980_home p1990_home p2000_home p2010_home : ℕ)
                          (p1980_part p1990_part p2000_part p2010_part : ℕ)
                          (h1980_home : p1980_home = 3)
                          (h1990_home : p1990_home = 16)
                          (h2000_home : p2000_home = 26)
                          (h2010_home : p2010_home = 40)
                          (h1980_part : p1980_part = 10)
                          (h1990_part : p1990_part = 20)
                          (h2000_part : p2000_part = 35)
                          (h2010_part : p2010_part = 45) :
    ∃ g : Graph, accurate_representation g :=
by {
  -- data
  let data_home := [(1980, p1980_home), (1990, p1990_home), (2000, p2000_home), (2010, p2010_home)],
  let data_part := [(1980, p1980_part), (1990, p1990_part), (2000, p2000_part), (2010, p2010_part)],
  
  -- define weighted percentage increment trends
  let trend_home := [ (1990, p1990_home - p1980_home), (2000, p2000_home - p1990_home), (2010, p2010_home - p2000_home) ],
  let trend_part := [ (1990, p1990_part - p1980_part), (2000, p2000_part - p1990_part), (2010, p2010_part - p2000_part) ],
  
  -- prove existence
  have hh : ∃ g, graph_property g data_home trend_home,
  have hp : ∃ g, graph_property g data_part trend_part,
  exact combine_graphs hh hp
}

end work_modes_growth_l109_109731


namespace trajectory_of_point_A_l109_109326

theorem trajectory_of_point_A (m : ℝ) (A B C : ℝ × ℝ) (hBC : B = (-1, 0) ∧ C = (1, 0)) (hBC_dist : dist B C = 2)
  (hRatio : dist A B / dist A C = m) :
  (m = 1 → ∀ x y : ℝ, A = (x, y) → x = 0) ∧
  (m = 0 → ∀ x y : ℝ, A = (x, y) → x^2 + y^2 - 2 * x + 1 = 0) ∧
  (m ≠ 0 ∧ m ≠ 1 → ∀ x y : ℝ, A = (x, y) → (x + (1 + m^2) / (1 - m^2))^2 + y^2 = (2 * m / (1 - m^2))^2) := 
sorry

end trajectory_of_point_A_l109_109326


namespace water_to_be_poured_l109_109205

-- Definitions from the conditions
def four_cup := 4
def eight_cup := 8
def fraction (x : ℝ) (capacity : ℝ) : ℝ := x / capacity

-- Condition: 5.333333333333333 cups in 8-cup bottle
def cups_in_eight_cup_bottle : ℝ := 5.333333333333333

-- Derived fraction from the 8-cup bottle
def f : ℝ := fraction cups_in_eight_cup_bottle eight_cup

-- Amount of water in the 4-cup bottle
def cups_in_four_cup_bottle := f * four_cup

-- Total water in both bottles
def total_water := cups_in_eight_cup_bottle + cups_in_four_cup_bottle

-- The proof statement
theorem water_to_be_poured : total_water = 8 := by
  sorry

end water_to_be_poured_l109_109205


namespace expression_divisible_by_1968_l109_109459

theorem expression_divisible_by_1968 (n : ℕ) : 
  ( -1 ^ (2 * n) +  9 ^ (4 * n) - 6 ^ (8 * n) + 8 ^ (16 * n) ) % 1968 = 0 :=
by
  sorry

end expression_divisible_by_1968_l109_109459


namespace front_view_correct_l109_109249

def grid : List (List Nat) :=
  [[2, 1, 2, 4], [3, 2, 3, 3], [1, 3, 1, 2]]

def max_height_in_columns (grid : List (List Nat)) : List Nat :=
  List.map List.maximum grid.transpose

theorem front_view_correct :
  max_height_in_columns grid = [3, 3, 3, 4] :=
by
  sorry

end front_view_correct_l109_109249


namespace number_of_cats_l109_109211

def cats_on_ship (C S : ℕ) : Prop :=
  (C + S + 2 = 16) ∧ (4 * C + 2 * S + 3 = 45)

theorem number_of_cats (C S : ℕ) (h : cats_on_ship C S) : C = 7 :=
by
  sorry

end number_of_cats_l109_109211


namespace ratio_time_A_B_l109_109223

theorem ratio_time_A_B (B : ℕ) (A : ℕ) (hB : B = 30) (h_work : 1/A + 1/B = 1/5) : A/B = 1/5 :=
by
  subst hB
  have h1 : 1/A = 1/5 - 1/30 := sorry  -- Step for detailed proof
  have h2 : 1/A = 1/6 := sorry         -- Step for detailed proof
  have h3 : A = 6 := sorry             -- Solving for A
  rw [h3, hB, Nat.div] 
  norm_num

end ratio_time_A_B_l109_109223


namespace geometric_mean_l109_109998

open Real

/--
Given points A, B, C, D, E on a line with distances:
1. AB = a,
2. AC = b,
3. BD = b,
4. From points D and C, draw arcs with radius b intersecting at point E,
prove that AE is the geometric mean of a and b.
-/
theorem geometric_mean (a b : ℝ) (A B C D E : ℝ → Prop)
  (hAB : dist A B = a) 
  (hAC : dist A C = b) 
  (hBD : dist B D = b) 
  (hArcs : ∃ r, r = b ∧ dist D E = r ∧ dist C E = r) :
  dist A E = sqrt (a * b) :=
  sorry

end geometric_mean_l109_109998


namespace max_dominoes_l109_109474

-- Define the problem data:
-- The figure as a grid with alternating coloring (think of it as NxM chessboard pattern)
def figure : Type := sorry -- You can define your geometric figure in terms of grid or other data structure

-- Define what it means to be a valid 1x2 rectangle covering within the grid
def valid_domino_placement (f: figure) (x y: nat) : Prop := sorry -- This would assert that the placement of 1x2 covers two aligned unit squares

-- Each domino must cover one black and one white square.
def domino_covers_black_white (f: figure) (x y: nat) : Prop := sorry -- Check alternating coloring properties

-- Counting black squares around the figure implies we have certain constraints
def number_of_black_squares (f: figure) : nat :=
  5  -- This is given in problem

-- Maximum number of 1x2 dominos that fit into the grid without overlap or gap
theorem max_dominoes (f: figure) (h1: (number_of_black_squares f = 5)) : 
  ∃ (n: nat), (n ≤ 5) := 
sorry

end max_dominoes_l109_109474


namespace largest_possible_value_of_k_l109_109125

theorem largest_possible_value_of_k :
  ∃ (p : ℕ → polynomial ℝ), 
  (x^12 - (1 : polynomial ℝ)) = p 0 * p 1 * p 2 * p 3 * p 4 * p 5 ∧
  ∀ i, 0 ≤ i ∧ i < 6 → ¬ is_constant (p i) :=
sorry

end largest_possible_value_of_k_l109_109125


namespace octahedron_sum_l109_109826

def sum_numbers := (list.range 8).sum + 8  -- sum of the numbers 1 to 8

def multiply_sum (n : ℕ) := n * 4  -- each number appearing 4 times

theorem octahedron_sum : multiply_sum sum_numbers = 144 := by
  unfold sum_numbers
  unfold multiply_sum
  norm_num
  sorry

end octahedron_sum_l109_109826


namespace max_distance_to_line_l_l109_109408

noncomputable def C1_param (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sqrt 3 * Real.sin β)

noncomputable def C_param (α : ℝ) : ℝ × ℝ := (-2 + Real.cos α, -1 + Real.sin α)

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def distance_to_line (M : ℝ × ℝ) : ℝ :=
|M.1 - M.2| / Real.sqrt 2

theorem max_distance_to_line_l 
(β : ℝ) (hβ : β = 2 * Real.pi / 3) :
let Q : ℝ × ℝ := C1_param β in 
let P : ℝ × ℝ := C_param (Real.pi / 2) in 
let M : ℝ × ℝ := midpoint P Q in 
distance_to_line M = Real.sqrt 2 := 
by
  sorry

end max_distance_to_line_l_l109_109408


namespace length_of_DF_l109_109745

-- Definitions derived from the conditions
variable (A B C D E F : Point)
variable (AB DC BC : Line)
variable [Parallelogram ABCD]

def is_parallelogram (A B C D : Point) : Prop := 
  Parallelogram ABCD

def is_altitude (line : Line) (base : Line) (height : ℝ) : Prop :=
  -- Definition of altitude in Lean would be more complex
  -- Here it is simplified for the sake of the problem's context
  base ⊥ line

-- Given conditions
axiom h1 : is_parallelogram A B C D
axiom h2 : is_altitude DE AB 8
axiom h3 : is_altitude DF BC _
axiom h4 : length DC = 15
axiom h5 : length EB = 3
axiom h6 : length DE = 8

-- To be proved
theorem length_of_DF : length DF = 8 :=
by
  sorry

end length_of_DF_l109_109745


namespace hypotenuse_length_l109_109916

theorem hypotenuse_length (x y : ℝ)
  (h₁ : (1 / 3) * real.pi * y^2 * x = 1200 * real.pi)
  (h₂ : (1 / 3) * real.pi * x^2 * (2 * x) = 3840 * real.pi) :
  real.sqrt (x^2 + y^2) = 2 * real.sqrt 131 :=
by
  sorry

end hypotenuse_length_l109_109916


namespace inequality_prod_geometric_mean_l109_109063

noncomputable def geom_mean (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  Real.exp ((1 / n) * (Finset.univ.sum (λ i, Real.log (x i))))

theorem inequality_prod_geometric_mean (x : ℕ → ℝ) (n : ℕ) (hx : ∀ i, x i > 1) :
    let A := geom_mean n (λ i, x i) in
    (Finset.prod (Finset.range n) (λ i, (x i + 1) / (x i - 1))) ≥ ((A + 1) / (A - 1))^n :=
by
  sorry

end inequality_prod_geometric_mean_l109_109063


namespace line_through_two_points_l109_109180

theorem line_through_two_points (x_1 y_1 x_2 y_2 x y : ℝ) :
  (x - x_1) * (y_2 - y_1) = (y - y_1) * (x_2 - x_1) :=
sorry

end line_through_two_points_l109_109180


namespace min_avg_score_less_than_record_l109_109305

theorem min_avg_score_less_than_record
  (old_record_avg : ℝ := 287.5)
  (players : ℕ := 6)
  (rounds : ℕ := 12)
  (total_points_11_rounds : ℝ := 19350.5)
  (bonus_points_9_rounds : ℕ := 300) :
  ∀ final_round_avg : ℝ, (final_round_avg = (old_record_avg * players * rounds - total_points_11_rounds + bonus_points_9_rounds) / players) →
  old_record_avg - final_round_avg = 12.5833 :=
by {
  sorry
}

end min_avg_score_less_than_record_l109_109305


namespace choose_positions_from_8_people_l109_109743

theorem choose_positions_from_8_people : 
  ∃ (ways : ℕ), ways = 8 * 7 * 6 := 
sorry

end choose_positions_from_8_people_l109_109743


namespace cannot_reach_all_points_in_segment_l109_109078

def point_on_OX (a : ℝ) : Prop :=
  ∃ x : ℝ, a ≤ x ∧ x ≤ a + 1

def can_reach_all_points_on_segment (a : ℝ) (reach_point : ℝ → Prop) : Prop :=
  ∀ x : ℝ, point_on_OX a x → reach_point x

theorem cannot_reach_all_points_in_segment (a : ℝ) : 
  ¬ can_reach_all_points_on_segment a (λ x, ∃ n : ℕ, ∃ m : ℕ, x = a + m / 10^n) :=
by
  sorry

end cannot_reach_all_points_in_segment_l109_109078


namespace mid_point_congruence_l109_109082

open Classical

variables {A B C D E : Type*} [InnerProductSpace ℝ A]

-- Definitions of the conditions given in part a)
def is_midpoint (A C D : A) : Prop := dist A D = dist D C
def intersects (D B E C : A) : Prop := lies_on_line D B E ∧ lies_on_line D C E

-- Main theorem statement
theorem mid_point_congruence (A B C D E : A) (h1 : is_midpoint A C D)
  (h2 : intersects D B E C) (h3 : dist (dist B D) (dist B E) = dist (dist A E) (dist E C)) : dist B E = dist B C :=
sorry

end mid_point_congruence_l109_109082


namespace problem_statement_l109_109364

def contains_digit_3 (n : ℕ) : Prop :=
  n.digits 10.contains 3

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def in_range (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 499

theorem problem_statement : 
  ∃ (count : ℕ), count = (finset.filter (λ n, in_range n ∧ contains_digit_3 n ∧ divisible_by_5 n) (finset.Icc 200 499)).card ∧ count = 12 :=
begin
  sorry
end

end problem_statement_l109_109364


namespace max_daily_sales_profit_l109_109559

-- Define the conditions
def cost_price : ℝ := 30
def sales_price (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 30 then 0.5 * x + 35
  else if 31 ≤ x ∧ x ≤ 60 then 50
  else 0

def daily_sales_volume (x : ℕ) : ℝ :=
  124 - 2 * x

-- Define the profit function w(x)
def daily_sales_profit (x : ℕ) : ℝ :=
  (sales_price x - cost_price) * daily_sales_volume x

-- Define the main Lean 4 statement to prove
theorem max_daily_sales_profit :
∀ x : ℕ, 1 ≤ x ∧ x ≤ 60 →
  (daily_sales_profit x = 
    (if 1 ≤ x ∧ x ≤ 30 then -x^2 + 52*x + 620
     else if 31 ≤ x ∧ x ≤ 60 then -40*x + 2480
     else 0)) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 60 → daily_sales_profit 26 ≥ daily_sales_profit y) :=
by
  intros x h1 hx
  sorry

end max_daily_sales_profit_l109_109559


namespace find_EC_length_l109_109327

variables (A B C D E : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (a b c d e : A)

-- Given conditions
variables (AC : Segment a c)
variables (BD : Segment b d)
variables (AE : Segment a e) (AB : Segment a b) (DC : Segment d c) (AD : Segment a d) (BE : Segment b e)
variables (intersect : Intersect AC BD e)
variables (AE_shorter_AB : dist a e = dist a b - 1)
variables (AE_eq_DC : dist a e = dist d c)
variables (AD_eq_BE : dist a d = dist b e)
variables (angle_ADC_eq_DEC : ∠ a d c = ∠ d e c)

-- Proof goal
theorem find_EC_length : dist e c = 1 :=
sorry

end find_EC_length_l109_109327


namespace valid_sandwiches_l109_109112

def types_of_bread := 5
def types_of_meat := 6
def types_of_cheese := 6

def forbidden_combinations : List (Nat × Nat) :=
  [(0, 0), -- turkey/swiss
   (4, 1), -- rye/beef
   (4, 0)] -- rye/turkey

def total_sandwiches := types_of_bread * types_of_meat * types_of_cheese

def count_forbidden_sandwiches : Nat :=
  5 + -- turkey/swiss
  6 + -- rye/beef
  6 + -- rye/turkey
  -1  -- rye/turkey/swiss counted twice

theorem valid_sandwiches : total_sandwiches - count_forbidden_sandwiches = 164 := by
  sorry

end valid_sandwiches_l109_109112


namespace count_two_digit_numbers_with_one_even_digit_is_14_l109_109235

def is_even_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 4

def is_valid_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def count_valid_two_digit_numbers_with_one_even_digit : ℕ :=
  let digits := [1, 2, 3, 4, 5]
  let count := list.sum ($
    digits.product digits
    .filter (λ ⟨d1, d2⟩ => (is_even_digit d1 ∧ ¬is_even_digit d2) ∨ 
                           (¬is_even_digit d1 ∧ is_even_digit d2)))
  count

theorem count_two_digit_numbers_with_one_even_digit_is_14 : 
  count_valid_two_digit_numbers_with_one_even_digit = 14 :=
sorry

end count_two_digit_numbers_with_one_even_digit_is_14_l109_109235


namespace integer_solution_count_l109_109455

theorem integer_solution_count {a b c d : ℤ} (h : a ≠ b) :
  (∀ x y : ℤ, (x + a * y + c) * (x + b * y + d) = 2 →
    ∃ a b : ℤ, (|a - b| = 1 ∨ (|a - b| = 2 ∧ (d - c) % 2 = 1))) :=
sorry

end integer_solution_count_l109_109455


namespace find_number_written_on_board_l109_109490

theorem find_number_written_on_board (n k r : ℤ) 
    (h_k : k = 3 * n + 7) 
    (h_r : r = 7 * n + 3) 
    (h_diff : r = k + 84) : 
    n = 22 := 
begin
    sorry
end

end find_number_written_on_board_l109_109490


namespace necessary_and_sufficient_condition_l109_109648

variable (a b : ℝ)

theorem necessary_and_sufficient_condition:
  (ab + 1 ≠ a + b) ↔ (a ≠ 1 ∧ b ≠ 1) :=
sorry

end necessary_and_sufficient_condition_l109_109648


namespace sum_of_solutions_l109_109778

theorem sum_of_solutions :
  let solutions := [(-8, -2), (-1, 5), (10, 4), (10, 4)],
  (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 :=
by
  let solutions : List (ℤ × ℤ) := [(-8, -2), (-1, 5), (10, 4), (10, 4)]
  have h1 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 4| = |y - 10| := sorry
  have h2 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 10| = 3 * |y - 4| := sorry
  have solution_sum : (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 := by
    simp [solutions]
    norm_num
  exact solution_sum

end sum_of_solutions_l109_109778


namespace gcf_120_180_300_l109_109509

theorem gcf_120_180_300 : Nat.gcd (Nat.gcd 120 180) 300 = 60 := 
by eval_gcd 120 180 300

end gcf_120_180_300_l109_109509


namespace tunnel_build_equation_l109_109561

theorem tunnel_build_equation (x : ℝ) (h1 : 1280 > 0) (h2 : x > 0) : 
  (1280 - x) / x = (1280 - x) / (1.4 * x) + 2 := 
by
  sorry

end tunnel_build_equation_l109_109561


namespace problem_concurrency_l109_109079

open EuclideanGeometry

-- Define the circumcenter of a triangle
def circumcenter (A B C : Point) : Point := sorry

-- Define the cyclic quadrilateral
variables (A B C D E O O1 O2 : Point)
variables (h1 : CyclicQuad A B C D) -- ABCD is cyclic quadrilateral
variables (h2 : E ∈ Line A C) -- E is on diagonal AC
variables (h3 : ∠ A B E = ∠ C B D) -- ∠ABE = ∠CBD  

-- Define the circumcenters of the triangles
variables (hO : O = circumcenter A B C)
variables (hO1 : O1 = circumcenter A B E)
variables (hO2 : O2 = circumcenter C B E)

-- The theorem to prove the concurrency of the lines
theorem problem_concurrency (h1 : CyclicQuad A B C D)
    (h2 : E ∈ Line A C) 
    (h3 : ∠ A B E = ∠ C B D)
    (hO : O = circumcenter A B C)
    (hO1 : O1 = circumcenter A B E)
    (hO2 : O2 = circumcenter C B E) :
    Concurrent (Line D O) (Line A O1) (Line C O2) := 
by
  sorry

end problem_concurrency_l109_109079


namespace round_to_hundredth_l109_109176

theorem round_to_hundredth:
  (∀ (x : Float), x ∈ {34.561, 34.558, 34.5601, 34.56444} → Float.round (x * 100) / 100 = 34.56) ∧ 
  (Float.round (34.5539999 * 100) / 100 ≠ 34.56) :=
by
  sorry

end round_to_hundredth_l109_109176


namespace time_between_peanuts_l109_109155

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := 4
def flight_time_hours : ℕ := 2

theorem time_between_peanuts (peanuts_per_bag number_of_bags flight_time_hours : ℕ) (h1 : peanuts_per_bag = 30) (h2 : number_of_bags = 4) (h3 : flight_time_hours = 2) :
  (flight_time_hours * 60) / (peanuts_per_bag * number_of_bags) = 1 := by
  sorry

end time_between_peanuts_l109_109155


namespace imaginary_part_of_complex_z_l109_109871

def complex (a b : ℝ) : ℂ := ⟨a, b⟩
instance : Coe ℂ (ℝ × ℝ) := ⟨λ z, (z.re, z.im)⟩

theorem imaginary_part_of_complex_z :
  let i := complex 0 1
  let z := (1 + i) ^ 2 * (2 + i)
  (z : ℝ × ℝ).snd = 4 := by
  -- Proof goes here
  sorry

end imaginary_part_of_complex_z_l109_109871


namespace no_intersection_tangent_graph_l109_109381

theorem no_intersection_tangent_graph (k : ℝ) (m : ℤ) : 
  (∀ x: ℝ, x = (k * Real.pi) / 2 → (¬ 4 * k ≠ 4 * m + 1)) → 
  (-1 ≤ k ∧ k ≤ 1) →
  (k = 1 / 4 ∨ k = -3 / 4) :=
sorry

end no_intersection_tangent_graph_l109_109381


namespace Olivia_paint_area_l109_109623

theorem Olivia_paint_area
  (length width height : ℕ) (door_window_area : ℕ) (bedrooms : ℕ)
  (h_length : length = 14) 
  (h_width : width = 11) 
  (h_height : height = 9) 
  (h_door_window_area : door_window_area = 70) 
  (h_bedrooms : bedrooms = 4) :
  (2 * (length * height) + 2 * (width * height) - door_window_area) * bedrooms = 1520 :=
by
  sorry

end Olivia_paint_area_l109_109623


namespace problem_solution_l109_109132

theorem problem_solution (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 500 / 9 :=
by
  sorry

end problem_solution_l109_109132


namespace positive_difference_of_perimeters_l109_109148

-- Defining the dimensions and conditions
def rect1_length : ℕ := 3
def rect1_width : ℕ := 6
def rect2_length : ℕ := 2
def rect2_width : ℕ := 7
def extra_square_side : ℕ := 1

-- Calculate the perimeters
def perimeter_rect1 : ℕ := 2 * (rect1_length + rect1_width)
def perimeter_rect2_with_square : ℕ := 2 * (rect2_length + rect2_width) + 3 * extra_square_side

-- Lean theorem statement
theorem positive_difference_of_perimeters :
  abs (perimeter_rect1 - perimeter_rect2_with_square) = 3 :=
by 
  sorry

end positive_difference_of_perimeters_l109_109148


namespace largest_four_digit_number_divisible_by_5_l109_109516

theorem largest_four_digit_number_divisible_by_5 : ∃ n : ℕ, n < 10000 ∧ n % 5 = 0 ∧ ∀ m : ℕ, m < 10000 ∧ m % 5 = 0 → m ≤ n :=
begin
  use 9995,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m h1 h2,
    have : m ≤ 9999 := h1,
    nlinarith,
    sorry
  }
end

end largest_four_digit_number_divisible_by_5_l109_109516


namespace toy_distribution_ratio_l109_109462

theorem toy_distribution_ratio (total_toys friends : ℕ) 
  (hc : total_toys = 118) 
  (hf : friends = 4) : 
  (29 / 118 : ℚ) = 1 / 4 :=
by
  have h : 118 = 4 * 29 + 2, by norm_num
  have hc : 29 / 118 = (29 / 29) / (118 / 29), by exact rat.cast_div 29 118
  have hr : 29 / 118 = 1 / 4, by norm_num
  exact hr

end toy_distribution_ratio_l109_109462


namespace area_of_region_Q_inside_strip_outside_triangle_l109_109425

theorem area_of_region_Q_inside_strip_outside_triangle (A B C D F : Point) 
  (h_square : is_unit_square A B C D) 
  (h_F_on_diagonal : F ∈ diagonal B C)
  (h_triangle_isosceles : is_isosceles_right_triangle A B F (hypotenuse AB))
  (h_strip : y_range := (1/4, 3/4)) : 
  area_of_region_Q (strip y_range) (outside_triangle (triangle A B F)) = 1/2 :=
sorry

end area_of_region_Q_inside_strip_outside_triangle_l109_109425


namespace kolakoski_13_to_20_l109_109495

def isKolakoski (seq : List ℕ) : Prop :=
  seq = ([1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2].take (seq.length)) ∧
  ∀ (n : ℕ), seq.take n.groupAdj.length == seq.take n

theorem kolakoski_13_to_20 : 
  ∃ seq : List ℕ, isKolakoski seq ∧ seq.drop 12.take 8 = [1, 1, 2, 1, 1, 2, 2, 1] :=
by
  sorry

end kolakoski_13_to_20_l109_109495


namespace correct_inverse_g_l109_109118

variables {X Y Z W : Type} 
  (s : X → Y) (t : Z → X) (u : W → Z)
  (s_inv : Y → X) (t_inv : X → Z) (u_inv : Z → W)

-- Assuming s, t, u are invertible functions with their respective inverses
noncomputable def s_invertible : (X → Y) := s
lemma s_left_inv : ∀ x, s_inv (s x) = x := sorry
lemma s_right_inv : ∀ y, s (s_inv y) = y := sorry

noncomputable def t_invertible : (Z → X) := t
lemma t_left_inv : ∀ z, t_inv (t z) = z := sorry
lemma t_right_inv : ∀ x, t (t_inv x) = x := sorry

noncomputable def u_invertible : (W → Z) := u
lemma u_left_inv : ∀ w, u_inv (u w) = w := sorry
lemma u_right_inv : ∀ z, u (u_inv z) = z := sorry

-- g is defined as a composition of s, u, and t
def g (w : W) : Y := s (u (t w))

-- Proof that the inverse of g is t⁻¹ ∘ u⁻¹ ∘ s⁻¹
theorem correct_inverse_g : 
  (∀ y, (t_inv ∘ u_inv ∘ s_inv) (g y) = y) :=
sorry

end correct_inverse_g_l109_109118


namespace probability_of_digit_six_l109_109237

theorem probability_of_digit_six :
  let total_numbers := 90
  let favorable_numbers := 18
  0 < total_numbers ∧ 0 < favorable_numbers →
  (favorable_numbers / total_numbers : ℚ) = 1 / 5 :=
by
  intros total_numbers favorable_numbers h
  sorry

end probability_of_digit_six_l109_109237


namespace winning_vote_majority_l109_109395

theorem winning_vote_majority (h1 : 0.70 * 900 = 630)
                             (h2 : 0.30 * 900 = 270) :
  630 - 270 = 360 :=
by
  sorry

end winning_vote_majority_l109_109395


namespace gain_percent_is_25_l109_109725

theorem gain_percent_is_25 (C S : ℝ) (h : 50 * C = 40 * S) : (S - C) / C * 100 = 25 :=
  sorry

end gain_percent_is_25_l109_109725


namespace largest_consecutive_odd_integer_sum_l109_109893

theorem largest_consecutive_odd_integer_sum :
  ∃ N : ℤ, N + (N + 2) + (N + 4) = -147 ∧ (N + 4) = -47 :=
begin
  sorry
end

end largest_consecutive_odd_integer_sum_l109_109893


namespace problem_statement_l109_109789

theorem problem_statement (x : Fin 50 → ℝ)
  (h1 : ∑ i, x i = 0)
  (h2 : ∑ i, x i / (1 + x i) = 0) :
  ∑ i, x i^2 / (1 + x i) = 0 := 
  sorry

end problem_statement_l109_109789


namespace oleg_traveling_in_15th_wagon_l109_109908

noncomputable def oleg_in_fifteenth_wagon : Prop :=
  let train_length := 20
  let time_to_meet := 36
  let time_to_pass := 44
  let total_time := time_to_meet + time_to_pass
  let distance_covered := 2 * train_length
  let relative_speed := distance_covered / total_time
  let vova_wagon := 4
  let distance_covered_in_meeting := relative_speed * time_to_meet
  let oleg_wagon := vova_wagon + distance_covered_in_meeting := 
  18 --Oleg's wagon should actually be set to the difference 
  (train_length if total train distance + meeting time)

  oleg_wagon = 15

theorem oleg_traveling_in_15th_wagon : oleg_in_fifteenth_wagon := by 
sorry

end oleg_traveling_in_15th_wagon_l109_109908


namespace shortest_side_len_l109_109413

-- Definitions and conditions from the problem
variable (A B C D : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable [InCircle ABC D]
variable [Segment AD : Real] [Segment DC : Real]
variable [Radius r : Real]
variable (lenShortestSide : Real)

-- Given conditions
def AD_eq_4 : AD = 4 := sorry
def DC_eq_5 : DC = 5 := sorry
def radius_eq_3 : r = 3 := sorry

-- Proof statement
theorem shortest_side_len : lenShortestSide = 9 :=
  by
    -- Assume the given conditions
    assume h₁ : AD_eq_4,
    assume h₂ : DC_eq_5,
    assume h₃ : radius_eq_3,

    -- Add proof details here
    -- Using these conditions, we need to show that lenShortestSide = 9
    sorry

end shortest_side_len_l109_109413


namespace expected_length_after_2012_repetitions_l109_109531

noncomputable def expected_length_remaining (n : ℕ) := (11/18 : ℚ)^n

theorem expected_length_after_2012_repetitions :
  expected_length_remaining 2012 = (11 / 18 : ℚ) ^ 2012 :=
by
  sorry

end expected_length_after_2012_repetitions_l109_109531


namespace max_r_value_for_same_range_l109_109126

theorem max_r_value_for_same_range (r : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = 2 * x^2 - 3 * x + r ∧ ((range (λ x, y) = range (λ x, 2 * (2 * x^2 - 3 * x + r)^2 - 3 * (2 * x^2 - 3 * x + r) + r)))) → 
  r ≤ 15 / 8 :=
by {
  sorry
}

end max_r_value_for_same_range_l109_109126


namespace parabola_equation_l109_109281

def parabola_condition (x y : ℝ) (a b : ℝ) : Prop :=
  (y^2 = a * x) ∨ (x^2 = b * y)

def point_on_parabola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P = (4, -2) ∧ parabola_condition P.1 P.2 a b

theorem parabola_equation :
  ∃ (a b : ℝ), point_on_parabola (4, -2) a b :=
by
  existsi (1)
  existsi (-8)
  split
  · exact rfl
  · left
    exact rfl
    sorry

end parabola_equation_l109_109281


namespace f_is_odd_l109_109328

-- Definitions of functions and conditions from the problem
def F (x : ℝ) (f : ℝ → ℝ) : ℝ := (x^3 - 2 * x) * (f x)

-- Main statement we need to prove
theorem f_is_odd (f : ℝ → ℝ)
  (hF_even : ∀ x : ℝ, F x f = F (-x) f)
  (hf_nonzero : ∃ x : ℝ, f x ≠ 0) :
  ∀ x : ℝ, f (-x) = -f x :=
begin
  sorry
end

end f_is_odd_l109_109328


namespace find_all_x_in_range_20_50_l109_109276

open Int

theorem find_all_x_in_range_20_50 (x : ℤ) : 20 ≤ x → x ≤ 50 → 6 * x + 5 ≡ -19 [MOD 10] → 
  x = 21 ∨ x = 26 ∨ x = 31 ∨ x = 36 ∨ x = 41 ∨ x = 46 :=
by
  intros h1 h2 h3
  sorry

end find_all_x_in_range_20_50_l109_109276


namespace smallest_y_l109_109957

noncomputable def x : ℕ := 3 * 40 * 75

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k^3 = n

theorem smallest_y (y : ℕ) (hy : y = 3) :
  ∀ (x : ℕ), x = 3 * 40 * 75 → is_perfect_cube (x * y) :=
by
  intro x hx
  unfold is_perfect_cube
  exists 5 -- This is just a placeholder value; the proof would find the correct k
  sorry

end smallest_y_l109_109957


namespace no_fixed_points_range_l109_109286

theorem no_fixed_points_range (a : ℝ) (h : ¬ ∃ x : ℝ, f(x) = x) : a^2 - 2*a - 3 < 0 :=
by
  let f := λ (x : ℝ), x^2 + a*x + 1
  have h1 : ¬ ∃ x: ℝ, x^2 + (a-1)*x + 1 = 0,
    from h
  sorry

end no_fixed_points_range_l109_109286


namespace solve_for_n_l109_109097

theorem solve_for_n : ∃ n : ℝ, 0.03 * n + 0.08 * (20 + n) = 12.6 ∧ n = 100 :=
by
  use 100
  split
  · -- Proof that 0.03 * 100 + 0.08 * (20 + 100) = 12.6
    calc
      0.03 * 100 + 0.08 * (20 + 100)
        = 3.0 + 0.08 * 120 : by ring
    ... = 3.0 + 9.6 : by ring
    ... = 12.6 : by ring
  · -- Proof that n = 100
    rfl

end solve_for_n_l109_109097


namespace full_tank_cost_l109_109548

-- Definitions from the conditions
def total_liters_given := 36
def total_cost_given := 18
def tank_capacity := 64

-- Hypothesis based on the conditions
def price_per_liter := total_cost_given / total_liters_given

-- Conclusion we need to prove
theorem full_tank_cost: price_per_liter * tank_capacity = 32 :=
  sorry

end full_tank_cost_l109_109548


namespace all_three_items_fans_l109_109642

theorem all_three_items_fans 
  (h1 : ∀ n, n = 4800 % 80 → n = 0)
  (h2 : ∀ n, n = 4800 % 40 → n = 0)
  (h3 : ∀ n, n = 4800 % 60 → n = 0)
  (h4 : ∀ n, n = 4800):
  (∃ k, k = 20):=
by
  sorry

end all_three_items_fans_l109_109642


namespace line_canonical_form_l109_109179

theorem line_canonical_form :
  ∃ (x y z : ℝ),
  x + y + z - 2 = 0 ∧
  x - y - 2 * z + 2 = 0 →
  ∃ (k : ℝ),
  x / k = -1 ∧
  (y - 2) / (3 * k) = 1 ∧
  z / (-2 * k) = 1 :=
sorry

end line_canonical_form_l109_109179


namespace number_of_fandoms_l109_109498

open Real

noncomputable def shirt_price : ℝ := 15
noncomputable def discount_rate : ℝ := 0.20
noncomputable def discounted_shirt_price : ℝ := shirt_price * (1 - discount_rate)
noncomputable def shirts_per_fandom : ℝ := 5
noncomputable def tax_rate : ℝ := 0.10
noncomputable def total_paid_after_tax : ℝ := 264
noncomputable def total_cost_before_tax (n : ℝ) : ℝ := (discounted_shirt_price * shirts_per_fandom * n)

theorem number_of_fandoms :
  ∃ (n : ℝ), total_cost_before_tax n * (1 + tax_rate) = total_paid_after_tax ∧ n = 4 :=
begin
  -- No proof required
  sorry
end

end number_of_fandoms_l109_109498


namespace unit_vector_sum_magnitudes_l109_109332

-- Define that a0 and b0 are unit vectors
variables {a0 b0 : Vector (ℝ, 3)}

-- Assume the magnitudes of these unit vectors are 1
def unit_vector_a0 : Prop := abs (a0.norm) = 1
def unit_vector_b0 : Prop := abs (b0.norm) = 1

-- Prove that the sum of the magnitudes is 2
theorem unit_vector_sum_magnitudes (h1 : unit_vector_a0) (h2 : unit_vector_b0) :
  abs (a0.norm) + abs (b0.norm) = 2 :=
by
  sorry

end unit_vector_sum_magnitudes_l109_109332


namespace total_effective_diameter_correct_l109_109972

noncomputable def C1 := 6.28
noncomputable def C2 := 10.47
noncomputable def C3 := 18.85

noncomputable def r := 0.95
noncomputable def π := Real.pi

noncomputable def D (C : ℝ) : ℝ := C / π
noncomputable def ED (D : ℝ) : ℝ := D * r

noncomputable def D1 := D C1
noncomputable def D2 := D C2
noncomputable def D3 := D C3

noncomputable def ED1 := ED D1
noncomputable def ED2 := ED D2
noncomputable def ED3 := ED D3

noncomputable def totalEffectiveDiameter := ED1 + ED2 + ED3

theorem total_effective_diameter_correct :
  totalEffectiveDiameter ≈ 10.77 := by
  sorry

end total_effective_diameter_correct_l109_109972


namespace parabolas_intersection_l109_109150

theorem parabolas_intersection : 
  let y1 := 3 * (3 / 2 + sqrt 21 / 2)^2 - 12 * (3 / 2 + sqrt 21 / 2) + 4,
      y2 := 3 * (3 / 2 - sqrt 21 / 2)^2 - 12 * (3 / 2 - sqrt 21 / 2) + 4,
      parabola1 := λ x, 3 * x^2 - 12 * x + 4,
      parabola2 := λ x, x^2 - 6 * x + 10 in
    (parabola1 (3 / 2 + sqrt 21 / 2) = parabola2 (3 / 2 + sqrt 21 / 2)) ∧ 
    (parabola1 (3 / 2 - sqrt 21 / 2) = parabola2 (3 / 2 - sqrt 21 / 2)) :=
by
  let y1 := 3 * (3 /2 + sqrt 21 / 2)^2 - 12 * (3 / 2 + sqrt 21 / 2) + 4
  let y2 := 3 * (3 / 2 - sqrt 21 / 2)^2 - 12 * (3 / 2 - sqrt 21 / 2) + 4
  let parabola1 := λ x, 3 * x^2 - 12 * x + 4
  let parabola2 := λ x, x^2 - 6 * x + 10
  split
  sorry -- proof for first intersection
  sorry -- proof for second intersection

end parabolas_intersection_l109_109150


namespace range_of_f_l109_109880

def f (x : ℝ) : ℝ := 2 * Real.cos x + Real.sin (2 * x)

theorem range_of_f : 
  Set.range f = Set.Icc (-3 * Real.sqrt 3 / 2) (3 * Real.sqrt 3 / 2) := 
sorry

end range_of_f_l109_109880


namespace point_divides_segment_l109_109634

theorem point_divides_segment (x₁ y₁ x₂ y₂ m n : ℝ) (h₁ : (x₁, y₁) = (3, 7)) (h₂ : (x₂, y₂) = (5, 1)) (h₃ : m = 1) (h₄ : n = 3) :
  ( (m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n) ) = (3.5, 5.5) :=
by
  sorry

end point_divides_segment_l109_109634


namespace complex_number_solution_l109_109787

-- Define the given problem as a hypothesis and prove the relationship between a and b
theorem complex_number_solution (a b : ℝ) (h : (1 + 2 * complex.i) / (a + b * complex.i) = 1 + complex.i) :
  a = 3 / 2 ∧ b = 1 / 2 :=
by
  sorry

end complex_number_solution_l109_109787


namespace sufficient_not_necessary_condition_l109_109540

theorem sufficient_not_necessary_condition (x : ℤ) :
  (x = 1 → x^2 = 1) ∧ (x^2 = 1 → x = 1 ∨ x = -1) →
  (∀ y, (y = 1 → y^2 = 1) ∧ (y^2 = 1 → y = 1 ∨ y = -1) → "sufficient but not necessary") := 
by
  intros h
  sorry

end sufficient_not_necessary_condition_l109_109540


namespace sets_geometric_sequence_l109_109527

def is_geometric_sequence (s : List ℚ) : Prop :=
  ∀ i, i < s.length - 1 → s[i + 1] / s[i] = s[1] / s[0]

def set_A := [1/3, 1/6, 1/9]
def set_B := [Real.log 3, Real.log 9, Real.log 27]
def set_C := [6, 8, 10]
def set_D := [3, -3 * Real.sqrt 3, 9]

theorem sets_geometric_sequence :
  ¬ is_geometric_sequence set_A ∧
  ¬ is_geometric_sequence set_B ∧
  ¬ is_geometric_sequence set_C ∧
  is_geometric_sequence set_D :=
by
  sorry

end sets_geometric_sequence_l109_109527


namespace integral_of_rational_func_l109_109934

noncomputable def integrand (x : ℝ) : ℝ :=
  (x^3 + 9*x^2 + 21*x + 21) / ((x + 3)^2 * (x^2 + 3))

noncomputable def expected_integral (x : ℝ) : ℝ :=
  -1 / (x + 3) + (1 / 2) * Real.log (x^2 + 3) + (2 / Real.sqrt 3) * Real.arctan (x / Real.sqrt 3)

theorem integral_of_rational_func :
  ∃ C : ℝ, ∀ x : ℝ, (∫ t in 0..x, integrand t) = expected_integral x + C :=
by
  intros
  sorry

end integral_of_rational_func_l109_109934


namespace machine_output_l109_109365

theorem machine_output (input : ℕ) (output : ℕ) (h : input = 26) (h_out : output = input + 15 - 6) : output = 35 := 
by 
  sorry

end machine_output_l109_109365


namespace count_valid_n_l109_109294

theorem count_valid_n :
  {n : ℕ | (∀ t : ℝ, (Complex.sin t - Complex.I * Complex.cos t) ^ n = Complex.sin (n * t) - Complex.I * Complex.cos (n * t)) ∧ n ≤ 500}.to_finset.card = 125 := by
sorry

end count_valid_n_l109_109294


namespace ted_age_l109_109981

theorem ted_age (t s : ℝ) 
  (h1 : t = 3 * s - 20) 
  (h2: t + s = 70) : 
  t = 47.5 := 
by
  sorry

end ted_age_l109_109981


namespace area_of_triangle_OAB_is_4_l109_109685

variable (a : ℝ) (h : a ≠ 0)

def center_of_circle_C := (a, 2 / a)
def passes_through_origin := ((0 - a)^2 + (0 - 2 / a)^2 = a^2 + (2 / a)^2)
def x_axis_intersection := (2 * a, 0)
def y_axis_intersection := (0, 4 / a)

noncomputable def area_of_triangle_OAB :=
  1 / 2 * abs (4 / a) * abs (2 * a)

theorem area_of_triangle_OAB_is_4 :
  passes_through_origin a h →
  area_of_triangle_OAB a h = 4 :=
by
  intros
  sorry

end area_of_triangle_OAB_is_4_l109_109685


namespace charlie_extra_charge_l109_109191

-- Define the data plan and cost structure
def data_plan_limit : ℕ := 8  -- GB
def extra_cost_per_gb : ℕ := 10  -- $ per GB

-- Define Charlie's data usage over each week
def usage_week_1 : ℕ := 2  -- GB
def usage_week_2 : ℕ := 3  -- GB
def usage_week_3 : ℕ := 5  -- GB
def usage_week_4 : ℕ := 10  -- GB

-- Calculate the total data usage and the extra data used
def total_usage : ℕ := usage_week_1 + usage_week_2 + usage_week_3 + usage_week_4
def extra_usage : ℕ := if total_usage > data_plan_limit then total_usage - data_plan_limit else 0
def extra_charge : ℕ := extra_usage * extra_cost_per_gb

-- Theorem to prove the extra charge
theorem charlie_extra_charge : extra_charge = 120 := by
  -- Skipping the proof
  sorry

end charlie_extra_charge_l109_109191


namespace unique_f_l109_109062
noncomputable def valid_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (u v : ℝ), x > 1 → y > 1 → u > 0 → v > 0 →
    f (x^u * y^v) ≤ (f x)^(1 / 44) * (f y)^(1 / 40)

theorem unique_f (f : ℝ → ℝ) (c : ℝ) (h : c > 1) :
  (∀ (x : ℝ), x > 1 → f x = c^(1 / (Real.log x))) ↔ valid_function f :=
begin
  sorry
end

end unique_f_l109_109062


namespace student_activities_arrangement_l109_109896

theorem student_activities_arrangement : 
  ∃ arrangements, 
    (arrangements = Nat.choose 6 4 + Nat.choose 6 3) 
    ∧ arrangements = 35 :=
begin
  sorry
end

end student_activities_arrangement_l109_109896


namespace solid_produces_quadrilateral_l109_109572

-- Define the solids and their properties
inductive Solid 
| cone 
| cylinder 
| sphere

-- Define the condition for a plane cut resulting in a quadrilateral cross-section
def can_produce_quadrilateral_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.cone => False
  | Solid.cylinder => True
  | Solid.sphere => False

-- Theorem to prove that only a cylinder can produce a quadrilateral cross-section
theorem solid_produces_quadrilateral : 
  ∃ s : Solid, can_produce_quadrilateral_cross_section s :=
by
  existsi Solid.cylinder
  trivial

end solid_produces_quadrilateral_l109_109572


namespace derivative_of_constant_function_l109_109783

noncomputable def f : ℝ → ℝ := sorry
variable (a b : ℝ)

-- Condition: f is defined on the interval [a, b]
-- Condition: M is the maximum value of f(x) on [a, b]
-- Condition: m is the minimum value of f(x) on [a, b]
-- Condition: M = m

theorem derivative_of_constant_function (M m : ℝ) (hM : ∀ x ∈ set.Icc a b, f x ≤ M) (hm : ∀ x ∈ set.Icc a b, m ≤ f x) (h_eq : M = m) : ∀ x ∈ set.Icc a b, deriv f x = 0 :=
by
  sorry

end derivative_of_constant_function_l109_109783


namespace sphere_speed_at_C_l109_109907

noncomputable def speed_at_C (Q q : ℝ) (AB AC m g k : ℝ) : ℝ :=
  Real.sqrt ((2 / m) * (k * Q * q * ((1 / AB) - (1 / AC)) + m * g * AB))

theorem sphere_speed_at_C 
  (Q : ℝ := -20 * 10 ^ (-6)) 
  (q : ℝ := 50 * 10 ^ (-6)) 
  (AB : ℝ := 2)
  (AC : ℝ := 3) 
  (m : ℝ := 0.2) 
  (g : ℝ := 10) 
  (k : ℝ := 9 * 10 ^ 9) :
  speed_at_C Q q AB AC m g k = 5 := 
by
  -- Proof to be filled in
  sorry

end sphere_speed_at_C_l109_109907


namespace total_prize_money_l109_109019

theorem total_prize_money (P1 P2 P3 : ℕ) (d : ℕ) (total : ℕ) 
(h1 : P1 = 2000) (h2 : d = 400) (h3 : P2 = P1 - d) (h4 : P3 = P2 - d) 
(h5 : total = P1 + P2 + P3) : total = 4800 :=
sorry

end total_prize_money_l109_109019


namespace complex_coordinate_l109_109649

open Complex

theorem complex_coordinate : 
  let i := Complex.I in
  (1 - i) / (1 + i) = -i := by
sorry

end complex_coordinate_l109_109649


namespace set_pattern_l109_109074

theorem set_pattern (k : ℕ) (k2_minus_1_div_2 : ℤ) (k2_plus_1_div_2 : ℤ):
  (k, k2_minus_1_div_2, k2_plus_1_div_2) = 
       (k, (k^2 - 1) / 2, (k^2 + 1) / 2) :=
by 
  sorry

end set_pattern_l109_109074


namespace marketing_firm_l109_109565

variable (Total_households : ℕ) (A_only : ℕ) (A_and_B : ℕ) (B_to_A_and_B_ratio : ℕ)

def neither_soap_households : ℕ :=
  Total_households - (A_only + (B_to_A_and_B_ratio * A_and_B) + A_and_B)

theorem marketing_firm (h1 : Total_households = 300)
                       (h2 : A_only = 60)
                       (h3 : A_and_B = 40)
                       (h4 : B_to_A_and_B_ratio = 3)
                       : neither_soap_households 300 60 40 3 = 80 :=
by {
  sorry
}

end marketing_firm_l109_109565


namespace blake_poured_out_water_l109_109982

theorem blake_poured_out_water :
  ∀ (initial remaining poured_out : ℝ), initial = 0.8 ∧ remaining = 0.6 →
  poured_out = initial - remaining →
  poured_out = 0.2 :=
by
  intros initial remaining poured_out h cond
  obtain ⟨h_initial, h_remaining⟩ := h
  rw [h_initial, h_remaining] at cond
  simp only [sub_eq_add_neg, add_right_neg, add_zero] at cond
  exact cond

end blake_poured_out_water_l109_109982


namespace real_solutions_count_is_two_l109_109262

def equation_has_two_real_solutions (a b c : ℝ) : Prop :=
  (3*a^2 - 8*b + 2 = c) → (∀ x : ℝ, 3*x^2 - 8*x + 2 = 0) → ∃! x₁ x₂ : ℝ, (3*x₁^2 - 8*x₁ + 2 = 0) ∧ (3*x₂^2 - 8*x₂ + 2 = 0)

theorem real_solutions_count_is_two : equation_has_two_real_solutions (3 : ℝ) (-8 : ℝ) (2 : ℝ) := by
  sorry

end real_solutions_count_is_two_l109_109262


namespace number_of_sparkly_integers_l109_109573

def is_multiple (n d : ℕ) : Prop := d % n = 0 ∧ 1 ≤ d ∧ d ≤ 9

def is_sparkly (num : ℕ) : Prop :=
  (1 <= num / 10^8 % 10 ∧ is_multiple 1 (num / 10^8 % 10)) ∧
  (1 <= num / 10^7 % 10 ∧ is_multiple 2 (num / 10^7 % 10)) ∧
  (1 <= num / 10^6 % 10 ∧ is_multiple 3 (num / 10^6 % 10)) ∧
  (1 <= num / 10^5 % 10 ∧ is_multiple 4 (num / 10^5 % 10)) ∧
  (1 <= num / 10^4 % 10 ∧ is_multiple 5 (num / 10^4 % 10)) ∧
  (1 <= num / 10^3 % 10 ∧ is_multiple 6 (num / 10^3 % 10)) ∧
  (1 <= num / 10^2 % 10 ∧ is_multiple 7 (num / 10^2 % 10)) ∧
  (1 <= num / 10^1 % 10 ∧ is_multiple 8 (num / 10^1 % 10)) ∧
  (1 <= num / 10^0 % 10 ∧ is_multiple 9 (num / 10^0 % 10))

theorem number_of_sparkly_integers : 
  {num : ℕ // is_sparkly num}.card = 216 :=
sorry

end number_of_sparkly_integers_l109_109573


namespace proj_on_line_equals_p_l109_109526

-- Define the line and the projection result
def line_eq (x : ℝ) : ℝ := (5 / 2) * x + 4
def v_on_line (a : ℝ) : ℝ × ℝ := (a, line_eq a)

-- Define the projection operation
def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let (vx, vy) := v
  let (wx, wy) := w
  ((vx * wx + vy * wy) / (wx^2 + wy^2)) * wx,
  ((vx * wx + vy * wy) / (wx^2 + wy^2)) * wy

-- Define the target vector p
def p : ℝ × ℝ := (-40 / 29, 16 / 29)

-- Proof statement
theorem proj_on_line_equals_p (d : ℝ) : 
  ∀ (a : ℝ), proj (v_on_line a) (- 5 / 2 * d, d) = p :=
sorry

end proj_on_line_equals_p_l109_109526


namespace negation_of_universal_proposition_l109_109816

def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0
def A : set ℤ := {x | is_odd x}
def B : set ℤ := {x | is_even x}
def p : Prop := ∀ x ∈ A, 2 * x ∈ B

theorem negation_of_universal_proposition :
  ¬p ↔ ∃ x ∈ A, 2 * x ∉ B :=
by
sorry

end negation_of_universal_proposition_l109_109816


namespace sufficient_not_necessary_l109_109372

variable (x : ℝ)
def p := x^2 > 4
def q := x > 2

theorem sufficient_not_necessary : (∀ x, q x -> p x) ∧ ¬ (∀ x, p x -> q x) :=
by sorry

end sufficient_not_necessary_l109_109372


namespace matrices_inverse_sum_l109_109874

variables {x y z k l m n : ℝ}

theorem matrices_inverse_sum : 
  (matrix.of (λ i j, [![![x, 2, x^2],
                        ![3, y, 4],
                        ![z, 3, z^2]]]).mul 
   (matrix.of (λ i j, [![![ -8, k, -x^3],
                        ![l, -y^2, m],
                        ![  3, n,  z^3]])])) = 1
   → x + y + z + k + l + m + n = -1 / 3 :=
sorry

end matrices_inverse_sum_l109_109874


namespace population_at_end_of_third_year_l109_109878

-- Definitions directly from the conditions
def initial_population : ℝ := 4999.999999999999

def end_of_first_year (pop : ℝ) : ℝ := pop - 0.10 * pop
def end_of_second_year (pop : ℝ) : ℝ := pop + 0.10 * pop
def end_of_third_year (pop : ℝ) : ℝ := pop - 0.10 * pop

-- Statement of the proof problem
theorem population_at_end_of_third_year :
  end_of_third_year (end_of_second_year (end_of_first_year initial_population)) = 4455 :=
begin
  sorry   -- Proof to be filled in
end

end population_at_end_of_third_year_l109_109878


namespace part1_part2_l109_109664

-- Define the universal set ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | 2 ≤ 2^x ∧ 2^x ≤ 8 }

-- Define set B
def B : Set ℝ := { x | x > 2 }

-- Define set C parameterized by a
def C (a : ℝ) : Set ℝ := { x | 1 < x ∧ x < a }

-- Prove part (1)
theorem part1 : (U \ B) ∪ A = { x | x ≤ 3 } := sorry

-- Prove part (2)
theorem part2 (a : ℝ) : (C a ⊆ A) ↔ (1 < a ∧ a ≤ 3) := sorry

end part1_part2_l109_109664


namespace arc_length_of_sector_l109_109692

-- Define the conditions and problem statement
theorem arc_length_of_sector (P : ℝ) (α: ℝ) (r: ℝ) (l: ℝ) (h1: P = 12) (h2: α = 4) (h3: l = α * r) :
  l + 2 * r = P → l = 8 :=
begin
  intros h,
  have h4: 4 * r + 2 * r = 12 := by rw [←h3, h2, h1],
  have h5: 6 * r = 12 := h4,
  have r_eq_2: r = 2 := by linarith,
  have l_eq_8: l = 4 * 2 := by rw [←h3, h2, r_eq_2],
  rw l_eq_8,
  exact rfl,
end

end arc_length_of_sector_l109_109692


namespace distinct_values_expression_l109_109997

theorem distinct_values_expression (a b c d e f : ℤ) (h1 : 3 ^ (3 ^ (3 ^ 3)) = a) 
  (h2 : 3 ^ ((3 ^ 3) ^ 3) = b) (h3 : ((3 ^ 3) ^ 3) ^ 3 = c) 
  (h4 : (3 ^ (3 ^ 3)) ^ 3 = d) (h5 : (3 ^ 3) ^ (3 ^ 3) = e) :
  (finset.card (finset.filter (λ x, x ≠ a) ⟨{a, b, c, d, e}.to_finset.to_list, sorry⟩)) = 4 := sorry

end distinct_values_expression_l109_109997


namespace AM_bisects_BAC_l109_109216

open Real

namespace Geometry

theorem AM_bisects_BAC
  (a b : ℝ) 
  (h_a_pos : 0 < a) 
  (h_b_pos : 0 < b)
  (A B C D K M : ℝ × ℝ)
  (h_A : A = (0, 0))
  (h_B : B = (a, 0))
  (h_C : C = (a, b))
  (h_D : D = (0, b))
  (h_K_def : K = (0, b + sqrt (a^2 + b^2)))
  (h_M_def : M = (a/2, (b + sqrt (a^2 + b^2)) / 2)) :
  angle_bisector A B C M :=
sorry

end Geometry

end AM_bisects_BAC_l109_109216


namespace inequality_x_add_inv_x_ge_two_l109_109095

theorem inequality_x_add_inv_x_ge_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
  sorry

end inequality_x_add_inv_x_ge_two_l109_109095


namespace difference_between_mean_and_median_l109_109014

noncomputable def students_scores : List ℕ := [75, 75, 82, 87, 87, 87, 87, 90, 90, 98, 98]

def median (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.sort
  let n := sorted_lst.length
  if (n % 2 = 0) then
    (sorted_lst.get! (n/2 - 1) + sorted_lst.get! (n/2)) / 2
  else 
    sorted_lst.get! (n/2)

def mean (lst : List ℕ) : ℝ :=
  (lst.map (λ x, x : ℝ)).sum / lst.length

theorem difference_between_mean_and_median :
  (| mean students_scores - median students_scores |).toNat = 9 := by
  sorry

end difference_between_mean_and_median_l109_109014


namespace natural_number_representable_l109_109803

def strictly_decreasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) < a n

def sum_of_first_k_minus_one (a : ℕ → ℕ) (k : ℕ) : ℕ :=
(if k = 1 then 0 else (Nat.range (k - 1)).sum a)

theorem natural_number_representable
  (a : ℕ → ℕ)
  (h_decreasing : strictly_decreasing_seq a) :
  (∀ n : ℕ, ∃ (s : set ℕ), s.finite ∧ (∀ i ∈ s, i < n) ∧ (s.sum id = n)) ↔ (∀ k : ℕ, a k ≤ sum_of_first_k_minus_one a k + 1) :=
sorry

end natural_number_representable_l109_109803


namespace least_integer_gt_sqrt_300_l109_109163

theorem least_integer_gt_sqrt_300 : ∃ (n : ℤ), n > (sqrt 300) ∧ ∀ m : ℤ, m > (sqrt 300) → n ≤ m :=
by
  have h1 : 17^2 = 289 := by norm_num
  have h2 : 18^2 = 324 := by norm_num
  have h_sqrt_300 : 17 < (sqrt 300) := by
    rw [←real.sqrt_eq_rpow]
    exact real.sqrt_lt' h1 (by norm_num : 0 < 300)
  have h_sqrt_300_lt_18 : (sqrt 300) < 18 := by
    rw [←real.sqrt_eq_rpow]
    exact real.rpow_lt (by norm_num : 0 < 300) (by norm_num)
  use 18
  split
  exact h_sqrt_300_lt_18
  intro m h_m_gt_sqrt_300
  exact le_of_lt (lt_min (int.lt_add_one_iff.mpr h_m_gt_sqrt_300) h_sqrt_300_lt_18)
sorry

end least_integer_gt_sqrt_300_l109_109163


namespace possible_values_of_expression_l109_109679

-- Defining the nonzero real numbers a, b, c, and d
variables {a b c d : ℝ}

-- Assuming that a, b, c, and d are nonzero
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- The expression to prove
def expression := (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem to state the possible values of the expression
theorem possible_values_of_expression : 
  expression ha hb hc hd ∈ ({5, 1, -3, -5} : set ℝ) := sorry

end possible_values_of_expression_l109_109679


namespace find_number_l109_109958

theorem find_number (number : ℤ) (h : number + 7 = 6) : number = -1 :=
by
  sorry

end find_number_l109_109958


namespace triangle_angle_solution_exists_l109_109194

noncomputable def possible_angles (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (A = 120 ∨ B = 120 ∨ C = 120) ∧
  (
    ((A = 40 ∧ B = 20) ∨ (A = 20 ∧ B = 40)) ∨
    ((A = 45 ∧ B = 15) ∨ (A = 15 ∧ B = 45))
  )
  
theorem triangle_angle_solution_exists :
  ∃ A B C : ℝ, possible_angles A B C :=
sorry

end triangle_angle_solution_exists_l109_109194


namespace solution_to_exponential_equation_l109_109263

theorem solution_to_exponential_equation : ∀ (x : ℝ), (∃ c : ℝ, 9^(x+6) = 10^x ∧ x = Real.log (9^6) / Real.log c) → c = 10 / 9 := by
  sorry

end solution_to_exponential_equation_l109_109263


namespace sin_cos_identity_count_sin_cos_identity_l109_109289

theorem sin_cos_identity (n : ℕ) (h₁ : n ≤ 500) :
  (∀ t : ℝ, (sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)) ↔
  (∃ k : ℕ, n = 4 * k + 1) :=
sorry

theorem count_sin_cos_identity :
  ∃ m : ℕ, m = 125 ∧ ∀ n : ℕ, n ≤ 500 → (∀ t : ℝ, (sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)) ↔
  n = 4 * (n / 4) + 1 :=
sorry

end sin_cos_identity_count_sin_cos_identity_l109_109289


namespace sum_of_solutions_eq_46_l109_109775

theorem sum_of_solutions_eq_46 (x y : ℤ) (sols : List (ℤ × ℤ)) :
  (∀ (xi yi : ℤ), (xi, yi) ∈ sols →
    (|xi - 4| = |yi - 10| ∧ |xi - 10| = 3 * |yi - 4|)) →
  (sols = [(10, 4), (5, -1), (10, 4), (-5, 19)]) →
  List.sum (sols.map (λ p, p.1 + p.2)) = 46 :=
by
  intro h1 h2
  rw [h2]
  dsimp
  norm_num

end sum_of_solutions_eq_46_l109_109775


namespace discount_difference_l109_109979

open Real

noncomputable def single_discount (B : ℝ) (d1 : ℝ) : ℝ :=
  B * (1 - d1)

noncomputable def successive_discounts (B : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (B * (1 - d2)) * (1 - d3)

theorem discount_difference (B : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) :
  B = 12000 →
  d1 = 0.30 →
  d2 = 0.25 →
  d3 = 0.05 →
  abs (single_discount B d1 - successive_discounts B d2 d3) = 150 := by
  intros h_B h_d1 h_d2 h_d3
  rw [h_B, h_d1, h_d2, h_d3]
  rw [single_discount, successive_discounts]
  sorry

end discount_difference_l109_109979


namespace mrs_hilt_money_l109_109823

-- Definitions and given conditions
def cost_of_pencil := 5  -- in cents
def number_of_pencils := 10

-- The theorem we need to prove
theorem mrs_hilt_money : cost_of_pencil * number_of_pencils = 50 := by
  sorry

end mrs_hilt_money_l109_109823


namespace bug_path_direction_l109_109015

/-- In a plane, a network is drawn where each cell of the network is a regular hexagon with a side length of one unit. 
    A bug crawled along the lines of the network from a point A to a point B along a path that is 20 units long. 
    Show that if there is no path along the network lines between A and B that is shorter than 20 units, 
    then the bug crawled in the same direction for half of the path. -/
theorem bug_path_direction 
  (A B : Type*) 
  (hexagonal_network : A → B → Prop) 
  (path_length : ℕ)
  (shortest_path : ∀ (P Q : A), ¬ hexagonal_network P Q → ¬(∃ (k:ℕ), k < path_length)) :
  (∀ (a b : A), hexagonal_network a b → (path_length = 20) → (∃ (n : ℕ), n = path_length / 2)) → 
  ∃ (x : ℕ), x = 10 :=
begin
  sorry
end

end bug_path_direction_l109_109015


namespace find_rate_per_kg_of_mangoes_l109_109242

theorem find_rate_per_kg_of_mangoes (r : ℝ) 
  (total_units_paid : ℝ) (grapes_kg : ℝ) (grapes_rate : ℝ)
  (mangoes_kg : ℝ) (total_grapes_cost : ℝ)
  (total_mangoes_cost : ℝ) (total_cost : ℝ) :
  grapes_kg = 8 →
  grapes_rate = 70 →
  mangoes_kg = 10 →
  total_units_paid = 1110 →
  total_grapes_cost = grapes_kg * grapes_rate →
  total_mangoes_cost = total_units_paid - total_grapes_cost →
  r = total_mangoes_cost / mangoes_kg →
  r = 55 := by
  intros
  sorry

end find_rate_per_kg_of_mangoes_l109_109242


namespace hamza_bucket_problem_l109_109358

theorem hamza_bucket_problem : 
  let initial_volume_in_5_liter_bucket := 5
      volume_in_3_liter_bucket := 3
      remaining_volume := initial_volume_in_5_liter_bucket - volume_in_3_liter_bucket
      volume_in_6_liter_bucket := remaining_volume
      additional_volume_needed := 6 - volume_in_6_liter_bucket
  in 
  additional_volume_needed = 4 :=
by
  let initial_volume_in_5_liter_bucket := 5
  let volume_in_3_liter_bucket := 3
  let remaining_volume := initial_volume_in_5_liter_bucket - volume_in_3_liter_bucket
  let volume_in_6_liter_bucket := remaining_volume
  let additional_volume_needed := 6 - volume_in_6_liter_bucket
  show additional_volume_needed = 4 from sorry

end hamza_bucket_problem_l109_109358


namespace product_of_solutions_abs_eq_40_l109_109619

theorem product_of_solutions_abs_eq_40 :
  (∃ x1 x2 : ℝ, (|3 * x1 - 5| = 40) ∧ (|3 * x2 - 5| = 40) ∧ ((x1 * x2) = -175)) :=
by
  sorry

end product_of_solutions_abs_eq_40_l109_109619


namespace part1_x_intercept_part2_general_equation_l109_109689

-- Definitions based on given conditions
def l1 (x y : ℚ) : Prop := x - y + 1 = 0
def l2 (x y : ℚ) : Prop := 2 * x + y - 4 = 0
def l3 (x y : ℚ) : Prop := 4 * x + 5 * y - 12 = 0

-- Part 1 statement
theorem part1_x_intercept:
  ∃ (a : ℚ), 
  (∃ (x y : ℚ), l1 x y ∧ l2 x y ∧ (x = 1) ∧ (y = 2)) →
  (∃ (slope : ℚ), slope = 1 / 2) →
  (∃ (l : ℚ → ℚ → ℚ), l = λ x y, y - 2 = 1 / 2 * (x - 1)) →
  (∃ x, l x 0 ∧ x = -3) :=
sorry

-- Part 2 statement
theorem part2_general_equation:
  ∃ (eqn: ℚ → ℚ → Prop),
  (∃ (x y : ℚ), l1 x y ∧ l2 x y ∧ (x = 1) ∧ (y = 2)) →
  (∃ (parallel : ℚ), parallel = -4 / 5) →
  (∃ (l : ℚ → ℚ → ℚ), l = λ x y, y - 2 = -4 / 5 * (x - 1)) →
  (∃ x y, eqn x y ∧ eqn = λ x y, 4 * x + 5 * y - 14 = 0) :=
sorry

end part1_x_intercept_part2_general_equation_l109_109689


namespace time_to_cross_tunnel_l109_109534

-- Definitions of constants based on conditions
def length_of_train : ℝ := 100  -- in meters
def speed_of_train_kmph : ℝ := 72  -- in kilometers per hour
def length_of_tunnel : ℝ := 1400  -- in meters

-- Convert speed from kmph to m/s
def speed_of_train : ℝ := speed_of_train_kmph * (1000 / 3600)

-- Total distance to be covered
def total_distance : ℝ := length_of_train + length_of_tunnel

-- Time calculation, which we need to prove is 75 seconds
theorem time_to_cross_tunnel : (total_distance / speed_of_train) = 75 :=
by
  sorry

end time_to_cross_tunnel_l109_109534


namespace tangent_normal_parabola_tangent_normal_circle_tangent_normal_cycloid_tangent_normal_abs_curve_l109_109544

-- Definitions
def parabola (x : ℝ) : ℝ := x^2 - 4 * x
def circle (x y : ℝ) : Bool := x^2 + y^2 - 2 * x + 4 * y - 3 = 0
def cycloid_x (t : ℝ) : ℝ := t - sin t
def cycloid_y (t : ℝ) : ℝ := 1 - cos t
def abs_curve (x : ℝ) : ℝ := |x^3 - 1|

-- Theorem Statements without proofs
theorem tangent_normal_parabola : 
    (∀ x y : ℝ, x = 1 → y = parabola x → 2 * x + y + 1 = 0 ∧ x - 2 * y - 7 = 0) :=
begin
    sorry
end

theorem tangent_normal_circle :
    (∀ x y : ℝ, circle x y ∧ y = 0 → 
                ((x = 3 ∨ x = -1) ∧ 
                ((x - y + 1 = 0 ∧ x + y - 3 = 0) ∨ 
                (x + y + 1 = 0 ∧ x - y - 3 = 0)))) :=
begin
    sorry
end

theorem tangent_normal_cycloid :
    (∀ t : ℝ, t = π / 2 → 
              ((cycloid_x t = π / 2 - 1 ∧ 
                cycloid_y t = 1) → 
                (2 * cycloid_x t - 2 * cycloid_y t - π + 4 = 0 ∧ 
                2 * cycloid_x t + 2 * cycloid_y t - π = 0))) :=
begin
    sorry
end

theorem tangent_normal_abs_curve :
    (∀ x : ℝ, x = 1 → 
              abs_curve x = 0 → 
              (3 * x - abs_curve x - 2 = 0 ∧ 
              -3 * x - abs_curve x - 2 = 0 ∧ 
              x + 3 * abs_curve x - 1 = 0 ∧ 
              x - 3 * abs_curve x - 1 = 0)) :=
begin
    sorry
end

end tangent_normal_parabola_tangent_normal_circle_tangent_normal_cycloid_tangent_normal_abs_curve_l109_109544


namespace probability_at_least_one_head_and_die_3_l109_109940

-- Define the probability of an event happening
noncomputable def probability_of_event (total_outcomes : ℕ) (successful_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

-- Define the problem specific values
def total_coin_outcomes : ℕ := 4
def successful_coin_outcomes : ℕ := 3
def total_die_outcomes : ℕ := 8
def successful_die_outcome : ℕ := 1
def total_outcomes : ℕ := total_coin_outcomes * total_die_outcomes
def successful_outcomes : ℕ := successful_coin_outcomes * successful_die_outcome

-- Prove that the probability of at least one head in two coin flips and die showing a 3 is 3/32
theorem probability_at_least_one_head_and_die_3 : 
  probability_of_event total_outcomes successful_outcomes = 3 / 32 := by
  sorry

end probability_at_least_one_head_and_die_3_l109_109940


namespace relatively_prime_perfect_squares_l109_109847

theorem relatively_prime_perfect_squares (a b c : ℤ) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_eq : (1:ℚ) / a + (1:ℚ) / b = (1:ℚ) / c) :
    ∃ x y z : ℤ, (a + b = x^2 ∧ a - c = y^2 ∧ b - c = z^2) :=
  sorry

end relatively_prime_perfect_squares_l109_109847


namespace min_value_a_l109_109477

-- Define the function f(x)
def f (x : Real) : Real := 
  Real.sin (Real.pi / 6 * x) * Real.cos (Real.pi / 6 * x) - Real.sqrt 3 * (Real.sin (Real.pi / 6 * x))^2

-- Define the minimum value of a
def min_a (a : Nat) : Prop :=
  ∃ (a : Nat), ∀ x ∈ [-1, a], f(x) takes at least 2 maximum values and a is minimal

-- Prove the minimum value of a is 8
theorem min_value_a : min_a 8 :=
  sorry

end min_value_a_l109_109477


namespace smallest_argument_proof_l109_109580

noncomputable def smallest_argument : ℂ := 12 + 16 * complex.I

theorem smallest_argument_proof (p : ℂ) 
  (h : |p - 25 * complex.I| ≤ 15) : 
  (∃ q : ℂ, |q - 25 * complex.I| ≤ 15 ∧ ∀ r : ℂ, |r - 25 * complex.I| ≤ 15 → complex.arg q ≤ complex.arg r) → 
  p = 12 + 16 * complex.I :=
begin
  sorry
end

end smallest_argument_proof_l109_109580


namespace complex_with_smallest_argument_l109_109578

theorem complex_with_smallest_argument
  (p : ℂ)
  (h : abs (p - (0 + 25 * complex.I)) ≤ 15) :
  p = 12 + 16 * complex.I :=
sorry

end complex_with_smallest_argument_l109_109578


namespace prove_a_value_l109_109977

theorem prove_a_value (a : ℝ) (h : (a - 2) * 0^2 + 0 + a^2 - 4 = 0) : a = -2 := 
by
  sorry

end prove_a_value_l109_109977


namespace find_a_l109_109638

variable (x y a : ℝ)

theorem find_a (h1 : (a * x + 8 * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : a = 7 :=
sorry

end find_a_l109_109638


namespace find_g_five_l109_109870

variable {g : ℝ → ℝ}

theorem find_g_five (h1 : ∀ x y : ℝ, g(x - y) = g(x) * g(y))
(h2 : ∀ x : ℝ, g(x) ≠ 0) : g 5 = 1 :=
sorry

end find_g_five_l109_109870


namespace largest_four_digit_divisible_by_5_l109_109512

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

theorem largest_four_digit_divisible_by_5 : 
  ∃ n : ℕ, n ≤ 9999 ∧ is_divisible_by_5 n ∧ ∀ m : ℕ, m ≤ 9999 → is_divisible_by_5 m → m ≤ n :=
begin
  use 9995,
  split,
  { linarith },
  split,
  { right, norm_num },
  { intros m hm hdiv,
    by_cases h : m = 9995,
    { rw h },
    have : m % 10 ≠ 0 ∧ m % 10 ≠ 5,
    { intro h,
      cases h,
      { rw [h, nat.sub_self, zero_mod] at hdiv, linarith },
      { norm_num at h } },
    have : m < 9995 := sorry,
    linarith }
end

end largest_four_digit_divisible_by_5_l109_109512


namespace range_of_a1_l109_109319

-- Definitions for the sequences a and b
def a_sequence (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := a_sequence n / (3 * a_sequence n + 1)

def b_sequence (a : ℝ) (n : ℕ) : ℝ :=
n * a_sequence a n

-- Definition for the sum of the first n terms of sequence b
def S_n (a : ℝ) (n : ℕ) : ℝ :=
∑ i in Finset.range n, b_sequence a (i + 1)

-- The main theorem statement
theorem range_of_a1 (a₁ : ℝ) (h₀ : a₁ < 0)
  (h₁ : ∀ n : ℕ, a_sequence a₁ (n + 1) = a_sequence a₁ n / (3 * a_sequence a₁ n + 1))
  (h₂ : ∀ n : ℕ, b_sequence a₁ n = n * a_sequence a₁ n)
  (h₃ : ∃! n : ℕ, n > 0 ∧ S_n a₁ n < S_n a₁ (n - 1)) :
  - (1 / 18) < a₁ ∧ a₁ < - (1 / 21) :=
sorry

end range_of_a1_l109_109319


namespace arithmetic_sequence_inequality_l109_109131

variable {α : Type*}

noncomputable def arithmetic_sequence_sum (a : ℕ → α) (n : ℕ) [ring α] : α :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * a 1

theorem arithmetic_sequence_inequality 
  (a : ℕ → ℝ) (d : ℝ) (h_d_pos : d > 0)
  (h_S : (arithmetic_sequence_sum a 7 - arithmetic_sequence_sum a 4) * 
          (arithmetic_sequence_sum a 8 - arithmetic_sequence_sum a 4) < 0) :
  |a 6| < |a 7| :=
sorry

end arithmetic_sequence_inequality_l109_109131


namespace largest_last_digit_sequence_l109_109116

theorem largest_last_digit_sequence:
  ∀ (s : String), 
  s.length = 2024 ∧ s.get 0 = '1' ∧ 
  (∀ i, i < 2023 → ((s.get i).to_nat * 10 + (s.get (i + 1)).to_nat) % 17 = 0 ∨ ((s.get i).to_nat * 10 + (s.get (i + 1)).to_nat) % 29 = 0) 
  → s.get 2023 = '8' :=
by 
  intros s h1 h2 h3
  sorry

end largest_last_digit_sequence_l109_109116


namespace Jill_time_to_school_l109_109257

noncomputable def Dave_walking_rate := 80 / 1  -- 80 steps per minute.
noncomputable def Dave_step_length := 65 / 1   -- 65 cm per step.
noncomputable def Dave_time_to_school := 20 / 1  -- 20 minutes.

noncomputable def Jill_walking_rate := 120 / 1  -- 120 steps per minute.
noncomputable def Jill_step_length := 55 / 1   -- 55 cm per step.

theorem Jill_time_to_school :
  let Dave_speed := Dave_walking_rate * Dave_step_length in
  let distance_to_school := Dave_speed * Dave_time_to_school in
  let Jill_speed := Jill_walking_rate * Jill_step_length in
  let Jill_time := distance_to_school / Jill_speed in
  Jill_time ≈ 16 :=
begin
  let Dave_speed := Dave_walking_rate * Dave_step_length,
  let distance_to_school := Dave_speed * Dave_time_to_school,
  let Jill_speed := Jill_walking_rate * Jill_step_length,
  let Jill_time := distance_to_school / Jill_speed,
  have : Jill_time ≈ 15.76, -- approximation step
  calc Jill_time ≈ 16 : sorry -- deriving Jill time is approximately 16
end

end Jill_time_to_school_l109_109257


namespace angle_AD_BC_zero_l109_109146

-- Define the two circles intersecting at points A and B
variables {ω1 ω2 : Circle}

-- Define the point of tangency A and point C on the second circle
variables {A B C D : Point}

-- Conditions given in the problem
axiom intersect_points (A B : Point) (ω1 ω2 : Circle) : A ∈ ω1 ∧ B ∈ ω2 ∧ A ∈ ω2 ∧ B ∈ ω1
axiom tangent_A_C (A C : Point) (ω1 ω2 : Circle) : (A ∈ ω1 ∧ C ∈ ω2) ∧ is_tangent_line A C ω1
axiom tangent_B_D (B D : Point) (ω2 ω1 : Circle) : (B ∈ ω2 ∧ D ∈ ω1) ∧ is_tangent_line B D ω2

-- The goal is to prove the angle between lines AD and BC is 0 degrees
theorem angle_AD_BC_zero (A B C D : Point) (ω1 ω2 : Circle)
  (h1 : intersect_points A B ω1 ω2)
  (h2 : tangent_A_C A C ω1 ω2)
  (h3 : tangent_B_D B D ω2 ω1) : 
  angle_between_lines A D B C = 0 :=
sorry

end angle_AD_BC_zero_l109_109146


namespace required_teachers_l109_109506

-- Define the conditions
def students : ℕ := 900
def classes_per_student : ℕ := 6
def students_per_class : ℕ := 25
def classes_per_teacher : ℕ := 5

-- Define the problem: Calculate the number of teachers needed
theorem required_teachers : 
  (students * classes_per_student) / students_per_class / classes_per_teacher + 
  ((students * classes_per_student) % (students_per_class * classes_per_teacher) ≠ 0) = 44 :=
by sorry

end required_teachers_l109_109506


namespace PA_PB_PC_concurrent_on_OI_l109_109022

-- Definitions related to acute-angled triangle and involved points
variables {A B C O I K : Point}
variables {P_A P_B P_C : Point}
variables {circle_circum : Circle}
variables {triangle_ABC : Triangle}

-- Assume the following conditions (1 to 4)
axiom (acute_angled_triangle : acute_triangle triangle_ABC)
axiom (O_center_circumcircle : is_center_circumcircle O triangle_ABC)
axiom (I_center_incircle : is_center_incircle I triangle_ABC)
axiom (K_on_segment_OI : on_segment K O I)
axiom (P_A_second_intersection : second_intersection P_A (line_through A K) circle_circum)
axiom (P_B_second_intersection : second_intersection P_B (line_through B K) circle_circum)
axiom (P_C_second_intersection : second_intersection P_C (line_through C K) circle_circum)

theorem PA_PB_PC_concurrent_on_OI :
  meets_at_single_point (line_through A K) (line_through B K) (line_through C K) O I :=
sorry

end PA_PB_PC_concurrent_on_OI_l109_109022


namespace rectangle_original_length_doubles_area_l109_109684

-- Let L and W denote the length and width of a rectangle respectively
-- Given the condition: (L + 2)W = 2LW
-- We need to prove that L = 2

theorem rectangle_original_length_doubles_area (L W : ℝ) (h : (L + 2) * W = 2 * L * W) : L = 2 :=
by 
  sorry

end rectangle_original_length_doubles_area_l109_109684


namespace total_number_of_students_l109_109489

theorem total_number_of_students 
    (group1 : Nat) (group2 : Nat) (group3 : Nat) (group4 : Nat) 
    (h1 : group1 = 5) (h2 : group2 = 8) (h3 : group3 = 7) (h4 : group4 = 4) : 
    group1 + group2 + group3 + group4 = 24 := 
by
  sorry

end total_number_of_students_l109_109489


namespace sum_solutions_eq_26_l109_109771

theorem sum_solutions_eq_26:
  (∃ (n : ℕ) (solutions: Fin n → (ℝ × ℝ)),
    (∀ i, let (x, y) := solutions i in |x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|)
    ∧ (Finset.univ.sum (λ i, let (x, y) := solutions i in x + y) = 26))
:= sorry

end sum_solutions_eq_26_l109_109771


namespace largest_four_digit_divisible_by_5_l109_109519

theorem largest_four_digit_divisible_by_5 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 5 = 0 → m ≤ n :=
begin
  use 9995,
  split, norm_num,
  split, norm_num,
  split, norm_num,
  intros m hm1 hm2 hm3,
  have H : m ≤ 9999, from hm2,
  norm_num at H,
  sorry
end

end largest_four_digit_divisible_by_5_l109_109519


namespace mean_greater_than_median_by_one_fifth_l109_109868

def num_students : ℕ := 20

def days_missed : list ℕ := [0, 0, 1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 7, 7]

def total_days_missed : ℕ := list.sum (days_missed.map (id))

def mean_days_missed : ℚ := total_days_missed / num_students

def median_days_missed : ℚ := 3 -- Median calculated from the given data

theorem mean_greater_than_median_by_one_fifth :
  mean_days_missed - median_days_missed = 1 / 5 := 
sorry

end mean_greater_than_median_by_one_fifth_l109_109868


namespace remainder_when_b_divided_by_13_is_6_l109_109811

theorem remainder_when_b_divided_by_13_is_6 :
  let b := (2⁻¹ + 3⁻¹ + 5⁻¹)⁻¹ in 
  b % 13 = 6 :=
begin
  sorry
end

end remainder_when_b_divided_by_13_is_6_l109_109811


namespace tina_made_more_140_dollars_l109_109818

def candy_bars_cost : ℕ := 2
def marvin_candy_bars : ℕ := 35
def tina_candy_bars : ℕ := 3 * marvin_candy_bars
def marvin_money : ℕ := marvin_candy_bars * candy_bars_cost
def tina_money : ℕ := tina_candy_bars * candy_bars_cost
def tina_extra_money : ℕ := tina_money - marvin_money

theorem tina_made_more_140_dollars :
  tina_extra_money = 140 := by
  sorry

end tina_made_more_140_dollars_l109_109818


namespace uphill_integers_divisible_by_9_l109_109989

def is_uphill_integer (n : ℕ) : Prop :=
  -- Convert the integer to a list of its digits and check the uphill property
  let digits := n.digits 10
  list.pairwise (<) digits

def is_divisible_by_9 (n : ℕ) : Prop :=
  n.digits 10.sum % 9 = 0

def number_of_uphill_integers_divisible_by_9 : ℕ :=
  (list.fin_range 1000).countp (λ n, is_uphill_integer n ∧ is_divisible_by_9 n)

theorem uphill_integers_divisible_by_9 :
  number_of_uphill_integers_divisible_by_9 = 1 := 
sorry

end uphill_integers_divisible_by_9_l109_109989


namespace david_more_heads_than_evan_l109_109612

theorem david_more_heads_than_evan :
  (let D : ℕ → ℝ := λ h, 1 / 2^(h+1),
       E : ℕ → ℝ := λ h, if h = 0 then 1 / 4 else 3^h / 4^(h+1) in
   ∑ n in (range 1000), E n * (∑ m in (range 1000), if m > n then D m else 0)) = 1/5 :=
begin
  -- Proof omitted
  sorry
end

end david_more_heads_than_evan_l109_109612


namespace four_times_remaining_marbles_l109_109838

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end four_times_remaining_marbles_l109_109838


namespace problem_part1_problem_part2_l109_109751

open Real

noncomputable def circle_rect_eq (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

noncomputable def min_len_PA_PB (a : ℝ) : ℝ :=
  2 * sqrt 2

theorem problem_part1 (theta : ℝ) :
  ∃ x y : ℝ, (let r := 4 * cos theta in r * r = x ^ 2 + y ^ 2 ∧ r * cos theta = x)
  → circle_rect_eq _ _ := sorry

theorem problem_part2 (a : ℝ) : 
  ∃ t1 t2 : ℝ, (let f := 2 * (sin a - cos a) in t1 ^ 2 + t2 ^ 2 + f = 0 ∧
                  t1 + t2 = -f)
  → min_len_PA_PB a = 2 * sqrt 2 := sorry

end problem_part1_problem_part2_l109_109751


namespace cars_with_neither_features_l109_109932

-- Define the given conditions
def total_cars : ℕ := 65
def cars_with_power_steering : ℕ := 45
def cars_with_power_windows : ℕ := 25
def cars_with_both_features : ℕ := 17

-- Define the statement to be proved
theorem cars_with_neither_features : total_cars - (cars_with_power_steering + cars_with_power_windows - cars_with_both_features) = 12 :=
by
  sorry

end cars_with_neither_features_l109_109932


namespace gcf_120_180_300_l109_109508

theorem gcf_120_180_300 : Nat.gcd (Nat.gcd 120 180) 300 = 60 := 
by eval_gcd 120 180 300

end gcf_120_180_300_l109_109508


namespace minimum_cuts_for_48_pieces_l109_109073

theorem minimum_cuts_for_48_pieces 
  (rearrange_without_folding : Prop)
  (can_cut_multiple_layers_simultaneously : Prop)
  (straight_line_cut : Prop)
  (cut_doubles_pieces : ∀ n, ∃ m, m = 2 * n) :
  ∃ n, (2^n ≥ 48 ∧ ∀ m, (m < n → 2^m < 48)) ∧ n = 6 := 
by 
  sorry

end minimum_cuts_for_48_pieces_l109_109073


namespace number_of_days_at_Tom_house_l109_109499

-- Define the constants and conditions
def total_people := 6
def plates_per_person_per_day := 6
def total_plates := 144

-- Prove that the number of days they were at Tom's house is 4
theorem number_of_days_at_Tom_house : total_plates / (total_people * plates_per_person_per_day) = 4 :=
  sorry

end number_of_days_at_Tom_house_l109_109499


namespace angle_EDF_equilateral_triangle_l109_109739

open EuclideanGeometry

theorem angle_EDF_equilateral_triangle 
  (A B C D E F : Point)
  (h_equilateral : EquilateralTriangle A B C)
  (h_angle_A : angle A B C = 60)
  (h_C_on_BC : LineSegment B C ∋ D)
  (h_E_on_AC : LineSegment A C ∋ E)
  (h_F_on_AB : LineSegment A B ∋ F)
  (h_CD_CE : dist C D = dist C E)
  (h_BD_BF : dist B D = dist B F) :
  angle E D F = 120 := 
by
  sorry

end angle_EDF_equilateral_triangle_l109_109739


namespace tournament_games_count_l109_109942

-- We define the conditions
def number_of_players : ℕ := 6

-- Function to calculate the number of games played in a tournament where each player plays twice with each opponent
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Now we state the theorem
theorem tournament_games_count : total_games number_of_players = 60 := by
  -- Proof goes here
  sorry

end tournament_games_count_l109_109942


namespace min_chopsticks_needed_l109_109895

/-- Structure representing chopsticks and conditions --/
structure ChopstickScenario where
  num_colors : ℕ                -- Number of colors
  pairs_per_color : ℕ           -- Pairs per each color

/-- Defining our specific scenario from the problem --/
def chopstickProblem : ChopstickScenario := {
  num_colors := 3,
  pairs_per_color := 4
}

/-- 
  We want to prove that to guarantee grabbing at least 
  two pairs of chopsticks with different colors in one go,
  the minimum number of chopsticks required is 11.
--/
theorem min_chopsticks_needed (cs : ChopstickScenario) :
  cs.num_colors = 3 ∧ cs.pairs_per_color = 4 → nat.succ (cs.pairs_per_color * cs.num_colors - 1) = 11 := by
  sorry

end min_chopsticks_needed_l109_109895


namespace divide_circle_with_equal_fences_l109_109956

-- Define the concept of a circle and its radius
structure Circle where
  r : ℝ

-- Define the concept of dividing a circle into quadrants using two perpendicular diameters
def divided_into_quadrants (c : Circle) : Prop :=
  ∃ d1 d2 : ℝ, d1 = 2 * c.r ∧ d2 = 2 * c.r ∧ (d1 ⊥ d2)

-- Define the concept of three equal semi-circles within each quadrant acting as fences
def equal_length_fences (c : Circle) (f1 f2 f3 : ℝ) : Prop :=
  f1 = π * (c.r / 2) ∧ f2 = π * (c.r / 2) ∧ f3 = π * (c.r / 2)

-- The theorem stating the problem
theorem divide_circle_with_equal_fences (c : Circle) :
  divided_into_quadrants c →
  ∃ f1 f2 f3 : ℝ, equal_length_fences c f1 f2 f3 :=
by
  intro h
  use [π * (c.r / 2), π * (c.r / 2), π * (c.r / 2)]
  exact ⟨rfl, rfl, rfl⟩

end divide_circle_with_equal_fences_l109_109956


namespace sum_first_nine_terms_l109_109026

variable {a : ℕ → ℝ}
variable (d : ℝ)

noncomputable def a_n (n : ℕ) : ℝ := a 1 + (n - 1) * d
noncomputable def S_n (n : ℕ) : ℝ := n * (a 1 + a_n n) / 2

-- given: a_3 + a_7 = 12
variable (h : a_n 3 + a_n 7 = 12)

-- goal: S_9 = 54
theorem sum_first_nine_terms
  (h : a_n 3 + a_n 7 = 12) : S_n 9 = 54 := by
  sorry

end sum_first_nine_terms_l109_109026


namespace base_b_not_divisible_by_5_l109_109300

theorem base_b_not_divisible_by_5 :
  ∀ b ∈ {4, 5, 7, 9, 10}, ¬ (2 * b^3 - 2 * b^2 - b + 1) % 5 = 0 ↔ b = 4 ∨ b = 7 ∨ b = 9 :=
by sorry

end base_b_not_divisible_by_5_l109_109300


namespace problem_proof_l109_109712

noncomputable def point (x y : ℝ) := (x, y)
noncomputable def Line (k : ℝ) (x₁ y₁ : ℝ) : ℝ × ℝ → Prop := 
  λ p, p.2 = k * (p.1 - x₁) + y₁

def passes_through (l : ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop := l p

def l1 (k₁ : ℝ) : ℝ × ℝ → Prop := Line k₁ 1 3
def l2 (k₂ : ℝ) : ℝ × ℝ → Prop := Line k₂ 2 3

def must_intersect_or_parallel_or_neither (k₁ k₂ : ℝ) : Prop :=
  ∃ p, passes_through (l1 k₁) p ∧ passes_through (l2 k₂) p ∧ k₁ = k₂ ∨ 
  ∃ p, passes_through (l1 k₁) p ∧ passes_through (l2 k₂) p ∧ k₁ ≠ k₂ ∨  
  (∀ p₁ p₂, passes_through (l1 k₁) p₁ ∧ passes_through (l2 k₂) p₂ → p₁ ≠ p₂ ∧ k₁ ≠ k₂)

theorem problem_proof {k₁ k₂ : ℝ} : must_intersect_or_parallel_or_neither k₁ k₂ := by
  sorry

end problem_proof_l109_109712


namespace sum_of_distinct_terms_if_and_only_if_l109_109797

theorem sum_of_distinct_terms_if_and_only_if
  (a : ℕ → ℕ)
  (h_decreasing : ∀ k, a k > a (k + 1))
  (S : ℕ → ℕ)
  (h_S : ∀ k, S k = (∑ i in finset.range (k - 1), a i))
  (h_S1 : S 1 = 0) :
  (∀ n : ℕ, ∃ (l : list ℕ), (∀ (i : ℕ), i ∈ l → ∃ k, i = a k) ∧  l.sum = n) ↔ (∀ k, a k ≤ S k + 1) :=
begin
  sorry,
end

end sum_of_distinct_terms_if_and_only_if_l109_109797


namespace determine_p_l109_109748

noncomputable def horizon_constant (r h p : ℝ) (d : ℝ) : Prop :=
  d = p * real.sqrt h

theorem determine_p (r : ℝ) (h : ℝ) (p : ℝ) : r = 6370 → 
  (h / 1000 < r) → horizon_constant r h p (p * real.sqrt h) → p = real.sqrt (2 * 6370) :=
begin
  intros,
  sorry,
end

end determine_p_l109_109748


namespace production_steps_use_process_flowchart_l109_109497

def describe_production_steps (task : String) : Prop :=
  task = "describe production steps of a certain product in a factory"

def correct_diagram (diagram : String) : Prop :=
  diagram = "Process Flowchart"

theorem production_steps_use_process_flowchart (task : String) (diagram : String) :
  describe_production_steps task → correct_diagram diagram :=
sorry

end production_steps_use_process_flowchart_l109_109497


namespace first_discount_percentage_l109_109882

-- Definitions based on the conditions provided
def listed_price : ℝ := 400
def final_price : ℝ := 334.4
def additional_discount : ℝ := 5

-- The equation relating these quantities
theorem first_discount_percentage (D : ℝ) (h : listed_price * (1 - D / 100) * (1 - additional_discount / 100) = final_price) : D = 12 :=
sorry

end first_discount_percentage_l109_109882


namespace matrix_result_l109_109054

open Matrix

variables {R : Type*} [Ring R]
variables {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
variables {M : Matrix m n R} {v w : n → R}
variables (h1 : M.mulVec v = ![1, -5])
variables (h2 : M.mulVec w = ![7, 2])

theorem matrix_result : M.mulVec (-2 • v + w) = ![5, 12] :=
by
  sorry

end matrix_result_l109_109054


namespace solve_for_x_l109_109718

theorem solve_for_x (x : ℝ) (h₁ : 3 * x^2 - 9 * x = 0) (h₂ : x ≠ 0) : x = 3 := 
by {
  sorry
}

end solve_for_x_l109_109718


namespace find_lambda_l109_109307

variable (A B C : Type)
variable (AB AC BC : A → A → A)
variable (λ : ℝ)

-- Given conditions
axiom (h1 : AB = 2 • AC)
axiom (h2 : AB = λ • BC)

-- Prove that λ = -2
theorem find_lambda : λ = -2 :=
sorry

end find_lambda_l109_109307


namespace monotone_increasing_condition_l109_109922

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotone_increasing_condition (t : ℝ) (h₁ : 0 < t) (h₂ : t < Real.pi / 6) :
  ∀ x y, x ∈ Ioo (-t) t → y ∈ Ioo (-t) t → x ≤ y → f x ≤ f y :=
sorry

end monotone_increasing_condition_l109_109922


namespace mod_remainder_l109_109810

theorem mod_remainder (b : ℤ) (h : b ≡ ((2⁻¹ * 2 + 3⁻¹ * 3 + 5⁻¹ * 5)⁻¹ : ℤ) [MOD 13]) : 
  b ≡ 6 [MOD 13] := 
  sorry

end mod_remainder_l109_109810


namespace sum_coefficients_excluding_constant_l109_109130

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := if h : k ≤ n then nat.choose n k else 0

-- Define the general term of the binomial expansion
def general_term (r : ℕ) (x : ℝ) : ℝ :=
  (-1)^r * binom 9 r * 2^(9-r) * x^((3 * r - 9) / 2)

-- Define the sum of coefficients of all terms except the constant term in the expansion
def sum_coefficients_except_constant : ℝ :=
  let constant_term : ℝ := general_term 3 1 in
  let sum_of_all_coefficients : ℝ := (2 - 1)^9 in
  sum_of_all_coefficients + constant_term.abs

theorem sum_coefficients_excluding_constant {x : ℝ} (h : x = 1) :
  sum_coefficients_except_constant = 5377 :=
by {
  sorry -- Proof will go here
}

end sum_coefficients_excluding_constant_l109_109130


namespace quadrilateral_perimeter_l109_109024

theorem quadrilateral_perimeter (EF FG HG EH : ℝ)
  (EF_val : EF = 24)
  (FG_val : FG = 32)
  (HG_val : HG = 20)
  (EH_val : EH = real.sqrt(2000)) :
  EF + FG + HG + EH = 76 + 20 * real.sqrt 5 :=
by {
  rw [EF_val, FG_val, HG_val, EH_val],
  -- This would reduce the arithmetic to straightforward calculation.
  sorry
}

end quadrilateral_perimeter_l109_109024


namespace Levi_brother_initial_score_l109_109441

def Levi_initial_score (B : ℕ) : Prop :=
  let Levi_initial := 8 in
  let Levi_additional := 12 in
  let brother_additional := 3 in
  let Levi_final := Levi_initial + Levi_additional in
  Levi_final = (B + brother_additional) + 5

theorem Levi_brother_initial_score :
  ∃ B : ℕ, Levi_initial_score B ∧ B = 12 :=
by
  existsi 12
  unfold Levi_initial_score
  simp
  sorry

end Levi_brother_initial_score_l109_109441


namespace find_a_l109_109379

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 4 → f y ≤ f x) → 
  a = -3 := 
by
  have f : ℝ → ℝ := λ x, x^2 + 2 * (a - 1) * x + 2
  intro h
  sorry

end find_a_l109_109379


namespace roots_of_P_n_are_real_and_negative_l109_109786

-- Define the set of permutations of {1, 2, ..., n}
def permutations (n : ℕ) : List (Equiv.Perm (Fin n)) := 
  List.bind (List.range n) (λ i, List.product (List.replicate n (Fin n)))

-- Define cyc to count the number of cycles in a permutation
def num_cycles {n : ℕ} (σ : Equiv.Perm (Fin n)) : ℕ := sorry

-- Define P_n(x) as the sum over permutations with x raised to the number of cycles
noncomputable def P (n : ℕ) (x : ℝ) : ℝ :=
  permutations n |> List.map (λ σ, x ^ num_cycles σ) |> List.sum

-- Main theorem to state that roots of P_n are real negative integers
theorem roots_of_P_n_are_real_and_negative (n : ℕ) (hn : n ≥ 1) :
  ∀ x, P n x = 0 → ∃ k : ℕ, (k < n ∧ x = -(k + 1)) :=
sorry

end roots_of_P_n_are_real_and_negative_l109_109786


namespace certain_event_is_D_l109_109172

-- Defining the conditions
def EventA : Prop := 
  ∃ (tv_on : Prop), tv_on ∧ (∀ (news_broadcast : Prop), news_broadcast → true)

def EventB : Prop := 
  let lottery_prob := 0.01 in
  let tickets_bought := 100 in
  ¬(tickets_bought * lottery_prob = 1)

def EventC : Prop :=
  let fair_coin := true in
  let tosses := 100 in
  let expected_heads := 50 in
  ¬(fair_coin ∧ expected_heads = (tosses / 2))

def EventD : Prop :=
  let students := 367 in
  let birthdays := 366 in
  students > birthdays

-- The main theorem to be proved
theorem certain_event_is_D (hA : ¬ EventA) (hB : ¬ EventB) (hC : ¬ EventC) (hD : EventD) : EventD := by
  exact hD

end certain_event_is_D_l109_109172


namespace shaded_region_complex_plane_l109_109980

-- Define the statement for the complex set corresponding to the shaded region
theorem shaded_region_complex_plane :
  {z : ℂ | abs z ≤ 1 ∧ z.im ≥ 1 / 2} = {z : ℂ | abs z ≤ 1 ∧ z.im ≥ 1 / 2} :=
by
  sorry

end shaded_region_complex_plane_l109_109980


namespace max_rooks_in_symmetric_half_l109_109655

theorem max_rooks_in_symmetric_half (n : ℕ) (h_rooks : n > 0) :
  let chessboard := fin (2 * n) → fin (2 * n) → bool -- Chessboard with rooks as boolean matrix
  in ∀ (rooks : fin (2 * n) → fin (2 * n)),
    (∀ i j, rooks i j = tt → (∀ k, k ≠ i → rooks k j = ff) ∧ (∀ l, l ≠ j → rooks i l = ff)) → -- No two rooks in the same row or column
    ∀ (partition : (fin (2 * n) × fin (2 * n)) → bool), -- Symmetric partition function
      (∀ (x y : fin (2 * n)), partition (x, y) = partition (fin (2 * n).sub 1 - x, fin (2 * n).sub 1 - y)) → -- Symmetry condition
      (∀ (a b : fin (2 * n) × fin (2 * n)),
        partition a = partition b → (∃ (path : list (fin (2 * n) × fin (2 * n))),
          path.head = a ∧ path.ilast = b ∧ (∀ (u v : fin (2 * n) × fin (2 * n)), u ∈ path → v ∈ path → 
          ((u.1.1 - v.1.1)^2 + (u.1.2 - v.1.2)^2 = 1) ∨ ((u.1.1 - v.1.1)^2 + (u.1.2 - v.1.2)^2 = 0)))) → -- Connectivity condition
      ∃ (part1 part2 : set (fin (2 * n) × fin (2 * n))),
        (∀ (p : fin (2 * n) × fin (2 * n)), part1 p ∨ part2 p) ∧ -- Division into two parts
        (∀ (p : fin (2 * n) × fin (2 * n)), part1 p → ¬ part2 p) ∧
        (∀ (p q : fin (2 * n) × fin (2 * n)), part1 p ∧ part1 q → (partition p = tt → partition q = tt)) ∧
        (∀ (p q : fin (2 * n) × fin (2 * n)), part2 p ∧ part2 q → (partition p = ff → partition q = ff)) ∧
        max (finset.filter (λ p, (partition p = tt)) (finset.univ.image (λ (p : fin (2 * n) × fin (2 * n)), rooks (p.1, p.2) = tt)).card 
             (finset.filter (λ p, (partition p = ff)) (finset.univ.image (λ (p : fin (2 * n) × fin (2 * n)), rooks (p.1, p.2) = tt)).card) = 2 * n - 1 :=
sorry

end max_rooks_in_symmetric_half_l109_109655


namespace simplest_sqrt_l109_109528

theorem simplest_sqrt :
  ¬ ∃ (xy : ℝ), sqrt 15 = xy.and (xy * xy = 15) ∧
  ( ∃ (a b : ℝ), sqrt 32 = a * sqrt b 
    ∧ (a = 4) ∧ (b = 2) ) ∧
  ( sqrt (1/5) = sqrt 5 / 5 ) ∧
  ( sqrt 49 = 7 ) :=
by sorry

end simplest_sqrt_l109_109528


namespace count_integer_k_equals_6_l109_109380

noncomputable def count_valid_k (k_bounds : ℕ) : ℕ := 
  (List.range k_bounds).filter (λ k, ∃ x : ℤ, (k - 45) * x = 4 - 47 * x).length

theorem count_integer_k_equals_6 : 
  count_valid_k 100 = 6 := 
sorry

end count_integer_k_equals_6_l109_109380


namespace sum_of_solutions_l109_109779

theorem sum_of_solutions :
  let solutions := [(-8, -2), (-1, 5), (10, 4), (10, 4)],
  (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 :=
by
  let solutions : List (ℤ × ℤ) := [(-8, -2), (-1, 5), (10, 4), (10, 4)]
  have h1 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 4| = |y - 10| := sorry
  have h2 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 10| = 3 * |y - 4| := sorry
  have solution_sum : (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 := by
    simp [solutions]
    norm_num
  exact solution_sum

end sum_of_solutions_l109_109779


namespace obtuse_triangle_l109_109010

variables {A B C : ℝ} {a b c R : ℝ}

theorem obtuse_triangle
  (h1 : sin A ^ 2 + sin B ^ 2 < sin C ^ 2)
  (h2 : sin A = a / (2 * R))
  (h3 : sin B = b / (2 * R))
  (h4 : sin C = c / (2 * R)) :
  ∠C > 90 :=
begin
  sorry,
end

end obtuse_triangle_l109_109010


namespace evaluate_fraction_l109_109931

theorem evaluate_fraction : (35 / 0.07) = 500 := 
by
  sorry

end evaluate_fraction_l109_109931


namespace sum_of_digits_of_10_pow_40_minus_46_l109_109167

theorem sum_of_digits_of_10_pow_40_minus_46 :
  let k := (10 ^ 40 - 46) in
  (nat.digits 10 k).sum = 369 := by
  sorry

end sum_of_digits_of_10_pow_40_minus_46_l109_109167


namespace count_three_digit_numbers_with_two_same_digits_l109_109576

theorem count_three_digit_numbers_with_two_same_digits : 
  ∃ n : ℕ, n = 75 ∧ 
  ∀ (d1 d2 : ℕ), 
  (d1 ∈ {0, 1, 2, 3, 4, 5}) ∧ 
  (d2 ∈ {0, 1, 2, 3, 4, 5}) ∧ 
  d1 ≠ d2 ∧ 
  (digit_count = 3) ∧ 
  (exactly_two_same_digits) ∧ 
  (one_repeats) → 
  (number_is_three_digit ∧ 
  formed_by_digits_only_from {0, 1, 2, 3, 4, 5}) → 
  n = 75 :=
by
  sorry

end count_three_digit_numbers_with_two_same_digits_l109_109576


namespace perimeter_triangle_A1DM_distance_D1_to_plane_A1DM_l109_109652

theorem perimeter_triangle_A1DM (a : ℝ) :
  let A1D := a * Real.sqrt 2,
      A1M := (a * Real.sqrt 5) / 2,
      DM := (a * Real.sqrt 5) / 2
  in
  (A1M + DM + A1D) = a * (Real.sqrt 5 + Real.sqrt 2) := 
sorry

theorem distance_D1_to_plane_A1DM (a : ℝ) :
  let A1M := (a * Real.sqrt 5) / 2,
      MK := (a * Real.sqrt 3) / 2,
      S_delta_A1MD := (1/2) * A1M * MK,
      V := (1/3) * S_delta_A1MD * a
  in
  let h := V * 24 / (a^2) / Real.sqrt 15 
  in
  h = a / Real.sqrt 6 :=
sorry

end perimeter_triangle_A1DM_distance_D1_to_plane_A1DM_l109_109652


namespace exists_city_with_8_companies_l109_109482

theorem exists_city_with_8_companies (companies : Nat) (roads : Nat) (cities : Nat) 
  (company_roads : Nat) :
  companies = 80 → 
  roads = 2400 →
  cities = 100 →
  (∀ c, 1 ≤ c ∧ c ≤ companies → company_roads = 30) →
  (∃ r, r ∈ roads ∧ r connects 2 cities) →
  (∀ r1 r2, r1 ≠ r2 → r1 connects two distinct cities → r2 connects two distinct cities → r1 ≠ r2 ) →
  (∀ c, 1 ≤ c ∧ c ≤ companies → ∃ city1 city2, city1 ≠ city2 ∧ (c has agencies in city1) ∧ (c has agencies in city2)) →
  ∃ city, ∃ agency_set, agency_set = {c | c ∈ companies ∧ has_agency_in(c, city)} ∧ agency_set.card ≥ 8 := by
    intros
    sorry

end exists_city_with_8_companies_l109_109482


namespace solution_in_Quadrant_III_l109_109061

theorem solution_in_Quadrant_III {c x y : ℝ} 
    (h1 : x - y = 4) 
    (h2 : c * x + y = 5) 
    (hx : x < 0) 
    (hy : y < 0) : 
    c < -1 := 
sorry

end solution_in_Quadrant_III_l109_109061


namespace moe_mowing_time_correct_l109_109819

noncomputable def time_to_mow_lawn 
  (length_lawn width_lawn : ℝ) 
  (swath_width overlap : ℝ) 
  (walking_speed : ℝ) : ℝ :=
  let effective_swath := (swath_width - overlap) / 12
  let number_of_strips := width_lawn / effective_swath
  let total_distance := number_of_strips * length_lawn
  total_distance / walking_speed

theorem moe_mowing_time_correct :
  time_to_mow_lawn 90 150 (28/12) (4/12) 5000 = 1.35 :=
by
  -- We denote the dimensions of the lawn and the parameters of mowing
  let length_lawn : ℝ := 90
  let width_lawn : ℝ := 150
  let swath_width : ℝ := (28:ℝ)/12  -- converting to feet here
  let overlap : ℝ := (4:ℝ)/12 -- converting to feet here
  let walking_speed : ℝ := 5000
  
  -- define effective swath width in feet
  let effective_swath := (swath_width - overlap) 
  
  -- define the number of strips needed
  let number_of_strips := width_lawn / effective_swath
  
  -- define the total distance mown
  let total_distance := number_of_strips * length_lawn
  
  -- define the time taken in hours
  let time := total_distance / walking_speed
  
  -- assert equivalence with the correct mowing time of 1.35 hours
  have h : time = 1.35 := sorry
  
  exact h

end moe_mowing_time_correct_l109_109819


namespace determine_number_l109_109948

theorem determine_number (x : ℝ) (number : ℝ) (h1 : number / x = 0.03) (h2 : x = 0.3) : number = 0.009 := by
  sorry

end determine_number_l109_109948


namespace term_formula_sum_S_l109_109658

variable {a : ℕ → ℤ} (d : ℤ)

-- Given that a_1 = 1
def a1 := a 1 = 1

-- Given that a_2, a_5, a_{14} form a geometric sequence
def geometric_seq_condition := 
  a 2 * a 14 = (a 5) ^ 2

-- The arithmetic sequence has a nonzero common difference 'd'
def arithmetic_seq (n : ℕ) := 
  a n = 1 + (n - 1) * d

-- Defining b_n in terms of a_n
def b (n : ℕ) := 2 ^ n * a n

-- Sum of the first n terms of the sequence {b_n}
def S (n : ℕ) := (Finset.range (n + 1)).sum b

-- Proving the general formula for the term a_n
theorem term_formula (n : ℕ) : a n = 2 * n - 1 := by
  sorry

-- Proving the sum of the first n terms S_n of the sequence {b_n}
theorem sum_S (n : ℕ) : 
  S n = -2 + 2 ^ (n + 1) * (3 - 2 * n) := by
  sorry

end term_formula_sum_S_l109_109658


namespace gcd_sum_of_abcd_dcba_is_2222_l109_109117

noncomputable def gcd_sum_of_abcd_dcba : ℕ :=
  let a b c d : ℤ := arbitrary ℤ
  have h1 : b = a + 2
  have h2 : c = a + 4
  have h3 : d = a + 6
  let abcd : ℤ := 1000 * a + 100 * b + 10 * c + d
  let dcba : ℤ := 1000 * d + 100 * c + 10 * b + a
  let sum_abcd_dcba := abcd + dcba
  gcd (map int.natAbs (2222 * (a + 3))) sorry = 2222
  
theorem gcd_sum_of_abcd_dcba_is_2222 : gcd_sum_of_abcd_dcba = 2222 :=
sorry

end gcd_sum_of_abcd_dcba_is_2222_l109_109117


namespace possible_values_of_expression_l109_109677

-- Defining the nonzero real numbers a, b, c, and d
variables {a b c d : ℝ}

-- Assuming that a, b, c, and d are nonzero
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- The expression to prove
def expression := (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem to state the possible values of the expression
theorem possible_values_of_expression : 
  expression ha hb hc hd ∈ ({5, 1, -3, -5} : set ℝ) := sorry

end possible_values_of_expression_l109_109677


namespace perfect_game_points_l109_109215

theorem perfect_game_points (points_per_game games_played total_points : ℕ) 
  (h1 : points_per_game = 21) 
  (h2 : games_played = 11) 
  (h3 : total_points = points_per_game * games_played) : 
  total_points = 231 := 
by 
  sorry

end perfect_game_points_l109_109215


namespace floor_euler_approximation_l109_109627

theorem floor_euler_approximation :
  let e := 2.71828 in ⌊e⌋ = 2 :=
by
  sorry

end floor_euler_approximation_l109_109627


namespace sum_of_solutions_eq_46_l109_109773

theorem sum_of_solutions_eq_46 (x y : ℤ) (sols : List (ℤ × ℤ)) :
  (∀ (xi yi : ℤ), (xi, yi) ∈ sols →
    (|xi - 4| = |yi - 10| ∧ |xi - 10| = 3 * |yi - 4|)) →
  (sols = [(10, 4), (5, -1), (10, 4), (-5, 19)]) →
  List.sum (sols.map (λ p, p.1 + p.2)) = 46 :=
by
  intro h1 h2
  rw [h2]
  dsimp
  norm_num

end sum_of_solutions_eq_46_l109_109773


namespace perpendicular_slope_l109_109622

theorem perpendicular_slope (a b c : ℝ) (h_line : a = 3 ∧ b = -4 ∧ c = 5) :
  let m := - (1 / (a / -b)) in m = -4 / 3 :=
by
  -- Parameters
  have ha := h_line.1
  have hb := h_line.2.1
  have hc := h_line.2.2

  -- Definition of the slope of the original line
  let original_slope := a / -b

  -- Calculate the slope of the perpendicular line
  let perp_slope := -1 / original_slope
  show perp_slope = -4 / 3 from sorry

end perpendicular_slope_l109_109622


namespace largest_four_digit_divisible_by_5_l109_109518

theorem largest_four_digit_divisible_by_5 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 5 = 0 → m ≤ n :=
begin
  use 9995,
  split, norm_num,
  split, norm_num,
  split, norm_num,
  intros m hm1 hm2 hm3,
  have H : m ≤ 9999, from hm2,
  norm_num at H,
  sorry
end

end largest_four_digit_divisible_by_5_l109_109518


namespace completing_square_transformation_l109_109909

theorem completing_square_transformation : ∀ x : ℝ, x^2 - 4 * x - 7 = 0 → (x - 2)^2 = 11 :=
by
  intros x h
  sorry

end completing_square_transformation_l109_109909


namespace problem_solution_l109_109133

theorem problem_solution (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 500 / 9 :=
by
  sorry

end problem_solution_l109_109133


namespace widgets_made_per_week_l109_109757

theorem widgets_made_per_week
  (widgets_per_hour : Nat)
  (hours_per_day : Nat)
  (days_per_week : Nat)
  (total_widgets : Nat) :
  widgets_per_hour = 20 →
  hours_per_day = 8 →
  days_per_week = 5 →
  total_widgets = widgets_per_hour * hours_per_day * days_per_week →
  total_widgets = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end widgets_made_per_week_l109_109757


namespace cindy_marbles_l109_109833

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end cindy_marbles_l109_109833


namespace compute_AT_l109_109064

variable {A B C D T X Y : Point}
variable {circle_center : Point}
variable {square_side_length : ℝ}
variable (is_square : is_square ABCD)
variable (side_length_eq : square_side_length = 5)
variable (circle_passes_through_A : OnCircle circle A)
variable (circle_tangent_T_CD : TangentToSegment circle T CD)
variable (circle_intersects_AB_at_X : OnCircle circle X ∧ X ≠ A)
variable (circle_intersects_AD_at_Y : OnCircle circle Y ∧ Y ≠ A)
variable (XY_eq_6 : distance X Y = 6)
variable {distance : Point → Point → ℝ}

theorem compute_AT :
  ∃ AT, distance A T = AT ∧ AT = Float.sqrt 30 := sorry

end compute_AT_l109_109064


namespace nat_sum_representation_l109_109807

noncomputable def S (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  if k = 1 then 0 else (Finset.range (k - 1)).sum a

theorem nat_sum_representation (a : ℕ → ℕ) (h_seq : ∀ n m, n < m → a n > a m) :
  (∀ n, ∃ U : Finset ℕ, U.sum a = n ∧ U.pairwise (≠)) ↔
  (∀ k, a k ≤ S a k + 1) := 
sorry

end nat_sum_representation_l109_109807


namespace train_crosses_signal_pole_in_12_seconds_l109_109552

noncomputable def time_to_cross_signal_pole (length_train : ℕ) (time_to_cross_platform : ℕ) (length_platform : ℕ) : ℕ :=
  let distance_train_platform := length_train + length_platform
  let speed_train := distance_train_platform / time_to_cross_platform
  let time_to_cross_pole := length_train / speed_train
  time_to_cross_pole

theorem train_crosses_signal_pole_in_12_seconds :
  time_to_cross_signal_pole 300 39 675 = 12 :=
by
  -- expected proof in the interactive mode
  sorry

end train_crosses_signal_pole_in_12_seconds_l109_109552


namespace move_line_down_l109_109105

theorem move_line_down (x : ℝ) : (y = -x + 1) → (y = -x - 2) := by
  sorry

end move_line_down_l109_109105


namespace count_valid_three_digit_numbers_l109_109051

theorem count_valid_three_digit_numbers : 
  (∃ count : ℕ, count = 120 ∧ 
    (∀ (a b c : ℕ), 
      a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
      (a ≠ b ∧ b ≠ c ∧ a ≠ c ∨ a = b ∨ b = c ∨ a = c) ∧
      (10 * a + b > 10 * b + c) ∧ 
      (10 * b + c > 10 * c + a) -> 
      true)) :=
by
  existsi 120
  split
  case inl =>
    sorry
  case inr =>
    assume a b c
    intro h
    simp at h

end count_valid_three_digit_numbers_l109_109051


namespace cos_A_and_bc_solutions_l109_109039

theorem cos_A_and_bc_solutions (
  (A B C : ℝ) (a b c : ℝ))
  (h_triangle : a^2 = b^2 + c^2 - 2 * b * c * cos A)
  (h_cos_eq : 3 * cos (B - C) - 1 = 6 * cos B * cos C)
  (h_a : a = 3)
  (h_area : 1 / 2 * b * c * sin A = 2 * sqrt 2)
  (h_sum_angles : A + B + C = π) :
  cos A = 1 / 3 ∧ (b = 2 ∧ c = 3 ∨ b = 3 ∧ c = 2) := 
by
  sorry

end cos_A_and_bc_solutions_l109_109039


namespace complex_exponential_sum_l109_109000

theorem complex_exponential_sum (γ δ : ℝ) 
  (h : Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -1 / 2 + 5 / 4 * Complex.I) :
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -1 / 2 - 5 / 4 * Complex.I :=
by
  sorry

end complex_exponential_sum_l109_109000


namespace midpoints_locus_is_circle_l109_109785

-- Let P be a point outside circle K, K is centered at O with radius r.
def locus_of_midpoints {r : ℝ} (P O : EuclideanSpace ℝ (Fin 2)) (h : dist P O > r) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {M : EuclideanSpace ℝ (Fin 2) | ∃ (A B : EuclideanSpace ℝ (Fin 2)), (A, B) ∈ K.intersects_line_through P ∧ dist A O = r ∧ dist B O = r ∧ M = midpoint A B}

-- The statement to be proven: The locus of all such midpoints is a circle centered at P.
theorem midpoints_locus_is_circle (P O : EuclideanSpace ℝ (Fin 2)) (r : ℝ) (h : dist P O > r) :
  ∃ R : ℝ, (locus_of_midpoints P O) = {M | dist M P = R} := by sorry

end midpoints_locus_is_circle_l109_109785


namespace suff_but_not_necessary_obtuse_l109_109384

theorem suff_but_not_necessary_obtuse (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 < c^2) : 
  (∃ (A B C : Type), (A = ℝ) ∧ (B = ℝ) ∧ (C = ℝ) ∧ (a = A) ∧ (b = B) ∧ (c = C)) ∧ 
  (cosine_law (A : Type) (B : Type) (C : Type) a b c h < 0) ∧ 
  (∀ ΔABC : Type, (obtuse_triangle (a b c))) :=
sorry

end suff_but_not_necessary_obtuse_l109_109384


namespace other_candidate_valid_votes_l109_109023

theorem other_candidate_valid_votes (total_votes : ℕ)
    (percent_invalid : ℚ)
    (percent_valid : ℚ)
    (percent_first_candidate : ℚ) 
    (valid_votes_first_candidate : ℕ) 
    (valid_votes_other_candidate : ℕ) :
    total_votes = 7000 →
    percent_invalid = 0.20 →
    percent_valid = 0.80 →
    percent_first_candidate = 0.55 →
    valid_votes_first_candidate = (total_votes * percent_valid * percent_first_candidate).to_nat →
    valid_votes_other_candidate = ((total_votes * percent_valid).to_nat - valid_votes_first_candidate) :=
by
  intros h1 h2 h3 h4 h5
  exact sorry

end other_candidate_valid_votes_l109_109023


namespace find_sin_beta_l109_109666

theorem find_sin_beta (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : cos (α - β) = -5 / 13)
  (h4 : sin α = 4 / 5) :
  sin β = -56 / 65 := 
by sorry

end find_sin_beta_l109_109666


namespace AD_perp_BC_l109_109423

variables {A B C I K X Y D : Type*}

-- Definitions based on conditions
def is_incenter (I : Type*) (A B C : Type*) : Prop := sorry
def incircle_tangent (K : Type*) (BC : Type*) : Prop := sorry
def on_segment (P : Type*) (L M : Type*) : Prop := sorry
def perpendicular (P Q R : Type*) : Prop := sorry
def circumscribed_circle (P Q R C : Type*) : Prop := sorry
def intersects_at (P L M : Type*) : Type* := sorry

-- Conditions
variables (I_is_incenter : is_incenter I A B C)
variables (K_tangent_BC : incircle_tangent K BC)
variables (X_on_BI : on_segment X B I)
variables (Y_on_CI : on_segment Y C I)
variables (KX_perp_AB : perpendicular K X AB)
variables (KY_perp_AC : perpendicular K Y AC)
variables (D_intersects_circum_XYK_BC : D = intersects_at (circumscribed_circle X Y K (circle X Y K))) -- assuming a function 'circle' that returns the circumscribed circle
variables (KBC_line : intersects_at K B C = BC)

-- Question (turned into theorem statement)
theorem AD_perp_BC : perpendicular A D BC :=
sorry

end AD_perp_BC_l109_109423


namespace inequality_holds_iff_m_range_l109_109287

theorem inequality_holds_iff_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 - 2 * m * x - 3 < 0) ↔ (-3 < m ∧ m ≤ 0) :=
by
  sorry

end inequality_holds_iff_m_range_l109_109287


namespace cross_section_is_rectangle_l109_109999

structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

structure Prism :=
  (A B C D A1 B1 C1 D1 : Point)
  (M N1 : Point)
  (plane_perpendicular_to_bases : ∀ (pt : Point), pt.z = A.z ∨ pt.z = A1.z → 
                                 (M.z = pt.z ∨ N1.z = pt.z))

-- Given prism and points
variable (prism : Prism)

-- To prove the intersection forms a rectangle
theorem cross_section_is_rectangle : 
  ∃ M M1 N1 N : Point, 
    (prism.plane_perpendicular_to_bases M) ∧
    (prism.plane_perpendicular_to_bases N1) ∧
    (prism.plane_perpendicular_to_bases M1) ∧ 
    (prism.plane_perpendicular_to_bases N) ∧
    (M1.z = N1.z) ∧
    (M.z = N.z) ∧
    (∃ c : ℝ, dist M M1 = c ∧ dist N N1 = c ∧ dist M N = dist M1 N1) := sorry

end cross_section_is_rectangle_l109_109999


namespace positive_integers_satisfying_sin_cos_equation_l109_109295

theorem positive_integers_satisfying_sin_cos_equation :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ∀ t : ℝ, (\sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)}
  in S.card = 125 :=
by
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ∀ t : ℝ, (\sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)}
  show S.card = 125,
  sorry

end positive_integers_satisfying_sin_cos_equation_l109_109295


namespace first_term_of_geometric_series_l109_109975

theorem first_term_of_geometric_series (r a S : ℝ) (h_r : r = 1 / 4) (h_S : S = 40) 
  (h_geometric_sum : S = a / (1 - r)) : a = 30 :=
by
  -- The proof would go here, but we place a sorry to skip the proof.
  sorry

end first_term_of_geometric_series_l109_109975


namespace Mika_birthday_stickers_l109_109444

variable (B : ℕ)

theorem Mika_birthday_stickers :
  ∀ (B : ℕ),
    let total_stickers := 20 + 26 + B in
    let used_stickers := 6 + 58 in
    total_stickers - used_stickers = 2 →
    B = 20 :=
by
  intro B
  let total_stickers := 20 + 26 + B
  let used_stickers := 6 + 58
  intro h
  -- proof steps would go here
  sorry

end Mika_birthday_stickers_l109_109444


namespace round_nearest_hundredth_problem_l109_109175

noncomputable def round_nearest_hundredth (x : ℚ) : ℚ :=
  let shifted := x * 100
  let rounded := if (shifted - shifted.floor) < 0.5 then shifted.floor else shifted.ceil
  rounded / 100

theorem round_nearest_hundredth_problem :
  let A := 34.561
  let B := 34.558
  let C := 34.5539999
  let D := 34.5601
  let E := 34.56444
  round_nearest_hundredth A = 34.56 ∧
  round_nearest_hundredth B = 34.56 ∧
  round_nearest_hundredth C ≠ 34.56 ∧
  round_nearest_hundredth D = 34.56 ∧
  round_nearest_hundredth E = 34.56 :=
sorry

end round_nearest_hundredth_problem_l109_109175


namespace height_of_cuboid_l109_109189

theorem height_of_cuboid (volume base_area : ℝ) (h1 : volume = 144) (h2 : base_area = 18) : volume / base_area = 8 := 
by
  rw [h1, h2]
  norm_num
  sorry

end height_of_cuboid_l109_109189


namespace y_minus_x_value_l109_109134

theorem y_minus_x_value (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 55.56 :=
sorry

end y_minus_x_value_l109_109134


namespace determine_B_l109_109711

open Set

-- Define the universal set U and the sets A and B
variable (U A B : Set ℕ)

-- Definitions based on the problem conditions
def U_def : U = A ∪ B := 
  by sorry

def cond1 : (U = {1, 2, 3, 4, 5, 6, 7}) := 
  by sorry

def cond2 : (A ∩ (U \ B) = {2, 4, 6}) := 
  by sorry

-- The main statement
theorem determine_B (h1 : U = {1, 2, 3, 4, 5, 6, 7}) (h2 : A ∩ (U \ B) = {2, 4, 6}) : B = {1, 3, 5, 7} :=
  by sorry

end determine_B_l109_109711


namespace circle_equation_standard_l109_109129

theorem circle_equation_standard (h k : ℝ) (r : ℝ) (H_center : h = -3) (K_center : k = 4) (R_rad : r = 2) :
  (λ x y : ℝ, (x + 3)^2 + (y - 4)^2 = 4) :=
by
  simp [H_center, K_center, R_rad]
  sorry

end circle_equation_standard_l109_109129


namespace natural_sum_representation_l109_109793

-- Definitions
def is_strictly_decreasing (a : ℕ → ℕ) := ∀ n, a (n + 1) < a n
def S (a : ℕ → ℕ) (k : ℕ) : ℕ := if k = 1 then 0 else finset.sum (finset.range (k - 1)) a

-- Theorem statement
theorem natural_sum_representation (a : ℕ → ℕ) 
  (h_dec : is_strictly_decreasing a) :
  (∀ (k : ℕ), a k ≤ S a k + 1) ↔ 
  ∀ n, ∃ s : finset ℕ, s.to_finset.sum a = n :=
sorry

end natural_sum_representation_l109_109793


namespace part_a_part_b_part_c_l109_109538

noncomputable def ctg_sq_sum (m : ℕ) : ℝ :=
  ((finset.range m).sum (λ k, real.cot ((k + 1 : ℕ) * real.pi / (2 * m + 1))) ^ 2)

theorem part_a (m : ℕ) : ctg_sq_sum m = m * (2 * m - 1) / 3 :=
  sorry

noncomputable def cosec_sq_sum (m : ℕ) : ℝ :=
  ((finset.range m).sum (λ k, real.csc ((k + 1 : ℕ) * real.pi / (2 * m + 1))) ^ 2)

theorem part_b (m : ℕ) : cosec_sq_sum m = m * (2 * m + 2) / 3 :=
  sorry

noncomputable def ctg_step_sum (n : ℕ) : ℝ :=
  ((finset.range n).sum (λ k, let x := 2 * k + 1 in real.cot (x * real.pi / (4 * n)) * if even x then 1 else -1))

theorem part_c (n : ℕ) (h : even n) : ctg_step_sum n = n :=
  sorry

end part_a_part_b_part_c_l109_109538


namespace moe_mowing_time_correct_l109_109820

noncomputable def time_to_mow_lawn 
  (length_lawn width_lawn : ℝ) 
  (swath_width overlap : ℝ) 
  (walking_speed : ℝ) : ℝ :=
  let effective_swath := (swath_width - overlap) / 12
  let number_of_strips := width_lawn / effective_swath
  let total_distance := number_of_strips * length_lawn
  total_distance / walking_speed

theorem moe_mowing_time_correct :
  time_to_mow_lawn 90 150 (28/12) (4/12) 5000 = 1.35 :=
by
  -- We denote the dimensions of the lawn and the parameters of mowing
  let length_lawn : ℝ := 90
  let width_lawn : ℝ := 150
  let swath_width : ℝ := (28:ℝ)/12  -- converting to feet here
  let overlap : ℝ := (4:ℝ)/12 -- converting to feet here
  let walking_speed : ℝ := 5000
  
  -- define effective swath width in feet
  let effective_swath := (swath_width - overlap) 
  
  -- define the number of strips needed
  let number_of_strips := width_lawn / effective_swath
  
  -- define the total distance mown
  let total_distance := number_of_strips * length_lawn
  
  -- define the time taken in hours
  let time := total_distance / walking_speed
  
  -- assert equivalence with the correct mowing time of 1.35 hours
  have h : time = 1.35 := sorry
  
  exact h

end moe_mowing_time_correct_l109_109820


namespace dime_probability_l109_109951

theorem dime_probability (dime_value quarter_value : ℝ) (dime_worth quarter_worth total_coins: ℕ) :
  dime_value = 0.10 ∧
  quarter_value = 0.25 ∧
  dime_worth = 10 ∧
  quarter_worth = 4 ∧
  total_coins = 14 →
  (dime_worth / total_coins : ℝ) = 5 / 7 :=
by
  sorry

end dime_probability_l109_109951


namespace hot_dogs_remainder_l109_109721

theorem hot_dogs_remainder :
  25197625 % 4 = 1 :=
by
  sorry

end hot_dogs_remainder_l109_109721


namespace chord_length_constant_l109_109859

open EuclideanGeometry

variables {P Q R A B C D : Point}

-- Definitions and assumptions in Lean
def circles_intersect (Q R : Circle) (A B : Point) : Prop :=
  A ∈ Q ∧ A ∈ R ∧ B ∈ Q ∧ B ∈ R

def is_on_arc_outside (P : Point) (Q R : Circle) (A B : Point) : Prop :=
  P ∈ Q ∧ ¬(P ∈ R)

def chord_defined_by_projection (P Q R A B C D : Point) : Prop :=
  -- P projects through A and B to define chord CD on circle R
  -- Here we need to confirm this logically/mathematically
  true -- Placeholder, as actual geometric construction proofs are complex.
  
def chord_length_invariant (Q R : Circle) (A B : Point) (C D : Point) : Prop :=
  -- The length of chord CD is invariant
  true -- Placeholder

theorem chord_length_constant (Q R : Circle) (A B : Point) :
  circles_intersect Q R A B →
  (∀ P, is_on_arc_outside P Q R A B → ∃ C D, chord_defined_by_projection P Q R A B C D) →
  (∀ P C D, is_on_arc_outside P Q R A B → chord_defined_by_projection P Q R A B C D →
    chord_length_invariant Q R A B C D) :=
by
  intros h_intersect h_projection h_invariant
  sorry

end chord_length_constant_l109_109859


namespace star_contains_2011_l109_109042

theorem star_contains_2011 :
  ∃ (n : ℕ), n = 183 ∧ 
  (∃ (seq : List ℕ), seq = List.range' (2003) 11 ∧ 2011 ∈ seq) :=
by
  sorry

end star_contains_2011_l109_109042


namespace probability_of_double_application_2012_l109_109065

-- Define the set of numbers from 1 to 2012
def S : finset ℕ := (finset.range 2012).image (λ x, x + 1)

-- Define a permutation type
def perm (α : Type*) := equiv.perm α

-- Define the specific permutation condition for 2012
def perm_condition (π : perm (fin 2012)) : Prop :=
  (π (π ⟨2011, by simp⟩) = ⟨2011, by simp⟩)

-- The theorem stating the probability
theorem probability_of_double_application_2012 :
  let permutations := fintype.card (perm (fin 2012)),
      favorable := fintype.card { π : perm (fin 2012) // perm_condition π } in
  (favorable.to_float / permutations.to_float : ℝ) = 1 / 1006 :=
by
  sorry

end probability_of_double_application_2012_l109_109065


namespace number_of_permutations_l109_109360

def total_letters : ℕ := 10
def freq_s : ℕ := 3
def freq_t : ℕ := 2
def freq_i : ℕ := 2
def freq_a : ℕ := 1
def freq_c : ℕ := 1

theorem number_of_permutations : 
  (total_letters.factorial / (freq_s.factorial * freq_t.factorial * freq_i.factorial * freq_a.factorial * freq_c.factorial)) = 75600 :=
by
  sorry

end number_of_permutations_l109_109360


namespace share_cookies_l109_109196

/-- Each person gets 4 cookies when 24 cookies are shared equally among 6 people. -/
theorem share_cookies : ∀ (cookies : ℕ) (people : ℕ), cookies = 24 → people = 6 → cookies / people = 4 := 
by {
  intros cookies people h_cookies h_people,
  rw [h_cookies, h_people],
  norm_num,
  sorry
}

end share_cookies_l109_109196


namespace nat_sum_representation_l109_109806

noncomputable def S (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  if k = 1 then 0 else (Finset.range (k - 1)).sum a

theorem nat_sum_representation (a : ℕ → ℕ) (h_seq : ∀ n m, n < m → a n > a m) :
  (∀ n, ∃ U : Finset ℕ, U.sum a = n ∧ U.pairwise (≠)) ↔
  (∀ k, a k ≤ S a k + 1) := 
sorry

end nat_sum_representation_l109_109806


namespace find_total_results_l109_109107

noncomputable def total_results (S : ℕ) (n : ℕ) (sum_first6 sum_last6 sixth_result : ℕ) :=
  (S = 52 * n) ∧ (sum_first6 = 6 * 49) ∧ (sum_last6 = 6 * 52) ∧ (sixth_result = 34)

theorem find_total_results {S n sum_first6 sum_last6 sixth_result : ℕ} :
  total_results S n sum_first6 sum_last6 sixth_result → n = 11 :=
by
  intros h
  sorry

end find_total_results_l109_109107


namespace range_of_a_l109_109377

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, 2^x + 1 + Real.log2(a) = 0) : 0 < a ∧ a < 1 / 2 :=
sorry

end range_of_a_l109_109377


namespace sum_exponents_prime_factors_sqrt_largest_perfect_square_l109_109523

/-- The sum of the exponents of the prime factors of the square root 
    of the largest perfect square that divides 12! is 8. -/
theorem sum_exponents_prime_factors_sqrt_largest_perfect_square (n : ℕ) (h : n = 12!) :
  ∑ p in (12!).factors.toFinset, (nat.sqrt 12!).factors.count p = 8 :=
sorry

end sum_exponents_prime_factors_sqrt_largest_perfect_square_l109_109523


namespace perpendicular_vectors_eq_one_l109_109353

noncomputable def vec_a : (ℝ × ℝ) := (3, 1)
noncomputable def vec_b (t : ℝ) : (ℝ × ℝ) := (t, -3)

theorem perpendicular_vectors_eq_one (t : ℝ) (h : vec_a.1 * vec_b t.1 + vec_a.2 * vec_b t.2 = 0) : t = 1 :=
by
  -- skip the actual proof
  sorry

end perpendicular_vectors_eq_one_l109_109353


namespace find_integer_solutions_l109_109631

theorem find_integer_solutions :
  {n : ℤ | n + 2 ∣ n^2 + 3} = {-9, -3, -1, 5} :=
  sorry

end find_integer_solutions_l109_109631


namespace ratio_area_of_squares_l109_109099

theorem ratio_area_of_squares :
  ∀ (t : ℝ), 
  (∀ (WI IX : ℝ), WI = 3 * IX → let WX := 4 * t in WI + IX = WX) →
  ∃ (side_IJKL side_WXYZ : ℝ),
  let area_IJKL := (side_IJKL)^2,
  let area_WXYZ := (side_WXYZ)^2 in
  side_WXYZ = 4 * t ∧ side_IJKL = t * Real.sqrt 2 →
  area_IJKL / area_WXYZ = 1 / 8 :=
by 
  intros t h wi ix hwi hwx
  have h1 : wi + ix = 4 * t := h wi ix hwi
  have side_WXYZ := 4 * t
  have side_IJKL := t * Real.sqrt 2
  use [side_IJKL, side_WXYZ]
  use Real.sqrt 2
  split
  { exact side_WXYZ, }
  { exact side_IJKL, }
  { sorry }

end ratio_area_of_squares_l109_109099


namespace hamza_bucket_problem_l109_109357

theorem hamza_bucket_problem : 
  let initial_volume_in_5_liter_bucket := 5
      volume_in_3_liter_bucket := 3
      remaining_volume := initial_volume_in_5_liter_bucket - volume_in_3_liter_bucket
      volume_in_6_liter_bucket := remaining_volume
      additional_volume_needed := 6 - volume_in_6_liter_bucket
  in 
  additional_volume_needed = 4 :=
by
  let initial_volume_in_5_liter_bucket := 5
  let volume_in_3_liter_bucket := 3
  let remaining_volume := initial_volume_in_5_liter_bucket - volume_in_3_liter_bucket
  let volume_in_6_liter_bucket := remaining_volume
  let additional_volume_needed := 6 - volume_in_6_liter_bucket
  show additional_volume_needed = 4 from sorry

end hamza_bucket_problem_l109_109357


namespace sqrt2_same_type_sqrt_1_over_8_l109_109234

def sqrt_same_type (x y : ℝ) : Prop := ∃ (a : ℝ), is_rational a ∧ y = a * x

theorem sqrt2_same_type_sqrt_1_over_8 :
  sqrt_same_type (Real.sqrt 2) (Real.sqrt (1 / 8)) :=
by sorry

end sqrt2_same_type_sqrt_1_over_8_l109_109234


namespace sum_of_solutions_eq_46_l109_109776

theorem sum_of_solutions_eq_46 (x y : ℤ) (sols : List (ℤ × ℤ)) :
  (∀ (xi yi : ℤ), (xi, yi) ∈ sols →
    (|xi - 4| = |yi - 10| ∧ |xi - 10| = 3 * |yi - 4|)) →
  (sols = [(10, 4), (5, -1), (10, 4), (-5, 19)]) →
  List.sum (sols.map (λ p, p.1 + p.2)) = 46 :=
by
  intro h1 h2
  rw [h2]
  dsimp
  norm_num

end sum_of_solutions_eq_46_l109_109776


namespace measure_weights_l109_109525

/-- A statement that all whole number weights up to 40 grams are measurable using a balance
    and weights of 1, 3, 9, 27 grams.
-/
theorem measure_weights : 
  ∀ (n : ℕ), n ≤ 40 → ∃ (a b c d : ℤ), a ∈ {0, 1, -1} ∧ b ∈ {0, 1, -1} ∧ c ∈ {0, 1, -1} ∧ d ∈ {0, 1, -1} ∧
  n = 27 * a + 9 * b + 3 * c + d :=
sorry

end measure_weights_l109_109525


namespace sum_of_distinct_terms_if_and_only_if_l109_109800

theorem sum_of_distinct_terms_if_and_only_if
  (a : ℕ → ℕ)
  (h_decreasing : ∀ k, a k > a (k + 1))
  (S : ℕ → ℕ)
  (h_S : ∀ k, S k = (∑ i in finset.range (k - 1), a i))
  (h_S1 : S 1 = 0) :
  (∀ n : ℕ, ∃ (l : list ℕ), (∀ (i : ℕ), i ∈ l → ∃ k, i = a k) ∧  l.sum = n) ↔ (∀ k, a k ≤ S k + 1) :=
begin
  sorry,
end

end sum_of_distinct_terms_if_and_only_if_l109_109800


namespace geometric_sequence_a7_value_l109_109031

theorem geometric_sequence_a7_value 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h₁ : a 4 = 8) 
  (h₂ : q = -2) 
  : a 7 = -64 :=
by 
  have h₃ : a 4 = a 1 * q^3 := by sorry
  have h₄ : 8 = a 1 * (-2)^3 := by rw [h₁, h₂]
  have h₅ : 8 = a 1 * (-8) := by rw [pow_succ' (-2) 2, mul_neg, pow_two]
  have h₆ : 8 = -8 * a 1 := by rw [mul_comm]
  have h₇ : a 1 = -1 := by linarith
  have h₈ : a n = -(-2)^(n-1) := by sorry
  show a 7 = -64 from by 
    rw [h₈, h₂]
    norm_num

end geometric_sequence_a7_value_l109_109031


namespace possible_values_expression_l109_109671

theorem possible_values_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (val : ℤ), val ∈ {5, 2, 1, -2, -3} ∧ 
  val = (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (a * b * c * d / |a * b * c * d|) :=
begin
  sorry
end

end possible_values_expression_l109_109671


namespace actual_distance_A_to_B_km_l109_109825

-- Given conditions
def map_scale : ℝ := 20000
def measured_distance_cm : ℝ := 8
def actual_distance_cm := measured_distance_cm * map_scale
def cm_to_km (cm : ℝ) : ℝ := cm / 100000

-- Statement of the problem
theorem actual_distance_A_to_B_km :
  cm_to_km actual_distance_cm = 1.6 :=
by
  sorry

end actual_distance_A_to_B_km_l109_109825


namespace parallelogram_area_l109_109610

open Matrix

noncomputable def vector3 := (ℝ × ℝ × ℝ)
noncomputable def vec_sub (a b : vector3) : vector3 :=
  (a.1 - b.1, a.2 - b.2, a.3 - b.3)

noncomputable def cross_product (a b : vector3) : vector3 :=
  ((a.2 * b.3 - a.3 * b.2), (a.3 * b.1 - a.1 * b.3), (a.1 * b.2 - a.2 * b.1))

noncomputable def magnitude (v : vector3) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def P : vector3 := (2, -5, 3)
noncomputable def Q : vector3 := (4, -9, 6)
noncomputable def R : vector3 := (1, -4, 1)
noncomputable def S : vector3 := (3, -8, 4)

theorem parallelogram_area :
  let pq := vec_sub Q P,
      sr := vec_sub S R,
      rp := vec_sub R P,
      cross := cross_product pq rp in
  (pq = sr) ∧ (magnitude cross = real.sqrt 62) :=
by
  sorry

end parallelogram_area_l109_109610


namespace midpoint_of_parabola_and_line_l109_109479

theorem midpoint_of_parabola_and_line :
  ∀ {x1 x2 y1 y2 : ℝ},
  (y1^2 = 4 * x1) ∧ (y2^2 = 4 * x2) ∧ (y1 = x1 - 1) ∧ (y2 = x2 - 1) → 
  ((x1 + x2) / 2 = 3) ∧ ((y1 + y2) / 2 = 2) :=
by
  intros x1 x2 y1 y2 h,
  sorry

end midpoint_of_parabola_and_line_l109_109479


namespace value_of_a_plus_b_l109_109312

variable (a b : ℝ)
variable (h1 : |a| = 5)
variable (h2 : |b| = 2)
variable (h3 : a < 0)
variable (h4 : b > 0)

theorem value_of_a_plus_b : a + b = -3 :=
by
  sorry

end value_of_a_plus_b_l109_109312


namespace umbrella_number_count_l109_109224

/-- A three-digit number formed by selecting three different digits 
    from the set {2, 3, 4, 5, 6, 9} is called an "umbrella number" if 
    the digit in the tens place is greater than both the digit in the 
    units place and the digit in the hundreds place. -/
def umbrella_number (d1 d2 d3 : ℕ) : Prop :=
  (d1 ≠ d2) ∧ (d2 ≠ d3) ∧ (d1 ≠ d3) ∧ 
  (d2 > d1) ∧ (d2 > d3) ∧ 
  d1 ∈ {2, 3, 4, 5, 6, 9} ∧ 
  d2 ∈ {2, 3, 4, 5, 6, 9} ∧ 
  d3 ∈ {2, 3, 4, 5, 6, 9}

theorem umbrella_number_count : 
  ∃ (n : ℕ), n = (Σ (d1 d2 d3 : ℕ), if umbrella_number d1 d2 d3 then 1 else 0) ∧ n = 40 :=
by
  sorry

end umbrella_number_count_l109_109224


namespace full_price_shoes_l109_109088

variable (P : ℝ)

def full_price (P : ℝ) : ℝ := P
def discount_1_year (P : ℝ) : ℝ := 0.80 * P
def discount_3_years (P : ℝ) : ℝ := 0.75 * discount_1_year P
def price_after_discounts (P : ℝ) : ℝ := 0.60 * P

theorem full_price_shoes : price_after_discounts P = 51 → full_price P = 85 :=
by
  -- Placeholder for proof steps,
  sorry

end full_price_shoes_l109_109088


namespace length_RS_14_l109_109884

-- Definitions of conditions
def edges : List ℕ := [8, 14, 19, 28, 37, 42]
def PQ_length : ℕ := 42

-- Problem statement
theorem length_RS_14 (edges : List ℕ) (PQ_length : ℕ) (h : PQ_length = 42) (h_edges : edges = [8, 14, 19, 28, 37, 42]) :
  ∃ RS_length : ℕ, RS_length ∈ edges ∧ RS_length = 14 :=
by
  sorry

end length_RS_14_l109_109884


namespace y_minus_x_value_l109_109135

theorem y_minus_x_value (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 55.56 :=
sorry

end y_minus_x_value_l109_109135


namespace problem1_stmt_problem2_stmt_l109_109603

noncomputable def problem1 : ℤ :=
  ( -1 : ℤ ) ^ (2020 : ℤ) * ((2020 : ℝ) - real.pi)^0 - 1

theorem problem1_stmt : problem1 = 0 :=
by sorry

noncomputable def problem2 (x y : ℝ) : ℝ :=
  x * (y - 5) + y * (3 - x)

theorem problem2_stmt (x y : ℝ) : problem2 x y = 3 * y - 5 * x :=
by sorry

end problem1_stmt_problem2_stmt_l109_109603


namespace min_measurements_3x3_grid_l109_109930

/--
Given a \(3 \times 3\) electrical grid with 16 nodes where each pair of nodes needs to be checked for connectivity,
prove that the minimum number of measurements required is 8.
-/
theorem min_measurements_3x3_grid (nodes : ℕ) (pairs : ℕ) : 
  nodes = 16 ∧ pairs = 8 → 
  ∃ (m : ℕ), m = pairs ∧ 
  (∀ (x y : ℕ), x ≠ y → measure_connection(x, y, m)) := sorry

end min_measurements_3x3_grid_l109_109930


namespace intersection_of_tangents_lies_on_line_l109_109456

def parabola (p : ℝ) : set (ℝ × ℝ) := {P | P.2^2 = 2 * p * P.1}

theorem intersection_of_tangents_lies_on_line (p : ℝ) (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) ∈ parabola p) (h2 : (x2, y2) ∈ parabola p)
  (h_tangent1 : ∀ x y, y = y1 → y1 * y = p * (x + x1)) -- ensure tangent constraint
  (h_tangent2 : ∀ x y, y = y2 → y2 * y = p * (x + x2))  -- ensure tangent constraint
  (h_y1_ne : y1 ≠ y2) :
  true := 
sorry

end intersection_of_tangents_lies_on_line_l109_109456


namespace daily_shoppers_correct_l109_109550

noncomputable def daily_shoppers (P : ℝ) : Prop :=
  let weekly_taxes : ℝ := 6580
  let daily_taxes := weekly_taxes / 7
  let percent_taxes := 0.94
  percent_taxes * P = daily_taxes

theorem daily_shoppers_correct : ∃ P : ℝ, daily_shoppers P ∧ P = 1000 :=
by
  sorry

end daily_shoppers_correct_l109_109550


namespace largest_pentagon_angle_is_179_l109_109208

-- Define the interior angles of the pentagon
def angle1 (x : ℝ) := x + 2
def angle2 (x : ℝ) := 2 * x + 3
def angle3 (x : ℝ) := 3 * x - 5
def angle4 (x : ℝ) := 4 * x + 1
def angle5 (x : ℝ) := 5 * x - 1

-- Define the sum of the interior angles of a pentagon
def pentagon_angle_sum := angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36

-- Define the largest angle function
def largest_angle (x : ℝ) := 5 * x - 1

-- The main theorem stating the largest angle measure
theorem largest_pentagon_angle_is_179 (h : angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36 = 540) :
  largest_angle 36 = 179 :=
sorry

end largest_pentagon_angle_is_179_l109_109208


namespace range_of_m_l109_109687

theorem range_of_m (m : ℝ) : (∃ x : ℝ, 3^x - m = 0) → m > 0 :=
by 
  intros h,
  obtain ⟨x, hx⟩ := h,
  have hm : m = 3^x := eq_add_of_sub_eq' hx,
  have h3x : 3^x > 0 := by sorry, 
  rw hm,
  exact h3x

end range_of_m_l109_109687


namespace f_is_periodic_l109_109052

variables {R : Type*} [Real R]

def is_periodic (f : R → R) (T : R) : Prop := ∀ x, f(x + T) = f(x)

theorem f_is_periodic 
  (a : R) (b c d : R) (f : R → R) 
  (h_rational : ∃ p q : ℤ, q ≠ 0 ∧ a = p / q) 
  (h_func_eq : ∀ x, f(x + a + b) - f(x + b) = c * (x + 2 * a + ⌊x⌋ - 2 * ⌊x + a⌋ - ⌊b⌋) + d)
  (h_range : ∀ x, f x ∈ -1..1) : 
  ∃ T : R, is_periodic f T := 
sorry

end f_is_periodic_l109_109052


namespace find_hourly_charge_computer_B_l109_109960

noncomputable def hourly_charge_computer_B (B : ℝ) :=
  ∃ (A h : ℝ),
    A = 1.4 * B ∧
    B * (h + 20) = 550 ∧
    A * h = 550 ∧
    B = 7.86

theorem find_hourly_charge_computer_B : ∃ B : ℝ, hourly_charge_computer_B B :=
  sorry

end find_hourly_charge_computer_B_l109_109960


namespace quadrilateral_area_and_ratio_l109_109904

noncomputable def triangles (O Q A B C D E : Point) : Prop :=
  circles_intersect_at O Q A B ∧
  angle_bisector_intersects O Q A B C D ∧
  intersect_at AD OQ E ∧
  area_triangle O A E = 18 ∧
  area_triangle Q A E = 42

theorem quadrilateral_area_and_ratio (O Q A B C D E : Point) (h : triangles O Q A B C D E) :
  area_quadrilateral O A Q D = 200 ∧ ratio BC BD = 3 / 7 :=
sorry

end quadrilateral_area_and_ratio_l109_109904


namespace vector_magnitude_l109_109354

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-2, 3)
def vector_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)

theorem vector_magnitude : ∥vector_sum∥ = Real.sqrt 26 :=
by
  sorry

end vector_magnitude_l109_109354


namespace angle_CBA_is_110_l109_109754

noncomputable def triangle (A B C : Type) := (M : Type)

variables {A B C M : Type}
variables (triangleABC : triangle A B C)
variables (BM : M)
variables (AB : A → B → Prop)
variables (BM_length_half_of_AB : ∀ a b m, AB a b → 2 * BM_length m = AB_length a b)
variables (angle_MBA : ∀ a m b, BM_length m = β(a, b) → angle m b a = 40)

theorem angle_CBA_is_110 : ∃ c b a, ‖CB‖ = triangleABC → angle c b a = 110 :=
by
  sorry

end angle_CBA_is_110_l109_109754


namespace exchanges_count_l109_109181

theorem exchanges_count (n : ℕ) :
  ∀ (initial_pencils_XZ initial_pens_XL : ℕ) 
    (pencils_per_exchange pens_per_exchange : ℕ)
    (final_pencils_multiplier : ℕ)
    (pz : initial_pencils_XZ = 200) 
    (pl : initial_pens_XL = 20)
    (pe : pencils_per_exchange = 6)
    (se : pens_per_exchange = 1)
    (fm : final_pencils_multiplier = 11),
    (initial_pencils_XZ - pencils_per_exchange * n = final_pencils_multiplier * (initial_pens_XL - pens_per_exchange * n)) ↔ n = 4 :=
by
  intros initial_pencils_XZ initial_pens_XL pencils_per_exchange pens_per_exchange final_pencils_multiplier pz pl pe se fm
  sorry

end exchanges_count_l109_109181


namespace multiples_of_7_between_20_and_150_l109_109361

def number_of_multiples_of_7_between (a b : ℕ) : ℕ :=
  (b / 7) - (a / 7) + (if a % 7 = 0 then 1 else 0)

theorem multiples_of_7_between_20_and_150 : number_of_multiples_of_7_between 21 147 = 19 := by
  sorry

end multiples_of_7_between_20_and_150_l109_109361


namespace g_inv_f_of_15_l109_109309

variable (f g : ℝ → ℝ)

theorem g_inv_f_of_15 (hf : ∀ x, f⁻¹(g(x)) = x^4 - 4) (hg_inv : ∃ g_inv, ∀ y, g (g_inv y) = y ∧ f⁻¹(g_inv (f 15)) = g⁻¹(f 15)) :
  g⁻¹(f 15) = real.root 4 19 :=
by
  sorry

end g_inv_f_of_15_l109_109309


namespace area_of_triangle_intercepts_l109_109617

theorem area_of_triangle_intercepts (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2 * (x - 4) * (x + 3)) : 
  let intercepts := {p : ℝ × ℝ | f p.1 = 0 ∨ p.1 = 0}
  ∀ p q r ∈ intercepts, 
  p ≠ q → q ≠ r → p ≠ r → 
  ∃ (base height : ℝ), (base = 7) ∧ (height = 0) ∧ 
  let area := (1/2) * base * height in 
  area = 0 :=
by
  sorry

end area_of_triangle_intercepts_l109_109617


namespace sequence_an_properties_l109_109320

theorem sequence_an_properties
(S : ℕ → ℝ) (a : ℕ → ℝ)
(h_mean : ∀ n, 2 * a n = S n + 2) :
a 1 = 2 ∧ a 2 = 4 ∧ ∀ n, a n = 2 ^ n :=
by
  sorry

end sequence_an_properties_l109_109320


namespace sum_of_solutions_l109_109777

theorem sum_of_solutions :
  let solutions := [(-8, -2), (-1, 5), (10, 4), (10, 4)],
  (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 :=
by
  let solutions : List (ℤ × ℤ) := [(-8, -2), (-1, 5), (10, 4), (10, 4)]
  have h1 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 4| = |y - 10| := sorry
  have h2 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 10| = 3 * |y - 4| := sorry
  have solution_sum : (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 := by
    simp [solutions]
    norm_num
  exact solution_sum

end sum_of_solutions_l109_109777


namespace intersect_sum_l109_109344

noncomputable def intersect_points (y : ℝ → ℝ) (k : ℝ) (x1 x2 x3 : ℝ) : Prop :=
  y x1 = k ∧ y x2 = k ∧ y x3 = k ∧ x1 < x2 ∧ x2 < x3

theorem intersect_sum (x1 x2 x3 : ℝ) (h1 : x1 + x2 = π / 3) (h2 : x2 + x3 = 4 * π / 3) :
  x1 + 2 * x2 + x3 = 5 * π / 3 := by
  sorry

end intersect_sum_l109_109344


namespace enlarged_area_ratio_l109_109964

theorem enlarged_area_ratio (s : ℝ) : 
  let original_area := s^2,
      new_side := 3 * s,
      new_area := (3 * s)^2
  in new_area / original_area = 9 :=
by
  let original_area := s^2
  let new_side := 3 * s
  let new_area := new_side^2
  have h : new_area = 9 * s^2 := by sorry
  show new_area / original_area = 9 from by sorry

end enlarged_area_ratio_l109_109964


namespace binomial_expansion_example_l109_109600

theorem binomial_expansion_example :
  57^3 + 3 * (57^2) * 4 + 3 * 57 * (4^2) + 4^3 = 226981 :=
by
  -- The proof would go here, using the steps outlined.
  sorry

end binomial_expansion_example_l109_109600


namespace supermarket_flour_import_l109_109965

theorem supermarket_flour_import :
  let long_grain_rice := (9 : ℚ) / 20
  let glutinous_rice := (7 : ℚ) / 20
  let combined_rice := long_grain_rice + glutinous_rice
  let less_amount := (3 : ℚ) / 20
  let flour : ℚ := combined_rice - less_amount
  flour = (13 : ℚ) / 20 :=
by
  sorry

end supermarket_flour_import_l109_109965


namespace product_of_sums_is_integer_product_is_square_of_integer_l109_109230

theorem product_of_sums_is_integer 
  (a : ℕ → ℝ) 
  (h : ∀ i, 1 ≤ i ∧ i ≤ 100 → a i = Real.sqrt i) :
  ∃ (P : ℤ), P = ∏ (s : Fin 100 → ℝ) in Finset.univ.image (λ f, ∑ i in Finset.range 100, ↑(2 * (if f i then 1 else 0) - 1) * a i), id :=
sorry

theorem product_is_square_of_integer 
  (a : ℕ → ℝ) 
  (h : ∀ i, 1 ≤ i ∧ i ≤ 100 → a i = Real.sqrt i) :
  ∃ (P : ℤ), P^2 = ∏ (s : Fin 100 → ℝ) in Finset.univ.image (λ f, ∑ i in Finset.range 100, ↑(2 * (if f i then 1 else 0) - 1) * a i), id :=
sorry

end product_of_sums_is_integer_product_is_square_of_integer_l109_109230


namespace hamza_bucket_problem_l109_109356

theorem hamza_bucket_problem :
  let bucket_sizes := (3, 5, 6)
  let initial_full_bucket := 5
  let transfer_to_small_bucket := min initial_full_bucket 3
  let remaining_in_full_bucket := initial_full_bucket - transfer_to_small_bucket
  let transferred_to_large_bucket := remaining_in_full_bucket
  let total_large_bucket_capacity := 6
  total_large_bucket_capacity - transferred_to_large_bucket = 4 := by
  let bucket_sizes := (3, 5, 6)
  let initial_full_bucket := 5
  let transfer_to_small_bucket := min initial_full_bucket 3
  let remaining_in_full_bucket := initial_full_bucket - transfer_to_small_bucket
  let transferred_to_large_bucket := remaining_in_full_bucket
  let total_large_bucket_capacity := 6
  show total_large_bucket_capacity - transferred_to_large_bucket = 4 from sorry

end hamza_bucket_problem_l109_109356


namespace natural_number_representable_l109_109804

def strictly_decreasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) < a n

def sum_of_first_k_minus_one (a : ℕ → ℕ) (k : ℕ) : ℕ :=
(if k = 1 then 0 else (Nat.range (k - 1)).sum a)

theorem natural_number_representable
  (a : ℕ → ℕ)
  (h_decreasing : strictly_decreasing_seq a) :
  (∀ n : ℕ, ∃ (s : set ℕ), s.finite ∧ (∀ i ∈ s, i < n) ∧ (s.sum id = n)) ↔ (∀ k : ℕ, a k ≤ sum_of_first_k_minus_one a k + 1) :=
sorry

end natural_number_representable_l109_109804


namespace find_a_l109_109002

theorem find_a (a b c : ℝ) (h1 : b = 15) (h2 : c = 5)
  (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) 
  (result : a * 15 * 5 = 2) : a = 6 := by 
  sorry

end find_a_l109_109002


namespace horner_method_complexity_l109_109157

variable {α : Type*} [Field α]

/-- Evaluating a polynomial of degree n using Horner's method requires exactly n multiplications
    and n additions, and 0 exponentiations.  -/
theorem horner_method_complexity (n : ℕ) (a : Fin (n + 1) → α) (x₀ : α) :
  ∃ (muls adds exps : ℕ), 
    (muls = n) ∧ (adds = n) ∧ (exps = 0) :=
by
  sorry

end horner_method_complexity_l109_109157


namespace arrange_abc_l109_109056

noncomputable def a : ℝ := ∫ x in 0..2, x^2
noncomputable def b : ℝ := ∫ x in 0..2, x^3
noncomputable def c : ℝ := ∫ x in 0..2, Real.sin x

theorem arrange_abc : c < a ∧ a < b := 
by 
  sorry

end arrange_abc_l109_109056


namespace total_cost_without_measures_minimum_preventive_cost_l109_109659

-- Define constants and conditions
def p_event : ℝ := 0.3
def loss : ℝ := 4
def cost_A : ℝ := 0.45
def p_not_event_A : ℝ := 0.9
def cost_B : ℝ := 0.3
def p_not_event_B : ℝ := 0.85

-- Statement of the main theorems
theorem total_cost_without_measures : total_cost == 120 :=
by
  let expected_loss := p_event * loss
  expected_loss == 120
  sorry

theorem minimum_preventive_cost : minimum_cost == 81 :=
by
  let cost_no_measures := p_event * loss
  let cost_A_alone := cost_A + loss * (1 - p_not_event_A)
  let cost_B_alone := cost_B + loss * (1 - p_not_event_B)
  let cost_both := cost_A + cost_B + loss * (1 - p_not_event_A) * (1 - p_not_event_B)
  minimum_cost == min cost_no_measures (min cost_A_alone (min cost_B_alone cost_both)) == 81
  sorry

end total_cost_without_measures_minimum_preventive_cost_l109_109659


namespace inequality_proof_l109_109285

variable (n : ℕ) (a : ℕ → ℕ) (x : ℝ)
hypothesis h1 : n ≥ 2
hypothesis h2 : ∀ i : ℕ, 1 ≤ i → i ≤ n → a i < a (i + 1)
hypothesis h3 : ∑ i in Finset.range n, (1 / a (i + 1)) ≤ 1

theorem inequality_proof : (∑ i in Finset.range n, (1 / (a (i + 1))^2 + x^2))^2 ≤ (1 / 2) * (1 / (a 1 * (a 1 - 1) + x^2)) :=
by
  sorry

end inequality_proof_l109_109285


namespace boys_laps_eq_27_l109_109036

noncomputable def miles_per_lap : ℝ := 3 / 4
noncomputable def girls_miles : ℝ := 27
noncomputable def girls_extra_laps : ℝ := 9

theorem boys_laps_eq_27 :
  (∃ boys_laps girls_laps : ℝ, 
    girls_laps = girls_miles / miles_per_lap ∧ 
    boys_laps = girls_laps - girls_extra_laps ∧ 
    boys_laps = 27) :=
by
  sorry

end boys_laps_eq_27_l109_109036


namespace fifteenth_entry_in_sequence_is_30_l109_109640

def r_8 (n : ℕ) : ℕ := n % 8

def sequence_condition (n : ℕ) : Prop := r_8 (7 * n) ≤ 3

noncomputable def sequence : List ℕ :=
  List.filter sequence_condition (List.range 200) -- A large enough range to cover the first 15 terms

theorem fifteenth_entry_in_sequence_is_30 :
  sequence.nth 14 = some 30 :=
by
  sorry

end fifteenth_entry_in_sequence_is_30_l109_109640


namespace collinear_dot_product_l109_109350

open Real

noncomputable def a (λ : ℝ) : ℝ × ℝ := (2, λ)
def b : ℝ × ℝ := (-3, 1)

theorem collinear_dot_product (λ : ℝ) (h_collinear : 2 * 1 = -3 * λ) :
  (a λ).1 * b.1 + (a λ).2 * b.2 = -20 / 3 :=
by
  rcases h_collinear with rfl
  sorry

end collinear_dot_product_l109_109350


namespace root_interval_2_5_to_2_75_l109_109253

noncomputable def f (x : ℝ) := log x + x - 3

-- Check the signs of the function values at given points
axiom f_225_neg : f 2.25 < 0
axiom f_275_pos : f 2.75 > 0
axiom f_25_neg : f 2.5 < 0
axiom f_3_pos : f 3 > 0
axiom f_continuous : continuous_on f (Icc 2 3)

theorem root_interval_2_5_to_2_75 : ∃ x ∈ Ioo 2.5 2.75, f x = 0 :=
by {
  apply exists_has_deriv_at_eq_zero_on_Icc,
  exact f_continuous,
  left,
  apply f_25_neg,
  apply f_275_pos,
  sorry
}

end root_interval_2_5_to_2_75_l109_109253


namespace sin_cos_identity_count_sin_cos_identity_l109_109290

theorem sin_cos_identity (n : ℕ) (h₁ : n ≤ 500) :
  (∀ t : ℝ, (sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)) ↔
  (∃ k : ℕ, n = 4 * k + 1) :=
sorry

theorem count_sin_cos_identity :
  ∃ m : ℕ, m = 125 ∧ ∀ n : ℕ, n ≤ 500 → (∀ t : ℝ, (sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)) ↔
  n = 4 * (n / 4) + 1 :=
sorry

end sin_cos_identity_count_sin_cos_identity_l109_109290


namespace cost_price_of_ball_l109_109077

theorem cost_price_of_ball (x : ℝ) (h : 17 * x - 5 * x = 720) : x = 60 :=
by {
  sorry
}

end cost_price_of_ball_l109_109077


namespace cindy_marbles_l109_109834

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end cindy_marbles_l109_109834


namespace mutually_exclusive_but_not_contrary_l109_109392

-- Definitions of events based on the conditions
def at_least_one_girl (selection : list string) : Prop :=
  list.count selection "girl" ≥ 1

def both_girls (selection : list string) : Prop :=
  list.count selection "girl" = 2

def exactly_one_girl (selection : list string) : Prop :=
  list.count selection "girl" = 1

def exactly_two_girls (selection : list string) : Prop :=
  list.count selection "girl" = 2

-- Proof that "Exactly 1 girl" and "Exactly 2 girls" are mutually exclusive and not contrary.
theorem mutually_exclusive_but_not_contrary :
  (∀ selection, ¬ (exactly_one_girl selection ∧ exactly_two_girls selection)) ∧
  (∃ selection, ¬ at_least_one_girl selection ∧ ¬ both_girls selection) :=
by
  sorry

end mutually_exclusive_but_not_contrary_l109_109392


namespace perpendicular_lines_l109_109691

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, (a * x - y + 2 * a = 0) → ((2 * a - 1) * x + a * y + a = 0) -> 
  (a ≠ 0 → ∃ k : ℝ, k = (a * ((1 - 2 * a) / a)) ∧ k = -1) -> a * ((1 - 2 * a) / a) = -1) →
  a = 0 ∨ a = 1 := by sorry

end perpendicular_lines_l109_109691


namespace pentagon_area_l109_109567

theorem pentagon_area (a b c d e : ℝ) (r s : ℝ) 
    (h1 : a = 14 ∨ a = 21 ∨ a = 22 ∨ a = 28 ∨ a = 35)
    (h2 : b = 14 ∨ b = 21 ∨ b = 22 ∨ b = 28 ∨ b = 35)
    (h3 : c = 14 ∨ c = 21 ∨ c = 22 ∨ c = 28 ∨ c = 35)
    (h4 : d = 14 ∨ d = 21 ∨ d = 22 ∨ d = 28 ∨ d = 35)
    (h5 : e = 14 ∨ e = 21 ∨ e = 22 ∨ e = 28 ∨ e = 35)
    (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
          b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
          c ≠ d ∧ c ≠ e ∧ 
          d ≠ e)
    (h6 : r^2 + s^2 = e^2)
    (h7 : r = b - d)
    (h8 : s = c - a) 
    : (a + b + c + d + e) * 1 / 2 + (b * c - 1 / 2 * r * s) = 1421 := 
begin
  sorry
end

end pentagon_area_l109_109567


namespace prime_condition_l109_109277

theorem prime_condition (p : ℕ) (h_prime: Nat.Prime p) :
  (∃ m n : ℤ, p = m^2 + n^2 ∧ (m^3 + n^3 - 4) % p = 0) ↔ p = 2 ∨ p = 5 :=
by
  sorry

end prime_condition_l109_109277


namespace trigonometric_identity_l109_109330

theorem trigonometric_identity
  (α : Real)
  (hcos : Real.cos α = -4/5)
  (hquad : π/2 < α ∧ α < π) :
  (-Real.sin (2 * α) / Real.cos α) = -6/5 := 
by
  sorry

end trigonometric_identity_l109_109330


namespace fly_traverse_mobius_strip_l109_109359

theorem fly_traverse_mobius_strip :
  ∃ (strip : Type) (fly_path : strip → Prop), (
    -- strip represents a Möbius strip
    (∀ p : strip, fly_path p) ∧
    -- fly_path traverses all squares on both sides continuously
    (∀ p₁ p₂ : strip, path_continuous p₁ p₂ fly_path) ∧
    -- fly_path returns to starting point
    (∃ start : strip, fly_path start ∧ ∃ end : strip, fly_path end ∧ end = start)
  ) :=
sorry

end fly_traverse_mobius_strip_l109_109359


namespace dot_product_v_and_w_l109_109430

variables {E : Type*} [inner_product_space ℝ E]

variables (u v w : E)
variables (hu : ∥u∥ = 1) (hv : ∥v∥ = 1) (huv : ∥u + v∥ = 2) 
          (hw : w - u - 3 • v = 4 • (u × v))

theorem dot_product_v_and_w : 
  ⟪v, w⟫ = 4 :=
begin
  sorry
end

end dot_product_v_and_w_l109_109430


namespace exists_n_gt_2012_with_f_lt_1_0001_pow_n_l109_109994

-- Definitions for the conditions and problem requirements
def directed_graph (V : Type) := V → V → Prop

variable {V : Type} [fintype V]

-- Out-neighborhood definition
def N_plus (G : directed_graph V) (S : finset V) : finset V :=
  S.bind (λ v, finset.univ.filter (λ u, G v u))

-- Iterative out-neighborhood definition
def N_plus_iter (G : directed_graph V) : ℕ → finset V → finset V
| 0     S := S
| (k+1) S := N_plus G (N_plus_iter k S)

-- Function f definition
def f (G : directed_graph V) : ℕ :=
  finset.univ.powerset.image (λ X, finset.univ.filter (λ k, N_plus_iter G k X)).card

-- Main theorem statement
theorem exists_n_gt_2012_with_f_lt_1_0001_pow_n (V : Type) [fintype V] :
  ∃ n, n > 2012 ∧ f (λ (v1 v2 : V), true) < 1.0001^n :=
by
  sorry

end exists_n_gt_2012_with_f_lt_1_0001_pow_n_l109_109994


namespace rectangular_to_spherical_l109_109255

theorem rectangular_to_spherical (x y z : ℝ) (h1 : 0 < x) (h2 : y < 2 * Math.pi) (h3 : 0 ≤ z) :
  (sqrt 2, -sqrt 2, 2) = (2 * sqrt 2, 7 * Math.pi / 4, Math.pi / 4) :=
by
  sorry

end rectangular_to_spherical_l109_109255


namespace symmetric_point_on_line_l109_109021

theorem symmetric_point_on_line (A B C A1 B1 C1 : Type) [EuclideanSpace ℝ (fin 3)]
  (h_acute : acute_triangle A B C)
  (h_altitudes : altitudes A B C A1 B1 C1)
  (h_A1 : foot_of_altitude A B C A1)
  (h_sym : symmetric_point A1 AC) :
  lies_on_line A1' B1 C1 :=
sorry

end symmetric_point_on_line_l109_109021


namespace full_tank_cost_l109_109547

-- Definitions from the conditions
def total_liters_given := 36
def total_cost_given := 18
def tank_capacity := 64

-- Hypothesis based on the conditions
def price_per_liter := total_cost_given / total_liters_given

-- Conclusion we need to prove
theorem full_tank_cost: price_per_liter * tank_capacity = 32 :=
  sorry

end full_tank_cost_l109_109547


namespace intersection_A_B_l109_109352

def A : Set ℝ := {x | abs x < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l109_109352


namespace new_year_markup_l109_109214

variables (C : ℝ) (x : ℝ) (h1 : (1 + 0.2) * C = 1.2 * C)
          (h2 : (1.2 * C) * (1 + x / 100) = 1.2 * (1 + x / 100) * C)
          (h3 : (1.2 * C) * (1 + x / 100) * 0.8 = (1.2 * C))

theorem new_year_markup : x = 25 :=
by
  have h4 : (1 + x / 100) * 0.8 = 1 := sorry
  have h5 : x / 100 = 0.25 := sorry
  have h6 : x = 25 := by linarith
  exact h6

end new_year_markup_l109_109214


namespace widgets_per_week_l109_109759

theorem widgets_per_week 
  (widgets_per_hour : ℕ) 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (h1 : widgets_per_hour = 20) 
  (h2 : hours_per_day = 8) 
  (h3 : days_per_week = 5) :
  widgets_per_hour * hours_per_day * days_per_week = 800 :=
by
  rw [h1, h2, h3]
  exact rfl

end widgets_per_week_l109_109759


namespace leak_empties_tank_in_10_hours_l109_109536

theorem leak_empties_tank_in_10_hours :
  (∀ (A L : ℝ), (A = 1/5) → (A - L = 1/10) → (1 / L = 10)) 
  := by
  intros A L hA hAL
  sorry

end leak_empties_tank_in_10_hours_l109_109536


namespace sample_size_correct_l109_109901

def population_size : Nat := 8000
def sampled_students : List Nat := List.replicate 400 1 -- We use 1 as a placeholder for the heights

theorem sample_size_correct : sampled_students.length = 400 := by
  sorry

end sample_size_correct_l109_109901


namespace cost_of_blue_cap_l109_109990

theorem cost_of_blue_cap (cost_tshirt cost_backpack cost_cap total_spent discount: ℝ) 
  (h1 : cost_tshirt = 30) 
  (h2 : cost_backpack = 10) 
  (h3 : discount = 2)
  (h4 : total_spent = 43) 
  (h5 : total_spent = cost_tshirt + cost_backpack + cost_cap - discount) : 
  cost_cap = 5 :=
by sorry

end cost_of_blue_cap_l109_109990


namespace cost_of_grapes_and_watermelon_l109_109443

theorem cost_of_grapes_and_watermelon (p g w f : ℝ)
  (h1 : p + g + w + f = 30)
  (h2 : f = 2 * p)
  (h3 : p - g = w) :
  g + w = 7.5 :=
by
  sorry

end cost_of_grapes_and_watermelon_l109_109443


namespace colbert_planks_needed_to_buy_l109_109251

variables (total_planks : ℕ) (planks_from_storage : ℕ) 
          (planks_from_parents : ℕ) (planks_from_friends : ℕ)

def planks_needed_from_store := 
  total_planks - (planks_from_storage + planks_from_parents + planks_from_friends)

theorem colbert_planks_needed_to_buy : 
  total_planks = 200 → planks_from_storage = total_planks / 4 → 
  planks_from_parents = total_planks / 2 → planks_from_friends = 20 → 
  planks_needed_from_store total_planks planks_from_storage planks_from_parents planks_from_friends = 30 :=
by
  -- proof steps here
  sorry

end colbert_planks_needed_to_buy_l109_109251


namespace real_m_of_complex_product_l109_109004

-- Define the conditions that m is a real number and (m^2 + i)(1 - mi) is a real number
def is_real (z : ℂ) : Prop := z.im = 0
def cplx_eq (m : ℝ) : ℂ := (⟨m^2, 1⟩ : ℂ) * (⟨1, -m⟩ : ℂ)

theorem real_m_of_complex_product (m : ℝ) : is_real (cplx_eq m) ↔ m = 1 :=
by
  sorry

end real_m_of_complex_product_l109_109004


namespace excursion_sharing_l109_109839

theorem excursion_sharing :
  ∀ (x : ℝ), 
    3 * (13 / 33) * 9 = 6 ∧
    3 * (12 / 33) * 9 = 3  :=
by
  intro x
  split
  { sorry }
  { sorry }

end excursion_sharing_l109_109839


namespace middle_part_proportional_l109_109197

theorem middle_part_proportional (x : ℚ) (s : ℚ) (h : s = 120) 
    (proportional : (2 * x) + (1/2 * x) + (1/4 * x) = s) : 
    (1/2 * x) = 240/11 := 
by
  sorry

end middle_part_proportional_l109_109197


namespace cannot_be_20_l109_109400

def distinct_integers_from_1_to_9 (F G H J : ℕ) : Prop :=
  F ≠ G ∧ F ≠ H ∧ F ≠ J ∧ G ≠ H ∧ G ≠ J ∧ H ≠ J ∧ 1 ≤ F ∧ F ≤ 9 ∧ 1 ≤ G ∧ G ≤ 9 ∧ 1 ≤ H ∧ H ≤ 9 ∧ 1 ≤ J ∧ J ≤ 9

theorem cannot_be_20 (F G H J : ℕ) (h : distinct_integers_from_1_to_9 F G H J) :
  let K := F * G,
      L := H + J,
      M := K / L in
  M ≠ 20 :=
sorry

end cannot_be_20_l109_109400


namespace rice_in_each_container_is_50_ounces_l109_109086

theorem rice_in_each_container_is_50_ounces :
  let total_weight_pounds := 25 / 2 
  let number_of_containers := 4
  let pounds_to_ounces := 16
  let weight_per_container_pounds := total_weight_pounds / number_of_containers in
  weight_per_container_pounds * pounds_to_ounces = 50 := 
by
  sorry

end rice_in_each_container_is_50_ounces_l109_109086


namespace bug_travel_paths_example_l109_109204

-- Definitions representing conditions of the problem
def hexagonal_lattice_paths (A B : Point) : Nat :=
  -- Formalize the logic that calculates the number of distinct paths from A to B
  sorry

-- Statement that we need to prove
theorem bug_travel_paths_example (A B : Point) (lattice : Lattice) (conditions : lattice_conditions lattice) : 
  hexagonal_lattice_paths A B = 2400 := 
sorry

end bug_travel_paths_example_l109_109204


namespace min_value_fraction_sum_l109_109706

/-- Given the conditions that a > 0, b > 0, and the line ax - by + 8 = 0 passes through the 
    center of the circle x^2 + y^2 + 4x - 4y = 0, prove that the minimum value of 1/a + 1/b is 1. 
    Equality holds if and only if a = b = 2. -/
theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (hline : ∃ x y : ℝ, a * x - b * y + 8 = 0 ∧ (x, y) = (-2, 2))
  (hcenter : ∃ g f : ℝ, (g, f) = (-2, 2) ∧ ∀ x y : ℝ, x^2 + y^2 + 2*g*x + 2*f*y = 0 → x = 0 ∧ y = 0) :
∃ a b : ℝ, a + b = 4 ∧ 1/a + 1/b = 1 :=
begin
  sorry -- Proof placeholder
end

end min_value_fraction_sum_l109_109706


namespace sqrt_sum_inequality_l109_109650

theorem sqrt_sum_inequality (n : ℕ) :
  (2 / 3 : ℝ) * n * √n < ∑ i in (Finset.range n).map (λ i, i + 1), √(i + 1) ∧ 
  ∑ i in (Finset.range n).map (λ i, i + 1), √(i + 1) < ((4 * n + 3) / 6 : ℝ) * √n := 
sorry

end sqrt_sum_inequality_l109_109650


namespace permutation_remainder_l109_109426

theorem permutation_remainder :
  let N := (∑ k in Finset.range 5,
              Nat.choose 5 (k + 1) * Nat.choose 6 k * Nat.choose 7 (k + 2)) in
  N % 1000 = 996 := by
  sorry

end permutation_remainder_l109_109426


namespace length_of_other_leg_l109_109873

theorem length_of_other_leg (c a b : ℕ) (h1 : c = 10) (h2 : a = 6) (h3 : c^2 = a^2 + b^2) : b = 8 :=
by
  sorry

end length_of_other_leg_l109_109873


namespace rectangular_prism_volume_increase_l109_109121

theorem rectangular_prism_volume_increase (L B H : ℝ) :
  let V_original := L * B * H
  let L_new := L * 1.07
  let B_new := B * 1.18
  let H_new := H * 1.25
  let V_new := L_new * B_new * H_new
  let increase_in_volume := (V_new - V_original) / V_original * 100
  increase_in_volume = 56.415 :=
by
  sorry

end rectangular_prism_volume_increase_l109_109121


namespace largest_of_three_consecutive_odds_l109_109891

theorem largest_of_three_consecutive_odds (n : ℤ) (h_sum : n + (n + 2) + (n + 4) = -147) : n + 4 = -47 :=
by {
  -- Proof steps here, but we're skipping for this exercise
  sorry
}

end largest_of_three_consecutive_odds_l109_109891


namespace area_of_triangle_OAOB_is_25_div_9_l109_109633

noncomputable def rightFocus (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)
noncomputable def ellipseLineIntersectionPoints (a b slope : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let c := rightFocus a b
  let y := λ x => slope * (x - c)
  let pts := [((x : ℝ) * x / a^2 + (y x) * y x / b^2 = 1)
    for x in [0, a / slope]].filter (λ pt => true)
  match pts with
  | ((x1, y1), (x2, y2)) => ((x1, y1), (x2, y2))

theorem area_of_triangle_OAOB_is_25_div_9 : 
  let a := 5
  let b := 4
  let slope := 2
  let origin := (0, 0)
  let ((x1, y1), (x2, y2)) := ellipseLineIntersectionPoints a b slope
  tri_area origin (x1, y1) (x2, y2) = 25 / 9 :=
by
  sorry

end area_of_triangle_OAOB_is_25_div_9_l109_109633


namespace translation_correctness_l109_109501

-- Define the original function
def original_function (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

-- Define the function obtained by translating to the left by π/6 units
def translated_function (x : ℝ) : ℝ := original_function (x + Real.pi / 6)

-- Define the expected result function
def expected_function (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- The theorem stating the equality of the translation and the expected result
theorem translation_correctness : ∀ x : ℝ, translated_function x = expected_function x := by
    sorry

end translation_correctness_l109_109501


namespace at_least_one_not_greater_than_neg_two_l109_109055

open Real

theorem at_least_one_not_greater_than_neg_two
  {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + (1 / b) ≤ -2 ∨ b + (1 / c) ≤ -2 ∨ c + (1 / a) ≤ -2 :=
sorry

end at_least_one_not_greater_than_neg_two_l109_109055


namespace part_a_part_b_l109_109016

variable {A B C D X : Point}
variable {AD BC AB CD : ℝ}

-- Given conditions
def diagonals_perpendicular (A B C D X : Point) (AC BD : Line) : Prop :=
  AC ⊥ BD

def parallel_lines (AB CD : Line) : Prop :=
  AB ∥ CD

-- Proof Statements
theorem part_a (h1 : diagonals_perpendicular A B C D X) 
               (h2 : parallel_lines AB CD) :
  AD * BC ≥ AB * CD := 
sorry

theorem part_b (h1 : diagonals_perpendicular A B C D X) 
               (h2 : parallel_lines AB CD) :
  AD + BC ≥ AB + CD :=
sorry

end part_a_part_b_l109_109016


namespace number_of_yellow_marbles_l109_109403

/-- 
 In a jar with blue, red, and yellow marbles:
  - there are 7 blue marbles
  - there are 11 red marbles
  - the probability of picking a yellow marble is 1/4
 Show that the number of yellow marbles is 6.
-/
theorem number_of_yellow_marbles 
  (blue red y : ℕ) 
  (h_blue : blue = 7) 
  (h_red : red = 11) 
  (h_prob : y / (18 + y) = 1 / 4) : 
  y = 6 := 
sorry

end number_of_yellow_marbles_l109_109403


namespace cindy_marbles_problem_l109_109831

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end cindy_marbles_problem_l109_109831


namespace rebus_to_math_teasers_ratio_l109_109575

theorem rebus_to_math_teasers_ratio:
  (total brainiacs: ℕ) 
  (both : ℕ) 
  (neither : ℕ) 
  (only_math : ℕ) 
  (R M : ℕ) 
  (gcd_58_38 : ℕ) 
  (zero_lt_rebus: 0 < R) 
  (zero_lt_math: 0 < M)
  (total = 100)
  (both = 18)
  (neither = 4)
  (only_math = 20)
  (96 = total - neither)
  (M = both + only_math)
  (R = total - neither - only_math)
  (gcd_58_38 = Int.gcd 58 38)
  : R / gcd_58_38 = 29 ∧ M / gcd_58_38 = 19 := by 
begin
  sorry
end

end rebus_to_math_teasers_ratio_l109_109575


namespace penthouse_units_l109_109212

theorem penthouse_units (total_floors : ℕ) (regular_units_per_floor : ℕ) (penthouse_floors : ℕ) (total_units : ℕ) :
  total_floors = 23 →
  regular_units_per_floor = 12 →
  penthouse_floors = 2 →
  total_units = 256 →
  (total_units - (total_floors - penthouse_floors) * regular_units_per_floor) / penthouse_floors = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end penthouse_units_l109_109212


namespace grams_of_fat_per_serving_is_correct_l109_109081

noncomputable def grams_of_fat_per_serving : ℝ :=
  let ratio_cream_cheese_butter := (5:ℝ, 3:ℝ, 2:ℝ)
  let total_parts := ratio_cream_cheese_butter.1 + ratio_cream_cheese_butter.2 + ratio_cream_cheese_butter.3
  let total_mix := 1.5
  let part_size := total_mix / total_parts
  let cream_amount := ratio_cream_cheese_butter.1 * part_size
  let cheese_amount := ratio_cream_cheese_butter.2 * part_size
  let butter_part := ratio_cream_cheese_butter.3 * part_size
  let butter_amount := butter_part * 2
  let cream_fat := cream_amount * 88
  let cheese_fat := cheese_amount * 110
  let butter_fat := butter_amount * 184
  let total_fat := cream_fat + cheese_fat + butter_fat
  let servings := 6
  total_fat / servings

theorem grams_of_fat_per_serving_is_correct :
  grams_of_fat_per_serving = 37.65 := by
  sorry

end grams_of_fat_per_serving_is_correct_l109_109081


namespace line_quadrant_conditions_l109_109690

theorem line_quadrant_conditions (k b : ℝ) 
  (H1 : ∃ x : ℝ, x > 0 ∧ k * x + b > 0)
  (H3 : ∃ x : ℝ, x < 0 ∧ k * x + b < 0)
  (H4 : ∃ x : ℝ, x > 0 ∧ k * x + b < 0) : k > 0 ∧ b < 0 :=
sorry

end line_quadrant_conditions_l109_109690


namespace total_boys_in_camp_l109_109389

variable (T : ℕ) -- Total number of boys in the camp
variable (school_a_boys school_b_boys school_c_boys : ℕ) -- Boys from each school
variable (a_students sci_a not_sci_a : ℕ) -- Students' distribution in School A
variable (b_students math_b : ℕ) -- Students' distribution in School B

-- Conditions
def condition1 : Prop := school_a_boys = 20 * T / 100
def condition2 : Prop := school_b_boys = 30 * T / 100
def condition3 : Prop := school_c_boys = T - school_a_boys - school_b_boys
def condition4a : Prop := a_students = school_a_boys 
def condition4b : Prop := sci_a = 30 * school_a_boys / 100
def condition4c : Prop := not_sci_a = 70 * school_a_boys / 100
def condition5a : Prop := b_students = school_b_boys 
def condition5b : Prop := math_b = 35 * school_b_boys / 100
def condition7 : Prop := 35 = not_sci_a
def condition8 : Prop := 20 = math_b

-- Theorem to be proved
theorem total_boys_in_camp (h1 : condition1) (h2 : condition2) (h3 : condition3) 
    (h4a : condition4a) (h4b : condition4b) (h4c : condition4c)  
    (h5a : condition5a) (h5b : condition5b) (h7 : condition7)
    (h8 : condition8) : T = 250 := sorry

end total_boys_in_camp_l109_109389


namespace solution_set_l109_109815

def f (x : ℝ) : ℝ :=
  if x < 2 then 2 * real.exp (x - 1) else real.log (x^2 - 1) / real.log 3

theorem solution_set :
  { x : ℝ | f x > 2 } = { x : ℝ | 1 < x ∧ x < 2 } ∪ { x : ℝ | x > real.sqrt 10 } :=
by
  sorry

end solution_set_l109_109815


namespace angle_PMF_eq_angle_FPN_l109_109496

-- Define the parabola as y^2 = 2px
def parabola (p : ℝ) : Set (ℝ × ℝ) :=
  { pt | pt.2 ^ 2 = 2 * p * pt.1 }

-- Define P as a point outside the parabola
def point_outside_parabola (P : ℝ × ℝ) (p : ℝ) : Prop :=
  ¬ (parabola p).has P

-- Define the focus of the parabola y^2 = 2px
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- Statement of the proof problem
theorem angle_PMF_eq_angle_FPN
  (p : ℝ)
  (P : ℝ × ℝ)
  (hP : point_outside_parabola P p)
  (M N : ℝ × ℝ)
  (hM : parabola p M)
  (hN : parabola p N)
  (tangent_PM : is_tangent P M p)
  (tangent_PN : is_tangent P N p) :
  let F := focus p in
  ∠PMF = ∠FPN := sorry

end angle_PMF_eq_angle_FPN_l109_109496


namespace minimize_C_ratio_l109_109668

noncomputable def C_x_m (x : ℝ) (m : ℕ) : ℝ :=
  if h : m = 0 then 1 else x * (x - 1) * (x - 2) * ... * (x - m + 1) / m.factorial

theorem minimize_C_ratio (x : ℝ) (hx : 0 < x) :
  let Cx3 := C_x_m x 3
      Cx1_square := (C_x_m x 1) ^ 2 in
  min ((Cx3) / Cx1_square) = (Real.sqrt 2 / 3 - 1 / 2) :=
sorry

end minimize_C_ratio_l109_109668


namespace find_dimensions_l109_109657

def is_solution (m n r : ℕ) : Prop :=
  ∃ k0 k1 k2 : ℕ, 
    k0 = (m - 2) * (n - 2) * (r - 2) ∧
    k1 = 2 * ((m - 2) * (n - 2) + (n - 2) * (r - 2) + (r - 2) * (m - 2)) ∧
    k2 = 4 * ((m - 2) + (n - 2) + (r - 2)) ∧
    k0 + k2 - k1 = 1985

theorem find_dimensions (m n r : ℕ) (h : m ≤ n ∧ n ≤ r) (hp : 0 < m ∧ 0 < n ∧ 0 < r) : 
  is_solution m n r :=
sorry

end find_dimensions_l109_109657


namespace floor_of_7_9_l109_109266

theorem floor_of_7_9 : (Int.floor 7.9) = 7 :=
by
  sorry

end floor_of_7_9_l109_109266


namespace log_xy_value_l109_109368

theorem log_xy_value (x y : ℝ) (h1 : log (x * y^4) = 2) (h2 : log (x^3 * y) = 2) : log (x * y) = 10 / 11 :=
by
  -- The proof will be inserted here
  sorry

end log_xy_value_l109_109368


namespace not_all_triangles_obtuse_l109_109583

/-- 
An acute-angled triangle is repeatedly subdivided using straight line cuts. 
Prove that it is impossible for all resulting triangles to be obtuse-angled.
-/

theorem not_all_triangles_obtuse (T : Type) [acute_angled_triangle T] (subdivide : T → T × T) :
  ∃ t : T, ¬ obtuse_angled t :=
sorry

end not_all_triangles_obtuse_l109_109583


namespace fifth_perpendicular_passes_through_O_l109_109646

-- Define the setup for the convex pentagon and conditions
structure ConvexPentagon (A B C D E O : Type) :=
(perp_O_A_to_CD : A → A → Prop)
(perp_O_B_to_DE : B → B → Prop)
(perp_O_C_to_EA : C → C → Prop)
(perp_O_D_to_AB : D → D → Prop)

-- The proof problem statement in Lean
theorem fifth_perpendicular_passes_through_O 
  (A B C D E O : Type)
  [ConvexPentagon A B C D E O] : 
  ∀ (v_A : A) (v_B : B) (v_C : C) (v_D : D) (v_E : E),
  (perp_O_A_to_CD A v_A) = true → 
  (perp_O_B_to_DE B v_B) = true → 
  (perp_O_C_to_EA C v_C) = true → 
  (perp_O_D_to_AB D v_D) = true → 
  (perp_O_E_to_AB E v_E) = true :=
  sorry

end fifth_perpendicular_passes_through_O_l109_109646


namespace minimum_common_perimeter_l109_109149

noncomputable def is_integer (x: ℝ) : Prop := ∃ (n: ℤ), x = n

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ is_triangle a b c

theorem minimum_common_perimeter :
  ∃ (a b : ℝ),
    is_integer a ∧ is_integer b ∧
    4 * a = 5 * b - 18 ∧
    is_isosceles_triangle a a (2 * a - 12) ∧
    is_isosceles_triangle b b (3 * b - 30) ∧
    (2 * a + (2 * a - 12) = 2 * b + (3 * b - 30)) ∧
    (2 * a + (2 * a - 12) = 228) := sorry

end minimum_common_perimeter_l109_109149


namespace regions_with_n_plus_one_lines_l109_109737

-- Conditions
variables (n : ℕ) (f : ℕ → ℕ)

-- Hypotheses based on given conditions
hypothesis no_parallel_lines : ∀ (i j : ℕ), i < j → ¬ parallel (lines i) (lines j)
hypothesis no_triple_intersection : ∀ (i j k : ℕ), i < j ∧ j < k → ¬ collinear (lines i) (lines j) (lines k)
hypothesis f_n_regions : f(n) = -- Number of regions with n lines 

-- Prove the statement
theorem regions_with_n_plus_one_lines : f(n + 1) = f(n) + n + 1 :=
by sorry

end regions_with_n_plus_one_lines_l109_109737


namespace exchanges_count_l109_109182

theorem exchanges_count (n : ℕ) :
  ∀ (initial_pencils_XZ initial_pens_XL : ℕ) 
    (pencils_per_exchange pens_per_exchange : ℕ)
    (final_pencils_multiplier : ℕ)
    (pz : initial_pencils_XZ = 200) 
    (pl : initial_pens_XL = 20)
    (pe : pencils_per_exchange = 6)
    (se : pens_per_exchange = 1)
    (fm : final_pencils_multiplier = 11),
    (initial_pencils_XZ - pencils_per_exchange * n = final_pencils_multiplier * (initial_pens_XL - pens_per_exchange * n)) ↔ n = 4 :=
by
  intros initial_pencils_XZ initial_pens_XL pencils_per_exchange pens_per_exchange final_pencils_multiplier pz pl pe se fm
  sorry

end exchanges_count_l109_109182


namespace find_m_l109_109027

theorem find_m (a : ℕ → ℝ) (m : ℝ)
  (h1 : (∀ (x : ℝ), x^2 + m * x - 8 = 0 → x = a 2 ∨ x = a 8))
  (h2 : a 4 + a 6 = a 5 ^ 2 + 1) :
  m = -2 :=
sorry

end find_m_l109_109027


namespace problem_solution_l109_109127

theorem problem_solution : 
  let a := (3 / 8 : ℝ)
  let b := (real.sqrt 183 / 8 : ℝ)
  a + b^2 = 207 / 64 :=
by
  sorry

end problem_solution_l109_109127


namespace difference_of_squares_simplifies_to_neg_two_l109_109986

theorem difference_of_squares_simplifies_to_neg_two :
  let a := (1 : ℝ)
  let b := real.sqrt 3
  (a + b) * (a - b) = -2 := 
by
  sorry

end difference_of_squares_simplifies_to_neg_two_l109_109986


namespace sahil_sold_machine_at_correct_price_l109_109090

def total_expense_purchase (purchase_price import_tax_percent : ℝ) :=
  purchase_price + (import_tax_percent * purchase_price)

def total_expense_repair (repair_cost_EUR : ℝ) (exchange_rate_EUR_USD : ℝ) (tax_percent_repair : ℝ) :=
  let repair_cost_USD := repair_cost_EUR / exchange_rate_EUR_USD
  repair_cost_USD + (tax_percent_repair * repair_cost_USD)

def total_expense_transport (transport_cost_GBP : ℝ) (exchange_rate_GBP_USD : ℝ) (tax_percent_transport : ℝ) :=
  let transport_cost_USD := transport_cost_GBP / exchange_rate_GBP_USD
  transport_cost_USD + (tax_percent_transport * transport_cost_USD)

def total_expense (purchase_cost_USD repair_cost_USD transport_cost_USD : ℝ) :=
  purchase_cost_USD + repair_cost_USD + transport_cost_USD

def selling_price (total_expense USD profit_percent : ℝ) :=
  total_expense + (profit_percent * total_expense)

def sahil_selling_price (purchase_price_USD repair_cost_EUR transport_cost_GBP import_tax_percent tax_percent_repair tax_percent_transport : ℝ)
  (exchange_rate_EUR_USD_at_purchase exchange_rate_GBP_USD_at_purchase profit_percent : ℝ) : ℝ :=
  let purchase_expense := total_expense_purchase purchase_price_USD import_tax_percent
  let repair_expense := total_expense_repair repair_cost_EUR exchange_rate_EUR_USD_at_purchase tax_percent_repair
  let transport_expense := total_expense_transport transport_cost_GBP exchange_rate_GBP_USD_at_purchase tax_percent_transport
  let total_expense_USD := total_expense purchase_expense repair_expense transport_expense
  selling_price total_expense_USD profit_percent

theorem sahil_sold_machine_at_correct_price :
  sahil_selling_price 12000 4000 1000 0.05 0.10 0.03 0.85 0.75 0.50 = 28724.70 :=
by sorry

end sahil_sold_machine_at_correct_price_l109_109090


namespace count_valid_n_l109_109292

theorem count_valid_n :
  {n : ℕ | (∀ t : ℝ, (Complex.sin t - Complex.I * Complex.cos t) ^ n = Complex.sin (n * t) - Complex.I * Complex.cos (n * t)) ∧ n ≤ 500}.to_finset.card = 125 := by
sorry

end count_valid_n_l109_109292


namespace sugar_needed_for_frosting_l109_109070

-- Definitions
def sugar_at_home : ℕ := 3
def sugar_per_bag : ℕ := 6
def bags_purchased : ℕ := 2
def sugar_per_twelve_cupcakes : ℕ := 1
def dozen_cupcakes : ℕ := 12
def dozen_baked : ℕ := 5

-- Query
theorem sugar_needed_for_frosting :
  let total_sugar := sugar_at_home + bags_purchased * sugar_per_bag in
  let sugar_for_batter := dozen_baked * sugar_per_twelve_cupcakes in
  let sugar_for_frosting := total_sugar - sugar_for_batter in
  sugar_for_frosting / dozen_baked = 2 :=
sorry

end sugar_needed_for_frosting_l109_109070


namespace complex_sum_evaluation_l109_109271

theorem complex_sum_evaluation : 
  (∑ k in finset.range 1999, (k + 1) * complex.I ^ (k + 1)) = -1002 * complex.I - 998 := 
sorry

end complex_sum_evaluation_l109_109271


namespace average_problem_l109_109861

theorem average_problem
  (h : (20 + 40 + 60) / 3 = (x + 50 + 45) / 3 + 5) :
  x = 10 :=
by
  sorry

end average_problem_l109_109861


namespace island_of_truth_l109_109080

variable (A B C D : Prop)

-- Definitions based on the conditions
def is_knight (X : Prop) : Prop := X
def is_liar (X : Prop) : Prop := ¬X

-- Conditions based on the inhabitants' statements
def statement_A := ∀ x, is_liar A ↔ (x = A ∨ x = B ∨ x = C ∨ x = D)
def statement_B := is_liar B ↔ (forall x, is_liar x)

-- Tourist asks C and gets a definitive answer about A
def tourist_ask_C (c_response : Prop) := c_response ↔ is_liar A

theorem island_of_truth :
  statement_A ∧ statement_B ∧ tourist_ask_C C → 
  is_liar A ∧ is_liar B ∧ (is_liar D ∨ is_knight D) := 
sorry

end island_of_truth_l109_109080


namespace distance_AB_eq_31_l109_109310

noncomputable def i : ℂ := Complex.I

def z1 : ℂ := (-1 + 3 * i) / (1 + 2 * i)
def z2 : ℂ := 1 + Complex.exp (10 * Complex.log (1 + i))

def A : ℝ × ℝ := (z1.re, z1.im)
def B : ℝ × ℝ := (z2.re, z2.im)

def dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem distance_AB_eq_31 : dist A B = 31 :=
  sorry

end distance_AB_eq_31_l109_109310


namespace johann_mail_l109_109417

def pieces_of_mail_total : ℕ := 180
def pieces_of_mail_friends : ℕ := 41
def friends : ℕ := 2
def pieces_of_mail_johann : ℕ := pieces_of_mail_total - (pieces_of_mail_friends * friends)

theorem johann_mail : pieces_of_mail_johann = 98 := by
  sorry

end johann_mail_l109_109417


namespace sum_of_possible_intersections_of_five_lines_l109_109641

/-- Definition of combinatorial calculation for intersection points of lines -/
def num_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

theorem sum_of_possible_intersections_of_five_lines : 
  let possible_points := finset.sum (finset.range (num_intersections 5 + 1)) id in 
  possible_points = 55 :=
by
  let num_intersections_5 := num_intersections 5
  have h1 : num_intersections_5 = 10 := by
    rw [num_intersections, nat.sub_one_eq_pred, nat.pred_succ]
    simp only [nat.mul_div_cancel_left, nat.sub_self, zero_mul, nat.succ_sub_succ_eq_sub, eq_self_iff_true, zero_add]
  have sum_n : ∑ i in finset.range (num_intersections_5 + 1), i = 55 := by
    calc ∑ i in finset.range (num_intersections_5 + 1), i = 
      ∑ i in finset.range 11, i : by rw h1
      ... = 55 : by norm_num
  rw ← sum_n
  exact rfl

end sum_of_possible_intersections_of_five_lines_l109_109641


namespace sum_of_solutions_eq_46_l109_109774

theorem sum_of_solutions_eq_46 (x y : ℤ) (sols : List (ℤ × ℤ)) :
  (∀ (xi yi : ℤ), (xi, yi) ∈ sols →
    (|xi - 4| = |yi - 10| ∧ |xi - 10| = 3 * |yi - 4|)) →
  (sols = [(10, 4), (5, -1), (10, 4), (-5, 19)]) →
  List.sum (sols.map (λ p, p.1 + p.2)) = 46 :=
by
  intro h1 h2
  rw [h2]
  dsimp
  norm_num

end sum_of_solutions_eq_46_l109_109774


namespace sonya_fell_times_l109_109466

theorem sonya_fell_times (steven_falls : ℕ) (stephanie_falls : ℕ) (sonya_falls : ℕ) :
  steven_falls = 3 →
  stephanie_falls = steven_falls + 13 →
  sonya_falls = 6 →
  sonya_falls = (stephanie_falls / 2) - 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at *
  sorry

end sonya_fell_times_l109_109466


namespace power_function_through_point_l109_109006

theorem power_function_through_point (m : ℝ) : (2 ^ m = real.sqrt 2) → m = 1 / 2 :=
by
  intro h,
  sorry

end power_function_through_point_l109_109006


namespace prime_product_l109_109984

def largest_one_digit_prime : ℕ := 7
def second_largest_one_digit_prime : ℕ := 5
def second_largest_two_digit_prime : ℕ := 89

theorem prime_product :
  largest_one_digit_prime * second_largest_one_digit_prime * second_largest_two_digit_prime = 3115 := by
  -- Use the known values
  have h₁ : largest_one_digit_prime = 7 := rfl
  have h₂ : second_largest_one_digit_prime = 5 := rfl
  have h₃ : second_largest_two_digit_prime = 89 := rfl
  -- Calculate the product
  calc
    7 * 5 * 89 = 35 * 89   : by rw [h₁, h₂]
           ... = 3115     : sorry

end prime_product_l109_109984


namespace average_cost_per_individual_l109_109954

-- Definitions based on conditions
def total_individuals : ℕ := 6
def total_bill : ℝ := 720
def gratuity_rate : ℝ := 0.20

-- Theorem stating the average cost per individual
theorem average_cost_per_individual 
  (total_individuals = 6) 
  (total_bill = 720) 
  (gratuity_rate = 0.20) : 
  let meal_cost := total_bill / (1 + gratuity_rate) in
  let average_cost := meal_cost / total_individuals in
  average_cost = 100 := 
sorry

end average_cost_per_individual_l109_109954


namespace frog_jumps_l109_109543

theorem frog_jumps (k : ℕ) (i : ℕ) (hk : k ≥ 2) :
  (min_jumps (2^i * k) > min_jumps (2^i)) :=
begin
  sorry
end

-- Supporting axioms and definitions, these would be defined as per the frog's jumping rules.
axiom min_jumps : ℕ -> ℕ

end frog_jumps_l109_109543


namespace first_step_of_testing_circuit_broken_l109_109917

-- Definitions based on the problem
def circuit_broken : Prop := true
def binary_search_method : Prop := true
def test_first_step_at_midpoint : Prop := true

-- The theorem stating the first step in testing a broken circuit using the binary search method
theorem first_step_of_testing_circuit_broken (h1 : circuit_broken) (h2 : binary_search_method) :
  test_first_step_at_midpoint :=
sorry

end first_step_of_testing_circuit_broken_l109_109917


namespace find_m_n_l109_109632

theorem find_m_n (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) 
  (h1 : m * n ∣ 3 ^ m + 1) (h2 : m * n ∣ 3 ^ n + 1) : 
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) :=
by
  sorry

end find_m_n_l109_109632


namespace chord_length_l109_109120

theorem chord_length
  (h1: ∀ x y : ℝ, x^2 + y^2 = 1)
  (h2: ∀ x y : ℝ, sqrt 3 * x + y - 1 = 0) :
  ∃ l : ℝ, l = sqrt 3 :=
by
  sorry

end chord_length_l109_109120


namespace find_p_l109_109750

theorem find_p (m n p : ℝ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by sorry

end find_p_l109_109750


namespace ratio_eq_one_l109_109373

theorem ratio_eq_one (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
sorry

end ratio_eq_one_l109_109373


namespace sally_weekend_reading_l109_109848

theorem sally_weekend_reading (pages_on_weekdays : ℕ) (total_pages : ℕ) (weeks : ℕ) (weekdays_per_week : ℕ) (total_days : ℕ) 
  (finishing_time : ℕ) (weekend_days : ℕ) (pages_weekdays_total : ℕ) :
  pages_on_weekdays = 10 →
  total_pages = 180 →
  weeks = 2 →
  weekdays_per_week = 5 →
  weekend_days = (total_days - weekdays_per_week * weeks) →
  total_days = 7 * weeks →
  finishing_time = weeks →
  pages_weekdays_total = pages_on_weekdays * weekdays_per_week * weeks →
  (total_pages - pages_weekdays_total) / weekend_days = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end sally_weekend_reading_l109_109848


namespace triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l109_109844

theorem triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero
  (A B C : ℝ) (h : A + B + C = 180): 
    (A = 60 ∨ B = 60 ∨ C = 60) ↔ (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 0) := 
by
  sorry

end triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l109_109844


namespace center_of_circle_l109_109507

-- Define the parametric equations x and y
def x (θ : Real) : Real := 2 * Real.cos θ
def y (θ : Real) : Real := 2 * Real.sin θ + 2

-- Define the circle equation in terms of x and y
def circle_equation (x y : Real) : Prop :=
  (y - 2) ^ 2 + x ^ 2 = 4

-- Prove that the center of the circle is (0, 2)
theorem center_of_circle : 
  ∃ h k, (h = 0) ∧ (k = 2) ∧ ∀ (θ : Real), circle_equation (x θ) (y θ) :=
by
  use 0
  use 2
  split
  · refl
  split
  · refl
  intro θ
  unfold x y
  sorry

end center_of_circle_l109_109507


namespace merchant_gross_profit_l109_109187

noncomputable def grossProfit (purchase_price : ℝ) (selling_price : ℝ) (discount : ℝ) : ℝ :=
  (selling_price - discount * selling_price) - purchase_price

theorem merchant_gross_profit :
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  grossProfit P S discount = 8 := 
by
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  unfold grossProfit
  sorry

end merchant_gross_profit_l109_109187


namespace first_term_of_geometric_series_l109_109973

theorem first_term_of_geometric_series (r a S : ℝ) (h_r : r = 1 / 4) (h_S : S = 40) 
  (h_geometric_sum : S = a / (1 - r)) : a = 30 :=
by
  -- The proof would go here, but we place a sorry to skip the proof.
  sorry

end first_term_of_geometric_series_l109_109973


namespace sum_of_solutions_l109_109781

theorem sum_of_solutions :
  let solutions := [(-8, -2), (-1, 5), (10, 4), (10, 4)],
  (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 :=
by
  let solutions : List (ℤ × ℤ) := [(-8, -2), (-1, 5), (10, 4), (10, 4)]
  have h1 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 4| = |y - 10| := sorry
  have h2 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 10| = 3 * |y - 4| := sorry
  have solution_sum : (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 := by
    simp [solutions]
    norm_num
  exact solution_sum

end sum_of_solutions_l109_109781


namespace find_y_coordinate_of_P_l109_109053

-- Definitions of points
def A := (-3: ℝ, 0: ℝ)
def B := (-2: ℝ, 1: ℝ)
def C := (2: ℝ, 1: ℝ)
def D := (3: ℝ, 0: ℝ)

-- Definition of distance function
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Conditions for point P
def condition1 (P : ℝ × ℝ) : Prop := dist P A + dist P D = 8
def condition2 (P : ℝ × ℝ) : Prop := dist P B + dist P C = 8

-- Proof statement
theorem find_y_coordinate_of_P (P : ℝ × ℝ) (h1 : condition1 P) (h2 : condition2 P) :
  ∃ (a b c d : ℕ), a = 7 ∧ b = 2 ∧ c = 21 ∧ d = 5 ∧ P.2 = (-(a:ℝ) + (b:ℝ) * Real.sqrt (c : ℝ)) / (d : ℝ) ∧ a + b + c + d = 35 :=
sorry

end find_y_coordinate_of_P_l109_109053


namespace number_of_different_partitions_l109_109375

variable {α : Type*} [DecidableEq α]

def different_partitions (A : Finset α) : Finset (Finset α × Finset α) :=
  (A.powerset.product A.powerset).filter (λ p, p.1 ∪ p.2 = A)

theorem number_of_different_partitions (A : Finset α) [Fintype α] (hA : A = {1, 2, 3}) :
  (different_partitions A).card = 27 := sorry

end number_of_different_partitions_l109_109375


namespace valid_card_pair_probability_l109_109503

def is_number_or_face_card (card : ℕ) : Prop :=
(card ≥ 2 ∧ card ≤ 10) ∨ (card = 11 ∨ card = 12 ∨ card = 13)

def valid_card_pair (card1 card2 : ℕ) : Prop :=
is_number_or_face_card card1 ∧ is_number_or_face_card card2 ∧ (card1 + card2 = 14)

noncomputable def probability_valid_card_pair : ℚ :=
74 / 331

theorem valid_card_pair_probability :
  (∃ (card1 card2 : ℕ), valid_card_pair card1 card2 ∧
  let p := 1 / 52 * (if card1 = card2 then (1 / 51) else (3 / 51)) in
  p = probability_valid_card_pair) := 
sorry

end valid_card_pair_probability_l109_109503


namespace problem_proof_l109_109258

def op (a b : ℝ) : ℝ :=
if a * b >= 0 then a * b else a / b

def f (x : ℝ) : ℝ := op (Real.log x) x

theorem problem_proof : f(2) + f(1 / 2) = 0 := by
  have h1 : f 2 = 2 * Real.log 2 := by
    unfold f op
    rw if_pos
    { rw Real.log
    { exact mul_nonneg_of_pos_of_pos (Real.log_pos two_pos) zero_lt_one } }
  have h2 : f (1 / 2) = - 2 * Real.log 2 := by
    unfold f op
    rw if_neg
    { rw [Real.log, div_div_eq_div_mul, one_div_eq_inv, mul_comm, neg_mul_eq_mul_neg]
    { exact mul_neg_of_neg_of_pos (Real.log_neg of_pos_div_zero one_pos pos.ne.symm) (Real.inv_pos.2 zero_lt_one) } }
  rw [h1, h2]
  linarith

end problem_proof_l109_109258


namespace ω_value_g_min_value_l109_109339

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 
  sin (π - ω * x) * cos (ω * x) + cos (ω * x) ^ 2

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sqrt 2 / 2) * sin (4 * x + π / 4) + 1 / 2

theorem ω_value (ω : ℝ) (h : ω > 0)
  (h1 : ∀ x, f x ω = sin (ω * x) * cos (ω * x) + (1 + cos (2 * ω * x)) / 2) 
  (h2 : ∀ x, f x ω = (Real.sqrt 2 / 2) * sin (2 * ω * x + π / 4) + 1 / 2)
  (h3 : ∀ x, ∀ y, (ω > 0) → (x=y) → (sin (2 * ω * x + (π / 4)) = sin (2 * ω * y + (π / 4))))
  (h4 : ω = 1): 
  (2 * π / (2 * ω) = π) := 
sorry

theorem g_min_value :
  ∀ x ∈ Icc (0 : ℝ) (π / 16), ∃ x, g x = 1 :=
sorry

end ω_value_g_min_value_l109_109339


namespace degree_of_polynomial_sum_l109_109059

-- Definitions based on conditions
def f : ℝ[X] := polynomial.X ^ 3 -- f(x) is a polynomial of degree 3
def g : ℝ[X] := polynomial.X ^ 2 -- g(x) is a polynomial of degree 2
def h : ℝ[X] := polynomial.X ^ 6 -- h(x) is a polynomial of degree 6

-- Statement to prove
theorem degree_of_polynomial_sum :
  (f.comp (polynomial.X ^ 4) * g.comp (polynomial.X ^ 5) + h).degree = 22 :=
by sorry

end degree_of_polynomial_sum_l109_109059


namespace inequality_system_no_solution_l109_109007

theorem inequality_system_no_solution (m : ℝ) :
  (∀ x : ℝ, ¬ (x < m + 1 ∧ x > 2m - 1)) ↔ m ≥ 2 :=
by sorry

end inequality_system_no_solution_l109_109007


namespace siyeon_running_distance_l109_109306

-- Definitions of the conditions
def kilometers_to_meters (km : ℝ) : ℝ :=
  km * 1000

def meters_to_kilometers (m : ℝ) : ℝ :=
  m / 1000

-- Problem Statement: Given that Giljun ran 1.05 km and Siyeon ran 460 m less than Giljun
-- Proof: Calculate Siyeon's running distance in kilometers.
theorem siyeon_running_distance :
  let giljun_distance_km := 1.05
  let siyeon_distance_m := kilometers_to_meters giljun_distance_km - 460
  meters_to_kilometers siyeon_distance_m = 0.59 := by
  sorry

end siyeon_running_distance_l109_109306


namespace g_value_at_172_l109_109260

-- Definition of the function g according to the problem statement.
def g : ℤ → ℤ
| n := if n ≥ 2000 then n - 4 else g (g (n + 6))

-- The theorem statement to be proven.
theorem g_value_at_172 : g 172 = 2000 :=
sorry

end g_value_at_172_l109_109260


namespace max_n_for_positive_sum_l109_109738

open Int

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem max_n_for_positive_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_seq : arithmetic_sequence a)
  (h_cond1 : -1 < (a 7 / a 6) ∧ (a 7 / a 6) < 0)
  (h_max_sum : ∃ n, S n = (n * (a 1 + a n)) / 2 ∧ (∀ m, S m ≤ S n)) :
  ∃ (n : ℕ), (S n > 0) ∧ n = 12 := sorry

end max_n_for_positive_sum_l109_109738


namespace exists_k_and_m_l109_109710

def P : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def f (m : ℕ) (k : ℕ) : ℕ :=
  ∑ i in Finset.range 5, (⌊m * (Real.sqrt ((k + 1) / (i + 2)))⌋ : ℕ)

theorem exists_k_and_m (n : ℕ) (hn : 0 < n) :
  ∃ k ∈ P, ∃ m, f m k = n :=
sorry

end exists_k_and_m_l109_109710


namespace find_omega_l109_109726

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ :=
  sin (ω * x + φ)

theorem find_omega (ω : ℝ) (φ : ℝ) 
  (hω : ω > 0)
  (h_mono : ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < (π / 3) → f x ω φ < f y ω φ)
  (h_sum : f (π / 6) ω φ + f (π / 3) ω φ = 0)
  (h_init : f 0 ω φ = -1) :
  ω = 2 :=
sorry

end find_omega_l109_109726


namespace sqrt_equation_solution_l109_109369

theorem sqrt_equation_solution (x : ℝ) (h : sqrt (2 + sqrt x) = 3) : x = 49 := by
  sorry

end sqrt_equation_solution_l109_109369


namespace interest_difference_l109_109570

theorem interest_difference (P R T : ℝ) (SI : ℝ) (Diff : ℝ) :
  P = 250 ∧ R = 4 ∧ T = 8 ∧ SI = (P * R * T) / 100 ∧ Diff = P - SI → Diff = 170 :=
by sorry

end interest_difference_l109_109570


namespace prove_solutions_l109_109186

-- Define the logarithm base change
def log_base_change (a b x : ℝ) : ℝ := log x / log a * log b / log x

-- Given conditions
axiom condition1 (x : ℝ) : 7.34 * log 3 x * log_base_change 9 3 x * log_base_change 27 3 x * log_base_change 81 3 x = 2 / 3
axiom condition2 (x : ℝ) : 7.351 + 2 * log 2 x * log 4 (10 - x) = 2 / log 4 x

-- Theorem to be proven
theorem prove_solutions (x : ℝ) : (x = 1/9 ∨ x = 9) ↔ (condition1 x ∧ condition2 x) :=
sorry

end prove_solutions_l109_109186


namespace find_f_3_l109_109699

noncomputable def f : ℝ → ℝ
| x => if x ≥ 6 then x - 5 else f (x + 2)

theorem find_f_3 : f 3 = 2 := by
  sorry

end find_f_3_l109_109699


namespace sum_phi_2_4_to_2m_eq_l109_109252

noncomputable def complex_sum_phi (m : ℕ) : ℝ :=
  let z : ℂ := complex.exp (2 * real.pi * complex.I * (1 / (12 * m)))
  let z_k : ℂ := λ k, z ^ (k + 1)
  let φ_k : ℝ := λ k, (k + 1) * (360 / (12 * m))
  Σ (k : ℕ) in finset.filter (λ k, (k + 1) % 2 = 0) (finset.range (2 * m)), φ_k k

theorem sum_phi_2_4_to_2m_eq (m : ℕ) : 
  (finset.sum (finset.filter (λ k, (k + 1) % 2 = 0) (finset.range (2 * m))) (λ k, (k + 1) * (360 / (12 * m)))) = 1290 :=
  sorry

end sum_phi_2_4_to_2m_eq_l109_109252


namespace max_value_of_quadratic_l109_109164

theorem max_value_of_quadratic : ∃ p : ℝ, ∀ x : ℝ, -3 * x^2 + 24 * x + 5 ≤ 53 :=
by
  use 4
  intro x
  have : -3 * (x - 4)^2 + 53 = -3 * ((x - 4)^2 - 16) + 5 := by sorry  -- Completing the square step skipped
  have : -3 * (x - 4)^2 + 53 ≤ 53 := by sorry                     -- It achieves max at p=4 or x-4=0
  exact this

end max_value_of_quadratic_l109_109164


namespace shopkeeper_gain_percentage_l109_109928

theorem shopkeeper_gain_percentage
  (true_weight false_weight : ℝ)
  (H_true_weight : true_weight = 1000)
  (H_false_weight : false_weight = 980) :
  let gain := true_weight - false_weight in
  let gain_percentage := (gain / true_weight) * 100 in
  gain_percentage = 2 := 
by 
  sorry

end shopkeeper_gain_percentage_l109_109928


namespace minimum_problems_45_l109_109405

-- Define the types for problems and their corresponding points
structure Problem :=
(points : ℕ)

def isValidScore (s : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s

def minimumProblems (s : ℕ) (min_problems : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s ∧ x + y + z = min_problems

-- Main statement
theorem minimum_problems_45 : minimumProblems 45 6 :=
by 
  sorry

end minimum_problems_45_l109_109405


namespace trajectory_of_moving_point_l109_109663

noncomputable def point (x y : ℝ) := (x, y)

theorem trajectory_of_moving_point :
  ∀ (P : ℝ × ℝ),
  let M := point 2 0,
      N := point (-2) 0
  in
    (Euclidean.dist P M - Euclidean.dist P N = 2) →
    (P.1^2 - P.2^2 / 3 = 1 ∧ P.1 ≤ -1) :=
begin
  sorry
end

end trajectory_of_moving_point_l109_109663


namespace stamps_problem_l109_109755

def largest_common_divisor (a b c : ℕ) : ℕ :=
  gcd (gcd a b) c

theorem stamps_problem :
  largest_common_divisor 1020 1275 1350 = 15 :=
by
  sorry

end stamps_problem_l109_109755


namespace total_price_of_25_shirts_l109_109018

theorem total_price_of_25_shirts (S W : ℝ) (H1 : W = S + 4) (H2 : 75 * W = 1500) : 
  25 * S = 400 :=
by
  -- Proof would go here
  sorry

end total_price_of_25_shirts_l109_109018


namespace unique_solution_1919_l109_109274

def superdivisor : ℕ → ℕ 
| 0 := 0
| 1 := 0
| n + 1 := (List.range (n + 1)).filter (λ d, (d ∣ n + 1) && (d < n + 1)).reverse.headD 0

def f (n : ℕ) : ℕ :=
  n + nat.rec_on n 0 (λ k, superdivisor)

theorem unique_solution_1919 :
  ∀ n : ℕ, (n + f(superdivisor n) + f(superdivisor (superdivisor n)) + ... = 2021) → n = 1919 :=
  sorry

end unique_solution_1919_l109_109274


namespace xiao_hong_home_to_school_distance_l109_109209

-- Definition of conditions
def distance_from_drop_to_school := 1000 -- in meters
def time_from_home_to_school_walking := 22.5 -- in minutes
def time_from_home_to_school_biking := 40 -- in minutes
def walking_speed := 80 -- in meters per minute
def bike_speed_slowdown := 800 -- in meters per minute

-- The main theorem statement
theorem xiao_hong_home_to_school_distance :
  ∃ d : ℝ, d = 12000 ∧ 
            distance_from_drop_to_school = 1000 ∧
            time_from_home_to_school_walking = 22.5 ∧
            time_from_home_to_school_biking = 40 ∧
            walking_speed = 80 ∧
            bike_speed_slowdown = 800 := 
sorry

end xiao_hong_home_to_school_distance_l109_109209


namespace remainder_242_when_divided_by_13_l109_109169

theorem remainder_242_when_divided_by_13 :
  ∃ R, (∃ k : ℤ, 242 = k * 13 + R) ∧ 
       (∃ m : ℤ, 698 = m * 13 + 9) ∧ 
       (940 = (42:ℤ) * 13 + 4) ∧  R = 8 :=
begin
  sorry
end

end remainder_242_when_divided_by_13_l109_109169


namespace math_problem_l109_109987

theorem math_problem : abs (sqrt 3 - 1) + (π - 3)^0 - tan (real.pi/3) = 0 :=
by
  sorry

end math_problem_l109_109987


namespace grade10_students_in_competition_average_score_of_top10_variance_of_top10_l109_109557

-- We need to define the conditions as constants

constant total_students : ℕ := 1200 + 1000 + 1800
constant num_selected : ℕ := 100
constant scores : list ℕ := [75, 78, 80, 84, 84, 85, 88, 92, 92, 92]

-- Prove the statements

theorem grade10_students_in_competition : 
  let proportion_grade10 := 1200 / total_students
  in num_selected * proportion_grade10 = 30 :=
by
  -- proof will go here
  sorry

theorem average_score_of_top10 : 
  list.sum scores / 10 = 85 :=
by
  -- proof will go here
  sorry

theorem variance_of_top10 :
  let mean := list.sum scores / 10
      squared_diffs := list.map (λ score, (score - mean) ^ 2) scores
  in list.sum squared_diffs / 10 = 33.2 :=
by
  -- proof will go here
  sorry

end grade10_students_in_competition_average_score_of_top10_variance_of_top10_l109_109557


namespace angle_BOC_l109_109147

-- Definitions for the problem conditions
variables {O A B C K : Point}
noncomputable def radius_large (L : Circle) : ℝ := 2
noncomputable def radius_small (S : Circle) : ℝ := 1
def eq_triangle (t : Triangle) : Prop := t.is_equilateral

-- Midpoint of side considered
def is_midpoint (K : Point) (B C : Point) : Prop := dist B K = dist K C

axiom O_center_L : O = center L
axiom O_center_S : O = center S
axiom A_on_L : on_circle A L
axiom K_mid_BC : is_midpoint K B C
axiom K_on_S : on_circle K S
axiom ABC_is_equilateral : eq_triangle (Triangle.mk A B C)

theorem angle_BOC :
  (∠ B O C = 60 ∨ ∠ B O C = 120) :=
sorry

end angle_BOC_l109_109147


namespace length_perpendicular_segment_l109_109849

open Lean
open Classical

variables {A B C D E F : Type*}
variables (AD BE CF : ℝ) (PQ : Set ℝ) (intersectsPQ : D ∉ PQ ∧ E ∉ PQ ∧ F ∉ PQ)

-- Given conditions
/-- Segments AD, BE, CF from vertices A, B, C of a triangle are perpendicular to PQ --/
def segments_and_perpendiculars : Prop :=
  (AD = 12) ∧ (BE = 8) ∧ (CF = 30) ∧ (forall pt : ℝ, (pt = D ∨ pt = E ∨ pt = F) → pt ∈ PQ)

-- To be proven: The length z of the perpendicular segment IJ
/-- Length z from the centroid to line PQ --/
theorem length_perpendicular_segment {A B C D E F : ℝ} (h : segments_and_perpendiculars AD BE CF PQ intersectsPQ) :
  let y_A := 12
  let y_B := 8
  let y_C := 30 in
  let y_I := (y_A + y_B + y_C) / 3 in
  y_I = 50 / 3 := by
  sorry

end length_perpendicular_segment_l109_109849


namespace unique_real_value_c_l109_109298

theorem unique_real_value_c (c : ℝ) : (abs (1 - (c + 1) * complex.I) = 1) → c = -1 :=
by
  intro h
  sorry

end unique_real_value_c_l109_109298


namespace regular_price_of_fish_l109_109959

theorem regular_price_of_fish (discounted_price_per_quarter_pound : ℝ)
  (discount : ℝ) (hp1 : discounted_price_per_quarter_pound = 2) (hp2 : discount = 0.4) :
  ∃ x : ℝ, x = (40 / 3) :=
by
  sorry

end regular_price_of_fish_l109_109959


namespace unique_function_l109_109273

noncomputable def f (x : ℝ) : ℝ := sorry

theorem unique_function (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f(x * f(y)) = y * f(x)) 
  (h2 : ∀ ε : ℝ, ε > 0 → ∃ M : ℝ, ∀ x : ℝ, x > M → f(x) < ε) : 
  ∀ x : ℝ, 0 < x → f(x) = 1 / x :=
sorry

end unique_function_l109_109273


namespace birds_more_than_nests_l109_109494

theorem birds_more_than_nests : 
  (number_of_birds number_of_nests: ℕ) (h_birds : number_of_birds = 6) (h_nests : number_of_nests = 3) :
  number_of_birds - number_of_nests = 3 :=
by
  sorry

end birds_more_than_nests_l109_109494


namespace area_of_EFCD_l109_109037

theorem area_of_EFCD (AB CD h : ℝ) (H_AB : AB = 10) (H_CD : CD = 30) (H_h : h = 15) :
  let EF := (AB + CD) / 2
  let h_EFCD := h / 2
  let area_EFCD := (1 / 2) * (CD + EF) * h_EFCD
  area_EFCD = 187.5 :=
by
  intros EF h_EFCD area_EFCD
  sorry

end area_of_EFCD_l109_109037


namespace angle_alpha_value_fraction_value_l109_109696

-- Proof Problem 1
theorem angle_alpha_value
  (A : ℝ × ℝ := (4, 0))
  (B : ℝ × ℝ := (0, 4))
  (C : ℝ → ℝ → ℝ × ℝ := λ α, (3 * Real.cos α, 3 * Real.sin α))
  (α : ℝ)
  (h1 : α ∈ Set.Ioo (-Real.pi) 0)
  (h2 : ∥C α - A∥ = ∥C α - B∥) :
  α = -3 * Real.pi / 4 :=
sorry

-- Proof Problem 2
theorem fraction_value
  (A : ℝ × ℝ := (4, 0))
  (B : ℝ × ℝ := (0, 4))
  (C : ℝ → ℝ → ℝ × ℝ := λ α, (3 * Real.cos α, 3 * Real.sin α))
  (α : ℝ)
  (h3 : ((C α.1 - A.1) * C α.1 + C α.2 * (C α.2 - 4)) = 0) :
  (2 * (Real.sin α)^2 + Real.sin (2 * α)) / (1 + Real.tan α) = -7 / 16 :=
sorry

end angle_alpha_value_fraction_value_l109_109696


namespace even_divisor_probability_of_18_factorial_l109_109122

theorem even_divisor_probability_of_18_factorial :
  let d_total := (15 + 1) * (8 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let d_even := d_total - (8 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  d_even / d_total = 3 / 4 :=
by {
  let d_total := 16 * 9 * 4 * 3 * 2 * 2 * 2,
  let d_even := d_total - 9 * 4 * 3 * 2 * 2 * 2,
  show d_even / d_total = 3 / 4,
  sorry
}

end even_divisor_probability_of_18_factorial_l109_109122


namespace sum_of_reciprocals_of_roots_l109_109057

-- Given polynomial and its roots
def polynomial := (λ x : ℂ, x^12 + x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x - 650)

def roots (𝕜 : Type*) [Field 𝕜] [Algebra ℂ 𝕜] : Fin 12 → 𝕜 := sorry

theorem sum_of_reciprocals_of_roots (ascs: ∀ i, polynomial (roots ℂ i) = 0) :
  (∑ n in Finset.range 12, 1 / (1 - (roots ℂ n))) = (39 : ℚ) / 319 :=
sorry

end sum_of_reciprocals_of_roots_l109_109057


namespace number_of_tickets_bought_l109_109156

theorem number_of_tickets_bought (ticket_cost total_spent : ℕ) (h1 : ticket_cost = 44) (h2 : total_spent = 308) :
  total_spent / ticket_cost = 7 :=
by
  rw [h1, h2]
  norm_num
  sorry

end number_of_tickets_bought_l109_109156


namespace first_instance_height_35_l109_109113
noncomputable def projectile_height (t : ℝ) : ℝ := -5 * t^2 + 30 * t

theorem first_instance_height_35 {t : ℝ} (h : projectile_height t = 35) :
  t = 3 - Real.sqrt 2 :=
sorry

end first_instance_height_35_l109_109113


namespace math_equivalence_example_l109_109304

theorem math_equivalence_example :
  ((3.242^2 * (16 + 8)) / (100 - (3 * 25))) + (32 - 10)^2 = 494.09014144 := 
by
  sorry

end math_equivalence_example_l109_109304


namespace num_divisors_not_multiple_of_14_l109_109788

theorem num_divisors_not_multiple_of_14 :
  ∃ n : ℕ, 
    (∃ a b c d : ℕ, n = 2 * a^2 ∧ n = 3 * b^3 ∧ n = 5 * c^5 ∧ n = 7 * d^7) ∧
    (let total_divisors := (14 + 1) * (21 + 1) * (35 + 1) * (7 + 1),
         divisors_multiple_of_14 := (13 + 1) * (21 + 1) * (35 + 1) * (6 + 1) in
        total_divisors - divisors_multiple_of_14 = 16416) :=
begin
  sorry
end

end num_divisors_not_multiple_of_14_l109_109788


namespace theta_value_l109_109171

theorem theta_value (Theta : ℕ) (h_digit : Θ < 10) (h_eq : 252 / Θ = 30 + 2 * Θ) : Θ = 6 := 
by
  sorry

end theta_value_l109_109171


namespace sheila_monthly_savings_l109_109850

-- Define the conditions and the question in Lean
def initial_savings : ℕ := 3000
def family_contribution : ℕ := 7000
def years : ℕ := 4
def final_amount : ℕ := 23248

-- Function to calculate the monthly saving given the conditions
def monthly_savings (initial_savings family_contribution years final_amount : ℕ) : ℕ :=
  (final_amount - (initial_savings + family_contribution)) / (years * 12)

-- The theorem we need to prove in Lean
theorem sheila_monthly_savings :
  monthly_savings initial_savings family_contribution years final_amount = 276 :=
by
  sorry

end sheila_monthly_savings_l109_109850


namespace sufficient_condition_l109_109058

def f (x : ℝ) : ℝ := x^5 + Real.log (x + Real.sqrt (x^2 + 1))

theorem sufficient_condition (a b : ℝ) (h : a + b ≥ 0) : f(a) + f(b) ≥ 0 := sorry

end sufficient_condition_l109_109058


namespace buckets_required_l109_109537

theorem buckets_required (C : ℝ) (tank_volume : ℝ) (H : tank_volume = 200 * C) : 
  let new_capacity := (4/5) * C in (tank_volume / new_capacity) = 250 := 
by
  sorry

end buckets_required_l109_109537


namespace round_nearest_hundredth_problem_l109_109174

noncomputable def round_nearest_hundredth (x : ℚ) : ℚ :=
  let shifted := x * 100
  let rounded := if (shifted - shifted.floor) < 0.5 then shifted.floor else shifted.ceil
  rounded / 100

theorem round_nearest_hundredth_problem :
  let A := 34.561
  let B := 34.558
  let C := 34.5539999
  let D := 34.5601
  let E := 34.56444
  round_nearest_hundredth A = 34.56 ∧
  round_nearest_hundredth B = 34.56 ∧
  round_nearest_hundredth C ≠ 34.56 ∧
  round_nearest_hundredth D = 34.56 ∧
  round_nearest_hundredth E = 34.56 :=
sorry

end round_nearest_hundredth_problem_l109_109174


namespace line_through_intersection_and_parallel_line_perpendicular_and_distance_l109_109938

-- First problem: equation of the line through intersection of l1 and l2, parallel to x+2y-3=0
theorem line_through_intersection_and_parallel :
  ∃ (a b c : ℝ), (a = 9) ∧ (b = 18) ∧ (c = -4) ∧
  (∀ (x y : ℝ), (2*x + 3*y - 5 = 0) → (7*x + 15*y + 1 = 0) → a*x + b*y + c = 0) ∧
  (∀ (x y : ℝ), (x + 2*y - 3 = 0) → a*x + b*y + c = 0) := 
sorry

-- Second problem: equation of the line perpendicular to 3x+4y-7=0 and at distance 6 from origin
theorem line_perpendicular_and_distance :
  ∃ (a b c d : ℝ), 
    ((a = 4) ∧ (b = -3) ∧ (d = 30) ∧ 
    (∀ (x y : ℝ), (3*x + 4*y - 7 = 0) → (a*x + b*y + c ≠ 0) ∧ 
    (real.dist 0 0 (a*x + b*y + c) = 6)) ∧
    (∀ (x y : ℝ), (a*x + b*y - d = 0) ∧ (a*x + b*y + d = 0))) := 
sorry

end line_through_intersection_and_parallel_line_perpendicular_and_distance_l109_109938


namespace sabrina_basil_leaves_l109_109089

-- Definitions of variables
variables (S B V : ℕ)

-- Conditions as definitions in Lean
def condition1 : Prop := B = 2 * S
def condition2 : Prop := S = V - 5
def condition3 : Prop := B + S + V = 29

-- Problem statement
theorem sabrina_basil_leaves (h1 : condition1 S B) (h2 : condition2 S V) (h3 : condition3 S B V) : B = 12 :=
by {
  sorry
}

end sabrina_basil_leaves_l109_109089


namespace monotonic_increasing_interval_l109_109480

noncomputable def f (x : ℝ) := 2 * x - Real.log x

theorem monotonic_increasing_interval : 
  {x : ℝ | x > 0 ∧ ∀ x₁ x₂ : ℝ, x₁ > x₂ → f'(x₁) > f'(x₂)} = (1/2 : ℝ, +∞) := 
by
  -- Proof goes here
  sorry

-- Calculating the derivative to be used in the proof
noncomputable def f' (x : ℝ) := 2 - (1 / x)

end monotonic_increasing_interval_l109_109480


namespace earl_stuffing_rate_l109_109265

theorem earl_stuffing_rate :
  let E := 36 in
  ∀ (E: ℝ) (L: ℝ)
    (h1: L = (2/3) * E)
    (h2: E + L = 60),
  E = 36 :=
by 
  intros E L h1 h2
  sorry

end earl_stuffing_rate_l109_109265


namespace length_AC_of_triangle_l109_109385

theorem length_AC_of_triangle
  (A B C : ℝ)
  (BC : ℝ := 1)
  (angle_B : ℝ := (real.pi / 3))
  (area_ABC : ℝ := real.sqrt 3)
  (AB : ℝ := 4)
  (AC : ℝ)
  (hBC : BC = 1)
  (hAngleB : angle_B = real.pi / 3)
  (hAreaABC : area_ABC = real.sqrt 3)
  (hAB : AB = 4) :
  AC = real.sqrt 13 :=
sorry

end length_AC_of_triangle_l109_109385


namespace correlation_comparison_l109_109864

def data_X_Y : list (ℝ × ℝ) := [(10, 1), (11.3, 2), (11.8, 3), (12.5, 4), (13, 5)]
def data_U_V : list (ℝ × ℝ) := [(10, 5), (11.3, 4), (11.8, 3), (12.5, 2), (13, 1)]

def mean (lst : list ℝ) : ℝ :=
  lst.sum / lst.length

-- Calculate means
def mean_X : ℝ := mean (data_X_Y.map Prod.fst)
def mean_Y : ℝ := mean (data_X_Y.map Prod.snd)
def mean_U : ℝ := mean (data_U_V.map Prod.fst)
def mean_V : ℝ := mean (data_U_V.map Prod.snd)

-- Calculate the correlation coefficients (placeholders to insert actual computation below)
def correlation_coefficient (data : list (ℝ × ℝ)) : ℝ :=
  sorry

def r1 : ℝ := correlation_coefficient data_X_Y
def r2 : ℝ := correlation_coefficient data_U_V

theorem correlation_comparison :
  r2 < 0 ∧ 0 < r1 :=
sorry

end correlation_comparison_l109_109864


namespace widgets_made_per_week_l109_109756

theorem widgets_made_per_week
  (widgets_per_hour : Nat)
  (hours_per_day : Nat)
  (days_per_week : Nat)
  (total_widgets : Nat) :
  widgets_per_hour = 20 →
  hours_per_day = 8 →
  days_per_week = 5 →
  total_widgets = widgets_per_hour * hours_per_day * days_per_week →
  total_widgets = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end widgets_made_per_week_l109_109756


namespace find_possible_values_l109_109676

noncomputable def abs_sign (x : ℝ) : ℝ := x / |x|

theorem find_possible_values (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) :
  ∃ (v : ℝ), v ∈ {-5, -2, 1, 4, 5} ∧ 
  v = abs_sign a + abs_sign b + abs_sign c + abs_sign d + abs_sign (a * b * c * d) :=
by
  sorry

end find_possible_values_l109_109676


namespace ratio_of_other_triangle_l109_109143

noncomputable def ratioAreaOtherTriangle (m : ℝ) : ℝ := 1 / (4 * m)

theorem ratio_of_other_triangle (m : ℝ) (h : m > 0) : ratioAreaOtherTriangle m = 1 / (4 * m) :=
by
  -- Proof will be provided here
  sorry

end ratio_of_other_triangle_l109_109143


namespace find_C_l109_109140

theorem find_C (A B C : ℕ) (h1 : 3 * A - A = 10) (h2 : B + A = 12) (h3 : C - B = 6) : C = 13 :=
by
  sorry

end find_C_l109_109140


namespace tens_digit_of_n_sq_odd_count_l109_109288

theorem tens_digit_of_n_sq_odd_count :
  {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧
           ((n * n / 10) % 10 % 2 = 1)}.card = 20 := sorry

end tens_digit_of_n_sq_odd_count_l109_109288


namespace sum_of_solutions_l109_109763

theorem sum_of_solutions :
  ∀ (x y : ℤ), (|x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|) →
  ({ (2, 8), (4, 10), (10, 4), (10, 4) }).sum (λ p, p.1 + p.2) = 52 :=
by
  sorry

end sum_of_solutions_l109_109763


namespace train_pass_bridge_time_l109_109966

-- Definitions of the given conditions
def train_length : ℝ := 750 -- in meters
def train_speed_kmh : ℝ := 72 -- in km/h
def bridge_length : ℝ := 280 -- in meters

-- Convert the speed from km/h to m/s
def kmh_to_ms (speed : ℝ) : ℝ := speed * (1000 / 3600)
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Total distance to be covered
def total_distance : ℝ := train_length + bridge_length

-- Calculate the time to pass the bridge
def time_to_pass : ℝ := total_distance / train_speed_ms

-- Theorem stating the solution to the problem
theorem train_pass_bridge_time : time_to_pass = 51.5 := by
  -- This is where the proof would go, but we will use sorry for now
  sorry

end train_pass_bridge_time_l109_109966


namespace fraction_of_KJ_l109_109596

variable (KJ_stamps AJ_stamps CJ_stamps total_stamps : ℤ)

def AJ_stamps : ℤ := 370
def total_stamps : ℤ := 930

theorem fraction_of_KJ (f : ℚ) (h : KJ_stamps = f * AJ_stamps) (h1 : CJ_stamps = 2 * KJ_stamps + 5) 
    (h2 : AJ_stamps + KJ_stamps + CJ_stamps = total_stamps) : 
    f = 1 / 2 := 
  sorry

end fraction_of_KJ_l109_109596


namespace peanuts_added_l109_109492

theorem peanuts_added (initial final added : ℕ) (h1 : initial = 4) (h2 : final = 8) (h3 : final = initial + added) : added = 4 :=
by
  rw [h1] at h3
  rw [h2] at h3
  sorry

end peanuts_added_l109_109492


namespace find_m_l109_109331

open Real

noncomputable def m_value (x : ℝ) : ℝ :=
  let sinx := sin x
  let cosx := cos x
  4 * sqrt (1.02)

theorem find_m (x : ℝ) (m : ℝ)
  (h1 : log 10 (sin x) + log 10 (cos x) = -2)
  (h2 : log 2 (sin x + cos x) = log 2 m - 2) :
  m = m_value x := 
sorry

end find_m_l109_109331


namespace room_count_l109_109937

theorem room_count : ∀ (x : ℕ), (7 * x + 7 = 9 * (x - 1)) → x = 8 :=
by
  intros x h,
  sorry

end room_count_l109_109937


namespace remainder_when_sum_div_by_3_l109_109723

theorem remainder_when_sum_div_by_3 
  (m n p q : ℕ)
  (a : ℕ := 6 * m + 4)
  (b : ℕ := 6 * n + 4)
  (c : ℕ := 6 * p + 4)
  (d : ℕ := 6 * q + 4)
  : (a + b + c + d) % 3 = 1 :=
by
  sorry

end remainder_when_sum_div_by_3_l109_109723


namespace arithmetic_geometric_sequences_l109_109693

theorem arithmetic_geometric_sequences :
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℤ), 
  (a₂ = a₁ + (a₁ - (-1))) ∧ 
  (-4 = -1 + 3 * (a₂ - a₁)) ∧ 
  (-4 = -1 * (b₃/b₁)^4) ∧ 
  (b₂ = b₁ * (b₂/b₁)^2) →
  (a₂ - a₁) / b₂ = 1 / 2 := 
by
  intros a₁ a₂ b₁ b₂ b₃ h
  sorry

end arithmetic_geometric_sequences_l109_109693


namespace races_condition_possible_l109_109200

variables {τ : Type} [LinearOrder τ]

def beats (x y : τ) (races : list (list τ)) :=
  (list.countp (λ race, race.index_of x < race.index_of y) races) > (races.length / 2)

def condition (A B C : τ) (races : list (list τ)) := 
  beats A B races ∧ beats B C races ∧ beats C A races

theorem races_condition_possible (A B C : τ) :
  ∃ races : list (list τ), condition A B C races :=
begin
  -- proof would go here
  sorry
end

end races_condition_possible_l109_109200


namespace geometric_sequence_offset_zero_l109_109314

variables {α : Type*} [linear_ordered_field α]

-- Definitions of geometric sequences and conditions
def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_geometric_sequence_offset (a : ℕ → α) (c : α) (q : α) : Prop :=
  ∀ n : ℕ, (a n + c) * (a (n + 2) + c) = (a (n + 1) + c) * (a (n + 1) + c)

-- Lean 4 statement of the problem
theorem geometric_sequence_offset_zero
  (a : ℕ → α) (q : α) (c : α)
  (h1 : q ≠ 1)
  (h2 : is_geometric_sequence a q)
  (h3 : is_geometric_sequence_offset a c q) :
  c = 0 :=
sorry

end geometric_sequence_offset_zero_l109_109314


namespace find_a_l109_109702

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (x + 1) = 3 * x + 2) (h2 : f a = 5) : a = 2 :=
sorry

end find_a_l109_109702


namespace regular_polygon_sides_l109_109005

theorem regular_polygon_sides (θ : ℝ) (hθ : θ = 45) : 360 / θ = 8 := by
  sorry

end regular_polygon_sides_l109_109005


namespace proof_problem_l109_109259

def star (x y : ℝ) : ℝ := x^3 - y

theorem proof_problem :
  star (3 ^ star 5 18) (2 ^ star 2 9) = 3 ^ 321 - 1 / 2 :=
by
  have h₁ : star 5 18 = 107 := by rfl
  have h₂ : star 2 9 = -1 := by rfl
  have h₃ : 3 ^ star 5 18 = 3 ^ 107 := by simp [h₁]
  have h₄ : 2 ^ star 2 9 = 1 / 2 := by simp [h₂]
  simp [star, h₃, h₄]
  sorry

end proof_problem_l109_109259


namespace volume_is_216_l109_109475

-- Define the conditions
def space_diagonal (s : ℝ) : ℝ := s * (real.sqrt 3)

def volume_of_cube (s : ℝ) : ℝ := s^3

-- Assume the space diagonal is 6√3
axiom distance_AG : ∃ s, space_diagonal s = 6 * (real.sqrt 3)

-- Main theorem to prove the volume of the cube is 216 cubic units
theorem volume_is_216 (s : ℝ) (h : space_diagonal s = 6 * (real.sqrt 3)) : volume_of_cube s = 216 := by
  sorry

end volume_is_216_l109_109475


namespace full_time_and_year_l109_109075

variable (Total F Y N FY : ℕ)

theorem full_time_and_year (h1 : Total = 130)
                            (h2 : F = 80)
                            (h3 : Y = 100)
                            (h4 : N = 20)
                            (h5 : Total = FY + (F - FY) + (Y - FY) + N) :
    FY = 90 := 
sorry

end full_time_and_year_l109_109075


namespace parabola_equation_l109_109283

/--
Given a point P (4, -2) on a parabola, prove that the equation of the parabola is either:
1) y^2 = x or
2) x^2 = -8y.
-/
theorem parabola_equation (p : ℝ) (x y : ℝ) (h1 : (4 : ℝ) = 4) (h2 : (-2 : ℝ) = -2) :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ 4 = 4 ∧ y = -2) ∨ (∃ p : ℝ, x^2 = 2 * p * y ∧ 4 = 4 ∧ x = 4) :=
sorry

end parabola_equation_l109_109283


namespace allocation_methods_count_l109_109138

theorem allocation_methods_count :
  let doctors := 3
  let nurses := 6
  let schools := 3
  (doctors = 3 ∧ nurses = 6 ∧ schools = 3) →
  (number_of_ways : ℕ) →
  (number_of_ways = (3 * (nurses.choose 2) * 2 * ((nurses - 2).choose 2) * 1 * ((nurses - 4).choose 2))) →
  number_of_ways = 540 :=
by
  intros doctors nurses schools h number_of_ways h_ways
  have h_doctors : doctors = 3 := by tauto
  have h_nurses : nurses = 6 := by tauto
  have h_schools : schools = 3 := by tauto
  rw [h_doctors, h_nurses, h_schools] at h_ways
  sorry

end allocation_methods_count_l109_109138


namespace rose_bushes_in_park_l109_109890

theorem rose_bushes_in_park (current_bushes : ℕ) (newly_planted : ℕ) (h1 : current_bushes = 2) (h2 : newly_planted = 4) : current_bushes + newly_planted = 6 :=
by
  sorry

end rose_bushes_in_park_l109_109890


namespace equal_AN_NC_l109_109949

open EuclideanGeometry

theorem equal_AN_NC 
(triangle_abc : Triangle ℝ α β γ)
(circumcircle : Circle ℝ α β γ)
(tangent_A : Line ℝ α)
(tangent_B : Line ℝ β)
(M : Point ℝ)
(intersect_MB : intersects M tangent_B)
(intersect_MA : intersects M tangent_A)
(N : Point ℝ)
(lies_on_BC : lies_on N β γ)
(parallel_MN_AC : parallel (line_through M N) (line_through α γ))
: dist α N = dist N γ := 
sorry

end equal_AN_NC_l109_109949


namespace largest_four_digit_divisible_by_5_l109_109517

theorem largest_four_digit_divisible_by_5 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 5 = 0 → m ≤ n :=
begin
  use 9995,
  split, norm_num,
  split, norm_num,
  split, norm_num,
  intros m hm1 hm2 hm3,
  have H : m ≤ 9999, from hm2,
  norm_num at H,
  sorry
end

end largest_four_digit_divisible_by_5_l109_109517


namespace smallest_positive_even_multiple_l109_109521

theorem smallest_positive_even_multiple (n : ℕ) (h_even : n % 2 = 0) (h_mul_5 : n % 5 = 0) (h_mul_8 : n % 8 = 0) : n = 40 :=
by
  -- We define the conditions for the answer to be the smallest positive even integer 
  -- that is a multiple of both 5 and 8.
  have lcm_val : Nat.lcm 5 8 = 40 := by 
    sorry
  exact lcm_val

end smallest_positive_even_multiple_l109_109521


namespace number_of_diagonals_in_octagon_l109_109221

theorem number_of_diagonals_in_octagon (n : ℕ) (h1 : n = 8) (h2 : ∀ (p : Polygon), convex p) (h3 : ∀ (p : Polygon), has_right_angle p) : 
  (n * (n - 3)) / 2 = 20 :=
by
  have h : (8 * (8 - 3)) / 2 = 20 := by simp
  exact h

end number_of_diagonals_in_octagon_l109_109221


namespace cross_section_area_l109_109879

-- Definitions representing the conditions
variables (AK KD BP PC DM DC : ℝ)
variable (h : ℝ)
variable (Volume : ℝ)

-- Conditions
axiom hyp1 : AK = KD
axiom hyp2 : BP = PC
axiom hyp3 : DM = 0.4 * DC
axiom hyp4 : h = 1
axiom hyp5 : Volume = 5

-- Proof problem: Prove that the area S of the cross-section of the pyramid is 3
theorem cross_section_area (S : ℝ) : S = 3 :=
by sorry

end cross_section_area_l109_109879


namespace find_sides_and_angle_a_in_triangle_l109_109411

noncomputable def triangle_sides_angle_a (a b c : ℝ) (sin_ratio : ℝ) (cos_a : ℝ) : Prop :=
b = 5 ∧ acos(cos_a) = 120 * (Real.pi / 180)

theorem find_sides_and_angle_a_in_triangle (a b c : ℝ) (sin_ratio : ℝ) (cos_a : ℝ) 
    (ha : a = 7)
    (hc : c = 3)
    (hsin_ratio : sin_ratio = 3 / 5)
    (hcos_a : cos_a = -1/2) : 
    triangle_sides_angle_a a b c sin_ratio cos_a :=
by
  rw [ha, hc, hsin_ratio, hcos_a]
  sorry

end find_sides_and_angle_a_in_triangle_l109_109411


namespace shaded_region_ratio_l109_109210

-- Define the main conditions
def large_square_side_length : ℕ := 10  -- since each smaller square's side length is 2 and there are 5 such squares on each side
def grid_dimension : ℕ := 5
def square_side_length : ℕ := 2

-- Define the areas
def area_of_large_square : ℕ := (large_square_side_length * large_square_side_length)
def area_of_smaller_square : ℕ := (square_side_length * square_side_length)

-- The shaded region forms a quadrilateral with specific midpoints
-- Define the midpoints for clarity:
def midpoints := {(1, 1), (3, 2), (4, 3), (2, 4)}

-- Calculate the area of the quadrilateral (shaded region)
def area_of_shaded_region : ℕ := 2  -- as derived from the solution steps

-- Calculate the ratio
def ratio := (area_of_shaded_region : ℚ) / (area_of_large_square : ℚ)

-- State the lean problem
theorem shaded_region_ratio : ratio = 1 / 50 := by
  sorry

end shaded_region_ratio_l109_109210


namespace area_of_figure_is_13_l109_109827

noncomputable def figure_area : ℝ :=
  if h : ∃ S : set (ℝ × ℝ), 
    (∀ p ∈ S, 2 * p.1^2 - 3 * p.1 * p.2 - 2 * p.2^2 ≤ 0 ∧
      |2 * p.1 + 3 * p.2| + |3 * p.1 - 2 * p.2| ≤ 13) ∧
    measure_theory.measure (set.univ).restrict S = 13 then
    measure_theory.measure (set.univ).restrict (classical.some h)
  else 0

theorem area_of_figure_is_13 : figure_area = 13 :=
sorry

end area_of_figure_is_13_l109_109827


namespace average_speed_is_correct_l109_109487

-- Definitions for the conditions
def speed_first_hour : ℕ := 140
def speed_second_hour : ℕ := 40
def total_distance : ℕ := speed_first_hour + speed_second_hour
def total_time : ℕ := 2

-- The statement we need to prove
theorem average_speed_is_correct : total_distance / total_time = 90 := by
  -- We would place the proof here
  sorry

end average_speed_is_correct_l109_109487


namespace largest_consecutive_odd_integer_sum_l109_109894

theorem largest_consecutive_odd_integer_sum :
  ∃ N : ℤ, N + (N + 2) + (N + 4) = -147 ∧ (N + 4) = -47 :=
begin
  sorry
end

end largest_consecutive_odd_integer_sum_l109_109894


namespace walter_hourly_wage_l109_109159

-- Define the conditions as Lean variables and constants
def daysPerWeek : ℕ := 5
def hoursPerDay : ℕ := 4
def fractionForSchool : ℚ := 3/4
def amountForSchool : ℚ := 75

-- Define the question as a Lean theorem statement
theorem walter_hourly_wage : 
  let totalHoursWeek := daysPerWeek * hoursPerDay in
  let totalWeeklyEarnings := amountForSchool / fractionForSchool in
  totalWeeklyEarnings / totalHoursWeek = 5 := 
by
  -- Proof omitted
  sorry

end walter_hourly_wage_l109_109159


namespace population_present_l109_109376

variable (P : ℝ)

theorem population_present (h1 : P * 0.90 = 450) : P = 500 :=
by
  sorry

end population_present_l109_109376


namespace oscar_leap_vs_elmer_stride_l109_109012

/--
Given:
1. The 51st telephone pole is exactly 6600 feet from the first pole.
2. Elmer the emu takes 50 equal strides to walk between consecutive telephone poles.
3. Oscar the ostrich can cover the same distance in 15 equal leaps.
4. There are 50 gaps between the 51 poles.

Prove:
Oscar's leap is 6 feet longer than Elmer's stride.
-/
theorem oscar_leap_vs_elmer_stride : 
  let total_distance := 6600 
  let elmer_strides_per_gap := 50
  let oscar_leaps_per_gap := 15
  let num_gaps := 50
  let elmer_total_strides := elmer_strides_per_gap * num_gaps
  let oscar_total_leaps := oscar_leaps_per_gap * num_gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = 6 := 
by {
  -- The proof would go here.
  sorry
}

end oscar_leap_vs_elmer_stride_l109_109012


namespace fraction_of_students_with_buddy_l109_109393

theorem fraction_of_students_with_buddy (t s : ℕ) (h1 : (t / 4) = (3 * s / 5)) :
  (t / 4 + 3 * s / 5) / (t + s) = 6 / 17 :=
by
  sorry

end fraction_of_students_with_buddy_l109_109393


namespace inequality_range_l109_109704

theorem inequality_range (a : ℝ) : 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 ≤ y ∧ y ≤ 3 → x * y ≤ a * x^2 + 2 * y^2) ↔ a ≥ -1 := by 
  sorry

end inequality_range_l109_109704


namespace total_goals_l109_109742

-- Definitions
def louie_goals_last_match := 4
def louie_previous_goals := 40
def brother_multiplier := 2
def seasons := 3
def games_per_season := 50

-- Total number of goals scored by Louie and his brother
theorem total_goals : (louie_previous_goals + louie_goals_last_match) 
                      + (brother_multiplier * louie_goals_last_match * seasons * games_per_season) 
                      = 1244 :=
by sorry

end total_goals_l109_109742


namespace p_element_subsets_with_sum_divisible_by_p_l109_109424

theorem p_element_subsets_with_sum_divisible_by_p (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  let s := (Finset.range (2 * p) ).filter (λ A, A.card = p ∧ (A.sum % p = 0))
  s.card = (Nat.choose (2 * p) p - 2) / p + 2 :=
sorry

end p_element_subsets_with_sum_divisible_by_p_l109_109424


namespace range_of_a_l109_109817

variable {a : ℝ}

def proposition_p : Prop := 
  ∀ x : ℝ, ¬(x^2 - (a-1)*x + 1 ≤ 0)

def proposition_q : Prop := 
  ∀ x y : ℝ, x < y → (a+1)^x < (a+1)^y

def p_and_q_false : Prop := ¬(proposition_p ∧ proposition_q)

def p_or_q_true : Prop := proposition_p ∨ proposition_q

theorem range_of_a : p_and_q_false ∧ p_or_q_true → (a > -1 ∧ a ≤ 0) ∨ (a ≥ 3) :=
by
  sorry

end range_of_a_l109_109817


namespace pencils_per_row_l109_109630

-- Define the conditions
def total_pencils := 25
def number_of_rows := 5

-- Theorem statement: The number of pencils per row is 5 given the conditions
theorem pencils_per_row : total_pencils / number_of_rows = 5 :=
by
  -- The proof should go here
  sorry

end pencils_per_row_l109_109630


namespace exists_sequence_l109_109661

theorem exists_sequence (k : ℕ) (h : 0 < k) :
  ∃ (a : ℕ → ℕ), (a 1 = k) ∧ (∀ n, ∑ i in Finset.range (n + 1), (a i)^2 ∣ ∑ i in Finset.range (n + 1), a i) :=
by
  sorry

end exists_sequence_l109_109661


namespace rate_of_stream_is_5_l109_109554

-- Define the conditions
def boat_speed : ℝ := 16  -- Boat speed in still water
def time_downstream : ℝ := 3  -- Time taken downstream
def distance_downstream : ℝ := 63  -- Distance covered downstream

-- Define the rate of the stream as an unknown variable
def rate_of_stream (v : ℝ) : Prop := 
  distance_downstream = (boat_speed + v) * time_downstream

-- Statement to prove
theorem rate_of_stream_is_5 : 
  ∃ (v : ℝ), rate_of_stream v ∧ v = 5 :=
by
  use 5
  simp [boat_speed, time_downstream, distance_downstream, rate_of_stream]
  sorry

end rate_of_stream_is_5_l109_109554


namespace Moe_mows_in_correct_time_l109_109822

open Real

-- Definition of the conditions
def lawn_length := 90
def lawn_width := 150
def swath_width_in_inches := 28
def overlap_in_inches := 4
def walking_speed := 5000

-- The effective swath width in feet
def effective_swath_width := (swath_width_in_inches - overlap_in_inches) / 12

-- The correct answer we need to prove
def mowing_time := 1.35

-- The theorem we need to prove
theorem Moe_mows_in_correct_time
  (lawn_length lawn_width : ℝ)
  (effective_swath_width : ℝ)
  (walking_speed : ℝ)
  (mowing_time : ℝ) :
  (lawn_length * real.ceil (lawn_width / effective_swath_width)) / walking_speed = mowing_time := by
  sorry

end Moe_mows_in_correct_time_l109_109822


namespace problem_ACD_l109_109683

noncomputable theory
open Classical Real

def line_l (m : ℝ) : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ mx - y - 3 * m + 4 = 0}
def circle_O : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ x ^ 2 + y ^ 2 = 4}

theorem problem_ACD :
  (∀ m, ∃ P : ℝ × ℝ, ∃ Q : ℝ × ℝ, line_l m P ∧ circle_O Q ∧
  -- A: The maximum distance from point Q to line l is 7
  max_distance Q P m = 7 ∧ 

  -- B: When the chord length of line l intersecting circle O is maximum, the value of m is not 1
  ¬(chord_intersection_max m = 1) ∧ 

  -- C: If line l is tangent to circle O, the value of m is (12 ± 2√21)/5
  (is_tangent m → (m = (12 + 2 * sqrt 21) / 5 ∨ m = (12 - 2 * sqrt 21) / 5)) ∧

  -- D: If the chord length intercepted by line l and circle O is 2√3, the value of m is (6 ± √6)/4
  (chord_length_intersection m = 2 * sqrt 3 → 
    (m = (6 + sqrt 6) / 4 ∨ m = (6 - sqrt 6) / 4))) := sorry

end problem_ACD_l109_109683


namespace volume_of_rotated_solid_l109_109717

noncomputable def volume_of_solid (r : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, π * (r x)^2

def abs_val_ratio (x y : ℝ) : Prop :=
  |x / 3| + |y / 3| = 2

theorem volume_of_rotated_solid :
  ∃ V : ℝ, 
  ∀ x y : ℝ, 
  abs_val_ratio x y → 
  V = volume_of_solid (λ y, abs ((3 * (2 - |x / 3|)) / 2)) (-6) 6 :=
sorry

end volume_of_rotated_solid_l109_109717


namespace geometric_sequence_and_area_l109_109730

-- Define the conditions
variables {A B C a b c : Real}
variable h1 : sin B * (tan A + tan C) = tan A * tan C
variable ha : a = 1
variable hc : c = 2

-- Define the theorem
theorem geometric_sequence_and_area 
  (h1 : sin B * (tan A + tan C) = tan A * tan C)
  (ha : a = 1)
  (hc : c = 2) :
  b^2 = a * c ∧ (1 / 2 * a * c * sin B) = sqrt 7 / 4 :=
by
  sorry

end geometric_sequence_and_area_l109_109730


namespace right_triangle_legs_distances_l109_109539

theorem right_triangle_legs_distances 
  (a b : ℝ) :
  (∃ (OB OC : ℝ), OB = √5 ∧ OC = √10 ∧ ∃ r : ℝ, 
   a = (√(5 - r^2)) + r ∧ b = (√(10 - r^2)) + r ∧ 
   a^2 + b^2 = (a + b - r)^2) → 
  (a = 4 ∧ b = 3) 
  ∨ (a = 3 ∧ b = 4) :=
sorry

end right_triangle_legs_distances_l109_109539


namespace spinner_final_direction_north_l109_109415

def start_direction := "north"
def clockwise_revolutions := (7 : ℚ) / 2
def counterclockwise_revolutions := (5 : ℚ) / 2
def net_revolutions := clockwise_revolutions - counterclockwise_revolutions

theorem spinner_final_direction_north :
  net_revolutions = 1 → start_direction = "north" → 
  start_direction = "north" :=
by
  intro h1 h2
  -- Here you would prove that net_revolutions of 1 full cycle leads back to start
  exact h2 -- Skipping proof

end spinner_final_direction_north_l109_109415


namespace largest_4_digit_divisible_by_88_l109_109192

-- Define a predicate for a 4-digit number
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define a constant representing the number 88
def eighty_eight : ℕ := 88

-- State the main theorem
theorem largest_4_digit_divisible_by_88 : ∃ n : ℕ, is_four_digit n ∧ eighty_eight ∣ n ∧ ∀ m : ℕ, is_four_digit m ∧ eighty_eight ∣ m → m ≤ n :=
begin
  -- We assert that 9944 is the largest 4-digit number divisible by 88
  use 9944,
  split,
  -- Prove that 9944 is a four-digit number
  { split,
    { norm_num },
    { norm_num } },
  -- Prove that 9944 is divisible by 88
  { use 113,
    norm_num },
  -- Prove that 9944 is the largest such number
  { intros m Hm,
    cases Hm with Hm₁ Hm₂,
    have Hm₃ : m ≤ 9999 := by exact Hm₁.2,
    have Hm₄ : ∃ k, m = eighty_eight * k,
    { exact Hm₂ },
    cases Hm₄ with k Hk,
    have key : k ≤ 113 := by sorry,
    rw Hk,
    exact mul_le_mul_right' key eighty_eight }
end

end largest_4_digit_divisible_by_88_l109_109192


namespace sum_of_3_digit_permutations_l109_109166

open Finset

theorem sum_of_3_digit_permutations : 
  let digits := {2, 4, 5}
  let permutations := {245, 254, 425, 452, 524, 542}
  permutations.sum id = 2442 :=
by
  let digits : Finset ℕ := {2, 4, 5}
  let permutations : Finset ℕ := {245, 254, 425, 452, 524, 542}
  have : permutations = {245, 254, 425, 452, 524, 542},
  { sorry },
  rw this,
  simp,
  sorry

end sum_of_3_digit_permutations_l109_109166


namespace find_a_for_parallel_lines_l109_109398

-- Definitions from the problem conditions
def line_l1 (s : ℝ) : ℝ × ℝ := (2 * s + 1, s)
def line_l2 (a : ℝ) (t : ℝ) : ℝ × ℝ := (a * t, 2 * t - 1)

-- The proof statement goes here, but we will only state the problem
theorem find_a_for_parallel_lines (a : ℝ) :
  (∀ s t, 2 * (2 * s + 1 - at) = a * (s - 2 * t + 1)) → a = 4 :=
sorry

end find_a_for_parallel_lines_l109_109398


namespace smallest_argument_proof_l109_109581

noncomputable def smallest_argument : ℂ := 12 + 16 * complex.I

theorem smallest_argument_proof (p : ℂ) 
  (h : |p - 25 * complex.I| ≤ 15) : 
  (∃ q : ℂ, |q - 25 * complex.I| ≤ 15 ∧ ∀ r : ℂ, |r - 25 * complex.I| ≤ 15 → complex.arg q ≤ complex.arg r) → 
  p = 12 + 16 * complex.I :=
begin
  sorry
end

end smallest_argument_proof_l109_109581


namespace det_E_l109_109429

open Real Matrix

-- Definitions and conditions in the problem
noncomputable def D : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

noncomputable def R : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.sqrt 2 / 2, -Real.sqrt 2 / 2], ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]]

noncomputable def E : Matrix (Fin 2) (Fin 2) ℝ :=
  D ⬝ R

-- Statement to prove
theorem det_E : det E = 25 := by
  sorry

end det_E_l109_109429


namespace sqrt_add_sub_eq_frac_sqrt_mul_div_sub_eq_l109_109604

-- Problem 1
theorem sqrt_add_sub_eq_frac (x y z: ℝ) (h1: x = 27) (h2: y = 1 / 3) (h3: z = 12) :
  (√x + √y - √z = 4 * √3 / 3) :=
by 
  rw [h1, h2, h3]
  -- the detailed proof would go here
  sorry

-- Problem 2
theorem sqrt_mul_div_sub_eq (a b c d e: ℝ) (h1: a = 6) (h2: b = 2) (h3: c = 24) (h4: d = 3) (h5: e = 48) :
  (√a * √b + √c / √d - √e = 2 * √2 - 2 * √3) :=
by
  rw [h1, h2, h3, h4, h5]
  -- the detailed proof would go here
  sorry

end sqrt_add_sub_eq_frac_sqrt_mul_div_sub_eq_l109_109604


namespace distance_between_points_is_sqrt_89_l109_109913

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_points_is_sqrt_89 :
  distance 3 7 (-5) 2 = real.sqrt 89 := 
sorry

end distance_between_points_is_sqrt_89_l109_109913


namespace sum_to_2n_formula_additional_terms_l109_109845

theorem sum_to_2n_formula (n : ℕ) : (∑ i in Finset.range (2 * n + 1), (i + 1)) = n * (2 * n + 1) := by
  induction n with k ih
  case zero =>
    -- Base case: n = 0
    simp
  case succ =>
    -- Inductive step
    calc
      (∑ i in Finset.range (2 * (k + 1) + 1), (i + 1))
        = (∑ i in Finset.range (2 * k + 1), (i + 1)) + (2 * k + 1 + 1) + (2 * k + 1 + 2 + 1) :
          by sorry

theorem additional_terms (k : ℕ) : (2 * k + 1) + (2 * k + 2) = 4 * k + 3 := by
  calc
    (2 * k + 1) + (2 * k + 2) = 2 * k + 1 + 2 * k + 2 : by rw add_assoc
                          ... = 4 * k + 3 : by rw [two_mul, add_assoc, add_comm 1 2, add_assoc]

end sum_to_2n_formula_additional_terms_l109_109845


namespace x5_y5_z5_value_is_83_l109_109421

noncomputable def find_x5_y5_z5_value (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧ 
  (x^3 + y^3 + z^3 = 15) ∧
  (x^4 + y^4 + z^4 = 35) ∧
  (x^2 + y^2 + z^2 < 10) →
  x^5 + y^5 + z^5 = 83

theorem x5_y5_z5_value_is_83 (x y z : ℝ) :
  find_x5_y5_z5_value x y z :=
  sorry

end x5_y5_z5_value_is_83_l109_109421


namespace tangents_intersect_on_altitude_l109_109207

-- Define triangle ABC
variable {A B C : Point}

-- Define points P and Q on AC and BC respectively
variable {P Q : Point}

-- Define circle with AB as diameter
axiom circle_through_A_B (A B P Q : Point) : Circle

-- Define tangents at intersection points P and Q
axiom tangent_line_at_P (P : Point) : Tangent
axiom tangent_line_at_Q (Q : Point) : Tangent

-- Define altitude from point C to AB
axiom altitude_from_C_to_AB (C : Point) : Line

-- Define the intersection point H of the two tangents at P and Q
axiom tangents_intersect_at_H (tangent_at_P tangent_at_Q : Tangent) : Point

-- The theorem to prove
theorem tangents_intersect_on_altitude {A B C P Q H : Point}
  (h_circle : circle_through_A_B A B P Q)
  (h_tangent_P : tangent_line_at_P P)
  (h_tangent_Q : tangent_line_at_Q Q)
  (h_altitude : altitude_from_C_to_AB C)
  (h_intersection : tangents_intersect_at_H (h_tangent_P) (h_tangent_Q) = H) :
  H ∈ h_altitude := sorry

end tangents_intersect_on_altitude_l109_109207


namespace translate_cosine_l109_109500

theorem translate_cosine:
  let f : ℝ → ℝ := λ x, cos (2 * x),
      g : ℝ → ℝ := λ x, f (x + π / 6),
      h : ℝ → ℝ := λ x, g x + 1
  in ∀ x, h x = cos (2 * x + π / 3) + 1 :=
by {
  sorry
}

end translate_cosine_l109_109500


namespace odd_n_iff_exists_non_integer_rationals_l109_109301

theorem odd_n_iff_exists_non_integer_rationals
  (n : ℕ) (h_pos : 0 < n) :
  (∃ (a b : ℚ), a > 0 ∧ b > 0 ∧ a.denom ≠ 1 ∧ b.denom ≠ 1 ∧ (a + b).denom = 1 ∧ (a^n + b^n).denom = 1) ↔ n % 2 = 1 := 
sorry

end odd_n_iff_exists_non_integer_rationals_l109_109301


namespace max_consecutive_sum_l109_109587

theorem max_consecutive_sum :
  ∀ (a : Fin 40 → ℕ), (∀ i : Fin (40-7), (∑ j in Finset.range 8, a (i + j)) ≥ 164) :=
by sorry

end max_consecutive_sum_l109_109587


namespace smallest_prime_with_digit_sum_25_l109_109915

-- Definitions used in Lean statement:
-- 1. Prime predicate based on primality check.
-- 2. Digit sum function.

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement to prove that the smallest prime whose digits sum to 25 is 1699.

theorem smallest_prime_with_digit_sum_25 : ∃ n : ℕ, is_prime n ∧ digit_sum n = 25 ∧ n = 1699 :=
by
  sorry

end smallest_prime_with_digit_sum_25_l109_109915


namespace BrianchonsTheorem_l109_109457

theorem BrianchonsTheorem (A B C D E F : Point) (circle : Circle) (hexagon : is_inscribed_in_circle (A B C D E F) circle) : 
  ¬ ∃ P : Point, (line_through A D).contains P ∧ (line_through B E).contains P ∧ (line_through C F).contains P :=
sorry

end BrianchonsTheorem_l109_109457


namespace triangle_perimeter_l109_109041

theorem triangle_perimeter (a b c : ℝ) (largest_angle_eq_120 : ∠A = 120) 
  (h1 : a - b = 4) (h2 : a + c = 2 * b) : 
  a + b + c = 30 :=
sorry

end triangle_perimeter_l109_109041


namespace coordinate_and_distance_problem_l109_109409

theorem coordinate_and_distance_problem
    (x y : ℝ)
    (α : ℝ)
    (ρ θ : ℝ)
    (cos_sq_sin_sq_identity : ∀ α : ℝ, cos α ^ 2 + sin α ^ 2 = 1)
    (line_l_polar : ∀ ρ θ : ℝ, ρ * cos θ - 2 * ρ * sin θ - 4 = 0)
    (curve_C_parametric : ParametricCartesian (x y : ℝ) (C : Set (ℝ × ℝ)) := ∃ α : ℝ, (x = 2 * cos α ∧ y = 3 * sin α)): 
    (∀ α, ∃ (x y : ℝ), (x = 2 * cos α ∧ y = 3 * sin α) →
        (x^2 / 4) + (y^2 / 9) = 1)
    ∧ ∀ (x y : ℝ), (x = ρ * cos θ ∧ y = ρ * sin θ ∧ ρ * cos θ - 2 * ρ * sin θ - 4 = 0 →
        x - 2 * y - 4 = 0)
    ∧ (∀ x y, line_l_cartesian_passing_point (4, 0, x, y : ℝ) →
        let AB_AD_product := -- some intermediary computations yielding 27 / 2
        (|AB| * |AD|) = 27 / 2)
    sorry

end coordinate_and_distance_problem_l109_109409


namespace line_equation_l109_109863

-- Given conditions
def param_x (t : ℝ) : ℝ := 3 * t + 6
def param_y (t : ℝ) : ℝ := 5 * t - 7

-- Proof problem: for any real t, the parameterized line can be described by the equation y = 5x/3 - 17.
theorem line_equation (t : ℝ) : ∃ (m b : ℝ), (∃ t : ℝ, param_y t = m * (param_x t) + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  exists 5 / 3
  exists -17
  sorry

end line_equation_l109_109863


namespace num_clerks_l109_109391

def manager_daily_salary := 5
def clerk_daily_salary := 2
def num_managers := 2
def total_daily_salary := 16

theorem num_clerks (c : ℕ) (h1 : num_managers * manager_daily_salary + c * clerk_daily_salary = total_daily_salary) : c = 3 :=
by 
  sorry

end num_clerks_l109_109391


namespace number_of_subsets_l109_109877

theorem number_of_subsets {α : Type} (a b : α) : 
  set.card {M : set α | M ⊆ {a, b}} = 4 :=
sorry

end number_of_subsets_l109_109877


namespace circle_area_l109_109151

theorem circle_area (A B : (ℝ × ℝ))
                   (hA : A = (7, 14))
                   (hB : B = (13, 12))
                   (intersects_on_x_axis : ∃ P : ℝ × ℝ, P.2 = 0 ∧ tangent_to_circle ω A P ∧ tangent_to_circle ω B P) :
                   ∃ (r : ℝ), area_of_circle_with_radius r = 196 * π :=
begin
  -- conditions
  have h1: A = (7, 14), from hA,
  have h2: B = (13, 12), from hB,
  sorry
end

end circle_area_l109_109151


namespace count_special_integers_l109_109714

-- Define a two-digit integer and its proper constraints.
def is_valid_integer (n : ℕ) : Prop :=
  10 < n ∧ n < 100 ∧ ∃ (x y : ℕ), n = 10*x + y ∧ x + 9 + y = 10*y + x

-- The theorem to be proved: there are 8 such integers.
theorem count_special_integers : 
  (finset.filter is_valid_integer (finset.range 100)).card = 8 := 
sorry

end count_special_integers_l109_109714


namespace distance_between_girls_l109_109383

-- Define conditions as variables
variables (speed_girl1 speed_girl2 time : ℝ)

-- Define the speeds and time based on problem conditions
def speed_girl1 := 5 -- 5 km/hr
def speed_girl2 := 10 -- 10 km/hr
def time := 5 -- 5 hours

-- Calculate the distances each girl walks
def distance_girl1 := speed_girl1 * time
def distance_girl2 := speed_girl2 * time

-- Calculate the total distance when walking in opposite directions
def total_distance := distance_girl1 + distance_girl2

-- State the theorem that proves the total distance
theorem distance_between_girls : total_distance = 75 := by
  sorry

end distance_between_girls_l109_109383


namespace find_other_denomination_l109_109158

theorem find_other_denomination
  (total_spent : ℕ)
  (twenty_bill_value : ℕ) (other_denomination_value : ℕ)
  (twenty_bill_count : ℕ) (other_bill_count : ℕ)
  (h1 : total_spent = 80)
  (h2 : twenty_bill_value = 20)
  (h3 : other_bill_count = 2)
  (h4 : twenty_bill_count = other_bill_count + 1)
  (h5 : total_spent = twenty_bill_value * twenty_bill_count + other_denomination_value * other_bill_count) : 
  other_denomination_value = 10 :=
by
  sorry

end find_other_denomination_l109_109158


namespace correct_conclusions_count_l109_109595

def is_perpendicular (A B : Type) : Prop := sorry

variables {Line Plane : Type}
variables (l1 l2 l : Line) (P Q : Plane)

axiom perp_line_same_line_is_not_parallel : (is_perpendicular l1 l) ∧ (is_perpendicular l2 l) → ¬ (parallel l1 l2)
axiom perp_line_same_plane_is_parallel : (is_perpendicular l1 P) ∧ (is_perpendicular l2 P) → (parallel l1 l2)
axiom perp_plane_same_line_is_parallel : (is_perpendicular P l) ∧ (is_perpendicular Q l) → (parallel P Q)
axiom perp_plane_same_plane_is_not_parallel : (is_perpendicular P Q) → ¬ (parallel P Q ∧ P ≠ Q)

theorem correct_conclusions_count : 2 = 
  if (perp_line_same_line_is_not_parallel l1 l2 l ∧
    perp_line_same_plane_is_parallel l1 l2 P ∧
    perp_plane_same_line_is_parallel P Q l ∧
    perp_plane_same_plane_is_not_parallel P Q) then 2 else sorry :=
sorry

end correct_conclusions_count_l109_109595


namespace math_exam_high_scores_l109_109744

open Real

variables (n : ℝ) (mu : ℝ) (sigma : ℝ) (students : ℝ)

-- Conditions from the problem
def total_students : ℝ := 1000
def exam_score_distribution : NormalDist :=
  { mean := 90, variance := sigma^2, variance_pos := sorry } -- assume sigma > 0
def score_range_70_110 : ℝ := 0.6 * total_students
def score_at_least_110 := 200

-- The theorem to be proven
theorem math_exam_high_scores:
  students = total_students →
  score_range_70_110 = 0.6 * students →
  ∃ score_above_110, score_above_110 = 200 := 
sorry

end math_exam_high_scores_l109_109744


namespace ellipse_problem_l109_109407

noncomputable theory

open_locale pointwise

-- Define the conditions and the conclusion
def ellipse_equation (a b x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def on_circle (x y : ℝ) := x^2 + y^2 = 1

def slope_product (x1 y1 x2 y2 : ℝ) :=
  (y1 * y2) / (x1 * x2)

def point_condition (x1 y1 x2 y2 θ : ℝ) (M : ℝ × ℝ) :=
  let Mx := fst M in let My := snd M in
  Mx = x1 * cos θ + x2 * sin θ ∧ My = y1 * cos θ + y2 * sin θ

-- The overall statement
theorem ellipse_problem :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (let e := (real.sqrt (a^2 - b^2)) / a in e = (real.sqrt 2) / 2) ∧
  (∃ (c : ℝ), c = 1 ∧ on_circle c 0 ∧
  ellipse_equation a b c 0 ∧ a = real.sqrt 2 ∧ b = 1 ∧
  (∀ (x1 y1 x2 y2 θ : ℝ), 
    (point_condition x1 y1 x2 y2 θ (x1 * cos θ + x2 * sin θ, y1 * cos θ + y2 * sin θ)) →
    (slope_product x1 y1 x2 y2) = -1 / 2 ∧ 
    (x1^2 + x2^2 = 2 ∧ y1^2 + y2^2 = 1 → 
    x1^2 + y1^2 + x2^2 + y2^2 = 3))) :=
sorry

end ellipse_problem_l109_109407


namespace geometric_locus_l109_109128

variable {x y a : ℝ}

def A : ℝ × ℝ := (x, y)
def B : ℝ × ℝ := (y, -x)
def C : ℝ × ℝ := (-x, -y)
def D : ℝ × ℝ := (-y, x)

def l (y : ℝ) : Prop := y = a

def P : ℝ × ℝ := (-y, a)

def Q : ℝ × ℝ := ((x + y) / 2, (y - x) / 2)

def M : ℝ × ℝ := ((-y + (x + y) / 2) / 2, (a + (y - x) / 2) / 2)

theorem geometric_locus (t : ℝ) : M = (t, -t + a / 2) :=
by
  sorry

end geometric_locus_l109_109128


namespace cab_speed_ratio_l109_109555

variable (S_u S_c : ℝ)

theorem cab_speed_ratio (h1 : ∃ S_u S_c : ℝ, S_u * 25 = S_c * 30) : S_c / S_u = 5 / 6 :=
by
  sorry

end cab_speed_ratio_l109_109555


namespace periodic_sequence_criteria_l109_109669

theorem periodic_sequence_criteria (K : ℕ) (hK_pos : K > 0) :
  ( ∀ p, ∃ m, ∀ n ≥ m, (Nat.choose (2 * n) n) % K = (Nat.choose (2 * (n + p)) (n + p)) % K ) →
  (K = 1 ∨ K = 2) :=
begin
  sorry,
end

end periodic_sequence_criteria_l109_109669


namespace cos2_minus_sin2_eq_neg_three_fifths_l109_109694

theorem cos2_minus_sin2_eq_neg_three_fifths (θ : Real) 
  (h_vertex : θ.vertex = (0, 0))
  (h_initial_side : θ.initial_side = {p : Real × Real | p.2 = 0 ∧ p.1 ≥ 0})
  (h_terminal_side : θ.terminal_side = {p : Real × Real | p.2 = 2 * p.1}) :
  Real.cos^2 θ - Real.sin^2 θ = -3 / 5 :=
by
  sorry

end cos2_minus_sin2_eq_neg_three_fifths_l109_109694


namespace sum_even_integers_less_than_73_l109_109522

theorem sum_even_integers_less_than_73 : 
  (∑ k in finset.range 37, 2 * k) = 1332 := 
by
  sorry

end sum_even_integers_less_than_73_l109_109522


namespace number_of_keepers_l109_109388

theorem number_of_keepers (k : ℕ) :
  let hens := 50
  let goats := 45
  let camels := 8
  let keepers := k
  let heads := hens + goats + camels + keepers
  let feet := 2 * hens + 4 * goats + 4 * camels + 2 * keepers
  feet = heads + 224 → keepers = 15 :=
by
  intros hens goats camels keepers heads feet h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_keepers_l109_109388


namespace count_ordered_triples_satisfying_equation_l109_109362

theorem count_ordered_triples_satisfying_equation :
  { (x, y, z) : ℕ × ℕ × ℕ // 0 < x ∧ 0 < y ∧ 0 < z ∧ (x ^ y) ^ z = 64 }.card = 9 := 
by sorry

end count_ordered_triples_satisfying_equation_l109_109362


namespace ratio_of_height_to_width_l109_109478

-- Define the constants and variables
variable {V : ℝ} {W H L x : ℝ}
constant volume_eq : V = 86436
constant width_approx : W = 7
constant height_ratio : H = x * W
constant length_ratio : L = 7 * H
constant volume_formula : V = W * H * L

-- Define the theorem to be proven
theorem ratio_of_height_to_width :
  height_ratio ∧ length_ratio ∧ volume_eq ∧ width_approx ∧ volume_formula →
  x = 6 :=
by
  sorry

end ratio_of_height_to_width_l109_109478


namespace optimal_play_second_player_wins_l109_109449

theorem optimal_play_second_player_wins : 
  (∃ (board : List ℕ), board = List.range' 1 (1000 + 1) ∧ 
   (∀ (player1_turns player2_turns : ℕ), 
    (player1_turns + player2_turns = 998) ∧ 
     (∀ n, n ∈ board → n < 1001) ∧ 
     (∀ x y, x ∈ board ∧ y ∈ board ∧ ¬ (x = y) → 
      (((x + y) % 3 = 0 → player1_turns < player2_turns) ∧ 
       ((x + y) % 3 ≠ 0 → player2_turns < player1_turns)))) → 
  ∃ optimal_strategy, optimal_strategy (board) → 
  (∃ (x y : ℕ), x ∈ board ∧ y ∈ board ∧ 
   ((x + y) % 3 ≠ 0))) :=
sorry

end optimal_play_second_player_wins_l109_109449


namespace quadratic_roots_product_l109_109243

theorem quadratic_roots_product (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, 4 * x^2 + 8 * x - 12 = a * x^2 + b * x + c) → 
  c = -12 → a = 4 → (α β : ℝ), is_root (λ x : ℝ, 4 * x^2 + 8 * x - 12) α ∧ is_root (λ x : ℝ, 4 * x^2 + 8 * x - 12) β → α * β = -3 := 
by 
  sorry

end quadratic_roots_product_l109_109243


namespace julie_reimbursement_l109_109093

def lollipops_shared_reimbursement (total_lollipops : ℕ) (total_cost : ℝ) (discount_rate : ℝ) (shared_fraction : ℚ) :=
  let cost_per_lollipop := total_cost / total_lollipops
  let discounted_cost_per_lollipop := cost_per_lollipop - (cost_per_lollipop * discount_rate)
  let shared_lollipops := total_lollipops * shared_fraction.toNat
  let reimbursement := shared_lollipops * discounted_cost_per_lollipop
  (reimbursement * 100).toNat

theorem julie_reimbursement :
  lollipops_shared_reimbursement 12 3 0.20 (1/4 : ℚ) = 60 :=
by
  sorry

end julie_reimbursement_l109_109093


namespace min_value_exprB_four_min_value_exprC_four_l109_109173

noncomputable def exprB (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def exprC (x : ℝ) : ℝ := 1 / (Real.sin x)^2 + 1 / (Real.cos x)^2

theorem min_value_exprB_four : ∃ x : ℝ, exprB x = 4 := sorry

theorem min_value_exprC_four : ∃ x : ℝ, exprC x = 4 := sorry

end min_value_exprB_four_min_value_exprC_four_l109_109173


namespace coefficient_of_third_term_in_binomial_expansion_l109_109109

theorem coefficient_of_third_term_in_binomial_expansion:
  let a := 2
  let b := λ x, x
  let n := 3
  let k := 2
  -- Using binomial coefficient and binomial expansion
  -- The term for the k-th coefficient in expanding (a + b x)^n
  have term := @nat.choose n k * (a ^ (n - k)) * (b(k) ^ k)
  -- Simplify to get the third term's coefficient
  shows term = 6
  by
    sorry

end coefficient_of_third_term_in_binomial_expansion_l109_109109


namespace total_cost_second_set_l109_109941

variable (A V : ℝ)

-- Condition declarations
axiom cost_video_cassette : V = 300
axiom cost_second_set : 7 * A + 3 * V = 1110

-- Proof goal
theorem total_cost_second_set :
  7 * A + 3 * V = 1110 :=
by
  sorry

end total_cost_second_set_l109_109941


namespace fifth_term_of_sequence_is_31_l109_109321

namespace SequenceProof

def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + 1

theorem fifth_term_of_sequence_is_31 :
  ∃ a : ℕ → ℕ, sequence a ∧ a 5 = 31 :=
by
  sorry

end SequenceProof

end fifth_term_of_sequence_is_31_l109_109321


namespace average_between_12_and_150_divisible_by_7_l109_109278

noncomputable def avg_divisible_by_seven (a b : ℕ) (h1 : 12 ≤ a) (h2 : a ≤ 150) (h3 : 12 ≤ b) (h4 : b ≤ 150) (h5 : ∀ x, x = a → a % 7 = 0) (h6 : ∀ x, x = b → b % 7 = 0) : ℝ :=
( a + b ) / 2

theorem average_between_12_and_150_divisible_by_7 : avg_divisible_by_seven 14 147 12 150 12 150 (by norm_num) (by norm_num) = 80.5 := 
sorry

end average_between_12_and_150_divisible_by_7_l109_109278


namespace boat_speed_in_still_water_l109_109486

theorem boat_speed_in_still_water:
  ∃ x : ℚ, (∀ rate_of_current distance time, 
    rate_of_current = 5 → 
    distance = 5 → 
    time = 12/60 → 
    distance = (x + rate_of_current) * time) → 
  x = 20 :=
begin
  use 20,
  intros rate_of_current distance time h,
  specialize h 5 5 (12/60),
  simp [h],
  sorry
end

end boat_speed_in_still_water_l109_109486


namespace cindy_marbles_problem_l109_109830

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end cindy_marbles_problem_l109_109830


namespace car_not_sold_probability_l109_109727

theorem car_not_sold_probability (a b : ℕ) (h : a = 5) (k : b = 6) : (b : ℚ) / (a + b : ℚ) = 6 / 11 :=
  by
    rw [h, k]
    norm_num

end car_not_sold_probability_l109_109727


namespace number_of_efficient_paths_2016_l109_109608

noncomputable def number_of_efficient_paths (n : ℕ) : ℕ :=
  Nat.choose (2 * (n - 1)) (n - 1) 

theorem number_of_efficient_paths_2016 :
  number_of_efficient_paths 2016 = Nat.choose 4030 2015 :=
by
  rw [number_of_efficient_paths]
  rw [Nat.sub_eq_of_eq_add]
  sorry

end number_of_efficient_paths_2016_l109_109608


namespace f_inequality_solution_set_l109_109613

noncomputable def f : ℝ → ℝ :=
sorry -- assume the function definition details are provided

axiom f_domain : ∀ x, x > 0 → 0 < f x ∧ f (2 : ℝ) = 4
axiom f_derivative : ∀ x, x > 0 → f x > x * (deriv f x)

theorem f_inequality_solution_set :
  {x : ℝ | 0 < x ∧ f(x) - 2*x > 0} = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end f_inequality_solution_set_l109_109613


namespace num_divisors_of_m_cubed_l109_109566

theorem num_divisors_of_m_cubed (m : ℕ) (h : ∃ p : ℕ, Nat.Prime p ∧ m = p ^ 4) :
    Nat.totient (m ^ 3) = 13 := 
sorry

end num_divisors_of_m_cubed_l109_109566


namespace sum_of_solutions_l109_109762

theorem sum_of_solutions :
  ∀ (x y : ℤ), (|x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|) →
  ({ (2, 8), (4, 10), (10, 4), (10, 4) }).sum (λ p, p.1 + p.2) = 52 :=
by
  sorry

end sum_of_solutions_l109_109762


namespace triangle_condition_l109_109967

variable {A B C : Type} [AddCommGroup A] [Module ℝ A] [InnerProductSpace ℝ A]

variables (a b c : ℝ)
variables (cos_A : ℝ)

axiom law_of_cosines (a b c : ℝ) : a^2 = b^2 + c^2 - 2 * b * c * cos_A

theorem triangle_condition (a b c : ℝ) (cos_A : ℝ) (h : (b^2 + c^2) / (2 * b * c * cos_A) = b^2 + c^2 - 2 * b * c * cos_A) : 
  a = b ∨ a = c :=
begin
  sorry
end

end triangle_condition_l109_109967


namespace num_ways_to_select_team_l109_109824

theorem num_ways_to_select_team (boys girls : ℕ) (team_size boys_selected girls_selected : ℕ) :
  boys = 7 → girls = 10 → team_size = 6 → boys_selected = 4 → girls_selected = 2 →
  (Nat.choose boys boys_selected) * (Nat.choose girls girls_selected) = 1575 :=
by
  intros hb hg ht hbs hgs
  rw [hb, hg, ht, hbs, hgs]
  have hb_comb : Nat.choose 7 4 = 35 := by sorry
  have hg_comb : Nat.choose 10 2 = 45 := by sorry
  rw [hb_comb, hg_comb]
  norm_num

end num_ways_to_select_team_l109_109824


namespace Gandalf_reachability_l109_109137

theorem Gandalf_reachability (n : ℕ) (h : n ≥ 1) :
  ∃ (m : ℕ), m = 1 :=
sorry

end Gandalf_reachability_l109_109137


namespace proof_ineq_l109_109317

variables {x : ℕ → ℝ} (n : ℕ)

def condition1 := x 1 ^ 2 = 1

def lhs (n : ℕ) : ℝ :=
  ∑ i in (finset.divisors n), ∑ j in (finset.divisors n), x i * x j / nat.lcm i j

def rhs (n : ℕ) : ℝ :=
  ∏ p in (finset.filter nat.prime (finset.divisors n)), (1 - (1 : ℝ) / p)

theorem proof_ineq (h1 : condition1) (hn : n ≥ 2) :
  lhs x n ≥ rhs n :=
sorry

end proof_ineq_l109_109317


namespace appropriate_weight_design_l109_109968

def weight_design (w_l w_s w_r w_w : ℕ) : Prop :=
  w_l > w_s ∧ w_l > w_w ∧ w_w > w_r ∧ w_s = w_w

theorem appropriate_weight_design :
  weight_design 5 2 1 2 :=
by {
  sorry -- skipped proof
}

end appropriate_weight_design_l109_109968


namespace sin_function_monotone_l109_109921

def f (x : Real) : Real := sin (2 * x + π / 6)
def interval (t : Real) (h : 0 < t ∧ t < π / 6) : Set Real := {x | -t < x ∧ x < t}

theorem sin_function_monotone (t : Real) (h : 0 < t ∧ t < π / 6) : 
  ∀ x ∈ interval t h, is_monotone_on (f) (interval t h) :=
sorry

end sin_function_monotone_l109_109921


namespace g_neg501_l109_109857

noncomputable def g : ℝ → ℝ := sorry

axiom g_eq (x y : ℝ) : g (x * y) + 2 * x = x * g y + g x

axiom g_neg1 : g (-1) = 7

theorem g_neg501 : g (-501) = 507 :=
by
  sorry

end g_neg501_l109_109857


namespace mints_ratio_l109_109553

theorem mints_ratio (n : ℕ) (green_mints red_mints : ℕ) (h1 : green_mints + red_mints = n) (h2 : green_mints = 3 * (n / 4)) : green_mints / red_mints = 3 :=
by
  sorry

end mints_ratio_l109_109553


namespace mistaken_divisor_l109_109735

theorem mistaken_divisor (x : ℕ) (h : 49 * x = 28 * 21) : x = 12 :=
sorry

end mistaken_divisor_l109_109735


namespace police_officers_on_duty_l109_109828

theorem police_officers_on_duty (F : ℕ) (hF : F = 400) (h1 : 0.18 * F = d_female) (h2 : 2 * d_female = d_total) : d_total = 144 :=
by
  sorry

end police_officers_on_duty_l109_109828


namespace solution_inequality_l109_109887

-- Define the condition as a predicate
def inequality_condition (x : ℝ) : Prop :=
  (x - 1) * (x + 1) < 0

-- State the theorem that we need to prove
theorem solution_inequality : ∀ x : ℝ, inequality_condition x → (-1 < x ∧ x < 1) :=
by
  intro x hx
  sorry

end solution_inequality_l109_109887


namespace calculation_is_correct_l109_109597

theorem calculation_is_correct : 450 / (6 * 5 - 10 / 2) = 18 :=
by {
  -- Let me provide an outline for solving this problem
  -- (6 * 5 - 10 / 2) must be determined first
  -- After that substituted into the fraction
  sorry
}

end calculation_is_correct_l109_109597


namespace solutions_to_cubic_equation_l109_109096

noncomputable def solve_cubic_equation : set ℂ :=
  { x : ℂ | (x^3 + 3*x^2*real.sqrt 2 + 6*x + 2*real.sqrt 2) + (x + real.sqrt 2) = 0 }

theorem solutions_to_cubic_equation : solve_cubic_equation = {-real.sqrt 2, -real.sqrt 2 + complex.I, -real.sqrt 2 - complex.I} :=
by
  sorry

end solutions_to_cubic_equation_l109_109096


namespace range_sin_diff_abs_sin_l109_109620

theorem range_sin_diff_abs_sin : Set.range (λ x : ℝ, Real.sin x - Real.sin (|x|)) = Set.Icc (-2 : ℝ) 2 := 
sorry

end range_sin_diff_abs_sin_l109_109620


namespace segments_nonfunctional_total_l109_109028

noncomputable def first_position_nonfunctional : Nat := 5
noncomputable def second_position_nonfunctional : Nat := 2
noncomputable def third_position_nonfunctional : Nat := 4
noncomputable def fourth_position_nonfunctional : Nat := 2

theorem segments_nonfunctional_total : first_position_nonfunctional + second_position_nonfunctional + third_position_nonfunctional + fourth_position_nonfunctional = 13 :=
by
  calc
  first_position_nonfunctional + second_position_nonfunctional + third_position_nonfunctional + fourth_position_nonfunctional
      = 5 + 2 + 4 + 2 : by rfl
  ... = 13 : by rfl

end segments_nonfunctional_total_l109_109028


namespace value_of_a_plus_b_l109_109313

variable (a b : ℝ)
variable (h1 : |a| = 5)
variable (h2 : |b| = 2)
variable (h3 : a < 0)
variable (h4 : b > 0)

theorem value_of_a_plus_b : a + b = -3 :=
by
  sorry

end value_of_a_plus_b_l109_109313


namespace part2_x_values_part3_no_real_x_for_2000_l109_109560

noncomputable def average_daily_sales (x : ℝ) : ℝ :=
  24 + 4 * x

noncomputable def profit_per_unit (x : ℝ) : ℝ :=
  60 - 5 * x

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  (60 - 5 * x) * (24 + 4 * x)

theorem part2_x_values : 
  {x : ℝ | daily_sales_profit x = 1540} = {1, 5} := sorry

theorem part3_no_real_x_for_2000 : 
  ∀ x : ℝ, daily_sales_profit x ≠ 2000 := sorry

end part2_x_values_part3_no_real_x_for_2000_l109_109560


namespace quadratic_root_and_coefficient_l109_109343

theorem quadratic_root_and_coefficient (k : ℝ) :
  (∃ x : ℝ, 5 * x^2 + k * x - 6 = 0 ∧ x = 2) →
  (∃ x₁ : ℝ, (5 * x₁^2 + k * x₁ - 6 = 0 ∧ x₁ ≠ 2) ∧ x₁ = -3/5 ∧ k = -7) :=
by
  sorry

end quadratic_root_and_coefficient_l109_109343


namespace temperature_reaches_90_at_17_l109_109387

def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem temperature_reaches_90_at_17 :
  ∃ t : ℝ, temperature t = 90 ∧ t = 17 :=
by
  exists 17
  dsimp [temperature]
  norm_num
  sorry

end temperature_reaches_90_at_17_l109_109387


namespace train_speed_kmph_l109_109226

/-- A train covers a distance of 22.5 km in 15 minutes. -/
def train_distance : ℝ := 22.5

/-- The time taken by the train in hours. -/
def train_time_hours : ℝ := 15 / 60

/-- The speed of the train is calculated by dividing the distance by the time. -/
theorem train_speed_kmph : train_distance / train_time_hours = 90 := by
  sorry

end train_speed_kmph_l109_109226


namespace johns_current_income_l109_109563

theorem johns_current_income
  (prev_income : ℝ := 1000000)
  (prev_tax_rate : ℝ := 0.20)
  (new_tax_rate : ℝ := 0.30)
  (extra_taxes_paid : ℝ := 250000) :
  ∃ (X : ℝ), 0.30 * X - 0.20 * prev_income = extra_taxes_paid ∧ X = 1500000 :=
by
  use 1500000
  -- Proof would come here
  sorry

end johns_current_income_l109_109563


namespace tan_sum_pi_div_12_l109_109461

theorem tan_sum_pi_div_12 (h1 : Real.tan (Real.pi / 12) ≠ 0) (h2 : Real.tan (5 * Real.pi / 12) ≠ 0) :
  Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 := 
by
  sorry

end tan_sum_pi_div_12_l109_109461


namespace natural_sum_representation_l109_109796

-- Definitions
def is_strictly_decreasing (a : ℕ → ℕ) := ∀ n, a (n + 1) < a n
def S (a : ℕ → ℕ) (k : ℕ) : ℕ := if k = 1 then 0 else finset.sum (finset.range (k - 1)) a

-- Theorem statement
theorem natural_sum_representation (a : ℕ → ℕ) 
  (h_dec : is_strictly_decreasing a) :
  (∀ (k : ℕ), a k ≤ S a k + 1) ↔ 
  ∀ n, ∃ s : finset ℕ, s.to_finset.sum a = n :=
sorry

end natural_sum_representation_l109_109796


namespace area_of_trapezoid_PQRS_eq_8256_div_25_l109_109410

-- Define basic properties and variables used in the problem
variable (PQRS : Type) [Trapezoid PQRS]
variable (P Q R S : PQRS)
variable (SQ RH SH : ℝ)
variable (angle_PQR angle_QRS : ℝ)
variable (distance_R_to_QS : ℝ)

-- State given values as conditions
axiom angle_PQR_90 : angle_PQR = 90
axiom angle_QRS_lt_90 : angle_QRS < 90
axiom angle_bisector_S : ∠QRS = ∠PSQ
axiom SQ_length : SQ = 24
axiom distance_R_to_QS_16 : distance_R_to_QS = 16

-- Prove that the area of trapezoid PQRS is 8256/25
theorem area_of_trapezoid_PQRS_eq_8256_div_25 :
  TrapezoidArea PQRS P Q R S = 8256 / 25 :=
by
  -- Proof goes here
  sorry

end area_of_trapezoid_PQRS_eq_8256_div_25_l109_109410


namespace polygon_eq_quadrilateral_l109_109382

theorem polygon_eq_quadrilateral (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 := 
sorry

end polygon_eq_quadrilateral_l109_109382


namespace range_of_f_pos_l109_109068

-- Define f as a odd function and define the conditions
variable {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x ) (h_f_neg1_eq_0 : f (-1) = 0)
         (h_inequality : ∀ x > 0, x * (deriv f x) - f x > 0)

-- Theorem that proves the range of x for which f(x) > 0
theorem range_of_f_pos : {x : ℝ | f x > 0} = set.Ioo (-1) 0 ∪ set.Ioi 1 :=
by
  sorry

end range_of_f_pos_l109_109068


namespace max_marks_l109_109454

theorem max_marks (M : ℕ) (h_pass : 55 / 100 * M = 510) : M = 928 :=
sorry

end max_marks_l109_109454


namespace pentagon_area_proof_l109_109963

noncomputable def area_pentagon_PTRSQ (PT TR : ℝ) (PT_perpendicular_TR : PT ≠ 0 ∧ TR ≠ 0 ∧ PT^2 + TR^2 = (real.sqrt (PT^2 + TR^2))^2) : ℝ :=
  let PR := real.sqrt (PT^2 + TR^2) in
  let side := PR / real.sqrt 2 in
  let area_square := side^2 in
  let area_triangle := 0.5 * PT * TR in
  area_square - area_triangle

theorem pentagon_area_proof : area_pentagon_PTRSQ 12 9 ⟨by norm_num, by norm_num, by norm_num⟩ = 58.5 := 
by 
  sorry

end pentagon_area_proof_l109_109963


namespace maximum_value_problem_l109_109939

theorem maximum_value_problem (x : ℝ) (h : 0 < x ∧ x < 4/3) : ∃ M, M = (4 / 3) ∧ ∀ y, 0 < y ∧ y < 4/3 → x * (4 - 3 * x) ≤ M :=
sorry

end maximum_value_problem_l109_109939


namespace number_of_fractions_is_4_l109_109233

noncomputable def expressions := [
  (x + y) / 2,
  -(3 * b) / a,
  1 / (x + y),
  (3 + y) / Real.pi
]

def is_fraction (e : Real) : Prop :=
  ∃ (num denom : Real), e = num / denom

def count_fractions (lst : List Real) : Nat :=
  lst.countp is_fraction

theorem number_of_fractions_is_4 (x y a b : Real) :
  count_fractions expressions = 4 :=
by
  sorry

end number_of_fractions_is_4_l109_109233


namespace azalea_wool_price_l109_109228

noncomputable def sheep_count : ℕ := 200
noncomputable def wool_per_sheep : ℕ := 10
noncomputable def shearing_cost : ℝ := 2000
noncomputable def profit : ℝ := 38000

-- Defining total wool and total revenue based on these definitions
noncomputable def total_wool : ℕ := sheep_count * wool_per_sheep
noncomputable def total_revenue : ℝ := profit + shearing_cost
noncomputable def price_per_pound : ℝ := total_revenue / total_wool

-- Problem statement: Proving that the price per pound of wool is equal to $20
theorem azalea_wool_price :
  price_per_pound = 20 := 
sorry

end azalea_wool_price_l109_109228


namespace password_probability_l109_109594

theorem password_probability : 
  (5/10) * (51/52) * (9/10) = 459 / 1040 := by
  sorry

end password_probability_l109_109594


namespace find_initial_snatch_weight_l109_109760

def initial_snatch_weight (initial_clean_jerk: ℝ) (new_total: ℝ) (multiplier: ℝ) (new_clean_jerk: ℝ) (S: ℝ) : Prop :=
    initial_clean_jerk * 2 = new_clean_jerk ∧ 
    new_total = new_clean_jerk + multiplier * S ∧ 
    new_total - new_clean_jerk = multiplier * S

theorem find_initial_snatch_weight : initial_snatch_weight 80 250 1.8 160 50 :=
by
  unfold initial_snatch_weight
  split
  simp
  split
  simp
  sorry

end find_initial_snatch_weight_l109_109760


namespace tangent_circle_ray_intersect_l109_109324

-- Define the geometric setup and conditions.
variable {O A B C E K : Point}
variable (tan_to_sides : Tangent(O, A) ∧ Tangent(O, B))
variable (ray_AC_parallel_OB : Parallel(Ray(A, C), Segment(O, B)))
variable (seg_OC_intersects_E : Intersects(Segment(O, C), Circle(E)))
variable (lines_AE_OB_intersect_K : Intersect(Line(A, E), Line(O, B)) = K)

-- Define the goal which is to prove OK = KB given the conditions.
theorem tangent_circle_ray_intersect
  (tan_to_sides : Tangent(O, A) ∧ Tangent(O, B))
  (ray_AC_parallel_OB : Parallel(Ray(A, C), Segment(O, B)))
  (seg_OC_intersects_E : Intersects(Segment(O, C), Circle(E)))
  (lines_AE_OB_intersect_K : Intersect(Line(A, E), Line(O, B)) = K) :
  dist(O, K) = dist(K, B) :=
by
  sorry

end tangent_circle_ray_intersect_l109_109324


namespace floor_abs_S_eq_503_l109_109103

theorem floor_abs_S_eq_503 (x : Fin 1004 → ℝ)
  (h : ∀ a : Fin 1004, x a + (a + 1 : ℕ) = ∑ i, x i + 1005) :
  (⌊|∑ i, x i|⌋) = 503 := by
  sorry

end floor_abs_S_eq_503_l109_109103


namespace smallest_n_satisfying_condition_l109_109520

noncomputable def exists_n_and_satisfy_condition : Prop :=
  ∃ (n : ℕ) (s : Fin n → ℕ), 
    (∀ i j : Fin n, i ≠ j → s i ≠ s j) ∧
    (Π i : Fin n, 1 < s i) ∧
    ((∏ i in Finset.range n, (1 - (1 / (s i)))) = (7 / 66))

theorem smallest_n_satisfying_condition :
  exists_n_and_satisfy_condition ∧ ∀ m, 
    (∃ (s : Fin m → ℕ),
      (∀ i j : Fin m, i ≠ j → s i ≠ s j) ∧
      (Π i : Fin m, 1 < s i) ∧
      ((∏ i in Finset.range m, (1 - (1 / (s i)))) = (7 / 66)))
    → m ≥ 9 :=
begin
  sorry
end

end smallest_n_satisfying_condition_l109_109520


namespace complex_modulus_l109_109311

noncomputable def z : ℂ := (1 - 2 * (I^3)) / (2 + I)

theorem complex_modulus : complex.abs z = 1 :=
by sorry

end complex_modulus_l109_109311


namespace josh_initial_marbles_l109_109418

def marbles_initial (lost : ℕ) (left : ℕ) : ℕ := lost + left

theorem josh_initial_marbles :
  marbles_initial 5 4 = 9 :=
by sorry

end josh_initial_marbles_l109_109418


namespace sequence_sum_bound_l109_109322

def sequence_a : ℕ → ℕ
| 1     := 0
| 2     := 3
| (n+1) := if n < 2 then 0 else 
              let a := sequence_a n
              let b := sequence_a (n-1)
              let c := sequence_a (n-2)
              (b + 2) * (c + 2) / a

def sequence_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_a

theorem sequence_sum_bound (n : ℕ) : sequence_sum n ≤ (n * (n + 1)) / 2 :=
sorry

end sequence_sum_bound_l109_109322


namespace jon_weekly_speed_gain_l109_109045

-- Definitions based on the conditions
def initial_speed : ℝ := 80
def speed_increase_percentage : ℝ := 0.20
def training_sessions : ℕ := 4
def weeks_per_session : ℕ := 4
def total_training_duration : ℕ := training_sessions * weeks_per_session

-- The calculated final speed
def final_speed : ℝ := initial_speed + initial_speed * speed_increase_percentage

theorem jon_weekly_speed_gain : 
  (final_speed - initial_speed) / total_training_duration = 1 :=
by
  -- This is the statement we want to prove
  sorry

end jon_weekly_speed_gain_l109_109045


namespace conical_heap_height_l109_109950

structure Cylinder :=
(radius : ℝ)
(height : ℝ)

structure Cone :=
(radius : ℝ)
(height : ℝ)

noncomputable def volume_cylinder (c : Cylinder) : ℝ :=
  π * c.radius^2 * c.height

noncomputable def volume_cone (c : Cone) : ℝ :=
  (1 / 3) * π * c.radius^2 * c.height

theorem conical_heap_height (cyl : Cylinder) (cone_radius : ℝ) (v_eq : volume_cylinder cyl = volume_cone {radius := cone_radius, height := _}) : 
  {c : Cone // c.radius = cone_radius ∧ c.height = 12} :=
by
  use {radius := cone_radius, height := 12}
  split
  · exact rfl
  · sorry

-- Instantiate the Cylinder with the conditions given
def bucket : Cylinder := { radius := 21, height := 36 }

-- Provide the known radius of the conical heap and the condition that volumes are equal
example : conical_heap_height bucket 63 (by sorry) :=
by 
  sorry

end conical_heap_height_l109_109950


namespace employee_salary_proof_l109_109897

variable (x : ℝ) (M : ℝ) (P : ℝ)

theorem employee_salary_proof (h1 : x + 1.2 * x + 1.8 * x = 1500)
(h2 : M = 1.2 * x)
(h3 : P = 1.8 * x)
: x = 375 ∧ M = 450 ∧ P = 675 :=
sorry

end employee_salary_proof_l109_109897


namespace visitors_on_2nd_oct_day_with_most_visitors_total_revenue_calculation_l109_109264

def changes_in_visitors : List ℝ := [+1.6, +0.8, +0.4, -0.4, -0.8, +0.2, -1.2]

def visitors_on_day (a : ℝ) (day : ℕ) : ℝ :=
  a + (List.take day changes_in_visitors).sum

def total_revenue (initial_visitors : ℝ) (ticket_price : ℝ) : ℝ :=
  let total_visitors := (List.range 7).map (visitors_on_day initial_visitors)
  (7 * initial_visitors + (List.sum changes_in_visitors)) * 10000 * ticket_price

theorem visitors_on_2nd_oct (a : ℝ) : visitors_on_day a 2 = a + 2.4 :=
by
  unfold visitors_on_day
  simp [changes_in_visitors]

theorem day_with_most_visitors (a : ℝ) : ∀ day, visitors_on_day a 3 ≥ visitors_on_day a day :=
by
  intro day
  have : [a + 1.6, a + 2.4, a + 2.8, a + 2.4, a + 1.6, a + 1.8, a + 0.6] =
    List.range 7).map (visitors_on_day a)
  simp [changes_in_visitors]

theorem total_revenue_calculation : total_revenue 2 15 = 4.08 * 10^6 :=
by
  unfold total_revenue
  simp [changes_in_visitors]
  linarith

sorry

end visitors_on_2nd_oct_day_with_most_visitors_total_revenue_calculation_l109_109264


namespace problem1_problem2_l109_109541

-- Trigonometric, algebraic, or numerical calculations may often require computational interpretation.
noncomputable theory 

-- Problem 1: Calculation proof
theorem problem1 : (-1: ℤ) ^ 2023 + real.cbrt (-8) + |real.sqrt 3 - 3| = - real.sqrt 3 :=
by
  sorry

-- Problem 2: Solving quadratic equation
theorem problem2 (x : ℝ) : (x - 1) ^ 2 = 4 / 9 ↔ x = 5 / 3 ∨ x = 1 / 3 :=
by
  sorry

end problem1_problem2_l109_109541


namespace find_p_l109_109881

theorem find_p :
  ∀ r s : ℝ, (3 * r^2 + 4 * r + 2 = 0) → (3 * s^2 + 4 * s + 2 = 0) →
  (∀ p q : ℝ, (p = - (1/(r^2)) - (1/(s^2))) → (p = -1)) :=
by 
  intros r s hr hs p q hp
  sorry

end find_p_l109_109881


namespace maximizing_point_l109_109453

variables {A B C A₁ B₁ C₁ M : Type*}
variables [Triangle ABC]
variables (on_BC : PointOnLineSegment A₁ BC)
variables (on_CA : PointOnLineSegment B₁ CA)
variables (on_AB : PointOnLineSegment C₁ AB)
variables (intersect : LinesIntersect AA₁ BB₁ CC₁ M)

theorem maximizing_point (M_at_centroid : is_centroid ABC M) :
  let α := M A₁ / (A A₁),
      β := M B₁ / (B B₁),
      γ := M C₁ / (C C₁)
  in α + β + γ = 1 → α * β * γ = (1 / 3) :=
sorry

end maximizing_point_l109_109453


namespace geometric_log_sum_l109_109653

open Real 

theorem geometric_log_sum :
  let a : ℕ → ℝ := λ n => 1 * (2 : ℝ) ^ (n - 1)
  log_sum := ∑ i in Finset.range 11, log (2 : ℝ) (a (i + 1))
in log_sum = 55 :=
by
  let a : ℕ → ℝ := λ n => 1 * (2 : ℝ) ^ (n - 1)
  have h₁ : ∀ n, log (2 : ℝ) (a n) = (n - 1 : ℝ),
    from λ n => by {rw [a, log_pow (2 : ℝ)], norm_num, exact_mod_cast le_of_lt (by decide : 0 < 2), }
  have h₂ : ∑ i in Finset.range 11, log (2 : ℝ) (a (i + 1)) = ∑ i in Finset.range 11, (i : ℝ),
    from Finset.sum_congr rfl (λ x _, h₁ (x + 1)),
  norm_num at h₂,
  rw h₂,
  exact (Finset.sum_range_id 11)


end geometric_log_sum_l109_109653


namespace harry_says_1111_l109_109625

-- Define the condition for skipping sequences
def skips (sequence: List ℕ) (skip_every: ℕ → ℕ): List ℕ :=
  sequence.enum.filter (λ i, i.fst % skip_every i.fst ≠ 0).map (λ i, i.snd)

-- Define the sequences for each student
def adam (n: ℕ): List ℕ := skips (List.range (n + 1)) (λ _, 4)

def beth (n: ℕ): List ℕ := skips (adam n) (λ _, 2)

def claire (n: ℕ): List ℕ := skips (beth n) (λ i, if i % 2 = 0 then 2 else 3)

def debby (n: ℕ): List ℕ := skips (claire n) (λ _, 2)

def eva (n: ℕ): List ℕ := skips (debby n) (λ i, if i % 2 = 0 then 2 else 3)

def frank (n: ℕ): List ℕ := skips (eva n) (λ _, 2)

def gina (n: ℕ): List ℕ := skips (frank n) (λ i, if i % 2 = 0 then 2 else 3)

def harry (n: ℕ): List ℕ := skips (gina n) (λ _, 2)

-- Theorem stating Harry says 1111
theorem harry_says_1111 : harry 1200 = [1111] :=
sorry

end harry_says_1111_l109_109625


namespace second_train_speed_l109_109153

theorem second_train_speed (length_train1 length_train2 : ℝ) (time_crossing : ℝ) (speed_train1 : ℝ)
    (h_length1 : length_train1 = 140)
    (h_length2 : length_train2 = 150)
    (h_time : time_crossing = 10.439164866810657)
    (h_speed1 : speed_train1 = 60) :
    let total_distance := length_train1 + length_train2 in
    let relative_speed_m_s := total_distance / time_crossing in
    let relative_speed_km_h := relative_speed_m_s * 3.6 in
    let speed_train2 := relative_speed_km_h - speed_train1 in
    speed_train2 = 40 := by
    sorry

end second_train_speed_l109_109153


namespace number_of_pairs_sold_l109_109188

-- Define the conditions
def total_amount_made : ℝ := 588
def average_price_per_pair : ℝ := 9.8

-- The theorem we want to prove
theorem number_of_pairs_sold : total_amount_made / average_price_per_pair = 60 := 
by sorry

end number_of_pairs_sold_l109_109188


namespace longest_segment_l109_109115

-- Definitions for the given angles
def angle_ABD : ℝ := 40
def angle_ADB : ℝ := 55
def angle_CBD : ℝ := 75
def angle_BDC : ℝ := 55

-- Condition that the sum of angles in a triangle is 180 degrees
def sum_of_angles_in_triangle (α β γ : ℝ) : Prop := α + β + γ = 180

noncomputable def angle_BAD : ℝ := 180 - angle_ABD - angle_ADB
noncomputable def angle_BCD : ℝ := 180 - angle_CBD - angle_BDC

-- Proving that CD is the longest segment
theorem longest_segment (h1 : sum_of_angles_in_triangle angle_ABD angle_ADB angle_BAD)
(h2 : sum_of_angles_in_triangle angle_CBD angle_BDC angle_BCD)
: ∀ (AD AB BD BC CD : ℝ), 
   AD < AB → AB < BD → BD < BC → BC < CD → CD = max (max (max (max AD AB) BD) BC) CD :=
begin
  sorry
end

end longest_segment_l109_109115


namespace imo2_hun4_l109_109936

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {x₁ x₂ : ℝ}

theorem imo2_hun4 (h1: ∃ y: ℝ → ℝ, y = (∑ i in Finset.range n.succ, (i : ℕ + 1) * Real.cos (a i + x)) ∧ y x₁ = 0 ∧ y x₂ = 0)
                  (h2: ¬ ∃ k: ℤ, (x₁ - x₂) = k * Real.pi) : 
                  (∑ i in Finset.range n.succ, (i : ℕ + 1) * Real.cos (a i + x) = 0) :=
sorry

end imo2_hun4_l109_109936


namespace problem1_problem2_l109_109707

-- Definitions based on the given conditions
def p (a : ℝ) (x : ℝ) : Prop := a < x ∧ x < 3 * a
def q (x : ℝ) : Prop := x^2 - 5 * x + 6 < 0

-- Problem (1)
theorem problem1 (a x : ℝ) (h : a = 1) (hp : p a x) (hq : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : ∀ x, q x → p a x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end problem1_problem2_l109_109707


namespace no_solution_exists_l109_109637

open Nat

theorem no_solution_exists : ¬ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 ^ x + 3 ^ y - 5 ^ z = 2 * 11 :=
by
  sorry

end no_solution_exists_l109_109637


namespace distinct_flags_l109_109564

noncomputable def num_distinct_flags : ℕ :=
  let colors := { "red", "white", "blue", "green", "yellow" }
  let num_colors := colors.size
  5 * 4 * 4 * 4

theorem distinct_flags : num_distinct_flags = 320 := 
  by
    sorry

end distinct_flags_l109_109564


namespace num_clients_visited_garage_l109_109220

theorem num_clients_visited_garage :
  ∃ (num_clients : ℕ), num_clients = 24 ∧
    ∀ (num_cars selections_per_car selections_per_client : ℕ),
        num_cars = 16 → selections_per_car = 3 → selections_per_client = 2 →
        (num_cars * selections_per_car) / selections_per_client = num_clients :=
by
  sorry

end num_clients_visited_garage_l109_109220


namespace impossible_even_both_or_sum_even_l109_109469

theorem impossible_even_both_or_sum_even (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) :
  ¬ (even n ∧ even m) ∧ ¬ even (n + m) :=
by {
  sorry
}

end impossible_even_both_or_sum_even_l109_109469


namespace beijing_olympics_problem_l109_109593

theorem beijing_olympics_problem
  (M T J D: Type)
  (sports: M → Type)
  (swimming gymnastics athletics volleyball: M → Prop)
  (athlete_sits: M → M → Prop)
  (Maria Tania Juan David: M)
  (woman: M → Prop)
  (left right front next_to: M → M → Prop)
  (h1: ∀ x, swimming x → left x Maria)
  (h2: ∀ x, gymnastics x → front x Juan)
  (h3: next_to Tania David)
  (h4: ∀ x, volleyball x → ∃ y, woman y ∧ next_to y x) :
  athletics David := 
sorry

end beijing_olympics_problem_l109_109593


namespace point_C_on_circle_O_l109_109753

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B : (ℝ × ℝ) := (13, 0)
noncomputable def C : (ℝ × ℝ) := (12, 5)
noncomputable def O : (ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem point_C_on_circle_O :
  let OA := real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) in
  real.sqrt ((C.1 - O.1)^2 + (C.2 - O.2)^2) = OA :=
by sorry

end point_C_on_circle_O_l109_109753


namespace shortest_tangent_length_example_l109_109813

noncomputable def shortest_tangent_length (C1 C2 : set (ℝ × ℝ)) : ℝ :=
  let A := (12, 0)
  let B := (-18, 0)
  let r1 := 7
  let r2 := 10
  let d := dist A B
  let AD := (r2 * d) / (r1 + r2)
  let DB := (r1 * d) / (r1 + r2)
  let PD := real.sqrt (AD^2 - r1^2)
  let QD := real.sqrt (DB^2 - r2^2)
  PD + QD

theorem shortest_tangent_length_example :
  shortest_tangent_length ({p | (p.1 - 12)^2 + p.2^2 = 49}) ({q | (q.1 + 18)^2 + q.2^2 = 100}) = 47.60 :=
begin
  sorry
end

end shortest_tangent_length_example_l109_109813


namespace largest_common_term_l109_109471

theorem largest_common_term (a : ℕ) (k l : ℕ) (hk : a = 4 + 5 * k) (hl : a = 5 + 10 * l) (h : a < 300) : a = 299 :=
by {
  sorry
}

end largest_common_term_l109_109471


namespace number_of_yellow_marbles_l109_109401

theorem number_of_yellow_marbles (Y : ℕ) (h : Y / (7 + 11 + Y) = 1 / 4) : Y = 6 :=
by
  -- Proof to be filled in
  sorry

end number_of_yellow_marbles_l109_109401


namespace inverse_function_value_l109_109001

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^3 + 4) :
  f⁻¹ 58 = 3 :=
by sorry

end inverse_function_value_l109_109001


namespace sum_of_reciprocals_of_roots_l109_109284

theorem sum_of_reciprocals_of_roots :
  (∀ r₁ r₂ : ℚ, (r₁ + r₂ = 14) ∧ (r₁ * r₂ = 8) → (1 / r₁ + 1 / r₂ = 7 / 4)) :=
begin
  sorry
end

end sum_of_reciprocals_of_roots_l109_109284


namespace common_root_quadratic_l109_109644

theorem common_root_quadratic (a x1: ℝ) :
  (x1^2 + a * x1 + 1 = 0) ∧ (x1^2 + x1 + a = 0) ↔ a = -2 :=
sorry

end common_root_quadratic_l109_109644


namespace min_sum_of_labels_l109_109574

def label (i j : ℕ) : ℚ := 1 / (2 * i + j)

theorem min_sum_of_labels :
  ∃ (selection : Fin 10 → Fin 10), 
  (∀ k, ∃ r, selection k = ⟨r.val, r.isLt⟩) ∧ 
  (∀ i j, i ≠ j → selection i ≠ selection j) ∧ 
  (∑ k, label (selection k).val (k.val)) = 100 / 165 :=
sorry

end min_sum_of_labels_l109_109574


namespace max_number_of_balloons_l109_109829

noncomputable def max_balloons (p : ℝ) := 36 * p

theorem max_number_of_balloons (p : ℝ) (h₁ : p > 0) :
  let total_cost := max_balloons p in
  let cost_per_set := (3 / 2) * p in
  let num_sets := total_cost / cost_per_set in
  let total_balloons := 2 * num_sets in
  total_balloons = 48 :=
by
  let total_cost := max_balloons p
  let cost_per_set := (3 / 2) * p
  let num_sets := total_cost / cost_per_set
  let total_balloons := 2 * num_sets
  sorry

end max_number_of_balloons_l109_109829


namespace divisible_sequence_exists_l109_109505

theorem divisible_sequence_exists : 
  ∃ s : List ℕ, 
  s = [1, 2, 4, 8, 975360] ∧ 
  (∀ i, i < s.length - 1 → s.get i ≠ 0 ∧ s.get (i + 1) % s.get i = 0) ∧ 
  ((s.bind Nat.digits).toFinset = Finset.range 10) :=
by
  sorry

end divisible_sequence_exists_l109_109505


namespace axis_of_symmetry_min_possible_value_of_a_l109_109700

-- Given function f
def f (x : ℝ) := Real.cos (2 * x - (4 * Real.pi / 3)) + 2 * Real.cos x ^ 2

-- Condition in triangle ABC
variable (A B C a b c : ℝ)

-- Condition: f(A/2) = 1/2
axiom f_A_over_2 : f (A / 2) = 1 / 2

-- Condition: b + c = 2
axiom b_plus_c : b + c = 2

-- Proof of axis of symmetry
theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f x = f (Real.pi * k / 2 - Real.pi / 6) := sorry

-- Proof of minimum possible value of a
theorem min_possible_value_of_a : a = 1 := sorry

end axis_of_symmetry_min_possible_value_of_a_l109_109700


namespace greatest_integer_AD_l109_109746

theorem greatest_integer_AD (AB AD : ℝ) (E : ℝ) (perpendicular : ℝ) :
  AB = 100 ∧ E = AD / 2 ∧ E * perpendicular = -1 → int.floor AD = 141 :=
by
  sorry

end greatest_integer_AD_l109_109746


namespace positive_integers_satisfying_sin_cos_equation_l109_109297

theorem positive_integers_satisfying_sin_cos_equation :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ∀ t : ℝ, (\sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)}
  in S.card = 125 :=
by
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ∀ t : ℝ, (\sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)}
  show S.card = 125,
  sorry

end positive_integers_satisfying_sin_cos_equation_l109_109297


namespace smallest_n_divisible_by_13_l109_109643

theorem smallest_n_divisible_by_13 : ∃ (n : ℕ), 5^n + n^5 ≡ 0 [MOD 13] ∧ ∀ (m : ℕ), m < n → ¬(5^m + m^5 ≡ 0 [MOD 13]) :=
sorry

end smallest_n_divisible_by_13_l109_109643


namespace remainder_when_b_divided_by_13_is_6_l109_109812

theorem remainder_when_b_divided_by_13_is_6 :
  let b := (2⁻¹ + 3⁻¹ + 5⁻¹)⁻¹ in 
  b % 13 = 6 :=
begin
  sorry
end

end remainder_when_b_divided_by_13_is_6_l109_109812


namespace exponent_problem_proof_l109_109607

theorem exponent_problem_proof :
  3 * 3^4 - 27^60 / 27^58 = -486 :=
by
  sorry

end exponent_problem_proof_l109_109607


namespace championship_games_l109_109929

theorem championship_games (n : ℕ) (h_n : n = 7) :
  (∑ i in finset.range n, i) = 21 :=
by {
  rw h_n,
  simp,
  sorry
}

end championship_games_l109_109929


namespace part_one_part_two_part_three_l109_109883

section Problem

variable (n k m : ℕ)

def A_n_set (n : ℕ) := {a | ∃ i : ℕ, i < n ∧ a = 2 * i + 1}
def A_n_set_alt (n : ℕ) := {a | ∃ i : ℕ, i < n ∧ a = 2 ^ (i + 1) - 1}

def T_k (A : Set ℕ) (k : ℕ) : ℕ := sorry  -- Sum of all products of any k elements in A

theorem part_one (n : ℕ) (h : n = 3) :
  let A_3 := A_n_set 3;
  T_k A_3 1 = 11 ∧ T_k A_3 2 = 31 ∧ T_k A_3 3 = 21 :=
sorry

theorem part_two (k m : ℕ) (hk : k > 0) (hm : 2 ≤ m ∧ m ≤ k) :
  let A_k := A_n_set_alt k;
  let A_k1 := A_n_set_alt (k+1);
  T_k A_k1 m = (2 ^ (k + 1) - 1) * T_k A_k (m - 1) + T_k A_k m :=
sorry

def S_n (A : Set ℕ) : ℕ := (finset.range n.succ).sum (λ i, T_k A i)  -- Sum of T_i from 1 to n

theorem part_three (n : ℕ) (hn : 0 < n) :
  let A_n := A_n_set_alt n;
  S_n A_n = 2^(n*(n+1)/2) - 1 :=
sorry

end Problem

end part_one_part_two_part_three_l109_109883


namespace integer_part_of_E_l109_109049

theorem integer_part_of_E (n : ℕ) (x : ℕ → ℝ) 
  (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < x i)
  (h_sum : (∑ i in Finset.range n, x (i+1)) = 1) :
  (⌊ x 1 + ∑ i in Finset.range (n - 1), (x (i + 2) / (Real.sqrt (1 - (∑ j in Finset.range (i + 1), (x (j + 1))^2)))) ⌋ = 1) :=
by
  sorry

end integer_part_of_E_l109_109049


namespace total_points_l109_109969

noncomputable def points_question_33 : ℕ := 3
noncomputable def points_question_34 : ℕ := 6
noncomputable def points_question_35 : ℕ := 4

theorem total_points : (points_question_33 + points_question_34 + points_question_35) = 13 :=
by
  simp [points_question_33, points_question_34, points_question_35]
  sorry

end total_points_l109_109969


namespace number_of_regions_in_convex_polygon_l109_109390

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_regions_in_convex_polygon (n : ℕ) (h1 : 3 ≤ n) :
  let num_regions := 1 + binom n 2 - n + binom n 4
  num_regions = 1 + (n * (n - 1)) / 2 - n + (n * (n - 1) * (n - 2) * (n - 3)) / 24 :=
begin
  sorry
end

end number_of_regions_in_convex_polygon_l109_109390


namespace range_of_a_l109_109869

variable (a : ℝ)
def f (x : ℝ) := x^2 + 2 * (a - 1) * x + 2
def f_deriv (x : ℝ) := 2 * x + 2 * (a - 1)

theorem range_of_a (h : ∀ x ≥ -4, f_deriv a x ≥ 0) : a ≥ 5 :=
sorry

end range_of_a_l109_109869


namespace intersection_infinite_l109_109111

-- Define the equations of the curves
def curve1 (x y : ℝ) : Prop := 2 * x^2 - x * y - y^2 - x - 2 * y - 1 = 0
def curve2 (x y : ℝ) : Prop := 3 * x^2 - 4 * x * y + y^2 - 3 * x + y = 0

-- Theorem statement
theorem intersection_infinite : ∃ (f : ℝ → ℝ), ∀ x, curve1 x (f x) ∧ curve2 x (f x) :=
sorry

end intersection_infinite_l109_109111


namespace h_squared_right_triangle_l109_109992

noncomputable def complex_values : Type :=
  {a b c : ℂ // (∃ (q r : ℂ), (polynomial.map (complex.of_real) (z^3 + q * z + r)).roots = {a, b, c}) 
                ∧ abs a ^ 2 + abs b ^ 2 + abs c ^ 2 = 250
                ∧ (triangle_formed_by_complex_points a b c is_right_angle)}

def h_squared (a b c : ℂ) [h : (a, b, c) ∈ complex_values] : ℝ :=
  let x := abs (b - c)
      y := abs (a - b)
  in x ^ 2 + y ^ 2

theorem h_squared_right_triangle :
  ∀ (a b c : ℂ) [h : (a, b, c) ∈ complex_values],
  ∃ (h_sq : ℝ), h_sq = 375 :=
by 
  intro a b c h
  use (h_squared a b c)
  sorry

end h_squared_right_triangle_l109_109992


namespace round_to_hundredth_l109_109177

theorem round_to_hundredth:
  (∀ (x : Float), x ∈ {34.561, 34.558, 34.5601, 34.56444} → Float.round (x * 100) / 100 = 34.56) ∧ 
  (Float.round (34.5539999 * 100) / 100 ≠ 34.56) :=
by
  sorry

end round_to_hundredth_l109_109177


namespace problem_statement_l109_109476

noncomputable def f (x: ℝ) : ℝ := sin x + cos (abs x)

theorem problem_statement :
  (∀ x, f (x + 2 * π) = f x) ∧
  ¬ (∀ x ∈ (Icc 0 (5 * π / 4 : ℝ)), ∀ y ∈ (Icc x (5 * π / 4 : ℝ)), f x ≤ f y) ∧
  (∃ x y : ℝ, -π ≤ x ∧ x ≤ π ∧ -π ≤ y ∧ y ≤ π ∧ f x = 1 ∧ f y = 1 ∧ x ≠ y) ∧
  (∀ x, f x ≥ -sqrt 2) :=
by
  sorry

end problem_statement_l109_109476


namespace problem_l109_109855

variable (a b : ℝ)

theorem problem (h : a = 1.25 * b) : (4 * b) / a = 3.2 :=
by
  sorry

end problem_l109_109855


namespace find_m_l109_109976

-- Given conditions as Lean definitions
def is_isosceles_triangle (A B C : Type) : Prop := (A B = A C)
def is_inscribed_in_circle (A B C : Type) := ∃ (O : Type), True 
def are_tangents_meet_at_D (B C D : Type) (circle : Type) := ∃ (tangent1 tangent2 : Type), True
def angles_relationship (ABC ACB D : Type) := (ABC = 3 * D) ∧ (ACB = 3 * D)
def angle_BAC (BAC : Type) (m : ℝ) (π : ℝ) := BAC = m * π

-- Theorem statement in Lean
theorem find_m (A B C D : Type) (π m : ℝ) :
  is_isosceles_triangle A B C →
  is_inscribed_in_circle A B C →
  are_tangents_meet_at_D B C D A →
  angles_relationship B C D →
  angle_BAC A m π →
  m = 5 / 11 :=
begin
  sorry
end

end find_m_l109_109976


namespace hyperbola_equation_l109_109347

theorem hyperbola_equation 
    (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
    (parabola_eq : ∀ x y : ℝ, y^2 = 4 * (real.sqrt 5) * x) 
    (directrix : ∀ x : ℝ, x = - (real.sqrt 5))
    (left_focus : c = real.sqrt 5)
    (hyperbola_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
    (angle_condition : angle F₁ F₂ A = (real.pi / 4))
    : x^2 - (y^2 / 4) = 1 :=
sorry

end hyperbola_equation_l109_109347


namespace product_in_fourth_quadrant_l109_109695

noncomputable def z1 : ℂ := 3 + complex.i
noncomputable def z2 : ℂ := 1 - complex.i
noncomputable def z_product : ℂ := z1 * z2

-- Define what it means for a complex number to be in the fourth quadrant.
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Statement of the proof.
theorem product_in_fourth_quadrant : in_fourth_quadrant z_product :=
sorry

end product_in_fourth_quadrant_l109_109695


namespace S_n_formula_l109_109432

noncomputable def S_n (a : ℝ) [h : a ≠ 0]: ℕ → ℝ
| n := let a_n := (λ n, a * (-a)^(n-1)) in
       let b_n := λ n, a_n n * log (abs (a_n n)) in
       ∑ i in finset.range n, b_n (i + 1)

theorem S_n_formula (a : ℝ) (h : a ≠ -1) (n : ℕ) : 
  S_n a (λ h_a : a ≠ 0, S_n a h_a) n = 
  a * log (abs a) * (a + (-1)^(n+1) * (1 + n + n * a) * (a^n)) / (1 + a)^2 :=
by sorry

end S_n_formula_l109_109432


namespace ratio_goats_sold_to_total_l109_109448

-- Define the conditions
variables (G S : ℕ) (total_revenue goat_sold : ℕ)
-- The ratio of goats to sheep is 5:7
axiom ratio_goats_to_sheep : G = (5/7) * S
-- The total number of sheep and goats is 360
axiom total_animals : G + S = 360
-- Mr. Mathews makes $7200 from selling some goats and 2/3 of the sheep
axiom selling_conditions : 40 * goat_sold + 30 * (2/3) * S = 7200

-- Prove the ratio of the number of goats sold to the total number of goats
theorem ratio_goats_sold_to_total : goat_sold / G = 1 / 2 := by
  sorry

end ratio_goats_sold_to_total_l109_109448


namespace zero_not_in_range_of_g_l109_109060

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈(Real.cos x) / (x + 3)⌉
  else if x < -3 then ⌊(Real.cos x) / (x + 3)⌋
  else 0 -- arbitrary value since it's undefined

theorem zero_not_in_range_of_g :
  ¬ (∃ x : ℝ, g x = 0) :=
by
  intro h
  sorry

end zero_not_in_range_of_g_l109_109060


namespace sum_solutions_eq_26_l109_109770

theorem sum_solutions_eq_26:
  (∃ (n : ℕ) (solutions: Fin n → (ℝ × ℝ)),
    (∀ i, let (x, y) := solutions i in |x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|)
    ∧ (Finset.univ.sum (λ i, let (x, y) := solutions i in x + y) = 26))
:= sorry

end sum_solutions_eq_26_l109_109770


namespace h_even_l109_109101

-- Given conditions
def k : ℝ → ℝ := sorry -- Assume some function k
axiom k_even : ∀ x, k(-x) = k(x)

def h (x : ℝ) := |k(x^4)|

-- Proof problem: Prove that h is an even function
theorem h_even : ∀ x, h(-x) = h(x) := by
  sorry

end h_even_l109_109101


namespace log_base_3_of_243_l109_109267

theorem log_base_3_of_243 : ∃ x : ℤ, (∃ y : ℤ, y = 243 ∧ 3^5 = y) ∧ 3^x = 243 ∧ log 3 243 = 5 := by
  -- Introducing the given conditions
  let y := 243
  have h1: 3^5 = y := by sorry -- Known condition in the problem
  -- Prove the main statement
  use 5
  split
  . use y
    exact ⟨rfl, h1⟩
  . split
    . exact h1
    . sorry

end log_base_3_of_243_l109_109267


namespace pentagon_area_l109_109715

-- Definitions of the side lengths of the pentagon
def side1 : ℕ := 12
def side2 : ℕ := 17
def side3 : ℕ := 25
def side4 : ℕ := 18
def side5 : ℕ := 17

-- Definitions for the rectangle and triangle dimensions
def rectangle_width : ℕ := side4
def rectangle_height : ℕ := side1
def triangle_base : ℕ := side4
def triangle_height : ℕ := side3 - side1

-- The area of the pentagon proof statement
theorem pentagon_area : rectangle_width * rectangle_height +
    (triangle_base * triangle_height) / 2 = 333 := by
  sorry

end pentagon_area_l109_109715


namespace natural_number_representable_l109_109802

def strictly_decreasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) < a n

def sum_of_first_k_minus_one (a : ℕ → ℕ) (k : ℕ) : ℕ :=
(if k = 1 then 0 else (Nat.range (k - 1)).sum a)

theorem natural_number_representable
  (a : ℕ → ℕ)
  (h_decreasing : strictly_decreasing_seq a) :
  (∀ n : ℕ, ∃ (s : set ℕ), s.finite ∧ (∀ i ∈ s, i < n) ∧ (s.sum id = n)) ↔ (∀ k : ℕ, a k ≤ sum_of_first_k_minus_one a k + 1) :=
sorry

end natural_number_representable_l109_109802


namespace side_length_of_dodecagon_inscribed_in_circle_l109_109621

variable (d : ℝ)

-- Define the side length a12 given the diameter d
def a12 := d / 2 * Real.sqrt (2 - Real.sqrt 3)

-- Define the condition given in the problem
def condition := (d / 2 - Real.sqrt ((d / 2) ^ 2 - (d / 4) ^ 2)) ^ 2 + (d / 4) ^ 2

theorem side_length_of_dodecagon_inscribed_in_circle :
  a12 d ^ 2 = condition d := by
  sorry

end side_length_of_dodecagon_inscribed_in_circle_l109_109621


namespace problem_l109_109733

-- Definitions according to the conditions
def red_balls : ℕ := 1
def black_balls (n : ℕ) : ℕ := n
def total_balls (n : ℕ) : ℕ := red_balls + black_balls n
noncomputable def probability_red (n : ℕ) : ℚ := (red_balls : ℚ) / (total_balls n : ℚ)
noncomputable def variance (n : ℕ) : ℚ := (black_balls n : ℚ) / (total_balls n : ℚ)^2

-- The theorem we want to prove
theorem problem (n : ℕ) (h : 0 < n) : 
  (∀ m : ℕ, n < m → probability_red m < probability_red n) ∧ 
  (∀ m : ℕ, n < m → variance m < variance n) :=
sorry

end problem_l109_109733


namespace solve_log_equation_l109_109098

theorem solve_log_equation (x : ℝ) (h : 2^(2*x + 1) - 6 > 0) :
     log 2 (2^(2*x + 1) - 6) = x + log 2 (2^x + 1) ↔ x = log 2 3 :=
by
  sorry

end solve_log_equation_l109_109098


namespace seed_germination_probability_l109_109119

-- Define necessary values and variables
def n : ℕ := 3
def p : ℚ := 0.7
def k : ℕ := 2

-- Define the binomial probability formula
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- State the proof problem
theorem seed_germination_probability :
  binomial_probability n k p = 0.441 := 
sorry

end seed_germination_probability_l109_109119


namespace find_probabilities_credits_probability_l109_109558

variables (m n : ℝ)

/-- Given conditions:
(1) The probability of entering all three clubs is \(\frac{1}{24}\).
(2) The probability of entering the Calligraphy, Poetry, and Science club
    are \(m, \frac{1}{3},\) and \(n\) respectively.
(3) The probability of entering at least one club is \(\frac{3}{4}\).
(4) It is also given that \(m > n\).
-/
theorem find_probabilities (h1 : m * (1 / 3) * n = 1 / 24)
  (h2 : 1 - (1 - m) * (2 / 3) * (1 - n) = 3 / 4)
  (h3 : m > n) :
  m = 1 / 2 ∧ n = 1 / 4 :=
sorry

variables (P_C P_P P_S: ℝ) (P_CP: ℝ) (P_CS: ℝ) (P_PS: ℝ)

def students_Earn_at_least_4_credits : ℝ :=
  -- Probability of earning 4 credits
  (P_C * (2/3) * (1 - n)) +
  -- Probability of earning 5 credits
  (P_C * (1 / 3) * (3 / 4)) +
  -- Probability of earning 6 credits
  ((1 - m) * (1 / 3) * (1 / 4))

/-- Given:
(1) Each credit probability given
(2) Total probability of earning at least 4 credits
-/
theorem credits_probability (h4 : P_C = 1/2) (h5 : P_P = 1/3) (h6 : P_S = 1/4) :
  students_Earn_at_least_4_credits P_C P_P P_S P_CP P_CS P_PS = 1 / 6 :=
sorry

end find_probabilities_credits_probability_l109_109558


namespace lcm_technicians_schedule_l109_109239

theorem lcm_technicians_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := 
sorry

end lcm_technicians_schedule_l109_109239


namespace parallel_line_through_point_distance_from_origin_to_line_equal_intercepts_l109_109397

-- Part (1)
theorem parallel_line_through_point (a : ℝ) (h : a = 1) :
  let l : ℝ → ℝ → Prop := λ x y, (a + 1) * x - y + 4 * a = 0 in
  ∀ x y, l x y → (2 * x - y - 2 = 0) := sorry

-- Part (2)
theorem distance_from_origin_to_line (a : ℝ) (h : ∀ x y, (a + 1) * x - y + 4 * a = 0) :
  let d : ℝ := 4 in
  |d| = 4 → a = -1 := sorry

-- Part (3)
theorem equal_intercepts (a : ℝ) :
  let l : ℝ → ℝ → Prop := λ x y, (a + 1) * x - y + 4 * a = 0 in
  ∀ x y, l x y → 
    (∃ x_intercept y_intercept, x_intercept = y_intercept) → (a = 0 ∨ a = -2) := sorry

end parallel_line_through_point_distance_from_origin_to_line_equal_intercepts_l109_109397


namespace possible_values_of_expression_l109_109678

-- Defining the nonzero real numbers a, b, c, and d
variables {a b c d : ℝ}

-- Assuming that a, b, c, and d are nonzero
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- The expression to prove
def expression := (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem to state the possible values of the expression
theorem possible_values_of_expression : 
  expression ha hb hc hd ∈ ({5, 1, -3, -5} : set ℝ) := sorry

end possible_values_of_expression_l109_109678


namespace sum_of_solutions_l109_109780

theorem sum_of_solutions :
  let solutions := [(-8, -2), (-1, 5), (10, 4), (10, 4)],
  (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 :=
by
  let solutions : List (ℤ × ℤ) := [(-8, -2), (-1, 5), (10, 4), (10, 4)]
  have h1 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 4| = |y - 10| := sorry
  have h2 : ∀ (x y : ℤ), (x, y) ∈ solutions → |x - 10| = 3 * |y - 4| := sorry
  have solution_sum : (sum (solutions.map (λ (p : ℤ × ℤ), p.1 + p.2))) = 22 := by
    simp [solutions]
    norm_num
  exact solution_sum

end sum_of_solutions_l109_109780


namespace carol_emily_balance_equal_l109_109605

theorem carol_emily_balance_equal :
  ∃ t : ℕ, (200 + 30 * t = 250 + 25 * t) ∧ t = 10 :=
begin
  use 10,
  split,
  {
    linarith,
  },
  {
    refl,
  }
end

end carol_emily_balance_equal_l109_109605


namespace degree_f_plus_g_is_3_l109_109856

noncomputable def degree_of_sum_polynomials (a3 a2 a1 a0 b2 b1 b0 : ℂ) (h : a3 ≠ 0) : ℤ :=
  let f := (λ z : ℂ, a3 * z^3 + a2 * z^2 + a1 * z + a0)
  let g := (λ z : ℂ, b2 * z^2 + b1 * z + b0)
  degree (f + g)

theorem degree_f_plus_g_is_3 (a3 a2 a1 a0 b2 b1 b0 : ℂ) (h : a3 ≠ 0) : degree_of_sum_polynomials a3 a2 a1 a0 b2 b1 b0 h = 3 :=
  sorry

end degree_f_plus_g_is_3_l109_109856


namespace inequality_proof_l109_109651

variable {x : ℝ}
variable {n : ℕ}
variable {a : ℝ}

theorem inequality_proof (h1 : x > 0) (h2 : n > 0) (h3 : x + a / x^n ≥ n + 1) : a = n^n := 
sorry

end inequality_proof_l109_109651


namespace champion_class_l109_109020

theorem champion_class (class3 class4 class5 : Prop)
    (judgeA : ¬ class3 ∧ ¬ class4)
    (judgeB : ¬ class3 ∧ class5)
    (judgeC : ¬ class5 ∧ class3)
    (judge_conditions : 
        (judgeA ∨ judgeB ∨ judgeC) ∧ 
        (¬ (judgeA ∧ judgeB ∧ judgeC)) ∧
        (judgeA → ¬ judgeB) ∧
        (judgeA → ¬ judgeC) ∧
        (judgeB → ¬ judgeC) ∧ 
        ((judgeA ∧ ¬ judgeB ∧ ¬ judgeC) ∨ 
        (judgeB ∧ ¬ judgeA ∧ ¬ judgeC) ∨ 
        (judgeC ∧ ¬ judgeA ∧ ¬ judgeB))) :
  class3 :=
by
  sorry

end champion_class_l109_109020


namespace positive_integers_satisfying_sin_cos_equation_l109_109296

theorem positive_integers_satisfying_sin_cos_equation :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ∀ t : ℝ, (\sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)}
  in S.card = 125 :=
by
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ∀ t : ℝ, (\sin t - complex.I * cos t)^n = sin (n * t) - complex.I * cos (n * t)}
  show S.card = 125,
  sorry

end positive_integers_satisfying_sin_cos_equation_l109_109296


namespace area_of_highest_points_l109_109470

noncomputable def highest_point_area (u g : ℝ) : ℝ :=
  let x₁ := u^2 / (2 * g)
  let x₂ := 2 * u^2 / g
  (1/4) * ((x₂^2) - (x₁^2))

theorem area_of_highest_points (u g : ℝ) : highest_point_area u g = 3 * u^4 / (4 * g^2) :=
by
  sorry

end area_of_highest_points_l109_109470


namespace expand_expression_l109_109629

theorem expand_expression (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 5) = x^4 - 4 * x^2 - 45 := 
by
  sorry

end expand_expression_l109_109629


namespace n_gon_partition_exists_l109_109927

/-- If a convex n-gon X can be partitioned into triangles with n-3 diagonals 
which do not intersect except at vertices, and there is a closed path that 
includes each side and each of the n-3 diagonals exactly once, then n must 
be a multiple of 3. Conversely, if n is a multiple of 3, then such a partition and 
closed path exists. -/
theorem n_gon_partition_exists (X : Type) (n : ℕ) (h_convex: convex X n)
  (h_partition : ∃ (triangles : finset (finset ℕ)), 
    (∀ t ∈ triangles, card t = 3 ∧ (∀ (v ∈ t), v ∈ (finset.range n)))
    ∧ triangles.card = n - 2 ∧ 
    (∀ (t1 t2 ∈ triangles), t1 ≠ t2 → disjoint t1 t2))
  (h_path : ∃ (path : list (finset ℕ)), 
    (∀ p ∈ path, (∃ t ∈ triangles, p ⊆ t)) ∧ path.length = n + (n - 3)) : 
  n % 3 = 0 ∧ (if n % 3 = 0 then (∃ (triangles : finset (finset ℕ)), 
    (∀ t ∈ triangles, card t = 3 ∧ (∀ (v ∈ t), v ∈ (finset.range n)))
    ∧ triangles.card = n - 2 ∧ 
    (∀ (t1 t2 ∈ triangles), t1 ≠ t2 → disjoint t1 t2)
    ∧ ∃ (path : list (finset ℕ)), 
      (∀ p ∈ path, (∃ t ∈ triangles, p ⊆ t)) ∧ path.length = n + (n - 3)) else false) :=
sorry

end n_gon_partition_exists_l109_109927


namespace concurrency_of_lines_l109_109315

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def similar (Δ₁ Δ₂ : Triangle) : Prop := sorry

variables (A B C : Point)
variables (nonisosceles_nonright_triangle : ¬is_isosceles A B C ∧ ¬is_right A B C)

def O := circumcenter A B C
def A1 := midpoint B C
def B1 := midpoint C A
def C1 := midpoint A B

def A2 := sorry -- Point on the ray OA1 such that Δ OAA1 ∼ Δ OA2A
def B2 := sorry -- Point on the ray OB1 such that Δ OBB1 ∼ Δ OB2B
def C2 := sorry -- Point on the ray OC1 such that Δ OCC1 ∼ Δ OC2C

theorem concurrency_of_lines :
  concurrent (line_through A A2) (line_through B B2) (line_through C C2) :=
sorry

end concurrency_of_lines_l109_109315


namespace max_slope_avoiding_lattice_points_l109_109996

theorem max_slope_avoiding_lattice_points :
  ∃ a : ℝ, (1 < a ∧ ∀ m : ℝ, (1 < m ∧ m < a) → (∀ x : ℤ, (10 < x ∧ x ≤ 200) → ∃ k : ℝ, y = m * x + 5 ∧ (m * x + 5 ≠ k))) ∧ a = 101 / 100 :=
sorry

end max_slope_avoiding_lattice_points_l109_109996


namespace parabola_equation_l109_109280

def parabola_condition (x y : ℝ) (a b : ℝ) : Prop :=
  (y^2 = a * x) ∨ (x^2 = b * y)

def point_on_parabola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P = (4, -2) ∧ parabola_condition P.1 P.2 a b

theorem parabola_equation :
  ∃ (a b : ℝ), point_on_parabola (4, -2) a b :=
by
  existsi (1)
  existsi (-8)
  split
  · exact rfl
  · left
    exact rfl
    sorry

end parabola_equation_l109_109280


namespace new_energy_vehicle_sales_growth_l109_109178

theorem new_energy_vehicle_sales_growth (x : ℝ) :
  let sales_jan := 64
  let sales_feb := 64 * (1 + x)
  let sales_mar := 64 * (1 + x)^2
  (sales_jan + sales_feb + sales_mar = 244) :=
sorry

end new_energy_vehicle_sales_growth_l109_109178


namespace longer_diagonal_of_parallelogram_l109_109905

variable (h1 h2 α : ℝ)

theorem longer_diagonal_of_parallelogram (h1 h2 : ℝ) (α : ℝ) : 
  ∃ d : ℝ, d = (real.sqrt (h1 ^ 2 + h2 ^ 2 + 2 * h1 * h2 * real.cos α)) / real.sin α :=
by 
  sorry

end longer_diagonal_of_parallelogram_l109_109905


namespace find_AC_length_l109_109841

-- Definitions based on given conditions
def circle_radius : ℝ := 5
def length_AB : ℝ := 6

-- Define the points A, B, O (center), and C
structure Point where
  x : ℝ
  y : ℝ

def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def midpoint (P Q : Point) : Point :=
  {x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2}

-- Given: A, B on a circle with the origin as the center, radius = 5, AB = 6.
-- To Prove: AC = sqrt(10)
theorem find_AC_length (A B O C : Point) (rAB : distance A B = length_AB)
  (rA : distance A O = circle_radius) (rB : distance B O = circle_radius)
  (rC : (distance C O = circle_radius) ∧ C = midpoint A B) : distance A C = Real.sqrt 10 := 
  sorry

end find_AC_length_l109_109841


namespace table_covered_with_three_layers_l109_109142

theorem table_covered_with_three_layers (A T table_area two_layers : ℕ)
    (hA : A = 204)
    (htable : table_area = 175)
    (hcover : 140 = 80 * table_area / 100)
    (htwo_layers : two_layers = 24) :
    3 * T + 2 * two_layers + (140 - two_layers - T) = 204 → T = 20 := by
  sorry

end table_covered_with_three_layers_l109_109142


namespace set_union_complement_l109_109708

open Set

variable (I A B : Set ℕ)

theorem set_union_complement :
  I = {1, 2, 3, 4} →
  A = {1} →
  B = {2, 4} →
  A ∪ (I \ B) = {1, 3} :=
by
  intros hI hA hB
  rw [hI, hA, hB]
  have H : I \ B = {1, 3} := by simp
  rw H
  simp
  sorry

end set_union_complement_l109_109708


namespace ternary_to_binary_equiv_binary_to_ternary_equiv_l109_109136

-- Definition and conditions for Problem 1
def ternary_x : ℝ := ∑' (n : ℕ), (1 / (3:ℝ)^(2*n + 1))
def binary_x : ℝ := 3 / 8

-- Proof problem for Problem 1
theorem ternary_to_binary_equiv : ternary_x = binary_x := by
  sorry

-- Definition and conditions for Problem 2
def binary_y : ℝ := ∑' (n : ℕ), (if n % 3 == 0 then 0 else 1) / 2^n
def ternary_y : ℝ := 6 / 7

-- Proof problem for Problem 2
theorem binary_to_ternary_equiv : binary_y = ternary_y := by
  sorry

end ternary_to_binary_equiv_binary_to_ternary_equiv_l109_109136


namespace complement_intersection_l109_109665

open Set

noncomputable def A : Set ℝ := {x | abs (x - 1) ≥ 2}
def B : Set ℕ := {x | x < 4}

theorem complement_intersection :
  (compl A) ∩ (B : Set ℝ) = {0, 1, 2} := by
  sorry

end complement_intersection_l109_109665


namespace monotonic_intervals_l109_109261

noncomputable def f : ℝ → ℝ :=
  λ x, Real.sin ((1/2) * x + (Real.pi / 3))

theorem monotonic_intervals :
  ∀ x : ℝ, x ∈ Set.Icc (-2 * Real.pi) (2 * Real.pi) →
    (Set.Icc (-5 * Real.pi / 3) (Real.pi / 3)).Subset { x | ∃ x ∈ { x }, ∀ y ∈ { y | f y = x }, y < x } ∧
    (Set.Icc (Real.pi / 3) (2 * Real.pi)).Subset { x | ∃ x ∈ { x }, ∀ y ∈ { y | f y = x }, y > x } ∧
    (Set.Icc (-2 * Real.pi) (-5 * Real.pi / 3)).Subset { x | ∃ x ∈ { x }, ∀ y ∈ { y | f y = x }, y > x } :=
  sorry

end monotonic_intervals_l109_109261


namespace ratio_of_candies_l109_109043

theorem ratio_of_candies (candiesEmily candiesBob : ℕ) (candiesJennifer : ℕ) 
  (hEmily : candiesEmily = 6) 
  (hBob : candiesBob = 4)
  (hJennifer : candiesJennifer = 3 * candiesBob) : 
  (candiesJennifer / Nat.gcd candiesJennifer candiesEmily) = 2 ∧ (candiesEmily / Nat.gcd candiesJennifer candiesEmily) = 1 := 
by
  sorry

end ratio_of_candies_l109_109043


namespace number_of_yellow_marbles_l109_109404

/-- 
 In a jar with blue, red, and yellow marbles:
  - there are 7 blue marbles
  - there are 11 red marbles
  - the probability of picking a yellow marble is 1/4
 Show that the number of yellow marbles is 6.
-/
theorem number_of_yellow_marbles 
  (blue red y : ℕ) 
  (h_blue : blue = 7) 
  (h_red : red = 11) 
  (h_prob : y / (18 + y) = 1 / 4) : 
  y = 6 := 
sorry

end number_of_yellow_marbles_l109_109404


namespace common_point_condition_l109_109784

variables {P A B C A1 B1 C1 A2 B2 C2 : Type*}

-- Given conditions
-- P is a point inside triangle ABC
-- A1, B1, C1 are the points where lines PA, PB, PC intersect sides BC, CA, AB respectively.
-- A2, B2, C2 are the points where lines B1C1, C1A1, A1B1 intersect sides BC, CA, AB respectively.
-- ω1, ω2, ω3 are circles with diameters A1A2, B1B2, C1C2 respectively.

def is_point_in_triangle (P A B C : Type*) : Prop := sorry
def is_intersection_point (L1 L2 : Type*) (P : Type*) : Prop := sorry
def is_circle_with_diameter (A B : Type*) (ω : Type*) : Prop := sorry
def has_common_point (ω1 ω2 : Type*) (K : Type*) : Prop := sorry

-- Lean theorem statement
theorem common_point_condition
    {P A B C A1 B1 C1 A2 B2 C2 K : Type*}
    (h1 : is_point_in_triangle P A B C)
    (h2 : is_intersection_point (line_through P A) (line_through B C) A1)
    (h3 : is_intersection_point (line_through P B) (line_through C A) B1)
    (h4 : is_intersection_point (line_through P C) (line_through A B) C1)
    (h5 : is_intersection_point (line_through B1 C1) (line_through B C) A2)
    (h6 : is_intersection_point (line_through C1 A1) (line_through C A) B2)
    (h7 : is_intersection_point (line_through A1 B1) (line_through A B) C2)
    (h8 : is_circle_with_diameter A1 A2 ω1)
    (h9 : is_circle_with_diameter B1 B2 ω2)
    (h10 : is_circle_with_diameter C1 C2 ω3)
    (h11 : has_common_point ω1 ω2 K) :
    has_common_point ω3 (common_point ω1 ω2) :=
sorry

end common_point_condition_l109_109784


namespace sixth_number_is_811_l109_109589

noncomputable def sixth_number_in_21st_row : ℕ := 
  let n := 21 
  let k := 6
  let total_numbers_up_to_previous_row := n * n
  let position_in_row := total_numbers_up_to_previous_row + k
  2 * position_in_row - 1

theorem sixth_number_is_811 : sixth_number_in_21st_row = 811 := by
  sorry

end sixth_number_is_811_l109_109589


namespace figure_is_convex_if_supporting_lines_l109_109843

-- Definition of a supporting line
def is_supporting_line (F : set (ℝ × ℝ)) (l : ℝ × ℝ → Prop) : Prop :=
  ∀ (q r : ℝ × ℝ), q ∈ F → r ∈ F → l q → l r

-- Given condition: For every boundary point of F, there is at least one supporting line.
def has_supporting_line_through_each_boundary_point (F : set (ℝ × ℝ)) (boundary : ℝ × ℝ → Prop) : Prop :=
  ∀ (p : ℝ × ℝ), boundary p → ∃ (l : ℝ × ℝ → Prop), is_supporting_line F l

-- Defining convexity of a set F
def is_convex (F : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ × ℝ), x ∈ F → y ∈ F → ∀ (t : ℝ), 0 ≤ t → t ≤ 1 → (t • x + (1 - t) • y) ∈ F

-- Main theorem: If there is a supporting line through every boundary point of F, then F is convex.
theorem figure_is_convex_if_supporting_lines (
  F : set (ℝ × ℝ)) (boundary : ℝ × ℝ → Prop)
  (h : has_supporting_line_through_each_boundary_point F boundary) :
  is_convex F :=
  sorry

end figure_is_convex_if_supporting_lines_l109_109843


namespace Moe_mows_in_correct_time_l109_109821

open Real

-- Definition of the conditions
def lawn_length := 90
def lawn_width := 150
def swath_width_in_inches := 28
def overlap_in_inches := 4
def walking_speed := 5000

-- The effective swath width in feet
def effective_swath_width := (swath_width_in_inches - overlap_in_inches) / 12

-- The correct answer we need to prove
def mowing_time := 1.35

-- The theorem we need to prove
theorem Moe_mows_in_correct_time
  (lawn_length lawn_width : ℝ)
  (effective_swath_width : ℝ)
  (walking_speed : ℝ)
  (mowing_time : ℝ) :
  (lawn_length * real.ceil (lawn_width / effective_swath_width)) / walking_speed = mowing_time := by
  sorry

end Moe_mows_in_correct_time_l109_109821


namespace largest_sum_is_three_fourths_l109_109606

-- Definitions of sums
def sum1 := (1 / 4) + (1 / 2)
def sum2 := (1 / 4) + (1 / 9)
def sum3 := (1 / 4) + (1 / 3)
def sum4 := (1 / 4) + (1 / 10)
def sum5 := (1 / 4) + (1 / 6)

-- The theorem stating that sum1 is the maximum of the sums
theorem largest_sum_is_three_fourths : max (max (max (max sum1 sum2) sum3) sum4) sum5 = 3 / 4 := 
sorry

end largest_sum_is_three_fourths_l109_109606


namespace value_of_f_1998_l109_109468

variable {R : Type} [Add R] [Inhabited R]

def f (x : R) : R
  
axiom axiom1 (x y : R) : f (x + y) = x + f y
axiom axiom2 : f 0 = 2

theorem value_of_f_1998 : f 1998 = 2000 :=
by
  sorry

end value_of_f_1998_l109_109468


namespace sum_solutions_eq_26_l109_109769

theorem sum_solutions_eq_26:
  (∃ (n : ℕ) (solutions: Fin n → (ℝ × ℝ)),
    (∀ i, let (x, y) := solutions i in |x - 4| = |y - 10| ∧ |x - 10| = 3 * |y - 4|)
    ∧ (Finset.univ.sum (λ i, let (x, y) := solutions i in x + y) = 26))
:= sorry

end sum_solutions_eq_26_l109_109769


namespace triangle_AFC_area_l109_109591

theorem triangle_AFC_area (a : ℕ) (h_a : a = 8) :
  let s := (1 / 2) * a * a
  in s = 32 := by
  -- Define the side lengths of the squares
  let s1 := a
  let s2 := a

  -- Calculate the diagonal of the square ABCD
  have h_diag : (s1 * s1 + s2 * s2 : ℝ) = 2 * (a * a) := 
    by rw [←sq s1, ←sq s2]; rw [sq (a : ℝ)];
    exact add_eq_coe.mpr (by ring)
  
  -- Calculate area of triangle AFC
  let area_triangle := (1 / 2) * a * a

  -- State the final proof for area of triangle AFC
  exact h_a.symm ▸ rfl

end triangle_AFC_area_l109_109591


namespace min_cells_to_mark_l109_109452

noncomputable def check_tetromino_coverage (board : ℕ × ℕ) (marked_cells : finset (ℕ × ℕ)) :=
  ∀ (tetromino_position : finset (ℕ × ℕ)), 
  ∃ (unique_position : finset (ℕ × ℕ)), 
    tetromino_position = unique_position ∧ unique_position ⊆ marked_cells

theorem min_cells_to_mark (board : ℕ × ℕ) (tetromino_size : ℕ) :
  board = (8, 8) → tetromino_size = 4 → ∀ marked_cells : finset (ℕ × ℕ),
  (check_tetromino_coverage board marked_cells → marked_cells.card ≥ 48) :=
by {
  intros,
  sorry
}

end min_cells_to_mark_l109_109452


namespace number_of_exchanges_l109_109184

theorem number_of_exchanges (n : ℕ) (hz_initial : ℕ) (hl_initial : ℕ) 
  (hz_decrease : ℕ) (hl_decrease : ℕ) (k : ℕ) :
  hz_initial = 200 →
  hl_initial = 20 →
  hz_decrease = 6 →
  hl_decrease = 1 →
  k = 11 →
  (hz_initial - n * hz_decrease) = k * (hl_initial - n * hl_decrease) →
  n = 4 := 
sorry

end number_of_exchanges_l109_109184


namespace bookstore_cost_effective_l109_109445

-- Bookstore A's pricing function
def storeA (x : ℝ) : ℝ := 0.8 * x

-- Bookstore B's pricing function
def storeB (x : ℝ) : ℝ :=
  if x ≤ 100 then x else 0.6 * x + 40

-- Cost-effectiveness comparison
def cost_effective (x : ℝ) : String :=
  if x < 200 then "Choose Bookstore A"
  else if x = 200 then "Costs are the same"
  else "Choose Bookstore B"

-- Proof of correctness of both functions and cost-effectiveness
theorem bookstore_cost_effective (x : ℝ) : cost_effective x :=
by 
  sorry

end bookstore_cost_effective_l109_109445


namespace deaths_during_operation_l109_109256

noncomputable def initial_count : ℕ := 1000
noncomputable def first_day_remaining (n : ℕ) := 5 * n / 6
noncomputable def second_day_remaining (n : ℕ) := (35 * n / 48) - 1
noncomputable def third_day_remaining (n : ℕ) := (105 * n / 192) - 3 / 4

theorem deaths_during_operation : ∃ n : ℕ, initial_count - n = 472 ∧ n = 528 :=
  by sorry

end deaths_during_operation_l109_109256


namespace cars_meeting_time_proof_l109_109645

-- Let's define some structures and theorems for our problem setup.
noncomputable def car_meeting_time
  (V_A V_B V_C V_D : ℝ) (circumference : ℝ) (t_ac t_bd t_ab : ℝ) : ℝ :=
  if h : (V_A + V_C = V_B + V_D) ∧ (t_ac = t_bd) ∧ (t_ac = 7) ∧ (circumference / |V_A - V_B| = t_ab) ∧ (t_ab = 53) then
    53  -- By the conditions of the problem, the cars C and D will meet at 53 minutes.
  else
    0  -- This should never happen with the given conditions.

-- Lean theorem statement for our translated problem
theorem cars_meeting_time_proof :
  ∀ (V_A V_B V_C V_D : ℝ) (circumference : ℝ),
    (V_A + V_C = V_B + V_D) →
    (circumference / |V_A - V_B| = 53) →
    car_meeting_time V_A V_B V_C V_D circumference 7 7 53 = 53 :=
by
  intros,
  -- proof steps are omitted here,
  -- as the prompt specifies only the statement is required
  sorry

end cars_meeting_time_proof_l109_109645


namespace number_of_cos_solutions_l109_109716

theorem number_of_cos_solutions : 
  (∃ x : ℝ, -90 ≤ x ∧ x ≤ 90 ∧ real.cos (x * real.pi / 180) = 0.5) → 
  (∃ a b : ℝ, a ≠ b ∧ a = 60 ∧ b = -60 ∧ real.cos (a * real.pi / 180) = 0.5 ∧ real.cos (b * real.pi / 180) = 0.5) :=
by
  sorry

end number_of_cos_solutions_l109_109716


namespace four_times_remaining_marbles_l109_109837

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end four_times_remaining_marbles_l109_109837


namespace average_age_in_club_l109_109011

theorem average_age_in_club (women men children : ℕ) 
    (avg_age_women avg_age_men avg_age_children : ℤ)
    (hw : women = 12) (hm : men = 18) (hc : children = 20)
    (haw : avg_age_women = 32) (ham : avg_age_men = 36) (hac : avg_age_children = 10) :
    (12 * 32 + 18 * 36 + 20 * 10) / (12 + 18 + 20) = 24 := by
  sorry

end average_age_in_club_l109_109011


namespace exists_min_value_f_l109_109660

def f (x y : ℝ) : ℝ := min x (x / (x^2 + y^2))

theorem exists_min_value_f :
  ∃ x0 y0 : ℝ, ∀ x y : ℝ, f x y ≤ f x0 y0 :=
by
  use 1, 0
  intros x y
  simp [f]
  split
  · exact min_le_left x (x / (x^2 + y^2))
  · exact min_le_right x (x / (x^2 + y^2))
  sorry

end exists_min_value_f_l109_109660


namespace each_persons_tip_l109_109420

theorem each_persons_tip
  (cost_julie cost_letitia cost_anton : ℕ)
  (H1 : cost_julie = 10)
  (H2 : cost_letitia = 20)
  (H3 : cost_anton = 30)
  (total_people : ℕ)
  (H4 : total_people = 3)
  (tip_percentage : ℝ)
  (H5 : tip_percentage = 0.20) :
  ∃ tip_per_person : ℝ, tip_per_person = 4 := 
by
  sorry

end each_persons_tip_l109_109420


namespace unnamed_racer_at_10th_l109_109732

-- Definitions of the positions of the racers
variables {Pos : Type} [linear_order Pos]
variables (Eda Simon Jacob Naomi Cal Iris : Pos)
variables (place10 : Pos)

-- Conditions of the problem
variables (racing_conditions :
  Jacob = Eda + 4 ∧
  Naomi = Simon + 1 ∧
  Jacob = Cal - 3 ∧
  Simon = Iris - 2 ∧
  Cal = Iris + 2 ∧
  Naomi = 7)

-- Question translated
theorem unnamed_racer_at_10th (h : racing_conditions) : 
  ∃ (Unnamed : Pos), place10 = Unnamed :=
by sorry

end unnamed_racer_at_10th_l109_109732


namespace count_fractions_l109_109970

theorem count_fractions :
  let numbers := [(-1 / 2 : ℚ), (7 / 10 : ℚ), (-9 : ℚ), (1 / 5 : ℚ), (-((Real.pi : ℚ) / 2)), (1 / 3 : ℚ)]
  in (numbers.filter (λ x, x ∈ ℚ)).length = 4 := 
by 
  sorry

end count_fractions_l109_109970


namespace initial_men_count_l109_109853

theorem initial_men_count 
  (M : ℕ)
  (h1 : 8 * M * 30 = (M + 77) * 6 * 50) :
  M = 63 :=
by
  sorry

end initial_men_count_l109_109853


namespace f_555_l109_109953

def linear_function (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x1 x2 y1 y2 z1 z2 a b c d e f',
  f(a * x1 + b * x2, c * y1 + d * y2, e * z1 + f' * z2)
  = a * c * e * f(x1, y1, z1) + a * c * f' * f(x1, y1, z2)
  + a * d * e * f(x1, y2, z1) + a * d * f' * f(x1, y2, z2)
  + b * c * e * f(x2, y1, z1) + b * c * f' * f(x2, y1, z2)
  + b * d * e * f(x2, y2, z1) + b * d * f' * f(x2, y2, z2)

axiom f : ℝ → ℝ → ℝ → ℝ
axiom linear_f : linear_function f
axiom f_values : ∀ (x y z : ℝ), (x = 3 ∨ x = 4) ∧ (y = 3 ∨ y = 4) ∧ (z = 3 ∨ z = 4) → f(x, y, z) = 1 / (x * y * z)

theorem f_555 : f 5 5 5 = 1 / 216 :=
  sorry

end f_555_l109_109953


namespace find_line_m_l109_109102

noncomputable def nine_circles := 
  {P : ℝ × ℝ | 
    ∃ (x y : ℤ), x ∈ set.Icc 0 3 ∧ y ∈ set.Icc 0 3 ∧ 
    (P.1 - (x:ℝ))^2 + (P.2 - (y:ℝ))^2 ≤ 1}

noncomputable def S := 
  {P : ℝ × ℝ | ∃ C ∈ nine_circles, dist P C ≤ 1}

noncomputable def line_divides_S (m : ℝ) (b : ℝ) : Prop :=
  ∃ A B : set (ℝ × ℝ), A ∪ B = S ∧ A ∩ B = ∅ ∧
  ∀ P ∈ A, m * P.1 + b ≤ P.2 ∧ ∀ P ∈ B, P.2 < m * P.1 + b

theorem find_line_m : 
  ∃ (a b c : ℤ), gcd a b c = 1 ∧ ∀ P ∈ S, P.2 = 2 * P.1 
  ∧ a^2 + b^2 + c^2 = 5 :=
begin
  sorry
end

end find_line_m_l109_109102


namespace effect_on_revenue_decrease_l109_109933

theorem effect_on_revenue_decrease (T C : ℝ) :
  let T_new := T * 0.81 in
  let C_new := C * 1.15 in
  let R := T / 100 * C in
  let R_new := T_new / 100 * C_new in
  let Effect_on_revenue := R_new - R in
  let Percentage_change_in_revenue := (Effect_on_revenue / R) * 100 in
  Percentage_change_in_revenue = -6.85 := by
  sorry

end effect_on_revenue_decrease_l109_109933


namespace mutually_exclusive_not_contradictory_l109_109303

inductive Color
| red
| black

noncomputable def bag : multiset Color := {Color.red, Color.red, Color.black, Color.black}.to_finset

noncomputable def draw_two_balls_from_bag := {s : finset (finset Color) | s.card = 2}

def event_exactly_one_black (s : finset Color) : Prop :=
s.filter (λ c, c = Color.black).card = 1

def event_exactly_two_blacks (s : finset Color) : Prop :=
s.filter (λ c, c = Color.black).card = 2

theorem mutually_exclusive_not_contradictory :
  ∀ s ∈ draw_two_balls_from_bag, event_exactly_one_black s ∨ event_exactly_two_blacks s → event_exactly_one_black s ↔ ¬ event_exactly_two_blacks s :=
by
  sorry

end mutually_exclusive_not_contradictory_l109_109303


namespace eq_zero_l109_109852

variable {x y z : ℤ}

theorem eq_zero (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end eq_zero_l109_109852


namespace total_spent_on_toys_l109_109231

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_spent : ℝ := 12.30

theorem total_spent_on_toys : football_cost + marbles_cost = total_spent :=
by sorry

end total_spent_on_toys_l109_109231


namespace nat_sum_representation_l109_109808

noncomputable def S (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  if k = 1 then 0 else (Finset.range (k - 1)).sum a

theorem nat_sum_representation (a : ℕ → ℕ) (h_seq : ∀ n m, n < m → a n > a m) :
  (∀ n, ∃ U : Finset ℕ, U.sum a = n ∧ U.pairwise (≠)) ↔
  (∀ k, a k ≤ S a k + 1) := 
sorry

end nat_sum_representation_l109_109808


namespace minimum_people_bought_most_popular_book_l109_109201

theorem minimum_people_bought_most_popular_book
  (n : ℕ)
  (num_people : ℕ)
  (book_matrix : Fin num_people → Fin 3 → Fin n)
  (common_book : ∀ i j : Fin num_people, i ≠ j → ∃ k l : Fin 3, book_matrix i k = book_matrix j l) :
  num_people = 510 → 
  ∀ x : ℕ, (∀ m : Fin n, m ∈ (Finset.univ.image (λ k, (Finset.univ.val.mk k : Fin num_people).choose (λ i j, x = (Finset.univ.val.mk k).count i)))) → x ≥ 5 :=
begin
  intros h_num_people h_x,
  sorry
end

end minimum_people_bought_most_popular_book_l109_109201


namespace largest_of_three_consecutive_odds_l109_109892

theorem largest_of_three_consecutive_odds (n : ℤ) (h_sum : n + (n + 2) + (n + 4) = -147) : n + 4 = -47 :=
by {
  -- Proof steps here, but we're skipping for this exercise
  sorry
}

end largest_of_three_consecutive_odds_l109_109892


namespace gcd_gx_x_is_450_l109_109336

def g (x : ℕ) : ℕ := (3 * x + 2) * (8 * x + 3) * (14 * x + 5) * (x + 15)

noncomputable def gcd_gx_x (x : ℕ) (h : 49356 ∣ x) : ℕ :=
  Nat.gcd (g x) x

theorem gcd_gx_x_is_450 (x : ℕ) (h : 49356 ∣ x) : gcd_gx_x x h = 450 := by
  sorry

end gcd_gx_x_is_450_l109_109336


namespace parabola_equation_l109_109282

/--
Given a point P (4, -2) on a parabola, prove that the equation of the parabola is either:
1) y^2 = x or
2) x^2 = -8y.
-/
theorem parabola_equation (p : ℝ) (x y : ℝ) (h1 : (4 : ℝ) = 4) (h2 : (-2 : ℝ) = -2) :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ 4 = 4 ∧ y = -2) ∨ (∃ p : ℝ, x^2 = 2 * p * y ∧ 4 = 4 ∧ x = 4) :=
sorry

end parabola_equation_l109_109282


namespace overlapping_region_area_l109_109100

theorem overlapping_region_area :
  ∀ (side_length : ℝ) (m n : ℕ),
  side_length = 12 → 
  let equilateral_height := (side_length * sqrt 3 / 2) in
  let equilateral_area := (sqrt 3 / 4) * (side_length^2) in
  let right_triangle_area := (side_length * equilateral_height / 2) in
  let overlapping_area := equilateral_area + 2 * right_triangle_area in
  overlapping_area = m * sqrt n →
  m = 108 ∧ n = 3 :=
by
  intros side_length m n h_side_length equilateral_height equilateral_area right_triangle_area overlapping_area h_area,
  sorry

end overlapping_region_area_l109_109100


namespace find_m_for_slope_45_degree_l109_109885

theorem find_m_for_slope_45_degree (m : ℝ) :
  let slope := (2 * m ^ 2 - 5 * m + 2) / (-(m ^ 2 - 4)) in
  slope = 1 → m = 3 :=
by
  intros slope h
  -- Insert proof here
  sorry

end find_m_for_slope_45_degree_l109_109885


namespace math_problem_l109_109141

theorem math_problem (A B C : ℕ) (h_pos : A > 0 ∧ B > 0 ∧ C > 0) (h_gcd : Nat.gcd (Nat.gcd A B) C = 1) (h_eq : A * Real.log 5 / Real.log 200 + B * Real.log 2 / Real.log 200 = C) : A + B + C = 6 :=
sorry

end math_problem_l109_109141


namespace ice_cream_flavors_l109_109713

theorem ice_cream_flavors (F : ℕ) (h1 : F / 4 + F / 2 + 25 = F) : F = 100 :=
by
  sorry

end ice_cream_flavors_l109_109713


namespace sum_of_distinct_terms_if_and_only_if_l109_109798

theorem sum_of_distinct_terms_if_and_only_if
  (a : ℕ → ℕ)
  (h_decreasing : ∀ k, a k > a (k + 1))
  (S : ℕ → ℕ)
  (h_S : ∀ k, S k = (∑ i in finset.range (k - 1), a i))
  (h_S1 : S 1 = 0) :
  (∀ n : ℕ, ∃ (l : list ℕ), (∀ (i : ℕ), i ∈ l → ∃ k, i = a k) ∧  l.sum = n) ↔ (∀ k, a k ≤ S k + 1) :=
begin
  sorry,
end

end sum_of_distinct_terms_if_and_only_if_l109_109798


namespace sum_of_possible_values_of_n_l109_109485

noncomputable def problem := {4, 7, 8, 12}

def median (s : Finset ℝ) : ℝ :=
  let sorted := s.sort
  if (sorted.card % 2 = 0) then
    (sorted.get (sorted.card / 2 - 1) + sorted.get (sorted.card / 2)) / 2
  else
    sorted.get (sorted.card / 2)

def mean (s : Finset ℝ) : ℝ :=
  s.sum / s.card

theorem sum_of_possible_values_of_n : (Finset.sum (Finset.filter (λ n, 
  median (problem ∪ {n}) = mean (problem ∪ {n})) (Finset.range 100))) = 9 :=
sorry

end sum_of_possible_values_of_n_l109_109485


namespace juliet_supporter_in_capulet_probability_l109_109549

/-- Theorem: The probability that a Juliet supporter chosen at random resides in Capulet
is 33%, given the population distribution and support percentages in the provinces of Venezia. -/
theorem juliet_supporter_in_capulet_probability :
  ∀ (P : ℕ),
    let montague_population := (4 / 6 : ℚ) * P,
        capulet_population := (1 / 6 : ℚ) * P,
        verona_population := (1 / 6 : ℚ) * P,
        juliet_montague := 0.2 * montague_population,
        juliet_capulet := 0.7 * capulet_population,
        juliet_verona := 0.6 * verona_population,
        total_juliet := juliet_montague + juliet_capulet + juliet_verona in
    round ((juliet_capulet / total_juliet) * 100) = 33 :=
begin
  sorry
end

end juliet_supporter_in_capulet_probability_l109_109549


namespace purchase_ingredients_A_B_1_minimize_total_cost_A_find_m_A_less_than_250_l109_109592

-- Part (1)
theorem purchase_ingredients_A_B_1 (x : ℕ) (h1 : x < 300) (h2 : 5 * (600 - x) + 9 * x = 3800) : 
  (600 - x) = 400 ∧ x = 200 :=
sorry

-- Part (2)
theorem minimize_total_cost_A (A B : ℕ) (x : ℕ) 
  (h1 : x > 300) 
  (h2 : 600 - x ≥ x / 2) 
  (h3 : x % 10 = 0) :
  600 - x = 200 ∧ -0.01 * x^2 + 7 * x + 3000 = 4200 :=
sorry

-- Part (3)
theorem find_m_A_less_than_250 (m x : ℕ) 
  (h1 : x > 300) 
  (h2 : m < 250) 
  (h3 : -0.01 * x^2 + 7 * x + 3000 = 4000) :
  m = 100 :=
sorry

end purchase_ingredients_A_B_1_minimize_total_cost_A_find_m_A_less_than_250_l109_109592


namespace value_of_x_find_x_value_l109_109749

theorem value_of_x (perimeter_square height_triangle : ℝ) (equal_areas : bool) : ℝ :=
  let side_square := perimeter_square / 4
  let area_square := side_square * side_square
  let area_triangle := (1 / 2) * height_triangle * (area_square / 30)
  if equal_areas then
    area_square / 30
  else 0

theorem find_x_value (perimeter_square height_triangle : ℝ) (equal_areas : bool) (h1 : perimeter_square = 40) (h2 : height_triangle = 60) (h3 : equal_areas = tt) : value_of_x perimeter_square height_triangle equal_areas = 10 / 3 := by
  rw [h1, h2, h3]
  dsimp [value_of_x]
  norm_num
  sorry

end value_of_x_find_x_value_l109_109749


namespace initial_water_amount_l109_109946

theorem initial_water_amount (W : ℝ) (h1 : ∀ t, t = 50 -> 0.008 * t = 0.4) (h2 : 0.04 * W = 0.4) : W = 10 :=
by
  sorry

end initial_water_amount_l109_109946


namespace circulation_zero_stokes_theorem_zero_l109_109598

variables {ρ φ z : ℝ}
variables (t : ℝ)

def vec_A (ρ φ z : ℝ) : (ℝ × ℝ × ℝ) := (ρ * sin φ, ρ * z, ρ^3)

def curve_L (t : ℝ) : (ℝ × ℝ × ℝ) := (sin t, 0, t)

def A_dot_dr {ρ φ z : ℝ} (t : ℝ) : ℝ := 
  let (Aρ, Aφ, Az) := vec_A ρ φ z in
  Aρ * derivative (λ t, sin t) t + Aφ * derivative (λ t, t) t + Az * derivative (λ t, 0) t

def stokes_theorem {A : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ} {S : ℝ × ℝ × ℝ → Prop} 
    (v : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ → Prop) :=
  ∫ (x : ℝ × ℝ × ℝ), (A x) • v x ∂μ = ∫ (x : ℝ × ℝ × ℝ) in S, (curl f (A x)) • normal_vector C x ∂μ

theorem circulation_zero : 
  ∫ (t : ℝ) in 0..π, A_dot_dr t ∂μ = 0 :=
sorry  -- proof not required as per the instructions

theorem stokes_theorem_zero : 
  stokes_theorem vec_A (λ (x : ℝ × ℝ × ℝ), x.2 = 0 ∧ 0 ≤ x.1 ∧ x.1 ≤ sin π) (0, 0, 1) = 0 :=
sorry  -- proof not required as per the instructions

end circulation_zero_stokes_theorem_zero_l109_109598


namespace common_ratio_geom_seq_l109_109030

variable {a : ℕ → ℝ} {q : ℝ}

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a n = a 0 * q ^ n

theorem common_ratio_geom_seq (h₁ : a 5 = 1) (h₂ : a 8 = 8) (hq : geom_seq a q) : q = 2 :=
by
  sorry

end common_ratio_geom_seq_l109_109030
