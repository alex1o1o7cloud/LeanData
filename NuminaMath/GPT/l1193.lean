import Mathlib

namespace NUMINAMATH_GPT_smallest_positive_integer_divides_l1193_119324

theorem smallest_positive_integer_divides (m : ℕ) : 
  (∀ z : ℂ, z ≠ 0 → (z^11 + z^10 + z^8 + z^7 + z^5 + z^4 + z^2 + 1) ∣ (z^m - 1)) →
  (m = 88) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_divides_l1193_119324


namespace NUMINAMATH_GPT_find_Q_l1193_119345

-- We define the circles and their centers
def circle1 (x y r : ℝ) : Prop := (x + 1) ^ 2 + (y - 1) ^ 2 = r ^ 2
def circle2 (x y R : ℝ) : Prop := (x - 2) ^ 2 + (y + 2) ^ 2 = R ^ 2

-- Coordinates of point P
def P : ℝ × ℝ := (1, 2)

-- Defining the symmetry about the line y = -x
def symmetric_about (p q : ℝ × ℝ) : Prop := p.1 = -q.2 ∧ p.2 = -q.1

-- Theorem stating that if P is (1, 2), Q should be (-2, -1)
theorem find_Q {r R : ℝ} (h1 : circle1 1 2 r) (h2 : circle2 1 2 R) (hP : P = (1, 2)) :
  ∃ Q : ℝ × ℝ, symmetric_about P Q ∧ Q = (-2, -1) :=
by
  sorry

end NUMINAMATH_GPT_find_Q_l1193_119345


namespace NUMINAMATH_GPT_rational_sqrts_l1193_119365

def is_rational (n : ℝ) : Prop := ∃ (q : ℚ), n = q

theorem rational_sqrts 
  (x y z : ℝ) 
  (hxr : is_rational x) 
  (hyr : is_rational y) 
  (hzr : is_rational z)
  (hw : is_rational (Real.sqrt x + Real.sqrt y + Real.sqrt z)) :
  is_rational (Real.sqrt x) ∧ is_rational (Real.sqrt y) ∧ is_rational (Real.sqrt z) :=
sorry

end NUMINAMATH_GPT_rational_sqrts_l1193_119365


namespace NUMINAMATH_GPT_sum_other_y_coordinates_l1193_119340

-- Given points
structure Point where
  x : ℝ
  y : ℝ

def opposite_vertices (p1 p2 : Point) : Prop :=
  -- conditions defining opposite vertices of a rectangle
  (p1.x ≠ p2.x) ∧ (p1.y ≠ p2.y)

-- Function to sum y-coordinates of two points
def sum_y_coords (p1 p2 : Point) : ℝ :=
  p1.y + p2.y

-- Main theorem to prove
theorem sum_other_y_coordinates (p1 p2 : Point) (h : opposite_vertices p1 p2) :
  sum_y_coords p1 p2 = 11 ↔ 
  (p1 = {x := 1, y := 19} ∨ p1 = {x := 7, y := -8}) ∧ 
  (p2 = {x := 1, y := 19} ∨ p2 = {x := 7, y := -8}) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_other_y_coordinates_l1193_119340


namespace NUMINAMATH_GPT_geometric_series_sum_l1193_119323

variable (a r : ℤ) (n : ℕ) 

theorem geometric_series_sum :
  a = -1 ∧ r = 2 ∧ n = 10 →
  (a * (r^n - 1) / (r - 1)) = -1023 := 
by
  intro h
  rcases h with ⟨ha, hr, hn⟩
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1193_119323


namespace NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l1193_119334

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := 
by
  sorry

end NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l1193_119334


namespace NUMINAMATH_GPT_evaluate_expression_equals_128_l1193_119308

-- Define the expression as a Lean function
def expression : ℕ := (8^6) / (4 * 8^3)

-- Theorem stating that the expression equals 128
theorem evaluate_expression_equals_128 : expression = 128 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_equals_128_l1193_119308


namespace NUMINAMATH_GPT_alfred_gain_percent_l1193_119304

theorem alfred_gain_percent (P : ℝ) (R : ℝ) (S : ℝ) (H1 : P = 4700) (H2 : R = 800) (H3 : S = 6000) : 
  (S - (P + R)) / (P + R) * 100 = 9.09 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end NUMINAMATH_GPT_alfred_gain_percent_l1193_119304


namespace NUMINAMATH_GPT_intersection_A_B_l1193_119388

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | ∃ (n : ℤ), (x : ℝ) = n }

theorem intersection_A_B : A ∩ B = {0, 1} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1193_119388


namespace NUMINAMATH_GPT_final_replacement_weight_l1193_119383

theorem final_replacement_weight (W : ℝ) (a b c d e : ℝ) 
  (h1 : a = W / 10)
  (h2 : b = (W - 70 + e) / 10)
  (h3 : b - a = 4)
  (h4 : c = (W - 70 + e - 110 + d) / 10)
  (h5 : c - b = -2)
  (h6 : d = (W - 70 + e - 110 + d + 140 - 90) / 10)
  (h7 : d - c = 5)
  : e = 110 ∧ d = 90 ∧ 140 = e + 50 := sorry

end NUMINAMATH_GPT_final_replacement_weight_l1193_119383


namespace NUMINAMATH_GPT_prove_b_div_c_equals_one_l1193_119378

theorem prove_b_div_c_equals_one
  (a b c d : ℕ)
  (h_a : a > 0 ∧ a < 4)
  (h_b : b > 0 ∧ b < 4)
  (h_c : c > 0 ∧ c < 4)
  (h_d : d > 0 ∧ d < 4)
  (h_eq : 4^a + 3^b + 2^c + 1^d = 78) :
  b / c = 1 :=
by
  sorry

end NUMINAMATH_GPT_prove_b_div_c_equals_one_l1193_119378


namespace NUMINAMATH_GPT_irreducible_fraction_l1193_119318

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by sorry

end NUMINAMATH_GPT_irreducible_fraction_l1193_119318


namespace NUMINAMATH_GPT_developer_lots_l1193_119361

theorem developer_lots (acres : ℕ) (cost_per_acre : ℕ) (lot_price : ℕ) 
  (h1 : acres = 4) 
  (h2 : cost_per_acre = 1863) 
  (h3 : lot_price = 828) : 
  ((acres * cost_per_acre) / lot_price) = 9 := 
  by
    sorry

end NUMINAMATH_GPT_developer_lots_l1193_119361


namespace NUMINAMATH_GPT_sandy_paid_cost_shop2_l1193_119386

-- Define the conditions
def books_shop1 : ℕ := 65
def cost_shop1 : ℕ := 1380
def books_shop2 : ℕ := 55
def avg_price_per_book : ℕ := 19

-- Calculation of the total amount Sandy paid for the books from the second shop
def cost_shop2 (total_books: ℕ) (avg_price: ℕ) (cost1: ℕ) : ℕ :=
  (total_books * avg_price) - cost1

-- Define the theorem we want to prove
theorem sandy_paid_cost_shop2 : cost_shop2 (books_shop1 + books_shop2) avg_price_per_book cost_shop1 = 900 :=
sorry

end NUMINAMATH_GPT_sandy_paid_cost_shop2_l1193_119386


namespace NUMINAMATH_GPT_students_not_enrolled_in_either_l1193_119352

-- Definitions based on conditions
def total_students : ℕ := 120
def french_students : ℕ := 65
def german_students : ℕ := 50
def both_courses_students : ℕ := 25

-- The proof statement
theorem students_not_enrolled_in_either : total_students - (french_students + german_students - both_courses_students) = 30 := by
  sorry

end NUMINAMATH_GPT_students_not_enrolled_in_either_l1193_119352


namespace NUMINAMATH_GPT_intersecting_chords_ratio_l1193_119336

theorem intersecting_chords_ratio {XO YO WO ZO : ℝ} 
    (hXO : XO = 5) 
    (hWO : WO = 7) 
    (h_power_of_point : XO * YO = WO * ZO) : 
    ZO / YO = 5 / 7 :=
by
    rw [hXO, hWO] at h_power_of_point
    sorry

end NUMINAMATH_GPT_intersecting_chords_ratio_l1193_119336


namespace NUMINAMATH_GPT_cannot_determine_right_triangle_l1193_119368

-- Definitions of conditions
variables {a b c : ℕ}
variables {angle_A angle_B angle_C : ℕ}

-- Context for the proof
def is_right_angled_triangle_via_sides (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def triangle_angle_sum_theorem (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Statements for conditions as used in the problem
def condition_A (a2 b2 c2 : ℕ) : Prop :=
  a2 = 1 ∧ b2 = 2 ∧ c2 = 3

def condition_B (a b c : ℕ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5

def condition_C (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A + angle_B = angle_C

def condition_D (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A = 45 ∧ angle_B = 60 ∧ angle_C = 75

-- Proof statement
theorem cannot_determine_right_triangle (a b c angle_A angle_B angle_C : ℕ) :
  condition_D angle_A angle_B angle_C →
  ¬(is_right_angled_triangle_via_sides a b c) :=
sorry

end NUMINAMATH_GPT_cannot_determine_right_triangle_l1193_119368


namespace NUMINAMATH_GPT_value_multiplied_by_15_l1193_119370

theorem value_multiplied_by_15 (x : ℝ) (h : 3.6 * x = 10.08) : x * 15 = 42 :=
sorry

end NUMINAMATH_GPT_value_multiplied_by_15_l1193_119370


namespace NUMINAMATH_GPT_oranges_difference_l1193_119310

-- Defining the number of sacks of ripe and unripe oranges
def sacks_ripe_oranges := 44
def sacks_unripe_oranges := 25

-- The statement to be proven
theorem oranges_difference : sacks_ripe_oranges - sacks_unripe_oranges = 19 :=
by
  -- Provide the exact calculation and result expected
  sorry

end NUMINAMATH_GPT_oranges_difference_l1193_119310


namespace NUMINAMATH_GPT_stock_price_drop_l1193_119348

theorem stock_price_drop (P : ℝ) (h1 : P > 0) (x : ℝ)
  (h3 : (1.30 * (1 - x/100) * 1.20 * P) = 1.17 * P) :
  x = 25 :=
by
  sorry

end NUMINAMATH_GPT_stock_price_drop_l1193_119348


namespace NUMINAMATH_GPT_probability_green_or_yellow_l1193_119329

def green_faces : ℕ := 3
def yellow_faces : ℕ := 2
def blue_faces : ℕ := 1
def total_faces : ℕ := 6

theorem probability_green_or_yellow : 
  (green_faces + yellow_faces) / total_faces = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_green_or_yellow_l1193_119329


namespace NUMINAMATH_GPT_binomial_expansion_five_l1193_119305

open Finset

theorem binomial_expansion_five (a b : ℝ) : 
  (a + b)^5 = a^5 + 5 * a^4 * b + 10 * a^3 * b^2 + 10 * a^2 * b^3 + 5 * a * b^4 + b^5 := 
by sorry

end NUMINAMATH_GPT_binomial_expansion_five_l1193_119305


namespace NUMINAMATH_GPT_find_x_l1193_119373

theorem find_x :
  ∃ x : ℕ, (5 * 12) / (x / 3) + 80 = 81 ∧ x = 180 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1193_119373


namespace NUMINAMATH_GPT_simplify_fraction_l1193_119376

/-- Given the numbers 180 and 270, prove that 180 / 270 is equal to 2 / 3 -/
theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := 
sorry

end NUMINAMATH_GPT_simplify_fraction_l1193_119376


namespace NUMINAMATH_GPT_smallest_prime_dividing_4_pow_11_plus_6_pow_13_l1193_119306

-- Definition of the problem
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem smallest_prime_dividing_4_pow_11_plus_6_pow_13 :
  ∃ p : ℕ, is_prime p ∧ p ∣ (4^11 + 6^13) ∧ ∀ q : ℕ, is_prime q ∧ q ∣ (4^11 + 6^13) → p ≤ q :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_prime_dividing_4_pow_11_plus_6_pow_13_l1193_119306


namespace NUMINAMATH_GPT_answer_l1193_119350

-- Definitions of geometric entities in terms of vectors
structure Square :=
  (A B C D E : ℝ × ℝ)
  (side_length : ℝ)
  (hAB_eq : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = side_length ^ 2)
  (hBC_eq : (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = side_length ^ 2)
  (hCD_eq : (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = side_length ^ 2)
  (hDA_eq : (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 = side_length ^ 2)
  (hE_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def EC_ED_dot_product (s : Square) : ℝ :=
  let EC := (s.C.1 - s.E.1, s.C.2 - s.E.2)
  let ED := (s.D.1 - s.E.1, s.D.2 - s.E.2)
  dot_product EC ED

theorem answer (s : Square) (h_side_length : s.side_length = 2) :
  EC_ED_dot_product s = 3 :=
sorry

end NUMINAMATH_GPT_answer_l1193_119350


namespace NUMINAMATH_GPT_priya_speed_l1193_119311

theorem priya_speed (Riya_speed Priya_speed : ℝ) (time_separation distance_separation : ℝ)
  (h1 : Riya_speed = 30) 
  (h2 : time_separation = 45 / 60) -- 45 minutes converted to hours
  (h3 : distance_separation = 60)
  : Priya_speed = 50 :=
sorry

end NUMINAMATH_GPT_priya_speed_l1193_119311


namespace NUMINAMATH_GPT_avery_donation_clothes_l1193_119316

theorem avery_donation_clothes :
  let shirts := 4
  let pants := 2 * shirts
  let shorts := pants / 2
  shirts + pants + shorts = 16 :=
by
  let shirts := 4
  let pants := 2 * shirts
  let shorts := pants / 2
  show shirts + pants + shorts = 16
  sorry

end NUMINAMATH_GPT_avery_donation_clothes_l1193_119316


namespace NUMINAMATH_GPT_car_B_speed_is_50_l1193_119372

def car_speeds (v_A v_B : ℕ) (d_init d_ahead t : ℝ) : Prop :=
  v_A * t = v_B * t + d_init + d_ahead

theorem car_B_speed_is_50 :
  car_speeds 58 50 10 8 2.25 :=
by
  sorry

end NUMINAMATH_GPT_car_B_speed_is_50_l1193_119372


namespace NUMINAMATH_GPT_smallest_solution_l1193_119395

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_smallest_solution_l1193_119395


namespace NUMINAMATH_GPT_prove_equations_and_PA_PB_l1193_119333

noncomputable def curve_C1_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

noncomputable def curve_C2_polar (ρ θ : ℝ) : Prop :=
  ρ + 7 / ρ = 4 * Real.cos θ + 4 * Real.sin θ

theorem prove_equations_and_PA_PB :
  (∀ (α : ℝ), 0 ≤ α ∧ α < π → 
    (∃ (C1_cart : ℝ → ℝ → Prop), ∀ x y, C1_cart x y ↔ x^2 = 4 * y) ∧
    (∃ (C1_polar : ℝ → ℝ → Prop), ∀ ρ θ, C1_polar ρ θ ↔ ρ^2 * Real.cos θ^2 = 4 * ρ * Real.sin θ) ∧
    (∃ (C2_cart : ℝ → ℝ → Prop), ∀ x y, C2_cart x y ↔ (x - 2)^2 + (y - 2)^2 = 1)) ∧
  (∃ (P A B : ℝ × ℝ), P = (0, 1) ∧ 
    curve_C1_parametric t (Real.pi / 2) = A ∧ 
    curve_C1_parametric t (Real.pi / 2) = B ∧ 
    |P - A| * |P - B| = 4) :=
sorry

end NUMINAMATH_GPT_prove_equations_and_PA_PB_l1193_119333


namespace NUMINAMATH_GPT_average_age_of_10_students_l1193_119312

theorem average_age_of_10_students
  (avg_age_25_students : ℕ)
  (num_students_25 : ℕ)
  (avg_age_14_students : ℕ)
  (num_students_14 : ℕ)
  (age_25th_student : ℕ)
  (avg_age_10_students : ℕ)
  (h_avg_age_25 : avg_age_25_students = 25)
  (h_num_students_25 : num_students_25 = 25)
  (h_avg_age_14 : avg_age_14_students = 28)
  (h_num_students_14 : num_students_14 = 14)
  (h_age_25th : age_25th_student = 13)
  : avg_age_10_students = 22 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_10_students_l1193_119312


namespace NUMINAMATH_GPT_max_area_right_triangle_in_semicircle_l1193_119346

theorem max_area_right_triangle_in_semicircle :
  ∀ (r : ℝ), r = 1/2 → 
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ y > 0 ∧ 
  (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 ∧ y' > 0 → (1/2) * x * y ≥ (1/2) * x' * y') ∧ 
  (1/2) * x * y = 3 * Real.sqrt 3 / 32 := 
sorry

end NUMINAMATH_GPT_max_area_right_triangle_in_semicircle_l1193_119346


namespace NUMINAMATH_GPT_equation_of_parabola_max_slope_OQ_l1193_119321

section parabola

variable (p : ℝ)
variable (y : ℝ) (x : ℝ)
variable (n : ℝ) (m : ℝ)

-- Condition: p > 0 and distance from focus F to directrix being 2
axiom positive_p : p > 0
axiom distance_focus_directrix : ∀ {F : ℝ}, F = 2 * p → 2 * p = 2

-- Prove these two statements
theorem equation_of_parabola : (y^2 = 4 * x) :=
  sorry

theorem max_slope_OQ : (∃ K : ℝ, K = 1 / 3) :=
  sorry

end parabola

end NUMINAMATH_GPT_equation_of_parabola_max_slope_OQ_l1193_119321


namespace NUMINAMATH_GPT_fraction_calculation_correct_l1193_119397

noncomputable def calculate_fraction : ℚ :=
  let numerator := (1 / 2) - (1 / 3)
  let denominator := (3 / 4) + (1 / 8)
  numerator / denominator

theorem fraction_calculation_correct : calculate_fraction = 4 / 21 := 
  by
    sorry

end NUMINAMATH_GPT_fraction_calculation_correct_l1193_119397


namespace NUMINAMATH_GPT_possible_number_of_students_l1193_119301

theorem possible_number_of_students (n : ℕ) 
  (h1 : n ≥ 1) 
  (h2 : ∃ k : ℕ, 120 = 2 * n + 2 * k) :
  n = 58 ∨ n = 60 :=
sorry

end NUMINAMATH_GPT_possible_number_of_students_l1193_119301


namespace NUMINAMATH_GPT_matrix_pow_expression_l1193_119366

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 4, 2; 0, 2, 3; 0, 0, 1]
def I : Matrix (Fin 3) (Fin 3) ℝ := 1

theorem matrix_pow_expression :
  A^5 - 3 • A^4 = !![0, 4, 2; 0, -1, 3; 0, 0, -2] := by
  sorry

end NUMINAMATH_GPT_matrix_pow_expression_l1193_119366


namespace NUMINAMATH_GPT_remainder_p11_minus_3_div_p_minus_2_l1193_119327

def f (p : ℕ) : ℕ := p^11 - 3

theorem remainder_p11_minus_3_div_p_minus_2 : f 2 = 2045 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_p11_minus_3_div_p_minus_2_l1193_119327


namespace NUMINAMATH_GPT_part_a_part_b_l1193_119317

-- Problem (a)
theorem part_a :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f x ≠ 0) ∧ (∀ x, 2 * f (f x) = f x ∧ f x ≥ 0) ∧ Differentiable ℝ f :=
sorry

-- Problem (b)
theorem part_b :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f x ≠ 0) ∧ (∀ x, -1 ≤ 2 * f (f x) ∧ 2 * f (f x) = f x ∧ f x ≤ 1) ∧ Differentiable ℝ f :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1193_119317


namespace NUMINAMATH_GPT_parabola_zero_sum_l1193_119359

-- Define the original parabola equation and transformations
def original_parabola (x : ℝ) : ℝ := (x - 3) ^ 2 + 4

-- Define the resulting parabola after transformations
def transformed_parabola (x : ℝ) : ℝ := -(x - 7) ^ 2 + 1

-- Prove that the resulting parabola has zeros at p and q such that p + q = 14
theorem parabola_zero_sum : 
  ∃ (p q : ℝ), transformed_parabola p = 0 ∧ transformed_parabola q = 0 ∧ p + q = 14 :=
by
  sorry

end NUMINAMATH_GPT_parabola_zero_sum_l1193_119359


namespace NUMINAMATH_GPT_geometric_sequence_when_k_is_neg_one_l1193_119335

noncomputable def S (n : ℕ) (k : ℝ) : ℝ := 3^n + k

noncomputable def a (n : ℕ) (k : ℝ) : ℝ :=
  if n = 1 then S 1 k else S n k - S (n-1) k

theorem geometric_sequence_when_k_is_neg_one :
  ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, ∀ m : ℕ, m ≥ 1 → a m (-1) = a 1 (-1) * r^(m-1) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_when_k_is_neg_one_l1193_119335


namespace NUMINAMATH_GPT_triangle_groups_count_l1193_119379

theorem triangle_groups_count (total_points collinear_groups groups_of_three total_combinations : ℕ)
    (h1 : total_points = 12)
    (h2 : collinear_groups = 16)
    (h3 : groups_of_three = (total_points.choose 3))
    (h4 : total_combinations = groups_of_three - collinear_groups) :
    total_combinations = 204 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_triangle_groups_count_l1193_119379


namespace NUMINAMATH_GPT_ratio_M_N_l1193_119353

variable {R P M N : ℝ}

theorem ratio_M_N (h1 : P = 0.3 * R) (h2 : M = 0.35 * R) (h3 : N = 0.55 * R) : M / N = 7 / 11 := by
  sorry

end NUMINAMATH_GPT_ratio_M_N_l1193_119353


namespace NUMINAMATH_GPT_sum_of_consecutive_evens_is_162_l1193_119358

-- Define the smallest even number
def smallest_even : ℕ := 52

-- Define the next two consecutive even numbers
def second_even : ℕ := smallest_even + 2
def third_even : ℕ := smallest_even + 4

-- The sum of these three even numbers
def sum_of_consecutive_evens : ℕ := smallest_even + second_even + third_even

-- Assertion that the sum must be 162
theorem sum_of_consecutive_evens_is_162 : sum_of_consecutive_evens = 162 :=
by 
  -- To be proved
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_evens_is_162_l1193_119358


namespace NUMINAMATH_GPT_sequence_value_l1193_119343

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 3) ∧ (11 - 5 = 6) ∧ (20 - 11 = 9) ∧ (x - 20 = 12) → x = 32 := 
by intros; sorry

end NUMINAMATH_GPT_sequence_value_l1193_119343


namespace NUMINAMATH_GPT_two_x_plus_two_y_value_l1193_119392

theorem two_x_plus_two_y_value (x y : ℝ) (h1 : x^2 - y^2 = 8) (h2 : x - y = 6) : 2 * x + 2 * y = 8 / 3 := 
by sorry

end NUMINAMATH_GPT_two_x_plus_two_y_value_l1193_119392


namespace NUMINAMATH_GPT_math_problem_l1193_119393

theorem math_problem : 
  ( - (1 / 12 : ℚ) + (1 / 3 : ℚ) - (1 / 2 : ℚ) ) / ( - (1 / 18 : ℚ) ) = 4.5 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1193_119393


namespace NUMINAMATH_GPT_jenny_reading_time_l1193_119398

theorem jenny_reading_time 
  (days : ℕ)
  (words_first_book : ℕ)
  (words_second_book : ℕ)
  (words_third_book : ℕ)
  (reading_speed : ℕ) : 
  days = 10 →
  words_first_book = 200 →
  words_second_book = 400 →
  words_third_book = 300 →
  reading_speed = 100 →
  (words_first_book + words_second_book + words_third_book) / reading_speed / days * 60 = 54 :=
by
  intros hdays hwords1 hwords2 hwords3 hspeed
  rw [hdays, hwords1, hwords2, hwords3, hspeed]
  norm_num
  sorry

end NUMINAMATH_GPT_jenny_reading_time_l1193_119398


namespace NUMINAMATH_GPT_Bill_Sunday_miles_l1193_119391

-- Definitions based on problem conditions
def Bill_Saturday (B : ℕ) : ℕ := B
def Bill_Sunday (B : ℕ) : ℕ := B + 4
def Julia_Sunday (B : ℕ) : ℕ := 2 * (B + 4)
def Alex_Total (B : ℕ) : ℕ := B + 2

-- Total miles equation based on conditions
def total_miles (B : ℕ) : ℕ := Bill_Saturday B + Bill_Sunday B + Julia_Sunday B + Alex_Total B

-- Proof statement
theorem Bill_Sunday_miles (B : ℕ) (h : total_miles B = 54) : Bill_Sunday B = 14 :=
by {
  -- calculations and proof would go here if not omitted
  sorry
}

end NUMINAMATH_GPT_Bill_Sunday_miles_l1193_119391


namespace NUMINAMATH_GPT_balance_relationship_l1193_119363

theorem balance_relationship (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 200 - 36 * x := 
sorry

end NUMINAMATH_GPT_balance_relationship_l1193_119363


namespace NUMINAMATH_GPT_not_p_and_p_l1193_119389

theorem not_p_and_p (p : Prop) : ¬ (p ∧ ¬ p) :=
by 
  sorry

end NUMINAMATH_GPT_not_p_and_p_l1193_119389


namespace NUMINAMATH_GPT_binom_18_4_l1193_119356

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end NUMINAMATH_GPT_binom_18_4_l1193_119356


namespace NUMINAMATH_GPT_mr_lee_broke_even_l1193_119382

theorem mr_lee_broke_even (sp1 sp2 : ℝ) (p1_loss2 : ℝ) (c1 c2 : ℝ) (h1 : sp1 = 1.50) (h2 : sp2 = 1.50) 
    (h3 : c1 = sp1 / 1.25) (h4 : c2 = sp2 / 0.8333) (h5 : p1_loss2 = (sp1 - c1) + (sp2 - c2)) : 
  p1_loss2 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_mr_lee_broke_even_l1193_119382


namespace NUMINAMATH_GPT_max_trig_expression_l1193_119349

theorem max_trig_expression (A : ℝ) : (2 * Real.sin (A / 2) + Real.cos (A / 2) ≤ Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_max_trig_expression_l1193_119349


namespace NUMINAMATH_GPT_x_intercept_is_correct_l1193_119315

-- Define the original line equation
def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

-- Define the perpendicular line's y-intercept
def y_intercept (y : ℝ) : Prop := y = -3

-- Define the equation of the perpendicular line in slope-intercept form
def perpendicular_line (x y : ℝ) : Prop := y = (5 / 4) * x + -3

-- Prove that the x-intercept of the perpendicular line is 12/5
theorem x_intercept_is_correct : ∃ x : ℝ, x ≠ 0 ∧ (∃ y : ℝ, y = 0) ∧ (perpendicular_line x y) :=
sorry

end NUMINAMATH_GPT_x_intercept_is_correct_l1193_119315


namespace NUMINAMATH_GPT_tenth_term_in_sequence_l1193_119390

def seq (n : ℕ) : ℚ :=
  (-1) ^ (n + 1) * ((2 * n - 1) / (n ^ 2 + 1))

theorem tenth_term_in_sequence :
  seq 10 = -19 / 101 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_tenth_term_in_sequence_l1193_119390


namespace NUMINAMATH_GPT_chengdu_gdp_scientific_notation_l1193_119354

theorem chengdu_gdp_scientific_notation :
  15000 = 1.5 * 10^4 :=
sorry

end NUMINAMATH_GPT_chengdu_gdp_scientific_notation_l1193_119354


namespace NUMINAMATH_GPT_income_calculation_l1193_119314

theorem income_calculation
  (x : ℕ)
  (income : ℕ := 5 * x)
  (expenditure : ℕ := 4 * x)
  (savings : ℕ := income - expenditure)
  (savings_eq : savings = 3000) :
  income = 15000 :=
sorry

end NUMINAMATH_GPT_income_calculation_l1193_119314


namespace NUMINAMATH_GPT_modulus_remainder_l1193_119319

theorem modulus_remainder (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_modulus_remainder_l1193_119319


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1193_119357

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem statement
theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ∈ P → x ∈ Q) ∧ (¬(x ∈ Q → x ∈ P)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1193_119357


namespace NUMINAMATH_GPT_probability_range_l1193_119338

theorem probability_range (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1)
  (h3 : (4 * p * (1 - p)^3) ≤ (6 * p^2 * (1 - p)^2)) : 
  2 / 5 ≤ p ∧ p ≤ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_range_l1193_119338


namespace NUMINAMATH_GPT_milan_total_minutes_l1193_119325

-- Conditions
variables (x : ℝ) -- minutes on the second phone line
variables (minutes_first : ℝ := x + 20) -- minutes on the first phone line
def total_cost (x : ℝ) := 3 + 0.15 * (x + 20) + 4 + 0.10 * x

-- Statement to prove
theorem milan_total_minutes (x : ℝ) (h : total_cost x = 56) :
  x + (x + 20) = 252 :=
sorry

end NUMINAMATH_GPT_milan_total_minutes_l1193_119325


namespace NUMINAMATH_GPT_point_on_imaginary_axis_point_in_fourth_quadrant_l1193_119331

-- (I) For what value(s) of the real number m is the point A on the imaginary axis?
theorem point_on_imaginary_axis (m : ℝ) :
  m^2 - 8 * m + 15 = 0 ∧ m^2 + m - 12 ≠ 0 ↔ m = 5 := sorry

-- (II) For what value(s) of the real number m is the point A located in the fourth quadrant?
theorem point_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8 * m + 15 > 0 ∧ m^2 + m - 12 < 0) ↔ -4 < m ∧ m < 3 := sorry

end NUMINAMATH_GPT_point_on_imaginary_axis_point_in_fourth_quadrant_l1193_119331


namespace NUMINAMATH_GPT_exists_sphere_tangent_to_lines_l1193_119341

variables
  (A B C D K L M N : Point)
  (AB BC CD DA : Line)
  (sphere : Sphere)

-- Given conditions
def AN_eq_AK : AN = AK := sorry
def BK_eq_BL : BK = BL := sorry
def CL_eq_CM : CL = CM := sorry
def DM_eq_DN : DM = DN := sorry
def sphere_tangent (s : Sphere) (l : Line) : Prop := sorry -- define tangency condition

-- Problem statement
theorem exists_sphere_tangent_to_lines :
  ∃ S : Sphere, 
    sphere_tangent S AB ∧
    sphere_tangent S BC ∧
    sphere_tangent S CD ∧
    sphere_tangent S DA := sorry

end NUMINAMATH_GPT_exists_sphere_tangent_to_lines_l1193_119341


namespace NUMINAMATH_GPT_tetrahedron_side_length_l1193_119385

theorem tetrahedron_side_length (s : ℝ) (area : ℝ) (d : ℝ) :
  area = 16 → s^2 = area → d = s * Real.sqrt 2 → 4 * Real.sqrt 2 = d :=
by
  intros _ h1 h2
  sorry

end NUMINAMATH_GPT_tetrahedron_side_length_l1193_119385


namespace NUMINAMATH_GPT_problem_statement_l1193_119380

theorem problem_statement : 15 * 30 + 45 * 15 + 15 * 15 = 1350 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1193_119380


namespace NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1193_119351

theorem ratio_of_ages_in_two_years (S M : ℕ) (h1: M = S + 28) (h2: M + 2 = (S + 2) * 2) (h3: S = 26) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1193_119351


namespace NUMINAMATH_GPT_find_extrema_A_l1193_119313

def eight_digit_number(n : ℕ) : Prop := n ≥ 10^7 ∧ n < 10^8

def coprime_with_thirtysix(n : ℕ) : Prop := Nat.gcd n 36 = 1

def transform_last_to_first(n : ℕ) : ℕ := 
  let last := n % 10
  let rest := n / 10
  last * 10^7 + rest

theorem find_extrema_A :
  ∃ (A_max A_min : ℕ), 
    (∃ B_max B_min : ℕ, 
      eight_digit_number B_max ∧ 
      eight_digit_number B_min ∧ 
      coprime_with_thirtysix B_max ∧ 
      coprime_with_thirtysix B_min ∧ 
      B_max > 77777777 ∧ 
      B_min > 77777777 ∧ 
      transform_last_to_first B_max = A_max ∧ 
      transform_last_to_first B_min = A_min) ∧ 
    A_max = 99999998 ∧ 
    A_min = 17777779 := 
  sorry

end NUMINAMATH_GPT_find_extrema_A_l1193_119313


namespace NUMINAMATH_GPT_total_amount_invested_l1193_119303

def annualIncome (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

def totalInvestment (T x y : ℝ) : Prop :=
  T - x = y

def condition (T : ℝ) : Prop :=
  let income_10_percent := annualIncome (T - 800) 0.10
  let income_8_percent := annualIncome 800 0.08
  income_10_percent - income_8_percent = 56

theorem total_amount_invested :
  ∃ (T : ℝ), condition T ∧ totalInvestment T 800 800 ∧ T = 2000 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_invested_l1193_119303


namespace NUMINAMATH_GPT_polynomial_multiplication_l1193_119307

noncomputable def multiply_polynomials (a b : ℤ) :=
  let p1 := 3 * a ^ 4 - 7 * b ^ 3
  let p2 := 9 * a ^ 8 + 21 * a ^ 4 * b ^ 3 + 49 * b ^ 6 + 6 * a ^ 2 * b ^ 2
  let result := 27 * a ^ 12 + 18 * a ^ 6 * b ^ 2 - 42 * a ^ 2 * b ^ 5 - 343 * b ^ 9
  p1 * p2 = result

-- The main statement to prove
theorem polynomial_multiplication (a b : ℤ) : multiply_polynomials a b :=
by
  sorry

end NUMINAMATH_GPT_polynomial_multiplication_l1193_119307


namespace NUMINAMATH_GPT_value_of_a_l1193_119339

theorem value_of_a (P Q : Set ℝ) (a : ℝ) :
  (P = {x | x^2 = 1}) →
  (Q = {x | ax = 1}) →
  (Q ⊆ P) →
  (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1193_119339


namespace NUMINAMATH_GPT_expansion_dissimilar_terms_count_l1193_119377

def number_of_dissimilar_terms (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem expansion_dissimilar_terms_count :
  number_of_dissimilar_terms 7 4 = 120 := by
  sorry

end NUMINAMATH_GPT_expansion_dissimilar_terms_count_l1193_119377


namespace NUMINAMATH_GPT_problem_inequality_solution_problem_prove_inequality_l1193_119381

-- Function definition for f(x)
def f (x : ℝ) := |2 * x - 3| + |2 * x + 3|

-- Problem 1: Prove the solution set for the inequality f(x) ≤ 8
theorem problem_inequality_solution (x : ℝ) : f x ≤ 8 ↔ -2 ≤ x ∧ x ≤ 2 :=
sorry

-- Problem 2: Prove a + 2b + 3c ≥ 9 given conditions
theorem problem_prove_inequality (a b c : ℝ) (M : ℝ) (h1 : M = 6)
  (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 1 / a + 1 / (2 * b) + 1 / (3 * c) = M / 6) :
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end NUMINAMATH_GPT_problem_inequality_solution_problem_prove_inequality_l1193_119381


namespace NUMINAMATH_GPT_cost_per_bag_l1193_119300

-- Definitions and variables based on the conditions
def sandbox_length : ℝ := 3  -- Sandbox length in feet
def sandbox_width : ℝ := 3   -- Sandbox width in feet
def bag_area : ℝ := 3        -- Area of one bag of sand in square feet
def total_cost : ℝ := 12     -- Total cost to fill up the sandbox in dollars

-- Statement to prove
theorem cost_per_bag : (total_cost / (sandbox_length * sandbox_width / bag_area)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_bag_l1193_119300


namespace NUMINAMATH_GPT_dutch_exam_problem_l1193_119342

theorem dutch_exam_problem (a b c d : ℝ) : 
  (a * b + c + d = 3) ∧ 
  (b * c + d + a = 5) ∧ 
  (c * d + a + b = 2) ∧ 
  (d * a + b + c = 6) → 
  (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) := 
by
  sorry

end NUMINAMATH_GPT_dutch_exam_problem_l1193_119342


namespace NUMINAMATH_GPT_gold_coins_equality_l1193_119384

theorem gold_coins_equality (pouches : List ℕ) 
  (h_pouches_length : pouches.length = 9)
  (h_pouches_sum : pouches.sum = 60)
  : (∃ s_2 : List (List ℕ), s_2.length = 2 ∧ ∀ l ∈ s_2, l.sum = 30) ∧
    (∃ s_3 : List (List ℕ), s_3.length = 3 ∧ ∀ l ∈ s_3, l.sum = 20) ∧
    (∃ s_4 : List (List ℕ), s_4.length = 4 ∧ ∀ l ∈ s_4, l.sum = 15) ∧
    (∃ s_5 : List (List ℕ), s_5.length = 5 ∧ ∀ l ∈ s_5, l.sum = 12) :=
sorry

end NUMINAMATH_GPT_gold_coins_equality_l1193_119384


namespace NUMINAMATH_GPT_base_7_units_digit_l1193_119344

theorem base_7_units_digit (a : ℕ) (b : ℕ) (h₁ : a = 326) (h₂ : b = 57) : ((a * b) % 7) = 4 := by
  sorry

end NUMINAMATH_GPT_base_7_units_digit_l1193_119344


namespace NUMINAMATH_GPT_roots_of_cubic_eq_l1193_119399

theorem roots_of_cubic_eq (r s t a b c d : ℂ) (h1 : a ≠ 0) (h2 : r ≠ 0) (h3 : s ≠ 0) 
  (h4 : t ≠ 0) (hrst : ∀ x : ℂ, a * x ^ 3 + b * x ^ 2 + c * x + d = 0 → (x = r ∨ x = s ∨ x = t) ∧ (x = r <-> r + s + t - x = -b / a)) 
  (Vieta1 : r + s + t = -b / a) (Vieta2 : r * s + r * t + s * t = c / a) (Vieta3 : r * s * t = -d / a) :
  (1 / r ^ 3 + 1 / s ^ 3 + 1 / t ^ 3 = c ^ 3 / d ^ 3) := 
by sorry

end NUMINAMATH_GPT_roots_of_cubic_eq_l1193_119399


namespace NUMINAMATH_GPT_math_problem_l1193_119328

open Real

theorem math_problem
  (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h : x^2 + y^2 + z^2 = 3) :
  sqrt (3 - ( (x + y) / 2) ^ 2) + sqrt (3 - ( (y + z) / 2) ^ 2) + sqrt (3 - ( (z + x) / 2) ^ 2) ≥ 3 * sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_math_problem_l1193_119328


namespace NUMINAMATH_GPT_single_reduction_equivalent_l1193_119371

/-- If a price is first reduced by 25%, and the new price is further reduced by 30%, 
the single percentage reduction equivalent to these two reductions together is 47.5%. -/
theorem single_reduction_equivalent :
  ∀ P : ℝ, (1 - 0.25) * (1 - 0.30) * P = P * (1 - 0.475) :=
by
  intros
  sorry

end NUMINAMATH_GPT_single_reduction_equivalent_l1193_119371


namespace NUMINAMATH_GPT_number_of_chlorine_atoms_l1193_119360

def molecular_weight_of_aluminum : ℝ := 26.98
def molecular_weight_of_chlorine : ℝ := 35.45
def molecular_weight_of_compound : ℝ := 132.0

theorem number_of_chlorine_atoms :
  ∃ n : ℕ, molecular_weight_of_compound = molecular_weight_of_aluminum + n * molecular_weight_of_chlorine ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_chlorine_atoms_l1193_119360


namespace NUMINAMATH_GPT_max_area_angle_A_l1193_119326

open Real

theorem max_area_angle_A (A B C : ℝ) (tan_A tan_B : ℝ) :
  tan A * tan B = 1 ∧ AB = sqrt 3 → 
  (∃ A, A = π / 4 ∧ area_maximized)
  :=
by sorry

end NUMINAMATH_GPT_max_area_angle_A_l1193_119326


namespace NUMINAMATH_GPT_xyz_cubed_over_xyz_eq_21_l1193_119369

open Complex

theorem xyz_cubed_over_xyz_eq_21 {x y z : ℂ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 18)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 21 :=
sorry

end NUMINAMATH_GPT_xyz_cubed_over_xyz_eq_21_l1193_119369


namespace NUMINAMATH_GPT_percentage_increase_of_cars_l1193_119387

theorem percentage_increase_of_cars :
  ∀ (initial final : ℕ), initial = 24 → final = 48 → ((final - initial) * 100 / initial) = 100 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_increase_of_cars_l1193_119387


namespace NUMINAMATH_GPT_find_b_l1193_119374

theorem find_b (x y z a b : ℝ) (h1 : x + y = 2) (h2 : xy - z^2 = a) (h3 : b = x + y + z) : b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1193_119374


namespace NUMINAMATH_GPT_lions_after_one_year_l1193_119362

def initial_lions : ℕ := 100
def birth_rate : ℕ := 5
def death_rate : ℕ := 1
def months_in_year : ℕ := 12

theorem lions_after_one_year : 
  initial_lions + (birth_rate * months_in_year) - (death_rate * months_in_year) = 148 :=
by
  sorry

end NUMINAMATH_GPT_lions_after_one_year_l1193_119362


namespace NUMINAMATH_GPT_intercepts_equal_l1193_119330

theorem intercepts_equal (m : ℝ) :
  (∃ x y: ℝ, mx - y - 3 - m = 0 ∧ y ≠ 0 ∧ (x = 3 + m ∧ y = -(3 + m))) ↔ (m = -3 ∨ m = -1) :=
by 
  sorry

end NUMINAMATH_GPT_intercepts_equal_l1193_119330


namespace NUMINAMATH_GPT_total_time_in_cocoons_l1193_119309

theorem total_time_in_cocoons (CA CB CC: ℝ) 
    (h1: 4 * CA = 90)
    (h2: 4 * CB = 120)
    (h3: 4 * CC = 150) 
    : CA + CB + CC = 90 := 
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_total_time_in_cocoons_l1193_119309


namespace NUMINAMATH_GPT_rational_root_contradiction_l1193_119347

theorem rational_root_contradiction 
(a b c : ℤ) 
(h_odd_a : a % 2 ≠ 0) 
(h_odd_b : b % 2 ≠ 0)
(h_odd_c : c % 2 ≠ 0)
(rational_root_exists : ∃ (r : ℚ), a * r^2 + b * r + c = 0) :
false :=
sorry

end NUMINAMATH_GPT_rational_root_contradiction_l1193_119347


namespace NUMINAMATH_GPT_grocery_store_distance_l1193_119322

theorem grocery_store_distance 
    (park_house : ℕ) (park_store : ℕ) (total_distance : ℕ) (grocery_store_house: ℕ) :
    park_house = 5 ∧ park_store = 3 ∧ total_distance = 16 → grocery_store_house = 8 :=
by 
    sorry

end NUMINAMATH_GPT_grocery_store_distance_l1193_119322


namespace NUMINAMATH_GPT_count_divisors_of_54_greater_than_7_l1193_119337

theorem count_divisors_of_54_greater_than_7 : ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ n ∈ S, n ∣ 54 ∧ n > 7 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_count_divisors_of_54_greater_than_7_l1193_119337


namespace NUMINAMATH_GPT_find_k_l1193_119396

theorem find_k (k : ℕ) (hk : 0 < k) (h : (k + 4) / (k^2 - 1) = 9 / 35) : k = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1193_119396


namespace NUMINAMATH_GPT_number_division_reduction_l1193_119364

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 48) : x = 72 := 
sorry

end NUMINAMATH_GPT_number_division_reduction_l1193_119364


namespace NUMINAMATH_GPT_odd_function_property_l1193_119332

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Lean 4 statement of the problem
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd f) : ∀ x : ℝ, f x + f (-x) = 0 := 
  by sorry

end NUMINAMATH_GPT_odd_function_property_l1193_119332


namespace NUMINAMATH_GPT_find_reflection_line_l1193_119394

/-*
Triangle ABC has vertices with coordinates A(2,3), B(7,8), and C(-4,6).
The triangle is reflected about line L.
The image points are A'(2,-5), B'(7,-10), and C'(-4,-8).
Prove that the equation of line L is y = -1.
*-/
theorem find_reflection_line :
  ∃ (L : ℝ), (∀ (x : ℝ), (∃ (k : ℝ), L = k) ∧ (L = -1)) :=
by sorry

end NUMINAMATH_GPT_find_reflection_line_l1193_119394


namespace NUMINAMATH_GPT_mul_mixed_number_eq_l1193_119367

theorem mul_mixed_number_eq :
  99 + 24 / 25 * -5 = -499 - 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_mul_mixed_number_eq_l1193_119367


namespace NUMINAMATH_GPT_teachers_can_sit_in_middle_l1193_119302

-- Definitions for the conditions
def num_students : ℕ := 4
def num_teachers : ℕ := 3
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def permutations (n r : ℕ) : ℕ := factorial n / factorial (n - r)

-- Definition statements
def num_ways_teachers : ℕ := permutations num_teachers num_teachers
def num_ways_students : ℕ := permutations num_students num_students

-- Main theorem statement
theorem teachers_can_sit_in_middle : num_ways_teachers * num_ways_students = 144 := by
  -- Calculation goes here but is omitted for brevity
  sorry

end NUMINAMATH_GPT_teachers_can_sit_in_middle_l1193_119302


namespace NUMINAMATH_GPT_area_of_rectangle_l1193_119320

-- Define the conditions
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def area (length width : ℕ) : ℕ := length * width

-- Assumptions based on the problem conditions
variable (length : ℕ) (width : ℕ) (P : ℕ) (A : ℕ)
variable (h1 : width = 25)
variable (h2 : P = 110)

-- Goal: Prove the area is 750 square meters
theorem area_of_rectangle : 
  ∃ l : ℕ, perimeter l 25 = 110 → area l 25 = 750 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1193_119320


namespace NUMINAMATH_GPT_exists_not_holds_l1193_119355

variable (S : Type) [Nonempty S] [Inhabited S]
variable (op : S → S → S)
variable (h : ∀ a b : S, op a (op b a) = b)

theorem exists_not_holds : ∃ a b : S, (op (op a b) a) ≠ a := sorry

end NUMINAMATH_GPT_exists_not_holds_l1193_119355


namespace NUMINAMATH_GPT_xy_leq_half_x_squared_plus_y_squared_l1193_119375

theorem xy_leq_half_x_squared_plus_y_squared (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_xy_leq_half_x_squared_plus_y_squared_l1193_119375
