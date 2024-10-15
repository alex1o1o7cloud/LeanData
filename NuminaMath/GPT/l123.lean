import Mathlib

namespace NUMINAMATH_GPT_find_center_of_ellipse_l123_12359

-- Defining the equation of the ellipse
def ellipse (x y : ℝ) : Prop := 2*x^2 + 2*x*y + y^2 + 2*x + 2*y - 4 = 0

-- The coordinates of the center
def center_of_ellipse : ℝ × ℝ := (0, -1)

-- The theorem asserting the center of the ellipse
theorem find_center_of_ellipse (x y : ℝ) (h : ellipse x y) : (x, y) = center_of_ellipse :=
sorry

end NUMINAMATH_GPT_find_center_of_ellipse_l123_12359


namespace NUMINAMATH_GPT_simplify_expression_l123_12351

theorem simplify_expression (x y : ℝ) : 7 * x + 8 * y - 3 * x + 4 * y + 10 = 4 * x + 12 * y + 10 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l123_12351


namespace NUMINAMATH_GPT_find_integer_l123_12333

theorem find_integer (N : ℤ) (hN : N^2 + N = 12) (h_pos : 0 < N) : N = 3 :=
sorry

end NUMINAMATH_GPT_find_integer_l123_12333


namespace NUMINAMATH_GPT_triangle_side_lengths_m_range_l123_12394

theorem triangle_side_lengths_m_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (m : ℝ) :
  (2 - Real.sqrt 3) < m ∧ m < (2 + Real.sqrt 3) ↔
  (x + y) + Real.sqrt (x^2 + x * y + y^2) > m * Real.sqrt (x * y) ∧
  (x + y) + m * Real.sqrt (x * y) > Real.sqrt (x^2 + x * y + y^2) ∧
  Real.sqrt (x^2 + x * y + y^2) + m * Real.sqrt (x * y) > (x + y) :=
by sorry

end NUMINAMATH_GPT_triangle_side_lengths_m_range_l123_12394


namespace NUMINAMATH_GPT_bicycles_purchased_on_Friday_l123_12363

theorem bicycles_purchased_on_Friday (F : ℕ) : (F - 10) - 4 + 2 = 3 → F = 15 := by
  intro h
  sorry

end NUMINAMATH_GPT_bicycles_purchased_on_Friday_l123_12363


namespace NUMINAMATH_GPT_solve_for_x_l123_12323

theorem solve_for_x 
  (a b c d x y z w : ℝ) 
  (H1 : x + y + z + w = 360)
  (H2 : a = x + y / 2) 
  (H3 : b = y + z / 2) 
  (H4 : c = z + w / 2) 
  (H5 : d = w + x / 2) : 
  x = (16 / 15) * (a - b / 2 + c / 4 - d / 8) :=
sorry


end NUMINAMATH_GPT_solve_for_x_l123_12323


namespace NUMINAMATH_GPT_empty_vessel_percentage_l123_12332

theorem empty_vessel_percentage
  (P : ℝ) -- weight of the paint that completely fills the vessel
  (E : ℝ) -- weight of the empty vessel
  (h1 : 0.5 * (E + P) = E + 0.42857142857142855 * P)
  (h2 : 0.07142857142857145 * P = 0.5 * E):
  (E / (E + P) * 100) = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_empty_vessel_percentage_l123_12332


namespace NUMINAMATH_GPT_Tony_science_degree_years_l123_12358

theorem Tony_science_degree_years (X : ℕ) (Total : ℕ)
  (h1 : Total = 14)
  (h2 : Total = X + 2 * X + 2) :
  X = 4 :=
by
  sorry

end NUMINAMATH_GPT_Tony_science_degree_years_l123_12358


namespace NUMINAMATH_GPT_k_polygonal_intersects_fermat_l123_12361

theorem k_polygonal_intersects_fermat (k : ℕ) (n m : ℕ) (h1: k > 2) 
  (h2 : ∃ n m, (k - 2) * n * (n - 1) / 2 + n = 2 ^ (2 ^ m) + 1) : 
  k = 3 ∨ k = 5 :=
  sorry

end NUMINAMATH_GPT_k_polygonal_intersects_fermat_l123_12361


namespace NUMINAMATH_GPT_opposite_of_reciprocal_negative_one_third_l123_12314

theorem opposite_of_reciprocal_negative_one_third : -(1 / (-1 / 3)) = 3 := by
  sorry

end NUMINAMATH_GPT_opposite_of_reciprocal_negative_one_third_l123_12314


namespace NUMINAMATH_GPT_largest_possible_n_l123_12318

theorem largest_possible_n :
  ∃ (m n : ℕ), (0 < m) ∧ (0 < n) ∧ (m + n = 10) ∧ (n = 9) :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_n_l123_12318


namespace NUMINAMATH_GPT_mass_percentage_O_in_N2O_is_approximately_36_35_l123_12369

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def number_of_N : ℕ := 2
noncomputable def number_of_O : ℕ := 1

noncomputable def molar_mass_N2O : ℝ := (number_of_N * atomic_mass_N) + (number_of_O * atomic_mass_O)

noncomputable def mass_percentage_O : ℝ := (atomic_mass_O / molar_mass_N2O) * 100

theorem mass_percentage_O_in_N2O_is_approximately_36_35 :
  abs (mass_percentage_O - 36.35) < 0.01 := sorry

end NUMINAMATH_GPT_mass_percentage_O_in_N2O_is_approximately_36_35_l123_12369


namespace NUMINAMATH_GPT_chord_length_l123_12370

theorem chord_length (ρ θ : ℝ) (p : ℝ) : 
  (∀ θ, ρ = 6 * Real.cos θ) ∧ (θ = Real.pi / 4) → 
  ∃ l : ℝ, l = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l123_12370


namespace NUMINAMATH_GPT_gumballs_per_package_l123_12389

theorem gumballs_per_package (total_gumballs : ℕ) (packages : ℝ) (h1 : total_gumballs = 100) (h2 : packages = 20.0) :
  total_gumballs / packages = 5 :=
by sorry

end NUMINAMATH_GPT_gumballs_per_package_l123_12389


namespace NUMINAMATH_GPT_sin_minus_pi_over_3_eq_neg_four_fifths_l123_12347

theorem sin_minus_pi_over_3_eq_neg_four_fifths
  (α : ℝ)
  (h : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (α - π / 3) = - (4 / 5) :=
by
  sorry

end NUMINAMATH_GPT_sin_minus_pi_over_3_eq_neg_four_fifths_l123_12347


namespace NUMINAMATH_GPT_smallest_q_p_difference_l123_12317

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_smallest_q_p_difference_l123_12317


namespace NUMINAMATH_GPT_values_of_x_plus_y_l123_12376

theorem values_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 :=
sorry

end NUMINAMATH_GPT_values_of_x_plus_y_l123_12376


namespace NUMINAMATH_GPT_ratio_of_perimeters_l123_12303

-- Define lengths of the rectangular patch
def length_rect : ℝ := 400
def width_rect : ℝ := 300

-- Define the length of the side of the square patch
def side_square : ℝ := 700

-- Define the perimeters of both patches
def P_square : ℝ := 4 * side_square
def P_rectangle : ℝ := 2 * (length_rect + width_rect)

-- Theorem stating the ratio of the perimeters
theorem ratio_of_perimeters : P_square / P_rectangle = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l123_12303


namespace NUMINAMATH_GPT_minimal_erasure_l123_12306

noncomputable def min_factors_to_erase : ℕ :=
  2016

theorem minimal_erasure:
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = g x) → 
    (∃ f' g' : ℝ → ℝ, (∀ x, f x ≠ g x) ∧ 
      ((∃ s : Finset ℕ, s.card = min_factors_to_erase ∧ (∀ i ∈ s, f' x = (x - i) * f x)) ∧ 
      (∃ t : Finset ℕ, t.card = min_factors_to_erase ∧ (∀ i ∈ t, g' x = (x - i) * g x)))) :=
by
  sorry

end NUMINAMATH_GPT_minimal_erasure_l123_12306


namespace NUMINAMATH_GPT_set_equivalence_l123_12378

theorem set_equivalence :
  {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ 2 * p.1 - p.2 = 2} = {(1, 0)} :=
by
  sorry

end NUMINAMATH_GPT_set_equivalence_l123_12378


namespace NUMINAMATH_GPT_solve_equation_l123_12301

theorem solve_equation (x : ℝ) : x^2 = 5 * x → x = 0 ∨ x = 5 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l123_12301


namespace NUMINAMATH_GPT_total_amount_shared_l123_12325

theorem total_amount_shared (X_share Y_share Z_share total_amount : ℝ) 
                            (h1 : Y_share = 0.45 * X_share) 
                            (h2 : Z_share = 0.50 * X_share) 
                            (h3 : Y_share = 45) : 
                            total_amount = X_share + Y_share + Z_share := 
by 
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_amount_shared_l123_12325


namespace NUMINAMATH_GPT_exists_x_gt_zero_negation_l123_12302

theorem exists_x_gt_zero_negation :
  (∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_exists_x_gt_zero_negation_l123_12302


namespace NUMINAMATH_GPT_tom_purchases_l123_12357

def total_cost_before_discount (price_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  price_per_box * num_boxes

def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

def total_cost_after_discount (total_cost : ℝ) (discount_amount : ℝ) : ℝ :=
  total_cost - discount_amount

def remaining_boxes (total_boxes : ℕ) (given_boxes : ℕ) : ℕ :=
  total_boxes - given_boxes

def total_pieces (num_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  num_boxes * pieces_per_box

theorem tom_purchases
  (price_per_box : ℝ) (num_boxes : ℕ) (discount_rate : ℝ) (given_boxes : ℕ) (pieces_per_box : ℕ) :
  (price_per_box = 4) →
  (num_boxes = 12) →
  (discount_rate = 0.15) →
  (given_boxes = 7) →
  (pieces_per_box = 6) →
  total_cost_after_discount (total_cost_before_discount price_per_box num_boxes) 
                             (discount (total_cost_before_discount price_per_box num_boxes) discount_rate)
  = 40.80 ∧
  total_pieces (remaining_boxes num_boxes given_boxes) pieces_per_box
  = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tom_purchases_l123_12357


namespace NUMINAMATH_GPT_train_half_speed_time_l123_12395

-- Definitions for Lean
variables (S T D : ℝ)

-- Conditions
axiom cond1 : D = S * T
axiom cond2 : D = (1 / 2) * S * (T + 4)

-- Theorem Statement
theorem train_half_speed_time : 
  (T = 4) → (4 + 4 = 8) := 
by 
  intros hT
  simp [hT]

end NUMINAMATH_GPT_train_half_speed_time_l123_12395


namespace NUMINAMATH_GPT_intersection_M_N_l123_12354

def M : Set ℝ := { x : ℝ | x + 1 ≥ 0 }
def N : Set ℝ := { x : ℝ | x^2 < 4 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -1 ≤ x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l123_12354


namespace NUMINAMATH_GPT_complementary_angles_not_obtuse_l123_12300

-- Define the concept of complementary angles.
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

-- Define that neither angle should be obtuse.
def not_obtuse (a b : ℝ) : Prop :=
  a < 90 ∧ b < 90

-- Proof problem statement
theorem complementary_angles_not_obtuse (a b : ℝ) (ha : a < 90) (hb : b < 90) (h_comp : is_complementary a b) : 
  not_obtuse a b :=
by
  sorry

end NUMINAMATH_GPT_complementary_angles_not_obtuse_l123_12300


namespace NUMINAMATH_GPT_find_integer_n_l123_12349

theorem find_integer_n :
  ∃ n : ℤ, 
    50 ≤ n ∧ n ≤ 120 ∧ (n % 5 = 0) ∧ (n % 6 = 3) ∧ (n % 7 = 4) ∧ n = 165 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_n_l123_12349


namespace NUMINAMATH_GPT_negation_proposition_l123_12346

theorem negation_proposition :
  ¬(∀ x : ℝ, x^2 > x) ↔ ∃ x : ℝ, x^2 ≤ x :=
sorry

end NUMINAMATH_GPT_negation_proposition_l123_12346


namespace NUMINAMATH_GPT_trig_identity_cos2theta_tan_minus_pi_over_4_l123_12335

variable (θ : ℝ)

-- Given condition
def tan_theta_is_2 : Prop := Real.tan θ = 2

-- Proof problem 1: Prove that cos(2θ) = -3/5
def cos2theta (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.cos (2 * θ) = -3 / 5

-- Proof problem 2: Prove that tan(θ - π/4) = 1/3
def tan_theta_minus_pi_over_4 (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.tan (θ - Real.pi / 4) = 1 / 3

-- Main theorem statement
theorem trig_identity_cos2theta_tan_minus_pi_over_4 
  (θ : ℝ) (h : tan_theta_is_2 θ) :
  cos2theta θ h ∧ tan_theta_minus_pi_over_4 θ h :=
sorry

end NUMINAMATH_GPT_trig_identity_cos2theta_tan_minus_pi_over_4_l123_12335


namespace NUMINAMATH_GPT_total_weight_of_shells_l123_12364

noncomputable def initial_weight : ℝ := 5.25
noncomputable def weight_large_shell_g : ℝ := 700
noncomputable def grams_per_pound : ℝ := 453.592
noncomputable def additional_weight : ℝ := 4.5

/-
We need to prove:
5.25 pounds (initial weight) + (700 grams * (1 pound / 453.592 grams)) (weight of large shell in pounds) + 4.5 pounds (additional weight) = 11.293235835 pounds
-/
theorem total_weight_of_shells :
  initial_weight + (weight_large_shell_g / grams_per_pound) + additional_weight = 11.293235835 := by
    -- Proof will be inserted here
    sorry

end NUMINAMATH_GPT_total_weight_of_shells_l123_12364


namespace NUMINAMATH_GPT_solution_set_of_inequality_l123_12386

theorem solution_set_of_inequality :
  {x : ℝ | abs (x^2 - 5 * x + 6) < x^2 - 4} = { x : ℝ | x > 2 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l123_12386


namespace NUMINAMATH_GPT_max_value_sqrt_add_l123_12355

noncomputable def sqrt_add (a b : ℝ) : ℝ := Real.sqrt (a + 1) + Real.sqrt (b + 3)

theorem max_value_sqrt_add (a b : ℝ) (h : 0 < a) (h' : 0 < b) (hab : a + b = 5) :
  sqrt_add a b ≤ 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_sqrt_add_l123_12355


namespace NUMINAMATH_GPT_y_intercept_tangent_line_l123_12311

/-- Three circles have radii 3, 2, and 1 respectively. The first circle has center at (3,0), 
the second at (7,0), and the third at (11,0). A line is tangent to all three circles 
at points in the first quadrant. Prove the y-intercept of this line is 36.
-/
theorem y_intercept_tangent_line
  (r1 r2 r3 : ℝ) (h1 : r1 = 3) (h2 : r2 = 2) (h3 : r3 = 1)
  (c1 c2 c3 : ℝ × ℝ) (hc1 : c1 = (3, 0)) (hc2 : c2 = (7, 0)) (hc3 : c3 = (11, 0)) :
  ∃ y_intercept : ℝ, y_intercept = 36 :=
sorry

end NUMINAMATH_GPT_y_intercept_tangent_line_l123_12311


namespace NUMINAMATH_GPT_boxes_containing_pans_l123_12312

def num_boxes : Nat := 26
def num_teacups_per_box : Nat := 20
def num_cups_broken_per_box : Nat := 2
def teacups_left : Nat := 180

def num_teacup_boxes (num_boxes : Nat) (num_teacups_per_box : Nat) (num_cups_broken_per_box : Nat) (teacups_left : Nat) : Nat :=
  teacups_left / (num_teacups_per_box - num_cups_broken_per_box)

def num_remaining_boxes (num_boxes : Nat) (num_teacup_boxes : Nat) : Nat :=
  num_boxes - num_teacup_boxes

def num_pans_boxes (num_remaining_boxes : Nat) : Nat :=
  num_remaining_boxes / 2

theorem boxes_containing_pans : ∀ (num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left : Nat),
  num_boxes = 26 →
  num_teacups_per_box = 20 →
  num_cups_broken_per_box = 2 →
  teacups_left = 180 →
  num_pans_boxes (num_remaining_boxes num_boxes (num_teacup_boxes num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left)) = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boxes_containing_pans_l123_12312


namespace NUMINAMATH_GPT_smallest_t_for_circle_l123_12345

theorem smallest_t_for_circle (t : ℝ) :
  (∀ r θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) → t ≥ π :=
by sorry

end NUMINAMATH_GPT_smallest_t_for_circle_l123_12345


namespace NUMINAMATH_GPT_find_length_QS_l123_12371

theorem find_length_QS 
  (cosR : ℝ) (RS : ℝ) (QR : ℝ) (QS : ℝ)
  (h1 : cosR = 3 / 5)
  (h2 : RS = 10)
  (h3 : cosR = QR / RS) :
  QS = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_length_QS_l123_12371


namespace NUMINAMATH_GPT_sum_of_powers_of_two_l123_12381

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_two_l123_12381


namespace NUMINAMATH_GPT_ratio_of_segments_l123_12350

variable (F S T : ℕ)

theorem ratio_of_segments : T = 10 → F = 2 * (S + T) → F + S + T = 90 → (T / S = 1 / 2) :=
by
  intros hT hF hSum
  sorry

end NUMINAMATH_GPT_ratio_of_segments_l123_12350


namespace NUMINAMATH_GPT_quoted_value_of_stock_l123_12341

theorem quoted_value_of_stock (D Y Q : ℝ) (h1 : D = 8) (h2 : Y = 10) (h3 : Y = (D / Q) * 100) : Q = 80 :=
by 
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_quoted_value_of_stock_l123_12341


namespace NUMINAMATH_GPT_maximize_wz_xy_zx_l123_12352

-- Variables definition
variables {w x y z : ℝ}

-- Main statement
theorem maximize_wz_xy_zx (h_sum : w + x + y + z = 200) (h_nonneg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  (w * z + x * y + z * x) ≤ 7500 :=
sorry

end NUMINAMATH_GPT_maximize_wz_xy_zx_l123_12352


namespace NUMINAMATH_GPT_no_equalities_l123_12305

def f1 (x : ℤ) : ℤ := x * (x - 2007)
def f2 (x : ℤ) : ℤ := (x - 1) * (x - 2006)
def f1004 (x : ℤ) : ℤ := (x - 1003) * (x - 1004)

theorem no_equalities (x : ℤ) (h : 0 ≤ x ∧ x ≤ 2007) :
  ¬(f1 x = f2 x ∨ f1 x = f1004 x ∨ f2 x = f1004 x) :=
by
  sorry

end NUMINAMATH_GPT_no_equalities_l123_12305


namespace NUMINAMATH_GPT_non_zero_number_is_nine_l123_12326

theorem non_zero_number_is_nine {x : ℝ} (h1 : (x + x^2) / 2 = 5 * x) (h2 : x ≠ 0) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_non_zero_number_is_nine_l123_12326


namespace NUMINAMATH_GPT_chord_length_y_eq_x_plus_one_meets_circle_l123_12304

noncomputable def chord_length (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem chord_length_y_eq_x_plus_one_meets_circle 
  (A B : ℝ × ℝ) 
  (hA : A.2 = A.1 + 1) 
  (hB : B.2 = B.1 + 1) 
  (hA_on_circle : A.1^2 + A.2^2 + 2 * A.2 - 3 = 0)
  (hB_on_circle : B.1^2 + B.2^2 + 2 * B.2 - 3 = 0) :
  chord_length A B = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_chord_length_y_eq_x_plus_one_meets_circle_l123_12304


namespace NUMINAMATH_GPT_minimize_product_l123_12342

theorem minimize_product
    (a b c : ℕ) 
    (h_positive: a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq: 10 * a^2 - 3 * a * b + 7 * c^2 = 0) : 
    (gcd a b) * (gcd b c) * (gcd c a) = 3 :=
sorry

end NUMINAMATH_GPT_minimize_product_l123_12342


namespace NUMINAMATH_GPT_parcel_total_weight_l123_12393

theorem parcel_total_weight (x y z : ℝ) 
  (h1 : x + y = 132) 
  (h2 : y + z = 146) 
  (h3 : z + x = 140) : 
  x + y + z = 209 :=
by
  sorry

end NUMINAMATH_GPT_parcel_total_weight_l123_12393


namespace NUMINAMATH_GPT_unique_solution_l123_12338

theorem unique_solution (p : ℕ) (a b n : ℕ) : 
  p.Prime → 2^a + p^b = n^(p-1) → (p, a, b, n) = (3, 0, 1, 2) ∨ (p = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_solution_l123_12338


namespace NUMINAMATH_GPT_total_amount_is_4000_l123_12320

-- Define the amount put at a 3% interest rate
def amount_at_3_percent : ℝ := 2800

-- Define the total annual interest from both investments
def total_annual_interest : ℝ := 144

-- Define the interest rate for the amount put at 3% and 5%
def interest_rate_3_percent : ℝ := 0.03
def interest_rate_5_percent : ℝ := 0.05

-- Define the total amount to be proved
def total_amount_divided (T : ℝ) : Prop :=
  interest_rate_3_percent * amount_at_3_percent + 
  interest_rate_5_percent * (T - amount_at_3_percent) = total_annual_interest

-- The theorem that states the total amount divided is Rs. 4000
theorem total_amount_is_4000 : ∃ T : ℝ, total_amount_divided T ∧ T = 4000 :=
by
  use 4000
  unfold total_amount_divided
  simp
  sorry

end NUMINAMATH_GPT_total_amount_is_4000_l123_12320


namespace NUMINAMATH_GPT_K_time_9_hours_l123_12309

theorem K_time_9_hours
  (x : ℝ) -- x is the speed of K
  (hx : 45 / x = 9) -- K's time for 45 miles is 9 hours
  (y : ℝ) -- y is the speed of M
  (h₁ : x = y + 0.5) -- K travels 0.5 mph faster than M
  (h₂ : 45 / y - 45 / x = 3 / 4) -- K takes 3/4 hour less than M
  : 45 / x = 9 :=
by
  sorry

end NUMINAMATH_GPT_K_time_9_hours_l123_12309


namespace NUMINAMATH_GPT_find_other_endpoint_l123_12383

theorem find_other_endpoint :
  ∀ (A B M : ℝ × ℝ),
  M = (2, 3) →
  A = (7, -4) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B = (-3, 10) :=
by
  intros A B M hM1 hA hM2
  sorry

end NUMINAMATH_GPT_find_other_endpoint_l123_12383


namespace NUMINAMATH_GPT_find_triplets_l123_12322

theorem find_triplets (m n k : ℕ) (pos_m : 0 < m) (pos_n : 0 < n) (pos_k : 0 < k) : 
  (k^m ∣ m^n - 1) ∧ (k^n ∣ n^m - 1) ↔ (k = 1) ∨ (m = 1 ∧ n = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_triplets_l123_12322


namespace NUMINAMATH_GPT_average_of_all_digits_l123_12331

theorem average_of_all_digits (d : List ℕ) (h_len : d.length = 9)
  (h1 : (d.take 4).sum = 32)
  (h2 : (d.drop 4).sum = 130) : 
  (d.sum / d.length : ℚ) = 18 := 
by
  sorry

end NUMINAMATH_GPT_average_of_all_digits_l123_12331


namespace NUMINAMATH_GPT_square_of_1027_l123_12344

theorem square_of_1027 :
  1027 * 1027 = 1054729 :=
by
  sorry

end NUMINAMATH_GPT_square_of_1027_l123_12344


namespace NUMINAMATH_GPT_min_value_ineq_inequality_proof_l123_12377

variable (a b x1 x2 : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx1_pos : 0 < x1) (hx2_pos : 0 < x2) (hab_sum : a + b = 1)

-- First problem: Prove that the minimum value of the given expression is 6.
theorem min_value_ineq : (x1 / a) + (x2 / b) + (2 / (x1 * x2)) ≥ 6 := by
  sorry

-- Second problem: Prove the given inequality.
theorem inequality_proof : (a * x1 + b * x2) * (a * x2 + b * x1) ≥ x1 * x2 := by
  sorry

end NUMINAMATH_GPT_min_value_ineq_inequality_proof_l123_12377


namespace NUMINAMATH_GPT_remainder_div_1234_567_89_1011_mod_12_l123_12329

theorem remainder_div_1234_567_89_1011_mod_12 :
  (1234^567 + 89^1011) % 12 = 9 := 
sorry

end NUMINAMATH_GPT_remainder_div_1234_567_89_1011_mod_12_l123_12329


namespace NUMINAMATH_GPT_molly_christmas_shipping_cost_l123_12390

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_sisters_in_law_per_brother : ℕ := 1
def num_children_per_brother : ℕ := 2

def total_relatives : ℕ :=
  num_parents + num_brothers + (num_brothers * num_sisters_in_law_per_brother) + (num_brothers * num_children_per_brother)

theorem molly_christmas_shipping_cost : total_relatives * cost_per_package = 70 :=
by
  sorry

end NUMINAMATH_GPT_molly_christmas_shipping_cost_l123_12390


namespace NUMINAMATH_GPT_subtracted_result_correct_l123_12382

theorem subtracted_result_correct (n : ℕ) (h1 : 96 / n = 6) : 34 - n = 18 :=
by
  sorry

end NUMINAMATH_GPT_subtracted_result_correct_l123_12382


namespace NUMINAMATH_GPT_supermarket_spent_more_than_collected_l123_12356

-- Given conditions
def initial_amount : ℕ := 53
def collected_amount : ℕ := 91
def amount_left : ℕ := 14

-- Finding the total amount before shopping and amount spent in supermarket
def total_amount : ℕ := initial_amount + collected_amount
def spent_amount : ℕ := total_amount - amount_left

-- Prove that the difference between spent amount and collected amount is 39
theorem supermarket_spent_more_than_collected : (spent_amount - collected_amount) = 39 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_supermarket_spent_more_than_collected_l123_12356


namespace NUMINAMATH_GPT_decompose_series_l123_12360

-- Define the 11-arithmetic Fibonacci sequence using the given series
def Φ₁₁₀ (n : ℕ) : ℕ :=
  if n % 11 = 0 then 0 else
  if n % 11 = 1 then 1 else
  if n % 11 = 2 then 1 else
  if n % 11 = 3 then 2 else
  if n % 11 = 4 then 3 else
  if n % 11 = 5 then 5 else
  if n % 11 = 6 then 8 else
  if n % 11 = 7 then 2 else
  if n % 11 = 8 then 10 else
  if n % 11 = 9 then 1 else
  0

-- Define the two geometric progressions
def G₁ (n : ℕ) : ℤ := 3 * (8 ^ n)
def G₂ (n : ℕ) : ℤ := 8 * (4 ^ n)

-- The decomposed sequence
def decomposedSequence (n : ℕ) : ℤ := G₁ n + G₂ n

-- The theorem to prove the decomposition
theorem decompose_series : ∀ n : ℕ, Φ₁₁₀ n = decomposedSequence n := by
  sorry

end NUMINAMATH_GPT_decompose_series_l123_12360


namespace NUMINAMATH_GPT_find_c_quadratic_solution_l123_12374

theorem find_c_quadratic_solution (c : ℝ) :
  (Polynomial.eval (-5) (Polynomial.C (-45) + Polynomial.X * Polynomial.C c + Polynomial.X^2) = 0) →
  c = -4 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_find_c_quadratic_solution_l123_12374


namespace NUMINAMATH_GPT_mixed_gender_groups_l123_12387

theorem mixed_gender_groups (boys girls : ℕ) (h_boys : boys = 28) (h_girls : girls = 4) :
  ∃ groups : ℕ, (groups ≤ girls) ∧ (groups * 2 ≤ boys) ∧ groups = 4 :=
by
   sorry

end NUMINAMATH_GPT_mixed_gender_groups_l123_12387


namespace NUMINAMATH_GPT_seating_profession_solution_l123_12385

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end NUMINAMATH_GPT_seating_profession_solution_l123_12385


namespace NUMINAMATH_GPT_alice_speed_exceed_l123_12380

theorem alice_speed_exceed (d : ℝ) (t₁ t₂ : ℝ) (t₃ : ℝ) :
  d = 220 →
  t₁ = 220 / 40 →
  t₂ = t₁ - 0.5 →
  t₃ = 220 / t₂ →
  t₃ = 44 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_alice_speed_exceed_l123_12380


namespace NUMINAMATH_GPT_solve_inequality_l123_12343

theorem solve_inequality (x : ℝ) : 
  3 * (2 * x - 1) - 2 * (x + 1) ≤ 1 → x ≤ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l123_12343


namespace NUMINAMATH_GPT_f_positive_when_a_1_f_negative_solution_sets_l123_12396

section

variable (f : ℝ → ℝ) (a x : ℝ)

def f_def := f x = (x - a) * (x - 2)

-- (Ⅰ) Problem statement
theorem f_positive_when_a_1 : (∀ x, f_def f 1 x → f x > 0 ↔ (x < 1) ∨ (x > 2)) :=
by sorry

-- (Ⅱ) Problem statement
theorem f_negative_solution_sets (a : ℝ) : 
  (∀ x, f_def f a x ∧ a = 2 → False) ∧ 
  (∀ x, f_def f a x ∧ a > 2 → 2 < x ∧ x < a) ∧ 
  (∀ x, f_def f a x ∧ a < 2 → a < x ∧ x < 2) :=
by sorry

end

end NUMINAMATH_GPT_f_positive_when_a_1_f_negative_solution_sets_l123_12396


namespace NUMINAMATH_GPT_fraction_division_l123_12373

theorem fraction_division (a b : ℚ) (ha : a = 3) (hb : b = 4) :
  (1 / b) / (1 / a) = 3 / 4 :=
by 
  -- Solve the proof
  sorry

end NUMINAMATH_GPT_fraction_division_l123_12373


namespace NUMINAMATH_GPT_contains_zero_l123_12391

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end NUMINAMATH_GPT_contains_zero_l123_12391


namespace NUMINAMATH_GPT_max_value_expr_l123_12398

theorem max_value_expr (x y : ℝ) : (2 * x + 3 * y + 4) / (Real.sqrt (x^4 + y^2 + 1)) ≤ Real.sqrt 29 := sorry

end NUMINAMATH_GPT_max_value_expr_l123_12398


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l123_12313

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x + 1/y) ≥ 2 ∨ (y + 1/z) ≥ 2 ∨ (z + 1/x) ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l123_12313


namespace NUMINAMATH_GPT_find_f6_l123_12340

variable {R : Type} [LinearOrderedField R]

def f : R → R := sorry

theorem find_f6 (h1 : ∀ x y : R, f (x - y) = f x * f y) (h2 : ∀ x : R, f x ≠ 0) : f 6 = 1 :=
sorry

end NUMINAMATH_GPT_find_f6_l123_12340


namespace NUMINAMATH_GPT_james_bought_400_fish_l123_12330

theorem james_bought_400_fish
  (F : ℝ)
  (h1 : 0.80 * F = 320)
  (h2 : F / 0.80 = 400) :
  F = 400 :=
by
  sorry

end NUMINAMATH_GPT_james_bought_400_fish_l123_12330


namespace NUMINAMATH_GPT_unique_two_digit_integer_s_l123_12336

-- We define s to satisfy the two given conditions.
theorem unique_two_digit_integer_s (s : ℕ) (h1 : 13 * s % 100 = 52) (h2 : 1 ≤ s) (h3 : s ≤ 99) : s = 4 :=
sorry

end NUMINAMATH_GPT_unique_two_digit_integer_s_l123_12336


namespace NUMINAMATH_GPT_ball_count_difference_l123_12328

open Nat

theorem ball_count_difference :
  (total_balls = 145) →
  (soccer_balls = 20) →
  (basketballs > soccer_balls) →
  (tennis_balls = 2 * soccer_balls) →
  (baseballs = soccer_balls + 10) →
  (volleyballs = 30) →
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  (basketballs - soccer_balls = 5) :=
by
  intros
  let tennis_balls := 2 * soccer_balls
  let baseballs := soccer_balls + 10
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  exact sorry

end NUMINAMATH_GPT_ball_count_difference_l123_12328


namespace NUMINAMATH_GPT_g_18_equals_324_l123_12319

def is_strictly_increasing (g : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → g (n + 1) > g n

def multiplicative (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n

def m_n_condition (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m ≠ n → m > 0 → n > 0 → m ^ n = n ^ m → (g m = n ∨ g n = m)

noncomputable def g : ℕ → ℕ := sorry

theorem g_18_equals_324 :
  is_strictly_increasing g →
  multiplicative g →
  m_n_condition g →
  g 18 = 324 :=
sorry

end NUMINAMATH_GPT_g_18_equals_324_l123_12319


namespace NUMINAMATH_GPT_second_number_is_40_l123_12353

-- Defining the problem
theorem second_number_is_40
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a = (3/4 : ℚ) * b)
  (h3 : c = (5/4 : ℚ) * b) :
  b = 40 :=
sorry

end NUMINAMATH_GPT_second_number_is_40_l123_12353


namespace NUMINAMATH_GPT_complex_power_identity_l123_12397

theorem complex_power_identity (i : ℂ) (hi : i^2 = -1) :
  ( (1 + i) / (1 - i) ) ^ 2013 = i :=
by sorry

end NUMINAMATH_GPT_complex_power_identity_l123_12397


namespace NUMINAMATH_GPT_expand_product_l123_12307

theorem expand_product : ∀ (x : ℝ), (x + 2) * (x^2 - 4 * x + 1) = x^3 - 2 * x^2 - 7 * x + 2 :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_expand_product_l123_12307


namespace NUMINAMATH_GPT_height_of_pole_l123_12365

/-- A telephone pole is supported by a steel cable extending from the top of the pole to a point on the ground 3 meters from its base.
When Leah, who is 1.5 meters tall, stands 2.5 meters from the base of the pole towards the point where the cable is attached to the ground,
her head just touches the cable. Prove that the height of the pole is 9 meters. -/
theorem height_of_pole 
  (cable_length_from_base : ℝ)
  (leah_distance_from_base : ℝ)
  (leah_height : ℝ)
  : cable_length_from_base = 3 → leah_distance_from_base = 2.5 → leah_height = 1.5 → 
    (∃ height_of_pole : ℝ, height_of_pole = 9) := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_height_of_pole_l123_12365


namespace NUMINAMATH_GPT_sum_of_three_smallest_positive_solutions_l123_12308

theorem sum_of_three_smallest_positive_solutions :
  let sol1 := 2
  let sol2 := 8 / 3
  let sol3 := 7 / 2
  sol1 + sol2 + sol3 = 8 + 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_smallest_positive_solutions_l123_12308


namespace NUMINAMATH_GPT_second_supply_cost_is_24_l123_12337

-- Definitions based on the given problem conditions
def cost_first_supply : ℕ := 13
def last_year_remaining : ℕ := 6
def this_year_budget : ℕ := 50
def remaining_budget : ℕ := 19

-- Sum of last year's remaining budget and this year's budget
def total_budget : ℕ := last_year_remaining + this_year_budget

-- Total amount spent on school supplies
def total_spent : ℕ := total_budget - remaining_budget

-- Cost of second school supply
def cost_second_supply : ℕ := total_spent - cost_first_supply

-- The theorem to prove
theorem second_supply_cost_is_24 : cost_second_supply = 24 := by
  sorry

end NUMINAMATH_GPT_second_supply_cost_is_24_l123_12337


namespace NUMINAMATH_GPT_adam_cat_food_packages_l123_12388

theorem adam_cat_food_packages (c : ℕ) 
  (dog_food_packages : ℕ := 7) 
  (cans_per_cat_package : ℕ := 10) 
  (cans_per_dog_package : ℕ := 5) 
  (extra_cat_food_cans : ℕ := 55) 
  (total_dog_cans : ℕ := dog_food_packages * cans_per_dog_package) 
  (total_cat_cans : ℕ := c * cans_per_cat_package)
  (h : total_cat_cans = total_dog_cans + extra_cat_food_cans) : 
  c = 9 :=
by
  sorry

end NUMINAMATH_GPT_adam_cat_food_packages_l123_12388


namespace NUMINAMATH_GPT_ram_leela_money_next_week_l123_12315

theorem ram_leela_money_next_week (x : ℕ)
  (initial_money : ℕ := 100)
  (total_money_after_52_weeks : ℕ := 1478)
  (sum_of_series : ℕ := 1378) :
  let n := 52
  let a1 := x
  let an := x + 51
  let S := (n / 2) * (a1 + an)
  initial_money + S = total_money_after_52_weeks → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_ram_leela_money_next_week_l123_12315


namespace NUMINAMATH_GPT_least_possible_area_l123_12392

def perimeter (x y : ℕ) : ℕ := 2 * (x + y)

def area (x y : ℕ) : ℕ := x * y

theorem least_possible_area :
  ∃ (x y : ℕ), 
    perimeter x y = 120 ∧ 
    (∀ x y, perimeter x y = 120 → area x y ≥ 59) ∧ 
    area x y = 59 := 
sorry

end NUMINAMATH_GPT_least_possible_area_l123_12392


namespace NUMINAMATH_GPT_tens_digit_of_72_pow_25_l123_12324

theorem tens_digit_of_72_pow_25 : (72^25 % 100) / 10 = 3 := 
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_72_pow_25_l123_12324


namespace NUMINAMATH_GPT_two_circles_common_tangents_l123_12362

theorem two_circles_common_tangents (r : ℝ) (h_r : 0 < r) :
  ¬ ∃ (n : ℕ), n = 2 ∧
  (∀ (config : ℕ), 
    (config = 0 → n = 4) ∨
    (config = 1 → n = 0) ∨
    (config = 2 → n = 3) ∨
    (config = 3 → n = 1)) :=
by
  sorry

end NUMINAMATH_GPT_two_circles_common_tangents_l123_12362


namespace NUMINAMATH_GPT_Keith_picked_6_apples_l123_12339

def m : ℝ := 7.0
def n : ℝ := 3.0
def t : ℝ := 10.0

noncomputable def r_m := m - n
noncomputable def k := t - r_m

-- Theorem Statement confirming Keith picked 6.0 apples
theorem Keith_picked_6_apples : k = 6.0 := by
  sorry

end NUMINAMATH_GPT_Keith_picked_6_apples_l123_12339


namespace NUMINAMATH_GPT_total_time_to_make_cookies_l123_12399

def time_to_make_batter := 10
def baking_time := 15
def cooling_time := 15
def white_icing_time := 30
def chocolate_icing_time := 30

theorem total_time_to_make_cookies : 
  time_to_make_batter + baking_time + cooling_time + white_icing_time + chocolate_icing_time = 100 := 
by
  sorry

end NUMINAMATH_GPT_total_time_to_make_cookies_l123_12399


namespace NUMINAMATH_GPT_number_of_students_l123_12348

-- Definitions based on conditions
def candy_bar_cost : ℝ := 2
def chips_cost : ℝ := 0.5
def total_cost_per_student : ℝ := candy_bar_cost + 2 * chips_cost
def total_amount : ℝ := 15

-- Statement to prove
theorem number_of_students : (total_amount / total_cost_per_student) = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l123_12348


namespace NUMINAMATH_GPT_rival_awards_l123_12372

theorem rival_awards (scott_awards jessie_awards rival_awards : ℕ)
  (h1 : scott_awards = 4)
  (h2 : jessie_awards = 3 * scott_awards)
  (h3 : rival_awards = 2 * jessie_awards) :
  rival_awards = 24 :=
by sorry

end NUMINAMATH_GPT_rival_awards_l123_12372


namespace NUMINAMATH_GPT_hemisphere_surface_area_l123_12367

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h1: 0 < π) (h2: A = 3) (h3: S = 4 * π * r^2):
  ∃ t, t = 9 :=
by
  sorry

end NUMINAMATH_GPT_hemisphere_surface_area_l123_12367


namespace NUMINAMATH_GPT_sum_of_intercepts_l123_12379

theorem sum_of_intercepts (x y : ℝ) (hx : y + 3 = 5 * (x - 6)) : 
  let x_intercept := 6 + 3/5;
  let y_intercept := -33;
  x_intercept + y_intercept = -26.4 := by
  sorry

end NUMINAMATH_GPT_sum_of_intercepts_l123_12379


namespace NUMINAMATH_GPT_table_area_l123_12334

/-- Given the combined area of three table runners is 224 square inches, 
     overlapping the runners to cover 80% of a table results in exactly 24 square inches being covered by 
     two layers, and the area covered by three layers is 30 square inches,
     prove that the area of the table is 175 square inches. -/
theorem table_area (A : ℝ) (S T H : ℝ) (h1 : S + 2 * T + 3 * H = 224)
   (h2 : 0.80 * A = S + T + H) (h3 : T = 24) (h4 : H = 30) : A = 175 := 
sorry

end NUMINAMATH_GPT_table_area_l123_12334


namespace NUMINAMATH_GPT_Lewis_found_20_items_l123_12327

-- Define the number of items Tanya found
def Tanya_items : ℕ := 4

-- Define the number of items Samantha found
def Samantha_items : ℕ := 4 * Tanya_items

-- Define the number of items Lewis found
def Lewis_items : ℕ := Samantha_items + 4

-- Theorem to prove the number of items Lewis found
theorem Lewis_found_20_items : Lewis_items = 20 := by
  sorry

end NUMINAMATH_GPT_Lewis_found_20_items_l123_12327


namespace NUMINAMATH_GPT_plane_through_points_l123_12316

def point := (ℝ × ℝ × ℝ)

def plane_equation (A B C D : ℤ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_through_points : 
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1) ∧
  plane_equation A B C D 2 (-3) 5 ∧
  plane_equation A B C D (-1) (-3) 7 ∧
  plane_equation A B C D (-4) (-5) 6 ∧
  (A = 2) ∧ (B = -9) ∧ (C = 3) ∧ (D = -46) :=
sorry

end NUMINAMATH_GPT_plane_through_points_l123_12316


namespace NUMINAMATH_GPT_ratio_c_a_l123_12368

theorem ratio_c_a (a b c : ℚ) (h1 : a * b = 3) (h2 : b * c = 8 / 5) : c / a = 8 / 15 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_c_a_l123_12368


namespace NUMINAMATH_GPT_homothety_maps_C_to_E_l123_12321

-- Defining Points and Circles
variable {Point Circle : Type}
variable [Inhabited Point] -- assuming Point type is inhabited

-- Definitions for points H, K_A, I_A, K_B, I_B, K_C, I_C
variables (H K_A I_A K_B I_B K_C I_C : Point)

-- Define midpoints
def is_midpoint (A B M : Point) : Prop := sorry -- In a real proof, you would define midpoint in terms of coordinates

-- Define homothety function
def homothety (center : Point) (ratio : ℝ) (P : Point) : Point := sorry -- In a real proof, you would define the homothety transformation

-- Defining Circles
variables (C E : Circle)

-- Define circumcircle of a triangle
def is_circumcircle (a b c : Point) (circle : Circle) : Prop := sorry

-- Statements from conditions
axiom midpointA : is_midpoint H K_A I_A
axiom midpointB : is_midpoint H K_B I_B
axiom midpointC : is_midpoint H K_C I_C

axiom circumcircle_C : is_circumcircle K_A K_B K_C C
axiom circumcircle_E : is_circumcircle I_A I_B I_C E

-- Lean theorem stating the proof problem
theorem homothety_maps_C_to_E :
  ∀ (H K_A I_A K_B I_B K_C I_C : Point) (C E : Circle),
  (is_midpoint H K_A I_A) →
  (is_midpoint H K_B I_B) →
  (is_midpoint H K_C I_C) →
  (is_circumcircle K_A K_B K_C C) →
  (is_circumcircle I_A I_B I_C E) →
  (homothety H 0.5 K_A = I_A ) →
  (homothety H 0.5 K_B = I_B ) →
  (homothety H 0.5 K_C = I_C ) →
  C = E :=
by intro; sorry

end NUMINAMATH_GPT_homothety_maps_C_to_E_l123_12321


namespace NUMINAMATH_GPT_no_intersection_abs_eq_l123_12366

theorem no_intersection_abs_eq (x : ℝ) : ∀ y : ℝ, y = |3 * x + 6| → y = -|2 * x - 4| → false := 
by
  sorry

end NUMINAMATH_GPT_no_intersection_abs_eq_l123_12366


namespace NUMINAMATH_GPT_balloon_permutations_l123_12310

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end NUMINAMATH_GPT_balloon_permutations_l123_12310


namespace NUMINAMATH_GPT_c_n_monotonically_decreasing_l123_12375

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ)

theorem c_n_monotonically_decreasing 
    (h_a0 : a 0 = 0)
    (h_b : ∀ n ≥ 1, b n = a n - a (n - 1))
    (h_c : ∀ n ≥ 1, c n = a n / n)
    (h_bn_decrease : ∀ n ≥ 1, b n ≥ b (n + 1)) : 
    ∀ n ≥ 2, c n ≤ c (n - 1) := 
by
  sorry

end NUMINAMATH_GPT_c_n_monotonically_decreasing_l123_12375


namespace NUMINAMATH_GPT_largest_stores_visited_l123_12384

theorem largest_stores_visited 
  (stores : ℕ) (total_visits : ℕ) (shoppers : ℕ) 
  (two_store_visitors : ℕ) (min_visits_per_person : ℕ)
  (h1 : stores = 8)
  (h2 : total_visits = 22)
  (h3 : shoppers = 12)
  (h4 : two_store_visitors = 8)
  (h5 : min_visits_per_person = 1)
  : ∃ (max_stores : ℕ), max_stores = 3 := 
by 
  -- Define the exact details given in the conditions
  have h_total_two_store_visits : two_store_visitors * 2 = 16 := by sorry
  have h_remaining_visits : total_visits - 16 = 6 := by sorry
  have h_remaining_shoppers : shoppers - two_store_visitors = 4 := by sorry
  have h_each_remaining_one_visit : 4 * 1 = 4 := by sorry
  -- Prove the largest number of stores visited by any one person is 3
  have h_max_stores : 1 + 2 = 3 := by sorry
  exact ⟨3, h_max_stores⟩

end NUMINAMATH_GPT_largest_stores_visited_l123_12384
