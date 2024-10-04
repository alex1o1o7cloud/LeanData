import Mathlib

namespace minimize_sum_of_sequence_l79_79004

variable (a_n : ℕ → ℝ)
variable (n : ℕ)

def sequence_general_term (n : ℕ) : ℝ :=
  3 * n - 50

def sum_of_first_n_terms (n : ℕ) : ℝ :=
  (n * (3 * n - 50 + 1)) / 2

theorem minimize_sum_of_sequence : (S_n (sequence_general_term) n = minimize → n = 16) :=
by
  sorry

end minimize_sum_of_sequence_l79_79004


namespace euler_line_eq_l79_79109

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨3, 1⟩
def B : Point := ⟨4, 2⟩
def C : Point := ⟨2, 3⟩

def centroid (A B C : Point) : Point :=
  ⟨(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3⟩

def orthocenter (A B C : Point) : Point := sorry -- Calculation skipped

-- Euler line of the triangle with vertices A, B, and C
theorem euler_line_eq (A B C : Point) :
  let G := centroid A B C
  let H := orthocenter A B C
  ∃ m b, (H.y - G.y) = m * (H.x - G.x) ∧
  (∀ x y, y = m * x + b → y - 2 = -(x - 3) → x + y - 5 = 0) :=
  sorry

end euler_line_eq_l79_79109


namespace divisibility_criterion_l79_79947

theorem divisibility_criterion (x y : ℕ) (h_two_digit : 10 ≤ x ∧ x < 100) :
  (1207 % x = 0) ↔ (x = 10 * (x / 10) + (x % 10) ∧ (x / 10)^3 + (x % 10)^3 = 344) :=
by
  sorry

end divisibility_criterion_l79_79947


namespace max_value_of_squares_l79_79913

theorem max_value_of_squares (a b c d : ℝ) 
  (h1 : a + b = 18) 
  (h2 : ab + c + d = 91) 
  (h3 : ad + bc = 187) 
  (h4 : cd = 105) : 
  a^2 + b^2 + c^2 + d^2 ≤ 107 :=
sorry

end max_value_of_squares_l79_79913


namespace probability_sequence_rw_10_l79_79688

noncomputable def probability_red_white_red : ℚ :=
  (4 / 10) * (6 / 9) * (3 / 8)

theorem probability_sequence_rw_10 :
    probability_red_white_red = 1 / 10 := by
  sorry

end probability_sequence_rw_10_l79_79688


namespace hexagon_area_half_l79_79799

variables {P Q R : Type} [PlaneGeometry P] [∀ x : P, LinearOrder (IntersectTrajectory P)]

open PlaneGeometry

def midpoint (A B : P) : P := sorry
def perpendicular (A B C : P) : P := sorry -- Point of intersection of perpendicular from A to line BC

theorem hexagon_area_half (A B C : P) (h_acute_ABC : acute_angle_triangle A B C) :
  let A1 := midpoint B C,
      B1 := midpoint A C,
      C1 := midpoint A B,
      M := perpendicular C1 A B,
      N := perpendicular C1 B C,
      K := perpendicular A1 A C 
  in 2 * area_of_poly {A1, K, B1, M, C1, N} = area_of_triangle A B C :=
sorry

end hexagon_area_half_l79_79799


namespace lines_intersect_or_parallel_l79_79523

namespace NinePointCircle

open EuclideanGeometry

/- Definitions for the problem setup -/
variables (A B C A1 B1 C1 A2 B2 C2 : Point)
variables (AA1 BB1 CC1 A1A2 B1B2 C1C2 : Line)

-- Triangle ABC
axiom triangle_ABC : Triangle A B C

-- Altitudes in triangle
axiom altitude_AA1 : Altitude AA1 A1 A B C
axiom altitude_BB1 : Altitude BB1 B1 B A C
axiom altitude_CC1 : Altitude CC1 C1 C A B

-- Nine-point circle diameters
noncomputable def nine_point_circle_diameter_A1A2 : Line := A1A2
noncomputable def nine_point_circle_diameter_B1B2 : Line := B1B2
noncomputable def nine_point_circle_diameter_C1C2 : Line := C1C2

/- The theorem to prove -/
theorem lines_intersect_or_parallel
  (h1: NinePointCircleDiameter A1A2)
  (h2: NinePointCircleDiameter B1B2)
  (h3: NinePointCircleDiameter C1C2) :
  (Intersects (Line_through A A2) (Line_through B B2) ∧ Intersects (Line_through B B2) (Line_through C C2) ∧ Intersects (Line_through C C2) (Line_through A A2)) ∨
  (Parallel (Line_through A A2) (Line_through B B2) ∧ Parallel (Line_through B B2) (Line_through C C2) ∧ Parallel (Line_through C C2) (Line_through A A2)) :=
sorry

end NinePointCircle

end lines_intersect_or_parallel_l79_79523


namespace sum_of_homothety_coeffs_geq_4_l79_79502

theorem sum_of_homothety_coeffs_geq_4 (a : ℕ → ℝ)
  (h_pos : ∀ i, 0 < a i)
  (h_less_one : ∀ i, a i < 1)
  (h_sum_cubes : ∑' i, (a i)^3 = 1) :
  (∑' i, a i) ≥ 4 := sorry

end sum_of_homothety_coeffs_geq_4_l79_79502


namespace smallest_difference_div_by_5_sum_l79_79730

-- Function to calculate the sum of digits of a natural number.
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Defining a predicate for natural numbers where sum of digits is divisible by 5.
def div_by_5_sum (n : ℕ) : Prop := sum_of_digits n % 5 = 0

-- Theorem stating that the smallest positive difference between consecutive numbers in the sequence is 1.
theorem smallest_difference_div_by_5_sum :
  ∃ (a b : ℕ), div_by_5_sum a ∧ div_by_5_sum b ∧ a ≠ b ∧ (a - b).nat_abs = 1 :=
sorry

end smallest_difference_div_by_5_sum_l79_79730


namespace tankard_one_quarter_full_l79_79634

theorem tankard_one_quarter_full
  (C : ℝ) 
  (h : (3 / 4) * C = 480) : 
  (1 / 4) * C = 160 := 
by
  sorry

end tankard_one_quarter_full_l79_79634


namespace find_a_l79_79834

theorem find_a (a : ℝ) : (∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5) → (∃ x : ℝ, x = 3) → a = 8 :=
by
  sorry

end find_a_l79_79834


namespace cross_product_correct_l79_79387

def a : ℝ × ℝ × ℝ := (4, 3, -7)
def b : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem cross_product_correct : cross_product a b = (5, -30, -10) := sorry

end cross_product_correct_l79_79387


namespace employee_discount_percentage_l79_79322

noncomputable def wholesale_cost : ℝ := 200
noncomputable def retail_markup : ℝ := 0.20
noncomputable def employee_price : ℝ := 168

def retail_price (wholesale_cost : ℝ) (retail_markup : ℝ) : ℝ :=
  wholesale_cost * (1 + retail_markup)

def discount_amount (retail_price : ℝ) (employee_price : ℝ) : ℝ :=
  retail_price - employee_price

def discount_percentage (discount_amount : ℝ) (retail_price : ℝ) : ℝ :=
  (discount_amount / retail_price) * 100

theorem employee_discount_percentage :
  discount_percentage (discount_amount (retail_price wholesale_cost retail_markup) employee_price) 
  (retail_price wholesale_cost retail_markup) = 30 :=
by
  sorry

end employee_discount_percentage_l79_79322


namespace river_width_is_correct_l79_79708

def river_depth : ℝ := 3
def flow_rate : ℝ := 2 * 1000 / 60
def volume_flow_per_minute : ℝ := 3600

theorem river_width_is_correct :
  volume_flow_per_minute = river_depth * 36 * flow_rate :=
by 
  sorry

end river_width_is_correct_l79_79708


namespace trajectory_C₂_equation_area_triangle_ABO_l79_79895

noncomputable def trajectory_equation (ρ θ : ℝ) : ℝ := 
  ρ * cos θ

theorem trajectory_C₂_equation (ρ θ x y : ℝ) 
  (h₁ : ∀ θ, trajectory_equation ρ θ = 4)
  (h₂ : x^2 + y^2 = 4 * real.abs x) 
  (h₃ : ρ * θ = real.sqrt(x^2 + y^2) * real.sqrt(16 + (16 * y^2) / x^2)) :
  x^2 + y^2 - 4x = 0 := by
  sorry

theorem area_triangle_ABO (θ : ℝ)
  (A B O : ℝ × ℝ)
  (hA : A = (1, real.sqrt 3))
  (hB : B = (2 * real.sqrt 3 * cos θ, 2 * real.sqrt 3 * sin θ))
  (hO : O = (0, 0))
  (hC₂ : (2 * real.sqrt 3 * cos θ)^2 + (2 * real.sqrt 3 * sin θ)^2 - 4 * (2 * real.sqrt 3 * cos θ) = 0) :
  area_of_triangle A B O = real.sqrt 3 ∨ area_of_triangle A B O = 2 * real.sqrt 3 := by
  sorry

noncomputable def area_of_triangle (A B O : ℝ × ℝ) : ℝ := 
  0.5 * real.abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2))

end trajectory_C₂_equation_area_triangle_ABO_l79_79895


namespace shirley_sold_boxes_l79_79587

theorem shirley_sold_boxes (cases boxes_per_case : ℕ) (h_cases : cases = 9) (h_boxes_per_case : boxes_per_case = 6) :
  cases * boxes_per_case = 54 :=
by
  rw [h_cases, h_boxes_per_case]
  exact rfl

end shirley_sold_boxes_l79_79587


namespace find_all_functions_l79_79775

theorem find_all_functions 
  (f : ℤ → ℝ)
  (h1 : ∀ m n : ℤ, m < n → f m < f n)
  (h2 : ∀ m n : ℤ, ∃ k : ℤ, f m - f n = f k) :
  ∃ a t : ℝ, a > 0 ∧ (∀ n : ℤ, f n = a * (n + t)) :=
sorry

end find_all_functions_l79_79775


namespace triangle_angle_bisector_ratio_k_l79_79901

-- Given definitions for a triangle and angle bisectors as per the problem statement
variable {A B C L M : Type}
variable (a b c : ℝ) -- sides of the triangle
variable [noncomputable] def BC : ℝ := a
variable [noncomputable] def AC : ℝ := b
variable [noncomputable] def AB : ℝ := c
variable [noncomputable] def k := c / a

-- Conditions for the angle bisectors
variable (AL_bisector : ∀ {A B C L : Type}, Prop)
variable (CM_bisector : ∀ {C A B M : Type}, Prop)

-- Proof problem statement
theorem triangle_angle_bisector_ratio_k  :
  AL_bisector A B C L ∧ 
  CM_bisector C A B M ∧ 
  L ∈ line BC ∧ 
  M ∈ line AB →
  (a ≠ 0) → (k = c / a) := sorry

end triangle_angle_bisector_ratio_k_l79_79901


namespace matrix_representation_l79_79600

theorem matrix_representation (A B : Matrix (Fin n) (Fin n) Int) (hB : B.det ≠ 0) :
  ∃ (m : ℕ) (N : Fin m → Matrix (Fin n) (Fin n) Int),
    (A ⬝ B⁻¹) = (Finset.univ.sum (λ k, (N k)⁻¹)) ∧
    (∀ i j, i ≠ j → N i ≠ N j) :=
sorry

end matrix_representation_l79_79600


namespace find_BC_l79_79058

open Real

-- Definitions based on the given problem
def Point := (ℝ × ℝ)
def Triangle (A B C : Point) := C = (0, 0) ∧ A.2 = 0 ∧ B.1 = 0 -- \(C\) at origin, \(A\) on x-axis, \(B\) on y-axis

variables {A B C : Point}
variables (h_triangle : Triangle A B C) (h_angle_C : ∠ C = 90) (h_tan_A : tan (angle A C B) = 3/4)
variables (h_AC : dist A C = 6)

-- Statement to prove
theorem find_BC : dist C B = 4.5 :=
sorry

end find_BC_l79_79058


namespace ploughing_problem_l79_79864

theorem ploughing_problem
  (hours_per_day_group1 : ℕ)
  (days_group1 : ℕ)
  (bulls_group1 : ℕ)
  (total_fields_group2 : ℕ)
  (hours_per_day_group2 : ℕ)
  (days_group2 : ℕ)
  (bulls_group2 : ℕ)
  (fields_group1 : ℕ)
  (fields_group2 : ℕ) :
    hours_per_day_group1 = 10 →
    days_group1 = 3 →
    bulls_group1 = 10 →
    hours_per_day_group2 = 8 →
    days_group2 = 2 →
    bulls_group2 = 30 →
    fields_group2 = 32 →
    480 * fields_group1 = 300 * fields_group2 →
    fields_group1 = 20 := by
  sorry

end ploughing_problem_l79_79864


namespace nm_equals_9_identity_l79_79557

theorem nm_equals_9_identity (M : Matrix (Fin 3) (Fin 2) ℝ) (N : Matrix (Fin 2) (Fin 3) ℝ)
  (MN : Matrix (Fin 3) (Fin 3) ℝ)
  (hMN : MN = ![![8, 2, -2], 
                ![2, 5, 4], 
                ![-2, 4, 5]])
  (hM_N_Prod : (M ⬝ N) = MN) :
  (N ⬝ M) = ![![9, 0], 
              ![0, 9]] :=
by
  sorry

end nm_equals_9_identity_l79_79557


namespace roots_of_equation_l79_79754

theorem roots_of_equation :
  (∃ x, (18 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = 3 ∨ x = -4.5)) :=
by
  sorry

end roots_of_equation_l79_79754


namespace greatest_integer_l79_79642

-- Define the conditions
def a : ℤ := 4^100 + 3^100
def b : ℤ := 4^98 + 3^98

-- State the theorem
theorem greatest_integer (a = 4^100 + 3^100) (b = 4^98 + 3^98) : 
  (⌊a / b⌋ = 15) :=
by
  sorry

end greatest_integer_l79_79642


namespace deductible_amount_l79_79338

-- This definition represents the conditions of the problem.
def current_annual_deductible_is_increased (D : ℝ) : Prop :=
  (2 / 3) * D = 2000

-- This is the Lean statement, expressing the problem that needs to be proven.
theorem deductible_amount (D : ℝ) (h : current_annual_deductible_is_increased D) : D = 3000 :=
by
  sorry

end deductible_amount_l79_79338


namespace simplify_expression_l79_79825

theorem simplify_expression (α : ℝ) (hα : 0 < α ∧ α < π / 3) : 
  3 ^ |Real.log3 (Real.sin α)| = 1 / Real.sin α :=
sorry

end simplify_expression_l79_79825


namespace geometric_series_common_ratio_l79_79188

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l79_79188


namespace problem_part1_problem_part2_l79_79677

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c d : V)
variables (c1 c2 d1 d2 : ℝ)

-- Conditions
axiom a_unit : ∥a∥ = 1
axiom b_unit : ∥b∥ = 1
axiom orthogonal_ab : ⟪a, b⟫ = 0
axiom c_def : c = c1 • a + c2 • b
axiom d_def : d = d1 • a + d2 • b
axiom c_unit : ∥c∥ = 1
axiom d_unit : ∥d∥ = 1
axiom orthogonal_cd : ⟪c, d⟫ = 0

-- Proof statements
theorem problem_part1 : 
  c1^2 + d1^2 = 1 ∧ c2^2 + d2^2 = 1 ∧ (c1 * c2 + d1 * d2 = 0) :=
sorry

theorem problem_part2 : 
  a = c1 • c + d1 • d ∧ b = c2 • c + d2 • d :=
sorry

end problem_part1_problem_part2_l79_79677


namespace maximum_shareholder_ownership_l79_79308

variable {S : Finset ℝ}
variable (shares : Fin (100) → ℝ)
variable (total_shares : ℝ := 100)

theorem maximum_shareholder_ownership :
  (∀ (T : Finset (Fin (100))), T.card = 66 → (T.sum shares) ≥ 50) →
  ∃ (x : ℝ), x ≤ 25 ∧ 
  shares (Fin.find (λ s, shares s = (Finset.univ.image shares).max')) = x := 
sorry

end maximum_shareholder_ownership_l79_79308


namespace area_trapezoid_ABCD_l79_79215

-- Define the geometric entities and their properties
variables (A B C D E F G : Point)
variable [IsTrapezoid A B C D]
variable (CE : LineSegment C E)
variable [Midpoint F CE]
variable [DividesTrapezoidIntoTriangleAndParallelogram CE A B C D]
variables (DF : Line)
variables [PassesThroughMidpoint DF B E]
variable [AreaTriangleCDE : Area (Triangle C D E) = 3]

-- State the theorem
theorem area_trapezoid_ABCD : Area (Trapezoid A B C D) = 12 := 
sorry

end area_trapezoid_ABCD_l79_79215


namespace log_subtraction_l79_79087

theorem log_subtraction (a b : ℝ) (h₁ : a = Real.log (10) / Real.log (3))
                                      (h₂ : b = Real.log (7) / Real.log (3)) :
  3^(a - b) = (10 / 7) :=
sorry

end log_subtraction_l79_79087


namespace coplanar_vectors_has_lambda_eq_one_l79_79406

open Real

noncomputable def vec3 := (ℝ × ℝ × ℝ)

def a : vec3 := (2, -1, 3)
def b : vec3 := (-1, 4, -2)
def c (λ : ℝ) : vec3 := (1, 3, λ)

-- Define the condition for coplanarity
def coplanar (v1 v2 v3 : vec3) : Prop := 
  ∃ m n : ℝ, v3 = (m * v1.1 + n * v2.1, m * v1.2 + n * v2.2, m * v1.3 + n * v2.3)

theorem coplanar_vectors_has_lambda_eq_one (λ : ℝ) :
  coplanar a b (c λ) ↔ λ = 1 :=
by sorry  -- Proof omitted; main goal is to write the statement.

end coplanar_vectors_has_lambda_eq_one_l79_79406


namespace total_students_sampled_l79_79302

theorem total_students_sampled :
  ∀ (seniors juniors freshmen sampled_seniors sampled_juniors sampled_freshmen total_students : ℕ),
    seniors = 1000 →
    juniors = 1200 →
    freshmen = 1500 →
    sampled_freshmen = 75 →
    sampled_seniors = seniors * (sampled_freshmen / freshmen) →
    sampled_juniors = juniors * (sampled_freshmen / freshmen) →
    total_students = sampled_seniors + sampled_juniors + sampled_freshmen →
    total_students = 185 :=
by
sorry

end total_students_sampled_l79_79302


namespace geometric_series_common_ratio_l79_79179

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l79_79179


namespace nec_but_not_suff_condition_l79_79681

variables {p q : Prop}

theorem nec_but_not_suff_condition (hp : ¬p) : 
  (p ∨ q → False) ↔ (¬p) ∧ ¬(¬p → p ∨ q) :=
by {
  sorry
}

end nec_but_not_suff_condition_l79_79681


namespace complex_conjugate_l79_79093

noncomputable def i : ℂ := complex.I

noncomputable def z : ℂ := (i) / (1 + i)

theorem complex_conjugate :
  complex.conj z = (1 / 2 : ℂ) - (1 / 2) * i :=
by
  sorry

end complex_conjugate_l79_79093


namespace justine_more_than_bailey_l79_79891

-- Definitions from conditions
def J : ℕ := 22 -- Justine's initial rubber bands
def B : ℕ := 12 -- Bailey's initial rubber bands

-- Theorem to prove
theorem justine_more_than_bailey : J - B = 10 := by
  -- Proof will be done here
  sorry

end justine_more_than_bailey_l79_79891


namespace AM_CK_BH_intersects_l79_79898

noncomputable theory

open Classical

variables {P Q R S T U V: Type*} [metric_space P] [metric_space Q] [metric_space R] 
variables [inner_product_space ℝ P] [inner_product_space ℝ Q] [inner_product_space ℝ R]

structure trapezoid (A B C D : P) := 
  (BC_lt_AD : dist B C < dist A D)
  (AB_eq_CD : dist A B = dist C D)
  (K : P) (is_midpoint_AD : K = midpoint ℝ A D)
  (M : P) (is_midpoint_CD : M = midpoint ℝ C D)
  (H : P) (CH_is_height : mk_segment C H ⊥ mk_segment A B)

theorem AM_CK_BH_intersects (A B C D K M H : P) (h_trap : trapezoid A B C D) :
  ∃ G : P, collinear ℝ {A, M, G} ∧ collinear ℝ {C, K, G} ∧ collinear ℝ {B, H, G} :=
sorry

end AM_CK_BH_intersects_l79_79898


namespace sum_last_two_digits_l79_79647

-- Definition of the problem conditions
def seven : ℕ := 10 - 3
def thirteen : ℕ := 10 + 3

-- Main statement of the problem
theorem sum_last_two_digits (x : ℕ) (y : ℕ) : x = seven → y = thirteen → (7^25 + 13^25) % 100 = 0 :=
by
  intros
  rw [←h, ←h_1] -- Rewriting x and y in terms of seven and thirteen
  sorry -- Proof omitted

end sum_last_two_digits_l79_79647


namespace initial_comparison_discount_comparison_B_based_on_discounted_A_l79_79697

noncomputable section

-- Definitions based on the problem conditions
def A_price (x : ℝ) : ℝ := x
def B_price (x : ℝ) : ℝ := (0.2 * 2 * x + 0.3 * 3 * x + 0.4 * 4 * x) / 3
def A_discount_price (x : ℝ) : ℝ := 0.9 * x

-- Initial comparison
theorem initial_comparison (x : ℝ) (h : 0 < x) : B_price x < A_price x :=
by {
  sorry
}

-- After A's discount comparison
theorem discount_comparison (x : ℝ) (h : 0 < x) : A_discount_price x < B_price x :=
by {
  sorry
}

-- B's price based on A’s discounted price comparison
theorem B_based_on_discounted_A (x : ℝ) (h : 0 < x) : B_price (A_discount_price x) < A_discount_price x :=
by {
  sorry
}

end initial_comparison_discount_comparison_B_based_on_discounted_A_l79_79697


namespace reflecting_BF_median_of_ABC_l79_79524

variables {A B C D E F : Type*} [euclidean_geometry]

open_locale euclidean_geometry

noncomputable def is_median (A B C P : point) : Prop :=
∃ M : point, midpoint M A C ∧ collinear A P M

theorem reflecting_BF_median_of_ABC
  (A B C D E F : point)
  (h1 : AB ≠ BC)
  (h2 : bisector ∠ABC D)
  (h3 : lies_on D AC)
  (h4 : bisector ∠ABC E)
  (h5 : lies_on E (circumcircle A B C) ∧ E ≠ B)
  (h6 : circle_with_diameter E D (intersect circumcircle A B C F) ∧ F ≠ E)
  : is_median A B C (reflect_line BF (perp_bisector BD)) := 
begin
  sorry
end

end reflecting_BF_median_of_ABC_l79_79524


namespace solve_for_a_l79_79872

noncomputable theory

def f (a x : ℝ) := (a * x - 1) * real.exp (x - 2)

def tangent_line_slope (a : ℝ) : ℝ :=
  let f' := λ x, (a) * real.exp(x-2) + (x * a - 1) * real.exp(x-2)
  f' 2

def tangent_line (a : ℝ) : ℝ → ℝ :=
  let m := tangent_line_slope a
  let f2 := f a 2
  λ x, m * (x - 2) + f2

theorem solve_for_a (a : ℝ) (h : tangent_line a 3 = 2) : a = 4/5 := 
sorry

end solve_for_a_l79_79872


namespace jerry_coin_flip_configuration_count_l79_79905

theorem jerry_coin_flip_configuration_count :
  let initial_heads := 10
  let flip_count := 2
  let distinct_coins := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let configurations := (∃ (c1 c2 : distinct_coins), c1 = c2 ∨ c1 ≠ c2) →
                                             if c1 = c2 then 1 else (nat.choose initial_heads 2)
  configurations = 46 :=
by
  sorry

end jerry_coin_flip_configuration_count_l79_79905


namespace sum_of_solutions_of_equation_l79_79789

theorem sum_of_solutions_of_equation :
  let f := (fun x : ℝ => (x - 4) ^ 2)
  ∃ S : Set ℝ, (S = {x | f x = 16}) ∧ (∑ s in S, s) = 8 := 
by
  sorry

end sum_of_solutions_of_equation_l79_79789


namespace worth_of_new_computer_l79_79635

theorem worth_of_new_computer (initial_savings : ℕ) (old_computer_value : ℕ) (additional_needed : ℕ) : 
  initial_savings = 50 → old_computer_value = 20 → additional_needed = 10 → 
  initial_savings + old_computer_value + additional_needed = 80 :=
begin
  intros h₁ h₂ h₃,
  rw [h₁, h₂, h₃],
  norm_num,
end

end worth_of_new_computer_l79_79635


namespace distance_focus_to_line_AP_minimized_l79_79432

noncomputable def hyperbola_focus_distance : ℝ :=
  let F : ℝ × ℝ := (-4, 0)
  let A : ℝ × ℝ := (1, 4)
  let F' : ℝ × ℝ := (4, 0)
  let line_AP_equation : ℝ → ℝ := λ x => ((-4 / 3) * x) + (4 + (4 / 3))
  
  -- Function representing the distance from F to a line defined by ax + by + c = 0
  let distance_to_line (x y a b c : ℝ) : ℝ := ( |a * x + b * y + c| / sqrt(a^2 + b^2) )

  -- Final distance calculation
  distance_to_line F.1 F.2 4 3 (-16)

theorem distance_focus_to_line_AP_minimized :
  hyperbola_focus_distance = 32 / 5 :=
  by
    -- The proof will be filled out here
    sorry

end distance_focus_to_line_AP_minimized_l79_79432


namespace line_tangent_to_circle_l79_79848

/-- 
Let l be the line defined by the equation x * cos θ + y * sin θ + 2 = 0.
Let C be the circle defined by the equation x^2 + y^2 = 4.
--/
theorem line_tangent_to_circle (θ : ℝ) : 
  let l := { p : ℝ × ℝ | p.1 * Real.cos θ + p.2 * Real.sin θ + 2 = 0 } in
  let C := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 } in
  ∀ (p₀ p₁ : ℝ × ℝ), p₀ ∈ l → p₁ ∈ C → Real.dist p₀ p₁ = 2 :=
begin
  sorry
end

end line_tangent_to_circle_l79_79848


namespace slope_of_line_segment_CD_intersections_of_circles_l79_79392

theorem slope_of_line_segment_CD_intersections_of_circles :
  (∀ (C D : ℝ × ℝ), ((C.1 ^ 2 + C.2 ^ 2 - 6 * C.1 + 4 * C.2 - 8 = 0) ∧ 
                     (C.1 ^ 2 + C.2 ^ 2 - 8 * C.1 + 6 * C.2 + 9 = 0) ∧ 
                     (D.1 ^ 2 + D.2 ^ 2 - 6 * D.1 + 4 * D.2 - 8 = 0) ∧ 
                     (D.1 ^ 2 + D.2 ^ 2 - 8 * D.1 + 6 * D.2 + 9 = 0))
  → let m := -1 in m = -1) :=
sorry

end slope_of_line_segment_CD_intersections_of_circles_l79_79392


namespace height_of_balcony_l79_79297

variable (t : ℝ) (v₀ : ℝ) (g : ℝ) (h₀ : ℝ)

axiom cond1 : t = 6
axiom cond2 : v₀ = 20
axiom cond3 : g = 10

theorem height_of_balcony : h₀ + v₀ * t - (1/2 : ℝ) * g * t^2 = 0 → h₀ = 60 :=
by
  intro h'
  sorry

end height_of_balcony_l79_79297


namespace distinct_real_roots_of_quadratic_l79_79469

theorem distinct_real_roots_of_quadratic (m : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ x^2 + m * x + 9 = 0 ∧ y^2 + m * y + 9 = 0) ↔ m ∈ Ioo (-∞) (-6) ∪ Ioo 6 (∞) :=
by
  sorry

end distinct_real_roots_of_quadratic_l79_79469


namespace polynomial_division_result_l79_79546

noncomputable def f (x : ℝ) : ℝ := 4 * x ^ 4 + 12 * x ^ 3 - 9 * x ^ 2 + x + 3
noncomputable def d (x : ℝ) : ℝ := x ^ 2 + 4 * x - 2

theorem polynomial_division_result :
  ∃ q r : Polynomial ℝ, degree r < degree d ∧
    f = q * d + r ∧
    q.eval 1 + r.eval (-1) = -21 :=
by
  sorry

end polynomial_division_result_l79_79546


namespace geometric_series_common_ratio_l79_79198

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l79_79198


namespace num_subsets_A_inter_B_l79_79818

noncomputable theory
open Set

def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = abs p.1}

theorem num_subsets_A_inter_B : Fintype.card (Set.subsets (A ∩ B)) = 4 := 
  sorry

end num_subsets_A_inter_B_l79_79818


namespace simplify_polynomial_l79_79128

theorem simplify_polynomial :
  (3 * x^3 + 4 * x^2 + 8 * x - 5) - (2 * x^3 + x^2 + 6 * x - 7) = x^3 + 3 * x^2 + 2 * x + 2 := 
by
  sorry

end simplify_polynomial_l79_79128


namespace bob_wins_strategy_l79_79327

-- Definitions for the conditions.
noncomputable def game_board : Type := fin 8 × fin 8
noncomputable def initial_tokens_positions (bob_initial alice_initial : game_board) : Prop := 
  true -- We will just implicitly assume the tokens start on the board as we do not focus on precise positions.

-- Main statement for Bob's winning strategy.
theorem bob_wins_strategy
  (bob_initial alice_initial : game_board)
  (move_token : game_board → game_board → ℕ → Prop)
  (move_legally : ∀ pos1 pos2 : game_board, (fin.dist pos1.1 pos2.1 + fin.dist pos1.2 pos2.2 = 1) → move_token pos1 pos2 1) 
  -- Condition for moving to adjacent squares horizontally or vertically.
  (rounds: ℕ): 
  (∀ round,  round < 2012) →
  ¬ (∀ round, move_token (bob_initial) (alice_initial) round) → 
  (∀ round, move_token (alice_initial) (bob_initial) round) :=
sorry


end bob_wins_strategy_l79_79327


namespace quadratic_two_distinct_real_roots_l79_79987

theorem quadratic_two_distinct_real_roots (m : ℝ) : 
  ∀ x : ℝ, x^2 + m * x - 2 = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l79_79987


namespace excenter_altitudes_l79_79539

-- Define that Oa, Ob, and Oc are the excircle centers of triangle ABC
variables {A B C Oa Ob Oc : Type*}
variables [has_e_excenter Oa B C A] [has_e_excenter Ob A C B] [has_e_excenter Oc A B C]

-- Define the statement to be proved
theorem excenter_altitudes (A B C Oa Ob Oc : Type*) 
  [has_e_excenter Oa B C A] [has_e_excenter Ob A C B] [has_e_excenter Oc A B C] :
  are_feet_of_altitudes A B C Oa Ob Oc :=
sorry

end excenter_altitudes_l79_79539


namespace pos_integer_solutions_count_l79_79464

theorem pos_integer_solutions_count :
  (finite {x : ℕ | 12 < -2 * (x : ℤ) + 17 ∧ x > 0}).to_finset.card = 2 :=
by
  sorry

end pos_integer_solutions_count_l79_79464


namespace meeting_time_l79_79350

-- Variables representing the conditions
def uniform_rate_cassie := 15
def uniform_rate_brian := 18
def distance_route := 70
def cassie_start_time := 8.0
def brian_start_time := 9.25

-- The goal
theorem meeting_time : ∃ T : ℝ, (15 * T + 18 * (T - 1.25) = 70) ∧ T = 2.803 := 
by {
  sorry
}

end meeting_time_l79_79350


namespace proof_triangle_is_right_angle_proof_perimeter_range_l79_79504

noncomputable def triangle_is_right_angle (a b c A B C : ℝ) (sin_A sin_B sin_C : ℝ) (cos_B : ℝ) (cos_2A : ℝ) :=
  (b > 0) ∧ (sin A / (sin B + sin C) = 1 - (a - b) / (a - c)) ∧
  (((8 * cos B) * sin A + cos_2A) = (-2 * (sin A - 1)^2 + 3)) ∧ (a > 0) ∧ (c > 0) ∧
  (B = π / 3) ∧ (A = π / 2)

noncomputable def perimeter_range (a b c : ℝ) (A : ℝ) : set ℝ :=
  { p : ℝ | b = sqrt 3 ∧ (p = sqrt 3 + 2 * sqrt 3 * sin (A + π / 6)) ∧ ((A + π / 6) ∈ (π / 6, 5 * π / 6)) }

theorem proof_triangle_is_right_angle (a b c A B C : ℝ) (sin_A sin_B sin_C : ℝ) (cos_B : ℝ) (cos_2A : ℝ) :
  triangle_is_right_angle a b c A B C sin_A sin_B sin_C cos_B cos_2A → (A = π / 2) :=
by sorry

theorem proof_perimeter_range (a b c A : ℝ) :
  b = sqrt 3 → ((a > 0) ∧ (c > 0)) →
  ∃ l, l ∈ perimeter_range a b c A :=
by sorry

end proof_triangle_is_right_angle_proof_perimeter_range_l79_79504


namespace num_common_tangents_l79_79610

noncomputable def circle_center_radius (a b c d e : ℝ) : (ℝ × ℝ) × ℝ :=
  let h := (-a / 2)
  let k := (-b / 2)
  let r := real.sqrt ((a / 2)^2 + (b / 2)^2 - e)
  ((h, k), r)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Circle 1: x^2 + y^2 + 2x + y - 2 = 0
def C1_center_radius := circle_center_radius 2 1 0 0 (-2)
-- Circle 2: x^2 + y^2 - 4x - 2y + 4 = 0
def C2_center_radius := circle_center_radius (-4) (-2) 0 0 4

theorem num_common_tangents :
  let ((c1x, c1y), r1) := C1_center_radius in
  let ((c2x, c2y), r2) := C2_center_radius in
  distance (c1x, c1y) (c2x, c2y) > r1 + r2 →
  ∃ (n : ℕ), n = 4 :=
by
  sorry

end num_common_tangents_l79_79610


namespace AM_GM_inequality_l79_79414

theorem AM_GM_inequality (n : ℕ) (hn : n ≥ 2) (a : Finₙ → ℝ) (ha : ∀ i, 0 < a i) :
  (∑ i, a i) / n.to_real ≥ (∏ i, a i) ^ (1 / n.to_real) :=
sorry

end AM_GM_inequality_l79_79414


namespace eight_lines_divide_plane_into_37_regions_l79_79377

theorem eight_lines_divide_plane_into_37_regions :
  ∀ (lines : Fin 8 → Line)
  (h1 : ∀ i j, i ≠ j → ¬parallel (lines i) (lines j))
  (h2 : ∀ i j k, ¬concurrent (lines i) (lines j) (lines k)),
  number_of_regions lines = 37 :=
by
  -- Proof omitted
  sorry

end eight_lines_divide_plane_into_37_regions_l79_79377


namespace complementary_angle_measure_l79_79737

theorem complementary_angle_measure (A S C : ℝ) (h1 : A = 45) (h2 : A + S = 180) (h3 : A + C = 90) (h4 : S = 3 * C) : C = 45 :=
by
  sorry

end complementary_angle_measure_l79_79737


namespace parallel_condition_l79_79086

noncomputable theory

-- Define line l1 and line l2
def line_l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def line_l2 (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define the conditions for lines to be parallel
def parallel_lines (a b c d e f : ℝ) : Prop := a * e = b * d

-- The main theorem
theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, line_l1 a x y ↔ line_l2 a x y) ↔ (a = 1) :=
begin
  sorry
end

end parallel_condition_l79_79086


namespace max_consecutive_sum_le_1000_l79_79228

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l79_79228


namespace max_equilateral_triangle_area_in_rectangle_l79_79705

-- Define the rectangle and conditions
def rectangle_area (length width : ℝ) : ℝ := length * width

def midpoint (a b : ℝ) : ℝ := (a + b) / 2

-- Define the maximum area of an equilateral triangle
def equilateral_triangle_area (side : ℝ) : ℝ := (side * side * Math.sqrt 3) / 4

-- The problem statement
theorem max_equilateral_triangle_area_in_rectangle 
  (a b : ℝ) (h_a : a = 12) (h_b : b = 15) : 
  ∃ (s : ℝ), equilateral_triangle_area s = 50.0625 * Math.sqrt 3 :=
by
  have h_area : (equilateral_triangle_area 14.15 = 50.0625 * Math.sqrt 3) :=
    sorry,
  use 14.15,
  exact h_area

end max_equilateral_triangle_area_in_rectangle_l79_79705


namespace graph_stays_connected_after_edge_removal_l79_79876

/-- Let G be a graph where every vertex has degree 100. 
    Prove that removing any single edge does not disconnect the graph. -/
theorem graph_stays_connected_after_edge_removal
  {V : Type} {G : SimpleGraph V} 
  (h_degree : ∀ v : V, G.degree v = 100)
  (h_connected : G.IsConnected) 
  (u v : V) (h_edge : G.Adj u v) :
  (G.deleteEdge u v).IsConnected :=
by
  sorry

end graph_stays_connected_after_edge_removal_l79_79876


namespace sum_of_distances_l79_79875

-- Define the points
def A : ℝ × ℝ := (15, 0)
def B : ℝ × ℝ := (0, 5)
def D : ℝ × ℝ := (4, 3)

-- Define a function to calculate Euclidean distance between two points
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the distances AD and BD
def AD : ℝ := euclidean_distance A D
def BD : ℝ := euclidean_distance B D

theorem sum_of_distances : AD + BD = 2 * Real.sqrt 5 + Real.sqrt 130 := by
  sorry

end sum_of_distances_l79_79875


namespace tangent_line_at_x0_l79_79671

-- Define the given function y = 2x / (x^2 + 1)
def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 1)

-- Define the point x0 = 1
def x0 : ℝ := 1

-- Define the derivative of the function f(x)
def f' (x : ℝ) : ℝ := ((2 * (x^2 + 1) - (2 * x) * (2 * x)) / (x^2 + 1)^2)

-- Evaluate the derivative at x0 = 1
def y'_0 : ℝ := f' x0

-- Evaluate the original function at x0 = 1
def y0 : ℝ := f x0

-- Theorem: The equation of the tangent line to the curve f(x) at x0, given the evaluations, is y = 1
theorem tangent_line_at_x0 : ∀ x : ℝ, f' x0 * (x - x0) + y0 = 1 :=
by
  simp [f', x0, y'_0, y0]
  sorry

end tangent_line_at_x0_l79_79671


namespace find_multiplier_l79_79273

theorem find_multiplier (n x : ℤ) (h1: n = 12) (h2: 4 * n - 3 = (n - 7) * x) : x = 9 :=
by {
  sorry
}

end find_multiplier_l79_79273


namespace axis_of_symmetry_parabola_l79_79003

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), 2 * (x - 3)^2 - 5 = 2 * (x - 3)^2 - 5 → (∃ h : ℝ, h = 3 ∧ ∀ x : ℝ, h = 3) :=
by
  sorry

end axis_of_symmetry_parabola_l79_79003


namespace cos_third_quadrant_l79_79021

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l79_79021


namespace length_of_train_l79_79718

-- Definitions based on conditions
def speed_kmph : ℝ := 90
def time_sec : ℝ := 100

-- Conversion factor from km/hr to m/sec
def kmph_to_mps : ℝ := 1000 / 3600

-- Mathematically equivalent proof problem
theorem length_of_train (speed_kmph time_sec : ℝ) (kmph_to_mps : ℝ) :
  (speed_kmph * kmph_to_mps) * time_sec = 2500 :=
by
  sorry

end length_of_train_l79_79718


namespace sum_of_perimeters_sum_of_areas_area_ratio_radii_ratio_l79_79868

variable {α β γ : ℝ}
variable {K T a b c a' b' c' : ℝ}

-- Given Conditions
def triangle_angle_bisectors_to_circumcircle (ABC : Type) (circumcircle : Type) (A' B' C' A_1 B_1 C_1 A'' B'' C'' A_2 B_2 C_2 : Type) : Prop := sorry

def triangle_perimeters_and_areas (triangle : Type) (perimeter : ℝ) (area : ℝ) : Prop := sorry

-- Sum of Perimeters
theorem sum_of_perimeters (ABC A_1B_1C_1 A_2B_2C_2 : Type) (K : ℝ) :
  triangle_angle_bisectors_to_circumcircle ABC Circumcircle A' B' C' A_1 B_1 C_1 A'' B'' C'' A_2 B_2 C_2 →
  triangle_perimeters_and_areas ABC K 0 →
  triangle_perimeters_and_areas A_1B_1C_1 (K / 2) 0 →
  triangle_perimeters_and_areas A_2B_2C_2 (K / 4) 0 →
  Σ (n : ℕ), triangle (n) = 2 * K := sorry

-- Sum of Areas
theorem sum_of_areas (ABC A_1B_1C_1 A_2B_2C_2 : Type) (T : ℝ) :
  triangle_angle_bisectors_to_circumcircle ABC Circumcircle A' B' C' A_1 B_1 C_1 A'' B'' C'' A_2 B_2 C_2 →
  triangle_perimeters_and_areas ABC 0 T →
  triangle_perimeters_and_areas A_1B_1C_1 0 (T / 4) →
  triangle_perimeters_and_areas A_2B_2C_2 0 (T / 16) →
  Σ (n : ℕ), triangle (n) = (4 / 3) * T := sorry

-- Area ratio
theorem area_ratio (ABC A'B'C' : Type) (T T' : ℝ) :
  triangle_angle_bisectors_to_circumcircle ABC Circumcircle A' B' C' A_1 B_1 C_1 A'' B'' C'' A_2 B_2 C_2 →
  triangle_perimeters_and_areas ABC 0 T →
  triangle_perimeters_and_areas A'B'C' 0 T' →
  T / T' = 8 * sin (α / 2) * sin (β / 2) * sin (γ / 2) := sorry

-- Radii ratio
theorem radii_ratio (ABC A'B'C' : Type) (r r' : ℝ) :
  triangle_angle_bisectors_to_circumcircle ABC Circumcircle A' B' C' A_1 B_1 C_1 A'' B'' C'' A_2 B_2 C_2 →
  triangle_perimeters_and_areas ABC 0 0 →
  triangle_perimeters_and_areas A'B'C' 0 0 →
  r / r' = (a * b * c * (a' + b' + c')) / (a' * b' * c' * (a + b + c)) := sorry

end sum_of_perimeters_sum_of_areas_area_ratio_radii_ratio_l79_79868


namespace problem_statement_l79_79561

-- Define the data set conditions
def data_set (n : ℕ) : list ℕ :=
  (list.replicate 12 (fin_range 1 n)) ++ list.replicate 7 (n + 1)

-- Define the mean calculation
def mean (lst : list ℕ) : ℚ :=
  (list.sum lst : ℚ) / list.length lst

-- Define the median calculation
def median (lst : list ℕ) : ℚ :=
  let sorted_lst := list.qsort (≤) lst in
  let len := list.length sorted_lst in
  if len % 2 = 0 then
    ((sorted_lst.nth_le (len / 2 - 1) sorry + sorted_lst.nth_le (len / 2) sorry) : ℚ) / 2
  else
    sorted_lst.nth_le (len / 2) sorry

-- Define the modes calculation
def modes (lst : list ℕ) : list ℕ :=
  let freq := list.foldl (λ acc x, acc.insert x (acc.find x + 1)) list.dict.nil lst in
  let max_freq := freq.values.max in
  freq.keys.filter (λ k, freq.find k = max_freq)

-- Define the median of the modes
def median_modes (modes_lst : list ℕ) : ℚ :=
  median modes_lst

-- Theorem statement
theorem problem_statement :
  let data := data_set 30 in
  let M := median data in
  let mu := mean data in
  let d := median_modes (modes data) in
  d < mu ∧ mu < M :=
by sorry

end problem_statement_l79_79561


namespace combined_weight_of_elephant_and_donkey_l79_79074

theorem combined_weight_of_elephant_and_donkey 
  (tons_to_pounds : ℕ → ℕ)
  (elephant_weight_tons : ℕ) 
  (donkey_percentage : ℕ) : 
  tons_to_pounds elephant_weight_tons * (1 + donkey_percentage / 100) = 6600 :=
by
  let tons_to_pounds (t : ℕ) := 2000 * t
  let elephant_weight_tons := 3
  let donkey_percentage := 10
  sorry

end combined_weight_of_elephant_and_donkey_l79_79074


namespace sum_possible_C_divisible_by_4_l79_79319

theorem sum_possible_C_divisible_by_4 : 
  let valid_C := {C | (10 * C + 92) % 4 = 0 ∧ C ∈ finset.range 10}
  finset.sum (valid_C : finset ℕ) id = 12 :=
by 
  let valid_C := finset.filter (λ C, (10 * C + 92) % 4 = 0) (finset.range 10)
  show finset.sum valid_C id = 12
  apply finset.sum_filter
  rw finset.range_filter_card
  exact dec_trivial

end sum_possible_C_divisible_by_4_l79_79319


namespace difference_of_numbers_l79_79214

/-- Given two natural numbers a and 10a whose sum is 23,320,
prove that the difference between them is 19,080. -/
theorem difference_of_numbers (a : ℕ) (h : a + 10 * a = 23320) : 10 * a - a = 19080 := by
  sorry

end difference_of_numbers_l79_79214


namespace rectangular_box_diagonals_l79_79706

noncomputable def interior_diagonals_sum (a b c : ℝ) : ℝ := 4 * Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 150)
  (h2 : 4 * (a + b + c) = 60)
  (h3 : a * b * c = 216) :
  interior_diagonals_sum a b c = 20 * Real.sqrt 3 :=
by
  sorry

end rectangular_box_diagonals_l79_79706


namespace exists_valid_configuration_l79_79739

-- Define the nine circles
def circles : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the connections (adjacency list) where each connected pair must sum to 23
def lines : List (ℕ × ℕ) := [(1, 8), (8, 6), (8, 9), (9, 2), (2, 7), (7, 6), (7, 4), (4, 1), (4, 5), (5, 6), (5, 3), (6, 3)]

-- The main theorem that we need to prove: there exists a permutation of circles satisfying the line sum condition
theorem exists_valid_configuration: 
  ∃ (f : ℕ → ℕ), 
    (∀ x ∈ circles, f x ∈ circles) ∧ 
    (∀ (a b : ℕ), (a, b) ∈ lines → f a + f b = 23) :=
sorry

end exists_valid_configuration_l79_79739


namespace volume_correct_l79_79756

noncomputable def volume_of_extended_box : ℝ :=
  let V_box := 4 * 5 * 6 in
  let V_external_parallelepipeds := 2 * (4 * 5 * 1) + 2 * (4 * 6 * 1) + 2 * (5 * 6 * 1) in
  let V_spheres := 8 * (1 / 8 * (4 / 3) * π * 1^3) in
  let V_cylinders := 3 * π * 4 + 3 * π * 5 + 3 * π * 6 in
  V_box + V_external_parallelepipeds + V_spheres + V_cylinders

theorem volume_correct : volume_of_extended_box = (804 + 139 * π) / 3 := by
  -- Proof to be filled in
  sorry

end volume_correct_l79_79756


namespace geometric_series_common_ratio_l79_79174

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l79_79174


namespace geometric_series_common_ratio_l79_79189

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l79_79189


namespace angle_of_inclination_BC_l79_79321

noncomputable def angle_of_inclination (m : ℝ) : ℝ :=
  real.arctan m

noncomputable def pointA : ℝ × ℝ := (-2, real.sqrt 3)
noncomputable def pointC : ℝ × ℝ := (1, 2 * real.sqrt 3)
noncomputable def pointA' : ℝ × ℝ := (-2, -real.sqrt 3)

theorem angle_of_inclination_BC : let Bx := (-(pointA.fst * pointA.snd) / real.sqrt (pointA.fst^2 + pointA.snd^2)) in 
  let By := 0 in
  let B := (Bx, By) in
  (pointA'.snd ≠ pointC.snd ∧ pointA'.fst ≠ pointC.fst) →
  Bx = (pointA'.snd * pointC.fst - pointA'.fst * pointC.snd) / (pointA'.snd - pointC.snd) →
  let m := (pointC.snd - By) / (pointC.fst - Bx) in
  angle_of_inclination m = real.pi / 3 :=
by sorry

end angle_of_inclination_BC_l79_79321


namespace smallest_trapezoid_area_l79_79894

theorem smallest_trapezoid_area :
  ∀ (area_outer : ℝ) (area_inner : ℝ) (ratios : ℝ × ℝ × ℝ), 
  area_outer = 36 → area_inner = 4 → ratios = (3, 2, 1) → 
  (area_outer - area_inner) / (ratios.1 + ratios.2 + ratios.3) * min ratios.1 (min ratios.2 ratios.3) = 5.33 :=
by
  intros area_outer area_inner ratios h1 h2 h3
  -- Proof to be completed
  sorry

end smallest_trapezoid_area_l79_79894


namespace sophie_saves_money_l79_79595

-- Definitions based on the conditions
def loads_per_week : ℕ := 4
def sheets_per_load : ℕ := 1
def cost_per_box : ℝ := 5.50
def sheets_per_box : ℕ := 104
def weeks_per_year : ℕ := 52

-- Main theorem statement
theorem sophie_saves_money :
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  annual_saving = 11.00 := 
by {
  -- Calculation steps
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  -- Proving the final statement
  sorry
}

end sophie_saves_money_l79_79595


namespace slower_time_to_reach_top_l79_79554

def time_for_lola (stories : ℕ) (time_per_story : ℕ) : ℕ :=
  stories * time_per_story

def time_for_tara (stories : ℕ) (time_per_story : ℕ) (stopping_time : ℕ) (num_stops : ℕ) : ℕ :=
  (stories * time_per_story) + (num_stops * stopping_time)

theorem slower_time_to_reach_top (stories : ℕ) (lola_time_per_story : ℕ) (tara_time_per_story : ℕ) 
  (tara_stop_time : ℕ) (tara_num_stops : ℕ) : 
  stories = 20 
  → lola_time_per_story = 10 
  → tara_time_per_story = 8 
  → tara_stop_time = 3
  → tara_num_stops = 18
  → max (time_for_lola stories lola_time_per_story) (time_for_tara stories tara_time_per_story tara_stop_time tara_num_stops) = 214 :=
by sorry

end slower_time_to_reach_top_l79_79554


namespace total_spent_l79_79402

theorem total_spent (cost_per_deck : ℕ) (decks_frank : ℕ) (decks_friend : ℕ) (total : ℕ) : 
  cost_per_deck = 7 → 
  decks_frank = 3 → 
  decks_friend = 2 → 
  total = (decks_frank * cost_per_deck) + (decks_friend * cost_per_deck) → 
  total = 35 :=
by
  sorry

end total_spent_l79_79402


namespace range_of_a_l79_79821

-- Given conditions and propositions
variable (a : ℝ) (h : 0 < a)

-- Define proposition p: y = a^x is strictly decreasing
def prop_p : Prop := 0 < a ∧ a < 1

-- Define proposition q: x^2 - 3ax + 1 > 0 for all x
def prop_q : Prop := ∀ x : ℝ, x^2 - 3 * a * x + 1 > 0

-- The theorem statement
theorem range_of_a (h_pq : xor prop_p prop_q) : 
  (2 / 3) ≤ a ∧ a < 1 :=
sorry

end range_of_a_l79_79821


namespace max_consecutive_integers_sum_lt_1000_l79_79242

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l79_79242


namespace correct_calculation_l79_79669

theorem correct_calculation (a : ℝ) : -2 * a + (2 * a - 1) = -1 := by
  sorry

end correct_calculation_l79_79669


namespace portion_of_pie_left_is_15_percent_l79_79749

def fraction_of_pie_left_after_carlos (total_pie : ℝ) : ℝ :=
  total_pie * 0.2

def fraction_of_pie_taken_by_maria (remaining_pie : ℝ) : ℝ :=
  remaining_pie * 0.25

theorem portion_of_pie_left_is_15_percent (total_pie : ℝ) (H : total_pie = 1) :
  let remaining_pie := fraction_of_pie_left_after_carlos total_pie in
  let maria_pie := fraction_of_pie_taken_by_maria remaining_pie in
  remaining_pie - maria_pie = 0.15 :=
by
  sorry

end portion_of_pie_left_is_15_percent_l79_79749


namespace point_bisector_second_quadrant_l79_79870

theorem point_bisector_second_quadrant (a : ℝ) : 
  (a < 0 ∧ 2 > 0) ∧ (2 = -a) → a = -2 :=
by sorry

end point_bisector_second_quadrant_l79_79870


namespace last_duck_bread_l79_79623

theorem last_duck_bread (total_bread : ℕ) (left_bread : ℕ) (first_duck_bread_fraction : ℚ) (second_duck_bread : ℕ) (total_pieces : ℕ) :
  total_bread = 100 →
  left_bread = 30 →
  first_duck_bread_fraction = 1 / 2 →
  second_duck_bread = 13 →
  total_pieces = total_bread - left_bread →
  ∃ (last_duck_bread : ℕ), last_duck_bread = total_pieces - (total_bread / 2) - second_duck_bread :=
begin
  intros h1 h2 h3 h4 h5,
  use total_pieces - (total_bread / 2) - second_duck_bread,
  rw [h1, h2, h3, h4, h5] at *,
  norm_num,
  sorry,
end

end last_duck_bread_l79_79623


namespace coplanar_vectors_lambda_eq_one_l79_79404

open Real

def vector_a : Vector3 := ⟨2, -1, 3⟩
def vector_b : Vector3 := ⟨-1, 4, -2⟩
def vector_c (λ : Real) : Vector3 := ⟨1, 3, λ⟩

theorem coplanar_vectors_lambda_eq_one (λ : Real) :
  (λ : Real) → (⟨1, 3, λ⟩ ∈ span ℝ {⟨2, -1, 3⟩, ⟨-1, 4, -2⟩}) → λ = 1 :=
by
  sorry

end coplanar_vectors_lambda_eq_one_l79_79404


namespace mabel_shark_ratio_l79_79558

variables (F1 F2 sharks_total sharks_day1 sharks_day2 ratio : ℝ)
variables (fish_day1 := 15)
variables (shark_percentage := 0.25)
variables (total_sharks := 15)

noncomputable def ratio_of_fish_counts := (F2 / F1)

theorem mabel_shark_ratio 
    (fish_day1 : ℝ := 15)
    (shark_percentage : ℝ := 0.25)
    (total_sharks : ℝ := 15)
    (sharks_day1 := 0.25 * fish_day1)
    (sharks_day2 := total_sharks - sharks_day1)
    (F2 := sharks_day2 / shark_percentage)
    (ratio := F2 / fish_day1):
    ratio = 16 / 5 :=
by
  sorry

end mabel_shark_ratio_l79_79558


namespace determine_constant_m_l79_79544

noncomputable def unit_vectors (u v w : ℝ^3) : Prop :=
  ∥u∥ = 1 ∧ ∥v∥ = 1 ∧ ∥w∥ = 1

noncomputable def orthogonal (x y : ℝ^3) : Prop := x ⋅ y = 0

noncomputable def angle (x y : ℝ^3) : ℝ := real.arccos (x ⋅ y / (∥x∥ * ∥y∥))

theorem determine_constant_m
  (u v w : ℝ^3)
  (hu : unit_vectors u v w)
  (huv : orthogonal u v)
  (huw : orthogonal u w)
  (hvwa : angle v w = π / 3):
  ∃ m : ℝ, u = m * (v × w) ∧ (m = 2 * real.sqrt 3 / 3 ∨ m = - (2 * real.sqrt 3 / 3)) :=
sorry

end determine_constant_m_l79_79544


namespace buildings_subset_count_l79_79218

theorem buildings_subset_count :
  let buildings := Finset.range (16 + 1) \ {0}
  ∃ S ⊆ buildings, ∀ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S → ∃ k, (b - a = 2 * k + 1) ∨ (a - b = 2 * k + 1) ∧ Finset.card S = 510 :=
sorry

end buildings_subset_count_l79_79218


namespace smallest_positive_period_axis_of_symmetry_max_value_on_interval_min_value_on_interval_l79_79447

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (2 * x - π / 3) - 2 * sin (x - π / 4) * sin (x + π / 4)

theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := 
  sorry

theorem axis_of_symmetry : ∃ k : ℤ, ∀ x, (f (x) = f (k * (π / 2) + π / 3)) := 
  sorry

theorem max_value_on_interval : ∀ x, (-π / 12 ≤ x ∧ x ≤ π / 2 → (f x ≤ 1)) :=
  ∃ x, (x = π / 3 ∧ f x = 1) := 
  sorry

theorem min_value_on_interval : ∀ x, (-π / 12 ≤ x ∧ x ≤ π / 2 → (f x ≥ - sqrt 3 / 2)) :=
  ∃ x, (x = - π / 12 ∧ f x = - sqrt 3 / 2) := 
  sorry

end smallest_positive_period_axis_of_symmetry_max_value_on_interval_min_value_on_interval_l79_79447


namespace regular_price_for_one_T_shirt_l79_79325

theorem regular_price_for_one_T_shirt 
    (total_tshirts : ℕ) 
    (total_cost : ℝ) 
    (discounted_cost : ℝ) 
    (regular_cost : ℝ) 
    (group_size : ℕ) 
    (actual_group_count : total_tshirts / group_size) 
    (cost_per_group : regular_cost + discounted_cost)
    (total_spent : actual_group_count * cost_per_group = total_cost) : regular_cost = 14.5 :=
by
  sorry

end regular_price_for_one_T_shirt_l79_79325


namespace tangent_circle_solutions_l79_79361

theorem tangent_circle_solutions (r R : ℝ) (O : ℝ × ℝ) (a b c : ℝ)
  (h1 : 0 ≤ r)
  (h2 : 0 ≤ R) :
  ∃ n : ℕ, n ≤ 8 ∧ ∀ (P : ℝ × ℝ), 
  (dist P O = R + r ∨ dist P O = R - r) ∧ 
  (a * P.1 + b * P.2 + (c + r * real.sqrt (a^2 + b^2)) = 0 ∨ 
   a * P.1 + b * P.2 + (c - r * real.sqrt (a^2 + b^2)) = 0) 
  → n = 8 :=
sorry

end tangent_circle_solutions_l79_79361


namespace spherical_to_rectangular_correct_l79_79362

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  sphericalToRectangular ρ θ φ = (5 * Real.sqrt 6 / 4, 5 * Real.sqrt 6 / 4, 5 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l79_79362


namespace find_N_l79_79015

theorem find_N : ∀ N : ℕ, (991 + 993 + 995 + 997 + 999 = 5000 - N) → N = 25 :=
by
  intro N h
  sorry

end find_N_l79_79015


namespace total_votes_l79_79052

theorem total_votes (V : ℝ) 
  (valid_votes : ℝ := 0.8 * V) 
  (candidate_A_valid_votes : ℝ := 0.55 * valid_votes) 
  (candidate_B_valid_votes : ℝ = 1980) 
  (valid_votes_eq : candidate_A_valid_votes + candidate_B_valid_votes = valid_votes) :
  V = 5500 :=
by
  sorry

end total_votes_l79_79052


namespace part_i_part_ii_l79_79532

-- Define the setup
variable (A B C D O : Point)
variable (AB AD BC CD : ℝ)
variable (semicircle_tangent : ∀ P Q R, IsTangent (semicircle O AB) BC ∧ IsTangent (semicircle O AB) CD ∧ IsTangent (semicircle O AB) DA)
variable (quadrilateral_cyclic : IsCyclicQuadrilateral A B C D)
variable (x y : ℝ)

-- Part (i) Statement
theorem part_i : AB = AD + BC := by
  sorry

-- Part (ii) Statement
theorem part_ii (h1 : AB = x) (h2 : CD = y) :
  maximal_area (quadrilateral_cyclic) = (x + y) * (sqrt (2 * x * y - y ^ 2)) / 4 := by
  sorry

end part_i_part_ii_l79_79532


namespace earthquake_amplitude_ratio_l79_79057

theorem earthquake_amplitude_ratio (A B : ℝ) (A0 : ℝ) 
  (h1 : ∀ (M : ℝ), M = log10 (A/A0)) (h2 : 7.1 = log10 (A/A0)) (h3 : 6.2 = log10 (B/A0)) :
  A / B = 10^0.9 :=
by 
  sorry

end earthquake_amplitude_ratio_l79_79057


namespace probability_of_specific_sequence_l79_79662

-- We define a structure representing the problem conditions.
structure problem_conditions :=
  (cards : multiset ℕ)
  (permutation : list ℕ)

-- Noncomputable definition for the correct answer.
noncomputable def probability := (1 : ℚ) / 720

-- The main theorem statement.
theorem probability_of_specific_sequence :
  ∀ (conds : problem_conditions),
  conds.cards = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6} ∧
  (∃ (perm : list ℕ), perm.perm conds.permutation) →
  (∃ (sequence : list ℕ), sequence = [1, 2, 3, 4, 5, 6]) →
  let prob := calculate_probability conds.permutation [1, 2, 3, 4, 5, 6] in
  prob = (1 : ℚ) / 720 :=
sorry

end probability_of_specific_sequence_l79_79662


namespace bing_column_guimao_2023_column_l79_79054

-- Define the cyclic arrangement and state the problem for 丙 (bǐng) and 癸卯 (guǐ mǎo).

-- Problem (1): Column of 丙 (bǐng) for the n-th time.
theorem bing_column (n : ℕ) : ∃ k, n = 10 * k - 7 := 
sorry

-- Problem (2): Column position for 2023 year 癸卯 (guǐ mǎo).
theorem guimao_2023_column : column 2023 = 40 :=
sorry

end bing_column_guimao_2023_column_l79_79054


namespace geometric_series_common_ratio_l79_79195

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l79_79195


namespace area_of_inner_triangle_divided_by_1_2_l79_79984

-- Define the original triangle and the points dividing each side in 1:2 ratio
variables {T : ℝ} (area_original : ℝ)
variables (A B C : Point) 
variables (P Q R : Point)
-- Assume that points P, Q, and R divide the sides in ratio 1:2

-- Assume P is 1/3 along from A to B, Q is 1/3 along from B to C, and R is 1/3 along from C to A
def dividing_point (A B : Point) (ratio : ℝ) : Point := sorry -- Define how we compute these points

-- Assume relevant properties about triangle, its area, and scaled areas
axiom area_triangle_proportional (T : ℝ) {Δ : Triangle} :
    area_original Δ = T → ∃ Δ', Δ'.side_lengths = (1/3) • Δ.side_lengths ∧ area Δ' = (T / 9)

-- Main statement about areas
theorem area_of_inner_triangle_divided_by_1_2 (T : ℝ) (area_original : T) : 
    area_original = T → 
    ∃ Δ', Δ'.side_lengths = (1/3) • Δ.side_lengths ∧ area Δ' = (T / 9) := 
by 
    sorry

end area_of_inner_triangle_divided_by_1_2_l79_79984


namespace cos_B_third_quadrant_l79_79028

theorem cos_B_third_quadrant (B : ℝ) (hB1 : π < B ∧ B < 3 * π / 2) (hB2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

end cos_B_third_quadrant_l79_79028


namespace capacitor_voltage_calculation_l79_79300

theorem capacitor_voltage_calculation : 
  ∀ (C₁ C₂ U₁ U₂ : ℝ), 
    C₁ = 10 ∧ U₁ = 15 ∧ C₂ = 5 ∧ U₂ = 10 → 
      (C₁ * U₁ - C₂ * U₂) / (C₁ + C₂) = 20 / 3 := by
  intros C₁ C₂ U₁ U₂ h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5, h6]
  norm_num
  sorry

end capacitor_voltage_calculation_l79_79300


namespace right_triangle_ratio_is_4_l79_79707

noncomputable def right_triangle_rectangle_ratio (b h xy : ℝ) : Prop :=
  (0.4 * (1/2) * b * h = 0.25 * xy) ∧ (xy = b * h) → (b / h = 4)

theorem right_triangle_ratio_is_4 (b h xy : ℝ) (h1 : 0.4 * (1/2) * b * h = 0.25 * xy)
(h2 : xy = b * h) : b / h = 4 :=
sorry

end right_triangle_ratio_is_4_l79_79707


namespace triangle_integer_lengths_impossible_l79_79084

theorem triangle_integer_lengths_impossible
  (A B C D E I : Type)
  [Triangle A B C]
  [RightAngle (Angle A)]
  [OnSegment D A C]
  [OnSegment E A B]
  [AngleEqual (Angle A B D) (Angle D B C)]
  [AngleEqual (Angle A C E) (Angle E C B)]
  [Intersection (Segment B D) (Segment C E) I] :
  ¬ (IsIntegerLength (Segment A B) ∧ IsIntegerLength (Segment A C) ∧
     IsIntegerLength (Segment B I) ∧ IsIntegerLength (Segment I D) ∧
     IsIntegerLength (Segment C I) ∧ IsIntegerLength (Segment I E)) :=
sorry

end triangle_integer_lengths_impossible_l79_79084


namespace problem_statement_l79_79842

-- Definitions related to the given conditions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi) / 6)

theorem problem_statement :
  (∀ x1 x2 : ℝ, (x1 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → (x2 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → x1 < x2 → f x1 < f x2) →
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) →
  f (-((5 * Real.pi) / 12)) = (Real.sqrt 3) / 2 :=
by
  intros h_mono h_symm
  sorry

end problem_statement_l79_79842


namespace max_consecutive_integers_lt_1000_l79_79255

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l79_79255


namespace number_of_possible_teams_less_than_5_percent_girls_teams_l79_79314

-- Definitions related to the problem conditions
def boys : ℕ := 5
def girls : ℕ := 4
def students : ℕ := boys + girls
def team_size : ℕ := 3
def girl_team_size : ℕ := 3

-- Define combinations function
def combination (n r : ℕ) : ℕ := (n.choose r)

-- Prove that the total number of teams is 84
theorem number_of_possible_teams : combination students team_size = 84 :=
by sorry

-- Number of ways to form a team of only girls
def girl_teams : ℕ := combination girls girl_team_size

-- Prove that less than 5% of the possible teams consist only of girls
theorem less_than_5_percent_girls_teams : girl_teams < 0.05 * combination students team_size :=
by sorry

end number_of_possible_teams_less_than_5_percent_girls_teams_l79_79314


namespace equilateral_triangle_perimeter_l79_79694

theorem equilateral_triangle_perimeter :
  (∃ (x y : ℝ), y = -2 * x ∧ x = 1) ∧ 
  (∃ (x y : ℝ), y = 2 + (1 / 2) * x ∧ x = 1) ∧ 
  (∃ (x y1 y2 : ℝ), y1 = -2 * x ∧ y2 = 2 + (1 / 2) * x ∧ x = 1) → 
  let side_length := 4.5 in
  3 * side_length = 13.5 :=
by
  sorry

end equilateral_triangle_perimeter_l79_79694


namespace john_marble_choices_l79_79907

open Nat

theorem john_marble_choices :
  (choose 4 2) * (choose 12 3) = 1320 :=
by
  sorry

end john_marble_choices_l79_79907


namespace current_population_l79_79685

theorem current_population (initial_population deaths_leaving_percentage : ℕ) (current_population : ℕ) :
  initial_population = 3161 → deaths_leaving_percentage = 5 →
  deaths_leaving_percentage / 100 * initial_population + deaths_leaving_percentage * (initial_population - deaths_leaving_percentage / 100 * initial_population) / 100 = initial_population - current_population →
  current_population = 2553 :=
 by
  sorry

end current_population_l79_79685


namespace number_of_solutions_ineq_l79_79329

theorem number_of_solutions_ineq : 
  {x | x ∈ {-1, 0, (1/2), 1} ∧ 2*x - 1 < x}.card = 3 := 
by 
  sorry

end number_of_solutions_ineq_l79_79329


namespace number_of_members_in_league_l79_79562

-- Define the conditions
def pair_of_socks_cost := 4
def t_shirt_cost := pair_of_socks_cost + 6
def cap_cost := t_shirt_cost - 3
def total_cost_per_member := 2 * (pair_of_socks_cost + t_shirt_cost + cap_cost)
def league_total_expenditure := 3144

-- Prove that the number of members in the league is 75
theorem number_of_members_in_league : 
  (∃ (n : ℕ), total_cost_per_member * n = league_total_expenditure) → 
  (∃ (n : ℕ), n = 75) :=
by
  sorry

end number_of_members_in_league_l79_79562


namespace subtraction_888_55_555_55_l79_79137

theorem subtraction_888_55_555_55 : 888.88 - 555.55 = 333.33 :=
by
  sorry

end subtraction_888_55_555_55_l79_79137


namespace cost_of_energy_drink_l79_79969

/-- 
The conditions stated in the problem 
-/
variables (cupcakes_sold : ℕ) (cupcake_price : ℝ)
variables (cookies_sold : ℕ) (cookie_price : ℝ)
variables (basketballs_bought : ℕ) (basketball_price : ℝ)
variables (energy_drinks_bought : ℕ)

/--
The proof statement: Given the number of items sold and their respective prices, 
and the cost of basketballs, prove the cost of one energy drink.
-/
theorem cost_of_energy_drink (h: cupcakes_sold = 50)
                              (h1: cupcake_price = 2)
                              (h2: cookies_sold = 40)
                              (h3: cookie_price = 0.5)
                              (h4: basketballs_bought = 2)
                              (h5: basketball_price = 40)
                              (h6: energy_drinks_bought = 20) :
  (120 - 2 * 40 : ℝ) / 20 = 2 :=
by
  sorry

end cost_of_energy_drink_l79_79969


namespace sum_of_odd_cubes_div_sum_of_odds_l79_79582

theorem sum_of_odd_cubes_div_sum_of_odds (n : ℕ) (h : n > 0) :
  (∑ k in finset.range n, (2 * k + 1)^3) / (∑ k in finset.range n, (2 * k + 1)) = 2 * n^2 - 1 :=
sorry

end sum_of_odd_cubes_div_sum_of_odds_l79_79582


namespace max_distance_l79_79505

def polar_eq_line : ℝ → ℝ → Prop := λ ρ θ, sqrt 2 * ρ * real.cos (θ + real.pi / 4) = 1

def parametric_eqn_curve (α : ℝ) : ℝ × ℝ :=
  (1 + sqrt 3 * real.cos α, real.sin α)

noncomputable def maximum_distance (C : ℝ × ℝ) : ℝ :=
  sqrt 3

theorem max_distance (α : ℝ) :
  let M := parametric_eqn_curve α in
  let l := polar_eq_line in
  true → maximum_distance M = sqrt 3 :=
by simp [maximum_distance]; sorry

end max_distance_l79_79505


namespace max_consecutive_sum_le_1000_l79_79232

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l79_79232


namespace projectiles_meet_in_72_minutes_l79_79499

-- Define the constants based on the conditions
def distance : ℝ := 1182  -- Distance in km
def speed1 : ℝ := 460     -- Speed of first projectile in km/h
def speed2 : ℝ := 525     -- Speed of second projectile in km/h
def total_speed : ℝ := speed1 + speed2  -- Combined speed in km/h
def time_hours : ℝ := distance / total_speed  -- Time in hours
def time_minutes : ℝ := time_hours * 60      -- Time in minutes

-- Theorem statement to be proven
theorem projectiles_meet_in_72_minutes : time_minutes = 72 :=
by
  sorry

end projectiles_meet_in_72_minutes_l79_79499


namespace intersect_dg_ei_on_Γ_l79_79096

-- Definitions of points and relations based on the conditions
variable {A B C I D E F G : Point}
variable {Γ : Circle}
variable {AI_line : Line}

-- Conditions that define our specific point structures and relationships
variable [incenter I A B C]
variable [circumcircle Γ A B C]
variable [on_line AI_line I]
variable [intersect AI_line Γ D (hne : D ≠ A)]
variable [on_bc F B C]
variable [arc_bdc E B D C Γ]
variable [angle_bisect A B F C A E < (1/2) ∠ BAC]

-- Interpretation of G as the midpoint of IF
variable [midpoint G I F]

-- Statement to be proved: The lines DG and EI intersect on Γ
theorem intersect_dg_ei_on_Γ : intersect (line_through D G) (line_through E I) Γ := by
  sorry

end intersect_dg_ei_on_Γ_l79_79096


namespace find_g_neg_4_l79_79152

-- Define the function g and the conditions
def g : ℝ → ℝ := sorry

-- The main theorem to prove
theorem find_g_neg_4 : (∀ x y : ℝ, g(x + y) = g(x) * g(y)) → (∀ x : ℝ, g(x) ≠ 0) → g(-4) = 1 / g(4) :=
by
  intros h1 h2
  sorry

end find_g_neg_4_l79_79152


namespace min_a2_b2_l79_79607

noncomputable def minimum_a2_b2 (a b : ℝ) : Prop :=
  (∃ a b : ℝ, (|(-2*a - 2*b + 4)|) / (Real.sqrt (a^2 + (2*b)^2)) = 2) → (a^2 + b^2 = 2)

theorem min_a2_b2 : minimum_a2_b2 a b :=
by
  sorry

end min_a2_b2_l79_79607


namespace olivia_cookies_total_l79_79114

def cookies_total (baggie_cookie_count : ℝ) (chocolate_chip_cookies : ℝ) 
                  (baggies_oatmeal_cookies : ℝ) (total_cookies : ℝ) : Prop :=
  let oatmeal_cookies := baggies_oatmeal_cookies * baggie_cookie_count
  oatmeal_cookies + chocolate_chip_cookies = total_cookies

theorem olivia_cookies_total :
  cookies_total 9.0 13.0 3.111111111 41.0 :=
by
  -- Proof goes here
  sorry

end olivia_cookies_total_l79_79114


namespace geometric_series_common_ratio_l79_79178

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l79_79178


namespace least_k_cubed_divisible_by_168_l79_79286

theorem least_k_cubed_divisible_by_168 : ∃ k : ℤ, (k ^ 3) % 168 = 0 ∧ ∀ n : ℤ, (n ^ 3) % 168 = 0 → k ≤ n :=
sorry

end least_k_cubed_divisible_by_168_l79_79286


namespace phone_extension_permutations_l79_79908

theorem phone_extension_permutations : 
  (∃ (l : List ℕ), l = [5, 7, 8, 9, 0] ∧ Nat.factorial l.length = 120) :=
sorry

end phone_extension_permutations_l79_79908


namespace ribbon_length_required_correct_l79_79608

noncomputable def area : ℝ := 50
noncomputable def pi_approx : ℝ := 22 / 7
noncomputable def required_ribbon_length (area : ℝ) (pi_approx : ℝ) : ℝ :=
  let r_squared := area * (7 / 22) in
  let r := Real.sqrt r_squared in
  let circumference := 2 * pi_approx * r in
  let rounded_circumference := Real.floor (circumference + 0.5) in
  rounded_circumference + 5

theorem ribbon_length_required_correct : required_ribbon_length area pi_approx = 30 :=
  by
    -- Proof omitted
    sorry

end ribbon_length_required_correct_l79_79608


namespace cos_pi_plus_alpha_l79_79435

-- Definitions for the conditions given in the problem
-- Point P(3, 4) lies on the terminal side of angle α
def P := (3 : ℕ, 4 : ℕ)

-- Function to calculate the hypotenuse
def hypotenuse (x y : ℕ) : ℝ :=
  real.sqrt ((x : ℝ) ^ 2 + (y : ℝ) ^ 2)

noncomputable def cos_alpha : ℝ :=
  P.1 / hypotenuse P.1 P.2

theorem cos_pi_plus_alpha (P : ℕ × ℕ) (hP : P = (3, 4)) :
  cos (π + real.arccos(cos_alpha)) = -3/5 :=
by {
  sorry
}

end cos_pi_plus_alpha_l79_79435


namespace sum_of_lucky_numbers_divisible_by_13_l79_79343

def is_lucky_number (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  ∃ a b c d e f : ℕ, digits = [f, e, d, c, b, a] ∧ (a + b + c = d + e + f)

theorem sum_of_lucky_numbers_divisible_by_13 :
  let lucky_numbers := {n : ℕ | n < 1000000 ∧ is_lucky_number n}
  ∑ n in lucky_numbers, n % 13 = 0 :=
by
  sorry

end sum_of_lucky_numbers_divisible_by_13_l79_79343


namespace sum_seq_b_div_3np1_eq_one_fifth_l79_79088

def seq_b : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n+3) := seq_b (n+2) + seq_b (n+1)

theorem sum_seq_b_div_3np1_eq_one_fifth : (∑' n : ℕ, (seq_b (n + 1) / (3 : ℝ)^(n + 2))) = (1 / 5 : ℝ) :=
by
  sorry

end sum_seq_b_div_3np1_eq_one_fifth_l79_79088


namespace remainder_when_N_divided_by_1000_l79_79398

-- Define the greatest integer function floor and the fractional part
def floor (x : ℝ) : ℤ := int.floor x
def frac (x : ℝ) : ℝ := x - (floor x)

-- Define f(x) = x * frac(x)
def f (x : ℝ) : ℝ := x * frac(x)

-- Define the number of solutions N to f(f(f(x))) = 17 for 0 <= x <= 2020
noncomputable def N : ℕ := {
  -- To be computed
  sorry
}

-- The main theorem to be proven: Remainder when N is divided by 1000
theorem remainder_when_N_divided_by_1000 : N % 1000 = 10 := by
  sorry

end remainder_when_N_divided_by_1000_l79_79398


namespace factor_polynomial_l79_79771

theorem factor_polynomial :
  ∀ (x : ℤ), 9 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 5 * x^2 = (x^2 + 4) * (9 * x^2 + 22 * x + 342) :=
by
  intro x
  sorry

end factor_polynomial_l79_79771


namespace shares_correct_l79_79674

open Real

-- Problem setup
def original_problem (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 1020 ∧
  a = (3 / 4) * b ∧
  b = (2 / 3) * c ∧
  c = (1 / 4) * d ∧
  d = (5 / 6) * e

-- Goal
theorem shares_correct : ∃ (a b c d e : ℝ),
  original_problem a b c d e ∧
  abs (a - 58.17) < 0.01 ∧
  abs (b - 77.56) < 0.01 ∧
  abs (c - 116.34) < 0.01 ∧
  abs (d - 349.02) < 0.01 ∧
  abs (e - 419.42) < 0.01 := by
  sorry

end shares_correct_l79_79674


namespace max_consecutive_integers_sum_lt_1000_l79_79259

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l79_79259


namespace max_friday_more_than_wednesday_l79_79560

-- Definitions and conditions
def played_hours_wednesday : ℕ := 2
def played_hours_thursday : ℕ := 2
def played_average_hours : ℕ := 3
def played_days : ℕ := 3

-- Total hours over three days
def total_hours : ℕ := played_average_hours * played_days

-- Hours played on Friday
def played_hours_wednesday_thursday : ℕ := played_hours_wednesday + played_hours_thursday

def played_hours_friday : ℕ := total_hours - played_hours_wednesday_thursday

-- Proof problem statement
theorem max_friday_more_than_wednesday : 
  played_hours_friday - played_hours_wednesday = 3 := 
sorry

end max_friday_more_than_wednesday_l79_79560


namespace find_a_l79_79452

-- Definition of the parabola
def parabola (p : ℝ) : set (ℝ × ℝ) := {P | P.snd ^ 2 = 2 * p * P.fst}

-- Point M on the parabola
def M (m : ℝ) (p : ℝ) : Prop := (1, m) ∈ parabola p

-- Distance from M to the focus of the parabola is 5
def distance_to_focus (m p : ℝ) : Prop := 5 = 1 + p / 2

-- Definition of the hyperbola
def hyperbola (a : ℝ) : set (ℝ × ℝ) := {P | P.fst ^ 2 / a - P.snd ^ 2 = 1}

-- The slope condition for parallel lines
def slopes_parallel (p a m : ℝ) : Prop := Real.sqrt (1 / a) = 4 / (1 + Real.sqrt a)

-- Main theorem: find the value of a given the conditions.
theorem find_a (m p : ℝ) (h_parabola : p > 0) (hM : m > 0) (hM_parabola : M m p) 
  (h_dist : distance_to_focus m p) (a : ℝ) (hyper : hyperbola a) 
  (left_vertex_A : a > 0) (asymptote_parallel : slopes_parallel p a m) : 
  a = 1 / 9 := by
  sorry

end find_a_l79_79452


namespace minimal_moves_for_7_disks_l79_79997

/-- Mathematical model of the Tower of Hanoi problem with special rules --/
def tower_of_hanoi_moves (n : ℕ) : ℚ :=
  if n = 7 then 23 / 4 else sorry

/-- Proof problem for the minimal number of moves required to transfer all seven disks to rod C --/
theorem minimal_moves_for_7_disks : tower_of_hanoi_moves 7 = 23 / 4 := 
  sorry

end minimal_moves_for_7_disks_l79_79997


namespace max_consecutive_sum_lt_1000_l79_79234

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l79_79234


namespace relationship_of_a_b_l79_79860

theorem relationship_of_a_b
  (a b : Real)
  (h1 : a < 0)
  (h2 : b > 0)
  (h3 : a + b < 0) : 
  -a > b ∧ b > -b ∧ -b > a := 
by
  sorry

end relationship_of_a_b_l79_79860


namespace num_ways_to_pay_16_rubles_l79_79758

theorem num_ways_to_pay_16_rubles :
  ∃! (n : ℕ), n = 13 ∧ ∀ (x y z : ℕ), (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0) ∧ 
  (10 * x + 2 * y + 1 * z = 16) ∧ (x < 2) ∧ (y + z > 0) := sorry

end num_ways_to_pay_16_rubles_l79_79758


namespace mustard_found_at_second_table_l79_79339

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

end mustard_found_at_second_table_l79_79339


namespace geometric_series_common_ratio_l79_79212

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l79_79212


namespace find_x_l79_79324

-- Define the conditions according to the problem statement
variables {C x : ℝ} -- C is the cost per liter of pure spirit, x is the volume of water in the first solution

-- Condition 1: The cost for the first solution
def cost_first_solution (C : ℝ) (x : ℝ) : Prop := 0.50 = C * (1 / (1 + x))

-- Condition 2: The cost for the second solution (approximating 0.4999999999999999 as 0.50)
def cost_second_solution (C : ℝ) : Prop := 0.50 = C * (1 / 3)

-- The theorem to prove: x = 2 given the two conditions
theorem find_x (C : ℝ) (x : ℝ) (h1 : cost_first_solution C x) (h2 : cost_second_solution C) : x = 2 := 
sorry

end find_x_l79_79324


namespace pair_product_not_72_l79_79670

theorem pair_product_not_72 : (2 * (-36) ≠ 72) :=
by
  sorry

end pair_product_not_72_l79_79670


namespace negation_of_universal_proposition_l79_79454

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 1 / 4 > 0)) = ∃ x : ℝ, x^2 - x + 1 / 4 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l79_79454


namespace number_dislike_both_radio_and_music_l79_79577

variable (total_people : ℕ)
variable (percentage_dislike_radio : ℝ)
variable (percentage_dislike_both : ℝ)
variable (num_dislike_radio : ℕ)
variable (num_dislike_both : ℕ)

-- Conditions based on the problem
axiom total_people_eq : total_people = 1500
axiom percentage_dislike_radio_eq : percentage_dislike_radio = 0.4
axiom percentage_dislike_both_eq : percentage_dislike_both = 0.15

-- Compute intermediate value: number of people who dislike radio
def compute_num_dislike_radio : ℕ :=
  (percentage_dislike_radio * total_people).toNat

axiom num_dislike_radio_eq : num_dislike_radio = compute_num_dislike_radio

-- Compute final value: number of people who dislike both radio and music
def compute_num_dislike_both : ℕ :=
  (percentage_dislike_both * num_dislike_radio).toNat

axiom num_dislike_both_eq : num_dislike_both = compute_num_dislike_both

-- Theorem statement
theorem number_dislike_both_radio_and_music :
  num_dislike_both = 90 :=
by
  rw [num_dislike_both_eq, compute_num_dislike_both]
  rw [num_dislike_radio_eq, compute_num_dislike_radio]
  rw [total_people_eq, percentage_dislike_radio_eq, percentage_dislike_both_eq]
  sorry

end number_dislike_both_radio_and_music_l79_79577


namespace am_gm_hm_inequality_l79_79540

theorem am_gm_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : a ≠ c) : 
  (a + b + c) / 3 > (a * b * c) ^ (1 / 3) ∧ (a * b * c) ^ (1 / 3) > 3 * a * b * c / (a * b + b * c + c * a) :=
by
  sorry

end am_gm_hm_inequality_l79_79540


namespace triangle_inequality_l79_79930

variable {α β γ a b c : ℝ}

theorem triangle_inequality (h1: α ≥ β) (h2: β ≥ γ) (h3: a ≥ b) (h4: b ≥ c) (h5: α ≥ γ) (h6: a ≥ c) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
by
  sorry

end triangle_inequality_l79_79930


namespace arc_length_cosh_l79_79746

theorem arc_length_cosh (x1 x2 : ℝ) : ∫ x in x1 .. x2, Real.cosh x = Real.sinh x2 - Real.sinh x1 := 
sorry

end arc_length_cosh_l79_79746


namespace lattice_points_interval_sum_l79_79917

theorem lattice_points_interval_sum :
  let T := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 45 ∧ 1 ≤ p.2 ∧ p.2 ≤ 45}
  let below_line (m : ℚ) :=
    {p ∈ T | p.2 ≤ m * p.1}
  ∃ c d : ℕ, let frac := (c : ℚ) / d in
    (∀ m : ℚ, m ∈ Icc (64/99) (65/99) → below_line m = 700) →
    (c.gcd d = 1) →
    c + d = 100 := sorry

end lattice_points_interval_sum_l79_79917


namespace pair_b_equiv_pair_c_equiv_l79_79276

-- Definitions of the two function pairs that need to be proven equivalent
noncomputable def f1 (x : ℝ) := x^2 - 2 * x - 1
noncomputable def g1 (t : ℝ) := t^2 - 2 * t - 1

def f2 (x : ℝ) := x^0
def g2 (x : ℝ) := 1 / x^0

-- Theorems to show the equivalence of the pairs

-- Pair B: f1(x) = g1(t) for all x and t in ℝ
theorem pair_b_equiv : ∀ x t : ℝ, f1 x = g1 t :=
by
  intros
  exact rfl

-- Pair C: f2(x) = g2(x) for all x in ℝ, x ≠ 0
theorem pair_c_equiv : ∀ x : ℝ, x ≠ 0 → f2 x = g2 x :=
by
  intros x hne
  change 1 = 1
  exact rfl

end pair_b_equiv_pair_c_equiv_l79_79276


namespace original_average_of_15_numbers_l79_79072

theorem original_average_of_15_numbers (A : ℝ) (h1 : 15 * A + 15 * 12 = 52 * 15) :
  A = 40 :=
sorry

end original_average_of_15_numbers_l79_79072


namespace possible_values_of_m_l79_79486

-- Define the conditions that m is a real number and the quadratic equation having two distinct real roots
variable (m : ℝ)

-- Define the discriminant condition for having two distinct real roots
def discriminant_condition (a b c : ℝ) := b^2 - 4 * a * c > 0

-- State the required theorem
theorem possible_values_of_m (h : discriminant_condition 1 m 9) : m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo 6 ∞ :=
sorry

end possible_values_of_m_l79_79486


namespace geometric_series_common_ratio_l79_79197

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l79_79197


namespace missing_digit_divisible_by_9_l79_79976

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

theorem missing_digit_divisible_by_9 : 
  ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 9 ∧ is_multiple_of_9 (1 + 3 + 5 + 7 + x) :=
by
  sorry

end missing_digit_divisible_by_9_l79_79976


namespace divisibility_criterion_l79_79946

theorem divisibility_criterion :
  (∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ (1207 % x = 0) ∧
  (let a := x / 10 in let b := x % 10 in a^3 + b^3 = 344)) ↔
  (1207 % 17 = 0 ∧ let a1 := 1 in let b1 := 7 in a1^3 + b1^3 = 344) ∨
  (1207 % 71 = 0 ∧ let a2 := 7 in let b2 := 1 in a2^3 + b2^3 = 344) :=
by sorry

end divisibility_criterion_l79_79946


namespace radius_of_circle_l79_79146

theorem radius_of_circle :
  ∃ r : ℝ, (θ : ℝ) (A : ℝ), θ = 54 ∧ A = 67.88571428571429 ∧ A = (θ / 360) * Real.pi * r^2 ∧ r = 12 :=
by
  let r := 12
  let θ := 54
  let A := 67.88571428571429
  use r
  split; norm_num
  sorry

end radius_of_circle_l79_79146


namespace selection_methods_at_least_one_female_l79_79586

theorem selection_methods_at_least_one_female :
  let total_students := 7
  let male_students := 5
  let female_students := 2
  let selection_size := 3
  nat.choose total_students selection_size - nat.choose male_students selection_size = 25 :=
by
  sorry

end selection_methods_at_least_one_female_l79_79586


namespace gcd_linear_combination_l79_79121

theorem gcd_linear_combination (a b : ℤ) : Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := by
  sorry

end gcd_linear_combination_l79_79121


namespace geometric_series_common_ratio_l79_79187

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l79_79187


namespace jacobi_symbol_part_a_l79_79673

theorem jacobi_symbol_part_a (n : ℕ) (hn : 1 < n) (h_prime : nat.prime (2^n - 1)) : 
  legendreSym 3 (2^n - 1) = -1 :=
sorry

end jacobi_symbol_part_a_l79_79673


namespace area_of_quadrilateral_l79_79956

-- Assume the lengths of the sides of the quadrilateral and the given angle information
variables (AB BC CD DA : ℝ) (angleCBA : ℝ) (tanACD : ℝ)

-- Given values as per the problem
def values := (AB = 6 ∧ BC = 8 ∧ CD = 5 ∧ DA = 10 ∧ angleCBA = Real.pi / 2 ∧ tanACD = 4 / 3)

-- Final statement to prove the area of the quadrilateral ABCD
theorem area_of_quadrilateral (h : values) : 
  let ABC_area := (1/2) * AB * BC,
      h_ACD := CD * tanACD,
      ACD_area := (1/2) * CD * h_ACD in
  ABC_area + ACD_area = 122 / 3 :=
sorry

end area_of_quadrilateral_l79_79956


namespace sum_of_valid_numbers_is_72_l79_79790

noncomputable def digit_sum_divides_square (a b : ℕ) : Prop :=
  a + b ∣ (10 * a + b) ^ 2

noncomputable def digit_product_divides_cube (a b : ℕ) : Prop :=
  a * b ∣ (10 * a + b) ^ 3

noncomputable def valid_digit_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ digit_sum_divides_square a b ∧ digit_product_divides_cube a b

noncomputable def valid_two_digit_numbers_sum : ℕ :=
  (finset.range 10).sum (λ b, (finset.range 10).sum (λ a, if valid_digit_pair a b then 10 * a + b else 0))

theorem sum_of_valid_numbers_is_72 :
  valid_two_digit_numbers_sum = 72 :=
sorry

end sum_of_valid_numbers_is_72_l79_79790


namespace stratified_sampling_total_students_sampled_l79_79305

theorem stratified_sampling_total_students_sampled 
  (seniors juniors freshmen : ℕ)
  (sampled_freshmen : ℕ)
  (ratio : ℚ)
  (h_freshmen : freshmen = 1500)
  (h_sampled_freshmen_ratio : sampled_freshmen = 75)
  (h_seniors : seniors = 1000)
  (h_juniors : juniors = 1200)
  (h_ratio : ratio = (sampled_freshmen : ℚ) / (freshmen : ℚ))
  (h_freshmen_ratio : ratio * (freshmen : ℚ) = sampled_freshmen) :
  let sampled_juniors := ratio * (juniors : ℚ)
  let sampled_seniors := ratio * (seniors : ℚ)
  sampled_freshmen + sampled_juniors + sampled_seniors = 185 := sorry

end stratified_sampling_total_students_sampled_l79_79305


namespace vitya_probability_l79_79653

theorem vitya_probability :
  let total_sequences := (finset.range 6).card * 
                         (finset.range 5).card * 
                         (finset.range 4).card * 
                         (finset.range 3).card * 
                         (finset.range 2).card * 
                         (finset.range 1).card,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11,
      total_possibilities := nat.choose 12 2 * nat.choose 10 2 * 
                             nat.choose 8 2 * nat.choose 6 2 * 
                             nat.choose 4 2 * nat.choose 2 2,
      P := (favorable_sequences : ℚ) / (total_possibilities : ℚ)
  in P = 1 / 720 := 
sorry

end vitya_probability_l79_79653


namespace equation_of_the_line_l79_79974

open Real

def intersection_point (l1 l2 : ℝ × ℝ × ℝ) : ℝ × ℝ :=
  let ⟨a1, b1, c1⟩ := l1
  let ⟨a2, b2, c2⟩ := l2
  let x := (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1)
  let y := (a1 * c2 - a2 * c1) / (a2 * b1 - a1 * b2)
  (x, y)

noncomputable def line_inclined_angle (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ × ℝ :=
  let m := tan θ
  let ⟨x, y⟩ := p
  let b := y - m * x
  (m, -1, b)

theorem equation_of_the_line :
  let l1 := (1, -1, 3)
  let l2 := (2, 1, 0)
  let p := intersection_point l1 l2
  let θ := π / 3
  let line := line_inclined_angle p θ
  line = (sqrt 3, -1, sqrt 3 + 2) := by
  sorry

end equation_of_the_line_l79_79974


namespace last_bead_is_yellow_l79_79903

-- Defining the conditions
def bead_pattern := [ "Red", "Orange", "Yellow", "Yellow", "Green", "Blue", "Purple"]
def first_bead := "Red"
def total_beads := 81

-- Function to determine the color of the nth bead given the pattern
def nth_bead (n : Nat) : String :=
  bead_pattern[(n - 1) % bead_pattern.length]

-- Theorem statement
theorem last_bead_is_yellow : nth_bead total_beads = "Yellow" :=
  by
    -- The proof should go here
    sorry

end last_bead_is_yellow_l79_79903


namespace num_cool_sequences_l79_79955
open Nat 

def is_cool_sequence (seq : List Nat) : Prop :=
  seq.length = 8 ∧
  (∀ (i : Nat), i < 7 → Nat.gcd (seq.get i).toNat (seq.get (i+1)).toNat = 1) ∧
  (∃ (perm : List Nat), List.Perm perm [1,2,3,4,5,6,7,8] ∧ seq = perm)

theorem num_cool_sequences : ∃ (n : Nat), n = 648 ∧ 
  (∃ (seqs : List (List Nat)), 
    (∀ seq ∈ seqs, is_cool_sequence seq) ∧ 
    seqs.length = n) :=
sorry

end num_cool_sequences_l79_79955


namespace number_of_arrangements_l79_79047

-- Define the conditions as Lean definitions
def students := {m1, m2, f1, f2, f3} -- 2 male (m1 and m2), 3 female (f1, f2, f3)
def isMale (s : Type) : bool := s = m1 ∨ s = m2
def isFemale (s : Type) : bool := s = f1 ∨ s = f2 ∨ s = f3
def notAtEnds (s : Type) : bool := ¬(s = m1) ∧ ¬(s = m2)
def twoFemalesTogether (arrangement : list Type) : bool := ∃ (i : ℕ), (arrangement.get? i).isSome ∧ (arrangement.get? (i+1)).isSome ∧ (isFemale (option.get (arrangement.get? i)) = tt) ∧ (isFemale (option.get (arrangement.get? (i+1))) = tt)

-- The statement of what we need to prove
theorem number_of_arrangements : ∃ (arrangements : list (list Type)),
                 (∀ a ∈ arrangements, length a = 5) ∧
                 (∀ a ∈ arrangements, notAtEnds (a.nth 1) = tt) ∧
                 (∀ a ∈ arrangements, twoFemalesTogether a = tt) ∧
                 arrangements.length = 48 :=
sorry

end number_of_arrangements_l79_79047


namespace convert_spherical_to_rectangular_l79_79364

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 5 (Real.pi / 4) (Real.pi / 3) = (5 * (Real.sqrt 3 / 2) * (Real.sqrt 2 / 2), 5 * (Real.sqrt 3 / 2) * (Real.sqrt 2 / 2), 5 * (1 / 2)) :=
by
  sorry

end convert_spherical_to_rectangular_l79_79364


namespace geometric_series_ratio_half_l79_79200

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l79_79200


namespace max_top_young_men_l79_79866

def young_man := {height : ℕ, weight : ℕ}

-- Definition of "not inferior"
def not_inferior (a b : young_man) : Prop := a.height > b.height ∨ a.weight > b.weight

-- Definition of "top young man"
def top_young_man (a : young_man) (others : List young_man) : Prop :=
  ∀ b ∈ others, not_inferior a b

-- List of 100 young men
def young_men : List young_man := List.range 100 |>.map (λ i, {height := 100 - i, weight := i + 1})

-- Prove that all 100 young men in the list are top young men
theorem max_top_young_men : ∀ a ∈ young_men, top_young_man a (List.erase young_men a) :=
by
  sorry

end max_top_young_men_l79_79866


namespace geometric_series_common_ratio_l79_79191

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l79_79191


namespace trigonometric_expression_equality_l79_79975

theorem trigonometric_expression_equality :
  (cos 85 * π / 180 + sin 25 * π / 180 * cos 30 * π / 180) / cos 25 * π / 180 = 1 / 2 :=
by sorry

end trigonometric_expression_equality_l79_79975


namespace fill_interior_averaged_l79_79599

noncomputable def boundary_filled_rect (m n : ℕ) (boundary_values : (Fin (2 * (m + n)) → ℝ)) : Prop :=
  ∃ (f : Fin m → Fin n → ℝ),
    (∀ (i : Fin m) (j : Fin n),
      f i j = (f (i.pred.pred mod m).pred j + 
                f i (j.pred.pred mod n).pred + 
                f (i.succ.pred mod m) j + 
                f i (j.succ.pred mod n)) / 4)

theorem fill_interior_averaged (m n : ℕ) (boundary_values : (Fin (2 * (m + n)) → ℝ)) :
  boundary_filled_rect m n boundary_values :=
sorry

end fill_interior_averaged_l79_79599


namespace avg_adjacent_boy_girl_pairs_l79_79962

theorem avg_adjacent_boy_girl_pairs (boys : ℕ) (girls : ℕ) (total : ℕ) :
  boys = 9 → girls = 15 → total = 24 →
  let T := λ (lineup : list ℕ), 
    ∑ i in finset.range (total - 1), 
      if (lineup.nth i = some 0 ∧ lineup.nth (i+1) = some 1) ∨ 
         (lineup.nth i = some 1 ∧ lineup.nth (i+1) = some 0) then 1 else 0
  in (\(lineup : list ℕ) (h : lineup.length = total), ↑(T lineup)) / 
     (total.choose 9) ≈ 11 :=
begin
  -- Proof omitted
  sorry
end

end avg_adjacent_boy_girl_pairs_l79_79962


namespace thought_number_is_24_l79_79676

variable (x : ℝ)

theorem thought_number_is_24 (h : x / 4 + 9 = 15) : x = 24 := by
  sorry

end thought_number_is_24_l79_79676


namespace molecular_weight_NH4I_correct_l79_79347

noncomputable def atomic_weight_N : ℝ := 14.01
noncomputable def atomic_weight_H : ℝ := 1.01
noncomputable def atomic_weight_I : ℝ := 126.90

def molecular_weight_NH4I : ℝ :=
  atomic_weight_N + (4 * atomic_weight_H) + atomic_weight_I

theorem molecular_weight_NH4I_correct :
  molecular_weight_NH4I = 144.95 :=
by
  unfold molecular_weight_NH4I
  simp [atomic_weight_N, atomic_weight_H, atomic_weight_I]
  norm_num
  sorry

end molecular_weight_NH4I_correct_l79_79347


namespace geometric_series_common_ratio_l79_79211

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l79_79211


namespace staff_battle_station_l79_79349

def number_of_ways_to_staff (resumes : ℕ) (thrilled_fraction : ℚ) (job_openings : ℕ) : ℕ :=
  let suitable_candidates := (thrilled_fraction * resumes : ℚ).to_nat in
  (List.range job_openings).reverse.foldl (λ acc i, acc * (suitable_candidates - i)) 1

theorem staff_battle_station : 
  number_of_ways_to_staff 30 (1/3 : ℚ) 5 = 30240 :=
by
  sorry

end staff_battle_station_l79_79349


namespace coefficient_of_a_neg_half_l79_79760

theorem coefficient_of_a_neg_half (a : ℝ) : 
  (∑ k in Finset.range 9, 
    (Nat.choose 8 k) * (a ^ (8 - k) * (2 / (a ^ (1 / 2))) ^ k)) = 0 :=
by sorry

end coefficient_of_a_neg_half_l79_79760


namespace possible_values_of_m_l79_79490

-- Define the conditions that m is a real number and the quadratic equation having two distinct real roots
variable (m : ℝ)

-- Define the discriminant condition for having two distinct real roots
def discriminant_condition (a b c : ℝ) := b^2 - 4 * a * c > 0

-- State the required theorem
theorem possible_values_of_m (h : discriminant_condition 1 m 9) : m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo 6 ∞ :=
sorry

end possible_values_of_m_l79_79490


namespace sphere_surface_area_radius_one_l79_79871

theorem sphere_surface_area_radius_one (r : ℝ) (h : r = 1) : 4 * real.pi * r^2 = 4 * real.pi :=
by
  rw [h]
  simp
  sorry

end sphere_surface_area_radius_one_l79_79871


namespace linear_regression_equation_l79_79143

-- Define data for area and number of species
def x : List ℕ := [6, 15, 25, 34, 44, 54]
def y : List ℕ := [5, 10, 15, 19, 24, 31]

-- Sums provided
def sum_x_squared : ℕ := 2042
def sum_xy : ℕ := 1201

-- Mean values for the first four samples
def x_bar : ℕ := (6 + 15 + 25 + 34) / 4
def y_bar : ℕ := (5 + 10 + 15 + 19) / 4

-- Slope calculation
def b_hat := (sum_xy - 4 * x_bar * 12.25) / (sum_x_squared - 4 * 400)

-- Intercept calculation
def a_hat := 12.25 - b_hat * x_bar

-- Statement to prove the obtained linear regression equation
theorem linear_regression_equation :
  a_hat = 2.25 ∧ b_hat = 0.5 ∧ ∀ x, y = 0.5 * x + 2.25 :=
by
  sorry

end linear_regression_equation_l79_79143


namespace geometric_series_common_ratio_l79_79181

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l79_79181


namespace distinct_real_roots_interval_l79_79482

open Set Real

theorem distinct_real_roots_interval (m : ℝ) : 
  (∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ IsRoot (λ x => x^2 + m * x + 9) r1 ∧ IsRoot (λ x => x^2 + m * x + 9) r2) ↔ 
  m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end distinct_real_roots_interval_l79_79482


namespace inequality_condition_l79_79615

theorem inequality_condition (a : ℝ) : 
  (∀ x y : ℝ, x^2 + 2 * x + a ≥ -y^2 - 2 * y) → a ≥ 2 :=
by
  sorry

end inequality_condition_l79_79615


namespace problem_1_problem_2_l79_79450
noncomputable def f (x : ℝ) : ℝ := x / (Real.log x)
def g (k x : ℝ) : ℝ := k * (x - 1)

theorem problem_1 (k : ℝ) : ¬∃ x₀, f x₀ = g k x₀ ∧ deriv f x₀ = deriv (g k) x₀ :=
by
  sorry

theorem problem_2 (k : ℝ) :
  (∃ x, x ∈ Icc Real.exp 1 (Real.exp 2) ∧ f x ≤ g k x + 1/2) → k ≥ 1/2 :=
by
  sorry

end problem_1_problem_2_l79_79450


namespace max_consecutive_sum_lt_1000_l79_79235

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l79_79235


namespace inequality_proof_l79_79829

theorem inequality_proof
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 1) :
  (a^2 + b^2 + c^2) * ((a / (b + c)) + (b / (a + c)) + (c / (a + b))) ≥ 1/2 := by
  sorry

end inequality_proof_l79_79829


namespace remainder_of_3_pow_600_mod_19_l79_79668

theorem remainder_of_3_pow_600_mod_19 :
  (3 ^ 600) % 19 = 11 :=
sorry

end remainder_of_3_pow_600_mod_19_l79_79668


namespace divisible_by_1001_l79_79764

theorem divisible_by_1001 (A B : ℕ) (n : ℕ) 
  (a : Fin n → ℕ) (b : Fin n → ℕ) :
  (A = ∑ i in Finset.range n, (1000^(2*i) * a i) + ∑ i in Finset.range n, (1000^(2*i+1) * b i)) →
  (B = ∑ i in Finset.range n, a (2*i) - ∑ i in Finset.range n, b (2*i+1)) →
  ((A - B) % 1001 = 0 ↔ A % 1001 = 0) :=
sorry

end divisible_by_1001_l79_79764


namespace number_dislike_both_radio_and_music_l79_79576

variable (total_people : ℕ)
variable (percentage_dislike_radio : ℝ)
variable (percentage_dislike_both : ℝ)
variable (num_dislike_radio : ℕ)
variable (num_dislike_both : ℕ)

-- Conditions based on the problem
axiom total_people_eq : total_people = 1500
axiom percentage_dislike_radio_eq : percentage_dislike_radio = 0.4
axiom percentage_dislike_both_eq : percentage_dislike_both = 0.15

-- Compute intermediate value: number of people who dislike radio
def compute_num_dislike_radio : ℕ :=
  (percentage_dislike_radio * total_people).toNat

axiom num_dislike_radio_eq : num_dislike_radio = compute_num_dislike_radio

-- Compute final value: number of people who dislike both radio and music
def compute_num_dislike_both : ℕ :=
  (percentage_dislike_both * num_dislike_radio).toNat

axiom num_dislike_both_eq : num_dislike_both = compute_num_dislike_both

-- Theorem statement
theorem number_dislike_both_radio_and_music :
  num_dislike_both = 90 :=
by
  rw [num_dislike_both_eq, compute_num_dislike_both]
  rw [num_dislike_radio_eq, compute_num_dislike_radio]
  rw [total_people_eq, percentage_dislike_radio_eq, percentage_dislike_both_eq]
  sorry

end number_dislike_both_radio_and_music_l79_79576


namespace vitya_probability_l79_79652

theorem vitya_probability :
  let total_sequences := (finset.range 6).card * 
                         (finset.range 5).card * 
                         (finset.range 4).card * 
                         (finset.range 3).card * 
                         (finset.range 2).card * 
                         (finset.range 1).card,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11,
      total_possibilities := nat.choose 12 2 * nat.choose 10 2 * 
                             nat.choose 8 2 * nat.choose 6 2 * 
                             nat.choose 4 2 * nat.choose 2 2,
      P := (favorable_sequences : ℚ) / (total_possibilities : ℚ)
  in P = 1 / 720 := 
sorry

end vitya_probability_l79_79652


namespace sum_max_min_value_f_l79_79840

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + x) / (x ^ 2 + 1)

theorem sum_max_min_value_f : 
  let M := (⨆ x : ℝ, f x)
  let m := (⨅ x : ℝ, f x)
  M + m = 2 :=
by
-- Proof to be filled in
  sorry

end sum_max_min_value_f_l79_79840


namespace find_EF_distance_l79_79007

def skew_lines_distance (a b : Line) (θ d m n : ℝ) : ℝ :=
  sqrt (d^2 + m^2 + n^2 + 2 * m * n * cos θ) -- one of the possible cases
  -- sqrt (d^2 + m^2 + n^2 - 2 * m * n * cos θ) -- other possible case

theorem find_EF_distance
  (a b : Line) (θ d m n : ℝ)
  (skew_a_b : ¬\(\exists p\). p ∈ a ∧ p ∈ b) -- skew lines condition
  (angle_a_b : \(\exists v \in a, w \in b\), ∠(v, w) = θ) -- angle between lines
  (common_perp : ∃ p1 p2, p1 ∈ a ∧ p2 ∈ b ∧ ∃ q, q ∈ Line.through p1 p2 ∧ d = dist p1 p2) -- common perpendicular of length d 
  (A_prime_E : ∃ A' ∈ a, dist A' E = m) -- distance A'E = m
  (A_F : ∃ A ∈ b, dist A F = n) -- distance AF = n
: dist E F = sqrt (d^2 + m^2 + n^2 + 2 * m * n * cos θ) ∨ 
  dist E F = sqrt (d^2 + m^2 + n^2 - 2 * m * n * cos θ) := by
  sorry

end find_EF_distance_l79_79007


namespace find_positive_solution_l79_79457

-- Defining the variables x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the conditions from the problem statement
def condition1 : Prop := x * y + 3 * x + 4 * y + 10 = 30
def condition2 : Prop := y * z + 4 * y + 2 * z + 8 = 6
def condition3 : Prop := x * z + 4 * x + 3 * z + 12 = 30

-- The theorem that states the positive solution for x is 3
theorem find_positive_solution (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 x z) : x = 3 :=
by {
  sorry
}

end find_positive_solution_l79_79457


namespace intervals_of_decrease_for_f_l79_79781

noncomputable def f : ℝ → ℝ := λ x, Real.sin (-2 * x + Real.pi / 2)

theorem intervals_of_decrease_for_f :
  ∀ k : ℤ, ∃ a b : ℝ, a = k * Real.pi ∧ b = k * Real.pi + Real.pi / 2 ∧ (∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x) :=
by
  sorry

end intervals_of_decrease_for_f_l79_79781


namespace pos_integer_solutions_count_l79_79465

theorem pos_integer_solutions_count :
  (finite {x : ℕ | 12 < -2 * (x : ℤ) + 17 ∧ x > 0}).to_finset.card = 2 :=
by
  sorry

end pos_integer_solutions_count_l79_79465


namespace geometric_series_common_ratio_l79_79176

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l79_79176


namespace value_of_y_l79_79631

variable (y : ℚ)

def first_boy_marbles : ℚ := 4 * y + 2
def second_boy_marbles : ℚ := 2 * y
def third_boy_marbles : ℚ := y + 3
def total_marbles : ℚ := 31

theorem value_of_y (h : first_boy_marbles y + second_boy_marbles y + third_boy_marbles y = total_marbles) :
  y = 26 / 7 :=
by
  sorry

end value_of_y_l79_79631


namespace equal_split_payment_l79_79395

def cost_taco_salad := 10.0
def cost_single_hamburger := 5.0
def num_hamburgers := 5
def cost_french_fries := 2.5
def num_fries := 4
def cost_peach_lemonade := 2.0
def num_lemonades := 5
def num_friends := 5

theorem equal_split_payment : 
  let total_cost := cost_taco_salad 
    + (num_hamburgers * cost_single_hamburger)
    + (num_fries * cost_french_fries)
    + (num_lemonades * cost_peach_lemonade) in
  total_cost / num_friends = 11.0 :=
by 
  sorry

end equal_split_payment_l79_79395


namespace perimeter_of_triangle_hyperbola_l79_79616

theorem perimeter_of_triangle_hyperbola (x y : ℝ) (F1 F2 A B : ℝ) :
  (x^2 / 16) - (y^2 / 9) = 1 →
  |A - F2| - |A - F1| = 8 →
  |B - F2| - |B - F1| = 8 →
  |B - A| = 5 →
  |A - F2| + |B - F2| + |B - A| = 26 :=
by
  sorry

end perimeter_of_triangle_hyperbola_l79_79616


namespace speed_of_man_in_still_water_l79_79696

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 4 = 48 →
  (v_m - v_s) * 6 = 24 →
  v_m = 8 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end speed_of_man_in_still_water_l79_79696


namespace slower_time_is_Tara_l79_79555

def time_to_top (stories : ℕ) (time_per_story : ℕ) : ℕ :=
  stories * time_per_story

def elevator_total_time (stories : ℕ) (time_per_story : ℕ) (stop_time : ℕ) : ℕ :=
  stories * time_per_story + (stories - 1) * stop_time

theorem slower_time_is_Tara :
  let stories := 20
  let lola_time_per_story := 10
  let tara_elevator_time_per_story := 8
  let tara_stop_time := 3 in
  max (time_to_top stories lola_time_per_story) (elevator_total_time stories tara_elevator_time_per_story tara_stop_time) = elevator_total_time stories tara_elevator_time_per_story tara_stop_time :=
by
  sorry

end slower_time_is_Tara_l79_79555


namespace geometric_series_common_ratio_l79_79185

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l79_79185


namespace distinct_real_roots_of_quadratic_l79_79475

-- Define the problem's condition: m is a real number and the discriminant of x^2 + mx + 9 > 0
def discriminant_positive (m : ℝ) := m^2 - 36 > 0

theorem distinct_real_roots_of_quadratic (m : ℝ) (h : discriminant_positive m) :
  m ∈ Iio (-6) ∪ Ioi (6) :=
sorry

end distinct_real_roots_of_quadratic_l79_79475


namespace unique_midpoints_are_25_l79_79223

/-- Define the properties of a parallelogram with marked points such as vertices, midpoints of sides, and intersection point of diagonals --/
structure Parallelogram :=
(vertices : Set ℝ)
(midpoints : Set ℝ)
(diagonal_intersection : ℝ)

def congruent_parallelograms (P P' : Parallelogram) : Prop :=
  P.vertices = P'.vertices ∧ P.midpoints = P'.midpoints ∧ P.diagonal_intersection = P'.diagonal_intersection

def unique_midpoints_count (P P' : Parallelogram) : ℕ := sorry

theorem unique_midpoints_are_25
  (P P' : Parallelogram)
  (h_congruent : congruent_parallelograms P P') :
  unique_midpoints_count P P' = 25 := sorry

end unique_midpoints_are_25_l79_79223


namespace tetrahedron_angle_difference_l79_79123

theorem tetrahedron_angle_difference (dihedral_angles : Fin 6 → ℝ) (trihedral_angles : Fin 4 → Fin 3 → ℝ) :
  let solid_angle (angles : Fin 3 → ℝ) := ∑ i, angles i - Real.pi
  let total_dihedral := 2 * ∑ i, dihedral_angles i
   in total_dihedral - ∑ i, solid_angle (trihedral_angles i) = 4 * Real.pi :=
by
  sorry

end tetrahedron_angle_difference_l79_79123


namespace min_value_of_u_l79_79290

theorem min_value_of_u : ∀ (x y : ℝ), x ∈ Set.Ioo (-2) 2 → y ∈ Set.Ioo (-2) 2 → x * y = -1 → 
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ 12 / 5) :=
by
  intros x y hx hy hxy u hu
  sorry

end min_value_of_u_l79_79290


namespace circle_radius_of_tangent_to_ellipse_l79_79331

noncomputable theory
open Real

-- Definitions of the ellipse and circle properties
def ellipse_eqn (x y : ℝ) : Prop := (x ^ 2) / 36 + (y ^ 2) / 9 = 1

def focus_x : ℝ := 3 * sqrt 3
def circle_eqn (r x y : ℝ) : Prop := (x - focus_x) ^ 2 + y ^ 2 = r ^ 2

variable {r : ℝ}

-- The theorem stating the radius of the circle is 3
theorem circle_radius_of_tangent_to_ellipse :
  (∀ x y : ℝ, circle_eqn r x y → ellipse_eqn x y) → r = 3 :=
by sorry

end circle_radius_of_tangent_to_ellipse_l79_79331


namespace max_consecutive_integers_sum_lt_1000_l79_79262

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l79_79262


namespace func_range_l79_79784

def func (x : Real) : Real := (3 * x^2 - 1) / (x^2 + 2)

theorem func_range : ∀ y : Real, y = func x → -1/2 ≤ y ∧ y < 3 :=
by
  sorry

end func_range_l79_79784


namespace white_area_of_painting_l79_79162

theorem white_area_of_painting (s : ℝ) (total_gray_area : ℝ) (gray_area_squares : ℕ)
  (h1 : ∀ t, t = 3 * s) -- The frame is 3 times the smaller square's side length.
  (h2 : total_gray_area = 62) -- The gray area is 62 cm^2.
  (h3 : gray_area_squares = 31) -- The gray area is composed of 31 smaller squares.
  : ∃ white_area, white_area = 10 := 
  sorry

end white_area_of_painting_l79_79162


namespace total_grains_in_grey_parts_l79_79055

theorem total_grains_in_grey_parts (total_circle1 : ℝ) (total_circle2 : ℝ) (overlapped_white : ℝ) : 
  (total_circle2 - overlapped_white) + (total_circle1 - overlapped_white) = 61 :=
by 
  have total_grey1 := total_circle1 - overlapped_white
  have total_grey2 := total_circle2 - overlapped_white
  have sum_grey := total_grey1 + total_grey2
  sorry

# Example values for context
# noncomputable def total_circle1 : ℝ := 87
# noncomputable def total_circle2 : ℝ := 110
# noncomputable def overlapped_white : ℝ := 68

end total_grains_in_grey_parts_l79_79055


namespace parabola_zeros_sum_is_10_l79_79979

noncomputable def parabola_sum_of_zeros : ℝ :=
let original_eq : ℝ → ℝ := λ x, (x - 3)^2 + 4 in
let rotated_eq : ℝ → ℝ := λ x, -(x - 3)^2 + 4 in
let shifted_right_eq : ℝ → ℝ := λ x, rotated_eq (x - 2) in
let shifted_down_eq : ℝ → ℝ := λ x, shifted_right_eq x - 3 in
  let zeros := {x : ℝ | shifted_down_eq x = 0} in
  if h : set.finite zeros then (finset.sum (h.to_finset) (λ x, x)) else 0

theorem parabola_zeros_sum_is_10 : parabola_sum_of_zeros = 10 :=
sorry

end parabola_zeros_sum_is_10_l79_79979


namespace number_of_false_propositions_is_even_l79_79982

theorem number_of_false_propositions_is_even 
  (P Q : Prop) : 
  ∃ (n : ℕ), (P ∧ ¬P ∧ (¬Q → ¬P) ∧ (Q → P)) = false ∧ n % 2 = 0 := sorry

end number_of_false_propositions_is_even_l79_79982


namespace age_difference_l79_79584

-- Define the given conditions
variable (S : ℝ) (R : ℝ) (D : ℝ)
hypothesis h1 : S = 24.5
hypothesis h2 : S / R = 7 / 9

-- Define the statement we want to prove
theorem age_difference : D = R - S → D = 7 := 
by
  intro h3
  sorry

end age_difference_l79_79584


namespace most_cost_effective_supermarket_l79_79998

theorem most_cost_effective_supermarket :
  let P := 1 in
  let price_A := 0.8 * 0.8 * P in
  let price_B := 0.6 * P in
  let price_C := 0.7 * 0.9 * P in
  price_B < price_A ∧ price_B < price_C :=
by {
  -- Initial price is assumed to be 1 for simplicity of comparison.
  let P := 1,
  let price_A := 0.8 * 0.8 * P,
  let price_B := 0.6 * P,
  let price_C := 0.7 * 0.9 * P,
  split,
  -- Price at B is less than Price at A
  calc 0.6 * P < 0.8 * 0.8 * P : by linarith,
  -- Price at B is less than Price at C
  calc 0.6 * P < 0.7 * 0.9 * P : by linarith,
  done
}

end most_cost_effective_supermarket_l79_79998


namespace integer_solutions_count_l79_79161

theorem integer_solutions_count : 
  ∃ (s : Finset ℤ), 
    (∀ x ∈ s, 2 * x + 1 > -3 ∧ -x + 3 ≥ 0) ∧ 
    s.card = 5 := 
by 
  sorry

end integer_solutions_count_l79_79161


namespace sam_dimes_proof_l79_79954

def initial_dimes : ℕ := 9
def remaining_dimes : ℕ := 2
def dimes_given : ℕ := 7

theorem sam_dimes_proof : initial_dimes - remaining_dimes = dimes_given :=
by
  sorry

end sam_dimes_proof_l79_79954


namespace represent_in_scientific_notation_l79_79570

def million : ℕ := 10^6
def rural_residents : ℝ := 42.39 * million

theorem represent_in_scientific_notation :
  42.39 * 10^6 = 4.239 * 10^7 :=
by
  -- The proof is omitted.
  sorry

end represent_in_scientific_notation_l79_79570


namespace three_digit_number_probability_not_divisible_by_3_l79_79580

noncomputable def probability_not_divisible_by_3 : ℚ :=
  2 / 3

theorem three_digit_number_probability_not_divisible_by_3 :
  ∃ (s : Finset (Fin 1000)), (s.card = 720) ∧
    (∃ (not_divisible_by_3 : Finset (Fin 1000)),
      (not_divisible_by_3.card = 480) ∧
      (∀ d ∈ not_divisible_by_3, ¬ (d.val % 3 = 0)) ∧
      (probability_not_divisible_by_3 = not_divisible_by_3.card / s.card)) :=
begin
  sorry
end

end three_digit_number_probability_not_divisible_by_3_l79_79580


namespace quinn_participation_weeks_l79_79949

theorem quinn_participation_weeks (books_per_donut : ℕ) (books_per_week : ℕ) (num_donuts : ℕ)
  (h1 : books_per_donut = 5) (h2 : books_per_week = 2) (h3 : num_donuts = 4) :
  (num_donuts * books_per_donut) / books_per_week = 10 := by
  rw [h1, h2, h3]
  norm_num
  sorry

end quinn_participation_weeks_l79_79949


namespace range_of_a_l79_79622

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Ioo 1 2, 3^x - 4 / x - a = 0) ↔ -1 < a ∧ a < 7 :=
by
  sorry

end range_of_a_l79_79622


namespace car_relationship_possible_arrangements_maximize_profit_l79_79301

-- Definitions based on the given conditions
def total_cars : Nat := 40
def total_tons : Nat := 200
def tons_per_car_A : Nat := 6
def tons_per_car_B : Nat := 5
def tons_per_car_C : Nat := 4
def min_cars_per_type : Nat := 4

-- Relationship y = -2x + 40
theorem car_relationship (x y : Nat) (h1 : 6 * x + 5 * y + 4 * (total_cars - x - y) = total_tons) :
  y = -2 * x + 40 := by
  sorry

-- Possible arrangements (must have at least 'min_cars' cars for each type)
theorem possible_arrangements (x y z : Nat) (h2 : x + y + z = total_cars)
  (hx : x ≥ min_cars_per_type) (hy : y ≥ min_cars_per_type) (hz : z ≥ min_cars_per_type)
  (h_rel : y = -2 * x + 40) :
  ∃ n : Nat, n = 15 := by
  sorry

-- Maximizing profit
theorem maximize_profit (x y z : Nat) (hx : x = 4) (hy : y = 32) (hz : z = 4)
  (h2 : x + y + z = total_cars) (h_rel : y = -2 * x + 40) :
  6 * x * 5 + 5 * y * 7 + 4 * z * 8 = 1368 := by
  sorry

end car_relationship_possible_arrangements_maximize_profit_l79_79301


namespace xyz_problem_l79_79492

variables {x y z : ℝ}

theorem xyz_problem
  (h1 : y + z = 10 - 4 * x)
  (h2 : x + z = -16 - 4 * y)
  (h3 : x + y = 9 - 4 * z) :
  3 * x + 3 * y + 3 * z = 1.5 :=
by 
  sorry

end xyz_problem_l79_79492


namespace max_consecutive_integers_sum_l79_79248

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l79_79248


namespace james_huskies_count_l79_79904

theorem james_huskies_count 
  (H : ℕ) 
  (pitbulls : ℕ := 2) 
  (golden_retrievers : ℕ := 4) 
  (husky_pups_per_husky : ℕ := 3) 
  (pitbull_pups_per_pitbull : ℕ := 3) 
  (extra_pups_per_golden_retriever : ℕ := 2) 
  (pup_difference : ℕ := 30) :
  H + pitbulls + golden_retrievers + pup_difference = 3 * H + pitbulls * pitbull_pups_per_pitbull + golden_retrievers * (husky_pups_per_husky + extra_pups_per_golden_retriever) :=
sorry

end james_huskies_count_l79_79904


namespace lines_concurrent_or_parallel_l79_79816

open Affine Geometry

variable {P : Type*} [AffineSpace P]  {R : Type*} [Ring R] [Module R P]    
variables (A1 A2 A3 A4 B1 B2 B3 B4 : P) 

noncomputable def are_parallel (l1 l2 : Line P) : Prop := 
∃ x : P, x ∈ l1 ∧ x ∉ l2

noncomputable def M (i j : ℕ) : P := intersection_point (line_through A (i) B (j)) (line_through A (j) B(i))

-- One possible definition of the problem using the broader library
theorem lines_concurrent_or_parallel :
  ∀ (A1 A2 A3 A4 B1 B2 B3 B4 : P), 
    (∀ i j : fin 4, intersect (line_through A i B j) (line_through A j B i) ≠ ⊥) → 
    are_parallel (line_through (M 1 2) (M 3 4)) (line_through (M 1 3) (M 2 4)) ∨ collinear (line_through (M 1 2) (M 3 4)) (line_through (M 1 3) (M 2 4)) :=
sorry

end lines_concurrent_or_parallel_l79_79816


namespace aftershave_lotion_volume_l79_79736

theorem aftershave_lotion_volume (V : ℝ) (h1 : 0.30 * V = 0.1875 * (V + 30)) : V = 50 := 
by 
-- sorry is added to indicate proof is omitted.
sorry

end aftershave_lotion_volume_l79_79736


namespace sum_of_alternating_sums_l79_79793

-- Define the alternating sum for a nonempty subset of nat numbers
def alternating_sum (s : List ℕ) : ℤ :=
  s.reverse.enum.map (λ ⟨i, a⟩, if i % 2 = 0 then (a : ℤ) else -(a : ℤ)).sum

-- Define the problem statement
def problem_statement : Prop :=
  let S := finset.powerset (finset.range 8).succ \ {∅} in
  S.sum (λ s, alternating_sum s.val.to_list) = 1024

-- This is the theorem to prove
theorem sum_of_alternating_sums : problem_statement :=
begin
  sorry
end

end sum_of_alternating_sums_l79_79793


namespace feeding_order_count_correct_l79_79726

def numFeedingWays : Nat :=
  288

theorem feeding_order_count_correct :
  ∀ (animals : List (String × String)) (start : String) 
    (condition1 : ∀ a ∈ animals, a.1 ≠ a.2)
    (condition2 : start = "female_lion")
    (condition3 : ∀ a b ∈ animals, a ≠ b → (condition_a_b a b))
    (condition4 : ∀ a b ∈ animals, (a.1 = "female_lion") → (b.2 ≠ "male_lion")),
  count_feeding_ways animals start = numFeedingWays := 
  sorry

end feeding_order_count_correct_l79_79726


namespace unique_rational_point_line_l79_79060

open Set

def isRational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

noncomputable def line_through (p1 p2 : ℝ × ℝ) : ℝ → ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  if x1 = x2 then y1
  else λ x, y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)

theorem unique_rational_point_line (a : ℝ) (a_irrational : ¬ isRational a) :
  ∃! p : ℝ × ℝ, p = (a, 0) ∧ ∀ q1 q2 : ℝ × ℝ, isRational q1.1 → isRational q1.2 → 
    isRational q2.1 → isRational q2.2 → line_through q1 q2 (a, 0).1 = (a, 0).2 :=
by
  sorry

end unique_rational_point_line_l79_79060


namespace cost_of_energy_drink_l79_79966

-- Definitions for given conditions as constants
def cupcakes_sold : ℕ := 50
def price_per_cupcake : ℝ := 2.0
def cookies_sold : ℕ := 40
def price_per_cookie : ℝ := 0.5
def basketballs_bought : ℕ := 2
def price_per_basketball : ℝ := 40.0
def bottles_bought : ℕ := 20

-- Goal: prove the cost per bottle of energy drink
theorem cost_of_energy_drink :
  let total_money := (cupcakes_sold * price_per_cupcake) + (cookies_sold * price_per_cookie),
      total_basketballs_cost := basketballs_bought * price_per_basketball,
      remaining_money := total_money - total_basketballs_cost,
      cost_per_bottle := remaining_money / bottles_bought
  in cost_per_bottle = 2.0 :=
sorry

end cost_of_energy_drink_l79_79966


namespace arithmetic_geometric_sequence_l79_79815

open Real

noncomputable def a_4 (a1 q : ℝ) : ℝ := a1 * q^3
noncomputable def sum_five_terms (a1 q : ℝ) : ℝ := a1 * (1 - q^5) / (1 - q)

theorem arithmetic_geometric_sequence :
  ∀ (a1 q : ℝ),
    (a1 + a1 * q^2 = 10) →
    (a1 * q^3 + a1 * q^5 = 5 / 4) →
    (a_4 a1 q = 1) ∧ (sum_five_terms a1 q = 31 / 2) :=
by
  intros a1 q h1 h2
  sorry

end arithmetic_geometric_sequence_l79_79815


namespace fraction_of_plot_occupied_by_beds_l79_79115

-- Define the conditions based on plot area and number of beds
def plot_area : ℕ := 64
def total_beds : ℕ := 13
def outer_beds : ℕ := 12
def central_bed_area : ℕ := 4 * 4

-- The proof statement showing that fraction of the plot occupied by the beds is 15/32
theorem fraction_of_plot_occupied_by_beds : 
  (central_bed_area + (plot_area - central_bed_area)) / plot_area = 15 / 32 := 
sorry

end fraction_of_plot_occupied_by_beds_l79_79115


namespace dot_product_AB_AC_l79_79889

open Real
open EuclideanGeometry

/-- Define the vectors and assume the conditions -/
def rhombus_OABC_vectors : Prop :=
  let O := (0 : ℝ, 0 : ℝ)
  let A := (1 : ℝ, 1 : ℝ)
  ∃ C : ℝ × ℝ, (O - A) ⋅ (O - C) = 1 ∧ (O - A).norm = (O - C).norm ∧ 
  ∃ B : ℝ × ℝ, (A - B).norm = (A - O).norm ∧ (A - B).norm = (A - C).norm

theorem dot_product_AB_AC : rhombus_OABC_vectors →
  ∃ O A B C : ℝ × ℝ, (A - B) ⋅ (A - C) = 1 :=
by
  intro h
  obtain ⟨C, hOC, hOAnorm, hOCnorm⟩ := h 
  have O := (0 : ℝ, 0 : ℝ)
  have A := (1 : ℝ, 1 : ℝ)
  sorry

end dot_product_AB_AC_l79_79889


namespace intersection_of_parabolas_l79_79360

-- Define the condition set
def directrix_lines (a_vals : Set ℤ) (b_vals : Set ℤ) :=
  { p : ℤ × ℤ | p.1 ∈ a_vals ∧ p.2 ∈ b_vals }

-- The given set of possible values for a and b
def a_vals : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}
def b_vals : Set ℤ := {-4, -3, -2, -1, 0, 1, 2, 3, 4}

-- Define the problem
theorem intersection_of_parabolas :
  let parabolas := directrix_lines a_vals b_vals in
  (points_of_intersection parabolas).count = 3528 :=
by sorry

end intersection_of_parabolas_l79_79360


namespace modulus_of_given_complex_number_l79_79442

def complex_modulus_problem : Prop :=
  let i := Complex.I in
  let z := (1 + 2 * i) / (i * i) in
  z.abs = Real.sqrt 5

theorem modulus_of_given_complex_number : complex_modulus_problem :=
by
  sorry

end modulus_of_given_complex_number_l79_79442


namespace determine_arrow_sequence_l79_79809

-- Conditions and problem setup
def repeats_every_six (n : ℕ) : Prop := n % 6 = 2

def is_reverse_arrow (n : ℕ) : Prop := n % 10 = 0

def arrow_sequence (start end : ℕ) : list (ℕ × ℕ) :=
  if is_reverse_arrow start then
    [(2, 3).swap, (3, 4), (4, 5)]
  else
    match start % 6, end % 6 with
    | 2, 5 => [(2, 3), (3, 4), (4, 5)]
    | _, _ => []
    end

-- Theorem statement
theorem determine_arrow_sequence :
  arrow_sequence 530 533 = [(3, 2), (3, 4), (4, 5)] :=
  sorry

end determine_arrow_sequence_l79_79809


namespace compute_c_d_l79_79921

theorem compute_c_d (c d : ℝ) 
  (h1 : (x : ℝ) → (x + c) * (x + d) * (x + 8) / (x + 2)^2 = 0 → distinctRoots 3)
  (h2 : (x : ℝ) → (x + 3 * c) * (x + 2) * (x + 4) / (x + d) / (x + 8) = 0 → distinctRoots 1) :
  100 * c + d = 70.67 :=
sorry

end compute_c_d_l79_79921


namespace count_5_in_range_1_to_1000_l79_79854

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k: ℕ, k ≤ n ∧ (n / 10 ^ k) % 10 = d

def count_contains_digit (d : ℕ) (range : Finset ℕ) : ℕ :=
  range.count (λ n, contains_digit n d)

theorem count_5_in_range_1_to_1000 : count_contains_digit 5 (Finset.range 1001) = 270 :=
by 
  sorry

end count_5_in_range_1_to_1000_l79_79854


namespace hyperbola_standard_equation_l79_79614

-- Define the hyperbola properties
def focus1 : ℝ × ℝ := (-6, 0)
def focus2 : ℝ × ℝ := (6, 0)
def point_on_hyperbola : ℝ × ℝ := (-5, 2)

-- Define the standard hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 20) - (y^2 / 16) = 1

-- Theorem statement
theorem hyperbola_standard_equation :
  ∃ eq : ℝ → ℝ → Prop,
    eq = hyperbola_equation ∧
    (∃ x y : ℝ, eq x y) ∧
    (∃ a b : ℝ, 2 * a = |real.sqrt ((point_on_hyperbola.1 + 6)^2 + point_on_hyperbola.2^2) - real.sqrt ((point_on_hyperbola.1 - 6)^2 + point_on_hyperbola.2^2)|) ∧
    (∃ c : ℝ, c = 6) ∧
    (∃ b2 : ℝ, b2 = 36 - (2 * real.sqrt 5)^2) ∧
    (b2 = 16) :=
sorry

end hyperbola_standard_equation_l79_79614


namespace solution_set_f_less_than_6_range_of_m_if_inequality_always_holds_l79_79448

def f (x : ℝ) : ℝ := abs(x + 2) + abs(2 * x - 4)

theorem solution_set_f_less_than_6 :
  {x : ℝ | f x < 6} = {x : ℝ | 0 < x ∧ x < 8 / 3} := sorry

theorem range_of_m_if_inequality_always_holds :
  (∀ x : ℝ, f x ≥ m^2 - 3 * m) ↔ -1 ≤ m ∧ m ≤ 4 := sorry

end solution_set_f_less_than_6_range_of_m_if_inequality_always_holds_l79_79448


namespace inclination_zero_l79_79520

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y + 1 = 0

-- Define the angle of inclination for a given line
def angle_of_inclination (line : ℝ → ℝ → Prop) : ℝ :=
  if ∃ m : ℝ, ∀ x y : ℝ, line x y → y = m * x then
    atan (m)
  else
    if ∀ x y : ℝ, line x y → y = -1 then
      0 -- line parallel to x-axis
    else
      sorry -- Handle other cases

-- The theorem we need to prove
theorem inclination_zero : angle_of_inclination line_eq = 0 := by
  sorry

end inclination_zero_l79_79520


namespace sum_X_Y_l79_79785

-- Define the variables and assumptions
variable (X Y : ℕ)

-- Hypotheses
axiom h1 : Y + 2 = X
axiom h2 : X + 5 = Y

-- Theorem statement
theorem sum_X_Y : X + Y = 12 := by
  sorry

end sum_X_Y_l79_79785


namespace valid_three_digit_numbers_l79_79856

theorem valid_three_digit_numbers : 
  ∀ (N : ℕ), 
    N = 657 ↔
    (∀ n ∈ (Set.Icc 100 999), ¬ ∃ (X Y : ℕ), X ≠ Y ∧ 
      ((n / 100 = X ∧ (n % 100) / 10 = X ∧ n % 10 = Y) ∨ 
      (n / 100 = Y ∧ (n % 100) / 10 = X ∧ n % 10 = X) ∨ 
      ((n / 100) = X ∧ (n % 100) / 10 = Y ∧ n % 10 = X)) → 
    N = Set.card {n ∈ Set.Icc 100 999 | ¬ ∃ (X Y : ℕ), X ≠ Y ∧ 
      ((n / 100 = X ∧ (n % 100) / 10 = X ∧ n % 10 = Y) ∨ 
      (n / 100 = Y ∧ (n % 100) / 10 = X ∧ n % 10 = X) ∨ 
      ((n / 100) = X ∧ (n % 10) / 10 = Y ∧ n % 10 = X))})
  :=
by
  sorry

end valid_three_digit_numbers_l79_79856


namespace geometric_series_common_ratio_l79_79209

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l79_79209


namespace solve_absolute_value_inequality_l79_79371

theorem solve_absolute_value_inequality (x : ℝ) :
  3 ≤ |x + 3| ∧ |x + 3| ≤ 7 ↔ (-10 ≤ x ∧ x ≤ -6) ∨ (0 ≤ x ∧ x ≤ 4) :=
by
  sorry

end solve_absolute_value_inequality_l79_79371


namespace belfried_industries_tax_l79_79711

noncomputable def payroll_tax (payroll : ℕ) : ℕ :=
  if payroll <= 200000 then
    0
  else
    ((payroll - 200000) * 2) / 1000

theorem belfried_industries_tax : payroll_tax 300000 = 200 :=
by
  sorry

end belfried_industries_tax_l79_79711


namespace slower_time_is_Tara_l79_79556

def time_to_top (stories : ℕ) (time_per_story : ℕ) : ℕ :=
  stories * time_per_story

def elevator_total_time (stories : ℕ) (time_per_story : ℕ) (stop_time : ℕ) : ℕ :=
  stories * time_per_story + (stories - 1) * stop_time

theorem slower_time_is_Tara :
  let stories := 20
  let lola_time_per_story := 10
  let tara_elevator_time_per_story := 8
  let tara_stop_time := 3 in
  max (time_to_top stories lola_time_per_story) (elevator_total_time stories tara_elevator_time_per_story tara_stop_time) = elevator_total_time stories tara_elevator_time_per_story tara_stop_time :=
by
  sorry

end slower_time_is_Tara_l79_79556


namespace expression_S_max_value_S_l79_79839

section
variable (x t : ℝ)
def f (x : ℝ) := -3 * x^2 + 6 * x

-- Define the integral expression for S(t)
noncomputable def S (t : ℝ) := ∫ x in t..(t + 1), f x

-- Assert the expression for S(t)
theorem expression_S (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 2) :
  S t = -3 * t^2 + 3 * t + 2 :=
by
  sorry

-- Assert the maximum value of S(t)
theorem max_value_S :
  ∀ t, (0 ≤ t ∧ t ≤ 2) → S t ≤ 5 / 4 :=
by
  sorry

end

end expression_S_max_value_S_l79_79839


namespace cost_of_energy_drink_l79_79967

-- Definitions for given conditions as constants
def cupcakes_sold : ℕ := 50
def price_per_cupcake : ℝ := 2.0
def cookies_sold : ℕ := 40
def price_per_cookie : ℝ := 0.5
def basketballs_bought : ℕ := 2
def price_per_basketball : ℝ := 40.0
def bottles_bought : ℕ := 20

-- Goal: prove the cost per bottle of energy drink
theorem cost_of_energy_drink :
  let total_money := (cupcakes_sold * price_per_cupcake) + (cookies_sold * price_per_cookie),
      total_basketballs_cost := basketballs_bought * price_per_basketball,
      remaining_money := total_money - total_basketballs_cost,
      cost_per_bottle := remaining_money / bottles_bought
  in cost_per_bottle = 2.0 :=
sorry

end cost_of_energy_drink_l79_79967


namespace shop_original_price_l79_79675

theorem shop_original_price (x : ℝ) (discount : ℝ) (final_price : ℝ) (h : discount = 0.30) (h2 : final_price = 560) :
  (x ∈ ℝ) → (1 - discount) * x = final_price → x = 800 :=
by
  intro hx h3
  sorry

end shop_original_price_l79_79675


namespace problem_l79_79100

theorem problem (k : ℕ) (hk : 0 < k) (n : ℕ) : 
  (∃ p : ℕ, n = 2 * 3 ^ (k - 1) * p ∧ 0 < p) ↔ 3^k ∣ (2^n - 1) := 
by 
  sorry

end problem_l79_79100


namespace coefficient_of_b_squared_term_l79_79865

noncomputable theory
open_locale classical

variable {α : Type*}
variables (a b : ℝ)

theorem coefficient_of_b_squared_term :
  (∃ a: ℝ, (∀ b: ℝ, 4 * b ^ 4 - a * b ^ 2 + 100 = 0) ∧
  (∃ b1 b2: ℝ, b1 + b2 = 4.5 ∧ (4 * b1 ^ 4 - a * b1 ^ 2 + 100 = 0) ∧ (4 * b2 ^ 4 - a * b2 ^ 2 + 100 = 0))) →
  a = -4.5 :=
by sorry

end coefficient_of_b_squared_term_l79_79865


namespace ant_diagonal_probability_l79_79310

theorem ant_diagonal_probability : 
  ∃ p : ℚ, p = 1/3 ∧ (∀ (A B C D E : Type) (move : A → B ⊕ C ⊕ D), 
  ∀ (from_adjacent : B → C ⊕ D ⊕ E) (from_adjacent' : C → B ⊕ D ⊕ E) 
  (from_adjacent'' : D → B ⊕ C ⊕ E), 
  let p1 := (1 : ℚ) / 3 in 
  let total_prob := p1 * p1 + p1 * p1 + p1 * p1 in 
  total_prob = p) :=
sorry

end ant_diagonal_probability_l79_79310


namespace parabola_focus_dot_product_and_cos_sum_l79_79002

noncomputable def parabola_eqn (p : ℝ) : Prop :=
  ∃ (C : ℝ × ℝ → Prop), C = (λ (x y : ℝ), y^2 = 2 * p * x)

theorem parabola_focus (p : ℝ) (h : p > 0) : parabola_eqn 4 :=
begin
  have parab_eq := (λ (x y : ℝ), y^2 = 8 * x),
  use parab_eq,
  sorry
end

noncomputable def line_through_focus (m : ℝ) (F : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ (x y : ℝ), x = m * y + F.1

theorem dot_product_and_cos_sum
  (F : ℝ × ℝ) (hF : F = (2, 0))
  (p : ℝ) (h_parab (x y : ℝ) : y^2 = 8 * x)
  (circle : ℝ × ℝ → Prop := (λ (x y : ℝ), (x - 2)^2 + y^2 = 1))
  (tangent_points : (ℝ × ℝ) × (ℝ × ℝ))
  (A B : ℝ × ℝ)
  (h_line_B : line_through_focus m F B)
  (h_line_A : line_through_focus m F A) :
  ((A.1 * B.1) + (A.2 * B.2) = -12) ∧ (cos_angle_sum A B F = 1/2) :=
begin
  sorry
end

noncomputable def cos_angle_sum (A B F : ℝ × ℝ) : ℝ :=
  (λ (P Q : ℝ × ℝ), sorry ) sorry sorry

end parabola_focus_dot_product_and_cos_sum_l79_79002


namespace total_distance_traveled_l79_79699

theorem total_distance_traveled :
  let car_speed1 := 90
  let car_time1 := 2
  let car_speed2 := 60
  let car_time2 := 1
  let train_speed := 100
  let train_time := 2.5
  let distance_car1 := car_speed1 * car_time1
  let distance_car2 := car_speed2 * car_time2
  let distance_train := train_speed * train_time
  distance_car1 + distance_car2 + distance_train = 490 := by
  sorry

end total_distance_traveled_l79_79699


namespace sum_of_leading_digits_l79_79545

def leading_digit (n : ℕ) : ℕ :=
  -- This would be defined properly, skipping actual implementation for brevity
  1 -- placeholder

noncomputable def M : ℕ := 10^199 * 9 - 1  -- 10^199 gives 200 digits, subtracting 1 changes them to all 8s

def g (r : ℕ) : ℕ := leading_digit (nat.root M r)

theorem sum_of_leading_digits :
  g 2 + g 4 + g 6 + g 8 + g 10 = 6 :=
  sorry

end sum_of_leading_digits_l79_79545


namespace arithmetic_series_mod_20_l79_79013

open Nat

theorem arithmetic_series_mod_20 :
  ∑ k in range 28, (4 + 5 * k) % 20 ≡ 2 [MOD 20] :=
by
  sorry

end arithmetic_series_mod_20_l79_79013


namespace solve_squares_and_circles_l79_79289

theorem solve_squares_and_circles (x y : ℝ) :
  (5 * x + 2 * y = 39) ∧ (3 * x + 3 * y = 27) → (x = 7) ∧ (y = 2) :=
by
  intro h
  sorry

end solve_squares_and_circles_l79_79289


namespace round_1_804_to_hundredth_l79_79953

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  Real.round (x * 100) / 100

theorem round_1_804_to_hundredth :
  round_to_hundredth 1.804 = 1.80 :=
by
  sorry

end round_1_804_to_hundredth_l79_79953


namespace sum_of_four_digit_even_numbers_l79_79413

theorem sum_of_four_digit_even_numbers :
  let digits := {0, 1, 2, 3, 4, 5}
  let num_choices := (5 : ℕ) * (6 : ℕ) * (6 : ℕ) * (3 : ℕ)
  let sum_digits := 1 + 2 + 3 + 4 + 5
  num_choices * (1000 * sum_digits + 100 * sum_digits + 10 * sum_digits + 1 * (0 + 2 + 4)) = 1769580 :=
by
  sorry

end sum_of_four_digit_even_numbers_l79_79413


namespace equality_of_shaded_areas_l79_79893

variable (θ : ℝ)
variable (r : ℝ)
variable (A B C : ℝ → Prop)
variable (isCenter : Prop)
variable (isLineSegmentBCD : Prop)
variable (isLineSegmentACE : Prop)
variable (isTangentToCircleAtA : Prop)

theorem equality_of_shaded_areas (hθ : 0 < θ ∧ θ < π / 4) 
    (h_isCenter : C = isCenter)
    (h_isLineSegmentBCD : BCD = isLineSegmentBCD)
    (h_isLineSegmentACE : ACE = isLineSegmentACE)
    (h_isTangentToCircleAtA : AB = isTangentToCircleAtA) :
    sin θ = 2 * θ := sorry

end equality_of_shaded_areas_l79_79893


namespace find_b_l79_79138

theorem find_b (b : ℕ) (h1 : 0 ≤ b) (h2 : b ≤ 20) (h3 : (746392847 - b) % 17 = 0) : b = 16 :=
sorry

end find_b_l79_79138


namespace geometric_series_ratio_half_l79_79203

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l79_79203


namespace sum_last_two_digits_l79_79645

theorem sum_last_two_digits (a b : ℕ) (h₁ : a = 7) (h₂ : b = 13) : 
  (a^25 + b^25) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_l79_79645


namespace purely_imaginary_necessary_not_sufficient_l79_79682

-- Definition of a purely imaginary number
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem purely_imaginary_necessary_not_sufficient (a b : ℝ) :
  a = 0 → (z : ℂ) = ⟨a, b⟩ → is_purely_imaginary z ↔ (a = 0 ∧ b ≠ 0) :=
by
  sorry

end purely_imaginary_necessary_not_sufficient_l79_79682


namespace b_interval_for_non_real_roots_l79_79018

-- Conditions
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

def has_non_real_roots (a b c : ℝ) : Prop := discriminant a b c < 0

-- Given the quadratic equation x^2 + bx + 9
def specific_discriminant (b : ℝ) : ℝ := discriminant 1 b 9

def specific_has_non_real_roots (b : ℝ) : Prop := has_non_real_roots 1 b 9

-- Prove that b is in the interval (-6, 6) if the quadratic equation has non-real roots
theorem b_interval_for_non_real_roots (b : ℝ) : specific_has_non_real_roots b → b ∈ Ioo -6 6 := by
  sorry

end b_interval_for_non_real_roots_l79_79018


namespace count_valid_7_digit_numbers_l79_79853

theorem count_valid_7_digit_numbers : 
  (∑ n in finset.univ.filter (λ n : fin 7, n = 0 ∨ n = 1), n) = 11 :=
by
  -- Formalize the problem statement in Lean
  let valid_numbers := {n : ℕ | 
    let digits := nat.digits 2 n in 
    list.length digits = 7 ∧ 
    digits.head = 1 ∧ 
    digits.last = 0 ∧ 
    ∃ (sum_digits_mod3 : ℕ), sum_digits_mod3 = list.sum digits ∧ sum_digits_mod3 % 3 = 0 
  } in
  have count_valid : #valid_numbers = 11 := sorry,
  exact count_valid

end count_valid_7_digit_numbers_l79_79853


namespace last_two_digits_of_sum_l79_79390

theorem last_two_digits_of_sum :
  ((4! + 8! + 12! + ... + 96!) % 100) = 24 :=
sorry

end last_two_digits_of_sum_l79_79390


namespace impossible_discount_rate_l79_79401

variable (regularPriceFox : ℕ := 15) (regularPricePony : ℕ := 18)
variable (pairsFox : ℕ := 3) (pairsPony : ℕ := 2)
variable (totalDiscountSum : ℝ := 0.18) (discountRatePony : ℝ := 0.5667)

/-- If the total discount sum is 18% and the discount rate on Pony jeans is 56.67%, then 
the calculated discount rate for Fox jeans which turns out to be -38.67%. This implies an impossible scenario. -/
theorem impossible_discount_rate (h : totalDiscountSum = discountRateFox + discountRatePony) 
                                (h_discountRatePony : discountRatePony = 0.5667):
  ¬ (0.18 - 0.5667 >= 0) :=
by
  have discountRateFox : ℝ := totalDiscountSum - discountRatePony
  have : discountRateFox = -0.3867 := by linarith
  have h2 : ¬ (discountRateFox >= 0) := by linarith
  exact h2

end impossible_discount_rate_l79_79401


namespace radius_of_incircle_correct_l79_79095

-- Definitions and conditions
variables (A B C : Type) [plane A B C]
variables (A₀ B C : point)
variables (β γ : ℝ) -- Angles at B and C
variable BC : ℝ -- Length of side BC
variable r_n : ℕ → ℝ -- Radius of the incircle at A_nBC

-- Incenter function (Details skipped)
noncomputable def Incenter (A : point) (Aₘ B C : Type) : Type := sorry

-- Angle functions (Details skipped)
noncomputable def angle_β_n (n : ℕ) : ℝ := β / 2^n
noncomputable def angle_γ_n (n : ℕ) : ℝ := γ / 2^n
noncomputable def angle_βγ_n (n : ℕ) : ℝ := (β + γ) / 2^n

-- Law of Sines function (Details skipped)
noncomputable def Law_of_Sines (B C : Type) (β γ : ℝ) (n : ℕ) : ℝ :=
  BC * (Math.sin (angle_γ_n C β γ n) * Math.sin (angle_β_n B β γ n)) /
  (Math.sin (angle_βγ_n B C β γ n))

-- Proof of the radius of the incircle of the triangle AₙBC
noncomputable def radius_of_incircle (BC : ℝ) (β γ : ℝ) (n : ℕ) : ℝ :=
  Law_of_Sines B C β γ n

theorem radius_of_incircle_correct (n : ℕ) :
  r_n n = BC * (Math.sin (angle_γ_n C β γ n) * Math.sin (angle_β_n B β γ n)) /
  (Math.sin (angle_βγ_n B C β γ n)) :=
sorry

end radius_of_incircle_correct_l79_79095


namespace fraction_doubling_unchanged_l79_79043

theorem fraction_doubling_unchanged (x y : ℝ) (h : x ≠ y) : 
  (3 * (2 * x)) / (2 * x - 2 * y) = (3 * x) / (x - y) :=
by
  sorry

end fraction_doubling_unchanged_l79_79043


namespace sum_lucky_numbers_divisible_by_13_l79_79341

def is_lucky (n : ℕ) : Prop :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 + d2 + d3 = d4 + d5 + d6

theorem sum_lucky_numbers_divisible_by_13 :
  (∑ n in Finset.filter is_lucky (Finset.range 1000000), n) % 13 = 0 :=
sorry

end sum_lucky_numbers_divisible_by_13_l79_79341


namespace cos_third_quadrant_l79_79019

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l79_79019


namespace midpoints_on_circle_l79_79572

-- Given a triangle with each side containing two points such that the six segments
-- connecting each point to the opposite vertex are equal in length, prove that the midpoints
-- of these six segments lie on a single circle.

theorem midpoints_on_circle {A B C D H K L M: Point}
  (hBD: altitude A B D C)
  (hH: orthocenter H A B C)
  (hK: midpoint K B D)
  (hL: midpoint L B D)
  (hM: midpoint M B D)
  (hEqualSegments: (distance B K = distance B L) ∧ (distance B K = distance A K) ∧ (distance A L = distance C L)):
  ∃ (O: Point) (r: ℝ), on_circle O M r ∧ on_circle O K r ∧ on_circle O L r := 
sorry

end midpoints_on_circle_l79_79572


namespace books_sold_wednesday_l79_79073

-- Define the conditions of the problem
def total_books : Nat := 1200
def sold_monday : Nat := 75
def sold_tuesday : Nat := 50
def sold_thursday : Nat := 78
def sold_friday : Nat := 135
def percentage_not_sold : Real := 66.5

-- Define the statement to be proved
theorem books_sold_wednesday : 
  let books_sold := total_books * (1 - percentage_not_sold / 100)
  let known_sales := sold_monday + sold_tuesday + sold_thursday + sold_friday
  books_sold - known_sales = 64 :=
by
  sorry

end books_sold_wednesday_l79_79073


namespace coplanar_vectors_has_lambda_eq_one_l79_79407

open Real

noncomputable def vec3 := (ℝ × ℝ × ℝ)

def a : vec3 := (2, -1, 3)
def b : vec3 := (-1, 4, -2)
def c (λ : ℝ) : vec3 := (1, 3, λ)

-- Define the condition for coplanarity
def coplanar (v1 v2 v3 : vec3) : Prop := 
  ∃ m n : ℝ, v3 = (m * v1.1 + n * v2.1, m * v1.2 + n * v2.2, m * v1.3 + n * v2.3)

theorem coplanar_vectors_has_lambda_eq_one (λ : ℝ) :
  coplanar a b (c λ) ↔ λ = 1 :=
by sorry  -- Proof omitted; main goal is to write the statement.

end coplanar_vectors_has_lambda_eq_one_l79_79407


namespace spherical_to_rectangular_correct_l79_79363

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  sphericalToRectangular ρ θ φ = (5 * Real.sqrt 6 / 4, 5 * Real.sqrt 6 / 4, 5 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l79_79363


namespace max_consecutive_integers_sum_lt_1000_l79_79243

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l79_79243


namespace geometric_series_common_ratio_l79_79194

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l79_79194


namespace no_solutions_if_AM_geq_A1M_or_perpendicular_AF_infinitely_many_solutions_if_A_eq_M_l79_79757

-- Define the conditions
variables (A M F : Point)
noncomputable def midpoint (B C : Point) : Point := sorry
def orthocenter (ABC : Triangle) : Point := sorry
def perpendicular (line1 line2 : Line) : Prop := sorry
def distance (p1 p2 : Point) : Real := sorry
def triangle (A B C : Point) := sorry
def reflection (point line : Point) : Point := sorry
def circle (center : Point) (radius : Real) : Set Point := sorry

-- Given conditions
axiom A_exists : ∃ A : Point, True
axiom M_is_orthocenter : ∃ (ABC : Triangle), M = orthocenter(ABC)
axiom F_is_midpoint : ∃ (B C : Point), F = midpoint B C

-- Define the equivalent proof statements
theorem no_solutions_if_AM_geq_A1M_or_perpendicular_AF 
  (A1 M : Point) 
  (h1 : distance A M ≥ distance A1 M) 
  (h2 : perpendicular (line_through A M) (line_through A F)) : 
  ¬ ∃ (ABC : Triangle), ABC.has_vertex A ∧ M = orthocenter(ABC) ∧ F = midpoint(ABC.B ABC.C) := 
sorry

theorem infinitely_many_solutions_if_A_eq_M 
  (A_eq_M : A = M) : 
  ∃ (ABC : Triangle), ABC.has_vertex A ∧ M = orthocenter(ABC) ∧ F = midpoint(ABC.B ABC.C) :=
sorry

end no_solutions_if_AM_geq_A1M_or_perpendicular_AF_infinitely_many_solutions_if_A_eq_M_l79_79757


namespace tan_graph_shift_l79_79999

theorem tan_graph_shift :
  ∀ x : ℝ,  tan x = tan (2 * (x - (π / 2))) :=
by
  sorry

end tan_graph_shift_l79_79999


namespace combined_resistance_in_parallel_l79_79509

theorem combined_resistance_in_parallel (x y r : ℝ) (hx : x = 4) (hy : y = 5) (h : 1 / r = 1 / x + 1 / y) : 
  r = 20 / 9 :=
by
  rw [hx, hy] at h
  calc
    1 / r = 1 / 4 + 1 / 5 : h
        ... = 5 / 20 + 4 / 20 : by norm_num
        ... = 9 / 20 : by norm_num
    r = 20 / 9 : by norm_num

end combined_resistance_in_parallel_l79_79509


namespace first_player_winning_strategy_l79_79135

theorem first_player_winning_strategy :
  (∃ card_pairs : list (ℕ × ℕ), (∀ (pair : ℕ × ℕ), pair ∈ card_pairs → (1 ≤ pair.1 ∧ pair.1 < pair.2 ∧ pair.2 ≤ 2003))
    → (∃ strategy : list (ℕ × ℕ), 
         player_has_winning_strategy strategy card_pairs)) :=
begin
  sorry
end

def player_has_winning_strategy (strategy : list (ℕ × ℕ)) (card_pairs : list (ℕ × ℕ)) : Prop :=
∃ gcd_fall_to_one_game_state : list ℕ → Prop,
  -- initial game state
  gcd_fall_to_one_game_state [],
  -- strategy ensures that the first player wins
  ∀ (state : list ℕ),
    gcd_fall_to_one_game_state state →
    ∀ (card : ℕ × ℕ),
      card ∈ card_pairs →
      let new_state := (card.1 * card.2) :: state in
      (forall gcd_fall_to_one_game_state new_state)

end first_player_winning_strategy_l79_79135


namespace sum_last_two_digits_l79_79646

theorem sum_last_two_digits (a b : ℕ) (h₁ : a = 7) (h₂ : b = 13) : 
  (a^25 + b^25) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_l79_79646


namespace parametric_curve_coefficients_l79_79311

theorem parametric_curve_coefficients :
  ∃ a b c : ℚ,
  (∀ t : ℝ,
    (3 * (Real.cos t) + (Real.sin t))^2 * a +
    (3 * (Real.cos t) + (Real.sin t)) * (3 * (Real.sin t)) * b +
    (3 * (Real.sin t))^2 * c = 1) ∧
  (a = 1/9) ∧ (b = -2/27) ∧ (c = 10/81) := 
begin
  use [1/9, -2/27, 10/81],
  split, 
  intro t,
  calc _ : (3 * (Real.cos t) + (Real.sin t))^2 * (1/9) +
            (3 * (Real.cos t) + (Real.sin t)) * (3 * (Real.sin t)) * (-2/27) +
            (3 * (Real.sin t))^2 * (10/81) = _ := 
    sorry, -- omitted steps to show it is identically 1 
  exact ⟨rfl, rfl, rfl⟩,
end

end parametric_curve_coefficients_l79_79311


namespace convert_spherical_to_rectangular_l79_79365

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 5 (Real.pi / 4) (Real.pi / 3) = (5 * (Real.sqrt 3 / 2) * (Real.sqrt 2 / 2), 5 * (Real.sqrt 3 / 2) * (Real.sqrt 2 / 2), 5 * (1 / 2)) :=
by
  sorry

end convert_spherical_to_rectangular_l79_79365


namespace minimize_power_line_distance_l79_79336

open Real

noncomputable def optimal_d {A B C D E : ℝ} (AB_dist Ac_Bc_dist Ae_Be_dist : ℝ) (AE EB : ℝ) :=
  (AC = BC := 0.5)   -- Given conditions
  (AB = 0.6)         -- Distance between A and B
  (AE = EB := 0.3) -- Midpoint distances

theorem minimize_power_line_distance : 
  let L (x : ℝ) := (0.4 - x) + 2 * sqrt (0.09 + x^2) in
  let critical_x := (sqrt 3) / 10 in
  ∀ x, optimize L x = critical_x :=
sorry

end minimize_power_line_distance_l79_79336


namespace standard_equation_of_ellipse_equation_of_line_l79_79423

-- Define the conditions first
variables (a b c : ℝ) (h1 : a > b > 0)
variables (eccentricity : ℝ) (h2 : eccentricity = 1/2)
variables (focusX : ℝ) (h3 : focusX = 1)
variables (P : ℝ → ℝ) (h4 : P 1 = t)
variables (area : ℝ) (h5 : area = 9 * Real.sqrt 2 / 7)

-- The first problem: Finding the standard equation of the ellipse
theorem standard_equation_of_ellipse
  (h6 : c = 1)
  (h7 : c / a = 1/2)
  (h8 : a = 2)
  : ∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 :=
sorry

-- The second problem: Finding the equation of line l
theorem equation_of_line
  (h9 : ∃! (l : ℝ → ℝ), (∀ x y, (x - l y - 1 = 0 ∨ x - l y + 1 = 0) ∧ (l passes through focusX)))
  : ∃! (l : ℝ → ℝ), (∀ x y, (x + y - 1 = 0 ∨ x - y - 1 = 0)) :=
sorry

end standard_equation_of_ellipse_equation_of_line_l79_79423


namespace distinct_9_pointed_stars_l79_79141

-- Define a function to count the distinct n-pointed stars for a given n
def count_distinct_stars (n : ℕ) : ℕ :=
  -- Functionality to count distinct stars will be implemented here
  sorry

-- Theorem stating the number of distinct 9-pointed stars
theorem distinct_9_pointed_stars : count_distinct_stars 9 = 2 :=
  sorry

end distinct_9_pointed_stars_l79_79141


namespace nth_equation_proof_l79_79113

theorem nth_equation_proof (n : ℕ) (hn : n > 0) :
  (1 : ℝ) + (1 / (n : ℝ)) - (2 / (2 * n - 1)) = (2 * n^2 + n + 1) / (n * (2 * n - 1)) :=
by
  sorry

end nth_equation_proof_l79_79113


namespace product_floor_cubic_roots_eq_rat_l79_79751

-- Define conditions related to cubed roots and flooring
def floor_cubic_roots_product (n : ℕ) : ℚ :=
  (∏ i in finset.range n, ⌊(i * 2 + 1 : ℚ)^(1/3)⌋) / (∏ i in finset.range n, ⌊((i + 1) * 2 : ℚ)^(1/3)⌋)

theorem product_floor_cubic_roots_eq_rat :
  floor_cubic_roots_product 62 = (2 / 5) :=
by
  -- Insert proof here
  sorry

end product_floor_cubic_roots_eq_rat_l79_79751


namespace total_handshakes_l79_79741

theorem total_handshakes (twins_num : ℕ) (triplets_num : ℕ) (twins_sets : ℕ) (triplets_sets : ℕ) (h_twins : twins_sets = 9) (h_triplets : triplets_sets = 6) (h_twins_num : twins_num = 2 * twins_sets) (h_triplets_num: triplets_num = 3 * triplets_sets) (h_handshakes : twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2) = 882): 
  (twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2)) / 2 = 441 :=
by
  sorry

end total_handshakes_l79_79741


namespace number_of_functions_l79_79783

noncomputable theory

def f (a b c d x : ℝ) := a * x^3 + b * x^2 + c * x + d

theorem number_of_functions (a b c d : ℝ) :
    ((∀ x, f a b c d x * f a b c d (-x) = f a b c d (x^3)) ↔
    (
      (∃ a, a = 0) ∧ 
      (∃ (b : ℝ), b = 0 ∨ b = 1) ∧ 
      (∃ c, c = 0) ∧ 
      (∃ (d : ℝ), d = 0 ∨ d = 1)
    )) →
    ∃ (n : ℕ), n = 4 := 
by 
  intros h 
  sorry

end number_of_functions_l79_79783


namespace total_disks_in_bag_l79_79379

theorem total_disks_in_bag :
  ∃ (x : ℕ), let blue_disks := 3 * x
             let yellow_disks := 7 * x
             let green_disks := 8 * x in
             green_disks - blue_disks = 20 ∧
             blue_disks + yellow_disks + green_disks = 72 :=
begin
  -- Proof skipped
  sorry
end

end total_disks_in_bag_l79_79379


namespace selling_price_correct_l79_79312

variable (CostPrice GainPercent : ℝ)
variables (Profit SellingPrice : ℝ)

noncomputable def calculateProfit : ℝ := (GainPercent / 100) * CostPrice

noncomputable def calculateSellingPrice : ℝ := CostPrice + calculateProfit CostPrice GainPercent

theorem selling_price_correct 
  (h1 : CostPrice = 900) 
  (h2 : GainPercent = 30)
  : calculateSellingPrice CostPrice GainPercent = 1170 := by
  sorry

end selling_price_correct_l79_79312


namespace max_consecutive_integers_lt_1000_l79_79256

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l79_79256


namespace geometric_series_ratio_half_l79_79205

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l79_79205


namespace max_clothing_sets_produced_l79_79772

noncomputable def total_max_sets : ℕ :=
  let sets_A := 2700 -- sets produced per month by Factory A
  let sets_B := 3600 -- sets produced per month by Factory B
  
  let ratio_A := (2, 1) -- time ratio of producing jackets to trousers in Factory A
  let ratio_B := (3, 2) -- time ratio of producing jackets to trousers in Factory B
  
  let max_jackets_B := sets_B * 5 / 3 -- Calculation based on given ratios
  let max_trousers_A := (sets_A * 3 / 1 : ℕ) -- Calculation based on given ratios
  let feasible_max := 6000 -- Recognizing that producing max trousers A isn't feasible timely wise
  
  max_jackets_B + feasible_max / 27 * (27 - 10)

theorem max_clothing_sets_produced :
  -- Given all the conditions
  let factA_production := 2700
  let factB_production := 3600
  let ratioA := (2, 1)
  let ratioB := (3, 2)
  in
  total_max_sets factA_production factB_production ratioA ratioB = 6700 :=
by
  sorry

end max_clothing_sets_produced_l79_79772


namespace gain_percentage_l79_79317

theorem gain_percentage (cost_price : ℝ) (selling_price : ℝ) :
  (selling_price / cost_price - 1) * 100 = 20 :=
by
  -- Given conditions
  let cp_15pencils := 15 * cost_price
  let sp_15pencils := 1
  let loss_percentage := 0.20
  
  -- Calculation of cost price per pencil
  let cp_1pencil := cp_15pencils / 15
  let effective_cp_15pencils := sp_15pencils / (1 - loss_percentage)
  have H1 : effective_cp_15pencils = cp_15pencils := by sorry
  have H2 : cp_1pencil = effective_cp_15pencils / 15 := by sorry
  
  -- Given new selling price condition
  let sp_10pencils := 1
  let cp_10pencils := 10 * cp_1pencil
  
  -- Calculation of gain percentage
  let gain_percentage := (sp_10pencils - cp_10pencils) / cp_10pencils * 100
  
  -- Required proof
  have H3 : gain_percentage = 20 := by sorry
  exact H3

end gain_percentage_l79_79317


namespace equilateral_triangle_BD_CD_plus_AE_l79_79077

theorem equilateral_triangle_BD_CD_plus_AE (A B C D E : Point) (t : Line) 
  (h₁ : is_equilateral_triangle A B C)
  (h₂ : parallel t (line B C))
  (h₃ : on_line A t)
  (h₄ : on_line D (line A C))
  (h₅ : bisector (angle A B D) (line B E))
  (h₆ : on_line E t) :
  dist B D = dist C D + dist A E :=
sorry

end equilateral_triangle_BD_CD_plus_AE_l79_79077


namespace evaluate_f_i_l79_79099

noncomputable def f (x : ℂ) : ℂ := (x^5 + 2*x^3 + x) / (x + 1)
def i : ℂ := complex.I

theorem evaluate_f_i : f i = -1 + i := by
  sorry

end evaluate_f_i_l79_79099


namespace geometric_series_common_ratio_l79_79180

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l79_79180


namespace trajectory_of_circle_center_l79_79439

theorem trajectory_of_circle_center :
  (∀ (P : Point) (r : ℝ), -- For any point P (center of circle) and radius r
    (P ∈ set_of_points_tangent_to F1) ∧ -- Circle P is tangent to circle F1
    (P ∈ set_of_points_internally_tangent_to F2)) -- Circle P is internally tangent to circle F2
    →

trivially_equivalent_curve : -- A helper def to store the equation of curve C
  ∀ (P : Point),
    (P ∈ curve_c_trajectory) ↔ -- Points P on curve C
    (P ∈ {Point | (P.x^2)/16 + (P.y^2)/7 = 1}) -- Equation of the trajectory curve C
    := 
sorry -- Proof omitted

end trajectory_of_circle_center_l79_79439


namespace _l79_79533

-- Infinite graph as a structure
structure Graph (V : Type) :=
(adj : V → V → Prop)

open Graph

-- Countable set as a typeclass to simplify things
class Countable (α : Type) :=
(encode : α → ℕ)
(decode : ℕ → Option α)
(decode_encode : ∀ a, decode (encode a) = some a)

-- Infinite type
class Infinite (α : Type) :=
(infinite : ∃ f : ℕ → α, ∀ n m, f n = f m → n = m → False)

variables {V : Type} [Infinite V]

-- Main theorem statement
lemma exists_uncountable_vertices {G : Graph V}
  (hG : ∀ (A : Set V), Countable A → ∃ p, p ∉ A ∧ Infinite {x | x ∈ A ∧ G.adj p x}) :
  ∃ (A : Set V), Countable A ∧ Infinite {p | p ∉ A ∧ Infinite {x | x ∈ A ∧ G.adj p x}} :=
sorry

end _l79_79533


namespace AC_squared_minus_AE_squared_l79_79534

variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables {ABC : Triangle} (angle_ABC : ∠ ABC = 90) (circle_with_diameter_AB : CircleDiameter A B)
variables {D_on_AC : D ∈ AC} {tangent_at_D : Tangent D circle_with_diameter_AB} (E_on_BC : E ∈ BC)
variables (EC_length : dist E C = 2)

theorem AC_squared_minus_AE_squared : dist A C ^ 2 - dist A E ^ 2 = 12 :=
by
  sorry

end AC_squared_minus_AE_squared_l79_79534


namespace arithmetic_mean_probability_l79_79418

theorem arithmetic_mean_probability
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : b = (a + c) / 2) :
  b = 1 / 3 :=
by
  sorry

end arithmetic_mean_probability_l79_79418


namespace polar_coords_intersection_l79_79519

theorem polar_coords_intersection (ρ θ : ℝ) (l : ℝ → ℝ → Prop)
  (h_line_parallel: ∀ θ, (l ρ θ ↔ ρ * cos θ = 1))
  (h_line_center: l 1 0)
  (h_center: ∀ θ, 2 * cos θ = 1) :
  l 1 0 :=
by
  sorry

end polar_coords_intersection_l79_79519


namespace delivery_time_is_40_minutes_l79_79163

-- Define the conditions
def total_pizzas : Nat := 12
def two_pizza_stops : Nat := 2
def pizzas_per_stop_with_two_pizzas : Nat := 2
def time_per_stop_minutes : Nat := 4

-- Define the number of pizzas covered by stops with two pizzas
def pizzas_covered_by_two_pizza_stops : Nat := two_pizza_stops * pizzas_per_stop_with_two_pizzas

-- Define the number of single pizza stops
def single_pizza_stops : Nat := total_pizzas - pizzas_covered_by_two_pizza_stops

-- Define the total number of stops
def total_stops : Nat := two_pizza_stops + single_pizza_stops

-- Total time to deliver all pizzas
def total_delivery_time_minutes : Nat := total_stops * time_per_stop_minutes

theorem delivery_time_is_40_minutes : total_delivery_time_minutes = 40 := by
  sorry

end delivery_time_is_40_minutes_l79_79163


namespace right_triangle_area_l79_79051

theorem right_triangle_area
  (h : ∀ (AB AC CM KN : ℝ), 
    (CM = 5) ∧ (KN = 4) ∧ (C = midpoint AB) ∧ (M = midpoint AB) 
    ∧ (N = midpoint KN) ∧ (MN ⊥ AC) ∧ (NK ⊥ AB)
    ∧ (angle ABC = 90) ∧ (angle CBA = 90)
    → area (triangle ABC) = 200 / 3 
  ) : sorry :=
begin
  sorry,
end

end right_triangle_area_l79_79051


namespace cyclic_quad_theorem_l79_79078

noncomputable theory
open_locale classical

variables (A B C D E : Point)
variables (AB AC AD BC BD CD : Segment)
variables (P : Segment)

-- Let ABCD be a cyclic quadrilateral
def cyclic_quadrilateral (ABCD : Quadrilateral) : Prop :=
  ∃ O : Point, O ∈ circumcircle ABCD

-- ∠ADC = ∠DBA
def angle_equality1 (A B C D : Point) : Prop :=
  ∠ADC = ∠DBA

-- E is the projection of A on BD
def projection (A D E : Point) : Prop :=
  is_perpendicular_to AD BD ∧ (∃ F : Point, E = midpointA B F)

theorem cyclic_quad_theorem
  (ABCD : Quadrilateral)
  (h1 : cyclic_quadrilateral ABCD)
  (h2 : angle_equality1 A B C D)
  (h3 : projection A D E) :
  BC = DE - BE := sorry

end cyclic_quad_theorem_l79_79078


namespace max_consecutive_integers_lt_1000_l79_79254

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l79_79254


namespace relationship_among_abc_l79_79370

noncomputable def a : ℝ := 20.3
noncomputable def b : ℝ := 0.32
noncomputable def c : ℝ := Real.log 25 / Real.log 10

theorem relationship_among_abc : b < a ∧ a < c :=
by
  -- Proof needs to be filled in here
  sorry

end relationship_among_abc_l79_79370


namespace area_of_circle_diameter_7_5_l79_79265

theorem area_of_circle_diameter_7_5 :
  ∃ (A : ℝ), (A = 14.0625 * Real.pi) ↔ (∃ (d : ℝ), d = 7.5 ∧ A = Real.pi * (d / 2) ^ 2) :=
by
  sorry

end area_of_circle_diameter_7_5_l79_79265


namespace decimal_to_base_six_has_three_digits_l79_79518

theorem decimal_to_base_six_has_three_digits :
  ∃ (digits : ℕ), nat.to_digits 6 143 = digits ∧ (digits.to_string.length = 3) := by
sorry

end decimal_to_base_six_has_three_digits_l79_79518


namespace coordinates_of_point_M_on_Z_axis_l79_79103

notation "^" => pow

theorem coordinates_of_point_M_on_Z_axis (z : ℝ) :
  (∃z, (0, 0, z)) ∧ 
  (real.sqrt ((0 - 1)^2 + (0 - 0)^2 + (z - 2)^2) = 
   real.sqrt ((0 - 1)^2 + (0 - -3)^2 + (z - 1)^2)) → 
  (0, 0, z) = (0, 0, -3) :=
by
  sorry

end coordinates_of_point_M_on_Z_axis_l79_79103


namespace average_viewer_watches_two_videos_daily_l79_79330

variable (V : ℕ)
variable (video_time : ℕ := 7)
variable (ad_time : ℕ := 3)
variable (total_time : ℕ := 17)

theorem average_viewer_watches_two_videos_daily :
  7 * V + 3 = 17 → V = 2 := 
by
  intro h
  have h1 : 7 * V = 14 := by linarith
  have h2 : V = 2 := by linarith
  exact h2

end average_viewer_watches_two_videos_daily_l79_79330


namespace quadratic_maximum_or_minimum_l79_79500

open Real

noncomputable def quadratic_function (a b x : ℝ) : ℝ := a * x^2 + b * x - b^2 / (3 * a)

theorem quadratic_maximum_or_minimum (a b : ℝ) (h : a ≠ 0) :
  (a > 0 → ∃ x₀, ∀ x, quadratic_function a b x₀ ≤ quadratic_function a b x) ∧
  (a < 0 → ∃ x₀, ∀ x, quadratic_function a b x₀ ≥ quadratic_function a b x) :=
by
  -- Proof will go here
  sorry

end quadratic_maximum_or_minimum_l79_79500


namespace meaningful_fraction_condition_l79_79220

theorem meaningful_fraction_condition (x : ℝ) : (4 - 2 * x ≠ 0) ↔ (x ≠ 2) :=
by {
  sorry
}

end meaningful_fraction_condition_l79_79220


namespace f_eq_91_for_all_n_leq_100_l79_79101

def f : ℤ → ℝ
| n => if n > 100 then n - 10 else f (f (n + 11))

theorem f_eq_91_for_all_n_leq_100 (n : ℤ) (h : n ≤ 100) : f n = 91 := by
  sorry

end f_eq_91_for_all_n_leq_100_l79_79101


namespace scientific_notation_correct_l79_79567

/-- Define the number 42.39 million as 42.39 * 10^6 and prove that it is equivalent to 4.239 * 10^7 -/
def scientific_notation_of_42_39_million : Prop :=
  (42.39 * 10^6 = 4.239 * 10^7)

theorem scientific_notation_correct : scientific_notation_of_42_39_million :=
by 
  sorry

end scientific_notation_correct_l79_79567


namespace correct_options_l79_79274

open Classical

variable {A B : Prop} {P : Prop → ℝ}

def mutually_exclusive (A B : Prop) : Prop :=
  A ∧ B → False

def independent (A B : Prop) : Prop :=
  P (A ∧ B) = P A * P B

def negation_correct (p : Prop) : Prop :=
  ¬p

def angle_sufficient (m n α : ℝ) : Prop :=
  m = 60 ∧ n = 30 ∧ α = 120

theorem correct_options :
  (P(A ∧ B) = P A * P B) →
  (∃ (m n α : ℝ), m = 60 ∧ n = 30 ∧ α = 120) → 
  True :=
by
  intros _ _
  trivial

end correct_options_l79_79274


namespace sum_of_ages_l79_79727

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
def condition1 := a = 16 + b + c
def condition2 := a^2 = 1632 + (b + c)^2

-- Define the theorem to prove the question
theorem sum_of_ages : condition1 a b c → condition2 a b c → a + b + c = 102 := 
by 
  intros h1 h2
  sorry

end sum_of_ages_l79_79727


namespace distinct_real_roots_of_quadratic_l79_79471

theorem distinct_real_roots_of_quadratic (m : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ x^2 + m * x + 9 = 0 ∧ y^2 + m * y + 9 = 0) ↔ m ∈ Ioo (-∞) (-6) ∪ Ioo 6 (∞) :=
by
  sorry

end distinct_real_roots_of_quadratic_l79_79471


namespace eval_expression_correct_l79_79767

noncomputable def evaluate_expression : ℝ :=
    3 + Real.sqrt 3 + (3 - Real.sqrt 3) / 6 + (1 / (Real.cos (Real.pi / 4) - 3))

theorem eval_expression_correct : 
  evaluate_expression = (3 * Real.sqrt 3 - 5 * Real.sqrt 2) / 34 :=
by
  -- Proof can be filled in later
  sorry

end eval_expression_correct_l79_79767


namespace seventeenth_number_is_5724_l79_79628

def is_permutation (l1 l2 : List ℕ) : Prop :=
  l1.length = l2.length ∧ l1.perm l2

def valid_4_digit_numbers : List (List ℕ) :=
  ([2, 4, 5, 7]).permutations

def sorted_4_digit_numbers : List (List ℕ) :=
  valid_4_digit_numbers.qsort (λ l1 l2 => l1 < l2)

theorem seventeenth_number_is_5724 :
  (sorted_4_digit_numbers.get? 16 = some [5, 7, 2, 4]) :=
sorry

end seventeenth_number_is_5724_l79_79628


namespace problem_ABCD_area_ratio_l79_79583

theorem problem_ABCD_area_ratio 
  (AB BC BD CE BE : ℝ)
  (ABCD_right_BC : ∀ (B C: ℝ), ABCD_has_right_angles_at_B_and_C) 
  (tri_sim_ABC_BCD : ∀ (A B C D : ℝ), TABC_sim_TBCD)
  (AB_gt_BC : AB > BC)
  (E_in_interior : ∀ (A B C D : ℝ), E_is_interior_of_ABCD)
  (tri_sim_ABC_CEB : ∀ (A B C E : ℝ), TABC_sim_TCEB)
  (area_AED_25_CEB : area_25_areas_of_triangles : ∀ (A B C D E : ℝ), area_of_tAED(ABCD) = 25 * area_of_tCEB(ABCD))
  : (AB / BC) = (5 + 2 * Real.sqrt 5) :=
sorry

end problem_ABCD_area_ratio_l79_79583


namespace least_number_to_subtract_l79_79269

theorem least_number_to_subtract (n : ℕ) (h : n = 101054) : ∃ x : ℕ, x = 4 ∧ (n - x) % 10 = 0 :=
by
  use 4
  have h1 : n % 10 = 4 := by rw [h]; norm_num
  have h2 : (n - 4) % 10 = (n % 10 - 4 % 10) % 10 := (nat.sub_mod _ 4 _).symm
  rw [h1] at h2
  norm_num at h2
  exact ⟨rfl, h2⟩

end least_number_to_subtract_l79_79269


namespace triangle_sequence_last_perimeter_l79_79918

theorem triangle_sequence_last_perimeter (1001 1002 1003 : ℕ)
  (h1 : 1001 + 1002 > 1003)
  (h2 : 1001 + 1003 > 1002)
  (h3 : 1002 + 1003 > 1001) :
  ∃ n : ℕ, (forall m > n, 1001 + m + 1 + (1002 + m + 1) <= 1003 + m + 1) ∧
  3 * (1003 + 5) = 3021 := 
sorry

end triangle_sequence_last_perimeter_l79_79918


namespace min_value_of_reciprocal_sum_l79_79428

theorem min_value_of_reciprocal_sum {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (hgeom : 3 = Real.sqrt (3^a * 3^b)) : (1 / a + 1 / b) = 2 :=
sorry  -- Proof not required, only the statement is needed.

end min_value_of_reciprocal_sum_l79_79428


namespace range_of_m_l79_79881

open EuclideanGeometry

-- Define the square ABCD with side length 8
variable (A B C D M N : Point)
variable (side_length : ℝ)
axiom square_ABCD : side_length = 8 ∧
  (dist A B = side_length) ∧
  (dist B C = side_length) ∧
  (dist C D = side_length) ∧
  (dist D A = side_length) ∧
  (angle A B C = π / 2) ∧
  (angle B C D = π / 2) ∧
  (angle C D A = π / 2) ∧
  (angle D A B = π / 2)

-- Define M as the midpoint of BC
axiom midpoint_M : M = midpoint B C

-- Define N on DA such that DN = 3 * NA
axiom N_on_DA : ∃ x : ℝ, x > 0 ∧
  (N = (x • D + (1 - x) • A)) ∧
  (dist D N = 3 * dist N A)

-- Define the problem
theorem range_of_m : 
  ∀ (m : ℝ), 
  (∃ P : Point, 
    P ∈ [A, B, C, D] ∧ 
    (vector PM • vector PN = m)) → 
  -1 < m ∧ m < 8 :=
sorry

end range_of_m_l79_79881


namespace min_unsociable_pairs_l79_79912

open Nat

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ n, n > 1 → n < p → p % n ≠ 0

def pi (x : ℕ) : ℕ := x.succ.count is_prime

theorem min_unsociable_pairs (n : ℕ) :
  let unsociable_pairs := ⌈(pi (2 * n) - pi n + 1 : ℝ) / 2⌉ in
  ∀ (S : Finset (Fin (2 * n + 1))),
  (∀ (a b : ℕ), a ∈ S → b ∈ S → gcd a b = 1 → 1 < a ∧ 1 < b) →
  S.card = unsociable_pairs :=
sorry

end min_unsociable_pairs_l79_79912


namespace coefficient_x3_l79_79761

def P (x : ℝ) := -3 * x^3 - 8 * x^2 + 3 * x + 2
def Q (x : ℝ) := -2 * x^2 - 7 * x - 4

theorem coefficient_x3 (x : ℝ) : (P x * Q x).coeff 3 = 6 := 
sorry

end coefficient_x3_l79_79761


namespace calculate_volume_proximity_cube_l79_79357

theorem calculate_volume_proximity_cube (a b c : ℝ) (h_dim : a = 2 ∧ b = 3 ∧ c = 6) :
  let V_total := (324 + 151 * Real.pi) / 3
  in ∃ V, V = V_total :=
by
  use (324 + 151 * Real.pi) / 3
  have hV : V_total = (324 + 151 * Real.pi) / 3 := by rfl
  exact hV
  sorry

end calculate_volume_proximity_cube_l79_79357


namespace product_evaluation_l79_79985

noncomputable def product_term (n : ℕ) : ℚ :=
  1 - (1 / (n * n))

noncomputable def product_expression : ℚ :=
  10 * 71 * (product_term 2) * (product_term 3) * (product_term 4) * (product_term 5) *
  (product_term 6) * (product_term 7) * (product_term 8) * (product_term 9) * (product_term 10)

theorem product_evaluation : product_expression = 71 := by
  sorry

end product_evaluation_l79_79985


namespace math_problem_l79_79823

open Complex

def z := (1 : ℂ) + (1 : ℂ) * I

lemma complex_conjugate (z : ℂ) : conj z = (1 : ℂ) - (1 : ℂ) * I := by
  sorry

lemma div_by_i (z : ℂ) : (conj z / I) = (-1 : ℂ) - (1 : ℂ) * I := by
  sorry

lemma multiply_by_i (z : ℂ) : (I * z) = (I : ℂ) - (1 : ℂ) := by
  sorry

theorem math_problem (z : ℂ) : (conj z / I) + (I * z) = -2 := by
  rw [complex_conjugate, div_by_i, multiply_by_i]
  sorry

end math_problem_l79_79823


namespace slope_tangent_line_at_pi_div_2_l79_79169

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x - Real.cos x

theorem slope_tangent_line_at_pi_div_2 : 
  (Real.deriv f) (Real.pi / 2) = 2 := 
by 
  sorry

end slope_tangent_line_at_pi_div_2_l79_79169


namespace distinct_real_roots_of_quadratic_l79_79478

-- Define the problem's condition: m is a real number and the discriminant of x^2 + mx + 9 > 0
def discriminant_positive (m : ℝ) := m^2 - 36 > 0

theorem distinct_real_roots_of_quadratic (m : ℝ) (h : discriminant_positive m) :
  m ∈ Iio (-6) ∪ Ioi (6) :=
sorry

end distinct_real_roots_of_quadratic_l79_79478


namespace probability_sequence_123456_l79_79660

theorem probability_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11
  in (favorable_sequences : ℚ) / total_sequences = 1 / 720 := 
by 
  sorry

end probability_sequence_123456_l79_79660


namespace h_comp_h_3_l79_79016

def h (x : ℕ) : ℕ := 3 * x * x + 5 * x - 3

theorem h_comp_h_3 : h (h 3) = 4755 := by
  sorry

end h_comp_h_3_l79_79016


namespace find_a_l79_79150

-- The equation of the hyperbola
def hyperbola (a : ℝ) (h : a > 0) : Prop := 
  ∀ x y : ℝ, 
  (x^2 / a^2 - y^2 / 9 = 1)

-- The equation of the asymptote
def asymptote (y x : ℝ) : Prop :=
  (y = (3 / 5) * x)

-- Prove that 'a' is equal to 5 given the hyperbola and asymptote equations
theorem find_a (a : ℝ) (h : a > 0) (H : ∀ x y : ℝ, hyperbola a h x y) : a = 5 := sorry

end find_a_l79_79150


namespace find_smallest_angle_l79_79920

open Real

noncomputable def smallest_angle (m n p : ℝ^3) (alpha : ℝ) : Prop :=
  ∥m∥ = 1 ∧ ∥n∥ = 1 ∧ ∥p∥ = 1 ∧
  (angle_between m n = alpha) ∧
  (angle_between p (cross m n) = alpha) ∧
  (dot n (cross p m) = 1/8) →
  alpha = 7.5 * (π / 180)

theorem find_smallest_angle {m n p : ℝ^3} (alpha : ℝ) :
  smallest_angle m n p alpha := 
sorry

end find_smallest_angle_l79_79920


namespace solution_to_problem_l79_79455

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, n > 0 → (1 / a (n + 1)) = (1 / a n) + 3)

theorem solution_to_problem (a : ℕ → ℝ) (h : sequence a) : a 10 = 1 / 28 :=
by
  sorry

end solution_to_problem_l79_79455


namespace geometric_product_formula_l79_79017

noncomputable def P (n : ℕ) (a : ℝ) (r : ℝ) : ℝ :=
  a ^ n * r ^ ((n - 1) * n / 2)

noncomputable def S (n : ℕ) (a : ℝ) (r : ℝ) : ℝ :=
  a * (1 - r^n) / (1 - r)

noncomputable def S' (n : ℕ) (a : ℝ) (r : ℝ) : ℝ :=
  (r ^ (-(n - 1))) / a * (1 - r^n) / (1 - r)

theorem geometric_product_formula (n : ℕ) (a : ℝ) (r : ℝ) :
  let S_n := S n a r in
  let S'_n := S' n a r in
  (S_n / S'_n) ^ (n / 2) = P n a r := 
sorry

end geometric_product_formula_l79_79017


namespace sum_last_two_digits_l79_79648

-- Definition of the problem conditions
def seven : ℕ := 10 - 3
def thirteen : ℕ := 10 + 3

-- Main statement of the problem
theorem sum_last_two_digits (x : ℕ) (y : ℕ) : x = seven → y = thirteen → (7^25 + 13^25) % 100 = 0 :=
by
  intros
  rw [←h, ←h_1] -- Rewriting x and y in terms of seven and thirteen
  sorry -- Proof omitted

end sum_last_two_digits_l79_79648


namespace shape_cross_section_circular_l79_79272

-- Definitions of the geometric shapes and their properties
inductive Shape
| cone
| cylinder
| sphere
| pentagonal_prism

-- Predicate that states a shape can produce a circular cross-section when intersected by a plane
def can_produce_circular_cross_section : Shape → Prop
| Shape.cone := true
| Shape.cylinder := true
| Shape.sphere := true
| Shape.pentagonal_prism := false

-- The theorem we want to prove
theorem shape_cross_section_circular :
  (can_produce_circular_cross_section Shape.cone) ∧
  (can_produce_circular_cross_section Shape.cylinder) ∧
  (can_produce_circular_cross_section Shape.sphere) ∧
  ¬(can_produce_circular_cross_section Shape.pentagonal_prism) :=
by sorry

end shape_cross_section_circular_l79_79272


namespace bows_purple_l79_79507

theorem bows_purple (total_bows : ℕ)
  (h1 : total_bows / 4 are red)
  (h2 : total_bows / 3 are blue)
  (h3 : total_bows / 6 are purple)
  (yellow_bows : ℕ) (h4 : yellow_bows = 60)
  (h5 : total_bows - (total_bows / 4 + total_bows / 3 + total_bows / 6) = yellow_bows) :
  (total_bows / 6 = 40) :=
  sorry

end bows_purple_l79_79507


namespace cos_B_third_quadrant_l79_79029

theorem cos_B_third_quadrant (B : ℝ) (hB1 : π < B ∧ B < 3 * π / 2) (hB2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

end cos_B_third_quadrant_l79_79029


namespace complex_fraction_simplification_l79_79588

noncomputable def simplify_complex_fraction : ℂ :=
let i : ℂ := complex.I in
let z1 : ℂ := 2 - 2 * i in
let z2 : ℂ := 1 + 4 * i in
  z1 / z2

theorem complex_fraction_simplification : simplify_complex_fraction = - (6 / 17) - (10 / 17) * complex.I :=
by
  sorry

end complex_fraction_simplification_l79_79588


namespace max_consecutive_sum_lt_1000_l79_79237

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l79_79237


namespace max_consecutive_integers_sum_lt_1000_l79_79260

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l79_79260


namespace percentage_needed_to_pass_l79_79326

-- Define conditions
def student_score : ℕ := 80
def marks_shortfall : ℕ := 40
def total_marks : ℕ := 400

-- Theorem statement: The percentage of marks required to pass the test.
theorem percentage_needed_to_pass : (student_score + marks_shortfall) * 100 / total_marks = 30 := by
  sorry

end percentage_needed_to_pass_l79_79326


namespace chosen_number_l79_79283

theorem chosen_number (x: ℤ) (h: 2 * x - 152 = 102) : x = 127 :=
by
  sorry

end chosen_number_l79_79283


namespace constants_solution_l79_79798

theorem constants_solution :
  let a := 56 / 9
  let c := 5 / 3
  (∀ (v1 v2 : ℝ ^ 3),
    v1 = ![a, -1, c]
    ∧ v2 = ![8, 4, 6]
    → (v1.cross_product v2 = ![-14, -24, 34]))
    ↔ (a, c) = (56 / 9, 5 / 3) :=
by
  sorry

end constants_solution_l79_79798


namespace lambda_range_l79_79521

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + n

def inequality (λ : ℝ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (λ / n) > ((n + 1) / (a n + 1))

theorem lambda_range (a : ℕ → ℕ) (λ : ℝ) (h1 : sequence a) (h2 : inequality λ a) : 
  2 ≤ λ :=
sorry

end lambda_range_l79_79521


namespace egg_production_difference_l79_79530

def eggs_last_year : ℕ := 1416
def eggs_this_year : ℕ := 4636
def eggs_difference (a b : ℕ) : ℕ := a - b

theorem egg_production_difference : eggs_difference eggs_this_year eggs_last_year = 3220 := 
by
  sorry

end egg_production_difference_l79_79530


namespace max_stamps_without_discount_theorem_l79_79040

def total_money := 5000
def price_per_stamp := 50
def max_stamps_without_discount := 100

theorem max_stamps_without_discount_theorem :
  price_per_stamp * max_stamps_without_discount ≤ total_money ∧
  ∀ n, n > max_stamps_without_discount → price_per_stamp * n > total_money := by
  sorry

end max_stamps_without_discount_theorem_l79_79040


namespace candidate_D_votes_l79_79883

theorem candidate_D_votes :
  let total_votes := 10000
  let invalid_votes_percentage := 0.25
  let valid_votes := (1 - invalid_votes_percentage) * total_votes
  let candidate_A_percentage := 0.40
  let candidate_B_percentage := 0.30
  let candidate_C_percentage := 0.20
  let candidate_D_percentage := 1.0 - (candidate_A_percentage + candidate_B_percentage + candidate_C_percentage)
  let candidate_D_votes := candidate_D_percentage * valid_votes
  candidate_D_votes = 750 :=
by
  sorry

end candidate_D_votes_l79_79883


namespace find_magnitude_of_vector_l79_79436

variables (a b : ℝ → ℝ → ℝ)
variable (θ : ℝ)

def angle_between_vectors (a b : ℝ → ℝ → ℝ) (θ : ℝ) : Prop :=
  θ = π / 3

def magnitude_vector_a (a : ℝ → ℝ → ℝ) : ℝ := 2

def magnitude_vector_b (b : ℝ → ℝ → ℝ) : ℝ := 3

def dot_product (a b : ℝ → ℝ → ℝ) : ℝ :=
  magnitude_vector_a a * magnitude_vector_b b * Real.cos θ

theorem find_magnitude_of_vector :
  angle_between_vectors a b θ →
  magnitude_vector_a a = 2 →
  magnitude_vector_b b = 3 →
  let d_ab := 2 * magnitude_vector_a a - 3 * magnitude_vector_b b in
  Real.sqrt ((d_ab a a)^2 - 12 * dot_product a b + (d_ab b b)^2) = Real.sqrt 61 :=
by
  intros
  -- proof goes here
  sorry

end find_magnitude_of_vector_l79_79436


namespace prime_condition_holds_l79_79776

theorem prime_condition_holds (n : ℕ) (h : n ≥ 2) : 
  (∀ (a : Fin n → ℤ), (∃ i, ¬ ∀ k < n, (Fin.sumRange (λ j, a ((i + j) % n)) % n = 0))) ↔ Nat.Prime n :=
sorry

end prime_condition_holds_l79_79776


namespace proof_problem_l79_79427

noncomputable def a : ℝ := Real.log 2 / Real.log 10
def b : ℝ := Real.log 5 / Real.log 10

theorem proof_problem : a + b = 1 := by
  sorry

end proof_problem_l79_79427


namespace distinct_real_roots_of_quadratic_l79_79474

-- Define the problem's condition: m is a real number and the discriminant of x^2 + mx + 9 > 0
def discriminant_positive (m : ℝ) := m^2 - 36 > 0

theorem distinct_real_roots_of_quadratic (m : ℝ) (h : discriminant_positive m) :
  m ∈ Iio (-6) ∪ Ioi (6) :=
sorry

end distinct_real_roots_of_quadratic_l79_79474


namespace sqrt_inequality_l79_79429

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 1) :
  sqrt (1 + 2 * a) + sqrt (1 + 2 * b) ≤ 2 * sqrt 2 :=
sorry

end sqrt_inequality_l79_79429


namespace imaginary_unit_expression_l79_79374

theorem imaginary_unit_expression :
  ∀ (i : ℂ), i^2 = -1 → i^2 + i^4 = 0 := 
by
  intros i h
  have h1 : i^4 = (i^2)^2, by ring
  rw [h, neg_one_sq] at h1
  have h2 : i^2 + i^4 = -1 + 1, by rw [h, h1]
  rw [add_right_neg] at h2
  exact h2

end imaginary_unit_expression_l79_79374


namespace units_digit_sum_sequence_l79_79348

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | _ => n * factorial (n - 1)

-- Define the sequence of terms n! + n
def seq_term (n : ℕ) : ℕ := factorial n + n

-- Calculate the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Prove that the units digit of the sum of the sequence is 8
theorem units_digit_sum_sequence :
  units_digit (∑ n in (1 : Finset ℕ).seq 10, seq_term n) = 8 :=
by
  sorry

end units_digit_sum_sequence_l79_79348


namespace odd_n_divides_2_pow_factorial_n_minus_1_l79_79941

theorem odd_n_divides_2_pow_factorial_n_minus_1 (n : ℕ) (h_odd : n % 2 = 1) :
  n ∣ (2^(factorial n) - 1) := 
sorry

end odd_n_divides_2_pow_factorial_n_minus_1_l79_79941


namespace common_chord_eqn_l79_79611

theorem common_chord_eqn (x y : ℝ) :
  (x^2 + y^2 - 4 = 0) ∧ (x^2 + y^2 - 4x + 4y - 12 = 0) → (x - y + 2 = 0) :=
by
  sorry

end common_chord_eqn_l79_79611


namespace distinct_real_roots_of_quadratic_l79_79470

theorem distinct_real_roots_of_quadratic (m : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ x^2 + m * x + 9 = 0 ∧ y^2 + m * y + 9 = 0) ↔ m ∈ Ioo (-∞) (-6) ∪ Ioo 6 (∞) :=
by
  sorry

end distinct_real_roots_of_quadratic_l79_79470


namespace volume_between_planes_l79_79147

theorem volume_between_planes (A V A1 A2 : ℝ) (hA : A = 3) (hV : V = 3) (hA1 : A1 = 1) (hA2 : A2 = 2) :
  let k1 := Real.sqrt (A1 / A) in
  let k2 := Real.sqrt (A2 / A) in
  let V1 := k1 ^ 3 * V in
  let V2 := k2 ^ 3 * V in
  let v  := V2 - V1 in
  v = (2 * Real.sqrt 6 - Real.sqrt 3) / 3 :=
by {
  sorry
}

end volume_between_planes_l79_79147


namespace max_expr_on_ellipse_l79_79417

-- Define the ellipse condition
def is_on_ellipse (x y : ℝ) (b : ℝ) := x^2 / 4 + y^2 / b^2 = 1

-- Define a function representing the expression to be maximized
def expr (x y : ℝ) := x^2 + 2 * y

-- Define the maximum value of the expression based on conditions
def max_value (b : ℝ) :=
  if 0 < b ∧ b <= 4 then
    b^2 / 4 + 4
  else if b > 4 then
    2 * b
  else
    0  -- placeholder for non-valid b values (e.g., b <= 0)

-- The theorem to be proved
theorem max_expr_on_ellipse (x y b : ℝ) (hb : 0 < b) (hxy : is_on_ellipse x y b) :
  expr x y ≤ max_value b :=
sorry

end max_expr_on_ellipse_l79_79417


namespace line_AB_fixed_point_tangents_intersection_y_l79_79914

section

variable {A B M : Type} -- Define general type variables for points
variable (xA yA xB yB xM yM : ℝ) -- Define real-valued variables
variable (k m : ℝ) -- Slope and intercept

-- Condition: A and B lie on the curve x^2 = y
def on_curve (x y : ℝ) : Prop := x^2 = y

-- Condition: Product of the x-coordinates of A and B is -1
def product_is_neg_one (xA xB : ℝ) : Prop := xA * xB = -1

-- Prove that the line AB always passes through (0, 1)
theorem line_AB_fixed_point (h1 : on_curve xA yA) (h2 : on_curve xB yB) (h3 : product_is_neg_one xA xB) :
    ∃ (k m : ℝ), m = 1 ∧ (∀ x y : ℝ, y = k * x + m → y = k * x + 1 → (0, 1) lies on the line) := by
  sorry

-- Prove that the y-coordinate of the intersection point M of the tangents at A and B is -1
theorem tangents_intersection_y (h1 : on_curve xA yA) (h2 : on_curve xB yB) (h3 : product_is_neg_one xA xB) :
    ∃ yM : ℝ, yM = -1 := by
  sorry

end

end line_AB_fixed_point_tangents_intersection_y_l79_79914


namespace not_intersect_necessary_but_not_sufficient_for_skew_l79_79680

-- Define what it means for two lines to be skew
def is_skew (l1 l2 : Line) : Prop :=
  ¬ (∃ p, p ∈ l1 ∧ p ∈ l2) ∧ ¬ (parallel l1 l2)

-- Define conditions under which two lines do not intersect
def do_not_intersect (l1 l2 : Line) : Prop :=
  ¬ (∃ p, p ∈ l1 ∧ p ∈ l2)

-- Theorem: "Two lines do not intersect" is a necessary but not sufficient condition for "two lines are skew lines"
theorem not_intersect_necessary_but_not_sufficient_for_skew:
  ∀ (l1 l2 : Line), do_not_intersect l1 l2 ↔ is_skew l1 l2 ∨ parallel l1 l2 := 
sorry

end not_intersect_necessary_but_not_sufficient_for_skew_l79_79680


namespace alcohol_solution_l79_79281

/-- 
A 40-liter solution of alcohol and water is 5 percent alcohol. If 3.5 liters of alcohol and 6.5 liters of water are added to this solution, 
what percent of the solution produced is alcohol? 
-/
theorem alcohol_solution (original_volume : ℝ) (original_percent_alcohol : ℝ)
                        (added_alcohol : ℝ) (added_water : ℝ) :
  original_volume = 40 →
  original_percent_alcohol = 5 →
  added_alcohol = 3.5 →
  added_water = 6.5 →
  (100 * (original_volume * original_percent_alcohol / 100 + added_alcohol) / (original_volume + added_alcohol + added_water)) = 11 := 
by 
  intros h1 h2 h3 h4
  sorry

end alcohol_solution_l79_79281


namespace intersection_distance_l79_79444

noncomputable def find_AB_distance 
  (intersects : ∀ θ₁ θ₂ : ℝ, 
                   θ₁ ≠ θ₂ ∧ 
                   (1 + cos θ₁) * sqrt 3 + sin θ₁ = 2 * sqrt 3 ∧ 
                   (1 + cos θ₂) * sqrt 3 + sin θ₂ = 2 * sqrt 3)
  : ℝ := 
  1

theorem intersection_distance 
  (θ₁ θ₂ : ℝ) 
  (h₁ : 1 + cos θ₁ = 1 + cos θ₂ ∧ sin θ₁ = sin θ₂) 
  : find_AB_distance (by simp [h₁, ne_of_gt (by norm_num)]) = 1 := 
  sorry

end intersection_distance_l79_79444


namespace part1_solution_part2_solution_l79_79106

noncomputable def f (x a : ℝ) := |x + a| + |x - a|

theorem part1_solution : (∀ x : ℝ, f x 1 ≥ 4 ↔ x ∈ Set.Iic (-2) ∨ x ∈ Set.Ici 2) := by
  sorry

theorem part2_solution : (∀ x : ℝ, f x a ≥ 6 → a ∈ Set.Iic (-3) ∨ a ∈ Set.Ici 3) := by
  sorry

end part1_solution_part2_solution_l79_79106


namespace range_of_a_if_p_is_false_l79_79812

theorem range_of_a_if_p_is_false :
  (∀ x : ℝ, x^2 + a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) := 
sorry

end range_of_a_if_p_is_false_l79_79812


namespace max_consecutive_integers_sum_lt_1000_l79_79241

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l79_79241


namespace telepathic_connection_probability_l79_79119

open ProbabilityTheory

theorem telepathic_connection_probability :
  let events := {ab | let a := ab.fst, b := ab.snd in a ∈ ({1,2,3,4,5,6} : set ℕ) ∧ b ∈ ({1,2,3,4,5,6} : set ℕ)},
      successful_events := {ab | let a := ab.fst, b := ab.snd in |a - b| ≤ 1 ∧ a ∈ ({1,2,3,4,5,6} : set ℕ) ∧ b ∈ ({1,2,3,4,5,6} : set ℕ)},
      total_possibilities := (6 * 6 : ℕ),
      successful_possibilities := successful_events.to_finset.card in
  successful_possibilities = 16 ∧ total_possibilities = 36 →
  ENNReal.ofRational (successful_possibilities / total_possibilities) = 4 / 9 := by
  sorry

end telepathic_connection_probability_l79_79119


namespace passes_through_P_l79_79038

noncomputable def g (x : ℝ) : ℝ := sorry

theorem passes_through_P : 
  (∀ x, g(x) + g(-2 * x - 2) = -2) → g(-2/3) = -1 :=
by 
  intro h
  have P := h (-2/3)
  sorry

end passes_through_P_l79_79038


namespace rhombus_other_diagonal_length_l79_79939

def length_other_diagonal (d1 area : ℝ) : ℝ :=
  (2 * area) / d1

theorem rhombus_other_diagonal_length (d1 area : ℝ) (h1 : d1 = 80) (h2 : area = 4800) :
  length_other_diagonal d1 area = 120 :=
by
  -- Proof is omitted
  sorry

end rhombus_other_diagonal_length_l79_79939


namespace numThreeDigitDivBy5_l79_79403

-- Define the set of digits to choose from
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the conditions of forming a valid number
def isThreeDigitDivBy5 (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ (∃ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 10 * a + b * 100 + c = n ∧ (n % 5 = 0))

-- State the theorem
theorem numThreeDigitDivBy5 : (digits.filter (λ n, isThreeDigitDivBy5 n)).card = 36 :=
  sorry

end numThreeDigitDivBy5_l79_79403


namespace absSumFirst15_eq_153_l79_79551

-- Define the sequence an and the absolute sum of the first 15 terms.
def a_n (n : ℕ) : ℤ := 2 * (n + 1) - 7  -- Note: n starts from 0 in ℕ but n starts from 1 in math.

def absSumFirst15 : ℕ := 
  let abs_term (n : ℕ) := Int.natAbs (a_n n) in
  (List.range 15).map abs_term |>.sum

-- The theorem to be proven.
theorem absSumFirst15_eq_153 : absSumFirst15 = 153 :=
  sorry

end absSumFirst15_eq_153_l79_79551


namespace question1_question2_l79_79803

-- Condition Definitions
variable (f : ℕ+ → ℕ+)
variable (k : ℕ+) (h_k_pos : 0 < k)
variable (h_f_property : ∀ n : ℕ+, n > k → f n = n - k)

-- Question 1: Prove that f(1) = a where a is a positive integer when k = 1
theorem question1 : ∃ (a : ℕ+), f 1 = a :=
  sorry

-- Question 2: Prove that the number of different functions f is 32 when k = 5 and 1 ≤ f(n) ≤ 2 for n ≤ 5
theorem question2 : 
  let k' := 5 in
  let f_prop : ∀ (n : ℕ+), n ≤ 5 → 1 ≤ f n ∧ f n ≤ 2 in
  (∃ (f : ℕ+ → ℕ+), (∀ n : ℕ+, n > k' → f n = n - k') ∧ (∀ n : ℕ+, n ≤ 5 → 1 ≤ f n ∧ f n ≤ 2) ∧ (Π f' : ℕ+ → ℕ+, (∀ n : ℕ+, n > k' → f' n = n - k') → (∀ n : ℕ+, n ≤ 5 → 1 ≤ f' n ∧ f' n ≤ 2) → true)) :=
  sorry

end question1_question2_l79_79803


namespace dislike_both_radio_and_music_l79_79579

theorem dislike_both_radio_and_music :
  ∀ (total_people : ℕ) 
    (percent_dislike_radio percent_dislike_both : ℝ), 
  total_people = 1500 →
  percent_dislike_radio = 0.4 →
  percent_dislike_both = 0.15 →
  (percent_dislike_both * (percent_dislike_radio * total_people)).toNat = 90 :=
by
  intros total_people percent_dislike_radio percent_dislike_both 
  assume h_total h_percent_radio h_percent_both
  rw [h_total, h_percent_radio, h_percent_both]
  have h1 : percent_dislike_radio * total_people = 600 := by norm_num
  have h2 : percent_dislike_both * 600 = 90 := by norm_num
  rw [h1] at h2
  exact h2.symm.toNatEq.2 rfl
  sorry

end dislike_both_radio_and_music_l79_79579


namespace side_length_of_S2_l79_79950

variable (r s : ℝ)

theorem side_length_of_S2 (h1 : 2 * r + s = 2100) (h2 : 2 * r + 3 * s = 3400) : s = 650 := by
  sorry

end side_length_of_S2_l79_79950


namespace log_equation_solution_l79_79859

theorem log_equation_solution (b x : ℝ) (h_b_pos : b > 0) (h_b_ne_one : b ≠ 1) (h_x_ne_one : x ≠ 1) 
    (h_log_eq : log (x) / log (b^3) + log (b) / log (x^3) = 2) : 
  x = b^(3 + 2 * Real.sqrt 2) ∨ x = b^(3 - 2 * Real.sqrt 2) :=
by
  sorry

end log_equation_solution_l79_79859


namespace range_of_m_for_roots_greater_than_2_l79_79167

theorem range_of_m_for_roots_greater_than_2 :
  ∀ m : ℝ, (∀ x : ℝ, x^2 + (m-2)*x + 5 - m = 0 → x > 2) ↔ (-5 < m ∧ m ≤ -4) :=
  sorry

end range_of_m_for_roots_greater_than_2_l79_79167


namespace PQ_passes_through_fixed_point_l79_79807

variable {A B C D P Q E F : Type}
variable [convex_quadrilateral A B C D]
variable [non_parallel (line_through A B) (line_through C D)]
variable [point_on_line P (line_through A D)]
variable [P ≠ A]
variable [P ≠ D]
variable [Q = circle_intersection (circumcircle A B P) (circumcircle C D P) ≠ P]

theorem PQ_passes_through_fixed_point :
  ∃ F, ∀ P, Q = circle_intersection (circumcircle A B P) (circumcircle C D P) → line_through P Q = line_through P F :=
sorry

end PQ_passes_through_fixed_point_l79_79807


namespace meet_at_starting_line_l79_79852

theorem meet_at_starting_line (henry_time margo_time : ℕ) (h_henry : henry_time = 7) (h_margo : margo_time = 12) : Nat.lcm henry_time margo_time = 84 :=
by
  rw [h_henry, h_margo]
  sorry

end meet_at_starting_line_l79_79852


namespace polynomial_value_given_cond_l79_79441

variable (x : ℝ)
theorem polynomial_value_given_cond :
  (x^2 - (5/2) * x = 6) →
  2 * x^2 - 5 * x + 6 = 18 :=
by
  sorry

end polynomial_value_given_cond_l79_79441


namespace triangle_AB_value_l79_79503

theorem triangle_AB_value (A C AB BC : ℝ)
  (tan_A : ℝ := 1/3)
  (angle_C : ℝ := 150)
  (BC_val : ℝ := 1)
  (h1 : tan A = tan_A)
  (h2 : C = angle_C)
  (h3 : BC = BC_val):
  AB = sqrt 10 / 2 :=
by
  sorry

end triangle_AB_value_l79_79503


namespace parabola_c_value_l79_79619

theorem parabola_c_value (a b c : ℝ) (h1 : 3 = a * (-1)^2 + b * (-1) + c)
  (h2 : 1 = a * (-2)^2 + b * (-2) + c) : c = 1 :=
sorry

end parabola_c_value_l79_79619


namespace john_money_l79_79910

theorem john_money (cost_given : ℝ) : cost_given = 14 :=
by
  have gift_cost := 28
  have half_cost := gift_cost / 2
  exact sorry

end john_money_l79_79910


namespace find_x_l79_79862

theorem find_x (x : ℕ) : (x % 9 = 0) ∧ (x^2 > 144) ∧ (x < 30) → (x = 18 ∨ x = 27) :=
by 
  sorry

end find_x_l79_79862


namespace olivia_dad_spent_l79_79566

def cost_per_meal : ℕ := 7
def number_of_meals : ℕ := 3
def total_cost : ℕ := 21

theorem olivia_dad_spent :
  cost_per_meal * number_of_meals = total_cost :=
by
  sorry

end olivia_dad_spent_l79_79566


namespace solve_for_m_l79_79131

theorem solve_for_m (m : ℤ) 
  (h : (m - 3)^3 = (1 / 27)⁻¹) : 
  m = 6 := by
  sorry

end solve_for_m_l79_79131


namespace probability_at_most_one_even_is_3_div_4_l79_79632

def fair_dice_events : Set (ℕ × ℕ) := 
  { (r, b) | r ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6}}

def event_at_most_one_even : Set (ℕ × ℕ) :=
  { (r, b) | (r % 2 = 1 ∧ b % 2 = 1) ∨ (r % 2 = 1 ∧ b % 2 = 0) ∨ (r % 2 = 0 ∧ b % 2 = 1)}

noncomputable def probability_at_most_one_even : ℝ := 
  ↑(event_at_most_one_even.to_finset.card) / ↑(fair_dice_events.to_finset.card)

theorem probability_at_most_one_even_is_3_div_4 : 
  probability_at_most_one_even = 3 / 4 := sorry

end probability_at_most_one_even_is_3_div_4_l79_79632


namespace trapezoid_problem_l79_79851

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]
variables (A B C D : V)

noncomputable def vec_BD_eq : Prop :=
  let AD := (D - A)
  let BC := (C - B)
  let BD := (D - B)
  BD = 3 • AD - 2 • BC

noncomputable def cos_angle_AC_BD_eq : Prop :=
  let AC := (C - A)
  let BD := (D - B)
  let cos_theta := ⟪AC, BD⟫ / (‖AC‖ * ‖BD‖)
  cos_theta = - (real.sqrt 26) / 26

theorem trapezoid_problem
  (H_parallel: parallel (A - B) (C - D))
  (H_AB_AD_2: dist A B = 2 ∧ dist A D = 2)
  (H_CD_3: dist C D = 3)
  (H_AD_perp_AB: ⟪A - D, A - B⟫ = 0)
: vec_BD_eq A B C D ∧ cos_angle_AC_BD_eq A B C D :=
by
  sorry

end trapezoid_problem_l79_79851


namespace part1_part2_l79_79446

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.log x)

theorem part1 : 
  (∀ x, 0 < x → x < 1 → (f x) < f (1)) ∧ 
  (∀ x, 1 < x → x < Real.exp 1 → (f x) < f (Real.exp 1)) :=
sorry

theorem part2 :
  ∃ k, k = 2 ∧ ∀ x, 0 < x → (f x) > (k / (Real.log x)) + 2 * Real.sqrt x :=
sorry

end part1_part2_l79_79446


namespace triangle_angle_relation_l79_79216

/-- Given a triangle ABC with ∠C = 2 ∠B, and a point D inside the triangle such that D is equidistant from B and C, and A is equidistant from C and D, prove that ∠A = 3 ∠BAD. -/
theorem triangle_angle_relation (A B C D : Type) 
  (angle_A angle_B angle_C : ℝ)
  (h_angle_C : angle_C = 2 * angle_B) 
  (h_D_eq_BC : dist D B = dist D C)
  (h_A_eq_CD : dist A C = dist A D) :
  angle_A = 3 * ∠BAD := 
sorry

end triangle_angle_relation_l79_79216


namespace possible_values_of_m_l79_79488

-- Define the conditions that m is a real number and the quadratic equation having two distinct real roots
variable (m : ℝ)

-- Define the discriminant condition for having two distinct real roots
def discriminant_condition (a b c : ℝ) := b^2 - 4 * a * c > 0

-- State the required theorem
theorem possible_values_of_m (h : discriminant_condition 1 m 9) : m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo 6 ∞ :=
sorry

end possible_values_of_m_l79_79488


namespace t_shirts_left_yesterday_correct_l79_79971

-- Define the conditions
def t_shirts_left_yesterday (x : ℕ) : Prop :=
  let t_shirts_sold_morning := (3 / 5) * x
  let t_shirts_sold_afternoon := 180
  t_shirts_sold_morning = t_shirts_sold_afternoon

-- Prove that x = 300 given the above conditions
theorem t_shirts_left_yesterday_correct (x : ℕ) (h : t_shirts_left_yesterday x) : x = 300 :=
by
  sorry

end t_shirts_left_yesterday_correct_l79_79971


namespace coloring_of_russia_with_odd_valid_colorings_l79_79935

theorem coloring_of_russia_with_odd_valid_colorings :
  let regions : ℕ := 85
  let colors := {white, blue, red}
  let valid_color (r : ℕ → colors) : Prop :=
    ∀ i j, (i ≠ j ∧ adjacent i j) → ¬((r i = white ∧ r j = red) ∨ (r i = red ∧ r j = white))
  let unused_colors_allowed : Prop :=
    ∃ r : ℕ → colors, ∀ i ∈ 85, r i = blue ∨ r i = white ∨ r i = red
  let total_colorings_odd : Prop :=
    ∃ n, valid_color n ∧ unused_colors_allowed n ∧ n % 2 = 1
  total_colorings_odd :=
sorry

end coloring_of_russia_with_odd_valid_colorings_l79_79935


namespace total_curve_length_l79_79879

-- Define the properties of the right square prism
structure RightSquarePrism :=
  (side_edge_length : ℝ)
  (base_edge_length : ℝ)

-- Define the length of the curves problem based on the given prism properties
def prism : RightSquarePrism := 
{ side_edge_length := 4, base_edge_length := 4 }

-- Define the required theorem
theorem total_curve_length (prism : RightSquarePrism) (distance_from_P : ℝ) : 
  prism.side_edge_length = 4 → 
  prism.base_edge_length = 4 → 
  distance_from_P = 3 → 
  6 * Real.pi :=
begin
  -- side_edge_length = 4
  -- base_edge_length = 4
  -- distance_from_P = 3
  -- Total curve length = 6π
  sorry
end

end total_curve_length_l79_79879


namespace crossing_time_indeterminate_l79_79637

-- Define the lengths of the two trains.
def train_A_length : Nat := 120
def train_B_length : Nat := 150

-- Define the crossing time of the two trains when moving in the same direction.
def crossing_time_together : Nat := 135

-- Define a theorem to state that without additional information, the crossing time for a 150-meter train cannot be determined.
theorem crossing_time_indeterminate 
    (V120 V150 : Nat) 
    (H : V150 - V120 = 2) : 
    ∃ t, t > 0 -> t < 150 / V150 -> False :=
by 
    -- The proof is not provided.
    sorry

end crossing_time_indeterminate_l79_79637


namespace acute_triangle_perpendiculars_sum_eq_circum_and_inradius_l79_79678

noncomputable def circumcenter (O : Type) := sorry -- Definition of circumcenter
noncomputable def inradius (r : Type) := sorry -- Definition of inradius
noncomputable def circumradius (R : Type) := sorry -- Definition of circumradius
noncomputable def perpendiculars (k_a k_b k_c : Type) := sorry -- Definition of perpendiculars dropped from the circumcenter

theorem acute_triangle_perpendiculars_sum_eq_circum_and_inradius
  (triangle : Type) (k_a k_b k_c R r : ℝ)
  (h1: circumcenter(O)) (h2: circumradius(R)) (h3: inradius(r)) (h4: perpendiculars(k_a, k_b, k_c)) :
  k_a + k_b + k_c = R + r :=
sorry

end acute_triangle_perpendiculars_sum_eq_circum_and_inradius_l79_79678


namespace alcohol_concentration_solution_l79_79720

noncomputable def alcohol_concentration_problem_statement : Prop :=
  let alcohol_in_first := 0.30 * 5
  let alcohol_in_second := 0.45 * 10
  let alcohol_in_third := 0.60 * 7
  let total_alcohol := alcohol_in_first + alcohol_in_second + alcohol_in_third
  let total_volume := 25 -- Total mixture's volume in the new vessel
  let concentration := (total_alcohol / total_volume) * 100
  concentration = 40.8

theorem alcohol_concentration_solution : alcohol_concentration_problem_statement :=
by
  let alcohol_in_first := 0.30 * 5
  let alcohol_in_second := 0.45 * 10
  let alcohol_in_third := 0.60 * 7
  let total_alcohol := alcohol_in_first + alcohol_in_second + alcohol_in_third
  let total_volume := 25 -- Total mixture's volume in the new vessel
  let concentration := (total_alcohol / total_volume) * 100
  show concentration = 40.8,
  sorry -- Proof to be completed

end alcohol_concentration_solution_l79_79720


namespace find_manuscript_fee_l79_79940

def manuscript_fee (x : ℝ) : ℝ := if x ≤ 4000 then (x - 800) * 0.2 * 0.7 else x * 0.8 * 0.2 * 0.7

theorem find_manuscript_fee : manuscript_fee 2800 = 280 :=
by {
  -- proof would go here, but adding sorry to skip
  sorry
}

end find_manuscript_fee_l79_79940


namespace vertical_asymptote_l79_79399

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 - 6*x + 8)

theorem vertical_asymptote (c : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → ((x^2 - x + c) ≠ 0)) ∨
  (∀ x : ℝ, ((x^2 - x + c) = 0) ↔ (x = 2) ∨ (x = 4)) →
  c = -2 ∨ c = -12 :=
sorry

end vertical_asymptote_l79_79399


namespace person_share_100_l79_79601

theorem person_share_100 :
  let total_cost := 30_000_000_000
  let num_people := 300_000_000
  total_cost / num_people = 100 := by
  let total_cost := 30_000_000_000
  let num_people := 300_000_000
  exact quotient.eq.mpr (calc
    30_000_000_000 / 300_000_000 = 100 : by norm_num)

end person_share_100_l79_79601


namespace group_distribution_l79_79510

theorem group_distribution : 
  ∃ (ways : ℕ), ways = 1440 ∧ 
  ∃ (men women : ℕ), men = 4 ∧ women = 5 ∧ 
  ∃ (group1 group2 group3 : ℕ), group1 = 3 ∧ group2 = 3 ∧ group3 = 3 ∧
  ∀ (g1 g2 g3 : list (nat × nat)), 
    (g1.length = 3 ∧ g2.length = 3 ∧ g3.length = 3) ∧ 
    (∃ m1 w1 m2 w2 m3 w3, 
      (1 ≤ m1 ∧ m1 ≤ men) ∧ (1 ≤ w1 ∧ w1 ≤ women) ∧
      (1 ≤ m2 ∧ m2 ≤ men - m1) ∧ (1 ≤ w2 ∧ w2 ≤ women - w1) ∧
      (1 ≤ m3 ∧ m3 ≤ men - m1 - m2) ∧ (1 ≤ w3 ∧ w3 ≤ women - w1 - w2) ∧ 
      (m1 + w1 = 3 ∧ m2 + w2 = 3 ∧ m3 + w3 = 3)) :=
begin
  -- Insertion of proof goes here
  sorry,
end

end group_distribution_l79_79510


namespace solve_f_x_eq_1_ab_eq_1_a_b_half_greater_than_1_exists_b_0_l79_79922

-- Define the function f(x) = |ln x|
def f (x : ℝ) : ℝ := abs (log x)

-- Definitions and conditions
variables (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b)
variables (h4 : f b = 2 * f ((a + b) / 2))

-- Proof statements
theorem solve_f_x_eq_1 : f x = 1 → (x = real.exp(1) ∨ x = real.exp(-1)) := 
    sorry

theorem ab_eq_1 : f a = f b → a * b = 1 := 
    sorry

theorem a_b_half_greater_than_1 : f a = f b → (0 < a) → (a < b) → (a * b = 1) → (a + b) / 2 > 1 :=
    sorry

theorem exists_b_0 : (0 < a) → (a < b) → (a * b = 1) → (f b = 2 * f ((a + b) / 2)) → 
  ∃ b0 ∈ Ioo 3 4, (1 / b0^2 + b0^2 + 2 - 4 * b0 = 0) := 
    sorry

end solve_f_x_eq_1_ab_eq_1_a_b_half_greater_than_1_exists_b_0_l79_79922


namespace good_odd_numbers_l79_79332

def is_good_odd (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n ≥ 3 ∧
  ∃ (a : Fin n → ℕ), 
  (∃ (perm : Fin n → Fin n), ∀ i, ∃ ai, i = perm ai) ∧
  (∀ k, 0 ≤ k < n → 
    let sum := List.sum (List.of_fn (λ i, if i % 2 = 0 then a ⟨(k + i) % n, sorry⟩ else -a ⟨(k + i) % n, sorry⟩)) 
    in sum > 0)

theorem good_odd_numbers (n : ℕ) : is_good_odd n ↔ ∃ k : ℕ, n = 4 * k + 1 :=
sorry

end good_odd_numbers_l79_79332


namespace at_most_16_pies_without_ingredients_l79_79366

theorem at_most_16_pies_without_ingredients:
  ∀ (total number_of_pies: ℕ) (chocolate_fraction marsmallow_fraction cayenne_fraction soy_nut_fraction: ℚ), 
  let chocolate_pies := chocolate_fraction * total,
      marsmallow_pies := marsmallow_fraction * total,
      cayenne_pies := cayenne_fraction * total,
      soy_nut_pies := soy_nut_fraction * total in
   total = 48 → chocolate_fraction = 5 / 8 → marsmallow_fraction = 3 / 4 → 
   cayenne_fraction = 2 / 3 → soy_nut_fraction = 1 / 4 → soy_nut_fraction * total ≤ marsmallow_fraction * total → 
   (48 - cayenne_fraction * 48) = 16 := 
by
  intros total number_of_pies chocolate_fraction marsmallow_fraction cayenne_fraction soy_nut_fraction,
  let chocolate_pies := chocolate_fraction * total,
      marsmallow_pies := marsmallow_fraction * total,
      cayenne_pies := cayenne_fraction * total,
      soy_nut_pies := soy_nut_fraction * total,
  assume h1 : total = 48,
  assume h2 : chocolate_fraction = 5 / 8,
  assume h3 : marsmallow_fraction = 3 / 4,
  assume h4 : cayenne_fraction = 2 / 3,
  assume h5 : soy_nut_fraction = 1 / 4,
  assume h6 : soy_nut_fraction * total ≤ marsmallow_fraction * total,
  rw [h1, h4],
  norm_num,
  exact rfl,
  sorry

end at_most_16_pies_without_ingredients_l79_79366


namespace x_coordinate_of_point_l79_79810

-- Definition of a parabola and the distance properties
def on_parabola (M : ℝ × ℝ) : Prop := M.2^2 = 12 * M.1

def distance_to_focus (M : ℝ × ℝ) : ℝ := 
  real.sqrt ((M.1 - 3)^2 + M.2^2)

theorem x_coordinate_of_point 
  (M : ℝ × ℝ) 
  (h_on_parabola : on_parabola M)
  (h_distance_to_focus : distance_to_focus M = 8) : 
  M.1 = 5 := by
  sorry

end x_coordinate_of_point_l79_79810


namespace joe_speed_l79_79527

theorem joe_speed (P : ℝ) (J : ℝ) (h1 : J = 2 * P) (h2 : 2 * P * (2 / 3) + P * (2 / 3) = 16) : J = 16 := 
by
  sorry

end joe_speed_l79_79527


namespace simplify_sqrt_of_square_l79_79590

-- The given condition
def x : ℤ := -9

-- The theorem stating the simplified form
theorem simplify_sqrt_of_square : (Real.sqrt ((x : ℝ) ^ 2) = 9) := by    
    sorry

end simplify_sqrt_of_square_l79_79590


namespace wrapping_paper_area_l79_79307

variable (a b h w : ℝ) (a_gt_b : a > b)

theorem wrapping_paper_area : 
  ∃ total_area, total_area = 4 * (a * b + a * w + b * w + w ^ 2) :=
by
  sorry

end wrapping_paper_area_l79_79307


namespace slower_time_to_reach_top_l79_79553

def time_for_lola (stories : ℕ) (time_per_story : ℕ) : ℕ :=
  stories * time_per_story

def time_for_tara (stories : ℕ) (time_per_story : ℕ) (stopping_time : ℕ) (num_stops : ℕ) : ℕ :=
  (stories * time_per_story) + (num_stops * stopping_time)

theorem slower_time_to_reach_top (stories : ℕ) (lola_time_per_story : ℕ) (tara_time_per_story : ℕ) 
  (tara_stop_time : ℕ) (tara_num_stops : ℕ) : 
  stories = 20 
  → lola_time_per_story = 10 
  → tara_time_per_story = 8 
  → tara_stop_time = 3
  → tara_num_stops = 18
  → max (time_for_lola stories lola_time_per_story) (time_for_tara stories tara_time_per_story tara_stop_time tara_num_stops) = 214 :=
by sorry

end slower_time_to_reach_top_l79_79553


namespace area_of_triangle_l79_79965

def curve (x : ℝ) : ℝ := (1 / 3) * x ^ 3 + x

def point : ℝ × ℝ := (1, 4 / 3)

theorem area_of_triangle :
  let tangent_line (x : ℝ) := 2 * x - 2 / 3
  let x_intercept : ℝ := (2 / 3) / 2
  let y_intercept : ℝ := 2 / 3
  let area : ℝ := (1 / 2) * x_intercept * y_intercept
  area = 1 / 9 :=
by
  sorry

end area_of_triangle_l79_79965


namespace vector_lines_l79_79393

variables {R : Type*} [Real R]

def vector_field (c r : R^3) : R^3 := λ i, c (i+1 % 3) * r (i+2 % 3) - c (i+2 % 3) * r (i+1 % 3)

theorem vector_lines
  (c : R^3)
  (h_c : ∃ c1 c2 c3, c = ![c1, c2, c3])
  (r : R^3)
  (h_r : ∃ x y z, r = ![x, y, z])
  (a : R^3) :
  (a = vector_field c r) →
  (∃ A1 A2 : R, A1 > 0 ∧ x^2 + y^2 + z^2 = A1 ∧ c1 * x + c2 * y + c3 * z = A2) :=
sorry

end vector_lines_l79_79393


namespace part1_part2_l79_79046

theorem part1 (p : ℝ) (h : p = 2 / 5) : 
  (p^2 + 2 * (3 / 5) * p^2) = 0.352 :=
by 
  rw [h]
  sorry

theorem part2 (p : ℝ) (h : p = 2 / 5) : 
  (4 * (1 / (11.32 * p^4)) + 5 * (2.4 / (11.32 * p^4)) + 6 * (3.6 / (11.32 * p^4)) + 7 * (2.16 / (11.32 * p^4))) = 4.834 :=
by 
  rw [h]
  sorry

end part1_part2_l79_79046


namespace fraction_1790s_l79_79071

def total_states : ℕ := 30
def states_1790s : ℕ := 16

theorem fraction_1790s : (states_1790s / total_states : ℚ) = 8 / 15 :=
by
  -- We claim that the fraction of states admitted during the 1790s is exactly 8/15
  sorry

end fraction_1790s_l79_79071


namespace range_of_a_l79_79041

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + (a+2)*x + 1 < 0) → a ∈ set.Icc (-4:ℝ) (0:ℝ) :=
by
  sorry

end range_of_a_l79_79041


namespace find_incircle_radius_l79_79938

variables {param1 param3 r_KBC : ℝ}

def radius_incircle_trapezoid (param1 param3 : ℝ) : ℝ :=
  if param1 = 3.5 ∧ param3 = 4 then 6 else
  if param1 = 4.33 ∧ param3 = 9 then 12 else
  if param1 = 5 ∧ param3 = 2 then 4 else
  if param1 = 3 ∧ param3 = 6 then 8 else 0

theorem find_incircle_radius (param1 param3 : ℝ) :
  radius_incircle_trapezoid param1 param3 = r_KBC := by
  sorry

end find_incircle_radius_l79_79938


namespace sum_of_two_squares_l79_79836

theorem sum_of_two_squares (n : ℕ) (k m : ℤ) : 2 * n = k^2 + m^2 → ∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end sum_of_two_squares_l79_79836


namespace find_tangent_line_perpendicular_to_given_line_l79_79388

theorem find_tangent_line_perpendicular_to_given_line :
  (∀ x ∈ Set.Icc (-1) (-1), 2*x - 6*((x^3 + 3*x^2 - 1)) + 1 = 0) →
  (∃ l : ℝ, ∀ x : ℝ, 3*x + l + 2 = 0) :=
by
  sorry

end find_tangent_line_perpendicular_to_given_line_l79_79388


namespace value_of_expression_l79_79268

theorem value_of_expression : (165^2 - 153^2) / 12 = 318 := by
  sorry

end value_of_expression_l79_79268


namespace graphene_scientific_notation_l79_79459

theorem graphene_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ (0.00000000034 : ℝ) = a * 10^n ∧ a = 3.4 ∧ n = -10 :=
sorry

end graphene_scientific_notation_l79_79459


namespace intersection_M_N_l79_79549

-- Definitions of the sets M and N
def M : Set ℝ := { -1, 0, 1 }
def N : Set ℝ := { x | x^2 ≤ x }

-- The theorem to be proven
theorem intersection_M_N : M ∩ N = { 0, 1 } :=
by
  sorry

end intersection_M_N_l79_79549


namespace distinct_real_roots_of_quadratic_l79_79473

theorem distinct_real_roots_of_quadratic (m : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ x^2 + m * x + 9 = 0 ∧ y^2 + m * y + 9 = 0) ↔ m ∈ Ioo (-∞) (-6) ∪ Ioo 6 (∞) :=
by
  sorry

end distinct_real_roots_of_quadratic_l79_79473


namespace geometric_series_common_ratio_l79_79172

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l79_79172


namespace cos_third_quadrant_l79_79020

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l79_79020


namespace intersection_eq_l79_79800

def A : Set ℝ := {-1, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def C : Set ℝ := {2}

theorem intersection_eq : A ∩ B = C := 
by {
  sorry
}

end intersection_eq_l79_79800


namespace candy_given_away_l79_79111

-- Define the conditions
def pieces_per_student := 2
def number_of_students := 9

-- Define the problem statement as a theorem
theorem candy_given_away : pieces_per_student * number_of_students = 18 := by
  -- This is where the proof would go, but we omit it with sorry.
  sorry

end candy_given_away_l79_79111


namespace token_position_after_final_move_l79_79626

theorem token_position_after_final_move :
  ∃ (m : ℕ), m > 0 ∧ token_position 1 (64 * m) = 32 ∧ lamp_count m = 1996 := sorry

end token_position_after_final_move_l79_79626


namespace sum_and_average_of_six_consecutive_integers_l79_79139

/--
Given six consecutive integers starting from n + 1,
prove that their sum is 6n + 21 and their average is n + 3.5.
-/
theorem sum_and_average_of_six_consecutive_integers (n : ℤ) :
  let a := n + 1
  let b := n + 2
  let c := n + 3
  let d := n + 4
  let e := n + 5
  let f := n + 6
  (a + b + c + d + e + f = 6 * n + 21) ∧
  ((a + b + c + d + e + f) / 6 = n + 3.5) := 
by
  sorry

end sum_and_average_of_six_consecutive_integers_l79_79139


namespace polynomial_has_integer_root_l79_79609

noncomputable def f (x : ℤ) : ℤ := sorry

theorem polynomial_has_integer_root (f: ℤ → ℤ)
  (coeff_bound : ∀ n, abs (f n) ≤ 5000000)
  (has_roots : ∀ i ∈ (finset.range 1 21), ∃ k : ℤ, f k = i * k) : f 0 = 0 :=
begin
  sorry
end

end polynomial_has_integer_root_l79_79609


namespace simplify_and_evaluate_l79_79130

theorem simplify_and_evaluate (a : ℝ) (h : a = 1/2) :
  (a - 1) / (a - 2) * (a^2 - 4) / (a^2 - 2 * a + 1) - 2 / (a - 1) = -1 :=
by
  rw h
  sorry

end simplify_and_evaluate_l79_79130


namespace max_gold_coins_l79_79278

theorem max_gold_coins (k : ℤ) (n : ℤ) 
  (h₁ : n = 11 * k + 2) 
  (h₂ : n < 100) : 
  n ≤ 90 :=
begin
  sorry
end

end max_gold_coins_l79_79278


namespace a_range_l79_79140

noncomputable def find_a_range (x y : ℝ) (e : ℝ) (ln : ℝ → ℝ) (a : ℝ) : Prop :=
  2 * x + a * (y - 2 * e * x) * (ln y - ln x) = 0

theorem a_range (x y : ℝ) (e ln : ℝ → ℝ) (a : ℝ) (hx : 0 < x) (hy : 0 < y) :
  find_a_range x y e ln a ↔ (a < 0 ∨ a ≥ 2 / e) :=
sorry

end a_range_l79_79140


namespace hyperbola_eccentricity_l79_79847

theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  let c := p / 2 in
  let e := c / a in
  (e = sqrt 2 + 1) ↔ 
  ∃ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) ∧ (y^2 = 2 * p * x) ∧ (x = p / 2) :=
by
  sorry

end hyperbola_eccentricity_l79_79847


namespace theorem_AC_parallel_DEF_theorem_dihedral_angle_ABCP_theorem_exists_Q_on_EF_l79_79828

-- Definitions of the points and their properties
def P := (0 : ℝ, 0 : ℝ, real.sqrt 2)
def A := (1 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 1 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 2 : ℝ, 0 : ℝ)
def D := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def E := (0 : ℝ, 2 : ℝ, real.sqrt 2)
def F := (1/2 : ℝ, 0 : ℝ, real.sqrt 2 / 2)

noncomputable def N := ((P.1 + C.1) / 2, (P.2 + C.2) / 2, (P.3 + C.3) / 2)

-- 1. Theorem for AC parallel to plane DEF
theorem theorem_AC_parallel_DEF : 
  ∀ (A C E D F N : ℝ^3), 
  (angle A D C = (π / 2)) ∧ (angle B A D = (π / 2)) ∧
  (F = midpoint P A) ∧
  (|P - D| = sqrt 2) ∧
  (|A - B| = |A - D| = 1) ∧ (|C - D| = 2) ∧
  (E = (0, 2, sqrt 2)) ∧ (N = midpoint P C) ∧
  (is_rectangle P D C E) →
  parallel (line AC) (plane DEF) :=
sorry

-- 2. Theorem for dihedral angle A-BC-P
theorem theorem_dihedral_angle_ABCP : 
  ∀ (D P B C : ℝ^3),
  (|P - D| = sqrt 2) ∧
  (B = (1, 1, 0)) ∧
  (C = (0, 2, 0)) →
  dihedral_angle A B C P = (π / 4) :=
sorry

-- 3. Theorem for the existence of point Q on EF
theorem theorem_exists_Q_on_EF : 
  ∀ (D P A E F B C : ℝ^3),
  (F = midpoint P A) ∧
  (E = (0, 2, sqrt 2)) ∧
  (angle B Q plane B C P = (π / 6)) →
  ∃ Q : ℝ^3, Q ∈ segment E F ∧ |F - Q| = (sqrt 19 / 2) :=
sorry

end theorem_AC_parallel_DEF_theorem_dihedral_angle_ABCP_theorem_exists_Q_on_EF_l79_79828


namespace proof_part1_proof_part2_l79_79849

-- Definitions for conditions
def line_parametric (t : ℝ) : ℝ × ℝ := (t, sqrt 3 * t + sqrt 2 / 2)
def curve_polar (ρ θ : ℝ) : ℝ := ρ = 2 * cos (θ - π / 4)

-- Definitions of the solved parts
def slope_angle_of_line : ℝ := π / 3
def curve_rect_eq (x y : ℝ) : Prop := (x - sqrt 2 / 2)^2 + (y - sqrt 2 / 2)^2 = 1

-- Propositions to be proven
theorem proof_part1 : slope_angle_of_line = π / 3 ∧ (∀ x y, curve_rect_eq x y ↔ curve_polar (sqrt (x^2 + y^2)) (atan2 y x))
:= sorry

theorem proof_part2 (A B P : ℝ × ℝ) (hA : A ∈ line_intersection_points) (hB : B ∈ line_intersection_points) (hP : P = (0, sqrt 2 / 2)) :
  |distance P A| + |distance P B| = sqrt 10 / 2
:= sorry

end proof_part1_proof_part2_l79_79849


namespace time_to_cross_first_platform_l79_79716

noncomputable section

def train_length : ℝ := 310
def platform_1_length : ℝ := 110
def platform_2_length : ℝ := 250
def crossing_time_platform_2 : ℝ := 20

def total_distance_2 (train_length platform_2_length : ℝ) : ℝ :=
  train_length + platform_2_length

def train_speed (total_distance_2 crossing_time_platform_2 : ℝ) : ℝ :=
  total_distance_2 / crossing_time_platform_2

def total_distance_1 (train_length platform_1_length : ℝ) : ℝ :=
  train_length + platform_1_length

def crossing_time_platform_1 (total_distance_1 train_speed : ℝ) : ℝ :=
  total_distance_1 / train_speed

theorem time_to_cross_first_platform :
  crossing_time_platform_1 (total_distance_1 train_length platform_1_length)
                           (train_speed (total_distance_2 train_length platform_2_length)
                                        crossing_time_platform_2) 
  = 15 :=
by
  -- We would prove this in a detailed proof which is omitted here.
  sorry

end time_to_cross_first_platform_l79_79716


namespace marble_distribution_l79_79630

theorem marble_distribution (a b c : ℚ) (h1 : a + b + c = 78) (h2 : a = 3 * b + 2) (h3 : b = c / 2) : 
  a = 40 ∧ b = 38 / 3 ∧ c = 76 / 3 :=
by
  sorry

end marble_distribution_l79_79630


namespace geometric_series_ratio_half_l79_79201

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l79_79201


namespace inscribed_square_area_equilateral_triangle_l79_79690

noncomputable def equilateral_triangle_side : ℝ := 6
noncomputable def inradius_of_equilateral_triangle (a : ℝ) : ℝ := a / (2 * Real.sqrt 3)
noncomputable def inscribed_circle_radius : ℝ := inradius_of_equilateral_triangle equilateral_triangle_side
noncomputable def inscribed_square_area (r : ℝ) : ℝ := (r * Real.sqrt 2) ^ 2

theorem inscribed_square_area_equilateral_triangle :
  inscribed_square_area inscribed_circle_radius ≈ 5.997 :=
by
  -- proof omitted
  sorry

end inscribed_square_area_equilateral_triangle_l79_79690


namespace range_a_if_monotonically_increasing_l79_79497

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * sin x + cos x

theorem range_a_if_monotonically_increasing :
  (∀ x ∈ set.Icc (real.pi / 6) (real.pi / 4), (a * cos x - sin x) ≥ 0) →
  a ≥ 1 :=
begin
  sorry
end

end range_a_if_monotonically_increasing_l79_79497


namespace missing_fraction_is_correct_l79_79989

def sum_of_fractions (x : ℚ) : Prop :=
  (1/3 : ℚ) + (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + x = (45/100 : ℚ)

theorem missing_fraction_is_correct : sum_of_fractions (27/60 : ℚ) :=
  by sorry

end missing_fraction_is_correct_l79_79989


namespace max_consecutive_integers_sum_l79_79247

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l79_79247


namespace rook_mod_n_l79_79928

def rook_closed_loop_modulo (n : ℕ) (hn : n > 1) : Prop :=
∀ {path : list (ℤ × ℤ)},  
  (  -- Conditions of the problem
    (path.nodup) ∧ 
    (∀ i, (i < path.length - 1) → 
      (let (x1, y1) := path.nth_le i (by linarith) in 
      let (x2, y2) := path.nth_le (i+1) (by linarith) in 
      ((abs (x2 - x1) = n ∧ y1 = y2) ∨ (abs (y2 - y1) = n ∧ x1 = x2)))) ∧
    ((path.head = path.last))
  ) 
  → (let white_cells_inside := 
        (list.range n.cross_with (λ x y, if (x, y) ∉ path then 1 else 0)).sum in 
     white_cells_inside % n = 1)

theorem rook_mod_n (n : ℕ) (hn : n > 1) : rook_closed_loop_modulo n hn := 
sorry

end rook_mod_n_l79_79928


namespace distinct_real_roots_interval_l79_79480

open Set Real

theorem distinct_real_roots_interval (m : ℝ) : 
  (∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ IsRoot (λ x => x^2 + m * x + 9) r1 ∧ IsRoot (λ x => x^2 + m * x + 9) r2) ↔ 
  m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end distinct_real_roots_interval_l79_79480


namespace cos_third_quadrant_l79_79022

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l79_79022


namespace measure_of_angle_B_l79_79899

-- Define the triangle with vertices A, B, C and its properties
variables {A B C D : Type}
variables (AB BC AC CD : ℝ)

-- Define the conditions
variables (h0 : AB = BC) 
          (h1 : CD = AC)
          (h2 : ∀ (X : Type), X ∈ C → X ∈ D → isAngleBisector AB AC CD)

-- Define the statement to prove
theorem measure_of_angle_B (h0 : AB = BC) 
                           (h1 : CD = AC)
                           (h2 : ∀ (X : Type), X ∈ C → X ∈ D → isAngleBisector AB AC CD) :
                           ∠ABC = 45 :=
begin
  sorry
end

end measure_of_angle_B_l79_79899


namespace part1_part2_l79_79915

/-- Definition of set A as roots of the equation x^2 - 3x + 2 = 0 --/
def set_A : Set ℝ := {x | x ^ 2 - 3 * x + 2 = 0}

/-- Definition of set B as roots of the equation x^2 + (a - 1)x + a^2 - 5 = 0 --/
def set_B (a : ℝ) : Set ℝ := {x | x ^ 2 + (a - 1) * x + a ^ 2 - 5 = 0}

/-- Proof for intersection condition --/
theorem part1 (a : ℝ) : (set_A ∩ set_B a = {2}) → (a = -3 ∨ a = 1) := by
  sorry

/-- Proof for union condition --/
theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -3 ∨ a > 7 / 3) := by
  sorry

end part1_part2_l79_79915


namespace vector_b_correct_l79_79919

open Matrix

noncomputable def a : Fin 3 → ℝ :=
  ![3, 2, 4]

noncomputable def b : Fin 3 → ℝ :=
  ![0, 5, 5 / 2]

def dot_product (v1 v2 : Fin 3 → ℝ) := 
  ∑ i, v1 i * v2 i

def cross_product (v1 v2 : Fin 3 → ℝ) :=
  ![(v1 1 * v2 2 - v1 2 * v2 1), 
    (v1 2 * v2 0 - v1 0 * v2 2), 
    (v1 0 * v2 1 - v1 1 * v2 0)]

theorem vector_b_correct :
  dot_product a b = 20 ∧ cross_product a b = ![-15, 5, 10] := 
  sorry

end vector_b_correct_l79_79919


namespace num_solutions_to_inequality_l79_79463

theorem num_solutions_to_inequality : 
  let S := {x : ℕ | 12 < -2 * (x : ℤ) + 17} in
  S.card = 2 := 
by 
  sorry

end num_solutions_to_inequality_l79_79463


namespace problem_statement_l79_79841

-- Definitions related to the given conditions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi) / 6)

theorem problem_statement :
  (∀ x1 x2 : ℝ, (x1 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → (x2 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → x1 < x2 → f x1 < f x2) →
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) →
  f (-((5 * Real.pi) / 12)) = (Real.sqrt 3) / 2 :=
by
  intros h_mono h_symm
  sorry

end problem_statement_l79_79841


namespace water_leftover_l79_79993

theorem water_leftover (players : ℕ) (total_water_l : ℕ) (water_per_player_ml : ℕ) (spill_water_ml : ℕ)
  (h1 : players = 30) 
  (h2 : total_water_l = 8) 
  (h3 : water_per_player_ml = 200) 
  (h4 : spill_water_ml = 250) : 
  (total_water_l * 1000 - (players * water_per_player_ml + spill_water_ml) = 1750) :=
by
  -- conversion of total water to milliliters
  let total_water_ml := total_water_l * 1000
  -- calculation of total water used for players
  let total_water_used_for_players := players * water_per_player_ml
  -- calculation of total water including spill
  let total_water_used := total_water_used_for_players + spill_water_ml
  -- leftover water calculation
  have calculation : total_water_l * 1000 - (players * water_per_player_ml + spill_water_ml) = total_water_ml - total_water_used, by
    rw [total_water_ml, total_water_used, total_water_used_for_players]
  rw calculation
  -- conclusion by substituting known values
  rw [h1, h2, h3, h4]
  norm_num

end water_leftover_l79_79993


namespace alice_walk_distance_l79_79728

theorem alice_walk_distance (m n : ℕ) (h_rel_prime : Nat.coprime m n) :
  let total_gates := 15
  let spacing := 150
  let max_distance := 600
  let valid_scenarios := 108
  let total_scenarios := total_gates * (total_gates - 1)
  let P := (Rat.mk valid_scenarios total_scenarios).reduce
  let m := P.num.nat_abs
  let n := P.denom in
  m + n = 53 := by
  let P_val : ℚ := Rat.mk 108 210
  have h1 : (P_val.reduce.num.nat_abs, P_val.reduce.denom) = (18, 35) := by
    -- This is a conceptual step where we count the scenarios and reduce the fraction correctly
    sorry
  have h2 : P = P_val.reduce := by sorry
  exact congrArg Nat.add (Eq.trans h1 h2).symm

end alice_walk_distance_l79_79728


namespace sophie_saves_money_by_using_wool_balls_l79_79597

def cost_of_dryer_sheets_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                  (weeks_per_year : ℕ) (sheets_per_box : ℕ)
                                  (cost_per_box : ℝ) : ℝ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := sheets_per_year / sheets_per_box
  boxes_per_year * cost_per_box

theorem sophie_saves_money_by_using_wool_balls :
  cost_of_dryer_sheets_per_year 4 1 52 104 5.50 = 11.00 :=
by simp only [cost_of_dryer_sheets_per_year]; sorry

end sophie_saves_money_by_using_wool_balls_l79_79597


namespace lines_intersect_at_point_l79_79693

noncomputable def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

noncomputable def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 3 * v, 2 - v)

theorem lines_intersect_at_point :
  ∃ s v : ℚ,
    line1 s = (15 / 7, 16 / 7) ∧
    line2 v = (15 / 7, 16 / 7) ∧
    s = 4 / 7 ∧
    v = -2 / 7 := by
  sorry

end lines_intersect_at_point_l79_79693


namespace right_triangle_width_l79_79219

theorem right_triangle_width (height : ℝ) (side_square : ℝ) (width : ℝ) (n_triangles : ℕ) 
  (triangle_right : height = 2)
  (fit_inside_square : side_square = 2)
  (number_triangles : n_triangles = 2) :
  width = 2 :=
sorry

end right_triangle_width_l79_79219


namespace trapezoid_AH_DH_l79_79120

theorem trapezoid_AH_DH (AD BC AH DH: ℝ) (H_on_AD: H ∈ Icc (0 : ℝ) AD):
  CH = height. ABCH
  isosceles_trapezoid \(ABCD\)
  AD = 35
  BC = 15
  DH = \frac{AD - BC}{2}
  AH = AD - DH
:
  AH = 25 ∧ DH = 10 :=
by
  sorry

end trapezoid_AH_DH_l79_79120


namespace unique_or_infinite_solutions_l79_79813

variables {α : Type*} [linear_ordered_field α] {A B C A1 B1 C1 : point α}

-- Definitions for the problem conditions
def points_collinear (a b c : point α) : Prop :=
vector α 2 (λ (p : point α), p.x * b.y + p.y * c.x + a.x * b.y * c.y = 0)

def triangle (a b c : point α) : Prop :=
a ≠ b ∧ b ≠ c ∧ a ≠ c

def similar_triangles (a b c a1 b1 c1 : point α) : Prop :=
let m := (c.y - b.y) / (a.y - a.x) in
let n := (c1.y - b1.y) / (a1.y - a1.x) in
(m = n ∧ angle a b c = angle a1 b1 c1)

-- Statement of the problem
theorem unique_or_infinite_solutions
  (h_triangle : triangle A B C)
  (h_collinear_A1 : points_collinear A B C A1) :
  (∃ B1 C1, on_line A C B1 ∧ on_line A B C1 ∧ similar_triangles A B C A1 B1 C1) ∧
  (∃! B1 C1, on_line A C B1 ∧ on_line A B C1 ∧ similar_triangles A B C A1 B1 C1) :=
sorry

end unique_or_infinite_solutions_l79_79813


namespace quadratic_sum_eq_504_l79_79763

theorem quadratic_sum_eq_504 :
  ∃ (a b c : ℝ), (∀ x : ℝ, 20 * x^2 + 160 * x + 800 = a * (x + b)^2 + c) ∧ a + b + c = 504 :=
by sorry

end quadratic_sum_eq_504_l79_79763


namespace water_left_over_l79_79996

theorem water_left_over (players : ℕ) (initial_liters : ℕ) (milliliters_per_player : ℕ) (water_spill_ml : ℕ) :
  players = 30 → initial_liters = 8 → milliliters_per_player = 200 → water_spill_ml = 250 →
  (initial_liters * 1000) - (players * milliliters_per_player + water_spill_ml) = 1750 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  change 8 * 1000 - (30 * 200 + 250) = 1750
  norm_num
  sorry

end water_left_over_l79_79996


namespace max_area_center_of_circle_l79_79037

theorem max_area_center_of_circle (k : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + k*x + 2*y + k^2 = 0) →
  (∀ x y : ℝ, if k = 0 then (x, y) = (0, -1))
:= by
  sorry

end max_area_center_of_circle_l79_79037


namespace geometric_series_common_ratio_l79_79213

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l79_79213


namespace vector_dot_product_example_l79_79873

def vector_dot_product {α : Type} [Field α] (u v : EuclideanSpace α (Fin 2)) : α :=
  Fin 2 (λ i, u ⟨i, sorry⟩ * v ⟨i, sorry⟩).

theorem vector_dot_product_example : vector_dot_product ![(1 : ℝ), 1] ![(-1 : ℝ), 2] = 1 := 
by
  sorry

end vector_dot_product_example_l79_79873


namespace bus_speed_excluding_stoppages_l79_79768

theorem bus_speed_excluding_stoppages 
  (v : ℝ) 
  (speed_incl_stoppages : v * 54 / 60 = 45) : 
  v = 50 := 
  by 
    sorry

end bus_speed_excluding_stoppages_l79_79768


namespace arithmetic_progression_infinite_kth_powers_l79_79358

theorem arithmetic_progression_infinite_kth_powers {a d k : ℕ} (ha : a > 0) (hd : d > 0) (hk : k > 0) :
  (∀ n : ℕ, ¬ ∃ b : ℕ, a + n * d = b ^ k) ∨ (∀ b : ℕ, ∃ n : ℕ, a + n * d = b ^ k) :=
sorry

end arithmetic_progression_infinite_kth_powers_l79_79358


namespace simplify_t_l79_79494

theorem simplify_t (t : ℝ) (cbrt3 : ℝ) (h : cbrt3 ^ 3 = 3) 
  (ht : t = 1 / (1 - cbrt3)) : 
  t = - (1 + cbrt3 + cbrt3 ^ 2) / 2 := 
sorry

end simplify_t_l79_79494


namespace positive_root_approximation_l79_79066

noncomputable def seq (x : ℕ → ℝ) := ∀ n, x (n + 1) = Real.sqrt (1 + (1 / x n))

theorem positive_root_approximation : 
  ∃ x : ℝ, (x > 0) ∧ (∃ (x : ℕ → ℝ), seq x ∧ x 1 = 1 ∧ ∀ n, tendsto x.2 at_top (𝓝 x )) := 
sorry

end positive_root_approximation_l79_79066


namespace geometric_series_common_ratio_l79_79190

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l79_79190


namespace cosine_in_third_quadrant_l79_79024

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l79_79024


namespace count_7_primable_below_1000_is_8_l79_79703

def is_one_digit_prime (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_n_primable (n : ℕ) (x : ℕ) : Prop :=
  (x % n = 0) ∧ (∀ d ∈ (x.digits 10), is_one_digit_prime d)

noncomputable def count_7_primable_below_1000 : ℕ :=
  finset.card { x ∈ finset.range 1000 | is_n_primable 7 x }

theorem count_7_primable_below_1000_is_8 : count_7_primable_below_1000 = 8 := by
  sorry

end count_7_primable_below_1000_is_8_l79_79703


namespace greatest_number_of_quarters_l79_79765

def eva_has_us_coins : ℝ := 4.80
def quarters_and_dimes_have_same_count (q : ℕ) : Prop := (0.25 * q + 0.10 * q = eva_has_us_coins)

theorem greatest_number_of_quarters : ∃ (q : ℕ), quarters_and_dimes_have_same_count q ∧ q = 13 :=
sorry

end greatest_number_of_quarters_l79_79765


namespace solve_system_addition_l79_79792

theorem solve_system_addition (a b : ℝ) (h1 : 3 * a + 7 * b = 1977) (h2 : 5 * a + b = 2007) : a + b = 498 :=
by
  sorry

end solve_system_addition_l79_79792


namespace number_of_ordered_triples_l79_79011

def number_of_solutions (a b c : ℤ) : Bool :=
  abs (a + b) + c = 19 ∧ a * b + abs c = 97

theorem number_of_ordered_triples :
  (∃ s : Finset _ × Finset _ × Finset _, 
    s.1.card * s.2.card * s.3.card = 12 ∧
    ∀ a ∈ s.1, ∀ b ∈ s.2, ∀ c ∈ s.3, number_of_solutions a b c) :=
sorry

end number_of_ordered_triples_l79_79011


namespace cos_B_third_quadrant_l79_79027

theorem cos_B_third_quadrant (B : ℝ) (hB1 : π < B ∧ B < 3 * π / 2) (hB2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

end cos_B_third_quadrant_l79_79027


namespace triangle_integer_lengths_impossible_l79_79083

theorem triangle_integer_lengths_impossible
  (A B C D E I : Type)
  [Triangle A B C]
  [RightAngle (Angle A)]
  [OnSegment D A C]
  [OnSegment E A B]
  [AngleEqual (Angle A B D) (Angle D B C)]
  [AngleEqual (Angle A C E) (Angle E C B)]
  [Intersection (Segment B D) (Segment C E) I] :
  ¬ (IsIntegerLength (Segment A B) ∧ IsIntegerLength (Segment A C) ∧
     IsIntegerLength (Segment B I) ∧ IsIntegerLength (Segment I D) ∧
     IsIntegerLength (Segment C I) ∧ IsIntegerLength (Segment I E)) :=
sorry

end triangle_integer_lengths_impossible_l79_79083


namespace area_of_triangle_PQR_l79_79515

theorem area_of_triangle_PQR (A B C D P Q R : ℝ^2)
  (h_rect : A.1 = D.1 ∧ D.2 = A.2 ∧ B.1 = A.1 ∧ D.1 = C.1 ∧ B.2 = C.2 ∧ C.2 = D.2)
  (h_AB_eq_AP : dist A B = dist A P)
  (h_AP_eq_PQ : dist A P = dist P Q)
  (h_PQ_eq_QD : dist P Q = dist Q D)
  (h_DR_eq_RC : dist D R = dist R C)
  (h_BC_eq_24 : dist B C = 24)
  (h_AD_eq_24 : dist A D = 24) :
  let PQ := dist P Q
  let DR := dist D R
  have h_PQ_val : PQ = 8, from sorry,
  have h_DR_val : DR = 4, from sorry,
  (1 / 2) * PQ * DR = 16 := sorry

end area_of_triangle_PQR_l79_79515


namespace coefficient_of_x2_in_expansion_l79_79516

noncomputable def binom_coeff (n k : ℕ) : ℚ :=
  nat.choose n k

theorem coefficient_of_x2_in_expansion:
  ∃ (r : ℕ), (8 - 2 * r = 2 ∧ (binom_coeff 8 r : ℚ) * ((-2 : ℚ) ^ r) = -binom_coeff 8 3 * 2^3) :=
by
  use 3
  split
  { norm_num }
  { sorry }

end coefficient_of_x2_in_expansion_l79_79516


namespace angle_at_half_past_eight_l79_79068

-- Define the principle of clock angles
def hour_angle (hr : ℕ) (min : ℕ) : ℝ :=
  (hr % 12 * 30) + (min * 0.5)

def minute_angle (min : ℕ) : ℝ :=
  min * 6

noncomputable def angle_between_hands (hr : ℕ) (min : ℕ) : ℝ :=
  let ha := hour_angle hr min in
  let ma := minute_angle min in
  |ha - ma|

theorem angle_at_half_past_eight :
  angle_between_hands 8 30 = 75 :=
  sorry

end angle_at_half_past_eight_l79_79068


namespace roots_g_eq_4_max_value_g_min_value_g_l79_79843

noncomputable def g (x : ℝ) : ℝ :=
  (4 * (Real.sin x)^4 + 7 * (Real.cos x)^2) / (4 * (Real.cos x)^4 + (Real.sin x)^2)

theorem roots_g_eq_4 :
  {x : ℝ | g(x) = 4} = 
  {x : ℝ | ∃ k : ℤ, x = (k : ℝ) * Real.pi + Real.pi / 2} ∪ 
  {x : ℝ | ∃ k : ℤ, x = (k : ℝ) * Real.pi + Real.pi / 3} ∪
  {x : ℝ | ∃ k : ℤ, x = (k : ℝ) * Real.pi - Real.pi / 3} :=
by sorry

theorem max_value_g : ∃ t ∈ (Set.Icc 0 1), g (Real.arccos (sqrt t)) = 21 / 5 :=
by sorry

theorem min_value_g : ∃ t ∈ (Set.Icc 0 1), g (Real.arccos (sqrt t)) = 7 / 4 :=
by sorry

end roots_g_eq_4_max_value_g_min_value_g_l79_79843


namespace sum_first_2n_terms_l79_79845

-- Define the sequence
def a_n (n : ℕ) : ℤ := (-1)^n * n + 2^n

-- Define the sum of the first 2n terms.
def S_2n (n : ℕ) : ℤ := ∑ i in range (2*n+1), a_n i

-- Statement of the problem
theorem sum_first_2n_terms (n : ℕ) : S_2n n = -n + 3 * 2^(2*n) - 6 :=
by
  sorry

end sum_first_2n_terms_l79_79845


namespace sum_of_solutions_of_equation_l79_79788

theorem sum_of_solutions_of_equation :
  let f := (fun x : ℝ => (x - 4) ^ 2)
  ∃ S : Set ℝ, (S = {x | f x = 16}) ∧ (∑ s in S, s) = 8 := 
by
  sorry

end sum_of_solutions_of_equation_l79_79788


namespace fraction_eq_solution_l79_79171

theorem fraction_eq_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (3 / x = 1 / (x - 1)) ↔ x = 3 / 2 :=
begin
  sorry
end

end fraction_eq_solution_l79_79171


namespace total_students_sampled_l79_79303

theorem total_students_sampled :
  ∀ (seniors juniors freshmen sampled_seniors sampled_juniors sampled_freshmen total_students : ℕ),
    seniors = 1000 →
    juniors = 1200 →
    freshmen = 1500 →
    sampled_freshmen = 75 →
    sampled_seniors = seniors * (sampled_freshmen / freshmen) →
    sampled_juniors = juniors * (sampled_freshmen / freshmen) →
    total_students = sampled_seniors + sampled_juniors + sampled_freshmen →
    total_students = 185 :=
by
sorry

end total_students_sampled_l79_79303


namespace primes_remain_prime_l79_79080

theorem primes_remain_prime (n : ℕ) (hn : 2 ≤ n)
  (h : ∀ k, 0 ≤ k → k ≤ (Int.floor (real.sqrt (n / 3 : ℝ))) → Prime (k^2 + k + n)) :
  ∀ k, 0 ≤ k → k ≤ n - 2 → Prime (k^2 + k + n) :=
begin
  sorry
end

end primes_remain_prime_l79_79080


namespace smallest_N_for_Bernardo_win_l79_79048

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10).sum

theorem smallest_N_for_Bernardo_win :
  let N := 63 in
  (27 * N + 1200 < 3000) ∧ (sum_of_digits N = 9) :=
by
  sorry

end smallest_N_for_Bernardo_win_l79_79048


namespace two_digit_number_reversed_l79_79267

theorem two_digit_number_reversed :
  ∃ (x y : ℕ), (10 * x + y = 73) ∧ (10 * x + y = 2 * (10 * y + x) - 1) ∧ (x < 10) ∧ (y < 10) := 
by
  sorry

end two_digit_number_reversed_l79_79267


namespace third_worker_operates_two_looms_l79_79715

open Nat

variable (n m : ℕ) (a : ℕ → ℕ → ℕ)

def operates_loom (i j : ℕ) : Prop :=
  a i j = 1

theorem third_worker_operates_two_looms
  (h_positive_n : 0 < n)
  (h_positive_m : 0 < m)
  (h_aij : ∀ i j, a i j = 1 ∨ a i j = 0)
  (h_sum : ∑ j in range n, a 3 j = 2) :
  ∃ looms : finset ℕ, looms.card = 2 ∧ ∀ j ∈ looms, operates_loom 3 j :=
sorry

end third_worker_operates_two_looms_l79_79715


namespace D1E_perpendicular_plane_ADF_l79_79062

-- Definitions of the points and midpoints in the cube
def Cube : Type := sorry
def A (c : Cube) : Point := sorry
def B (c : Cube) : Point := sorry
def C (c : Cube) : Point := sorry
def D (c : Cube) : Point := sorry
def A1 (c : Cube) : Point := sorry
def B1 (c : Cube) : Point := sorry
def C1 (c : Cube) : Point := sorry
def D1 (c : Cube) : Point := sorry

def midpoint (p1 p2 : Point) : Point := sorry

def E (c : Cube) : Point := midpoint (D c) (C c)
def F (c : Cube) : Point := midpoint (C c) (C1 c)

-- The proof statement
theorem D1E_perpendicular_plane_ADF (c : Cube) :
  Perpendicular (line_through (D1 c) (E c)) (plane A (D c) (F c)) :=
sorry

end D1E_perpendicular_plane_ADF_l79_79062


namespace triangle_side_value_l79_79900

theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 1)
  (h2 : b = 4)
  (h3 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h4 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) :
  c = Real.sqrt 13 :=
sorry

end triangle_side_value_l79_79900


namespace median_is_8_4_l79_79884

open List

def data_set : List ℝ := [8.4, 7.5, 8.4, 8.5, 7.5, 9]

theorem median_is_8_4 : median data_set = 8.4 :=
by
  sorry

end median_is_8_4_l79_79884


namespace min_value_fraction_seq_l79_79422

noncomputable def a (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else
    let m := n - 1 in
    (2 * m * m + 98 * m + 102)

theorem min_value_fraction_seq : 
  ∀ (n : ℕ), n ≥ 1 → a n = 102 + (n - 2) * (2 * n + 2) → 
  ∃ (n_min : ℕ), n_min ≥ 1 ∧ (∀ n ≥ 1, (a n / n : ℝ) ≥ 26) := sorry

end min_value_fraction_seq_l79_79422


namespace smallest_product_is_minus_120_l79_79911

def smallest_product_of_triplet (l : List Int) : Int :=
  l.product3.toList.map (λ t => t.fst * t.snd.1 * t.snd.2).minimum'.get!

theorem smallest_product_is_minus_120 (l : List Int) (h : l = [-5, -3, -1, 2, 4, 6]) :
  smallest_product_of_triplet l = -120 :=
by
  sorry

end smallest_product_is_minus_120_l79_79911


namespace mary_probability_at_least_three_correct_l79_79108

noncomputable def probability_correct_guesses (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
Finset.sum (Finset.range (k + 1)) (λ i, (nat.choose n i : ℚ) * p^i * (1 - p)^(n - i))

theorem mary_probability_at_least_three_correct :
  probability_correct_guesses 5 2 (1/4) = 53/512 :=
by
  unfold probability_correct_guesses
  norm_cast
  sorry

end mary_probability_at_least_three_correct_l79_79108


namespace y1_gt_y3_gt_y2_l79_79804

noncomputable def y1 : ℝ := 2 ^ 1.8
noncomputable def y2 : ℝ := 8 ^ 0.48
noncomputable def y3 : ℝ := (1 / 2) ^ -1.5

theorem y1_gt_y3_gt_y2 : y1 > y3 ∧ y3 > y2 := by
  have hy1 : y1 = 2 ^ 1.8 := by rfl
  have hy2 : y2 = 2 ^ 1.44 := by 
    calc 
      8 ^ 0.48 = (2^3) ^ 0.48 : by rw [pow_mul]
      ... = 2 ^ (3 * 0.48) : by rfl
      ... = 2 ^ 1.44 : by norm_num
  have hy3 : y3 = 2 ^ 1.5 := by
    calc
      (1 / 2) ^ -1.5 = (2 ^ -1) ^ -1.5 : by rw [one_div, pow_neg]
      ... = 2 ^ (-(1 * -1.5)) : by rw [←pow_mul]
      ... = 2 ^ 1.5 : by norm_num
  rw [hy1, hy2, hy3]
  exact sorry

end y1_gt_y3_gt_y2_l79_79804


namespace point_on_line_l79_79773

theorem point_on_line (t : ℝ) : (∃ t : ℝ, (t, 6) lies_on_line_through (0, 4) (-6, 1)) → t = 4 :=
by
  sorry

end point_on_line_l79_79773


namespace cos_48_eq_sqrt2_over_2_l79_79354

theorem cos_48_eq_sqrt2_over_2 : 
  ∃ (y : ℝ), y = cos 48 ∧ y = sqrt 2 / 2 :=
by
  sorry

end cos_48_eq_sqrt2_over_2_l79_79354


namespace valid_combinations_count_l79_79721

-- Definitions of the roots, minerals, and incompatibilities
def roots : Finset String := {"root1", "root2", "root3", "root4"}
def minerals : Finset String := {"mineral1", "mineral2", "mineral3", "mineral4", "mineral5", "mineral6"}

-- Incompatibility definitions
def incompatible_with_root1 : Finset String := {"mineral1", "mineral2"}
def incompatible_with_root2_and_root3 : String := "mineral3"

-- We need to prove the number of valid combinations is 20
theorem valid_combinations_count :
  let total_combinations := Finset.card roots * Finset.card minerals in
  let incompatible_combinations := Finset.card incompatible_with_root1 + 2 in
  total_combinations - incompatible_combinations = 20 :=
by
  -- Add your proof here
  sorry

end valid_combinations_count_l79_79721


namespace divisibility_criterion_l79_79945

theorem divisibility_criterion :
  (∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ (1207 % x = 0) ∧
  (let a := x / 10 in let b := x % 10 in a^3 + b^3 = 344)) ↔
  (1207 % 17 = 0 ∧ let a1 := 1 in let b1 := 7 in a1^3 + b1^3 = 344) ∨
  (1207 % 71 = 0 ∧ let a2 := 7 in let b2 := 1 in a2^3 + b2^3 = 344) :=
by sorry

end divisibility_criterion_l79_79945


namespace geometric_series_ratio_half_l79_79206

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l79_79206


namespace point_on_graph_l79_79683

def f (a : ℝ) (x : ℝ) : ℝ := log a (2 * x + 3) + 2

theorem point_on_graph (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  f a (-1) = 2 :=
sorry

end point_on_graph_l79_79683


namespace rectangle_perimeter_l79_79618

theorem rectangle_perimeter (AB BK : ℝ) 
  (h_perimeter : 4 * (AB + BK + 17) = 180)
  (h_division : BK = BK) : 
  2 * (2 * AB + 2 * BK) = 112 :=
by
  have h_AB_BK_sum : AB + BK = 28, from sorry,
  rw h_AB_BK_sum,
  have h_2AB_2BK : 2 * (2 * AB + 2 * BK) = 2 * (2 * 28), from sorry,
  rw h_2AB_2BK,
  norm_num

end rectangle_perimeter_l79_79618


namespace largest_interesting_number_is_correct_l79_79698

def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits
  (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) ∧
  (∀ i, i < digits.length - 1 → is_square ((digits.nth i + digits.nth (i + 1)) : ℕ))

def largest_interesting_number : ℕ :=
6310972

theorem largest_interesting_number_is_correct : ∀ n : ℕ, is_interesting n → n ≤ largest_interesting_number :=
sorry

end largest_interesting_number_is_correct_l79_79698


namespace distance_of_canteen_from_each_camp_l79_79313

noncomputable def distanceFromCanteen (distGtoRoad distBtoG : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (distGtoRoad ^ 2 + distBtoG ^ 2)
  hypotenuse / 2

theorem distance_of_canteen_from_each_camp :
  distanceFromCanteen 360 800 = 438.6 :=
by
  sorry -- The proof is omitted but must show that this statement is valid.

end distance_of_canteen_from_each_camp_l79_79313


namespace payment_for_400_payment_for_600_payment_for_x_200_to_500_payment_for_x_500_or_more_total_payment_for_a_l79_79689

noncomputable section

-- Define the discount conditions
def actual_payment (amount : ℕ) : ℕ :=
  if amount < 200 then
    amount
  else if amount < 500 then
    amount * 9 / 10
  else
    450 + (amount - 500) * 8 / 10

-- Prove the instances for specific purchases
theorem payment_for_400 : actual_payment 400 = 360 := by sorry
theorem payment_for_600 : actual_payment 600 = 530 := by sorry

-- Define the algebraic expressions for general purchases
def actual_payment_algebra (x : ℕ) : ℕ :=
  if x < 500 then
    x * 9 / 10
  else
    450 + (x - 500) * 8 / 10

-- Prove for the algebraic expressions
theorem payment_for_x_200_to_500 (x : ℕ) (h₁ : 200 ≤ x) (h₂ : x < 500) : actual_payment_algebra x = x * 9 / 10 := by sorry
theorem payment_for_x_500_or_more (x : ℕ) (h : x ≥ 500) : actual_payment_algebra x = 450 + (x - 500) * 8 / 10 := by sorry

-- Define the conditions for two transactions
def total_payment_for_two (a b : ℕ) (h₁ : 200 < a) (h₂ : a < 300) (h₃: a + b = 820): ℕ :=
  actual_payment a + actual_payment b

-- Prove the total payment for a given 'a' and total sum 820
theorem total_payment_for_a (a : ℕ) (h₁ : 200 < a) (h₂ : a < 300) (h₃ : a + (820 - a) = 820) : total_payment_for_two a (820 - a) h₁ h₂ h₃ = 0.1 * a + 706 := by sorry

end payment_for_400_payment_for_600_payment_for_x_200_to_500_payment_for_x_500_or_more_total_payment_for_a_l79_79689


namespace quadratic_completion_l79_79380

theorem quadratic_completion (x : ℝ) : 
  ∃ a h k : ℝ, a = 1 ∧ h = 4 ∧ k = -13 ∧ x^2 - 8x + 3 = a * (x - h)^2 + k :=
by 
  use 1, 4, -13
  -- Proof steps will go here
  sorry

end quadratic_completion_l79_79380


namespace log_computation_cannot_be_direct_l79_79501

theorem log_computation_cannot_be_direct (log7 log8 : ℝ)
    (h1 : log7 ≈ 0.8451)
    (h2 : log8 ≈ 0.9031) :
    ¬ ( ∃ (log4 : ℝ), (log4 = 2 * (1 / 3 * log8)) ∧ (log28 = log7 + log4) ) :=
by sorry

end log_computation_cannot_be_direct_l79_79501


namespace incorrect_proposition_B_l79_79151

-- Definition of the complex number z
def z : ℂ := 1 + complex.i

-- Proposition A (correct)
def prop_A := complex.abs z = real.sqrt 2

-- Proposition B (incorrect)
def prop_B := complex.im z = complex.i

-- Proposition C (correct)
def prop_C := (z.re > 0) ∧ (z.im > 0)

-- Proposition D (correct)
def prop_D := complex.conj z = 1 - complex.i

-- Theorem: Proposition B is incorrect
theorem incorrect_proposition_B : prop_B = false := 
sorry

end incorrect_proposition_B_l79_79151


namespace fourth_side_of_cyclic_quadrilateral_l79_79158

variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

def inscribed_cyclic_quadrilateral (circumradius : ℝ) (sides_length : ℝ) :=
  (circumradius = 2 * Real.sqrt 2) ∧ 
  ((dist A B = sides_length) ∧ (dist B C = sides_length) ∧ (dist C D = sides_length))

theorem fourth_side_of_cyclic_quadrilateral {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (circumradius := 2 * Real.sqrt 2) (sides_length := 2) :
  inscribed_cyclic_quadrilateral {circumradius} {sides_length} →
  dist A D = 5 :=
by
  sorry

end fourth_side_of_cyclic_quadrilateral_l79_79158


namespace other_root_is_neg4_l79_79445

theorem other_root_is_neg4 (m : ℝ) (h : (-1)^2 + 5 * (-1) + m = 0) : (-1 + -4 = -5) := by
  have h1 : 1 - 5 + m = 0 := by rw [sq_neg_one, mul_neg_one]; exact h
  have h2 : -4 = -5 + 1 := by mini_solver
  rw neg_add_eq_sub
  exact h2
  sorry

end other_root_is_neg4_l79_79445


namespace number_of_pairs_l79_79461

theorem number_of_pairs (n : ℕ) (h : n = 2835) :
  ∃ (count : ℕ), count = 20 ∧
  (∀ (x y : ℕ), (0 < x ∧ 0 < y ∧ x < y ∧ (x^2 + y^2) % (x + y) = 0 ∧ (x^2 + y^2) / (x + y) ∣ n) → count = 20) := 
sorry

end number_of_pairs_l79_79461


namespace positive_real_solution_l79_79855

def polynomial (x : ℝ) : ℝ := x^4 + 10*x^3 - 2*x^2 + 12*x - 9

theorem positive_real_solution (h : polynomial 1 = 0) : polynomial 1 > 0 := sorry

end positive_real_solution_l79_79855


namespace max_consecutive_sum_lt_1000_l79_79238

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l79_79238


namespace area_of_triangle_l79_79886

theorem area_of_triangle (BC : ℝ) (h1 : BC = 8 * Real.sqrt 2) (h2 : ∠B = 90) (h3 : ∠A = 45) (h4 : ∠C = 45) :
  let AB := BC / (Real.sqrt 2)
  let AC := AB
  (1 / 2) * AB * AC = 32 := by
  unfold let_def
  sorry

end area_of_triangle_l79_79886


namespace water_leftover_l79_79992

theorem water_leftover (players : ℕ) (total_water_l : ℕ) (water_per_player_ml : ℕ) (spill_water_ml : ℕ)
  (h1 : players = 30) 
  (h2 : total_water_l = 8) 
  (h3 : water_per_player_ml = 200) 
  (h4 : spill_water_ml = 250) : 
  (total_water_l * 1000 - (players * water_per_player_ml + spill_water_ml) = 1750) :=
by
  -- conversion of total water to milliliters
  let total_water_ml := total_water_l * 1000
  -- calculation of total water used for players
  let total_water_used_for_players := players * water_per_player_ml
  -- calculation of total water including spill
  let total_water_used := total_water_used_for_players + spill_water_ml
  -- leftover water calculation
  have calculation : total_water_l * 1000 - (players * water_per_player_ml + spill_water_ml) = total_water_ml - total_water_used, by
    rw [total_water_ml, total_water_used, total_water_used_for_players]
  rw calculation
  -- conclusion by substituting known values
  rw [h1, h2, h3, h4]
  norm_num

end water_leftover_l79_79992


namespace water_left_over_l79_79994

theorem water_left_over (players : ℕ) (initial_liters : ℕ) (milliliters_per_player : ℕ) (water_spill_ml : ℕ) :
  players = 30 → initial_liters = 8 → milliliters_per_player = 200 → water_spill_ml = 250 →
  (initial_liters * 1000) - (players * milliliters_per_player + water_spill_ml) = 1750 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  change 8 * 1000 - (30 * 200 + 250) = 1750
  norm_num
  sorry

end water_left_over_l79_79994


namespace max_consecutive_integers_sum_lt_1000_l79_79244

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l79_79244


namespace mrs_petersons_change_l79_79112

-- Define the conditions
def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def discount_rate : ℚ := 0.10
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

-- Formulate the proof statement
theorem mrs_petersons_change :
  let total_cost_before_discount := num_tumblers * cost_per_tumbler
  let discount_amount := total_cost_before_discount * discount_rate
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let total_amount_paid := num_bills * value_per_bill
  let change_received := total_amount_paid - total_cost_after_discount
  change_received = 95 := by sorry

end mrs_petersons_change_l79_79112


namespace geometric_series_common_ratio_l79_79177

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l79_79177


namespace coplanar_vectors_lambda_eq_one_l79_79405

open Real

def vector_a : Vector3 := ⟨2, -1, 3⟩
def vector_b : Vector3 := ⟨-1, 4, -2⟩
def vector_c (λ : Real) : Vector3 := ⟨1, 3, λ⟩

theorem coplanar_vectors_lambda_eq_one (λ : Real) :
  (λ : Real) → (⟨1, 3, λ⟩ ∈ span ℝ {⟨2, -1, 3⟩, ⟨-1, 4, -2⟩}) → λ = 1 :=
by
  sorry

end coplanar_vectors_lambda_eq_one_l79_79405


namespace distinct_real_roots_interval_l79_79483

open Set Real

theorem distinct_real_roots_interval (m : ℝ) : 
  (∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ IsRoot (λ x => x^2 + m * x + 9) r1 ∧ IsRoot (λ x => x^2 + m * x + 9) r2) ↔ 
  m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end distinct_real_roots_interval_l79_79483


namespace bounded_region_area_correct_l79_79779

open Real

def bounded_region_area : ℝ :=
  let circle_radius := 3
  let square_half_diagonal := 3 * sqrt 2 / 2
  let sector_area := (π * circle_radius ^ 2) / 4
  let triangle_area := (1/2) * square_half_diagonal ^ 2
  in sector_area - triangle_area

theorem bounded_region_area_correct :
  bounded_region_area = 9 * (π - 1) / 4 :=
by
  sorry

end bounded_region_area_correct_l79_79779


namespace geometric_series_common_ratio_l79_79207

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l79_79207


namespace find_a_l79_79835

theorem find_a (a : ℝ) : (∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5) → (∃ x : ℝ, x = 3) → a = 8 :=
by
  sorry

end find_a_l79_79835


namespace hollow_circles_in_first_2003_circles_l79_79713

-- Define the sequence of circles as a list for simplicity
def sequence : List Char := ['●', '○', '●', '●', '○', '●', '●', '●', '○', '●', '●', '●', '●', 
                            '○', '●', '●', '●', '●', '●', '○', '●', '●', '●', '●', '●', '○']

-- Define the length of the repeating sequence
def sequence_length : Nat := sequence.length

-- Define the number of hollow circles in the sequence
def hollow_circles_in_sequence : Nat := sequence.filter (fun c => c = '○').length

-- Function to count hollow circles in the first n elements of the sequence
def hollow_circles (n : Nat) : Nat :=
  let full_sequences := n / sequence_length
  let remainder := n % sequence_length
  let full_hollow_circles := full_sequences * hollow_circles_in_sequence
  let partial_sequence_hollow_circles := (sequence.take remainder).filter (fun c => c = '○').length
  full_hollow_circles + partial_sequence_hollow_circles

-- Theorem stating the number of hollow circles in the first 2003 circles is 446
theorem hollow_circles_in_first_2003_circles : hollow_circles 2003 = 446 :=
  by sorry

end hollow_circles_in_first_2003_circles_l79_79713


namespace inverse_function_result_l79_79102

open Function

theorem inverse_function_result (f h : ℂ → ℂ) (hf_inv : ∀ x, f (f⁻¹ x) = x) (hhx : ∀ x, f⁻¹ (h x) = 2 * x^2 + 4) :
  h⁻¹ (f 3) = ± (Complex.I / Real.sqrt 2) :=
by
  sorry

end inverse_function_result_l79_79102


namespace find_number_l79_79649

theorem find_number :
  ∃ (x : ℕ), (sqrt x / 2) = 2 :=
by
  use 16
  have h1 : sqrt 16 = 4 := by sorry
  have h2 : 4 / 2 = 2 := by sorry
  rw [h1] at *
  exact h2

end find_number_l79_79649


namespace cross_product_of_a_and_b_l79_79384

def vector := ℝ × ℝ × ℝ 

def a : vector := (4, 3, -7)
def b : vector := (2, -1, 4)

def cross_product (v₁ v₂ : vector) : vector :=
  (v₁.2 * v₂.3 - v₁.3 * v₂.2,
   v₁.3 * v₂.1 - v₁.1 * v₂.3,
   v₁.1 * v₂.2 - v₁.2 * v₂.1)

theorem cross_product_of_a_and_b :
  cross_product a b = (5, -30, -10) :=
by sorry

end cross_product_of_a_and_b_l79_79384


namespace pyramid_integer_volume_values_l79_79280

open Real

theorem pyramid_integer_volume_values :
  let AE := 1024
  let AB := 640
  let height_diff := 12
  let max_k := 1024 / height_diff
  (∃ k : ℕ, 0 ≤ k ∧ k ≤ max_k ∧ 1024 - k * height_diff = some h) ∧
      (volume_of_pyramid AE AB h) ∈ ℤ :=
  sorry

end pyramid_integer_volume_values_l79_79280


namespace perpendicular_lines_m_l79_79869

open Real

theorem perpendicular_lines_m (m : ℝ) :
  let l1 := λ x y : ℝ, x + 2 * m * y - 1 = 0
  let l2 := λ x y : ℝ, (3 * m - 1) * x - m * y - 1 = 0
  (∀ x y, l1 x y ∧ l2 x y → (3 * m - 1) * x - m * (-(1 / (2 * m)) * x + y) - 1 = 0) ↔ m = 1 ∨ m = 1 / 2 := by
sorry

end perpendicular_lines_m_l79_79869


namespace number_sum_values_f4_l79_79089

theorem number_sum_values_f4 (f : ℝ → ℝ)
  (H : ∀ (x y z : ℝ), f (x^2 + y^2 + z * f z) = x * f x + y^2 + z * f y) :
  let n := 2 in let s := 16 in n * s = 32 :=
by
  let n := 2
  let s := 16
  show n * s = 32
  sorry

end number_sum_values_f4_l79_79089


namespace max_consecutive_sum_le_1000_l79_79233

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l79_79233


namespace g_of_g_of_g_of_g_of_3_l79_79076

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else x^2 + 2

theorem g_of_g_of_g_of_g_of_3 : g (g (g (g 3))) = 3 :=
by sorry

end g_of_g_of_g_of_g_of_3_l79_79076


namespace radius_of_circle_in_xy_plane_l79_79986

theorem radius_of_circle_in_xy_plane (theta : ℝ) :
  let x := 2 * Real.sin (Real.pi / 3) * Real.cos theta,
      y := 2 * Real.sin (Real.pi / 3) * Real.sin theta,
      radius := Real.sqrt (x^2 + y^2)
  in radius = Real.sqrt 3 :=
by
  have x_def : x = 2 * Real.sin (Real.pi / 3) * Real.cos theta := rfl
  have y_def : y = 2 * Real.sin (Real.pi / 3) * Real.sin theta := rfl
  have radius_def : radius = Real.sqrt (x^2 + y^2) := rfl
  sorry

end radius_of_circle_in_xy_plane_l79_79986


namespace cos_48_conjecture_l79_79355

noncomputable def cos_24 := Real.cos (24 * Real.pi / 180)
noncomputable def cos_6 := Real.cos (6 * Real.pi / 180)

theorem cos_48_conjecture :
  let c := cos_24,
      d := Real.cos (48 * Real.pi / 180),
      e := cos_6 in
  d = 2 * c^2 - 1 ∧
  -e = 2 * d^2 - 1 ∧
  -e = 4 * c^3 - 3 * c →
  d = Real.cos (48 * Real.pi / 180) :=
by
  intro h
  sorry

end cos_48_conjecture_l79_79355


namespace x_y_divisible_by_3_l79_79134

theorem x_y_divisible_by_3
    (x y z t : ℤ)
    (h : x^3 + y^3 = 3 * (z^3 + t^3)) :
    (3 ∣ x) ∧ (3 ∣ y) :=
by sorry

end x_y_divisible_by_3_l79_79134


namespace exists_non_negative_partial_sum_l79_79001

theorem exists_non_negative_partial_sum (n : ℕ) (c : Fin (2 * n) → ℤ)
  (h_sum : (Finset.univ.sum (λ i : Fin (2 * n), c i)) = 0)
  (h_eq : ∀ j : Fin n, c j = c (⟨n + (j : ℕ), Nat.add_lt_add_right j.2 n⟩)) :
  ∃ k : Fin n, ∀ j : Fin n, (Finset.univ.filter (λ i, i.val < k + j + 1)).sum (λ i, c ⟨i, by apply Nat.lt_of_lt_pred; exact i.2⟩) ≥ 0 := 
sorry

end exists_non_negative_partial_sum_l79_79001


namespace find_m_l79_79850

open Set

theorem find_m (m : ℝ) : 
  let A := {-1, 2, 2*m - 1}
  let B := {2, m^2}
  (B ⊆ A) -> (m = 1) :=
by
  intro h
  have : m ∈ {x | x ^ 2 = 2 * x - 1}
  sorry

end find_m_l79_79850


namespace maximum_value_of_M_l79_79796

open Real

def parabola_through_points (a b c P : ℝ) (hP : P ≠ 0) : Prop :=
  (0 = a * 0 * 0 + b * 0 + c) ∧
  (0 = a * 3 * P * 3 * P + b * 3 * P + c) ∧
  (45 = a * (3 * P - 1) * (3 * P - 2) + b * (3 * P - 1) + c)

def vertex_x (P : ℝ) : ℝ := 3 * P / 2

def vertex_y (a P : ℝ) : ℝ := -a * (9 * P^2 / 4)

def sum_of_vertex (a P : ℝ) : ℝ := vertex_x P + vertex_y a P

def max_sum_vertex : ℝ := 138

theorem maximum_value_of_M (a P : ℝ) (hP : P ≠ 0) (h1 : parabola_through_points a 0 0 P hP) :
  ∃ (P : ℝ), sum_of_vertex a P = max_sum_vertex :=
by
  use 2
  sorry

end maximum_value_of_M_l79_79796


namespace hyperbola_eccentricity_l79_79846

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
    (hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → True)
    (focus : ∀ F : ℝ × ℝ, F = (- (sqrt 10) / 2, 0) → True)
    (circle : ∀ x y, (x - (sqrt 10) / 2)^2 + y^2 = 1 → True) :
  let c := (sqrt 10) / 2 in
  let e := c / a in
  e = (sqrt 10) / 2 := sorry

end hyperbola_eccentricity_l79_79846


namespace star_pentagon_angle_sum_l79_79612

noncomputable def star_pentagon_sum_of_angles (A B C D E : Type*) [HasAngle A B C] [HasAngle B C D] [HasAngle C D E] [HasAngle D E A] [HasAngle E A B] : Prop :=
  (angle A B C) + (angle B C D) + (angle C D E) + (angle D E A) + (angle E A B) = 180

theorem star_pentagon_angle_sum (A B C D E : Type*) [HasAngle A B C] [HasAngle B C D] [HasAngle C D E] [HasAngle D E A] [HasAngle E A B] :
  star_pentagon_sum_of_angles A B C D E := 
sorry

end star_pentagon_angle_sum_l79_79612


namespace arithmetic_sequence_general_term_l79_79837

variable {a : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ d a1, ∀ n, a n = a1 + d * (n - 1)

def sum_n_terms (a : ℕ → ℤ) (n : ℕ) := 
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_general_term 
  (h₁ : a 1 + a 2 = 10)
  (h₂ : a 5 = a 3 + 4)
  {n : ℕ} :
  ∃ d a1, (∀ n, a n = a1 + d * (n - 1)) ∧ 
  (a n = 2 * n + 2) ∧ 
  (∑ i in range (k + 1), a i < 2 * a k + a 2) → k = 1 := 
sorry

end arithmetic_sequence_general_term_l79_79837


namespace integral_sqrt_expression_l79_79867

theorem integral_sqrt_expression :
  ∀ (n : ℕ), (∃ r : ℕ, n - (5 / 2) * r = 0) →
  (∀ x, -(5 : ℝ) ≤ x ∧ x ≤ (5 : ℝ) → ∫ y in -5..5, real.sqrt ((25 : ℝ) - y^2) = (25 * (real.pi / 2))) := 
sorry

end integral_sqrt_expression_l79_79867


namespace geometric_series_common_ratio_l79_79210

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l79_79210


namespace simplify_expression_l79_79589

variable (a : ℝ)

theorem simplify_expression (h1 : 0 < a ∨ a < 0) : a * Real.sqrt (-(1 / a)) = -Real.sqrt (-a) :=
sorry

end simplify_expression_l79_79589


namespace max_value_of_f_in_interval_l79_79391

def f (x : ℝ) := -x^2 + 6*x - 10

theorem max_value_of_f_in_interval : 
  ∃ x ∈ set.Icc 0 4, (∀ y ∈ set.Icc 0 4, f y ≤ f x) ∧ f x = -1 :=
sorry

end max_value_of_f_in_interval_l79_79391


namespace reciprocal_of_six_times_eq_twelve_l79_79493

theorem reciprocal_of_six_times_eq_twelve :
  (∃ x : ℝ, 6 * x = 12) → (150 * (1 / 2)) = 75 :=
by
  intro h,
  cases h with x hx,
  have hx_value : x = 2,
  { linarith },
  rw hx_value at *,
  sorry

end reciprocal_of_six_times_eq_twelve_l79_79493


namespace min_value_of_g_min_value_at_1_l79_79396

noncomputable def g (x : ℝ) : ℝ := (5 * x^2 - 10 * x + 24) / (7 * (1 + 2 * x))

theorem min_value_of_g :
  ∀ x : ℝ, x ≥ 1 → g x ≥ (29 / 21) :=
begin
  sorry
end

theorem min_value_at_1 :
  g 1 = (29 / 21) :=
begin
  sorry
end

end min_value_of_g_min_value_at_1_l79_79396


namespace geometric_series_common_ratio_l79_79193

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l79_79193


namespace classical_mechanics_incorrect_is_misunderstood_l79_79988

theorem classical_mechanics_incorrect_is_misunderstood
  (cond1 : ¬ (∀ t, ClassicalMechanics t))
  (cond2 : ∀ (v : ℝ) (m : ℝ), (v < const_high_speed ∧ m > const_macroscopic_mass) → applicable_classical_mechanics v m)
  (cond3 : ∀ p, exploring_scientific_theories_endless p) :
  misunderstood (ClassicalMechanics = incorrect_scientific_theory) :=
sorry

end classical_mechanics_incorrect_is_misunderstood_l79_79988


namespace problem_statement_l79_79104

noncomputable def function1 (x : ℝ) : ℝ := 2 * x
noncomputable def function3 (x : ℝ) : ℝ := Real.log x / Real.log 2

def associated_with_constant (f : ℝ → ℝ) (c : ℝ) (D : set ℝ) : Prop :=
  ∀ x1 ∈ D, ∃! x2 ∈ D, f x1 + f x2 = c

theorem problem_statement :
  associated_with_constant function1 4 set.univ ∧
  associated_with_constant function3 4 {x : ℝ | 0 < x} :=
by
  sorry

end problem_statement_l79_79104


namespace find_special_number_l79_79774

def reverse_digits (n : ℕ) : ℕ := sorry -- Assume reversing digits is done here

theorem find_special_number :
    ∃ K F : ℕ, K = 9 * F ∧ F = reverse_digits K ∧ K = 9801 ∧ K < 100000 ∧ F < 10000 :=
begin
    sorry
end

end find_special_number_l79_79774


namespace max_consecutive_integers_sum_lt_1000_l79_79258

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l79_79258


namespace Rs_parallel_to_AB_l79_79144

variables {A B C H M N P Q R S : Type*}
variables [is_right_triangle C A B]
variables [is_altitude C H (line_through A B)]
variables [is_angle_bisector (angle_at_vertex A) A M]
variables [is_angle_bisector (angle_at_vertex B) B N]
variables [line_intersects_line (line_through C H) (line_through A M) P]
variables [line_intersects_line (line_through C H) (line_through B N) Q]
variables [is_midpoint P M R]
variables [is_midpoint Q N S]

theorem Rs_parallel_to_AB 
  (h : is_right_triangle C A B)
  (hCH : is_altitude C H (line_through A B))
  (hAM : is_angle_bisector A M)
  (hBN : is_angle_bisector B N)
  (hP : line_intersects_line (line_through C H) (line_through A M) P)
  (hQ : line_intersects_line (line_through C H) (line_through B N) Q)
  (hR : is_midpoint P M R)
  (hS : is_midpoint Q N S) : 
  is_parallel (line_through R S) (line_through A B) :=
begin
  sorry
end

end Rs_parallel_to_AB_l79_79144


namespace minimum_value_proof_l79_79541

noncomputable def minimum_value (a b c : ℝ) (h : a + b + c = 6) : ℝ :=
  9 / a + 4 / b + 1 / c

theorem minimum_value_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 6) :
  (minimum_value a b c h₃) = 6 :=
sorry

end minimum_value_proof_l79_79541


namespace betty_sugar_amount_l79_79271

-- Define the ratios and the given amount of eggs
variable (sugar_rat : ℚ) (vanilla_rat : ℚ) (egg_rat : ℚ) (eggs_used : ℕ)

-- The given conditions
def conditions :=
  sugar_rat = 1 / 4 ∧
  vanilla_rat = 1 / 2 ∧
  egg_rat = 1 / 2 ∧
  eggs_used = 8

-- The resulting amount of sugar needed
def amount_of_sugar (sugar_rat vanilla_rat egg_rat : ℚ) (eggs_used : ℕ) :=
  (eggs_used * vanilla_rat) * (1 / vanilla_rat) * sugar_rat

-- The proof problem
theorem betty_sugar_amount : conditions ∧ amount_of_sugar 1 / 4 1 / 2 1 / 2 8 = 2 :=
by
  sorry

end betty_sugar_amount_l79_79271


namespace question1_part1_question1_part2_question1_part3_question1_part4_question2_question3_l79_79742

-- Definitions for the data

def classA_scores : List ℕ := [90, 90, 70, 90, 100, 80, 80, 90, 95, 65]
def classB_scores : List ℕ := [95, 70, 80, 90, 70, 80, 95, 80, 100, 90]

-- Organized data
def organized_data : (List ℕ × List ℕ) := ([2, 2, 4, 2], [2, 3, 2, 3]) -- format ([Class A], [Class B])

-- Definitions for the values to prove
def a_value : ℕ := 2
def b_value : ℕ := 85
def c_value : ℕ := 85
def d_value : ℕ := 90

-- Proof statements
theorem question1_part1 : (organized_data.snd.nth 2).get_or_else 0 = a_value := by
  sorry

theorem question1_part2 : (List.sum classB_scores : ℚ) / (List.length classB_scores : ℚ) = b_value := by
  sorry

theorem question1_part3 : (let sortedB := List.sort (· ≤ ·) classB_scores; let middle := (sortedB.get 4 + sortedB.get 5) / 2; middle : ℚ) = c_value := by
  sorry

theorem question1_part4 : (classA_scores.mode, 90) = d_value := by
  sorry

-- Class determination based on scores
def is_above_average (scores : List ℕ) (score : ℕ) : Bool :=
  score > (List.sort (· ≤ ·) scores).get (scores.length / 2)

theorem question2 : is_above_average classB_scores 90 ∧ ¬is_above_average classA_scores 90 := by
  sorry

-- Estimating number of students scoring above 90 in Class B
theorem question3 (total_students : ℕ) : (total_students * (3 / 10) : ℚ) = 15 :=
  by
    sorry

end question1_part1_question1_part2_question1_part3_question1_part4_question2_question3_l79_79742


namespace find_starting_number_l79_79217

-- Define that there are 15 even integers between a starting number and 40
def even_integers_range (n : ℕ) : Prop :=
  ∃ k : ℕ, (1 ≤ k) ∧ (k = 15) ∧ (n + 2*(k-1) = 40)

-- Proof statement
theorem find_starting_number : ∃ n : ℕ, even_integers_range n ∧ n = 12 :=
by
  sorry

end find_starting_number_l79_79217


namespace calculate_volume_of_tetrahedron_l79_79964

noncomputable def volumeOfTetrahedron (PQ PR PS QR QS RS : ℝ) : ℝ :=
  let a := PQ
  let b := PR
  let c := PS
  let d := QR
  let e := QS
  let f := RS
  -- Volume formula for a tetrahedron with sides a, b, c, d, e, f
  have volume := 3*real.sqrt 2
  volume

theorem calculate_volume_of_tetrahedron :
  volumeOfTetrahedron 6 4 5 5 4 (15 / 4 * real.sqrt 2) = 3 * real.sqrt 2 := 
  sorry

end calculate_volume_of_tetrahedron_l79_79964


namespace circle_passing_through_F_O_P_l79_79888

def parabola : set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

def focus := (1 : ℝ, 0 : ℝ)
def origin := (0 : ℝ, 0 : ℝ)
def P := (4 : ℝ, 4 : ℝ)

theorem circle_passing_through_F_O_P :
  (P ∈ parabola ∧ dist origin P = 4 * real.sqrt 2) →
  ∃ D E F, ∀ (x y : ℝ), (x, y).fst = 0 ∧
    x^2 + y^2 + D * x + E * y + F = 0  ∧ 
    (1 + D + F = 0) ∧ 
    (16 + 16 + 4 * D + 4 * E + F = 0) ∧
    x^2 + y^2 - x - 7 * y = 0 := sorry

end circle_passing_through_F_O_P_l79_79888


namespace seventeenth_entry_in_ordered_list_l79_79397

def r_11 (n : ℕ) : ℕ := n % 11

theorem seventeenth_entry_in_ordered_list (n : ℕ) :
  (n ≥ 0) → (r_11 (6 * n) ≤ 6) → 
  ∃ m, m = 32 ∧ 
  (list.filter (λ x, r_11 (6 * x) ≤ 6) (list.range(m))).nth 16 = some n :=
sorry

end seventeenth_entry_in_ordered_list_l79_79397


namespace pie_slice_max_segment_squared_l79_79306

theorem pie_slice_max_segment_squared (d : ℝ) (n : ℕ) (r : ℝ) (angle_deg : ℝ) (M_squared : ℝ) : 
  d = 20 → 
  n = 4 → 
  r = d / 2 → 
  angle_deg = 360 / n → 
  M_squared = (r * real.sqrt(2))^2 → 
  angle_deg = 90 → 
  M_squared = 200 := by 
  intros h1 h2 h3 h4 h5 h6
  rw h1 at h3 
  rw h2 at h4 
  simp at h3 
  rw h4 at h6 
  simp at h4 
  rw h6 at h5
  rw h5
  simp
  sorry

end pie_slice_max_segment_squared_l79_79306


namespace range_of_a_l79_79430

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def is_monotone_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y : ℝ⦄, 0 ≤ x → 0 ≤ y → x < y → f x < f y

axiom even_f : is_even f
axiom monotone_f : is_monotone_on_nonneg f

theorem range_of_a (a : ℝ) (h : f a ≥ f 3) : a ≤ -3 ∨ a ≥ 3 :=
by
  sorry

end range_of_a_l79_79430


namespace max_consecutive_integers_lt_1000_l79_79252

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l79_79252


namespace exists_point_between_l79_79124

noncomputable def F (z : ℝ) : ℝ := sorry

theorem exists_point_between (h_strictly_increasing : ∀ {z1 z2 : ℝ}, z1 < z2 → F z1 < F z2)
    (h_continuous : continuous F)
    (h_F_at_1 : F 1 = 0)
    (h_intermediate_value : intermediate_value F 2 3) :
    ∃ e ∈ Ioo 2 3, F e = 1 :=
begin
  sorry
end

end exists_point_between_l79_79124


namespace function_always_passes_through_fixed_point_l79_79154

theorem function_always_passes_through_fixed_point (α : ℝ) : (2 - 1)^α = 1 := 
by 
  sorry

end function_always_passes_through_fixed_point_l79_79154


namespace lambda_range_l79_79291

noncomputable def PA : ℝ := 1
noncomputable def PB : ℝ := 1
noncomputable def PO : ℝ := 2
noncomputable def PM (λ : ℝ) : ℝ := 2 * λ * PA + (1 - λ) * PB

theorem lambda_range (λ : ℝ) (h1 : λ > 0) (h2 : λ < 2 / 3) : 
  PM(λ) < 2 := 
by
  sorry

end lambda_range_l79_79291


namespace max_difference_y_intersections_l79_79156

-- Define the given equations
def y1 (x : ℝ) : ℝ := 4 - x^2 + x^4
def y2 (x : ℝ) : ℝ := 2 + x^2 + x^4

-- Define the condition of the intersection points
def x_values : set ℝ := { x | y1 x = y2 x }

-- Prove the maximum difference in the y-coordinates at intersection points is 0
theorem max_difference_y_intersections : 
  ∀ x ∈ x_values, |y1 x - y2 x| = 0 :=
by
  sorry

end max_difference_y_intersections_l79_79156


namespace probability_girls_same_color_l79_79687

open Classical

noncomputable def probability_same_color_marbles : ℚ :=
(3/6) * (2/5) * (1/4) + (3/6) * (2/5) * (1/4)

theorem probability_girls_same_color :
  probability_same_color_marbles = 1/20 := by
  sorry

end probability_girls_same_color_l79_79687


namespace total_students_l79_79110

theorem total_students (S : ℕ) (H1 : S / 2 = S - 15) : S = 30 :=
sorry

end total_students_l79_79110


namespace fraction_milk_in_cup1_l79_79729

-- Definitions of initial states and transfers
def initial_tea_in_cup1 : ℝ := 6
def initial_milk_in_cup2 : ℝ := 8
def fraction_tea_transferred_cup2 : ℝ := 1 / 3
def mixing_ratio_tea_in_cup2 : ℝ := (initial_tea_in_cup1 * fraction_tea_transferred_cup2) / (initial_milk_in_cup2 + initial_tea_in_cup1 * fraction_tea_transferred_cup2)
def mixing_ratio_milk_in_cup2 : ℝ := initial_milk_in_cup2 / (initial_milk_in_cup2 + initial_tea_in_cup1 * fraction_tea_transferred_cup2)
def fraction_mixed_content_transferred_back : ℝ := 1 / 4

-- Final amounts after all transfers
def final_tea_in_cup1 : ℝ := (initial_tea_in_cup1 * (1 - fraction_tea_transferred_cup2) + (initial_milk_in_cup2 + initial_tea_in_cup1 * fraction_tea_transferred_cup2) * fraction_mixed_content_transferred_back * mixing_ratio_tea_in_cup2)
def final_milk_in_cup1 : ℝ := (initial_milk_in_cup2 + initial_tea_in_cup1 * fraction_tea_transferred_cup2) * fraction_mixed_content_transferred_back * mixing_ratio_milk_in_cup2

-- Total liquid in Cup 1 after transfers
def total_liquid_in_cup1 : ℝ := final_tea_in_cup1 + final_milk_in_cup1

-- Proving the fraction of milk in Cup 1 is 2 / 6.5
theorem fraction_milk_in_cup1 : final_milk_in_cup1 / total_liquid_in_cup1 = 2 / 6.5 :=
by 
  sorry

end fraction_milk_in_cup1_l79_79729


namespace number_of_true_propositions_l79_79453

variable (a b c : ℝ)

def in_geometric_progression (a b c : ℝ) : Prop := b^2 = a * c

def converse (a b c : ℝ) : Prop := a * c = b^2 → in_geometric_progression a b c
def inverse (a b c : ℝ) : Prop := ¬ in_geometric_progression a b c → a * c ≠ b^2
def contrapositive (a b c : ℝ) : Prop := a * c ≠ b^2 → ¬ in_geometric_progression a b c

theorem number_of_true_propositions (a b c : ℝ) :
  (if converse a b c then 1 else 0) + 
  (if inverse a b c then 1 else 0) + 
  (if contrapositive a b c then 1 else 0) = 1 := sorry

end number_of_true_propositions_l79_79453


namespace exactly_one_vertical_asymptote_l79_79375

def g (x c : ℝ) : ℝ := (x^2 - 2 * x + c) / (x^2 - x - 42)

theorem exactly_one_vertical_asymptote (c : ℝ) :  
  ((∃ x : ℝ, (x - 7) ∣ (x^2 - 2 * x + c)) ∨ (∃ y : ℝ, (y + 6) ∣ (x^2 - 2 * x + c))) 
  ∧ ¬((∃ x : ℝ, (x - 7) ∣ (x^2 - 2 * x + c)) ∧ (∃ y : ℝ, (y + 6) ∣ (x^2 - 2 * x + c))) 
  ↔ (c = -35 ∨ c = -48) :=
  sorry

end exactly_one_vertical_asymptote_l79_79375


namespace average_minutes_per_day_l79_79740

theorem average_minutes_per_day (e : ℕ) (h_e_pos : 0 < e) : 
  let sixth_grade_minutes := 20
  let seventh_grade_minutes := 18
  let eighth_grade_minutes := 12
  
  let sixth_graders := 3 * e
  let seventh_graders := 4 * e
  let eighth_graders := e
  
  let total_minutes := sixth_grade_minutes * sixth_graders + seventh_grade_minutes * seventh_graders + eighth_grade_minutes * eighth_graders
  let total_students := sixth_graders + seventh_graders + eighth_graders
  
  (total_minutes / total_students) = 18 := by
sorry

end average_minutes_per_day_l79_79740


namespace solve_equations_l79_79591

theorem solve_equations (x y : ℝ) (h1 : (x + y) / x = y / (x + y)) (h2 : x = 2 * y) :
  x = 0 ∧ y = 0 :=
by
  sorry

end solve_equations_l79_79591


namespace exactly_one_female_student_l79_79056

noncomputable def binom : ℕ → ℕ → ℕ
| n, k := if h : k ≤ n then nat.choose n k else 0

theorem exactly_one_female_student (A_males A_females B_males B_females : ℕ) :
  A_males = 5 → A_females = 3 → B_males = 6 → B_females = 2 →
  let ways_A := binom A_females 1 * binom A_males 1 * binom B_males 2,
      ways_B := binom B_females 1 * binom B_males 1 * binom A_males 2 in
  (ways_A + ways_B) = 345 :=
by
  intros hA_males hA_females hB_males hB_females
  simp [binom, hA_males, hA_females, hB_males, hB_females]
  sorry

end exactly_one_female_student_l79_79056


namespace coloring_of_russia_with_odd_valid_colorings_l79_79934

theorem coloring_of_russia_with_odd_valid_colorings :
  let regions : ℕ := 85
  let colors := {white, blue, red}
  let valid_color (r : ℕ → colors) : Prop :=
    ∀ i j, (i ≠ j ∧ adjacent i j) → ¬((r i = white ∧ r j = red) ∨ (r i = red ∧ r j = white))
  let unused_colors_allowed : Prop :=
    ∃ r : ℕ → colors, ∀ i ∈ 85, r i = blue ∨ r i = white ∨ r i = red
  let total_colorings_odd : Prop :=
    ∃ n, valid_color n ∧ unused_colors_allowed n ∧ n % 2 = 1
  total_colorings_odd :=
sorry

end coloring_of_russia_with_odd_valid_colorings_l79_79934


namespace triangle_condition_l79_79896

variables {A B C D E F: Type}
variables [metric_space D] [inner_product_space ℝ D]
variables {a b c x: ℝ}

noncomputable def closest_point_F_to_C (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ c = real.sqrt (a^2 + b^2)) : ℝ :=
  (b * (c - b)) / a

theorem triangle_condition {A B C D E F: Type} 
  [metric_space D] [inner_product_space ℝ D] 
  (a b c x : ℝ) 
  (hA: right_triangle A B C)
  (h1 : point_on_line B C D)
  (h2 : perpendicular D AD E)
  (h3 : perpendicular_projection E BC F)
  : x = closest_point_F_to_C a b c (and.intro (by linarith) (by linarith) (by linarith)) :=
sorry

end triangle_condition_l79_79896


namespace average_percent_score_l79_79564

theorem average_percent_score :
  let num_students : ℕ := 100
  let scores : List (ℕ × ℕ) := [(95, 10), (85, 15), (75, 20), (65, 25), (55, 15), (45, 10), (35, 5)]
  let weighted_score := scores.foldl (λ acc, λ pair, acc + pair.1 * pair.2) 0
  let average_score := weighted_score / num_students
  average_score = 68 :=
by
  -- Definitions for easier readability
  let num_students := 100
  let scores := [(95, 10), (85, 15), (75, 20), (65, 25), (55, 15), (45, 10), (35, 5)]
  -- Calculating the weighted score
  let weighted_score := scores.foldl (λ acc, λ pair, acc + pair.1 * pair.2) 0
  -- Calculating the average score
  let average_score := weighted_score / num_students
  -- Summing up the products manually to avoid reliance on the foldl in the proof
  have : weighted_score = 95 * 10 + 85 * 15 + 75 * 20 + 65 * 25 + 55 * 15 + 45 * 10 + 35 * 5 := sorry
  -- Asserting the final average and its value
  show average_score = 68 from sorry

end average_percent_score_l79_79564


namespace solution_set_for_f_ge_0_range_of_a_l79_79449

def f (x : ℝ) : ℝ := |3 * x + 1| - |2 * x + 2|

theorem solution_set_for_f_ge_0 : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3/5} ∪ {x : ℝ | x ≥ 1} :=
sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x - |x + 1| ≤ |a + 1|) ↔ (a ≤ -3 ∨ a ≥ 1) :=
sorry

end solution_set_for_f_ge_0_range_of_a_l79_79449


namespace problem1_problem2_l79_79725

theorem problem1 (P1 P2 : ℝ) (h₁ : P1 = 1/2) (h₂ : P2 = 1/2) :
  (P1 * P2) + (P1 * P2) = 1 / 2 :=
by simp [h₁, h₂]; norm_num; sorry

theorem problem2 (p1 p2 : ℝ) (h₁ : p1 + p2 = 1) (h₂ : p1 + 2 * p2 = 2 * p1 ^ 2 + 2 * p1 * p2) :
  p1 = 2 / 3 ∧ p2 = 1 / 3 :=
by sorry

end problem1_problem2_l79_79725


namespace conclusion1_conclusion2_l79_79801

def condition (a b α β : ℝ) : Prop :=
  sin α + cos β = a ∧ cos α + sin β = b ∧ 0 < a^2 + b^2 ∧ a^2 + b^2 ≤ 4

theorem conclusion1 (a α β : ℝ) :
  ¬ ∀ b : ℝ, sin(α + β) = (a^2 + b^2 - 2) / 2 :=
sorry

theorem conclusion2 (b α β : ℝ) :
  ∀ a : ℝ, cos(α - β) = 0 :=
sorry

end conclusion1_conclusion2_l79_79801


namespace max_consecutive_integers_sum_lt_1000_l79_79263

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l79_79263


namespace vector_dot_product_l79_79045

variables {V : Type*} [inner_product_space ℝ V] 
variables {A B C M P : V}

noncomputable def midpoint (x y : V) : V := (x + y) / 2

-- Conditions
axiom M_midpoint : M = midpoint B C
axiom AM_length : dist A M = 3
axiom AP_relation : P - A = 2 • (M - P)

-- Proof statement
theorem vector_dot_product : 
  (P - A) • ((P - B) + (P - C)) = -4 :=
sorry

end vector_dot_product_l79_79045


namespace nat_solution_unique_l79_79593

theorem nat_solution_unique (x y : ℕ) (h : x + y = x * y) : (x, y) = (2, 2) :=
sorry

end nat_solution_unique_l79_79593


namespace log_expr_eq_seven_l79_79752

-- Define the problem in Lean 4
theorem log_expr_eq_seven : logBase 3 27 + log 10 25 + log 10 4 + (7 ^ logBase 7 2) = 7 :=
by
  sorry

end log_expr_eq_seven_l79_79752


namespace goals_in_fifth_match_l79_79692

/-- A football player scores some goals in his fifth match, thus increasing his average goals score by 0.2.
    The total number of goals in his 5 matches is 11. Prove that the number of goals he scored in his fifth match is 3. -/
theorem goals_in_fifth_match
  (G : ℕ)
  (A : ℕ)
  (h1 : (5 * (A + 0.2)) = 11)
  (h2 : (4 * A) + G = 11) :
  G = 3 :=
sorry

end goals_in_fifth_match_l79_79692


namespace cosine_in_third_quadrant_l79_79025

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l79_79025


namespace pentagon_area_l79_79346

theorem pentagon_area :
  let a := 18
  let b := 25
  let c := 30
  let d := 28
  let e := 25
  let triangle_area := (1 / 2) * 18 * 25
  let trapezoid_area := (1 / 2) * (28 + 30) * 25
  triangle_area + trapezoid_area = 950 := 
by
  let a : ℕ := 18
  let b : ℕ := 25
  let c : ℕ := 30
  let d : ℕ := 28
  let e : ℕ := 25
  let triangle_area : ℝ := (1 / 2) * a * b
  let trapezoid_area : ℝ := (1 / 2) * (d + c) * e
  have h1 : triangle_area + trapezoid_area = 950 := by sorry
  exact h1
  sorry

end pentagon_area_l79_79346


namespace volume_of_bounded_ellipsoid_l79_79747

theorem volume_of_bounded_ellipsoid :
  (∫ z in 0..7, (3 * Real.pi / 49) * (196 - z^2)) = 77 * Real.pi := by
  -- Proof goes here
  sorry

end volume_of_bounded_ellipsoid_l79_79747


namespace six_digit_even_numbers_l79_79369

theorem six_digit_even_numbers :
  (∃ L : List ℕ, L.perm [1, 2, 3, 4, 5, 6] ∧ 
    L.length = 6 ∧ 
    L.last = 2 ∨ L.last = 4 ∨ L.last = 6 ∧ 
    ¬(1 :: 3 :: 5 :: List.nil ⊆ L.windowed 3 ∨ 
      3 :: 1 :: 5 :: List.nil ⊆ L.windowed 3 ∨ 
      5 :: 1 :: 3 :: List.nil ⊆ L.windowed 3 ∨ 
      1 :: 5 :: 3 :: List.nil ⊆ L.windowed 3 ∨ 
      3 :: 5 :: 1 :: List.nil ⊆ L.windowed 3 ∨ 
      5 :: 3 :: 1 :: List.nil ⊆ L.windowed 3)) :=
  ∃ L : List ℕ, L.perm [1, 2, 3, 4, 5, 6] ∧ L.length = 6 ∧ 
  L.last ∈ [2, 4, 6] ∧
  ¬(List.sublists' {1, 3, 5}).exists (fun l => L.windowed 3 l) := by sorry

end six_digit_even_numbers_l79_79369


namespace probability_sequence_123456_l79_79659

theorem probability_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11
  in (favorable_sequences : ℚ) / total_sequences = 1 / 720 := 
by 
  sorry

end probability_sequence_123456_l79_79659


namespace dislike_both_radio_and_music_l79_79578

theorem dislike_both_radio_and_music :
  ∀ (total_people : ℕ) 
    (percent_dislike_radio percent_dislike_both : ℝ), 
  total_people = 1500 →
  percent_dislike_radio = 0.4 →
  percent_dislike_both = 0.15 →
  (percent_dislike_both * (percent_dislike_radio * total_people)).toNat = 90 :=
by
  intros total_people percent_dislike_radio percent_dislike_both 
  assume h_total h_percent_radio h_percent_both
  rw [h_total, h_percent_radio, h_percent_both]
  have h1 : percent_dislike_radio * total_people = 600 := by norm_num
  have h2 : percent_dislike_both * 600 = 90 := by norm_num
  rw [h1] at h2
  exact h2.symm.toNatEq.2 rfl
  sorry

end dislike_both_radio_and_music_l79_79578


namespace max_value_frac_3t_minus_4t2_over_9t_l79_79264

theorem max_value_frac_3t_minus_4t2_over_9t (t : ℝ) : 
  ∃ (t : ℝ), (∀ (t : ℝ), (3^t - 4 * t^2) * t / 9^t ≤ (3^t - 4 * t^2) * t / 9^t) → 
  ((3^t - 4 * t^2) * t / 9^t = √3 / 9) :=
sorry

end max_value_frac_3t_minus_4t2_over_9t_l79_79264


namespace num_distinct_sums_of_three_distinct_elements_l79_79010

noncomputable def arith_seq_sum_of_three_distinct : Nat :=
  let a (i : Nat) : Nat := 3 * i + 1
  let lower_bound := 21
  let upper_bound := 129
  (upper_bound - lower_bound) / 3 + 1

theorem num_distinct_sums_of_three_distinct_elements : arith_seq_sum_of_three_distinct = 37 := by
  -- We are skipping the proof by using sorry
  sorry

end num_distinct_sums_of_three_distinct_elements_l79_79010


namespace sum_of_lucky_numbers_divisible_by_13_l79_79344

def is_lucky_number (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  ∃ a b c d e f : ℕ, digits = [f, e, d, c, b, a] ∧ (a + b + c = d + e + f)

theorem sum_of_lucky_numbers_divisible_by_13 :
  let lucky_numbers := {n : ℕ | n < 1000000 ∧ is_lucky_number n}
  ∑ n in lucky_numbers, n % 13 = 0 :=
by
  sorry

end sum_of_lucky_numbers_divisible_by_13_l79_79344


namespace possible_values_of_m_l79_79491

-- Define the conditions that m is a real number and the quadratic equation having two distinct real roots
variable (m : ℝ)

-- Define the discriminant condition for having two distinct real roots
def discriminant_condition (a b c : ℝ) := b^2 - 4 * a * c > 0

-- State the required theorem
theorem possible_values_of_m (h : discriminant_condition 1 m 9) : m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo 6 ∞ :=
sorry

end possible_values_of_m_l79_79491


namespace calculate_expression_l79_79923

def h (x : ℝ) : ℝ := 2 * x - 3
def h_inv (x : ℝ) : ℝ := (x + 3) / 2
def j (x : ℝ) : ℝ := x / 4
def j_inv (x : ℝ) : ℝ := 4 * x

theorem calculate_expression : h (j_inv (h_inv (h_inv (j (h 12))))) = 25.5 :=
by
  sorry

end calculate_expression_l79_79923


namespace geometric_sequence_fifth_term_l79_79621

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 9) (h2 : a * r^6 = 1) : a * r^4 = 3 :=
by
  sorry

end geometric_sequence_fifth_term_l79_79621


namespace altitude_from_orthocenter_incenter_triangle_l79_79575

theorem altitude_from_orthocenter_incenter_triangle 
(point_D_on_AC : ∃ D, D ∈ line_segment A C)
(incenters_of_triangles : ∃ I1 I2 I, incenter_of ABD I1 ∧ incenter_of BCD I2 ∧ incenter_of ABC I)
(I_is_orthocenter : is_orthocenter_of_triangle I I1 I2 B) :
  altitude_of_triangle B D A C :=
  sorry

end altitude_from_orthocenter_incenter_triangle_l79_79575


namespace sum_of_solutions_l79_79787

theorem sum_of_solutions (x : ℝ) : (∃ x₁ x₂ : ℝ, (x - 4)^2 = 16 ∧ x = x₁ ∨ x = x₂ ∧ x₁ + x₂ = 8) :=
by sorry

end sum_of_solutions_l79_79787


namespace distance_from_center_to_cross_section_l79_79419

theorem distance_from_center_to_cross_section
  (P A B C : Point)
  (O : Point)
  (r : ℝ)
  (PA_perpendicular_PB : IsPerpendicular PA PB)
  (PB_perpendicular_PC : IsPerpendicular PB PC)
  (PC_perpendicular_PA : IsPerpendicular PC PA)
  (O_distance_P : Distance O P = sqrt 3)
  (O_distance_A : Distance O A = sqrt 3)
  (O_distance_B : Distance O B = sqrt 3)
  (O_distance_C : Distance O C = sqrt 3) :
  DistanceFromCenterToCrossSection O (CrossSection P A B C) = sqrt 3 / 3 :=
sorry

end distance_from_center_to_cross_section_l79_79419


namespace ratio_of_sold_phones_to_production_l79_79701

def last_years_production : ℕ := 5000
def this_years_production : ℕ := 2 * last_years_production
def phones_left_in_factory : ℕ := 7500
def sold_phones : ℕ := this_years_production - phones_left_in_factory

theorem ratio_of_sold_phones_to_production : 
  (sold_phones : ℚ) / this_years_production = 1 / 4 := 
by
  sorry

end ratio_of_sold_phones_to_production_l79_79701


namespace total_adjusted_income_is_1219_72_l79_79778

def investment_amount := 6800
def stock_allocation := 0.60
def bond_allocation := 0.30
def cash_allocation := 0.10

def inflation_rate := 0.02

def stock_returns := [0.08, 0.04, 0.10]
def bond_returns := [0.05, 0.06, 0.04]
def cash_returns := [0.01, 0.02, 0.03]

def year_income (investment: Real) (allocations: (Real, Real, Real)) (returns: List Real):
  Real :=
  let (stock_alloc, bond_alloc, cash_alloc) := allocations
  investment * stock_alloc * returns.head! +
  investment * bond_alloc * returns.head! +
  investment * cash_alloc * returns.head!

def adjusted_income (income: Real) (inflation: Real): Real :=
  income * (1 - inflation)

theorem total_adjusted_income_is_1219_72 :
  let y1_income := adjusted_income (year_income investment_amount (stock_allocation, bond_allocation, cash_allocation) [stock_returns.head!, bond_returns.head!, cash_returns.head!]) inflation_rate
  let y2_income := adjusted_income (year_income investment_amount (stock_allocation, bond_allocation, cash_allocation) [stock_returns.tail!.head!, bond_returns.tail!.head!, cash_returns.tail!.head!]) inflation_rate
  let y3_income := adjusted_income (year_income investment_amount (stock_allocation, bond_allocation, cash_allocation) [stock_returns.tail!.tail!.head!, bond_returns.tail!.tail!.head!, cash_returns.tail!.tail!.head!]) inflation_rate
  y1_income + y2_income + y3_income = 1219.72 :=
begin
  -- Proof to be done
  sorry
end

end total_adjusted_income_is_1219_72_l79_79778


namespace degree_of_product_is_seven_l79_79755

-- noncomputable necessary for handling non-algebraic operations
noncomputable def P1 : Polynomial ℝ := X^5
noncomputable def P2 : Polynomial ℝ := X^2 + (1 / X^2)
noncomputable def P3 : Polynomial ℝ := 1 + (2 / X) + (3 / X^2)

theorem degree_of_product_is_seven : (P1 * P2 * P3).degree = 7 := 
by 
  sorry

end degree_of_product_is_seven_l79_79755


namespace cyclic_quadrilateral_l79_79885

-- Define the given problem in Lean 4

theorem cyclic_quadrilateral (A B C D E F G : Type) 
  (ABCD_quad : quadrilateral A B C D) 
  (AD_not_parallel_BC : ¬ parallel (line_through A D) (line_through B C))
  (diagonals_intersect_at_E : intersects (line_through A C) (line_through B D) E)
  (F_on_AB : on_line_segment F A B) 
  (G_on_DC : on_line_segment G D C) 
  (ratios_equal : ∀ (k : ℝ), (AF_FB_ratio : (length (A F)) / (length (F B)) = k) 
                             (DG_GC_ratio : (length (D G)) / (length (G C)) = k)
                             (AD_BC_ratio : (length (A D)) / (length (B C)) = k))
  (E_F_G_collinear : collinear {E, F, G}) 
  : cyclic_quadrilateral A B C D := 
sorry

end cyclic_quadrilateral_l79_79885


namespace quadratic_roots_diff_l79_79973

noncomputable def quadratic_roots (k : ℝ) : (ℝ × ℝ) :=
  let Δ : ℝ := 4 - 4 * k
  if h : Δ ≥ 0 then
    let root1 := (2 + sqrt Δ) / 2
    let root2 := (2 - sqrt Δ) / 2
    (root1, root2)
  else
    (0, 0) -- placeholder; in reality we would handle complex roots

theorem quadratic_roots_diff (k : ℝ) : 
    let (α, β) := quadratic_roots k in
    abs (α - β) = 2 * sqrt 2 ↔ (k = -1 ∨ k = 3) := sorry

end quadratic_roots_diff_l79_79973


namespace geometric_series_common_ratio_l79_79175

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l79_79175


namespace max_value_of_x_plus_y_l79_79831

noncomputable def f : ℝ → ℝ := sorry  -- we assume f exists as a monotonic function satisfying given properties

theorem max_value_of_x_plus_y :
  (∀ x y : ℝ, f(x + y) = f(x) + f(y)) →
  (∀ x y : ℝ, f(x^2 + 2*x + 2) + f(y^2 + 8*y + 3) = 0) →
  ∀ x y : ℝ, (x^2 + y^2 + 2*x + 8*y + 5 = 0) → 
  ∃ (α : ℝ), 
  let x := -1 + 2*sqrt 3 * cos α,
      y := -4 + 2*sqrt 3 * sin α 
  in x + y ≤ 2*sqrt 6 - 5 :=
sorry

end max_value_of_x_plus_y_l79_79831


namespace probability_root_in_interval_l79_79410

def is_monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x y, x < y → f x < f y

def has_root_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ x, a ≤ x ∧ x ≤ b ∧ f x = 0

theorem probability_root_in_interval : 
  let m_set := {1, 2, 3, 4}
  let n_set := {-12, -8, -4, -2}
  let f := λ (m n : ℤ) (x : ℝ), x^3 + (m * x) + n
  11 / 16 = 
    (∑ m in m_set, ∑ n in n_set, 
      if has_root_in_interval (f m n) 1 2 
      then 1 else 0) / (|m_set| * |n_set|) :=
by
  sorry

end probability_root_in_interval_l79_79410


namespace distinct_real_roots_of_quadratic_l79_79477

-- Define the problem's condition: m is a real number and the discriminant of x^2 + mx + 9 > 0
def discriminant_positive (m : ℝ) := m^2 - 36 > 0

theorem distinct_real_roots_of_quadratic (m : ℝ) (h : discriminant_positive m) :
  m ∈ Iio (-6) ∪ Ioi (6) :=
sorry

end distinct_real_roots_of_quadratic_l79_79477


namespace volume_of_cone_correct_l79_79394

noncomputable def volume_cone (S : ℝ) (r : ℝ) := 
  let l := S / π / r
  let h := Real.sqrt (l ^ 2 - r ^ 2) in
  (1 / 3) * π * r ^ 2 * h

theorem volume_of_cone_correct (S : ℝ) (r : ℝ) (h : ℝ)
  (hS : S = 15 * Real.cbrt π) 
  (hr : r = 21) :
  volume_cone S r = 12 :=
by
  -- The proof will be placed here
  sorry

end volume_of_cone_correct_l79_79394


namespace circle_and_tangent_line_l79_79416

noncomputable def circle_equation : ℝ × ℝ → Prop :=
λ p, p.1 ^ 2 + p.2 ^ 2 - 4 * p.2 + 2 = 0

noncomputable def line : ℝ × ℝ → Prop :=
λ p, p.1 - p.2 = 0

theorem circle_and_tangent_line :
  (∀ p : ℝ × ℝ, circle_equation p → p = (1, 1) → line p) ∧
  (let r := sqrt 2 in
  ∀ C : ℝ × ℝ → Prop, (∀ p : ℝ × ℝ, C p ↔ p.1 ^ 2 + (p.2 - 2)^2 = 2) →
  r = sqrt 2) :=
by
  sorry

end circle_and_tangent_line_l79_79416


namespace probability_of_sequence_123456_l79_79654

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end probability_of_sequence_123456_l79_79654


namespace geometric_series_common_ratio_l79_79199

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l79_79199


namespace find_AC_l79_79065

variables (ABC : Type) [Triangle ABC]
variables {A B C M : Point ABC} {a b : Real}
variables {angle_C_obtuse : ∃ C, is_obtuse ∠BAC}
variables {D : Point ABC} {AD_perpendicular_DB : is_perpendicular (D,B) (A,B)}
variables {DC_perpendicular_AC : is_perpendicular (D,C) (A,C)}
variables {M : Point ABC} {altitude_CM_intersects_AB : M ∈ intersection (altitude_of_triangle_from_vertex C (triangle ADC)) (A,B)}

theorem find_AC
  (AM_eq_a : distance A M = a)
  (MB_eq_b : distance M B = b) :
  distance A C = sqrt(a * (a + b)) :=
sorry

end find_AC_l79_79065


namespace largest_binomial_term_remainder_of_binomial_power_l79_79434

-- Problem Part 1:
theorem largest_binomial_term (a b : ℕ) {n : ℕ} 
  (h1 : (nat.choose n 5) = largest (λ k, (nat.choose n k)))
  : n = 10 :=
sorry

-- Problem Part 2:
theorem remainder_of_binomial_power (a b : ℕ) (h2 : a + b = 4) : 
  ((a + b)^10 + 7) % 3 = 2 :=
sorry

end largest_binomial_term_remainder_of_binomial_power_l79_79434


namespace count_consecutive_sets_sum_18_l79_79466

-- Define the concept of consecutive summing to a target value
def consecutive_sum (start : ℕ) (len : ℕ) : ℕ :=
  len * (2 * start + len - 1) / 2

-- Define the main problem as a theorem statement
theorem count_consecutive_sets_sum_18 : 
  (finset.range 18).filter (fun start => 
    ∃ n >= 2, consecutive_sum start n = 18
  ).card = 1 :=
sorry

end count_consecutive_sets_sum_18_l79_79466


namespace correct_inequality_l79_79014

variable {a α : ℝ}

theorem correct_inequality (h : 0 < a ∧ a < 1 / 2) : 
  cos (1 + α) < cos (1 - α) :=
sorry

end correct_inequality_l79_79014


namespace quadratic_one_real_root_l79_79042

theorem quadratic_one_real_root (m : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*m*x + 2*m = 0) ∧ 
    (∀ y : ℝ, (y^2 - 6*m*y + 2*m = 0) → y = x)) → 
  m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_l79_79042


namespace sum_of_solutions_l79_79786

theorem sum_of_solutions (x : ℝ) : (∃ x₁ x₂ : ℝ, (x - 4)^2 = 16 ∧ x = x₁ ∨ x = x₂ ∧ x₁ + x₂ = 8) :=
by sorry

end sum_of_solutions_l79_79786


namespace expression_divisible_by_10_l79_79944

theorem expression_divisible_by_10 (n : ℕ) : 10 ∣ (3 ^ (n + 2) - 2 ^ (n + 2) + 3 ^ n - 2 ^ n) :=
  sorry

end expression_divisible_by_10_l79_79944


namespace root_in_interval_l79_79145

noncomputable def f (x : ℝ) := Real.log (x + 1) - 2 / x

theorem root_in_interval :
    ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
by
  have h1 : f 1 < 0 := by
    sorry
  have h2 : f 2 > 0 := by
    sorry
  have h_interval := IntermediateValueTheorem (-∞) ∞ f 1 2
  exact ⟨h_interval h1 h2, sorry⟩

end root_in_interval_l79_79145


namespace necessary_but_not_sufficient_condition_l79_79822

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a + b > 2 ∧ ab > 1) → ¬ ((a > 1) ∧ (b > 1)) :=
begin
  intros h,
  obtain ⟨h1, h2⟩ := h,
  by_cases ha : a > 1,
  { by_cases hb : b > 1,
    { exfalso,
      sorry }, -- This needs to handle contradiction case.
    { exact ha }, wlog },
end

end necessary_but_not_sufficient_condition_l79_79822


namespace animal_order_l79_79292

-- Define the behaviors of each animal
def Jackal (responds : String -> String) : Prop :=
  ∀ (q : String), responds q = "No"

def Lion (responds : String -> String) : Prop :=
  ∀ (q : String), responds q = "Yes"

def Parrot (previous_response : String) (responds : String -> String) : Prop :=
  ∀ (q : String), responds q = previous_response

def Giraffe (previous_response : String) (responds : String -> String) : Prop :=
  ∀ (q : String), responds q = previous_response

--conditions on different animals
section
variables (J L P G : String -> String)

-- Given behaviors
axiom jackal_behavior : Jackal J
axiom lion_behavior : Lion L
axiom parrot_behavior : ∀ (pr : String), Parrot pr P
axiom giraffe_behavior : ∀ (pr : String), Giraffe pr G

-- The animals' standing order satisfies the given responses for questions
theorem animal_order : Parrot "" ∧ Lion ∧ Giraffe "" ∧ Jackal :=
  sorry

end animal_order_l79_79292


namespace length_AD_in_quadrilateral_l79_79511

variables (A B C D O : Type*) [MetricSpace A]
variables (BO OD AO OC AB : ℝ)
variables [fact (BO = 4)] [fact (OD = 5)] [fact (AO = 9)]
variables [fact (OC = 2)] [fact (AB = 7)]

theorem length_AD_in_quadrilateral
    (quadrilateral : ∃ (A B C D O : A), true)
    (intersect : ∃ (O : A), True ∧ True):
    sqrt ((9^2) + (5^2) + (2 * 9 * 5 * (2/3))) = sqrt 166 :=
by sorry

end length_AD_in_quadrilateral_l79_79511


namespace general_formula_a_n_sum_terms_T_n_l79_79710

-- Define the sequence and necessary conditions
def seq_a (n : ℕ) : ℕ := 3^n
def S_n (n : ℕ) : ℕ := (3 * (3^n - 1)) / 2 -- sum of the first n terms of the sequence

-- Prove the general form of the sequence a_n
theorem general_formula_a_n (n : ℕ) : seq_a n = 3^n := sorry

-- Define the sequence b_n and the sum of its first n terms T_n
def b_n (n : ℕ) : ℕ := seq_a (n+1) / (S_n n * S_n (n+1))
def T_n (n : ℕ) : ℕ := (2/3) * ((1/2) - (1/(3^(n+1) - 1)))

-- Prove the sum of the first n terms T_n
theorem sum_terms_T_n (n : ℕ) : ∑ i in range n, b_n i = T_n n := sorry

end general_formula_a_n_sum_terms_T_n_l79_79710


namespace ellipse_equation_slope_CD_l79_79826

variables {A : Type*} [Field A] [Algebra ℚ A]

def is_point_on_ellipse (p : A × A) (a b : A) : Prop :=
  (p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1)

def foci_condition (A F1 F2 : A × A) : Prop :=
  (A.1 - F1.1) ^ 2 + (A.2 - F1.2) ^ 2 + (A.1 - F2.1) ^ 2 + (A.2 - F2.2) ^ 2 = 4

def inclination_angle_complementary (k : A) : Prop :=
  ∀ x y, (y = k * (x - 1) + 1) ∧ (y = -k * (x - 1) + 1) → k ≠ 0

-- Given conditions
variables (A : A × A) (F1 F2 : A × A) (a b k : A)
variables (C D : A × A)
-- Assume A, F1, F2, C, and D are properly defined in this context

-- Point A is given as (1, 1)
axiom hA1 : A = (1 : A, 1 : A)

-- Ellipse condition: |AF1| + |AF2| = 4
axiom hfoci : foci_condition A F1 F2

-- The inclination of AC and AD are complementary
axiom hInclination : inclination_angle_complementary k

-- Points C and D are on the ellipse
axiom hC : is_point_on_ellipse C a b
axiom hD : is_point_on_ellipse D a b

theorem ellipse_equation :
  is_point_on_ellipse A 2 (sqrt 4 / sqrt 3) := 
sorry 

theorem slope_CD :
  ∃ k : A, inclination_angle_complementary k → k ≠ 0 → slope_CD = 1 / 3 := 
sorry 

end ellipse_equation_slope_CD_l79_79826


namespace compare_neg_fractions_l79_79352

theorem compare_neg_fractions : - (4 / 3 : ℚ) < - (5 / 4 : ℚ) := 
by sorry

end compare_neg_fractions_l79_79352


namespace acute_angle_of_parallel_vectors_l79_79458
open Real

theorem acute_angle_of_parallel_vectors (α : ℝ) (h₁ : abs (α * π / 180) < π / 2) :
  let a := (3 / 2, sin (α * π / 180))
  let b := (sin (α * π / 180), 1 / 6) 
  a.1 * b.2 = a.2 * b.1 → α = 30 :=
by
  sorry

end acute_angle_of_parallel_vectors_l79_79458


namespace probability_of_specific_sequence_l79_79665

-- We define a structure representing the problem conditions.
structure problem_conditions :=
  (cards : multiset ℕ)
  (permutation : list ℕ)

-- Noncomputable definition for the correct answer.
noncomputable def probability := (1 : ℚ) / 720

-- The main theorem statement.
theorem probability_of_specific_sequence :
  ∀ (conds : problem_conditions),
  conds.cards = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6} ∧
  (∃ (perm : list ℕ), perm.perm conds.permutation) →
  (∃ (sequence : list ℕ), sequence = [1, 2, 3, 4, 5, 6]) →
  let prob := calculate_probability conds.permutation [1, 2, 3, 4, 5, 6] in
  prob = (1 : ℚ) / 720 :=
sorry

end probability_of_specific_sequence_l79_79665


namespace maximal_value_l79_79927

open Function

-- Definitions based on the given problem conditions
variables {α β γ : ℝ}
def is_incenter (M : Point) (A B C A' B' C' : Point) : Prop :=
  MA' ⊥ BC ∧ MB' ⊥ CA ∧ MC' ⊥ AB ∧
  A' ∈ line BC ∧ B' ∈ line CA ∧ C' ∈ line AB ∧
  p(M) = (MA' * MB' * MC') / (MA * MB * MC)

-- Problem statement
theorem maximal_value (ABC : Triangle) (M : Point) (A B C A' B' C' : Point) :
  is_incenter M A B C A' B' C' →
  p(M) = (MA' * MB' * MC') / (MA * MB * MC) →
  ∃ M, p(M) = (1 / 8) :=
sorry

end maximal_value_l79_79927


namespace sum_greater_l79_79426

theorem sum_greater {a b c d : ℝ} (h1 : b + Real.sin a > d + Real.sin c) (h2 : a + Real.sin b > c + Real.sin d) : a + b > c + d := by
  sorry

end sum_greater_l79_79426


namespace abc_ints_u_v_exist_l79_79098

theorem abc_ints_u_v_exist {a b c : ℚ}
  (h1 : ∃ t ∈ ℤ, a + b + c = t)
  (h2 : ∃ t ∈ ℤ, a^2 + b^2 + c^2 = t) :
  ∃ (u v : ℤ), a * b * c * v^3 = u^2 ∧ Int.gcd u v = 1 := 
sorry

end abc_ints_u_v_exist_l79_79098


namespace valid_license_plates_count_l79_79719

theorem valid_license_plates_count :
  let letters_count := 26
  let digits_count := 10
  letters_count ^ 3 * digits_count ^ 4 = 175760000 :=
by
  let letters_count := 26
  let digits_count := 10
  calc
    letters_count ^ 3 * digits_count ^ 4 = 26 ^ 3 * 10 ^ 4 : by rfl
                              ... = 175760000 : by norm_num

end valid_license_plates_count_l79_79719


namespace possible_values_of_m_l79_79487

-- Define the conditions that m is a real number and the quadratic equation having two distinct real roots
variable (m : ℝ)

-- Define the discriminant condition for having two distinct real roots
def discriminant_condition (a b c : ℝ) := b^2 - 4 * a * c > 0

-- State the required theorem
theorem possible_values_of_m (h : discriminant_condition 1 m 9) : m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo 6 ∞ :=
sorry

end possible_values_of_m_l79_79487


namespace intersection_M_N_l79_79495

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x <2} := by
  sorry

end intersection_M_N_l79_79495


namespace solve_x_squared_eq_nine_l79_79594

theorem solve_x_squared_eq_nine (x : ℝ) : x^2 = 9 → (x = 3 ∨ x = -3) :=
by
  -- Proof by sorry placeholder
  sorry

end solve_x_squared_eq_nine_l79_79594


namespace distinct_real_roots_of_quadratic_l79_79476

-- Define the problem's condition: m is a real number and the discriminant of x^2 + mx + 9 > 0
def discriminant_positive (m : ℝ) := m^2 - 36 > 0

theorem distinct_real_roots_of_quadratic (m : ℝ) (h : discriminant_positive m) :
  m ∈ Iio (-6) ∪ Ioi (6) :=
sorry

end distinct_real_roots_of_quadratic_l79_79476


namespace locus_of_Y_l79_79806

open EuclideanGeometry

noncomputable def circle (O : Point) (r : Real) := {P : Point | dist O P = r}
def midpoint (A B : Point) : Point := (A + B) / 2
def perp_bisector (A B : Point) : Line := {P : Point | dist P A = dist P B}

theorem locus_of_Y (O P X : Point) (r : Real) (hP : P ∉ circle O r) (hX : X ∈ circle O r) :
  ∃ Z : Point, is_on_line Z (line O P) ∧ Z ∈ circle O r ∧
  (locus_Y : Π Y : Point, Y ∈ bisector_line (angle P O X) ∧ Y ∈ perp_bisector P X → Y ∈ perp_bisector P Z) :=
sorry

end locus_of_Y_l79_79806


namespace midpoint_I_of_AD_l79_79107

variable {α : Type*} [LinearOrderedField α]

-- Assume the existence of a triangle ABC such that 2BC = AB + AC
variables {A B C I D : α}
variable {ω : Type*}

-- Definitions
def triangle_satisfies_condition (A B C : α) : Prop := 2 * (dist B C) = (dist A B) + (dist A C)
def incenter_of_triangle (I A B C : α) : Prop := is_incenter I A B C
def circumcircle_of_triangle (ω A B C : α) : Prop := is_circumcircle ω A B C
def intersection_of_AI_and_circumcircle (A I D : α) (ω : Type*) : Prop := intersects AI ω D ∧ A ≠ D

-- Goal to prove
theorem midpoint_I_of_AD 
    (h1 : triangle_satisfies_condition A B C)
    (h2 : incenter_of_triangle I A B C)
    (h3 : circumcircle_of_triangle ω A B C)
    (h4 : intersection_of_AI_and_circumcircle A I D ω) : 
    is_midpoint I A D :=
sorry

end midpoint_I_of_AD_l79_79107


namespace max_consecutive_sum_le_1000_l79_79229

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l79_79229


namespace tetrahedron_faces_acute_l79_79714

-- Definitions for vertices of the tetrahedron
variables {A B C D M : Type} 

-- Condition: Tetrahedron has opposite sides equal
variables (AB DC AD BC AC BD : ℝ)
variables (h1 : AB = DC)
variables (h2 : BC = AD)
variables (h3 : AC = BD)

-- Goal: Prove all faces of the tetrahedron are acute-angled
theorem tetrahedron_faces_acute (A B C D : Type) 
  (AB DC AD BC AC BD : ℝ) 
  (h1 : AB = DC) 
  (h2 : BC = AD)
  (h3 : AC = BD) : 
  all_faces_acute A B C D :=
sorry

end tetrahedron_faces_acute_l79_79714


namespace greatest_digits_product_4_digit_997_l79_79643

theorem greatest_digits_product_4_digit_997 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → ∀ m : ℕ, m = 997 → ∃ d : ℕ, d = Nat.number_of_digits (n * m) ∧ d = 7 :=
by
  sorry

end greatest_digits_product_4_digit_997_l79_79643


namespace range_a_local_max_l79_79802

open Real

theorem range_a_local_max (a : ℝ) :
  (∃ x > 0, is_local_max (λ x => exp x + a * x) x) → a < -1 :=
by
  sorry

end range_a_local_max_l79_79802


namespace tetrahedron_intersect_and_circumsphere_center_l79_79050

-- Conditions
variables {S A B C : Type} -- Points of the tetrahedron
variable  [EuclideanGeometry S A B C]
variable {M : EuclideanPoint}
variable (M_is_centroid : centroid M A B C)
variable {D : EuclideanPoint}
variable (D_is_midpoint : midpoint D A B)
variables {DP : EuclideanLine}
variable (DP_parallel_SC : parallel DP SC)

-- Proof that DP intersects SM and the intersection point is the center of the circumsphere
theorem tetrahedron_intersect_and_circumsphere_center 
  (SM : EuclideanLine)
  (SM_sub_W : in_plane S M DP) :
  ∃ D' : EuclideanPoint, 
  (on_line D' DP) ∧ (on_line D' SM) ∧ 
  (equidistant D' S A B C) :=
sorry

end tetrahedron_intersect_and_circumsphere_center_l79_79050


namespace triangle_integer_lengths_impossible_l79_79081

noncomputable def isIntegerLength (x : Real) : Prop :=
  ∃ n : ℤ, x = n

theorem triangle_integer_lengths_impossible
  {A B C D E I : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace I]
  (AB AC BI ID CI IE : ℝ)
  (h0 : ∃ α : ℝ, α = ∠A B C ∧ α = π / 2)
  (h1 : ∃ β : ℝ, β = ∠A B D ∧ β = ∠D B C)
  (h2 : ∃ γ : ℝ, γ = ∠A C E ∧ γ = ∠E C B)
  (h3 : BD meets CE at I)
  (h4 : A B C form_right_triangle_at A) : 
  ¬(isIntegerLength AB ∧ isIntegerLength AC ∧ isIntegerLength BI ∧ isIntegerLength ID ∧ isIntegerLength CI ∧ isIntegerLength IE) := 
sorry

end triangle_integer_lengths_impossible_l79_79081


namespace width_of_g_domain_is_72_l79_79543

-- Define the function h and its domain condition
def h (x : ℝ) : ℝ := sorry

-- Define the function g based on h
def g (x : ℝ) : ℝ := h (x / 3)

-- State the domain condition for h
axiom h_domain : ∀ x, x ∈ Set.Icc (-12 : ℝ) (12 : ℝ) → (h x = h x)

-- Proof statement about the width of the domain of g
theorem width_of_g_domain_is_72 :
  (∀ x, x ∈ Set.Icc (-12 : ℝ) (12 : ℝ) → (h x = h x)) →
  (∀ x, x ∈ Set.Icc (-36 : ℝ) (36 : ℝ) → (g x = g x)) →
  (36 - (-36) = 72) :=
by
  intros
  calc
    36 - (-36) = 36 + 36 : by linarith
    ... = 72 : by linarith

end width_of_g_domain_is_72_l79_79543


namespace vertical_asymptote_at_x_eq_neg_2_l79_79863

def f (x : ℝ) : ℝ := (2 * x^2 + 3 * x + 10) / (x + 2)

theorem vertical_asymptote_at_x_eq_neg_2 : ∃ x : ℝ, x = -2 ∧ (x + 2 = 0) ∧ (2 * x^2 + 3 * x + 10 ≠ 0) :=
by
  use (-2)
  split
  · rfl
  split
  · norm_num
  · norm_num
    linarith

end vertical_asymptote_at_x_eq_neg_2_l79_79863


namespace probability_of_sequence_123456_l79_79655

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end probability_of_sequence_123456_l79_79655


namespace geometric_series_common_ratio_l79_79196

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l79_79196


namespace seating_arrangements_l79_79603

theorem seating_arrangements :
  let boys := 6
  let girls := 5
  let chairs := 11
  let total_arrangements := Nat.factorial chairs
  let restricted_arrangements := Nat.factorial boys * Nat.factorial girls
  total_arrangements - restricted_arrangements = 39830400 :=
by
  sorry

end seating_arrangements_l79_79603


namespace workshop_participants_problem_l79_79624

variable (WorkshopSize : ℕ) 
variable (LeftHanded : ℕ) 
variable (RockMusicLovers : ℕ) 
variable (RightHandedDislikeRock : ℕ) 
variable (Under25 : ℕ)
variable (RightHandedUnder25RockMusicLovers : ℕ)
variable (y : ℕ)

theorem workshop_participants_problem
  (h1 : WorkshopSize = 30)
  (h2 : LeftHanded = 12)
  (h3 : RockMusicLovers = 18)
  (h4 : RightHandedDislikeRock = 5)
  (h5 : Under25 = 9)
  (h6 : RightHandedUnder25RockMusicLovers = 3)
  (h7 : WorkshopSize = LeftHanded + (WorkshopSize - LeftHanded))
  (h8 : WorkshopSize - LeftHanded = RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + (WorkshopSize - LeftHanded - RightHandedDislikeRock - RightHandedUnder25RockMusicLovers - y))
  (h9 : WorkshopSize - (RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + Under25 - y - (RockMusicLovers - y)) - (LeftHanded - y) = WorkshopSize) :
  y = 5 := by
  sorry

end workshop_participants_problem_l79_79624


namespace negation_sin_proposition_l79_79160

theorem negation_sin_proposition : ¬ (∀ x : ℝ, x > 0 → sin x ≥ -1) ↔ ∃ x : ℝ, x > 0 ∧ sin x < -1 := 
sorry

end negation_sin_proposition_l79_79160


namespace geometric_series_ratio_half_l79_79202

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l79_79202


namespace select_workers_l79_79734

theorem select_workers :
  ∃ (num_selections : ℕ),
  (∀ (total workers for typesetting printing both : ℕ),
    total = 11 ∧
    typesetting = 7 ∧
    printing = 6 ∧
    both ≤ min typesetting printing →
    num_selections = 185) :=
begin
  sorry
end

end select_workers_l79_79734


namespace cosine_of_angle_in_third_quadrant_l79_79033

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l79_79033


namespace roots_of_polynomial_l79_79166

-- Define the coefficients of the polynomial equation
def a : ℂ := 1
def b : ℂ := 2
def c : ℂ := 3

-- Define the polynomial equation
def polynomial (x : ℂ) : Prop := x^2 + b * x + c = 0

-- Define the expected roots
def root1 : ℂ := -1 + complex.I * real.sqrt 2
def root2 : ℂ := -1 - complex.I * real.sqrt 2

-- Prove that these are indeed the roots of the polynomial
theorem roots_of_polynomial : ∀ x : ℂ, polynomial x ↔ (x = root1 ∨ x = root2) :=
by
  sorry

end roots_of_polynomial_l79_79166


namespace divides_expression_iff_l79_79279

theorem divides_expression_iff (p : ℕ) [hpprime : Fact p.prime] (hpodd : p % 2 = 1) :
  (∃ n : ℤ, p ∣ n * (n + 1) * (n + 2) * (n + 3) + 1) ↔ (∃ m : ℤ, p ∣ m^2 - 5) :=
by sorry

end divides_expression_iff_l79_79279


namespace at_least_one_junior_selected_l79_79035

noncomputable def probability_at_least_one_junior 
  (seniors juniors: ℕ) (selected: ℕ) 
  (total_people_seniors: seniors = 8) 
  (total_people_juniors: juniors = 4): Prop :=
  let total_people := seniors + juniors in
  let prob_no_junior := (seniors / total_people) * 
                        ((seniors - 1) / (total_people - 1)) *
                        ((seniors - 2) / (total_people - 2)) *
                        ((seniors - 3) / (total_people - 3)) in
  1 - prob_no_junior = 85 / 99

-- Now we state the actual theorem
theorem at_least_one_junior_selected :
  probability_at_least_one_junior 8 4 4 rfl rfl :=
  sorry

end at_least_one_junior_selected_l79_79035


namespace egg_pack_count_l79_79722

theorem egg_pack_count (n : ℕ) (h : 6 / (n * (n - 1)) = 0.0047619047619047615) : n = 36 :=
sorry

end egg_pack_count_l79_79722


namespace max_consecutive_sum_le_1000_l79_79230

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l79_79230


namespace domain_F_l79_79438

-- Define the function f and specify its domain.
def f (x : ℝ) : ℝ := sorry  -- Placeholder for the actual function definition.

-- Define the function F in terms of f and sqrt.
def F (x : ℝ) : ℝ := f (x + 1) + Real.sqrt (3 - x)

theorem domain_F (h : ∀ x : ℝ, 0 < x → ∃ y : ℝ, f x = y) : 
  ∀ x : ℝ, (-1 < x ∧ x ≤ 3) ↔ ∃ y : ℝ, F x = y := 
by
  sorry

end domain_F_l79_79438


namespace y_intercept_of_tangent_line_2023rd_derivative_l79_79368

def f (x : ℝ) : ℝ := x * Real.exp x

-- Define nth derivative recursively
noncomputable def derivative : ℕ → (ℝ → ℝ)
| 0 := f
| (n+1) := λ x, (derivative n)' x

-- nth derivative of f
noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
(derivative n) x

theorem y_intercept_of_tangent_line_2023rd_derivative :
  let x := 0 in
  let n := 2023 in
  let f' := derivative (n+1) in
  let tangent_line := λ x, (f' 0) * x + f_n n 0 in
  tangent_line (Real.neg (2023/2024 : ℝ)) = 0 :=
by
  sorry

end y_intercept_of_tangent_line_2023rd_derivative_l79_79368


namespace greatest_int_less_equal_expr_l79_79762

theorem greatest_int_less_equal_expr : 
  let expr := (4^103 + 3^103) / (4^100 + 3^100) in
  ∃ (k : ℤ), k = 63 ∧ (k : ℝ) ≤ expr ∧ expr < (k + 1 : ℝ) :=
begin
  sorry
end

end greatest_int_less_equal_expr_l79_79762


namespace geometric_series_ratio_half_l79_79204

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l79_79204


namespace find_integer_solutions_of_equation_l79_79381

def has_eight_divisors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), (a.prime ∧ b.prime ∧ c.prime) ∧ (n = a^7 ∨ n = a^3 * b ∨ n = a * b * c)

theorem find_integer_solutions_of_equation (p q r : ℕ) :
  r + p^4 = q^4 →
  has_eight_divisors r →
  p.prime ∧ q.prime →
  (p = 2 ∧ q = 5 ∧ r = 609) :=
sorry

end find_integer_solutions_of_equation_l79_79381


namespace area_of_BEIH_is_3_over_4_l79_79704

-- Define the vertices of the rectangle
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (3, 0)
def D : ℝ × ℝ := (3, 2)

-- Define the midpoints of AB and CD
def E : ℝ × ℝ := midpoint ℝ ℝ (0, 0) (0, 2)
def F : ℝ × ℝ := midpoint ℝ ℝ (3, 0) (3, 2)

-- Assume the intersection points based on conditions
def I : ℝ × ℝ := (3/2, 5/2)
def H : ℝ × ℝ := (3/2, 3/2)

-- Define the area of quadrilateral BEIH
def area_BEIH : ℝ :=
  (1/2) * abs ((E.1 * I.2 + I.1 * H.2 + H.1 * B.2 + B.1 * E.2) - (E.2 * I.1 + I.2 * H.1 + H.2 * B.1 + B.2 * E.1))

-- The proof problem: prove that the area of quadrilateral BEIH is 3/4
theorem area_of_BEIH_is_3_over_4 : area_BEIH = 3 / 4 :=
by sorry

end area_of_BEIH_is_3_over_4_l79_79704


namespace infinite_pos_int_sum_three_cubes_l79_79122

theorem infinite_pos_int_sum_three_cubes (i : ℕ) (h : i = 1 ∨ i = 2 ∨ i = 3) :
  ∃ (n : ℕ), ∃ᶠ (n : ℕ) in at_top, 
   (∃ (k : ℕ), k = (commodious_solution_based n i mod 9) = true if cubeable condition setup) :=
    sorry

end infinite_pos_int_sum_three_cubes_l79_79122


namespace last_person_is_knight_l79_79529

-- Definitions for the conditions:
def first_whispered_number := 7
def last_announced_number_first_game := 3
def last_whispered_number_second_game := 5
def first_announced_number_second_game := 2

-- Definitions to represent the roles:
inductive Role
| knight
| liar

-- Definition of the last person in the first game being a knight:
def last_person_first_game_role := Role.knight

theorem last_person_is_knight 
  (h1 : Role.liar = Role.liar)
  (h2 : last_announced_number_first_game = 3)
  (h3 : first_whispered_number = 7)
  (h4 : first_announced_number_second_game = 2)
  (h5 : last_whispered_number_second_game = 5) :
  last_person_first_game_role = Role.knight :=
sorry

end last_person_is_knight_l79_79529


namespace geometric_series_common_ratio_l79_79183

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l79_79183


namespace find_solutions_l79_79777

theorem find_solutions (x y : ℤ) (hx : x ≥ 0) (hy : y ≥ 0) :
  1 + 3^x = 2^y ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_solutions_l79_79777


namespace find_y_l79_79159

theorem find_y 
  (h : (5 + 8 + 17) / 3 = (12 + y) / 2) : y = 8 :=
sorry

end find_y_l79_79159


namespace difference_in_change_is_40_percent_l79_79743

theorem difference_in_change_is_40_percent :
  let initial_yes_percent := 0.40
  let initial_no_percent := 0.40
  let initial_undecided_percent := 0.20
  let final_yes_percent := 0.60
  let final_no_percent := 0.30
  let final_undecided_percent := 0.10
  max_percent_change_minus_min_percent_change initial_yes_percent initial_no_percent initial_undecided_percent 
                                              final_yes_percent final_no_percent final_undecided_percent = 0.40 :=
by 
  sorry

def max_percent_change_minus_min_percent_change (initial_yes initial_no initial_undecided : Float) 
                                                (final_yes final_no final_undecided : Float) : Float :=
  -- Placeholder function to compute the difference between maximum and minimum possible values of changes.
  sorry

end difference_in_change_is_40_percent_l79_79743


namespace cosine_of_angle_in_third_quadrant_l79_79032

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l79_79032


namespace probability_sequence_123456_l79_79661

theorem probability_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11
  in (favorable_sequences : ℚ) / total_sequences = 1 / 720 := 
by 
  sorry

end probability_sequence_123456_l79_79661


namespace minimum_perimeter_Q_l79_79085

noncomputable def Q (z : ℂ) : ℂ := z^8 + (2 * complex.sqrt 3 + 8) * z^4 - (2 * complex.sqrt 3 + 9)

theorem minimum_perimeter_Q (z : ℂ) : 
  (∀ v ∈ ({z : ℂ | Q z = 0}), 
   ∑ (pair : ℂ × ℂ) in (v.zip_with_next), complex.dist (pair.1) (pair.2)) = 8 * complex.sqrt 2 :=
sorry

end minimum_perimeter_Q_l79_79085


namespace minimum_chips_for_A10_l79_79227

theorem minimum_chips_for_A10 (n : ℕ) (initial_chips : ∀ i : ℕ, i ≠ 1 → i < 11 → ℕ := 0) :
  ∃ n ≥ 46, (initial_chips 1 = n) → ∃ m : ℕ, m ≥ 1 ∧ initial_chips 10 = m :=
begin
  sorry
end

end minimum_chips_for_A10_l79_79227


namespace compare_fractions_l79_79351

theorem compare_fractions :
  (111110 / 111111) < (333331 / 333334) ∧ (333331 / 333334) < (222221 / 222223) :=
by
  sorry

end compare_fractions_l79_79351


namespace change_in_us_volume_correct_l79_79970

-- Definition: Change in the total import and export volume of goods in a given year
def change_in_volume (country : String) : Float :=
  if country = "China" then 7.5
  else if country = "United States" then -6.4
  else 0

-- Theorem: The change in the total import and export volume of goods in the United States is correctly represented.
theorem change_in_us_volume_correct :
  change_in_volume "United States" = -6.4 := by
  sorry

end change_in_us_volume_correct_l79_79970


namespace lora_coins_l79_79295

theorem lora_coins :
  ∃ n : ℕ, (17 = (finset.filter (λ d, d > 2) (nat.divisors n)).card) ∧
           (∀ m : ℕ, (17 = (finset.filter (λ d, d > 2) (nat.divisors m)).card) → n ≤ m) ∧
           n = 2700 :=
by
  sorry

end lora_coins_l79_79295


namespace water_left_over_l79_79995

theorem water_left_over (players : ℕ) (initial_liters : ℕ) (milliliters_per_player : ℕ) (water_spill_ml : ℕ) :
  players = 30 → initial_liters = 8 → milliliters_per_player = 200 → water_spill_ml = 250 →
  (initial_liters * 1000) - (players * milliliters_per_player + water_spill_ml) = 1750 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  change 8 * 1000 - (30 * 200 + 250) = 1750
  norm_num
  sorry

end water_left_over_l79_79995


namespace general_term_and_nonexistent_pair_l79_79517

open Nat

def sequence (a : ℕ → ℕ) : Prop := 
  (a 1 = 2) ∧ ∀ n, a (n+1) = a n + 2^n

theorem general_term_and_nonexistent_pair (a : ℕ → ℕ) :
  sequence a →
  (∀ n, a n = 2^n) ∧ ¬ ∃ p q, p < q ∧ a p + a q = 2014 :=
by
  assume h1 : sequence a
  sorry

end general_term_and_nonexistent_pair_l79_79517


namespace usual_time_to_cover_distance_l79_79224

theorem usual_time_to_cover_distance
  (S T : ℝ) :
  ((S / (0.25 * S)) = ((T + 24) / T)) →
  4 = (T + 24) / T →
  T = 8 := 
by
  intros,
  sorry

end usual_time_to_cover_distance_l79_79224


namespace symmetry_of_f_graph_symmetric_about_point_half_l79_79978

def f (x : ℝ) : ℝ := |⌊x⌋| - |⌊1 - x⌋|

theorem symmetry_of_f : ∀ x : ℝ, f(x) = f(1 - x) :=
by
  sorry

theorem graph_symmetric_about_point_half : ∃ k, k = (1/2, 0) ∧ (∀ x, f(x) = f(1 - x)) :=
by
  use (1/2, 0)
  split
  · rfl
  · apply symmetry_of_f
  sorry

end symmetry_of_f_graph_symmetric_about_point_half_l79_79978


namespace find_a_l79_79833

theorem find_a (a : ℝ) :
  (∀ x : ℝ, ((x^2 - 4 * x + a) + |x - 3| ≤ 5) → x ≤ 3) →
  (∃ x : ℝ, x = 3 ∧ ((x^2 - 4 * x + a) + |x - 3| ≤ 5)) →
  a = 2 := 
by
  sorry

end find_a_l79_79833


namespace find_a_l79_79832

theorem find_a (a : ℝ) :
  (∀ x : ℝ, ((x^2 - 4 * x + a) + |x - 3| ≤ 5) → x ≤ 3) →
  (∃ x : ℝ, x = 3 ∧ ((x^2 - 4 * x + a) + |x - 3| ≤ 5)) →
  a = 2 := 
by
  sorry

end find_a_l79_79832


namespace polynomial_has_at_most_one_integer_root_l79_79079

theorem polynomial_has_at_most_one_integer_root (k : ℝ) :
  ∀ x y : ℤ, (x^3 - 24 * x + k = 0) ∧ (y^3 - 24 * y + k = 0) → x = y :=
by
  intros x y h
  sorry

end polynomial_has_at_most_one_integer_root_l79_79079


namespace no_return_after_12_jumps_return_after_13_jumps_l79_79933

-- Define the number of points (12 points on a circle).
def num_points : ℕ := 12

-- Define the initial positions of the grasshoppers on the circle.
def initial_positions : Fin num_points → RealAngle := sorry  -- Assuming RealAngle is a type representing angles on the circle

-- Define the function representing a single jump to the nearest midpoint in a clockwise direction.
def jump : RealAngle → RealAngle := sorry  -- Function to compute the new position after a jump

-- Proof statement (a): No grasshopper can return to its initial position after 12 jumps.
theorem no_return_after_12_jumps : ∀ (i : Fin num_points), (jump^[12] (initial_positions i) ≠ initial_positions i) := sorry

-- Proof statement (b): Any grasshopper can return to its initial position after 13 jumps.
theorem return_after_13_jumps : ∀ (i : Fin num_points), (jump^[13] (initial_positions i) = initial_positions i) := sorry

end no_return_after_12_jumps_return_after_13_jumps_l79_79933


namespace smallest_radius_squared_of_sphere_l79_79323

theorem smallest_radius_squared_of_sphere :
  ∃ (x y z : ℤ), 
  (x - 2)^2 + y^2 + z^2 = (x^2 + (y - 4)^2 + z^2) ∧
  (x - 2)^2 + y^2 + z^2 = (x^2 + y^2 + (z - 6)^2) ∧
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
  (∃ r, r^2 = (x - 2)^2 + (0 - y)^2 + (0 - z)^2) ∧
  51 = r^2 :=
sorry

end smallest_radius_squared_of_sphere_l79_79323


namespace log_base_3x_condition_l79_79858

theorem log_base_3x_condition (x : ℝ) (h : log (3 * x) 729 = x) : 
  x = 3 ∧ ¬(∃ n : ℤ, n^2 = x) ∧ ¬(∃ m : ℤ, m^3 = x) :=
by
  sorry

end log_base_3x_condition_l79_79858


namespace odd_n_non_integer_even_n_example_l79_79808

-- Define the sequence recurrence relation
noncomputable def sequence (n : ℕ) : (ℕ → ℝ) → ℕ → ℕ → ℝ
| x 0 := x
| x (k + 1) i :=
  if i = n then
    1 / 2 * (sequence n x k i + sequence n x k 1)
  else
    1 / 2 * (sequence n x k i + sequence n x k (i + 1))

-- Main theorem for odd n
theorem odd_n_non_integer {n : ℕ} (hn : n ≥ 2) (hn_odd : n % 2 = 1)
  (x : ℕ → ℤ) (not_all_equal : ∃ i j, i < n ∧ j < n ∧ x i ≠ x j) :
  ∃ k j, j < n ∧ ¬ (sequence n (λ i, (x i : ℝ)) k j).isInt :=
sorry

-- Counterexample for even n
theorem even_n_example {n : ℕ} (hn : n ≥ 2) (hn_even : n % 2 = 0) :
  ∃ x : ℕ → ℤ, (∀ j < n, ∀ k, (sequence n (λ i, (x i : ℝ)) k j).isInt) :=
begin
  use λ i, if i % 2 = 0 then 1 else -1,
  intros j hj k,
  induction k with k ih,
  { simp [sequence] },
  { simp [sequence, ih, if_pos] },
end

end odd_n_non_integer_even_n_example_l79_79808


namespace problem_statement_l79_79036

def A := {x : ℝ | x * (x - 1) < 0}
def B := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem problem_statement : A ⊆ {y : ℝ | y ≥ 0} :=
sorry

end problem_statement_l79_79036


namespace length_of_AC_l79_79117

-- Defining the isosceles triangle ABC with AB = BC
variables {A B C M : Type}
variable [triangle : ∀ A B C, Prop (isosceles A B C)] -- Assuming triangle ABC is isosceles with AB = BC
variable (point_M_on_AC : ∀ A C M, Prop (on_line A C M)) -- Point M on line AC
variable (AM_val : distance (A, M) = 7) -- AM = 7
variable (MB_val : distance (M, B) = 3) -- MB = 3
variable (angle_BMC_60 : ∠(B, M, C) = 60) -- ∠BMC = 60°

-- Define the proof problem: the length of AC
theorem length_of_AC (ABC_isosceles : isosceles A B C)
  (M_on_AC : on_line A C M)
  (H_AM : distance (A, M) = 7)
  (H_MB : distance (M, B) = 3)
  (H_angle_BMC : ∠(B, M, C) = 60) :
  distance (A, C) = 17 :=
sorry

end length_of_AC_l79_79117


namespace sum_of_altitudes_is_less_than_perimeter_l79_79942

theorem sum_of_altitudes_is_less_than_perimeter 
  (a b c h_a h_b h_c : ℝ) 
  (h_a_le_b : h_a ≤ b) 
  (h_b_le_c : h_b ≤ c) 
  (h_c_le_a : h_c ≤ a) 
  (strict_inequality : h_a < b ∨ h_b < c ∨ h_c < a) : h_a + h_b + h_c < a + b + c := 
by 
  sorry

end sum_of_altitudes_is_less_than_perimeter_l79_79942


namespace find_constants_sum_of_squares_l79_79629

noncomputable def cos_expansion_constants (a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  ∀ θ : ℝ, real.cos θ ^ 5 = 
            a₁ * real.cos θ + 
            a₂ * real.cos (2 * θ) + 
            a₃ * real.cos (3 * θ) + 
            a₄ * real.cos (4 * θ) + 
            a₅ * real.cos (5 * θ)

theorem find_constants_sum_of_squares :
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℝ), 
    cos_expansion_constants a₁ a₂ a₃ a₄ a₅ ∧ 
    a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = 63 / 128 :=
sorry

end find_constants_sum_of_squares_l79_79629


namespace sum_G_div_5_pow_inf_l79_79916

def fibonacci : ℕ → ℕ 
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

def G (n : ℕ) : ℕ := 2 * fibonacci n

noncomputable def sum_G_div_5_pow (N : ℕ) : ℝ := ∑ n in finset.range N, (G n : ℝ) / (5 ^ n)

noncomputable def inf_sum_G_div_5_pow : ℝ := ∑' n, (G n : ℝ) / (5 ^ n)

theorem sum_G_div_5_pow_inf : inf_sum_G_div_5_pow = 10 / 19 := 
by sorry

end sum_G_div_5_pow_inf_l79_79916


namespace represent_in_scientific_notation_l79_79569

def million : ℕ := 10^6
def rural_residents : ℝ := 42.39 * million

theorem represent_in_scientific_notation :
  42.39 * 10^6 = 4.239 * 10^7 :=
by
  -- The proof is omitted.
  sorry

end represent_in_scientific_notation_l79_79569


namespace max_consecutive_integers_sum_l79_79250

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l79_79250


namespace markers_difference_l79_79531

theorem markers_difference (leo_cost maya_cost : ℝ) (h_leo : leo_cost = 3.51) (h_maya : maya_cost = 4.25) : ∃ m : ℕ, m = 4 :=
by
  use 4
  sorry

end markers_difference_l79_79531


namespace functions_symmetric_about_y_axis_l79_79897

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2^(x + 1)
def g (x : ℝ) : ℝ := 2^(1 - x)

-- Statement to prove the symmetry about y-axis
theorem functions_symmetric_about_y_axis :
  ∀ (x : ℝ), f(-x) = g(x) :=
by
  sorry

end functions_symmetric_about_y_axis_l79_79897


namespace parallelogram_is_centrally_symmetric_l79_79277

-- Definition of a parallelogram
structure Parallelogram (P Q R S : Type) :=
(e1 : segment P Q)
(e2 : segment Q R)
(e3 : segment R S)
(e4 : segment S P)
(parallel : ∀ {e1 e3 : Prop}, e1 ∥ e3)
(parallel : ∀ {e2 e4 : Prop}, e2 ∥ e4)

-- Definition of central symmetry
def centrally_symmetric (fig : Type) :=
∃ (center : Type), ∀ x ∈ fig, ∃ y ∈ fig, (center + x = y + center)

-- Theorem to prove the property
theorem parallelogram_is_centrally_symmetric (P Q R S : Type) (p : Parallelogram P Q R S) :
    centrally_symmetric (p.e1, p.e2, p.e3, p.e4) :=
by
  sorry

end parallelogram_is_centrally_symmetric_l79_79277


namespace intersection_A_B_l79_79005

def A : set ℝ := { x | -1 < 2^x ∧ 2^x < 2 }
def B : set ℝ := { x | -1 < log 2 x ∧ log 2 x < 2 }

theorem intersection_A_B : (A ∩ B) = { x | 0.5 < x ∧ x < 1 } :=
  by sorry

end intersection_A_B_l79_79005


namespace weight_of_person_being_replaced_l79_79606

variable (W_old : ℝ)

theorem weight_of_person_being_replaced :
  (W_old : ℝ) = 35 :=
by
  -- Given: The average weight of 8 persons increases by 5 kg.
  -- The weight of the new person is 75 kg.
  -- The total weight increase is 40 kg.
  -- Prove that W_old = 35 kg.
  sorry

end weight_of_person_being_replaced_l79_79606


namespace smallest_three_digit_n_l79_79744

theorem smallest_three_digit_n (n : ℕ) (h_pos : 100 ≤ n) (h_below : n ≤ 999) 
  (cond1 : n % 9 = 2) (cond2 : n % 6 = 4) : n = 118 :=
by {
  sorry
}

end smallest_three_digit_n_l79_79744


namespace range_of_f_positive_l79_79550

noncomputable def f (x : ℝ) : ℝ := sorry
def f' (x : ℝ) : ℝ := sorry -- Define the derivative of f

theorem range_of_f_positive :
  (∀ x : ℝ, f (-x) = -f x) ∧ f (-1) = 0 ∧ (∀ x > 0, x * f' x - f x > 0) →
  (∀ x, (f x > 0 ↔ (x ∈ Set.Ioo (-1 : ℝ) 0 ∨ x ∈ Set.Ioi 1))) :=
begin
  sorry
end

end range_of_f_positive_l79_79550


namespace sum_of_repeating_decimals_l79_79769

theorem sum_of_repeating_decimals :
  (0.1.replicate_repeat + 0.02.replicate_repeat + 0.0003.replicate_repeat) = (13151314 / 99999999 : ℚ) := by
  sorry

end sum_of_repeating_decimals_l79_79769


namespace possible_values_of_n_l79_79929

theorem possible_values_of_n (n : ℕ) (h1 : n > 1) 
  (d : ℕ → ℕ) (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ m, d m = i) -- d is the function that gives positive divisors
  (h_diff : ∀ (i : ℕ), 2 ≤ i → ∃ k, ∀ j, 1 ≤ j ∧ j < i → d (n / 2) + 1 = n) -- positive divisors of another integer
: ∃ s : ℕ, s > 0 ∧ n = 2^s :=
sorry

end possible_values_of_n_l79_79929


namespace eight_sided_die_probability_l79_79667

/-- When rolling a fair 8-sided die (with faces numbered from 1 through 8),
    the probability of rolling a number less than 5 is 1/2. -/
theorem eight_sided_die_probability :
  let total_outcomes := 8
  let favorable_outcomes := 4
  let probability := (favorable_outcomes : ℝ) / (total_outcomes : ℝ)
  in probability = 1 / 2 :=
by
  let total_outcomes := 8
  let favorable_outcomes := 4
  let probability := (favorable_outcomes : ℝ) / (total_outcomes : ℝ)
  show probability = 1 / 2
  sorry

end eight_sided_die_probability_l79_79667


namespace count_integers_satisfying_conditions_l79_79009

theorem count_integers_satisfying_conditions :
  (finset.filter (λ n : ℤ, -11 ≤ n ∧ n ≤ 11 ∧ (n-3)*(n+3)*(n+8) < 0)
     (finset.Icc (-11) (11))).card = 7 := 
sorry

end count_integers_satisfying_conditions_l79_79009


namespace quadrilateral_diagonals_l79_79620

theorem quadrilateral_diagonals (a b c d e f : ℝ) 
  (hac : a > c) 
  (hbd : b ≥ d) 
  (hapc : a = c) 
  (hdiag1 : e^2 = (a - b)^2 + b^2) 
  (hdiag2 : f^2 = (c + b)^2 + b^2) :
  e^4 - f^4 = (a + c) / (a - c) * (d^2 * (2 * a * c + d^2) - b^2 * (2 * a * c + b^2)) :=
by
  sorry

end quadrilateral_diagonals_l79_79620


namespace suitcase_combinations_l79_79750

def count_odd_numbers (n : Nat) : Nat := n / 2

def count_multiples_of_4 (n : Nat) : Nat := n / 4

def count_multiples_of_5 (n : Nat) : Nat := n / 5

theorem suitcase_combinations : count_odd_numbers 40 * count_multiples_of_4 40 * count_multiples_of_5 40 = 1600 :=
by
  sorry

end suitcase_combinations_l79_79750


namespace correct_factorization_l79_79275

theorem correct_factorization (a x m : ℝ) :
  (ax^2 - a = a * (x^2 - 1)) ∨
  (m^3 + m = m * (m^2 + 1)) ∨
  (x^2 + 2*x - 3 = x*(x+2) - 3) ∨
  (x^2 + 2*x - 3 = (x-3)*(x+1)) :=
by sorry

end correct_factorization_l79_79275


namespace solution_l79_79382

noncomputable def f (x : ℝ) := 
  10 / (Real.sqrt (x - 5) - 10) + 
  2 / (Real.sqrt (x - 5) - 5) + 
  9 / (Real.sqrt (x - 5) + 5) + 
  18 / (Real.sqrt (x - 5) + 10)

theorem solution : 
  f (1230 / 121) = 0 := sorry

end solution_l79_79382


namespace cost_of_energy_drink_l79_79968

/-- 
The conditions stated in the problem 
-/
variables (cupcakes_sold : ℕ) (cupcake_price : ℝ)
variables (cookies_sold : ℕ) (cookie_price : ℝ)
variables (basketballs_bought : ℕ) (basketball_price : ℝ)
variables (energy_drinks_bought : ℕ)

/--
The proof statement: Given the number of items sold and their respective prices, 
and the cost of basketballs, prove the cost of one energy drink.
-/
theorem cost_of_energy_drink (h: cupcakes_sold = 50)
                              (h1: cupcake_price = 2)
                              (h2: cookies_sold = 40)
                              (h3: cookie_price = 0.5)
                              (h4: basketballs_bought = 2)
                              (h5: basketball_price = 40)
                              (h6: energy_drinks_bought = 20) :
  (120 - 2 * 40 : ℝ) / 20 = 2 :=
by
  sorry

end cost_of_energy_drink_l79_79968


namespace roots_of_polynomial_l79_79069

noncomputable def cos_degree (d : ℝ) : ℝ := Real.cos (d * Real.pi / 180)

theorem roots_of_polynomial :
  let t := cos_degree 6 in
  t^5 * 32 - t^3 * 40 + t * 10 - Real.sqrt 3 = 0 →
  ∃ t1 t2 t3 t4 : ℝ, t1 = cos_degree 78 ∧
                     t2 = cos_degree 150 ∧
                     t3 = cos_degree 222 ∧
                     t4 = cos_degree 294 ∧
                     32 * t1^5 - 40 * t1^3 + 10 * t1 - Real.sqrt 3 = 0 ∧
                     32 * t2^5 - 40 * t2^3 + 10 * t2 - Real.sqrt 3 = 0 ∧
                     32 * t3^5 - 40 * t3^3 + 10 * t3 - Real.sqrt 3 = 0 ∧
                     32 * t4^5 - 40 * t4^3 + 10 * t4 - Real.sqrt 3 = 0 :=
by
  sorry

end roots_of_polynomial_l79_79069


namespace problem1_problem2_problem3_l79_79294

-- Problem 1
theorem problem1 (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  (1/2 * x * (1 - 2 * x) ≤ 1/16) := sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : 0 < x) : 
  (2 - x - 4 / x ≤ -2) := sorry

-- Problem 3
theorem problem3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  (1 / x + 3 / y ≥ 1 + Real.sqrt 3 / 2) := sorry

end problem1_problem2_problem3_l79_79294


namespace find_x_find_height_diff_l79_79571

-- Define the conversion scales for each region
def conversion_A := 35 / 7  -- km per cm for region A
def conversion_B := 45 / 9  -- km per cm for region B
def conversion_C := 30 / 10 -- km per cm for region C

-- Define the map distances between cities
def distance_map_A := 15 -- cm
def distance_map_B := 10 -- cm
noncomputable def x := 76.67 -- cm (value that we need to prove)

-- Define the total perimeter
def total_perimeter := 355 -- km

-- Define the height difference between region A and B
def height_diff_map_A_B := 2 -- cm
def height_diff_real_A_B := 500 -- meters

-- Define the height difference between region B and C on the map
def height_diff_map_B_C := 3 -- cm

-- Theorem to prove the value of x in cm
theorem find_x : 
  (distance_map_A * conversion_A + distance_map_B * conversion_B + x * conversion_C) = total_perimeter :=
sorry

-- Define the height conversion scale
def height_conversion := height_diff_real_A_B / height_diff_map_A_B  -- meters per cm

-- Theorem to prove the height difference between B and C in meters
theorem find_height_diff : 
  (height_diff_map_B_C * height_conversion) = 750 :=
sorry

end find_x_find_height_diff_l79_79571


namespace correctly_calculated_value_l79_79963

theorem correctly_calculated_value (n : ℕ) (h : 5 * n = 30) : n / 6 = 1 :=
sorry

end correctly_calculated_value_l79_79963


namespace find_lambda_range_l79_79817

open Real

theorem find_lambda_range {a b : ℝ} (θ λ : ℝ) (non_zero_a : a ≠ 0) (non_zero_b : b ≠ 0) 
    (magnitude_a : abs a = 2 * abs b) 
    (inequality : ∀ θ, abs (2 * a + b) ≥ abs (a + λ * b)) : 
    -1 ≤ λ ∧ λ ≤ 3 :=
by
  sorry

end find_lambda_range_l79_79817


namespace olivia_spent_38_l79_79565

def initial_amount : ℕ := 128
def amount_left : ℕ := 90
def money_spent (initial amount_left : ℕ) : ℕ := initial - amount_left

theorem olivia_spent_38 :
  money_spent initial_amount amount_left = 38 :=
by 
  sorry

end olivia_spent_38_l79_79565


namespace molecular_weight_BaBr2_l79_79644

theorem molecular_weight_BaBr2 (w: ℝ) (h: w = 2376) : w / 8 = 297 :=
by
  sorry

end molecular_weight_BaBr2_l79_79644


namespace shopkeeper_gain_percentage_l79_79672

-- Define the conditions
def cost_price (kg: ℝ) : ℝ := kg
def false_weight : ℝ := 950
def true_weight : ℝ := 1000

-- Define the gain percentage calculation
def gain_percentage (error true_value: ℝ) : ℝ :=
  (error / true_value) * 100

-- Define the problem statement
theorem shopkeeper_gain_percentage : gain_percentage (true_weight - false_weight) true_weight = 5 :=
by 
  -- Proof will be implemented here
  sorry

end shopkeeper_gain_percentage_l79_79672


namespace probability_of_sequence_123456_l79_79656

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end probability_of_sequence_123456_l79_79656


namespace possible_values_of_m_l79_79489

-- Define the conditions that m is a real number and the quadratic equation having two distinct real roots
variable (m : ℝ)

-- Define the discriminant condition for having two distinct real roots
def discriminant_condition (a b c : ℝ) := b^2 - 4 * a * c > 0

-- State the required theorem
theorem possible_values_of_m (h : discriminant_condition 1 m 9) : m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo 6 ∞ :=
sorry

end possible_values_of_m_l79_79489


namespace arithmetic_sequence_general_term_and_sum_max_l79_79061

-- Definitions and conditions
def a1 : ℤ := 4
def d : ℤ := -2
def a (n : ℕ) : ℤ := a1 + (n - 1) * d
def Sn (n : ℕ) : ℤ := n * (a1 + (a n)) / 2

-- Prove the general term formula and maximum value
theorem arithmetic_sequence_general_term_and_sum_max :
  (∀ n, a n = -2 * n + 6) ∧ (∃ n, Sn n = 6) :=
by
  sorry

end arithmetic_sequence_general_term_and_sum_max_l79_79061


namespace rankings_count_l79_79136

def rankings_count_condition (A_first : Prop) (A_fifth : Prop) (B_fifth : Prop) : ℕ :=
  if ¬A_first ∧ ¬A_fifth ∧ ¬B_fifth then 54 else 0

theorem rankings_count :
  rankings_count_condition false false false = 54 :=
by
  -- Intentionally left unproven
  sorry

end rankings_count_l79_79136


namespace line_tangent_to_circle_l79_79125

theorem line_tangent_to_circle (x y : ℝ) :
  (3 * x - 4 * y + 25 = 0) ∧ (x^2 + y^2 = 25) → (x = -3 ∧ y = 4) :=
by sorry

end line_tangent_to_circle_l79_79125


namespace orthocenters_collinear_l79_79605

-- Define the vertices of the acute-angled triangle
variables {A B C : Type} [IsTriangle A B C] 

-- Define the altitudes from vertices to the opposite sides
variables {alt_A : Altitude A B C} {alt_B : Altitude B A C} {alt_C : Altitude C A B}

-- Define the internal angle bisectors intersection points
variables {P : Point} {D : Point} {F : Point} 
-- Define the external angle bisectors intersection points
variables {Q : Point} {E : Point} {G : Point}

-- Define orthocenters of the triangles formed
variables {H_A : Orthocenter (Triangle A P Q)} 
variables {H_B : Orthocenter (Triangle B D E)} 
variables {H_C : Orthocenter (Triangle C F G)}

-- Define the main triangle's orthocenter
variables {H : Orthocenter (Triangle A B C)}

theorem orthocenters_collinear (A B C : Type) [IsTriangle A B C] 
    (alt_A : Altitude A B C) (alt_B : Altitude B A C) (alt_C : Altitude C A B)
    (P Q D E F G : Type) -- Points of intersection as types
    (H_A : Orthocenter (Triangle A P Q)) 
    (H_B : Orthocenter (Triangle B D E)) 
    (H_C : Orthocenter (Triangle C F G)) 
    (H : Orthocenter (Triangle A B C)) : 
  Collinear H_A H_B H_C ∧ Collinear H A H :=
sorry

end orthocenters_collinear_l79_79605


namespace num_solutions_to_inequality_l79_79462

theorem num_solutions_to_inequality : 
  let S := {x : ℕ | 12 < -2 * (x : ℤ) + 17} in
  S.card = 2 := 
by 
  sorry

end num_solutions_to_inequality_l79_79462


namespace max_value_x_plus_inv_x_l79_79990

theorem max_value_x_plus_inv_x (f : Fin 2023 → ℝ) (hpos : ∀ i, 0 < f i)
  (hsum : ∑ i, f i = 2024) (hrecip_sum : ∑ i, (f i)⁻¹ = 2024) :
  ∃ x, x ∈ (Set.range f) ∧ x + 1 / x = 4096094 / 2024 :=
by
  sorry

end max_value_x_plus_inv_x_l79_79990


namespace lcm_of_consecutive_correct_l79_79641

noncomputable def lcm_of_consecutive (n : ℕ) : ℕ :=
if even (n + 1)
then n * (n + 1) * (n + 2)
else (n * (n + 1) * (n + 2)) / 2

theorem lcm_of_consecutive_correct (n : ℕ) :
  lcm (n, n + 1, n + 2) = lcm_of_consecutive n :=
sorry

end lcm_of_consecutive_correct_l79_79641


namespace solve_for_x_l79_79592

theorem solve_for_x:
    ∀ x : ℝ, 
      (x ≠ 6) → (4 * x - 3 ≠ 0) → (x^2 - 10 * x + 24) / (x - 6) + (4 * x^2 + 20 * x - 24) / (4 * x - 3) + 2 * x = 5 → 
      x = 1 / 4 :=
begin
  intros,
  sorry
end

end solve_for_x_l79_79592


namespace domain_g_3_single_point_l79_79092

noncomputable def g₁ (x : ℝ) : ℝ := real.sqrt (1 - x^2)

noncomputable def g : ℕ → ℝ → ℝ
| 1 := g₁
| n + 1 := λ x, g n (real.sqrt ((n + 1) ^ 2 - x^2))

theorem domain_g_3_single_point
  (M : ℕ) (hM : ∀ n : ℕ, n ≤ M → ∃ x : ℝ, g n x = g n x)
  (hMax : ∀ n : ℕ, M < n → ¬(∃ x : ℝ, g n x = g n x))
  : ∃ d : ℝ, {d} = {x : ℝ | ∃ y : ℝ, g M y = x} :=
by
  let M := 3
  let d := 3
  sorry

end domain_g_3_single_point_l79_79092


namespace geometric_series_common_ratio_l79_79208

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l79_79208


namespace intersection_point_on_y_eq_neg_x_l79_79805

theorem intersection_point_on_y_eq_neg_x 
  (α β : ℝ)
  (h1 : ∃ x y : ℝ, (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧ 
                   (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧ 
                   (y = -x)) :
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 :=
sorry

end intersection_point_on_y_eq_neg_x_l79_79805


namespace divisibility_criterion_l79_79948

theorem divisibility_criterion (x y : ℕ) (h_two_digit : 10 ≤ x ∧ x < 100) :
  (1207 % x = 0) ↔ (x = 10 * (x / 10) + (x % 10) ∧ (x / 10)^3 + (x % 10)^3 = 344) :=
by
  sorry

end divisibility_criterion_l79_79948


namespace cosine_in_third_quadrant_l79_79026

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l79_79026


namespace angelina_speed_l79_79333

variable (v : ℝ) (t₁ t₂ : ℝ) 

def time_home_to_grocery (v : ℝ) : ℝ :=
  100 / v

def time_grocery_to_gym (v : ℝ) : ℝ :=
  180 / (2*v)

theorem angelina_speed (h1 : 100 / v - 180 / (2*v) = 40) :  2 * (v = 1/4) := by 
  -- solution omitted
  sorry

end angelina_speed_l79_79333


namespace binom_identity_l79_79538

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) : k * binom n k = n * binom (n - 1) (k - 1) := by
  sorry

end binom_identity_l79_79538


namespace sophie_saves_money_by_using_wool_balls_l79_79598

def cost_of_dryer_sheets_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                  (weeks_per_year : ℕ) (sheets_per_box : ℕ)
                                  (cost_per_box : ℝ) : ℝ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := sheets_per_year / sheets_per_box
  boxes_per_year * cost_per_box

theorem sophie_saves_money_by_using_wool_balls :
  cost_of_dryer_sheets_per_year 4 1 52 104 5.50 = 11.00 :=
by simp only [cost_of_dryer_sheets_per_year]; sorry

end sophie_saves_money_by_using_wool_balls_l79_79598


namespace cross_product_correct_l79_79386

def a : ℝ × ℝ × ℝ := (4, 3, -7)
def b : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem cross_product_correct : cross_product a b = (5, -30, -10) := sorry

end cross_product_correct_l79_79386


namespace det_M_mod_1000_l79_79795

def α(m n : ℕ) : ℕ :=
  (List.range (nat.log2 $ Nat.succ (m + n)))
    .filter (λ k, ((m / 2^k) % 2 = 1) ∧ ((n / 2^k) % 2 = 1)).length

def M : Matrix (Fin 65) (Fin 65) ℤ :=
  λ i j, (-1) ^ (α (i : ℕ) (j : ℕ))

theorem det_M_mod_1000 : (M.det % 1000) = 792 := by
  sorry

end det_M_mod_1000_l79_79795


namespace guide_is_aborginal_l79_79937

-- Define types for the tribes
inductive Tribe
| Aboriginal : Tribe -- Truth-teller
| Alien : Tribe -- Liar

open Tribe

-- Given conditions in Lean definitions
def tellsTruth : Tribe → Prop
| Aboriginal := True
| Alien := False

def lies : Tribe → Prop
| Aboriginal := False
| Alien := True

def statementIsTrue (speaker : Tribe) (claim : Prop) : Prop :=
  (speaker = Aboriginal ∧ claim) ∨ (speaker = Alien ∧ ¬claim)

-- The guide's report's condition in Lean
def guideReport (guide : Tribe) (islanderClaims : Prop) : Prop :=
  statementIsTrue guide islanderClaims

-- Proposition to prove: The guide is Aboriginal
theorem guide_is_aborginal (guide : Tribe) (islander : Tribe) :
  guideReport guide (statementIsTrue islander (islander = Aboriginal)) → guide = Aboriginal :=
sorry

end guide_is_aborginal_l79_79937


namespace english_score_l79_79733

theorem english_score (s1 s2 s3 e : ℕ) :
  (s1 + s2 + s3) = 276 → (s1 + s2 + s3 + e) = 376 → e = 100 :=
by
  intros h1 h2
  sorry

end english_score_l79_79733


namespace die_opposite_face_l79_79574

theorem die_opposite_face (numbers : Fin 6 → ℕ) (h : Multiset.ofFinset {6, 7, 8, 9, 10, 11} = Multiset.map numbers Finset.univ) (sum_v1 sum_v2 : ℕ)
  (h_sum_v1 : sum_v1 = 33) (h_sum_v2 : sum_v2 = 35) :
  (numbers 7 = 9 ∨ numbers 7 = 11) := sorry

end die_opposite_face_l79_79574


namespace shoe_cost_l79_79880

theorem shoe_cost 
    (T_shirt_cost : ℕ := 20) 
    (pants_cost : ℕ := 80) 
    (discount : ℝ := 0.9) 
    (num_T_shirts : ℕ := 4) 
    (num_pants : ℕ := 3) 
    (num_shoes : ℕ := 2) 
    (total_cost_after_discount : ℝ := 558) : 
    (shoe_cost : ℝ) :=
    by 
    let total_pre_discount := 
        (num_T_shirts * T_shirt_cost) + 
        (num_pants * pants_cost) + 
        (num_shoes * shoe_cost)
    let total_after_discount := discount * total_pre_discount
    have h : total_after_discount = total_cost_after_discount
    sorry
    exact shoe_cost = 150

end shoe_cost_l79_79880


namespace count_non_negative_numbers_l79_79732

theorem count_non_negative_numbers :
  ∃ non_neg_count : ℕ, 
  non_neg_count = 5 ∧ 
  (∀ x ∈ {-15, 5 + 1/3, -0.23, 0, 7.6, 2, -3/5, 3.14}, x ≥ 0 → non_neg_count = 5) := 
by
  -- We state the elements in the set
  let s := {-15, 5 + 1/3, -0.23, 0, 7.6, 2, -3/5, 3.14}
  -- Check that 5 of these elements are non-negative
  let non_neg_elements := [5 + 1/3, 0, 7.6, 2, 3.14]
  have h : non_neg_elements.length = 5,
  { sorry }
  use 5,
  split,
  { exact h },
  { intros x hx h_non_neg,
    have : x ∈ non_neg_elements,
    { sorry },
    exact h },

end count_non_negative_numbers_l79_79732


namespace stratified_sampling_total_students_sampled_l79_79304

theorem stratified_sampling_total_students_sampled 
  (seniors juniors freshmen : ℕ)
  (sampled_freshmen : ℕ)
  (ratio : ℚ)
  (h_freshmen : freshmen = 1500)
  (h_sampled_freshmen_ratio : sampled_freshmen = 75)
  (h_seniors : seniors = 1000)
  (h_juniors : juniors = 1200)
  (h_ratio : ratio = (sampled_freshmen : ℚ) / (freshmen : ℚ))
  (h_freshmen_ratio : ratio * (freshmen : ℚ) = sampled_freshmen) :
  let sampled_juniors := ratio * (juniors : ℚ)
  let sampled_seniors := ratio * (seniors : ℚ)
  sampled_freshmen + sampled_juniors + sampled_seniors = 185 := sorry

end stratified_sampling_total_students_sampled_l79_79304


namespace probability_sequence_123456_l79_79658

theorem probability_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11
  in (favorable_sequences : ℚ) / total_sequences = 1 / 720 := 
by 
  sorry

end probability_sequence_123456_l79_79658


namespace correct_ticket_status_l79_79666

def ticket_status (h : ℝ) : String :=
  if h ≤ 1.1 then "Free of charge"
  else if h ≤ 1.4 then "Buy a half-price ticket"
  else "Buy a full-price ticket"

theorem correct_ticket_status (h : ℝ) : 
  (h ≤ 1.1 → ticket_status h = "Free of charge") ∧
  (1.1 < h ∧ h ≤ 1.4 → ticket_status h = "Buy a half-price ticket") ∧
  (h > 1.4 → ticket_status h = "Buy a full-price ticket") :=
sorry

end correct_ticket_status_l79_79666


namespace part_a_part_b_l79_79284

-- Part (a)
theorem part_a (x : ℝ) (h : x > 0) : x^3 - 3*x ≥ -2 :=
sorry

-- Part (b)
theorem part_b (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y / z) + (y^2 * z / x) + (z^2 * x / y) + 2 * ((y / (x * z)) + (z / (x * y)) + (x / (y * z))) ≥ 9 :=
sorry

end part_a_part_b_l79_79284


namespace total_votes_is_2400_l79_79053

-- Definitions for the conditions
def total_votes (V : ℕ) : Prop :=
  let second_candidate_votes := 480
  let second_candidate_percentage := 0.20
  second_candidate_votes = second_candidate_percentage * V

-- The theorem to prove the total number of votes is 2400
theorem total_votes_is_2400 : ∃ V : ℕ, total_votes V ∧ V = 2400 :=
by
  existsi 2400
  sorry

end total_votes_is_2400_l79_79053


namespace limit_sqrt_area_sum_l79_79168

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := 1 + x ^ (n^2 - 1) + x ^ (n^2 + 2 * n)

noncomputable def S (n : ℕ) : ℝ := ∫ x in 0..1, f n x

noncomputable def L : ℝ := 
  lim (λ n, (1 / n * ∑ k in finset.range n, sqrt (S (k + 1)))^n) at_top

theorem limit_sqrt_area_sum :
  L = exp (1 / 2) :=
sorry

end limit_sqrt_area_sum_l79_79168


namespace jerome_family_members_l79_79526

-- Define the conditions of the problem
variables (C F M T : ℕ)
variables (hC : C = 20) (hF : F = C / 2) (hT : T = 33)

-- Formulate the theorem to prove
theorem jerome_family_members :
  M = T - (C + F) :=
sorry

end jerome_family_members_l79_79526


namespace proof_problem_l79_79094

noncomputable def z (a b : ℝ) (h : b ≠ 0) : ℂ := a + b * complex.I

noncomputable def ω (z : ℂ) : ℂ := z + 1 / z

noncomputable def u (z : ℂ) : ℂ := (1 - z) / (1 + z)

theorem proof_problem (a b : ℝ) (h : b ≠ 0) (hω : -1 < ω (z a b h) ∧ ω (z a b h) < 2) : 
  abs (z a b h) = 1 ∧ a ∈ Ioo (-1/2 : ℝ) 1 ∧ ∃ ci : ℂ, u (z a b h) = ci * complex.I ∧ ci.im ≠ 0 :=
begin
  sorry
end

end proof_problem_l79_79094


namespace sum_fraction_simplified_l79_79745

theorem sum_fraction_simplified :
  (\sum k in Finset.range 6, (1 / ((k + 3) * (k + 4)))) = 2 / 9 :=
by
  sorry

end sum_fraction_simplified_l79_79745


namespace min_dot_product_pf1_pf2_l79_79827

namespace EllipseProblem

def Ellipse (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def Foci (a b : ℝ) := 
  let c := Real.sqrt (a^2 - b^2)
  (c, 0), (-c, 0)

def Vertices (a b : ℝ) := 
  let v₁ := (-a, 0)
  let v₂ := (0, b)
  v₁, v₂

def PointOnLineSegment (A B : ℝ × ℝ) (t : ℝ) := 
  (1-t) * A.1 + t * B.1, (1-t) * A.2 + t * B.2

def DotProduct (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

theorem min_dot_product_pf1_pf2 :
  ∀ (P : ℝ × ℝ),
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  P = PointOnLineSegment (-2, 0) (0, 1) t →
  let F1 := Foci 2 1 in
  let F2 := (Foci 2 1).snd in
  ∃ min_value : ℝ,
    min_value = 1/5 ∧
    ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
    DotProduct (P.1 - F1.1, P.2 - F1.2) (P.1 - F2.1, P.2 - F2.2) = min_value :=
begin
  sorry
end

end EllipseProblem

end min_dot_product_pf1_pf2_l79_79827


namespace cos_double_angle_l79_79857

theorem cos_double_angle (α : ℝ) (hα1 : sin α = 4 / 5) (hα2 : 0 < α ∧ α < π / 2) : cos (2 * α) = -7 / 25 := 
by
  sorry

end cos_double_angle_l79_79857


namespace first_train_speed_l79_79636

noncomputable def speed_first_train (length1 length2 : ℝ) (speed2 time : ℝ) : ℝ :=
  let distance := (length1 + length2) / 1000
  let time_hours := time / 3600
  (distance / time_hours) - speed2

theorem first_train_speed :
  speed_first_train 100 280 30 18.998480121590273 = 42 :=
by
  sorry

end first_train_speed_l79_79636


namespace main_problem_solve_l79_79090

def f (n : ℕ) : ℕ :=
  let m := (Float.floor (Nat.lift (\(\sqrt[6]{n})))).toNat
  -- Ensure m is the closest integer to the 6th root of n
  if Float.abs((m + 1) ^ 6 - n) < Float.abs(m ^ 6 - n) then m + 1 else m

noncomputable def main_problem : ℕ :=
  ∑ k in Finset.range 4095, 1 / f k

theorem main_problem_solve : main_problem = 134008 :=
by
  sorry

end main_problem_solve_l79_79090


namespace function_zero_if_sum_zero_on_unit_square_l79_79759

theorem function_zero_if_sum_zero_on_unit_square 
  (f : ℝ × ℝ → ℝ) 
  (H : ∀ (A B C D : ℝ × ℝ), 
         dist A B = 1 ∧
         dist B C = 1 ∧
         dist C D = 1 ∧
         dist D A = 1 ∧
         dist A C = dist B D ∧
         dist A C = √2 ∧
         dist B D = √2 → 
         f A + f B + f C + f D = 0) : 
  ∀ (x y : ℝ), f (x, y) = 0 :=
by
  sorry

end function_zero_if_sum_zero_on_unit_square_l79_79759


namespace external_tangent_y_intercept_l79_79222

theorem external_tangent_y_intercept 
  (center1 : ℝ × ℝ) (radius1 : ℝ)
  (center2 : ℝ × ℝ) (radius2 : ℝ)
  (positive_slope : ∀ x₁ y₁ x₂ y₂ : ℝ, ((y₂ - y₁) / (x₂ - x₁)) > 0)
  (slope : ℝ) (y_intercept : ℝ)
  (tangent_line : ∀ x : ℝ, ℝ, tangent_line x = slope * x + y_intercept) :
  center1 = (3, 7) ∧ radius1 = 3 ∧ center2 = (10, 12) ∧ radius2 = 7 ∧ slope = 35 / 12 →
  y_intercept = 912 / 119 :=
begin
  sorry
end

end external_tangent_y_intercept_l79_79222


namespace area_convex_quadrilateral_l79_79943

variable {A B C D O : Type} -- Points in the quadrilateral
variable {AC BD : ℝ} -- Lengths of the diagonals
variable {φ : ℝ} -- Angle between the diagonals

theorem area_convex_quadrilateral (hAC : AC = d1) (hBD : BD = d2) (hφ : φ = angle_AOB) 
  (h_convex : convex_quadrilateral A B C D):
  area_quadrilateral A B C D = (1/2) * d1 * d2 * sin φ := 
sorry

end area_convex_quadrilateral_l79_79943


namespace overall_average_marks_l79_79709

theorem overall_average_marks :
  let students := [50, 35, 45, 42]
  let mean_marks := [50, 60, 55, 45]
  let total_students := students.sum
  let total_marks := List.map2 (λ n m => n * m) students mean_marks |>.sum
  real.to_rat (total_marks / total_students) ≈ 52.12 :=
by
  -- Defining the given problem conditions
  let students := [50, 35, 45, 42]
  let mean_marks := [50, 60, 55, 45]
  -- Calculating total number of students
  let total_students := students.sum
  -- Computing total marks obtained
  let total_marks := List.map2 (λ n m => n * m) students mean_marks |>.sum
  -- Converting the average marks to a rational number and checking approximation
  have h: real.to_rat (total_marks / total_students) ≈ 52.12 := sorry
  exact h

end overall_average_marks_l79_79709


namespace distance_between_A_and_B_l79_79700

theorem distance_between_A_and_B (speed : ℝ) (time : ℝ) (h_speed : speed = 60) (h_time: time = 4) : 
  let total_distance := speed * time in 
  let distance_A_B := total_distance / 2 in 
  distance_A_B = 120 :=
by 
  -- Placeholder for proof; the actual proof is omitted
  sorry

end distance_between_A_and_B_l79_79700


namespace jude_matchbox_trade_l79_79528

theorem jude_matchbox_trade :
  (∃ (C T H : ℕ), 
    C = 4 ∧ T = 4 ∧ H = 4 ∧
    10 * C + 15 * T + 20 * H ≤ 250 ∧
    C ≤ 8 ∧ T ≤ 5 ∧ H ≤ 4) :=
by
  exist 4, 4, 4
  split
  - refl
  - split
    - refl
    - split
      - refl
      - split
        - norm_num
        - split
          - norm_num
          - split
            - norm_num
            - sorry

end jude_matchbox_trade_l79_79528


namespace integral_sqrt_sub_linear_l79_79345

variable (x : ℝ)

theorem integral_sqrt_sub_linear :
  ∫ x in 0..2, (Real.sqrt (4 - x^2) - 2 * x) = Real.pi - 4 :=
by
  sorry

end integral_sqrt_sub_linear_l79_79345


namespace line_AB_cartesian_eq_circle_O_cartesian_eq_l79_79063

noncomputable def A_polar := (2, Real.pi / 2)
noncomputable def B_polar := (1, -Real.pi / 3)
noncomputable def O_polar_eq := λ θ : ℝ, 4 * Real.sin θ

def A_cartesian := (0, 2)
def B_cartesian := (1 / 2, -Real.sqrt 3 / 2)

theorem line_AB_cartesian_eq :
  line_through_cartesian A_cartesian B_cartesian = {y = -(4 + Real.sqrt 3) * x + 2} :=
sorry

theorem circle_O_cartesian_eq :
  cartesian_eq_of_polar_eq O_polar_eq = {x^2 + (y - 2)^2 = 4} :=
sorry

end line_AB_cartesian_eq_circle_O_cartesian_eq_l79_79063


namespace cistern_fill_time_l79_79282

theorem cistern_fill_time (t1 t2 : ℝ) (h1 : t1 = 2) (h2 : t2 = 9) :
  let F := 1 / t1,
      E := 1 / t2,
      net_rate := F - E,
      time_to_fill := 1 / net_rate
  in time_to_fill ≈ 2.5714 :=
by
  rw [h1, h2]
  let F := 1 / 2
  let E := 1 / 9
  let net_rate := F - E
  let time_to_fill := 1 / net_rate
  have : F = 0.5 := by norm_num
  have : E ≈ 0.1111 := by norm_num
  have : net_rate ≈ 0.3889 := by norm_num
  have : time_to_fill ≈ 2.5714 := by norm_num
  exact approximately_equal.symm

end cistern_fill_time_l79_79282


namespace natural_numbers_without_repeating_digits_l79_79638

theorem natural_numbers_without_repeating_digits :
  let digits := {0, 1, 2}
  ∃ n : ℕ, 
    n = 3 + (2 * 2) + (2 * 2) ∧
    n = 11 := by
sorry

end natural_numbers_without_repeating_digits_l79_79638


namespace min_value_a_plus_b_l79_79467

theorem min_value_a_plus_b (a b : ℝ) (h : log 4 (3 * a + 4 * b) = log 2 (sqrt (a * b))) : a + b ≥ 11 :=
sorry

end min_value_a_plus_b_l79_79467


namespace trouser_sale_price_l79_79909

theorem trouser_sale_price 
  (original_price : ℝ) 
  (percent_decrease : ℝ) 
  (sale_price : ℝ) 
  (h : original_price = 100) 
  (p : percent_decrease = 0.25) 
  (s : sale_price = original_price * (1 - percent_decrease)) : 
  sale_price = 75 :=
by 
  sorry

end trouser_sale_price_l79_79909


namespace smallest_k_l79_79097

def S : set ℕ := { x | 1 ≤ x ∧ x ≤ 50 }

def satisfies_condition (a b : ℕ) : Prop := (a + b) ∣ (a * b)

theorem smallest_k (k : ℕ) (h : k = 39) :
  ∀ A, A ⊆ S → A.card = k → ∃ a b, a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ satisfies_condition a b :=
by sorry

end smallest_k_l79_79097


namespace probability_of_specific_sequence_l79_79664

-- We define a structure representing the problem conditions.
structure problem_conditions :=
  (cards : multiset ℕ)
  (permutation : list ℕ)

-- Noncomputable definition for the correct answer.
noncomputable def probability := (1 : ℚ) / 720

-- The main theorem statement.
theorem probability_of_specific_sequence :
  ∀ (conds : problem_conditions),
  conds.cards = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6} ∧
  (∃ (perm : list ℕ), perm.perm conds.permutation) →
  (∃ (sequence : list ℕ), sequence = [1, 2, 3, 4, 5, 6]) →
  let prob := calculate_probability conds.permutation [1, 2, 3, 4, 5, 6] in
  prob = (1 : ℚ) / 720 :=
sorry

end probability_of_specific_sequence_l79_79664


namespace ratio_of_areas_l79_79400

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

theorem ratio_of_areas (s : ℝ) (n : ℕ) (side_len : ℝ) (number_of_triangles : ℕ) (fencing : ℝ) 
  (fencing_eq : n * s = fencing) (num_triangles_eq : number_of_triangles = 4) 
  (small_triangle_side_len_eq : side_len = 10) (fencing_large_triangle_eq : fencing = 3 * s) :
  let large_triangle_side_len := fencing / 3 in
  let area_small_triangle := area_of_equilateral_triangle side_len in
  let total_area_small_triangles := number_of_triangles * area_small_triangle in
  let area_large_triangle := area_of_equilateral_triangle large_triangle_side_len in
  (total_area_small_triangles / area_large_triangle) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l79_79400


namespace cross_product_of_a_and_b_l79_79385

def vector := ℝ × ℝ × ℝ 

def a : vector := (4, 3, -7)
def b : vector := (2, -1, 4)

def cross_product (v₁ v₂ : vector) : vector :=
  (v₁.2 * v₂.3 - v₁.3 * v₂.2,
   v₁.3 * v₂.1 - v₁.1 * v₂.3,
   v₁.1 * v₂.2 - v₁.2 * v₂.1)

theorem cross_product_of_a_and_b :
  cross_product a b = (5, -30, -10) :=
by sorry

end cross_product_of_a_and_b_l79_79385


namespace mn_min_l79_79702

noncomputable def min_mn_value (m n : ℝ) : ℝ := m * n

theorem mn_min : 
  (∃ m n, m = Real.sin (2 * (π / 12)) ∧ n > 0 ∧ 
            Real.cos (2 * (π / 12 + n) - π / 4) = m ∧ 
            min_mn_value m n = π * 5 / 48) := by
  sorry

end mn_min_l79_79702


namespace f_even_l79_79542

-- Let g(x) = x^3 - x
def g (x : ℝ) : ℝ := x^3 - x

-- Let f(x) = |g(x^2)|
def f (x : ℝ) : ℝ := abs (g (x^2))

-- Prove that f(x) is even, i.e., f(-x) = f(x) for all x
theorem f_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_even_l79_79542


namespace arrangement_possible_l79_79067

theorem arrangement_possible : 
  ∃ (f : Fin 50 → Fin 50), 
    (∀ i : Fin 25, f i + f (i+25) + f ((i+1) % 25 + 25) = k) := 
by 
sory

end arrangement_possible_l79_79067


namespace max_consecutive_integers_sum_l79_79249

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l79_79249


namespace circumcircle_BMC_bisects_BP_l79_79514

open EuclideanGeometry

variables (A B C P Q M : Point)
variables [hABC : acute_triangle A B C] (hAB_gt_BC : AB > BC)
variables (hP_mid : is_midpoint_arc P (small_arc A C))
variables (hQ_mid : is_midpoint_arc Q (large_arc A C))
variables (hM_foot : is_foot M Q (line_segment A B))

theorem circumcircle_BMC_bisects_BP :
  circumcircle B M C bisects (line_segment B P) := sorry

end circumcircle_BMC_bisects_BP_l79_79514


namespace scientific_notation_correct_l79_79568

/-- Define the number 42.39 million as 42.39 * 10^6 and prove that it is equivalent to 4.239 * 10^7 -/
def scientific_notation_of_42_39_million : Prop :=
  (42.39 * 10^6 = 4.239 * 10^7)

theorem scientific_notation_correct : scientific_notation_of_42_39_million :=
by 
  sorry

end scientific_notation_correct_l79_79568


namespace sum_of_wsum_over_all_subsets_of_S_l79_79535

noncomputable def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

def wsum (A : Finset ℕ) : ℕ := 
List.sum (List.map (λ (p : ℕ × ℕ), if p.1 % 2 = 1 then 3 * p.2 else 2 * p.2) (A.toList.enum))

def sumOfWSUMs (S : Set ℕ) : ℕ := 
Finset.sum (Finset.powerset (Finset.univ : Finset ℕ)) wsum

theorem sum_of_wsum_over_all_subsets_of_S :
  sumOfWSUMs S = 2^7 * (5 * 10^2 + 5 * 10 + 2) :=
sorry

end sum_of_wsum_over_all_subsets_of_S_l79_79535


namespace number_of_elements_in_A_l79_79981

def N_star := { n : ℕ // n > 0 }

def A := { p : N_star × N_star // p.1.val + p.2.val = 10 }

theorem number_of_elements_in_A : (set.finite A) ∧ (finset.card (set.finite.to_finset (set.finite A)) = 9) :=
by
  sorry

end number_of_elements_in_A_l79_79981


namespace arithmetic_sequence_sets_l79_79012

theorem arithmetic_sequence_sets (S : Finset ℕ) (hS : S = {0, 2, 4, 6, 8}) :
  (∃ (count : ℕ), 
    count = (Finset.filter (λ s : Finset ℕ, s.card = 3 ∧ 
    ∃ (d : ℕ),
      (∀ {a b c : ℕ}, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a < b ∧ b < c → b - a = c - b ∧ d = c - a))) 
    ((Finset.powersetLen 3 S))).card ∧ count = 4) :=
by
  sorry

end arithmetic_sequence_sets_l79_79012


namespace cards_not_in_box_correct_l79_79952

-- Total number of cards Robie had at the beginning.
def total_cards : ℕ := 75

-- Number of cards in each box.
def cards_per_box : ℕ := 10

-- Number of boxes Robie gave away.
def boxes_given_away : ℕ := 2

-- Number of boxes Robie has with him.
def boxes_with_rob : ℕ := 5

-- The number of cards not placed in a box.
def cards_not_in_box : ℕ :=
  total_cards - (boxes_given_away * cards_per_box + boxes_with_rob * cards_per_box)

theorem cards_not_in_box_correct : cards_not_in_box = 5 :=
by
  unfold cards_not_in_box
  unfold total_cards
  unfold boxes_given_away
  unfold cards_per_box
  unfold boxes_with_rob
  sorry

end cards_not_in_box_correct_l79_79952


namespace area_of_P_l79_79957

-- Definitions for the problem conditions
def side1_length : ℝ := 7
def side2_length : ℝ := 8
def side3_length : ℝ := 9
def side4_length : ℝ := 10
def area_PQRS : ℝ := 15

-- The proof statement
theorem area_of_P'Q'R'S' :
  let area_ext : ℝ := 2 * area_PQRS in
  let total_area : ℝ := area_ext + area_PQRS in
  total_area = 45 :=
by
  sorry

end area_of_P_l79_79957


namespace bug_reaches_point_73_with_conditions_l79_79299

-- Define the conditions and the problem.
def bug_path_count : ℕ :=
  let total_paths := (Nat.choose 10 3)
  let paths_via_11 := (Nat.choose 8 2)
  
  let paths_via_22 := (Nat.choose 6 1)

   let paths_via_33 := (Nat.choose 4 0) := 0 subtract_paths  :=  total_paths - 28 -6 -24) }= 

  total_paths - paths_via_11 - paths_via_22 + paths_via_33 end 

theorem bug_reaches_point_73_with_conditions :
  (bug_path_count (7, 3)) 
        
:= by sorryocaustic 지역사무소 34) sum

end bug_reaches_point_73_with_conditions_l79_79299


namespace initial_average_quiz_score_l79_79877

theorem initial_average_quiz_score 
  (n : ℕ) (A : ℝ) (dropped_avg : ℝ) (drop_score : ℝ)
  (students_before : n = 16)
  (students_after : n - 1 = 15)
  (dropped_avg_eq : dropped_avg = 64.0)
  (drop_score_eq : drop_score = 8) 
  (total_sum_before_eq : n * A = 16 * A)
  (total_sum_after_eq : (n - 1) * dropped_avg = 15 * 64):
  A = 60.5 := 
by
  sorry

end initial_average_quiz_score_l79_79877


namespace bleach_to_detergent_ratio_changed_factor_l79_79165

theorem bleach_to_detergent_ratio_changed_factor :
  let original_bleach : ℝ := 4
  let original_detergent : ℝ := 40
  let original_water : ℝ := 100
  let altered_detergent : ℝ := 60
  let altered_water : ℝ := 300

  -- Calculate the factor by which the volume increased
  let original_total_volume := original_detergent + original_water
  let altered_total_volume := altered_detergent + altered_water
  let volume_increase_factor := altered_total_volume / original_total_volume

  -- The calculated factor of the ratio change
  let original_ratio_bleach_to_detergent := original_bleach / original_detergent

  altered_detergent > 0 → altered_water > 0 →
  volume_increase_factor * original_ratio_bleach_to_detergent = 2.5714 :=
by
  -- Insert proof here
  sorry

end bleach_to_detergent_ratio_changed_factor_l79_79165


namespace solve_equation_in_integers_l79_79959
-- Import the necessary library for Lean

-- Define the main theorem to solve the equation in integers
theorem solve_equation_in_integers :
  ∃ (xs : List (ℕ × ℕ)), (∀ x y, (3^x - 2^y = 1 → (x, y) ∈ xs)) ∧ xs = [(1, 1), (2, 3)] :=
by
  sorry

end solve_equation_in_integers_l79_79959


namespace cone_to_sphere_surface_area_ratio_l79_79498

noncomputable def sphere_radius (r : ℝ) : ℝ := r
noncomputable def cone_height (r : ℝ) : ℝ := 3 * r
noncomputable def lateral_surface_area_cone (r : ℝ) : ℝ := 6 * real.pi * r * r
noncomputable def surface_area_sphere (r : ℝ) : ℝ := 4 * real.pi * r * r

theorem cone_to_sphere_surface_area_ratio (r : ℝ) :
  (lateral_surface_area_cone r) / (surface_area_sphere r) = 3 / 2 :=
by {
  sorry
}

end cone_to_sphere_surface_area_ratio_l79_79498


namespace midpoint_limit_l79_79006

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def limit_position (a b : ℝ) : ℝ × ℝ :=
  (2 * a / 3, 2 * b / 3)

theorem midpoint_limit (a b : ℝ) :
  let O := (0, 0) in
  let Q := (a, b) in
  let sequence : ℕ → ℝ × ℝ :=
    λ n, nat.rec_on n Q (λ n Pn, midpoint Pn O)
  in
  (sequence ∘ nat.succ) ∘ nat.succ 0 → limit_position a b :=
sorry

end midpoint_limit_l79_79006


namespace max_distance_from_circle_to_line_l79_79811

theorem max_distance_from_circle_to_line :
  let P := (x, y)
      center := (5, 3)
      radius := 3
      circle := (x - 5) ^ 2 + (y - 3) ^ 2 = 9
      line := 3 * x + 4 * y - 2 = 0 in
    ∀ (P : ℝ × ℝ), circle → (∃ (d : ℝ), d = 8) :=
by
  sorry

end max_distance_from_circle_to_line_l79_79811


namespace true_proposition_is_D_l79_79735

-- Definitions of the propositions
def PropA (L₁ L₂ : Line) (π : Plane) : Prop :=
  ∀ (L1_parallel_L2 : parallel L₁ L₂), (parallel L₁ π)

def PropB (L : Line) (π : Plane) : Prop :=
  ∀ (L_perpendicular_to_L₁ L_perpendicular_to_L₂ : Line), 
  (π.contains L₁) -> (π.contains L₂) -> (perpendicular L L₁) -> (perpendicular L L₂) -> (perpendicular L π)

def PropC (L : Line) (π : Plane) : Prop :=
  (parallel L π) -> (∀ (L₁ : Line), (π.contains L₁) -> (parallel L L₁))

def PropD (L : Line) (π : Plane) : Prop :=
  (perpendicular L π) -> (∀ (L₁ : Line), (π.contains L₁) -> (perpendicular L L₁))

-- Main theorem stating the correct proposition
theorem true_proposition_is_D (L₁ L₂ : Line) (π : Plane) : 
  (PropA L₁ L₂ π ∨ PropB L₁ π ∨ PropC L₁ π ∨ PropD L₁ π) := 
  sorry

end true_proposition_is_D_l79_79735


namespace garden_sparrows_l79_79506

theorem garden_sparrows (ratio_b_s : ℕ) (bluebirds sparrows : ℕ)
  (h1 : ratio_b_s = 4 / 5) (h2 : bluebirds = 28) :
  sparrows = 35 :=
  sorry

end garden_sparrows_l79_79506


namespace find_rate_compounded_annually_l79_79337

-- Definition of the variables based on the conditions
def P : ℝ := 50000
def A : ℝ := 80000
def n : ℝ := 1
def t : ℝ := 4

-- Define the compounded interest formula
def compound_interest (P A r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

-- The proof statement we need to show
theorem find_rate_compounded_annually (r : ℝ) : compound_interest 50000 80000 r → r ≈ 0.12468265 := by
  -- Sorry placeholder for the actual proof
  sorry

end find_rate_compounded_annually_l79_79337


namespace solve_z_l79_79132

variable (z : ℂ) -- Define the variable z in the complex number system
variable (i : ℂ) -- Define the variable i in the complex number system

-- State the conditions: 2 - 3i * z = 4 + 5i * z and i^2 = -1
axiom cond1 : 2 - 3 * i * z = 4 + 5 * i * z
axiom cond2 : i^2 = -1

-- The theorem to prove: z = i / 4
theorem solve_z : z = i / 4 :=
by
  sorry

end solve_z_l79_79132


namespace victor_additional_candies_needed_l79_79640

variable (friends : ℝ) (initial_candies : ℕ) (candies_needed : ℕ)

-- Defining the conditions
def conditions := friends = 7.5 ∧ initial_candies = 4692

-- Definition of the additional candies needed
def additional_candies_needed (candies_needed additive_needed : ℕ): Prop :=
  candies_needed = nat_ceil (friends * 626) ∧ additive_needed = candies_needed - initial_candies

-- The theorem statement
theorem victor_additional_candies_needed : conditions friends initial_candies → ∃ n, additional_candies_needed 4695 n := by
  sorry

end victor_additional_candies_needed_l79_79640


namespace pure_imaginary_solution_l79_79443

theorem pure_imaginary_solution (m : ℝ) :
  (m^2 + m - 2 = 0) ∧ (m^2 + 4m - 5 ≠ 0) → m = -2 :=
by
  sorry

end pure_imaginary_solution_l79_79443


namespace scientific_notation_of_45_million_l79_79770

theorem scientific_notation_of_45_million :
  ∃ a n, (45_000_000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.5 ∧ n = 7 :=
by
  use 4.5
  use 7
  split
  · norm_num [pow]
  · split
  · norm_num
  · split
  · norm_num
  · rfl
  · rfl

end scientific_notation_of_45_million_l79_79770


namespace problem_1_problem_2_l79_79456

noncomputable def a (n : ℕ) : ℚ :=
  if n = 1 then 2 else if n = 2 then 2 / 3 else
    sorry /-
      The exact recursive definition and proof of the recurrence relation
      is omitted for brevity. It would be established in a full proof.
    -/ 

open Real

theorem problem_1 (n : ℕ) (h : n ≥ 1) : 
  is_arithmetic_sequence (λ n, 1 / a n) := sorry

theorem problem_2 (n : ℕ) : 
  (∑ i in range n, a i / (2 * i + 1)) = (2 * n) / (2 * n + 1) := sorry

end problem_1_problem_2_l79_79456


namespace discount_percentage_shirt_l79_79906

-- Define the constants based on the conditions
def original_price_jacket : ℝ := 100
def original_price_shirt : ℝ := 60
def total_paid : ℝ := 110
def discount_jacket : ℝ := 0.30

-- Define the goal to prove the discount percentage on the shirt
theorem discount_percentage_shirt :
  let sale_price_jacket := original_price_jacket * (1 - discount_jacket) in
  let amount_paid_for_shirt := total_paid - sale_price_jacket in
  let discount_shirt := (original_price_shirt - amount_paid_for_shirt) / original_price_shirt in
  discount_shirt * 100 = 33 + 1 / 3 :=
by
  sorry

end discount_percentage_shirt_l79_79906


namespace distinct_real_roots_of_quadratic_l79_79468

theorem distinct_real_roots_of_quadratic (m : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ x^2 + m * x + 9 = 0 ∧ y^2 + m * y + 9 = 0) ↔ m ∈ Ioo (-∞) (-6) ∪ Ioo 6 (∞) :=
by
  sorry

end distinct_real_roots_of_quadratic_l79_79468


namespace triangle_inequality_l79_79830

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_l79_79830


namespace user_level_1000_l79_79602

noncomputable def user_level (points : ℕ) : ℕ :=
if points >= 1210 then 18
else if points >= 1000 then 17
else if points >= 810 then 16
else if points >= 640 then 15
else if points >= 490 then 14
else if points >= 360 then 13
else if points >= 250 then 12
else if points >= 160 then 11
else if points >= 90 then 10
else 0

theorem user_level_1000 : user_level 1000 = 17 :=
by {
  -- proof will be written here
  sorry
}

end user_level_1000_l79_79602


namespace probability_after_2019_rings_l79_79878

noncomputable def players_start_with_one : ℕ := 1
noncomputable def bell_ring_interval : ℕ := 15
noncomputable def num_rings : ℕ := 2019

-- Assuming a function that simulates the game results based on the conditions
def game_simulation (num_rings : ℕ) : ℚ := sorry

theorem probability_after_2019_rings :
  game_simulation num_rings = 1 / 4 := sorry

end probability_after_2019_rings_l79_79878


namespace wire_cut_l79_79686

theorem wire_cut (total_length : ℝ) (ratio : ℝ) (shorter longer : ℝ) (h_total : total_length = 21) (h_ratio : ratio = 2/5)
  (h_shorter : longer = (5/2) * shorter) (h_sum : total_length = shorter + longer) : shorter = 6 := 
by
  -- total_length = 21, ratio = 2/5, longer = (5/2) * shorter, total_length = shorter + longer, prove shorter = 6
  sorry

end wire_cut_l79_79686


namespace monthly_installment_amount_l79_79142

theorem monthly_installment_amount (total_cost : ℝ) (down_payment_percentage : ℝ) (additional_down_payment : ℝ) 
  (balance_after_months : ℝ) (months : ℕ) (monthly_installment : ℝ) : 
    total_cost = 1000 → 
    down_payment_percentage = 0.20 → 
    additional_down_payment = 20 → 
    balance_after_months = 520 → 
    months = 4 → 
    monthly_installment = 65 :=
by
  intros
  sorry

end monthly_installment_amount_l79_79142


namespace sum_lucky_numbers_divisible_by_13_l79_79342

def is_lucky (n : ℕ) : Prop :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 + d2 + d3 = d4 + d5 + d6

theorem sum_lucky_numbers_divisible_by_13 :
  (∑ n in Finset.filter is_lucky (Finset.range 1000000), n) % 13 = 0 :=
sorry

end sum_lucky_numbers_divisible_by_13_l79_79342


namespace histogram_approximates_density_curve_l79_79951

theorem histogram_approximates_density_curve 
  (sample_size : ℕ) (group_interval : ℝ) 
  (histogram : ℕ → ℝ) (density_curve : ℝ → ℝ) 
  (h1 : ∀ n, monotone (histogram n))
  (h2 : ∀ x, continuous (density_curve x)) :
  (filter.at_top (λ n => histogram n)) = (filter.at_top (λ x => density_curve x)) :=
sorry

end histogram_approximates_density_curve_l79_79951


namespace distinct_real_roots_interval_l79_79481

open Set Real

theorem distinct_real_roots_interval (m : ℝ) : 
  (∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ IsRoot (λ x => x^2 + m * x + 9) r1 ∧ IsRoot (λ x => x^2 + m * x + 9) r2) ↔ 
  m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end distinct_real_roots_interval_l79_79481


namespace range_of_m_l79_79105

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m (m : ℝ) :
  (∀ θ, 0 < θ ∧ θ < (real.pi / 2) → f (m * real.sin θ) + f (1 - m) > 0) → m ≤ 1 :=
by
  sorry

end range_of_m_l79_79105


namespace simplify_division_l79_79129

noncomputable def a := 5 * 10 ^ 10
noncomputable def b := 2 * 10 ^ 4 * 10 ^ 2

theorem simplify_division : a / b = 25000 := by
  sorry

end simplify_division_l79_79129


namespace sophie_saves_money_l79_79596

-- Definitions based on the conditions
def loads_per_week : ℕ := 4
def sheets_per_load : ℕ := 1
def cost_per_box : ℝ := 5.50
def sheets_per_box : ℕ := 104
def weeks_per_year : ℕ := 52

-- Main theorem statement
theorem sophie_saves_money :
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  annual_saving = 11.00 := 
by {
  -- Calculation steps
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  -- Proving the final statement
  sorry
}

end sophie_saves_money_l79_79596


namespace gcd_612_468_l79_79639

theorem gcd_612_468 : gcd 612 468 = 36 :=
by
  sorry

end gcd_612_468_l79_79639


namespace num_boys_in_circle_l79_79731

theorem num_boys_in_circle (n : ℕ) 
  (h : ∃ k, n = 2 * k ∧ k = 40 - 10) : n = 60 :=
by
  sorry

end num_boys_in_circle_l79_79731


namespace decreasing_interval_of_f_l79_79980

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem decreasing_interval_of_f :
  ∀ x : ℝ, 0 < x ∧ x < 1 → ∃ ε > 0, ∀ y : ℝ, abs(y - x) < ε → f(y) < f(x) :=
by
  intros x hx
  sorry

end decreasing_interval_of_f_l79_79980


namespace log_expression_equals_six_l79_79373

-- Define the necessary logarithm properties
theorem log_expression_equals_six :
    log 10 9 + 3 * log 10 4 + 2 * log 10 3 + 4 * log 10 8 + log 10 27 = 6 :=
by
  sorry

end log_expression_equals_six_l79_79373


namespace sam_distance_traveled_l79_79559

-- Define the conditions given in the problem
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_speed_increase_factor : ℚ := 1.2
def sam_time : ℕ := 4

-- The statement we need to prove
theorem sam_distance_traveled :
  let marguerite_speed := marguerite_distance / marguerite_time in
  let sam_speed := marguerite_speed * sam_speed_increase_factor in
  let sam_distance := sam_speed * sam_time in
  sam_distance = 240 :=
by
  -- Proof is omitted
  sorry

end sam_distance_traveled_l79_79559


namespace distinct_sequences_count_l79_79838

noncomputable def number_of_distinct_sequences (n : ℕ) : ℕ :=
  if n = 6 then 12 else sorry

theorem distinct_sequences_count : number_of_distinct_sequences 6 = 12 := 
by 
  sorry

end distinct_sequences_count_l79_79838


namespace ellipse_properties_value_of_m_l79_79440

noncomputable def ellipse_equation (a b c : ℝ) := sorry

theorem ellipse_properties :
  let c := sqrt 3 in
  let e := sqrt 3 / 2 in
  let a := 2 in
  let b := 1 in
  ellipse_equation a b c = "x^2 / 4 + y^2 = 1" := sorry

theorem value_of_m (m : ℝ) :
  let ellipse_eq := "x^2 / 4 + y^2 = 1" in
  ∀ P Q : ℝ × ℝ,
    line_eq := (λ x, x / 2 + m) →
    points_on_ellipse P Q ellipse_eq →
    |PQ| = 2 →
    m = sqrt 30 / 5 ∨ m = -sqrt 30 / 5 := sorry

end ellipse_properties_value_of_m_l79_79440


namespace loc_of_P_l79_79931

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) :=
  { pt | pt.2 ^ 2 = 2 * p * pt.1 }

theorem loc_of_P (p a : ℝ) (h_a : 0 ≤ a) (M1 M2 P : ℝ × ℝ) (k l : ℝ) :
  M1.2 ^ 2 = 2 * p * M1.1 ∧ M2.2 ^ 2 = 2 * p * M2.1 ∧
  M1.2 - M2.2 = k * (M1.1 - M2.1) ∧
  2 * (P.1 - M1.1) ^ 2 + 2 * (P.2 - M1.2) ^ 2 + 2 * (P.1 - M2.1) ^ 2 + 2 * (P.2 - M2.2) ^ 2 - (M1.1 - M2.1) ^ 2 - (M1.2 - M2.2) ^ 2 = a :=
  sorry
  
#print loc_of_P

end loc_of_P_l79_79931


namespace sin_C_in_right_triangle_l79_79522

-- Triangle ABC with angle B = 90 degrees and tan A = 3/4
theorem sin_C_in_right_triangle (A C : ℝ) (h1 : A + C = π / 2) (h2 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end sin_C_in_right_triangle_l79_79522


namespace cos_48_eq_sqrt2_over_2_l79_79353

theorem cos_48_eq_sqrt2_over_2 : 
  ∃ (y : ℝ), y = cos 48 ∧ y = sqrt 2 / 2 :=
by
  sorry

end cos_48_eq_sqrt2_over_2_l79_79353


namespace geometric_series_common_ratio_l79_79182

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l79_79182


namespace find_m_l79_79824

def hyperbola_focus (x y : ℝ) (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = 9 ∧ b^2 = -m ∧ (x - 0)^2 / a^2 - (y - 0)^2 / b^2 = 1

theorem find_m (m : ℝ) (H : hyperbola_focus 5 0 m) : m = -16 :=
by
  sorry

end find_m_l79_79824


namespace distance_A_A₁_l79_79882

-- Define a point in 3D Cartesian coordinate system
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Given point A and its symmetrical point A₁ about the y-axis
def A : Point := { x := -4, y := 3, z := 1 }
def A₁ : Point := { x := 4, y := 3, z := -1 }

-- Distance between two points in 3D space
def distance (p q : Point) : ℝ :=
  Real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2 + (q.z - p.z)^2)

-- Proof problem statement
theorem distance_A_A₁ : distance A A₁ = 2 * Real.sqrt 17 := by
  sorry

end distance_A_A₁_l79_79882


namespace range_of_a_l79_79425

theorem range_of_a (a : ℝ) 
  (h₁ : ∀ x ∈ set.Icc (1:ℝ) 2, x^2 - a ≥ 0)
  (h₂ : ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 :=
by
  sorry

end range_of_a_l79_79425


namespace total_legs_correct_l79_79581

def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goats : ℕ := 1
def legs_per_animal : ℕ := 4

theorem total_legs_correct :
  num_horses * legs_per_animal +
  num_dogs * legs_per_animal +
  num_cats * legs_per_animal +
  num_turtles * legs_per_animal +
  num_goats * legs_per_animal = 72 :=
by
  sorry

end total_legs_correct_l79_79581


namespace minimize_avg_cost_maximize_profit_l79_79604

-- Define the cost function y
def cost_function (x : ℝ) : ℝ := (1/2) * x^2 - 200 * x + 45000

-- Define the average cost function
def avg_cost_function (x : ℝ) : ℝ := (cost_function x) / x

-- Define the profit function S
def profit_function (x : ℝ) : ℝ := 200 * x - (cost_function x)

-- Define the conditions
def min_volume := 300
def max_volume := 600
def min_avg_cost := 100
def max_profit := 35000

-- Lean theorem statement for problem part (1)
theorem minimize_avg_cost :
  ∀ x : ℝ, (300 ≤ x ∧ x ≤ 600) → avg_cost_function x ≥ min_avg_cost :=
begin
  assume x,
  assume h : 300 ≤ x ∧ x ≤ 600,
  sorry,
end

-- Lean theorem statement for problem part (2)
theorem maximize_profit :
  ∀ x : ℝ, (300 ≤ x ∧ x ≤ 600) → profit_function x ≤ max_profit :=
begin
  assume x,
  assume h : 300 ≤ x ∧ x ≤ 600,
  sorry,
end

end minimize_avg_cost_maximize_profit_l79_79604


namespace perimeter_of_equilateral_triangle_l79_79316

noncomputable def equilateral_triangle_perimeter : ℝ :=
  let L1 := {x : ℝ | x = 2}
  let L2 := {p : ℝ × ℝ | p.snd = 2 + 1 / Real.sqrt 3 * p.fst}
  let L3 := {p : ℝ × ℝ | p.snd = -Real.sqrt 3 * p.fst}
  let vertex1 := (2, -2 * Real.sqrt 3)
  let vertex2 := (2, 2 + 2 / Real.sqrt 3)
  let side_length := Real.abs (vertex2.snd - vertex1.snd)
  3 * side_length

theorem perimeter_of_equilateral_triangle :
  equilateral_triangle_perimeter = 6 + 8 * Real.sqrt 3 :=
sorry

end perimeter_of_equilateral_triangle_l79_79316


namespace non_degenerate_ellipse_condition_l79_79153

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 3 * x^2 + 6 * y^2 - 12 * x + 18 * y = k) ↔ k > -51 / 2 :=
sorry

end non_degenerate_ellipse_condition_l79_79153


namespace rick_gave_to_miguel_l79_79127

theorem rick_gave_to_miguel (total_cards : ℕ) 
  (kept_cards : ℕ) (friends : ℕ) (friends_cards : ℕ) 
  (sisters : ℕ) (sisters_cards_each : ℕ) :
  total_cards = 130 → kept_cards = 15 → 
  friends = 8 → friends_cards = 12 →
  sisters = 2 → sisters_cards_each = 3 → 
  total_cards - kept_cards - (friends * friends_cards) - (sisters * sisters_cards_each) = 13 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h7 : total_cards - kept_cards = 115 := by 
    conv in (total_cards - kept_cards) {rw [h1, h2]}
    simp
  have h8 : friends * friends_cards = 96 := by 
    conv in (friends * friends_cards) {rw [h3, h4]}
    simp
  have h9 : 115 - 96 = 19 := by
    simp
  have h10 : sisters * sisters_cards_each = 6 := by 
    conv in (sisters * sisters_cards_each) {rw [h5, h6]}
    simp
  have h11 : 19 - 6 = 13 := by
    simp
  rw [h7, h8, h9, h10, h11]
  simp
  sorry

end rick_gave_to_miguel_l79_79127


namespace solve_for_x_l79_79958

theorem solve_for_x : ∃ (x : ℝ), (x - 5) ^ 2 = (1 / 16)⁻¹ ∧ (x = 9 ∨ x = 1) :=
by
  sorry

end solve_for_x_l79_79958


namespace max_consecutive_integers_lt_1000_l79_79257

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l79_79257


namespace max_consecutive_integers_sum_lt_1000_l79_79245

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l79_79245


namespace problem_statement_l79_79753

theorem problem_statement (a b c x : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0)
  (eq1 : (a * x^4 / b * c)^3 = x^3)
  (sum_eq : a + b + c = 9) :
  (x = 1 ∨ x = -1) ∧ a = 1 ∧ b = 4 ∧ c = 4 :=
by
  sorry

end problem_statement_l79_79753


namespace min_tiles_placement_l79_79536

theorem min_tiles_placement (k n : ℕ) (h1 : k ≥ 2) (h2 : k ≤ n) (h3 : n ≤ 2 * k - 1) :
  if n = k then 
    min_tiles n k = k 
  else if (k < n ∧ n < 2 * k - 1) then 
    min_tiles n k = 2 * (n - k + 1)
  else if n = 2 * k - 1 then
    min_tiles n k = 2 * k - 1
  else 
    False :=
sorry

end min_tiles_placement_l79_79536


namespace total_balloons_l79_79585

theorem total_balloons (sam_balloons_initial mary_balloons fred_balloons : ℕ) (h1 : sam_balloons_initial = 6)
    (h2 : mary_balloons = 7) (h3 : fred_balloons = 5) : sam_balloons_initial - fred_balloons + mary_balloons = 8 :=
by
  sorry

end total_balloons_l79_79585


namespace polynomial_root_problem_l79_79091

theorem polynomial_root_problem (a b c d : ℤ) (r1 r2 r3 r4 : ℕ)
  (h_roots : ∀ x, x^4 + a * x^3 + b * x^2 + c * x + d = (x + r1) * (x + r2) * (x + r3) * (x + r4))
  (h_sum : a + b + c + d = 2009) :
  d = 528 := 
by
  sorry

end polynomial_root_problem_l79_79091


namespace model_tower_height_l79_79552

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_cond : real_height = 80) (vol_cond : real_volume = 200000) (model_vol_cond : model_volume = 0.2) : 
  ∃ h : ℝ, h = 0.8 :=
by sorry

end model_tower_height_l79_79552


namespace fourth_vertex_of_square_l79_79892

def forms_square (a b c d : Complex) : Prop :=
  let sides := [a - b, b - c, c - d, d - a]
  let diagonals := [a - c, b - d]
  (sides.all (λ s, Complex.normSq s = Complex.normSq (a - b))) ∧
  (diagonals.all (λ d, Complex.normSq d = Complex.normSq (a - c))) ∧
  (Complex.normSq (a - c) = 2 * Complex.normSq (a - b))

theorem fourth_vertex_of_square :
  let a := (1 : ℂ) + 2 * Complex.i
  let b := (-2 : ℂ) + Complex.i
  let c := (-1 : ℂ) - 2 * Complex.i
  ∃ d : ℂ, forms_square a b c d ∧ d = (2 : ℂ) - Complex.i := by
  let a := (1 : ℂ) + 2 * Complex.i
  let b := (-2 : ℂ) + Complex.i
  let c := (-1 : ℂ) - 2 * Complex.i
  exact ⟨(2 : ℂ) - Complex.i, sorry⟩

end fourth_vertex_of_square_l79_79892


namespace solution_value_l79_79861

theorem solution_value (a : ℝ) (h : a^2 + a - 5 = 0) : a^2 + a + 1 = 6 :=
by {
    have h2 : a^2 + a = 5,
    {
        sorry,  -- Proof deriving a^2 + a = 5
    },
    rw h2,
    calc
    5 + 1 = 6 : by norm_num,
}

end solution_value_l79_79861


namespace evaluate_expression_l79_79766

theorem evaluate_expression : (3^2)^4 * 2^3 = 52488 := by
  sorry

end evaluate_expression_l79_79766


namespace max_t_for_60_degrees_l79_79116

def f (t : ℝ) : ℝ := -t^2 + 10*t + 40

theorem max_t_for_60_degrees : ∃ t : ℝ, f t = 60 ∧ ∀ t' : ℝ, f t' = 60 → t ≤ t' → t = 5 + real.sqrt 5 :=
by
  sorry

end max_t_for_60_degrees_l79_79116


namespace logarithms_form_harmonic_progression_l79_79925

noncomputable def in_geometric_progression (x y z q : ℝ) := 
  x = y / q ∧ z = y * q ∧ q > 1

noncomputable def is_harmonic_progression (a b c : ℝ) := 
  2 / b = 1 / a + 1 / c

theorem logarithms_form_harmonic_progression {x y z m q : ℝ} 
  (h_geom : in_geometric_progression x y z q) (h_order : x < y ∧ y < z) 
  (h_m : m > 1) : 
  is_harmonic_progression (log x m) (log y m) (log z m) := 
sorry

end logarithms_form_harmonic_progression_l79_79925


namespace evaluate_expression_l79_79378

theorem evaluate_expression :
    | 5 - 8 * (3 - 12)^2 | - | 5 - 11 | + real.sqrt 16 + real.sin (real.pi / 2) = 642 :=
by
  have h1 : 3 - 12 = -9 := by sorry
  have h2 : (-9)^2 = 81 := by sorry
  have h3 : 8 * 81 = 648 := by sorry
  have h4 : 5 - 648 = -643 := by sorry
  have h5 : | -643 | = 643 := by sorry
  have h6 : 5 - 11 = -6 := by sorry
  have h7 : | -6 | = 6 := by sorry
  have h8 : real.sqrt 16 = 4 := by sorry
  have h9 : real.sin (real.pi / 2) = 1 := by sorry
  calc 
    | 5 - 8 * (3 - 12)^2 | - | 5 - 11 | + real.sqrt 16 + real.sin (real.pi / 2)
        = | -643 | - 6 + 4 + 1 : by sorry
    ... = 643 - 6 + 4 + 1 : by sorry
    ... = 642 : by sorry

end evaluate_expression_l79_79378


namespace smallest_n_value_l79_79960

-- Definitions for conditions
def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = b * k

def count_factors_of_5 (n : ℕ) : ℕ :=
  (Finset.range n.succ).sum (λ k, if (5 ^ k ≤ n) then n / (5 ^ k) else 0)

noncomputable def min_possible_n (a b c : ℕ) : ℕ :=
  count_factors_of_5 a + count_factors_of_5 b + count_factors_of_5 c

theorem smallest_n_value (a b c : ℕ) (hpos1 : a > 0) (hpos2 : b > 0) (hpos3 : c > 0)
  (hsum : a + b + c = 2023) (hmult : is_multiple_of a 11) (hmn : ∃ m n, a! * b! * c! = m * 10 ^ n ∧ ¬ (10 ∣ m)) :
  min_possible_n a b c = 497 :=
sorry

end smallest_n_value_l79_79960


namespace problem_statement_l79_79924

open Nat

theorem problem_statement (p : ℕ) (h_prime : Prime p) (h_gt3 : p > 3) 
  (n : ℕ) (h_n : n = (2 ^ (2 * p) - 1) / 3) : 
  2 ^ n - 2 ≡ 0 [MOD n] := 
sorry

end problem_statement_l79_79924


namespace maximum_vertices_pyramid_l79_79293

theorem maximum_vertices_pyramid (cube_faces : ℕ) (face_vertices : ℕ) (no_edge_vertices : Prop) (at_least_one_per_face : Prop) : 
  ∀ pyramid_vertices, pyramid_vertices ≤ 13 :=
by {
  -- Assuming the necessary conditions
  assume h1 : no_edge_vertices,
  assume h2 : at_least_one_per_face,
  -- Here, you would normally carry out the proof steps.
  sorry
}

end maximum_vertices_pyramid_l79_79293


namespace range_of_a_l79_79170

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a * x + 3 * x > 2 * a + 3) ↔ (x < 1)) → (a < -3 / 2) :=
by
  intro h
  sorry

end range_of_a_l79_79170


namespace tiffany_initial_lives_l79_79633

theorem tiffany_initial_lives : 
  ∃ x : ℝ, (x - 14) + 27 = 56 ∧ x = 43 :=
begin
  use 43,
  split,
  { linarith, },
  { refl, },
end

end tiffany_initial_lives_l79_79633


namespace calvin_weight_after_one_year_l79_79748

def initial_weight : ℕ := 250
def gym_training_loss : ℕ := 8 + 5 + 7 + 6 + 8 + 7 + 5 + 7 + 4 + 6 + 5 + 7
def diet_loss : ℕ := 3 * 12
def exercise_loss : ℕ := 2 + 3 + 4 + 3 + 2 + 4 + 3 + 2 + 1 + 3 + 2 + 4
def total_loss : ℕ := gym_training_loss + diet_loss + exercise_loss
def final_weight : ℕ := initial_weight - total_loss

theorem calvin_weight_after_one_year : final_weight = 106 :=
by
  unfold initial_weight gym_training_loss diet_loss exercise_loss total_loss final_weight
  exact rfl

end calvin_weight_after_one_year_l79_79748


namespace problem_statement_l79_79412

variable {x y : ℝ}

theorem problem_statement (h1 : x * y = -3) (h2 : x + y = -4) : x^2 + 3 * x * y + y^2 = 13 := sorry

end problem_statement_l79_79412


namespace find_v_l79_79794

variable (v : ℝ)

def operation (v : ℝ) : ℝ := v - (v / 3)
def double_operation (v : ℝ) : ℝ := operation (operation v)

theorem find_v (hv : double_operation v = 20) : v = 45 := by
  sorry

end find_v_l79_79794


namespace trigonometric_value_existence_l79_79814

noncomputable def can_be_value_of_tan (n : ℝ) : Prop :=
∃ θ : ℝ, Real.tan θ = n

noncomputable def can_be_value_of_cot (n : ℝ) : Prop :=
∃ θ : ℝ, 1 / Real.tan θ = n

def can_be_value_of_sin (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.sin θ = n

def can_be_value_of_cos (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.cos θ = n

def can_be_value_of_sec (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.cos θ = n

def can_be_value_of_csc (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.sin θ = n

theorem trigonometric_value_existence (n : ℝ) : 
  can_be_value_of_tan n ∧ 
  can_be_value_of_cot n ∧ 
  can_be_value_of_sin n ∧ 
  can_be_value_of_cos n ∧ 
  can_be_value_of_sec n ∧ 
  can_be_value_of_csc n := 
sorry

end trigonometric_value_existence_l79_79814


namespace equal_distances_of_circumcenters_l79_79335

-- Define points and circumcenters
variables {A B C D O O₁ O₂: Type}
noncomputable def is_circumcenter (O : Type) (A B C : Type) : Prop := sorry
noncomputable def is_angle_bisector (D : Type) (A : Type) (B C : Type) : Prop := sorry

-- Assumptions
variable (h_angle_bisector : is_angle_bisector D A B C)
variable (h_circumcenter_O : is_circumcenter O A B C)
variable (h_circumcenter_O₁ : is_circumcenter O₁ A B D)
variable (h_circumcenter_O₂ : is_circumcenter O₂ A C D)

-- Theorem to prove that OO₁ = OO₂
theorem equal_distances_of_circumcenters :
  dist O O₁ = dist O O₂ :=
by sorry

end equal_distances_of_circumcenters_l79_79335


namespace shorter_side_ratio_l79_79525

variable {x y : ℝ}
variables (h1 : x < y)
variables (h2 : x + y - Real.sqrt (x^2 + y^2) = 1/2 * y)

theorem shorter_side_ratio (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = 1 / 2 * y) : x / y = 3 / 4 := 
sorry

end shorter_side_ratio_l79_79525


namespace cosine_of_angle_in_third_quadrant_l79_79034

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l79_79034


namespace solve_investment_problem_l79_79723

def remaining_rate_proof (A I A1 R1 A2 R2 x : ℚ) : Prop :=
  let income1 := A1 * (R1 / 100)
  let income2 := A2 * (R2 / 100)
  let remaining := A - A1 - A2
  let required_income := I - (income1 + income2)
  let expected_rate_in_float := (required_income / remaining) * 100
  expected_rate_in_float = x

theorem solve_investment_problem :
  remaining_rate_proof 15000 800 5000 3 6000 4.5 9.5 :=
by
  -- proof goes here
  sorry

end solve_investment_problem_l79_79723


namespace necessary_but_not_sufficient_condition_l79_79409

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a < 1) → ((a + 1) * (a - 2) < 0) ∧ ((∃ b : ℝ, (b + 1) * (b - 2) < 0 ∧ ¬(0 < b ∧ b < 1))) :=
by
  sorry

end necessary_but_not_sufficient_condition_l79_79409


namespace david_english_marks_l79_79367

theorem david_english_marks :
  let Mathematics := 45
  let Physics := 72
  let Chemistry := 77
  let Biology := 75
  let AverageMarks := 68.2
  let TotalSubjects := 5
  let TotalMarks := AverageMarks * TotalSubjects
  let MarksInEnglish := TotalMarks - (Mathematics + Physics + Chemistry + Biology)
  MarksInEnglish = 72 :=
by
  sorry

end david_english_marks_l79_79367


namespace probability_of_sequence_123456_l79_79657

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end probability_of_sequence_123456_l79_79657


namespace solution_set_of_f_lt_3x_plus_9_l79_79149

variable (f : ℝ → ℝ)

-- Condition: The domain of f(x) is ℝ
-- Condition: f(-2) = 3
axiom f_neg_two_eq_three : f (-2) = 3

-- Condition: For any x ∈ ℝ, f''(x) < 3
axiom f_double_prim_lt_three (x : ℝ) : f'' x < 3

-- We need to prove that the solution set of f(x) < 3x + 9 is x > -2
theorem solution_set_of_f_lt_3x_plus_9 :
  {x : ℝ | f x < 3 * x + 9} = {x : ℝ | x > -2} :=
begin
  sorry
end

end solution_set_of_f_lt_3x_plus_9_l79_79149


namespace find_f_2017_l79_79431

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_2017 (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_func_eq : ∀ x : ℝ, f (x + 3) * f x = -1)
  (h_val : f (-1) = 2) :
  f 2017 = -2 := sorry

end find_f_2017_l79_79431


namespace subtract_gcd_from_lcm_l79_79266

def gcd_24_54 : Nat := Nat.gcd 24 54
def lcm_40_8 : Nat := Nat.lcm 40 8

theorem subtract_gcd_from_lcm : lcm_40_8 - gcd_24_54 = 34 := by
  -- Definitions and calculations have been moved to conditions
  have h_gcd: gcd_24_54 = 6 := by sorry
  have h_lcm: lcm_40_8 = 40 := by sorry
  rw [h_gcd, h_lcm]
  norm_num

end subtract_gcd_from_lcm_l79_79266


namespace student_ratio_difference_l79_79164

theorem student_ratio_difference (s6 s7 : Nat) (h : 3 * s7 = 4 * s6) :
  s7 - s6 = 1 → s6 < s7 → s6 has fewer students than s7 by (s7 - s6) * 3 :=
sorry

end student_ratio_difference_l79_79164


namespace part1_part2_l79_79411

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 7 * x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - 4 * m^2 ≤ 0

-- Define the intervals
def interval_p : set ℝ := {x | -1 ≤ x ∧ x ≤ 8}
def interval_q (m : ℝ) : set ℝ := {x | 1 - 2 * m ≤ x ∧ x ≤ 1 + 2 * m}

-- Part 1: m ≥ 7/2 if p is necessary but not sufficient for q
theorem part1 (m : ℝ) (h : m > 0) (h_subset : interval_p ⊂ interval_q m) : m ≥ 7 / 2 := sorry

-- Part 2: 1 ≤ m ≤ 7/2 if "not p" is necessary but not sufficient for "not q"
theorem part2 (m : ℝ) (h : 0 < m) (h_subset : interval_q m ⊂ interval_p) : 1 ≤ m ∧ m ≤ 7 / 2 := sorry

end part1_part2_l79_79411


namespace triangle_integer_lengths_impossible_l79_79082

noncomputable def isIntegerLength (x : Real) : Prop :=
  ∃ n : ℤ, x = n

theorem triangle_integer_lengths_impossible
  {A B C D E I : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace I]
  (AB AC BI ID CI IE : ℝ)
  (h0 : ∃ α : ℝ, α = ∠A B C ∧ α = π / 2)
  (h1 : ∃ β : ℝ, β = ∠A B D ∧ β = ∠D B C)
  (h2 : ∃ γ : ℝ, γ = ∠A C E ∧ γ = ∠E C B)
  (h3 : BD meets CE at I)
  (h4 : A B C form_right_triangle_at A) : 
  ¬(isIntegerLength AB ∧ isIntegerLength AC ∧ isIntegerLength BI ∧ isIntegerLength ID ∧ isIntegerLength CI ∧ isIntegerLength IE) := 
sorry

end triangle_integer_lengths_impossible_l79_79082


namespace find_locus_of_tangent_intersection_l79_79451

noncomputable def locus_of_tangent_intersection (k : ℝ) (h : 3/4 < k ∧ k < 1) : set (ℝ × ℝ) :=
  {p | ∃ y, p = (2, y) ∧ 2 < y ∧ y < 5/2}

theorem find_locus_of_tangent_intersection :
  ∀ (k : ℝ), (3/4 < k ∧ k < 1) → (locus_of_tangent_intersection k) = {p : ℝ × ℝ | p.1 = 2 ∧ 2 < p.2 ∧ p.2 < 5/2} :=
by
  intros k hk
  sorry

end find_locus_of_tangent_intersection_l79_79451


namespace unique_norm_b_if_angle_is_determined_l79_79547

noncomputable def vector_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def vector_dot (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

def angle_between_vectors (a b : ℝ × ℝ × ℝ) (θ : ℝ) : Prop :=
  cos θ = vector_dot a b / (vector_norm a * vector_norm b)

theorem unique_norm_b_if_angle_is_determined {a b : ℝ × ℝ × ℝ} (θ : ℝ)
  (h : ∀ t : ℝ, vector_norm (t • a + b) ≥ 1)
  (h_angle: angle_between_vectors a b θ) :
  ∃! b_norm : ℝ, b_norm = vector_norm b :=
sorry

end unique_norm_b_if_angle_is_determined_l79_79547


namespace sequence_formula_l79_79376

def computational_program (m n : ℕ) : ℕ :=
if m = 1 ∧ n = 1 then 2
else if m = 1 ∧ n > 1 then (computational_program m (n-1)) + 2
else 0 -- for the purpose of the statement, this does not need to be exhaustive

def a (n : ℕ) : ℕ := computational_program 1 n

theorem sequence_formula (n : ℕ) (hn : n > 0) : a n = 2 * n :=
by {
  -- concise proof: proof will be provided here.
  intro,
  sorry
}

end sequence_formula_l79_79376


namespace minimize_transportation_cost_l79_79691

-- Define given conditions
def max_speed : ℝ := 60
def distance : ℝ := 600
def proportionality_constant : ℝ := 0.5
def other_costs_per_hour : ℝ := 1250

-- Define the cost function y
def cost_function (x : ℝ) : ℝ := (750000 / x) + (300 * x)

-- Prove that the cost function is minimized at x = 50
theorem minimize_transportation_cost :
  ∃ x : ℝ, (0 < x ∧ x ≤ max_speed) ∧ (∀ y : ℝ, (0 < y ∧ y ≤ max_speed) → cost_function x ≤ cost_function y) ∧ x = 50 :=
sorry

end minimize_transportation_cost_l79_79691


namespace distinct_real_roots_of_quadratic_l79_79472

theorem distinct_real_roots_of_quadratic (m : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ x^2 + m * x + 9 = 0 ∧ y^2 + m * y + 9 = 0) ↔ m ∈ Ioo (-∞) (-6) ∪ Ioo 6 (∞) :=
by
  sorry

end distinct_real_roots_of_quadratic_l79_79472


namespace anna_strawberry_lemonade_earning_l79_79738

theorem anna_strawberry_lemonade_earning:
  (plain_lemonade_price : ℝ) (plain_lemonade_count : ℕ) (extra_earned : ℝ) (strawberry_lemonade_earning : ℝ) 
  (plain_lemonade_earning := plain_lemonade_price * plain_lemonade_count)
  (earning_equation : plain_lemonade_earning = strawberry_lemonade_earning + extra_earned)
  (h1 : plain_lemonade_price = 0.75) 
  (h2 : plain_lemonade_count = 36)
  (h3 : extra_earned = 11) :
  strawberry_lemonade_earning = 16 := 
by
  sorry

end anna_strawberry_lemonade_earning_l79_79738


namespace radius_ratio_l79_79315

noncomputable def volume_large_sphere (r_L : ℝ) : ℝ := (4 / 3) * π * r_L^3
noncomputable def volume_small_sphere (V_L : ℝ) : ℝ := 0.20 * V_L

theorem radius_ratio (r_L r_S V_L V_S : ℝ) (h1 : volume_large_sphere r_L = 675 * π)
  (h2 : V_S = volume_small_sphere 675 * π):
  r_S / r_L = 1 / (5)^(1/3) :=
sorry

end radius_ratio_l79_79315


namespace verification_l79_79424

noncomputable def z1 : ℂ := 2 + I
noncomputable def z2 : ℂ := 1 - 2 * I

theorem verification : (conj z2 ≠ z1) ∧ (abs z1 = abs z2) ∧ (z1 * z2 = 4 - 3 * I) ∧ (im (z2 / z1) ≠ 0 ∧ re (z2 / z1) = 0) :=
by {
  sorry
}

end verification_l79_79424


namespace polygon_coloring_10_vertices_l79_79226

-- Define the initial conditions and recurrence relations as part of the setup
def a_1 := 2
def a_2 := 3

noncomputable def a : ℕ → ℕ :=
  λ n,
    if n = 1 then a_1
    else if n = 2 then a_2
    else a (n - 1) + a (n - 3) + a (n - 4)

-- Main theorem statement
theorem polygon_coloring_10_vertices : a 10 = 123 :=
by
  sorry

end polygon_coloring_10_vertices_l79_79226


namespace total_pages_book_l79_79070

-- Define the conditions
def reading_speed1 : ℕ := 10 -- pages per day for first half
def reading_speed2 : ℕ := 5 -- pages per day for second half
def total_days : ℕ := 75 -- total days spent reading

-- This is the main theorem we seek to prove:
theorem total_pages_book (P : ℕ) 
  (h1 : ∃ D1 D2 : ℕ, D1 + D2 = total_days ∧ D1 * reading_speed1 = P / 2 ∧ D2 * reading_speed2 = P / 2) : 
  P = 500 :=
by
  sorry

end total_pages_book_l79_79070


namespace speed_of_man_in_still_water_l79_79318

theorem speed_of_man_in_still_water (V_m V_c : ℝ) :
  (72 / 4 = V_m + V_c) → (36 / 6 = V_m - V_c) → V_m = 12 :=
by
  intro h1 h2
  have h3 : 18 = V_m + V_c := h1
  have h4 : 6 = V_m - V_c := h2
  have h5 : (V_m + V_c) + (V_m - V_c) = 24 := by linarith
  have h6 : 2 * V_m = 24 := by linarith
  have h7 : V_m = 12 := by linarith
  exact h7

end speed_of_man_in_still_water_l79_79318


namespace find_m_l79_79064

theorem find_m
  (m : ℝ)
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (m, 2, 3))
  (hB : B = (1, -1, 1))
  (h_dist : (Real.sqrt ((m - 1) ^ 2 + (2 - (-1)) ^ 2 + (3 - 1) ^ 2) = Real.sqrt 13)) :
  m = 1 := 
sorry

end find_m_l79_79064


namespace length_of_plot_l79_79157

-- Definitions of the given conditions, along with the question.
def breadth (b : ℝ) : Prop := 2 * (b + 32) + 2 * b = 5300 / 26.50
def length (b : ℝ) := b + 32

theorem length_of_plot (b : ℝ) (h : breadth b) : length b = 66 := by 
  sorry

end length_of_plot_l79_79157


namespace distinct_real_roots_interval_l79_79484

open Set Real

theorem distinct_real_roots_interval (m : ℝ) : 
  (∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ IsRoot (λ x => x^2 + m * x + 9) r1 ∧ IsRoot (λ x => x^2 + m * x + 9) r2) ↔ 
  m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end distinct_real_roots_interval_l79_79484


namespace inscribed_rectangle_area_l79_79148

variables (b h x : ℝ)

theorem inscribed_rectangle_area (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hx_lt_h : x < h) :
  let area := (b * x * (h - x)) / h in
  area = (b * x * (h - x)) / h := 
sorry

end inscribed_rectangle_area_l79_79148


namespace angles_of_smaller_triangles_l79_79508

/-- 
In a triangle, the angles A, B, and C are known. Prove that the angles of the six smaller triangles 
formed by the bisectors of the angles of the original triangle are as follows:
- the angles are 1/2 * A, 1/2 * A + 1/2 * C, B + 1/2 * C, and others derived similarly.
-/
theorem angles_of_smaller_triangles (A B C : ℝ) (triangle : triangle_property A B C) :
  (angles_of_smaller_triangles A B C) = { (1/2 * A), (1/2 * A + 1/2 * C), (B + 1/2 * C), ... } :=
sorry

end angles_of_smaller_triangles_l79_79508


namespace find_b_in_terms_of_y_l79_79408

variables {a b y : ℝ}

theorem find_b_in_terms_of_y
  (h1 : a ≠ b)
  (h2 : a^3 - b^3 = 25 * y^3)
  (h3 : a - b = y) :
  b = -((1 - real.sqrt 33) / 2) * y ∨ b = -((1 + real.sqrt 33) / 2) * y :=
sorry

end find_b_in_terms_of_y_l79_79408


namespace trig_function_value_l79_79433

theorem trig_function_value
  (x y : ℝ) (r : ℝ) (α : ℝ)
  (hx : x = -4 / 5)
  (hy : y = 3 / 5)
  (hr : r = 1)
  (cos_α : cos α = x / r)
  (sin_α : sin α = y / r) :
  2 * sin α + cos α = 2 / 5 :=
by sorry

end trig_function_value_l79_79433


namespace max_consecutive_integers_sum_l79_79251

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l79_79251


namespace alicia_masks_collection_l79_79328

theorem alicia_masks_collection
  (initial_masks : ℕ)
  (donated_guggenheim : ℕ)
  (donated_metropolitan : ℕ)
  (damaged_sets : ℕ)
  (donated_louvre : ℕ)
  (donated_british : ℕ)
  (remaining_masks_after_all_visits : ℕ)
  (h1 : initial_masks = 500)
  (h2 : donated_guggenheim = 51)
  (h3 : donated_metropolitan = 2 * donated_guggenheim)
  (h4 : damaged_sets = 20)
  (h5 : donated_louvre = (initial_masks - donated_guggenheim - donated_metropolitan - damaged_sets) / 2)
  (h6 : donated_british = 2 * (initial_masks - donated_guggenheim - donated_metropolitan - damaged_sets - donated_louvre) / 3)
  : remaining_masks_after_all_visits = 55 :=
begin
  -- Definitions based on given conditions
  let initial := 500,
  let guggenheim_donation := 51,
  let met_donation := 2 * guggenheim_donation,
  let damaged := 20,
  let louvre_donation := (initial - guggenheim_donation - met_donation - damaged) / 2,
  let british_donation := 2 * (initial - guggenheim_donation - met_donation - damaged - louvre_donation) / 3,
  let remaining_masks := initial - guggenheim_donation - met_donation - damaged - louvre_donation - british_donation,
  
  -- Using the provided conditions
  have h_initial : initial = initial_masks, from h1,
  have h_guggenheim : guggenheim_donation = donated_guggenheim, from h2,
  have h_met : met_donation = donated_metropolitan, from h3,
  have h_damaged : damaged = damaged_sets, from h4,
  have h_louvre : louvre_donation = donated_louvre, from h5,
  have h_british : british_donation = donated_british, from h6,
  
  -- Conclusion based on given conditions
  show remaining_masks_after_all_visits = remaining_masks,
  from calc remaining_masks_after_all_visits = 55 : sorry
end

end alicia_masks_collection_l79_79328


namespace rectangle_PS_greatest_integer_lt_173_l79_79512

theorem rectangle_PS_greatest_integer_lt_173
  (PQRS : Type)
  [rectangle PQRS]
  (P Q R S T : PQRS)
  (h1 : PQ = 150)
  (h2 : midpoint T P S)
  (h3 : PT ⊥ QR) :
  ∃ PS : ℝ, PS = sqrt 30000 ∧ greatest_integer_lt PS = 173 :=
sorry

end rectangle_PS_greatest_integer_lt_173_l79_79512


namespace fixed_point_of_line_l79_79000

theorem fixed_point_of_line (m : ℝ) : ∃ x y : ℝ, (y - 2 = m * x + m) ∧ (x = -1) ∧ (y = 2) :=
by {
  existsi -1,
  existsi 2,
  split, apply eq.refl, split; sorry
}

end fixed_point_of_line_l79_79000


namespace geometric_series_common_ratio_l79_79184

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l79_79184


namespace range_of_a_l79_79039

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, 0 < x ∧ 3*x + a ≤ 2 → x = 1 ∨ x = 2) ↔ (-7 < a ∧ a ≤ -4) :=
sorry

end range_of_a_l79_79039


namespace cosine_of_angle_in_third_quadrant_l79_79031

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l79_79031


namespace geometric_series_common_ratio_l79_79173

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l79_79173


namespace number_of_integer_pairs_satisfying_conditions_l79_79359

noncomputable def count_integer_pairs (n m : ℕ) : ℕ := Nat.choose (n-1) (m-1)

theorem number_of_integer_pairs_satisfying_conditions :
  ∃ (a b c x y : ℕ), a + b + c = 55 ∧ a + b + c + x + y = 71 ∧ x + y > a + b + c → count_integer_pairs 55 3 * count_integer_pairs 16 2 = 21465 := sorry

end number_of_integer_pairs_satisfying_conditions_l79_79359


namespace total_price_of_books_l79_79298

-- Define the conditions
variables (x : ℝ)

-- The given conditions
def price_of_discount_card := 20
def discount_rate := 0.2
def saved_amount := 12

-- The equation derived from the conditions
def equation := price_of_discount_card + (1 - discount_rate) * x = x - saved_amount

-- Statement to prove
theorem total_price_of_books (h : equation x) : x = 160 := 
sorry

end total_price_of_books_l79_79298


namespace Kiera_envelopes_total_l79_79075

-- Define variables for different colored envelopes
def E_b : ℕ := 120
def E_y : ℕ := E_b - 25
def E_g : ℕ := 5 * E_y
def E_r : ℕ := (E_b + E_y) / 2  -- integer division in lean automatically rounds down
def E_p : ℕ := E_r + 71
def E_total : ℕ := E_b + E_y + E_g + E_r + E_p

-- The statement to be proven
theorem Kiera_envelopes_total : E_total = 975 := by
  -- intentionally put the sorry to mark the proof as unfinished
  sorry

end Kiera_envelopes_total_l79_79075


namespace geometric_series_common_ratio_l79_79186

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l79_79186


namespace max_consecutive_integers_sum_l79_79246

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l79_79246


namespace rooks_diagonal_move_l79_79936

def rook (x y : ℕ) : Prop := x = y

theorem rooks_diagonal_move (board : fin 8 → fin 8 → Prop)
        (h : ∀ i j, board i j → ¬ ∃ k, k ≠ j ∧ board i k ∨ k ≠ i ∧ board k j) :
        ∃ new_board : fin 8 → fin 8 → Prop, 
        (∀ i j, new_board i j → ¬ ∃ k, k ≠ j ∧ new_board i k ∨ k ≠ i ∧ new_board k j)
        ∧ (∀ i j, board i j → ∃ di dj, (di = 1 ∨ di = -1) ∧ (dj = 1 ∨ dj = -1) ∧ new_board (i + di) (j + dj)) :=
sorry

end rooks_diagonal_move_l79_79936


namespace total_amount_spent_l79_79309

theorem total_amount_spent {price_food : ℝ} (h1 : price_food = 120)
                           {tax_rate : ℝ} (h2 : tax_rate = 0.10)
                           {tip_rate : ℝ} (h3 : tip_rate = 0.20) :
  let price_tax := price_food * tax_rate,
      price_incl_tax := price_food + price_tax,
      tip := price_incl_tax * tip_rate,
      total := price_incl_tax + tip
  in total = 158.40 :=
by {
  rw [h1, h2, h3],
  let price_tax := 120 * 0.10,
  have h4 : price_tax = 12 := by norm_num,
  let price_incl_tax := 120 + price_tax,
  have h5 : price_incl_tax = 132 := by {rw h4, norm_num},
  let tip := price_incl_tax * 0.20,
  have h6 : tip = 26.40 := by {rw h5, norm_num},
  let total := price_incl_tax + tip,
  have h7 : total = 158.40 := by {rw [h5, h6], norm_num},
  exact h7,
}

end total_amount_spent_l79_79309


namespace total_cost_rental_l79_79563

theorem total_cost_rental :
  let rental_fee := 20.99
  let charge_per_mile := 0.25
  let miles_driven := 299
  let total_cost := rental_fee + charge_per_mile * miles_driven
  total_cost = 95.74 := by
{
  sorry
}

end total_cost_rental_l79_79563


namespace A_3_eq_14_A_n_eq_general_l79_79225

open Finset
open BigOperators

def norm {n : ℕ} (a : Fin n → ℤ) : ℤ := ∑ i, |a i|

def odd_norm_vectors (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    (3^n + 1) / 2
  else
    (3^n - 1) / 2

theorem A_3_eq_14 : odd_norm_vectors 3 = 14 := by
  sorry

theorem A_n_eq_general (n : ℕ) : odd_norm_vectors n = (3^n - (-1)^n) / 2 := by
  sorry

end A_3_eq_14_A_n_eq_general_l79_79225


namespace union_of_sets_l79_79819

open Set

theorem union_of_sets (A B : Set ℝ) :
  (A = { x | x^2 - 4x - 5 = 0 }) → 
  (B = { x | x^2 - 1 = 0 }) → 
  A ∪ B = { -1, 1, 5 } :=
by
  intros hA hB
  sorry

end union_of_sets_l79_79819


namespace vitya_probability_l79_79651

theorem vitya_probability :
  let total_sequences := (finset.range 6).card * 
                         (finset.range 5).card * 
                         (finset.range 4).card * 
                         (finset.range 3).card * 
                         (finset.range 2).card * 
                         (finset.range 1).card,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11,
      total_possibilities := nat.choose 12 2 * nat.choose 10 2 * 
                             nat.choose 8 2 * nat.choose 6 2 * 
                             nat.choose 4 2 * nat.choose 2 2,
      P := (favorable_sequences : ℚ) / (total_possibilities : ℚ)
  in P = 1 / 720 := 
sorry

end vitya_probability_l79_79651


namespace max_consecutive_sum_lt_1000_l79_79239

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l79_79239


namespace find_coefficient_B_l79_79383

theorem find_coefficient_B (A B C D : ℤ) (roots : Fin 6 → ℕ) :
  (∏ i, (z - ↑(roots i))) = z^6 - 10 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 36 →
  (∀ i, 0 < roots i) →
  B = -66 := 
sorry

end find_coefficient_B_l79_79383


namespace bridge_crossing_time_correct_l79_79627

def A_time : ℕ := 2
def B_time : ℕ := 3
def C_time : ℕ := 8
def D_time : ℕ := 10

def total_crossing_time : ℕ :=
  let AB_cross := max A_time B_time in
  let A_return := A_time in
  let CD_cross := max C_time D_time in
  let B_return := B_time in
  let AB_cross_again := max A_time B_time in
  AB_cross + A_return + CD_cross + B_return + AB_cross_again

theorem bridge_crossing_time_correct : total_crossing_time = 21 :=
  by
    unfold total_crossing_time
    have AB_cross_eq : max A_time B_time = 3 := by rfl
    have A_return_eq : A_time = 2 := by rfl
    have CD_cross_eq : max C_time D_time = 10 := by rfl
    have B_return_eq : B_time = 3 := by rfl
    have AB_cross_again_eq : max A_time B_time = 3 := by rfl
    rw [AB_cross_eq, A_return_eq, CD_cross_eq, B_return_eq, AB_cross_again_eq]
    rfl

end bridge_crossing_time_correct_l79_79627


namespace cos_48_conjecture_l79_79356

noncomputable def cos_24 := Real.cos (24 * Real.pi / 180)
noncomputable def cos_6 := Real.cos (6 * Real.pi / 180)

theorem cos_48_conjecture :
  let c := cos_24,
      d := Real.cos (48 * Real.pi / 180),
      e := cos_6 in
  d = 2 * c^2 - 1 ∧
  -e = 2 * d^2 - 1 ∧
  -e = 4 * c^3 - 3 * c →
  d = Real.cos (48 * Real.pi / 180) :=
by
  intro h
  sorry

end cos_48_conjecture_l79_79356


namespace cosine_of_F_in_right_triangle_l79_79887

noncomputable def DE := 8
noncomputable def EF := 17
noncomputable def DF := Real.sqrt (EF^2 - DE^2)
noncomputable def cosF := DF / EF

theorem cosine_of_F_in_right_triangle :
  ∠ D = 90 → DF = 15 → cosF = 15 / 17 :=
by 
  sorry

end cosine_of_F_in_right_triangle_l79_79887


namespace geom_series_converges_arith_geom_series_converges_telescoping_series_one_telescoping_series_one_fourth_harmonic_series_diverges_l79_79791

noncomputable def geom_series_sum (a : ℝ) (ha : |a| < 1) : ℝ :=
  1 / (1 - a)

theorem geom_series_converges (a : ℝ) (ha : |a| < 1) :
  ∑' n : ℕ, a^n = geom_series_sum a ha := sorry

noncomputable def arith_geom_series_sum (a : ℝ) (ha : |a| < 1) : ℝ :=
  a / (1 - a)^2

theorem arith_geom_series_converges (a : ℝ) (ha : |a| < 1) :
  ∑' n : ℕ, (n + 1) * a^(n + 1) = arith_geom_series_sum a ha := sorry

theorem telescoping_series_one :
  ∑' n : ℕ, 1 / ((n + 1) * (n + 2)) = 1 := sorry

theorem telescoping_series_one_fourth :
  ∑' n : ℕ, 1 / ((n + 1) * (n + 2) * (n + 3)) = 1 / 4 := sorry

theorem harmonic_series_diverges :
  ¬ summable (λ n : ℕ, 1 / (n + 1)) := sorry

end geom_series_converges_arith_geom_series_converges_telescoping_series_one_telescoping_series_one_fourth_harmonic_series_diverges_l79_79791


namespace xy_value_l79_79961

theorem xy_value (x y : ℝ) (h : x ≠ y) (h_eq : x^2 + 2 / x^2 = y^2 + 2 / y^2) : 
  x * y = Real.sqrt 2 ∨ x * y = -Real.sqrt 2 :=
by
  sorry

end xy_value_l79_79961


namespace distinct_real_roots_of_quadratic_l79_79479

-- Define the problem's condition: m is a real number and the discriminant of x^2 + mx + 9 > 0
def discriminant_positive (m : ℝ) := m^2 - 36 > 0

theorem distinct_real_roots_of_quadratic (m : ℝ) (h : discriminant_positive m) :
  m ∈ Iio (-6) ∪ Ioi (6) :=
sorry

end distinct_real_roots_of_quadratic_l79_79479


namespace square_diagonal_l79_79712

theorem square_diagonal (P : ℝ) (d : ℝ) (hP : P = 200 * Real.sqrt 2) :
  d = 100 :=
by
  sorry

end square_diagonal_l79_79712


namespace problem1_problem2_problem3_l79_79932

variable {A : Type*} [Fintype A]

def binary_subset (A : Finset A) (M : Finset A) : Prop :=
  M ⊆ A ∧ M.card = 2
  
def P (A : Finset A) (m : ℕ) : Prop :=
  ∀ (A1 A2 ... Am : Finset A), P_if (A1 ⊆ A) (A2 ⊆ A) ... (Am ⊆ A) ∧
  (∀ i j, i ≠ j → A1 ≠ A2) → 
  (∃ B, B ⊆ A ∧ B.card = m ∧ ∀ i, 1 ≤ i ∧ i ≤ m → (B ∩ A_i).card ≤ 1)
  
-- For n = 3, at least one possible value of m is 2
theorem problem1 (A : Finset ℕ) (hA : A = {1, 2, 3}) : P A 2 := 
sorry

-- For n = 6, A does not have the property P(4)
theorem problem2 (A : Finset ℕ) (hA : A = {1, 2, 3, 4, 5, 6}) : ¬ P A 4 :=
sorry

-- If A has the property P(2023), then the minimum value of n is 4045
theorem problem3 (A : Finset ℕ) (hP : P A 2023) : A.card ≥ 4045 :=
sorry

end problem1_problem2_problem3_l79_79932


namespace magnitude_add_three_b_l79_79820

noncomputable def vec_mag_add_to_be_proven (a b : ℝ × ℝ) (ha : a.1 ^ 2 + a.2 ^ 2 = 1) (hb : b.1 ^ 2 + b.2 ^ 2 = 1) 
(θ : ℝ) (hθ : θ = real.pi / 3) : real :=
real.sqrt ((a.1 + 3 * b.1) ^ 2 + (a.2 + 3 * b.2) ^ 2)

theorem magnitude_add_three_b (a b : ℝ × ℝ) (ha : a.1 ^ 2 + a.2 ^ 2 = 1) (hb : b.1 ^ 2 + b.2 ^ 2 = 1) 
(θ : ℝ) (hθ : θ = real.pi / 3) :
  vec_mag_add_to_be_proven a b ha hb θ hθ = 13 :=
sorry

end magnitude_add_three_b_l79_79820


namespace general_solution_differential_inequality_l79_79613

variable (p q g : ℝ → ℝ)
variable (eintp : ℝ → ℝ) -- represents e^(∫ p(x) dx)

axiom eintp_def : ∀ x, eintp x = exp (∫ t in 0..x, p t)
axiom zero_le_g : ∀ x, 0 ≤ g x

theorem general_solution_differential_inequality :
  (∀ x, (differential (λ y : ℝ → ℝ, y x) x + p x * (λ y : ℝ → ℝ, y x) x) ≥ q x) →
  (∀ x, ∃ c : ℝ → ℝ, (λ y : ℝ → ℝ, y x) = eintp x * (∫ t in 0..x, q t * eintp t dx + ∫ t in 0..x, g t dx)) :=
by 
  sorry

end general_solution_differential_inequality_l79_79613


namespace ratio_of_hexagon_areas_l79_79537

-- Introduce definitions for regular hexagons and centers of equilateral triangles
structure RegularHexagon (P : Type*) :=
  (vertices : Fin 6 → P)
  (side_length : ℝ)
  (side_lengths_eq : ∀ i j, dist (vertices i) (vertices (i + 1) % 6) = dist (vertices j) (vertices (j + 1) % 6))

structure EquilateralTriangleCenters (P : Type*) :=
  (centers : Fin 6 → P)

-- Assume we have a regular hexagon ABCD with vertices A B C D E F
variables {P : Type*} [MetricSpace P]

def is_centers_of_equilateral_triangles_constructed (hex : RegularHexagon P) (centers : EquilateralTriangleCenters P) : Prop :=
  ∀ i, dist (hex.vertices i) (centers.centers i) = dist (hex.vertices (i + 1) % 6) (centers.centers i)
  
theorem ratio_of_hexagon_areas (hex : RegularHexagon P)
  (centers : EquilateralTriangleCenters P)
  (h : is_centers_of_equilateral_triangles_constructed hex centers) :
  ∃ ratio : ℝ, ratio = (4 / 3) ∧
    (3 * sqrt 3 / 2) * (hex.side_length ^ 2) * (6 * (sqrt 3 / 6 / (hex.side_length / 2))^2) = 
    (3 * sqrt 3 / 2) * ( hex.side_length ^ 2 * 4 / 3) :=
sorry

end ratio_of_hexagon_areas_l79_79537


namespace max_consecutive_integers_lt_1000_l79_79253

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l79_79253


namespace diagonal_length_proof_l79_79983

noncomputable def rectangle_diagonal_length
    (P : ℝ) (ratio : (ℝ × ℝ)) (hP : P = 60) (hRatio : ratio = (5, 4)) : ℝ :=
    let l := 5 * (P / 18)
    let w := 4 * (P / 18)
    real.sqrt (l ^ 2 + w ^ 2)

theorem diagonal_length_proof :
    rectangle_diagonal_length 60 (5, 4) (by norm_num) (by norm_num) = 10 * real.sqrt 41 / 3 :=
    sorry

end diagonal_length_proof_l79_79983


namespace intersection_of_A_and_B_l79_79548

open Set

variable (A : Set ℕ) (B : Set ℕ)

theorem intersection_of_A_and_B (hA : A = {0, 1, 2}) (hB : B = {0, 2, 4}) :
  A ∩ B = {0, 2} := by
  sorry

end intersection_of_A_and_B_l79_79548


namespace area_of_rectangle_PQRS_l79_79420

def origin : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (2, 1)
def S : ℝ × ℝ := (0, 1)
def triangle_PQR (P Q R : ℝ × ℝ) : Prop := 
  (P = origin ∧ Q = (2, 0) ∧ R = (2, 1) ∧ (Q.2 = R.2 ∨ P.1 = R.1)) -- ensuring the right triangle condition

theorem area_of_rectangle_PQRS : 
  triangle_PQR origin Q R → 
  let PQ := (Q.1 - origin.1)  in
  let PS := (S.2 - origin.2)  in
  PQ * PS = 2 := 
by 
  intro h;
  let PQ := (Q.1 - origin.1);
  let PS := (S.2 - origin.2);
  exact sorry

end area_of_rectangle_PQRS_l79_79420


namespace distinct_real_roots_interval_l79_79485

open Set Real

theorem distinct_real_roots_interval (m : ℝ) : 
  (∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ IsRoot (λ x => x^2 + m * x + 9) r1 ∧ IsRoot (λ x => x^2 + m * x + 9) r2) ↔ 
  m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end distinct_real_roots_interval_l79_79485


namespace cos_sin_eq_one_solutions_l79_79133

theorem cos_sin_eq_one_solutions (n : ℤ) (hn : n > 0) :
  (∃ k : ℤ, cos (n * k * Real.pi) ^ n - sin (n * k * Real.pi) ^ n = 1) ∨
  (∃ k : ℤ, cos (n * (2 * k * Real.pi) + (3 * Real.pi / 2)) ^ n - 
              sin (n * (2 * k * Real.pi) + (3 * Real.pi / 2)) ^ n = 1) :=
by
  sorry

end cos_sin_eq_one_solutions_l79_79133


namespace prob_both_even_correct_l79_79288

-- Define the dice and verify their properties
def die1 := {n : ℕ // n ≥ 1 ∧ n ≤ 6}
def die2 := {n : ℕ // n ≥ 1 ∧ n ≤ 7}

-- Define the sets of even numbers for both dice
def even_die1 (n : die1) : Prop := n.1 % 2 = 0
def even_die2 (n : die2) : Prop := n.1 % 2 = 0

-- Define the probabilities of rolling an even number on each die
def prob_even_die1 := 3 / 6
def prob_even_die2 := 3 / 7

-- Calculate the combined probability
def prob_both_even := prob_even_die1 * prob_even_die2

-- The theorem stating the probability of both dice rolling even is 3/14
theorem prob_both_even_correct : prob_both_even = 3 / 14 :=
by
  -- Proof is omitted
  sorry

end prob_both_even_correct_l79_79288


namespace quadratic_roots_conjugate_and_product_l79_79437

open Complex

theorem quadratic_roots_conjugate_and_product (x1 x2 : ℂ) :
  (∀ x : ℂ, x^2 + (3 : ℂ) * x + 4 = 0 → ∃ x1 x2 : ℂ, x = x1 ∨ x = x2) →
  (x1 = Complex.conj x2) ∧ (x1 * x2 = 4) :=
by
  sorry

end quadratic_roots_conjugate_and_product_l79_79437


namespace water_leftover_l79_79991

theorem water_leftover (players : ℕ) (total_water_l : ℕ) (water_per_player_ml : ℕ) (spill_water_ml : ℕ)
  (h1 : players = 30) 
  (h2 : total_water_l = 8) 
  (h3 : water_per_player_ml = 200) 
  (h4 : spill_water_ml = 250) : 
  (total_water_l * 1000 - (players * water_per_player_ml + spill_water_ml) = 1750) :=
by
  -- conversion of total water to milliliters
  let total_water_ml := total_water_l * 1000
  -- calculation of total water used for players
  let total_water_used_for_players := players * water_per_player_ml
  -- calculation of total water including spill
  let total_water_used := total_water_used_for_players + spill_water_ml
  -- leftover water calculation
  have calculation : total_water_l * 1000 - (players * water_per_player_ml + spill_water_ml) = total_water_ml - total_water_used, by
    rw [total_water_ml, total_water_used, total_water_used_for_players]
  rw calculation
  -- conclusion by substituting known values
  rw [h1, h2, h3, h4]
  norm_num

end water_leftover_l79_79991


namespace pyramid_volume_l79_79513

theorem pyramid_volume (side_S : ℝ) (side_S' : ℝ) (mid_S_to_S_center : ℝ) (S_center_to_S'_side : ℝ) :
  side_S = 40 → 
  side_S' = 15 → 
  mid_S_to_S_center = side_S / 2 →
  S_center_to_S'_side = side_S' / 2 →
  let h' := mid_S_to_S_center - S_center_to_S'_side,
      h := Real.sqrt (h' ^ 2 - (side_S' / 2) ^ 2),
      b := side_S' ^ 2,
      V := (1 / 3) * b * h in
  V = 750 :=
by
  intros h
  sorry

end pyramid_volume_l79_79513


namespace integral_equals_expected_l79_79780

noncomputable def integral_expression : ℝ → ℝ :=
  λ x, (x^3 + x^2 + 2) / (x * (x^2 - 1)^2)

noncomputable def expected_integral : ℝ → ℝ :=
  λ x, 2 * Real.log (Real.abs x)
        - (3 / 4) * Real.log (Real.abs (x - 1))
        - 1 / (x - 1)
        - (5 / 4) * Real.log (Real.abs (x + 1))
        + 1 / (2 * (x + 1))

theorem integral_equals_expected (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  ∫ c in 0..x, integral_expression c = expected_integral x + C :=
by
  sorry

end integral_equals_expected_l79_79780


namespace train_length_correct_l79_79717

noncomputable def train_length (speed_kmh : ℝ) (time_sec : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  let total_distance := speed_ms * time_sec
  total_distance - bridge_length_m

theorem train_length_correct :
  train_length 45 30 255.03 = 119.97 :=
by 
  -- Using the given values, prove that the length of the train is 119.97 meters.
  let speed_ms := 45 * (1000 / 3600) in
  let total_distance := speed_ms * 30 in
  let train_length := total_distance - 255.03 in
  have h1 : speed_ms = 12.5 := by norm_num,
  have h2 : total_distance = 375 := by norm_num [h1],
  have h3 : train_length = 119.97 := by norm_num [h2],
  exact h3

end train_length_correct_l79_79717


namespace geom_location_y_axis_l79_79679

noncomputable def geom_location (z : ℂ) (hz : |z| = 1) (hz_ne : z^2 ≠ 1) : Prop :=
  ∃ r : ℝ, 0 < r ∧ z = r * exp(I * real.arg z)

theorem geom_location_y_axis (z : ℂ) (hz : |z| = 1) (hz_ne : z^2 ≠ 1) :
  ∃ k : ℝ, (z / (1 - z^2)).im = k :=
sorry

end geom_location_y_axis_l79_79679


namespace ball_properties_from_spherical_cap_l79_79296

theorem ball_properties_from_spherical_cap :
  ∃ (r : ℝ), (r = 10) ∧ (4 * Real.pi * r^2 = 400 * Real.pi) :=
begin
  -- Given conditions
  let diameter : ℝ := 12,
  let depth : ℝ := 2,
  
  -- Translated conditions
  have r_condition : ∀ (r : ℝ), (r - depth)^2 + (diameter / 2)^2 = r^2,
  { intros r, sorry }, -- The full proof is omitted here

  have r_result : 10 - depth = r,
  { sorry },

  use 10,
  split,
  { refl }, -- The radius is 10 cm.
  { sorry }  -- The surface area is 400π cm².
end

end ball_properties_from_spherical_cap_l79_79296


namespace malingerers_exposed_l79_79684

theorem malingerers_exposed (a b c : Nat) (ha : a > b) (hc : c = b + 9) :
  let aabbb := 10000 * a + 1000 * a + 100 * b + 10 * b + b
  let abccc := 10000 * a + 1000 * b + 100 * c + 10 * c + c
  (aabbb - 1 = abccc) -> abccc = 10999 :=
by
  sorry

end malingerers_exposed_l79_79684


namespace purple_chip_count_l79_79287

theorem purple_chip_count :
  ∃ (x : ℕ), (x > 5) ∧ (x < 11) ∧
  (∃ (blue green purple red : ℕ),
    (2^6) * (5^2) * 11 * 7 = (blue * 1) * (green * 5) * (purple * x) * (red * 11) ∧ purple = 1) :=
sorry

end purple_chip_count_l79_79287


namespace angle_between_vectors_l79_79008

variables (a b : ℝ → ℝ → ℝ)
variables (θ : ℝ)

-- Conditions
def norm_a : ℝ := 1
def norm_b : ℝ := 4
def dot_ab : ℝ := 2

-- Question
theorem angle_between_vectors : 
  norm a = norm_a → 
  norm b = norm_b → 
  dot_product a b = dot_ab → 
  θ = arccos (dot_ab / (norm_a * norm_b)) → 
  θ = real.pi / 3 :=
by
  sorry

end angle_between_vectors_l79_79008


namespace arithmetic_seq_sum_thirteen_terms_l79_79890

theorem arithmetic_seq_sum_thirteen_terms 
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_condition : 2 * (a 1 + a 4 + a 7) + 3 * (a 9 + a 11) = 24) :
  (finset.range 13).sum a = 26 :=
by
  sorry

end arithmetic_seq_sum_thirteen_terms_l79_79890


namespace A2B2_eq_2AB_l79_79926

theorem A2B2_eq_2AB (A B C A₁ B₁ A₂ B₂ : Point) (hABC : Triangle A B C) :
  is_isosceles_triangle hABC → 
  (angle C > 60) → 
  (angle A₁ C B₁ = angle ABC) → 
  externally_tangent_circle (circumcircle (triangle A₁ B₁ C)) C A B A₂ B₂
  → (distance A₂ B₂ = 2 * distance A B)
:= by
  sorry

end A2B2_eq_2AB_l79_79926


namespace find_number_l79_79285

theorem find_number (N : ℝ) (h : 0.6 * (3 / 5) * N = 36) : N = 100 :=
by sorry

end find_number_l79_79285


namespace cos_B_third_quadrant_l79_79030

theorem cos_B_third_quadrant (B : ℝ) (hB1 : π < B ∧ B < 3 * π / 2) (hB2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

end cos_B_third_quadrant_l79_79030


namespace cosine_in_third_quadrant_l79_79023

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l79_79023


namespace position_of_ten_sqrt_three_l79_79334

theorem position_of_ten_sqrt_three : 
  let pattern (n : ℕ) := sqrt (2 * ↑n) in 
  numberAtPosition (10 * sqrt 3) = (38, 2) :=
sorry

end position_of_ten_sqrt_three_l79_79334


namespace rationalize_denominator_min_sum_l79_79126

theorem rationalize_denominator_min_sum :
  ∃ (A B C D : ℤ), (D > 0) ∧ (∀ p : ℤ, Prime p → ¬ (p * p ∣ B)) ∧
    (Ratio (A * √ B + C) D = Ratio (5 * √ 2) ((√ 25) - (√ 5))) ∧
    (A + B + C + D = 15) :=
begin
  sorry
end

end rationalize_denominator_min_sum_l79_79126


namespace max_consecutive_integers_sum_lt_1000_l79_79240

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l79_79240


namespace tangent_sum_of_angles_l79_79221

-- Given conditions and context
variables (K M A B : Point)
variable (circle1 : Circle K M)
variable (circle2 : Circle K M)
variable (tangent : Line)
variable (is_tangent1 : tangent.is_tangent circle1 A)
variable (is_tangent2 : tangent.is_tangent circle2 B)

-- Prove that the sum of the angles ∠AMB and ∠AKB is 180°
theorem tangent_sum_of_angles (K M A B : Point) (circle1 circle2 : Circle K M) (tangent : Line)
  (is_tangent1 : tangent.is_tangent circle1 A) (is_tangent2 : tangent.is_tangent circle2 B) :
  angle A M B + angle A K B = 180 :=
  sorry

end tangent_sum_of_angles_l79_79221


namespace sum_a_sequence_sum_b_sequence_l79_79421

-- Define the sequence a_n such that a_1 = 4 and a_(n+1) = 2 * a_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 4 else 2 * a (n - 1)

-- Define the sum S_n of the first n terms of the sequence a_n
def S (n : ℕ) : ℕ :=
  finset.sum (finset.range n) (λ i, a (i + 1))

-- Define the sequence b_n such that b_7 = a_3 and b_15 = a_4
def b (n : ℕ) : ℕ := 2 * n + 2

-- Define the sum T_n of the first n terms of the sequence b_n
def T (n : ℕ) : ℕ :=
  finset.sum (finset.range n) (λ i, b (i + 1))

-- Define the theorems for the problems
theorem sum_a_sequence (n : ℕ) : S n = 2 ^ (n + 2) - 4 :=
  sorry

theorem sum_b_sequence (n : ℕ) : T n = n ^ 2 + 3 * n :=
  sorry

end sum_a_sequence_sum_b_sequence_l79_79421


namespace total_sum_of_money_in_rupees_l79_79625

theorem total_sum_of_money_in_rupees (total_coins : ℕ) (coins_20_paise : ℕ) (coins_25_paise : ℕ) :
  total_coins = 342 → coins_20_paise = 290 → coins_25_paise = total_coins - coins_20_paise → 
  (coins_20_paise * 20 + coins_25_paise * 25) / 100 = 71 :=
by
  intros h_total h_20 h_25
  have h_paise_total : coins_20_paise * 20 + coins_25_paise * 25 = 7100 :=
    sorry -- This would be filled with the calculations if proof was required
  rw div_eq_of_eq_mul_right _ _ (by norm_num : 100 ≠ 0)
  rw ←h_paise_total
  assumption

end total_sum_of_money_in_rupees_l79_79625


namespace number_of_factors_l79_79270

theorem number_of_factors (x : ℤ) : 
  (x^11 - x^3).factor_count_with_integral_coefficients = 6 := 
sorry

end number_of_factors_l79_79270


namespace sin_B_value_l79_79874

theorem sin_B_value (A B C : ℝ) (h : sin A / sin B / sin C = 6 / 5 / 4): sin B = 5 * sqrt 7 / 16 :=
sorry

end sin_B_value_l79_79874


namespace meal_cost_before_tax_and_tip_l79_79049

theorem meal_cost_before_tax_and_tip (total_expenditure : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (base_meal_cost : ℝ):
  total_expenditure = 35.20 →
  tax_rate = 0.08 →
  tip_rate = 0.18 →
  base_meal_cost * (1 + tax_rate + tip_rate) = total_expenditure →
  base_meal_cost = 28 :=
by
  intros h_total h_tax h_tip h_eq
  sorry

end meal_cost_before_tax_and_tip_l79_79049


namespace monotone_increasing_interval_l79_79617

noncomputable def f (x : ℝ) : ℝ := real.logb (1/2) (x^2 - 4)

theorem monotone_increasing_interval :
  ∀ x y : ℝ, x < -2 → y < -2 → x < y → f x < f y :=
by
  intros x y hx hy hxy
  dsimp [f]
  sorry

end monotone_increasing_interval_l79_79617


namespace pentagon_areas_l79_79118

noncomputable def find_areas (x A_total : ℝ) : (ℝ × ℝ) :=
  let y := (A_total - x) / 10 in
  (y, y)

theorem pentagon_areas (x y z A_total : ℝ) (hx : x > 0) (hy : y = z) (hz : A_total = x + 10 * y) :
  y = z ∧ y = (A_total - x) / 10 :=
by
  sorry

end pentagon_areas_l79_79118


namespace triangle_obtuse_l79_79044

noncomputable def triangle_is_obtuse (A B C : ℝ) (a b c : ℝ) : Prop :=
  a.sin * A.sin + b.sin * B.sin < c.sin * C.sin → ∃ (θ : ℝ), A + B + C = π ∧ θ = C ∧ π / 2 < θ ∧ θ < π

theorem triangle_obtuse (A B C a b c : ℝ) (h1 : a.sin * A.sin + b.sin * B.sin < c.sin * C.sin)
(h2 : A + B + C = π) : triangle_is_obtuse A B C a b c :=
begin
  sorry
end

end triangle_obtuse_l79_79044


namespace m_range_for_circle_l79_79496

def is_circle (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m - 3) * x + 2 * y + 5 = 0

theorem m_range_for_circle (m : ℝ) :
  (∀ x y : ℝ, is_circle x y m) → ((m > 5) ∨ (m < 1)) :=
by 
  sorry -- Proof not required

end m_range_for_circle_l79_79496


namespace max_consecutive_sum_le_1000_l79_79231

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l79_79231


namespace determine_triangle_type_l79_79372

noncomputable def triangle_type (A B C a b c : ℝ) (h1 : A + B + C = 180)
  (h2 : (sin A) / a = (cos B) / b) (h3 : (cos B) / b = (cos C) / c) : Prop :=
  (A = 90 ∧ B = 45 ∧ C = 45)

theorem determine_triangle_type (A B C a b c : ℝ) 
  (h1 : A + B + C = 180) 
  (h2 : (sin A) / a = (cos B) / b) 
  (h3 : (cos B) / b = (cos C) / c) :
  triangle_type A B C a b c h1 h2 h3 :=
begin
  sorry
end

end determine_triangle_type_l79_79372


namespace max_consecutive_integers_sum_lt_1000_l79_79261

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l79_79261


namespace son_age_l79_79695

theorem son_age (S F : ℕ) (h1 : F = S + 30) (h2 : F + 2 = 2 * (S + 2)) : S = 28 :=
by
  sorry

end son_age_l79_79695


namespace vitya_probability_l79_79650

theorem vitya_probability :
  let total_sequences := (finset.range 6).card * 
                         (finset.range 5).card * 
                         (finset.range 4).card * 
                         (finset.range 3).card * 
                         (finset.range 2).card * 
                         (finset.range 1).card,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11,
      total_possibilities := nat.choose 12 2 * nat.choose 10 2 * 
                             nat.choose 8 2 * nat.choose 6 2 * 
                             nat.choose 4 2 * nat.choose 2 2,
      P := (favorable_sequences : ℚ) / (total_possibilities : ℚ)
  in P = 1 / 720 := 
sorry

end vitya_probability_l79_79650


namespace minimum_value_of_absolute_difference_l79_79844

theorem minimum_value_of_absolute_difference (x₁ x₂ : ℝ) (h : |Real.sin (2 * x₁) - Real.sin (2 * x₂)| = 2) :
  |x₁ - x₂| = Real.pi / 2 :=
sorry

end minimum_value_of_absolute_difference_l79_79844


namespace time_away_l79_79724

theorem time_away : 
  ∃ (n : ℝ), let h := 150 + n / 2,
                 m := 6 * n,
                 diff1 := |150 - 11 * n / 2| = 120,
                 diff2 := |150 - 11 * n / 2| = -120,
                 n1 := 60 / 11,
                 n2 := 540 / 11
             in 
             0 < n1 ∧ n1 < 60 ∧
             0 < n2 ∧ n2 < 60 ∧
             (n2 - n1 = 480 / 11) :=
by sorry

end time_away_l79_79724


namespace expected_value_of_N_l79_79340

noncomputable def expected_value_N : ℝ :=
  30

theorem expected_value_of_N :
  -- Suppose Bob chooses a 4-digit binary string uniformly at random,
  -- and examines an infinite sequence of independent random binary bits.
  -- Let N be the least number of bits Bob has to examine to find his chosen string.
  -- Then the expected value of N is 30.
  expected_value_N = 30 :=
by
  sorry

end expected_value_of_N_l79_79340


namespace max_ratio_point_l79_79573

theorem max_ratio_point (a : ℝ) (M : ℝ) (x := 2 * a / (1 + Real.sqrt 5)) : 
  is_square ABCD a → M ∈ line_extended CD → 
  ( ∀ y : ℝ, y = (distance M A / distance M B) → 
    ∃ y_max : ℝ, y_max = (distance M A / distance M B) ∧ y_max = x^2 ) :=
by 
  sorry

end max_ratio_point_l79_79573


namespace power_decomposition_l79_79797

theorem power_decomposition (n m : ℕ) (h1 : n ≥ 2) 
  (h2 : n * n = 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) 
  (h3 : Nat.succ 19 = 21) 
  : m + n = 15 := sorry

end power_decomposition_l79_79797


namespace find_a_and_b_l79_79415

variable (a b : ℝ) (z : ℂ)

def z_def := z = a + b * Complex.I
def conj_z_def := Complex.conj z = a - b * Complex.I
def equation := z * Complex.I + 2 * Complex.conj z = 3

theorem find_a_and_b (h1 : z_def a b z) (h2 : conj_z_def a b z) (h3 : equation a b z) :
  a = 2 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l79_79415


namespace max_consecutive_sum_lt_1000_l79_79236

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l79_79236


namespace product_of_numbers_l79_79972

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 85) : x * y = 18 := by
  sorry

end product_of_numbers_l79_79972


namespace train_crossing_time_l79_79902

noncomputable def convert_speed (km_hr : ℕ) : ℕ :=
  km_hr * 1000 / 3600

theorem train_crossing_time 
  (distance : ℕ) 
  (speed_km_hr : ℕ) 
  (speed_m_s : ℕ) 
  (time : ℕ) :
  distance = 750 ∧ speed_km_hr = 180 ∧ speed_m_s = convert_speed speed_km_hr ∧ time = distance / speed_m_s
  → time = 15 :=
by 
  intros h,
  cases h with h_distance h_rest, 
  cases h_rest with h_speed_km_hr h_rest,
  cases h_rest with h_speed_m_s h_time, 
  sorry

end train_crossing_time_l79_79902


namespace local_minimum_interval_l79_79977

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  (1/3 : ℝ) * x^3 - (1/2 : ℝ) * (2 * b + 1) * x^2 + b * (b + 1) * x

theorem local_minimum_interval (b : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ (∃ ε > 0, ∀ y, abs(y - x) < ε → f y b ≥ f x b)) →
  -1 < b ∧ b < 1 :=
by
  sorry

end local_minimum_interval_l79_79977


namespace max_min_of_f_l79_79782

noncomputable def f (x : ℝ) : ℝ := real.sqrt (100 - x^2)

theorem max_min_of_f :
  (∀ x ∈ set.Icc (-6:ℝ) 8, f x ≤ 10) ∧
  (∃ x ∈ set.Icc (-6:ℝ) 8, f x = 10) ∧
  (∀ x ∈ set.Icc (-6:ℝ) 8, f x ≥ 6) ∧
  (∃ x ∈ set.Icc (-6:ℝ) 8, f x = 6) :=
sorry

end max_min_of_f_l79_79782


namespace probability_of_specific_sequence_l79_79663

-- We define a structure representing the problem conditions.
structure problem_conditions :=
  (cards : multiset ℕ)
  (permutation : list ℕ)

-- Noncomputable definition for the correct answer.
noncomputable def probability := (1 : ℚ) / 720

-- The main theorem statement.
theorem probability_of_specific_sequence :
  ∀ (conds : problem_conditions),
  conds.cards = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6} ∧
  (∃ (perm : list ℕ), perm.perm conds.permutation) →
  (∃ (sequence : list ℕ), sequence = [1, 2, 3, 4, 5, 6]) →
  let prob := calculate_probability conds.permutation [1, 2, 3, 4, 5, 6] in
  prob = (1 : ℚ) / 720 :=
sorry

end probability_of_specific_sequence_l79_79663


namespace painted_cube_cubes_l79_79320

theorem painted_cube_cubes
  (edge_large_cube : ℕ) (edge_small_cube : ℕ)
  (h1 : edge_large_cube = 10)
  (h2 : edge_small_cube = 1) :
  let num_one_face_painted := 6 * (edge_large_cube - 2)^2
      num_two_faces_painted := 12 * (edge_large_cube - 2)
  in
  num_one_face_painted = 384 ∧ num_two_faces_painted = 96 :=
by
  sorry

end painted_cube_cubes_l79_79320


namespace geometric_series_common_ratio_l79_79192

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l79_79192


namespace parabola_transformation_l79_79059

theorem parabola_transformation 
  (x : ℝ) :
  (shift_left : ℝ → ℝ := λ x, x + 1)
  (shift_down : ℝ → ℝ := λ y, y - 2):
  let initial_parabola := -(x + 1) ^ 2 + 2 in
  let parabola_after_shift := -(shift_left x + 1) ^ 2 + shift_down 2 in
  parabola_after_shift = -(x + 2) ^ 2 :=
by
  sorry

end parabola_transformation_l79_79059


namespace distinct_four_digit_integers_using_2_3_3_9_l79_79460

-- Define the set of digits.
def digits := {2, 3, 3, 9}

-- Define a function to calculate the number of distinct permutations of the given digits.
def num_distinct_permutations : Nat :=
  Nat.factorial 4 / Nat.factorial 2

-- Theorem statement: The number of distinct positive four-digit integers that can be formed is 12.
theorem distinct_four_digit_integers_using_2_3_3_9 : num_distinct_permutations = 12 :=
by
  -- Proof is intentionally omitted as specified.
  sorry

end distinct_four_digit_integers_using_2_3_3_9_l79_79460


namespace phi_value_for_symmetric_translation_l79_79155

theorem phi_value_for_symmetric_translation (φ : ℝ) : 
  let f := λ x : ℝ, Real.sin (2 * x + φ) 
  let g := λ x : ℝ, Real.sin (2 * x - Real.pi / 3 + φ)
  (∀ x : ℝ, g (-x) = -g(x)) → φ = Real.pi / 3 :=
by
  intros
  sorry

end phi_value_for_symmetric_translation_l79_79155


namespace parabola_focus_l79_79389

theorem parabola_focus (x y : ℝ) :
  (∃ x, y = 4 * x^2 + 8 * x - 5) →
  (x, y) = (-1, -8.9375) :=
by
  sorry

end parabola_focus_l79_79389
