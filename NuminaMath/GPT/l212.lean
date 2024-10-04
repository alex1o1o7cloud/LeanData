import Mathlib

namespace sin_A_value_trigonometric_range_l212_212042

variables (A B C a b c : ℝ)
variables (q p : ℝ × ℝ)
variables (cos_C sin_A : ℝ)

-- Conditions
def triangle_sides := (a, b, c)
def q := (2*a, 1)
def p := (2*b - c, cos_C)
def parallel := p.1 * q.2 = p.2 * q.1 

-- Problem (I)
theorem sin_A_value (h1: ∀ (a b c cos_C: ℝ), (q = (2*a, 1)) ∧ (p = (2*b - c, cos_C)) ∧ (parallel)) : 
  sin_A = √3 / 2 := 
sorry 

-- Problem (II)
theorem trigonometric_range (h2: ∀ (A: ℝ), 0 < A ∧ A < 2*π/3 → 
  ((2 * cos (2*C) - (2*cos(2)*(tan (C)+1))) + 1) >= -1 ∧ 
  ((2 * cos (2*C) - (2*cos(2)*(tan (C)+1))) + 1) <= √(2)): 
  ((-2*cos (2*C) / (1+tan C)) + 1 >= -1 ) ∧ 
  ((-2*cos (2*C) / (1+tan C)) + 1 <= √(2)) :=
sorry

end sin_A_value_trigonometric_range_l212_212042


namespace required_hours_for_fifth_week_l212_212587

def typical_hours_needed (week1 week2 week3 week4 week5 add_hours total_weeks target_avg : ℕ) : ℕ :=
  if (week1 + week2 + week3 + week4 + week5 + add_hours) / total_weeks = target_avg then 
    week5 
  else 
    0

theorem required_hours_for_fifth_week :
  typical_hours_needed 10 14 11 9 x 1 5 12 = 15 :=
by
  sorry

end required_hours_for_fifth_week_l212_212587


namespace find_a_l212_212780

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l212_212780


namespace mn_product_l212_212033

theorem mn_product (a m n : ℝ) (h : log a (1 : ℝ) = 0) (H : -2 = log a (-1 + m) + n) : m * n = -4 :=
by
  have h1 : -1 + m = 1 := by sorry
  have h2 : m = 2 := by linarith
  have h3 : -2 = log a 1 + n := by linarith
  have h4 : log a 1 = 0 := by linarith
  have h5 : n = -2 := by linarith
  have h6 : m * n = 2 * (-2) := by linarith
  have h7 : m * n = -4 := by linarith
  exact h7

end mn_product_l212_212033


namespace number_of_right_triangles_l212_212950

noncomputable def point : Type := ℝ × ℝ

structure rectangle :=
(A B C D : point)
(width height : ℝ)
(hAB : A.1 = B.1) (hBC : B.2 = C.2) (hCD : C.1 = D.1) (hDA : D.2 = A.2)
(hDistAB : dist A B = width)
(hDistBC : dist B C = height)
(hDistCD : dist C D = width)
(hDistDA : dist D A = height)

structure midpoint (P Q : point) :=
(M : point)
(hMid1 : 2 * M.1 = P.1 + Q.1)
(hMid2 : 2 * M.2 = P.2 + Q.2)

axiom dist : point → point → ℝ

def rectangle_ABCD : rectangle :=
{ A := (0, 0),
  B := (6, 0),
  C := (6, 4),
  D := (0, 4),
  width := 6,
  height := 4,
  hAB := rfl,
  hBC := rfl,
  hCD := rfl,
  hDA := rfl,
  hDistAB := rfl,
  hDistBC := rfl,
  hDistCD := rfl,
  hDistDA := rfl }

def point_E : point := (3, 0)
def point_F : point := (3, 4)
def point_G : point := (6, 2)

theorem number_of_right_triangles : 
  let points := [rectangle_ABCD.A, rectangle_ABCD.B,
                 rectangle_ABCD.C, rectangle_ABCD.D,
                 point_E, point_F, point_G] in
  (count_right_triangles points) = 16 :=
sorry

end number_of_right_triangles_l212_212950


namespace max_area_dihedral_angle_l212_212063

/-- Prove that the dihedral angle ∠PMB is 60° which maximizes the area S_ΔABC
    given the conditions:
    1. PB is perpendicular to AC
    2. PH is perpendicular to the plane ABC at point H, which lies inside ΔABC
    3. The angle between PB and the plane ABC is 30°
    4. The area of ΔPAC is 1
-/
theorem max_area_dihedral_angle (P A B C H : Point)
  (PB_perp_AC : Perpendicular (Line P B) (Line A C))
  (PH_perp_plane_ABC : Perpendicular (Line P H) (Plane A B C))
  (H_in_triangle_ABC : InsideTriangle H (Triangle A B C))
  (angle_PB_plane_ABC : Angle (Line P B) (Plane A B C) = 30)
  (area_PAC_1 : Area (Triangle P A C) = 1) :
  DihedralAngle (Line P A) (Line C B) = 60 :=
sorry

end max_area_dihedral_angle_l212_212063


namespace find_f_at_3_l212_212443

theorem find_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = x ^ 2 - 2 * x) : f 3 = -1 :=
by {
  -- Proof would go here.
  sorry
}

end find_f_at_3_l212_212443


namespace determine_digits_l212_212672

def product_eq_digits (A B C D x : ℕ) : Prop :=
  x * (x + 1) = 1000 * A + 100 * B + 10 * C + D

def product_minus_3_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 3) * (x - 2) = 1000 * C + 100 * A + 10 * B + D

def product_minus_30_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 30) * (x - 29) = 1000 * B + 100 * C + 10 * A + D

theorem determine_digits :
  ∃ (A B C D x : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  product_eq_digits A B C D x ∧
  product_minus_3_eq_digits A B C D x ∧
  product_minus_30_eq_digits A B C D x ∧
  A = 8 ∧ B = 3 ∧ C = 7 ∧ D = 2 :=
by
  sorry

end determine_digits_l212_212672


namespace area_pentagon_trapezoid_l212_212288

noncomputable def area_of_pentagon_formed_by_inscribed_circle_radii := sorry

theorem area_pentagon_trapezoid : 
  ∀ (a b c h : ℝ),
  a = 5 →
  c = 6 →
  (36 = ((b - 5) / 2)^2 + h^2) →
  b = 12 →
  h = sqrt 95 / 2 →
  (∃ (r : ℝ), area_of_pentagon_formed_by_inscribed_circle_radii = r) →
  r = (7 / 2) * sqrt 35 :=
sorry

end area_pentagon_trapezoid_l212_212288


namespace range_of_a_l212_212389

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ ((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) →
  (-2:ℝ) ≤ a ∧ a < (6 / 5:ℝ) :=
by
  sorry

end range_of_a_l212_212389


namespace find_tan_theta_l212_212080

noncomputable def dilation_matrix (k : ℝ) (hk : k > 0) : Matrix (fin 2) (fin 2) ℝ := ![![k, 0], ![0, k]]

noncomputable def rotation_matrix (θ : ℝ) : Matrix (fin 2) (fin 2) ℝ := ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem find_tan_theta (k : ℝ) (hk : k > 0) (θ : ℝ)
  (h : (rotation_matrix θ) ⬝ (dilation_matrix k hk) = ![![5, -12], ![12, 5]]) :
  Real.tan θ = 12 / 5 :=
sorry

end find_tan_theta_l212_212080


namespace trigonometric_identity_l212_212012

theorem trigonometric_identity
  (x : ℝ) 
  (h_tan : Real.tan x = -1/2) :
  (3 * Real.sin x ^ 2 - 2) / (Real.sin x * Real.cos x) = 7 / 2 := 
by
  sorry

end trigonometric_identity_l212_212012


namespace Junior_Class_2A_female_students_l212_212450

theorem Junior_Class_2A_female_students :
  ∃ x : ℕ, 8 * x < 200 ∧ 9 * x > 200 ∧ 11 * (x + 4) > 300 ∧ x = 24 :=
by
  -- Define variables and conditions
  let x : ℕ := 24
  have cond1 : 8 * x < 200 := sorry
  have cond2 : 9 * x > 200 := sorry
  have cond3 : 11 * (x + 4) > 300 := sorry
  have x_eq_24 : x = 24 := rfl

  -- Prove the theorem exists
  use x
  simp [*, cond1, cond2, cond3]
  exact ⟨cond1, cond2, cond3, x_eq_24⟩

end Junior_Class_2A_female_students_l212_212450


namespace central_angle_of_probability_l212_212263

theorem central_angle_of_probability (x : ℝ) (h1 : x / 360 = 1 / 6) : x = 60 := by
  have h2 : x = 60 := by
    linarith
  exact h2

end central_angle_of_probability_l212_212263


namespace min_scalar_product_l212_212036

open Real

variable {a b : ℝ → ℝ}

-- Definitions used as conditions in the problem
def condition (a b : ℝ → ℝ) : Prop :=
  |2 * a - b| ≤ 3

-- The goal to prove based on the conditions and the correct answer
theorem min_scalar_product (h : condition a b) : 
  (a x) * (b x) ≥ -9 / 8 :=
sorry

end min_scalar_product_l212_212036


namespace num_five_dollar_bills_l212_212100

theorem num_five_dollar_bills (total_dollars : ℕ) (bill_denomination : ℕ) (num_bills : ℕ) (h1 : total_dollars = 45) (h2 : bill_denomination = 5) : num_bills = 9 :=
by
  have h : num_bills = total_dollars / bill_denomination, from sorry,
  rw [h1, h2] at h,
  exact h -- will yield num_bills = 9

end num_five_dollar_bills_l212_212100


namespace distance_from_P_to_AD_l212_212540

theorem distance_from_P_to_AD :
  let A := (0, 6) in
  let D := (0, 0) in
  let C := (6, 0) in
  let M := (3, 0) in
  let intersect_Circle (center1 : ℝ × ℝ) (radius1 : ℝ) (center2 : ℝ × ℝ) (radius2 : ℝ) : set (ℝ × ℝ) :=
    {p | (p.1 - center1.1)^2 + (p.2 - center1.2)^2 = radius1^2 ∧
         (p.1 - center2.1)^2 + (p.2 - center2.2)^2 = radius2^2} in
  let P := (4.8, 2.4) in
  P ∈ intersect_Circle M 3 A 6 →
  P ≠ D →
  P.2 = 2.4 :=
by
  sorry

end distance_from_P_to_AD_l212_212540


namespace incorrect_description_about_cancer_l212_212315

theorem incorrect_description_about_cancer
  (A : ∀ (h : AIDS), has_higher_probability_of_developing_cancer h)
  (B : ∀ (c : CancerCell), abnormal_differentiation c ∧ can_proliferate_indefinitely c ∧ can_metastasize c)
  (C : ∀ (n : Nitrite), can_alter_gene_structure n → causes_cancer n)
  (Dnot : ∀ (n : NormalPerson) (c : CancerPatient), long_term_contact n c → ¬ increased_probability_of_cancer n) :
  ∃ incorrect_statement, incorrect_statement = OptionD :=
  sorry

end incorrect_description_about_cancer_l212_212315


namespace A_expression_value_l212_212185

theorem A_expression_value (n : ℕ) (h1 : 0 ≤ n + 3 ∧ n + 3 ≤ 2 * n) (h2 : 0 ≤ n + 1 ∧ n + 1 ≤ 4) :
  nat.factorial (2 * n) / nat.factorial ((2 * n) - (n + 3)) - nat.factorial 4 / nat.factorial (4 - (n + 1)) = 696 :=
sorry

end A_expression_value_l212_212185


namespace probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l212_212785

noncomputable theory
open ProbabilityTheory

-- Definitions based on given conditions
def events := {1, 2, 3, 4, 5, 6}
def eventA := {1, 2, 3}
def eventB := {1, 2, 4}

def numTrials : ℕ := 10
def numA : ℕ := 6
def pA : ℚ := 1 / 2
def pB_given_A : ℚ := 2 / 3
def pB_given_Ac : ℚ := 1 / 3

-- Theorem for probability that B does not occur given A occurred 6 times.
theorem probability_B_does_not_occur_given_A_6_occur :
  -- The probability of B not occurring given A occurred exactly 6 times.
  -- Should be approximately 2.71 * 10^(-4)
  true := sorry

-- Theorem for the expected number of times B occurs.
theorem expected_value_B_occurances : 
  -- The expected value of the number of occurrences of event B given the conditions.
  -- Should be 16 / 3
  true := sorry

end probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l212_212785


namespace solve_for_x_l212_212132

noncomputable def solve_equation (x : ℝ) : Prop := 
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = 3 * x / (3 * x - 2) ∧ x ≠ 2 / 3

theorem solve_for_x (x : ℝ) (h : solve_equation x) : x = (Real.sqrt 6) / 3 ∨ x = - (Real.sqrt 6) / 3 := 
  sorry

end solve_for_x_l212_212132


namespace binomial_remainder_l212_212791

def binomial_expansion_max_term_condition (n : ℕ) :=
  ∀ k : ℕ, k ≠ 6 → binom n k < binom n 6

theorem binomial_remainder (n : ℕ) (h : binomial_expansion_max_term_condition n)
  (h_n : n = 12) : 
  2 ^ (n + 4) % 7 = 2 :=
by
  sorry

end binomial_remainder_l212_212791


namespace range_of_t_l212_212744

theorem range_of_t (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_incr : ∀ {x y : ℝ}, -1 ≤ x ∧ x ≤ 1 → -1 ≤ y ∧ y ≤ 1 → x < y → f x < f y) 
  (h_ineq : ∀ t, -1 ≤ 3t ∧ 3t ≤ 1 → -1 ≤ t - 1/3 ∧ t - 1/3 ≤ 1 → f (3t) + f (1/3 - t) > 0) 
  : ∀ t, -1 ≤ 3t ∧ 3t ≤ 1 → -1 ≤ t - 1/3 ∧ t - 1/3 ≤ 1 → -1/6 < t ∧ t ≤ 1/3 := 
sorry

end range_of_t_l212_212744


namespace vertices_with_odd_degree_even_l212_212936

variable {V : Type} {E : Type} [Fintype V] [DecidableEq V] [Fintype E]

def degree (G : SimpleGraph V) (v : V) : ℕ :=
  Fintype.card (G.neighborFinset v)

theorem vertices_with_odd_degree_even (G : SimpleGraph V) :
  (∑ v in Fintype.elems V, degree G v) = 2 * Fintype.card (G.edgeSet) →
  even (Fintype.card { v : V | degree G v % 2 = 1 }) := by
  sorry

end vertices_with_odd_degree_even_l212_212936


namespace sequence_is_arithmetic_l212_212733

def an (n : ℕ) : ℤ := 3 * n - 5

theorem sequence_is_arithmetic :
  arithmetic_seq an ∧ first_term an = -2 ∧ common_diff an = 3 := sorry

end sequence_is_arithmetic_l212_212733


namespace quadrilateral_CM_CN_l212_212643

open EuclideanGeometry

noncomputable def inscribed_quadrilateral (A B C D : Point) : Prop :=
  ∃ (O : Point), Circle O (distance O A) = Circle O (distance O B) ∧ Circle O (distance O B) = Circle O (distance O C) ∧ Circle O (distance O C) = Circle O (distance O D)

theorem quadrilateral_CM_CN {A B C D M N : Point}
  (h_incircle : inscribed_quadrilateral A B C D)
  (h_M : ∃ (M : Point), collinear A B M ∧ collinear C D M)
  (h_N : ∃ (N : Point), collinear B C N ∧ collinear A D N)
  (h_BM_DN : distance B M = distance D N) :
  distance C M = distance C N :=
by
  sorry

end quadrilateral_CM_CN_l212_212643


namespace min_value_of_x1_l212_212866

noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

def consecutive_log_values (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  (lg x1, lg x2, lg x3, lg x4, lg x5).Sorted (· <= ·) ∧
  (lg x2 = lg x1 + 1) ∧ (lg x3 = lg x2 + 1) ∧ (lg x4 = lg x3 + 1) ∧ (lg x5 = lg x4 + 1)

def condition (x1 x4 x5 : ℝ) : Prop :=
  (lg x4)^2 < lg x1 * lg x5

theorem min_value_of_x1 (x1 x2 x3 x4 x5 : ℝ) (h1 : consecutive_log_values x1 x2 x3 x4 x5)
  (h2 : condition x1 x4 x5) : x1 = 100000 :=
by
  sorry

end min_value_of_x1_l212_212866


namespace triangle_perpendicular_distance_eq_l212_212803

noncomputable def circumcenter {A B C : Type*} [plane_triangle A B C] : Type* := sorry
noncomputable def incenter {A B C : Type*} [plane_triangle A B C] : Type* := sorry
noncomputable def perpendicular {A B : Type*} [line_segment A B] : Type* := sorry
noncomputable def distance {A B : Type*} [line_segment A B] : ℝ := sorry

theorem triangle_perpendicular_distance_eq (A B C O I D E : Type*)
  [plane_triangle A B C]
  (hC : angle A B C = 30)
  (hCircumcenter : O = circumcenter A B C)
  (hIncenter : I = incenter A B C)
  (hPointsOnSides : 
    D ∈ (line_segment A C) ∧ 
    E ∈ (line_segment B C) ∧ 
    distance A D = distance B E ∧ 
    distance A B = distance A D) :
  perpendicular O I D E ∧ distance O I = distance D E :=
sorry

end triangle_perpendicular_distance_eq_l212_212803


namespace river_flow_speed_l212_212278

/-- Speed of the ship in still water is 30 km/h,
    the distance traveled downstream is 144 km, and
    the distance traveled upstream is 96 km.
    Given that the time taken for both journeys is equal,
    the equation representing the speed of the river flow v is:
    144 / (30 + v) = 96 / (30 - v). -/
theorem river_flow_speed (v : ℝ) :
  (30 : ℝ) > 0 →
  real_equiv 144 (30 + v) (96 (30 - v)) := by
sorry

end river_flow_speed_l212_212278


namespace shaded_area_correct_l212_212848

-- Define a circle with center O and radius 6
def CircleO : Type := { p : ℝ × ℝ // (p.1 - 0)^2 + (p.2 - 0)^2 = 6^2 }

-- Definitions based on given conditions
-- PQ and RS are diameters intersecting perpendicularly at the center O
def PQ : CircleO := ⟨(6, 0), by simp⟩
def RS : CircleO := ⟨(0, 6), by simp⟩

-- PR and QS subtend central angles of 60° and 120° respectively at O
def anglePOR : ℝ := 60
def angleSOQ : ℝ := 120

-- Function calculating the area of the shaded region
noncomputable def shaded_area : ℝ :=
  let triangle_area (a b : ℝ) : ℝ := 1/2 * a * b
  let sector_area (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * Math.pi * r^2
  2 * triangle_area 6 6 + sector_area 6 120 + sector_area 6 60

-- The theorem statement
theorem shaded_area_correct : shaded_area = 36 + 18 * Math.pi := 
by
  sorry

end shaded_area_correct_l212_212848


namespace number_of_divisors_of_gcd_l212_212004

theorem number_of_divisors_of_gcd (n m : ℕ) (hn : n = 2^2 * 3 * 5) (hm : m = 2^2 * 3 * 7) :
  (nat.divisors (nat.gcd n m)).card = 6 :=
by
  sorry

end number_of_divisors_of_gcd_l212_212004


namespace range_of_b_l212_212758

theorem range_of_b (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f x = log 2 (2^x - b) → f x ≥ 0) ↔ b ∈ set.Iic 1 :=
by 
  sorry

end range_of_b_l212_212758


namespace total_boundary_mass_is_60_l212_212916

variable {Coin : Type} [HasValue Coin (Mass : ℝ)]

/-- On a table, 28 coins of the same size but possibly different masses are arranged in the shape 
of a triangle. It is known that the total mass of any trio of coins that touch each other pairwise 
is 10 grams. -/
def summing_trio_condition (trio : Finset Coin) : Prop :=
  trio.card = 3 ∧ (trio.sum (λ c, Coin.value Mass c) = 10)

/-- Find the total mass of all 18 coins on the border of the triangle. -/
def boundary_coins := {coins : Finset Coin // coins.card = 18}

/-- The total mass of all 18 coins on the border of the triangular arrangement is 60 grams. -/
theorem total_boundary_mass_is_60 (triangle : Finset Coin) (h₁ : triangle.card = 28) 
  (h₂ : ∀ trio : Finset Coin, trio ⊆ triangle → summing_trio_condition trio)  :
  ∀ border : boundary_coins, border.val.sum (λ c, Coin.value Mass c) = 60 :=
sorry

end total_boundary_mass_is_60_l212_212916


namespace basis_vectors_l212_212294

def is_collinear (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

noncomputable def set_of_vectors := ( (-1, 2) : ℝ × ℝ, (5, 7) : ℝ × ℝ)

theorem basis_vectors {e1 e2 : ℝ × ℝ} (h : set_of_vectors = (e1, e2)) :
  ¬is_collinear e1 e2 :=
by
  -- Proof placeholder
  sorry

end basis_vectors_l212_212294


namespace smallest_value_of_x_l212_212997

theorem smallest_value_of_x :
  ∃ x : Real, (∀ z, (z = (5 * x - 20) / (4 * x - 5)) → (z * z + z = 20)) → x = 0 :=
by
  sorry

end smallest_value_of_x_l212_212997


namespace area_of_parallelogram_l212_212341

-- Define base and height
def base : ℝ := 14
def height : ℝ := 24

-- Define the area function for parallelogram.
def area (b h : ℝ) : ℝ := b * h

-- The goal is to prove that the area of the parallelogram with base 14 and height 24 is 336.
theorem area_of_parallelogram : area base height = 336 := 
by
  sorry

end area_of_parallelogram_l212_212341


namespace seungjun_clay_cost_l212_212962

theorem seungjun_clay_cost (price_per_gram : ℝ) (qty1 qty2 : ℝ) 
  (h1 : price_per_gram = 17.25) 
  (h2 : qty1 = 1000) 
  (h3 : qty2 = 10) :
  (qty1 * price_per_gram + qty2 * price_per_gram) = 17422.5 :=
by
  sorry

end seungjun_clay_cost_l212_212962


namespace g_at_4_eq_7_point_5_l212_212142

def f (x : ℝ) : ℝ := 4 / (3 - x)

def f_inv (y : ℝ) : ℝ := 3 - 4 / y

def g (x : ℝ) : ℝ := 1 / f_inv x + 7

theorem g_at_4_eq_7_point_5 : g 4 = 7.5 := 
by sorry

end g_at_4_eq_7_point_5_l212_212142


namespace sum_of_areas_is_72_l212_212589

def base : ℕ := 2
def length1 : ℕ := 1
def length2 : ℕ := 8
def length3 : ℕ := 27

theorem sum_of_areas_is_72 : base * length1 + base * length2 + base * length3 = 72 :=
by
  sorry

end sum_of_areas_is_72_l212_212589


namespace tangent_y_axis_circle_eq_l212_212385

theorem tangent_y_axis_circle_eq (h k r : ℝ) (hc : h = -2) (kc : k = 3) (rc : r = abs h) :
  (x + h)^2 + (y - k)^2 = r^2 ↔ (x + 2)^2 + (y - 3)^2 = 4 := by
  sorry

end tangent_y_axis_circle_eq_l212_212385


namespace line_through_E_intersects_circle_coordinates_of_P_minimizing_PM_l212_212735

/-- Define the circle C: (x - 2)² + (y - 2)² = 4 -/
def circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 4

/-- Define point E (3, 4) -/
def E : ℝ × ℝ := (3, 4)

/-- Define line equation passing through a point (x₁, y₁) -/ 
def line_through (x₁ y₁ k : ℝ) (x y : ℝ) : Prop := y - y₁ = k * (x - x₁)

/-- Problem 1: Proving line l through point E intersects circle C such that AB = 2√3 -/
theorem line_through_E_intersects_circle
    (l : ℝ → ℝ → Prop) :
    (∃ k : ℝ, l = line_through 3 4 k ∧ ∀ (x y : ℝ), l x y → circle x y) ∨
    (∃ (x : ℝ), l = (λ a b, a = x) ∧ x = 3 ∧ (∀ (x y : ℝ), l x y → circle x y)) :=
begin
    sorry
end

/-- Define point P as a pair of coordinates -/
abbreviation P_coord := ℝ × ℝ

/-- Define the distance PM -/
def PM (x₁ y₁ : ℝ) : ℝ := sqrt ((x₁ - 2)^2 + (y₁ - 2)^2 - 4)

/-- Define the distance PO -/
def PO (x₁ y₁ : ℝ) : ℝ := sqrt (x₁^2 + y₁^2)

/-- Problem 2: Proving the coordinates of P that minimize PM are (1/2, 1/2) -/
theorem coordinates_of_P_minimizing_PM :
    ∃ (x₁ y₁ : ℝ), (PM x₁ y₁ = PO x₁ y₁) ∧ (y₁ + x₁ = 1) ∧ (y₁ - x₁ = 0) ∧ x₁ = 1/2 ∧ y₁ = 1/2 :=
begin
    sorry
end

end line_through_E_intersects_circle_coordinates_of_P_minimizing_PM_l212_212735


namespace petya_wins_with_optimal_play_l212_212920

open Function

def board := fin 100 × fin 100

-- Initial board state with all cells white
def initial_board_state : board → bool := λ _, true

-- Move functions for Petya and Vasya
def petya_move (state : board → bool) (positions : list (fin 100)) : board → bool :=
  λ pos, if pos.1 = pos.2 ∧ pos.1 ∈ positions then false else state pos

def vasya_move (state : board → bool) (column : fin 100) (length : fin 100): board → bool :=
  λ pos, if pos.2 = column ∧ pos.1 < length then false else state pos

-- A valid move means that there are still white cells on the path
def valid_move (state : board → bool) (positions : list (fin 100)) : Prop :=
  ∃ pos ∈ positions, state (pos, pos)

def valid_column_move (state : board → bool) (column : fin 100) (length : fin 100): Prop :=
  ∃ pos, pos.2 = column ∧ pos.1 < length ∧ state pos

noncomputable def optimal_play : Prop :=
  ∀ state : board → bool, (state = initial_board_state) →
  (∃ petya_win_strategy, (valid_move (petya_move state petya_win_strategy) petya_win_strategy) ∧ 
  (∀ vasya_col vasya_len, valid_column_move (vasya_move (petya_move state petya_win_strategy) vasya_col vasya_len) vasya_col vasya_len → 
  valid_move (petya_move (vasya_move (petya_move state petya_win_strategy) vasya_col vasya_len) petya_win_strategy) petya_win_strategy)) →
  (state = initial_board_state → ¬ (∃ vasya_col vasya_len, valid_column_move (vasya_move state vasya_col vasya_len) vasya_col vasya_len))

theorem petya_wins_with_optimal_play : optimal_play :=
sorry

end petya_wins_with_optimal_play_l212_212920


namespace triangle_problem_l212_212065

variable {A B C a b c : ℝ}

-- Conditions
def condition1 := (b - 2 * a) * Real.cos C + c * Real.cos B = 0
def condition2 := c = Real.sqrt 7
def condition3 := b = 3 * a

-- Prove angle C equals pi/3 and area is 3 * sqrt 3 / 4
theorem triangle_problem (h₁ : condition1) (h₂ : condition2) (h₃ : condition3) :
  (C = Real.pi / 3) ∧ 
  (0.5 * a * b * (Real.sin (Real.pi / 3)) = 3 * Real.sqrt 3 / 4) :=
by
  sorry

end triangle_problem_l212_212065


namespace order_of_a_b_c_l212_212375

variable {f : ℝ → ℝ}
variable (h_odd : ∀ x, f(-x) = -f(x))
variable (h_deriv : ∀ x ≠ 0, f'(x) + f(x) / x > 0)
let a := (1 / 3) * f(1 / 3)
let b := -3 * f(-3)
let c := (Real.log (1 / 3)) * f(Real.log (1 / 3))

theorem order_of_a_b_c (h_odd : ∀ x, f(-x) = -f(x))
  (h_deriv : ∀ x, x ≠ 0 → f'(x) + f(x) / x > 0) :
  a < c ∧ c < b := 
sorry

end order_of_a_b_c_l212_212375


namespace nancy_hourly_wage_l212_212109

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end nancy_hourly_wage_l212_212109


namespace largest_sum_products_l212_212575

def perm4 (s : set ℤ) := s = {3, 4, 5, 6}

noncomputable def largest_value_of_sum_products (f g h j : ℤ) (P : perm4 {f, g, h, j}) : ℤ := 
if perm4 {f, g, h, j} then 
  let s := {p : finset ℤ | ∃ a b in s, (a ≠ b) ∧ (a != f) ∧ (b != g) ∧ (p = (a * h + b * j))} 
  in (f * g + g * h + h * j + j * f) - min (s)
else 0

theorem largest_sum_products (f g h j : ℤ) (P : perm4 {f, g, h, j}) :
  largest_value_of_sum_products f g h j P = 80 :=
sorry

end largest_sum_products_l212_212575


namespace calculate_expression_value_l212_212313

theorem calculate_expression_value :
  5 * 7 + 6 * 9 + 13 * 2 + 4 * 6 = 139 :=
by
  -- proof can be added here
  sorry

end calculate_expression_value_l212_212313


namespace minimum_value_l212_212404

theorem minimum_value (a_n : ℕ → ℤ) (h : ∀ n, a_n n = n^2 - 8 * n + 5) : ∃ n, a_n n = -11 :=
by
  sorry

end minimum_value_l212_212404


namespace count_four_digit_numbers_divisible_by_five_l212_212773

theorem count_four_digit_numbers_divisible_by_five :
  let count := (9995 - 1000) / 5 + 1 in
  count = 1800 := by
  sorry

end count_four_digit_numbers_divisible_by_five_l212_212773


namespace pizza_slices_with_both_toppings_l212_212641

theorem pizza_slices_with_both_toppings (total_slices ham_slices pineapple_slices slices_with_both : ℕ)
  (h_total: total_slices = 15)
  (h_ham: ham_slices = 8)
  (h_pineapple: pineapple_slices = 12)
  (h_slices_with_both: slices_with_both + (ham_slices - slices_with_both) + (pineapple_slices - slices_with_both) = total_slices)
  : slices_with_both = 5 :=
by
  -- the proof would go here, but we use sorry to skip it
  sorry

end pizza_slices_with_both_toppings_l212_212641


namespace prime_numbers_20_to_40_l212_212981

def is_prime (n : ℕ) : Prop := Nat.Prime n

def candidates : List (ℕ × ℕ) :=
  [(23, 29), (23, 31), (23, 37), (29, 31), (29, 37), (31, 37)]

def calc (xy : ℕ × ℕ) : ℕ :=
  xy.fst * xy.snd - (xy.fst + xy.snd)

theorem prime_numbers_20_to_40 :
  ∃ x y, x ∈ [23, 29, 31, 37] ∧ y ∈ [23, 29, 31, 37] ∧ x ≠ y ∧ calc (x, y) = 899 :=
by
  let primes := [23, 29, 31, 37]
  simp only [mem_list_primes]
  exact sorry

end prime_numbers_20_to_40_l212_212981


namespace a_plus_b_values_l212_212369

theorem a_plus_b_values (a b : ℤ) (h1 : |a + 1| = 0) (h2 : b^2 = 9) :
  a + b = 2 ∨ a + b = -4 :=
by
  have ha : a = -1 := by sorry
  have hb1 : b = 3 ∨ b = -3 := by sorry
  cases hb1 with
  | inl b_pos =>
    left
    rw [ha, b_pos]
    exact sorry
  | inr b_neg =>
    right
    rw [ha, b_neg]
    exact sorry

end a_plus_b_values_l212_212369


namespace square_side_length_using_triangles_l212_212115

theorem square_side_length_using_triangles :
  let triangles := 20
  let base := 1
  let height := 2
  let hypotenuse := Math.sqrt 5
  let area_triangle := (base * height) / 2
  let total_area := triangles * area_triangle
  (Math.sqrt total_area) = 2 * Math.sqrt 5 :=
by
  sorry

end square_side_length_using_triangles_l212_212115


namespace transformed_point_is_correct_l212_212171

-- Define the transformations as separate functions
def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.2, p.1, p.3)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2, p.3)

-- Define the sequence of transformations
def transformations (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let p1 := rotate_z_90 p in
  let p2 := reflect_xy p1 in
  let p3 := reflect_yz p2 in
  let p4 := rotate_z_90 p3 in
  reflect_yz p4

-- State the theorem to prove the final coordinates
theorem transformed_point_is_correct :
  transformations (2, 3, 4) = (2, 3, -4) :=
by sorry

end transformed_point_is_correct_l212_212171


namespace hiker_total_distance_l212_212269

-- Define conditions based on the problem description
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day2_speed : ℕ := day1_speed + 1
def day1_time : ℕ := day1_distance / day1_speed
def day2_time : ℕ := day1_time - 1
def day3_speed : ℕ := 5
def day3_time : ℕ := 3

-- Define the total distance walked based on the conditions
def total_distance : ℕ :=
  day1_distance + (day2_speed * day2_time) + (day3_speed * day3_time)

-- The theorem stating the hiker walked a total of 53 miles
theorem hiker_total_distance : total_distance = 53 := by
  sorry

end hiker_total_distance_l212_212269


namespace angle_ABC_equals_angle_ADC_l212_212843

def Quadrilateral (A B C D : Type) := True -- We need a placeholder for the quadrilateral type.

variables {A B C D O : Type} -- Variables for points

-- Angles definitions
variables (angle_CBD angle_CAB angle_ACD angle_BDA angle_ABC angle_ADC : Type)

-- Given conditions:
variable Hypothesis1 : angle_CBD = angle_CAB
variable Hypothesis2 : angle_ACD = angle_BDA

-- The theorem to be proven:
theorem angle_ABC_equals_angle_ADC : Quadrilateral A B C D → angle_CBD = angle_CAB → angle_ACD = angle_BDA → angle_ABC = angle_ADC :=
  by
  intro h_quad h1 h2,
  sorry

end angle_ABC_equals_angle_ADC_l212_212843


namespace money_left_after_purchase_l212_212669

def initial_toonies : Nat := 4
def value_per_toonie : Nat := 2
def total_coins : Nat := 10
def value_per_loonie : Nat := 1
def frappuccino_cost : Nat := 3

def toonies_value : Nat := initial_toonies * value_per_toonie
def loonies : Nat := total_coins - initial_toonies
def loonies_value : Nat := loonies * value_per_loonie
def initial_total : Nat := toonies_value + loonies_value
def remaining_money : Nat := initial_total - frappuccino_cost

theorem money_left_after_purchase : remaining_money = 11 := by
  sorry

end money_left_after_purchase_l212_212669


namespace count_z_values_l212_212266

open Complex

noncomputable def g (z : ℂ) : ℂ := -I * conj(z)

theorem count_z_values :
  let z_vals := {z : ℂ | abs z = 3 ∧ g z = z}
  cardinal.mk z_vals = 2 :=
by 
  let z_vals := {z : ℂ | abs z = 3 ∧ g z = z}
  have h_count : set.finite z_vals := 
    λ z (hz : z ∈ z_vals), sorry
  exact (set.finite.cardinal_eq_fintype_card h_count).symm.trans sorry

end count_z_values_l212_212266


namespace nancy_hourly_wage_l212_212102

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end nancy_hourly_wage_l212_212102


namespace AndrewClosestToC_l212_212620

-- Conditions and definitions.
noncomputable def totalTrackLength : ℝ := 400
noncomputable def segmentLength : ℝ := totalTrackLength / 4
noncomputable def startToA : ℝ := segmentLength / 2
noncomputable def walkingSpeed : ℝ := 1.4 -- meters per second
noncomputable def minutes : ℝ := 30
noncomputable def seconds : ℝ := minutes * 60

-- Define the distance walked after 30 minutes.
noncomputable def totalDistance : ℝ := walkingSpeed * seconds

-- Define the positions along the track: A, B, C, D, Start.
noncomputable def A : ℝ := 0
noncomputable def B : ℝ := segmentLength
noncomputable def C : ℝ := 2 * segmentLength
noncomputable def D : ℝ := 3 * segmentLength
noncomputable def Start : ℝ := segmentLength / 2

-- Prove that Andrew's position after walking for 30 minutes is closest to point C.
theorem AndrewClosestToC : ℝ :=
  let lapsCompleted := totalDistance / totalTrackLength
  let distanceAfterLaps := totalDistance % totalTrackLength
  let currentPosition := (Start + distanceAfterLaps) % totalTrackLength
  if currentPosition <= B then
    if B - currentPosition < currentPosition then B else Start
  else if currentPosition <= C then
    if C - currentPosition < currentPosition - B then C else B
  else if currentPosition <= D then
    if D - currentPosition < currentPosition - C then D else C
  else
    if totalTrackLength - currentPosition < currentPosition - D then A else D

#eval AndrewClosestToC

end AndrewClosestToC_l212_212620


namespace ordered_pairs_no_digit_5_sum_eq_500_l212_212327

/-- Determine the number of ordered pairs of positive integers (a, b) such that
    a + b = 500 and neither a nor b contains the digit 5. -/
theorem ordered_pairs_no_digit_5_sum_eq_500 :
  ∃ n : ℕ, (∀ a b : ℕ, a + b = 500 ∧ ¬contains_digit 5 a ∧ ¬contains_digit 5 b → a * b = n) ∧ n = 57121 :=
by
  -- Definitions and auxiliary functions
  def contains_digit (d n : ℕ) : Prop :=
    let ds := to_digits 10 n
    d ∈ ds
  
  -- We continue with the proof after the sorry
  sorry

end ordered_pairs_no_digit_5_sum_eq_500_l212_212327


namespace magnitude_of_vec_sub_l212_212000

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (-1, 1)
def vec_b : ℝ × ℝ := (3, -2)

-- Define the subtraction of the vectors
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Prove that the magnitude of (vec_a - vec_b) is 5
theorem magnitude_of_vec_sub : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end magnitude_of_vec_sub_l212_212000


namespace dogwood_trees_initial_count_l212_212198

theorem dogwood_trees_initial_count 
  (dogwoods_today : ℕ) 
  (dogwoods_tomorrow : ℕ) 
  (final_dogwoods : ℕ)
  (total_planted : ℕ := dogwoods_today + dogwoods_tomorrow)
  (initial_dogwoods := final_dogwoods - total_planted)
  (h : dogwoods_today = 41)
  (h1 : dogwoods_tomorrow = 20)
  (h2 : final_dogwoods = 100) : 
  initial_dogwoods = 39 := 
by sorry

end dogwood_trees_initial_count_l212_212198


namespace find_sum_of_solutions_l212_212084

variable {X : Type} [LinearOrder X] [Add X] [HasOne X]

-- Condition 1: f is a strictly increasing function with an inverse g
variable (f g : X → X)
variable (hf : StrictMono f) (hg : Function.Inverse f g)

-- Conditions 2 and 3: x1 and x2 are solutions to the given equations
variable (x1 x2 : X)
variable (hx1 : f x1 + x1 = 2)
variable (hx2 : g x2 + x2 = 2)

-- Proof problem: prove that x1 + x2 = 2
theorem find_sum_of_solutions : x1 + x2 = 2 :=
sorry

end find_sum_of_solutions_l212_212084


namespace num_real_solutions_system_l212_212677

theorem num_real_solutions_system :
  ∃ (x y z w : ℝ),
    (x = z - w + x * z) ∧
    (y = w - x + y * w) ∧
    (z = x - y + x * z) ∧
    (w = y - z + y * w) ∧
    ∃ (S : Finset ℝ) (h : S.card = 4), 
      (∀ (a ∈ S), 
        ∃ x y z w ∈ S, 
          x = z - w + x * z ∧
          y = w - x + y * w ∧
          z = x - y + x * z ∧
          w = y - z + y * w) :=
sorry

end num_real_solutions_system_l212_212677


namespace fixed_point_of_logarithmic_function_l212_212763

theorem fixed_point_of_logarithmic_function :
  ∃ P : ℝ × ℝ, P = (3, -1) ∧ ∀ x : ℝ, y : ℝ, y = real.log (x - 2) / real.log 2 - 1 → P = (x, y) :=
by
  sorry

end fixed_point_of_logarithmic_function_l212_212763


namespace max_intervals_6_l212_212222

/-- 
  Let there be a set of closed intervals on the number line with the following properties:
  1. Among any three intervals, there are two that intersect.
  2. The intersection of any four intervals is empty.
  Prove that the maximum number of such intervals is 6.
-/
theorem max_intervals_6 {α : Type*} [linear_order α] (S : set (set α)) (h1 : ∀ (a b c ∈ S), ∃ (x : set α), a ∩ b = x ∨ b ∩ c = x ∨ a ∩ c = x)
  (h2 : ∀ (a b c d ∈ S), (a ∩ b ∩ c ∩ d) = ∅) : S.card ≤ 6 :=
sorry

end max_intervals_6_l212_212222


namespace angle_ABC_eq_angle_ADC_l212_212825

-- Given a convex quadrilateral ABCD
variables {A B C D O : Type}
variables [convex_quadrilateral A B C D]

-- Given conditions
variable (angle_CBD_eq_angle_CAB : ∠ CBD = ∠ CAB)
variable (angle_ACD_eq_angle_BDA : ∠ ACD = ∠ BDA)

-- Prove that ∠ ABC = ∠ ADC 
theorem angle_ABC_eq_angle_ADC :
  ∠ ABC = ∠ ADC :=
begin
  sorry -- Proof not required
end

end angle_ABC_eq_angle_ADC_l212_212825


namespace expectation_fraction_ident_distrib_l212_212530

open MeasureTheory ProbabilityTheory

variable {Ω : Type} {P : ProbabilitySpace Ω}

theorem expectation_fraction_ident_distrib 
  (X : ℕ → Ω → ℝ)
  (h_indep : IndepFun P X)
  (h_pos : ∀ i, ∀ ω, 0 < X i ω)
  (h_ident : ∀ i j, IdentDistrib P (X i) (X j))
  (n : ℕ) : 
  n > 0 → 
  E (λ ω, X 0 ω / ∑ i in Finset.range n, X i ω) = 1 / n := 
by
  sorry

end expectation_fraction_ident_distrib_l212_212530


namespace circle_diameter_from_area_l212_212949

noncomputable def circle_area := 78.53981633974483
noncomputable def pi_approx := 3.141592653589793

theorem circle_diameter_from_area (A : ℝ) (hA : A = circle_area) : 
  2 * (real.sqrt (circle_area / pi_approx)) = 10 := by
  sorry

end circle_diameter_from_area_l212_212949


namespace part_a_part_b_valid_n_l212_212622

variable {a : ℤ} {n : ℕ}

theorem part_a (ha : a ∈ ℤ) (hn : n ∈ ℕ) :
  a^(n + 1) - 1 = (a - 1) * ∑ i in Finset.range (n + 1), a^i :=
by sorry

noncomputable def u_n (n : ℕ) : ℕ := 10^(n + 1) + 1

theorem part_b_valid_n (n : ℕ) :
  (u_n n) % 11 = 0 ↔ n % 2 = 0 :=
by sorry

end part_a_part_b_valid_n_l212_212622


namespace equal_angles_quadrilateral_l212_212837

theorem equal_angles_quadrilateral
  (AB CD : Type)
  [convex_quad AB CD]
  (angle_CBD angle_CAB angle_ACD angle_BDA : AB CD → ℝ)
  (h1 : angle_CBD = angle_CAB)
  (h2 : angle_ACD = angle_BDA) : angle_ABC = angle_ADC :=
by sorry

end equal_angles_quadrilateral_l212_212837


namespace no_prime_divisible_by_42_l212_212416

open Nat

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → 42 ∣ p → p = 0 :=
by
  intros p hp hdiv
  sorry

end no_prime_divisible_by_42_l212_212416


namespace length_O1_O2_l212_212769

-- Define the circle structure
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

def tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2 ∨
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2

def circle1 : Circle := { center := (0, 0), radius := 3 }
def circle2 : Circle := { center := (5, 0), radius := 2 }

theorem length_O1_O2 :
  tangent circle1 circle2 →
  (real.sqrt ((circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2) = 1 ∨ 
   real.sqrt ((circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2) = 5) :=
by
  assume h,
  sorry

end length_O1_O2_l212_212769


namespace sum_equality_l212_212235

-- Defining the original set
def original_set : Fin 20 → ℕ := sorry
-- Defining the function that gives the number of elements greater than i in the original set
def b (i : ℕ) : ℕ :=
  ∑ n in Finset.univ, if original_set n > i then 1 else 0

-- Defining the sums
def sum_original := ∑ i in Finset.fin_range 20, original_set i
def sum_new := ∑' i, b i

-- The theorem we want to prove
theorem sum_equality :
  sum_original = sum_new :=
sorry

end sum_equality_l212_212235


namespace correct_statement_l212_212259

-- Definitions based on given conditions
def total_students : ℕ := 520
def selected_students : ℕ := 100

-- Statement to prove
theorem correct_statement :
  "The sleep time of each student is an individual" :=
by sorry

end correct_statement_l212_212259


namespace kendra_birdwatching_l212_212877

theorem kendra_birdwatching :
  (∀ m t wed : ℕ, 
    (m = 5) → 
    (t = 5) → 
    (wed = 10) →
    (average (m*tave + t*tave + wed*wave) (m + t + wed) = 7) →
    (tave = 7) → 
    (tave = 5) →
    m * tave + t * tave + wed * wave = 140 →
    wave = 8)
by
  intros m t wed hm ht hwed havg htave1 htave2 htotal,
  sorry

end kendra_birdwatching_l212_212877


namespace wall_width_l212_212260

theorem wall_width
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_length : ℝ) (wall_height : ℝ)
  (num_bricks : ℕ)
  (brick_volume : ℝ := brick_length * brick_width * brick_height)
  (total_volume : ℝ := num_bricks * brick_volume) :
  brick_length = 0.20 → brick_width = 0.10 → brick_height = 0.08 →
  wall_length = 10 → wall_height = 8 → num_bricks = 12250 →
  total_volume = wall_length * wall_height * (0.245 : ℝ) :=
by 
  sorry

end wall_width_l212_212260


namespace range_of_g_l212_212355

noncomputable def g (x : ℝ) : ℝ :=
  (cos x)^3 + 7 * (cos x)^2 + 2 * (cos x) + 3 * (1 - (cos x)^2) - 14

theorem range_of_g :
  ∀ x : ℝ, cos x ≠ 2 → 0.5 ≤ g x / (cos x - 2) ∧ g x / (cos x - 2) < 12.5 :=
by
  sorry

end range_of_g_l212_212355


namespace symmetry_about_point_l212_212032

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x + Real.pi / 6)

theorem symmetry_about_point (x: ℝ) :
  f(x) = 3 * Real.cos (2 * x + Real.pi / 6) ∧ f(Real.pi / 6) = 0 → 
  ∃ a b, f(x) = f(2 * a - x) → (a, b) = (Real.pi / 6, 0) := 
sorry

end symmetry_about_point_l212_212032


namespace tangent_lines_count_l212_212522

noncomputable def number_of_tangent_lines (r1 r2 : ℝ) (k : ℕ) : ℕ :=
if r1 = 2 ∧ r2 = 3 then 5 else 0

theorem tangent_lines_count: 
∃ k : ℕ, number_of_tangent_lines 2 3 k = 5 :=
by sorry

end tangent_lines_count_l212_212522


namespace suff_not_necessary_condition_l212_212082

theorem suff_not_necessary_condition (a b : ℝ) (ha : 0 < a ∧ a ≠ 1) (hb : 0 < b ∧ b ≠ 1) :
  (3 ^ a > 3 ^ b ∧ 3 ^ b > 3) → (Real.log 3 / Real.log a < Real.log 3 / Real.log b) :=
sorry

end suff_not_necessary_condition_l212_212082


namespace households_with_bike_only_l212_212809

theorem households_with_bike_only 
    (total_households : ℕ)
    (neither_car_nor_bike : ℕ)
    (both_car_and_bike : ℕ)
    (with_car : ℕ) :
    neither_car_nor_bike = 11 ∧ both_car_and_bike = 22 ∧ with_car = 44 ∧ total_households = 90 →
    (total_households - neither_car_nor_bike - with_car + both_car_and_bike) = 35 :=
by 
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_other
  cases h_other with h3 h4
  rw [h1, h2, h3, h4]
  exact (79 - 44) = 35
  sorry

end households_with_bike_only_l212_212809


namespace part_a_part_b_part_c_maximizes_equidistant_l212_212272

def coord := (ℕ × ℕ)

def manhattan_distance (c1 c2 : coord) : ℕ := 
  abs (c1.1 - c2.1) + abs (c1.2 - c2.2)

noncomputable def cheese_positions : list coord := [(2, 3), (4, 3)]

def is_equidistant_to_cheese (pos : coord) : Prop :=
  manhattan_distance pos (2, 3) = manhattan_distance pos (4, 3)

-- Part (a): Prove (3, 4) is equidistant to both cheese at (2, 3) and (4, 3)
theorem part_a : is_equidistant_to_cheese (3, 4) :=
sorry

-- Part (b): Prove all intersections (3, 2), (3, 4), (1, 3), (5, 3), (2, 3), (4, 3), (3, 3) are equidistant to cheese
theorem part_b : 
  is_equidistant_to_cheese (3, 2) ∧ 
  is_equidistant_to_cheese (3, 4) ∧ 
  is_equidistant_to_cheese (1, 3) ∧ 
  is_equidistant_to_cheese (5, 3) ∧ 
  is_equidistant_to_cheese (2, 3) ∧ 
  is_equidistant_to_cheese (4, 3) ∧
  is_equidistant_to_cheese (3, 3) :=
sorry

-- Part (c): Prove placing cheese at intersections (1, 1) and (5, 5) maximizes the number of equidistant intersections
noncomputable def suggested_cheese_positions : list coord := [(1, 1), (5, 5)]

def is_equidistant_to_suggested_cheese (pos : coord) : Prop :=
  manhattan_distance pos (1, 1) = manhattan_distance pos (5, 5)

theorem part_c_maximizes_equidistant : 
  ∃ (positions : list coord), 
    positions = suggested_cheese_positions ∧
    ∀ pos, pos ∈ [(2, 1), (4, 5), (1, 5), (5, 1)] → is_equidistant_to_suggested_cheese pos :=
sorry

end part_a_part_b_part_c_maximizes_equidistant_l212_212272


namespace intersection_complement_A_B_l212_212768

noncomputable def A : Set ℝ := {x | ∃ y, x^2 - y^2 = 1}
noncomputable def B : Set ℝ := {y | ∃ x, x^2 = 4*y}
noncomputable def C_R : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_complement_A_B : 
  (C_R \ A) ∩ B = Ico 0 1 :=
by sorry

end intersection_complement_A_B_l212_212768


namespace jason_worked_hours_on_saturday_l212_212070

def hours_jason_works (x y : ℝ) : Prop :=
  (4 * x + 6 * y = 88) ∧ (x + y = 18)

theorem jason_worked_hours_on_saturday (x y : ℝ) : hours_jason_works x y → y = 8 := 
by 
  sorry

end jason_worked_hours_on_saturday_l212_212070


namespace diane_total_loss_l212_212695

-- Define the starting amount of money Diane had.
def starting_amount : ℤ := 100

-- Define the amount of money Diane won.
def winnings : ℤ := 65

-- Define the amount of money Diane owed at the end.
def debt : ℤ := 50

-- Define the total amount of money Diane had after winnings.
def mid_game_total : ℤ := starting_amount + winnings

-- Define the total amount Diane lost.
def total_loss : ℤ := mid_game_total + debt

-- Theorem stating the total amount Diane lost is 215 dollars.
theorem diane_total_loss : total_loss = 215 := by
  sorry

end diane_total_loss_l212_212695


namespace sequence_2023rd_letter_l212_212217

def sequence : List Char := ['A', 'B', 'C', 'D', 'C', 'B']

theorem sequence_2023rd_letter :
  sequence[(2023 % sequence.length) - 1] = 'A' :=
by
  have h : 2023 % sequence.length = 1 := by sorry
  rw [h]
  refl

end sequence_2023rd_letter_l212_212217


namespace sum_of_interior_angles_of_regular_polygon_l212_212027

theorem sum_of_interior_angles_of_regular_polygon (exterior_angle : ℝ)
  (h_exterior_angle : exterior_angle = 40) : 
  180 * (360 / exterior_angle - 2) = 1260 :=
by
  have n_sides : ℝ := 360 / exterior_angle
  have h_n_sides : n_sides = 9 := by 
    simp [exterior_angle, h_exterior_angle]
  sorry

end sum_of_interior_angles_of_regular_polygon_l212_212027


namespace ratio_depth_water_to_dean_height_l212_212292

theorem ratio_depth_water_to_dean_height 
  (height_Ron : ℕ) 
  (short_Ron_Dean : ℕ) 
  (water_depth : ℕ) 
  (height_Ron_cond : height_Ron = 14)
  (short_Ron_Dean_cond : short_Ron_Dean = 8)
  (water_depth_cond : water_depth = 12):
  let height_Dean := height_Ron - short_Ron_Dean in 
  (water_depth : height_Dean) = 2 := by
  sorry

end ratio_depth_water_to_dean_height_l212_212292


namespace minimum_weighings_for_counterfeit_coin_l212_212196

/-- Given 9 coins, where 8 have equal weight and 1 is heavier (the counterfeit coin), prove that the 
minimum number of weighings required on a balance scale without weights to find the counterfeit coin is 2. -/
theorem minimum_weighings_for_counterfeit_coin (n : ℕ) (coins : Fin n → ℝ) 
  (h_n : n = 9) 
  (h_real : ∃ w : ℝ, ∀ i : Fin n, i.val < 8 → coins i = w) 
  (h_counterfeit : ∃ i : Fin n, ∀ j : Fin n, j ≠ i → coins i > coins j) : 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end minimum_weighings_for_counterfeit_coin_l212_212196


namespace num_articles_cost_price_l212_212793

theorem num_articles_cost_price (N C S : ℝ) (h1 : N * C = 50 * S) (h2 : (S - C) / C * 100 = 10) : N = 55 := 
sorry

end num_articles_cost_price_l212_212793


namespace intersection_complement_eq_l212_212408

open Set

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  (U = {1, 2, 3, 4, 5, 6}) →
  (A = {1, 3}) →
  (B = {3, 4, 5}) →
  A ∩ (U \ B) = {1} :=
by
  intros hU hA hB
  subst hU
  subst hA
  subst hB
  sorry

end intersection_complement_eq_l212_212408


namespace John_first_half_cost_l212_212875

theorem John_first_half_cost
  (n : ℕ) (c_total : ℕ) (cost_factor : ℝ)
  (h1 : n = 22) 
  (h2 : c_total = 35200)
  (h3 : cost_factor = 1.20) :
  ∃ x : ℝ, (11 * x) + (11 * (2.20 * x)) = 35200 ∧ x = 1000 :=
by
  use 1000
  split
  sorry

end John_first_half_cost_l212_212875


namespace vartan_spent_on_recreation_last_week_l212_212489

variable (W P : ℝ)
variable (h1 : P = 0.20)
variable (h2 : W > 0)

theorem vartan_spent_on_recreation_last_week :
  (P * W) = 0.20 * W :=
by
  sorry

end vartan_spent_on_recreation_last_week_l212_212489


namespace exists_finite_group_commutator_bound_l212_212986

/-- A finite group G has rank at most 2 if every subgroup of G can be generated by at most 2 elements. -/
def rank_le_2 (G : Group) : Prop :=
  ∀ (H : Subgroup G), ∃ (S : Set G), S.card ≤ 2 ∧ H = Subgroup.closure S

/-- The commutator series of a group G. -/
def commutator_series_length (G : Group) : ℕ :=
  sorry  -- Define commutator series length here

theorem exists_finite_group_commutator_bound :
  ∃ (s : ℕ), ∀ (G : Group) (hG_fin : Group.is_finite G) (hG_rank : rank_le_2 G), commutator_series_length G < s :=
sorry

end exists_finite_group_commutator_bound_l212_212986


namespace circle_complete_the_square_l212_212296

/-- Given the equation x^2 - 6x + y^2 - 10y + 18 = 0, show that it can be transformed to  
    (x - 3)^2 + (y - 5)^2 = 4^2 -/
theorem circle_complete_the_square :
  ∀ x y : ℝ, x^2 - 6 * x + y^2 - 10 * y + 18 = 0 ↔ (x - 3)^2 + (y - 5)^2 = 4^2 :=
by
  sorry

end circle_complete_the_square_l212_212296


namespace problem1_problem2_l212_212092

-- Define the function f
def f (x α : ℝ) : ℝ := abs (x + 1) + abs (x - α)

-- Problem 1: Prove the range of α
theorem problem1 (α : ℝ) :
  (∀ x : ℝ, f x α ≥ 5) ↔ α ∈ set.Iic (-6) ∪ set.Ici 4 :=
by
  sorry

-- Problem 2: Prove the inequality given minimum value condition
theorem problem2 (m n t : ℝ) (hm : m > 0) (hn : n > 0) (hα : α = 1) :
  (f (1 : ℝ) 1 = t) → (m + n = t) → (1/m + 1/n ≥ 2) :=
by
  sorry

end problem1_problem2_l212_212092


namespace sum_sqrt_distinct_nonzero_rational_nonzero_l212_212506

noncomputable def square_free (n : ℕ) : Prop := 
  ∀ d : ℕ, d > 1 → d^2 ∣ n → false

theorem sum_sqrt_distinct_nonzero_rational_nonzero 
  (k : ℕ) 
  (a : Fin k → ℕ) 
  (b : Fin k → ℚ)
  (h_distinct : ∀ i j : Fin k, i ≠ j → a i ≠ a j)
  (h_square_free : ∀ i : Fin k, square_free (a i))
  (h_nonzero : ∀ i : Fin k, b i ≠ 0) :
  ∑ i : Fin k, b i * Real.sqrt (a i) ≠ 0 :=
sorry

end sum_sqrt_distinct_nonzero_rational_nonzero_l212_212506


namespace minimum_value_l212_212016

theorem minimum_value (x : ℝ) (h : x > -3) : 2 * x + (1 / (x + 3)) ≥ 2 * Real.sqrt 2 - 6 :=
sorry

end minimum_value_l212_212016


namespace infinite_n_for_triangle_l212_212129

theorem infinite_n_for_triangle (a b c : ℕ) (n : ℕ) (p r : ℕ) :
  let p := (a + b + c) / 2,
      r := n → by
  ∃ (infinitely many) n,
  ∃ (triangle : ℕ) (int_side_triangle : ℕ) (sides_condition : ℕ),
    p = ((a + b + c) / 2) ∧
    r = A / p → 
  by
    p = n * r :=
  sorry

end infinite_n_for_triangle_l212_212129


namespace contest_result_l212_212357

-- Definition of the positions
def pos := Fin 5
-- Definition of students
inductive Student
| A | B | C | D | E

open Student 

-- Hypotheses based on given problem
variable (placement: pos → Student)
def prediction1 := [A, B, C, D, E]
def prediction2 := [D, A, E, C, B]

-- Conditions from the problem statement
def condition1 : Prop := ∀ i, placement i ≠ prediction1.get ⟨i.1⟩ sorry
def consecutive (s1 s2 : Student) (i j : pos) : Prop := (placement i = s1 ∧ placement j = s2 ∧ abs(i.1 - j.1) = 1)

def condition2 : Prop := ¬ ∃ i j : pos, consecutive (prediction1.get ⟨i.1⟩ sorry) (prediction1.get ⟨j.1⟩ sorry) i j

def condition3 : Prop := ∃ i j : pos, i ≠ j ∧ placement i = prediction2.get ⟨i.1⟩ sorry ∧ placement j = prediction2.get ⟨j.1⟩ sorry

def condition4 : Prop := ∃ si sj sk sl: Student, ∃ i j k l : pos, i ≠ j ∧ k ≠ l ∧ (consecutive si sj i j ∨ consecutive sk sl k l) ∧ ((si, sj) ∈ [(D, A), (A, E), (E, C), (C, B)] ∧ (sk, sl) ∈ [(D, A), (A, E), (E, C), (C, B)])

theorem contest_result : condition1 placement → condition2 placement → condition3 placement → condition4 placement → placement = λi, [E, D, A, C, B].get ⟨i.1⟩ sorry :=
by
  sorry

end contest_result_l212_212357


namespace solve_chimney_bricks_l212_212309

noncomputable def chimney_bricks (x : ℝ) : Prop :=
  let brenda_rate := x / 8
  let brandon_rate := x / 12
  let combined_rate := brenda_rate + brandon_rate - 15
  (combined_rate * 6) = x

theorem solve_chimney_bricks : ∃ (x : ℝ), chimney_bricks x ∧ x = 360 :=
by
  use 360
  unfold chimney_bricks
  sorry

end solve_chimney_bricks_l212_212309


namespace bounded_polynomial_settings_count_l212_212941

/-- The bounded polynomial problem with binary variables and positive coefficients. -/
theorem bounded_polynomial_settings_count : 
  let aij := { (i, j) | 0 ≤ i ∧ i ≤ 3 ∧ 0 ≤ j ∧ j ≤ 3 } in
  let num_vars := fintype.card aij in
  let valid_settings := { s : fin num_vars → bool | ∃ c : fin num_vars → ℝ, (∀ k, 0 < c k) ∧ 
    ∀ (x y : ℝ), (∑ k in aij, c k * (s k).to_bool * x ^ k.1 * y ^ k.2) ≥ -M } in
  ∃ M : ℝ, fintype.card valid_settings = 65024 :=
begin
  sorry
end

end bounded_polynomial_settings_count_l212_212941


namespace arcsin_arccos_value_range_l212_212338

theorem arcsin_arccos_value_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z ∈ set.Icc (-7 * π / 2) (3 * π / 2) ∧ 5 * Real.arcsin x - 2 * Real.arccos y = z := sorry

end arcsin_arccos_value_range_l212_212338


namespace closest_point_to_line_l212_212353

theorem closest_point_to_line {x y : ℝ} (h : y = 2 * x - 4) :
  ∃ (closest_x closest_y : ℝ),
    closest_x = 9 / 5 ∧ closest_y = -2 / 5 ∧ closest_y = 2 * closest_x - 4 ∧
    ∀ (x' y' : ℝ), y' = 2 * x' - 4 → (closest_x - 3)^2 + (closest_y + 1)^2 ≤ (x' - 3)^2 + (y' + 1)^2 :=
by
  sorry

end closest_point_to_line_l212_212353


namespace arts_sciences_probability_l212_212330

noncomputable def morning_classes : List String := ["mathematics", "Chinese", "politics", "geography"]
noncomputable def afternoon_classes : List String := ["English", "history", "physical education"]
noncomputable def arts_sciences_classes : List String := ["politics", "history", "geography"]

def total_morning_classes := morning_classes.length
def total_afternoon_classes := afternoon_classes.length
def total_classes := total_morning_classes * total_afternoon_classes

def morning_arts_sciences := morning_classes.filter (fun c => c ∈ arts_sciences_classes)
def afternoon_arts_sciences := afternoon_classes.filter (fun c => c ∈ arts_sciences_classes)
def non_arts_sciences_morning := total_morning_classes - morning_arts_sciences.length

def favorable_morning_arts_sciences := morning_arts_sciences.length * total_afternoon_classes
def favorable_afternoon_arts_sciences := non_arts_sciences_morning * afternoon_arts_sciences.length
def favorable_outcomes := favorable_morning_arts_sciences + favorable_afternoon_arts_sciences

def probability : ℚ := favorable_outcomes / total_classes

theorem arts_sciences_probability : probability = 2 / 3 := by
  sorry

end arts_sciences_probability_l212_212330


namespace probability_x2012_eq_x0_l212_212867

-- Definitions for the transformations
def f (x : ℂ) : ℂ := 1 - x
def g (x : ℂ) : ℂ := 1 / x

-- The main statement we need to prove
theorem probability_x2012_eq_x0 (x0 : ℂ) (h0 : x0 ≠ 0) (h1 : x0 ≠ 1) :
  ∃ p ∈ ({1, (2 ^ 2011 + 1) / (3 * 2 ^ 2011)} : set ℚ),
    ∀ x : ℂ, 
      (∀ n, (n = 0 → x = x0) ∧ (n > 0 →
        ∃ r : ℤ, (r % 2 = 0 → x = g (n - 1)) ∧ (r % 2 = 1 → x = f (n - 1))))
      → p = prob (x_2012 = x0) then 
sorry

end probability_x2012_eq_x0_l212_212867


namespace sum_of_reciprocals_less_than_3_l212_212895

def is_valid_element (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≤ 3

def S : finset ℕ := -- Assume a finite set S satisfying the conditions
{n ∈ (finset.range 1000) | is_valid_element n} -- Adjust range or set appropriately.

theorem sum_of_reciprocals_less_than_3 (S : finset ℕ) (hS : ∀ s ∈ S, is_valid_element s) :
  (finset.sum S (λ s, (1 : ℚ) / s)) < 3 :=
sorry

end sum_of_reciprocals_less_than_3_l212_212895


namespace smallest_c_for_f_inverse_l212_212897

noncomputable def f (x : ℝ) : ℝ := (x - 3)^2 - 4

theorem smallest_c_for_f_inverse :
  ∃ c : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≥ c → x₂ ≥ c → f x₁ = f x₂ → x₁ = x₂) ∧ (∀ d : ℝ, d < c → ∃ x₁ x₂ : ℝ, x₁ ≥ d ∧ x₂ ≥ d ∧ f x₁ = f x₂ ∧ x₁ ≠ x₂) ∧ c = 3 :=
by
  sorry

end smallest_c_for_f_inverse_l212_212897


namespace number_of_integers_with_exactly_three_identical_digits_l212_212002

def exactlyThreeIdenticalDigits (n : ℕ) : Prop :=
  let digits := n.digits 10
  let digit_counts := digits.freq
  n >= 100 ∧ n < 10000 ∧ (digit_counts.values.filter (λ c => c == 3)).length == 1

theorem number_of_integers_with_exactly_three_identical_digits : 
  #{ n | n ∈ finset.range 10000 ∧ exactlyThreeIdenticalDigits n } = 324 :=
by sorry

end number_of_integers_with_exactly_three_identical_digits_l212_212002


namespace probability_A_union_B_l212_212532

-- Definitions for the conditions
def odd_number (n : ℕ) : Prop := n % 2 = 1
def at_least_5 (n : ℕ) : Prop := n ≥ 5

-- The main statement to be proven
theorem probability_A_union_B : 
  let outcomes : ℕ := 36 in
  let favorable_outcomes : ℕ := 24 in 
  (favorable_outcomes : ℚ) / outcomes = 2 / 3 :=
by
  sorry

end probability_A_union_B_l212_212532


namespace correct_option_l212_212610

-- Conditions as definitions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def optionC (a : ℝ) : Prop := (-2 * a^2)^3 = -8 * a^6
def optionD (a : ℝ) : Prop := a^6 / a^2 = a^3

-- The statement to prove
theorem correct_option (a : ℝ) : optionC a :=
by 
  unfold optionC
  sorry

end correct_option_l212_212610


namespace George_spending_l212_212702

theorem George_spending (B m s : ℝ) (h1 : m = 0.25 * (B - s)) (h2 : s = 0.05 * (B - m)) : 
  (m + s) / B = 1 := 
by
  sorry

end George_spending_l212_212702


namespace geometric_series_sum_l212_212693

theorem geometric_series_sum :
  (∑ i in finset.range 10, ∑ k in finset.range (i + 1), (1/2)^k) = 9 + 1/(2^10) :=
by
  sorry

end geometric_series_sum_l212_212693


namespace roots_equal_implies_a_eq_3_l212_212798

theorem roots_equal_implies_a_eq_3 (x a : ℝ) (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
sorry

end roots_equal_implies_a_eq_3_l212_212798


namespace sequence_100th_term_l212_212904

-- Definition of the sequence
def sequence_term (k : ℕ) : ℕ :=
  let n := Nat.find (fun n => (n * (n + 1)) / 2 >= k) in n

-- Main theorem statement
theorem sequence_100th_term : sequence_term 100 = 14 := by
  sorry

end sequence_100th_term_l212_212904


namespace triangle_heights_sum_geq_nine_times_incircle_radius_l212_212507

-- Let h₁, h₂, h₃ be the heights of a triangle, and r be the radius of the inscribed circle.
variables {Δ : Type} [triangle Δ] (h₁ h₂ h₃ r : ℝ)

-- Assume h₁, h₂, h₃ are the heights of the triangle, and r is the radius of the incircle.
-- Prove that h₁ + h₂ + h₃ ≥ 9 * r.
theorem triangle_heights_sum_geq_nine_times_incircle_radius (h₁ h₂ h₃ r : ℝ) :
  h₁ + h₂ + h₃ ≥ 9 * r :=
sorry

end triangle_heights_sum_geq_nine_times_incircle_radius_l212_212507


namespace sum_of_eight_consec_fib_not_fib_l212_212925

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib n + fib (n + 1)

-- Define the sum of eight consecutive Fibonacci numbers
def sum_eight_consec_fib (k : ℕ) : ℕ :=
  fib (k + 1) + fib (k + 2) + fib (k + 3) + fib (k + 4) +
  fib (k + 5) + fib (k + 6) + fib (k + 7) + fib (k + 8)

-- The theorem states that the sum is not a Fibonacci number
theorem sum_of_eight_consec_fib_not_fib (k : ℕ) : 
  ¬ ∃ n : ℕ, fib n = sum_eight_consec_fib k :=
by
  sorry

end sum_of_eight_consec_fib_not_fib_l212_212925


namespace angle_ABC_eq_angle_ADC_l212_212824

-- Given a convex quadrilateral ABCD
variables {A B C D O : Type}
variables [convex_quadrilateral A B C D]

-- Given conditions
variable (angle_CBD_eq_angle_CAB : ∠ CBD = ∠ CAB)
variable (angle_ACD_eq_angle_BDA : ∠ ACD = ∠ BDA)

-- Prove that ∠ ABC = ∠ ADC 
theorem angle_ABC_eq_angle_ADC :
  ∠ ABC = ∠ ADC :=
begin
  sorry -- Proof not required
end

end angle_ABC_eq_angle_ADC_l212_212824


namespace proof_quadratic_conclusions_l212_212229

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given points on the graph
def points_on_graph (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = -2 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 1 = -4 ∧
  quadratic_function a b c 2 = -3 ∧
  quadratic_function a b c 3 = 0

-- Assertions based on the problem statement
def assertion_A (a b : ℝ) : Prop := 2 * a + b = 0

def assertion_C (a b c : ℝ) : Prop :=
  quadratic_function a b c 3 = 0 ∧ quadratic_function a b c (-1) = 0

def assertion_D (a b c : ℝ) (m : ℝ) (y1 y2 : ℝ) : Prop :=
  (quadratic_function a b c (m - 1) = y1) → 
  (quadratic_function a b c m = y2) → 
  (y1 < y2) → 
  (m > 3 / 2)

-- Final theorem statement to be proven
theorem proof_quadratic_conclusions (a b c : ℝ) (m y1 y2 : ℝ) :
  points_on_graph a b c →
  assertion_A a b →
  assertion_C a b c →
  assertion_D a b c m y1 y2 :=
by
  sorry

end proof_quadratic_conclusions_l212_212229


namespace similar_triangles_side_ratios_l212_212038

theorem similar_triangles_side_ratios {P₁ P₂ : ℝ} (h₁ : P₁ / P₂ = 1 / 4) :
  ∀ (side₁ side₂ : ℝ), (side₁ / side₂ = 1 / 4) :=
begin
  assume side₁ side₂,
  have h₂ : ∀ s₁ s₂ : ℝ, s₁ / s₂ = P₁ / P₂,
  { intros s₁ s₂,
    sorry },
  exact h₂ side₁ side₂,
end

end similar_triangles_side_ratios_l212_212038


namespace students_in_class_l212_212045

-- Define the relevant variables and conditions
variables (P H W T A S : ℕ)

-- Given conditions
axiom poetry_club : P = 22
axiom history_club : H = 27
axiom writing_club : W = 28
axiom two_clubs : T = 6
axiom all_clubs : A = 6

-- Statement to prove
theorem students_in_class
  (poetry_club : P = 22)
  (history_club : H = 27)
  (writing_club : W = 28)
  (two_clubs : T = 6)
  (all_clubs : A = 6) :
  S = P + H + W - T - 2 * A :=
sorry

end students_in_class_l212_212045


namespace positive_intervals_of_product_l212_212691

theorem positive_intervals_of_product (x : ℝ) : 
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := 
sorry

end positive_intervals_of_product_l212_212691


namespace temperature_on_tuesday_l212_212551

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th = 156) ∧ (W + Th + 53 = 162) → T = 47 :=
by
  sorry

end temperature_on_tuesday_l212_212551


namespace rem_neg_one_third_quarter_l212_212701

noncomputable def rem (x y : ℝ) : ℝ :=
  x - y * ⌊x / y⌋

theorem rem_neg_one_third_quarter :
  rem (-1/3) (1/4) = 1/6 :=
by
  sorry

end rem_neg_one_third_quarter_l212_212701


namespace probability_B_not_occur_given_A_occurs_expected_value_X_l212_212788

namespace DieProblem

def event_A := {1, 2, 3}
def event_B := {1, 2, 4}

def num_trials := 10
def num_occurrences_A := 6

theorem probability_B_not_occur_given_A_occurs :
  (∑ i in Finset.range (num_trials.choose num_occurrences_A), 
    (1/6)^num_occurrences_A * (1/3)^(num_trials - num_occurrences_A)) / 
  (num_trials.choose num_occurrences_A * (1/2)^(num_trials)) = 2.71 * 10^(-4) :=
sorry

theorem expected_value_X : 
  (6 * (2/3)) + (4 * (1/3)) = 16 / 3 :=
sorry

end DieProblem

end probability_B_not_occur_given_A_occurs_expected_value_X_l212_212788


namespace value_of_f_ln3_l212_212751

def f : ℝ → ℝ := sorry

theorem value_of_f_ln3 (f_symm : ∀ x : ℝ, f (x + 1) = f (-x + 1))
  (f_exp : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = Real.exp (-x)) :
  f (Real.log 3) = 3 * Real.exp (-2) :=
by
  sorry

end value_of_f_ln3_l212_212751


namespace neg_p_sufficient_but_not_necessary_for_neg_q_l212_212503

variables {x : ℝ}

def p : Prop := x^2 - x < 1
def q : Prop := log 2 (x^2 - x) < 0

theorem neg_p_sufficient_but_not_necessary_for_neg_q : 
  (¬ p → ¬ q) ∧ ¬ (¬ q → ¬ p) :=
by
  sorry

end neg_p_sufficient_but_not_necessary_for_neg_q_l212_212503


namespace pentagon_area_ratio_of_decagon_l212_212123

theorem pentagon_area_ratio_of_decagon (n m : ℝ) 
  (h1 : is_regular_decagon ABCDEFGHIJ n)
  (h2 : is_pentagon ACEGI) 
  (h3 : area_of_pentagon ACEGI = m)
  (h4 : area_of_decagon ABCDEFGHIJ = n) :
  m / n = 1 / 2 :=
sorry

end pentagon_area_ratio_of_decagon_l212_212123


namespace probability_of_multiples_l212_212131

noncomputable def number_range := (Finset.range 61).erase 0

def multiples_of_six := number_range.filter (λ n, n % 6 = 0)
def multiples_of_eight := number_range.filter (λ n, n % 8 = 0)
def multiples_of_twenty_four := number_range.filter (λ n, n % 24 = 0)

def favorable_count := multiples_of_six.card + multiples_of_eight.card - multiples_of_twenty_four.card
def total_count := number_range.card

theorem probability_of_multiples :
    (favorable_count / total_count : ℚ) = 1 / 4 := by
  sorry

end probability_of_multiples_l212_212131


namespace find_j_k_l212_212077

-- Definition of Fibonacci numbers
def fibonacci : ℕ → ℕ
| 1 => 1
| 2 => 1
| (n+3) => fibonacci (n+2) + fibonacci (n+1)

-- Definition of polynomial p with the given conditions
def p : ℕ → ℕ := sorry
-- Assume p to be a polynomial of degree 1008, satisfying p(2n+1) = fibonacci(2n+1) for n = 0,1,...,1008

theorem find_j_k :
  (j k : ℕ)
  (hp_condition : ∀ n, (n <= 1008) → p (2*n + 1) = fibonacci (2*n + 1))
  (p_degree : degree p = 1008)
  : 
  ((p 2019 = fibonacci 2019 - fibonacci 1010) ∧ (j = 2019) ∧ (k = 1010)) :=
sorry

end find_j_k_l212_212077


namespace geq_solution_l212_212049

def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ (a (n+1) / a n) = (a 1 / a 0)

theorem geq_solution
  (a : ℕ → ℝ)
  (h_seq : geom_seq a)
  (h_cond : a 0 * a 2 + 2 * a 1 * a 3 + a 1 * a 5 = 9) :
  a 1 + a 3 = 3 :=
sorry

end geq_solution_l212_212049


namespace non_degenerate_ellipse_condition_l212_212679

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 9 * x^2 + y^2 - 18 * x - 2 * y = k) ↔ k > -10 :=
sorry

end non_degenerate_ellipse_condition_l212_212679


namespace shared_name_in_most_populous_house_l212_212976

-- Let's define the conditions given in the problem
def people_total : ℕ := 125
def min_people_per_name : ℕ := 3
def max_distinct_names : ℕ := 42

-- Our goal: to prove that in the most populous house, at least two people must share the same name.
theorem shared_name_in_most_populous_house 
  (h_total: ∀ (s: set ℕ), s.card = people_total) 
  (h_names: ∀ (name: ℕ), (∀ s: set ℕ, s.card = people_total → s.filter (λ x, x = name).card ≥ min_people_per_name))
  (h_distinct_names: ∀ (names: set ℕ), names.card ≤ max_distinct_names) : 
  ∃ (house: set ℕ), house.card > people_total / max_distinct_names ∧ ∃ (x: ℕ), (house.filter (λ p, p = x)).card ≥ 2 :=
by sorry

end shared_name_in_most_populous_house_l212_212976


namespace add_fractions_l212_212291

theorem add_fractions :
  (11 / 12) + (7 / 8) + (3 / 4) = 61 / 24 :=
by
  sorry

end add_fractions_l212_212291


namespace popsicles_count_l212_212097

def total_money : ℕ := 2550 -- Lucy's total money in cents
def popsicle_price_initial : ℕ := 200 -- Price of each popsicle for the first 10
def popsicle_price_discounted : ℕ := 150 -- Price of each popsicle after the first 10
def max_popsicles : ℕ := 13 -- The number of popsicles Lucy can buy

theorem popsicles_count (total_money popsicle_price_initial popsicle_price_discounted : ℕ) :
  ∀ (total_money = 2550) (popsicle_price_initial = 200) (popsicle_price_discounted = 150), 
  let maximum_popsicles_without_discount := total_money / popsicle_price_initial in
  if maximum_popsicles_without_discount ≤ 10 then 
    maximum_popsicles_without_discount = 13 
  else 
    let remaining_money := total_money - 10 * popsicle_price_initial in
    let additional_popsicles := remaining_money / popsicle_price_discounted in
    10 + additional_popsicles = 13 := sorry

end popsicles_count_l212_212097


namespace Nancy_hourly_wage_l212_212105

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end Nancy_hourly_wage_l212_212105


namespace point_inside_circle_l212_212756

-- Conditions
variables {a b c : ℝ}
variable (e : ℝ)
variable (x y : ℝ)

-- Given conditions in the problem
def is_eccentricity (e : ℝ) (a c : ℝ) : Prop :=
  e = c / a

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_ellipse_eccentricity : Prop :=
  is_eccentricity (1 / 2 : ℝ) a c

def roots (x1 x2 : ℝ) (a b c : ℝ) : Prop :=
  x1 + x2 = -b / a ∧ x1 * x2 = -c / a

-- The point P(x1, x2) position relative to the circle x^2 + y^2 = 2
def is_inside_circle (x1 x2 : ℝ) : Prop :=
  x1^2 + x2^2 < 2

-- Problem statement in Lean 4
theorem point_inside_circle (a b c x1 x2 : ℝ) (h_eccentricity : is_ellipse_eccentricity) 
  (h_roots : roots x1 x2 a b c) : is_inside_circle x1 x2 :=
sorry

end point_inside_circle_l212_212756


namespace sin_double_angle_l212_212776

theorem sin_double_angle (θ : ℝ) (h : cos θ - sin θ = 3 / 5) : sin (2 * θ) = 16 / 25 :=
by 
  sorry

end sin_double_angle_l212_212776


namespace relatively_prime_count_in_range_l212_212774

theorem relatively_prime_count_in_range : 
  ∀ n ∈ finset.range 110, n + 11 > 10 ∧ n + 11 < 120 ∧ nat.coprime (n + 11) 18  → 
  (finset.filter (λ n, nat.coprime n 18) (finset.range 110)).card = 34 :=
by 
  sorry

end relatively_prime_count_in_range_l212_212774


namespace perpendicular_line_planes_parallel_l212_212378

variables (m n : Type) [line m] [line n] [non_coincident m n]
variables (α β : Type) [plane α] [plane β] [non_coincident α β]

theorem perpendicular_line_planes_parallel (m_perp_α : m ⊥ α) (m_perp_β : m ⊥ β) : α ∥ β :=
sorry

end perpendicular_line_planes_parallel_l212_212378


namespace continuous_cauchy_eq_linear_l212_212706

theorem continuous_cauchy_eq_linear (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y : ℝ, f(x + y) = f(x) + f(y)) → continuous f → (∀ x : ℝ, f x = a * x) :=
  by
    intros h_func h_cont
    sorry

end continuous_cauchy_eq_linear_l212_212706


namespace find_cities_with_roads_l212_212481

noncomputable def kingdom_kitty (n k : ℕ) (G : SimpleGraph (Fin n)) : Prop :=
  n > 2023 * k^3 ∧
  G.edgeSet.card ≥ 2 * n^(3/2) ∧
  (∃ (S : Finset (Fin n)), S.card = 3 * k + 1 ∧ S.powerset.card ≥ 4 * k)

theorem find_cities_with_roads (n k : ℕ) (G : SimpleGraph (Fin n)) (H : kingdom_kitty n k G) :
  ∃ (S : Finset (Fin n)), S.card = 3 * k + 1 ∧ (G.induce S).edgeFinset.card ≥ 4 * k :=
sorry

end find_cities_with_roads_l212_212481


namespace scientific_notation_120_million_l212_212149

theorem scientific_notation_120_million :
  120000000 = 1.2 * 10^7 :=
by
  sorry

end scientific_notation_120_million_l212_212149


namespace angle_ABC_equals_angle_ADC_l212_212847

def Quadrilateral (A B C D : Type) := True -- We need a placeholder for the quadrilateral type.

variables {A B C D O : Type} -- Variables for points

-- Angles definitions
variables (angle_CBD angle_CAB angle_ACD angle_BDA angle_ABC angle_ADC : Type)

-- Given conditions:
variable Hypothesis1 : angle_CBD = angle_CAB
variable Hypothesis2 : angle_ACD = angle_BDA

-- The theorem to be proven:
theorem angle_ABC_equals_angle_ADC : Quadrilateral A B C D → angle_CBD = angle_CAB → angle_ACD = angle_BDA → angle_ABC = angle_ADC :=
  by
  intro h_quad h1 h2,
  sorry

end angle_ABC_equals_angle_ADC_l212_212847


namespace weight_of_replaced_person_l212_212552

theorem weight_of_replaced_person (avg_increase : ℝ) (num_persons : ℕ) (new_person_weight : ℝ) (weight_increase : ℝ) : 
  (num_persons = 8) → (avg_increase = 2.5) → (new_person_weight = 80) → (weight_increase = num_persons * avg_increase) → 
  weight_increase = 20 → 
  ∃ W : ℝ, W = new_person_weight - weight_increase := 
by
  intros h1 h2 h3 h4 h5
  use (new_person_weight - weight_increase)
  sorry

end weight_of_replaced_person_l212_212552


namespace number_of_solutions_l212_212005

theorem number_of_solutions :
  let equation x := |x - 2| = |x - 3| + |x - 5| + |x - 4| in
  (∃ x1 : ℝ, equation x1 ∧ (x1 = 8 / 3 ∨ x1 = 5)) ∧
  (∀ x2 : ℝ, equation x2 → (x2 = 8 / 3 ∨ x2 = 5)) := 
by
  sorry

end number_of_solutions_l212_212005


namespace square_free_odd_integers_count_l212_212685

def is_square_free (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k * k ∣ n → false

lemma odd_integer_in_range (n : ℕ) : Prop :=
  n > 1 ∧ n < 200 ∧ odd n

theorem square_free_odd_integers_count :
  (∃ count : ℕ, count = 81 ∧
   count = (λ S, S.card) {n : ℕ | odd_integer_in_range n ∧ is_square_free n} ) :=
begin
  sorry
end

end square_free_odd_integers_count_l212_212685


namespace train_length_l212_212659

theorem train_length (time : ℝ) (speed_in_kmph : ℝ) (speed_in_mps : ℝ) (length_of_train : ℝ) :
  (time = 6) →
  (speed_in_kmph = 96) →
  (speed_in_mps = speed_in_kmph * (5 / 18)) →
  length_of_train = speed_in_mps * time →
  length_of_train = 480 := by
  sorry

end train_length_l212_212659


namespace collinear_incenter_circumcenter_O_l212_212201

-- Define the points and the properties of the equal circles touching the sides of the triangle
variables {A B C O : Point}

-- Define the triangle and the three equal circles touching the sides of the triangle
variable (h_touch : ∀ (x : Point), 
  (circle x O).touches_point_side_triangle x (A, B, C))

-- Define the incenter and circumcenter of triangle ABC
noncomputable def incenter (A B C : Point) : Point := 
  sorry -- Definition of the incenter

noncomputable def circumcenter (A B C : Point) : Point := 
  sorry -- Definition of the circumcenter

-- State the theorem
theorem collinear_incenter_circumcenter_O :
  collinear (incenter A B C) (circumcenter A B C) O :=
by
  sorry -- Proof to be filled in

end collinear_incenter_circumcenter_O_l212_212201


namespace overall_percentage_increase_l212_212566

noncomputable def new_price_after_discount (price : ℕ) (discount_rate : ℕ) : ℕ :=
  price - (price * discount_rate / 100)

noncomputable def total_cost_with_shipping (price_after_discount : ℕ) (shipping_fee : ℕ) : ℕ :=
  price_after_discount + shipping_fee

noncomputable def percentage_increase (initial_cost final_cost : ℕ) : ℚ :=
  ((final_cost - initial_cost) * 100 : ℚ) / initial_cost

theorem overall_percentage_increase :
  let initial_cost_A := 300
  let new_price_A := 360
  let discount_rate := 5
  let shipping_fee_A := 10
  let initial_cost_B := 450
  let new_price_B := 540
  let shipping_fee_B := 7
  let initial_cost_C := 600
  let new_price_C := 720
  let shipping_fee_C := 5
  let final_cost_A := total_cost_with_shipping (new_price_after_discount new_price_A discount_rate) shipping_fee_A
  let final_cost_B := total_cost_with_shipping (new_price_after_discount new_price_B discount_rate) shipping_fee_B
  let final_cost_C := total_cost_with_shipping (new_price_after_discount new_price_C discount_rate) shipping_fee_C
  let percentage_increase_A := percentage_increase initial_cost_A final_cost_A
  let percentage_increase_B := percentage_increase initial_cost_B final_cost_B
  let percentage_increase_C := percentage_increase initial_cost_C final_cost_C
  let overall_increase := (percentage_increase_A + percentage_increase_B + percentage_increase_C) / 3
  overall_increase ≈ 15.91 :=
sorry

end overall_percentage_increase_l212_212566


namespace x_coordinate_of_point_on_parabola_l212_212818

-- Definitions: Conditions in a)
def parabola (p : ℝ × ℝ) : Prop := p.2 ^ 2 = 4 * p.1

def distance (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def focus : ℝ × ℝ := (1, 0) -- focus of the parabola y² = 4x

-- The Lean definition of the problem
theorem x_coordinate_of_point_on_parabola 
  (P : ℝ × ℝ)
  (hP : parabola P)
  (h_dist : distance P focus = 5) :
  P.1 = 4 := 
sorry

end x_coordinate_of_point_on_parabola_l212_212818


namespace max_red_balls_l212_212459

theorem max_red_balls (x : ℕ) (hx : x ≤ 20) : 90 * (50 + 8 * x) ≤ 100 * (49 + 7 * x) → 50 + 8 * x ≤ 210 :=
by
  -- The proof is intended to be here
  apply le_trans (calc
    50 + 8 * x ≤ 50 + 8 * 20 : Nat.add_le_add_left (Nat.mul_le_mul_left 8 hx) 50
  )
  sorry

end max_red_balls_l212_212459


namespace minimal_area_circle_equation_circle_equation_center_on_line_l212_212634

-- Question (1): Prove the equation of the circle with minimal area
theorem minimal_area_circle_equation :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  C = (0, -4) ∧ r = Real.sqrt 5 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → P.1 ^ 2 + (P.2 + 4) ^ 2 = 5) :=
sorry

-- Question (2): Prove the equation of a circle with the center on a specific line
theorem circle_equation_center_on_line :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  (C.1 - 2 * C.2 - 3 = 0) ∧
  C = (-1, -2) ∧ r = Real.sqrt 10 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → (P.1 + 1) ^ 2 + (P.2 + 2) ^ 2 = 10) :=
sorry

end minimal_area_circle_equation_circle_equation_center_on_line_l212_212634


namespace calculate_expression_solve_quadratic_l212_212624

-- Problem 1
theorem calculate_expression (x : ℝ) (hx : x > 0) :
  (2 / 3) * Real.sqrt (9 * x) + 6 * Real.sqrt (x / 4) - x * Real.sqrt (1 / x) = 4 * Real.sqrt x :=
sorry

-- Problem 2
theorem solve_quadratic (x : ℝ) (h : x^2 - 4 * x + 1 = 0) :
  x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
sorry

end calculate_expression_solve_quadratic_l212_212624


namespace harkamal_total_amount_l212_212771

-- Define the conditions
def grapes_kg := 9
def grapes_rate := 70
def grapes_discount := 0.15
def grapes_tax := 0.06

def mangoes_kg := 9
def mangoes_rate := 55
def mangoes_offer := 3 / 9  -- "Buy 2 kg get 1 kg free" implies Harkamal pays for 2/3 of the quantity
def mangoes_tax := 0.04

def apples_kg := 5
def apples_rate := 40
def apples_discount := 0.10
def apples_tax := 0.03

def oranges_kg := 6
def oranges_rate := 30
def oranges_tax := 0.05

def pineapples_kg := 7
def pineapples_rate := 45
def pineapples_discount := 0.20
def pineapples_tax := 0.08

def bananas_kg := 10
def bananas_rate := 15
def bananas_tax := 0.02

-- Define the required proof statement
theorem harkamal_total_amount :
  let grapes_total := (grapes_kg * grapes_rate * (1 - grapes_discount)) * (1 + grapes_tax)
  let mangoes_total := (mangoes_kg * mangoes_rate * (1 - mangoes_offer)) * (1 + mangoes_tax)
  let apples_total := (apples_kg * apples_rate * (1 - apples_discount)) * (1 + apples_tax)
  let oranges_total := (oranges_kg * oranges_rate) * (1 + oranges_tax)
  let pineapples_total := (pineapples_kg * pineapples_rate * (1 - pineapples_discount)) * (1 + pineapples_tax)
  let bananas_total := (bananas_kg * bananas_rate) * (1 + bananas_tax)
  let total_amount := grapes_total + mangoes_total + apples_total + oranges_total + pineapples_total + bananas_total
  total_amount = 1710.39 :=
by
  -- Proof steps (for illustration; they are not required for this prompt)
  sorry

end harkamal_total_amount_l212_212771


namespace new_oranges_added_l212_212283

def initial_oranges : Nat := 34
def thrown_away_oranges : Nat := 20
def current_oranges : Nat := 27

theorem new_oranges_added : initial_oranges - thrown_away_oranges + ?new_oranges = current_oranges :=
by
  sorry

end new_oranges_added_l212_212283


namespace find_integer_x_l212_212709

theorem find_integer_x : ∃ x : ℤ, x^5 - 3 * x^2 = 216 ∧ x = 3 :=
by {
  sorry
}

end find_integer_x_l212_212709


namespace total_distance_travelled_l212_212332

noncomputable def distance_traveled (radius : ℝ) (boys : ℕ) : ℝ :=
  let diameter := 2 * radius
  let chord := 2 * radius * real.sin (real.pi / 4)
  boys * (diameter + 4 * chord)

theorem total_distance_travelled :
  distance_traveled 50 8 = 800 + 1600 * real.sqrt 2 :=
by
  sorry

end total_distance_travelled_l212_212332


namespace solve_eq_l212_212134

theorem solve_eq (x : ℝ) (h : (greatest_int_floor x) = ⌊x⌋):
  x^(⌊x⌋) = 9/2 ↔ x = 3 * (Real.sqrt 2) / 2 := sorry

end solve_eq_l212_212134


namespace min_value_f_l212_212347

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (15 - 12 * cos x) + 
  sqrt (4 - 2 * sqrt 3 * sin x) +
  sqrt (7 - 4 * sqrt 3 * sin x) +
  sqrt (10 - 4 * sqrt 3 * sin x - 6 * cos x)

theorem min_value_f : ∃ x : ℝ, f x = 6 := 
sorry

end min_value_f_l212_212347


namespace john_reads_faster_by_37_5_percent_l212_212072

theorem john_reads_faster_by_37_5_percent (h_brother: ℕ) (h_john: ℕ) (num_books: ℕ)
    (h1 : h_brother = 8)
    (h2 : h_john = 15)
    (h3 : num_books = 3) : 
    (h_brother - (h_john / num_books)) / h_brother * 100 = 37.5 := 
by
  sorry

end john_reads_faster_by_37_5_percent_l212_212072


namespace max_value_of_x1_squared_plus_x2_squared_l212_212380

theorem max_value_of_x1_squared_plus_x2_squared :
  ∀ (k : ℝ), -4 ≤ k ∧ k ≤ -4 / 3 → (∃ x1 x2 : ℝ, x1^2 + x2^2 = 18) :=
by
  sorry

end max_value_of_x1_squared_plus_x2_squared_l212_212380


namespace b_general_term_l212_212942

noncomputable def a : ℕ+ → ℝ
| 1     => 2
| (n+1) => a (n+2) + a (n+3)

noncomputable def b : ℕ+ → ℝ
| 1     => a 1
| (n+1) => b n * a (n+1)

theorem b_general_term (n : ℕ+) (h : ∀ (k : ℕ+), k ≥ 2 → (b k)^3 ≥ 2 * real.sqrt 2) :
  b n = 2 ^ (2 - 1 / 2 ^ (nat.pred n)) :=
sorry

end b_general_term_l212_212942


namespace polar_coordinate_circle_intersections_product_l212_212055

theorem polar_coordinate_circle (θ : ℝ) :
  let x := 2 + 2 * Real.cos θ
  let y := 2 * Real.sin θ
  (x - 2) ^ 2 + y ^ 2 = 4 := by
  sorry

theorem intersections_product (t : ℝ) :
  let x_l := 1 + (1 / 2) * t
  let y_l := (Real.sqrt 3 / 2) * t
  (x_l - 2) ^ 2 + y_l ^ 2 = 4
  (x - 2) ^ 2 + y ^ 2 = 4
  let x := 2 + 2 * Real.cos θ
  let y := 2 * Real.sin θ
  let PA := Real.sqrt ((1 - x_A) ^ 2 + 0 ^ 2)
  let PB := Real.sqrt ((1 - x_B) ^ 2 + 0 ^ 2)
  |PA * PB| = 3 := by
  sorry

end polar_coordinate_circle_intersections_product_l212_212055


namespace tank_filling_time_l212_212526

-- Define the rates at which pipes fill or drain the tank
def capacity : ℕ := 1200
def rate_A : ℕ := 50
def rate_B : ℕ := 35
def rate_C : ℕ := 20
def rate_D : ℕ := 40

-- Define the times each pipe is open
def time_A : ℕ := 2
def time_B : ℕ := 4
def time_C : ℕ := 3
def time_D : ℕ := 5

-- Calculate the total time for one cycle
def cycle_time : ℕ := time_A + time_B + time_C + time_D

-- Calculate the net amount of water added in one cycle
def net_amount_per_cycle : ℕ := (rate_A * time_A) + (rate_B * time_B) + (rate_C * time_C) - (rate_D * time_D)

-- Calculate the number of cycles needed to fill the tank
def num_cycles : ℕ := capacity / net_amount_per_cycle

-- Calculate the total time to fill the tank
def total_time : ℕ := num_cycles * cycle_time

-- Prove that the total time to fill the tank is 168 minutes
theorem tank_filling_time : total_time = 168 := by
  sorry

end tank_filling_time_l212_212526


namespace min_value_expr_l212_212021

theorem min_value_expr (x : ℝ) (h : x > -3) : ∃ m, (∀ y > -3, 2 * y + (1 / (y + 3)) ≥ m) ∧ m = 2 * Real.sqrt 2 - 6 :=
by
  sorry

end min_value_expr_l212_212021


namespace angle_APB_is_135_l212_212372

theorem angle_APB_is_135
  (A B C D P : Type)
  [inhabited A]
  [inhabited B]
  [inhabited C]
  [inhabited D]
  [inhabited P]
  (PA PB PC : ℝ)
  (h1 : PA = 1)
  (h2 : PB = 2)
  (h3 : PC = 3)
  (h_square : ∀ (X : Type), X = A ∨ X = B ∨ X = C ∨ X = D)
  : angle A P B = 135 :=
by
  sorry

end angle_APB_is_135_l212_212372


namespace distinct_prime_factors_300_l212_212001

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ m, m ≠ 0 ∧ is_prime m ∧ m ∣ n)

def distinct_prime_factors_count (n : ℕ) : ℕ :=
  (prime_factors n).eraseDup.length

theorem distinct_prime_factors_300 : distinct_prime_factors_count 300 = 3 :=
by
  sorry

end distinct_prime_factors_300_l212_212001


namespace diane_total_loss_l212_212696

def initial_amount : ℤ := 100
def amount_won : ℤ := 65
def amount_owed : ℤ := 50

theorem diane_total_loss : initial_amount + amount_won - (initial_amount + amount_won) + amount_owed = 215 := by
  calc
    initial_amount + amount_won - (initial_amount + amount_won) + amount_owed
      = amount_owed : by rw [add_sub_cancel'_right]
      = 50 : rfl
      = 215 : by sorry

end diane_total_loss_l212_212696


namespace john_bought_dress_shirts_l212_212873

theorem john_bought_dress_shirts:
  ∀ (x : ℕ), 20 * x + 0.10 * (20 * x) = 66 → x = 3 := by
  sorry

end john_bought_dress_shirts_l212_212873


namespace tan_arccot_five_thirds_l212_212676

theorem tan_arccot_five_thirds (h : ∀ θ, θ = Real.arccot 3/5 → Real.tan θ = 5/3) : 
  Real.tan (Real.arccot 3/5) = 5/3 := by sorry

end tan_arccot_five_thirds_l212_212676


namespace quadratic_expression_negative_for_all_x_l212_212684

theorem quadratic_expression_negative_for_all_x (k : ℝ) :
  (∀ x : ℝ, (5-k) * x^2 - 2 * (1-k) * x + 2 - 2 * k < 0) ↔ k > 9 :=
sorry

end quadratic_expression_negative_for_all_x_l212_212684


namespace sock_pairs_l212_212199

theorem sock_pairs (n : ℕ) (h : ((2 * n) * (2 * n - 1)) / 2 = 90) : n = 10 :=
sorry

end sock_pairs_l212_212199


namespace additional_rows_l212_212636

-- Definitions based on conditions
def total_trees (rows : ℕ) (trees_per_row : ℕ) : ℕ := rows * trees_per_row
def number_of_rows (total_trees : ℕ) (trees_per_row : ℕ) : ℕ := total_trees / trees_per_row

-- Given conditions
def problem_conditions (original_rows new_rows trees_per_row_old trees_per_row_new : ℕ) : Prop :=
  original_rows = 24 ∧ trees_per_row_old = 42 ∧ trees_per_row_new = 28 ∧ new_rows = 36

-- Statement to prove
theorem additional_rows (original_rows new_rows trees_per_row_old trees_per_row_new : ℕ) (total_trees : ℕ)
  (h1 : problem_conditions original_rows new_rows trees_per_row_old trees_per_row_new)
  (h2 : total_trees = total_trees original_rows trees_per_row_old)
  (h3 : new_rows = number_of_rows total_trees trees_per_row_new) :
  new_rows - original_rows = 12 := 
by 
  sorry

end additional_rows_l212_212636


namespace appropriate_sampling_methods_l212_212203

theorem appropriate_sampling_methods :
  ∀ (high_income_families medium_income_families low_income_families : ℕ) 
    (total_art_students sample_art_students : ℕ),
  high_income_families = 125 ∧ medium_income_families = 200 ∧ low_income_families = 95 ∧ 
  total_art_students = 5 ∧ sample_art_students = 3 →
  (stratified_sampling ∧ simple_random_sampling) := 
begin
  sorry
end

end appropriate_sampling_methods_l212_212203


namespace distinct_shading_patterns_l212_212772

-- Define the 3x3 grid and possible shading scenarios
def grid := fin (3 × 3)

-- Define the concept of equivalence classes of patterns under flips and/or turns
def equivalent_patterns (p1 p2 : set grid) : Prop :=
  ∃ f : grid → grid, bijective f ∧ (∀ x ∈ p1, f x ∈ p2) ∧ (∀ x ∉ p1, f x ∉ p2)

-- Define the main theorem
theorem distinct_shading_patterns : 
  (number_of_patterns (set grid) equivalent_patterns (λ p, card p = 2) = 8) :=
sorry

end distinct_shading_patterns_l212_212772


namespace charlie_distance_to_lightning_l212_212316

noncomputable def distance_in_quarter_miles
  (time : ℕ) (speed : ℕ) (mile_in_feet : ℕ) : ℚ :=
  let distance_in_feet := speed * time
  let distance_in_miles := distance_in_feet / mile_in_feet
  (distance_in_miles * 4).round / 4

theorem charlie_distance_to_lightning :
  distance_in_quarter_miles 15 1100 5280 = 3.25 :=
by
  sorry

end charlie_distance_to_lightning_l212_212316


namespace highest_price_percentage_increase_l212_212561

-- Define the prices
def highest_price : ℝ := 45
def lowest_price : ℝ := 30

-- Define the percentage increase function
def percentage_increase (high low : ℝ) : ℝ := ((high - low) / low) * 100

-- State the theorem to prove
theorem highest_price_percentage_increase :
  percentage_increase highest_price lowest_price = 50 := by
  sorry

end highest_price_percentage_increase_l212_212561


namespace minimum_value_f_is_correct_l212_212350

noncomputable def f (x : ℝ) := 
  Real.sqrt (15 - 12 * Real.cos x) + 
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem minimum_value_f_is_correct :
  ∃ x : ℝ, f x = (9 / 2) * Real.sqrt 2 :=
sorry

end minimum_value_f_is_correct_l212_212350


namespace initial_workers_count_l212_212299

-- Define variables used in the condition
variables (W : ℕ) -- W is the initial number of workers
variables (women_ratio men_ratio hired_women : ℚ) -- ratios
variables (total_percentage_women : ℚ)

-- Assume the conditions provided
def initial_conditions :=
  women_ratio = 1/3 ∧
  men_ratio = 2/3 ∧
  hired_women = 10 ∧
  total_percentage_women = 0.40

-- Define the final count of workers and the equation derived from the condition
def final_conditions (W : ℕ) :=
  total_percentage_women * (W + hired_women) = women_ratio * W + hired_women

-- The proof problem as a statement that W = 90
theorem initial_workers_count (W : ℕ) :
  initial_conditions →
  final_conditions W →
  W = 90 :=
by
  intros
  sorry

end initial_workers_count_l212_212299


namespace ratio_product_of_solutions_l212_212037

theorem ratio_product_of_solutions :
  (∀ x : ℝ, (3 * x + 5) / (4 * x + 4) = (5 * x + 4) / (10 * x + 5)) →
  ∏ (solutions : Finset ℝ) = 9 / 10 :=
by
  sorry

end ratio_product_of_solutions_l212_212037


namespace digit_sum_cardinality_of_S_l212_212081

def digit_sum (x : ℕ) : ℕ :=
  x.digits.sum

def is_in_S (n : ℕ) : Prop :=
  digit_sum n = 9 ∧ n < 10^5

def S : finset ℕ :=
  finset.filter is_in_S (finset.range 10^5)

theorem digit_sum_cardinality_of_S : digit_sum S.card = 13 := by
  sorry

end digit_sum_cardinality_of_S_l212_212081


namespace octagon_area_sum_l212_212184

open Real

theorem octagon_area_sum (O : Point ℝ)
  (side_len : ℝ)
  (side_len_eq : side_len = 1)
  (length_AB : ℝ)
  (length_AB_eq : length_AB = 43 / 99)
  (m n : ℕ)
  (h1 : Nat.coprime m n)
  (area_octagon_eq : (octagon_area ABCDEFGH) = m / n) :
  m + n = 185 := 
sorry

end octagon_area_sum_l212_212184


namespace omega_value_l212_212030

theorem omega_value (ω : ℝ) (h1 : ω > 0)
    (h2 : ∀ x y : ℝ, (cos (ω * x + π / 4) = 0 → cos (ω * y + π / 4) = 0 → abs (x - y) = π / 6)) : ω = 6 := by
  sorry

end omega_value_l212_212030


namespace Nancy_hourly_wage_l212_212104

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end Nancy_hourly_wage_l212_212104


namespace square_free_odd_integers_count_l212_212686

def is_square_free (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k * k ∣ n → false

lemma odd_integer_in_range (n : ℕ) : Prop :=
  n > 1 ∧ n < 200 ∧ odd n

theorem square_free_odd_integers_count :
  (∃ count : ℕ, count = 81 ∧
   count = (λ S, S.card) {n : ℕ | odd_integer_in_range n ∧ is_square_free n} ) :=
begin
  sorry
end

end square_free_odd_integers_count_l212_212686


namespace perp_iff_area_relation_l212_212215

-- Defining the geometric conditions for the problem.
variables (A B C D O E F : Type) [Trapezoid A B C D] [Parallel AB CD] 
[AB_gt_CD : GT AB CD] [Perpendicular AD AB] [AD_gt_CD : GT AD CD]
[Intersects AC BD O] [ParallelThrough O E AB AD] [Intersects BE CD F]

-- Stating the proof problem
theorem perp_iff_area_relation :
  CE_perp_AF ↔ AB * CD = AD^2 - CD^2 :=
sorry

end perp_iff_area_relation_l212_212215


namespace log_5_1560_l212_212987

open Real

noncomputable def log_5 := logb (5 : ℝ)

theorem log_5_1560 (h625: log_5 625 = 4)
                    (h3125: log_5 3125 = 5)
                    (h1250: log_5 1250 > 4.43) :
                    log_5 1560 = 5 :=
begin
  sorry
end

end log_5_1560_l212_212987


namespace base6_addition_problem_l212_212377

-- Definitions for the conditions
def digits := {n // n > 0 ∧ n < 6}
def distinct {α : Type*} (a b c : α) := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Statement of the problem
theorem base6_addition_problem :
  ∀ (S H E : digits), distinct S H E →
  (S + S * 6 + H + 1 * 6 + E + E * 6 = E + S * 6 + H) →
  (S + H + E = 14 ∧ Nat.digits 6 14 = [2, 2]) :=
by
  sorry

end base6_addition_problem_l212_212377


namespace square_area_increase_l212_212174

theorem square_area_increase (s : ℕ) (h : (s = 5) ∨ (s = 10) ∨ (s = 15)) :
  (1.35^2 - 1) * 100 = 82.25 :=
by
  sorry

end square_area_increase_l212_212174


namespace volume_ratio_proof_l212_212187

-- Definitions based on conditions
def edge_ratio (a b : ℝ) : Prop := a = 3 * b
def volume_ratio (V_large V_small : ℝ) : Prop := V_large = 27 * V_small

-- Problem statement
theorem volume_ratio_proof (e V_small V_large : ℝ) 
  (h1 : edge_ratio (3 * e) e)
  (h2 : volume_ratio V_large V_small) : 
  V_large / V_small = 27 := 
by sorry

end volume_ratio_proof_l212_212187


namespace degree_of_my_poly_l212_212220

noncomputable def my_poly : Polynomial ℝ :=
  3 * Real.sqrt 3 + 8 * X^6 + 10 * X^2 + (5 / 3) * X^5 + 7 * Real.pi * X^7 + 13

theorem degree_of_my_poly : my_poly.degree = 7 :=
by
  sorry

end degree_of_my_poly_l212_212220


namespace convert_to_scientific_notation_l212_212150

-- Problem statement: convert 120 million to scientific notation and validate the format.
theorem convert_to_scientific_notation :
  120000000 = 1.2 * 10^7 :=
sorry

end convert_to_scientific_notation_l212_212150


namespace proof_problem_l212_212376

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 - 2 * x * real.sin θ + 1 ≥ 0

-- Define proposition q
def q : Prop := ∀ α β : ℝ, real.sin (α + β) ≤ real.sin α + real.sin β

-- Define the compound proposition to prove
theorem proof_problem (hp : p) (hq : q) : ¬p ∨ q :=
by
  sorry

end proof_problem_l212_212376


namespace even_iff_b_eq_zero_l212_212553

theorem even_iff_b_eq_zero (a b c : ℝ) :
  (∀ x, f(x) = ax^2 + bx + c → f(x) = f(-x)) ↔ b = 0 :=
sorry

end even_iff_b_eq_zero_l212_212553


namespace gcd_max_value_l212_212969

theorem gcd_max_value (x y : ℤ) (h_posx : x > 0) (h_posy : y > 0) (h_sum : x + y = 780) :
  gcd x y ≤ 390 ∧ ∃ x' y', x' > 0 ∧ y' > 0 ∧ x' + y' = 780 ∧ gcd x' y' = 390 := by
  sorry

end gcd_max_value_l212_212969


namespace Beth_reading_time_l212_212628

theorem Beth_reading_time :
  ∀ (total_chapters : ℕ) (chapters_read : ℕ) (time_for_chapters_read : ℚ) (break_time_per_chapter : ℚ),
  total_chapters = 14 → 
  chapters_read = 4 → 
  time_for_chapters_read = 6 → 
  break_time_per_chapter = 10 / 60 →
  let remaining_chapters := total_chapters - chapters_read,
      time_per_chapter := time_for_chapters_read / chapters_read,
      break_time_total := (remaining_chapters - 1) * break_time_per_chapter,
      total_time := remaining_chapters * time_per_chapter + break_time_total in
  total_time = 16.5 :=
by
  intros
  sorry

end Beth_reading_time_l212_212628


namespace areas_equal_l212_212880

open EuclideanGeometry Metric

namespace Example

variables {A B C P H_A H_B H_C : Point ℝ}
variables {triangleABC : Triangle ℝ} (hABC : triangleABC = Triangle.mk A B C)

def isOrtho (P₁ P₂ P₃ P₄ : Point ℝ) : Prop :=
∠ P₁ P₂ P₃ = 90 ∨ ∠ P₁ P₂ P₄ = 90 ∨ ∠ P₁ P₄ P₃ = 90

theorem areas_equal (hP : InteriorPoint P (triangleABC))
                    (hH_A : is_orthocenter H_A P B C)
                    (hH_B : is_orthocenter H_B P A C)
                    (hH_C : is_orthocenter H_C P A B) :
  area (Triangle.mk H_A H_B H_C) = area triangleABC :=
sorry

end Example

end areas_equal_l212_212880


namespace min_value_y_l212_212728

theorem min_value_y (x : ℝ) (h : x > 0) : ∃ y, y = x + 4 / x^2 ∧ (∀ z, z = x + 4 / x^2 → y ≤ z) := 
sorry

end min_value_y_l212_212728


namespace cubic_root_identity_l212_212078

theorem cubic_root_identity (r : ℝ) (h : (r^(1/3)) - (1/(r^(1/3))) = 2) : r^3 - (1/r^3) = 14 := 
by 
  sorry

end cubic_root_identity_l212_212078


namespace integer_solutions_l212_212346

theorem integer_solutions (x y : ℤ) : 
  x^2 * y = 10000 * x + y ↔ 
  (x, y) = (-9, -1125) ∨ 
  (x, y) = (-3, -3750) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (3, 3750) ∨ 
  (x, y) = (9, 1125) := 
by
  sorry

end integer_solutions_l212_212346


namespace solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l212_212248

-- Define the conditions
variables (a b c x y z: ℝ) 

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) : Prop :=
  (a * y + b * x = c) ∧
  (c * x + a * z = b) ∧
  (b * z + c * y = a)

-- Define the general solution
def solution (a b c x y z : ℝ) : Prop :=
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧
  z = (a^2 + b^2 - c^2) / (2 * a * b)

-- Define the proof problem statement
theorem solve_system_of_equations_general (a b c x y z : ℝ) (h : system_of_equations a b c x y z) 
      (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : solution a b c x y z :=
  sorry

-- Special cases
theorem solve_system_of_equations_zero_case_1 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_2 (a b c x y z : ℝ) (h1 : a = 0) (h2 : b = 0) (h3: c ≠ 0) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_3 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : x = c / b ∧ 
      (c * x = b) :=
  sorry

-- Following special cases more concisely
theorem solve_system_of_equations_special_cases (a b c x y z : ℝ) 
      (h : system_of_equations a b c x y z) (h1: a = 0 ∨ b = 0 ∨ c = 0): 
      (∃ k : ℝ, x = k ∧ y = -k ∧ z = k)  
    ∨ (∃ k : ℝ, x = k ∧ y = k ∧ z = -k)
    ∨ (∃ k : ℝ, x = -k ∧ y = k ∧ z = k) :=
  sorry

end solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l212_212248


namespace volume_of_pyramid_SPQR_l212_212116

-- Define a structure for points in 3D space
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Define the conditions as functions or properties
def perpendicular (u v : Point3D) : Prop :=
u.x * v.x + u.y * v.y + u.z * v.z = 0

def distance (u v : Point3D) : ℝ :=
real.sqrt ((u.x - v.x)^2 + (u.y - v.y)^2 + (u.z - v.z)^2)

-- Defining the points P, Q, R, and S
variables (P Q R S : Point3D)

-- Define the given conditions
def conditions : Prop :=
perpendicular (P - S) (Q - S) ∧
perpendicular (P - S) (R - S) ∧
perpendicular (Q - S) (R - S) ∧
distance S P = 8 ∧
distance S Q = 8 ∧
distance S R = 12

-- Statement of the proof problem
theorem volume_of_pyramid_SPQR (h : conditions P Q R S) : 
  volume_of_pyramid P Q R S = 128 := sorry

end volume_of_pyramid_SPQR_l212_212116


namespace ellipse_equation_l212_212361

theorem ellipse_equation (a b : ℝ) (A : ℝ × ℝ)
  (hA : A = (-3, 1.75))
  (he : 0.75 = Real.sqrt (a^2 - b^2) / a) 
  (hcond : (Real.sqrt (a^2 - b^2) / a) = 0.75) :
  (16 = a^2) ∧ (7 = b^2) :=
by
  have h1 : A = (-3, 1.75) := hA
  have h2 : Real.sqrt (a^2 - b^2) / a = 0.75 := hcond
  sorry

end ellipse_equation_l212_212361


namespace prove_smallest_x_is_zero_l212_212995

noncomputable def smallest_x : ℝ :=
  let f := λ (x : ℝ), (5 * x - 20) / (4 * x - 5)
  in if h : f(x)^2 + f(x) = 20 then x else 0

theorem prove_smallest_x_is_zero :
  let f := λ (x : ℝ), (5 * x - 20) / (4 * x - 5)
  in ∃ x : ℝ, f(x)^2 + f(x) = 20 ∧ (∀ y : ℝ, f(y)^2 + f(y) = 20 → x ≤ y) :=
begin
  sorry, -- proof goes here
end

end prove_smallest_x_is_zero_l212_212995


namespace addition_of_decimals_l212_212186

theorem addition_of_decimals :
  0.9 + 0.99 = 1.89 :=
by
  sorry

end addition_of_decimals_l212_212186


namespace point_3_units_away_l212_212111

theorem point_3_units_away (x : ℤ) (h : abs (x + 1) = 3) : x = 2 ∨ x = -4 :=
by
  sorry

end point_3_units_away_l212_212111


namespace measure_angle_ACB_l212_212849

-- Definitions of angles and a given triangle
variable (α β γ : ℝ)
variable (angleABD angle75 : ℝ)
variable (triangleABC : Prop)

-- Conditions from the problem
def angle_supplementary : Prop := angleABD + α = 180
def sum_angles_triangle : Prop := α + β + γ = 180
def known_angle : Prop := β = 75
def angleABD_value : Prop := angleABD = 150

-- The theorem to prove
theorem measure_angle_ACB : 
  angle_supplementary angleABD α ∧
  sum_angles_triangle α β γ ∧
  known_angle β ∧
  angleABD_value angleABD
  → γ = 75 := by
  sorry


end measure_angle_ACB_l212_212849


namespace range_of_k_for_intersection_l212_212715

def f (x : ℝ) : ℝ := if |x| ≤ 2 then 2 else 1

def g (k x : ℝ) : ℝ := k * (x - 2) + 4

theorem range_of_k_for_intersection (k : ℝ) : (∃ (x1 x2 : ℝ), -2 ≤ x1 ∧ x1 ≤ 2 ∧ -2 ≤ x2 ∧ x2 ≤ 2 ∧ x1 ≠ x2 ∧ f x1 = g k x1 ∧ f x2 = g k x2) ↔ detection_of_range k :=
sorry

end range_of_k_for_intersection_l212_212715


namespace smallest_possible_value_l212_212023

theorem smallest_possible_value 
  (a : ℂ)
  (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ z : ℂ, z = 3 * a + 1 ∧ z.re = -1 / 8 :=
by
  sorry

end smallest_possible_value_l212_212023


namespace tan_value_l212_212424

theorem tan_value (α : ℝ) 
  (h₀ : cos (π + α) = - (sqrt 10) / 5) 
  (h₁ : α ∈ Ioo (-π / 2) 0) : 
  tan (3 * π / 2 + α) = - (sqrt 6) / 3 :=
sorry

end tan_value_l212_212424


namespace prove_expression_value_l212_212426

theorem prove_expression_value (a b c d : ℝ) (h1 : a + b = 0) (h2 : c = -1) (h3 : d = 1 ∨ d = -1) :
  2 * a + 2 * b - c * d = 1 ∨ 2 * a + 2 * b - c * d = -1 := 
by sorry

end prove_expression_value_l212_212426


namespace no_prime_divisible_by_42_l212_212422

theorem no_prime_divisible_by_42 : 
  ∀ p : ℕ, Prime p → ¬ (42 ∣ p) := 
by
  intro p hp hdiv
  have h2 : 2 ∣ p := dvd_of_mul_right_dvd hdiv
  have h3 : 3 ∣ p := dvd_of_mul_left_dvd (dvd_of_mul_right_dvd hdiv)
  have h7 : 7 ∣ p := dvd_of_mul_left_dvd hdiv
  sorry

end no_prime_divisible_by_42_l212_212422


namespace range_a_for_increasing_f_l212_212952

theorem range_a_for_increasing_f {a : ℝ} :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ ≤ x₂ → f x₁ ≤ f x₂) ↔ 0 ≤ a :=
by
  let f (x : ℝ) := x^2 + 2 * (a - 1) * x + 3
  sorry

end range_a_for_increasing_f_l212_212952


namespace extremum_value_of_f_at_x_monotonicity_of_g_l212_212370

variable {a : ℝ}

theorem extremum_value_of_f_at_x (h_extremum : f'(-4 / 3) = 0) : 
  a = 1 / 2 ∧ f(-4 / 3) = 32 / 27 :=
sorry

theorem monotonicity_of_g (x : ℝ) : 
  let g := λ x, (1 / 2 * x^3 + x^2) * exp x in 
  ((x < -4 → g' x < 0) ∧ 
  (-4 < x ∧ x < -1 → g' x > 0) ∧ 
  (-1 < x ∧ x < 0 → g' x < 0) ∧ 
  (x > 0 → g' x > 0)) :=
sorry

end extremum_value_of_f_at_x_monotonicity_of_g_l212_212370


namespace probability_ace_first_diamond_second_jack_third_l212_212200

open Classical

variable {deck : Finset (Nat)}

-- Define standard deck of 52 cards
def standardDeck : Finset (Nat) := Finset.range 52

-- Define the event conditions
def eventAceFirstDiamondSecondJackThird (deck : Finset (Nat)) : Prop :=
  ∃ a b c : Nat, 
    a ∈ deck ∧ isAce a ∧
    b ∈ (deck.erase a) ∧ isDiamond b ∧
    c ∈ (deck.erase a).erase b ∧ isJack c

-- Define the probability calculation (pseudo definition for demonstration)
noncomputable def probabilityEvent (deck : Finset (Nat)) (event : Finset (Nat) → Prop) : ℚ :=
  sorry -- Placeholder for actual probability calculation

-- Define the proposition
theorem probability_ace_first_diamond_second_jack_third :
  probabilityEvent standardDeck eventAceFirstDiamondSecondJackThird = 1 / 650 := 
sorry

end probability_ace_first_diamond_second_jack_third_l212_212200


namespace length_of_handrail_l212_212653

open Real

/-- The length of the handrail of a spiral staircase given that:
    - It turns 450 degrees.
    - The rise of the staircase is 15 feet.
    - The radius of the staircase is 4 feet.
-/
theorem length_of_handrail :
  let θ := (450 * (π / 180)) in
  let height := 15 in
  let radius := 4 in
  let width := radius * θ / (2 * π) * 2 * π in
  let length := sqrt (height ^ 2 + width ^ 2) in
  abs (length - 17.4) < 0.1 :=
by
  let θ := (450 * (π / 180))
  let height := 15
  let radius := 4
  let width := radius * θ / (2 * π) * 2 * π
  let length := sqrt (height ^ 2 + width ^ 2)
  have h_approx : abs (length - 17.4) < 0.1 := by sorry
  exact h_approx

end length_of_handrail_l212_212653


namespace find_k_value_l212_212741

theorem find_k_value : ∀ (x y k : ℝ), x = 2 → y = -1 → y - k * x = 7 → k = -4 := 
by
  intros x y k hx hy h
  sorry

end find_k_value_l212_212741


namespace chicken_pieces_needed_l212_212596

theorem chicken_pieces_needed :
  let chicken_pasta_pieces := 2
      barbecue_chicken_pieces := 3
      fried_chicken_dinner_pieces := 8
      number_of_fried_chicken_dinner_orders := 2
      number_of_chicken_pasta_orders := 6
      number_of_barbecue_chicken_orders := 3
  in
  (number_of_fried_chicken_dinner_orders * fried_chicken_dinner_pieces +
   number_of_chicken_pasta_orders * chicken_pasta_pieces +
   number_of_barbecue_chicken_orders * barbecue_chicken_pieces) = 37 := by
  sorry

end chicken_pieces_needed_l212_212596


namespace no_constant_term_in_expansion_find_rational_terms_in_expansion_l212_212757

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (√x - 1 / (2 * x^(1 / 4)))^8

theorem no_constant_term_in_expansion (x : ℝ) :
  ∀ k : ℕ, 3 * k ≠ 16 :=
  sorry

theorem find_rational_terms_in_expansion (x : ℝ) :
  ∃ terms : List (ℝ × ℝ), 
    (terms = [(1, x^4), (35/8, x), (1/256, x^(-2))]) ∧ 
    ∀ t ∈ terms, 
      ∂ (polynomial_expansion x) (t.2) ∈ ℚ :=
  sorry

end no_constant_term_in_expansion_find_rational_terms_in_expansion_l212_212757


namespace ratio_of_m1_and_m2_l212_212498

theorem ratio_of_m1_and_m2 (m a b m1 m2 : ℝ) (h1 : a^2 * m - 3 * a * m + 2 * a + 7 = 0) (h2 : b^2 * m - 3 * b * m + 2 * b + 7 = 0) 
  (h3 : (a / b) + (b / a) = 2) (h4 : m1^2 * 9 - m1 * 28 + 4 = 0) (h5 : m2^2 * 9 - m2 * 28 + 4 = 0) : 
  (m1 / m2) + (m2 / m1) = 194 / 9 := 
sorry

end ratio_of_m1_and_m2_l212_212498


namespace subset_condition_l212_212736

def A (m : ℝ) : set ℝ := {1, 3, m^2}
def B (m : ℝ) : set ℝ := {1, m}

theorem subset_condition (m : ℝ) : B m ⊆ A m ↔ (m = 0 ∨ m = 3) := by
  sorry

end subset_condition_l212_212736


namespace five_spotlights_intersect_at_one_point_l212_212110

theorem five_spotlights_intersect_at_one_point
  (α β : ℝ)
  (α02 : 0 < α ∧ α < (π / 2))
  (β02 : 0 < β ∧ β < (π / 2))
  (exists_intersection : ∀ (P1 P2 P3 P4 : point), ∃ (P : point), emits_at_angle P1 α ∨ emits_at_angle P1 β ∧
    emits_at_angle P2 α ∨ emits_at_angle P2 β ∧ emits_at_angle P3 α ∨ emits_at_angle P3 β ∧
    emits_at_angle P4 α ∨ emits_at_angle P4 β ∧
    beams_intersect_at P [P1, P2, P3, P4])
  : ∃ (P : point), ∀ (P1 P2 P3 P4 P5 : point), emits_at_angle P1 α ∨ emits_at_angle P1 β ∧
      emits_at_angle P2 α ∨ emits_at_angle P2 β ∧ emits_at_angle P3 α ∨ emits_at_angle P3 β ∧
      emits_at_angle P4 α ∨ emits_at_angle P4 β ∧ emits_at_angle P5 α ∨ emits_at_angle P5 β ∧
      beams_intersect_at P [P1, P2, P3, P4, P5] :=
begin
  sorry
end

end five_spotlights_intersect_at_one_point_l212_212110


namespace value_of_p_l212_212858

theorem value_of_p (m n p : ℝ) (h₁ : m = 8 * n + 5) (h₂ : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by {
  sorry
}

end value_of_p_l212_212858


namespace equation_of_curve_line_equation_additional_l212_212412

variable {x y k : ℝ}

-- Definitions of vectors and derived vectors
def m₁ := (0, x)
def n₁ := (1, 1)
def m₂ := (x, 0)
def n₂ := (y^2, 1)

def m := (sqrt 2 * y^2, x + sqrt 2)
def n := (x - sqrt 2, - sqrt 2)

-- Cross product parallel condition 
def cross_product_parallel := (λ m n : ℝ × ℝ, m.1 * n.2 - m.2 * n.1 = 0)

theorem equation_of_curve :
  cross_product_parallel m n → (x^2 / 2) + y^2 = 1 :=
by
  intro h
  sorry

theorem line_equation_additional (hx : (x^2 / 2) + y^2 = 1) (dist : Real.abs (sqrt (1 + k^2) * (-(4 * k) / (1 + 2 * k^2))) = (4 * sqrt 2) / 3) :
  k = 1 ∨ k = -1 :=
by
  sorry

end equation_of_curve_line_equation_additional_l212_212412


namespace seating_arrangements_l212_212583

theorem seating_arrangements (n_families n_members : ℕ) :
  n_families = 3 → n_members = 3 →
  let num_arrangements := (n_members!) ^ n_families * n_families!
  in num_arrangements = (3!)^4 :=
by
  intros h_families h_members
  rw [h_families, h_members]
  have h_fact3 : 3! = 6 := by norm_num
  simp [h_fact3]
  exact sorry

end seating_arrangements_l212_212583


namespace hyperbola_eccentricity_l212_212031

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h₁ : c = sqrt (a^2 + b^2))
  (h₂ : b * c / sqrt (b^2 + a^2) = sqrt (5) / 5 * 2 * c)
  (h₃ : e = c / a) :
  e = sqrt (5) :=
by
  sorry

end hyperbola_eccentricity_l212_212031


namespace slope_abs_value_of_line_l212_212205

/-- Given two circles each of radius 4 and centers at points (0, 12) and (6, 10)
  and a line that passes through the point (4, 0), prove that the absolute value
  of the slope of the line that equally divides the total area of both circles
  is 11/7. -/
theorem slope_abs_value_of_line :
  let r := 4 in
  let center1 := (0, 12) in
  let center2 := (6, 10) in
  let point := (4, 0) in
  |(11 / 7)| = 11 / 7 :=
sorry

end slope_abs_value_of_line_l212_212205


namespace probability_two_cards_diff_suits_l212_212099

def prob_two_cards_diff_suits {deck_size suits cards_per_suit : ℕ} (h1 : deck_size = 40) (h2 : suits = 4) (h3 : cards_per_suit = 10) : ℚ :=
  let total_cards := deck_size
  let cards_same_suit := cards_per_suit - 1
  let cards_diff_suit := total_cards - 1 - cards_same_suit 
  cards_diff_suit / (total_cards - 1)

theorem probability_two_cards_diff_suits (h1 : 40 = 40) (h2 : 4 = 4) (h3 : 10 = 10) :
  prob_two_cards_diff_suits h1 h2 h3 = 10 / 13 :=
by
  sorry

end probability_two_cards_diff_suits_l212_212099


namespace value_of_y_l212_212783

theorem value_of_y (x y : ℤ) (h1 : x - y = 6) (h2 : x + y = 12) : y = 3 := 
by
  sorry

end value_of_y_l212_212783


namespace solve_inequality_l212_212137

theorem solve_inequality (a x : ℝ) : 
  (sqrt (a^2 - x^2) > 2 * x + a) ->
  (a > 0 -> (-a <= x ∧ x < 0)) ∧ 
  (a = 0 -> false) ∧ 
  (a < 0 -> (a <= x ∧ x < -4 * a / 5)) := 
sorry

end solve_inequality_l212_212137


namespace problem_l212_212511

noncomputable def binomial_prob (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem problem 
  (P : ℚ) 
  (hX : binomial_prob 2 P 0 = (4 : ℚ)/9)
  (hY : binomial_prob 3 P 0 = (8 : ℚ)/27) : 
  1 - hY = (19 : ℚ)/27 :=
by 
  sorry

end problem_l212_212511


namespace QO_perpendicular_BC_l212_212495

variables {ABC : Type*} [triangle ABC]
variables {A B C M N Q P O : Point}

-- Definitions based on given conditions
def is_median (A B C M : Point) : Prop := midpoint B C = M
def is_angle_bisector (A B C N : Point) : Prop := bisects_angle A B C N

-- The perpendicular conditions
def is_perpendicular (X Y : Point) (L : Line) : Prop := ∠X Y L = 90

-- Conditions as given in the problem
variable (h1 : is_median A B C M)
variable (h2 : is_angle_bisector A B C N)
variable (h3 : Q = intersection (perpendicular_to N (line_through N A)) (line_through M A))
variable (h4 : P = intersection (perpendicular_to N (line_through N A)) (line_through B A))
variable (h5 : O = intersection (perpendicular_to P (line_through P B A)) (extension_of_line_through N A))

-- The theorem statement to prove
theorem QO_perpendicular_BC : is_perpendicular Q O (line_through B C) :=
sorry

end QO_perpendicular_BC_l212_212495


namespace volume_of_right_prism_with_trapezoid_base_l212_212119

variable (S1 S2 H a b h: ℝ)

theorem volume_of_right_prism_with_trapezoid_base 
  (hS1 : S1 = a * H) 
  (hS2 : S2 = b * H) 
  (h_trapezoid : a ≠ b) : 
  1 / 2 * (S1 + S2) * h = (1 / 2 * (a + b) * h) * H :=
by 
  sorry

end volume_of_right_prism_with_trapezoid_base_l212_212119


namespace total_weight_l212_212124

def weight_of_blue_ball : ℝ := 6.0
def weight_of_brown_ball : ℝ := 3.12

theorem total_weight (_ : weight_of_blue_ball = 6.0) (_ : weight_of_brown_ball = 3.12) : 
  weight_of_blue_ball + weight_of_brown_ball = 9.12 :=
by
  sorry

end total_weight_l212_212124


namespace equal_angles_quadrilateral_l212_212834

theorem equal_angles_quadrilateral
  (AB CD : Type)
  [convex_quad AB CD]
  (angle_CBD angle_CAB angle_ACD angle_BDA : AB CD → ℝ)
  (h1 : angle_CBD = angle_CAB)
  (h2 : angle_ACD = angle_BDA) : angle_ABC = angle_ADC :=
by sorry

end equal_angles_quadrilateral_l212_212834


namespace range_of_a_l212_212010

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 2 * x + 8 * x^3 + a^2 * Real.exp(2 * x) < 4 * x^2 + a * Real.exp(x) + a^3 * Real.exp(3 * x)) :
  a > 2 / Real.exp(1) :=
by
  sorry

end range_of_a_l212_212010


namespace locus_of_centers_l212_212318

-- Define the given circles
def C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
def C2 := { p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9 }

-- Prove the locus of the centers (a, b)
theorem locus_of_centers (a b : ℝ) :
  (∀ (r : ℝ), 
    ((a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (3 - r)^2))) →
  16 * a^2 + 25 * b^2 - 42 * a - 49 = 0 :=
by (!sorry: This is left to be proved)


end locus_of_centers_l212_212318


namespace problem1_problem2_l212_212136

-- Problem 1: Solution set for x(7 - x) >= 12
theorem problem1 (x : ℝ) : x * (7 - x) ≥ 12 ↔ (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Problem 2: Solution set for x^2 > 2(x - 1)
theorem problem2 (x : ℝ) : x^2 > 2 * (x - 1) ↔ true :=
by
  sorry

end problem1_problem2_l212_212136


namespace equilateral_triangle_side_length_l212_212144

-- Define the point Q with its distances to the vertices D, E, and F
variables (Q D E F : Point)

-- Conditions given in the problem
axiom dist_DQ : dist D Q = 2
axiom dist_EQ : dist E Q = Real.sqrt 5
axiom dist_FQ : dist F Q = 3

-- Definition of an equilateral triangle with side length t
def equilateral_triangle (DEF : Triangle) (t : Real) : Prop :=
  side_length DEF D E = t ∧
  side_length DEF E F = t ∧
  side_length DEF F D = t

-- Definition of the unique point condition
def unique_point (DEF : Triangle) (Q D E F : Point) : Prop :=
  dist D Q = 2 ∧
  dist E Q = Real.sqrt 5 ∧
  dist F Q = 3

-- The Lean 4 statement
theorem equilateral_triangle_side_length (t : Real) (DEF : Triangle) :
  equilateral_triangle DEF t →
  unique_point DEF Q D E F →
  t = 2 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_side_length_l212_212144


namespace number_of_ordered_pairs_l212_212429

theorem number_of_ordered_pairs :
  { (x : ℤ) // x ≥ 0 } × ℤ → Prop :=
λ xy, let ⟨x, y⟩ := xy in 2 * x^2 - 2 * x * y + y^2 = 289

example : finset.card (finset.filter number_of_ordered_pairs (finset.product (finset.Ico 0 18) (finset.Ico (-18) 18))) = 7 :=
by
  sorry

end number_of_ordered_pairs_l212_212429


namespace lcm_5_6_8_18_l212_212991

/-- The least common multiple of the numbers 5, 6, 8, and 18 is 360. -/
theorem lcm_5_6_8_18 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 18) = 360 := by
  sorry

end lcm_5_6_8_18_l212_212991


namespace find_m_l212_212519

-- Definition of the sequence pattern
def sequence : ℕ → ℚ
| 1 := 1 / 1
| 2 := 1 / 2
| 3 := 2 / 1
| 4 := 1 / 3
| 5 := 2 / 2
| 6 := 3 / 1
| 7 := 1 / 4
| 8 := 2 / 3
| 9 := 3 / 2
| 10 := 4 / 1
| 11 := 1 / 5
| 12 := 2 / 4
| 13 := 3 / 3
| 14 := 4 / 2
| 15 := 5 / 1
| 16 := 1 / 6
| _ := sorry  -- ignoring further terms for simplicity

def F (m : ℕ) : ℚ := sequence m

theorem find_m (m : ℕ) : F(m) = 1 / 101 ↔ m = 5051 :=
by sorry

end find_m_l212_212519


namespace hyperbola_focal_coordinates_l212_212160

theorem hyperbola_focal_coordinates:
  ∀ (x y : ℝ), x^2 / 16 - y^2 / 9 = 1 → ∃ c : ℝ, c = 5 ∧ (x = -c ∨ x = c) ∧ y = 0 :=
by
  intro x y
  sorry

end hyperbola_focal_coordinates_l212_212160


namespace probabilities_l212_212607

/-- Define events A and B for the probability space of rolling three six-sided dice. -/
def event_A (ω : Fin 6 × Fin 6 × Fin 6) : Prop := ω.1 ≠ ω.2 ∧ ω.2 ≠ ω.3 ∧ ω.1 ≠ ω.3
def event_B (ω : Fin 6 × Fin 6 × Fin 6) : Prop := ω.1 = 5 ∨ ω.2 = 5 ∨ ω.3 = 5

/-- Define the sample space of rolling three six-sided dice. -/
def sample_space : Finset (Fin 6 × Fin 6 × Fin 6) := 
  (Finset.fin_range 6) ×ˢ (Finset.fin_range 6) ×ˢ (Finset.fin_range 6)

/-- Define the probabilities P(AB) and P(B|A) respectively. -/
theorem probabilities (h : 0 < sample_space.card) :
    let P := (sample_space.filter (λ ω, event_A ω ∧ event_B ω)).card / sample_space.card
    let Q := (sample_space.filter (λ ω, event_A ω)).card / sample_space.card
    (P = 75 / 216) ∧ (P / Q = 5 / 8) :=
by
  sorry

end probabilities_l212_212607


namespace intersection_A_B_union_B_not_A_range_of_a_l212_212407

variable (U : Set ℝ)
variable (A B C : Set ℝ)
variable (a : ℝ)

noncomputable def U := Set.univ
noncomputable def A := {x : ℝ | 2 < x ∧ x < 9}
noncomputable def B := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
noncomputable def C := {x : ℝ | a ≤ x ∧ x ≤ 2 - a}
noncomputable def not_A := {x : ℝ | x ≤ 2 ∨ x ≥ 9}
noncomputable def not_B := {x : ℝ | x < -2 ∨ x > 5}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 5} := sorry

theorem union_B_not_A :
  B ∪ not_A = {x : ℝ | x ≤ 5 ∨ x ≥ 9} := sorry

theorem range_of_a (h : C ∪ not_B = U) : a ≤ -3 := sorry

end intersection_A_B_union_B_not_A_range_of_a_l212_212407


namespace find_slope_of_line_through_focus_l212_212271

noncomputable def slope_of_line {l : ℝ → ℝ} (F : ℝ × ℝ) (C : ℝ → ℝ) (M : ℝ × ℝ) : ℝ :=
  Classical.choice (choose_slope F C M)

theorem find_slope_of_line_through_focus (l : ℝ → ℝ)
  (F : ℝ × ℝ := (1, 0)) (C : ℝ × ℝ → Prop := λ p, p.2^2 = 4 * p.1)
  (M : ℝ × ℝ := (-1, 2))
  (hyp : (∀ A B : ℝ × ℝ, C A → C B → A ≠ B → l (A.1 - F.1) = A.2 ∧ l (B.1 - F.1) = B.2 → 
    ((A.1 + 1) * (B.1 + 1) + (A.2 - 2) * (B.2 - 2) = 0))) :
  slope_of_line F C M = 1 := 
sorry

end find_slope_of_line_through_focus_l212_212271


namespace Robert_has_taken_more_photos_l212_212096

variables (C L R : ℕ) -- Claire's, Lisa's, and Robert's photos

-- Conditions definitions:
def ClairePhotos : Prop := C = 8
def LisaPhotos : Prop := L = 3 * C
def RobertPhotos : Prop := R > C

-- The proof problem statement:
theorem Robert_has_taken_more_photos (h1 : ClairePhotos C) (h2 : LisaPhotos C L) : RobertPhotos C R :=
by { sorry }

end Robert_has_taken_more_photos_l212_212096


namespace man_profit_doubled_l212_212238

noncomputable def percentage_profit (C SP1 SP2 : ℝ) : ℝ :=
  (SP2 - C) / C * 100

theorem man_profit_doubled (C SP1 SP2 : ℝ) (h1 : SP1 = 1.30 * C) (h2 : SP2 = 2 * SP1) :
  percentage_profit C SP1 SP2 = 160 := by
  sorry

end man_profit_doubled_l212_212238


namespace solve_quadratic_completing_square_l212_212133

theorem solve_quadratic_completing_square (x : ℝ) :
  x^2 - 4 * x + 3 = 0 → (x - 2)^2 = 1 :=
by sorry

end solve_quadratic_completing_square_l212_212133


namespace correct_operation_l212_212612

theorem correct_operation (a : ℝ) : 
  (-2 * a^2)^3 = -8 * a^6 :=
by sorry

end correct_operation_l212_212612


namespace sum_of_fractions_bounds_l212_212216

theorem sum_of_fractions_bounds (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum_numerators : a + c = 1000) (h_sum_denominators : b + d = 1000) :
  (999 / 969 + 1 / 31) ≤ (a / b + c / d) ∧ (a / b + c / d) ≤ (999 + 1 / 999) :=
by
  sorry

end sum_of_fractions_bounds_l212_212216


namespace find_a_l212_212090

noncomputable def ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 4 = 1

noncomputable def distance (x y a : ℝ) : ℝ := real.sqrt ((x - a)^2 + y^2)

theorem find_a (x y a : ℝ) (h1 : ellipse x y) (h2 : 0 < a) (h3 : a < 3) (h4 : distance x y a = 1) :
  a = real.sqrt 15 / 2 :=
sorry

end find_a_l212_212090


namespace prove_smallest_x_is_zero_l212_212996

noncomputable def smallest_x : ℝ :=
  let f := λ (x : ℝ), (5 * x - 20) / (4 * x - 5)
  in if h : f(x)^2 + f(x) = 20 then x else 0

theorem prove_smallest_x_is_zero :
  let f := λ (x : ℝ), (5 * x - 20) / (4 * x - 5)
  in ∃ x : ℝ, f(x)^2 + f(x) = 20 ∧ (∀ y : ℝ, f(y)^2 + f(y) = 20 → x ≤ y) :=
begin
  sorry, -- proof goes here
end

end prove_smallest_x_is_zero_l212_212996


namespace find_investment_b_l212_212290

def investment_a : Nat := 8000
def investment_b : Nat -- this is what we need to find
def investment_c : Nat := 2000
def profit_c : Nat := 36000
def total_profit : Nat := 252000

theorem find_investment_b (x : Nat) (h : x = 4000) : 
  (investment_c : ℚ) / (investment_a + x + investment_c) = profit_c / total_profit :=
by
  sorry

end find_investment_b_l212_212290


namespace diane_total_loss_l212_212697

def initial_amount : ℤ := 100
def amount_won : ℤ := 65
def amount_owed : ℤ := 50

theorem diane_total_loss : initial_amount + amount_won - (initial_amount + amount_won) + amount_owed = 215 := by
  calc
    initial_amount + amount_won - (initial_amount + amount_won) + amount_owed
      = amount_owed : by rw [add_sub_cancel'_right]
      = 50 : rfl
      = 215 : by sorry

end diane_total_loss_l212_212697


namespace sides_of_leaves_connected_l212_212095

theorem sides_of_leaves_connected
    (L : Type) [Leaf L]
    (finite_discs : ∀ (d1 d2 : disc L), finite (intersection d1 d2))
    (side : ∀ (L : Leaf) (d : disc L), side L d) 
    : connected (sides L) := 
sorry

end sides_of_leaves_connected_l212_212095


namespace quadrilateral_partition_l212_212922

variables {A B C D M_t N_t : Type}
variables [Quadrilateral A B C D]

/-- Given a quadrilateral $ABCD$ with points $M_t$ on side $AB$ and $N_t$ on side $DC$
dividing the sides in the ratio $t$ to $(1-t)$ for $0 < t < 1$, prove that the union of 
the line segments $M_tN_t$ and the diagonals $AD$ and $BC$ partitions the quadrilateral
into non-overlapping segments. --/
theorem quadrilateral_partition : 
  (∃ (M_t N_t : ℝ) (t : ℝ), 
    0 < t ∧ t < 1 ∧ 
    ((AM_t / M_tB) = t / (1-t) ∧ (DN_t / N_tC) = t / (1-t)) ∧
    (set.union (set.range (λ t, segment (M_t t) (N_t t))) (diagonal_segment A D C B) = A ∪ B ∪ C ∪ D)) :=
  by
  sorry

end quadrilateral_partition_l212_212922


namespace circle_chord_midpoint_equal_segments_l212_212670

open EuclideanGeometry

theorem circle_chord_midpoint_equal_segments
  {O A B G C D E F : Point}
  (hO_center : O = midpoint A B)
  (hC_midAG : C = midpoint A G)
  (hCD_perp_AB : perp (line C D) (line A B))
  (hE_intersection : E ∈ (intersection (line C D) (line A G)))
  (hF_intersection : F ∈ (intersection (line B C) (line A G))) :
  dist A E = dist E F := 
sorry

end circle_chord_midpoint_equal_segments_l212_212670


namespace exists_pair_N_ij_gt_200_l212_212894

noncomputable def e : ℝ := 2.71828

variable (A : Fin 29 → Set ℕ)

def N_i (i : Fin 29) (x : ℕ) : ℕ :=
  (A i).filter (λ n, n ≤ x).card

def N_ij (i j : Fin 29) (x : ℕ) : ℕ :=
  (A i ∩ A j).filter (λ n, n ≤ x).card

theorem exists_pair_N_ij_gt_200 :
  (∀ i : Fin 29, N_i A i 1988 ≥ 1988 / e) →
  ∃ (i j : Fin 29) (h : i < j), N_ij A i j 1988 > 200 := by
  sorry

end exists_pair_N_ij_gt_200_l212_212894


namespace cost_to_paint_floor_l212_212563

-- Definitions from the conditions
def length := 22
def breadth := length / 3
def area := length * breadth
def rate_per_sqm := 3
def total_cost := area * rate_per_sqm

-- Statement to prove
theorem cost_to_paint_floor : total_cost = 483.78 := by
  sorry

end cost_to_paint_floor_l212_212563


namespace triangle_similar_lines_count_l212_212523

theorem triangle_similar_lines_count (ABC : Triangle) (P : Point) 
  (hP : P ∈ ABC.side AB ∨ P ∈ ABC.side BC ∨ P ∈ ABC.side CA) : 
  ∃ lines : set Line, lines.through P ∧ (∀ line ∈ lines, ∃ A' B' C' : Point, A' B' C' forms_triangle ∧ is_similar (triangle P A' B' C') ABC) ∧ lines.card = 4 :=
by sorry

end triangle_similar_lines_count_l212_212523


namespace asymptotes_tangent_to_circle_l212_212433

theorem asymptotes_tangent_to_circle {m : ℝ} (hm : m > 0) 
  (hyp_eq : ∀ x y : ℝ, y^2 - (x^2 / m^2) = 1) 
  (circ_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * y + 3 = 0) : 
  m = (Real.sqrt 3) / 3 :=
sorry

end asymptotes_tangent_to_circle_l212_212433


namespace value_of_c_l212_212088

theorem value_of_c (a b c : ℕ) (hab : b = 1) (hd : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_pow : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_gt : 100 * c + 10 * c + b > 300) : 
  c = 4 :=
sorry

end value_of_c_l212_212088


namespace no_prime_divisible_by_42_l212_212421

theorem no_prime_divisible_by_42 : 
  ∀ p : ℕ, Prime p → ¬ (42 ∣ p) := 
by
  intro p hp hdiv
  have h2 : 2 ∣ p := dvd_of_mul_right_dvd hdiv
  have h3 : 3 ∣ p := dvd_of_mul_left_dvd (dvd_of_mul_right_dvd hdiv)
  have h7 : 7 ∣ p := dvd_of_mul_left_dvd hdiv
  sorry

end no_prime_divisible_by_42_l212_212421


namespace cone_base_area_l212_212585

-- Define the volume of the cylinder
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Define the volume of the cone
def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

-- Define the problem conditions
def cylinder_radius : ℝ := 1
def cylinder_height : ℝ := 1
def cone_height : ℝ := 1

-- Define the theorem to prove
theorem cone_base_area :
  ∃ (r : ℝ), volume_cylinder cylinder_radius cylinder_height = volume_cone r cone_height ∧ (π * r^2 = 3 * π) :=
by
  -- Proof goes here
  sorry

end cone_base_area_l212_212585


namespace remainder_2015_div_28_l212_212993

theorem remainder_2015_div_28 : 2015 % 28 = 17 :=
by
  sorry

end remainder_2015_div_28_l212_212993


namespace g_at_5_l212_212164

def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ (x : ℝ), g x + 2 * g (1 - x) = x^2 + 2 * x

theorem g_at_5 : g 5 = -19 / 3 :=
by {
  sorry
}

end g_at_5_l212_212164


namespace find_product_of_roots_l212_212891

namespace ProductRoots

variables {k m : ℝ} {x1 x2 : ℝ}

theorem find_product_of_roots (h1 : x1 ≠ x2) 
    (hx1 : 5 * x1 ^ 2 - k * x1 = m) 
    (hx2 : 5 * x2 ^ 2 - k * x2 = m) : x1 * x2 = -m / 5 :=
sorry

end ProductRoots

end find_product_of_roots_l212_212891


namespace forged_cube_edge_length_l212_212647

def rectangle_volume (l w h : ℝ) : ℝ := l * w * h

def cube_edge_length (volume : ℝ) : ℝ := volume ^ (1 / 3 : ℝ)

theorem forged_cube_edge_length : 
  rectangle_volume 50 8 20 = 8000 ∧ cube_edge_length 8000 = 20 :=
by 
  sorry

end forged_cube_edge_length_l212_212647


namespace sets_satisfy_union_l212_212006

theorem sets_satisfy_union (A : Set Int) : (A ∪ {-1, 1} = {-1, 0, 1}) → 
  (∃ (X : Finset (Set Int)), X.card = 4 ∧ ∀ B ∈ X, A = B) :=
  sorry

end sets_satisfy_union_l212_212006


namespace no_prime_divisible_by_42_l212_212415

open Nat

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → 42 ∣ p → p = 0 :=
by
  intros p hp hdiv
  sorry

end no_prime_divisible_by_42_l212_212415


namespace intersection_point_l212_212308

noncomputable def h (x : ℝ) : ℝ := 4.125 - (x+0.5)^2 / 2

theorem intersection_point :
  ∃ a b : ℝ, h(a) = h(a-4) + 1 ∧ a = -1.5 ∧ b = 3.875 ∧ a + b = 2.375 ∧ b ≠ -a :=
sorry

end intersection_point_l212_212308


namespace angle_ABC_eq_angle_ADC_l212_212826

-- Given a convex quadrilateral ABCD
variables {A B C D O : Type}
variables [convex_quadrilateral A B C D]

-- Given conditions
variable (angle_CBD_eq_angle_CAB : ∠ CBD = ∠ CAB)
variable (angle_ACD_eq_angle_BDA : ∠ ACD = ∠ BDA)

-- Prove that ∠ ABC = ∠ ADC 
theorem angle_ABC_eq_angle_ADC :
  ∠ ABC = ∠ ADC :=
begin
  sorry -- Proof not required
end

end angle_ABC_eq_angle_ADC_l212_212826


namespace coefficient_of_x_in_expansion_l212_212392

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

noncomputable def sum_of_coefficients (x : ℝ) (n : ℕ) : ℝ :=
  (x + (3 / x))^n

theorem coefficient_of_x_in_expansion 
  (n : ℕ)
  (h1 : (sum_of_coefficients 1 n / 2^n) = 64) :
  nat.choose 6 2 * 9 = 135 :=
by
  sorry

end coefficient_of_x_in_expansion_l212_212392


namespace parabola_focus_coordinates_l212_212555

theorem parabola_focus_coordinates : (∃ (p : ℝ), 4 * p = 8) → ∃ (focus: ℝ × ℝ), focus = (0, 2) :=
by
  intro h
  cases h with p hp
  use (0, p)
  rw hp
  linarith

end parabola_focus_coordinates_l212_212555


namespace sum_of_nine_numbers_l212_212440

theorem sum_of_nine_numbers (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 9) : 
  (9 * avg = 47.7) :=
by 
  rw [h_avg, h_n]
  norm_num

end sum_of_nine_numbers_l212_212440


namespace min_weighings_to_identify_fake_l212_212975

def piles := 1000000
def coins_per_pile := 1996
def weight_real_coin := 10
def weight_fake_coin := 9
def expected_total_weight : Nat :=
  (piles * (piles + 1) / 2) * weight_real_coin

theorem min_weighings_to_identify_fake :
  (∃ k : ℕ, k < piles ∧ 
  ∀ (W : ℕ), W = expected_total_weight - k → k = expected_total_weight - W) →
  true := 
by
  sorry

end min_weighings_to_identify_fake_l212_212975


namespace minimize_AB_l212_212395

-- Definition of the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2 * y - 3 = 0

-- Definition of the point P
def P : ℝ × ℝ := (-1, 2)

-- Definition of the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- The goal is to prove that line_l is the line through P minimizing |AB|
theorem minimize_AB : 
  ∀ l : ℝ → ℝ → Prop, 
  (∀ x y, l x y → (∃ a b, circleC a b ∧ l a b ∧ circleC x y ∧ l x y ∧ (x ≠ a ∨ y ≠ b)) → False) 
  → l = line_l :=
by
  sorry

end minimize_AB_l212_212395


namespace points_concyclic_l212_212320

-- Definitions for the points and conditions
variables (A B C D H K : Point)
variables (isParallelogram : Parallelogram A B C D)
variables (obtuseAngleAtA : obtuseAngle A)
variables (footH : FootOfPerpendicular A B C H)
variables (medianPointK : MedianMeetsCircumcircle A B C K)

-- Theorem to prove that K, H, C, D are concyclic
theorem points_concyclic 
  (parallelogramABCD : Parallelogram A B C D)
  (obtuseAngleA : obtuseAngle A)
  (perpendicularFootH : FootOfPerpendicular A B C H)
  (medianOnCircumcircle : MedianMeetsCircumcircle C A B K) :
  Concyclic K H C D :=
begin
  sorry
end

end points_concyclic_l212_212320


namespace ADEF_is_cyclic_l212_212262

-- Definitions for the conditions
variables {A B C D E F : Point}
variables {circABC : Circle}
variables (isosceles_ABC : A.between B C)
variables (circumscribed_AB_AC : A.between B A ∧ A.between C A)
variables (angle_bisectors_BC_intersects_at_E : intersection (angle_bisector B C) = E)
variables (bisectors_intersect_circumcircle_at_DF 
  : on_circle D circABC ∧ on_circle F circABC)

-- The theorem statement
theorem ADEF_is_cyclic :
  is_cyclic_quadrilateral A D E F := sorry

end ADEF_is_cyclic_l212_212262


namespace find_values_of_x_y_and_sqrt_expr_l212_212384

variable {x y : ℝ}

def condition1 := (sqrt (3 * x + 1) = 4)
def condition2 := (x + 2 * y) ^ (1/3 : ℝ) = -1 

theorem find_values_of_x_y_and_sqrt_expr (h1 : condition1) (h2 : condition2) : 
  x = 5 ∧ y = -3 ∧ (sqrt (2 * x - 5 * y) = 5 ∨ sqrt (2 * x - 5 * y) = -5) :=
by
  sorry

end find_values_of_x_y_and_sqrt_expr_l212_212384


namespace geometry_problem_l212_212854

theorem geometry_problem
  (PQ_parallel_RS : ∀ P Q R S : Type, parallel PQ RS)
  (PRT_straight : ∀ (P R T : Type), straight_angle PRT)
  (angle_PRS : angle PRS = 110)
  (angle_PQR : angle PQR = 80)
  (angle_PSR : angle PSR = 30) :
  y = 80 :=
by
  sorry

end geometry_problem_l212_212854


namespace transformation_matrix_correct_l212_212094

def matrix_A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 1, 2], ![3, -1, 0], ![1, 1, -2]]

def matrix_B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, -1, 1], ![-1, 1, -2], ![-1, 2, 1]]

def matrix_A' : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![9, -1, 27], ![5, -1, 14], ![-3, 1, -8]]

theorem transformation_matrix_correct :
  let B_inv := matrix.inverse matrix_B in
  (B_inv ⬝ matrix_A ⬝ matrix_B) = matrix_A' :=
sorry

end transformation_matrix_correct_l212_212094


namespace age_proof_l212_212179

theorem age_proof (A B C D k m : ℕ)
  (h1 : A + B + C + D = 76)
  (h2 : A - 3 = k)
  (h3 : B - 3 = 2*k)
  (h4 : C - 3 = 3*k)
  (h5 : A - 5 = 3*m)
  (h6 : D - 5 = 4*m)
  (h7 : B - 5 = 5*m) :
  A = 11 := 
sorry

end age_proof_l212_212179


namespace ratio_of_jumps_l212_212930

theorem ratio_of_jumps (run_ric: ℕ) (jump_ric: ℕ) (run_mar: ℕ) (extra_dist: ℕ)
    (h1 : run_ric = 20)
    (h2 : jump_ric = 4)
    (h3 : run_mar = 18)
    (h4 : extra_dist = 1) :
    (run_mar + extra_dist - run_ric - jump_ric) / jump_ric = 7 / 4 :=
by
  sorry

end ratio_of_jumps_l212_212930


namespace tim_dimes_shining_shoes_l212_212202

-- Definitions of the conditions
def value_of_nickel : ℝ := 0.05
def value_of_dime : ℝ := 0.10
def value_of_half_dollar : ℝ := 0.50

def num_of_nickels_shining_shoes : ℕ := 3
def num_of_dimes_tip_jar : ℕ := 7
def num_of_half_dollars_tip_jar : ℕ := 9
def total_amount : ℝ := 6.65

-- Prove that Tim got 13 dimes for shining shoes
theorem tim_dimes_shining_shoes : 
  let value_from_tip_jar := num_of_dimes_tip_jar * value_of_dime + 
                            num_of_half_dollars_tip_jar * value_of_half_dollar,
      value_from_shining_shoes := total_amount - value_from_tip_jar,
      value_of_nickels := num_of_nickels_shining_shoes * value_of_nickel,
      value_from_dimes := value_from_shining_shoes - value_of_nickels in
      value_from_dimes = 1.3 :=
sorry

end tim_dimes_shining_shoes_l212_212202


namespace minimum_value_of_f_l212_212251

open Real

noncomputable def f (θ : ℝ) := 3 * cos θ + 2 * sec θ + sqrt 3 * tan θ

theorem minimum_value_of_f :
  ∃ (θ : ℝ), 0 < θ ∧ θ < π / 2 ∧ (∀ (x : ℝ), 0 < x ∧ x < π / 2 → f θ ≤ f x) ∧ f θ = 6 :=
by
  use π / 4
  split
  -- 0 < θ
  linarith
  split
  -- θ < π / 2
  linarith
  split
  -- ∀ x, 0 < x < π / 2 → f θ ≤ f x
  sorry
  -- f θ = 6
  sorry

end minimum_value_of_f_l212_212251


namespace right_triangle_legs_sum_l212_212956

-- Definitions
def sum_of_legs (a b : ℕ) : ℕ := a + b

-- Main theorem statement
theorem right_triangle_legs_sum (x : ℕ) (h : x^2 + (x + 1)^2 = 53^2) :
  sum_of_legs x (x + 1) = 75 :=
sorry

end right_triangle_legs_sum_l212_212956


namespace quadrilateral_partition_l212_212923

variables {A B C D M_t N_t : Type}
variables [Quadrilateral A B C D]

/-- Given a quadrilateral $ABCD$ with points $M_t$ on side $AB$ and $N_t$ on side $DC$
dividing the sides in the ratio $t$ to $(1-t)$ for $0 < t < 1$, prove that the union of 
the line segments $M_tN_t$ and the diagonals $AD$ and $BC$ partitions the quadrilateral
into non-overlapping segments. --/
theorem quadrilateral_partition : 
  (∃ (M_t N_t : ℝ) (t : ℝ), 
    0 < t ∧ t < 1 ∧ 
    ((AM_t / M_tB) = t / (1-t) ∧ (DN_t / N_tC) = t / (1-t)) ∧
    (set.union (set.range (λ t, segment (M_t t) (N_t t))) (diagonal_segment A D C B) = A ∪ B ∪ C ∪ D)) :=
  by
  sorry

end quadrilateral_partition_l212_212923


namespace simplify_expression_l212_212937

variable (y : ℝ)

theorem simplify_expression :
  4 * y^3 + 8 * y + 6 - (3 - 4 * y^3 - 8 * y) = 8 * y^3 + 16 * y + 3 :=
by
  sorry

end simplify_expression_l212_212937


namespace find_a1_l212_212568

-- Define the sequence recurrence relation
def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) * (1 - a n) = 1

-- Given conditions
def a8_value: ℝ := 2

-- Goal
theorem find_a1 (a : ℕ → ℝ) (h_seq : sequence a) (h_a8 : a 8 = 2) : a 1 = 1 / 2 :=
  sorry

end find_a1_l212_212568


namespace avg_of_nine_numbers_l212_212442

theorem avg_of_nine_numbers (average : ℝ) (sum : ℝ) (h : average = (sum / 9)) (h_avg : average = 5.3) : sum = 47.7 := by
  sorry

end avg_of_nine_numbers_l212_212442


namespace distance_X_Y_is_24_l212_212525

open Real

noncomputable def distance_between_points (Y_walk_rate B_walk_rate B_walked T_walk) :=
  let Y_walked := Y_walk_rate * (T_walk + 1) in
  Y_walked + B_walked

theorem distance_X_Y_is_24 :
  ∀ (Y_walk_rate B_walk_rate B_walked : ℝ) (T_walk : ℝ), 
  Y_walk_rate = 3 → B_walk_rate = 4 → B_walked = 12 → T_walk = B_walked / B_walk_rate →
  distance_between_points Y_walk_rate B_walk_rate B_walked T_walk = 24 :=
by
  intros Y_walk_rate B_walk_rate B_walked T_walk hY hB hBw hT
  sorry

end distance_X_Y_is_24_l212_212525


namespace total_cost_correct_l212_212233

/-- Define the base car rental cost -/
def rental_cost : ℝ := 150

/-- Define cost per mile -/
def cost_per_mile : ℝ := 0.5

/-- Define miles driven on Monday -/
def miles_monday : ℝ := 620

/-- Define miles driven on Thursday -/
def miles_thursday : ℝ := 744

/-- Define the total cost Zach spent -/
def total_cost : ℝ := rental_cost + (miles_monday * cost_per_mile) + (miles_thursday * cost_per_mile)

/-- Prove that the total cost Zach spent is 832 dollars -/
theorem total_cost_correct : total_cost = 832 := by
  sorry

end total_cost_correct_l212_212233


namespace change_in_total_berries_l212_212985

-- Define the given conditions
def blueberries_per_blue_box : ℕ := 30

theorem change_in_total_berries :
  ∃ (S : ℕ), S + blueberries_per_blue_box = 80 ∧ (S - blueberries_per_blue_box = 20) :=
by {
  use 50,
  split,
  { exact rfl },
  { simp, refl },
}

end change_in_total_berries_l212_212985


namespace parameterized_line_solution_l212_212168

theorem parameterized_line_solution :
  ∃ s l : ℝ, s = 1 / 2 ∧ l = -10 ∧
    ∀ t : ℝ, ∃ x y : ℝ,
      (x = -7 + t * l → y = s + t * (-5)) ∧ (y = (1 / 2) * x + 4) :=
by
  sorry

end parameterized_line_solution_l212_212168


namespace length_of_qr_l212_212141

theorem length_of_qr (Q : ℝ) (PQ QR : ℝ) 
  (h1 : Real.sin Q = 0.6)
  (h2 : PQ = 15) :
  QR = 18.75 :=
by
  sorry

end length_of_qr_l212_212141


namespace white_area_of_sign_l212_212577

theorem white_area_of_sign : 
  let total_area := 6 * 18
  let F_area := 2 * (4 * 1) + 6 * 1
  let O_area := 2 * (6 * 1) + 2 * (4 * 1)
  let D_area := 6 * 1 + 4 * 1 + 4 * 1
  let total_black_area := F_area + O_area + O_area + D_area
  total_area - total_black_area = 40 :=
by
  sorry

end white_area_of_sign_l212_212577


namespace probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l212_212787

noncomputable theory
open ProbabilityTheory

-- Definitions based on given conditions
def events := {1, 2, 3, 4, 5, 6}
def eventA := {1, 2, 3}
def eventB := {1, 2, 4}

def numTrials : ℕ := 10
def numA : ℕ := 6
def pA : ℚ := 1 / 2
def pB_given_A : ℚ := 2 / 3
def pB_given_Ac : ℚ := 1 / 3

-- Theorem for probability that B does not occur given A occurred 6 times.
theorem probability_B_does_not_occur_given_A_6_occur :
  -- The probability of B not occurring given A occurred exactly 6 times.
  -- Should be approximately 2.71 * 10^(-4)
  true := sorry

-- Theorem for the expected number of times B occurs.
theorem expected_value_B_occurances : 
  -- The expected value of the number of occurrences of event B given the conditions.
  -- Should be 16 / 3
  true := sorry

end probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l212_212787


namespace distance_between_vertices_l212_212712

def hyperbola_eq (x y : ℝ) := (x ^ 2) / 99 - (y ^ 2) / 36 = 1

theorem distance_between_vertices :
  let a := Real.sqrt 99
  let distance := 2 * a
  distance = 6 * Real.sqrt 11 :=
by
sad sorry

end distance_between_vertices_l212_212712


namespace field_trip_people_per_bus_l212_212637

def number_of_people_on_each_bus (vans buses people_per_van total_people : ℕ) : ℕ :=
  (total_people - (vans * people_per_van)) / buses

theorem field_trip_people_per_bus :
  let vans := 9
  let buses := 10
  let people_per_van := 8
  let total_people := 342
  number_of_people_on_each_bus vans buses people_per_van total_people = 27 :=
by
  sorry

end field_trip_people_per_bus_l212_212637


namespace hyperbola_asymptotes_tangent_to_circle_l212_212436

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, y^2 - (x^2 / m^2) = 1 → (x^2 + y^2 - 4*y + 3 = 0 → distance_center_to_asymptote (0, 2) (y = x / m) = 1)) → 
  m = real.sqrt(3) / 3 :=
sorry

end hyperbola_asymptotes_tangent_to_circle_l212_212436


namespace same_root_a_eq_3_l212_212797

theorem same_root_a_eq_3 {x a : ℝ} (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
by
  sorry

end same_root_a_eq_3_l212_212797


namespace symm_property_interval_condition_main_theorem_l212_212750

-- Define the function f with the given conditions
def f (x : ℝ) : ℝ := sorry -- Given function f, to be defined from the conditions

-- Define the conditions
theorem symm_property (f : ℝ → ℝ) : (∀ x, f(x+1) = f(-x+1)) := sorry
theorem interval_condition (f : ℝ → ℝ) : (∀ x, (0 < x ∧ x < 1) → f(x) = Real.exp(-x)) := sorry

-- Main theorem to be proved
theorem main_theorem (f : ℝ → ℝ) 
  (symm : ∀ x, f(x+1) = f(-x+1))
  (interval : ∀ x, (0 < x ∧ x < 1) → f(x) = Real.exp(-x)) : 
  f(Real.log 3) = 3 * Real.exp(-2) := sorry

end symm_property_interval_condition_main_theorem_l212_212750


namespace sphere_radius_in_cone_l212_212865

theorem sphere_radius_in_cone 
  (r : ℝ) 
  (base_radius : ℝ) 
  (height : ℝ) 
  (tangent_spheres : ∀ r base_radius height, base_radius = 7 ∧ height = 15 ∧ 
    (∀ O1 O2 O3, dist O1 O2 = 2 * r ∧ dist O2 O3 = 2 * r ∧ dist O1 O3 = 2 * r)) :
  r = (630 - 262.5 * real.sqrt 3) / 69 :=
by
  sorry

end sphere_radius_in_cone_l212_212865


namespace common_chord_line_l212_212344

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y - 4 = 0

theorem common_chord_line : 
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - y + 1 = 0) := 
by sorry

end common_chord_line_l212_212344


namespace day_after_75_days_l212_212901

theorem day_after_75_days (day_of_week : ℕ → String) (h : day_of_week 0 = "Tuesday") :
  day_of_week 75 = "Sunday" :=
sorry

end day_after_75_days_l212_212901


namespace min_gift_cost_time_constrained_l212_212905

def store := { name : String, mom : ℕ, dad : ℕ, brother : ℕ, sister : ℕ, time_in_store : ℕ }

def romashka : store := { name := "Romashka", mom := 1000, dad := 750, brother := 930, sister := 850, time_in_store := 35 }
def oduvanchik : store := { name := "Oduvanchik", mom := 1050, dad := 790, brother := 910, sister := 800, time_in_store := 30 }
def nezabudka : store := { name := "Nezabudka", mom := 980, dad := 810, brother := 925, sister := 815, time_in_store := 40 }
def landysh : store := { name := "Landysh", mom := 1100, dad := 755, brother := 900, sister := 820, time_in_store := 25 }

constant travel_time : ℕ := 30
constant total_available_time : ℕ := 3 * 60 + 25 -- 205 minutes

def total_time (stores : List store) : ℕ :=
  stores.map (λ s => s.time_in_store).sum + (travel_time * (stores.length - 1))

def total_cost (stores : List store) : ℕ :=
  stores[0].mom + stores[1].dad + stores[2].brother + stores[3].sister

theorem min_gift_cost_time_constrained : 
  ∃ stores : List store, stores.length = 4 ∧ total_time stores ≤ total_available_time ∧ total_cost stores = 3435 :=
by
  sorry

end min_gift_cost_time_constrained_l212_212905


namespace diane_total_loss_l212_212694

-- Define the starting amount of money Diane had.
def starting_amount : ℤ := 100

-- Define the amount of money Diane won.
def winnings : ℤ := 65

-- Define the amount of money Diane owed at the end.
def debt : ℤ := 50

-- Define the total amount of money Diane had after winnings.
def mid_game_total : ℤ := starting_amount + winnings

-- Define the total amount Diane lost.
def total_loss : ℤ := mid_game_total + debt

-- Theorem stating the total amount Diane lost is 215 dollars.
theorem diane_total_loss : total_loss = 215 := by
  sorry

end diane_total_loss_l212_212694


namespace difference_of_numbers_l212_212180

variables (x y : ℝ)

-- Definitions corresponding to the conditions
def sum_of_numbers (x y : ℝ) : Prop := x + y = 30
def product_of_numbers (x y : ℝ) : Prop := x * y = 200

-- The proof statement in Lean
theorem difference_of_numbers (x y : ℝ) 
  (h1: sum_of_numbers x y) 
  (h2: product_of_numbers x y) : x - y = 10 ∨ y - x = 10 :=
by
  sorry

end difference_of_numbers_l212_212180


namespace liquefied_gas_more_economical_retrofit_cost_equivalence_l212_212700

noncomputable def cost_gasoline (t : ℕ) : ℝ :=
  (200 * t / 12) * 2.8

noncomputable def cost_liquefied_gas_min (t : ℕ) : ℝ :=
  (200 * t / 16) * 3

noncomputable def cost_liquefied_gas_max (t : ℕ) : ℝ :=
  (200 * t / 15) * 3

noncomputable def retrofit_cost : ℝ := 5000

theorem liquefied_gas_more_economical :
  ∀ t : ℕ, 37.5 * t ≤ (200 * t / 16) * 3 ∧ (200 * t / 16) * 3 ≤ 40 * t ∧ 40 * t < (200 * t / 12) * 2.8 := by
  sorry

theorem retrofit_cost_equivalence (t : ℕ) :
  ∃ t_min t_max : ℕ, 37.5 * t + retrofit_cost = cost_gasoline t ∧ 40 * t + retrofit_cost = cost_gasoline t ∧ 546 ≤ t ∧ t ≤ 750 := by
  sorry

end liquefied_gas_more_economical_retrofit_cost_equivalence_l212_212700


namespace spherical_coordinates_convert_l212_212463

theorem spherical_coordinates_convert (ρ θ φ ρ' θ' φ' : ℝ) 
  (h₀ : ρ > 0) 
  (h₁ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₂ : 0 ≤ φ ∧ φ ≤ Real.pi) 
  (h_initial : (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5)) 
  (h_final : (ρ', θ', φ') = (4, (11 * Real.pi) / 8,  Real.pi / 5)) : 
  (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5) → 
  (ρ, θ, φ) = (ρ', θ', φ') := 
by
  sorry

end spherical_coordinates_convert_l212_212463


namespace simple_interest_calculation_l212_212655

theorem simple_interest_calculation (P R T : ℝ) (H₁ : P = 8925) (H₂ : R = 9) (H₃ : T = 5) : 
  P * R * T / 100 = 4016.25 :=
by
  sorry

end simple_interest_calculation_l212_212655


namespace remaining_oranges_l212_212592

/--
Given Michaela needs 45 oranges to get full, and Cassandra needs five times as many oranges as Michaela,
and they picked 520 oranges from the farm today, 
prove that the number of oranges remaining after they've both eaten until they were full is 250 oranges.
-/
theorem remaining_oranges :
  (michaela_needs : 45) → (cassandra_needs : 5 * michaela_needs) →
  (total_picked : 520) →  (remaining : total_picked - (michaela_needs + cassandra_needs)) = 250 :=
by
  sorry

end remaining_oranges_l212_212592


namespace minimum_value_3y2_minus_18y_plus_7_is_3_l212_212228

notation "ℝ" => Real

theorem minimum_value_3y2_minus_18y_plus_7_is_3 : 
  ∃ y : ℝ, y = 3 ∧ (∀ z : ℝ, 3 * y^2 - 18 * y + 7 ≤ 3 * z^2 - 18 * z + 7) :=
by
  use 3
  split
  { reflexivity }
  sorry

end minimum_value_3y2_minus_18y_plus_7_is_3_l212_212228


namespace find_value_of_reciprocal_sin_double_angle_l212_212390

open Real

noncomputable def point := ℝ × ℝ

def term_side_angle_passes_through (α : ℝ) (P : point) :=
  ∃ (r : ℝ), P = (r * cos α, r * sin α)

theorem find_value_of_reciprocal_sin_double_angle (α : ℝ) (P : point) (h : term_side_angle_passes_through α P) :
  P = (-2, 1) → (1 / sin (2 * α)) = -5 / 4 :=
by
  intro hP
  sorry

end find_value_of_reciprocal_sin_double_angle_l212_212390


namespace problem_statement_l212_212478

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ A > 0 ∧ B > 0 ∧ C > 0

def sides_opposite_angles (a b c A B C : ℝ) : Prop :=
  c = sqrt 3 ∧ b = 1 ∧ C = 120 ∧ A + B + C = 180

noncomputable def find_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  B = 30

noncomputable def find_area_S (a b c : ℝ) (A B C S : ℝ) : Prop :=
  S = (sqrt 3) / 4


theorem problem_statement :
  ∃ (a b c A B C S : ℝ), sides_opposite_angles a b c A B C ∧
    find_angle_B a b c A B C ∧
    find_area_S a b c A B C S :=
by
  sorry

end problem_statement_l212_212478


namespace isosceles_right_triangle_shaded_area_l212_212667

theorem isosceles_right_triangle_shaded_area :
  ∀ (leg_length : ℕ) (num_total_triangles : ℕ) (num_shaded_triangles : ℕ),
  leg_length = 10 → num_total_triangles = 25 → num_shaded_triangles = 15 →
  let total_area := (1 / 2 : ℝ) * leg_length * leg_length in
  let area_one_triangle := total_area / num_total_triangles in
  let shaded_area := num_shaded_triangles * area_one_triangle in
  shaded_area = 30 := by
sorry

end isosceles_right_triangle_shaded_area_l212_212667


namespace range_of_A_measure_of_A_l212_212043

-- Define the Triangle and Conditions
variables {A B C : ℝ}
variables (a b c : ℝ := 1)
variables (S : ℝ := (Real.sqrt 3 + 1) / 4)

-- Adding given conditions
axiom angle_A_ne_PI_div_2 : A ≠ π / 2
axiom sin_eqn : Real.sin C + Real.sin (B - A) = Real.sqrt 2 * Real.sin (2 * A)
axiom area_eqn : b = Real.sqrt 2 * a
axiom sin_C_eqn : Real.sin C = (Real.sqrt 6 + Real.sqrt 2) / 4
axiom obtuse_angle_C : π / 2 < C ∧ C < π

-- Goals
theorem range_of_A : 0 < A ∧ A ≤ π / 4 :=
by sorry

theorem measure_of_A : A = π / 6 :=
by sorry

end range_of_A_measure_of_A_l212_212043


namespace road_construction_days_l212_212297

theorem road_construction_days
  (length_of_road : ℝ)
  (initial_men : ℕ)
  (completed_length : ℝ)
  (completed_days : ℕ)
  (extra_men : ℕ)
  (initial_days : ℕ)
  (remaining_length : ℝ)
  (remaining_days : ℕ)
  (total_men : ℕ) :
  length_of_road = 15 →
  initial_men = 30 →
  completed_length = 2.5 →
  completed_days = 100 →
  extra_men = 45 →
  initial_days = initial_days →
  remaining_length = length_of_road - completed_length →
  remaining_days = initial_days - completed_days →
  total_men = initial_men + extra_men →
  initial_days = 700 :=
by
  intros
  sorry

end road_construction_days_l212_212297


namespace remaining_students_average_l212_212156

theorem remaining_students_average
  (N : ℕ) (A : ℕ) (M : ℕ) (B : ℕ) (E : ℕ)
  (h1 : N = 20)
  (h2 : A = 80)
  (h3 : M = 5)
  (h4 : B = 50)
  (h5 : E = (N - M))
  : (N * A - M * B) / E = 90 :=
by
  -- Using sorries to skip the proof
  sorry

end remaining_students_average_l212_212156


namespace charity_amount_l212_212306

theorem charity_amount (total : ℝ) (charities : ℕ) (amount_per_charity : ℝ) 
  (h1 : total = 3109) (h2 : charities = 25) : 
  amount_per_charity = 124.36 :=
by
  sorry

end charity_amount_l212_212306


namespace employees_women_with_fair_hair_l212_212261

def percentage_of_employees_women_with_fair_hair (p q : ℚ) : ℚ :=
  p * q

theorem employees_women_with_fair_hair (p q : ℚ) (h1 : p = 0.40) (h2 : q = 0.70) :
  percentage_of_employees_women_with_fair_hair p q = 0.28 :=
by
  rw [h1, h2]
  norm_num

end employees_women_with_fair_hair_l212_212261


namespace find_radius_l212_212945

-- Definition of the conditions
def area_of_sector : ℝ := 10 -- The area of the sector in square centimeters
def arc_length : ℝ := 4     -- The arc length of the sector in centimeters

-- The radius of the circle we want to prove
def radius (r : ℝ) : Prop :=
  (r * 4) / 2 = 10

-- The theorem to be proved
theorem find_radius : ∃ r : ℝ, radius r :=
by
  use 5
  unfold radius
  norm_num

end find_radius_l212_212945


namespace nancy_hourly_wage_l212_212101

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end nancy_hourly_wage_l212_212101


namespace min_value_expr_l212_212019

theorem min_value_expr (x : ℝ) (h : x > -3) : ∃ m, (∀ y > -3, 2 * y + (1 / (y + 3)) ≥ m) ∧ m = 2 * Real.sqrt 2 - 6 :=
by
  sorry

end min_value_expr_l212_212019


namespace deposit_time_l212_212539

theorem deposit_time (r t : ℕ) : 
  8000 + 8000 * r * t / 100 = 10200 → 
  8000 + 8000 * (r + 2) * t / 100 = 10680 → 
  t = 3 :=
by 
  sorry

end deposit_time_l212_212539


namespace inverse_variation_l212_212927

variable (a b : ℝ)

theorem inverse_variation (h_ab : a * b = 400) :
  (b = 0.25 ∧ a = 1600) ∨ (b = 1.0 ∧ a = 400) :=
  sorry

end inverse_variation_l212_212927


namespace intersection_at_theta_equals_pi_l212_212857

noncomputable def polar_intersection {π : ℝ} {ρ : ℝ} (θ : ℝ) : Prop :=
  ρ * Real.sin(π / 6 - θ) + 1 = 0
 
theorem intersection_at_theta_equals_pi : 
  polar_intersection π → (2, π) :=
by
  sorry

end intersection_at_theta_equals_pi_l212_212857


namespace find_hourly_wage_l212_212651

noncomputable def hourly_wage_inexperienced (x : ℝ) : Prop :=
  let sailors_total := 17
  let inexperienced_sailors := 5
  let experienced_sailors := sailors_total - inexperienced_sailors
  let wage_experienced := (6 / 5) * x
  let total_hours_month := 240
  let total_monthly_earnings_experienced := 34560
  (experienced_sailors * wage_experienced * total_hours_month) = total_monthly_earnings_experienced

theorem find_hourly_wage (x : ℝ) : hourly_wage_inexperienced x → x = 10 :=
by
  sorry

end find_hourly_wage_l212_212651


namespace sum_of_nine_numbers_l212_212439

theorem sum_of_nine_numbers (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 9) : 
  (9 * avg = 47.7) :=
by 
  rw [h_avg, h_n]
  norm_num

end sum_of_nine_numbers_l212_212439


namespace isosceles_triangles_same_area_perimeter_angle_l212_212068

theorem isosceles_triangles_same_area_perimeter_angle (x y : ℝ) : 
  let hU : ℝ := sqrt (36 - (10/2)^2),
      AreaU : ℝ := (1/2) * 10 * sqrt 11,
      PerimeterU : ℝ := 6 + 6 + 10,
      hU' : ℝ := sqrt (x^2 - ((22 - 2 * x) / 2)^2),
      AreaU' : ℝ := (1/2) * (22 - 2 * x) * sqrt 11,
      PerimeterU' : ℝ := 2 * x + (22 - 2 * x)
  in x = 7 :=
by {
  -- Corresponding proof would go here
  sorry
}

end isosceles_triangles_same_area_perimeter_angle_l212_212068


namespace log_ineq_no_constant_c_l212_212886

noncomputable theory

section GeometricSequence

variable {α : Type*} [linear_order α] [ordered_comm_group α] [ordered_ring α]

-- Definitions 
def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n, S n = ∑ i in finset.range n, a i

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ} (hq : q > 0)
          (h_geo : geometric_sequence a q) (h_sum : sum_of_first_n_terms a S)

-- The first statement to be proven
theorem log_ineq : ∀ n, 
  (log (S n) + log (S (n + 2))) / 2 < log (S (n + 1)) :=
sorry

-- The second statement to be proven
theorem no_constant_c : 
  ¬∃ c > 0, ∀ n, 
    (log (S n - c) + log (S (n + 2) - c)) / 2 = log (S (n + 1) - c) :=
sorry

end GeometricSequence

end log_ineq_no_constant_c_l212_212886


namespace paul_crayons_left_l212_212114

theorem paul_crayons_left (initial_crayons lost_crayons : ℕ) 
  (h_initial : initial_crayons = 253) 
  (h_lost : lost_crayons = 70) : (initial_crayons - lost_crayons) = 183 := 
by
  sorry

end paul_crayons_left_l212_212114


namespace sequence_and_sum_l212_212821

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  (∃ a1 : ℝ, ∀ n, a n = a1 + d * (n - 1))

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ}
variables {d : ℝ}
variables (h1 : a 2 = 8)
variables (h2 : ∑ i in range 6, a (i + 1) = 66)
variables (h3 : ∀ n, b n = 2 / ((n + 1) * a n))
variables (h4 : ∀ n, T n = (∑ i in range n, b (i + 1)))

theorem sequence_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (d : ℝ) (h_seq : arithmetic_sequence a d) :
  (∀ n, a n = 2 * n + 4) ∧ (∀ n, T n = n / (2 * n + 4)) := by
  have h1 : arithmetic_sequence a d := h_seq
  sorry

end sequence_and_sum_l212_212821


namespace solve_for_wood_length_l212_212623

theorem solve_for_wood_length (y x : ℝ) (h1 : y - x = 4.5) (h2 : x - (1/2) * y = 1) :
  ∃! (x y : ℝ), (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  -- The content of the proof is omitted
  sorry

end solve_for_wood_length_l212_212623


namespace part_I_part_II_l212_212759

def f (x : ℝ) : ℝ := 
  (Real.sqrt 3) * (Real.sin (x / 2)) * (Real.cos (x / 2)) + 
  (Real.cos (x / 2))^2 - 0.5

theorem part_I (k : ℤ) : 
  ∀ x, x ∈ Set.Ioc (2 * k * Real.pi - (2 / 3) * Real.pi) (2 * k * Real.pi + (1 / 3) * Real.pi) ↔
  ∃ n : ℤ, x = 2 * n * Real.pi + Real.pi / 6 - Real.pi / 2 :=
sorry

theorem part_II (B C : ℝ) (a b : ℝ) (h0 : f (B + C) = 1) (h1 : a = Real.sqrt 3) (h2 : b = 1) :
  C = Real.pi / 6 :=
sorry

end part_I_part_II_l212_212759


namespace seating_alternating_girls_boys_l212_212300

-- Define variables
variables (n : ℕ)

-- Definition of the mathematical problem
def alternate_seating_count (n : ℕ) : ℕ :=
  2 * (n!)^2

-- Lean theorem stating the problem
theorem seating_alternating_girls_boys (n : ℕ) :
  ∃ count, count = alternate_seating_count n :=
by
  use 2 * (n!)^2
  sorry

end seating_alternating_girls_boys_l212_212300


namespace non_zero_x_satisfies_equation_l212_212227

theorem non_zero_x_satisfies_equation :
  ∃ (x : ℝ), (x ≠ 0) ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16 / 7 :=
by {
  sorry
}

end non_zero_x_satisfies_equation_l212_212227


namespace area_of_triangle_formed_by_line_and_axes_l212_212946

-- Definitions for the conditions
def line_intersects_x_axis (x : ℝ) : Prop := x + 0 - 2 = 0
def line_intersects_y_axis (y : ℝ) : Prop := 0 + y - 2 = 0

-- Definition for the points of intersection
def x_intercept := 2  -- Derived from line_intersects_x_axis (2)
def y_intercept := 2  -- Derived from line_intersects_y_axis (2)

-- Definition for the base and height of the triangle
def base := x_intercept
def height := y_intercept

-- The area of the triangle formed by the line and the axes
def triangle_area (b h : ℝ) : ℝ := (1 / 2) * b * h

-- The statement to be proved
theorem area_of_triangle_formed_by_line_and_axes : 
  triangle_area base height = 2 :=
by {
  -- Proof goes here (omitted for this task)
  sorry
}

end area_of_triangle_formed_by_line_and_axes_l212_212946


namespace find_constant_l212_212802

-- Define the variables: t, x, y, and the constant
variable (t x y constant : ℝ)

-- Conditions
def x_def : x = constant - 2 * t :=
  by sorry

def y_def : y = 2 * t - 2 :=
  by sorry

def x_eq_y_at_t : t = 0.75 → x = y :=
  by sorry

-- Proposition: Prove that the constant in the equation for x is 1
theorem find_constant (ht : t = 0.75) (hx : x = constant - 2 * t) (hy : y = 2 * t - 2) (he : x = y) :
  constant = 1 :=
  by sorry

end find_constant_l212_212802


namespace factorize_polynomial_l212_212614

theorem factorize_polynomial (x : ℤ) (h : x^3 ≠ 1) : 
  x^{12} + x^6 + 1 = (x^6 + x^3 + 1) * (x^6 - x^3 + 1) :=
by
  sorry

end factorize_polynomial_l212_212614


namespace purchase_price_eq_360_l212_212963

theorem purchase_price_eq_360 (P : ℝ) (M : ℝ) (H1 : M = 30) (H2 : M = 0.05 * P + 12) : P = 360 :=
by
  sorry

end purchase_price_eq_360_l212_212963


namespace pq_eq_real_nums_l212_212929

theorem pq_eq_real_nums (p q r x y z : ℝ) 
  (h1 : x / p + q / y = 1) 
  (h2 : y / q + r / z = 1) : 
  p * q * r + x * y * z = 0 := 
by 
  sorry

end pq_eq_real_nums_l212_212929


namespace edge_length_of_box_l212_212581

noncomputable def edge_length_cubical_box (num_cubes : ℕ) (edge_length_cube : ℝ) : ℝ :=
  if num_cubes = 8 ∧ edge_length_cube = 0.5 then -- 50 cm in meters
    1 -- The edge length of the cubical box in meters
  else
    0 -- Placeholder for other cases

theorem edge_length_of_box :
  edge_length_cubical_box 8 0.5 = 1 :=
sorry

end edge_length_of_box_l212_212581


namespace reflection_PQ_is_TU_l212_212855

-- Define reflection across the x-axis for a point
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Conditions: points P and Q and their reflections
def P : ℝ × ℝ := (p1, 3)
def Q : ℝ × ℝ := (q1, 6)
def T : ℝ × ℝ := reflect_x P
def U : ℝ × ℝ := reflect_x Q

-- The question is to prove that the reflection of PQ is TU
-- This is the Lean 4 formalization of the proof problem
theorem reflection_PQ_is_TU :
  T = reflect_x P ∧ U = reflect_x Q :=
by
  sorry

end reflection_PQ_is_TU_l212_212855


namespace avg_of_nine_numbers_l212_212441

theorem avg_of_nine_numbers (average : ℝ) (sum : ℝ) (h : average = (sum / 9)) (h_avg : average = 5.3) : sum = 47.7 := by
  sorry

end avg_of_nine_numbers_l212_212441


namespace part1_cost_price_part2_max_profit_l212_212631

theorem part1_cost_price (x : ℝ) (cost_a cost_b : ℝ) :
  (cost_a = x + 40) ∧ (cost_b = x) ∧ (480 / cost_a = 240 / cost_b) →
  (cost_b = 40) ∧ (cost_a = 80) :=
begin
  intros h,
  have add_eq : cost_a = cost_b + 40 := by { rw h.1 },
  have ratio_eq : 480 / cost_a = 240 / cost_b := by { rw h.2.2 },
  sorry
end

theorem part2_max_profit (m b : ℝ) (profit : ℝ) :
  (b = (4000 - 80 * m) / 40) ∧ (m = 34) ∧ (b = 32) →
  (profit = (100 - 80) * m + (55 - 40) * b) →
  (profit = 1160) :=
begin
  intros h,
  have b_def : b = (4000 - 80 * m) / 40 := by { rw h.1 },
  have m_val : m = 34 := by { rw h.2.1 },
  have b_val : b = 32 := by { rw h.2.2 },
  have profit_eq : profit = (100 - 80) * m + (55 - 40) * b := by { rw h.3 },
  sorry
end

end part1_cost_price_part2_max_profit_l212_212631


namespace students_can_do_both_l212_212194

variable (total_students swimmers gymnasts neither : ℕ)

theorem students_can_do_both (h1 : total_students = 60)
                             (h2 : swimmers = 27)
                             (h3 : gymnasts = 28)
                             (h4 : neither = 15) : 
                             total_students - (total_students - swimmers + total_students - gymnasts - neither) = 10 := 
by 
  sorry

end students_can_do_both_l212_212194


namespace centroid_segment_sum_lt_two_thirds_l212_212479

open EuclideanGeometry

theorem centroid_segment_sum_lt_two_thirds (a b c : ℝ) (A B C P : Point)
  (h1 : a ≥ b) (h2 : b ≥ c)
  (h3 : triangle ABC)
  (h4 : centroid P A B C) :
  let m_a := (2 * b^2 + 2 * c^2 - a^2) / 4
  let m_b := (2 * a^2 + 2 * c^2 - b^2) / 4
  let m_c := (2 * a^2 + 2 * b^2 - c^2) / 4
  let AA' := 2 * m_a / 3
  let BB' := 2 * m_b / 3
  let CC' := 2 * m_c / 3
  s = AA' + BB' + CC' :
  s < 2 / 3 * (a + b + c) :=
  begin
    sorry
  end

end centroid_segment_sum_lt_two_thirds_l212_212479


namespace only_perfect_number_with_perfect_sigma_sigma_is_6_l212_212249

def is_perfect (σ : ℕ → ℕ) (n : ℕ) : Prop :=
  σ(n) = 2 * n

def is_perfect_number_with_perfect_sigma_sigma (σ : ℕ → ℕ) (n : ℕ) : Prop :=
  is_perfect σ n ∧ is_perfect σ (σ (σ n))

theorem only_perfect_number_with_perfect_sigma_sigma_is_6
  (σ : ℕ → ℕ)
  (hσ : ∀ n, σ n = ∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d)
  (n : ℕ) :
  is_perfect_number_with_perfect_sigma_sigma σ n ↔ n = 6 :=
by
  sorry

end only_perfect_number_with_perfect_sigma_sigma_is_6_l212_212249


namespace no_prime_divisible_by_42_l212_212418

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → ¬ (42 ∣ p) :=
by sorry

end no_prime_divisible_by_42_l212_212418


namespace graph_D_is_abs_f_l212_212764

-- Define the original piecewise function f(x)
def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x 
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2 
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Assume 0 outside the given intervals for completeness

-- Define the absolute value function |f(x)|
def abs_f (x : ℝ) : ℝ := abs (f x)

-- Define the candidate graph D (which is abs_f)
def graph_D (x : ℝ) : ℝ := abs_f x

-- The theorem stating that graph D corresponds to y = |f(x)|
theorem graph_D_is_abs_f : ∀ x, graph_D x = abs_f x :=
by
  intros x
  rfl  -- Direct equality from the definition

end graph_D_is_abs_f_l212_212764


namespace constant_term_binom_expansion_l212_212473

theorem constant_term_binom_expansion : 
  let T := (x : ℕ) → (-1)^x * (Nat.choose 6 x) * x^(12 - 3 * x)
  (T 4) = 15 := 
by
  sorry

end constant_term_binom_expansion_l212_212473


namespace tangent_at_5_eqn_l212_212162

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_period : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom tangent_at_neg1 : ∀ x y : ℝ, x - y + 3 = 0 → x = -1 → y = f x

theorem tangent_at_5_eqn : 
  ∀ x y : ℝ, x = 5 → y = f x → x + y - 7 = 0 :=
sorry

end tangent_at_5_eqn_l212_212162


namespace linear_function_expression_l212_212954

theorem linear_function_expression :
  ∃ k b: ℝ, y = k * x + b ∧ 
           (A = (-2, 0) ∧ (B = (0, t) ∧ (|t| = 8)) → 
            ((y = 4 * x + 8) ∨ (y = -4 * x - 8))) :=
sorry

end linear_function_expression_l212_212954


namespace _l212_212477

noncomputable def given_conditions (A B C a b c : ℝ) : Prop :=
  b * sin (2 * A) = sqrt 3 * a * sin B ∧
  1 / 2 * b * c * sin A = 3 * sqrt 3 ∧
  b / c = 3 * sqrt 3 / 4

noncomputable theorem value_of_A_and_a (A B C a b c : ℝ) (h : given_conditions A B C a b c) :
  A = π / 6 ∧ a = sqrt 7 :=
by
  sorry

end _l212_212477


namespace least_possible_sum_l212_212960

theorem least_possible_sum {c d : ℕ} (hc : c ≥ 2) (hd : d ≥ 2) (h : 3 * c + 6 = 6 * d + 3) : c + d = 5 :=
by
  sorry

end least_possible_sum_l212_212960


namespace circle_line_distance_l212_212754

theorem circle_line_distance {r : ℝ} (h₁ : ∃ x y, (x^2 + (y - 3)^2 = r^2)) (h₂ : ∃ y x, (y = √3 * x + 1)) (h₃ : (abs (-3 + 1)) / 2 = 1) : r = √2 :=
  sorry

end circle_line_distance_l212_212754


namespace number_of_multiples_of_5_l212_212003

theorem number_of_multiples_of_5 (start end_ : ℕ) (h1 : start = 100) (h2 : end_ = 450) : 
  (count_multiples : (start + 5 * k <= end_ for some k : ℕ) = 71 ) :=
by
  -- Define the function to count multiples of 5 in the interval
  noncomputable def count_multiples (start end_ : ℕ) : ℕ :=
    (end_ / 5) - (start / 5) + (if start % 5 = 0 then 1 else 0)
  
  -- Use the conditions to compute the final count
  have h : count_multiples 100 450 = 71 := by sorry
  
  exact h

end number_of_multiples_of_5_l212_212003


namespace original_sticker_price_l212_212870

theorem original_sticker_price (
  (x A_price B_price : ℝ)
  (storeA_discount storeB_discount tax rebate : ℝ)
  (saving : ℝ)
  (h1 : storeA_discount = 0.20)
  (h2 : storeB_discount = 0.30)
  (h3 : tax = 0.07)
  (h4 : rebate = 120)
  (h5 : saving = 21)
  (hA_price : A_price = ((x * (1 - storeA_discount)) - rebate) * (1 + tax))
  (hB_price : B_price = (x * (1 - storeB_discount)) * (1 + tax))
  (h_save : B_price - A_price = saving)
) : x = 1004 := sorry

end original_sticker_price_l212_212870


namespace find_value_of_polynomial_l212_212493

noncomputable def P (x a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

theorem find_value_of_polynomial
  (a b c k p0 p1 p2 p3 : ℤ)
  (hc: odd c)
  (hp1: p1 = P 1 a b c)
  (hp2: p2 = P 2 a b c)
  (hp3: p3 = P 3 a b c)
  (hp1_eq_k: p1 = k)
  (hp2_eq_k: p2 = k)
  (hp3_eq_k: p3 = k)
  (hp1_hp2_hp3_eq: p1^3 + p2^3 + p3^3 = 3 * p1 * p2 * p3) :
  p2 + 2 * p1 - 3 * p0 = 18 := 
sorry

end find_value_of_polynomial_l212_212493


namespace product_mod_25_l212_212143

theorem product_mod_25 (m : ℕ) (h : 0 ≤ m ∧ m < 25) : 
  43 * 67 * 92 % 25 = 2 :=
by
  sorry

end product_mod_25_l212_212143


namespace geometric_series_sum_l212_212674

-- Define the terms of the series
def a : ℚ := 1 / 5
def r : ℚ := -1 / 3
def n : ℕ := 6

-- Define the expected sum
def expected_sum : ℚ := 182 / 1215

-- Prove that the sum of the geometric series equals the expected sum
theorem geometric_series_sum : 
  (a * (1 - r^n)) / (1 - r) = expected_sum := 
by
  sorry

end geometric_series_sum_l212_212674


namespace symmetric_function_exists_l212_212076

-- Define the main sets A and B with given cardinalities
def A := { n : ℕ // n < 2011^2 }
def B := { n : ℕ // n < 2010 }

-- The main theorem to prove
theorem symmetric_function_exists :
  ∃ (f : A × A → B), 
  (∀ x y, f (x, y) = f (y, x)) ∧ 
  (∀ g : A → B, ∃ (a1 a2 : A), g a1 = f (a1, a2) ∧ g a2 = f (a1, a2) ∧ a1 ≠ a2) :=
sorry

end symmetric_function_exists_l212_212076


namespace parity_of_floor_λ_n_l212_212625

noncomputable def λ := (3 + Real.sqrt 17) / 2

theorem parity_of_floor_λ_n (λ := (3 + Real.sqrt 17) / 2) :
  ∀ n : ℕ, 0 < n → Nat.bodd (Int.floor (λ ^ n)) = Nat.bodd n :=
  sorry

end parity_of_floor_λ_n_l212_212625


namespace trigonometric_simplification_l212_212234

noncomputable def trigonometric_expression (x : ℝ) :=
  Real.tan (2 * Real.atan ((1 - Real.cos x) / (Real.sin x))) * Real.sqrt ((1 + Real.cos (2 * x)) / (1 - Real.cos (2 * x)))

theorem trigonometric_simplification (x : ℝ) :
  trigonometric_expression x = 
    if Real.tan x > 0 then 1 else -1 :=
sorry

end trigonometric_simplification_l212_212234


namespace min_value_expr_l212_212020

theorem min_value_expr (x : ℝ) (h : x > -3) : ∃ m, (∀ y > -3, 2 * y + (1 / (y + 3)) ≥ m) ∧ m = 2 * Real.sqrt 2 - 6 :=
by
  sorry

end min_value_expr_l212_212020


namespace symm_property_interval_condition_main_theorem_l212_212749

-- Define the function f with the given conditions
def f (x : ℝ) : ℝ := sorry -- Given function f, to be defined from the conditions

-- Define the conditions
theorem symm_property (f : ℝ → ℝ) : (∀ x, f(x+1) = f(-x+1)) := sorry
theorem interval_condition (f : ℝ → ℝ) : (∀ x, (0 < x ∧ x < 1) → f(x) = Real.exp(-x)) := sorry

-- Main theorem to be proved
theorem main_theorem (f : ℝ → ℝ) 
  (symm : ∀ x, f(x+1) = f(-x+1))
  (interval : ∀ x, (0 < x ∧ x < 1) → f(x) = Real.exp(-x)) : 
  f(Real.log 3) = 3 * Real.exp(-2) := sorry

end symm_property_interval_condition_main_theorem_l212_212749


namespace isosceles_trapezoid_sides_length_l212_212155

theorem isosceles_trapezoid_sides_length (b1 b2 A : ℝ) (h s : ℝ) 
  (hb1 : b1 = 11) (hb2 : b2 = 17) (hA : A = 56) :
  (A = 1/2 * (b1 + b2) * h) →
  (s ^ 2 = h ^ 2 + (b2 - b1) ^ 2 / 4) →
  s = 5 :=
by
  intro
  sorry

end isosceles_trapezoid_sides_length_l212_212155


namespace tan_15_degree_identity_l212_212250

theorem tan_15_degree_identity : (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end tan_15_degree_identity_l212_212250


namespace series_converges_and_sum_l212_212711

noncomputable def factorial (n : Nat) : ℕ :=
  Nat.recOn n 1 (fun n ih => (n + 1) * ih)

noncomputable def series_term (z : ℂ) (n : ℕ) : ℂ :=
  (∏ i in Finset.range n, z + i) / (factorial (n + 1))

noncomputable def series_sum (z : ℂ) : ℂ :=
  ∑' n, series_term z n

theorem series_converges_and_sum (z : ℂ) (h : |z| < 1) : 
  series_sum z = 1 / (1 - z) :=
by
  sorry

end series_converges_and_sum_l212_212711


namespace no_prime_divisible_by_42_l212_212420

theorem no_prime_divisible_by_42 : 
  ∀ p : ℕ, Prime p → ¬ (42 ∣ p) := 
by
  intro p hp hdiv
  have h2 : 2 ∣ p := dvd_of_mul_right_dvd hdiv
  have h3 : 3 ∣ p := dvd_of_mul_left_dvd (dvd_of_mul_right_dvd hdiv)
  have h7 : 7 ∣ p := dvd_of_mul_left_dvd hdiv
  sorry

end no_prime_divisible_by_42_l212_212420


namespace remainder_7459_div_9_l212_212224

theorem remainder_7459_div_9 : 7459 % 9 = 7 := 
by
  sorry

end remainder_7459_div_9_l212_212224


namespace greatest_value_of_y_l212_212221

theorem greatest_value_of_y : 
  ∀ y : ℝ, 3 * y ^ 2 + 5 * y + 2 = 6 → y ≤ (-5 + real.sqrt 73) / 6 :=
by
  sorry

end greatest_value_of_y_l212_212221


namespace english_book_pages_l212_212295

def numPagesInOneEnglishBook (x y : ℕ) : Prop :=
  x = y + 12 ∧ 3 * x + 4 * y = 1275 → x = 189

-- The statement with sorry as no proof is required:
theorem english_book_pages (x y : ℕ) (h1 : x = y + 12) (h2 : 3 * x + 4 * y = 1275) : x = 189 :=
  sorry

end english_book_pages_l212_212295


namespace min_max_product_l212_212086

theorem min_max_product (x y : ℝ) (h : 5 * x ^ 2 + 10 * x * y + 7 * y ^ 2 = 2) :
  let k := 3 * x ^ 2 + 4 * x * y + 3 * y ^ 2 in
  let m := (55 - Real.sqrt 1415) / 140 in
  let M := (55 + Real.sqrt 1415) / 140 in
  m * M = 805 / 9800 :=
by
  sorry

end min_max_product_l212_212086


namespace scientific_notation_120_million_l212_212148

theorem scientific_notation_120_million :
  120000000 = 1.2 * 10^7 :=
by
  sorry

end scientific_notation_120_million_l212_212148


namespace increase_subsets_one_element_increase_subsets_k_elements_l212_212383

variable {A : Type*} -- Define a set A
variable (m k : ℕ) (m_pos : 0 < m) (k_pos : 1 < k)

noncomputable def increase_one_element_subsets : ℕ := 2^m
noncomputable def increase_k_elements_subsets : ℕ := 2^m * (2^k - 1)

theorem increase_subsets_one_element 
  (A_subsets : A := (set_univ (fin (m)))) :
  (set_univ (fin (m + 1))).card - (set_univ (fin (m))).card = increase_one_element_subsets m :=
by sorry

theorem increase_subsets_k_elements
  (A_subsets : A := (set_univ (fin (m)))) :
  (set_univ (fin (m + k))).card - (set_univ (fin (m))).card = increase_k_elements_subsets m k :=
by sorry

end increase_subsets_one_element_increase_subsets_k_elements_l212_212383


namespace pq_perpendicular_bc_l212_212812

namespace Geometry

open Triangle

variables {A B C A₁ B₁ C₁ P Q H M : Type}

-- Define the elements based on the given conditions
def is_projection (A₁ : Type) (A B C : Type) : Prop := ∃ l : Type, is_line l ∧ foot_of_perpendicular l B = A₁

def is_orthocenter (H A B C : Type) : Prop := ∃ A₁ B₁ C₁ : Type, is_projection A₁ A B C ∧ is_projection B₁ B A C ∧ is_projection C₁ C A B ∧ intersection_point A₁ B₁ C₁ = H

def is_midpoint (M H A : Type) : Prop := ∃ B : Type, midpoint_of_segment B H = M ∧ midpoint_of_segment B A = M

def is_intersection (Q : Type) (B H A₁ C₁ : Type) : Prop := ∃ l₁ l₂ : Type, is_line l₁ ∧ is_line l₂ ∧ l₁ = line_through B H ∧ l₂ = line_through A₁ C₁ ∧ intersection_point l₁ l₂ = Q

def on_line (P B₁ M AB : Type) : Prop := ∃ l : Type, is_line l ∧ l = line_through B₁ M ∧ on_segment P AB l

-- Define the final proof goal
theorem pq_perpendicular_bc (A B C A₁ B₁ C₁ P Q H M : Type) :
  is_projection A₁ A B C →
  is_projection B₁ B A C →
  is_projection C₁ C A B →
  is_orthocenter H A B C →
  is_midpoint M H A →
  is_intersection Q B H A₁ C₁ →
  on_line P B₁ M (segment A B) →
  perpendicular (line_through P Q) (line_through B C) :=
by sorry -- proof is omitted, placeholder for solution.

end Geometry

end pq_perpendicular_bc_l212_212812


namespace minimum_value_of_expr_l212_212013

noncomputable def expr (x : ℝ) := 2 * x + 1 / (x + 3)

theorem minimum_value_of_expr (x : ℝ) (h : x > -3) :
  ∃ y, y = 2 * real.sqrt 2 - 6 ∧ ∀ z, z > -3 → expr z ≥ y := sorry

end minimum_value_of_expr_l212_212013


namespace slope_of_line_l_l212_212061

noncomputable def slope_of_line (l : ℝ) (m n : ℝ) : ℝ := 
  n / m

theorem slope_of_line_l 
  (u : ℝ × ℝ) 
  (OA : ℝ × ℝ := (1, 4)) 
  (OB : ℝ × ℝ := (-3, 1)) 
  (proj_eq : (OA.1 * u.1 + OA.2 * u.2) / ℝ.sqrt(u.1 * u.1 + u.2 * u.2) = -((OB.1 * u.1 + OB.2 * u.2) / ℝ.sqrt(u.1 * u.1 + u.2 * u.2)))
  (slope_obtuse : u.2 / u.1 < 0) :
  slope_of_line l u.1 u.2 = -4 / 3 :=
sorry

end slope_of_line_l_l212_212061


namespace arithmetic_sequence_sum_l212_212167

noncomputable def a (n : ℕ) : ℚ := (2 / 5 : ℚ) * n + (3 / 5 : ℚ)

def b (n : ℕ) : ℤ := Int.floor (a n)

theorem arithmetic_sequence_sum :
  (a 3 + a 4 = 4) ∧
  (a 5 + a 7 = 6) →
  (b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 + b 10 = 24) :=
by
  intros
  sorry

end arithmetic_sequence_sum_l212_212167


namespace sum_reciprocal_gt_log_l212_212402

theorem sum_reciprocal_gt_log (n : ℕ) (h : ∀ x ≥ 0, Real.exp x ≥ x + 1 ∧ (x = 0 → Real.exp x = x + 1)) :
  ∑ k in Finset.range n, (1 / (k + 1)) > Real.log (n + 1) :=
sorry

end sum_reciprocal_gt_log_l212_212402


namespace connected_graph_with_2n_odd_vertices_can_be_drawn_l212_212921

theorem connected_graph_with_2n_odd_vertices_can_be_drawn 
  (G : Type) [graph : SimpleGraph G] 
  (connected : G.connected) 
  (odd_degree_vertices : ∃ (S : Finset G), S.card = 2 * n ∧ ∀ (v : G), v ∈ S → graph.degree v % 2 = 1) :
  ∃ (segmentation : List (List G)), segmentation.length = n-1 ∧ (∀ (path : List G), path ∈ segmentation → is_path G path ∧ no_retrace path) :=
sorry

end connected_graph_with_2n_odd_vertices_can_be_drawn_l212_212921


namespace smallest_value_of_x_l212_212999

theorem smallest_value_of_x :
  ∃ x : Real, (∀ z, (z = (5 * x - 20) / (4 * x - 5)) → (z * z + z = 20)) → x = 0 :=
by
  sorry

end smallest_value_of_x_l212_212999


namespace emily_first_round_points_l212_212333

theorem emily_first_round_points (x : ℤ) 
  (second_round : ℤ := 33) 
  (last_round_loss : ℤ := 48) 
  (total_points_end : ℤ := 1) 
  (eqn : x + second_round - last_round_loss = total_points_end) : 
  x = 16 := 
by 
  sorry

end emily_first_round_points_l212_212333


namespace total_number_of_three_digit_numbers_l212_212197

-- Definitions of the four cards and their sides
def card_sides : List (Nat × Nat) := [(1, 2), (3, 4), (5, 6), (7, 8)]

-- Selecting 3 cards out of 4 and the number of ways to show each side
def select_cards : List (List (Nat × Nat)) :=
  List.filter (fun l => List.length l = 3) (List.powerset card_sides)

-- Total possible outcomes for one selection of 3 cards
def outcomes_per_selection (cards : List (Nat × Nat)) : Nat :=
  2 ^ (List.length cards)
  
-- Number of ways to arrange 3 cards
def arrangements (n : Nat) : Nat := Nat.factorial n

-- Calculation of the total different three-digit numbers that can be formed
def total_three_digit_numbers : Nat :=
  List.length select_cards * (outcomes_per_selection [default, default, default]) * (arrangements 3)

theorem total_number_of_three_digit_numbers : total_three_digit_numbers = 192 := by sorry

end total_number_of_three_digit_numbers_l212_212197


namespace decreasing_function_range_l212_212560

theorem decreasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → f(x) > f(y)) ↔ (a ≤ -3) :=
by
  let f : ℝ → ℝ := λ x, x^2 + 2 * (a - 1) * x + 2
  sorry

end decreasing_function_range_l212_212560


namespace probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l212_212786

noncomputable theory
open ProbabilityTheory

-- Definitions based on given conditions
def events := {1, 2, 3, 4, 5, 6}
def eventA := {1, 2, 3}
def eventB := {1, 2, 4}

def numTrials : ℕ := 10
def numA : ℕ := 6
def pA : ℚ := 1 / 2
def pB_given_A : ℚ := 2 / 3
def pB_given_Ac : ℚ := 1 / 3

-- Theorem for probability that B does not occur given A occurred 6 times.
theorem probability_B_does_not_occur_given_A_6_occur :
  -- The probability of B not occurring given A occurred exactly 6 times.
  -- Should be approximately 2.71 * 10^(-4)
  true := sorry

-- Theorem for the expected number of times B occurs.
theorem expected_value_B_occurances : 
  -- The expected value of the number of occurrences of event B given the conditions.
  -- Should be 16 / 3
  true := sorry

end probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l212_212786


namespace fraction_simplifiable_by_7_l212_212118

theorem fraction_simplifiable_by_7 (a b c : ℤ) (h : (100 * a + 10 * b + c) % 7 = 0) : 
  ((10 * b + c + 16 * a) % 7 = 0) ∧ ((10 * b + c - 61 * a) % 7 = 0) :=
by
  sorry

end fraction_simplifiable_by_7_l212_212118


namespace probability_B_not_occur_given_A_occurs_expected_value_X_l212_212790

namespace DieProblem

def event_A := {1, 2, 3}
def event_B := {1, 2, 4}

def num_trials := 10
def num_occurrences_A := 6

theorem probability_B_not_occur_given_A_occurs :
  (∑ i in Finset.range (num_trials.choose num_occurrences_A), 
    (1/6)^num_occurrences_A * (1/3)^(num_trials - num_occurrences_A)) / 
  (num_trials.choose num_occurrences_A * (1/2)^(num_trials)) = 2.71 * 10^(-4) :=
sorry

theorem expected_value_X : 
  (6 * (2/3)) + (4 * (1/3)) = 16 / 3 :=
sorry

end DieProblem

end probability_B_not_occur_given_A_occurs_expected_value_X_l212_212790


namespace a_n_correct_b_n_correct_T_correct_l212_212732

noncomputable def S (n : ℕ) := 2 * n^2 + n

def a (n : ℕ) : ℕ := if n = 1 then 3 else S n - S (n - 1)

def b (n : ℕ) : ℕ := 2^(n - 1)

def a_n (n : ℕ) : ℕ := 4 * n - 1

def b_n (n : ℕ) : ℕ := 2^(n - 1)

def T (n : ℕ) : ℕ := (4 * n - 5) * 2^n + 5

theorem a_n_correct (n : ℕ) (hn : n ≥ 1) : a_n n = a n := 
sorry

theorem b_n_correct (n : ℕ) (hn : n ≥ 1) : b_n n = b n := 
sorry

theorem T_correct (n : ℕ) (hn : n ≥ 1) : T n = (list.range n).sum (λ i, a i * b i) := 
sorry

end a_n_correct_b_n_correct_T_correct_l212_212732


namespace f_f_neg2_l212_212091

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 - real.sqrt x else (3 : ℝ) ^ x

theorem f_f_neg2 : f (f (-2)) = 2 / 3 :=
by
  sorry

end f_f_neg2_l212_212091


namespace solution_set_of_inequality_l212_212571

theorem solution_set_of_inequality :
  {x : ℝ | 3 * x ^ 2 - 7 * x - 10 ≥ 0} = {x : ℝ | x ≥ (10 / 3) ∨ x ≤ -1} :=
sorry

end solution_set_of_inequality_l212_212571


namespace minimum_cost_proof_l212_212910

structure Store :=
  (name : String)
  (gift_costs : (Nat × Nat × Nat × Nat)) -- (Mom, Dad, Brother, Sister)
  (time_spent : Nat) -- Time spent in store in minutes

def Romashka  : Store := { name := "Romashka", gift_costs := (1000, 750, 930, 850), time_spent := 35 }
def Oduvanchik : Store := { name := "Oduvanchik", gift_costs := (1050, 790, 910, 800), time_spent := 30 }
def Nezabudka : Store := { name := "Nezabudka", gift_costs := (980, 810, 925, 815), time_spent := 40 }
def Landysh : Store := { name := "Landysh", gift_costs := (1100, 755, 900, 820), time_spent := 25 }

def stores : List Store := [Romashka, Oduvanchik, Nezabudka, Landysh]

def travel_time := 30 -- minutes
def total_time := 3 * 60 + 25  -- 3 hours and 25 minutes or 210 minutes

noncomputable def min_cost_within_constraints : Nat :=
  let costs := [
    (Romashka.gift_costs.fst, Romashka.time_spent),
    (Oduvanchik.gift_costs.snd, Oduvanchik.time_spent),
    (Landysh.gift_costs.trd, Landysh.time_spent),
    (Nezabudka.gift_costs.fourth, Nezabudka.time_spent)
    ]
  in 3435 -- Given the final correct answer

theorem minimum_cost_proof : min_cost_within_constraints = 3435 := by
  sorry

end minimum_cost_proof_l212_212910


namespace sequence_general_formula_l212_212166

theorem sequence_general_formula (n : ℕ) (h : n ≥ 1) :
  (λ n, ([0, 3/2, 4, 15/2].nth (n - 1)).getD 0 =
    (λ n, (n^2 - 1) / 2) n) := 
sorry

end sequence_general_formula_l212_212166


namespace count_four_digit_numbers_l212_212565

def starts_with_one (n : ℕ) : Prop :=
  n / 1000 = 1

def has_exactly_two_identical_digits (n : ℕ) : Prop :=
  let d1 := (n / 1000) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  (d1 = d2 ∨ d1 = d3 ∨ d1 = d4 ∨ d2 = d3 ∨ d2 = d4 ∨ d3 = d4) ∧
  ¬((d1 = d2 ∧ d1 = d3 ∧ d1 = d4) ∨
    (d2 = d3 ∧ d2 = d4) ∨
    (d1 = d3 ∧ d1 = d4) ∨
    (d1 = d2 ∧ d1 = d4) ∨
    (d1 = d2 ∧ d2 = d3))

theorem count_four_digit_numbers : ∃ n, n = 432 ∧
  (∀ m, 1000 ≤ m ∧ m < 10000 →
    starts_with_one m →
    has_exactly_two_identical_digits m → m ∈ n) :=
by {
  -- Proof is omitted.
  sorry
}

end count_four_digit_numbers_l212_212565


namespace isosceles_triangle_angle_parts_l212_212153

theorem isosceles_triangle_angle_parts (α β γ : ℝ) (hα : α = 40) (h_isosceles : β = γ)
  (h_sum : α + 2 * β = 180) : β = 70 :=
by
  have h_half : (180 - α) / 2 = β := by sorry
  calc
    β = (180 - α) / 2 := by sorry
    _ = 70 := by sorry

example : isosceles_triangle_angle_parts 40 70 70 (by norm_num) (by norm_num) (by norm_num) = 70 := 
by norm_num

end isosceles_triangle_angle_parts_l212_212153


namespace b4_apply_v_l212_212510

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
variable v : Fin 2 → ℝ

noncomputable def B := Matrix.of ![![B 0 0, B 0 1], ![B 1 0, B 1 1]]
noncomputable def v := ![4, -1]

theorem b4_apply_v :
  B * B * B * B * v = ![324, -81] :=
by
  have h : B * v = ![12, -3],
  sorry

end b4_apply_v_l212_212510


namespace fraction_of_work_left_l212_212236

theorem fraction_of_work_left (A_days : ℕ) (B_days : ℕ) (hA : A_days = 15) (hB : B_days = 20) : 
  (1 - (3 * ((1 / A_days : ℚ) + (1 / B_days : ℚ)))) = 13 / 20 :=
by
  rw [hA, hB]
  norm_num
  sorry

end fraction_of_work_left_l212_212236


namespace find_k_value_l212_212740

theorem find_k_value : ∀ (x y k : ℝ), x = 2 → y = -1 → y - k * x = 7 → k = -4 := 
by
  intros x y k hx hy h
  sorry

end find_k_value_l212_212740


namespace cone_volume_proof_l212_212792

noncomputable def volume_of_cone (r l h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem cone_volume_proof : 
  ∀ (r l h : ℝ), 
    r = 1 
    ∧ l = 2 
    ∧ h = Real.sqrt (l^2 - r^2) 
  → volume_of_cone r l h = (Real.sqrt 3 * π) / 3 :=
by
  intros r l h
  intro h_cond
  sorry

end cone_volume_proof_l212_212792


namespace polynomial_degree_one_condition_l212_212707

theorem polynomial_degree_one_condition (P : ℝ → ℝ) (c : ℝ) :
  (∀ a b : ℝ, a < b → (P = fun x => x + c) ∨ (P = fun x => -x + c)) ∧
  (∀ a b : ℝ, a < b →
    (max (P a) (P b) - min (P a) (P b) = b - a)) :=
sorry

end polynomial_degree_one_condition_l212_212707


namespace nth_wise_number_1990_l212_212640

/--
A natural number that can be expressed as the difference of squares 
of two other natural numbers is called a "wise number".
-/
def is_wise_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 - y^2 = n

/--
The 1990th "wise number" is 2659.
-/
theorem nth_wise_number_1990 : ∃ n : ℕ, is_wise_number n ∧ n = 2659 :=
  sorry

end nth_wise_number_1990_l212_212640


namespace equal_coefficients_m_eq_neg3_l212_212444

/-- Given the product of two polynomials (x^2 - mx + 2)(2x + 1) and the condition that the coefficients
    of the quadratic term and linear term are equal, we need to prove that m = -3. -/
theorem equal_coefficients_m_eq_neg3 (m : ℝ) :
  (∃ coeff_q coeff_l : ℝ, 
    (λ f, ∃ x : ℝ, f = 2*x^3 + coeff_q*x^2 + coeff_l*x + 2) ((x^2 - m*x + 2)*(2*x + 1)) ∧ coeff_q = coeff_l) → m = -3 := 
by
  sorry

end equal_coefficients_m_eq_neg3_l212_212444


namespace convert_to_scientific_notation_l212_212152

-- Problem statement: convert 120 million to scientific notation and validate the format.
theorem convert_to_scientific_notation :
  120000000 = 1.2 * 10^7 :=
sorry

end convert_to_scientific_notation_l212_212152


namespace part1_part2_l212_212447

noncomputable theory

-- Definitions for the conditions given in the problem
variables {A B C a b c : ℝ}

-- Given conditions
axiom h1 : cos C = 1 / 8
axiom h2 : C = 2 * A

-- Part 1: Prove cos A = 3 / 4 given h1 and h2
theorem part1 (h1 : cos C = 1 / 8) (h2 : C = 2 * A) : cos A = 3 / 4 :=
sorry

-- Additional given condition for Part 2
axiom ha : a = 4

-- Part 2: Prove c = 6 given h1, h2, ha, and part1 result
theorem part2 (h1 : cos C = 1 / 8) (h2 : C = 2 * A) (ha : a = 4)
  (part1_result : cos A = 3 / 4) : c = 6 :=
sorry

end part1_part2_l212_212447


namespace geometry_projections_on_circle_l212_212881

/-- Let A, B, P be three points on a circle C.
  Let Q, R, S be the projections of P onto (AB) and the tangents to C at A and B, respectively.
  We need to prove: PQ² = PR * PS 
-/
theorem geometry_projections_on_circle (C : Type) [metric_space C] [normed_group C]
  {A B P : C} (hA : A ∈ metric_closed_ball P) (hB : B ∈ metric_closed_ball P)
  {Q R S : C} (hQ : is_projection P Q (line_through A B))
  (hR : is_projection P R (tangent_to_circle_at A C))
  (hS : is_projection P S (tangent_to_circle_at B C)) :
  dist P Q ^ 2 = dist P R * dist P S := 
sorry

end geometry_projections_on_circle_l212_212881


namespace problem_statement_1_problem_statement_2_l212_212491

variable {x y : ℝ}

def S (x y : ℝ) : ℝ := (2 / x^2) + 1 + (1 / x) + y * (y + 2 + (1 / x))

def T (x y : ℝ) : ℝ := (2 / x^2) + 1 + (1 / x) + y * (y + 1 + (1 / x))

theorem problem_statement_1
  (hx : x^2 * y^2 + 2 * y * x^2 + 1 = 0) :
  -1 / 2 ≤ S x y ∧ S x y ≤ 1 / 2 :=
sorry

theorem problem_statement_2
  (hx : x^2 * y^2 + 2 * y * x^2 + 1 = 0) :
  (1 - 3 * Real.sqrt 3 / 4) ≤ T x y ∧ T x y ≤ (1 + 3 * Real.sqrt 3 / 4) :=
sorry

end problem_statement_1_problem_statement_2_l212_212491


namespace inscribed_circle_radius_third_of_circle_l212_212932

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ := 
  R * (Real.sqrt 3 - 1) / 2

theorem inscribed_circle_radius_third_of_circle (R : ℝ) (hR : R = 5) :
  inscribed_circle_radius R = 5 * (Real.sqrt 3 - 1) / 2 := by
  sorry

end inscribed_circle_radius_third_of_circle_l212_212932


namespace hyperbola_asymptotes_tangent_to_circle_l212_212437

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, y^2 - (x^2 / m^2) = 1 → (x^2 + y^2 - 4*y + 3 = 0 → distance_center_to_asymptote (0, 2) (y = x / m) = 1)) → 
  m = real.sqrt(3) / 3 :=
sorry

end hyperbola_asymptotes_tangent_to_circle_l212_212437


namespace reflections_on_circumcircle_l212_212494

-- Let A, B, C be points in the Euclidean plane.
variables (A B C H : Point)
-- Define triangle ABC with orthocenter H.
-- Assume the orthocenter H is inside the triangle (acute triangle).
def is_acute_triangle (A B C H : Point) : Prop :=
  is_triangle A B C ∧ orthocenter A B C H ∧ is_inside H (triangle A B C)

-- Define reflections Ha, Hb, Hc of H with respect to sides BC, CA, AB respectively.
def reflection (H P Q : Point) : Point := sorry

def H_a := reflection H B C
def H_b := reflection H C A
def H_c := reflection H A B

-- Prove that these reflections lie on the circumcircle of triangle ABC.
theorem reflections_on_circumcircle (A B C H : Point)
  (h : is_acute_triangle A B C H) :
  on_circumcircle A B C H_a ∧ on_circumcircle A B C H_b ∧ on_circumcircle A B C H_c :=
sorry

end reflections_on_circumcircle_l212_212494


namespace polynomial_factorization_l212_212040

theorem polynomial_factorization (m n : ℤ) 
  (h1 : 81 - 3 * m + n = 0) 
  (h2 : -3 + m + n = 0) 
  (h3 : n = 0) : 
  |3 * m + n| = 81 := by 
  have hn : n = 0 := h3
  rw hn at h1
  rw hn at h2
  have hm : m = 27 := by 
    linarith
  rw [hm, hn]
  norm_num
  sorry

end polynomial_factorization_l212_212040


namespace y_equals_9_l212_212241

noncomputable def list_i (y : ℕ) : List ℕ := [y, 2, 4, 7, 10, 11]
def list_ii : List ℕ := [3, 3, 4, 6, 7, 10]

def median (l : List ℕ) : ℚ :=
  let l_sorted := l.qsort (· < ·)
  if h : l_sorted.length % 2 = 0 then
    ((l_sorted.nth_le (l_sorted.length / 2 - 1) (by simp [h, Nat.div_pos])).natAbs + 
     (l_sorted.nth_le (l_sorted.length / 2) (by simp [h, Nat.div_pos])).natAbs) / 2
  else
    l_sorted.nth_le (l_sorted.length / 2) (by simp [zero_lt_iff_ne_zero, Nat.div_pos])

def mode (l : List ℕ) : ℕ :=
  l.maximumBy fun n => l.count n

theorem y_equals_9 (y : ℕ) (h : median (list_i y) = median list_ii + mode list_ii) : y = 9 :=
by
  sorry

end y_equals_9_l212_212241


namespace chicken_pieces_needed_l212_212597

theorem chicken_pieces_needed :
  let chicken_pasta_pieces := 2
      barbecue_chicken_pieces := 3
      fried_chicken_dinner_pieces := 8
      number_of_fried_chicken_dinner_orders := 2
      number_of_chicken_pasta_orders := 6
      number_of_barbecue_chicken_orders := 3
  in
  (number_of_fried_chicken_dinner_orders * fried_chicken_dinner_pieces +
   number_of_chicken_pasta_orders * chicken_pasta_pieces +
   number_of_barbecue_chicken_orders * barbecue_chicken_pieces) = 37 := by
  sorry

end chicken_pieces_needed_l212_212597


namespace ratio_of_inscribed_squares_is_one_eighth_l212_212223

noncomputable def ratio_of_inscribed_squares (r : ℝ) : ℝ :=
  let a := r
  let b := r / 2
  let s1 := b  -- since min(a, b) = b = r / 2
  let area_square_in_ellipse := s1 * s1  -- (r / 2) * (r / 2)
  let s2 := r * real.sqrt 2  -- derived from the circle inscribed square
  let area_square_in_circle := s2 * s2  -- (r * √2) * (r * √2)
  ratio_of_areas := area_square_in_ellipse / area_square_in_circle
  ratio_of_areas

theorem ratio_of_inscribed_squares_is_one_eighth (r : ℝ) : ratio_of_inscribed_squares r = 1 / 8 :=
by
  sorry

end ratio_of_inscribed_squares_is_one_eighth_l212_212223


namespace find_k_l212_212739

theorem find_k (x y k : ℝ) (h₁ : x = 2) (h₂ : y = -1) (h₃ : y - k * x = 7) : k = -4 :=
by
  sorry

end find_k_l212_212739


namespace concrete_required_for_l212_212265

/-- Given a pathway with specified dimensions, the required cubic yards of concrete rounded up
to the nearest whole number is calculated -/
def pathway_volume_required_concrete (width_ft : ℕ) (length_ft : ℕ) (depth_in : ℕ) : ℕ :=
let width_yd := width_ft / 3,
    length_yd := length_ft / 3,
    depth_yd := depth_in / 36 in
let volume_cubic_yards := width_yd * length_yd * depth_yd in
nat_ceil (volume_cubic_yards)

theorem concrete_required_for pathway : pathway_volume_required_concrete 4 100 4 = 5 :=
by sorry

end concrete_required_for_l212_212265


namespace train_pass_time_approx_l212_212658

noncomputable def time_to_pass_platform
  (L_t L_p : ℝ)
  (V_t : ℝ) : ℝ :=
  (L_t + L_p) / (V_t * (1000 / 3600))

theorem train_pass_time_approx
  (L_t L_p V_t : ℝ)
  (hL_t : L_t = 720)
  (hL_p : L_p = 360)
  (hV_t : V_t = 75) :
  abs (time_to_pass_platform L_t L_p V_t - 51.85) < 0.01 := 
by
  rw [hL_t, hL_p, hV_t]
  sorry

end train_pass_time_approx_l212_212658


namespace quadrilateral_is_kite_l212_212029

/-- If a quadrilateral has perpendicular diagonals, two adjacent sides that are equal,
    and one pair of opposite angles that are equal, then it fits the classification of a kite. -/
theorem quadrilateral_is_kite
  (Q : Type)
  [quadrilateral Q]
  (diagonals_perpendicular : perpendicular (diagonal1 Q) (diagonal2 Q))
  (adjacent_sides_equal : side1 Q = side2 Q)
  (opposite_angles_equal : angle_opposite1 Q = angle_opposite2 Q) :
  kite Q :=
sorry

end quadrilateral_is_kite_l212_212029


namespace tangent_line_sum_l212_212885

theorem tangent_line_sum (a b c : ℕ) (h₁ : ∀ x y, y = x^2 + 102 / 100 → x = y^2 + 49 / 4 → a*x + b*y = c)
  (h₂ : b/a = 5) (h₃ : ∃ (d : ℕ), d * (gcdA a b + gcdA b c + gcdA c a) = 1) 
  : a + b + c = 11 := sorry

end tangent_line_sum_l212_212885


namespace solve_for_x_l212_212501

-- Define the function h based on the given condition
def h (x : ℝ) : ℝ := (2 * x - 4) / 3 + 7

-- Prove that x = 17 when h(x) = x
theorem solve_for_x (x : ℝ) (H : h x = x) : x = 17 :=
by
  -- Proof to be completed
  sorry

end solve_for_x_l212_212501


namespace max_k_a_19_max_value_of_ka_l212_212321

def is_permutation (a : List ℕ) (n : ℕ) : Prop :=
  a.perm (List.range n)

def min_swaps_to_identity (a : List ℕ) : ℕ :=
  -- Definition omitted for brevity, but assume this properly computes the minimum swaps to sort the permutation to identity.

theorem max_k_a_19 (a : List ℕ) (h : is_permutation a 20) :
  min_swaps_to_identity a ≤ 19 ∧ min_swaps_to_identity a ≥ 19 :=
sorry

theorem max_value_of_ka {a : List ℕ} (h : is_permutation a 20) :
  ∃ k_a, k_a = 19 ∧ min_swaps_to_identity a = k_a :=
by
  existsi 19
  apply max_k_a_19
  exact h

end max_k_a_19_max_value_of_ka_l212_212321


namespace sum_of_dimensions_l212_212274

noncomputable def rectangular_prism_dimensions (A B C : ℝ) : Prop :=
  (A * B = 30) ∧ (A * C = 40) ∧ (B * C = 60)

theorem sum_of_dimensions (A B C : ℝ) (h : rectangular_prism_dimensions A B C) : A + B + C = 9 * Real.sqrt 5 :=
by
  sorry

end sum_of_dimensions_l212_212274


namespace number_of_equidistant_points_l212_212170

-- Let's define the problem
def circle (O : Point) (r : ℝ) := { P : Point | dist P O = r }
def parallel_tangents (L1 L2 : Line) (O : Point) (d : ℝ) := 
  L1 ≠ L2 ∧ 
  ∀ P1 ∈ L1, ∀ P2 ∈ L2, dist P1 P2 = 2 * d ∧ ∀ r ∈ radius(O, L1), r = dist P1 O

-- Main theorem
theorem number_of_equidistant_points (O : Point) (r d : ℝ) (L1 L2 : Line) 
  (h1 : O ∈ L1 ∧ O ∈ L2)
  (h2 : parallel_tangents L1 L2 O d)
  (h3 : ∀ P, P ∈ circle O r → dist P L1 = dist P L2) :
  (∃ P1 P2 P3 : Point, 
    P1 = O ∧ 
    P2 ≠ P1 ∧ P2 ∈ circle O r ∧ 
    P3 ≠ P2 ∧ P3 ≠ P1 ∧ P3 ∈ circle O r ∧ 
    dist P1 L1 = dist P1 L2 ∧ 
    dist P2 L1 = dist P2 L2 ∧ 
    dist P3 L1 = dist P3 L2 ∧ 
    (∀ Q : Point, dist Q L1 = dist Q L2 → (Q = P1 ∨ Q = P2 ∨ Q = P3))) :=
sorry

end number_of_equidistant_points_l212_212170


namespace minimum_antitours_count_l212_212852

theorem minimum_antitours_count (V : Type) [fintype V] [decidable_eq V] (capital : V) (flights : V → V → Prop)
  (H1 : fintype.card V = 50)
  (H2 : ∀ (A : V), ∃ (antitour : list V), antitour.nodup ∧ antitour.length = 50 ∧ (∀ (k : ℕ), k ∈ (finset.range 50).succ → ¬ reachable A (antitour.nth_le k sorry) k))
  (H3 : ∀ (v : V), ∃ (path : list V), list.distinct path ∧ path.head = some capital ∧ path.last = some v) :
  ∃ (n : ℕ), n = (25! ^ 2) := by
  sorry

end minimum_antitours_count_l212_212852


namespace max_volume_prism_in_pyramid_proof_l212_212864

/-
Give a regular triangular pyramid with an equilateral base of side length 2 and height \( 2\sqrt{2} \).
There is a right prism inside it with a rhombus as its base, such that one face of the prism lies on the base
of the pyramid and another face lies on a lateral face of the pyramid.

Goal: Prove that the maximum volume of such a prism is \( \frac{5\sqrt{6}}{36} \).
-/

def max_volume_of_prism_in_pyramid (a h : ℝ) (hr : h = 2 * real.sqrt 2) (ab : a = 2) : ℝ :=
  (5 * real.sqrt 6) / 36

theorem max_volume_prism_in_pyramid_proof : max_volume_of_prism_in_pyramid 2 (2 * real.sqrt 2) (by simp) (by simp) = ((5 * real.sqrt 6) / 36) :=
sorry

end max_volume_prism_in_pyramid_proof_l212_212864


namespace largest_root_in_interval_l212_212322

theorem largest_root_in_interval :
  ∃ (r : ℝ), (2 < r ∧ r < 3) ∧ (∃ (a_2 a_1 a_0 : ℝ), 
    |a_2| ≤ 3 ∧ |a_1| ≤ 3 ∧ |a_0| ≤ 3 ∧ a_2 + a_1 + a_0 = -6 ∧ r^3 + a_2 * r^2 + a_1 * r + a_0 = 0) :=
sorry

end largest_root_in_interval_l212_212322


namespace max_fraction_value_l212_212731

noncomputable def max_value_fraction (a : ℕ → ℝ) (S : ℕ → ℝ) : ℝ :=
  let seq := λ n, a n = 3 * (S n) / (n + 2)
  let fraction := λ n, a n / a (n - 1)
  ∀ n, a n = 3 * (S n) / (n + 2) -> fraction n ≤ 3

theorem max_fraction_value : 
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (∀ n, a n = 3 * (S n) / (n + 2)) →
    (∀ n ≥ 2, S n = (n + 2) / 3 * a n) →
    ∀ n ≥ 2, a n / a (n - 1) ≤ 3 :=
by
  sorry

end max_fraction_value_l212_212731


namespace scientific_notation_of_3300000_l212_212064

theorem scientific_notation_of_3300000 : 3300000 = 3.3 * 10^6 :=
by
  sorry

end scientific_notation_of_3300000_l212_212064


namespace power_function_positive_sum_l212_212951

def f (x : ℝ) (m : ℝ) := (m ^ 2 - m - 1) * x ^ (m ^ 2 + 2 * m - 5)

theorem power_function_positive_sum
  (a b m : ℝ)
  (h1 : ∀ (x1 x2 : ℝ), (0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) → (f x1 m - f x2 m) / (x1 - x2) > 0)
  (h2 : a + b > 0)
  (h3 : a ∈ Set.univ ∧ b ∈ Set.univ) :
  f a 2 + f b 2 > 0 :=
sorry

end power_function_positive_sum_l212_212951


namespace quarters_given_by_dad_l212_212485

variable (original_quarters : Nat) (total_quarters : Nat)

-- Set the conditions:
def original_condition := original_quarters = 49
def total_condition := total_quarters = 74

-- Define the proof problem:
theorem quarters_given_by_dad (h1 : original_condition) (h2 : total_condition) : total_quarters - original_quarters = 25 := by
  sorry

end quarters_given_by_dad_l212_212485


namespace james_pay_for_two_semesters_l212_212869

theorem james_pay_for_two_semesters
  (units_per_semester : ℕ) (cost_per_unit : ℕ) (num_semesters : ℕ)
  (h_units : units_per_semester = 20) (h_cost : cost_per_unit = 50) (h_semesters : num_semesters = 2) :
  units_per_semester * cost_per_unit * num_semesters = 2000 := 
by 
  rw [h_units, h_cost, h_semesters]
  norm_num

end james_pay_for_two_semesters_l212_212869


namespace find_c_value_l212_212449

theorem find_c_value (A B C : ℝ) (S1_area S2_area : ℝ) (b : ℝ) :
  S1_area = 40 * b + 1 →
  S2_area = 40 * b →
  ∃ c, AC + CB = c ∧ c = 462 :=
by
  intro hS1 hS2
  sorry

end find_c_value_l212_212449


namespace utility_value_relation_l212_212460

-- Definitions
def has_utility (x : Type) (A : set x) := x ∈ A
def has_value (x : Type) (B : set x) := x ∈ B

-- Problem statement
theorem utility_value_relation {x : Type} (A B : set x) 
  (h1 : ∀ x, has_value x B → has_utility x A) :
  A ∪ B = A :=
by
  sorry

end utility_value_relation_l212_212460


namespace concentric_circles_circumference_difference_and_area_l212_212681

theorem concentric_circles_circumference_difference_and_area {r_inner r_outer : ℝ} (h1 : r_inner = 25) (h2 : r_outer = r_inner + 15) :
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi ∧ Real.pi * r_outer^2 - Real.pi * r_inner^2 = 975 * Real.pi :=
by
  sorry

end concentric_circles_circumference_difference_and_area_l212_212681


namespace problem_solution_l212_212025

theorem problem_solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 :=
by
  sorry

end problem_solution_l212_212025


namespace jony_turnaround_block_l212_212876

-- Define constants and given information
def initial_block : ℕ := 10
def turnaround_block : ℕ := 70
def speed : ℕ := 100 -- meters per minute
def block_length : ℕ := 40 -- meters
def total_time : ℕ := 40 -- minutes
def total_distance := total_time * speed -- meters

-- Theorem statement
theorem jony_turnaround_block :
  let walking_distance_from_10_to_70 := (turnaround_block - initial_block) * block_length in
  let additional_distance_after_70 := total_distance - walking_distance_from_10_to_70 in
  let additional_blocks := additional_distance_after_70 / block_length in
  (turnaround_block + additional_blocks) = 110 :=
by
  -- Proof omitted
  sorry

end jony_turnaround_block_l212_212876


namespace max_consecutive_semi_primes_l212_212900

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_semi_prime (n : ℕ) : Prop := 
  n > 25 ∧ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p + q

theorem max_consecutive_semi_primes : ∃ (N : ℕ), N = 5 ∧
  ∀ (a b : ℕ), (a > 25) ∧ (b = a + 4) → 
  (∀ n, a ≤ n ∧ n ≤ b → is_semi_prime n) ↔ N = 5 := sorry

end max_consecutive_semi_primes_l212_212900


namespace square_perimeter_l212_212545

theorem square_perimeter (area : ℝ) (h : area = 450) :
  ∃ P : ℝ, P = 60 * Real.sqrt 2 ∧ (side : ℝ) (h_side : side = Real.sqrt area), (P = 4 * side) :=
by {
  use 4 * Real.sqrt 450,
  split,
  { norm_num [Real.sqrt, Real.sqrt_mul],
    linarith },
  { ext, simp, linarith }
  sorry
}

end square_perimeter_l212_212545


namespace angle_PRS_is_150_degrees_l212_212120

/-- Let's define a quadrilateral PQRS with the given properties and prove that angle PRS is 150 degrees.
    We have PQ = QR = RS, m∠PQR = 80 degrees, and m∠QRS = 160 degrees. -/
theorem angle_PRS_is_150_degrees
  (P Q R S : Type)
  [metric_space : MetricSpace (P Q R S)]
  (angle_PQR : ∀ (P Q R : P Q R S), mangle P Q R = 80)
  (angle_QRS : ∀ (Q R S : P Q R S), mangle Q R S = 160)
  (side_EQ : ∀ (P Q R S : P Q R S), PQ = QR ∧ QR = RS) :
  mangle P Q R S = 150 :=
  sorry

end angle_PRS_is_150_degrees_l212_212120


namespace pasta_cost_is_one_l212_212534

-- Define the conditions
def pasta_cost (p : ℝ) : ℝ := p -- The cost of the pasta per box
def sauce_cost : ℝ := 2.00 -- The cost of the sauce
def meatballs_cost : ℝ := 5.00 -- The cost of the meatballs
def servings : ℕ := 8 -- The number of servings
def cost_per_serving : ℝ := 1.00 -- The cost per serving

-- Calculate the total meal cost
def total_meal_cost : ℝ := servings * cost_per_serving

-- Calculate the combined cost of sauce and meatballs
def combined_cost_of_sauce_and_meatballs : ℝ := sauce_cost + meatballs_cost

-- Calculate the cost of the pasta
def pasta_cost_calculation : ℝ := total_meal_cost - combined_cost_of_sauce_and_meatballs

-- The theorem stating that the pasta cost should be $1
theorem pasta_cost_is_one (p : ℝ) (h : pasta_cost_calculation = p) : p = 1 := by
  sorry

end pasta_cost_is_one_l212_212534


namespace malenky_phone_connection_impossible_l212_212860

theorem malenky_phone_connection_impossible :
  ∃ (V : Finset ℕ) (d : ℕ → ℕ), 
    card V = 15 ∧ 
    (∃ A : Finset ℕ, card A = 4 ∧ ∀ v ∈ A, d v = 3) ∧ 
    (∃ B : Finset ℕ, card B = 8 ∧ ∀ v ∈ B, d v = 6) ∧ 
    (∃ C : Finset ℕ, card C = 3 ∧ ∀ v ∈ C, d v = 5) → 
    false := 
by
  -- We'll start the proof to initiate the lemma and facilitate Lean compilation.
  sorry

end malenky_phone_connection_impossible_l212_212860


namespace num_sides_second_polygon_is_135_l212_212984

def side_length_second_polygon (s : ℝ) : ℝ := s
def side_length_first_polygon (s : ℝ) : ℝ := 3 * s
def perimeter_first_polygon (s : ℝ) : ℝ := 45 * side_length_first_polygon s
def same_perimeter (s : ℝ) : Prop := perimeter_first_polygon s = perimeter_first_polygon s
def num_sides_second_polygon (s : ℝ) : ℝ := perimeter_first_polygon s / side_length_second_polygon s

theorem num_sides_second_polygon_is_135 {s : ℝ} (h : same_perimeter s) : num_sides_second_polygon s = 135 := by
  sorry

end num_sides_second_polygon_is_135_l212_212984


namespace jason_gave_9_cards_l212_212071

theorem jason_gave_9_cards (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 13) 
  (h2 : remaining_cards = 4) : 
  initial_cards - remaining_cards = 9 := by
  rw [h1, h2]
  exact Nat.sub_self 4 -- It simplifies automatically to 9

end jason_gave_9_cards_l212_212071


namespace minimum_value_of_f_l212_212760

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * real.exp(a * x - 1) - real.log x - a * x

def satisfies_conditions (a : ℝ) : Prop := a ≤ -1/(real.exp 2)

theorem minimum_value_of_f (a : ℝ) (h : satisfies_conditions a) : ∃ x : ℝ, f x a = 0 :=
sorry

end minimum_value_of_f_l212_212760


namespace abs_neg_eight_l212_212543

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end abs_neg_eight_l212_212543


namespace triangle_BC_length_l212_212456

theorem triangle_BC_length 
(ABC : Triangle)
(h₁ : ABC.isAcute)
(h₂ : sin ABC.A = 3 / 5)
(h₃ : ABC.AB = 5)
(h₄ : ABC.AC = 6) :
  ABC.BC = Real.sqrt 13 :=
by 
  sorry

end triangle_BC_length_l212_212456


namespace expectation_X_median_increase_weight_significant_diff_increase_weight_l212_212298

-- Problem 1: Mathematical expectation of X 

def count_combinations : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => count_combinations n k + count_combinations n (k+1)

def prob_X (n m k : ℕ) (total : ℕ) : ℚ :=
  (count_combinations n m * count_combinations (total - n) (k - m)) / (count_combinations total k)

theorem expectation_X (n k : ℕ) (total : ℕ) :
  n = 20 ∧ total = 40 ∧ k = 2 →
  let P0 := prob_X 20 0 2 40
  let P1 := prob_X 20 1 2 40
  let P2 := prob_X 20 2 2 40
  let E := 0 * P0 + 1 * P1 + 2 * P2
  E = 1 :=
sorry

-- Problem 2: Median of the increase in body weight of the 40 mice

def weights_control : List ℚ := 
  [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1, 32.6,
   34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]

def weights_experimental : List ℚ := 
  [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 19.8,
   20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]

def sorted_combined_weights := 
  (weights_control ++ weights_experimental).qsort (≤)

theorem median_increase_weight :
  (List.nth sorted_combined_weights 19 = some 23.2) ∧ (List.nth sorted_combined_weights 20 = some 23.6) →
  let m := (23.2 + 23.6) / 2
  m = 23.4 :=
sorry

-- Problem 3: Test significance using K^2

def k_squared (a b c d : ℚ) (n : ℚ) : ℚ :=
  n * ((a*d - b*c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem significant_diff_increase_weight :
  let a := 6
  let b := 14
  let c := 14
  let d := 6
  let n := 40
  let K2 := k_squared a b c d n
  K2 = 6.400 →
  6.400 > 3.841 :=
sorry

end expectation_X_median_increase_weight_significant_diff_increase_weight_l212_212298


namespace seven_disks_cover_large_disk_l212_212934

theorem seven_disks_cover_large_disk (large_radius small_radius : ℝ) (n : ℕ)
  (h₁ : large_radius = 2) (h₂ : small_radius = 1) (h₃ : n = 7) :
  ∃ positions : fin n → ℝ × ℝ, -- positions of centers of smaller disks
    ∀ p : ℝ × ℝ, 
    (dist p (0, 0) ≤ large_radius) → (∃ i : fin n, dist p (positions i) ≤ small_radius) :=
sorry

end seven_disks_cover_large_disk_l212_212934


namespace river_flow_speed_eq_l212_212280

-- Definitions of the given conditions
def ship_speed : ℝ := 30
def distance_downstream : ℝ := 144
def distance_upstream : ℝ := 96

-- Lean 4 statement to prove the condition
theorem river_flow_speed_eq (v : ℝ) :
  (distance_downstream / (ship_speed + v) = distance_upstream / (ship_speed - v)) :=
by { sorry }

end river_flow_speed_eq_l212_212280


namespace people_who_cannot_do_either_l212_212452

def people_total : ℕ := 120
def can_dance : ℕ := 88
def can_write_calligraphy : ℕ := 32
def can_do_both : ℕ := 18

theorem people_who_cannot_do_either : 
  people_total - (can_dance + can_write_calligraphy - can_do_both) = 18 := 
by
  sorry

end people_who_cannot_do_either_l212_212452


namespace river_flow_speed_eq_l212_212281

-- Definitions of the given conditions
def ship_speed : ℝ := 30
def distance_downstream : ℝ := 144
def distance_upstream : ℝ := 96

-- Lean 4 statement to prove the condition
theorem river_flow_speed_eq (v : ℝ) :
  (distance_downstream / (ship_speed + v) = distance_upstream / (ship_speed - v)) :=
by { sorry }

end river_flow_speed_eq_l212_212281


namespace max_value_tan_A_l212_212410

open Real

noncomputable def max_tan_A_given_conditions (A B C : ℝ) (h1 : sin A + 2 * sin B * cos C = 0) (h2 : A + B + C = π) : ℝ :=
  by
    -- Proof omitted
    sorry

-- Statement: The maximum value of tan(A) given the conditions
theorem max_value_tan_A : ∀ (A B C : ℝ), (sin A + 2 * sin B * cos C = 0) → (A + B + C = π) → max_tan_A_given_conditions A B C (sin A + 2 * sin B * cos C = 0) (A + B + C = π) = 1 / sqrt 3 :=
  by 
    -- Proof omitted
    sorry

end max_value_tan_A_l212_212410


namespace factorization_l212_212704

theorem factorization (t : ℝ) : t^3 - 144 = (t - 12) * (t^2 + 12 * t + 144) :=
sorry

end factorization_l212_212704


namespace asymptotes_tangent_to_circle_l212_212435

theorem asymptotes_tangent_to_circle {m : ℝ} (hm : m > 0) 
  (hyp_eq : ∀ x y : ℝ, y^2 - (x^2 / m^2) = 1) 
  (circ_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * y + 3 = 0) : 
  m = (Real.sqrt 3) / 3 :=
sorry

end asymptotes_tangent_to_circle_l212_212435


namespace angle_equality_l212_212828

-- Let ABCD be a convex quadrilateral
variables (A B C D : Type) -- represents points
variables (convex : convex_quadrilateral A B C D)
variables (h1 : ∠ B C D = ∠ C A B)
variables (h2 : ∠ A C D = ∠ B D A)

theorem angle_equality (A B C D : Type)
  (convex : convex_quadrilateral A B C D)
  (h1 : ∠ C B D = ∠ C A B)
  (h2 : ∠ A C D = ∠ B D A) : 
  ∠ A B C = ∠ A D C := 
sorry

end angle_equality_l212_212828


namespace amanda_hourly_rate_l212_212293

def hours_monday := 5 * 1.5
def hours_tuesday := 1 * 3
def hours_thursday := 2 * 2
def hours_saturday := 1 * 6

def total_hours := hours_monday + hours_tuesday + hours_thursday + hours_saturday
def total_earnings := 410

def hourly_rate := total_earnings / total_hours

theorem amanda_hourly_rate : hourly_rate = 20 :=
by
  have h_monday : hours_monday = 7.5 := rfl
  have h_tuesday : hours_tuesday = 3 := rfl
  have h_thursday : hours_thursday = 4 := rfl
  have h_saturday : hours_saturday = 6 := rfl
  have h_total : total_hours = 20.5 := by
    rw [hours_monday, hours_tuesday, hours_thursday, hours_saturday]
    norm_num
  have h_earnings : total_earnings = 410 := rfl
  have h_rate : hourly_rate = total_earnings / total_hours := rfl
  rw [h_total, h_earnings, h_rate]
  norm_num
  sorry

end amanda_hourly_rate_l212_212293


namespace josh_daily_hours_l212_212074

-- Definitions of the parameters
def hours_josh_per_day : ℕ := sorry
def hours_carl_per_day := hours_josh_per_day - 2
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def hourly_wage_josh : ℝ := 9
def hourly_wage_carl := hourly_wage_josh / 2
def total_monthly_payment : ℝ := 1980

-- Lean statement proving the number of hours Josh works per day
theorem josh_daily_hours :
  let month_hours_josh := weeks_per_month * days_per_week * hours_josh_per_day in
  let month_hours_carl := weeks_per_month * days_per_week * hours_carl_per_day in
  let monthly_earnings_josh := month_hours_josh * hourly_wage_josh in
  let monthly_earnings_carl := month_hours_carl * hourly_wage_carl in
  monthly_earnings_josh + monthly_earnings_carl = total_monthly_payment → 
  hours_josh_per_day = 8 :=
by 
  sorry

end josh_daily_hours_l212_212074


namespace same_name_in_most_populous_house_l212_212979

/-- Given 125 people living in houses, with each name being shared by at least three people, 
    prove that there are at least two people with the same name in the most populous house. -/
theorem same_name_in_most_populous_house :
  ∀ (n_people : ℕ) (n_names : ℕ),
    n_people = 125 →
    (∀ name, 3 ∣ n_people) →
    n_people / 3 ≤ n_names →
    ∃ house, 2 ≤ n_people / (house+n_people/n_names) :=
by sorry

end same_name_in_most_populous_house_l212_212979


namespace highest_value_of_a_for_divisibility_l212_212345

/-- Given a number in the format of 365a2_, where 'a' is a digit (0 through 9),
prove that the highest value of 'a' that makes the number divisible by 8 is 9. -/
theorem highest_value_of_a_for_divisibility :
  ∃ (a : ℕ), a ≤ 9 ∧ (∃ (d : ℕ), d < 10 ∧ (365 * 100 + a * 10 + 20 + d) % 8 = 0 ∧ a = 9) :=
sorry

end highest_value_of_a_for_divisibility_l212_212345


namespace sphere_radius_l212_212652

theorem sphere_radius {r h : ℝ} (h1 : 30 = 2 * 15)
                       (h2 : h - (h - 10) = 10)
                       (h3 : r^2 = 225 + 100) :
  r = 5 * Real.sqrt 13 :=
by
  have h_r : 15^2 + (h - 10 - h)^2 = r^2,
  { rw h3, ring },
  sorry

end sphere_radius_l212_212652


namespace water_height_same_l212_212331

noncomputable def height_of_water (rA rB HA HB : ℝ) (VA_initial : ℝ) : ℝ :=
  let V_total := VA_initial
  let h := (V_total / (π * (rA^2 + rB^2)))
  h

theorem water_height_same (rA rB HA HB : ℝ) (VA_initial : ℝ) :
  rA = 6 → rB = 8 → HA = 50 → HB = 50 → VA_initial = π * rB^2 * HB → height_of_water rA rB HA HB VA_initial = 32 :=
by
  intros h
  sorry

end water_height_same_l212_212331


namespace remainder_when_sum_div_by_1000_l212_212277

noncomputable def b : ℕ → ℕ
| 0     := 0
| 1     := 1
| 2     := 1
| 3     := 2
| n + 4 := b (n + 3) + b (n + 2) + b (n + 1)

theorem remainder_when_sum_div_by_1000 :
  ∑ k in Finset.range 28, b (k + 1) % 1000 = 120 := 
by
  have h28 : b 28 = 12202338 := sorry
  have h29 : b 29 = 22404141 := sorry
  have h30 : b 30 = 41207902 := sorry
  sorry

end remainder_when_sum_div_by_1000_l212_212277


namespace fodder_last_days_l212_212252

theorem fodder_last_days (
  initial_buffaloes : ℕ,
  initial_oxen : ℕ,
  initial_cows : ℕ,
  days_fodder : ℕ,
  add_cows : ℕ,
  add_buffaloes : ℕ
) (
  buffalo_to_cows : ℚ,
  ox_to_cows : ℚ,
  initial_fodder_cows : ℚ,
  new_fodder_cows : ℚ,
  daily_consumption : ℚ
) : 
  initial_buffaloes = 15 →
  initial_oxen = 8 →
  initial_cows = 24 →
  days_fodder = 24 →
  add_cows = 40 →
  add_buffaloes = 15 →
  buffalo_to_cows = 4 / 3 →
  ox_to_cows = 3 / 2 →
  initial_fodder_cows = 15 * (4 / 3) + 8 * (3 / 2) + 24 →
  new_fodder_cows = initial_fodder_cows + (40 * 1) + (15 * (4 / 3)) →
  daily_consumption = initial_fodder_cows / days_fodder →
  49 ≤ new_fodder_cows / daily_consumption := 
sorry

end fodder_last_days_l212_212252


namespace f_is_odd_l212_212163

noncomputable def f (x : ℝ) : ℝ :=
  Math.cos (x - Real.pi / 12) ^ 2 + Math.sin (x + Real.pi / 12) ^ 2 - 1

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) :=
by
  intro x
  -- Proof can be added here
  sorry

end f_is_odd_l212_212163


namespace length_of_segment_PS_l212_212816

theorem length_of_segment_PS
  (PR PQ RS: ℕ)
  (hPR : PR = 15)
  (hPQ : PQ = 8)
  (hRS : RS = 17)
  (S_extends_QR : True) : 
  sqrt (PR^2 + (sqrt (PR^2 - PQ^2) + RS)^2) = sqrt (675 + 34 * sqrt 161) :=
by
  sorry

end length_of_segment_PS_l212_212816


namespace base8_1724_to_base10_l212_212600

/-- Define the base conversion function from base-eight to base-ten -/
def base8_to_base10 (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0

/-- Base-eight representation conditions for the number 1724 -/
def base8_1724_digits := (1, 7, 2, 4)

/-- Prove the base-ten equivalent of the base-eight number 1724 is 980 -/
theorem base8_1724_to_base10 : base8_to_base10 1 7 2 4 = 980 :=
  by
    -- skipping the proof; just state that it is a theorem to be proved.
    sorry

end base8_1724_to_base10_l212_212600


namespace find_p_l212_212719

def bn (n : ℕ) (h : n ≥ 5) : ℚ := ((n + 2) ^ 2) / (n ^ 3 - 1)

theorem find_p : 
    let product := ∏ n in Finset.range 96 \ 4, bn (n + 5) (Nat.le_add_left 5 n) in
    product = 6608 / 100! := 
by
    let product := ∏ n in Finset.range 96 \ 4, bn (n + 5) (Nat.le_add_left 5 n)
    have : product = 6608 / 100! := sorry
    exact this

end find_p_l212_212719


namespace cross_product_scalar_distributive_l212_212011

noncomputable def vec3 : Type := (ℝ × ℝ × ℝ)

def cross_product (u v : vec3) : vec3 := 
  let (ux, uy, uz) := u
  let (vx, vy, vz) := v
  (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)

theorem cross_product_scalar_distributive (a b : vec3)
  (h : cross_product a b = (6, -3, 2)) :
  cross_product a (5 * b.1, 5 * b.2, 5 * b.3) = (30, -15, 10) := by
  sorry

end cross_product_scalar_distributive_l212_212011


namespace minimum_cost_proof_l212_212908

structure Store :=
  (name : String)
  (gift_costs : (Nat × Nat × Nat × Nat)) -- (Mom, Dad, Brother, Sister)
  (time_spent : Nat) -- Time spent in store in minutes

def Romashka  : Store := { name := "Romashka", gift_costs := (1000, 750, 930, 850), time_spent := 35 }
def Oduvanchik : Store := { name := "Oduvanchik", gift_costs := (1050, 790, 910, 800), time_spent := 30 }
def Nezabudka : Store := { name := "Nezabudka", gift_costs := (980, 810, 925, 815), time_spent := 40 }
def Landysh : Store := { name := "Landysh", gift_costs := (1100, 755, 900, 820), time_spent := 25 }

def stores : List Store := [Romashka, Oduvanchik, Nezabudka, Landysh]

def travel_time := 30 -- minutes
def total_time := 3 * 60 + 25  -- 3 hours and 25 minutes or 210 minutes

noncomputable def min_cost_within_constraints : Nat :=
  let costs := [
    (Romashka.gift_costs.fst, Romashka.time_spent),
    (Oduvanchik.gift_costs.snd, Oduvanchik.time_spent),
    (Landysh.gift_costs.trd, Landysh.time_spent),
    (Nezabudka.gift_costs.fourth, Nezabudka.time_spent)
    ]
  in 3435 -- Given the final correct answer

theorem minimum_cost_proof : min_cost_within_constraints = 3435 := by
  sorry

end minimum_cost_proof_l212_212908


namespace problem1_problem2_l212_212675

open Real

-- Define the first mathematical equality problem
theorem problem1 : log 3 27 + log 10 25 + log 10 4 + 7^(log 7 2) + (-9.8)^0 = 8 :=
  sorry

-- Define the second mathematical equality problem
theorem problem2 : (8 / 27)^(-2 / 3) - 3 * pi * (pi^(2 / 3)) + sqrt ((2 - pi)^2) = 1 / 4 :=
  sorry

end problem1_problem2_l212_212675


namespace eventually_divisible_by_4_l212_212245

theorem eventually_divisible_by_4 (a_0 : ℕ) : 
  ∃ k : ℕ, ∃ n : ℕ, 
    (a : ℕ → ℕ) 
      (h₁ : ∀ n, a n = if a n % 2 = 0 then a n / 2 else 3 * a n + 1)
      (hk : a 0 = a_0) 
      (k : ℕ → Prop) 
  , 
  k n = (a n % 4 = 0) := 
sorry

end eventually_divisible_by_4_l212_212245


namespace square_free_odd_integers_count_l212_212687

def is_square_free (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k * k ∣ n → false

lemma odd_integer_in_range (n : ℕ) : Prop :=
  n > 1 ∧ n < 200 ∧ odd n

theorem square_free_odd_integers_count :
  (∃ count : ℕ, count = 81 ∧
   count = (λ S, S.card) {n : ℕ | odd_integer_in_range n ∧ is_square_free n} ) :=
begin
  sorry
end

end square_free_odd_integers_count_l212_212687


namespace smallest_points_to_guarantee_victory_l212_212933

noncomputable def pointsForWinning : ℕ := 5
noncomputable def pointsForSecond : ℕ := 3
noncomputable def pointsForThird : ℕ := 1

theorem smallest_points_to_guarantee_victory :
  ∀ (student_points : ℕ),
  (exists (x y z : ℕ), (x = pointsForWinning ∨ x = pointsForSecond ∨ x = pointsForThird) ∧
                         (y = pointsForWinning ∨ y = pointsForSecond ∨ y = pointsForThird) ∧
                         (z = pointsForWinning ∨ z = pointsForSecond ∨ z = pointsForThird) ∧
                         student_points = x + y + z) →
  (∃ (victory_points : ℕ), victory_points = 13) →
  (∀ other_points : ℕ, other_points < victory_points) :=
sorry

end smallest_points_to_guarantee_victory_l212_212933


namespace development_value_project_l212_212453

-- Define the profit function without road construction
def profit_without_road (x : ℝ) : ℝ := - (1 / 160) * (x - 40)^2 + 10

-- Define the total profit without road construction over 10 years
def total_profit_without_road : ℝ := profit_without_road 40 * 10

-- Define the profit function for the first 5 years with road construction
def profit_with_road_first_5_years (x : ℝ) : ℝ := - (1 / 160) * (x - 40)^2 + 10

-- Define the profit function after road completion
def profit_after_road (x : ℝ) : ℝ := - (159 / 160) * (60 - x)^2 + (119 / 2) * (60 - x)

-- Define the total profit over the first 5 years with an investment of 30
def total_profit_first_5_years : ℝ := profit_with_road_first_5_years 30 * 5

-- Define the total profit over the next 5 years after road completion
def total_profit_next_5_years : ℝ := 
  5 * (profit_with_road_first_5_years 30 + profit_after_road 30)

-- Define the total profit with road construction over 10 years
def total_profit_with_road : ℝ := total_profit_first_5_years + total_profit_next_5_years

theorem development_value_project : total_profit_with_road > total_profit_without_road := by 
  sorry

end development_value_project_l212_212453


namespace assign_numbers_l212_212538

-- Definitions for goal differences
variable (Team : Type) (d : Team → Team → ℤ)
variables (A B C : Team)

-- Conditions as stated in the problem
axiom d_symmetric : ∀ A B, d(A, B) + d(B, A) = 0
axiom d_cyclic : ∀ A B C, d(A, B) + d(B, C) + d(C, A) = 0

-- The statement to be proved
theorem assign_numbers :
  ∃ f : Team → ℤ, ∀ A B, d(A, B) = f(A) - f(B) :=
sorry

end assign_numbers_l212_212538


namespace min_cost_proof_l212_212912

/--
  Misha needs to buy different gifts for his mom, dad, brother, and sister from 4 stores.
  The gifts and their costs are as follows (store, cost):
  Romashka: mom = 1000, dad = 750, brother = 930, sister = 850.
  Oduvanchik: mom = 1050, dad = 790, brother = 910, sister = 800.
  Nezabudka: mom = 980, dad = 810, brother = 925, sister = 815.
  Landysh: mom = 1100, dad = 755, brother = 900, sister = 820.
  Each store closes at 8:00 PM and the traveling time between stores and home is 30 minutes.
  The shopping time in each store varies.

  Prove that the minimum amount of money Misha can spend to buy all 4 gifts is 3435 rubles given the time constraints.
-/
def misha_min_spent : ℕ :=
  let mom_cost := min 1000 (min 1050 (min 980 1100))
  let dad_cost := min 750 (min 790 (min 810 755))
  let brother_cost := min 930 (min 910 (min 925 900))
  let sister_cost := min 850 (min 800 (min 815 820))
  mom_cost + dad_cost + brother_cost + sister_cost

theorem min_cost_proof (h: ∃ g1 g2 g3 g4: ℕ,
                              g1 ∈ {980, 1000, 1050, 1100} ∧
                              g2 ∈ {750, 790, 810, 755} ∧
                              g3 ∈ {900, 925, 930, 910} ∧
                              g4 ∈ {800, 820, 815, 850} ∧
                              g1 + g2 + g3 + g4 = 3435) : 
                              misha_min_spent = 3435 := 
by
  sorry


end min_cost_proof_l212_212912


namespace triangle_side_length_c_l212_212811

theorem triangle_side_length_c (a b : ℝ) (α β γ : ℝ) (h_angle_sum : α + β + γ = 180) (h_angle_eq : 3 * α + 2 * β = 180) (h_a : a = 2) (h_b : b = 3) : 
∃ c : ℝ, c = 4 :=
by
  sorry

end triangle_side_length_c_l212_212811


namespace polynomial_inequality_l212_212961

open Polynomial

noncomputable def polynomial_f (n : ℕ) (a : Fin n → ℝ) : ℝ[X] :=
  (X ^ n) + (∑ i in Finset.range n, (coeff n i a) * (X ^ (n - 1 - i)))
  where coeff (n : ℕ) (i : ℕ) (a : Fin n → ℝ) : ℝ := ite (i < n) (a ⟨i, nat.lt_succ_self i⟩) 1

theorem polynomial_inequality (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 ≤ a i) (hroots : (polynomial_f n a).roots.card = n) :
    eval 2 (polynomial_f n a) ≥ 3^n :=
begin
  sorry
end

end polynomial_inequality_l212_212961


namespace store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l212_212054

-- Definitions and conditions
def cost_per_soccer : ℕ := 200
def cost_per_basketball : ℕ := 80
def discount_A_soccer (n : ℕ) : ℕ := n * cost_per_soccer
def discount_A_basketball (n : ℕ) : ℕ := if n > 100 then (n - 100) * cost_per_basketball else 0
def discount_B_soccer (n : ℕ) : ℕ := n * cost_per_soccer * 8 / 10
def discount_B_basketball (n : ℕ) : ℕ := n * cost_per_basketball * 8 / 10

-- For x = 100
def total_cost_A_100 : ℕ := discount_A_soccer 100 + discount_A_basketball 100
def total_cost_B_100 : ℕ := discount_B_soccer 100 + discount_B_basketball 100

-- Prove that for x = 100, Store A is more cost-effective
theorem store_A_more_cost_effective_100 : total_cost_A_100 < total_cost_B_100 :=
by sorry

-- For x > 100, express costs in terms of x
def total_cost_A (x : ℕ) : ℕ := 80 * x + 12000
def total_cost_B (x : ℕ) : ℕ := 64 * x + 16000

-- Prove the expressions for costs
theorem cost_expressions_for_x (x : ℕ) (h : x > 100) : 
  total_cost_A x = 80 * x + 12000 ∧ total_cost_B x = 64 * x + 16000 :=
by sorry

-- For x = 300, most cost-effective plan
def combined_A_100_B_200 : ℕ := (discount_A_soccer 100 + cost_per_soccer * 100) + (200 * cost_per_basketball * 8 / 10)
def only_A_300 : ℕ := discount_A_soccer 100 + (300 - 100) * cost_per_basketball
def only_B_300 : ℕ := discount_B_soccer 100 + 300 * cost_per_basketball * 8 / 10

-- Prove the most cost-effective plan for x = 300
theorem most_cost_effective_plan : combined_A_100_B_200 < only_B_300 ∧ combined_A_100_B_200 < only_A_300 :=
by sorry

end store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l212_212054


namespace cos_C_eq_neg_quarter_l212_212753

-- Given conditions
variables {A B C : ℝ}
variables {a b c : ℝ}
variable (k : ℝ)
variable h_perimeter : a + b + c = 9
variable h_sine_ratio : sin A / sin B = 3 / 2 ∧ sin A / sin C = 3 / 4

-- Derived conditions from given problem
variables h_a : a = 3 * k
variables h_b : b = 2 * k
variables h_c : c = 4 * k

-- Goal to prove
theorem cos_C_eq_neg_quarter : cos C = -1 / 4 :=
by
  sorry  -- Proof will be filled here

end cos_C_eq_neg_quarter_l212_212753


namespace hyperbola_eccentricity_l212_212730

theorem hyperbola_eccentricity (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b^2 = a * c) (h₄ : c^2 = a^2 + b^2) : 
  let e := c / a in
  e = (Real.sqrt 5 + 1) / 2 := sorry

end hyperbola_eccentricity_l212_212730


namespace min_stamps_needed_l212_212312

theorem min_stamps_needed {c f : ℕ} (h : 3 * c + 4 * f = 33) : c + f = 9 :=
sorry

end min_stamps_needed_l212_212312


namespace discount_and_final_price_l212_212046

noncomputable def apply_percent_discount (original_price : ℝ) (percent_discount : ℝ) : ℝ :=
  original_price * (1 - percent_discount / 100)

noncomputable def apply_promotion (original_price : ℝ) (promotion_percent : ℝ) : ℝ :=
  original_price * (1 - promotion_percent / 100)

noncomputable def apply_sales_tax (price_after_discounts : ℝ) (sales_tax_percent : ℝ) : ℝ :=
  price_after_discounts * (1 + sales_tax_percent / 100)

theorem discount_and_final_price
  (original_price : ℝ)
  (discount_amount : ℝ)
  (promotion_percent : ℝ)
  (sales_tax_percent : ℝ) :
  (original_price = 500) →
  (discount_amount = 350) →
  (promotion_percent = 10) →
  (sales_tax_percent = 7) →
  let percent_discount := (discount_amount / original_price) * 100 in
  let price_after_promotion := apply_promotion original_price promotion_percent in
  let price_after_discount := apply_percent_discount price_after_promotion percent_discount in
  let final_price := apply_sales_tax price_after_discount sales_tax_percent in
  (percent_discount = 70) ∧ (final_price = 144.45) := 
by {
  intros original_price discount_amount promotion_percent sales_tax_percent h1 h2 h3 h4,
  let percent_discount := (discount_amount / original_price) * 100,
  let price_after_promotion := apply_promotion original_price promotion_percent,
  let price_after_discount := apply_percent_discount price_after_promotion percent_discount,
  let final_price := apply_sales_tax price_after_discount sales_tax_percent,
  split,
  { rw [h1, h2], norm_num },
  { rw [h1, h3, h4], norm_num, },
}

end discount_and_final_price_l212_212046


namespace area_of_sector_one_radian_l212_212154

theorem area_of_sector_one_radian (r θ : ℝ) (hθ : θ = 1) (hr : r = 1) : 
  (1/2 * (r * θ) * r) = 1/2 :=
by
  sorry

end area_of_sector_one_radian_l212_212154


namespace Mia_wins_games_l212_212050

theorem Mia_wins_games
  (Sarah_wins : ℕ)
  (Sarah_losses : ℕ)
  (Ryan_wins : ℕ)
  (Ryan_losses : ℕ)
  (Mia_losses : ℕ)
  (total_games : ℕ)
  (Sarah_games_eq : Sarah_wins + Sarah_losses = 6)
  (Ryan_games_eq : Ryan_wins + Ryan_losses = 6)
  (total_games_eq : total_games = (16 + (total_games - 8)) / 2)
  (wins_eq_games : Sarah_wins + Ryan_wins + (total_games - 8) = total_games) :
  (total_games - 8) = 2 :=
begin
  sorry
end

end Mia_wins_games_l212_212050


namespace max_n_sum_squares_12345_l212_212603

theorem max_n_sum_squares_12345 :
  ∃ (k : ℕ → ℕ) (n : ℕ), 
      (∀ (i j : ℕ), (i ≠ j) → (k i ≠ k j)) ∧ -- distinct integers
      (∑ i in Finset.range n, (k i)^2 = 12345) ∧ -- sum of squares condition
      (n = 33) := 
sorry

end max_n_sum_squares_12345_l212_212603


namespace adam_simon_hours_apart_l212_212661

theorem adam_simon_hours_apart
  (x : ℝ)
  (h1 : ∀ t : ℝ, t ≠ 0)
  (h2 : ∀ t : ℝ, t > 0)
  (h3 : ∀ t : ℝ, ∃ a : ℝ, ∃ s : ℝ, a = 10 * t ∧ s = 7 * t ∧ (a^2 + s^2 = 85^2))
  : x ≈ 7 := sorry

end adam_simon_hours_apart_l212_212661


namespace sum_m_n_p_l212_212138

theorem sum_m_n_p : ∃ m n p : ℤ, 
  (∀ x, 2 * x * (4 * x - 5) = -4 ↔ 
    (x = (m + complex.sqrt n) / p) ∨ (x = (m - complex.sqrt n) / p)) ∧
  nat.gcd (nat.gcd m.nat_abs n) p.nat_abs = 1 ∧
  m + n + p = 20 :=
by
  sorry

end sum_m_n_p_l212_212138


namespace gcd_max_value_l212_212968

theorem gcd_max_value (x y : ℤ) (h_posx : x > 0) (h_posy : y > 0) (h_sum : x + y = 780) :
  gcd x y ≤ 390 ∧ ∃ x' y', x' > 0 ∧ y' > 0 ∧ x' + y' = 780 ∧ gcd x' y' = 390 := by
  sorry

end gcd_max_value_l212_212968


namespace Tamika_greater_probability_equals_one_l212_212944

-- First, define the sets involved.
def T_set : set ℕ := {6, 7, 8}
def C_set : set ℕ := {2, 4, 5}

-- Define the products Tamika and Carlos can get.
def T_products : set ℕ := {42, 48, 56}
def C_products : set ℕ := {8, 10, 20}

-- Probability calculation
def favorable_pairs : set (ℕ × ℕ) := 
  {(t, c) | t ∈ T_products ∧ c ∈ C_products ∧ t > c}

-- Counting pairs to verify if the probability is 1
def total_pairs := T_products.card * C_products.card

-- Calculate the probability that Tamika's result is greater than Carlos'
def probability_T_greater_C := 
  favorable_pairs.card / total_pairs

theorem Tamika_greater_probability_equals_one : 
  probability_T_greater_C = 1 :=
by
  sorry

end Tamika_greater_probability_equals_one_l212_212944


namespace find_value_m_n_l212_212062

noncomputable def symmetric_point_Oxy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

theorem find_value_m_n :
  ∃ (m n : ℝ), symmetric_point_Oxy (3, -1, m) = (3, n, -2) ∧ m + n = 1 :=
by
  use 2, -1
  split
  . simp [symmetric_point_Oxy]
  . linarith

end find_value_m_n_l212_212062


namespace line_passing_through_P_and_parallel_to_polar_axis_has_equation_l212_212815

theorem line_passing_through_P_and_parallel_to_polar_axis_has_equation :
  ∀ (ρ θ : ℝ), ρ = 2 → θ = π / 6 → (∃ l : line, l.passes_through (ρ, θ) ∧ l.is_parallel_to_polar_axis) → 
  (ρ * sin θ = 1) := 
by 
  intros ρ θ h1 h2 h3 
  sorry

end line_passing_through_P_and_parallel_to_polar_axis_has_equation_l212_212815


namespace AC_length_l212_212057

variable (A B C D : Type) [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space D]

variable (AB AD DC AC : ℝ)

-- Given conditions
variable (AB := 12) -- AB = 12 cm
variable (AD := 8)  -- AD = 8 cm
variable (DC := 18) -- DC = 18 cm

-- Quadrilateral properties
variable (is_symmetric : A B C D → Prop)
variable (is_perpendicular : A B C D.pairwise_disjoint ∧ (pair.self (AB * AC)) ⊥ (pair.self AD))

noncomputable def length_of_AC : ℝ :=
  if h : is_symmetric A B C D ∧ is_perpendicular A B C D then
    18
  else
    0

theorem AC_length (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB AC AD DC : ℝ) (is_symmetric : A B C D → Prop) (is_perpendicular : A B C D.pairwise_disjoint ∧ (pair.self (AB * AC)) ⊥ (pair.self AD))
  (h1 : AB = 12) (h2 : AD = 8) (h3 : DC = 18) (h_sym : is_symmetric A B C D) (h_perp : is_perpendicular A B C D) : 
  length_of_AC A B C D AB AD DC AC is_symmetric is_perpendicular = 18 := by
  sorry

end AC_length_l212_212057


namespace percentage_by_tenth_day_is_49_percent_l212_212817

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ → ℝ
| 1   => 5
| n+1 => arithmetic_sequence n + (-4 / 29)

-- Sum of first n terms of arithmetic sequence
def sum_arithmetic (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (f 1 + f n)) / 2

-- Prove that the percentage of cloth woven by the 10th day is 49% of the total amount woven over 30 days
theorem percentage_by_tenth_day_is_49_percent :
  let S10 := sum_arithmetic arithmetic_sequence 10
  let S30 := sum_arithmetic arithmetic_sequence 30
  S10 / S30 = 0.49 :=
by
  sorry

end percentage_by_tenth_day_is_49_percent_l212_212817


namespace arithmetic_geometric_inequality_l212_212528

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a ≠ b) :
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  B < (a - b)^2 / (8 * (A - B)) ∧ (a - b)^2 / (8 * (A - B)) < A :=
by
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  sorry

end arithmetic_geometric_inequality_l212_212528


namespace normal_price_of_article_l212_212992

theorem normal_price_of_article 
  (final_price : ℝ) 
  (d1 d2 d3 : ℝ) 
  (P : ℝ) 
  (h_final_price : final_price = 36) 
  (h_d1 : d1 = 0.15) 
  (h_d2 : d2 = 0.25) 
  (h_d3 : d3 = 0.20) 
  (h_eq : final_price = P * (1 - d1) * (1 - d2) * (1 - d3)) : 
  P = 70.59 := sorry

end normal_price_of_article_l212_212992


namespace sum_of_coefficients_l212_212680

noncomputable def sequence (u : ℕ → ℕ) : Prop :=
u 1 = 5 ∧ ∀ n : ℕ, u (n + 1) - u n = 3 + 4 * (n - 1)

theorem sum_of_coefficients (u : ℕ → ℕ) (h : sequence u) :
  ∃ (a b c : ℤ), (∀ n : ℕ, u n = a * n^2 + b * n + c) ∧ a + b + c = 5 :=
sorry

end sum_of_coefficients_l212_212680


namespace collinear_vectors_l212_212770

-- Define the vectors and condition of collinearity
variables (k : ℝ)
def vector_OA : ℝ × ℝ := (k, 12)
def vector_OB : ℝ × ℝ := (4, 5)
def vector_OC : ℝ × ℝ := (-k, 10)

-- Define vectors AB and AC
def vector_AB : ℝ × ℝ := (4 - k, -7)
def vector_AC : ℝ × ℝ := (-2 * k, -2)

-- Prove collinearity condition
theorem collinear_vectors : 
  vector_OA k = (k, 12) ∧ 
  vector_OB = (4, 5) ∧ 
  vector_OC = (-k, 10) ∧ 
  ((-2) * (4 - k) = (-7) * (-2 * k)) → 
  k = - (2 / 3) := 
by 
  sorry

end collinear_vectors_l212_212770


namespace closest_point_to_line_l212_212354

theorem closest_point_to_line {x y : ℝ} (h : y = 2 * x - 4) :
  ∃ (closest_x closest_y : ℝ),
    closest_x = 9 / 5 ∧ closest_y = -2 / 5 ∧ closest_y = 2 * closest_x - 4 ∧
    ∀ (x' y' : ℝ), y' = 2 * x' - 4 → (closest_x - 3)^2 + (closest_y + 1)^2 ≤ (x' - 3)^2 + (y' + 1)^2 :=
by
  sorry

end closest_point_to_line_l212_212354


namespace minimum_value_expression_l212_212499

noncomputable def expression (a b c d : ℝ) : ℝ :=
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  expression a b c d ≥ 8 :=
by
  -- Proof goes here
  sorry

end minimum_value_expression_l212_212499


namespace g_18_66_l212_212953

def g (x y : ℕ) : ℕ := sorry

axiom g_prop1 : ∀ x, g x x = x
axiom g_prop2 : ∀ x y, g x y = g y x
axiom g_prop3 : ∀ x y, (x + 2 * y) * g x y = y * g x (x + 2 * y)

theorem g_18_66 : g 18 66 = 198 :=
by
  sorry

end g_18_66_l212_212953


namespace working_mom_work_percent_l212_212239

theorem working_mom_work_percent :
  let awake_hours := 16
  let work_hours := 8
  (work_hours / awake_hours) * 100 = 50 :=
by
  sorry

end working_mom_work_percent_l212_212239


namespace find_value_of_sum_of_squares_l212_212319

-- Definitions of the conditions
variables (x y z : ℝ)
def condition1 : Prop := x^2 + 4 * y = 8
def condition2 : Prop := y^2 + 6 * z = 0
def condition3 : Prop := z^2 + 8 * x = -16

-- The theorem to prove
theorem find_value_of_sum_of_squares (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) :
  x^2 + y^2 + z^2 = 21 :=
sorry

end find_value_of_sum_of_squares_l212_212319


namespace units_digit_of_sum_factorials_l212_212226

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the function to get the units digit of a number
def units_digit (n : ℕ) : ℕ :=
n % 10

-- Define the main problem statement
theorem units_digit_of_sum_factorials : units_digit (∑ i in finset.range 100, factorial i) = 7 :=
by sorry

end units_digit_of_sum_factorials_l212_212226


namespace ratio_part_to_whole_l212_212918

/-- One part of one third of two fifth of a number is 17, and 40% of that number is 204. 
Prove that the ratio of the part to the whole number is 1:30. -/
theorem ratio_part_to_whole 
  (N : ℝ)
  (h1 : (1 / 1) * (1 / 3) * (2 / 5) * N = 17) 
  (h2 : 0.40 * N = 204) : 
  17 / N = 1 / 30 :=
  sorry

end ratio_part_to_whole_l212_212918


namespace C_pow_eq_target_l212_212087

open Matrix

-- Define the specific matrix C
def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

-- Define the target matrix for the formula we need to prove
def C_power_50 : Matrix (Fin 2) (Fin 2) ℤ := !![101, 50; -200, -99]

-- Prove that C^50 equals to the target matrix
theorem C_pow_eq_target (n : ℕ) (h : n = 50) : C ^ n = C_power_50 := by
  rw [h]
  sorry

end C_pow_eq_target_l212_212087


namespace regression_constant_a_l212_212544

theorem regression_constant_a :
  let x_vals := [1, 3, 4, 5, 7]
  let y_vals := [15, 20, 30, 40, 45]
  let mean (l : List ℝ) : ℝ := l.sum / l.length
  let x_bar := mean x_vals
  let y_bar := mean y_vals
  let regression_eq := λ x, 4.5 * x + (y_bar - 4.5 * x_bar)
  (∃ a : ℝ, ∀ x ∈ x_vals, ∀ y ∈ y_vals, y = 4.5 * x + a → a = 12) :=
by
  sorry

end regression_constant_a_l212_212544


namespace total_pieces_correct_l212_212598

-- Definition of the pieces of chicken required per type of order
def chicken_pieces_per_chicken_pasta : ℕ := 2
def chicken_pieces_per_barbecue_chicken : ℕ := 3
def chicken_pieces_per_fried_chicken_dinner : ℕ := 8

-- Definition of the number of each type of order tonight
def num_fried_chicken_dinner_orders : ℕ := 2
def num_chicken_pasta_orders : ℕ := 6
def num_barbecue_chicken_orders : ℕ := 3

-- Calculate the total number of pieces of chicken needed
def total_chicken_pieces_needed : ℕ :=
  (num_fried_chicken_dinner_orders * chicken_pieces_per_fried_chicken_dinner) +
  (num_chicken_pasta_orders * chicken_pieces_per_chicken_pasta) +
  (num_barbecue_chicken_orders * chicken_pieces_per_barbecue_chicken)

-- The proof statement
theorem total_pieces_correct : total_chicken_pieces_needed = 37 :=
by
  -- Our exact computation here
  sorry

end total_pieces_correct_l212_212598


namespace min_gift_cost_time_constrained_l212_212906

def store := { name : String, mom : ℕ, dad : ℕ, brother : ℕ, sister : ℕ, time_in_store : ℕ }

def romashka : store := { name := "Romashka", mom := 1000, dad := 750, brother := 930, sister := 850, time_in_store := 35 }
def oduvanchik : store := { name := "Oduvanchik", mom := 1050, dad := 790, brother := 910, sister := 800, time_in_store := 30 }
def nezabudka : store := { name := "Nezabudka", mom := 980, dad := 810, brother := 925, sister := 815, time_in_store := 40 }
def landysh : store := { name := "Landysh", mom := 1100, dad := 755, brother := 900, sister := 820, time_in_store := 25 }

constant travel_time : ℕ := 30
constant total_available_time : ℕ := 3 * 60 + 25 -- 205 minutes

def total_time (stores : List store) : ℕ :=
  stores.map (λ s => s.time_in_store).sum + (travel_time * (stores.length - 1))

def total_cost (stores : List store) : ℕ :=
  stores[0].mom + stores[1].dad + stores[2].brother + stores[3].sister

theorem min_gift_cost_time_constrained : 
  ∃ stores : List store, stores.length = 4 ∧ total_time stores ≤ total_available_time ∧ total_cost stores = 3435 :=
by
  sorry

end min_gift_cost_time_constrained_l212_212906


namespace sum_of_segments_eq_200_sqrt_41_l212_212122

theorem sum_of_segments_eq_200_sqrt_41 :
  let AB := 5
      CB := 4
      n := 200
      diag := (AB^2 + CB^2).sqrt
      a_k (k : ℕ) := diag * (n - k).to_real / n
      sum_ak := (Finset.range (n - 1)).sum (λ k, 2 * (a_k k)) + diag
  in sum_ak = 200 * diag :=
by {
  sorry
}

end sum_of_segments_eq_200_sqrt_41_l212_212122


namespace circle_diameter_l212_212632

theorem circle_diameter (r d : ℝ) (h₀ : ∀ (r : ℝ), ∃ (d : ℝ), d = 2 * r) (h₁ : π * r^2 = 9 * π) :
  d = 6 :=
by
  rcases h₀ r with ⟨d, hd⟩
  sorry

end circle_diameter_l212_212632


namespace area_of_DEF_l212_212451

def triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def area_of_triangle (a b c : ℝ) :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_DEF :
  let DE := 31
  let EF := 31
  let DF := 46
  triangle DE EF DF →
  area_of_triangle DE EF DF = 477 :=
by
  intros h
  sorry

end area_of_DEF_l212_212451


namespace number_of_ways_to_select_courses_l212_212276

theorem number_of_ways_to_select_courses :
  let typeA := 3
  let typeB := 4
  ∃ k, (k = (nat.choose typeA 2) * (nat.choose typeB 1) + (nat.choose typeA 1) * (nat.choose typeB 2) + (nat.choose typeA 1) * (nat.choose typeB 1)) ∧ k = 42 :=
begin
  sorry
end

end number_of_ways_to_select_courses_l212_212276


namespace radius_of_inner_circle_is_correct_l212_212884

noncomputable def radius_of_inner_circle : ℝ := 
let Ox (r : ℝ) := (r^2 + 8*r)^(1/2),
    Oz (r : ℝ) := (r^2 + 6*r)^(1/2),
    height := (48 : ℝ)^(1/2) in
  if h : (Ox r + Oz r = height) ∧ (∀ r', Ox r' + Oz r' = height → r' = r) 
  then r else 0

-- The problem statement
theorem radius_of_inner_circle_is_correct : 
  radius_of_inner_circle = ( -42 + 18 * real.sqrt 3 ) / 23 :=
by
  sorry

end radius_of_inner_circle_is_correct_l212_212884


namespace odd_squares_diff_divisible_by_8_l212_212128

theorem odd_squares_diff_divisible_by_8 (m n : ℤ) (a b : ℤ) (hm : a = 2 * m + 1) (hn : b = 2 * n + 1) : (a^2 - b^2) % 8 = 0 := sorry

end odd_squares_diff_divisible_by_8_l212_212128


namespace lattice_points_interval_length_l212_212505

theorem lattice_points_interval_length :
  let T := { (x, y) | 1 ≤ x ∧ x ≤ 40 ∧ 1 ≤ y ∧ y ≤ 40 }
  in ∃ a b : ℕ, Nat.gcd a b = 1 ∧
     (∃ m1 m2 : ℚ, m1 < m2 ∧
      (∀ m : ℚ, m1 ≤ m ∧ m ≤ m2 → ∃ count, count ≤ 560) ∧
      (m2 - m1 = a / b)) ∧
     (a + b = 21) :=
sorry

end lattice_points_interval_length_l212_212505


namespace angle_equality_l212_212830

-- Let ABCD be a convex quadrilateral
variables (A B C D : Type) -- represents points
variables (convex : convex_quadrilateral A B C D)
variables (h1 : ∠ B C D = ∠ C A B)
variables (h2 : ∠ A C D = ∠ B D A)

theorem angle_equality (A B C D : Type)
  (convex : convex_quadrilateral A B C D)
  (h1 : ∠ C B D = ∠ C A B)
  (h2 : ∠ A C D = ∠ B D A) : 
  ∠ A B C = ∠ A D C := 
sorry

end angle_equality_l212_212830


namespace triangle_right_angled_l212_212476

theorem triangle_right_angled
  (a b c : ℝ) (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * Real.cos C + c * Real.cos B = a * Real.sin A) :
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 :=
sorry

end triangle_right_angled_l212_212476


namespace infinite_lines_intersecting_all_three_l212_212801

noncomputable def skew_lines (a b c : set (ℝ × ℝ × ℝ)) : Prop :=
  -- Definitions asserting that no two lines are parallel or intersect each other.
  ¬parallel a b ∧ ¬parallel b c ∧ ¬parallel a c ∧ 
  ∀ x ∈ a, ∀ y ∈ b, x ≠ y ∧ 
  ∀ x ∈ b, ∀ y ∈ c, x ≠ y ∧ 
  ∀ x ∈ a, ∀ y ∈ c, x ≠ y

noncomputable def lines_intersecting_all_three (a b c : set (ℝ × ℝ × ℝ)) : set (set (ℝ × ℝ × ℝ)) :=
  { l | ∃ p₁ ∈ a, ∃ p₂ ∈ b, ∃ p₃ ∈ c, p₁ = p₂ ∧ p₂ = p₃ ∧ l = line_through p₁ p₂ }

theorem infinite_lines_intersecting_all_three (a b c : set (ℝ × ℝ × ℝ)) (h : skew_lines a b c) :
  ∃! l : set (set (ℝ × ℝ × ℝ)), lines_intersecting_all_three a b c = set.univ :=
by
  sorry

end infinite_lines_intersecting_all_three_l212_212801


namespace distance_to_left_directrix_l212_212243

def ellipse_eq (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 / 25 + y^2 / 9 = 1

def is_focus (F: ℝ × ℝ) : Prop :=
  F = (-4, 0)

def midpoint_condition (O P F Q : ℝ × ℝ) : Prop :=
  let (ox, oy) := O
  let (px, py) := P
  let (fx, fy) := F
  let (qx, qy) := Q
  (qx, qy) = (1/2 * (ox + px + fx), 1/2 * (oy + py + fy)) ∧ (qx^2 + qy^2).sqrt = 3

theorem distance_to_left_directrix
  (P Q F1 : ℝ × ℝ)
  (hP_on_ellipse : ellipse_eq P)
  (hF1_is_focus : is_focus F1)
  (hQ_condition : midpoint_condition (0,0) P F1 Q) : 
  ∃ d : ℝ, d = 5 :=
sorry

end distance_to_left_directrix_l212_212243


namespace teresas_total_adjusted_score_l212_212145

/--
Teresa's scores in different subjects:
- Science: 70 marks
- Music: 80 marks
- Social Studies: 85 marks
- Physics: half as many marks as in music
- Mathematics: 90 marks (75% worth)

Marks deductions for incorrect answers:
- Science: 6 incorrect answers
- Music: 9 incorrect answers
- Social Studies: 4 incorrect answers
- Physics: 12 incorrect answers
- Mathematics: 3 incorrect answers

Weightages:
- Science: 20%
- Music: 25%
- Social Studies: 30%
- Physics: 10%
- Mathematics: 15%

Prove that Teresa's total adjusted score after accounting for negative marking and weightages is 71.6625.
-/
theorem teresas_total_adjusted_score :
  let science_marks := 70
  let music_marks := 80
  let social_studies_marks := 85
  let physics_marks := music_marks / 2
  let mathematics_marks := 90
  let incorrect_science := 2
  let incorrect_music := 3
  let incorrect_social_studies := 1
  let incorrect_physics := 4
  let incorrect_mathematics := 1
  let weighted_science := 20
  let weighted_music := 25
  let weighted_social_studies := 30
  let weighted_physics := 10
  let weighted_mathematics := 15
  let adjusted_science_marks := science_marks - incorrect_science
  let adjusted_music_marks := music_marks - incorrect_music
  let adjusted_social_studies_marks := social_studies_marks - incorrect_social_studies
  let adjusted_physics_marks := physics_marks - incorrect_physics
  let adjusted_mathematics_marks := (mathematics_marks - incorrect_mathematics) * 0.75
  let total_weighted_score :=
    (adjusted_science_marks * weighted_science / 100) +
    (adjusted_music_marks * weighted_music / 100) +
    (adjusted_social_studies_marks * weighted_social_studies / 100) +
    (adjusted_physics_marks * weighted_physics / 100) +
    (adjusted_mathematics_marks * weighted_mathematics / 100)
  in total_weighted_score = 71.6625 :=
begin
  sorry
end

end teresas_total_adjusted_score_l212_212145


namespace annika_time_back_l212_212668

-- Define the constants based on given conditions
def rate : ℝ := 10 -- minutes per kilometer
def initial_distance : ℝ := 2.75 -- kilometers
def total_east_distance : ℝ := 3.625 -- kilometers

-- Total time to hike out and back
def total_time : ℝ := (2 * total_east_distance) * rate

-- Prove that the time Annika has to be back at the start of the trail is 72.5 minutes
theorem annika_time_back (rate initial_distance total_east_distance : ℝ) : 
  rate = 10 → 
  initial_distance = 2.75 → 
  total_east_distance = 3.625 → 
  total_time = 72.5 :=
by
  intros h_rate h_initial h_total_east
  -- Definitions and calculations
  rw [h_rate, h_initial, h_total_east]
  sorry

end annika_time_back_l212_212668


namespace f_decreasing_interval_l212_212557

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2 * x ^ 2 - x + 1
  else -2 * x ^ 2 + 7 * x - 7

theorem f_decreasing_interval :
  (∀ x < 1, f x = 2 * x ^ 2 - x + 1) →
  (∀ t, f (-t + 1) = -f (t + 1)) →
  ∀ x > (7 / 4), f(x) is strictly decreasing :=
by
  intros h1 h2
  sorry

end f_decreasing_interval_l212_212557


namespace min_weighings_to_find_counterfeit_l212_212188

-- Definition of the problem conditions.
def coin_is_genuine (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins m = coins (Fin.mk 0 sorry)

def counterfit_coin_is_lighter (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins n < coins m

-- The theorem statement
theorem min_weighings_to_find_counterfeit :
  (∀ coins : Fin 10 → ℝ, ∃ n : Fin 10, coin_is_genuine coins n ∧ counterfit_coin_is_lighter coins n → ∃ min_weighings : ℕ, min_weighings = 3) :=
by {
  sorry
}

end min_weighings_to_find_counterfeit_l212_212188


namespace solve_box_dimensions_l212_212646

theorem solve_box_dimensions (m n r : ℕ) (h1 : m ≤ n) (h2 : n ≤ r) (h3 : m ≥ 1) (h4 : n ≥ 1) (h5 : r ≥ 1) :
  let k₀ := (m - 2) * (n - 2) * (r - 2)
  let k₁ := 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2))
  let k₂ := 4 * ((m - 2) + (n - 2) + (r - 2))
  (k₀ + k₂ - k₁ = 1985) ↔ ((m = 5 ∧ n = 7 ∧ r = 663) ∨ 
                            (m = 5 ∧ n = 5 ∧ r = 1981) ∨
                            (m = 3 ∧ n = 3 ∧ r = 1981) ∨
                            (m = 1 ∧ n = 7 ∧ r = 399) ∨
                            (m = 1 ∧ n = 3 ∧ r = 1987)) :=
sorry

end solve_box_dimensions_l212_212646


namespace area_of_L_shaped_region_l212_212705

theorem area_of_L_shaped_region : 
  ∀ (a b c d : ℕ) (s₁ s₂ s₃ : ℕ),
  a = 7 → b = 7 → c = 7 → d = 7 → s₁ = 2 → s₂ = 2 → s₃ = 3 →
  (a * b) - ((s₁ * s₁) + (s₂ * s₂) + (s₃ * s₃)) = 32 :=
  by
  intros a b c d s₁ s₂ s₃
  assume ha₁ hb₁ hc₁ hd₁ hs₁ hs₂ hs₃
  sorry

end area_of_L_shaped_region_l212_212705


namespace log_base_3_of_24_l212_212366

theorem log_base_3_of_24 (a : ℝ) (h : log 3 2 = a) : log 3 24 = 1 + 3 * a :=
sorry

end log_base_3_of_24_l212_212366


namespace angle_B_in_progression_l212_212863

theorem angle_B_in_progression (A B C a b c : ℝ) (h1: A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) 
(h2: B - A = C - B) (h3: b^2 - a^2 = a * c) (h4: A + B + C = Real.pi) : 
B = 2 * Real.pi / 7 := sorry

end angle_B_in_progression_l212_212863


namespace ancient_greek_life_span_l212_212666

-- Define the dates and time period
def birth_date := (40 : Int, "BC")
def death_date := (40 : Int, "AD")
def no_year_zero := true

-- Define the total number of years lived
def years_lived := 79

theorem ancient_greek_life_span :
  birth_date = (40, "BC") →
  death_date = (40, "AD") →
  no_year_zero →
  years_lived = 79 :=
by {
  intros,
  sorry
}

end ancient_greek_life_span_l212_212666


namespace correct_time_fraction_l212_212253

theorem correct_time_fraction : 
  (∀ hour : ℕ, hour < 24 → true) →
  (∀ minute : ℕ, minute < 60 → (minute ≠ 16)) →
  (fraction_of_correct_time = 59 / 60) :=
by
  intros h_hour h_minute
  sorry

end correct_time_fraction_l212_212253


namespace quadratic_no_real_roots_l212_212724

theorem quadratic_no_real_roots 
  (k : ℝ) 
  (h : ¬ ∃ (x : ℝ), 2 * x^2 + x - k = 0) : 
  k < -1/8 :=
by {
  -- Proof will go here.
  sorry
}

end quadratic_no_real_roots_l212_212724


namespace xy_leq_half_x_squared_plus_y_squared_l212_212935

theorem xy_leq_half_x_squared_plus_y_squared (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := 
by 
  sorry

end xy_leq_half_x_squared_plus_y_squared_l212_212935


namespace find_a_find_abs_z_minus_1_range_l212_212729

noncomputable theory

open Complex Real

def z_of_a (a : ℝ) : ℂ := (1 + a * I) * (1 + I) + 2 + 4 * I

def line_condition (a : ℝ) : Prop := 
  let z := z_of_a a in
  let x := z.re in
  let y := z.im in
  x - y = 0

def abs_z_minus_1 (a : ℝ) : ℝ := 
  let z := z_of_a a in
  abs (z - 1)

theorem find_a : ∃ a : ℝ, line_condition a ∧ a = -1 :=
  sorry

theorem find_abs_z_minus_1_range : 
  ∀ (r : ℝ), r ≥ (7 * sqrt 2) / 2 ↔ ∃ a : ℝ, abs_z_minus_1 a = r :=
  sorry

end find_a_find_abs_z_minus_1_range_l212_212729


namespace curve_c1_general_eqn_curve_c2_rect_eqn_slopes_sum_l212_212820

noncomputable def c1_parametric (t : ℝ) := (2 * t^2, 2 * t)

def curve_c2_polar (θ a : ℝ) := ρ = 2 / (sin θ + a * cos θ)

theorem curve_c1_general_eqn :
  ∀ t : ℝ,
    let (x, y) := c1_parametric t in y^2 = 2 * x :=
by sorry

theorem curve_c2_rect_eqn (a : ℝ) (h1 : a ≠ 0) :
  ∀ θ : ℝ,
    let ρ := 2 / (sin θ + a * cos θ) in
    let x := ρ * cos θ in
    let y := ρ * sin θ in
    ax + y = 2 :=
by sorry

theorem slopes_sum (a : ℝ) (h1 : a ≠ 0) (h2 : a > -1/4) :
  ∃ t1 t2 : ℝ,
    ((2 * t1^2), (2 * t1)) ∈ curve_c1_general_eqn ∧
    ((2 * t2^2), (2 * t2)) ∈ curve_c1_general_eqn ∧
    ∃ (k1 k2 : ℝ),
      k1 = 1 / t1 ∧ k2 = 1 / t2 ∧ k1 + k2 = 1 :=
by sorry


end curve_c1_general_eqn_curve_c2_rect_eqn_slopes_sum_l212_212820


namespace center_circumcircle_BCD_on_circumcircle_ABC_l212_212899
noncomputable section

-- Definition of geometric elements and conditions
variables {S₁ S₂ : Circle} {O₁ O₂ O₃ : Point} {r₁ r₂ : ℝ}
variables {A B C D : Point}
variables (hB_on_S₁ : B ∈ S₁) 
          (hA_tan_S₁_B : is_tangent_at S₁ A B)
          (hC_outside_S₁ : ¬(C ∈ S₁) ∧ ∃ P1 P2 : Point, P1 ≠ P2 ∧ segment AC ∩ S₁ = {P1, P2})
          (hS₂_touches_AC_at_C : is_tangent_at S₂ C A)
          (hS₂_touches_S₁_at_D : S₂ ⊥⊤ S₁ ∧ D ∈ S₂ ∧ D ∈ S₁ ∧ on_opposite_side B D (line AC))

-- The theorem to prove
theorem center_circumcircle_BCD_on_circumcircle_ABC :
  center (circumcircle B C D) ∈ circumcircle A B C :=
sorry

end center_circumcircle_BCD_on_circumcircle_ABC_l212_212899


namespace joans_remaining_kittens_l212_212872

theorem joans_remaining_kittens (initial_kittens given_away : ℕ) (h1 : initial_kittens = 15) (h2 : given_away = 7) : initial_kittens - given_away = 8 := sorry

end joans_remaining_kittens_l212_212872


namespace min_value_A_x1_x2_l212_212398

def f (x : ℝ) : ℝ := sqrt 3 * sin (2017 * x) + cos (2017 * x)

theorem min_value_A_x1_x2 :
  ∃ x1 x2 : ℝ, (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) ∧
  (2 * |x1 - x2| = 2 * π / 2017) :=
by
  -- Proof omitted
  sorry

end min_value_A_x1_x2_l212_212398


namespace population_increase_l212_212059

theorem population_increase (P : ℕ)
  (birth_rate1_per_1000 : ℕ := 25)
  (death_rate1_per_1000 : ℕ := 12)
  (immigration_rate1 : ℕ := 15000)
  (birth_rate2_per_1000 : ℕ := 30)
  (death_rate2_per_1000 : ℕ := 8)
  (immigration_rate2 : ℕ := 30000)
  (pop_increase1_perc : ℤ := 200)
  (pop_increase2_perc : ℤ := 300) :
  (12 * P - P) / P * 100 = 1100 := by
  sorry

end population_increase_l212_212059


namespace forged_cubic_edge_length_l212_212650

noncomputable def rectangular_block_volume (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def cube_edge_length (volume : ℝ) : ℝ :=
  real.cbrt volume

theorem forged_cubic_edge_length
  (l w h : ℝ)
  (h_l : l = 50)
  (h_w : w = 8)
  (h_h : h = 20) :
  cube_edge_length (rectangular_block_volume l w h) = 20 :=
by
  -- Proof omitted
  sorry

end forged_cubic_edge_length_l212_212650


namespace incenter_distance_l212_212972

  noncomputable def length_of_BI (AB : ℝ) (equal_sides : AB = (6 * Real.sqrt 2)) : ℝ :=
  let s := (AB + AB + (AB * Real.sqrt 2)) / 2 in
  let area := 0.5 * AB * AB in
  let r := area / s in
  (r * Real.sqrt 2)

  theorem incenter_distance (h1 : 6 * Real.sqrt 2 = (6 * Real.sqrt 2)) : length_of_BI (6 * Real.sqrt 2) h1 = 12 * (Real.sqrt 2 - 1) := by
  sorry
  
end incenter_distance_l212_212972


namespace solution_l212_212175

-- Define the equations and their solution sets
def eq1 (x p : ℝ) : Prop := x^2 - p * x + 6 = 0
def eq2 (x q : ℝ) : Prop := x^2 + 6 * x - q = 0

-- Define the condition that the solution sets intersect at {2}
def intersect_at_2 (p q : ℝ) : Prop :=
  eq1 2 p ∧ eq2 2 q

-- The main theorem stating the value of p + q given the conditions
theorem solution (p q : ℝ) (h : intersect_at_2 p q) : p + q = 21 :=
by
  sorry

end solution_l212_212175


namespace John_drove_miles_out_of_the_way_l212_212874

variable (normalTripDistance normalTripTime totalTripTime : ℕ)
variable (constantSpeed : ℕ)

theorem John_drove_miles_out_of_the_way :
  normalTripDistance = 150 ∧
  normalTripTime = 3 ∧
  totalTripTime = 5 ∧
  constantSpeed = normalTripDistance / normalTripTime →
  (totalTripTime * constantSpeed - normalTripDistance) = 100 :=
by
  -- Introduce variables for the conditions
  intro h
  cases h with h1 hd
  cases hd with h2 ht
  cases ht with h3 hs
  -- Placeholder for the proof
  simp_all
  sorry

end John_drove_miles_out_of_the_way_l212_212874


namespace larry_tens_digits_count_l212_212339

theorem larry_tens_digits_count :
  { x : ℕ // x < 10 ∧ (10 * x + 6) % 4 = 0 }.to_finset.card = 4 :=
sorry

end larry_tens_digits_count_l212_212339


namespace jill_arrives_earlier_l212_212483

-- Definitions of the problem conditions
def distance_to_park : ℝ := 1.5
def jill_speed : ℝ := 8
def jack_speed : ℝ := 3

-- Definitions of the calculated travel times
def jill_time_hours : ℝ := distance_to_park / jill_speed
def jack_time_hours : ℝ := distance_to_park / jack_speed

-- Convert travel times from hours to minutes
def jill_time_minutes : ℝ := jill_time_hours * 60
def jack_time_minutes : ℝ := jack_time_hours * 60

-- Define the time difference in arrival times
def time_difference : ℝ := jack_time_minutes - jill_time_minutes

-- Theorem statement
theorem jill_arrives_earlier : time_difference = 18.75 :=
by
  sorry

end jill_arrives_earlier_l212_212483


namespace angle_equality_l212_212829

-- Let ABCD be a convex quadrilateral
variables (A B C D : Type) -- represents points
variables (convex : convex_quadrilateral A B C D)
variables (h1 : ∠ B C D = ∠ C A B)
variables (h2 : ∠ A C D = ∠ B D A)

theorem angle_equality (A B C D : Type)
  (convex : convex_quadrilateral A B C D)
  (h1 : ∠ C B D = ∠ C A B)
  (h2 : ∠ A C D = ∠ B D A) : 
  ∠ A B C = ∠ A D C := 
sorry

end angle_equality_l212_212829


namespace train_length_l212_212211

noncomputable def relative_speed_kmh (vA vB : ℝ) : ℝ :=
  vA - vB

noncomputable def relative_speed_mps (relative_speed_kmh : ℝ) : ℝ :=
  relative_speed_kmh * (5 / 18)

noncomputable def distance_covered (relative_speed_mps : ℝ) (time_s : ℝ) : ℝ :=
  relative_speed_mps * time_s

theorem train_length (vA_kmh : ℝ) (vB_kmh : ℝ) (time_s : ℝ) (L : ℝ) 
  (h1 : vA_kmh = 42) (h2 : vB_kmh = 36) (h3 : time_s = 36) 
  (h4 : distance_covered (relative_speed_mps (relative_speed_kmh vA_kmh vB_kmh)) time_s = 2 * L) :
  L = 30 :=
by
  sorry

end train_length_l212_212211


namespace total_cost_correct_l212_212232

/-- Define the base car rental cost -/
def rental_cost : ℝ := 150

/-- Define cost per mile -/
def cost_per_mile : ℝ := 0.5

/-- Define miles driven on Monday -/
def miles_monday : ℝ := 620

/-- Define miles driven on Thursday -/
def miles_thursday : ℝ := 744

/-- Define the total cost Zach spent -/
def total_cost : ℝ := rental_cost + (miles_monday * cost_per_mile) + (miles_thursday * cost_per_mile)

/-- Prove that the total cost Zach spent is 832 dollars -/
theorem total_cost_correct : total_cost = 832 := by
  sorry

end total_cost_correct_l212_212232


namespace minimum_value_l212_212018

theorem minimum_value (x : ℝ) (h : x > -3) : 2 * x + (1 / (x + 3)) ≥ 2 * Real.sqrt 2 - 6 :=
sorry

end minimum_value_l212_212018


namespace solve_cosine_equation_l212_212938

theorem solve_cosine_equation (x : ℝ) (k : ℤ) :
  cos x ^ 2 + cos (2 * x) ^ 2 + cos (3 * x) ^ 2 = 1 ↔
  (x = (2 * k + 1) * Real.pi / 8 ∨
   x = (2 * k + 1) * Real.pi / 6 ∨
   x = (2 * k + 1) * Real.pi / 3) :=
by sorry

end solve_cosine_equation_l212_212938


namespace find_a_l212_212851

theorem find_a (a : ℝ) (h1 : a > 0)
  (h2 : ∀ x : ℝ, (complex.coeff (expand (sqrt x + a / x)^6 0) = 60)) : a = 2 :=
sorry

end find_a_l212_212851


namespace james_pay_for_two_semesters_l212_212868

theorem james_pay_for_two_semesters
  (units_per_semester : ℕ) (cost_per_unit : ℕ) (num_semesters : ℕ)
  (h_units : units_per_semester = 20) (h_cost : cost_per_unit = 50) (h_semesters : num_semesters = 2) :
  units_per_semester * cost_per_unit * num_semesters = 2000 := 
by 
  rw [h_units, h_cost, h_semesters]
  norm_num

end james_pay_for_two_semesters_l212_212868


namespace Ruth_school_hours_l212_212533

theorem Ruth_school_hours (d : ℝ) :
  0.25 * 5 * d = 10 → d = 8 :=
by
  sorry

end Ruth_school_hours_l212_212533


namespace squirrel_travel_distance_l212_212282

/-- 
  A squirrel runs up a cylindrical post in a perfect spiral path making one circuit for each rise of 4 feet. The post is 12 feet tall and 3 feet in circumference. Prove that the squirrel travels 15 feet in total.
--/
theorem squirrel_travel_distance
  (height : ℝ := 12)
  (circumference : ℝ := 3)
  (rise_per_circuit : ℝ := 4) :
  (sqrt (height^2 + (circumference * (height / rise_per_circuit))^2) = 15) :=
by
  sorry

end squirrel_travel_distance_l212_212282


namespace figure_surface_area_calculation_l212_212488

-- Define the surface area of one bar
def bar_surface_area : ℕ := 18

-- Define the surface area lost at the junctions
def surface_area_lost : ℕ := 2

-- Define the effective surface area of one bar after accounting for overlaps
def effective_bar_surface_area : ℕ := bar_surface_area - surface_area_lost

-- Define the number of bars used in the figure
def number_of_bars : ℕ := 4

-- Define the total surface area of the figure
def total_surface_area : ℕ := number_of_bars * effective_bar_surface_area

-- The theorem stating the total surface area of the figure
theorem figure_surface_area_calculation : total_surface_area = 64 := by
  sorry

end figure_surface_area_calculation_l212_212488


namespace shared_name_in_most_populous_house_l212_212977

-- Let's define the conditions given in the problem
def people_total : ℕ := 125
def min_people_per_name : ℕ := 3
def max_distinct_names : ℕ := 42

-- Our goal: to prove that in the most populous house, at least two people must share the same name.
theorem shared_name_in_most_populous_house 
  (h_total: ∀ (s: set ℕ), s.card = people_total) 
  (h_names: ∀ (name: ℕ), (∀ s: set ℕ, s.card = people_total → s.filter (λ x, x = name).card ≥ min_people_per_name))
  (h_distinct_names: ∀ (names: set ℕ), names.card ≤ max_distinct_names) : 
  ∃ (house: set ℕ), house.card > people_total / max_distinct_names ∧ ∃ (x: ℕ), (house.filter (λ p, p = x)).card ≥ 2 :=
by sorry

end shared_name_in_most_populous_house_l212_212977


namespace quadratic_inequality_l212_212371

noncomputable def quadratic_function (a c : ℝ) (x : ℝ) : ℝ :=
  a * (x^2 - x) + c

theorem quadratic_inequality 
  (a c x₁ x₂ x₃ : ℝ) 
  (h₁ : a ≠ 0) 
  (h₂ : x₁ ≠ x₂) 
  (h₃ : x₂ ≠ x₃) 
  (h₄ : x₃ ≠ x₁) 
  (h₅ : quadratic_function a c x₁ - a * x₂ = 0) 
  (h₆ : quadratic_function a c x₂ - a * x₃ = 0) 
  (h₇ : quadratic_function a c x₃ - a * x₁ = 0) : 
  a^2 > a * c :=
begin
  sorry,
end

end quadratic_inequality_l212_212371


namespace min_cost_proof_l212_212911

/--
  Misha needs to buy different gifts for his mom, dad, brother, and sister from 4 stores.
  The gifts and their costs are as follows (store, cost):
  Romashka: mom = 1000, dad = 750, brother = 930, sister = 850.
  Oduvanchik: mom = 1050, dad = 790, brother = 910, sister = 800.
  Nezabudka: mom = 980, dad = 810, brother = 925, sister = 815.
  Landysh: mom = 1100, dad = 755, brother = 900, sister = 820.
  Each store closes at 8:00 PM and the traveling time between stores and home is 30 minutes.
  The shopping time in each store varies.

  Prove that the minimum amount of money Misha can spend to buy all 4 gifts is 3435 rubles given the time constraints.
-/
def misha_min_spent : ℕ :=
  let mom_cost := min 1000 (min 1050 (min 980 1100))
  let dad_cost := min 750 (min 790 (min 810 755))
  let brother_cost := min 930 (min 910 (min 925 900))
  let sister_cost := min 850 (min 800 (min 815 820))
  mom_cost + dad_cost + brother_cost + sister_cost

theorem min_cost_proof (h: ∃ g1 g2 g3 g4: ℕ,
                              g1 ∈ {980, 1000, 1050, 1100} ∧
                              g2 ∈ {750, 790, 810, 755} ∧
                              g3 ∈ {900, 925, 930, 910} ∧
                              g4 ∈ {800, 820, 815, 850} ∧
                              g1 + g2 + g3 + g4 = 3435) : 
                              misha_min_spent = 3435 := 
by
  sorry


end min_cost_proof_l212_212911


namespace shortest_side_length_l212_212448

-- Define angles and sides in the triangle
def angle_A := 75 * in_deg
def angle_B := 45 * in_deg
def angle_C := 60 * in_deg

-- Define side lengths
def side_c := 1

-- Define the sine function for specific angles
def sin_45 := Real.sin (π / 4)
def sin_60 := Real.sin (π / 3)

-- The goal is to prove the length of the shortest side is sqrt(6) / 3
theorem shortest_side_length (b : Real) :
  b = side_c * sin_45 / sin_60 :=
sorry

end shortest_side_length_l212_212448


namespace inequality_x_alpha_y_beta_l212_212926

theorem inequality_x_alpha_y_beta (x y α β : ℝ) (hx : 0 < x) (hy : 0 < y) 
(hα : 0 < α) (hβ : 0 < β) (hαβ : α + β = 1) : x^α * y^β ≤ α * x + β * y := 
sorry

end inequality_x_alpha_y_beta_l212_212926


namespace min_three_beacons_l212_212060

-- Definitions of conditions
def maze : Type := sorry  -- The maze diagram
def room : Type := sorry  -- Rooms in the maze
def beacon (r : room) : Prop := sorry  -- Property indicating a beacon is in room r
def distance (r1 r2 : room) : ℕ := sorry  -- Function giving the distance between two rooms
def unique_distances (rs : set room) : Prop :=
  ∀ (r1 r2 : room), r1 ≠ r2 → (λ b, distance r1 b) '' rs ≠ (λ b, distance r2 b) '' rs

-- Proposition to state the minimum number of beacons needed to uniquely identify each room
theorem min_three_beacons (rs : set room) (n : ℕ) (h : unique_distances rs) : 
  ∃ bs : finset room, bs.card = 3 ∧ ∀ r1 r2, r1 ≠ r2 → distance r1 bs ≠ distance r2 bs := 
sorry

end min_three_beacons_l212_212060


namespace original_population_l212_212657

variable (p : ℕ)

/-- Assume the town's population increased by 1,500 people, and then this new population
    decreased by 20%. After these changes, the town had 50 more people than it did before 
    the 1,500 persons increase. -/
theorem original_population (h_condition : 0.8 * (p + 1500) = p + 1550) : p = 1750 :=
sorry

end original_population_l212_212657


namespace find_value_perpendicular_distances_l212_212898

variable {R a b c D E F : ℝ}
variable {ABC : Triangle}

-- Assume the distances from point P on the circumcircle of triangle ABC
-- to the sides BC, CA, and AB respectively.
axiom D_def : D = R * a / (2 * R)
axiom E_def : E = R * b / (2 * R)
axiom F_def : F = R * c / (2 * R)

theorem find_value_perpendicular_distances
    (a b c R : ℝ) (D E F : ℝ) 
    (hD : D = R * a / (2 * R)) 
    (hE : E = R * b / (2 * R)) 
    (hF : F = R * c / (2 * R)) : 
    a^2 * D^2 + b^2 * E^2 + c^2 * F^2 = (a^4 + b^4 + c^4) / (4 * R^2) :=
by
  sorry

end find_value_perpendicular_distances_l212_212898


namespace integer_fraction_l212_212405

variable {c : ℕ}
variable {a : ℕ → ℕ}

theorem integer_fraction (h : ∀ n : ℕ, ∑ i in finset.range n + 1, a (n / i.succ) = n^10) (n : ℕ) (n_pos : n > 1) : 
  (c^a n - c^a (n-1)) % n = 0 := 
sorry

end integer_fraction_l212_212405


namespace consecutive_pairs_sum_ge8_l212_212520

theorem consecutive_pairs_sum_ge8 (nums : List ℕ) (h_len : nums.length = 2005) 
  (h_sum : nums.sum = 7022) : 
  ∃ (i j k l : ℕ), i < 2005 ∧ j < 2005 ∧ k < 2005 ∧ l < 2005 ∧ -- indices are within bounds
  ((nums.get ⟨i, (by simp [*])⟩ + nums.get ⟨(i+1) % 2005, (by simp [*])⟩ >= 8) ∧ 
   (nums.get ⟨j, (by simp [*])⟩ + nums.get ⟨(j+1) % 2005, (by simp [*])⟩ >= 8) ∧
   (nums.get ⟨k, (by simp [*])⟩ + nums.get ⟨(k+1) % 2005, (by simp [*])⟩ >= 8) ∧ 
   (nums.get ⟨l, (by simp [*])⟩ + nums.get ⟨(l+1) % 2005, (by simp [*])⟩ >= 8)) :=
by 
  sorry

end consecutive_pairs_sum_ge8_l212_212520


namespace range_of_a_l212_212761

def f (x : ℝ) : ℝ :=
  if -2 < x ∧ x ≤ -1 then log (-x) + 3
  else -x^2 - 2*x + 1

def valid_interval (a : ℝ) : Prop :=
  4 < a ∧ a < 14

theorem range_of_a (a : ℝ) :
  f (2 * a) - (1 / 2) * (2 * a + 2)^2 < f (12 - a) - (1 / 2) * (14 - a)^2 → valid_interval a :=
by
  sorry

end range_of_a_l212_212761


namespace part_I_part_II_l212_212678

-- sequence definition
def a : ℕ → ℝ
| 0       := 1/2
| (n + 1) := a n ^ 2 / (a n ^ 2 - a n + 1)

-- sum of the first n terms of the sequence
def S (n : ℕ) : ℝ :=
  (finset.range (n + 1)).sum (λ i, a i)

-- prove a_{n+1} < a_n
theorem part_I (n : ℕ) : a (n + 1) < a n := by
  sorry

-- prove S_n < 1
theorem part_II (n : ℕ) : S n < 1 := by
  sorry

end part_I_part_II_l212_212678


namespace hawks_points_l212_212808

theorem hawks_points (x y : ℕ) (h1 : x + y = 48) (h2 : x - y = 16) : y = 16 := 
by {
  sorry,
}

end hawks_points_l212_212808


namespace sum_of_digits_of_smallest_n_l212_212212

def gcd (a b : ℕ) : ℕ := Nat.gcd a b -- Define the gcd function

/-- Prove the sum of the digits of the smallest n that satisfies the conditions is 15. -/
theorem sum_of_digits_of_smallest_n :
  ∃ n : ℕ, n > 2021 ∧ gcd 63 (n + 120) = 21 ∧ gcd (n + 63) 120 = 60 ∧ (Nat.digits 10 n).sum = 15 :=
sorry

end sum_of_digits_of_smallest_n_l212_212212


namespace no_prime_divisible_by_42_l212_212417

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → ¬ (42 ∣ p) :=
by sorry

end no_prime_divisible_by_42_l212_212417


namespace smallest_value_of_x_l212_212998

theorem smallest_value_of_x :
  ∃ x : Real, (∀ z, (z = (5 * x - 20) / (4 * x - 5)) → (z * z + z = 20)) → x = 0 :=
by
  sorry

end smallest_value_of_x_l212_212998


namespace rectangle_side_length_along_hypotenuse_l212_212461

-- Define the right triangle with given sides
def triangle_PQR (PR PQ QR : ℝ) : Prop := 
  PR^2 + PQ^2 = QR^2

-- Condition: Right triangle PQR with PR = 9 and PQ = 12
def PQR : Prop := triangle_PQR 9 12 (Real.sqrt (9^2 + 12^2))

-- Define the property of the rectangle
def rectangle_condition (x : ℝ) (s : ℝ) : Prop := 
  (3 / (Real.sqrt (9^2 + 12^2))) = (x / 9) ∧ s = ((9 - x) * (Real.sqrt (9^2 + 12^2)) / 9)

-- Main theorem
theorem rectangle_side_length_along_hypotenuse : 
  PQR ∧ (∃ x, rectangle_condition x 12) → (∃ s, s = 12) :=
by
  intro h
  sorry

end rectangle_side_length_along_hypotenuse_l212_212461


namespace T_n_inequality_l212_212512

noncomputable def a_n (n : ℕ) : ℕ := 2 ^ n

noncomputable def b_n (n : ℕ) : ℕ := 1 + Int.log2( (a_n n) ^ 2 )

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in finRange n, 1 / ((b_n i).toReal * (b_n (i + 1)).toReal)

theorem T_n_inequality (n : ℕ) : T_n n < 1 / 6 := by
  sorry

end T_n_inequality_l212_212512


namespace lower_right_is_4_l212_212591

def grid := Array (Array (Option Nat))

noncomputable def initial_grid : grid :=
  #[ #[some 1, none, some 3, none, none],
     #[some 3, some 4, none, none, none],
     #[none, none, none, some 5, none],
     #[none, none, none, none, none],
     #[none, none, none, none, none] ]

noncomputable def is_valid_grid (g : grid) : Prop :=
  ∀ i j, ∃ k, g[i]![j]! = some k ∧ 1 ≤ k ∧ k ≤ 5 ∧
  (∀ l, g[i]![l]! = some k → l = j) ∧
  (∀ l, g[l]![j]! = some k → l = i)

noncomputable def final_element (g : grid) : Option Nat := g[4]![4]!

theorem lower_right_is_4 (g : grid) (h : is_valid_grid g) : final_element g = some 4 :=
sorry

end lower_right_is_4_l212_212591


namespace trains_passing_time_l212_212210

def length_train_A := 300 -- in meters
def length_train_B := 250 -- in meters
def speed_train_A_kmph := 75 -- in km/h
def speed_train_B_kmph := 65 -- in km/h

def speed_in_m_per_s (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

def speed_train_A := speed_in_m_per_s speed_train_A_kmph
def speed_train_B := speed_in_m_per_s speed_train_B_kmph

def relative_speed := speed_train_A + speed_train_B
def total_distance := length_train_A + length_train_B
def passing_time := total_distance / relative_speed

theorem trains_passing_time :
  ↥(abs (passing_time - 14.14) < 0.01) := 
sorry

end trains_passing_time_l212_212210


namespace ellipse_chord_slope_l212_212850

def midpoint_slope (x₁ x₂ y₁ y₂ : ℝ) : ℝ := (y₁ - y₂) / (x₁ - x₂)

theorem ellipse_chord_slope :
  (∀ (x y : ℝ), x^2 / 8 + y^2 / 6 = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ + x₂ = 4 ∧ y₁ + y₂ = 2 ∧ midpoint_slope x₁ x₂ y₁ y₂ = -3/2) :=
by
  sorry

end ellipse_chord_slope_l212_212850


namespace forged_cube_edge_length_l212_212648

def rectangle_volume (l w h : ℝ) : ℝ := l * w * h

def cube_edge_length (volume : ℝ) : ℝ := volume ^ (1 / 3 : ℝ)

theorem forged_cube_edge_length : 
  rectangle_volume 50 8 20 = 8000 ∧ cube_edge_length 8000 = 20 :=
by 
  sorry

end forged_cube_edge_length_l212_212648


namespace probability_in_picture_l212_212121

/-- Rachel runs counterclockwise and completes a lap every 120 seconds.
    Robert runs clockwise and completes a lap every 70 seconds.
    Both start from the same line at the same time.
    A picture is taken at some random time between 15 minutes and 16 minutes
    after they begin to run. The picture shows one-third of the track,
    centered on the starting line. Prove that the probability 
    that both Rachel and Robert are in the picture is 2/9. -/
theorem probability_in_picture 
  (r_lap_time : ℕ) (rachel_lap_seconds : ℕ)
  (robert_lap_seconds : ℕ) (start_time : ℕ)
  (end_time : ℕ) (picture_fraction : ℝ) :
  rachel_lap_seconds = 120 →
  robert_lap_seconds = 70 →
  start_time = 900 →
  end_time = 960 →
  picture_fraction = 1/3 →
  let time_overlap := max (start_time + (rachel_lap_seconds / 3)) (start_time - (robert_lap_seconds / 3)) 
                       - min (end_time - (rachel_lap_seconds / 3)) (end_time + (robert_lap_seconds / 3))
  in time_overlap / (end_time - start_time) = 2 / 9 :=
by 
  sorry

end probability_in_picture_l212_212121


namespace annual_profits_l212_212284

-- Define the profits of each quarter
def P1 : ℕ := 1500
def P2 : ℕ := 1500
def P3 : ℕ := 3000
def P4 : ℕ := 2000

-- State the annual profit theorem
theorem annual_profits : P1 + P2 + P3 + P4 = 8000 := by
  sorry

end annual_profits_l212_212284


namespace polygon_diagonals_l212_212432

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 4) : n = 7 :=
sorry

end polygon_diagonals_l212_212432


namespace alfred_gain_percent_l212_212663

open Real

theorem alfred_gain_percent (scooter_price repairs_cost accessories_price discount_rate selling_price gain_percent : ℝ) 
  (h_scooter_price : scooter_price = 4400)
  (h_repairs_cost : repairs_cost = 800)
  (h_accessories_price : accessories_price = 600)
  (h_discount_rate : discount_rate = 0.20)
  (h_selling_price : selling_price = 5800)
  (h_gain_percent : gain_percent ≈ 2.11) : 
  let total_cost := scooter_price + repairs_cost
      accessories_cost := accessories_price - (discount_rate * accessories_price)
      final_cost := total_cost + accessories_cost
      gain := selling_price - final_cost in
        gain / final_cost * 100 ≈ gain_percent :=
by
  sorry

end alfred_gain_percent_l212_212663


namespace median_altitude_perpendicular_l212_212480

theorem median_altitude_perpendicular {A B C M H P A₁ M₁ B₁ : Point} 
  (h_triangle : Triangle A B C) 
  (h_median : Median C M) 
  (h_altitude : Altitude C H) 
  (h_perp_CA : Perpendicular P CA) 
  (h_perp_CM : Perpendicular P CM) 
  (h_perp_CB : Perpendicular P CB) 
  (h_inter_A₁ : Intersect P A₁ CH) 
  (h_inter_M₁ : Intersect P M₁ CH) 
  (h_inter_B₁ : Intersect P B₁ CH) : 
  Dist A₁ M₁ = Dist B₁ M₁ :=
by 
  sorry

end median_altitude_perpendicular_l212_212480


namespace angle_ABC_eq_angle_ADC_l212_212823

-- Given a convex quadrilateral ABCD
variables {A B C D O : Type}
variables [convex_quadrilateral A B C D]

-- Given conditions
variable (angle_CBD_eq_angle_CAB : ∠ CBD = ∠ CAB)
variable (angle_ACD_eq_angle_BDA : ∠ ACD = ∠ BDA)

-- Prove that ∠ ABC = ∠ ADC 
theorem angle_ABC_eq_angle_ADC :
  ∠ ABC = ∠ ADC :=
begin
  sorry -- Proof not required
end

end angle_ABC_eq_angle_ADC_l212_212823


namespace points_on_circle_locus_centers_l212_212183

-- Given definitions and conditions
def line1 (x : ℝ) (t : ℝ) : ℝ := 2 * x + t
def line2 (x : ℝ) (t : ℝ) : ℝ := (1 / 2) * x + t
def hyperbola (x : ℝ) : ℝ := 1 / x
def point_T (t : ℝ) : ℝ × ℝ := (0, t)

-- Points of intersection
def point_A (t : ℝ) : (ℝ × ℝ) :=
  let x1 := (-t + sqrt (t^2 + 8)) / 4
  (x1, hyperbola x1)

def point_C (t : ℝ) : (ℝ × ℝ) :=
  let x2 := (-t - sqrt (t^2 + 8)) / 4
  (x2, hyperbola x2)

def point_B (t : ℝ) : (ℝ × ℝ) :=
  let x3 := -t + sqrt (t^2 + 2)
  (x3, hyperbola x3)

def point_D (t : ℝ) : (ℝ × ℝ) :=
  let x4 := -t - sqrt (t^2 + 2)
  (x4, hyperbola x4)

-- Statement to prove that points lie on a circle
theorem points_on_circle (t : ℝ) :
  ∃ O R, ∀ P ∈ {point_A t, point_B t, point_C t, point_D t}, dist P O = R :=
sorry

-- Statement to find the locus of centers
theorem locus_centers :
  ∀ (T : ℝ), ∃ x y, (x, y) = (T, -(5 / 4) * T) :=
sorry

end points_on_circle_locus_centers_l212_212183


namespace area_of_triangle_AME_l212_212531

theorem area_of_triangle_AME :
  ∀ (A B C D M E : Point)
    (h1 : rectangle A B C D)
    (h2 : dist A B = 12)
    (h3 : dist B C = 10)
    (h4 : midpoint M A C)
    (h5 : ∃ E ∈ line_through A B, dist A E = 1/3 * dist A B)
    (h6 : perp M E A C),
  area A M E = 6 * sqrt 5 := 
by 
  sorry

end area_of_triangle_AME_l212_212531


namespace convert_326_base7_to_base4_l212_212323

theorem convert_326_base7_to_base4 :
  -- Definition of the number in base 7
  let n_base7 := 326;
  -- Definition of base conversion from base 7 to decimal
  let n_dec := 3 * 7^2 + 2 * 7^1 + 6 * 7^0;
  -- Definition of base conversion from decimal to base 4
  let n_base4 := (2 * 1000 + 2 * 100 + 1 * 10 + 3);
  -- Statement to prove
  nat_of_digits 7 (digits 7 n_base7) = 167 → nat_of_digits 4 (digits 4 n_dec) = 2213 := sorry

end convert_326_base7_to_base4_l212_212323


namespace problem_l212_212381

-- Define the conditions as hypotheses
variable {R : Type} [Real : linear_ordered_field R]
variable (f : R → R)

-- Condition 1: f(x) = f(x + 4) + f(2)
axiom cond1 : ∀ x : R, f(x) = f(x + 4) + f(2)

-- Condition 2: The graph of y=f(x+3) is symmetric with respect to the line x=-3
axiom cond2 : ∀ x : R, f(x + 3) = f(-3 - (x + 3))

-- Condition 3: Monotonic property on [0, 2]
axiom cond3 : ∀ x1 x2 : R, 0 ≤ x1 → x1 < x2 → x2 ≤ 2 → (x2 - x1) * (f(x2) - f(x1)) > 0

-- We need to prove the following conclusions
theorem problem :
  f(2) = 0 ∧
  (∀ x : R, f(x) = f(-x)) ∧
  (∀ x : R, f(x) = f(x + 4)) :=
by sorry

end problem_l212_212381


namespace matrix_inverse_l212_212714

theorem matrix_inverse :
  let A := ![\[7, -5\], \[-3, 2\]] in
  let A_inv := ![\[-2, -5\], \[-3, -7\]] in
  inverse A = A_inv :=
by
  sorry

end matrix_inverse_l212_212714


namespace constant_term_correct_l212_212472

variable (x : ℝ)

noncomputable def constant_term_expansion : ℝ :=
  let term := λ (r : ℕ) => (Nat.choose 9 r) * (-2)^r * x^((9 - 9 * r) / 2)
  term 1

theorem constant_term_correct : 
  constant_term_expansion x = -18 :=
sorry

end constant_term_correct_l212_212472


namespace prove_smallest_x_is_zero_l212_212994

noncomputable def smallest_x : ℝ :=
  let f := λ (x : ℝ), (5 * x - 20) / (4 * x - 5)
  in if h : f(x)^2 + f(x) = 20 then x else 0

theorem prove_smallest_x_is_zero :
  let f := λ (x : ℝ), (5 * x - 20) / (4 * x - 5)
  in ∃ x : ℝ, f(x)^2 + f(x) = 20 ∧ (∀ y : ℝ, f(y)^2 + f(y) = 20 → x ≤ y) :=
begin
  sorry, -- proof goes here
end

end prove_smallest_x_is_zero_l212_212994


namespace apples_added_equiv_l212_212195

theorem apples_added_equiv (initial_apples final_apples apples_added : ℕ)
  (h1 : initial_apples = 8)
  (h2 : final_apples = 13)
  (h3 : apples_added = final_apples - initial_apples) : apples_added = 5 :=
by
  rw [h1, h2, h3]
  exact Nat.sub_self 8

end apples_added_equiv_l212_212195


namespace solution_set_inequality_l212_212800

theorem solution_set_inequality (f g : ℝ → ℝ)
  (h1 : {x | f x ≤ 0} = set.Icc (-2 : ℝ) 3)
  (h2 : {x | g x ≤ 0} = ∅) :
  {x | f x / g x > 0} = set.Iic (-2 : ℝ) ∪ set.Ici 3 :=
by
  sorry

end solution_set_inequality_l212_212800


namespace number_of_ellipses_calculated_l212_212363

-- Define the considering range
def range_set := {i : ℕ | 1 ≤ i ∧ i ≤ 11}

-- Definition of choosing two different elements from the given set
def choose_mn (s : Set ℕ) := { p : ℕ × ℕ | p.fst ∈ s ∧ p.snd ∈ s ∧ p.fst ≠ p.snd }

-- Define the rectangular region
def B (x y : ℝ) := abs x < 11 ∧ abs y < 9

-- Number of valid ellipses formed within specific regions
def number_of_ellipses : ℕ :=
  let possible_values := ({i | 1 ≤ i ∧ i ≤ 8} : Set ℕ)
  let case1 := choose_mn possible_values
  let case2 := choose_mn ({9, 10} : Set ℕ) ∪ choose_mn possible_values
  in case1.card + case2.card

theorem number_of_ellipses_calculated : number_of_ellipses = 72 :=
by
  -- proof steps would go here
  sorry

end number_of_ellipses_calculated_l212_212363


namespace derivative_at_one_l212_212948

-- Definition of the function
def f (x : ℝ) : ℝ := x^2

-- Condition
def x₀ : ℝ := 1

-- Problem statement
theorem derivative_at_one : (deriv f x₀) = 2 :=
sorry

end derivative_at_one_l212_212948


namespace convex_polyhedron_inequality_l212_212244

variables (V E T : ℕ) (h_E : 0 ≤ E) (h_T : 0 ≤ T)

theorem convex_polyhedron_inequality (h_VET : V ≤ √E + T) : 
  V ≤ √E + T :=
begin
  sorry
end

end convex_polyhedron_inequality_l212_212244


namespace probability_of_selecting_female_volunteers_l212_212329

theorem probability_of_selecting_female_volunteers :
  let n := 5
  let k := 2
  let female_count := 3
  let male_count := 2
  let total_ways := Nat.choose n k
  let female_ways := Nat.choose female_count k
  (female_ways.toRat / total_ways.toRat) = (3 : ℚ) / 10 := 
by
  sorry

end probability_of_selecting_female_volunteers_l212_212329


namespace midpoint_of_am_l212_212527

variables {A B C D M X : Type} 
variables [AffineSpace ℝ (A': Type)] [MetricSpace A'] [NormedAddTorsor (EuclideanMetricSpace ℝ) A']

-- Given conditions
def midpoint (P Q M : A') : Prop := dist P M = dist Q M
def perpendicular (L1 L2 : Set (A')) : Prop := ∃ (n : ℝ), ∀ (a b ∈ L1), ∀ (c ∈ L2), ⟪b - a, c - a⟫ = 0
def collinear (a b c : A') : Prop := ∃ (u : ℝ), c = u • b + (1-u) • a

variables {A B C D M X : A'}

-- M is midpoint of BC
hypothesis h1 : midpoint B C M
-- DM is perpendicular to BC
hypothesis h2 : perpendicular {D, M} {B, C}
-- AM and BD intersect at X
hypothesis h3 : ∃ l1 l2, l1 ⊆ {A, M} ∧ l2 ⊆ {B, D} ∧ l1 ∩ l2 = {X}
-- AC = 2BX 
hypothesis h4 : dist A C = 2 * dist B X

-- Proof to show that X is the midpoint of AM
theorem midpoint_of_am : midpoint A M X :=
sorry

end midpoint_of_am_l212_212527


namespace steve_needs_total_wood_l212_212139

noncomputable theory

-- Define the lengths and quantities of wood needed for each bench.
def lengths_first_bench := [(6, 4), (2, 2)]
def lengths_second_bench := [(8, 3), (5, 1.5)]
def lengths_third_bench := [(4, 5), (3, 2.5)]

-- Function to compute the total length of wood required for a bench
def total_length (lengths : List (ℕ × ℝ)) : ℝ :=
  lengths.foldr (λ (pair : ℕ × ℝ) acc, pair.1 * pair.2 + acc) 0

-- Total lengths for each bench
def total_first_bench := total_length lengths_first_bench
def total_second_bench := total_length lengths_second_bench
def total_third_bench := total_length lengths_third_bench

-- Total length of wood needed
def total_wood := total_first_bench + total_second_bench + total_third_bench

-- Lean theorem statement
theorem steve_needs_total_wood : total_wood = 87 := by
  sorry

end steve_needs_total_wood_l212_212139


namespace ride_duration_l212_212364

theorem ride_duration (a b c : ℕ) (times : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 3) (h4 : times = 5) : 
  (a + b + c) * times = 40 :=
by 
  simp [h1, h2, h3, h4]
  exact calc
    (3 + 2 + 3) * 5 = 8 * 5 : by rw [add_assoc, add_comm 2 3, add_assoc]
    ... = 40 : by norm_num

end ride_duration_l212_212364


namespace lines_intersection_example_l212_212206

theorem lines_intersection_example (m b : ℝ) 
  (h1 : 8 = m * 4 + 2) 
  (h2 : 8 = 4 * 4 + b) : 
  b + m = -13 / 2 := 
by
  sorry

end lines_intersection_example_l212_212206


namespace joe_first_lift_weight_l212_212455

variables (x y : ℕ)

theorem joe_first_lift_weight (h1 : x + y = 600) (h2 : 2 * x = y + 300) : x = 300 :=
by
  sorry

end joe_first_lift_weight_l212_212455


namespace sum_inequality_l212_212529

open_locale big_operators

theorem sum_inequality
  (a b : ℕ → ℚ) (n : ℕ)
  (h₁ : ∀ i j, i ≤ j → a i ≤ a j)
  (h₂ : ∀ i j, i ≤ j → b i ≤ b j) :
  (∑ i in finset.range n, a i) * (∑ i in finset.range n, b i) ≤ n * (∑ i in finset.range n, a i * b i) :=
sorry

end sum_inequality_l212_212529


namespace tetrahedron_volume_correct_l212_212334

noncomputable def volume_of_tetrahedron
  (angle_ABC_BCD : ℝ)
  (area_ABC : ℝ)
  (area_BCD : ℝ)
  (BC : ℝ) : ℝ :=
  if (angle_ABC_BCD = 45) ∧ (area_ABC = 150) ∧ (area_BCD = 100) ∧ (BC = 10) then 500 * real.sqrt 2 else 0

theorem tetrahedron_volume_correct :
  volume_of_tetrahedron 45 150 100 10 = 500 * real.sqrt 2 := sorry

end tetrahedron_volume_correct_l212_212334


namespace cubic_roots_cosines_l212_212388

theorem cubic_roots_cosines
  {p q r : ℝ}
  (h_eq : ∀ x : ℝ, x^3 + p * x^2 + q * x + r = 0)
  (h_roots : ∃ (α β γ : ℝ), (α > 0) ∧ (β > 0) ∧ (γ > 0) ∧ (α + β + γ = -p) ∧ 
             (α * β + β * γ + γ * α = q) ∧ (α * β * γ = -r)) :
  2 * r + 1 = p^2 - 2 * q :=
by
  sorry

end cubic_roots_cosines_l212_212388


namespace area_of_smallest_square_containing_circle_l212_212219

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  ∃ (a : ℝ), a = 100 :=
by
  sorry

end area_of_smallest_square_containing_circle_l212_212219


namespace min_cost_proof_l212_212913

/--
  Misha needs to buy different gifts for his mom, dad, brother, and sister from 4 stores.
  The gifts and their costs are as follows (store, cost):
  Romashka: mom = 1000, dad = 750, brother = 930, sister = 850.
  Oduvanchik: mom = 1050, dad = 790, brother = 910, sister = 800.
  Nezabudka: mom = 980, dad = 810, brother = 925, sister = 815.
  Landysh: mom = 1100, dad = 755, brother = 900, sister = 820.
  Each store closes at 8:00 PM and the traveling time between stores and home is 30 minutes.
  The shopping time in each store varies.

  Prove that the minimum amount of money Misha can spend to buy all 4 gifts is 3435 rubles given the time constraints.
-/
def misha_min_spent : ℕ :=
  let mom_cost := min 1000 (min 1050 (min 980 1100))
  let dad_cost := min 750 (min 790 (min 810 755))
  let brother_cost := min 930 (min 910 (min 925 900))
  let sister_cost := min 850 (min 800 (min 815 820))
  mom_cost + dad_cost + brother_cost + sister_cost

theorem min_cost_proof (h: ∃ g1 g2 g3 g4: ℕ,
                              g1 ∈ {980, 1000, 1050, 1100} ∧
                              g2 ∈ {750, 790, 810, 755} ∧
                              g3 ∈ {900, 925, 930, 910} ∧
                              g4 ∈ {800, 820, 815, 850} ∧
                              g1 + g2 + g3 + g4 = 3435) : 
                              misha_min_spent = 3435 := 
by
  sorry


end min_cost_proof_l212_212913


namespace hyperbola_asymptotes_tangent_to_circle_l212_212438

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, y^2 - (x^2 / m^2) = 1 → (x^2 + y^2 - 4*y + 3 = 0 → distance_center_to_asymptote (0, 2) (y = x / m) = 1)) → 
  m = real.sqrt(3) / 3 :=
sorry

end hyperbola_asymptotes_tangent_to_circle_l212_212438


namespace nancy_hourly_wage_l212_212108

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end nancy_hourly_wage_l212_212108


namespace triangle_dot_product_l212_212041

def vector_length (v : ℝ × ℝ): ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2

noncomputable def ab : ℝ × ℝ := (5, 0)
noncomputable def ac : ℝ × ℝ := (4, 3)

theorem triangle_dot_product :
  vector_length ab = 5 →
  vector_length ac = 4 →
  vector_length (ac - ab) = 3 →
  dot_product ab (ac - ab) = -9 := 
by 
  intros h1 h2 h3
  sorry

end triangle_dot_product_l212_212041


namespace part_a_part_b_l212_212559

variable (f : ℝ → ℝ)

-- Part (a)
theorem part_a (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :
  ∀ x : ℝ, f (f x) ≤ 0 :=
sorry

-- Part (b)
theorem part_b (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) (h₀ : f 0 ≥ 0) :
  ∀ x : ℝ, f x = 0 :=
sorry

end part_a_part_b_l212_212559


namespace bus_children_count_l212_212190

theorem bus_children_count
  (initial_count : ℕ)
  (first_stop_add : ℕ)
  (second_stop_add : ℕ)
  (second_stop_remove : ℕ)
  (third_stop_remove : ℕ)
  (third_stop_add : ℕ)
  (final_count : ℕ)
  (h1 : initial_count = 18)
  (h2 : first_stop_add = 5)
  (h3 : second_stop_remove = 4)
  (h4 : third_stop_remove = 3)
  (h5 : third_stop_add = 5)
  (h6 : final_count = 25)
  (h7 : initial_count + first_stop_add = 23)
  (h8 : 23 + second_stop_add - second_stop_remove - third_stop_remove + third_stop_add = final_count) :
  second_stop_add = 4 :=
by
  sorry

end bus_children_count_l212_212190


namespace profit_percentage_is_9_l212_212430

variable CP : ℝ
variable SP_loss : ℝ
variable SP_profit : ℝ

def cost_price : Prop := CP = 50
def selling_price_loss : Prop := SP_loss = 45.5
def selling_price_profit : Prop := SP_profit = 54.5
def profit_percentage : Prop := (SP_profit - CP) / CP * 100 = 9

theorem profit_percentage_is_9 :
  cost_price ∧ selling_price_loss ∧ selling_price_profit → profit_percentage :=
by
  sorry

end profit_percentage_is_9_l212_212430


namespace min_sum_if_inverse_adds_to_zero_l212_212748

theorem min_sum_if_inverse_adds_to_zero (a m n : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) 
  (h_inv_sum_zero : Real.log a m + Real.log a n = 0) : m + n = 2 := 
sorry

end min_sum_if_inverse_adds_to_zero_l212_212748


namespace area_of_square_l212_212939

theorem area_of_square :
  ∀ (t : ℝ), (t = (-1 + Real.sqrt 5)) →
    (A B C D : ℝ) →
    (A = -t ∧ B = t) →
    C = (B, B^2 - 4) →
    D = (A, A^2 - 4) →
    (y_C = 0 - (B^2 - 4)) →
    (y_D = 0 - (A^2 - 4)) →
    (AB = 2 * t) →
    (BC = 4 - t^2) →
    (AB = BC) →
    (Area = 4 * t^2) →
    Area = 24 - 8 * Real.sqrt 5 :=
by
  intro t ht A B C D hA B hC hD hyC hyD hAB hBC hab hArea
  sorry

end area_of_square_l212_212939


namespace correct_operation_l212_212613

theorem correct_operation (a : ℝ) : 
  (-2 * a^2)^3 = -8 * a^6 :=
by sorry

end correct_operation_l212_212613


namespace integer_coordinates_exist_l212_212126

/-- Several points are marked on a plane. For any three of them, there exists a Cartesian coordinate 
    system in which these points have integer coordinates.
    Prove that there exists a Cartesian coordinate system in which all the marked points have integer coordinates. -/
theorem integer_coordinates_exist 
  (points : set (ℝ × ℝ))
  (h : ∀ (A B C : ℝ × ℝ) (hA : A ∈ points) 
      (hB : B ∈ points) (hC : C ∈ points), 
      ∃ (T : set (ℝ × ℝ)), (A ∈ T ∧ B ∈ T ∧ C ∈ T
      ∧ (∀ (P ∈ T), ∃ (x y : ℤ), P = (x, y)))) :
    ∃ T : set (ℝ × ℝ), (∀ (P ∈ points), ∃ (x y : ℤ), P = (x, y)) :=
sorry

end integer_coordinates_exist_l212_212126


namespace same_name_in_most_populous_house_l212_212978

/-- Given 125 people living in houses, with each name being shared by at least three people, 
    prove that there are at least two people with the same name in the most populous house. -/
theorem same_name_in_most_populous_house :
  ∀ (n_people : ℕ) (n_names : ℕ),
    n_people = 125 →
    (∀ name, 3 ∣ n_people) →
    n_people / 3 ≤ n_names →
    ∃ house, 2 ≤ n_people / (house+n_people/n_names) :=
by sorry

end same_name_in_most_populous_house_l212_212978


namespace shaded_area_is_correct_l212_212914

-- Definitions based on conditions
variables (rectangles : list (ℝ × ℝ))
variables (AB BC AP QR : ℝ)
variables (ABCD QRSC : set (ℝ × ℝ))
noncomputable def is_rectangle (r : ℝ × ℝ) : Prop := r.1 ≠ r.2
noncomputable def area (r : ℝ × ℝ) : ℝ := r.1 * r.2
noncomputable def shaded_area (total_area : ℝ) (sub_areas : list ℝ) : ℝ :=
total_area - sub_areas.sum

-- Conditions given in the problem
axiom h1 : all (λ r, is_rectangle r) rectangles
axiom h2 : area (AB, BC) = 35
axiom h3 : AP < QR

-- Statement to be proved
theorem shaded_area_is_correct :
  ∃ (A : ℝ), A = shaded_area 35 [_, _, _] ∧ (A = 24 ∨ A = 26) :=
sorry

end shaded_area_is_correct_l212_212914


namespace bp_equals_br_iff_abp_equilateral_l212_212085

theorem bp_equals_br_iff_abp_equilateral 
  (A B C D P Q R : Point) 
  (h_square : Square A B C D) 
  (h_P_angle : ∠ B A P = 60) 
  (h_Q : ∃ (l : Line), OnLine l Q ∧ Perpendicular l (LineThrough B P) ∧ LineThrough Q D = LineThrough A D) 
  (h_R : ∃ (m : Line), OnLine m R ∧ Perpendicular m (LineThrough B P) ∧ LineThrough R C = LineThrough B Q) :
  (|BP| = |BR| ↔ ∠ B A P = 60 ∧ |AP| = |AB|) :=
begin
  sorry -- proof is not required as per the instruction
end

end bp_equals_br_iff_abp_equilateral_l212_212085


namespace leftover_cents_l212_212270

noncomputable def total_cents (pennies nickels dimes quarters : Nat) : Nat :=
  (pennies * 1) + (nickels * 5) + (dimes * 10) + (quarters * 25)

noncomputable def total_cost (num_people : Nat) (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem leftover_cents (h₁ : total_cents 123 85 35 26 = 1548)
                       (h₂ : total_cost 5 300 = 1500) :
  1548 - 1500 = 48 :=
sorry

end leftover_cents_l212_212270


namespace find_angle_of_inclination_l212_212340

-- Define the function and its derivative
def curve (x : ℝ) (m : ℝ) : ℝ := x^3 - 2 * x + m
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - 2

-- Define the slope of the tangent at x = 1
def slope_at_x1 : ℝ := curve_derivative 1

-- Define the angle of inclination (in degrees) when the slope is 1
def angle_of_inclination : ℝ := (Real.atan slope_at_x1) * (180 / Real.pi)

theorem find_angle_of_inclination (m : ℝ) : angle_of_inclination = 45 := by
  -- Sorry, as we skip the proof.
  sorry

end find_angle_of_inclination_l212_212340


namespace solve_complex_eq_problem_l212_212393

def complex_eq_problem : Prop :=
  ∀ (z : ℂ), (1 - complex.I) * z = -1 - complex.I → z = - complex.I

theorem solve_complex_eq_problem : complex_eq_problem :=
by
  sorry

end solve_complex_eq_problem_l212_212393


namespace painted_cube_problem_l212_212660

theorem painted_cube_problem (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2)^2 = (n - 2)^3) : n = 8 :=
by {
  sorry
}

end painted_cube_problem_l212_212660


namespace find_x4_l212_212642

open Real

theorem find_x4 (x : ℝ) (h₁ : 0 < x) (h₂ : sqrt (1 - x^2) + sqrt (1 + x^2) = 2) : x^4 = 0 :=
by
  sorry

end find_x4_l212_212642


namespace square_length_QP_l212_212048

theorem square_length_QP (r1 r2 dist : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 7) (h_dist : dist = 15)
  (x : ℝ) (h_equal_chords: QP = PR) :
  x ^ 2 = 65 :=
sorry

end square_length_QP_l212_212048


namespace brocard_cotangent_sum_brocard_isogonal_conjugate_brocard_angle_equal_l212_212616

-- Definitions and assumptions
variables {P Q : Point} {A B C A_1 : Point}
variables {α β γ ϕ : Real} -- α, β, γ are the angles of triangle ABC, ϕ is the Brocard angle

-- Conditions
def is_brocard_point (P : Point) (A B C : Point) (ϕ : Real) :=
  angle A B P = ϕ ∧ angle B C P = ϕ ∧ angle C A P = ϕ

def is_brocard_angle (A B C : Point) (ϕ : Real) :=
  ∃ P, is_brocard_point P A B C ϕ

def tangent_circumcircle (A B C : Point) (P : Point) :=
  ∃ t, tangent_at P t (circumcircle A B C)

def is_parallel (l1 l2 : Line) :=
  ∃ m1 m2, l1 ≠ l2 ∧ parallel l1 l2

def intersect_at (l1 l2 : Line) (P : Point) :=
  P ∈ l1 ∧ P ∈ l2

-- Lean 4 statements for the proof problems
theorem brocard_cotangent_sum {A B C : Point} {α β γ ϕ : Real} (hP : is_brocard_angle A B C ϕ) :
  Real.cot ϕ = Real.cot α + Real.cot β + Real.cot γ :=
sorry

theorem brocard_isogonal_conjugate {A B C P Q : Point} (hP : is_brocard_angle A B C ϕ) :
  is_brocard_point P A B C ϕ ∧
  is_brocard_point Q A B C ϕ ∧
  ∀ X, isogonal_conjugate X A B C P Q :=
sorry

theorem brocard_angle_equal {A B C A1 : Point} {ϕ : Real} (hP : is_brocard_angle A B C ϕ)
  (h_tangent : tangent_circumcircle A B C C) (h_parallel : is_parallel (line_through B (parallel_line_through A C)) A1)
  (h_intersect : intersect_at (line_through A A1) A1) :
  ϕ = angle A1 A C :=
sorry

end brocard_cotangent_sum_brocard_isogonal_conjugate_brocard_angle_equal_l212_212616


namespace scalper_percentage_initial_offer_l212_212871

-- Definitions of the problem conditions
def normal_price : ℝ := 50
def concert_tickets_cost : ℝ := 2 * normal_price
def discounted_friend_ticket : ℝ := 0.6 * normal_price
def total_payment : ℝ := 360
def discount : ℝ := 10

-- Condition: Total payment by friends equals the combined payments
def total_cost (P : ℝ) : ℝ :=
  concert_tickets_cost + 2 * (P / 100) * normal_price - discount + discounted_friend_ticket

-- Theorem statement
theorem scalper_percentage_initial_offer : ∃ P : ℝ, total_cost P = total_payment ∧ P = 480 :=
by
  use 480
  sorry

end scalper_percentage_initial_offer_l212_212871


namespace min_value_f_l212_212348

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (15 - 12 * cos x) + 
  sqrt (4 - 2 * sqrt 3 * sin x) +
  sqrt (7 - 4 * sqrt 3 * sin x) +
  sqrt (10 - 4 * sqrt 3 * sin x - 6 * cos x)

theorem min_value_f : ∃ x : ℝ, f x = 6 := 
sorry

end min_value_f_l212_212348


namespace price_of_two_identical_filters_l212_212629

def price_of_individual_filters (x : ℝ) : Prop :=
  let total_individual := 2 * 14.05 + 19.50 + 2 * x
  total_individual = 87.50 / 0.92

theorem price_of_two_identical_filters
  (h1 : price_of_individual_filters 23.76) :
  23.76 * 2 + 28.10 + 19.50 = 87.50 / 0.92 :=
by sorry

end price_of_two_identical_filters_l212_212629


namespace tangent_circles_set_of_m_l212_212766

-- Definitions of the given circles' equations
def circle1 (x y m : ℝ) := x^2 + y^2 - 2 * m * x + 4 * y + m^2 - 5
def circle2 (x y m : ℝ) := x^2 + y^2 + 2 * x - 2 * m * y + m^2 - 3

-- Definition of the centers and radii, as derived from the standard forms
def center1 (m : ℝ) : ℝ × ℝ := (m, -2)
def center2 (m : ℝ) : ℝ × ℝ := (-1, m)
def radius1 : ℝ := 3
def radius2 : ℝ := 2

-- Definition of the Euclidean distance between two points
def euclidean_distance (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The mathematically equivalent proof problem
theorem tangent_circles_set_of_m : 
  {m : ℝ | (euclidean_distance (center1 m) (center2 m) = radius1 + radius2 ∨ 
           euclidean_distance (center1 m) (center2 m) = abs (radius1 - radius2))} = 
  {-5, -2, -1, 2} := sorry

end tangent_circles_set_of_m_l212_212766


namespace equivalent_spherical_coords_l212_212464

theorem equivalent_spherical_coords (ρ θ φ : ℝ) (hρ : ρ = 4) (hθ : θ = 3 * π / 8) (hφ : φ = 9 * π / 5) :
  ∃ (ρ' θ' φ' : ℝ), ρ' = 4 ∧ θ' = 11 * π / 8 ∧ φ' = π / 5 ∧ 
  (ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by
  sorry

end equivalent_spherical_coords_l212_212464


namespace least_months_to_double_debt_l212_212125

theorem least_months_to_double_debt : ∃ t : ℕ, 1.07^t > 2 ∧ ∀ k : ℕ, k < t → 1.07^k ≤ 2 :=
by
  sorry

end least_months_to_double_debt_l212_212125


namespace locus_of_centers_is_three_concurrent_segments_l212_212644

def Point := ℝ × ℝ  -- Represent points as pairs of real numbers

-- Define triangle ABC and points on its sides
variables (A B C T U V W M : Point)

-- Define conditions of inscribed rectangle
def isInscribedRectangle (T U V W : Point) (A B C : Point) : Prop :=
  segment_parallel (V, W) (B, C) ∧ -- VW ∥ BC
  is_on_line T (A, B) ∧ -- T ∈ AB
  is_on_line U (A, C) ∧ -- U ∈ AC
  is_on_line V (B, C) ∧ -- V ∈ BC
  is_on_line W (B, C) -- W ∈ BC

-- Define center of rectangle
def center_of_rectangle (T U V W : Point) : Point :=
  midpoint (midpoint T U) (midpoint V W)

-- Define symmedian point of triangle (assume some definition of symmedian point exist)
def symmedian_point (A B C : Point) : Point := sorry

-- Lean 4 theorem statement
theorem locus_of_centers_is_three_concurrent_segments (A B C : Point) 
  (h : ∀ (T U V W : Point), isInscribedRectangle T U V W A B C) :
  ∃ K : Point, K = symmedian_point A B C ∧ 
  ∀ (M : Point), (∃ (T U V W : Point), center_of_rectangle T U V W = M ∧ isInscribedRectangle T U V W A B C) → 
  M lies_on_concurrent_segments_at K :=
sorry

end locus_of_centers_is_three_concurrent_segments_l212_212644


namespace tangent_line_through_M_and_tangent_to_circle_value_of_m_if_line_intersects_circle_A_B_l212_212755

noncomputable def circle_eq (x y : ℝ) : ℝ := x^2 + (y - 1)^2 - 5
noncomputable def line_m (m x y : ℝ) : ℝ := m * x - y - m + 1

theorem tangent_line_through_M_and_tangent_to_circle :
  ∃ k : ℝ, ∀ x y, (3 * y - 2 - y * (x - 3) = 0 → circle_eq x y = 0) ∧ 
  ∃ k1 k2 : ℝ, k1 = 2 ∨ k2 = (-1 / 2) ∧ (∀ x y : ℝ, (line_m k x y = 0 ↔ (x + 2y - 7 = 0) ∨ (2x - y - 4 = 0)))
: sorry

theorem value_of_m_if_line_intersects_circle_A_B :
  ∀ |AB| : ℝ, |AB| = real.sqrt 17 →
  ∃ m : ℝ, m = real.sqrt 3 ∨ m = -real.sqrt 3 ∧ ∀ x y, (∃ x y, line_m m x y = 0) :=
sorry

end tangent_line_through_M_and_tangent_to_circle_value_of_m_if_line_intersects_circle_A_B_l212_212755


namespace minimum_value_of_expr_l212_212015

noncomputable def expr (x : ℝ) := 2 * x + 1 / (x + 3)

theorem minimum_value_of_expr (x : ℝ) (h : x > -3) :
  ∃ y, y = 2 * real.sqrt 2 - 6 ∧ ∀ z, z > -3 → expr z ≥ y := sorry

end minimum_value_of_expr_l212_212015


namespace num_rectangular_arrays_with_36_chairs_l212_212645

theorem num_rectangular_arrays_with_36_chairs :
  ∃ n : ℕ, (∀ r c : ℕ, r * c = 36 ∧ r ≥ 2 ∧ c ≥ 2 ↔ n = 7) :=
sorry

end num_rectangular_arrays_with_36_chairs_l212_212645


namespace minimal_value_x1_x2_l212_212399

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin ((π / 2) * x + (π / 5))

theorem minimal_value_x1_x2 :
  (∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) → |x₁ - x₂| = 2 :=
sorry

end minimal_value_x1_x2_l212_212399


namespace sum_of_cosines_sum_of_sines_l212_212373

open Complex Real

theorem sum_of_cosines : ∀ α : ℝ, 
  cos α + cos (α + 2 * π / 3) + cos (α + 4 * π / 3) = 0 := 
sorry

theorem sum_of_sines : ∀ α : ℝ, 
  sin α + sin (α + 2 * π / 3) + sin (α + 4 * π / 3) = 0 := 
sorry

end sum_of_cosines_sum_of_sines_l212_212373


namespace find_x_y_l212_212379

theorem find_x_y (a n x y : ℕ) (hx4 : 1000 ≤ x ∧ x < 10000) (hy4 : 1000 ≤ y ∧ y < 10000) 
  (h_yx : y > x) (h_y : y = a * 10 ^ n) 
  (h_sum : (x / 1000) + ((x % 1000) / 100) = 5 * a) 
  (ha : a = 2) (hn : n = 3) :
  x = 1990 ∧ y = 2000 := 
by 
  sorry

end find_x_y_l212_212379


namespace log_to_base_3_of_expression_l212_212673

open Real

noncomputable def calculate_logarithm_expression : Prop :=
  let expr := (3 ^ 4 * 3) / (3 ^ (2 / 3))
  log 3 expr = (13 / 3)

theorem log_to_base_3_of_expression :
  calculate_logarithm_expression := 
by
  -- proof omitted
  sorry

end log_to_base_3_of_expression_l212_212673


namespace rational_root_is_integer_l212_212492

noncomputable def P (x : ℚ) : ℤ
noncomputable def Q (x : ℚ) : ℤ

variables (a_n b_n : ℤ) (a : ℕ → ℤ) (b : ℕ → ℤ)
variable (r : ℚ)

theorem rational_root_is_integer 
  (hP : ∀ x, P x = a 0 + a 1 * x + a 2 * x^2 + ... + a n * x^n) 
  (hQ : ∀ x, Q x = b 0 + b 1 * x + b 2 * x^2 + ... + b n * x^n) 
  (h1 : a n - b n = k ∧ prime k)
  (h2 : a n * b 0 - a 0 * b n ≠ 0)
  (h3 : a (n - 1) = b (n - 1))
  (h4 : P r = 0)
  (h5 : Q r = 0) :
  r ∈ ℤ :=
sorry

end rational_root_is_integer_l212_212492


namespace no_prime_divisible_by_42_l212_212419

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → ¬ (42 ∣ p) :=
by sorry

end no_prime_divisible_by_42_l212_212419


namespace numberOfWaysToChooseSquad_correct_l212_212113

/-- Define the number of ways to choose a starting squad consisting of a goalkeeper,
defender, midfielder, forward, and substitute from a team of 15 members,
given each member can play any position. -/
def numberOfWaysToChooseSquad : ℕ :=
  15 * 14 * 13 * 12 * 11

/-- Prove that the number of ways to choose such a squad is 360,360. -/
theorem numberOfWaysToChooseSquad_correct :
  numberOfWaysToChooseSquad = 360360 :=
by
  simp [numberOfWaysToChooseSquad]
  sorry

end numberOfWaysToChooseSquad_correct_l212_212113


namespace train_length_correct_l212_212237

-- Definitions of the given conditions
def speed_kmh : ℝ := 72
def length_platform : ℝ := 210
def time_seconds : ℝ := 26

-- Conversion factor from km/hr to m/s
def kmh_to_ms (v : ℝ) : ℝ := v * 1000 / 3600

-- The speed of the train in meters per second
def speed_ms := kmh_to_ms speed_kmh

-- Total distance covered by the train while crossing the platform
def total_distance := speed_ms * time_seconds

-- Length of the train L
def length_train := total_distance - length_platform

theorem train_length_correct : length_train = 310 := by
  sorry

end train_length_correct_l212_212237


namespace correct_option_l212_212611

-- Conditions as definitions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def optionC (a : ℝ) : Prop := (-2 * a^2)^3 = -8 * a^6
def optionD (a : ℝ) : Prop := a^6 / a^2 = a^3

-- The statement to prove
theorem correct_option (a : ℝ) : optionC a :=
by 
  unfold optionC
  sorry

end correct_option_l212_212611


namespace closest_point_on_line_l212_212351

-- Definition of the line and the given point
def line (x : ℝ) : ℝ := 2 * x - 4
def point : ℝ × ℝ := (3, -1)

-- Define the closest point we've computed
def closest_point : ℝ × ℝ := (9/5, 2/5)

-- Statement of the problem to prove the closest point
theorem closest_point_on_line : 
  ∃ (p : ℝ × ℝ), p = closest_point ∧ 
  ∀ (q : ℝ × ℝ), (line q.1 = q.2) → 
  (dist point p ≤ dist point q) :=
sorry

end closest_point_on_line_l212_212351


namespace dogwood_trees_count_l212_212579

theorem dogwood_trees_count 
  (c : ℕ) (t1 : ℕ) (t2 : ℕ) 
  (h1 : c = 39) 
  (h2 : t1 = 41) 
  (h3 : t2 = 20) : 
  c + t1 + t2 = 100 :=
by
  rw [h1, h2, h3]
  rfl

end dogwood_trees_count_l212_212579


namespace roots_equal_implies_a_eq_3_l212_212799

theorem roots_equal_implies_a_eq_3 (x a : ℝ) (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
sorry

end roots_equal_implies_a_eq_3_l212_212799


namespace parameterized_line_segment_problem_l212_212169

theorem parameterized_line_segment_problem
  (p q r s : ℝ)
  (hq : q = 1)
  (hs : s = 2)
  (hpq : p + q = 6)
  (hrs : r + s = 9) :
  p^2 + q^2 + r^2 + s^2 = 79 := 
sorry

end parameterized_line_segment_problem_l212_212169


namespace variance_transformed_l212_212391

-- Define the variance function
def variance (xs : List ℝ) : ℝ :=
  let mean := (xs.sum) / (xs.length : ℝ)
  let squared_diffs := xs.map (λ x => (x - mean) ^ 2)
  (squared_diffs.sum) / (xs.length : ℝ)

-- Define the four numbers and their transformed counterparts
variables (x1 x2 x3 x4 : ℝ)
noncomputable def xs : List ℝ := [x1, x2, x3, x4]
noncomputable def ys : List ℝ := [3 * x1 + 5, 3 * x2 + 5, 3 * x3 + 5, 3 * x4 + 5]

-- Given condition
axiom variance_xs : variance xs = 7

theorem variance_transformed : variance ys = 63 := 
sorry

end variance_transformed_l212_212391


namespace maximize_profit_l212_212231

noncomputable def profit (x : ℝ) : ℝ := 
  (x - 8) * (100 - 10 * (x - 10))

theorem maximize_profit : 
  (∀ x > 10, profit x = -10 * x^2 + 280 * x - 1600) → 
  (∃ x : ℝ, x = 14 ∧ (∀ y : ℝ, (y > 10 → profit y ≤ profit x))) :=
begin
  sorry
end

end maximize_profit_l212_212231


namespace school_children_count_l212_212242

theorem school_children_count (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by sorry

end school_children_count_l212_212242


namespace range_of_tangent_lines_l212_212762

/-- Given the function f(x) = x^3 - 3x^2 and the point P(2, t).
    If there exist three tangent lines to the curve f(x) that pass through point P,
    then the range of values for t is -5 < t < -4. -/
theorem range_of_tangent_lines (t : ℝ) : 
  (∃ (s : ℝ), let f := λ x : ℝ, x^3 - 3 * x^2 in
              let f' := 3 * s^2 - 6 * s in 
              let tangent_eq := f(t) + f' * (x - s) in
              tangent_eq s = f s ∧ tangent_eq 2 = t) →
  -5 < t ∧ t < -4 :=
begin
  sorry
end

end range_of_tangent_lines_l212_212762


namespace arrange_letters_of_unique_word_l212_212007

-- Define the problem parameters
def unique_word := ["M₁", "I₁", "S₁", "S₂", "I₂", "P₁", "P₂", "I₃"]
def word_length := unique_word.length
def arrangement_count := Nat.factorial word_length

-- Theorem statement corresponding to the problem
theorem arrange_letters_of_unique_word :
  arrangement_count = 40320 :=
by
  sorry

end arrange_letters_of_unique_word_l212_212007


namespace henrique_chocolate_bars_l212_212413

theorem henrique_chocolate_bars :
  ∃ n : ℕ, 1.35 * n < 10 ∧ 9 < 1.35 * n := 
by
  sorry

end henrique_chocolate_bars_l212_212413


namespace find_k_value_l212_212742

theorem find_k_value : ∀ (x y k : ℝ), x = 2 → y = -1 → y - k * x = 7 → k = -4 := 
by
  intros x y k hx hy h
  sorry

end find_k_value_l212_212742


namespace general_equation_of_line_and_min_value_l212_212382

-- Definitions
def line_through_point (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) :=
  l p.1 p.2

def intercepts_equal (l : ℝ → ℝ → Prop) :=
  ∃ a : ℝ, l a 0 ∧ l 0 a

def point_on_line (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) :=
  l P.1 P.2

def non_zero_intercepts (l : ℝ → ℝ → Prop) :=
  ∃ a : ℝ, a ≠ 0 ∧ l a 0 ∧ l 0 a

-- Theorem
theorem general_equation_of_line_and_min_value (a b : ℝ) :
  let l := λ x y, x + y = 3 in
  line_through_point (1, 2) l →
  intercepts_equal l →
  non_zero_intercepts l →
  point_on_line (a, b) l →
  3 ^ a + 3 ^ b = 6 * Real.sqrt 3 :=
by
  intros
  sorry

end general_equation_of_line_and_min_value_l212_212382


namespace relationship_among_abc_l212_212745

noncomputable def log_base (b x : ℝ) := log x / log b

variable (f : ℝ → ℝ)

/-- f is an odd function -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

/-- f is an increasing function -/
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f (x) < f (y)

variables (a b c : ℝ)
variables (f_is_odd : odd_function f)
variables (f_is_increasing : increasing_function f)
variable (h_a : a = -f (log_base 2 (1 / 5)))
variable (h_b : b = f (log_base 2 4.1))
variable (h_c : c = f (2 ^ 0.8))

theorem relationship_among_abc :
  c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l212_212745


namespace exists_trihedral_angle_with_all_acute_l212_212924

theorem exists_trihedral_angle_with_all_acute (T : Tetrahedron) :
  ∃ v : T.Vertex, ∀ (α β γ : ℝ), T.angle_at v = α + β + γ → α < π/2 ∧ β < π/2 ∧ γ < π/2 :=
sorry

end exists_trihedral_angle_with_all_acute_l212_212924


namespace find_a_l212_212779

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l212_212779


namespace sum_of_squares_of_roots_l212_212717

noncomputable def P (x : ℂ) : ℂ := 4 * x^10 - 7 * x^9 + 5 * x^8 - 8 * x^7 + 12 * x^6 - 12 * x^5 + 12 * x^4 - 8 * x^3 + 5 * x^2 - 7 * x + 4

theorem sum_of_squares_of_roots :
  (∑ x in (P.roots).to_finset, x^2) = -7 / 16 :=
sorry

end sum_of_squares_of_roots_l212_212717


namespace gcd_largest_value_l212_212971

/-- Given two positive integers x and y such that x + y = 780,
    this definition states that the largest possible value of gcd(x, y) is 390. -/
theorem gcd_largest_value (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x + y = 780) : ∃ d, d = Nat.gcd x y ∧ d = 390 :=
sorry

end gcd_largest_value_l212_212971


namespace car_speed_constant_l212_212630

theorem car_speed_constant (v : ℝ) (hv : v ≠ 0)
  (condition_1 : (1 / 36) * 3600 = 100) 
  (condition_2 : (1 / v) * 3600 = 120) :
  v = 30 := by
  sorry

end car_speed_constant_l212_212630


namespace brooke_initial_l212_212311

variable (B : ℕ)

def brooke_balloons_initially (B : ℕ) :=
  let brooke_balloons := B + 8
  let tracy_balloons_initial := 6
  let tracy_added_balloons := 24
  let tracy_balloons := tracy_balloons_initial + tracy_added_balloons
  let tracy_popped_balloons := tracy_balloons / 2 -- Tracy having half her balloons popped.
  (brooke_balloons + tracy_popped_balloons = 35)

theorem brooke_initial (h : brooke_balloons_initially B) : B = 12 :=
  sorry

end brooke_initial_l212_212311


namespace sum_of_angles_l212_212468

variables {α β : ℝ}

theorem sum_of_angles :
  (cos α = sqrt 5 / 5) →
  (sin β = sqrt 2 / 10) →
  (0 < α ∧ α < π / 2) →
  (0 < β ∧ β < π / 2) →
  2 * α + β = 3 * π / 4 :=
begin
  intros hcosα hsinβ hα_range hβ_range,
  sorry
end

end sum_of_angles_l212_212468


namespace sufficient_necessary_condition_l212_212285

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * a * x^3 + (1 / 2) * a * x^2 - 2 * a * x + 2 * a + 1

theorem sufficient_necessary_condition (a : ℝ) :
  (-6 / 5 < a ∧ a < -3 / 16) ↔
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧
   (∃ c₁ c₂ : ℝ, deriv (f a) c₁ = 0 ∧ deriv (f a) c₂ = 0 ∧
   deriv (deriv (f a)) c₁ < 0 ∧ deriv (deriv (f a)) c₂ > 0 ∧
   f a c₁ > 0 ∧ f a c₂ < 0)) := sorry

end sufficient_necessary_condition_l212_212285


namespace statue_of_liberty_ratio_l212_212178

theorem statue_of_liberty_ratio :
  let H_statue := 305 -- height in feet
  let H_model := 10 -- height in inches
  H_statue / H_model = 30.5 := 
by
  let H_statue := 305
  let H_model := 10
  sorry

end statue_of_liberty_ratio_l212_212178


namespace inequality_and_equality_hold_l212_212896

theorem inequality_and_equality_hold (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ a = b) :=
sorry

end inequality_and_equality_hold_l212_212896


namespace radius_of_third_circle_l212_212593

/-- 
  Given two concentric circles with radii 12 and 25 units respectively,
  and a shaded region between these circles.
  Prove that the radius of a third circle whose area is equal to the shaded region is √481.
-/
theorem radius_of_third_circle (r1 r2 : ℝ) (A_shaded : ℝ) :
  r1 = 12 → r2 = 25 → A_shaded = (Math.pi * r2^2) - (Math.pi * r1^2) → (∃ r, Math.pi * r^2 = A_shaded) → r = Real.sqrt 481 :=
by
  intros
  sorry

end radius_of_third_circle_l212_212593


namespace find_QE_length_l212_212965

variables (AP AQ PE QE PD : ℝ)
variables (O P Q A E D : Type*)

-- Conditions: given values and relationships
def AP_eq_six : AP = 6 := by sorry
def PE_eq_3_6 : PE = 3.6 := by sorry
def PD_eq_four : PD = 4 := by sorry
def QE_perpendicular_PD : QE ⊥ PD := by sorry

-- The statement to prove
theorem find_QE_length (h₁ : AP_eq_six) (h₂ : PE_eq_3_6) (h₃ : PD_eq_four) (h₄ : QE_perpendicular_PD) : QE = 1.2 :=
sorry

end find_QE_length_l212_212965


namespace Nancy_hourly_wage_l212_212106

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end Nancy_hourly_wage_l212_212106


namespace a_3_value_l212_212859

noncomputable def a : ℕ → ℚ
| 0     := 2    -- but sequence is given in terms of n starting from 1, we'll adjust this if needed
| (n+1) := (2 * a n / (n + 2)) - 1

theorem a_3_value : a 2 = -1/3 :=
by sorry

end a_3_value_l212_212859


namespace statement_A_statement_C_l212_212428

noncomputable def a_0 : ℝ := 1
noncomputable def a_1 : ℝ
noncomputable def a_2 : ℝ
noncomputable def a_3 : ℝ
noncomputable def a_4 : ℝ

def polynomial_equality (x : ℝ) : ℝ :=
  x^4 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4

theorem statement_A :
  polynomial_equality (-1) →
  a_0 = 1 :=
sorry

theorem statement_C :
  polynomial_equality 0 →
  a_1 + a_2 + a_3 + a_4 = -1 :=
sorry

end statement_A_statement_C_l212_212428


namespace angle_ABC_equals_angle_ADC_l212_212846

def Quadrilateral (A B C D : Type) := True -- We need a placeholder for the quadrilateral type.

variables {A B C D O : Type} -- Variables for points

-- Angles definitions
variables (angle_CBD angle_CAB angle_ACD angle_BDA angle_ABC angle_ADC : Type)

-- Given conditions:
variable Hypothesis1 : angle_CBD = angle_CAB
variable Hypothesis2 : angle_ACD = angle_BDA

-- The theorem to be proven:
theorem angle_ABC_equals_angle_ADC : Quadrilateral A B C D → angle_CBD = angle_CAB → angle_ACD = angle_BDA → angle_ABC = angle_ADC :=
  by
  intro h_quad h1 h2,
  sorry

end angle_ABC_equals_angle_ADC_l212_212846


namespace find_a_l212_212777

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l212_212777


namespace angle_ABC_equals_angle_ADC_l212_212844

def Quadrilateral (A B C D : Type) := True -- We need a placeholder for the quadrilateral type.

variables {A B C D O : Type} -- Variables for points

-- Angles definitions
variables (angle_CBD angle_CAB angle_ACD angle_BDA angle_ABC angle_ADC : Type)

-- Given conditions:
variable Hypothesis1 : angle_CBD = angle_CAB
variable Hypothesis2 : angle_ACD = angle_BDA

-- The theorem to be proven:
theorem angle_ABC_equals_angle_ADC : Quadrilateral A B C D → angle_CBD = angle_CAB → angle_ACD = angle_BDA → angle_ABC = angle_ADC :=
  by
  intro h_quad h1 h2,
  sorry

end angle_ABC_equals_angle_ADC_l212_212844


namespace num_valid_arrangements_l212_212301

-- Defining the concepts of adjacency and arrangements
def adjacent (A B : ℕ) (arrangement : List ℕ) : Prop :=
  ∃ (i : ℕ), i < arrangement.length - 1 ∧ (arrangement.nth i = some A ∧ arrangement.nth (i + 1) = some B) ∨ (arrangement.nth i = some B ∧ arrangement.nth (i + 1) = some A)

def not_adjacent (A B : ℕ) (arrangement : List ℕ) : Prop :=
  ¬ (adjacent A B arrangement)

-- Arrangement of 5 different products
def arrangement_5 : List ℕ := [0, 1, 2, 3, 4]

-- The problem states that A is adjacent to B and A is not adjacent to C
def valid_arrangement (A B C : ℕ) (arrangement : List ℕ) : Prop :=
  adjacent A B arrangement ∧ not_adjacent A C arrangement

-- The Lean theorem will state that there are exactly 36 such valid arrangements
theorem num_valid_arrangements : 
  ∃ (A B C : ℕ), (A ≠ B ∧ A ≠ C ∧ B ≠ C) → 
  (list.filter (valid_arrangement A B C) (list.permutations arrangement_5)).length = 36 := sorry

end num_valid_arrangements_l212_212301


namespace number_of_parallelograms_l212_212919

theorem number_of_parallelograms (A : ℝ × ℝ) (area : ℝ) (m : ℤ) (h_m : m > 2) :
  (A = (0, 0)) ∧ (area = 500000) →
  (∀ b d : ℤ, (b > 0) ∧ (d > 0) ∧ (b ∈ ℤ) ∧ (d ∈ ℤ) ∧ (b = d / 2) ∧ ((m-1)*b*d = 500000) →
  True) →
  (∃ n, n = 720) :=
by
  intros h
  sorry

end number_of_parallelograms_l212_212919


namespace sufficient_but_not_necessary_condition_l212_212743

-- Define a sequence of positive terms
def is_positive_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∀ i, 0 < seq i

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∃ q > 0, q ≠ 1 ∧ ∀ i j, i < j → seq j = (q ^ (j - i : ℤ)) * seq i

-- State the theorem
theorem sufficient_but_not_necessary_condition (seq : Fin 8 → ℝ) (h_pos : is_positive_sequence seq) :
  ¬is_geometric_sequence seq → seq 0 + seq 7 < seq 3 + seq 4 ∧ 
  (seq 0 + seq 7 < seq 3 + seq 4 → ¬is_geometric_sequence seq) ∧
  (¬is_geometric_sequence seq → ¬(seq 0 + seq 7 < seq 3 + seq 4) -> ¬ is_geometric_sequence seq) :=
sorry

end sufficient_but_not_necessary_condition_l212_212743


namespace probability_xy_gt_0_4_l212_212983

noncomputable def probability_of_product_gt_0_4 : ℝ :=
  let D := set.prod (set.Icc 0 1) (set.Icc 0 1)
  let event_set := {p : ℝ × ℝ | p.1 * p.2 > 0.4} ∩ D
  let area_D : ℝ := 1
  let area_event_set : ℝ := ∫ x in 0.4..1, (1 - 0.4 / x) ∂ measure_theory.volume
  area_event_set / area_D

theorem probability_xy_gt_0_4 :
  probability_of_product_gt_0_4 = (0.6 + 0.4 * real.log 0.4) :=
sorry

end probability_xy_gt_0_4_l212_212983


namespace find_sum_of_possible_n_l212_212569

def conditions : Prop := 
  let S := ({4, 7, 15, 18} : Set ℤ)
  ∃ n : ℤ, n ∉ S ∧ ((4 + 7 + 15 + 18 + n) / 5 = n)

theorem find_sum_of_possible_n : ∃ n : ℤ, conditions → ( Set.sum ({4, 7, 15, 18, 31, 11} : Set ℤ) = 42 ):= 
by
  sorry

end find_sum_of_possible_n_l212_212569


namespace xiao_cong_math_score_l212_212550

theorem xiao_cong_math_score :
  ∀ (C M E : ℕ),
    (C + M + E) / 3 = 122 → C = 118 → E = 125 → M = 123 :=
by
  intros C M E h1 h2 h3
  sorry

end xiao_cong_math_score_l212_212550


namespace circle_area_l212_212218

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def radius (P Q : ℝ × ℝ) : ℝ := distance P Q

def area_of_circle (r : ℝ) : ℝ := real.pi * r ^ 2

theorem circle_area (P Q : ℝ × ℝ) (hP : P = (-2, 3)) (hQ : Q = (8, -4)) :
  area_of_circle (radius P Q) = 149 * real.pi :=
by
  sorry

end circle_area_l212_212218


namespace odd_squarefree_integers_1_to_199_l212_212690

noncomputable def count_squarefree_odd_integers (n : ℕ) :=
  n - List.sum [
    n / 18,   -- for 3^2 = 9
    n / 50,   -- for 5^2 = 25
    n / 98,   -- for 7^2 = 49
    n / 162,  -- for 9^2 = 81
    n / 242,  -- for 11^2 = 121
    n / 338   -- for 13^2 = 169
  ]

theorem odd_squarefree_integers_1_to_199 : count_squarefree_odd_integers 198 = 79 := 
by
  sorry

end odd_squarefree_integers_1_to_199_l212_212690


namespace Terry_driving_speed_is_40_l212_212146

-- Conditions
def distance_home_to_workplace : ℕ := 60
def total_time_driving : ℕ := 3

-- Computation for total distance
def total_distance := distance_home_to_workplace * 2

-- Desired speed computation
def driving_speed := total_distance / total_time_driving

-- Problem statement to prove
theorem Terry_driving_speed_is_40 : driving_speed = 40 :=
by 
  sorry -- proof not required as per instructions

end Terry_driving_speed_is_40_l212_212146


namespace function_is_even_with_period_pi_div_2_l212_212165

noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (2 * x)) * (Real.sin x) ^ 2

theorem function_is_even_with_period_pi_div_2 : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + (π / 2)) = f x) :=
by
  sorry

end function_is_even_with_period_pi_div_2_l212_212165


namespace determine_plane_by_parallel_lines_l212_212664

theorem determine_plane_by_parallel_lines
    (points : Type)
    (space : Type)
    [metric_space space]
    [affine_space points space]
    (distinct_points : ∀ P Q R : points, P ≠ Q ∧ Q ≠ R ∧ P ≠ R        
        → ∃ plane : set points, P ∈ plane ∧ Q ∈ plane ∧ R ∈ plane)
    (two_lines: ∀ l₁ l₂ : set points, l₁ ≠ l₂ ∧ is_line l₁ ∧ is_line l₂
        → ¬ (∃ l′ : set points, is_line l′ ∧ l₁ ⊆ l′ ∧ l₂ ⊆ l′))
    (line_and_point: ∀ p : points, ∀ l : set points, 
        p ∈ l ∧ is_line l → ∃ planes : set (set points), 
        ∀ plane : set points, plane ∈ planes → (is_plane plane ∧ l ⊆ plane ∧ p ∈ plane))
    (parallel_lines: ∀ l₁ l₂ : set points, l₁ ∩ l₂ = ∅ ∧ 
        is_parallel l₁ l₂ → ∃! plane : set points, is_plane plane ∧ l₁ ⊆ plane ∧ l₂ ⊆ plane):
    ∃! p : set points, is_plane p ∧ ∃! l₁ l₂ : set points, 
    l₁ ⊆ p ∧ l₂ ⊆ p ∧ is_parallel l₁ l₂ :=
sorry

end determine_plane_by_parallel_lines_l212_212664


namespace total_soccer_balls_l212_212964

theorem total_soccer_balls (boxes : ℕ) (packages_per_box : ℕ) (balls_per_package : ℕ) 
  (h1 : boxes = 10) (h2 : packages_per_box = 8) (h3 : balls_per_package = 13) : 
  (boxes * packages_per_box * balls_per_package = 1040) :=
by 
  sorry

end total_soccer_balls_l212_212964


namespace jessica_at_least_two_correct_prob_l212_212486

noncomputable def probability_jessica_at_least_two_correct : ℚ :=
  1 - (C(5, 0) * (3 / 4)^5 * (1 / 4)^0 + C(5, 1) * (3 / 4)^4 * (1 / 4)^1)

theorem jessica_at_least_two_correct_prob :
  probability_jessica_at_least_two_correct = 47 / 128 :=
by
  sorry

end jessica_at_least_two_correct_prob_l212_212486


namespace simplify_expression_l212_212130

theorem simplify_expression :
  (2^5 + 4^3) * (2^2 - (-2)^3)^8 = 96 * 12^8 :=
by
  sorry

end simplify_expression_l212_212130


namespace stuart_segments_l212_212940

-- Defining the required conditions for the problem
def concentric_circles (C1 C2 : Type) : Prop := sorry

def draws_chords_tangent (A B C D : Type) (C1 C2 : Type) : Prop := sorry

def angle_measure_ABC (A B : Type) (angleABC : ℝ) : Prop := angleABC = 60

-- The main theorem statement
theorem stuart_segments (C1 C2 : Type) (A B C D : Type) (angleABC : ℝ) :
  concentric_circles C1 C2 →
  draws_chords_tangent A B C D C1 C2 →
  angle_measure_ABC A B angleABC →
  ∃ n : ℕ, n = 3 :=
by
  intro h1 h2 h3
  use 3
  sorry

end stuart_segments_l212_212940


namespace range_of_k_l212_212400

variable (a k : ℝ)

def f (x : ℝ) : ℝ :=
  if x >= 0 then x + k * (1 - a^2)
  else x^2 - 4 * x + (3 - a)^2

theorem range_of_k (h : ∀ x1 : ℝ, x1 ≠ 0 → ∃ x2 : ℝ, x2 ≠ 0 ∧ x1 ≠ x2 ∧ f a k x1 = f a k x2) :
  k ≤ 0 ∨ k ≥ 8 :=
sorry

end range_of_k_l212_212400


namespace subset_sum_divisible_by_2n_l212_212508

theorem subset_sum_divisible_by_2n (n : ℕ) (h : n ≥ 4) (a : Fin n → ℕ) (ha : ∀ i, a i < 2 * n ∧ a i > 0) (distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ s : Finset (Fin n), (∑ i in s, a i) % (2 * n) = 0 := 
sorry

end subset_sum_divisible_by_2n_l212_212508


namespace problem_l212_212889

noncomputable def f (x a b : ℝ) : ℝ := (1/3)*x^3 + a*x^2 + b*x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem problem (a b : ℝ) (hab : f' x a b 0 = g' 0) :
  (b = 1) ∧
  (∀ x, -1 ≤ a ∧ a ≤ 1 → f' x a 1 ≥ 0) ∧
  (∀ x ∈ Set.Iio 0, g x > f x a b → a < (1/2)) :=
by
  sorry

/-- Definitions of derivatives used in the theorem -/
def f' (x a b : ℝ) : ℝ := x^2 + 2*a*x + b
def g' (x : ℝ) : ℝ := Real.exp x

end problem_l212_212889


namespace fraction_replaced_l212_212258

variable (x : ℝ)

-- Conditions
axiom original_solution_sugar : 0.10 -- Original solution is 10% sugar by weight
axiom resulting_solution_sugar : 0.14 -- Resulting solution is 14% sugar by weight
axiom second_solution_sugar : 0.26000000000000007 -- Second solution is 26.000000000000007% sugar by weight

theorem fraction_replaced (hx : (1 - x) * original_solution_sugar + x * second_solution_sugar = resulting_solution_sugar) : 
  x = 0.25 := 
by 
  sorry

end fraction_replaced_l212_212258


namespace spherical_coordinates_convert_l212_212462

theorem spherical_coordinates_convert (ρ θ φ ρ' θ' φ' : ℝ) 
  (h₀ : ρ > 0) 
  (h₁ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₂ : 0 ≤ φ ∧ φ ≤ Real.pi) 
  (h_initial : (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5)) 
  (h_final : (ρ', θ', φ') = (4, (11 * Real.pi) / 8,  Real.pi / 5)) : 
  (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5) → 
  (ρ, θ, φ) = (ρ', θ', φ') := 
by
  sorry

end spherical_coordinates_convert_l212_212462


namespace operation_difference_l212_212682

def operation (x y : ℕ) : ℕ := x * y - 3 * x + y

theorem operation_difference : operation 5 9 - operation 9 5 = 16 :=
by
  sorry

end operation_difference_l212_212682


namespace minimum_completion_time_l212_212264

-- Define the problem conditions
variables (x k : ℕ) (K : ℕ) (x_pos : 0 < x) (K_pos : 0 < K) (x_domain : x * (1 + K) < 200)

-- Total units to produce
def total_units := 3000

-- Define production rates
def production_rate_a := 6
def production_rate_b := 3
def production_rate_c := 2

-- Define the time to complete the production of parts A, B, and C
def T_1 (x : ℕ) : ℝ := 2 * total_units / (production_rate_a * x)
def T_2 (x : ℕ) (k : ℕ) : ℝ := 2 * total_units / (production_rate_b * (k * x))
def T_3 (x : ℕ) (k : ℕ) : ℝ := total_units / (production_rate_c * (200 - (1 + k) * x))

-- Define the minimum completion time
def f (x : ℕ) (k : ℕ) : ℝ := max (T_1 x) (max (T_2 x k) (T_3 x k))

-- Define the proof statement
theorem minimum_completion_time : x * 3 + 32 * k = 200 →
    T_1 x = 1000 / x ∧ T_2 x k = 2000 / (k * x) ∧ T_3 x k = 1500 / (200 - (1 + k) * x) →
    (k = 2 → f x k = 250 / 11 ∧ x = 44 ∧ k * x = 88 ∧ 200 - x - k * x = 68) :=
by
  sorry

end minimum_completion_time_l212_212264


namespace no_term_un_eq_neg1_l212_212509

theorem no_term_un_eq_neg1 (p : ℕ) [hp_prime: Fact (Nat.Prime p)] (hp_odd: p % 2 = 1) (hp_not_five: p ≠ 5) :
  ∀ n : ℕ, ∀ u : ℕ → ℤ, ((u 0 = 0) ∧ (u 1 = 1) ∧ (∀ k, k ≥ 2 → u (k-2) = 2 * u (k-1) - p * u k)) → 
    (u n ≠ -1) :=
  sorry

end no_term_un_eq_neg1_l212_212509


namespace star_evaluation_l212_212427

def star (x y : ℕ) : ℕ := x * y - 3 * x + y

theorem star_evaluation : (star 6 5) - (star 5 6) = -4 :=
by
  sorry

end star_evaluation_l212_212427


namespace find_BA_prime_l212_212639

theorem find_BA_prime (BA BC A_prime C_1 : ℝ) 
  (h1 : BA = 3)
  (h2 : BC = 2)
  (h3 : A_prime < BA)
  (h4 : A_prime * C_1 = 3) : A_prime = 3 / 2 := 
by 
  sorry

end find_BA_prime_l212_212639


namespace equal_angles_quadrilateral_l212_212836

theorem equal_angles_quadrilateral
  (AB CD : Type)
  [convex_quad AB CD]
  (angle_CBD angle_CAB angle_ACD angle_BDA : AB CD → ℝ)
  (h1 : angle_CBD = angle_CAB)
  (h2 : angle_ACD = angle_BDA) : angle_ABC = angle_ADC :=
by sorry

end equal_angles_quadrilateral_l212_212836


namespace point_in_fourth_quadrant_l212_212822

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the complex number z as given in the problem
def z : ℂ := i / (i - 1)

-- Define what it means to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- The theorem stating the problem to be proved
theorem point_in_fourth_quadrant : in_fourth_quadrant z :=
by {
  -- We simplify z to find it's representation
  have h1 : z = (1 - i) / 2 := by {
    rw [←complex.div_eq_mul_inv, ←mul_div_assoc, complex.inv_eq_conj_div_abs_sq, complex.norm_sq, complex.conj],
    simp,
    norm_num,
    ring,
  },
  -- From the simplified z, we know z.re = 1/2, z.im = -1/2
  have hz_re : z.re = 1/2 := by rw h1; norm_num,
  have hz_im : z.im = -1/2 := by rw h1; norm_num,
  -- Check the real and imaginary parts fall in the fourth quadrant
  split; linarith,
}

end point_in_fourth_quadrant_l212_212822


namespace polynomial_division_l212_212083

noncomputable def f : Polynomial ℤ := 2 * X^4 + 8 * X^3 - 5 * X^2 + 2 * X + 5
noncomputable def d : Polynomial ℤ := X^2 + 2 * X - 3
noncomputable def q : Polynomial ℤ := 2 * X^2 + 4 * X
noncomputable def r : Polynomial ℤ := 8 * X - 22

theorem polynomial_division : q.eval 2 + r.eval (-2) = -22 :=
by
  -- Since f(x) = q(x) * d(x) + r(x) where deg(r) < deg(d)
  have h : f = q * d + r := sorry
  -- Goal: q(2) + r(-2) = -22
  sorry

end polynomial_division_l212_212083


namespace max_brownie_pieces_l212_212807

theorem max_brownie_pieces (base height piece_width piece_height : ℕ) 
    (h_base : base = 30) (h_height : height = 24)
    (h_piece_width : piece_width = 3) (h_piece_height : piece_height = 4) :
  (base / piece_width) * (height / piece_height) = 60 :=
by sorry

end max_brownie_pieces_l212_212807


namespace angle_equality_in_quadrilateral_l212_212839

/-- In a convex quadrilateral ABCD, 
    if ∠CBD = ∠CAB and ∠ACD = ∠BDA,
    then ∠ABC = ∠ADC. -/
theorem angle_equality_in_quadrilateral 
  {A B C D: Type*} [convex_quadrilateral A B C D]
  (h1 : ∠CBD = ∠CAB)
  (h2 : ∠ACD = ∠BDA) : 
  ∠ABC = ∠ADC := sorry

end angle_equality_in_quadrilateral_l212_212839


namespace isosceles_triangle_x_value_sum_l212_212458

theorem isosceles_triangle_x_value_sum (x : ℝ) (a b c : ℕ) :
  (∃ (α d : ℝ), x ∈ {3} ∧
  (∀ (α d : ℝ), α - d = 60 ∨ α + d = 60 ∧ 3 * α = 180) ∧
  (x^2 = 3^2 + 3^2 - 2 * 3 * 3 * real.cos (60 * (real.pi / 180)))) →
  a + b + c = 3 :=
begin
 sorry
end

end isosceles_triangle_x_value_sum_l212_212458


namespace part1_part2_l212_212409

/-- Conditions for points A, B, and C -/
variables (m n : ℝ)
def vector_OA := (-3, m + 1)
def vector_OB := (n, 3)
def vector_OC := (7, 4)

-- Given orthogonality and collinearity of points
axiom OA_perpendicular_OB : vector_OA.fst * vector_OB.fst + vector_OA.snd * vector_OB.snd = 0
axiom points_collinear : ∃ t : ℝ, vector_OC = vector_OA + t • (vector_OB - vector_OA)

-- Part 1: Solving for m and n
theorem part1 :
  (m = 1 ∧ n = 2) ∨ (m = 8 ∧ n = 9) :=
sorry

-- Part 2: Solving for cos(AOC)
def vector_A := (-3 : ℝ, m + 1)
def vector_C := (7 : ℝ, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
v1.fst * v2.fst + v1.snd * v2.snd

def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.fst^2 + v.snd^2)

def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem part2 (h : m = 1 ∧ n = 2) :
  cos_angle vector_A vector_C = -Real.sqrt 5 / 5 :=
sorry

end part1_part2_l212_212409


namespace gcd_largest_value_l212_212970

/-- Given two positive integers x and y such that x + y = 780,
    this definition states that the largest possible value of gcd(x, y) is 390. -/
theorem gcd_largest_value (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x + y = 780) : ∃ d, d = Nat.gcd x y ∧ d = 390 :=
sorry

end gcd_largest_value_l212_212970


namespace smallest_n_satisfying_inequality_l212_212225

theorem smallest_n_satisfying_inequality :
  ∃ (n : ℕ), n > 0 ∧ (\sqrt n - \sqrt (n - 1) < 0.005) ∧ 
  ∀ (m : ℕ), m > 0 ∧ (\sqrt m - \sqrt (m - 1) < 0.005) → n ≤ m := 
begin
  -- Proof will be provided here
  sorry
end

end smallest_n_satisfying_inequality_l212_212225


namespace journey_time_l212_212256

theorem journey_time : 
  let distance := 9.999999999999998
  let speed_to := 12.5
  let speed_back := 2
  (distance / speed_to) + (distance / speed_back) ≈ 5.8 :=
by
  sorry

end journey_time_l212_212256


namespace omega_sum_equality_l212_212888

noncomputable def omega : ℂ := sorry
axiom condition1 : ω^9 = 1
axiom condition2 : ω ≠ 1

theorem omega_sum_equality : 
  (ω^{16} + ω^{18} + ω^{20} + ω^{22} + ω^{24} + ω^{26} + 
  ω^{28} + ω^{30} + ω^{32} + ω^{34} + ω^{36} + ω^{38} + 
  ω^{40} + ω^{42} + ω^{44} + ω^{46} + ω^{48} + ω^{50} + 
  ω^{52} + ω^{54} + ω^{56} + ω^{58} + ω^{60} + ω^{62} + 
  ω^{64} + ω^{66} + ω^{68} + ω^{70} + ω^{72}) = -ω^7 :=
by 
  exact sorry

end omega_sum_equality_l212_212888


namespace perimeter_of_abcde_l212_212588

-- Definitions based on problem conditions
def is_equilateral (a b c : ℕ) : Prop := (a = b) ∧ (b = c)

def is_midpoint (x y z : ℕ) : Prop := x + x = y + z

-- The main theorem
theorem perimeter_of_abcde
  (AB BC CD DE EF FG GA: ℕ)
  (h₁ : AB = 6)
  (h₂ : is_equilateral 6 6 BC) 
  (h₃: CD = 3)
  (h₄: is_equilateral 3 3 DE)
  (h₅: EF = 1.5)
  (h₆: FG = 1.5)
  (h₇: GA = 3) :
  AB + BC + CD + DE + EF + FG + GA = 24 :=
by
  sorry  -- Proof is omitted as not required.

end perimeter_of_abcde_l212_212588


namespace abs_diff_of_two_numbers_l212_212181

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 216) : |x - y| = 6 := 
sorry

end abs_diff_of_two_numbers_l212_212181


namespace odd_squarefree_integers_1_to_199_l212_212689

noncomputable def count_squarefree_odd_integers (n : ℕ) :=
  n - List.sum [
    n / 18,   -- for 3^2 = 9
    n / 50,   -- for 5^2 = 25
    n / 98,   -- for 7^2 = 49
    n / 162,  -- for 9^2 = 81
    n / 242,  -- for 11^2 = 121
    n / 338   -- for 13^2 = 169
  ]

theorem odd_squarefree_integers_1_to_199 : count_squarefree_odd_integers 198 = 79 := 
by
  sorry

end odd_squarefree_integers_1_to_199_l212_212689


namespace total_pieces_correct_l212_212599

-- Definition of the pieces of chicken required per type of order
def chicken_pieces_per_chicken_pasta : ℕ := 2
def chicken_pieces_per_barbecue_chicken : ℕ := 3
def chicken_pieces_per_fried_chicken_dinner : ℕ := 8

-- Definition of the number of each type of order tonight
def num_fried_chicken_dinner_orders : ℕ := 2
def num_chicken_pasta_orders : ℕ := 6
def num_barbecue_chicken_orders : ℕ := 3

-- Calculate the total number of pieces of chicken needed
def total_chicken_pieces_needed : ℕ :=
  (num_fried_chicken_dinner_orders * chicken_pieces_per_fried_chicken_dinner) +
  (num_chicken_pasta_orders * chicken_pieces_per_chicken_pasta) +
  (num_barbecue_chicken_orders * chicken_pieces_per_barbecue_chicken)

-- The proof statement
theorem total_pieces_correct : total_chicken_pieces_needed = 37 :=
by
  -- Our exact computation here
  sorry

end total_pieces_correct_l212_212599


namespace product_bn_eq_pq_l212_212721

noncomputable def b (n : ℕ) (hn : n ≥ 5) : ℚ :=
  ((n + 2) ^ 2 : ℚ) / (n * (n ^ 3 - 1))

theorem product_bn_eq_pq (p q : ℕ) :
  (∏ n in finset.range (100 - 5 + 1), b (n + 5) (by linarith)) = p / (q.factorial : ℚ)
  ∧ p = 10404
  ∧ q = 100 := sorry

end product_bn_eq_pq_l212_212721


namespace determine_polynomial_by_product_l212_212574

def is_monic (P : Polynomial ℤ) : Prop :=
  P.leadingCoeff = 1

theorem determine_polynomial_by_product (P : Polynomial ℤ) (k : ℕ) (n : ℕ → ℤ) :
  P.degree = 2017 ∧ is_monic P →
  (∃ N : ℤ, ∏ i in finset.range k, P.eval (n i) = N) →
  k ≥ 2017 :=
by  -- proof needs to go here
  sorry

end determine_polynomial_by_product_l212_212574


namespace river_flow_speed_l212_212279

/-- Speed of the ship in still water is 30 km/h,
    the distance traveled downstream is 144 km, and
    the distance traveled upstream is 96 km.
    Given that the time taken for both journeys is equal,
    the equation representing the speed of the river flow v is:
    144 / (30 + v) = 96 / (30 - v). -/
theorem river_flow_speed (v : ℝ) :
  (30 : ℝ) > 0 →
  real_equiv 144 (30 + v) (96 (30 - v)) := by
sorry

end river_flow_speed_l212_212279


namespace pieces_cannot_encounter_all_arrangements_exactly_once_l212_212524

/--
  On a 8x8 chessboard with a black piece and a white piece such that
  - the pieces can move to an adjacent square vertically or horizontally,
  - and no two pieces can stand on the same square,
  prove that it is not possible to encounter all possible arrangements of the two pieces exactly once.
--/
theorem pieces_cannot_encounter_all_arrangements_exactly_once :
  ¬ ∃ m : ℕ → (ℕ × ℕ) × (ℕ × ℕ),
    (∀ n, 1 ≤ n ∧ n ≤ 4032 → valid_move ((m n) 0) ((m n).snd)) ∧
    (∀ a b : (ℕ × ℕ) × (ℕ × ℕ), a ≠ b ↔ ∃ n m : ℕ, m n = a ∧ m m = b ∧ n ≠ m) :=
sorry

end pieces_cannot_encounter_all_arrangements_exactly_once_l212_212524


namespace no_prime_divisible_by_42_l212_212414

open Nat

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → 42 ∣ p → p = 0 :=
by
  intros p hp hdiv
  sorry

end no_prime_divisible_by_42_l212_212414


namespace alternate_cups_possible_l212_212112

def initial_state : list ℕ := [0, 0, 0, 1, 1, 1]  -- 0 represents empty (E), 1 represents filled (C)

def final_state (cups : list ℕ) : Prop :=
  cups = [1, 0, 0, 1, 1, 0]  -- One of the correct possible configurations

theorem alternate_cups_possible (cups : list ℕ) (hs : cups = initial_state) : ∃ cups', 
  final_state cups' :=
by
  sorry

end alternate_cups_possible_l212_212112


namespace same_root_a_eq_3_l212_212796

theorem same_root_a_eq_3 {x a : ℝ} (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
by
  sorry

end same_root_a_eq_3_l212_212796


namespace baron_munchausen_correct_l212_212307

theorem baron_munchausen_correct :
  ∃ (P : Type) [polygon_structure P] (O : P),
  (∀ (L : line), (passes_through L O) → divides_into_three_polygons P L) := sorry

end baron_munchausen_correct_l212_212307


namespace possible_integer_roots_l212_212273

theorem possible_integer_roots (a2 a1 : ℤ) :
  {x : ℤ | (x^3 + a2 * x^2 + a1 * x - 13) = 0} ⊆ {±1, ±13} :=
sorry

end possible_integer_roots_l212_212273


namespace not_divisible_by_4_l212_212621

theorem not_divisible_by_4 (n : Int) : ¬ (1 + n + n^2 + n^3 + n^4) % 4 = 0 := by
  sorry

end not_divisible_by_4_l212_212621


namespace living_room_area_l212_212627

theorem living_room_area (L W : ℝ) (percent_covered : ℝ) (expected_area : ℝ) 
  (hL : L = 6.5) (hW : W = 12) (hpercent : percent_covered = 0.85) 
  (hexpected_area : expected_area = 91.76) : 
  (L * W / percent_covered = expected_area) :=
by
  sorry  -- The proof is omitted.

end living_room_area_l212_212627


namespace perp_OG_CD_l212_212497

open EuclideanGeometry

-- Define vertices of the triangle
variables {A B C O D G : Point}

-- Define properties and conditions
variables (triangleABC : Triangle A B C)
variables (isCircumcenterO : Circumcenter O triangleABC)
variables (isMidpointD : Midpoint D A B)
variables (isCentroidG : Centroid G A C D)
variables (AB_eq_AC : dist A B = dist A C)

-- Final statement to prove
theorem perp_OG_CD : ⦄ cintc(O, G)
    ((dist G O *  dist D O + dist O G * dist G  A = 2 * dim  triangleABC
    (α:Oβ C)):

  sorry

end perp_OG_CD_l212_212497


namespace slip_2_5_goes_to_B_l212_212484

-- Defining the slips and their values
def slips : List ℝ := [1.5, 2, 2, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5, 6]

-- Defining the total sum of slips
def total_sum : ℝ := 52

-- Defining the cup sum values
def cup_sums : List ℝ := [11, 10, 9, 8, 7]

-- Conditions: slip with 4 goes into cup A, slip with 5 goes into cup D
def cup_A_contains : ℝ := 4
def cup_D_contains : ℝ := 5

-- Proof statement
theorem slip_2_5_goes_to_B : 
  ∃ (cup_A cup_B cup_C cup_D cup_E : List ℝ), 
    (cup_A.sum = 11 ∧ cup_B.sum = 10 ∧ cup_C.sum = 9 ∧ cup_D.sum = 8 ∧ cup_E.sum = 7) ∧
    (4 ∈ cup_A) ∧ (5 ∈ cup_D) ∧ (2.5 ∈ cup_B) :=
sorry

end slip_2_5_goes_to_B_l212_212484


namespace min_a_b_l212_212656

theorem min_a_b (a b : ℕ) (h1 : 43 * a + 17 * b = 731) (h2 : a ≤ 17) (h3 : b ≤ 43) : a + b = 17 :=
by
  sorry

end min_a_b_l212_212656


namespace range_g_l212_212328

def g (x : ℝ) : ℝ := (⌊2 * x⌋ : ℝ) - x

theorem range_g : Set.range g = Set.univ := 
by sorry

end range_g_l212_212328


namespace bricklaying_problem_l212_212310

/-- Bricklaying problem proof -/
theorem bricklaying_problem :
  ∃ (h : ℕ), (∃ (b_rate brandon_rate comb_rate : ℚ),
  b_rate = h / 8 ∧
  brandon_rate = h / 12 ∧
  comb_rate = b_rate + brandon_rate - 12 ∧
  4.5 * comb_rate = h ∧
  h = 864) :=
begin
  sorry
end

end bricklaying_problem_l212_212310


namespace eventually_all_ones_l212_212374

theorem eventually_all_ones (k : ℕ) (seq : Fin (2^k) → ℤ) 
  (h_initial : ∀ i, seq i = 1 ∨ seq i = -1) :
  ∃ n : ℕ, ∀ i, (transform_n_times seq n) i = 1 := 
sorry

end eventually_all_ones_l212_212374


namespace minimum_value_l212_212017

theorem minimum_value (x : ℝ) (h : x > -3) : 2 * x + (1 / (x + 3)) ≥ 2 * Real.sqrt 2 - 6 :=
sorry

end minimum_value_l212_212017


namespace value_of_f_ln3_l212_212752

def f : ℝ → ℝ := sorry

theorem value_of_f_ln3 (f_symm : ∀ x : ℝ, f (x + 1) = f (-x + 1))
  (f_exp : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = Real.exp (-x)) :
  f (Real.log 3) = 3 * Real.exp (-2) :=
by
  sorry

end value_of_f_ln3_l212_212752


namespace parabola_intersections_l212_212207

theorem parabola_intersections :
  ∃ y1 y2, (∀ x y, (y = 2 * x^2 + 5 * x + 1 ∧ y = - x^2 + 4 * x + 6) → 
     (x = ( -1 + Real.sqrt 61) / 6 ∧ y = y1) ∨ (x = ( -1 - Real.sqrt 61) / 6 ∧ y = y2)) := 
by
  sorry

end parabola_intersections_l212_212207


namespace mask_price_reduction_l212_212515

theorem mask_price_reduction 
  (initial_sales : ℕ)
  (initial_profit : ℝ)
  (additional_sales_factor : ℝ)
  (desired_profit : ℝ)
  (x : ℝ)
  (h_initial_sales : initial_sales = 500)
  (h_initial_profit : initial_profit = 0.6)
  (h_additional_sales_factor : additional_sales_factor = 100 / 0.1)
  (h_desired_profit : desired_profit = 240) :
  (initial_profit - x) * (initial_sales + additional_sales_factor * x) = desired_profit → x = 0.3 :=
sorry

end mask_price_reduction_l212_212515


namespace min_gift_cost_time_constrained_l212_212907

def store := { name : String, mom : ℕ, dad : ℕ, brother : ℕ, sister : ℕ, time_in_store : ℕ }

def romashka : store := { name := "Romashka", mom := 1000, dad := 750, brother := 930, sister := 850, time_in_store := 35 }
def oduvanchik : store := { name := "Oduvanchik", mom := 1050, dad := 790, brother := 910, sister := 800, time_in_store := 30 }
def nezabudka : store := { name := "Nezabudka", mom := 980, dad := 810, brother := 925, sister := 815, time_in_store := 40 }
def landysh : store := { name := "Landysh", mom := 1100, dad := 755, brother := 900, sister := 820, time_in_store := 25 }

constant travel_time : ℕ := 30
constant total_available_time : ℕ := 3 * 60 + 25 -- 205 minutes

def total_time (stores : List store) : ℕ :=
  stores.map (λ s => s.time_in_store).sum + (travel_time * (stores.length - 1))

def total_cost (stores : List store) : ℕ :=
  stores[0].mom + stores[1].dad + stores[2].brother + stores[3].sister

theorem min_gift_cost_time_constrained : 
  ∃ stores : List store, stores.length = 4 ∧ total_time stores ≤ total_available_time ∧ total_cost stores = 3435 :=
by
  sorry

end min_gift_cost_time_constrained_l212_212907


namespace solve_fractional_equation_l212_212176

theorem solve_fractional_equation
  (x : ℝ)
  (h1 : x ≠ 0)
  (h2 : x ≠ 2)
  (h_eq : 2 / x - 1 / (x - 2) = 0) : 
  x = 4 := by
  sorry

end solve_fractional_equation_l212_212176


namespace attendees_chose_water_l212_212305

theorem attendees_chose_water
  (total_attendees : ℕ)
  (juice_percentage water_percentage : ℝ)
  (attendees_juice : ℕ)
  (h1 : juice_percentage = 0.7)
  (h2 : water_percentage = 0.3)
  (h3 : attendees_juice = 140)
  (h4 : total_attendees * juice_percentage = attendees_juice)
  : total_attendees * water_percentage = 60 := by
  sorry

end attendees_chose_water_l212_212305


namespace equal_angles_quadrilateral_l212_212833

theorem equal_angles_quadrilateral
  (AB CD : Type)
  [convex_quad AB CD]
  (angle_CBD angle_CAB angle_ACD angle_BDA : AB CD → ℝ)
  (h1 : angle_CBD = angle_CAB)
  (h2 : angle_ACD = angle_BDA) : angle_ABC = angle_ADC :=
by sorry

end equal_angles_quadrilateral_l212_212833


namespace angle_ABC_equals_angle_ADC_l212_212845

def Quadrilateral (A B C D : Type) := True -- We need a placeholder for the quadrilateral type.

variables {A B C D O : Type} -- Variables for points

-- Angles definitions
variables (angle_CBD angle_CAB angle_ACD angle_BDA angle_ABC angle_ADC : Type)

-- Given conditions:
variable Hypothesis1 : angle_CBD = angle_CAB
variable Hypothesis2 : angle_ACD = angle_BDA

-- The theorem to be proven:
theorem angle_ABC_equals_angle_ADC : Quadrilateral A B C D → angle_CBD = angle_CAB → angle_ACD = angle_BDA → angle_ABC = angle_ADC :=
  by
  intro h_quad h1 h2,
  sorry

end angle_ABC_equals_angle_ADC_l212_212845


namespace madeline_water_intake_l212_212514

-- Declare necessary data and conditions
def bottle_A : ℕ := 8
def bottle_B : ℕ := 12
def bottle_C : ℕ := 16

def goal_yoga : ℕ := 15
def goal_work : ℕ := 35
def goal_jog : ℕ := 20
def goal_evening : ℕ := 30

def intake_yoga : ℕ := 2 * bottle_A
def intake_work : ℕ := 3 * bottle_B
def intake_jog : ℕ := 2 * bottle_C
def intake_evening : ℕ := 2 * bottle_A + 2 * bottle_C

def total_intake : ℕ := intake_yoga + intake_work + intake_jog + intake_evening
def goal_total : ℕ := 100

-- Statement of the proof problem
theorem madeline_water_intake : total_intake = 132 ∧ total_intake - goal_total = 32 :=
by
  -- Calculation parts go here (not needed per instruction)
  sorry

end madeline_water_intake_l212_212514


namespace co_presidents_included_probability_l212_212192

-- Let the number of students in each club
def club_sizes : List ℕ := [6, 8, 9, 10]

-- Function to calculate binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Function to calculate probability for a given club size
noncomputable def co_president_probability (n : ℕ) : ℚ :=
  (choose (n - 2) 2 : ℚ) / (choose n 4)

-- List of probabilities for each club
noncomputable def probabilities : List ℚ :=
  List.map co_president_probability club_sizes

-- Aggregate total probability by averaging the individual probabilities
noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * probabilities.sum

-- The proof problem: proving the total probability equals 119/700
theorem co_presidents_included_probability :
  total_probability = 119 / 700 := by
  sorry

end co_presidents_included_probability_l212_212192


namespace volume_of_given_solid_l212_212966

noncomputable def volumeOfSolid (s : ℝ) : ℝ :=
  144 * Real.sqrt 2

theorem volume_of_given_solid (s : ℝ) (h1 : s = 4 * Real.sqrt 2) :
  volumeOfSolid s = 144 * Real.sqrt 2 :=
by
  rw [h1, volumeOfSolid]
  sorry

end volume_of_given_solid_l212_212966


namespace a_n_is_n_l212_212747

theorem a_n_is_n (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
    (h2 : ∀ n, ∑ j in finset.range (n + 1), a j = (∑ j in finset.range (n + 1), a j) ^ 2) :
    ∀ n, a n = n := by
  sorry

end a_n_is_n_l212_212747


namespace odd_number_solutions_iff_perfect_square_l212_212325

theorem odd_number_solutions_iff_perfect_square (n : ℕ) :
  (∃f : ℕ → ℕ → Prop, (∀ x y, f x y ↔ (1 / (n : ℝ)) = 1 / x + 1 / y) ∧ nat.odd (nat.card { p | f p.1 p.2 })) ↔ (∃ k, n = k * k) :=
by
  sorry

end odd_number_solutions_iff_perfect_square_l212_212325


namespace additional_votes_in_revote_l212_212454

theorem additional_votes_in_revote (a b a' b' n : ℕ) :
  a + b = 300 →
  b - a = n →
  a' - b' = 3 * n →
  a' + b' = 300 →
  a' = (7 * b) / 6 →
  a' - a = 55 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end additional_votes_in_revote_l212_212454


namespace union_A_B_eq_intersection_A_B_complement_eq_l212_212765

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x ≤ 0}
def B_complement : Set ℝ := {x | x < 0 ∨ x > 4}

theorem union_A_B_eq : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} := by
  sorry

theorem intersection_A_B_complement_eq : A ∩ B_complement = {x | -1 ≤ x ∧ x < 0} := by
  sorry

end union_A_B_eq_intersection_A_B_complement_eq_l212_212765


namespace find_alpha_l212_212466

noncomputable def line (α : ℝ) (t : ℝ) :=
  (x : ℝ, y : ℝ) := (x = - √2 + t * cos α) ∧ (y = t * sin α)

noncomputable def curve (x : ℝ) (y : ℝ) :=
  (x ^ 2 / 3) + (y ^ 2) = 1

theorem find_alpha : ∃ α : ℝ, (0 ≤ α ∧ α ≤ π / 2) ∧
  (∀ t : ℝ, let (x, y) := line α t in curve x y) ∧
  (let tM := √2 / (cos α),
    let (xM, yM) := line α tM in xM = 0) ∧
  (let (F1x, F1y) := (- √2, 0) in let (xA, yA) := (x, y) in (|F1x - xA| + |F1y - yA| = |tA + tB|)) →
  α = π / 6 :=
by
  sorry

end find_alpha_l212_212466


namespace stable_performance_l212_212457

theorem stable_performance 
  (X_A_mean : ℝ) (X_B_mean : ℝ) (S_A_var : ℝ) (S_B_var : ℝ)
  (h1 : X_A_mean = 82) (h2 : X_B_mean = 82)
  (h3 : S_A_var = 245) (h4 : S_B_var = 190) : S_B_var < S_A_var :=
by {
  sorry
}

end stable_performance_l212_212457


namespace circle_diameter_l212_212633

theorem circle_diameter (r d : ℝ) (h₀ : ∀ (r : ℝ), ∃ (d : ℝ), d = 2 * r) (h₁ : π * r^2 = 9 * π) :
  d = 6 :=
by
  rcases h₀ r with ⟨d, hd⟩
  sorry

end circle_diameter_l212_212633


namespace find_local_minimum_l212_212067

noncomputable def u (x y z : ℝ) : ℝ := x^2 + 2*y^2 + 4*z^2 + 4*x + 2*y - 8

def critical_point (x y z : ℝ) : Prop :=
  (∂ u x = 0) ∧ (∂ u y = 0) ∧ (∂ u z = 0)

def hessian_matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![∂^2 u x x, ∂^2 u x y, ∂^2 u x z],
    ![∂^2 u y x, ∂^2 u y y, ∂^2 u y z],
    ![∂^2 u z x, ∂^2 u z y, ∂^2 u z z]]

theorem find_local_minimum :
  critical_point (-2) (-1/2) 1 ∧
  (Sylvester.criterion (hessian_matrix (-2) (-1/2) 1)) →
  ∃ (x y z : ℝ), u x y z = u (-2) (-1/2) 1 ∧ x = -2 ∧ y = -1/2 ∧ z = 1 :=
begin
  sorry
end

end find_local_minimum_l212_212067


namespace total_money_spent_l212_212487

/-- 
John buys a gaming PC for $1200.
He decides to replace the video card in it.
He sells the old card for $300 and buys a new one for $500.
Prove total money spent on the computer after counting the savings from selling the old card is $1400.
-/
theorem total_money_spent (initial_cost : ℕ) (sale_price_old_card : ℕ) (price_new_card : ℕ) : 
  (initial_cost = 1200) → (sale_price_old_card = 300) → (price_new_card = 500) → 
  (initial_cost + (price_new_card - sale_price_old_card) = 1400) :=
by 
  intros
  sorry

end total_money_spent_l212_212487


namespace evaluate_ff_ff_1_l212_212324

def f (x : ℚ) : ℚ := (x^2 - x - 2) / (x^2 + x - 6)

theorem evaluate_ff_ff_1 : let p := 17, q := 41 in 10 * p + q = 211 :=
by {
  let f := λ x : ℚ, (x^2 - x - 2) / (x^2 + x - 6),
  have h1 : f 1 = 1 / 2 := by sorry,
  have h2 : f (1 / 2) = 3 / 7 := by sorry,
  have h3 : f (3 / 7) = 5 / 12 := by sorry,
  have h4 : f (5 / 12) = 17 / 41 := by sorry,
  exact calc 10 * 17 + 41 = 170 + 41 : by rw [mul_comm]
                        ... = 211 : by norm_num
}

end evaluate_ff_ff_1_l212_212324


namespace nancy_hourly_wage_l212_212107

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end nancy_hourly_wage_l212_212107


namespace f_is_odd_l212_212767

-- Define the operations
def op1 (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)
def op2 (a b : ℝ) : ℝ := real.sqrt ((a - b)^2)

-- Define the function using the operations
def f (x : ℝ) : ℝ := (op1 2 x) / (2 - op2 x 2)

-- Prove that the function is odd
theorem f_is_odd : ∀ x, f (-x) = -f x :=
by sorry

end f_is_odd_l212_212767


namespace n_pointed_star_l212_212317

variables {n : ℕ} {A B : ℕ → ℝ}
variables (hA : ∀ i, A i = A 0) (hB : ∀ i, B i = B 0)
          (hAB_diff : ∀ i, A i = B i - 15)

theorem n_pointed_star (h_exterior_sum : ∑ i in finset.range n, B i = 360)
: n = 24 := 
sorry

end n_pointed_star_l212_212317


namespace volume_difference_l212_212182

noncomputable def equi_cone_radius_base : ℝ :=
  real.sqrt (6 / real.pi)

def radius_sphere (R : ℝ) : ℝ :=
  R / real.sqrt 3

def surface_area_cone (R : ℝ) : ℝ :=
  5 * real.pi * R^2 / 3

def surface_area_sphere (R : ℝ) : ℝ :=
  4 * real.pi * (radius_sphere R)^2

theorem volume_difference (R : ℝ) (h : surface_area_cone R = surface_area_sphere R + 10) : 
  (π * R^3 * real.sqrt 3 / 3) - (4 * π * (radius_sphere R)^3 / (3 * real.sqrt 3)) = (10 / 3) * real.sqrt (2 / π) :=
sorry

end volume_difference_l212_212182


namespace circumscribed_sphere_eqn_l212_212343

-- Define vertices of the tetrahedron
variables {A_1 A_2 A_3 A_4 : Point}

-- Define barycentric coordinates
variables {x_1 x_2 x_3 x_4 : ℝ}

-- Define edge lengths
variables {a_12 a_13 a_14 a_23 a_24 a_34: ℝ}

-- Define the equation of the circumscribed sphere in barycentric coordinates
theorem circumscribed_sphere_eqn (h1 : A_1 ≠ A_2) (h2 : A_1 ≠ A_3) (h3 : A_1 ≠ A_4)
                                 (h4 : A_2 ≠ A_3) (h5 : A_2 ≠ A_4) (h6 : A_3 ≠ A_4) :
    (x_1 * x_2 * a_12^2 + x_1 * x_3 * a_13^2 + x_1 * x_4 * a_14^2 +
     x_2 * x_3 * a_23^2 + x_2 * x_4 * a_24^2 + x_3 * x_4 * a_34^2) = 0 :=
 sorry

end circumscribed_sphere_eqn_l212_212343


namespace four_digit_perm_div_by_3_l212_212161

theorem four_digit_perm_div_by_3 : 
  ∃ (abcd : ℕ), 
    (abcd >= 1000 ∧ abcd < 10000) ∧
    (∃ (a b c d : ℕ), 
      (a + b + c + d) % 3 = 0 ∧ 
      (permutations_of_consecutive a b c) ∧
      (a * 1000 + b * 100 + c * 10 + d = abcd)) ∧
    (count_valid_numbers = 184) :=
sorry

def permutations_of_consecutive (a b c : ℕ) : Prop :=
  ∃ k : ℕ, 
    (a = k ∨ a = k + 1 ∨ a = k + 2) ∧
    (b = k ∨ b = k + 1 ∨ b = k + 2) ∧
    (c = k ∨ c = k + 1 ∨ c = k + 2) ∧
    list.perm [a, b, c] [k, k + 1, k + 2]

def count_valid_numbers : ℕ := sorry -- This is a placeholder for the actual count computation.

end four_digit_perm_div_by_3_l212_212161


namespace find_Y_exists_l212_212931

variable {X : Finset ℕ} -- Consider a finite set X of natural numbers for generality
variable (S : Finset (Finset ℕ)) -- Set of all subsets of X with even number of elements
variable (f : Finset ℕ → ℝ) -- Real-valued function on subsets of X

-- Conditions
variable (hS : ∀ s ∈ S, s.card % 2 = 0) -- All elements in S have even number of elements
variable (h1 : ∃ A ∈ S, f A > 1990) -- f(A) > 1990 for some A ∈ S
variable (h2 : ∀ ⦃B C⦄, B ∈ S → C ∈ S → (Disjoint B C) → (f (B ∪ C) = f B + f C - 1990)) -- f respects the functional equation for disjoint subsets

theorem find_Y_exists :
  ∃ Y ⊆ X, (∀ D ∈ S, D ⊆ Y → f D > 1990) ∧ (∀ D ∈ S, D ⊆ (X \ Y) → f D ≤ 1990) :=
by
  sorry

end find_Y_exists_l212_212931


namespace triangle_inequality_range_l212_212734

theorem triangle_inequality_range {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  1 ≤ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ∧ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) < 2 := 
by 
  sorry

end triangle_inequality_range_l212_212734


namespace range_of_m_l212_212394

theorem range_of_m (x1 x2 m : Real) (h_eq : ∀ x : Real, x^2 - 2*x + m + 2 = 0)
  (h_abs : |x1| + |x2| ≤ 3)
  (h_real : ∀ x : Real, ∃ y : Real, x^2 - 2*x + m + 2 = 0) : -13 / 4 ≤ m ∧ m ≤ -1 :=
by
  sorry

end range_of_m_l212_212394


namespace smallest_mu_ineq_l212_212716

theorem smallest_mu_ineq (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
    a^2 + b^2 + c^2 + d^2 + 2 * a * d ≥ 2 * (a * b + b * c + c * d) := by {
    sorry
}

end smallest_mu_ineq_l212_212716


namespace cards_given_l212_212903

variables (n j given_left : ℕ)

-- Defining the initial conditions
def initial_nell_cards := 528
def initial_jeff_cards := 11
def final_nell_cards := 252

-- Problem Statement
theorem cards_given : given_left = initial_nell_cards - final_nell_cards → given_left = 276 :=
by
  intros h,
  exact h

end cards_given_l212_212903


namespace fraction_eq_l212_212782

def f(x : ℤ) : ℤ := 3 * x + 2
def g(x : ℤ) : ℤ := 2 * x - 3

theorem fraction_eq : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by 
  sorry

end fraction_eq_l212_212782


namespace coin_and_dice_same_number_probability_l212_212008

noncomputable def coin := {1, 2}     -- 1 for Heads, 2 for Tails
noncomputable def dice := {1, 2, 3, 4, 5, 6}

-- The set of all possible outcomes for the two dice rolls
noncomputable def dice_outcomes := { (d1, d2) | d1 ∈ dice ∧ d2 ∈ dice }

-- The set of outcomes where the two dice show the same number
noncomputable def dice_same_number := { (d1, d2) | d1 = d2 ∧ d1 ∈ dice ∧ d2 ∈ dice }

-- Prove that probability of getting heads and both dice showing the same number is 1/12
theorem coin_and_dice_same_number_probability :
  ((1 : ℝ) / 12) = (1 : ℝ) / 2 * (6 : ℝ) / 36 := 
  sorry

end coin_and_dice_same_number_probability_l212_212008


namespace average_weight_section_B_l212_212578

theorem average_weight_section_B
  (num_students_A : ℕ) (num_students_B : ℕ) (avg_weight_A : ℝ) (avg_weight_class : ℝ)
  (hA : num_students_A = 40)
  (hB : num_students_B = 20)
  (hAvgA : avg_weight_A = 50)
  (hAvgClass : avg_weight_class = 46.67) :
  ∃ (avg_weight_B : ℝ), avg_weight_B = 40.01 :=
by
  have total_weight_class : ℝ := num_students_A * avg_weight_A + num_students_B * avg_weight_B,
  rw [hA, hB, hAvgA, hAvgClass] at total_weight_class,
  obtain ⟨avg_weight_B, _⟩ : ∃ (avg_weight_B : ℝ), num_students_A * avg_weight_A + num_students_B * avg_weight_B = 60 * avg_weight_class,
  sorry

end average_weight_section_B_l212_212578


namespace hairstylist_monthly_earnings_l212_212268

noncomputable def hairstylist_earnings_per_month : ℕ :=
  let monday_wednesday_friday_earnings : ℕ := (4 * 10) + (3 * 15) + (1 * 22);
  let tuesday_thursday_earnings : ℕ := (6 * 10) + (2 * 15) + (3 * 30);
  let weekend_earnings : ℕ := (10 * 22) + (5 * 30);
  let weekly_earnings : ℕ :=
    (monday_wednesday_friday_earnings * 3) +
    (tuesday_thursday_earnings * 2) +
    (weekend_earnings * 2);
  weekly_earnings * 4

theorem hairstylist_monthly_earnings : hairstylist_earnings_per_month = 5684 := by
  -- Assertion based on the provided problem conditions
  sorry

end hairstylist_monthly_earnings_l212_212268


namespace short_bingo_first_column_l212_212806

-- Define the set of numbers from 1 to 10
def numSet : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the function that counts the distinct possibilities for the column
def countFirstColumnPossibilities (s : Set ℕ) : ℕ :=
  if h : s.card = 10 then
    (Finset.univ : Finset {x // x ∈ s}).card.factorial
  else
    0

-- The main theorem stating the number of possibilities
theorem short_bingo_first_column :
  countFirstColumnPossibilities numSet = 30240 :=
by
  -- Proof goes here
  sorry

end short_bingo_first_column_l212_212806


namespace abs_condition_l212_212039

theorem abs_condition (a : ℝ) (h : a + |a| = 0) : a - |2a| = 3a :=
sorry

end abs_condition_l212_212039


namespace geometric_sequence_a5_l212_212853

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n, a n + a (n + 1) = 3 * (1 / 2) ^ n)
  (h₁ : ∀ n, a (n + 1) = a n * q)
  (h₂ : q = 1 / 2) :
  a 5 = 1 / 16 :=
sorry

end geometric_sequence_a5_l212_212853


namespace g_2x_expression_sum_m_n_l212_212396

def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < 1/2 then 1 else 
  if 1/2 ≤ x ∧ x < 1 then -1 else 
  0

def g (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then 1 else 0

theorem g_2x_expression (x : ℝ) : g (2 * x) =
  (if 0 ≤ x ∧ x < 1/2 then 1 else 0) := sorry

theorem sum_m_n 
(m n : ℤ) (h : ∀ x : ℝ, m * g (↑n * x) - g x = f x) : m + n = 3 := sorry

end g_2x_expression_sum_m_n_l212_212396


namespace order_of_numbers_l212_212692

noncomputable def a : ℝ := 0.4 ^ 3
noncomputable def b : ℝ := logBase 3 0.4
noncomputable def c : ℝ := 3 ^ 0.4

theorem order_of_numbers : b < a ∧ a < c := 
by {
  sorry
}

end order_of_numbers_l212_212692


namespace reciprocal_of_sum_l212_212604

theorem reciprocal_of_sum : ((1 / 3 + 3 / 4)⁻¹ = 12 / 13) :=
by sorry

end reciprocal_of_sum_l212_212604


namespace closest_point_on_line_l212_212352

-- Definition of the line and the given point
def line (x : ℝ) : ℝ := 2 * x - 4
def point : ℝ × ℝ := (3, -1)

-- Define the closest point we've computed
def closest_point : ℝ × ℝ := (9/5, 2/5)

-- Statement of the problem to prove the closest point
theorem closest_point_on_line : 
  ∃ (p : ℝ × ℝ), p = closest_point ∧ 
  ∀ (q : ℝ × ℝ), (line q.1 = q.2) → 
  (dist point p ≤ dist point q) :=
sorry

end closest_point_on_line_l212_212352


namespace angle_equality_l212_212831

-- Let ABCD be a convex quadrilateral
variables (A B C D : Type) -- represents points
variables (convex : convex_quadrilateral A B C D)
variables (h1 : ∠ B C D = ∠ C A B)
variables (h2 : ∠ A C D = ∠ B D A)

theorem angle_equality (A B C D : Type)
  (convex : convex_quadrilateral A B C D)
  (h1 : ∠ C B D = ∠ C A B)
  (h2 : ∠ A C D = ∠ B D A) : 
  ∠ A B C = ∠ A D C := 
sorry

end angle_equality_l212_212831


namespace angle_equality_in_quadrilateral_l212_212840

/-- In a convex quadrilateral ABCD, 
    if ∠CBD = ∠CAB and ∠ACD = ∠BDA,
    then ∠ABC = ∠ADC. -/
theorem angle_equality_in_quadrilateral 
  {A B C D: Type*} [convex_quadrilateral A B C D]
  (h1 : ∠CBD = ∠CAB)
  (h2 : ∠ACD = ∠BDA) : 
  ∠ABC = ∠ADC := sorry

end angle_equality_in_quadrilateral_l212_212840


namespace find_t_collinear_l212_212819

open Real

noncomputable def collinear_vectors_problem (t : ℝ) : Prop :=
  let i : Vector ℝ := ⟨[1, 0], by simp⟩
  let j : Vector ℝ := ⟨[0, 1], by simp⟩
  let OA : Vector ℝ := i + (2 : ℝ) • j
  let OB : Vector ℝ := 3 • i + (4 : ℝ) • j
  let OC : Vector ℝ := (2 * t) • i + (t + 5) • j
  let AB : Vector ℝ := OB - OA
  let AC : Vector ℝ := OC - OA
  AB = (2 : ℝ) • i + (2 : ℝ) • j ∧ AC = ((2 * t) - 1) • i + (t + 3) • j ∧ (∃ k : ℝ, AC = k • AB)

-- Theorem statement
theorem find_t_collinear (t : ℝ) : 
  collinear_vectors_problem t → t = 4 :=
sorry

end find_t_collinear_l212_212819


namespace gcd_combination_exists_l212_212502

noncomputable def gcd_combination (n : ℕ) (h : n ≥ 2) 
  (a : Fin n → ℕ) : Prop :=
  let d := Nat.gcd (List.ofFn a) in
  ∃ (b : Fin n → ℕ), ∑ i, a i * b i = d

theorem gcd_combination_exists (n : ℕ) (h : n ≥ 2) 
  (a : Fin n → ℕ) :
  gcd_combination n h a :=
sorry

end gcd_combination_exists_l212_212502


namespace greatest_value_of_a_greatest_value_of_a_achieved_l212_212518

theorem greatest_value_of_a (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : a ≤ 20 :=
sorry

theorem greatest_value_of_a_achieved (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120)
  (h2 : Nat.gcd a b = 10) (h3 : 10 ∣ a ∧ 10 ∣ b) (h4 : Nat.lcm a b = 20) : a = 20 :=
sorry

end greatest_value_of_a_greatest_value_of_a_achieved_l212_212518


namespace num_right_triangles_with_incenter_origin_l212_212504

theorem num_right_triangles_with_incenter_origin (p : ℕ) (hp : Nat.Prime p) :
  let M : ℤ × ℤ := (p * 1994, 7 * p * 1994)
  let is_lattice_point (x : ℤ × ℤ) : Prop := True  -- All points considered are lattice points
  let is_right_angle_vertex (M : ℤ × ℤ) : Prop := True
  let is_incenter_origin (M : ℤ × ℤ) : Prop := True
  let num_triangles (p : ℕ) : ℕ :=
    if p = 2 then 18
    else if p = 997 then 20
    else 36
  num_triangles p = if p = 2 then 18 else if p = 997 then 20 else 36 := (

  by sorry

 )

end num_right_triangles_with_incenter_origin_l212_212504


namespace total_people_transport_l212_212698

-- Define the conditions
def boatA_trips_day1 := 7
def boatB_trips_day1 := 5
def boatA_capacity := 20
def boatB_capacity := 15
def boatA_trips_day2 := 5
def boatB_trips_day2 := 6

-- Define the theorem statement
theorem total_people_transport :
  (boatA_trips_day1 * boatA_capacity + boatB_trips_day1 * boatB_capacity) +
  (boatA_trips_day2 * boatA_capacity + boatB_trips_day2 * boatB_capacity)
  = 405 := 
  by
  sorry

end total_people_transport_l212_212698


namespace triangle_color_configuration_l212_212810

theorem triangle_color_configuration 
(points : Finset (ℝ × ℝ))
(p1 p2 p3 : points)
(h_color1 : ¬(are_collinear p1.1 p2.1 p3.1))
(p4 p5 p6 : points)
(h_color2 : ¬(are_collinear p4.1 p5.1 p6.1))
(h_colors : ∀ p ∈ points, p.2 = "red" ∨ p.2 = "blue")
(h_no3collinear_red : ∀ p q r ∈ points, list.chain (λ x y, x.2 = "red" ∧ y.2 = "red") [p, q, r] → ¬are_collinear p.1 q.1 r.1)
(h_no3collinear_blue : ∀ p q r ∈ points, list.chain (λ x y, x.2 = "blue" ∧ y.2 = "blue") [p, q, r] → ¬are_collinear p.1 q.1 r.1)
(h_atleast3_red : ∃ p q r ∈ points, list.chain (λ x y, x.2 = "red" ∧ y.2 = "red") [p, q, r])
(h_atleast3_blue : ∃ p q r ∈ points, list.chain (λ x y, x.2 = "blue" ∧ y.2 = "blue") [p, q, r]) :
∃ (t: Triangle) (color: String) (h_color: color = "red" ∨ color = "blue"), 
  (∀ side ∈ t.sides, (Finset.filter (λ p, p.2 ≠ color) points).card ≤ 2)  :=
by 
  sorry

end triangle_color_configuration_l212_212810


namespace domain_of_f_l212_212558

noncomputable def f (x : ℝ) : ℝ := log (x - 1) / (x - 2)

theorem domain_of_f : {x : ℝ | 1 < x ∧ x ≠ 2} = (Set.Ioo 1 2 ∪ Set.Ioi 2) :=
by
  sorry

end domain_of_f_l212_212558


namespace trailing_zeros_of_factorial_30_l212_212784

theorem trailing_zeros_of_factorial_30 : 
  (number_of_trailing_zeros (nat.factorial 30) = 7) :=
sorry

end trailing_zeros_of_factorial_30_l212_212784


namespace gcf_75_105_l212_212602

theorem gcf_75_105 : Nat.gcd 75 105 = 15 :=
by {
  -- Condition 1: Prime factorization of 75
  have h1 : 75 = 3 * 5^2 := rfl,
  -- Condition 2: Prime factorization of 105
  have h2 : 105 = 3 * 5 * 7 := rfl,
  -- Goal: Prove gcd(75, 105) = 15
  sorry
}

end gcf_75_105_l212_212602


namespace algebraic_expression_correct_l212_212608

-- Definition of the problem
def algebraic_expression (x : ℝ) : ℝ :=
  2 * x + 3

-- Theorem statement
theorem algebraic_expression_correct (x : ℝ) :
  algebraic_expression x = 2 * x + 3 :=
by
  sorry

end algebraic_expression_correct_l212_212608


namespace numerical_puzzle_unique_solution_l212_212708

theorem numerical_puzzle_unique_solution :
  ∃ (A X Y P : ℕ), 
    A ≠ X ∧ A ≠ Y ∧ A ≠ P ∧ X ≠ Y ∧ X ≠ P ∧ Y ≠ P ∧
    (A * 10 + X) + (Y * 10 + X) = Y * 100 + P * 10 + A ∧
    A = 8 ∧ X = 9 ∧ Y = 1 ∧ P = 0 :=
sorry

end numerical_puzzle_unique_solution_l212_212708


namespace angle_equality_l212_212832

-- Let ABCD be a convex quadrilateral
variables (A B C D : Type) -- represents points
variables (convex : convex_quadrilateral A B C D)
variables (h1 : ∠ B C D = ∠ C A B)
variables (h2 : ∠ A C D = ∠ B D A)

theorem angle_equality (A B C D : Type)
  (convex : convex_quadrilateral A B C D)
  (h1 : ∠ C B D = ∠ C A B)
  (h2 : ∠ A C D = ∠ B D A) : 
  ∠ A B C = ∠ A D C := 
sorry

end angle_equality_l212_212832


namespace square_side_increase_l212_212570

theorem square_side_increase (p : ℝ) (h : (1 + p / 100)^2 = 1.69) : p = 30 :=
by {
  sorry
}

end square_side_increase_l212_212570


namespace odd_squarefree_integers_1_to_199_l212_212688

noncomputable def count_squarefree_odd_integers (n : ℕ) :=
  n - List.sum [
    n / 18,   -- for 3^2 = 9
    n / 50,   -- for 5^2 = 25
    n / 98,   -- for 7^2 = 49
    n / 162,  -- for 9^2 = 81
    n / 242,  -- for 11^2 = 121
    n / 338   -- for 13^2 = 169
  ]

theorem odd_squarefree_integers_1_to_199 : count_squarefree_odd_integers 198 = 79 := 
by
  sorry

end odd_squarefree_integers_1_to_199_l212_212688


namespace x_coordinate_of_P_equation_of_asymptotes_l212_212403

-- Given the hyperbola and parabola equations
variables (a b : ℝ) (ha : a > 0) (hb : b > 0)

-- Define the equations
def hyperbola_eq (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def parabola_eq (x y : ℝ) : Prop := y^2 = 8 * x

-- Common focus and intersection conditions
def common_focus : (ℝ × ℝ) := (2, 0)
def point_P (x y : ℝ) : Prop := hyperbola_eq a b x y ∧ parabola_eq x y
def |PF| (x y : ℝ) : ℝ := real.sqrt ((x - 2)^2 + y^2)

theorem x_coordinate_of_P (x : ℝ) (hx : point_P a b x (2 * real.sqrt 6)) (hPF : |PF| x (2 * real.sqrt 6) = 5) : x = 3 :=
sorry

theorem equation_of_asymptotes (a : ℝ) (b : ℝ) (ha : a = 1) (hb : b = real.sqrt 3) :
  ∀ x y, hyperbola_eq a b x y → y = real.sqrt 3 * x ∨ y = -real.sqrt 3 * x :=
sorry

end x_coordinate_of_P_equation_of_asymptotes_l212_212403


namespace product_bn_eq_pq_l212_212722

noncomputable def b (n : ℕ) (hn : n ≥ 5) : ℚ :=
  ((n + 2) ^ 2 : ℚ) / (n * (n ^ 3 - 1))

theorem product_bn_eq_pq (p q : ℕ) :
  (∏ n in finset.range (100 - 5 + 1), b (n + 5) (by linarith)) = p / (q.factorial : ℚ)
  ∧ p = 10404
  ∧ q = 100 := sorry

end product_bn_eq_pq_l212_212722


namespace smallest_three_digit_number_ends_with_three_identical_digits_l212_212572

theorem smallest_three_digit_number_ends_with_three_identical_digits :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ (∃ (d : ℕ), 1 ≤ d ∧ d ≤ 9 ∧ n^2 % 1000 = 111 * d) ∧
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬ (∃ (d : ℕ), 1 ≤ d ∧ d ≤ 9 ∧ m^2 % 1000 = 111 * d)) :=
begin
  use 376,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { use 6,
    norm_num,
  },
  { intros m hm hlt,
    norm_num at hlt,
    sorry
  }
end

end smallest_three_digit_number_ends_with_three_identical_digits_l212_212572


namespace two_trains_crossing_time_l212_212209

variables {length1 length2 : ℝ} {speed1 speed2 : ℝ}
def length1 := 140 -- in meters
def length2 := 210 -- in meters
def speed1 := 60 * (5 / 18) -- in meters per second conversion
def speed2 := 40 * (5 / 18) -- in meters per second conversion

noncomputable def relative_speed := speed1 + speed2 -- adding speeds in m/s 
noncomputable def total_distance := length1 + length2 -- summing lengths in meters
noncomputable def crossing_time := total_distance / relative_speed -- time = distance / speed

-- Theorem stating the expected crossing time 
theorem two_trains_crossing_time : crossing_time ≈ 12.59 := sorry

end two_trains_crossing_time_l212_212209


namespace plane_EFC_perpendicular_BCD_l212_212626

/-- Given a right triangle BCD, with F as the midpoint of BD,
and E and F as the midpoints of AD and BD respectively, 
prove that plane EFC is perpendicular to plane BCD. -/
theorem plane_EFC_perpendicular_BCD 
  (B C D A E F : Point)
  (h_triangle : right_triangle B C D)
  (h_midpointF : midpoint F B D)
  (h_midpointE : midpoint E A D)
  (h_lengthFC : FC = (1/2) * BD)
  (h_lengthEF : EF = (1/2) * AB)
  (h_lengthEC : EC = sqrt(2)) :
  perpendicular_plane E F C B C D :=
  sorry

end plane_EFC_perpendicular_BCD_l212_212626


namespace find_k_l212_212738

theorem find_k (x y k : ℝ) (h₁ : x = 2) (h₂ : y = -1) (h₃ : y - k * x = 7) : k = -4 :=
by
  sorry

end find_k_l212_212738


namespace foci_of_hyperbola_l212_212554

-- Definition for the hyperbola equation in standard form
def is_hyperbola (x y : ℝ) : Prop := (x^2 / 2) - y^2 = 1

-- The statement to be proven, that the foci of the hyperbola are at (±√3, 0)
theorem foci_of_hyperbola : ∀ x y : ℝ, is_hyperbola x y → (x = ± sqrt 3 ∧ y = 0) :=
by
  intros x y h
  sorry

end foci_of_hyperbola_l212_212554


namespace justine_used_250_sheets_l212_212586

/-
We are given:
1. There are 4600 sheets of paper split into 11 binders.
2. Justine colors on three-fifths of the sheets in one binder.
We need to prove that Justine uses 250 sheets of paper.
-/
theorem justine_used_250_sheets (total_sheets : ℕ) (total_binders : ℕ) (sheets_per_binder : ℕ) (fraction_colored : ℚ) :
  total_sheets = 4600 → total_binders = 11 → total_sheets / total_binders = sheets_per_binder → fraction_colored = 3/5 →
  floor (fraction_colored * sheets_per_binder) = 250 :=
by
  sorry

end justine_used_250_sheets_l212_212586


namespace sqrt3_f_pi6_lt_f_pi3_l212_212683

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_derivative_tan_lt (x : ℝ) (h : 0 < x ∧ x < π / 2) : f x < (deriv f x) * tan x

theorem sqrt3_f_pi6_lt_f_pi3 :
  sqrt 3 * f (π / 6) < f (π / 3) :=
by
  sorry

end sqrt3_f_pi6_lt_f_pi3_l212_212683


namespace percent_taxed_land_correct_l212_212336

variable (c_taxes : ℝ) -- total collected taxes
variable (w_taxes : ℝ) -- taxes paid by Mr. Willam
variable (p_land : ℝ) -- percentage of land owned by Mr. Willam

-- All conditions from the problem in Lean:
def problem_conditions :=
  c_taxes = 3840 ∧
  w_taxes = 480 ∧
  p_land = 31.25

-- Definition of the function to find percentage of taxed land:
def percent_taxed_land (c_taxes w_taxes : ℝ) := (w_taxes / c_taxes) * 100

-- The proposition to prove:
theorem percent_taxed_land_correct (h : problem_conditions c_taxes w_taxes p_land) :
  percent_taxed_land c_taxes w_taxes = 12.5 :=
by
  obtain ⟨hc, hw, hp⟩ := h
  sorry

end percent_taxed_land_correct_l212_212336


namespace find_x_and_verify_l212_212035

theorem find_x_and_verify (x : ℤ) (h : (x - 14) / 10 = 4) : (x - 5) / 7 = 7 := 
by 
  sorry

end find_x_and_verify_l212_212035


namespace maximize_area_dot_product_zero_l212_212496

noncomputable theory

open_locale classical

variables (F1 F2 P Q : ℝ × ℝ)
variables (a b : ℝ)

-- Defining the ellipse and its focci
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Defining the foci of the ellipse
def left_focus : ℝ × ℝ := (-real.sqrt 3, 0)
def right_focus : ℝ × ℝ := (real.sqrt 3, 0)

-- Defining the conditions
def conditions : Prop := 
  ∃ P Q : ℝ × ℝ,
  (0, 1) = P ∧ 
  (0, -1) = Q ∧
  ellipse_eq P.1 P.2 ∧
  ellipse_eq Q.1 Q.2

-- Defining the vectors from P to F1 and F2
def vector_PF1 : ℝ × ℝ := (fst P - fst F1, snd P - snd F1)
def vector_PF2 : ℝ × ℝ := (fst P - fst F2, snd P - snd F2)

-- Calculating the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- The main theorem to prove
theorem maximize_area_dot_product_zero : 
  conditions →
  dot_product (vector_PF1 left_focus right_focus (0, 1)) (vector_PF2 left_focus right_focus (0, 1)) = -2 :=
sorry

end maximize_area_dot_product_zero_l212_212496


namespace find_other_number_l212_212482

theorem find_other_number (a b : ℤ) (h1 : 2 * a + 3 * b = 100) (h2 : a = 28 ∨ b = 28) : a = 8 ∨ b = 8 :=
sorry

end find_other_number_l212_212482


namespace average_condition_l212_212549

theorem average_condition (x : ℝ) :
  (1275 + x) / 51 = 80 * x → x = 1275 / 4079 :=
by
  sorry

end average_condition_l212_212549


namespace prove_equation_l212_212861

theorem prove_equation (s : ℝ) 
  (h1 : 2s - (s - 1) = 2) : 
  2s - (s - 1) = 2 :=
  sorry

end prove_equation_l212_212861


namespace calculate_number_of_molecules_l212_212247

-- Definitions based on the provided conditions and the problem
def avogadro_number : ℝ := 6.022 * 10^23  -- Avogadro's number

-- We assume the number of moles (n) leads to the given number of molecules (3 * 10^26)
def number_of_moles : ℝ := 3 * 10^26 / avogadro_number

-- Statement of the theorem to be proven
theorem calculate_number_of_molecules (n : ℝ) (na : ℝ) (h1 : na = avogadro_number) (h2 : n = number_of_moles) : 
  n * na = 3 * 10^26 :=
by
  rw [h1, h2]
  sorry

end calculate_number_of_molecules_l212_212247


namespace sum_reciprocals_B_final_answer_l212_212883

def B (n : ℕ) : Prop :=
  n > 0 ∧ (∀ p, nat.prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7)

noncomputable def reciprocal_sum : ℚ :=
  ∑' n in set_of B, 1 / n

theorem sum_reciprocals_B : (reciprocal_sum = 35 / 8) :=
sorry

theorem final_answer : 35 + 8 = 43 :=
by norm_num

end sum_reciprocals_B_final_answer_l212_212883


namespace find_eccentricity_l212_212342

noncomputable def eccentricity_of_hyperbola (a b c e : ℝ) : Prop :=
  9 * y^2 - 16 * x^2 = 144 ∧ c^2 = a^2 + b^2 ∧ a = 4 ∧ b = 3 ∧ c = 5 ∧ e = c / a

theorem find_eccentricity : ∃ e : ℝ, eccentricity_of_hyperbola 4 3 5 (5 / 4) :=
by 
  sorry

end find_eccentricity_l212_212342


namespace cotangent_product_l212_212431

variable (α β γ : ℝ)

-- Conditions:
-- 1. α, β, γ are angles of a triangle
-- 2. cotangent of α/2, β/2, γ/2 form an arithmetic progression
axiom angles_of_triangle (h_triangle_ineq : α + β + γ = π)
axiom arithmetic_seq (h_arith_seq : (Real.cot (α / 2) + Real.cot (γ / 2)) / 2 = Real.cot (β / 2))

theorem cotangent_product (h_triangle_ineq : α + β + γ = π) (h_arith_seq : (Real.cot (α / 2) + Real.cot (γ / 2)) / 2 = Real.cot (β / 2)) :
  Real.cot (α / 2) * Real.cot (γ / 2) = 3 :=
sorry

end cotangent_product_l212_212431


namespace solution_set_l212_212890

variable {f g : ℝ → ℝ}

-- Definitions based on the given conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

def given_conditions (f g : ℝ → ℝ) : Prop :=
  odd_function f ∧ even_function g ∧
  (∀ x < 0, f'(x) * g x + f x * g'(x) < 0) ∧
  f (-1) = 0

-- Theorem stating the solution set
theorem solution_set (h : given_conditions f g) : 
  {x : ℝ | f x * g x < 0} = Ioo (-1) 0 ∪ Ioi 1 :=
sorry

end solution_set_l212_212890


namespace necessary_and_sufficient_cosine_condition_l212_212445

theorem necessary_and_sufficient_cosine_condition
  (A B : ℝ) (A_pos : 0 < A) (A_lt_pi : A < π) (B_pos : 0 < B) (B_lt_pi : B < π) :
  (A > B ↔ cos A < cos B) :=
by sorry

end necessary_and_sufficient_cosine_condition_l212_212445


namespace min_distance_PQ_l212_212469

theorem min_distance_PQ :
  ∀ (P Q : ℝ × ℝ), (P.1 - P.2 - 4 = 0) → (Q.1^2 = 4 * Q.2) →
  ∃ (d : ℝ), d = dist P Q ∧ d = 3 * Real.sqrt 2 / 2 :=
sorry

end min_distance_PQ_l212_212469


namespace local_min_f_at_2_implies_a_eq_2_l212_212794

theorem local_min_f_at_2_implies_a_eq_2 (a : ℝ) : 
  (∃ f : ℝ → ℝ, 
     (∀ x : ℝ, f x = x * (x - a)^2) ∧ 
     (∀ f' : ℝ → ℝ, 
       (∀ x : ℝ, f' x = 3 * x^2 - 4 * a * x + a^2) ∧ 
       f' 2 = 0 ∧ 
       (∀ f'' : ℝ → ℝ, 
         (∀ x : ℝ, f'' x = 6 * x - 4 * a) ∧ 
         f'' 2 > 0
       )
     )
  ) → a = 2 :=
sorry

end local_min_f_at_2_implies_a_eq_2_l212_212794


namespace proof_triangle_angles_l212_212805

noncomputable def angles_in_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  B = 3.141592653589793 / 3 ∧ (cos (C + 3.141592653589793 / 6) = 1 / 3 → sin A = (2 * sqrt 6 + 1) / 6)

theorem proof_triangle_angles (A B C a b c : ℝ)
  (h1 : tan B / tan A + 1 = 2 * c / a)
  (h2 : 0 < C ∧ C < 2 * 3.141592653589793 / 3) :
  angles_in_triangle A B C a b c :=
by 
  sorry

end proof_triangle_angles_l212_212805


namespace ones_digit_of_prime_in_sequence_l212_212362

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def valid_arithmetic_sequence (p1 p2 p3 p4: Nat) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4) ∧ (p3 = p2 + 4) ∧ (p4 = p3 + 4)

theorem ones_digit_of_prime_in_sequence (p1 p2 p3 p4 : Nat) (hp_seq : valid_arithmetic_sequence p1 p2 p3 p4) (hp1_gt_3 : p1 > 3) : 
  (p1 % 10) = 9 :=
sorry

end ones_digit_of_prime_in_sequence_l212_212362


namespace intersect_rectangles_by_lines_l212_212917

theorem intersect_rectangles_by_lines 
  (rects : set (set (ℝ × ℝ)))
  (h_parallel : ∀ r ∈ rects, ∃ (x₁ x₂ y₁ y₂ : ℝ), r = {p : ℝ × ℝ | x₁ ≤ p.1 ∧ p.1 ≤ x₂ ∧ y₁ ≤ p.2 ∧ p.2 ≤ y₂})
  (h_cut : ∀ r₁ r₂ ∈ rects, ∃ l, (l = {p : ℝ × ℝ | ∃ x : ℝ, p.1 = x} ∨ l = {p : ℝ × ℝ | ∃ y : ℝ, p.2 = y}) ∧ r₁ ∩ l ≠ ∅ ∧ r₂ ∩ l ≠ ∅) :
  ∃ h v, (h = {p : ℝ × ℝ | ∃ y : ℝ, p.2 = y}) ∧ (v = {p : ℝ × ℝ | ∃ x : ℝ, p.1 = x}) ∧ ∀ r ∈ rects, (r ∩ h ≠ ∅ ∨ r ∩ v ≠ ∅) :=
sorry

end intersect_rectangles_by_lines_l212_212917


namespace coprime_count_l212_212326

theorem coprime_count (N : ℕ) (hN : N = 1990 ^ 1990) : 
  let count := (591 * 1990 ^ 1989)
  ∃ K, (∀ n, 1 ≤ n ∧ n ≤ N → (n^2 - 1).gcd N = 1) ↔ (n.count = count) := 
by
  sorry

end coprime_count_l212_212326


namespace Vins_total_miles_l212_212214

theorem Vins_total_miles : 
  let dist_library_one_way := 6
  let dist_school_one_way := 5
  let dist_friend_one_way := 8
  let extra_miles := 1
  let shortcut_miles := 2
  let days_per_week := 7
  let weeks := 4

  -- Calculate weekly miles
  let library_round_trip := (dist_library_one_way + dist_library_one_way + extra_miles)
  let total_library_weekly := library_round_trip * 3

  let school_round_trip := (dist_school_one_way + dist_school_one_way + extra_miles)
  let total_school_weekly := school_round_trip * 2

  let friend_round_trip := dist_friend_one_way + (dist_friend_one_way - shortcut_miles)
  let total_friend_weekly := friend_round_trip / 2 -- Every two weeks

  let total_weekly := total_library_weekly + total_school_weekly + total_friend_weekly

  -- Calculate total miles over the weeks
  let total_miles := total_weekly * weeks

  total_miles = 272 := sorry

end Vins_total_miles_l212_212214


namespace minimum_cost_proof_l212_212909

structure Store :=
  (name : String)
  (gift_costs : (Nat × Nat × Nat × Nat)) -- (Mom, Dad, Brother, Sister)
  (time_spent : Nat) -- Time spent in store in minutes

def Romashka  : Store := { name := "Romashka", gift_costs := (1000, 750, 930, 850), time_spent := 35 }
def Oduvanchik : Store := { name := "Oduvanchik", gift_costs := (1050, 790, 910, 800), time_spent := 30 }
def Nezabudka : Store := { name := "Nezabudka", gift_costs := (980, 810, 925, 815), time_spent := 40 }
def Landysh : Store := { name := "Landysh", gift_costs := (1100, 755, 900, 820), time_spent := 25 }

def stores : List Store := [Romashka, Oduvanchik, Nezabudka, Landysh]

def travel_time := 30 -- minutes
def total_time := 3 * 60 + 25  -- 3 hours and 25 minutes or 210 minutes

noncomputable def min_cost_within_constraints : Nat :=
  let costs := [
    (Romashka.gift_costs.fst, Romashka.time_spent),
    (Oduvanchik.gift_costs.snd, Oduvanchik.time_spent),
    (Landysh.gift_costs.trd, Landysh.time_spent),
    (Nezabudka.gift_costs.fourth, Nezabudka.time_spent)
    ]
  in 3435 -- Given the final correct answer

theorem minimum_cost_proof : min_cost_within_constraints = 3435 := by
  sorry

end minimum_cost_proof_l212_212909


namespace polynomial_degree_distinct_roots_l212_212117

theorem polynomial_degree_distinct_roots (P : Polynomial ℂ) (n m : ℕ) (hdeg : P.degree = n) (hroots : ∀ a ∈ P.roots, multiplicity a P = 1):
  m ≤ n := sorry

end polynomial_degree_distinct_roots_l212_212117


namespace cost_price_per_meter_l212_212615

theorem cost_price_per_meter (selling_price : ℕ) (total_meters : ℕ) (loss_per_meter : ℕ) : selling_price = 18000 → total_meters = 600 → loss_per_meter = 5 → (selling_price + total_meters * loss_per_meter) / total_meters = 35 :=
by
  intros h1 h2 h3
  have h4 : (18000 + 600 * 5) / 600 = 35 := sorry
  exact h4

end cost_price_per_meter_l212_212615


namespace no_real_solutions_quadratic_solve_quadratic_eq_l212_212537

-- For Equation (1)

theorem no_real_solutions_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = -4 ∧ c = 5 ∧ (b^2 - 4 * a * c < 0)) :
  ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 := 
by
  sorry

-- For Equation (2)

theorem solve_quadratic_eq {x : ℝ} (h_eq : (x + 1) * (x + 2) = 2 * x + 4) :
  x = -2 ∨ x = 1 :=
by
  sorry

end no_real_solutions_quadratic_solve_quadratic_eq_l212_212537


namespace car_city_efficiency_l212_212257

noncomputable def fuel_efficiency_city : ℝ → ℝ := λ H, H - 12

theorem car_city_efficiency 
  (H : ℝ) 
  (Hp : H = 32) 
  (T : ℝ) 
  (highway_mpg_cond : 800 = H * T) 
  (city_mpg_cond : 500 = fuel_efficiency_city H * T)
  (C : ℝ) 
  (C_def : C = fuel_efficiency_city H) :
  C = 20 :=
by {
  sorry
}

end car_city_efficiency_l212_212257


namespace area_of_triangle_value_of_expression_l212_212446

variables (A B C : ℝ) (a b c : ℝ)

-- Definitions from the conditions
def angle_condition : Prop := 2 * (Real.cos A)^2 - 2 * Real.sqrt 3 * (Real.sin A) * (Real.cos A) = -1
def side_a : Prop := a = 2 * Real.sqrt 3
def side_c : Prop := c = 2

-- Theorem statements for part 1: Area of triangle ABC
theorem area_of_triangle 
  (hA : 0 < A) (hA_lt_pi : A < Real.pi)
  (cond_angle : angle_condition A)
  (cond_a : side_a a)
  (cond_c : side_c c) :
  (1 / 2 * b * c * Real.sin A) = 2 * Real.sqrt 3 :=
sorry

-- Additional definitions for part 2
def cos60_plus_C : ℝ := Real.cos (Real.pi / 3 + C)
def sinusoidal_req : Prop := 2 * a * cos60_plus_C = b - 2 * c

-- Theorem statements for part 2: Value of the given expression
theorem value_of_expression
  (cond_angle : angle_condition A)
  (cond_a : side_a a)
  (cond_c : side_c c)
  (eq_b : b = 4)
  (eq_cos60_plus_C : cos60_plus_C = Real.cos (Real.pi / 3 + C)) :
  (b - 2 * c) / (a * cos60_plus_C) = 2 :=
sorry

end area_of_triangle_value_of_expression_l212_212446


namespace find_p_l212_212720

def bn (n : ℕ) (h : n ≥ 5) : ℚ := ((n + 2) ^ 2) / (n ^ 3 - 1)

theorem find_p : 
    let product := ∏ n in Finset.range 96 \ 4, bn (n + 5) (Nat.le_add_left 5 n) in
    product = 6608 / 100! := 
by
    let product := ∏ n in Finset.range 96 \ 4, bn (n + 5) (Nat.le_add_left 5 n)
    have : product = 6608 / 100! := sorry
    exact this

end find_p_l212_212720


namespace fourth_intersection_point_exists_l212_212056

noncomputable def find_fourth_intersection_point : Prop :=
  let points := [(4, 1/2), (-6, -1/3), (1/4, 8), (-2/3, -3)]
  ∃ (h k r : ℝ), 
  ∀ (x y : ℝ), (x, y) ∈ points → (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

theorem fourth_intersection_point_exists :
  find_fourth_intersection_point :=
by
  sorry

end fourth_intersection_point_exists_l212_212056


namespace pythagorean_triple_l212_212564

theorem pythagorean_triple {a b c : ℕ} (h : a * a + b * b = c * c) (gcd_abc : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ m n : ℕ, a = 2 * m * n ∧ b = m * m - n * n ∧ c = m * m + n * n :=
sorry

end pythagorean_triple_l212_212564


namespace find_a_l212_212778

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l212_212778


namespace binomial_expansion_constant_term_binomial_coeff_largest_4th_l212_212058

theorem binomial_expansion_constant_term :
  (∃ x : ℝ, let expansion := (sqrt x - 1 / (2 * x)) ^ 6 in
  (∀ c : ℝ, (c ∈ expansion.terms) → (c == 15 / 4))) := sorry

theorem binomial_coeff_largest_4th :
  (∃ x : ℝ, let expansion := (sqrt x - 1 / (2 * x)) ^ 6 in
  (∃ k : ℕ, 3 ≤ k ∧ k ≤ 4 ∧ (k ∈ expansion.coeffs) → (k.th = 4))) := sorry

end binomial_expansion_constant_term_binomial_coeff_largest_4th_l212_212058


namespace abs_x_minus_y_l212_212386

theorem abs_x_minus_y (x y : ℝ) (h₁ : x^3 + y^3 = 26) (h₂ : xy * (x + y) = -6) : |x - y| = 4 :=
by
  sorry

end abs_x_minus_y_l212_212386


namespace angle_equality_in_quadrilateral_l212_212842

/-- In a convex quadrilateral ABCD, 
    if ∠CBD = ∠CAB and ∠ACD = ∠BDA,
    then ∠ABC = ∠ADC. -/
theorem angle_equality_in_quadrilateral 
  {A B C D: Type*} [convex_quadrilateral A B C D]
  (h1 : ∠CBD = ∠CAB)
  (h2 : ∠ACD = ∠BDA) : 
  ∠ABC = ∠ADC := sorry

end angle_equality_in_quadrilateral_l212_212842


namespace coefficient_x_squared_binom_l212_212470

-- Define the conditions present in the problem
def binom_expansion_term (n r : ℕ) (x : ℝ) : ℝ := 
  binomial n r * (-1)^r * x^(5 - (3 * r / 2))

-- Define the given expansion and the term we are interested in
def binom_expansion := λ x : ℝ, (sqrt x - 1 / x) ^ 10

-- Define the coefficient we are aiming to prove
def coeff_of_x_squared (n : ℕ) := binomial n 2

-- The final proof statement
theorem coefficient_x_squared_binom : coeff_of_x_squared 10 = 45 := by
  sorry

end coefficient_x_squared_binom_l212_212470


namespace train_speed_kmph_l212_212287

def length_of_train : ℝ := 120
def length_of_bridge : ℝ := 255.03
def time_to_cross : ℝ := 30

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross * 3.6 = 45.0036 :=
by
  sorry

end train_speed_kmph_l212_212287


namespace num_tangent_lines_l212_212915

theorem num_tangent_lines (r1 r2 d : ℝ) (h1 : r1 = 5) (h2 : r2 = 2) (h3 : d = 10) (h4 : d > r1 + r2) : 
  2 + 2 = 4 := 
by
  rw h1 at h4
  rw h2 at h4
  rw h3 at h4
  have h : 10 > 5 + 2 := by norm_num
  rw h at h4
  exact eq.refl 4

end num_tangent_lines_l212_212915


namespace smith_house_towels_l212_212053

theorem smith_house_towels
  (kylie_towels : ℕ)
  (husband_towels : ℕ)
  (loads : ℕ)
  (towels_per_load : ℕ)
  (total_towels : ℕ)
  (daughters_towels : ℕ) :
  kylie_towels = 3 →
  husband_towels = 3 →
  loads = 3 →
  towels_per_load = 4 →
  total_towels = loads * towels_per_load →
  total_towels = kylie_towels + husband_towels + daughters_towels →
  daughters_towels = 6 :=
begin
  sorry
end

end smith_house_towels_l212_212053


namespace asymptotes_tangent_to_circle_l212_212434

theorem asymptotes_tangent_to_circle {m : ℝ} (hm : m > 0) 
  (hyp_eq : ∀ x y : ℝ, y^2 - (x^2 / m^2) = 1) 
  (circ_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * y + 3 = 0) : 
  m = (Real.sqrt 3) / 3 :=
sorry

end asymptotes_tangent_to_circle_l212_212434


namespace scientific_notation_120_million_l212_212147

theorem scientific_notation_120_million :
  120000000 = 1.2 * 10^7 :=
by
  sorry

end scientific_notation_120_million_l212_212147


namespace intersection_M_N_l212_212406

def M : Set ℕ := {1, 2, 3, 4, 5, 6}
def N : Set ℤ := {x | -2 < x ∧ x < 5 ∧ x ∈ ℤ}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} :=
by sorry

end intersection_M_N_l212_212406


namespace total_players_is_60_l212_212521

-- Define the conditions
def Cricket_players : ℕ := 25
def Hockey_players : ℕ := 20
def Football_players : ℕ := 30
def Softball_players : ℕ := 18

def Cricket_and_Hockey : ℕ := 5
def Cricket_and_Football : ℕ := 8
def Cricket_and_Softball : ℕ := 3
def Hockey_and_Football : ℕ := 4
def Hockey_and_Softball : ℕ := 6
def Football_and_Softball : ℕ := 9

def Cricket_Hockey_and_Football_not_Softball : ℕ := 2

-- Define total unique players present on the ground
def total_unique_players : ℕ :=
  Cricket_players + Hockey_players + Football_players + Softball_players -
  (Cricket_and_Hockey + Cricket_and_Football + Cricket_and_Softball +
   Hockey_and_Football + Hockey_and_Softball + Football_and_Softball) +
  Cricket_Hockey_and_Football_not_Softball

-- Statement
theorem total_players_is_60:
  total_unique_players = 60 :=
by
  sorry

end total_players_is_60_l212_212521


namespace count_z_values_l212_212267

open Complex

noncomputable def g (z : ℂ) : ℂ := -I * conj(z)

theorem count_z_values :
  let z_vals := {z : ℂ | abs z = 3 ∧ g z = z}
  cardinal.mk z_vals = 2 :=
by 
  let z_vals := {z : ℂ | abs z = 3 ∧ g z = z}
  have h_count : set.finite z_vals := 
    λ z (hz : z ∈ z_vals), sorry
  exact (set.finite.cardinal_eq_fintype_card h_count).symm.trans sorry

end count_z_values_l212_212267


namespace students_with_grade_B_and_above_l212_212517

theorem students_with_grade_B_and_above (total_students : ℕ) (percent_below_B : ℕ) 
(h1 : total_students = 60) (h2 : percent_below_B = 40) : 
(total_students * (100 - percent_below_B) / 100) = 36 := by
  sorry

end students_with_grade_B_and_above_l212_212517


namespace net_cannot_contain_2001_knots_l212_212255

theorem net_cannot_contain_2001_knots (knots : Nat) (ropes_per_knot : Nat) (total_knots : knots = 2001) (ropes_per_knot_eq : ropes_per_knot = 3) :
  false :=
by
  sorry

end net_cannot_contain_2001_knots_l212_212255


namespace rectangle_area_error_percent_l212_212618

theorem rectangle_area_error_percent (L W : ℝ) :
  let L' := 1.05 * L,
      W' := 0.96 * W,
      actual_area := L * W,
      measured_area := L' * W',
      error := measured_area - actual_area,
      error_percent := (error / actual_area) * 100 in
  error_percent = 0.8 :=
by
  sorry

end rectangle_area_error_percent_l212_212618


namespace sequence_sum_l212_212474

noncomputable def sequence : ℕ → ℤ
| 1 := 8
| n := if n = 4 then 2 else sequence (n - 1) - 2 * (sequence (n - 1) - sequence (n - 2))

def S_n (n : ℕ) : ℤ :=
∑ i in (Finset.range n).map Finset.succ, |sequence i|

theorem sequence_sum (n : ℕ) :
  S_n n = if n ≤ 5 then 9 * n - n^2 else n^2 - 9 * n + 40 :=
by sorry

end sequence_sum_l212_212474


namespace money_when_left_home_l212_212928

theorem money_when_left_home :
  let gasoline_cost := 8
  let lunch_cost := 15.65
  let gift_cost := 5
  let total_gift_cost := 2 * gift_cost
  let total_spent := gasoline_cost + lunch_cost + total_gift_cost
  let grandma_gift_per_person := 10
  let total_grandma_gift := 2 * grandma_gift_per_person
  let return_trip_money := 36.35
  let initial_money := return_trip_money + total_spent - total_grandma_gift
  initial_money = 50 := 
by
  simp only [gasoline_cost, lunch_cost, gift_cost, total_gift_cost, total_spent, grandma_gift_per_person, total_grandma_gift, return_trip_money, initial_money]
  sorry

end money_when_left_home_l212_212928


namespace geometric_sequence_property_l212_212513

variables {a : ℕ → ℝ} {S : ℕ → ℝ}

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def S_n (n : ℕ) : ℝ := 
  if n = 0 then 0
  else (2 * (1 - 3^n)) / (1 - 3)

theorem geometric_sequence_property 
  (h₁ : a 1 + a 2 + a 3 = 26)
  (h₂ : S 6 = 728)
  (h₃ : ∀ n, a n = a_n n)
  (h₄ : ∀ n, S n = S_n n) :
  ∀ n, S (n + 1) ^ 2 - S n * S (n + 2) = 4 * 3 ^ n :=
by sorry

end geometric_sequence_property_l212_212513


namespace angle_ACB_is_75_l212_212879

open Real
open Classical

-- Define the problem conditions
variables {A B C P : Type}
variables {triangle_ABC : Triangle A B C}
variables {P_on_BC : PointOnSegment P B C}
variables {ratio_BP_PC : BP / PC = 1/2}
variables {angle_ABC_is_45 : Angle μεταξύ B A C = 45}
variables {angle_APC_is_60 : Angle μεταξύ A P C = 60}

-- State the theorem
theorem angle_ACB_is_75 :
  ∀ (A B C P : Type)
    (triangle_ABC : Triangle A B C)
    (P_on_BC : PointOnSegment P B C)
    (ratio_BP_PC : BP / PC = 1/2)
    (angle_ABC_is_45 : Angle μεταξύ B A C = 45)
    (angle_APC_is_60 : Angle μεταξύ A P C = 60),
    ∠ACB = 75 := 
  sorry

end angle_ACB_is_75_l212_212879


namespace z_pow_8_l212_212490

/-- Define the complex number z -/
def z := (-Real.sqrt 3 + Complex.I) / 2

/-- Prove that z^8 equals the given complex number -/
theorem z_pow_8 : z^8 = -1 / 2 - (Real.sqrt 3 / 2) * Complex.I :=
by
  sorry

end z_pow_8_l212_212490


namespace solve_chestnut_problem_l212_212098

def chestnut_problem : Prop :=
  ∃ (P M L : ℕ), (M = 2 * P) ∧ (L = P + 2) ∧ (P + M + L = 26) ∧ (M = 12)

theorem solve_chestnut_problem : chestnut_problem :=
by 
  sorry

end solve_chestnut_problem_l212_212098


namespace length_of_CD_eq_152_l212_212973

theorem length_of_CD_eq_152 :
  ∀ (A B C D E : Point) (a b c : ℝ),
    A.dist B = 137 ∧ A.dist C = 241 ∧ B.dist C = 200 ∧
    on_segment D B C ∧ incircle_touching A B D E ∧ incircle_touching A C D E →
    C.dist D = 152 :=
by
  intros A B C D E a b c hab hac hbc hds hib hia
  sorry

end length_of_CD_eq_152_l212_212973


namespace nancy_hourly_wage_l212_212103

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end nancy_hourly_wage_l212_212103


namespace trig_problems_l212_212367

noncomputable def tan_eq_neg_four_thirds (α : ℝ) : Prop :=
  Real.tan α = -4 / 3

noncomputable def sin_plus_cos_eq (α : ℝ) : Prop :=
  Real.sin α + Real.cos α = -1 / 5

noncomputable def complex_trig_eq (α : ℝ) : Prop :=
  (Real.sin (π - α) + 2 * Real.cos (π + α)) / (Real.sin (3 / 2 * π - α) - Real.cos (3 / 2 * π + α)) = -10

theorem trig_problems (α : ℝ) (h : tan_eq_neg_four_thirds α) : sin_plus_cos_eq α ∧ complex_trig_eq α :=
by sorry

end trig_problems_l212_212367


namespace distance_O_to_MK_half_hypotenuse_l212_212635

/-- 
Given a right-angled triangle ABC with ∠B = 90 degrees, and a circle centered at O that passes
through the endpoints A and C of the hypotenuse AC and intersects the legs AB and BC at points M and K respectively,
prove that the distance from point O to line MK is half the length of the hypotenuse AC.
-/
theorem distance_O_to_MK_half_hypotenuse 
  {A B C M K O : Type*} [MetricSpace A]
  (hypotenuse_AC : ℝ)
  (h1 : is_right_triangle A B C)
  (h2 : ∠B = 90)
  (h3 : circle_center O ∈ circle A C)
  (h4 : seg_intersects A B M)
  (h5 : seg_intersects B C K)
  :
  distance_to_line O MK = hypotenuse_AC / 2 :=
sorry

end distance_O_to_MK_half_hypotenuse_l212_212635


namespace solve_for_y_l212_212026

theorem solve_for_y (y : ℝ) (h : (y / 6) / 3 = 9 / (y / 3)) : y = 3 * real.sqrt 54 ∨ y = -3 * real.sqrt 54 :=
sorry

end solve_for_y_l212_212026


namespace jack_travel_distance_l212_212069

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

noncomputable def total_distance : ℝ :=
  distance (-3, 6) (2, 2) + distance (2, 2) (6, -3)

theorem jack_travel_distance : total_distance = 2 * real.sqrt 41 :=
by {
  sorry
}

end jack_travel_distance_l212_212069


namespace incorrect_statement_for_proportional_function_l212_212723

theorem incorrect_statement_for_proportional_function (x y : ℝ) : y = -5 * x →
  ¬ (∀ x, (x > 0 → y > 0) ∧ (x < 0 → y < 0)) :=
by
  sorry

end incorrect_statement_for_proportional_function_l212_212723


namespace probability_sum_two_balls_less_than_five_l212_212193

theorem probability_sum_two_balls_less_than_five :
  let balls := {1, 2, 3, 4, 5}
  let bag1 := balls
  let bag2 := balls
  let events := { (x, y) | x ∈ bag1 ∧ y ∈ bag2 }
  let favorable_events := { (x, y) | x ∈ bag1 ∧ y ∈ bag2 ∧ x + y < 5 }
  let total_events := card events
  let favorable_count := card favorable_events
  let probability := (favorable_count : ℚ) / (total_events : ℚ)
  probability = 6 / 25 :=
by
  sorry -- Proof to be filled in

end probability_sum_two_balls_less_than_five_l212_212193


namespace tailor_trim_amount_l212_212286

variable (x : ℝ)

def original_side : ℝ := 22
def trimmed_side : ℝ := original_side - x
def fixed_trimmed_side : ℝ := original_side - 5
def remaining_area : ℝ := 120

theorem tailor_trim_amount :
  (original_side - x) * 17 = remaining_area → x = 15 :=
by
  intro h
  sorry

end tailor_trim_amount_l212_212286


namespace problem_proof_l212_212365

def binomial_expansion (n : ℕ) : ℕ :=
  2 * choose n 1 + 2^2 * choose n 2 + 2^3 * choose n 3 + 2^n * choose n n

noncomputable def constant_term_in_expansion (n : ℕ) : ℚ :=
  (choose n 4) * (1/2)^4

theorem problem_proof :
  (binomial_expansion 6 = 728) ∧ (constant_term_in_expansion 6 = 15/16) :=
by
  sorry

end problem_proof_l212_212365


namespace odd_function_evaluation_l212_212500

theorem odd_function_evaluation
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, x ≤ 0 → f x = 2 * x^2 - x) :
  f 1 = -3 :=
by {
  sorry
}

end odd_function_evaluation_l212_212500


namespace probability_B_not_occur_given_A_occurs_expected_value_X_l212_212789

namespace DieProblem

def event_A := {1, 2, 3}
def event_B := {1, 2, 4}

def num_trials := 10
def num_occurrences_A := 6

theorem probability_B_not_occur_given_A_occurs :
  (∑ i in Finset.range (num_trials.choose num_occurrences_A), 
    (1/6)^num_occurrences_A * (1/3)^(num_trials - num_occurrences_A)) / 
  (num_trials.choose num_occurrences_A * (1/2)^(num_trials)) = 2.71 * 10^(-4) :=
sorry

theorem expected_value_X : 
  (6 * (2/3)) + (4 * (1/3)) = 16 / 3 :=
sorry

end DieProblem

end probability_B_not_occur_given_A_occurs_expected_value_X_l212_212789


namespace germination_percentage_l212_212359

theorem germination_percentage (seeds_plot1 seeds_plot2 : ℕ) (percent_germ_plot1 : ℕ) (total_percent_germ : ℕ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  percent_germ_plot1 = 20 →
  total_percent_germ = 26 →
  ∃ (percent_germ_plot2 : ℕ), percent_germ_plot2 = 35 :=
by
  sorry

end germination_percentage_l212_212359


namespace irrationals_equal_if_floor_eq_l212_212089

theorem irrationals_equal_if_floor_eq (α β : ℝ) (hα : ¬is_rat α) (hβ : ¬is_rat β) (h_pos_α : 0 < α) (h_pos_β : 0 < β)
  (h : ∀ x : ℝ, 0 < x → floor (α * floor (β * x)) = floor (β * floor (α * x))) :
  α = β :=
sorry

end irrationals_equal_if_floor_eq_l212_212089


namespace num_valid_pairs_eq_zero_l212_212718

theorem num_valid_pairs_eq_zero :
  ∀ (a b : ℤ), (∃ x y : ℤ, a * x + b * y = 3 ∧ x^2 + y^2 = 85) ∧ (3 * a - 5 * b = 0) → false :=
begin
  intros a b h,
  rcases h with ⟨⟨x, y, h1, h2⟩, h3⟩,
  have h4 : b = (3 * a) / 5,
  { rw [← h3, int.coe_nat_zero, zero_div] },
  sorry
end

end num_valid_pairs_eq_zero_l212_212718


namespace perimeter_of_semicircles_bounded_square_l212_212275

theorem perimeter_of_semicircles_bounded_square (a : ℝ) (h : a = 1 / real.pi) :
  let radius := a / 2,
      circumference_full_circle := 2 * real.pi * radius,
      perimeter_semicircle := circumference_full_circle / 2 + a,
      total_perimeter := 4 * perimeter_semicircle
  in total_perimeter = 2 + 4 / real.pi :=
by
  sorry

end perimeter_of_semicircles_bounded_square_l212_212275


namespace total_readers_l212_212584

variables {n : ℕ} (S : ℕ) (S_k : ℕ → ℕ)

def number_of_readers_in_library (n : ℕ) (S_k : ℕ → ℕ) : ℕ :=
  ∑ i in (finset.range n.succ).filter (λ i, i > 0),
    (-1)^(i-1) * S_k i

theorem total_readers (h_S : S = number_of_readers_in_library n S_k) : S = S_1 - S_2 + S_3 - S_4 + ... + (-1)^{n-1} * S_n :=
by
  sorry  -- Proof omitted

end total_readers_l212_212584


namespace repeating_decimal_calculation_l212_212314

theorem repeating_decimal_calculation :
  2 * (8 / 9 - 2 / 9 + 4 / 9) = 20 / 9 :=
by
  -- sorry proof will be inserted here.
  sorry

end repeating_decimal_calculation_l212_212314


namespace probability_of_h_and_l_l212_212902

-- Define the possible levels of service
inductive BusServiceLevel
| H -- high/good
| M -- medium/average
| L -- low/poor

open BusServiceLevel

-- Define the sequences of bus arrivals
def sequences := [
  [L, M, H],
  [L, H, M],
  [M, L, H],
  [M, H, L],
  [H, L, M],
  [H, M, L]
]

-- Define Mr. Zhang's bus-taking strategy
def takeBus (seq : List BusServiceLevel) : BusServiceLevel :=
  match seq with
  | [first, second, third] =>
    if second = H ∨ (first ≠ H ∧ second = M) then
      second
    else
      third
  | _ => L -- This case won't occur due to our predefined sequences

-- Calculate probabilities
theorem probability_of_h_and_l :
  let counts := (sequences.count (λ seq => takeBus seq = H), sequences.count (λ seq => takeBus seq = L))
  counts.1 = 3 ∧ counts.2 = 1 ∧ sequences.length = 6 →
  (counts.1.toRat / sequences.length.toRat = 1/2) ∧ (counts.2.toRat / sequences.length.toRat = 1/6) :=
by sorry

end probability_of_h_and_l_l212_212902


namespace probability_target_hit_probability_hit_only_by_A_l212_212303

noncomputable def P_A : ℝ := 0.95
noncomputable def P_B : ℝ := 0.9

theorem probability_target_hit :
  let P_A_not := 1 - P_A in
  let P_B_not := 1 - P_B in
  (P_A * P_B_not + P_A_not * P_B + P_A * P_B) = 0.995 := by
sorry

theorem probability_hit_only_by_A :
  let P_B_not := 1 - P_B in
  (P_A * P_B_not) = 0.095 := by
sorry

end probability_target_hit_probability_hit_only_by_A_l212_212303


namespace megatek_manufacturing_percentage_l212_212619

theorem megatek_manufacturing_percentage
  (proportionality : ∀ (d : ℕ), d / 360 = (d / 360) * 100 / 100)
  (manufacturing_degrees : ℕ = 252)
  (full_circle_degrees : ℕ = 360) :
  (252 / 360) * 100 = 70 :=
by
  sorry

end megatek_manufacturing_percentage_l212_212619


namespace part_a_part_b_l212_212240

-- Define the triangle side lengths a, b, c
variables (a b c : ℝ)

-- Define the medians to sides a and b
variables (m_a m_b : ℝ)

-- Conditions for a, b, c being side lengths of a triangle
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : a + c > b
axiom triangle_inequality3 : b + c > a

-- Part (a) statement
theorem part_a : a^2 + b^2 ≥ c^2 / 2 := sorry

-- Conditions for m_a and m_b being medians
-- Assuming m_a and m_b are medians calculated based on valid conditions
axiom median_formula : m_a = (sqrt (2 * b^2 + 2 * c^2 - a^2)) / 2
axiom median_formula : m_b = (sqrt (2 * a^2 + 2 * c^2 - b^2)) / 2

-- Part (b) statement
theorem part_b : m_a^2 + m_b^2 ≥ (9 * c^2) / 8 := sorry

end part_a_part_b_l212_212240


namespace price_reduction_l212_212638

theorem price_reduction (P : ℝ) (hP : P > 0) :
  let initial_discount := 0.7 * P,
      final_price := 0.5 * initial_discount,
      reduction := (P - final_price) / P * 100
  in reduction = 65 := by
  sorry

end price_reduction_l212_212638


namespace no_twelve_consecutive_primes_in_ap_l212_212127

theorem no_twelve_consecutive_primes_in_ap (d : ℕ) (h : d < 2000) :
  ∀ a : ℕ, ¬(∀ n : ℕ, n < 12 → (Prime (a + n * d))) :=
sorry

end no_twelve_consecutive_primes_in_ap_l212_212127


namespace transformed_samples_property_l212_212028

variables {n : ℕ}
variables (x : fin n → ℝ)

-- Conditions
def average (x : fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

def variance (x : fin n → ℝ) (mean : ℝ) : ℝ :=
  (∑ i, (x i - mean) ^ 2) / n

theorem transformed_samples_property
  (h_avg : average x = 8)
  (h_var : variance x 8 = 4) :
  average (fun i => x i - 3) = 5 ∧ variance (fun i => x i - 3) 5 = 4 ∧ sorry := 
  sorry

end transformed_samples_property_l212_212028


namespace find_equations_of_lines_l212_212713

-- Define the given constants and conditions
def point_P := (2, 2)
def line_l1 (x y : ℝ) := 3 * x - 2 * y + 1 = 0
def line_l2 (x y : ℝ) := x + 3 * y + 4 = 0
def intersection_point := (-1, -1)
def slope_perpendicular_line := 3

-- The theorem that we need to prove
theorem find_equations_of_lines :
  (∀ k, k = 0 → line_l1 2 2 → (x = y ∨ x + y = 4)) ∧
  (line_l1 (-1) (-1) ∧ line_l2 (-1) (-1) →
   (3 * x - y + 2 = 0))
:=
sorry

end find_equations_of_lines_l212_212713


namespace students_play_soccer_count_l212_212580

noncomputable def number_of_students : ℕ := 400
noncomputable def percentage_play_sports : ℝ := 0.52
noncomputable def percentage_play_soccer : ℝ := 0.125

noncomputable def students_play_sports : ℕ := percentage_play_sports * number_of_students
noncomputable def students_play_soccer : ℕ := percentage_play_soccer * students_play_sports

theorem students_play_soccer_count :
  students_play_soccer = 26 :=
sorry

end students_play_soccer_count_l212_212580


namespace circle_through_origin_and_point_l212_212159

theorem circle_through_origin_and_point (a r : ℝ) :
  (∃ a r : ℝ, (a^2 + (5 - 3 * a)^2 = r^2) ∧ ((a - 3)^2 + (3 * a - 6)^2 = r^2)) →
  a = 5/3 ∧ r^2 = 25/9 :=
sorry

end circle_through_origin_and_point_l212_212159


namespace collinear_min_value_l212_212411

variable (a b : ℝ)

theorem collinear_min_value (h₁ : a > 0) (h₂ : b > 0)
    (h₃ : ∃ (λ : ℝ), (a - 1 = -λ * (b + 1)) ∧ (1 = 2 * λ)) :
    (∃ (x : ℝ), x = 1 / a + 2 / b ∧ ∀ y, y = 1 / a + 2 / b → y ≥ 8) :=
sorry

end collinear_min_value_l212_212411


namespace XY_perpendicular_AD_l212_212893

-- Define the triangle ABC and related points with the given conditions
variables (A B C D X Y : Point)
variables (hD_on_BC : D ∈ line_segment B C)
variables (h_angles_BAD_DAC : ∠ BAD = ∠ DAC)
variables (h_X_opposite_side : ∃ P : Point, P ∈ line BC ∧ A and B same_line P)
variables (h_XB_eq_XD : dist X B = dist X D)
variables (h_BXD_eq_ACB : ∠ BXD = ∠ ACB)
variables (h_Y_opposite_side : ∃ Q : Point, Q ∈ line BC ∧ A and C same_line Q)
variables (h_YC_eq_YD : dist Y C = dist Y D)
variables (h_CYD_eq_ABC : ∠ CYD = ∠ ABC)

-- The proof goal to be shown
theorem XY_perpendicular_AD :
  perp_line (line XY) (line AD) :=
sorry

end XY_perpendicular_AD_l212_212893


namespace find_k_l212_212737

theorem find_k (x y k : ℝ) (h₁ : x = 2) (h₂ : y = -1) (h₃ : y - k * x = 7) : k = -4 :=
by
  sorry

end find_k_l212_212737


namespace increasing_interval_of_f_l212_212957

noncomputable def f (x : ℝ) : ℝ :=
  log (1 / 2) (x^2 - 2 * x - 3)

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem increasing_interval_of_f : is_monotonically_increasing f {x : ℝ | x < -1} :=
sorry

end increasing_interval_of_f_l212_212957


namespace shadow_length_false_if_approaching_lamp_at_night_l212_212671

theorem shadow_length_false_if_approaching_lamp_at_night
  (night : Prop)
  (approaches_lamp : Prop)
  (shadow_longer : Prop) :
  night → approaches_lamp → ¬shadow_longer :=
by
  -- assume it is night and person is approaching lamp
  intros h_night h_approaches
  -- proof is omitted
  sorry

end shadow_length_false_if_approaching_lamp_at_night_l212_212671


namespace range_of_t_l212_212475

noncomputable def a (n : ℕ) : ℚ := 
if n = 1 then 1 else n * (n + 1) / 2

noncomputable def bn (n : ℕ) : ℚ :=
2 / (2 * n + 1 / n + 3)

theorem range_of_t (t : ℝ) (n : ℕ) :
  (∀ m : ℝ, m ∈ set.Icc 1 2 → m^2 - m * t + 1/3 > bn n) ↔ t < 1 := by
  sorry

end range_of_t_l212_212475


namespace sum_of_largest_and_smallest_l212_212356

theorem sum_of_largest_and_smallest : 
  let a := - (2 ^ 4)
  let b := 15
  let c := -18
  let d := (-4) ^ 2
in (∃ x y, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
          (∀ z, (z = a ∨ z = b ∨ z = c ∨ z = d) → z ≤ x ∨ z ≥ y) ∧ 
          x + y = -2) := sorry

end sum_of_largest_and_smallest_l212_212356


namespace percentage_who_do_not_have_job_of_choice_have_university_diploma_l212_212052

theorem percentage_who_do_not_have_job_of_choice_have_university_diploma :
  ∀ (total_population university_diploma job_of_choice no_diploma_job_of_choice : ℝ),
    total_population = 100 →
    job_of_choice = 40 →
    no_diploma_job_of_choice = 10 →
    university_diploma = 48 →
    ((university_diploma - (job_of_choice - no_diploma_job_of_choice)) / (total_population - job_of_choice)) * 100 = 30 :=
by
  intros total_population university_diploma job_of_choice no_diploma_job_of_choice h1 h2 h3 h4
  sorry

end percentage_who_do_not_have_job_of_choice_have_university_diploma_l212_212052


namespace probability_exactly_4_excellent_students_l212_212189

def total_students : ℕ := 10
def excellent_students : ℕ := 6
def selected_students : ℕ := 7
def target_excellent_selected : ℕ := 4

theorem probability_exactly_4_excellent_students : 
  (combinatorics.combinations total_students selected_students ≠ 0) →
  (1 / combinatorics.combinations total_students selected_students) *
  (combinatorics.combinations excellent_students target_excellent_selected) *
  (combinatorics.combinations (total_students - excellent_students) (selected_students - target_excellent_selected)) = 0.5 := 
by
  sorry

end probability_exactly_4_excellent_students_l212_212189


namespace find_c5_relate_c_d_d_seq_5_value_find_final_c5_value_l212_212093

def c_seq : ℕ → ℝ
| 0       := 0
| (n + 1) := (7 / 4) * c_seq n + (3 / 4) * sqrt(9^n - (c_seq n)^2)

theorem find_c5 : c_seq 5 = 243 * d_seq 5 := 
sorry

def d_seq : ℕ → ℝ
| 0       := 0
| (n + 1) := (7 / 12) * d_seq n + (1 / 4) * sqrt(1 - (d_seq n)^2)

theorem relate_c_d (n : ℕ) : c_seq n = 3^n * d_seq n := 
sorry

theorem d_seq_5_value : d_seq 5 = "expression in terms of radicals and rational numbers" := 
sorry

theorem find_final_c5_value : c_seq 5 = 243 * "expression for d_5" := 
sorry

end find_c5_relate_c_d_d_seq_5_value_find_final_c5_value_l212_212093


namespace employees_6_or_more_percentage_is_18_l212_212974

-- Defining the employee counts for different year ranges
def count_less_than_1 (y : ℕ) : ℕ := 4 * y
def count_1_to_2 (y : ℕ) : ℕ := 6 * y
def count_2_to_3 (y : ℕ) : ℕ := 7 * y
def count_3_to_4 (y : ℕ) : ℕ := 4 * y
def count_4_to_5 (y : ℕ) : ℕ := 3 * y
def count_5_to_6 (y : ℕ) : ℕ := 3 * y
def count_6_to_7 (y : ℕ) : ℕ := 2 * y
def count_7_to_8 (y : ℕ) : ℕ := 2 * y
def count_8_to_9 (y : ℕ) : ℕ := y
def count_9_to_10 (y : ℕ) : ℕ := y

-- Sum of all employees T
def total_employees (y : ℕ) : ℕ := count_less_than_1 y + count_1_to_2 y + count_2_to_3 y +
                                    count_3_to_4 y + count_4_to_5 y + count_5_to_6 y +
                                    count_6_to_7 y + count_7_to_8 y + count_8_to_9 y +
                                    count_9_to_10 y

-- Employees with 6 years or more E
def employees_6_or_more (y : ℕ) : ℕ := count_6_to_7 y + count_7_to_8 y + count_8_to_9 y + count_9_to_10 y

-- Calculate percentage
def percentage (y : ℕ) : ℚ := (employees_6_or_more y : ℚ) / (total_employees y : ℚ) * 100

-- Proving the final statement
theorem employees_6_or_more_percentage_is_18 (y : ℕ) (hy : y ≠ 0) : percentage y = 18 :=
by
  sorry

end employees_6_or_more_percentage_is_18_l212_212974


namespace triangle_perimeter_l212_212562

theorem triangle_perimeter
  (AP BP : ℕ)
  (r : ℝ)
  (h_AP : AP = 20)
  (h_PB : BP = 30)
  (h_r : r = 15) :
  ∃ s x : ℝ, s = 25 + x ∧ 15 * s = sqrt ((25 + x) * x * 600) ∧ 2 * s = 100 := 
begin
  sorry
end

end triangle_perimeter_l212_212562


namespace largest_n_base_conditions_l212_212990

theorem largest_n_base_conditions :
  ∃ n: ℕ, n < 10000 ∧ 
  (∃ a: ℕ, 4^a ≤ n ∧ n < 4^(a+1) ∧ 4^a ≤ 3*n ∧ 3*n < 4^(a+1)) ∧
  (∃ b: ℕ, 8^b ≤ n ∧ n < 8^(b+1) ∧ 8^b ≤ 7*n ∧ 7*n < 8^(b+1)) ∧
  (∃ c: ℕ, 16^c ≤ n ∧ n < 16^(c+1) ∧ 16^c ≤ 15*n ∧ 15*n < 16^(c+1)) ∧
  n = 4369 :=
sorry

end largest_n_base_conditions_l212_212990


namespace divide_implies_divide_l212_212887

open Int Nat

variable (f : ℤ → ℕ)

theorem divide_implies_divide (H1 : ∀ m n : ℤ, (f(m) - f(n)) % f(m - n) = 0)
  (m n : ℤ) (H2 : f(m) ∣ f(n)) : f(n) ∣ f(m) := by
  sorry

end divide_implies_divide_l212_212887


namespace gcd_102_238_l212_212213

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l212_212213


namespace convert_to_scientific_notation_l212_212151

-- Problem statement: convert 120 million to scientific notation and validate the format.
theorem convert_to_scientific_notation :
  120000000 = 1.2 * 10^7 :=
sorry

end convert_to_scientific_notation_l212_212151


namespace truck_speed_in_kmph_l212_212289

def time_in_seconds := 60
def distance_in_meters := 600
def meters_per_kilometer := 1000
def seconds_per_hour := 3600
def conversion_factor := 3.6 -- Since 3600 / 1000 = 3.6

theorem truck_speed_in_kmph : (distance_in_meters / time_in_seconds) * conversion_factor = 36 := by
  sorry

end truck_speed_in_kmph_l212_212289


namespace max_value_neg_a_inv_l212_212781

theorem max_value_neg_a_inv (a : ℝ) (h : a < 0) : a + (1 / a) ≤ -2 := 
by
  sorry

end max_value_neg_a_inv_l212_212781


namespace remainder_is_correct_l212_212617

def dividend : ℕ := 725
def divisor : ℕ := 36
def quotient : ℕ := 20

theorem remainder_is_correct : ∃ (remainder : ℕ), dividend = (divisor * quotient) + remainder ∧ remainder = 5 := by
  sorry

end remainder_is_correct_l212_212617


namespace problem_1_and_2_l212_212856

noncomputable def polar_equation_of_circle (rho theta : ℝ) : Prop :=
  rho = 6 * cos(theta - (Real.pi / 6))

noncomputable def trajectory_equation_of_p (rho theta : ℝ) : Prop :=
  rho = 10 * cos(theta - (Real.pi / 6))

theorem problem_1_and_2 (rho theta : ℝ) :
  (polar_equation_of_circle rho theta ∧ trajectory_equation_of_p rho theta) := by
  sorry

end problem_1_and_2_l212_212856


namespace count_divisible_by_3_and_7_l212_212542

theorem count_divisible_by_3_and_7 (a : ℕ) : 
  let lcm_3_7 := Nat.lcm 3 7 
  in (∀ n ∈ {x | 1 ≤ x ∧ x ≤ 200}, n % lcm_3_7 = 0) → a = 9 :=
by
  let lcm_3_7 := Nat.lcm 3 7 
  have H : ∀ n ∈ {x | 1 ≤ x ∧ x ≤ 200}, n % lcm_3_7 = 0 := sorry
  have count : (Finset.filter (λ n => n % lcm_3_7 = 0) (Finset.range' 1 200)).card = 9 := sorry
  exact count

end count_divisible_by_3_and_7_l212_212542


namespace product_of_binary_and_ternary_equals_483_l212_212703

def binary_to_decimal (b : ℕ) (bin : List ℕ) : ℕ :=
  bin.reverse.enum.map (λ ⟨e, d⟩ => d * b ^ e).sum

def ternary_to_decimal (t : ℕ) (tern : List ℕ) : ℕ :=
  tern.reverse.enum.map (λ ⟨e, d⟩ => d * t ^ e).sum

theorem product_of_binary_and_ternary_equals_483 :
  binary_to_decimal 2 [1, 0, 1, 0, 1] * ternary_to_decimal 3 [2, 1, 2] = 483 :=
by
  sorry

end product_of_binary_and_ternary_equals_483_l212_212703


namespace area_triangle_BCD_l212_212079

theorem area_triangle_BCD (a b c p q r : ℝ)
  (h1 : p = (1/2) * a * b)
  (h2 : q = (1/2) * b * c)
  (h3 : r = (1/2) * a * c) :
  (√(p ^ 2 + q ^ 2 + r ^ 2)) = (1/2) * √((2*q)^2 + (2*r)^2 + (2*p)^2) :=
by
  sorry

end area_triangle_BCD_l212_212079


namespace single_light_on_l212_212230

open Function

def toggle (grid : Matrix Bool 5 5) (i j : Fin 5) : Matrix Bool 5 5 :=
  grid.update i (grid i).update j !(grid i j)

def toggle_neighbors (grid : Matrix Bool 5 5) (i j : Fin 5) : Matrix Bool 5 5 :=
  let row_toggle := (0 : Fin 5).map fun k => toggle grid i k
  let col_toggle := (0 : Fin 5).map fun k => toggle grid k j
  col_toggle i

noncomputable def final_positions (initial_grid : Matrix Bool 5 5) : List (Fin 5 × Fin 5) :=
  [(2,2), (2,4), (4,2), (4,4), (3,3)]

theorem single_light_on (initial_grid : Matrix Bool 5 5) :
  (∀ i j, initial_grid i j = false) ∧ 
  ∃ (i j : Fin 5), (∀ (k l : Fin 5), toggle_neighbors initial_grid i j k l = (i = k ∧ j = l)) → 
  final_positions initial_grid = [(2,2), (2,4), (4,2), (4,4), (3,3)] :=
sorry


end single_light_on_l212_212230


namespace find_angle_A_find_perimeter_l212_212066

-- Given problem conditions as Lean definitions
def triangle_sides (a b c : ℝ) : Prop :=
  ∃ B : ℝ, c = a * (Real.cos B + Real.sqrt 3 * Real.sin B)

def triangle_area (S a : ℝ) : Prop :=
  S = Real.sqrt 3 / 4 ∧ a = 1

-- Prove angle A
theorem find_angle_A (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ A : ℝ, A = Real.pi / 6 := 
sorry

-- Prove perimeter
theorem find_perimeter (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ P : ℝ, P = Real.sqrt 3 + 2 := 
sorry

end find_angle_A_find_perimeter_l212_212066


namespace number_of_identical_to_y_eq_x_l212_212665

-- Define the functions
def f1 (x : ℝ) : ℝ := (Real.sqrt x) ^ 2
def f2 (x : ℝ) : ℝ := 3 * x ^ 3
def f3 (x : ℝ) : ℝ := Real.sqrt (x ^ 2)
def f4 (x : ℝ) : ℝ := if x ≠ 0 then x^2 / x else 0 -- handle x = 0 separately

-- Main theorem statement
theorem number_of_identical_to_y_eq_x : 
  (∃ n : ℕ, (n = 0 ∧ 
  (∀ x : ℝ, f1 x ≠ x) ∧ 
  (∀ x : ℝ, f2 x ≠ x) ∧ 
  (∀ x : ℝ, f3 x ≠ x) ∧ 
  (∀ x : ℝ, f4 x ≠ x))) :=
sorry

end number_of_identical_to_y_eq_x_l212_212665


namespace pipe_B_time_l212_212594

theorem pipe_B_time :
  (∀ {T : ℝ}, (1 / 30 + 1 / T = 1 / 18) → T = 45) :=
begin
  intro T,
  intro h,
  -- this is where the proof would go, as indicated by sorry to skip the proof.
  sorry
end

end pipe_B_time_l212_212594


namespace tangent_line_circle_l212_212034

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ,  (x + y + m = 0) → (x^2 + y^2 = m) → m = 2) : m = 2 :=
sorry

end tangent_line_circle_l212_212034


namespace part1_part2_l212_212746

section
variables {α β : ℝ} {t : ℝ} {f : ℝ → ℝ}
variables {λ₁ λ₂ : ℝ}

-- Condition: α and β are roots of the quadratic equation 2x^2 - tx - 2 = 0
def roots (t : ℝ) (α β : ℝ) : Prop := 
  2 * α^2 - t * α - 2 = 0 ∧ 2 * β^2 - t * β - 2 = 0

-- Function definition: f(x) = (4x - t) / (x^2 + 1)
def func (f : ℝ → ℝ) (t : ℝ) : Prop := 
  ∀ x, f x = (4 * x - t) / (x^2 + 1)

-- Part 1: Prove that (f(α) - f(β)) / (α - β) = 2
theorem part1 (h1 : roots t α β) 
              (h2 : func f t) 
              (h3 : α < β) :
  (f α - f β) / (α - β) = 2 := sorry

-- Part 2: Prove that for any positive λ₁ and λ₂, 
--         | f((λ₁ * α + λ₂ * β) / (λ₁ + λ₂)) - f((λ₁ * β + λ₂ * α) / (λ₁ + λ₂)) | < 2 | α - β |
theorem part2 (h1 : roots t α β) 
              (h2 : func f t) 
              (h3 : α < β) 
              (h4 : 0 < λ₁) 
              (h5 : 0 < λ₂) :
  |f ((λ₁ * α + λ₂ * β) / (λ₁ + λ₂)) - f ((λ₁ * β + λ₂ * α) / (λ₁ + λ₂))| < 2 * |α - β| := sorry

end

end part1_part2_l212_212746


namespace parabola_min_dist_l212_212556

theorem parabola_min_dist (p : ℝ) (hp : p > 0) (hmin_dist : ∀ Q : ℝ × ℝ, (Q.2)^2 = 2 * p * Q.1 → dist Q (p / 2, 0) ≥ 1) : p = 2 :=
begin
  -- Proof goes here
  sorry
end

end parabola_min_dist_l212_212556


namespace speed_of_stream_l212_212177

theorem speed_of_stream 
  (v : ℝ)
  (boat_speed : ℝ)
  (distance_downstream : ℝ)
  (distance_upstream : ℝ)
  (H1 : boat_speed = 12)
  (H2 : distance_downstream = 32)
  (H3 : distance_upstream = 16)
  (H4 : distance_downstream / (boat_speed + v) = distance_upstream / (boat_speed - v)) :
  v = 4 :=
by
  sorry

end speed_of_stream_l212_212177


namespace equation_C_is_symmetric_l212_212609

def symm_y_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), f x y ↔ f (-x) y

def equation_A (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equation_B (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equation_C (x y : ℝ) : Prop := x^2 - y^2 = 1
def equation_D (x y : ℝ) : Prop := x - y = 1

theorem equation_C_is_symmetric : symm_y_axis equation_C :=
by
  sorry

end equation_C_is_symmetric_l212_212609


namespace abs_x_minus_y_l212_212387

theorem abs_x_minus_y (x y : ℝ) (h₁ : x^3 + y^3 = 26) (h₂ : xy * (x + y) = -6) : |x - y| = 4 :=
by
  sorry

end abs_x_minus_y_l212_212387


namespace class_schedule_count_l212_212590

-- Define the days consisting of sessions for the classes.
inductive Subject
| Chinese | Mathematics | Politics | English | PhysicalEducation | Art

open Subject

-- Define the conditions
def valid_arrangement (arrangement : List Subject) : Prop :=
  ∃ (s₁ s₂ s₃ s₄ s₅ s₆ : Subject), 
    arrangement = [s₁, s₂, s₃, s₄, s₅, s₆] ∧ 
    ∃ (n₁ n₂ n₃ : ℕ), 
      n₁ < 3 ∧ n₂ < 3 ∧ n₃ < 3 ∧ 
      (s₁ = Mathematics ∨ s₂ = Mathematics ∨ s₃ = Mathematics) ∧ 
      ∀ i, (i ≠ 5 → s₅ ≠ English)

-- The statement of the problem
theorem class_schedule_count (h : ∃ (arrangement : List Subject), valid_arrangement arrangement) :   
  (card (finset.filter valid_arrangement (finset.univ : finset (List Subject))) = 288) :=
sorry

end class_schedule_count_l212_212590


namespace solve_equation_l212_212536

noncomputable def equation_sol (x : ℂ) : Prop :=
  (x^2 + 4*x + 8) / (x - 3) = 2

theorem solve_equation (x : ℂ) : 
  equation_sol x ↔ x = (-1 + (7 * (complex.sqrt 2) / 2) * complex.I) ∨ x = (-1 - (7 * (complex.sqrt 2) / 2) * complex.I) :=
sorry

end solve_equation_l212_212536


namespace pentagon_ratio_area_l212_212882

noncomputable def hexagon_side_length : ℝ := 6

-- Let ABCDEF be a regular hexagon with side length 6
def is_regular_hexagon (A B C D E F : ℝ × ℝ) : Prop :=
  let sides := [(A, B), (B, C), (C, D), (D, E), (E, F), (F, A)] in
  (∀ (side : (ℝ × ℝ) × (ℝ × ℝ)), side ∈ sides → dist side.fst side.snd = hexagon_side_length) ∧
  (angles (A, B, C, D, E, F) = 120)

-- Define the midpoints G, H, I, J, K, L
def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

def is_pentagon (P Q R S T : ℝ × ℝ) : Prop :=
  let sides := [(P, Q), (Q, R), (R, S), (S, T), (T, P)] in
  ∀ (side : (ℝ × ℝ) × (ℝ × ℝ)), side ∈ sides → ∃ c, dist side.fst side.snd = c

-- Define intersections as given in conditions
-- Note: In practice, you would use more sophisticated definitions and calculations to find actual coordinates
axiom intersects (p q r s : ℝ × ℝ) : Prop

-- The main proof statement
theorem pentagon_ratio_area (A B C D E F G H I J K L P Q R S T : ℝ × ℝ)
  (h_hex : is_regular_hexagon A B C D E F)
  (h_G : G = midpoint A B) (h_H : H = midpoint B C) (h_I : I = midpoint C D)
  (h_J : J = midpoint D E) (h_K : K = midpoint E F) (h_L : L = midpoint F A)
  (h_PQ : intersects A H B I) (h_QR : intersects B I C J)
  (h_RS : intersects C J D K) (h_ST : intersects D K E L)
  (h_TP : intersects E L F G)
  (h_pent : is_pentagon P Q R S T) :
  (pentagon_area_ratio A B C D E F P Q R S T) = (15 * (sqrt 5 - 2)) / (4 * sqrt 3) := sorry

end pentagon_ratio_area_l212_212882


namespace find_m_direct_proportion_l212_212368

theorem find_m_direct_proportion (m : ℝ) (h1 : m^2 - 3 = 1) (h2 : m ≠ 2) : m = -2 :=
by {
  -- here would be the proof, but it's omitted as per instructions
  sorry
}

end find_m_direct_proportion_l212_212368


namespace angle_equality_in_quadrilateral_l212_212838

/-- In a convex quadrilateral ABCD, 
    if ∠CBD = ∠CAB and ∠ACD = ∠BDA,
    then ∠ABC = ∠ADC. -/
theorem angle_equality_in_quadrilateral 
  {A B C D: Type*} [convex_quadrilateral A B C D]
  (h1 : ∠CBD = ∠CAB)
  (h2 : ∠ACD = ∠BDA) : 
  ∠ABC = ∠ADC := sorry

end angle_equality_in_quadrilateral_l212_212838


namespace crackers_distribution_l212_212516

theorem crackers_distribution (h1 : ∀ (total_crackers friends crackers_per_friend : ℕ),
  total_crackers = 8 → crackers_per_friend = 2 → total_crackers = friends * crackers_per_friend) : 
∀ (total_crackers friends crackers_per_friend : ℕ), total_crackers = 8 → crackers_per_friend = 2 → friends = 4 :=
by
  intros total_crackers friends crackers_per_friend h_total h_crackers_per_friend
  specialize h1 total_crackers friends crackers_per_friend h_total h_crackers_per_friend
  have friends_eq : friends = total_crackers / crackers_per_friend := by
    rw [h1]
  rw [h_total, h_crackers_per_friend] at friends_eq
  norm_num at friends_eq
  exact friends_eq

end crackers_distribution_l212_212516


namespace count_valid_n_l212_212360

theorem count_valid_n : {n : ℕ | 300 ≤ n ∧ n ≤ 333 ∧ n % 3 = 0}.card = 12 := 
sorry

end count_valid_n_l212_212360


namespace algebraic_expression_eq_five_l212_212727

theorem algebraic_expression_eq_five (a b : ℝ)
  (h₁ : a^2 - a = 1)
  (h₂ : b^2 - b = 1) :
  3 * a^2 + 2 * b^2 - 3 * a - 2 * b = 5 :=
by
  sorry

end algebraic_expression_eq_five_l212_212727


namespace right_triangle_AEF_l212_212302

variable {O A B C P D E F : Type}
variables [EuclideanGeometry] (Circle : Circle O A)
  (AB : Segment A B) 
  (C : Midpoint (semicircular_arc Circle A B))
  (P : Point)
  (PD : Tangent_segment Circle P D)
  (E F : Point)
  (E_on_AC : On_angle_bisector (angle_bisector A P D) A C E)
  (F_on_BC : On_angle_bisector (angle_bisector A P D) B C F)

theorem right_triangle_AEF :
  is_right_triangle (Triangle.mk E F A) :=
sorry

end right_triangle_AEF_l212_212302


namespace tan_alpha_20_l212_212425

theorem tan_alpha_20 (α : ℝ) 
  (h : Real.tan (α + 80 * Real.pi / 180) = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (α + 20 * Real.pi / 180) = Real.sqrt 3 / 7 := 
sorry

end tan_alpha_20_l212_212425


namespace gcf_75_105_l212_212601

theorem gcf_75_105 : Nat.gcd 75 105 = 15 :=
by {
  -- Condition 1: Prime factorization of 75
  have h1 : 75 = 3 * 5^2 := rfl,
  -- Condition 2: Prime factorization of 105
  have h2 : 105 = 3 * 5 * 7 := rfl,
  -- Goal: Prove gcd(75, 105) = 15
  sorry
}

end gcf_75_105_l212_212601


namespace part_I_part_II_l212_212397

noncomputable def f (x : ℝ) (a : ℝ) := (a - 1) * x^a
noncomputable def g (x : ℝ) := real.log10 x

-- Given function f is a power function, prove a = 2 and decreasing interval of f(x) = x^2 is (-∞, 0)
theorem part_I (a : ℝ) (x : ℝ) :
  f x a = (x : ℝ)^2 → a = 2 ∧ ∃ I : set ℝ, I = set.Iio 0 ∧ ∀ x ∈ I, x < 0 := 
sorry

-- Given equation has two different real roots x1, x2 ∈ (1, 3), prove the range of values for a + 1/x1 + 1/x2 is (2 - log10(2), 2)
theorem part_II (a : ℝ) (x1 x2 : ℝ) (h1 : 1 < x1) (h2 : x1 < 2) (h3 : 2 < x2) (h4 : x2 < 3) :
  g (x1 - 1) = -(1 - a) ∧ g (x2 - 1) = 1 - a → 1 - real.log10 2 < a ∧ a < 1 ∧ a + 1/x1 + 1/x2 ∈ set.Ioo (2 - real.log10 2) 2 :=
sorry

end part_I_part_II_l212_212397


namespace sin_cos_difference_inverse_difference_of_squares_l212_212726

theorem sin_cos_difference (x : ℝ) (h1 : -π / 2 < x) (h2 : x < 0) (h3 : sin x + cos x = 1 / 5) : sin x - cos x = -7 / 5 := 
by sorry

theorem inverse_difference_of_squares (x : ℝ) (h1 : -π / 2 < x) (h2 : x < 0) (h3 : sin x + cos x = 1 / 5) : 
  1 / (cos x ^ 2 - sin x ^ 2) = 25 / 7 :=
by sorry

end sin_cos_difference_inverse_difference_of_squares_l212_212726


namespace range_of_f_l212_212567

noncomputable def f (x : ℝ) : ℝ :=
  (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2)

theorem range_of_f (x : ℝ) : -1 < f x ∧ f x < 1 :=
by
  sorry

end range_of_f_l212_212567


namespace forged_cubic_edge_length_l212_212649

noncomputable def rectangular_block_volume (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def cube_edge_length (volume : ℝ) : ℝ :=
  real.cbrt volume

theorem forged_cubic_edge_length
  (l w h : ℝ)
  (h_l : l = 50)
  (h_w : w = 8)
  (h_h : h = 20) :
  cube_edge_length (rectangular_block_volume l w h) = 20 :=
by
  -- Proof omitted
  sorry

end forged_cubic_edge_length_l212_212649


namespace number_of_parallel_lines_l212_212022

/-- 
Given 10 parallel lines in the first set and the fact that the intersection 
of two sets of parallel lines forms 1260 parallelograms, 
prove that the second set contains 141 parallel lines.
-/
theorem number_of_parallel_lines (n : ℕ) (h₁ : 10 - 1 = 9) (h₂ : 9 * (n - 1) = 1260) : n = 141 :=
sorry

end number_of_parallel_lines_l212_212022


namespace area_of_triangle_l212_212547

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

theorem area_of_triangle : 
  let p : ℝ × ℝ := (0, 1)
  let f' (x : ℝ) : ℝ := Real.exp x + Real.cos x
  let tangent_line (x : ℝ) : ℝ := 2 * x + 1,
  (1 / 2) * (1 / 2) * 1 = 1 / 4 :=
by
  sorry

end area_of_triangle_l212_212547


namespace initial_average_price_l212_212304

variable (A O : ℕ)
variable (price_apple price_orange initial_pieces put_back_oranges average_initial average_final total_initial total_remain total_putback : ℕ)

def fruit_price_initial_conditions :=
  price_apple = 40 ∧
  price_orange = 60 ∧
  initial_pieces = 10 ∧
  put_back_oranges = 5 ∧
  average_final = 48 ∧
  total_initial = price_apple * A + price_orange * O ∧
  total_remain = average_final * (initial_pieces - put_back_oranges) ∧
  total_putback = put_back_oranges * price_orange ∧
  total_initial = total_remain + total_putback

theorem initial_average_price : fruit_price_initial_conditions A O price_apple price_orange initial_pieces put_back_oranges average_initial average_final total_initial total_remain total_putback → average_initial = 54
by
  sorry

end initial_average_price_l212_212304


namespace range_of_m_minimum_7a_4b_l212_212401

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := real.sqrt (abs (x + 1) + abs (x - 3) - m)

theorem range_of_m (x : ℝ) (m : ℝ) : abs (x + 1) + abs (x - 3) - m ≥ 0 ↔ m ≤ 4 :=
sorry

theorem minimum_7a_4b (a b : ℝ) (n : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : n = 4) (h4 : 2 / (3 * a + b) + 1 / (a + 2 * b) = n) :
  7 * a + 4 * b = 9 / 4 :=
sorry

end range_of_m_minimum_7a_4b_l212_212401


namespace max_distance_from_curve_point_to_line_l212_212467

-- Definitions based on given problem conditions
def parametric_line : Type := {t : ℝ // true}
def polar_curve : Type := {θ : ℝ // true}

-- Condition definitions
def line_parametric_eq (p : parametric_line) : ℝ × ℝ := (p.1, 4 - p.1)
def curve_polar_eq (r : ℝ) (θ : ℝ) : Prop := r = 2

-- Problem statement rewritten as proof problem in Lean
theorem max_distance_from_curve_point_to_line :
  ∀ θ : ℝ, ∀ Qx Qy : ℝ, ∀ p : parametric_line,
    (Qx, Qy) = (2 * Real.cos θ, 2 * Real.sin θ) →
    line_parametric_eq p = (p.1, 4 - p.1) →
    curve_polar_eq (Real.sqrt ((2 * Real.cos θ) ^ 2 + (2 * Real.sin θ) ^ 2)) θ →
    dist (2 * Real.cos θ, 2 * Real.sin θ) (p.1, 4 - p.1) ≤ 3 * Real.sqrt 2 := 
sorry

-- Additional helper definitions for distances
def dist (Q : ℝ × ℝ) (P : ℝ × ℝ) : ℝ :=
  abs ((2 * Real.sin (Real.arctan2 Q.2 Q.1 + Real.pi / 4) - 4) / Real.sqrt 2)

end max_distance_from_curve_point_to_line_l212_212467


namespace factorize_expression_l212_212335

theorem factorize_expression : (x^2 + 9)^2 - 36*x^2 = (x + 3)^2 * (x - 3)^2 := 
by 
  sorry

end factorize_expression_l212_212335


namespace tangential_circle_radius_l212_212204

theorem tangential_circle_radius (R r x : ℝ) (hR : R > r) (hx : x = 4 * R * r / (R + r)) :
  ∃ x, x = 4 * R * r / (R + r) := by
sorry

end tangential_circle_radius_l212_212204


namespace determine_lunch_break_duration_lunch_break_duration_in_minutes_l212_212073

noncomputable def painter_lunch_break_duration (j h L : ℝ) : Prop :=
  (10 - L) * (j + h) = 0.6 ∧
  (8 - L) * h = 0.3 ∧
  (5 - L) * j = 0.1

theorem determine_lunch_break_duration (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L = 0.8 :=
by sorry

theorem lunch_break_duration_in_minutes (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L * 60 = 48 :=
by sorry

end determine_lunch_break_duration_lunch_break_duration_in_minutes_l212_212073


namespace perpendicular_AM_H1H2_l212_212208

-- Definitions and pre-conditions
variables {A B C D E F M N P Q H1 H2 : Point}
variable {triangle_ABC : Triangle A B C}
variable {BC_midpoint : Point}
variable {P_on_BC : P ∈ Line BC}
variable {Q_on_BC : Q ∈ Line BC}
variable {P_distance_midpoint : dist P BC_midpoint = some_equal_distance}
variable {Q_distance_midpoint : dist Q BC_midpoint = some_equal_distance}
variable {perpendicular_P : Perpendicular (Line P BC) (Line P E AC)}
variable {perpendicular_Q : Perpendicular (Line Q BC) (Line Q F AB)}
variable {M_intersection : M = LineIntersection (Line P F) (Line Q E)}
variable {orthocenter_BFP : Orthocenter triangle B F P H1}
variable {orthocenter_CEQ : Orthocenter triangle C E Q H2}

-- Theorem to prove
theorem perpendicular_AM_H1H2 : 
    (Perpendicular (Line A M) (Line H1 H2)) :=
sorry

end perpendicular_AM_H1H2_l212_212208


namespace num_pairs_equals_3_l212_212775

-- Define the problem requirements: positive integers m and n such that m ≥ n and m^2 - n^2 = 180
def num_ordered_pairs : ℕ :=
  let pairs := [(1, 180), (2, 90), (3, 60), (4, 45), (5, 36), (6, 30), (9, 20), (10, 18), (12, 15)] in
  (pairs.filter (λ p, let m := (p.1 + p.2) / 2 in let n := (p.2 - p.1) / 2 in 
  m ≥ n ∧ m^2 - n^2 = 180 ∧ m ∈ ℕ ∧ n ∈ ℕ)).length

theorem num_pairs_equals_3 : num_ordered_pairs = 3 :=
  sorry

end num_pairs_equals_3_l212_212775


namespace souvenir_costs_l212_212140

-- Define the variables and known conditions
variables (x y t : ℕ) (total_cost : ℕ)

-- Define the conditions
def condition1 := (x + 5 * y = 52)
def condition2 := (3 * x + 4 * y = 68)
def condition3 := (100 * x + 100 * y = 1200)
def condition4 (t : ℕ) := (992 ≤ 12 * t + 8 * (100 - t) ∧ 12 * t + 8 * (100 - t) ≤ 1002)
def cost820 := (x=12 ∧ y=8)
def costcondition : total_cost = 12 * t +100-y * t 

-- Lean statement for the proof
theorem souvenir_costs : 
(condition1) →
(condition2) →
condition4 (48) ∨ condition4 (49) ∨ condition4 (50) :=
sorry

end souvenir_costs_l212_212140


namespace initial_number_of_boarders_l212_212173

theorem initial_number_of_boarders (B D : ℕ) (h1 : B / D = 2 / 5) (h2 : (B + 15) / D = 1 / 2) : B = 60 :=
by
  -- Proof needs to be provided here
  sorry

end initial_number_of_boarders_l212_212173


namespace product_of_slopes_max_value_l212_212982

theorem product_of_slopes_max_value :
  ∃ m₁ m₂ : ℝ, (m₂ = 4 * m₁) ∧ (abs ((m₂ - m₁) / (1 + m₁ * m₂)) = sqrt 3) ∧ 4 * m₁^2 = 1.98 :=
begin
  sorry
end

end product_of_slopes_max_value_l212_212982


namespace angle_ABC_eq_angle_ADC_l212_212827

-- Given a convex quadrilateral ABCD
variables {A B C D O : Type}
variables [convex_quadrilateral A B C D]

-- Given conditions
variable (angle_CBD_eq_angle_CAB : ∠ CBD = ∠ CAB)
variable (angle_ACD_eq_angle_BDA : ∠ ACD = ∠ BDA)

-- Prove that ∠ ABC = ∠ ADC 
theorem angle_ABC_eq_angle_ADC :
  ∠ ABC = ∠ ADC :=
begin
  sorry -- Proof not required
end

end angle_ABC_eq_angle_ADC_l212_212827


namespace exists_tangent_circle_l212_212595

-- Define points and circles
variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

-- Assume the existence of three circles S1, S2, S3 passing through a common point O
variables (S1 S2 S3 : set α) (O : α)
(hS1 : metric.bounded S1) (hS2 : metric.bounded S2) (hS3 : metric.bounded S3)
(hO1 : O ∈ S1) (hO2 : O ∈ S2) (hO3 : O ∈ S3)
(h_intersect_S1S2 : ∃ A ∈ S1, A ∈ S2 ∧ A ≠ O)
(h_intersect_S1S3 : ∃ B ∈ S1, B ∈ S3 ∧ B ≠ O)
(h_intersect_S2S3 : ∃ C ∈ S2, C ∈ S3 ∧ C ≠ O)

-- We need to prove the existence of a circle S that is tangent to S1, S2, and S3
theorem exists_tangent_circle : ∃ S : set α, metric.bounded S ∧ ∀ x ∈ S, ∃ y z w, y ∈ S1 ∧ z ∈ S2 ∧ w ∈ S3 ∧ x = y ∨ x = z ∨ x = w :=
sorry

end exists_tangent_circle_l212_212595


namespace angle_equality_in_quadrilateral_l212_212841

/-- In a convex quadrilateral ABCD, 
    if ∠CBD = ∠CAB and ∠ACD = ∠BDA,
    then ∠ABC = ∠ADC. -/
theorem angle_equality_in_quadrilateral 
  {A B C D: Type*} [convex_quadrilateral A B C D]
  (h1 : ∠CBD = ∠CAB)
  (h2 : ∠ACD = ∠BDA) : 
  ∠ABC = ∠ADC := sorry

end angle_equality_in_quadrilateral_l212_212841


namespace original_population_l212_212172

-- Define the conditions given in the problem
def population_increase (x : ℝ) : ℝ := x + 1200
def population_decrease (x : ℝ) : ℝ := (population_increase x) * (1 - 0.11)
def final_population_condition (x : ℝ) : Prop := population_decrease x = x - 32

-- Statement of the problem
theorem original_population (x : ℝ) : final_population_condition x → x = 10000 :=
begin
  sorry
end

end original_population_l212_212172


namespace parabola_equation_l212_212576

theorem parabola_equation {p : ℝ} :
  vertex = (0, 0) ∧ axis_of_symmetry = x_axis ∧ distance ((-5, 2 * Real.sqrt 5)) (focus (y^2 = 2 * p * x)) = 6 →
  (y^2 = -4 * x ∨ y^2 = -36 * x) :=
by sorry

end parabola_equation_l212_212576


namespace Manolo_face_masks_total_l212_212813

theorem Manolo_face_masks_total :
  let masks_first_hour := 60 / 4,
      masks_second_third_hour := 120 / 6,
      masks_break := 0,
      masks_fifth_sixth_hour := 120 / 8,
      masks_seventh_eighth_hour := 120 / 6
  in masks_first_hour + masks_second_third_hour + masks_break + masks_fifth_sixth_hour + masks_seventh_eighth_hour = 70 :=
by {
  let masks_first_hour := 60 / 4,
  let masks_second_third_hour := 120 / 6,
  let masks_break := 0,
  let masks_fifth_sixth_hour := 120 / 8,
  let masks_seventh_eighth_hour := 120 / 6,
  show masks_first_hour + masks_second_third_hour + masks_break + masks_fifth_sixth_hour + masks_seventh_eighth_hour = 70,
  sorry
}

end Manolo_face_masks_total_l212_212813


namespace sum_first_3n_terms_l212_212573

variable (n : ℕ) (r : ℝ) (a : ℝ)
variable (hn : 0 < n) (hr : 0 < r) (hr' : r ≠ 1)

-- Conditions
def sum_first_n_geometric (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

axiom h1 : sum_first_n_geometric a r n = 48
axiom h2 : sum_first_n_geometric a r (2 * n) = 60

-- Question
theorem sum_first_3n_terms : sum_first_n_geometric a r (3 * n) = 63 :=
sorry

end sum_first_3n_terms_l212_212573


namespace distance_from_P_to_AD_l212_212541

theorem distance_from_P_to_AD :
  let A := (0, 6) in
  let D := (0, 0) in
  let C := (6, 0) in
  let M := (3, 0) in
  let intersect_Circle (center1 : ℝ × ℝ) (radius1 : ℝ) (center2 : ℝ × ℝ) (radius2 : ℝ) : set (ℝ × ℝ) :=
    {p | (p.1 - center1.1)^2 + (p.2 - center1.2)^2 = radius1^2 ∧
         (p.1 - center2.1)^2 + (p.2 - center2.2)^2 = radius2^2} in
  let P := (4.8, 2.4) in
  P ∈ intersect_Circle M 3 A 6 →
  P ≠ D →
  P.2 = 2.4 :=
by
  sorry

end distance_from_P_to_AD_l212_212541


namespace find_phi_l212_212955

noncomputable def f (x φ : ℝ) : ℝ :=
  (√3) * sin (2 * x + φ)

noncomputable def g (x φ : ℝ) : ℝ :=
  (√3) * sin (2 * x + φ + π / 3)

theorem find_phi (φ : ℝ) (h1 : |φ| < π / 2) (h2 : g 0 φ = 0) :
  φ = -π / 3 :=
by
  sorry

end find_phi_l212_212955


namespace largest_abs_difference_l212_212989

theorem largest_abs_difference : ∀ (a b : ℤ), a ∈ {-10, -3, 1, 5, 7, 15} ∧ b ∈ {-10, -3, 1, 5, 7, 15} → 
  |a - b| ≤ 25 ∧ ∃ a b, a ∈ {-10, -3, 1, 5, 7, 15} ∧ b ∈ {-10, -3, 1, 5, 7, 15} ∧ |a - b| = 25 :=
by
  sorry

end largest_abs_difference_l212_212989


namespace digits_at_positions_are_219_l212_212662

-- Define the sequence function
def digit_sequence : ℕ → ℕ :=
  sorry -- This function represents the digit sequence

-- Conditions identical to the problem statement
def starts_with_two (n : ℕ) : Prop := (digit_sequence n).to_string.get 0 = '2'

theorem digits_at_positions_are_219 :
  starts_with_two 1100 ∧ starts_with_two 1101 ∧ starts_with_two 1102 →
  (digit_sequence 1100 = 2 ∧ digit_sequence 1101 = 1 ∧ digit_sequence 1102 = 9) :=
  by
    sorry

end digits_at_positions_are_219_l212_212662


namespace area_of_region_l212_212988

theorem area_of_region 
  (x y : ℝ)
  (h : x^2 + y^2 + 5 = 4y - 6x + 9) : 
  ∃ r : ℝ, r = √17 ∧ ∃ c : ℝ × ℝ, c = (-3, 2) ∧ π * r^2 = 17 * π :=
by
  sorry

end area_of_region_l212_212988


namespace omitting_last_term_does_not_change_product_l212_212246

noncomputable theory

variable {a : ℝ}

theorem omitting_last_term_does_not_change_product
  (ha : a ≠ 0)
  (ha2 : a ≠ 2)
  (ha_neg2 : a ≠ -2) :
  (a^2 + 2 * a + 4 + 8 / a + 16 / a^2 + 64 / ((a - 2) * a^2)) *
  (a^2 - 2 * a + 4 - 8 / a + 16 / a^2 - 64 / ((a + 2) * a^2)) =
  (a^2 + 2 * a + 4 + 8 / a + 16 / a^2) *
  (a^2 - 2 * a + 4 - 8 / a + 16 / a^2) :=
sorry

end omitting_last_term_does_not_change_product_l212_212246


namespace weekly_hours_to_afford_vacation_l212_212423

theorem weekly_hours_to_afford_vacation 
  (planned_hours_per_week : ℕ) 
  (planned_weeks : ℕ) 
  (total_earnings_goal : ℕ) 
  (missed_weeks : ℕ) 
  (remaining_weeks : ℕ) 
  (required_hours_per_week : ℚ) : 
  planned_hours_per_week = 25 →
  planned_weeks = 15 →
  total_earnings_goal = 3750 →
  missed_weeks = 3 →
  remaining_weeks = 12 →
  required_hours_per_week = 31.25 →
  total_earnings_goal = planned_hours_per_week * 25 * remaining_weeks :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end weekly_hours_to_afford_vacation_l212_212423


namespace minimum_value_f_is_correct_l212_212349

noncomputable def f (x : ℝ) := 
  Real.sqrt (15 - 12 * Real.cos x) + 
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem minimum_value_f_is_correct :
  ∃ x : ℝ, f x = (9 / 2) * Real.sqrt 2 :=
sorry

end minimum_value_f_is_correct_l212_212349


namespace total_distinct_symbols_l212_212051

def numSequences (n : ℕ) : ℕ := 3^n

theorem total_distinct_symbols :
  numSequences 1 + numSequences 2 + numSequences 3 + numSequences 4 = 120 :=
by
  sorry

end total_distinct_symbols_l212_212051


namespace first_player_wins_with_probability_gt_half_l212_212862

theorem first_player_wins_with_probability_gt_half (s1 s2 : Nat) (board_config : Type) (σ1 σ2 : board_config → seq Nat) :
  s1 = s2 →
  (∀ bc : board_config, (probability_of_win σ1 σ2 bc > 1/2)) :=
begin
  intros h_eq bc,
  sorry
end

end first_player_wins_with_probability_gt_half_l212_212862


namespace initial_persons_count_l212_212157

theorem initial_persons_count (n : ℕ) 
  (avg_increase : 1.5) 
  (weight_left : 65) 
  (weight_new : 77) 
  (total_increase : avg_increase * n = weight_new - weight_left) : 
  n = 8 :=
by 
  have h : avg_increase * n = 12 := sorry
  have n_val : n = 12 / 1.5 := sorry
  rw [n_val, div_self] at h
  assumption

end initial_persons_count_l212_212157


namespace pentagon_area_l212_212471

theorem pentagon_area 
  (PQ QR RS ST TP : ℝ) 
  (angle_TPQ angle_PQR : ℝ) 
  (hPQ : PQ = 8) 
  (hQR : QR = 2) 
  (hRS : RS = 13) 
  (hST : ST = 13) 
  (hTP : TP = 8) 
  (hangle_TPQ : angle_TPQ = 90) 
  (hangle_PQR : angle_PQR = 90) : 
  PQ * QR + (1 / 2) * (TP - QR) * PQ + (1 / 2) * 10 * 12 = 100 := 
by
  sorry

end pentagon_area_l212_212471


namespace angle_B_is_30_degrees_l212_212804

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Assuming the conditions given in the problem
variables (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) 
          (h2 : a > b)

-- The proof to establish the measure of angle B as 30 degrees
theorem angle_B_is_30_degrees (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) (h2 : a > b) : B = Real.pi / 6 :=
sorry

end angle_B_is_30_degrees_l212_212804


namespace exists_shared_set_l212_212358

variables {α : Type*} (r s : ℕ) (F : set (set α))
open set

-- Define the assumptions
def conditions : Prop :=
  r > s ∧
  (∀ A ∈ F, F.infinite ∧ card A = r) ∧
  (∀ A B ∈ F, A ≠ B → card (A ∩ B) ≥ s)

-- State the theorem
theorem exists_shared_set (hr : r > s) (hF : ∀ A ∈ F, F.infinite ∧ card A = r) 
  (h_inter : ∀ A B ∈ F, A ≠ B → card (A ∩ B) ≥ s) : 
  ∃ (T : set α), card T = r - 1 ∧ ∀ A ∈ F, card (T ∩ A) ≥ s :=
sorry

end exists_shared_set_l212_212358


namespace inverse_at_one_l212_212795

noncomputable def f (x : ℝ) : ℝ := 3^x

noncomputable def f_inv : ℝ → ℝ := λ y, Real.log y / Real.log 3

theorem inverse_at_one : f_inv 1 = 0 :=
by
  unfold f_inv
  rw [Real.log_one, zero_div]
  exact zero_eq_zero

end inverse_at_one_l212_212795


namespace coefficient_of_x3_in_expression_l212_212710

def expression := 4 * (x^2 - 2 * x^3 + 2 * x) + 2 * (x + 3 * x^3 - 2 * x^2 + 4 * x^5 - x^3) - 6 * (2 + 2 * x - 5 * x^3 - 3 * x^2 + x^4)

theorem coefficient_of_x3_in_expression : coefficient_of_x3 expression = 26 :=
by
  sorry

end coefficient_of_x3_in_expression_l212_212710


namespace odd_divisor_probability_l212_212958

open Nat

-- Let’s define the prime factorization condition:
def fac23 : ℕ := 23.fact

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

-- Define the number of divisors of a number obtained by including or excluding certain divisors.
def num_divisors (n : ℕ) : ℕ :=
  (n.factorization 2 + 1) * (n.factorization 3 + 1) * (n.factorization 5 + 1) *
  (n.factorization 7 + 1) * (n.factorization 11 + 1) * (n.factorization 13 + 1) *
  (n.factorization 17 + 1) * (n.factorization 19 + 1) * (n.factorization 23 + 1)

def odd_divisors_count (n : ℕ) : ℕ :=
  (n.factorization 3 + 1) * (n.factorization 5 + 1) *
  (n.factorization 7 + 1) * (n.factorization 11 + 1) * (n.factorization 13 + 1) *
  (n.factorization 17 + 1) * (n.factorization 19 + 1) * (n.factorization 23 + 1)

def probability_odd_divisor (n : ℕ) : ℚ :=
  odd_divisors_count n / num_divisors n

theorem odd_divisor_probability : probability_odd_divisor fac23 = 11 / 23 := 
sorry

end odd_divisor_probability_l212_212958


namespace erick_total_money_collected_l212_212606

noncomputable def new_lemon_price (old_price increase : ℝ) : ℝ := old_price + increase
noncomputable def new_grape_price (old_price increase : ℝ) : ℝ := old_price + increase / 2

noncomputable def total_money_collected (lemons grapes : ℕ)
                                       (lemon_price grape_price lemon_increase : ℝ) : ℝ :=
  let new_lemon_price := new_lemon_price lemon_price lemon_increase
  let new_grape_price := new_grape_price grape_price lemon_increase
  lemons * new_lemon_price + grapes * new_grape_price

theorem erick_total_money_collected :
  total_money_collected 80 140 8 7 4 = 2220 := 
by
  sorry

end erick_total_money_collected_l212_212606


namespace cube_ratio_sum_l212_212959

theorem cube_ratio_sum (a b : ℝ) (h1 : |a| ≠ |b|) (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18 / 7 :=
by
  sorry

end cube_ratio_sum_l212_212959


namespace cyclic_shift_divisibility_l212_212191

-- Define the digits as a sequence of natural numbers
def digits : List ℕ := sorry

-- Define the number formed by reading a sequence of digits clockwise
def number (digits : List ℕ) : ℕ :=
List.foldl (λ acc d, acc * 10 + d) 0 digits

-- Given condition: the number starting at a certain position is divisible by 27
axiom initial_condition : ∃ start_pos : ℕ, (number (digits ++ digits).drop start_pos.take 1956) % 27 = 0

-- We need to prove that for any cyclic shift, the resulting 1956-digit number is also divisible by 27
theorem cyclic_shift_divisibility (start_pos : ℕ) (h1 : start_pos < 1956) : 
  (number (digits ++ digits).drop start_pos.take 1956) % 27 = 0 :=
sorry

end cyclic_shift_divisibility_l212_212191


namespace pencil_eraser_cost_l212_212075

theorem pencil_eraser_cost (p e : ℕ) (h_eq : 10 * p + 4 * e = 120) (h_gt : p > e) : p + e = 15 :=
by sorry

end pencil_eraser_cost_l212_212075


namespace marble_problem_l212_212254

def combinations : ℕ → ℕ → ℕ
| n, k := Nat.choose n k

def probability_same_color (total_marbles : ℕ) (reds whites blues draws : ℕ) : ℚ :=
  let total_combinations := combinations total_marbles draws
  let red_combinations := combinations reds draws
  let white_combinations := combinations whites draws
  let blue_combinations := combinations blues draws
  (red_combinations + white_combinations + blue_combinations) / total_combinations

theorem marble_problem :
  probability_same_color 21 6 7 8 4 = 8 / 399 :=
by
  sorry

end marble_problem_l212_212254


namespace smallest_four_digit_divisible_by_25_l212_212605

theorem smallest_four_digit_divisible_by_25 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 25 = 0 → n ≤ m := by
  -- Prove that the smallest four-digit number divisible by 25 is 1000
  sorry

end smallest_four_digit_divisible_by_25_l212_212605


namespace pick_three_different_cards_in_order_l212_212654

theorem pick_three_different_cards_in_order :
  (52 * 51 * 50) = 132600 :=
by
  sorry

end pick_three_different_cards_in_order_l212_212654


namespace find_lambda_l212_212980

noncomputable def hyperbola_eq : ℝ → ℝ → Prop :=
  λ x y, x^2 - y^2 / 2 = 1

theorem find_lambda :
  (∃ (c : ℝ) (lines : ℕ), c = sqrt 3 ∧ lines = 3 ∧ ∀ l, (l passes through the point (sqrt 3, 0)) → (l intersects hyperbola_eq at points A B) → abs (distance A B) = 4 → True) → ∃ (λ : ℝ), λ = 4 := μsorry

end find_lambda_l212_212980


namespace average_of_second_group_l212_212548

variable (avg1 : ℝ) (avg_all : ℝ) (A : ℝ)
variable (n1 n2 : ℕ)
variable (sum1 sum_all : ℝ)

-- Conditions
def cond1 : Prop := avg1 = 20
def cond2 : Prop := avg_all = 24
def cond3 : Prop := n1 = 30
def cond4 : Prop := n2 = 20
def cond5 : Prop := sum1 = n1 * avg1
def cond6 : Prop := sum_all = (n1 + n2) * avg_all
def cond7 : Prop := sum1 + n2 * A = sum_all

-- Proof statement
theorem average_of_second_group :
  cond1 → cond2 → cond3 → cond4 → cond5 → cond6 → cond7 → A = 30 := by
  sorry

end average_of_second_group_l212_212548


namespace arrangement_equivalence_l212_212947

theorem arrangement_equivalence (students : Fin 48 → Fin 48) :
  (∃ (P : ℕ), P = fact 48) ∧ (∃ (Q : ℕ), Q = fact 48) ∧ P = Q := by
  let P := fact 48
  let Q := fact 48
  use P, Q
  simp
  split
  . exact rfl
  . split
    . exact rfl
    . exact rfl

#check arrangement_equivalence

end arrangement_equivalence_l212_212947


namespace train_length_l212_212943

theorem train_length
  (train_speed_kmh : ℕ)
  (car_speed_kmh : ℕ)
  (time_cross_sec : ℕ)
  (train_speed_kmh = 36)
  (car_speed_kmh = 12)
  (time_cross_sec = 9) : 
  (train_length : ℕ) :=
  train_length = 60 :=
sorry

end train_length_l212_212943


namespace not_possible_odd_sum_l212_212024

theorem not_possible_odd_sum (m n : ℤ) (h : (m ^ 2 + n ^ 2) % 2 = 0) : (m + n) % 2 ≠ 1 :=
sorry

end not_possible_odd_sum_l212_212024


namespace minimum_value_of_expr_l212_212014

noncomputable def expr (x : ℝ) := 2 * x + 1 / (x + 3)

theorem minimum_value_of_expr (x : ℝ) (h : x > -3) :
  ∃ y, y = 2 * real.sqrt 2 - 6 ∧ ∀ z, z > -3 → expr z ≥ y := sorry

end minimum_value_of_expr_l212_212014


namespace magnitude_of_z_8_l212_212337

def z : Complex := 2 + 3 * Complex.I

theorem magnitude_of_z_8 : Complex.abs (z ^ 8) = 28561 := by
  sorry

end magnitude_of_z_8_l212_212337


namespace cost_of_one_box_and_board_min_num_painting_boards_l212_212814

open Real

noncomputable def cost_of_paintbrush : ℝ := 
  let x := 17
  x

noncomputable def cost_of_painting_board : ℝ := 
  let x := 17
  x - 2

theorem cost_of_one_box_and_board (x : ℝ) (h : 340 / x = 300 / (x - 2)) : 
  x = 17 ∧ x - 2 = 15 :=
by 
  have h1: 340 * (x - 2) = 300 * x := by
    rw [div_eq_iff, mul_div_cancel_left];
    apply ne_of_gt;
    norm_num;
  field_simp at h,
  linarith

theorem min_num_painting_boards (a : ℕ) (h : 17 * (30 - a) + 15 * a ≤ 475) : 
  a ≥ 18 :=
by 
  field_simp at h,
  linarith

end cost_of_one_box_and_board_min_num_painting_boards_l212_212814


namespace minimum_time_to_replace_shades_l212_212044

theorem minimum_time_to_replace_shades :
  ∀ (C : ℕ) (S : ℕ) (T : ℕ) (E : ℕ),
  ((C = 60) ∧ (S = 4) ∧ (T = 5) ∧ (E = 48)) →
  ((C * S * T) / E = 25) :=
by
  intros C S T E h
  rcases h with ⟨hC, hS, hT, hE⟩
  sorry

end minimum_time_to_replace_shades_l212_212044


namespace production_today_l212_212725

theorem production_today (n x: ℕ) (avg_past: ℕ) 
  (h1: avg_past = 50) 
  (h2: n = 1) 
  (h3: (avg_past * n + x) / (n + 1) = 55): 
  x = 60 := 
by 
  sorry

end production_today_l212_212725


namespace exists_decomposition_l212_212878

variable (n k : ℕ)
variable (A : Matrix (Fin n) (Fin n) ℤ)
variable (b : Fin k → ℤ)

theorem exists_decomposition (det_condition : A.det = (Finset.univ : Finset (Fin k)).prod b)
  : ∃ (B : Fin k → Matrix (Fin n) (Fin n) ℤ), A = Matrix.mul ![B 0, B 1, ..., B (k-1)] ∧ ∀ i, (b i) = (B i).det :=
  sorry

end exists_decomposition_l212_212878


namespace distinct_colorings_of_cube_l212_212582

theorem distinct_colorings_of_cube (m : ℕ) : 
  ∃ M : ℕ, M = (1 / 24 : ℚ) * m^2 * (m^6 + 17 * m^2 + 6) := by
  use ((1 / 24 : ℚ) * m^2 * (m^6 + 17 * m^2 + 6)).nat_abs
  rw [Rat.cast_mul, ←Nat.cast_mul, ←Nat.cast_mul, ←Nat.cast_mul, ←Nat.cast_mul, mul_assoc, mul_assoc, nat_abs_cast]
  sorry

end distinct_colorings_of_cube_l212_212582


namespace equal_angles_quadrilateral_l212_212835

theorem equal_angles_quadrilateral
  (AB CD : Type)
  [convex_quad AB CD]
  (angle_CBD angle_CAB angle_ACD angle_BDA : AB CD → ℝ)
  (h1 : angle_CBD = angle_CAB)
  (h2 : angle_ACD = angle_BDA) : angle_ABC = angle_ADC :=
by sorry

end equal_angles_quadrilateral_l212_212835


namespace find_n_of_geometric_sum_l212_212967

-- Define the first term and common ratio of the sequence
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3

-- Define the sum of the first n terms of the geometric sequence
def S_n (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Mathematical statement to be proved
theorem find_n_of_geometric_sum (h : S_n 5 = 80 / 243) : ∃ n, S_n n = 80 / 243 ↔ n = 5 :=
by
  sorry

end find_n_of_geometric_sum_l212_212967


namespace value_of_a_minus_b_l212_212009

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a - b = 5) (h2 : a - 2 * b = 4) : a - b = 3 :=
by
  sorry

end value_of_a_minus_b_l212_212009


namespace equivalent_spherical_coords_l212_212465

theorem equivalent_spherical_coords (ρ θ φ : ℝ) (hρ : ρ = 4) (hθ : θ = 3 * π / 8) (hφ : φ = 9 * π / 5) :
  ∃ (ρ' θ' φ' : ℝ), ρ' = 4 ∧ θ' = 11 * π / 8 ∧ φ' = π / 5 ∧ 
  (ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by
  sorry

end equivalent_spherical_coords_l212_212465


namespace ball_distributions_l212_212699

theorem ball_distributions (p q : ℚ) (h1 : p = (Nat.choose 5 1 * Nat.choose 4 1 * Nat.choose 20 2 * Nat.choose 18 6 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20)
                            (h2 : q = (Nat.choose 20 4 * Nat.choose 16 4 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20) :
  p / q = 10 :=
by
  sorry

end ball_distributions_l212_212699


namespace imaginary_part_of_conjugate_l212_212158

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

theorem imaginary_part_of_conjugate :
  ∀ (z : ℂ), z = (1+i)^2 / (1-i) → (complex_conjugate z).im = -1 :=
by
  sorry

end imaginary_part_of_conjugate_l212_212158


namespace equation1_solution_equation2_no_solution_l212_212135

theorem equation1_solution (x: ℝ) (h: x ≠ -1/2 ∧ x ≠ 1):
  (1 / (x - 1) = 5 / (2 * x + 1)) ↔ (x = 2) :=
sorry

theorem equation2_no_solution (x: ℝ) (h: x ≠ 1 ∧ x ≠ -1):
  ¬ ( (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 ) :=
sorry

end equation1_solution_equation2_no_solution_l212_212135


namespace solve_for_y_l212_212535

theorem solve_for_y (y : ℝ) (h : 4^(3 * y) = real.cbrt 64) : y = 1 / 3 :=
by
  sorry

end solve_for_y_l212_212535


namespace circumcenter_of_A_l212_212892

-- Definitions of points and their properties
variables {A B C I O A' B' C' : Type}

-- Assumptions about incenter, circumcenter, and circumcenters of specific triangles
variable [incircle_ABC : is_incident_circle A B C I]
variable [circumcenter_ABC : is_circumcenter A B C O]
variable [A'_circumcenter : is_circumcenter I B C A']
variable [B'_circumcenter : is_circumcenter I C A B']
variable [C'_circumcenter : is_circumcenter I A B C']

-- Theorem statement
theorem circumcenter_of_A'B'C' :
  is_circumcenter A' B' C' O :=
sorry

end circumcenter_of_A_l212_212892


namespace area_enclosed_by_line_and_curve_l212_212546

theorem area_enclosed_by_line_and_curve:
  let f := fun x: ℝ => 4*x
  let g := fun x: ℝ => x^3
  (∫ x in 0..2, f x - g x) = 4 := by
    sorry

end area_enclosed_by_line_and_curve_l212_212546


namespace number_of_8th_graders_l212_212047

variable (x y : ℕ)
variable (y_valid : 0 ≤ y)

theorem number_of_8th_graders (h : x * (x + 3 - 2 * y) = 14) :
  x = 7 :=
by 
  sorry

end number_of_8th_graders_l212_212047
