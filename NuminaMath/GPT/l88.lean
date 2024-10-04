import Mathlib

namespace triangle_side_b_eq_l88_88553

noncomputable def law_of_cosines_solution : ℝ :=
  let a : ℝ := 2
  let c : ℝ := 3
  let B : ℝ := real.pi * (120 / 180) -- converting degrees to radians
  let b_squared : ℝ := c^2 + a^2 - 2 * a * c * (Real.cos B)
  real.sqrt b_squared

theorem triangle_side_b_eq : law_of_cosines_solution = Real.sqrt 19 := by
  sorry

end triangle_side_b_eq_l88_88553


namespace range_of_a_l88_88239

theorem range_of_a {a : ℝ} : 
  (∃ x : ℝ, (1 / 2 < x ∧ x < 3) ∧ (x ^ 2 - a * x + 1 = 0)) ↔ (2 ≤ a ∧ a < 10 / 3) :=
by
  sorry

end range_of_a_l88_88239


namespace probability_three_heads_l88_88320

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l88_88320


namespace prob_1_less_X_less_2_l88_88889

noncomputable def NormalDist (mean variance : ℝ) : Type := sorry -- Placeholder for normal distribution type

variable (X : NormalDist 1 4)

axiom prob_X_less_than_2 : P(X < 2) = 0.72

theorem prob_1_less_X_less_2 : P(1 < X < 2) = 0.22 :=
by
  have h1 : P(X ≥ 2) = 1 - P(X < 2) := sorry
  have h2 : P(X > 1) = 0.5 := sorry
  have h3 : P(1 < X < 2) = P(X > 1) - P(X ≥ 2) := sorry
  show P(1 < X < 2) = 0.22
sorry

end prob_1_less_X_less_2_l88_88889


namespace choosing_officers_l88_88572

theorem choosing_officers :
  let n := 6 in
  (n * (n - 1) * (n - 2)) = 120 :=
by
  let n := 6
  show (n * (n - 1) * (n - 2)) = 120
  sorry

end choosing_officers_l88_88572


namespace sum_abcd_l88_88753

variables (a b c d : ℚ)

theorem sum_abcd :
  3 * a + 4 * b + 6 * c + 8 * d = 48 →
  4 * (d + c) = b →
  4 * b + 2 * c = a →
  c + 1 = d →
  a + b + c + d = 513 / 37 :=
by
sorry

end sum_abcd_l88_88753


namespace truck_capacity_rental_plan_l88_88790

-- Define the variables for the number of boxes each type of truck can carry
variables {x y : ℕ}

-- Define the conditions for the number of boxes carried by trucks
axiom cond1 : 15 * x + 25 * y = 750
axiom cond2 : 10 * x + 30 * y = 700

-- Problem 1: Prove x = 25 and y = 15
theorem truck_capacity : x = 25 ∧ y = 15 :=
by
  sorry

-- Define the variables for the number of each type of truck
variables {m : ℕ}

-- Define the conditions for the total number of trucks and boxes to be carried
axiom cond3 : 25 * m + 15 * (70 - m) ≤ 1245
axiom cond4 : 70 - m ≤ 3 * m

-- Problem 2: Prove there is one valid rental plan with m = 18 and 70-m = 52
theorem rental_plan : 17 ≤ m ∧ m ≤ 19 ∧ 70 - m ≤ 3 * m ∧ (70-m = 52 → m = 18) :=
by
  sorry

end truck_capacity_rental_plan_l88_88790


namespace part_a_l88_88215

theorem part_a : let x := (0.12:ℝ) in 
                 let y := (0.122:ℝ) in
                 x + y = (4158 / 1089:ℝ) := by
  sorry

end part_a_l88_88215


namespace decimal_to_binary_89_l88_88043

theorem decimal_to_binary_89 : nat.toBinary 89 = "1011001" :=
by 
  sorry

end decimal_to_binary_89_l88_88043


namespace cross_product_scalar_multiplication_dot_product_scalar_multiplication_l88_88138

variables (a b : ℝ×ℝ×ℝ) (c d : ℝ)

-- Given Conditions
def cross_product_condition : a × b = ⟨-3, 7, 2⟩ := sorry
def dot_product_condition : a • b = 4 := sorry

-- Proof Problem Statements
theorem cross_product_scalar_multiplication :
  a × (5 • b) = ⟨-15, 35, 10⟩ :=
by
  -- Add the given conditions as assumptions
  assume h1 : cross_product_condition a b,
  sorry

theorem dot_product_scalar_multiplication :
  a • (5 • b) = 20 :=
by
  -- Add the given conditions as assumptions
  assume h2 : dot_product_condition a b,
  sorry

end cross_product_scalar_multiplication_dot_product_scalar_multiplication_l88_88138


namespace shaded_area_is_correct_l88_88561

-- Define the radii of semicircles ADB, BEC, and DFE
def r_ADB : ℝ := 2
def r_BEC : ℝ := 3
def r_DFE : ℝ := 2.5

-- Define the areas of semicircles
def area_ADB : ℝ := (1/2) * π * r_ADB^2
def area_BEC : ℝ := (1/2) * π * r_BEC^2
def area_DFE : ℝ := (1/2) * π * r_DFE^2

-- Define the assumed overlap area
def overlap_area : ℝ := π

-- Define the shaded area
def shaded_area : ℝ := area_ADB + area_BEC + area_DFE - overlap_area

-- Prove that the shaded area is equal to 8.625π square units
theorem shaded_area_is_correct : shaded_area = 8.625 * π := by
  sorry

end shaded_area_is_correct_l88_88561


namespace calculate_q_div_p_l88_88059

noncomputable def cards : (Fin 50 → Fin 10) :=
  sorry -- Assume a function that assigns each of 50 cards a number from 1 to 10 

def all_same_number (drawn_cards : Fin 5 → Fin 50) : Prop :=
  ∀ i j, cards (drawn_cards i) = cards (drawn_cards j)

def four_same_one_different (drawn_cards : Fin 5 → Fin 50) : Prop :=
  ∃ a b : Fin 10, a ≠ b ∧
    (∃ count : Fin 5 → Fin 2, (∃ count_a, (count i = 0 ↔ cards (drawn_cards i) = a) ∧ (count_a 0 = 4)) ∧
                    ∃ count_b, (count i = 1 ↔ cards (drawn_cards i) = b) ∧ (count_b 1 = 1))

theorem calculate_q_div_p :
  let p := 10 / choose 50 5 in
  let q := 2250 / choose 50 5 in
  q / p = 225 :=
by
  sorry

end calculate_q_div_p_l88_88059


namespace geometric_statement_l88_88267

open EuclideanGeometry

-- Assume the plane existence
variables (P : Type) [metric_space P] [normed_group P] [normed_space ℝ P]

-- Define the circles, intersection points and lines
variables
  (Γ1 Γ2 : circle P)
  (A B K M Q R : P)
  (ℓ ℓ1 ℓ2 : line P)

-- Circle intersections at A and B
axiom circle_intersects_at_two_points : ∀ (Γ1 Γ2 : circle P), intersects Γ1 Γ2 A ∧ intersects Γ1 Γ2 B

-- ℓ passes through B and intersects Γ1 at K and Γ2 at M (K, M ≠ B)
axiom line_intersects_circles : passes_through ℓ B ∧ touches ℓ Γ1 K ∧ touches ℓ Γ2 M ∧ K ≠ B ∧ M ≠ B

-- ℓ1 is parallel to AM and tangent to Γ1 at Q
axiom line_parallel_tangent : parallel ℓ1 (join A M) ∧ tangent_to ℓ1 Γ1 Q

-- QA intersects Γ2 again at R (R ≠ A)
axiom line_intersects_again : intersects_again (join Q A) Γ2 R ∧ R ≠ A

-- ℓ2 is tangent to Γ2 at R
axiom tangent_line : tangent_to ℓ2 Γ2 R

theorem geometric_statement :
  (parallel ℓ2 (join A K)) ∧ ∃ S, intersects ℓ ℓ1 S ∧ intersects ℓ2 ℓ1 S ∧ intersects ℓ ℓ2 S :=
  by 
  sorry

end geometric_statement_l88_88267


namespace cannot_determine_sum_l88_88897

-- Given conditions
variables (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
variables (h_invertible : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)

-- Function values provided
def f_values : ℕ → ℝ
| 1 := 2
| 2 := 4
| 3 := 6
| 4 := 9
| 5 := 10
| _ := 0

-- Assume that f(3) = 6 and f_inv(6) = 3
axiom f_3 : f 3 = 6
axiom f_inv_6 : f_inv 6 = 3

-- The main theorem we want to prove
theorem cannot_determine_sum : f (f 3) + f (f_inv 6) + f_inv (f (f_inv 6)) = 0 → False :=
by
  sorry

end cannot_determine_sum_l88_88897


namespace kim_distance_traveled_l88_88020

-- Definitions based on the problem conditions:
def infantry_column_length : ℝ := 1  -- The length of the infantry column in km.
def distance_inf_covered : ℝ := 2.4  -- Distance the infantrymen covered in km.

-- Theorem statement:
theorem kim_distance_traveled (column_length : ℝ) (inf_covered : ℝ) :
  column_length = 1 →
  inf_covered = 2.4 →
  ∃ d : ℝ, d = 3.6 :=
by
  sorry

end kim_distance_traveled_l88_88020


namespace probability_vertex_B_is_bottom_vertex_l88_88364

-- Define the dodecahedron structure
structure Dodecahedron :=
  (top_vertex : Vertex)
  (bottom_vertex : Vertex)
  (adjacent_vertices : Vertex → Finset Vertex)
  (middle_ring_size : Nat := 5)
  (pyramid_base_vertices : Finset Vertex)

variable {d : Dodecahedron}

-- Define the vertices
variables (A B : d.pyramid_base_vertices)

-- Define the conditions
def ant_start_at_top_vertex (d : Dodecahedron) : Prop :=
  ∃ A ∈ d.adjacent_vertices d.top_vertex, true

def ant_walking_to_B_from_A (d : Dodecahedron) : Prop :=
  ∃ B ∈ d.adjacent_vertices A, true

-- Define the main theorem to be proved
theorem probability_vertex_B_is_bottom_vertex :
  ant_start_at_top_vertex d → 
  ant_walking_to_B_from_A d → 
  (1 / d.middle_ring_size : ℝ) = 1 / 5 :=
by
  sorry

end probability_vertex_B_is_bottom_vertex_l88_88364


namespace log35_28_l88_88875

variable (a b : ℝ)
variable (log : ℝ → ℝ → ℝ)

-- Conditions
axiom log14_7_eq_a : log 14 7 = a
axiom log14_5_eq_b : log 14 5 = b

-- Theorem to prove
theorem log35_28 (h1 : log 14 7 = a) (h2 : log 14 5 = b) : log 35 28 = (2 - a) / (a + b) :=
sorry

end log35_28_l88_88875


namespace domino_arrangements_l88_88657

def numDominoArrangements := 126

theorem domino_arrangements (R D : ℕ) (hf_moves:  R = 4 ∧ D = 5)
  (hf_path: ∀ p : ℕ × ℕ, p.1 ≤ 6 ∧ p.2 ≤ 4 ∧ p.1 + p.2 <= 9):
  R + D = 9 → ( ∑ n in finset.range (5+1), (n choose 4)) = numDominoArrangements := 
by 
  sorry

end domino_arrangements_l88_88657


namespace max_area_of_triangle_ABC_l88_88886

noncomputable def parabola (x : ℝ) : ℝ := (6 * x)^(1/2)

def point_on_parabola (x1 x2 y1 y2 : ℝ) : Prop :=
  (y1^2 = 6 * x1) ∧ (y2^2 = 6 * x2)

def parabola_conditions (x1 x2 : ℝ) : Prop :=
  (x1 ≠ x2) ∧ (x1 + x2 = 4)

def perp_bisector_point (x1 x2 y1 y2 : ℝ) : Prop :=
  (- (y1^2 - y2^2)/(4 * (x2 - x1)) + 2)

def max_area (x1 x2 y1 y2 : ℝ) : ℝ :=
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  let m_c := (0 - ym) / (5 - 2)
  let area := 1 / 2 * abs (t * ym + 3 * m_c) in
  max area

theorem max_area_of_triangle_ABC :
  ∀ x1 x2 y1 y2 : ℝ,
    point_on_parabola x1 x2 y1 y2 →
    parabola_conditions x1 x2 →
    max_area x1 x2 y1 y2 = 7*sqrt(7)/6 :=
by
  sorry

end max_area_of_triangle_ABC_l88_88886


namespace max_value_MP_l88_88716

noncomputable def O := Type -- Assume the existence of point O
noncomputable def A := Type -- Assume the existence of vertex A
noncomputable def B := Type -- Assume the existence of vertex B
noncomputable def C := Type -- Assume the existence of vertex C
noncomputable def M := Type -- Assume the existence of point M
noncomputable def P := Type -- Assume the existence of point P

variables (OA OB OC : ℝ) -- Vectors OA, OB, OC can be represented as real numbers

-- Given conditions
axiom condition1 : ∀ (OA OB OC : ℝ), OA - 2*OB - 3*OC = 0
axiom condition2 : ∀ (A B C : Type) (side_length : ℝ), side_length = 8
axiom point_on_side : ∀ (M : Type) (A B C : Type), (M ∈ [A, B] ∨ M ∈ [B, C] ∨ M ∈ [C, A])
axiom O_P_distance : ∀ (O P : Type), (|P - O| = sqrt 19)

-- Target proof (we use ℝ for distances)
theorem max_value_MP : ∀ (MP : ℝ), (|MP| ≤ 3 * sqrt 19) := 
by
  sorry

end max_value_MP_l88_88716


namespace smallest_number_of_cubes_l88_88806

noncomputable def container_cubes (length_ft : ℕ) (height_ft : ℕ) (width_ft : ℕ) (prime_inch : ℕ) : ℕ :=
  let length_inch := length_ft * 12
  let height_inch := height_ft * 12
  let width_inch := width_ft * 12
  (length_inch / prime_inch) * (height_inch / prime_inch) * (width_inch / prime_inch)

theorem smallest_number_of_cubes :
  container_cubes 60 24 30 3 = 2764800 :=
by
  sorry

end smallest_number_of_cubes_l88_88806


namespace prob_three_heads_is_one_eighth_l88_88333

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l88_88333


namespace triangle_BD_length_l88_88764

theorem triangle_BD_length (b c p q : ℝ) (hpr : 0 < p)
  (hqr : 0 < q) (hangleA : real.cos (real.pi / 3) = 1 / 2) :
  let AC := 2 * b,
      AB := c,
      BC := real.sqrt (4 * b^2 + c^2 - 2 * 2 * b * c * real.cos (real.pi / 3))
  in ∃ BD : ℝ, BD = (p / (p + 2 * q)) * BC :=
begin
  sorry
end

end triangle_BD_length_l88_88764


namespace count_four_digit_integers_with_product_18_l88_88965

def valid_digits (n : ℕ) : Prop := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def digit_product_18 (a b c d : ℕ) : Prop := 
  a * b * c * d = 18

def four_digit_integer (a b c d : ℕ) : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d

theorem count_four_digit_integers_with_product_18 : 
  (∑ a b c d in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    ite (four_digit_integer a b c d ∧ digit_product_18 a b c d) 1 0) = 48 := 
sorry

end count_four_digit_integers_with_product_18_l88_88965


namespace actual_discount_and_difference_l88_88810

variable (P : ℝ) (original_price discounted_once discounted_twice final_discounted_price : ℝ)

def step_one (P: ℝ) : ℝ :=
  0.75 * P

def step_two (discounted_once: ℝ) : ℝ :=
  0.85 * discounted_once

def step_three (final_discounted_price: ℝ) : ℝ :=
  100 - (final_discounted_price / P * 100)

theorem actual_discount_and_difference (P : ℝ) :
  let discounted_once := step_one P in
  let discounted_twice := step_two discounted_once in
  let actual_discount := step_three discounted_twice in
  actual_discount = 36.25 ∧ (40 - actual_discount) = 3.75 :=
by
  sorry

end actual_discount_and_difference_l88_88810


namespace max_intersections_in_first_quadrant_l88_88435

theorem max_intersections_in_first_quadrant :
  ∀ (X Y : Finset ℕ), 
    X.card = 15 →
    Y.card = 10 →
    (∃ S, S.card = 4725 ∧ 
          ∀ s ∈ S, 
            ∃ (x1 x2 ∈ X) (y1 y2 ∈ Y), 
              x1 < x2 ∧ y1 < y2 ∧ s = (x1, y1, x2, y2)) :=
by sorry

end max_intersections_in_first_quadrant_l88_88435


namespace dan_cards_left_l88_88835

-- Problem conditions
def original_cards : ℤ := 97
def torn_cards : ℤ := 8
def sold_cards : ℤ := 15

-- The theorem statement
theorem dan_cards_left (original_cards = 97) (torn_cards = 8) (sold_cards = 15) : 
  original_cards - sold_cards = 82 :=
by
  sorry

end dan_cards_left_l88_88835


namespace distinct_four_digit_numbers_product_18_l88_88983

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l88_88983


namespace tangent_slope_perpendicular_l88_88107

theorem tangent_slope_perpendicular (a b : ℝ)
  (h1 : b = a^3)
  (h2 : differentiable ℝ (λ x, x^3))
  (h3 : (deriv (λ x, x^3) a) * (-1/3) = -1) :
  a = 1 ∨ a = -1 :=
by
  sorry

end tangent_slope_perpendicular_l88_88107


namespace probability_of_first_three_heads_l88_88293

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l88_88293


namespace answer_l88_88575

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

end answer_l88_88575


namespace probability_three_heads_l88_88291

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l88_88291


namespace total_height_rounded_is_correct_l88_88012

def pole1_height_in := 12
def pole2_height_in := 36
def pole3_height_in := 44
def conversion_factor := 2.58

noncomputable def total_height_cm : Float :=
  (pole1_height_in * conversion_factor) + 
  (pole2_height_in * conversion_factor) + 
  (pole3_height_in * conversion_factor)

noncomputable def round_to_nearest_tenth (x : Float) : Float :=
  (Float.ofInt (Int.ofFloat (10 * x + 0.5))) / 10

theorem total_height_rounded_is_correct :
  round_to_nearest_tenth total_height_cm = 237.4 := 
by
  sorry

end total_height_rounded_is_correct_l88_88012


namespace problem_1_problem_2_l88_88536

def vector (ℝ: Type *) := (ℝ × ℝ)

variables x : ℝ

def a : vector ℝ := (1, 2)
def b : vector ℝ := (x, 1)

def vector_add (v1 v2 : vector ℝ) : vector ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)
def vector_scalar_mul (c : ℝ) (v : vector ℝ) : vector ℝ :=
  (c * v.1, c * v.2)
def vector_sub (v1 v2 : vector ℝ) : vector ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)
def cross_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.2 - v1.2 * v2.1
def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem problem_1 :
  cross_product (vector_add a (vector_scalar_mul 2 b))
                (vector_sub (vector_scalar_mul 2 a) b) = 0 → x = 1 / 2 :=
by
  sorry

theorem problem_2 :
  dot_product (vector_add a (vector_scalar_mul 2 b))
              (vector_sub (vector_scalar_mul 2 a) b) = 0 → x = -2 ∨ x = 7 / 2 :=
by
  sorry

end problem_1_problem_2_l88_88536


namespace incorrect_statement_l88_88102

-- Definitions for given conditions
variables (α β : Plane) (l a b : Line)

-- Intersecting planes along a line
axiom planes_intersect : l ∈ α ∧ l ∈ β

-- Lines lie within respective planes
axiom a_in_alpha : a ∈ α
axiom b_in_beta : b ∈ β

-- Definitions for perpendicularity
axiom perp_planes (α β : Plane) : Prop
axiom perp_lines (l1 l2 : Line) : Prop

-- Conditions for the statement D's examination
axiom a_perp_l : perp_lines a l
axiom b_perp_l : perp_lines b l

-- Problem statement (which needs to be proven incorrect)
theorem incorrect_statement :
  (a_perp_l ∧ b_perp_l) → ¬ perp_planes α β := sorry

end incorrect_statement_l88_88102


namespace area_inside_C_outside_A_B_l88_88831

def radiusA := 1
def radiusB := 1
def radiusC := 2

def circle_tangent (r1 r2: ℝ) : Prop :=
  ∃ (p1 p2: ℝ × ℝ), dist p1 p2 = r1 + r2

def midpoint_tangent (p1 p2: ℝ × ℝ) (r: ℝ) : Prop :=
  ∃ (pm (px : ℝ × ℝ)), pm = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2) ∧ dist pm px = r

theorem area_inside_C_outside_A_B : 
  circle_tangent radiusA radiusB ∧
  midpoint_tangent ((0, 0) : ℝ × ℝ) ((2, 0) : ℝ × ℝ) radiusC
  → π * radiusC ^ 2 - 2 * (π * radiusA ^ 2 / 3 - sqrt 3 / 2 * radiusA ^ 2) = 4 * π - 4.912 := 
by
  sorry

end area_inside_C_outside_A_B_l88_88831


namespace three_heads_in_a_row_l88_88310

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l88_88310


namespace quadrant_of_tan_and_cos_l88_88544

theorem quadrant_of_tan_and_cos (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  ∃ Q, (Q = 2) :=
by
  sorry


end quadrant_of_tan_and_cos_l88_88544


namespace sum_two_digit_integers_square_ends_with_09_l88_88741

theorem sum_two_digit_integers_square_ends_with_09 :
  let ns := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n ^ 2) % 100 = 9} in
  (∑ n in ns, n) = 100 :=
by
  sorry

end sum_two_digit_integers_square_ends_with_09_l88_88741


namespace _l88_88639

noncomputable theorem bisection_method_example :
  (∀ x : ℝ, f (x) = 4*x^3 + x - 8) →
  f 1 < 0 →
  f 1.5 > 0 →
  ∃ x ∈ Ioo 1 1.5, f x = 0 :=
by
  intro f_def f_1_lt_0 f_1_5_gt_0
  sorry

end _l88_88639


namespace problem_part1_problem_part2_l88_88527

noncomputable def polar_to_rectangular (ρ θ : ℝ) : Prop := 
  ρ * (sin θ) ^ 2 = 6 * cos θ

-- We define the parametric equations of the line l
def line_l (t : ℝ) : ℝ × ℝ := 
  ((3 / 2 + 1 / 2 * t), (sqrt 3 / 2 * t))

-- Stating the main proof goals
theorem problem_part1 (ρ θ x y : ℝ) (h1 : polar_to_rectangular ρ θ)
    (hx : x = ρ * cos θ)
    (hy : y = ρ * sin θ) :
    y^2 = 6 * x := sorry

theorem problem_part2 (t1 t2 : ℝ)
    (h2 : t1 = 6)
    (h3 : t2 = -2)
    (A B : ℝ × ℝ)
    (hA : A = line_l t1)
    (hB : B = line_l t2) :
    dist A B = sqrt 73 := sorry

end problem_part1_problem_part2_l88_88527


namespace prob_first_three_heads_all_heads_l88_88316

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l88_88316


namespace probability_three_heads_l88_88286

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l88_88286


namespace number_of_subsets_l88_88876

open Finset

/-- Given that {2, 3} ⊆ Y ⊆ {1, 2, 3, 4, 5, 6}, prove that the number of such subsets Y is 16. -/
theorem number_of_subsets (Y : Finset ℕ) (h : {2, 3} ⊆ Y ∧ Y ⊆ {1, 2, 3, 4, 5, 6}) :
    (univ.filter (λ X : Finset ℕ, {2, 3} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5, 6})).card = 16 := by
  sorry

end number_of_subsets_l88_88876


namespace find_speeds_l88_88481

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l88_88481


namespace baron_munchausen_claim_l88_88252

-- Definitions based on conditions
def cube := ℕ
def color := ℕ -- 0 represents white, 1 represents black, 2 represents red

-- Assume all 16 cubes
noncomputable def cubes : ℕ := 16

-- Function to determine arrangement validity
def can_arrange (C : cube) : Prop :=
  ∃ (arrangement : ℕ → ℕ → ℕ → color),
  (∀ i j k, arrangement i j k = 0) ∨ 
  (∀ i j k, arrangement i j k = 1) ∨ 
  (∀ i j k, arrangement i j k = 2)

-- Main theorem to be proven
theorem baron_munchausen_claim : can_arrange cubes :=
sorry

end baron_munchausen_claim_l88_88252


namespace solve_for_p_l88_88759

variable (P : ℝ)
def q_r_combined := (2 / 7) * P
def p_has_more := P = q_r_combined + 35

theorem solve_for_p (h : p_has_more P) : P = 49 :=
by
  sorry

end solve_for_p_l88_88759


namespace find_ellipse_eq_maximize_area_l88_88162
open Real

-- Conditions
variable (a b : ℝ) (c : ℝ := sqrt 6) (h_ab : a > b > 0)
variable (h_focal : 2 * c = 2 * sqrt 6)
variable (h_point : sqrt 2^2 / a^2 + sqrt 5^2 / b^2 = 1)

-- Definitions for ellipses and points
def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem 1: Finding the Ellipse Equation
theorem find_ellipse_eq (h_ab : a > b > 0) (h_focal : 2 * c = 2 * sqrt 6)
    (h_point : ellipse_eq a b (sqrt 2) (sqrt 5)) :
  ellipse_eq 12 6 x y :=
sorry

-- Conditions for maximizing the area of the triangle
variable (x0 y0 : ℝ) (h_x0 : 2 < x0 ∧ x0 ≤ 2 * sqrt 3)
variable (h_P_on_ellipse : ellipse_eq 12 6 x0 y0)

-- Theorem 2: Maximizing Area of Triangle PAB
theorem maximize_area (h_x0 : 2 < x0 ∧ x0 ≤ 2 * sqrt 3) (h_P_on_ellipse : ellipse_eq 12 6 x0 y0) :
  (x0, y0) = (2 * sqrt 3, 0) :=
sorry

end find_ellipse_eq_maximize_area_l88_88162


namespace a2_plus_b2_minus_abc_is_perfect_square_l88_88986

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem a2_plus_b2_minus_abc_is_perfect_square {a b c : ℕ} (h : 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c) :
  is_perfect_square (a^2 + b^2 - a * b * c) :=
by
  sorry

end a2_plus_b2_minus_abc_is_perfect_square_l88_88986


namespace circle_partition_contains_small_circle_l88_88271

theorem circle_partition_contains_small_circle :
  ∀ (R : ℝ), R = 2000 → ∀ (n : ℕ), n = 1996 →
  ∃ (r : ℝ), r = 1 → ∃ (region : Set (ℝ × ℝ)), 
    (region ⊆ metric.ball (0, 0) R ∧ ∀ (line : ℝ × ℝ), line ∈ region → line ∉ ⋃ i : fin n, strip (line i) r) :=
begin
  sorry  
end

end circle_partition_contains_small_circle_l88_88271


namespace sum_of_distances_l88_88559

def point := (ℝ × ℝ)

def A : point := (20, 0)
def B : point := (1, 0)
def D : point := (1, 7)

noncomputable def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem sum_of_distances :
  let AD := distance A D,
      BD := distance B D
  in 27 < AD + BD ∧ AD + BD < 28 :=
by
  let AD := distance A D
  let BD := distance B D
  sorry

end sum_of_distances_l88_88559


namespace prob_three_heads_is_one_eighth_l88_88329

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l88_88329


namespace cuboid_height_third_dimension_l88_88796

theorem cuboid_height_third_dimension : 
  ∃ (h : ℕ), (∀ (side_length : ℕ), side_length = 3 → 
  (6 * 9 * h = 24 * (side_length ^ 3))) → 
  h = 12 := 
by
  -- Given conditions
  assume h : ℕ
  assume s : ℕ
  assume hc : s = 3

  -- Insert proof here
  sorry

end cuboid_height_third_dimension_l88_88796


namespace distinct_primes_in_product_l88_88957

theorem distinct_primes_in_product : 
  let primes := ({95, 97, 99, 101, 103}: finset ℕ).bUnion (λ n, (n.factorization.keys.to_finset : finset ℕ))
  primes.card = 7 := 
by {
  sorry
}

end distinct_primes_in_product_l88_88957


namespace calculate_T6_l88_88199

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + 1 / y^m

theorem calculate_T6 (y : ℝ) (h : y + 1 / y = 5) : T y 6 = 12098 := 
by
  sorry

end calculate_T6_l88_88199


namespace find_value_of_expression_l88_88892

/-- Proving the value of the expression based on given conditions in a triangle with specific geometric properties. -/
theorem find_value_of_expression 
  (A B C : Triangle) 
  (angle_A : A.ang = 60) 
  (angle_B : B.ang = 75) 
  (angle_C : C.ang = 45) 
  (H : A.orthocenter) 
  (O : A.circumcenter) 
  (F : midpoint A B) 
  (Q : foot_perpendicular B C) 
  (X : intersection (line FH) (line QO))
  (R : circumradius A)
  (FX_R_ratio : FX / R = (1 + sqrt 3) / 2) :
  1000 * 1 + 100 * 1 + 10 * 3 + 2 = 1132 := 
by
  sorry

end find_value_of_expression_l88_88892


namespace base_log_eq_l88_88718

theorem base_log_eq (x : ℝ) : (5 : ℝ)^(x + 7) = (6 : ℝ)^x → x = Real.logb (6 / 5 : ℝ) (5^7 : ℝ) := by
  sorry

end base_log_eq_l88_88718


namespace markese_earnings_l88_88651

-- Define the conditions
def earnings_relation (E M : ℕ) : Prop :=
  M = E - 5 ∧ M + E = 37

-- The theorem to prove
theorem markese_earnings (E M : ℕ) (h : earnings_relation E M) : M = 16 :=
by
  sorry

end markese_earnings_l88_88651


namespace zero_point_in_interval_23_l88_88941

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem zero_point_in_interval_23 :
  ∃ c ∈ Icc 2 3, f c = 0 :=
sorry

end zero_point_in_interval_23_l88_88941


namespace min_value_expression_l88_88870

theorem min_value_expression (a b : ℝ) (h : a ≠ 0) : 
  ∃ (m : ℝ), m = Real.sqrt(8 / 3) ∧ (∀ (x y : ℝ), x ≠ 0 → (1 / x^2 + 2 * x^2 + 3 * y^2 + 4 * x * y) ≥ Real.sqrt(8 / 3)) :=
by 
  use Real.sqrt(8 / 3)
  split
  -- Here we would provide the proof that the minimum value is sqrt(8 / 3)
  sorry

  -- Here we would provide the proof that for all x and y, the expression is greater or equal to sqrt(8 / 3)
  sorry

end min_value_expression_l88_88870


namespace length_of_DC_l88_88060

variables {A B C D O : Type} [parallelogram A B C D]
  (AB CD AD BC : ℝ)
  (angle_BCD angle_CDA : ℝ)
  (ratio_AO_OC : ℝ)

-- Conditions and given values
def parallelogram_property_1 : AB = 6 := sorry
def parallelogram_property_2 : BC = 4 * real.sqrt 3 := sorry
def parallelogram_property_3 : angle_BCD = 60 := sorry
def parallelogram_property_4 : angle_CDA = 45 := sorry
def diagonal_ratio : ratio_AO_OC = 2 / 1 := sorry

-- Proof statement
theorem length_of_DC : CD = 6 :=
  sorry

end length_of_DC_l88_88060


namespace area_of_rotated_polygon_l88_88081

variable (A : Type) [ConvexPolygon A] (S : ℝ) (M : Point) (α : ℝ)

theorem area_of_rotated_polygon (A : ConvexPolygon A) (S : ℝ) (M : Point) (α : ℝ) :
  (area (rotate_polygon A M α ˢ A) = 4 * (sin (α / 2))^2 * S) :=
by sorry

end area_of_rotated_polygon_l88_88081


namespace perfect_squares_divisors_360_l88_88540

open Nat

noncomputable def factorial_product : ℕ := fact 2 * fact 4 * fact 6 * fact 8 * fact 10

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def num_perfect_square_divisors (n : ℕ) : ℕ :=
  (factors n).foldr (λ p c, 
    let exp := count p (factors n);
    if exp % 2 == 0 then 
      c * ((exp / 2) + 1) 
    else 
      c * (((exp + 1) / 2) + 1)
  ) 1

theorem perfect_squares_divisors_360 : num_perfect_square_divisors factorial_product = 360 :=
sorry


end perfect_squares_divisors_360_l88_88540


namespace Robert_ate_chocolates_l88_88216

theorem Robert_ate_chocolates (n r : ℕ) (h1 : n = 4) (h2 : r = n + 9) : r = 13 :=
by
  rw [h1] at h2
  rw [h2]
  rw [Nat.add_comm]   -- Optional, just to show commutativity
  refl

end Robert_ate_chocolates_l88_88216


namespace exponents_geom_progression_l88_88456

open Nat

-- Define a condition for a sequence to be a geometric progression
def is_geometric_progression (seq : List ℕ) : Prop :=
  ∃ r : ℚ, ∀ i, 0 < i ∧ i < seq.length → 
    seq.nth_le i sorry / seq.nth_le (i - 1) sorry = r

-- Define the exponents of the prime factorization of n! 
def prime_factorization_exponents (n : ℕ) (p : ℕ) : ℕ :=
  if p.prime then 
    List.sum ((List.range (n / p + 1)).map (λ k => n / (p ^ k)))
  else 0

-- The main theorem statement
theorem exponents_geom_progression (n : ℕ) (h : n ≥ 3) :
  ∃ m : ℕ, m = n ∧
  (let exponents := (List.range (n + 1)).filter prime.subtype.map (prime_factorization_exponents n) 
  in is_geometric_progression exponents) ↔ 
  (n = 3 ∨ n = 6 ∨ n = 10) := 
by {
  sorry
}

end exponents_geom_progression_l88_88456


namespace chips_swap_cannot_reverse_order_l88_88345

def swap (pos1 pos2 : ℕ) (chips : List ℕ) : List ℕ :=
  if pos1 < chips.length ∧ pos2 < chips.length ∧ abs (pos1 - pos2) = 2 then
    let chip1 := chips.get! pos1
    let chip2 := chips.get! pos2
    chips.set! pos1 chip2 |>.set! pos2 chip1
  else
    chips

theorem chips_swap_cannot_reverse_order :
  ¬ ∃ (chips : List ℕ), chips.length = 100 ∧
    (∀ (pos1 pos2 : ℕ), abs (pos1 - pos2) = 2 → swap pos1 pos2 chips = reverse chips) := sorry

end chips_swap_cannot_reverse_order_l88_88345


namespace range_of_f_l88_88523

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * Real.sin x - Real.cos x

theorem range_of_f :
  ∀ x : ℝ, x ∈ Icc (- Real.pi / 2) (Real.pi / 2) → f x ∈ Icc (-2 : ℝ) (sqrt 3) := by
sorry

end range_of_f_l88_88523


namespace FH_bisects_BC_l88_88504

variables (A B C H F : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty H] [Nonempty F]
noncomputable theory

-- Assume points A, B, and C are vertices of a triangle with orthocenter H
variables [IsOrthocenter A B C H]

-- Assume circle with diameter AH intersects the circumcircle of triangle ABC at F
variables [CircleWithDiameterIntersectsCircumcircleOfTriangle A H (Circumcenter A B C) F]

-- Main theorem: FH bisects BC
theorem FH_bisects_BC (A B C H F : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty H] [Nonempty F]
  [IsOrthocenter A B C H] [CircleWithDiameterIntersectsCircumcircleOfTriangle A H (Circumcenter A B C) F]
  : Bisects (Segment H F) (Segment B C) :=
sorry

end FH_bisects_BC_l88_88504


namespace number_of_irrationals_is_two_l88_88820

-- Definitions of the given numbers
def a := -22 / 7
def b := Real.sqrt 5
def c := 0
def d := Real.cbrt 8
def e := -Real.pi 
def f := Real.sqrt 64
def g := 1.101001000100001

-- Statement of the proof
theorem number_of_irrationals_is_two : 
  (∃ i j k l m n o: ℝ, 
    i = a ∧ j = b ∧ k = c ∧ l = d ∧ m = e ∧ n = f ∧ o = g ∧ 
    (irrational j ∧ irrational m ∧ 
     ¬irrational i ∧ ¬irrational k ∧ ¬irrational l ∧ ¬irrational n ∧ ¬irrational o) ) 
  → 2 := 
sorry

end number_of_irrationals_is_two_l88_88820


namespace distance_foci_ellipse_l88_88436

noncomputable def distance_between_foci (a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem distance_foci_ellipse :
  ∀ (x y : ℝ), (x^2 / 36 + y^2 / 9 = 4) →
  distance_between_foci 3 (3 / 2) = 3 * real.sqrt 3 :=
by
  intros x y h
  sorry

end distance_foci_ellipse_l88_88436


namespace calculate_selling_price_l88_88203

-- Define the constants
def CP : ℝ := 540
def markupPercentage : ℝ := 15 / 100
def discountPercentage : ℝ := 19.999999999999996 / 100

-- Define the mark price and selling price calculation
def MP : ℝ := CP + (markupPercentage * CP)
def SP : ℝ := MP - (discountPercentage * MP)

-- The theorem to prove
theorem calculate_selling_price :
  SP = 496.8 := by
  sorry

end calculate_selling_price_l88_88203


namespace xiaoMingFatherAge_penTotalCost_xiaoMingFatherAge_sorry_penTotalCost_sorry_l88_88749

-- Definitions for conditions and statements
def xiaoMingAge : ℕ := 9
def fatherAge : ℕ := 5 * xiaoMingAge
def penCost : ℕ := 2
def totalPens : ℕ := 60
def totalCost : ℕ := penCost * totalPens

-- Theorem to prove
theorem xiaoMingFatherAge : fatherAge = 45 :=
by
  -- proof will be here

theorem penTotalCost : totalCost = 120 :=
by
  -- proof will be here

-- Use sorry to complete with valid Lean code.
theorem xiaoMingFatherAge_sorry : fatherAge = 45 := 
  by 
  sorry

theorem penTotalCost_sorry : totalCost = 120 :=
  by
  sorry

end xiaoMingFatherAge_penTotalCost_xiaoMingFatherAge_sorry_penTotalCost_sorry_l88_88749


namespace binomial_20_10_l88_88913

open Nat

theorem binomial_20_10 :
  (binomial 18 8 = 43758) →
  (binomial 18 9 = 48620) →
  (binomial 18 10 = 43758) →
  binomial 20 10 = 184756 :=
by
  intros h1 h2 h3
  sorry

end binomial_20_10_l88_88913


namespace problem_statement_l88_88188

noncomputable theory

def omega : ℂ := complex.exp (2 * complex.pi * complex.I / 3)

lemma omega_nonreal_root_of_unity : omega ^ 3 = 1 ∧ omega ≠ 1 := 
begin
  split,
  {
    -- proof that omega ^ 3 = 1
    sorry
  },
  {
    -- proof that omega ≠ 1
    sorry
  }
end

lemma omega_squared_equation : omega^2 + omega + 1 = 0 :=
begin
  -- proof that omega^2 + omega + 1 = 0
  sorry
end

theorem problem_statement : (2 - 3 * omega + 4 * omega^2)^3 + (3 + 2 * omega - omega^2)^3 = 1191 :=
begin
  -- proof goes here
  sorry
end

end problem_statement_l88_88188


namespace standard_equation_of_ellipse_find_range_m_l88_88499

theorem standard_equation_of_ellipse 
  (a b : ℝ) (h : 0 < b ∧ b < a) 
  (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ set_of (λ (p : ℝ × ℝ), let ⟨x, y⟩ := p in (x^2 / a^2 + y^2 / b^2 = 1)))
  (focal_distance : 2 * sqrt 3 = 2 * (sqrt (a^2 - b^2))) 
  (P : ℝ × ℝ) 
  (F1 F2 : ℝ × ℝ) 
  (F1F2_dist : F1.1 * F2.1 = 2)
  (angle_P90 : ∀ (F1 F2 P : ℝ × ℝ), ∠ (P - F1) (P - F2) = pi / 2)
  (area_triangle : let base := dist (F1) (F2) in 0.5 * base * dist P (midpoint F1 F2) = 1)
  (vertex_B : ℝ × ℝ) 
  (inner_point_M : ℝ × ℝ)
  (BMC_BMD_ratio : let BM := affine_segment ℝ vertex_B inner_point_M in ∀ (C D : ℝ × ℝ), C ≠ D → lies_on BM C → lies_on line_through inner_point_M C D) 
  : (a = 2 ∧ b^2 = 1 ∧ ∀ x y : ℝ, x^2 / 4 + y^2 = 1) :=
begin
  -- proof here
  sorry
end

theorem find_range_m 
  (a b : ℝ) (h : 0 < b ∧ b < a) 
  (vertex_B : ℝ × ℝ) 
  (inner_point_M : 0 = inner_point_M.1) 
  (BMC_BMD_ratio : ∀ (C D : ℝ × ℝ), let BM := affine_segment ℝ vertex_B inner_point_M in C ≠ D → lies_on BM C → lies_on BM D ∧ (area (triangle vertex_B inner_point_M C) / area (triangle vertex_B inner_point_M D) = 2/1))
  : (1/3 < inner_point_M.2 ∧ inner_point_M.2 < 1 ∨ -1 <  inner_point_M.2 ∧ inner_point_M.2 < -1/3) :=
begin
  -- proof here
  sorry
end

end standard_equation_of_ellipse_find_range_m_l88_88499


namespace determine_g_expression_l88_88937

theorem determine_g_expression :
  ∀ x, ∀ f : ℝ → ℝ, ∀ g : ℝ → ℝ,
    (∀ x, f(x) = 2 * x + 3) →
    (∀ x, g(x + 2) = f(x)) →
    (g(x) = 2 * x - 1) :=
by 
  intros x f g h₁ h₂
  sorry

end determine_g_expression_l88_88937


namespace value_of_d_l88_88862

noncomputable def d : ℚ :=
  let x := classical.some (exists.intro (-5) (by norm_num))
  let y := classical.some (exists.intro (1/2) (by norm_num))
  x + y

theorem value_of_d:
  (∃ x, (3 * x^2 + 10 * x - 40 = 0) ∧ (x ∈ ℤ)) →
  (∃ y, (4 * y^2 - 20 * y + 19 = 0) ∧ (0 ≤ y ∧ y < 1)) →
  d = -9/2 :=
by
  intros hx hy
  sorry

end value_of_d_l88_88862


namespace maintenance_check_days_l88_88400

theorem maintenance_check_days (x : ℝ) (hx : x + 0.20 * x = 60) : x = 50 :=
by
  -- this is where the proof would go
  sorry

end maintenance_check_days_l88_88400


namespace smallest_integer_k_distinct_real_roots_l88_88074

theorem smallest_integer_k_distinct_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, x^2 - x + 2 - k = 0 → x ≠ 0) ∧ k = 2 :=
by
  sorry

end smallest_integer_k_distinct_real_roots_l88_88074


namespace arithmetic_sequence_sum_l88_88607

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (h_arith : is_arithmetic_sequence a)
  (h_condition : a 2 + a 6 = 37) : 
  a 1 + a 3 + a 5 + a 7 = 74 :=
  sorry

end arithmetic_sequence_sum_l88_88607


namespace xy_equals_252_l88_88100

-- Definitions and conditions
variables (x y : ℕ) -- positive integers
variable (h1 : x + y = 36)
variable (h2 : 4 * x * y + 12 * x = 5 * y + 390)

-- Statement of the problem
theorem xy_equals_252 (h1 : x + y = 36) (h2 : 4 * x * y + 12 * x = 5 * y + 390) : x * y = 252 := by 
  sorry

end xy_equals_252_l88_88100


namespace trapezium_area_l88_88853

theorem trapezium_area (a b h : ℝ) (h₁ : a = 20) (h₂ : b = 16) (h₃ : h = 15) : 
  (1/2 * (a + b) * h = 270) :=
by
  rw [h₁, h₂, h₃]
  -- The following lines of code are omitted as they serve as solving this proof, and the requirement is to provide the statement only. 
  sorry

end trapezium_area_l88_88853


namespace largest_n_for_a_n_eq_2020_l88_88624

-- Define Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

-- Define a_n as the number of sets S of positive integers such that the sum of fibonacci numbers indexed by S equals n
def a_n (n : ℕ) : ℕ := sorry -- Placeholder definition

-- State the theorem for the largest n such that a_n = 2020
theorem largest_n_for_a_n_eq_2020 : ∃ n, a_n n = 2020 ∧ n = fibonacci 2022 - 1 := 
  sorry -- Proof omitted

end largest_n_for_a_n_eq_2020_l88_88624


namespace intersection_of_M_and_N_eq_interval_l88_88948

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2x ≥ 0}

theorem intersection_of_M_and_N_eq_interval :
  M ∩ N = {x | x ≥ 2} :=
sorry

end intersection_of_M_and_N_eq_interval_l88_88948


namespace solve_equation_l88_88246

theorem solve_equation : ∃ x : ℝ, 4^x - 2^(x + 1) - 3 = 0 ∧ x = Real.log 3 / Real.log 2 := by
  sorry

end solve_equation_l88_88246


namespace maximize_power_speed_l88_88695

variable (C S ρ v₀ : ℝ)

-- Given the formula for force F
def force (v : ℝ) : ℝ := (C * S * ρ * (v₀ - v)^2) / 2

-- Given the formula for power N
def power (v : ℝ) : ℝ := force C S ρ v₀ v * v

theorem maximize_power_speed : ∀ C S ρ v₀ : ℝ, ∃ v : ℝ, v = v₀ / 3 ∧ (∀ v' : ℝ, power C S ρ v₀ v ≤ power C S ρ v₀ v') :=
by
  sorry

end maximize_power_speed_l88_88695


namespace emma_goal_l88_88026

theorem emma_goal (total_quizzes : ℕ) (percent_goal : ℝ) (quizzes_taken : ℕ) (quizzes_with_A : ℕ)
  (quizzes_remaining : ℕ) (quizzes_needed_with_A : ℕ) :
  total_quizzes = 60 →
  percent_goal = 0.85 →
  quizzes_taken = 40 →
  quizzes_with_A = 30 →
  quizzes_remaining = total_quizzes - quizzes_taken →
  quizzes_needed_with_A = percent_goal * total_quizzes - quizzes_with_A →
  quizzes_needed_with_A = quizzes_remaining →
  0 = quizzes_remaining - quizzes_needed_with_A :=
begin
  intros,
  sorry
end

end emma_goal_l88_88026


namespace black_pieces_more_than_white_l88_88256

theorem black_pieces_more_than_white (B W : ℕ) 
  (h₁ : (B - 1) * 7 = 9 * W)
  (h₂ : B * 5 = 7 * (W - 1)) :
  B - W = 7 :=
sorry

end black_pieces_more_than_white_l88_88256


namespace andrew_purchase_grapes_l88_88823

theorem andrew_purchase_grapes (G : ℕ) (h : 70 * G + 495 = 1055) : G = 8 :=
by
  sorry

end andrew_purchase_grapes_l88_88823


namespace pages_after_break_l88_88178

-- Formalize the conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.70

-- Define the proof problem
theorem pages_after_break : 
  let pages_read_before_break := (break_percentage * total_pages)
  let pages_remaining := total_pages - pages_read_before_break
  pages_remaining = 9 :=
by
  sorry

end pages_after_break_l88_88178


namespace determine_a_l88_88936

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then
    sin x
  else
    x^3 - 9 * x^2 + 25 * x + a

def has_three_intersections_with_y_eq_x (a : ℝ) : Prop :=
  let h1 := ∃ x < 1, f x a = x
  let h2 := ∃ x ≥ 1, f x a = x
  h1 ∧ h2

theorem determine_a (a : ℝ) :
  has_three_intersections_with_y_eq_x a ↔ (a = -20 ∨ a = -16) :=
sorry

end determine_a_l88_88936


namespace tank_inflow_rate_l88_88259

/-- 
  Tanks A and B have the same capacity of 20 liters. Tank A has
  an inflow rate of 2 liters per hour and takes 5 hours longer to
  fill than tank B. Show that the inflow rate in tank B is 4 liters 
  per hour.
-/
theorem tank_inflow_rate (capacity : ℕ) (rate_A : ℕ) (extra_time : ℕ) (rate_B : ℕ) 
  (h1 : capacity = 20) (h2 : rate_A = 2) (h3 : extra_time = 5) (h4 : capacity / rate_A = (capacity / rate_B) + extra_time) :
  rate_B = 4 :=
sorry

end tank_inflow_rate_l88_88259


namespace eccentricity_range_l88_88088

noncomputable def ellipse_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) : Prop :=
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ e = c / a ∧ (2 * ((-a) * (c + a / 2) - (b / 2) * b) + b^2 + c^2 ≥ 0)

theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ e : ℝ, ellipse_eccentricity_range a b h e ∧ (0 < e ∧ e ≤ -1 + Real.sqrt 3) :=
sorry

end eccentricity_range_l88_88088


namespace has_max_min_no_extreme_l88_88630

noncomputable def polynomial (R : Type*) [Ring R] := R →₁₀

theorem has_max_min_no_extreme (f : polynomial ℝ) (a b : ℝ) (h : a ≤ b) :
  (∃ x ∈ Icc a b, ∀ y ∈ Icc a b, f x ≥ f y) ∧
  (∃ x ∈ Icc a b, ∀ y ∈ Icc a b, f x ≤ f y) ∧
  ¬(∃ x ∈ Icc a b, (∀ h : ℝ, DifferentiableAt ℝ f x ∧ (deriv f x = 0) ∧ (∀ h:ℝ, x > a ∧ x < b → (deriv f x > 0 → x > h → deriv f x < 0 →
  h > x)))))
:= sorry

end has_max_min_no_extreme_l88_88630


namespace max_value_of_sides_thm_l88_88098

variable (a b c : ℝ) (A B C S : ℝ)

def max_value_of_sides : Prop :=
  (a^2 + b^2 + c^2 ≤ 4)

theorem max_value_of_sides_thm
  (h1 : S = 1/2 * c^2)
  (h2 : a * b = sqrt 2) :
  max_value_of_sides a b c A B C S := by
  sorry

end max_value_of_sides_thm_l88_88098


namespace parabola_directrix_eq_l88_88440

theorem parabola_directrix_eq (x : ℝ) : 
  (∀ y : ℝ, y = 3 * x^2 - 6 * x + 2 → True) →
  y = -13/12 := 
  sorry

end parabola_directrix_eq_l88_88440


namespace find_first_term_arithmetic_progression_l88_88038

theorem find_first_term_arithmetic_progression
  (a1 a2 a3 : ℝ)
  (h1 : a1 + a2 + a3 = 12)
  (h2 : a1 * a2 * a3 = 48)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a1 < a2 ∧ a2 < a3) :
  a1 = 2 :=
by
  sorry

end find_first_term_arithmetic_progression_l88_88038


namespace island_perimeter_correct_l88_88704

noncomputable def perimeter_of_island : ℝ :=
  let base_triangle := 4.0      -- the base of the equilateral triangle in miles
  let height_triangle := 5.0    -- the height of the equilateral triangle in miles
  let side_triangle := base_triangle   -- each side of equilateral triangle (since base = side)
  let triangle_sides := 3 * side_triangle  -- total length of the triangle sides
  let radius_half_circle := base_triangle / 2  -- radius of each half circle
  let circumference_half_circle := Real.pi * radius_half_circle   -- circumference of one half circle
  let total_half_circles := 2 * circumference_half_circle  -- total length of the half circles
  triangle_sides + total_half_circles

theorem island_perimeter_correct : perimeter_of_island ≈ 24.56636 :=
  sorry

end island_perimeter_correct_l88_88704


namespace electric_power_calculation_l88_88437

-- Define the constants and the main theorem to prove
theorem electric_power_calculation :
  ∀ (k_star : ℝ) (e_tau : ℝ) (N_H : ℝ), 
    k_star = 1 / 3 → 
    e_tau = 0.15 → 
    N_H = 80 * (1) → 
    k_star * e_tau * N_H = 4 :=
by
  intros k_star e_tau N_H hk he hNH
  rw [hk, he, hNH]
  simp
  norm_num
  sorry

end electric_power_calculation_l88_88437


namespace probability_zero_units_digit_seven_divisible_by_three_l88_88453

theorem probability_zero_units_digit_seven_divisible_by_three :
  ∀ (a b : ℕ), a ∈ (Finset.range 21).image (+1) → b ∈ (Finset.range 21).image (+1) →
  (2^a + 5^b) % 10 = 7 → (2^a + 5^b) % 3 = 0 → false :=
by
  intros a b ha hb units_digit_seven divisible_by_three
  sorry

end probability_zero_units_digit_seven_divisible_by_three_l88_88453


namespace reduced_rate_fraction_l88_88335

-- Definitions
def hours_in_a_week := 7 * 24
def hours_with_reduced_rates_on_weekdays := (12 * 5)
def hours_with_reduced_rates_on_weekends := (24 * 2)

-- Question in form of theorem
theorem reduced_rate_fraction :
  (hours_with_reduced_rates_on_weekdays + hours_with_reduced_rates_on_weekends) / hours_in_a_week = 9 / 14 := 
by
  sorry

end reduced_rate_fraction_l88_88335


namespace find_angle_l88_88121

open Real EuclideanSpace

variables {E : Type*} [InnerProductSpace ℝ E] (a b : E)

-- Conditions
def magnitude_a := ‖a‖ = Real.sqrt 2
def magnitude_b := ‖b‖ = 2
def perp_condition := inner (a - b) a = 0

-- Theorem statement
theorem find_angle (h1 : magnitude_a a) (h2 : magnitude_b b) (h3 : perp_condition a b) :
  ∠ a b = π / 4 :=
sorry

end find_angle_l88_88121


namespace maximize_power_speed_l88_88696

variable (C S ρ v₀ : ℝ)

-- Given the formula for force F
def force (v : ℝ) : ℝ := (C * S * ρ * (v₀ - v)^2) / 2

-- Given the formula for power N
def power (v : ℝ) : ℝ := force C S ρ v₀ v * v

theorem maximize_power_speed : ∀ C S ρ v₀ : ℝ, ∃ v : ℝ, v = v₀ / 3 ∧ (∀ v' : ℝ, power C S ρ v₀ v ≤ power C S ρ v₀ v') :=
by
  sorry

end maximize_power_speed_l88_88696


namespace variance_of_data_l88_88722

def data : List ℝ := [0.7, 1, 0.8, 0.9, 1.1]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.foldr (λ x acc => x + acc) 0) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.foldr (λ x acc => (x - m) ^ 2 + acc) 0) / l.length

theorem variance_of_data :
  variance data = 0.02 :=
by
  sorry

end variance_of_data_l88_88722


namespace train_length_l88_88391

theorem train_length (speed_kph : ℕ) (tunnel_length_m : ℕ) (time_s : ℕ) : 
  speed_kph = 54 → 
  tunnel_length_m = 1200 → 
  time_s = 100 → 
  ∃ train_length_m : ℕ, train_length_m = 300 := 
by
  intros h1 h2 h3
  have speed_mps : ℕ := (speed_kph * 1000) / 3600 
  have total_distance_m : ℕ := speed_mps * time_s
  have train_length_m : ℕ := total_distance_m - tunnel_length_m
  use train_length_m
  sorry

end train_length_l88_88391


namespace heather_total_distance_l88_88955

-- Definitions for distances walked
def distance_car_to_entrance : ℝ := 0.33
def distance_entrance_to_rides : ℝ := 0.33
def distance_rides_to_car : ℝ := 0.08

-- Statement of the problem to be proven
theorem heather_total_distance :
  distance_car_to_entrance + distance_entrance_to_rides + distance_rides_to_car = 0.74 :=
by
  sorry

end heather_total_distance_l88_88955


namespace prob_1_less_X_less_2_l88_88890

noncomputable def NormalDist (mean variance : ℝ) : Type := sorry -- Placeholder for normal distribution type

variable (X : NormalDist 1 4)

axiom prob_X_less_than_2 : P(X < 2) = 0.72

theorem prob_1_less_X_less_2 : P(1 < X < 2) = 0.22 :=
by
  have h1 : P(X ≥ 2) = 1 - P(X < 2) := sorry
  have h2 : P(X > 1) = 0.5 := sorry
  have h3 : P(1 < X < 2) = P(X > 1) - P(X ≥ 2) := sorry
  show P(1 < X < 2) = 0.22
sorry

end prob_1_less_X_less_2_l88_88890


namespace find_a_l88_88092

theorem find_a (a α : Real) (h1 : sin α + cos α = 2 / 3)
  (h2 : sin α * cos α = a / 3) :
  a = -5 / 6 := 
  sorry

end find_a_l88_88092


namespace binomial_20_10_l88_88911

open Nat

theorem binomial_20_10 :
  (binomial 18 8 = 43758) →
  (binomial 18 9 = 48620) →
  (binomial 18 10 = 43758) →
  binomial 20 10 = 184756 :=
by
  intros h1 h2 h3
  sorry

end binomial_20_10_l88_88911


namespace arithmetic_sequence_b_sum_reciprocal_S_n_sum_T_n_bound_l88_88083

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 2 else 2 - 1 / (sequence_a (n - 1))

def sequence_b (n : ℕ) : ℚ :=
  1 / (sequence_a n - 1)

def S_n (n : ℕ) : ℚ :=
  (finset.range n).sum (λ i, 1 / 3 * sequence_b (i + 1))

def T_n (n : ℕ) : ℚ :=
  (finset.range n).sum (λ i, (1 / 3) ^ (i + 1) * sequence_b (i + 1))

theorem arithmetic_sequence_b :
  ∀ n : ℕ, sequence_b (n + 1) - sequence_b n = 1 := sorry

theorem sum_reciprocal_S_n (n : ℕ) :
  (finset.range n).sum (λ i, 1 / S_n (i + 1)) = 6 * n / (n + 1) := sorry

theorem sum_T_n_bound (n : ℕ) :
  T_n n < 3 / 4 := sorry

end arithmetic_sequence_b_sum_reciprocal_S_n_sum_T_n_bound_l88_88083


namespace value_of_c_l88_88541

noncomputable def mixed_to_improper (a b c : Int) := (a * c + b : Int) / (c : Int)

def only_one_integer_point (x1 y1 x2 y2 : Rat) : Prop :=
  ∃x y : Rat, (x, y) = (x1, y1) ∧ (x, y) = (x2, y2)

theorem value_of_c (c : ℚ) (H : only_one_integer_point 22 (mixed_to_improper 12 2 3) c (mixed_to_improper 17 2 3)) :
  c = 23 :=
by
  intro h
  sorry

end value_of_c_l88_88541


namespace three_heads_in_a_row_l88_88309

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l88_88309


namespace range_of_m_l88_88108

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  sqrt 3 * sin (ω * x) * cos (ω * x) +
  cos (ω * x) ^ 2 - 1 / 2

def smallest_period_of_f (ω : ℝ) : Prop :=
  ∃ T > 0, ∀ x, f (x + T) ω = f x ω ∧ T = π / 2

def transformed_function (x : ℝ) : ℝ := sin x

def has_exactly_one_solution (m : ℝ) : Prop :=
  ∃! x ∈ set.Icc 0 (5 * π / 6), transformed_function x + m = 0

theorem range_of_m :
  ∀ m : ℝ,
  (has_exactly_one_solution m ↔ m ∈ set.Icc (-1/2) 0 ∪ set.Icc (-1) (-1)) :=
sorry

end range_of_m_l88_88108


namespace part_a_exists_part_b_not_exists_l88_88431

theorem part_a_exists :
  ∃ (a b : ℤ), (∀ x : ℝ, x^2 + a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a*x + b = 0) :=
sorry

theorem part_b_not_exists :
  ¬ ∃ (a b : ℤ), (∀ x : ℝ, x^2 + 2*a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + 2*a*x + b = 0) :=
sorry

end part_a_exists_part_b_not_exists_l88_88431


namespace three_heads_in_a_row_l88_88306

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l88_88306


namespace soccer_tournament_l88_88566

theorem soccer_tournament (teams : Finset ℕ) (matches : Finset (ℕ × ℕ))
  (h_card_teams : teams.card = 18)
  (h_match_condition : ∀ t ∈ teams, (matches.filter (λ m, m.1 = t ∨ m.2 = t)).card = 8) :
  ∃ (t1 t2 t3 : ℕ), t1 ∈ teams ∧ t2 ∈ teams ∧ t3 ∈ teams ∧
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t1 ∧ t2 ≠ t3 ∧
  ¬ (matches ∈ ({(t1, t2), (t1, t3), (t2, t3)} : Finset (ℕ × ℕ)) ∨ matches ∈ ({(t2, t1), (t3, t1), (t3, t2)} : Finset (ℕ × ℕ))) :=
by sorry

end soccer_tournament_l88_88566


namespace beyonce_total_songs_l88_88030

theorem beyonce_total_songs (s a b t : ℕ) (h_s : s = 5) (h_a : a = 2 * 15) (h_b : b = 20) (h_t : t = s + a + b) : t = 55 := by
  rw [h_s, h_a, h_b] at h_t
  exact h_t

end beyonce_total_songs_l88_88030


namespace range_of_k_l88_88161

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 9

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k * x + 3

-- Define the distance function from a point (x1, y1) to a line ax + by + c = 0
def distance_point_to_line (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

-- Define the line l in the form ax + by + c = 0
def line_l_reduced (k x : ℝ) : ℝ := k * x - 1 * x + 3

-- Define the midpoint condition
def midpoint_distance_condition (k : ℝ) : Prop :=
  (distance_point_to_line 1 1 k (-1) 3 + 2) ≥ 3

-- State the main theorem about the range of k
theorem range_of_k (k : ℝ) : midpoint_distance_condition k ↔ k ≥ -3 / 4 :=
by
  sorry

end range_of_k_l88_88161


namespace p_q_r_cubic_sum_l88_88418

theorem p_q_r_cubic_sum (p q r : ℚ) (h1 : p + q + r = 4) (h2 : p * q + p * r + q * r = 6) (h3 : p * q * r = -8) : 
  p^3 + q^3 + r^3 = 8 := by
  sorry

end p_q_r_cubic_sum_l88_88418


namespace bacteria_cells_count_l88_88795

def initial_cells : ℕ := 5
def division_rate : ℕ := 3
def time_period : ℕ := 15
def time_interval : ℕ := 3
def expected_cells : ℕ := 1215

theorem bacteria_cells_count :
  ∀ (n : ℕ), n = time_period / time_interval + 1 → 
  let a := initial_cells in 
  let r := division_rate in 
  let a_n := a * r ^ (n - 1) in
  a_n = expected_cells := 
by
  intros n hn, 
  let a := initial_cells,
  let r := division_rate,
  let a_n := a * r ^ (n - 1),
  sorry

end bacteria_cells_count_l88_88795


namespace probability_1_lt_X_lt_2_l88_88887

noncomputable theory

-- Assume X is a random variable following normal distribution N(1, 4)
def X : ProbabilityTheory.ProbMeasure ℝ := 
  ProbabilityTheory.ProbMeasure.normal 1 (real.sqrt 4)  -- N(mean=1, std=sqrt(variance))

-- Given P(X < 2) = 0.72
axiom P_X_less_than_2 : ProbabilityTheory.ProbMeasure.cdf X 2 = 0.72

-- The theorem to be proven
theorem probability_1_lt_X_lt_2 : 
  ProbabilityTheory.ProbMeasure.prob (set.Ioo 1 2) = 0.22 :=
sorry

end probability_1_lt_X_lt_2_l88_88887


namespace binomial_20_10_l88_88914

open Nat

theorem binomial_20_10 :
  (binomial 18 8 = 43758) →
  (binomial 18 9 = 48620) →
  (binomial 18 10 = 43758) →
  binomial 20 10 = 184756 :=
by
  intros h1 h2 h3
  sorry

end binomial_20_10_l88_88914


namespace angle_bisector_length_l88_88502

/-- In a right triangle DEF, where DE is 6 and EF is 8,
     the length of the angle bisector from E to the hypotenuse DF is 12√6/7. -/
theorem angle_bisector_length (DE EF : ℝ) (h1 : DE = 6) (h2 : EF = 8) :
  let DF := Real.sqrt (DE^2 + EF^2),
      DQ := (3/7) * DF,
      EQ := (DE^2 - DQ^2)^0.5
  in EQ = 12 * Real.sqrt 6 / 7 :=
by
  -- leaving proof as a placeholder
  sorry

end angle_bisector_length_l88_88502


namespace find_z_find_range_a_l88_88930

-- Problem 1: Prove the value of z
theorem find_z (b : ℝ) (hb : (b - 2 * (0 : ℂ).im) / (2 - (0 : ℂ).im) ∈ ℝ) : b = 4 ∧ b - 2 * (0 : ℂ).im = 4 - 2 * (0 : ℂ).im :=
by sorry

-- Problem 2: Find the range of a such that the point corresponding to the complex number (z + a * i) ^ 2 is in the fourth quadrant, given z = 4 - 2 * i.

theorem find_range_a (a : ℝ) (hz : (4 - 2 * (0 : ℂ).im) + a * (0 : ℂ).i) (h_in_fourth : (16 - (a - 2) ^ 2 > 0) ∧ (8 * (a - 2) < 0)) : -2 < a ∧ a < 2 :=
by sorry

end find_z_find_range_a_l88_88930


namespace probability_three_heads_l88_88288

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l88_88288


namespace find_a_value_l88_88457

noncomputable def M (I : Set ℝ) := Sup (Set.Image (fun x => Real.sin x) I)

theorem find_a_value (a : ℝ) (h : 0 < a) (h_max : M (Set.Icc 0 a) = 2 * M (Set.Icc a (2 * a))):
  a = (5 / 6) * Real.pi ∨ a = (13 / 12) * Real.pi :=
  sorry

end find_a_value_l88_88457


namespace sum_of_coefficients_l88_88719

theorem sum_of_coefficients (a : ℤ) (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (a + x) * (1 + x) ^ 4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_3 + a_5 = 32 →
  a = 3 :=
by sorry

end sum_of_coefficients_l88_88719


namespace numberOfHandshakes_is_correct_l88_88731

noncomputable def numberOfHandshakes : ℕ :=
  let gremlins := 30
  let imps := 20
  let friendlyImps := 5
  let gremlinHandshakes := gremlins * (gremlins - 1) / 2
  let impGremlinHandshakes := imps * gremlins
  let friendlyImpHandshakes := friendlyImps * (friendlyImps - 1) / 2
  gremlinHandshakes + impGremlinHandshakes + friendlyImpHandshakes

theorem numberOfHandshakes_is_correct : numberOfHandshakes = 1045 := by
  sorry

end numberOfHandshakes_is_correct_l88_88731


namespace min_value_f_l88_88880

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1)^2 + (Real.exp (-x) - 1)^2

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 :=
sorry

end min_value_f_l88_88880


namespace intersect_perpendicular_lines_on_AD_l88_88342

variables (A B C D M E : Point)
variables [rectangle A B C D]
variables [midpoint M C D]
variables [perpendicular_line CE BM]
variables [perpendicular_line EM BD]

theorem intersect_perpendicular_lines_on_AD :
  intersect_on_line CE EM AD := sorry

end intersect_perpendicular_lines_on_AD_l88_88342


namespace solution_l88_88191

noncomputable def problem_statement (x y z : ℝ) : Prop :=
  (cos x + cos y + cos z = 3) ∧ (sin x + sin y + sin z = 0) → (cos (2 * x) + cos (2 * y) + cos (2 * z) = 0)

theorem solution : ∀ x y z : ℝ, problem_statement x y z :=
by
  intros x y z
  unfold problem_statement
  intros h
  sorry

end solution_l88_88191


namespace sum_two_digit_integers_square_ends_with_09_l88_88740

theorem sum_two_digit_integers_square_ends_with_09 :
  let ns := {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n ^ 2) % 100 = 9} in
  (∑ n in ns, n) = 100 :=
by
  sorry

end sum_two_digit_integers_square_ends_with_09_l88_88740


namespace financing_term_years_l88_88954

def monthly_payment : Int := 150
def total_financed_amount : Int := 9000

theorem financing_term_years : 
  (total_financed_amount / monthly_payment) / 12 = 5 := 
by
  sorry

end financing_term_years_l88_88954


namespace coordinate_fifth_point_l88_88768

theorem coordinate_fifth_point : 
  ∃ (a : Fin 16 → ℝ), 
    a 0 = 2 ∧ 
    a 15 = 47 ∧ 
    (∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) ∧ 
    a 4 = 14 := 
sorry

end coordinate_fifth_point_l88_88768


namespace original_number_of_people_l88_88663

theorem original_number_of_people (x : ℕ) (h1 : 3 ∣ x) (h2 : 6 ∣ x) (h3 : (x / 2) = 18) : x = 36 :=
by
  sorry

end original_number_of_people_l88_88663


namespace proof_problem_theorem_l88_88585

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l88_88585


namespace correct_polynomials_are_l88_88852

noncomputable def polynomial_solution (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, p.eval (x^2) = (p.eval x) * (p.eval (x - 1))

theorem correct_polynomials_are (p : Polynomial ℝ) :
  polynomial_solution p ↔ ∃ n : ℕ, p = (Polynomial.C (1 : ℝ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℝ) * Polynomial.X + Polynomial.C (1 : ℝ)) ^ n :=
by
  sorry

end correct_polynomials_are_l88_88852


namespace find_speeds_l88_88475

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l88_88475


namespace range_of_f_l88_88244

def f (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem range_of_f : (finset.image f {-1, 0, 1, 2, 3}) = {1, 2, 5} :=
by
  sorry

end range_of_f_l88_88244


namespace dad_strawberries_weight_proof_l88_88645

/-
Conditions:
1. total_weight (the combined weight of Marco's and his dad's strawberries) is 23 pounds.
2. marco_weight (the weight of Marco's strawberries) is 14 pounds.
We need to prove that dad_weight (the weight of dad's strawberries) is 9 pounds.
-/

def total_weight : ℕ := 23
def marco_weight : ℕ := 14

def dad_weight : ℕ := total_weight - marco_weight

theorem dad_strawberries_weight_proof : dad_weight = 9 := by
  sorry

end dad_strawberries_weight_proof_l88_88645


namespace range_of_g_l88_88508

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^m

theorem range_of_g (m : ℝ) (h : m > 0) : set.range (λ x, g x m) = set.Ioc 0 1 := by
  sorry

end range_of_g_l88_88508


namespace sin_70_eq_1_minus_2k_squared_l88_88482

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : sin (10 * real.pi / 180) = k) :
  sin (70 * real.pi / 180) = 1 - 2 * k^2 :=
sorry

end sin_70_eq_1_minus_2k_squared_l88_88482


namespace arrange_TOOTH_l88_88429

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem arrange_TOOTH : 
  let total_permutations := factorial 5
  let repeat_T := factorial 3
  let repeat_O := factorial 2
  let corrected_permutations := total_permutations / (repeat_T * repeat_O)
  corrected_permutations = 10 := by
    simp [factorial]
    sorry

end arrange_TOOTH_l88_88429


namespace intersection_first_quadrant_l88_88146

theorem intersection_first_quadrant (a : ℝ) : 
  (∃ x y : ℝ, (ax + y = 4) ∧ (x - y = 2) ∧ (0 < x) ∧ (0 < y)) ↔ (-1 < a ∧ a < 2) :=
by
  sorry

end intersection_first_quadrant_l88_88146


namespace Paul_age_in_twelve_years_l88_88830

-- Definitions based on the conditions provided
def Christian_age (Brian_age : ℝ) := 3.5 * Brian_age
def Brian_age (years_from_now : ℝ) := 45 - years_from_now
def Margaret_age (Brian_age : ℝ) := 2 * Brian_age
def Margaret_age2 (Christian_age : ℝ) := Christian_age - 15
def Paul_age (Margaret_age : ℝ) (Christian_age : ℝ) := (Margaret_age + Christian_age) / 2
def Paul_future_age (Paul_age : ℝ) (years_from_now : ℝ) := Paul_age + years_from_now

-- The main theorem to prove Paul's age in twelve years
theorem Paul_age_in_twelve_years 
    (Brian_current_age : ℝ)
    (Brian_current_age = 33)
    (Christian_current_age = 3.5 * Brian_current_age) 
    (Margaret_current_age = 2 * Brian_current_age)
    (Paul_current_age = (2 * Brian_current_age + 3.5 * Brian_current_age) / 2) : 
    Paul_future_age Paul_current_age 12 = 102.75 :=
by
  -- replace sorry with the actual proof
  sorry

end Paul_age_in_twelve_years_l88_88830


namespace math_mpt_proof_l88_88171

noncomputable def problem_statement : ℕ :=
  let X := (0 : ℝ, 0 : ℝ)
  let Y := (1 : ℝ, 0 : ℝ)
  let Z := (1 / 2 : ℝ, (sqrt 3) / 2 : ℝ) -- coordinates derived from equilateral properties
  let N := ((1 / 2 : ℝ), (0 : ℝ)) -- Midpoint of XY
  -- Find T on XZ and B such that given conditions are met
  -- Skipping the geometric constructions, assumptions directly

  sorry
  -- The result should ultimately yield 12, as asked in the problem.

theorem math_mpt_proof {p q : ℕ} (hpq_coprime : nat.coprime p q)
  (h_length : (5 * sqrt 3 + 1) / 6 = (p : ℝ) / q) : p + q = 12 :=
begin
  sorry,
end

end math_mpt_proof_l88_88171


namespace compare_abc_l88_88079

theorem compare_abc (a b c : ℝ) (h_a : a = (4 - log 4) / exp 2)
                                  (h_b : b = log 2 / 2)
                                  (h_c : c = 1 / exp 1) :
  b < a ∧ a < c :=
by
  sorry

end compare_abc_l88_88079


namespace train_A_time_l88_88735

-- Variables for train speeds and distances
variables (v_A v_B : ℝ)
-- Distance of the route
def route := 75
-- Train B's time to complete the trip
def time_B := 2
-- Distance traveled by Train A when they meet
def distance_A_meet := 30
-- Distance traveled by Train B when they meet
def distance_B_meet := route - distance_A_meet

-- Speed of Train B
def speed_B := (75 / 2 : ℝ)
-- Time to meet
def time_to_meet := distance_B_meet / speed_B
-- Speed of Train A
def speed_A := distance_A_meet / time_to_meet
-- Time for Train A to complete the trip
def time_A := route / speed_A

-- Theorem statement
theorem train_A_time : time_A = 3 := by
  sorry

end train_A_time_l88_88735


namespace semi_minor_axis_of_ellipse_l88_88155

def distance (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem semi_minor_axis_of_ellipse 
  (center focus semi_major_end: ℝ × ℝ)
  (h1 : center = (2, -4))
  (h2 : focus = (2, -6))
  (h3 : semi_major_end = (2, -1)) :
  let a := distance center semi_major_end in
  let c := distance center focus in
  ∃ b, b = real.sqrt (a^2 - c^2) ∧ b = real.sqrt 5 :=
by 
  sorry

end semi_minor_axis_of_ellipse_l88_88155


namespace beyonce_total_songs_l88_88029

theorem beyonce_total_songs (s a b t : ℕ) (h_s : s = 5) (h_a : a = 2 * 15) (h_b : b = 20) (h_t : t = s + a + b) : t = 55 := by
  rw [h_s, h_a, h_b] at h_t
  exact h_t

end beyonce_total_songs_l88_88029


namespace max_product_is_2331_l88_88268

open Nat

noncomputable def max_product (a b : ℕ) : ℕ :=
  if a + b = 100 ∧ a % 5 = 2 ∧ b % 6 = 3 then a * b else 0

theorem max_product_is_2331 (a b : ℕ) (h_sum : a + b = 100) (h_mod_a : a % 5 = 2) (h_mod_b : b % 6 = 3) :
  max_product a b = 2331 :=
  sorry

end max_product_is_2331_l88_88268


namespace radius_of_inscribed_sphere_l88_88388

theorem radius_of_inscribed_sphere 
  (ρₐ ρ_b ρ_c ρ_d : ℝ)
  (hₐ : ρₐ = 9)
  (h_b : ρ_b = 12)
  (h_c : ρ_c = 36)
  (h_d : ρ_d = 39) :
  ∃ ρ : ℝ, ρ = 48 := 
by
  use 48
  sorry

end radius_of_inscribed_sphere_l88_88388


namespace quarters_fit_l88_88382

def table_radius := 12

def quarter_radius := 0.955

def spacing_radius := 2 * quarter_radius

def effective_radius := table_radius - quarter_radius

def subtended_angle(n : ℕ) : Real := 360 / n

theorem quarters_fit (n : ℕ) : (spacing_radius + quarter_radius) * 2 * Math.sin (subtended_angle n / 2) ≤ effective_radius → n = 10 :=
by
  -- proof will go here
  sorry

end quarters_fit_l88_88382


namespace base8_to_base10_l88_88274

theorem base8_to_base10 (n : ℕ) (h : n = 23456) : 
  let d0 := 6 in 
  let d1 := 5 in 
  let d2 := 4 in 
  let d3 := 3 in
  let d4 := 2 in
  let b := 8 in
  d0 * b^0 + d1 * b^1 + d2 * b^2 + d3 * b^3 + d4 * b^4 = 5934 := 
by 
  unfold d0 d1 d2 d3 d4 b 
  sorry

end base8_to_base10_l88_88274


namespace min_distance_eq_sqrt2_l88_88104

open Real

variables {P Q : ℝ × ℝ}
variables {x y : ℝ}

/-- Given that point P is on the curve y = e^x and point Q is on the curve y = ln x, prove that the minimum value of the distance |PQ| is sqrt(2). -/
theorem min_distance_eq_sqrt2 : 
  (P.2 = exp P.1) ∧ (Q.2 = log Q.1) → (dist P Q) = sqrt 2 :=
by
  sorry

end min_distance_eq_sqrt2_l88_88104


namespace minimum_omega_l88_88113

/-- Given function f and its properties, determine the minimum valid ω. -/
theorem minimum_omega {f : ℝ → ℝ} 
  (Hf : ∀ x : ℝ, f x = (1 / 2) * Real.cos (ω * x + φ) + 1)
  (Hsymmetry : ∃ k : ℤ, ω * (π / 3) + φ = k * π)
  (Hvalue : ∃ n : ℤ, f (π / 12) = 1 ∧ ω * (π / 12) + φ = n * π + π / 2)
  (Hpos : ω > 0) : ω = 2 := 
sorry

end minimum_omega_l88_88113


namespace distinct_four_digit_numbers_product_18_l88_88981

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l88_88981


namespace distinct_four_digit_positive_integers_product_18_l88_88974

theorem distinct_four_digit_positive_integers_product_18 :
  Finset.card {n | ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
                             (1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) ∧
                             a * b * c * d = 18} = 24 :=
by
  sorry

end distinct_four_digit_positive_integers_product_18_l88_88974


namespace wheel_revolutions_l88_88816

theorem wheel_revolutions (d : ℝ) (initial_offset : ℝ) (distance_miles : ℝ) :
  d = 8 → initial_offset = 2 → distance_miles = 2 → 
  (1 + (distance_miles * 5280 - (d * π - 2)) / (d * π)) = 10562 / (8 * π) :=
by
  intro hd hio hdm
  rw [hd, hio, hdm]
  sorry

end wheel_revolutions_l88_88816


namespace tom_cheaper_than_jane_l88_88734

-- Define constants for Store A
def store_a_full_price : ℝ := 125
def store_a_discount_one : ℝ := 0.08
def store_a_discount_two : ℝ := 0.12
def store_a_tax : ℝ := 0.07

-- Define constants for Store B
def store_b_full_price : ℝ := 130
def store_b_discount_one : ℝ := 0.10
def store_b_discount_three : ℝ := 0.15
def store_b_tax : ℝ := 0.05

-- Define the number of smartphones bought by Tom and Jane
def tom_quantity : ℕ := 2
def jane_quantity : ℕ := 3

-- Define the final amount Tom pays
def final_amount_tom : ℝ :=
  let full_price := tom_quantity * store_a_full_price
  let discount := store_a_discount_two * full_price
  let discounted_price := full_price - discount
  let tax := store_a_tax * discounted_price
  discounted_price + tax

-- Define the final amount Jane pays
def final_amount_jane : ℝ :=
  let full_price := jane_quantity * store_b_full_price
  let discount := store_b_discount_three * full_price
  let discounted_price := full_price - discount
  let tax := store_b_tax * discounted_price
  discounted_price + tax

-- Prove that Tom's total cost is $112.68 cheaper than Jane's total cost
theorem tom_cheaper_than_jane : final_amount_jane - final_amount_tom = 112.68 :=
by
  have tom := final_amount_tom
  have jane := final_amount_jane
  sorry

end tom_cheaper_than_jane_l88_88734


namespace ratio_of_girls_participated_to_total_l88_88242

noncomputable def ratio_participating_girls {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ℚ :=
    girls_participated / (girls_participated + boys_participated)

theorem ratio_of_girls_participated_to_total {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ratio_participating_girls h1 equal_boys_girls girls_participated boys_participated = 9 / 17 :=
by
    sorry

end ratio_of_girls_participated_to_total_l88_88242


namespace best_pit_numbers_l88_88777

-- Definitions based on given conditions
def total_distance_walked (n : ℕ) (x : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, (abs (i + 1 - x) * 10 : ℝ))

-- Prove the two best pit numbers to minimize the total distance is 10 and 11
theorem best_pit_numbers (n : ℕ) (h1 : n = 20) :
  ∃ x y, x = 10 ∧ y = 11 ∧ total_distance_walked n x = total_distance_walked n y :=
by
  -- The proof part is not required, so we add sorry.
  sorry

end best_pit_numbers_l88_88777


namespace markese_earnings_16_l88_88654

theorem markese_earnings_16 (E M : ℕ) (h1 : M = E - 5) (h2 : E + M = 37) : M = 16 :=
by
  sorry

end markese_earnings_16_l88_88654


namespace problem_l88_88626

noncomputable def y1 : ℝ := 4^0.9
noncomputable def y2 : ℝ := Real.logb (1/2) 5
noncomputable def y3 : ℝ := (1/2)^(-1.5)

theorem problem :
  y1 > y3 ∧ y3 > y2 :=
by
  sorry

end problem_l88_88626


namespace log_inequality_l88_88635

theorem log_inequality (b : ℝ) (h : b ≥ 1) : 
  ∀ x : ℝ, 0 ≤ x → x ≤ 1 → 2 * log 2 (2 * x + b) ≥ log 2 (x + 1) :=
by 
  intro x hx0 hx1
  have h₀ := Real.nonneg_of_log_ge_zero (2 * x + b)
  have h₁ := Real.nonneg_of_log_ge_zero (x + 1)
  sorry

end log_inequality_l88_88635


namespace term_with_largest_binomial_coeffs_and_largest_coefficient_l88_88519

theorem term_with_largest_binomial_coeffs_and_largest_coefficient :
  ∀ x : ℝ,
    (∀ k : ℕ, k = 2 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 90 * x ^ 6) ∧
    (∀ k : ℕ, k = 3 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 270 * x ^ (22 / 3)) ∧
    (∀ r : ℕ, r = 4 → (Nat.choose 5 4) * (x ^ (2 / 3)) ^ (5 - 4) * (3 * x ^ 2) ^ 4 = 405 * x ^ (26 / 3)) :=
by sorry

end term_with_largest_binomial_coeffs_and_largest_coefficient_l88_88519


namespace delta_y_over_delta_x_l88_88112

def f (x : ℝ) : ℝ := 2 * x^2 + 1

theorem delta_y_over_delta_x (Δx : ℝ) :
  let Δy := f (1 + Δx) - f 1 in
  Δy / Δx = 4 + 2 * Δx := by
sorry

end delta_y_over_delta_x_l88_88112


namespace painted_cells_l88_88619

theorem painted_cells (k l : ℕ) (h : k * l = 74) :
    (2 * k + 1) * (2 * l + 1) - k * l = 301 ∨ 
    (2 * k + 1) * (2 * l + 1) - k * l = 373 :=
sorry

end painted_cells_l88_88619


namespace pages_after_break_l88_88179

-- Formalize the conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.70

-- Define the proof problem
theorem pages_after_break : 
  let pages_read_before_break := (break_percentage * total_pages)
  let pages_remaining := total_pages - pages_read_before_break
  pages_remaining = 9 :=
by
  sorry

end pages_after_break_l88_88179


namespace min_value_expression_min_value_attained_l88_88095

theorem min_value_expression (a b : ℝ) (h : a > b ∧ b > 0) : 
  a^2 + 1 / (b * (a - b)) ≥ 4 :=
by
  sorry

theorem min_value_attained : ∃ a b : ℝ, a > b ∧ b > 0 ∧ a^2 + 1 / (b * (a - b)) = 4 :=
by
  use [real.sqrt 2, real.sqrt 2 / 2]
  split
  { linarith [real.sqrt_pos.2 (by norm_num)] }
  split
  { norm_num }
  sorry

end min_value_expression_min_value_attained_l88_88095


namespace double_series_sum_l88_88415

theorem double_series_sum : 
  (∑' n from 2, ∑' k from 1 to n-1, (k: ℝ) / 3^(n + k)) = (9 / 128) := 
  sorry

end double_series_sum_l88_88415


namespace problem_statement_l88_88891

noncomputable def a : ℕ → ℝ
| 0 := 1
| n+1 := if n = 1 then 3 - a n else 2 * a n - a (nat.pred n)

def c (n : ℕ) : ℝ := a n + 1 / 2^a n

def S (n : ℕ) : ℝ := (finset.range n).sum (λ i, c (i+1))

theorem problem_statement (n : ℕ) :
  S n = (n^2 + n + 2) / 2 - 1 / 2^n := 
sorry

end problem_statement_l88_88891


namespace find_speeds_l88_88463

noncomputable def speed_pedestrian := 5
noncomputable def speed_cyclist := 30

def distance_AB := 40
def starting_time_pedestrian := 4 -- In hours (24-hour format)
def starting_time_cyclist_1 := 7 + 20 / 60 -- 7:20 AM in hours
def halfway_distance := distance_AB / 2
def midpoint_meeting_time := 1 -- Time (in hours) after the first meeting
def starting_time_cyclist_2 := 8 + 30 / 60 -- 8:30 AM in hours

theorem find_speeds (x y : ℝ) (hx : x = speed_pedestrian) (hy : y = speed_cyclist) :
  let time_to_halfway := halfway_distance / x in
  let cyclist_time := (midpoint_meeting_time + time_to_halfway) in
  distance_AB = 
    cyclist_time * y + 
    time_to_halfway * x + 
    (midpoint_meeting_time - 1) * x :=
    x = speed_pedestrian ∧ y = speed_cyclist :=
begin
  sorry
end

end find_speeds_l88_88463


namespace john_gym_hours_l88_88615

theorem john_gym_hours :
  (2 * (1 + 1/3)) + (2 * (1 + 1/2)) + (1.5 + 3/4) = 7.92 :=
by
  sorry

end john_gym_hours_l88_88615


namespace regular_square_pyramid_side_edge_length_l88_88513

theorem regular_square_pyramid_side_edge_length 
  (base_edge_length : ℝ)
  (volume : ℝ)
  (h_base_edge_length : base_edge_length = 4 * Real.sqrt 2)
  (h_volume : volume = 32) :
  ∃ side_edge_length : ℝ, side_edge_length = 5 :=
by sorry

end regular_square_pyramid_side_edge_length_l88_88513


namespace exist_equilateral_triangle_on_parallel_lines_l88_88532

-- Define the concept of lines and points in a relation to them
def Line := ℝ → ℝ -- For simplicity, let's assume lines are functions

-- Define the points A1, A2, A3
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the concept of parallel lines
def parallel (D1 D2 : Line) : Prop :=
  ∀ x y, D1 x - D2 x = D1 y - D2 y

axiom D1 : Line
axiom D2 : Line
axiom D3 : Line

-- Ensure the lines are parallel
axiom parallel_D1_D2 : parallel D1 D2
axiom parallel_D2_D3 : parallel D2 D3

-- Main statement to prove
theorem exist_equilateral_triangle_on_parallel_lines :
  ∃ (A1 A2 A3 : Point), 
    (A1.y = D1 A1.x) ∧ 
    (A2.y = D2 A2.x) ∧ 
    (A3.y = D3 A3.x) ∧ 
    ((A1.x - A2.x)^2 + (A1.y - A2.y)^2 = (A2.x - A3.x)^2 + (A2.y - A3.y)^2) ∧ 
    ((A2.x - A3.x)^2 + (A2.y - A3.y)^2 = (A3.x - A1.x)^2 + (A3.y - A1.y)^2) := sorry

end exist_equilateral_triangle_on_parallel_lines_l88_88532


namespace last_bill_amount_l88_88024

theorem last_bill_amount :
  ∃ x : ℝ, 
    (x + 120) + 240 + 430 = 1234 ∧ x = 444 :=
begin
  use 444,
  split,
  {
    calc
      (444 + 120) + 240 + 430 = 564 + 240 + 430 : by norm_num
      ... = 804 + 430 : by norm_num
      ... = 1234 : by norm_num
  },
  {
    refl,
  }
end

end last_bill_amount_l88_88024


namespace decimal_to_binary_89_l88_88045

theorem decimal_to_binary_89 :
  (∃ (b : Nat), b = 2^6 + 2^4 + 2^3 + 2^0 ∧ nat2bin 89 = b) :=
by
  use 1011001  -- binary representation
  have : 2^6 + 2^4 + 2^3 + 2^0 = 64 + 16 + 8 + 1 := by norm_num
  rw [this]
  sorry  -- placeholder for the proof that nat2bin 89 = 1011001

end decimal_to_binary_89_l88_88045


namespace distinct_four_digit_integers_with_digit_product_18_l88_88963

theorem distinct_four_digit_integers_with_digit_product_18 : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in digits.prod = 18) → 
  (finset.univ.filter (λ m, (let mdigits := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in mdigits.prod = 18))).card = 36 :=
by
  sorry

end distinct_four_digit_integers_with_digit_product_18_l88_88963


namespace maximal_radius_of_cherry_in_goblet_l88_88807

-- Definition of the goblet (axial cross-section y = x^4)
def goblet (x : ℝ) : ℝ := x ^ 4

-- Definition of the circle centered at (0, r) that touches the origin
def circle (x r : ℝ) : ℝ := x^2 + (x^4 - r)^2 - r^2

-- Problem statement: Prove that the maximal radius r is 3 * real.cbrt(2) / 4
theorem maximal_radius_of_cherry_in_goblet :
  ∀ r : ℝ, (∀ x : ℝ, goblet x ≥ 0 → circle x r ≤ 0) ↔ r ≤ (3 * real.cbrt(2) / 4) :=
sorry

end maximal_radius_of_cherry_in_goblet_l88_88807


namespace dot_product_square_ABCD_l88_88600

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l88_88600


namespace triangle_bisectors_inequality_l88_88604

theorem triangle_bisectors_inequality
  (ABC : Triangle)
  (A1 A2 B1 B2 C1 C2 : Point)
  (hA1 : angle_bisector_point ABC A₁)
  (hA2 : circumcircle_intersection_point ABC A₂)
  (hB1 : angle_bisector_point ABC B₁)
  (hB2 : circumcircle_intersection_point ABC B₂)
  (hC1 : angle_bisector_point ABC C₁)
  (hC2 : circumcircle_intersection_point ABC C₂) :
  (distance A₁ A₂ / (distance B A₂ + distance A₂ C)) 
  + (distance B₁ B₂ / (distance C B₂ + distance B₂ A)) 
  + (distance C₁ C₂ / (distance A C₂ + distance C₂ B)) ≥ 3 / 4 :=
sorry

end triangle_bisectors_inequality_l88_88604


namespace two_bags_remainder_l88_88058

-- Given conditions
variables (n : ℕ)

-- Assume n ≡ 8 (mod 11)
def satisfied_mod_condition : Prop := n % 11 = 8

-- Prove that 2n ≡ 5 (mod 11)
theorem two_bags_remainder (h : satisfied_mod_condition n) : (2 * n) % 11 = 5 :=
by 
  unfold satisfied_mod_condition at h
  sorry

end two_bags_remainder_l88_88058


namespace expression_is_five_l88_88451

-- Define the expression
def given_expression : ℤ := abs (abs (-abs (-2 + 1) - 2) + 2)

-- Prove that the expression equals 5
theorem expression_is_five : given_expression = 5 :=
by
  -- We skip the proof for now
  sorry

end expression_is_five_l88_88451


namespace nonneg_int_solutions_to_x2_eq_6x_l88_88127

theorem nonneg_int_solutions_to_x2_eq_6x : {x : ℕ // x^2 = 6 * x}.card = 2 := by
  sorry

end nonneg_int_solutions_to_x2_eq_6x_l88_88127


namespace solve_for_y_l88_88678

theorem solve_for_y : 
  ∃ y : ℚ, 8 + 3.2 * y = 0.8 * y + 40 ∧ y = 40 / 3 :=
begin
  sorry
end

end solve_for_y_l88_88678


namespace remaining_empty_pages_l88_88022

def total_pages := 500
def first_week_pages := 150
def second_week_percentage := 0.30
def damaged_percentage := 0.20

theorem remaining_empty_pages : 
    let remaining_after_first_week := total_pages - first_week_pages in
    let second_week_pages := second_week_percentage * remaining_after_first_week in
    let remaining_after_second_week := remaining_after_first_week - second_week_pages in
    let damaged_pages := damaged_percentage * remaining_after_second_week in
    let final_remaining_pages := remaining_after_second_week - damaged_pages in
    final_remaining_pages = 196 :=
by
    sorry

end remaining_empty_pages_l88_88022


namespace mike_buys_rose_bushes_for_friend_l88_88204
-- Importing the relevant library

-- Define conditions as constants
variables (n_roses : ℕ) (cost_rose : ℕ) (n_aloes : ℕ) (cost_aloe : ℕ) (total_self : ℕ)

-- Assign values to constants according to the problem statement
def n_roses := 6
def cost_rose := 75
def n_aloes := 2
def cost_aloe := 100
def total_self := 500

-- The target is to prove that Mike bought 2 rose bushes for his friend
theorem mike_buys_rose_bushes_for_friend :
  let total_cost_roses := n_roses * cost_rose in
  let total_cost_aloes := n_aloes * cost_aloe in
  let cost_roses_self := total_self - total_cost_aloes in
  let n_roses_self := cost_roses_self / cost_rose in
  let n_roses_friend := n_roses - n_roses_self in
  n_roses_friend = 2 :=
by
  sorry

end mike_buys_rose_bushes_for_friend_l88_88204


namespace system_of_equations_solution_l88_88054

theorem system_of_equations_solution (x y z : ℝ) (hx : x = Real.exp (Real.log y))
(hy : y = Real.exp (Real.log z)) (hz : z = Real.exp (Real.log x)) : x = y ∧ y = z ∧ z = x ∧ x = Real.exp 1 :=
by
  sorry

end system_of_equations_solution_l88_88054


namespace markese_earnings_16_l88_88652

theorem markese_earnings_16 (E M : ℕ) (h1 : M = E - 5) (h2 : E + M = 37) : M = 16 :=
by
  sorry

end markese_earnings_16_l88_88652


namespace negation_proposition_false_l88_88241

variable (a : ℝ)

theorem negation_proposition_false : ¬ (∃ a : ℝ, a ≤ 2 ∧ a^2 ≥ 4) :=
sorry

end negation_proposition_false_l88_88241


namespace find_cos_pi_over_4_plus_alpha_l88_88505

-- Definitions of the conditions
def cos_alpha := -3/5
def alpha_third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

-- The main theorem: 
theorem find_cos_pi_over_4_plus_alpha (α : ℝ) (h_cos : cos α = cos_alpha) (h_quad : alpha_third_quadrant α) :
  cos (π / 4 + α) = (√2 / 10) := sorry

end find_cos_pi_over_4_plus_alpha_l88_88505


namespace digits_of_expression_l88_88410

theorem digits_of_expression : 
  let n := (2^14) * (5^12) in
  Nat.digits 10 n = 13 :=
by
  sorry

end digits_of_expression_l88_88410


namespace remaining_bread_after_three_days_l88_88643

namespace BreadProblem

def InitialBreadCount : ℕ := 200

def FirstDayConsumption (bread : ℕ) : ℕ := bread / 4
def SecondDayConsumption (remainingBreadAfterFirstDay : ℕ) : ℕ := 2 * remainingBreadAfterFirstDay / 5
def ThirdDayConsumption (remainingBreadAfterSecondDay : ℕ) : ℕ := remainingBreadAfterSecondDay / 2

theorem remaining_bread_after_three_days : 
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  breadAfterThirdDay = 45 := 
by
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  have : breadAfterThirdDay = 45 := sorry
  exact this

end BreadProblem

end remaining_bread_after_three_days_l88_88643


namespace trig_identity_l88_88828

theorem trig_identity : 
  cos (42 * (π / 180)) * cos (18 * (π / 180)) - cos (48 * (π / 180)) * sin (18 * (π / 180)) = 1 / 2 :=
by 
  sorry

end trig_identity_l88_88828


namespace fx_leq_one_l88_88114

noncomputable def f (x : ℝ) : ℝ := (x + 1) / Real.exp x

theorem fx_leq_one : ∀ x : ℝ, f x ≤ 1 := by
  sorry

end fx_leq_one_l88_88114


namespace convert_10203_base4_to_base10_l88_88420

def base4_to_base10 (n : ℕ) (d₀ d₁ d₂ d₃ d₄ : ℕ) : ℕ :=
  d₄ * 4^4 + d₃ * 4^3 + d₂ * 4^2 + d₁ * 4^1 + d₀ * 4^0

theorem convert_10203_base4_to_base10 :
  base4_to_base10 10203 3 0 2 0 1 = 291 :=
by
  -- proof goes here
  sorry

end convert_10203_base4_to_base10_l88_88420


namespace hyperbola_foci_coordinates_l88_88690

-- Define the hyperbola and prove the coordinates of the foci
theorem hyperbola_foci_coordinates :
  let hyperbola := (x y : ℝ) → x^2 / 16 - y^2 / 9 = 1
  ∀ x y : ℝ, hyperbola x y → ((x, y) = (-5, 0) ∨ (x, y) = (5, 0)) :=
by
  sorry

end hyperbola_foci_coordinates_l88_88690


namespace distance_between_foci_l88_88854

theorem distance_between_foci (a b : ℝ) (h₁ : a^2 = 18) (h₂ : b^2 = 2) :
  2 * (Real.sqrt (a^2 + b^2)) = 4 * Real.sqrt 5 :=
by
  sorry

end distance_between_foci_l88_88854


namespace locus_of_vertex_D_l88_88119

noncomputable def square_locus_conditions (a : Line) (c : Line) (o : Line) (A : Point) (C : Point) (O : Point) : Prop :=
  A ∈ a ∧ C ∈ c ∧ O ∈ o

noncomputable def rotated_line_90_pos (l : Line) (O : Point) : Line := sorry -- function defining rotation by +90°
noncomputable def rotated_line_90_neg (l : Line) (O : Point) : Line := sorry -- function defining rotation by -90°

theorem locus_of_vertex_D (a c o : Line) (A C O : Point) 
  (Hac : a ≠ c) (H_conditions : square_locus_conditions a c o A C O) :
  ∃ (F G H : ℝ), 
  ∀ (D : Point), D ∈ rotated_line_90_neg a O ∧ D ∈ rotated_line_90_pos c O → 
  (F * D.x + G * D.y + H = 0) :=
sorry

end locus_of_vertex_D_l88_88119


namespace sum_first_nine_terms_l88_88946

def seq (n : ℕ) : ℤ :=
  if n % 2 = 1 then 2 * n - 3 else 2 ^ (n - 1)

theorem sum_first_nine_terms : (List.range 9).map (λ n => seq (n + 1)).sum = 720 := by
  sorry

end sum_first_nine_terms_l88_88946


namespace speeds_correct_l88_88471

-- Definitions for conditions
def distance (A B : Type) := 40 -- given distance between A and B is 40 km
def start_time_pedestrian : Real := 4 -- pedestrian starts at 4:00 AM
def start_time_cyclist : Real := 7 + (20 / 60) -- cyclist starts at 7:20 AM
def midpoint_distance : Real := 20 -- the midpoint distance where cyclist catches up with pedestrian is 20 km

noncomputable def speeds (x y : Real) : Prop :=
  let t_catch_up := (20 - (10 / 3) * x) / (y - x) in -- time taken by the cyclist to catch up
  let t_total := (10 / 3) + t_catch_up + 1 in -- total time for pedestrian until meeting second cyclist
  4.5 = t_total ∧ -- total time in hours from 4:00 AM to 8:30 AM
  10 * x * (y - x) + 60 * x - 10 * x^2 = 60 * y - 60 * x ∧ -- initial condition simplification step
  y = 6 * x -- relationship between speeds based on derived equations

-- The proposition to prove
theorem speeds_correct : ∃ x y : Real, speeds x y ∧ x = 5 ∧ y = 30 :=
by
  sorry

end speeds_correct_l88_88471


namespace boat_cost_per_foot_l88_88205

theorem boat_cost_per_foot (total_savings : ℝ) (license_cost : ℝ) (docking_fee_multiplier : ℝ) (max_boat_length : ℝ) 
  (h1 : total_savings = 20000) 
  (h2 : license_cost = 500) 
  (h3 : docking_fee_multiplier = 3) 
  (h4 : max_boat_length = 12) 
  : (total_savings - (license_cost + docking_fee_multiplier * license_cost)) / max_boat_length = 1500 :=
by
  sorry

end boat_cost_per_foot_l88_88205


namespace binom_20_10_eq_184756_l88_88907

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l88_88907


namespace max_eccentricity_of_ellipse_l88_88533

theorem max_eccentricity_of_ellipse
  (R : ℝ)
  (r : ℝ)
  (a : ℝ)
  (b : ℝ)
  (hR : R = 1)
  (hr : r = 1 / 4)
  (ha : a = 10 / 3)
  (hb : b = 2) :
  sqrt (1 - (b / a) ^ 2) = 4 / 5 :=
by
  sorry

end max_eccentricity_of_ellipse_l88_88533


namespace distinct_four_digit_integers_with_digit_product_18_l88_88961

theorem distinct_four_digit_integers_with_digit_product_18 : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in digits.prod = 18) → 
  (finset.univ.filter (λ m, (let mdigits := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in mdigits.prod = 18))).card = 36 :=
by
  sorry

end distinct_four_digit_integers_with_digit_product_18_l88_88961


namespace solve_inequality_l88_88679

theorem solve_inequality (x : ℝ) : 
  (3 * x^2 - 5 * x + 2 > 0) ↔ (x < 2 / 3 ∨ x > 1) := 
by
  sorry

end solve_inequality_l88_88679


namespace vector_addition_correct_dot_product_correct_l88_88539

-- Define the two vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- Define the expected results
def a_plus_b_expected : ℝ × ℝ := (4, 3)
def a_dot_b_expected : ℝ := 5

-- Prove the sum of vectors a and b
theorem vector_addition_correct : a + b = a_plus_b_expected := by
  sorry

-- Prove the dot product of vectors a and b
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = a_dot_b_expected := by
  sorry

end vector_addition_correct_dot_product_correct_l88_88539


namespace steve_writing_time_per_regular_letter_l88_88681

-- Definitions based on the conditions
def time_to_write_regular_letter_per_page : ℕ := 10
def total_pages_written_per_month : ℕ := 24
def time_spent_for_long_letter : ℕ := 80
def long_letter_time_multiplier : ℕ := 2
def days_per_month : ℕ := 30
def days_per_regular_letter : ℕ := 3

-- The goal is to prove that Steve spends 20 minutes on each regular letter
theorem steve_writing_time_per_regular_letter : 
  ∀ (number_of_pages_written_per_month 
     time_to_write_regular_letter_per_page_in_minutes
     time_spent_for_long_letter_in_minutes
     long_letter_multiplier
     days_in_month
     days_per_regular_letter : ℕ),
  number_of_pages_written_per_month = 24 → 
  time_to_write_regular_letter_per_page_in_minutes = 10 → 
  time_spent_for_long_letter_in_minutes = 80 → 
  long_letter_multiplier = 2 → 
  days_in_month = 30 → 
  days_per_regular_letter = 3 → 
  (200 / (days_in_month / days_per_regular_letter)) = 20 := 
by
  intro number_of_pages_written_per_month
  intro time_to_write_regular_letter_per_page_in_minutes
  intro time_spent_for_long_letter_in_minutes
  intro long_letter_multiplier
  intro days_in_month
  intro days_per_regular_letter
  assume h1 : number_of_pages_written_per_month = 24
  assume h2 : time_to_write_regular_letter_per_page_in_minutes = 10
  assume h3 : time_spent_for_long_letter_in_minutes = 80
  assume h4 : long_letter_multiplier = 2
  assume h5 : days_in_month = 30
  assume h6 : days_per_regular_letter = 3
  let pages_for_long_letter := time_spent_for_long_letter_in_minutes / (time_to_write_regular_letter_per_page_in_minutes * long_letter_multiplier)
  let pages_for_regular_letters := number_of_pages_written_per_month - pages_for_long_letter
  let total_time_for_regular_letters := pages_for_regular_letters * time_to_write_regular_letter_per_page_in_minutes
  let number_of_regular_letters := days_in_month / days_per_regular_letter
  let time_per_regular_letter := total_time_for_regular_letters / number_of_regular_letters
  have hw: time_per_regular_letter = 20 := sorry
  exact hw

end steve_writing_time_per_regular_letter_l88_88681


namespace not_possible_to_arrange_l88_88725

theorem not_possible_to_arrange (
  instruments : Fin 13,
  colors : Fin 12,
  connected_by_wire : instruments → instruments → Prop
) :
  ¬ (∀ i : instruments, ∃ (f : Fin 12 → colors), function.injective f ∧ ∀ (j : instruments), connected_by_wire i j → ∃ c : colors, ∀ d : colors, d ≠ c) :=
by
  sorry

end not_possible_to_arrange_l88_88725


namespace prob_three_heads_is_one_eighth_l88_88327

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l88_88327


namespace cosine_of_A_l88_88061

theorem cosine_of_A (A B C : Type) 
  [right_triangle A B C] 
  (AB : ℕ) (BC : ℕ) (hAB : AB = 8) (hBC : BC = 12) : 
  cos_angle A B C BC (hypotenuse A B C AB BC hAB hBC) = (3 * real.sqrt 13) / 13 :=
by
  sorry

end cosine_of_A_l88_88061


namespace faye_initial_flowers_l88_88434

theorem faye_initial_flowers (bouquets : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ): 
  (bouquets = 8) → 
  (flowers_per_bouquet = 5) → 
  (wilted_flowers = 48) → 
  (bouquets * flowers_per_bouquet + wilted_flowers = 88) := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end faye_initial_flowers_l88_88434


namespace locus_of_points_M_is_circle_l88_88884

variables {K : Type*} [Field K]

structure Circle (K : Type*) :=
(center : K × K)
(radius : K)

variables {O : K × K} (r : K) (A B N : K × K)

def tangent_circle (C : Circle K) (P : K × K) (Q : K × K) : Circle K := sorry
def intersection_points (C1 C2 : Circle K) : set (K × K) := sorry

theorem locus_of_points_M_is_circle (O : K × K) (r : K) (A B N : K × K)
  (hA : (A - O).norm = r) (hB : (B - O).norm = r) (hN : ∃ λ : K, A + λ • (B - A) = N)
  (C_given : Circle K) :
  let C1 := tangent_circle C_given A N, C2 := tangent_circle C_given B N in
  ∃ C : Circle K, ∀ M ∈ intersection_points C1 C2, M ∈ C.center ∨ M ∈ C :=
sorry

end locus_of_points_M_is_circle_l88_88884


namespace boxes_containing_neither_l88_88385

theorem boxes_containing_neither (total_boxes markers erasers both : ℕ) 
  (h_total : total_boxes = 15) (h_markers : markers = 8) (h_erasers : erasers = 5) (h_both : both = 4) :
  total_boxes - (markers + erasers - both) = 6 :=
by
  sorry

end boxes_containing_neither_l88_88385


namespace cubic_polynomial_solution_l88_88048

noncomputable def q (x : ℚ) : ℚ := (51/13) * x^3 + (-31/13) * x^2 + (16/13) * x + (3/13)

theorem cubic_polynomial_solution : 
  q 1 = 3 ∧ q 2 = 23 ∧ q 3 = 81 ∧ q 5 = 399 :=
by {
  sorry
}

end cubic_polynomial_solution_l88_88048


namespace bug_distance_total_l88_88787

theorem bug_distance_total : 
  ∀ (p1 p2 p3 p4 : ℤ), 
    p1 = 3 → p2 = -5 → p3 = -2 → p4 = 6 →
    |p2 - p1| + |p3 - p2| + |p4 - p3| = 19 :=
by
  intros p1 p2 p3 p4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  exact sorry

end bug_distance_total_l88_88787


namespace digit_A_unique_solution_l88_88745

theorem digit_A_unique_solution :
  ∃ (A : ℕ), 0 ≤ A ∧ A < 10 ∧ (100 * A + 72 - 23 = 549) ∧ A = 5 :=
by
  sorry

end digit_A_unique_solution_l88_88745


namespace prob_three_heads_is_one_eighth_l88_88332

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l88_88332


namespace probability_of_first_three_heads_l88_88296

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l88_88296


namespace part_I_part_II_l88_88152

-- Define the problem conditions for Part (I)
def cards : Finset ℕ := {1, 2, 3, 4, 5}
def draw_two_without_replacement (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ t, t.card = 2)

-- Define the probability calculation for Part (I)
def prob_not_both_odd_or_even : ℚ :=
  let total_ways := (draw_two_without_replacement cards).card in
  let suitable_ways := (draw_two_without_replacement cards).filter 
    (λ t, t.to_list.head % 2 ≠ t.to_list.tail.head' 2) in
  suitable_ways.card / total_ways

-- The theorem for Part (I)
theorem part_I : prob_not_both_odd_or_even = 3/5 :=
  sorry

-- Define the problem conditions for Part (II)
def draw_with_replacement (s : Finset ℕ) (k : ℕ) : Finset (Multiset ℕ) :=
  Multiset.replicate k s.to_multiset.powerset.to_finset.to_multiset

-- Define the probability calculation for Part (II)
def prob_exactly_two_even_in_three_draws : ℚ :=
  let total_outcomes := (draw_with_replacement cards 3).card in
  let suitable_outcomes := (draw_with_replacement cards 3).filter 
    (λ m, m.count 2 + m.count 4 = 2) in
  suitable_outcomes.card / total_outcomes

-- The theorem for Part (II)
theorem part_II : prob_exactly_two_even_in_three_draws = 36/125 :=
  sorry

end part_I_part_II_l88_88152


namespace solve_equation_l88_88067

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48 ↔ x = 6 ∨ x = 8 := 
by
  sorry

end solve_equation_l88_88067


namespace train_crosses_bridge_in_59_seconds_l88_88340

variable (length_train length_bridge : ℝ)
variable (speed_kmph : ℝ)

def speed_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000/3600)

def total_distance (length_train length_bridge : ℝ) : ℝ :=
  length_train + length_bridge

def time_to_cross (total_distance speed_mps : ℝ) : ℝ :=
  total_distance / speed_mps

theorem train_crosses_bridge_in_59_seconds :
  length_train = 165 → length_bridge = 720 → speed_kmph = 54 →
  time_to_cross (total_distance length_train length_bridge) (speed_mps speed_kmph) = 59 := 
by
  intros
  sorry

end train_crosses_bridge_in_59_seconds_l88_88340


namespace find_a_and_sin_2A_l88_88450

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {triangle_ABC : ∀ (A B C : ℝ), 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π}
variable {cos_C : ℝ}

-- Conditions
def conditions : Prop :=
  triangle_ABC A B C ∧ cos_C = 3 / 4

-- Theorem
theorem find_a_and_sin_2A (h : conditions) : a = 1 ∧ sin (2 * A) = 5 * sqrt 7 / 16 := by
  sorry

end find_a_and_sin_2A_l88_88450


namespace exists_five_digit_palindrome_sum_of_four_digit_palindromes_l88_88378

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ is_palindrome n

def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ is_palindrome n

theorem exists_five_digit_palindrome_sum_of_four_digit_palindromes :
  ∃ n₁ n₂ : ℕ, is_four_digit_palindrome n₁ ∧ is_four_digit_palindrome n₂ ∧ is_five_digit_palindrome (n₁ + n₂) :=
by
  sorry

end exists_five_digit_palindrome_sum_of_four_digit_palindromes_l88_88378


namespace price_of_A_correct_l88_88792

noncomputable def A_price : ℝ := 25

theorem price_of_A_correct (H1 : 6000 / A_price - 4800 / (1.2 * A_price) = 80) 
                           (H2 : ∀ B_price : ℝ, B_price = 1.2 * A_price) : A_price = 25 := 
by
  sorry

end price_of_A_correct_l88_88792


namespace union_A_B_l88_88117

def A : Set ℕ := {1, 3}

def B : Set ℕ := {x | 0 < Real.log (x + 1) / Real.log 10 ∧ Real.log (x + 1) / Real.log 10 < 1 / 2 ∧ x ∈ Set.univ.to_list.to_set}

/-- Prove the union of sets A and B equals {1, 2, 3} -/
theorem union_A_B : (A ∪ B) = {1, 2, 3} := by
  sorry

end union_A_B_l88_88117


namespace dot_product_EC_ED_l88_88596

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l88_88596


namespace solve_inequality_l88_88222

theorem solve_inequality :
  {x : ℝ | x ∈ { y | (y^2 - 5*y + 6) / (y - 3)^2 > 0 }} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solve_inequality_l88_88222


namespace bicycle_meets_light_vehicle_l88_88257

noncomputable def meeting_time (v_1 v_2 v_3 v_4 : ℚ) : ℚ :=
  let x := 2 * (v_1 + v_4)
  let y := 6 * (v_2 - v_4)
  (x + y) / (v_3 + v_4) + 12

theorem bicycle_meets_light_vehicle (v_1 v_2 v_3 v_4 : ℚ) (h1 : 2 * (v_1 + v_4) = x)
  (h2 : x + y = 4 * (v_1 + v_2))
  (h3 : x + y = 5 * (v_2 + v_3))
  (h4 : 6 * (v_2 - v_4) = y) :
  meeting_time v_1 v_2 v_3 v_4 = 15 + 1/3 :=
by
  sorry

end bicycle_meets_light_vehicle_l88_88257


namespace person_looking_at_son_l88_88379

-- Definitions based on given conditions
variable (Person : Type) -- Type for representing a person
variable (father : Person → Person) -- Function representing the father of a person
variable (portrait : Person → Person) -- Function representing the person in the portrait that someone is looking at
variable (P : Person) -- The person looking at the portrait

-- Conditions given in the problem
def only_child (P : Person) := ∀ x : Person, father x = father P → x = P

def father_of_person_in_portrait := father (portrait P) = P

-- The theorem we need to prove
theorem person_looking_at_son (h1 : only_child P) (h2 : father_of_person_in_portrait P) : portrait P = father P :=
by
  sorry

end person_looking_at_son_l88_88379


namespace count_N_such_that_gcd_not_one_l88_88458

/-- There are 164 integers N such that 1 ≤ N ≤ 2000 and gcd(N^3 + 11, N + 5) > 1. -/
theorem count_N_such_that_gcd_not_one : 
  (∃ count : ℕ, count = (finset.filter (λ N, Nat.gcd (N^3 + 11) (N + 5) ≠ 1) 
  (finset.range 2001)).card ∧ count = 164) := by
  sorry

end count_N_such_that_gcd_not_one_l88_88458


namespace men_population_percentage_of_women_l88_88995

theorem men_population_percentage_of_women (M W : ℝ) (h : W = 0.90 * M) : (M / W) * 100 = 111.11 :=
by
  rw [←h]
  field_simp
  norm_num
  sorry

end men_population_percentage_of_women_l88_88995


namespace cost_price_marked_price_ratio_l88_88374

theorem cost_price_marked_price_ratio (x : ℝ) (hx : x > 0) :
  let selling_price := (2 / 3) * x
  let cost_price := (3 / 4) * selling_price 
  cost_price / x = 1 / 2 := 
by
  let selling_price := (2 / 3) * x 
  let cost_price := (3 / 4) * selling_price 
  have hs : selling_price = (2 / 3) * x := rfl 
  have hc : cost_price = (3 / 4) * selling_price := rfl 
  have ratio := hc.symm 
  simp [ratio, hs]
  sorry

end cost_price_marked_price_ratio_l88_88374


namespace min_value_proof_l88_88516

noncomputable def minimum_value (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a + 3 * b = 7 then
    if h' : a + 3 * b = 7 then
      min (1 / (1 + a) + 4 / (2 + b))
    else 0
  else 0

theorem min_value_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 7) : 
  (1 / (1 + a) + 4 / (2 + b)) = (13 + 4 * real.sqrt 3) / 14 :=
sorry

end min_value_proof_l88_88516


namespace directrix_of_parabola_l88_88439

-- Define the given parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

-- Define the directrix equation for the parabola
def directrix_eq : ℝ := -13 / 12

-- Problem statement: Proof that the equation of the directrix of the parabola is given by directrix_eq.
theorem directrix_of_parabola : ∀ x : ℝ, (∃ (a b c : ℝ), parabola x = a*x^2 + b*x + c) → (∃ y : ℝ, y = directrix_eq) :=
by
  intros x H
  use directrix_eq
  sorry

end directrix_of_parabola_l88_88439


namespace polynomial_root_sum_l88_88198

theorem polynomial_root_sum (c d : ℝ) (h_roots : c^2 - 6*c + 10 = 0 ∧ d^2 - 6*d + 10 = 0) :
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16036 := 
by 
  -- condition of Vieta's formulas
  have cd_eq : c + d = 6 ∧ c * d = 10 := sorry,
  sorry

end polynomial_root_sum_l88_88198


namespace ratio_proof_l88_88407

noncomputable def total_capacity : ℝ := 10 -- million gallons
noncomputable def amount_end_month : ℝ := 6 -- million gallons
noncomputable def normal_level : ℝ := total_capacity - 5 -- million gallons

theorem ratio_proof (h1 : amount_end_month = 0.6 * total_capacity)
                    (h2 : normal_level = total_capacity - 5) :
  (amount_end_month / normal_level) = 1.2 :=
by sorry

end ratio_proof_l88_88407


namespace part1_solution_l88_88115

theorem part1_solution (x : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x|) :
  f x + f (x - 1) ≤ 2 ↔ x ∈ set.Icc (-1 / 2 : ℝ) (3 / 2 : ℝ) :=
sorry

end part1_solution_l88_88115


namespace count_four_digit_integers_with_product_18_l88_88967

def valid_digits (n : ℕ) : Prop := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def digit_product_18 (a b c d : ℕ) : Prop := 
  a * b * c * d = 18

def four_digit_integer (a b c d : ℕ) : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d

theorem count_four_digit_integers_with_product_18 : 
  (∑ a b c d in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    ite (four_digit_integer a b c d ∧ digit_product_18 a b c d) 1 0) = 48 := 
sorry

end count_four_digit_integers_with_product_18_l88_88967


namespace parallelogram_angles_l88_88213

variable {A B C D : Type}

-- Define angles
variable (angle : Type)
variable [has_add angle]
variable adjSum180 : angle → angle → Prop -- Adjacent angles sum up to 180
variable oppEqual : angle → angle → Prop -- Opposite angles are equal

-- Define a parallelogram with vertices A, B, C, D
structure Parallelogram (A B C D : Type) :=
(AB_parallel_CD : Prop) -- AB is parallel to CD
(AD_parallel_BC : Prop) -- AD is parallel to BC

-- The properties we want to prove
theorem parallelogram_angles (P : Parallelogram A B C D)
  (angle_A angle_B angle_C angle_D : angle)
  (adjSum180_A_B : adjSum180 angle_A angle_B)
  (adjSum180_B_C : adjSum180 angle_B angle_C)
  (adjSum180_C_D : adjSum180 angle_C angle_D)
  (adjSum180_D_A : adjSum180 angle_D angle_A)
  (oppEqual_A_C : oppEqual angle_A angle_C)
  (oppEqual_B_D : oppEqual angle_B angle_D) :
  (adjSum180 angle_A angle_B) ∧ (adjSum180 angle_B angle_C) ∧ 
  (adjSum180 angle_C angle_D) ∧ (adjSum180 angle_D angle_A) ∧
  (oppEqual angle_A angle_C) ∧ (oppEqual angle_B angle_D) := by
  sorry

end parallelogram_angles_l88_88213


namespace place_mat_length_l88_88384

noncomputable def octagon_side_length (r : ℝ) : ℝ := 2 * r * Real.sin (Real.pi / 8)

noncomputable def mat_length (r : ℝ) (d : ℝ) : ℝ :=
  let inner_leg := r * Real.cos (Real.pi / 8) - 0.5
  in Real.sqrt (r^2 - (0.5)^2) - inner_leg

theorem place_mat_length :
  let r := 5
  let d := octagon_side_length r
  mat_length r d = 0.8554 :=
by
  sorry

end place_mat_length_l88_88384


namespace river_width_l88_88805

def boat_width : ℕ := 3
def num_boats : ℕ := 8
def space_between_boats : ℕ := 2
def riverbank_space : ℕ := 2

theorem river_width : 
  let boat_space := num_boats * boat_width
  let between_boat_space := (num_boats - 1) * space_between_boats
  let riverbank_space_total := 2 * riverbank_space
  boat_space + between_boat_space + riverbank_space_total = 42 :=
by
  sorry

end river_width_l88_88805


namespace valid_y_values_for_triangle_l88_88815

-- Define the triangle inequality conditions for sides 8, 11, and y^2
theorem valid_y_values_for_triangle (y : ℕ) (h_pos : y > 0) :
  (8 + 11 > y^2) ∧ (8 + y^2 > 11) ∧ (11 + y^2 > 8) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by
  sorry

end valid_y_values_for_triangle_l88_88815


namespace circle_intersection_problem_l88_88065

theorem circle_intersection_problem :
  ∃ (C : ℝ) (A : ℝ) (l m : ℝ → ℝ → ℝ), 
  let f := λ (x y : ℝ), x^2 + y^2 + 4 * x - 3
  let g := λ (x y : ℝ), x^2 + y^2 - 4 * y - 3
  let line := λ (x y : ℝ), 2 * x - y - 4
  ∃ h k : ℝ, (∀ x y, (x = h) ∧ (y = k) ↔ C * (1 + -4/3) * x^2 + C * (1 + -4/3) * y^2 + 4 * x + 4 * (-4/3) * y - 3 + -3 * (-4/3) = 0) ∧ 
  (C * x^2 + C * y^2 - 12 * x - 16 * y - 3 = 0) ∧
  line h k = 0 :=
by sorry

end circle_intersection_problem_l88_88065


namespace true_propositions_among_converse_inverse_contrapositive_l88_88709

theorem true_propositions_among_converse_inverse_contrapositive
  (x : ℝ)
  (h1 : x^2 ≥ 1 → x ≥ 1) :
  (if x ≥ 1 then x^2 ≥ 1 else true) ∧ 
  (if x^2 < 1 then x < 1 else true) ∧ 
  (if x < 1 then x^2 < 1 else true) → 
  ∃ n, n = 2 :=
by sorry

end true_propositions_among_converse_inverse_contrapositive_l88_88709


namespace general_term_l88_88927

-- Define the sequence of sums S and the general term sequence a
noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := ∑ i in finset.range n, a (i + 1)

-- Given condition: S_n = 3 - 2a_n
axiom condition (a : ℕ → ℝ) (n : ℕ) : S n a = 3 - 2 * a n

-- Define the required proof of the general term of the sequence satisfying the condition
theorem general_term (a : ℕ → ℝ) : (∀ n, a 1 = 1 ∧ a n = (2 / 3)^(n - 1)) :=
begin
  intros n,
  induction n with d hd,
  { simp [a] }, -- base case for n=0
  sorry, -- induction step
end

end general_term_l88_88927


namespace directrix_of_parabola_l88_88438

-- Define the given parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

-- Define the directrix equation for the parabola
def directrix_eq : ℝ := -13 / 12

-- Problem statement: Proof that the equation of the directrix of the parabola is given by directrix_eq.
theorem directrix_of_parabola : ∀ x : ℝ, (∃ (a b c : ℝ), parabola x = a*x^2 + b*x + c) → (∃ y : ℝ, y = directrix_eq) :=
by
  intros x H
  use directrix_eq
  sorry

end directrix_of_parabola_l88_88438


namespace min_c_unique_solution_thm_l88_88856

noncomputable def min_c_unique_solution : ℝ :=
  let c := 24 in
  c

theorem min_c_unique_solution_thm :
  ∀ x y : ℝ, (8 * (x + 7) ^ 4 + (y - 4) ^ 4 = min_c_unique_solution ∧ (x + 4) ^ 4 + 8 * (y - 7) ^ 4 = min_c_unique_solution) 
  → min_c_unique_solution = 24 :=
by
  intro x y h
  let c := min_c_unique_solution
  cases' h with h1 h2
  sorry

end min_c_unique_solution_thm_l88_88856


namespace average_minutes_correct_l88_88405

noncomputable def average_minutes_run_per_day : ℚ :=
  let f (fifth_graders : ℕ) : ℚ := (48 * (4 * fifth_graders) + 30 * (2 * fifth_graders) + 10 * fifth_graders) / (4 * fifth_graders + 2 * fifth_graders + fifth_graders)
  f 1

theorem average_minutes_correct :
  average_minutes_run_per_day = 88 / 7 :=
by
  sorry

end average_minutes_correct_l88_88405


namespace func_decreasing_iff_a_ge_one_l88_88924

theorem func_decreasing_iff_a_ge_one (a : ℝ) : 
  (∀ x : ℝ, x < 0 -> deriv (λ x, real.exp x - a * x) x ≤ 0) ↔ (a ≥ 1) :=
by
  sorry

end func_decreasing_iff_a_ge_one_l88_88924


namespace problem1_problem2_l88_88035

open Real

theorem problem1 : 2 * log 3 2 - log 3 (32 / 9) + log 3 8 - 5 ^ (log 5 3) = -1 := 
by 
  sorry

theorem problem2 : 0.064 ^ (-1 / 3) - (-1 / 8) ^ 0 + 16 ^ (3 / 4) + 0.25 ^ (1 / 2) + 2 * log 3 6 - log 3 12 = 11 := 
by 
  sorry

end problem1_problem2_l88_88035


namespace remaining_bread_after_three_days_l88_88644

namespace BreadProblem

def InitialBreadCount : ℕ := 200

def FirstDayConsumption (bread : ℕ) : ℕ := bread / 4
def SecondDayConsumption (remainingBreadAfterFirstDay : ℕ) : ℕ := 2 * remainingBreadAfterFirstDay / 5
def ThirdDayConsumption (remainingBreadAfterSecondDay : ℕ) : ℕ := remainingBreadAfterSecondDay / 2

theorem remaining_bread_after_three_days : 
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  breadAfterThirdDay = 45 := 
by
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  have : breadAfterThirdDay = 45 := sorry
  exact this

end BreadProblem

end remaining_bread_after_three_days_l88_88644


namespace range_of_f_l88_88037

-- Function definition
def f (x : ℝ) : ℝ := |x + 3| - |x - 5| + 3 * x

-- Theorem statement for the range of the function
theorem range_of_f : set.range f = set.univ :=
by sorry

end range_of_f_l88_88037


namespace estimated_probability_of_excellence_l88_88362

def dart_throw_simulation (rounds : List (Fin 2 × Fin 2 × Fin 2)) : ℕ :=
  rounds.count (λ ⟨(x₁, x₂, x₃)⟩, (x₁ + x₂ + x₃) ≥ 2)

def probability_of_excellence (excellent_rounds : ℕ) (total_rounds : ℕ) : ℝ :=
  excellent_rounds / total_rounds

theorem estimated_probability_of_excellence :
  let simulation_data : List (Fin 2 × Fin 2 × Fin 2) := [(1,0,1), (1,1,1), (0,1,1), (1,0,1), (0,1,0), (1,0,0), (1,0,0), (0,1,1), (1,1,1), (1,1,0), (0,0,0), (0,1,1), (0,1,0), (0,0,1), (1,1,1), (0,1,1), (1,0,0), (0,0,0), (1,0,1), (1,0,1)]
  in probability_of_excellence (dart_throw_simulation simulation_data) 20 = 0.6 :=
by
  sorry

end estimated_probability_of_excellence_l88_88362


namespace intervals_and_range_of_function_l88_88442

noncomputable def func (x : ℝ) : ℝ := (1/4)^x - (1/2)^x + 1

theorem intervals_and_range_of_function :
  (∀ x ∈ Icc (-3 : ℝ) 1, func x > func (x + ε) ∀ ε > 0) ∧
  (∀ x ∈ Icc (1 : ℝ) 2, func x < func (x + ε) ∀ ε > 0) ∧
  (Icc (3/4 : ℝ) (57 : ℝ) = {y : ℝ | ∃ x ∈ Icc (-3 : ℝ) 2, func x = y }) :=
by sorry

end intervals_and_range_of_function_l88_88442


namespace probability_calc_correct_l88_88670

noncomputable def probability_same_number_sum_ge_20 : ℚ :=
  let favorable_cases : ℚ := 7 / 1944 in
  let total_cases : ℚ := 1 in
  favorable_cases / total_cases

theorem probability_calc_correct :
  probability_same_number_sum_ge_20 = 7 / 1944 :=
by
  sorry

end probability_calc_correct_l88_88670


namespace curve_is_ellipse_l88_88839

theorem curve_is_ellipse (r θ : ℝ) : 
  (r = 1 / (1 - cos θ - sin θ)) → 
  ∃ (a b : ℝ), (a ≠ 0 ∧ b ≠ 0) ∧ 
  ∀ (x y : ℝ), ((x - a)^2 + (y - b)^2 + 2 * x * y = 3) := 
by 
  intro h
  sorry

end curve_is_ellipse_l88_88839


namespace midpoints_form_equilateral_triangle_l88_88560

theorem midpoints_form_equilateral_triangle
  (A B C D E F : ℂ) (r : ℝ)
  (h1 : abs A = r)
  (h2 : abs B = r)
  (h3 : abs C = r)
  (h4 : abs D = r)
  (h5 : abs E = r)
  (h6 : abs F = r)
  (h7 : abs (A - B) = r)
  (h8 : abs (C - D) = r)
  (h9 : abs (E - F) = r)
  (h10 : abs (B - A) = r)
  (h11 : abs (D - C) = r)
  (h12 : abs (F - E) = r) :
  let P := (B + C) / 2,
      Q := (D + E) / 2,
      R := (F + A) / 2 in
  abs (P - Q) = abs (Q - R) ∧ abs (Q - R) = abs (R - P) := sorry

end midpoints_form_equilateral_triangle_l88_88560


namespace answer_l88_88576

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

end answer_l88_88576


namespace find_arithmetic_general_formula_find_sum_arithmetic_sequence_l88_88496

variable (aₙ : ℕ → ℕ)
variable (d : ℕ)
variable (Sₙ : ℕ → ℕ)
variable (n : ℕ)

-- Condition: Sequence {aₙ} is arithmetic with first term aₙ = 2 and common difference d ≠ 0
axiom h₁ : d ≠ 0
axiom h₂ : aₙ(1) = 2
axiom h₃ : aₙ(n) = aₙ(1) + (n-1) * d

-- Condition: {aₙ} forms a geometric sequence with aₙ(1), aₙ(3), aₙ(9)
axiom h₄ : (aₙ(3))^2 = aₙ(1) * aₙ(9)

-- Goal: Prove the general formula for the sequence {aₙ}
theorem find_arithmetic_general_formula : aₙ(n) = 2 * n :=
by
  sorry

-- Given the general term of the new sequence {2^aₙ - 1}
axiom h₅ : ∀ n, Sₙ(n) = ∑ k in finset.range n, ∃ aₙ(k), 4^k - 1 - n

-- Goal: Prove the sum of the first n terms, Sₙ, of the sequence {2^aₙ - 1}
theorem find_sum_arithmetic_sequence : Sₙ(n) = (4/3) * (4^n - 1) - n :=
by
  sorry

end find_arithmetic_general_formula_find_sum_arithmetic_sequence_l88_88496


namespace calculate_percentage_l88_88347

variable (Part Whole : ℝ)

theorem calculate_percentage (hPart : Part = 294.16) (hWhole : Whole = 1,258.37) : 
  (Part / Whole) * 100 = 23.37 := by 
  sorry

end calculate_percentage_l88_88347


namespace possible_values_f_l88_88631

noncomputable def f (x y z : ℝ) : ℝ := (y / (y + x)) + (z / (z + y)) + (x / (x + z))

theorem possible_values_f (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x^2 + y^3 = z^4) : 
  1 < f x y z ∧ f x y z < 2 :=
sorry

end possible_values_f_l88_88631


namespace OHara_triple_example_l88_88262

def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a + Real.sqrt b = x)

theorem OHara_triple_example : is_OHara_triple 36 25 11 :=
by {
  sorry
}

end OHara_triple_example_l88_88262


namespace distinct_four_digit_integers_with_product_18_l88_88971

theorem distinct_four_digit_integers_with_product_18 :
  ∃ n : ℕ, n = 24 ∧ ∀ (d1 d2 d3 d4 : ℕ), (d1 * d2 * d3 * d4 = 18 ∧ 1000 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000) →
    set.finite { x | ∃ (d1 d2 d3 d4 : ℕ), x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ d1 * d2 * d3 * d4 = 18 ∧ ∀ i ∈ [d1, d2, d3, d4], 1 ≤ i ∧ i ≤ 9 } :=
begin
  sorry
end

end distinct_four_digit_integers_with_product_18_l88_88971


namespace parabola_properties_l88_88710

theorem parabola_properties :
  ∃ m : ℝ, 
    (∀ x y : ℝ, (y = x^2 - 2 * m * x + m^2 - 1) ∧ x = 0 → y = 3) ∧
    (∃ v : ℝ, v = -2 * m / (2 * 1) ∧ v > 0 → m = 2) ∧
    (∀ x y : ℝ, (y = x^2 - 4 * x + 3) ∧ 0 ≤ x ∧ x ≤ 3 → -1 ≤ y ∧ y ≤ 3) :=
by
  use 2
  intros x y hxy x0 y0 hx0 y3
  sorry

end parabola_properties_l88_88710


namespace dot_product_EC_ED_l88_88592

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l88_88592


namespace find_cos_theta_l88_88538

variables {V : Type*} [inner_product_space ℝ V] (a b : V)
variables (θ : ℝ)

-- Given conditions
def condition_1 := ∥a∥ = 2
def condition_2 := ∥b∥ = real.sqrt 2
def condition_3 := inner (a + b) (2 • a - b) = 0

-- The proof statement
theorem find_cos_theta
  (ha : condition_1 a)
  (hb : condition_2 b)
  (hc : condition_3 a b) :
  real.cos θ = -3 * real.sqrt 2 / 2 :=
  sorry

end find_cos_theta_l88_88538


namespace prob_first_three_heads_all_heads_l88_88313

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l88_88313


namespace sugar_per_cup_correct_l88_88613

variable (total_sugar : ℝ) (num_cups : ℕ) (sugar_per_cup : ℝ)

noncomputable def sugar_in_cup (total_sugar : ℝ) (num_cups : ℕ) : ℝ :=
  total_sugar / num_cups

theorem sugar_per_cup_correct :
  total_sugar = 84.6 → 
  num_cups = 12 → 
  sugar_in_cup total_sugar num_cups = 7.05 :=
by
  intros h1 h2
  rw [h1, h2]
  exact (rfl : 84.6 / 12 = 7.05)

#check sugar_per_cup_correct -- This line ensures the theorem is correctly stated

end sugar_per_cup_correct_l88_88613


namespace range_of_m_l88_88521

open Real

noncomputable def e : ℝ := Real.exp 1

noncomputable def g (x : ℝ) : ℝ := x^2 + 5 * ln x

theorem range_of_m :
  ∀ m : ℝ, 
  (∃ x : ℝ, e ≤ x ∧ x ≤ 2 * e ∧ m + 1 = g x) ↔
  (e^2 + 4 ≤ m ∧ m ≤ 4 * e^2 + 4 + 5 * ln 2) := 
begin
  sorry
end

end range_of_m_l88_88521


namespace percentage_markup_is_correct_l88_88784

def selling_price : ℝ := 11.00
def cost_price : ℝ := 9.90
def markup : ℝ := selling_price - cost_price
def percentage_markup : ℝ := (markup / cost_price) * 100

theorem percentage_markup_is_correct : percentage_markup = 11.11 :=
by
  unfold percentage_markup
  unfold markup
  unfold selling_price
  unfold cost_price
  sorry

end percentage_markup_is_correct_l88_88784


namespace cube_probability_red_blue_faces_l88_88004

theorem cube_probability_red_blue_faces :
  let Ω := Finset.range (2^6) -- The set of all possible color combinations (total 64)
  ∃ (f : Finset (fin 6) → String), (∀ x, f (Finset.singleton x) ∈ {"red", "blue"}) ∧
  let P := (n : ℕ) → ((n < 6) ∧ (f (Finset.range n) = "blue") ∧ (f (Finset.range (6 - n)) = "red")) →
  (Finset.card (Finset.filter P Ω)) / (Finset.card Ω) = 27 / 64
:= sorry

end cube_probability_red_blue_faces_l88_88004


namespace probability_penny_dime_same_nickel_quarter_different_l88_88686

def coin := Type

def penny : coin := sorry
def nickel : coin := sorry
def dime : coin := sorry
def quarter : coin := sorry
def half_dollar : coin := sorry

def outcome (c : coin) := c = "H" ∨ c = "T"

def same (c1 c2 : coin) := (c1 = "H" ∧ c2 = "H") ∨ (c1 = "T" ∧ c2 = "T")
def different (c1 c2 : coin) := (c1 = "H" ∧ c2 = "T") ∨ (c1 = "T" ∧ c2 = "H")

theorem probability_penny_dime_same_nickel_quarter_different :
  (probability (same penny dime) (different nickel quarter) = 1 / 4) := sorry

end probability_penny_dime_same_nickel_quarter_different_l88_88686


namespace cube_dimension_ratio_l88_88361

theorem cube_dimension_ratio (V1 V2 : ℕ) (h1 : V1 = 27) (h2 : V2 = 216) :
  ∃ r : ℕ, r = 2 ∧ (∃ l1 l2 : ℕ, l1 * l1 * l1 = V1 ∧ l2 * l2 * l2 = V2 ∧ l2 = r * l1) :=
by
  sorry

end cube_dimension_ratio_l88_88361


namespace find_a_minus_b_l88_88931

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x + 1

theorem find_a_minus_b (a b : ℝ)
  (h1 : deriv (f a b) 1 = -2)
  (h2 : deriv (f a b) (2 / 3) = 0) :
  a - b = 10 :=
sorry

end find_a_minus_b_l88_88931


namespace triangle_shading_probability_l88_88562

theorem triangle_shading_probability (n_triangles: ℕ) (n_shaded: ℕ) (h1: n_triangles > 4) (h2: n_shaded = 4) (h3: n_triangles = 10) :
  (n_shaded / n_triangles) = 2 / 5 := 
by
  sorry

end triangle_shading_probability_l88_88562


namespace max_value_of_expression_is_infinite_l88_88443

theorem max_value_of_expression_is_infinite : 
  ∀ x : ℝ, (Sup (set.image (λ x : ℝ, 4^x - 2^x + 5) set.univ)) = ⊤ := 
by
  sorry

end max_value_of_expression_is_infinite_l88_88443


namespace theater_earnings_l88_88375

theorem theater_earnings :
  let matinee_price := 5
  let evening_price := 7
  let opening_night_price := 10
  let popcorn_price := 10
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let half_of_customers_that_bought_popcorn := 
    (matinee_customers + evening_customers + opening_night_customers) / 2
  let total_earnings := 
    (matinee_price * matinee_customers) + 
    (evening_price * evening_customers) + 
    (opening_night_price * opening_night_customers) + 
    (popcorn_price * half_of_customers_that_bought_popcorn)
  total_earnings = 1670 :=
by
  sorry

end theater_earnings_l88_88375


namespace probability_xiao_ming_chooses_king_of_sky_l88_88432

theorem probability_xiao_ming_chooses_king_of_sky :
  let choices := ["Life is Unfamiliar", "King of the Sky", "Prosecution Storm"]
  in Probability (Xiao Ming chooses "King of the Sky") = 1/3 :=
by sorry

end probability_xiao_ming_chooses_king_of_sky_l88_88432


namespace jane_stopped_babysitting_years_ago_l88_88175

-- Definitions based on the problem conditions
def jane_start_age := 18
def jane_current_age := 32
def max_child_age_when_jane_started_babysitting := jane_start_age / 2
def current_oldest_babysat_age := 23

-- The actual proof goal
theorem jane_stopped_babysitting_years_ago :
  jane_current_age - (current_oldest_babysat_age - max_child_age_when_jane_started_babysitting) = 14 :=
by 
  calc
  jane_current_age - (current_oldest_babysat_age - max_child_age_when_jane_started_babysitting)
      = 32 - (23 - 9) : by sorry
  ... = 32 - 14        : by sorry
  ... = 14             : by sorry

end jane_stopped_babysitting_years_ago_l88_88175


namespace find_k_l88_88864

theorem find_k (k : ℝ) : 2 + (2 + k) / 3 + (2 + 2 * k) / 3^2 + (2 + 3 * k) / 3^3 + 
  ∑' (n : ℕ), (2 + (n + 1) * k) / 3^(n + 1) = 7 ↔ k = 16 / 3 := 
sorry

end find_k_l88_88864


namespace negation_of_forall_geq_l88_88707

theorem negation_of_forall_geq {x : ℝ} : ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end negation_of_forall_geq_l88_88707


namespace cost_of_individual_rose_l88_88217

theorem cost_of_individual_rose (cost_of_dozen : ℕ) (cost_of_two_dozen : ℕ) (max_roses : ℕ) (total_amount : ℕ) :
  cost_of_dozen = 36 → cost_of_two_dozen = 50 → max_roses = 316 → total_amount = 680 → (30 / 4 : ℚ) = 7.5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end cost_of_individual_rose_l88_88217


namespace proof_problem_theorem_l88_88580

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l88_88580


namespace factorial_sum_representation_100_l88_88666

theorem factorial_sum_representation_100 :
  ∃ (s : Multiset ℕ), (∀ x ∈ s, x ∈ (Multiset.range 100).map Nat.factorial) ∧
    s.sum = 100! ∧ s.card ≥ 100! :=
sorry

end factorial_sum_representation_100_l88_88666


namespace pythagorean_triplet_unique_solution_l88_88063

-- Define the conditions given in the problem
def is_solution (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  2000 ≤ a ∧ a ≤ 3000 ∧
  2000 ≤ b ∧ b ≤ 3000 ∧
  2000 ≤ c ∧ c ≤ 3000

-- Prove that the only set of integers (a, b, c) meeting the conditions
-- equals the specific tuple (2100, 2059, 2941)
theorem pythagorean_triplet_unique_solution : 
  ∀ a b c : ℕ, is_solution a b c ↔ (a = 2100 ∧ b = 2059 ∧ c = 2941) :=
by
  sorry

end pythagorean_triplet_unique_solution_l88_88063


namespace distinct_four_digit_integers_with_product_18_l88_88972

theorem distinct_four_digit_integers_with_product_18 :
  ∃ n : ℕ, n = 24 ∧ ∀ (d1 d2 d3 d4 : ℕ), (d1 * d2 * d3 * d4 = 18 ∧ 1000 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000) →
    set.finite { x | ∃ (d1 d2 d3 d4 : ℕ), x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ d1 * d2 * d3 * d4 = 18 ∧ ∀ i ∈ [d1, d2, d3, d4], 1 ≤ i ∧ i ≤ 9 } :=
begin
  sorry
end

end distinct_four_digit_integers_with_product_18_l88_88972


namespace rad_prod_simplify_l88_88826

theorem rad_prod_simplify (q : ℝ) : 
  Real.sqrt (12 * q) * Real.sqrt (8 * q^2) * Real.sqrt (9 * q^5) = 12 * q^4 * Real.sqrt(6) :=
by sorry

end rad_prod_simplify_l88_88826


namespace card_M_l88_88708

open Set

noncomputable def M : Set ℤ := {x | (x - 5) * (x - 1) ≤ 0 ∧ x ≠ 1}

theorem card_M : (card M) = 4 :=
by
  sorry

end card_M_l88_88708


namespace inscribed_polygon_larger_area_l88_88664

theorem inscribed_polygon_larger_area (N M : Type) [convex_polygon N]
  [convex_polygon M] (same_side_lengths : same_side_lengths N M)
  (N_inscribed_in_circle : inscribed_in_circle N) :
  area N > area M := 
sorry

end inscribed_polygon_larger_area_l88_88664


namespace part_1_part_2_part_3_l88_88939

def f (x : ℝ) : ℝ := x + 1 / x

theorem part_1 : ∀ x : ℝ, f (-x) = -f x :=
by sorry

theorem part_2 : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f x2 < f x1 :=
by sorry

theorem part_3 : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 0 → f x2 < f x1 :=
by sorry

end part_1_part_2_part_3_l88_88939


namespace binom_20_10_l88_88915

-- Definitions for the provided conditions
def binom_18_8 := 43758
def binom_18_9 := 48620
def binom_18_10 := 43758

-- The theorem we need to prove
theorem binom_20_10 : ∀
  (binom_18_8 = 43758)
  (binom_18_9 = 48620)
  (binom_18_10 = 43758),
  binomial 20 10 = 184756 :=
by
  sorry

end binom_20_10_l88_88915


namespace find_c_for_deg3_l88_88041

-- Definitions of the polynomials f and g
def f (x : ℝ) : ℝ := 2 - 6 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℝ) : ℝ := 4 - 3 * x - 7 * x^3 + 11 * x^4

-- Statement to prove
theorem find_c_for_deg3 : ∃ c : ℝ, (∀ x : ℝ, f(x) + c * g(x) = 2 - 6 * x + 4 * x^2 - 5 * x^3 ∧ c = -7/11) :=
sorry

end find_c_for_deg3_l88_88041


namespace limit_of_ratio_l88_88506

def a_n (n : ℕ) : ℕ := n * (n - 1) / 2

theorem limit_of_ratio :
  (Real.limit (λ n : ℕ, (2 * a_n n : ℝ) / (n^2 + 1) ) atTop) = 1 :=
by 
  sorry 

end limit_of_ratio_l88_88506


namespace find_b_l88_88247

theorem find_b (a b : ℝ) (h_inv_var : a^2 * Real.sqrt b = k) (h_ab : a * b = 72) (ha3 : a = 3) (hb64 : b = 64) : b = 18 :=
sorry

end find_b_l88_88247


namespace no_odd_digits_1_to_9999_l88_88123

-- Lean statement for the given mathematical problem
theorem no_odd_digits_1_to_9999 : 
  let digits := [0, 2, 4, 6, 8]
  in ∀ n : ℕ, (1 ≤ n ∧ n ≤ 9999) →
  (∀ d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10], d ∈ digits) →
  (finset.filter (λ k, (∀ d ∈ [k / 1000 % 10, k / 100 % 10, k / 10 % 10, k % 10], d ∈ digits)) (finset.range 10000).to_set).card = 624 :=
by
  sorry

end no_odd_digits_1_to_9999_l88_88123


namespace prob_first_three_heads_all_heads_l88_88314

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l88_88314


namespace intersection_of_sphere_is_circle_intersection_of_cylinder_is_circle_or_ellipse_l88_88380

def intersect_sphere_with_plane (plane : Type) (sphere : Type) : Type :=
Sorry -- Definition of the intersection of a plane and a sphere, typically a circle.

theorem intersection_of_sphere_is_circle (plane : Type) (sphere : Type) :
  intersect_sphere_with_plane plane sphere = circle := 
sorry

def intersect_cylinder_with_plane (plane : Type) (cylinder : Type) : Type :=
Sorry -- Definition of the intersection of a plane and a cylinder, either a circle or an ellipse.

theorem intersection_of_cylinder_is_circle_or_ellipse (plane : Type) (cylinder : Type) :
  intersect_cylinder_with_plane plane cylinder = circle ∨ intersect_cylinder_with_plane plane cylinder = ellipse := 
sorry

end intersection_of_sphere_is_circle_intersection_of_cylinder_is_circle_or_ellipse_l88_88380


namespace simplify_expression_l88_88833

theorem simplify_expression (N : ℕ) (h : N ≥ 1) :
  (N - 1)! * N * (N + 1) / (N + 2)! = 1 / (N + 2) :=
by
  sorry

end simplify_expression_l88_88833


namespace question_1_question_2_question_3_l88_88491

section Sequences
variables {a : ℕ → ℝ} {c d : ℕ → ℝ} {p : ℕ}
variable (S : ℕ → ℝ)
variable (n m : ℕ)
variable [fact (0 : ℝ < a 1)] [fact (0 : ℝ < a 2)]
variable [fact (3 <= p)]

-- Assume the given conditions
axiom cond_1 (h : 0 < m ∧ 0 < n) : (S (m + n) + S 1)^2 = 4 * (a (2 * m) * a (2 * n))
axiom cond_2 : ∀ n, a n > 0
axiom cond_3 (n : ℕ) : abs (c n) = a n ∧ abs (d n) = a n
axiom cond_4 : T p = R p

-- Define the sums T and R
def T (p : ℕ) := ∑ i in Finset.range p, c i
def R (p : ℕ) := ∑ i in Finset.range p, d i

-- Define the properties we want to prove
theorem question_1 : a 2 / a 1 = 2 := sorry
theorem question_2 (n : ℕ) : a n = a 1 * (2 ^ (n - 1)) := sorry
theorem question_3 (k : ℕ) (hk : 1 ≤ k ∧ k ≤ p) : c k = d k := sorry

end Sequences

end question_1_question_2_question_3_l88_88491


namespace ordered_pairs_2028_l88_88243

theorem ordered_pairs_2028 :
    ∃! (n : ℕ), n = 27 ∧ (∀ (x y : ℕ), (prime_factors 2028 = [2^2, 3^2, 13^2] → xy = 2028 → true)) :=
begin
    sorry
end

end ordered_pairs_2028_l88_88243


namespace probability_three_heads_l88_88323

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l88_88323


namespace committee_count_l88_88794

theorem committee_count (club_members founding_members : ℕ) [fact (club_members = 30)] [fact (founding_members = 10)]
  : ∀ (k : ℕ) [fact (k = 5)], 
  choose club_members k - choose (club_members - founding_members) k = 126992 :=
by
  intros
  rw [fact.out (club_members = 30), fact.out (founding_members = 10), fact.out (k = 5)]
  norm_num
  sorry

end committee_count_l88_88794


namespace no_odd_digits_1_to_9999_l88_88124

-- Lean statement for the given mathematical problem
theorem no_odd_digits_1_to_9999 : 
  let digits := [0, 2, 4, 6, 8]
  in ∀ n : ℕ, (1 ≤ n ∧ n ≤ 9999) →
  (∀ d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10], d ∈ digits) →
  (finset.filter (λ k, (∀ d ∈ [k / 1000 % 10, k / 100 % 10, k / 10 % 10, k % 10], d ∈ digits)) (finset.range 10000).to_set).card = 624 :=
by
  sorry

end no_odd_digits_1_to_9999_l88_88124


namespace value_of_f_g_3_l88_88545

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x^2 - 2*x + 1

theorem value_of_f_g_3 : f (g 3) = 2134 :=
by 
  sorry

end value_of_f_g_3_l88_88545


namespace inequality_holds_iff_rk_equals_ak_l88_88089

theorem inequality_holds_iff_rk_equals_ak
  {n : ℕ} {a r x : Fin n → ℝ}
  (h_nonzero : ∀ i, a i ≠ 0)
  (h_ineq : ∀ x, 
     (∑ k : Fin n, r k * (x k - a k)) 
     ≤ (∑ k : Fin n, x k ^ 2) ^ (1/2) 
     - (∑ k : Fin n, a k ^ 2) ^ (1/2)) :
  (∀ k, r k = a k / (∑ k, a k ^ 2) ^ (1/2)) ↔
  (∑ k, r k ^ 2 = 1) :=
sorry

end inequality_holds_iff_rk_equals_ak_l88_88089


namespace seating_arrangements_count_l88_88224

theorem seating_arrangements_count :
  ∃ n : ℕ, n = 10010 ∧
  ∃ (teachers : Finset ℕ), 
    teachers.card = 10 ∧
    ∀ (i : ℕ), (1 ≤ i ∧ i ≤ 10) → i ∈ teachers ∧
    ∀ (seats : Finset ℕ),
      seats.card = 25 →
      ∀ (assignments : teachers → Finset ℕ),
        (∀ i, assignments ⟨i, sorry⟩.card = 1)
        ∧ (∀ i j (hij : i ≠ j), Disjoint (assignments ⟨i, sorry⟩) (assignments ⟨j, sorry⟩))
        ∧ (∀ i, (assignments ⟨i, sorry⟩) ∈ seats)
        ∧ (∀ i, (assignments ⟨(i % 10) + 1, sorry⟩ ∪ (assignments ⟨(i % 10) + 1, sorry⟩)) ≠ ∅)
        ∧ ((assignments ⟨1, sorry⟩ ∧ assignments ⟨10, sorry⟩ ≠ ∅)) → assignments = 10010 := sorry

end seating_arrangements_count_l88_88224


namespace find_smallest_angle_exists_l88_88066

noncomputable def smallest_angle (θ : ℝ) : Prop :=
  θ > 0 ∧ cos θ = sin (45 * (π / 180)) + cos (30 * (π / 180)) - sin (18 * (π / 180)) - cos (12 * (π / 180))

theorem find_smallest_angle_exists : ∃ θ > 0, smallest_angle θ := 
sorry

end find_smallest_angle_exists_l88_88066


namespace range_of_t_l88_88190

theorem range_of_t (a : ℝ) (t : ℝ) (f : ℝ → ℝ) (f_def : ∀ x, f x = (1/2) * x^2 - 4 * x + a * (ln x)) :
  (∀ x1 x2, x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ (x^2 - 4 * x + a = 0) ∧ (f(x1) + f(x2) ≥ x1 + x2 + t)) → t ≤ -13 :=
by { sorry }

end range_of_t_l88_88190


namespace dot_product_EC_ED_l88_88589

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l88_88589


namespace area_of_region_below_diagonal_of_triangle_l88_88808

theorem area_of_region_below_diagonal_of_triangle:
  let A := (0, 12)
  let B := (0, 0)
  let C := (12, 0)
  let D := (12, 12)
  let F := (24, 0)
  -- The area of the region below the diagonal of the triangle DF inside the square ADBC
  area_region: ℕ := 72
  in area_region = 72 := sorry

end area_of_region_below_diagonal_of_triangle_l88_88808


namespace binom_20_10_l88_88902

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l88_88902


namespace probability_three_heads_l88_88289

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l88_88289


namespace average_speed_l88_88393

-- Define the initial and final odometer readings, and the times spent riding.
def initial_odometer : Nat := 1221
def final_odometer : Nat := 1881
def time_first_day : Nat := 5
def time_second_day : Nat := 7

-- Define the total distance traveled and total time spent.
def total_distance : Nat := final_odometer - initial_odometer
def total_time : Nat := time_first_day + time_second_day

-- State the theorem to be proven.
theorem average_speed : total_distance / total_time = 55 := by
  -- Explicitly define calculations
  have total_distance_calc : total_distance = 1881 - 1221 := by rfl
  have total_time_calc : total_time = 5 + 7 := by rfl
  -- Simplify the expressions
  have distance_simplified : total_distance = 660 := by
    rw total_distance_calc
    exact Nat.sub_self_eq 1881 1221 (by norm_num)
  have time_simplified : total_time = 12 := by
    rw total_time_calc
    exact Nat.add_self_eq 5 7

  -- Compute average speed and compare to expected 55
  show total_distance / total_time = 55
  rw [distance_simplified, time_simplified]
  exact Nat.div_eq 660 12 (by norm_num)


end average_speed_l88_88393


namespace probability_1_lt_X_lt_2_l88_88888

noncomputable theory

-- Assume X is a random variable following normal distribution N(1, 4)
def X : ProbabilityTheory.ProbMeasure ℝ := 
  ProbabilityTheory.ProbMeasure.normal 1 (real.sqrt 4)  -- N(mean=1, std=sqrt(variance))

-- Given P(X < 2) = 0.72
axiom P_X_less_than_2 : ProbabilityTheory.ProbMeasure.cdf X 2 = 0.72

-- The theorem to be proven
theorem probability_1_lt_X_lt_2 : 
  ProbabilityTheory.ProbMeasure.prob (set.Ioo 1 2) = 0.22 :=
sorry

end probability_1_lt_X_lt_2_l88_88888


namespace wilson_total_withdrawal_l88_88732

noncomputable def total_withdrawal (initial_balance : ℝ) : ℝ :=
  let first_deposit := 0.10 * initial_balance
  let after_first_deposit := initial_balance + first_deposit
  
  let first_withdrawal := 0.15 * after_first_deposit
  let after_first_withdrawal := after_first_deposit - first_withdrawal
  
  let second_deposit := 0.25 * initial_balance
  let after_second_deposit := after_first_withdrawal + second_deposit
  
  let final_balance := initial_balance + 16
  let before_final_deposit := final_balance / 1.30
  
  let total_withdraw := after_second_deposit - before_final_deposit
  
  total_withdraw

theorem wilson_total_withdrawal :
  total_withdrawal 150 ≈ 50.06 :=
by
  sorry

end wilson_total_withdrawal_l88_88732


namespace min_value_of_f_l88_88878

noncomputable def f (x : ℝ) : ℝ := (real.exp x - 1) ^ 2 + (real.exp (-x) - 1) ^ 2

theorem min_value_of_f : ∃ x : ℝ, f x = 0 := 
begin
  use 0,
  -- Here we should show that f(0) = 0
  simp [f, real.exp_zero],
  norm_num,
end

end min_value_of_f_l88_88878


namespace find_a_plus_b_l88_88097

-- Define that a is a prime number, b is an odd number, and they satisfy a^2 + b = 2001.
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem find_a_plus_b (a b : ℕ) (h1 : is_prime a) (h2 : is_odd b) (h3 : a^2 + b = 2001) : a + b = 1999 :=
sorry

end find_a_plus_b_l88_88097


namespace fruit_drink_total_volume_l88_88336

theorem fruit_drink_total_volume (T : ℝ) (h_orange : 0.15 * T) (h_watermelon : 0.60 * T) (h_grape : 0.25 * T = 35) :
  T = 140 :=
sorry

end fruit_drink_total_volume_l88_88336


namespace sequence_solution_l88_88062

theorem sequence_solution (a : ℕ → ℝ) :
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m + n) = a m + a n - m * n) ∧ 
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m * n) = m^2 * a n + n^2 * a m + 2 * a m * a n) →
    (∀ n, a n = -n*(n-1)/2) ∨ (∀ n, a n = -n^2/2) :=
  by
  sorry

end sequence_solution_l88_88062


namespace count_arrangements_of_students_l88_88729

theorem count_arrangements_of_students :
  (∃ (A B C D E F G : Type),
    ∀ (positions : Fin 7 → A ⊕ B ⊕ C ⊕ D ⊕ E ⊕ F ⊕ G),
      positions ⟨3, sorry⟩ = (Sum.inl A) ∧
      ((∃ p : Fin 6, (positions p = Sum.inr (Sum.inl (B ⊕ C)) ∧
      (positions (p+1) = Sum.inr (Sum.inr (Σ _ : B ⊕ C → Prop, sorry)))) ∨
      (positions (p+1) = Sum.inr (Sum.inl (B ⊕ C)) ∧
      (positions p = Sum.inr (Sum.inr (Σ _ : B ⊕ C → Prop, sorry))))) ∧
      (∀ q : Fin 7, q ≠ ⟨3, sorry⟩ ∧ q ≠ p ∧ q ≠ (p + 1) →
       positions q = Sum.inr (Sum.inl (D ⊕ E ⊕ F ⊕ G)))) →
  192 := sorry

end count_arrangements_of_students_l88_88729


namespace tangent_no_intersection_l88_88461

variable {F : Type*} [linear_ordered_field F]
variables (A B C O D E K : F)
variables (R a : F)
variable (h1 : (AB = AC))
variable (h2 : (OA = a))
variable (h3 : (D = midpoint O A))
variable (h4 : (E = midpoint O A))
variable (h5 : (K = (DE ∩ OA)))

theorem tangent_no_intersection (h_nonneg : 0 < a ∧ 0 < R) : 
  ( (a^2 + R^2) / (2 * a) > R ) :=
by
  sorry

end tangent_no_intersection_l88_88461


namespace probability_three_heads_l88_88325

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l88_88325


namespace constant_term_in_expansion_l88_88276

theorem constant_term_in_expansion : 
  (∃ c : ℕ, (c = 
    ∑ k in finset.range (11), 
    (nat.choose 10 k) * ((√x ^ k) * ((5 / x) ^ (10 - k))) ∧ 2 * (2 * k = 10))) ∧ 
    (c = 787500) := by
  sorry

end constant_term_in_expansion_l88_88276


namespace num_distinct_terms_expanded_l88_88131

-- Define the multinomial expression and the problem
def multinomial_expr (x : ℝ) : ℝ := 4 * x^3 + x^(-3) + 2

-- State the main theorem
theorem num_distinct_terms_expanded (n : ℕ) (h : n = 2016) : 
  let expr := (multinomial_expr x)^n in
  -- Express that the number of distinct terms is equal to 4033
  -- when expanded and like terms are combined
  (number_of_distinct_terms expr) = 4033 :=
sorry

end num_distinct_terms_expanded_l88_88131


namespace percent_owning_only_cats_l88_88157

theorem percent_owning_only_cats (total_students dogs cats both : ℕ) (h1 : total_students = 500)
  (h2 : dogs = 150) (h3 : cats = 80) (h4 : both = 25) : (cats - both) / total_students * 100 = 11 :=
by
  sorry

end percent_owning_only_cats_l88_88157


namespace find_pq_l88_88416

noncomputable def area_of_triangle (p q : ℝ) : ℝ := 1/2 * (12 / p) * (12 / q)

theorem find_pq (p q : ℝ) (hp : p > 0) (hq : q > 0) (harea : area_of_triangle p q = 12) : p * q = 6 := 
by
  sorry

end find_pq_l88_88416


namespace domain_of_g_l88_88514

/-- Given that the domain of the function f(x+2) is (-3, 4), prove that the domain of the function g(x) = f(x) / sqrt(x-1) is (1, 6). -/
theorem domain_of_g {f : ℝ → ℝ} :
  (∀ x, -3 < x ∧ x < 4 → f(x + 2) ∈ set.univ) → 
  (∀ x, (1 < x ∧ x < 6) ↔ (x > 1 ∧ f(x) ∈ set.univ ∧ ∃ y, f(x+2) = y ∧ x - 1 > 0)) :=
begin
  sorry,
end

end domain_of_g_l88_88514


namespace value_of_s_l88_88173

theorem value_of_s {r s : ℝ} (h : 2 * (Complex.i + 4) + r * Complex.i + s = 0) : s = 26 := 
sorry

end value_of_s_l88_88173


namespace bah_equivalent_to_1500_yahs_l88_88992

theorem bah_equivalent_to_1500_yahs :
  (∃ (bahs rahs yahs : Type) [hasMul bahs] [hasMul rahs] [hasMul yahs],
    (20 : bahs) = 36 * (1 : rahs) ∧ 
    (1 : rahs) = 20 * (1 : yahs) / 12 →
    1500 * (1 : yahs) = 500 * (1 : bahs)) :=
begin
  sorry
end

end bah_equivalent_to_1500_yahs_l88_88992


namespace part1_part2_l88_88483

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 1 / (x + a)

theorem part1 (a x : ℝ):
  a ≥ 1 → x > 0 → f x a ≥ 0 := 
sorry

theorem part2 (a : ℝ):
  0 < a ∧ a ≤ 2 / 3 → ∃! x, x > -a ∧ f x a = 0 :=
sorry

end part1_part2_l88_88483


namespace edward_remaining_money_l88_88433

def initial_amount : ℕ := 19
def spent_amount : ℕ := 13
def remaining_amount : ℕ := initial_amount - spent_amount

theorem edward_remaining_money : remaining_amount = 6 := by
  sorry

end edward_remaining_money_l88_88433


namespace dot_product_EC_ED_l88_88591

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l88_88591


namespace find_angles_and_sides_l88_88877

noncomputable def angle_a {a b c : ℝ} (h1 : a * sin C - sqrt 3 * c * cos A = 0)
  [triangle : ∀ A B C : ℝ, angle A ∧ angle B ∧ angle C] : Prop :=
  A = π / 3

noncomputable def sides_bc {a b c : ℝ} (h1 : a * sin C - sqrt 3 * c * cos A = 0)
  (h2 : a = 2) (area : triangle_area = sqrt 3) : Prop :=
  b = 2 ∧ c = 2

theorem find_angles_and_sides
  (a b c A B C : ℝ)
  (h1 : a * sin C - sqrt 3 * c * cos A = 0)
  (h2 : a = 2)
  (area : triangle_area = sqrt 3)
  [triangle : ∀ A B C : ℝ, angle A ∧ angle B ∧ angle C] :
  angle_a h1 ∧ sides_bc h1 h2 area := by
  sorry

end find_angles_and_sides_l88_88877


namespace maximize_power_speed_l88_88694

variable (C S ρ v₀ : ℝ)

-- Given the formula for force F
def force (v : ℝ) : ℝ := (C * S * ρ * (v₀ - v)^2) / 2

-- Given the formula for power N
def power (v : ℝ) : ℝ := force C S ρ v₀ v * v

theorem maximize_power_speed : ∀ C S ρ v₀ : ℝ, ∃ v : ℝ, v = v₀ / 3 ∧ (∀ v' : ℝ, power C S ρ v₀ v ≤ power C S ρ v₀ v') :=
by
  sorry

end maximize_power_speed_l88_88694


namespace CK_bisects_arc_BC_of_BOC_l88_88039

-- Definitions of the given conditions
variables (A B C O K P : Type) [triangle ABC]
variables (circumcenter O ABC)
variables (circumcircle_O_BOC : Type) [circumcircle B O C]
variables (BP_parallel_AC : BP ∥ AC)
variables (K_on_AP : lies_on K (segment A P))
variables (BK_eq_BC : distance B K = distance B C)
variables (angle_A_gt_60 : angle A > 60)

-- Statement to prove
theorem CK_bisects_arc_BC_of_BOC :
  bisects (segment C K) (arc BC circumcircle_O_BOC) :=
sorry

end CK_bisects_arc_BC_of_BOC_l88_88039


namespace writer_birth_day_l88_88231

theorem writer_birth_day : 
  ∀ (birth_anniversary_day : nat) (years : nat),
  (birth_anniversary_day = 4) → 
  (years = 250) → 
  ((years / 4 - (years / 100 - years / 400)) × 2 + (years - (years / 4 - (years / 100 - years / 400)))) % 7 = 2 →  
  day_of_week (birth_anniversary_day - ((years / 4 - (years / 100 - years / 400)) * 2 + (years - (years / 4 - (years / 100 - years / 400)))) / 7) = 2 := 
by 
  sorry

end writer_birth_day_l88_88231


namespace cost_price_of_one_toy_l88_88801

-- Definitions translating the conditions into Lean
def total_revenue (toys_sold : ℕ) (price_per_toy : ℕ) : ℕ := toys_sold * price_per_toy
def gain (cost_per_toy : ℕ) (toys_gained : ℕ) : ℕ := cost_per_toy * toys_gained

-- Given the conditions in the problem
def total_cost_price_of_sold_toys := 18 * (1300 : ℕ)
def gain_from_sale := 3 * (1300 : ℕ)
def selling_price := total_cost_price_of_sold_toys + gain_from_sale

-- The target theorem we want to prove
theorem cost_price_of_one_toy : (selling_price = 27300) → (1300 = 27300 / 21) :=
by
  intro h
  sorry

end cost_price_of_one_toy_l88_88801


namespace maximum_number_of_rooks_l88_88270

-- Define the chessboard game setting
def rook_placement_condition (n : ℕ) : Prop :=
  ∀ (R : set (fin n × fin n)), (∀ r1 r2 ∈ R, r1 ≠ r2 → ∃ x y, x ∈ set.range (λ i, (i, i)) ∧ y ∉ R ∧ (r1 = x ∨ r2 = x)) → (R.card ≤ (if n % 2 = 1 then (3 * n - 1) / 2 else (3 * n - 2) / 2))

-- Proof statement for the maximum rooks
theorem maximum_number_of_rooks (n : ℕ) : rook_placement_condition n :=
sorry

end maximum_number_of_rooks_l88_88270


namespace john_receives_correct_amount_l88_88674

def ratio := [3, 5, 7, 2, 4]
def total_amount := 12000
def john_share := 3

theorem john_receives_correct_amount :
  ((john_share / (3 + 5 + 7 + 2 + 4).toFloat) * total_amount.toFloat).toInt = 1714 :=
by
  sorry

end john_receives_correct_amount_l88_88674


namespace problem_statement_l88_88033

def sum_odds_from_1_to_2023 : ℕ := 
  let n := (2023 - 1) / 2 + 1 in n * n

def sum_evens_from_2_to_2022 : ℕ := 
  let n := (2022 - 2) / 2 + 1 in n * (n + 1)

def result : ℕ := sum_odds_from_1_to_2023 - sum_evens_from_2_to_2022 - 2025

theorem problem_statement : result = 7 := by
  sorry

end problem_statement_l88_88033


namespace find_speeds_l88_88473

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l88_88473


namespace sum_of_digits_of_N_l88_88683

def sequence_sum_n : ℕ :=
  (0 to 50).sum (λ n, 10^n + 1)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum
    
theorem sum_of_digits_of_N : sum_of_digits sequence_sum_n = 58 :=
  sorry

end sum_of_digits_of_N_l88_88683


namespace average_minutes_run_per_day_l88_88404

theorem average_minutes_run_per_day (f : ℕ) :
  let third_grade_minutes := 12
  let fourth_grade_minutes := 15
  let fifth_grade_minutes := 10
  let third_graders := 4 * f
  let fourth_graders := 2 * f
  let fifth_graders := f
  let total_minutes := third_graders * third_grade_minutes + fourth_graders * fourth_grade_minutes + fifth_graders * fifth_grade_minutes
  let total_students := third_graders + fourth_graders + fifth_graders
  total_minutes / total_students = 88 / 7 :=
by
  sorry

end average_minutes_run_per_day_l88_88404


namespace distribute_percentage_each_child_l88_88000

noncomputable def percentage_distributed_to_each_child (income : ℝ) : ℝ :=
let deposited_to_wife := 0.20 * income in
let remaining_after_deposit := income - deposited_to_wife in
let donated_to_orphan_house := 0.10 * remaining_after_deposit in
let remaining_after_donation := remaining_after_deposit - donated_to_orphan_house in
let final_amount := 500 in
let amount_distributed_to_children := remaining_after_donation - final_amount in
(amount_distributed_to_children / 2) / income * 100

theorem distribute_percentage_each_child (income : ℝ) (h_income : income = 1000) :
  percentage_distributed_to_each_child income = 11 :=
by
  rw [h_income]
  sorry

end distribute_percentage_each_child_l88_88000


namespace f_prime_at_0_l88_88408

def f : ℝ → ℝ :=
λ x, if x = 0 then 0 else exp (x * sin (5 * x)) - 1

noncomputable def f_prime_0 : ℝ :=
lim
  (λ Δx, (f Δx - f 0) / Δx)
  (nhds 0)

theorem f_prime_at_0 : f_prime_0 = 0 :=
sorry

end f_prime_at_0_l88_88408


namespace distinct_four_digit_numbers_product_18_l88_88980

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l88_88980


namespace interval_representation_l88_88691

def S : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem interval_representation : S = Set.Ioc (-1) 3 :=
sorry

end interval_representation_l88_88691


namespace subject_representatives_count_l88_88673

-- Define the problem context
def number_of_students : ℕ := 5
def number_of_selected_students : ℕ := 4
def number_of_subjects : ℕ := 3
def students : finset (fin 5) := finset.univ  -- represents the 5 students

-- Define the subject representative function and constraints
def chinese (s : finset (fin 5)) : Prop := s.card = 1
def foreign_language (s : finset (fin 5)) : Prop := s.card = 1
def mathematics (s : finset (fin 5)) : Prop := s.card = 2

-- State the main theorem
theorem subject_representatives_count : 
  ∃ S₁ S₂ S₃ : finset (fin 5), S₁ ∪ S₂ ∪ S₃ = students ∧ 
    chinese S₁ ∧ foreign_language S₂ ∧ mathematics S₃ ∧ S₁ ∩ S₂ = ∅ ∧ S₁ ∩ S₃ = ∅ ∧ 
    S₂ ∩ S₃ = ∅ ∧ S₁.card + S₂.card + S₃.card = number_of_selected_students ∧ 
    number_of_ways = 60 :=
begin
  sorry
end

end subject_representatives_count_l88_88673


namespace newLampTaller_l88_88612

-- Define the heights of the old and new lamps
def oldLampHeight : ℝ := 1
def newLampHeight : ℝ := 2.33

-- Define the proof statement
theorem newLampTaller : newLampHeight - oldLampHeight = 1.33 :=
by
  sorry

end newLampTaller_l88_88612


namespace last_score_entered_is_92_l88_88206

theorem last_score_entered_is_92 (scores : List ℕ) (h_len : scores.length = 6)
  (h_scores : scores ~ [72, 77, 81, 83, 92, 95])
  (h_integer_averages : ∀ (l : List ℕ), l ⊆ scores → l.length ≠ 0 →
    ∃ r : ℕ, List.sum l = r * l.length) :
  List.last scores (by simp [h_len]) = 92 :=
by
  sorry

end last_score_entered_is_92_l88_88206


namespace rectangle_area_ratio_l88_88865

theorem rectangle_area_ratio (a : ℝ) :
  let square_area := a^2
  let rect_width := a
  let rect_length := 2 * a
  let rect_area := rect_width * rect_length
  let combined_rect_area := 2 * rect_area
  combined_rect_area / square_area = 4 :=
by
  let square_area := a^2
  let rect_width := a
  let rect_length := 2 * a
  let rect_area := rect_width * rect_length
  let combined_rect_area := 2 * rect_area
  have h1 : square_area = a^2 := by rfl
  have h2 : combined_rect_area = 4 * a^2 := by 
    rw [rect_area, rect_width, rect_length]
    calc
      2 * (a * (2 * a)) = 2 * (2 * a^2) : by ring
      ... = 4 * a^2 : by ring
  calc
    combined_rect_area / square_area
      = (4 * a^2) / (a^2) : by rw [h1, h2]
      ... = 4 : by ring

end rectangle_area_ratio_l88_88865


namespace product_of_real_roots_eq_one_l88_88712

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, x ^ Real.log x = Real.exp 1 → (x = Real.exp 1 ∨ x = Real.exp (-1)) →
  x * (if x = Real.exp 1 then Real.exp (-1) else Real.exp 1) = 1 :=
by sorry

end product_of_real_roots_eq_one_l88_88712


namespace grain_demand_prediction_l88_88824

theorem grain_demand_prediction (x : ℕ) (y : ℕ) (h1 : x = 2012) (h2 : y = 6.5 * (x - 2006) + 261) : y = 300 := by
  sorry

end grain_demand_prediction_l88_88824


namespace compute_b100_l88_88047

theorem compute_b100 :
  let b : ℕ → ℝ := λ n, if n = 0 then 1 else 4^n
  in b 99 = 4^99 :=
by {
  let b : ℕ → ℝ := λ n, if n = 0 then 1 else 4^n,
  sorry
}

end compute_b100_l88_88047


namespace increasing_iff_m_range_l88_88110

def f (m x : ℝ) : ℝ := (1 / 3) * x ^ 3 - (4 * m - 1) * x ^ 2 + (15 * m ^ 2 - 2 * m - 7) * x + 2

theorem increasing_iff_m_range (m : ℝ) : (∀ x y : ℝ, f m x ≤ f m y) ↔ (2 ≤ m ∧ m ≤ 4) :=
sorry

end increasing_iff_m_range_l88_88110


namespace katie_earnings_l88_88618

theorem katie_earnings 
  (bead_necklaces : ℕ)
  (gem_stone_necklaces : ℕ)
  (bead_cost : ℕ)
  (gem_stone_cost : ℕ)
  (h1 : bead_necklaces = 4)
  (h2 : gem_stone_necklaces = 3)
  (h3 : bead_cost = 5)
  (h4 : gem_stone_cost = 8) :
  (bead_necklaces * bead_cost + gem_stone_necklaces * gem_stone_cost = 44) :=
by
  sorry

end katie_earnings_l88_88618


namespace triangle_largest_angle_l88_88706

theorem triangle_largest_angle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180)
  (h2 : ∃ k, 3 * k + 4 * k + 5 * k = 180) :
  5 * k = 75 :=
sorry

end triangle_largest_angle_l88_88706


namespace exists_regular_tetrahedron_on_planes_l88_88485

-- Definitions of planes
variable {E1 E2 E3 E4 : Plane}
-- Definitions of distance between planes
variable (d1 d2 d3 : ℝ)
-- Conditions: four non-overlapping, parallel planes
variable (h1 : E1 ≠ E2) (h2 : E2 ≠ E3) (h3 : E3 ≠ E4)
variable (p1 : isParallel E1 E2) (p2 : isParallel E2 E3) (p3 : isParallel E3 E4)

-- Statement: There exists a regular tetrahedron with vertices on the planes
theorem exists_regular_tetrahedron_on_planes
  (E1 E2 E3 E4 : Plane) (d1 d2 d3 : ℝ)
  (h1 : E1 ≠ E2) (h2 : E2 ≠ E3) (h3 : E3 ≠ E4)
  (p1 : isParallel E1 E2) (p2 : isParallel E2 E3) (p3 : isParallel E3 E4) :
  ∃ (P1 P2 P3 P4 : Point3D), 
  P1 ∈ E1 ∧ P2 ∈ E2 ∧ P3 ∈ E3 ∧ P4 ∈ E4 ∧ is_regular_tetrahedron P1 P2 P3 P4 :=
sorry

end exists_regular_tetrahedron_on_planes_l88_88485


namespace number_of_ways_to_sum_75_l88_88570

theorem number_of_ways_to_sum_75 : 
  ∃ (N : ℕ → ℕ) (n : ℕ), 
    (∀ (N' : ℕ), 
      (N' < 12) → 
      (∃ (A' : ℚ), N' * A' = 75 → 
      (N = λ n, (N' ≤ n) ∧ (n ≤ 2 * (75 / N'))))) ↔ 
      n = 5 := 
  sorry

end number_of_ways_to_sum_75_l88_88570


namespace dot_product_EC_ED_l88_88593

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l88_88593


namespace seeds_germination_percentage_l88_88868

theorem seeds_germination_percentage :
  (let seeds1 := 300
       seeds2 := 200
       seeds3 := 400
       germination1 := 0.15
       germination2 := 0.35
       germination3 := 0.25
       germinated1 := germination1 * seeds1
       germinated2 := germination2 * seeds2
       germinated3 := germination3 * seeds3
       total_seeds := seeds1 + seeds2 + seeds3
       total_germinated := germinated1 + germinated2 + germinated3
   in (total_germinated / total_seeds) * 100 = 23.89) :=
by
  sorry

end seeds_germination_percentage_l88_88868


namespace physical_fitness_stats_fitness_test_probs_l88_88160

theorem physical_fitness_stats :
  let scores := [38, 41, 44, 51, 54, 56, 58, 64, 74, 80],
      sum_of_squares := 33050,
      avg := (List.sum scores : ℝ) / (List.length scores : ℝ),
      variance := (sum_of_squares : ℝ) / (List.length scores : ℝ) - avg^2
  in avg = 56 ∧ variance = 161 := by
  let scores := [38, 41, 44, 51, 54, 56, 58, 64, 74, 80]
  let sum_of_squares := 33050
  let n := (List.length scores : ℝ)
  have avg : ℝ := (List.sum scores : ℝ) / n
  have avg_eq : avg = 56 := by
    sorry
  have variance : ℝ := sum_of_squares / n - avg^2
  have variance_eq : variance = 161 := by
    sorry
  exact ⟨avg_eq, variance_eq⟩

theorem fitness_test_probs :
  let scores := [38, 41, 44, 51, 54, 56, 58, 64, 74, 80],
      unqualified := (List.countp (· < 50) scores),
      n := List.length scores,
      N := 10,
      prob :=
        let C (n k : ℕ) := Nat.choose n k
        in
        [ (C (n - unqualified) 3 : ℝ) / C n 3,
          (C (n - unqualified) 2 * C unqualified 1 : ℝ) / C n 3,
          (C (n - unqualified) 1 * C unqualified 2 : ℝ) / C n 3,
          (C unqualified 3 : ℝ) / C n 3
        ],
      E := prob[1] * (1 : ℝ) + prob[2] * (2 : ℝ) + prob[3] * (3 : ℝ)
  in prob = [7/24, 21/40, 7/40, 1/120] ∧ E = 9/10 := by
  let scores := [38, 41, 44, 51, 54, 56, 58, 64, 74, 80]
  let unqualified := (scores.filter (· < 50)).length
  let n := scores.length
  let C (n k : ℕ) := Nat.choose n k
  have probs : List ℝ := [
    (C (n - unqualified) 3 : ℝ) / C n 3,
    (C (n - unqualified) 2 * C unqualified 1 : ℝ) / C n 3,
    (C (n - unqualified) 1 * C unqualified 2 : ℝ) / C n 3,
    (C unqualified 3 : ℝ) / C n 3
  ]
  have probs_eq : probs = [7/24, 21/40, 7/40, 1/120] := by
    sorry
  have E : ℝ := probs[1] * 1 + probs[2] * 2 + probs[3] * 3
  have E_eq : E = 9/10 := by
    sorry
  exact ⟨probs_eq, E_eq⟩

end physical_fitness_stats_fitness_test_probs_l88_88160


namespace beyonce_total_songs_l88_88028

-- Define the conditions as given in the problem
def singles : ℕ := 5
def albums_15_songs : ℕ := 2
def songs_per_15_album : ℕ := 15
def albums_20_songs : ℕ := 1
def songs_per_20_album : ℕ := 20

-- Define a function to calculate the total number of songs released by Beyonce
def total_songs_released : ℕ :=
  singles + (albums_15_songs * songs_per_15_album) + (albums_20_songs * songs_per_20_album)

-- Theorem statement for the total number of songs released
theorem beyonce_total_songs {singles albums_15_songs songs_per_15_album albums_20_songs songs_per_20_album : ℕ} :
  singles = 5 →
  albums_15_songs = 2 →
  songs_per_15_album = 15 →
  albums_20_songs = 1 →
  songs_per_20_album = 20 →
  total_songs_released = 55 :=
by {
  intros h_singles h_albums_15_songs h_songs_per_15_album h_albums_20_songs h_songs_per_20_album,
  -- replace with the proven result
  sorry
}

end beyonce_total_songs_l88_88028


namespace distinct_four_digit_integers_with_digit_product_18_l88_88962

theorem distinct_four_digit_integers_with_digit_product_18 : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in digits.prod = 18) → 
  (finset.univ.filter (λ m, (let mdigits := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in mdigits.prod = 18))).card = 36 :=
by
  sorry

end distinct_four_digit_integers_with_digit_product_18_l88_88962


namespace angle_XZY_l88_88165

def midpoint (M Y Z : Point) : Prop :=
  dist M Y = dist M Z

variables {X Y Z M : Point}

theorem angle_XZY (h1 : midpoint M Y Z) (h2 : angle X M Z = 30) (h3 : angle X Y Z = 15) :
  angle X Z Y = 75 :=
by
  sorry

end angle_XZY_l88_88165


namespace inclination_angle_l88_88605

theorem inclination_angle (x y : ℝ) (θ : ℝ) :
  (x + √3 * y + 1 = 0) → θ = 5 * Real.pi / 6 :=
sorry

end inclination_angle_l88_88605


namespace find_n_find_m_constant_term_find_m_max_coefficients_l88_88949

-- 1. Prove that if the sum of the binomial coefficients is 256, then n = 8.
theorem find_n (n : ℕ) (h : 2^n = 256) : n = 8 :=
by sorry

-- 2. Prove that if the constant term is 35/8, then m = ±1/2.
theorem find_m_constant_term (m : ℚ) (h : m^4 * (Nat.choose 8 4) = 35/8) : m = 1/2 ∨ m = -1/2 :=
by sorry

-- 3. Prove that if only the 6th and 7th terms have the maximum coefficients, then m = 2.
theorem find_m_max_coefficients (m : ℚ) (h1 : m ≠ 0) (h2 : m^5 * (Nat.choose 8 5) = m^6 * (Nat.choose 8 6)) : m = 2 :=
by sorry

end find_n_find_m_constant_term_find_m_max_coefficients_l88_88949


namespace inclination_angle_of_line_l88_88717

theorem inclination_angle_of_line (c : ℝ) : 
  ∃ α : ℝ, (0 ≤ α ∧ α < 180 ∧ tan α = sqrt 3 / 3 ∧ α = 30) :=
by
  sorry

end inclination_angle_of_line_l88_88717


namespace derek_books_ratio_l88_88425

theorem derek_books_ratio :
  ∃ (T : ℝ), 960 - T - (1/4) * (960 - T) = 360 ∧ T / 960 = 1 / 2 :=
by
  sorry

end derek_books_ratio_l88_88425


namespace second_part_shorter_l88_88817

def length_wire : ℕ := 180
def length_part1 : ℕ := 106
def length_part2 : ℕ := length_wire - length_part1
def length_difference : ℕ := length_part1 - length_part2

theorem second_part_shorter :
  length_difference = 32 :=
by
  sorry

end second_part_shorter_l88_88817


namespace prob_first_three_heads_all_heads_l88_88319

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l88_88319


namespace b_120_eq_l88_88046

noncomputable def b : ℕ → ℚ
| 1     := 2
| 2     := 1
| (n+1) := (2 - b n) / (3 * b (n-1))

theorem b_120_eq : b 120 = -194 / 3 := by
    sorry

end b_120_eq_l88_88046


namespace find_number_satisfies_l88_88991

noncomputable def find_number (m : ℤ) (n : ℤ) : Prop :=
  (m % n = 2) ∧ (3 * m % n = 1)

theorem find_number_satisfies (m : ℤ) : ∃ n : ℤ, find_number m n ∧ n = 5 :=
by
  sorry

end find_number_satisfies_l88_88991


namespace bounded_area_l88_88409

noncomputable def param_x (t : ℝ) : ℝ := 2 * Real.cos t
noncomputable def param_y (t : ℝ) : ℝ := 6 * Real.sin t

theorem bounded_area :
  (∫ t in (Real.pi / 6)..(5 * Real.pi / 6), param_x t * (derivative param_y t)) = 4 * Real.pi - 3 * Real.sqrt 3 :=
by
  sorry

end bounded_area_l88_88409


namespace binom_20_10_l88_88918

-- Definitions for the provided conditions
def binom_18_8 := 43758
def binom_18_9 := 48620
def binom_18_10 := 43758

-- The theorem we need to prove
theorem binom_20_10 : ∀
  (binom_18_8 = 43758)
  (binom_18_9 = 48620)
  (binom_18_10 = 43758),
  binomial 20 10 = 184756 :=
by
  sorry

end binom_20_10_l88_88918


namespace incorrect_statements_count_l88_88017

theorem incorrect_statements_count :
  (¬ (∀ x, variance (dataset.add x) = variance dataset)) ∧
  (¬ (∀ x, regression (linear_eqn 3 (-5) x) = regression (linear_eqn 3 (-5) x) + 5)) ∧
  (∀ b a x y, regression (linear_eqn b a x) = regression (linear_eqn b a x + x) + y) ∧
  (¬ (∀ p : point, curve_point_correspondence p)) ∧
  (¬ (probability_related (contingency_table 2 2 13.079) = 0.9)) →
  incorrect_statements = 3 :=
begin
  -- Sorry is used as a placeholder for the proof
  sorry
end

end incorrect_statements_count_l88_88017


namespace optimal_messenger_strategy_l88_88208

theorem optimal_messenger_strategy (p : ℝ) (hp : 0 < p ∧ p < 1) :
  (p < 1/3 → (1 - p^3 * (4 - 3*p)) > (1 - p^2) ∧ (1 - p^3 * (4 - 3*p)) > (1 - p^2 * (2 - p)))
  ∧
  (1/3 <= p → (1 - p^2) > (1 - p^3 * (4 - 3*p)) ∧ (1 - p^2) > (1 - p^2 * (2 - p))) :=
by {
  sorry,
}

end optimal_messenger_strategy_l88_88208


namespace finite_decimals_are_rational_l88_88399

-- Conditions as definitions
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_infinite_decimal (x : ℝ) : Prop := ¬∃ (n : ℤ), x = ↑n
def is_finite_decimal (x : ℝ) : Prop := ∃ (a b : ℕ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Equivalence to statement C: Finite decimals are rational numbers
theorem finite_decimals_are_rational : ∀ (x : ℝ), is_finite_decimal x → is_rational x := by
  sorry

end finite_decimals_are_rational_l88_88399


namespace older_brother_catches_younger_brother_l88_88723

theorem older_brother_catches_younger_brother
  (y_time_reach_school o_time_reach_school : ℕ) 
  (delay : ℕ) 
  (catchup_time : ℕ) 
  (h1 : y_time_reach_school = 25) 
  (h2 : o_time_reach_school = 15) 
  (h3 : delay = 8) 
  (h4 : catchup_time = 17):
  catchup_time = delay + ((8 * y_time_reach_school) / (o_time_reach_school - y_time_reach_school) * (y_time_reach_school / 25)) :=
by
  sorry

end older_brother_catches_younger_brother_l88_88723


namespace integral_cos_over_2_plus_cos_l88_88846

-- Define the function to be integrated
def integrand (x : ℝ) : ℝ :=
  cos x / (2 + cos x)

-- Define the limits of integration
def a := 0
def b := π / 2

-- State the definite integral and expected result
theorem integral_cos_over_2_plus_cos :
  ∫ x in a..b, integrand x = (9 - 4 * sqrt 3) * π / 18 :=
by sorry

end integral_cos_over_2_plus_cos_l88_88846


namespace quadratic_roots_l88_88994

noncomputable def roots_quadratic : Prop :=
  ∀ (a b : ℝ), (a + b = 7) ∧ (a * b = 7) → (a^2 + b^2 = 35)

theorem quadratic_roots (a b : ℝ) (h : a + b = 7 ∧ a * b = 7) : a^2 + b^2 = 35 :=
by
  sorry

end quadratic_roots_l88_88994


namespace parabola_directrix_eq_l88_88441

theorem parabola_directrix_eq (x : ℝ) : 
  (∀ y : ℝ, y = 3 * x^2 - 6 * x + 2 → True) →
  y = -13/12 := 
  sorry

end parabola_directrix_eq_l88_88441


namespace dice_sum_prime_probability_l88_88733

noncomputable def probability_prime_sum_dice : ℚ := 9 / 16

theorem dice_sum_prime_probability :
  (∃ (s : set (ℕ × ℕ)), s = {(1, 1), (1, 2), (2, 1), (1, 4), (4, 1), (2, 3), (3, 2), (3, 4), (4, 3)}
  ∧ (∀ (d1 d2 : ℕ), d1 ∈ {1, 2, 3, 4} → d2 ∈ {1, 2, 3, 4} → (d1 + d2) ∈ {2, 3, 5, 7} ↔ (d1, d2) ∈ s))
  → (∃ (prob : ℚ), prob = probability_prime_sum_dice) :=
begin
  sorry
end

end dice_sum_prime_probability_l88_88733


namespace value_of_k_l88_88551

open Real

noncomputable def line := λ k : ℝ, λ (x : ℝ), k * x + 3

def circle := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4}

theorem value_of_k (k : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ circle ∧ B ∈ circle ∧ A ≠ B ∧ (A.1 * B.2 - A.2 * B.1) ^ 2 / (A.1 ^ 2 + A.2 ^ 2) / (B.1 ^ 2 + B.2 ^ 2) = 3 / 4) →
  k = -sqrt 2 ∨ k = sqrt 2 :=
sorry

end value_of_k_l88_88551


namespace only_D_is_quadratic_l88_88747

-- Conditions
def eq_A (x : ℝ) : Prop := x^2 + 1/x - 1 = 0
def eq_B (x : ℝ) : Prop := (2*x + 1) + x = 0
def eq_C (m x : ℝ) : Prop := 2*m^2 + x = 3
def eq_D (x : ℝ) : Prop := x^2 - x = 0

-- Proof statement
theorem only_D_is_quadratic :
  ∃ (x : ℝ), eq_D x ∧ 
  (¬(∃ x : ℝ, eq_A x) ∧ ¬(∃ x : ℝ, eq_B x) ∧ ¬(∃ (m x : ℝ), eq_C m x)) :=
by
  sorry

end only_D_is_quadratic_l88_88747


namespace part_one_part_two_part_three_l88_88082

noncomputable def a_seq (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else if n = 1 then 1 
  else 3 * a_seq (n - 1) + 1

theorem part_one (n : ℕ) : (a_seq (n + 1) + 1/2) = 3 * (a_seq n + 1/2) :=
by sorry

theorem part_two (n : ℕ) : a_seq n = (3^n - 1) / 2 :=
by sorry

theorem part_three (n : ℕ) : 
  (∑ i in Finset.range n, 1 / a_seq (i + 1)) < 3 / 2 :=
by sorry

end part_one_part_two_part_three_l88_88082


namespace area_bound_l88_88255

open Set

-- Define the conditions
variables {n : ℕ} (h_n : n ≥ 3)
variables (C : Fin n → Metric.Sphere (0 : ℝ × ℝ) 1)

-- Condition: Every three circles have at least one pair of intersecting circles
def condition (C : Fin n → Metric.Sphere (0 : ℝ × ℝ) 1) : Prop :=
∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → (Metric.sphere_intersect (C i) (C j) ∨
                                               Metric.sphere_intersect (C j) (C k) ∨
                                               Metric.sphere_intersect (C i) (C k))

-- Main theorem that we want to prove
theorem area_bound (C : Fin n → Metric.Sphere (0 : ℝ × ℝ) 1) (h : condition C) : 
  ∑ i, Fin n, MeasureTheory.measure (C i) < 35 := 
sorry

end area_bound_l88_88255


namespace product_of_sequence_is_256_l88_88032

-- Definitions for conditions
def seq : List ℚ := [1 / 4, 16 / 1, 1 / 64, 256 / 1, 1 / 1024, 4096 / 1, 1 / 16384, 65536 / 1]

-- The main theorem
theorem product_of_sequence_is_256 : (seq.prod = 256) :=
by
  sorry

end product_of_sequence_is_256_l88_88032


namespace find_valid_sets_l88_88130

-- Definition of sets of consecutive integers summing to 30
def valid_consecutive_set_sum (n k : ℕ) : Prop :=
  k > 1 ∧ (k * (2 * n + k - 1)) / 2 = 30

-- Definition for the number of such sets
def num_valid_sets : ℕ :=
  Finset.card ((Finset.range 31).filter (λ k, ∃ n : ℕ, valid_consecutive_set_sum n k))

-- The theorem to prove
theorem find_valid_sets : num_valid_sets = 1 :=
by sorry

end find_valid_sets_l88_88130


namespace square_has_four_axes_of_symmetry_l88_88016

inductive Shape 
| Square
| Rhombus
| Rectangle
| IsoscelesTrapezoid

open Shape

def axes_of_symmetry : Shape → ℕ
| Square := 4
| Rhombus := 2
| Rectangle := 2
| IsoscelesTrapezoid := 1

theorem square_has_four_axes_of_symmetry (s : Shape) :
  s = Square → axes_of_symmetry s = 4 :=
by intro h; rw h; exact rfl

end square_has_four_axes_of_symmetry_l88_88016


namespace continued_fraction_a_l88_88036

theorem continued_fraction_a : 
  [5; (1, 2, 1, 10)] = sqrt 33 := 
  sorry

end continued_fraction_a_l88_88036


namespace required_draws_for_pairs_l88_88761

-- Definitions of the conditions
def red_items := 41
def green_items := 23
def orange_items := 11

-- The main statement/question: Prove that 78 draws are required to obtain a pair of each color.
theorem required_draws_for_pairs : 
  red_items + 1 + green_items + 1 + orange_items + 1 = 78 :=
begin
  -- Given the mathematical conditions:
  -- There are 41 red items, requiring 42 draws to guarantee a pair.
  -- There are 23 green items, requiring 24 draws to guarantee a pair.
  -- There are 11 orange items, requiring 12 draws to guarantee a pair.
  -- Total draws required is:
  calc  41 + 1 + 23 + 1 + 11 + 1 : sorry,
end

end required_draws_for_pairs_l88_88761


namespace imaginary_part_of_complex_num_l88_88277

-- Define the complex number and the imaginary part condition
def complex_num : ℂ := ⟨1, 2⟩

-- Define the theorem to prove the imaginary part is 2
theorem imaginary_part_of_complex_num : complex_num.im = 2 :=
by
  -- The proof steps would go here
  sorry

end imaginary_part_of_complex_num_l88_88277


namespace total_shelves_needed_l88_88177

def regular_shelf_capacity : Nat := 45
def large_shelf_capacity : Nat := 30
def regular_books : Nat := 240
def large_books : Nat := 75

def shelves_needed (book_count : Nat) (shelf_capacity : Nat) : Nat :=
  (book_count + shelf_capacity - 1) / shelf_capacity

theorem total_shelves_needed :
  shelves_needed regular_books regular_shelf_capacity +
  shelves_needed large_books large_shelf_capacity = 9 := by
sorry

end total_shelves_needed_l88_88177


namespace current_prices_l88_88354

theorem current_prices (initial_ram_price initial_ssd_price : ℝ) 
  (ram_increase_1 ram_decrease_1 ram_decrease_2 : ℝ) 
  (ssd_increase_1 ssd_decrease_1 ssd_decrease_2 : ℝ) 
  (initial_ram : initial_ram_price = 50) 
  (initial_ssd : initial_ssd_price = 100) 
  (ram_increase_factor : ram_increase_1 = 0.30 * initial_ram_price) 
  (ram_decrease_factor_1 : ram_decrease_1 = 0.15 * (initial_ram_price + ram_increase_1)) 
  (ram_decrease_factor_2 : ram_decrease_2 = 0.20 * ((initial_ram_price + ram_increase_1) - ram_decrease_1)) 
  (ssd_increase_factor : ssd_increase_1 = 0.10 * initial_ssd_price) 
  (ssd_decrease_factor_1 : ssd_decrease_1 = 0.05 * (initial_ssd_price + ssd_increase_1)) 
  (ssd_decrease_factor_2 : ssd_decrease_2 = 0.12 * ((initial_ssd_price + ssd_increase_1) - ssd_decrease_1)) 
  : 
  ((initial_ram_price + ram_increase_1 - ram_decrease_1 - ram_decrease_2) = 44.20) ∧ 
  ((initial_ssd_price + ssd_increase_1 - ssd_decrease_1 - ssd_decrease_2) = 91.96) := 
by
  sorry

end current_prices_l88_88354


namespace prob_three_heads_is_one_eighth_l88_88328

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l88_88328


namespace maximize_ab2c3_l88_88136

def positive_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def sum_constant (a b c A : ℝ) : Prop :=
  a + b + c = A

noncomputable def maximize_expression (a b c : ℝ) : ℝ :=
  a * b^2 * c^3

theorem maximize_ab2c3 (a b c A : ℝ) (h1 : positive_numbers a b c)
  (h2 : sum_constant a b c A) : 
  maximize_expression a b c ≤ maximize_expression (A / 6) (A / 3) (A / 2) :=
sorry

end maximize_ab2c3_l88_88136


namespace seats_per_row_l88_88714

-- Definitions of conditions
def initial_boarding_count : ℕ := 16
def first_stop_boarding : ℕ := 15
def first_stop_departure : ℕ := 3
def first_stop_net_load := first_stop_boarding - first_stop_departure
def second_stop_boarding : ℕ := 17
def second_stop_departure : ℕ := 10
def second_stop_net_load := second_stop_boarding - second_stop_departure
def total_rows : ℕ := 23
def empty_seats_after_second_stop : ℕ := 57

-- The theorem we need to prove
theorem seats_per_row : ∃ x : ℕ, (total_rows * x - 35 = empty_seats_after_second_stop) ∧ x = 4 :=
by
  use 4
  simp [total_rows, empty_seats_after_second_stop]
  sorry

end seats_per_row_l88_88714


namespace probability_first_three_heads_l88_88300

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l88_88300


namespace sasha_informed_anya_of_product_l88_88671

noncomputable def chosen_numbers : set ℕ := {1, 2, 3, 4, 5, 6, 7}

def product_of_chosen (s : set ℕ) : ℕ :=
  s.fold (λ x y => x * y) 1

theorem sasha_informed_anya_of_product :
  ∃ s : set ℕ, s ⊆ chosen_numbers ∧ s.card = 5 ∧ product_of_chosen s = 420 :=
sorry

end sasha_informed_anya_of_product_l88_88671


namespace collinear_unit_vectors_MN_correct_l88_88500

structure Point (α : Type) := (x : α) (y : α)
def M : Point ℝ := ⟨1, 1⟩
def N : Point ℝ := ⟨4, -3⟩

noncomputable def collinear_unit_vectors 
  (M N : Point ℝ) : set (ℝ × ℝ) :=
  let mn := (N.x - M.x, N.y - M.y)
  let magnitude := Real.sqrt ((mn.1)^2 + (mn.2)^2)
  {(mn.1 / magnitude, mn.2 / magnitude), (-mn.1 / magnitude, -mn.2 / magnitude)}

theorem collinear_unit_vectors_MN_correct :
  collinear_unit_vectors M N = 
  { (3/5, -4/5), (-3/5, 4/5) } := sorry

end collinear_unit_vectors_MN_correct_l88_88500


namespace max_factors_of_2_in_Ik_l88_88454

def Ik (k : ℕ) : ℕ := 10^(k + 2) + 64

def N (k : ℕ) : ℕ :=
  let factors : ℕ → List ℕ := fun n => n.factorization.keys.filter (fun x => x = 2)
  (Ik k).factorization.findWithDefault 2 0

theorem max_factors_of_2_in_Ik (k : ℕ) (hk : k > 0) : N k = 6 :=
sorry

end max_factors_of_2_in_Ik_l88_88454


namespace sum_of_integers_between_2_and_14_l88_88743

theorem sum_of_integers_between_2_and_14 :
  (∑ i in finset.Icc 2 14, i) = 104 :=
by
  sorry

end sum_of_integers_between_2_and_14_l88_88743


namespace no_solution_eqn_l88_88423

def f (n : ℕ) (x : ℝ) : ℝ := (Real.sin x) ^ n + (Real.cos x) ^ n

theorem no_solution_eqn :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi), 
  ¬ (8 * f 4 x - 6 * f 6 x = 4 * f 2 x) :=
by
  sorry

end no_solution_eqn_l88_88423


namespace find_side_length_l88_88120

theorem find_side_length
  (a : ℝ) (B C : ℝ) (h1 : a = real.sqrt 3) (h2 : real.sin B = 1/2) (h3 : C = real.pi / 6) :
  ∃ b : ℝ, b = 1 :=
by 
  sorry

end find_side_length_l88_88120


namespace fifth_coordinate_is_14_l88_88765

theorem fifth_coordinate_is_14
  (a : Fin 16 → ℝ)
  (h_1 : a 0 = 2)
  (h_16 : a 15 = 47)
  (h_avg : ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) :
  a 4 = 14 :=
by
  sorry

end fifth_coordinate_is_14_l88_88765


namespace find_lambda_l88_88537

theorem find_lambda :
  let a := (1, 2)
  let b := (2, 1)
  let c := (5, -2)
  ∃ (λ : ℝ), a - λ • b = (k : ℝ) • c → λ = 4 / 3 := sorry

end find_lambda_l88_88537


namespace answer_l88_88579

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

end answer_l88_88579


namespace baseball_card_value_change_l88_88783

-- Definitions for yearly changes
def year1_decrease (initial_value : ℝ) := initial_value - 0.50 * initial_value
def year2_increase (value_after_year1 : ℝ) := value_after_year1 + 0.30 * value_after_year1
def year3_decrease (value_after_year2 : ℝ) := value_after_year2 - 0.20 * value_after_year2
def year4_increase (value_after_year3 : ℝ) := value_after_year3 + 0.15 * value_after_year3

-- Total percent change calculation
def total_percent_change (initial_value final_value : ℝ) : ℝ :=
  ((final_value - initial_value) / initial_value) * 100

-- Theorem statement
theorem baseball_card_value_change :
  ∀ (initial_value : ℝ),
  total_percent_change initial_value (year4_increase (year3_decrease (year2_increase (year1_decrease initial_value)))) = -40.2 :=
by
  intros
  sorry

end baseball_card_value_change_l88_88783


namespace convex_pentagon_exists_l88_88486

theorem convex_pentagon_exists
  (points : Fin 9 → Point)
  (no_three_collinear : ∀ i j k : Fin 9, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k))
  (convex_hull_is_quadrilateral : ∃ (A1 A2 A3 A4 : Fin 9), convex_hull {points A1, points A2, points A3, points A4} = convex_hull (set_of (λ i, points i))) :
  ∃ (P1 P2 P3 P4 P5 : Fin 9), convex_hull {points P1, points P2, points P3, points P4, points P5}.has_five_sides :=
sorry

end convex_pentagon_exists_l88_88486


namespace ring_is_finite_l88_88804

variables (R : Type*) [Ring R] (nonZeroZeroDivisorExists : ∃ u v : R, u ≠ 0 ∧ v ≠ 0 ∧ u * v = 0) (finiteZeroDivisors : (setOf {a : R | ∃ b : R, a ≠ 0 ∧ b ≠ 0 ∧ a * b = 0}).finite)

theorem ring_is_finite : finite R :=
sorry

end ring_is_finite_l88_88804


namespace non_zero_no_multiple_of_reverse_l88_88755

def reverse_number (n : ℕ) : ℕ := sorry -/ define the function that reverses an integer -/

theorem non_zero_no_multiple_of_reverse (n : ℕ) (h₀ : n ≠ 0) :
  ¬ ∃ a ∈ ({2, 3, 5, 6, 7, 8} : Set ℕ), n = a * reverse_number n := sorry

end non_zero_no_multiple_of_reverse_l88_88755


namespace number_of_ordered_triples_l88_88444

-- Definition of conditions
def condition1 (a b c : ℤ) : Prop :=
  a^2 + b^2 + c^2 - ab - bc - ca - 1 ≤ 4042 * b - 2021 * a - 2021 * c - 2021^2

def condition2 (a b c : ℤ) : Prop :=
  abs a ≤ 2021 ∧ abs b ≤ 2021 ∧ abs c ≤ 2021

-- The theorem statement
theorem number_of_ordered_triples :
  {n : ℕ | ∃ a b c : ℤ, condition1 a b c ∧ condition2 a b c}.card = 14152 :=
  sorry

end number_of_ordered_triples_l88_88444


namespace magnitude_reciprocal_l88_88929

noncomputable def complex_z : ℂ := (Complex.mk (Real.sqrt 3) 1) / ((Complex.mk 1 (-Real.sqrt 3)).pow 2)

theorem magnitude_reciprocal (z : ℂ) (h : z = complex_z) : Complex.abs (1 / z) = 2 :=
by
  rw [h]
  sorry

end magnitude_reciprocal_l88_88929


namespace sine_cosine_fraction_l88_88943

theorem sine_cosine_fraction (θ : ℝ) (h : Matrix.det !![!![Real.sin θ, 2], !![Real.cos θ, 1]] = 0) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 3 :=
by
  sorry

end sine_cosine_fraction_l88_88943


namespace binom_20_10_eq_184756_l88_88906

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l88_88906


namespace fifth_coordinate_is_14_l88_88766

theorem fifth_coordinate_is_14
  (a : Fin 16 → ℝ)
  (h_1 : a 0 = 2)
  (h_16 : a 15 = 47)
  (h_avg : ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) :
  a 4 = 14 :=
by
  sorry

end fifth_coordinate_is_14_l88_88766


namespace probability_product_div_by_4_l88_88227

open Nat finset

def setOfIntegers := (4 : ℕ) +range 17

def isProductDivBy4 (a b : ℕ) : Prop :=
  (a * b) % 4 = 0

theorem probability_product_div_by_4 :
  (∃ (s : Finset ℕ) (h₁ : s = setOfIntegers) (h₂ : s.card = 17),
  let pairs := (s.product s).filter (λ p, p.1 ≠ p.2),
  let favorable := pairs.filter (λ p, isProductDivBy4 p.1 p.2),
  let total_pairs := pairs.card,
  let favorable_count := favorable.card
  in (favorable_count : ℚ) / total_pairs = 15 / 68) := sorry

end probability_product_div_by_4_l88_88227


namespace trapezoid_bases_l88_88661

theorem trapezoid_bases (c d x y : ℝ) (h1 : c > 2 * d) (h2 : (x + (y : ℝ) = c - d)) (h3 : (4 * x * y = d ^ 2)) : 
  (x = (c - d + real.sqrt (c * (c - 2 * d))) / 2) ∧ (y = (c - d - real.sqrt (c * (c - 2 * d))) / 2) :=
by {
  sorry
}

end trapezoid_bases_l88_88661


namespace probability_at_least_2_red_and_non_blue_l88_88782

noncomputable def total_balls : ℕ := 5 + 6 + 7 + 4
noncomputable def red_balls : ℕ := 5
noncomputable def blue_balls : ℕ := 6
noncomputable def non_blue_balls : ℕ := 7 + 4

def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_at_least_2_red_and_non_blue : 
  let total_ways := comb total_balls 3
      ways_2_red_1_non_blue := comb red_balls 2 * comb non_blue_balls 1
      ways_3_red := comb red_balls 3
  in (ways_2_red_1_non_blue + ways_3_red) / total_ways = 12 / 154 := 
by
  -- mathematical proof goes here.
  sorry

end probability_at_least_2_red_and_non_blue_l88_88782


namespace area_union_square_circle_l88_88809

noncomputable def side_length_square : ℝ := 12
noncomputable def radius_circle : ℝ := 15
noncomputable def area_union : ℝ := 144 + 168.75 * Real.pi

theorem area_union_square_circle : 
  let area_square := side_length_square ^ 2
  let area_circle := Real.pi * radius_circle ^ 2
  let area_quarter_circle := area_circle / 4
  area_union = area_square + area_circle - area_quarter_circle :=
by
  -- The actual proof is omitted
  sorry

end area_union_square_circle_l88_88809


namespace remainder_4015_div_32_l88_88280

theorem remainder_4015_div_32 : 4015 % 32 = 15 := by
  sorry

end remainder_4015_div_32_l88_88280


namespace three_heads_in_a_row_l88_88311

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l88_88311


namespace probability_first_three_heads_l88_88303

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l88_88303


namespace probability_first_three_heads_l88_88301

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l88_88301


namespace inequality_inequality_l88_88219

theorem inequality_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end inequality_inequality_l88_88219


namespace proposition_1_proposition_2_proposition_3_proposition_4_l88_88748

theorem proposition_1 : ∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0 := sorry

theorem proposition_2 : ¬ (∀ x ∈ ({-1, 0, 1} : Set ℤ), 2 * x + 1 > 0) := sorry

theorem proposition_3 : ∃ x : ℕ, x^2 ≤ x := sorry

theorem proposition_4 : ∃ x : ℕ, x ∣ 29 := sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l88_88748


namespace distance_AB_l88_88168

def parametric_equation_l (t : ℝ) : ℝ × ℝ :=
  (2 - 3 * t, real.sqrt 3 * t)

def polar_equation_C1 (θ : ℝ) : ℝ :=
  4 * real.cos θ

def polar_equation_C2 : ℝ :=
  real.pi / 6

theorem distance_AB :
  let ρ_A := (2 * real.sqrt 3) / 3,
      ρ_B := 2 * real.sqrt 3
  in |ρ_A - ρ_B| = (4 * real.sqrt 3) / 3 :=
  sorry

end distance_AB_l88_88168


namespace fibonacci_factorial_series_sum_l88_88744

theorem fibonacci_factorial_series_sum : 
  (1! + 1! + 2! + 3! + (8! % 100) + (13! % 100) + (21! % 100) + (34! % 100) + (55! % 100) + (89! % 100) + (144! % 100)) % 100 = 30 := 
by sorry

end fibonacci_factorial_series_sum_l88_88744


namespace equal_circumference_and_perimeter_l88_88780

constant length_wire : ℝ
constant circle_circumference : ℝ → ℝ
constant square_perimeter : ℝ → ℝ

axiom wire_length : length_wire = 2019
axiom circle_formula : circle_circumference length_wire = 2019
axiom square_formula : square_perimeter length_wire = 2019

theorem equal_circumference_and_perimeter :
  circle_circumference length_wire = square_perimeter length_wire :=
by 
  rw [circle_formula, square_formula]
  sorry

end equal_circumference_and_perimeter_l88_88780


namespace binom_20_10_l88_88916

-- Definitions for the provided conditions
def binom_18_8 := 43758
def binom_18_9 := 48620
def binom_18_10 := 43758

-- The theorem we need to prove
theorem binom_20_10 : ∀
  (binom_18_8 = 43758)
  (binom_18_9 = 48620)
  (binom_18_10 = 43758),
  binomial 20 10 = 184756 :=
by
  sorry

end binom_20_10_l88_88916


namespace sum_of_coefficients_l88_88073

theorem sum_of_coefficients (a b c d : ℤ)
  (h1 : a + c = 2)
  (h2 : a * c + b + d = -3)
  (h3 : a * d + b * c = 7)
  (h4 : b * d = -6) :
  a + b + c + d = 7 :=
sorry

end sum_of_coefficients_l88_88073


namespace max_power_speed_l88_88702

def aerodynamic_force (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v)^2) / 2

def power (C S ρ v₀ v : ℝ) : ℝ :=
  aerodynamic_force C S ρ v₀ v * v

theorem max_power_speed (C S ρ v₀ v : ℝ) (h₁ : v = v₀ / 3) :
  ∃ v, power C S ρ v₀ v = (C * S * ρ * v₀^3) / 54 :=
begin
  use v₀ / 3,
  sorry
end

end max_power_speed_l88_88702


namespace infinite_solutions_eq_a_l88_88118

variable (a x y: ℝ)

-- Define the two equations
def eq1 : Prop := a * x + y - 1 = 0
def eq2 : Prop := 4 * x + a * y - 2 = 0

theorem infinite_solutions_eq_a (h : ∃ x y, eq1 a x y ∧ eq2 a x y) :
  a = 2 := 
sorry

end infinite_solutions_eq_a_l88_88118


namespace distinct_four_digit_numbers_product_18_l88_88982

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l88_88982


namespace coffee_mix_price_per_pound_l88_88182

-- Definitions based on conditions
def total_weight : ℝ := 100
def columbian_price_per_pound : ℝ := 8.75
def brazilian_price_per_pound : ℝ := 3.75
def columbian_weight : ℝ := 52
def brazilian_weight : ℝ := total_weight - columbian_weight

-- Goal to prove
theorem coffee_mix_price_per_pound :
  (columbian_weight * columbian_price_per_pound + brazilian_weight * brazilian_price_per_pound) / total_weight = 6.35 :=
by
  sorry

end coffee_mix_price_per_pound_l88_88182


namespace decimal_to_binary_89_l88_88044

theorem decimal_to_binary_89 :
  (∃ (b : Nat), b = 2^6 + 2^4 + 2^3 + 2^0 ∧ nat2bin 89 = b) :=
by
  use 1011001  -- binary representation
  have : 2^6 + 2^4 + 2^3 + 2^0 = 64 + 16 + 8 + 1 := by norm_num
  rw [this]
  sorry  -- placeholder for the proof that nat2bin 89 = 1011001

end decimal_to_binary_89_l88_88044


namespace five_thursdays_in_july_l88_88682

theorem five_thursdays_in_july (N : ℕ) :
  (∃ ts : finset ℕ, ts.card = 5 ∧ ts ⊆ finset.range 30 ∧ 
  (∀ n ∈ ts, (n - 1) % 7 = 1 ∨ (n - 1) % 7 = 2 ∨ (n - 1) % 7 = 3 ∨ (n - 1) % 7 = 4 ∨ 
  (n - 1) % 7 = 5)) →
  ∃ th : finset ℕ, th.card = 5 ∧ th ⊆ finset.range 31 ∧ 
  (∃ July1 : ℕ, (July1 - 1) % 7 = 3 ∧ 
  (∀ d ∈ th, (d + July1 - 1) % 7 = 3)) :=
sorry

end five_thursdays_in_july_l88_88682


namespace pages_after_break_correct_l88_88181

-- Definitions based on conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.7
def pages_before_break : ℕ := (total_pages : ℝ * break_percentage).to_nat
def pages_after_break : ℕ := total_pages - pages_before_break

-- Theorem statement
theorem pages_after_break_correct : pages_after_break = 9 :=
by
  -- The proof is unnecessary as per instructions
  sorry

end pages_after_break_correct_l88_88181


namespace triangle_area_ratio_l88_88151

noncomputable def area_ratio (h : ℝ) : ℝ := 2 * h / (3.5 * h)

theorem triangle_area_ratio :
  ∀ (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D],
    let BD := 4,
    let DC := 7,
    area_ratio 1 = 4 / 7 :=
by
  intros
  unfold area_ratio
  norm_num
  rfl

#lint

end triangle_area_ratio_l88_88151


namespace vector_subtraction_example_l88_88860

theorem vector_subtraction_example : 
  let v1 := ⟨(3 : ℝ), (-8 : ℝ)⟩
  let v2 := ⟨(-2 : ℝ), (6 : ℝ)⟩
  v1 - 3 • v2 = ⟨9, -26⟩ :=
by 
  -- place proof here
  sorry

end vector_subtraction_example_l88_88860


namespace relationship_between_a_and_b_l88_88638

theorem relationship_between_a_and_b
  {a b : ℝ}
  (h1 : 3 ^ a + 13 ^ b = 17 ^ a)
  (h2 : 5 ^ a + 7 ^ b = 11 ^ b) :
  a < b :=
by
  sorry

end relationship_between_a_and_b_l88_88638


namespace smallest_integer_in_set_of_seven_l88_88154

theorem smallest_integer_in_set_of_seven (n : ℤ) (h : n + 6 < 3 * (n + 3)) : n = -1 :=
sorry

end smallest_integer_in_set_of_seven_l88_88154


namespace num_real_solutions_eq_121_l88_88445

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 120).sum (λ n, ((n+1:ℝ)^2) / (x - (n+1:ℝ)))

theorem num_real_solutions_eq_121 : 
  ∃ (n : ℕ), n = 121 ∧ ∀ x : ℝ, f x = x → count_solutions x = n
  sorry

end num_real_solutions_eq_121_l88_88445


namespace staircase_handrail_length_l88_88007

/--
Given a spiral staircase that turns 180 degrees as it rises 15 feet and has a radius of 3 feet,
prove the length of the handrail is approximately 17.7 feet.
-/
theorem staircase_handrail_length :
  let arc_length := 3 * Real.pi in
  let handrail_length := Real.sqrt (15^2 + arc_length^2) in
  handrail_length ≈ 17.7 := by
  sorry

end staircase_handrail_length_l88_88007


namespace gold_coins_proof_l88_88223

theorem gold_coins_proof :
  ∃ (n c : ℕ), n = 109 ∧ n = 12 * (c - 4) ∧ n = 8 * c + 5 :=
begin
  sorry
end

end gold_coins_proof_l88_88223


namespace dot_product_square_ABCD_l88_88602

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l88_88602


namespace jim_min_percentage_needed_l88_88563

noncomputable def geoff_votes : ℕ := (0.5 / 100) * 6000
noncomputable def laura_votes : ℕ := 2 * geoff_votes
noncomputable def total_votes : ℕ := 6000
noncomputable def jim_votes : ℕ := total_votes - (geoff_votes + laura_votes)

theorem jim_min_percentage_needed (h_geoff_needs_3000_more : geoff_votes + 3000 >= 3030):
  ((geoff_votes + 3000) < jim_votes + 1) →
  ((3031 : ℕ) / total_votes.to_real * 100 ≥ 50.52) :=
by
  sorry

end jim_min_percentage_needed_l88_88563


namespace angle_bisectors_triangle_inscribed_circle_center_l88_88031

theorem angle_bisectors_triangle_inscribed_circle_center (A B C : Type) [euclidean_space A] [triangle A B C] :
  ∃ (P : point A), (is_incenter_triangle P A B C) ∧ (angle_bisectors_intersect_at A B C P) :=
by
  sorry

end angle_bisectors_triangle_inscribed_circle_center_l88_88031


namespace dot_product_EC_ED_l88_88597

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l88_88597


namespace distinct_four_digit_positive_integers_product_18_l88_88975

theorem distinct_four_digit_positive_integers_product_18 :
  Finset.card {n | ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
                             (1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) ∧
                             a * b * c * d = 18} = 24 :=
by
  sorry

end distinct_four_digit_positive_integers_product_18_l88_88975


namespace range_of_a_l88_88149

theorem range_of_a (a : ℝ) :
  (∃ x₀ ∈ set.Icc (-1 : ℝ) 1, |4^x₀ - a * 2^x₀ + 1| ≤ 2^(x₀ + 1)) ↔ 0 ≤ a ∧ a ≤ 9 / 2 :=
by
  sorry

end range_of_a_l88_88149


namespace proof_problem1_proof_problem2_l88_88772

-- Proof Problem 1
def problem1 : Real :=
  sqrt 16 + (1 - sqrt 3)^0 - 2^(-1)

theorem proof_problem1 : problem1 = (4 + 1 - 1/2 : Real) :=
  by sorry

-- Proof Problem 2
theorem proof_problem2 (x : Real) :
  (-2 * x + 6 >= 4) ∧ ((4 * x + 1) / 3 > x - 1) -> x ∈ [0, 1] :=
  by sorry

end proof_problem1_proof_problem2_l88_88772


namespace find_a_b_l88_88920

theorem find_a_b (a b : ℝ) (h1 : sqrt (2 + 2 / 3) = 2 * sqrt (2 / 3))
  (h2 : sqrt (3 + 3 / 8) = 3 * sqrt (3 / 8))
  (h3 : sqrt (4 + 4 / 15) = 4 * sqrt (4 / 15))
  (h4 : sqrt (6 + a / b) = 6 * sqrt (a / b)) :
  a = 6 ∧ b = 35 :=
sorry

end find_a_b_l88_88920


namespace distribution_problem_l88_88818

open Nat

noncomputable def countValidDistributions : Nat :=
  ∑ k in finRange 8, (-1)^k * (binomial 12 k) * (12 - k)^7

theorem distribution_problem :
  ∃ N : Nat, N = countValidDistributions :=
by
  use countValidDistributions
  rfl

end distribution_problem_l88_88818


namespace speeds_correct_l88_88468

-- Definitions for conditions
def distance (A B : Type) := 40 -- given distance between A and B is 40 km
def start_time_pedestrian : Real := 4 -- pedestrian starts at 4:00 AM
def start_time_cyclist : Real := 7 + (20 / 60) -- cyclist starts at 7:20 AM
def midpoint_distance : Real := 20 -- the midpoint distance where cyclist catches up with pedestrian is 20 km

noncomputable def speeds (x y : Real) : Prop :=
  let t_catch_up := (20 - (10 / 3) * x) / (y - x) in -- time taken by the cyclist to catch up
  let t_total := (10 / 3) + t_catch_up + 1 in -- total time for pedestrian until meeting second cyclist
  4.5 = t_total ∧ -- total time in hours from 4:00 AM to 8:30 AM
  10 * x * (y - x) + 60 * x - 10 * x^2 = 60 * y - 60 * x ∧ -- initial condition simplification step
  y = 6 * x -- relationship between speeds based on derived equations

-- The proposition to prove
theorem speeds_correct : ∃ x y : Real, speeds x y ∧ x = 5 ∧ y = 30 :=
by
  sorry

end speeds_correct_l88_88468


namespace complex_problem_solution_l88_88193

noncomputable def complex_problem_condition (z : ℂ) : Prop :=
  10 * complex.norm_sq z = 3 * complex.norm_sq (z + 3) + complex.norm_sq (z^2 + 4) + 50

theorem complex_problem_solution (z : ℂ) (h : complex_problem_condition z) : 2 * z + 12 / z = -37 / 3 :=
sorry

end complex_problem_solution_l88_88193


namespace emails_left_in_inbox_l88_88680

-- Define the initial conditions and operations
def initial_emails : ℕ := 600

def move_half_to_trash (emails : ℕ) : ℕ := emails / 2
def move_40_percent_to_work (emails : ℕ) : ℕ := emails - (emails * 40 / 100)
def move_25_percent_to_personal (emails : ℕ) : ℕ := emails - (emails * 25 / 100)
def move_10_percent_to_miscellaneous (emails : ℕ) : ℕ := emails - (emails * 10 / 100)
def filter_30_percent_to_subfolders (emails : ℕ) : ℕ := emails - (emails * 30 / 100)
def archive_20_percent (emails : ℕ) : ℕ := emails - (emails * 20 / 100)

-- Statement we need to prove
theorem emails_left_in_inbox : 
  archive_20_percent
    (filter_30_percent_to_subfolders
      (move_10_percent_to_miscellaneous
        (move_25_percent_to_personal
          (move_40_percent_to_work
            (move_half_to_trash initial_emails))))) = 69 := 
by sorry

end emails_left_in_inbox_l88_88680


namespace fixed_point_graph_l88_88428

theorem fixed_point_graph (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) : ∃ x y : ℝ, (x = 2 ∧ y = 2 ∧ y = a^(x-2) + 1) :=
by
  use 2
  use 2
  sorry

end fixed_point_graph_l88_88428


namespace proj_MN_eq_KL_l88_88770

variables {ABC : Type*} {AB AC BC : Segment ABC}
variables {M N K L S : Point ABC}
variables (is_isosceles : AB = AC)
variables (M_on_AB : lies_on M AB)
variables (N_on_AC : lies_on N AC)
variables (MN_parallel_BC : parallel MN BC)
variables (S_midpoint_MN : midpoint S M N)
variables (KL_parallel_MN : parallel KL MN)
variables (K_on_AB : lies_on K AB)
variables (L_on_AC : lies_on L AC)

theorem proj_MN_eq_KL :
  proj_of_segment MN BC = KL :=
sorry

end proj_MN_eq_KL_l88_88770


namespace probability_first_three_heads_l88_88305

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l88_88305


namespace find_m_range_l88_88797

noncomputable def f : ℝ → ℝ :=
  λ x, if x % 2 < 1 then (1/2 - 2 * (x % 2) ^ 2)
       else -2 ^ (1 - abs ((x % 2) - 3/2))

noncomputable def g (m : ℝ) : ℝ → ℝ :=
  λ x, (2 * x - x^2) * Real.exp x + m

def main_range (m : ℝ) := m ≤ 3 / Real.exp 1 - 2

theorem find_m_range (m : ℝ) :
  (∀ x1 ∈ Set.Icc (-4 : ℝ) (-2),
   ∃ x2 ∈ Set.Icc (-1 : ℝ) 2, f x1 - g m x2 ≥ 0) ↔ main_range m := sorry

end find_m_range_l88_88797


namespace prove_sequence_terms_irrational_l88_88245

noncomputable def sequence_terms_irrational (a : ℕ → ℝ) : Prop :=
  (∀ k : ℕ, (a (k + 1) + k) * a k = 1) →
  (∀ k : ℕ, irrational (a k))

theorem prove_sequence_terms_irrational
  (a : ℕ → ℝ)
  (h1 : ∀ k : ℕ, 0 < a k)
  (h2 : ∀ k : ℕ, (a (k + 1) + k) * a k = 1)
  : sequence_terms_irrational a :=
by {
  sorry
}

end prove_sequence_terms_irrational_l88_88245


namespace probability_of_sum_of_two_Fermat_primes_l88_88350

def is_Fermat_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, p = 2^(2^n) + 1

def positive_even_numbers_not_exceeding_30 : list ℕ :=
  [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

def can_be_sum_of_two_different_Fermat_primes (n : ℕ) : Prop :=
  ∃ (a b : ℕ), is_Fermat_prime a ∧ is_Fermat_prime b ∧ a ≠ b ∧ n = a + b

theorem probability_of_sum_of_two_Fermat_primes : 
  (∃ (p : ℚ), p = 1/5) ↔ 
  (let relevant_numbers := list.filter can_be_sum_of_two_different_Fermat_primes positive_even_numbers_not_exceeding_30 in
  (relevant_numbers.length / positive_even_numbers_not_exceeding_30.length : ℚ) = 1/5) :=
by sorry

end probability_of_sum_of_two_Fermat_primes_l88_88350


namespace operation_days_correct_rounding_correct_l88_88232

-- Define the start date (November 24, 2012) and today's date (December 18, 2012).
def start_date := { year := 2012, month := 11, day := 24 }
def today := { year := 2012, month := 12, day := 18 }

-- Define the conditions given.
def train_trips_per_day := 100
def transport_capacity := 287000

-- Define the required days of operation.
def operating_days := 25

-- Define the rounding function and approximate number.
def round_to_nearest_ten_thousand (n : ℕ) : ℕ :=
  (n + 5000) / 10000 * 10000

def rounded_capacity := 290000

-- The first proof statement (number of operating days is 25).
theorem operation_days_correct : 
  let d1 := ⟨2012, 11, 24⟩ in
  let d2 := ⟨2012, 12, 18⟩ in
  nat.abs ((d2.day - d1.day) + ((d2.month - d1.month) * 30) + ((d2.year - d1.year) * 365)) = operating_days :=
begin
  sorry
end

-- The second proof statement (287,000 rounds to 2.9 * 10^5).
theorem rounding_correct : round_to_nearest_ten_thousand transport_capacity = rounded_capacity :=
begin
  sorry
end

end operation_days_correct_rounding_correct_l88_88232


namespace dot_product_EC_ED_l88_88590

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l88_88590


namespace find_EC_and_altitude_from_A_to_BC_l88_88166

open Real

-- Define the conditions
variables (BC BD DA : ℝ)
variable h_diameter : BC = sqrt 257
variable h_BD : BD = 1
variable h_DA : DA = 12

-- The theorem with required proofs
theorem find_EC_and_altitude_from_A_to_BC 
  (h_diameter : BC = sqrt 257) 
  (h_BD : BD = 1) 
  (h_DA : DA = 12) :
  let DC := sqrt (BC^2 - BD^2),
      AC := sqrt (DA^2 + DC^2),
      AE := 16 / 3,
      EC := 20 - AE,
      AY := sqrt (AC^2 - EC^2)
  in
    EC = 56 / 3 ∧ AY = 7.18 :=
by
  sorry

end find_EC_and_altitude_from_A_to_BC_l88_88166


namespace daniel_noodles_l88_88421

theorem daniel_noodles : 
  ∀ (initial_noodles given_away_noodles : ℕ), 
  initial_noodles = 66 → 
  given_away_noodles = 12 → 
  initial_noodles - given_away_noodles = 54 := 
by
  intros initial_noodles given_away_noodles h1 h2
  rw [h1, h2]
  norm_num

end daniel_noodles_l88_88421


namespace train_speed_approx_60_kmph_l88_88814

def train_length : ℝ := 110
def bridge_length : ℝ := 200
def crossing_time : ℝ := 18.598512119030477

theorem train_speed_approx_60_kmph :
  let total_distance := train_length + bridge_length,
      speed_m_s := total_distance / crossing_time,
      speed_kmph := speed_m_s * 3.6
  in abs (speed_kmph - 60) < 0.1 :=
by
  sorry

end train_speed_approx_60_kmph_l88_88814


namespace probability_three_heads_l88_88321

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l88_88321


namespace amount_each_friend_received_l88_88263

theorem amount_each_friend_received :
  ∀ (winning: ℝ) (percentage: ℝ) (friends: ℕ), winning = 100 ∧ percentage = 0.5 ∧ friends = 3 →
  (winning * percentage) / friends = 16.67 := 
by
  intros winning percentage friends h,
  rcases h with ⟨hw, hp, hf⟩,
  sorry

end amount_each_friend_received_l88_88263


namespace inscribed_circle_radius_l88_88218

-- Given: Sector OAB is a third of a circle with radius 6 cm.
-- Required to prove: The radius of the inscribed circle is 6 * sqrt(2) - 6 cm.

noncomputable def radius_of_inscribed_circle : ℝ :=
  let r : ℝ := 6 * (Real.sqrt 2 - 1) in
  r

theorem inscribed_circle_radius:
  is_third_of_a_circle (sector := 6) →
  ∃ r : ℝ, r = radius_of_inscribed_circle ∧
           tangent_at_three_points (circle_radius := radius_of_inscribed_circle) (sector_radius := 6) :=
by
  intros h_sector
  use radius_of_inscribed_circle
  split
  { unfold radius_of_inscribed_circle
    simp }
  { sorry }

end inscribed_circle_radius_l88_88218


namespace AliceFavoriteNumber_l88_88395

theorem AliceFavoriteNumber :
  ∃ n : ℕ, (90 < n ∧ n < 150) ∧ (n % 13 = 0) ∧ ¬(n % 3 = 0) ∧ ((n.digits.sum) % 4 = 0) ∧ n = 130 :=
by
  use 130
  -- Conditions to verify
  split
  { exact 130 < 150 }
  split
  { exact 90 < 130 }
  split
  { exact 130 % 13 = 0 }
  split
  { exact ¬ (130 % 3 = 0) }
  split
  { exact (130.digits.sum) % 4 = 0 }
  exact rfl

end AliceFavoriteNumber_l88_88395


namespace polynomial_roots_and_factorization_l88_88922

theorem polynomial_roots_and_factorization (m : ℤ) :
  (∀ x : ℤ, IsRoot (Polynomial.mk [0, 0, 2, 0, m, 8]) x) →
  m = -10 ∧ (∀ x : ℤ, Polynomial.mk [0, 0, 2, 0, m, 8] = 
    Polynomial.mul (Polynomial.mul (Polynomial.mul (Polynomial.C 2) (Polynomial.X + 1)) (Polynomial.X - 1)) (Polynomial.mul (Polynomial.X + 2) (Polynomial.X - 2))) :=
by
  intros h
  sorry

end polynomial_roots_and_factorization_l88_88922


namespace max_value_neg_domain_l88_88840

theorem max_value_neg_domain (x : ℝ) (h : x < 0) : 
  ∃ y, y = 2 * x + 2 / x ∧ y ≤ -4 :=
sorry

end max_value_neg_domain_l88_88840


namespace exists_good_10_element_subset_l88_88632

theorem exists_good_10_element_subset (f : Finset ℕ → ℕ)
  (h₁ : ∀ S, S.card = 9 → ∃ n, n ∈ S ∧ f S = n) :
  ∃ T : Finset ℕ, T.card = 10 ∧ ∀ k ∈ T, f (T.erase k) ≠ k :=
by
  let M := (Finset.range 21).erase 0
  have hM : M.card = 20 := by simp [Finset.card_erase_of_mem, Finset.mem_range]
  obtain ⟨T, hT₁, hT₂⟩ := exists_good_set M f h₁ hM
  exact ⟨T, hT₁, hT₂⟩


end exists_good_10_element_subset_l88_88632


namespace binom_20_10_l88_88904

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l88_88904


namespace markese_earned_16_l88_88647

def evan_earnings (E : ℕ) : Prop :=
  (E : ℕ)

def markese_earnings (M : ℕ) (E : ℕ) : Prop :=
  (M : ℕ) = E - 5

def total_earnings (E M : ℕ) : Prop :=
  E + M = 37

theorem markese_earned_16 (E : ℕ) (M : ℕ) 
  (h1 : markese_earnings M E) 
  (h2 : total_earnings E M) : M = 16 :=
sorry

end markese_earned_16_l88_88647


namespace area_of_triangle_ABC_l88_88371

variables {α : ℝ} -- Define variable for angle α

-- Define our geometric elements, points and segments
variables (A B C D E F : Type*)
          [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]

-- Define lengths DE and BE
variables (DE BE : ℝ)

-- Define the theorem that proves the area of triangle ABC
theorem area_of_triangle_ABC (h1 : DE = 2) (h2 : BE = 1) (h3 : BF = 1) (h4 : ∠FCB = α) :
  ∃ (area : ℝ), area = (1 / 2) * (2 * (real.cos (2 * α)) + 1) ^ 2 * (real.tan (2 * α)) :=
sorry

end area_of_triangle_ABC_l88_88371


namespace smallest_sum_of_consecutive_primes_divisible_by_four_l88_88064

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := nat.prime n

-- Define a predicate to check if four consecutive primes sum to a specific value
def sum_of_primes_is (a b c d sum : ℕ) : Prop := 
  is_prime a ∧ 
  is_prime b ∧ 
  is_prime c ∧ 
  is_prime d ∧ 
  a + 2 = b ∧ 
  b + 2 = c ∧ 
  c + 2 = d ∧ 
  a + b + c + d = sum

-- Define the main theorem statement
theorem smallest_sum_of_consecutive_primes_divisible_by_four :
  ∃ a b c d, sum_of_primes_is a b c d 36 ∧ (a + b + c + d) % 4 = 0 := 
sorry

end smallest_sum_of_consecutive_primes_divisible_by_four_l88_88064


namespace hyperbola_real_axis_length_l88_88705

theorem hyperbola_real_axis_length :
  (∃ a : ℝ, (∀ x y : ℝ, (x^2 / 9 - y^2 = 1) → (2 * a = 6))) :=
sorry

end hyperbola_real_axis_length_l88_88705


namespace distinct_four_digit_positive_integers_product_18_l88_88976

theorem distinct_four_digit_positive_integers_product_18 :
  Finset.card {n | ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
                             (1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) ∧
                             a * b * c * d = 18} = 24 :=
by
  sorry

end distinct_four_digit_positive_integers_product_18_l88_88976


namespace jack_grove_trees_column_l88_88174

theorem jack_grove_trees_column :
  ∀ (x : ℕ), 
    (∀ (total_trees minutes_per_tree time_spent : ℕ), 
       total_trees = 4 * x → 
       minutes_per_tree = 3 → 
       time_spent = 60 → 
       time_spent / minutes_per_tree = total_trees) → 
    x = 5 := 
by
  intros x H
  have H1 : 4 * x = (60 / 3) := sorry
  have H2 : 4 * x = 20 := sorry
  have H3 : x = 5 := by
    apply H2
  assumption

end jack_grove_trees_column_l88_88174


namespace dot_product_square_ABCD_l88_88601

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l88_88601


namespace train_crosses_post_in_approximately_18_seconds_l88_88348

noncomputable def train_length : ℕ := 300
noncomputable def platform_length : ℕ := 350
noncomputable def crossing_time_platform : ℕ := 39

noncomputable def combined_length : ℕ := train_length + platform_length
noncomputable def speed_train : ℝ := combined_length / crossing_time_platform

noncomputable def crossing_time_post : ℝ := train_length / speed_train

theorem train_crosses_post_in_approximately_18_seconds :
  abs (crossing_time_post - 18) < 1 :=
by
  admit

end train_crosses_post_in_approximately_18_seconds_l88_88348


namespace obtuse_triangle_area_bounds_square_sum_l88_88459

theorem obtuse_triangle_area_bounds_square_sum :
  let t := λ s : ℝ, {Δ | s > 0 ∧ Δ.is_obtuse ∧ Δ.area = s ∧ (Δ.side1 = 4 ∨ Δ.side2 = 4) ∧ (Δ.side1 = 10 ∨ Δ.side2 = 10)}
  ∃ a b : ℝ, (∀ Δ Δ' ∈ t (a^2 + b^2), Δ ≅ Δ') ∧ (a^2 + b^2 = 736) :=
by
  sorry

end obtuse_triangle_area_bounds_square_sum_l88_88459


namespace anne_cleaning_time_l88_88756

theorem anne_cleaning_time :
  ∃ (A B : ℝ), (4 * (B + A) = 1) ∧ (3 * (B + 2 * A) = 1) ∧ (1 / A = 12) :=
by
  sorry

end anne_cleaning_time_l88_88756


namespace shortest_distance_is_correct_l88_88715

open Real

-- Define the parametric equations for a point on the graph
def parametric_eqns (θ : ℝ) := (3 + 3 * cos θ, -3 + 3 * sin θ)

-- Define the line equation y = x
def line_eq (x y : ℝ) := y = x

-- Define the shortest distance function
def shortest_distance_from_graph_to_line : ℝ :=
  let center := (3 : ℝ, -3 : ℝ)
  let radius := 3
  let distance := abs (-1 * center.1 + 1 * center.2) / sqrt (1^2 + (-1)^2)
  distance - radius

theorem shortest_distance_is_correct :
  shortest_distance_from_graph_to_line = 3 * (sqrt 2 - 1) :=
by sorry

end shortest_distance_is_correct_l88_88715


namespace vector_magnitude_parallel_l88_88950

theorem vector_magnitude_parallel (m : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (6, -2)) 
  (hb : b = (3, m))
  (h_parallel : 6 / 3 = -2 / m) : 
  | (6, -2) - (3, m) | = 3 * Real.sqrt 2 := by
  sorry

end vector_magnitude_parallel_l88_88950


namespace number_of_ways_sum_divisible_by_4_l88_88858

theorem number_of_ways_sum_divisible_by_4 (n : ℕ) :
    let S := finset.range (4 * n + 1)
    ∃ (k : ℕ), (k = n^3 + 3 * n * nat.choose n 2 + nat.choose n 3) ∧
    (finset.card {a ∈ S | ∃ b ∈ S, ∃ c ∈ S, (a + b + c) % 4 = 0}) = k :=
by
  sorry

end number_of_ways_sum_divisible_by_4_l88_88858


namespace sin_neg_1740_eq_neg_sqrt3_div_2_l88_88449

theorem sin_neg_1740_eq_neg_sqrt3_div_2 : 
  sin (-1740 * real.pi / 180) = - (real.sqrt 3 / 2) := by
  sorry

end sin_neg_1740_eq_neg_sqrt3_div_2_l88_88449


namespace decimal_to_binary_89_l88_88042

theorem decimal_to_binary_89 : nat.toBinary 89 = "1011001" :=
by 
  sorry

end decimal_to_binary_89_l88_88042


namespace total_earnings_per_week_correct_l88_88617

noncomputable def weekday_fee_kid : ℝ := 3
noncomputable def weekday_fee_adult : ℝ := 6
noncomputable def weekend_surcharge_ratio : ℝ := 0.5

noncomputable def num_kids_weekday : ℕ := 8
noncomputable def num_adults_weekday : ℕ := 10

noncomputable def num_kids_weekend : ℕ := 12
noncomputable def num_adults_weekend : ℕ := 15

noncomputable def weekday_earnings_kids : ℝ := (num_kids_weekday : ℝ) * weekday_fee_kid
noncomputable def weekday_earnings_adults : ℝ := (num_adults_weekday : ℝ) * weekday_fee_adult

noncomputable def weekday_earnings_total : ℝ := weekday_earnings_kids + weekday_earnings_adults

noncomputable def weekday_earning_per_week : ℝ := weekday_earnings_total * 5

noncomputable def weekend_fee_kid : ℝ := weekday_fee_kid * (1 + weekend_surcharge_ratio)
noncomputable def weekend_fee_adult : ℝ := weekday_fee_adult * (1 + weekend_surcharge_ratio)

noncomputable def weekend_earnings_kids : ℝ := (num_kids_weekend : ℝ) * weekend_fee_kid
noncomputable def weekend_earnings_adults : ℝ := (num_adults_weekend : ℝ) * weekend_fee_adult

noncomputable def weekend_earnings_total : ℝ := weekend_earnings_kids + weekend_earnings_adults

noncomputable def weekend_earning_per_week : ℝ := weekend_earnings_total * 2

noncomputable def total_weekly_earnings : ℝ := weekday_earning_per_week + weekend_earning_per_week

theorem total_earnings_per_week_correct : total_weekly_earnings = 798 := by
  sorry

end total_earnings_per_week_correct_l88_88617


namespace circle_equation_l88_88248

theorem circle_equation (x y : ℝ) : (x-2)^2 + (y+1)^2 = 25 ↔ (∃ (R:ℝ), R = 5 ∧ (2, -1) = (2, -1) ∧ (-1, 3) ∈ set_of (λ p : ℝ × ℝ, (p.1 - 2)^2 + (p.2 + 1)^2 = R^2)) :=
by
  sorry

end circle_equation_l88_88248


namespace arithmetic_sequence_a6_l88_88163

-- Definitions representing the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n (n + m) = a_n n + a_n m + n

def sum_of_first_n_terms (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = (n / 2) * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))

theorem arithmetic_sequence_a6 (S : ℕ → ℝ) (a_n : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_first_n_terms S a_n)
  (h_cond : S 9 - S 2 = 35) : 
  a_n 6 = 5 :=
by
  sorry

end arithmetic_sequence_a6_l88_88163


namespace profit_percentage_correct_l88_88387

variable (C : ℕ)  -- cost price of each article

def profit_percentage : ℕ :=
  let SellingPrice := 35 * C
  let CostPrice     := 30 * C
  let Profit        := SellingPrice - CostPrice
  let ProfitPercent := (Profit * 100) / CostPrice
  ProfitPercent

theorem profit_percentage_correct (C : ℕ) (hC : C > 0) : profit_percentage C = 16.67 :=
  by
    sorry

end profit_percentage_correct_l88_88387


namespace domain_of_f_l88_88427

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 1)) / (x + 1)

theorem domain_of_f :
  ∀ x, (x - 1 ≥ 0) ∧ (x + 1 ≠ 0) ↔ (x ≥ 1) := 
by
  sorry

end domain_of_f_l88_88427


namespace prob_contestant_A_makes_it_to_final_expected_value_of_xi_l88_88812

/-- Problem conditions -/

def prob_correct : ℚ := 2 / 3
def prob_incorrect : ℚ := 1 / 3

def make_it_to_final_3_correct : ℚ :=
(2 / 3) ^ 3

def make_it_to_final_4_questions : ℚ :=
(binomial 3 2) * (2 / 3)^2 * (1 / 3) * (2 / 3)

def make_it_to_final_5_questions : ℚ :=
(binomial 4 2) * (2 / 3) ^ 3 * (1 / 3) ^ 2

def prob_make_it_to_final : ℚ :=
make_it_to_final_3_correct + make_it_to_final_4_questions + make_it_to_final_5_questions

/-- Proof statement (1) -/
theorem prob_contestant_A_makes_it_to_final
  (p_correct : prob_correct = 2 / 3)
  (p_incorrect : prob_incorrect = 1 / 3)
  : prob_make_it_to_final = 64 / 81 := by
  sorry

def xi_dist_3 : ℚ := ((2 / 3) ^ 3) + (1 / 3) ^ 3
def xi_dist_4 : ℚ := (binomial 3 2) * (2 / 3)^2 * (1 / 3) * (2 / 3) + (binomial 3 2) * (1 / 3)^2 * (2 / 3) * (1 / 3)
def xi_dist_5 : ℚ := (binomial 4 2) * (2 / 3) ^ 3 * (1 / 3) ^ 2 + (binomial 4 2) * (2 / 3) ^ 2 * (1 / 3) ^ 3

def expected_value_xi : ℚ :=
3 * xi_dist_3 + 4 * xi_dist_4 + 5 * xi_dist_5

/-- Proof statement (2) -/
theorem expected_value_of_xi
  (dist_3 : xi_dist_3 = 1 / 3)
  (dist_4 : xi_dist_4 = 10 / 27)
  (dist_5 : xi_dist_5 = 8 / 27)
  : expected_value_xi = 107 / 27 := by
  sorry

end prob_contestant_A_makes_it_to_final_expected_value_of_xi_l88_88812


namespace max_power_speed_l88_88700

def aerodynamic_force (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v)^2) / 2

def power (C S ρ v₀ v : ℝ) : ℝ :=
  aerodynamic_force C S ρ v₀ v * v

theorem max_power_speed (C S ρ v₀ v : ℝ) (h₁ : v = v₀ / 3) :
  ∃ v, power C S ρ v₀ v = (C * S * ρ * v₀^3) / 54 :=
begin
  use v₀ / 3,
  sorry
end

end max_power_speed_l88_88700


namespace trapezoid_diagonal_ratio_l88_88355

theorem trapezoid_diagonal_ratio {A B C D M N : Point}
    (h1 : ∠A = 90°) (h2 : ∠B = 90°) 
    (h3 : circle_through ABCD A B D)
    (h4 : extension_intersection BC CD M N)
    (h5 : ratio_CM_CB : CM / CB = 1 / 2)
    (h6 : ratio_CN_CD : CN / CD = 1 / 2) :
    BD / AC = 2 * sqrt(3) / sqrt(7) :=
begin
  sorry
end

end trapezoid_diagonal_ratio_l88_88355


namespace regression_model_fitting_effect_l88_88933

/-- Conditions for regression model fitting effect -/
def Condition1 (R2 : ℝ) : Prop := R2 ≥ 0 → ∀ (R2' : ℝ), R2' ≥ R2 → R2' indicates better fit
def Condition2 (SSR : ℝ) : Prop := SSR ≥ 0 → ∀ (SSR' : ℝ), SSR' ≤ SSR → SSR' indicates better fit
def Condition3 (r : ℝ) : Prop := ∀ r, (r ≥ 0 → |r| indicates better fit)
def Condition4 : Prop := ∀ residual_plot, (evenly_distributed_band residual_plot → more appropriate_model residual_plot) ∧ 
        (narrower_band residual_plot → higher_precision residual_plot)

/-- Main predicate to check the correct conditions -/
def correct_conditions (R2 SSR r : ℝ) : Prop :=
  Condition1 R2 ∧ ¬ Condition2 SSR ∧ ¬ Condition3 r ∧ Condition4

theorem regression_model_fitting_effect (R2 SSR r : ℝ) 
  (h1 : Condition1 R2) 
  (h2 : Condition2 SSR) 
  (h3 : Condition3 r) 
  (h4 : Condition4) :
  correct_conditions R2 SSR r :=
by {
  sorry -- Proof is not required
}

end regression_model_fitting_effect_l88_88933


namespace smallest_n_for_factorial_sum_2001_l88_88861

theorem smallest_n_for_factorial_sum_2001 (n : ℕ) (a : Fin n → ℕ) (h₀ : ∀ i, a i ≤ 15) 
  (h₁ : (Finset.univ.sum (λ i : (Fin n), (nat.factorial (a i)))) % 10000 = 2001) : 
  n = 3 :=
sorry

end smallest_n_for_factorial_sum_2001_l88_88861


namespace find_range_a_l88_88116

-- Given conditions
def hyperbola_eq (a x y : ℝ) : Prop := (1 - a^2) * x^2 + a^2 * y^2 = a^2
def line_eq (x y : ℝ) : Prop := y = -x
def parabola_eq (x y m : ℝ) : Prop := x^2 = -4*(m - 1)*(y - m)
def slope_eq (m a : ℝ) : Prop := 1/4 ≤ (m - a)/a ∧ (m - a)/a ≤ 1/3

-- Prove the range of a
theorem find_range_a (a : ℝ) (h1 : a > 1)
    (h2 : ∃ m, hyperbola_eq a 0 1 ∧ parabola_eq 0 1 m ∧ m > 1)
    (h3 : ∃ P : ℝ × ℝ, P = (-a, a) ∧ hyperbola_eq a (-a) a ∧ line_eq (-a) a)
    (h4 : ∃ k, slope_eq (a * k + a) a ∧ slope_eq (a * k + a) a ∧ parabola_eq (-a) a (a * k + a)) : 
    (frac 12 7 ≤ a ∧ a ≤ 4) :=
begin
  sorry,
end

end find_range_a_l88_88116


namespace find_speeds_l88_88477

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l88_88477


namespace ananthu_work_days_l88_88398

theorem ananthu_work_days (Amit_days : ℕ) (work_days : ℕ) (Ananthu_work : (ℕ → ℚ) → Prop) : 
  Amit_days = 10 ∧ 
  work_days = 18 ∧ 
  (∀ rate : ℕ → ℚ, rate Amit_days 2 + rate (work_days - 2) (rate work_days - rate Amit_days 2)) =
  20 := 
by
  sorry

end ananthu_work_days_l88_88398


namespace least_number_to_add_l88_88283

theorem least_number_to_add (n : ℕ) : 
  (∀ k : ℕ, n = 1 + k * 425 ↔ n + 1019 % 425 = 0) → n = 256 := 
sorry

end least_number_to_add_l88_88283


namespace remainder_divisor_l88_88547

theorem remainder_divisor (d r : ℤ) (h1 : d > 1) 
  (h2 : 2024 % d = r) (h3 : 3250 % d = r) (h4 : 4330 % d = r) : d - r = 2 := 
by
  sorry

end remainder_divisor_l88_88547


namespace rectangle_division_l88_88956

theorem rectangle_division : ∃ rectangles : list (ℕ × ℕ),
  (∀ rect ∈ rectangles, rect.fst * rect.snd > 0) ∧
  list.foldl (λ acc rect, acc + rect.fst * rect.snd) 0 rectangles = 13 * 7 ∧
  list.nodup rectangles ∧
  list.length rectangles = 13 :=
sorry

end rectangle_division_l88_88956


namespace eccentricity_of_hyperbola_l88_88692

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let c := 2 * b
  let e := c / a
  e

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_cond : hyperbola_eccentricity a b h_a h_b = 2 * (b / a)) :
  hyperbola_eccentricity a b h_a h_b = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l88_88692


namespace pie_left_l88_88819

/--
Alice took 80% of a whole pie. Bob took one fourth of the remainder. Cindy then took half of what remained after Alice and Bob. Prove that the portion of the whole pie that was left is 7.5%.
-/
theorem pie_left (alice_share : ℝ) (bob_share : ℝ) (cindy_share : ℝ) (remaining_pie : ℝ) :
  alice_share = 0.80 →
  bob_share = 1 / 4 →
  cindy_share = 1 / 2 →
  remaining_pie = (1 - alice_share) * (1 - bob_share * (1 - alice_share)) * (1 - cindy_share * (1 - bob_share * (1 - alice_share))) →
  remaining_pie = 0.075 :=
begin
  sorry
end

end pie_left_l88_88819


namespace min_value_expression_l88_88869

theorem min_value_expression (a b : ℝ) (h : a ≠ 0) : 
  ∃ (m : ℝ), m = Real.sqrt(8 / 3) ∧ (∀ (x y : ℝ), x ≠ 0 → (1 / x^2 + 2 * x^2 + 3 * y^2 + 4 * x * y) ≥ Real.sqrt(8 / 3)) :=
by 
  use Real.sqrt(8 / 3)
  split
  -- Here we would provide the proof that the minimum value is sqrt(8 / 3)
  sorry

  -- Here we would provide the proof that for all x and y, the expression is greater or equal to sqrt(8 / 3)
  sorry

end min_value_expression_l88_88869


namespace markese_earnings_16_l88_88653

theorem markese_earnings_16 (E M : ℕ) (h1 : M = E - 5) (h2 : E + M = 37) : M = 16 :=
by
  sorry

end markese_earnings_16_l88_88653


namespace expression_is_five_l88_88452

-- Define the expression
def given_expression : ℤ := abs (abs (-abs (-2 + 1) - 2) + 2)

-- Prove that the expression equals 5
theorem expression_is_five : given_expression = 5 :=
by
  -- We skip the proof for now
  sorry

end expression_is_five_l88_88452


namespace graph_symmetric_l88_88072

theorem graph_symmetric {f : ℝ → ℝ} :
  (∀ x : ℝ, f(x) + f(2 - x) + 2 = 0) →
  (∀ x : ℝ, f(2 - x) = -2 - f(x)) →
  (∀ (x y : ℝ), y = f(x) → ∃ (x' y' : ℝ), x' = 2 - x ∧ y' = -2 - y ∧ y' = f(x')) :=
begin
  intros H1 H2 x y hy,
  use [2 - x, -2 - y],
  split,
  { exact rfl },
  split,
  { exact rfl },
  { rw H2, exact hy }
end

end graph_symmetric_l88_88072


namespace lunch_break_duration_l88_88211

-- Assume the conditions as given

variables (p h L : ℝ)
-- Wednesday condition
def wednesday_condition : Prop := (9 - L) * (p + h) = 0.6
-- Thursday condition
def thursday_condition : Prop := (7.4 - L) * h = 0.28
-- Friday condition
def friday_condition : Prop := (2.6 - L) * p = 0.12

-- The goal
theorem lunch_break_duration : wednesday_condition p h L → thursday_condition p h L → friday_condition p h L → L = 1 :=
by intros; sorry

end lunch_break_duration_l88_88211


namespace complex_division_l88_88487

-- Define real part 'a' condition
def realPart (a : ℝ) : Prop := a < 0

-- Define complex number z and its conjugate
def z (a : ℝ) (i : ℂ) : ℂ := a + 1 * Complex.I
def conj_z (a : ℝ) : ℂ := a - 1 * Complex.I

-- Magnitude condition
def magnitude_z (a : ℝ) : Prop := Complex.abs (z a Complex.I) = 2

-- Conjugate condition
def conjugate_z (a : ℝ) : Prop := Complex.conj (z a Complex.I) = conj_z a

-- Main theorem to prove
theorem complex_division (a : ℝ) (h1 : realPart a) (h2 : magnitude_z a) (h3 : conjugate_z a): 
  (1 + Complex.I * Real.sqrt 3) / (conj_z a) = -(Complex.I + Real.sqrt 3) / 2 :=
by
  sorry

end complex_division_l88_88487


namespace integer_root_divisor_of_constant_term_l88_88220

def polynomial (n : ℕ) (a : Fin n → ℤ) (x : ℤ) : ℤ :=
  (Fin n).foldr (fun i acc => a i * x ^ (n - i) + acc) 0

theorem integer_root_divisor_of_constant_term
  {n : ℕ} {a : Fin n → ℤ} {x : ℤ}
  (h_root : polynomial n a x = 0) :
  x ∣ a (Fin.last n) := sorry

end integer_root_divisor_of_constant_term_l88_88220


namespace power_problem_l88_88548

theorem power_problem (k : ℕ) (h : 6 ^ k = 4) : 6 ^ (2 * k + 3) = 3456 := 
by 
  sorry

end power_problem_l88_88548


namespace proof_problem_theorem_l88_88581

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l88_88581


namespace probability_s_neg_l88_88932

noncomputable def p_values_with_s_neg := { p ∈ Finset.range 11 | (p : ℕ) > 0 ∧ p^2 - 13 * p + 40 < 0 }

theorem probability_s_neg (h : p ∈ Finset.range 11): 
  (p_values_with_s_neg.card / 10 : ℝ) = 0.2 :=
by
  sorry

end probability_s_neg_l88_88932


namespace least_positive_t_geometric_progression_l88_88834

theorem least_positive_t_geometric_progression :
  ∃ (t : ℝ) (α : ℝ), 0 < α ∧ α < π / 2 ∧
  arccos (cos α) = α ∧ arccos (cos (3 * α)) = 3 * α ∧
  arccos (cos (6 * α)) = 6 * α ∧ arccos (cos (t * α)) = t * α ∧
  t = 27 := sorry

end least_positive_t_geometric_progression_l88_88834


namespace range_of_g_on_interval_l88_88509

-- Defining the function g(x) = x^m
def g (x : ℝ) (m : ℝ) : ℝ := x ^ m

-- Defining the conditions
variable {m : ℝ}
variable {x : ℝ}
variable h1 : 0 < m
variable h2 : 0 < x
variable h3 : x ≤ 1

-- Theorem stating the range of g(x) on (0, 1] is (0, 1]
theorem range_of_g_on_interval (h1 : 0 < m) (h2 : 0 < x) (h3 : x ≤ 1) :
  ∃ y : Set.Icc (0 : ℝ) 1, g x m = y :=
sorry

end range_of_g_on_interval_l88_88509


namespace sum_of_real_values_satisfying_equation_l88_88447

theorem sum_of_real_values_satisfying_equation :
  (∑ x in ({x : ℝ | |x - 1| = 3 * |x + 3|}), x) = -7 :=
by
  sorry

end sum_of_real_values_satisfying_equation_l88_88447


namespace melanie_plums_l88_88658

variable (initialPlums : ℕ) (givenPlums : ℕ)

theorem melanie_plums :
  initialPlums = 7 → givenPlums = 3 → initialPlums - givenPlums = 4 :=
by
  intro h1 h2
  -- proof omitted
  exact sorry

end melanie_plums_l88_88658


namespace triangle_cos_area_l88_88099

-- Given that \(a\), \(b\), and \(c\) are the sides opposite to angles A, B, and C respectively in \(\triangle ABC\).
variables {A B C : ℝ} -- angles
variables {a b c : ℝ} -- sides

-- Given conditions
def cond1 : Prop := sin B ^ 2 = 2 * sin A * sin C
def cond2 : Prop := c = sqrt 3
def cond3 : Prop := a = sqrt 3

-- Theorem to prove cos B = 0 and area is \frac{3}{2}
theorem triangle_cos_area (h1 : cond1) (h2 : cond2) (h3 : cond3) : 
  cos B = 0 ∧ (1/2 * a * c * sin A = 3/2) :=
sorry

end triangle_cos_area_l88_88099


namespace product_greater_than_zero_l88_88269

noncomputable theory

def probability_product_greater_zero : Prop :=
  let interval := set.Icc (-15 : ℝ) 15 in
  let prob_positive_or_negative (x : set ℝ) : ℝ := (set.size x) / (set.size interval) in
  let prob_positive := prob_positive_or_negative (set.Ioc 0 15) in
  let prob_negative := prob_positive_or_negative (set.Ico (-15) 0) in
  let prob_positive_product := prob_positive * prob_positive in
  let prob_negative_product := prob_negative * prob_negative in
  prob_positive_product + prob_negative_product = 0.5

theorem product_greater_than_zero :
  probability_product_greater_zero :=
sorry

end product_greater_than_zero_l88_88269


namespace real_roots_lambda_le_three_fourths_no_real_roots_lambda_ge_one_l88_88040

-- Part (a)
theorem real_roots_lambda_le_three_fourths (a b c λ : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_le : λ ≤ 3 / 4) :
  ∃ x : ℝ, x^2 + (a + b + c) * x + λ * (a * b + b * c + c * a) = 0 := sorry

-- Part (b)
theorem no_real_roots_lambda_ge_one (a b c λ : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_ge : λ ≥ 1) :
  ¬ ∃ x : ℝ, x^2 + (a + b + c) * x + λ * (a * b + b * c + c * a) = 0 := sorry

end real_roots_lambda_le_three_fourths_no_real_roots_lambda_ge_one_l88_88040


namespace probability_is_correct_l88_88069

noncomputable def probability_odd_and_greater_than_12 : ℚ :=
  let outcomes := [(x, y) | x ← [1, 2, 3, 4, 5], y ← [1, 2, 3, 4, 5]]
  let favorable := [(3, 5), (5, 3), (5, 5)]
  favorable.length / outcomes.length

theorem probability_is_correct :
  probability_odd_and_greater_than_12 = 3 / 25 :=
by
  sorry

end probability_is_correct_l88_88069


namespace no_same_graphs_l88_88430

-- Define the equations
def equation_I := ∀ x : ℝ, y = 2 * x - 1
def equation_II := ∀ x : ℝ, x ≠ -1 / 2 → y = (4 * x^2 - 1) / (2 * x + 1)
def equation_III := ∀ x : ℝ, (2 * x + 1) * y = 4 * x^2 - 1

-- State the theorem to prove none of the equations have the same graph
theorem no_same_graphs : (∀ x : ℝ, y = 2 * x - 1) ≠ 
                         (∀ x : ℝ, x ≠ -1 / 2 → y = (4 * x^2 - 1) / (2 * x + 1)) 
                         ∧ 
                         (∀ x : ℝ, y = 2 * x - 1) ≠ 
                         (∀ x : ℝ, (2 * x + 1) * y = 4 * x^2 - 1)
                         ∧ 
                         (∀ x : ℝ, x ≠ -1 / 2 → y = (4 * x^2 - 1) / (2 * x + 1)) ≠ 
                         (∀ x : ℝ, (2 * x + 1) * y = 4 * x^2 - 1) :=
sorry

end no_same_graphs_l88_88430


namespace distinct_four_digit_positive_integers_product_18_l88_88978

theorem distinct_four_digit_positive_integers_product_18 :
  Finset.card {n | ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
                             (1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) ∧
                             a * b * c * d = 18} = 24 :=
by
  sorry

end distinct_four_digit_positive_integers_product_18_l88_88978


namespace max_planes_six_points_l88_88050

theorem max_planes_six_points (P : Finset (Fin 6 → ℝ)) :
  (∀ p ∈ P.powerset, p.card ≥ 4 → ∃ plane : Finset (Fin 6 → ℝ), plane.card ≥ 4 ∧ p ⊆ plane)
  ∧ (∀ p ∈ (P.powerset.filter (λ p, p.card = 4)), ¬Collinear ℝ p) → (∃! n, n = 6) :=
sorry

variables (ℝ : Type) [field ℝ] [vector_space ℝ (Fin 6 → ℝ)] 
variables [add_comm_group (Fin 6 → ℝ)] [module ℝ (Fin 6 → ℝ)]

namespace collinear_detection

-- Assume collinearity detection module

def Collinear (K : Type*) [field K] (s : Finset (Fin 6 → K)) : Prop :=
∃ l : (Fin 6 → K), ∀ p ∈ s, ∃ k : K, p = k • l

end collinear_detection

end max_planes_six_points_l88_88050


namespace log_sum_of_geometric_sequence_l88_88942

variable {x : ℕ → ℝ}

def geometric_sequence (x : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, x (n + 1) = x n * q

theorem log_sum_of_geometric_sequence
  (h : geometric_sequence x)
  (h2 : x 2 * x 5 * x 8 = Real.exp 1) :
  (∑ n in Finset.range 9, Real.log (x n)) = 3 :=
sorry

end log_sum_of_geometric_sequence_l88_88942


namespace count_four_digit_integers_with_product_18_l88_88966

def valid_digits (n : ℕ) : Prop := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def digit_product_18 (a b c d : ℕ) : Prop := 
  a * b * c * d = 18

def four_digit_integer (a b c d : ℕ) : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d

theorem count_four_digit_integers_with_product_18 : 
  (∑ a b c d in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    ite (four_digit_integer a b c d ∧ digit_product_18 a b c d) 1 0) = 48 := 
sorry

end count_four_digit_integers_with_product_18_l88_88966


namespace geometric_sequence_a1_cannot_be_2_l88_88489

theorem geometric_sequence_a1_cannot_be_2
  (a : ℕ → ℕ)
  (q : ℕ)
  (h1 : 2 * a 2 + a 3 = a 4)
  (h2 : (a 2 + 1) * (a 3 + 1) = a 5 - 1)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 1 ≠ 2 :=
by sorry

end geometric_sequence_a1_cannot_be_2_l88_88489


namespace weight_of_new_person_l88_88234

theorem weight_of_new_person : 
  ∀ (avg_increase : ℝ) (replaced_weight : ℝ), 
  avg_increase = 5.5 → 
  replaced_weight = 86 → 
  let total_increase := 9 * avg_increase in 
  let new_weight := replaced_weight + total_increase in 
  new_weight = 135.5 :=
by
  intros avg_increase replaced_weight h_avg h_replaced
  let total_increase := 9 * avg_increase
  have h_total_increase : total_increase = 49.5 := by sorry
  let new_weight := replaced_weight + total_increase
  have h_new_weight : new_weight = 135.5 := by sorry
  exact h_new_weight

end weight_of_new_person_l88_88234


namespace central_angle_l88_88926

-- Define the conditions as hypotheses
variables {r θ x : ℝ}

-- Conditions
def condition1 : Prop := θ/(2 * π) = x
def condition2 : Prop := π * x * r^2 = 1 / 2
def condition3 : Prop := 2 * r + 2 * π * x * r = 3

-- Proposition to prove
theorem central_angle (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  θ = 1 ∨ θ = 4 :=
sorry

end central_angle_l88_88926


namespace tan_center_of_symmetry_l88_88822

-- Define the main function y = tan(x + π/5)
def f (x : ℝ) : ℝ := Real.tan (x + Real.pi / 5)

-- Define the condition for x where x should not be kπ + 3π/10, ∀ k ∈ ℤ
def valid_x (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 10

-- The final theorem statement proving the center of symmetry
theorem tan_center_of_symmetry :
  (valid_x (3 * Real.pi / 10)) →
  ∃ c : ℝ × ℝ, c = (3 * Real.pi / 10, 0) ∧ 
  ∀ x : ℝ, valid_x x → f (2 * 3 * Real.pi / 10 - x) = - f x :=
sorry

end tan_center_of_symmetry_l88_88822


namespace student_chose_number_l88_88337

theorem student_chose_number : ∃ x : ℤ, 5 * x - 138 = 102 ∧ x = 48 := by
  exists 48
  constructor
  {
    have eq1 : 5 * 48 - 138 = 102 := by
    calc
      5 * 48 - 138 = 240 - 138 := by rw [Int.mul_eq_mul, Int.sub_eq_sub]
      _ = 102 := by norm_num
    exact eq1
  }
  exact rfl

end student_chose_number_l88_88337


namespace net_profit_expression_and_break_even_point_l88_88573

-- Definitions based on the conditions in a)
def investment : ℝ := 600000
def initial_expense : ℝ := 80000
def expense_increase : ℝ := 20000
def annual_income : ℝ := 260000

-- Define the net profit function as given in the solution
def net_profit (n : ℕ) : ℝ :=
  - (n : ℝ)^2 + 19 * n - 60

-- Statement about the function and where the dealer starts making profit
theorem net_profit_expression_and_break_even_point :
  net_profit n = - (n : ℝ)^2 + 19 * n - 60 ∧ ∃ n ≥ 5, net_profit n > 0 :=
sorry

end net_profit_expression_and_break_even_point_l88_88573


namespace quadratic_expression_representation_quadratic_expression_integer_iff_l88_88341

theorem quadratic_expression_representation (A B C : ℤ) :
  ∃ (k l m : ℤ), 
    (k = 2 * A) ∧ 
    (l = A + B) ∧ 
    (m = C) ∧ 
    (∀ x : ℤ, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) := 
sorry

theorem quadratic_expression_integer_iff (A B C : ℤ) :
  (∀ x : ℤ, ∃ k l m : ℤ, (k = 2 * A) ∧ (l = A + B) ∧ (m = C) ∧ (A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m)) ↔ 
  (A % 1 = 0 ∧ B % 1 = 0 ∧ C % 1 = 0) := 
sorry

end quadratic_expression_representation_quadratic_expression_integer_iff_l88_88341


namespace speeds_correct_l88_88467

-- Definitions for conditions
def distance (A B : Type) := 40 -- given distance between A and B is 40 km
def start_time_pedestrian : Real := 4 -- pedestrian starts at 4:00 AM
def start_time_cyclist : Real := 7 + (20 / 60) -- cyclist starts at 7:20 AM
def midpoint_distance : Real := 20 -- the midpoint distance where cyclist catches up with pedestrian is 20 km

noncomputable def speeds (x y : Real) : Prop :=
  let t_catch_up := (20 - (10 / 3) * x) / (y - x) in -- time taken by the cyclist to catch up
  let t_total := (10 / 3) + t_catch_up + 1 in -- total time for pedestrian until meeting second cyclist
  4.5 = t_total ∧ -- total time in hours from 4:00 AM to 8:30 AM
  10 * x * (y - x) + 60 * x - 10 * x^2 = 60 * y - 60 * x ∧ -- initial condition simplification step
  y = 6 * x -- relationship between speeds based on derived equations

-- The proposition to prove
theorem speeds_correct : ∃ x y : Real, speeds x y ∧ x = 5 ∧ y = 30 :=
by
  sorry

end speeds_correct_l88_88467


namespace minimum_value_of_f_l88_88990

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x + 1)

theorem minimum_value_of_f (x : ℝ) (h : x > -1) : f x = 1 ↔ x = 0 :=
by
  sorry

end minimum_value_of_f_l88_88990


namespace minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l88_88925

theorem minute_hand_angle_is_pi_six (radius : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : fast_min = 5) :
  (fast_min / 60 * 2 * Real.pi = Real.pi / 6) :=
by sorry

theorem minute_hand_arc_length_is_2pi_third (radius : ℝ) (angle : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : angle = Real.pi / 6) (h3 : fast_min = 5) :
  (radius * angle = 2 * Real.pi / 3) :=
by sorry

end minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l88_88925


namespace find_interest_rate_l88_88018

theorem find_interest_rate (initial_investment : ℚ) (duration_months : ℚ) 
  (first_rate : ℚ) (final_value : ℚ) (s : ℚ) :
  initial_investment = 15000 →
  duration_months = 9 →
  first_rate = 0.09 →
  final_value = 17218.50 →
  (∃ s : ℚ, 16012.50 * (1 + (s * 0.75) / 100) = final_value) →
  s = 10 := 
by
  sorry

end find_interest_rate_l88_88018


namespace minimize_water_tank_construction_cost_l88_88365

theorem minimize_water_tank_construction_cost 
  (volume : ℝ := 4800)
  (depth : ℝ := 3)
  (cost_bottom_per_m2 : ℝ := 150)
  (cost_walls_per_m2 : ℝ := 120)
  (x : ℝ) :
  (volume = x * x * depth) →
  (∀ y, y = cost_bottom_per_m2 * x * x + cost_walls_per_m2 * 4 * x * depth) →
  (x = 40) ∧ (y = 297600) :=
by
  sorry

end minimize_water_tank_construction_cost_l88_88365


namespace dot_product_EC_ED_l88_88587

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l88_88587


namespace probability_of_first_three_heads_l88_88295

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l88_88295


namespace min_sum_of_squares_of_roots_l88_88874

theorem min_sum_of_squares_of_roots :
  ∃ a : ℝ, (∀ b : ℝ, (b ≠ a) → 
    let f (a : ℝ) := (a - 2)^2 + 2 * (a + 1) in 
    f b ≥ f a) ∧ a = 1 :=
by
  -- Summarize the theorem
  sorry

end min_sum_of_squares_of_roots_l88_88874


namespace unbounded_diff_then_density_zero_l88_88620

-- Definitions
variables {A : set ℕ} (hA_nonempty : A.nonempty) (hA_pos : ∀ a ∈ A, 1 ≤ a)

def N (x : ℕ) : ℕ := {a ∈ A | a ≤ x}.to_finset.card

def B : set ℕ := {b | ∃ a a' ∈ A, b = a - a'}

def b_seq (n : ℕ) : ℕ :=
  if h : ∃ b ∈ B, ∀ b' ∈ B, b' < b → n ≤ b'
  then classical.some h else 0

-- Theorem
theorem unbounded_diff_then_density_zero
  (h_unbounded_diff : ∀ n : ℕ, ∃ k : ℕ, b_seq (k + 1) - b_seq k > n) :
  tendsto (λ x, (N x : ℝ) / x) at_top (𝓝 0) :=
begin
  sorry
end

end unbounded_diff_then_density_zero_l88_88620


namespace probability_three_heads_l88_88326

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l88_88326


namespace perfect_square_divisors_of_product_factorials_l88_88827

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def product_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).map factorial |>.foldl (*) 1

theorem perfect_square_divisors_of_product_factorials :
  (number_of_perfect_square_divisors (product_factorials 12) = 98304) :=
by
  sorry

end perfect_square_divisors_of_product_factorials_l88_88827


namespace find_k_find_range_of_g_l88_88515

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

noncomputable def f (a : ℝ) (k : ℝ) := λ x: ℝ, k * (a^x) - (a^(-x))

def g (a : ℝ) := λ x: ℝ, (a^(2*x)) + (a^(-2*x)) - 4 * (f a 1 x)

theorem find_k {a: ℝ} (h1 : a > 0) (h2 : a ≠ 1)
(h3 : is_odd_function (f a 1)) :
  (1:ℝ) = 1 := sorry

theorem find_range_of_g {a: ℝ} (h1 : a > 0) (h2 : a ≠ 1) (h4 : f a 1 1 = (3 / 2)) :
  let g := g a in
  ∀ x ∈ set.Icc (1:ℝ) 2, g x ∈ set.Icc (-2:ℝ) (17 / 16) := sorry

end find_k_find_range_of_g_l88_88515


namespace num_distinct_terms_expanded_l88_88132

-- Define the multinomial expression and the problem
def multinomial_expr (x : ℝ) : ℝ := 4 * x^3 + x^(-3) + 2

-- State the main theorem
theorem num_distinct_terms_expanded (n : ℕ) (h : n = 2016) : 
  let expr := (multinomial_expr x)^n in
  -- Express that the number of distinct terms is equal to 4033
  -- when expanded and like terms are combined
  (number_of_distinct_terms expr) = 4033 :=
sorry

end num_distinct_terms_expanded_l88_88132


namespace text_message_costs_equal_l88_88334

theorem text_message_costs_equal (x : ℝ) : 
  (0.25 * x + 9 = 0.40 * x) ∧ (0.25 * x + 9 = 0.20 * x + 12) → x = 60 :=
by 
  sorry

end text_message_costs_equal_l88_88334


namespace binomial_20_10_l88_88910

open Nat

theorem binomial_20_10 :
  (binomial 18 8 = 43758) →
  (binomial 18 9 = 48620) →
  (binomial 18 10 = 43758) →
  binomial 20 10 = 184756 :=
by
  intros h1 h2 h3
  sorry

end binomial_20_10_l88_88910


namespace number_of_digits_in_product_is_20_l88_88837

-- Define the product and the number of digits function
def product := (6 ^ 7) * (7 ^ 14) * (3 ^ 5)

def number_of_digits (n : ℕ) : ℕ := (Real.log10 n).floor + 1

-- Statement of the theorem to prove the number of digits in the product
theorem number_of_digits_in_product_is_20 : number_of_digits product = 20 := by
  sorry

end number_of_digits_in_product_is_20_l88_88837


namespace equation_of_ellipse_exist_point_E_and_constant_value_l88_88497

noncomputable def ellipse_eccentricity (a b c : ℝ) := c / a = (Real.sqrt 6) / 3

noncomputable def circle_tangent_line (a : ℝ) : Prop := 
  let radius := a in 
  ∃ x y : ℝ, x^2 + y^2 = a^2 ∧ 2 * x - Real.sqrt 2 * y + 6 = 0

noncomputable def points_intersection (k a b : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 
  let quad := (1 + 3 * k^2) * x1^2 - 12 * k^2 * x2 + 12 * k^2 - 6 = 0 in
  (x1 + x2 = (12 * k^2) / (1 + 3 * k^2)) ∧ 
  (x1 * x2 = (12 * k^2 - 6) / (1 + 3 * k^2))

theorem equation_of_ellipse (a b c : ℝ) :
  (a > 0 ∧ b > 0 ∧ a > b ∧ ellipse_eccentricity a b c ∧ circle_tangent_line a) → 
  (c = 2 ∧ b^2 = 2 ∧ ∀ x y : ℝ, x^2 / 6 + y^2 / 2 = 1) := by
  sorry

theorem exist_point_E_and_constant_value (k : ℝ) (a b : ℝ) :
  (k ≠ 0 ∧ a > 0 ∧ b > 0 ∧ points_intersection k a b) → 
  ∃ m : ℝ, m = 7 / 3 ∧ (m^2 - 6) = -5 / 9 := by
  sorry

end equation_of_ellipse_exist_point_E_and_constant_value_l88_88497


namespace distinct_four_digit_integers_with_product_18_l88_88970

theorem distinct_four_digit_integers_with_product_18 :
  ∃ n : ℕ, n = 24 ∧ ∀ (d1 d2 d3 d4 : ℕ), (d1 * d2 * d3 * d4 = 18 ∧ 1000 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000) →
    set.finite { x | ∃ (d1 d2 d3 d4 : ℕ), x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ d1 * d2 * d3 * d4 = 18 ∧ ∀ i ∈ [d1, d2, d3, d4], 1 ≤ i ∧ i ≤ 9 } :=
begin
  sorry
end

end distinct_four_digit_integers_with_product_18_l88_88970


namespace range_of_g_l88_88507

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^m

theorem range_of_g (m : ℝ) (h : m > 0) : set.range (λ x, g x m) = set.Ioc 0 1 := by
  sorry

end range_of_g_l88_88507


namespace parcel_post_cost_l88_88236

theorem parcel_post_cost (P : ℕ) (h : 1 ≤ P) :
  let first_pound_cost := 10
  let additional_pound_cost := 3
  let C := first_pound_cost + additional_pound_cost * (P - 1) in
  C = 10 + 3 * (P - 1) :=
by
  sorry

end parcel_post_cost_l88_88236


namespace length_of_platform_l88_88011

theorem length_of_platform 
  (speed_kmph : ℕ)
  (time_cross_platform : ℕ)
  (time_cross_man : ℕ)
  (speed_mps : ℕ)
  (length_of_train : ℕ)
  (distance_platform : ℕ)
  (length_of_platform : ℕ) :
  speed_kmph = 72 →
  time_cross_platform = 30 →
  time_cross_man = 16 →
  speed_mps = speed_kmph * 1000 / 3600 →
  length_of_train = speed_mps * time_cross_man →
  distance_platform = speed_mps * time_cross_platform →
  length_of_platform = distance_platform - length_of_train →
  length_of_platform = 280 := by
  sorry

end length_of_platform_l88_88011


namespace divides_expression_l88_88675

theorem divides_expression (n : ℤ) : (n - 1) ∣ (n^(3*n + 1) - 3*n^4 + 2) :=
sorry

end divides_expression_l88_88675


namespace max_power_speed_l88_88701

def aerodynamic_force (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v)^2) / 2

def power (C S ρ v₀ v : ℝ) : ℝ :=
  aerodynamic_force C S ρ v₀ v * v

theorem max_power_speed (C S ρ v₀ v : ℝ) (h₁ : v = v₀ / 3) :
  ∃ v, power C S ρ v₀ v = (C * S * ρ * v₀^3) / 54 :=
begin
  use v₀ / 3,
  sorry
end

end max_power_speed_l88_88701


namespace sequences_sum_l88_88494

noncomputable def arithmetic_seq (a1 d : ℕ) : ℕ → ℕ
| 0       := a1
| (n + 1) := a1 + (n + 1) * d

noncomputable def geometric_seq (c1 q : ℕ) : ℕ → ℕ
| 0       := c1
| (n + 1) := c1 * q^n

theorem sequences_sum (a1 a5 b1 b5 : ℕ) (h_a : a5 = a1 + 4 * 3) (h_b : b5 = a1 + 4 * 3 + (b1 + 28 * 1)) :
  let a_n := λ n : ℕ, a1 + n * 3 in
  let c_n := λ n : ℕ, (b1 - a1) * 2^(n - 1) in
  let b_n := λ n : ℕ, a_n n + c_n n in
  (a_n 4 = 3 * 4 ∧ c_n 4 = 16) ∧ (finset.range n).sum b_n = (3 * n^2 + 3 * n) / 2 + 2^n - 1 :=
by sorry

end sequences_sum_l88_88494


namespace max_value_sin_transform_l88_88265

open Real

theorem max_value_sin_transform (φ : ℝ) (hφ : |φ| < π / 2) :
  let f (x : ℝ) := sin (2 * x + φ)
  ∃ x ∈ Icc (0 : ℝ) (π / 2), f x = 1 := 
begin
  let g (x : ℝ) := sin (2 * x + 2 * π / 3 + φ),
  have hg_symm : ∀ x, g x = - g (-x), from sorry, -- Using symmetry about the origin
  have hφ_eq : 2 * π / 3 + φ = π, from sorry, -- Derived from symmetry condition
  let φ := π / 3,
  have f_def : f = (λ x, sin (2 * x + π / 3)), 
  { unfold f, rw [hφ_eq], sorry },
  
  have max_value_interval : ∃ x ∈ Icc (0 : ℝ) (π / 2), sin (2 * x + π / 3) = 1,
  { use π / 6,
    split,
    { split,
      { linarith },
      { linarith }},
    { rw [sin_add],
      have : sin (π / 2) = 1 := sin_pi_div_two,
      rw [this, mul_div_cancel_left π (show 2 ≠ 0, by linarith)], 
      rw [sin_pi, cos_pi_div_two, zero_mul, add_zero],
      exact this }},
  exact max_value_interval
end

end max_value_sin_transform_l88_88265


namespace answer_l88_88574

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

end answer_l88_88574


namespace probability_of_transformation_l88_88381

noncomputable def unit_square := set.prod (set.Icc 0 1) (set.Icc 0 1)

def transformation (p : ℝ × ℝ) : ℝ × ℝ :=
  (3 * p.1 + 2 * p.2, p.1 + 4 * p.2)

def is_in_unit_square (p : ℝ × ℝ) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1

theorem probability_of_transformation :
  let a := 7
  let b := 120
  let probability := (7:ℝ) / 120
  ∀ p : ℝ × ℝ, p ∈ unit_square → is_in_unit_square (transformation p) → 100 * a + b = 820 :=
by
  sorry

end probability_of_transformation_l88_88381


namespace pears_equivalence_l88_88225

theorem pears_equivalence :
  (3 / 4 : ℚ) * 16 * (5 / 6) = 10 → 
  (2 / 5 : ℚ) * 20 * (5 / 6) = 20 / 3 := 
by
  intros h
  sorry

end pears_equivalence_l88_88225


namespace probability_of_first_three_heads_l88_88292

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l88_88292


namespace dot_product_square_ABCD_l88_88603

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l88_88603


namespace limit_a_n_limit_p1_an_n_l88_88625

noncomputable theory

open Real
open TopologicalSpace

variables {p : ℕ} (f : ℝ → ℝ) (hf : ContinuousOn f (Icc 0 1)) (hf_pos : ∀ x ∈ Icc (0:ℝ) 1, f x > 0)

def a_n (n: ℕ) : ℝ :=
  ∫ x in 0..1, (x ^ p) * (f x) ^ (1 / n.toFloat)

-- Part (a)
theorem limit_a_n : 
  (tendsto (λ n : ℕ, ∫ x in (0:ℝ)..1, (x ^ p) * (f x) ^ (1 / n.toFloat)) atTop (𝓝 (1 / (p + 1)))) :=
sorry

-- Part (b)
theorem limit_p1_an_n :
  (tendsto (λ n : ℕ, ((p + 1) * (∫ x in (0:ℝ)..1, (x ^ p) * (f x) ^ (1 / n.toFloat))) ^ n) atTop 
    (𝓝 (exp ((p + 1) * (∫ x in (0:ℝ)..1, (x ^ p) * log (f x))))) :=
sorry

end limit_a_n_limit_p1_an_n_l88_88625


namespace markese_earned_16_l88_88646

def evan_earnings (E : ℕ) : Prop :=
  (E : ℕ)

def markese_earnings (M : ℕ) (E : ℕ) : Prop :=
  (M : ℕ) = E - 5

def total_earnings (E M : ℕ) : Prop :=
  E + M = 37

theorem markese_earned_16 (E : ℕ) (M : ℕ) 
  (h1 : markese_earnings M E) 
  (h2 : total_earnings E M) : M = 16 :=
sorry

end markese_earned_16_l88_88646


namespace binom_20_10_l88_88919

-- Definitions for the provided conditions
def binom_18_8 := 43758
def binom_18_9 := 48620
def binom_18_10 := 43758

-- The theorem we need to prove
theorem binom_20_10 : ∀
  (binom_18_8 = 43758)
  (binom_18_9 = 48620)
  (binom_18_10 = 43758),
  binomial 20 10 = 184756 :=
by
  sorry

end binom_20_10_l88_88919


namespace find_AB_max_area_l88_88172

noncomputable def triangle_geom (A B C : ℝ) (AB AC BC : ℝ) : Prop :=
AC = 3 ∧ 
∃ a b c : ℝ, (2 * b = a + c) ∧ (a + b + c = π) ∧ 
cos c = sqrt 6 / 3

theorem find_AB 
  {A B C : ℝ} 
  {AB AC BC : ℝ} 
  (h1 : AC = 3) 
  (h2 : ∃ a b c : ℝ, 2 * b = a + c ∧ a + b + c = π) 
  (h3 : cos C = sqrt 6 / 3) : 
  AB = 2 := 
sorry

theorem max_area 
  {A B C : ℝ} 
  {AB AC BC : ℝ} 
  (h1 : AC = 3) 
  (h2 : ∃ a b c : ℝ, 2 * b = a + c ∧ a + b + c = π) 
  (h3 : cos C = sqrt 6 / 3) : 
  (area : ℝ) = 9 * sqrt 3 / 4 :=
sorry

end find_AB_max_area_l88_88172


namespace find_t_l88_88535

def vector (α : Type*) := list α

def dot_product {α : Type*} [semiring α] (v₁ v₂ : vector α) : α :=
list.sum (list.zip_with (*) v₁ v₂)

def is_perpendicular {α : Type*} [semiring α] (v₁ v₂ : vector α) : Prop :=
dot_product v₁ v₂ = 0

def a (t : ℝ) : vector ℝ := [1, 1, t]
def b : vector ℝ := [-1, 0, 2]

theorem find_t (t : ℝ) (h : is_perpendicular b (list.zip_with (+) (a t) b)) : t = -2 :=
sorry

end find_t_l88_88535


namespace fuel_needed_for_500_miles_l88_88353

theorem fuel_needed_for_500_miles (fuel_needed_for_1000_miles : ℕ) (rate_of_fuel : ℕ) :
  fuel_needed_for_1000_miles = 40 → rate_of_fuel = (fuel_needed_for_1000_miles / 2) → rate_of_fuel = 20 :=
by
  intros h1 h2
  rw h1 at h2
  exact h2

end fuel_needed_for_500_miles_l88_88353


namespace choose_3_from_12_l88_88571

theorem choose_3_from_12 : (Nat.choose 12 3) = 220 := by
  sorry

end choose_3_from_12_l88_88571


namespace prob_first_three_heads_all_heads_l88_88317

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l88_88317


namespace length_of_DC_l88_88164

theorem length_of_DC (AB : ℝ) (angle_ADB : ℝ) (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30) (h2 : angle_ADB = pi / 2) (h3 : sin_A = 3 / 5) (h4 : sin_C = 1 / 4) :
  ∃ DC : ℝ, DC = 18 * Real.sqrt 15 :=
by
  sorry

end length_of_DC_l88_88164


namespace parallel_line_length_l88_88689

theorem parallel_line_length (base : ℝ) (ratios : ℕ × ℕ × ℕ) (l1 l2 : ℝ) 
  (h_base : base = 18) 
  (h_ratios : ratios = (1, 2, 2))
  (h_ratios_areas : ∃ a b c, a + b + c = base ∧ a = (ratios.1 : ℝ) * l1 ∧ b = (ratios.2 : ℝ) * l2 
    ∧ c = (ratios.3 : ℝ) * l2) : l2 = 8.04 :=
by {
  sorry
}

end parallel_line_length_l88_88689


namespace cos_neg_alpha_l88_88928

theorem cos_neg_alpha (α : ℝ) (P : ℝ × ℝ) (hx : P.1 = 4) (hy : P.2 = -3) :
  ∃ r : ℝ, r = 5 ∧ cos (-α) = 4 / r := 
by 
  use 5
  split
  . rfl
  . sorry

end cos_neg_alpha_l88_88928


namespace coeff_x3_binom_expansion_l88_88275

theorem coeff_x3_binom_expansion : 
  ∀ (C : ℕ → ℕ → ℕ) [∀ n k, Decidable (C n k)], 
  ∀ (binomial_expand : (ℕ → ℕ → ℕ) → ℕ → (ℕ → ℕ → (ℕ → Type u)) → (ℕ → (ℕ → Type u)) → Type u), 
  ∀ (coeff_x_to_r : (ℕ → ℕ → ℕ) → (ℕ → Type u) → ℕ → (ℕ → (ℕ → Type u)) → (ℕ → (ℕ → Type u)) → ℕ) 
    [∀ (C : ℕ → ℕ → ℕ) (n : ℕ) (f : ℕ → ℕ → (ℕ → Type u)) (g : ℕ → (ℕ → Type u)), Decidable (coeff_x_to_r C n f g)], 
    coeff_x_to_r C (λ n, λ x, (2*x+1)^[n]) 3 binomial_expand = 80 :=
by
  -- Detailed proof goes here
sorry

end coeff_x3_binom_expansion_l88_88275


namespace min_value_of_f_l88_88879

noncomputable def f (x : ℝ) : ℝ := (real.exp x - 1) ^ 2 + (real.exp (-x) - 1) ^ 2

theorem min_value_of_f : ∃ x : ℝ, f x = 0 := 
begin
  use 0,
  -- Here we should show that f(0) = 0
  simp [f, real.exp_zero],
  norm_num,
end

end min_value_of_f_l88_88879


namespace triangle_side_lengths_l88_88554

theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) 
  (hcosA : Real.cos A = 1/4)
  (ha : a = 4)
  (hbc_sum : b + c = 6)
  (hbc_order : b < c) :
  b = 2 ∧ c = 4 := by
  sorry

end triangle_side_lengths_l88_88554


namespace prove_g_f_neg1_l88_88109

def f (x : ℤ) : ℤ :=
if x > 0 then -x^2 else 2^x

def g (x : ℤ) : ℤ :=
if x > 0 then -(1/x) else x - 1

theorem prove_g_f_neg1 :
  g (f (-1)) = -2 :=
by
-- Definitions of f and g in the context of Lean will be used to prove this theorem
-- Skipping the proof as per instructions
sorry

end prove_g_f_neg1_l88_88109


namespace arithmetic_sequence_properties_l88_88606

theorem arithmetic_sequence_properties :
  (∃ a : ℕ → ℕ, (2 * a 1 + 3 * a 2 = 11) ∧ (2 * a 3 = a 2 + a 6 - 4) ∧
  (∀ n : ℕ, a n = 2 * n - 1)) →
  let S : ℕ → ℕ := λ n, n^2
  let b : ℕ → ℚ := λ n, 1 / (S n + n)
  let T : ℕ → ℚ := λ n, ∑ i in finset.range (n + 1), b i
  (∀ n : ℕ, T n = n / (n + 1)) :=
sorry

end arithmetic_sequence_properties_l88_88606


namespace find_unit_costs_and_schemes_l88_88791

-- Define parameters and conditions
def AirConditionerPrice (x y : ℕ) : Prop :=
  3 * x + 2 * y = 39000 ∧ 4 * x - 5 * y = 6000

def PurchasingSchemes (a b : ℕ) : Prop :=
  a + b = 30 ∧ 2 * a ≥ b ∧ 9000 * a + 6000 * b ≤ 217000

theorem find_unit_costs_and_schemes :
  ∃ (x y : ℕ), AirConditionerPrice x y ∧
  x = 9000 ∧ y = 6000 ∧
  (∃ (a b : ℕ), PurchasingSchemes a b ∧
    (a, b) ∈ {(10, 20), (11, 19), (12, 18)} ∧
    (a = 10 ∧ b = 20 → 9000 * a + 6000 * b = 210000)) :=
  sorry -- Proof to be completed

end find_unit_costs_and_schemes_l88_88791


namespace triangle_external_angle_bisector_l88_88660

theorem triangle_external_angle_bisector {A B C M : Type}  
  [metric_space M] [has_dist M] 
  (triangle_ABC : triangle A B C)
  (M_on_bisector : ∃ (M : M), M ≠ C ∧ is_external_angle_bisector C A B M) :
  dist M A + dist M B > dist C A + dist C B :=
by
  sorry

end triangle_external_angle_bisector_l88_88660


namespace binom_20_10_eq_184756_l88_88905

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l88_88905


namespace no_such_triplets_of_positive_reals_l88_88129

-- Define the conditions that the problem states.
def satisfies_conditions (a b c : ℝ) : Prop :=
  a = b + c ∧ b = c + a ∧ c = a + b

-- The main theorem to prove.
theorem no_such_triplets_of_positive_reals :
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) → satisfies_conditions a b c → false :=
by
  intro a b c
  intro ha hb hc
  intro habc
  sorry

end no_such_triplets_of_positive_reals_l88_88129


namespace incorrect_companion_conclusions_l88_88071

noncomputable def l_companion_function (f : ℝ → ℝ) (l : ℝ) : Prop :=
  ∀ x : ℝ, f(x + l) + l * f(x) = 0

theorem incorrect_companion_conclusions (f : ℝ → ℝ) (l : ℝ) :
  continuous f →
  (∀ x : ℝ, f(x + l) + l * f(x) = 0) →
  ¬ (f = fun x => x^2 ∨ f = fun x => 0) :=
by
  intros h_cont h_l_companion
  have h1 : ¬(f = fun x => 0) := 
    sorry
  have h2 : ¬(f = fun x => x^2) := 
    sorry
  exact
    fun h =>
      match h with
      | Or.inl h_eq_zero => h1 h_eq_zero
      | Or.inr h_eq_x2 => h2 h_eq_x2

end incorrect_companion_conclusions_l88_88071


namespace length_of_road_l88_88369

theorem length_of_road (L : ℕ) 
  (h1 : ∃ k, L = 8 * k + (8 * 9))
  (h2 : ∃ m, L = 9 * m - (7 * 8)) : 
  L = 576 := 
begin
  sorry
end

end length_of_road_l88_88369


namespace crabs_first_day_is_72_l88_88254

noncomputable def crabs_first_day (total : ℕ) : ℕ :=
let oysters1 := 50 in
let oysters2 := oysters1 / 2 in
let crabs1 := (3 * (total - oysters1 - oysters2)) / 5 in
crabs1

theorem crabs_first_day_is_72 : crabs_first_day 195 = 72 := sorry

end crabs_first_day_is_72_l88_88254


namespace beyonce_total_songs_l88_88027

-- Define the conditions as given in the problem
def singles : ℕ := 5
def albums_15_songs : ℕ := 2
def songs_per_15_album : ℕ := 15
def albums_20_songs : ℕ := 1
def songs_per_20_album : ℕ := 20

-- Define a function to calculate the total number of songs released by Beyonce
def total_songs_released : ℕ :=
  singles + (albums_15_songs * songs_per_15_album) + (albums_20_songs * songs_per_20_album)

-- Theorem statement for the total number of songs released
theorem beyonce_total_songs {singles albums_15_songs songs_per_15_album albums_20_songs songs_per_20_album : ℕ} :
  singles = 5 →
  albums_15_songs = 2 →
  songs_per_15_album = 15 →
  albums_20_songs = 1 →
  songs_per_20_album = 20 →
  total_songs_released = 55 :=
by {
  intros h_singles h_albums_15_songs h_songs_per_15_album h_albums_20_songs h_songs_per_20_album,
  -- replace with the proven result
  sorry
}

end beyonce_total_songs_l88_88027


namespace min_value_f_l88_88881

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1)^2 + (Real.exp (-x) - 1)^2

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 :=
sorry

end min_value_f_l88_88881


namespace triangle_DFG_area_l88_88662

theorem triangle_DFG_area (a b x y : ℝ) (h_ab : a * b = 20) (h_xy : x * y = 8) : 
  (a * b - x * y) / 2 = 6 := 
by
  sorry

end triangle_DFG_area_l88_88662


namespace prob1_prob2_l88_88894

theorem prob1 (a : ℕ → ℝ) (a_1 : ℝ) :
  (∀ n, a (n + 1) = a n + 1) → 
  (1, a_1, a 3) forms_geometric_progression → 
  a_1 = 2 ∨ a_1 = -1 :=
sorry

theorem prob2 (a : ℕ → ℝ) (a_1 : ℝ) :
  (∀ n, a (n + 1) = a n + 1) → 
  S 5 a > a_1 * a 9 →
  -5 < a_1 ∧ a_1 < 2 :=
sorry

end prob1_prob2_l88_88894


namespace workers_combined_time_l88_88763

theorem workers_combined_time: 
  let rA := 1 / 7 in
  let rB := 1 / 10 in
  let rC := 1 / 12 in
  let combined_rate := rA + rB + rC in
  let time_to_complete := 1 / combined_rate in
  time_to_complete = 420 / 137 :=
by
  let rA := 1 / 7
  let rB := 1 / 10
  let rC := 1 / 12
  let combined_rate := rA + rB + rC
  let time_to_complete := 1 / combined_rate
  have h : combined_rate = 137 / 420 := by sorry
  have t : time_to_complete = 420 / 137 := by sorry
  sorry

end workers_combined_time_l88_88763


namespace project_selection_count_l88_88557

theorem project_selection_count :
  let key_projects := {1, 2, 3, 4} -- 4 key projects
  let general_projects := {5, 6, 7, 8, 9, 10} -- 6 general projects
  let A := 1 -- key project A
  let B := 5 -- general project B
  let count_selection_with_A : ℕ := (3.choose 1) * (6.choose 2) -- ways including A
  let count_selection_with_B : ℕ := (4.choose 2) * (5.choose 1) -- ways including B
  let count_selection_with_A_and_B : ℕ := (3.choose 1) * (5.choose 1) -- ways including both A and B
  count_selection_with_A + count_selection_with_B - count_selection_with_A_and_B = 60 :=
by
  let key_projects := {1, 2, 3, 4}
  let general_projects := {5, 6, 7, 8, 9, 10}
  let A := 1
  let B := 5
  let count_selection_with_A := (3.choose 1) * (6.choose 2)
  let count_selection_with_B := (4.choose 2) * (5.choose 1)
  let count_selection_with_A_and_B := (3.choose 1) * (5.choose 1)
  have h1 : count_selection_with_A = 45 := by
    sorry
  have h2 : count_selection_with_B = 30 := by
    sorry
  have h3 : count_selection_with_A_and_B = 15 := by
    sorry
  have h4 : count_selection_with_A + count_selection_with_B - count_selection_with_A_and_B = 60 := by
    rw [h1, h2, h3]
    exact rfl
  exact h4

end project_selection_count_l88_88557


namespace calculate_expression_l88_88412

theorem calculate_expression :
  abs (1 - 3) * ((-12) - 2^3) = -40 :=
by
  have h1 : abs (1 - 3) = 2 := by sorry
  have h2 : 2^3 = 8 := by sorry
  have h3 : (-12) - 8 = -20 := by sorry
  have h4 : 2 * (-20) = -40 := by sorry
  rw [h1, h2, h3, h4]
  rfl

end calculate_expression_l88_88412


namespace distinct_four_digit_numbers_product_18_l88_88979

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l88_88979


namespace chess_tournament_games_l88_88153

def stage1_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage2_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage3_games : ℕ := 4

def total_games (stage1 stage2 stage3 : ℕ) : ℕ := stage1 + stage2 + stage3

theorem chess_tournament_games : total_games (stage1_games 20) (stage2_games 10) stage3_games = 474 :=
by
  unfold stage1_games
  unfold stage2_games
  unfold total_games
  simp
  sorry

end chess_tournament_games_l88_88153


namespace problem_l88_88641

-- Given definitions
variables {n : ℕ} (a : ℕ → ℝ) 
def s (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, if i % 2 = 0 then a (2 * i + 1) else 0 -- sum of odd indexed terms
def t (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, if i % 2 = 1 then a (2 * i) else 0 -- sum of even indexed terms
def x (a : ℕ → ℝ) (k : ℕ) (n : ℕ) : ℝ := ∑ i in finset.range n, a ((k + i) % (2 * n)) -- x_k 

-- Theorem Statement
theorem problem 
  (h : ∀ i, 0 < a i) : 
  (∑ i in finset.range (2 * n), (if i % 2 = 0 then s a n / x a i n else t a n / x a i n)) > (2 * n ^ 2 / (n + 1)) :=
sorry

end problem_l88_88641


namespace circumcircles_common_point_l88_88196

-- Assume variables and geometric entities
variables (A B C M N O R K : Point)

-- Assume conditions are given as hypotheses
def is_acute_triangle (a b c : Point) : Prop := 
  acute_angle a b c ∧ acute_angle b c a ∧ acute_angle c a b

def on_diameter_circle (d1 d2 m : Point) : Prop := 
  diameter_circle d1 d2 m

def midpoint (o b c : Point) : Prop := 
  o = (b + c) / 2

def angle_bisectors_intersect (r a1 a2 : Point) : Prop := 
  intersect_bisectors r a1 a2

-- Main theorem statement
theorem circumcircles_common_point
  {A B C M N O R : Point}
  (h1 : is_acute_triangle A B C)
  (h2 : on_diameter_circle B C M)
  (h3 : on_diameter_circle B C N)
  (h4 : midpoint O B C)
  (h5 : angle_bisectors_intersect R A (segment_angle M O N))
  : (∃ K, lies_on K (line B C) ∧ circumcircle_contains K B M R ∧ circumcircle_contains K C N R) :=
sorry

end circumcircles_common_point_l88_88196


namespace number_of_real_solutions_l88_88958

theorem number_of_real_solutions : 
  (∃ (x : ℝ), (x^4 - 5)^4 = 81) ∧ set.countable { x : ℝ | (x^4 - 5)^4 = 81 } = 4 :=
sorry

end number_of_real_solutions_l88_88958


namespace average_age_l88_88688

-- Define the present ages of the two sons
variables (S1 S2 : ℕ)

-- Define the conditions
axiom h1 : (S1 - 5 + S2 - 5) / 2 = 15
axiom h2 : S1 - S2 = 4
axiom father_age : ℕ := 32

-- Define the proof problem (find the average age is 24)
theorem average_age (S1 S2 : ℕ) (h1 : (S1 - 5 + S2 - 5) / 2 = 15) (h2 : S1 - S2 = 4) : 
  (father_age + S1 + S2) / 3 = 24 := 
by 
  -- Here we skip the proof and provide the sorry placeholder
  sorry

end average_age_l88_88688


namespace inequality_least_one_l88_88629

theorem inequality_least_one {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4) :=
by
  sorry

end inequality_least_one_l88_88629


namespace Megatech_budget_allocation_l88_88356

theorem Megatech_budget_allocation :
  let total_degrees := 360
  let degrees_astrophysics := 90
  let home_electronics := 19
  let food_additives := 10
  let genetically_modified_microorganisms := 24
  let industrial_lubricants := 8

  let percentage_astrophysics := (degrees_astrophysics / total_degrees) * 100
  let known_percentages_sum := home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + percentage_astrophysics
  let percentage_microphotonics := 100 - known_percentages_sum

  percentage_microphotonics = 14 :=
by
  sorry

end Megatech_budget_allocation_l88_88356


namespace crow_distance_l88_88360

theorem crow_distance (trips: ℕ) (hours: ℝ) (speed: ℝ) (distance: ℝ) :
  trips = 15 → hours = 1.5 → speed = 4 → (trips * 2 * distance) = (speed * hours) → distance = 200 / 1000 :=
by
  intros h_trips h_hours h_speed h_eq
  sorry

end crow_distance_l88_88360


namespace arithmetic_sum_problem_l88_88893

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sum_problem
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms S a)
  (h_S10 : S 10 = 4) :
  a 3 + a 8 = 4 / 5 := 
sorry

end arithmetic_sum_problem_l88_88893


namespace ferris_wheel_time_l88_88351

theorem ferris_wheel_time {radius : ℝ} (r : radius = 30)
                         {rate : ℝ} (v : rate = 1 / 2)
                         {start_height bottom_height : ℝ} (start_bottom : start_height = 0)
                         {time_to_top descend_time total_time : ℝ} 
                         (h_period : time_to_top = 60) 
                         (descend_height : bottom_height = 20) 
                         (total_time_correct : total_time = 70) :
                         total_time = time_to_top + 
                                     (60 / π * real.arccos (- 1 / 3)) :=
begin
  sorry,
end

end ferris_wheel_time_l88_88351


namespace sequence_is_arithmetic_not_geometric_l88_88091

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 6 / Real.log 2
noncomputable def c := Real.log 12 / Real.log 2

theorem sequence_is_arithmetic_not_geometric : 
  (b - a = c - b) ∧ (b / a ≠ c / b) := 
by
  sorry

end sequence_is_arithmetic_not_geometric_l88_88091


namespace probability_of_first_three_heads_l88_88294

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l88_88294


namespace arithmetic_mean_add_two_l88_88068

def arithmetic_mean (a b c : ℝ) : ℝ :=
  (a + b + c) / 3

def new_mean (mean : ℝ) : ℝ :=
  mean + 2

theorem arithmetic_mean_add_two 
  (a b c : ℝ) (mean : ℝ) (new_mean_result : ℝ) :
  a = 13 → b = 27 → c = 34 →
  mean = arithmetic_mean a b c →
  new_mean_result = new_mean mean →
  Float.round (new_mean_result * 10) / 10 = 26.7 :=
by
  sorry

end arithmetic_mean_add_two_l88_88068


namespace binom_20_10_l88_88900

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l88_88900


namespace answer_l88_88578

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

end answer_l88_88578


namespace find_x_solution_l88_88863

theorem find_x_solution (x : ℝ) (h : sqrt (3 * x + 15) = 12) : x = 43 :=
by {
  sorry
}

end find_x_solution_l88_88863


namespace find_speeds_l88_88462

noncomputable def speed_pedestrian := 5
noncomputable def speed_cyclist := 30

def distance_AB := 40
def starting_time_pedestrian := 4 -- In hours (24-hour format)
def starting_time_cyclist_1 := 7 + 20 / 60 -- 7:20 AM in hours
def halfway_distance := distance_AB / 2
def midpoint_meeting_time := 1 -- Time (in hours) after the first meeting
def starting_time_cyclist_2 := 8 + 30 / 60 -- 8:30 AM in hours

theorem find_speeds (x y : ℝ) (hx : x = speed_pedestrian) (hy : y = speed_cyclist) :
  let time_to_halfway := halfway_distance / x in
  let cyclist_time := (midpoint_meeting_time + time_to_halfway) in
  distance_AB = 
    cyclist_time * y + 
    time_to_halfway * x + 
    (midpoint_meeting_time - 1) * x :=
    x = speed_pedestrian ∧ y = speed_cyclist :=
begin
  sorry
end

end find_speeds_l88_88462


namespace earl_start_floor_l88_88845

theorem earl_start_floor : ∃ (F : ℕ), F + 10 = 11 := by
  use 1
  sorry

end earl_start_floor_l88_88845


namespace value_of_f_of_x_minus_3_l88_88189

theorem value_of_f_of_x_minus_3 (x : ℝ) (f : ℝ → ℝ) (h : ∀ y : ℝ, f y = y^2) : f (x - 3) = x^2 - 6*x + 9 :=
by
  sorry

end value_of_f_of_x_minus_3_l88_88189


namespace find_speeds_l88_88479

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l88_88479


namespace find_speeds_l88_88465

noncomputable def speed_pedestrian := 5
noncomputable def speed_cyclist := 30

def distance_AB := 40
def starting_time_pedestrian := 4 -- In hours (24-hour format)
def starting_time_cyclist_1 := 7 + 20 / 60 -- 7:20 AM in hours
def halfway_distance := distance_AB / 2
def midpoint_meeting_time := 1 -- Time (in hours) after the first meeting
def starting_time_cyclist_2 := 8 + 30 / 60 -- 8:30 AM in hours

theorem find_speeds (x y : ℝ) (hx : x = speed_pedestrian) (hy : y = speed_cyclist) :
  let time_to_halfway := halfway_distance / x in
  let cyclist_time := (midpoint_meeting_time + time_to_halfway) in
  distance_AB = 
    cyclist_time * y + 
    time_to_halfway * x + 
    (midpoint_meeting_time - 1) * x :=
    x = speed_pedestrian ∧ y = speed_cyclist :=
begin
  sorry
end

end find_speeds_l88_88465


namespace two_trainees_same_number_known_l88_88556

variable (T : Type) [Fintype T] (knows : T → T → Prop)
  (mutual : ∀ a b, knows a b ↔ knows b a)

theorem two_trainees_same_number_known (h : Fintype.card T = 62) :
  ∃ a b : T, a ≠ b ∧ (Fintype.card {x // knows a x} = Fintype.card {x // knows b x}) := by
sorry

end two_trainees_same_number_known_l88_88556


namespace sum_of_fractions_l88_88737

theorem sum_of_fractions : (3/7 : ℚ) + (5/14 : ℚ) = 11/14 :=
by
  sorry

end sum_of_fractions_l88_88737


namespace third_nail_division_l88_88019

-- Define the properties of the equilateral triangle and divisions
structure EquilateralTriangle (A B C : Point) :=
(equilateral : distance A B = distance B C ∧ distance B C = distance C A)

def division_ratio (p q : ℝ) := p / q

-- Given ratios provided in the problem:
def ratio_AB (D : Point) (A B : Point) : Prop := division_ratio (distance A D) (distance D B) = 1 / 3
def ratio_BC (E : Point) (B C : Point) : Prop := division_ratio (distance B E) (distance E C) = 2 / 1

-- Main theorem statement
theorem third_nail_division (A B C D E F : Point) (h_triangle : EquilateralTriangle A B C)
    (h_AB : ratio_AB D A B) (h_BC : ratio_BC E B C) :
  division_ratio (distance A F) (distance F C) = 5 / 7 := sorry

end third_nail_division_l88_88019


namespace scientific_notation_of_0_000136_l88_88251

theorem scientific_notation_of_0_000136 :
  (0.000136 : ℝ) = 1.36 * 10 ^ (-4) := 
by
  sorry

end scientific_notation_of_0_000136_l88_88251


namespace train_length_eq_100_l88_88762

theorem train_length_eq_100 (V_fast V_slow time_pass : ℝ) (L : ℝ)
  (H_eq_len : V_fast = 46 / 3.6 ∧ V_slow = 36 / 3.6)
  (H_time : time_pass = 72)
  (H_dist : 2 * L = (V_fast - V_slow) * time_pass) :
  L = 100 :=
begin
  sorry
end

end train_length_eq_100_l88_88762


namespace no_integer_with_300_ones_perfect_square_l88_88843

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def has_300_ones_and_zeros (n : ℕ) : Prop :=
  let digits := to_digits 10 n in
  (digits.count 1 = 300) ∧ ∀ digit ∈ digits, digit = 1 ∨ digit = 0

theorem no_integer_with_300_ones_perfect_square :
  ¬ ∃ n : ℕ, has_300_ones_and_zeros n ∧ is_perfect_square n :=
sorry

end no_integer_with_300_ones_perfect_square_l88_88843


namespace problem_inequality_l88_88633

theorem problem_inequality (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  -- proof here
  sorry

end problem_inequality_l88_88633


namespace compare_neg_one_neg_sqrt_two_l88_88832

theorem compare_neg_one_neg_sqrt_two : -1 > -Real.sqrt 2 :=
  by
    sorry

end compare_neg_one_neg_sqrt_two_l88_88832


namespace number_of_pairs_divisible_by_33_l88_88534

theorem number_of_pairs_divisible_by_33 :
  let m n : ℕ := 1
  in let condition := (1 ≤ m ∧ m ≤ n ∧ n ≤ 40)
  in let product_condition := (33 ∣ (m * n))
  in Σ (m n : ℕ), (1 ≤ m ∧ m ≤ n ∧ n ≤ 40) ∧ (33 ∣ (m * n)) → 64 :=
begin
  sorry
end

end number_of_pairs_divisible_by_33_l88_88534


namespace alice_bob_meet_l88_88394

/-- Alice and Bob play a game involving a circle whose circumference is divided by 15 equally-spaced points.
    Both start on point 15. Alice moves clockwise and Bob moves counterclockwise. In a turn of the game, 
    Alice moves 7 points clockwise and Bob moves 11 points counterclockwise. The game ends when they 
    stop on the same point. Prove that this occurs after 5 turns. -/
theorem alice_bob_meet (n : ℕ) (m : ℕ) (start : ℕ) (steps_a : ℕ) (steps_b : ℕ) (turns : ℕ) (p : ℕ) :
  alice_bob_meet 15 = start :=
(n = 15) ∧ (m = 15) ∧ (start = 15) ∧ (steps_a = 7) ∧ (steps_b = -11) ∧ (turns = 5) ∧ (p ≡ 0 [MOD n])
→ 3 * turns ≡ 0 [MOD n] :=
by
  have h : 7 - (-11) = 18 := sorry,
  have h1 : 18 ≡ 3 [MOD 15] := sorry,
  have h2 : 3 * turns ≡ 0 [MOD 15] := sorry,
  triv

end alice_bob_meet_l88_88394


namespace range_of_m_l88_88640

def f (x : ℝ) : ℝ := x^3 - (1 / 2) * x^2 - 2 * x + 5

def domain (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 2

theorem range_of_m (m : ℝ) : (∀ x : ℝ, domain x → f x < m) ↔ (7 < m) :=
by
  sorry

end range_of_m_l88_88640


namespace sequence_general_formula_l88_88775

variable {S : ℕ → ℝ} 

def a_n (n : ℕ) := match n with
                   | 0   => 0
                   | n+1 => S (n+1) - S n

theorem sequence_general_formula (S : ℕ → ℝ) (h : ∀ n, S n = (3 * (3^n + 1)) / 2) : 
  ∀ n, a_n S n = 3 ^ n :=
by
  sorry

end sequence_general_formula_l88_88775


namespace parallel_lines_chord_distance_l88_88261

theorem parallel_lines_chord_distance
    (r : ℝ) -- radius of the circle
    (d : ℝ) -- distance between two adjacent parallel lines
    (h1 : ∀ P, dist P O = r) -- all points P on the circle are at distance r from the center
    (h2 : 40^2 = 4 * r^2 - d^2) -- Stewart's theorem for the first chord
    (h3 : (40^2 - d^2) + (36^2 - (2 * d)^2) + (40^2 - d^2) = 14400) -- combined equation for three chords intersecting circle
    : d = 12 :=
sorry

end parallel_lines_chord_distance_l88_88261


namespace binom_20_10_eq_184756_l88_88908

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l88_88908


namespace spring_mass_at_length_l88_88158

def spring_length (x : ℝ) : ℝ := 16 + 2 * x

theorem spring_mass_at_length (y : ℝ) (h : y = 27) : ∃ x : ℝ, spring_length x = y ∧ x = 5.5 :=
by
  use 5.5
  simp [spring_length]
  rw [←h]
  norm_num
  sorry

end spring_mass_at_length_l88_88158


namespace proof_problem_theorem_l88_88582

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l88_88582


namespace second_method_larger_volume_l88_88260

noncomputable def volume_first_method : ℝ := 
  let r := 2.5 / (2 * Real.pi)
  in Real.pi * r^2 * 5

noncomputable def volume_second_method : ℝ := 
  let r := 5 / (2 * Real.pi)
  in Real.pi * r^2 * 2.5

theorem second_method_larger_volume :
  volume_first_method < volume_second_method :=
by
  sorry

end second_method_larger_volume_l88_88260


namespace total_distance_hiked_east_l88_88023

-- Define Annika's constant rate of hiking
def constant_rate : ℝ := 10 -- minutes per kilometer

-- Define already hiked distance
def distance_hiked : ℝ := 2.75 -- kilometers

-- Define total available time to return
def total_time : ℝ := 45 -- minutes

-- Prove that the total distance hiked east is 4.5 kilometers
theorem total_distance_hiked_east : distance_hiked + (total_time - distance_hiked * constant_rate) / constant_rate = 4.5 :=
by
  sorry

end total_distance_hiked_east_l88_88023


namespace sequence_general_term_l88_88169

def a : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 1) := a n + (n + 1) 

theorem sequence_general_term (n : ℕ) : 
  (∀ (m n : ℕ), a (m + n) = a m + a n + m * n) → 
  a n = n * (n + 1) / 2 := by
  sorry

end sequence_general_term_l88_88169


namespace D_double_prime_coordinates_l88_88210

-- A definition for the original points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the original points of the parallelogram
def A : Point2D := ⟨3, 6⟩
def B : Point2D := ⟨5, 10⟩
def C : Point2D := ⟨7, 6⟩
def D : Point2D := ⟨5, 2⟩

-- Reflect a point across the x-axis
def reflect_x (p : Point2D) : Point2D :=
  { p with y := -p.y }

-- Reflect a point across the line y = x + 2
def reflect_y_eq_x_plus_2 (p : Point2D) : Point2D :=
  let translated_p := { p with y := p.y - 2 }
  let reflected_p := { x := translated_p.y, y := translated_p.x }
  { reflected_p with y := reflected_p.y + 2 }

-- Perform the transformations and define the final result of D
def D'' := reflect_y_eq_x_plus_2 (reflect_x D)

-- The theorem to prove
theorem D_double_prime_coordinates : D'' = ⟨-4, 7⟩ :=
  sorry -- Proof goes here

end D_double_prime_coordinates_l88_88210


namespace birds_cannot_gather_on_one_tree_l88_88779

theorem birds_cannot_gather_on_one_tree :
  let trees := fin 6, -- 6 trees at the vertices of a regular hexagon
  let birds_initial := (fin 6 → fin 2), -- one bird on each tree
  (∀ t : trees, birds_initial t = 1) ∧
  (∀ move : (fin 6 × fin 6), ∃ (t1 t2 : fin 6), 
    (neighbors t1 t2) ∧ 
    (birds_initial t1 = 1) ∧ 
    (birds_initial t2 = 1) ∧ 
    (birds_after_move t1 t2 = update (λ x, if x = t1 ∨ x = t2 then 0 else birds_initial x))) → 
  ¬ ∃ tree : fin 6, ∀ t, birds_after_time t = if t = tree then 6 else 0 := 
begin
  sorry
end

end birds_cannot_gather_on_one_tree_l88_88779


namespace cannot_determine_passed_students_l88_88564

variable {M S E : ℝ} -- average marks for Math, Science, and English
variable {wM wS wE : ℝ} -- weights for Math, Science, and English
variable {N : ℕ} -- number of students
variable {P : ℝ} -- passing weighted average

def weighted_average (M S E wM wS wE : ℝ) : ℝ :=
  (M * wM) + (S * wS) + (E * wE)

theorem cannot_determine_passed_students 
    (hM : M = 46) (hS : S = 57) (hE : E = 65) 
    (hwM : wM = 0.4) (hwS : wS = 0.3) (hwE : 0.3)
    (hP : P = 60) :
    N = 150 → weighted_average M S E wM wS wE < P → false :=
by
  sorry

end cannot_determine_passed_students_l88_88564


namespace product_of_roots_of_quadratics_l88_88253

theorem product_of_roots_of_quadratics :
  ( ∏ k in range (2020 + 1), (2021 - k) / (2020 - k + 1) ) = 2021 :=
by
  sorry

end product_of_roots_of_quadratics_l88_88253


namespace polynomial_comparison_l88_88944

theorem polynomial_comparison {x : ℝ} :
  let A := (x - 3) * (x - 2)
  let B := (x + 1) * (x - 6)
  A > B :=
by 
  sorry -- Proof is omitted.

end polynomial_comparison_l88_88944


namespace compute_fraction_l88_88282

theorem compute_fraction :
  ((5 * 4) + 6) / 10 = 2.6 :=
by
  sorry

end compute_fraction_l88_88282


namespace smallest_positive_period_find_radius_of_circumcircle_l88_88935

noncomputable def f (x : ℝ) : ℝ :=
  let m : ℝ × ℝ := (2 * Real.cos x, 1)
  let n : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
  m.1 * n.1 + m.2 * n.2

def period_f (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x

def decreasing_interval (x : ℝ) (k : ℤ) : Prop :=
  let interval := (π/6 + k * π, 2 * π / 3 + k * π)
  interval.1 ≤ x ∧ x ≤ interval.2

theorem smallest_positive_period : period_f π ∧ (∀ k : ℤ, ∀ x : ℝ, decreasing_interval x k) :=
sorry

noncomputable def triangle_abc (A : ℝ) (b : ℝ) (area : ℝ) : ℝ :=
  let c := 2
  let a := Real.sqrt 3
  c / 2

theorem find_radius_of_circumcircle (A : ℝ) (b : ℝ) (area : ℝ) : triangle_abc A b area = 1 :=
sorry

end smallest_positive_period_find_radius_of_circumcircle_l88_88935


namespace smallest_positive_period_of_f_interval_where_f_is_monotonically_increasing_l88_88111

def f (x : ℝ) : ℝ := (sqrt 3 * sin x + cos x) * (sqrt 3 * cos x - sin x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, T' ≤ T → (∀ x, f (x + T') = f x) → T' = T) :=
sorry

theorem interval_where_f_is_monotonically_increasing (k : ℤ) :
  ∀ x, (k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12) ↔
       (∃ x1 x2, (k * π - 5 * π / 12 ≤ x1 ∧ x1 ≤ x ∧ x ≤ x2 ∧ x2 ≤ k * π + π / 12) ∧
       ∀ y, (x1 ≤ y ∧ y ≤ x2 → f y ≤ f (y + 1e-9)) =
sorry

end smallest_positive_period_of_f_interval_where_f_is_monotonically_increasing_l88_88111


namespace zoo_animal_count_l88_88392

def num_tiger_enclosures : ℕ := 4
def num_tigers_per_enclosure : ℕ := 4
def num_tiger_enclosure_groups : ℕ := num_tiger_enclosures / 2
def num_zebra_enclosures_per_group : ℕ := 3
def num_zebra_enclosures : ℕ := num_tiger_enclosure_groups * num_zebra_enclosures_per_group
def num_zebras_per_enclosure : ℕ := 10

def num_elephant_enclosure_patterns : ℕ := 4
def num_elephant_enclosures : ℕ := num_elephant_enclosure_patterns
def num_elephants_per_enclosure : ℕ := 3

def num_giraffe_enclosures_per_pattern : ℕ := 2
def num_giraffe_enclosures : ℕ := num_elephant_enclosure_patterns * num_giraffe_enclosures_per_pattern
def num_giraffes_per_enclosure : ℕ := 2

def num_rhino_enclosures : ℕ := 5
def num_rhinos_per_enclosure : ℕ := 1
def num_chimpanzee_enclosures_per_rhino : ℕ := 2
def num_chimpanzee_enclosures : ℕ := num_rhino_enclosures * num_chimpanzee_enclosures_per_rhino
def num_chimpanzees_per_enclosure : ℕ := 8

def total_tigers : ℕ := num_tiger_enclosures * num_tigers_per_enclosure
def total_zebras : ℕ := num_zebra_enclosures * num_zebras_per_enclosure
def total_elephants : ℕ := num_elephant_enclosures * num_elephants_per_enclosure
def total_giraffes : ℕ := num_giraffe_enclosures * num_giraffes_per_enclosure
def total_rhinos : ℕ := num_rhino_enclosures * num_rhinos_per_enclosure
def total_chimpanzees : ℕ := num_chimpanzee_enclosures * num_chimpanzees_per_enclosure

def total_animals : ℕ := total_tigers + total_zebras + total_elephants + total_giraffes + total_rhinos + total_chimpanzees

theorem zoo_animal_count : total_animals = 189 := by
  unfold total_animals total_tigers total_zebras total_elephants total_giraffes total_rhinos total_chimpanzees
  unfold num_tiger_enclosures num_tigers_per_enclosure num_zebra_enclosures num_zebras_per_enclosure
         num_elephant_enclosures num_elephants_per_enclosure num_giraffe_enclosures num_giraffes_per_enclosure
         num_rhino_enclosures num_rhinos_per_enclosure num_chimpanzee_enclosures num_chimpanzees_per_enclosure
  unfold num_tiger_enclosure_groups num_zebra_enclosures_per_group num_elephant_enclosure_patterns
        num_giraffe_enclosures_per_pattern num_chimpanzee_enclosures_per_rhino
  simp only [Nat.mul_add, Nat.add_mul, Nat.one_mul, Nat.div_add_div_same, Nat.mul_div_cancel', Nat.div_self, Nat.mul_comm]
  exact sorry

end zoo_animal_count_l88_88392


namespace range_of_g_on_interval_l88_88510

-- Defining the function g(x) = x^m
def g (x : ℝ) (m : ℝ) : ℝ := x ^ m

-- Defining the conditions
variable {m : ℝ}
variable {x : ℝ}
variable h1 : 0 < m
variable h2 : 0 < x
variable h3 : x ≤ 1

-- Theorem stating the range of g(x) on (0, 1] is (0, 1]
theorem range_of_g_on_interval (h1 : 0 < m) (h2 : 0 < x) (h3 : x ≤ 1) :
  ∃ y : Set.Icc (0 : ℝ) 1, g x m = y :=
sorry

end range_of_g_on_interval_l88_88510


namespace scientific_notation_l88_88106

-- Given radius of a water molecule
def radius_of_water_molecule := 0.00000000192

-- Required scientific notation
theorem scientific_notation : radius_of_water_molecule = 1.92 * 10 ^ (-9) :=
by
  sorry

end scientific_notation_l88_88106


namespace identify_fake_pearl_l88_88015

theorem identify_fake_pearl (pearls : Fin 9 → ℕ) (h_fake : ∃ i, pearls i = min (pearls 0) (pearls 1) (pearls 2) (pearls 3) (pearls 4) (pearls 5) (pearls 6) (pearls 7) (pearls 8)) : ∃ n : ℕ, n ≤ 2 ∧ identify_fake pearl n sorry := sorry

end identify_fake_pearl_l88_88015


namespace equal_angles_EFB_EBF_l88_88628

-- Define the given setup and conditions.
variables {A B C E F : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space E] [metric_space F] 

-- Define right triangle ABC with AB as the hypotenuse and BC as the diameter of a circle.
variables (ABC_right : ∠ A C B = 90)

-- Point E is on segment AB such that BE = EA.
variables (midpoint_E : dist B E = dist E A)

-- Line through E perpendicular to AB intersects BC at F.
variables (perpendicular_EF_AB : perpendicular (line_through E F) (line_through A B))

-- Goal: Prove that ∠ EFB = ∠ EBF.
theorem equal_angles_EFB_EBF : ∠ E F B = ∠ E B F :=
by sorry

end equal_angles_EFB_EBF_l88_88628


namespace probability_of_first_three_heads_l88_88298

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l88_88298


namespace dot_product_square_ABCD_l88_88598

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l88_88598


namespace max_value_6a_3b_10c_l88_88636

theorem max_value_6a_3b_10c (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 25 * c ^ 2 = 1) : 
  6 * a + 3 * b + 10 * c ≤ (Real.sqrt 41) / 2 :=
sorry

end max_value_6a_3b_10c_l88_88636


namespace angle_equality_l88_88622

open Real

structure Point2D (α : Type) :=
  (x : α)
  (y : α)

noncomputable def semicircle (A B : Point2D ℝ) : set (Point2D ℝ) :=
  {C | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧
       C.x = (A.x + B.x) / 2 + ((A.x - B.x) / 2) * cos θ ∧
       C.y = ((A.x - B.x) / 2) * sin θ}

noncomputable structure Triangle (α : Type) :=
  (A B P : Point2D α)

noncomputable def incircle (T : Triangle ℝ) : set (Point2D ℝ) :=
  {C | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * π ∧
       let R := (dist T.A T.B + dist T.B T.P + dist T.P T.A) / 2 in
       dist C T.A = dist C T.B ∧
       dist C T.B = dist C T.P ∧
       dist C T.P = R}

axiom angle_obtuse (T : Triangle ℝ) : Prop

noncomputable def tangent_points (T : Triangle ℝ) (IC : set (Point2D ℝ)) :
  Point2D ℝ × Point2D ℝ :=
  ({ x := 0, y := 0 }, { x := 0, y := 0 }) -- Placeholder values

noncomputable def line_intersect_semicircle
  (M N : Point2D ℝ) (semicircle : set (Point2D ℝ)) :
  Point2D ℝ × Point2D ℝ :=
  ({ x := 0, y := 0 }, { x := 0, y := 0 }) -- Placeholder values

theorem angle_equality
  (A B P : Point2D ℝ)
  (hP_in_semicircle : P ∈ semicircle A B)
  (hAPB_obtuse : angle_obtuse { A := A, B := B, P := P } )
  (incircle_of_triangle : set (Point2D ℝ) := incircle { A := A, B := B, P := P })
  (M N : Point2D ℝ := (tangent_points { A := A, B := B, P := P } incircle_of_triangle).fst,
   (tangent_points { A := A, B := B, P := P } incircle_of_triangle).snd)
  (X Y : Point2D ℝ := (line_intersect_semicircle M N (semicircle A B)).fst,
    (line_intersect_semicircle M N (semicircle A B)).snd) :
  -- Placeholder proof outline
  sorry

end angle_equality_l88_88622


namespace probability_three_heads_l88_88322

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l88_88322


namespace add_neg_eleven_results_in_geometric_sequence_l88_88495

theorem add_neg_eleven_results_in_geometric_sequence :
  ∃ x, x = -11 ∧
  (∀ {a1 a3 a4 a5 : ℤ} (d : ℤ),
  a1 = 2 ∧ a3 = 6 ∧ a1 + 2*d = a3 ∧ a4 = a1 + 3*d ∧ a5 = a1 + 4*d →
  let a1' := a1 + x,
      a4' := a4 + x,
      a5' := a5 + x in
  a4' * a4' = a1' * a5') :=
begin
  use -11,
  split,
  { refl },
  { intros a1 a3 a4 a5 d h,
    cases h with ha1 h1,
    rcases h1 with ⟨ha3, h2, h3, h4⟩,
    have d_eq : d = 2 := by linarith,
    simp [d_eq] at *,
    let a1' := a1 + -11,
    let a4' := a4 + -11,
    let a5' := a5 + -11,
    calc a4' * a4'
        = (a1 + 3*d + -11) * (a1 + 3*d + -11) : by refl
    ... = (2 + 3*2 + -11) * (2 + 3*2 + -11) : by simp [ha1, d_eq]
    ... = 8 * 8 : by linarith
    ... = 4 * 16 : by norm_num
    ... = (2 + -11) * (2 + 4*2 + -11) : by norm_num
    ... = a1' * a5' : by refl }

end add_neg_eleven_results_in_geometric_sequence_l88_88495


namespace N_def_M_intersection_CU_N_def_M_union_N_def_l88_88776

section Sets

variable {α : Type}

-- Declarations of conditions
def U := {x : ℝ | -3 ≤ x ∧ x ≤ 3}
def M := {x : ℝ | -1 < x ∧ x < 1}
def CU (N : Set ℝ) := {x : ℝ | 0 < x ∧ x < 2}

-- Problem statements
theorem N_def (N : Set ℝ) : N = {x : ℝ | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)} ↔ CU N = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

theorem M_intersection_CU_N_def (N : Set ℝ) : (M ∩ CU N) = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

theorem M_union_N_def (N : Set ℝ) : (M ∪ N) = {x : ℝ | (-3 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
by sorry

end Sets

end N_def_M_intersection_CU_N_def_M_union_N_def_l88_88776


namespace length_of_solution_set_l88_88424

def floor (x : ℝ) : ℤ := int.floor x
def f (x : ℝ) := (floor x) * (x - floor x)
def g (x : ℝ) := x - 1
def length_of_interval (a b : ℝ) := b - a

theorem length_of_solution_set :
  length_of_interval 1 2012 = 2011 := by
  sorry

end length_of_solution_set_l88_88424


namespace exists_large_prime_divisor_l88_88866

def factorial_sum (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum (λ k, k.fact)

def has_large_prime_divisor (n : ℕ) : Prop :=
  ∃ p : ℕ, nat.prime p ∧ p > 10^2012 ∧ p ∣ factorial_sum n

theorem exists_large_prime_divisor : ∃ n : ℕ, has_large_prime_divisor n :=
begin
  sorry
end

end exists_large_prime_divisor_l88_88866


namespace inequality_holds_l88_88197

theorem inequality_holds (a b c : ℝ) 
  (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_holds_l88_88197


namespace find_m_value_l88_88988

variables (a b : ℝ × ℝ) (m : ℝ)

-- Conditions
def vector_a := (1, 2) : ℝ × ℝ
def vector_b := (-3, 0) : ℝ × ℝ

-- Proving m value
theorem find_m_value (h1 : a = vector_a) (h2 : b = vector_b) 
  (h3 : 2 • a + b = (-1, 4)) (h4 : ∀ m : ℝ, a - m • b = (1 + 3 * m, 2)) : 
  m = -1 / 2 := 
sorry

end find_m_value_l88_88988


namespace a_saves_per_month_l88_88754

theorem a_saves_per_month (I_B S_B S_A : ℝ) 
  (income_ratio : I_A / I_B = 5 / 6)
  (expenditure_ratio : E_A / E_B = 3 / 4)
  (B_saves : S_B = 1600)
  (I_B_value : I_B = 7200) :
  let I_A := 5 / 6 * I_B
  let E_B := I_B - S_B
  let E_A := 3 / 4 * E_B
  let S_A := I_A - E_A
  in S_A = 1800 := 
by 
  sorry

end a_saves_per_month_l88_88754


namespace range_of_m_l88_88934

def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x / x) - m * x

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 0 < x → f x m > 0) ↔ m ∈ Set.Ioo (-(Float.infinity)) (Real.exp 2 / 4) :=
by
  sorry

end range_of_m_l88_88934


namespace ball_arrangements_l88_88859

theorem ball_arrangements (blue_balls red_balls : ℕ) (h_blue : blue_balls = 13) (h_red : red_balls = 5) 
  (h_condition : ∀ i, i < red_balls - 1 → 1 ≤ blue_balls) :
  (finset.card (finset.range (blue_balls + red_balls - 1)).powerset.choose (red_balls - 1) = 2002) := 
sorry

end ball_arrangements_l88_88859


namespace run_time_is_48_minutes_l88_88750

noncomputable def cycling_speed : ℚ := 5 / 2
noncomputable def running_speed : ℚ := cycling_speed * 0.5
noncomputable def walking_speed : ℚ := running_speed * 0.5

theorem run_time_is_48_minutes (d : ℚ) (h : (d / cycling_speed) + (d / walking_speed) = 2) : 
  (60 * d / running_speed) = 48 :=
by
  sorry

end run_time_is_48_minutes_l88_88750


namespace count_four_digit_integers_with_product_18_l88_88964

def valid_digits (n : ℕ) : Prop := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def digit_product_18 (a b c d : ℕ) : Prop := 
  a * b * c * d = 18

def four_digit_integer (a b c d : ℕ) : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d

theorem count_four_digit_integers_with_product_18 : 
  (∑ a b c d in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    ite (four_digit_integer a b c d ∧ digit_product_18 a b c d) 1 0) = 48 := 
sorry

end count_four_digit_integers_with_product_18_l88_88964


namespace total_red_peaches_l88_88727

theorem total_red_peaches (baskets : ℕ) (red_peaches_per_basket : ℕ) (hb : baskets = 6) (hr : red_peaches_per_basket = 16) : baskets * red_peaches_per_basket = 96 :=
by
  rw [hb, hr]
  norm_num

end total_red_peaches_l88_88727


namespace count_satisfying_integers_l88_88985

theorem count_satisfying_integers :
  {n : ℕ // 9 < n ∧ n < 65}.card = 55 := by
  sorry

end count_satisfying_integers_l88_88985


namespace cubic_identity_l88_88549

variable (a b c : ℝ)
variable (h1 : a + b + c = 13)
variable (h2 : ab + ac + bc = 30)

theorem cubic_identity : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 :=
by 
  sorry

end cubic_identity_l88_88549


namespace valves_filling_time_l88_88844

theorem valves_filling_time (a b c V : ℝ) (h_all_open : V / (a + b + c) = 2)
                           (h_A_and_C : V / (a + c) = 3)
                           (h_B_and_C : V / (b + c) = 4) :
                           V / (a + b) = 2.4 :=
by
  have h1 : a + b + c = V / 2 := by linarith
  have h2 : a + c = V / 3 := by linarith
  have h3 : b + c = V / 4 := by linarith
  have h_b : b = (V / 2) - (V / 3) := by linarith
  have h_c : c = (V / 4) - h_b := by linarith
  have h_a : a = (V / 3) - h_c := by linarith
  show V / (a + b) = 2.4, sorry

end valves_filling_time_l88_88844


namespace max_distance_l88_88338

variable (highway_mpg city_mpg gallons : ℝ)
variable (highway_mpg_value : highway_mpg = 12.2)
variable (city_mpg_value : city_mpg = 7.6)
variable (gallons_value : gallons = 20)

theorem max_distance (highway_mpg city_mpg gallons : ℝ) 
  (highway_mpg_value : highway_mpg = 12.2) 
  (city_mpg_value : city_mpg = 7.6)
  (gallons_value : gallons = 20) : 
  gallons * highway_mpg = 244 :=
by
  rw [highway_mpg_value, gallons_value]
  norm_num
  sorry

end max_distance_l88_88338


namespace probability_first_three_heads_l88_88302

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l88_88302


namespace total_money_l88_88800

theorem total_money (n : ℕ) (h1 : n * 3 = 36) :
  let one_rupee := n * 1
  let five_rupee := n * 5
  let ten_rupee := n * 10
  (one_rupee + five_rupee + ten_rupee) = 192 :=
by
  -- Note: The detailed calculations would go here in the proof
  -- Since we don't need to provide the proof, we add sorry to indicate the omitted part
  sorry

end total_money_l88_88800


namespace find_speeds_l88_88474

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l88_88474


namespace probability_three_heads_l88_88287

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l88_88287


namespace area_triangle_APB_l88_88389

variable (Square : Type) [metric_space Square]

-- Assumptions/Definitions based on conditions
variables {A B C D F P : Square}
variables (side_length : ℝ) (h_square : side_length = 6)
variables (PA PB PC : ℝ) (h_PA_eq_PB_eq_PC : PA = PB ∧ PB = PC)
variables (h_perpendicular : segment PC ⊥ FD)

-- Point E on AB such that AE = EB = 3
variables {E : Square} (h_AE_eq_EB : distance A E = 3 ∧ distance E B = 3)

-- Proof that given these conditions, the area of the triangle APB is 27/4 square inches
theorem area_triangle_APB : area (triangle A P B) = 27 / 4 :=
by sorry

end area_triangle_APB_l88_88389


namespace sum_first_n_sequence_a_correct_l88_88873

def sequence_a (n : ℕ) : ℕ →
  | 0 => 1
  | (n + 1) => sequence_a n + 2^n

def sum_first_n_sequence_a (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_a

theorem sum_first_n_sequence_a_correct (n : ℕ) : sum_first_n_sequence_a n = 2^(n + 1) - 1 :=
  sorry

end sum_first_n_sequence_a_correct_l88_88873


namespace max_airlines_l88_88558

theorem max_airlines (n : ℕ) (hn : n = 100) : ∃ k, k = 50 ∧ 
  (∀ (E : set (ℕ × ℕ)), (∀ (x y : ℕ), x < n ∧ y < n ∧ x ≠ y → (x, y) ∈ E ∨ (y, x) ∈ E) ∧
  (∀ (x y : ℕ), ∃ (p : list ℕ), path E x y p)) :=
sorry

end max_airlines_l88_88558


namespace carol_rectangle_width_l88_88829

theorem carol_rectangle_width 
  (area_jordan : ℕ) (length_jordan width_jordan : ℕ) (width_carol length_carol : ℕ)
  (h1 : length_jordan = 12)
  (h2 : width_jordan = 10)
  (h3 : width_carol = 24)
  (h4 : area_jordan = length_jordan * width_jordan)
  (h5 : area_jordan = length_carol * width_carol) :
  length_carol = 5 :=
by
  sorry

end carol_rectangle_width_l88_88829


namespace sum_of_variables_l88_88484

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : 
  x + y + z = 2 :=
sorry

end sum_of_variables_l88_88484


namespace OPRQ_is_rectangle_l88_88623

-- Definitions and conditions
variables (AB : ℝ) (O P Q R : Type)
variables (S1 S2 : Set Point) (C1 C2 : Set Point)
variables [MetricSpace O] [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables {AB_center : O.center = (AB / 2)}
variables {S1_center : O.center = AB / 2}
variables {C1_center : P.center}
variables {S2_center : Q.center}
variables {C2_center : R.center}

-- R1, r1, R2, r2 definitions
def R1 : ℝ := AB / 2
def r1 : ℝ := R1 / 2
def R2 : ℝ := R1 / 3
def r2 : ℝ := R1 / 6

-- Proof that OPRQ is a rectangle
theorem OPRQ_is_rectangle (h1 : Circle P r1 ∈ TangentTo S1) (h2 : Circle P r1 ∈ TangentTo O) :
  (Quadrilateral O P Q R) = Rectangle :=
sorry

end OPRQ_is_rectangle_l88_88623


namespace paintable_sum_is_453_l88_88122

-- For simplicity, let's define a function that determines if railings are painted exactly once
def railings_painted_once (h t u : ℕ) : Prop :=
  (∀ n : ℕ, ∃! x : ℕ, x ∈ {m | (m = 1 + h * n) ∨ (m = 4 + t * n) ∨ (m = 7 + u * n)})

noncomputable def paintable_sum : ℕ :=
  ∑ h t u in  {453| railings_painted_once h t u}, (100 * h + 10 * t + u)

theorem paintable_sum_is_453 : paintable_sum = 453 :=
  sorry

end paintable_sum_is_453_l88_88122


namespace set_subset_l88_88530

-- Define the sets M and N
def M := {x : ℝ | abs x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≤ 0}

-- The mathematical statement to be proved
theorem set_subset : N ⊆ M := sorry

end set_subset_l88_88530


namespace even_function_iff_orthogonal_l88_88096

variables {K : Type*} [Field K] (a b : K) (x : K)

def is_even : (x : K) → K := λ x, (a * x + b)^2

def orthogonal (a b : K) : Prop := a * b = 0

theorem even_function_iff_orthogonal (a b : K) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∀ x : K, is_even a b x = is_even a b (-x)) ↔ orthogonal a b :=
sorry

end even_function_iff_orthogonal_l88_88096


namespace factorize_grouping_decomposition_factorize_term_splitting_find_perimeter_of_triangle_l88_88272

theorem factorize_grouping_decomposition (x y : ℝ) :
  4*x^2 + 4*x - y^2 + 1 = (2*x + y + 1) * (2*x - y + 1) :=
by
  ring

theorem factorize_term_splitting (x : ℝ) :
  x^2 - 6*x + 8 = (x - 4) * (x - 2) :=
by
  ring

theorem find_perimeter_of_triangle (a b c : ℝ) (h : a^2 + b^2 + c^2 - 4*a - 4*b - 6*c + 17 = 0) :
  a = 2 ∧ b = 2 ∧ c = 3 ∧ a + b + c = 7 :=
by
  have ha : (a - 2)^2 = 0 := by 
    rw <-sub_eq_zero_iff_eq
    sorry
  have hb : (b - 2)^2 = 0 := by 
    rw <-sub_eq_zero_iff_eq
    sorry
  have hc : (c - 3)^2 = 0 := by 
    rw <-sub_eq_zero_iff_eq
    sorry
  use (ha, hb, hc)
  rw [ha, hb, hc]
  ring
  sorry

end factorize_grouping_decomposition_factorize_term_splitting_find_perimeter_of_triangle_l88_88272


namespace tank_capacity_l88_88367

theorem tank_capacity (x : ℝ) (h₁ : (3/4) * x = (1/3) * x + 18) : x = 43.2 := sorry

end tank_capacity_l88_88367


namespace quarters_percentage_is_65_22_l88_88752

-- Given constants for the conditions
def num_nickels := 80
def num_quarters := 30

-- Definitions for their values in cents
def value_of_quarters := num_quarters * 25
def value_of_nickels := num_nickels * 5
def total_value := value_of_quarters + value_of_nickels

-- The proof statement of the percentage of the value in quarters.
theorem quarters_percentage_is_65_22 : 
  (value_of_quarters / total_value : ℝ) * 100 ≈ 65.22 := 
by
  sorry

end quarters_percentage_is_65_22_l88_88752


namespace solve_for_phi_l88_88145

theorem solve_for_phi (φ : ℝ) (h1 : abs φ < π / 2)
  (h2 : ∃ x y, x = π / 12 ∧ y = -sqrt 2 ∧ 
              y = 2 * sin (2 * (x - π / 6) + φ)) :
  φ = -π / 12 :=
by
  sorry

end solve_for_phi_l88_88145


namespace find_f_quarter_and_solve_f_neg_x_l88_88940

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^x else real.log 2 x

-- The Lean theorem statement without providing proof:

theorem find_f_quarter_and_solve_f_neg_x : 
  f (1 / 4) = -2 ∧ ((f (-1)) = (1 / 2) ∧ (f (-(-sqrt 2))) = (1 / 2)) :=
by 
  sorry

end find_f_quarter_and_solve_f_neg_x_l88_88940


namespace probability_first_three_heads_l88_88299

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l88_88299


namespace base_11_correct_l88_88838

noncomputable def char_to_value (ch : Char) : Nat :=
  if h : '0' ≤ ch ∧ ch ≤ '9' then ch.toNat - '0'.toNat else 
    if h : 'A' ≤ ch ∧ ch ≤ 'Z' then ch.toNat - 'A'.toNat + 10 else 0

noncomputable def base_to_nat (str : String) (base : Nat) : Nat :=
  str.toList.foldl (fun acc ch => acc * base + char_to_value ch) 0

theorem base_11_correct :
  ∀ a : Nat, a = 11 →
  base_to_nat "356" a + base_to_nat "791" a = base_to_nat "C42" a :=
by
  intros a ha
  have h : a = 11 := ha
  rw [h]
  simp
  sorry

end base_11_correct_l88_88838


namespace complex_number_z_l88_88077

theorem complex_number_z (z : ℂ) (i : ℂ) (hz : i^2 = -1) (h : (1 - i)^2 / z = 1 + i) : z = -1 - i :=
by
  sorry

end complex_number_z_l88_88077


namespace part1_part2_l88_88200

variables (a b c x x₁ x₂ x₀ : ℝ)

noncomputable def f (x : ℝ) := a * x^2 + b * x + c

variable (h1 : a > 0)
variable (h2 : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / a)
variable (h3 : f x₁ = x₁ ∧ f x₂ = x₂)
variable (h4 : x₀ = -b / (2 * a))

theorem part1 (hx : 0 < x ∧ x < x₁) : x < f x ∧ f x < x₁ := sorry

theorem part2 : x₀ < x₁ / 2 := sorry

end part1_part2_l88_88200


namespace calculate_expression_l88_88825

theorem calculate_expression : ( (3 / 20 + 5 / 200 + 7 / 2000) * 2 = 0.357 ) :=
by
  sorry

end calculate_expression_l88_88825


namespace part1_solution_l88_88850

theorem part1_solution : ∀ n : ℕ, ∃ k : ℤ, 2^n + 3 = k^2 ↔ n = 0 :=
by sorry

end part1_solution_l88_88850


namespace internship_assignment_plans_l88_88726

-- Define number of students and companies
def num_students : ℕ := 5
def num_companies : ℕ := 3

-- State the theorem that the number of different possible assignment plans is 150
theorem internship_assignment_plans : 
  (Σ (s : FiniteType (Fin num_companies) num_students), 
    ∀ i : Fin num_companies, ∃ j : Fin num_students, j ∈ s i) = 150 := by
  sorry

end internship_assignment_plans_l88_88726


namespace isabella_jumped_farthest_l88_88669

-- defining the jumping distances
def ricciana_jump : ℕ := 4
def margarita_jump : ℕ := 2 * ricciana_jump - 1
def isabella_jump : ℕ := ricciana_jump + 3 

-- defining the total distances
def ricciana_total : ℕ := 20 + ricciana_jump
def margarita_total : ℕ := 18 + margarita_jump
def isabella_total : ℕ := 22 + isabella_jump

-- stating the theorem
theorem isabella_jumped_farthest : isabella_total = 29 :=
by sorry

end isabella_jumped_farthest_l88_88669


namespace molecular_weight_CaO_is_56_l88_88279

def atomic_weight_Ca : ℕ := 40
def atomic_weight_O : ℕ := 16
def molecular_weight_CaO : ℕ := atomic_weight_Ca + atomic_weight_O

theorem molecular_weight_CaO_is_56 :
  molecular_weight_CaO = 56 := by
  sorry

end molecular_weight_CaO_is_56_l88_88279


namespace number_of_nonzero_terms_in_polynomial_is_4_l88_88053

def polynomial := (2*x + 5) * (3*x^2 + 4*x + 8) - 4*(x^3 - x^2 + 5*x + 2)

theorem number_of_nonzero_terms_in_polynomial_is_4 : 
  let p := polynomial
  ∃ p1 p2 p3 p4 : ℕ, p = (λ _ => 2 * x ^ 3 + 27 * x ^ 2 + 16 * x + 32) ∧ p1 ≠ 0 ∧ p2 ≠ 0 ∧ p3 ≠ 0 ∧ p4 ≠ 0 :=
sorry

end number_of_nonzero_terms_in_polynomial_is_4_l88_88053


namespace proof_problem_theorem_l88_88583

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l88_88583


namespace distance_after_20_seconds_l88_88266

noncomputable def velocity1 (t : ℝ) : ℝ := 5 * t
noncomputable def velocity2 (t : ℝ) : ℝ := 3 * t^2

noncomputable def position1 (t : ℝ) : ℝ := ∫ x in 0 .. t, velocity1 x
noncomputable def position2 (t : ℝ) : ℝ := ∫ x in 0 .. t, velocity2 x

theorem distance_after_20_seconds : 
  let s1 := position1 20;
      s2 := position2 20
  in s2 - s1 = 7000 :=
by
  sorry

end distance_after_20_seconds_l88_88266


namespace distinct_four_digit_integers_with_product_18_l88_88973

theorem distinct_four_digit_integers_with_product_18 :
  ∃ n : ℕ, n = 24 ∧ ∀ (d1 d2 d3 d4 : ℕ), (d1 * d2 * d3 * d4 = 18 ∧ 1000 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000) →
    set.finite { x | ∃ (d1 d2 d3 d4 : ℕ), x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ d1 * d2 * d3 * d4 = 18 ∧ ∀ i ∈ [d1, d2, d3, d4], 1 ≤ i ∧ i ≤ 9 } :=
begin
  sorry
end

end distinct_four_digit_integers_with_product_18_l88_88973


namespace prime_factorization_count_l88_88836

theorem prime_factorization_count :
  (∀ (x : ℕ), (x = 86 → x = 2 * 43) ∧
              (x = 88 → x = 2^3 * 11) ∧
              (x = 90 → x = 2 * 3^2 * 5) ∧
              (x = 92 → x = 2^2 * 23))
  → (number_of_different_primes_in_factorization (86 * 88 * 90 * 92) = 6) :=
by
  sorry

end prime_factorization_count_l88_88836


namespace find_platform_length_l88_88349

theorem find_platform_length
  (train_length : ℝ)
  (time_signal_pole : ℝ)
  (time_platform : ℝ)
  (train_speed : ℝ) 
  (V : ℝ := train_length / time_signal_pole)
  (platform_length : ℝ := (V * time_platform - train_length)) :
  train_length = 300 →
  time_signal_pole = 18 →
  time_platform = 33 →
  train_speed = 50/3 →
  platform_length = 250 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  change 300 / 18 * 33 - 300 = 250
  norm_num
  exact rfl

end find_platform_length_l88_88349


namespace probability_winning_prize_l88_88366

theorem probability_winning_prize :
  let cards := {1, 2, 3}  -- Represent the 3 types of cards
  let bags := {1, 2, 3, 4}  -- Represent 4 bags of food
  -- Total number of ways to distribute 3 types of cards into 4 bags
  let total_ways := 3^4
  -- Number of ways to distribute at least one of each type of card into 4 bags 
  let favorable_ways := 3 * Nat.choose 4 2 * Nat.choose 2 2
  -- The probability of winning a prize
  let probability := favorable_ways / total_ways in
  probability = 4 / 9 :=
by
  sorry

end probability_winning_prize_l88_88366


namespace impossible_to_remove_all_checkers_l88_88159

-- Define the structure and properties of the board
def board := List (List ℕ) -- A board represented as a list of list of natural numbers

-- Initial board setup: a 5x10 board with one checker in each cell
def initial_board : board := List.replicate 5 (List.replicate 10 1)

-- Define the structure of a move: (x1, y1) and (x2, y2) are the coordinates of the cells
structure move :=
  (x1 y1 x2 y2 : ℕ)

-- Function to check if a move is valid (moving to adjacent cells)
def move_is_valid (m : move) : Prop :=
  ((m.x1 = m.x2) ∧ (m.y1 = m.y2 + 1 ∨ m.y1 = m.y2 - 1)) ∨ 
  ((m.y1 = m.y2) ∧ (m.x1 = m.x2 + 1 ∨ m.x1 = m.x2 - 1))

-- Function to remove two checkers from any cell with two or more checkers
def remove_two_checkers (b : board) (x y : ℕ) : board :=
  if b.nth x >>= fun row => row.nth y < 2 then b
  else b.modify_nth x (fun row => row.modify_nth y (fun n => n - 2))

-- Proposition formulation in Lean 4
theorem impossible_to_remove_all_checkers :
  ¬∃ moves : List move, ∀ m ∈ moves, move_is_valid m ∧
    (let final_board := List.foldl (λ b m, remove_two_checkers (place_checkers b m) m.x1 m.y1.m.x2 m.y2) initial_board moves in
     ∀ row ∈ final_board, ∀ cell ∈ row, cell = 0):
sorry

end impossible_to_remove_all_checkers_l88_88159


namespace square_of_larger_number_is_1156_l88_88250

theorem square_of_larger_number_is_1156
  (x y : ℕ)
  (h1 : x + y = 60)
  (h2 : x - y = 8) :
  x^2 = 1156 := by
  sorry

end square_of_larger_number_is_1156_l88_88250


namespace number_of_correct_conclusions_l88_88668

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 - (2/3)^(Real.abs x) + 1/2

theorem number_of_correct_conclusions :
  let conclusions : Fin 4 → Bool := fun i =>
    match i with
    | ⟨0, _⟩ => False         -- Conclusion 1 is wrong
    | ⟨1, _⟩ => False         -- Conclusion 2 is wrong
    | ⟨2, _⟩ => False         -- Conclusion 3 is wrong
    | ⟨3, _⟩ => True          -- Conclusion 4 is correct
  ∑ i, if conclusions i then 1 else 0 = 1 :=
by
  sorry

end number_of_correct_conclusions_l88_88668


namespace diagonal_of_square_plot_l88_88233

theorem diagonal_of_square_plot (l b : ℝ) (hl : l = 90) (hb : b = 80) :
  let area_rect := l * b in
  let area_square := area_rect in
  let side_square := real.sqrt area_square in
  let diag_square := real.sqrt (2 * (side_square ^ 2)) in
  diag_square = 120 :=
by
  sorry

end diagonal_of_square_plot_l88_88233


namespace EF_bisects_AC_l88_88075

theorem EF_bisects_AC 
  (A B C K E F : Point) 
  (ABC : Triangle A B C) 
  (h1 : right_angle A B C)
  (h2 : altitude C K B)
  (h3 : angle_bisector A C K E)
  (h4 : parallel (line B F) (line C E)) 
  (h5 : intersect (line CK) (line BF) F):
  midpoint F A C :=
sorry

end EF_bisects_AC_l88_88075


namespace rectangular_board_area_l88_88383

variable (length width : ℕ)

theorem rectangular_board_area
  (h1 : length = 2 * width)
  (h2 : 2 * length + 2 * width = 84) :
  length * width = 392 := 
by
  sorry

end rectangular_board_area_l88_88383


namespace max_k_preserving_product_l88_88953

theorem max_k_preserving_product (
    a : ℕ → ℤ) 
    (h_distinct : ∀ i j : ℕ, i ≠ j → a i ≠ a j) 
    (h_len : ∃ i :ℕ, i < 100) :
  ∃ k : ℕ, (k = 99) ∧ (∀ i: ℕ, 0 ≤ i ∧  i ≤ 99 → ∏ (j : ℕ) in (finset.range 100), (a j + i) = ∏ (j : ℕ) in (finset.range 100), (a j)) :=
begin
    sorry
end

end max_k_preserving_product_l88_88953


namespace period_of_f_g_is_2_sin_x_g_is_odd_l88_88520

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 3)

-- Theorem 1: Prove that f has period 2π.
theorem period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

-- Define g and prove the related properties.
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 3)

-- Theorem 2: Prove that g(x) = 2 * sin x.
theorem g_is_2_sin_x : ∀ x : ℝ, g x = 2 * Real.sin x := by
  sorry

-- Theorem 3: Prove that g is an odd function.
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end period_of_f_g_is_2_sin_x_g_is_odd_l88_88520


namespace speed_of_current_eq_l88_88373

theorem speed_of_current_eq :
  ∃ (m c : ℝ), (m + c = 15) ∧ (m - c = 8.6) ∧ (c = 3.2) :=
by
  sorry

end speed_of_current_eq_l88_88373


namespace num_even_rows_in_pascals_triangle_l88_88419

theorem num_even_rows_in_pascals_triangle : 
  let rows_with_only_even_except_ends (n : ℕ) := ∃ k : ℕ, n = 2^k ∧ k < 5
  in ∃ (count : ℕ), count = 4 ∧ 
    (∀ r : ℕ, r < 30 ∧ r ≠ 0 ∧ r ≠ 1 → rows_with_only_even_except_ends r -> r ∈ [2, 4, 8, 16]) := 
by {
  sorry
}

end num_even_rows_in_pascals_triangle_l88_88419


namespace find_f_prime_zero_l88_88167

noncomputable def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

def f (x : ℝ) (a : Fin 8 → ℝ) : ℝ := x * ∏ i in Finset.finRange 8, (x - a i)

theorem find_f_prime_zero :
  ∀ (a₁ a₈ : ℝ),
  a₁ = 2 →
  a₈ = 4 →
  ∃ (r : ℝ), r ^ 7 = 2 ∧
  ∀ (a : Fin 8 → ℝ),
  (∀ n, a n = geometric_sequence a₁ r (n + 1)) →
  f' (f 0 a) = 2 ^ 12 :=
sorry

end find_f_prime_zero_l88_88167


namespace find_speeds_l88_88478

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l88_88478


namespace max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l88_88899

open Real

noncomputable def max_value_b_minus_inv_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
b - (1 / a)

noncomputable def min_value_inv_3a_plus_1_plus_inv_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
(1 / (3 * a + 1)) + (1 / (a + b))

theorem max_value_b_minus_inv_a_is_minus_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  max_value_b_minus_inv_a a b ha hb h = -1 :=
sorry

theorem min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  min_value_inv_3a_plus_1_plus_inv_a_plus_b a b ha hb h = 1 :=
sorry

end max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l88_88899


namespace slope_angle_of_chord_l88_88778

-- Define the 2D point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the parametric equations for the circle
def circle_parametric (θ : ℝ) : Point :=
  { x := 1 + 5 * Real.cos θ, y := 5 * Real.sin θ }

-- The given center of the circle
def A : Point := { x := 1, y := 0 }

-- The given midpoint of the chord
def P : Point := { x := 2, y := -1 }

-- The slope function
def slope (P1 P2 : Point) : ℝ :=
  (P2.y - P1.y) / (P2.x - P1.x)

-- Definition of the problem that needs to be proved
theorem slope_angle_of_chord : 
  ∀ θ1 θ2 : ℝ, 0 ≤ θ1 ∧ θ1 < 2*Real.pi ∧ 0 ≤ θ2 ∧ θ2 < 2*Real.pi → 
  P = { x := (circle_parametric θ1).x + (circle_parametric θ2).x / 2, 
        y := (circle_parametric θ1).y + (circle_parametric θ2).y / 2 } →
  angle (slope A P) (slope (circle_parametric θ1) (circle_parametric θ2)) = Real.pi / 4 :=
begin
  sorry
end

end slope_angle_of_chord_l88_88778


namespace find_k_l88_88798

theorem find_k (k : ℚ) : 
  ((3, -8) ≠ (k, 20)) ∧ 
  (∃ m, (4 * m = -3) ∧ (20 - (-8) = m * (k - 3))) → 
  k = -103/3 := 
by
  sorry

end find_k_l88_88798


namespace find_speeds_l88_88472

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l88_88472


namespace greatest_n_dividing_factorial_l88_88144

theorem greatest_n_dividing_factorial (n : ℕ) (m : ℕ) (h1 : m = 3 ^ n) (h2 : m ∣ nat.factorial 19) : n = 8 := 
sorry

end greatest_n_dividing_factorial_l88_88144


namespace tank_fill_time_l88_88390

theorem tank_fill_time (h_a : ℝ)
(h_d : ℝ) : 1 / ((1 / 20) + (3 / 10)) ≈ 2.86 :=
by {
  sorry
}

end tank_fill_time_l88_88390


namespace greatest_n_for_xy_le_0_l88_88185

theorem greatest_n_for_xy_le_0
  (a b : ℕ) (coprime_ab : Nat.gcd a b = 1) :
  ∃ n : ℕ, (n = a * b ∧ ∃ x y : ℤ, n = a * x + b * y ∧ x * y ≤ 0) :=
sorry

end greatest_n_for_xy_le_0_l88_88185


namespace line_intersects_circle_l88_88490

theorem line_intersects_circle
  (a b r : ℝ)
  (r_nonzero : r ≠ 0)
  (h_outside : a^2 + b^2 > r^2) :
  ∃ x y : ℝ, (x^2 + y^2 = r^2) ∧ (a * x + b * y = r^2) :=
sorry

end line_intersects_circle_l88_88490


namespace max_term_in_sequence_l88_88529

theorem max_term_in_sequence :
  ∀ (a_n : ℕ → ℝ), (a_n = λ n, n / (n^2 + 196)) → ∃ (n : ℕ), n = 14 ∧
    ∀ (m : ℕ), a_n m ≤ a_n n :=
by sorry

end max_term_in_sequence_l88_88529


namespace solve_for_x_l88_88677

theorem solve_for_x :
  ∀ x : ℝ, 4 * x + 9 * x = 360 - 9 * (x - 4) → x = 18 :=
by
  intros x h
  sorry

end solve_for_x_l88_88677


namespace range_of_a_l88_88522

def f (x : ℝ) : ℝ := 1 / exp (abs x) - x^2

theorem range_of_a (a : ℝ) :
  (f (3^(a-1)) > f (-1 / 9)) → a < -1 :=
by
  sorry

end range_of_a_l88_88522


namespace find_speeds_l88_88480

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l88_88480


namespace markese_earnings_l88_88650

-- Define the conditions
def earnings_relation (E M : ℕ) : Prop :=
  M = E - 5 ∧ M + E = 37

-- The theorem to prove
theorem markese_earnings (E M : ℕ) (h : earnings_relation E M) : M = 16 :=
by
  sorry

end markese_earnings_l88_88650


namespace pitcher_juice_percentage_l88_88002

noncomputable def juice_percentage_per_cup (C : ℝ) : ℝ := ((5 / 6 * C) / 3) / C * 100

theorem pitcher_juice_percentage :
  ∀ (C : ℝ), C > 0 → juice_percentage_per_cup C = 27.78 :=
by
  intro C hC
  have eq1 : juice_percentage_per_cup C = ((5 / 6 * C) / 3) / C * 100 := by rfl
  have eq2 : ((5 / 6 * C) / 3) / C * 100 = (5 / 18) * 100 := by
    field_simp [hC]
    ring
  have eq3 : (5 / 18) * 100 = 27.78 := by
    norm_num
  rw [eq1, eq2, eq3]
  sorry

end pitcher_juice_percentage_l88_88002


namespace find_ellipse_equation_find_line_equation_l88_88895

theorem find_ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (e : ℝ) (h3 : e = real.sqrt 2 / 2)
(c : ℝ) (h4 : c = 1) :
  let x2 := x^2 / (a^2),
      y2 := y^2 / (b^2)
  in x2 + y2 = 1 → (a = real.sqrt 2) → (b = 1)
  → ( ∀ (x y : ℝ), x^2 / 2 + y^2 = 1 ) := sorry

theorem find_line_equation (l : ℝ → ℝ) 
  (h1 : ∀ x, l x = x + 1 ∨ l x = x - 1) 
  (A B : ℝ × ℝ) 
  (h2 : 3 * A.1^2 + 4 * l(A.1) * A.1 + 2 * (l(A.1))^2 - 2 = 0 ∧ 
        3 * B.1^2 + 4 * l(B.1) * B.1 + 2 * (l(B.1))^2 - 2 = 0)
  (h3 : abs ((A.1 - B.1) * (l(A.1) - l(B.1))) = 4 * real.sqrt(2) / 3) :
  l (A.1) = A.2 ∧ l (B.1) = B.2 :=
sorry

end find_ellipse_equation_find_line_equation_l88_88895


namespace sqrt_x_minus_2_meaningful_l88_88284

theorem sqrt_x_minus_2_meaningful (x : ℝ) : (√(x - 2)).isReal ↔ x ≥ 2 := 
by sorry

end sqrt_x_minus_2_meaningful_l88_88284


namespace binom_20_10_l88_88917

-- Definitions for the provided conditions
def binom_18_8 := 43758
def binom_18_9 := 48620
def binom_18_10 := 43758

-- The theorem we need to prove
theorem binom_20_10 : ∀
  (binom_18_8 = 43758)
  (binom_18_9 = 48620)
  (binom_18_10 = 43758),
  binomial 20 10 = 184756 :=
by
  sorry

end binom_20_10_l88_88917


namespace distinct_four_digit_integers_with_product_18_l88_88969

theorem distinct_four_digit_integers_with_product_18 :
  ∃ n : ℕ, n = 24 ∧ ∀ (d1 d2 d3 d4 : ℕ), (d1 * d2 * d3 * d4 = 18 ∧ 1000 ≤ 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000) →
    set.finite { x | ∃ (d1 d2 d3 d4 : ℕ), x = 1000 * d1 + 100 * d2 + 10 * d3 + d4 ∧ d1 * d2 * d3 * d4 = 18 ∧ ∀ i ∈ [d1, d2, d3, d4], 1 ≤ i ∧ i ≤ 9 } :=
begin
  sorry
end

end distinct_four_digit_integers_with_product_18_l88_88969


namespace sequence_sum_60_l88_88170

theorem sequence_sum_60 :
  ∃ (a : ℕ → ℝ) (S : ℕ → ℝ),
    a 1 = 1 ∧
    (∀ n, a (n+2) + (-1)^n * a n = 2) ∧
    S 60 = (∑ i in finset.range 60, a (i+1)) → 
    S 60 = 930 :=
sorry

end sequence_sum_60_l88_88170


namespace determine_value_of_m_l88_88237

noncomputable def conics_same_foci (m : ℝ) : Prop :=
  let c1 := Real.sqrt (4 - m^2)
  let c2 := Real.sqrt (m + 2)
  (∀ (x y : ℝ),
    (x^2 / 4 + y^2 / m^2 = 1) → (x^2 / m - y^2 / 2 = 1) → c1 = c2) → 
  m = 1

theorem determine_value_of_m : ∃ (m : ℝ), conics_same_foci m :=
sorry

end determine_value_of_m_l88_88237


namespace distance_to_school_l88_88614

def usual_time := 25 / 60 -- 25 minutes in hours
def lighter_time := 10 / 60 -- 10 minutes in hours
def speed_increase := 15 -- mph increase during lighter traffic

theorem distance_to_school (v : ℝ) (d : ℝ) 
  (h1 : d = v * usual_time) 
  (h2 : d = (v + speed_increase) * lighter_time) :
  d = 1.6 :=
by
  let t₁ := 25 / 60
  let t₂ := 10 / 60
  have h : v * t₁ = (v + 15) * t₂ := h1.trans h2.symm
  sorry

end distance_to_school_l88_88614


namespace municipal_department_min_employees_l88_88802

theorem municipal_department_min_employees 
  (U P UCAP : ℕ)
  (hU : U = 120)
  (hP : P = 98)
  (hUCAP : UCAP = 40) : 
  (U + P - UCAP + UCAP = 218) :=
by
  -- Definitions
  let required_employees := U + P - UCAP
  -- Assertion
  have min_employees : required_employees + UCAP = 218
  sorry

end municipal_department_min_employees_l88_88802


namespace range_of_x_l88_88511

variable {f : ℝ → ℝ}

-- Define the conditions for the function
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom decreasing_on_nonneg : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_x (h : ∀ x : ℝ, f (Real.log x / Real.log 10) > f 1) :
  ∀ x : ℝ, Real.log ⁻¹'.range (f ∘ Real.log) ⊆ Ioo (1/10) 10 :=
sorry

end range_of_x_l88_88511


namespace sum_of_inradii_of_divided_triangles_greater_than_initial_radius_l88_88803

variable {n : ℕ}
variable {r : ℝ}
variable {P S : ℝ}
variable {P_i S_i : Fin n → ℝ}
variable {r_i : Fin n → ℝ}

-- Definitions based on the problem conditions
def isCircumscribedPolygon (r : ℝ) (P : ℝ) (S : ℝ) : Prop :=
  ∃ (circle_radius : ℝ), circle_radius = r ∧ circle_radius = 2 * S / P

def isDividedIntoTriangles (P_i S_i : Fin n → ℝ) (P S : ℝ) : Prop :=
  (∀ i, P_i i < P) ∧ ∀ i, r_i i = 2 * S_i i / P_i i

theorem sum_of_inradii_of_divided_triangles_greater_than_initial_radius
  (h_pol : isCircumscribedPolygon r P S)
  (h_tris : isDividedIntoTriangles P_i S_i P S) :
  (∑ i, r_i i) > r :=
sorry

end sum_of_inradii_of_divided_triangles_greater_than_initial_radius_l88_88803


namespace point_on_inverse_graph_and_sum_l88_88226

-- Definitions
variable (f : ℝ → ℝ)
variable (h : f 2 = 6)

-- Theorem statement
theorem point_on_inverse_graph_and_sum (hf : ∀ x, x = 2 → 3 = (f x) / 2) :
  (6, 1 / 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, (f⁻¹ x) / 2)} ∧
  (6 + (1 / 2) = 13 / 2) :=
by
  sorry

end point_on_inverse_graph_and_sum_l88_88226


namespace find_speeds_l88_88464

noncomputable def speed_pedestrian := 5
noncomputable def speed_cyclist := 30

def distance_AB := 40
def starting_time_pedestrian := 4 -- In hours (24-hour format)
def starting_time_cyclist_1 := 7 + 20 / 60 -- 7:20 AM in hours
def halfway_distance := distance_AB / 2
def midpoint_meeting_time := 1 -- Time (in hours) after the first meeting
def starting_time_cyclist_2 := 8 + 30 / 60 -- 8:30 AM in hours

theorem find_speeds (x y : ℝ) (hx : x = speed_pedestrian) (hy : y = speed_cyclist) :
  let time_to_halfway := halfway_distance / x in
  let cyclist_time := (midpoint_meeting_time + time_to_halfway) in
  distance_AB = 
    cyclist_time * y + 
    time_to_halfway * x + 
    (midpoint_meeting_time - 1) * x :=
    x = speed_pedestrian ∧ y = speed_cyclist :=
begin
  sorry
end

end find_speeds_l88_88464


namespace sum_seq_100_l88_88947

def seq := ℕ → ℕ

def conditions (a : seq) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧
  (∀ n, a n * a (n + 1) * a (n + 2) ≠ 1) ∧
  (∀ n, a n * a (n + 1) * a (n + 2) * a (n + 3) = 
       a n + a (n + 1) + a (n + 2) + a (n + 3))

theorem sum_seq_100 (a : seq) (h : conditions a) :
  (Finset.range 100).sum (λ n, a (n + 1)) = 200 :=
sorry

end sum_seq_100_l88_88947


namespace masha_lives_on_seventh_floor_l88_88656

/-- Masha lives in apartment No. 290, which is in the 4th entrance of a 17-story building.
The number of apartments is the same in all entrances of the building on all 17 floors; apartment numbers start from 1.
We need to prove that Masha lives on the 7th floor. -/
theorem masha_lives_on_seventh_floor 
  (n_apartments_per_floor : ℕ) 
  (total_floors : ℕ := 17) 
  (entrances : ℕ := 4) 
  (masha_apartment : ℕ := 290) 
  (start_apartment : ℕ := 1) 
  (h1 : (masha_apartment - start_apartment + 1) > 0) 
  (h2 : masha_apartment ≤ entrances * total_floors * n_apartments_per_floor)
  (h4 : masha_apartment > (entrances - 1) * total_floors * n_apartments_per_floor)  
   : ((masha_apartment - ((entrances - 1) * total_floors * n_apartments_per_floor) - 1) / n_apartments_per_floor) + 1 = 7 := 
by
  sorry

end masha_lives_on_seventh_floor_l88_88656


namespace line_intersects_ellipse_l88_88087

theorem line_intersects_ellipse
  (m : ℝ) :
  ∃ P : ℝ × ℝ, P = (3, 2) ∧ ((m + 2) * P.1 - (m + 4) * P.2 + 2 - m = 0) ∧ 
  (P.1^2 / 25 + P.2^2 / 9 < 1) :=
by 
  sorry

end line_intersects_ellipse_l88_88087


namespace rabbit_speed_l88_88057

theorem rabbit_speed (dog_speed : ℝ) (head_start : ℝ) (catch_time_minutes : ℝ) 
  (H1 : dog_speed = 24) (H2 : head_start = 0.6) (H3 : catch_time_minutes = 4) :
  let catch_time_hours := catch_time_minutes / 60
  let distance_dog_runs := dog_speed * catch_time_hours
  let distance_rabbit_runs := distance_dog_runs - head_start
  let rabbit_speed := distance_rabbit_runs / catch_time_hours
  rabbit_speed = 15 :=
  sorry

end rabbit_speed_l88_88057


namespace pages_after_break_correct_l88_88180

-- Definitions based on conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.7
def pages_before_break : ℕ := (total_pages : ℝ * break_percentage).to_nat
def pages_after_break : ℕ := total_pages - pages_before_break

-- Theorem statement
theorem pages_after_break_correct : pages_after_break = 9 :=
by
  -- The proof is unnecessary as per instructions
  sorry

end pages_after_break_correct_l88_88180


namespace determine_constants_l88_88055

theorem determine_constants (k a b : ℝ) :
  (3*x^2 - 4*x + 5)*(5*x^2 + k*x + 8) = 15*x^4 - 47*x^3 + a*x^2 - b*x + 40 →
  k = -9 ∧ a = 15 ∧ b = 72 :=
by
  sorry

end determine_constants_l88_88055


namespace speeds_correct_l88_88469

-- Definitions for conditions
def distance (A B : Type) := 40 -- given distance between A and B is 40 km
def start_time_pedestrian : Real := 4 -- pedestrian starts at 4:00 AM
def start_time_cyclist : Real := 7 + (20 / 60) -- cyclist starts at 7:20 AM
def midpoint_distance : Real := 20 -- the midpoint distance where cyclist catches up with pedestrian is 20 km

noncomputable def speeds (x y : Real) : Prop :=
  let t_catch_up := (20 - (10 / 3) * x) / (y - x) in -- time taken by the cyclist to catch up
  let t_total := (10 / 3) + t_catch_up + 1 in -- total time for pedestrian until meeting second cyclist
  4.5 = t_total ∧ -- total time in hours from 4:00 AM to 8:30 AM
  10 * x * (y - x) + 60 * x - 10 * x^2 = 60 * y - 60 * x ∧ -- initial condition simplification step
  y = 6 * x -- relationship between speeds based on derived equations

-- The proposition to prove
theorem speeds_correct : ∃ x y : Real, speeds x y ∧ x = 5 ∧ y = 30 :=
by
  sorry

end speeds_correct_l88_88469


namespace probability_three_heads_l88_88285

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l88_88285


namespace ac_bc_ratios_l88_88212

theorem ac_bc_ratios (A B C : ℝ) (m n : ℕ) (h : AC / BC = m / n) : 
  if m ≠ n then
    ((AC / AB = m / (m+n) ∧ BC / AB = n / (m+n)) ∨ 
     (AC / AB = m / (n-m) ∧ BC / AB = n / (n-m)))
  else 
    (AC / AB = 1 / 2 ∧ BC / AB = 1 / 2) := sorry

end ac_bc_ratios_l88_88212


namespace lola_dora_allowance_l88_88201

variable (total_cost deck_cost sticker_cost sticker_count packs_each : ℕ)
variable (allowance : ℕ)

theorem lola_dora_allowance 
  (h1 : deck_cost = 10)
  (h2 : sticker_cost = 2)
  (h3 : packs_each = 2)
  (h4 : sticker_count = 2 * packs_each)
  (h5 : total_cost = deck_cost + sticker_count * sticker_cost)
  (h6 : total_cost = 18) :
  allowance = 9 :=
sorry

end lola_dora_allowance_l88_88201


namespace boys_speed_l88_88786

-- Define the conditions
def sideLength : ℕ := 50
def timeTaken : ℕ := 72

-- Define the goal
theorem boys_speed (sideLength timeTaken : ℕ) (D T : ℝ) :
  D = (4 * sideLength : ℕ) / 1000 ∧
  T = timeTaken / 3600 →
  (D / T = 10) := by
  sorry

end boys_speed_l88_88786


namespace total_bales_in_barn_l88_88730

-- Definitions based on the conditions 
def initial_bales : ℕ := 47
def added_bales : ℕ := 35

-- Statement to prove the final number of bales in the barn
theorem total_bales_in_barn : initial_bales + added_bales = 82 :=
by
  sorry

end total_bales_in_barn_l88_88730


namespace distance_inequality_l88_88184

theorem distance_inequality
  (n : ℕ)
  (P : Fin (n+1) → ℝ × ℝ)
  (d : ℝ)
  (h0 : 0 < d)
  (h1 : ∀ i j : Fin (n+1), i ≠ j → dist (P i) (P j) ≥ d) :
  ∏ i in Finset.filter (≠ 0) Finset.univ, dist (P 0) (P i) > (d / 3)^n * real.sqrt (nat.factorial (n+1)) :=
by
  sorry

end distance_inequality_l88_88184


namespace dot_product_EC_ED_l88_88595

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l88_88595


namespace maximize_area_ratio_l88_88773

-- Define the conditions of the trapezium
variables {A B C D E F G : Type*}
variables {AB DC AE : ℝ} (h_desc: AB > DC)
variables (h_parallel : ∃ l : ℝ,  A = l * D + (1 - l) * E)

-- Define the points and their relationships
variable h_AE_eq_DC : AE = DC
variable point_E_on_AB : ∃ m : ℝ, E = m * A + (1 - m) * B
variable int_AC_DE_at_F : ∃ k : ℝ, F = k * D + (1 - k) * E
variable int_DB_AC_at_G : ∃ n : ℝ, G = n * D + (1 - n) * B

-- This ensures the correct calculation involved in the problem
theorem maximize_area_ratio (AB DC AE : ℝ) (H_parallel: ∃ l : ℝ, A = l * D + (1 - l) * E) (H1: AE = DC)
  (H2: AB > DC) (H3: ∃ m : ℝ, E = m * A + (1 - m) * B) (
    H4: ∃ k : ℝ, F = k * D + (1 - k) * E) (H5: ∃ n : ℝ, G = n * D + (1 - n) * B) 
  : ∃ x : ℝ, x = 3 := 
sorry

end maximize_area_ratio_l88_88773


namespace find_speeds_l88_88466

noncomputable def speed_pedestrian := 5
noncomputable def speed_cyclist := 30

def distance_AB := 40
def starting_time_pedestrian := 4 -- In hours (24-hour format)
def starting_time_cyclist_1 := 7 + 20 / 60 -- 7:20 AM in hours
def halfway_distance := distance_AB / 2
def midpoint_meeting_time := 1 -- Time (in hours) after the first meeting
def starting_time_cyclist_2 := 8 + 30 / 60 -- 8:30 AM in hours

theorem find_speeds (x y : ℝ) (hx : x = speed_pedestrian) (hy : y = speed_cyclist) :
  let time_to_halfway := halfway_distance / x in
  let cyclist_time := (midpoint_meeting_time + time_to_halfway) in
  distance_AB = 
    cyclist_time * y + 
    time_to_halfway * x + 
    (midpoint_meeting_time - 1) * x :=
    x = speed_pedestrian ∧ y = speed_cyclist :=
begin
  sorry
end

end find_speeds_l88_88466


namespace part1_solution_l88_88849

theorem part1_solution : ∀ n : ℕ, ∃ k : ℤ, 2^n + 3 = k^2 ↔ n = 0 :=
by sorry

end part1_solution_l88_88849


namespace distinct_four_digit_positive_integers_product_18_l88_88977

theorem distinct_four_digit_positive_integers_product_18 :
  Finset.card {n | ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
                             (1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) ∧
                             a * b * c * d = 18} = 24 :=
by
  sorry

end distinct_four_digit_positive_integers_product_18_l88_88977


namespace stone_length_is_correct_l88_88368

variable (length_m width_m : ℕ)
variable (num_stones : ℕ)
variable (width_stone dm : ℕ)

def length_of_each_stone (length_m : ℕ) (width_m : ℕ) (num_stones : ℕ) (width_stone : ℕ) : ℕ :=
  let length_dm := length_m * 10
  let width_dm := width_m * 10
  let area_hall := length_dm * width_dm
  let area_stone := width_stone * 5
  (area_hall / num_stones) / width_stone

theorem stone_length_is_correct :
  length_of_each_stone 36 15 5400 5 = 2 := by
  sorry

end stone_length_is_correct_l88_88368


namespace number_of_valid_n_l88_88426

theorem number_of_valid_n :
  {n : ℕ // 1 ≤ n ∧ n ≤ 30 ∧ ( ∃ k : ℕ, (k : ℕ)! * (n!)^n = (n^3-1)! )}.toFinset.card = 4 :=
sorry

end number_of_valid_n_l88_88426


namespace range_a_le_2_l88_88998
-- Import everything from Mathlib

-- Define the hypothesis and the conclusion in Lean 4
theorem range_a_le_2 (a : ℝ) : 
  (∀ x > 0, Real.log x + a * x + 1 - x * Real.exp (2 * x) ≤ 0) ↔ a ≤ 2 := 
sorry

end range_a_le_2_l88_88998


namespace christmas_bonus_remainder_l88_88758

theorem christmas_bonus_remainder (X : ℕ) (h : X % 5 = 2) : (3 * X) % 5 = 1 :=
by
  sorry

end christmas_bonus_remainder_l88_88758


namespace proof_problem_l88_88503

noncomputable def A := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
noncomputable def B := {(x, y) : ℝ × ℝ | y = x^2 + 1}

theorem proof_problem :
  ((1, 2) ∈ B) ∧
  (0 ∉ A) ∧
  ((0, 0) ∉ B) :=
by
  sorry

end proof_problem_l88_88503


namespace range_of_a_l88_88148

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, 3 * (x - 1) > (x - 6) ∧ 8 - 2 * x + 2 * a ≥ 0)
  → {x : ℤ | 3 * (x - 1) > (x - 6) ∧ 8 - 2 * x + 2 * a ≥ 0}.to_finset.card = 3
  → (-3 ≤ a ∧ a < -2) :=
by
  -- The provided solution steps are to be elaborated here
  sorry

end range_of_a_l88_88148


namespace prime_cube_plus_five_implies_prime_l88_88989

theorem prime_cube_plus_five_implies_prime (p : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime (p^3 + 5)) : p^5 - 7 = 25 := 
by
  sorry

end prime_cube_plus_five_implies_prime_l88_88989


namespace profit_percentage_is_40_l88_88214

-- Define the given conditions
def total_cost : ℚ := 44 * 150 + 36 * 125  -- Rs 11100
def total_weight : ℚ := 44 + 36            -- 80 kg
def selling_price_per_kg : ℚ := 194.25     -- Rs 194.25
def total_selling_price : ℚ := total_weight * selling_price_per_kg  -- Rs 15540
def profit : ℚ := total_selling_price - total_cost  -- Rs 4440

-- Define the statement about the profit percentage
def profit_percentage : ℚ := (profit / total_cost) * 100

-- State the theorem
theorem profit_percentage_is_40 :
  profit_percentage = 40 := by
  -- This is where the proof would go
  sorry

end profit_percentage_is_40_l88_88214


namespace seating_arrangements_l88_88781

theorem seating_arrangements (D R : ℕ) (circular : bool): D = 6 → R = 4 → circular = true →
  (∀ a b : ℕ, a ≠ b → a ≠ b + 1 → ∃! n : ℕ, 
    n = nat.factorial 3 * nat.factorial 6) → 
  ∃ n : ℕ, n = 4320 :=
by
  intros D R circular hD hR hcircular hno_two_adj
  have h:= hno_two_adj 6 4 hD hR
  rw[h]
  use 4320
  sorry

end seating_arrangements_l88_88781


namespace fivefold_composition_l88_88139

def f (x : ℚ) : ℚ := -2 / x

theorem fivefold_composition :
  f (f (f (f (f (3))))) = -2 / 3 := 
by
  -- Proof goes here
  sorry

end fivefold_composition_l88_88139


namespace tetrahedron_circumsphere_surface_area_eq_five_pi_l88_88070

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

noncomputable def circumscribed_sphere_radius (a b : ℝ) : ℝ :=
  rectangle_diagonal a b / 2

noncomputable def circumscribed_sphere_surface_area (a b : ℝ) : ℝ :=
  4 * Real.pi * (circumscribed_sphere_radius a b)^2

theorem tetrahedron_circumsphere_surface_area_eq_five_pi :
  circumscribed_sphere_surface_area 2 1 = 5 * Real.pi := by
  sorry

end tetrahedron_circumsphere_surface_area_eq_five_pi_l88_88070


namespace parallelepiped_ratio_l88_88417

noncomputable def v : ℝ^4 := ![1, 0, 0, 0]
noncomputable def w : ℝ^4 := ![0, 1, 0, 0]
noncomputable def u : ℝ^4 := ![0, 0, 1, 0]

noncomputable def AG2 : ℝ := ∥u + v + w∥^2
noncomputable def BH2 : ℝ := ∥u - v + w∥^2
noncomputable def CE2 : ℝ := ∥-u + v + w∥^2
noncomputable def DF2 : ℝ := ∥u + v - w∥^2
noncomputable def AB2 : ℝ := ∥v∥^2
noncomputable def AD2 : ℝ := ∥w∥^2
noncomputable def AE2 : ℝ := ∥u∥^2

theorem parallelepiped_ratio :
  (AG2 + BH2 + CE2 + DF2) / (AB2 + AD2 + AE2) = 4 := by
  sorry

end parallelepiped_ratio_l88_88417


namespace greatest_of_consecutive_even_numbers_l88_88339

theorem greatest_of_consecutive_even_numbers (n : ℤ) (h : ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 35) : n + 4 = 39 :=
by
  sorry

end greatest_of_consecutive_even_numbers_l88_88339


namespace maximize_wind_power_l88_88699

variable {C S ρ v_0 : ℝ}

theorem maximize_wind_power : 
  ∃ v : ℝ, (∀ (v' : ℝ),
           let F := (C * S * ρ * (v_0 - v)^2) / 2;
           let N := F * v;
           let N' := (C * S * ρ / 2) * (v_0^2 - 4 * v_0 * v + 3 * v^2);
           N' = 0
         → N ≤ (C * S * ρ / 2) * (v_0^2 * (v_0/3) - 2 * v_0 * (v_0/3)^2 + (v_0/3)^3)) ∧ v = v_0 / 3 :=
by sorry

end maximize_wind_power_l88_88699


namespace dot_product_square_ABCD_l88_88599

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l88_88599


namespace tangent_line_at_point_l88_88238

noncomputable def f (x : ℝ) : ℝ := x^2 + 1 / x

def point := (1, f 1)

theorem tangent_line_at_point : ∀ x y : ℝ, 
  (fderiv ℝ f 1).to_fun 1 * (x - 1) = y - f 1 ↔ x - y + 1 = 0 :=
by 
  sorry

end tangent_line_at_point_l88_88238


namespace centennial_park_cleanup_finished_at_l88_88555

-- Define the conditions as Lean definitions.
def constant_growth_rate_per_minute : ℝ := 1 -- weeds grow at 1 unit per minute

def first_day_finish_time : ℝ := 60 -- 60 minutes to finish on the first day

def second_day_workers : ℕ := 10 
def second_day_finish_time : ℝ := 30 -- 10 workers finish in 30 minutes on the second day
def second_day_total_growth : ℝ := (23 * 60 + 30) * constant_growth_rate_per_minute -- growth from the end of first day to second day

def worker_clear_rate (units: ℝ) (workers: ℕ) (time: ℝ) : ℝ := units / (workers * time)

def second_day_worker_rate := worker_clear_rate second_day_total_growth second_day_workers second_day_finish_time -- each worker rate in units/min

def third_day_workers : ℕ := 8
def third_day_finish_time_integer : ℝ := 37.5 -- calculated clearing time in minutes (non-integer part for consideration)
def third_day_rounded_finish_time : ℕ := 38 -- rounded to the nearest minute

def third_day_total_growth : ℝ := second_day_total_growth

-- Define the Lean theorem statement.
theorem centennial_park_cleanup_finished_at : 
  -- We need to prove that on the third day, the workers finished rounding the time to nearest minute.
  let total_clearing_rate := third_day_workers * second_day_worker_rate in
  total_clearing_rate * (real.ceil third_day_finish_time_integer : ℝ) = third_day_total_growth → 
  real.ceil third_day_finish_time_integer = third_day_rounded_finish_time :=
by
    sorry

end centennial_park_cleanup_finished_at_l88_88555


namespace ellipse_eccentricity_min_val_l88_88517

noncomputable def ellipse_conditions 
  (a b c : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) (h : b = sqrt (3 * c)) : Prop :=
  ∃ e, 
    (e = c / a) ∧
    (e = 1 / 2)

noncomputable def min_value_condition 
  (a b x₀ c : ℝ) (a_neq_b : a ≠ b) (h1 : -a ≤ x₀) (h2 : x₀ ≤ a) : Prop :=
  ∃ k_PM k_PN: ℝ,
    ∣(1 / k_PM) - (1 / k_PN) ∣ = 4 / 3

theorem ellipse_eccentricity_min_val 
  (a b c : ℝ) (x₀ : ℝ) (h_c : c = a * sqrt(1 - (b / a) ^ 2)) 
  (a_gt_b : a > b) (b_gt_0 : b > 0) (a_neq_b : a ≠ b) 
  (h : b = sqrt(3 * c)) 
  (h1 : -a ≤ x₀) (h2 : x₀ ≤ a) : 
  ellipse_conditions a b c h ∧ min_value_condition a b x₀ c a_neq_b h1 h2 :=
by {
  sorry
}

end ellipse_eccentricity_min_val_l88_88517


namespace correct_propositions_l88_88142

structure PerfectTriangle (a b c : ℕ) :=
(angle_60 : ∃ x y z, (x = a ∧ y = b ∧ z = c ∧ (x*x + y*y - x*y = z*z)) ∨ -- Using Law of Cosines for \(\frac{\pi}{3}\) angle
                     (x = b ∧ y = c ∧ z = a ∧ (x*x + y*y - x*y = z*z)) ∨ 
                     (x = c ∧ y = a ∧ z = b ∧ (x*x + y*y - x*y = z*z)))

def Proposition1 : Prop :=
  ¬ ∃ a b c : ℕ, RightTriangle a b c ∧ ((∃ x y z, PerfectTriangle x y z))

def Proposition2 : Prop :=
  ¬ ∃ a b c : ℕ, PerfectTriangle a b c ∧ (∃ k : ℕ, (√3 * (a * b) / 4 = k))

def Proposition3 : Prop :=
  ∃ (a b c : ℕ), PerfectTriangle a b c ∧ (a + b + c = 12) ∧ 
  let area := (√3 * (a * b) / 4) 
  in 4 * √3 = area

def TriangleCongruent (a1 b1 c1 a2 b2 c2 : ℕ) : Prop :=
  a1 = a2 ∧ b1 = b2 ∧ c1 = c2

def Proposition4 : Prop :=
  ∀ (a1 b1 c1 a2 b2 c2 : ℕ), (PerfectTriangle a1 b1 c1 ∧ 
                              PerfectTriangle a2 b2 c2 ∧ 
                              (∃ x y, (a1 = x ∧ b1 = y) ∧ (a2 = x ∧ b2 = y)) ∧ 
                              (let area := (√3 * (a1 * b1) / 4) in 
                               let area' := (√3 * (a2 * b2) / 4) in 
                               area = area')) → 
                               TriangleCongruent a1 b1 c1 a2 b2 c2

theorem correct_propositions : Proposition3 ∧ Proposition4 ∧ ¬ Proposition1 ∧ ¬ Proposition2 :=
by
  sorry

end correct_propositions_l88_88142


namespace handrail_length_proof_l88_88005

def handrail_length (height radius : ℝ) (turn_deg : ℝ) : ℝ :=
  let arc_length := turn_deg / 360 * 2 * Real.pi * radius
  Real.sqrt (height^2 + arc_length^2)

theorem handrail_length_proof :
  handrail_length 15 3 180 = 15.9 := 
by
  sorry

end handrail_length_proof_l88_88005


namespace contractor_fine_per_absent_day_l88_88359

noncomputable def fine_per_absent_day (total_days : ℕ) (pay_per_day : ℝ) (total_amount_received : ℝ) (days_absent : ℕ) : ℝ :=
  let days_worked := total_days - days_absent
  let earned := days_worked * pay_per_day
  let fine := (earned - total_amount_received) / days_absent
  fine

theorem contractor_fine_per_absent_day :
  fine_per_absent_day 30 25 425 10 = 7.5 := by
  sorry

end contractor_fine_per_absent_day_l88_88359


namespace range_of_f_l88_88713

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 18 * Real.pi / 180) - Real.cos (x + 48 * Real.pi / 180)

theorem range_of_f :
  Set.Range f = Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end range_of_f_l88_88713


namespace different_denominators_count_l88_88685

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def repeating_decimal_to_fraction (c d : ℕ) : ℚ := (10 * c + d) / 99

theorem different_denominators_count : 
  ∀ (c d : ℕ), is_digit c → is_digit d → c ≠ d → (c ≠ 0 ∨ d ≠ 0) → 
  ∃ denom_set : set ℕ, denom_set = {3, 9, 11, 33, 99} ∧ card denom_set = 5 :=
by
  intros c d hc hd hcd hne
  sorry

end different_denominators_count_l88_88685


namespace tablets_of_medicine_A_l88_88785

-- Given conditions as definitions
def B_tablets : ℕ := 16

def min_extracted_tablets : ℕ := 18

-- Question and expected answer encapsulated in proof statement
theorem tablets_of_medicine_A (A_tablets : ℕ) (h : A_tablets + B_tablets - 2 >= min_extracted_tablets) : A_tablets = 3 :=
sorry

end tablets_of_medicine_A_l88_88785


namespace stratified_sampling_correct_l88_88358

noncomputable def production_volumes : (ℕ × ℕ × ℕ) := (1200, 6000, 2000)
noncomputable def total_cars : ℕ := 9200
noncomputable def total_sample : ℕ := 46

def samples_needed (pv : ℕ × ℕ × ℕ) (tc : ℕ) (ts : ℕ) : (ℕ × ℕ × ℕ) :=
  let (v1, v2, v3) := pv in
  let s1 := (v1 * ts) / tc
  let s2 := (v2 * ts) / tc
  let s3 := (v3 * ts) / tc
  (s1, s2, s3)

theorem stratified_sampling_correct :
  samples_needed production_volumes total_cars total_sample = (6, 30, 10) := by
sorry

end stratified_sampling_correct_l88_88358


namespace three_heads_in_a_row_l88_88307

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l88_88307


namespace triangular_pyramid_volume_l88_88492

open Real

-- Define the edge length of the triangular pyramid
def edge_length : ℝ := sqrt 2

-- Define the base area of an equilateral triangle with side length sqrt(2)
def base_area : ℝ := (sqrt 3) / 2

-- Define the height of the triangular pyramid
def height : ℝ := 2 / sqrt 3

-- Define the expected volume of the triangular pyramid
def expected_volume : ℝ := 1 / 3

-- The proof problem to be proved
theorem triangular_pyramid_volume : 
  ∀ (a b c d e f : ℝ), 
  a = edge_length →
  b = edge_length →
  c = edge_length →
  d = edge_length →
  e = edge_length →
  f = edge_length →
  let base := base_area
  in let h := height
  in let volume := (1 / 3) * base * h
  in volume = expected_volume :=
by
  intros a b c d e f ha hb hc hd he hf
  rw [ha, hb, hc, hd, he, hf]
  simp [base_area, height]
  sorry

end triangular_pyramid_volume_l88_88492


namespace general_term_formula_sum_of_first_n_terms_l88_88488

section GeoSeq
variables {T : Type*} [linear_ordered_field T]

noncomputable def q : T := 1/2

def a (n : ℕ) : T := (1/2)^(n-1)

def S (n : ℕ) : T := (1-(1/2)^n) / (1-(1/2))

theorem general_term_formula 
  (hS6 : S 6 = 63/32)
  (hSeq : 2 * a 4 = -a 2 + 3 * a 3) 
  : ∀ n, a n = (1/2)^(n-1) :=
sorry

def b (n : ℕ) : T := a n * n

noncomputable def T (n : ℕ) : T :=
∑ i in finset.range n, b (i+1)

theorem sum_of_first_n_terms 
  (hS6 : S 6 = 63/32)
  (hSeq : 2 * a 4 = -a 2 + 3 * a 3)
  : ∀ n, T n = 4 - (2 * n + 4) * (1/2)^n :=
sorry
end GeoSeq

end general_term_formula_sum_of_first_n_terms_l88_88488


namespace thief_distance_l88_88813

variable (d : ℝ := 250)   -- initial distance in meters
variable (v_thief : ℝ := 12 * 1000 / 3600)  -- thief's speed in m/s (converted from km/hr)
variable (v_policeman : ℝ := 15 * 1000 / 3600)  -- policeman's speed in m/s (converted from km/hr)

noncomputable def distance_thief_runs : ℝ :=
  v_thief * (d / (v_policeman - v_thief))

theorem thief_distance :
  distance_thief_runs d v_thief v_policeman = 990.47 := sorry

end thief_distance_l88_88813


namespace distinct_four_digit_integers_with_digit_product_18_l88_88960

theorem distinct_four_digit_integers_with_digit_product_18 : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in digits.prod = 18) → 
  (finset.univ.filter (λ m, (let mdigits := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in mdigits.prod = 18))).card = 36 :=
by
  sorry

end distinct_four_digit_integers_with_digit_product_18_l88_88960


namespace divide_plane_into_regions_l88_88882

theorem divide_plane_into_regions (n : ℕ) (h₁ : n < 199) (h₂ : ∃ (k : ℕ), k = 99):
  n = 100 ∨ n = 198 :=
sorry

end divide_plane_into_regions_l88_88882


namespace least_integer_solution_l88_88278

theorem least_integer_solution :
  ∃ x : ℤ, (abs (3 * x - 4) ≤ 25) ∧ (∀ y : ℤ, (abs (3 * y - 4) ≤ 25) → x ≤ y) :=
sorry

end least_integer_solution_l88_88278


namespace cost_of_adult_ticket_l88_88013

theorem cost_of_adult_ticket (x : ℕ) (total_persons : ℕ) (total_collected : ℕ) (adult_tickets : ℕ) (child_ticket_cost : ℕ) (amount_from_children : ℕ) :
  total_persons = 280 →
  total_collected = 14000 →
  adult_tickets = 200 →
  child_ticket_cost = 25 →
  amount_from_children = 2000 →
  200 * x + amount_from_children = total_collected →
  x = 60 :=
by
  intros h_persons h_total h_adults h_child_cost h_children_amount h_eq
  sorry

end cost_of_adult_ticket_l88_88013


namespace prob_first_three_heads_all_heads_l88_88318

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l88_88318


namespace parabola_equation_dot_product_y_coordinate_range_C_l88_88372

-- Define the parabola parameter and condition on the product of y-coordinates
variables (p : ℝ) (h_p : p > 0) (y1 y2 : ℝ) (h_prod_y : y1 * y2 = -4)

-- State the first proof problem
theorem parabola_equation : y^2 = 4x :=
by admit

-- Define points M (x1, y1) and N (x2, y2) on the parabola
variables (x1 x2 : ℝ) (h_MN_x : x1 = (y1^2) / 4) (h_MN_x2 : x2 = (y2^2) / 4) -- Joined via the parabola equation

-- Origin O
variables (O : ℝ×ℝ) (h_O : O = (0, 0))

-- State the second proof problem
theorem dot_product : (O.1 * x1 + O.2 * y1) * (O.1 * x2 + O.2 * y2) = -3 :=
by admit

-- Define point A
variables (A : ℝ×ℝ) (h_A : A = (1, 2))

-- State the third proof problem
theorem y_coordinate_range_C (y3 y4 : ℝ) (h_B : B = (y3^2 / 4, y3)) (h_C : C = (y4^2 / 4, y4))
  (h_perpendicular : (A.1 - B.1) * (B.1 - C.1) + (A.2 - B.2) * (B.2 - C.2) = 0) :
  y4 ∈ (-∞, -6) ∪ [10, ∞) :=
by admit

end parabola_equation_dot_product_y_coordinate_range_C_l88_88372


namespace max_value_sqrt_sum_l88_88192

theorem max_value_sqrt_sum (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 8) :
    (sqrt (3 * x + 2) + sqrt (3 * y + 2) + sqrt (3 * z + 2)) ≤ 3 * sqrt 10 := 
by
  sorry

end max_value_sqrt_sum_l88_88192


namespace cosine_double_angle_identity_given_sine_l88_88093

theorem cosine_double_angle_identity_given_sine :
  (sin (π / 6 - α) = 1 / 3) → cos (2 * (π / 3 + α)) = -7 / 9 :=
by
  intro h
  sorry

end cosine_double_angle_identity_given_sine_l88_88093


namespace area_EFGH_area_MNPQ_area_common_l88_88209

-- Definitions of parameters and conditions
variable {a : ℝ}

-- Conditions
def square_ABCD (ABCD : ℝ × ℝ → Prop) : Prop :=
∀ P Q, ABCD (P, Q) →
  (∃ x y, P = (x, y) ∧ Q = (x + a, y)) ∧
  (∃ x y, P = (x, y) ∧ Q = (x, y + a)) ∧
  (∃ x y, P = (x, y) ∧ Q = (x + a, y))

def equilateral_on_square (ABCDE : ℝ × ℝ × Prop) : Prop :=
∀ P Q R, ABCDE (P, Q, R) →
  (∃ x y, P = (x, y) ∧ Q = (x + a / 2, y + a * sqrt 3 / 2) ∧ R = (x - a / 2, y + a * sqrt 3 / 2)) ∧
  (∃ x y, P = (x, y) ∧ Q = (x - a / 2, y + a * sqrt 3 / 2) ∧ R = (x, y + a * sqrt 3))

-- Proof statements
theorem area_EFGH (ABCD : ℝ × ℝ → Prop) (ABCDE : ℝ × ℝ × Prop)
  (hABCD : square_ABCD ABCD) (hABCDE : equilateral_on_square ABCDE) : 
  ∃ t : ℝ, t = a^2 * (2 + sqrt 3) :=
sorry

theorem area_MNPQ (ABCD : ℝ × ℝ → Prop) (ABCDE : ℝ × ℝ × Prop)
  (hABCD : square_ABCD ABCD) (hABCDE : equilateral_on_square ABCDE) : 
  ∃ t : ℝ, t = 3 * a^2 :=
sorry

theorem area_common (ABCD : ℝ × ℝ → Prop) (ABCDE : ℝ × ℝ × Prop) 
  (hABCD : square_ABCD ABCD) (hABCDE : equilateral_on_square ABCDE) : 
  ∃ t : ℝ, t = a^2 * (sqrt 3 + 1) :=
sorry

end area_EFGH_area_MNPQ_area_common_l88_88209


namespace employed_females_percentage_l88_88610

def P_total : ℝ := 0.64
def P_males : ℝ := 0.46

theorem employed_females_percentage : 
  ((P_total - P_males) / P_total) * 100 = 28.125 :=
by
  sorry

end employed_females_percentage_l88_88610


namespace sequence_2018_eq_half_l88_88945

noncomputable def sequence : ℕ → ℚ
| 0     := 2
| (n+1) := 1 - 1 / sequence n

theorem sequence_2018_eq_half : sequence 2018 = 1 / 2 :=
by
  sorry

end sequence_2018_eq_half_l88_88945


namespace bin_game_expected_value_l88_88352

theorem bin_game_expected_value (k : ℕ) (h : (3 * 10 - 1 * k) / (10 + k) = 0.75) : k = 13 := 
sorry

end bin_game_expected_value_l88_88352


namespace student_count_before_new_student_l88_88235

variable {W : ℝ} -- total weight of students before the new student joined
variable {n : ℕ} -- number of students before the new student joined
variable {W_new : ℝ} -- total weight including the new student
variable {n_new : ℕ} -- number of students including the new student

theorem student_count_before_new_student 
  (h1 : W = n * 28) 
  (h2 : W_new = W + 7) 
  (h3 : n_new = n + 1) 
  (h4 : W_new / n_new = 27.3) : n = 29 := 
by
  sorry

end student_count_before_new_student_l88_88235


namespace distinct_four_digit_integers_with_digit_product_18_l88_88959

theorem distinct_four_digit_integers_with_digit_product_18 : 
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in digits.prod = 18) → 
  (finset.univ.filter (λ m, (let mdigits := [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10] in mdigits.prod = 18))).card = 36 :=
by
  sorry

end distinct_four_digit_integers_with_digit_product_18_l88_88959


namespace mode_data_set_l88_88568

def data_set : List ℕ := [32, 35, 32, 33, 30, 32, 31]

theorem mode_data_set : List.mode data_set = 32 := 
sorry

end mode_data_set_l88_88568


namespace circumcenters_concyclic_l88_88621

/-- Given a point E inside parallelogram ABCD such that ∠BCE = ∠BAE, 
    prove that the circumcenters of triangles ABE, BCE, CDE, and DAE are concyclic. --/
theorem circumcenters_concyclic (A B C D E : Type*) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
  (parallelogram : parallelogram A B C D) 
  (h_angle : angle BCE = angle BAE) : 
  concyclic (circumcenter (△ABE)) (circumcenter (△BCE)) (circumcenter (△CDE)) (circumcenter (△DAE)) :=
sorry

end circumcenters_concyclic_l88_88621


namespace unfair_coin_probability_l88_88021

theorem unfair_coin_probability (P : ℕ → ℝ) :
  let heads := 3/4
  let initial_condition := P 0 = 1
  let recurrence_relation := ∀n, P (n + 1) = 3 / 4 * (1 - P n) + 1 / 4 * P n
  recurrence_relation →
  initial_condition →
  P 40 = 1 / 2 * (1 + (1 / 2) ^ 40) :=
by
  sorry

end unfair_coin_probability_l88_88021


namespace min_positive_integer_expression_l88_88446

theorem min_positive_integer_expression : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (m: ℝ) / 3 + 27 / (m: ℝ) ≥ (n: ℝ) / 3 + 27 / (n: ℝ)) ∧ (n / 3 + 27 / n = 6) :=
sorry

end min_positive_integer_expression_l88_88446


namespace reducible_fraction_l88_88851

theorem reducible_fraction (l : ℤ) : ∃ k : ℤ, l = 13 * k + 4 ↔ (∃ d > 1, d ∣ (5 * l + 6) ∧ d ∣ (8 * l + 7)) :=
sorry

end reducible_fraction_l88_88851


namespace coordinate_fifth_point_l88_88767

theorem coordinate_fifth_point : 
  ∃ (a : Fin 16 → ℝ), 
    a 0 = 2 ∧ 
    a 15 = 47 ∧ 
    (∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) ∧ 
    a 4 = 14 := 
sorry

end coordinate_fifth_point_l88_88767


namespace max_planes_with_conditions_l88_88051

def has_four_points_on_each_plane (planes : Set (Set Point))
  (points : Set Point) : Prop := 
  ∀ plane ∈ planes, ∃ subset ⊆ points, subset.card ≥ 4 ∧ subset ⊆ plane

def any_four_points_not_collinear (points : Set Point) : Prop :=
  ∀ (p1 p2 p3 p4 : Point) ∈ points, ¬collinear ({p1, p2, p3, p4} : Set Point)

theorem max_planes_with_conditions :
  ∃ (planes : Set (Set Point)) (points : Set Point), points.card = 6 ∧
  has_four_points_on_each_plane planes points ∧ 
  any_four_points_not_collinear points ∧
  (∀ (planes' : Set (Set Point)), planes_compat planes_plane' ∧ planes'.card > planes.card → false) ∧ 
  planes.card = 6 :=
sorry

end max_planes_with_conditions_l88_88051


namespace arrangements_of_four_from_six_l88_88141

theorem arrangements_of_four_from_six :
  (∃ volunteers : Finset ℕ, volunteers.card = 6) →
  (∃ roles : Finset ℕ, roles.card = 4) →
  (∃ arrangements : ℕ, arrangements = Nat.choose 6 4 * 4!) →
  arrangements = 360 :=
by
  intros h₁ h₂ h₃
  obtain ⟨volunteers, hVolunteers⟩ := h₁
  obtain ⟨roles, hRoles⟩ := h₂
  obtain ⟨arrangements, harrangements⟩ := h₃
  rw [Nat.choose_eq_factorial_div_factorial, mul_comm] at harrangements
  sorry

end arrangements_of_four_from_six_l88_88141


namespace speeds_correct_l88_88470

-- Definitions for conditions
def distance (A B : Type) := 40 -- given distance between A and B is 40 km
def start_time_pedestrian : Real := 4 -- pedestrian starts at 4:00 AM
def start_time_cyclist : Real := 7 + (20 / 60) -- cyclist starts at 7:20 AM
def midpoint_distance : Real := 20 -- the midpoint distance where cyclist catches up with pedestrian is 20 km

noncomputable def speeds (x y : Real) : Prop :=
  let t_catch_up := (20 - (10 / 3) * x) / (y - x) in -- time taken by the cyclist to catch up
  let t_total := (10 / 3) + t_catch_up + 1 in -- total time for pedestrian until meeting second cyclist
  4.5 = t_total ∧ -- total time in hours from 4:00 AM to 8:30 AM
  10 * x * (y - x) + 60 * x - 10 * x^2 = 60 * y - 60 * x ∧ -- initial condition simplification step
  y = 6 * x -- relationship between speeds based on derived equations

-- The proposition to prove
theorem speeds_correct : ∃ x y : Real, speeds x y ∧ x = 5 ∧ y = 30 :=
by
  sorry

end speeds_correct_l88_88470


namespace area_of_rhombus_l88_88760

theorem area_of_rhombus (d1 d2 : ℝ) (hd1 : d1 = 70) (hd2 : d2 = 160) : (d1 * d2) / 2 = 5600 :=
by
  rw [hd1, hd2]
  norm_num
  sorry

end area_of_rhombus_l88_88760


namespace range_of_m_for_inequality_l88_88346

theorem range_of_m_for_inequality (x y m : ℝ) :
  (∀ x y : ℝ, 3*x^2 + y^2 ≥ m * x * (x + y)) ↔ (-6 ≤ m ∧ m ≤ 2) := sorry

end range_of_m_for_inequality_l88_88346


namespace range_of_a_for_no_extreme_points_l88_88996

theorem range_of_a_for_no_extreme_points :
  ∀ (a : ℝ), (∀ x : ℝ, x * (x - 2 * a) * x + 1 ≠ 0) ↔ -1 ≤ a ∧ a ≤ 1 := sorry

end range_of_a_for_no_extreme_points_l88_88996


namespace certain_number_l88_88343

theorem certain_number (x : ℝ) (h : (2.28 * x) / 6 = 480.7) : x = 1265.0 := 
by 
  sorry

end certain_number_l88_88343


namespace binomial_20_10_l88_88912

open Nat

theorem binomial_20_10 :
  (binomial 18 8 = 43758) →
  (binomial 18 9 = 48620) →
  (binomial 18 10 = 43758) →
  binomial 20 10 = 184756 :=
by
  intros h1 h2 h3
  sorry

end binomial_20_10_l88_88912


namespace maximize_wind_power_l88_88697

variable {C S ρ v_0 : ℝ}

theorem maximize_wind_power : 
  ∃ v : ℝ, (∀ (v' : ℝ),
           let F := (C * S * ρ * (v_0 - v)^2) / 2;
           let N := F * v;
           let N' := (C * S * ρ / 2) * (v_0^2 - 4 * v_0 * v + 3 * v^2);
           N' = 0
         → N ≤ (C * S * ρ / 2) * (v_0^2 * (v_0/3) - 2 * v_0 * (v_0/3)^2 + (v_0/3)^3)) ∧ v = v_0 / 3 :=
by sorry

end maximize_wind_power_l88_88697


namespace wendi_initial_chickens_l88_88273

theorem wendi_initial_chickens (X : ℕ) 
    (h0 : ∀ y, y = 2 * X)
    (h1 : ∀ z, z = y - 1)
    (h2 : ∀ w, w = 10 - 4)
    (h3 : ∀ t, t = z + w)
    (h4 : t = 13) : X = 4 :=
by
  sorry

end wendi_initial_chickens_l88_88273


namespace always_negative_l88_88938

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x ^ 2 + 1) - x) - Real.sin x

theorem always_negative (a b : ℝ) (ha : a ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hb : b ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hab : a + b ≠ 0) : 
  (f a + f b) / (a + b) < 0 := 
sorry

end always_negative_l88_88938


namespace cos_double_angle_l88_88993

variable (θ : Real)

theorem cos_double_angle (h : ∑' n, (Real.cos θ)^(2 * n) = 7) : Real.cos (2 * θ) = 5 / 7 := 
  by sorry

end cos_double_angle_l88_88993


namespace maximize_wind_power_l88_88698

variable {C S ρ v_0 : ℝ}

theorem maximize_wind_power : 
  ∃ v : ℝ, (∀ (v' : ℝ),
           let F := (C * S * ρ * (v_0 - v)^2) / 2;
           let N := F * v;
           let N' := (C * S * ρ / 2) * (v_0^2 - 4 * v_0 * v + 3 * v^2);
           N' = 0
         → N ≤ (C * S * ρ / 2) * (v_0^2 * (v_0/3) - 2 * v_0 * (v_0/3)^2 + (v_0/3)^3)) ∧ v = v_0 / 3 :=
by sorry

end maximize_wind_power_l88_88698


namespace calculate_difference_l88_88736

theorem calculate_difference :
  let a := 3.56
  let b := 2.1
  let c := 1.5
  a - (b * c) = 0.41 :=
by
  let a := 3.56
  let b := 2.1
  let c := 1.5
  show a - (b * c) = 0.41
  sorry

end calculate_difference_l88_88736


namespace numberOfValidSequencesLength6_l88_88728

-- Definition of the conditions of the problem
def isValidSequence (seq : List ℕ) : Prop :=
  seq.head = 1 ∧
  ∀ (i : ℕ), i ∈ seq → i > 1 → i - 1 ∈ seq.take (seq.indexOf i)

-- Theorem statement saying that the number of valid sequences of length 6 is 203
theorem numberOfValidSequencesLength6 : 
  (List.filter isValidSequence (List.permutations [1, 2, 3, 4, 5, 6])).length = 203 :=
by sorry

end numberOfValidSequencesLength6_l88_88728


namespace valid_numbers_count_l88_88125

def is_valid_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

def count_valid_numbers : ℕ :=
  ∑ n in (finset.range 10000).filter (λ n, ∀ d ∈ n.digits 10, is_valid_digit d), 1

theorem valid_numbers_count : count_valid_numbers = 624 := by
  sorry

end valid_numbers_count_l88_88125


namespace range_of_a_l88_88552

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (a + 1) * x^2 + 4 * x + 1 < 0) ↔ a ∈ set.Iio 1 := 
sorry

end range_of_a_l88_88552


namespace ariel_age_l88_88014

theorem ariel_age : ∃ A : ℕ, (A + 15 = 4 * A) ∧ A = 5 :=
by
  -- Here we skip the proof
  sorry

end ariel_age_l88_88014


namespace inradius_plus_circumradius_le_height_l88_88634

theorem inradius_plus_circumradius_le_height {α β γ : ℝ} 
    (h : ℝ) (r R : ℝ)
    (h_triangle : α ≥ β ∧ β ≥ γ ∧ γ ≥ 0 ∧ α + β + γ = π )
    (h_non_obtuse : π / 2 ≥ α ∧ π / 2 ≥ β ∧ π / 2 ≥ γ)
    (h_greatest_height : true) -- Assuming this condition holds as given
    :
    r + R ≤ h :=
sorry

end inradius_plus_circumradius_le_height_l88_88634


namespace markese_earnings_l88_88649

-- Define the conditions
def earnings_relation (E M : ℕ) : Prop :=
  M = E - 5 ∧ M + E = 37

-- The theorem to prove
theorem markese_earnings (E M : ℕ) (h : earnings_relation E M) : M = 16 :=
by
  sorry

end markese_earnings_l88_88649


namespace polygon_sides_l88_88143

theorem polygon_sides (angle : ℝ) (h : angle = 30) : ∃ n : ℕ, n = 12 :=
by {
  have Hsum : 360 / angle = 12,
  { 
    sorry -- This is where we would normally provide a proof for Hsum.
  },
  use 12,
  rw Hsum,
}

end polygon_sides_l88_88143


namespace smaller_circle_radius_l88_88793

open Real

def is_geometric_progression (a b c : ℝ) : Prop :=
  (b / a = c / b)

theorem smaller_circle_radius 
  (B1 B2 : ℝ) 
  (r2 : ℝ) 
  (h1 : B1 + B2 = π * r2^2) 
  (h2 : r2 = 5) 
  (h3 : is_geometric_progression B1 B2 (B1 + B2)) :
  sqrt ((-1 + sqrt (1 + 100 * π)) / (2 * π)) = sqrt (B1 / π) :=
by
  sorry

end smaller_circle_radius_l88_88793


namespace expand_count_terms_l88_88134

theorem expand_count_terms : 
  ∃ n, n = 4033 ∧ ∀ a b c : ℕ, a + b + c = 2016 → 
  let exponents := 3 * a - 3 * b in 
  -2016 ≤ exponents ∧ exponents ≤ 2016 := 
sorry

end expand_count_terms_l88_88134


namespace divideDogs_l88_88229

def waysToDivideDogs : ℕ :=
  (10.choose 2) * (8.choose 4)

theorem divideDogs : waysToDivideDogs = 3150 := by
  sorry

end divideDogs_l88_88229


namespace total_number_of_boys_l88_88565

theorem total_number_of_boys (B : ℕ) 
  (h1 : 0.44 * B + 0.32 * B + 0.10 * B + 119 = B) 
  (h2 : 0.14 * B = 119) : B = 850 := 
by
  sorry

end total_number_of_boys_l88_88565


namespace markese_earned_16_l88_88648

def evan_earnings (E : ℕ) : Prop :=
  (E : ℕ)

def markese_earnings (M : ℕ) (E : ℕ) : Prop :=
  (M : ℕ) = E - 5

def total_earnings (E M : ℕ) : Prop :=
  E + M = 37

theorem markese_earned_16 (E : ℕ) (M : ℕ) 
  (h1 : markese_earnings M E) 
  (h2 : total_earnings E M) : M = 16 :=
sorry

end markese_earned_16_l88_88648


namespace exist_ai_for_xij_l88_88667

theorem exist_ai_for_xij (n : ℕ) (x : Fin n → Fin n → ℝ)
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ a : Fin n → ℝ, ∀ i j : Fin n, x i j = a i - a j :=
by
  sorry

end exist_ai_for_xij_l88_88667


namespace probability_of_rolling_prime_l88_88228

/-- Given a standard 12-sided die and the set of prime numbers between 1 and 12,
    the probability of rolling a prime number is 5/12. -/
theorem probability_of_rolling_prime :
  let die_faces := {n | n ∈ (Finset.range 12).map Nat.succ}
  let prime_faces := {2, 3, 5, 7, 11}
  (prime_faces.card : ℚ) / (die_faces.card : ℚ) = 5 / 12 :=
by
  sorry

end probability_of_rolling_prime_l88_88228


namespace solve_for_x_l88_88842

theorem solve_for_x : ∀ (x : ℝ), (-3 * x - 8 = 5 * x + 4) → (x = -3 / 2) := by
  intro x
  intro h
  sorry

end solve_for_x_l88_88842


namespace binom_20_10_eq_184756_l88_88909

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l88_88909


namespace locus_of_moving_circle_is_hyperbola_l88_88357

theorem locus_of_moving_circle_is_hyperbola :
  ∀ (h k : ℝ),
  (∃ (r : ℝ),
    (sqrt (h^2 + k^2) - 1) = r ∧
    (sqrt ((h - 4)^2 + k^2) - 2) = r) →
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (a * h^2 - b * h + 1 = 0) :=
by
  sorry

end locus_of_moving_circle_is_hyperbola_l88_88357


namespace sequence_formula_general_formula_l88_88084

open BigOperators

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2 * n + 2

def S_n (n : ℕ) : ℕ :=
  n^2 + 3 * n + 1

theorem sequence_formula :
  ∀ n, a_n n =
    if n = 1 then 5 else 2 * n + 2 := by
  sorry

theorem general_formula (n : ℕ) :
  a_n n =
    if n = 1 then S_n 1 else S_n n - S_n (n - 1) := by
  sorry

end sequence_formula_general_formula_l88_88084


namespace distinct_ball_box_distribution_l88_88135

theorem distinct_ball_box_distribution (balls : ℕ) (boxes : ℕ) (h_balls : balls = 6) (h_boxes : boxes = 3) :
  (number_of_ways_to_distribute balls boxes) = 5 :=
sorry

noncomputable def number_of_ways_to_distribute (balls boxes : ℕ) : ℕ :=
sorry

end distinct_ball_box_distribution_l88_88135


namespace probability_three_heads_l88_88324

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l88_88324


namespace midpoint_translation_l88_88672

open Real

theorem midpoint_translation :
  let p1 := (3 : ℝ, -2 : ℝ)
      p2 := (-7 : ℝ, 6 : ℝ)
      mp_s3 := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
      translation := (3, -4)
      mp_s4 := (mp_s3.1 + translation.1, mp_s3.2 + translation.2)
  in mp_s4 = (1, -2) :=
by
  let p1 := (3 : ℝ, -2 : ℝ)
  let p2 := (-7 : ℝ, 6 : ℝ)
  let mp_s3 := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let translation := (3, -4)
  let mp_s4 := (mp_s3.1 + translation.1, mp_s3.2 + translation.2)
  show mp_s4 = (1, -2)
  sorry

end midpoint_translation_l88_88672


namespace binom_20_10_l88_88901

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l88_88901


namespace domain_width_of_g_l88_88546

theorem domain_width_of_g
  (h : ℝ → ℝ)
  (domain_h : ∀ x, -10 ≤ x ∧ x ≤ 10 → x ∈ (set.univ : set ℝ))
  (g : ℝ → ℝ := fun x => h (x / 3)) :
  set.Icc (-30 : ℝ) 30 = (set.Icc (-10:ℝ) 10).preimage (λ x => x / 3) → 
  width (set.Icc (-30 : ℝ) 30) = 60 :=
sorry

end domain_width_of_g_l88_88546


namespace number_of_circles_l88_88550

noncomputable def parabola_focus : Point := (1, 0) -- Focus of the parabola y^2 = 4x
noncomputable def parabola_directrix : Line := {x | x = -1} -- Directrix of the parabola y^2 = 4x
noncomputable def M : Point := (1, 2) -- Point M(1, 2) on the parabola

theorem number_of_circles : 
  let F := parabola_focus
  let l := parabola_directrix in
  (∃ (C : Circle), C.passes_through F ∧ C.passes_through M ∧ C.tangent_to l) = 4 := 
sorry

end number_of_circles_l88_88550


namespace set_prod_l88_88543

def A : set ℝ := {x : ℝ | |x - 1 / 2| < 1}

def B : set ℝ := {x : ℝ | 1 / x ≥ 1}

def set_union_inter_diff (A B : set ℝ) : set ℝ :=
  (A ∪ B) \ (A ∩ B)

theorem set_prod (x : ℝ) (hx : x ∈ set_union_inter_diff A B) : 
  x ∈ Icc (-1/2) 0 ∪ Ioo 1 (3/2) :=
by sorry

end set_prod_l88_88543


namespace count_four_digit_integers_with_product_18_l88_88968

def valid_digits (n : ℕ) : Prop := 
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def digit_product_18 (a b c d : ℕ) : Prop := 
  a * b * c * d = 18

def four_digit_integer (a b c d : ℕ) : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d

theorem count_four_digit_integers_with_product_18 : 
  (∑ a b c d in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    ite (four_digit_integer a b c d ∧ digit_product_18 a b c d) 1 0) = 48 := 
sorry

end count_four_digit_integers_with_product_18_l88_88968


namespace prove_median_l88_88386

def mode (s : List ℕ) (x : ℕ) : Prop :=
    s.count x > ∀ y, y ≠ x → s.count y

def is_median (s : List ℕ) (m : ℕ) : Prop :=
    let sorted_s := s.sort
    if sorted_s.length % 2 = 1 then
      m = sorted_s.get ((sorted_s.length - 1) / 2)
    else
      (m = (sorted_s.get (sorted_s.length / 2 - 1) + sorted_s.get (sorted_s.length / 2)) / 2)

theorem prove_median :
  ∀ x : ℕ, mode [2, 3, 6, 7, 8, x] x → (1 / 6 * x - 4 > 0) →
  is_median [2, 3, 6, 7, 8, x] (9 / 2) ∨ is_median [2, 3, 6, 7, 8, x] 6 :=
by
  intros x h_mode h_ineq
  sorry

end prove_median_l88_88386


namespace sum_fraction_result_l88_88414

theorem sum_fraction_result :
  ∑ n in Finset.range 100 + 1, (1 / (n^3 + n^2)) = 
    1 - (∑ n in Finset.range 100 + 1, (1 / (n^2))) - (1 / 101) := 
by
    sorry

end sum_fraction_result_l88_88414


namespace not_lucky_1994_l88_88799

def is_valid_month (m : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12

def is_valid_day (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 31

def is_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), is_valid_month m ∧ is_valid_day d ∧ m * d = y

theorem not_lucky_1994 : ¬ is_lucky_year 94 := 
by
  sorry

end not_lucky_1994_l88_88799


namespace pie_chart_representation_l88_88001

-- Define the conditions
def red_portion (blue: ℝ) := 3 * blue
def green_portion (blue: ℝ) := blue
def yellow_portion (blue: ℝ) := 0.5 * blue

-- State of the theorem 
theorem pie_chart_representation :
  ∀ (blue : ℝ), blue = 1 ->
  (red_portion blue = 3) ∧ (green_portion blue = 1) ∧ (yellow_portion blue = 0.5) := by 
  intros blue h
  rw h
  split
  { rw red_portion; norm_num }
  { split
    { rw green_portion; exact h }
    { rw yellow_portion; norm_num } }
  sorry

end pie_chart_representation_l88_88001


namespace three_heads_in_a_row_l88_88308

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l88_88308


namespace minimal_base_for_magic_square_l88_88841

theorem minimal_base_for_magic_square : ∃ b : ℕ, ∀ (r : ℕ), r ∈ {0, 1, 2, 3} → 
  let num := [[2, 20, 21], [5, 20, 22], [7, 24, 3]] in
  let row_sum := num[r][0] + 2*b + 2*b+1 + num[r][2],
   let col_sum := 7 + num[1][0] + num[2][0] in
   let diag1_sum := 2 + (2*b) + (2*b+1),
   let diag2_sum := num[2][0] + (2*b+1) + num[0][2] → 
   b = 5 ∧ (row_sum = col_sum ∧ row_sum = diag1_sum ∧ row_sum = diag2_sum) :=
begin
  sorry
end

end minimal_base_for_magic_square_l88_88841


namespace answer_l88_88577

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

end answer_l88_88577


namespace sum_of_first_17_terms_l88_88147

variable {α : Type*} [LinearOrderedField α] 

-- conditions
def arithmetic_sequence (a : ℕ → α) : Prop := 
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

variable {a : ℕ → α}
variable {S : ℕ → α}

-- main theorem
theorem sum_of_first_17_terms (h_arith : arithmetic_sequence a)
  (h_S : sum_of_first_n_terms a S)
  (h_condition : a 7 + a 12 = 12 - a 8) :
  S 17 = 68 := sorry

end sum_of_first_17_terms_l88_88147


namespace residue_11_pow_1234_mod_19_l88_88281

theorem residue_11_pow_1234_mod_19 : 
  (11 ^ 1234) % 19 = 11 := 
by
  sorry

end residue_11_pow_1234_mod_19_l88_88281


namespace midpoint_product_l88_88194

theorem midpoint_product (x' y' : ℤ) 
  (h1 : (0 + x') / 2 = 2) 
  (h2 : (9 + y') / 2 = 4) : 
  (x' * y') = -4 :=
by
  sorry

end midpoint_product_l88_88194


namespace Jonathan_typing_time_l88_88616

variable (J : ℝ) -- The time it takes Jonathan to type the document alone.
def rate_Susan : ℝ := 1/30  -- Susan's typing rate.
def rate_Jack : ℝ := 1/24  -- Jack's typing rate.
def rate_combined : ℝ := 1/10  -- Combined typing rate when they work together.

theorem Jonathan_typing_time (h : (1/J) + rate_Susan + rate_Jack = rate_combined) : J = 40 :=
by {
  sorry -- Proof goes here.
}

end Jonathan_typing_time_l88_88616


namespace employees_in_four_restaurants_l88_88402

-- Definitions
def total_employees : ℕ := 75
def buffet_employees : ℕ := 35
def fine_dining_employees : ℕ := 28
def snack_bar_employees : ℕ := 22
def seafood_shack_employees : ℕ := 16
def international_employees : ℕ := 12
def two_restaurants_employees : ℕ := 10
def three_restaurants_employees : ℕ := 7

-- Main theorem statement
theorem employees_in_four_restaurants :
  let x := total_employees - (buffet_employees + fine_dining_employees + snack_bar_employees + seafood_shack_employees + international_employees - two_restaurants_employees - 2 * three_restaurants_employees) / 3
  in x = 0 :=
by {
  -- Dummy proof to ensure the theorem statement is syntactically correct
  sorry
}

end employees_in_four_restaurants_l88_88402


namespace cheese_partition_into_equal_weight_groups_l88_88724

-- Given six pieces of cheese with different weights
variable (cheese : Fin 6 → ℕ)

-- Conditions:
-- For any two pieces, we can identify the heavier piece (implied by weights being different)
variable (weights_different : ∀ i j : Fin 6, i ≠ j → cheese i ≠ cheese j)

-- We need to divide them into two groups of equal weight with three pieces each
theorem cheese_partition_into_equal_weight_groups:
  (∃ (A B : Finset (Fin 6)), A.card = 3 ∧ B.card = 3 ∧ A ≠ B ∧ A ∪ B = Finset.univ ∧ (∑ i in A, cheese i) = (∑ j in B, cheese j)) :=
by
  -- to be proved
  sorry

end cheese_partition_into_equal_weight_groups_l88_88724


namespace ordered_triples_lcm_count_l88_88984

theorem ordered_triples_lcm_count :
  let x, y, z : ℕ+ in
  ((∀ x y z : ℕ+, Nat.lcm x y = 120 ∧ Nat.lcm x z = 450 ∧ Nat.lcm y z = 180) → 
  (∃ (n : ℕ), n = 6)) :=
by 
  sorry

end ordered_triples_lcm_count_l88_88984


namespace minimum_area_l88_88703

noncomputable def f (x : ℝ) : ℝ :=
  (x^3)/2 + 1 - x * ∫ t in 0..x, g(t)

noncomputable def g (x : ℝ) : ℝ :=
  x - ∫ t in 0..1, f(t)

theorem minimum_area (a : ℝ) : 
  let l₁ := λ x, f(a) + f'(a) * (x - a)
  let l₂ := λ x, f(a) + f'(a) * (x - a)
  minimum_area_between_tangents_and_curve f l₁ l₂ = 0 :=
sorry

end minimum_area_l88_88703


namespace sum_of_squares_ending_with_09_eq_253_l88_88738

theorem sum_of_squares_ending_with_09_eq_253 :
  let two_digit_positive_integers := {n : ℕ | 10 ≤ n ∧ n < 100}
  let squares_end_with_09 := {n : ℕ | n ∈ two_digit_positive_integers ∧ (n^2 % 100 = 9)}
  finset.sum (finset.filter (λ n, n ∈ squares_end_with_09) (finset.range 100)) id = 253 :=
by
  sorry

end sum_of_squares_ending_with_09_eq_253_l88_88738


namespace triangle_area_from_squares_l88_88951

theorem triangle_area_from_squares (A B C : ℝ) (hA : A = 64) (hB : B = 225) (hC : C = 289) :
  ∃ (S : ℝ), S = 60 ∧ (let a := Real.sqrt A; let b := Real.sqrt B; let c := Real.sqrt C in a^2 + b^2 = c^2) := 
by
  use 60
  split
  · rfl
  · let a := Real.sqrt 64
    let b := Real.sqrt 225
    let c := Real.sqrt 289
  sorry

end triangle_area_from_squares_l88_88951


namespace exam_passed_percentage_l88_88569

theorem exam_passed_percentage (failed_students total_students : ℕ) (h_failed : failed_students = 455) (h_total : total_students = 700) :
  (total_students - failed_students) * 100 / total_students = 35 :=
by
  rw [h_failed, h_total]
  norm_num
  sorry

end exam_passed_percentage_l88_88569


namespace find_k_value_l88_88187

variable (S : ℕ → ℤ) (n : ℕ)

-- Conditions
def is_arithmetic_sum (S : ℕ → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ n : ℕ, S n = n * (2 * a + (n - 1) * d) / 2

axiom S3_eq_S8 (S : ℕ → ℤ) (hS : is_arithmetic_sum S) : S 3 = S 8
axiom Sk_eq_S7 (S : ℕ → ℤ) (k : ℕ) (hS: is_arithmetic_sum S)  : S 7 = S k

theorem find_k_value (S : ℕ → ℤ) (hS: is_arithmetic_sum S) :  S 3 = S 8 → S 7 = S 4 :=
by
  sorry

end find_k_value_l88_88187


namespace total_students_l88_88687

theorem total_students (N : ℕ)
  (h1 : (84 + 128 + 13 = 15 * N))
  : N = 15 :=
sorry

end total_students_l88_88687


namespace tan_ABO_eq_half_l88_88103

variables {xA xB : ℝ} (hA_pos : xA > 0) (hB_neg : xB < 0)
def yA := 1 / xA
def yB := -4 / xB

-- Slope of OA
def m1 := yA / xA

-- Slope of OB
def m2 := yB / xB

-- Given OA ⊥ OB
def perpendicular_slopes : Prop := m1 * m2 = -1

-- Calculating the tangent of the angle between AB and OA
def tan_angle_ABO : ℝ := abs (yB / xB)

theorem tan_ABO_eq_half :
  perpendicular_slopes →
  tan_angle_ABO = 1 / 2 := 
sorry

end tan_ABO_eq_half_l88_88103


namespace janet_rose_shampoo_l88_88176

theorem janet_rose_shampoo :
  ∃ (R : ℚ), (R + 1/4 = 7 * (1/12)) ∧ (R = 1/3) :=
by
  apply Exists.intro (1/3)
  split
  · have h1 : 7 * (1/12) = 7/12 := by norm_num
    rw h1
    norm_num
  · norm_num

end janet_rose_shampoo_l88_88176


namespace probability_distribution_correct_l88_88240

noncomputable def probability_of_hit : ℝ := 0.1
noncomputable def probability_of_miss : ℝ := 1 - probability_of_hit

def X_distribution : Fin 4 → ℝ
| ⟨3, _⟩ => probability_of_hit
| ⟨2, _⟩ => probability_of_miss * probability_of_hit
| ⟨1, _⟩ => probability_of_miss^2 * probability_of_hit
| ⟨0, _⟩ => probability_of_miss^3 * probability_of_hit + probability_of_miss^4

theorem probability_distribution_correct :
  X_distribution ⟨0, by simp⟩ = 0.729 ∧
  X_distribution ⟨1, by simp⟩ = 0.081 ∧
  X_distribution ⟨2, by simp⟩ = 0.09 ∧
  X_distribution ⟨3, by simp⟩ = 0.1 :=
by
  sorry

end probability_distribution_correct_l88_88240


namespace no_integer_right_triangle_side_x_l88_88871

theorem no_integer_right_triangle_side_x :
  ∀ (x : ℤ), (12 + 30 > x ∧ 12 + x > 30 ∧ 30 + x > 12) →
             (12^2 + 30^2 = x^2 ∨ 12^2 + x^2 = 30^2 ∨ 30^2 + x^2 = 12^2) →
             (¬ (∃ x : ℤ, 18 < x ∧ x < 42)) :=
by
  sorry

end no_integer_right_triangle_side_x_l88_88871


namespace isoland_license_plates_proof_l88_88693

def isoland_license_plates : ℕ :=
  let letters := ['A', 'B', 'D', 'E', 'I', 'L', 'N', 'O', 'R', 'U']
  let valid_letters := letters.erase 'B'
  let first_letter_choices := ['A', 'I']
  let last_letter := 'R'
  let remaining_letters:= valid_letters.erase last_letter
  (first_letter_choices.length * (remaining_letters.length - first_letter_choices.length) * (remaining_letters.length - first_letter_choices.length - 1) * (remaining_letters.length - first_letter_choices.length - 2))

theorem isoland_license_plates_proof :
  isoland_license_plates = 420 := by
  sorry

end isoland_license_plates_proof_l88_88693


namespace max_knights_on_island_l88_88207

theorem max_knights_on_island :
  ∃ n x, (n * (n - 1) = 90) ∧ (x * (10 - x) = 24) ∧ (x ≤ n) ∧ (∀ y, y * (10 - y) = 24 → y ≤ x) := sorry

end max_knights_on_island_l88_88207


namespace sum_of_squares_ending_with_09_eq_253_l88_88739

theorem sum_of_squares_ending_with_09_eq_253 :
  let two_digit_positive_integers := {n : ℕ | 10 ≤ n ∧ n < 100}
  let squares_end_with_09 := {n : ℕ | n ∈ two_digit_positive_integers ∧ (n^2 % 100 = 9)}
  finset.sum (finset.filter (λ n, n ∈ squares_end_with_09) (finset.range 100)) id = 253 :=
by
  sorry

end sum_of_squares_ending_with_09_eq_253_l88_88739


namespace frank_hawaiian_slices_l88_88422

theorem frank_hawaiian_slices:
  ∀ (total_slices dean_slices sammy_slices leftover_slices frank_slices : ℕ),
  total_slices = 24 →
  dean_slices = 6 →
  sammy_slices = 4 →
  leftover_slices = 11 →
  (total_slices - leftover_slices) = (dean_slices + sammy_slices + frank_slices) →
  frank_slices = 3 :=
by
  intros total_slices dean_slices sammy_slices leftover_slices frank_slices
  intros h_total h_dean h_sammy h_leftovers h_total_eaten
  sorry

end frank_hawaiian_slices_l88_88422


namespace find_speeds_l88_88476

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l88_88476


namespace total_tiles_in_square_hall_l88_88370

theorem total_tiles_in_square_hall
  (s : ℕ) -- integer side length of the square hall
  (black_tiles : ℕ)
  (total_tiles : ℕ)
  (all_tiles_white_or_black : ∀ (x : ℕ), x ≤ total_tiles → x = black_tiles ∨ x = total_tiles - black_tiles)
  (black_tiles_count : black_tiles = 153 + 3) : total_tiles = 6084 :=
by
  sorry

end total_tiles_in_square_hall_l88_88370


namespace theater_earnings_l88_88376

theorem theater_earnings :
  let matinee_price := 5
  let evening_price := 7
  let opening_night_price := 10
  let popcorn_price := 10
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let half_of_customers_that_bought_popcorn := 
    (matinee_customers + evening_customers + opening_night_customers) / 2
  let total_earnings := 
    (matinee_price * matinee_customers) + 
    (evening_price * evening_customers) + 
    (opening_night_price * opening_night_customers) + 
    (popcorn_price * half_of_customers_that_bought_popcorn)
  total_earnings = 1670 :=
by
  sorry

end theater_earnings_l88_88376


namespace sum_difference_even_odd_first_100_l88_88034

def nth_odd (n : ℕ) : ℕ := 2*n - 1
def nth_even (n : ℕ) : ℕ := 2*n

theorem sum_difference_even_odd_first_100 : 
  (∑ k in finset.range 100, nth_even (k + 1)) - (∑ k in finset.range 100, nth_odd (k + 1)) = 100 :=
by 
  sorry

end sum_difference_even_odd_first_100_l88_88034


namespace prob_first_three_heads_all_heads_l88_88315

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l88_88315


namespace find_an_and_Sn_find_Tn_l88_88086

-- Define the arithmetic sequence with the given conditions
def a_seq (n : ℕ) : ℕ := 2 * n

-- Define the sum of the first n terms of the arithmetic sequence
def S_seq (n : ℕ) : ℕ := n * n + n

-- Define the sequence b_n based on a_n
def b_seq (n : ℕ) : ℚ := 1 / (a_seq n ^ 2 - 1)

-- Define the sum of the first n terms of the sequence b_n
def T_seq (n : ℕ) : ℚ := (1 / 2) * (1 - 1 / (2 * n + 1))

-- Theorem statements
theorem find_an_and_Sn (n : ℕ) : a_seq n = 2 * n ∧ S_seq n = n * n + n := by
  sorry

theorem find_Tn (n : ℕ) : ∑ i in Finset.range n, b_seq (i + 1) = n / (2 * n + 1) := by
  sorry

end find_an_and_Sn_find_Tn_l88_88086


namespace normal_chores_per_week_is_12_l88_88056

variable (C : ℕ) -- number of chores per week normally
variable (total_chores_two_weeks : ℕ) (extra_chores_two_weeks : ℕ)
variable (ratio : ℕ)

-- Given conditions
def given_conditions : Prop :=
  total_chores_two_weeks = 56 ∧ extra_chores_two_weeks = 32 ∧ ratio = 2

-- Theorem: Prove that Edmund normally does 12 chores a week.
theorem normal_chores_per_week_is_12 (h : given_conditions) : C = 12 := by
  sorry

end normal_chores_per_week_is_12_l88_88056


namespace total_prime_dates_in_non_leap_year_l88_88156

def prime_dates_in_non_leap_year (days_in_months : List (Nat × Nat)) : Nat :=
  let prime_numbers := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  days_in_months.foldl 
    (λ acc (month, days) => 
      acc + (prime_numbers.filter (λ day => day ≤ days)).length) 
    0

def month_days : List (Nat × Nat) :=
  [(2, 28), (3, 31), (5, 31), (7, 31), (11,30)]

theorem total_prime_dates_in_non_leap_year : prime_dates_in_non_leap_year month_days = 52 :=
  sorry

end total_prime_dates_in_non_leap_year_l88_88156


namespace prob_three_heads_is_one_eighth_l88_88331

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l88_88331


namespace ratio_meerkats_to_lion_cubs_l88_88003

-- Defining the initial conditions 
def initial_animals : ℕ := 68
def gorillas_sent : ℕ := 6
def hippo_adopted : ℕ := 1
def rhinos_rescued : ℕ := 3
def lion_cubs : ℕ := 8
def final_animal_count : ℕ := 90

-- Calculating the number of meerkats
def animals_before_meerkats : ℕ := initial_animals - gorillas_sent + hippo_adopted + rhinos_rescued + lion_cubs
def meerkats : ℕ := final_animal_count - animals_before_meerkats

-- Proving the ratio of meerkats to lion cubs is 2:1
theorem ratio_meerkats_to_lion_cubs : meerkats / lion_cubs = 2 := by
  -- Placeholder for the proof
  sorry

end ratio_meerkats_to_lion_cubs_l88_88003


namespace Teena_speed_is_55_l88_88230

def Teena_speed (Roe_speed T : ℝ) (initial_gap final_gap time : ℝ) : Prop :=
  Roe_speed * time + initial_gap + final_gap = T * time

theorem Teena_speed_is_55 :
  Teena_speed 40 55 7.5 15 1.5 :=
by 
  sorry

end Teena_speed_is_55_l88_88230


namespace total_path_length_B_travels_l88_88401

theorem total_path_length_B_travels
  (radius : ℝ)
  (arc_deg : ℝ)
  (pivot_repeats : ℕ)
  (BC_radius : radius = 1)
  (arc_deg_val : arc_deg = 45)
  (pivot_repeats_val : pivot_repeats = 4) :
  let arc_length := (arc_deg / 360) * (2 * Real.pi * radius)
  in (pivot_repeats / 2) * arc_length = Real.pi :=
by
  sorry

end total_path_length_B_travels_l88_88401


namespace length_of_symmedian_l88_88848

theorem length_of_symmedian (a b c : ℝ) (AS : ℝ) :
  AS = (2 * b * c^2) / (b^2 + c^2) := sorry

end length_of_symmedian_l88_88848


namespace Sn_geom_seq_l88_88609

theorem Sn_geom_seq (a : ℕ → ℝ) (S_n : ℕ → ℝ) (λ : ℝ) (h1 : a 1 = 2)
    (h2 : ∀ n, a (n + 1) ^ 2 = a n * a (n + 2))
    (h3 : ∀ n, (a (n + 1) + λ) ^ 2 = (a n + λ) * (a (n + 2) + λ)) 
    (h4 : λ ≠ 0):
  S_n = λ (n : ℕ), 2 * n :=
by
  sorry

end Sn_geom_seq_l88_88609


namespace probability_of_exactly_3_out_of_5_winners_l88_88711

theorem probability_of_exactly_3_out_of_5_winners :
  let p := 0.9 in
  let n := 5 in
  let k := 3 in
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (nat.choose 5 3) * (0.9^3) * (0.1^2) :=
by
  let p := 0.9
  let n := 5
  let k := 3
  have h1: nat.choose n k = nat.choose 5 3 := by simp
  have h2: p^k = 0.9^3 := by simp
  have h3: (1 - p)^(n - k) = 0.1^2 := by simp
  rw [h1, h2, h3]
  sorry

end probability_of_exactly_3_out_of_5_winners_l88_88711


namespace exclude_chairs_l88_88344

-- Definitions
def total_chairs : ℕ := 10000
def perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Statement
theorem exclude_chairs (n : ℕ) (h₁ : n = total_chairs) :
  perfect_square n → (n - total_chairs) = 0 := 
sorry

end exclude_chairs_l88_88344


namespace area_ratio_triangle_l88_88090

open EuclideanGeometry

variables {A B C P : Point}

theorem area_ratio_triangle
  (h1 : collinear {A, B, C})
  (h2 : P ∈ Plane A B C)
  (h3 : (line_vector A P) + (line_vector B P) + (line_vector C P) = (line_vector A B)) :
  area (triangle A B P) / area (triangle B C P) = 1 / 2 := 
sorry

end area_ratio_triangle_l88_88090


namespace general_formula_sequence_l88_88720

theorem general_formula_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 2 * n^2 + n + 1) →
  (a 1 = S 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n - 1)) →
  (∀ n, a n = if n = 1 then 4 else 4 * n - 2) :=
by
  intro hS ha1 ha2
  funext n
  sorry

end general_formula_sequence_l88_88720


namespace probability_two_heads_in_a_row_l88_88413

open ProbabilityTheory

noncomputable def probability_two_heads_after_HTH : ℚ :=
  (1 / 2) ^ 3 * Q

theorem probability_two_heads_in_a_row
  (coin : ℕ → bool)
  (fair_coin : ∀ n, coin n = tt ∨ coin n = ff)
  (initial_sequence : list bool)
  (ht_sequence : initial_sequence = [tt, ff, tt])
  (stop_after_second_HH_or_TT : ∃ n, ((coin n = coin (n + 1) ∧ (coin n = tt ∨ coin n = ff)) ∧ (coin (n + 2) = tt ∨ coin (n + 2) = ff))) :
  probability_two_heads_after_HTH = 1 / 64 := sorry

end probability_two_heads_in_a_row_l88_88413


namespace complex_div_product_l88_88987

theorem complex_div_product (a b : ℝ) (h : (1 + 7 * complex.I) / (2 - complex.I) = a + b * complex.I) : a * b = -3 := 
by
  sorry

end complex_div_product_l88_88987


namespace expand_count_terms_l88_88133

theorem expand_count_terms : 
  ∃ n, n = 4033 ∧ ∀ a b c : ℕ, a + b + c = 2016 → 
  let exponents := 3 * a - 3 * b in 
  -2016 ≤ exponents ∧ exponents ≤ 2016 := 
sorry

end expand_count_terms_l88_88133


namespace sum_of_divisors_eq_600_l88_88249

variable (i j : ℕ)

theorem sum_of_divisors_eq_600 (i j : ℕ) (h1 : 2 ^ i * 3 ^ j) 
(h2 : (∑ k in Finset.range (i + 1), (2 : ℕ) ^ k) * (∑ k in Finset.range (j + 1), (3 : ℕ) ^ k) = 600) : 
  i + j = 6 := 
sorry

end sum_of_divisors_eq_600_l88_88249


namespace sum_count_x_y_l88_88757

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_count_x_y :
  let x := sum_integers 40 60 in
  let y := count_even_integers 40 60 in
  x + y = 1061 :=
by
  sorry

end sum_count_x_y_l88_88757


namespace lines_concurrent_l88_88567

noncomputable def concurrent_points
  (A B C D E F A1 C1 : Type)
  [CoordSheaf A B C D E F A1 C1] 
  (circle₁ : Circle A1 E F A) 
  (circle₂ : Circle C1 E F C) 
  (trapezoid : Trapezoid ABCD AD_parallel_BC)
  (points_on_AB : OnLine E AB)
  (points_on_CD : OnLine F CD)
  (points_on_AD : OnLine A1 AD)
  (points_on_BC : OnLine C1 BC) : Prop := 
  are_concurrent (Line A1 C1) (Line BD) (Line EF) 

-- Statement of the theorem
theorem lines_concurrent
  {A B C D E F A1 C1 : Type}
  [CoordSheaf A B C D E F A1 C1]
  (trapezoid : Trapezoid ABCD AD_parallel_BC)
  (points_on_AB : OnLine E AB)
  (points_on_CD : OnLine F CD)
  (circle₁ : Circle A1 E F A) 
  (circle₂ : Circle C1 E F C)
  (points_on_AD : OnLine A1 AD)
  (points_on_BC : OnLine C1 BC) :
  concurrent_points A B C D E F A1 C1 circle₁ circle₂ trapezoid points_on_AB points_on_CD points_on_AD points_on_BC := 
sorry

end lines_concurrent_l88_88567


namespace handrail_length_proof_l88_88006

def handrail_length (height radius : ℝ) (turn_deg : ℝ) : ℝ :=
  let arc_length := turn_deg / 360 * 2 * Real.pi * radius
  Real.sqrt (height^2 + arc_length^2)

theorem handrail_length_proof :
  handrail_length 15 3 180 = 15.9 := 
by
  sorry

end handrail_length_proof_l88_88006


namespace hyperbola_equation_and_conditions_l88_88885

-- Define the hyperbola and its conditions
def hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Define the eccentricity condition
def eccentricity_condition (a e : ℝ) : Prop :=
  e = real.sqrt 3 ∧ a > 0 ∧ a = 1

-- Coordinates of the left vertex
def left_vertex_condition : Prop :=
  (-1 : ℝ, 0)

-- Define the line equation
def line_eq (x y m : ℝ) : Prop :=
  x - y + m = 0

-- Define the circle where the midpoint of AB lies
def midpoint_circle (x y: ℝ) : Prop :=
  x^2 + y^2 = 5

-- Define the theorem to be proved
theorem hyperbola_equation_and_conditions (a b e m : ℝ) (x y : ℝ) :
  hyperbola x y a b →
  eccentricity_condition a e →
  line_eq x y m →
  midpoint_circle m 0 →
  x^2 - y^2 / 2 = 1 ∧ (m = 1 ∨ m = -1) ∧ ∃ A B : ℝ × ℝ, 4 * real.sqrt 2 = dist A B := sorry

end hyperbola_equation_and_conditions_l88_88885


namespace simplify_abs_expression_l88_88078

theorem simplify_abs_expression (a b c : ℝ) (h1 : a + c > b) (h2 : b + c > a) (h3 : a + b > c) :
  |a - b + c| - |a - b - c| = 2 * a - 2 * b :=
by
  sorry

end simplify_abs_expression_l88_88078


namespace max_planes_six_points_l88_88049

theorem max_planes_six_points (P : Finset (Fin 6 → ℝ)) :
  (∀ p ∈ P.powerset, p.card ≥ 4 → ∃ plane : Finset (Fin 6 → ℝ), plane.card ≥ 4 ∧ p ⊆ plane)
  ∧ (∀ p ∈ (P.powerset.filter (λ p, p.card = 4)), ¬Collinear ℝ p) → (∃! n, n = 6) :=
sorry

variables (ℝ : Type) [field ℝ] [vector_space ℝ (Fin 6 → ℝ)] 
variables [add_comm_group (Fin 6 → ℝ)] [module ℝ (Fin 6 → ℝ)]

namespace collinear_detection

-- Assume collinearity detection module

def Collinear (K : Type*) [field K] (s : Finset (Fin 6 → K)) : Prop :=
∃ l : (Fin 6 → K), ∀ p ∈ s, ∃ k : K, p = k • l

end collinear_detection

end max_planes_six_points_l88_88049


namespace total_items_sold_l88_88009

theorem total_items_sold (x : ℕ) (h : 59.8 * x = 2550) : 3 * 42 = 126 :=
by
  sorry

end total_items_sold_l88_88009


namespace simplify_expression_l88_88847

theorem simplify_expression (a : ℝ) (h : a ≠ 1/2) : 1 - (2 / (1 + (2 * a) / (1 - 2 * a))) = 4 * a - 1 :=
by
  sorry

end simplify_expression_l88_88847


namespace num_zeros_in_expansion_l88_88542

theorem num_zeros_in_expansion : 
  (let n := 10^12 - 1 in (n * n).toString.filter (· == '0')).length = 12 := 
sorry

end num_zeros_in_expansion_l88_88542


namespace sum_of_areas_of_triangles_in_cube_l88_88411

theorem sum_of_areas_of_triangles_in_cube :
  let m := 1008
  let n := 144^2 * 2
  let p := 216^2 * 3
  1008 + 144 * Real.sqrt 2 + 216 * Real.sqrt 3 = 1008 + Real.sqrt n + Real.sqrt p :=
by
  let m := 1008
  let n := 20736 * 2
  let p := 46656 * 3
  sorry

end sum_of_areas_of_triangles_in_cube_l88_88411


namespace total_exercises_l88_88811

theorem total_exercises (pts_needed : ℕ) (n : ℕ) (grp_size : ℕ) (ex_incr : ℕ) :
  pts_needed = 30 → grp_size = 6 → ex_incr = 1 →
  (n = pts_needed / grp_size) →
  (Σ i in Finset.range n, grp_size * (i + 1)) = 90 :=
by
  intro h1 h2 h3 h4
  sorry

end total_exercises_l88_88811


namespace tommy_house_price_l88_88264

variable (P : ℝ)

theorem tommy_house_price 
  (h1 : 1.25 * P = 125000) : 
  P = 100000 :=
by
  sorry

end tommy_house_price_l88_88264


namespace set_intersection_complement_eq_l88_88531

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem set_intersection_complement_eq :
  U = {1, 2, 3, 4, 5, 6} → 
  A = {2, 3} →
  B = {3, 5} →
  A ∩ (U \ B) = {2} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  simp
  sorry

end set_intersection_complement_eq_l88_88531


namespace clock_angle_at_315_l88_88025

theorem clock_angle_at_315 : 
  let minute_hand_angle := 90
  let hour_hand_angle := 90 + (3 / 4) * 30
  abs (minute_hand_angle - hour_hand_angle) = 22.5 :=
by
  sorry

end clock_angle_at_315_l88_88025


namespace staircase_handrail_length_l88_88008

/--
Given a spiral staircase that turns 180 degrees as it rises 15 feet and has a radius of 3 feet,
prove the length of the handrail is approximately 17.7 feet.
-/
theorem staircase_handrail_length :
  let arc_length := 3 * Real.pi in
  let handrail_length := Real.sqrt (15^2 + arc_length^2) in
  handrail_length ≈ 17.7 := by
  sorry

end staircase_handrail_length_l88_88008


namespace toll_calculation_correct_l88_88721

theorem toll_calculation_correct :
  ∀ (toll_weekday toll_weekend: ℕ → ℝ) (num_wheels front_wheels wheels_per_axle : ℕ),
  (∀ x, toll_weekday x = 2.50 + 0.70 * (x - 1)) →
  (∀ x, toll_weekend x = 3.00 + 0.80 * (x - 1)) →
  num_wheels = 18 →
  front_wheels = 2 →
  wheels_per_axle = 4 →
  let num_other_axles := (num_wheels - front_wheels) / wheels_per_axle in
  let num_axles := num_other_axles + 1 in
  (5 * toll_weekday num_axles) + (2 * toll_weekend num_axles) = 38.90 :=
by 
  intros toll_weekday toll_weekend num_wheels front_wheels wheels_per_axle h_toll_weekday h_toll_weekend h_num_wheels h_front_wheels h_wheels_per_axle;
  let num_other_axles := (num_wheels - front_wheels) / wheels_per_axle;
  let num_axles := num_other_axles + 1;
  sorry

end toll_calculation_correct_l88_88721


namespace red_cards_pick_ordered_count_l88_88363

theorem red_cards_pick_ordered_count :
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  (red_cards * (red_cards - 1) = 552) :=
by
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  show (red_cards * (red_cards - 1) = 552)
  sorry

end red_cards_pick_ordered_count_l88_88363


namespace probability_of_first_three_heads_l88_88297

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l88_88297


namespace binom_20_10_l88_88903

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l88_88903


namespace average_minutes_run_per_day_l88_88403

theorem average_minutes_run_per_day (f : ℕ) :
  let third_grade_minutes := 12
  let fourth_grade_minutes := 15
  let fifth_grade_minutes := 10
  let third_graders := 4 * f
  let fourth_graders := 2 * f
  let fifth_graders := f
  let total_minutes := third_graders * third_grade_minutes + fourth_graders * fourth_grade_minutes + fifth_graders * fifth_grade_minutes
  let total_students := third_graders + fourth_graders + fifth_graders
  total_minutes / total_students = 88 / 7 :=
by
  sorry

end average_minutes_run_per_day_l88_88403


namespace ellipse_equation_standard_form_l88_88105

-- Define the conditions under which the problem is set
def ellipse_passes_through (A : ℝ × ℝ) (x y a b : ℝ) : Prop :=
  (A.1 / a)^2 + (A.2 / b)^2 = 1

def right_focus (F : ℝ × ℝ) (c : ℝ) : Prop :=
  F.1 = c

-- Define the proposition to be proved
theorem ellipse_equation_standard_form :
  ∃ (a b : ℝ),
  (a > 0 ∧ b > 0) ∧
  ellipse_passes_through (2, 3) 2 3 4 (real.sqrt 12) ∧
  right_focus (2, 0) 2 ∧
  (a^2 = 16 ∧ b^2 = 12) ∧
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) :=
begin
  sorry
end

end ellipse_equation_standard_form_l88_88105


namespace tan_alpha_value_l88_88923

theorem tan_alpha_value {α : ℝ} (h1 : sin α + cos α = 7/5) (h2 : sin α > cos α) : tan α = 4/3 :=
sorry

end tan_alpha_value_l88_88923


namespace ratio_volume_surface_area_l88_88010

noncomputable def volume : ℕ := 10
noncomputable def surface_area : ℕ := 45

theorem ratio_volume_surface_area : volume / surface_area = 2 / 9 := by
  sorry

end ratio_volume_surface_area_l88_88010


namespace smallest_x_y_sum_l88_88501

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hne : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 24) :
  x + y = 100 :=
sorry

end smallest_x_y_sum_l88_88501


namespace allan_initial_balloons_l88_88397

theorem allan_initial_balloons :
  ∃ (A : ℕ), A + 3 + 1 = 6 ∧ A = 2 :=
begin
  use 2,
  split,
  {
    show 2 + 3 + 1 = 6,
    calc 2 + 3 + 1 = 6 : by norm_num,
  },
  {
    refl,
  }
end


end allan_initial_balloons_l88_88397


namespace obtuse_quadrilateral_maximum_obtuse_angles_l88_88128

def is_obtuse_angle (θ : ℝ) : Prop := θ > 90

def is_quadrilateral (angles : list ℝ) : Prop :=
  angles.length = 4 ∧ angles.sum = 360

def is_obtuse_quadrilateral (angles : list ℝ) : Prop :=
  is_quadrilateral angles ∧ (angles.any is_obtuse_angle)

theorem obtuse_quadrilateral_maximum_obtuse_angles (angles : list ℝ) :
  is_obtuse_quadrilateral angles → (angles.filter is_obtuse_angle).length ≤ 2 :=
sorry

end obtuse_quadrilateral_maximum_obtuse_angles_l88_88128


namespace number_of_proper_subsets_l88_88857

open Set

noncomputable def M : Set ℤ := {x | -4 < x - 1 ∧ x - 1 < 4 ∧ x ≠ 1}

theorem number_of_proper_subsets : ∀ M : Set ℤ, 
  (M = {x | -4 < x - 1 ∧ x - 1 < 4 ∧ x ≠ 1}) → (nat.pow 2 (card M) - 1 = 63) := 
by
  sorry

end number_of_proper_subsets_l88_88857


namespace polygon_independence_l88_88455

theorem polygon_independence (P : Type*) [polygon P] :
  ∀ (pentagon : P) (P1 : ∀ (s1 s2 s3 s4 s5 : ℝ), (s1 = s2 ∧ s2 = s3 ∧ s3 = s4 ∧ s4 = s5))
  (P2 : ∀ (a1 a2 a3 a4 a5 : ℝ), (a1 = a2 ∧ a2 = a3 ∧ a3 = a4 ∧ a4 = a5)), 
  ¬(P1 ∧ P2 → (regular pentagon P)) :=
by
  sorry

end polygon_independence_l88_88455


namespace count_non_adjacent_black_tiles_l88_88202

def width := 2
def length := 13
def num_black_tiles := 11
def num_white_tiles := 15

theorem count_non_adjacent_black_tiles :
  ∀ (hall_width hall_length : ℕ) (black_tiles white_tiles : ℕ),
  hall_width = width →
  hall_length = length →
  black_tiles = num_black_tiles →
  white_tiles = num_white_tiles →
  ∃ (configs : ℕ), 
  configs = 486 ∧ 
  (∀ (config : ℕ) (col : ℕ), config < configs → col < hall_length → 
  (no_adjacent_black_tiles config col): Prop) :=
sorry

end count_non_adjacent_black_tiles_l88_88202


namespace part1_part2_l88_88867

def f (n : ℕ) : ℕ := 
  if n = 1 then 2 
  else
    let rec find_indicator (m : ℕ) : ℕ :=
      if m ∣ n then find_indicator (m + 1) else m
    find_indicator 2

def a : ℕ → ℕ 
| 1     := 1
| 2     := 1
| n + 3 := (a (f (n + 3))) + 1

theorem part1 : ∃ C : ℕ, ∀ n : ℕ, n > 0 → a n ≤ 4 :=
sorry

theorem part2 : ¬ ∃ M T : ℕ, ∀ n : ℕ, n ≥ M → a n = a (n + T) :=
sorry

end part1_part2_l88_88867


namespace total_value_is_1_73_l88_88751

variable (a : ℝ := 7) -- Initial cookies
variable (b : ℝ := 2.5) -- Cookies eaten
variable (c : ℝ := 4.2) -- Cookies given by friend
variable (d : ℝ := 1.3) -- Additional cookies eaten
variable (e : ℝ := 3) -- More cookies given by another friend
variable (p : ℝ := 0.25) -- Cost per cookie

noncomputable def total_value_of_remaining_cookies (a b c d e p : ℝ) : ℝ :=
  let total_cookies := a - b + c - d + e
  let shared_cookies := total_cookies / 3
  let remaining_cookies := total_cookies - shared_cookies
  let total_value := remaining_cookies * p
  total_value

theorem total_value_is_1_73: total_value_of_remaining_cookies a b c d e p ≈ 1.73 := sorry

end total_value_is_1_73_l88_88751


namespace probability_first_three_heads_l88_88304

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l88_88304


namespace verify_calculations_l88_88746

theorem verify_calculations (m n x y a b : ℝ) :
  (2 * m - 3 * n) ^ 2 = 4 * m ^ 2 - 12 * m * n + 9 * n ^ 2 ∧
  (-x + y) ^ 2 = x ^ 2 - 2 * x * y + y ^ 2 ∧
  (a + 2 * b) * (a - 2 * b) = a ^ 2 - 4 * b ^ 2 ∧
  (-2 * x ^ 2 * y ^ 2) ^ 3 / (- x * y) ^ 3 ≠ -2 * x ^ 3 * y ^ 3 :=
by
  sorry

end verify_calculations_l88_88746


namespace value_of_a_b_l88_88085

theorem value_of_a_b (a b : ℕ) (ha : 2 * 100 + a * 10 + 3 + 326 = 5 * 100 + b * 10 + 9) (hb : (5 + b + 9) % 9 = 0): 
  a + b = 6 := 
sorry

end value_of_a_b_l88_88085


namespace arithmetic_sequence_term_seven_l88_88512

variable {a : ℕ → ℕ} -- a represents the arithmetic sequence

-- Define the condition
axiom h1 : a 4 + a 6 + a 8 + a 10 = 28

-- Define the theorem to prove
theorem arithmetic_sequence_term_seven (a : ℕ → ℕ) [arith_seq : (∀ n : ℕ, a n + a (n + 2) = 2 * a (n + 1))] : 
  a 7 = 7 :=
by
  sorry

end arithmetic_sequence_term_seven_l88_88512


namespace largest_prime_divisor_of_15_sq_plus_45_sq_l88_88855

theorem largest_prime_divisor_of_15_sq_plus_45_sq : 
  ∃ p, prime p ∧ (p ∣ (15 ^ 2 + 45 ^ 2)) ∧ ∀ q, (prime q ∧ (q ∣ (15 ^ 2 + 45 ^ 2))) → q ≤ p := 
by
  sorry

end largest_prime_divisor_of_15_sq_plus_45_sq_l88_88855


namespace find_a_l88_88526

theorem find_a {a : ℝ} :
  (∀ x : ℝ, (ax - 1) / (x + 1) < 0 → (x < -1 ∨ x > -1 / 2)) → a = -2 :=
by 
  intros h
  sorry

end find_a_l88_88526


namespace area_of_triangle_ADE_eq_3sqrt15div2_l88_88611

noncomputable def area_triangle_ADE (AB BC AC : ℝ) (A B C D E : Point)
  (hAB : dist A B = AB) (hBC : dist B C = BC) (hAC : dist A C = AC)
  (h_angle_bisector : is_angle_bisector ∠BCA C D)
  (h_circle : circle_passing_through A, D, C)
  (h_circle_intersects : circle_intersects_side_at C A B D E BC) : ℝ :=
  let AD := dist A D in
  let DE := dist D E in
  let sin_angle_ade := sin (angle A D E) in
  (1/2) * AD * DE * sin_angle_ade

theorem area_of_triangle_ADE_eq_3sqrt15div2
  (A B C D E : Point)
  (hAB : dist A B = 6)
  (hBC : dist B C = 4)
  (hAC : dist A C = 8)
  (h_angle_bisector : is_angle_bisector ∠BCA C D)
  (h_circle : circle_passing_through A, D, C)
  (h_circle_intersects : circle_intersects_side_at C A B D E BC) :
  area_triangle_ADE 6 4 8 A B C D E hAB hBC hAC h_angle_bisector h_circle h_circle_intersects =
  (3 * sqrt 15) / 2 :=
sorry

end area_of_triangle_ADE_eq_3sqrt15div2_l88_88611


namespace find_a_n_and_T_n_l88_88528

theorem find_a_n_and_T_n (n : ℕ) (h₀ : n ≥ 1)
  (S : ℕ → ℕ := λ n, n^2 + 2 * n)
  (b : ℕ → ℚ := λ n, 1 / ((2*n + 1) * (2*n + 3) : ℚ))
  (T : ℕ → ℚ := λ n, (∑ k in Finset.range n, b k)) :
  (∀ n, a n = 2 * n + 1) ∧ (∀ n, T n = n / (6 * n + 9)) :=
by
  suffices h₀: (∀ n, a n = 2 * n + 1),
    from ⟨h₀, λ n, rfl⟩
  sorry

end find_a_n_and_T_n_l88_88528


namespace derivative_at_zero_l88_88080

-- Definition of the function
noncomputable def f (x : ℝ) (n : ℕ) := ∏ i in (Finset.range n), (x + i + 1)

-- Theorem statement to prove f′(0) = n!
theorem derivative_at_zero (n : ℕ) : deriv (λ x, f x n) 0 = n! := sorry

end derivative_at_zero_l88_88080


namespace expected_value_sum_path_find_p_plus_q_l88_88183

noncomputable def expected_sum_path : ℚ := (100850 + 201700 + 2000)

theorem expected_value_sum_path 
    (a : Finset ℕ) (b : Finset ℕ)
    (H₁ : a.card = 100) 
    (H₂ : b.card = 200) 
    (H₃ : ∀ x ∈ a, 1 ≤ x ∧ x ≤ 2016) 
    (H₄ : ∀ y ∈ b, 1 ≤ y ∧ y ≤ 2016)
    (H₅ : a.val.nodup) (H₆ : b.val.nodup) : 
    (∑ i in a, i + ∑ j in b, j + 100 * (a.min' (by linarith))) = 304550 := 
begin
  sorry
end

theorem find_p_plus_q : 304550 + 1 = 304551 := 
begin
  norm_num,
end

end expected_value_sum_path_find_p_plus_q_l88_88183


namespace perpendicular_condition_l88_88771

theorem perpendicular_condition (l : Type) (a : Type) (is_perpendicular_to_many_lines_within_plane : l → a → Prop) :
  (∀ lines_in_plane, is_perpendicular_to_many_lines_within_plane l lines_in_plane) → necessary_but_not_sufficient (l ⟂ a) :=
by
  sorry

end perpendicular_condition_l88_88771


namespace concurrency_and_tangency_l88_88493

open EuclideanGeometry

variables {A B C I K A1 B1 C1 M N P : Point}
variables {triangle_ABC : Triangle}
variables {incircle_ABC : Circle}
variables {d : Line}
variables {circumcircle_XYZ : Circle}

-- Given the following conditions:
def conditions :=
  acute_triangle triangle_ABC ∧
  incircle triangle_ABC incircle_ABC I ∧
  tangent_line_circle d incircle_ABC K ∧
  perpendicular_from I IA d A1 ∧
  perpendicular_from I IB d B1 ∧
  perpendicular_from I IC d C1 ∧
  intersects_at d AB M ∧
  intersects_at d BC N ∧
  intersects_at d CA P ∧
  parallel_line P IA XYZ.line ∧
  parallel_line M IB XYZ.line ∧
  parallel_line N IC XYZ.line

-- We need to prove:
theorem concurrency_and_tangency (h : conditions) :
  concurrent (A, A1, B, B1, C, C1) ∧ tangent_line_circle (IK, circumcircle_XYZ) :=
sorry

end concurrency_and_tangency_l88_88493


namespace smallest_nat_satisfies_conditions_l88_88377

theorem smallest_nat_satisfies_conditions : 
  ∃ x : ℕ, (∃ m : ℤ, x + 13 = 5 * m) ∧ (∃ n : ℤ, x - 13 = 6 * n) ∧ x = 37 := by
  sorry

end smallest_nat_satisfies_conditions_l88_88377


namespace solve_problem_l88_88999

theorem solve_problem (a : ℝ) (x : ℝ) (h1 : 3 * x + |a - 2| = -3) (h2 : 3 * x + 4 = 0) :
  (a = 3 ∨ a = 1) → ((a - 2) ^ 2010 - 2 * a + 1 = -4 ∨ (a - 2) ^ 2010 - 2 * a + 1 = 0) :=
by {
  sorry
}

end solve_problem_l88_88999


namespace range_of_a_zero_value_of_a_minimum_l88_88524

noncomputable def f (x a : ℝ) : ℝ := Real.log x + (7 * a) / x

-- Problem 1: Range of a where f(x) has exactly one zero in its domain
theorem range_of_a_zero (a : ℝ) : 
  (∃! x : ℝ, (0 < x) ∧ f x a = 0) ↔ (a ∈ Set.Iic 0 ∪ {1 / (7 * Real.exp 1)}) := sorry

-- Problem 2: Value of a such that the minimum value of f(x) on [e, e^2] is 3
theorem value_of_a_minimum (a : ℝ) : 
  (∃ x : ℝ, (Real.exp 1 ≤ x ∧ x ≤ Real.exp 2) ∧ f x a = 3) ↔ (a = (Real.exp 2)^2 / 7) := sorry

end range_of_a_zero_value_of_a_minimum_l88_88524


namespace greatest_and_next_greatest_l88_88460

noncomputable def value1 := real.exp (real.log 2 / 2)
noncomputable def value2 := real.exp (real.log 3 / 3)
noncomputable def value3 := real.exp (real.log 8 / 8)
noncomputable def value4 := real.exp (real.log 9 / 9)

theorem greatest_and_next_greatest : 
  (value2 > value1) ∧ 
  (value1 > value3) ∧ 
  (value1 > value4) ∧ 
  (value2 > value3) ∧ 
  (value2 > value4) := 
sorry

end greatest_and_next_greatest_l88_88460


namespace complex_division_example_l88_88140

def complex_div (a b : ℂ) : ℂ := a / b

theorem complex_division_example : complex_div (complex.I) (1 + complex.I) = (1/2 : ℂ) + (1/2 : ℂ) * complex.I := 
by
  sorry

end complex_division_example_l88_88140


namespace _l88_88186

def p : ℕ := 2^16 + 1
def S : Set ℕ := {n | n % p ≠ 0}
def f (x : ℕ) : ℕ := -- appropriate function embedding for Lean, assuming comes from condition
  sorry

lemma main_theorem : 
  (∃ f : S → Fin p, (∀ x y ∈ S, f x * f y % p = (f (x * y) + f (x * y ^ (p - 2))) % p) 
  ∧ (∀ x ∈ S, f (x + p) = f x) 
  ∧ (let N := ∏ x in S, if f 81 ≠ 0 then f 81 else 1 in N % p = 16384)) := 
sorry

end _l88_88186


namespace prob_three_heads_is_one_eighth_l88_88330

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l88_88330


namespace min_value_of_sum_l88_88137

theorem min_value_of_sum (a b : ℝ) (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 2 = 6) :
  a + b ≥ 16 :=
sorry

end min_value_of_sum_l88_88137


namespace dot_product_EC_ED_l88_88586

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l88_88586


namespace slope_of_AB_l88_88518

-- Define the ellipse equation and point M on the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 3) = 1
def point_M (x y : ℝ) : Prop := x = real.sqrt 3 ∧ y = real.sqrt 2

-- Define the slopes of the lines MA and MB which are opposite
def slopes_opposite (k : ℝ) : Prop := ∀ x_M y_M, point_M x_M y_M → 
  ∃ k1 k2 : ℝ, k1 = k ∧ k2 = -k ∧ 
  ∀ x_A y_A x_B y_B, k1 = (y_A - y_M) / (x_A - x_M) ∧ k2 = (y_B - y_M) / (x_B - x_M)

-- Define the problem statement to find the slope of line AB
theorem slope_of_AB : ∀ x_A y_A x_B y_B, 
  slopes_opposite (sqrt 6 / 6) →
  ∃ slopeAB : ℝ, slopeAB = (y_B - y_A) / (x_B - x_A) ∧ slopeAB = real.sqrt 6 / 6 :=
sorry

end slope_of_AB_l88_88518


namespace average_minutes_correct_l88_88406

noncomputable def average_minutes_run_per_day : ℚ :=
  let f (fifth_graders : ℕ) : ℚ := (48 * (4 * fifth_graders) + 30 * (2 * fifth_graders) + 10 * fifth_graders) / (4 * fifth_graders + 2 * fifth_graders + fifth_graders)
  f 1

theorem average_minutes_correct :
  average_minutes_run_per_day = 88 / 7 :=
by
  sorry

end average_minutes_correct_l88_88406


namespace distance_sum_conditions_l88_88642

theorem distance_sum_conditions (a : ℚ) (k : ℚ) :
  abs (20 * a - 20 * k - 190) = 4460 ∧ abs (20 * a^2 - 20 * k - 190) = 2755 →
  a = -37 / 2 ∨ a = 39 / 2 :=
sorry

end distance_sum_conditions_l88_88642


namespace correct_equation_l88_88821

theorem correct_equation :
  ¬ (7^3 * 7^3 = 7^9) ∧ 
  (-3^7 / 3^2 = -3^5) ∧ 
  ¬ (2^6 + (-2)^6 = 0) ∧ 
  ¬ ((-3)^5 / (-3)^3 = -3^2) :=
by 
  sorry

end correct_equation_l88_88821


namespace sin_double_angle_of_angle_on_ray_l88_88774

variable (x y θ : ℝ)

theorem sin_double_angle_of_angle_on_ray :
  let θ := real.arctan 3 in
  (sin (2 * θ) = 3 / 5) :=
by {
  let θ := real.arctan 3,
  sorry
}

end sin_double_angle_of_angle_on_ray_l88_88774


namespace rest_stop_location_l88_88788

theorem rest_stop_location (km_A km_B : ℕ) (fraction : ℚ) (difference := km_B - km_A) 
  (rest_stop_distance := fraction * difference) : 
  km_A = 30 → km_B = 210 → fraction = 4 / 5 → rest_stop_distance + km_A = 174 :=
by 
  intros h1 h2 h3
  sorry

end rest_stop_location_l88_88788


namespace cos_sum_identity_l88_88676

theorem cos_sum_identity : 
  cos (2 * π / 7) + cos (4 * π / 7) + cos (8 * π / 7) = -1 / 2 := 
by sorry

end cos_sum_identity_l88_88676


namespace find_P_coordinates_l88_88898

-- Define points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -3)

-- Define the theorem
theorem find_P_coordinates :
  ∃ P : ℝ × ℝ, P = (8, -15) ∧ (P.1 - A.1, P.2 - A.2) = (3 * (B.1 - A.1), 3 * (B.2 - A.2)) :=
sorry

end find_P_coordinates_l88_88898


namespace dot_product_EC_ED_l88_88588

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l88_88588


namespace rectangle_area_l88_88608

theorem rectangle_area
  (x : ℝ)
  (perimeter_eq_160 : 10 * x = 160) :
  4 * (4 * x * x) = 1024 :=
by
  -- We would solve the problem and show the steps here
  sorry

end rectangle_area_l88_88608


namespace standard_equation_of_ellipse_maximize_area_ACN_l88_88498

noncomputable def eccentricity := 1 / 2
noncomputable def distance_bf := 2
noncomputable def m_ne_zero (m : ℝ) := m ≠ 0

-- Define the conditions
def ellipse (a b : ℝ) (h_ab : a > b ∧ b > 0) :=
  ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

def upper_vertex_to_focus_distance (a : ℝ) := a = 2

def ellipse_eccentricity (a c : ℝ) := c / a = 1 / 2

-- The standard equation of the ellipse is given:
theorem standard_equation_of_ellipse : 
  ∃ a b : ℝ, (a > b ∧ b > 0) ∧ 
              ellipse a b (a > b ∧ b > 0) ∧ 
              upper_vertex_to_focus_distance a ∧ 
              ellipse_eccentricity a 1 ∧ 
              a = 2 ∧ (b^2 = 3) :=
sorry

-- Define the line and the area to maximize
def line_intersects_ellipse (a b m : ℝ) (h_ab : a > b ∧ b > 0) (h_m : m ≠ 0) :=
  ∀ x y : ℝ, (y = x - 2*m) → ellipse a b h_ab

def area_ACN (m : ℝ) := (6 / 7) * Real.sqrt ((21 - (12 * m^2)) * m^2)

-- The equation of the line that maximizes the area of triangle ACN is:
theorem maximize_area_ACN : 
  ∃ m : ℝ, 
    m ≠ 0 ∧ 
    ∃ a b : ℝ, 
      a = 2 ∧ 
      b^2 = 3 ∧ 
      m^2 = 7 / 8 ∧ 
      line_intersects_ellipse a b m (a > b ∧ b > 0) (m ≠ 0) ∧ 
      area_ACN m = (3 * Real.sqrt 3) / 2 :=
sorry

end standard_equation_of_ellipse_maximize_area_ACN_l88_88498


namespace best_graph_representation_l88_88659

-- Definitions of conditions
def mike_trip_conds (graph : Type) : Prop :=
  -- Define each condition explicitly
  (∃ slow_city_traffic : Prop, slow_city_traffic ∧ 
  ∃ fast_highway_drive : Prop, fast_highway_drive ∧ 
  ∃ shopping_mall_stop : Prop, shopping_mall_stop ∧ 
  ∃ refueling_break : Prop, refueling_break ∧ 
  ∃ return_same_route : Prop, return_same_route)

-- Main statement: Given these conditions, the best representation is Graph B
theorem best_graph_representation (graphs : Type) (B : graphs) : 
  mike_trip_conds graphs → 
  (B = (λ graphs : Type, ∃ (f1 f2 : ℕ) (fast_highway_slope : Prop) (slow_city_slope : Prop),
    (f1 = 2 * 60 ∧ f2 = 30) ∧ fast_highway_slope ∧ slow_city_slope)) :=
by
  sorry

end best_graph_representation_l88_88659


namespace number_of_intersections_l88_88872

theorem number_of_intersections : ∃ (a_values : Finset ℚ), 
  ∀ a ∈ a_values, ∀ x y, y = 2 * x + a ∧ y = x^2 + 3 * a^2 ∧ x = 0 → 
  2 = a_values.card :=
by 
  sorry

end number_of_intersections_l88_88872


namespace solve_equation_l88_88221

theorem solve_equation (x : ℝ) : 
  (9 - 3 * x) * (3 ^ x) - (x - 2) * (x ^ 2 - 5 * x + 6) = 0 ↔ x = 3 :=
by sorry

end solve_equation_l88_88221


namespace cashier_five_dollar_bills_l88_88789

-- Define the conditions as a structure
structure CashierBills (x y : ℕ) : Prop :=
(total_bills : x + y = 126)
(total_value : 5 * x + 10 * y = 840)

-- State the theorem that we need to prove
theorem cashier_five_dollar_bills (x y : ℕ) (h : CashierBills x y) : x = 84 :=
sorry

end cashier_five_dollar_bills_l88_88789


namespace tiling_scenarios_unique_l88_88150

theorem tiling_scenarios_unique (m n : ℕ) 
  (h1 : 60 * m + 150 * n = 360) : m = 1 ∧ n = 2 :=
by {
  -- The proof will be provided here
  sorry
}

end tiling_scenarios_unique_l88_88150


namespace function_evaluation_l88_88525

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2 * x :=
by
  sorry

end function_evaluation_l88_88525


namespace cyclic_inequality_l88_88101

theorem cyclic_inequality
    (x1 x2 x3 x4 x5 : ℝ)
    (h1 : 0 < x1)
    (h2 : 0 < x2)
    (h3 : 0 < x3)
    (h4 : 0 < x4)
    (h5 : 0 < x5) :
    (x1 + x2 + x3 + x4 + x5)^2 > 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end cyclic_inequality_l88_88101


namespace strength_order_l88_88258

variable (Person : Type)
variable [LE Person] [DecidableRel ((≤) : Person → Person → Prop)]
variable (A B C D : Person)

-- Conditions
axiom match_condition_1 : A + B = C + D
axiom match_condition_2 : A + D > B + C
axiom match_condition_3 : B > A + C

-- Strength order to prove
theorem strength_order : D > B ∧ B > A ∧ A > C := sorry

end strength_order_l88_88258


namespace valid_numbers_count_l88_88126

def is_valid_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

def count_valid_numbers : ℕ :=
  ∑ n in (finset.range 10000).filter (λ n, ∀ d ∈ n.digits 10, is_valid_digit d), 1

theorem valid_numbers_count : count_valid_numbers = 624 := by
  sorry

end valid_numbers_count_l88_88126


namespace min_OG_locus_M_l88_88076

variables (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2)
variables (a b : ℝ)
variable (area_OPQ : ℝ)
variable (ab_product : ℝ)

-- Condition: The area of ΔOPQ is always 36
def triangle_area_condition : Prop := (1 / 2) * a * b * (sin θ) = 36

-- Calculate the product of lengths OP and OQ from the area condition
def ab_from_area : ab_product = 72 / (sin θ) := sorry

-- Minimum |OG|
def min_OG_value : ℝ := 4 * (sqrt (cot (θ / 2)))

theorem min_OG (G_x G_y : ℝ) (OG_value : ℝ) 
  (centroid_x_cond : G_x = (1 / 3) * (a + b) * cos (θ / 2))
  (centroid_y_cond : G_y = (1 / 3) * (a - b) * sin (θ / 2))
  (OG_sqrd : OG_value^2 = (1 / 9) * ((a^2 + b^2 + 2 * a * b * cos θ))) :
  OG_value = min_OG_value := sorry

-- Locus of M
def hyperbola_x_denominator : ℝ := 36 * cot (θ / 2)
def hyperbola_y_denominator : ℝ := 36 * tan (θ / 2)

theorem locus_M (M_x M_y : ℝ) (x_cond : M_x = (1 / 2) * (a + b) * cos (θ / 2))
  (y_cond : M_y = (1 / 2) * (a - b) * sin (θ / 2)) :
  (M_x^2 / hyperbola_x_denominator) - (M_y^2 / hyperbola_y_denominator) = 1 := sorry

end min_OG_locus_M_l88_88076


namespace minimum_value_of_a_l88_88921

-- Define the given condition
axiom a_pos : ℝ → Prop
axiom positive : ∀ (x : ℝ), 0 < x

-- Definition of the equation
def equation (x y a : ℝ) : Prop :=
  (2 * x - y / Real.exp 1) * Real.log (y / x) = x / (a * Real.exp 1)

-- The mathematical statement we need to prove
theorem minimum_value_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) (h_eq : equation x y a) : 
  a ≥ 1 / Real.exp 1 :=
sorry

end minimum_value_of_a_l88_88921


namespace find_ratio_l88_88195

variables {A B C D : Type} [plane α]
variables {a b c d : α}

-- Definition of the conditions
def is_internal_point_of_acute_triangle (D : α) : Prop :=
  ∃ ABC : triangle α, D ∈ int_triangle ABC

def angle_ADB_eq_angle_ACB_plus_90 (A B C D : α) : Prop :=
  ∃ (ABC : angle α), mangle ADB = mangle ACB + 90

def AC_times_BD_eq_AD_times_BC (A B C D : α) : Prop :=
  (dist A C) * (dist B D) = (dist A D) * (dist B C)

-- Main problem statement
theorem find_ratio (A B C D : α) 
  (h1 : is_internal_point_of_acute_triangle D)
  (h2 : angle_ADB_eq_angle_ACB_plus_90 A B C D)
  (h3 : AC_times_BD_eq_AD_times_BC A B C D) :
  (dist A B * dist C D) / (dist A C * dist B D) = real.sqrt 2 :=
sorry

end find_ratio_l88_88195


namespace dot_product_EC_ED_l88_88594

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l88_88594


namespace max_pairwise_disjoint_subsets_l88_88627

noncomputable def maxSubsets (n : ℕ) : ℕ :=
  Nat.choose n (n / 2)

theorem max_pairwise_disjoint_subsets (n : ℕ):
  ∀ (A : Finset (Fin n)) (A_subsets : Finset (Finset (Fin n))),
    (∀ X Y ∈ A_subsets, X ≠ Y → X ∩ Y = ∅) →
    A_subset.card = n →
    A_subsets.card ≤ maxSubsets n := sorry

end max_pairwise_disjoint_subsets_l88_88627


namespace parallel_vectors_x_value_l88_88952

variable {x : ℝ}

theorem parallel_vectors_x_value (h : (1 / x) = (2 / -6)) : x = -3 := sorry

end parallel_vectors_x_value_l88_88952


namespace arithmetic_and_geometric_properties_l88_88094

def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := n * (a₁ + a₁ + (n - 1) * d) / 2

def geometric_seq (b₁ q : ℤ) (n : ℕ) : ℤ := b₁ * q^(n - 1)

def sum_geometric_seq (b₁ q : ℤ) (n : ℕ) : ℤ := b₁ * (1 - q^n) / (1 - q)

theorem arithmetic_and_geometric_properties :
  let a₁ := 1
  let d := 2
  let a₄ := arithmetic_seq a₁ d 4
  let S₄ := sum_arithmetic_seq a₁ d 4
  (a₄ = 7 ∧ S₄ = 16) →
  ∃ q, q^2 - (a₄ + 1) * q + S₄ = 0 ∧ q = 4 ∧
  ∀ (n : ℕ),
  (arithmetic_seq a₁ d n = 2 * n - 1) ∧
  (sum_arithmetic_seq a₁ d n = n^2) ∧
  (geometric_seq 2 q n = 2 * 4^(n - 1)) ∧
  (sum_geometric_seq 2 q n = 2 * (1 - 4^n) / (1 - 4)) :=
by
  intros a₁ d a₄ S₄ h
  cases h with h₁ h₂
  use 4
  sorry  -- omitting actual proof

end arithmetic_and_geometric_properties_l88_88094


namespace exists_term_not_of_form_l88_88665

theorem exists_term_not_of_form (a d : ℕ) (h_seq : ∀ i j : ℕ, (i < 40 ∧ j < 40 ∧ i ≠ j) → a + i * d ≠ a + j * d)
  (pos_a : a > 0) (pos_d : d > 0) 
  : ∃ h : ℕ, h < 40 ∧ ¬ ∃ k l : ℕ, a + h * d = 2^k + 3^l :=
by {
  sorry
}

end exists_term_not_of_form_l88_88665


namespace squirrel_pine_cones_l88_88769

theorem squirrel_pine_cones (x y : ℕ) (hx : 26 - 10 + 9 + (x + 14)/2 = x/2) (hy : y + 5 - 18 + 9 + (x + 14)/2 = x/2) :
  x = 86 := sorry

end squirrel_pine_cones_l88_88769


namespace ellipse_problem_l88_88896

def ellipse_eq (a b : ℝ) : Prop := (a > b ∧ b > 0 ∧ 4 * b * b = a * a - 4)

def focal_distance_4 (a : ℝ) : Prop := a = 2 * real.sqrt 2

def chord_through_focus_and_perpendicular (l : ℝ) : Prop := l = 2 * real.sqrt 2

noncomputable def ellipse_E (a : ℝ) (b : ℝ) : Prop :=
  ∃ x y : ℝ, ((x^2) / (8:ℝ) + (y^2) / (4:ℝ) = 1 ∧ a = 2 * real.sqrt 2 ∧ b = 2)

theorem ellipse_problem (a b: ℝ) :
  ellipse_eq a b ∧
  focal_distance_4 a ∧
  chord_through_focus_and_perpendicular (2 * real.sqrt 2) →
  ellipse_E a b ∧
  ∀ (M N : ℝ), -4 ≤ (M * N) ∧ (M * N) ≤ 14 :=
by
  sorry

end ellipse_problem_l88_88896


namespace martha_age_l88_88655

theorem martha_age (ellen_age_now : ℕ) (h1 : ellen_age_now = 10) : 
  let ellen_age_in_six_years := ellen_age_now + 6 in
  let martha_age_now := 2 * ellen_age_in_six_years in
  martha_age_now = 32 :=
by
  sorry

end martha_age_l88_88655


namespace three_heads_in_a_row_l88_88312

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l88_88312


namespace range_of_a_for_no_extreme_points_l88_88997

theorem range_of_a_for_no_extreme_points :
  ∀ (a : ℝ), (∀ x : ℝ, x * (x - 2 * a) * x + 1 ≠ 0) ↔ -1 ≤ a ∧ a ≤ 1 := sorry

end range_of_a_for_no_extreme_points_l88_88997


namespace inequality_l88_88637

variable (a : ℕ → ℝ) (n : ℕ) (s k : ℝ)

-- Conditions
def positives := ∀ i : ℕ, (i < n) → a i > 0
def sum_eq_s := ∑ i in finset.range n, a i = s
def k_positive := k > 1

-- Statement
theorem inequality (h_pos : positives a n) (h_sum : sum_eq_s a s n) (h_k : k_positive k) :
  (finset.sum (finset.range n) (λ i, (a i)^k / (s - a i))) ≥ s^(k-1) / ((n-1) * n^(k-2)) := 
sorry

end inequality_l88_88637


namespace initial_milk_quantity_is_1248_l88_88396

noncomputable def initial_milk_quantity (A : ℝ) : Prop :=
  let B := 0.375 * A
  let C := 0.625 * A
  B + 156 = C - 156

theorem initial_milk_quantity_is_1248 : initial_milk_quantity 1248 :=
by 
  let A := 1248
  let B := 0.375 * A
  let C := 0.625 * A
  have h : B + 156 = C - 156 := by
    calc
      B + 156 = 0.375 * A + 156 : by rfl
      ... = 156 + 0.375 * 1248 : by rfl
      ... = 156 + 468 : by norm_num
      ... = 624 : by norm_num
    calc
      C - 156 = 0.625 * A - 156 : by rfl
      ... = 0.625 * 1248 - 156 : by rfl
      ... = 780 - 156 : by norm_num
      ... = 624 : by norm_num
  exact h

end initial_milk_quantity_is_1248_l88_88396


namespace proof_problem_theorem_l88_88584

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l88_88584


namespace sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l88_88742

-- Definitions based on conditions
def sum_arithmetic (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Theorem statements based on the correct answers
theorem sum_first_twelve_multiples_17 : 
  17 * sum_arithmetic 12 = 1326 := 
by
  sorry

theorem sum_squares_first_twelve_multiples_17 : 
  17^2 * sum_squares 12 = 187850 :=
by
  sorry

end sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l88_88742


namespace max_planes_with_conditions_l88_88052

def has_four_points_on_each_plane (planes : Set (Set Point))
  (points : Set Point) : Prop := 
  ∀ plane ∈ planes, ∃ subset ⊆ points, subset.card ≥ 4 ∧ subset ⊆ plane

def any_four_points_not_collinear (points : Set Point) : Prop :=
  ∀ (p1 p2 p3 p4 : Point) ∈ points, ¬collinear ({p1, p2, p3, p4} : Set Point)

theorem max_planes_with_conditions :
  ∃ (planes : Set (Set Point)) (points : Set Point), points.card = 6 ∧
  has_four_points_on_each_plane planes points ∧ 
  any_four_points_not_collinear points ∧
  (∀ (planes' : Set (Set Point)), planes_compat planes_plane' ∧ planes'.card > planes.card → false) ∧ 
  planes.card = 6 :=
sorry

end max_planes_with_conditions_l88_88052


namespace sum_of_squares_of_first_10_primes_l88_88448

theorem sum_of_squares_of_first_10_primes :
  ((2^2) + (3^2) + (5^2) + (7^2) + (11^2) + (13^2) + (17^2) + (19^2) + (23^2) + (29^2)) = 2397 :=
by
  sorry

end sum_of_squares_of_first_10_primes_l88_88448


namespace max_value_sum_of_weighted_real_numbers_l88_88684

theorem max_value_sum_of_weighted_real_numbers 
  (x : Fin 49 → ℝ)
  (h : ∑ i in Finset.range 49, (i+1) * x i ^ 2 = 1) : 
  ∑ i in Finset.range 49, (i + 1) * x i ≤ 35 := 
by
  sorry

end max_value_sum_of_weighted_real_numbers_l88_88684


namespace complex_conjugate_condition_l88_88883

theorem complex_conjugate_condition {a : ℝ} (h : (a + real.sqrt 3 * complex.I) * (a - real.sqrt 3 * complex.I) = 4) : a = 1 ∨ a = -1 := 
by sorry

end complex_conjugate_condition_l88_88883


namespace probability_three_heads_l88_88290

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l88_88290
