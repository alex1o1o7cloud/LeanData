import Mathlib

namespace fraction_division_l325_325838

theorem fraction_division :
  (3 / 4) / (5 / 8) = 6 / 5 :=
by
  sorry

end fraction_division_l325_325838


namespace complementA_intersect_B_l325_325306

noncomputable def setA : Set ℝ := {x | 1 / (1 - x) ≥ 1}

noncomputable def setB : Set ℝ := {x | x ^ 2 + 2 * x - 3 > 0}

theorem complementA_intersect_B :
  (set.compl setA ∩ setB) = (set.Iio (-3) ∪ set.Ioi 1) :=
by
  sorry

end complementA_intersect_B_l325_325306


namespace nth_term_is_112_l325_325032

noncomputable def arithmetic_sequence (x : ℝ) (n : ℕ) : ℝ :=
  let a1 := 3 * x - 4
  let d := (7 * x - 14) - a1
  a1 + (n - 1) * d

theorem nth_term_is_112 (x : ℝ) (n : ℕ) (h : arithmetic_sequence x n = 112) :
  n = 13 :=
begin
  sorry
end

end nth_term_is_112_l325_325032


namespace slope_CD_constant_l325_325091

-- Define the hypotheses and the main statement
theorem slope_CD_constant (x0 y0 p : ℝ) (h_p : p > 0)
  (P : ℝ × ℝ) (hP : P.snd^2 = 2 * p * P.fst)
  (A : ℝ × ℝ := (x0, y0))
  (B : ℝ × ℝ := (y0^2 / p - x0, y0)) :
  ∃ k : ℝ, ∀ C D : ℝ × ℝ,
    (C.snd^2 = 2 * p * C.fst ∧ D.snd^2 = 2 * p * D.fst) ∧
    (C ≠ P ∧ D ≠ P) ∧
    ((C.fst - A.fst) * (D.snd - A.snd) = (D.fst - A.fst) * (C.snd - A.snd)) ∧
    ((C.fst - B.fst) * (D.snd - B.snd) = (D.fst - B.fst) * (C.snd - B.snd)) → 
    (C.snd - D.snd) / (C.fst - D.fst) = k :=
begin
  sorry, -- Proof is omitted
end

end slope_CD_constant_l325_325091


namespace waiter_tables_l325_325012

theorem waiter_tables (total_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (number_of_tables : ℕ) 
  (h1 : total_customers = 22)
  (h2 : customers_left = 14)
  (h3 : people_per_table = 4)
  (h4 : remaining_customers = total_customers - customers_left)
  (h5 : number_of_tables = remaining_customers / people_per_table) :
  number_of_tables = 2 :=
by
  sorry

end waiter_tables_l325_325012


namespace polynomial_value_l325_325514

theorem polynomial_value (x : ℝ) (hx : x^2 - 4*x + 1 = 0) : 
  x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 - 32*Real.sqrt 3 ∨ 
  x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 + 32*Real.sqrt 3 :=
sorry

end polynomial_value_l325_325514


namespace triangles_with_positive_area_count_l325_325614

noncomputable def countTrianglesWithPositiveArea : ℕ :=
  let points := {(i, j) | i, j ∈ Finset.Icc 1 5}
  let totalCombinations := Nat.choose 25 3
  let collinearInRowsAndColumns := 10 * (5 + 5)
  let collinearInLongDiagonals := 20
  let collinearIn4PointsDiagonals := 4 * 4
  let collinearIn3PointsDiagonals := 4 * 1
  let collinearWithSpecificSlopes := 12
  let collinearSets := collinearInRowsAndColumns + collinearInLongDiagonals + collinearIn4PointsDiagonals + collinearIn3PointsDiagonals + collinearWithSpecificSlopes
  totalCombinations - collinearSets

theorem triangles_with_positive_area_count :
  countTrianglesWithPositiveArea = 2148 := by
  sorry

end triangles_with_positive_area_count_l325_325614


namespace number_of_factors_180_l325_325046

theorem number_of_factors_180 :
  let N := 180,
      p1 := 2, p2 := 3, p3 := 5,
      a1 := 2, a2 := 2, a3 := 1 in
  N = p1^a1 * p2^a2 * p3^a3 ∧ (N > 1) →
  ∏ i in [a1+1, a2+1, a3+1], i = 18 := by
  intros N p1 p2 p3 a1 a2 a3 hN
  sorry

end number_of_factors_180_l325_325046


namespace part_one_part_two_l325_325595

-- Assuming definitions for exponential, logarithmic functions and their properties exist

noncomputable def f (a x : ℝ) : ℝ := a / x - real.log x

noncomputable def g (a x x₀ : ℝ) : ℝ :=
  if x ≤ x₀ then
    a / x - real.log x - (real.log x) / x
  else
    real.log x - a / x - (real.log x) / x

theorem part_one (a x₀ : ℝ) (h_tangent : f a x₀ = 0) (h_tangent_derivative : f.deriv a x₀ = 0) :
    a = -1 / real.exp 1 :=
  sorry

theorem part_two (a x₀ : ℝ) (h_a_bounds : exp 1 + 1 < a ∧ a < exp 2) (h_f_zero : f a x₀ = 0) :
    ∃ x₁ x₂ : ℝ, g a x₀ x₁ = 0 ∧ g a x₀ x₂ = 0 ∧ (x₁ ≠ x₂) ∧ x₁ ∈ Ioo (exp 1) x₀ ∧ x₂ ∈ Ioo x₀ (exp 2) :=
  sorry

end part_one_part_two_l325_325595


namespace sequence_contains_composite_l325_325729

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem sequence_contains_composite (a : ℕ → ℕ) (h : ∀ n, a (n+1) = 2 * a n + 1 ∨ a (n+1) = 2 * a n - 1) :
  ∃ n, is_composite (a n) :=
sorry

end sequence_contains_composite_l325_325729


namespace maximize_fraction_product_l325_325671

theorem maximize_fraction_product :
  ∃ A B C D : ℕ, 
    A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
    B ≠ C ∧ B ≠ D ∧ 
    C ≠ D ∧ 
    (A / B : ℚ) * (C / D : ℚ) = 36 :=
by
  sorry

end maximize_fraction_product_l325_325671


namespace part_one_monotonicity_part_two_range_of_a_l325_325597

noncomputable def f (x a : ℝ) : ℝ := exp x + a * x + (1 / 2) * x^2

theorem part_one_monotonicity (x : ℝ) : 
  (a = -1) → 
  ((∀ x < 0, (f x -1).derivative x < 0) ∧ (∀ x > 0, (f x -1).derivative x > 0)) :=
by
  sorry

theorem part_two_range_of_a (x a : ℝ) :
  (x ≥ 0) → 
  (f x a ≥ (3/2) * x^2 + 3 * a * x + a^2 - 2 * exp x) → 
  (-sqrt 3 ≤ a ∧ a ≤ 2 - log 4 / 3) :=
by
  sorry

end part_one_monotonicity_part_two_range_of_a_l325_325597


namespace intersect_on_angle_bisector_l325_325303

-- Initial definitions for points and parallelograms
variables {A B C D P Q K : Type} [add_group A] [add_group B]
variables (parallelogram : Type) [add_group parallelogram]

-- Define when a set of points forms a parallelogram
def is_parallelogram (A B C D : parallelogram) : Prop :=
  -- TODO: Provide the actual definition, using sorry for now
  sorry

-- Define point on the side of a parallelogram
def on_side (P : parallelogram) (side : parallelogram) : Prop :=
  -- TODO: Provide the actual definition, using sorry for now
  sorry

-- Define distance equality between points
def dist_eq (P Q : parallelogram) : Prop :=
  -- TODO: Provide the actual definition, using sorry for now
  sorry

-- theorem statement
theorem intersect_on_angle_bisector
  (A B C D P Q : parallelogram)
  (h_parallelogram : is_parallelogram A B C D)
  (h_P_on_side_BC : on_side P B)
  (h_Q_on_side_CD : on_side Q C)
  (h_BP_DQ : dist_eq P Q) :
  ∃ K, (on_angle_bisector K A D) :=
begin
  sorry,
end

end intersect_on_angle_bisector_l325_325303


namespace solution_to_system_l325_325964

theorem solution_to_system (x y a b : ℝ) 
  (h1 : x = 1) (h2 : y = 2) 
  (h3 : a * x + b * y = 4) 
  (h4 : b * x - a * y = 7) : 
  a + b = 1 :=
by
  sorry

end solution_to_system_l325_325964


namespace find_f3_l325_325382

noncomputable def f : ℝ → ℝ := sorry

theorem find_f3 (h : ∀ x : ℝ, x ≠ 0 → f x - 2 * f (1 / x) = 3 ^ x) : f 3 = -11 :=
sorry

end find_f3_l325_325382


namespace compute_expression_l325_325966

variables (a b c : ℝ)

theorem compute_expression (h1 : a - b = 2) (h2 : a + c = 6) : 
  (2 * a + b + c) - 2 * (a - b - c) = 12 :=
by
  sorry

end compute_expression_l325_325966


namespace circle_divided_into_regions_l325_325257

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l325_325257


namespace mutually_exclusive_event_l325_325969

variables {Ω : Type*} {P : MeasureTheory.Measure Ω}
variables {A B : Set ℝ}

theorem mutually_exclusive_event (h : A ∩ B = ∅) :
  MeasureTheory.Probability.ofSet P A * MeasureTheory.Probability.ofSet P B = 0 :=
by sorry

end mutually_exclusive_event_l325_325969


namespace onewaynia_half_edges_l325_325650

theorem onewaynia_half_edges (V : Type*) [Fintype V] [DecidableEq V] (G : SimpleGraph V) : 
  (∀ v, G.degree v = 4) →
  ∃ k : ℕ, k ≥ 1 ∧ let number_of_ways := 2 ^ k in
    number_of_ways = (count_deletions_half_edges G) :=
begin
  sorry
end

end onewaynia_half_edges_l325_325650


namespace triangle_count_l325_325134

theorem triangle_count (i j : ℕ) (h₁ : 1 ≤ i ∧ i ≤ 6) (h₂ : 1 ≤ j ∧ j ≤ 6): 
  let points := { (x, y) | 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 } in
  fintype.card { t : finset (ℕ × ℕ) // t.card = 3 ∧ ∃ a b c : ℕ × ℕ, a ∉ t ∧ b ∉ t ∧ c ∉ t ∧ abs ((b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)) ≠ 0 } = 6800 := 
by
  sorry

end triangle_count_l325_325134


namespace translate_graph_l325_325384

theorem translate_graph (x : ℝ) : 
  ∃ (translation : ℝ), y = sin (2 * (x + translation)) = sin (2 * x + π / 6) := 
by
  use -π / 12
  sorry

end translate_graph_l325_325384


namespace quadrant_of_complex_number_l325_325972

noncomputable def quadrant_of_z (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "fourth quadrant"
  else "on an axis"

theorem quadrant_of_complex_number (z : ℂ) (h : (3 + 4 * complex.i) * z = 25) : 
  quadrant_of_z z = "fourth quadrant" :=
sorry

end quadrant_of_complex_number_l325_325972


namespace visitors_not_ill_l325_325479

theorem visitors_not_ill (total_visitors : ℕ) (ill_percentage : ℕ) (fall_ill : ℕ) : 
  total_visitors = 500 → 
  ill_percentage = 40 → 
  fall_ill = (ill_percentage * total_visitors) / 100 →
  total_visitors - fall_ill = 300 :=
by
  intros h1 h2 h3
  sorry

end visitors_not_ill_l325_325479


namespace cookout_kids_2004_l325_325635

variable (kids2005 kids2004 kids2006 : ℕ)

theorem cookout_kids_2004 :
  (kids2006 = 20) →
  (2 * kids2005 = 3 * kids2006) →
  (2 * kids2004 = kids2005) →
  kids2004 = 60 :=
by
  intros h1 h2 h3
  sorry

end cookout_kids_2004_l325_325635


namespace circles_radii_divide_regions_l325_325263

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l325_325263


namespace simplify_tan_product_l325_325339

theorem simplify_tan_product : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  -- use the angle addition formula for tangent
  have tan_sum : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12) :=
    by rw [Real.tan_add, Real.tan_pi_div_four]
  -- using the given condition tan(45 degrees) = 1
  have tan_45 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  sorry

end simplify_tan_product_l325_325339


namespace hexagon_diagonal_extension_projections_l325_325315

open Geometry

theorem hexagon_diagonal_extension_projections
{ABCDEF : RegularPolygon (Fin 6) ℝ} -- original hexagon ABCDEF
(K L M : Point ℝ ℝ) -- points K, L, M
(H : RegularPolygon (Fin 6) ℝ) -- hexagon H formed as described
(P Q R : Point ℝ ℝ) -- points P, Q, R
(h1 : K ∈ Line ABCDEF.diagonal₁) -- assumption for K
(h2 : L ∈ Line ABCDEF.diagonal₂) -- assumption for L
(h3 : M ∈ Line ABCDEF.diagonal₃) -- assumption for M
(h4 : H.formado_sub_intersec_sector KLM ABCDEF) -- H formation condition
(h5 : H.si_extensión ∉ Triangle KLM) -- extension condition for H
(h6 : P ∈ Intersection (Extension (H.edge₁)) (Extension (H.edge₂))) -- P intersection condition
(h7 : Q ∈ Intersection (Extension (H.edge₃)) (Extension (H.edge₄))) -- Q intersection condition
(h8 : R ∈ Intersection (Extension (H.edge₅)) (Extension (H.edge₆))) -- R intersection condition
: P ∈ Line (Extension ABCDEF.diagonal₁) ∧
  Q ∈ Line (Extension ABCDEF.diagonal₂) ∧
  R ∈ Line (Extension ABCDEF.diagonal₃) :=
by
  sorry

end hexagon_diagonal_extension_projections_l325_325315


namespace total_profit_l325_325021

theorem total_profit (A D : ℝ) (D_share : ℝ) (H_A : A = 2250) (H_D : D = 3200) (H_D_share : D_share = 810.28) :
  P = 1380.48 :=
by
  have ratio : A / D = 2250 / 3200, from sorry,
  have gcd_2250_3200 : gcd 2250 3200 = 50, from sorry,
  have simp_ratio : 2250 / 50 / (3200 / 50) = 45 / 64, from sorry,
  have parts_sum : 45 + 64 = 109, from sorry,
  have D_part : D_share = (64 / 109) * P, from sorry,
  have P := D_share / (64 / 109), from sorry,
  have P_val : P = 810.28 * (109 / 64), from sorry,
  exact P_val

end total_profit_l325_325021


namespace intersection_is_N_l325_325126

-- Define the sets M and N as given in the problem
def M := {x : ℝ | x > 0}
def N := {x : ℝ | Real.log x > 0}

-- State the theorem for the intersection of M and N
theorem intersection_is_N : (M ∩ N) = N := 
  by 
    sorry

end intersection_is_N_l325_325126


namespace cyclist_overtake_points_l325_325464

theorem cyclist_overtake_points (p c : ℝ) (track_length : ℝ) (h1 : c = 1.55 * p) (h2 : track_length = 55) : 
  ∃ n, n = 11 :=
by
  -- we'll add the proof steps later
  sorry

end cyclist_overtake_points_l325_325464


namespace repeating_decimal_fraction_sum_l325_325161

theorem repeating_decimal_fraction_sum : 
  let x := 36 / 99 in
  let a := 4 in
  let b := 11 in
  gcd a b = 1 ∧ (a : ℚ) / (b : ℚ) = x → a + b = 15 :=
by
  sorry

end repeating_decimal_fraction_sum_l325_325161


namespace percentage_of_500_l325_325431

theorem percentage_of_500 : (110 * 500) / 100 = 550 :=
by
  sorry

end percentage_of_500_l325_325431


namespace number_of_officers_l325_325991

theorem number_of_officers
  (avg_all : ℝ := 120)
  (avg_officer : ℝ := 420)
  (avg_non_officer : ℝ := 110)
  (num_non_officer : ℕ := 450) :
  ∃ O : ℕ, avg_all * (O + num_non_officer) = avg_officer * O + avg_non_officer * num_non_officer ∧ O = 15 :=
by
  sorry

end number_of_officers_l325_325991


namespace equality_or_neg_equality_of_eq_l325_325545

theorem equality_or_neg_equality_of_eq
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^3 / a = b^2 + a^3 / b) : a = b ∨ a = -b := 
  by
  sorry

end equality_or_neg_equality_of_eq_l325_325545


namespace index_50th_term_b_n_lt_0_l325_325508

noncomputable def b (n : ℕ) : ℝ := ∑ k in finset.range n, real.cos k 

theorem index_50th_term_b_n_lt_0 : ∃ (n : ℕ), n = floor (2 * real.pi * 50 + 1) ∧ b n < 0 := 
sorry

end index_50th_term_b_n_lt_0_l325_325508


namespace find_m_l325_325567

-- Define the sets A and B and the conditions
def A : Set ℝ := {x | x ≥ 3}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Define the conditions on these sets
def conditions (m : ℝ) : Prop :=
  (∀ x, x ∈ A ∨ x ∈ B m) ∧ (∀ x, ¬(x ∈ A ∧ x ∈ B m))

-- State the theorem
theorem find_m : ∃ m : ℝ, conditions m ∧ m = 3 :=
  sorry

end find_m_l325_325567


namespace largest_integer_x_l325_325756

theorem largest_integer_x (x : ℕ) : (1 / 4 : ℚ) + (x / 8 : ℚ) < 1 ↔ x <= 5 := sorry

end largest_integer_x_l325_325756


namespace tank_insulation_cost_l325_325468

def length := 5
def width := 3
def height := 2
def cost_per_square_foot := 20

def surface_area (l w h : ℕ) :=
  2 * l * w + 2 * l * h + 2 * w * h

def total_cost (sa cost : ℕ) := sa * cost

theorem tank_insulation_cost :
  total_cost (surface_area length width height) cost_per_square_foot = 1240 := 
  sorry

end tank_insulation_cost_l325_325468


namespace three_distinct_numbers_l325_325961

theorem three_distinct_numbers (s : ℕ) (A : Finset ℕ) (S : Finset ℕ) (hA : A = Finset.range (4 * s + 1) \ Finset.range 1)
  (hS : S ⊆ A) (hcard: S.card = 2 * s + 2) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x + y = 2 * z :=
by
  sorry

end three_distinct_numbers_l325_325961


namespace width_of_room_l325_325642

-- Conditions
def volume : ℝ := 10000 
def length : ℝ := 100
def height : ℝ := 10

-- Theorem to prove
theorem width_of_room : 
  (volume = length * width * height) → (width = 10) :=
begin
  sorry
end

end width_of_room_l325_325642


namespace max_value_of_m_l325_325944

-- Define the set M
def M (m : ℝ) : set (ℝ × ℝ) := {p | p.1 ≤ -1 ∧ p.2 ≤ m}

-- Translate the condition into a predicate
def condition (a b : ℝ) : Prop := a * 2^b - b - 3 * a ≥ 0

-- The proof statement that needs to be shown
theorem max_value_of_m (m : ℝ) : 
  (∀ (a b : ℝ), (a, b) ∈ M (m) → condition a b) ↔ m ≤ 1 :=
sorry

end max_value_of_m_l325_325944


namespace reconstruct_quadrilateral_l325_325916

def quadrilateralVectors (W W' X X' Y Y' Z Z' : ℝ) :=
  (W - Z = W/2 + Z'/2) ∧
  (X - Y = Y'/2 + X'/2) ∧
  (Y - X = Y'/2 + X'/2) ∧
  (Z - W = W/2 + Z'/2)

theorem reconstruct_quadrilateral (W W' X X' Y Y' Z Z' : ℝ) :
  quadrilateralVectors W W' X X' Y Y' Z Z' →
  W = (1/2) * W' + 0 * X' + 0 * Y' + (1/2) * Z' :=
sorry

end reconstruct_quadrilateral_l325_325916


namespace number_of_digits_of_1234_in_base5_l325_325952

def base5_representation_digits (n : ℕ) : ℕ :=
  if h : n > 0 then
    Nat.find (λ k, n < 5^(k + 1)) + 1
  else
    1

theorem number_of_digits_of_1234_in_base5 : base5_representation_digits 1234 = 5 := 
by
  unfold base5_representation_digits
  have h : ∃ k, 1234 < 5^(k+1), from Exists.intro 4 (by norm_num)
  simp [Nat.find_spec h]
  rfl

end number_of_digits_of_1234_in_base5_l325_325952


namespace percent_sparrows_not_pigeons_l325_325986

-- Definitions of percentages
def crows_percent : ℝ := 0.20
def sparrows_percent : ℝ := 0.40
def pigeons_percent : ℝ := 0.15
def doves_percent : ℝ := 0.25

-- The statement to prove
theorem percent_sparrows_not_pigeons :
  (sparrows_percent / (1 - pigeons_percent)) = 0.47 :=
by
  sorry

end percent_sparrows_not_pigeons_l325_325986


namespace track_length_approximation_l325_325615

theorem track_length_approximation (steps_100m : ℝ) (track_steps : ℝ) (length_100m : ℝ) : 
  steps_100m = 200 ∧ length_100m = 100 ∧ track_steps = 800 →
  (track_steps / steps_100m) * length_100m = 400 := 
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end track_length_approximation_l325_325615


namespace totalNameLengths_l325_325285

-- Definitions of the lengths of names
def JonathanNameLength := 8 + 10
def YoungerSisterNameLength := 5 + 10
def OlderBrotherNameLength := 6 + 10
def YoungestSisterNameLength := 4 + 15

-- Statement to prove
theorem totalNameLengths :
  JonathanNameLength + YoungerSisterNameLength + OlderBrotherNameLength + YoungestSisterNameLength = 68 :=
by
  sorry -- no proof required

end totalNameLengths_l325_325285


namespace num_nonempty_subsets_l325_325957

theorem num_nonempty_subsets {α : Type*} (s : set α) (h : s = {1, 2, 3}) : (s.powerset.erase ∅).to_finset.card = 7 :=
by
  sorry

end num_nonempty_subsets_l325_325957


namespace first_laptop_cost_l325_325818

variable (x : ℝ)

def cost_first_laptop (x : ℝ) : ℝ := x
def cost_second_laptop (x : ℝ) : ℝ := 3 * x
def total_cost (x : ℝ) : ℝ := cost_first_laptop x + cost_second_laptop x
def budget : ℝ := 2000

theorem first_laptop_cost : total_cost x = budget → x = 500 :=
by
  intros h
  sorry

end first_laptop_cost_l325_325818


namespace polygon_side_ratio_l325_325324

variables {n : ℕ} (a : Fin n → ℝ)
  (h_order : ∀ i j, i < j → a i ≥ a j)

theorem polygon_side_ratio (h : ∀ i, a 0 < a i + ∑ j in Finset.filter (λ x, x ≠ i) Finset.univ, a x) :
  ∃ (i j : Fin n), i < j ∧ (1 / 2 : ℝ) < a j / a i ∧ a j / a i < 2 :=
by
  sorry

end polygon_side_ratio_l325_325324


namespace regions_divided_by_radii_circles_l325_325244

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l325_325244


namespace find_some_number_l325_325444

theorem find_some_number :
  ∃ (some_number : ℝ), (481 + 426) * 2 - (some_number * 481 * 426) = 3025 ∧ some_number = -0.005918 :=
by
  use -0.005918
  split
  sorry  -- here comes the verification of the condition

end find_some_number_l325_325444


namespace fraction_sum_l325_325149

theorem fraction_sum (x a b : ℕ) (h1 : x = 36 / 99) (h2 : a = 4) (h3 : b = 11) (h4 : Nat.gcd a b = 1) : a + b = 15 :=
by
  sorry

end fraction_sum_l325_325149


namespace hostel_cost_23_days_l325_325172

def weekday_cost_first_week : ℕ := 20
def weekend_cost_first_week : ℕ := 25
def weekday_cost_add_weeks : ℕ := 15
def weekend_cost_add_weeks : ℕ := 20
def discount_threshold : ℕ := 15
def discount_rate : ℝ := 0.10

def first_week_days : ℕ := 7
def total_days : ℕ := 23
def weekdays_first_week : ℕ := 5
def weekends_first_week : ℕ := 2

def calculate_cost(first_weekdays : ℕ, first_weekends : ℕ, 
                   add_weekdays : ℕ, add_weekends : ℕ) : ℝ :=
  (first_weekdays * weekday_cost_first_week + first_weekends * weekend_cost_first_week) +
  (add_weekdays * weekday_cost_add_weeks + add_weekends * weekend_cost_add_weeks)

theorem hostel_cost_23_days : calculate_cost 5 2 12 4 * (if total_days ≥ discount_threshold then (1.0 - discount_rate) else 1.0) = 369 :=
by sorry

end hostel_cost_23_days_l325_325172


namespace polynomial_min_value_P_l325_325583

theorem polynomial_min_value_P (a b : ℝ) (h_root_pos : ∀ x, a * x^3 - x^2 + b * x - 1 = 0 → 0 < x) :
    (∀ x : ℝ, a * x^3 - x^2 + b * x - 1 = 0 → x > 0) →
    ∃ P : ℝ, P = 12 * Real.sqrt 3 :=
sorry

end polynomial_min_value_P_l325_325583


namespace horizontal_asymptote_l325_325511

-- Define the numerator and denominator polynomials
def N (x : ℝ) : ℝ := 15 * x^5 + 7 * x^3 + 10 * x^2 + 6 * x + 2
def D (x : ℝ) : ℝ := 4 * x^5 + 3 * x^3 + 11 * x^2 + 4 * x + 1

-- State the theorem about the horizontal asymptote
theorem horizontal_asymptote : 
  (∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x > M,  | N(x) / D(x) - L | < ε) ∧ L = 15 / 4 := 
sorry

end horizontal_asymptote_l325_325511


namespace sum_of_angles_of_9_pointed_star_is_540_l325_325696

-- Define the circle with nine evenly spaced points.
def circle_with_nine_points := { p : ℝ // 0 <= p ∧ p < 360 }

-- Define a 9-pointed star formed by connecting these nine points.
def nine_pointed_star (points : fin 9 → circle_with_nine_points) : Prop :=
  ∀ i : fin 9, points i = ⟨ i.1 * 40, sorry ⟩

-- Define the sum of the angle measurements at the tips of the 9-pointed star.
def sum_of_tip_angles (points : fin 9 → circle_with_nine_points) : ℝ :=
  (∑ i in finset.univ, 60)

-- Statement to be proved: 
theorem sum_of_angles_of_9_pointed_star_is_540 : ∀ points : fin 9 → circle_with_nine_points,  
  nine_pointed_star points → sum_of_tip_angles points = 540 := 
by
  intros points h
  sorry

end sum_of_angles_of_9_pointed_star_is_540_l325_325696


namespace circle_regions_division_l325_325248

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l325_325248


namespace recurring_fraction_sum_l325_325159

theorem recurring_fraction_sum (a b : ℕ) (h : 0.36̅ = ↑a / ↑b) (gcd_ab : Nat.gcd a b = 1) : a + b = 15 :=
sorry

end recurring_fraction_sum_l325_325159


namespace angle_AHB_l325_325822

variables (A B C D E H : Type) [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ D] [EuclideanSpace ℝ E] [EuclideanSpace ℝ H]

-- Definitions of the points and angles
axiom altitudes_intersect_at (h : line A D) (k : line B E) (h_alt : h ⊥ planar (triangle A B C)) (k_alt : k ⊥ planar (triangle A B C)) 
         : intersection_point h k H  -- Altitudes intersect at H 

axiom angle_BAC : angle A B C = 30 -- angle BAC = 30 degrees
axiom angle_ABC : angle B A C = 80 -- angle ABC = 80 degrees

theorem angle_AHB (h : line A D) (k : line B E) (h_alt : h ⊥ planar (triangle A B C)) (k_alt : k ⊥ planar (triangle A B C)) : 
  intersection_point h k H → angle A H B = 110 := 
by
  sorry -- Proof omitted

end angle_AHB_l325_325822


namespace odd_function_extended_l325_325100

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≥ 0 then 
    x * Real.log (x + 1)
  else 
    x * Real.log (-x + 1)

theorem odd_function_extended : (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x ≥ 0 → f x = x * Real.log (x + 1)) →
  (∀ x : ℝ, x < 0 → f x = x * Real.log (-x + 1)) :=
by
  intros h_odd h_def_neg
  sorry

end odd_function_extended_l325_325100


namespace area_KPNM_l325_325909

open EuclideanGeometry

variable {A B C D K M P N : Point}

-- Given conditions
axiom h1 : midpoint K A B
axiom h2 : midpoint M C D
axiom h3 : collinear P K C
axiom h4 : collinear P B M
axiom h5 : collinear N A M
axiom h6 : collinear N K D
axiom h7 : ∠CBP = 30
axiom h8 : ∠NDA = 30
axiom h9 : ∠BPC = 105
axiom h10 : ∠DAN = 15
axiom h11 : BP = 2 * Real.sqrt 2
axiom h12 : ND = Real.sqrt 3

-- Proof that the area of quadrilateral KPNM equals the provided value
theorem area_KPNM : area K P N M = (Real.sqrt 3 + 1) * (Real.sqrt 6 + 4 * Real.sqrt 2) / 4 := sorry

end area_KPNM_l325_325909


namespace solution_set_of_inequality_l325_325739

theorem solution_set_of_inequality :
  {x : ℝ | 6^(x - 2) < 1} = {x : ℝ | x < 2} :=
sorry

end solution_set_of_inequality_l325_325739


namespace pennies_thrown_total_l325_325701

theorem pennies_thrown_total (rachelle_pennies gretchen_pennies rocky_pennies : ℕ) 
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) : 
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 := 
by 
  sorry

end pennies_thrown_total_l325_325701


namespace regions_formed_l325_325219

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l325_325219


namespace logarithmic_relationship_l325_325737

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_relationship
  (h1 : 0 < Real.cos 1)
  (h2 : Real.cos 1 < Real.sin 1)
  (h3 : Real.sin 1 < 1)
  (h4 : 1 < Real.tan 1) :
  log_base (Real.sin 1) (Real.tan 1) < log_base (Real.cos 1) (Real.tan 1) ∧
  log_base (Real.cos 1) (Real.tan 1) < log_base (Real.cos 1) (Real.sin 1) ∧
  log_base (Real.cos 1) (Real.sin 1) < log_base (Real.sin 1) (Real.cos 1) :=
sorry

end logarithmic_relationship_l325_325737


namespace ab_squared_plus_four_composite_l325_325669

theorem ab_squared_plus_four_composite (a b c : ℕ) (hp : nat.prime (nat.abs (a - b)))
  (h_eq : nat.gcd a b + nat.lcm a b = 2021^c) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬ nat.prime ((a + b)^2 + 4) :=
by {
  sorry
}

end ab_squared_plus_four_composite_l325_325669


namespace speedster_convertibles_count_l325_325792

-- Definitions of conditions
def total_inventory (T : ℕ) : Prop := (T / 3) = 60
def number_of_speedsters (T S : ℕ) : Prop := S = (2 / 3) * T
def number_of_convertibles (S C : ℕ) : Prop := C = (4 / 5) * S

-- Primary statement to prove
theorem speedster_convertibles_count (T S C : ℕ) (h1 : total_inventory T) (h2 : number_of_speedsters T S) (h3 : number_of_convertibles S C) : C = 96 :=
by
  -- Conditions and given values are defined
  sorry

end speedster_convertibles_count_l325_325792


namespace simplify_tan_expr_l325_325350

-- Definition of the tangents of 30 degrees and 15 degrees
def tan_30 : ℝ := Real.tan (Real.pi / 6)
def tan_15 : ℝ := Real.tan (Real.pi / 12)

-- Theorem stating that (1 + tan_30) * (1 + tan_15) = 2
theorem simplify_tan_expr : (1 + tan_30) * (1 + tan_15) = 2 :=
by
  sorry

end simplify_tan_expr_l325_325350


namespace bells_lcm_l325_325513

/-
  Given the intervals at which five bells toll: 5, 8, 11, 15, and 20,
  we need to prove that their least common multiple (LCM) is 1320.
-/

theorem bells_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm 5 8) 11) (Nat.lcm 15 20) = 1320 := 
by 
  calc Nat.lcm (Nat.lcm (Nat.lcm 5 8) 11) (Nat.lcm 15 20)
    = 1320 : sorry

end bells_lcm_l325_325513


namespace sample_size_l325_325084

theorem sample_size (F n : ℕ) (FR : ℚ) (h1: F = 36) (h2: FR = 1/4) (h3: FR = F / n) : n = 144 :=
by 
  sorry

end sample_size_l325_325084


namespace regions_divided_by_radii_circles_l325_325241

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l325_325241


namespace Terry_has_20_more_stickers_than_Steven_l325_325713

theorem Terry_has_20_more_stickers_than_Steven :
  let Ryan_stickers := 30
  let Steven_stickers := 3 * Ryan_stickers
  let Total_stickers := 230
  let Ryan_Steven_Total := Ryan_stickers + Steven_stickers
  let Terry_stickers := Total_stickers - Ryan_Steven_Total
  (Terry_stickers - Steven_stickers) = 20 := 
by 
  sorry

end Terry_has_20_more_stickers_than_Steven_l325_325713


namespace integral_abs_x_plus_2_eq_29_div_2_integral_inv_x_minus_1_eq_1_l325_325030

open Real

noncomputable def integral_abs_x_plus_2 : ℝ :=
  ∫ x in (-4 : ℝ)..(3 : ℝ), |x + 2|

noncomputable def integral_inv_x_minus_1 : ℝ :=
  ∫ x in (2 : ℝ)..(Real.exp 1 + 1 : ℝ), 1 / (x - 1)

theorem integral_abs_x_plus_2_eq_29_div_2 :
  integral_abs_x_plus_2 = 29 / 2 :=
sorry

theorem integral_inv_x_minus_1_eq_1 :
  integral_inv_x_minus_1 = 1 :=
sorry

end integral_abs_x_plus_2_eq_29_div_2_integral_inv_x_minus_1_eq_1_l325_325030


namespace sum_diff_l325_325170

def sum_even (m n : ℕ) : ℕ := ∑ i in finset.Icc m n, if i % 2 = 0 then i else 0
def sum_odd (m n : ℕ) : ℕ := ∑ i in finset.Icc m n, if i % 2 ≠ 0 then i else 0

theorem sum_diff :
  let a := sum_even 2 40
  let b := sum_odd 1 39
  a - b = 20 :=
by
  sorry

end sum_diff_l325_325170


namespace bc_length_four_points_l325_325402

theorem bc_length_four_points (A B C D : ℝ) (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (segments : set ℝ) (h_segments : segments = {14, 21, 34, 35, 48, 69})
  (h_AB : abs (B - A) ∈ segments)
  (h_AC : abs (C - A) ∈ segments)
  (h_AD : abs (D - A) ∈ segments)
  (h_BC : abs (C - B) ∈ segments)
  (h_BD : abs (D - B) ∈ segments)
  (h_CD : abs (D - C) ∈ segments) :
  abs (C - B) = 14 :=
by
  sorry

end bc_length_four_points_l325_325402


namespace central_angle_measure_l325_325723

-- Given the problem definitions
variables (A : ℝ) (x : ℝ)

-- Condition: The probability of landing in the region is 1/8
def probability_condition : Prop :=
  (1 / 8 : ℝ) = (x / 360)

-- The final theorem to prove
theorem central_angle_measure (h : probability_condition x) : x = 45 := 
  sorry

end central_angle_measure_l325_325723


namespace paint_calculation_correct_l325_325625

-- Define the height of the original statue and the amount of paint required for it
def original_height : ℝ := 10
def original_paint : ℝ := 1

-- Define the height of the smaller statues and their number
def small_height : ℝ := 2
def number_of_small_statues : ℕ := 360

-- Define the ratio of heights and the ratio of surface areas
def height_ratio : ℝ := small_height / original_height
def surface_area_ratio : ℝ := height_ratio^2

-- Define the paint required for a smaller statue with double the thickness
def small_statue_paint : ℝ := 2 * original_paint * surface_area_ratio

-- Define the total paint required for all small statues
def total_paint_required : ℝ := number_of_small_statues * small_statue_paint

-- Statement to prove
theorem paint_calculation_correct :
  total_paint_required = 28.8 :=
sorry

end paint_calculation_correct_l325_325625


namespace andrew_total_days_l325_325486

noncomputable def hours_per_day : ℝ := 2.5
noncomputable def total_hours : ℝ := 7.5

theorem andrew_total_days : total_hours / hours_per_day = 3 := 
by 
  sorry

end andrew_total_days_l325_325486


namespace min_value_quadratic_l325_325529

noncomputable def quadratic_expr (x : ℝ) : ℝ :=
  x^2 - 4 * x - 2019

theorem min_value_quadratic :
  ∀ x : ℝ, quadratic_expr x ≥ -2023 :=
by
  sorry

end min_value_quadratic_l325_325529


namespace fencing_cost_per_foot_is_3_l325_325842

-- Definitions of the constants given in the problem
def side_length : ℕ := 9
def back_length : ℕ := 18
def total_cost : ℕ := 72
def neighbor_behind_rate : ℚ := 1/2
def neighbor_left_rate : ℚ := 1/3

-- The statement to be proved
theorem fencing_cost_per_foot_is_3 : 
  (total_cost / ((2 * side_length + back_length) - 
                (neighbor_behind_rate * back_length) -
                (neighbor_left_rate * side_length))) = 3 := 
by
  sorry

end fencing_cost_per_foot_is_3_l325_325842


namespace n_gon_angle_condition_l325_325482

theorem n_gon_angle_condition (n : ℕ) (h1 : 150 * (n-1) + (30 * n - 210) = 180 * (n-2)) (h2 : 30 * n - 210 < 150) (h3 : 30 * n - 210 > 0) :
  n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 :=
by
  sorry

end n_gon_angle_condition_l325_325482


namespace power_mod_cycle_three_mod_eleven_prob_mod3_exp1234_11_l325_325423

theorem power_mod_cycle :
  ∃ k : ℕ, (1234 = 5 * k + 4) :=
by {
  use 246,
  norm_num,
}

theorem three_mod_eleven :
  (3^1 % 11 = 3) ∧ (3^2 % 11 = 9) ∧ (3^3 % 11 = 5) ∧ (3^4 % 11 = 4) ∧ (3^5 % 11 = 1) := 
by norm_num

theorem prob_mod3_exp1234_11 :
  3^1234 % 11 = 4 :=
by {
  have step := three_mod_eleven,
  have cycle := power_mod_cycle,
  simp at step,
  simp at cycle,
  rw [← Nat.pow_mod(3, 1234, 11)],
  cases cycle with k hk,
  rw hk,
  rw [Nat.pow_add, Nat.pow_mul, step.right.right.right.right],
  norm_num,
  rw [step.right.right.right.left],
  exact step.right.right.right.left
}

end power_mod_cycle_three_mod_eleven_prob_mod3_exp1234_11_l325_325423


namespace median_ride_time_l325_325741

theorem median_ride_time :
  let times : List ℕ := [30, 40, 55, 68, 78, 92, 105, 110, 130, 147, 149, 160, 170, 192, 205, 208, 213, 225, 245, 250]
  (sorted_times := times.qsort (· < ·))
  ∃ median, List.nth sorted_times 9 = some median ∧ median = 147 :=
  begin
    sorry
  end

end median_ride_time_l325_325741


namespace card_probability_is_correct_l325_325405

section CardProbability

open ProbabilityTheory

-- Definitions for the question conditions
def total_cards := 52
def spades_count := 13
def hearts_count := 13
def kings_count := 4
def non_king_spades := 12
def non_king_hearts := 12
def king_spades := 1

-- Correct answer definition as rational number
def correct_answer : ℚ := 17 / 3683

-- The probability calculation as described in the problem statement
def calculate_probability : ℚ :=
  (non_king_spades / total_cards) * (non_king_hearts / (total_cards - 1)) * (kings_count / (total_cards - 2)) +
  (king_spades / total_cards) * (non_king_hearts / (total_cards - 1)) * ((kings_count - 1) / (total_cards - 2))

-- The theorem stating that the calculated probability is equal to the correct answer
theorem card_probability_is_correct :
  calculate_probability = correct_answer :=
by
  -- skipping the proof steps
  sorry

end CardProbability

end card_probability_is_correct_l325_325405


namespace probability_of_drawing_1_on_kth_draw_l325_325018

-- Define the context and conditions of the problem
variables (n k : ℕ)
variable (hne_zero : n > 0)
variable (hkn : 1 ≤ k ∧ k ≤ n)

-- Define the event of drawing ticket number 1 on the k-th draw
def probability_drawing_1_on_kth_draw : ℚ :=
  (1 : ℚ) / (n : ℚ)

-- Statement of the theorem
theorem probability_of_drawing_1_on_kth_draw :
  ∀ (n k : ℕ), n > 0 → (1 ≤ k ∧ k ≤ n) →
  probability_drawing_1_on_kth_draw n k = 1 / n :=
by intros; sorry

end probability_of_drawing_1_on_kth_draw_l325_325018


namespace petra_waddle_longer_by_four_feet_l325_325433

def feet_in_mile : ℕ := 5280

def total_distance : ℕ := 2 * feet_in_mile

def number_of_markers : ℕ := 81
def number_of_gaps : ℕ := number_of_markers - 1

def winston_hops_per_gap : ℕ := 88
def petra_waddles_per_gap : ℕ := 24

def total_winston_hops : ℕ := winston_hops_per_gap * number_of_gaps
def total_petra_waddles : ℕ := petra_waddles_per_gap * number_of_gaps

noncomputable def winston_hop_length : ℝ := total_distance / total_winston_hops.to_float
noncomputable def petra_waddle_length : ℝ := total_distance / total_petra_waddles.to_float

noncomputable def difference_in_length : ℝ := petra_waddle_length - winston_hop_length

theorem petra_waddle_longer_by_four_feet :
  difference_in_length = 4 := by
  sorry

end petra_waddle_longer_by_four_feet_l325_325433


namespace find_tan_G_l325_325864

def right_triangle (FG GH FH : ℕ) : Prop :=
  FG^2 = GH^2 + FH^2

def tan_ratio (GH FH : ℕ) : ℚ :=
  FH / GH

theorem find_tan_G
  (FG GH : ℕ)
  (H1 : FG = 13)
  (H2 : GH = 12)
  (FH : ℕ)
  (H3 : right_triangle FG GH FH) :
  tan_ratio GH FH = 5 / 12 :=
by
  sorry

end find_tan_G_l325_325864


namespace volume_of_adjacent_cubes_l325_325537

theorem volume_of_adjacent_cubes 
(side_length count : ℝ) 
(h_side : side_length = 5) 
(h_count : count = 5) : 
  (count * side_length ^ 3) = 625 :=
by
  -- Proof steps (skipped)
  sorry

end volume_of_adjacent_cubes_l325_325537


namespace fraction_sum_l325_325150

theorem fraction_sum (x a b : ℕ) (h1 : x = 36 / 99) (h2 : a = 4) (h3 : b = 11) (h4 : Nat.gcd a b = 1) : a + b = 15 :=
by
  sorry

end fraction_sum_l325_325150


namespace bruno_coconuts_per_trip_is_8_l325_325830

-- Definitions related to the problem conditions
def total_coconuts : ℕ := 144
def barbie_coconuts_per_trip : ℕ := 4
def trips : ℕ := 12
def bruno_coconuts_per_trip : ℕ := total_coconuts - (barbie_coconuts_per_trip * trips)

-- The main theorem stating the question and the answer
theorem bruno_coconuts_per_trip_is_8 : bruno_coconuts_per_trip / trips = 8 :=
by
  sorry

end bruno_coconuts_per_trip_is_8_l325_325830


namespace expression_equals_l325_325843

theorem expression_equals :
  (1 / 2 * log 25 / log 10 + log 2 / log 10 + log (e ^ (2 / 3)) / log 2 + (sqrt 2 - sqrt 3) ^ 0) = 8 / 3 :=
by
  sorry

end expression_equals_l325_325843


namespace fraction_solution_l325_325380

theorem fraction_solution (a : ℕ) (h : a > 0) (h_eq : (a : ℚ) / (a + 45) = 0.75) : a = 135 :=
sorry

end fraction_solution_l325_325380


namespace locus_of_P_l325_325085

noncomputable def square_vertices (s : ℝ) : ℕ → ℝ × ℝ
| 0 => (0, 0)
| 1 => (s, 0)
| 2 => (s, s)
| 3 => (0, s)
| _ => (0, 0) -- default (should not happen for valid input)

def squared_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

def centroid (A B C D : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1 + D.1) / 4, (A.2 + B.2 + C.2 + D.2) / 4)

theorem locus_of_P 
  (s : ℝ) 
  (A B C D : ℝ × ℝ := square_vertices s 0, square_vertices s 1, square_vertices s 2, square_vertices s 3) 
  (P : ℝ × ℝ) : Prop :=
  let G := centroid A B C D in
  let K := (squared_distance A G + squared_distance B G + squared_distance C G + squared_distance D G) in
  squared_distance P A + squared_distance P B + squared_distance P C + squared_distance P D = 2 * s^2 ↔
  squared_distance P G = (2 * s^2 - K) / 4

end locus_of_P_l325_325085


namespace triangle_count_l325_325133

theorem triangle_count (i j : ℕ) (h₁ : 1 ≤ i ∧ i ≤ 6) (h₂ : 1 ≤ j ∧ j ≤ 6): 
  let points := { (x, y) | 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 } in
  fintype.card { t : finset (ℕ × ℕ) // t.card = 3 ∧ ∃ a b c : ℕ × ℕ, a ∉ t ∧ b ∉ t ∧ c ∉ t ∧ abs ((b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)) ≠ 0 } = 6800 := 
by
  sorry

end triangle_count_l325_325133


namespace add_to_expression_to_get_91_l325_325425

theorem add_to_expression_to_get_91 : 
  let x := 90 
  in x + (5 * 12 / (180 / 3)) = 91 :=
by
  let x := 90
  have h1 : 180 / 3 = 60 := by sorry
  have h2 : 5 * 12 = 60 := by sorry
  have h3 : 60 / 60 = 1 := by sorry
  have h4 : x + 1 = 91 := by sorry
  exact h4

end add_to_expression_to_get_91_l325_325425


namespace price_comparison_l325_325437

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def reduced_price (P : ℝ) : ℝ := 0.75 * P
noncomputable def increased_price (P : ℝ) : ℝ := reduced_price(P) * 1.30

theorem price_comparison (P : ℝ) (hP : 0 < P) : 
  ¬ (increased_price(P) > original_price(P)) :=
by
  unfold original_price reduced_price increased_price
  calc
    0.75 * P * 1.30 = 0.975 * P : by ring
    _ ≤ P : by { have : 0.975 < 1 := by norm_num, linarith }

end price_comparison_l325_325437


namespace steel_ball_radius_correct_l325_325803

def radius_cylinder : ℝ := 3
def initial_water_depth : ℝ := 8
def final_water_depth : ℝ := 8.5
def steel_ball_radius : ℝ := 1.06

theorem steel_ball_radius_correct :
  let h := final_water_depth - initial_water_depth in
  let V_cylinder := π * radius_cylinder^2 * h in
  let V_sphere := (4 / 3) * π * steel_ball_radius^3 in
  V_cylinder = V_sphere :=
by
  sorry

end steel_ball_radius_correct_l325_325803


namespace unique_solution_l325_325865

-- We declare an arbitrary function f from ℕ to ℕ
variable (f : ℕ → ℕ)

-- The condition on the function f
def condition : Prop :=
  ∀ n : ℕ, f(f(n)) < f(n + 1)

-- The main theorem stating that the only function satisfying the condition is the identity function
theorem unique_solution : condition f → ∀ n : ℕ, f(n) = n :=
by
  sorry

end unique_solution_l325_325865


namespace circle_regions_l325_325273

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l325_325273


namespace pennies_thrown_total_l325_325700

theorem pennies_thrown_total (rachelle_pennies gretchen_pennies rocky_pennies : ℕ) 
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) : 
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 := 
by 
  sorry

end pennies_thrown_total_l325_325700


namespace third_part_of_156_division_proof_l325_325962

theorem third_part_of_156_division_proof :
  ∃ (x : ℚ), (2 * x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 156) ∧ ((1 / 4) * x = 13 + 15 / 23) :=
by
  sorry

end third_part_of_156_division_proof_l325_325962


namespace find_a_given_max_value_l325_325599

variable (a : ℝ) (x : ℝ)

def y_function (a x : ℝ) : ℝ := a^(2 * x) + 2 * a^x - 1

theorem find_a_given_max_value :
  (∀ x, -1 ≤ x ∧ x ≤ 1) → 
  (a > 0 ∧ a ≠ 1) → 
  (∀ x, y_function a x ≤ 14) → 
  (∃ (a : ℝ), (a = 3 ∨ a = 1/3)) :=
by 
  sorry

end find_a_given_max_value_l325_325599


namespace pure_imaginary_a_l325_325628

theorem pure_imaginary_a : ∀ (a : ℝ), (i : ℂ) (hi : i^2 = -1), 
  ((1 + 2 * i) * (1 + a * i)).re = 0 → a = 1 / 2 :=
by
  intros a i hi hRealPartZero
  sorry

end pure_imaginary_a_l325_325628


namespace parabola_properties_l325_325122

-- Define the parabolic equation and its properties
def parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Point on the parabola
def point_P (m : ℝ) : ℝ × ℝ := (m, 2)

-- Focus of the parabola
def focus_F : ℝ × ℝ := (1 / 2, 0)

-- Distance formula
def distance (P F : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

theorem parabola_properties (m : ℝ) : 
  parabola m 2 ∧ point_P m = (2, 2) ∧ distance (point_P 2) focus_F = 5 / 2 :=
by
  sorry

end parabola_properties_l325_325122


namespace number_of_triangles_is_correct_l325_325137

def points := Fin 6 × Fin 6

def is_collinear (p1 p2 p3 : points) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2)

noncomputable def count_triangles_with_positive_area : Nat :=
  let all_points := Finset.univ.product Finset.univ
  let all_combinations := all_points.powerset.filter (λ s, s.card = 3)
  let valid_triangles := all_combinations.filter (λ s, ¬is_collinear (s.choose 0) (s.choose 1) (s.choose 2))
  valid_triangles.card

theorem number_of_triangles_is_correct :
  count_triangles_with_positive_area = 6804 :=
by
  sorry

end number_of_triangles_is_correct_l325_325137


namespace ordered_pair_solution_l325_325531

theorem ordered_pair_solution :
  ∃ (x y : ℚ), 
  (3 * x - 2 * y = (6 - 2 * x) + (6 - 2 * y)) ∧
  (x + 3 * y = (2 * x + 1) - (2 * y + 1)) ∧
  x = 12 / 5 ∧
  y = 12 / 25 :=
by
  sorry

end ordered_pair_solution_l325_325531


namespace perpendicular_tangent_line_eqn_l325_325922

theorem perpendicular_tangent_line_eqn :
  ∀ x y : ℝ, (x = Real.pi / 3) ∧ (y = 1 / 2) →
  (∀ (m : ℝ), (m = 2 / Real.sqrt 3) →
  (∃ k : ℝ, 2 * x - Real.sqrt 3 * y + k = 0) →
  k = - (2 * Real.pi / 3) + (Real.sqrt 3 / 2))

end perpendicular_tangent_line_eqn_l325_325922


namespace train_length_correct_l325_325008

noncomputable def length_of_train (speed_train speed_man time_pass : ℝ) : ℝ :=
  let relative_speed_kmhr := speed_train + speed_man
  let relative_speed_ms := (relative_speed_kmhr * 1000) / 3600
  relative_speed_ms * time_pass

theorem train_length_correct :
  length_of_train 27 6 11.999040076793857 ≈ 110 := by
  sorry

end train_length_correct_l325_325008


namespace cookout_kids_2004_l325_325634

variable (kids2005 kids2004 kids2006 : ℕ)

theorem cookout_kids_2004 :
  (kids2006 = 20) →
  (2 * kids2005 = 3 * kids2006) →
  (2 * kids2004 = kids2005) →
  kids2004 = 60 :=
by
  intros h1 h2 h3
  sorry

end cookout_kids_2004_l325_325634


namespace car_sales_solutions_l325_325793

-- Define the conditions
variable (this_year_revenue last_year_revenue cost_per_A cost_per_B total_budget total_cars min_cars_A : ℕ)

-- Constants specified in the problem
def cost_per_A := 0.75
def cost_per_B := 0.6
def total_budget := 1.05
def total_cars := 15
def min_cars_A := 6
def this_year_revenue := 900000
def last_year_revenue := 1000000

-- Define the main goal as a theorem
theorem car_sales_solutions :
  ∃ (x : ℝ) (purchasing_plans : list ℕ),
    -- Condition for revenue calculations with price reduction
    (let sales_last_year := last_year_revenue / (x + 1)
         sales_this_year := this_year_revenue / x in
     sales_last_year = 100 ∧ sales_this_year = 90 ∧ x = 9) ∧
    -- Condition for purchasing plans
    ( ∀ y : ℕ, y ≥ min_cars_A ∧ y ≤ 10 → purchasing_plans.length = 5) :=
by sorry

end car_sales_solutions_l325_325793


namespace quadratic_roots_l325_325810

theorem quadratic_roots (A B C : ℝ) (r s p : ℝ) (h1 : 2 * A * r^2 + 3 * B * r + 4 * C = 0)
  (h2 : 2 * A * s^2 + 3 * B * s + 4 * C = 0) (h3 : r + s = -3 * B / (2 * A)) (h4 : r * s = 2 * C / A) :
  p = (16 * A * C - 9 * B^2) / (4 * A^2) :=
by
  sorry

end quadratic_roots_l325_325810


namespace radius_of_intersecting_circles_l325_325750

-- Define the centers of the circles and the intersection points A and B
variables (O1 O2 A B : Type)
variables [metric_space O1] [metric_space O2] [metric_space A] [metric_space B]

-- Define properties for circles
def circles_pass_through_centers (R : ℝ) := 
  dist O1 O2 = R ∧ dist O1 A = R ∧ dist O2 A = R ∧ dist O1 B = R ∧ dist O2 B = R

-- The given condition for area of quadrilateral
def area_quad_O1AO2B (O1 O2 A B : Type) (R : ℝ) : Prop := 
  dist O1 O2 = R ∧ dist O1 A = R ∧ dist O2 A = R ∧ dist O1 B = R ∧ dist O2 B = R ∧
  (sqrt 3 / 2 * R^2 = 2 * sqrt 3)

-- The main theorem
theorem radius_of_intersecting_circles (R : ℝ) (h : area_quad_O1AO2B O1 O2 A B R) : R = 2 :=
sorry

end radius_of_intersecting_circles_l325_325750


namespace isosceles_triangle_largest_angle_l325_325641

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_triangle : A + B + C = 180) 
  (h_isosceles : A = B) (h_given_angle : A = 40) : C = 100 :=
by
  sorry

end isosceles_triangle_largest_angle_l325_325641


namespace inverse_of_matrixA_l325_325869

def matrixA := ![
  [5, -3],
  [4, -2]
]

def inverseA := ![
  [-1, 1.5],
  [-2, 2.5]
]

def zeroMatrix := ![
  [0, 0],
  [0, 0]
]

noncomputable def inverseOrZero (A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
if A.det = 0 then zeroMatrix else inverseA

theorem inverse_of_matrixA : inverseOrZero matrixA = inverseA := by
  sorry

end inverse_of_matrixA_l325_325869


namespace directrix_hyperbola_l325_325925

def asymptotes_hyperbola := ∀ x y : ℝ, (x = 2 * y) ∨ (x = -2 * y)

def foci_on_y_axis := ∀ x : ℝ, ∃ y : ℝ, (x = 0)

def min_distance_to_A (P : ℝ × ℝ) (A : (ℝ × ℝ)) := sqrt ((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)

def equation_hyperbola (hyperbola_lambda : ℝ) (x y : ℝ) := 4 * y^2 - x^2 = 4 * hyperbola_lambda

theorem directrix_hyperbola :
  ∃ l : ℝ, (l > 0) ∧
  (asymptotes_hyperbola ∧ foci_on_y_axis ∧ min_distance_to_A P (5, 0) = sqrt(6)) →
  (equation_hyperbola 1 P.1 P.2) →
  (∃ y, (y = ± (sqrt 5) / 5)) :=
sorry

end directrix_hyperbola_l325_325925


namespace f_monotonically_decreasing_g_minimum_value_l325_325594

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * Real.log x

theorem f_monotonically_decreasing {a : ℝ} (h : ∀ x ∈ set.Icc (3 : ℝ) 5, deriv (f x) (λ x, 2 * x - a / x) ≤ 0) :
  a ≥ 50 :=
sorry

noncomputable def g (x a b : ℝ) : ℝ := x^2 + 2 * Real.log x - 2 * (b - 1) * x

theorem g_minimum_value {a b : ℝ} (h₁ : b ≥ 7 / 2) (h₂ : ∃ x1 x2 : ℝ, x1 < x2 ∧ deriv (g x1) (λ x, 2 * x + 2 / x - 2 * (b-1)) = 0 ∧ deriv (g x2) (λ x, 2 * x + 2 / x - 2 * (b-1)) = 0) :
  ∃ x1 x2 : ℝ, g x1 a b - g x2 a b = 15 / 4 - 4 * Real.log 2 :=
sorry

end f_monotonically_decreasing_g_minimum_value_l325_325594


namespace more_vegetables_to_plant_l325_325320

def tomatoes := 3 * 5
def cucumbers := 5 * 4
def potatoes := 30
def total_spaces := 10 * 15
def planted_vegetables := tomatoes + cucumbers + potatoes
def additional_vegetables := total_spaces - planted_vegetables

theorem more_vegetables_to_plant : additional_vegetables = 85 := 
by
  have h1 : tomatoes = 15 := by norm_num
  have h2 : cucumbers = 20 := by norm_num
  have h3 : planted_vegetables = 65 := by
    have h4 : planted_vegetables = tomatoes + cucumbers + potatoes := rfl
    rw [h1, h2, rfl] at h4
    norm_num at h4
    exact h4
  have h5 : total_spaces = 150 := by norm_num
  have h6 : additional_vegetables = total_spaces - planted_vegetables := rfl
  rw [h5, h3] at h6
  norm_num at h6
  exact h6

end more_vegetables_to_plant_l325_325320


namespace regions_divided_by_radii_circles_l325_325243

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l325_325243


namespace min_sum_reciprocal_roots_l325_325937

variable (b c : ℝ)

def f (x : ℝ) := 2 * x^2 + b * x + c

noncomputable def x1 : ℝ := -- positive root
noncomputable def x2 : ℝ := -- positive root

theorem min_sum_reciprocal_roots :
  (f (-10) = f 12) ->
  (f x1 = 0) ->
  (f x2 = 0) ->
  (0 < x1) ->
  (0 < x2) ->
  x1 + x2 = 2 ->
  1/x1 + 1/x2 = 2 :=
begin
  sorry
end

end min_sum_reciprocal_roots_l325_325937


namespace max_ratio_of_three_digit_to_sum_l325_325856

theorem max_ratio_of_three_digit_to_sum (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9)
  (hc : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c) / (a + b + c) ≤ 100 :=
by sorry

end max_ratio_of_three_digit_to_sum_l325_325856


namespace factor_complete_polynomial_l325_325049

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end factor_complete_polynomial_l325_325049


namespace point_M_on_y_axis_l325_325199

theorem point_M_on_y_axis (t : ℝ) (h : t - 3 = 0) : (t-3, 5-t) = (0, 2) :=
by
  sorry

end point_M_on_y_axis_l325_325199


namespace find_ellipse_equation_l325_325913

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

noncomputable def is_focus (E : ℝ × ℝ → Prop) (F : ℝ × ℝ) : Prop :=
  F = (3, 0)

noncomputable def midpoint_ab (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem find_ellipse_equation (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) 
(h₃ : is_focus (ellipse_equation a b) (3, 0)) : 
  ((1,-1) = midpoint_ab (x₁, y₁) (x₂, y₂)) → 
  ellipse_equation 18 9 :=
sorry

end find_ellipse_equation_l325_325913


namespace count_squares_within_region_l325_325509

theorem count_squares_within_region : 
  let region := {p : ℤ × ℤ | p.2 < 2 * p.1 ∧ p.2 > -1 ∧ p.1 < 5}
  let squares := {p : ℤ × ℤ | p ∈ region ∧ (p.1 - 1, p.2) ∈ region ∧ (p.1, p.2 - 1) ∈ region ∧ (p.1 - 1, p.2 - 1) ∈ region}
  squares.card = 16 :=
by
  sorry

end count_squares_within_region_l325_325509


namespace max_value_of_f_l325_325936

noncomputable def f (a x : ℝ) := a * sin x * cos x - sin x ^ 2 + 1 / 2

theorem max_value_of_f (a : ℝ) (h_sym : ∀ x : ℝ, f a x = f a (π/6 - x)) :
  ∃ x : ℝ, f a x = 1 := sorry

end max_value_of_f_l325_325936


namespace binom_20_10_l325_325571

open_locale nat

theorem binom_20_10 :
  (nat.choose 18 8 = 31824) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 172822 :=
by {
  intros h1 h2 h3,
  sorry
}

end binom_20_10_l325_325571


namespace circles_radii_divide_regions_l325_325262

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l325_325262


namespace tan_product_simplification_l325_325356

theorem tan_product_simplification :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  have h : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have tan_addition :
    ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) := Real.tan_add
  sorry

end tan_product_simplification_l325_325356


namespace intersection_at_most_one_l325_325117

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the statement to be proved
theorem intersection_at_most_one (a : ℝ) :
  ∀ x1 x2 : ℝ, f x1 = f x2 → x1 = x2 :=
by
  sorry

end intersection_at_most_one_l325_325117


namespace chord_line_equation_of_hyperbola_l325_325386

theorem chord_line_equation_of_hyperbola (x_1 y_1 x_2 y_2 : ℝ)
  (h_hyperbola1 : x_1^2 - y_1^2 = 1)
  (h_hyperbola2 : x_2^2 - y_2^2 = 1)
  (h_midpoint_x : x_1 + x_2 = 4)
  (h_midpoint_y : y_1 + y_2 = 2) :
  ∀ x y : ℝ, (y - 1) = 2 * (x - 2) → y = 2 * x - 3 := 
by
  intro a b h
  apply eq.myapplication 
  sorry

end chord_line_equation_of_hyperbola_l325_325386


namespace digits_in_base_5_l325_325955

theorem digits_in_base_5 (n : ℕ) (h : n = 1234) (h_largest_power : 5^4 < n ∧ n < 5^5) : 
  ∃ digits : ℕ, digits = 5 := 
sorry

end digits_in_base_5_l325_325955


namespace coloring_grid_l325_325848

theorem coloring_grid :
  ∃ (colorings : Finset (Fin 10 → Fin 3)), 
    (∀ (grid : Finset (Fin 10 → Fin 3)),
      grid.card = 1026 ↔
      ∃! (i : Fin 3), ∃! (j : Fin 3), ∃! (k : Fin 3), 
        (i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ (∃ (c : colorings), c 0 = i ∧ c 1 = i ∧ c 2 ≠ i ∧ c 3 ≠ i)) ∧
        (∃ (c : colorings), c 4 = j ∧ c 5 = j ∧ c 6 ≠ j ∧ c 7 ≠ j) ∧
        (∃ (c : colorings), c 8 = k ∧ c 9 = k ∧ c 10 ≠ k ∧ c 11 ≠ k)
    ) :=
  sorry

end coloring_grid_l325_325848


namespace total_children_l325_325332

theorem total_children (sons daughters : ℕ) (h1 : sons = 3) (h2 : daughters = 6 * sons) : (sons + daughters) = 21 :=
by
  sorry

end total_children_l325_325332


namespace divide_into_palindromic_pieces_example_requiring_at_least_15_pieces_l325_325419

-- Definition of a palindrome
def is_palindrome (s : List Bool) : Prop :=
  s = s.reverse

-- Main proof statement
theorem divide_into_palindromic_pieces :
  ∀ (s : List Bool), list.length s = 60 → ∃ (pieces : List (List Bool)), 
  list.length pieces ≤ 24 ∧ (∀ piece ∈ pieces, is_palindrome piece) := 
by
  sorry

-- Example requiring at least 15 pieces
theorem example_requiring_at_least_15_pieces :
  ∃ (s : List Bool), list.length s = 60 ∧ 
  (∀ (pieces : List (List Bool)), (∀ piece ∈ pieces, is_palindrome piece) → 
  list.length pieces ≥ 15) :=
by
  sorry

end divide_into_palindromic_pieces_example_requiring_at_least_15_pieces_l325_325419


namespace simplify_tan_expression_l325_325359

theorem simplify_tan_expression :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 := 
by 
  -- Angle addition formula for tangent
  have h : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12), 
  from by rw [Real.tan_add]; exact Real.tan_pi_div_four,
  -- Given that tan 45° = 1
  have h1 : Real.tan (Real.pi / 4) = 1, from Real.tan_pi_div_four,
  -- Derive the known value
  rw [Real.tan_pi_div_four, h] at h1,
  -- Simplify using the derived value
  suffices : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 
             1 + Real.tan (Real.pi / 6) + Real.tan (Real.pi / 12) + Real.tan (Real.pi / 6) * Real.tan (Real.pi / 12), 
  from by rw this; simp [←h1],
  sorry

end simplify_tan_expression_l325_325359


namespace distance_from_point_to_x_axis_l325_325587

theorem distance_from_point_to_x_axis (x y : ℤ) (h : (x, y) = (5, -12)) : |y| = 12 :=
by
  -- sorry serves as a placeholder for the proof
  sorry

end distance_from_point_to_x_axis_l325_325587


namespace g_increasing_on_neg_infinity_to_1_l325_325596

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def f_inv (y : ℝ) : ℝ := (y + 1) / (y - 1)
noncomputable def g (x : ℝ) : ℝ := f_inv (1 / x)

theorem g_increasing_on_neg_infinity_to_1 :
  ∀ x y : ℝ, x < y ∧ y < 1 → g(x) < g(y) :=
by
  sorry

end g_increasing_on_neg_infinity_to_1_l325_325596


namespace area_triangle_AOB_l325_325333

-- Definition for curve C in Cartesian system
def curve_C (t : ℝ) : ℝ × ℝ := (4 * t^2, 4 * t)

-- Definition for line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := ρ * (cos θ - sin θ) = 4

theorem area_triangle_AOB :
  ∃ (A B : ℝ × ℝ),
    -- Points A and B lie on both the parabola and the line
    (∃ t1 t2 : ℝ, curve_C t1 = A ∧ curve_C t2 = B ∧
    A.snd^2 = 4 * A.fst ∧ A.fst - A.snd - 4 = 0 ∧
    B.snd^2 = 4 * B.fst ∧ B.fst - B.snd - 4 = 0) ∧
    -- Distance AB is 4√10
    (|real.sqrt ((A.fst - B.fst) ^ 2 + (A.snd - B.snd) ^ 2)| = 4 * real.sqrt 10) ∧
    -- Distance from origin O to the line l
    (let d := |0 - 0 - 4| / real.sqrt 2 in 
    d = 2 * real.sqrt 2) ∧
    -- Area of triangle AOB
    (1 / 2 * d * |real.sqrt ((A.fst - B.fst) ^ 2 + (A.snd - B.snd) ^ 2)| = 8 * real.sqrt 5)
:= sorry

end area_triangle_AOB_l325_325333


namespace isosceles_triangle_ratio_l325_325731

theorem isosceles_triangle_ratio (a b : ℕ) (ABC : Triangle) (H : Point)
  (h_isosceles : ABC.b = ABC.c) 
  (h_altitudes_meet : orthocenter ABC = H)
  (h_H_incircle : incircle ABC H) :
  a : a : b = 3 : 3 : 4 := 
sorry

end isosceles_triangle_ratio_l325_325731


namespace midpoint_meeting_point_l325_325611

theorem midpoint_meeting_point : 
  let h := (10, -3)
  let s := (-2, 7)
  let midpoint := ( (fst h + fst s) / 2, (snd h + snd s) / 2 )
  in midpoint = (4, 2) :=
by
  sorry

end midpoint_meeting_point_l325_325611


namespace positive_rationals_l325_325436

variable {Q : Type} [LinearOrderedField Q]

def is_closed_add (X : set Q) : Prop :=
∀ x y, x ∈ X → y ∈ X → x + y ∈ X

def is_closed_mul (X : set Q) : Prop :=
∀ x y, x ∈ X → y ∈ X → x * y ∈ X

noncomputable def X : set Q := { q : Q | q > 0 }

theorem positive_rationals
  (X : set Q)
  (h1 : X ⊆ { q : Q | ∃ n : ℕ, q = n } ∪ { q : Q | ∃ n : ℕ, q = (1 : Q) / n })
  (h2 : is_closed_add X)
  (h3 : is_closed_mul X)
  (h4 : (0 : Q) ∉ X)
  (h5 : ∀ q, q ≠ 0 → (q ∈ X ↔ -q ∉ X)) :
  X = { q : Q | q > 0 } :=
sorry

end positive_rationals_l325_325436


namespace line_equation_correct_l325_325376

-- Define the problem conditions
def x_intercept : ℝ := 2
def inclination_angle : ℝ := 135
def slope : ℝ := -1 -- since tan 135° = -1

-- Define the expected equation of the line
def expected_line_equation (x : ℝ) : ℝ := -x + 2

-- Prove that the equation of the line is as expected
theorem line_equation_correct : ∀ (x y : ℝ), (y = 0 ∧ x = 2) ∧ inclination_angle = 135 → y = -x + 2 := by
  intros x y h
  cases h with h1 h2
  cases h1 with hy hx
  simp [hx, hy, slope, expected_line_equation]
  sorry

end line_equation_correct_l325_325376


namespace square_103_square_998x1002_l325_325836

theorem square_103 : 103^2 = 10609 :=
by
  have H : (100 + 3)^2 = 100^2 + 2 * 100 * 3 + 3^2 := by norm_num
  norm_num at H
  exact H

theorem square_998x1002 : 998 * 1002 = 999996 :=
by
  have H : (1000 - 2) * (1000 + 2) = 1000^2 - 2^2 := by norm_num
  norm_num at H
  exact H

end square_103_square_998x1002_l325_325836


namespace complex_multiplication_value_l325_325536

theorem complex_multiplication_value (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end complex_multiplication_value_l325_325536


namespace circle_region_count_l325_325237

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l325_325237


namespace total_amount_is_10350_l325_325461

variable (N_50 N_500 : ℕ)
variable (total_amount : ℕ)

-- Definitions
def condition1 : Prop := N_50 = 57
def condition2 : Prop := N_50 + N_500 = 72
def total_amount_proof : Prop := total_amount = (N_50 * 50) + (N_500 * 500)

-- Theorem Statement
theorem total_amount_is_10350 (h1 : condition1) (h2 : condition2) : total_amount = 10350 :=
by {
  sorry
}

end total_amount_is_10350_l325_325461


namespace numberOfEvenIntegersInRangeIs6_l325_325129

-- Define the fractions in the problem
def frac1 : ℚ := 13 / 3
def frac2 : ℚ := 52 / 3

-- Define the range within the integer part of the fractions
def lower_bound : ℕ := frac1.ceil
def upper_bound : ℕ := frac2.floor

-- Define the set of even integers within the range
def evenIntegersInRange (a b : ℕ) : List ℕ :=
  (List.range (b - a + 1)).map (λ n, a + n).filter (λ n, n % 2 = 0)

-- Define the number of even integers between the two fractions
def numberOfEvenIntegersInRange : ℕ :=
  evenIntegersInRange lower_bound upper_bound

-- State the theorem we want to prove
theorem numberOfEvenIntegersInRangeIs6 : numberOfEvenIntegersInRange = 6 := by
  -- this is where the proof would be placed
  sorry

end numberOfEvenIntegersInRangeIs6_l325_325129


namespace line_passes_fixed_point_l325_325539

theorem line_passes_fixed_point :
  ∀ (m : ℝ), ∃ (x y : ℝ), (2m + 1)*x + (m + 1)*y - 7m - 4 = 0 ∧ x = 3 ∧ y = 1 :=
by
  sorry

end line_passes_fixed_point_l325_325539


namespace sum_coefficients_l325_325678

noncomputable def s : ℕ → ℝ
| 0 => 3
| 1 => 6
| 2 => 14
| (n + 3) => a * s (n + 2) + b * s (n + 1) + c * s n

theorem sum_coefficients (a b c : ℝ) (h : s 3 = 36) : a + b + c = 13 :=
by
  sorry

end sum_coefficients_l325_325678


namespace verify_proportion_not_boarded_last_two_l325_325861

/-- Define the conditions under which the passenger proportions are to be calculated. -/
variables (n : ℕ)

/-- Define the proportion of passengers at any time who did not board at either of the last two docks. -/
def proportion_not_boarded_last_two :
  Prop :=
  let P := (1 / 4 : ℚ) + (1 / 4 : ℚ) - (1 / 4 * 1 / 10 : ℚ) in -- Proportion who boarded at either of the last two docks
  (1 - P = 21 / 40)

/-- Prove that the calculated proportion matches the given conditions. -/
theorem verify_proportion_not_boarded_last_two :
  proportion_not_boarded_last_two n :=
  by sorry

end verify_proportion_not_boarded_last_two_l325_325861


namespace password_count_l325_325794

noncomputable def permutations : ℕ → ℕ → ℕ
| n, r := (n! / (n-r)!)

theorem password_count:
    let n_eng := 26 in
    let r_eng := 2 in
    let n_num := 10 in
    let r_num := 2 in
    permutations n_eng r_eng * permutations n_num r_num = (26 * 25) * (10 * 9) :=
by
  sorry

end password_count_l325_325794


namespace a_n_le_n_div_n_plus_1_a_n_increasing_l325_325562

def a_n (n : ℕ) : ℝ := 
  if h : n > 0 then 
    classical.some (exists_unique a : ℝ, 0 < a ∧ a < 1 ∧ a^n + a - 1 = 0) 
  else 
    1

lemma a_n_pos (n : ℕ) (hn : n > 0) : 0 < a_n n :=
begin
  unfold a_n,
  split_ifs,
  exact (classical.some_spec (exists_unique a : ℝ, 0 < a ∧ a < 1 ∧ a^n + a - 1 = 0)).1,
end

lemma a_n_lt_one (n : ℕ) (hn : n > 0) : a_n n < 1 :=
begin
  unfold a_n,
  split_ifs,
  exact (classical.some_spec (exists_unique a : ℝ, 0 < a ∧ a < 1 ∧ a^n + a - 1 = 0)).2.1,
end

lemma a_n_spec (n : ℕ) (hn : n > 0) : a_n n ^ n + a_n n - 1 = 0 :=
begin
  unfold a_n,
  split_ifs,
  exact (classical.some_spec (exists_unique a : ℝ, 0 < a ∧ a < 1 ∧ a^n + a - 1 = 0)).2.right,
end

theorem a_n_le_n_div_n_plus_1 (n : ℕ) (hn : n > 0) : a_n n ≤ n / (n + 1) :=
begin
  sorry  -- Proof to show that ∀ n, 0 < a_n n ≤ n / (n + 1)
end

theorem a_n_increasing (n : ℕ) (hn : n > 0) : a_n n < a_n (n + 1) :=
begin
  sorry  -- Proof to show that ∀ n, a_n n < a_n (n + 1)
end

end a_n_le_n_div_n_plus_1_a_n_increasing_l325_325562


namespace sum_of_angles_nine_pointed_star_l325_325699

open Real

-- Definitions
def points_on_circle (n : ℕ) (circle : Set Point) : Prop :=
  ∀ (i j : ℕ), i ≠ j → dist (circle_point i) (circle_point j) = (360 / n) * abs (i - j)

def nine_pointed_star (circle : Set Point) : Prop :=
  points_on_circle 9 circle ∧ is_9_pointed_star circle

-- Theorem statement
theorem sum_of_angles_nine_pointed_star {circle : Set Point} (h : nine_pointed_star circle) :
  ∑ i in finset.range 9, angle (tip i) = 720 :=
sorry

end sum_of_angles_nine_pointed_star_l325_325699


namespace min_value_of_z_l325_325040

theorem min_value_of_z : ∃ x : ℝ, ∀ y : ℝ, 5 * x^2 + 20 * x + 25 ≤ 5 * y^2 + 20 * y + 25 :=
by
  sorry

end min_value_of_z_l325_325040


namespace max_ratio_three_digit_l325_325853

theorem max_ratio_three_digit (x a b c : ℕ) (h1 : 100 * a + 10 * b + c = x) (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : 0 ≤ b ∧ b ≤ 9) (h4 : 0 ≤ c ∧ c ≤ 9) : 
  (x : ℚ) / (a + b + c) ≤ 100 := sorry

end max_ratio_three_digit_l325_325853


namespace sequence_count_105_l325_325081

theorem sequence_count_105:
  let a : Fin 16 → Int := λ n, ite (n = 0) 1 (lambda k, k = 15 → -10 | n - 1);
  (∀ k : Fin 15, abs (a k.succ - a k) = 1) →
  (a 0 = 1) →
  (a 15 = -10) →
  (n : ℕ), n = 105 :=
by
  sorry

end sequence_count_105_l325_325081


namespace sum_of_initial_terms_that_reach_4_after_7_steps_l325_325195

def transformation (N : ℕ) : ℕ :=
  if N % 2 = 0 then N / 2 else 3 * N + 2

theorem sum_of_initial_terms_that_reach_4_after_7_steps :
  (Σ i in {8, 16, 32, 64, 128, 256, 512}, i) = 1016 :=
by
  sorry

end sum_of_initial_terms_that_reach_4_after_7_steps_l325_325195


namespace no_single_face_with_distinct_edges_l325_325414

variables {Polyhedron : Type} [ConvexPolyhedron Polyhedron]
variables (F : Polyhedron → Prop)
variables (w_1 w_2 : Polyhedron → ℕ)
variables (u : Polyhedron → ℤ)

theorem no_single_face_with_distinct_edges (ℓ_1 ℓ_2 : Polyhedron → Prop) :
  (∀ P : Polyhedron, F P → w_1 P ≠ w_2 P) → false :=
by
  sorry

end no_single_face_with_distinct_edges_l325_325414


namespace P_n_roots_real_and_negative_l325_325681

noncomputable def cyc (n : ℕ) (σ : equiv.perm (fin n)) : ℕ :=
-- implementation of cyc

noncomputable def P_n (n : ℕ) : polynomial ℤ :=
polynomial.sum (f : equiv.perm (fin n)), (polynomial.C (1 : ℤ)) * polynomial.X ^ (cyc n f)

theorem P_n_roots_real_and_negative (n : ℕ) (h : 1 ≤ n) : 
    ∃ roots : multiset ℤ,
      (∀ root ∈ roots, ∃ k : ℕ, root = -(k + 1)) ∧
      P_n n = polynomial.prod_X_sub_C roots :=
sorry

end P_n_roots_real_and_negative_l325_325681


namespace collinear_O1_O2_K_L_l325_325666

-- Definitions of the given conditions
variables (A B C D K L O1 O2 : Type)
variables [RightAngledTriangle A B C] [FootOfAltitude A BC D] [Incenter A D B O1] [Incenter A D C O2]
variables [Circle CenterA A RadiusAD (inter C AB K)] [Circle CenterA A RadiusAD (inter C AC L)]

-- The proof goal
theorem collinear_O1_O2_K_L : Collinear [O1, O2, K, L] := 
sorry

end collinear_O1_O2_K_L_l325_325666


namespace population_approx_20000_l325_325045

def population (year : ℕ) : ℕ :=
  if year < 2000 then 0 else 250 * 4 ^ ((year - 2000) / 20)

theorem population_approx_20000 : ∃ year, abs (20000 - population year) < 4000 ∧ year = 2060 := 
by
  sorry

end population_approx_20000_l325_325045


namespace evaluate_expression_eq_l325_325759

theorem evaluate_expression_eq :
  let x := 2
  let y := -3
  let z := 7
  x^2 + y^2 - z^2 - 2 * x * y + 3 * z = -15 := by
    sorry

end evaluate_expression_eq_l325_325759


namespace different_signing_schemes_l325_325973

theorem different_signing_schemes (n : ℕ) (A B C : Set ℕ) :
  n = 5 → 
  A ∪ B ∪ C = {1, 2, 3, 4, 5} →
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ →
  A.card ≤ 2 ∧ B.card ≤ 2 ∧ C.card ≤ 2 →
  card {p : Π (s : Set ℕ), Set {A, B, C} // A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ A.card ≤ 2 ∧ B.card ≤ 2 ∧ C.card ≤ 2} = 90 :=
by
  sorry

end different_signing_schemes_l325_325973


namespace ordered_pairs_m_n_l325_325131

theorem ordered_pairs_m_n :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ (p.1 ^ 2 - p.2 ^ 2 = 72)) ∧ s.card = 3 :=
by
  sorry

end ordered_pairs_m_n_l325_325131


namespace ab_perp_cd_l325_325208

variables {V : Type*} [inner_product_space ℝ V]
variables (b c d : V)

-- Given conditions
def condition1 := 1/2 • (b + c) ⬝ (d - b) = 0
def condition2 := 1/2 • (b + d) ⬝ (c - b) = 0

-- Conclusion to prove
theorem ab_perp_cd (hb : condition1 b c d) (hc : condition2 b c d) : b ⬝ (d - c) = 0 :=
sorry

end ab_perp_cd_l325_325208


namespace compute_x_squared_y_plus_x_y_squared_l325_325929

open Real

theorem compute_x_squared_y_plus_x_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + 2 * x + 2 * y = 7) : 
  x^2 * y + x * y^2 = 245 / 121 := 
by 
  sorry

end compute_x_squared_y_plus_x_y_squared_l325_325929


namespace chloromethane_formation_l325_325057

variable (CH₄ Cl₂ CH₃Cl : Type)
variable (molesCH₄ molesCl₂ molesCH₃Cl : ℕ)

theorem chloromethane_formation 
  (h₁ : molesCH₄ = 3)
  (h₂ : molesCl₂ = 3)
  (reaction : CH₄ → Cl₂ → CH₃Cl)
  (one_to_one : ∀ (x y : ℕ), x = y → x = y): 
  molesCH₃Cl = 3 :=
by
  sorry

end chloromethane_formation_l325_325057


namespace colin_avg_speed_goal_l325_325029

-- Define a structure to hold the running segments
structure RunSegment where
  distance : ℝ
  speed : ℝ

-- Define the time calculation for a segment
def time_for_segment (seg : RunSegment) : ℝ :=
  seg.distance / seg.speed

-- Define Colin's run segments as per the problem conditions
def colins_run : List RunSegment :=
  [ { distance := 1.5, speed := 6.5 },
    { distance := 1.25, speed := 8 },
    { distance := 2.25, speed := 9.5 } ]

-- Define the total distance of the run
def total_distance (segments : List RunSegment) : ℝ :=
  segments.foldl (fun acc seg => acc + seg.distance) 0

-- Define the total time for the run
def total_time (segments : List RunSegment) : ℝ :=
  segments.foldl (fun acc seg => acc + time_for_segment(seg)) 0

-- Define the average speed calculation
def average_speed (segments : List RunSegment) : ℝ :=
  total_distance(segments) / total_time(segments)

-- Define the proof problem for average speed
theorem colin_avg_speed_goal : 
  abs (average_speed(colins_run) - 8.014) < 0.001 :=
sorry

end colin_avg_speed_goal_l325_325029


namespace championship_winner_l325_325438

def Wang_predictions (champion runner_up : String) : Prop :=
  champion = "D" ∧ runner_up = "B"

def Li_predictions (runner_up fourth : String) : Prop :=
  runner_up = "A" ∧ fourth = "C"

def Zhang_predictions (third runner_up : String) : Prop :=
  third = "C" ∧ runner_up = "D"

def half_correct (pred : Prop) : Prop := 
  -- Assuming a practical definition of "half correct" specific to the problem
  sorry

theorem championship_winner : 
  (∃ champion runner_up fourth third, 
    (Wang_predictions champion runner_up ∧ half_correct (Wang_predictions champion runner_up)) ∧
    (Li_predictions runner_up fourth ∧ half_correct (Li_predictions runner_up fourth)) ∧
    (Zhang_predictions third runner_up ∧ half_correct (Zhang_predictions third runner_up))
  ) → 
  champion = "D" :=
begin
  sorry
end

end championship_winner_l325_325438


namespace sum_fractions_l325_325111

noncomputable def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

theorem sum_fractions : 
  ∑ k in finset.range 2017, f ((k + 1) / 2018) = 1008.5 :=
by
  sorry

end sum_fractions_l325_325111


namespace number_of_triangles_in_6x6_grid_l325_325141

def is_valid_point (n : ℕ) (i j : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n

def number_of_triangles (n : ℕ) : ℕ :=
  let points := (Finset.range (n + 1)).product (Finset.range (n + 1))
  let valid_points := points.filter (λ p, is_valid_point n p.1 p.2)
  let all_triangles := valid_points.powersetLen 3
  let collinear (a b c : (ℕ × ℕ)) : Prop :=
    (b.1 - a.1) * (c.2 - a.2) = (c.1 - a.1) * (b.2 - a.2)
  let non_collinear_triangles := all_triangles.filter (λ t, ¬ collinear t.nthLe! 0 t.nthLe! 1 t.nthLe! 2)
  non_collinear_triangles.card

theorem number_of_triangles_in_6x6_grid : number_of_triangles 6 = 6804 := 
sorry

end number_of_triangles_in_6x6_grid_l325_325141


namespace sequence_difference_squared_l325_325942

theorem sequence_difference_squared {a : ℕ → ℕ} (S : ℕ → ℕ) (hSn : ∀ n : ℕ, S n = n^2) :
  let a2 := S 2 - S 1,
      a3 := S 3 - S 2 
  in a3^2 - a2^2 = 16 :=
by
  sorry

end sequence_difference_squared_l325_325942


namespace james_speed_is_16_l325_325281

theorem james_speed_is_16
  (distance : ℝ)
  (time : ℝ)
  (distance_eq : distance = 80)
  (time_eq : time = 5) :
  (distance / time = 16) :=
by
  rw [distance_eq, time_eq]
  norm_num

end james_speed_is_16_l325_325281


namespace log_abs_sin_properties_l325_325780

noncomputable def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f (x)

noncomputable def has_period (f : ℝ → ℝ) (period : ℝ) := ∀ x : ℝ, f (x + period) = f (x)

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : Set ℝ) := ∀ x y ∈ I, x < y → f x < f y

theorem log_abs_sin_properties :
  is_even (λ x:ℝ, Real.log (abs (Real.sin x))) ∧
  has_period (λ x:ℝ, Real.log (abs (Real.sin x))) Real.pi ∧
  is_monotonically_increasing (λ x:ℝ, Real.log (abs (Real.sin x))) (Set.Ioo 0 (Real.pi / 2)) :=
sorry

end log_abs_sin_properties_l325_325780


namespace total_children_l325_325330

theorem total_children (sons daughters : ℕ) (h1 : sons = 3) (h2 : daughters = 6 * sons) : (sons + daughters) = 21 :=
by
  sorry

end total_children_l325_325330


namespace kernels_popped_in_first_bag_l325_325318

theorem kernels_popped_in_first_bag :
  ∀ (x : ℕ), 
    (total_kernels : ℕ := 75 + 50 + 100) →
    (total_popped : ℕ := x + 42 + 82) →
    (average_percentage_popped : ℚ := 82) →
    ((total_popped : ℚ) / total_kernels) * 100 = average_percentage_popped →
    x = 61 :=
by
  sorry

end kernels_popped_in_first_bag_l325_325318


namespace circle_regions_division_l325_325250

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l325_325250


namespace alfred_gain_percent_l325_325772

theorem alfred_gain_percent : 
  let purchase_price := 4700
  let repair_costs := 600
  let selling_price := 5800
  let total_cost := purchase_price + repair_costs
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost.toFloat) * 100
  gain_percent ≈ 9.43 :=
by
  sorry

end alfred_gain_percent_l325_325772


namespace problem_l325_325858

-- Define the problem
theorem problem {a b c : ℤ} (h1 : a = c + 1) (h2 : b - 1 = a) :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 = 6 := 
sorry

end problem_l325_325858


namespace sin_2pi_minus_theta_l325_325550

theorem sin_2pi_minus_theta (theta : ℝ) (k : ℤ) 
  (h1 : 3 * Real.cos theta ^ 2 = Real.tan theta + 3)
  (h2 : theta ≠ k * Real.pi) :
  Real.sin (2 * (Real.pi - theta)) = 2 / 3 := by
  sorry

end sin_2pi_minus_theta_l325_325550


namespace maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l325_325997

open Real

theorem maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism 
  (a b : ℝ)
  (ha : a^2 + b^2 = 25) 
  (AC_eq_5 : AC = 5) :
  ∃ (r : ℝ), 4 * π * r^2 = 25 * (3 - 3 * sqrt 2) * π :=
sorry

end maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l325_325997


namespace cats_remained_on_island_l325_325014

theorem cats_remained_on_island : 
  ∀ (n m1 : ℕ), 
  n = 1800 → 
  m1 = 600 → 
  (n - m1) / 2 = 600 → 
  (n - m1) - ((n - m1) / 2) = 600 :=
by sorry

end cats_remained_on_island_l325_325014


namespace sin_18_eq_l325_325499

theorem sin_18_eq : ∃ x : Real, x = (Real.sin (Real.pi / 10)) ∧ x = (Real.sqrt 5 - 1) / 4 := by
  sorry

end sin_18_eq_l325_325499


namespace edward_rides_l325_325517

theorem edward_rides (total_tickets : ℕ) (spent_tickets : ℕ) (ride_cost : ℕ) :
  total_tickets = 325 ∧ spent_tickets = 115 ∧ ride_cost = 13 →
  (total_tickets - spent_tickets) / ride_cost = 16 :=
by
  intro h
  cases h with ht h1
  cases h1 with hs hr
  rw [ht, hs, hr]
  sorry

end edward_rides_l325_325517


namespace circle_division_l325_325225

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l325_325225


namespace cody_games_still_has_l325_325028

def initial_games : ℕ := 9
def games_given_away_to_jake : ℕ := 4
def games_given_away_to_sarah : ℕ := 2
def games_bought_over_weekend : ℕ := 3

theorem cody_games_still_has : 
  initial_games - (games_given_away_to_jake + games_given_away_to_sarah) + games_bought_over_weekend = 6 := 
by
  sorry

end cody_games_still_has_l325_325028


namespace cuboid_intersection_impossible_l325_325043

theorem cuboid_intersection_impossible :
  (∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 12) ∧ (1 ≤ j ∧ j ≤ 12) → 
    ((abs (i - j) ≠ 1 ∧ abs (i - j) ≠ 11) → (∃ (xi yi zi xj yj zj : ℝ), 
      ¬(xj < xi ∨ yj < yi ∨ zj < zi)))) → False :=
by
  /* Conditions definitions in Lean */
  sorry

end cuboid_intersection_impossible_l325_325043


namespace binom_20_10_l325_325568

open_locale nat

theorem binom_20_10 :
  (nat.choose 18 8 = 31824) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 172822 :=
by {
  intros h1 h2 h3,
  sorry
}

end binom_20_10_l325_325568


namespace hyperbola_distance_condition_l325_325099

theorem hyperbola_distance_condition
  (O F1 F2 P : Type) (dist : Type → Type → ℝ)
  (center_hyp : P → Prop)
  (on_hyperbola : P → Prop) :
  (on_hyperbola P → dist P F1 * dist P F2 = 6) →
  (on_hyperbola P → dist P O = sqrt 6) :=
by
  sorry

end hyperbola_distance_condition_l325_325099


namespace circle_region_count_l325_325234

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l325_325234


namespace visitors_not_ill_l325_325476

theorem visitors_not_ill (total_visitors : ℕ) (percent_ill : ℕ) (H1 : total_visitors = 500) (H2 : percent_ill = 40) : 
  total_visitors * (100 - percent_ill) / 100 = 300 := 
by 
  sorry

end visitors_not_ill_l325_325476


namespace expected_number_of_2s_when_three_dice_rolled_l325_325407

def probability_of_rolling_2 : ℚ := 1 / 6
def probability_of_not_rolling_2 : ℚ := 5 / 6

theorem expected_number_of_2s_when_three_dice_rolled :
  (0 * (probability_of_not_rolling_2)^3 + 
   1 * 3 * (probability_of_rolling_2) * (probability_of_not_rolling_2)^2 + 
   2 * 3 * (probability_of_rolling_2)^2 * (probability_of_not_rolling_2) + 
   3 * (probability_of_rolling_2)^3) = 
   1 / 2 :=
by
  sorry

end expected_number_of_2s_when_three_dice_rolled_l325_325407


namespace dogs_on_mon_wed_fri_l325_325949

def dogs_on_tuesday : ℕ := 12
def dogs_on_thursday : ℕ := 9
def pay_per_dog : ℕ := 5
def total_earnings : ℕ := 210

theorem dogs_on_mon_wed_fri :
  ∃ (d : ℕ), d = 21 ∧ d * pay_per_dog = total_earnings - (dogs_on_tuesday + dogs_on_thursday) * pay_per_dog :=
by 
  sorry

end dogs_on_mon_wed_fri_l325_325949


namespace number_of_triangles_in_6x6_grid_l325_325142

def is_valid_point (n : ℕ) (i j : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n

def number_of_triangles (n : ℕ) : ℕ :=
  let points := (Finset.range (n + 1)).product (Finset.range (n + 1))
  let valid_points := points.filter (λ p, is_valid_point n p.1 p.2)
  let all_triangles := valid_points.powersetLen 3
  let collinear (a b c : (ℕ × ℕ)) : Prop :=
    (b.1 - a.1) * (c.2 - a.2) = (c.1 - a.1) * (b.2 - a.2)
  let non_collinear_triangles := all_triangles.filter (λ t, ¬ collinear t.nthLe! 0 t.nthLe! 1 t.nthLe! 2)
  non_collinear_triangles.card

theorem number_of_triangles_in_6x6_grid : number_of_triangles 6 = 6804 := 
sorry

end number_of_triangles_in_6x6_grid_l325_325142


namespace password_count_correct_l325_325797

-- Defining variables
def n_letters := 26
def n_digits := 10

-- The number of permutations for selecting 2 different letters
def perm_letters := n_letters * (n_letters - 1)
-- The number of permutations for selecting 2 different numbers
def perm_digits := n_digits * (n_digits - 1)

-- The total number of possible passwords
def total_permutations := perm_letters * perm_digits

-- The theorem we need to prove
theorem password_count_correct :
  total_permutations = (n_letters * (n_letters - 1)) * (n_digits * (n_digits - 1)) :=
by
  -- The proof goes here
  sorry

end password_count_correct_l325_325797


namespace steps_from_center_to_square_l325_325015

-- Define the conditions and question in Lean 4
def steps_to_center := 354
def total_steps := 582

-- Prove that the steps from Rockefeller Center to Times Square is 228
theorem steps_from_center_to_square : (total_steps - steps_to_center) = 228 := by
  sorry

end steps_from_center_to_square_l325_325015


namespace closest_integer_13_minus_sqrt_13_l325_325730

theorem closest_integer_13_minus_sqrt_13 :
  9 = Int.closest (13 - Real.sqrt 13) :=
by 
  have h : 3.5 < Real.sqrt 13 ∧ Real.sqrt 13 < 4 := sorry
  have h1 : 9 < 13 - Real.sqrt 13 := by linarith [h.left]
  have h2 : 13 - Real.sqrt 13 < 9.5 := by linarith [h.right]
  exact Int.closest_spec (13 - Real.sqrt 13) (by linarith)

end closest_integer_13_minus_sqrt_13_l325_325730


namespace part_I_monotonic_increase_part_II_non_negative_l325_325115

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x - 1 - x^2 / 2

-- Part I: f(x) is monotonically increasing for a = 1 / 2
theorem part_I_monotonic_increase :
  (∀ x : ℝ, x ∈ set.univ → (∂ (f x (1/2))) / (∂ x) >= 0) := 
sorry

-- Part II: f(x) >= 0 for all x >= 0 if and only if a <= 1
theorem part_II_non_negative (a : ℝ) :
  (∀ x : ℝ, x >= 0 → f x a >= 0) ↔ (a <= 1) :=
sorry

end part_I_monotonic_increase_part_II_non_negative_l325_325115


namespace infinite_triples_exists_l325_325709

-- Definitions from conditions
def isPrime (p : ℤ) : Prop := Nat.Prime p
def moduloEq (a b n : ℤ) : Prop := (a - b) % n = 0
def divisibility (m n : ℤ) : Prop := n ∣ m

-- Equivalent proof problem rewritten in Lean 4
theorem infinite_triples_exists :
  ∃ (a b p : ℤ), isPrime p ∧ moduloEq p 1 3 ∧ 0 < a ∧ a ≤ b ∧ b < p ∧ divisibility (p^5) ((a + b)^p - a^p - b^p) :=
sorry

end infinite_triples_exists_l325_325709


namespace C_n_bound_l325_325670

def sum_of_digits (x : ℕ) : ℕ := 
  (to_digits 10 x).sum

def C_n (n : ℕ) : ℕ :=
  (finset.Ico 1 (10^n)).filter (λ x, sum_of_digits (2 * x) < sum_of_digits x).card

theorem C_n_bound (n : ℕ) (hn : n > 0) : 
  C_n n ≥ (4 * (10^n - 1) / 9) :=
by sorry

end C_n_bound_l325_325670


namespace cube_difference_l325_325095

theorem cube_difference {a b : ℝ} (h1 : a - b = 5) (h2 : a^2 + b^2 = 35) : a^3 - b^3 = 200 :=
sorry

end cube_difference_l325_325095


namespace correct_fraction_l325_325637

theorem correct_fraction (incorrect_fraction : ℚ) (num : ℕ) (difference : ℕ) 
  (h : incorrect_fraction * num = (5/16 : ℚ) * num + difference) : 
  incorrect_fraction = (5/16 : ℚ) :=
by {
  sorry
}

#eval correct_fraction (5/6) 96 50 (by norm_num)

end correct_fraction_l325_325637


namespace find_a_l325_325917

-- Definitions and conditions
def cond (a : ℝ) : Prop := 3 ∈ ({1, -a^2, a-1} : Set ℝ)

-- The statement to be proved
theorem find_a (a : ℝ) (h : cond a) : a = 4 := 
by
  sorry

end find_a_l325_325917


namespace triangle_count_l325_325132

theorem triangle_count (i j : ℕ) (h₁ : 1 ≤ i ∧ i ≤ 6) (h₂ : 1 ≤ j ∧ j ≤ 6): 
  let points := { (x, y) | 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 } in
  fintype.card { t : finset (ℕ × ℕ) // t.card = 3 ∧ ∃ a b c : ℕ × ℕ, a ∉ t ∧ b ∉ t ∧ c ∉ t ∧ abs ((b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)) ≠ 0 } = 6800 := 
by
  sorry

end triangle_count_l325_325132


namespace cos_double_x0_zero_l325_325113

noncomputable def f (x : ℝ) : ℝ := (sin x)^2 + 2 * sqrt 3 * sin x * cos x + sin (x + π / 4) * sin (x - π / 4)

theorem cos_double_x0_zero (x0 : ℝ) (h : 0 ≤ x0 ∧ x0 ≤ π / 2) (hx0 : f x0 = 0) : 
  cos (2 * x0) = (3 * sqrt 5 + 1) / 8 :=
by
  sorry

end cos_double_x0_zero_l325_325113


namespace circle_regions_l325_325277

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l325_325277


namespace total_sticks_used_l325_325901

-- Define the number of sides an octagon has
def octagon_sides : ℕ := 8

-- Define the number of sticks each subsequent octagon needs, sharing one side with the previous one
def additional_sticks_per_octagon : ℕ := 7

-- Define the total number of octagons in the row
def total_octagons : ℕ := 700

-- Define the total number of sticks used
def total_sticks : ℕ := 
  let first_sticks := octagon_sides
  let additional_sticks := additional_sticks_per_octagon * (total_octagons - 1)
  first_sticks + additional_sticks

-- Statement to prove
theorem total_sticks_used : total_sticks = 4901 := by
  sorry

end total_sticks_used_l325_325901


namespace unique_root_quadratic_trinomial_l325_325192

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

noncomputable def R (a b c : ℝ) : ℝ -> ℝ :=
  λ x, (a + c) * x^2 + 2 * b * x + (a + c)

theorem unique_root_quadratic_trinomial (a b c : ℝ) (h : quadratic_discriminant (a + c) 2 * b (a + c) = 0) :
  (R a b c (-1) = 0 ∧ R a b c = 0) ∨ (R a b c (1) = 0 ∧ R a b c = 0) :=
sorry

end unique_root_quadratic_trinomial_l325_325192


namespace sequence_property_l325_325395

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ k : ℕ, 0 ≤ a k

theorem sequence_property (a : ℕ → ℝ) :
  seq a →
  (∀ k : ℕ, a k - 2 * a (k + 1) + a (k + 2) ≥ 0) →
  (∀ k : ℕ, (∑ i in finset.range k, a (i + 1)) ≤ 1) →
  ∀ k : ℕ, 0 ≤ a k - a (k + 1) ∧ a k - a (k + 1) < 2 / (k ^ 2) :=
by
  sorry

end sequence_property_l325_325395


namespace range_of_a_l325_325940

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x - a| > x - 1) → (a < 1 ∨ a > 3) :=
by
  assume h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x - a| > x - 1
  sorry

end range_of_a_l325_325940


namespace total_pennies_l325_325703

theorem total_pennies (R G K : ℕ) (h1 : R = 180) (h2 : G = R / 2) (h3 : K = G / 3) : R + G + K = 300 := by
  sorry

end total_pennies_l325_325703


namespace percentage_women_not_speaking_French_but_speaking_Spanish_German_or_both_is_36_l325_325191

noncomputable def percentage_women_not_speaking_French_but_speaking_Spanish_German_or_both :
  ℝ :=
  let total_women := 0.40 in
  let women_not_speaking_French := total_women * (1 - 0.40) in
  let women_speaking_Spanish_or_German := 0.35 + 0.25 in -- assuming mutually exclusive
  women_not_speaking_French * women_speaking_Spanish_or_German * 100

theorem percentage_women_not_speaking_French_but_speaking_Spanish_German_or_both_is_36 :
  percentage_women_not_speaking_French_but_speaking_Spanish_German_or_both = 36 :=
by
  sorry

end percentage_women_not_speaking_French_but_speaking_Spanish_German_or_both_is_36_l325_325191


namespace solve_equation_l325_325755

theorem solve_equation (x : ℤ) : x * (x + 2) + 1 = 36 ↔ x = 5 :=
by sorry

end solve_equation_l325_325755


namespace circle_regions_l325_325270

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l325_325270


namespace limit_seq_zero_l325_325493

open Filter Real
open_locale TopologicalSpace

noncomputable def seq (n : ℕ) : ℝ :=
  (sqrt (n - 1) - sqrt (n ^ 2 + 1)) / (cbrt (3 * n ^ 3 + 3) + cbrt (n ^ 5 + 1))

theorem limit_seq_zero : tendsto (λ n : ℕ, seq n) at_top (𝓝 0) :=
sorry

end limit_seq_zero_l325_325493


namespace candy_removal_time_l325_325496

theorem candy_removal_time :
  let prism := { cuboid : ℕ × ℕ × ℕ // cuboid.1 ≤ 3 ∧ cuboid.2 ≤ 4 ∧ cuboid.3 ≤ 5 } in
  (∀ cuboid ∈ prism, ∃ t : ℕ, (cuboid.1 + cuboid.2 + cuboid.3 - 3) = t) ∧ 
  ∀ t > 0, (∃ S : ℕ, t = S ∧ S = 10) :=
by
sorry

end candy_removal_time_l325_325496


namespace positive_difference_l325_325399

theorem positive_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 3 * x = 27) : y - x = 9 :=
sorry

end positive_difference_l325_325399


namespace quadratic_function_properties_l325_325077

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * (x - 1) ^ 2 - (16 / 3)

theorem quadratic_function_properties :
  (f 0 = -5) ∧ (f (-1) = -4) ∧ (f 2 = -5) ∧
  (∀ x ∈ Set.Icc 0 5, f x ≤ f 5) ∧
  (∀ x ∈ Set.Icc 0 5, f x ≥ f 1) :=
by
  have h1 : f 0 = -5 := by
    rw [f]; norm_num
  have h2 : f (-1) = -4 := by
    rw [f]; norm_num
  have h3 : f 2 = -5 := by
    rw [f]; norm_num
  have max_on_interval : ∀ x ∈ Set.Icc 0 5, f x ≤ 0 := by
    intro x hx
    rw [f]
    let y := (1 / 3) * (x - 1)^2 - (16 / 3)
    have hy : y ≤ 0 := sorry -- Detailed proof omitted; demonstrates y ≤ 0 in desired interval.
    exact hy
  have min_on_interval : ∀ x ∈ Set.Icc 0 5, f x ≥ -16 / 3 := by
    intro x hx
    rw [f]
    let y := (1 / 3) * (x - 1)^2 - (16 / 3)
    have hy : y ≥ -16 / 3 := sorry -- Detailed proof omitted; demonstrates y ≥ -16 / 3 in desired interval.
    exact hy
  exact ⟨h1, h2, h3, max_on_interval, min_on_interval⟩

end quadratic_function_properties_l325_325077


namespace coeff_x4_l325_325490

theorem coeff_x4 :
  (∃ (f : Polynomial ℤ), f = 3 * (Polynomial.C 1 * X ^ 2 - Polynomial.C 1 * X ^ 4) - 2 * (Polynomial.C 1 * X ^ 3 - Polynomial.C 1 * X ^ 4 + Polynomial.C 1 * X ^ 6) + 5 * (Polynomial.C 2 * X ^ 4 - Polynomial.C 1 * X ^ 10)
  ⇒ f.coeff 4 = 9) :=
sorry

end coeff_x4_l325_325490


namespace circle_division_l325_325227

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l325_325227


namespace num_common_elements_1000_multiples_5_9_l325_325673

def multiples_up_to (n k : ℕ) : ℕ := n / k

def num_common_elements_in_sets (k m n : ℕ) : ℕ :=
  multiples_up_to n (Nat.lcm k m)

theorem num_common_elements_1000_multiples_5_9 :
  num_common_elements_in_sets 5 9 5000 = 111 :=
by
  -- The proof is omitted as per instructions
  sorry

end num_common_elements_1000_multiples_5_9_l325_325673


namespace train_speed_l325_325009

/-- Define the lengths of the train and the bridge and the time taken to cross the bridge. --/
def len_train : ℕ := 360
def len_bridge : ℕ := 240
def time_minutes : ℕ := 4
def time_seconds : ℕ := 240 -- 4 minutes converted to seconds

/-- Define the speed calculation based on the given domain. --/
def total_distance : ℕ := len_train + len_bridge
def speed (distance : ℕ) (time : ℕ) : ℚ := distance / time

/-- The statement to prove that the speed of the train is 2.5 m/s. --/
theorem train_speed :
  speed total_distance time_seconds = 2.5 := sorry

end train_speed_l325_325009


namespace average_up_to_17_innings_l325_325790

-- Definitions extracted from the conditions
def average (runs : ℕ) (innings : ℕ) : ℚ := runs / innings

variables (R18 : ℕ) (A18 : ℕ) (R18_eq : R18 = A18 * 18) (runs_in_18th : ℕ) (runs_in_18th_eq : runs_in_18th = 1)
variable (A18_eq18 : A18 = 18)

-- Theorem to be proven
theorem average_up_to_17_innings : 
  ∀ (R17 : ℕ), (R17 = R18 - runs_in_18th) -> (average R17 17 = 19) :=
by
  intros R17 R17_eq
  rw [R18_eq, A18_eq18, runs_in_18th_eq, R17_eq]
  rw [← Nat.cast_sub (by norm_num : 324 ≥ 1)]
  norm_num
  sorry

end average_up_to_17_innings_l325_325790


namespace tan_product_simplification_l325_325354

theorem tan_product_simplification :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  have h : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have tan_addition :
    ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) := Real.tan_add
  sorry

end tan_product_simplification_l325_325354


namespace ab_equality_l325_325684

theorem ab_equality (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_div : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := sorry

end ab_equality_l325_325684


namespace greatest_and_smallest_int_3_7_l325_325388

noncomputable theory

def greatest_int_leq (x : ℝ) : ℤ := int.floor x
def smallest_int_geq (x : ℝ) : ℤ := int.ceil x

theorem greatest_and_smallest_int_3_7 :
  greatest_int_leq 3.7 = 3 ∧ smallest_int_geq 3.7 = 4 :=
by
  sorry

end greatest_and_smallest_int_3_7_l325_325388


namespace solve_for_a_l325_325904

variable (a b x : ℝ)

theorem solve_for_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x := sorry

end solve_for_a_l325_325904


namespace circle_equation_fixed_point_l325_325064

theorem circle_equation_fixed_point :
  ∃ P : ℝ × ℝ, 
  (∀ a ∈ ℝ, (∀ x y : ℝ, (x + y - 1) - a * (x + 1) = 0 → P = (-1, 2)))
  → (∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 5 ↔ x^2 + y^2 + 2 * x - 4 * y = 0) :=
by
  use (-1, 2)
  intros h
  sorry

end circle_equation_fixed_point_l325_325064


namespace circles_common_tangents_l325_325751

theorem circles_common_tangents 
  {r1 r2 : ℝ} {c1 c2 : ℝ} 
  (h1 : r1 ≠ r2) 
  (h2 : c1 ≠ c2) 
  (h3 : (c1 + r1, 0) = (c2 - r2, 0)) :
  (number_of_common_tangents r1 r2) = 1 ∨ (number_of_common_tangents r1 r2) = 3 := 
sorry

end circles_common_tangents_l325_325751


namespace min_distance_PQ_l325_325070

theorem min_distance_PQ :
  let ellipse := {p : ℝ × ℝ | (p.1)^2 / 9 + (p.2)^2 / 4 = 1}
  let minor_axis := 2
  (∀ M ∈ ellipse,
    let P := (4 / (3 * cos (atan(M.2 / 2))), 0)
    let Q := (0, 4 / (2 * sin (atan(M.2 / 2))))
    ∃ α,
    |P.1 - Q.1| + |P.2 - Q.2| = 10 / 3)
:= sorry

end min_distance_PQ_l325_325070


namespace regions_divided_by_radii_circles_l325_325245

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l325_325245


namespace find_c_d_l325_325748

theorem find_c_d (C D : ℤ) (h1 : 3 * C - 4 * D = 18) (h2 : C = 2 * D - 5) :
  C = 28 ∧ D = 33 / 2 := by
sorry

end find_c_d_l325_325748


namespace circle_divided_into_regions_l325_325255

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l325_325255


namespace proposition_correct_l325_325932

-- Defining the propositions based on the problem statement
def p : Prop := ∃ x : ℝ, x^2 + x - 1 < 0
def not_p : Prop := ∀ x : ℝ, x^2 + x - 1 ≥ 0
def prop_2 : Prop := p ∧ q → p ∧ ¬q
def prop_3 : Prop := (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 2) = (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 2)

-- The Lean statement proving Proposition ① is correct.
theorem proposition_correct : 
  (p ↔ not_p) ∧ ¬prop_2 ∧ ¬prop_3 :=
by
  sorry

end proposition_correct_l325_325932


namespace significant_points_cyclic_l325_325908

-- Define the necessary points and properties of the triangle
variables {A B C : Type}

-- Define lengths of sides
variables (AB AC BC : ℝ)

-- Given condition: BC is half the sum of other two sides
axiom BC_condition : BC = (AB + AC) / 2

-- Define special points we are interested in
variables (D I : Type) -- D - intersection of angle bisector with the circumcircle, I - incenter

-- Assume A, D, I are on the same plane with the circumcenter and midpoints of sides
axiom cyclic_points : ∀ (A B C : Type) (AB AC BC : ℝ) (D I : Type),
  BC = (AB + AC) / 2 →
  ∃ (circumcircle : Type), 
    (A ∈ circumcircle ∧ -- vertex A
     ∃ (M_N M_P : Type), -- midpoints of sides AB and AC
      (M_N ∈ circumcircle ∧ M_P ∈ circumcircle) ∧ 
     (I ∈ circumcircle) ∧ -- incenter
     ∀ (circumcenter : Type), circumcenter ∈ circumcircle) -- circumcenter

-- Goal: Prove all significant points lie on the same circle
theorem significant_points_cyclic (A B C : Type) (AB AC BC : ℝ) (D I : Type) :
  BC = (AB + AC) / 2 →
  ∃ (circumcircle : Type), 
    (A ∈ circumcircle ∧ -- vertex A
     ∃ (M_N M_P : Type), -- midpoints of sides AB and AC
      (M_N ∈ circumcircle ∧ M_P ∈ circumcircle) ∧ 
     (I ∈ circumcircle) ∧ -- incenter
     ∀ (circumcenter : Type), circumcenter ∈ circumcircle) := 
sorry

end significant_points_cyclic_l325_325908


namespace hyperbola_equation_l325_325874

theorem hyperbola_equation (P : ℝ × ℝ) (f : ℝ × ℝ → Prop) (a b : ℝ) 
  (hP : P = (0, -2)) 
  (hf : f = λ Q, Q = (0, -4)) 
  (ha : a^2 = 4) 
  (hb : b^2 = 12) : 
  (∀ (x y : ℝ), (P = (0, -2)) → ((x^2 = -16 * y) → (hf (0, -4))) → 
  (c = 4) → 16 = a^2 + b^2 → ∀ (q w : ℝ), ((q, w) = P → ((w^2 / a^2) - (q^2 / b^2) = 1))) → 
  ∀ x y, (y^2 / 4) - (x^2 / 12) = 1 := 
sorry

end hyperbola_equation_l325_325874


namespace log_m_eq_a_sub_log_p_l325_325970

theorem log_m_eq_a_sub_log_p {m p : ℝ} (a : ℝ) (hm : log 2 m = a - log 2 p) : m = 2^a / p :=
by
  sorry

end log_m_eq_a_sub_log_p_l325_325970


namespace circle_equation_l325_325740

theorem circle_equation (h k r : ℝ) :
  (h, k) = (2, 0) → r = 2 →
  (∃ c : ℝ × ℝ, c ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 - 4} ∧
    ∀ p : ℝ × ℝ, p ∈ {(0, 0), (2, 2)} → dist p c = r) →
  (x y : ℝ)
    → (x - h)^2 + y^2 = r^2 :=
by
  intros h_eq k_eq radius_eq exists_c x y
  rcases exists_c with ⟨c, ⟨c_line_eq, c_dist_eq⟩⟩
  subst h_eq
  subst k_eq
  subst radius_eq
  exact sorry

end circle_equation_l325_325740


namespace number_of_correct_statements_is_zero_l325_325734

theorem number_of_correct_statements_is_zero :
  (∀ x : ℝ, ¬(-x < 0 ∨ x = 0)) ∧
  (¬(degree (-3*a^2 * b + 5*a^2 * b^2 - 2*a*b - 3) = 3)) ∧
  (¬(coeff (-2 : ℝ) 0 = (-5 : ℝ))) ∧
  (∀ a : ℝ, ¬(|a| = -a ∧ a < 0)) →
  (number_of_true_statements 4 = 0) :=
by
  sorry

def degree (p : Polynomial) : ℕ := sorry 
def coeff (c : ℝ) (x : ℝ) : ℝ := c
def number_of_true_statements (n : ℕ) : ℕ := 0

end number_of_correct_statements_is_zero_l325_325734


namespace number_of_triangles_in_6x6_grid_l325_325143

def is_valid_point (n : ℕ) (i j : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n

def number_of_triangles (n : ℕ) : ℕ :=
  let points := (Finset.range (n + 1)).product (Finset.range (n + 1))
  let valid_points := points.filter (λ p, is_valid_point n p.1 p.2)
  let all_triangles := valid_points.powersetLen 3
  let collinear (a b c : (ℕ × ℕ)) : Prop :=
    (b.1 - a.1) * (c.2 - a.2) = (c.1 - a.1) * (b.2 - a.2)
  let non_collinear_triangles := all_triangles.filter (λ t, ¬ collinear t.nthLe! 0 t.nthLe! 1 t.nthLe! 2)
  non_collinear_triangles.card

theorem number_of_triangles_in_6x6_grid : number_of_triangles 6 = 6804 := 
sorry

end number_of_triangles_in_6x6_grid_l325_325143


namespace simplify_tan_expression_l325_325343

theorem simplify_tan_expression 
  (h30 : Real.tan (π / 6) = 1 / Real.sqrt 3)
  (h15 : Real.tan (π / 12) = 2 - Real.sqrt 3) :
  (1 + Real.tan (π / 6)) * (1 + Real.tan (π / 12)) = 2 :=
by
  -- State the tangent addition formula for the required angles
  have h_tan_add : Real.tan (π / 4) = (Real.tan (π / 6) + Real.tan (π / 12)) / (1 - Real.tan (π / 6) * Real.tan (π / 12)),
  {
    sorry,
  }
  -- The correct answer proof part is not provided in the brief
  sorry

end simplify_tan_expression_l325_325343


namespace repeating_decimal_fraction_sum_l325_325162

theorem repeating_decimal_fraction_sum : 
  let x := 36 / 99 in
  let a := 4 in
  let b := 11 in
  gcd a b = 1 ∧ (a : ℚ) / (b : ℚ) = x → a + b = 15 :=
by
  sorry

end repeating_decimal_fraction_sum_l325_325162


namespace problem_statement_l325_325682

def system_eq1 (x y : ℝ) := x^3 - 5 * x * y^2 = 21
def system_eq2 (y x : ℝ) := y^3 - 5 * x^2 * y = 28

theorem problem_statement
(x1 y1 x2 y2 x3 y3 : ℝ)
(h1 : system_eq1 x1 y1)
(h2 : system_eq2 y1 x1)
(h3 : system_eq1 x2 y2)
(h4 : system_eq2 y2 x2)
(h5 : system_eq1 x3 y3)
(h6 : system_eq2 y3 x3)
(h_distinct : (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :
  (11 - x1 / y1) * (11 - x2 / y2) * (11 - x3 / y3) = 1729 :=
sorry

end problem_statement_l325_325682


namespace area_ratio_of_triangle_to_trapezoid_l325_325654

theorem area_ratio_of_triangle_to_trapezoid (A B C D E : Type) [HasArea A] [HasArea B] [HasArea C] [HasArea D] [HasArea E]
  (trapezoid : Trapezoid A B C D)
  (AB CD EAB EDC : ℝ)
  (hAB : AB = 8)
  (hCD : CD = 17)
  (hE : legs_extended_to_meet_at_point E)
  (similar : similar_triangles EAB EDC):
  (area_ratio : (area EAB) / (area (trapezoid ABCD)) = (64 / 225)) := sorry

end area_ratio_of_triangle_to_trapezoid_l325_325654


namespace sin_cos_sum_l325_325176

theorem sin_cos_sum (x y r : ℝ) (h : r = Real.sqrt (x^2 + y^2)) (ha : (x = 5) ∧ (y = -12)) :
  (y / r) + (x / r) = -7 / 13 :=
by
  sorry

end sin_cos_sum_l325_325176


namespace smallest_non_real_root_product_l325_325505

noncomputable def Q : ℤ → ℤ := λ x => x^4 - 2 * x^3 - 3 * x^2 + 4 * x + 5

theorem smallest_non_real_root_product :
  let r1 r2 : ℝ := Q.roots.filter (λ r => r ∈ [-1, 1] ∨ r ∈ [2, 3])
  Q(1) = 5 ∧
  (∏ x in Q.roots, x) = 5 ∧
  (∏ x in Q.roots.filter (λ r => ¬(r ∈ [-1, 1] ∨ r ∈ [2, 3])), x) = 4 ∧
  (d + sum([1, -2, -3, 4])) = 5 ∧
  (sum Q.real_roots) = r1 + r2
  → (∏ x in Q.roots.filter (λ r => ¬(r ∈ [-1, 1] ∨ r ∈ [2, 3])), x) < 5 :=
by
  sorry

end smallest_non_real_root_product_l325_325505


namespace function_symmetric_about_line_period_pi_l325_325112

theorem function_symmetric_about_line_period_pi :
  ∀ (f : ℝ → ℝ) (ω : ℝ),
    (∀ x, f x = sin (ω * x + π / 3)) →
    (0 < ω) →
    (∀ x, f (x + π) = f x) →
    (∀ x, f (π / 6 - x) = f (π / 6 + x)) := 
begin
  intros f ω h1 h2 h3,
  -- The proof will be placed here
  sorry
end

end function_symmetric_about_line_period_pi_l325_325112


namespace regions_divided_by_radii_circles_l325_325238

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l325_325238


namespace circle_x_intersect_at_6_l325_325454

def circle_center (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def distance (A B : ℝ × ℝ) : ℝ :=
  ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2).sqrt

def circle_radius (A B : ℝ × ℝ) : ℝ :=
  distance A B / 2

def circle_equation (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { P | (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2 = radius ^ 2 }

def x_axis_intersect (center : ℝ × ℝ) (radius : ℝ) : set ℝ :=
  { x | (x - center.1) ^ 2 = radius ^ 2 - (center.2 - 0) ^ 2 }

theorem circle_x_intersect_at_6 :
  let A := (2, 2)
      B := (10, 8)
      center := circle_center A B
      radius := circle_radius A B in
  x_axis_intersect center radius = {6} :=
by
  sorry

end circle_x_intersect_at_6_l325_325454


namespace circle_divided_into_regions_l325_325254

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l325_325254


namespace area_quad_ABDF_l325_325502

-- Definitions
def side_length : ℝ := 40
def AB := (1 / 4) * side_length
def AF := (3 / 4) * side_length

-- Main theorem
theorem area_quad_ABDF : 
  let ACDE_area := side_length^2 in -- Area of the square
  let BCD_area := (1 / 2) * (side_length - AB) * side_length in -- Area of triangle BCD
  let EFD_area := (1 / 2) * (side_length - AF) * side_length in -- Area of triangle EFD
  ACDE_area - BCD_area - EFD_area = 800 := -- Calculate area of quadrilateral ABDF
by {
  let ACDE_area := side_length^2,
  let BCD_area := (1 / 2) * (side_length - AB) * side_length,
  let EFD_area := (1 / 2) * (side_length - AF) * side_length,
  sorry
}

end area_quad_ABDF_l325_325502


namespace num_monomials_degree_7_l325_325744

theorem num_monomials_degree_7 : 
  ∃ (count : Nat), 
    (∀ (a b c : ℕ), a + b + c = 7 → (1 : ℕ) = 1) ∧ 
    count = 15 := 
sorry

end num_monomials_degree_7_l325_325744


namespace max_area_of_quadrilateral_l325_325819

theorem max_area_of_quadrilateral (a b c d : ℝ) (α γ : ℝ) (t : ℝ) : 
  (∀ (t : ℝ), 2 * t = a * b * Real.sin α + c * d * Real.sin γ) →
  α + γ = Real.pi →
  (∀ (s : ℝ), 2 * s ≤ a * b * Real.sin α + c * d * Real.sin γ) :=
begin
  sorry
end

end max_area_of_quadrilateral_l325_325819


namespace max_sum_cross_l325_325863

open Nat

def cross_like_structure (a b c d e : ℤ) :=
  (a = d) ∧ (b = c) ∧ (a + b + e = b + d + e) ∧ (a + c + e = a + b + e)

theorem max_sum_cross (a b c d e : ℤ) 
  (h₁: a ≠ b) (h₂: a ≠ c) (h₃: a ≠ d) (h₄: a ≠ e) (h₅: b ≠ c) 
  (h₆: b ≠ d) (h₇: b ≠ e) (h₈: c ≠ d) (h₉: c ≠ e) (h₁₀: d ≠ e) 
  (h₁₁: cross_like_structure a b c d e) : 
  max_sum_row_or_column a b c d e := 36 :=
by 
  sorry

end max_sum_cross_l325_325863


namespace lock_combinations_correct_l325_325282

theorem lock_combinations_correct :
  let digits : List ℕ := [1, 2, 3, 4, 5, 6] in
  let odd_digits : List ℕ := [1, 3, 5] in
  let even_digits : List ℕ := [2, 4, 6] in
  ∀ (comb : List ℕ), comb.length = 6 →
    (∀ i, i < 5 → (comb.nth i).is_some → 
          ((comb.nth i).iget ∈ even_digits ↔ (comb.nth (i + 1)).iget ∈ odd_digits) ∧
          ((comb.nth i).iget ∈ odd_digits ↔ (comb.nth (i + 1)).iget ∈ even_digits)) →
    (list.filter (λ (c : List ℕ), ∀ i, i < 5 → 
                    (c.nth i).is_some → 
                    ((c.nth i).iget ∈ even_digits ↔ (c.nth (i + 1)).iget ∈ odd_digits) ∧
                    ((c.nth i).iget ∈ odd_digits ↔ (c.nth (i + 1)).iget ∈ even_digits))
              (list.replicate 729 <| list.replicate 6 0)).length + 
    (list.filter (λ (c : List ℕ), ∀ i, i < 5 →
                    (c.nth i).is_some → 
                    ((c.nth i).iget ∈ odd_digits ↔ (c.nth (i + 1)).iget ∈ even_digits) ∧
                    ((c.nth i).iget ∈ even_digits ↔ (c.nth (i + 1)).iget ∈ odd_digits))
              (list.replicate 729 <| list.replicate 6 0)).length = 1458 := 
by
  sorry

end lock_combinations_correct_l325_325282


namespace simplify_product_l325_325362

theorem simplify_product (x t : ℕ) : (x^2 * t^3) * (x^3 * t^4) = (x^5) * (t^7) := 
by 
  sorry

end simplify_product_l325_325362


namespace winning_votes_casted_is_7241_l325_325188

-- Define the agreed percentages and the known number of votes difference.
def total_votes : Type := ℝ -- Assume total_votes as a real number for generalization

def winning_percentage : ℝ := 0.48
def runner_up_percentage : ℝ := 0.34
def combined_remaining_percentage : ℝ := 0.15
def spoiled_percentage : ℝ := 0.03
def vote_difference : ℝ := 2112

-- Given conditions set up within the Lean framework:
def valid_votes (V : total_votes) := V * (1 - spoiled_percentage)

-- Main statement to be proven:
theorem winning_votes_casted_is_7241 (V : total_votes) (H : valid_votes V = V - V * spoiled_percentage) :
  (vote_difference = (winning_percentage - runner_up_percentage) * V) →
  (V = 15085.71) →
  (winning_percentage * V = 7241) :=
begin
  intro H0,
  intro H1,
  rw [H0, H1],
  sorry
end

end winning_votes_casted_is_7241_l325_325188


namespace binomial_identity_l325_325574

theorem binomial_identity :
  (nat.choose 18 8 = 31824) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 172822 :=
by
  intros h1 h2 h3
  have h4: nat.choose 19 9 = nat.choose 18 8 + nat.choose 18 9 := by sorry
  have h5: nat.choose 19 9 = 31824 + 48620 := by sorry
  have h6: nat.choose 19 10 = nat.choose 18 9 + nat.choose 18 10 := by sorry
  have h7: nat.choose 19 10 = 48620 + 43758 := by sorry
  show nat.choose 20 10 = nat.choose 19 9 + nat.choose 19 10 from sorry
  have h8: nat.choose 20 10 = 80444 + 92378 := by sorry
  exact sorry

end binomial_identity_l325_325574


namespace no_pos_integers_such_that_product_is_power_of_11_l325_325516

theorem no_pos_integers_such_that_product_is_power_of_11 :
  ¬ ∃ (n : Fin 2022 → ℕ), 
        (∀ i, 0 < n i) ∧ 
        let term i := n i ^ 2020 + n ((i + 1) % 2022) ^ 2019 in
        (∏ i, term i) = 11^k for some k : ℕ :=
by
  sorry

end no_pos_integers_such_that_product_is_power_of_11_l325_325516


namespace triangle_count_l325_325135

theorem triangle_count (i j : ℕ) (h₁ : 1 ≤ i ∧ i ≤ 6) (h₂ : 1 ≤ j ∧ j ≤ 6): 
  let points := { (x, y) | 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 } in
  fintype.card { t : finset (ℕ × ℕ) // t.card = 3 ∧ ∃ a b c : ℕ × ℕ, a ∉ t ∧ b ∉ t ∧ c ∉ t ∧ abs ((b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)) ≠ 0 } = 6800 := 
by
  sorry

end triangle_count_l325_325135


namespace number_of_random_events_l325_325019

-- Define each event as a boolean indicating whether it is random
def event1 : Prop := true  -- Throwing dice is random
def event2 : Prop := true  -- Winning the lottery is random
def event3 : Prop := true  -- Random selection from set is random
def event4 : Prop := false -- Boiling temperature of water is not random

-- Define the function to count the number of true (random) events
def count_random_events (events : List Prop) : Nat :=
  events.foldr (fun ev acc => if ev then acc + 1 else acc) 0

-- List of events
def events : List Prop := [event1, event2, event3, event4]

-- The theorem to be proven
theorem number_of_random_events : count_random_events events = 3 := by
  sorry -- Proof is to be provided separately

end number_of_random_events_l325_325019


namespace repeating_decimal_fraction_sum_l325_325164

theorem repeating_decimal_fraction_sum : 
  let x := 36 / 99 in
  let a := 4 in
  let b := 11 in
  gcd a b = 1 ∧ (a : ℚ) / (b : ℚ) = x → a + b = 15 :=
by
  sorry

end repeating_decimal_fraction_sum_l325_325164


namespace median_fuel_cost_per_100_miles_l325_325749

/-- Definition of fuel types and their corresponding prices per unit
    (gallon, kWh, kg) and required amounts per 100 miles. -/
structure FuelType where
  name : String
  price : ℝ -- price per gallon or kWh or kg
  unit : String -- "gallon", "kWh", "kg"
  amount_per_100_miles : ℝ -- amount needed per 100 miles (for kWh and kg only)

def fuels : List FuelType :=
  [ { name := "Premium Gas", price := 3.59, unit := "gallon", amount_per_100_miles := 0 },
    { name := "Regular Gas", price := 2.99, unit := "gallon", amount_per_100_miles := 0 },
    { name := "Diesel", price := 3.29, unit := "gallon", amount_per_100_miles := 0 },
    { name := "Ethanol", price := 2.19, unit := "gallon", amount_per_100_miles := 0 },
    { name := "Biodiesel", price := 3.69, unit := "gallon", amount_per_100_miles := 0 },
    { name := "Electricity", price := 0.12, unit := "kWh", amount_per_100_miles := 34 },
    { name := "Hydrogen", price := 4.50, unit := "kg", amount_per_100_miles := 0.028 } ]

variable (fuel_efficiencies : List ℝ) -- efficiency for each fuel type, i.e., miles per gallon

/-- Function to calculate the cost per 100 miles given fuel efficiencies
    and fuels list. For electricity and hydrogen, the direct amount is used. -/
noncomputable def cost_per_100_miles (fuel_efficiencies : List ℝ) : List ℝ :=
  (fuels.zip fuel_efficiencies).map (λ (fuel, efficiency) =>
    if efficiency ≠ 0 then (100 / efficiency) * fuel.price
    else fuel.amount_per_100_miles * fuel.price)

/-- Function to find the median of a list of real numbers. -/
noncomputable def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· < ·)
  if sorted.length % 2 = 1 then
    sorted.get (sorted.length / 2)
  else
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2

/-- Main statement to prove: Given the efficiencies, prices, and conditions,
    compute and prove the median fuel cost per 100 miles. -/
theorem median_fuel_cost_per_100_miles (fuel_efficiencies : List ℝ) :
  median (cost_per_100_miles fuel_efficiencies) = sorry :=
sorry

end median_fuel_cost_per_100_miles_l325_325749


namespace circle_division_l325_325228

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l325_325228


namespace sum_of_solutions_l325_325777

theorem sum_of_solutions :
  (∑ x in {x : ℝ | (x < 5 ∧ x^2 - 8*x + 21 = -(x - 5) + 4) ∨ (x ≥ 5 ∧ x^2 - 8*x + 21 = x - 1)}, x) = 18 :=
by
  sorry

end sum_of_solutions_l325_325777


namespace problem_solution_l325_325067

def otimes (x y z : ℝ) (h : y ≠ z) : ℝ := x / (y - z)

theorem problem_solution :
  otimes (otimes 1 3 2 (by linarith))
         (otimes 2 1 3 (by linarith))
         (otimes 3 2 1 (by linarith))
         (by linarith) = -1/4 := 
  sorry

end problem_solution_l325_325067


namespace no_balanced_set_of_2013_elements_l325_325665

open Function

-- Define what it means for a set to be balanced
def is_balanced (A : Finset ℕ) : Prop :=
  let multiples_of_3 := (A.powerset.filter (λ t, (∑ x in t, x) % 3 = 0)).card
  let not_multiples_of_3 := (A.powerset.filter (λ t, (∑ x in t, x) % 3 ≠ 0)).card
  multiples_of_3 = not_multiples_of_3

-- Prove that no set of 2013 elements can be balanced
theorem no_balanced_set_of_2013_elements :
  ¬ ∃ (A : Finset ℕ), A.card = 2013 ∧ is_balanced A :=
by
  sorry

end no_balanced_set_of_2013_elements_l325_325665


namespace problem_statement_l325_325089

open Real -- Assuming arithmetic sequence with real numbers

-- Definitions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) * a 0 + n * (n + 1) / 2 * d

def bounded_sum_sequence (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n ≤ S 10

-- Theorem statement
theorem problem_statement (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) 
  (h_arith_seq : arithmetic_sequence a d)
  (h_non_zero_d : d ≠ 0)
  (h_sum_terms : sum_of_first_n_terms a S)
  (h_bounded_sum : bounded_sum_sequence S) :
  S 19 ≥ 0 :=
sorry -- Proof not required

end problem_statement_l325_325089


namespace problem_statement_l325_325923

theorem problem_statement (a1 a2 b1 b2 b3 : ℝ)
  (h1 : -9, a1, a2, -1 form an arithmetic sequence)
  (h2 : -9, b1, b2, b3, -1 form a geometric sequence) :
  b2 * (a2 - a1) = -8 := 
sorry

end problem_statement_l325_325923


namespace count_distinct_reals_a_with_integer_roots_l325_325890

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l325_325890


namespace binom_20_10_l325_325579

-- Given conditions
def binom_18_8 : ℕ := 31824
def binom_18_9 : ℕ := 48620
def binom_18_10 : ℕ := 43758

theorem binom_20_10 : nat.choose 20 10 = 172822 := by
  have h1 : nat.choose 19 9 = binom_18_8 + binom_18_9 := rfl
  have h2 : nat.choose 19 10 = binom_18_9 + binom_18_10 := rfl
  have h3 : nat.choose 20 10 = nat.choose 19 9 + nat.choose 19 10 := rfl
  rw [h1, h2, h3]
  exact rfl

end binom_20_10_l325_325579


namespace range_of_a_l325_325974

theorem range_of_a (x1 x2 : ℝ) (a : ℝ) :
  (a > 0) ∧ (a ≠ 1) ∧
  (∀ x1 x2, x1 < x2 ∧ x2 ≤ a / 2 
    → f a x1 - f a x2 > 0) 
  → ∃ a, a ∈ Ioo 1 (2 * real.sqrt 3) :=
by
  sorry

def f (a x : ℝ) : ℝ := real.log (x ^ 2 - a * x + 3)

end range_of_a_l325_325974


namespace monomial_forms_l325_325601

theorem monomial_forms : 
  (∃ n : ℕ, 3 ≤ n ∧ n ≤ 8 ∧ 
  ((-1 : ℤ)^n * a^(n-2 : ℤ) * b^(9-n : ℤ) = -a*b^6 ∨ 
  (-1 : ℤ)^n * a^(n-2 : ℤ) * b^(9-n : ℤ) = a^2*b^5 ∨ 
  (-1 : ℤ)^n * a^(n-2 : ℤ) * b^(9-n : ℤ) = -a^3*b^4 ∨ 
  (-1 : ℤ)^n * a^(n-2 : ℤ) * b^(9-n : ℤ) = a^4*b^3 ∨ 
  (-1 : ℤ)^n * a^(n-2 : ℤ) * b^(9-n : ℤ) = -a^5*b^2 ∨ 
  (-1 : ℤ)^n * a^(n-2 : ℤ) * b^(9-n : ℤ) = a^6*b)) :=
sorry

end monomial_forms_l325_325601


namespace count_real_numbers_a_with_integer_roots_l325_325885

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l325_325885


namespace residents_watch_all_four_residents_watch_all_four_correct_l325_325998

def Clermontville := 800
def watch_IS := 0.3 * Clermontville
def watch_LLL := 0.35 * Clermontville
def watch_ME := 0.45 * Clermontville
def watch_MM := 0.25 * Clermontville
def watch_exactly_two := 0.22 * Clermontville
def watch_exactly_three := 0.08 * Clermontville

theorem residents_watch_all_four :
  watch_IS + watch_LLL + watch_ME + watch_MM - watch_exactly_two + watch_exactly_three - X = Clermontville :=
by
  sorry

def number_of_residents_watching_all_four : ℕ :=
  168

-- Asserting the theorem now
theorem residents_watch_all_four_correct :
  residents_watch_all_four ⟨ 168 ⟩ := 
by
  sorry

end residents_watch_all_four_residents_watch_all_four_correct_l325_325998


namespace compute_sum_of_i_powers_l325_325491

theorem compute_sum_of_i_powers :
  let i : ℂ := complex.I in
  let s1 := (finset.range 2013).sum (λ n, i ^ n) in
  let s2 := (finset.range 5).sum (λ n, i ^ (n + 4)) in
  s1 - 2 * s2 = -2 :=
by
  -- Complex summation powers should be manually expanded for the theorem prover
  sorry

end compute_sum_of_i_powers_l325_325491


namespace circle_region_count_l325_325235

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l325_325235


namespace lower_limit_of_range_l325_325714

theorem lower_limit_of_range (A : Set ℕ) (range_A : ℕ) (h1 : ∀ n ∈ A, Prime n∧ n ≤ 36) (h2 : range_A = 14)
  (h3 : ∃ x, x ∈ A ∧ ¬(∃ y, y ∈ A ∧ y > x)) (h4 : ∃ x, x ∈ A ∧ x = 31): 
  ∃ m, m ∈ A ∧ m = 17 := 
sorry

end lower_limit_of_range_l325_325714


namespace total_yield_l325_325663

noncomputable def johnson_hectare_yield_2months : ℕ := 80
noncomputable def neighbor_hectare_yield_multiplier : ℕ := 2
noncomputable def neighbor_hectares : ℕ := 2
noncomputable def months : ℕ := 6

theorem total_yield (jh2 : ℕ := johnson_hectare_yield_2months) 
                    (nhm : ℕ := neighbor_hectare_yield_multiplier) 
                    (nh : ℕ := neighbor_hectares) 
                    (m : ℕ := months): 
                    3 * jh2 + 3 * nh * jh2 * nhm = 1200 :=
by
  sorry

end total_yield_l325_325663


namespace num_distinct_products_of_elements_S_40000_l325_325296

def is_divisor (n d : ℕ) : Prop := d ∣ n

def S (n : ℕ) : set ℕ := {d | is_divisor n d ∧ d > 0}

noncomputable def num_distinct_products_of_elements_S (n : ℕ) (S : set ℕ) : ℕ :=
  (S.to_finset.product S.to_finset).filter (λ p, p.1 ≠ p.2).card

theorem num_distinct_products_of_elements_S_40000 :
  num_distinct_products_of_elements_S 40000 (S 40000) = 73 :=
sorry

end num_distinct_products_of_elements_S_40000_l325_325296


namespace jellybean_problem_l325_325788

theorem jellybean_problem:
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  black + green + orange = 27 →
  green - orange = 1 :=
by
  intros black green orange h_black h_green h_total
  sorry

end jellybean_problem_l325_325788


namespace proof_ratio_area_of_quadrilateral_PNZM_triangle_QZR_l325_325194

open EuclideanGeometry

def ratio_area_of_quadrilateral_PNZM_triangle_QZR (P Q R M N Z : Point) : Prop :=
  let angle_Q := 90                                             -- ∠Q = 90 degrees
  let PQ := 15                                                  -- PQ = 15
  let QR := 20                                                  -- QR = 20
  let PR := Real.sqrt (PQ ^ 2 + QR ^ 2)                         -- PR calculated via Pythagoras
  let M := midpoint P Q                                         -- M is midpoint of PQ
  let N := midpoint P R                                         -- N is midpoint of PR
  let Z := centroid_of_triangle P Q R                           -- Z is centroid
  let quadrilateral_PNZM_area := polygon_area P N Z M           -- Area of quadrilateral PNZM
  let triangle_QZR_area := triangle_area Q Z R                  -- Area of triangle QZR
  quadrilateral_PNZM_area / triangle_QZR_area = 1               -- Ratio of the areas is 1

-- Proof is omitted
theorem proof_ratio_area_of_quadrilateral_PNZM_triangle_QZR (P Q R M N Z : Point) 
  (hQ : angle P Q R = 90) 
  (hPQ : dist P Q = 15) 
  (hQR : dist Q R = 20) 
  (hM : M = midpoint P Q) 
  (hN : N = midpoint P R) 
  (hZ : Z = centroid P Q R) : 
  ratio_area_of_quadrilateral_PNZM_triangle_QZR P Q R M N Z :=
sorry

end proof_ratio_area_of_quadrilateral_PNZM_triangle_QZR_l325_325194


namespace period_and_monotonic_decreasing_intervals_range_of_fx_on_interval_l325_325110

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

-- Statement for the first question
theorem period_and_monotonic_decreasing_intervals (k : ℤ) :
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x, k * π + π / 6 ≤ x → x ≤ k * π + 2 * π / 3 → (f x' = f x ∧ x' > x → x' = x)) :=
sorry

-- Statement for the second question
theorem range_of_fx_on_interval :
  ∀ x ∈ Icc (-π / 6) (π / 4), f x ∈ Icc (-1) 2 :=
sorry

end period_and_monotonic_decreasing_intervals_range_of_fx_on_interval_l325_325110


namespace regions_formed_l325_325221

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l325_325221


namespace sum_of_angles_nine_pointed_star_l325_325698

open Real

-- Definitions
def points_on_circle (n : ℕ) (circle : Set Point) : Prop :=
  ∀ (i j : ℕ), i ≠ j → dist (circle_point i) (circle_point j) = (360 / n) * abs (i - j)

def nine_pointed_star (circle : Set Point) : Prop :=
  points_on_circle 9 circle ∧ is_9_pointed_star circle

-- Theorem statement
theorem sum_of_angles_nine_pointed_star {circle : Set Point} (h : nine_pointed_star circle) :
  ∑ i in finset.range 9, angle (tip i) = 720 :=
sorry

end sum_of_angles_nine_pointed_star_l325_325698


namespace convex_quadratic_solution_l325_325522

theorem convex_quadratic_solution (f : ℝ → ℝ) :
  (∀ x y z : ℝ, x < y → y < z → 
    f(y) - ((z - y) / (z - x) * f(x) + (y - x) / (z - x) * f(z)) 
    ≤ f((x + z) / 2) - (f(x) + f(z)) / 2) →
  ∃ a b c : ℝ, a ≤ 0 ∧ (∀ x : ℝ, f(x) = a * x^2 + b * x + c) :=
sorry

end convex_quadratic_solution_l325_325522


namespace monomials_non_zero_coefficients_l325_325995

theorem monomials_non_zero_coefficients : 
  let expr := (λ (x y z : ℕ), (x + y + z) ^ 2036 + (x - y - z) ^ 2036) in
  (∑ n in (finset.range 2037).filter (λ k, k % 2 = 0), n + 1) = 1038361 :=
by 
  sorry

end monomials_non_zero_coefficients_l325_325995


namespace geometric_sequence_sum_of_first_four_terms_l325_325559

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def arithmetic_sequence (b c d : ℝ) : Prop :=
2 * c = b + d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
finset.sum (finset.range n) a

theorem geometric_sequence_sum_of_first_four_terms
    (a : ℕ → ℝ)
    (q : ℝ)
    (h_geo_seq : geometric_sequence a q)
    (h_pos_terms : ∀ n, a n > 0)
    (h_arith_seq : arithmetic_sequence (4 * a 3) (2 * a 4) (a 5))
    (h_a1 : a 1 = 1) :
  sum_first_n_terms a 4 = 15 :=
sorry

end geometric_sequence_sum_of_first_four_terms_l325_325559


namespace eq_2x_squared_plus_1_of_eq_exponential_l325_325616

theorem eq_2x_squared_plus_1_of_eq_exponential
  (x : ℝ)
  (h : 2^(3 * x) + 18 = 26 * 2^(x + 1)) :
  (2 * x) ^ 2 + 1 = 5 := 
sorry

end eq_2x_squared_plus_1_of_eq_exponential_l325_325616


namespace tennis_championship_l325_325815

theorem tennis_championship (n : ℕ) (initial_players : ℕ) (external_judge : Prop) : 
  (∀ participants judges judges_used games_played remaining_players : ℕ, 
    (initial_players ≥ 2) → 
    (games_played + 1 = initial_players - remaining_players) →
    (games_played ≤ initial_players - 1) → 
    (judges ≤ remaining_players) → 
    (judges_used = games_played) → 
    (remaining_players ≥ 1) →
    (n = games_played + judges + remaining_players) →
    (external_judge) →
    ∃ next_judge : ℕ, next_judge ≤ remaining_players) := 
begin
  sorry
end

end tennis_championship_l325_325815


namespace max_value_is_9_l325_325994

noncomputable def max_value_s_t_u_v (p q r s t u v : ℕ) : ℕ :=
  s + t + u + v

theorem max_value_is_9 (p q r s t u v : ℕ) 
  (h1 : ∀ x ∈ {p, q, r, s, t, u, v}, x = 1 ∨ x = 2 ∨ x = 3)
  (h2 : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h3 : q ≠ s ∧ s ≠ t ∧ q ≠ t)
  (h4 : r ≠ u ∧ u ≠ v ∧ r ≠ v) :
  max_value_s_t_u_v p q r s t u v = 9 :=
sorry

end max_value_is_9_l325_325994


namespace ants_count_l325_325013

noncomputable def total_ants_found (abe beth charlie daisy edward freda georgia : ℕ) : ℕ :=
  abe + beth + charlie + daisy + edward + freda + georgia

theorem ants_count :
  ∃ (abe beth charlie daisy edward freda georgia : ℕ),
    abe = 7 ∧
    beth = 7 + Nat.round (1.3 * 7 : ℚ).to_nat ∧
    charlie = 3 * 7 ∧
    daisy = Nat.round (0.5 * 7 : ℚ).to_nat ∧
    edward = charlie + daisy ∧
    freda = Nat.round (0.8 * (abe + beth + charlie) : ℚ).to_nat ∧
    georgia = Nat.round ((freda : ℚ) / 3).to_nat ∧
    total_ants_found abe beth charlie daisy edward freda georgia = 120 := 
by 
  use 7, Nat.round (9.1), 21, 4, 25, 35, 12
  split; refl
  split; refl
  split; refl
  split; refl
  split; refl
  split; refl
  refl
  sorry

end ants_count_l325_325013


namespace exists_m_square_between_l325_325121

theorem exists_m_square_between (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : a * d = b * c) : 
  ∃ m : ℤ, a < m^2 ∧ m^2 < d := 
sorry

end exists_m_square_between_l325_325121


namespace max_value_of_S_l325_325101

theorem max_value_of_S :
  ∀ (a b : Fin 31 → ℕ),
  (∀ i, a i < a (i + 1 ∧ b i < b (i + 1))) ∧ (∀ i, a i ≤ 2015 ∧ b i ≤ 2015) ∧
  (Finset.univ.sum (λ i => a i) = Finset.univ.sum (λ i => b i))
  → ∑ i, |a i - b i| ≤ 30720 :=
sorry

end max_value_of_S_l325_325101


namespace circle_division_l325_325229

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l325_325229


namespace distinct_a_count_l325_325893

theorem distinct_a_count :
  ∃ (a_set : Set ℝ), (∀ x ∈ a_set, ∃ r s : ℤ, r + s = -x ∧ r * s = 9 * x) ∧ a_set.toFinset.card = 3 :=
by 
  sorry

end distinct_a_count_l325_325893


namespace third_derivative_y_l325_325054

noncomputable def y (x : ℝ) : ℝ := x * Real.cos (x^2)

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = (8 * x^4 - 6) * Real.sin (x^2) - 24 * x^2 * Real.cos (x^2) :=
by
  sorry

end third_derivative_y_l325_325054


namespace algebraic_expression_needed_l325_325752

theorem algebraic_expression_needed (n : ℕ) :
  let left_n := (n+1) * (n+2) * ... * (n+n),
      left_k := (k+1) * (k+2) * ... * (2k)
      left_k1 := (k+2) * (k+3) * ... * (2k) * (2k+1) * (2k+2)
  in 
  left_k1 = left_k * 2 * (2k + 1) :=
by
  sorry

end algebraic_expression_needed_l325_325752


namespace games_needed_in_single_elimination_l325_325816

theorem games_needed_in_single_elimination (teams : ℕ) (h : teams = 23) : 
  ∃ games : ℕ, games = teams - 1 ∧ games = 22 :=
by
  existsi (teams - 1)
  sorry

end games_needed_in_single_elimination_l325_325816


namespace perimeter_of_triangle_ABF2_l325_325931

theorem perimeter_of_triangle_ABF2 :
  let a := 4 in
  let major_axis := 2 * a in
  let F1 := (-a, 0) in
  let F2 := (a, 0) in
  ∀ (A B : ℝ × ℝ),
    (∀ x y, (x,y) ∈ (A,B) → x^2 / 16 + y^2 / 7 = 1) →
    (A.1, A.2) = F1 →
    (¬(A.1 = B.1) ∨ ¬(A.2 = B.2)) →
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = major_axis^2 →
  let AF1 := 0 in
  let BF1 := major_axis in
  let AF2 := major_axis in
  let BF2 := ∥B - F2∥ in
  4 * a = 16 :=
by
  sorry

end perimeter_of_triangle_ABF2_l325_325931


namespace num_int_values_not_satisfying_l325_325069

theorem num_int_values_not_satisfying:
  (∃ n : ℕ, n = 7 ∧ (∃ x : ℤ, 7 * x^2 + 25 * x + 24 ≤ 30)) :=
sorry

end num_int_values_not_satisfying_l325_325069


namespace total_pennies_l325_325702

theorem total_pennies (R G K : ℕ) (h1 : R = 180) (h2 : G = R / 2) (h3 : K = G / 3) : R + G + K = 300 := by
  sorry

end total_pennies_l325_325702


namespace square_root_3a_minus_4b_is_pm4_l325_325104

theorem square_root_3a_minus_4b_is_pm4
  (a b : ℝ)
  (h1 : sqrt (2*a + 1) = 3 ∨ sqrt (2*a + 1) = -3)
  (h2 : sqrt (5*a + 2*b - 2) = 4) :
  sqrt (3*a - 4*b) = 4 ∨ sqrt (3*a - 4*b) = -4 := 
by
  sorry

end square_root_3a_minus_4b_is_pm4_l325_325104


namespace find_multiple_of_t_l325_325177

theorem find_multiple_of_t (k t x y : ℝ) (h1 : x = 1 - k * t) (h2 : y = 2 * t - 2) :
  t = 0.5 → x = y → k = 4 :=
by
  intros ht hxy
  sorry

end find_multiple_of_t_l325_325177


namespace purely_imaginary_iff_l325_325629

theorem purely_imaginary_iff (a : ℝ) :
  (a^2 - a - 2 = 0 ∧ (|a - 1| - 1 ≠ 0)) ↔ a = -1 :=
by
  sorry

end purely_imaginary_iff_l325_325629


namespace train_crossing_signal_pole_l325_325786

theorem train_crossing_signal_pole (L_t : ℝ) (T_p : ℝ) (L_p : ℝ) 
  (h1 : L_t = 300) 
  (h2 : T_p = 38) 
  (h3 : L_p = 333.33) 
  : ∃ (T_s : ℝ), T_s = 18 :=
by {
  -- We need to prove that time to cross signal pole T_s = 18
  let distance := L_t + L_p,
  let speed := distance / T_p,
  have h4 : distance = 633.33, from sorry,
  have h5 : speed = 16.67, from sorry,
  let T_s := L_t / speed,
  have h6 : T_s = 18, from sorry,
  exact ⟨T_s, h6⟩
}

end train_crossing_signal_pole_l325_325786


namespace company_x_installation_charge_l325_325415

theorem company_x_installation_charge:
  let price_X := 575
  let surcharge_X := 0.04 * price_X
  let installation_charge_X := 82.50
  let total_cost_X := price_X + surcharge_X + installation_charge_X
  let price_Y := 530
  let surcharge_Y := 0.03 * price_Y
  let installation_charge_Y := 93.00
  let total_cost_Y := price_Y + surcharge_Y + installation_charge_Y
  let savings := 41.60
  total_cost_X - total_cost_Y = savings → installation_charge_X = 82.50 :=
by
  intros h
  sorry

end company_x_installation_charge_l325_325415


namespace sqrt_3a_4b_eq_pm4_l325_325106

variable (a b : ℝ)

theorem sqrt_3a_4b_eq_pm4
  (h1 : sqrt (2 * a + 1) = 3 ∨ sqrt (2 * a + 1) = -3)
  (h2 : sqrt (5 * a + 2 * b - 2) = 4) :
  sqrt (3 * a - 4 * b) = 4 ∨ sqrt (3 * a - 4 * b) = -4 :=
by
  sorry

end sqrt_3a_4b_eq_pm4_l325_325106


namespace number_of_triangles_in_6x6_grid_l325_325140

def is_valid_point (n : ℕ) (i j : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n

def number_of_triangles (n : ℕ) : ℕ :=
  let points := (Finset.range (n + 1)).product (Finset.range (n + 1))
  let valid_points := points.filter (λ p, is_valid_point n p.1 p.2)
  let all_triangles := valid_points.powersetLen 3
  let collinear (a b c : (ℕ × ℕ)) : Prop :=
    (b.1 - a.1) * (c.2 - a.2) = (c.1 - a.1) * (b.2 - a.2)
  let non_collinear_triangles := all_triangles.filter (λ t, ¬ collinear t.nthLe! 0 t.nthLe! 1 t.nthLe! 2)
  non_collinear_triangles.card

theorem number_of_triangles_in_6x6_grid : number_of_triangles 6 = 6804 := 
sorry

end number_of_triangles_in_6x6_grid_l325_325140


namespace find_colorable_n_l325_325495

noncomputable def colorable (n : ℕ) : Prop :=
∀ (lines : set (List (ℝ × ℝ → ℝ))), 
  set.finite lines ∧
  (∀ l1 l2 l3 : List (ℝ × ℝ → ℝ), l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → collinear l1 l2 l3 → false) →
  ∃ (coloring : set (List (ℝ × ℝ → ℝ) × (ℕ × ℕ))),
    (∀ r1 r2 : List (ℝ × ℝ → ℝ),
      adj r1 r2 → ∀ (a1 a2 : ℕ) (b1 b2 : ℕ), (a1 ≠ a2 ∨ b1 ≠ b2) ∧ coloring r1 = some (a1, b1) ∧ 
      coloring r2 = some (a2, b2)) ∧
    (∀ ai bi : ℕ, (ai = 1 ∨ ai = 2) ∧ (bi = 1 ∨ bi = 2 ∨ bi = 3) → ∃ r : List (ℝ × ℝ → ℝ), coloring r = some (ai, bi))

theorem find_colorable_n (n : ℕ) : colorable n ↔ n ≥ 5 := sorry

end find_colorable_n_l325_325495


namespace perfect_squares_divisible_by_12_l325_325959

theorem perfect_squares_divisible_by_12 :
  { n : ℕ | ∃ k : ℕ, n = k^2 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n % 12 = 0 }.card = 4 := 
sorry

end perfect_squares_divisible_by_12_l325_325959


namespace fraction_sum_l325_325154

theorem fraction_sum (a b : ℕ) (h1 : 0.36 = a / b) (h2: Nat.gcd a b = 1) : a + b = 15 := by
  sorry

end fraction_sum_l325_325154


namespace probability_of_point_in_sphere_l325_325466

open real

noncomputable def probability_inside_sphere : ℝ :=
let cube_volume := 4^3 in
let sphere_volume := (4 / 3) * π * (2^3) in
sphere_volume / cube_volume

theorem probability_of_point_in_sphere :
  probability_inside_sphere = π / 6 := by
  sorry

end probability_of_point_in_sphere_l325_325466


namespace find_y_for_orthogonality_l325_325875

namespace OrthogonalVectors

-- Define the vectors
def vector_v : ℝ^3 := ![2, -4, -3]
def vector_w (y : ℝ) : ℝ^3 := ![-3, y, 2]

-- Constant assumptions for orthogonality
def orthogonal (u v : ℝ^3) : Prop := u.dot_product v = 0

theorem find_y_for_orthogonality : orthogonal vector_v (vector_w (-3)) :=
by sorry

end OrthogonalVectors

end find_y_for_orthogonality_l325_325875


namespace binom_20_10_l325_325576

-- Given conditions
def binom_18_8 : ℕ := 31824
def binom_18_9 : ℕ := 48620
def binom_18_10 : ℕ := 43758

theorem binom_20_10 : nat.choose 20 10 = 172822 := by
  have h1 : nat.choose 19 9 = binom_18_8 + binom_18_9 := rfl
  have h2 : nat.choose 19 10 = binom_18_9 + binom_18_10 := rfl
  have h3 : nat.choose 20 10 = nat.choose 19 9 + nat.choose 19 10 := rfl
  rw [h1, h2, h3]
  exact rfl

end binom_20_10_l325_325576


namespace circle_regions_division_l325_325247

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l325_325247


namespace no_pos_integers_exist_l325_325507

theorem no_pos_integers_exist (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬ (3 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
sorry

end no_pos_integers_exist_l325_325507


namespace visitors_not_ill_l325_325478

theorem visitors_not_ill (total_visitors : ℕ) (ill_percentage : ℕ) (fall_ill : ℕ) : 
  total_visitors = 500 → 
  ill_percentage = 40 → 
  fall_ill = (ill_percentage * total_visitors) / 100 →
  total_visitors - fall_ill = 300 :=
by
  intros h1 h2 h3
  sorry

end visitors_not_ill_l325_325478


namespace regions_divided_by_radii_circles_l325_325239

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l325_325239


namespace find_AC_length_l325_325210

noncomputable def length_of_AC (B : ℝ) (AB BC AC area : ℝ) : Prop :=
  tan B = √3 ∧
  AB = 3 ∧
  area = (3 * √3) / 2 →
  B = π / 3 ∧
  BC = 2 →
  AC = √7

theorem find_AC_length : ∃ AC, AC = √7 :=
by
  use √7
  sorry

end find_AC_length_l325_325210


namespace circles_radii_divide_regions_l325_325268

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l325_325268


namespace seventh_number_fifth_row_l325_325732

theorem seventh_number_fifth_row : 
  ∀ (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ → ℕ), 
  (∀ i, 1 <= i ∧ i <= n  → b 1 i = 2 * i - 1) →
  (∀ j i, 2 <= j ∧ 1 <= i ∧ i <= n - (j-1)  → b j i = b (j-1) i + b (j-1) (i+1)) →
  (b : ℕ → ℕ → ℕ) →
  b 5 7 = 272 :=
by {
  sorry
}

end seventh_number_fifth_row_l325_325732


namespace equal_angles_l325_325278

noncomputable theory

variables {A B C D K L M N : EuclideanGeometry.Point ℝ}

-- Definitions for midpoints
def is_midpoint (M X Y : EuclideanGeometry.Point ℝ) : Prop := dist M X = dist M Y

-- Given conditions
variables (h1 : is_midpoint L A D) (h2 : is_midpoint M C D) 
variables (h3 : dist L K = dist L C) (h4 : dist M K = dist M A)
variables (N : EuclideanGeometry.Point ℝ) (h5 : dist N B = dist N K)

-- Prove the statement
theorem equal_angles (h1 h2 h3 h4 h5 : Prop) : ∠ N A K = ∠ N C K := 
by sorry

end equal_angles_l325_325278


namespace highest_profit_rate_l325_325800

-- Definitions for the problem conditions
def f (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 20 then 1
  else if 21 ≤ x ∧ x ≤ 60 then x / 10
  else 0

def total_funds_before_x (x : ℕ) : ℚ :=
  8100000 / 10000 + ∑ i in Finset.range (x - 1), f (i + 1)

def g (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 60 then f x / total_funds_before_x x else 0

-- Proof problem statement
theorem highest_profit_rate :
  g 10 = 1 / 90 ∧
  (∀ x, 1 ≤ x ∧ x ≤ 20 → g x = 1 / (x + 80)) ∧
  (∀ x, 21 ≤ x ∧ x ≤ 60 → g x = 2 * x / (x * x - x + 1600)) ∧
  (∃ m, 1 ≤ m ∧ m ≤ 60 ∧ g m = 2 / 79 ∧ (∀ x, 1 ≤ x ∧ x ≤ 60 → g x ≤ g m) ∧ m = 40) :=
by
  sorry

end highest_profit_rate_l325_325800


namespace find_g_at_9_l325_325292

noncomputable def f (x : ℝ) := x^3 + x + 1

noncomputable def g (x : ℝ) : ℝ :=
  let rts := Complex.isCubicRoots (λ x => x^3 + x + 1) in
  x - rts.1^2

theorem find_g_at_9 :
  let rts := Complex.isCubicRoots (λ x => x^3 + x + 1)
  g 0 = -1 →
  g (Complex.ofReal rts.1^2) = g (Complex.ofReal rts.2^2) = g (Complex.ofReal rts.3^2) = 0 →
  g 9 = 899 :=
by
  sorry

end find_g_at_9_l325_325292


namespace count_valid_integers_l325_325956

def satisfies_properties (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d = b + c ∧ b = a - 1

theorem count_valid_integers :
  (finset.filter satisfies_properties (finset.range 1101 2201)).card = 19 :=
sorry

end count_valid_integers_l325_325956


namespace reinforcement_calculation_l325_325770

theorem reinforcement_calculation
  (initial_men : ℕ := 2000)
  (initial_days : ℕ := 40)
  (days_until_reinforcement : ℕ := 20)
  (additional_days_post_reinforcement : ℕ := 10)
  (total_initial_provisions : ℕ := initial_men * initial_days)
  (remaining_provisions_post_20_days : ℕ := total_initial_provisions / 2)
  : ∃ (reinforcement_men : ℕ), reinforcement_men = 2000 :=
by
  have remaining_provisions := remaining_provisions_post_20_days
  have total_post_reinforcement := initial_men + ((remaining_provisions) / (additional_days_post_reinforcement))

  use (total_post_reinforcement - initial_men)
  sorry

end reinforcement_calculation_l325_325770


namespace expected_twos_three_dice_l325_325409

def expected_twos (n : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), k * (nat.choose n k) * (1/6)^k * (5/6)^(n - k)

theorem expected_twos_three_dice : expected_twos 3 = 1/2 :=
by
  sorry

end expected_twos_three_dice_l325_325409


namespace patrick_savings_l325_325319

theorem patrick_savings :
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  saved_money - lent_money = 25 := by
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  sorry

end patrick_savings_l325_325319


namespace log_problem_solution_l325_325602

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Definitions based on conditions
def a (x : ℝ) : ℝ := Real.log (x - 4) / Real.log (sqrt (2 * x - 8))
def b (x : ℝ) : ℝ := Real.log (2 * x - 8) / Real.log (sqrt (5 * x - 26))
def c (x : ℝ) : ℝ := Real.log (5 * x - 26) / Real.log ((x - 4)^2)

-- Main theorem statement
theorem log_problem_solution (x : ℝ) : 
  (a x = b x ∧ a x = c x - 1 ∨
   b x = c x ∧ c x = a x - 1 ∨
   c x = a x ∧ a x = b x - 1) →
  x = 6 := by
  sorry

end log_problem_solution_l325_325602


namespace equilateral_hexagon_implication_l325_325316

-- Define the structure of the hexagon and equilateral triangles on each side
structure Hexagon (A B C D E F : Type) :=
  (equilateral_A'AB : Equilateral A B A')
  (equilateral_B'BC : Equilateral B C B')
  (equilateral_C'CD : Equilateral C D C')
  (equilateral_D'DE : Equilateral D E D')
  (equilateral_E'EF : Equilateral E F E')
  (equilateral_F'FA : Equilateral F A F')

structure RegularTriangle (P Q R : Type) :=
  (equilateral : Equilateral P Q R)

-- Given a convex hexagon with vertices A, B, C, D, E, F and external equilateral triangles on each side
-- Given ΔA'B'C' is equilateral
-- Prove that ΔA''B''C'' formed by similar construction is also equilateral
theorem equilateral_hexagon_implication {A B C D E F A' B' C' A'' B'' C'' : Type}
  (hex : Hexagon A B C D E F)
  (t1 : RegularTriangle A' B' C' ⟨hex.equilateral_A'AB, hex.equilateral_B'BC, hex.equilateral_C'CD⟩) :
  RegularTriangle A'' B'' C'' ⟨hex.equilateral_A''AF, hex.equilateral_B''BC, hex.equilateral_C''CD⟩ :=
sorry

end equilateral_hexagon_implication_l325_325316


namespace regions_formed_l325_325214

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l325_325214


namespace form_square_from_trapezoid_l325_325038

noncomputable def trapezoid_area (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

theorem form_square_from_trapezoid (a b h : ℝ) (trapezoid_area_eq_five : trapezoid_area a b h = 5) :
  ∃ s : ℝ, s^2 = 5 :=
by
  use (Real.sqrt 5)
  sorry

end form_square_from_trapezoid_l325_325038


namespace total_initial_yield_l325_325656

variable (x y z : ℝ)

theorem total_initial_yield (h1 : 0.4 * x + 0.2 * y = 5) 
                           (h2 : 0.4 * y + 0.2 * z = 10) 
                           (h3 : 0.4 * z + 0.2 * x = 9) 
                           : x + y + z = 40 := 
sorry

end total_initial_yield_l325_325656


namespace quotient_of_f_div_g_l325_325533

-- Define the polynomial f(x) = x^5 + 5
def f (x : ℝ) : ℝ := x ^ 5 + 5

-- Define the divisor polynomial g(x) = x - 1
def g (x : ℝ) : ℝ := x - 1

-- Define the expected quotient polynomial q(x) = x^4 + x^3 + x^2 + x + 1
def q (x : ℝ) : ℝ := x ^ 4 + x ^ 3 + x ^ 2 + x + 1

-- State and prove the main theorem
theorem quotient_of_f_div_g (x : ℝ) :
  ∃ r : ℝ, f x = g x * (q x) + r :=
by
  sorry

end quotient_of_f_div_g_l325_325533


namespace remaining_grass_area_l325_325455

def diameter : ℝ := 20
def path_width : ℝ := 2

theorem remaining_grass_area : 
  let r := diameter / 2 in
  let circle_area := Real.pi * r ^ 2 in
  let path_length := diameter in
  let path_rect_area := path_width * path_length in
  let path_cap_area := 0.5 * Real.pi * (path_width / 2) ^ 2 in
  let total_path_area := 2 * (path_rect_area + path_cap_area) - (path_width ^ 2) in
  (circle_area - total_path_area) = 98 * Real.pi - 76 := 
  sorry

end remaining_grass_area_l325_325455


namespace no_real_r_l325_325849

def P (x : ℂ) : ℂ := x^3 + x^2 - x + 2

theorem no_real_r 
  (r : ℝ) : 
  (∀ (z : ℂ), (z ∉ set.univ → ℂ) → P z ≠ r) ↔ r ∈ (-∞, 49/27) ∪ (3, ∞) :=
sorry

end no_real_r_l325_325849


namespace sandro_children_l325_325327

variables (sons daughters children : ℕ)

-- Conditions
def has_six_times_daughters (sons daughters : ℕ) : Prop := daughters = 6 * sons
def has_three_sons (sons : ℕ) : Prop := sons = 3

-- Theorem to be proven
theorem sandro_children (h1 : has_six_times_daughters sons daughters) (h2 : has_three_sons sons) : children = 21 :=
by
  -- Definitions from the conditions
  unfold has_six_times_daughters has_three_sons at h1 h2

  -- Skip the proof
  sorry

end sandro_children_l325_325327


namespace gwen_average_speed_l325_325947

def average_speed (distance1 distance2 speed1 speed2 : ℕ) : ℕ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time

theorem gwen_average_speed :
  average_speed 40 40 15 30 = 20 :=
by
  sorry

end gwen_average_speed_l325_325947


namespace sign_change_impossible_l325_325639

noncomputable def decagon_diagonals : ℕ := 10 * (10 - 3) / 2 -- number of diagonals in a decagon

def initial_marks (pts : ℕ) : ℕ := pts -- initial marking of points (vertices and intersections) with +1

theorem sign_change_impossible :
  ∀ (pts : ℕ), (initial_marks pts = pts) ∧ (decagon_diagonals = 35) ∧
  (∀ side : ℕ, side = 10 → ∀ diag : ℕ, diag = 35 → 
  (∃ op_sequence : (finset ℕ → finset ℕ), 
   (∀ si di : (finset ℕ), op_sequence si di = si.sdiff di ∨ 
   op_sequence si di = di.sdiff si)) → 
   ∀ t : ℕ, t ≠ pts → false) :=
by
  intros pts h1 h2 h3 side hs diag hd op_sequence hop t ht
  sorry

end sign_change_impossible_l325_325639


namespace square_root_calc_l325_325102

theorem square_root_calc (x : ℤ) (hx : x^2 = 1764) : (x + 2) * (x - 2) = 1760 := by
  sorry

end square_root_calc_l325_325102


namespace problem_I_problem_II_l325_325783

-- Question I
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 :=
by
  sorry

-- Question II
theorem problem_II (a : ℝ) : (∀ x : ℝ, |x - a| + |2 * x - 1| ≥ 2) ↔ (a ≤ -3/2 ∨ a ≥ 5/2) :=
by
  sorry

end problem_I_problem_II_l325_325783


namespace circle_division_l325_325223

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l325_325223


namespace measure_angle_delta_l325_325528

def sum_cos_range_eq_one (start end : ℤ) : ℤ :=
  ∑ k in finset.range (end - start + 1), int.cos (start + k)

def sum_sin_range_eq_sin_25 (start end : ℤ) : ℤ :=
  ∑ k in finset.range (end - start + 1), int.sin (start + k)

theorem measure_angle_delta :
  let sum_cos := sum_cos_range_eq_one 2880 6480
  let sum_sin := sum_sin_range_eq_sin_25 2905 6505
  \delta = real.arccos (sum_sin^sum_cos) -> \delta = 65
:= by
  sorry

end measure_angle_delta_l325_325528


namespace trapezoid_perimeter_l325_325494

def EF : ℝ := 40 -- longer base
def GH : ℝ := 20 -- shorter base
def EG : ℝ := 30 -- segment EG
def FH : ℝ := 45 -- segment FH

theorem trapezoid_perimeter : 
  let FG := Real.sqrt (EF^2 - EG^2), HE := Real.sqrt (FH^2 - GH^2)
  in EF + FG + GH + HE = 60 + 10 * Real.sqrt 7 + 5 * Real.sqrt 65 := 
  by 
    let FG := Real.sqrt (EF^2 - EG^2)
    let HE := Real.sqrt (FH^2 - GH^2)
    have FG_eq : FG = 10 * Real.sqrt 7 := by sorry
    have HE_eq : HE = 5 * Real.sqrt 65 := by sorry
    have perim_eq : EF + FG + GH + HE = 60 + 10 * Real.sqrt 7 + 5 * Real.sqrt 65 := by sorry
    exact perim_eq

end trapezoid_perimeter_l325_325494


namespace triangle_inequality_l325_325290

theorem triangle_inequality (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  sqrt(a * (a + b - sqrt(a * b))) + sqrt(b * (a + c - sqrt(a * c))) + sqrt(c * (b + c - sqrt(b * c))) ≥ a + b + c := 
sorry

end triangle_inequality_l325_325290


namespace grant_earnings_proof_l325_325128

noncomputable def total_earnings (X Y Z W : ℕ): ℕ :=
  let first_month := X
  let second_month := 3 * X + Y
  let third_month := 2 * second_month - Z
  let average := (first_month + second_month + third_month) / 3
  let fourth_month := average + W
  first_month + second_month + third_month + fourth_month

theorem grant_earnings_proof : total_earnings 350 30 20 50 = 5810 := by
  sorry

end grant_earnings_proof_l325_325128


namespace simplify_tan_product_l325_325340

theorem simplify_tan_product : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  -- use the angle addition formula for tangent
  have tan_sum : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12) :=
    by rw [Real.tan_add, Real.tan_pi_div_four]
  -- using the given condition tan(45 degrees) = 1
  have tan_45 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  sorry

end simplify_tan_product_l325_325340


namespace series_sum_l325_325500

noncomputable def sum_series : Real :=
  ∑' n: ℕ, (4 * (n + 1) + 2) / (3 : ℝ)^(n + 1)

theorem series_sum : sum_series = 3 := by
  sorry

end series_sum_l325_325500


namespace percentage_reduction_is_20_l325_325002

def original_employees : ℝ := 243.75
def reduced_employees : ℝ := 195

theorem percentage_reduction_is_20 :
  (original_employees - reduced_employees) / original_employees * 100 = 20 := 
  sorry

end percentage_reduction_is_20_l325_325002


namespace sum_of_sequences_l325_325026

-- Definition of the problem conditions
def seq1 := [2, 12, 22, 32, 42]
def seq2 := [10, 20, 30, 40, 50]
def sum_seq1 := 2 + 12 + 22 + 32 + 42
def sum_seq2 := 10 + 20 + 30 + 40 + 50

-- Lean statement of the problem
theorem sum_of_sequences :
  sum_seq1 + sum_seq2 = 260 := by
  sorry

end sum_of_sequences_l325_325026


namespace equality_or_neg_equality_of_eq_l325_325546

theorem equality_or_neg_equality_of_eq
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^3 / a = b^2 + a^3 / b) : a = b ∨ a = -b := 
  by
  sorry

end equality_or_neg_equality_of_eq_l325_325546


namespace abc_sqrt_proof_l325_325585

noncomputable def find_abc_and_sqrt (a b c : ℤ) : Prop :=
  (a - 2) + (7 - 2 * a) = 0 ∧
  ∛(3 * b + 1) = -2 ∧
  c = Int.floor (Real.sqrt 39) ∧
  (∀ a b c, (a = 5 → b = -3 → c = 6 → Real.sqrt (5 * a + 2 * b - c) = Real.sqrt 13 ∨
  Real.sqrt (5 * a + 2 * b - c) = - Real.sqrt 13))

theorem abc_sqrt_proof :
  ∃ a b c, find_abc_and_sqrt a b c :=
begin
  -- Proof is not required
  sorry
end

end abc_sqrt_proof_l325_325585


namespace smallest_N_bench_sections_l325_325470

theorem smallest_N_bench_sections (N : ℕ) (h₁ : 8 * N = 12 * N):
  N = 3 :=
begin
  sorry
end

end smallest_N_bench_sections_l325_325470


namespace intersection1_intersection2_l325_325866

theorem intersection1 :
  ∃ x : ℚ, (5 * x - 20 = 190 - 3 * x) ∧ (x = 105 / 4) :=
by
  use 105 / 4
  split
  · norm_num
    linarith

theorem intersection2 :
  ∃ x : ℚ, (5 * x - 20 = 2 * x + 15) ∧ (x = 35 / 3) :=
by
  use 35 / 3
  split
  · norm_num
    linarith

end intersection1_intersection2_l325_325866


namespace simplify_tan_expression_l325_325358

theorem simplify_tan_expression :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 := 
by 
  -- Angle addition formula for tangent
  have h : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12), 
  from by rw [Real.tan_add]; exact Real.tan_pi_div_four,
  -- Given that tan 45° = 1
  have h1 : Real.tan (Real.pi / 4) = 1, from Real.tan_pi_div_four,
  -- Derive the known value
  rw [Real.tan_pi_div_four, h] at h1,
  -- Simplify using the derived value
  suffices : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 
             1 + Real.tan (Real.pi / 6) + Real.tan (Real.pi / 12) + Real.tan (Real.pi / 6) * Real.tan (Real.pi / 12), 
  from by rw this; simp [←h1],
  sorry

end simplify_tan_expression_l325_325358


namespace no_solution_in_positive_rationals_l325_325291

theorem no_solution_in_positive_rationals (n : ℕ) (hn : n > 0) (x y : ℚ) (hx : x > 0) (hy : y > 0) :
  x + y + (1 / x) + (1 / y) ≠ 3 * n :=
sorry

end no_solution_in_positive_rationals_l325_325291


namespace pq_solution_l325_325426

theorem pq_solution (A B C p q : ℝ) (h1 : A ≠ 0) :
  (A (2 * q - p^2) = B ∧ A * q^2 = C) ↔ (A * x^4 + B * x^2 + C = A * (x^2 + p * x + q) * (x^2 - p * x + q)) :=
sorry

end pq_solution_l325_325426


namespace range_of_f_area_of_ABC_l325_325114

variable {x A : ℝ}
variable {a b c : ℕ}

-- Define the given function f(x)
def f (x : ℝ) : ℝ := sqrt 3 * sin^2 x + sin x * cos x

-- The first statement to prove the range of f(x)
theorem range_of_f (h₁ : 0 ≤ x) (h₂ : x ≤ π / 3) : 0 ≤ f x ∧ f x ≤ sqrt 3 := sorry

-- The second statement regarding the area of ΔABC
theorem area_of_ABC (h₁ : f (A / 2) = sqrt 3 / 2) (h₂ : a = 4) (h₃ : b + c = 5) : 
  let bc := 3 in 
  (1 / 2 * bc * sin A) = 3 * sqrt 3 / 4 := sorry

end range_of_f_area_of_ABC_l325_325114


namespace exists_line_with_n_beetles_l325_325987

theorem exists_line_with_n_beetles (n : ℕ) 
  (initial_positions : fin (n + 1) × fin (n + 1) → Prop)
  (move_constraint : ∀ (i j : fin (n + 1) × fin (n + 1)),
                     initial_positions i → initial_positions j →
                     dist i j = 1 → dist (i ∘ f) (j ∘ f) ≤ 1)
  : ∃ line_with_slope_1 : fin (n + 1) × fin (n + 1) → Prop, 
    (∃ p q : fin (n + 1) × fin (n + 1),
              initial_positions p ∧ initial_positions q ∧
              line_with_slope_1 p ∧ line_with_slope_1 q ∧
              dist p q = n) :=
sorry

end exists_line_with_n_beetles_l325_325987


namespace sum_log_geometric_seq_l325_325996

noncomputable def a_n (n : ℕ) : ℝ := sorry -- definition of geometric sequence (not explicitly needed, so left as sorry)

variable (a_n : ℕ → ℝ)
variable (h1 : a_n 4 = 2)
variable (h2 : a_n 7 = 5)

theorem sum_log_geometric_seq : 
  (∑ i in finset.range 10, real.log (a_n (i + 1))) = 5 := 
sorry

end sum_log_geometric_seq_l325_325996


namespace bisection_method_next_value_l325_325753

noncomputable def f (x : ℝ) : ℝ := sorry

theorem bisection_method_next_value :
  f 1 < 0 → f 1.5 > 0 → ∃ x₀ ∈ (set.Ioo 1 2), x₀ = 1.25 :=
by
  intro h1 h1_5
  use 1.25
  split
  · exact Ioo.mem.mpr ⟨by norm_num, by norm_num⟩
  · refl

end bisection_method_next_value_l325_325753


namespace max_value_3absx_2absy_l325_325623

theorem max_value_3absx_2absy (x y : ℝ) (h : x^2 + y^2 = 9) : 
  3 * abs x + 2 * abs y ≤ 9 :=
sorry

end max_value_3absx_2absy_l325_325623


namespace wilsons_theorem_l325_325322

theorem wilsons_theorem (p : ℕ) (hp : Nat.Prime p) : (Nat.factorial (p - 1)) % p = p - 1 :=
by
  sorry

end wilsons_theorem_l325_325322


namespace sum_binom_mod_500_l325_325845

-- Define a binomial coefficient function.
noncomputable def binom : ℕ → ℕ → ℕ
| n 0       := 1
| 0 k       := 0
| (n+1) (k+1) := (binom n k) * (n+1) / (k+1)
| _ _       := 0

-- Define the sum of binomial coefficients for multiples of 4.
def sum_binom_multiples_of_4 (n : ℕ) : ℕ :=
  ∑ i in finset.filter (λ i, i % 4 = 0) (finset.range (n+1)), binom n i

-- Theorem statement
theorem sum_binom_mod_500 : sum_binom_multiples_of_4 2015 % 500 = 5 :=
begin
  sorry
end

end sum_binom_mod_500_l325_325845


namespace count_distinct_reals_a_with_integer_roots_l325_325887

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l325_325887


namespace hyperbola_eq_1_hyperbola_eq_2_l325_325782

-- Problem 1: Hyperbola with foci on the y-axis
theorem hyperbola_eq_1 (a b : ℝ) (h1 : b = 3 * a / 2) (h2 : (4 / a ^ 2) - (6 / b ^ 2) = 1) :
  (∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (2 / a) * sqrt(3) = (1 / b) / 2 ∧ 
  ∀ x y : ℝ, (y ^ 2 / a ^ 2) - (x ^ 2 / b ^ 2) = 1 → 
             (y ^ 2 = (4 / 3)) - (x ^ 2 = (3 * y))⁻¹ ) :=
begin
  sorry
end

-- Problem 2: Hyperbola with foci on the x-axis
theorem hyperbola_eq_2 (a b : ℝ) (h3 : b = (4 / 3) * a) (h4 : (9 / a ^ 2) - (12 / b ^ 2) = 1) :
  (∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (5 ^ 2 / 3) = 1 + b ^ 2 / a ^ 2 ∧ 
  ∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1 → 
             (x ^ 2 / (9 / 4) = (y ^ 2 = (12 / 4))) :=
begin
  sorry
end

end hyperbola_eq_1_hyperbola_eq_2_l325_325782


namespace password_count_l325_325795

noncomputable def permutations : ℕ → ℕ → ℕ
| n, r := (n! / (n-r)!)

theorem password_count:
    let n_eng := 26 in
    let r_eng := 2 in
    let n_num := 10 in
    let r_num := 2 in
    permutations n_eng r_eng * permutations n_num r_num = (26 * 25) * (10 * 9) :=
by
  sorry

end password_count_l325_325795


namespace max_points_3D_l325_325526

-- Defining our conditions and type for the problem
def points_3D (n : ℕ) := vector (euclidean_space ℝ (fin 3)) n

-- Defining a function that checks if no three points are collinear
def non_collinear (pts : points_3D n) : Prop :=
  ∀ i j k : fin n, i ≠ j → i ≠ k → j ≠ k → ¬ collinear (pts.nth i) (pts.nth j) (pts.nth k)

-- Defining a function that checks if no triangles formed are obtuse
def non_obtuse_triangle (pts : points_3D n) : Prop :=
  ∀ i j k : fin n, i < j → j < k → ¬ obtuse_triangle (pts.nth i) (pts.nth j) (pts.nth k)

-- Main theorem stating that the maximum n under these conditions is 8
theorem max_points_3D (n : ℕ) (pts : points_3D n) :
  non_collinear pts → non_obtuse_triangle pts → n ≤ 8 :=
begin
  sorry
end

end max_points_3D_l325_325526


namespace smallest_w_l325_325169

theorem smallest_w (w : ℕ) (w_pos : 0 < w) : 
  (∀ n : ℕ, (2^5 ∣ 936 * n) ∧ (3^3 ∣ 936 * n) ∧ (11^2 ∣ 936 * n) ↔ n = w) → w = 4356 :=
sorry

end smallest_w_l325_325169


namespace min_value_of_expression_l325_325305

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 4) :
  (9 / x + 1 / y + 25 / z) ≥ 20.25 :=
by 
  sorry

end min_value_of_expression_l325_325305


namespace particle_position_after_120_moves_l325_325462

noncomputable def ω : Complex := Complex.cis (Real.pi / 6)

noncomputable def z : ℕ → Complex
| 0       := 3
| (n + 1) := ω * z n + 8

theorem particle_position_after_120_moves : z 120 = 3 := 
by
  sorry

end particle_position_after_120_moves_l325_325462


namespace rational_solution_square_l325_325547

theorem rational_solution_square (x : ℚ) :
  (∃ y : ℚ, 3 * x^2 - 5 * x + 9 = y^2) ↔
  (∃ m n : ℤ, (Int.gcd m n = 1) ∧
               x = ⟨5 * n^2 + 6 * m * n, 3 * n^2 - m^2, sorry⟩) :=
sorry

end rational_solution_square_l325_325547


namespace distinct_roots_range_l325_325449

theorem distinct_roots_range (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + 1 = 0 ∧ x2^2 + m * x2 + 1 = 0) ↔ m ∈ Set.Ioo (-∞) (-2) ∪ Set.Ioo 2 ∞ := 
sorry

end distinct_roots_range_l325_325449


namespace prime_product_is_2009_l325_325896

theorem prime_product_is_2009 (a b c : ℕ) 
  (h_primeA : Prime a) 
  (h_primeB : Prime b) 
  (h_primeC : Prime c)
  (h_div1 : a ∣ (b + 8)) 
  (h_div2a : a ∣ (b^2 - 1)) 
  (h_div2c : c ∣ (b^2 - 1)) 
  (h_sum : b + c = a^2 - 1) : 
  a * b * c = 2009 := 
sorry

end prime_product_is_2009_l325_325896


namespace simplify_tan_expression_l325_325344

theorem simplify_tan_expression 
  (h30 : Real.tan (π / 6) = 1 / Real.sqrt 3)
  (h15 : Real.tan (π / 12) = 2 - Real.sqrt 3) :
  (1 + Real.tan (π / 6)) * (1 + Real.tan (π / 12)) = 2 :=
by
  -- State the tangent addition formula for the required angles
  have h_tan_add : Real.tan (π / 4) = (Real.tan (π / 6) + Real.tan (π / 12)) / (1 - Real.tan (π / 6) * Real.tan (π / 12)),
  {
    sorry,
  }
  -- The correct answer proof part is not provided in the brief
  sorry

end simplify_tan_expression_l325_325344


namespace hancho_tape_length_l325_325948

noncomputable def tape_length (x : ℝ) : Prop :=
  (1 / 4) * (4 / 5) * x = 1.5

theorem hancho_tape_length : ∃ x : ℝ, tape_length x ∧ x = 7.5 :=
by sorry

end hancho_tape_length_l325_325948


namespace max_ratio_three_digit_l325_325854

theorem max_ratio_three_digit (x a b c : ℕ) (h1 : 100 * a + 10 * b + c = x) (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : 0 ≤ b ∧ b ≤ 9) (h4 : 0 ≤ c ∧ c ≤ 9) : 
  (x : ℚ) / (a + b + c) ≤ 100 := sorry

end max_ratio_three_digit_l325_325854


namespace largest_int_with_remainder_l325_325525

theorem largest_int_with_remainder (k : ℤ) (h₁ : k < 95) (h₂ : k % 7 = 5) : k = 94 := by
sorry

end largest_int_with_remainder_l325_325525


namespace watch_correction_needed_l325_325472

theorem watch_correction_needed :
  ∀ (loss_per_day : ℝ) (start_hour : ℝ) (start_day : ℝ) (end_hour : ℝ) (end_day : ℝ),
  loss_per_day = 3 →
  start_hour = 12 →
  start_day = 1 →
  end_hour = 18 →
  end_day = 10 →
  let days_passed := end_day - start_day
  let total_hours := days_passed * 24 + (end_hour - start_hour)
  let hourly_loss := loss_per_day / 24
  ∃ (n : ℝ), n = total_hours * hourly_loss ∧ n = 27.75 :=
by {
  intros,
  simp [days_passed, total_hours, hourly_loss],
  use 27.75,
  split; sorry
}

end watch_correction_needed_l325_325472


namespace solve_problem_l325_325166

noncomputable def problem_statement (α : ℝ) : Prop :=
  (sin (α - π / 4) / cos (2 * α) = -sqrt 2) → (sin α + cos α = 1 / 2)

theorem solve_problem (α : ℝ) : problem_statement α :=
  by
  sorry

end solve_problem_l325_325166


namespace four_digit_number_exists_l325_325443

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), 
  B = 3 * A ∧ 
  C = A + B ∧ 
  D = 3 * B ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ 
  1000 * A + 100 * B + 10 * C + D = 1349 :=
by {
  sorry 
}

end four_digit_number_exists_l325_325443


namespace smallest_angle_measure_l325_325828

-- Definitions based on given conditions
def is_isosceles (A B C : ℝ) : Prop := A = B ∨ B = C ∨ C = A
def is_obtuse (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90
def is_20_percent_larger (A B C : ℝ) : Prop := C = 1.2 * 60

-- Lean 4 statement for the proof task
theorem smallest_angle_measure (A B C : ℝ) : 
  is_isosceles A B C ∧ is_obtuse A B C ∧ is_20_percent_larger A B C ∧ A + B + C = 180 →
  (A = B ∨ A = C ∨ B = C) ∧ 54.0 ∈ {A, B, C} :=
sorry

end smallest_angle_measure_l325_325828


namespace simplify_tan_product_l325_325341

theorem simplify_tan_product : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  -- use the angle addition formula for tangent
  have tan_sum : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12) :=
    by rw [Real.tan_add, Real.tan_pi_div_four]
  -- using the given condition tan(45 degrees) = 1
  have tan_45 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  sorry

end simplify_tan_product_l325_325341


namespace number_of_possible_values_of_prime_p_l325_325367

theorem number_of_possible_values_of_prime_p : 
  (∀ p : ℕ, prime p → 
  (1 * p^3 + 0 * p^2 + 1 * p + 4 + 5 * p^2 + 0 * p + 2 + 2 * p^2 + 1 * p + 7 + 2 * p^2 + 3 * p + 1 + 1 * p + 2 = 
  2 * p^2 + 5 * p + 4 + 5 * p^2 + 4 * p + 7 + 6 * p^2 + 7 * p + 5)) → 
  0 :=
sorry

end number_of_possible_values_of_prime_p_l325_325367


namespace arithmetic_sum_S11_l325_325200

theorem arithmetic_sum_S11 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ)
  (h_arith : ∀ n, a (n+1) - a n = d) -- The sequence is arithmetic with common difference d
  (h_sum : S n = n * (a 1 + a n) / 2) -- Sum of the first n terms definition
  (h_condition: a 3 + a 6 + a 9 = 54) :
  S 11 = 198 := 
sorry

end arithmetic_sum_S11_l325_325200


namespace probability_between_R_and_S_l325_325707

variables (P Q R S : ℝ)

theorem probability_between_R_and_S
  (h1 : Q = 4 * P)
  (h2 : Q = 8 * R) :
  let PS := P, QR := R, PQ := Q in
  PQ - PS - QR = 5 / 8 :=
by
  let PS := P,
  let QR := R,
  let PQ := Q,
  sorry

end probability_between_R_and_S_l325_325707


namespace length_of_EG_l325_325198

theorem length_of_EG {E F G H : Type} [EuclideanSpace ℝ (E × F × G × H)]
  (EF FG GH HE DiagonalEG : ℝ) (angleEHG : ℝ)
  (hEF : EF = 12) (hFG : FG = 12) (hGH : GH = 20) (hHE : HE = 20) (hAngle : angleEHG = 60) :
  DiagonalEG = 20 := by
  sorry

end length_of_EG_l325_325198


namespace find_OP_2016_l325_325588

open Matrix

def vector_transform (v : ℕ → ℕ × ℕ) (n : ℕ) : ℕ × ℕ :=
  (v n).fst + (0 : ℕ × ℕ).fst,
  (v n).fst + (v n).snd

def initial_vector : ℕ × ℕ := (2, 0)

noncomputable def vector_seq : ℕ → ℕ × ℕ
| 0 := initial_vector
| (n + 1) := vector_transform vector_seq n

theorem find_OP_2016 : vector_seq 2015 = (2, 4030) :=
by sorry

end find_OP_2016_l325_325588


namespace exam_percentage_l325_325980

theorem exam_percentage (x : ℝ) (h_cond : 100 - x >= 0 ∧ x >= 0 ∧ 60 * x + 90 * (100 - x) = 69 * 100) : x = 70 := by
  sorry

end exam_percentage_l325_325980


namespace value_range_of_function_l325_325063

noncomputable def function_value_range :=
  (λ x : ℝ, real.log (1 + real.sin x) / real.log 2 + real.log (1 - real.sin x) / real.log 2)

theorem value_range_of_function :
  set.range (λ x : ℝ, function_value_range x) = set.Icc (-1 : ℝ) (0 : ℝ) :=
begin
  sorry
end

end value_range_of_function_l325_325063


namespace possible_guesses_l325_325804

-- Defining the digits available
def digits := [2, 2, 2, 3, 3, 3, 3]

-- Conditions
def valid_guess (x y z : List ℕ) : Prop :=
  x.length + y.length + z.length = digits.length ∧
  x.length = z.length ∧
  (∀ d ∈ x ++ y ++ z, d = 2 ∨ d = 3) ∧
  (list.perm (x ++ y ++ z) digits)

-- Statement of the problem
theorem possible_guesses : ∃ n, n = 14 ∧ 
  (∃ (x y z : List ℕ), valid_guess x y z) :=
begin
  sorry
end

end possible_guesses_l325_325804


namespace probability_even_sum_l325_325178

theorem probability_even_sum :
  let x_set := {1, 2, 3, 4, 5}
  let y_set := {7, 8, 9, 10}
  let even x := x % 2 = 0
  let odd x := x % 2 = 1
  let prob_even_x := 2 / 5
  let prob_odd_x := 3 / 5
  let prob_even_y := 1 / 2
  let prob_odd_y := 1 / 2
  in
  (prob_even_x * prob_even_y + prob_odd_x * prob_odd_y) = 1 / 2 :=
by {
  sorry
}

end probability_even_sum_l325_325178


namespace num_real_values_for_integer_roots_l325_325880

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l325_325880


namespace expected_number_of_2s_when_three_dice_rolled_l325_325408

def probability_of_rolling_2 : ℚ := 1 / 6
def probability_of_not_rolling_2 : ℚ := 5 / 6

theorem expected_number_of_2s_when_three_dice_rolled :
  (0 * (probability_of_not_rolling_2)^3 + 
   1 * 3 * (probability_of_rolling_2) * (probability_of_not_rolling_2)^2 + 
   2 * 3 * (probability_of_rolling_2)^2 * (probability_of_not_rolling_2) + 
   3 * (probability_of_rolling_2)^3) = 
   1 / 2 :=
by
  sorry

end expected_number_of_2s_when_three_dice_rolled_l325_325408


namespace probability_of_specific_combination_l325_325960

def total_shirts : ℕ := 3
def total_shorts : ℕ := 7
def total_socks : ℕ := 4
def total_clothes : ℕ := total_shirts + total_shorts + total_socks
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def favorable_outcomes : ℕ := (choose total_shirts 2) * (choose total_shorts 1) * (choose total_socks 1)
def total_outcomes : ℕ := choose total_clothes 4

theorem probability_of_specific_combination :
  favorable_outcomes / total_outcomes = 84 / 1001 :=
by
  -- Proof omitted
  sorry

end probability_of_specific_combination_l325_325960


namespace num_intersections_l325_325971

-- Define a structure for a Circle in a plane.
structure Circle (α : Type _) [Field α] :=
(center : α × α)
(radius : α)

-- Define a type for lines in a plane.
structure Line (α : Type _) [Field α] :=
(a : α) -- Coefficient of x
(b : α) -- Coefficient of y
(c : α) -- Constant term

-- Define the condition that a line intersects a circle.
def intersects_circle {α : Type _} [Field α] (l : Line α) (c : Circle α) : Prop :=
  let sq := (c.radius)^2
  let discr := (l.a * c.center.1 + l.b * c.center.2 + l.c)^2 - (l.a^2 + l.b^2) * sq
  discr > 0

-- Conditions for the problem
variables {α : Type _} [Field α]
variables (L1 L2 L3 : Line α) (C : Circle α)

-- The theorem to be proved
theorem num_intersections (hL1 : intersects_circle L1 C) (hL2 : intersects_circle L2 C) (hL3 : intersects_circle L3 C) : 
  ∃ n, n ∈ {2, 4, 6} :=
sorry

end num_intersections_l325_325971


namespace range_of_f_range_of_a_l325_325090

noncomputable def f (x : ℝ) : ℝ := 3 * |x - 1| + |3 * x + 1|
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := |x + 2| + |x - a|

def range_f : set ℝ := {y : ℝ | ∃ x : ℝ, y = f x}
def range_g (a : ℝ) : set ℝ := {y : ℝ | ∃ x : ℝ, y = g x a}

theorem range_of_f :
  range_f = {y : ℝ | 4 ≤ y} :=
sorry

theorem range_of_a (a : ℝ) : 
  (range_f ∪ range_g a = range_g a) → (-6 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_f_range_of_a_l325_325090


namespace prob_sqrt_sum_l325_325298

noncomputable def T : ℝ := ∑ n in Finset.range (9900 + 1).filter (fun n => n > 0), 1 / (Real.sqrt (n + Real.sqrt (n^2 - 1)))

theorem prob_sqrt_sum :
  T = 4 + 99 * Real.sqrt 2 :=
sorry

end prob_sqrt_sum_l325_325298


namespace ratio_of_area_shaded_triangle_to_large_square_l325_325757

-- Define the side length of each smaller square
def sideLengthSmallSquare : ℝ := 1

-- Define the side length of the large square
def sideLengthLargeSquare : ℝ := 5 * sideLengthSmallSquare

-- Define the area of the large square
def areaLargeSquare : ℝ := sideLengthLargeSquare * sideLengthLargeSquare

-- Define the base and height of the shaded triangle
def baseTriangle : ℝ := sideLengthLargeSquare / 2
def heightTriangle : ℝ := sideLengthLargeSquare / 2

-- Define the area of the shaded triangle
def areaShadedTriangle : ℝ := 1 / 2 * baseTriangle * heightTriangle

-- Define the ratio of the area of the shaded triangle to the area of the large square
def ratio : ℝ := areaShadedTriangle / areaLargeSquare

theorem ratio_of_area_shaded_triangle_to_large_square : ratio = 0.125 := by
  sorry

end ratio_of_area_shaded_triangle_to_large_square_l325_325757


namespace last_remaining_card_l325_325400

noncomputable def largest_power_of_2_leq (n : ℕ) : ℕ :=
  let L := Nat.log2 n in
  L

noncomputable def josephus_step2 (n : ℕ) : ℕ :=
  let L := largest_power_of_2_leq n in
  2 * (n - 2^L) + 1

theorem last_remaining_card (n : ℕ) : ℕ :=
  josephus_step2 n

example : last_remaining_card 52 = 41 := by
  unfold last_remaining_card josephus_step2 largest_power_of_2_leq
  rfl

end last_remaining_card_l325_325400


namespace imag_part_complex_l325_325492

def complex_number := 4 + 3 * Complex.i / (1 + 2 * Complex.i)

theorem imag_part_complex :
  Complex.im complex_number = -1 :=
by 
  sorry

end imag_part_complex_l325_325492


namespace limit_of_growing_line_length_l325_325460

noncomputable def growing_line_length (n : ℕ) : ℝ := 
  2 + ∑ k in Finset.range(n+1).erase(0), (1/3^k + (1/3^k) * Real.sqrt 3)

theorem limit_of_growing_line_length :
  tendsto growing_line_length at_top (nhds (3 + 1/2 * Real.sqrt 3)) :=
sorry

end limit_of_growing_line_length_l325_325460


namespace sum_of_integers_between_cubrt_and_sqrt_l325_325657

open Real Finset

noncomputable def sum_integers_between_cubrt_2006_and_sqrt_2006 : ℝ := 
  let a := Nat.ceil (real.cbrt 2006)
  let b := Nat.floor (real.sqrt 2006)
  let sum := (b - a + 1) * (a + b) / 2
  sum

theorem sum_of_integers_between_cubrt_and_sqrt:
  sum_integers_between_cubrt_2006_and_sqrt_2006 = 912 :=
by
  let a := Nat.ceil (real.cbrt 2006)
  let b := Nat.floor (real.sqrt 2006)
  have h1 : a = 13 := by sorry
  have h2 : b = 44 := by sorry
  calc
    sum_integers_between_cubrt_2006_and_sqrt_2006
        = (44 - 13 + 1) * (13 + 44) / 2 : by rw [h1, h2]
    ... = 32 * 57 / 2 : by norm_num
    ... = 16 * 57 : by norm_num
    ... = 912 : by norm_num

end sum_of_integers_between_cubrt_and_sqrt_l325_325657


namespace total_profit_for_the_month_l325_325813

theorem total_profit_for_the_month (mean_profit_month : ℕ) (num_days_month : ℕ)
(mean_profit_first15 : ℕ) (num_days_first15 : ℕ) 
(mean_profit_last15 : ℕ) (num_days_last15 : ℕ) 
(h1 : mean_profit_month = 350) (h2 : num_days_month = 30) 
(h3 : mean_profit_first15 = 285) (h4 : num_days_first15 = 15) 
(h5 : mean_profit_last15 = 415) (h6 : num_days_last15 = 15) : 
(mean_profit_first15 * num_days_first15 + mean_profit_last15 * num_days_last15) = 10500 := by
  sorry

end total_profit_for_the_month_l325_325813


namespace ratio_of_areas_l325_325907

open Set

noncomputable def vector_space : Type := ℝ^3

def point_in_triangle (O A B C : vector_space) : Prop := 
∃ α β γ : ℝ, α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧ 
  O = α • A + β • B + γ • C

def vector_relation (O A B C : vector_space) : Prop :=
  O + 2 • B + 3 • C = 0

theorem ratio_of_areas (O A B C : vector_space)
  (h1 : point_in_triangle O A B C)
  (h2 : vector_relation O A B C) :
  ∃ k : ℝ, k = 3 :=
sorry

end ratio_of_areas_l325_325907


namespace count_positive_area_triangles_l325_325145

-- Define the grid size
def grid_size : ℕ := 6

-- Defining the main theorem
theorem count_positive_area_triangles :
  (set.univ.powerset.to_finset.filter (λ s : finset (ℤ × ℤ), s.card = 3 ∧ ¬collinear s)).card = 6628 :=
sorry

end count_positive_area_triangles_l325_325145


namespace distinct_a_count_l325_325891

theorem distinct_a_count :
  ∃ (a_set : Set ℝ), (∀ x ∈ a_set, ∃ r s : ℤ, r + s = -x ∧ r * s = 9 * x) ∧ a_set.toFinset.card = 3 :=
by 
  sorry

end distinct_a_count_l325_325891


namespace sharon_plums_l325_325715

-- Define the given conditions
variables (S A : ℕ)
axiom allan_plums : A = 10
axiom plum_difference : S - A = 3

-- Goal: Prove that Sharon has 13 plums
theorem sharon_plums : S = 13 :=
by
  have h1 : A = 10 := allan_plums,
  have h2 : S - A = 3 := plum_difference,
  sorry

end sharon_plums_l325_325715


namespace percentage_profit_is_correct_l325_325735

-- Variables
noncomputable def C : ℝ := 424 -- Derived from condition 872 - C = C - 448

-- Sale Price
def SP : ℝ := 990

-- Profit calculation
def profit : ℝ := SP - C

-- Percentage profit calculation
def perc_profit : ℝ := (profit / C) * 100

-- Expected percentage profit
def expected_perc_profit : ℝ := 133.49

-- The theorem statement
theorem percentage_profit_is_correct :
  perc_profit = expected_perc_profit :=
by
  -- Proof omitted with sorry
  sorry

end percentage_profit_is_correct_l325_325735


namespace most_pieces_day_and_maximum_number_of_popular_days_l325_325183

-- Definitions for conditions:
def a_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then 3 * n
else 65 - 2 * n

def S_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then (3 + 3 * n) * n / 2
else 273 + (51 - n) * (n - 13)

-- Propositions to prove:
theorem most_pieces_day_and_maximum :
  ∃ k a_k, (1 ≤ k ∧ k ≤ 31) ∧
           (a_k = a_n k) ∧
           (∀ n, 1 ≤ n ∧ n ≤ 31 → a_n n ≤ a_k) ∧
           k = 13 ∧ a_k = 39 := 
sorry

theorem number_of_popular_days :
  ∃ days_popular,
    (∃ n1, 1 ≤ n1 ∧ n1 ≤ 13 ∧ S_n n1 > 200) ∧
    (∃ n2, 14 ≤ n2 ∧ n2 ≤ 31 ∧ a_n n2 < 20) ∧
    days_popular = (22 - 12 + 1) :=
sorry

end most_pieces_day_and_maximum_number_of_popular_days_l325_325183


namespace infinitely_many_ns_l325_325710

theorem infinitely_many_ns {n : ℕ} : ∃ (n_ℕ : ℕ) (infinitely_many : ∀ k: ℕ, ∃ n: ℕ, k ≤ n) 
  (x y: ℕ), (x^2 - 2*y^2 = -1) → (⌊sqrt 2 * n⌋ = x^2) :=
begin
  sorry
end

end infinitely_many_ns_l325_325710


namespace max_one_acute_point_l325_325078

-- Given 2006 points on a plane, prove there can be at most one point 
-- that determines an acute-angled triangle with any other two points.
theorem max_one_acute_point (points : Finset (ℝ × ℝ))
  (h_size : points.card = 2006) :
  ∃ p : (ℝ × ℝ), ∀ x y ∈ points, x ≠ y → 
    ¬ (∃ a b c ∈ points, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
        (triangle_acute (a, b, c))) := sorry

noncomputable def triangle_acute (a b c : (ℝ × ℝ)) : Prop :=
  -- This function will return if a given triangle formed by the points a, b, c is an acute-angled triangle.
  sorry

end max_one_acute_point_l325_325078


namespace length_MN_in_trapezoid_l325_325768

variable {a b c d : ℝ}

theorem length_MN_in_trapezoid (h_parallel : BC ∥ AD)
  (h_M : M = intersection_angle_bisectors A B)
  (h_N : N = intersection_angle_bisectors C D)
  (h_AB : length AB = a)
  (h_BC : length BC = b)
  (h_CD : length CD = c)
  (h_DA : length DA = d) : 
  length MN = (b + d - a - c) / 2 :=
sorry

end length_MN_in_trapezoid_l325_325768


namespace range_of_S_l325_325171

theorem range_of_S (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 1) (S : ℝ) (hS : S = 3 * x^2 - 2 * y^2) :
  -2 / 3 < S ∧ S ≤ 3 / 2 :=
sorry

end range_of_S_l325_325171


namespace president_vicepresident_selection_l325_325812

theorem president_vicepresident_selection :
  let boys := 18
  let girls := 12
  let boys_seniors := boys / 2
  let boys_juniors := boys / 2
  let girls_seniors := girls / 2
  let girls_juniors := girls / 2
  let total_ways := 2 * 2 * 9 * 9
  total_ways = 324 :=
by
  -- Definitions
  let boys := 18
  let girls := 12
  let boys_seniors := boys / 2
  let boys_juniors := boys / 2
  let girls_seniors := girls / 2
  let girls_juniors := girls / 2
  let total_ways := 2 * 2 * 9 * 9
  -- Statement
  show total_ways = 324 from sorry

end president_vicepresident_selection_l325_325812


namespace calc_f_prime_pi_div_2_l325_325096

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * x * (f' 0) -- f'(0) will be defined in the scope of the problem

theorem calc_f_prime_pi_div_2 : 
  let f' (x : ℝ) : ℝ := (λ x, Real.cos x + 2 * (λ _, f' 0) x) x in
  f' 0 = -1 → f' (Real.pi / 2) = -2 :=
by
  intro h
  have f'_def : ∀ x, (λ x, Real.cos x + 2 * (λ _, f' 0) x) x = f' x := by
    intro x
    reflexivity
  -- Additional steps needed for the proof would be here
  sorry

end calc_f_prime_pi_div_2_l325_325096


namespace square_area_from_circle_l325_325005

noncomputable def circle_area : ℝ := 39424
noncomputable def radius_of_circle (A : ℝ) : ℝ := real.sqrt (A / real.pi)
noncomputable def radius : ℝ := radius_of_circle circle_area
noncomputable def perimeter_of_square (r : ℝ) : ℝ := r
noncomputable def side_length_of_square (P : ℝ) : ℝ := P / 4
noncomputable def area_of_square (s : ℝ) : ℝ := s * s

theorem square_area_from_circle :
  let r := radius in let P := perimeter_of_square r in let s := side_length_of_square P in
  area_of_square s ≈ 785.12 :=
by
  sorry

end square_area_from_circle_l325_325005


namespace quadratic_coefficients_l325_325036

theorem quadratic_coefficients :
  ∀ x : ℝ, x * (x + 2) = 5 * (x - 2) → ∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 10 ∧ a * x^2 + b * x + c = 0 := by
  intros x h
  use 1, -3, 10
  sorry

end quadratic_coefficients_l325_325036


namespace no_integer_solution_for_system_l325_325325

theorem no_integer_solution_for_system :
  ¬ ∃ (a b c d : ℤ), 
    (a * b * c * d - a = 1961) ∧ 
    (a * b * c * d - b = 961) ∧ 
    (a * b * c * d - c = 61) ∧ 
    (a * b * c * d - d = 1) :=
by {
  sorry
}

end no_integer_solution_for_system_l325_325325


namespace min_packs_needed_l325_325721

theorem min_packs_needed (P8 P15 P30 : ℕ) (h: P8 * 8 + P15 * 15 + P30 * 30 = 120) : P8 + P15 + P30 = 4 :=
by
  sorry

end min_packs_needed_l325_325721


namespace hash_fn_triple_40_l325_325851

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_fn_triple_40 : hash_fn (hash_fn (hash_fn 40)) = 12.56 := by
  sorry

end hash_fn_triple_40_l325_325851


namespace f_of_9_l325_325604

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

def condition : Prop := f (Real.sqrt 2) α = 2

theorem f_of_9 (α : ℝ) (h : condition α) : f 9 α = 81 :=
by
  have h_alpha : α = 2 :=
    sorry -- solve for α from the given condition
  rw [h_alpha]
  show f 9 2 = 81
  simp [f]
  norm_num

end f_of_9_l325_325604


namespace altered_solution_contains_60_liters_of_detergent_l325_325393

/-- Given the conditions:
  * The original volume ratio of bleach to detergent to water is 4:40:100.
  * The ratio of bleach to detergent is tripled.
  * The ratio of detergent to water is halved.
  * The altered solution contains 300 liters of water.
  
  Prove that the amount of detergent in the altered solution is 60 liters.
--/
theorem altered_solution_contains_60_liters_of_detergent :
  ∃ (detergent bleach water : ℕ), 
    let r_orig := (4, 40, 100) in
    let r_bleach_detergent := 3 * r_orig.1 / r_orig.2 in
    let r_detergent_water := r_orig.2 / (2 * r_orig.3) in
    let r_new := (r_bleach_detergent, 10, 50) in
    water = 300 →
    r_new.3 * water / 50 = 60 :=
by
  sorry

end altered_solution_contains_60_liters_of_detergent_l325_325393


namespace distinct_sum_product_problem_l325_325323

theorem distinct_sum_product_problem (S : ℤ) (hS : S ≥ 100) :
  ∃ a b c P : ℤ, a > b ∧ b > c ∧ a + b + c = S ∧ a * b * c = P ∧ 
    ¬(∀ x y z : ℤ, x > y ∧ y > z ∧ x + y + z = S → a = x ∧ b = y ∧ c = z) := 
sorry

end distinct_sum_product_problem_l325_325323


namespace area_triangle_sum_l325_325977

theorem area_triangle_sum (A B C D E : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (AD_DE : dist A D = dist D E)
  (AC_2 : dist A C = 2)
  (BAC_30 : angle A B C = 30)
  (ABC_80 : angle B A C = 80)
  (ACB_70 : angle C A B = 70)
  (DEC_70 : angle D E C = 70)
  (mid_E : is_midpoint E B C)
  (AD_cond : dist A D = dist D C)
  : area_triangle A B C + 2 * area_triangle A D E = 2 * sin 80 + sin 70 := sorry

end area_triangle_sum_l325_325977


namespace minimum_value_fraction_l325_325921

theorem minimum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (2 / x + 1 / y) >= 2 * Real.sqrt 2 :=
sorry

end minimum_value_fraction_l325_325921


namespace option_b_right_triangle_l325_325825

theorem option_b_right_triangle : 
    (sqrt 2)^2 + (sqrt 3)^2 = (sqrt 5)^2 :=
sorry

end option_b_right_triangle_l325_325825


namespace DimaPassword_l325_325859

theorem DimaPassword :
  ∃ a : List ℕ,
  (a.length = 5 ∧
  a.get! 0 = 5 ∧
  a.get! 1 = 12 ∧
  a.get! 2 = 19 ∧
  a.get! 3 = 26 ∧
  a.get! 4 = 33 ∧
  ∀ i : ℕ, i < 4 → a.get! (i + 1) - a.get! i = 7) :=
begin
  sorry
end

end DimaPassword_l325_325859


namespace symmetry_point_sum_l325_325309

def f (x : ℝ) : ℝ := x^3 + Real.sin x + 2

theorem symmetry_point_sum :
  (f (-1) + f (-9/10) + f (-8/10) + f (-7/10) + f (-6/10) + f (-5/10) + f (-4/10) + f (-3/10) + f (-2/10) + f (-1/10) +
   f (1/10) + f (2/10) + f (3/10) + f (4/10) + f (5/10) + f (6/10) + f (7/10) + f (8/10) + f (9/10) + f (1) + f 0) = 42 :=
by
  sorry

end symmetry_point_sum_l325_325309


namespace angle_BAC_90_degrees_l325_325289

theorem angle_BAC_90_degrees
  (A B C M P Q : Type*)
  [linear_ordered_field Type*]
  [euclidean_geometry geometric_space motif] -- assumes specific geometric context exists
  -- Conditions
  (hMmid : M = midpoint B C)
  (hPfoot : P = foot_of_perpendicular M A B)
  (hQfoot : Q = foot_of_perpendicular M A C)
  (harea : area_triangle M P Q = (1 / 4) * area_triangle A B C) :
  angle A B C = 90 :=
sorry

end angle_BAC_90_degrees_l325_325289


namespace no_bounded_constant_exists_l325_325895

def digitSum (n : ℕ) : ℕ :=
  -- Implementation to calculate the sum of the digits of n
  sorry

theorem no_bounded_constant_exists :
  ¬ ∃ c : ℝ, (0 < c) ∧ ∀ n : ℕ, (digitSum n) / (digitSum (n * n)) ≤ c :=
begin
  sorry
end

end no_bounded_constant_exists_l325_325895


namespace cannot_connect_points_l325_325807

theorem cannot_connect_points 
  (x y : ℤ) 
  (A B : ℤ × ℤ) 
  (hA : A = (19, 47)) 
  (hB : B = (12, 17)) 
  (transformation1 : Π (p : ℤ × ℤ), p = (p.1 + 3 * p.2, p.2)) 
  (transformation2 : Π (p : ℤ × ℤ), p = (p.1, p.2 - 2 * p.1)) : 
  (A = (19, 47) → B = (12, 17) → ∀ x y, (x ≡ 1 [MOD 3] → B.1 ≡ 0 [MOD 3]) → (x ≡ B.1 [MOD 3] → A.1 ≡ 0 [MOD 3]) → false) := 
sorry

end cannot_connect_points_l325_325807


namespace convex_polygon_diagonals_nine_sides_l325_325463

theorem convex_polygon_diagonals_nine_sides (n : ℕ) (h₁ : n = 9) : 
  ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 := 
by
  use (n * (n - 3)) / 2
  split
  . refl
  . rw [h₁]
    norm_num
    sorry

end convex_polygon_diagonals_nine_sides_l325_325463


namespace total_children_l325_325331

theorem total_children (sons daughters : ℕ) (h1 : sons = 3) (h2 : daughters = 6 * sons) : (sons + daughters) = 21 :=
by
  sorry

end total_children_l325_325331


namespace cricket_team_members_l325_325801

-- Define variables and conditions
variable (n : ℕ) -- let n be the number of team members
variable (T : ℕ) -- let T be the total age of the team
variable (average_team_age : ℕ := 24) -- given average age of the team
variable (wicket_keeper_age : ℕ := average_team_age + 3) -- wicket keeper is 3 years older
variable (remaining_players_average_age : ℕ := average_team_age - 1) -- remaining players' average age

-- Given condition which relates to the total age
axiom total_age_condition : T = average_team_age * n

-- Given condition for the total age of remaining players
axiom remaining_players_total_age : T - 24 - 27 = remaining_players_average_age * (n - 2)

-- Prove the number of members in the cricket team
theorem cricket_team_members : n = 5 :=
by
  sorry

end cricket_team_members_l325_325801


namespace arithmetic_sequence_sum_l325_325992

variable {a : ℕ → ℝ} 

-- Condition: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Condition: Given sum of specific terms in the sequence
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 10 = 16

-- Problem: Proving the correct answer
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : given_condition a) :
  a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l325_325992


namespace building_height_l325_325412

theorem building_height (H : ℝ) 
                        (bounced_height : ℕ → ℝ) 
                        (h_bounce : ∀ n, bounced_height n = H / 2 ^ (n + 1)) 
                        (h_fifth : bounced_height 5 = 3) : 
    H = 96 := 
by {
  sorry
}

end building_height_l325_325412


namespace tan_product_simplification_l325_325352

theorem tan_product_simplification :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  have h : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have tan_addition :
    ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) := Real.tan_add
  sorry

end tan_product_simplification_l325_325352


namespace intersection_x_value_l325_325060

theorem intersection_x_value:
  ∃ x y : ℝ, y = 4 * x - 29 ∧ 3 * x + y = 105 ∧ x = 134 / 7 :=
by
  sorry

end intersection_x_value_l325_325060


namespace number_of_solutions_l325_325860

def g (x : ℝ) : ℝ := abs (3^x - 1)

theorem number_of_solutions (k : ℝ) :
  ((k < 0) → ∀ x : ℝ, g x ≠ k) ∧
  ((k = 0) → ∃! x : ℝ, g x = k) ∧
  ((k ≥ 1) → ∃! x : ℝ, g x = k) ∧
  ((0 < k ∧ k < 1) → ∃ x1 x2 : ℝ, g x1 = k ∧ g x2 = k ∧ x1 ≠ x2) :=
by
  sorry

end number_of_solutions_l325_325860


namespace regions_formed_l325_325216

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l325_325216


namespace triangle_area_max_l325_325582

noncomputable def area_max {a b c : ℝ} (A B C : ℝ) (h1 : a = 2)
  (h2 : a * Real.sin C = c * Real.cos A) : ℝ :=
  Real.sqrt 2 + 1

theorem triangle_area_max {a b c A B C : ℝ} (h1 : a = 2)
  (h2 : a * Real.sin C = c * Real.cos A) :
  ∃ b c, (area_max A B C h1 h2 = Real.sqrt 2 + 1) :=
sorry

end triangle_area_max_l325_325582


namespace trajectory_of_midpoint_l325_325295

variables (x y : ℝ) (P O M : ℝ × ℝ)

-- Condition 2: O is the origin
def O := (0, 0) 

-- Condition 3: M is the midpoint of OP
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def M := midpoint O (2*x, 2*y) 

-- Definition of the trajectory equation
noncomputable def trajectory_equation : Prop := 
  x^2 - 4*y^2 = 1

-- The theorem to be proved
theorem trajectory_of_midpoint : 
  (P = (2*x, 2*y)) ∧ (O = (0, 0)) ∧ (M = (x, y)) → trajectory_equation x y :=
by {
  sorry
}

end trajectory_of_midpoint_l325_325295


namespace simplify_tan_product_l325_325337

theorem simplify_tan_product : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  -- use the angle addition formula for tangent
  have tan_sum : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12) :=
    by rw [Real.tan_add, Real.tan_pi_div_four]
  -- using the given condition tan(45 degrees) = 1
  have tan_45 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  sorry

end simplify_tan_product_l325_325337


namespace x_n_general_term_t_range_sum_sn_bound_l325_325935

-- Define the function
def f (x : ℝ) : ℝ := (Real.log (x + 1) / Real.log 3) / (x + 1)

-- Define the sequence rule
def x_seq : ℕ → ℝ
| 1 => 2
| (n+1) => 3 * (x_seq n) + 2

-- Define the general term we're aiming to prove
def x_n_general (n : ℕ) : ℝ := 3^n - 1

-- Define y_n
def y_seq (n : ℕ) : ℝ := f (x_seq n)

-- Definition of S_n
def S (n : ℕ) : ℝ := (1/2) * (y_seq (n + 1) + y_seq n) * (x_seq (n + 1) - x_seq n)

theorem x_n_general_term (n : ℕ) : x_seq n = x_n_general n := by
  sorry

theorem t_range (t : ℝ) (m : ℝ) (n : ℕ) (h1 : m ∈ Set.Icc (-1) 1) :
  3 * t ^ 2 - 6 * m * t + 1/3 > y_seq n → t ∈ Set.Ioi (2 : ℝ) ∪ Set.Iio (-2 : ℝ) := by
  sorry

theorem sum_sn_bound (n : ℕ) : ∑ i in Finset.range n, 1 / (i + 1) * S (i + 1) < 5 / 4 := by
  sorry

end x_n_general_term_t_range_sum_sn_bound_l325_325935


namespace angle_ABQ_eq_angle_BAS_l325_325743

variables {A B C T U P Q R S : Type} [AcuteAngledTriangle ABC] 
variables (h1 : tangent B (circumcircle ABC) = tangent C T) 
variables (h2 : tangent A (circumcircle ABC) = tangent C U)
variables (h3 : line AT ∩ BC = P)
variables (h4 : midpoint Q AP)
variables (h5 : line BU ∩ CA = R)
variables (h6 : midpoint S BR)

open Real EuclideanGeometry

theorem angle_ABQ_eq_angle_BAS
  (h_acute : isAcuteAngledTriangle A B C):
  ∠ABQ = ∠BAS := 
sorry

end angle_ABQ_eq_angle_BAS_l325_325743


namespace odd_function_periodic_period_l325_325300

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def f_def (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x) else 0

theorem odd_function_periodic_period:
  let f := f_def
  (isOddFunction f) ∧ (hasPeriod f 2) →
  f ( - 5 / 2 ) = - 1 / 2 :=
  sorry

end odd_function_periodic_period_l325_325300


namespace sum_first_13_terms_l325_325640

variable {a : ℕ → ℤ}

-- Defining the conditions of the arithmetic sequence
axiom arithmetic_seq (n d : ℤ) (a₁ : ℤ) : a n = a₁ + (n - 1) * d 

-- Defining the given condition
axiom given_condition : a 7 - a 5 + 8 = a 9

-- The statement to prove
theorem sum_first_13_terms : (∑ i in finset.range 13, a (i + 1)) = 104 :=
sorry

end sum_first_13_terms_l325_325640


namespace triangle_bisector_length_l325_325779

theorem triangle_bisector_length (A B C D M O : Point) (AB AC BC AD BD BO : ℝ) 
    (angle_A : Real.Angle) (circum_radius_ADC : ℝ)
    (AB_is_1 : AB = 1) 
    (A_eq_60 : angle_A = 60)
    (R_eq : circum_radius_ADC = (2 * sqrt 3) / 3)
    (bisector_AD : bisects_angle A) :
    ∃ BM : ℝ, BM = (sqrt 21) / 9 := 
  sorry

end triangle_bisector_length_l325_325779


namespace circle_region_count_l325_325236

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l325_325236


namespace hexagon_shell_arrangements_l325_325661

theorem hexagon_shell_arrangements : (12.factorial / 6) = 79833600 := 
by
  -- math proof here
  sorry

end hexagon_shell_arrangements_l325_325661


namespace triangle_ABC_BC_length_l325_325209

theorem triangle_ABC_BC_length :
  ∀ (ABC : Type) [triangle ABC]
  (A B C : points ABC)
  (hA : angle A = 45)
  (hB : angle B = 90)
  (hAC : segment AC = 6),
  segment BC = 3 * sqrt 2 :=
by
  intros ABC _ A B C hA hB hAC
  -- Proof would go here
  sorry

end triangle_ABC_BC_length_l325_325209


namespace fraction_sum_l325_325155

theorem fraction_sum (a b : ℕ) (h1 : 0.36 = a / b) (h2: Nat.gcd a b = 1) : a + b = 15 := by
  sorry

end fraction_sum_l325_325155


namespace sqrt_defined_range_l325_325618

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by sorry

end sqrt_defined_range_l325_325618


namespace triangle_PQL_angles_l325_325781

-- Definitions based on the conditions
variables (P Q R L M : Type) [euclidean_space P Q R] -- P, Q, R, L, M are points in the Euclidean space
-- we assume here that we can define an angle function which measures angles between points in the Euclidean space

-- Conditions as definitions
def is_angle_bisector (QL : Triangle) : Prop := 
-- To be implemented: QL is the angle bisector of triangle PQR
sorry

def is_circumcenter (M : Triangle) : Prop := 
-- To be implemented: M is the circumcenter of triangle PQL
sorry

def symmetric_about (M L PQ : Line) : Prop :=
-- To be implemented: M and L are symmetric with respect to line PQ
sorry

-- Given the conditions, we assert the conclusion
theorem triangle_PQL_angles 
  (P Q R L M : Point)
  (H1 : is_angle_bisector QL)
  (H2 : is_circumcenter M)
  (H3 : symmetric_about M L PQ) :
  (angle P Q L = 30) ∧ (angle Q P L = 30) ∧ (angle P L Q = 120) :=
sorry

end triangle_PQL_angles_l325_325781


namespace solve_inequality_l325_325722

theorem solve_inequality :
  (x^2 - 4*x - 45) / (x + 7) < 0 ↔ (x ∈ -7 < x ∧ x < -5 ∨ -5 < x ∧ x < 9) :=
by
  sorry

end solve_inequality_l325_325722


namespace binom_20_10_l325_325578

-- Given conditions
def binom_18_8 : ℕ := 31824
def binom_18_9 : ℕ := 48620
def binom_18_10 : ℕ := 43758

theorem binom_20_10 : nat.choose 20 10 = 172822 := by
  have h1 : nat.choose 19 9 = binom_18_8 + binom_18_9 := rfl
  have h2 : nat.choose 19 10 = binom_18_9 + binom_18_10 := rfl
  have h3 : nat.choose 20 10 = nat.choose 19 9 + nat.choose 19 10 := rfl
  rw [h1, h2, h3]
  exact rfl

end binom_20_10_l325_325578


namespace present_value_of_amount_l325_325047

-- Define constants and functions according to problem conditions and solution
def annual_increase (x : ℝ) : ℝ := x * (9 / 8)
def after_two_years (p : ℝ) : ℝ := annual_increase (annual_increase p)

-- State the main theorem we want to prove
theorem present_value_of_amount (p : ℝ) (h : after_two_years p = 3645) : p ≈ 2880 :=
sorry -- Proof is not required.

end present_value_of_amount_l325_325047


namespace arithmetic_seq_a7_l325_325989

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  (h3 : ∀ n, a (n + 1) = a n + d) : a 7 = 8 :=
sorry

end arithmetic_seq_a7_l325_325989


namespace Jackson_emails_l325_325279

theorem Jackson_emails : 
    ∀ (deleted1 deleted2 received1 received2 received_final : ℕ), 
    deleted1 = 50 → deleted2 = 20 → received1 = 15 → received2 = 5 → received_final = 10 → 
    (received1 + received2 + received_final = 30) :=
begin
    intros deleted1 deleted2 received1 received2 received_final,
    intros h1 h2 h3 h4 h5,
    sorry
end

end Jackson_emails_l325_325279


namespace sin_squared_sum_l325_325450

theorem sin_squared_sum (A B C : ℝ) (h1 : sin A + sin B + sin C = 0) (h2 : cos A + cos B + cos C = 0) : 
    sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 3 / 2 :=
by
  sorry

end sin_squared_sum_l325_325450


namespace solve_diff_eq_l325_325534

def solution_of_diff_eq (x y : ℝ) (y' : ℝ → ℝ) : Prop :=
  (x + y) * y' x = 1

def initial_condition (y x : ℝ) : Prop :=
  y = 0 ∧ x = -1

theorem solve_diff_eq (x : ℝ) (y : ℝ) (y' : ℝ → ℝ) (h1 : initial_condition y x) (h2 : solution_of_diff_eq x y y') :
  y = -(x + 1) :=
by 
  sorry

end solve_diff_eq_l325_325534


namespace quadratic_radical_equivalence_l325_325975

theorem quadratic_radical_equivalence {a : ℝ} (h : a + 2 = 3a) : a = 1 :=
by 
  sorry

end quadratic_radical_equivalence_l325_325975


namespace angle_BCD_measure_l325_325648

-- Definitions based on given conditions
def in_circle (E B D C A : Point) : Prop :=
  diameter E B ∧
  parallel E B D C ∧
  parallel A B E D ∧
  ratio (angle A E B) (angle A B E) 3 7

-- Correct answer statement
theorem angle_BCD_measure (E B D C A : Point) (h_circle : in_circle E B D C A) :
  measure_angle B C D = 117 := 
  sorry

end angle_BCD_measure_l325_325648


namespace precision_of_rounded_value_l325_325447

-- Definition of the original problem in Lean 4
def original_value := 27390000000

-- Proof statement to check the precision of the rounded value to the million place
theorem precision_of_rounded_value :
  (original_value % 1000000 = 0) :=
sorry

end precision_of_rounded_value_l325_325447


namespace f_decreasing_implies_f_x1_gt_f_x2_l325_325308

variable (f : ℝ → ℝ)

-- Given conditions
axiom has_deriv_f : ∀ x : ℝ, Differentiable ℝ f
axiom symm_property : ∀ x : ℝ, f (2 - x) = f x
axiom derivative_condition : ∀ x : ℝ, f' x < 0 → x > 1 →  f x < 0

-- Guard conditions
variables (x1 x2 : ℝ)
axiom x1_less_x2 : x1 < x2
axiom sum_greater_two : x1 + x2 > 2

-- Prove f(x1) > f(x2)
theorem f_decreasing_implies_f_x1_gt_f_x2 : f x1 > f x2 :=
begin
  sorry -- The proof can be completed here
end

end f_decreasing_implies_f_x1_gt_f_x2_l325_325308


namespace problem_statement_l325_325107

open Complex

noncomputable def z : ℂ := ((1 - I)^2 + 3 * (1 + I)) / (2 - I)

theorem problem_statement :
  z = 1 + I ∧ (∀ (a b : ℝ), (z^2 + a * z + b = 1 - I) → (a = -3 ∧ b = 4)) :=
by
  sorry

end problem_statement_l325_325107


namespace teams_solving_Q3_and_Q4_l325_325481

noncomputable def teams_solving_Q1 : ℕ := 45
noncomputable def teams_solving_Q2 : ℕ := 40
noncomputable def teams_solving_Q3 : ℕ := 35
noncomputable def teams_solving_Q4 : ℕ := 30
noncomputable def total_teams : ℕ := 50

theorem teams_solving_Q3_and_Q4 :
  ∀ T : ℕ, T = total_teams →
  ∀ a b c d : ℕ, a = teams_solving_Q1 →
  b = teams_solving_Q2 →
  c = teams_solving_Q3 →
  d = teams_solving_Q4 →
  (∀ t ∈ (finset.range T), (teams_solving_Q1 t + teams_solving_Q2 t + teams_solving_Q3 t + teams_solving_Q4 t ≤ 3 * T)
    →
  (∃ x : ℕ, x = 15)) :=
begin
  intros T hT a ha b hb c hc d hd h,
  use 15,
  sorry,
end

end teams_solving_Q3_and_Q4_l325_325481


namespace find_n_l325_325484

noncomputable def geometric_sequence_problem (S1 S2 : ℝ) (n: ℝ) (a₁ b₁ a₂ b₂: ℝ) (r s: ℝ) :=
  a₁ = 18 ∧
  b₁ = 6 ∧
  a₂ = 18 ∧
  b₂ = 6 + n ∧
  b₁ = a₁ * r ∧
  b₂ = a₂ * s ∧
  S1 = a₁ / (1 - r) ∧
  S2 = a₂ / (1 - s) ∧
  S2 = 5 * S1

theorem find_n : ∃ n: ℝ, geometric_sequence_problem 27 135 n 18 6 18 (6 + n) (1 / 3) ((6 + n) / 18) :=
begin
  use 9.6,
  split,
  { refl },  -- For a₁ = 18
  split,
  { refl },  -- For b₁ = 6
  split,
  { refl },  -- For a₂ = 18
  split,
  { refl },  -- For b₂ = 6 + n
  split,
  { linarith },  -- For b₁ = a₁ * r
  split,
  { linarith },  -- For b₂ = a₂ * s
  split,
  { linarith },  -- For S1 = a₁ / (1 - r)
  split,
  { linarith },  -- For S2 = a₂ / (1 - s)
  { linarith },  -- For S2 = 5 * S1
end

end find_n_l325_325484


namespace regions_formed_l325_325218

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l325_325218


namespace minimum_sets_purchased_l325_325767

theorem minimum_sets_purchased (total_amount : ℕ) (price_per_set : ℕ) (sets_purchased_20 : ℕ) (total_amount_spent : ℕ) :
  total_amount = 6800 → price_per_set = 20 → sets_purchased_20 = 178 → total_amount_spent = sets_purchased_20 * price_per_set →
  ∃ n : ℕ, n ≥ sets_purchased_20 :=
by
  intros htotal_amount hprice_per_set hsets_purchased_20 htotal_amount_spent htotal_amount_eq
  use sets_purchased_20
  rw hsets_purchased_20
  exact le_refl _

end minimum_sets_purchased_l325_325767


namespace second_derivative_f_l325_325905

variable {x : ℝ}

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem second_derivative_f : (deriv^[2] f) x = -Real.cos x - Real.sin x :=
by sorry

end second_derivative_f_l325_325905


namespace simplify_tan_expr_l325_325349

-- Definition of the tangents of 30 degrees and 15 degrees
def tan_30 : ℝ := Real.tan (Real.pi / 6)
def tan_15 : ℝ := Real.tan (Real.pi / 12)

-- Theorem stating that (1 + tan_30) * (1 + tan_15) = 2
theorem simplify_tan_expr : (1 + tan_30) * (1 + tan_15) = 2 :=
by
  sorry

end simplify_tan_expr_l325_325349


namespace visitors_not_ill_l325_325477

theorem visitors_not_ill (total_visitors : ℕ) (percent_ill : ℕ) (H1 : total_visitors = 500) (H2 : percent_ill = 40) : 
  total_visitors * (100 - percent_ill) / 100 = 300 := 
by 
  sorry

end visitors_not_ill_l325_325477


namespace all_players_same_flip_probability_l325_325017

-- Define the probability of a single player flipping their first head on the n-th flip
def single_flip_probability (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 2) ^ n

-- Define the combined probability for all five players flipping their first head on the n-th flip
def combined_probability (n : ℕ) : ℝ :=
  (single_flip_probability n) ^ 5

-- Define the infinite geometric series sum for the probability of all five players
def geometric_series_sum (a r : ℝ) : ℝ :=
  a / (1 - r)

-- Define the sum of combined probability over all n starting from 1
noncomputable def total_probability : ℝ :=
  geometric_series_sum ((1 / 2) ^ 5) ((1 / 2) ^ 5)

-- Statement of the theorem
theorem all_players_same_flip_probability : total_probability = 1 / 31 := by
  sorry

end all_players_same_flip_probability_l325_325017


namespace circle_regions_l325_325271

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l325_325271


namespace range_of_m_l325_325672

-- Define the ellipse and conditions
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2 / m) + (y^2 / 2) = 1
def point_exists (M : ℝ × ℝ) (C : ℝ → ℝ → ℝ → Prop) : Prop := ∃ p : ℝ × ℝ, C p.1 p.2 (M.1 + M.2)

-- State the theorem
theorem range_of_m (m : ℝ) (h₁ : ellipse x y m) (h₂ : point_exists M ellipse) :
  (0 < m ∧ m <= 1/2) ∨ (8 <= m) := 
sorry

end range_of_m_l325_325672


namespace closest_integer_to_cube_root_of_80_l325_325761

theorem closest_integer_to_cube_root_of_80 : 
  ∃ (n : ℤ), n = 4 ∧ (∀ m : ℤ, m ≠ n → |(m ^ 3 - 80 : ℤ)| > |(n ^ 3 - 80 : ℤ)) :=
by
  use 4
  sorry

end closest_integer_to_cube_root_of_80_l325_325761


namespace quadratic_roots_equation_l325_325391

theorem quadratic_roots_equation :
  let (r₁, r₂) := complex_roots 2 1 (-5) in
  let sum := r₁ + r₂ in
  let product := r₁ * r₂ in
  let new_sum := sum + product in
  let new_product := sum * product in
  ∃ a b c : ℚ, (a = 4) ∧ (b = 12) ∧ (c = 5) ∧ (a * x^2 + b * x + c = 0) = (4 * x^2 + 12 * x + 5 = 0) :=
sorry

end quadratic_roots_equation_l325_325391


namespace parabola_vertex_above_x_axis_l325_325427

theorem parabola_vertex_above_x_axis (k : ℝ) (h : k > 9 / 4) : 
  ∃ y : ℝ, ∀ x : ℝ, y = (x - 3 / 2) ^ 2 + k - 9 / 4 ∧ y > 0 := 
by
  sorry

end parabola_vertex_above_x_axis_l325_325427


namespace find_A_from_eq_l325_325428

theorem find_A_from_eq (A : ℕ) (h : 10 - A = 6) : A = 4 :=
by
  sorry

end find_A_from_eq_l325_325428


namespace cos_210_eq_neg_sqrt3_over_2_l325_325448

theorem cos_210_eq_neg_sqrt3_over_2 :
  Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by sorry

end cos_210_eq_neg_sqrt3_over_2_l325_325448


namespace radius_of_C1_is_sqrt_29_l325_325027

theorem radius_of_C1_is_sqrt_29
(O X Y Z : Type)
(center_O : is_center O C1)
(O_on_C2 : on O C2)
(XY_on_intersection : on X C1 ∧ on X C2 ∧ on Y C1 ∧ on Y C2)
(Z_properties : on Z C2 ∧ ¬ on Z C1)
(dist_XZ : dist X Z = 11)
(dist_OZ : dist O Z = 13)
(dist_YZ : dist Y Z = 5) :
radius C1 = sqrt 29 :=
sorry

end radius_of_C1_is_sqrt_29_l325_325027


namespace simplify_tan_expr_l325_325347

-- Definition of the tangents of 30 degrees and 15 degrees
def tan_30 : ℝ := Real.tan (Real.pi / 6)
def tan_15 : ℝ := Real.tan (Real.pi / 12)

-- Theorem stating that (1 + tan_30) * (1 + tan_15) = 2
theorem simplify_tan_expr : (1 + tan_30) * (1 + tan_15) = 2 :=
by
  sorry

end simplify_tan_expr_l325_325347


namespace find_term_position_l325_325943

theorem find_term_position :
  let sequence := λ n : ℕ, (List.range (3 ^ (n + 1) - 1)).map (λ k, (k + 1) * 2 / (3 ^ (n + 1)))
  let flattened_sequence := List.join (List.range 7).map (sequence)
  let desired_term := (2020 / 2187 : ℚ)
  flattened_sequence.indexOf desired_term + 1 = 1553 :=
by
  sorry

end find_term_position_l325_325943


namespace even_function_implies_f2_eq_neg5_l325_325630

def f (x a : ℝ) : ℝ := (x - a) * (x + 3)

theorem even_function_implies_f2_eq_neg5 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) :
  f 2 a = -5 :=
by
  sorry

end even_function_implies_f2_eq_neg5_l325_325630


namespace triangle_cot_difference_l325_325999

theorem triangle_cot_difference (A B C Q : Type)
  [normed_add_comm_group A] [inner_product_space ℝ A]
  [normed_add_comm_group B] [inner_product_space ℝ B]
  [normed_add_comm_group C] [inner_product_space ℝ C]
  [normed_add_comm_group Q] [inner_product_space ℝ Q]
  (h_angle_bisector_30_deg : ∠BAC = 30)
  (h_angle_bisector_eq : ∃ n: ℝ, AQ = 2n ∧ BQ = CQ) :
  |cot B - cot C| = 1 :=
sorry

end triangle_cot_difference_l325_325999


namespace find_ordered_pair_l325_325512

theorem find_ordered_pair (x y : ℝ)
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x - 2y = (x - 2) + (2y - 2)) :
  (x, y) = (6, 1) :=
by {
  sorry
}

end find_ordered_pair_l325_325512


namespace eq_satisfied_l325_325530

-- Definitions for the problem conditions
def mixed_fraction_val (a b c : ℚ) : ℚ := a + (b / c)
def condition_expr (x : ℚ) : ℚ :=
  ((mixed_fraction_val 2 2 3 + x) / mixed_fraction_val 3 3 4) - 0.4

-- The theorem stating the problem
theorem eq_satisfied : 
  ∃ (x : ℚ), condition_expr x = 32 / 45 ↔ x = 1 + (1 / 2) := sorry

end eq_satisfied_l325_325530


namespace binomial_sum_mod_l325_325846

theorem binomial_sum_mod :
  (∑ k in finset.range (nat.succ (513 / 3)), nat.choose 513 (3 * k)) % 512 = 0 :=
begin
  sorry
end

end binomial_sum_mod_l325_325846


namespace cos_neg_3pi_plus_alpha_l325_325918

/-- Given conditions: 
  1. 𝚌𝚘𝚜(3π/2 + α) = -3/5,
  2. α is an angle in the fourth quadrant,
Prove: cos(-3π + α) = -4/5 -/
theorem cos_neg_3pi_plus_alpha (α : Real) (h1 : Real.cos (3 * Real.pi / 2 + α) = -3 / 5) (h2 : 0 ≤ α ∧ α < 2 * Real.pi ∧ Real.sin α < 0) :
  Real.cos (-3 * Real.pi + α) = -4 / 5 := 
sorry

end cos_neg_3pi_plus_alpha_l325_325918


namespace james_vs_combined_l325_325280

def james_balloons : ℕ := 1222
def amy_balloons : ℕ := 513
def felix_balloons : ℕ := 687
def olivia_balloons : ℕ := 395
def combined_balloons : ℕ := amy_balloons + felix_balloons + olivia_balloons

theorem james_vs_combined :
  1222 = 1222 ∧ 513 = 513 ∧ 687 = 687 ∧ 395 = 395 → combined_balloons - james_balloons = 373 := by
  sorry

end james_vs_combined_l325_325280


namespace sequence_properties_l325_325910

noncomputable def arithmetic_sequence (n : ℕ) : ℕ := 3 * n

noncomputable def geometric_sequence (n : ℕ) : ℕ := 3^(n - 1)

noncomputable def T_n (n : ℕ) : ℕ := ((2 * n - 1) * 3^(n + 1) + 3) / 4

theorem sequence_properties (n : ℕ) :
  (arithmetic_sequence 1 = 3) ∧
  (geometric_sequence 1 = 1) →
  let S_2 := arithmetic_sequence 1 + arithmetic_sequence 2 in
  (geometric_sequence 2 + S_2 = 12) ∧
  (geometric_sequence 2 = S_2 / geometric_sequence 2) →
  (∀ n, arithmetic_sequence n = 3 * n) ∧ 
  (∀ n, geometric_sequence n = 3^(n - 1)) ∧
  (∀ n, ∑ i in finset.range n, (arithmetic_sequence i * geometric_sequence (i + 1)) = T_n n) :=
by
  intros 
  sorry

end sequence_properties_l325_325910


namespace not_divisible_by_5_l325_325524

theorem not_divisible_by_5 (b : ℕ) : 
  (b = 3 ∨ b = 6 ∨ b = 7) → ¬ (5 ∣ (3 * b^3 - 4 * b^2 - 2 * b)) :=
begin
  sorry
end

end not_divisible_by_5_l325_325524


namespace moles_H2O_produced_l325_325871

section chemistry

variable (HCl NaHCO3 NaCl CO2 H2O : Type) [Semiring HCl] [Semiring NaHCO3] [Semiring NaCl] [Semiring CO2] [Semiring H2O]

-- Conditions
axiom balanced_eq : HCl + NaHCO3 = NaCl + CO2 + H2O
axiom moles_HCl : ℕ
axiom moles_NaHCO3 : ℕ

-- Given moles of HCl and NaHCO3
axiom given_moles_HCl : moles_HCl = 3
axiom given_moles_NaHCO3 : moles_NaHCO3 = 3

-- Statement to prove: the reaction produces 3 moles of H2O
theorem moles_H2O_produced : (moles_HCl = 3 ∧ moles_NaHCO3 = 3) → moles_H2O = 3 :=
by
  sorry

end chemistry

end moles_H2O_produced_l325_325871


namespace find_base_k_l325_325542

theorem find_base_k :
  ∃ k : ℕ, (k > 0) ∧ (∃ (n : ℕ), 0.\overline{45}_k = 0.\overline{45}_k) →
    (k = 16) :=
begin
  sorry
end

end find_base_k_l325_325542


namespace investment_months_after_A_l325_325457

variable (x : ℕ) (months : ℕ)
variable (A_invest B_invest : ℝ)
variable (total_profit A_profit B_profit : ℝ)

-- Given conditions
def conditions :=
  A_invest = 400 ∧
  B_invest = 200 ∧
  total_profit = 100 ∧
  A_profit = 80 ∧
  B_profit = total_profit - A_profit ∧
  (A_profit / B_profit) = (A_invest * 12) / (B_invest * (12 - x))

-- Define the question as a goal to prove
theorem investment_months_after_A (h : conditions x 400 200 100 80) : x = 6 := by
  sorry

end investment_months_after_A_l325_325457


namespace circle_regions_l325_325276

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l325_325276


namespace find_length_of_segment_AC_l325_325314

noncomputable def length_segment_AC : ℝ :=
  let AB := BC 
  let AM := 7
  let MB := 3
  let angleBMC := 60
  17

theorem find_length_of_segment_AC :
  ∃ (AC : ℝ), AC = length_segment_AC :=
by {
  -- The proof goes here.
  sorry
}

end find_length_of_segment_AC_l325_325314


namespace value_of_x_plus_inv_x_l325_325586

theorem value_of_x_plus_inv_x (x : ℝ) (hx : x ≠ 0) (t : ℝ) (ht : t = x^2 + (1 / x)^2) : x + (1 / x) = 5 :=
by
  have ht_val : t = 23 := by
    rw [ht] -- assuming t = 23 by condition
    sorry -- proof continuation placeholder

  -- introduce y and relate it to t
  let y := x + (1 / x)

  -- express t in terms of y and handle the algebra:
  have t_expr : t = y^2 - 2 := by
    sorry -- proof continuation placeholder

  -- show that y^2 = 25 and therefore y = 5 as the only valid solution:
  have y_val : y = 5 := by
    sorry -- proof continuation placeholder

  -- hence, the required value is found:
  exact y_val

end value_of_x_plus_inv_x_l325_325586


namespace area_of_triangle_PQR_l325_325416

open Real

def point := ℝ × ℝ

def line (slope : ℝ) (p : point) : (ℝ → point) :=
  λ x, (x, slope * (x - p.1) + p.2)

lemma area_of_triangle {P Q R : point} :
  area P Q R = half * (|Q.1 - R.1|) * |P.2 - Q.2| :=
by sorry

theorem area_of_triangle_PQR :
  let P : point := (2, 5)
  let Q : point := (1 / 3, 0)
  let R : point := (-3, 0)
  area P Q R = 25 / 3 :=
by sorry

end area_of_triangle_PQR_l325_325416


namespace fixed_point_of_line_l325_325379

theorem fixed_point_of_line :
  let line : ℝ → ℝ × ℝ → ℝ := λ λ (x,y), (2 + λ) * x + (λ - 1) * y - 2 * λ - 10
  let L1 : ℝ × ℝ → ℝ := λ (x,y), 2 * x - y
  let L2 : ℝ × ℝ → ℝ := λ (x,y), x + 2
  let intersection_point := (1, 1)

  (∀ (x y : ℝ), L1 (x, y) = 0 ∧ L2 (x, y) = 0 → (x, y) = intersection_point) →
  ∀ λ, line λ intersection_point = 0 :=
  
sorry

end fixed_point_of_line_l325_325379


namespace scientific_notation_15_7_trillion_l325_325647

theorem scientific_notation_15_7_trillion :
  ∃ n : ℝ, n = 15.7 * 10^12 ∧ n = 1.57 * 10^13 :=
by
  sorry

end scientific_notation_15_7_trillion_l325_325647


namespace circle_regions_l325_325275

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l325_325275


namespace inequality_system_solution_l325_325365

theorem inequality_system_solution (x : ℝ) :
  (x + 7) / 3 ≤ x + 3 ∧ 2 * (x + 1) < x + 3 ↔ -1 ≤ x ∧ x < 1 :=
by
  sorry

end inequality_system_solution_l325_325365


namespace Kayla_points_on_first_level_l325_325664

theorem Kayla_points_on_first_level
(points_2 : ℕ) (points_3 : ℕ) (points_4 : ℕ) (points_5 : ℕ) (points_6 : ℕ)
(h2 : points_2 = 3) (h3 : points_3 = 5) (h4 : points_4 = 8) (h5 : points_5 = 12) (h6 : points_6 = 17) :
  ∃ (points_1 : ℕ), 
    (points_3 - points_2 = 2) ∧ 
    (points_4 - points_3 = 3) ∧ 
    (points_5 - points_4 = 4) ∧ 
    (points_6 - points_5 = 5) ∧ 
    (points_2 - points_1 = 1) ∧ 
    points_1 = 2 :=
by
  use 2
  repeat { split }
  sorry

end Kayla_points_on_first_level_l325_325664


namespace line_slope_l325_325644

-- Define the points and the line properties
def point1 := (5, 1)
def point2 := (5, 5)
def origin := (0, 0)

-- Define the condition that a line passes through the origin and these points
def on_line (p : ℝ × ℝ) (m : ℝ) : Prop := ∃ c, p.2 = m * p.1 + c

-- Define the slope function
def slope (p1 p2 : ℝ × ℝ) : ℝ := 
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem line_slope : slope origin point1 = 1 / 5 :=
by
  -- The proof will be added here
  sorry

end line_slope_l325_325644


namespace brogan_total_red_apples_l325_325834

def red_apples (total_apples percentage_red : ℕ) : ℕ :=
  (total_apples * percentage_red) / 100

theorem brogan_total_red_apples :
  red_apples 20 40 + red_apples 20 50 = 18 :=
by
  sorry

end brogan_total_red_apples_l325_325834


namespace fraction_exponentiation_l325_325844

theorem fraction_exponentiation :
  (⟨1/3⟩ : ℝ) ^ 5 = (⟨1/243⟩ : ℝ) :=
by
  sorry

end fraction_exponentiation_l325_325844


namespace beth_total_crayons_l325_325832

theorem beth_total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) 
  (h1 : packs = 8) (h2 : crayons_per_pack = 20) (h3 : extra_crayons = 15) :
  packs * crayons_per_pack + extra_crayons = 175 :=
by
  sorry

end beth_total_crayons_l325_325832


namespace probability_of_quarter_or_dime_l325_325805

theorem probability_of_quarter_or_dime :
  let value_of_quarters := 5.00
  let value_of_dimes := 6.00
  let value_of_nickels := 2.00
  let quarter_value := 0.25
  let dime_value := 0.10
  let nickel_value := 0.05
  let num_quarters := value_of_quarters / quarter_value
  let num_dimes := value_of_dimes / dime_value
  let num_nickels := value_of_nickels / nickel_value
  let total_coins := num_quarters + num_dimes + num_nickels
  let probability := (num_quarters + num_dimes) / total_coins
  in probability = 2 / 3 :=
by
  sorry

end probability_of_quarter_or_dime_l325_325805


namespace angle_A_eq_60_l325_325655

-- Define triangle vertices as points in 2D plane.
variables {A B C D E : Type} 

-- Assume the conditions given in the problem.
variables (h1 : ∃ (u : B = C), AB = AC) -- AB = AC and notation
variables (h2 : (bisector BD (angle ABC)) -- BD bisects angle ABC
variables (h3 : DE = BD ∧ (bisector DE (angle BDC))) -- DE = BD and DE bisects angle BDC
variables (h4 : BD = BC) -- BD = BC

-- Define angle function (not considering specific implementation).
noncomputable def angle (x y z : A) : ℝ := sorry

-- The main theorem: Prove that angle A is 60 degrees given the above conditions.
theorem angle_A_eq_60 : 
  (angle A B C + angle A C B + angle B C A = 180) →
  (angle_B = angle_C) →
  (h4) →
  (angle C B D = 1/2 * angle_A B C) →
  (angle D B E = 60) → 
  (angle A B C = 120) →
  (∠A = 180 - ∠AB C = 180 - 120 = 60)
  (angle A = 60) :=
sorry -- proof omitted

end angle_A_eq_60_l325_325655


namespace nonagon_diagonals_l325_325951

theorem nonagon_diagonals (n : ℕ) (h1 : n = 9) : (n * (n - 3)) / 2 = 27 := by
  sorry

end nonagon_diagonals_l325_325951


namespace shortest_distance_from_curve_to_line_l325_325396

noncomputable def shortest_distance_to_line (x : ℝ) : ℝ :=
  let y := Real.exp (2 * x) in
  let distance := (abs (2 * x - y - 4)) / Real.sqrt (2^2 + (-1)^2) in
  distance

theorem shortest_distance_from_curve_to_line :
  ∃ x : ℝ, shortest_distance_to_line x = Real.sqrt 5 :=
sorry

end shortest_distance_from_curve_to_line_l325_325396


namespace solve_problem_l325_325179

noncomputable def ratio_of_external_bisector (A B C D : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (distance : A → B → ℝ) (triangle_ABC : Prop) (ratio_AC_CB : distance A C / distance C B = 2 / 3)
  (external_bisector_C_D : Prop) (B_between_A_D : Prop) : Prop :=
  distance D A / distance A B = 2 / 5

axiom AC : Type
axiom CB : Type
axiom A : Type
axiom B : Type
axiom C : Type
axiom D : Type
axiom distance : A → B → ℝ
axiom triangle_ABC : Prop := True
axiom ratio_AC_CB : distance A C / distance C B = 2 / 3
axiom external_bisector_C_D : Prop := True
axiom B_between_A_D : Prop := True

theorem solve_problem : 
  ratio_of_external_bisector A B C D distance triangle_ABC ratio_AC_CB external_bisector_C_D B_between_A_D := 
  sorry

end solve_problem_l325_325179


namespace count_positive_area_triangles_l325_325144

-- Define the grid size
def grid_size : ℕ := 6

-- Defining the main theorem
theorem count_positive_area_triangles :
  (set.univ.powerset.to_finset.filter (λ s : finset (ℤ × ℤ), s.card = 3 ∧ ¬collinear s)).card = 6628 :=
sorry

end count_positive_area_triangles_l325_325144


namespace count_distinct_reals_a_with_integer_roots_l325_325888

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l325_325888


namespace arithmetic_sequence_zero_term_l325_325565

theorem arithmetic_sequence_zero_term (a : ℕ → ℤ) (d : ℤ) (h : d ≠ 0) 
  (h_seq : ∀ n, a n = a 1 + (n-1) * d)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  ∃ n, a n = 0 ∧ n = 5 :=
by { sorry }

end arithmetic_sequence_zero_term_l325_325565


namespace area_of_region_inside_circle_outside_rectangle_l325_325193

theorem area_of_region_inside_circle_outside_rectangle
  (EF FH : ℝ)
  (hEF : EF = 6)
  (hFH : FH = 5)
  (r : ℝ)
  (h_radius : r = (EF^2 + FH^2).sqrt) :
  π * r^2 - EF * FH = 61 * π - 30 :=
by
  sorry

end area_of_region_inside_circle_outside_rectangle_l325_325193


namespace ellipse_properties_l325_325912

theorem ellipse_properties (a b : ℝ) (C : set (ℝ × ℝ))
  (h1 : C = {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / a^2) + (y^2 / b^2) = 1})
  (h2 : 0 < b ∧ b < a)
  (h3 : 2 * a = 8)
  (h4 : ∀ x y : ℝ, (x, y) ∈ C → (y / (x + 4)) * (y / (x - 4)) = -3 / 4) :
  (C = {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / 16) + (y^2 / 12) = 1}) ∧
  ∀ M : ℝ × ℝ, M = (0, 2) →
    ∀ P Q : ℝ × ℝ, (P.1, P.2) ∈ C → (Q.1, Q.2) ∈ C →
      M.2 ≠ P.2 →
      let OP := P.1 * Q.1 + P.2 * Q.2 in
      let MP := (P.1 - M.1) * (Q.1 - M.1) + (P.2 - M.2) * (Q.2 - M.2) in
      -20 ≤ OP + MP ∧ OP + MP ≤ -52 / 3 :=
sorry

end ellipse_properties_l325_325912


namespace persons_in_second_group_l325_325784

-- Conditions from the problem
def num_persons_first_group : ℕ := 39
def days_first_group : ℕ := 12
def hours_per_day_first_group : ℕ := 5

def num_persons_second_group : ℕ
def days_second_group : ℕ := 26
def hours_per_day_second_group : ℕ := 3

-- Correct answer
def correct_answer : ℕ := 30

-- Total man-hours calculation for both groups
def man_hours (persons : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  persons * days * hours_per_day

-- Hypothesis: the total man-hours for both groups are equal
def total_man_hours_first_group : ℕ :=
  man_hours num_persons_first_group days_first_group hours_per_day_first_group

def total_man_hours_second_group : ℕ :=
  man_hours num_persons_second_group days_second_group hours_per_day_second_group

theorem persons_in_second_group (x : ℕ) (h : total_man_hours_first_group = total_man_hours_second_group x) : x = correct_answer :=
by {
  sorry
}

end persons_in_second_group_l325_325784


namespace smallest_positive_period_and_monotonic_decrease_intervals_range_of_b_l325_325589

noncomputable def f (x : ℝ) : ℝ :=
  cos x * (sqrt 3 * sin x - cos x)

theorem smallest_positive_period_and_monotonic_decrease_intervals :
  (∀ x, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x ∈ set.Icc (k * π + π / 3) (k * π + 5 * π / 6), 
    f (x) ≤ f (x + ε) ∨ f (x) ≥ f (x - ε)) :=
by
  sorry

theorem range_of_b 
  (a c b : ℝ) (A B C : ℝ) (h : f B = 1 / 2) (h_triangle : A + B + C = π)
  (h_sides : a + c = 1) (ha : 0 < a ∧ a < 1) :
  1 / 2 ≤ b ∧ b < 1 :=
by
  sorry

end smallest_positive_period_and_monotonic_decrease_intervals_range_of_b_l325_325589


namespace sum_of_solutions_l325_325058

theorem sum_of_solutions (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 225) :
  ∑ x in {x : ℝ | x^2 + y^2 = 225}, x = 0 :=
by
  -- Proof to be filled
  sorry

end sum_of_solutions_l325_325058


namespace exists_matching_mittens_l325_325190

structure Child (Color : Type) :=
(left_mitten : Color)
(right_mitten : Color)

def initial_children {Color : Type} (n : ℕ) (diff_colors : List Color) (h_colors : diff_colors.length = n) : List (Child Color) :=
sorry

def pair_children (pairs : List (Child ℕ × Child ℕ)) : List (Child ℕ × Child ℕ) :=
sorry

def exchange_left (pair : Child ℕ × Child ℕ) : Child ℕ × Child ℕ :=
sorry

def exchange_right (pair : Child ℕ × Child ℕ) : Child ℕ × Child ℕ :=
sorry

theorem exists_matching_mittens : 
  ∀ (Color : Type) (n : ℕ) (diff_colors : List Color) (h_colors: diff_colors.length = n) (children : List (Child Color)),
  n = 26 →
  (∀ c ∈ children, (∃ color, c.left_mitten = color ∧ c.right_mitten = color)) →
  (∀ c1 c2 ∈ children, c1 ≠ c2 → (c1.left_mitten ≠ c2.left_mitten ∧ c1.right_mitten ≠ c2.right_mitten)) →
  (∃ pairs1 pairs2 pairs3 : List (Child Color × Child Color),
   pair_children pairs1 = children.length/2 ∧
   pair_children pairs2 = children.length/2 ∧
   pair_children pairs3 = children.length/2 ∧
   ∀ (pair1 ∈ pairs1), (∀ (c ∈ [fst pair1, snd pair1]), ∃ cl cr, (exchange_left pair1) = (cl, cr)) ∧
   ∀ (pair2 ∈ pairs2), (∀ (c ∈ [fst pair2, snd pair2]), ∃ cl cr, (exchange_right pair2) = (cl, cr)) ∧
   ∀ (pair3 ∈ pairs3), (∀ (c ∈ [fst pair3, snd pair3]), ∃ cl cr, (c.left_mitten = cl ∧ c.right_mitten = cr))) →
  ∃ c ∈ children, c.left_mitten = c.right_mitten :=
by {
  sorry
}

end exists_matching_mittens_l325_325190


namespace eq_or_neg_eq_of_eq_frac_l325_325544

theorem eq_or_neg_eq_of_eq_frac (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h : a^2 + b^3 / a = b^2 + a^3 / b) :
  a = b ∨ a = -b :=
by
  sorry

end eq_or_neg_eq_of_eq_frac_l325_325544


namespace max_distance_c1_to_c2_l325_325651

-- Definitions for the conditions
def parametric_curves_c1 (θ : ℝ) :=
  let x := sqrt 2 * cos θ
  let y := sqrt 3 * sin θ
  (x, y)

def parametric_curves_c2 (t : ℝ) :=
  let x := t
  let y := -t + 4 * sqrt 5
  (x, y)

-- The general equations to be proved
def curve_equation_c1 (x y : ℝ) :=
  x^2 / 2 + y^2 / 3 = 1

def curve_equation_c2 (x y : ℝ) :=
  x + y - 4 * sqrt 5 = 0

-- Defining the distance formula
def distance_to_curve_c2 (P : ℝ × ℝ) : ℝ :=
  abs (P.1 + P.2 - 4 * sqrt 5) / sqrt 2

-- Proof statement
theorem max_distance_c1_to_c2 :
  (∀ θ : ℝ, let P := parametric_curves_c1 θ in curve_equation_c1 P.1 P.2) ∧
  (∀ t : ℝ, let Q := parametric_curves_c2 t in curve_equation_c2 Q.1 Q.2) ∧
  (∃ θ : ℝ, let P := parametric_curves_c1 θ in distance_to_curve_c2 P = 5 * sqrt 10 / 2) :=
by
  -- The proof would go here
  sorry

end max_distance_c1_to_c2_l325_325651


namespace fraction_of_tips_l325_325817

variable (S T : ℝ) -- assuming S is salary and T is tips
variable (h : T / (S + T) = 0.7142857142857143)

/-- 
If the fraction of the waiter's income from tips is 0.7142857142857143,
then the fraction of his salary that were his tips is 2.5.
-/
theorem fraction_of_tips (h : T / (S + T) = 0.7142857142857143) : T / S = 2.5 :=
sorry

end fraction_of_tips_l325_325817


namespace no_HCl_formed_l325_325872

-- Definitions
def NaCl_moles : Nat := 3
def HNO3_moles : Nat := 3
def HCl_moles : Nat := 0

-- Hypothetical reaction context
-- if the reaction would produce HCl
axiom hypothetical_reaction : (NaCl_moles = 3) → (HNO3_moles = 3) → (∃ h : Nat, h = 3)

-- Proof under normal conditions that no HCl is formed
theorem no_HCl_formed : (NaCl_moles = 3) → (HNO3_moles = 3) → HCl_moles = 0 := by
  intros hNaCl hHNO3
  sorry

end no_HCl_formed_l325_325872


namespace tangent_line_at_1_2_is_2x_minus_y_eq_0_l325_325728

noncomputable def f (x : ℝ) : ℝ := x^2 + 1

theorem tangent_line_at_1_2_is_2x_minus_y_eq_0 :
  ∀ (x y : ℝ), y = f x → (x, y) = (1, 2) → 2 * x - y = 0 :=
by
  intros x y hyp1 hyp2
  subst hyp1
  subst hyp2
  sorry

end tangent_line_at_1_2_is_2x_minus_y_eq_0_l325_325728


namespace redder_permutations_no_palindrome_l325_325613

-- Define the word REDDER and the concept of palindromic substring.
def is_palindromic (s : List Char) : Bool :=
  s = s.reverse

def valid_permutation (perm : List Char) : Bool :=
  ∀ (n : ℕ), n < perm.length - 1 → ¬is_palindromic [perm[n], perm[n+1]] ∧
             (n < perm.length - 2 → ¬is_palindromic [perm[n], perm[n+1], perm[n+2]])

def permutations (l : List Char) : List (List Char) :=
  -- function generating all distinct permutations of l
  List.perm l

-- Statement of the problem
theorem redder_permutations_no_palindrome : 
  (permutations ['R', 'E', 'D', 'D', 'E', 'R']).count (λ perm, valid_permutation perm) = 6 :=
sorry

end redder_permutations_no_palindrome_l325_325613


namespace correct_divisor_l325_325983

theorem correct_divisor (dividend incorrect_divisor quotient correct_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 63) 
  (h2 : quotient = 24) 
  (h3 : correct_quotient = 42) 
  (h4 : dividend = incorrect_divisor * quotient) 
  (h5 : dividend / correct_divisor = correct_quotient) : 
  correct_divisor = 36 := 
by 
  sorry

end correct_divisor_l325_325983


namespace solve_for_x_l325_325168

theorem solve_for_x (x : ℝ) (h : sqrt (3/x + 3) = 2/3) : x = -27/23 :=
by
  sorry

end solve_for_x_l325_325168


namespace find_m_condition_l325_325963

theorem find_m_condition (m : ℕ) (h : 9^4 = 3^(2*m)) : m = 4 := by
  sorry

end find_m_condition_l325_325963


namespace simplify_tan_expression_l325_325357

theorem simplify_tan_expression :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 := 
by 
  -- Angle addition formula for tangent
  have h : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12), 
  from by rw [Real.tan_add]; exact Real.tan_pi_div_four,
  -- Given that tan 45° = 1
  have h1 : Real.tan (Real.pi / 4) = 1, from Real.tan_pi_div_four,
  -- Derive the known value
  rw [Real.tan_pi_div_four, h] at h1,
  -- Simplify using the derived value
  suffices : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 
             1 + Real.tan (Real.pi / 6) + Real.tan (Real.pi / 12) + Real.tan (Real.pi / 6) * Real.tan (Real.pi / 12), 
  from by rw this; simp [←h1],
  sorry

end simplify_tan_expression_l325_325357


namespace total_number_of_notes_l325_325439

theorem total_number_of_notes (x : ℕ) (h : 192 = 1 * x + 5 * x + 10 * x) : 3 * x = 36 := by
  have h1 : 16 * x = 192 := by linarith
  have h2 : x = 192 / 16 := by linarith
  have hx : x = 12 := by norm_num
  rw [hx]
  norm_num

end total_number_of_notes_l325_325439


namespace complex_square_l325_325840

theorem complex_square (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end complex_square_l325_325840


namespace ellipse_eq_l325_325375

noncomputable def ellipse_equation (x y : ℝ) :=
  (x^2 / 16) + (y^2 / 12) = 1

theorem ellipse_eq :
  let f1 := (-2, 0), f2 := (2, 0)
  ∃ (x y : ℝ), (x, y) = (2, 3) → ellipse_equation x y :=
by 
  have f1 : (-2 : ℝ, 0 : ℝ) := (-2, 0)
  have f2 : (2 : ℝ, 0 : ℝ) := (2, 0)
  use (2 : ℝ, 3 : ℝ)
  intros,
  simp [ellipse_equation],
  sorry

end ellipse_eq_l325_325375


namespace magnitude_a_minus_2b_l325_325299

noncomputable def a : ℝ × ℝ × ℝ := (3, 5, -4)
noncomputable def b : ℝ × ℝ × ℝ := (2, -1, -2)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem magnitude_a_minus_2b : magnitude (a.1 - 2 * b.1, a.2 - 2 * b.2, a.3 - 2 * b.3) = 5 * Real.sqrt 2 := by
  sorry

end magnitude_a_minus_2b_l325_325299


namespace correct_misread_number_l325_325372

theorem correct_misread_number (s : List ℕ) (wrong_avg correct_avg n wrong_num correct_num : ℕ) 
  (h1 : s.length = 10) 
  (h2 : (s.sum) / n = wrong_avg) 
  (h3 : wrong_num = 26) 
  (h4 : correct_avg = 16) 
  (h5 : n = 10) 
  : correct_num = 36 :=
sorry

end correct_misread_number_l325_325372


namespace find_x_l325_325627

theorem find_x (p q : ℕ) (h1 : 1 < p) (h2 : 1 < q) (h3 : 17 * (p + 1) = (14 * (q + 1))) (h4 : p + q = 40) : 
    x = 14 := 
by
  sorry

end find_x_l325_325627


namespace part_I_part_II_part_III_l325_325933

open Real

noncomputable def f (x : ℝ) : ℝ := (1 + log (x + 1)) / x

-- Statement for Part I
theorem part_I (x : ℝ) (h : 0 < x) : deriv (f) x < 0 := 
sorry

-- Statement for Part II
theorem part_II (x : ℝ) (h : 0 < x) : 
  ∃ (k : ℕ), k = 3 ∧ (f x > (k : ℝ) / (x + 1)) := 
sorry

-- Statement for Part III
theorem part_III (n : ℕ) : 
  (∏ i in finset.range n, (1 + i * (i + 1)) : ℝ) > exp (2 * n - 3) := 
sorry

end part_I_part_II_part_III_l325_325933


namespace simplify_tan_expr_l325_325351

-- Definition of the tangents of 30 degrees and 15 degrees
def tan_30 : ℝ := Real.tan (Real.pi / 6)
def tan_15 : ℝ := Real.tan (Real.pi / 12)

-- Theorem stating that (1 + tan_30) * (1 + tan_15) = 2
theorem simplify_tan_expr : (1 + tan_30) * (1 + tan_15) = 2 :=
by
  sorry

end simplify_tan_expr_l325_325351


namespace find_dianas_uniform_number_l325_325548

noncomputable def is_two_digit_prime (n : ℕ) : Prop :=
  nat.prime n ∧ n >= 10 ∧ n < 100

def ashley_uniform_number := 23
def bethany_uniform_number := 17
def caitlin_uniform_number := 19
def diana_uniform_number := 19

axiom ashley_condition (a c b_birthday : ℕ) : a + c = b_birthday
axiom bethany_condition (b d c_birthday : ℕ) : b + d = c_birthday
axiom caitlin_condition (c d todays_date : ℕ) : c + d = 2 * todays_date
axiom product_condition (a d : ℕ) : a * d = 437

theorem find_dianas_uniform_number 
  (a b c d : ℕ)
  (a_prime : is_two_digit_prime a)
  (b_prime : is_two_digit_prime b)
  (c_prime : is_two_digit_prime c)
  (d_prime : is_two_digit_prime d)
  (ashley_cond : ashley_condition a c (b+7))
  (bethany_cond : bethany_condition b d (c+12))
  (caitlin_cond : caitlin_condition c d 18)
  (prod_cond : product_condition a d) : 
  d = 19 :=
by 
  sorry

end find_dianas_uniform_number_l325_325548


namespace min_value_of_function_l325_325076

theorem min_value_of_function (x : Real) (hx : x * log 3 2 ≥ -1) :
  ∃ (m : Real), m = (4^x - 2^(x+1) - 3) ∧ (∀ y : Real, y ≥ (4^x - 2^(x+1) - 3) → y ≥ m) ∧ m = -4 := by
  sorry

end min_value_of_function_l325_325076


namespace projection_a_on_b_l325_325946

variables {ℝ : Type*} [NormedSpace ℝ ℝ]
variables (a b : ℝ) -- Let's assume these represent vectors of some appropriate type with norm defined.

-- Conditions:
def norm_a_one : ∥a∥ = 1 := sorry
def norm_b_sqrt3 : ∥b∥ = sqrt 3 := sorry
def norm_a_add_b_sqrt2 : ∥a + b∥ = sqrt 2 := sorry

-- Proof problem statement:
theorem projection_a_on_b :
  let θ := acos (inner_product a b / (∥a∥ * ∥b∥))
  in (∥a∥ * cos θ ⟂ b) = - sqrt 3 / 3 :=
by 
  sorry

end projection_a_on_b_l325_325946


namespace count_four_digit_even_numbers_l325_325056

theorem count_four_digit_even_numbers : 
  let digits := [0, 1, 2, 3, 4, 5]
  let four_digit_evens := 
    {n | let d := n.digits 10 in 
         d.length = 4 ∧ 
         n % 2 = 0 ∧ 
         d.nodup ∧ 
         ∀ x ∈ d, x ∈ digits}
  four_digit_evens.card = 156 := 
sorry

end count_four_digit_even_numbers_l325_325056


namespace ship_catch_up_direction_l325_325334

theorem ship_catch_up_direction
  (a : ℝ) -- distance between ships
  (α : ℝ) -- 60 degree, direction
  (β : ℝ) -- 120 degree, angle B
  (s_A : ℝ) -- speed of Ship A
  (s_B : ℝ) -- speed of Ship B
  (h_A_speed : s_A = real.sqrt 3 * s_B)
  (h_angle : α = 60)
  (h_angle_B : β = 120)
  (x : ℝ) -- time to catch up
  (BC : ℝ) -- distance BC
  (AC : ℝ) -- distance AC
  (h_BC : BC = x)
  (h_AC : AC = real.sqrt 3 * x)
  : ∃ θ : ℝ, θ = 30 :=
by 
  sorry

end ship_catch_up_direction_l325_325334


namespace find_b2_l325_325738

noncomputable def sequence_b (n : ℕ) : ℕ := by
  if n = 1 then exact 41
  else if n = 10 then exact 101
  else if n >= 3 then exact 101
  else exact n

theorem find_b2 (b : ℕ → ℕ)  (h₁ : b 1 = 41) (h₁₀ : b 10 = 101)
  (h_mean : ∀ n, n ≥ 3 → b n = (b (nat.pred n))): 
  b 2 = 161 := sorry

end find_b2_l325_325738


namespace bus_distance_covered_l325_325404

theorem bus_distance_covered (speedTrain speedCar speedBus distanceBus : ℝ) (h1 : speedTrain / speedCar = 16 / 15)
                            (h2 : speedBus = (3 / 4) * speedTrain) (h3 : 450 / 6 = speedCar) (h4 : distanceBus = 8 * speedBus) :
                            distanceBus = 480 :=
by
  sorry

end bus_distance_covered_l325_325404


namespace total_amount_spent_l325_325489

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = (1/2) * B
def condition2 : Prop := B = D + 15

-- Proof statement
theorem total_amount_spent (h1 : condition1 B D) (h2 : condition2 B D) : B + D = 45 := by
  sorry

end total_amount_spent_l325_325489


namespace evaluate_expression_l325_325519

theorem evaluate_expression : 
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12 - 13 + 14 - 15 + 16 - 17 + 18 - 19 + 20)
  = 10 / 11 := 
by
  sorry

end evaluate_expression_l325_325519


namespace prove_total_bill_is_correct_l325_325023

noncomputable def totalCostAfterDiscounts : ℝ :=
  let adultsMealsCost := 8 * 12
  let teenagersMealsCost := 4 * 10
  let childrenMealsCost := 3 * 7
  let adultsSodasCost := 8 * 3.5
  let teenagersSodasCost := 4 * 3.5
  let childrenSodasCost := 3 * 1.8
  let appetizersCost := 4 * 8
  let dessertsCost := 5 * 5

  let subtotal := adultsMealsCost + teenagersMealsCost + childrenMealsCost +
                  adultsSodasCost + teenagersSodasCost + childrenSodasCost +
                  appetizersCost + dessertsCost

  let discountAdultsMeals := 0.10 * adultsMealsCost
  let discountDesserts := 5
  let discountChildrenMealsAndSodas := 0.15 * (childrenMealsCost + childrenSodasCost)

  let adjustedSubtotal := subtotal - discountAdultsMeals - discountDesserts - discountChildrenMealsAndSodas

  let additionalDiscount := if subtotal > 200 then 0.05 * adjustedSubtotal else 0
  let total := adjustedSubtotal - additionalDiscount

  total

theorem prove_total_bill_is_correct : totalCostAfterDiscounts = 230.70 :=
by sorry

end prove_total_bill_is_correct_l325_325023


namespace problem_a0_a6_l325_325903

theorem problem_a0_a6 :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
  (2 - 0)*(2 * 0 + 1)^5 = a₀ →
  (((2 * x + 1)^5).expand' (2 - x)).coeff 0 = a₀ →
  (((2 * x + 1)^5).expand' (2 - x)).coeff 6 = a₆ →
  a₀ + a₆ = -30 :=
by 
  intros a₀ a₁ a₂ a₃ a₄ a₅ a₆ h₀ h₆,
  sorry

end problem_a0_a6_l325_325903


namespace sqrt_defined_range_l325_325619

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by sorry

end sqrt_defined_range_l325_325619


namespace jane_remaining_hours_l325_325434

def time_to_complete_task (time_jane time_roy time_mary total_work: ℕ) : ℕ :=
  let rate_jane := 1 / (time_jane:ℚ)
  let rate_roy := 1 / (time_roy:ℚ)
  let rate_mary := 1 / (time_mary:ℚ)
  let work_together_2hrs := 2 * (rate_jane + rate_roy + rate_mary)
  let work_1hr_after_roy_leaves := rate_jane + rate_mary
  let completed_work := work_together_2hrs + work_1hr_after_roy_leaves
  let remaining_work := total_work - completed_work
  let time_for_jane := remaining_work / rate_jane
  time_for_jane

theorem jane_remaining_hours (total_work: ℚ) : time_to_complete_task 4 5 6 1 = 1.4 :=
sorry

end jane_remaining_hours_l325_325434


namespace locus_of_midpoint_l325_325475

noncomputable def circle_center (O : Point) (r : Float) := ∀ (P : Point), dist O P = r

def is_tangent (P E F : Point) (O : Point) :=
  -- Placeholder for the tangent condition from P to circle centered at O to points E and F
  sorry

def midpoint (E F M : Point) :=
  M.x = (E.x + F.x) / 2 ∧ M.y = (E.y + F.y) / 2

def perpendicular (A O D : Point) :=
  -- Placeholder for the perpendicular condition
  sorry

theorem locus_of_midpoint :
  ∀ (O P E F M D : Point), 
    circle_center O 1 → -- Assuming radius r = 1 for simplicity
    is_tangent P E F O → 
    midpoint E F M → 
    perpendicular A O D → 
    P moves along the line →
    locus M = circle_with_diameter (dist O D) := 
begin
  sorry
end

end locus_of_midpoint_l325_325475


namespace compare_f_l325_325934

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem compare_f (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 = 0) : 
  f x1 < f x2 :=
by sorry

end compare_f_l325_325934


namespace equilateral_triangles_similar_l325_325826

theorem equilateral_triangles_similar : 
  ∀ (Δ1 Δ2 : Triangle), 
    (is_equilateral Δ1 → is_equilateral Δ2 → similar Δ1 Δ2) := 
by
  sorry

end equilateral_triangles_similar_l325_325826


namespace konigsberg_bridges_l325_325418

theorem konigsberg_bridges:
  ∀ (G : SimpleGraph V) [Fintype V] [DecidableRel G.Adj],
  (∃ v : V, G.degree v % 2 = 1) →
  (∃ v : V, ∃ e : G.edgeSet, e ∈ G.edgeSet ∧ G.incidence e v) →
  ¬(∃ P : G.Walk v v, P.edges.nodup ∧ ∀ e : G.edgeSet, e ∈ P.edges) :=
by
  sorry

end konigsberg_bridges_l325_325418


namespace binary_110101_is_53_l325_325850

def binary_to_decimal (n : Nat) : Nat :=
  let digits := [1, 1, 0, 1, 0, 1]  -- Define binary digits from the problem statement
  digits.reverse.foldr (λ d (acc, pow) => (acc + d * (2^pow), pow + 1)) (0, 0) |>.fst

theorem binary_110101_is_53 : binary_to_decimal 110101 = 53 := by
  sorry

end binary_110101_is_53_l325_325850


namespace cesaroSum_extended_l325_325065

variable (B : Fin 150 → ℝ)
variable (T : Fin 150 → ℝ)
variable (k : ℕ)
variable hc : (∑ i in Finset.range 150, ∑ j in Finset.range (i + 1), B j) / 150 = 500

def cesaroSum (b : Fin n → ℝ) := (∑ i in Finset.range n, ∑ j in Finset.range (i + 1), b j) / n

theorem cesaroSum_extended :
  (cesaroSum ((λ i, if i = 0 then 2 else B ⟨i - 1, sorry⟩) : Fin (150+1) → ℝ) = 499 :=
begin
  -- Intuition: derive it step by step or add assumptions/parameters for sums.
  sorry
end

end cesaroSum_extended_l325_325065


namespace quadratic_nonneg_iff_l325_325555

variable {a b c : ℝ}

theorem quadratic_nonneg_iff :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4 * a * c ≤ 0) :=
by sorry

end quadratic_nonneg_iff_l325_325555


namespace minimum_f2020_minimum_f2020_mul_f2021_l325_325668

-- Definitions for the sequence and conditions need to be stated explicitly.
def sequence_conditions (z : ℕ → ℂ) : Prop :=
  (∀ (n : ℕ), n % 2 = 1 → ∃ r ∈ ℝ, z n = r) ∧
  (∀ (n : ℕ), n % 2 = 0 → ∃ r ∈ ℝ, z n = r * complex.I) ∧
  (∀ (k : ℕ), k > 0 → complex.abs (z k * z (k + 1)) = 2^k)

def f (z : ℕ → ℂ) (n : ℕ) : ℝ :=
  complex.abs (∑ i in finset.range n, z i)

-- The minimum of f_2020
theorem minimum_f2020 (z : ℕ → ℂ) (h : sequence_conditions z) : 
  ∃ c, c = 2^1010 - 1 ∧ ∀ n, f z n ≥ c := by
  sorry

-- The minimum of f_2020 * f_2021
theorem minimum_f2020_mul_f2021 (z : ℕ → ℂ) (h : sequence_conditions z) : 
  ∃ c, c = 2^1011 ∧ ∀ n, (f z n) * (f z (n+1)) ≥ c := by
  sorry

end minimum_f2020_minimum_f2020_mul_f2021_l325_325668


namespace equal_partition_of_weights_l325_325988

theorem equal_partition_of_weights 
  (weights : Fin 2009 → ℕ) 
  (h1 : ∀ i : Fin 2008, (weights i + 1 = weights (i + 1)) ∨ (weights i = weights (i + 1) + 1))
  (h2 : ∀ i : Fin 2009, weights i ≤ 1000)
  (h3 : (Finset.univ.sum weights) % 2 = 0) :
  ∃ (A B : Finset (Fin 2009)), (A ∪ B = Finset.univ ∧ A ∩ B = ∅ ∧ A.sum weights = B.sum weights) :=
sorry

end equal_partition_of_weights_l325_325988


namespace sum_of_unique_four_digit_numbers_l325_325535

theorem sum_of_unique_four_digit_numbers : 
  let digits := [1, 2, 3, 4, 5]
  in (∑ n in {n | ∃ d1 d2 d3 d4, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
                                              d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ 
                                              n = d1 * 1000 + d2 * 100 + d3 * 10 + d4}.to_finset, n) = 399960 := 
by 
  sorry

end sum_of_unique_four_digit_numbers_l325_325535


namespace probability_one_pair_no_three_of_a_kind_l325_325862

/--
Each of five, standard, six-sided dice is rolled once. The probability that there is at least 
one pair but not a three-of-a-kind (that is, there are two dice showing the same value, but no 
three dice show the same value) is calculated as follows:
-/

theorem probability_one_pair_no_three_of_a_kind :
  let total_outcomes := 6^5 in
  let successful_case1 := 6 * 10 * 5 * 4 * 3 in
  let successful_case2 := 15 * 4 * 30 in
  let successful_outcomes := successful_case1 + successful_case2 in
  (successful_outcomes : ℚ) / total_outcomes = 25 / 36 :=
sorry

end probability_one_pair_no_three_of_a_kind_l325_325862


namespace markdown_calculation_l325_325042

noncomputable def markdown_percentage (P S : ℝ) (h_inc : P = S * 1.1494) : ℝ :=
  1 - (1 / 1.1494)

theorem markdown_calculation (P S : ℝ) (h_sale : S = P * (1 - markdown_percentage P S sorry / 100)) (h_inc : P = S * 1.1494) :
  markdown_percentage P S h_inc = 12.99 := 
sorry

end markdown_calculation_l325_325042


namespace segment_length_AE_l325_325727

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

theorem segment_length_AE :
  let A := (0, 4)
  let B := (7, 0)
  let C := (5, 0)
  let D := (6, 4)
  let E := (21 / 4, 1)
  distance A E ≈ 6.05 :=
by
  sorry

end segment_length_AE_l325_325727


namespace derivative_at_2_l325_325967

noncomputable def f (x : ℝ) := (1/3) * x^3 - (f 1)^2 * x^2 + x + 5

theorem derivative_at_2 : 
  (λ x, (1/3) * x^3 - (f 1)^2 * x^2 + x + 5).deriv 2 = 7/3 := 
by 
  -- Start of the proof, but only a statement is required.
  sorry

end derivative_at_2_l325_325967


namespace vector_sum_l325_325073

-- Definitions of the vectors
def a : ℝ × ℝ × ℝ := (3, -2, 1)
def b : ℝ × ℝ × ℝ := (-2, 4, 0)
def c : ℝ × ℝ × ℝ := (3, 0, 2)

-- The theorem to prove
theorem vector_sum : a.1 - 2 * b.1 + 4 * c.1 = 19 ∧
                     a.2 - 2 * b.2 + 4 * c.2 = -10 ∧
                     a.3 - 2 * b.3 + 4 * c.3 = 9 :=
by sorry

end vector_sum_l325_325073


namespace min_points_in_convex_n_gon_l325_325422

theorem min_points_in_convex_n_gon {n : ℕ} (h_n : n ≥ 3) :
  ∃ (m : ℕ), (∀ (tri : set (fin n) → set (fin n) → set (fin n)), is_triangle tri → ∃ p : fin n, p ∈ tri) ↔ m = n - 2 :=
sorry

end min_points_in_convex_n_gon_l325_325422


namespace number_of_triangles_is_correct_l325_325136

def points := Fin 6 × Fin 6

def is_collinear (p1 p2 p3 : points) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2)

noncomputable def count_triangles_with_positive_area : Nat :=
  let all_points := Finset.univ.product Finset.univ
  let all_combinations := all_points.powerset.filter (λ s, s.card = 3)
  let valid_triangles := all_combinations.filter (λ s, ¬is_collinear (s.choose 0) (s.choose 1) (s.choose 2))
  valid_triangles.card

theorem number_of_triangles_is_correct :
  count_triangles_with_positive_area = 6804 :=
by
  sorry

end number_of_triangles_is_correct_l325_325136


namespace arithmetic_formula_arithmetic_sum_geometric_bounds_geometric_formula_geometric_sum_l325_325919

section ArithmeticSequence

variable {a : ℕ → ℤ}

-- Assume an arithmetic sequence with specific properties
axiom a_property_1 : a 2 + a 5 = 16
axiom a_property_2 : a 5 - a 3 = 4

-- Prove the general formula
theorem arithmetic_formula : ∀ n : ℕ, a n = 2 * n + 1 := sorry

-- Prove the sum of terms from 2^(n-1) to 2^n - 1
theorem arithmetic_sum (n : ℕ) : (∑ i in Finset.range (2^n), if i ≥ 2^(n-1) then a i else 0) = 3 * (4^(n-1)) := sorry

end ArithmeticSequence

section GeometricSequence

variable {b : ℕ → ℤ}

-- Assume a geometric sequence with specific properties
axiom geometric_property_k : ∀ k n : ℕ, (2^(k-1) ≤ n) → (n ≤ 2^k - 1) → b k < a n ∧ a n < b (k+1)

-- Prove the bounds of the geometric sequence
theorem geometric_bounds (k : ℕ) (h : k ≥ 2) : 2^k - 1 < b k ∧ b k < 2^k + 1 := sorry

-- Prove the general formula for the geometric sequence
theorem geometric_formula : ∀ n : ℕ, b n = 2^n := sorry

-- Prove the sum of the first n terms of the geometric sequence
theorem geometric_sum (n : ℕ) : (∑ i in Finset.range n, b i) = 2^(n+1) - 2 := sorry

end GeometricSequence

end arithmetic_formula_arithmetic_sum_geometric_bounds_geometric_formula_geometric_sum_l325_325919


namespace payment_for_A_and_B_l325_325451

theorem payment_for_A_and_B 
  (rate_A : ℚ) (rate_B : ℚ) (rate_total : ℚ) (payment_C : ℚ)
  (h_rate_A : rate_A = 1 / 6)
  (h_rate_B : rate_B = 1 / 8)
  (h_rate_total : rate_total = 1 / 3)
  (h_payment_C : payment_C = 450) :
  (24 * payment_C) = 10800 :=
by
  unfold rate_A at h_rate_A -- A's work rate is 1/6 of the work per day
  unfold rate_B at h_rate_B -- B's work rate is 1/8 of the work per day
  unfold rate_total at h_rate_total -- A, B, and C together can do 1/3 of the work per day
  have h_combined_rate := h_rate_A + h_rate_B -- Combined rate of A and B
  sorry

end payment_for_A_and_B_l325_325451


namespace Hillary_newspaper_weeks_l325_325950

theorem Hillary_newspaper_weeks :
  (∀ (price_per_newspaper : ℕ → ℝ) (total_spent : ℝ),
   (price_per_newspaper 3 = 0.50) →
   (price_per_newspaper 4 = 2.0) →
   (total_spent = 28.0) →
   let weekly_expense := 3 * (price_per_newspaper 3) + (price_per_newspaper 4) in
   weeks := total_spent / weekly_expense in
   weeks = 8) :=
by
  intros,
  sorry

end Hillary_newspaper_weeks_l325_325950


namespace eval_neg_a_l325_325075

noncomputable def f (x : ℝ) : ℝ := x + 1/x - 2

theorem eval_neg_a (a : ℝ) (h : f a = 3) : f (-a) = -7 := by
  have h1 : a + 1/a = 5 := sorry
  have h2 : f (-a) = - (a + 1/a) - 2 := sorry
  show f (-a) = -7 := sorry

end eval_neg_a_l325_325075


namespace avg_ac_l325_325370

-- Define the ages of a, b, and c as variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def avg_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 26
def age_b (B : ℕ) : Prop := B = 20

-- State the theorem to prove
theorem avg_ac {A B C : ℕ} (h1 : avg_abc A B C) (h2 : age_b B) : (A + C) / 2 = 29 := 
by sorry

end avg_ac_l325_325370


namespace bridge_length_calculation_l325_325733

def length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := (train_speed_kmph * 1000) / 3600
  let distance_covered := speed_mps * time_seconds
  distance_covered - train_length

theorem bridge_length_calculation :
  length_of_bridge 140 45 30 = 235 :=
by
  unfold length_of_bridge
  norm_num
  sorry

end bridge_length_calculation_l325_325733


namespace div_by_binomial_condition_l325_325304

-- Define the polynomial P_n(x) as a function with degree n and coefficient a_0 no zero
def Pn (x : ℝ) (coeffs : List ℝ) (n : ℕ) : ℝ :=
  coeffs.foldr (λ (a : ℝ) (acc : ℝ) → acc * x + a) 0

-- Define the quotient polynomial Q_{n-1}(x)
def Qn_1 (x : ℝ) (quotCoeffs : List ℝ) (n : ℕ) : ℝ :=
  quotCoeffs.foldr (λ (a : ℝ) (acc : ℝ) → acc * x + a) 0

-- State the main theorem to be proven
theorem div_by_binomial_condition (α : ℝ) (coeffs : List ℝ) (n : ℕ) (a0_nonzero : coeffs.head! ≠ 0) :
  (∀ (quotCoeffs : List ℝ), ∃ (R : ℝ), Pn x coeffs n = (x - α) * Qn_1 x quotCoeffs (n - 1) + R ∧ ∀ x, Pn x coeffs n = 0 ↔ x = α ) ↔ Pn α coeffs n = 0 :=
begin
  sorry,
end

end div_by_binomial_condition_l325_325304


namespace polynomial_roots_G_l325_325924

noncomputable def G (p : Polynomial ℤ) (roots : List ℤ) : ℤ :=
  if h : p.degree = 7 ∧ ∀ r ∈ roots, r > 0 then (-1) * roots.map ((· ^ 3)).sum else 0

theorem polynomial_roots_G (p : Polynomial ℤ) (roots : List ℤ) :
  (p = Polynomial.C 36 + Polynomial.X * Polynomial.C I + Polynomial.X^2 * Polynomial.C H +
  Polynomial.X^3 * Polynomial.C G + Polynomial.X^4 * Polynomial.C F + Polynomial.X^5 * Polynomial.C E +
  Polynomial.X^6 * Polynomial.C (-13) + Polynomial.X^7) →
  (roots : List ℤ) → (∀ r ∈ roots, r > 0) →
  G p roots = -82 := sorry

end polynomial_roots_G_l325_325924


namespace sum_series_eq_l325_325501

def sum_series (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (3 + 9 * (k + 1)) / (8 ^ (n - k))

theorem sum_series_eq (n : ℕ) (h : n = 100) : sum_series n = 128.5714286 :=
by
  rw h
  sorry

end sum_series_eq_l325_325501


namespace arith_seq_ratio_l325_325088

variables {a₁ d : ℝ} (h₁ : d ≠ 0) (h₂ : (a₁ + 2*d)^2 ≠ a₁ * (a₁ + 8*d))

theorem arith_seq_ratio:
  (a₁ + 2*d) / (a₁ + 5*d) = 1 / 2 :=
sorry

end arith_seq_ratio_l325_325088


namespace count_positive_area_triangles_l325_325146

-- Define the grid size
def grid_size : ℕ := 6

-- Defining the main theorem
theorem count_positive_area_triangles :
  (set.univ.powerset.to_finset.filter (λ s : finset (ℤ × ℤ), s.card = 3 ∧ ¬collinear s)).card = 6628 :=
sorry

end count_positive_area_triangles_l325_325146


namespace sphere_surface_area_l325_325374

theorem sphere_surface_area (edge_length : ℝ) (surface_area : ℝ) 
    (h_edge_length : edge_length = 2)
    (h_vertices_on_sphere : ∀ (vertices : set (ℝ × ℝ × ℝ)), vertices.card = 8 ∧ ∀ v ∈ vertices, ∃ (R : ℝ), dist v (0, 0, 0) = R) :
    surface_area = 12 * Real.pi :=
by
  sorry -- Proof will be written here

end sphere_surface_area_l325_325374


namespace triangle_inequality_120_deg_l325_325683

theorem triangle_inequality_120_deg
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt (a^2 - ab + b^2) + sqrt (b^2 - bc + c^2) ≥ sqrt (a^2 + ac + c^2) ∧
  (sqrt (a^2 - ab + b^2) + sqrt (b^2 - bc + c^2) = sqrt (a^2 + ac + c^2) ↔ (1 / a + 1 / c = 1 / b)) :=
by sorry

end triangle_inequality_120_deg_l325_325683


namespace find_angle_C_find_side_c_l325_325182

-- Definitions using the given conditions
variables {A B C : ℝ}
variables {a b c : ℝ}
variables (h₁ : 4 * sin^2 ((A - B) / 2) + 4 * sin A * sin B = 2 + sqrt 2)
variables (h₂ : b = 4)
variables (h₃ : (1/2) * a * b * sin C = 6)

-- Proof part (no proof body, just the statements)
theorem find_angle_C : C = π / 4 :=
by
  sorry

theorem find_side_c
  (h₁ : 4 * sin^2 ((A - B) / 2) + 4 * sin A * sin B = 2 + sqrt 2)
  (h₂ : b = 4)
  (h₃ : (1/2) * a * b * sin C = 6)
  (hC : C = π / 4) :
  c = sqrt 10 :=
by
  sorry

end find_angle_C_find_side_c_l325_325182


namespace probability_sum_7_9_12_l325_325802

/-- A cubical die has faces marked with numbers {1, 1, 2, 2, 4, 4}.
Another die has faces marked with numbers {1, 2, 5, 5, 6, 7}.
What is the probability that the sum of the top-facing numbers after a roll of both dice will be 7, 9, or 12? -/
def first_die_faces := [1, 1, 2, 2, 4, 4]
def second_die_faces := [1, 2, 5, 5, 6, 7]

theorem probability_sum_7_9_12 :
  let successful_outcomes := [(1, 6), (2, 5), (2, 7), (4, 3), (4, 5)] in
  let total_outcomes := first_die_faces.length * second_die_faces.length in
  let successful_count := successful_outcomes.length in
  (successful_count : ℚ) / (total_outcomes : ℚ) = 5 / 18 :=
by
  sorry

end probability_sum_7_9_12_l325_325802


namespace circle_regions_division_l325_325246

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l325_325246


namespace square_root_3a_minus_4b_is_pm4_l325_325103

theorem square_root_3a_minus_4b_is_pm4
  (a b : ℝ)
  (h1 : sqrt (2*a + 1) = 3 ∨ sqrt (2*a + 1) = -3)
  (h2 : sqrt (5*a + 2*b - 2) = 4) :
  sqrt (3*a - 4*b) = 4 ∨ sqrt (3*a - 4*b) = -4 := 
by
  sorry

end square_root_3a_minus_4b_is_pm4_l325_325103


namespace suff_and_nec_condition_l325_325098

variable {a b : ℝ}
def f (x : ℝ) : ℝ := a ^ x
def g (x : ℝ) : ℝ := b ^ x

theorem suff_and_nec_condition (a_pos : a > 0) (a_neq_one : a ≠ 1) (b_pos : b > 0) (b_neq_one : b ≠ 1) :
  (f 2 > g 2) ↔ (a > b) :=
sorry

end suff_and_nec_condition_l325_325098


namespace min_value_of_2_a_plus_2_b_l325_325920

theorem min_value_of_2_a_plus_2_b (a b : ℝ) (h : a + b = 3) : 2^a + 2^b ≥ 4 * real.sqrt 2 :=
sorry

end min_value_of_2_a_plus_2_b_l325_325920


namespace distinct_a_count_l325_325894

theorem distinct_a_count :
  ∃ (a_set : Set ℝ), (∀ x ∈ a_set, ∃ r s : ℤ, r + s = -x ∧ r * s = 9 * x) ∧ a_set.toFinset.card = 3 :=
by 
  sorry

end distinct_a_count_l325_325894


namespace number_of_digits_of_1234_in_base5_l325_325953

def base5_representation_digits (n : ℕ) : ℕ :=
  if h : n > 0 then
    Nat.find (λ k, n < 5^(k + 1)) + 1
  else
    1

theorem number_of_digits_of_1234_in_base5 : base5_representation_digits 1234 = 5 := 
by
  unfold base5_representation_digits
  have h : ∃ k, 1234 < 5^(k+1), from Exists.intro 4 (by norm_num)
  simp [Nat.find_spec h]
  rfl

end number_of_digits_of_1234_in_base5_l325_325953


namespace average_speed_l325_325662

noncomputable def start_time : ℝ := 7.5 -- 7:30 a.m. in hours
noncomputable def end_time : ℝ := 14.75 -- 2:45 p.m. in hours
noncomputable def distance : ℝ := 273 -- Distance in miles

-- Average speed calculation: distance / (end_time - start_time)
theorem average_speed :
  let duration := end_time - start_time in
  duration = 7.25 →
  distance / duration = 37.6551724138 :=
by
  intros duration duration_eq
  rw duration_eq
  sorry

end average_speed_l325_325662


namespace three_monotonic_intervals_iff_a_lt_zero_l325_325381

-- Definition of the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Definition of the first derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main statement: Prove that f(x) has exactly three monotonic intervals if and only if a < 0.
theorem three_monotonic_intervals_iff_a_lt_zero (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ a < 0 :=
by
  sorry

end three_monotonic_intervals_iff_a_lt_zero_l325_325381


namespace find_average_age_and_experience_change_l325_325982

variable (A E WE ME W1 W2 : ℝ) (W1_range : 26 ≤ W1 ∧ W1 ≤ 30) (W2_range : 32 ≤ W2 ∧ W2 ≤ 36)

-- Given conditions
def average_age_increase (A : ℝ) : Prop :=
  8 * (A + 2) = 8 * A - (20 + 24) + W1 + W2

def work_experience_change (E WE ME : ℝ) : Prop :=
  8 * (E + 1) = 8 * E - ME + WE

-- Proof problem: Prove the average age of the two women and the change in their combined work experience years
theorem find_average_age_and_experience_change
    (h1 : average_age_increase A)
    (h2 : work_experience_change E WE ME) :
    (W1 + W2) / 2 = 30 ∧ WE - ME = 8 :=
by
  sorry

end find_average_age_and_experience_change_l325_325982


namespace minimum_rooks_5_l325_325725

-- Define a 9x9 board and a color function that returns if a cell is white based on checkerboard pattern
def is_white (i j : ℕ) : Prop := (i + j) % 2 = 0

-- Define a rook's attack function
def attacks (rook_i rook_j cell_i cell_j : ℕ) : Prop :=
  rook_i = cell_i ∨ rook_j = cell_j

-- Define the condition that all white cells are under attack by a set of rooks
def all_white_cells_attacked (rooks : list (ℕ × ℕ)) : Prop :=
  ∀ i j, is_white i j → ∃ rook, rook ∈ rooks ∧ attacks rook.fst rook.snd i j

-- Define the specific problem for a 9x9 board
def minimum_rooks (n : ℕ) : Prop :=
  ∃ rooks : list (ℕ × ℕ), rooks.length = n ∧ all_white_cells_attacked rooks

-- Statement of the problem
theorem minimum_rooks_5 : minimum_rooks 5 :=
  sorry

end minimum_rooks_5_l325_325725


namespace seeds_in_fourth_pot_l325_325518

-- Define the conditions as variables
def total_seeds : ℕ := 10
def number_of_pots : ℕ := 4
def seeds_per_pot : ℕ := 3

-- Define the theorem to prove the quantity of seeds planted in the fourth pot
theorem seeds_in_fourth_pot :
  (total_seeds - (seeds_per_pot * (number_of_pots - 1))) = 1 := by
  sorry

end seeds_in_fourth_pot_l325_325518


namespace smallest_prime_divisor_524_plus_718_l325_325758

theorem smallest_prime_divisor_524_plus_718 (x y : ℕ) (h1 : x = 5 ^ 24) (h2 : y = 7 ^ 18) :
  ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ p ∣ (x + y) :=
by
  sorry

end smallest_prime_divisor_524_plus_718_l325_325758


namespace expected_twos_three_dice_l325_325410

def expected_twos (n : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), k * (nat.choose n k) * (1/6)^k * (5/6)^(n - k)

theorem expected_twos_three_dice : expected_twos 3 = 1/2 :=
by
  sorry

end expected_twos_three_dice_l325_325410


namespace pq_eq_qr_l325_325711

noncomputable section

open EuclideanGeometry

variables (A B C D K P Q R : Point) (circle : Circle)
variables [InscribeQuadrilateral circle A B C D]
variables (tangentAtB : TangentLine circle B A K) (tangentAtD : TangentLine circle D K C)
variables (lineAC : Line_on A C)

theorem pq_eq_qr 
  (H1 : CyclicQuadrilateral A B C D)
  (H2 : LineParallel line t KB)
  (H3 : line_intersect A K P)
  (H4 : line_intersect B K Q)
  (H5 : line_intersect C K R) :
  PQ = QR := 
sorry 

end pq_eq_qr_l325_325711


namespace median_of_scores_is_39_l325_325984

noncomputable def student_scores : List ℕ := [36, 36, 37, 37, 37, 38, 39, 39, 39, 39, 40, 40]

theorem median_of_scores_is_39 :
  let score_list := (repeat 36 1) ++ (repeat 37 2) ++ (repeat 38 1) ++ (repeat 39 4) ++ (repeat 40 2)
  median score_list = 39 := 
by
  hint sorry

end median_of_scores_is_39_l325_325984


namespace eval_expr_correct_l325_325062

noncomputable def eval_expr : ℝ :=
  let a := (12:ℝ)^5 * (6:ℝ)^4
  let b := (3:ℝ)^2 * (36:ℝ)^2
  let c := Real.sqrt 9 * Real.log (27:ℝ)
  (a / b) + c

theorem eval_expr_correct : eval_expr = 27657.887510597983 := by
  sorry

end eval_expr_correct_l325_325062


namespace xy_difference_l325_325617

theorem xy_difference (x y : ℚ) (h1 : 3 * x - 4 * y = 17) (h2 : x + 3 * y = 5) : x - y = 73 / 13 :=
by
  sorry

end xy_difference_l325_325617


namespace proof_problem_l325_325092

variable (x y : ℝ)

def condition : Prop := 3^x + 5^y > 3^(-y) + 5^(-x)

theorem proof_problem (h : condition x y) : x + y > 0 := by
  sorry

end proof_problem_l325_325092


namespace problem_1_problem_2_l325_325127

open Real

noncomputable def minimum_value (x y z : ℝ) : ℝ :=
  1 / x + 1 / y + 1 / z

theorem problem_1 (x y z : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_posz : 0 < z) (h_sum : x + 2 * y + 3 * z = 1) :
  minimum_value x y z = 6 + 2 * sqrt 2 + 2 * sqrt 3 + 2 * sqrt 6 :=
sorry

theorem problem_2 (x y z : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_posz : 0 < z) (h_sum : x + 2 * y + 3 * z = 1) :
  x^2 + y^2 + z^2 ≥ 1 / 14 :=
sorry

end problem_1_problem_2_l325_325127


namespace space_shuttle_speed_conversion_l325_325814

theorem space_shuttle_speed_conversion : 
  (let speed_per_second := 2 
   let seconds_in_minute := 60
   let minutes_in_hour := 60 in
   let speed_per_hour := speed_per_second * seconds_in_minute * minutes_in_hour in
   speed_per_hour = 7200) :=
by
  sorry

end space_shuttle_speed_conversion_l325_325814


namespace count_positive_area_triangles_l325_325147

-- Define the grid size
def grid_size : ℕ := 6

-- Defining the main theorem
theorem count_positive_area_triangles :
  (set.univ.powerset.to_finset.filter (λ s : finset (ℤ × ℤ), s.card = 3 ∧ ¬collinear s)).card = 6628 :=
sorry

end count_positive_area_triangles_l325_325147


namespace number_of_zero_sequences_l325_325297
-- Lean 4 statement for the given problem:

theorem number_of_zero_sequences :
  let T := { t : ℤ × ℤ × ℤ // 1 ≤ t.1 ∧ t.1 ≤ 15 ∧ 1 ≤ t.2 ∧ t.2 ≤ 15 ∧ 1 ≤ t.3 ∧ t.3 ≤ 15 }
  let generates_zero (b1 b2 b3 : ℤ) : Prop :=
    ∃ (n : ℕ), (n ≥ 4) ∧ (b_n = 0)
      where b : ℕ → ℤ
        | 1 => b1
        | 2 => b2
        | 3 => b3
        | (n + 1) => b n * (b (n - 1) - b (n - 2))^2
  in
  ∃ n ∈ T, generates_zero n.1 n.2 n.3 :=
  435
:= 
-- proof is not required as per instructions
sorry

end number_of_zero_sequences_l325_325297


namespace reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l325_325394

theorem reciprocal_opposite_of_neg_neg_3_is_neg_one_third : 
  (1 / (-(-3))) = -1 / 3 :=
by
  sorry

end reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l325_325394


namespace complex_ratio_range_l325_325307

theorem complex_ratio_range (x y : ℝ)
  (h : (z : ℂ) = x + y * complex.I ∧
       (real_part : ℝ) = ((x + 1) / (x + 2)) ∧
       (imag_part : ℝ) = (y / (x + 2)) ∧
       ((real_part / imag_part) = sqrt 3)) :
    let k := y / x in
    k ∈ (set.Icc ((-3 * sqrt 3 - 4 * sqrt 2) / 5) ((-3 * sqrt 3 + 4 * sqrt 2) / 5)) :=
begin
  sorry
end

end complex_ratio_range_l325_325307


namespace union_of_sets_l325_325293

def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}

theorem union_of_sets :
  A ∪ B = {x | 3 < x ∧ x ≤ 10} :=
by
  sorry

end union_of_sets_l325_325293


namespace coloring_two_corners_removed_l325_325746

noncomputable def coloring_count (total_ways : Nat) (ways_without_corner_a : Nat) : Nat :=
  total_ways - 2 * (total_ways - ways_without_corner_a) / 2 + 
  (ways_without_corner_a - (total_ways - ways_without_corner_a) / 2)

theorem coloring_two_corners_removed : coloring_count 120 96 = 78 := by
  sorry

end coloring_two_corners_removed_l325_325746


namespace conference_games_l325_325004

theorem conference_games (teams_per_division : ℕ) (divisions : ℕ) 
  (intradivision_games_per_team : ℕ) (interdivision_games_per_team : ℕ) 
  (total_teams : ℕ) (total_games : ℕ) : 
  total_teams = teams_per_division * divisions →
  intradivision_games_per_team = (teams_per_division - 1) * 2 →
  interdivision_games_per_team = teams_per_division →
  total_games = (total_teams * (intradivision_games_per_team + interdivision_games_per_team)) / 2 →
  total_games = 133 :=
by
  intros
  sorry

end conference_games_l325_325004


namespace probability_closer_to_6_l325_325467

theorem probability_closer_to_6 :
  let interval : Set ℝ := Set.Icc 0 6
  let subinterval : Set ℝ := Set.Icc 3 6
  let length_interval := 6
  let length_subinterval := 3
  (length_subinterval / length_interval) = 0.5 := by
    sorry

end probability_closer_to_6_l325_325467


namespace tasty_residue_count_2016_l325_325852

def tasty_residue (n : ℕ) (a : ℕ) : Prop :=
  1 < a ∧ a < n ∧ ∃ m : ℕ, m > 1 ∧ a ^ m ≡ a [MOD n]

theorem tasty_residue_count_2016 : 
  (∃ count : ℕ, count = 831 ∧ ∀ a : ℕ, 1 < a ∧ a < 2016 ↔ tasty_residue 2016 a) :=
sorry

end tasty_residue_count_2016_l325_325852


namespace books_per_shelf_l325_325006

theorem books_per_shelf 
  (initial_books : ℕ) 
  (sold_books : ℕ) 
  (num_shelves : ℕ) 
  (remaining_books : ℕ := initial_books - sold_books) :
  initial_books = 40 → sold_books = 20 → num_shelves = 5 → remaining_books / num_shelves = 4 :=
by
  sorry

end books_per_shelf_l325_325006


namespace number_of_triangles_is_correct_l325_325138

def points := Fin 6 × Fin 6

def is_collinear (p1 p2 p3 : points) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2)

noncomputable def count_triangles_with_positive_area : Nat :=
  let all_points := Finset.univ.product Finset.univ
  let all_combinations := all_points.powerset.filter (λ s, s.card = 3)
  let valid_triangles := all_combinations.filter (λ s, ¬is_collinear (s.choose 0) (s.choose 1) (s.choose 2))
  valid_triangles.card

theorem number_of_triangles_is_correct :
  count_triangles_with_positive_area = 6804 :=
by
  sorry

end number_of_triangles_is_correct_l325_325138


namespace area_of_shaded_figure_is_half_l325_325203

structure Hexagon :=
  (A B C D E F : Point)

structure Midpoint (P Q R : Point) :=
  (is_midpoint : dist P R = dist R Q)

def area (hex : Hexagon) : ℝ := sorry -- assuming presence of a function to calculate area of hexagon

def shaded_area (hex : Hexagon) (M : Point) (N : Point) : ℝ := sorry -- assuming presence of a function to calculate area of the shaded region

theorem area_of_shaded_figure_is_half (hex : Hexagon) (M N : Point)
  (hM : Midpoint hex.A hex.C M)
  (hN : Midpoint hex.C hex.E N) :
  shaded_area hex M N = (1 / 2) * area hex := sorry

end area_of_shaded_figure_is_half_l325_325203


namespace measure_angle_E_36_degrees_l325_325686

def angle_A_measure : ℝ := sorry -- We will prove it is 36 degrees
def angle_B_measure : ℝ := 4 * angle_A_measure
def angle_E_measure : ℝ := angle_A_measure

theorem measure_angle_E_36_degrees
  (h_parallel : ∀ x : ℝ, x ∥ 0)
  (h_angle_relation : angle_A_measure = angle_B_measure / 4)
  (h_straight_line : angle_B_measure + angle_E_measure = 180) :
  angle_E_measure = 36 :=
by
  sorry

end measure_angle_E_36_degrees_l325_325686


namespace amanda_coffee_shop_l325_325823

-- Let x be the number of pounds of type A coffee.
-- Type B coffee is twice the pounds of type A coffee, i.e., 2x.
-- Costs and total cost are provided as conditions.

theorem amanda_coffee_shop :
  ∃ (x : ℝ), (4.60 * x + 11.90 * x = 511.50) ∧ (x = 31) :=
by
  -- Let x be the number of pounds of type A coffee and set up the equation
  use 31
  -- Conditions
  have h1 : 4.60 * 31 + 11.90 * 31 = 511.50,
    from sorry
  exact ⟨h1, rfl⟩

end amanda_coffee_shop_l325_325823


namespace nancy_clay_pots_total_l325_325689

theorem nancy_clay_pots_total :
  let pots_monday := 12 in
  let pots_tuesday := 2 * pots_monday in
  let pots_wednesday := 14 in
  pots_monday + pots_tuesday + pots_wednesday = 50 := by
  sorry

end nancy_clay_pots_total_l325_325689


namespace subtract_three_from_binary_l325_325165

theorem subtract_three_from_binary (M : ℕ) (M_binary: M = 0b10110000) : (M - 3) = 0b10101101 := by
  sorry

end subtract_three_from_binary_l325_325165


namespace men_meet_4_miles_nearer_R_than_S_l325_325417

def distance_between_points : ℝ := 76
def rate_at_R : ℝ := 4.5
def initial_rate_at_S : ℝ := 3.25
def rate_increase_at_S_per_hour : ℝ := 0.5

theorem men_meet_4_miles_nearer_R_than_S :
  ∃ (h : ℕ) (x : ℝ), x = 4 ∧
  (rate_at_R * h + (h / 2 * (2 * initial_rate_at_S + (h - 1) * rate_increase_at_S_per_hour)) = distance_between_points) :=
begin
  sorry
end

end men_meet_4_miles_nearer_R_than_S_l325_325417


namespace average_age_of_students_l325_325186

variables (A : ℕ)  -- Define A to be the average age of the students without the teacher

-- Conditions
constants (num_students : ℕ) (teacher_age combined_avg_age : ℕ)
axiom h1 : num_students = 10
axiom h2 : teacher_age = 26
axiom h3 : combined_avg_age = 16

-- Total age of the students without the teacher
def students_total_age : ℕ := num_students * A

-- Equation including the teacher's age
def total_age_with_teacher : ℕ := students_total_age + teacher_age

-- Combined average age condition
axiom h4 : (total_age_with_teacher = combined_avg_age * (num_students + 1))

-- Prove the average age of the students without the teacher
theorem average_age_of_students : A = 15 := 
by sorry

end average_age_of_students_l325_325186


namespace circle_region_count_l325_325232

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l325_325232


namespace area_of_triangle_l325_325420

variable (x y : ℝ)

def line1 := x + y = 2005
def line2 := x / 2005 + y / 2006 = 1
def line3 := x / 2006 + y / 2005 = 1

theorem area_of_triangle :
  let a := 2005 in
  let b := 2006 in
  let P1 := (a, 0) in
  let P2 := (0, a) in
  let P3 := (a * b / (a + b), a * b / (a + b)) in
  let area := abs ((P1.1 * (P2.2 - P3.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P1.2 - P2.2)) / 2) in
  area = 501.13 :=
by
  sorry

end area_of_triangle_l325_325420


namespace no_positive_integer_a_n_nonnegative_integer_l325_325068

def a_n (n : ℕ) : ℚ :=
  ∑ k in finset.range (n+1), (2*(k + n) + 1 : ℚ)^n / (k + n : ℚ)

theorem no_positive_integer_a_n_nonnegative_integer (n : ℕ) (h : 0 < n) : ¬ ∃ m : ℤ, m ≥ 0 ∧ a_n n = m :=
by
  sorry

end no_positive_integer_a_n_nonnegative_integer_l325_325068


namespace proof_problem_l325_325206

-- Definitions of curve C and line l
def curve_C (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, Real.sin α)
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin(θ - π / 4) = Real.sqrt 2

-- Definitions for finding equations
def ordinary_equation_C (x y : ℝ) : Prop := (x^2 / 9) + (y^2) = 1
def rectangular_equation_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Definition of point P
def P := (0, 2)

-- Definition to find PA and PB
noncomputable def length_PA_PB (A B : ℝ × ℝ) : ℝ :=
  Real.abs (P.1 - A.1) + Real.abs (P.2 - A.2) + Real.abs (P.1 - B.1) + Real.abs (P.2 - B.2)

-- Main proof statement
theorem proof_problem (α : ℝ) (ρ θ : ℝ) (A B : ℝ × ℝ) :
  (ordinary_equation_C (3 * Real.cos α) (Real.sin α)) ∧ 
  (line_l ρ θ) ∧ 
  (rectangular_equation_l A.1 A.2) ∧ 
  (rectangular_equation_l B.1 B.2)
  → length_PA_PB A B = 18 * Real.sqrt 2 / 5 :=
  by sorry

end proof_problem_l325_325206


namespace cube_surface_area_with_holes_l325_325458

theorem cube_surface_area_with_holes (a b : ℕ) (h₁ : a = 5) (h₂ : b = 2) :
  let original_surface_area := 6 * a^2,
      hole_area := 6 * b^2,
      internal_exposed_area := 96  -- Calculation based on the problem description
  in original_surface_area - hole_area + internal_exposed_area = 222 :=
by
  -- The proof is skipped as per instructions
  sorry

end cube_surface_area_with_holes_l325_325458


namespace min_packs_needed_for_soda_l325_325718

def soda_pack_sizes : List ℕ := [8, 15, 30]
def total_cans_needed : ℕ := 120

theorem min_packs_needed_for_soda : ∃ n, n = 4 ∧
  (∀ p ∈ {a // (a ∈ soda_pack_sizes)}, (n*p) ≤ total_cans_needed) ∧
  (∀ m, m < n → ∀ q ∈ {a // (a ∈ soda_pack_sizes)}, (m*q) < total_cans_needed) := by
  sorry

end min_packs_needed_for_soda_l325_325718


namespace percentage_smelling_rotten_l325_325317

def total_apples : ℕ := 200
def percentage_rotten : ℕ := 40
def number_not_smelling_rotten : ℕ := 24

theorem percentage_smelling_rotten :
  let total_rotten := percentage_rotten * total_apples / 100 in
  let smelling_rotten := total_rotten - number_not_smelling_rotten in
  (smelling_rotten * 100 / total_rotten) = 70 :=
by 
  let total_rotten := percentage_rotten * total_apples / 100
  let smelling_rotten := total_rotten - number_not_smelling_rotten
  sorry

end percentage_smelling_rotten_l325_325317


namespace petya_sum_of_digits_l325_325321

theorem petya_sum_of_digits (M N : ℕ) :
  (∀ k : ℕ, sum_of_digits (N + k * M) = sum_of_digits N) → 
  ¬ (∀ k : ℕ, sum_of_digits (N + k * M) = 2) :=
by
  sorry

end petya_sum_of_digits_l325_325321


namespace simplify_expression_l325_325716

theorem simplify_expression (x : ℤ) : 
  (2 * x ^ 13 + 3 * x ^ 12 - 4 * x ^ 9 + 5 * x ^ 7) + 
  (8 * x ^ 11 - 2 * x ^ 9 + 3 * x ^ 7 + 6 * x ^ 4 - 7 * x + 9) + 
  (x ^ 13 + 4 * x ^ 12 + x ^ 11 + 9 * x ^ 9) = 
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 3 * x ^ 9 + 8 * x ^ 7 + 6 * x ^ 4 - 7 * x + 9 :=
sorry

end simplify_expression_l325_325716


namespace min_distance_complex_l325_325557

theorem min_distance_complex (z : ℂ) (h1 : complex.abs z = complex.abs (z + (2 + 2 * complex.i))) :
  complex.abs (z - (1 - complex.i)) ≥ complex.abs (1 - complex.i) := 
sorry

end min_distance_complex_l325_325557


namespace probability_three_black_balls_probability_white_ball_l325_325833

-- Definitions representing conditions
def total_ratio (A B C : ℕ) := A / B = 5 / 4 ∧ B / C = 4 / 6

-- Proportions of black balls in each box
def proportion_black_A (black_A total_A : ℕ) := black_A = 40 * total_A / 100
def proportion_black_B (black_B total_B : ℕ) := black_B = 25 * total_B / 100
def proportion_black_C (black_C total_C : ℕ) := black_C = 50 * total_C / 100

-- Problem 1: Probability of selecting a black ball from each box
theorem probability_three_black_balls
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C) :
  (black_A / total_A) * (black_B / total_B) * (black_C / total_C) = 1 / 20 :=
  sorry

-- Problem 2: Probability of selecting a white ball from the mixed total
theorem probability_white_ball
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (white_A white_B white_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C)
  (h5 : white_A = total_A - black_A)
  (h6 : white_B = total_B - black_B)
  (h7 : white_C = total_C - black_C) :
  (white_A + white_B + white_C) / (total_A + total_B + total_C) = 3 / 5 :=
  sorry

end probability_three_black_balls_probability_white_ball_l325_325833


namespace min_surface_area_of_prism_l325_325653

noncomputable def minimum_surface_area_of_circumscribed_sphere
  (volume : ℝ) (AB : ℝ) (AC : ℝ) : ℝ :=
  if volume = 3 * Real.sqrt 7 ∧ AB = 2 ∧ AC = 3 then 18 * Real.pi else 0

theorem min_surface_area_of_prism :
  minimum_surface_area_of_circumscribed_sphere 3.sqrt 7 2 3 = 18 * Real.pi :=
by
  sorry

end min_surface_area_of_prism_l325_325653


namespace binomial_identity_l325_325573

theorem binomial_identity :
  (nat.choose 18 8 = 31824) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 172822 :=
by
  intros h1 h2 h3
  have h4: nat.choose 19 9 = nat.choose 18 8 + nat.choose 18 9 := by sorry
  have h5: nat.choose 19 9 = 31824 + 48620 := by sorry
  have h6: nat.choose 19 10 = nat.choose 18 9 + nat.choose 18 10 := by sorry
  have h7: nat.choose 19 10 = 48620 + 43758 := by sorry
  show nat.choose 20 10 = nat.choose 19 9 + nat.choose 19 10 from sorry
  have h8: nat.choose 20 10 = 80444 + 92378 := by sorry
  exact sorry

end binomial_identity_l325_325573


namespace same_functions_l325_325762

-- Definitions
def f (x : ℝ) : ℝ := abs (x - 3)
def g (x : ℝ) : ℝ := real.sqrt ((x - 3)^2)

-- Statement
theorem same_functions : ∀ x, f x = g x :=
by
  -- Proof not required
  sorry

end same_functions_l325_325762


namespace max_wx_plus_xy_plus_yz_plus_wz_l325_325679

theorem max_wx_plus_xy_plus_yz_plus_wz (w x y z : ℝ) (h_nonneg : 0 ≤ w ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : w + x + y + z = 200) :
  wx + xy + yz + wz ≤ 10000 :=
sorry

end max_wx_plus_xy_plus_yz_plus_wz_l325_325679


namespace value_of_a_minus_b_minus_c_l325_325676

theorem value_of_a_minus_b_minus_c :
  let a := 1
  let b := -1
  let c := 0
  a - b - c = 2 := by
  let a := 1
  let b := -1
  let c := 0
  have : a - b - c = 2 := by
    calc
      a - b - c = 1 - (-1) - 0 : by rw [a, b, c]
              ... = 1 + 1 - 0 : by simp
              ... = 2 : by simp
  exact this
  sorry

end value_of_a_minus_b_minus_c_l325_325676


namespace complement_union_l325_325674

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4, 5}
def complementU (A B : Set ℕ) : Set ℕ := U \ (A ∪ B)

theorem complement_union :
  complementU A B = {2, 6} := by
  sorry

end complement_union_l325_325674


namespace soup_weight_after_four_days_l325_325024
noncomputable def remainingWeightAfterReductions (initialWeight : ℝ) (reductions : List ℝ) : ℝ :=
  initialWeight * reductions.foldl (λ acc r, acc * (1 - r)) 1

theorem soup_weight_after_four_days :
  remainingWeightAfterReductions 80 [0.40, 0.35, 0.55, 0.50] ≈ 7.02 :=
by
  -- We substitute the list of reduction rates to check for equivalence with 7.02
  -- This equivalence can be proven using appropriate numerical methods or bounds
  sorry

end soup_weight_after_four_days_l325_325024


namespace minimum_value_f_l325_325870

def f (x : ℝ) : ℝ := x^2 / (x + 1)

theorem minimum_value_f :
  ∃ x : ℝ, x > -1 ∧ (∀ y: ℝ, y > -1 → f(y) ≥ 0) ∧ f(x) = 0 :=
by
  sorry

end minimum_value_f_l325_325870


namespace find_difference_l325_325742

-- Define the necessary constants and variables
variables (u v : ℝ)

-- Define the conditions
def condition1 := u + v = 360
def condition2 := u = (1/1.1) * v

-- Define the theorem to prove
theorem find_difference (h1 : condition1 u v) (h2 : condition2 u v) : v - u = 17 := 
sorry

end find_difference_l325_325742


namespace fraction_green_after_tripling_l325_325636

theorem fraction_green_after_tripling 
  (x : ℕ)
  (h₁ : ∃ x, 0 < x) -- Total number of marbles is a positive integer
  (h₂ : ∀ g y, g + y = x ∧ g = 1/4 * x ∧ y = 3/4 * x) -- Initial distribution
  (h₃ : ∀ y : ℕ, g' = 3 * g ∧ y' = y) -- Triple the green marbles, yellow stays the same
  : (g' / (g' + y')) = 1/2 := 
sorry

end fraction_green_after_tripling_l325_325636


namespace sum_c_eq_138600_l325_325878

def c (q : ℕ) : ℕ :=
  if ∃ m : ℕ, 0 < m ∧ |m - real.sqrt q| < 1 / 3 then classical.some (exists_pos_int_of_abs_sub_lt q) else 0

theorem sum_c_eq_138600 : (∑ q in finset.range 3000, c q) = 138600 := sorry

end sum_c_eq_138600_l325_325878


namespace min_value_xyz_l325_325556

theorem min_value_xyz (x y z : ℝ) (h1 : 0 < x ∧ 0 < y ∧ 0 < z) (h2 : x^2 + y^2 + z^2 = 1) : 
  (min (frac (y * z) x + frac (z * x) y + frac (x * y) z)) = sqrt 3 := 
sorry

end min_value_xyz_l325_325556


namespace circle_divided_into_regions_l325_325259

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l325_325259


namespace regions_divided_by_radii_circles_l325_325240

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l325_325240


namespace circles_radii_divide_regions_l325_325269

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l325_325269


namespace num_real_values_for_integer_roots_l325_325879

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l325_325879


namespace sum_binom_identity_l325_325680

theorem sum_binom_identity (n r : ℕ) (h1 : 0 < r) (h2 : r < n) (h3 : (n - r) % 2 = 0) :
  ∑ k in Finset.range (n - r + 1), (-2) ^ (-k : ℤ) * (Nat.choose n (r + k)) * (Nat.choose (n + r + k) k) = 
  2 ^ (r - n : ℤ) * (-1) ^ ((n - r) / 2) * Nat.choose n ((n - r) / 2) := 
sorry

end sum_binom_identity_l325_325680


namespace fraction_sum_l325_325152

theorem fraction_sum (x a b : ℕ) (h1 : x = 36 / 99) (h2 : a = 4) (h3 : b = 11) (h4 : Nat.gcd a b = 1) : a + b = 15 :=
by
  sorry

end fraction_sum_l325_325152


namespace first_term_of_arithmetic_sequence_l325_325066

theorem first_term_of_arithmetic_sequence (a d : ℚ)
  (h1 : 15 * (2 * a + 29 * d) = 300)
  (h2 : 25 * (2 * a + 109 * d) = 3750) :
  a = -217 / 16 :=
by {
  sorry,
}

end first_term_of_arithmetic_sequence_l325_325066


namespace simplify_fraction_l325_325363

theorem simplify_fraction (h1 : 90 = 2 * 3^2 * 5) (h2 : 150 = 2 * 3 * 5^2) : (90 / 150 : ℚ) = 3 / 5 := by
  sorry

end simplify_fraction_l325_325363


namespace distinct_a_count_l325_325892

theorem distinct_a_count :
  ∃ (a_set : Set ℝ), (∀ x ∈ a_set, ∃ r s : ℤ, r + s = -x ∧ r * s = 9 * x) ∧ a_set.toFinset.card = 3 :=
by 
  sorry

end distinct_a_count_l325_325892


namespace tan_alpha_mul_tan_beta_l325_325902

theorem tan_alpha_mul_tan_beta (α β : Real) 
  (h1 : Real.cos(α + β) = 2 / 3) 
  (h2 : Real.cos(α - β) = 1 / 3) : 
  Real.tan α * Real.tan β = -1 / 3 := 
by
  sorry

end tan_alpha_mul_tan_beta_l325_325902


namespace circle_regions_l325_325272

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l325_325272


namespace elmer_saves_14_3_percent_l325_325044

-- Define the problem statement conditions and goal
theorem elmer_saves_14_3_percent (old_efficiency new_efficiency : ℝ) (old_cost new_cost : ℝ) :
  new_efficiency = 1.75 * old_efficiency →
  new_cost = 1.5 * old_cost →
  (500 / old_efficiency * old_cost - 500 / new_efficiency * new_cost) / (500 / old_efficiency * old_cost) * 100 = 14.3 := by
  -- sorry to skip the actual proof
  sorry

end elmer_saves_14_3_percent_l325_325044


namespace main_theorem_l325_325592

-- Definitions based on the problem conditions
def f (a x : ℝ) := a * Real.log x + (a + 1) / 2 * x ^ 2 + 1

-- Part Ⅰ: Maximum and minimum when a = -1/2 in [1/e, e]
noncomputable def part1_max (a : ℝ) (x : ℝ) := f a x = (1/2 : ℝ) + (Real.exp 2) / (4 : ℝ)
noncomputable def part1_min (a : ℝ) (x : ℝ) := f a x = (5/4 : ℝ)

-- Part Ⅱ: Discussing monotonicity
def f' (a x : ℝ) := ((a + 1) * x ^ 2 + a) / x

-- Part Ⅲ: Finding the range of 'a'
def part3_inequality (a : ℝ) : Prop := -1 < a ∧ a < 0 ∧ ∀ x, f a x > 1 + (a / 2) * Real.log (-a) 
def range_of_a (a : ℝ) : Prop := (1 / Real.exp 1 - 1 < a) ∧ (a < 0)

-- The main theorem combining all parts
theorem main_theorem:
  ∀ a x : ℝ, 
  (a = -1/2 → part1_max a x ∧ part1_min a x) ∧
  (a <= -1 → ∀ x, f' a x < 0 ∧ 
    ∀ (ha : a >= 0), ∀ x, f' a x > 0 ∧ 
    ∀ (ha : -1 < a < 0), (∀ x, x > Real.sqrt ((-a)/(a+1)) → f' a x > 0) ∧ (∀ x, x < Real.sqrt ((-a)/(a+1)) → f' a x < 0)) ∧
  part3_inequality a → range_of_a a :=
by sorry

end main_theorem_l325_325592


namespace nine_pointed_star_sum_of_angles_l325_325691

theorem nine_pointed_star_sum_of_angles (h : ∀i, ((i : ℤ) % 9) * 40 = 0) :
  ∑ (k : Fin 9), (1 / 2) * 4 * 40 = 720 := 
sorry

end nine_pointed_star_sum_of_angles_l325_325691


namespace ball_is_green_probability_l325_325506

noncomputable def probability_green_ball : ℚ :=
  let containerI_red := 8
  let containerI_green := 4
  let containerII_red := 3
  let containerII_green := 5
  let containerIII_red := 4
  let containerIII_green := 6
  let probability_container := (1 : ℚ) / 3
  let probability_green_I := (containerI_green : ℚ) / (containerI_red + containerI_green)
  let probability_green_II := (containerII_green : ℚ) / (containerII_red + containerII_green)
  let probability_green_III := (containerIII_green : ℚ) / (containerIII_red + containerIII_green)
  probability_container * probability_green_I +
  probability_container * probability_green_II +
  probability_container * probability_green_III

theorem ball_is_green_probability :
  probability_green_ball = 187 / 360 :=
by
  -- The detailed proof is omitted and left as an exercise
  sorry

end ball_is_green_probability_l325_325506


namespace kids_in_2004_l325_325633

-- Set up the variables
variables (kids2004 kids2005 kids2006 : ℕ)

-- Given conditions
def condition1 : Prop := kids2005 = kids2004 / 2
def condition2 : Prop := kids2006 = (2 * kids2005) / 3
def condition3 : Prop := kids2006 = 20

-- Theorem statement
theorem kids_in_2004 :
  condition1 →
  condition2 →
  condition3 →
  kids2004 = 60 :=
by
  intros h1 h2 h3
  sorry

end kids_in_2004_l325_325633


namespace person_hired_l325_325310

theorem person_hired (A_truth : ¬A)
  (B_truth : ¬(C))
  (C_truth : ¬(D))
  (D_truth : ¬D)
  (one_truth : (A_truth ∧ ¬B_truth ∧ ¬C_truth ∧ ¬D_truth) ∨
               (¬A_truth ∧ B_truth ∧ ¬C_truth ∧ ¬D_truth) ∨
               (¬A_truth ∧ ¬B_truth ∧ C_truth ∧ ¬D_truth) ∨
               (¬A_truth ∧ ¬B_truth ∧ ¬C_truth ∧ D_truth))
  (one_hired : (¬A ∧ B ∧ C ∧ D) ∨
               (A ∧ ¬B ∧ C ∧ D) ∨
               (A ∧ B ∧ ¬C ∧ D) ∨
               (A ∧ B ∧ C ∧ ¬D)) :
  A := sorry

end person_hired_l325_325310


namespace find_smallest_k_l325_325789

-- The initial conditions described as constants
constant h0 : ℝ := 1500
constant r : ℝ := 0.4
constant decay : ℝ := 0.95
constant threshold : ℝ := 2

-- Definition of the bounce height formula hk
noncomputable def hk (k : ℕ) : ℝ :=
  h0 * (r ^ k) * (decay ^ (k * (k - 1) / 2))

-- The proof goal is to find the smallest k for which hk < threshold
theorem find_smallest_k : ∃ k : ℕ, hk k < threshold ∧ ∀ m < k, ¬ hk m < threshold := sorry

end find_smallest_k_l325_325789


namespace set_intersection_l325_325685

theorem set_intersection (A B : Set ℝ)
  (hA : A = { x : ℝ | 1 < x ∧ x < 4 })
  (hB : B = { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }) :
  A ∩ (Set.univ \ B) = { x : ℝ | 3 < x ∧ x < 4 } :=
by
  sorry

end set_intersection_l325_325685


namespace max_value_on_circle_l325_325083

noncomputable def max_value (x y : ℝ) : ℝ := 2 * x + y

theorem max_value_on_circle {x y : ℝ} (h : x^2 + y^2 = 4) :
  max_value x y ≤ 2 * Real.sqrt 5 :=
by
  have h₀ : |max_value x y| = |2 * x + y| := rfl
  have h₁ : max_value x y = 2 * x + y := rfl
  have h₂ : (2 * x + y)^2 ≤ 20 := sorry
  show max_value x y ≤ 2 * Real.sqrt 5 from sorry

end max_value_on_circle_l325_325083


namespace fifteenth_term_geometric_sequence_l325_325421

theorem fifteenth_term_geometric_sequence :
  let a1 := 5
  let r := (1 : ℝ) / 2
  let fifteenth_term := a1 * r^(14 : ℕ)
  fifteenth_term = (5 : ℝ) / 16384 := by
sorry

end fifteenth_term_geometric_sequence_l325_325421


namespace time_per_potato_l325_325798

-- Definitions from the conditions
def total_potatoes : ℕ := 12
def cooked_potatoes : ℕ := 6
def remaining_potatoes : ℕ := total_potatoes - cooked_potatoes
def total_time : ℕ := 36
def remaining_time_per_potato : ℕ := total_time / remaining_potatoes

-- Theorem to be proved
theorem time_per_potato : remaining_time_per_potato = 6 := by
  sorry

end time_per_potato_l325_325798


namespace probability_red_ball_l325_325581

theorem probability_red_ball :
  let bagA := {red := 1, yellow := 1}
  let bagB := {red := 2, yellow := 1}
  let total_outcomes := (bagA.red + bagA.yellow) * (bagB.red + bagB.yellow)
  let yellow_outcomes := bagA.yellow * bagB.yellow
  let prob := 1 - yellow_outcomes / total_outcomes
  prob = 5 / 6 :=
by
  let bagA := {red := 1, yellow := 1}
  let bagB := {red := 2, yellow := 1}
  let total_outcomes := (bagA.red + bagA.yellow) * (bagB.red + bagB.yellow)
  let yellow_outcomes := bagA.yellow * bagB.yellow
  let prob := 1 - yellow_outcomes / total_outcomes
  have h1 : total_outcomes = 6 := by sorry
  have h2 : yellow_outcomes = 1 := by sorry
  have h3 : prob = 1 - 1 / 6 := by sorry
  exact h3.trans (by norm_num)

end probability_red_ball_l325_325581


namespace wire_leftover_length_l325_325766

-- Define given conditions as variables/constants
def initial_wire_length : ℝ := 60
def side_length : ℝ := 9
def sides_in_square : ℕ := 4

-- Define the theorem: prove leftover wire length is 24 after creating the square
theorem wire_leftover_length :
  initial_wire_length - sides_in_square * side_length = 24 :=
by
  -- proof steps are not required, so we use sorry to indicate where the proof should be
  sorry

end wire_leftover_length_l325_325766


namespace number_of_papers_above_120_correct_l325_325185

noncomputable def number_of_papers_above_120
  (total_students : ℕ)
  (mean : ℝ)
  (variance : ℝ)
  (p_80_to_100 : ℝ)
  (total_samples : ℕ)
  : ℕ :=
  let p_above_120 := 0.5 * (1 - 2 * p_80_to_100)
  total_samples * p_above_120

theorem number_of_papers_above_120_correct :
  ∀ (total_students : ℕ) (mean variance : ℝ) (p_80_to_100 : ℝ) (total_samples : ℕ),
  total_students = 15000 →
  mean = 100 →
  variance > 0 →
  p_80_to_100 = 0.35 →
  total_samples = 100 →
  number_of_papers_above_120 total_students mean variance p_80_to_100 total_samples = 15 :=
by
  intros
  sorry

end number_of_papers_above_120_correct_l325_325185


namespace eq_or_neg_eq_of_eq_frac_l325_325543

theorem eq_or_neg_eq_of_eq_frac (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h : a^2 + b^3 / a = b^2 + a^3 / b) :
  a = b ∨ a = -b :=
by
  sorry

end eq_or_neg_eq_of_eq_frac_l325_325543


namespace sum_of_10_smallest_n_values_divisible_by_5_l325_325540

noncomputable def T_n (n : ℕ) : ℕ :=
  (n - 1) * n * (n + 1) * (3 * n + 2) / 24

def is_divisible_by_5 (x : ℕ) : Prop :=
  x % 5 = 0

def smallest_n_values_divisible_by_5 (num_values : ℕ) : ℕ :=
  (List.range (num_values * 5 + 2)).filter (λ n, n >= 2 ∧ is_divisible_by_5 (T_n n)).take num_values |>.sum

theorem sum_of_10_smallest_n_values_divisible_by_5 :
  smallest_n_values_divisible_by_5 10 = 265 :=
sorry

end sum_of_10_smallest_n_values_divisible_by_5_l325_325540


namespace S_infinite_and_not_all_primes_l325_325287

noncomputable def a : ℕ → ℕ
| 0       := 1
| (n + 1) := (a n) ^ 2 + n * (a n)

def is_in_S (p : ℕ) : Prop :=
  ∃ i : ℕ, p ∣ a i

def S : set ℕ := { p | is_in_S p }

theorem S_infinite_and_not_all_primes :
  infinite S ∧ ¬ ∀ p : ℕ, is_prime p → p ∈ S := sorry

end S_infinite_and_not_all_primes_l325_325287


namespace degree_condition_degree_difference_condition_number_of_cables_used_l325_325456

def computer_network (n : ℕ) :=
  {computers : finset ℕ // computers.card = n}

def independent_set (G : simple_graph ℕ) (S : finset ℕ) :=
  ∀ x y ∈ S, x ≠ y → ¬G.adj x y

variable (G : simple_graph ℕ)

-- There are 2004 computers in the network
def network_2004 : computer_network 2004 := sorry

-- The size of any independent set is at most 50
axiom independent_set_condition : ∀ S : finset ℕ, independent_set G S → S.card ≤ 50

-- The number of cables is minimized given the above conditions
def min_cables_used : ℕ := sorry

-- show conditions on degree of computers connected by a cable
theorem degree_condition (A B : ℕ) (hAB : G.adj A B) :
  G.degree A = G.degree B :=
sorry

theorem degree_difference_condition (A B : ℕ) (hAB : ¬ G.adj A B) :
  abs (G.degree A - G.degree B) ≤ 1 :=
sorry

-- find the number of cables used in the network
theorem number_of_cables_used (G.network_2004) (independent_set_condition) : 
  min_cables_used = 39160 :=
sorry

end degree_condition_degree_difference_condition_number_of_cables_used_l325_325456


namespace percent_decrease_april_to_may_l325_325736

theorem percent_decrease_april_to_may :
  ∃ (D : ℝ), (∀ (P : ℝ), 
  P * 1.20 * (1 - D / 100) * 1.50 = P * 1.4399999999999999) 
  ∧ D = 20 :=
by
  -- Define the assumption of the profit in March
  assume (P : ℝ),

  -- Use the condition of profit changes over months
  have h1 : P * 1.20 * (1 - D / 100) * 1.50 = P * 1.4399999999999999,
  
  -- Define the calculation of D
  have h2 : (1 - D / 100) = 1.4399999999999999 / (1.20 * 1.50),

  -- Solve for D
  have h3 : (1 - D / 100) = 0.7999999999999999,

  -- Final result of D
  have h4 : D / 100 = 0.2000000000000001,
  
  -- Convert to percent
  have h5 : D = 0.2000000000000001 * 100,
  
  -- Final conclusion
  exact ⟨20, by 
  ⟩


end percent_decrease_april_to_may_l325_325736


namespace train_cross_pole_in_5_seconds_l325_325774

/-- A train 100 meters long traveling at 72 kilometers per hour 
    will cross an electric pole in 5 seconds. -/
theorem train_cross_pole_in_5_seconds (L : ℝ) (v : ℝ) (t : ℝ) : 
  L = 100 → v = 72 * (1000 / 3600) → t = L / v → t = 5 :=
by
  sorry

end train_cross_pole_in_5_seconds_l325_325774


namespace nine_pointed_star_sum_of_angles_l325_325693

theorem nine_pointed_star_sum_of_angles (h : ∀i, ((i : ℤ) % 9) * 40 = 0) :
  ∑ (k : Fin 9), (1 / 2) * 4 * 40 = 720 := 
sorry

end nine_pointed_star_sum_of_angles_l325_325693


namespace sequences_satisfy_conditions_l325_325446

noncomputable def a (k : ℤ) : ℤ := 24 * k^2 + 12 * k + 14
noncomputable def b (k : ℤ) : ℤ := 16 * k^3 - 2
noncomputable def c (k : ℤ) : ℤ := -16 * k^3 - 24 * k^2 - 12 * k - 4
def d (k : ℤ) : ℤ := (a k)^2 + (b k)^2 + (c k)^2 

theorem sequences_satisfy_conditions (k : ℤ) :
  a k + b k + c k = 8 ∧ (∃ m : ℤ, m^3 = d k) := 
by sorry

end sequences_satisfy_conditions_l325_325446


namespace find_am_of_sum_conditions_l325_325086

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Definition of an arithmetic sequence sum
noncomputable def Sn (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem find_am_of_sum_conditions
  (a₁ d : ℝ)
  (m : ℕ)
  (h₁ : Sn a₁ d (m - 2) = -4)
  (h₂ : Sn a₁ d m = 0)
  (h₃ : Sn a₁ d (m + 2) = 12) :
  let a_m := a₁ + (m - 1) * d in
  a_m = 3 :=
by
  sorry

end find_am_of_sum_conditions_l325_325086


namespace oil_depth_correct_l325_325469

noncomputable def oil_depth_in_upright_tank : ℝ :=
  let tank_height := 12
  let tank_diameter := 6
  let radius := (tank_diameter : ℝ) / 2
  let lying_depth := 2
  let chord_height := radius - lying_depth
  let theta := 2 * real.arccos (chord_height / radius)
  let segment_area := (radius^2 / 2) * (theta - real.sin theta)
  let base_area := real.pi * radius^2
  let fraction_oil := segment_area / base_area
  let upright_depth := tank_height * fraction_oil
  2.4

theorem oil_depth_correct :
  let tank_height := 12
  let tank_diameter := 6
  let radius := (tank_diameter : ℝ) / 2
  let lying_depth := 2
  let chord_height := radius - lying_depth
  let theta := 2 * real.arccos (chord_height / radius)
  let segment_area := (radius^2 / 2) * (theta - real.sin theta)
  let base_area := real.pi * radius^2
  let fraction_oil := segment_area / base_area
  let upright_depth := tank_height * fraction_oil
  upright_depth ≈ 2.4 :=
by
  sorry

end oil_depth_correct_l325_325469


namespace lines_perpendicular_iff_equal_sides_l325_325189

-- Definitions of points, midpoints, centroids, and perpendicularity
variables {A B C O D E: Type} [EuclideanSpace ℝ ℝ]
variables (A B C O D E: ℝ)
variables [C.ircumcenter A B C = O]
variables [D.midpoint A B = D]
variables [E.centroid A C D = E]

-- Proof statement
theorem lines_perpendicular_iff_equal_sides :
  (CD ⊥ OE) ↔ (dist A B = dist A C) :=
sorry

end lines_perpendicular_iff_equal_sides_l325_325189


namespace circle_rect_eq_of_polar_distance_from_center_to_line_l325_325205

-- Problem (I): Proving the rectangular equation of circle C
theorem circle_rect_eq_of_polar (ρ θ : ℝ) (h : ρ = 2 * cos θ) :
  let x := ρ * cos θ
  let y := ρ * sin θ
  in x^2 + y^2 - 2 * x = 0 := by
  sorry

-- Problem (II): Proving the distance between the center of circle C and line l
theorem distance_from_center_to_line (A B C x0 y0 : ℝ) (hA : A = 2) (hB : B = -1) (hC : C = 1) (hx0 : x0 = 1) (hy0 : y0 = 0) :
  let d := abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)
  in d = 3 / sqrt 5 := by
  sorry

end circle_rect_eq_of_polar_distance_from_center_to_line_l325_325205


namespace solve_exp_log_equation_l325_325364

theorem solve_exp_log_equation
  (a b α : ℝ) (ha : 0 < a) (hb : 0 < b) (hb1 : b ≠ 1) (hα : 0 < α) :
  ∃ x : ℝ, (a ^ (Real.log x / Real.log (Real.sqrt b)) - 5 * a ^ (Real.log α / Real.log b) + 6 = 0)
          ∧ (x = b ^ (Real.log 3 / Real.log a) ∨ x = b ^ (Real.log 2 / Real.log a)) :=
begin
  sorry
end.

end solve_exp_log_equation_l325_325364


namespace simple_annual_interest_rate_l325_325485

theorem simple_annual_interest_rate (monthly_interest : ℝ) (principal : ℝ) :
  (monthly_interest = 240) → (principal = 32000) → 
  (∃ R : ℝ, (monthly_interest * 12) = (principal * R * 1) ∧ R = 0.09) :=
begin
  intros h1 h2,
  sorry
end

end simple_annual_interest_rate_l325_325485


namespace area_of_polygon_is_14_cm2_l325_325020

noncomputable def polygon_vertices : List (ℝ × ℝ) := 
  [(1,2), (1,3), (2,4), (3,5), (4,6), (5,5), (6,5), (7,4), (7,3), (6,2), (5,1), (4,1), (3,1), (2,2)]

theorem area_of_polygon_is_14_cm2 : area_of_polygon polygon_vertices = 14 := 
by
  sorry

end area_of_polygon_is_14_cm2_l325_325020


namespace solve_problem_range_m_l325_325915

noncomputable def function_expression (ω : ℝ) (ϕ : ℝ) : ℝ → ℝ :=
  λ (x : ℝ), 2 * sin (ω * x + ϕ)

theorem solve_problem (ϕ : ℝ) (ω : ℝ) (x₁ x₂ : ℝ) (fx₁ fx₂ : ℝ)
  (cond1 : ω > 0)
  (cond2 : -π / 2 < ϕ ∧ ϕ < 0)
  (cond3 : tan ϕ = -√3)
  (cond4 : ∀ (x: ℝ), fx₁ = function_expression ω ϕ x₁ ∧ fx₂ = function_expression ω ϕ x₂)
  (cond5 : |fx₁ - fx₂| = 4)
  (cond6 : |x₁ - x₂| = π / 3) :
  ∃ (ω' : ℝ) (ϕ' : ℝ), ω' = 3 ∧ ϕ' = -π / 3 ∧ function_expression 3 (-π / 3) = function_expression ω ϕ :=
sorry

theorem range_m (m : ℝ)
  (cond_a : m * function_expression 3 (-π / 3) + 2 * m ≥ function_expression 3 (-π / 3))
  (cond_b : ∀ x, 0 ≤ x ∧ x ≤ π / 6) :
  m ≥ 1 / 3 :=
sorry

end solve_problem_range_m_l325_325915


namespace total_number_of_questions_l325_325184

/-
  Given:
    1. There are 20 type A problems.
    2. Type A problems require twice as much time as type B problems.
    3. 32.73 minutes are spent on type A problems.
    4. Total examination time is 3 hours.

  Prove that the total number of questions is 199.
-/

theorem total_number_of_questions
  (type_A_problems : ℕ)
  (type_B_to_A_time_ratio : ℝ)
  (time_spent_on_type_A : ℝ)
  (total_exam_time_hours : ℝ)
  (total_number_of_questions : ℕ)
  (h_type_A_problems : type_A_problems = 20)
  (h_time_ratio : type_B_to_A_time_ratio = 2)
  (h_time_spent_on_type_A : time_spent_on_type_A = 32.73)
  (h_total_exam_time_hours : total_exam_time_hours = 3) :
  total_number_of_questions = 199 := 
sorry

end total_number_of_questions_l325_325184


namespace simplify_tan_expression_l325_325342

theorem simplify_tan_expression 
  (h30 : Real.tan (π / 6) = 1 / Real.sqrt 3)
  (h15 : Real.tan (π / 12) = 2 - Real.sqrt 3) :
  (1 + Real.tan (π / 6)) * (1 + Real.tan (π / 12)) = 2 :=
by
  -- State the tangent addition formula for the required angles
  have h_tan_add : Real.tan (π / 4) = (Real.tan (π / 6) + Real.tan (π / 12)) / (1 - Real.tan (π / 6) * Real.tan (π / 12)),
  {
    sorry,
  }
  -- The correct answer proof part is not provided in the brief
  sorry

end simplify_tan_expression_l325_325342


namespace circle_region_count_l325_325233

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l325_325233


namespace m_p_relation_l325_325785

variable (a b c d e : ℝ)
def m := (a + b + c + d + e) / 5
def k := (a + b) / 2
def l := (c + d + e) / 3
def p := (k + l) / 2

theorem m_p_relation : m = p ∨ m > p ∨ m < p :=
by sorry

end m_p_relation_l325_325785


namespace circle_regions_division_l325_325253

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l325_325253


namespace range_alpha_beta_l325_325302

noncomputable def centroid (A B C : Point) : Point :=
(sorry : Point) -- This would be the formal definition of centroid of triangle ABC

theorem range_alpha_beta {A D E G B C P : Point} 
  (hG : G = centroid A D E)
  (PB : P ∈ triangle D E G)
  (hB : B = (2/3) • A + (1/3) • D)
  (hC : C = (2/3) • A + (1/3) • E) 
  (hP : P = α • B + β • C)
  : ∃ α β : ℝ, α + (1/2) * β ∈ set.Icc (3/2) 3 :=
begin
  sorry
end

end range_alpha_beta_l325_325302


namespace event_relationship_l325_325326

def die_events : set ℕ := {1, 2, 3, 4, 5, 6}
def event_A : set ℕ := {n | n ∈ die_events ∧ n % 2 = 1}
def event_B : set ℕ := {n | n = 3 ∨ n = 4}

lemma not_mutually_exclusive : ∃ n ∈ die_events, n ∈ event_A ∧ n ∈ event_B :=
by {
  use 3,
  split,
  { unfold die_events, finish, },
  { unfold event_A, unfold event_B, finish, }
}

lemma independent_events : independent_events event_A event_B :=
by {
  sorry
}

theorem event_relationship : ¬ (disjoint event_A event_B) ∧ independent_events event_A event_B :=
by {
  split,
  { exact not_mutually_exclusive, },
  { exact independent_events, }
}


end event_relationship_l325_325326


namespace range_of_a_l325_325877

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l325_325877


namespace number_of_integers_satisfying_inequalities_l325_325510

theorem number_of_integers_satisfying_inequalities :
  let sol1 := λ x : ℤ, -4 * x ≥ 2 * x + 10
  let sol2 := λ x : ℤ, 3 * x ≤ 18
  let sol3 := λ x : ℤ, -5 * x ≥ x - 8
  (∃ (x : ℤ), sol1 x ∧ sol2 x ∧ sol3 x ∧ x = 0) ∧ 
  (∃ (x : ℤ), sol1 x ∧ sol2 x ∧ sol3 x ∧ x = -1) ∧
  ∀ x : ℤ, (sol1 x ∧ sol2 x ∧ sol3 x) → (x = 0 ∨ x = -1) :=
by
  sorry

end number_of_integers_satisfying_inequalities_l325_325510


namespace recurring_fraction_sum_l325_325160

theorem recurring_fraction_sum (a b : ℕ) (h : 0.36̅ = ↑a / ↑b) (gcd_ab : Nat.gcd a b = 1) : a + b = 15 :=
sorry

end recurring_fraction_sum_l325_325160


namespace exists_k_swap_thimbles_l325_325007

theorem exists_k_swap_thimbles (n : Nat) (hn : n = 2021)
  (arr : Fin n → Fin n) (harr : ∀ i, (arr i).val < n)
  (moves : Fin n → Fin n × Fin n)
  (hmoves : ∀ k, moves k = (if k = 0 then (Fin.ofNat (n - 1), Fin.ofNat 1) else 
                             if k = n - 1 then (Fin.ofNat (k - 1), Fin.ofNat 0)
                             else (Fin.ofNat (k - 1), Fin.ofNat (k + 1)))) :
  ∃ k, let (a, b) := moves k in a.val < k ∧ k < b.val :=
by {
  sorry
}

end exists_k_swap_thimbles_l325_325007


namespace recurring_fraction_sum_l325_325158

theorem recurring_fraction_sum (a b : ℕ) (h : 0.36̅ = ↑a / ↑b) (gcd_ab : Nat.gcd a b = 1) : a + b = 15 :=
sorry

end recurring_fraction_sum_l325_325158


namespace tan_product_simplification_l325_325353

theorem tan_product_simplification :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  have h : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have tan_addition :
    ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) := Real.tan_add
  sorry

end tan_product_simplification_l325_325353


namespace sum_of_angles_nine_pointed_star_l325_325697

open Real

-- Definitions
def points_on_circle (n : ℕ) (circle : Set Point) : Prop :=
  ∀ (i j : ℕ), i ≠ j → dist (circle_point i) (circle_point j) = (360 / n) * abs (i - j)

def nine_pointed_star (circle : Set Point) : Prop :=
  points_on_circle 9 circle ∧ is_9_pointed_star circle

-- Theorem statement
theorem sum_of_angles_nine_pointed_star {circle : Set Point} (h : nine_pointed_star circle) :
  ∑ i in finset.range 9, angle (tip i) = 720 :=
sorry

end sum_of_angles_nine_pointed_star_l325_325697


namespace cylindrical_tin_volume_l325_325442

noncomputable theory

def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

theorem cylindrical_tin_volume :
  let d := 14 in
  let h := 5 in
  let r := d / 2 in
  abs (cylinder_volume r h - 769.69) < 0.01 :=
by
  sorry

end cylindrical_tin_volume_l325_325442


namespace X_stationary_l325_325123

noncomputable def X (t : ℝ) (ϕ : ℝ) : ℝ := Real.cos (t + ϕ)

def φ_dist : MeasureTheory.Measure ℝ := MeasureTheory.Measure.uniform 0 (2 * Real.pi)

noncomputable def E_X (t : ℝ) : ℝ := 
  ∫ ϕ in 0..(2 * Real.pi), X t ϕ ∂φ_dist.toFinMeas

def R_X (t1 t2 : ℝ) : ℝ := 
  (∫ ϕ in 0..(2 * Real.pi), X t1 ϕ * X t2 ϕ ∂φ_dist.toFinMeas) / (2 * Real.pi)

theorem X_stationary : 
  (∀ t : ℝ, E_X t = 0) ∧ (∀ t1 t2 : ℝ, R_X t1 t2 = (Real.cos (t1 - t2) / 2)) :=
by
  sorry

end X_stationary_l325_325123


namespace person_A_work_days_l325_325704

theorem person_A_work_days (A : ℕ) (h1 : ∀ (B : ℕ), B = 45) (h2 : 4 * (1/A + 1/45) = 2/9) : A = 30 := 
by
  sorry

end person_A_work_days_l325_325704


namespace geometric_sequence_properties_l325_325580

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q: ℝ, q > 0 ∧ ∀ n: ℕ, a (n + 1) = a n * q

noncomputable def increasing (a : ℕ → ℝ) : Prop :=
  ∀ n: ℕ, a (n + 1) > a n

theorem geometric_sequence_properties 
  (a : ℕ → ℝ) 
  (h_geom: geometric_sequence a) 
  (h_incr: increasing a)
  (h_a2: a 2 = 2)
  (h_a4_minus_a3: a 4 - a 3 = 4) : 
  a 1 = 1 ∧ ∃ q: ℝ, q = 2 ∧ a 5 = 16 ∧ (∑ i in finset.range 5, a (i + 1)) = 31 :=
by
  sorry

end geometric_sequence_properties_l325_325580


namespace area_of_triangle_given_conditions_l325_325213

noncomputable def area_triangle_ABC (a b B : ℝ) : ℝ :=
  0.5 * a * b * Real.sin B

theorem area_of_triangle_given_conditions :
  area_triangle_ABC 2 (Real.sqrt 3) (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end area_of_triangle_given_conditions_l325_325213


namespace area_of_triangle_l325_325867

-- Define the vertices
def A : ℝ³ := (0, 3, 6)
def B : ℝ³ := (-2, 2, 2)
def C : ℝ³ := (-5, 5, 2)

-- Define the distances (sides of the triangle)
def AB : ℝ := Real.sqrt ((0 - (-2))^2 + (3 - 2)^2 + (6 - 2)^2)
def AC : ℝ := Real.sqrt ((0 - (-5))^2 + (3 - 5)^2 + (6 - 2)^2)
def BC : ℝ := Real.sqrt ((-2 - (-5))^2 + (2 - 5)^2 + (2 - 2)^2)

-- Define the semi-perimeter
noncomputable def s : ℝ := (AB + AC + BC) / 2

-- Define Heron's formula for the area
noncomputable def area : ℝ := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

-- The theorem to be proved
theorem area_of_triangle : area = Real.sqrt ((s * (s - AB) * (s - AC) * (s - BC)) : ℝ) :=
by sorry

end area_of_triangle_l325_325867


namespace children_tickets_count_l325_325474

theorem children_tickets_count (A C : ℕ) (h1 : 8 * A + 5 * C = 201) (h2 : A + C = 33) : C = 21 :=
by
  sorry

end children_tickets_count_l325_325474


namespace problem_statement_l325_325366

noncomputable def polynomial := (polynomial.X^4 - 6 * polynomial.X^3 + 23 * polynomial.X^2 - 72 * polynomial.X + 8 : polynomial ℝ)

theorem problem_statement (p q r s : ℝ) (hpqr : p * q * r * s = 8)
  (sum_pairs : ∑ i in {p, q, r, s}.powerset.filter (λ t, t.card = 2), t.prod id = 72)
  (roots : polynomial.roots = {p, q, r, s}) :
  (1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s)) = -9 :=
by sorry

end problem_statement_l325_325366


namespace correct_second_number_l325_325371

theorem correct_second_number 
    (avg_incorrect : Float) (sum_incorrect : Float) (n : Int) 
    (first_incorrect_difference : Float) (second_incorrect : Float) 
    (avg_correct : Float) (correct_sum_diff : Float) :
    avg_incorrect = 40.2 ->
    sum_incorrect = n * avg_incorrect ->
    n = 10 ->
    first_incorrect_difference = 16 ->
    second_incorrect = 13 ->
    avg_correct = 40 ->
    correct_sum_diff = 0 ->
    ∃ (second_correct : Float), 
      let correct_sum := sum_incorrect - first_incorrect_difference - second_incorrect + second_correct in
      correct_sum = n * avg_correct ∧ second_correct = 27 := 
by
  intros
  sorry

end correct_second_number_l325_325371


namespace factor_complete_polynomial_l325_325050

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end factor_complete_polynomial_l325_325050


namespace sqrt_3a_4b_eq_pm4_l325_325105

variable (a b : ℝ)

theorem sqrt_3a_4b_eq_pm4
  (h1 : sqrt (2 * a + 1) = 3 ∨ sqrt (2 * a + 1) = -3)
  (h2 : sqrt (5 * a + 2 * b - 2) = 4) :
  sqrt (3 * a - 4 * b) = 4 ∨ sqrt (3 * a - 4 * b) = -4 :=
by
  sorry

end sqrt_3a_4b_eq_pm4_l325_325105


namespace fraction_order_l325_325432

theorem fraction_order:
  let frac1 := (21 : ℚ) / 17
  let frac2 := (22 : ℚ) / 19
  let frac3 := (18 : ℚ) / 15
  let frac4 := (20 : ℚ) / 16
  frac2 < frac3 ∧ frac3 < frac1 ∧ frac1 < frac4 := 
sorry

end fraction_order_l325_325432


namespace induction_step_l325_325708

theorem induction_step (x y : ℕ) (k : ℕ) (odd_k : k % 2 = 1) 
  (hk : (x + y) ∣ (x^k + y^k)) : (x + y) ∣ (x^(k+2) + y^(k+2)) :=
sorry

end induction_step_l325_325708


namespace harmonic_mean_pairs_count_l325_325541

theorem harmonic_mean_pairs_count :
  let x y : ℕ := sorry
  (0 < x) ∧ (0 < y) ∧ (x < y) ∧ (2*x*y / (x + y) = 8^20) → (x * y - 4^20 * (x + y) = 0) →
  (x - 4^20) * (y - 4^20) = 4^40 → -- Condition derived from Simon's Favorite Factoring Trick
  (filter (λ p : ℕ × ℕ, p.1 < p.2) (list.range (nat.succ 80)).length / 2 = 40) :=
sorry

end harmonic_mean_pairs_count_l325_325541


namespace sum_of_angles_of_9_pointed_star_is_540_l325_325694

-- Define the circle with nine evenly spaced points.
def circle_with_nine_points := { p : ℝ // 0 <= p ∧ p < 360 }

-- Define a 9-pointed star formed by connecting these nine points.
def nine_pointed_star (points : fin 9 → circle_with_nine_points) : Prop :=
  ∀ i : fin 9, points i = ⟨ i.1 * 40, sorry ⟩

-- Define the sum of the angle measurements at the tips of the 9-pointed star.
def sum_of_tip_angles (points : fin 9 → circle_with_nine_points) : ℝ :=
  (∑ i in finset.univ, 60)

-- Statement to be proved: 
theorem sum_of_angles_of_9_pointed_star_is_540 : ∀ points : fin 9 → circle_with_nine_points,  
  nine_pointed_star points → sum_of_tip_angles points = 540 := 
by
  intros points h
  sorry

end sum_of_angles_of_9_pointed_star_is_540_l325_325694


namespace difference_in_biking_distance_l325_325480

def biking_rate_alberto : ℕ := 18  -- miles per hour
def biking_rate_bjorn : ℕ := 20    -- miles per hour

def start_time_alberto : ℕ := 9    -- a.m.
def start_time_bjorn : ℕ := 10     -- a.m.

def end_time : ℕ := 15            -- 3 p.m. in 24-hour format

def biking_duration_alberto : ℕ := end_time - start_time_alberto
def biking_duration_bjorn : ℕ := end_time - start_time_bjorn

def distance_alberto : ℕ := biking_rate_alberto * biking_duration_alberto
def distance_bjorn : ℕ := biking_rate_bjorn * biking_duration_bjorn

theorem difference_in_biking_distance : 
  (distance_alberto - distance_bjorn) = 8 := by
  sorry

end difference_in_biking_distance_l325_325480


namespace count_real_numbers_a_with_integer_roots_l325_325884

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l325_325884


namespace ak_perpendicular_lm_l325_325667

variables (Γ : Type) [circle Γ]
variables (O : point Γ) (A B C : point Γ) (H K L M : point Γ)

-- Conditions
def triangle_inscribed (Γ : circle) (A B C : point Γ) : Prop := inscribed_in_circle A B C Γ
def orthocenter (A B C H : point Γ) : Prop := orthocenter_of_triangle A B C = H
def midpoint (K : point Γ) (O H : point Γ) : Prop := midpoint K O H
def tangent_and_bisector_intersect (Γ : circle)
  (B C : point Γ) (L M : point Γ) : Prop :=
  (tangent_to_circle_at_point B Γ).intersect (perpendicular_bisector_of_base_and_median A C) = L &
  (tangent_to_circle_at_point C Γ).intersect (perpendicular_bisector_of_base_and_median A B) = M

-- Problem statement
theorem ak_perpendicular_lm 
  (triangle_inscribed Γ A B C)
  (orthocenter A B C H)
  (midpoint K O H)
  (tangent_and_bisector_intersect Γ B C L M) : 
  ⊥ ℝ (line_through_points A K) (line_through_points L M) :=
sorry

end ak_perpendicular_lm_l325_325667


namespace digits_in_base_5_l325_325954

theorem digits_in_base_5 (n : ℕ) (h : n = 1234) (h_largest_power : 5^4 < n ∧ n < 5^5) : 
  ∃ digits : ℕ, digits = 5 := 
sorry

end digits_in_base_5_l325_325954


namespace initial_percentage_of_alcohol_l325_325787

-- Define necessary conditions
variables (P : ℝ)
variables (initial_volume : ℝ) (added_alcohol : ℝ) (added_water : ℝ) (final_volume : ℝ) (final_percentage : ℝ)

-- The problem conditions
def conditions : Prop := 
  initial_volume = 40 ∧
  added_alcohol = 5.5 ∧
  added_water = 4.5 ∧
  final_volume = 50 ∧
  final_percentage = 15

-- The statement to prove
theorem initial_percentage_of_alcohol (P : ℝ) (h : conditions P initial_volume added_alcohol added_water final_volume final_percentage)
  : P = 5 :=
by {
  -- Unwrap the conditions
  use [40, 5.5, 4.5, 50, 15],
  simp [conditions] at h,
  -- Establish the equation (P / 100) * 40 + 5.5 = 7.5
  have eq1 : (P / 100) * 40 + 5.5 = 7.5 := sorry,
  -- Solve for P
  have eq2 : P * 40 + 550 = 750 := by calc
    (P / 100) * 40 + 5.5 = 7.5 : eq1
    ... == P * .1d),
  rw add_comm at eq2, 
  simp [eq1] at eq2,
  sorry
}

end initial_percentage_of_alcohol_l325_325787


namespace expected_value_of_length_l325_325445

noncomputable def expectedLength : ℝ :=
  let pH : ℝ := 1 / 2
  let pM : ℝ := 1 / 4
  let pT : ℝ := 1 / 4
  let q : ℝ := pM ^ 2 + 2 * pM * (pH + pT) + (pH + pT) ^ 2
  let E : ℝ := (2 * pM + (pH + pT) * (expectedLength + 1) + pM * (pH + pT) * (expectedLength + 2)) / q
  E

theorem expected_value_of_length : expectedLength = 6 := 
  sorry

end expected_value_of_length_l325_325445


namespace circle_region_count_l325_325231

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l325_325231


namespace isosceles_triangle_perimeter_l325_325990

/-- 
  Given an isosceles triangle with two sides of length 6 and the third side of length 2,
  prove that the perimeter of the triangle is 14.
-/
theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 6) (h3 : c = 2) 
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a + b + c = 14 :=
  sorry

end isosceles_triangle_perimeter_l325_325990


namespace tooth_extraction_cost_l325_325754

variable (c f b e : ℕ)

-- Conditions
def cost_cleaning := c = 70
def cost_filling := f = 120
def bill := b = 5 * f

-- Proof Problem
theorem tooth_extraction_cost (h_cleaning : cost_cleaning c) (h_filling : cost_filling f) (h_bill : bill b f) :
  e = b - (c + 2 * f) :=
sorry

end tooth_extraction_cost_l325_325754


namespace deer_initial_money_l325_325841

theorem deer_initial_money :
  ∃ y : ℚ, 
    (let y1 := 3 * y - 50 in 
    let y2 := 3 * y1 - 50 in 
    let y3 := 4 * y2 - 50 in 
    y3 = 0) ∧ y = 425 / 18 :=
by
  sorry

end deer_initial_money_l325_325841


namespace find_missing_distance_l325_325392

noncomputable def distances (d1 d2 d3 d4 d5 d6 : ℕ) : Prop :=
  ∃ (A B C D : ℕ), ∀ (d ∈ {d1, d2, d3, d4, d5, d6}), 
    d ∈ {abs (A - B), abs (A - C), abs (A - D), abs (B - C), abs (B - D), abs (C - D)}

theorem find_missing_distance :
  distances 2 5 7 17 22 24 :=
sorry

end find_missing_distance_l325_325392


namespace circle_regions_division_l325_325249

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l325_325249


namespace quadratic_decomposition_l325_325378

theorem quadratic_decomposition : ∀ y : ℝ, y^2 + 14*y + 60 = (y + 7)^2 + 11 := 
by {
  intro y,
  sorry
}

end quadratic_decomposition_l325_325378


namespace cash_realized_before_brokerage_l325_325724

theorem cash_realized_before_brokerage (C : ℝ) (h1 : 0.25 / 100 * C = C / 400)
(h2 : C - C / 400 = 108) : C = 108.27 :=
by
  sorry

end cash_realized_before_brokerage_l325_325724


namespace tan_seq_zero_l325_325167

theorem tan_seq_zero (x : ℝ) (h_seq : cos x ^ 2 = sin x * (cos x / sin x)) (h_cos : cos x = 1) : 
  (tan x) ^ 6 - (tan x) ^ 2 = 0 := 
by
  sorry

end tan_seq_zero_l325_325167


namespace trapezoid_median_median_of_trapezoid_is_12_l325_325011

theorem trapezoid_median {h x : ℝ} :
  let triangle_area := 12 * h,
      trapezoid_area := (x + 5) * h in
  (triangle_area = trapezoid_area) → x = 7 := 
by
  sorry

theorem median_of_trapezoid_is_12 {h x : ℝ} (h_ne_zero : h ≠ 0) :
  let triangle_area := 12 * h,
      trapezoid_area := (x + 5) * h in
  (triangle_area = trapezoid_area) → let median := (x + (x + 10)) / 2 in median = 12 :=
by
  intros h_eq
  apply trapezoid_median.mp h_eq
  sorry

end trapezoid_median_median_of_trapezoid_is_12_l325_325011


namespace carl_watermelons_left_l325_325497

-- Define the conditions
def price_per_watermelon : ℕ := 3
def profit : ℕ := 105
def starting_watermelons : ℕ := 53

-- Define the main proof statement
theorem carl_watermelons_left :
  (starting_watermelons - (profit / price_per_watermelon) = 18) :=
sorry

end carl_watermelons_left_l325_325497


namespace circles_radii_divide_regions_l325_325264

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l325_325264


namespace AD_parallel_OC_AD_OC_value_l325_325645

-- Define the problem's conditions
variables {O O' : Eucl.Point} {A B C D : Eucl.Point}
variables (r : ℝ) (h : r = 1)

-- Assume AB is the diameter of circle O and BC, CD are tangents to circle O'
def AB_is_diameter (A B O : Eucl.Point) : Prop := Eucl.colinear A B O ∧ Eucl.dist A O = Eucl.dist O B

def tangents_BC_CD (B C D O' : Eucl.Point) : Prop := 
  Eucl.tangent B C O' ∧ Eucl.tangent C D O'

-- Part 1: Showing AD parallel to OC
theorem AD_parallel_OC
  (h1 : AB_is_diameter A B O)
  (h2 : tangents_BC_CD B C D O')
  : Eucl.parallel (Eucl.line_through A D) (Eucl.line_through O C) := 
sorry

-- Part 2: Given radius of circle O = 1, prove AD·OC = 2
theorem AD_OC_value
  (h1 : AB_is_diameter A B O)
  (h2 : tangents_BC_CD B C D O')
  (h3 : Eucl.dist O A = 1)
  : Eucl.dist A D * Eucl.dist O C = 2 := 
sorry

end AD_parallel_OC_AD_OC_value_l325_325645


namespace correct_factorization_A_l325_325430

theorem correct_factorization_A (x : ℝ) : x^2 - 4 * x + 4 = (x - 2)^2 :=
by sorry

end correct_factorization_A_l325_325430


namespace lambda_inequality_l325_325087

noncomputable def a (n : ℕ) : ℕ := 2 * n
noncomputable def b (n : ℕ) : ℕ := 2 ^ (n - 1)
def c (n : ℕ) (λ : ℝ) : ℝ := 2 * b n - λ * (3 : ℝ) ^ (a n / 2 : ℝ)

theorem lambda_inequality (λ : ℝ) :
  (∀ n : ℕ, n > 0 → c (n + 1) λ < c n λ) → λ > 1 / 3 :=
sorry

end lambda_inequality_l325_325087


namespace max_parts_divided_by_three_planes_l325_325406

theorem max_parts_divided_by_three_planes : max_parts 3 = 8 :=
by
  -- Assume conditions as definitions (hypotheses)
  let two_planes_divide := 4
  let each_part_divided_by_third_plane := 2
  sorry

end max_parts_divided_by_three_planes_l325_325406


namespace c_share_is_correct_l325_325515

-- Define the total amount to be divided
def totalAmount : ℝ := 5000

-- Define the parts in the ratio for A, B, C, D
def partA : ℝ := 1
def partB : ℝ := 3
def partC : ℝ := 5
def partD : ℝ := 7

-- Define the total parts
def totalParts : ℝ := partA + partB + partC + partD

-- Define the value of each part
def valuePerPart : ℝ := totalAmount / totalParts

-- Define the share of C
def cShare : ℝ := valuePerPart * partC

-- Prove that C's share is $1562.50
theorem c_share_is_correct : cShare = 1562.50 := by
  sorry

end c_share_is_correct_l325_325515


namespace intersection_of_M_and_N_l325_325945

def M : Set ℝ := { x | x ≤ 0 }
def N : Set ℝ := { -2, 0, 1 }

theorem intersection_of_M_and_N : M ∩ N = { -2, 0 } := 
by
  sorry

end intersection_of_M_and_N_l325_325945


namespace final_result_repeated_subtraction_sum_digits_l325_325900

-- Definition of sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Problem statement in Lean 4
theorem final_result_repeated_subtraction_sum_digits (n : ℕ) :
  ∃ k, (iterate (λ x, x - sum_of_digits x) k n) = 0 :=
by
  sorry

end final_result_repeated_subtraction_sum_digits_l325_325900


namespace ratio_of_radii_l325_325471

theorem ratio_of_radii (R r a : ℝ) (A B C M N K L: Point) (circle_large circle_small: Circle) 
  (h_inscribe : InscribedInTriangle circle_large A B C) 
  (h_concentric : ConcentricCircles circle_large circle_small)
  (h_touch : TouchesSide circle_small A B) 
  (h_divides_BC : DividesEqually circle_small B C 3 M N)
  (h_divides_AC : DividesEqually circle_small A C 3 K L) :
  r / R = 5 / 9 :=
sorry

end ratio_of_radii_l325_325471


namespace doubles_tournament_handshakes_l325_325829

theorem doubles_tournament_handshakes :
  let num_teams := 3
  let players_per_team := 2
  let total_players := num_teams * players_per_team
  let handshakes_per_player := total_players - 2
  let total_handshakes := total_players * handshakes_per_player / 2
  total_handshakes = 12 :=
by
  sorry

end doubles_tournament_handshakes_l325_325829


namespace coloring_count_correct_l325_325041

def proper_divisors (n : ℕ) : List ℕ := 
  List.filter (λ d, d ≠ n ∧ n % d = 0) (List.range' 1 n)

def valid_coloring (f : ℕ → ℕ) : Prop :=
  (∀ n ∈ (List.range' 2 11), 
    f n ≠ 0 ∧ f n ≤ 3 ∧ (∀ d ∈ proper_divisors n, f n ≠ f d))

theorem coloring_count_correct :
  (∃ (f : ℕ → ℕ), valid_coloring f) →
  (3^5 * 2^3 * 1^4 = 5832) :=
by
  sorry

end coloring_count_correct_l325_325041


namespace num_real_values_for_integer_roots_l325_325881

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l325_325881


namespace X_independent_iff_independent_from_prefix_l325_325335

variables {Ω : Type*} {X : ℕ → Ω → ℝ}

def independent_system (X : ℕ → Ω → ℝ) : Prop := 
  ∀ s : finset ℕ, ∀ A : Π i ∈ s, set (Ω → ℝ),
  probability_theory.indep_set (λ i, X i) s A 

def independent_from_prefix (X : ℕ → Ω → ℝ) (n : ℕ) : Prop := 
  ∀ A_n B_{n-1}, 
  probability_theory.indep_set (X n) (λ i, X i) (fin.range (n-1)) (λ _, set.univ)

theorem X_independent_iff_independent_from_prefix : 
  independent_system X ↔ ∀ n ≥ 1, independent_from_prefix X n :=
sorry

end X_independent_iff_independent_from_prefix_l325_325335


namespace how_many_toys_l325_325286

theorem how_many_toys (initial_savings : ℕ) (allowance : ℕ) (toy_cost : ℕ)
  (h1 : initial_savings = 21)
  (h2 : allowance = 15)
  (h3 : toy_cost = 6) :
  (initial_savings + allowance) / toy_cost = 6 :=
by
  sorry

end how_many_toys_l325_325286


namespace trapezium_area_is_correct_l325_325053

noncomputable def trapezium_area (a b h : ℝ) : ℝ :=
  1 / 2 * (a + b) * h

theorem trapezium_area_is_correct :
  let a := 20
  let b := 18
  let c := 15
  let theta := Real.pi / 3 -- 60 degrees in radians
  let h := c * Real.sin theta
  trapezium_area a b h = 285 * Real.sqrt 3 / 2 :=
by
  -- Definitions
  let a := 20
  let b := 18
  let c := 15
  let theta := Real.pi / 3 -- 60 degrees in radians
  have h_def : h = c * Real.sin theta := by sorry
  have area_def : trapezium_area a b h = (1 / 2 * (a + b) * h) := by sorry
  -- Calculation
  rw [h_def, area_def]
  sorry

end trapezium_area_is_correct_l325_325053


namespace product_value_l325_325839

theorem product_value : 
  (∏ n in finset.range (2020), (1 - (1 / (n + 2)^2))) = 1011 / 2021 :=
by
  sorry

end product_value_l325_325839


namespace average_temperature_l325_325831

def highTemps : List ℚ := [51, 60, 56, 55, 48, 63, 59]
def lowTemps : List ℚ := [42, 50, 44, 43, 41, 46, 45]

def dailyAverage (high low : ℚ) : ℚ :=
  (high + low) / 2

def averageOfAverages (tempsHigh tempsLow : List ℚ) : ℚ :=
  (List.sum (List.zipWith dailyAverage tempsHigh tempsLow)) / tempsHigh.length

theorem average_temperature :
  averageOfAverages highTemps lowTemps = 50.2 :=
  sorry

end average_temperature_l325_325831


namespace smallest_positive_angle_l325_325847

theorem smallest_positive_angle (x : ℝ) (hx : x > 0) :
  (tan (5 * x) = (1 - sin x) / (1 + sin x)) ↔ x = 9 * Real.pi / 180 :=
by
  sorry

end smallest_positive_angle_l325_325847


namespace accommodation_ways_l325_325898

theorem accommodation_ways (people rooms : ℕ) (h1 : people = 4) (h2 : rooms = 3) (h3 : rooms ≤ people) : 
  ∃ (ways : ℕ), ways = 36 :=
by
  have h : (∃ ways, ways = 36),
  from ⟨36, rfl⟩,
  exact h

end accommodation_ways_l325_325898


namespace average_of_first_10_even_numbers_l325_325776

theorem average_of_first_10_even_numbers : 
  let evens := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] in
  (evens.sum / evens.length : ℕ) = 11 := 
by
  admit

end average_of_first_10_even_numbers_l325_325776


namespace circle_regions_division_l325_325251

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l325_325251


namespace recurring_fraction_sum_l325_325157

theorem recurring_fraction_sum (a b : ℕ) (h : 0.36̅ = ↑a / ↑b) (gcd_ab : Nat.gcd a b = 1) : a + b = 15 :=
sorry

end recurring_fraction_sum_l325_325157


namespace number_of_ways_for_top_to_be_plus_l325_325985

def bottom_row_signs := Fin 5 → ιR = ±1

def sign_pyramid (bottom_row : bottom_row_signs) : ιR :=
(by calc
  let a := bottom_row 0; let b := bottom_row 1; let c := bottom_row 2; let d := bottom_row 3; let e := bottom_row 4;

  let r1 := [(a * b), (b * c), (c * d), (d * e)];
  let r2 := [(r1 0 * r1 1), (r1 1 * r1 2), (r1 2 * r1 3)];
  let r3 := [(r2 0 * r2 1), (r2 1 * r2 2)];
  let top := r3 0 * r3 1;

  top)

theorem number_of_ways_for_top_to_be_plus (correct_number : ιR = 12) : ∃ ways : Finset bottom_row_signs, ways.card = correct_number ∧ ∀ row ∈ ways, sign_pyramid row = 1 := sorry

end number_of_ways_for_top_to_be_plus_l325_325985


namespace number_of_fourth_quadrant_angles_l325_325483

-- Define a function that normalizes an angle to the range [0, 360).
def normalize_angle (θ : Int) : Int := 
  let n := θ % 360
  if n < 0 then n + 360 else n

-- Define a function to check if an angle is in the fourth quadrant.
def is_in_fourth_quadrant (θ : Int) : Bool := 
  let n := normalize_angle θ
  n > 270 || n < 360

-- List of angles to be checked.
def angles : List Int := [-20, -400, -2000, 1600]

-- Count the number of angles in the fourth quadrant.
def count_fourth_quadrant : Int :=
  angles.countp is_in_fourth_quadrant

theorem number_of_fourth_quadrant_angles :
  count_fourth_quadrant = 2 := by
  sorry

end number_of_fourth_quadrant_angles_l325_325483


namespace f_two_l325_325938

def f (n : ℕ+) : ℤ := match n with
  | ⟨1, _⟩ => 8
  | ⟨k+1, _⟩ => f ⟨k, by linarith⟩ + 7

theorem f_two : f ⟨2, by decide⟩ = 15 := 
sorry

end f_two_l325_325938


namespace square_coloring_contradiction_l325_325487

theorem square_coloring_contradiction :
  ∀ color : (Fin 11 → Fin 3),
  ∃ (i j : Fin 11), i ≠ j ∧ adjacent i j ∧ color i = color j :=
by
  sorry

-- Additional definitions related to adjacency and other specifics
def adjacent (i j : Fin 11) : Prop := 
  -- define adjacency relation on 11 squares, for instance, 1-2, 2-3, etc.
  sorry

end square_coloring_contradiction_l325_325487


namespace min_packs_needed_l325_325720

theorem min_packs_needed (P8 P15 P30 : ℕ) (h: P8 * 8 + P15 * 15 + P30 * 30 = 120) : P8 + P15 + P30 = 4 :=
by
  sorry

end min_packs_needed_l325_325720


namespace cost_of_paintbrush_l325_325016

noncomputable def cost_of_paints : ℝ := 4.35
noncomputable def cost_of_easel : ℝ := 12.65
noncomputable def amount_already_has : ℝ := 6.50
noncomputable def additional_amount_needed : ℝ := 12.00

-- Let's define the total cost needed and the total costs of items
noncomputable def total_cost_of_paints_and_easel : ℝ := cost_of_paints + cost_of_easel
noncomputable def total_amount_needed : ℝ := amount_already_has + additional_amount_needed

-- And now we can state our theorem that needs to be proved.
theorem cost_of_paintbrush : total_amount_needed - total_cost_of_paints_and_easel = 1.50 :=
by
  sorry

end cost_of_paintbrush_l325_325016


namespace length_of_segment_P_to_P_l325_325706

/-- Point P is given as (-4, 3) and P' is the reflection of P over the x-axis. 
    We need to prove that the length of the segment connecting P to P' is 6. -/
theorem length_of_segment_P_to_P' :
  let P := (-4, 3)
  let P' := (-4, -3)
  dist P P' = 6 :=
by
  sorry

end length_of_segment_P_to_P_l325_325706


namespace seq_periodic_l325_325125

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1/4
  else ite (n > 1) (1 - (1 / (seq (n-1)))) 0 -- handle invalid cases with a default zero

theorem seq_periodic {n : ℕ} (h : seq 1 = 1/4) (h2 : ∀ k ≥ 2, seq k = 1 - (1 / (seq (k-1)))) :
  seq 2014 = 1/4 :=
sorry

end seq_periodic_l325_325125


namespace face_opposite_yellow_is_blue_l325_325717

/-- Definition of the colors of the squares -/
inductive Color
| R | B | O | Y | G | W

open Color

/-- The face opposite the yellow (Y) face on the constructed cube is blue (B) -/
theorem face_opposite_yellow_is_blue
    (hinged_squares : CubeConfiguration)
    (color_front : Color = G)
    (color_back : Color = W)
    (color_right : Color = O)
    (color_left : Color = R)
    (color_bottom : Color = B)
    (color_top : Color = Y) :
    face_opposite color_top = color_bottom :=
by
  sorry

end face_opposite_yellow_is_blue_l325_325717


namespace sin_X_of_triangle_l325_325211

theorem sin_X_of_triangle (X Y Z : Type) [InnerProductSpace ℝ X] [InnerProductSpace ℝ Y] [InnerProductSpace ℝ Z] 
  (HX : innerProductSpace.angle X Z = real.pi / 2) (HXY : dist X Y = 12) (HYZ : dist Y Z = real.sqrt 51) : 
  innerProductSpace.angle Y X = real.arcsin (real.sqrt 51 / 12) := by
  sorry

end sin_X_of_triangle_l325_325211


namespace count_real_numbers_a_with_integer_roots_l325_325883

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l325_325883


namespace sequence_ninth_term_l325_325820

theorem sequence_ninth_term (a b : ℚ) :
  ∀ n : ℕ, n = 9 → (-1 : ℚ) ^ n * (n * b ^ n) / ((n + 1) * a ^ (n + 2)) = -9 * b^9 / (10 * a^11) :=
by
  sorry

end sequence_ninth_term_l325_325820


namespace PQ_bisects_angle_CPD_l325_325488

variable (P A B C D Q : Type)
variable (a b c d : ℝ)

-- The conditions given in the problem
axiom lengths : PA = a ∧ PB = b ∧ PC = c ∧ PD = d
axiom intersection : ∃ Q, Q ∈ AB ∩ CD
axiom lengths_relation : (1 / a) + (1 / b) = (1 / c) + (1 / d)

-- The theorem to be proved
theorem PQ_bisects_angle_CPD : ∃ Q, (CQ / QD) = (c / d) :=
  sorry

end PQ_bisects_angle_CPD_l325_325488


namespace problem_part1_problem_part2_l325_325610

noncomputable def a (α : ℝ) := (4 : ℝ, 5 * Real.cos α)
noncomputable def b (α : ℝ) := (3 : ℝ, -4 * Real.tan α)

theorem problem_part1 (α : ℝ) (hα : α ∈ Ioo 0 (π / 2)) (h_perpendicular : (a α).fst * (b α).fst + (a α).snd * (b α).snd = 0) : 
  Real.norm (Prod.mk (a α).fst - (b α).fst (a α).snd - (b α).snd) = 5 * Real.sqrt 2 := 
  sorry

theorem problem_part2 (α : ℝ) (hα : α ∈ Ioo 0 (π / 2)) (h_cos_alpha : Real.cos α = 4 / 5) : 
  Real.sin (3 * π / 2 + 2 * α) + Real.cos (2 * α - π) = -14 / 25 := 
  sorry

end problem_part1_problem_part2_l325_325610


namespace exist_distinct_xy_divisibility_divisibility_implies_equality_l325_325771

-- Part (a)
theorem exist_distinct_xy_divisibility (n : ℕ) (h_n : n > 0) :
  ∃ (x y : ℕ), x ≠ y ∧ (∀ j : ℕ, 1 ≤ j ∧ j ≤ n → (x + j) ∣ (y + j)) :=
sorry

-- Part (b)
theorem divisibility_implies_equality (x y : ℕ) (h : ∀ j : ℕ, (x + j) ∣ (y + j)) : 
  x = y :=
sorry

end exist_distinct_xy_divisibility_divisibility_implies_equality_l325_325771


namespace minimum_set_size_l325_325554

theorem minimum_set_size {r t : ℕ} (A : fin t → finset (fin r)) : 
  let X := finset.biUnion finset.univ A,
      n := Nat.find (λ n, Nat.choose n r ≥ t) in
  X.card = n :=
sorry

end minimum_set_size_l325_325554


namespace sandro_children_l325_325328

variables (sons daughters children : ℕ)

-- Conditions
def has_six_times_daughters (sons daughters : ℕ) : Prop := daughters = 6 * sons
def has_three_sons (sons : ℕ) : Prop := sons = 3

-- Theorem to be proven
theorem sandro_children (h1 : has_six_times_daughters sons daughters) (h2 : has_three_sons sons) : children = 21 :=
by
  -- Definitions from the conditions
  unfold has_six_times_daughters has_three_sons at h1 h2

  -- Skip the proof
  sorry

end sandro_children_l325_325328


namespace scout_troop_profit_l325_325000

theorem scout_troop_profit 
  (num_bars : ℕ) 
  (cost_per_three_bars : ℕ) 
  (selling_per_three_bars : ℕ)
  (total_bars : ℕ)
  (cost_dollars_per_bars : ℚ := cost_per_three_bars / 3) 
  (selling_dollars_per_bars : ℚ := selling_per_three_bars / 3) 
  (total_cost : ℚ := total_bars * cost_dollars_per_bars) 
  (total_revenue : ℚ := total_bars * selling_dollars_per_bars) 
  (profit : ℚ := total_revenue - total_cost) :
  num_bars = 1500 → cost_per_three_bars = 1 → selling_per_three_bars = 2 → total_bars = 1500 → profit = 500 := 
by
  intros
  have h1 : cost_dollars_per_bars = 1 / 3 := by rw [a_1]
  have h2 : selling_dollars_per_bars = 2 / 3 := by rw [a_2]
  have h3 : total_cost = 1500 * (1 / 3) := by rw [a_3, h1]
  have h4 : total_revenue = 1500 * (2 / 3) := by rw [a_3, h2]
  have h5 : total_cost = 500 := by norm_num
  have h6 : total_revenue = 1000 := by norm_num
  have h7 : profit = 1000 - 500 := by rw [h5, h6]
  norm_num at h7 ; exact h7

end scout_troop_profit_l325_325000


namespace part1_part2_l325_325593

noncomputable def f (x a : ℝ) : ℝ := x - 1 - a * Real.log x

theorem part1 (a : ℝ) : (∀ x : ℝ, x > 0 → f x a ≥ 0) ↔ a = 1 := by
  sorry

theorem part2 (m : ℤ) : (∀ n : ℕ, n > 0 → (Π i in Finset.range n, (1 + (1 / 2^i)) < m)) ↔ m = 3 := by
  sorry

end part1_part2_l325_325593


namespace nine_pointed_star_sum_of_angles_l325_325692

theorem nine_pointed_star_sum_of_angles (h : ∀i, ((i : ℤ) % 9) * 40 = 0) :
  ∑ (k : Fin 9), (1 / 2) * 4 * 40 = 720 := 
sorry

end nine_pointed_star_sum_of_angles_l325_325692


namespace least_3_digit_number_l325_325775

variables (k S h t u : ℕ)

def is_3_digit_number (k : ℕ) : Prop := k ≥ 100 ∧ k < 1000

def digits_sum_eq (k h t u S : ℕ) : Prop :=
  k = 100 * h + 10 * t + u ∧ S = h + t + u

def difference_condition (h t : ℕ) : Prop :=
  t - h = 8

theorem least_3_digit_number (k S h t u : ℕ) :
  is_3_digit_number k →
  digits_sum_eq k h t u S →
  difference_condition h t →
  k * 3 < 200 →
  k = 19 * S :=
sorry

end least_3_digit_number_l325_325775


namespace problem_1_probability_increasing_problem_2_probability_increasing_l325_325564

noncomputable def f (x a b : ℝ) : ℝ := a * x^2 - 4 * b * x + 1

theorem problem_1_probability_increasing:
  let P := {1, 2, 3} in
  let Q := {-1, 1, 2, 3, 4} in
  let event_space := P.prod Q in
  let favorable_events := 
    { (a, b) ∈ event_space | a > 0 ∧ 2 * b ≤ a } in 
    (favorable_events.card : ℚ) / (event_space.card : ℚ) = 1 / 3 :=
sorry

theorem problem_2_probability_increasing:
  let region := { (a, b) : ℝ × ℝ | a + b ≤ 8 ∧ a > 0 ∧ b > 0 } in
  let favorable_region := 
    { (a, b) ∈ region | 2 * b ≤ a } in
    (favorable_region.area : ℚ) / (region.area : ℚ) = 1 / 3 :=
sorry

end problem_1_probability_increasing_problem_2_probability_increasing_l325_325564


namespace cos_C_length_b_l325_325181

-- Conditions
variables {A B C a b c : ℝ}
variables (triangle_ABC : a > 0 ∧ b > 0 ∧ c > 0)
variables (C_obtuse : C > π / 2 ∧ C < π)
variables (cos_condition : cos (A + B - C) = 1 / 4)
variables (side_a : a = 2)
variables (sin_ratio : sin (B + A) / sin A = 2)

-- Question 1: Prove \(\cos C = -\frac{\sqrt{6}}{4}\)
theorem cos_C (h1 : cos (A + B - C) = 1 / 4) (h2 : C > π / 2 ∧ C < π) : cos C = -sqrt 6 / 4 := sorry

-- Conditions for Question 2
def cos_C_value := -sqrt 6 / 4
variables (cos_C_def : cos C = cos_C_value)

-- Question 2: Prove \(b = \sqrt{6}\)
theorem length_b (h3 : a = 2) (h4 : sin (B + A) / sin A = 2) (cos_C_def : cos C = cos_C_value) : b = sqrt 6 := sorry

end cos_C_length_b_l325_325181


namespace kids_in_2004_l325_325632

-- Set up the variables
variables (kids2004 kids2005 kids2006 : ℕ)

-- Given conditions
def condition1 : Prop := kids2005 = kids2004 / 2
def condition2 : Prop := kids2006 = (2 * kids2005) / 3
def condition3 : Prop := kids2006 = 20

-- Theorem statement
theorem kids_in_2004 :
  condition1 →
  condition2 →
  condition3 →
  kids2004 = 60 :=
by
  intros h1 h2 h3
  sorry

end kids_in_2004_l325_325632


namespace proposition_A_l325_325609

variables {m n : Line} {α β : Plane}

def parallel (x y : Line) : Prop := sorry -- definition for parallel lines
def perpendicular (x : Line) (P : Plane) : Prop := sorry -- definition for perpendicular line to plane
def parallel_planes (P Q : Plane) : Prop := sorry -- definition for parallel planes

theorem proposition_A (hmn : parallel m n) (hperp_mα : perpendicular m α) (hperp_nβ : perpendicular n β) : parallel_planes α β :=
sorry

end proposition_A_l325_325609


namespace max_ratio_of_three_digit_to_sum_l325_325855

theorem max_ratio_of_three_digit_to_sum (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9)
  (hc : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c) / (a + b + c) ≤ 100 :=
by sorry

end max_ratio_of_three_digit_to_sum_l325_325855


namespace AT_bisects_XY_l325_325204

-- Declare the geometric entities: points, lines, segments, and operations such as midpoint and intersection
variables {A B C D P Q X Y T O : Type} 
           [incidence_geometry A B C D P Q X Y T] [euclidean_geometry O] 

-- Assuming D is the midpoint of arc BC of the circumcircle of triangle ABC
def midpoint_arc (A B C D : Type) [circumcircle A B C O] : Prop := 
  ∃ O, circle O A ∧ circle O B ∧ circle O C ∧ arc_midpoint O B C D ∧ not (in_arc O A B C)

-- Tangents to the circle at points B and C intersect at points P and Q respectively
def tangents_intersect (A B C D P Q : Type) [tangent_circle A B O] [tangent_circle A C O] : Prop :=
  (is_tangent O B P) ∧ (is_tangent O C Q) ∧ (line_intersection PQ (tangent_circle A B O) (tangent_circle A C O))

-- points X = BQ ∩ AC and Y = CP ∩ AB
def point_intersections (A B C X Y : Type) [line_segment BQ AC] [line_segment CP AB] : Prop :=
  (intersect BQ AC X) ∧ (intersect CP AB Y)

-- point T is defined as the intersection of BQ and CP
def intersection_pt_T (BQ CP T : Type) [line_intersection BQ CP T] : Prop := 
  intersection BQ CP T

-- Defining the main theorem to prove that AT bisects segment XY given the previous conditions
theorem AT_bisects_XY (A B C D P Q X Y T : Type)
  [incidence_geometry A B C D P Q X Y T] [euclidean_geometry O]
  (h1 : midpoint_arc A B C D)
  (h2 : tangents_intersect A B C D P Q)
  (h3 : point_intersections A B C X Y)
  (h4 : intersection_pt_T BQ CP T):
  bisects (line_segment A T) (line_segment X Y) := 
begin
  sorry
end

end AT_bisects_XY_l325_325204


namespace question1_question2_l325_325930

-- Definitions for the given conditions
def a_1 : ℤ := -3
def d : ℤ := 2  -- determined from the given equation

def a_seq (n : ℕ) : ℤ := -3 + (n - 1) * 2

def S_n (n : ℕ) : ℤ := n * (a_1 + a_seq n) / 2

-- Proof statements for the questions
theorem question1 (n : ℕ) : a_seq n = 2 * n - 5 := by
  sorry

theorem question2 : ∃ n : ℕ, S_n n ≤ S_n m ∀ m : ℕ ∧ S_n 2 = -4 := by
  sorry

end question1_question2_l325_325930


namespace smallest_possible_value_of_M_l325_325677

theorem smallest_possible_value_of_M (a b c d e : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_sum : a + b + c + d + e = 2023) : 
  let M := max (max (a + b) (b + c)) (max (c + d) (d + e)) in
  M ≥ 405 :=
by
  let M := max (max (a + b) (b + c)) (max (c + d) (d + e))
  sorry

end smallest_possible_value_of_M_l325_325677


namespace ellipse_and_slope_condition_l325_325603

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def right_isosceles_triangle (A B P : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xP, yP) := P
  (xP - xA)^2 + (yP - yA)^2 = (xP - xB)^2 + (yP - yB)^2

def vector_ratio (P Q B : ℝ × ℝ) : Prop :=
  let (xP, yP) := P
  let (xQ, yQ) := Q
  let (xB, yB) := B
  let PQx := xQ - xP
  let PQy := yQ - yP
  let QBx := xB - xQ
  let QBy := yB - yQ
  PQx = 3/2 * QBx ∧ PQy = 3/2 * QBy

theorem ellipse_and_slope_condition
  (P A B Q : ℝ × ℝ)
  (a b : ℝ)
  (hP : P = (0, -2))
  (hA : A = (a, 0))
  (hB : B = (-a, 0))
  (hE : ellipse_eq a b)
  (hABP : right_isosceles_triangle A B P)
  (hPQ : vector_ratio P Q B)
  (ha_pos : a > 0) (hb_pos : b > 0)
  (hb_less_than_a : b < a) :
  (hE 2 (sqrt (3 - 2 * sqrt 2)) = True) ∧
  (∀ k : ℝ, (-2 < k ∧ k < -sqrt 3 / 2) ∨ (sqrt 3 / 2 < k ∧ k < 2)) :=
sorry

end ellipse_and_slope_condition_l325_325603


namespace circle_region_count_l325_325230

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l325_325230


namespace find_intersection_distance_l325_325560

noncomputable def linear_function : Type := ℝ → ℝ

def intersection_distance_1 (a b : ℝ) : ℝ :=
real.sqrt (a^2 + 4*b - 4)

def intersection_distance_2 (a b : ℝ) : ℝ :=
real.sqrt (a^2 + 4*b - 8)

def final_intersection_distance (a b : ℝ) : ℝ :=
real.sqrt 13

theorem find_intersection_distance {a b : ℝ}
  (h₁ : intersection_distance_1 a b = 3 * real.sqrt 2)
  (h₂ : intersection_distance_2 a b = real.sqrt 10)
  (h₃ : a^2 = 1 ∧ b = 3) :
  final_intersection_distance a b = real.sqrt 26 :=
sorry

end find_intersection_distance_l325_325560


namespace valid_points_region_equivalence_l325_325705

def valid_point (x y : ℝ) : Prop :=
  |x - 1| + |x + 1| + |2 * y| ≤ 4

def region1 (x y : ℝ) : Prop :=
  x ≤ -1 ∧ y ≤ x + 2 ∧ y ≥ -x - 2

def region2 (x y : ℝ) : Prop :=
  -1 < x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

def region3 (x y : ℝ) : Prop :=
  1 < x ∧ y ≤ 2 - x ∧ y ≥ x - 2

def solution_region (x y : ℝ) : Prop :=
  region1 x y ∨ region2 x y ∨ region3 x y

theorem valid_points_region_equivalence : 
  ∀ x y : ℝ, valid_point x y ↔ solution_region x y :=
sorry

end valid_points_region_equivalence_l325_325705


namespace problem_statement_l325_325503

noncomputable theory

def satisfies_condition (P : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (|y^2 - P x| ≤ 2 * |x| ↔ |x^2 - P y| ≤ 2 * |y|)

def possible_values_p0 (P : ℝ → ℝ) := 
  satisfies_condition P → (P 0 ∈ Iio 0 ∨ P 0 = 1)

theorem problem_statement : ∀ P : ℝ → ℝ, possible_values_p0 P :=
sorry

end problem_statement_l325_325503


namespace perpendicular_planes_l325_325608

variables {Line : Type} [has_perp Line]
variables {Plane : Type} [has_parallel Line Plane]
variables (l m n : Line) (α β : Plane)

theorem perpendicular_planes (h1: m ⟂ α) (h2: m ∥ n) (h3: n ∥ β) : α ⟂ β :=
sorry

end perpendicular_planes_l325_325608


namespace min_people_with_all_luxuries_l325_325196

variables (U : Type) [finite U] (P : U → Prop)
variables (fridge tv computer ac : set U)
variables (h_fridge : ∀ u, u ∈ fridge ↔ P u)
variables (h_tv : ∀ u, u ∈ tv ↔ P u)
variables (h_computer : ∀ u, u ∈ computer ↔ P u)
variables (h_ac : ∀ u, u ∈ ac ↔ P u)

def people_with_all_luxuries : set U := fridge ∩ tv ∩ computer ∩ ac

noncomputable def total_people := fintype.card U
noncomputable def people_with_fridge := (fridge.to_finset.card : ℝ) / total_people
noncomputable def people_with_tv := (tv.to_finset.card : ℝ) / total_people
noncomputable def people_with_computer := (computer.to_finset.card : ℝ) / total_people
noncomputable def people_with_ac := (ac.to_finset.card : ℝ) / total_people

theorem min_people_with_all_luxuries
  (hf : people_with_fridge = 0.70) 
  (ht : people_with_tv = 0.75) 
  (hc : people_with_computer = 0.90) 
  (ha : people_with_ac = 0.85) :
  people_with_all_luxuries.to_finset.card ≥ (0.70 * total_people) :=
by sorry

end min_people_with_all_luxuries_l325_325196


namespace dice_probability_l325_325311

theorem dice_probability :
  let prob_roll_less_than_four := 3 / 6
  let prob_roll_even := 3 / 6
  let prob_roll_greater_than_four := 2 / 6
  prob_roll_less_than_four * prob_roll_even * prob_roll_greater_than_four = 1 / 12 :=
by
  sorry

end dice_probability_l325_325311


namespace mean_is_one_l325_325763

-- Define variables and conditions
variable (X : ℝ → Prop) (μ σ : ℝ)

-- Assume X follows normal distribution and given condition P(X > -2) + P(X ≥ 4) = 1
axiom normal_dist : ∀ x, X x ↔ (x - μ) / σ = X x
axiom prob_condition : ∀ P : ℝ → Prop, (∫ x in set.Ioi (-2), P x) + (∫ x in set.Ici 4, P x) = 1

-- Prove the mean μ is 1
theorem mean_is_one (X : ℝ → Prop) (μ σ : ℝ) (normal_dist : ∀ x, X x ↔ (x - μ) / σ = X x)
  (prob_condition : ∀ P : ℝ → Prop, (∫ x in set.Ioi (-2), P x) + (∫ x in set.Ici 4, P x) = 1) :
  μ = 1 :=
sorry

end mean_is_one_l325_325763


namespace cost_of_remaining_shirt_l325_325401

theorem cost_of_remaining_shirt :
  ∀ (shirts total_cost cost_per_shirt remaining_shirt_cost : ℕ),
  shirts = 5 →
  total_cost = 85 →
  cost_per_shirt = 15 →
  (3 * cost_per_shirt) + (2 * remaining_shirt_cost) = total_cost →
  remaining_shirt_cost = 20 :=
by
  intros shirts total_cost cost_per_shirt remaining_shirt_cost
  intros h_shirts h_total h_cost_per_shirt h_equation
  sorry

end cost_of_remaining_shirt_l325_325401


namespace simple_interest_less_than_principal_l325_325397

def principal : ℝ := 2800
def rate : ℝ := 4
def time : ℝ := 5
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100
def difference (P SI : ℝ) : ℝ := P - SI

theorem simple_interest_less_than_principal :
  difference principal (simple_interest principal rate time) = 2240 :=
by
  sorry

end simple_interest_less_than_principal_l325_325397


namespace circle_regions_l325_325274

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l325_325274


namespace find_m_l325_325965

theorem find_m 
  (m n : ℝ) (b : ℝ) 
  (h : log 10 (m^2) = b - 2 * log 10 n) : 
  m = (10^(b/2) / n) ∨ m = -(10^(b/2) / n) := 
by sorry

end find_m_l325_325965


namespace number_of_triangles_is_correct_l325_325139

def points := Fin 6 × Fin 6

def is_collinear (p1 p2 p3 : points) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2)

noncomputable def count_triangles_with_positive_area : Nat :=
  let all_points := Finset.univ.product Finset.univ
  let all_combinations := all_points.powerset.filter (λ s, s.card = 3)
  let valid_triangles := all_combinations.filter (λ s, ¬is_collinear (s.choose 0) (s.choose 1) (s.choose 2))
  valid_triangles.card

theorem number_of_triangles_is_correct :
  count_triangles_with_positive_area = 6804 :=
by
  sorry

end number_of_triangles_is_correct_l325_325139


namespace circle_division_l325_325222

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l325_325222


namespace remainder_14_plus_x_mod_31_l325_325301

theorem remainder_14_plus_x_mod_31 (x : ℕ) (hx : 7 * x ≡ 1 [MOD 31]) : (14 + x) % 31 = 23 := 
sorry

end remainder_14_plus_x_mod_31_l325_325301


namespace hyperbola_eccentricity_correct_l325_325119

open Real

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
    let PF1 := (12 * a / 5)
    let PF2 := PF1 - 2 * a
    let c := (2 * sqrt 37 * a / 5)
    sqrt (1 + (b^2 / a^2))  -- Assuming the geometric properties hold, the eccentricity should match
-- Lean function to verify the result
def verify_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
    hyperbola_eccentricity a b ha hb = sqrt 37 / 5

-- Statement to be verified
theorem hyperbola_eccentricity_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
    verify_eccentricity a b ha hb := sorry

end hyperbola_eccentricity_correct_l325_325119


namespace a_n_divisible_by_n_l325_325039

noncomputable def a : ℕ → ℕ
| 0       := 0
| (n + 1) := 2^(n + 1) - ∑ d in (finset.divisors n).erase (n + 1), a d -- Define according to given condition

theorem a_n_divisible_by_n (n : ℕ) : n ∣ a n :=
by
  sorry

end a_n_divisible_by_n_l325_325039


namespace find_m_l325_325368

noncomputable def polynomial := Polynomial ℝ

theorem find_m (m : ℝ) (h_root : ∃ a b : ℝ, (a = b) ∧ 
    (∀ (x : ℝ), polynomial.eval x (3 * (Polynomial.X ^ 3) + 9 * (Polynomial.X ^ 2) - 135 * Polynomial.X + Polynomial.C m) = 0 → 
    (x = a ∨ x = b)))
    (h_pos : m > 0) : m = 22275 :=
sorry

end find_m_l325_325368


namespace inequality_holds_if_and_only_if_l325_325968

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_and_only_if (hx : |x-5| + |x-3| + |x-2| < b) : b > 4 :=
sorry

end inequality_holds_if_and_only_if_l325_325968


namespace petya_vasya_meet_at_lamp_64_l325_325821

-- Definitions of positions of Petya and Vasya
def Petya_position (x : ℕ) : ℕ := x - 21 -- Petya starts from the 1st lamp and is at the 22nd lamp
def Vasya_position (x : ℕ) : ℕ := 88 - x -- Vasya starts from the 100th lamp and is at the 88th lamp

-- Condition that both lanes add up to 64
theorem petya_vasya_meet_at_lamp_64 : ∀ x y : ℕ, 
    Petya_position x = Vasya_position y -> x = 64 :=
by
  intro x y
  rw [Petya_position, Vasya_position]
  sorry

end petya_vasya_meet_at_lamp_64_l325_325821


namespace equal_distances_l325_325498

-- Definition of a point and a circle in Lean
structure Point :=
(x : ℝ) (y : ℝ)

structure Circle :=
(center : Point) (radius : ℝ)

-- Conditions from the problem
variables
(z1 z2 : Circle) -- Circles z1 and z2
(M N A B C D E P Q : Point) -- Points M, N, A, B, C, D, E, P, Q
(l : Line) -- Line l which is a common tangent
(H1 : z1.center ≠ z2.center) -- Circles intersect at two points
(H2 : M ≠ N)
(H3 : A ≠ B)
(H4 : l.isTangent z1 A) -- l touches z1 at A
(H5 : l.isTangent z2 B) -- l touches z2 at B
(H6 : lineThrough M.parallelTo l)
(H7 : ∃ C, C ∈ z1 ∧ lineThrough M.parallelTo l)
(H8 : ∃ D, D ∈ z2 ∧ lineThrough M.parallelTo l)
(H9 : ∃ E, (lineThrough C A).meets (lineThrough D B) E)
(H10 : ∃ P, (lineThrough A N).meets (lineThrough C D) P)
(H11 : ∃ Q, (lineThrough B N).meets (lineThrough C D) Q)

-- Goal to prove
theorem equal_distances (EP_eq_EQ : dist E P = dist E Q) : True := 
by
  sorry -- Proof omitted since only the problem statement is required

end equal_distances_l325_325498


namespace area_quadrilateral_ADQD_l325_325649

noncomputable def area_ADQD' (A B C D Q D': Point)
  (H1: square ABCD)
  (H2: length (side CD) = 4)
  (H3: midpoint Q CD)
  (H4: reflection_square_ABCD_AQ_forms_AB'C'D'D')
  : ℝ := sorry

theorem area_quadrilateral_ADQD'_is_8
  (A B C D Q D' : Point)
  (H1: square ABCD)
  (H2: length (side CD) = 4)
  (H3: midpoint Q CD)
  (H4: reflection_square_ABCD_AQ_forms_AB'C'D'D')
  : area_ADQD' A B C D Q D' = 8 := sorry

end area_quadrilateral_ADQD_l325_325649


namespace min_value_bn_l325_325124

noncomputable def sequence {n : ℕ} (hn : n > 0) : ℕ := n

noncomputable def Sn (n : ℕ) : ℚ :=
  if n > 0 then (n * (n + 1)) / 2 else 0

noncomputable def bn (n : ℕ) : ℚ := 
  if n > 0 then (2 * Sn n + 7) / n else 0

theorem min_value_bn : bn 3 ≤ bn n :=
begin
  -- Define the domain of n
  assume n,
  cases n,
  { simp [bn], linarith },
  cases n,
  { simp [bn], linarith },
  cases n,
  iterate 4 {
    simp [bn, Sn, sequence],
    linarith,
  },
end

end min_value_bn_l325_325124


namespace interest_rate_same_l325_325978

theorem interest_rate_same (initial_amount: ℝ) (interest_earned: ℝ) 
  (time_period1: ℝ) (time_period2: ℝ) (principal: ℝ) (initial_rate: ℝ) : 
  initial_amount * initial_rate * time_period2 = interest_earned * 100 ↔ initial_rate = 12 
  :=
by
  sorry

end interest_rate_same_l325_325978


namespace blown_out_sand_dune_contains_treasure_and_coupon_l325_325313

theorem blown_out_sand_dune_contains_treasure_and_coupon :
  let
    prob_remain := 1 / 3,
    prob_treasure := 1 / 5,
    prob_lucky_coupon := 2 / 3,
    total_probability := prob_remain * prob_treasure * prob_lucky_coupon
  in total_probability = 2 / 45 := by
  sorry

end blown_out_sand_dune_contains_treasure_and_coupon_l325_325313


namespace circle_divided_into_regions_l325_325260

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l325_325260


namespace fraction_sum_l325_325153

theorem fraction_sum (a b : ℕ) (h1 : 0.36 = a / b) (h2: Nat.gcd a b = 1) : a + b = 15 := by
  sorry

end fraction_sum_l325_325153


namespace intersection_eq_l325_325093

noncomputable def A : Set ℝ := { x | x < 2 }
noncomputable def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} :=
by
  sorry

end intersection_eq_l325_325093


namespace regions_divided_by_radii_circles_l325_325242

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l325_325242


namespace largest_possible_value_x_plus_y_l325_325413

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((a.1 * (b.2 - c.2)) + (b.1 * (c.2 - a.2)) + (c.1 * (a.2 - b.2)))

def median_slope_condition (p m : ℝ × ℝ) (slope : ℝ) : Prop :=
  (m.2 - p.2) = slope * (m.1 - p.1)

theorem largest_possible_value_x_plus_y :
  ∃ (x y : ℝ), 
    area_of_triangle (x, y) (14, 20) (27, 22) = 85 ∧
    median_slope_condition (x, y) (midpoint (14, 20) (27, 22)) (-2) ∧
    x + y = 33.5 :=
sorry

end largest_possible_value_x_plus_y_l325_325413


namespace num_real_values_for_integer_roots_l325_325882

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l325_325882


namespace count_distinct_reals_a_with_integer_roots_l325_325889

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l325_325889


namespace max_a_value_l325_325561

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem max_a_value :
  (∀ x : ℝ, ∃ y : ℝ, f y a b = f x a b + y) → a ≤ 1/2 :=
by
  sorry

end max_a_value_l325_325561


namespace min_perimeter_of_triangle_with_conditions_l325_325180

-- Define the context of the problem
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the triangle with required properties
def isosceles_triangle (ABC : Triangle) : Prop :=
  ABC.is_triangle ∧ ABC.a = ABC.b

-- Define the conditions related to the incircle and excircle tangencies
def incircle_and_excircle_conditions (ABC : Triangle) (ω : Circle) : Prop :=
  ABC.incenter ∈ ω.center ∧
  (∀ ξ : Circle, ξ ∈ ABC.excircles → 
     if ξ.is_tangent_to_side ABC.BC then
       ξ.is_internally_tangent_to ω
     else
       ξ.is_externally_tangent_to ω)

-- Main theorem statement in Lean 4
theorem min_perimeter_of_triangle_with_conditions (ABC : Triangle) (ω : Circle) :
  isosceles_triangle ABC ∧ incircle_and_excircle_conditions ABC ω →
  ABC.perimeter = 20 :=
begin
  sorry  -- Proof goes here
end

end min_perimeter_of_triangle_with_conditions_l325_325180


namespace acute_angle_range_collinearity_value_l325_325072

-- Define vectors OA, OB, and OC
def vec_OA : ℝ × ℝ := (-2, 3)
def vec_OB (n : ℝ) : ℝ × ℝ := (n, 1)
def vec_OC : ℝ × ℝ := (5, -1)

-- Definition that the angle between OA and OB is acute
def is_acute_angle (OA OB : ℝ × ℝ) : Prop :=
  let dot_product := OA.1 * OB.1 + OA.2 * OB.2 in
  dot_product > 0

-- Definition that points are collinear (A, B, C in this context)
def collinear (OA OB OC : ℝ × ℝ) : Prop :=
  ∃ μ : ℝ, (OB.1 + OA.1, OB.2 + OA.2) = (μ * (OC.1 + OA.1), μ * (OC.2 + OA.2))

-- Lean statement for the first problem
theorem acute_angle_range (n : ℝ) : is_acute_angle vec_OA (vec_OB n) → 
  n ∈ Set.Ioo (-∞) (-2/3) ∪ Set.Ioo (-2/3) (3/2) := 
sorry

-- Lean statement for the second problem
theorem collinearity_value (n : ℝ) : collinear vec_OA (vec_OB n) vec_OC → n = 3/2 := 
sorry

end acute_angle_range_collinearity_value_l325_325072


namespace find_y_coordinate_l325_325652

theorem find_y_coordinate (x2 : ℝ) (y1 : ℝ) :
  (∃ m : ℝ, m = (y1 - 0) / (10 - 4) ∧ (-8 - y1) = m * (x2 - 10)) →
  y1 = -8 :=
by
  sorry

end find_y_coordinate_l325_325652


namespace log_domain_is_open_interval_l325_325373

def log_function_domain : Set ℝ :=
  {x : ℝ | x > 4}

theorem log_domain_is_open_interval :
  ∀ x : ℝ, x ∈ log_function_domain ↔ x > 4 :=
by
  intro x
  simp [log_function_domain]
  sorry

end log_domain_is_open_interval_l325_325373


namespace abs_val_eq_option_b_l325_325383

-- Define the absolute value function and other candidate functions.
def abs_val (x : ℝ) : ℝ := |x|
def option_b (x : ℝ) : ℝ := real.sqrt (x ^ 2)

-- The theorem stating the equivalence of the functions.
theorem abs_val_eq_option_b : ∀ x : ℝ, abs_val x = option_b x := by
  sorry

end abs_val_eq_option_b_l325_325383


namespace arrangements_not_adjacent_l325_325120

theorem arrangements_not_adjacent (a b c d e : ℕ) :
  let S := {a, b, c, d, e}
  ∃ n : ℕ, n = 36 ∧ ∀ l : list ℕ, l.perm S → 
  (no_adjacent l a c ∧ no_adjacent l b c) :=
sorry

end arrangements_not_adjacent_l325_325120


namespace trig_inequality_l325_325876

theorem trig_inequality (ϕ : ℝ) (h : 0 < ϕ ∧ ϕ < π / 2) : 
  sin (cos ϕ) < cos ϕ ∧ cos ϕ < cos (sin ϕ) := 
sorry

end trig_inequality_l325_325876


namespace range_of_a_l325_325631

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 3) - 2 * a ≥ abs (x + a)) ↔ ( -3/2 ≤ a ∧ a < -1/2) := 
by sorry

end range_of_a_l325_325631


namespace simplify_tan_expression_l325_325346

theorem simplify_tan_expression 
  (h30 : Real.tan (π / 6) = 1 / Real.sqrt 3)
  (h15 : Real.tan (π / 12) = 2 - Real.sqrt 3) :
  (1 + Real.tan (π / 6)) * (1 + Real.tan (π / 12)) = 2 :=
by
  -- State the tangent addition formula for the required angles
  have h_tan_add : Real.tan (π / 4) = (Real.tan (π / 6) + Real.tan (π / 12)) / (1 - Real.tan (π / 6) * Real.tan (π / 12)),
  {
    sorry,
  }
  -- The correct answer proof part is not provided in the brief
  sorry

end simplify_tan_expression_l325_325346


namespace triple_H_of_2_l325_325939

def H (x : ℝ) : ℝ := (x - 2)^2 / 3 - 4

theorem triple_H_of_2 : H (H (H 2)) = 8 := 
  sorry

end triple_H_of_2_l325_325939


namespace light_flash_interval_l325_325459

theorem light_flash_interval
    (flashes : ℕ) (time_hour : ℚ) (time_seconds : ℕ)
    (flashes = 240) 
    (time_hour = 3/4) 
    (time_seconds = 2700) :
    (time_seconds / flashes : ℚ) = 11.25 := 
by
  sorry

end light_flash_interval_l325_325459


namespace ten_factorial_perfect_square_probability_sum_l325_325808

theorem ten_factorial_perfect_square_probability_sum :
  let ten_factorial := (2^8) * (3^4) * (5^2) * (7^1),
      total_divisors := (8+1) * (4+1) * (2+1) * (1+1),
      perfect_square_divisors := 5 * 3 * 2 * 1,
      probability := perfect_square_divisors / total_divisors,
      m := Nat.gcdNumerator probability,
      n := Nat.gcdDenominator probability
  in m + n = 10 :=
by
  sorry

end ten_factorial_perfect_square_probability_sum_l325_325808


namespace similarity_definition_insufficient_l325_325764

theorem similarity_definition_insufficient (fig1 fig2 : Type) [Geometry fig1] [Geometry fig2] 
    (shape_similar : fig1 → fig2 → Prop) : 
    (∀ (F G : fig1), shape_similar F G → (similar F G ↔ G = dilation F ∧ ∀ x: fig1, shape_similar x x)) → 
    (shape_similar fig1 fig2 → ¬ similar fig1 fig2) :=
by
  intros
  sorry

end similarity_definition_insufficient_l325_325764


namespace concyclic_quad_iff_AP_eq_CP_l325_325187

-- Define initial conditions
variables {A B C D P : Type} [convex_quadrilateral A B C D]

-- Assumptions about angles and diagonal
variables (BD : diagonal A B C D)
variables [angle_not_bisector BD (angle A B C)]
variables [angle_not_bisector BD (angle C D A)]

-- Assumptions about point P within the quadrilateral and given angle conditions
variables [within_quadrilateral P A B C D]
variables [given_conditions (angle P B C = angle A B D) (angle P D C = angle B D A)]

-- Conclude that A, B, C, and D are concyclic if and only if AP = CP
theorem concyclic_quad_iff_AP_eq_CP : (concyclic A B C D) ↔ (distance A P = distance C P) := 
sorry

end concyclic_quad_iff_AP_eq_CP_l325_325187


namespace cos_square_alpha_pi_over_4_l325_325552

variable (α : ℝ)

theorem cos_square_alpha_pi_over_4 (h : sin (2 * α) = 2 / 3) : cos^2(α + π / 4) = 1 / 6 :=
by
  sorry

end cos_square_alpha_pi_over_4_l325_325552


namespace regions_formed_l325_325220

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l325_325220


namespace cos_B_value_cos_B_plus_pi_4_l325_325976

variables {A B C : ℝ} {a b c : ℝ}

-- Variable declarations for the angles and sides of the triangle.
def triangle_conditions := (c = (sqrt 5) / 2 * b) ∧ (C = 2 * B)

-- Proof problem for part (1): Prove that cos B = sqrt 5 / 4 given the conditions.
theorem cos_B_value (h : triangle_conditions) : cos B = sqrt 5 / 4 :=
sorry

-- Additional condition for part (2): dot product relationships.
def dot_product_condition := (a = c) ∧ (cos B = 3 / 5) 

-- Proof problem for part (2): Prove that cos (B + π / 4) = - sqrt 2 / 10 given the conditions.
theorem cos_B_plus_pi_4 (h : dot_product_condition) : cos (B + π / 4) = - sqrt 2 / 10 :=
sorry

end cos_B_value_cos_B_plus_pi_4_l325_325976


namespace range_b_over_a_l325_325591

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := Real.log x - a * x - b

lemma monotonicity (a b : ℝ) : 
  (a ≤ 0 → ∀ x > 0, (f x a b).derivative x > 0) ∧ 
  (a > 0 → (∀ x ∈ Ioo 0 (1/a), (f x a b).derivative x > 0) ∧ 
          (∀ x ∈ Ioi (1/a), (f x a b).derivative x < 0)) :=
by sorry

theorem range_b_over_a (a b: ℝ) (h : ∀ x > 0, f x a b ≤ 0) : 
  a > 0 → -1 ≤ b / a :=
by sorry

end range_b_over_a_l325_325591


namespace arithmetic_sequence_diff_l325_325202

theorem arithmetic_sequence_diff (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 7 = a 3 + 4 * d) :
  a 2008 - a 2000 = 8 * d :=
by
  sorry

end arithmetic_sequence_diff_l325_325202


namespace regions_formed_l325_325217

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l325_325217


namespace rectangle_contains_polygon_with_area_two_l325_325294

variable {P : Type} [ConvexPolygon P]

def area1 (P : P) : Prop :=
  P.area = 1

theorem rectangle_contains_polygon_with_area_two (P : P) (h1 : area1 P) :
  ∃ (R : Rectangle), R.area = 2 ∧ P ⊆ R :=
  sorry

end rectangle_contains_polygon_with_area_two_l325_325294


namespace regions_formed_l325_325215

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l325_325215


namespace tangent_line_at_1_1_max_k_l325_325116

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem tangent_line_at_1_1 :
  let f' := λ (x : ℝ), Real.log x + 2 in
  let slope := f' 1 in
  ∃ (m b : ℝ), m = 2 ∧ b = -1 ∧ ∀ y x, y - 1 = slope * (x - 1) ↔ y = m * x + b :=
sorry

theorem max_k :
  ∃ k : ℤ, (∀ x > 1, (k : ℝ) * (x - 1) < f x) ∧ (k = 3) :=
sorry

end tangent_line_at_1_1_max_k_l325_325116


namespace jesse_rooms_l325_325283

theorem jesse_rooms:
  ∀ (l w A n: ℕ), 
  l = 19 ∧ 
  w = 18 ∧ 
  A = 6840 ∧ 
  n = A / (l * w) → 
  n = 20 :=
by
  intros
  sorry

end jesse_rooms_l325_325283


namespace christina_walking_speed_l325_325659

-- Definitions based on the conditions
def initial_distance : ℝ := 150  -- Jack and Christina are 150 feet apart
def jack_speed : ℝ := 7  -- Jack's speed in feet per second
def lindy_speed : ℝ := 10  -- Lindy's speed in feet per second
def lindy_total_distance : ℝ := 100  -- Total distance Lindy travels

-- Proof problem: Prove that Christina's walking speed is 8 feet per second
theorem christina_walking_speed : 
  ∃ c : ℝ, (lindy_total_distance / lindy_speed) * jack_speed + (lindy_total_distance / lindy_speed) * c = initial_distance ∧ 
  c = 8 :=
by {
  use 8,
  sorry
}

end christina_walking_speed_l325_325659


namespace range_of_sqrt_meaningful_real_l325_325620

theorem range_of_sqrt_meaningful_real (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
by
  sorry

end range_of_sqrt_meaningful_real_l325_325620


namespace conditional_probability_even_given_six_l325_325033

open ProbabilityTheory

-- Define the sample space of rolling a six-sided die twice
def sampleSpace : Set (ℕ × ℕ) := {p | p.1 ∈ {1, 2, 3, 4, 5, 6} ∧ p.2 ∈ {1, 2, 3, 4, 5, 6}}

-- Event A: First roll results in a six
def EventA : Set (ℕ × ℕ) := {p | p.1 = 6}

-- Event B: Sum of the two rolls is even
def EventB : Set (ℕ × ℕ) := {p | (p.1 + p.2) % 2 = 0}

-- Conditional probability P(B|A) = 1/2
theorem conditional_probability_even_given_six :
  P(EventB | EventA) = 1/2 := sorry

end conditional_probability_even_given_six_l325_325033


namespace circles_radii_divide_regions_l325_325265

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l325_325265


namespace largest_square_area_approx_l325_325312

-- Define that we are working with a square on a standard coordinate plane where it might not necessarily align with axis lines and its sides are not parallel to the axes.
variables {R : Type*} [real_number : IsIR R]

structure Square :=
(vertices : list (R × R))
(is_square : is_square vertices)
(not_parallel_to_axes : ¬ (aligned_with_axes vertices))

-- Define the number of interior lattice points within the square.
def num_interior_lattice_points (sq : Square) : ℕ :=
number_of_points_with_integer_coordinates sq.vertices

-- Define the proposition stating the largest possible area of a square given the number of interior lattice points.
def largest_square_area_with_five_interior_lattice_points (sq : Square) : R :=
if num_interior_lattice_points sq = 5 then area sq else 0

-- Prove that the area is approximately 10.47 when there are exactly five interior lattice points
theorem largest_square_area_approx (sq : Square) 
  (cond1 : sq.is_square)
  (cond2 : sq.not_parallel_to_axes)
  (cond3 : num_interior_lattice_points sq = 5) : 
  abs (largest_square_area_with_five_interior_lattice_points sq - 10.47) < 0.01 :=
sorry

end largest_square_area_approx_l325_325312


namespace graph_of_equation_is_two_intersecting_lines_l325_325429

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x - 2 * y)^2 = x^2 + y^2 ↔ (y = 0 ∨ y = 4 / 3 * x) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l325_325429


namespace find_divisor_l325_325527

-- Define the given and calculated values in the conditions
def initial_value : ℕ := 165826
def subtracted_value : ℕ := 2
def resulting_value : ℕ := initial_value - subtracted_value

-- Define the goal: to find the smallest divisor of resulting_value other than 1
theorem find_divisor (d : ℕ) (h1 : initial_value - subtracted_value = resulting_value)
  (h2 : resulting_value % d = 0) (h3 : d > 1) : d = 2 := by
  sorry

end find_divisor_l325_325527


namespace number_of_arrangements_l325_325148

theorem number_of_arrangements :
  ∃ n : ℕ, n = 6! ∧ n = 720 :=
by
  have : 6! = 720 := rfl
  use 720
  exact ⟨this, rfl⟩

end number_of_arrangements_l325_325148


namespace ratio_of_shares_l325_325791

theorem ratio_of_shares (A B C : ℝ) (x : ℝ):
  A = 240 → 
  A + B + C = 600 →
  A = x * (B + C) →
  B = (2/3) * (A + C) →
  A / (B + C) = 2 / 3 :=
by
  intros hA hTotal hFraction hB
  sorry

end ratio_of_shares_l325_325791


namespace greatest_possible_bxa_l325_325773

-- Define the property of the number being divisible by 35
def div_by_35 (n : ℕ) : Prop :=
  n % 35 = 0

-- Define the main proof problem
theorem greatest_possible_bxa :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ div_by_35 (10 * a + b) ∧ (∀ (a' b' : ℕ), a' < 10 → b' < 10 → div_by_35 (10 * a' + b') → b * a ≥ b' * a') :=
sorry

end greatest_possible_bxa_l325_325773


namespace total_detergent_is_19_l325_325688

-- Define the quantities and usage of detergent
def detergent_per_pound_cotton := 2
def detergent_per_pound_woolen := 3
def detergent_per_pound_synthetic := 1

def pounds_of_cotton := 4
def pounds_of_woolen := 3
def pounds_of_synthetic := 2

-- Define the function to calculate the total amount of detergent needed
def total_detergent_needed := 
  detergent_per_pound_cotton * pounds_of_cotton +
  detergent_per_pound_woolen * pounds_of_woolen +
  detergent_per_pound_synthetic * pounds_of_synthetic

-- The theorem to prove the total amount of detergent used
theorem total_detergent_is_19 : total_detergent_needed = 19 :=
  by { sorry }

end total_detergent_is_19_l325_325688


namespace mu_n_lt_log3_4_l325_325082

noncomputable def l_intersects_ellipse (k : ℝ) (h : k ≠ 0) (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 = 1) ∧ (y = k * x)

noncomputable def slope_condition (k k1 k2 : ℝ) : Prop :=
  3 * (k1 + k2) = 8 * k

noncomputable def line_through_point (x y : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def area_ratio_condition (k : ℝ) (t : ℝ) (h : k^2 < 5/12) : Prop :=
  2 < t ∧ t < 3

noncomputable def sequence_term (n1 n2 : ℝ) (n : ℕ) : ℝ :=
  1 / ((n2)^n - 0.5 * n1)

noncomputable def sum_of_sequence (n1 n2: ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (sequence_term n1 n2)

theorem mu_n_lt_log3_4 (k k1 k2 n1 n2 : ℝ) (n : ℕ) :
  l_intersects_ellipse k (by sorry) (0 : ℝ) (1 / 2) ∧
  slope_condition k k1 k2 ∧
  line_through_point 1 (0 : ℝ) ∧
  area_ratio_condition k (by sorry) (by sorry) ∧
  (sequence_term n1 n2 n = by sorry) ∧ 
  (sum_of_sequence n1 n2 n < real.log 4 / real.log 3) := sorry

end mu_n_lt_log3_4_l325_325082


namespace problem_statement_l325_325097

noncomputable def f (x : ℝ) (t : ℝ) : ℝ :=
  if x >= 3 then log 3 (x + t) else 3 ^ x

theorem problem_statement (t : ℝ) (h : log 3 (3 + t) = 0) : f (f 6 t) t = 4 :=
  sorry

end problem_statement_l325_325097


namespace circle_regions_division_l325_325252

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l325_325252


namespace distance_travelled_by_circle_center_l325_325080

noncomputable def distance_travelled (a b c r : ℝ) : ℝ :=
  let s := a + b + c - 2 * (a + b + c) / (a*b*c) * r in 
  s * 3

theorem distance_travelled_by_circle_center :
  distance_travelled 9 12 15 2 = 20 :=
  sorry

end distance_travelled_by_circle_center_l325_325080


namespace f_symmetric_l325_325558

variable (T : ℝ)
variable (f : ℝ → ℝ)

theorem f_symmetric (x : ℝ) :
  (∀ x, f(x + 2 * T) = f(x)) → 
  (∀ x, T / 2 ≤ x ∧ x ≤ T → f(x) = f(T - x)) →
  (∀ x, T ≤ x ∧ x ≤ (3 / 2) * T → f(x) = -f(x - T)) →
  (∀ x, (3 / 2) * T ≤ x ∧ x ≤ 2 * T → f(x) = -f(2 * T - x)) →
  f(x) = f(T - x) :=
by
  sorry

end f_symmetric_l325_325558


namespace distance_between_points_l325_325835

-- Definitions based on the conditions given
def point1 : (ℝ × ℝ × ℝ) := (3, 5, 0)
def point2 : (ℝ × ℝ × ℝ) := (-2, 1, 4)

-- The main statement to prove
theorem distance_between_points :
  let distance := real.sqrt (((point2.1 - point1.1)^2 +
                              (point2.2 - point1.2)^2 +
                              (point2.3 - point1.3)^2)) in
  distance = real.sqrt 57 :=
by
  sorry

end distance_between_points_l325_325835


namespace distance_between_points_l325_325868

open Real

theorem distance_between_points : 
  let p1 := (2, 2)
  let p2 := (5, 9)
  dist (p1 : ℝ × ℝ) p2 = sqrt 58 :=
by
  let p1 := (2, 2)
  let p2 := (5, 9)
  have h1 : p1.1 = 2 := rfl
  have h2 : p1.2 = 2 := rfl
  have h3 : p2.1 = 5 := rfl
  have h4 : p2.2 = 9 := rfl
  sorry

end distance_between_points_l325_325868


namespace evaluate_expression_l325_325520

theorem evaluate_expression :
  let S := (1 / (4 - Real.sqrt 10)) - (1 / (Real.sqrt 10 - Real.sqrt 9)) + 
           (1 / (Real.sqrt 9 - Real.sqrt 8)) - (1 / (Real.sqrt 8 - Real.sqrt 7)) + 
           (1 / (Real.sqrt 7 - 3)) in
  S = 7 :=
by
  sorry

end evaluate_expression_l325_325520


namespace sum_is_correct_l325_325398

def number : ℕ := 81
def added_number : ℕ := 15
def sum_value (x : ℕ) (y : ℕ) : ℕ := x + y

theorem sum_is_correct : sum_value number added_number = 96 := 
by 
  sorry

end sum_is_correct_l325_325398


namespace complement_union_l325_325606

def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | x > 0}

theorem complement_union (x : ℝ) : (x ∈ Aᶜ ∪ B) ↔ (x ∈ Set.Iic (-1) ∪ Set.Ioi 0) := by
  sorry

end complement_union_l325_325606


namespace diameter_length_l325_325646

/-- Given geometric conditions involving circle arrangements and areas, 
    prove the length of the diameter AB. -/
theorem diameter_length (r s : ℝ) (A_shaded A_C : ℝ) (tangent1 tangent2 tangentPQ : Prop)
  (h1 : A_shaded = 39 * Real.pi)
  (h2 : A_C = 9 * Real.pi)
  (h3 : tangent1)
  (h4 : tangent2)
  (h5 : tangentPQ) :
  2 * (r + s) = 32 :=
begin
  -- sorry, the proof is skipped here
  sorry
end

end diameter_length_l325_325646


namespace simplify_tan_expression_l325_325345

theorem simplify_tan_expression 
  (h30 : Real.tan (π / 6) = 1 / Real.sqrt 3)
  (h15 : Real.tan (π / 12) = 2 - Real.sqrt 3) :
  (1 + Real.tan (π / 6)) * (1 + Real.tan (π / 12)) = 2 :=
by
  -- State the tangent addition formula for the required angles
  have h_tan_add : Real.tan (π / 4) = (Real.tan (π / 6) + Real.tan (π / 12)) / (1 - Real.tan (π / 6) * Real.tan (π / 12)),
  {
    sorry,
  }
  -- The correct answer proof part is not provided in the brief
  sorry

end simplify_tan_expression_l325_325345


namespace length_of_train_l325_325010

-- Define the speed of the train in km/hr
def speed_kmh : ℝ := 150

-- Define the time to cross the pole in seconds
def time_sec : ℝ := 12

-- Define the conversion factor from km/hr to m/s
def kmh_to_ms : ℝ := 5 / 18

-- Convert speed from km/hr to m/s
def speed_ms : ℝ := speed_kmh * kmh_to_ms

-- Define the distance formula
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- The expected length of the train
def expected_length : ℝ := 500.04

-- Lean statement to prove the length of the train
theorem length_of_train : distance speed_ms time_sec = expected_length :=
by
  sorry

end length_of_train_l325_325010


namespace acute_triangle_after_increase_l325_325173

theorem acute_triangle_after_increase (a b c m : ℝ)
  (h1 : c^2 ≤ a^2 + b^2)
  (h2 : 0 < m) :
  let a' := a + m,
      b' := b + m,
      c' := c + m in (c'^2 < a'^2 + b'^2) :=
sorry

end acute_triangle_after_increase_l325_325173


namespace arrange_3x3_grid_l325_325643

-- Define the problem conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := ¬ is_odd n

-- Define the function to count the number of such arrangements
noncomputable def count_arrangements : ℕ :=
  6 * 3^6 * 4^3 + 9 * 3^4 * 4^5 + 4^9

-- State the main theorem
theorem arrange_3x3_grid (nums : ℕ → Prop) (table : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 7) :
  (∀ i, is_odd (table i 0 + table i 1 + table i 2)) ∧ (∀ j, is_odd (table 0 j + table 1 j + table 2 j)) →
  count_arrangements = 6 * 3^6 * 4^3 + 9 * 3^4 * 4^5 + 4^9 :=
by sorry

end arrange_3x3_grid_l325_325643


namespace probability_div_by_3_l325_325440

-- Define the set of prime digits
def prime_digits : set ℕ := {2, 3, 5, 7}

-- Define the set of two-digit numbers with both digits prime
def two_digit_primes : set ℕ := {n | ∃ d1 d2, d1 ∈ prime_digits ∧ d2 ∈ prime_digits ∧ n = 10 * d1 + d2}

-- Define the set of two-digit numbers in two_digit_primes that are divisible by 3
def divisible_by_3 : set ℕ := {n ∈ two_digit_primes | (n % 3 = 0)}

-- Define the total number of two-digit prime numbers
def total_count : ℕ := 16

-- Define the count of numbers divisible by 3
def count_div_3 : ℕ := 5

-- Define the probability
def probability (num favorable total : ℕ) : ℚ := favorable / total

-- The theorem
theorem probability_div_by_3 : probability count_div_3 total_count = 5 / 16 := by
  sorry

end probability_div_by_3_l325_325440


namespace binom_20_10_l325_325569

open_locale nat

theorem binom_20_10 :
  (nat.choose 18 8 = 31824) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 172822 :=
by {
  intros h1 h2 h3,
  sorry
}

end binom_20_10_l325_325569


namespace abs_inequality_solution_l325_325175

theorem abs_inequality_solution {a : ℝ} (h : ∀ x : ℝ, |2 - x| + |x + 1| ≥ a) : a ≤ 3 :=
sorry

end abs_inequality_solution_l325_325175


namespace factorial_expression_simplifies_l325_325079

theorem factorial_expression_simplifies (n : ℕ) : 
    (n! - 1)! * (n + 1)! / (n!)! = n + 1 :=
by sorry

end factorial_expression_simplifies_l325_325079


namespace ellipse_equation_l325_325897

noncomputable def c : ℝ := real.sqrt (4 + 3)
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 3) = 1
def point_on_ellipse (ellipse : ℝ → ℝ → Prop) : Prop := ellipse 2 (3 * real.sqrt 3 / 2)

theorem ellipse_equation :
  (∃ ellipse : ℝ → ℝ → Prop, (∀ x y, ellipse x y ↔ (x^2 / 16) + (y^2 / 9) = 1) ∧ point_on_ellipse ellipse) →
  ∀ (x y : ℝ), (x^2 / 16) + (y^2 / 9) = 1 → ellipse x y :=
by
  sorry

end ellipse_equation_l325_325897


namespace product_of_repeating_decimal_l325_325532

theorem product_of_repeating_decimal (p : ℝ) (h : p = 0.6666666666666667) : p * 6 = 4 :=
sorry

end product_of_repeating_decimal_l325_325532


namespace find_distance_to_focus_l325_325906

-- Define the hyperbola and the foci
def isPointOnHyperbola (P : ℝ × ℝ) : Prop := 
  let (x, y) := P
  (x^2 / 16) - (y^2 / 20) = 1

def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem find_distance_to_focus
  (P : ℝ × ℝ)
  (F₁ : ℝ × ℝ := (-6, 0))
  (F₂ : ℝ × ℝ := (6, 0))
  (hP : isPointOnHyperbola P)
  (hF1 : distance P F₁ = 9) :
  distance P F₂ = 17 :=
sorry

end find_distance_to_focus_l325_325906


namespace num_even_integers_between_fractions_l325_325130

theorem num_even_integers_between_fractions : 
  let a := 19 / 4
  let b := 40 / 2
  ∃ n, (n = 8) ∧ (∀ k, k ∈ (Set.Icc (Int.ceil a) (Int.floor b)) → even k) :=
begin
  sorry
end

end num_even_integers_between_fractions_l325_325130


namespace gypsy_pheasants_l325_325958

theorem gypsy_pheasants (x : ℚ) (h1 : (3 : ℚ) = x / 8 - 7 / 8) : x = 31 := 
sorry

end gypsy_pheasants_l325_325958


namespace total_monthly_bill_working_from_home_l325_325687

def original_monthly_bill : ℝ := 60
def percentage_increase : ℝ := 0.30

theorem total_monthly_bill_working_from_home :
  original_monthly_bill + (original_monthly_bill * percentage_increase) = 78 := by
  sorry

end total_monthly_bill_working_from_home_l325_325687


namespace alpha_plus_beta_l325_325071

theorem alpha_plus_beta (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hcos : cos α = -3 * (sqrt 10) / 10) (hsin : sin (2 * α + β) = 1 / 2 * sin β) :
  α + β = 5 / 4 * π :=
sorry

end alpha_plus_beta_l325_325071


namespace simplify_tan_expression_l325_325361

theorem simplify_tan_expression :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 := 
by 
  -- Angle addition formula for tangent
  have h : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12), 
  from by rw [Real.tan_add]; exact Real.tan_pi_div_four,
  -- Given that tan 45° = 1
  have h1 : Real.tan (Real.pi / 4) = 1, from Real.tan_pi_div_four,
  -- Derive the known value
  rw [Real.tan_pi_div_four, h] at h1,
  -- Simplify using the derived value
  suffices : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 
             1 + Real.tan (Real.pi / 6) + Real.tan (Real.pi / 12) + Real.tan (Real.pi / 6) * Real.tan (Real.pi / 12), 
  from by rw this; simp [←h1],
  sorry

end simplify_tan_expression_l325_325361


namespace problem_solution_l325_325377

theorem problem_solution (a b : ℤ) (h1 : 6 * b + 4 * a = -50) (h2 : a * b = -84) : a + 2 * b = -17 := 
  sorry

end problem_solution_l325_325377


namespace polynomial_factorization_l325_325052

theorem polynomial_factorization :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 = 5 * x^4 + 180 * x^3 + 1431 * x^2 + 4900 * x + 5159 :=
by sorry

end polynomial_factorization_l325_325052


namespace circles_radii_divide_regions_l325_325266

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l325_325266


namespace circle_divided_into_regions_l325_325258

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l325_325258


namespace general_term_b_seq_Sn_inequality_l325_325928

-- Definitions and conditions for the sequences a_n and b_n
def a_seq (n : ℕ) : ℝ :=
  match n with
  | 1 => 3
  | 3 => 15
  | _ => (n : ℝ) * (n + 2)

def b_seq (n : ℕ) : ℝ := n

def S_seq (n : ℕ) : ℝ :=
  (1 / 2) * ∑ i in Finset.range (n + 1), (1 / (i + 1 : ℝ)) - (1 / ((i + 3) : ℝ))

-- Statement 1
theorem general_term_b_seq : 
  ∀ n : ℕ, (b_seq n = n ∨ b_seq n = -n) := 
sorry

-- Statement 2
theorem Sn_inequality (n : ℕ) : 
  S_seq n < 3 / 4 := 
sorry

end general_term_b_seq_Sn_inequality_l325_325928


namespace calculate_expression_l325_325837

-- Define the expression using the provided conditions
def expression := 6 + 15 / 3 - 4^2

-- Theorem statement: the result of the expression is -5
theorem calculate_expression : expression = -5 :=
by
  sorry

end calculate_expression_l325_325837


namespace fraction_sum_l325_325151

theorem fraction_sum (x a b : ℕ) (h1 : x = 36 / 99) (h2 : a = 4) (h3 : b = 11) (h4 : Nat.gcd a b = 1) : a + b = 15 :=
by
  sorry

end fraction_sum_l325_325151


namespace solve_quadratic_solve_cubic_l325_325061

theorem solve_quadratic :
  ∀ (x : ℝ), 25 * x ^ 2 - 9 = 7 → x = 4/5 ∨ x = -4/5 :=
begin
  intros x h,
  sorry
end

theorem solve_cubic :
  ∀ (x : ℝ), 8 * (x - 2) ^ 3 = 27 → x = 7/2 :=
begin
  intros x h,
  sorry
end

end solve_quadratic_solve_cubic_l325_325061


namespace volume_of_cube_within_pyramid_l325_325809

noncomputable def pyramid_has_hexagonal_base
  (side_length : ℝ) : Prop :=
  side_length = 2

noncomputable def pyramid_lateral_faces_are_equilateral
  (side_length : ℝ) : Prop :=
  ∀ (face : ℝ), face = (sqrt 3 / 2) * side_length

noncomputable def cube_within_pyramid
  (base_side : ℝ) : Prop :=
  ∀ (cube_side : ℝ), cube_side = (sqrt 2 / 2)

theorem volume_of_cube_within_pyramid
  (base_side : ℝ)
  (side_length : ℝ)
  (cube_side : ℝ) :
  pyramid_has_hexagonal_base base_side →
  pyramid_lateral_faces_are_equilateral side_length →
  cube_within_pyramid base_side →
  cube_side^3 = sqrt 2 / 4 :=
by
  assume h_hex_base h_lat_faces h_cube_within,
  -- Proof omitted
  sorry

end volume_of_cube_within_pyramid_l325_325809


namespace sum_of_angles_of_9_pointed_star_is_540_l325_325695

-- Define the circle with nine evenly spaced points.
def circle_with_nine_points := { p : ℝ // 0 <= p ∧ p < 360 }

-- Define a 9-pointed star formed by connecting these nine points.
def nine_pointed_star (points : fin 9 → circle_with_nine_points) : Prop :=
  ∀ i : fin 9, points i = ⟨ i.1 * 40, sorry ⟩

-- Define the sum of the angle measurements at the tips of the 9-pointed star.
def sum_of_tip_angles (points : fin 9 → circle_with_nine_points) : ℝ :=
  (∑ i in finset.univ, 60)

-- Statement to be proved: 
theorem sum_of_angles_of_9_pointed_star_is_540 : ∀ points : fin 9 → circle_with_nine_points,  
  nine_pointed_star points → sum_of_tip_angles points = 540 := 
by
  intros points h
  sorry

end sum_of_angles_of_9_pointed_star_is_540_l325_325695


namespace find_matrix_N_l325_325055

def cross_product (u v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ k, match k with
  | 0 => u 1 * v 2 - u 2 * v 1
  | 1 => u 2 * v 0 - u 0 * v 2
  | 2 => u 0 * v 1 - u 1 * v 0
  | _ => 0

def matrix_N : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 7, -4],
    ![-7, 0, -3],
    ![4, 3, 0]
  ]

theorem find_matrix_N (w : Fin 3 → ℝ) :
  matrix_N.mul_vec w = cross_product ![3, -4, 7] w :=
by
  sorry

end find_matrix_N_l325_325055


namespace binomial_identity_l325_325572

theorem binomial_identity :
  (nat.choose 18 8 = 31824) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 172822 :=
by
  intros h1 h2 h3
  have h4: nat.choose 19 9 = nat.choose 18 8 + nat.choose 18 9 := by sorry
  have h5: nat.choose 19 9 = 31824 + 48620 := by sorry
  have h6: nat.choose 19 10 = nat.choose 18 9 + nat.choose 18 10 := by sorry
  have h7: nat.choose 19 10 = 48620 + 43758 := by sorry
  show nat.choose 20 10 = nat.choose 19 9 + nat.choose 19 10 from sorry
  have h8: nat.choose 20 10 = 80444 + 92378 := by sorry
  exact sorry

end binomial_identity_l325_325572


namespace ramsey3_3_6_l325_325745
-- Import Mathlib for graph theory and related functions

-- Define the problem setup and the theorem statement
theorem ramsey3_3_6 : ∀ (G : SimpleGraph (Fin 6)), ∃ (S : Finset (Fin 6)), S.card = 3 ∧ G.edgeColoring IsMonochromatic S :=
by
  -- Formalize the structure and assumptions
  sorry -- Proof is omitted as per the instruction.

end ramsey3_3_6_l325_325745


namespace range_of_sqrt_meaningful_real_l325_325621

theorem range_of_sqrt_meaningful_real (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
by
  sorry

end range_of_sqrt_meaningful_real_l325_325621


namespace max_xy_l325_325622

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 9 * y = 12) : xy ≤ 4 :=
by
sorry

end max_xy_l325_325622


namespace determine_y_value_l325_325857

theorem determine_y_value : 
  let y : ℚ := (2023^2 - 1012) / 2023 in 
  y = 2023 - (1012 / 2023) := 
by 
  sorry

end determine_y_value_l325_325857


namespace union_A_B_eq_intersection_complements_eq_subset_intersection_imp_range_l325_325607

open Set

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := {x | -1 < x ∧ x < 3}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}
noncomputable def A_c : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
noncomputable def B_c : Set ℝ := {x | x ≤ 0 ∨ x > 4}

theorem union_A_B_eq : A ∪ B = {x | -1 < x ∧ x ≤ 4} := sorry

theorem intersection_complements_eq : (A_c ∩ B_c) = {x | x ≤ -1 ∨ x > 4} := sorry

theorem subset_intersection_imp_range (a : ℝ) : 
  (∀ x, C a x → x ∈ (A ∩ B)) → (0 ≤ a ∧ a ≤ 2) := sorry

end union_A_B_eq_intersection_complements_eq_subset_intersection_imp_range_l325_325607


namespace third_speed_correct_l325_325465

variable (total_time : ℝ := 11)
variable (total_distance : ℝ := 900)
variable (speed1_km_hr : ℝ := 3)
variable (speed2_km_hr : ℝ := 9)

noncomputable def convert_speed_km_hr_to_m_min (speed: ℝ) : ℝ := speed * 1000 / 60

noncomputable def equal_distance : ℝ := total_distance / 3

noncomputable def third_speed_m_min : ℝ :=
  let speed1_m_min := convert_speed_km_hr_to_m_min speed1_km_hr
  let speed2_m_min := convert_speed_km_hr_to_m_min speed2_km_hr
  let d := equal_distance
  300 / (total_time - (d / speed1_m_min + d / speed2_m_min))

noncomputable def third_speed_km_hr : ℝ := third_speed_m_min * 60 / 1000

theorem third_speed_correct : third_speed_km_hr = 6 := by
  sorry

end third_speed_correct_l325_325465


namespace fraction_sum_l325_325156

theorem fraction_sum (a b : ℕ) (h1 : 0.36 = a / b) (h2: Nat.gcd a b = 1) : a + b = 15 := by
  sorry

end fraction_sum_l325_325156


namespace count_real_numbers_a_with_integer_roots_l325_325886

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l325_325886


namespace simplify_tan_product_l325_325338

theorem simplify_tan_product : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  -- use the angle addition formula for tangent
  have tan_sum : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12) :=
    by rw [Real.tan_add, Real.tan_pi_div_four]
  -- using the given condition tan(45 degrees) = 1
  have tan_45 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  sorry

end simplify_tan_product_l325_325338


namespace princess_sissi_reservoir_l325_325979

theorem princess_sissi_reservoir :
  ∃ t : ℕ, t = 15 ∧
  ∃ r R V : ℕ,
  R = 2 * r ∧ V = 30 * r ∧
  (90 * r - 30 * R = V) ∧
  (50 * r - 10 * R = V) ∧
  (4 * r * t - R * t = V) :=
by {
  use 15,
  sorry
}

end princess_sissi_reservoir_l325_325979


namespace natural_numbers_property_l325_325523

noncomputable section

variables {x a : ℕ}

theorem natural_numbers_property (a_non_zero : a ≠ 0) (hx : ∀ d ∈ digits x, a ≤ d ∧ d ≤ a + 1) :
  x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 :=
by
  sorry

end natural_numbers_property_l325_325523


namespace circles_radii_divide_regions_l325_325267

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l325_325267


namespace incorrect_description_proof_l325_325824

-- Definitions for each of the conditions stated in the problem
def solid_of_revolution_sphere (semicircle : Type) (diameter : semicircle) : Prop :=
  ∃ (sphere : Type), sphere = rotate_around_line semicircle diameter

def closed_surface_cone (triangle : Type) (base_height_line : triangle) : Prop :=
  ∃ (cone : Type), cone = rotate_around_line triangle base_height_line 180

def frustum_cone_section (cone : Type) (plane : Type) (base : cone) : Prop :=
  ∃ (frustum : Type), plane ≠ parallel_to base ∧ frustum = section_between base plane

def solid_of_revolution_cylinder (rectangle : Type) (side : rectangle) : Prop :=
  ∃ (cylinder : Type), cylinder = rotate_around_line rectangle side

-- The main theorem to be proved
theorem incorrect_description_proof (semicircle : Type) (diameter : semicircle)
  (triangle : Type) (base_height_line : triangle)
  (cone : Type) (plane : Type) (base : cone)
  (rectangle : Type) (side : rectangle) :
  solid_of_revolution_sphere semicircle diameter ∧
  closed_surface_cone triangle base_height_line ∧
  solid_of_revolution_cylinder rectangle side →
  ¬ frustum_cone_section cone plane base :=
sorry

end incorrect_description_proof_l325_325824


namespace distance_from_center_to_line_is_sqrt_2_l325_325941

-- Define the parametric line equation
def line (t : ℝ) : ℝ × ℝ := (t, t + 1)

-- Define the polar circle equation
def circle (θ : ℝ) : ℝ := 2 * cos θ

-- Define the standard Cartesian forms derived from the conditions
def line_standard : ℝ → ℝ → Prop := λ x y, (x - y + 1) = 0
def circle_standard : ℝ → ℝ → Prop := λ x y, (x^2 + y^2 - 2*x) = 0

-- Define the center of the circle as identified from the conversion
def circle_center : ℝ × ℝ := (1, 0)

-- Define the point-to-line distance formula as per standard geometric distance calculation
def point_to_line_distance (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / sqrt (a^2 + b^2)

-- Prove that the distance from the center of the circle to the line is sqrt 2
theorem distance_from_center_to_line_is_sqrt_2 :
  point_to_line_distance (1, 0) 1 (-1) 1 = sqrt 2 := by
  -- Proof goes here
  sorry

end distance_from_center_to_line_is_sqrt_2_l325_325941


namespace triangle_AB_CD_eq_BC_l325_325212
-- importing math library

-- defining points and triangle ABC
variables {A B C D : Type} [add_comm_group B] [normed_add_comm_group B] [normed_space ℝ B]

-- defining angle
variables {α : Type} [measure_space α] [linear_order α] [order_top α] [order_bot α]
variables (angle : α) (BAC C DAC : B)

-- conditions
axiom angle_A_eq_3_angle_C (C : B) : angle = 3 * (angle_C : B)
axiom ∠DAC_eq_2_angle_C (D : B) : angle = 2 * (angle_C : B)

-- goal
theorem triangle_AB_CD_eq_BC (C BAC DAC : B) (h1: angle_A_eq_3_angle_C C) (h2: ∠DAC_eq_2_angle_C D) : 
B ≠ C → 
(A ≠ B → 
angle = 3 * angle_C ∧
angle = 2 * angle_C) →
  by sorry

end triangle_AB_CD_eq_BC_l325_325212


namespace binom_20_10_l325_325577

-- Given conditions
def binom_18_8 : ℕ := 31824
def binom_18_9 : ℕ := 48620
def binom_18_10 : ℕ := 43758

theorem binom_20_10 : nat.choose 20 10 = 172822 := by
  have h1 : nat.choose 19 9 = binom_18_8 + binom_18_9 := rfl
  have h2 : nat.choose 19 10 = binom_18_9 + binom_18_10 := rfl
  have h3 : nat.choose 20 10 = nat.choose 19 9 + nat.choose 19 10 := rfl
  rw [h1, h2, h3]
  exact rfl

end binom_20_10_l325_325577


namespace quadratic_inequality_solution_l325_325174

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : b = 2 * a) (h3 : c = -3 * a) :
  (b < 0 ∧ c > 0) ∧
  (∀ x : ℝ, ax - b < 0 ↔ x > 2) ∧
  (∀ x : ℝ, ax^2 - bx + c < 0 ↔ x ∈ (-∞, -1) ∪ (3, +∞)) :=
by 
  sorry

end quadratic_inequality_solution_l325_325174


namespace password_count_correct_l325_325796

-- Defining variables
def n_letters := 26
def n_digits := 10

-- The number of permutations for selecting 2 different letters
def perm_letters := n_letters * (n_letters - 1)
-- The number of permutations for selecting 2 different numbers
def perm_digits := n_digits * (n_digits - 1)

-- The total number of possible passwords
def total_permutations := perm_letters * perm_digits

-- The theorem we need to prove
theorem password_count_correct :
  total_permutations = (n_letters * (n_letters - 1)) * (n_digits * (n_digits - 1)) :=
by
  -- The proof goes here
  sorry

end password_count_correct_l325_325796


namespace remaining_lawn_after_john_mows_one_hour_l325_325284

theorem remaining_lawn_after_john_mows_one_hour :
  let john_mowing_rate := (1 : ℝ) / 3;
  let hours_john_works := 1;
  let lawn_mowed_by_john := john_mowing_rate * hours_john_works;
  let initial_lawn := 1 in
  initial_lawn - lawn_mowed_by_john = (2 : ℝ) / 3 := by
sorry

end remaining_lawn_after_john_mows_one_hour_l325_325284


namespace problem_1_problem_2_l325_325914

def f (x : ℝ) : ℝ := x^2 + 4 * x
def g (a : ℝ) : ℝ := |a - 2| + |a + 1|

theorem problem_1 (x : ℝ) :
    (f x ≥ g 3) ↔ (x ≥ 1 ∨ x ≤ -5) :=
  sorry

theorem problem_2 (a : ℝ) :
    (∃ x : ℝ, f x + g a = 0) → (-3 / 2 ≤ a ∧ a ≤ 5 / 2) :=
  sorry

end problem_1_problem_2_l325_325914


namespace binomial_identity_l325_325575

theorem binomial_identity :
  (nat.choose 18 8 = 31824) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 172822 :=
by
  intros h1 h2 h3
  have h4: nat.choose 19 9 = nat.choose 18 8 + nat.choose 18 9 := by sorry
  have h5: nat.choose 19 9 = 31824 + 48620 := by sorry
  have h6: nat.choose 19 10 = nat.choose 18 9 + nat.choose 18 10 := by sorry
  have h7: nat.choose 19 10 = 48620 + 43758 := by sorry
  show nat.choose 20 10 = nat.choose 19 9 + nat.choose 19 10 from sorry
  have h8: nat.choose 20 10 = 80444 + 92378 := by sorry
  exact sorry

end binomial_identity_l325_325575


namespace chinese_chess_sets_l325_325001

theorem chinese_chess_sets (x y : ℕ) 
  (h1 : 24 * x + 18 * y = 300) 
  (h2 : x + y = 14) : 
  y = 6 := 
sorry

end chinese_chess_sets_l325_325001


namespace no_primes_in_sequence_l325_325765

-- Define Q as the product of all prime numbers up to 53
def Q : ℕ := ∏ p in (Finset.filter Nat.prime (Finset.range 54)), p

-- Define the sequence based on n ranging from 2 to 51
def sequence : Finset ℕ := (Finset.range 50).image (λ n, Q + (n + 2))

-- Define M as the count of prime elements in the sequence
def M := (sequence.filter (λ k, Nat.prime k)).card

-- Prove that M is 0
theorem no_primes_in_sequence : M = 0 :=
by
  -- Proof goes here
  sorry

end no_primes_in_sequence_l325_325765


namespace probability_abd_together_l325_325769

-- Definitions based on the conditions in a)
def individuals := {a, b, c, d, e, f, g, h}
def units := {super_person, c, e, f, g, h}  -- super_person represents {a, b, d}

-- Number of arrangements of 6 units
def arrangements_units : Nat := 6.factorial

-- Number of arrangements within the super_person unit
def arrangements_within_super_person : Nat := 3.factorial

-- Total number of favorable arrangements
def favorable_arrangements : Nat := arrangements_units * arrangements_within_super_person

-- Total number of arrangements of 8 individuals
def total_arrangements : Nat := 8.factorial

-- The probability that a, b, and d are sitting together
def probability : Float := favorable_arrangements.toFloat / total_arrangements.toFloat

-- Target probability
def target_probability : Float := 1 / 9.375

-- The proof statement that the calculated probability equals the target probability
theorem probability_abd_together : probability = target_probability := by
  sorry

end probability_abd_together_l325_325769


namespace multiple_7_proposition_l325_325390

theorem multiple_7_proposition : (47 % 7 ≠ 0 ∨ 49 % 7 = 0) → True :=
by
  intros h
  sorry

end multiple_7_proposition_l325_325390


namespace soccer_team_solution_l325_325003

def soccer_team_problem : Prop :=
  ∃ (P : ℕ),
    (let goals_scored_by_one_third := (1/3 : ℝ) * P * 1 * 15 in
    let goals_scored_by_others := 30 in
    goals_scored_by_one_third + goals_scored_by_others = 150) ∧
    P = 24

theorem soccer_team_solution : soccer_team_problem :=
by {
  use 24,
  simp,
  unfold goals_scored_by_one_third,
  linarith,
  sorry
}

end soccer_team_solution_l325_325003


namespace exists_f_m_eq_n_plus_2017_l325_325873

theorem exists_f_m_eq_n_plus_2017 (m : ℕ) (h : m > 0) :
  (∃ f : ℤ → ℤ, ∀ n : ℤ, (f^[m] n = n + 2017)) ↔ (m = 1 ∨ m = 2017) :=
by
  sorry

end exists_f_m_eq_n_plus_2017_l325_325873


namespace female_officers_on_police_force_l325_325441

theorem female_officers_on_police_force (F : ℕ) (h1 : 160 / 2 = 80) (h2 : 0.16 * F = 80) :
  F = 500 := sorry

end female_officers_on_police_force_l325_325441


namespace problem_l325_325600

def g (a b c : ℤ) : ℤ := (c^2 + a) / (c^2 - b)

theorem problem : g 2 4 (-1) = -1 := 
by 
  -- details of proof
  simp [g]
  sorry

end problem_l325_325600


namespace geometric_sequence_common_ratio_l325_325584

theorem geometric_sequence_common_ratio (S : ℕ → ℝ) (a : ℕ → ℝ)
  (q : ℝ) (h1 : a 1 = 2) (h2 : S 3 = 6)
  (geo_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  q = 1 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l325_325584


namespace lovely_books_earning_l325_325473

theorem lovely_books_earning :
  ∀ (n : ℕ) (p1 p2 : ℝ), n = 10 → p1 = 2.50 → p2 = 2 →
    (2/5 * n * p1 + (n - 2/5 * n) * p2) = 22 := 
by
  intros n p1 p2 h_n h_p1 h_p2
  rw [h_n, h_p1, h_p2]
  have h1 : 2 / 5 * 10 = 4 := by norm_num
  have h2 : 10 - 4 = 6 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end lovely_books_earning_l325_325473


namespace amount_in_excess_l325_325806

theorem amount_in_excess (total_value : ℝ) (tax_paid : ℝ) (tax_rate : ℝ) (X : ℝ) :
  total_value = 2570 → tax_paid = 109.90 → tax_rate = 0.07 → 0.07 * (2570 - X) = 109.90 → X = 1000 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  solve_by_elim

end amount_in_excess_l325_325806


namespace graph_comparison_l325_325035

theorem graph_comparison :
  (∀ x : ℝ, (x^2 - x + 3) < (x^2 - x + 5)) :=
by
  sorry

end graph_comparison_l325_325035


namespace tan_product_simplification_l325_325355

theorem tan_product_simplification :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 :=
by
  have h : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have tan_addition :
    ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) := Real.tan_add
  sorry

end tan_product_simplification_l325_325355


namespace derivative_at_point_pi_over_4_value_of_derivative_at_pi_over_4_l325_325074

theorem derivative_at_point_pi_over_4 (x : ℝ) (f : ℝ → ℝ) (h : f x = 2 * Real.sin x + 1)
  : HasDerivAt f (2 * Real.cos (π / 4)) (π / 4) :=
sorry

theorem value_of_derivative_at_pi_over_4 (h : f' (π / 4) = 2 * Real.cos (π / 4)) 
  : f' (π / 4) = Real.sqrt 2 :=
sorry

end derivative_at_point_pi_over_4_value_of_derivative_at_pi_over_4_l325_325074


namespace triangle_area_is_sqrt2_l325_325504

-- Definitions of polynomial conditions and roots
def polynomial (x : ℝ) : Prop := x^3 - 4 * x^2 + 5 * x - 1 = 0

def is_root (a b c : ℝ) : Prop := polynomial a ∧ polynomial b ∧ polynomial c

def sum_of_roots (a b c : ℝ) : Prop := a + b + c = 4

def product_of_roots_two_at_a_time (a b c : ℝ) : Prop := a * b + a * c + b * c = 5

def product_of_roots (a b c : ℝ) : Prop := a * b * c = 1

-- Proof problem: the area of the triangle with sides a, b, and c
theorem triangle_area_is_sqrt2 
  (a b c : ℝ) 
  (hroots : is_root a ∧ is_root b ∧ is_root c)
  (hsum : sum_of_roots a b c)
  (hprod2 : product_of_roots_two_at_a_time a b c)
  (hprod : product_of_roots a b c)
  : ∃ K : ℝ, K = sqrt 2 := 
sorry

end triangle_area_is_sqrt2_l325_325504


namespace sequence_parity_behavior_l325_325538

noncomputable def a1 := Real.sqrt 2008
noncomputable def a2 := Real.sqrt 2009

def f (n : ℕ) : ℕ := ⌊(2^n : ℝ) * a1⌋ + ⌊(2^n : ℝ) * a2⌋

theorem sequence_parity_behavior :
  (∃ N : ℕ, ∀ n ≥ N, even (f n)) ∨
  (∃ N : ℕ, ∀ n ≥ N, odd (f n)) →
  false :=
begin
  sorry,
end

end sequence_parity_behavior_l325_325538


namespace cube_edge_relation_l325_325403

noncomputable def cube_relation (a α β : ℝ) : Prop :=
  (a * real.cos α * real.cos β) ^ 2 +
  (a * real.sin α) ^ 2 +
  (a * real.sin α * real.cos β) ^ 2 +
  (a * real.cos α) ^ 2 +
  a ^ 2 = 2 * a ^ 2

theorem cube_edge_relation (a α β : ℝ) (h₁ : 0 ≤ a) 
  (h₂ : 0 ≤ α) (h₃ : α ≤ real.pi / 2)
  (h₄ : 0 ≤ β) (h₅ : β ≤ real.pi / 2) 
  : cube_relation a α β := 
by sorry

end cube_edge_relation_l325_325403


namespace circle_divided_into_regions_l325_325261

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l325_325261


namespace top_width_is_76_l325_325726

-- Definitions of the conditions
def bottom_width : ℝ := 4
def area : ℝ := 10290
def depth : ℝ := 257.25

-- The main theorem to prove that the top width equals 76 meters
theorem top_width_is_76 (x : ℝ) (h : 10290 = 1/2 * (x + 4) * 257.25) : x = 76 :=
by {
  sorry
}

end top_width_is_76_l325_325726


namespace line_through_point_chord_l325_325626

-- Define the given conditions using Lean definitions
def point_P : ℝ × ℝ := (-3, -3 / 2)

def circle (x y : ℝ) : Prop := x^2 + y^2 = 25

def chord_length (point1 point2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((point1.1 - point2.1)^2 + (point1.2 - point2.2)^2)

def line_equations (x y : ℝ) : Prop :=
  (x = -3) ∨ (3 * x + 4 * y + 15 = 0)

-- The proof problem statement in Lean 4
theorem line_through_point_chord (x y : ℝ) :
  (line_equations x y) ∧
  chord_length (point_P) (x, y) = 8 →
  circle x y :=
by
  sorry

end line_through_point_chord_l325_325626


namespace expand_expression_l325_325521

variable {R : Type*} [CommRing R]
variable (x y : R)

theorem expand_expression : 
  ((10 * x - 6 * y + 9) * 3 * y) = (30 * x * y - 18 * y * y + 27 * y) :=
by
  sorry

end expand_expression_l325_325521


namespace find_true_statements_l325_325566

noncomputable def statement1 (l m : Line) (α β : Plane) :=
  l ⊆ α ∧ m ⊆ α ∧ l ∥ β ∧ m ∥ β → α ∥ β

noncomputable def statement2 (l m : Line) (α β : Plane) :=
  l ⊆ α ∧ l ∥ β ∧ α ∩ β = m → l ∥ m

noncomputable def statement3 (l : Line) (α β : Plane) :=
  α ∥ β ∧ l ∥ α → l ∥ β

noncomputable def statement4 (l m : Line) (α β : Plane) :=
  l ⊥ α ∧ m ∥ l ∧ α ∥ β → m ⊥ β

theorem find_true_statements (l m : Line) (α β : Plane) :
  ¬(statement1 l m α β) ∧ statement2 l m α β ∧ ¬(statement3 l α β) ∧ statement4 l m α β :=
by
  sorry

end find_true_statements_l325_325566


namespace eccentricity_of_hyperbola_equation_of_hyperbola_l325_325118

variables (a b c : ℝ) (M : ℝ × ℝ)
variables (h_a_pos : a > 0) (h_b_pos : b > 0) (h_x0_ne_a : M.1 ≠ a ∧ M.1 ≠ -a)
variables (h_hyperbola_eq : M.1^2 / a^2 - M.2^2 / b^2 = 1)
variables (h_slope_product : (M.2 / (M.1 + a)) * (M.2 / (M.1 - a)) = 16 / 9)
variables (h_focus_to_asymptote : 4 * c / 5 = 4)
variables (h_c_rel : c^2 = a^2 + b^2)

def hyperbola_eclecticity_eq : Prop :=
  eccentricity = 5 / 3

def hyperbola_equation : Prop :=
  ∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1) ↔ (x, y) lies on the hyperbola

theorem eccentricity_of_hyperbola :
  hyperbola_eclecticity_eq a b := sorry

theorem equation_of_hyperbola :
  hyperbola_equation a b := sorry

end eccentricity_of_hyperbola_equation_of_hyperbola_l325_325118


namespace min_abs_ab_perpendicular_lines_l325_325926

theorem min_abs_ab_perpendicular_lines (a b : ℝ) (h : a * b = a ^ 2 + 1) : |a * b| = 1 :=
by sorry

end min_abs_ab_perpendicular_lines_l325_325926


namespace equilateral_midpoints_l325_325675

open Complex

noncomputable def midpoint (X Y : ℂ) : ℂ := (X + Y) / 2

theorem equilateral_midpoints {R : ℝ} (A B C D E F : ℂ)
  (hA : abs A = R) (hB : abs B = R) (hC : abs C = R) (hD : abs D = R) (hE : abs E = R) (hF : abs F = R)
  (hAB : abs (B - A) = R) (hCD : abs (D - C) = R) (hEF : abs (F - E) = R) :
  abs ((midpoint B C) - (midpoint D E)) = abs ((midpoint D E) - (midpoint F A)) ∧
  abs ((midpoint D E) - (midpoint F A)) = abs ((midpoint F A) - (midpoint B C)) := 
sorry

end equilateral_midpoints_l325_325675


namespace repeating_decimals_subtraction_l325_325048

def x : Rat := 1 / 3
def y : Rat := 2 / 99

theorem repeating_decimals_subtraction :
  x - y = 31 / 99 :=
sorry

end repeating_decimals_subtraction_l325_325048


namespace max_S_over_a_l325_325201

variable {α : Type*} [LinearOrder α] [DivisionRing α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def S (a : ℕ → α) : ℕ → α
| 0     => 0
| (n+1) => S a n + a (n+1)

theorem max_S_over_a 
  {a : ℕ → α} 
  (h_arith : is_arithmetic_sequence a) 
  (h_S15 : S a 15 > 0) 
  (h_S16 : S a 16 < 0) :
  ∃ (n ≤ 15), (∀ m ≤ 15, (S a n / a n) ≥ (S a m / a m)) ∧ n = 8 :=
sorry

end max_S_over_a_l325_325201


namespace car_average_speed_l325_325452

noncomputable def average_speed (segments : List (ℕ × ℕ)) : ℕ :=
  let total_distance := segments.foldr (λ (pair : ℕ × ℕ) acc => acc + pair.2) 0
  let total_time := segments.foldr (λ (pair : ℕ × ℕ) acc => acc + (pair.2.toRat / pair.1.toRat).toNat) 0
  total_distance / total_time

theorem car_average_speed :
  average_speed [(40, 20), (50, 25), (60, 45), (48, 12)] = 51 := 
by
  sorry

end car_average_speed_l325_325452


namespace find_m_l325_325551

theorem find_m (m : ℝ) : 
  let a := (m, 1)
      b := (1, 2) in
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = a.1^2 + a.2^2 + b.1^2 + b.2^2 → m = -2 := 
by
  sorry

end find_m_l325_325551


namespace circle_divided_into_regions_l325_325256

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l325_325256


namespace repeating_decimal_fraction_sum_l325_325163

theorem repeating_decimal_fraction_sum : 
  let x := 36 / 99 in
  let a := 4 in
  let b := 11 in
  gcd a b = 1 ∧ (a : ℚ) / (b : ℚ) = x → a + b = 15 :=
by
  sorry

end repeating_decimal_fraction_sum_l325_325163


namespace sum_of_multiples_of_4_between_63_and_151_l325_325778

theorem sum_of_multiples_of_4_between_63_and_151 :
  (∑ x in finset.Icc 64 148, if x % 4 = 0 then x else 0) = 2332 :=
by
  sorry

end sum_of_multiples_of_4_between_63_and_151_l325_325778


namespace find_x_l325_325385

theorem find_x : 
  ∃ x : ℤ, 
  (∃ s : set ℤ, s = {12, 38, 45, x, 14} ∧ 
   (median s = (mean s) - 5) ∧ 
   x < 0)
→ 
  x = -14 :=
begin
  -- The proof will go here
  sorry
end

end find_x_l325_325385


namespace part1_part2_part3_l325_325911

noncomputable theory

variable (a_n b_n : ℕ → ℤ) (c d : ℤ) (A B C D G H n : ℕ)
variable (ac_seq : ∀ n, a_n n = c + (n - 1) * d)
variable (geo_seq : ∀ n, b_n n = d * c^(n - 1))
variable (seq_order : a_n 1 < b_n 1 ∧ b_n 1 < a_n 2 ∧ a_n 2 < b_n 2 ∧ b_n 2 < a_n 3)
variable (sum_A : A = ∑ i in finset.range n, a_n i)
variable (sum_B : B = ∑ i in finset.range (2*n), a_n i - ∑ i in finset.range n, a_n i)
variable (sum_C : C = ∑ i in finset.range (3*n), a_n i - ∑ i in finset.range (2*n), a_n i)
variable (sum_D : D = ∑ i in finset.range n, b_n i)
variable (sum_G : G = ∑ i in finset.range (2*n), b_n i)
variable (sum_H : H = ∑ i in finset.range (3*n), b_n i)

theorem part1 : 0 < c ∧ c < d ∧ c = 2 := sorry

theorem part2 : (B^2 - A * C) / (A - C)^2 = 1 / 4 := sorry

theorem part3 : H = (G^2 / D) + D - G := sorry

end part1_part2_part3_l325_325911


namespace house_cost_ratio_l325_325690

theorem house_cost_ratio {base_salary commission house_A_cost total_income : ℕ}
    (H_base_salary: base_salary = 3000)
    (H_commission: commission = 2)
    (H_house_A_cost: house_A_cost = 60000)
    (H_total_income: total_income = 8000)
    (H_total_sales_price: ℕ)
    (H_house_B_cost: ℕ)
    (H_house_C_cost: ℕ)
    (H_m: ℕ)
    (h1: total_income - base_salary = 5000)
    (h2: total_sales_price * commission / 100 = 5000)
    (h3: total_sales_price = 250000)
    (h4: house_B_cost = 3 * house_A_cost)
    (h5: total_sales_price = house_A_cost + house_B_cost + house_C_cost)
    (h6: house_C_cost = m * house_A_cost - 110000)
  : m = 2 :=
by
  sorry

end house_cost_ratio_l325_325690


namespace count_six_digit_numbers_with_at_least_one_zero_l325_325612

theorem count_six_digit_numbers_with_at_least_one_zero : 
  900000 - 531441 = 368559 :=
by
  sorry

end count_six_digit_numbers_with_at_least_one_zero_l325_325612


namespace circle_division_l325_325224

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l325_325224


namespace Q_has_no_zeros_in_interval_l325_325034

def Q (x : ℝ) : ℂ := 2 + exp (complex.I * x) + exp (2 * complex.I * x) - exp (3 * complex.I * x)

theorem Q_has_no_zeros_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x < 2 * real.pi → Q x ≠ 0 := 
by
  intro x hx
  sorry

end Q_has_no_zeros_in_interval_l325_325034


namespace min_packs_needed_for_soda_l325_325719

def soda_pack_sizes : List ℕ := [8, 15, 30]
def total_cans_needed : ℕ := 120

theorem min_packs_needed_for_soda : ∃ n, n = 4 ∧
  (∀ p ∈ {a // (a ∈ soda_pack_sizes)}, (n*p) ≤ total_cans_needed) ∧
  (∀ m, m < n → ∀ q ∈ {a // (a ∈ soda_pack_sizes)}, (m*q) < total_cans_needed) := by
  sorry

end min_packs_needed_for_soda_l325_325719


namespace part1_S_value_part2_triangle_area_l325_325563

theorem part1_S_value (n : ℕ) (hn : n = 2023) : 
  ∏ i in finset.range(2023) \ {0, 1}, 
  (1 - (1 / (i + 1 : ℝ))^2) = (2024 / 4046 : ℝ) := 
sorry

theorem part2_triangle_area (k n : ℕ) (hk : k = 2023) (hn : n = 2023) : 
  let S := (1 / 2) * ((n + 1) / n : ℝ) in
  let A := (n, S - (1 / 2) : ℝ) in
  let B := ((S - (1 / 2)) / k : ℝ, 0) in
  let P := (n, 0) in
  let area_ΔABP := n * ((S - (1 / 2)) / k : ℝ) / 2 in
  area_ΔABP = 1 :=
sorry

end part1_S_value_part2_triangle_area_l325_325563


namespace systematic_sampling_l325_325453

theorem systematic_sampling (E P: ℕ) (a b: ℕ) (g: ℕ) 
  (hE: E = 840)
  (hP: P = 42)
  (ha: a = 61)
  (hb: b = 140)
  (hg: g = E / P)
  (hEpos: 0 < E)
  (hPpos: 0 < P)
  (hgpos: 0 < g):
  (b - a + 1) / g = 4 := 
by
  sorry

end systematic_sampling_l325_325453


namespace triangle_area_in_square_l325_325336

theorem triangle_area_in_square (points : Fin 9 → (ℝ × ℝ)) :
  (∀ (x : Fin 9), 0 ≤ (points x).1 ∧ (points x).1 ≤ 1 ∧ 0 ≤ (points x).2 ∧ (points x).2 ≤ 1) →
  ∃ (a b c : Fin 9),
    let triangle_area := (λ a b c : (ℝ × ℝ), 0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))) in
    triangle_area (points a) (points b) (points c) ≤ 1 / 8 := by
  sorry

end triangle_area_in_square_l325_325336


namespace shift_graph_of_sine_function_l325_325927

theorem shift_graph_of_sine_function
  (f : ℝ → ℝ)
  (φ : ℝ)
  (h1 : ∀ x, f x = 2 * Real.sin (3 * x + φ))
  (h2 : 0 < φ ∧ φ < π / 2)
  (h3 : ∃ x, f φ = f (-x + 2 * φ) ∧ f φ = 0) :
  ∃ c d, (∀ x, 2 * Real.sin (3 * x + φ) = 2 * Real.sin 3 (x + c) + d) ∧
  c = -π / 12 ∧ d = 1 := by
  sorry

end shift_graph_of_sine_function_l325_325927


namespace average_is_3_5_l325_325389

def student_participations : List ℕ := [2, 1, 3, 3, 4, 5, 3, 6, 5, 3]

noncomputable def total_students : ℕ := 10

noncomputable def average_participations : Float := 
  (student_participations.foldl (· + ·) 0 : Float) / total_students

theorem average_is_3_5 : average_participations = 3.5 := 
by 
  -- proof will be here
  sorry

end average_is_3_5_l325_325389


namespace problem_1_problem_2_l325_325108

noncomputable def f1 (x : ℝ) (a b c : ℝ) : ℝ := a * x ^ 2 + b * x + c

noncomputable def F1 (x : ℝ) : ℝ :=
if x > 0 then f1 x 1 2 1 else -f1 x 1 2 1

theorem problem_1 : F1 2 + F1 (-2) = 8 := 
by
  sorry

noncomputable def f2 (x : ℝ) (b : ℝ) : ℝ := x ^ 2 + b * x

theorem problem_2 (b : ℝ) : 
  (-1 ≤ ∀ (x : ℝ) (W : (0 < x) ∧ (x ≤ 1)), f2 x b) 
  ∧ (∀ (x : ℝ) (W : (0 < x) ∧ (x ≤ 1)), f2 x b ≤ 1) → 
  -2 ≤ b ∧ b ≤ 0 :=
by
  sorry

end problem_1_problem_2_l325_325108


namespace problem_statement_l325_325037

def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def noDigit2 (n : ℕ) : Prop := ¬ ∃ d ∈ [2], n.digits ℕ contains d
def isOdd (n : ℕ) : Prop := n % 2 = 1
def isMultipleOf3 (n : ℕ) : Prop := n % 3 = 0
def lessThan5000 (n : ℕ) : Prop := n < 5000

def C : Finset ℕ := (Finset.range 10000).filter (λ n, isFourDigit n ∧ noDigit2 n ∧ isOdd n)
def D : Finset ℕ := (Finset.range 5000).filter (λ n, isFourDigit n ∧ isMultipleOf3 n)

theorem problem_statement : C.card + D.card = 4573 := by sorry

end problem_statement_l325_325037


namespace find_m_l325_325094

open Classical

noncomputable def vec_a : ℝ × ℝ × ℝ := (-1, 1, 3)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ × ℝ := (1, 3, m)

def perp (u v : ℝ × ℝ × ℝ) : Prop := (u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0)

theorem find_m (m : ℝ) : perp (2 • vec_a + vec_b m) vec_a ↔ m = -8 := by
  sorry

end find_m_l325_325094


namespace indeterminate_C_l325_325207

variable (m n C : ℝ)

theorem indeterminate_C (h1 : m = 8 * n + C)
                      (h2 : m + 2 = 8 * (n + 0.25) + C) : 
                      False :=
by
  sorry

end indeterminate_C_l325_325207


namespace polynomial_factorization_l325_325051

theorem polynomial_factorization :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 = 5 * x^4 + 180 * x^3 + 1431 * x^2 + 4900 * x + 5159 :=
by sorry

end polynomial_factorization_l325_325051


namespace marks_per_correct_answer_l325_325197

-- Definition and conditions
def total_questions : ℕ := 75
def total_marks : ℕ := 125
def correct_answers : ℕ := 40
def incorrect_answers : ℕ := total_questions - correct_answers
def marks_lost_per_incorrect : ℕ := 1

-- Proving the mark per correct answer
theorem marks_per_correct_answer :
  ∃ x : ℕ, (correct_answers * x - incorrect_answers * marks_lost_per_incorrect = total_marks) ∧ x = 4 :=
by
  use 4
  split
  · -- Prove the equation holds
    calc
      (correct_answers * 4 - incorrect_answers * marks_lost_per_incorrect)
          = (40 * 4 - (total_questions - correct_answers) * 1) : by rfl
      ... = (160 - 35) : by rw [total_questions, correct_answers, sub_self]
      ... = 125 : by rfl
  · -- Prove x = 4
    rfl

end marks_per_correct_answer_l325_325197


namespace lowest_score_is_46_l325_325369

variable (scores : Fin 12 → ℕ)
variable (highest lowest : ℕ)

theorem lowest_score_is_46 (h1 : (∑ i, scores i) / 12 = 82)
                          (h2 : (∑ i in Finset.erase (Finset.erase Finset.univ (Fin 0)) (Fin 11), scores i) / 10 = 84)
                          (h3 : highest = 98) 
                          (h4 : (∑ i, scores i) - (∑ i in Finset.erase (Finset.erase Finset.univ (Fin 0)) (Fin 11), scores i) = highest + lowest) :
                          lowest = 46 := 
sorry

end lowest_score_is_46_l325_325369


namespace ellipse_curve_relation_l325_325827

theorem ellipse_curve_relation (a b c : ℝ)
    (h_curve : ∀ x, ∃ y, y = c * real.sin (x / a))
    (h_rolling_no_slipping : 2 * real.pi * real.sqrt((a ^ 2 + b ^ 2) / 2) = 2 * real.pi * a)
    (h_curvature_matching : c / (a * a) = 1 / a) :
    a = b ∧ b = c :=
begin
    sorry
end

end ellipse_curve_relation_l325_325827


namespace jim_saves_money_l325_325660

noncomputable def cost_after_tax (cost : ℚ) (tax_rate : ℚ) : ℚ :=
  cost + (cost * tax_rate)

noncomputable def total_cost (quantity_needed : ℚ) (container_size : ℚ) (container_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  let num_containers := (quantity_needed / container_size).ceil
  in cost_after_tax (num_containers * container_cost) tax_rate

noncomputable def savings (min_option_cost : ℚ) (best_option_cost : ℚ) : ℚ :=
  min_option_cost - best_option_cost

theorem jim_saves_money :
  let store_a_cost := total_cost 75 128 8 0.05 in
  let store_b_cost := total_cost 75 48 5 0.05 in
  let store_c_cost := total_cost 75 24 3.5 0.05 in
  let best_option_cost := store_a_cost in
  let min_option_cost := store_c_cost in
  savings min_option_cost best_option_cost = 6.30 := by
  sorry

end jim_saves_money_l325_325660


namespace geometric_progression_product_l325_325031

variables {n : ℕ} {b q S S' P : ℝ} 

theorem geometric_progression_product (hb : b ≠ 0) (hq : q ≠ 1)
  (hP : P = b^n * q^(n*(n-1)/2))
  (hS : S = b * (1 - q^n) / (1 - q))
  (hS' : S' = (q^n - 1) / (b * (q - 1)))
  : P = (S * S')^(n/2) := 
sorry

end geometric_progression_product_l325_325031


namespace rectangle_integer_side_length_l325_325811

theorem rectangle_integer_side_length
  (original_rectangle : Rect)
  (tile : Rect → Prop)
  (h1 : ∀ t, tile t → has_integer_side t)
  (h2 : tiles original_rectangle tile) :
  has_integer_side original_rectangle :=
sorry

-- Definitions
structure Rect := (width height : ℝ)
def has_integer_side (r : Rect) : Prop := ∃ n : ℕ, r.width = n ∨ r.height = n
def tiles (r : Rect) (tile : Rect → Prop) : Prop := ∃ tiles : list Rect, (∀ t ∈ tiles, tile t) ∧ (∀ t ∈ tiles, t.width ≤ r.width ∧ t.height ≤ r.height) ∧ sum (tiles.map (λ t, t.width * t.height)) = r.width * r.height

end rectangle_integer_side_length_l325_325811


namespace binom_20_10_l325_325570

open_locale nat

theorem binom_20_10 :
  (nat.choose 18 8 = 31824) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 172822 :=
by {
  intros h1 h2 h3,
  sorry
}

end binom_20_10_l325_325570


namespace dan_destroyed_l325_325899

def balloons_initial (fred: ℝ) (sam: ℝ) : ℝ := fred + sam

theorem dan_destroyed (fred: ℝ) (sam: ℝ) (final_balloons: ℝ) (destroyed_balloons: ℝ) :
  fred = 10.0 →
  sam = 46.0 →
  final_balloons = 40.0 →
  destroyed_balloons = (balloons_initial fred sam) - final_balloons →
  destroyed_balloons = 16.0 := by
  intros h1 h2 h3 h4
  sorry

end dan_destroyed_l325_325899


namespace angle_KAD_eq_angle_KCD_l325_325993

variable (A B C D K : Type)
variable [plane_convex_quadrilateral A B C D]
variable (h1 : A.dist B = B.dist D)
variable (h2 : ∠ A B D = ∠ D B C)
variable (K : line_segment B D)
variable (h3 : K.midpoint = B.dist C)

theorem angle_KAD_eq_angle_KCD :
  ∠ K A D = ∠ K C D :=
by
  sorry

end angle_KAD_eq_angle_KCD_l325_325993


namespace range_of_a_minus_b_l325_325549

theorem range_of_a_minus_b (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 1) (h₃ : 1 < b) (h₄ : b < 3) : 
  -4 < a - b ∧ a - b < 0 := by
  sorry

end range_of_a_minus_b_l325_325549


namespace typing_speed_ratio_l325_325435

-- Define Tim's and Tom's typing speeds
variables (T t : ℝ)

-- Conditions from the problem
def condition1 : Prop := T + t = 15
def condition2 : Prop := T + 1.6 * t = 18

-- The proposition to prove: the ratio of Tom's typing speed to Tim's is 1:2
theorem typing_speed_ratio (h1 : condition1 T t) (h2 : condition2 T t) : t / T = 1 / 2 :=
sorry

end typing_speed_ratio_l325_325435


namespace winning_candidate_percentage_l325_325747

theorem winning_candidate_percentage (v1 v2 v3 : ℕ) (total_votes : ℕ) (winning_votes : ℕ) (percentage : ℚ) :
  v1 = 2500 → v2 = 5000 → v3 = 15000 → total_votes = v1 + v2 + v3 → winning_votes = 15000 →
  percentage = (winning_votes : ℚ) / (total_votes : ℚ) * 100 → percentage = 75 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h5] at h4
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end winning_candidate_percentage_l325_325747


namespace females_with_advanced_degrees_only_l325_325981

theorem females_with_advanced_degrees_only (T F A C_only M M_C M_M F_M : ℕ)
  (hT : T = 148)
  (hF : F = 92)
  (hA : A = 78)
  (hC_only : C_only = 55)
  (hM : M = 15)
  (hM_C : M_C = 31)
  (hM_M : M_M = 8)
  (hF_M : F_M = 10) :
  (F - (T - F) = 56) ∧  -- Male employees
  (let males_adv_degrees := M_M + (56 - 31 - 8) in A - F_M - M_M - males_adv_degrees = 35) :=
by {
  -- Definitions:
  let males := T - F,
  have h1 : males = 56 := by rw [hT, hF]; simp,
  let males_adv_degrees := M_M + (males - M_C - M_M),
  have h2 : males_adv_degrees = 25 := by rw [h1, hM_M, hM_C]; simp,
  let employees_adv_degrees_only := A - F_M - M_M,
  have h3 : employees_adv_degrees_only = 60 := by rw [hA, hF_M, hM_M]; simp,
  have females_adv_degrees_only := employees_adv_degrees_only - males_adv_degrees,
  show (F - (T - F) = 56) ∧ (females_adv_degrees_only = 35),
  split,
  { exact h1 },
  { rw [h3, h2], simp }
}

end females_with_advanced_degrees_only_l325_325981


namespace monotone_decreasing_interval_l325_325387

open Real

theorem monotone_decreasing_interval (k : ℤ) : 
  ∀ x ∈ Set.Icc (k * π + π / 8) (k * π + 5 * π / 8), 
    MonotoneDecreasing (λ x, 2 * cos (2 * x - π / 4)) :=
by
  sorry

end monotone_decreasing_interval_l325_325387


namespace sandro_children_l325_325329

variables (sons daughters children : ℕ)

-- Conditions
def has_six_times_daughters (sons daughters : ℕ) : Prop := daughters = 6 * sons
def has_three_sons (sons : ℕ) : Prop := sons = 3

-- Theorem to be proven
theorem sandro_children (h1 : has_six_times_daughters sons daughters) (h2 : has_three_sons sons) : children = 21 :=
by
  -- Definitions from the conditions
  unfold has_six_times_daughters has_three_sons at h1 h2

  -- Skip the proof
  sorry

end sandro_children_l325_325329


namespace sin_sum_geq_zero_l325_325288

open Real

theorem sin_sum_geq_zero (A B C : ℝ)
    (hA : 0 < A ∧ A < π / 2)
    (hB : 0 < B ∧ B < π / 2)
    (hC : 0 < C ∧ C < π / 2) :
    let X := (sin A * sin (A - B) * sin (A - C)) / sin (B + C)
    let Y := (sin B * sin (B - C) * sin (B - A)) / sin (C + A)
    let Z := (sin C * sin (C - A) * sin (C - B)) / sin (A + B)
    in X + Y + Z ≥ 0 := 
by {
  sorry
}

end sin_sum_geq_zero_l325_325288


namespace simplify_tan_expr_l325_325348

-- Definition of the tangents of 30 degrees and 15 degrees
def tan_30 : ℝ := Real.tan (Real.pi / 6)
def tan_15 : ℝ := Real.tan (Real.pi / 12)

-- Theorem stating that (1 + tan_30) * (1 + tan_15) = 2
theorem simplify_tan_expr : (1 + tan_30) * (1 + tan_15) = 2 :=
by
  sorry

end simplify_tan_expr_l325_325348


namespace average_brown_mms_l325_325712

def brown_mms_bag_1 := 9
def brown_mms_bag_2 := 12
def brown_mms_bag_3 := 8
def brown_mms_bag_4 := 8
def brown_mms_bag_5 := 3

def total_brown_mms : ℕ := brown_mms_bag_1 + brown_mms_bag_2 + brown_mms_bag_3 + brown_mms_bag_4 + brown_mms_bag_5

theorem average_brown_mms :
  (total_brown_mms / 5) = 8 := by
  rw [total_brown_mms]
  norm_num
  sorry

end average_brown_mms_l325_325712


namespace negation_problem_l325_325553

variable {a b c : ℝ}

theorem negation_problem (h : a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) : 
  a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3 :=
sorry

end negation_problem_l325_325553


namespace work_completion_time_extension_l325_325658

theorem work_completion_time_extension
    (total_men : ℕ) (initial_days : ℕ) (remaining_men : ℕ) (man_days : ℕ) :
    total_men = 100 →
    initial_days = 20 →
    remaining_men = 50 →
    man_days = total_men * initial_days →
    (man_days / remaining_men) - initial_days = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end work_completion_time_extension_l325_325658


namespace part1_part2_l325_325109
-- Import all necessary libraries

-- Define the function f(x) for part (1)
def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - 0.5 * a * x ^ 2 - x

-- Part 1: Prove that f(x) is a decreasing function when a = 1
theorem part1 (a : ℝ) : a = 1 → ∀ x > 0, (Real.log x - x) ≤ 0 :=
sorry

-- Part 2: Prove the range of λ when ln x1 + λ ln x2 > 1 + λ
theorem part2 (x1 x2 : ℝ) (λ : ℝ) (h₁ : 0 < x1) (h₂ : 0 < x2) (h₃ : x1 < x2) :
  (Real.log x1 + λ * Real.log x2 > 1 + λ) → 1 ≤ λ :=
sorry

end part1_part2_l325_325109


namespace problem1_problem2_problem3_l325_325025

-- Problem 1
theorem problem1 : (real.cbrt ((-4: ℝ) ^ 3)) - ( (1 / 2) ^ 0) + (0.25 ^ (1 / 2) * (real.sqrt 2) ^ 4) = -3 := by
  sorry

-- Problem 2
theorem problem2 : (real.log10 4 + real.log10 25 + 4 ^ (-1 / 2) - (4 - real.pi) ^ 0) = 3 / 2 := by
  sorry

-- Problem 3
theorem problem3 : ((real.log10 8 + real.log10 125 - real.log10 2 - real.log10 5) / (real.log10 (real.sqrt 10) * real.log10 0.1)) = -4 := by
  sorry

end problem1_problem2_problem3_l325_325025


namespace simplify_tan_expression_l325_325360

theorem simplify_tan_expression :
  (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 2 := 
by 
  -- Angle addition formula for tangent
  have h : Real.tan (Real.pi / 4) = Real.tan (Real.pi / 6 + Real.pi / 12), 
  from by rw [Real.tan_add]; exact Real.tan_pi_div_four,
  -- Given that tan 45° = 1
  have h1 : Real.tan (Real.pi / 4) = 1, from Real.tan_pi_div_four,
  -- Derive the known value
  rw [Real.tan_pi_div_four, h] at h1,
  -- Simplify using the derived value
  suffices : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 12)) = 
             1 + Real.tan (Real.pi / 6) + Real.tan (Real.pi / 12) + Real.tan (Real.pi / 6) * Real.tan (Real.pi / 12), 
  from by rw this; simp [←h1],
  sorry

end simplify_tan_expression_l325_325360


namespace ellipse_circle_radius_l325_325799

/-- An ellipse given by the equation 4x^2 + 9y^2 = 36 -/
def ellipse_eq (x y : ℝ) : Prop :=
  4 * x^2 + 9 * y^2 = 36

/-- Foci distance c of the ellipse calculated from a^2 - b^2 -/
def foci_dist : ℝ :=
  Real.sqrt (9 - 4)

/-- The circle passes through both foci and exactly four points of the ellipse with radius r -/
def circle (x0 r : ℝ) (x y : ℝ) : Prop :=
  (x - x0)^2 + y^2 = r^2

/-- The interval [a, b) for r with a circle passing through both foci and four points of the ellipse -/
def possible_radius_interval : Set ℝ :=
  Set.Ico (Real.sqrt 5) (Real.sqrt 5 + 7)

/-- Prove that a + b = sqrt(5) + 7 -/
theorem ellipse_circle_radius :
  let a := Real.sqrt 5
  let b := Real.sqrt 5 + 7
  a + b = Real.sqrt 5 + 7 := 
  by
    /- Lean Code proof steps are skipped for providing definitions and statement only -/
    sorry

end ellipse_circle_radius_l325_325799


namespace area_section_AMK_l325_325638

variables (A B C D A1 B1 C1 D1 M K : Point)
variables (AB AC AA1 : ℝ)

-- Given conditions
def conditions (AB AC AA1 : ℝ) (A B C D A1 B1 C1 D1 : Point) : Prop :=
  distance A B = 2 ∧ distance A C = 3 ∧ distance A A1 = 4

-- Prove the area of the section AMK
theorem area_section_AMK (h : conditions AB AC AA1 A B C D A1 B1 C1 D1) :
  let M := midpoint B B1,
      K := midpoint D D1,
      A := Point.mk 0 0 0 in
  let area_section := calc_area_section A M K in
  area_section = sqrt 22 := 
sorry

end area_section_AMK_l325_325638


namespace units_digit_of_sum_sequence_is_8_l325_325424

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit_sum_sequence : ℕ :=
  let term (n : ℕ) := (factorial n + n * n) % 10
  (term 1 + term 2 + term 3 + term 4 + term 5 + term 6 + term 7 + term 8 + term 9) % 10

theorem units_digit_of_sum_sequence_is_8 :
  units_digit_sum_sequence = 8 :=
sorry

end units_digit_of_sum_sequence_is_8_l325_325424


namespace correct_option_l325_325605

def M : Set ℝ := { x | x^2 - 4 = 0 }

theorem correct_option : -2 ∈ M :=
by
  -- Definitions and conditions from the problem
  -- Set M is defined as the set of all x such that x^2 - 4 = 0
  have hM : M = { x | x^2 - 4 = 0 } := rfl
  -- Goal is to show that -2 belongs to the set M
  sorry

end correct_option_l325_325605


namespace find_x_l325_325760

theorem find_x (x : ℝ) : 0.5 * x + (0.3 * 0.2) = 0.26 ↔ x = 0.4 := by
  sorry

end find_x_l325_325760


namespace complex_square_l325_325624

-- Define z and the condition on i
def z := 5 + (6 * Complex.I)
axiom i_squared : Complex.I ^ 2 = -1

-- State the theorem to prove z^2 = -11 + 60i
theorem complex_square : z ^ 2 = -11 + (60 * Complex.I) := by {
  sorry
}

end complex_square_l325_325624


namespace quadratic_roots_form_l325_325059

theorem quadratic_roots_form (c : ℝ) (h : ∀ (x : ℝ), x^2 - 3*x + c = 0 ↔ (x = (3 + sqrt c) / 2 ∨ x = (3 - sqrt c) / 2)) :
  c = 9 / 5 :=
by
  sorry

end quadratic_roots_form_l325_325059


namespace smaller_angle_at_8_15_l325_325022

noncomputable def hour_hand_position (h m : ℕ) : ℝ := (↑h % 12) * 30 + (↑m / 60) * 30

noncomputable def minute_hand_position (m : ℕ) : ℝ := ↑m / 60 * 360

noncomputable def angle_between_hands (h m : ℕ) : ℝ :=
  let θ := |hour_hand_position h m - minute_hand_position m|
  min θ (360 - θ)

theorem smaller_angle_at_8_15 : angle_between_hands 8 15 = 157.5 := by
  sorry

end smaller_angle_at_8_15_l325_325022


namespace find_a_l325_325590

noncomputable def f (x a : ℝ) : ℝ := 4 * x ^ 2 - 4 * a * x + a ^ 2 - 2 * a + 2

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 2 → f y a ≤ f x a) ∧ f 0 a = 3 ∧ f 2 a = 3 → 
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := 
sorry

end find_a_l325_325590


namespace problem_1_problem_2_l325_325598

-- Definitions according to the conditions
def f (x a : ℝ) := |2 * x + a| + |x - 2|

-- The first part of the problem: Proof when a = -4, solve f(x) >= 6
theorem problem_1 (x : ℝ) : 
  f x (-4) ≥ 6 ↔ x ≤ 0 ∨ x ≥ 4 := by
  sorry

-- The second part of the problem: Prove the range of a for inequality f(x) >= 3a^2 - |2 - x|
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 * a^2 - |2 - x|) ↔ (-1 ≤ a ∧ a ≤ 4 / 3) := by
  sorry

end problem_1_problem_2_l325_325598


namespace bisect_secant_l325_325411

-- Define the relevant points and elements in the problem
variables {A B C D M : Point}

-- Assuming we have already defined what it means for circles to intersect at points A and B,
-- and that CD is a secant line passing through A between the two circles
-- and that the circle with diameter AB intersects CD at point M
axiom circles_intersect_at (circle1 circle2 : Circle) (h_eq : circle1.radius = circle2.radius) : Intersect circle1 circle2 A B
axiom secant_line_between_circles (circle1 circle2 : Circle) (C D : Point) (H1 : OnCircle C circle1) (H2 : OnCircle D circle2) : SecantLine CD A
axiom circle_diameter (AB_circle : Circle) (h_diameter : CircleDiameter AB_circle A B) : OnCircle M AB_circle

-- The main theorem to be proved
theorem bisect_secant (circle1 circle2 AB_circle : Circle) (h_eq : circle1.radius = circle2.radius)
                      (H1 : OnCircle C circle1) (H2 : OnCircle D circle2) (h_cd : SecantLine CD A)
                      (h_intersect : Intersect circle1 circle2 A B) (h_diameter : CircleDiameter AB_circle A B) 
                      (h_M : OnCircle M AB_circle) :
  Distance C M = Distance M D :=
by
  -- skipping proof for now
  sorry

end bisect_secant_l325_325411


namespace circle_division_l325_325226

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l325_325226
