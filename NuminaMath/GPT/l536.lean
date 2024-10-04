import Mathlib

namespace quadratic_root_shift_l536_536420

theorem quadratic_root_shift (d e : ℝ) :
  (∀ r s : ℝ, (r^2 - 2 * r + 0.5 = 0) → (r-3)^2 + (r-3) * (s-3) * d + e = 0) → e = 3.5 := 
by
  intros
  sorry

end quadratic_root_shift_l536_536420


namespace perpendicular_lines_l536_536002

open EuclideanGeometry

variables {A B C D E F T X : Point}

/-- Given triangle ABC with AB ≠ AC. Let D, E, and F be the points of tangency of the incircle with BC, CA, and AB, respectively.
    Let the perpendicular from D to EF intersect AB at X. Let Τ be the second intersection point of the circumcircles of triangles AEF and ABC.
    Prove that TX ⊥ TF. --/
theorem perpendicular_lines
    (h1 : AB ≠ AC)
    (h2 : incircle_tangent_points A B C D E F)
    (h3 : line_perpendicular D EF (line_intersects AB D X))
    (h4 : second_intersection_point_circumcircles A E F B C T):
    perpendicular TX TF := 
sorry

end perpendicular_lines_l536_536002


namespace min_removed_numbers_l536_536645

theorem min_removed_numbers : 
  ∃ S : Finset ℤ, 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 1982) ∧ 
    (∀ a b c : ℤ, a ∈ S → b ∈ S → c ∈ S → c ≠ a * b) ∧
    ∀ T : Finset ℤ, 
      ((∀ y ∈ T, 1 ≤ y ∧ y ≤ 1982) ∧ 
       (∀ p q r : ℤ, p ∈ T → q ∈ T → r ∈ T → r ≠ p * q) → 
       T.card ≥ 1982 - 43) :=
sorry

end min_removed_numbers_l536_536645


namespace range_of_a_l536_536360

def f (a x : ℝ) : ℝ := x^2 + 2 * a * x + 2

theorem range_of_a (a : ℝ) : (∀ x ∈ Iic (4 : ℝ), ∀ y ∈ Iic (4 : ℝ), x ≤ y → f a y ≤ f a x) → a ≤ -4 := 
by 
  sorry

end range_of_a_l536_536360


namespace find_general_term_find_minimum_sum_l536_536316

open Nat

-- Define the arithmetic sequence's general term
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arith_seq (a d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
variables (a d : ℤ)
axiom cond1 : a + (a + 4 * d) = -20 -- a_1 + a_5 = -20
axiom cond2 : (a + 2 * d) + (a + 7 * d) = -10 -- a_3 + a_8 = -10

-- Statements to prove
theorem find_general_term :
  let a := -14;
      d := 2 in
  ∀ n, arithmetic_seq a d n = 2 * n - 16 := sorry

theorem find_minimum_sum :
  let a := -14;
      d := 2;
      S_n (n : ℕ) := sum_arith_seq a d n in
  (∀ n, n = 7 ∨ n = 8 → S_n n = -56) ∧
  ∀ n, S_n n ≥ -56 := sorry

end find_general_term_find_minimum_sum_l536_536316


namespace num_students_l536_536156

theorem num_students (S : ℕ) (h1 : 0.5 * S = 0.5 * S)
  (h2 : (1/5) * (0.5 * S) = (1/10) * S)
  (h3 : (0.5 * S) - (1/10) * S = 160) :
  S = 400 :=
by
  sorry

end num_students_l536_536156


namespace whole_numbers_in_interval_7_4_3pi_l536_536858

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536858


namespace count_whole_numbers_in_interval_l536_536837

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536837


namespace find_T_l536_536161

theorem find_T : 
  ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (1 / 5) * (1 / 4) * 120 ∧ T = 48 :=
by
  sorry

end find_T_l536_536161


namespace angle_CAB_is_60_l536_536371

-- Define the geometric entities and conditions of the problem
variables {A B C D H O : Point}
variables (triangle_ABC : Triangle A B C)
variables (is_acute : isAcute triangle_ABC)
variables (is_altitude_CD : isAltitude C D A B)
variables (is_orthocenter_H : isOrthocenter H triangle_ABC)
variables (is_circumcenter_O : isCircumcenter O triangle_ABC)
variables (on_bisector_O : liesOnBisectorOf O D H B)

-- Assert the angle CAB is 60 degrees based on the given conditions
theorem angle_CAB_is_60 (triangle_ABC : Triangle A B C) (is_acute : isAcute triangle_ABC)
                         (is_altitude_CD : isAltitude C D A B) (is_orthocenter_H : isOrthocenter H triangle_ABC)
                         (is_circumcenter_O : isCircumcenter O triangle_ABC)
                         (on_bisector_O : liesOnBisectorOf O D H B):
  angle A C B = 60 :=
sorry

end angle_CAB_is_60_l536_536371


namespace domain_f_l536_536489

noncomputable def f (x : ℝ) := real.sqrt (2^x - 1)

theorem domain_f : {x : ℝ | 0 ≤ x} = {x : ℝ | ∃y, f y = f x} :=
by
  sorry

end domain_f_l536_536489


namespace count_ways_to_get_50_cents_with_coins_l536_536828

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536828


namespace exists_faces_with_common_vertex_same_number_l536_536265

-- Define the concept of an icosahedron with specific properties
structure Icosahedron where
  faces : Fin 20 → ℕ -- Fin 20 represents the 20 faces
  vertices : Fin 12 -- 12 vertices

-- Define that five edges meet at each vertex
def vertices_have_five_edges (I : Icosahedron) : Prop :=
  ∀ v : Fin 12, card ({f | I.faces f ≠ 0} ∩ Finset.univ) = 5

-- Define that the sum of numbers on all faces is 39
def faces_sum_to_39 (I : Icosahedron) : Prop :=
  (Finset.sum Finset.univ I.faces) = 39

-- The main theorem we need to prove
theorem exists_faces_with_common_vertex_same_number (I : Icosahedron) (h1 : vertices_have_five_edges I) (h2 : faces_sum_to_39 I) : 
  ∃ f1 f2 : Fin 20, f1 ≠ f2 ∧ f1 ∈ Finset.univ ∧ f2 ∈ Finset.univ ∧ ∃ v : Fin 12, v ∈ ({f1, f2} : Finset _) ∧ I.faces f1 = I.faces f2 := 
sorry

end exists_faces_with_common_vertex_same_number_l536_536265


namespace problem1_problem2_l536_536609

-- Problem 1: Prove that the expression simplifies correctly given the condition
theorem problem1 (a : ℝ) (h : a > 0) : (a^2 / (sqrt a * 3 * a^2) = a^(5/6)) := 
  sorry

-- Problem 2: Prove that the logarithmic expression simplifies correctly
theorem problem2 : (2 * log 2 + log 3) / (1 + (1 / 2) * log 0.36 + (1 / 3) * log 8) = 1 := 
  sorry

end problem1_problem2_l536_536609


namespace combinations_of_coins_with_50_cents_l536_536747

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536747


namespace find_higher_selling_price_l536_536357

noncomputable def cost_price : ℝ := 400
noncomputable def lower_selling_price : ℝ := 340
noncomputable def expected_higher_selling_price : ℝ := 360

theorem find_higher_selling_price
  (h1 : cost_price = 400)
  (h2 : ∀ profit_at_hsp : ℝ, profit_at_hsp = (sample rs. 400 * (5/100))),
  shnould beyadt_cost : ℝ = 340 - 400 + profit_at_hsp1 * 5)  (total_profit : 5 + 20)
  (A_cal : [profit_ate_hxp : sh 400 - 4-(80)]
  [given_anyt : profit_at_wh : 360 
  [solut_5% : ups 0, (_40),
by\[R 5 result : 

_profit_grop_low, _5 ans: 11,_then 20)= -14c (Relectotal give_any progh (story_1 over]) ⓧ),

  exist solution given (Add_proff(true: _: (-given (loss at-tax: state_given),-20 (proof.

end_0 ###.

5 % returns _hspr14: using)←

totalneen 
liht sh_provin_θ :(sp (profit_4 gives) -

end_then,

_true(};_by, 

(nith

:Given *[5-:totalg⇛]

 
end find_higher_selling_price_l536_536357


namespace projectile_reaches_100_feet_l536_536584

theorem projectile_reaches_100_feet :
  ∃ (t : ℝ), t > 0 ∧ (-16 * t ^ 2 + 80 * t = 100) ∧ (t = 2.5) := by
sorry

end projectile_reaches_100_feet_l536_536584


namespace abs_prod_diff_le_sum_abs_diff_l536_536401

theorem abs_prod_diff_le_sum_abs_diff {n : ℕ} (a b : ℕ → ℝ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → -1 ≤ a i ∧ a i ≤ 1)
  (h2 : ∀ i, 1 ≤ i → i ≤ n → -1 ≤ b i ∧ b i ≤ 1) :
  abs (∏ i in finset.range n, a i - ∏ i in finset.range n, b i) ≤ ∑ i in finset.range n, abs (a i - b i) :=
sorry

end abs_prod_diff_le_sum_abs_diff_l536_536401


namespace coin_combinations_l536_536806

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536806


namespace area_of_square_with_adjacent_points_l536_536080

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l536_536080


namespace area_of_square_l536_536060

-- We define the points as given in the conditions
def point1 : ℝ × ℝ := (1, 2)
def point2 : ℝ × ℝ := (4, 6)

-- Lean's "def" defines the concept of a square given two adjacent points.
def is_square (p1 p2: ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  in ∃ (l : ℝ), l = d ∧ (l^2 = 25)

-- The theorem assumes the points are adjacent points on a square and proves that their area is 25.
theorem area_of_square :
  is_square point1 point2 :=
by
  -- Insert formal proof here, skipped with 'sorry' for this task
  sorry

end area_of_square_l536_536060


namespace kyiv_math_olympiad_1963_problem_l536_536950

theorem kyiv_math_olympiad_1963_problem:
  ∀ (d1 d2 d3 d4 d5 d6 d7 d8 d9: ℕ), 
  {d1, d2, d3, d4, d5, d6, d7, d8, d9} = {0,1,2,3,4,5,6,7,8,9} \ 
  → 396 ∣ (3 * 10^length(d1)*d1 + d2 * 10^length(d3)*d3 + 4 * 10^length(d4)*d4 + d5 10 * 10^length(d6)*d6 + 1 * 10^length(d7)*d7 + d8 + 0 * 10^length(d9)).

end kyiv_math_olympiad_1963_problem_l536_536950


namespace platform_length_eq_train_length_l536_536141

noncomputable def length_of_train : ℝ := 900
noncomputable def speed_of_train_kmh : ℝ := 108
noncomputable def speed_of_train_mpm : ℝ := (speed_of_train_kmh * 1000) / 60
noncomputable def crossing_time_min : ℝ := 1
noncomputable def total_distance_covered : ℝ := speed_of_train_mpm * crossing_time_min

theorem platform_length_eq_train_length :
  total_distance_covered - length_of_train = length_of_train :=
by
  sorry

end platform_length_eq_train_length_l536_536141


namespace triangle_niche_dimensions_l536_536552

theorem triangle_niche_dimensions : ∀ (s : ℝ), s = 3.414 → let x := s / (2 + Real.sqrt 2) in x = 1 := 
by
  intro s hs
  let x := s / (2 + Real.sqrt 2)
  have h : x = 1 := by sorry
  exact h

# Example of verifying usage
# example : triangle_niche_dimensions 3.414 := by apply triangle_niche_dimensions 3.414 rfl

end triangle_niche_dimensions_l536_536552


namespace matrix_multiplication_correct_l536_536612

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -4], ![6, 2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![0, 3], ![-2, 1]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![8, 5], ![-4, 20]]

theorem matrix_multiplication_correct :
  A ⬝ B = C :=
by {
  -- The proof would go here, but it's omitted per instruction.
  sorry
}

end matrix_multiplication_correct_l536_536612


namespace find_AP_l536_536438

theorem find_AP 
  (P is on diagonal_AC: Point)
  (circumcenter_ABP O1: CircleCenter)
  (circumcenter_CDP O2: CircleCenter)
  (AB_eq_24: length AB = 24)
  : (angle O1 P O2 = 90°) → (length AP = 24) :=
by
  sorry

end find_AP_l536_536438


namespace rectangle_perimeter_from_squares_l536_536529

/-- 
  Given two squares, each with an area of 25 cm², placed side by side to form a rectangle,
  the perimeter of the resulting rectangle is 30 cm. 
-/
theorem rectangle_perimeter_from_squares :
  ∀ (A : ℝ), (A = 25) → 
  let s := real.sqrt A in
  let L := s + s in
  let W := s in
  2 * L + 2 * W = 30 :=
by
  intros A hA s L W
  unfold s
  unfold L
  unfold W
  sorry

end rectangle_perimeter_from_squares_l536_536529


namespace fib_mod_five_l536_536041

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

theorem fib_mod_five (a : ℕ → ℕ) (h : sequence a) (k : ℕ) : 5 ∣ a (5 * k) :=
begin
  sorry
end

end fib_mod_five_l536_536041


namespace days_for_B_l536_536571

theorem days_for_B
  (x : ℝ)
  (hA : 15 ≠ 0)
  (h_nonzero_fraction : 0.5833333333333334 ≠ 0)
  (hfraction : 0 <  0.5833333333333334 ∧ 0.5833333333333334 < 1)
  (h_fraction_work_left : 5 * (1 / 15 + 1 / x) = 0.5833333333333334) :
  x = 20 := by
  sorry

end days_for_B_l536_536571


namespace possible_values_of_xi_l536_536681

def number_of_defective_products_selected (total_products : ℕ) (defective_products : ℕ) (selected_products : ℕ) : set ℕ :=
{ n | n ≤ defective_products ∧ n ≤ selected_products ∧ defective_products - n ≤ total_products - selected_products }

theorem possible_values_of_xi :
  number_of_defective_products_selected 8 2 3 = {0, 1, 2} :=
by
  sorry

end possible_values_of_xi_l536_536681


namespace vejoTudo_TV_price_l536_536144

theorem vejoTudo_TV_price :
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ (36792 = a * 10000 + 6790 + b) ∧ (36792 / 72 = 511) := by
  use 3, 2
  split
  . exact rfl
  split
  . exact rfl
  split
  . norm_num
  . norm_num

end vejoTudo_TV_price_l536_536144


namespace evaluate_fraction_l536_536633

theorem evaluate_fraction : 
  (7/3) / (8/15) = 35/8 :=
by
  -- we don't need to provide the proof as per instructions
  sorry

end evaluate_fraction_l536_536633


namespace count_whole_numbers_in_interval_l536_536840

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536840


namespace trajectory_equation_l536_536652

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem trajectory_equation (P : ℝ × ℝ) (h : |distance P (1, 0) - P.1| = 1) :
  (P.1 ≥ 0 → P.2 ^ 2 = 4 * P.1) ∧ (P.1 < 0 → P.2 = 0) :=
by
  sorry

end trajectory_equation_l536_536652


namespace number_of_integers_in_interval_l536_536923

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536923


namespace coin_combinations_50_cents_l536_536729

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536729


namespace rainfall_sunday_l536_536994

theorem rainfall_sunday 
  (rain_sun rain_mon rain_tue : ℝ)
  (h1 : rain_mon = rain_sun + 3)
  (h2 : rain_tue = 2 * rain_mon)
  (h3 : rain_sun + rain_mon + rain_tue = 25) :
  rain_sun = 4 :=
by
  sorry

end rainfall_sunday_l536_536994


namespace square_area_adjacency_l536_536048

-- Definition of points as pairs of integers
def Point := ℤ × ℤ

-- Define the points (1,2) and (4,6)
def P1 : Point := (1, 2)
def P2 : Point := (4, 6)

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement for proving the area of a square given the side length
theorem square_area_adjacency (h : distance P1 P2 = 5) : ∃ area : ℝ, area = 25 :=
by
  use 25
  sorry

end square_area_adjacency_l536_536048


namespace coin_combinations_l536_536807

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536807


namespace square_area_from_points_l536_536064

theorem square_area_from_points :
  let P1 := (1, 2)
  let P2 := (4, 6)
  let side_length := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let area := side_length^2
  P1.1 = 1 ∧ P1.2 = 2 ∧ P2.1 = 4 ∧ P2.2 = 6 →
  area = 25 :=
by
  sorry

end square_area_from_points_l536_536064


namespace find_radius_of_sphere_l536_536224

noncomputable def radius_of_sphere (R : ℝ) : Prop :=
  ∃ a b c : ℝ, 
  (R = |a| ∧ R = |b| ∧ R = |c|) ∧ 
  ((3 - R)^2 + (2 - R)^2 + (1 - R)^2 = R^2)

theorem find_radius_of_sphere : radius_of_sphere (3 + Real.sqrt 2) ∨ radius_of_sphere (3 - Real.sqrt 2) :=
sorry

end find_radius_of_sphere_l536_536224


namespace coin_combinations_sum_50_l536_536722

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536722


namespace combinations_of_coins_with_50_cents_l536_536755

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536755


namespace linear_function_quadrant_check_l536_536502

theorem linear_function_quadrant_check (x y : ℝ) :
  let m : ℝ := -2
  let b : ℝ := -1
  let f : ℝ → ℝ := λ x, m * x + b in
  ∀ (x : ℝ), (f x = y) → ((x > 0 ∧ y > 0) → false) :=
by {
  intros x y m b f hxy good_quadrant,
  let rule := -2 * x - 1,
  rw [[←rule] rhos_rule, hxy], -- ensures f(x) equals y correctly
  simp only [not_and_simp, ne.def],
	sorry
}

end linear_function_quadrant_check_l536_536502


namespace polynomial_divisibility_l536_536282

theorem polynomial_divisibility (m : ℤ) : (4 * m + 5) ^ 2 - 9 ∣ 8 := by
  sorry

end polynomial_divisibility_l536_536282


namespace digit_count_gt_2_005_l536_536469

def isValidDigit (d : ℕ) : Prop := d >= 0 ∧ d <= 9
def isGreaterThan2_005 (d : ℕ) : Prop := 2.0 + d * 0.001 + 0.0005 > 2.005

theorem digit_count_gt_2_005 : 
  { d : ℕ // isValidDigit d ∧ isGreaterThan2_005 d }.card = 5 := 
sorry

end digit_count_gt_2_005_l536_536469


namespace count_whole_numbers_in_interval_l536_536851

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536851


namespace last_four_digits_pow_product_is_5856_l536_536241

noncomputable def product : ℕ := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349

theorem last_four_digits_pow_product_is_5856 :
  (product % 10000) ^ 4 % 10000 = 5856 := by
  sorry

end last_four_digits_pow_product_is_5856_l536_536241


namespace barium_atom_count_l536_536577

-- Defining the atomic weights of the elements
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- The molecular weight of the compound
def molecular_weight_compound : ℝ := 171

-- Number of atoms in the compound (given conditions)
def num_oxygen_atoms : ℕ := 2
def num_hydrogen_atoms : ℕ := 2

-- Defining the number of Barium atoms we need to prove
def num_barium_atoms : ℕ := 1

-- The mathematical equivalency proof statement
theorem barium_atom_count : 
  (molecular_weight_compound - (num_oxygen_atoms * atomic_weight_O + num_hydrogen_atoms * atomic_weight_H)) / atomic_weight_Ba ≈ num_barium_atoms := 
sorry

end barium_atom_count_l536_536577


namespace distance_planes_A_B_l536_536437

noncomputable def distance_between_planes : ℝ :=
  let d1 := 1
  let d2 := 2
  let a := 1
  let b := 1
  let c := 1
  (|d2 - d1|) / (Real.sqrt (a^2 + b^2 + c^2))

theorem distance_planes_A_B :
  let A := fun (x y z : ℝ) => x + y + z = 1
  let B := fun (x y z : ℝ) => x + y + z = 2
  distance_between_planes = 1 / Real.sqrt 3 :=
  by
    -- Proof steps will be here
    sorry

end distance_planes_A_B_l536_536437


namespace combinations_of_coins_l536_536756

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536756


namespace vertex_angle_of_equal_cones_l536_536208

noncomputable def vertex_angle_of_cones (n : ℕ) (n ≥ 3) : ℝ :=
  2 * Real.arcsin (Real.tan (Real.pi / n) / Real.sqrt (1 + 2 * (Real.tan (Real.pi / n) ^ 2)))

theorem vertex_angle_of_equal_cones (n : ℕ) (hn : n ≥ 3) :
  ∃ θ : ℝ, θ = vertex_angle_of_cones n hn :=
begin
  use vertex_angle_of_cones n hn,
  reflexivity,
end

end vertex_angle_of_equal_cones_l536_536208


namespace integers_between_3_and_15_with_perfect_cube_base_1331_l536_536833

theorem integers_between_3_and_15_with_perfect_cube_base_1331 :
  {n : ℕ | 3 ≤ n ∧ n ≤ 15 ∧ (∃ m : ℕ, n^3 + 3 * n^2 + 3 * n + 1 = m^3)}.card = 12 :=
by
  sorry

end integers_between_3_and_15_with_perfect_cube_base_1331_l536_536833


namespace geometric_progression_problem_l536_536258

open Real

theorem geometric_progression_problem
  (a b c r : ℝ)
  (h1 : a = 20)
  (h2 : b = 40)
  (h3 : c = 10)
  (h4 : b = r * a)
  (h5 : c = r * b) :
  (a - (b - c)) - ((a - b) - c) = 20 := by
  sorry

end geometric_progression_problem_l536_536258


namespace divides_m_l536_536034

theorem divides_m (p a m n : ℤ) (hp : p.prime)
  (S_a : ℤ → ℚ := λ a, ∑ k in finset.range (p - 1), (a ^ (k + 1) / (k + 1)))
  (hmn : S_a 3 + S_a 4 - 3 * S_a 2 = m / n) :
  p ∣ m :=
by sorry

end divides_m_l536_536034


namespace count_whole_numbers_in_interval_l536_536839

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536839


namespace number_of_pieces_of_colored_paper_distributed_l536_536566

theorem number_of_pieces_of_colored_paper_distributed 
  (number_of_students : ℕ) (pieces_per_student : ℕ) : 
  number_of_students = 230 → pieces_per_student = 15 → 
  number_of_students * pieces_per_student = 3450 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end number_of_pieces_of_colored_paper_distributed_l536_536566


namespace train_crossing_time_l536_536835

def train_length : ℕ := 100  -- length of the train in meters
def bridge_length : ℕ := 180  -- length of the bridge in meters
def train_speed_kmph : ℕ := 36  -- speed of the train in kmph

theorem train_crossing_time 
  (TL : ℕ := train_length) 
  (BL : ℕ := bridge_length) 
  (TSK : ℕ := train_speed_kmph) : 
  (TL + BL) / ((TSK * 1000) / 3600) = 28 := by
  sorry

end train_crossing_time_l536_536835


namespace count_whole_numbers_in_interval_l536_536888

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536888


namespace integer_x_cubed_prime_l536_536254

theorem integer_x_cubed_prime (x : ℕ) : 
  (∃ p : ℕ, Prime p ∧ (2^x + x^2 + 25 = p^3)) → x = 6 :=
by
  sorry

end integer_x_cubed_prime_l536_536254


namespace original_number_l536_536583

theorem original_number (x : ℝ) (h : 1.50 * x = 165) : x = 110 :=
sorry

end original_number_l536_536583


namespace unique_elements_condition_l536_536283

theorem unique_elements_condition (x : ℝ) : 
  (∀ a b, a ≠ b → a ∈ {3, x^2 - 2x} → b ∈ {3, x^2 - 2x} → x ≠ 3 ∧ x ≠ -1) :=
by
  sorry

end unique_elements_condition_l536_536283


namespace f_2016_is_cos_l536_536668

noncomputable def f_sequence : ℕ → (ℝ → ℝ)
| 0     := cos
| (n+1) := deriv (f_sequence n)

theorem f_2016_is_cos : f_sequence 2016 = cos :=
sorry

end f_2016_is_cos_l536_536668


namespace triangle_area_formula_l536_536200

def base := 2
def height := 3

-- Proof that the area of the triangle is 3 square meters.
theorem triangle_area_formula (b h : ℕ) (hb : b = base) (hh : h = height) : (b * h) / 2 = 3 := by
  simp [hb, hh]
  rfl

end triangle_area_formula_l536_536200


namespace log_relationship_l536_536648

theorem log_relationship : 
  let a := log 576 / log 16
  let b := log 24 / log 4
  a = 1.6 * b := 
by {
  -- Definitions by conditions
  let a := real.log 576 / real.log 16
  let b := real.log 24 / real.log 4
  -- Proof will go here
  sorry
}

end log_relationship_l536_536648


namespace arthur_spent_l536_536600

/-- Given prices of hamburgers and hotdogs, calculate the cost on the second day. -/
theorem arthur_spent :
  ∃ (H D : ℝ), D = 1 ∧ (3 * H + 4 * D = 10) ∧ (2 * H + 3 * D = 7) :=
begin
  use [2, 1],
  split,
  { refl, },
  split,
  { norm_num, },
  { norm_num, },
end

end arthur_spent_l536_536600


namespace first_divisor_exists_l536_536188

theorem first_divisor_exists (m d : ℕ) :
  (m % d = 47) ∧ (m % 24 = 23) ∧ (d > 47) → d = 72 :=
by
  sorry

end first_divisor_exists_l536_536188


namespace combinations_of_coins_l536_536765

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536765


namespace count_ordered_triples_l536_536416

def S : set ℕ := {x | 1 ≤ x ∧ x ≤ 2013}

theorem count_ordered_triples (A B C : set ℕ) (h₁ : A ⊆ B) (h₂ : A ∪ B ∪ C = S) :
  ∃ n, n = 5^(2013) :=
sorry

end count_ordered_triples_l536_536416


namespace find_a5_l536_536983

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a5 (a : ℕ → ℝ) (h_seq : geometric_sequence a) (h_a2 : a 2 = 2) (h_a8 : a 8 = 32) :
  a 5 = 8 :=
by
  sorry

end find_a5_l536_536983


namespace minimum_positive_period_cos_omega_cos_pi_over_2_minus_omega_eq_pi_l536_536955

theorem minimum_positive_period_cos_omega_cos_pi_over_2_minus_omega_eq_pi (ω : ℝ) (hω : 0 < ω)
  (h : ∀ x, f (x + π) = f x → (f x = cos (ω * x) * cos (π / 2 - ω * x))) : ω = 1 :=

end minimum_positive_period_cos_omega_cos_pi_over_2_minus_omega_eq_pi_l536_536955


namespace count_whole_numbers_in_interval_l536_536881

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536881


namespace minimum_value_frac_inv_is_one_third_l536_536018

noncomputable def min_value_frac_inv (x y : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ x + y = 12 then 1/x + 1/y else 0

theorem minimum_value_frac_inv_is_one_third (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x + y = 12) :
  min_value_frac_inv x y = 1/3 :=
begin
  -- Proof to be provided
  sorry
end

end minimum_value_frac_inv_is_one_third_l536_536018


namespace coin_combinations_l536_536805

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536805


namespace coin_combinations_count_l536_536738

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536738


namespace double_angle_value_l536_536627

theorem double_angle_value : 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 2 := 
sorry

end double_angle_value_l536_536627


namespace number_of_integers_in_interval_l536_536924

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536924


namespace probability_of_selecting_2_red_balls_l536_536122

-- Definitions based on conditions
def total_balls := 10
def red_balls := 4
def white_balls := 6
def balls_selected := 5
def X := Nat

-- Condition result definitions
def prob_X_2 := (X = 2) -> (P(X = 2) = 10 / 21)

-- Lean 4 statement
theorem probability_of_selecting_2_red_balls :
  ∀ (X : Nat), 
    (total_balls = red_balls + white_balls) ∧
    (balls_selected = red_balls ∨ balls_selected = white_balls) ∧
    (X = 2) →
    (P(X = 2) = 10 / 21) :=
  by
  intro X
  intros
  sorry

end probability_of_selecting_2_red_balls_l536_536122


namespace color_paint_cans_l536_536607

theorem color_paint_cans:
  ∀ (bedrooms other_rooms paint_per_room color_cans white_paint_per_can total_cans : ℕ),
  bedrooms = 3 →
  other_rooms = 2 * bedrooms →
  paint_per_room = 2 →
  total_cans = 10 →
  white_paint_per_can = 3 →
  other_rooms * paint_per_room / white_paint_per_can = 4 →
  ((bedrooms * paint_per_room) + (other_rooms * paint_per_room)) / (total_cans - 4) = 1 :=
by
  intros bedrooms other_rooms paint_per_room color_cans white_paint_per_can total_cans
  assume h1 h2 h3 h4 h5 h6
  sorry

end color_paint_cans_l536_536607


namespace incircles_tangent_incircle_tangency_points_form_inscribed_quad_l536_536104

-- Part (a)
theorem incircles_tangent (ABCD : quadrilateral) (h1 : is_circumscribed ABCD) :
  ∃ P, tangent_at_point (incircle (triangle_ABC ABCD)) (incircle (triangle_ACD ABCD)) P :=
begin
  sorry
end

-- Part (b)
theorem incircle_tangency_points_form_inscribed_quad (ABCD : quadrilateral)
  (h1 : is_circumscribed ABCD) 
  (h2 : ∃ P, tangent_at_point (incircle (triangle_ABC ABCD)) (incircle (triangle_ACD ABCD)) P) :
  ∃ K L M N, is_inscribed (quadrilateral_of_tangency_points ABCD K L M N) :=
begin
  sorry
end

end incircles_tangent_incircle_tangency_points_form_inscribed_quad_l536_536104


namespace min_degree_polynomial_l536_536119

noncomputable def P : polynomial ℚ :=
  (X^2 - 6*X + 1) * (X^2 - 10*X + 12)

lemma rational_coeffs (a b : ℚ) : polynomial ℚ :=
  (X - C a) * (X - C b)

lemma roots_inclusion :
  (P.eval (3 - real.sqrt 8) = 0) ∧ 
  (P.eval (5 + real.sqrt 13) = 0) ∧ 
  (P.eval (3 + real.sqrt 8) = 0) ∧ 
  (P.eval (5 - real.sqrt 13) = 0) :=
sorry

theorem min_degree_polynomial : ∃ P : polynomial ℚ, 
  P.eval (3 - real.sqrt 8) = 0 ∧ 
  P.eval (5 + real.sqrt 13) = 0 ∧ 
  P.eval (3 + real.sqrt 8) = 0 ∧ 
  P.eval (5 - real.sqrt 13) = 0 ∧ 
  nat_degree P = 4 :=
begin
  use P,
  split,
  exact roots_inclusion.left,
  split,
  exact roots_inclusion.right.left,
  split,
  exact roots_inclusion.right.right.left,
  split,
  exact roots_inclusion.right.right.right,
  exact eq.refl 4,
end

end min_degree_polynomial_l536_536119


namespace number_of_correct_statements_is_three_l536_536138

-- Definitions of the statements
def is_equilateral_is_isosceles : Prop := 
  ∀ (T : Triangle), T.is_equilateral → T.is_isosceles

def is_isosceles_can_be_right : Prop := 
  ∃ (T : Triangle), T.is_isosceles ∧ T.is_right

def valid_classification_by_sides : Prop := 
  ∃ (T : Triangle), T.is_equilateral ∨ T.is_isosceles ∨ T.is_scalene

def valid_classification_by_angles : Prop := 
  ∃ (T : Triangle), T.is_acute ∨ T.is_right ∨ T.is_obtuse

-- Main theorem stating the number of correct statements
theorem number_of_correct_statements_is_three :
  (is_equilateral_is_isosceles ∧
   is_isosceles_can_be_right ∧
   valid_classification_by_angles) ∧ 
   ¬valid_classification_by_sides →
  3 = 3 := by
  sorry

end number_of_correct_statements_is_three_l536_536138


namespace solution_set_inequality_l536_536134

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x, x ∈ ℝ
axiom f_at_0 : f 0 = 2
axiom f_derivative_condition : ∀ x, f x + deriv f x > 1

theorem solution_set_inequality :
  {x : ℝ | e^x * f x > e^x + 1} = {x : ℝ | x > 0} :=
by
  sorry

end solution_set_inequality_l536_536134


namespace cooks_in_restaurant_l536_536196

theorem cooks_in_restaurant
  (C W : ℕ) 
  (h1 : C * 8 = 3 * W) 
  (h2 : C * 4 = (W + 12)) :
  C = 9 :=
by
  sorry

end cooks_in_restaurant_l536_536196


namespace term_2014th_2014_sequence_l536_536137

def sumOfCubesOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.map (λ d => d^3) |>.sum

noncomputable def sequence (n : ℕ) : ℕ :=
  Nat.iterate sumOfCubesOfDigits n (n - 1)

theorem term_2014th_2014_sequence :
  sequence 2014 2014 = 370 := by
  sorry

end term_2014th_2014_sequence_l536_536137


namespace tangent_perpendicular_centers_l536_536204

noncomputable theory
open_locale classical

structure Circle (α : Type*) := 
(center : α) 
(radius : ℝ) -- Assuming radius is real for simplicity

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

def is_tangent (C O : α) (r : ℝ) (A : α) : Prop :=
dist C A = dist C O + r

variables (Ω1 Ω2 : Circle α) (C : α) (A B : α)
variable h : dist Ω1.center (Ω2.center) = Ω1.radius

theorem tangent_perpendicular_centers
  (hC : dist Ω1.center C = Ω1.radius)
  (hA : is_tangent C Ω2.center Ω2.radius A)
  (hB : is_tangent C Ω2.center Ω2.radius B) :
  ∃ A B : α, ∃ C : α, ∃ Ω1 Ω2 : Circle α,
  dist Ω1.center C = Ω1.radius ∧ 
  is_tangent C Ω2.center Ω2.radius A ∧ 
  is_tangent C Ω2.center Ω2.radius B ∧ 
  inner (A - B) (Ω1.center - Ω2.center) = 0 := 
begin
  sorry
end

end tangent_perpendicular_centers_l536_536204


namespace good_time_more_than_bad_time_l536_536578

def good_time (hour min sec : ℕ) : Prop :=
  (hour * 3600 + min * 60 + sec)/43200 < 0.5 ∨ (hour * 3600 + min * 60 + sec)/43200 > 0.5

def bad_time (hour min sec : ℕ) : Prop := ¬ (good_time hour min sec)

theorem good_time_more_than_bad_time : 
  (∑ h in finset.range 24, ∑ m in finset.range 60, ∑ s in finset.range 60, if good_time h m s then 1 else 0) 
  ≥ 
  (∑ h in finset.range 24, ∑ m in finset.range 60, ∑ s in finset.range 60, if bad_time h m s then 1 else 0) :=
sorry

end good_time_more_than_bad_time_l536_536578


namespace range_of_x_for_sqrt_l536_536944

theorem range_of_x_for_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 :=
sorry

end range_of_x_for_sqrt_l536_536944


namespace cos_2α_plus_sin_2α_eq_l536_536292

variable (α : ℝ)

theorem cos_2α_plus_sin_2α_eq : 
  tan (π + α) = 2 → cos (2 * α) + sin (2 * α) = 1 / 5 := 
by
  sorry

end cos_2α_plus_sin_2α_eq_l536_536292


namespace combinations_of_coins_l536_536774

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536774


namespace slope_ratio_symmetric_points_l536_536306

theorem slope_ratio_symmetric_points (x_A y_A x_B y_B x_2 y_2 : ℝ) (hA : x_A ≠ 0 ∧ y_A ≠ 0)
  (hB : x_B ≠ 0 ∧ y_B ≠ 0) (hsym : x_B = -x_A ∧ y_B = -y_A)
  (hellipseA : (x_A^2 / 6) + y_A^2 = 1) (hellipseD : (x_2^2 / 6) + y_2^2 = 1) :
  let k_AB := y_A / x_A in
  let k_BD := (y_2 + y_A) / (x_2 + x_A) in
  x_2 ≠ x_A ∧ x_A ≠ √6 ∧ x_A ≠ -√6 → 
    x_2 ≠ -x_A ∧ x_A ≠ -√6 ∧ x_A ≠ √6 → 
      x_2 ≠ 0 ∧ y_2 ≠ 0 ∧ 
        (k_AB / k_BD = 6) :=
sorry

end slope_ratio_symmetric_points_l536_536306


namespace find_postal_carriages_l536_536226

def postal_carriages (n : ℕ) (ps : list ℕ) : Prop := 
  ps.length = n ∧ n % 2 = 0 ∧ 
  ps.head = n ∧ ps.last = 4 * n ∧ 
  (∀ (i : ℕ), i ∈ ps → (i + 1) ∈ ps ∨ (i - 1) ∈ ps) ∧
  list.sorted ps

theorem find_postal_carriages :
  postal_carriages 4 [4, 5, 15, 16] :=
begin
  sorry
end

end find_postal_carriages_l536_536226


namespace distinguishable_large_equilateral_triangles_l536_536522

-- Definitions based on conditions.
def num_colors : ℕ := 8

def same_color_corners : ℕ := num_colors
def two_same_one_diff_colors : ℕ := num_colors * (num_colors - 1)
def all_diff_colors : ℕ := (num_colors * (num_colors - 1) * (num_colors - 2)) / 6

def corner_configurations : ℕ := same_color_corners + two_same_one_diff_colors + all_diff_colors
def triangle_between_center_and_corner : ℕ := num_colors
def center_triangle : ℕ := num_colors

def total_distinguishable_triangles : ℕ := corner_configurations * triangle_between_center_and_corner * center_triangle

theorem distinguishable_large_equilateral_triangles : total_distinguishable_triangles = 7680 :=
by
  sorry

end distinguishable_large_equilateral_triangles_l536_536522


namespace number_of_kettles_is_six_l536_536581

-- Define the conditions
def avg_pregnancies_per_kettle : ℕ := 15
def babies_per_pregnancy : ℕ := 4
def loss_fraction : ℝ := 0.25
def expected_babies : ℕ := 270

-- Define the proof problem
theorem number_of_kettles_is_six 
  (avg_pregnancies_per_kettle babies_per_pregnancy expected_babies : ℕ) 
  (loss_fraction : ℝ) 
  (h_avg_pregnancies : avg_pregnancies_per_kettle = 15) 
  (h_babies_per_pregnancy : babies_per_pregnancy = 4) 
  (h_loss_fraction : loss_fraction = 0.25) 
  (h_expected_babies : expected_babies = 270) :
  let babies_per_kettle : ℕ := avg_pregnancies_per_kettle * babies_per_pregnancy
  let surviving_babies_per_kettle : ℕ := (babies_per_kettle : ℝ) * (1 - loss_fraction) |> Int.to_nat
  let number_of_kettles := expected_babies / surviving_babies_per_kettle
  in number_of_kettles = 6 :=
by
  sorry

end number_of_kettles_is_six_l536_536581


namespace divides_expression_l536_536033

theorem divides_expression (y : ℕ) (hy : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end divides_expression_l536_536033


namespace combinations_of_coins_l536_536769

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536769


namespace area_of_square_l536_536056

-- We define the points as given in the conditions
def point1 : ℝ × ℝ := (1, 2)
def point2 : ℝ × ℝ := (4, 6)

-- Lean's "def" defines the concept of a square given two adjacent points.
def is_square (p1 p2: ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  in ∃ (l : ℝ), l = d ∧ (l^2 = 25)

-- The theorem assumes the points are adjacent points on a square and proves that their area is 25.
theorem area_of_square :
  is_square point1 point2 :=
by
  -- Insert formal proof here, skipped with 'sorry' for this task
  sorry

end area_of_square_l536_536056


namespace weekly_hours_proof_l536_536927

-- Variables representing conditions
variables (planned_weeks : ℕ) (sick_weeks : ℕ) (initial_weekly_hours : ℕ) (financial_goal : ℕ) 

-- Conditions based on given problem
def total_weeks := 15
def initial_hours := 15
def missed_weeks := 3
def goal := 4500

-- Defining what's needed to prove
def weekly_hours_needed : ℝ := (total_weeks.toReal / (total_weeks - missed_weeks).toReal) * initial_hours.toReal
def new_weekly_hours := weekly_hours_needed.ceil

-- Statement we want to prove
theorem weekly_hours_proof : 
  new_weekly_hours = 19 := 
by 
  sorry

end weekly_hours_proof_l536_536927


namespace area_of_square_with_adjacent_points_l536_536094

theorem area_of_square_with_adjacent_points (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 6)) :
  let side_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in 
  let area := side_length^2 in 
  area = 25 :=
by
  sorry

end area_of_square_with_adjacent_points_l536_536094


namespace coin_combinations_50_cents_l536_536727

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536727


namespace Pima_investment_value_at_week6_l536_536099

noncomputable def Pima_initial_investment : ℝ := 400
noncomputable def Pima_week1_gain : ℝ := 0.25
noncomputable def Pima_week1_addition : ℝ := 200
noncomputable def Pima_week2_gain : ℝ := 0.50
noncomputable def Pima_week2_withdrawal : ℝ := 150
noncomputable def Pima_week3_loss : ℝ := 0.10
noncomputable def Pima_week4_gain : ℝ := 0.20
noncomputable def Pima_week4_addition : ℝ := 100
noncomputable def Pima_week5_gain : ℝ := 0.05
noncomputable def Pima_week6_loss : ℝ := 0.15
noncomputable def Pima_week6_withdrawal : ℝ := 250
noncomputable def weekly_interest_rate : ℝ := 0.02

noncomputable def calculate_investment_value : ℝ :=
  let week0 := Pima_initial_investment
  let week1 := (week0 * (1 + Pima_week1_gain) * (1 + weekly_interest_rate)) + Pima_week1_addition
  let week2 := ((week1 * (1 + Pima_week2_gain) * (1 + weekly_interest_rate)) - Pima_week2_withdrawal)
  let week3 := (week2 * (1 - Pima_week3_loss) * (1 + weekly_interest_rate))
  let week4 := ((week3 * (1 + Pima_week4_gain) * (1 + weekly_interest_rate)) + Pima_week4_addition)
  let week5 := (week4 * (1 + Pima_week5_gain) * (1 + weekly_interest_rate))
  let week6 := ((week5 * (1 - Pima_week6_loss) * (1 + weekly_interest_rate)) - Pima_week6_withdrawal)
  week6

theorem Pima_investment_value_at_week6 : calculate_investment_value = 819.74 := 
  by
  sorry

end Pima_investment_value_at_week6_l536_536099


namespace area_triangle_ADM_l536_536977

def parallelogram_area (ABCD : Parallelogram) : ℝ := 50
def is_diagonal_bisector (AC : Diagonal) : Prop := 
  bisects_angle AC ∠BAD ∧ bisects_angle AC ∠BCD
def is_midpoint (M AC : Point) (A C : Point) : Prop := 
  midpoint M A C

theorem area_triangle_ADM (ABCD : Parallelogram) (AC : Diagonal) (M A D : Point) :
  parallelogram_area ABCD = 50 ∧
  is_diagonal_bisector AC ∧
  is_midpoint M A C →
  area (Triangle A D M) = 12.5 :=
begin
  sorry
end

end area_triangle_ADM_l536_536977


namespace complex_z_solution_l536_536376

theorem complex_z_solution :
  ∃ z : ℂ, (1 - complex.i) * z = 2 ∧ z = 1 + complex.i :=
sorry

end complex_z_solution_l536_536376


namespace fraction_simplification_l536_536244

theorem fraction_simplification :
  (1/2 * 1/3 * 1/4 * 1/5 + 3/2 * 3/4 * 3/5) / (1/2 * 2/3 * 2/5) = 41/8 :=
by
  sorry

end fraction_simplification_l536_536244


namespace count_whole_numbers_in_interval_l536_536883

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536883


namespace prove_y_equals_x_l536_536191

theorem prove_y_equals_x (x : ℝ) :
  (if x ≥ 0 then real.sqrt (x^2) else -real.sqrt (x^2)) = x :=
by
  sorry

end prove_y_equals_x_l536_536191


namespace min_inv_sum_l536_536022

open Real

theorem min_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) :
  min ((1 / x) + (1 / y)) (1 / 3) :=
sorry

end min_inv_sum_l536_536022


namespace find_expression_l536_536669

-- Definitions based on the conditions provided
def prop_rel (y x : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 2)

def prop_value_k (k : ℝ) : Prop :=
  k = -4

def prop_value_y (y x : ℝ) : Prop :=
  y = -4 * x + 8

theorem find_expression (y x k : ℝ) : 
  (prop_rel y x k) → 
  (x = 3) → 
  (y = -4) → 
  (prop_value_k k) → 
  (prop_value_y y x) :=
by
  intros h1 h2 h3 h4
  subst h4
  subst h3
  subst h2
  sorry

end find_expression_l536_536669


namespace correct_statements_l536_536520

-- Define the propositions p and q
variables (p q : Prop)

-- Define the given statements as logical conditions
def statement1 := (p ∧ q) → (p ∨ q)
def statement2 := ¬(p ∧ q) → (p ∨ q)
def statement3 := (p ∨ q) ↔ ¬¬p
def statement4 := (¬p) → ¬(p ∧ q)

-- Define the proof problem
theorem correct_statements :
  ((statement1 p q) ∧ (¬statement2 p q) ∧ (statement3 p q) ∧ (¬statement4 p q)) :=
by {
  -- Here you would prove that
  -- statement1 is correct,
  -- statement2 is incorrect,
  -- statement3 is correct,
  -- statement4 is incorrect
  sorry
}

end correct_statements_l536_536520


namespace square_area_adjacency_l536_536054

-- Definition of points as pairs of integers
def Point := ℤ × ℤ

-- Define the points (1,2) and (4,6)
def P1 : Point := (1, 2)
def P2 : Point := (4, 6)

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement for proving the area of a square given the side length
theorem square_area_adjacency (h : distance P1 P2 = 5) : ∃ area : ℝ, area = 25 :=
by
  use 25
  sorry

end square_area_adjacency_l536_536054


namespace ellipse_foci_coordinates_l536_536129

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (frac (x^2) 16 + frac (y^2) 9 = 1) → 
                ((x = sqrt 7 ∧ y = 0) ∨ (x = -sqrt 7 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l536_536129


namespace parabola_intersection_points_l536_536528

theorem parabola_intersection_points :
  (∃ (x y : ℝ), y = 4 * x ^ 2 + 3 * x - 7 ∧ y = 2 * x ^ 2 - 5)
  ↔ ((-2, 3) = (x, y) ∨ (1/2, -4.5) = (x, y)) :=
by
   -- To be proved (proof omitted)
   sorry

end parabola_intersection_points_l536_536528


namespace tangent_line_at_fourth_quadrant_l536_536954

theorem tangent_line_at_fourth_quadrant (k : ℝ) :
  (∃ x y : ℝ, 
     (x - 3)^2 + y^2 = 1 
     ∧ y = k * x 
     ∧ x > 3 
     ∧ y < 0) 
  → k = - (Real.sqrt 2 / 4) :=
begin
  sorry
end

end tangent_line_at_fourth_quadrant_l536_536954


namespace area_ratio_white_to_shaded_l536_536378

theorem area_ratio_white_to_shaded:
  let area_square := λ (s : ℕ), s^2 in
  let areas := list.map area_square [1, 2, 3, 4, 5] in
  let total_area := list.sum areas in
  let white_area := total_area in -- Assuming no overlap for simplicity
  let shaded_area := 0 in -- Assuming no overlap for simplicity
  white_area / shaded_area = (35: ℚ) / (20: ℚ) :=
begin
  -- Definitions
  let area_square := λ (s : ℕ), s^2,
  let areas := list.map area_square [1, 2, 3, 4, 5],
  let total_area := list.sum areas,
  let white_area := total_area, -- total area without overlap
  let shaded_area := 0, -- no overlap assumed
  
  -- Proof outline
  calc white_area / shaded_area = 55 / 0 : by sorry
                       ... = (35: ℚ) / (20: ℚ) : by sorry
end

end area_ratio_white_to_shaded_l536_536378


namespace combination_coins_l536_536813

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536813


namespace coin_combinations_sum_50_l536_536719

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536719


namespace two_digit_factors_of_3_18_minus_1_l536_536834

theorem two_digit_factors_of_3_18_minus_1 : ∃ n : ℕ, n = 6 ∧ 
  ∀ x, x ∈ {y : ℕ | y ∣ 3^18 - 1 ∧ y > 9 ∧ y < 100} → 
  (x = 13 ∨ x = 26 ∨ x = 52 ∨ x = 14 ∨ x = 28 ∨ x = 91) :=
by
  use 6
  sorry

end two_digit_factors_of_3_18_minus_1_l536_536834


namespace find_omega_cos_symmetric_monotonic_l536_536676

theorem find_omega_cos_symmetric_monotonic :
  ∃ ω : ℝ, (ω > 0) ∧ (∀ x : ℝ, cos (ω * x) = 0 ↔ x = (3 * π / 4)) ∧ (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 * π / (3 * ω) → cos (ω * x) > cos (ω * y)) ∧ (ω = 2 / 3) := 
sorry

end find_omega_cos_symmetric_monotonic_l536_536676


namespace nesbitt_inequality_l536_536564

variable (a b c d : ℝ)

-- Assume a, b, c, d are positive real numbers
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom pos_d : 0 < d

theorem nesbitt_inequality :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end nesbitt_inequality_l536_536564


namespace area_of_square_with_adjacent_points_l536_536077

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l536_536077


namespace number_of_integers_in_interval_l536_536919

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536919


namespace h_even_function_l536_536291

noncomputable def f (x : ℝ) : ℝ := x / (2^x - 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
def h (x : ℝ) := f(x) + g(x)

theorem h_even_function : ∀ x : ℝ, h(x) = h(-x) := by
  sorry

end h_even_function_l536_536291


namespace problem_proof_l536_536408

theorem problem_proof (a b c : ℝ) (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h2 : a / (b - c) + b / (c - a) + c / (a - b) = 0) : 
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
sorry

end problem_proof_l536_536408


namespace count_ways_to_get_50_cents_with_coins_l536_536823

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536823


namespace profit_percentage_l536_536143

theorem profit_percentage (initial_cost_per_pound : ℝ) (ruined_percent : ℝ) (selling_price_per_pound : ℝ) (desired_profit_percent : ℝ) : 
  initial_cost_per_pound = 0.80 ∧ ruined_percent = 0.10 ∧ selling_price_per_pound = 0.96 → desired_profit_percent = 8 := by
  sorry

end profit_percentage_l536_536143


namespace f_geq_3e2x_minus_2e3x_l536_536468

-- Define our problem statement in Lean 4
theorem f_geq_3e2x_minus_2e3x (f : ℝ → ℝ) 
    (hf_diff : Differentiable ℝ f)
    (hf_diff2 : Differentiable ℝ (f'))
    (hf0 : f 0 = 1) 
    (hf_prime0 : f' 0 = 0)
    (h_ineq : ∀ x : ℝ, 0 ≤ x → f'' x - 5 * f' x + 6 * f x ≥ 0) :
    ∀ x ≥ 0, f x ≥ 3 * Real.exp (2 * x) - 2 * Real.exp (3 * x) := 
begin
    sorry
end

end f_geq_3e2x_minus_2e3x_l536_536468


namespace exponential_sum_gt_two_l536_536327

def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem exponential_sum_gt_two (a b : ℝ) (h_distinct : a ≠ b) (h_eq : a * Real.exp b - b * Real.exp a = Real.exp a - Real.exp b) : 
  Real.exp a + Real.exp b > 2 :=
by
  sorry

end exponential_sum_gt_two_l536_536327


namespace loss_percentage_grinder_l536_536396

-- Conditions
def CP_grinder := 15000
def CP_mobile := 10000
def profit_mobile_percentage := 0.10
def overall_profit := 400

-- Intermediate calculations based on conditions
def SP_mobile := CP_mobile * (1 + profit_mobile_percentage)
def TCP := CP_grinder + CP_mobile
def TSP := TCP + overall_profit
def SP_grinder := TSP - SP_mobile
def loss_on_grinder := CP_grinder - SP_grinder
def loss_percentage_on_grinder := (loss_on_grinder / CP_grinder) * 100

-- Assertion: The loss percentage on the grinder is 4%
theorem loss_percentage_grinder : loss_percentage_on_grinder = 4 := by
  sorry

end loss_percentage_grinder_l536_536396


namespace angle_BAP_eq_angle_MAC_l536_536705

-- Given conditions
variables {A B C D E F M P : Type} [geometry]  -- Define the variables and assume geometry context
variable [triangle ABC : Type]  -- Given triangle ABC
variable [point_on_lines D BC]  -- Point D on BC
variable [circumcircle_abd E AC]  -- E on circumcircle of triangle ABD and AC meets this circle again at E
variable [circumcircle_acd F AB]  -- F on circumcircle of triangle ACD and AB meets this circle again at F
variable [midpoint M EF]  -- M is the midpoint of EF
variable [circumcircle_def_intersect P BC]  -- P is the intersection of BC with the circumcircle of triangle DEF

-- Theorem to prove
theorem angle_BAP_eq_angle_MAC : ∠BAP = ∠MAC := by
  sorry

end angle_BAP_eq_angle_MAC_l536_536705


namespace probability_base_sqrt10_l536_536335

def M : Set ℕ := {x | x ≤ 3}

noncomputable def f : ℕ → ℕ := sorry

def isosceles_triangle (f : ℕ → ℕ) : Prop :=
  f 0 = f 2 ∧ f 0 ≠ f 1

def valid_configuration_1 (f : ℕ → ℕ) : Prop :=
  f 0 = 0 ∧ f 2 = 0 ∧ f 1 = 3

def valid_configuration_2 (f : ℕ → ℕ) : Prop :=
  f 0 = 3 ∧ f 2 = 3 ∧ f 1 = 0

def base_sqrt10 (f : ℕ → ℕ) : Prop := valid_configuration_1 f ∨ valid_configuration_2 f

theorem probability_base_sqrt10 :
  ∃ (f : ℕ → ℕ), isosceles_triangle f →
  (density base_sqrt10 isosceles_triangle) = (1 / 6 : ℚ) :=
sorry

end probability_base_sqrt10_l536_536335


namespace volume_between_concentric_spheres_l536_536521

theorem volume_between_concentric_spheres (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 8) : 
  let V_small := (4 / 3) * π * r_small^3 in
  let V_large := (4 / 3) * π * r_large^3 in
  V_large - V_small = (1792 / 3) * π :=
by
  subst h_small,
  subst h_large,
  let V_small := (4 / 3) * π * 4^3,
  let V_large := (4 / 3) * π * 8^3,
  have h_V_small : V_small = (256 / 3) * π := by sorry,
  have h_V_large : V_large = (2048 / 3) * π := by sorry,
  calc
    V_large - V_small
        = (2048 / 3) * π - (256 / 3) * π : by rw [h_V_large, h_V_small]
    ... = (1792 / 3) * π : by norm_num

end volume_between_concentric_spheres_l536_536521


namespace total_notebooks_distributed_l536_536549

theorem total_notebooks_distributed (S : ℕ) (h1 : ∀ s, s = S → s / 2 > 0 → s / 8 = 16) : 
(total_notebooks : ℕ := S * (S / 8)) = 2048 :=
by
  sorry

end total_notebooks_distributed_l536_536549


namespace flower_distribution_l536_536517

theorem flower_distribution (students : Fin 12) (bouquets : ℕ) (initial_holding : students → ℕ) :
  bouquets = 13 →
  (∃ n, ∀ i : Fin 12, n = ∑ j : Fin 12, initial_holding j) →
  (∀ i : Fin 12, initial_holding i ≥ 2 → initial_holding (i.pred 1) + initial_holding (i.succ 1) ≥ 1) →
  (∃ i : Fin 12, initial_holding i ≥ 1) →
  ∃ k, 7 ≤ ∑ i : Fin 12, ite (initial_holding i ≥ 1) 1 0 :=
begin
  sorry  -- The proof is omitted as per the instruction
end

end flower_distribution_l536_536517


namespace find_p_l536_536957

theorem find_p 
  (h : {x | x^2 - 5 * x + p ≥ 0} = {x | x ≤ -1 ∨ x ≥ 6}) : p = -6 :=
by
  sorry

end find_p_l536_536957


namespace combinations_of_coins_with_50_cents_l536_536752

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536752


namespace coin_combinations_count_l536_536789

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536789


namespace count_whole_numbers_in_interval_l536_536870

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536870


namespace correct_conclusions_l536_536302

variable {a : ℕ → ℤ}
variable (S : ℕ → ℤ)
variable (d : ℤ)

-- Conditions given in the problem
hypothesis arithmetic_seq : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2
hypothesis S8_gt_S9 : S 8 > S 9
hypothesis S9_gt_S7 : S 9 > S 7

-- The statement to be proven
theorem correct_conclusions (S : ℕ → ℤ) (d : ℤ)
  (arithmetic_seq : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2)
  (S8_gt_S9 : S 8 > S 9)
  (S9_gt_S7 : S 9 > S 7) :
  S 15 > 0 ∧ S 10 < S 7 := by
  sorry

end correct_conclusions_l536_536302


namespace natalie_bushes_to_zucchinis_l536_536630

theorem natalie_bushes_to_zucchinis :
  ∃ (b : ℕ), 
    let containers_per_bush := 10 in
    let zucchinis_per_container := 3 / 4 in
    b * (containers_per_bush * zucchinis_per_container) ≥ 72 ∧ b = 10 :=
by
  sorry

end natalie_bushes_to_zucchinis_l536_536630


namespace count_whole_numbers_in_interval_l536_536868

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536868


namespace average_questions_answered_l536_536498

theorem average_questions_answered :
  let n := 6
  let questions := [16, 22, 30, 26, 18, 20]
  (∑ k in questions, k : ℝ) / n = 22 := 
by 
  sorry

end average_questions_answered_l536_536498


namespace solve_for_nabla_l536_536930

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 :=
by
  sorry

end solve_for_nabla_l536_536930


namespace count_ways_to_get_50_cents_with_coins_l536_536824

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536824


namespace problem_solution_l536_536624

theorem problem_solution :
  ∃ l : List ℕ, l = [15, 70, 125, 180, 235] ∧
                (∀ n ∈ l, n % 11 = 4) ∧
                (∀ n ∈ l, n % 5 = 0) :=
by
  existsi [15, 70, 125, 180, 235]
  split
  { refl }
  split
  { intros n hn
    cases hn with | inl hn75 | inr hn' | { exact_mod_cast hn } ; rw [← hn]
    repeat { simp }
  }
  { intros n hn
    cases hn with | inl hn75 | inr hn' | { exact_mod_cast hn } ; rw [← hn]
    repeat { simp }
  }

end problem_solution_l536_536624


namespace whole_numbers_in_interval_7_4_3pi_l536_536863

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536863


namespace area_of_square_with_adjacent_points_l536_536083

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def side_length := distance 1 2 4 6
def area_of_square (side : ℝ) : ℝ := side ^ 2

theorem area_of_square_with_adjacent_points :
  area_of_square side_length = 25 :=
by
  unfold side_length
  unfold area_of_square
  sorry

end area_of_square_with_adjacent_points_l536_536083


namespace count_whole_numbers_in_interval_l536_536873

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536873


namespace problem_inequality_l536_536688

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 8 * x + 4

theorem problem_inequality :
  (∀ x : ℝ, -2 < x ∧ x < 4 → f' x < 0) ∧
  (∀ x : ℝ, (x < -2 ∨ x > 4) → f' x > 0) ∧
  (∀ x : ℝ, f(-1) = 32/3 ∧ f(4) = -68/3 ∧ f(5) = -58/3) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 → f x ≤ 32/3) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 → f x ≥ -68/3) :=
by
  let f' := λ x : ℝ, x^2 - 2 * x - 8
  sorry

end problem_inequality_l536_536688


namespace solve_for_nabla_l536_536933

theorem solve_for_nabla : ∃ (∇ : ℤ), 3 * (-2) = ∇ + 2 ∧ ∇ = -8 :=
by { existsi (-8), split, exact rfl, exact rfl }

end solve_for_nabla_l536_536933


namespace knights_double_liars_l536_536434

-- Define the types knights and liars
inductive Person
| knight : Person
| liar : Person

-- Define properties of telling the truth or lying
def tells_truth (p : Person) : Prop :=
  p = Person.knight

def lies (p : Person) : Prop :=
  p = Person.liar

-- Define the main problem statement
theorem knights_double_liars
  (people : List Person)
  (h1 : 2 < people.length)
  (h2 : ∃ k, ∃ l, k ≠ l ∧ l = Person.liar ∧ k = Person.knight)
  (h3 : ∀ i, tells_truth (List.nthLe people i (by linarith)) ↔ 
             (((tells_truth (List.nthLe people ((i + 1) % people.length) (by linarith))) ∧
               (lies (List.nthLe people ((i - 1 + people.length) % people.length) (by linarith)))) ∨
              ((lies (List.nthLe people ((i + 1) % people.length) (by linarith))) ∧
               (tells_truth (List.nthLe people ((i - 1 + people.length) % people.length) (by linarith)))))) :
  (people.count (λ p, tells_truth p)) = 2 * (people.count (λ p, lies p)) :=
sorry

end knights_double_liars_l536_536434


namespace circle_line_tangent_l536_536381

-- Definitions based on conditions (equivalent transformation)
def circle_polar_eq (ρ θ : ℝ) (a : ℝ) : Prop := ρ = 2 * a * cos θ
def circle_rect_eq (x y a : ℝ) : Prop := x^2 + y^2 - 2 * a * x = 0

-- Parametric equation of line l
def line_param_eq (x y t : ℝ) : Prop := x = 3 * t + 1 ∧ y = 4 * t + 3

-- Conversion to polar coordinates (in terms of polar ρ and θ of line l)
def line_polar_eq (x y : ℝ) : Prop := ∃ t, x = 3 * t + 1 ∧ y = 4 * t + 3

-- Tangency condition
def tangency_condition (a : ℝ) : Prop := |(4 * a + 5) / 5| = |a|

-- Main theorem statement
theorem circle_line_tangent (a : ℝ) (h : a < 1) :
  ∀ x y t ρ θ,
  circle_polar_eq ρ θ a ↔ (circle_rect_eq x y a) ∧
  line_param_eq x y t ↔ line_polar_eq x y ∧
  tangency_condition a → 
  ∃ a_value, a = a_value := 
sorry

end circle_line_tangent_l536_536381


namespace sector_area_eq_25_l536_536973

theorem sector_area_eq_25 (r θ : ℝ) (h_r : r = 5) (h_θ : θ = 2) : (1 / 2) * θ * r^2 = 25 := by
  sorry

end sector_area_eq_25_l536_536973


namespace general_formula_seq_arithmetic_l536_536424

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Conditions from the problem
axiom sum_condition (n : ℕ) : (1 - q) * S n + q^n = 1
axiom nonzero_q : q * (q - 1) ≠ 0
axiom arithmetic_S : S 3 + S 9 = 2 * S 6

-- Stating the proof goals
theorem general_formula (n : ℕ) : a n = q^(n-1) :=
sorry

theorem seq_arithmetic : a 2 + a 8 = 2 * a 5 :=
sorry

end general_formula_seq_arithmetic_l536_536424


namespace minimum_t_no_geometric_sequence_l536_536679

noncomputable def a (n : ℕ) : ℝ := (1/2) ^ n
noncomputable def s (n : ℕ) : ℝ := 1 - (1/2) ^ n
noncomputable def b (n : ℕ) (t : ℕ) : ℝ := t - 5 * Real.logb 2 (1 - s n)
noncomputable def c (n : ℕ) (t : ℕ) : ℝ := a n * b n t

theorem minimum_t (t : ℕ) : (∀ n : ℕ, c (n + 1) t < c n t) → t ≥ 1 := sorry

theorem no_geometric_sequence (t : ℕ) : 
  ¬ (∃ k : ℕ, k > 0 ∧
   (∃ x₁ x₂ x₃ : ℕ, {k, k + 1, k + 2} = {x₁, x₂, x₃} ∧
    {c x₁ t, c x₂ t, c x₃ t}.to_finset.card = 3 ∧
    (∃ r : ℝ, r > 0 ∧
      (c x₂ t / c x₁ t = r ∧ c x₃ t / c x₂ t = r))) := sorry

end minimum_t_no_geometric_sequence_l536_536679


namespace min_value_PA_PB_dot_l536_536296

noncomputable def min_dot_product (t : ℝ) : ℝ :=
let x := t^2 - 2 * t + 4 in
((2 * x^2 + x) / (x + 1))

theorem min_value_PA_PB_dot (min_val : ℝ) : 
    (∃ P : ℝ × ℝ, P = (-1, 1)) →
    (∀ t : ℝ, Nonempty (C : ℝ × ℝ → ℝ, (λ (x y : ℝ), (x - t)^2 + (y - t + 2)^2 - 1 = 0) (P fst) (P snd))) →
    (∃ A B : ℝ × ℝ,
        (tangency_points : C = tangent_lines (P) (circle_family := C)) ∧
        P * A = min_dot_product) →
    min_val = 21 / 4 :=
by
  sorry

end min_value_PA_PB_dot_l536_536296


namespace max_removed_rooks_l536_536604

-- Define the chessboard as an 8x8 grid
def Chessboard := Fin 8 × Fin 8

-- Define a function to check if two rooks attack each other
def attacks (r1 r2 : Chessboard) : Prop :=
  r1.1 = r2.1 ∨ r1.2 = r2.2

-- Define a function to compute the number of rooks a given rook attacks
def numAttacks (remaining : Set Chessboard) (r : Chessboard) : Nat :=
  (remaining.filter (attacks r)).size

-- Define a condition stating a rook can be removed if it attacks an odd number of other rooks
def canBeRemoved (remaining : Set Chessboard) (r : Chessboard) : Prop :=
  numAttacks remaining r % 2 = 1

-- Define the initial set of rooks on the chessboard
def initialRooks : Set Chessboard :=
  { p | true }

-- Prove the maximum number of rooks that can be removed is 59
theorem max_removed_rooks : ∃ remaining : Set Chessboard,
  remaining ⊆ initialRooks ∧
  initialRooks.size - remaining.size = 59 ∧
  ∀ r ∈ initialRooks \ remaining, canBeRemoved remaining r := sorry

end max_removed_rooks_l536_536604


namespace infinite_primes_solution_l536_536440

theorem infinite_primes_solution :
  ∃^∞ p : ℕ, prime p ∧ ∃ x y : ℤ, x^2 + x + 1 = p :=
sorry

end infinite_primes_solution_l536_536440


namespace area_of_square_l536_536057

-- We define the points as given in the conditions
def point1 : ℝ × ℝ := (1, 2)
def point2 : ℝ × ℝ := (4, 6)

-- Lean's "def" defines the concept of a square given two adjacent points.
def is_square (p1 p2: ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  in ∃ (l : ℝ), l = d ∧ (l^2 = 25)

-- The theorem assumes the points are adjacent points on a square and proves that their area is 25.
theorem area_of_square :
  is_square point1 point2 :=
by
  -- Insert formal proof here, skipped with 'sorry' for this task
  sorry

end area_of_square_l536_536057


namespace value_at_17pi_over_6_l536_536471

variable (f : Real → Real)

-- Defining the conditions
def period (f : Real → Real) (T : Real) := ∀ x, f (x + T) = f x
def specific_value (f : Real → Real) (x : Real) (v : Real) := f x = v

-- The main theorem statement
theorem value_at_17pi_over_6 : 
  period f (π / 2) →
  specific_value f (π / 3) 1 →
  specific_value f (17 * π / 6) 1 :=
by
  intros h_period h_value
  sorry

end value_at_17pi_over_6_l536_536471


namespace alice_winning_strategy_l536_536411

-- Define the game conditions
variables (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2)

-- Define winning strategies
def alice_has_winning_strategy := ∀ turns, alice_strategy turns = win
def bob_has_winning_strategy := ∀ turns, bob_strategy turns = win

-- Main theorem statement
theorem alice_winning_strategy : ∃ strategy, alice_has_winning_strategy :=
sorry

end alice_winning_strategy_l536_536411


namespace initial_population_l536_536974

theorem initial_population (P : ℝ) 
  (increase1 : 1 + 5 / 100 = 1.05)
  (decrease2 : 1 - 7 / 100 = 0.93)
  (increase3 : 1 + 3 / 100 = 1.03)
  (increase4 : 1 + 10 / 100 = 1.10)
  (decrease5 : 1 - 5 / 100 = 0.95) :
  P * 1.05 * 0.93 * 1.03 * 1.10 * 0.95 = 10,450 ->
  P = 10,450 / (1.05 * 0.93 * 1.03 * 1.10 * 0.95) :=
by 
  sorry

end initial_population_l536_536974


namespace triangles_with_positive_area_l536_536349

theorem triangles_with_positive_area :
  let points := {p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5} in
  ∃ (n : ℕ), n = 2150 ∧ 
    (∃ (triangles : set (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)), 
      (∀ t ∈ triangles, 
        t.1 ∈ points ∧ t.2 ∈ points ∧ t.3 ∈ points ∧ 
        (∃ (area : ℚ), area > 0 ∧ 
          area = ((t.2.1 - t.1.1) * (t.3.2 - t.1.2) - (t.3.1 - t.1.1) * (t.2.2 - t.1.2)) / 2)) ∧ 
      ∃ (card_tris : ℕ), card_tris = n) :=
sorry

end triangles_with_positive_area_l536_536349


namespace solve_for_x_l536_536460

theorem solve_for_x (x : ℚ) (h : 3 / 4 - 1 / x = 1 / 2) : x = 4 :=
sorry

end solve_for_x_l536_536460


namespace range_of_a_l536_536691

theorem range_of_a (a : ℝ) (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) (M N : ℝ × ℝ)
  (h_line : ∀ x y, line x y ↔ ax - y + 3 = 0)
  (h_circle : ∀ x y, circle x y ↔ (x - 2)^2 + (y - a)^2 = 4)
  (h_MN : (circle M.1 M.2 ∧ circle N.1 N.2) ∧ (line M.1 M.2 ∧ line N.1 N.2))
  (h_dist : |dist M N| ≥ 2 * real.sqrt 3) :
  a ≤ -4 / 3 :=
by
  sorry

end range_of_a_l536_536691


namespace count_whole_numbers_in_interval_l536_536838

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536838


namespace number_of_ways_to_pick_three_cards_l536_536212

theorem number_of_ways_to_pick_three_cards :
  (∃ n : ℕ, n = 60) →
  (∃ k : ℕ, k = 3) →
  (∃ m : ℕ, m = 5) →
  (∃ s : ℕ, s = 12) →
  (∃ o : ℕ, o = 205320) → 
  (finset.card (finset.pi ((finset.range 60).finset_powerset_len 3) 
  ((λ _ => finset.range 60).map finset.pi)) = 205320) :=
by
  intros h1 h2 h3 h4 h5 
  sorry

end number_of_ways_to_pick_three_cards_l536_536212


namespace combination_coins_l536_536821

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536821


namespace area_of_square_with_adjacent_points_l536_536095

theorem area_of_square_with_adjacent_points (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 6)) :
  let side_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in 
  let area := side_length^2 in 
  area = 25 :=
by
  sorry

end area_of_square_with_adjacent_points_l536_536095


namespace angle_ABC_degrees_l536_536035

noncomputable def length (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

def A : ℝ × ℝ × ℝ := (1, 2, 3)
def B : ℝ × ℝ × ℝ := (4, 6, 3)
def C : ℝ × ℝ × ℝ := (4, 2, 6)

noncomputable def cosAngleABC : ℝ :=
  let AB := length A B
  let BC := length B C
  let AC := length A C
  (AB^2 + BC^2 - AC^2) / (2 * AB * BC)

noncomputable def angleABC_deg : ℝ := real.acos cosAngleABC * (180 / real.pi)

theorem angle_ABC_degrees : angleABC_deg ≈ 39.23 :=
  sorry

end angle_ABC_degrees_l536_536035


namespace bracket_mul_l536_536202

def bracket (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

theorem bracket_mul : bracket 6 * bracket 3 = 28 := by
  sorry

end bracket_mul_l536_536202


namespace area_of_square_with_adjacent_points_l536_536089

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def side_length := distance 1 2 4 6
def area_of_square (side : ℝ) : ℝ := side ^ 2

theorem area_of_square_with_adjacent_points :
  area_of_square side_length = 25 :=
by
  unfold side_length
  unfold area_of_square
  sorry

end area_of_square_with_adjacent_points_l536_536089


namespace range_of_a_l536_536685

-- Define the function
def f (x : ℝ) (a : ℝ) : ℝ := -real.log x + a * x^2 + (1 - 2 * a) * x + a - 1

-- Define the problem and required theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, 0 < x ∧ x < 1 → f x a > 0) ↔ a ≥ -1/2 := 
sorry

end range_of_a_l536_536685


namespace count_whole_numbers_in_interval_l536_536855

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536855


namespace combinations_of_coins_with_50_cents_l536_536751

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536751


namespace common_ratio_of_geometric_progression_l536_536484

-- Define the problem conditions
variables {a b c q : ℝ}

-- The sequence a, b, c is a geometric progression
def geometric_progression (a b c : ℝ) (q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- The sequence 577a, (2020b/7), (c/7) is an arithmetic progression
def arithmetic_progression (x y z : ℝ) : Prop :=
  2 * y = x + z

-- Main theorem statement to prove
theorem common_ratio_of_geometric_progression (h1 : geometric_progression a b c q) 
  (h2 : arithmetic_progression (577 * a) (2020 * b / 7) (c / 7)) 
  (h3 : b < a ∧ c < b) : q = 4039 :=
sorry

end common_ratio_of_geometric_progression_l536_536484


namespace trig_identity_l536_536419

theorem trig_identity
  (α β γ τ : ℝ)
  (hpos_α : 0 < α)
  (hpos_β : 0 < β)
  (hpos_γ : 0 < γ)
  (hpos_τ : 0 < τ)
  (heq : ∀ x : ℝ, sin (α * x) + sin (β * x) = sin (γ * x) + sin (τ * x)) :
  α = γ ∨ α = τ := 
sorry

end trig_identity_l536_536419


namespace count_whole_numbers_in_interval_l536_536904

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536904


namespace coin_combinations_l536_536808

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536808


namespace number_of_integers_in_interval_l536_536925

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536925


namespace combination_coins_l536_536820

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536820


namespace whole_numbers_in_interval_7_4_3pi_l536_536861

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536861


namespace triangle_relationship_l536_536619

variable (A B C D : Type) [Triangle A B C]
variable (AB AC BC : ℝ) (DA DB DC : ℝ)
variable (x y : ℝ)
variable (p s : ℝ)

-- Given conditions
axiom cond1 : p = AB + BC + AC
axiom cond2 : s = DA + DB + DC
axiom cond3 : x = y
axiom cond4 : BC = x + y

-- To prove
theorem triangle_relationship (cond1 : p = AB + BC + AC)
                             (cond2 : s = DA + DB + DC)
                             (cond3 : x = y)
                             (cond4 : BC = x + y) :
  s >= p / 2 := 
sorry

end triangle_relationship_l536_536619


namespace f_neg_expr_l536_536315

-- Define that f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Define the specific form of f for x >= 0
def f_nonneg (f : ℝ → ℝ) := ∀ x : ℝ, x ≥ 0 → f x = 2 * x ^ 2 - 4 * x

-- The theorem we want to prove
theorem f_neg_expr (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f_nonneg f) :
  ∀ x : ℝ, x < 0 → f x = - (2 * x ^ 2 + 4 * x) :=
begin
  sorry
end

end f_neg_expr_l536_536315


namespace game_completion_days_l536_536999

theorem game_completion_days (initial_playtime hours_per_day : ℕ) (initial_days : ℕ) (completion_percentage : ℚ) (increased_playtime : ℕ) (remaining_days : ℕ) :
  initial_playtime = 4 →
  hours_per_day = 2 * 7 →
  completion_percentage = 0.4 →
  increased_playtime = 7 →
  ((initial_playtime * hours_per_day) / completion_percentage) - (initial_playtime * hours_per_day) = increased_playtime * remaining_days →
  remaining_days = 12 :=
by
  intros
  sorry

end game_completion_days_l536_536999


namespace number_of_factors_is_16_l536_536157

-- Definitions of the characters as digits (between 0 and 9)
variables (好 妙 真 题 : ℕ)
variables (h0 : 0 ≤ 好 ∧ 好 ≤ 9)
variables (h1 : 0 ≤ 妙 ∧ 妙 ≤ 9)
variables (h2 : 0 ≤ 真 ∧ 真 ≤ 9)
variables (h3 : 0 ≤ 题 ∧ 题 ≤ 9)

-- Distinct digits condition
variables (h_distinct : 好 ≠ 妙 ∧ 好 ≠ 真 ∧ 好 ≠ 题 ∧ 妙 ≠ 真 ∧ 妙 ≠ 题 ∧ 真 ≠ 题)

-- The given conditions
variables (h_add : 好 + 好 = 妙)
variables (h_mult : 妙 * (10 * 好 + 好) * (10 * 真 + 好) = 妙 * 1000 + 题 * 100 + 题 * 10 + 妙)

-- The number of factors of the four-digit number 妙题题妙 is 16
theorem number_of_factors_is_16 : nat.factors (妙 * 1000 + 题 * 100 + 题 * 10 + 妙).length = 16 :=
by sorry

end number_of_factors_is_16_l536_536157


namespace coin_combinations_50_cents_l536_536728

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536728


namespace problem1_l536_536562

def f (x : ℝ) : ℝ :=
  if x > 0 then real.log x / real.log 2 else 2 ^ x

theorem problem1 : f (f (1 / 8)) = 1 / 8 :=
  by
    sorry

end problem1_l536_536562


namespace count_pairs_satisfying_l536_536344

theorem count_pairs_satisfying (N : ℕ) (hN : N = 50) : 
  (∑ m in finset.range 8, if m * m < N then N - m * m - 1 else 0) = 203 := 
by
  -- This theorem states that for N = 50, the count of pairs (m, n) satisfying m^2 + n < 50 is 203.
  -- Each condition translates the ranges of m from 1 to 7 and ensures the correct totaling of counts for n.
  sorry

end count_pairs_satisfying_l536_536344


namespace cab_driver_income_fifth_day_l536_536570

def income_for_first_four_days : list ℕ := [400, 250, 650, 400]
def average_income_for_five_days : ℕ := 440

theorem cab_driver_income_fifth_day (total_days : ℕ) 
    (income : fin total_days → ℕ) 
    (avg_income : ℕ) 
    (h : total_days = 5) 
    (h₁ : income 0 = 400) 
    (h₂ : income 1 = 250) 
    (h₃ : income 2 = 650) 
    (h₄ : income 3 = 400) 
    (h_avg : avg_income = 440) : 
  income 4 = 500 := 
by {
  sorry
}

end cab_driver_income_fifth_day_l536_536570


namespace opposite_of_9_is_neg_9_l536_536147

-- Definition of opposite number according to the given condition
def opposite (n : Int) : Int := -n

-- Proof statement that the opposite of 9 is -9
theorem opposite_of_9_is_neg_9 : opposite 9 = -9 :=
by
  sorry

end opposite_of_9_is_neg_9_l536_536147


namespace jessica_final_balance_l536_536550

variable (B : ℝ) (withdrawal : ℝ) (deposit : ℝ)

-- Conditions
def condition1 : Prop := withdrawal = (2 / 5) * B
def condition2 : Prop := deposit = (1 / 5) * (B - withdrawal)

-- Proof goal statement
theorem jessica_final_balance (h1 : condition1 B withdrawal)
                             (h2 : condition2 B withdrawal deposit) :
    (B - withdrawal + deposit) = 360 :=
by
  sorry

end jessica_final_balance_l536_536550


namespace max_red_dragons_l536_536377

variables (dragon : Type) [inhabited dragon]
variables (Color : Type) [inhabited Color] (Head : Type) [inhabited Head]

variables (is_red : dragon → Prop)
variables (is_green : dragon → Prop)
variables (is_blue : dragon → Prop)
variables (tells_truth : Head → Prop)
variables (has_truthful_head : dragon → Prop)
variables (left_of : dragon → dragon)
variables (right_of : dragon → dragon)
variables (head_statements : dragon → Head → Prop)

-- Definitions for the statements
def first_head_statement (d : dragon) : Prop := is_green (left_of d)
def second_head_statement (d : dragon) : Prop := is_blue (right_of d)
def third_head_statement (d : dragon) : Prop := ¬ ∃ x, (is_red x) ∧ (left_of d = x ∨ right_of d = x)

-- Condition for a dragon
def consistent_dragon (d : dragon) : Prop :=
(∃ h, tells_truth h ∧ head_statements d h) ∧ 
(∀ h, tells_truth h → 
  (head_statements d h = first_head_statement d ∨ 
   head_statements d h = second_head_statement d ∨ 
   head_statements d h = third_head_statement d))

theorem max_red_dragons (total_dragons : ℕ) (h_round_table : total_dragons = 530) :
  ∃ max_red : ℕ, max_red = 176 := 
by
  sorry

end max_red_dragons_l536_536377


namespace find_students_with_equal_homework_hours_l536_536497

theorem find_students_with_equal_homework_hours :
  let Dan := 6
  let Joe := 3
  let Bob := 5
  let Susie := 4
  let Grace := 1
  (Joe + Grace = Dan ∨ Joe + Bob = Dan ∨ Bob + Grace = Dan ∨ Dan + Bob = Dan ∨ Susie + Grace = Dan) → 
  (Bob + Grace = Dan) := 
by 
  intros
  sorry

end find_students_with_equal_homework_hours_l536_536497


namespace count_whole_numbers_in_interval_l536_536882

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536882


namespace coin_combinations_count_l536_536797

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536797


namespace combinations_of_coins_l536_536759

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536759


namespace avg_children_in_families_with_children_l536_536171

theorem avg_children_in_families_with_children
  (total_families : ℕ)
  (avg_children_per_family : ℕ)
  (childless_families : ℕ)
  (total_children : ℕ := total_families * avg_children_per_family)
  (families_with_children : ℕ := total_families - childless_families)
  (avg_children_in_families_with_children : ℚ := total_children / families_with_children) :
  avg_children_in_families_with_children = 4 :=
by
  have h1 : total_families = 12 := sorry
  have h2 : avg_children_per_family = 3 := sorry
  have h3 : childless_families = 3 := sorry
  have h4 : total_children = 12 * 3 := sorry
  have h5 : families_with_children = 12 - 3 := sorry
  have h6 : avg_children_in_families_with_children = (12 * 3) / (12 - 3) := sorry
  have h7 : ((12 * 3) / (12 - 3) : ℚ) = 4 := sorry
  exact h7

end avg_children_in_families_with_children_l536_536171


namespace solve_for_x_l536_536113

theorem solve_for_x : 
  (35 / (6 - (2 / 5)) = 25 / 4) := 
by
  sorry 

end solve_for_x_l536_536113


namespace area_relationship_l536_536223

theorem area_relationship (P Q R : ℝ) (h_square : 10 * 10 = 100)
  (h_triangle1 : P + R = 50)
  (h_triangle2 : Q + R = 50) :
  P - Q = 0 :=
by
  sorry

end area_relationship_l536_536223


namespace consecutive_page_numbers_sum_l536_536500

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) = 19881) : n + (n + 1) = 283 :=
sorry

end consecutive_page_numbers_sum_l536_536500


namespace jordan_travel_distance_heavy_traffic_l536_536397

theorem jordan_travel_distance_heavy_traffic (x : ℝ) (h1 : x / 20 + x / 10 + x / 6 = 7 / 6) : 
  x = 3.7 :=
by
  sorry

end jordan_travel_distance_heavy_traffic_l536_536397


namespace sum_of_first_3n_terms_l536_536514

-- Define the sums of the geometric sequence
variable (S_n S_2n S_3n : ℕ)

-- Given conditions
variable (h1 : S_n = 48)
variable (h2 : S_2n = 60)

-- The statement we need to prove
theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 := by
  sorry

end sum_of_first_3n_terms_l536_536514


namespace floor_sqrt_x_eq_8_has_17_values_l536_536935

theorem floor_sqrt_x_eq_8_has_17_values :
  {x : ℕ | 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9}.finite
  ∧ fintype.card {x : ℕ | 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9} = 17 :=
by {
  sorry,
}

end floor_sqrt_x_eq_8_has_17_values_l536_536935


namespace coin_combinations_50_cents_l536_536730

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536730


namespace smallest_integer_inverse_modulo_1260_l536_536183

theorem smallest_integer_inverse_modulo_1260 :
  ∃ (n : ℕ), n > 1 ∧ (gcd n 1260 = 1) ∧
  ∀ (m : ℕ), m > 1 ∧ (gcd m 1260 = 1) → n ≤ m :=
begin
  use 13,
  split,
  { linarith, },
  split,
  { norm_num, },
  { intros m hm hm',
    revert m hm,
    exact sorry, },
end

end smallest_integer_inverse_modulo_1260_l536_536183


namespace part_I_part_II_part_III_part_III_range_l536_536689

-- Define the function f(x)
def f (m x : ℝ) : ℝ := m * real.log x - x^2 + 2

-- Problem statements in Lean
theorem part_I (x : ℝ) (h : 0 < x) : 
0 < x ∧ x < real.sqrt 2 / 2 → deriv (f 1) x > 0 ∧ x > real.sqrt 2 / 2 → deriv (f 1) x < 0 := sorry

theorem part_II (x : ℝ) (h : x > 0) :
f 2 x - deriv (f 2) x ≤ 4 * x - 3 := sorry

theorem part_III (m x : ℝ) (hm : 2 ≤ m ∧ m ≤ 8) (hx : 1 ≤ x) :
f m x - deriv (f m) x ≤ 4 * x - 3 := sorry

-- To find range of m such that the inequality holds
theorem part_III_range :
∀ (m : ℝ), 2 ≤ m ∧ m ≤ 8 → ∀ (x : ℝ), x ≥ 1 → f m x - deriv (f m) x ≤ 4 * x - 3 := sorry

end part_I_part_II_part_III_part_III_range_l536_536689


namespace general_term_a_sum_terms_b_l536_536680

-- Define the sequences and constants
def S (n : ℕ) : ℝ := sorry
def a (n : ℕ) : ℝ := if n = 0 then 1 else 3^(n-1)
def b (n : ℕ) : ℝ := 3 / (n^2 + n)

-- Problem 1: Prove the general term of the sequence {a_n}
theorem general_term_a (n : ℕ) (hn : n > 0) : 
  2 * S n = 3 * a n - 1 → a n = 3^(n-1) := 
by sorry

-- Problem 2: Prove the sum of the first n terms of the sequence {b_n}
theorem sum_terms_b (n : ℕ) (hn : n > 0) :
  (∀ m, m ≤ n → a m * b m = 3^m / (m^2 + m)) →
  T n = ∑ i in range (n + 1), b i → T n = (3 * n)/(n + 1) :=
by sorry

end general_term_a_sum_terms_b_l536_536680


namespace combinations_of_coins_l536_536763

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536763


namespace range_of_x_for_sqrt_l536_536945

theorem range_of_x_for_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 :=
sorry

end range_of_x_for_sqrt_l536_536945


namespace problem_statement_l536_536012

-- Given conditions 
def g (n : ℕ) : ℝ := Real.logb 3 (3^n)

-- The problem statement to prove
theorem problem_statement (n : ℕ) : 
  (g n) / (Real.log 10 3) = n / (Real.log 10 3) := 
by 
  sorry

end problem_statement_l536_536012


namespace tan_alpha_parallel_vectors_l536_536706

theorem tan_alpha_parallel_vectors
    (α : ℝ)
    (a : ℝ × ℝ := (6, 8))
    (b : ℝ × ℝ := (Real.sin α, Real.cos α))
    (h : a.fst * b.snd = a.snd * b.fst) :
    Real.tan α = 3 / 4 := 
sorry

end tan_alpha_parallel_vectors_l536_536706


namespace coin_combinations_count_l536_536734

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536734


namespace complex_div_second_quadrant_l536_536657

theorem complex_div_second_quadrant (z1 z2 : ℂ) (h1 : z1 = (1 : ℂ) + 2 * complex.I) (h2 : z2 = (3 : ℂ) - 4 * complex.I) :
  let div := z1 / z2 in div.re < 0 ∧ 0 < div.im :=
by
  sorry

end complex_div_second_quadrant_l536_536657


namespace parallel_cd_perpendicular_cd_l536_536939

-- Definitions and given conditions
def a : ℝ × ℝ := (0, 3)
def b : ℝ × ℝ := (Real.sqrt 3, 1)
def c : ℝ × ℝ := (3 * a.1 + 5 * b.1, 3 * a.2 + 5 * b.2)
def d (m : ℝ) : ℝ × ℝ := (m * a.1 - 5 * b.1, m * a.2 - 5 * b.2)

-- Proof goals
theorem parallel_cd (m : ℝ) : (c.1 * d m.2 = c.2 * d m.1) ↔ (m = -3) :=
by sorry

theorem perpendicular_cd (m : ℝ) : (c.1 * d m.1 + c.2 * d m.2 = 0) ↔ (m = 145 / 42) :=
by sorry

end parallel_cd_perpendicular_cd_l536_536939


namespace sin_double_angle_CPD_l536_536439

variable (P : Type) [InnerProductSpace ℝ P]

theorem sin_double_angle_CPD (A B C D E P : P)
  (h1 : dist A B = dist B C) (h2 : dist B C = dist C D) (h3 : dist C D = dist D E)
  (hcos1 : ∀ P, (∠ B P C).cos = 1/3)
  (hcos2 : ∀ P, (∠ C P D).cos = 1/2) : 
  Real.sin (2 * ∠ C P D) = Real.sqrt 3 / 2 := 
  sorry

end sin_double_angle_CPD_l536_536439


namespace parabola_directrix_l536_536132

theorem parabola_directrix (a : ℝ) (h : -1 / (4 * a) = 2) : a = -1 / 8 :=
by
  sorry

end parabola_directrix_l536_536132


namespace square_area_adjacency_l536_536051

-- Definition of points as pairs of integers
def Point := ℤ × ℤ

-- Define the points (1,2) and (4,6)
def P1 : Point := (1, 2)
def P2 : Point := (4, 6)

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement for proving the area of a square given the side length
theorem square_area_adjacency (h : distance P1 P2 = 5) : ∃ area : ℝ, area = 25 :=
by
  use 25
  sorry

end square_area_adjacency_l536_536051


namespace required_range_of_a_l536_536625

variable (a : ℝ) (f : ℝ → ℝ)
def function_increasing_on (f : ℝ → ℝ) (a : ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, DifferentiableAt ℝ f x ∧ (deriv f x) ≥ 0

theorem required_range_of_a (h : function_increasing_on (fun x => a * Real.log x + x) a (Set.Icc 2 3)) :
  a ≥ -2 :=
sorry

end required_range_of_a_l536_536625


namespace count_whole_numbers_in_interval_l536_536849

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536849


namespace BYTS_concyclic_l536_536991

variables {A B C P E F S O T Y : Point}
variables (h1: ∠ A = 60)
variables (h2: lies_on_segment P B C)
variables (h3: lies_on_rays E A B ∧ lies_on_rays F A C)
variables (h4: dist B P = dist B E ∧ dist C P = dist C F)
variables (h5: tangents_intersect_circumcircle S A B C)
variables (h6: circumcenter O E P F)
variables (h7: intersects BO CS T)
variables (h8: intersects SO AB Y)

-- Statement to prove points B, Y, T, and S are concyclic
theorem BYTS_concyclic :
  concyclic B Y T S :=
sorry

end BYTS_concyclic_l536_536991


namespace projection_distance_yOz_l536_536386

noncomputable def point := ℝ × ℝ × ℝ

def A : point := (1, 2, 3)
def O : point := (0, 0, 0)
def projection_on_yOz (p : point) : point := (0, p.2, p.3)

def dist (p1 p2 : point) : ℝ := 
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

theorem projection_distance_yOz :
  dist O (projection_on_yOz A) = real.sqrt 13 := by
  sorry

end projection_distance_yOz_l536_536386


namespace find_next_score_l536_536239

def scores := [95, 85, 75, 65, 90]
def current_avg := (95 + 85 + 75 + 65 + 90) / 5
def target_avg := current_avg + 4

theorem find_next_score (s : ℕ) (h : (95 + 85 + 75 + 65 + 90 + s) / 6 = target_avg) : s = 106 :=
by
  -- Proof steps here
  sorry

end find_next_score_l536_536239


namespace sally_quarters_after_purchases_l536_536445

-- Given conditions
def initial_quarters : ℕ := 760
def first_purchase : ℕ := 418
def second_purchase : ℕ := 192

-- Define the resulting quarters after purchases
def quarters_after_first_purchase (initial : ℕ) (spent : ℕ) : ℕ := initial - spent
def quarters_after_second_purchase (remaining : ℕ) (spent : ℕ) : ℕ := remaining - spent

-- The main statement to be proved
theorem sally_quarters_after_purchases :
  quarters_after_second_purchase (quarters_after_first_purchase initial_quarters first_purchase) second_purchase = 150 :=
by
  unfold quarters_after_first_purchase quarters_after_second_purchase initial_quarters first_purchase second_purchase
  simp
  sorry

end sally_quarters_after_purchases_l536_536445


namespace manager_hourly_wage_l536_536602

open Real

theorem manager_hourly_wage (M D C : ℝ) 
  (hD : D = M / 2)
  (hC : C = 1.20 * D)
  (hC_manager : C = M - 3.40) :
  M = 8.50 :=
by
  sorry

end manager_hourly_wage_l536_536602


namespace coin_combinations_50_cents_l536_536723

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536723


namespace fourth_group_frequency_is_14_l536_536262

/-- 
Define the total number of data pieces.
-/
def total_data_pieces : ℕ := 50

/-- 
Define the number of groups.
-/
def num_groups : ℕ := 5

/-- 
Define the frequency of the first group.
-/
def freq_first_group : ℕ := 6

/-- 
Define the sum of the frequencies of the second and fifth groups.
-/
def sum_freq_second_fifth_groups : ℕ := 20

/-- 
Define the frequency of the third group.
-/
def freq_third_group : ℕ := (0.2 * total_data_pieces).toNat

/-- 
The proposition to prove the frequency of the fourth group is 14.
-/
def freq_fourth_group : ℕ := total_data_pieces - freq_first_group - sum_freq_second_fifth_groups - freq_third_group

theorem fourth_group_frequency_is_14 :
  freq_fourth_group = 14 := by
  sorry

end fourth_group_frequency_is_14_l536_536262


namespace solve_for_x_l536_536487

-- Definitions
variables (α x : ℝ)
def side_length := (1 / 2 : ℝ)
def inclined_angle := (2 * α)
def angle_QPR := (180 - 2 * α : ℝ)
def length_PQ := side_length
def length_PR := side_length
def points_congruent := ∀ (Q N R : ℝ), Q = N ∧ N = R

-- Proof Problem Statement
theorem solve_for_x (cos_alpha : ℝ) 
  (h1 : PQ = PR = side_length) 
  (h2 : angle_QPR = 180 - 2 * α) 
  (h3 : points_congruent Q N R)
  (trig_relation : cos α = x)
  : x = cos α := sorry

end solve_for_x_l536_536487


namespace square_area_adjacency_l536_536053

-- Definition of points as pairs of integers
def Point := ℤ × ℤ

-- Define the points (1,2) and (4,6)
def P1 : Point := (1, 2)
def P2 : Point := (4, 6)

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement for proving the area of a square given the side length
theorem square_area_adjacency (h : distance P1 P2 = 5) : ∃ area : ℝ, area = 25 :=
by
  use 25
  sorry

end square_area_adjacency_l536_536053


namespace multiple_of_anushas_share_l536_536466

theorem multiple_of_anushas_share : 
  ∃ M : ℕ, ∀ (A B E : ℕ), 
    (A = 84) ∧ (M * A = 8 * B) ∧ (8 * B = 6 * E) ∧ (A + B + E = 378) → 
    M = 12 :=
by
  intro A B E
  assume h
  sorry

end multiple_of_anushas_share_l536_536466


namespace successive_discounts_eq_single_discount_l536_536117

theorem successive_discounts_eq_single_discount :
  ∀ (x : ℝ), (1 - 0.15) * (1 - 0.25) * x = (1 - 0.3625) * x :=
by
  intro x
  sorry

end successive_discounts_eq_single_discount_l536_536117


namespace find_m_of_symmetry_l536_536673

-- Define the conditions for the parabola and the axis of symmetry
theorem find_m_of_symmetry (m : ℝ) :
  let a := (1 : ℝ)
  let b := (m - 2 : ℝ)
  let axis_of_symmetry := (0 : ℝ)
  (-b / (2 * a)) = axis_of_symmetry → m = 2 :=
by
  sorry

end find_m_of_symmetry_l536_536673


namespace problem_1_problem_2_problem_3_l536_536133

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for function definition

axiom domain_f : ∀ x, x > 0 → f x ≠ 0
axiom functional_eq : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y
axiom pos_cond : ∀ x, x > 1 → f x > 0

theorem problem_1 : f 1 = 0 := sorry

theorem problem_2 : ∀ x y, 0 < x → 0 < y → x < y → f x < f y := sorry

theorem problem_3 : f 6 = 1 → ∀ x, f (x + 3) - f (1 / 3) < 2 ↔ x ∈ Ioo (-3 : ℝ) 9 := sorry

end problem_1_problem_2_problem_3_l536_536133


namespace coin_combinations_count_l536_536798

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536798


namespace income_before_taxes_correct_l536_536233

-- Given conditions as definitions in Lean
def tax_5_percent (x : ℝ) : ℝ := x * 0.05
def tax_10_percent (x : ℝ) : ℝ := x * 0.10
def tax_15_percent (x : ℝ) : ℝ := x * 0.15

-- Define the income brackets and net income condition
def net_income_after_taxes (I : ℝ) : ℝ :=
  let income_over_10000 := max (0) (I - 10000)
  let income_6000_to_10000 := max (0) (min (I - 6000) 4000)
  let income_3000_to_6000 := max (0) (min (I - 3000) 3000)
  I - (tax_15_percent(income_over_10000) + tax_10_percent(income_6000_to_10000) + tax_5_percent(income_3000_to_6000))

-- Statement of the proof problem
theorem income_before_taxes_correct :
  ∃ I : ℝ, net_income_after_taxes I = 15000 ∧ abs (I - 11138.24) < 0.01 :=
sorry

end income_before_taxes_correct_l536_536233


namespace polar_equation_parabola_l536_536481

/-- Given a polar equation 4 * ρ * (sin(θ / 2))^2 = 5, prove that it represents a parabola in Cartesian coordinates. -/
theorem polar_equation_parabola (ρ θ : ℝ) (h : 4 * ρ * (Real.sin (θ / 2))^ 2 = 5) : 
  ∃ (a : ℝ), a ≠ 0 ∧ (∃ b c : ℝ, ∀ x y : ℝ, (y^2 = a * (x + b)) ∨ (x = c ∨ y = 0)) := 
sorry

end polar_equation_parabola_l536_536481


namespace geometric_sequence_b_proof_l536_536149

noncomputable def geometric_sequence_b_positive : ℝ :=
let b := (67.5).sqrt in
if h : 0 < b then b else 0

theorem geometric_sequence_b_proof (b : ℝ) (hpos : 0 < b) (hgeo_seq : 30 * ((67.5).sqrt / 30) = b ∧ b * (b / 30) = 9 / 4) :
  b = (67.5).sqrt :=
begin
  sorry
end

end geometric_sequence_b_proof_l536_536149


namespace rem_value_l536_536184

def floor_div (x y : ℝ) : ℝ := ⌊x / y⌋

def rem (x y : ℝ) : ℝ := x - y * floor_div x y

theorem rem_value :
  rem (5/7) (-3/11) = -8/77 := by
  sorry

end rem_value_l536_536184


namespace coin_combinations_50_cents_l536_536733

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536733


namespace max_distance_from_circle_to_line_l536_536495

theorem max_distance_from_circle_to_line :
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
  let center : ℝ × ℝ := (0, 0)
  let radius : ℝ := 1
  let line_x_eq_2 : set (ℝ × ℝ) := { p | p.1 = 2 }
  let distance_from_center_to_line := 2
  (radius + distance_from_center_to_line = 3) :=
by
  sorry

end max_distance_from_circle_to_line_l536_536495


namespace whole_numbers_in_interval_7_4_3pi_l536_536862

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536862


namespace average_value_of_T_l536_536121

noncomputable def expected_value_T : ℕ := 22

theorem average_value_of_T (boys girls : ℕ) (boy_pair girl_pair : Prop) (T : ℕ) :
  boys = 9 → girls = 15 →
  boy_pair ∧ girl_pair →
  T = expected_value_T :=
by
  intros h_boys h_girls h_pairs
  sorry

end average_value_of_T_l536_536121


namespace complete_the_square_l536_536465

theorem complete_the_square (x : ℝ) (h : x^2 - 4 * x + 3 = 0) : (x - 2)^2 = 1 :=
sorry

end complete_the_square_l536_536465


namespace area_of_triangle_pfg_correct_l536_536391

noncomputable def area_of_triangle_pfg : ℝ :=
  let PQ := 10 in
  let QR := 12 in
  let PR := 14 in
  let PF := 3 in
  let PG := 5 in
  let S := (PQ + QR + PR) / 2 in
  let A_PQR := real.sqrt (S * (S - PQ) * (S - QR) * (S - PR)) in
  let sinP := (2 * A_PQR) / (PQ * PR) in
  (1/2) * PF * PG * sinP

theorem area_of_triangle_pfg_correct :
  area_of_triangle_pfg = (45 * real.sqrt 2) / 14 :=
sorry

end area_of_triangle_pfg_correct_l536_536391


namespace chess_positions_after_one_move_each_l536_536341

def number_of_chess_positions (initial_positions : ℕ) (pawn_moves : ℕ) (knight_moves : ℕ) (active_pawns : ℕ) (active_knights : ℕ) : ℕ :=
  let pawn_move_combinations := active_pawns * pawn_moves
  let knight_move_combinations := active_knights * knight_moves
  pawn_move_combinations + knight_move_combinations

theorem chess_positions_after_one_move_each :
  number_of_chess_positions 1 2 2 8 2 * number_of_chess_positions 1 2 2 8 2 = 400 :=
by
  sorry

end chess_positions_after_one_move_each_l536_536341


namespace x_finishes_in_24_days_l536_536551

variable (x y : Type) [Inhabited x] [Inhabited y]

/-- 
  y can finish the work in 16 days,
  y worked for 10 days and left the job,
  x alone needs 9 days to finish the remaining work,
  How many days does x need to finish the work alone?
-/
theorem x_finishes_in_24_days
  (days_y : ℕ := 16)
  (work_done_y : ℕ := 10)
  (work_left_x : ℕ := 9)
  (D_x : ℕ) :
  (1 / days_y : ℚ) * work_done_y + (1 / D_x) * work_left_x = 1 / D_x :=
by
  sorry

end x_finishes_in_24_days_l536_536551


namespace enclosed_area_of_curve_l536_536477

theorem enclosed_area_of_curve :
  let side_length := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let arc_length := Real.pi
  let arc_angle := Real.pi / 2
  let arc_radius := arc_length / arc_angle
  let sector_area := (arc_angle / (2 * Real.pi)) * Real.pi * arc_radius^2
  let total_sector_area := 12 * sector_area
  let enclosed_area := octagon_area + total_sector_area + 3 * Real.pi
  enclosed_area = 54 + 38.4 * Real.sqrt 2 + 3 * Real.pi :=
by
  -- We will use sorry to indicate the proof is omitted.
  sorry

end enclosed_area_of_curve_l536_536477


namespace correct_statements_l536_536580

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom f_symmetry : ∀ x, f(2 - x) = f(x)
axiom f_value_at_1 : f(1) = 2
axiom f_odd_transformation : ∀ x, f(3 * x + 2) = -f(-3 * x - 2)
axiom g_symmetry : ∀ x, g(x) = -g(4 - x)
axiom intersection_points : fin 2023 → ℝ × ℝ 

theorem correct_statements :
  let intersections := (λ i, intersection_points i) in
  (∀ x, f(2 - x) = f(x)) ∧
  (f(2) = 0) ∧ 
  ∑ i in finset.range 2023, (intersections i).fst + (intersections i).snd = 4046 :=
sorry

end correct_statements_l536_536580


namespace roots_of_cubic_eq_sum_l536_536407

namespace MathProof

open Real

theorem roots_of_cubic_eq_sum :
  ∀ a b c : ℝ, 
  (Polynomial.eval a (Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 2023 * Polynomial.X + Polynomial.C 4012) = 0) ∧
  (Polynomial.eval b (Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 2023 * Polynomial.X + Polynomial.C 4012) = 0) ∧
  (Polynomial.eval c (Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 2023 * Polynomial.X + Polynomial.C 4012) = 0) 
  → (a + b)^3 + (b + c)^3 + (c + a)^3 = 3009 :=
by
  intros a b c h
  sorry

end MathProof

end roots_of_cubic_eq_sum_l536_536407


namespace area_of_square_with_adjacent_points_l536_536092

theorem area_of_square_with_adjacent_points (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 6)) :
  let side_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in 
  let area := side_length^2 in 
  area = 25 :=
by
  sorry

end area_of_square_with_adjacent_points_l536_536092


namespace find_consecutive_integers_sum_eq_l536_536632

theorem find_consecutive_integers_sum_eq 
    (M : ℤ) : ∃ n k : ℤ, (0 ≤ k ∧ k ≤ 9) ∧ (M = (9 * n + 45 - k)) := 
sorry

end find_consecutive_integers_sum_eq_l536_536632


namespace trapezoid_to_square_l536_536451

-- Define a trapezoid in a geometrical context
structure Trapezoid :=
(base1 base2 height : ℝ)

-- A theorem stating that a given trapezoid can be cut into three parts and reassembled into a square
theorem trapezoid_to_square (T : Trapezoid) :
  ∃ (pieces : list Set(ℝ × ℝ)), pieces.length = 3 ∧ is_square (reassemble T pieces) :=
sorry

end trapezoid_to_square_l536_536451


namespace sqrt_of_product_powers_eq_twelve_l536_536614

theorem sqrt_of_product_powers_eq_twelve : (real.sqrt (3^3 * 2^5)) = 12 := by
  sorry

end sqrt_of_product_powers_eq_twelve_l536_536614


namespace a_45_eq_1991_l536_536697

-- Define the sequence a based on initial conditions and recurrence relation
noncomputable def a : ℕ → ℤ
| 0     := 11
| 1     := 11
| (m+n) := 1/2 * (a (2*m) + a (2*n)) - (m - n)^2

-- State the theorem to prove
theorem a_45_eq_1991 : a 45 = 1991 := by
  sorry

end a_45_eq_1991_l536_536697


namespace range_of_a_l536_536326

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - 2 * a * x

theorem range_of_a (a : ℝ) (h₀ : a > 0) 
  (h₁ h₂ : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ a - f x₂ a ≥ (3/2) - 2 * Real.log 2) : 
  a ≥ (3/2) * Real.sqrt 2 :=
sorry

end range_of_a_l536_536326


namespace minimum_value_frac_inv_is_one_third_l536_536017

noncomputable def min_value_frac_inv (x y : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ x + y = 12 then 1/x + 1/y else 0

theorem minimum_value_frac_inv_is_one_third (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x + y = 12) :
  min_value_frac_inv x y = 1/3 :=
begin
  -- Proof to be provided
  sorry
end

end minimum_value_frac_inv_is_one_third_l536_536017


namespace probability_sunglasses_to_hat_l536_536367

variable (S H : Finset ℕ) -- S: set of people wearing sunglasses, H: set of people wearing hats
variable (num_S : Nat) (num_H : Nat) (num_SH : Nat)
variable (prob_hat_to_sunglasses : ℚ)

-- Conditions
def condition1 : num_S = 80 := sorry
def condition2 : num_H = 50 := sorry
def condition3 : prob_hat_to_sunglasses = 3 / 5 := sorry
def condition4 : num_SH = (3/5) * 50 := sorry

-- Question: Prove that the probability a person wearing sunglasses is also wearing a hat
theorem probability_sunglasses_to_hat :
  (num_SH : ℚ) / num_S = 3 / 8 :=
sorry

end probability_sunglasses_to_hat_l536_536367


namespace initial_candies_l536_536433

theorem initial_candies (x : ℕ) (h1 : x % 4 = 0) (h2 : x / 4 * 3 / 3 * 2 / 2 - 24 ≥ 6) (h3 : x / 4 * 3 / 3 * 2 / 2 - 24 ≤ 9) :
  x = 64 :=
sorry

end initial_candies_l536_536433


namespace area_of_region_l536_536270

-- Define the region
def region := { p : ℝ × ℝ | abs p.1 - 1 ≤ p.2 ∧ p.2 ≤ real.sqrt (1 - p.1^2) }

-- Statement of the theorem
theorem area_of_region : 
  ∫ x in -1..1, ∫ y in (max (-1 : ℝ) (abs x - 1))..real.sqrt(1 - x^2), 1 = (real.pi / 2) + 1 :=
by
  sorry

end area_of_region_l536_536270


namespace students_joined_l536_536971

theorem students_joined (A X : ℕ) (h1 : 100 * A = 5000) (h2 : (100 + X) * (A - 10) = 5400) :
  X = 35 :=
by
  sorry

end students_joined_l536_536971


namespace solve_for_nabla_l536_536931

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 :=
by
  sorry

end solve_for_nabla_l536_536931


namespace num_triangles_2164_l536_536347

noncomputable def is_valid_triangle (p1 p2 p3 : ℤ × ℤ) : Prop :=
  let det := (fun (x1 y1 x2 y2 x3 y3 : ℤ) => x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
  det p1.1 p1.2 p2.1 p2.2 p3.1 p3.2 ≠ 0

noncomputable def num_valid_triangles : ℕ :=
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5}.to_finset.powerset 3
  points.count (λ t, match t.elems with
    | [p1, p2, p3] => is_valid_triangle p1 p2 p3
    | _ => false
  end)

theorem num_triangles_2164 : num_valid_triangles = 2164 := by
  sorry

end num_triangles_2164_l536_536347


namespace correct_graph_l536_536140

def g (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 1 then x^2 - 1 else
  if 1 ≤ x ∧ x ≤ 4 then -(x - 3)^2 + 2 else 0

def transformed_g (x : ℝ) : ℝ :=
  - (2 / 3) * g x - 2

theorem correct_graph :
  graph_of_y_eq_transformed_g_eq_graph_C :=
sorry

end correct_graph_l536_536140


namespace hangar_length_l536_536236

-- Define the conditions
def num_planes := 7
def length_per_plane := 40 -- in feet

-- Define the main theorem to be proven
theorem hangar_length : num_planes * length_per_plane = 280 := by
  -- Proof omitted with sorry
  sorry

end hangar_length_l536_536236


namespace whole_numbers_in_interval_l536_536915

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536915


namespace common_ratio_geom_arith_prog_l536_536483

theorem common_ratio_geom_arith_prog (a b c q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = a * q^2)
  (h3 : 2 * (2020 * b / 7) = 577 * a + c / 7) : 
  q = 4039 :=
begin
  -- proof to be filled
  sorry
end

end common_ratio_geom_arith_prog_l536_536483


namespace count_whole_numbers_in_interval_l536_536845

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536845


namespace max_intersections_circle_rectangle_l536_536182

theorem max_intersections_circle_rectangle (circle_intersects_side_at_most_twice : ∀ (c : Circle) (s : LineSegment), intersects c s ≤ 2) : 
  ∃ (max_points : Nat), max_points = 8 :=
by
  have max_intersections : 4 * 2 = 8 := by simp
  existsi 8
  exact max_intersections

end max_intersections_circle_rectangle_l536_536182


namespace A_work_days_l536_536572

theorem A_work_days (x : ℝ) (h1 : 1 / 15 + 1 / x = 1 / 8.571428571428571) : x = 20 :=
by
  sorry

end A_work_days_l536_536572


namespace count_whole_numbers_in_interval_l536_536871

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536871


namespace count_whole_numbers_in_interval_l536_536896

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536896


namespace count_ways_to_get_50_cents_with_coins_l536_536830

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536830


namespace combinations_of_coins_l536_536777

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536777


namespace range_of_m_l536_536304

variable (x m : ℝ)

noncomputable def p := abs (x - 1) ≤ 2
noncomputable def q := x^2 - 2 * x + 1 - m^2 ≤ 0
noncomputable def neg_p := abs (x - 1) > 2
noncomputable def neg_q := x^2 - 2 * x + 1 - m^2 > 0

theorem range_of_m (h_m_gt_zero : m > 0) (h_neg_p_necessary : ∀ x, neg_q x m → neg_p x) (h_neg_p_not_sufficient : ∃ x, neg_p x ∧ ¬ neg_q x m) : 
  3 ≤ m := sorry

end range_of_m_l536_536304


namespace perimeter_of_square_B_l536_536116

theorem perimeter_of_square_B (area_A : ℝ) (prob_not_in_B : ℝ) (h1 : area_A = 25) (h2 : prob_not_in_B = 0.64) : 
  let a := Real.sqrt area_A,
      area_B := (1 - prob_not_in_B) * area_A,
      b := Real.sqrt area_B
  in 4 * b = 12 := by
  sorry

end perimeter_of_square_B_l536_536116


namespace exponential_increasing_l536_536136

theorem exponential_increasing (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x < a^y) ↔ a > 1 :=
by
  sorry

end exponential_increasing_l536_536136


namespace shading_no_symmetry_l536_536245

def shadable_cells (n : ℕ) : ℕ :=
if even n then n^2 - 2 * n else n^2 - 4 * n + 3

theorem shading_no_symmetry (n : ℕ) :
  shadable_cells n = if even n then n^2 - 2 * n else n^2 - 4 * n + 3 :=
by sorry

end shading_no_symmetry_l536_536245


namespace min_value_reciprocal_sum_l536_536029

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end min_value_reciprocal_sum_l536_536029


namespace coin_combination_l536_536781

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536781


namespace large_square_min_side_and_R_max_area_l536_536366

-- Define the conditions
variable (s : ℝ) -- the side length of the larger square
variable (rect_1_side1 rect_1_side2 : ℝ) -- sides of the first rectangle
variable (square_side : ℝ) -- side of the inscribed square
variable (R_area : ℝ) -- area of the rectangle R

-- The known dimensions
axiom h1 : rect_1_side1 = 2
axiom h2 : rect_1_side2 = 4
axiom h3 : square_side = 2
axiom h4 : ∀ x y : ℝ, x > 0 → y > 0 → R_area = x * y -- non-overlapping condition

-- Define the result to be proved
theorem large_square_min_side_and_R_max_area 
  (h_r_fit_1 : rect_1_side1 + square_side ≤ s)
  (h_r_fit_2 : rect_1_side2 + square_side ≤ s)
  (h_R_max_area : R_area = 4)
  : s = 4 ∧ R_area = 4 := 
by 
  sorry

end large_square_min_side_and_R_max_area_l536_536366


namespace solve_for_x_l536_536459

theorem solve_for_x (x : ℝ) (h : (4/7) * (2/5) * x = 8) : x = 35 :=
sorry

end solve_for_x_l536_536459


namespace sufficient_condition_for_ellipse_with_foci_y_axis_l536_536661

theorem sufficient_condition_for_ellipse_with_foci_y_axis (m n : ℝ) (h : m > n ∧ n > 0) :
  (∃ a b : ℝ, (a^2 = m / n) ∧ (b^2 = 1 / n) ∧ (a > b)) ∧ ¬(∀ u v : ℝ, (u^2 = m / v) → (v^2 = 1 / v) → (u > v) → (v = n ∧ u = m)) :=
by
  sorry

end sufficient_condition_for_ellipse_with_foci_y_axis_l536_536661


namespace square_area_from_points_l536_536065

theorem square_area_from_points :
  let P1 := (1, 2)
  let P2 := (4, 6)
  let side_length := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let area := side_length^2
  P1.1 = 1 ∧ P1.2 = 2 ∧ P2.1 = 4 ∧ P2.2 = 6 →
  area = 25 :=
by
  sorry

end square_area_from_points_l536_536065


namespace count_whole_numbers_in_interval_l536_536846

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536846


namespace binomial_expansion_coefficient_l536_536474

theorem binomial_expansion_coefficient 
  (a : ℝ) 
  (h : a > 0) 
  (area_condition : ∫ x in 0..1, 2 * real.sqrt (a * x) = 4 / 3) : 
  (binomial_coeff 20 19) = 20 := 
sorry

end binomial_expansion_coefficient_l536_536474


namespace pyramid_base_edge_length_l536_536476

theorem pyramid_base_edge_length (height : ℝ) (radius : ℝ) (side_len : ℝ) :
  height = 4 ∧ radius = 3 →
  side_len = (12 * Real.sqrt 14) / 7 :=
by
  intros h
  rcases h with ⟨h1, h2⟩
  sorry

end pyramid_base_edge_length_l536_536476


namespace perpendicular_vector_x_value_l536_536708

-- Definitions based on the given problem conditions
def dot_product_perpendicular (a1 a2 b1 b2 x : ℝ) : Prop :=
  (a1 * b1 + a2 * b2 = 0)

-- Statement to be proved
theorem perpendicular_vector_x_value (x : ℝ) :
  dot_product_perpendicular 4 x 2 4 x → x = -2 :=
by
  intros h
  sorry

end perpendicular_vector_x_value_l536_536708


namespace combinations_of_coins_with_50_cents_l536_536746

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536746


namespace pencil_counts_l536_536421

theorem pencil_counts (s p t: ℕ) (h1 : 6 * s = 12) (h2 : t = 8 * s) (h3 : p = (2.5 * s).to_nat + 3) :
  t = 16 ∧ p = 8 :=
by
  have h_s := Nat.eq (6 * s) 12,
  sorry

end pencil_counts_l536_536421


namespace integer_solutions_count_l536_536938

theorem integer_solutions_count (x : ℕ) (h : ⌊Real.sqrt x⌋ = 8) : (finset.Icc 64 80).card = 17 :=
by
  sorry

end integer_solutions_count_l536_536938


namespace emily_lives_total_l536_536427

variable (x : ℤ)

def total_lives_after_stages (x : ℤ) : ℤ :=
  let lives_after_stage1 := x + 25
  let lives_after_stage2 := lives_after_stage1 + 24
  let lives_after_stage3 := lives_after_stage2 + 15
  lives_after_stage3

theorem emily_lives_total : total_lives_after_stages x = x + 64 := by
  -- The proof will go here
  sorry

end emily_lives_total_l536_536427


namespace whole_numbers_in_interval_7_4_3pi_l536_536859

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536859


namespace linear_function_not_third_quadrant_l536_536984

theorem linear_function_not_third_quadrant (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ¬ (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ y = k * x + 1) :=
sorry

end linear_function_not_third_quadrant_l536_536984


namespace sam_morning_run_distance_l536_536447

variable (x : ℝ) -- The distance of Sam's morning run in miles

theorem sam_morning_run_distance (h1 : ∀ y, y = 2 * x) (h2 : 12 = 12) (h3 : x + 2 * x + 12 = 18) : x = 2 :=
by sorry

end sam_morning_run_distance_l536_536447


namespace ellipse_k_range_l536_536135

theorem ellipse_k_range (k : ℝ) (h1 : k > 0) (h2 : 4 > k) : 0 < k ∧ k < 4 :=
begin
  split,
  { exact h1 },
  { exact h2 }
end

end ellipse_k_range_l536_536135


namespace count_whole_numbers_in_interval_l536_536848

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536848


namespace problem_statement_l536_536269

theorem problem_statement (x : ℝ) (h : x ≠ 2) :
  (x * (x + 1)) / ((x - 2)^2) ≥ 8 ↔ (1 ≤ x ∧ x < 2) ∨ (32/7 < x) :=
by 
  sorry

end problem_statement_l536_536269


namespace whole_numbers_in_interval_l536_536908

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536908


namespace coin_combinations_l536_536809

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536809


namespace positive_integers_n_le_1200_l536_536276

theorem positive_integers_n_le_1200 (n : ℕ) :
  n ≤ 1200 ∧ ∃ k : ℕ, 14 * n = k^2 → n ∈ {14 * k^2 | k ∈ finset.range 10} :=
by
  sorry

end positive_integers_n_le_1200_l536_536276


namespace trigonometric_identity_l536_536667

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = -(2 + Real.sqrt 3) / 3 := 
sorry

end trigonometric_identity_l536_536667


namespace projection_unique_l536_536101

theorem projection_unique (v p : ℝ^3) (a b : ℝ^3) (t : ℝ) :
  a = matrix.vec3 (-3) 2 1 →
  b = matrix.vec3 4 (-1) 5 →
  (a + t • (b - a)) = p →
  ∀ d : ℝ^3, (d = b - a) → (p ⋅ d = 0) → t = 23 / 74 →
  p = matrix.vec3 (-61 / 74) (79 / 74) (166 / 74) :=
begin
  intros ha hb h1 hd hdot ht,
  sorry
end

end projection_unique_l536_536101


namespace triangle_is_obtuse_l536_536960

theorem triangle_is_obtuse (A B C : ℝ) (a b c : ℝ) (h₁ : sin A / sin B = 2 / 3)
  (h₂ : sin B / sin C = 3 / 4) (h₃ : a / b = 2 / 3) (h₄ : b / c = 3 / 4) :
  A + B + C = π ∧ c^2 > a^2 + b^2 → C > π / 2 :=
by
  sorry

end triangle_is_obtuse_l536_536960


namespace coin_combinations_sum_50_l536_536717

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536717


namespace area_ratio_of_circles_l536_536358

theorem area_ratio_of_circles 
  (CX : ℝ)
  (CY : ℝ)
  (RX RY : ℝ)
  (hX : CX = 2 * π * RX)
  (hY : CY = 2 * π * RY)
  (arc_length_equality : (90 / 360) * CX = (60 / 360) * CY) :
  (π * RX^2) / (π * RY^2) = 9 / 4 :=
by
  sorry

end area_ratio_of_circles_l536_536358


namespace slope_of_tangent_l536_536322

theorem slope_of_tangent {x : ℝ} (h : x = 2) : deriv (λ x, 2 * x^2) x = 8 :=
by
  sorry

end slope_of_tangent_l536_536322


namespace count_whole_numbers_in_interval_l536_536890

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536890


namespace simplify_polynomial_l536_536457

variable (x : ℝ)

theorem simplify_polynomial :
  (2 * x^10 + 8 * x^9 + 3 * x^8) + (5 * x^12 - x^10 + 2 * x^9 - 5 * x^8 + 4 * x^5 + 6)
  = 5 * x^12 + x^10 + 10 * x^9 - 2 * x^8 + 4 * x^5 + 6 := by
  sorry

end simplify_polynomial_l536_536457


namespace trapezoid_angles_l536_536985

theorem trapezoid_angles
  {A B C D : Point}
  (h1 : is_trapezoid A B C D)
  (h2 : A = AD ∧ B = BC ∧ C = CD)
  (h3 : BC + CD = AD) :
  all_angles A B C D (72, 108, 144, 36) :=
by
  sorry

end trapezoid_angles_l536_536985


namespace combinations_of_coins_l536_536772

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536772


namespace inverse_function_composition_l536_536120

def g (x : ℝ) : ℝ := 3 * x + 7

noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

theorem inverse_function_composition : g_inv (g_inv 20) = -8 / 9 := by
  sorry

end inverse_function_composition_l536_536120


namespace avg_children_in_families_with_children_l536_536170

theorem avg_children_in_families_with_children
  (total_families : ℕ)
  (avg_children_per_family : ℕ)
  (childless_families : ℕ)
  (total_children : ℕ := total_families * avg_children_per_family)
  (families_with_children : ℕ := total_families - childless_families)
  (avg_children_in_families_with_children : ℚ := total_children / families_with_children) :
  avg_children_in_families_with_children = 4 :=
by
  have h1 : total_families = 12 := sorry
  have h2 : avg_children_per_family = 3 := sorry
  have h3 : childless_families = 3 := sorry
  have h4 : total_children = 12 * 3 := sorry
  have h5 : families_with_children = 12 - 3 := sorry
  have h6 : avg_children_in_families_with_children = (12 * 3) / (12 - 3) := sorry
  have h7 : ((12 * 3) / (12 - 3) : ℚ) = 4 := sorry
  exact h7

end avg_children_in_families_with_children_l536_536170


namespace count_ways_to_get_50_cents_with_coins_l536_536832

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536832


namespace differential_approximation_l536_536553

-- Define the function f as stated in the problem
def f (x : ℝ) : ℝ := (3 * x + cos x)^(1/3)

-- Define the given value of x
def x : ℝ := 0.01

-- Define the approximation of y based on the conditions
def y_approx : ℝ := 1.01

-- Formalize the proof statement
theorem differential_approximation : f x ≈ y_approx :=
sorry

end differential_approximation_l536_536553


namespace unit_digit_7_14_l536_536958

theorem unit_digit_7_14 : (7^14) % 10 = 9 := 
by
  have h1 : 7^1 % 10 = 7 := by norm_num
  have h2 : 7^2 % 10 = 9 := by norm_num
  have h3 : 7^3 % 10 = 3 := by norm_num
  have h4 : 7^4 % 10 = 1 := by norm_num
  have cycle : 7^14 % 10 = 7^2 % 10 := by
    have cycle_length := 14 % 4 = 2
    exact congr_arg (λ x, 7^x % 10) cycle_length
  rw cycle
  exact h2

end unit_digit_7_14_l536_536958


namespace num_solutions_l536_536343

-- Definitions from the problem conditions
def satisfies_system (x y : ℝ) : Prop :=
  x^2 + 3 * y = 3 ∧ abs (abs x - abs y) = 2

-- Statement of the problem
theorem num_solutions : 
  {p : ℝ × ℝ | satisfies_system p.1 p.2}.finite.card = 6 :=
by sorry

end num_solutions_l536_536343


namespace paint_needed_270_statues_l536_536356

theorem paint_needed_270_statues:
  let height_large := 12
  let paint_large := 2
  let height_small := 3
  let num_statues := 270
  let ratio_height := (height_small : ℝ) / (height_large : ℝ)
  let ratio_area := ratio_height ^ 2
  let paint_small := paint_large * ratio_area
  let total_paint := num_statues * paint_small
  total_paint = 33.75 := by
  sorry

end paint_needed_270_statues_l536_536356


namespace evan_needs_7_more_dollars_for_watch_l536_536620

/--
David found $12 on the street. He then gave it to his friend Evan who has $1 and needed to buy a watch worth $20. Show that Evan still needs $7 to buy the watch.
-/
theorem evan_needs_7_more_dollars_for_watch (money_found: ℕ) (initial_money: ℕ) (watch_cost: ℕ) (money_given: ℕ)
  (h1: money_found = 12)
  (h2: initial_money = 1)
  (h3: watch_cost = 20)
  (h4: money_given = 12):
  (watch_cost - (initial_money + money_given) = 7) :=
by
  rw [h1, h2, h3, h4]
  simp
  sorry

end evan_needs_7_more_dollars_for_watch_l536_536620


namespace oblomov_lost_weight_l536_536098

-- Define the initial weight
variable (W : ℝ)

-- Define the weight changes in each season
def spring_factor : ℝ := 0.75
def summer_factor : ℝ := 1.20
def autumn_factor : ℝ := 0.90
def winter_factor : ℝ := 1.20

-- Define the final weight after all seasonal changes
def final_weight : ℝ := W * spring_factor * summer_factor * autumn_factor * winter_factor

-- The Lean statement checking whether Oblomov lost weight
theorem oblomov_lost_weight (hW : W > 0) : final_weight W < W :=
by
  rw [final_weight]
  simp only [spring_factor, summer_factor, autumn_factor, winter_factor]
  norm_num
  -- 0.75 * 1.20 * 0.90 * 1.20 = 0.972
  have h : 0.972 < 1 := by norm_num
  exact mul_lt_mul_of_pos_left h hW
sorry

end oblomov_lost_weight_l536_536098


namespace area_of_square_with_adjacent_points_l536_536082

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l536_536082


namespace sum_of_fourth_powers_l536_536515

theorem sum_of_fourth_powers (n : ℤ) (h : (n - 2)^2 + n^2 + (n + 2)^2 = 2450) :
  (n - 2)^4 + n^4 + (n + 2)^4 = 1881632 :=
sorry

end sum_of_fourth_powers_l536_536515


namespace does_not_pass_through_quadrant_I_l536_536503

def quadrant (x y : ℝ) : Option (Fin 4) :=
  match x, y with
  | x, y if x > 0 ∧ y > 0 => some 0 -- Quadrant I
  | x, y if x < 0 ∧ y > 0 => some 1 -- Quadrant II
  | x, y if x < 0 ∧ y < 0 => some 2 -- Quadrant III
  | x, y if x > 0 ∧ y < 0 => some 3 -- Quadrant IV
  | _, _ => none

theorem does_not_pass_through_quadrant_I :
  ∀ x : ℝ, quadrant x (-2 * x - 1) ≠ some 0 :=
by
  intros
  sorry

end does_not_pass_through_quadrant_I_l536_536503


namespace constant_term_expansion_l536_536513

theorem constant_term_expansion (a : ℝ) :
  (∑ x in (x - (a / x)) * (2 * x + 1) ^ 4) = -81 →
  (constant_term ((x - (2 : ℝ) / x) * (2 * x + 1)^4) = -16) :=
by
  sorry

end constant_term_expansion_l536_536513


namespace congruence_problem_l536_536353

theorem congruence_problem (x : ℤ) (h : 5 * x + 8 ≡ 3 [MOD 19]) : 5 * x + 9 ≡ 4 [MOD 19] :=
sorry

end congruence_problem_l536_536353


namespace prod_roots_eq_l536_536409

noncomputable def polynomial := (3 : ℚ) * X^3 - 9 * X^2 + X - 7

theorem prod_roots_eq :
  (∀ (a b c : ℚ), (polynomial.eval a = 0) ∧ (polynomial.eval b = 0) ∧ (polynomial.eval c = 0) →
    a * b * c = 7 / 3) :=
by {
  sorry
}

end prod_roots_eq_l536_536409


namespace tangent_circumcircle_DBC_l536_536710

open EuclideanGeometry

-- Definitions
variables {A B C D : Point}
variable {O : Circle}

-- Conditions
def angle_ACB_45 := ∠ A C B = π / 4
def angle_ADB_60 := ∠ A D B = π / 3
def ratio_AD_DC := dist A D / dist D C = 2 / 1

-- Tangency Condition
def tangent_condition (O : Circle) (A B : Point) :=
  ∀ (p : Point), p ∈ Circle.tangent_point_set O → p = A ∨ p = B

-- Theorem statement
theorem tangent_circumcircle_DBC (h_angle_ACB : angle_ACB_45)
    (h_angle_ADB : angle_ADB_60) (h_ratio : ratio_AD_DC) :
    tangent_condition (circumcircle D B C) A B :=
sorry

end tangent_circumcircle_DBC_l536_536710


namespace card_cube_product_count_l536_536518

theorem card_cube_product_count : 
  let cards := (finset.range 100).1.map (λ n, 2^(n + 1)) ++ (finset.range 100).1.map (λ n, 3^(n + 1))
  in (cards.card_combinations 2).card = 4389 :=
by 
  have h₁ : (finset.range 100).card = 100 := by simp
  have h₂ : cards.card = 200 :=
    by simp [cards, finset.card_of_finset_range, list.card_append, h₁]
  have h₃ : (cards.card_combinations 2).card = 4389 := sorry
  exact h₃

end card_cube_product_count_l536_536518


namespace chord_length_30_degrees_through_focus_l536_536664

noncomputable def parabola_focus : ℝ × ℝ :=
  (3/4, 0)

noncomputable def parabola : ℝ → ℝ :=
  λ x, real.sqrt (3 * x)

noncomputable def line_through_focus (x : ℝ): ℝ :=
  real.sqrt 3 / 3 * (x - 3 / 4)

theorem chord_length_30_degrees_through_focus : 
  let F := parabola_focus in
  ∀ A B : ℝ × ℝ, A.2 = parabola A.1 ∧ B.2 = parabola B.1 ∧
  A.2 = line_through_focus A.1 ∧ B.2 = line_through_focus B.1 →
  |A.1 - B.1| = 12 
  :=
by 
  sorry

end chord_length_30_degrees_through_focus_l536_536664


namespace perimeter_of_quadrilateral_divided_by_b_l536_536512

theorem perimeter_of_quadrilateral_divided_by_b (b : ℝ) :
  let square_vertices := [(-2 * b, -2 * b), (2 * b, -2 * b), (-2 * b, 2 * b), (2 * b, 2 * b)],
      line_eq := λ x : ℝ, x / 3,
      right_intersect := (2 * b, 2 * b / 3),
      left_intersect := (-2 * b, -2 * b / 3)
  in
    let vertical1 := abs(-2 * b + 2 * b / 3),
        horizontal := 4 * b,
        slant := sqrt((4 * b) ^ 2 + ( (4 * b / 3)) ^ 2),
        vertical2 := vertical1
    in 
      (vertical1 + horizontal + slant + vertical2) / b = 4 * (5 + real.sqrt 10) / 3 :=
by
  sorry

end perimeter_of_quadrilateral_divided_by_b_l536_536512


namespace count_ways_to_get_50_cents_with_coins_l536_536822

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536822


namespace minimum_value_of_reciprocal_expression_l536_536683

noncomputable def f (x a : ℝ) : ℝ := sqrt (abs (2 * x - 1) + abs (x + 1) - a)

theorem minimum_value_of_reciprocal_expression (a m n : ℝ) (h_a : a ≤ 3/2) (h_k : a = 3/2) (h_sum : m + n = 2 * a) (h_m : m > 0) (h_n : n > 0) :
  (1/m + 4/n) ≥ 3 :=
by
  sorry

end minimum_value_of_reciprocal_expression_l536_536683


namespace total_cards_l536_536519

theorem total_cards (C : ℕ) 
  (h1 : 2 / 5 * C = R)
  (h2 : 5 / 9 * (C - R) = B)
  (h3 : 32 = C - R - B) :
  C = 120 :=
begin
  sorry
end

end total_cards_l536_536519


namespace fewer_men_than_women_l536_536399

/-- Lauryn employs 80 men and there are 180 people working in total. Prove that there are 20 fewer men than women. -/
theorem fewer_men_than_women (total_people men : ℕ) (h1 : total_people = 180) (h2 : men = 80) :
  (total_people - men) - men = 20 :=
by
  simp [h1, h2]
  sorry

end fewer_men_than_women_l536_536399


namespace real_and_distinct_roots_l536_536285

-- Definition of the quadratic equation with parameter m
def quadratic_eq (m : ℝ) := λ x : ℝ, x^2 - m * x + 2 * m - 3

-- Condition for real and distinct roots: discriminant > 0
def discriminant (m : ℝ) : ℝ := m^2 - 8 * m + 12

theorem real_and_distinct_roots (m : ℝ) :
  (quadratic_eq m).discriminant > 0 ↔ m < 2 ∨ m > 6 := 
sorry

end real_and_distinct_roots_l536_536285


namespace whole_numbers_in_interval_7_4_3pi_l536_536857

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536857


namespace parallel_vectors_m_value_l536_536339

theorem parallel_vectors_m_value :
  ∀ (m : ℝ), let a := (1, 2) ∧ b := (-2, m) in (∃ k : ℝ, (1, 2) = (k * -2, k * m)) → m = -4 :=
by
  -- no proof required as per instructions
  sorry

end parallel_vectors_m_value_l536_536339


namespace xiao_zhang_password_l536_536545

theorem xiao_zhang_password : 
  ∃ password : String, 
    (∀ d₁ d₂ d₃ d₄ d₅, password = d₁ ++ d₂ ++ d₃ ++ d₄ ++ d₅ → d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₄ ≠ d₅) ∧ 
    (∀ d₁ d₂ d₃ d₄ d₅, (password = d₁ ++ d₂ ++ d₃ ++ d₄ ++ d₅) → 
      ((d₁ ∈ ['5', '1', '9', '3', '2']) ∧ (d₃ ∈ ['5', '1', '9', '3', '2'])) ∨ 
      ((d₁ ∈ ['8', '5', '4', '7', '8']) ∧ (d₅ ∈ ['8', '5', '4', '7', '8'])) ∨ 
      ((d₃ ∈ ['3', '4', '9', '0', '6']) ∧ (d₅ ∈ ['3', '4', '9', '0', '6']))) ∧ 
    (password = "55976" ∨ password = "75972") 
    := sorry

end xiao_zhang_password_l536_536545


namespace area_of_square_l536_536055

-- We define the points as given in the conditions
def point1 : ℝ × ℝ := (1, 2)
def point2 : ℝ × ℝ := (4, 6)

-- Lean's "def" defines the concept of a square given two adjacent points.
def is_square (p1 p2: ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  in ∃ (l : ℝ), l = d ∧ (l^2 = 25)

-- The theorem assumes the points are adjacent points on a square and proves that their area is 25.
theorem area_of_square :
  is_square point1 point2 :=
by
  -- Insert formal proof here, skipped with 'sorry' for this task
  sorry

end area_of_square_l536_536055


namespace ord_p_factorial_factorial_ratio_is_integer_another_factorial_ratio_is_integer_l536_536298

-- Part 1
theorem ord_p_factorial (n p : ℕ) (hp : p > 1) :
  ∏ ord_p n n! = (n - S_p(n)) / (p - 1) := 
sorry

-- Part 2
theorem factorial_ratio_is_integer (n : ℕ) :
  ∏ n > 0 → ∃ k : ℕ, ( ( (2 * n)! / ((n!) * ((n + 1)!) ) = k ) ) :=
sorry

-- Part 3
theorem another_factorial_ratio_is_integer (m n : ℕ) (Hmn : gcd m (n + 1) = 1) :
  ∃ k : ℕ, ( ( (m * n + n)! / ( (m * n)! * ( (n + 1)! ) ) = k ) ) :=
sorry

end ord_p_factorial_factorial_ratio_is_integer_another_factorial_ratio_is_integer_l536_536298


namespace exchange_process_time_limit_l536_536565

theorem exchange_process_time_limit :
  ∀ (init_positions final_positions : list ℕ) (n : ℕ),
  (init_positions = list.range n.map (λ i, 2 * (i + 1))) →
  (final_positions = list.range n.map (λ i, i + 1)) →
  (sum init_positions - sum final_positions = 55) →
  n = 10 →
  55 < 60 :=
by
  intros init_positions final_positions n h1 h2 h3 h4
  -- Goals are solvable as per given conditions, but we skip proof
  exact nat.lt_of_add_lt_add_right h4
  sorry

end exchange_process_time_limit_l536_536565


namespace two_digit_numbers_reverse_square_condition_l536_536229

theorem two_digit_numbers_reverse_square_condition :
  ∀ (a b : ℕ), 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 →
  (∃ n : ℕ, 10 * a + b + 10 * b + a = n^2) ↔ 
  (10 * a + b = 29 ∨ 10 * a + b = 38 ∨ 10 * a + b = 47 ∨ 10 * a + b = 56 ∨ 
   10 * a + b = 65 ∨ 10 * a + b = 74 ∨ 10 * a + b = 83 ∨ 10 * a + b = 92) :=
by {
  sorry
}

end two_digit_numbers_reverse_square_condition_l536_536229


namespace coin_combination_l536_536785

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536785


namespace measure_angle_D_l536_536527

def is_isosceles (D E F : ℝ) : Prop := D = F
def triangle_angle_sum (D E F : ℝ) : Prop := D + E + F = 180
def relation (F E : ℝ) : Prop := F = 3 * E

theorem measure_angle_D (D E F : ℝ) (h1 : is_isosceles D F) (h2 : relation F E) (h3 : triangle_angle_sum D E F) :
  D = 77 :=
sorry

end measure_angle_D_l536_536527


namespace triangle_with_sides_3x_4x_5x_is_right_l536_536509

theorem triangle_with_sides_3x_4x_5x_is_right (x : ℝ) (hx : 0 < x) : 
  let a := 3 * x, b := 4 * x, c := 5 * x in a^2 + b^2 = c^2 :=
by
  -- Definitions of sides
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  -- Proof of the Pythagorean identity
  sorry

end triangle_with_sides_3x_4x_5x_is_right_l536_536509


namespace median_locus_l536_536105

-- Assume we have a definition for areas of triangles and points inside a triangle
def area (A B C : Point) : Real := sorry  -- Area definition placeholder
def is_inside (M A B C : Point) : Prop := sorry  -- Inside triangle definition placeholder

theorem median_locus (A B C M : Point) (h_interior: is_inside M A B C)
  (h_area : area M A B = area M B C + area M C A) :
  is_on_median M A B C := 
sorry  -- Proof to be provided

end median_locus_l536_536105


namespace proof_problem_l536_536319

theorem proof_problem (k : ℝ) :
  (∃ (a b : ℝ), ((x - a)^2 + y^2 = 9) ∧ (a = 1) ∧ (b = 0)) ∧
  ((-1, 1) ∈ {p | ∃ k : ℝ, p.snd = k * (p.fst + 1) + 1}) ∧
  (∃ (d : ℝ), d = sqrt ((1 - (-1))^2 + (0 - 1)^2) ∧ d < 3) ∧
  (¬ (∃ (d : ℝ), d = 2 * sqrt (9 - ((3) / sqrt (1 + k^2))^2) ∧ d = 4 * sqrt 2)) :=
by
  sorry

end proof_problem_l536_536319


namespace trip_drop_probability_l536_536395

-- Definitions
def P_Trip : ℝ := 0.4
def P_Drop_not : ℝ := 0.9

-- Main theorem
theorem trip_drop_probability : ∀ (P_Trip P_Drop_not : ℝ), P_Trip = 0.4 → P_Drop_not = 0.9 → 1 - P_Drop_not = 0.1 :=
by
  intros P_Trip P_Drop_not h1 h2
  rw [h2]
  norm_num

end trip_drop_probability_l536_536395


namespace f_periodic_odd_l536_536678

noncomputable def f (x : ℝ) : ℝ := sorry -- Define f(x) properly only from the given conditions

theorem f_periodic_odd : 
  odd f ∧ periodic f π ∧ (∀ (x : ℝ), 0 < x ∧ x < π / 2 → f x = 2 * Real.sin x) → 
  f (11 * π / 6) = -1 := 
by
sorry

end f_periodic_odd_l536_536678


namespace total_gas_cost_l536_536598

/-
Definitions and conditions derived from part (a)
-/
def mpg_car1 := 50
def mpg_car2 := 10
def mpg_car3 := 15
def mpg_car4 := 25
def mpg_car5 := 45

def cost_per_gallon_car1 := 2.0
def cost_per_gallon_car2 := 2.5
def cost_per_gallon_car3 := 3.0
def cost_per_gallon_car4 := 2.75
def cost_per_gallon_car5 := 2.25

def total_miles := 450
def proportion_car1 := 0.30
def proportion_car2 := 0.15
def proportion_car3 := 0.10
def proportion_car4 := 0.25
def proportion_car5 := 0.20

/-
The proof problem we need to establish
-/
theorem total_gas_cost :
  let miles_car1 := total_miles * proportion_car1,
      miles_car2 := total_miles * proportion_car2,
      miles_car3 := total_miles * proportion_car3,
      miles_car4 := total_miles * proportion_car4,
      miles_car5 := total_miles * proportion_car5,

      gallons_car1 := miles_car1 / mpg_car1,
      gallons_car2 := miles_car2 / mpg_car2,
      gallons_car3 := miles_car3 / mpg_car3,
      gallons_car4 := miles_car4 / mpg_car4,
      gallons_car5 := miles_car5 / mpg_car5,

      cost_car1 := gallons_car1 * cost_per_gallon_car1,
      cost_car2 := gallons_car2 * cost_per_gallon_car2,
      cost_car3 := gallons_car3 * cost_per_gallon_car3,
      cost_car4 := gallons_car4 * cost_per_gallon_car4,
      cost_car5 := gallons_car5 * cost_per_gallon_car5 in
  
  cost_car1 + cost_car2 + cost_car3 + cost_car4 + cost_car5 = 48.15 :=
by
  -- Placeholder for the proof.
  sorry

end total_gas_cost_l536_536598


namespace exists_m_int_l536_536014

section
variable (k : ℕ) [Fact (k > 0)]

noncomputable def r : ℝ := k + 1 / 2

noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉

noncomputable def f (x : ℝ) : ℝ := x * (ceil x)

noncomputable def fn (l : ℕ) (x : ℝ) : ℝ :=
  Nat.recOn l f (λ l' h, f h) x

def v2 (n : ℕ) : ℕ :=
  Nat.find (λ m, 2 ^ m ∣ n ∧ ¬2^(m+1) ∣ n)

theorem exists_m_int (k_pos : 0 < k) : ∃ m > 0, ∃ x : ℝ, fn m r = x ∧ x % 1 = 0 :=
  by
  sorry
end

end exists_m_int_l536_536014


namespace sqrt_of_sqrt_49_cube_root_of_neg_8_div_27_l536_536511

theorem sqrt_of_sqrt_49 : ∃ (x : ℝ), x = sqrt(sqrt(49)) ∧ (x = sqrt(7) ∨ x = -sqrt(7)) := sorry

theorem cube_root_of_neg_8_div_27 : sqrt[3] (-8/27:ℝ) = -2/3 := sorry

end sqrt_of_sqrt_49_cube_root_of_neg_8_div_27_l536_536511


namespace area_of_square_with_adjacent_points_l536_536091

theorem area_of_square_with_adjacent_points (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 6)) :
  let side_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in 
  let area := side_length^2 in 
  area = 25 :=
by
  sorry

end area_of_square_with_adjacent_points_l536_536091


namespace area_of_square_with_adjacent_points_l536_536076

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l536_536076


namespace Marissa_sister_height_l536_536429

theorem Marissa_sister_height (sunflower_height_feet : ℕ) (height_difference_inches : ℕ) :
  sunflower_height_feet = 6 -> height_difference_inches = 21 -> 
  let sunflower_height_inches := sunflower_height_feet * 12
  let sister_height_inches := sunflower_height_inches - height_difference_inches
  let sister_height_feet := sister_height_inches / 12
  let sister_height_remainder_inches := sister_height_inches % 12
  sister_height_feet = 4 ∧ sister_height_remainder_inches = 3 :=
by
  intros
  sorry

end Marissa_sister_height_l536_536429


namespace count_whole_numbers_in_interval_l536_536897

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536897


namespace john_share_l536_536442

theorem john_share
  (total_amount : ℝ)
  (john_ratio jose_ratio binoy_ratio : ℝ)
  (total_amount_eq : total_amount = 6000)
  (ratios_eq : john_ratio = 2 ∧ jose_ratio = 4 ∧ binoy_ratio = 6) :
  (john_ratio / (john_ratio + jose_ratio + binoy_ratio)) * total_amount = 1000 :=
by
  -- Here we would derive the proof, but just use sorry for the moment.
  sorry

end john_share_l536_536442


namespace combinations_of_coins_with_50_cents_l536_536750

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536750


namespace range_of_x_for_sqrt_l536_536943

theorem range_of_x_for_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 :=
sorry

end range_of_x_for_sqrt_l536_536943


namespace combinations_of_coins_l536_536767

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536767


namespace determine_a_l536_536674

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 2 * x

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) → a = 0 :=
by
  intros h
  sorry

end determine_a_l536_536674


namespace find_slope_of_l3_l536_536043

-- Definitions of points and lines involved
def A := (3, -2) : ℝ × ℝ
def l1 (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def l2 (y : ℝ) : Prop := y = 2
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs (1/2 * (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Coordinates of point B where lines l1 and l2 intersect
def B := (0, 2) : ℝ × ℝ

-- Conditions and constraints from the problem
def line_l3 (m : ℝ) (x y : ℝ) : Prop := y = m * (x - A.1) + A.2
def intersects_l2 (C : ℝ × ℝ) : Prop := C.2 = 2

-- The proof problem statement
theorem find_slope_of_l3 :
  ∃ (m : ℝ), 
    m > 0 ∧ -- Positive slope
    (∃ C : ℝ × ℝ,
      intersects_l2 C ∧
      line_l3 m C.1 C.2 ∧
      area_of_triangle A B C = 10) ∧
    m = 2 :=
sorry

end find_slope_of_l3_l536_536043


namespace square_area_from_points_l536_536066

theorem square_area_from_points :
  let P1 := (1, 2)
  let P2 := (4, 6)
  let side_length := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let area := side_length^2
  P1.1 = 1 ∧ P1.2 = 2 ∧ P2.1 = 4 ∧ P2.2 = 6 →
  area = 25 :=
by
  sorry

end square_area_from_points_l536_536066


namespace count_whole_numbers_in_interval_l536_536905

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536905


namespace find_k_l536_536934

theorem find_k (k : ℤ) (h : ∃ p : ℤ[X], 9 * X^3 + k * X^2 + 16 * X + 64 = (3 * X + 4) * p) : k = -12 :=
sorry

end find_k_l536_536934


namespace value_of_x_l536_536185

theorem value_of_x :
  ∃ x : ℕ, 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 = 2 ^ x ∧ x = 53 :=
by {
  existsi (53),
  have h : 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 = 8 * 32 ^ 10 :=
  by sorry, -- Step required to show summation.
  have h1 : 8 * 32 ^ 10 = 2 ^ 53 :=
  by sorry, -- Step required to convert 8 and multiply exponent.
  exact ⟨h, h1.symm⟩,
}

end value_of_x_l536_536185


namespace sqrt_meaningful_real_l536_536942

theorem sqrt_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 5)) → x ≥ 5 :=
by
  intro h
  cases h with y hy
  have : x - 5 ≥ 0 := by sorry -- simplified proof of sqrt definition
  linarith

end sqrt_meaningful_real_l536_536942


namespace range_of_a_l536_536038

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ real.pi / 2 ∧ (real.cos x)^2 + 2 * real.cos x - a = 0) 
  ⊕ (∀ x : ℝ, x^2 + 2 * a * x - 8 + 6 * a ≥ 0) 
  → ¬((∃ x : ℝ, 0 ≤ x ∧ x ≤ real.pi / 2 ∧ (real.cos x)^2 + 2 * real.cos x - a = 0) 
  ∧ (∀ x : ℝ, x^2 + 2 * a * x - 8 + 6 * a ≥ 0)) 
  → a ∈ set.Ioo 0 2 ∪ set.Ioo 3 4 :=
by
  sorry

end range_of_a_l536_536038


namespace hyperbola_foci_x_axis_probability_l536_536611

-- Conditions
def is_valid_pair (m n : ℤ) : Prop :=
  m ∈ {-1, 2, 3} ∧ n ∈ {-1, 2, 3}

def is_hyperbola_with_foci_on_x_axis (m n : ℤ) : Prop :=
  m > 0 ∧ n > 0

-- Proof problem
theorem hyperbola_foci_x_axis_probability :
  let valid_pairs := [(m, n) | m ∈ {-1, 2, 3}, n ∈ {-1, 2, 3}, is_valid_pair m n];
  let count_valid := valid_pairs.length;
  let count_hyperbola := (valid_pairs.filter (λ p, is_hyperbola_with_foci_on_x_axis p.fst p.snd)).length in
  count_valid = 7 ∧ count_hyperbola = 4 →
  (count_hyperbola : ℚ) / count_valid = 4 / 7 :=
by
  sorry

end hyperbola_foci_x_axis_probability_l536_536611


namespace hillary_descending_rate_is_1000_l536_536340

-- Definitions from the conditions
def base_to_summit_distance : ℕ := 5000
def hillary_departure_time : ℕ := 6
def hillary_climbing_rate : ℕ := 800
def eddy_climbing_rate : ℕ := 500
def hillary_stop_distance_from_summit : ℕ := 1000
def hillary_and_eddy_pass_time : ℕ := 12

-- Derived definitions
def hillary_climbing_time : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) / hillary_climbing_rate
def hillary_stop_time : ℕ := hillary_departure_time + hillary_climbing_time
def eddy_climbing_time_at_pass : ℕ := hillary_and_eddy_pass_time - hillary_departure_time
def eddy_climbed_distance : ℕ := eddy_climbing_rate * eddy_climbing_time_at_pass
def hillary_distance_descended_at_pass : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) - eddy_climbed_distance
def hillary_descending_time : ℕ := hillary_and_eddy_pass_time - hillary_stop_time 

def hillary_descending_rate : ℕ := hillary_distance_descended_at_pass / hillary_descending_time

-- Statement to prove
theorem hillary_descending_rate_is_1000 : hillary_descending_rate = 1000 := 
by
  sorry

end hillary_descending_rate_is_1000_l536_536340


namespace min_value_inv_sum_l536_536027

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end min_value_inv_sum_l536_536027


namespace whole_numbers_in_interval_l536_536912

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536912


namespace final_quarters_l536_536443

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 760
def first_spent : ℕ := 418
def second_spent : ℕ := 192

-- Define the final amount of quarters Sally should have
theorem final_quarters (initial_quarters first_spent second_spent : ℕ) : initial_quarters - first_spent - second_spent = 150 :=
by
  sorry

end final_quarters_l536_536443


namespace problem_statement_l536_536981

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

def curve_polar (ρ θ : ℝ) : Prop :=
  ρ^2 * (Real.cos θ)^2 + 9 * (ρ * Real.sin θ)^2 = 9

def line_cartesian : Prop :=
  ∀ (x y : ℝ), (∃ t : ℝ, (x, y) = line_parametric t) ↔ x - Real.sqrt 3 * y - 2 = 0

def curve_cartesian : Prop :=
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ curve_polar ρ θ) ↔ x^2 / 9 + y^2 = 1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def intersection_dist_sum : ℝ :=
  ∑ q in (({p : ℝ × ℝ | ∃ t, p = line_parametric t ∧ (p.1 ^ 2 / 9 + p.2 ^ 2 = 1)}).to_finset \ {((2,0)})), distance (2,0) q

theorem problem_statement :
  line_cartesian ∧ curve_cartesian ∧ intersection_dist_sum = 2 * Real.sqrt 2 :=
by sorry

end problem_statement_l536_536981


namespace coin_combinations_l536_536804

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536804


namespace quadratic_roots_product_l536_536654

theorem quadratic_roots_product :
  ∀ (x1 x2: ℝ), (x1^2 - 4 * x1 - 2 = 0 ∧ x2^2 - 4 * x2 - 2 = 0) → (x1 * x2 = -2) :=
by
  -- Assume x1 and x2 are roots of the quadratic equation
  intros x1 x2 h
  sorry

end quadratic_roots_product_l536_536654


namespace area_of_square_with_adjacent_points_l536_536086

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def side_length := distance 1 2 4 6
def area_of_square (side : ℝ) : ℝ := side ^ 2

theorem area_of_square_with_adjacent_points :
  area_of_square side_length = 25 :=
by
  unfold side_length
  unfold area_of_square
  sorry

end area_of_square_with_adjacent_points_l536_536086


namespace find_major_axis_range_l536_536314

-- Definitions corresponding to each condition from part a:
variables (a b : ℝ) (h_a_gt_b : a > b) (h_b_gt_0 : b > 0)
variables (e : ℝ) (h_eccentricity : (Real.sqrt 3 / 3) ≤ e ∧ e ≤ Real.sqrt 2 / 2)
variables (h_eccentricity_def : e = Real.sqrt(1 - (b^2 / a^2)))

-- Derived conditions:
def major_axis_range : Prop := (Real.sqrt 5) ≤ 2 * a ∧ 2 * a ≤ (Real.sqrt 6)

-- Statement to prove the range of values for the major axis:
theorem find_major_axis_range (h1: a^2 + b^2 > 1) (h2: a^2 + b^2 = 2 * (a^2) * (b^2))
  (h3: b^2 = a^2 / (2 * a^2 - 1)) : major_axis_range a :=
sorry

end find_major_axis_range_l536_536314


namespace avg_children_in_families_with_children_l536_536169

theorem avg_children_in_families_with_children
  (total_families : ℕ)
  (avg_children_per_family : ℕ)
  (childless_families : ℕ)
  (total_children : ℕ := total_families * avg_children_per_family)
  (families_with_children : ℕ := total_families - childless_families)
  (avg_children_in_families_with_children : ℚ := total_children / families_with_children) :
  avg_children_in_families_with_children = 4 :=
by
  have h1 : total_families = 12 := sorry
  have h2 : avg_children_per_family = 3 := sorry
  have h3 : childless_families = 3 := sorry
  have h4 : total_children = 12 * 3 := sorry
  have h5 : families_with_children = 12 - 3 := sorry
  have h6 : avg_children_in_families_with_children = (12 * 3) / (12 - 3) := sorry
  have h7 : ((12 * 3) / (12 - 3) : ℚ) = 4 := sorry
  exact h7

end avg_children_in_families_with_children_l536_536169


namespace triangle_is_right_triangle_l536_536658

theorem triangle_is_right_triangle (A B C : ℝ) (hC_eq_A_plus_B : C = A + B) (h_angle_sum : A + B + C = 180) : C = 90 :=
by
  sorry

end triangle_is_right_triangle_l536_536658


namespace combinations_of_coins_l536_536776

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536776


namespace sequence_sum_1234_l536_536385

noncomputable def sequence_sum : ℕ → ℕ
| 0 := 1
| n := if n = 0 then 1 else (n / (n + 1) + 1)

theorem sequence_sum_1234 : (finset.range 1234).sum sequence_sum = 2419 := by
  sorry

end sequence_sum_1234_l536_536385


namespace combinations_of_coins_l536_536761

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536761


namespace extreme_points_f_x2_l536_536307

noncomputable def f (a x : ℝ) := x^2 + a * real.log (1 + x)

theorem extreme_points_f_x2:
  ∀ (a x1 x2 : ℝ), x1 < x2 ∧ (x1 + x2 = -1) ∧ (x1 * x2 = a) ∧ (x2 ∈ Ioo (-1/2 : ℝ) (0 : ℝ)) →
  f a x2 > (1 - 2 * real.log 2) / 4 := by
  sorry

end extreme_points_f_x2_l536_536307


namespace simplify_expression_l536_536455

variable {x : ℤ}

theorem simplify_expression : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 := 
by 
  calc
    3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = (3 + 6 + 9 + 12 + 15) * x + 18 : by ring
    ... = 45 * x + 18 : by norm_num

end simplify_expression_l536_536455


namespace coin_combinations_l536_536803

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536803


namespace find_quadruples_l536_536268

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

theorem find_quadruples (a b p n : ℕ) (hp : is_prime p) (h_ab : a + b ≠ 0) :
  a^3 + b^3 = p^n ↔ (a = 1 ∧ b = 1 ∧ p = 2 ∧ n = 1) ∨
               (a = 1 ∧ b = 2 ∧ p = 3 ∧ n = 2) ∨ 
               (a = 2 ∧ b = 1 ∧ p = 3 ∧ n = 2) ∨
               ∃ (k : ℕ), (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨ 
                          (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
                          (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end find_quadruples_l536_536268


namespace combination_coins_l536_536812

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536812


namespace find_a_l536_536653

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → 
     ∀ (x' y' : ℝ), (x' = x - 2 * (x - a * y + 2) / (1 + a^2)) ∧ (y' = y - 2 * a * (x - a * y + 2) / (1 + a^2)) → 
     (x'^2 + y'^2 + 2 * x' - 4 * y' + 1 = 0)) → 
  (a = -1 / 2) := 
sorry

end find_a_l536_536653


namespace count_whole_numbers_in_interval_l536_536889

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536889


namespace simplify_expression_l536_536454

variable {x : ℤ}

theorem simplify_expression : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 := 
by 
  calc
    3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = (3 + 6 + 9 + 12 + 15) * x + 18 : by ring
    ... = 45 * x + 18 : by norm_num

end simplify_expression_l536_536454


namespace number_of_integers_in_interval_l536_536926

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536926


namespace compound_interest_rate_approx_l536_536177

-- Assuming that the principal, final amount, number of times compounded per year, and number of years are defined
def principal : ℝ := 780
def final_amount : ℝ := 1300
def times_compounded_per_year : ℕ := 4
def years : ℕ := 4

-- Defining the compound interest rate
noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  ((A / P) ^ (1 / (n * t)) - 1) * n

-- Assertion of the value of the compound interest rate
theorem compound_interest_rate_approx :
  compound_interest_rate principal final_amount times_compounded_per_year years ≈ 0.1396 := by
  sorry

end compound_interest_rate_approx_l536_536177


namespace coin_combinations_50_cents_l536_536731

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536731


namespace complex_power_sum_l536_536418

noncomputable def z : ℂ := sorry

theorem complex_power_sum (hz : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -2 :=
sorry

end complex_power_sum_l536_536418


namespace area_of_part_of_circle_enclosed_between_chords_l536_536969

noncomputable def area_between_chords (R : ℝ) := 
  0.5 * R^2 * (Real.pi + Real.sqrt 3)

theorem area_of_part_of_circle_enclosed_between_chords (R : ℝ) :
  (∃ (A1 A2 : Set (ℝ^2)), 
    (chord_subtends_angle A1 R (Real.pi / 3)) ∧ 
    (chord_subtends_angle A2 R (2 * Real.pi / 3)) ∧ 
    (area_between A1 A2 = area_between_chords R)) := 
sorry

end area_of_part_of_circle_enclosed_between_chords_l536_536969


namespace infinite_integer_solutions_l536_536037

theorem infinite_integer_solutions (n : ℕ) 
  (a : Fin (n + 1) → ℕ) 
  (coprime_conditions : ∀ i : Fin n, Nat.coprime (a i) (a ⟨n, by simp⟩)) :
  ∃ infinitely_many (solutions : Fin (n + 1) → ℕ),
    (λ x, ∑ i : Fin n, x i ^ a i = (x ⟨n, by simp⟩) ^ (a ⟨n, by simp⟩)) solutions := sorry

end infinite_integer_solutions_l536_536037


namespace square_area_from_points_l536_536068

theorem square_area_from_points :
  let P1 := (1, 2)
  let P2 := (4, 6)
  let side_length := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let area := side_length^2
  P1.1 = 1 ∧ P1.2 = 2 ∧ P2.1 = 4 ∧ P2.2 = 6 →
  area = 25 :=
by
  sorry

end square_area_from_points_l536_536068


namespace count_whole_numbers_in_interval_l536_536842

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536842


namespace area_of_square_with_adjacent_points_l536_536096

theorem area_of_square_with_adjacent_points (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 6)) :
  let side_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in 
  let area := side_length^2 in 
  area = 25 :=
by
  sorry

end area_of_square_with_adjacent_points_l536_536096


namespace value_of_x_l536_536186

theorem value_of_x :
  ∃ x : ℕ, 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 = 2 ^ x ∧ x = 53 :=
by {
  existsi (53),
  have h : 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 + 32 ^ 10 = 8 * 32 ^ 10 :=
  by sorry, -- Step required to show summation.
  have h1 : 8 * 32 ^ 10 = 2 ^ 53 :=
  by sorry, -- Step required to convert 8 and multiply exponent.
  exact ⟨h, h1.symm⟩,
}

end value_of_x_l536_536186


namespace ladder_base_distance_l536_536567

theorem ladder_base_distance
  (c : ℕ) (b : ℕ) (hypotenuse : c = 13) (wall_height : b = 12) :
  ∃ x : ℕ, x^2 + b^2 = c^2 ∧ x = 5 := by
  sorry

end ladder_base_distance_l536_536567


namespace number_of_correct_statements_l536_536496

-- Define the five statements as per the problem conditions
def statement1 : Prop := ∀ (x : ℚ), (x ∈ ℤ ∨ (∃ p q : ℤ, q ≠ 0 ∧ x = p/q)) -- Rational numbers refer to integers and fractions
def statement2 : Prop := ∀ a : ℤ, a < 0 → -1 ≤ a -- The largest negative integer is -1
def statement3 : Prop := ∀ a : ℚ, abs a = 0 ↔ a = 0 -- The number with the smallest absolute value is 0
def statement4 : Prop := ∀ a : ℝ, (a^2 = a) ↔ (a = 1 ∨ a = 0) -- A number that is equal to its own square is 1 and 0
def statement5 : Prop := ∀ a : ℝ, (a^3 = a) → (a = 1 ∨ a = -1) -- A number that is equal to its own cube is only 1 and -1

-- Define the main problem: The number of correct statements among the five is 4
theorem number_of_correct_statements : (statement1 ∧ statement2 ∧ statement3 ∧ statement4 ∧ ¬statement5) → 4 := sorry

end number_of_correct_statements_l536_536496


namespace angle_MKO_perpendicular_l536_536618

noncomputable theory

-- Definitions for points on the plane
variables {A B O C D M K : Type}

-- The conditions of the problem
variables [EuclideanGeometry]
  (h_circle : circle O A B) -- Circle with diameter AB and center O
  (h_C_on_circle : point_on_circle C h_circle)  -- Point C on the circle.
  (h_D_on_circle : point_on_circle D h_circle) -- Point D on the circle.
  (h_CD_intersect_AB_at_M : line_intersect_line_at CD AB M)
  (h_M_conditions : M ∈ line_segment A B ∧ M ∈ line_segment C D)
  (h_MB_less_MA : distance M B < distance M A)
  (h_MD_less_MC : distance M D < distance M C)
  (h_K_intersection : K ∈ circumcircle_of_triangle A O C ∧ K ∈ circumcircle_of_triangle D O B ∧ K ≠ O)

-- The angle we need to prove is 90 degrees
theorem angle_MKO_perpendicular 
  : angle_between_lines MK OK = 90 := 
by 
  sorry

end angle_MKO_perpendicular_l536_536618


namespace max_distance_between_circle_centers_in_rectangle_l536_536174

theorem max_distance_between_circle_centers_in_rectangle :
  ∀ (length width diameter : ℝ),
    length = 15 ∧ width = 20 ∧ diameter = 10 →
    let radius := diameter / 2 in
    let center1 := (radius, radius) in
    let center2 := (width - radius, length - radius) in
    dist center1 center2 = 5 * Real.sqrt 5 :=
by
  sorry

end max_distance_between_circle_centers_in_rectangle_l536_536174


namespace dog_adult_weight_l536_536606

theorem dog_adult_weight 
  (w7 : ℕ) (w7_eq : w7 = 6)
  (w9 : ℕ) (w9_eq : w9 = 2 * w7)
  (w3m : ℕ) (w3m_eq : w3m = 2 * w9)
  (w5m : ℕ) (w5m_eq : w5m = 2 * w3m)
  (w1y : ℕ) (w1y_eq : w1y = w5m + 30) :
  w1y = 78 := by
  -- Proof is not required, so we leave it with sorry.
  sorry

end dog_adult_weight_l536_536606


namespace gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l536_536535

def a : ℕ := 2^1025 - 1
def b : ℕ := 2^1056 - 1
def answer : ℕ := 2147483647

theorem gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1 :
  Int.gcd a b = answer := by
  sorry

end gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l536_536535


namespace floor_sqrt_x_eq_8_has_17_values_l536_536936

theorem floor_sqrt_x_eq_8_has_17_values :
  {x : ℕ | 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9}.finite
  ∧ fintype.card {x : ℕ | 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9} = 17 :=
by {
  sorry,
}

end floor_sqrt_x_eq_8_has_17_values_l536_536936


namespace not_theorem_valid_non_perpendicular_axes_l536_536629

-- Define the properties of an equilateral triangle
def is_equilateral_triangle (A B C : Point) : Prop :=
  equilateral_triangle A B C ∧ (axes_of_symmetry A B C = 3) ∧ (angles_between_axes_of_symmetry A B C = 120)

-- Define the theorem validity given perpendicular axes of symmetry
def theorem_valid_perpendicular (T : Type) [metric_space T] : Prop :=
  ∃ (f : T → T), is_perpendicular f ∧ the_theorem_holds f

-- The Lean statement for the mathematically equivalent proof problem
theorem not_theorem_valid_non_perpendicular_axes (A B C : Point) (h : is_equilateral_triangle A B C) :
  ¬ theorem_valid_perpendicular (equilateral_triangle_space A B C) :=
sorry

end not_theorem_valid_non_perpendicular_axes_l536_536629


namespace problem_statement_l536_536293

theorem problem_statement (a b : ℝ) (h1 : a < 0) (h2 : -1 < b) (h3 : b < 0) : ab > ab^2 > a :=
by
  sorry

end problem_statement_l536_536293


namespace war_and_peace_completion_day_l536_536492

theorem war_and_peace_completion_day :
  ∀ (publication_date : ℕ) (days_before : ℕ) (day_of_week : ℕ),
  (publication_date = month_day_year.to_days (1865, 9, 4)) →
  (days_before = 1003) →
  (day_of_week = 2) →
  day_of_week_of_date (publication_date - days_before) = 6 := -- 6 signifies Saturday (0 = Sunday, 1 = Monday, ..., 6 = Saturday)
by
  sorry

end war_and_peace_completion_day_l536_536492


namespace count_whole_numbers_in_interval_l536_536844

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536844


namespace average_children_with_children_l536_536164

theorem average_children_with_children (total_families : ℕ) (average_children : ℚ) (childless_families : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  (total_families * average_children) / (total_families - childless_families) = 4.0 :=
by
  intros h1 h2 h3
  sorry

end average_children_with_children_l536_536164


namespace simplify_expression_l536_536452

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l536_536452


namespace accessories_per_doll_l536_536398

theorem accessories_per_doll (n dolls accessories time_per_doll time_per_accessory total_time : ℕ)
  (h0 : dolls = 12000)
  (h1 : time_per_doll = 45)
  (h2 : time_per_accessory = 10)
  (h3 : total_time = 1860000)
  (h4 : time_per_doll + accessories * time_per_accessory = n)
  (h5 : dolls * n = total_time) :
  accessories = 11 :=
by
  sorry

end accessories_per_doll_l536_536398


namespace number_of_pages_in_each_chapter_l536_536467

variable (x : ℕ)  -- Variable for number of pages in each chapter

-- Definitions based on the problem conditions
def pages_read_before_4_o_clock := 10 * x
def pages_read_at_4_o_clock := 20
def pages_read_after_4_o_clock := 2 * x
def total_pages_read := pages_read_before_4_o_clock x + pages_read_at_4_o_clock + pages_read_after_4_o_clock x

-- The theorem statement
theorem number_of_pages_in_each_chapter (h : total_pages_read x = 500) : x = 40 :=
sorry

end number_of_pages_in_each_chapter_l536_536467


namespace rightmost_three_digits_of_7_pow_1993_l536_536176

theorem rightmost_three_digits_of_7_pow_1993 :
  7^1993 % 1000 = 407 := 
sorry

end rightmost_three_digits_of_7_pow_1993_l536_536176


namespace solution_l536_536980

noncomputable section

open EuclideanGeometry

variables {A B C D E : Point}

def is_rhombus (A B C D : Point) : Prop :=
  is_parallelogram A B C D ∧ dist A B = dist B C

def midpoint (E D C : Point) : Prop :=
  dist D E = dist E C

def problem_statement (A B C D E : Point) (BAD_angle DEG_angle : Real) : Prop :=
  is_rhombus A B C D ∧
  BAD_angle = 60 ∧
  midpoint E D C ∧
  dist A B = 2

theorem solution (A B C D E : Point) (h : problem_statement A B C D E 60) :
  vector_dot (vector A E) (vector D B) = -1 :=
sorry

end solution_l536_536980


namespace craig_total_commission_correct_l536_536251

-- Define the commission structures
def refrigerator_commission (price : ℝ) : ℝ := 75 + 0.08 * price
def washing_machine_commission (price : ℝ) : ℝ := 50 + 0.10 * price
def oven_commission (price : ℝ) : ℝ := 60 + 0.12 * price

-- Define total sales
def total_refrigerator_sales : ℝ := 5280
def total_washing_machine_sales : ℝ := 2140
def total_oven_sales : ℝ := 4620

-- Define number of appliances sold
def number_of_refrigerators : ℝ := 3
def number_of_washing_machines : ℝ := 4
def number_of_ovens : ℝ := 5

-- Calculate total commissions for each appliance category
def total_refrigerator_commission : ℝ := number_of_refrigerators * refrigerator_commission total_refrigerator_sales
def total_washing_machine_commission : ℝ := number_of_washing_machines * washing_machine_commission total_washing_machine_sales
def total_oven_commission : ℝ := number_of_ovens * oven_commission total_oven_sales

-- Calculate total commission for the week
def total_commission : ℝ := total_refrigerator_commission + total_washing_machine_commission + total_oven_commission

-- Prove that the total commission is as expected
theorem craig_total_commission_correct : total_commission = 5620.20 := 
by
  sorry

end craig_total_commission_correct_l536_536251


namespace problem_l536_536615

def z : ℂ := 1 - complex.i

theorem problem : z^6 = 8 * complex.i := 
by
  -- Here would be the proof steps
  sorry

end problem_l536_536615


namespace coin_combination_l536_536786

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536786


namespace line_equation_l536_536494

theorem line_equation (a b : ℝ) 
  (h1 : -4 = (a + 0) / 2)
  (h2 : 6 = (0 + b) / 2) :
  (∀ x y : ℝ, y = (3 / 2) * (x + 4) → 3 * x - 2 * y + 24 = 0) :=
by
  sorry

end line_equation_l536_536494


namespace coefficient_of_x_cube_in_expansion_l536_536127

theorem coefficient_of_x_cube_in_expansion :
  let T_r := λ (r : ℕ), (Nat.choose 6 r : ℤ)
  let T_l := λ (l : ℕ), (-1)^l * (Nat.choose 4 l : ℤ)
  let coeff := ∑ k in {3 | 0 <= k ∧ k <= 3}, T_r k * T_l (3 - k) 
  coeff = -8 :=
by sorry

end coefficient_of_x_cube_in_expansion_l536_536127


namespace negation_of_proposition_l536_536146

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by sorry

end negation_of_proposition_l536_536146


namespace cheapest_store_for_60_balls_l536_536217

def cost_store_A (n : ℕ) (price_per_ball : ℕ) (free_per_10 : ℕ) : ℕ :=
  if n < 10 then n * price_per_ball
  else (n / 10) * 10 * price_per_ball + (n % 10) * price_per_ball * (n / (10 + free_per_10))

def cost_store_B (n : ℕ) (discount : ℕ) (price_per_ball : ℕ) : ℕ :=
  n * (price_per_ball - discount)

def cost_store_C (n : ℕ) (price_per_ball : ℕ) (cashback_threshold cashback_amt : ℕ) : ℕ :=
  let initial_cost := n * price_per_ball
  let cashback := (initial_cost / cashback_threshold) * cashback_amt
  initial_cost - cashback

theorem cheapest_store_for_60_balls
  (price_per_ball discount free_per_10 cashback_threshold cashback_amt : ℕ) :
  cost_store_A 60 price_per_ball free_per_10 = 1250 →
  cost_store_B 60 discount price_per_ball = 1200 →
  cost_store_C 60 price_per_ball cashback_threshold cashback_amt = 1290 →
  min (cost_store_A 60 price_per_ball free_per_10) (min (cost_store_B 60 discount price_per_ball) (cost_store_C 60 price_per_ball cashback_threshold cashback_amt))
  = 1200 :=
by
  sorry

end cheapest_store_for_60_balls_l536_536217


namespace coin_combination_l536_536782

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536782


namespace tangent_lines_from_point_to_circle_l536_536225

open Real

noncomputable def tangentLines (A : ℝ × ℝ) (C_center : ℝ × ℝ) (R : ℝ) : Prop :=
  -- Definitions
  let circle := λ x y, (x - C_center.1)^2 + (y - C_center.2)^2 = R^2
  let first_tangent := λ x, x = A.1
  let second_tangent := λ x y, 3 * x - 4 * y + 11 = 0
  -- Point and circle specifics
  A = (3, 5) ∧ 
  C_center = (2, 3) ∧ 
  R = 1 ∧ 
  -- Proof that these are the tangent lines
  (∃ k, (first_tangent k) ∨ (second_tangent k (4 * k - 3 * k + 11)))

theorem tangent_lines_from_point_to_circle : 
  tangentLines (3, 5) (2, 3) 1 :=
sorry

end tangent_lines_from_point_to_circle_l536_536225


namespace mean_median_mode_equal_l536_536642

theorem mean_median_mode_equal (x : ℕ) (h : x = (70 + 110 + x + 120 + x + 210 + x + 50 + 100) / 9 ∧ x = 110 ∧ x = mode [70, 110, x, 120, x, 210, x, 50, 100]) : x = 110 :=
sorry

end mean_median_mode_equal_l536_536642


namespace max_quotient_value_l536_536273

noncomputable def max_quotient_of_digits : ℚ :=
  sorry

theorem max_quotient_value :
  ∃ a b c d : ℕ, 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  max_quotient_of_digits = 329.2 :=
begin
  sorry
end

end max_quotient_value_l536_536273


namespace count_whole_numbers_in_interval_l536_536879

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536879


namespace largest_percentage_increase_l536_536603

theorem largest_percentage_increase :
  let students : ℕ → ℕ :=
    λ y, if y = 2010 then 80
         else if y = 2011 then 85
         else if y = 2012 then 88
         else if y = 2013 then 90
         else if y = 2014 then 95
         else if y = 2015 then 100
         else if y = 2016 then 120
         else 0
  let percentage_increase_from (y1 y2 : ℕ) : ℚ :=
    ((students y2 - students y1) : ℚ) / students y1 * 100 in
  percentage_increase_from 2015 2016 > percentage_increase_from 2010 2011 ∧
  percentage_increase_from 2015 2016 > percentage_increase_from 2011 2012 ∧
  percentage_increase_from 2015 2016 > percentage_increase_from 2012 2013 ∧
  percentage_increase_from 2015 2016 > percentage_increase_from 2013 2014 ∧
  percentage_increase_from 2015 2016 > percentage_increase_from 2014 2015 := 
by
  sorry

end largest_percentage_increase_l536_536603


namespace Mrs_Hilt_remaining_money_l536_536045

theorem Mrs_Hilt_remaining_money :
  let initial_amount : ℝ := 3.75
  let pencil_cost : ℝ := 1.15
  let eraser_cost : ℝ := 0.85
  let notebook_cost : ℝ := 2.25
  initial_amount - (pencil_cost + eraser_cost + notebook_cost) = -0.50 :=
by
  sorry

end Mrs_Hilt_remaining_money_l536_536045


namespace area_of_square_with_adjacent_points_l536_536078

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l536_536078


namespace area_enclosed_by_eqn_l536_536181

theorem area_enclosed_by_eqn : 
  (∃ E : set (ℝ × ℝ), ∀ (x y : ℝ), (x, y) ∈ E ↔ x^2 + y^2 = 2 * (abs x + abs y) ∧ ∫∫ (x, y) in E, 1 = 2 * π) :=
sorry

end area_enclosed_by_eqn_l536_536181


namespace sum_of_medians_is_32_l536_536965

def median (lst : List ℕ) : ℕ :=
  let sorted := lst.quickSort (≤)
  sorted.get (sorted.length / 2)

theorem sum_of_medians_is_32 :
  let scores_A := [7, 8, 9, 15, 17, 19, 23, 24, 26, 32, 41]
  let scores_B := [5, 7, 8, 11, 11, 13, 20, 22, 30, 31, 40]
  median scores_A + median scores_B = 32 :=
by 
  sorry

end sum_of_medians_is_32_l536_536965


namespace omega_decreasing_function_l536_536413

noncomputable def omega := Real

theorem omega_decreasing_function :
  ∀ (ω : omega),
  (0 < ω) → 
  (∀ x y : Real, (0 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3) → (2 * Real.cos (ω * x) > 2 * Real.cos (ω * y)))
  → (ω = 1 / 2) := by
  sorry

end omega_decreasing_function_l536_536413


namespace minimum_inequality_l536_536412

theorem minimum_inequality 
  (x_1 x_2 x_3 x_4 : ℝ) 
  (h1 : x_1 > 0) 
  (h2 : x_2 > 0) 
  (h3 : x_3 > 0) 
  (h4 : x_4 > 0) 
  (h_sum : x_1^2 + x_2^2 + x_3^2 + x_4^2 = 4) :
  (x_1 / (1 - x_1^2) + x_2 / (1 - x_2^2) + x_3 / (1 - x_3^2) + x_4 / (1 - x_4^2)) ≥ 6 * Real.sqrt 3 :=
by
  sorry

end minimum_inequality_l536_536412


namespace sam_morning_run_l536_536449

variable (X : ℝ)
variable (run_miles : ℝ) (walk_miles : ℝ) (bike_miles : ℝ) (total_miles : ℝ)

-- Conditions
def condition1 := walk_miles = 2 * run_miles
def condition2 := bike_miles = 12
def condition3 := total_miles = 18
def condition4 := run_miles + walk_miles + bike_miles = total_miles

-- Proof of the distance Sam ran in the morning
theorem sam_morning_run :
  (condition1 X run_miles walk_miles) →
  (condition2 bike_miles) →
  (condition3 total_miles) →
  (condition4 run_miles walk_miles bike_miles total_miles) →
  run_miles = 2 := by
  sorry

end sam_morning_run_l536_536449


namespace count_whole_numbers_in_interval_l536_536886

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536886


namespace UW_eq_RT_l536_536130

-- Define the points R, S, Q, T, A, U, W on a plane
variables (R S Q T A U W : Type) [Geometry R S Q T A U W]

-- Conditions of the problem
-- RSTQ is a trapezoid with bases RS and QT
def is_trapezoid (RSTQ : Type) [Geometry R S Q T] : Prop :=
∃(RS QT : Line), is_base R S T Q (RSQT)

-- Diagonals RT and SQ intersect at A at a right angle
def diagonals_right_angle (RT SQ : Line) (A : Point) : Prop :=
is_diagonal R T A ∧ is_diagonal S Q A ∧ angle RT SQ = 90°

-- RS is longer than QT
def base_comparison (RS QT : Line) : Prop :=
length RS > length QT

-- ∠R is a right angle
def right_angle_R (R : Point) : Prop :=
angle R = 90°

-- Angle bisector of ∠RAT intersects RT at U
def angle_bisector_RAT (R A T U : Type) [Angle RAT] : Prop :=
bisector RAT U

-- Line through U parallel to RS intersects SQ at W
def line_parallel_U_W (U W : Point) (RS : Line) : Prop :=
(∃ (parallel_line : Line), is_parallel parallel_line RS ∧ intersects parallel_line U ∧ intersects SQ W)

-- Definition of points R, S, Q, T, A, U, W
variables (R S Q T A U W : Point)

-- Assumptions for the proof
axiom H1 : is_trapezoid R S Q T
axiom H2 : diagonals_right_angle R T S Q A
axiom H3 : base_comparison R S Q T
axiom H4 : right_angle_R R
axiom H5 : angle_bisector_RAT R A T U
axiom H6 : line_parallel_U_W U W R S

-- The statement to be proved
theorem UW_eq_RT : length (U W) = length (R T) :=
sorry

end UW_eq_RT_l536_536130


namespace arctan_sum_zero_l536_536988
open Real

variable (a b c : ℝ)
variable (h : a^2 + b^2 = c^2)

theorem arctan_sum_zero (h : a^2 + b^2 = c^2) :
  arctan (a / (b + c)) + arctan (b / (a + c)) + arctan (c / (a + b)) = 0 := 
sorry

end arctan_sum_zero_l536_536988


namespace pieces_1994_impossible_pieces_1997_possible_l536_536222

def P (n : ℕ) : ℕ := 1 + 4 * n

theorem pieces_1994_impossible : ∀ n : ℕ, P n ≠ 1994 := 
by sorry

theorem pieces_1997_possible : ∃ n : ℕ, P n = 1997 := 
by sorry

end pieces_1994_impossible_pieces_1997_possible_l536_536222


namespace sum_primes_between_30_and_50_greater_than_35_l536_536540

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sums_between (a b c : ℕ) : ℕ :=
  (Finset.filter (λ x, is_prime x ∧ x > c) (Finset.Ico a b)).sum

theorem sum_primes_between_30_and_50_greater_than_35 :
  prime_sums_between 30 50 35 = 168 :=
by
  sorry

end sum_primes_between_30_and_50_greater_than_35_l536_536540


namespace coin_combinations_count_l536_536796

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536796


namespace find_number_set_l536_536643

noncomputable def num_set := {n : ℕ // n > 0}

variables (a b c d e : num_set)

theorem find_number_set
  (distinct_naturals : a.val ≠ b.val ∧ a.val ≠ c.val ∧ a.val ≠ d.val ∧ a.val ≠ e.val ∧ b.val ≠ c.val ∧ b.val ≠ d.val ∧ b.val ≠ e.val ∧ c.val ≠ d.val ∧ c.val ≠ e.val ∧ d.val ≠ e.val)
  (sorted : a.val < b.val ∧ b.val < c.val ∧ c.val < d.val ∧ d.val < e.val)
  (condition1 : a.val * b.val > 25)
  (condition2 : d.val * e.val < 75) :
  {a.val, b.val, c.val, d.val, e.val} = {5, 6, 7, 8, 9} :=
sorry

end find_number_set_l536_536643


namespace equilateral_triangle_l536_536148

variable {A B C : Type*} [Inhabited A] [Inhabited B] [Inhabited C]
variable (AB : ℝ) (BC : ℝ) (CA : ℝ)
variable {M : Type*} [Inhabited M]

-- Hypotheses
def is_centroid (T : ℝ → ℝ → ℝ → Prop) (M T : ℝ) := sorry
def equal_perimeters (T : ℝ → ℝ → ℝ → Prop) (P_abm P_bcm P_cam : ℝ) := sorry

-- Theorem
theorem equilateral_triangle
  (M : ℝ)
  (h1 : is_centroid M (triangle ABC))
  (h2 : equal_perimeters (triangle ABM) (triangle BCM) (triangle CAM)) :
  triangle ABC is_equilateral :=
  sorry

end equilateral_triangle_l536_536148


namespace find_m_l536_536042

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

end find_m_l536_536042


namespace count_whole_numbers_in_interval_l536_536891

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536891


namespace line_parallel_plane_l536_536647

-- Definitions for planes and lines
variable (Plane : Type) (Line : Type)

-- Definitions for planes α and β, and lines a and b
variable (α β : Plane) (a b : Line)

-- Definitions for the given conditions
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Definitions of the conditions for the problem statement
variable (H1 : parallel α β)
variable (H2 : subset a β)

-- Final theorem statement
theorem line_parallel_plane (α β : Plane) (a : Line) (H1 : parallel α β) (H2 : subset a β) : parallel a α := 
sorry

end line_parallel_plane_l536_536647


namespace batting_average_excl_highest_lowest_l536_536125

theorem batting_average_excl_highest_lowest (
    avg : ℕ,
    innings : ℕ,
    highest : ℕ,
    lowest : ℕ,
    diff : ℕ,
    T : ℕ,
    T_excl : ℕ,
    I_excl : ℕ,
    avg_excl : ℝ) 
    (h1 : avg = 58)
    (h2 : innings = 46)
    (h3 : highest = 133)
    (h4 : lowest = 0)
    (h5 : diff = 150)
    (h6 : T = avg * innings)
    (h7 : T_excl = T - (highest + lowest))
    (h8 : I_excl = innings - 2)
    (h9 : avg_excl = T_excl / I_excl.to_float) :

    avg_excl ≈ 57.61 :=
by sorry

end batting_average_excl_highest_lowest_l536_536125


namespace square_area_l536_536070

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (1, 2) ∧ p2 = (4, 6)) :
  let d := distance p1 p2 in
  d^2 = 25 :=
by
  sorry

end square_area_l536_536070


namespace trapezoid_area_sum_l536_536228

theorem trapezoid_area_sum (a b c d : ℝ) 
  (h1 : a = 4) (h2 : b = 6) (h3 : c = 8) (h4 : d = 10)
  (h_bases1 : ((a, d), (b, c)) = ((4, 10), (6, 8)))
  (h_bases2 : ((b, c), (a, d)) = ((6, 8), (4, 10))) :
  let h1_area := (7 * (3 * Real.sqrt 15) / 2)
      h2_area := (7 * (12 * Real.sqrt 6))  in
  h1_area + h2_area = (21 * Real.sqrt 15) / 2 + 84 * Real.sqrt 6 := 
sorry

end trapezoid_area_sum_l536_536228


namespace window_area_ratio_l536_536213

variables (AD AB : ℝ) (π : ℝ := Real.pi)
noncomputable def radius (ab: ℝ) := ab / 2
noncomputable def area_semicircles (r: ℝ) := π * r^2
noncomputable def area_rectangle (ad ab: ℝ) := ad * ab

theorem window_area_ratio
  (h1 : AD / AB = 3 / 2)
  (h2 : AB = 20) :
  area_rectangle AD AB / area_semicircles (radius AB) = 6 / π :=
by
  sorry

end window_area_ratio_l536_536213


namespace sequence_sum_1234_l536_536384

noncomputable def sequence_sum : ℕ → ℕ
| 0 := 1
| n := if n = 0 then 1 else (n / (n + 1) + 1)

theorem sequence_sum_1234 : (finset.range 1234).sum sequence_sum = 2419 := by
  sorry

end sequence_sum_1234_l536_536384


namespace square_area_l536_536073

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (1, 2) ∧ p2 = (4, 6)) :
  let d := distance p1 p2 in
  d^2 = 25 :=
by
  sorry

end square_area_l536_536073


namespace circle_tangent_l536_536951

theorem circle_tangent (t : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 4 → (x - t)^2 + y^2 = 1 → |t| = 3) :=
by
  sorry

end circle_tangent_l536_536951


namespace tiling_impossible_l536_536532

theorem tiling_impossible (w b : ℕ) (total_squares : ℕ) : 
  (w = 30) → (b = 32) → (total_squares = 62) → 
  ¬ ∃ (domino_pairing : ℕ → ℕ → ℕ), total_squares % 2 = 1 := 
by {
  intros,
  -- declare that each domino covers one white and one black square
  have dominos_cover : ∀ n, domino_pairing = ∃ (pair : ℕ), pair = w ∧ pair = b,
  sorry,
}

end tiling_impossible_l536_536532


namespace combinations_of_coins_with_50_cents_l536_536749

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536749


namespace does_not_pass_through_quadrant_I_l536_536504

def quadrant (x y : ℝ) : Option (Fin 4) :=
  match x, y with
  | x, y if x > 0 ∧ y > 0 => some 0 -- Quadrant I
  | x, y if x < 0 ∧ y > 0 => some 1 -- Quadrant II
  | x, y if x < 0 ∧ y < 0 => some 2 -- Quadrant III
  | x, y if x > 0 ∧ y < 0 => some 3 -- Quadrant IV
  | _, _ => none

theorem does_not_pass_through_quadrant_I :
  ∀ x : ℝ, quadrant x (-2 * x - 1) ≠ some 0 :=
by
  intros
  sorry

end does_not_pass_through_quadrant_I_l536_536504


namespace count_whole_numbers_in_interval_l536_536903

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536903


namespace five_lines_seven_intersections_exists_l536_536264

noncomputable def line : Type := ℝ → ℝ → Prop

def intersection_points (lines : list line) : set (ℝ × ℝ) :=
  { p | ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ l1 (fst p) (snd p) ∧ l2 (fst p) (snd p) }

def five_distinct_lines (lines : list line) : Prop :=
  ∃ l1 l2 l3 l4 l5, 
    lines = [l1, l2, l3, l4, l5] ∧ 
    l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l1 ≠ l5 ∧
    l2 ≠ l3 ∧ l2 ≠ l4 ∧ l2 ≠ l5 ∧
    l3 ≠ l4 ∧ l3 ≠ l5 ∧
    l4 ≠ l5

def exactly_seven_intersections (lines : list line) : Prop :=
  finset.card (intersection_points lines) = 7

theorem five_lines_seven_intersections_exists :
  ∃ lines : list line, five_distinct_lines lines ∧ exactly_seven_intersections lines :=
sorry

end five_lines_seven_intersections_exists_l536_536264


namespace disproves_statement_l536_536281

def is_consonant (c : Char) : Prop :=
  c = 'B'

def is_vowel (c : Char) : Prop :=
  c = 'A'

def is_odd (n : Nat) : Prop :=
  n % 2 = 1

def is_even (n : Nat) : Prop :=
  n % 2 = 0

noncomputable def card_front : Char :=
  '8'

noncomputable def card_back : Char :=
  'B'

theorem disproves_statement : card_front = '8' ∧ card_back = 'B' → (is_even (card_front.to_nat) ∧ is_consonant card_back) :=
by {
  unfold card_front card_back is_even is_consonant,
  sorry
}

end disproves_statement_l536_536281


namespace find_speed_l536_536995

-- Definitions corresponding to conditions
def JacksSpeed (x : ℝ) : ℝ := x^2 - 7 * x - 12
def JillsDistance (x : ℝ) : ℝ := x^2 - 3 * x - 10
def JillsTime (x : ℝ) : ℝ := x + 2

-- Theorem statement
theorem find_speed (x : ℝ) (hx : x ≠ -2) (h_speed_eq : JacksSpeed x = (JillsDistance x) / (JillsTime x)) : JacksSpeed x = 2 :=
by
  sorry

end find_speed_l536_536995


namespace g_inv_solution_l536_536013

def g (x : ℝ) : ℝ := x * abs (x) ^ 3

noncomputable def g_inv : ℝ → ℝ 
| y := if y > 0 then (y ^ (1 / 3)) else if y < 0 then (- ((-y) ^ (1 / 3))) else 0

theorem g_inv_solution :
  g_inv 8 + g_inv (-64) = -2 :=
by
  -- Statement of the proof problem; the proof itself is omitted.
  sorry

end g_inv_solution_l536_536013


namespace coin_combinations_sum_50_l536_536714

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536714


namespace avg_children_nine_families_l536_536166

theorem avg_children_nine_families
  (total_families : ℕ)
  (average_children : ℕ)
  (childless_families : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  total_children = total_families * average_children →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℝ) = 4.0 :=
begin
  intros,
  sorry
end

end avg_children_nine_families_l536_536166


namespace solve_inequality_l536_536139

theorem solve_inequality (k a b c : ℝ) :
  (∀ x : ℝ, (x ∈ (-2, -1) ∨ x ∈ (2, 3)) ↔ (k*x/(a*x-1) + (b*x-1)/(c*x-1) < 0)) →
  (∀ x : ℝ, (x ∈ (-1/2, -1/3) ∨ x ∈ (1/2, 1)) ↔ (k/(x + a) + (x + b)/(x + c) < 0)) :=
by
sor__()


end solve_inequality_l536_536139


namespace P_le_Q_l536_536670

variable (a b c d m n : ℝ)

-- Assume all variables are positive
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom pos_d : 0 < d
axiom pos_m : 0 < m
axiom pos_n : 0 < n

def P := Real.sqrt (a * b) + Real.sqrt (c * d)
def Q := Real.sqrt (m * a + n * c) * Real.sqrt (b / m + d / n)

theorem P_le_Q : P a b c d m n ≤ Q a b c d m n := by
  sorry

end P_le_Q_l536_536670


namespace combinations_of_coins_l536_536760

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536760


namespace working_days_l536_536548

/-- 
  Given a month with 30 days starting on a Saturday, where every second Saturday 
  and all Sundays are holidays, there are 23 working days in that month.
-/
theorem working_days (days : Fin 30) (starts_on_sat : days 0 = "Saturday")
    (holiday_sats : ∀ (d : Fin 30), d ∈ [8, 22]) 
    (holiday_suns : ∀ (d : Fin 30), d % 7 = 1) : 
    23 := 
sorry

end working_days_l536_536548


namespace whole_numbers_in_interval_l536_536910

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536910


namespace product_of_two_numbers_eq_a_mul_100_a_l536_536361

def product_of_two_numbers (a : ℝ) (b : ℝ) : ℝ := a * b

theorem product_of_two_numbers_eq_a_mul_100_a (a : ℝ) (b : ℝ) (h : a + b = 100) :
    product_of_two_numbers a b = a * (100 - a) :=
by
  sorry

end product_of_two_numbers_eq_a_mul_100_a_l536_536361


namespace children_sit_in_same_row_twice_l536_536968

theorem children_sit_in_same_row_twice
  (rows : ℕ) (seats_per_row : ℕ) (children : ℕ)
  (h_rows : rows = 7) (h_seats_per_row : seats_per_row = 10) (h_children : children = 50) :
  ∃ (morning_evening_pair : ℕ × ℕ), 
  (morning_evening_pair.1 < rows ∧ morning_evening_pair.2 < rows) ∧ 
  morning_evening_pair.1 = morning_evening_pair.2 :=
by
  sorry

end children_sit_in_same_row_twice_l536_536968


namespace rectangle_area_eq_l536_536554

variables (a b : ℝ) (A B C D M : Point) (𝓡₁ 𝓡₂ : Circle)
  (hA : IsRectangle A B C D)
  (hM : IsPointOnLineSegment M B C)
  (hInscribed1 : IsInscribedCircle 𝓡₁ [A, M, C, D])
  (hRadius1 : radius 𝓡₁ = a)
  (hInscribed2 : IsInscribedCircle 𝓡₂ [A, B, M])
  (hRadius2 : radius 𝓡₂ = b)

theorem rectangle_area_eq : 
  area_rectangle A B C D = (4 * a^3 - 2 * a^2 * b) / (2 * a - b) :=
sorry

end rectangle_area_eq_l536_536554


namespace coprime_with_others_in_seq_l536_536103

open Nat

theorem coprime_with_others_in_seq (n : ℤ) : 
  ∃ x ∈ (list.map (λ i, n + i) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
    ∀ y ∈ (list.map (λ i, n + i) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), y ≠ x → gcd x y = 1 :=
by
  sorry

end coprime_with_others_in_seq_l536_536103


namespace arithmetic_sequence_sum_eq_105_l536_536982

variable {α : Type} [AddGroup α] [AddCommGroup α] [Module ℚ α] [LinearIndependent ℚ α]
variable {a : ℕ → α}
variable (h₁ : a 6 + a 9 + a 13 + a 16 = (20 : α))

theorem arithmetic_sequence_sum_eq_105 : 
  S 21 = 105 :=
by
  sorry

end arithmetic_sequence_sum_eq_105_l536_536982


namespace integer_solution_count_in_inequality_l536_536280

theorem integer_solution_count_in_inequality (f : ℤ → ℝ)
  (h₁ : ∀ x : ℤ, f x = sqrt (3 * cos (π * x / 2) + cos (π * x / 4) + 1) + sqrt 6 * cos (π * x / 4))
  (h₂ : ∀ x : ℤ, 1991 ≤ x ∧ x ≤ 2013) :
  (∑ x in finset.range 2013, if f x ≥ 0 then 1 else 0) = 9 :=
sorry

end integer_solution_count_in_inequality_l536_536280


namespace log_one_eq_zero_l536_536559

theorem log_one_eq_zero : Real.log 1 = 0 := 
by
  sorry

end log_one_eq_zero_l536_536559


namespace AD_perp_IP_l536_536404

-- Definitions of given geometric entities and conditions
variables {A B C I D L M P : Type}
variables (triangle_ABC : Triangle A B C) (incenter : Center I)
variables (tangencyD : Tangent D (Segment B C))
variables (tangencyL : Tangent L (Segment A C))
variables (tangencyM : Tangent M (Segment A B))
variables (intersection_P : Meet (Line (Segment M L)) (Line (Segment B C)) P)
variables (perpendicular_AD_IP : Perpendicular (Line (Segment A D)) (Line (Segment I P)))

-- The theorem to be proven
theorem AD_perp_IP :
  AD ⟂ IP :=
sorry

end AD_perp_IP_l536_536404


namespace initial_girls_count_l536_536288

-- Define the variables
variables (b g : ℕ)

-- Conditions
def condition1 := b = 3 * (g - 20)
def condition2 := 4 * (b - 60) = g - 20

-- Statement of the problem
theorem initial_girls_count
  (h1 : condition1 b g)
  (h2 : condition2 b g) : g = 460 / 11 := 
sorry

end initial_girls_count_l536_536288


namespace f_neg_five_l536_536675

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then x^3 - 2 * x^2 else f (x - 2)

axiom f_is_odd : ∀ x : ℝ, f (-x) = -f x

theorem f_neg_five : f (-5) = 1 :=
by
  sorry

end f_neg_five_l536_536675


namespace cos_B_value_cos_2B_minus_pi_over_6_value_l536_536961

/-
  In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c, respectively.
  Given that a² - b² = bc and sin B = 2 sin C, prove:
  1. cos B = sqrt(6) / 4
  2. cos (2B - pi / 6) = (sqrt(15) - sqrt(3)) / 8
-/

variables {A B C : ℝ} -- angles of triangle ABC
variables {a b c : ℝ} -- sides opposite to angles A, B, and C respectively
variables (h1 : a^2 - b^2 = b * c) (h2 : sin B = 2 * sin C)

-- Statement 1: cos B = sqrt(6) / 4
theorem cos_B_value : cos B = sqrt(6) / 4 :=
by
  sorry

-- Statement 2: cos (2B - pi / 6) = (sqrt(15) - sqrt(3)) / 8
theorem cos_2B_minus_pi_over_6_value : cos (2 * B - π / 6) = (sqrt(15) - sqrt(3)) / 8 :=
by
  sorry

end cos_B_value_cos_2B_minus_pi_over_6_value_l536_536961


namespace baseball_cost_correct_l536_536110

def cost_of_football := 9.14
def total_amount_paid := 20.0
def change_returned := 4.05

def cost_of_baseball : ℝ := total_amount_paid - change_returned - cost_of_football

theorem baseball_cost_correct : cost_of_baseball = 6.81 := by
  unfold cost_of_baseball
  norm_num
  sorry

end baseball_cost_correct_l536_536110


namespace determine_g_l536_536253

theorem determine_g (g : ℝ → ℝ) : (∀ x : ℝ, 4 * x^4 + x^3 - 2 * x + 5 + g x = 2 * x^3 - 7 * x^2 + 4) →
  (∀ x : ℝ, g x = -4 * x^4 + x^3 - 7 * x^2 + 2 * x - 1) :=
by
  intro h
  sorry

end determine_g_l536_536253


namespace count_whole_numbers_in_interval_l536_536887

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536887


namespace A_on_curve_slope_at_A_l536_536321

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x ^ 2

-- Define the point A on the curve f
def A : ℝ × ℝ := (2, 8)

-- Define the condition that A is on the curve f
theorem A_on_curve : A.2 = f A.1 := by
  -- * left as a proof placeholder
  sorry

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

-- State and prove the main theorem
theorem slope_at_A : (deriv f) 2 = 8 := by
  -- * left as a proof placeholder
  sorry

end A_on_curve_slope_at_A_l536_536321


namespace distance_example_l536_536692

noncomputable def C1_intersection_line : ℝ × ℝ :=
  let x := Real.cos (π / 6)
  let y := 1 + Real.sin (π / 6)
  let ρ := Real.sqrt (x^2 + y^2)
  (ρ, π / 6)

noncomputable def C2_intersection_line : ℝ × ℝ :=
  let θ := π / 6
  let ρ := 4 * Real.sin (θ + π / 3)
  (ρ, θ)

noncomputable def distance_AB (A B : ℝ × ℝ) : ℝ :=
  Real.abs (A.1 - B.1)

theorem distance_example :
  let A := C1_intersection_line
  let B := C2_intersection_line
  distance_AB A B = 3 :=
by
  -- The proof is omitted.
  sorry

end distance_example_l536_536692


namespace count_whole_numbers_in_interval_l536_536900

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536900


namespace combinations_of_coins_l536_536762

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536762


namespace trigonometric_value_l536_536318

variable (α : ℝ)

-- Conditions
def vertex_is_origin : Prop := true
def initial_side_positive_x_axis : Prop := true
def terminal_side_on_line : Prop := tan α = 2

-- Problem statement
theorem trigonometric_value : terminal_side_on_line α → 
  (sin α + cos α) / (sin α - cos α) = 3 :=
by
  sorry

end trigonometric_value_l536_536318


namespace coin_combinations_count_l536_536739

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536739


namespace combination_coins_l536_536817

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536817


namespace frood_points_smallest_frood_points_l536_536379

theorem frood_points (n : ℕ) (h : n > 9) : (n * (n + 1)) / 2 > 5 * n :=
by {
  sorry
}

noncomputable def smallest_n : ℕ := 10

theorem smallest_frood_points (m : ℕ) (h : (m * (m + 1)) / 2 > 5 * m) : 10 ≤ m :=
by {
  sorry
}

end frood_points_smallest_frood_points_l536_536379


namespace first_investment_percentage_l536_536441

theorem first_investment_percentage :
  let total_inheritance := 4000
  let invested_6_5 := 1800
  let interest_rate_6_5 := 0.065
  let total_interest := 227
  let remaining_investment := total_inheritance - invested_6_5
  let interest_from_6_5 := invested_6_5 * interest_rate_6_5
  let interest_from_remaining := total_interest - interest_from_6_5
  let P := interest_from_remaining / remaining_investment
  P = 0.05 :=
by 
  sorry

end first_investment_percentage_l536_536441


namespace table_tennis_teams_equation_l536_536976

-- Variables
variable (x : ℕ)

-- Conditions
def total_matches : ℕ := 28
def teams_playing_equation : Prop := x * (x - 1) = 28 * 2

-- Theorem Statement
theorem table_tennis_teams_equation : teams_playing_equation x :=
sorry

end table_tennis_teams_equation_l536_536976


namespace inequality_solution_l536_536464

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) :=
by
  sorry

end inequality_solution_l536_536464


namespace sam_morning_run_distance_l536_536448

variable (x : ℝ) -- The distance of Sam's morning run in miles

theorem sam_morning_run_distance (h1 : ∀ y, y = 2 * x) (h2 : 12 = 12) (h3 : x + 2 * x + 12 = 18) : x = 2 :=
by sorry

end sam_morning_run_distance_l536_536448


namespace planned_daily_catch_l536_536591

theorem planned_daily_catch (x y : ℝ) 
  (h1 : x * y = 1800)
  (h2 : (x / 3) * (y - 20) + ((2 * x / 3) - 1) * (y + 20) = 1800) :
  y = 100 :=
by
  sorry

end planned_daily_catch_l536_536591


namespace count_whole_numbers_in_interval_l536_536853

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536853


namespace count_whole_numbers_in_interval_l536_536885

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536885


namespace average_children_with_children_l536_536163

theorem average_children_with_children (total_families : ℕ) (average_children : ℚ) (childless_families : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  (total_families * average_children) / (total_families - childless_families) = 4.0 :=
by
  intros h1 h2 h3
  sorry

end average_children_with_children_l536_536163


namespace ratio_of_areas_l536_536486

variables {A B C D M O1 O2 O3 O4 : Type}
variables [measure_space A] [measure_space B] [measure_space C] [measure_space D]
variables [measure_space M] [measure_space O1] [measure_space O2] [measure_space O3] [measure_space O4]

noncomputable def area (x : Type) [measure_space x] : ℝ := sorry 

variables (α : ℝ) (hM : ∃ M, M = (AC ∩ BD)) (hα : ∀ M, ∠M = α)
variables (O1 O2 O3 O4 : Type)
variables (hO1 : O1 = center_circle ABM) (hO2 : O2 = center_circle BCM)
variables (hO3 : O3 = center_circle CDM) (hO4 : O4 = center_circle DAM)

theorem ratio_of_areas (h_cond : (hM, hα, hO1, hO2, hO3, hO4)) : 
  (area O1O2O3O4) = (2 * (sin α)^2) * (area ABCD) := 
sorry

end ratio_of_areas_l536_536486


namespace area_of_square_l536_536059

-- We define the points as given in the conditions
def point1 : ℝ × ℝ := (1, 2)
def point2 : ℝ × ℝ := (4, 6)

-- Lean's "def" defines the concept of a square given two adjacent points.
def is_square (p1 p2: ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  in ∃ (l : ℝ), l = d ∧ (l^2 = 25)

-- The theorem assumes the points are adjacent points on a square and proves that their area is 25.
theorem area_of_square :
  is_square point1 point2 :=
by
  -- Insert formal proof here, skipped with 'sorry' for this task
  sorry

end area_of_square_l536_536059


namespace original_days_to_finish_work_l536_536230

theorem original_days_to_finish_work : 
  ∀ (D : ℕ), 
  (∃ (W : ℕ), 15 * D * W = 25 * (D - 3) * W) → 
  D = 8 :=
by
  intros D h
  sorry

end original_days_to_finish_work_l536_536230


namespace gilda_marbles_percentage_l536_536290

theorem gilda_marbles_percentage (M : ℝ) (hM : 0 ≤ M) :
  let M_after_Pedro := 0.70 * M in
  let M_after_Ebony := 0.90 * M_after_Pedro in
  let M_after_Lisa := 0.60 * M_after_Ebony in
  M_after_Lisa / M = 0.378 :=
begin
  sorry
end

end gilda_marbles_percentage_l536_536290


namespace tangent_XYZ_l536_536989

-- Given a triangle XYZ with the following conditions:
variables (X Y Z : Type) [InnerProductSpace ℝ X] 

-- Angle Y is 90 degrees.
def angle_Y_90 (a b c : X) : Prop := inner_product_space.angle a b c = 90

-- YZ = 4
def YZ_length_four (a b : X) : Prop := dist a b = 4

-- XZ = √20
def XZ_length_root_20 (a b : X) : Prop := dist a b = real.sqrt 20

-- The goal is to prove that tan X = 2
theorem tangent_XYZ
  (a b c : X) 
  (hy: angle_Y_90 a b c)
  (hyz: YZ_length_four b c)
  (hxz: XZ_length_root_20 a c) : 
  real.tan (inner_product_space.angle a c b) = 2 :=
sorry

end tangent_XYZ_l536_536989


namespace problem1_proof_problem2_proof_l536_536610

-- Definition of the first problem
def problem1_expression := (Real.sqrt 45 + Real.sqrt 50) - (Real.sqrt 18 - Real.sqrt 20)
def problem1_expected := 5 * Real.sqrt 5 + 2 * Real.sqrt 2

-- First proof problem statement
theorem problem1_proof : problem1_expression = problem1_expected := by
  sorry

-- Definition of the second problem
def problem2_expression := Real.sqrt 24 / (6 * Real.sqrt (1 / 6)) - Real.sqrt 12 * (Real.sqrt 3 / 2)
def problem2_expected := -1

-- Second proof problem statement
theorem problem2_proof : problem2_expression = problem2_expected := by
  sorry

end problem1_proof_problem2_proof_l536_536610


namespace range_of_a_is_l536_536334

def bounds_of_a : set ℝ :=
  {a : ℝ | ∀ A : set ℝ, (A = {x : ℝ | x^2 - a *x - a - 1 > 0}) → 
    (∃ z : ℝ, z ∈ (compl A : set ℝ) ∩ (set.has_mem.mem : set ℝ → set ℝ) ℤ → 
      (∀ y ∈ (compl A : set ℝ) ∩ (set.has_mem.mem : set ℝ → set ℝ) ℤ, y = z))}

theorem range_of_a_is :
  bounds_of_a = {a : ℝ | -3 < a ∧ a < -1} :=
sorry

end range_of_a_is_l536_536334


namespace min_value_inv_sum_l536_536025

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end min_value_inv_sum_l536_536025


namespace average_children_with_children_l536_536165

theorem average_children_with_children (total_families : ℕ) (average_children : ℚ) (childless_families : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  (total_families * average_children) / (total_families - childless_families) = 4.0 :=
by
  intros h1 h2 h3
  sorry

end average_children_with_children_l536_536165


namespace solve_system_of_equations_l536_536115

theorem solve_system_of_equations
  {a b c d x y z : ℝ}
  (h1 : x + y + z = 1)
  (h2 : a * x + b * y + c * z = d)
  (h3 : a^2 * x + b^2 * y + c^2 * z = d^2)
  (hne1 : a ≠ b)
  (hne2 : a ≠ c)
  (hne3 : b ≠ c) :
  x = (d - b) * (d - c) / ((a - b) * (a - c)) ∧
  y = (d - a) * (d - c) / ((b - a) * (b - c)) ∧
  z = (d - a) * (d - b) / ((c - a) * (c - b)) :=
sorry

end solve_system_of_equations_l536_536115


namespace kayla_less_than_vika_l536_536109

variable (S K V : ℕ)
variable (h1 : S = 216)
variable (h2 : S = 4 * K)
variable (h3 : V = 84)

theorem kayla_less_than_vika (S K V : ℕ) (h1 : S = 216) (h2 : S = 4 * K) (h3 : V = 84) : V - K = 30 :=
by
  sorry

end kayla_less_than_vika_l536_536109


namespace coin_combinations_l536_536801

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536801


namespace coin_combinations_count_l536_536742

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536742


namespace combination_coins_l536_536814

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536814


namespace number_of_integers_in_interval_l536_536921

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536921


namespace incorrect_statements_exactly_one_l536_536660

-- From given condition \(x^2 + y^2 - 4x + 1 = 0\)
def condition (x y : ℝ) := x^2 + y^2 - 4 * x + 1 = 0

-- Prove that there is exactly one incorrect statement among the four given
theorem incorrect_statements_exactly_one (x y : ℝ) (h : condition x y) :
  (statement_1 h = true ∧ statement_2 h = true ∧ statement_3 h = false ∧ statement_4 h = true) → 
  ∃! p, (p = 1) :=
by
  -- skipping the proof
  sorry

-- Definitions of each statement 
def statement_1 (h : condition x y) := y - x ≤ sqrt(6) - 2
def statement_2 (h : condition x y) := x^2 + y^2 ≤ 7 + 4 * sqrt(3)
def statement_3 (h : condition x y) := y / x ≤ sqrt(3) / 2
def statement_4 (h : condition x y) := x + y ≤ 2 + sqrt(3)

end incorrect_statements_exactly_one_l536_536660


namespace count_whole_numbers_in_interval_l536_536854

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536854


namespace sally_quarters_after_purchases_l536_536446

-- Given conditions
def initial_quarters : ℕ := 760
def first_purchase : ℕ := 418
def second_purchase : ℕ := 192

-- Define the resulting quarters after purchases
def quarters_after_first_purchase (initial : ℕ) (spent : ℕ) : ℕ := initial - spent
def quarters_after_second_purchase (remaining : ℕ) (spent : ℕ) : ℕ := remaining - spent

-- The main statement to be proved
theorem sally_quarters_after_purchases :
  quarters_after_second_purchase (quarters_after_first_purchase initial_quarters first_purchase) second_purchase = 150 :=
by
  unfold quarters_after_first_purchase quarters_after_second_purchase initial_quarters first_purchase second_purchase
  simp
  sorry

end sally_quarters_after_purchases_l536_536446


namespace regularSixteenGon_symmetries_add_angle_l536_536219

def isRegularSixteenGon (P : Type) [polygon P (16 : ℕ)] :=
  ∀ (x : P), true

theorem regularSixteenGon_symmetries_add_angle
  (P : Type) [polygon P (16 : ℕ)] (h : isRegularSixteenGon P) :
  let L := 16
  let R := 22.5
  L + R = 38.5 :=
by
  sorry

end regularSixteenGon_symmetries_add_angle_l536_536219


namespace game_completion_days_l536_536998

theorem game_completion_days (initial_playtime hours_per_day : ℕ) (initial_days : ℕ) (completion_percentage : ℚ) (increased_playtime : ℕ) (remaining_days : ℕ) :
  initial_playtime = 4 →
  hours_per_day = 2 * 7 →
  completion_percentage = 0.4 →
  increased_playtime = 7 →
  ((initial_playtime * hours_per_day) / completion_percentage) - (initial_playtime * hours_per_day) = increased_playtime * remaining_days →
  remaining_days = 12 :=
by
  intros
  sorry

end game_completion_days_l536_536998


namespace problem_1_problem_2_l536_536330

def f (x : ℝ) (a : ℝ) : ℝ := |x + 2| - |x + a|

theorem problem_1 (a : ℝ) (h : a = 3) :
  ∀ x, f x a ≤ 1/2 → x ≥ -11/4 := sorry

theorem problem_2 (a : ℝ) :
  (∀ x, f x a ≤ a) → a ≥ 1 := sorry

end problem_1_problem_2_l536_536330


namespace Smallest_number_ending_in_4_l536_536538

theorem Smallest_number_ending_in_4 :
  ∃ x: ℕ, (x % 10 = 4) ∧ (4 * x = 4 * 10^(nat.log10 x + 1) + x / 10) ∧ (x = 102564) := 
sorry

end Smallest_number_ending_in_4_l536_536538


namespace combinations_of_coins_l536_536766

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536766


namespace water_pressure_on_dam_l536_536240

theorem water_pressure_on_dam :
  let a := 10 -- length of upper base in meters
  let b := 20 -- length of lower base in meters
  let h := 3 -- height in meters
  let ρg := 9810 -- natural constant for water pressure in N/m^3
  let P := ρg * ((a + 2 * b) * h^2 / 6)
  P = 735750 :=
by
  sorry

end water_pressure_on_dam_l536_536240


namespace simplify_exponents_of_variables_outside_radical_l536_536189

-- Given conditions
variables {a b c : ℝ} (h : 0 < 120 * (a^6) * (b^8) * (c^{17}))

-- The problem statement
theorem simplify_exponents_of_variables_outside_radical (h : 0 < 120 * (a^6) * (b^8) * (c^{17})) :
    let radicand := 120 * (a^6) * (b^8) * (c^{17}),
        root := real.rpow radicand (1/4),
        outside_root_exponents_sum := 1 + 4 
    in outside_root_exponents_sum = 5 := 
sorry

end simplify_exponents_of_variables_outside_radical_l536_536189


namespace count_whole_numbers_in_interval_l536_536869

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536869


namespace common_ratio_geom_arith_prog_l536_536482

theorem common_ratio_geom_arith_prog (a b c q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = a * q^2)
  (h3 : 2 * (2020 * b / 7) = 577 * a + c / 7) : 
  q = 4039 :=
begin
  -- proof to be filled
  sorry
end

end common_ratio_geom_arith_prog_l536_536482


namespace projection_of_vector1_on_normalized_vector2_is_correct_l536_536639

def vector1 : ℝ × ℝ := (3, 2)
def vector2 : ℝ × ℝ := (4, -1)

def norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def normalize (v : ℝ × ℝ) : ℝ × ℝ :=
  let n := norm v
  (v.1 / n, v.2 / n)

def projection (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := (v1.1 * v2.1 + v1.2 * v2.2) / (v2.1 ^ 2 + v2.2 ^ 2)
  (dot_product * v2.1, dot_product * v2.2)

theorem projection_of_vector1_on_normalized_vector2_is_correct :
  projection vector1 (normalize vector2) = (40 / 17, -10 / 17) :=
by
  sorry

end projection_of_vector1_on_normalized_vector2_is_correct_l536_536639


namespace number_of_solutions_is_two_l536_536472

-- Define the absolute value and the equation
def equation (x : ℚ) : Prop := abs (x - abs (2 * x + 1)) = 3

-- Prove that the number of distinct solutions to the equation is 2
theorem number_of_solutions_is_two : 
  (finset.filter equation (finset.Icc (-10:ℚ) 10: ℚ)).card = 2 := 
by 
  sorry

end number_of_solutions_is_two_l536_536472


namespace length_of_shorter_side_l536_536218

/-- 
A rectangular plot measuring L meters by 50 meters is to be enclosed by wire fencing. 
If the poles of the fence are kept 5 meters apart, 26 poles will be needed.
What is the length of the shorter side of the rectangular plot?
-/
theorem length_of_shorter_side
(L: ℝ) 
(h1: ∃ L: ℝ, L > 0) -- There's some positive length for the side L
(h2: ∀ distance: ℝ, distance = 5) -- Poles are kept 5 meters apart
(h3: ∀ poles: ℝ, poles = 26) -- 26 poles will be needed
(h4: 125 = 2 * (L + 50)) -- Use the perimeter calculated
: L = 12.5
:= sorry

end length_of_shorter_side_l536_536218


namespace combinations_of_coins_l536_536770

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536770


namespace quadratic_solution_l536_536505

theorem quadratic_solution (a c: ℝ) (h1 : a + c = 7) (h2 : a < c) (h3 : 36 - 4 * a * c = 0) : 
  a = (7 - Real.sqrt 13) / 2 ∧ c = (7 + Real.sqrt 13) / 2 :=
by
  sorry

end quadratic_solution_l536_536505


namespace std_eq_of_ellipse_std_eq_of_hyperbola_l536_536561

-- 1. The standard equation of the ellipse
theorem std_eq_of_ellipse (λ : ℝ) (h1 : λ > -9) (h2 : (4:ℝ, 3:ℝ) ∈ set_of (λ xy, xy.1^2 / (16 + λ) + xy.2^2 / (9 + λ) = 1)) :
  (4:ℝ, 3:ℝ) ∈ set_of (λ xy, xy.1^2 / 28 + xy.2^2 / 21 = 1) :=
sorry

-- 2. The standard equations of the hyperbola
theorem std_eq_of_hyperbola (λ : ℝ) (h3 : λ ≠ 0) :
  (λ = 1 ∨ λ = -1) →
  (2 * real.sqrt (λ + if λ > 0 then λ else -λ)) = 2 * real.sqrt 13 →
  (set_of (λ xy, xy.1^2 / (4 * λ) - xy.2^2 / (9 * λ) = 1) ∨
   set_of (λ xy, xy.2^2 / (9 * λ) - xy.1^2 / (4 * λ) = 1)) :=
sorry

end std_eq_of_ellipse_std_eq_of_hyperbola_l536_536561


namespace cube_surface_area_l536_536211

theorem cube_surface_area (side_length : ℝ) (h : side_length = 8) : 6 * side_length^2 = 384 :=
by
  rw [h]
  sorry

end cube_surface_area_l536_536211


namespace probability_point_within_small_spheres_l536_536221

theorem probability_point_within_small_spheres :
  ∀ (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r),
  let s := Real.sqrt (h^2 + r^2) in
  let R := r + s in
  let small_sphere_radius := s / 3 in
  let circumscribed_sphere_volume := (4 / 3) * Real.pi * R^3 in
  let small_spheres_total_volume := 4 * (4 / 3) * Real.pi * (small_sphere_radius^3) in
  small_spheres_total_volume / circumscribed_sphere_volume ≈ 0.3 :=
begin
  intros h r h_pos r_pos,
  let s := Real.sqrt (h^2 + r^2),
  let R := r + s,
  let small_sphere_radius := s / 3,
  let circumscribed_sphere_volume := (4 / 3) * Real.pi * R^3,
  let small_spheres_total_volume := 4 * (4 / 3) * Real.pi * (small_sphere_radius^3),
  sorry
end

end probability_point_within_small_spheres_l536_536221


namespace shaded_area_is_500_l536_536638

-- Define the coordinates of the vertices
def square_vertices : List (ℕ × ℕ) := [(0, 0), (30, 0), (30, 30), (0, 30)]
def shaded_vertices : List (ℕ × ℕ) := [(0, 0), (10, 0), (30, 20), (30, 30), (20, 30), (0, 10)]

-- Define the area of a polygon using the vertex list
noncomputable def area_of_polygon (vertices : List (ℕ × ℕ)) : ℚ :=
  let n := vertices.length
  let cross_product (a b : ℕ × ℕ) : ℚ := (a.1 : ℚ) * b.2 - (a.2 : ℚ) * b.1
  (vertices.zip (vertices.tail ++ [vertices.head])).map (λ p, cross_product p.1 p.2).sum / 2

-- Conditions and proof statements
theorem shaded_area_is_500 : area_of_polygon shaded_vertices = 500 := by
  sorry

end shaded_area_is_500_l536_536638


namespace equal_games_per_month_l536_536523

-- Define the given conditions
def total_games : ℕ := 27
def months : ℕ := 3
def games_per_month := total_games / months

-- Proposition that needs to be proven
theorem equal_games_per_month : games_per_month = 9 := 
by
  sorry

end equal_games_per_month_l536_536523


namespace min_AB_dot_CD_l536_536372

theorem min_AB_dot_CD (a b : ℝ) (h1 : 0 <= (a - 1)^2 + (b - 3 / 2)^2 - 13/4) :
  ∃ (a b : ℝ), (a-1)^2 + (b - 3 / 2)^2 - 13/4 = 0 :=
by
  sorry

end min_AB_dot_CD_l536_536372


namespace square_area_l536_536075

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (1, 2) ∧ p2 = (4, 6)) :
  let d := distance p1 p2 in
  d^2 = 25 :=
by
  sorry

end square_area_l536_536075


namespace hall_length_width_difference_is_10_l536_536155

noncomputable def length_width_difference : ℝ :=
  let L := 20 in   -- We calculate L based on given conditions
  let W := 10 in   -- We calculate W based on given conditions
  L - W  -- The difference between length and width

theorem hall_length_width_difference_is_10
 (L W : ℝ)
 (h1 : W = 1 / 2 * L)
 (h2 : L * W = 200) :
  L - W = 10 := by
  sorry

end hall_length_width_difference_is_10_l536_536155


namespace rectangle_no_shaded_probability_l536_536617

theorem rectangle_no_shaded_probability :
  let total_rows := 2,
      total_cols := 2005,
      total_rectangles := (total_cols + 1) * total_cols / 2,
      shaded_square_position := 1003,
      rectangles_with_shaded_square := 1002 * 1002 in
  (total_rectangles - rectangles_with_shaded_square) / (total_rectangles * total_rows) = 1001 / 2005 :=
by
  sorry

end rectangle_no_shaded_probability_l536_536617


namespace no_member_of_T_is_divisible_by_4_or_5_l536_536007

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4_or_5 :
  ∀ (n : ℤ), ¬ (∃ (T : ℤ), T = sum_of_squares_of_four_consecutive_integers n ∧ (T % 4 = 0 ∨ T % 5 = 0)) :=
by
  sorry

end no_member_of_T_is_divisible_by_4_or_5_l536_536007


namespace second_order_derivative_l536_536640

noncomputable def x (t : ℝ) : ℝ := sqrt t
noncomputable def y (t : ℝ) : ℝ := 1 / sqrt (1 - t)
noncomputable def dx_dt (t : ℝ) : ℝ := (sqrt t)'
noncomputable def dy_dt (t : ℝ) : ℝ := (1 / sqrt (1 - t))'
noncomputable def dy_dx (t : ℝ) : ℝ := dy_dt t / dx_dt t
noncomputable def d_dy_dx_dt (t : ℝ) : ℝ := (dy_dx t)'

theorem second_order_derivative
  (t : ℝ) (ht : t ≠ 1) (ht_pos : 0 < t):
  (d_dy_dx_dt t / dx_dt t) = (1 + 2 * t) * sqrt (1 - t) :=
by
  sorry

end second_order_derivative_l536_536640


namespace reportersNotCoveringPoliticsPercentage_l536_536197

-- Define the total number of reporters as 100 for normalization in the percentage calculations.
def totalReporters : ℕ := 100

-- Define the percentage of reporters covering local politics in country X
def localPoliticsPercentage : ℝ := 0.30

-- Define the percentage of reporters covering politics in general
def nonLocalPoliticsPercentage : ℝ := 0.25

-- Define the assumptions in terms of normalized percentages
def reportersCoveringLocalPolitics : ℝ := localPoliticsPercentage * totalReporters

-- Define reporters who cover politics as percentage of those who cover local politics, given the inverse of the nonLocal percentage
def reportersCoveringPolitics : ℝ := reportersCoveringLocalPolitics / (1.0 - nonLocalPoliticsPercentage)

-- Define the final statement to prove the percentage of reporters not covering politics
theorem reportersNotCoveringPoliticsPercentage : (totalReporters - reportersCoveringPolitics) / totalReporters = 0.60 :=
by
  sorry

end reportersNotCoveringPoliticsPercentage_l536_536197


namespace coin_combinations_count_l536_536744

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536744


namespace sam_morning_run_l536_536450

variable (X : ℝ)
variable (run_miles : ℝ) (walk_miles : ℝ) (bike_miles : ℝ) (total_miles : ℝ)

-- Conditions
def condition1 := walk_miles = 2 * run_miles
def condition2 := bike_miles = 12
def condition3 := total_miles = 18
def condition4 := run_miles + walk_miles + bike_miles = total_miles

-- Proof of the distance Sam ran in the morning
theorem sam_morning_run :
  (condition1 X run_miles walk_miles) →
  (condition2 bike_miles) →
  (condition3 total_miles) →
  (condition4 run_miles walk_miles bike_miles total_miles) →
  run_miles = 2 := by
  sorry

end sam_morning_run_l536_536450


namespace coin_combinations_sum_50_l536_536721

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536721


namespace meet_at_starting_point_l536_536195

-- Speeds in km/h
def speed_a := 32
def speed_b := 48
def speed_c := 36
def speed_d := 60

-- Track length in meters
def track_length := 800

-- Conversion factor from km/h to m/min
def kmph_to_mpm (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 60

-- Speeds in m/min
def speed_a_mpm := kmph_to_mpm speed_a
def speed_b_mpm := kmph_to_mpm speed_b
def speed_c_mpm := kmph_to_mpm speed_c
def speed_d_mpm := kmph_to_mpm speed_d

-- Time to complete one lap in minutes
def time_to_complete_lap (speed_mpm : ℕ) : ℕ :=
  track_length / speed_mpm

-- Times in minutes
def time_a := time_to_complete_lap speed_a_mpm
def time_b := time_to_complete_lap speed_b_mpm
def time_c := time_to_complete_lap speed_c_mpm
def time_d := time_to_complete_lap speed_d_mpm

-- Convert minutes to seconds
def min_to_sec (time_min : ℕ) : ℕ :=
  time_min * 60

-- times in seconds
def time_a_sec := min_to_sec time_a
def time_b_sec := min_to_sec time_b
def time_c_sec := min_to_sec time_c
def time_d_sec := min_to_sec time_d

-- Least common multiple function
def lcm (a b : ℕ) : ℕ :=
  a * b / Nat.gcd a b

-- Compute lcm of four numbers
def lcm4 (a b c d : ℕ) : ℕ :=
  lcm (lcm a b) (lcm c d)

-- Theorem statement
theorem meet_at_starting_point : lcm4 time_a_sec time_b_sec time_c_sec time_d_sec = 1440 :=
  by
    sorry

end meet_at_starting_point_l536_536195


namespace fourth_term_sequence_l536_536123

theorem fourth_term_sequence : (4 + 2^4) = 20 := by
  calc
    4 + 2^4 = 4 + 16 : by rfl
    ... = 20 : by rfl

end fourth_term_sequence_l536_536123


namespace total_reduction_approx_l536_536568

-- Definitions for the conditions
def original_price : ℝ := 500
def first_reduction_rate : ℝ := 0.07
def second_reduction_rate : ℝ := 0.05
def third_reduction_rate : ℝ := 0.03

-- The main theorem statement showing the total reduction
theorem total_reduction_approx : 
  let first_reduction := first_reduction_rate * original_price
  let price_after_first := original_price - first_reduction
  let second_reduction := second_reduction_rate * price_after_first
  let price_after_second := price_after_first - second_reduction
  let third_reduction := third_reduction_rate * price_after_second
  let final_price := price_after_second - third_reduction
  original_price - final_price ≈ 71.50 := 
by sorry -- Here we use 'by sorry' to skip the proof body.

end total_reduction_approx_l536_536568


namespace combinations_of_coins_l536_536773

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536773


namespace find_functional_f_l536_536004

-- Define the problem domain and functions
variable (f : ℕ → ℕ)
variable (ℕ_star : Set ℕ) -- ℕ_star is {1,2,3,...}

-- Conditions
axiom f_increasing (h1 : ℕ) (h2 : ℕ) (h1_lt_h2 : h1 < h2) : f h1 < f h2
axiom f_functional (x : ℕ) (y : ℕ) : f (y * f x) = x^2 * f (x * y)

-- The proof problem
theorem find_functional_f : (∀ x ∈ ℕ_star, f x = x^2) :=
sorry

end find_functional_f_l536_536004


namespace find_vec_b_collinear_angle_bisect_l536_536406

variable {ℝ : Type*} [AddCommGroup ℝ] [Module ℝ ℝ]

def vec_b0 := ![1, 2, 3] : Fin 3 → ℝ
def vec_a := ![2, -3, 1] + vec_b0
def vec_c := ![-1, 5, -3] + vec_b0
def vec_b := (![20 / 23, 40 / 23, 65 / 23] : Fin 3 → ℝ)

theorem find_vec_b_collinear_angle_bisect : 
  ∃ (b : Fin 3 → ℝ), 
    (∀ (t : ℝ), b = t • (vec_a + vec_c)) ∧
    b = vec_b :=
  sorry

end find_vec_b_collinear_angle_bisect_l536_536406


namespace count_whole_numbers_in_interval_l536_536876

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536876


namespace area_of_square_with_adjacent_points_l536_536090

theorem area_of_square_with_adjacent_points (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 6)) :
  let side_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in 
  let area := side_length^2 in 
  area = 25 :=
by
  sorry

end area_of_square_with_adjacent_points_l536_536090


namespace ratio_of_CP_to_PA_l536_536986

theorem ratio_of_CP_to_PA
  (A B C D M P H : Type)
  [Point A] [Point B] [Point C] [Point D] [Point M] [Point P] [Point H]
  (AB AC : ℝ) (hAB : AB = 15) (hAC : AC = 8)
  (hAngleBisect : line_through A D ∧ angle_bisector A)
  (hMidpointM : midpoint M A D)
  (hP_intersection : intersection P AC BM)
  (hH : altitude B AC H) :
  ratio CP PA = 529 / 120 :=
by sorry

end ratio_of_CP_to_PA_l536_536986


namespace Martiza_study_time_l536_536430

theorem Martiza_study_time :
  ∀ (x : ℕ),
  (30 * x + 30 * 25 = 20 * 60) →
  x = 15 :=
by
  intros x h
  sorry

end Martiza_study_time_l536_536430


namespace comparison_inequality_l536_536651

variable {f : ℝ → ℝ}
variable (h_derivative : ∀ x ∈ set.Ioo 0 (π / 2), (f' x * Real.sin x - Real.cos x * f x) > 0)

theorem comparison_inequality:
  √3 * f (π / 6) < f (π / 3) :=
sorry

end comparison_inequality_l536_536651


namespace hyperbola_asymptotes_l536_536475

-- Define the hyperbola equation as a condition
def hyperbola (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

-- Define the asymptote condition that we need to prove
def asymptote (x y : ℝ) : Prop := y = sqrt 2 * x ∨ y = -sqrt 2 * x

-- State the theorem that proves the asymptotic line equations
theorem hyperbola_asymptotes (x y : ℝ) : hyperbola x y → asymptote x y :=
sorry

end hyperbola_asymptotes_l536_536475


namespace combinations_of_coins_l536_536757

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536757


namespace sum_first_2014_terms_l536_536665

def sequence_is_arithmetic (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + a 2

def first_arithmetic_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :=
  S n = (n * (n - 1)) / 2

theorem sum_first_2014_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : sequence_is_arithmetic a) 
  (h2 : a 3 = 2) : 
  S 2014 = 1007 * 2013 :=
sorry

end sum_first_2014_terms_l536_536665


namespace ABCD_is_trapezoid_l536_536655

-- Definitions and conditions
variables {A B C D K L : Type} [add_comm_group K] [module K L]
def is_centroid (G : K) (A B C D : L) : Prop :=
  G = (A + B + C + D) / 4

def is_intersection (P Q R : L) : Prop :=
  ∃ k : K, P = k • R ∧ Q = (1 - k) • R

def is_parallel (P Q R S : L) : Prop :=
  ∃ k : K, P - Q = k • (R - S)

-- Given quadrilateral ABCD with properties
variables (A B C D : L)
variables (K : L) (L : L)
hypothesis hK : is_intersection (A - B) (C - D) K
hypothesis hL : is_intersection (A - C) (B - D) L
hypothesis hCentroid : is_centroid K L A B C D

-- Prove that ABCD is a trapezoid (AD parallel to BC)
theorem ABCD_is_trapezoid (A B C D : L) (K L: L) 
  (hK : is_intersection (A - B) (C - D) K)
  (hL : is_intersection (A - C) (B - D) L)
  (hCentroid : is_centroid (K • L) A B C D) : 
  is_parallel A D B C := 
sorry

end ABCD_is_trapezoid_l536_536655


namespace remainder_of_10_pow_23_minus_7_mod_6_l536_536207

theorem remainder_of_10_pow_23_minus_7_mod_6 : ((10 ^ 23 - 7) % 6) = 3 := by
  sorry

end remainder_of_10_pow_23_minus_7_mod_6_l536_536207


namespace least_sum_value_l536_536359

noncomputable def least_sum (a b : ℝ) (h : log 3 a + log 3 b ≥ 5) : ℝ :=
  a + b

theorem least_sum_value (a b : ℝ) (h : log 3 a + log 3 b ≥ 5) : least_sum a b h ≥ 18 * Real.sqrt 3 :=
sorry

end least_sum_value_l536_536359


namespace count_whole_numbers_in_interval_l536_536893

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536893


namespace coin_combinations_count_l536_536741

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536741


namespace number_of_integers_in_interval_l536_536917

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536917


namespace term_in_geometric_sequence_l536_536301

variable {α : Type*} [LinearOrderedField α]

-- Define the arithmetic sequence
def arithmetic_sequence (a d : α) (n : ℕ) : α := a + (n - 1) * d

-- Define the geometric sequence
def geometric_sequence (a r : α) (n : ℕ) : α := a * r ^ (n - 1)

-- Define the indices such that respective terms follow a geometric pattern
def specific_indices : List ℕ := [2, 6, 22]

-- The property that certain terms of an arithmetic sequence form a geometric sequence
def forms_geometric_sequence (a d : α) : Prop :=
  let terms := specific_indices.map (arithmetic_sequence a d)
  terms.length ≥ 2 ∧
  ∀ n, n < terms.length - 1 → ∃ q, terms.nth (n + 1) = some (q * (terms.nth n).get_or_else 0)

-- The main statement to prove
theorem term_in_geometric_sequence (a d : α) (h : d ≠ 0) :
  forms_geometric_sequence a d →
  arithmetic_sequence a d 342 = geometric_sequence a (4 : α) 6 := sorry

end term_in_geometric_sequence_l536_536301


namespace previous_year_height_l536_536199

def current_height : ℝ := 147
def growth_percent : ℝ := 5

-- theorem statement to prove the previous year's height given the current height and growth percentage.
theorem previous_year_height (current_height : ℝ) (growth_percent : ℝ) : 
  ∃ (h_prev : ℝ), current_height = h_prev * (1 + growth_percent / 100) → h_prev = 140 :=
begin
  sorry  -- Proof goes here
end

end previous_year_height_l536_536199


namespace area_of_overlapping_region_l536_536247

noncomputable def triangle1_area : ℝ :=
(abs((0:ℝ) * (1:ℝ) + (2:ℝ) * (2:ℝ) + (0:ℝ) * (0:ℝ) - 
(1:ℝ) * (2:ℝ) - (2:ℝ) * (0:ℝ) - (2:ℝ) * (0:ℝ))) / 2

noncomputable def triangle2_area : ℝ :=
(abs((2:ℝ) * (1:ℝ) + (0:ℝ) * (2:ℝ) + (2:ℝ) * (2:ℝ) - 
(2:ℝ) * (2:ℝ) - (1:ℝ) * (2:ℝ) - (0:ℝ) * (0:ℝ))) / 2

noncomputable def overlapping_area : ℝ := 2

theorem area_of_overlapping_region : (overlapping_area : ℝ) = 2 := by
  sorry

end area_of_overlapping_region_l536_536247


namespace seashells_needed_to_reach_target_l536_536432

-- Definitions based on the conditions
def current_seashells : ℕ := 19
def target_seashells : ℕ := 25

-- Statement to prove
theorem seashells_needed_to_reach_target : target_seashells - current_seashells = 6 :=
by
  sorry

end seashells_needed_to_reach_target_l536_536432


namespace square_area_adjacency_l536_536050

-- Definition of points as pairs of integers
def Point := ℤ × ℤ

-- Define the points (1,2) and (4,6)
def P1 : Point := (1, 2)
def P2 : Point := (4, 6)

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement for proving the area of a square given the side length
theorem square_area_adjacency (h : distance P1 P2 = 5) : ∃ area : ℝ, area = 25 :=
by
  use 25
  sorry

end square_area_adjacency_l536_536050


namespace EB_eq_CF_l536_536005

-- Definitions and conditions stated as per the problem
variable {A B C D E F : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
variable (triangle_ABC : A → B → C → Prop)
variable (is_angle_bisector_foot : A → D → Prop)
variable (circumcircle_ACD_intersects_AB : A → C → D → E → Prop)
variable (circumcircle_ABD_intersects_AC : A → B → D → F → Prop)

-- The theorem we need to show
theorem EB_eq_CF 
  {A B C D E F : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
  (h_triangle : triangle_ABC A B C)
  (h_bisector : is_angle_bisector_foot A D)
  (h_ADC : circumcircle_ACD_intersects_AB A C D E)
  (h_ABD : circumcircle_ABD_intersects_AC A B D F) :
  E = F :=
sorry

end EB_eq_CF_l536_536005


namespace problem_statement_l536_536142

noncomputable def k_value (k : ℝ) : Prop :=
  (∀ (x y : ℝ), x + y = k → x^2 + y^2 = 4) ∧ (∀ (A B : ℝ × ℝ), (∃ (x y : ℝ), A = (x, y) ∧ x^2 + y^2 = 4) ∧ (∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4) ∧ 
  (∃ (xa ya xb yb : ℝ), A = (xa, ya) ∧ B = (xb, yb) ∧ |(xa - xb, ya - yb)| = |(xa, ya)| + |(xb, yb)|)) → k = 2

theorem problem_statement (k : ℝ) (h : k > 0) : k_value k :=
  sorry

end problem_statement_l536_536142


namespace solve_for_nabla_l536_536932

theorem solve_for_nabla : ∃ (∇ : ℤ), 3 * (-2) = ∇ + 2 ∧ ∇ = -8 :=
by { existsi (-8), split, exact rfl, exact rfl }

end solve_for_nabla_l536_536932


namespace whole_numbers_in_interval_l536_536911

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536911


namespace avg_children_nine_families_l536_536168

theorem avg_children_nine_families
  (total_families : ℕ)
  (average_children : ℕ)
  (childless_families : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  total_children = total_families * average_children →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℝ) = 4.0 :=
begin
  intros,
  sorry
end

end avg_children_nine_families_l536_536168


namespace carlton_school_earnings_l536_536975

theorem carlton_school_earnings :
  let students_days_adams := 8 * 4
  let students_days_byron := 5 * 6
  let students_days_carlton := 6 * 10
  let total_wages := 1092
  students_days_adams + students_days_byron = 62 → 
  62 * (2 * x) + students_days_carlton * x = total_wages → 
  x = (total_wages : ℝ) / 184 → 
  (students_days_carlton : ℝ) * x = 356.09 := 
by
  intros _ _ _ 
  sorry

end carlton_school_earnings_l536_536975


namespace fleas_cannot_reach_final_positions_l536_536524

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def initial_A : Point2D := ⟨0, 0⟩
def initial_B : Point2D := ⟨1, 0⟩
def initial_C : Point2D := ⟨0, 1⟩

def area (A B C : Point2D) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def final_A : Point2D := ⟨1, 0⟩
def final_B : Point2D := ⟨-1, 0⟩
def final_C : Point2D := ⟨0, 1⟩

theorem fleas_cannot_reach_final_positions : 
    ¬ (∃ (flea_move_sequence : List (Point2D → Point2D)), 
    area initial_A initial_B initial_C = area final_A final_B final_C) :=
by 
  sorry

end fleas_cannot_reach_final_positions_l536_536524


namespace equivalence_of_complements_union_l536_536425

open Set

-- Definitions as per the conditions
def U : Set ℝ := univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def complement_U (S : Set ℝ) : Set ℝ := U \ S

-- Mathematical statement to be proved
theorem equivalence_of_complements_union :
  (complement_U M ∪ complement_U N) = { x : ℝ | x < 1 ∨ x ≥ 5 } :=
by
  -- Non-trivial proof, hence skipped with sorry
  sorry

end equivalence_of_complements_union_l536_536425


namespace area_of_triangle_AEB_l536_536978

noncomputable def rectangle_area_AEB : ℝ :=
  let AB := 8
  let BC := 4
  let DF := 2
  let GC := 2
  let FG := 8 - DF - GC -- DC (8 units) minus DF and GC.
  let ratio := AB / FG
  let altitude_AEB := BC * ratio
  let area_AEB := 0.5 * AB * altitude_AEB
  area_AEB

theorem area_of_triangle_AEB : rectangle_area_AEB = 32 :=
by
  -- placeholder for detailed proof
  sorry

end area_of_triangle_AEB_l536_536978


namespace whole_numbers_in_interval_l536_536916

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536916


namespace sqrt_meaningful_real_l536_536940

theorem sqrt_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 5)) → x ≥ 5 :=
by
  intro h
  cases h with y hy
  have : x - 5 ≥ 0 := by sorry -- simplified proof of sqrt definition
  linarith

end sqrt_meaningful_real_l536_536940


namespace square_area_l536_536069

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (1, 2) ∧ p2 = (4, 6)) :
  let d := distance p1 p2 in
  d^2 = 25 :=
by
  sorry

end square_area_l536_536069


namespace eight_times_10x_plus_14pi_l536_536352

theorem eight_times_10x_plus_14pi (x : ℝ) (Q : ℝ) (h : 4 * (5 * x + 7 * π) = Q) : 
  8 * (10 * x + 14 * π) = 4 * Q := 
by {
  sorry  -- proof is omitted
}

end eight_times_10x_plus_14pi_l536_536352


namespace min_abs_w_product_l536_536417

noncomputable def g (x : ℝ) : ℝ := x^4 + 20 * x^3 + 98 * x^2 + 100 * x + 25

theorem min_abs_w_product : 
  ∃ (w : ℕ → ℝ), 
    (∀ n, w n ∈ {root : ℝ | g root = 0}) ∧ 
    (∃ (a b c d : fin 4), 
      (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
      |w a * w b + w c * w d| = 10) :=
sorry

end min_abs_w_product_l536_536417


namespace minimum_distance_PQ_l536_536311

theorem minimum_distance_PQ : 
  ∃ a b : ℝ, (b = 1 - real.log 2) ∧ (a = real.log (2 * b)) ∧ 
  (dist (a, (1/2) * real.exp a) (b, b) = (real.sqrt 2 / 2) * (1 - real.log 2)) :=
by
  sorry

end minimum_distance_PQ_l536_536311


namespace tom_mangoes_purchase_l536_536162

theorem tom_mangoes_purchase 
  (p_apples : ℕ) (r_apples : ℕ) (r_mangoes : ℕ) (total_paid : ℕ) (m : ℕ)
  (h1 : p_apples = 8)
  (h2 : r_apples = 70)
  (h3 : r_mangoes = 70)
  (h4 : total_paid = 1190) :
  m = 9 :=
by
  have total_cost_apples := h1 * h2
  have total_cost_mangoes := h4 - total_cost_apples
  have m_kg := total_cost_mangoes / h3
  exact Eq.symm m_kg ▸ rfl

end tom_mangoes_purchase_l536_536162


namespace sum_series_with_sign_reversal_l536_536626

theorem sum_series_with_sign_reversal : 
  let cube_intervals := fun n => filter (fun k => (n - 1)^3 < k ∧ k ≤ n^3) (range 12101) in 
  let sign_factor := fun n => if even n then -1 else 1 in 
  (sum (range 12101) (fun k => 
    (sum (cube_intervals (nat.ceilsqrt k / 3))
         (fun j => j) * 
      sign_factor (nat.ceilsqrt k / 3)))) = sorry := sorry

end sum_series_with_sign_reversal_l536_536626


namespace max_projection_area_of_tetrahedron_l536_536173

theorem max_projection_area_of_tetrahedron (A B C D : Point) (h1 : ∀ (u v w : ℝ), 
  equilateral_triangle u v w) (h2 : dihedral_angle ABCD 60) :
  ∃ area : ℝ, area = √3 / 4 := sorry

end max_projection_area_of_tetrahedron_l536_536173


namespace min_value_inv_sum_l536_536026

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end min_value_inv_sum_l536_536026


namespace linear_function_quadrant_check_l536_536501

theorem linear_function_quadrant_check (x y : ℝ) :
  let m : ℝ := -2
  let b : ℝ := -1
  let f : ℝ → ℝ := λ x, m * x + b in
  ∀ (x : ℝ), (f x = y) → ((x > 0 ∧ y > 0) → false) :=
by {
  intros x y m b f hxy good_quadrant,
  let rule := -2 * x - 1,
  rw [[←rule] rhos_rule, hxy], -- ensures f(x) equals y correctly
  simp only [not_and_simp, ne.def],
	sorry
}

end linear_function_quadrant_check_l536_536501


namespace pension_equality_l536_536597

theorem pension_equality (x c d r s: ℝ) (h₁ : d ≠ c) 
    (h₂ : x > 0) (h₃ : 2 * x * (d - c) + d^2 - c^2 ≠ 0)
    (h₄ : ∀ k:ℝ, k * (x + c)^2 - k * x^2 = r)
    (h₅ : ∀ k:ℝ, k * (x + d)^2 - k * x^2 = s) 
    : ∃ k : ℝ, k = (s - r) / (2 * x * (d - c) + d^2 - c^2) 
    → k * x^2 = (s - r) * x^2 / (2 * x * (d - c) + d^2 - c^2) :=
by {
    sorry
}

end pension_equality_l536_536597


namespace total_surface_area_of_solid_l536_536586

theorem total_surface_area_of_solid (r : ℝ) (h : ℝ) (hsurface_area : ℝ) 
  (base_area : ℝ) 
  (h_radius : π * r^2 = 144 * π)
  (h_height : h = 2 * r)
  (hsurface_area_hemisphere : hsurface_area = 2 * π * r^2)
  (l : ℝ) 
  (cone_lateral_area : ℝ)
  (h_l : l = real.sqrt (r^2 + (2 * r)^2))
  (h_cone_area : cone_lateral_area = π * r * l) :
  hsurface_area + cone_lateral_area = 288 * π + 144 * real.sqrt 5 * π :=
sorry

end total_surface_area_of_solid_l536_536586


namespace brad_zip_code_l536_536608

theorem brad_zip_code (a b c d e : ℕ) 
  (h1 : a = b) 
  (h2 : c = 0) 
  (h3 : d = 2 * a) 
  (h4 : d + e = 8) 
  (h5 : a + b + c + d + e = 10) : 
  (a, b, c, d, e) = (1, 1, 0, 2, 6) :=
by 
  -- Proof omitted on purpose
  sorry

end brad_zip_code_l536_536608


namespace closed_polygonal_chain_exists_l536_536508

theorem closed_polygonal_chain_exists (n m : ℕ) : 
  ((n % 2 = 1 ∨ m % 2 = 1) ↔ 
   ∃ (length : ℕ), length = (n + 1) * (m + 1) ∧ length % 2 = 0) :=
by sorry

end closed_polygonal_chain_exists_l536_536508


namespace coin_combination_l536_536783

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536783


namespace final_quarters_l536_536444

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 760
def first_spent : ℕ := 418
def second_spent : ℕ := 192

-- Define the final amount of quarters Sally should have
theorem final_quarters (initial_quarters first_spent second_spent : ℕ) : initial_quarters - first_spent - second_spent = 150 :=
by
  sorry

end final_quarters_l536_536444


namespace probability_of_same_number_l536_536593

noncomputable def probability_same_number : ℚ :=
  let less_than_250 := {n : ℕ | n < 250}
  let multiples_of_25 := {n | n ∈ less_than_250 ∧ n % 25 = 0}
  let multiples_of_30 := {n | n ∈ less_than_250 ∧ n % 30 = 0}
  let common_multiples := multiples_of_25 ∩ multiples_of_30
  if h : (multiples_of_25.card * multiples_of_30.card ≠ 0) then
    common_multiples.card / (multiples_of_25.card * multiples_of_30.card)
  else
    0

theorem probability_of_same_number :
  probability_same_number = 1 / 80 :=
by
  sorry

end probability_of_same_number_l536_536593


namespace sqrt_meaningful_real_l536_536941

theorem sqrt_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 5)) → x ≥ 5 :=
by
  intro h
  cases h with y hy
  have : x - 5 ≥ 0 := by sorry -- simplified proof of sqrt definition
  linarith

end sqrt_meaningful_real_l536_536941


namespace exponential_function_base_l536_536952

theorem exponential_function_base (a : ℝ) (f : ℝ → ℝ) (h_exp : f = λ x, (a^2 - 3a + 3) * (a^x)) 
  (h_cond_pos : a > 0) (h_cond_not1 : a ≠ 1) : a = 2 :=
sorry

end exponential_function_base_l536_536952


namespace range_of_f_l536_536010

-- Definitions based on the conditions
def f (a b x : ℝ) : ℝ := |a * x + b|

/-- Prove the range of f(x) given the conditions -/
theorem range_of_f (a b : ℝ) (h_pos : 0 < a) (h_neg : b < 0) :
  set.range (f a b) = set.Icc 0 (max (|b|) (|a + b|)) :=
sorry

end range_of_f_l536_536010


namespace count_whole_numbers_in_interval_l536_536856

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536856


namespace prob_event_A_l536_536949

-- Define the sample space of die rolls
def sample_space := { (a, b) : ℕ × ℕ // a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} }

-- Define the event that z^2 is purely imaginary
def event_A (p : sample_space) : Prop := p.val.1 = p.val.2

-- Define the probability of an event given the sample space
def probability (E : set sample_space) : ℚ :=
  (E.to_finset.card : ℚ) / (sample_space.to_finset.card : ℚ)

theorem prob_event_A : probability {p | event_A p} = 1/6 :=
by sorry

end prob_event_A_l536_536949


namespace total_area_correct_l536_536555

noncomputable def total_area (r p q : ℝ) : ℝ :=
  r^2 + 4*p^2 + 12*q

theorem total_area_correct
  (r p q : ℝ)
  (h : 12 * q = r^2 + 4 * p^2 + 45)
  (r_val : r = 6)
  (p_val : p = 1.5)
  (q_val : q = 7.5) :
  total_area r p q = 135 := by
  sorry

end total_area_correct_l536_536555


namespace largest_even_n_inequality_l536_536255

noncomputable theory

open Real

theorem largest_even_n_inequality :
  ∃ n : ℕ, (even n) ∧ (∀ x : ℝ, (sin x) ^ (2 * n) + (cos x) ^ (2 * n) + (tan x) ^ 2 ≥ 1 / n) ∧ (∀ (m : ℕ), (even m) ∧ (∀ x : ℝ, (sin x) ^ (2 * m) + (cos x) ^ (2 * m) + (tan x) ^ 2 ≥ 1 / m) → m ≤ n) :=
sorry

end largest_even_n_inequality_l536_536255


namespace count_whole_numbers_in_interval_l536_536880

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536880


namespace shortest_distance_l536_536332

def C1 (ρ θ : ℝ) : Prop := ρ^2 - 2 * ρ * Real.cos θ - 1 = 0

def C2 (x y t : ℝ) : Prop := x = 3 - t ∧ y = 1 + t

theorem shortest_distance :
  let circle_center := (1 : ℝ, 0 : ℝ)
  let line_eq (x y : ℝ) := x + y - 4 = 0
  distance_from_center_to_line (cx cy a b c : ℝ) : ℝ := |a * cx + b * cy + c| / Real.sqrt (a^2 + b^2)
  shortest_distance_from_circle_to_line (d r : ℝ) := d - r
  -- center of the circle
  let center_x, center_y := 1, 0
  -- coefficients of the line
  let a, b, c := (1 : ℝ), (1 : ℝ), (-4 : ℝ)
  -- radius of the circle
  let radius := Real.sqrt 2
  shortest_distance_from_circle_to_line (distance_from_center_to_line center_x center_y a b c) radius = Real.sqrt 2 / 2 :=
sorry

end shortest_distance_l536_536332


namespace coin_combinations_count_l536_536790

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536790


namespace triangles_congruent_l536_536001

variables (A B C D E K L M P : Point)
variables [inst1 : Parallelogram A B C D] (circACD : Circle A C D) (circCEL : Circle C E L)
variables (BAD_60deg : Angle A B D = 60)
variables (intE : Intersection (Diagonal A C) (Diagonal B D) = E)
variables (EP_int_circ : Intersect (Line E P) circCEL = {E, M})
variables (circACD_meet : Intersects circACD (Line B A) K)
variables (circACD_meet2 : Intersects circACD (Line B D) P)
variables (circACD_meet3 : Intersects circACD (Line B C) L)

theorem triangles_congruent : Congruent (Triangle K L M) (Triangle C A P) :=
sorry

end triangles_congruent_l536_536001


namespace common_ratio_of_geometric_progression_l536_536485

-- Define the problem conditions
variables {a b c q : ℝ}

-- The sequence a, b, c is a geometric progression
def geometric_progression (a b c : ℝ) (q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- The sequence 577a, (2020b/7), (c/7) is an arithmetic progression
def arithmetic_progression (x y z : ℝ) : Prop :=
  2 * y = x + z

-- Main theorem statement to prove
theorem common_ratio_of_geometric_progression (h1 : geometric_progression a b c q) 
  (h2 : arithmetic_progression (577 * a) (2020 * b / 7) (c / 7)) 
  (h3 : b < a ∧ c < b) : q = 4039 :=
sorry

end common_ratio_of_geometric_progression_l536_536485


namespace combinations_of_coins_l536_536764

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536764


namespace largest_area_polygons_l536_536250

-- Define the area of each polygon
def area_P := 4
def area_Q := 6
def area_R := 3 + 3 * (1 / 2)
def area_S := 6 * (1 / 2)
def area_T := 5 + 2 * (1 / 2)

-- Proof of the polygons with the largest area
theorem largest_area_polygons : (area_Q = 6 ∧ area_T = 6) ∧ area_Q ≥ area_P ∧ area_Q ≥ area_R ∧ area_Q ≥ area_S :=
by
  sorry

end largest_area_polygons_l536_536250


namespace count_whole_numbers_in_interval_l536_536843

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536843


namespace round_addition_l536_536232

theorem round_addition (a b : ℝ) (h1 : a = 45.23) (h2 : b = 78.569) :
  (Real.floor ((a + b) * 10 + 0.5) : ℝ) / 10 = 123.8 :=
by
  sorry

end round_addition_l536_536232


namespace area_increase_percentage_l536_536956

-- Define the conditions when the radius is increased by 50%
def original_radius (r : ℝ) := r
def new_radius (r : ℝ) := 1.5 * r

-- Define the original and new areas
def original_area (r : ℝ) := Real.pi * r^2
def new_area (r : ℝ) := Real.pi * (new_radius r)^2

-- Prove that the percentage increase in area is 125%
theorem area_increase_percentage (r : ℝ) (h : 0 ≤ r) :
  ((new_area r - original_area r) / original_area r) * 100 = 125 :=
by
  sorry

end area_increase_percentage_l536_536956


namespace triangle_has_angle_45_l536_536990

theorem triangle_has_angle_45
  (A B C : ℝ)
  (h1 : A + B + C = 180)
  (h2 : B + C = 3 * A) :
  A = 45 :=
by
  sorry

end triangle_has_angle_45_l536_536990


namespace increasing_f_and_min_g_l536_536666

variable {α β t : ℝ}

def quadratic_has_real_roots (t : ℝ) : Prop :=
  let discrim := (4 * t) ^ 2 - 4 * 4 * (-1)
  discrim ≥ 0

def root_condition (t : ℝ) (α β : ℝ) : Prop :=
  4 * α^2 - 4 * t * α - 1 = 0 ∧ 4 * β^2 - 4 * t * β - 1 = 0 ∧ α ≤ β

def f (x : ℝ) : ℝ := sorry

def g (x t : ℝ) : ℝ := max (f x) - min (f x)

theorem increasing_f_and_min_g :
  (∀ (t : ℝ), quadratic_has_real_roots t → ∃ α β : ℝ, root_condition t α β ∧ 
   (∀ x ∈ set.Icc α β, f x) is_increasing_on (set.Icc α β) ∧ (∃ t, g t (max f β - min f α) = 0)) :=
sorry

end increasing_f_and_min_g_l536_536666


namespace num_ints_less_than_200_l536_536346

theorem num_ints_less_than_200 :
  ∀ n m : ℕ,
    0 < n ∧ n < 200 ∧ (∃ (k : ℤ), k % 2 = 0 ∧ n = 2*k + 2 ∧ m = k*(k + 2) ∧ m % 5 = 0) →
    ∃ (N : ℕ), N = 20 :=
begin
  sorry,
end

end num_ints_less_than_200_l536_536346


namespace curve_eq_proof_line_eq_proof_max_distance_proof_l536_536373

-- Define the parametric equations of curve C
def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos θ, Real.sin θ)

-- Define the equations of curve C and line l
def general_curve_eq (x y : ℝ) :=
  x^2 / 3 + y^2 = 1

def cartesian_line_eq (x y : ℝ) :=
  x - y + 3 = 0

-- Point on curve C parameterized by θ
def point_on_curve (θ : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos θ, Real.sin θ)

-- Distance from point P to line l
def distance_from_point_to_line (P : ℝ × ℝ) :=
  (abs (fst P - snd P + 3)) / sqrt 2

-- Proving the general equation of curve C
theorem curve_eq_proof : 
  ∀ θ : ℝ, let (x, y) := parametric_curve θ in general_curve_eq x y :=
begin
  sorry
end

-- Proving the Cartesian equation of line l
theorem line_eq_proof : 
  ∀ θ : ℝ, let ρ := sqrt ((sqrt 3 * Real.cos θ)^2 + (Real.sin θ)^2) in ρ * Real.sin (θ - π / 4) = 3 → 
  cartesian_line_eq (sqrt 3 * Real.cos θ) (Real.sin θ) :=
begin
  sorry
end

-- Proving the maximum distance
theorem max_distance_proof : 
  ∀ θ : ℝ, let P := point_on_curve θ in (distance_from_point_to_line P) ≤ 5 * sqrt 2 / 2 :=
begin
  sorry
end

end curve_eq_proof_line_eq_proof_max_distance_proof_l536_536373


namespace area_of_square_with_adjacent_points_l536_536085

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def side_length := distance 1 2 4 6
def area_of_square (side : ℝ) : ℝ := side ^ 2

theorem area_of_square_with_adjacent_points :
  area_of_square side_length = 25 :=
by
  unfold side_length
  unfold area_of_square
  sorry

end area_of_square_with_adjacent_points_l536_536085


namespace ratio_of_areas_l536_536979

-- Given values
variables (a b : ℝ)
variable {A B C D M N : ℝ × ℝ}

-- Conditions in the problem
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2 * a, 0)
def C : ℝ × ℝ := (2 * a, b)
def D : ℝ × ℝ := (0, b)
def M : ℝ × ℝ := (a / 2, 0)
def N : ℝ × ℝ := (2 * a, b / 2)

-- Proof statement
theorem ratio_of_areas (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let triangle_AMN := (1 / 4 * b * real.sqrt((3 * a / 2) ^ 2 + (b / 2) ^ 2)) in
  let rectangle_ABCD := 2 * a * b in
  (triangle_AMN / rectangle_ABCD) = (real.sqrt(9 * a ^ 2 + b ^ 2) / (8 * a)) :=
by sorry

end ratio_of_areas_l536_536979


namespace coin_combinations_count_l536_536794

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536794


namespace medians_altitudes_angle_bisectors_coincide_l536_536145

theorem medians_altitudes_angle_bisectors_coincide (ABC : Triangle) (h_eq : ABC.is_equilateral) :
  (∀ (a b c : Point), (ABC.is_median a b c) ↔ (ABC.is_altitude a b c) ∧ (ABC.is_angle_bisector a b c)) :=
sorry

-- Definitions for Triangle, Point, is_equilateral, is_median, is_altitude, and is_angle_bisector
-- would generally be provided elsewhere in the formalization.

end medians_altitudes_angle_bisectors_coincide_l536_536145


namespace coin_combinations_count_l536_536792

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536792


namespace determine_dress_and_notebooks_l536_536525

structure Girl :=
  (name : String)
  (dress_color : String)
  (notebook_color : String)

def colors := ["red", "yellow", "blue"]

def Sveta : Girl := ⟨"Sveta", "red", "red"⟩
def Ira : Girl := ⟨"Ira", "blue", "yellow"⟩
def Tania : Girl := ⟨"Tania", "yellow", "blue"⟩

theorem determine_dress_and_notebooks :
  (Sveta.dress_color = Sveta.notebook_color) ∧
  (¬ Tania.dress_color = "red") ∧
  (¬ Tania.notebook_color = "red") ∧
  (Ira.notebook_color = "yellow") ∧
  (Sveta ∈ [Sveta, Ira, Tania]) ∧
  (Ira ∈ [Sveta, Ira, Tania]) ∧
  (Tania ∈ [Sveta, Ira, Tania]) →
  ([Sveta, Ira, Tania] = 
   [{name := "Sveta", dress_color := "red", notebook_color := "red"},
    {name := "Ira", dress_color := "blue", notebook_color := "yellow"},
    {name := "Tania", dress_color := "yellow", notebook_color := "blue"}])
:=
by
  intro h
  sorry

end determine_dress_and_notebooks_l536_536525


namespace sum_of_decimals_as_fraction_l536_536634

/-- Define the problem inputs as constants -/
def d1 : ℚ := 2 / 10
def d2 : ℚ := 4 / 100
def d3 : ℚ := 6 / 1000
def d4 : ℚ := 8 / 10000
def d5 : ℚ := 1 / 100000

/-- The main theorem statement -/
theorem sum_of_decimals_as_fraction : 
  d1 + d2 + d3 + d4 + d5 = 24681 / 100000 := 
by 
  sorry

end sum_of_decimals_as_fraction_l536_536634


namespace max_x_sqrtxy_sqrt4xyz_l536_536032

variable {x y z : ℝ}

theorem max_x_sqrtxy_sqrt4xyz (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  x + Real.sqrt (x * y) + Real.root 4 (x * y * z) ≤ 7 / 6 := 
sorry

end max_x_sqrtxy_sqrt4xyz_l536_536032


namespace function_representation_and_monotonic_interval_l536_536687

def f (A ω φ x : ℝ) : ℝ := A * sin (ω * x + φ)

variables (A ω φ : ℝ)
variables (hA : A = 2) (hω : ω = 4) (hφ : φ = π / 6)
variables (T : ℝ) (hT : T = π / 2)
variables (hx_min : f A ω φ (π / 3) = -2)
variables (hx_sym : ∀ x, f A ω φ x = f A ω φ (π / 6 - x))

noncomputable def transformed_f (x : ℝ) : ℝ :=
  f A 2 (π / 3 + φ) (x - π / 6)

theorem function_representation_and_monotonic_interval :
  (f A ω φ x = 2 * sin (4 * x + π / 6)) ∧
  (∀ k ∈ ℤ, ∃ I, I = [k * π / 2 - π / 6, k * π / 2 + π / 12]) :=
sorry

end function_representation_and_monotonic_interval_l536_536687


namespace problem_f_a_minus_f_neg_a_eq_zero_l536_536563

section
variable {α : Type*}

def f (x : α) [HasPow α ℕ] [Mul α] [Add α] [Sub α] [HasZero α] [HasOne α] : α := 3 * x ^ 2 - 1

theorem problem_f_a_minus_f_neg_a_eq_zero 
  [HasPow α ℕ] [Mul α] [Add α] [Sub α] [HasZero α] [HasOne α] (a : α) :
  f a - f (-a) = (0 : α) := 
by
  sorry
end

end problem_f_a_minus_f_neg_a_eq_zero_l536_536563


namespace Jim_speed_l536_536286

def driving_time (start_time end_time : ℕ) : ℕ := sorry -- Here, start_time and end_time can be expressed to compute driving time in minutes

def hours_of_driving_time (time_in_minutes : ℕ) : ℝ := (time_in_minutes : ℝ) / 60

def speed (distance hours : ℝ) : ℝ := distance / hours

theorem Jim_speed :
  driving_time 1945 2130 = 105 →
  hours_of_driving_time 105 = 1.75 →
  speed 84 1.75 = 48 :=
by
  intros h1 h2
  -- Proof steps not required
  sorry

end Jim_speed_l536_536286


namespace coin_combinations_50_cents_l536_536725

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536725


namespace sum_sequences_l536_536613

theorem sum_sequences : 
  (1 + 12 + 23 + 34 + 45) + (10 + 20 + 30 + 40 + 50) = 265 := by
  sorry

end sum_sequences_l536_536613


namespace circumference_of_smaller_circle_l536_536126

noncomputable def pi_approx : ℝ := 3.14159

noncomputable def circumference (r : ℝ) : ℝ :=
  2 * pi_approx * r

noncomputable def area (r : ℝ) : ℝ :=
  pi_approx * r ^ 2

theorem circumference_of_smaller_circle:
  let R := 704 / (2 * pi_approx) in
  let area_diff := 33893.63668085003 in
  let r := Real.sqrt (R^2 - area_diff / pi_approx) in
  circumference r ≈ 263.89356 :=
by
  sorry

end circumference_of_smaller_circle_l536_536126


namespace area_of_square_with_adjacent_points_l536_536087

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def side_length := distance 1 2 4 6
def area_of_square (side : ℝ) : ℝ := side ^ 2

theorem area_of_square_with_adjacent_points :
  area_of_square side_length = 25 :=
by
  unfold side_length
  unfold area_of_square
  sorry

end area_of_square_with_adjacent_points_l536_536087


namespace corina_problem_l536_536948

variable (P Q : ℝ)

theorem corina_problem (h1 : P + Q = 16) (h2 : P - Q = 4) : P = 10 :=
sorry

end corina_problem_l536_536948


namespace triangle_construction_exists_l536_536531

noncomputable def exist_triangle (O P H : Point) : Prop :=
∃ (A B C : Point), 
  circumcenter A B C = O ∧ 
  centroid A B C = P ∧ 
  altitude_foot A B C = H

theorem triangle_construction_exists (O P H : Point) : exist_triangle O P H :=
sorry

end triangle_construction_exists_l536_536531


namespace find_m_l536_536701

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def N (m : ℝ) : Set ℝ := {x | x*x - m*x < 0}
noncomputable def M_inter_N (m : ℝ) : Set ℝ := {x | 0 < x ∧ x < 1}

theorem find_m (m : ℝ) (h : M ∩ (N m) = M_inter_N m) : m = 1 :=
by sorry

end find_m_l536_536701


namespace simplify_expression_l536_536453

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l536_536453


namespace find_angle_B_perimeter_range_l536_536008

variable {a b c A B C : ℝ}

-- Condition given in the problem
axiom cond : a - b*(1 - 2*sin(C/2)^2) = 1/2 * c

-- Part 1: Prove that B = π / 3
theorem find_angle_B (h : cond) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
: B = π / 3 := 
  sorry

-- Part 2: Given b = 6, prove the perimeter range is (12, 18]
theorem perimeter_range (h : cond) (h_b : b = 6)
  (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
: 12 < a + b + c ∧ a + b + c ≤ 18 :=
  sorry

end find_angle_B_perimeter_range_l536_536008


namespace trainer_voice_radius_l536_536227

noncomputable def area_of_heard_voice (r : ℝ) : ℝ := (1/4) * Real.pi * r^2

theorem trainer_voice_radius :
  ∃ r : ℝ, abs (r - 140) < 1 ∧ area_of_heard_voice r = 15393.804002589986 :=
by
  sorry

end trainer_voice_radius_l536_536227


namespace find_x_l536_536707

def vector := prod ℝ ℝ

def a : vector := (1, 2)
def b (x : ℝ) : vector := (2 * x, x)
def c : vector := (3, 1)

def is_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v1 = (k * v2.1, k * v2.2)

theorem find_x (x : ℝ) (h : is_parallel (a.1 + (b x).1, a.2 + (b x).2) c) : x = -5 :=
  sorry

end find_x_l536_536707


namespace find_f_20_l536_536011

theorem find_f_20 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = (1/2) * f (x + 2))
  (h2 : f 2 = 1) :
  f 20 = 512 :=
sorry

end find_f_20_l536_536011


namespace minimum_area_triangle_AOB_l536_536152

theorem minimum_area_triangle_AOB : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) ∧ (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) → (1/2 * a * b ≥ 12)) := 
sorry

end minimum_area_triangle_AOB_l536_536152


namespace smallest_two_digit_switch_l536_536539

-- Definitions based on the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def digits_switched_result (n : ℕ) : ℕ := 
  let a := n / 10  -- tens digit
  let b := n % 10  -- units digit
  10 * b + a
def condition (n : ℕ) : Prop := 4 * digits_switched_result(n) = 2 * n

-- The Lean theorem statement
theorem smallest_two_digit_switch : 
  ∃ n : ℕ, is_two_digit(n) ∧ condition(n) ∧ (∀ m : ℕ, is_two_digit(m) ∧ condition(m) → n ≤ m) ∧ n = 52 :=
by
  -- This is the theorem statement, the proof is bypassed.
  sorry

end smallest_two_digit_switch_l536_536539


namespace whole_numbers_in_interval_l536_536913

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536913


namespace gcd_of_n13_minus_n_l536_536272

theorem gcd_of_n13_minus_n : 
  ∀ n : ℤ, n ≠ 0 → 2730 ∣ (n ^ 13 - n) :=
by sorry

end gcd_of_n13_minus_n_l536_536272


namespace integer_solutions_count_l536_536937

theorem integer_solutions_count (x : ℕ) (h : ⌊Real.sqrt x⌋ = 8) : (finset.Icc 64 80).card = 17 :=
by
  sorry

end integer_solutions_count_l536_536937


namespace least_add_to_divisible_least_subtract_to_divisible_l536_536536

theorem least_add_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (a : ℤ) : 
  n = 1100 → d = 37 → r = n % d → a = d - r → (n + a) % d = 0 :=
by sorry

theorem least_subtract_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (s : ℤ) : 
  n = 1100 → d = 37 → r = n % d → s = r → (n - s) % d = 0 :=
by sorry

end least_add_to_divisible_least_subtract_to_divisible_l536_536536


namespace janet_sharon_oranges_l536_536996

theorem janet_sharon_oranges (janet_oranges sharon_oranges : ℕ) (h₁ : janet_oranges = 9) (h₂ : sharon_oranges = 7) :
  janet_oranges + sharon_oranges = 16 :=
begin
  sorry
end

end janet_sharon_oranges_l536_536996


namespace count_whole_numbers_in_interval_l536_536906

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536906


namespace smallest_percent_increase_l536_536585

theorem smallest_percent_increase 
    (d1 d2 d3 d4 d5 d6 : ℕ) 
    (h1 : d1 = 15) 
    (h2 : d2 = 20) 
    (h3 : d3 = 25) 
    (h4 : d4 = 35) 
    (h5 : d5 = 50) 
    (h6 : d6 = 70) :
    (d2 - d1) / d1 * 100 < (d3 - d2) / d2 * 100 ∧ 
    (d3 - d2) / d2 * 100 < (d4 - d3) / d3 * 100 ∧ 
    (d3 - d2) / d2 * 100 < (d5 - d4) / d4 * 100 ∧ 
    (d3 - d2) / d2 * 100 < (d6 - d5) / d5 * 100 :=
by
    rw [h1, h2, h3, h4, h5, h6]
    unfold Nat.div Nat.mul
    change (20 - 15) / 15 * 100 < (25 - 20)/20 * 100 ∧ 
           (25 - 20) / 20 * 100 < (35 - 25) / 25 * 100 ∧ 
           (25 - 20) / 20 * 100 < (15 / 15) ∨ 
           (15 / 10) ∨ (25 - 15) / (15 / 10) ¹ ℕ
    sorry

end smallest_percent_increase_l536_536585


namespace combinations_of_coins_with_50_cents_l536_536748

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536748


namespace problem_solution_l536_536009

-- Define b_n as stated in the problem
def b (k : ℕ) : ℕ :=
  let a := List.foldl (λ acc x => acc * 10 + x) 0 (List.range (k + 1)).drop 1
  in a - k

-- Define the main theorem
noncomputable def count_b_k_divisible_by_9 : ℕ :=
  (List.range 101).drop 1 |>.count (λ k => b k % 9 == 0)

theorem problem_solution : count_b_k_divisible_by_9 = 22 := by
  sorry

end problem_solution_l536_536009


namespace shift_graph_l536_536526

variable (x : ℝ)

def f (x : ℝ) := 2 * sin (2 * x)
def g (x : ℝ) := sqrt 3 * sin (2 * x) - cos (2 * x)

theorem shift_graph :
  f x = g (x - π / 12) :=
by sorry

end shift_graph_l536_536526


namespace count_whole_numbers_in_interval_l536_536872

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536872


namespace intersection_C1_C2_minimum_distance_C1_C2_l536_536210

-- Define the parametric curve C1
def C1 (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α ^ 2)

-- Define the curve C2 in rectangular coordinates
def C2 (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Define the curve C3 in rectangular coordinates
def C3 (x y : ℝ) : Prop :=
  x ^ 2 + (y - 1) ^ 2 = 1

-- Prove that the intersection point of C1 and C2 is (-1, 0)
theorem intersection_C1_C2 :
  ∃ (α : ℝ), ∃ (x y : ℝ), C1 α = (x, y) ∧ C2 x y ∧ (x, y) = (-1, 0) :=
sorry

-- Prove that the minimum value of |AB| where A is on C1 and B is on C2 is sqrt 2 - 1
theorem minimum_distance_C1_C2 :
  ∀ (A B : ℝ × ℝ), (∃ α : ℝ, A = C1 α) ∧ C2 (B.1) (B.2) → 
  (B.1 + 1 = 0 ∧ B.2 = 0) → abs (dist A B) = Real.sqrt 2 - 1 :=
sorry

end intersection_C1_C2_minimum_distance_C1_C2_l536_536210


namespace number_of_cans_per_set_l536_536590

noncomputable def ice_cream_original_price : ℝ := 12
noncomputable def ice_cream_discount : ℝ := 2
noncomputable def ice_cream_sale_price : ℝ := ice_cream_original_price - ice_cream_discount
noncomputable def number_of_tubs : ℝ := 2
noncomputable def total_money_spent : ℝ := 24
noncomputable def cost_of_juice_set : ℝ := 2
noncomputable def number_of_cans_in_juice_set : ℕ := 10

theorem number_of_cans_per_set (n : ℕ) (h : cost_of_juice_set * n = number_of_cans_in_juice_set) : (n / 2) = 5 :=
by sorry

end number_of_cans_per_set_l536_536590


namespace negation_proposition_l536_536102

theorem negation_proposition {x : ℝ} (h : ∀ x > 0, Real.sin x > 0) : ∃ x > 0, Real.sin x ≤ 0 :=
sorry

end negation_proposition_l536_536102


namespace chickens_and_rabbits_system_l536_536375

variable (x y : ℕ)

theorem chickens_and_rabbits_system :
  (x + y = 16) ∧ (2 * x + 4 * y = 44) ↔ (
    (∃ x y, x + y = 16 ∧ 2 * x + 4 * y = 44) := sorry

end chickens_and_rabbits_system_l536_536375


namespace ratio_of_triangle_areas_l536_536220

theorem ratio_of_triangle_areas 
  (R : ℝ)
  (S₁ S₂ : ℝ) 
  (hS₁ : S₁ = R^2 * (3.sqrt 3 + Real.pi) / 6) 
  (hS₂ : S₂ = R^2 * (3.sqrt 3 - Real.pi) / 6) : 
  S₂ / S₁ = (3.sqrt 3 - Real.pi) / (3.sqrt 3 + Real.pi) := 
sorry

end ratio_of_triangle_areas_l536_536220


namespace find_x_value_l536_536641

theorem find_x_value :
  ∃ (x : ℤ), ∀ (y z w : ℤ), (x = 2 * y + 4) → (y = z + 5) → (z = 2 * w + 3) → (w = 50) → x = 220 :=
by
  sorry

end find_x_value_l536_536641


namespace area_of_square_with_adjacent_points_l536_536084

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def side_length := distance 1 2 4 6
def area_of_square (side : ℝ) : ℝ := side ^ 2

theorem area_of_square_with_adjacent_points :
  area_of_square side_length = 25 :=
by
  unfold side_length
  unfold area_of_square
  sorry

end area_of_square_with_adjacent_points_l536_536084


namespace probability_extreme_value_probability_extreme_value_exists_l536_536305

theorem probability_extreme_value
  (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (∃ x, f' x = 0) ↔ b < 1 / 3 * a^2 :=
begin
  sorry
end

def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x + b

theorem probability_extreme_value_exists 
  (a : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) :
  ∫ a in 0..1, 1 / 3 * a^2 = 1 / 9 :=
begin
  sorry
end

end probability_extreme_value_probability_extreme_value_exists_l536_536305


namespace min_value_inv_sum_l536_536024

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end min_value_inv_sum_l536_536024


namespace correct_number_of_candles_left_l536_536234

noncomputable def candles_left_first_room : Nat := 50 - (2/5 * 50).toNat - (3/7 * (50 - (2/5 * 50).toNat)).toNat - (4/9 * (50 - (2/5 * 50).toNat - (3/7 * (50 - (2/5 * 50).toNat)).toNat)).toNat

noncomputable def candles_left_second_room : Nat := 70 - (3/10 * 70).toNat - (2/5 * (70 - (3/10 * 70).toNat)).toNat - (5/8 * (70 - (3/10 * 70).toNat - (2/5 * (70 - (3/10 * 70).toNat)).toNat)).toNat

noncomputable def candles_left_third_room : Nat := 80 - (1/4 * 80).toNat - (5/12 * (80 - (1/4 * 80).toNat)).toNat - (3/7 * (80 - (1/4 * 80).toNat - (5/12 * (80 - (1/4 * 80).toNat)).toNat)).toNat

theorem correct_number_of_candles_left :
  candles_left_first_room = 10 ∧
  candles_left_second_room = 12 ∧
  candles_left_third_room = 20 :=
by
  sorry

end correct_number_of_candles_left_l536_536234


namespace percentage_reduction_bananas_l536_536574

theorem percentage_reduction_bananas
  (Pr : ℝ)  -- Reduced price per dozen bananas
  (additional_bananas : ℝ)  -- Additional bananas obtained for Rs. 40
  (cost : ℝ)  -- Total cost spent for additional bananas
  (original_price_condition : 40 = additional_bananas / 12 * Pr)  -- Condition for reduced price

  (Pr : ℝ := 3.84)  -- Reduced price per dozen bananas
  (additional_bananas : ℝ := 50)   -- Additional bananas obtained
  (cost : ℝ := 40)  -- Cost for these bananas

  : (let P : ℝ := additional_bananas * Pr / 40 in
     let percentage_reduction : ℝ := (P - Pr) / P * 100 in
     percentage_reduction = 60) :=
by
  sorry

end percentage_reduction_bananas_l536_536574


namespace minimize_expression_l536_536659

theorem minimize_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 30) :
  (a, b) = (15 / 4, 15) ↔ (∀ x y : ℝ, 0 < x → 0 < y → (4 * x + y = 30) → (1 / x + 4 / y) ≥ (1 / (15 / 4) + 4 / 15)) := by
sorry

end minimize_expression_l536_536659


namespace delivery_payment_l536_536436

-- Define the problem conditions and the expected outcome
theorem delivery_payment 
    (deliveries_Oula : ℕ) 
    (deliveries_Tona : ℕ) 
    (difference_in_pay : ℝ) 
    (P : ℝ) 
    (H1 : deliveries_Oula = 96) 
    (H2 : deliveries_Tona = 72) 
    (H3 : difference_in_pay = 2400) :
    96 * P - 72 * P = 2400 → P = 100 :=
by
  intro h1
  sorry

end delivery_payment_l536_536436


namespace UniqueTriangleABC_l536_536231

-- Define the triangles with their specific conditions
structure TriangleA where
  AB BC CA : ℝ
  h1 : AB = 3
  h2 : BC = 4
  h3 : CA = 8

structure TriangleB where
  AB BC : ℝ
  ∠A : ℝ
  h1 : AB = 4
  h2 : BC = 3
  h3 : ∠A = 60

structure TriangleC where
  ∠A ∠B : ℝ
  AB : ℝ
  h1 : ∠A = 60
  h2 : ∠B = 45
  h3 : AB = 4

structure TriangleD where
  ∠C ∠B ∠A : ℝ
  h1 : ∠C = 90
  h2 : ∠B = 30
  h3 : ∠A = 60

theorem UniqueTriangleABC : ∃ (tri : TriangleC), 
  ∀ (triA : TriangleA), ¬∃ triA ∈ {triA} ∧
  ∀ (triB : TriangleB), ¬∃ triB ∈ {triB} ∧ 
  ∀ (triD : TriangleD), ¬∃ triD ∈ {triD} := 
  sorry

end UniqueTriangleABC_l536_536231


namespace angus_caught_4_more_l536_536237

theorem angus_caught_4_more (
  angus ollie patrick: ℕ
) (
  h1: ollie = angus - 7
) (
  h2: ollie = 5
) (
  h3: patrick = 8
) : (angus - patrick) = 4 := 
sorry

end angus_caught_4_more_l536_536237


namespace min_inv_sum_l536_536020

open Real

theorem min_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) :
  min ((1 / x) + (1 / y)) (1 / 3) :=
sorry

end min_inv_sum_l536_536020


namespace f_monotonic_intervals_f_maximum_value_l536_536328

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 8 * x + 4

-- Prove f(x) is increasing on (-∞, -2) and (4, ∞) and decreasing on (-2, 4)
theorem f_monotonic_intervals :
  (∀ x, x < -2 → (f' x > 0)) ∧ 
  (∀ x, -2 < x ∧ x < 4 → (f' x < 0)) ∧ 
  (∀ x, x > 4 → (f' x > 0)) :=
sorry

-- Prove the maximum value of f(x) on [-1, 5] is 32/3
theorem f_maximum_value : ∃ x ∈ set.Icc (-1 : ℝ) (5 : ℝ), ∀ y ∈ set.Icc (-1:ℝ) (5:ℝ), f x ≥ f y ∧ f x = 32/3 :=
sorry

end f_monotonic_intervals_f_maximum_value_l536_536328


namespace abs_diff_eq_l536_536203

theorem abs_diff_eq (a b c d : ℤ) (h1 : a = 13) (h2 : b = 3) (h3 : c = 4) (h4 : d = 10) : 
  |a - b| - |c - d| = 4 := 
  by
  -- Proof goes here
  sorry

end abs_diff_eq_l536_536203


namespace inner_pyramid_edges_greater_l536_536530

open_locale classical

variables (ε : ℝ) (hε_pos : 0 < ε) (hε_small : ε < 1)

def sum_edges_ABCD (ε : ℝ) := 3 + 3 * ε
def sum_edges_ABCD' (ε : ℝ) := 6 * ε + 2

theorem inner_pyramid_edges_greater (ε : ℝ) (hε_pos : 0 < ε) (hε_small : ε < 1) :
  sum_edges_ABCD' ε > sum_edges_ABCD ε :=
by {
  calc
    sum_edges_ABCD' ε = 6 * ε + 2 : by refl
    ... > 3 + 3 * ε : sorry  -- a step that will require detailed calculation
}

end inner_pyramid_edges_greater_l536_536530


namespace problem1_l536_536456

theorem problem1 :
  0.064^(-1 / 3) - (-1 / 8)^0 + 16^(3 / 4) + 0.25^(1 / 2) = 10 :=
by
  sorry

end problem1_l536_536456


namespace area_under_curve_l536_536507

-- Define the vertices of the rectangle
structure Vertices where
  A B C D : (ℝ × ℝ)

-- Define a rotation function (not actually implemented here)
def rotate_clockwise (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the initial and final points
def initial_point := (1, 1 : ℝ × ℝ)
def final_point := (11, 1 : ℝ × ℝ)

-- The main proof statement
theorem area_under_curve : 
  (vertices : Vertices) 
  (rotate1 : rotate_clockwise initial_point (2, 0) = (2, -1)) 
  (rotate2 : rotate_clockwise (2, -1) (5, 0) = (4, 3)) 
  (rotate3 : rotate_clockwise (4, 3) (7, 0) = (10, -3)) 
  (rotate4 : rotate_clockwise (10, -3) (10, 0) = final_point)
  : 
  let radius1 := ((initial_point.1 - 2)^2 + (initial_point.2)^2)^0.5
  let radius2 := sqrt ((4 - 7)^2 + (3 - 0)^2)
  2 * (1/4 * (π * radius1^2) + 1/4 * (π * radius2^2)) + 2 + 4 = (7 / 2) * π + 6
  := sorry

end area_under_curve_l536_536507


namespace man_l536_536215

-- Defining the conditions as variables in Lean
variables (S : ℕ) (M : ℕ)
-- Given conditions
def son_present_age := S = 25
def man_present_age := M = S + 27

-- Goal: the ratio of the man's age to the son's age in two years is 2:1
theorem man's_age_ratio_in_two_years (h1 : son_present_age S) (h2 : man_present_age S M) :
  (M + 2) / (S + 2) = 2 := sorry

end man_l536_536215


namespace coin_combinations_sum_50_l536_536715

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536715


namespace coin_combinations_count_l536_536799

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536799


namespace combinations_of_coins_with_50_cents_l536_536753

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536753


namespace slices_per_pizza_l536_536159

def number_of_people : ℕ := 18
def slices_per_person : ℕ := 3
def number_of_pizzas : ℕ := 6
def total_slices : ℕ := number_of_people * slices_per_person

theorem slices_per_pizza : total_slices / number_of_pizzas = 9 :=
by
  -- proof steps would go here
  sorry

end slices_per_pizza_l536_536159


namespace number_of_integers_in_interval_l536_536922

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536922


namespace ratio_diana_grace_l536_536260

-- Conditions
def grace_age (t: ℕ) := if t = 0 then 3 else 3 + t
def diana_age := 8

-- Theorem statement: ratio of Diana's age to Grace's age equals 2
theorem ratio_diana_grace (t : ℕ) : grace_age 1 = 4 ∧ diana_age = 8 → diana_age / grace_age 1 = 2 :=
by
  intros h
  have grace_age_today := h.1
  have d_age := h.2
  rw [grace_age_today, d_age]
  sorry

end ratio_diana_grace_l536_536260


namespace a_arithmetic_max_sum_b_sum_l536_536300

/-- Definition of the sequence sum S_n --/
def S : ℕ → ℤ := λ n, 33 * n - n ^ 2

/-- Definition of the sequence a_n --/
def a (n : ℕ) : ℤ := S n - S (n - 1)

/-- Definition of the sequence b_n --/
def b (n : ℕ) : ℤ := Int.natAbs (a n)

/-- Sum of the first n terms of sequence b_n --/
def S' : ℕ → ℤ
| n => if n ≤ 17 then 33 * (n : ℤ) - n ^ 2 else n ^ 2 - 33 * (n : ℤ) + 544

/-- Proof that a_n is an arithmetic sequence --/
theorem a_arithmetic : ∃ d a1, ∀ n ≥ 1, a n = a1 + (n - 1) * d :=
by
  sorry

/-- Proof that the maximum sum is at n = 17 and is 272 --/
theorem max_sum : (∃ n, n = 17 ∨ n = 16) ∧ ∀ n, 33 * (n : ℤ) - n ^ 2 < 272 :=
by
  sorry

/-- Proof of the sum of the first n terms of b_n --/
theorem b_sum : ∀ n, S' n = S (if n ≤ 17 then n else 17) := 
by
  sorry

end a_arithmetic_max_sum_b_sum_l536_536300


namespace cos_tan_third_quadrant_l536_536478

theorem cos_tan_third_quadrant (θ : ℝ) : (cos θ < 0 ∧ tan θ > 0) ↔ (∃ k : ℤ, θ = (2 * k + 1) * π + (π / 2)) :=
by
  sorry

end cos_tan_third_quadrant_l536_536478


namespace tangent_line_equation_even_derived_l536_536684

def f (x a : ℝ) : ℝ := x^3 + (a - 2) * x^2 + a * x - 1

def f' (x a : ℝ) : ℝ := 3 * x^2 + 2 * (a - 2) * x + a

theorem tangent_line_equation_even_derived (a : ℝ) (h : ∀ x : ℝ, f' x a = f' (-x) a) :
  5 * 1 - (f 1 a) - 3 = 0 :=
by
  sorry

end tangent_line_equation_even_derived_l536_536684


namespace whole_numbers_in_interval_7_4_3pi_l536_536860

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536860


namespace exp_ineq_solution_set_l536_536278

theorem exp_ineq_solution_set (e : ℝ) (h : e = Real.exp 1) :
  {x : ℝ | e^(2*x - 1) < 1} = {x : ℝ | x < 1 / 2} :=
sorry

end exp_ineq_solution_set_l536_536278


namespace whole_numbers_in_interval_l536_536914

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536914


namespace ω1_eq_ω7_l536_536414

-- Definitions based on the conditions
variables (A1 A2 A3 : Point)
variables (ω1 ω2 ω3 ω4 ω5 ω6 ω7 : Circle)

-- Conditions as Lean statements
axiom triangle_exists : Triangle A1 A2 A3
axiom ω1_through_A1_A2 : ω1 ∋ A1 ∧ ω1 ∋ A2
axiom ωk_through_Ak_Ak1 : ∀ k ∈ {2, 3, 4, 5, 6, 7}, ωk ∋ A_k ∧ ωk ∋ A_{k+1} ∧ (A_i = A_{i+3})
axiom ωkωk1_touch : ∀ k ∈ {1, 2, 3, 4, 5, 6}, touches_externally ωk ω(k+1)

-- The theorem to prove
theorem ω1_eq_ω7 (A1 A2 A3 : Point) (ω1 ω2 ω3 ω4 ω5 ω6 ω7 : Circle)
  (triangle_exists : Triangle A1 A2 A3)
  (ω1_through_A1_A2 : ω1 ∋ A1 ∧ ω1 ∋ A2)
  (ωk_through_Ak_Ak1 : ∀ k ∈ {2, 3, 4, 5, 6, 7}, ωk ∋ A_k ∧ ωk ∋ A_{k+1} ∧ (A_i = A_{i+3}))
  (ωkωk1_touch : ∀ k ∈ {1, 2, 3, 4, 5, 6}, touches_externally ωk ω(k+1)) :
  ω1 = ω7 := 
sorry

end ω1_eq_ω7_l536_536414


namespace find_a_l536_536325

noncomputable def f : ℝ → ℝ :=
  λ x, if x >= 5 then x^2 - x + 12 else 2^x

theorem find_a (a : ℝ) (h : f (f a) = 16) : a = 2 :=
begin
  sorry
end

end find_a_l536_536325


namespace tulips_in_daniels_garden_l536_536506
-- Import all necessary libraries

-- Define the problem statement
theorem tulips_in_daniels_garden
  (initial_ratio_tulips_sunflowers : ℚ := 3 / 7)
  (initial_sunflowers : ℕ := 42)
  (additional_sunflowers : ℕ := 14) :
  let total_sunflowers := initial_sunflowers + additional_sunflowers,
      total_tulips := (initial_ratio_tulips_sunflowers * total_sunflowers) : ℕ
  in total_tulips = 24 :=
by
  sorry  -- Proof should be filled in here

end tulips_in_daniels_garden_l536_536506


namespace final_amoeba_type_l536_536369

theorem final_amoeba_type
  (initial_A : ℕ) (initial_B : ℕ) (initial_C : ℕ)
  (hA : initial_A = 20) (hB : initial_B = 21) (hC : initial_C = 22)
  (fusion_rule : ∀ (a b : ℕ), (a, b) = (initial_A, initial_B) ∨ (a, b) = (initial_A, initial_C) ∨ (a, b) = (initial_B, initial_C) → (initial_A + initial_B + initial_C - 1) = 1) :
  final_amoeba_type = B :=
sorry

end final_amoeba_type_l536_536369


namespace sum_of_first_1234_terms_l536_536383

-- Define the sequence
def seq : ℕ → ℕ
| 0 := 1
| (n + 1) := if n % (2 + seq n) == 1 then 1 else 2

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ :=
(nat.rec_on n 0 (λ n ih, ih + seq n))

-- Define the given conditions and the correct answer
theorem sum_of_first_1234_terms : sum_seq 1234 = 2419 := 
by sorry

end sum_of_first_1234_terms_l536_536383


namespace square_area_adjacency_l536_536049

-- Definition of points as pairs of integers
def Point := ℤ × ℤ

-- Define the points (1,2) and (4,6)
def P1 : Point := (1, 2)
def P2 : Point := (4, 6)

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement for proving the area of a square given the side length
theorem square_area_adjacency (h : distance P1 P2 = 5) : ∃ area : ℝ, area = 25 :=
by
  use 25
  sorry

end square_area_adjacency_l536_536049


namespace complement_A_l536_536959

open Set

variable U : Set ℝ := univ
variable A : Set ℝ := { x | 1 < x ∧ x ≤ 3 }

theorem complement_A :
  compl A = { x : ℝ | x ≤ 1 ∨ x > 3 } := by
  sorry

end complement_A_l536_536959


namespace right_triangle_side_lengths_l536_536595

theorem right_triangle_side_lengths :
  ¬ (4^2 + 5^2 = 6^2) ∧
  (12^2 + 16^2 = 20^2) ∧
  ¬ (5^2 + 10^2 = 13^2) ∧
  ¬ (8^2 + 40^2 = 41^2) := by
  sorry

end right_triangle_side_lengths_l536_536595


namespace distance_between_points_l536_536534

-- Define the points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the points
def point1 : Point3D := { x := 1, y := -3, z := 2 }
def point2 : Point3D := { x := 4, y := 4, z := -1 }

-- Define the 3D distance function
def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

-- Prove that the distance between point1 and point2 is sqrt(67)
theorem distance_between_points :
  distance point1 point2 = Real.sqrt 67 :=
by
  -- Proof goes here
  sorry

end distance_between_points_l536_536534


namespace count_ways_to_get_50_cents_with_coins_l536_536826

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536826


namespace AE_length_l536_536248

-- Definitions of points A, B, C, D
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (7, 0)
def C : ℝ × ℝ := (3, 0)
def D : ℝ × ℝ := (5, 3)

-- Definition of the function to calculate distance between two points
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  ((Q.1 - P.1)^2 + (Q.2 - P.2)^2).sqrt

-- Equation for length AE in the context of intersection point E of lines connecting AB and CD
noncomputable def length_AE (A B C D : ℝ × ℝ) (E : ℝ × ℝ) : ℝ := 
  dist A E

-- Hypothesis that AE is equal to the calculated result
theorem AE_length :
  ∃ E : ℝ × ℝ, 
    (E.1, E.2) ∈ ({(x, y) | (y - A.2) / (x - A.1) = (y - B.2) / (x - B.1)} ∩ {(x, y) | (y - C.2) / (x - C.1) = (y - D.2) / (x - D.1)}) →
    length_AE A B C D E = 7 * (dist A B) / 13 :=
begin
  sorry
end

end AE_length_l536_536248


namespace number_of_triangles_l536_536256

theorem number_of_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) :
  let total_points := 25, total_combinations := Nat.choose total_points 3 in
  let invalid_rows_cols := 100, invalid_diagonals := 24 in
  let valid_triangles := total_combinations - (invalid_rows_cols + invalid_diagonals) in
  valid_triangles = 2176 :=
by
  sorry

end number_of_triangles_l536_536256


namespace scientific_notation_l536_536458

theorem scientific_notation (a : ℕ) (b : ℝ) : 
  a = 113800 ∧ b = 1.138 → a = b * 10^5 :=
by
  intros h
  cases h with ha hb
  sorry

end scientific_notation_l536_536458


namespace triangle_area_ratio_l536_536380

theorem triangle_area_ratio
    (h1: inscribed_triangle ABC)
    (h2: on_arc D (arc AC))
    (h3: arc_measure DC = 30)
    (h4: on_arc G (arc BA))
    (h5: arc_measure BG > arc_measure GA)
    (h6: length AB = length AC)
    (h7: length AB = length DG)
    (h8: angle CAB = 30)
    (h9: chord_intersects_sides DG AC AB E F) :
    area_ratio (triangle AFE) (triangle ABC) = 7 * sqrt 3 - 12 :=
sorry

end triangle_area_ratio_l536_536380


namespace minimum_value_y_l536_536187

noncomputable def y (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_y (x : ℝ) (hx : x > 1) : ∃ A, (A = 3) ∧ (∀ y', y' = y x → y' ≥ A) := sorry

end minimum_value_y_l536_536187


namespace parabola_equation_l536_536636

theorem parabola_equation (a b c : ℝ) (x y : ℝ) :
  (∀ x y : ℝ, y = a * (x - 3)^2 + 5 → (0, 2) ∈ {(x, y)} → a = -1/3) →
  y = -1/3 * x^2 + 2 * x + 2 :=
by
  sorry

end parabola_equation_l536_536636


namespace whole_numbers_in_interval_l536_536909

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536909


namespace money_received_by_a_l536_536546

variables (total_profit : ℝ) (management_fee_percentage : ℝ)
          (capital_a : ℝ) (capital_b : ℝ) (profit_received_by_a : ℝ)

-- Conditions
def is_working_partner (a : Prop) : Prop := a
def is_sleeping_partner (b : Prop) : Prop := b
def capital_contributed (a_contrib b_contrib : ℝ) : Prop := 
  a_contrib = 3500 ∧ b_contrib = 2500
def management_fee (fee_percentage : ℝ) : Prop :=
  fee_percentage = 0.1
def total_profit_val (profit : ℝ) : Prop :=
  profit = 9600
def profit_sharing_proportion (capital_a capital_b : ℝ) (remaining_profit : ℝ) : Prop :=
  remaining_profit = total_profit - management_fee_percentage * total_profit ∧
  (capital_a / (capital_a + capital_b)) * remaining_profit +
  management_fee_percentage * total_profit = profit_received_by_a

-- Theorem statement
theorem money_received_by_a
  (h1 : is_working_partner True)
  (h2 : is_sleeping_partner True)
  (h3 : capital_contributed capital_a capital_b)
  (h4 : management_fee management_fee_percentage)
  (h5 : total_profit_val total_profit)
  (h6 : profit_sharing_proportion capital_a capital_b profit_received_by_a) :
  profit_received_by_a = 6000 :=
sorry

end money_received_by_a_l536_536546


namespace decreased_cost_proof_l536_536480

def original_cost : ℝ := 200
def percentage_decrease : ℝ := 0.5
def decreased_cost (original_cost : ℝ) (percentage_decrease : ℝ) : ℝ := 
  original_cost - (percentage_decrease * original_cost)

theorem decreased_cost_proof : decreased_cost original_cost percentage_decrease = 100 := 
by { 
  sorry -- Proof is not required
}

end decreased_cost_proof_l536_536480


namespace find_f_l536_536649

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) + 2 * (deriv^[2] f 0) * x

theorem find_f'''_at_neg1 :
  deriv^[3] f (-1) = 2 - Real.exp 1 :=
sorry

end find_f_l536_536649


namespace book_reading_average_l536_536569

theorem book_reading_average :
  ∀ (total_pages read_pages days_remaining : ℕ),
    total_pages = 212 →
    read_pages = 97 →
    days_remaining = 5 →
    (total_pages - read_pages) / days_remaining = 25 := by
  intros total_pages read_pages days_remaining
  assume h1 : total_pages = 212
  assume h2 : read_pages = 97
  assume h3 : days_remaining = 5
  sorry

end book_reading_average_l536_536569


namespace largest_possible_value_unbounded_l536_536662

theorem largest_possible_value_unbounded (x y : ℝ) (h1 : -3 ≤ x) (h2 : x ≤ 1) (h3 : 1 ≤ y) (h4 : y ≤ 3) :
  ∀ M : ℝ, ∃ ε > 0, x ∈ set.Ioo ε 1 → (1 + (y + 1) / x > M) :=
by
  sorry

end largest_possible_value_unbounded_l536_536662


namespace coin_combinations_count_l536_536743

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536743


namespace coin_combination_l536_536788

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536788


namespace combination_coins_l536_536815

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536815


namespace avg_children_nine_families_l536_536167

theorem avg_children_nine_families
  (total_families : ℕ)
  (average_children : ℕ)
  (childless_families : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  total_children = total_families * average_children →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℝ) = 4.0 :=
begin
  intros,
  sorry
end

end avg_children_nine_families_l536_536167


namespace triangles_with_positive_area_l536_536350

theorem triangles_with_positive_area :
  let points := {p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5} in
  ∃ (n : ℕ), n = 2150 ∧ 
    (∃ (triangles : set (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)), 
      (∀ t ∈ triangles, 
        t.1 ∈ points ∧ t.2 ∈ points ∧ t.3 ∈ points ∧ 
        (∃ (area : ℚ), area > 0 ∧ 
          area = ((t.2.1 - t.1.1) * (t.3.2 - t.1.2) - (t.3.1 - t.1.1) * (t.2.2 - t.1.2)) / 2)) ∧ 
      ∃ (card_tris : ℕ), card_tris = n) :=
sorry

end triangles_with_positive_area_l536_536350


namespace problem_AD_l536_536003

variable (AB BC AM CD CM AD : ℝ)
variable (M A B C D : Type) [MetricSpace M]
variable (f : M → ℝ) [Fact (AB = 2)] [Fact (BC = 5)] [Fact (AM = 4)] [Fact (CD / CM = 0.6)]
variable [Fact (ABCD : cyclic_quad A B C D)]

theorem problem_AD :
  AB = 2 →
  BC = 5 →
  AM = 4 →
  CD / CM = 0.6 →
  AD = 2 :=
by
  intros hab hbc ham hcdcm
  sorry

end problem_AD_l536_536003


namespace xy_conditions_l536_536947

theorem xy_conditions (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : x * y = 1) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end xy_conditions_l536_536947


namespace coin_combinations_50_cents_l536_536726

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536726


namespace combinations_of_coins_with_50_cents_l536_536754

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536754


namespace range_of_m_l536_536686

theorem range_of_m (m : ℝ) : 
  (∃ x ∈ Icc (-(π / 3)) (π / 3), 2 * sin x + (3 * sqrt 3) / π * x + m = 0) ↔ 
  m ∈ Icc (-2 * sqrt 3) (2 * sqrt 3) := 
sorry

end range_of_m_l536_536686


namespace combination_coins_l536_536811

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536811


namespace count_ways_to_get_50_cents_with_coins_l536_536831

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536831


namespace apron_sewing_ratio_l536_536711

def total_aprons : ℕ := 150
def aprons_before_today : ℕ := 13
def aprons_tomorrow : ℕ := 49
def aprons_needed_for_half_remaining : ℕ := 98 -- Since 49 aprons tomorrow implies half remaining is 98

theorem apron_sewing_ratio :
  let aprons_today := total_aprons - aprons_needed_for_half_remaining - aprons_before_today in
  aprons_today = 39 → 
  (aprons_today : ℚ) / aprons_before_today = 3 :=
by
  intros
  exact sorry

end apron_sewing_ratio_l536_536711


namespace geometric_series_sum_l536_536242

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (-1 / 4 : ℚ)
  let n := 6
  let sum := a * ((1 - r ^ n) / (1 - r))
  sum = (4095 / 5120 : ℚ) :=
by
  -- Proof goes here
  sorry

end geometric_series_sum_l536_536242


namespace jacob_lunch_calories_l536_536394

theorem jacob_lunch_calories (l : ℕ) 
  (planned_intake : ℕ = 1800)
  (breakfast_calories : ℕ = 400) 
  (dinner_calories : ℕ = 1100) 
  (extra_calories : ℕ = 600) :
  let total_intake := planned_intake + extra_calories in
  let food_intake := breakfast_calories + l + dinner_calories in
  l = 900 := 
by
  let total_intake := planned_intake + extra_calories
  let food_intake := breakfast_calories + l + dinner_calories
  have h₁ : total_intake = 2400 := by
    simp [planned_intake, extra_calories, show 1800 + 600 = 2400 from rfl]
  have h₂ : breakfast_calories + dinner_calories = 1500 := by
    simp [show 400 + 1100 = 1500 from rfl]
  have h₃ : total_intake = food_intake := by
    rw [h₁, show l + 1500 = 2400 from rfl, ← h₂]
  have h₄ : l + 1500 = 2400 := by
    rw [← h₂, ← h₃]
  have h₅ : l = 2400 - 1500 := by 
    rw [show l + 1500 - 1500 = 2400 - 1500 from rfl]
  exact show l = 900 from rfl

end jacob_lunch_calories_l536_536394


namespace export_volume_equation_l536_536387

variable (x : ℕ)

-- Statements and definitions
def export_volume_2023 := 107
def export_volume_relation := 4 * x + 3

-- Theorem to prove
theorem export_volume_equation (hx : export_volume_relation = export_volume_2023) : 
  4 * x + 3 = 107 := by
  rw [export_volume_relation, export_volume_2023] at hx
  exact hx

end export_volume_equation_l536_536387


namespace exists_arithmetic_progressions_covering_naturals_l536_536263

theorem exists_arithmetic_progressions_covering_naturals :
  ∃ (N : ℕ), N = 12 ∧ ∀ (n : ℕ), ∃ (d : ℕ), d ∈ {2, 3, 4, ..., 12} ∧ ∃ (k : ℕ), n = k * d := 
begin
  -- Proof will go here
  sorry
end

end exists_arithmetic_progressions_covering_naturals_l536_536263


namespace problem_l536_536308

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

theorem problem
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_nonzero : ∀ x, f x ≠ 0)
  (h_condition : ∀ x, x * f(x + 1) = (x + 1) * f(x)) :
  f (f (5 / 2)) = 0 :=
sorry

end problem_l536_536308


namespace coin_combinations_l536_536800

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536800


namespace count_whole_numbers_in_interval_l536_536850

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536850


namespace locus_of_centers_of_rectangles_l536_536271

open_locale classical

variables {A B C O M H P Q R S D E : Type} [MetricSpace A] 
  [MetricSpace B] [MetricSpace C] [MetricSpace O] 
  [MetricSpace M] [MetricSpace H] [MetricSpace P] 
  [MetricSpace Q] [MetricSpace R] [MetricSpace S] 
  [MetricSpace D] [MetricSpace E]

-- Define triangle and its properties
variable (triangle_ABC : Triangle A B C)
variable (rectangle_PQRS : Rectangle P Q R S)

-- Define midpoints and perpendiculars
variable (O_is_midpoint : Midpoint O (Altitude CH))
variable (M_is_midpoint : Midpoint M (Segment AB))
variable (D_is_midpoint : Midpoint D (Segment RQ))
variable (E_is_midpoint : Midpoint E (Segment PS))

-- Define the locus problem
theorem locus_of_centers_of_rectangles :
  geometric_locus (center_of rectangle_PQRS) with_side_on (Segment AB) == Segment OM - {O, M} :=
sorry

end locus_of_centers_of_rectangles_l536_536271


namespace count_whole_numbers_in_interval_l536_536894

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536894


namespace exterior_angle_BAC_is_54_l536_536587

def regular_decagon : Type := sorry
def square : Type := sorry
def coplanar (A B : Type) : Prop := sorry
def common_side (A B : Type) (AD : Type) : Prop := sorry
def interior_angle (A : Type) : ℝ := 144
def exterior_angle (θ : ℝ) : ℝ := 360 - θ
def angle_BAD : ℝ := exterior_angle (interior_angle regular_decagon)
def angle_CAD : ℝ := 90
def angle_BAC : ℝ := 360 - angle_BAD - angle_CAD

theorem exterior_angle_BAC_is_54
  (dec : regular_decagon)
  (sqr : square)
  (A B C D : ℝ)
  (h1 : coplanar sqr dec)
  (h2 : common_side sqr dec AD)
  : angle_BAC = 54 := by
  sorry

end exterior_angle_BAC_is_54_l536_536587


namespace count_ways_to_get_50_cents_with_coins_l536_536827

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536827


namespace coin_combinations_l536_536810

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536810


namespace train_crossing_time_l536_536393

theorem train_crossing_time 
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (train_length = 250)
  (train_speed_kmph = 120):
  let train_speed_mps := train_speed_kmph * (1000 / 3600) in
  let time := train_length / train_speed_mps in
  time = 7.5 :=
by 
  sorry

end train_crossing_time_l536_536393


namespace union_M_N_l536_536336

noncomputable def M := {x : ℝ | x^2 + 3 * x + 2 < 0}
noncomputable def N := {x : ℝ | (1 / 2)^x ≤ 4}

theorem union_M_N : ∀ x, x ∈ M ∪ N ↔ x ≥ -2 := by
  sorry

end union_M_N_l536_536336


namespace coin_combination_l536_536779

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536779


namespace combinations_of_coins_l536_536768

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536768


namespace length_BD_l536_536389

-- Define the triangle and its properties
variables (A B C D E F : Type*)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited F]

-- Length of sides in triangle ABC where ∠C is 90°
structure triangle :=
(AC BC : ℝ)
(angle_C : AC ≠ 0 ∧ BC ≠ 0)

-- Points on the sides of the triangle
structure points_on_sides :=
(D : Type*) (on_AB : D)
(E : Type*) (on_BC : E)
(F : Type*) (on_AC : F)

-- Given conditions
noncomputable def given_conditions : triangle × points_on_sides :=
⟨ { AC := 5, BC := 12, angle_C := ⟨5 ≠ 0 ∧ 12 ≠ 0⟩ },
  { D := D, on_AB := on_AB, 
    E := E, on_BC := on_BC, 
    F := F, on_AC := on_AC } ⟩

-- Prove the length BD
theorem length_BD (t : triangle) (p : points_on_sides) (AC BC : ℝ) (angle_C : AC ≠ 0 ∧ BC ≠ 0)
  (DE DF : ℝ) (h1 : DE = 5) (h2 : DF = 3) :
  let AB := real.sqrt (AC^2 + BC^2) in
  let similarity_condition := DF / DE = 3 / 5 in
  let BD := AB - (AC * (DF / DE)) in
  BD = 10 :=
sorry -- Proof to be filled in

end length_BD_l536_536389


namespace distinct_results_for_exponentiation_l536_536238

theorem distinct_results_for_exponentiation : 
  ∃ (distinct_results : Finset ℕ), 
    (∀ (expr : String), 
      expr ∈ {"3 ↑ (3 ↑ (3 ↑ 3))", 
              "3 ↑ ((3 ↑ 3) ↑ 3)", 
              "((3 ↑ 3) ↑ 3) ↑ 3", 
              "(3 ↑ (3 ↑ 3)) ↑ 3", 
              "(3 ↑ 3) ↑ (3 ↑ 3)"} →
          expr.eval ≠ none →
          expr.eval ∈ distinct_results) ∧ 
    distinct_results.card = 4 :=
sorry

end distinct_results_for_exponentiation_l536_536238


namespace coin_combinations_count_l536_536795

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536795


namespace alice_number_l536_536594

theorem alice_number (m : ℕ) 
  (h1 : 180 ∣ m) 
  (h2 : 240 ∣ m) 
  (h3 : 2000 ≤ m ∧ m ≤ 5000) : 
    m = 2160 ∨ m = 2880 ∨ m = 3600 ∨ m = 4320 := 
sorry

end alice_number_l536_536594


namespace coin_combinations_50_cents_l536_536724

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536724


namespace number_of_valid_triples_l536_536249

theorem number_of_valid_triples : 
  ∃! (abc : ℕ × ℕ × ℕ), 
       let (a, b, c) := abc in 
       a ≥ b ∧ b ≥ c ∧ c ≥ 1 ∧ (a * b * c = 2 * (a * b + a * c + b * c)) :=
sorry

end number_of_valid_triples_l536_536249


namespace exists_two_disjoint_paths_l536_536364

variable {Intersection : Type}
variable [Nonempty Intersection]
variable [DecidableEq Intersection]

notation "Path" := List Intersection

-- A function that determines if a path avoids a specific intersection
def avoids (p : Path) (c : Intersection) : Prop :=
  c ∉ p.tail

-- Condition 1: For any three intersections A, B, C, there exists a path from A to B avoiding C.
axiom exists_path_avoiding :
  ∀ (A B C : Intersection), ∃ p : Path, p.head = A ∧ p.reverse.head = B ∧ avoids p C

-- Main statement: From any intersection to any other, there are at least two disjoint paths.
theorem exists_two_disjoint_paths :
  ∀ (A B : Intersection), ∃ (P₁ P₂ : Path), (P₁.head = A ∧ P₁.reverse.head = B) ∧ (P₂.head = A ∧ P₂.reverse.head = B) ∧
  ∀ (x : Intersection), x ≠ A ∧ x ≠ B → (x ∈ P₁ → x ∉ P₂) ∧ (x ∈ P₂ → x ∉ P₁)
:= sorry

end exists_two_disjoint_paths_l536_536364


namespace a_10_correct_l536_536252

noncomputable def a : ℕ → ℚ
| 0       := 1
| (n + 1) := (7 / 4) * a n + (5 / 4) * real.sqrt (3^n - (a n)^2)

theorem a_10_correct : a 10 = 11907 / 16 := by
  sorry

end a_10_correct_l536_536252


namespace pyramid_volume_eq_4_l536_536558

def volume_of_pyramid (a b c : ℝ) (A B C G : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * dist A B * dist B C * dist C G

theorem pyramid_volume_eq_4 : 
  ∀ (a b c : ℝ) (A B C G : ℝ × ℝ × ℝ), 
    a = 2 → b = 3 → c = 4 →
    A = (0, 0, 0) → B = (2, 0, 0) → C = (2, 3, 0) → G = (2, 3, 4) →
    volume_of_pyramid a b c A B C G = 4 :=
by
  intros
  simp [volume_of_pyramid]
  sorry

end pyramid_volume_eq_4_l536_536558


namespace a_formula_f_n_inequality_l536_536333

-- Definitions based on the given conditions
def f_n (a : ℕ → ℤ) (n : ℕ) (x : ℚ) : ℚ :=
  finset.range n.succ.sum (λ i, a i * x ^ (i + 1))

axiom f_n_neg_one (a : ℕ → ℤ) (n : ℕ) : f_n a n (-1) = (-1) ^ n * n

-- Problem 1: Find a_1, a_2, and a_3
def a_1 := 1
def a_2 := 3
def a_3 := 5

-- Problem 2: Prove general term formula for a_n
def a (n : ℕ) : ℤ := 2 * n - 1

theorem a_formula (n : ℕ) : a n = 2 * n - 1 := by
  sorry

-- Problem 3: Prove the inequality
theorem f_n_inequality (a : ℕ → ℤ) (n : ℕ) : 
  let fn := f_n a n
  in 1 / 3 ≤ fn (1 / 3) ∧ fn (1 / 3) < 1 := by
  sorry

end a_formula_f_n_inequality_l536_536333


namespace count_whole_numbers_in_interval_l536_536841

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l536_536841


namespace find_a5_l536_536671

variables {a : ℕ → ℝ}  -- represent the arithmetic sequence

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom a3_a8_sum : a 3 + a 8 = 22
axiom a6_value : a 6 = 8
axiom arithmetic : is_arithmetic_sequence a

-- Target proof statement
theorem find_a5 (a : ℕ → ℝ) (arithmetic : is_arithmetic_sequence a) (a3_a8_sum : a 3 + a 8 = 22) (a6_value : a 6 = 8) : a 5 = 14 :=
by {
  sorry
}

end find_a5_l536_536671


namespace lucky_larry_e_value_l536_536428

theorem lucky_larry_e_value : 
  ∃ e : ℚ, 
    let a := 2
    let b := 3
    let c := 4 
    let d := 6 in
    (a * b + c * d - c * e + 1 = a * (b + (c * (d - e)))) ∧ e = 23 / 4 :=
begin
  use 23 / 4,
  sorry -- Proof to be filled in.
end

end lucky_larry_e_value_l536_536428


namespace trip_time_maple_to_oak_l536_536573

noncomputable def total_trip_time (d1 d2 v1 v2 t_break : ℝ) : ℝ :=
  (d1 / v1) + t_break + (d2 / v2)

theorem trip_time_maple_to_oak : 
  total_trip_time 210 210 50 40 0.5 = 5.75 :=
by
  sorry

end trip_time_maple_to_oak_l536_536573


namespace tetrahedral_vectors_equal_angles_l536_536287

-- Define 3D vector and point origin O
variable (O : ℝ × ℝ × ℝ := (0, 0, 0))

-- Define four vectors from point O to points A1, A2, A3, A4
structure Vectors :=
  (A1 A2 A3 A4 : ℝ × ℝ × ℝ)

-- Indicate that all vectors form equal angles with each other
def EqualAngles (v : Vectors) : Prop :=
  let OA1 := v.A1
  let OA2 := v.A2
  let OA3 := v.A3
  let OA4 := v.A4
  -- Placeholder for actual angle calculation - this needs geometric properties
  -- and vector dot product relations
  sorry

-- The main theorem stating the required angle
theorem tetrahedral_vectors_equal_angles :
  ∀ {v : Vectors}, EqualAngles v → ∀ (i j : ℕ) (h : 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 ∧ i ≠ j), 
    let angle := -- Calculation involving acos returned angle from dot product properties
      sorry 
    angle ≈ 109.4712 :=
by
  sorry

end tetrahedral_vectors_equal_angles_l536_536287


namespace minimum_value_frac_inv_is_one_third_l536_536019

noncomputable def min_value_frac_inv (x y : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ x + y = 12 then 1/x + 1/y else 0

theorem minimum_value_frac_inv_is_one_third (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x + y = 12) :
  min_value_frac_inv x y = 1/3 :=
begin
  -- Proof to be provided
  sorry
end

end minimum_value_frac_inv_is_one_third_l536_536019


namespace combinations_of_coins_with_50_cents_l536_536745

def coins : Type := ℕ × ℕ × ℕ × ℕ -- (number of pennies, number of nickels, number of dimes, number of quarters)

def value (c : coins) : ℕ :=
  match c with
  | (p, n, d, q) => p * 1 + n * 5 + d * 10 + q * 25 -- total value based on coin counts

-- The main theorem:
theorem combinations_of_coins_with_50_cents :
  {c : coins // value c = 50}.card = 16 :=
sorry

end combinations_of_coins_with_50_cents_l536_536745


namespace max_g_achieved_l536_536400

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt (x * (100 - x)) + Real.sqrt (x * (5 - x))

theorem max_g_achieved :
  ∃ x₁ N, (0 ≤ x₁ ∧ x₁ ≤ 5) ∧ g x₁ = N ∧ 
  (∀ x ∈ Icc 0 5, g x ≤ N) ∧ x₁ = 100 / 21 ∧ N = 10 * Real.sqrt 5 := by
  sorry

end max_g_achieved_l536_536400


namespace minimum_perimeter_of_triangle_l536_536962

theorem minimum_perimeter_of_triangle 
  (a b c : ℕ) -- side lengths of triangle
  (h1 : a + b > c) -- Triangle inequality
  (h2 : b + c > a) 
  (h3 : c + a > b) 
  (h4 : b = c) -- $XY = XZ$
  (incenter_excircle_tangent_conditions : -- complex condition about incenter and excircles
    ∀ (r R : ℝ), 
    -- condition specifying the tangency of incenter to excircle
    r > 0 ∧ R > r ∧ 
    (one_excircle_tangent : -- excircle tangent to side
      ∃ t : ℝ, 
      t = r + R ∧ t > 0) ∧
    (other_excircles_tangent : -- other two excircles tangent to extensions
      ∀ (R' : ℝ), 
      R' > r ∧ R' + R = 2 * R)) :
  (a + b + c) = 20 := 
by sorry

end minimum_perimeter_of_triangle_l536_536962


namespace coin_combinations_50_cents_l536_536732

theorem coin_combinations_50_cents :
  let P := 1
  let N := 5
  let D := 10
  let Q := 25
  ∃ p n d q : ℕ, p * P + n * N + d * D + q * Q = 50 :=
  ∃ p n d q : ℕ, (p + 5 * n + 10 * d + 25 * q = 50) :=
sorry

end coin_combinations_50_cents_l536_536732


namespace distance_between_homes_l536_536431

-- Definitions based on conditions
def maxwell_speed : ℝ := 4 -- km/h
def brad_speed : ℝ := 6 -- km/h
def maxwell_time : ℝ := 6 -- hours
def brad_time : ℝ := 5 -- hours -- (Maxwell walks for 6 hours, Brad starts one hour later, so Brad runs for 5 hours)

-- Proven statement
theorem distance_between_homes :
  let D := maxwell_speed * maxwell_time + brad_speed * brad_time in
  D = 54 := 
by
  sorry

end distance_between_homes_l536_536431


namespace solution_l536_536114

def solve_for_x (x : ℝ) : Prop :=
  7 + 3.5 * x = 2.1 * x - 25

theorem solution (x : ℝ) (h : solve_for_x x) : x = -22.857 :=
by
  sorry

end solution_l536_536114


namespace square_area_from_points_l536_536062

theorem square_area_from_points :
  let P1 := (1, 2)
  let P2 := (4, 6)
  let side_length := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let area := side_length^2
  P1.1 = 1 ∧ P1.2 = 2 ∧ P2.1 = 4 ∧ P2.2 = 6 →
  area = 25 :=
by
  sorry

end square_area_from_points_l536_536062


namespace max_value_of_sumsqrt_l536_536036

theorem max_value_of_sumsqrt (a b c : ℝ) (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) (h3 : a + b + c = 8) :
  sqrt (3 * a + 2) + sqrt (3 * b + 2) + sqrt (3 * c + 2) ≤ 3 * sqrt 26 :=
by
  sorry

end max_value_of_sumsqrt_l536_536036


namespace min_value_zero_l536_536158

noncomputable def f (k x y : ℝ) : ℝ :=
  3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

theorem min_value_zero (k : ℝ) :
  (∀ x y : ℝ, f k x y ≥ 0) ↔ (k = 3 / 2 ∨ k = -3 / 2) :=
by
  sorry

end min_value_zero_l536_536158


namespace count_ways_to_get_50_cents_with_coins_l536_536829

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536829


namespace intersection_of_lines_l536_536637

theorem intersection_of_lines :
  ∃ (x y : ℝ), 10 * x - 5 * y = 5 ∧ 8 * x + 2 * y = 22 ∧ x = 2 ∧ y = 3 := by
  sorry

end intersection_of_lines_l536_536637


namespace right_angled_triangle_not_necessarily_axisymmetric_l536_536235

-- Define the concept of an axisymmetric figure
def is_axisymmetric (figure : Type) : Prop := 
  ∃ axis : figure, ∀ points : figure, points = axis -- Simplified representation

-- Define the figures
structure LineSegment
structure RightAngle
structure RightAngledTriangle
structure IsoscelesTriangle

-- Conditions
def condition_A : Prop := is_axisymmetric LineSegment
def condition_B : Prop := is_axisymmetric RightAngle
def condition_C : Prop := ¬is_axisymmetric RightAngledTriangle -- Our key condition for the proof
def condition_D : Prop := is_axisymmetric IsoscelesTriangle

theorem right_angled_triangle_not_necessarily_axisymmetric :
  ¬is_axisymmetric RightAngledTriangle := 
by admittedly sorry

end right_angled_triangle_not_necessarily_axisymmetric_l536_536235


namespace abs_c_eq_1106_l536_536118

noncomputable def f (a b c : ℤ) (x : ℂ) : ℂ :=
  a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_eq_1106
  (a b c : ℤ)
  (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_eq : f a b c (3 + Complex.i) = 0) :
  |c| = 1106 := sorry

end abs_c_eq_1106_l536_536118


namespace right_triangle_hypotenuse_length_l536_536677

theorem right_triangle_hypotenuse_length :
  ∀ (a b : ℝ), a = 1 → b = 3 → (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = Real.sqrt 10) :=
begin
  intros a b ha hb,
  use Real.sqrt 10,
  split,
  { rw [ha, hb],
    simp,
    norm_num },
  { refl }
end

end right_triangle_hypotenuse_length_l536_536677


namespace min_inv_sum_l536_536023

open Real

theorem min_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) :
  min ((1 / x) + (1 / y)) (1 / 3) :=
sorry

end min_inv_sum_l536_536023


namespace P_intersect_Q_empty_l536_536702

def is_element_of_P (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 4

def is_element_of_Q (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 2

theorem P_intersect_Q_empty : ∀ x, is_element_of_P x → is_element_of_Q x → false :=
by
  intro x hP hQ
  sorry

end P_intersect_Q_empty_l536_536702


namespace angle_BDA_is_correct_l536_536390

-- Definitions 
variables {A B C D E : Type} [euclidean_geometry A B C D E]

def is_on_line_segment {A B C : Type} (x : A) (y : B) (z : C) : Prop :=
  collinear x y z ∧ dist x y + dist y z = dist x z

def trisect {A B C D E : Type} (x : A) (y : B) (z : C) (w : D) (v : E) : Prop :=
  is_on_line_segment x y z ∧ is_on_line_segment y w v ∧ colem x y w ∧ ∃ k, dist y w = k ∧ dist w v = k

def bisects_angle {A B D: Type} [has_angle A B D] (u : D) : Prop :=
  angle A u B = angle B u D

-- Given conditions
variables (ABC_trisected : trisect A B C D E)
          (AD_bisects_angle : bisects_angle A D C)
          (angle_BAC x : ℝ)
          (angle_ABC y : ℝ)

-- The theorem statement to prove equivalence
theorem angle_BDA_is_correct : angle B D A = 90 - (x / 4) :=
begin
  sorry
end

end angle_BDA_is_correct_l536_536390


namespace count_whole_numbers_in_interval_l536_536875

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536875


namespace katherine_time_20_l536_536601

noncomputable def time_katherine_takes (k : ℝ) :=
  let time_naomi_takes_per_website := (5/4) * k
  let total_websites := 30
  let total_time_naomi := 750
  time_naomi_takes_per_website = 25 ∧ k = 20

theorem katherine_time_20 :
  ∃ k : ℝ, time_katherine_takes k :=
by
  use 20
  sorry

end katherine_time_20_l536_536601


namespace blondes_range_l536_536047

-- Definitions corresponding to the conditions provided
constant knights_liars : ℕ → ℕ → Prop -- true if the number of knights and liars sum up to total population
constant blondes_brunettes : ℕ → ℕ → Prop -- true if the number of blondes and brunettes sum up to total population

-- The main theorem corresponding to the mathematically equivalent proof problem
theorem blondes_range : ∀ (knights liars blondes brunettes : ℕ),
  knights_liars knights liars ∧ blondes_brunettes blondes brunettes ∧
  ((knights + liars = 200) ∧
  ((knights - brunettes) * ℕ.ofNat (liars - blondes) > (knights + liars) / 2))
  → (93 ≤ blondes ∧ blondes ≤ 107) :=
sorry

end blondes_range_l536_536047


namespace jacket_price_restore_l536_536499

theorem jacket_price_restore (p : ℝ) (h : p > 0) :
  let p1 := p - 0.1 * p,
      p2 := p1 - 0.15 * p1,
      p3 := p2 - 0.25 * p2,
      required_increase := (p - p3) / p3
  in required_increase ≈ 0.743 :=
by
  sorry

end jacket_price_restore_l536_536499


namespace pennies_added_per_compartment_l536_536108

theorem pennies_added_per_compartment (pennies_initial each_compartments compartments total_pennies final_pennies : ℕ) 
    (h1 : pennies_initial = 2)
    (h2 : compartments = 12)
    (h3 : final_pennies = 96)
    (h4 : total_pennies = pennies_initial * compartments)
    (h5 : final_pennies - total_pennies) / compartments = 6) 
    (h6 : final_pennies = total_pennies + 12 * pennies_initial):
  pennies_initial = 6 :=
by 
  sorry

end pennies_added_per_compartment_l536_536108


namespace min_value_reciprocal_sum_l536_536030

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end min_value_reciprocal_sum_l536_536030


namespace combinations_of_coins_l536_536775

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536775


namespace coin_combination_l536_536787

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536787


namespace total_weight_l536_536575

-- Define the weights of almonds and pecans.
def weight_almonds : ℝ := 0.14
def weight_pecans : ℝ := 0.38

-- Prove that the total weight of nuts is 0.52 kilograms.
theorem total_weight (almonds pecans : ℝ) (h_almonds : almonds = 0.14) (h_pecans : pecans = 0.38) :
  almonds + pecans = 0.52 :=
by
  sorry

end total_weight_l536_536575


namespace shaded_area_fraction_l536_536589

-- Definitions based on the problem conditions
variable (square : Type)

-- Defining the fraction of shaded area in the entire square
def fractional_shaded_area : ℕ → ℝ 
| 0     := 1/4  -- Initial shaded area
| (n+1) := (1/16) * fractional_shaded_area n  -- Geometric series reduction

-- Infinite series sum for the shaded area
def total_shaded_area := (1/4) * (1 / (1 - 1/16))

-- The theorem to prove the fractional shaded area is 4/15
theorem shaded_area_fraction : 
  total_shaded_area = 4 / 15 := 
by 
  sorry

end shaded_area_fraction_l536_536589


namespace determine_c_value_l536_536628

theorem determine_c_value :
  ∃ c, (∀ x : ℝ, x ∈ Ioo (-5 / 2) 3 → x * (3 * x + 1) < c) ∧
       (∀ x : ℝ, x ∉ Ioo (-5 / 2) 3 → x * (3 * x + 1) ≥ c) ∧
       c = 30 :=
by
  sorry

end determine_c_value_l536_536628


namespace inequality_proof_l536_536650

theorem inequality_proof 
  (m : ℕ) (a : ℕ → ℝ) 
  (p q r : ℝ) (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ m → a i > 0) 
  (h2 : ∑ i in finset.range m, a (i + 1) = p)
  (h3 : q > 0) (h4 : r > 0) (h5 : 1 ≤ n) : 

  ∑ i in finset.range m, (q * a (i + 1) + r / a (i + 1)) ^ n 
  ≥ (q * p^2 + m ^ 2 * r) ^ n / (m ^ (n - 1) * p ^ n) :=
sorry

end inequality_proof_l536_536650


namespace only_a_zero_is_perfect_square_l536_536635

theorem only_a_zero_is_perfect_square (a : ℕ) : (∃ (k : ℕ), a^2 + 2 * a = k^2) → a = 0 := by
  sorry

end only_a_zero_is_perfect_square_l536_536635


namespace min_value_reciprocal_sum_l536_536028

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end min_value_reciprocal_sum_l536_536028


namespace coin_combination_l536_536784

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536784


namespace PA_bisects_angle_MPN_l536_536435

-- Given triangle ABC
variables {A B C M N P : Type}
variable [EuclideanGeometry ℝ] -- Assuming we have a Euclidean geometry context

-- Conditions
-- M lies on AB and N lies on AC such that MC = AC and NB = AB
axiom M_on_AB : M ∈ line A B
axiom N_on_AC : N ∈ line A C
axiom MC_eq_AC : dist M C = dist A C
axiom NB_eq_AB : dist N B = dist A B

-- P is symmetric to A with respect to line BC
axiom P_symmetric_A_BC : symmetric_point P A (line B C)

-- Prove PA is the angle bisector of ∠MPN
theorem PA_bisects_angle_MPN : is_angle_bisector (line A P) ∠(line M P) (line P N) :=
sorry

end PA_bisects_angle_MPN_l536_536435


namespace nat_number_bound_l536_536267

def f_k (n k : ℕ) : ℕ := k^2 + ⌊(n : ℚ) / k^2⌋.to_nat

theorem nat_number_bound (n : ℕ) :
  (∀ k : ℕ, k > 0 → f_k n k ≥ 1991) ∧ (∃ k : ℕ, k > 0 ∧ f_k n k = 1991) →
  990208 ≤ n ∧ n ≤ 991231 :=
sorry

end nat_number_bound_l536_536267


namespace complex_in_second_quadrant_l536_536128

theorem complex_in_second_quadrant : 
  let i := complex.I in
  let z := i * (1 + i) in
  z.re < 0 ∧ z.im > 0 :=
by
  let i := complex.I
  let z := i * (1 + i)
  have h1 : z.re = -1 := by sorry
  have h2 : z.im = 1 := by sorry
  exact ⟨by linarith, by linarith⟩

end complex_in_second_quadrant_l536_536128


namespace range_of_p_l536_536699

def set_A : Set ℝ := {x | x^2 - 3x - 10 ≤ 0}
def set_B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2p - 1}

theorem range_of_p (p : ℝ) : (set_B p ⊆ set_A) → 
    ((p + 1 > 2p - 1 ∨ (-2 ≤ p + 1 ∧ p + 1 ≤ 2p - 1 ∧ 2p - 1 ≤ 5)) →
    (-3 ≤ p ∧ p ≤ 3)) :=
by
  sorry

end range_of_p_l536_536699


namespace whole_numbers_in_interval_7_4_3pi_l536_536866

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536866


namespace black_squares_31x31_l536_536246

-- Definitions to express the checkerboard problem conditions
def isCheckerboard (n : ℕ) : Prop := 
  ∀ i j : ℕ,
    i < n → j < n → 
    ((i + j) % 2 = 0 → (i % 2 = 0 ∧ j % 2 = 0) ∨ (i % 2 = 1 ∧ j % 2 = 1))

def blackCornerSquares (n : ℕ) : Prop :=
  ∀ i j : ℕ,
    (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 0

-- The main statement to prove
theorem black_squares_31x31 :
  ∃ (n : ℕ) (count : ℕ), n = 31 ∧ isCheckerboard n ∧ blackCornerSquares n ∧ count = 481 := 
by 
  sorry -- Proof to be provided

end black_squares_31x31_l536_536246


namespace radius_of_given_circle_l536_536150

noncomputable def circle_radius (x y : ℝ) : ℝ :=
(x - 1)^2 + (y + 2)^2

theorem radius_of_given_circle: radius_of_given_circle: 
  (∃ center : ℝ × ℝ, radius : ℝ, (λ p, (p.1-1)^2 + (p.2+2)^2 = 4) ↔ radius = 2) := 
sorry

end radius_of_given_circle_l536_536150


namespace problem_solution_l536_536967

def scores : List ℕ := [90, 89, 90, 95, 93, 94, 93]

def average (lst : List ℕ) : ℕ := lst.sum / lst.length

def variance (lst : List ℕ) (avg : ℕ) : ℚ := 
  (lst.map (λ x => (x - avg) ^ 2)).sum / lst.length

theorem problem_solution :
  let scores_filtered := List.erase (List.erase scores 95) 89
  average scores_filtered = 92 ∧ variance scores_filtered 92 = 2.8 :=
by 
  sorry

end problem_solution_l536_536967


namespace problem1_solution_problem2_solution_l536_536987

-- Definitions and conditions for the first problem
noncomputable def problem1 (A B C : ℝ) (a b c m : ℝ) (h1 : a = m * b * Real.cos C) (h2 : m = 2) (h3 : Real.cos C = (Real.sqrt 10) / 10) : ℝ :=
  let A' := Real.acos ((a^2 + b^2 - c^2) / (2 * a * b)) in -- Cosine rule
  Real.cos A'

-- Assertion for problem 1
theorem problem1_solution (A B C : ℝ) (a b c : ℝ) (h1 : a = 2 * b * Real.cos C) (h3 : Real.cos C = (Real.sqrt 10) / 10) :
  problem1 A B C a b c 2 h1 h3 = 4 / 5 :=
sorry

-- Definitions and conditions for the second problem
noncomputable def problem2 (A B C : ℝ) (a b c m : ℝ) (h1 : a = m * b * Real.cos C) (h2 : m = 4) : ℝ :=
  let tanC := Real.tan C in
  let tanB := Real.tan B in
  Real.tan (C - B)

-- Assertion for problem 2
theorem problem2_solution (A B C : ℝ) (a b c : ℝ) (h1 : a = 4 * b * Real.cos C) :
  problem2 A B C a b c 4 h1 = (Real.sqrt 3) / 3 :=
sorry

end problem1_solution_problem2_solution_l536_536987


namespace count_whole_numbers_in_interval_l536_536874

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536874


namespace sequence_formula_l536_536695

-- Definition of the sequence S_n and its relationship with a_n
def S (n : ℕ) (a : ℕ → ℚ) := 2 * n - a n + 1

-- Conjecture that we need to prove using mathematical induction
def a (n : ℕ) : ℚ := (2^(n+1) - 1) / 2^n

-- Statement of the problem in Lean 4
theorem sequence_formula (n : ℕ) (hn: 0 < n) :
  S n (λ n, a n) = 2 * n - a n + 1 :=
sorry

end sequence_formula_l536_536695


namespace count_whole_numbers_in_interval_l536_536878

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536878


namespace simplify_tan_expression_l536_536112

noncomputable def tan_30 : ℝ := Real.tan (Real.pi / 6)
noncomputable def tan_15 : ℝ := Real.tan (Real.pi / 12)

theorem simplify_tan_expression : (1 + tan_30) * (1 + tan_15) = 2 := by
  sorry

end simplify_tan_expression_l536_536112


namespace solve_inequality_l536_536462

theorem solve_inequality (x : ℝ) (h₀ : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ set.Iic (-4) :=
by
  sorry

end solve_inequality_l536_536462


namespace parallel_DE_FG_l536_536405

open EuclideanGeometry

variables {A B C D E F G : Point} {Γ : Circle} {lineDE lineFG : Line}

noncomputable def triangle := {ABC : Triangle // AcuteTriangle ABC}

def bisectors_intersect (Γ : Circle) (BD CE : Segment) (F G : Point) : Prop :=
F ∈ minorArc A B Γ ∧ G ∈ minorArc A C Γ ∧
perpendicularBisector BD ∩ Γ = {F} ∧ perpendicularBisector CE ∩ Γ = {G}

def parallel_or_coincident (l1 l2 : Line) : Prop :=
l1 = l2 ∨ Parallel l1 l2

theorem parallel_DE_FG
  (Δ : triangle)
  (hΓ : Circumcircle Δ = Γ)
  (hD : D ∈ side AB Δ)
  (hE : E ∈ side AC Δ)
  (hAD_AE : SegmentLength (A, D) = SegmentLength (A, E))
  (hBisectors : bisectors_intersect Γ (segment B D) (segment C E) F G)
  : parallel_or_coincident (line D E) (line F G) :=
sorry

end parallel_DE_FG_l536_536405


namespace coin_combinations_sum_50_l536_536716

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536716


namespace find_monic_quartic_polynomial_l536_536266

noncomputable def monic_quartic_polynomial := 
  polynomial ℚ

theorem find_monic_quartic_polynomial : 
  ∃ p : monic_quartic_polynomial, 
    p.monic ∧
    (∀ x : ℂ, p.eval x = 0 → 
      (x = 3 + real.sqrt 5 ∨ x = 3 - real.sqrt 5 ∨ x = 2 - real.sqrt 2 ∨ x = 2 + real.sqrt 2)) ∧
    p = polynomial.C 1 * 
        (polynomial.X ^ 2 - polynomial.C 6 * polynomial.X + polynomial.C 4) *
        (polynomial.X ^ 2 - polynomial.C 4 * polynomial.X + polynomial.C 2) :=
begin
  sorry,
end

end find_monic_quartic_polynomial_l536_536266


namespace units_digit_of_product_l536_536541

-- Define the three given even composite numbers
def a := 4
def b := 6
def c := 8

-- Define the product of the three numbers
def product := a * b * c

-- State the units digit of the product
theorem units_digit_of_product : product % 10 = 2 :=
by
  -- Proof is skipped here
  sorry

end units_digit_of_product_l536_536541


namespace root_in_interval_l536_536175

noncomputable def f : ℝ → ℝ := sorry

theorem root_in_interval (h1 : f 2 * f 4 < 0) (h2 : f 2 * f 3 < 0) : ∃ x ∈ set.Ioo 2 3, f x = 0 :=
by {
  sorry
}

end root_in_interval_l536_536175


namespace sum_of_possible_remainders_l536_536488

theorem sum_of_possible_remainders :
  ∀ (n : ℕ), n < 10 → 
  (∃ a b c d : ℕ, a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ 1000 * a + 100 * b + 10 * c + d = n) →
  (∑ i in finset.range 7, (i + 28) % 37) = 217 :=
begin
  sorry
end

end sum_of_possible_remainders_l536_536488


namespace union_eq_M_l536_536415

def M : Set (ℝ × ℝ) := {p | (|p.1 * p.2| = 1 ∧ p.1 > 0)}
def N : Set (ℝ × ℝ) := {p | Real.arctan p.1 + Real.arccot p.2 = Real.pi}

theorem union_eq_M : M ∪ N = M := by
  sorry

end union_eq_M_l536_536415


namespace quadratic_inequality_solution_l536_536259

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + x - 12 > 0) → (x > 3 ∨ x < -4) :=
by
  sorry

end quadratic_inequality_solution_l536_536259


namespace coin_combinations_sum_50_l536_536720

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536720


namespace coin_combinations_count_l536_536793

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536793


namespace probability_sum_divisible_by_3_l536_536644

open Finset

-- Define the list of numbers
def numbers : Finset ℕ := {1, 2, 3, 4, 5}

-- Calculate all pairs of numbers chosen from the given list
def all_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s.filter (λ p, p.1 < p.2)  -- Ensure order so we don't count (a,b) and (b,a) separately

-- Define a predicate that checks if the sum of a pair is divisible by 3
def divisible_by_three_sum (p : ℕ × ℕ) : Prop :=
  (p.1 + p.2) % 3 = 0

-- Count the pairs fulfilling the condition
def successful_pairs : Finset (ℕ × ℕ) :=
  (all_pairs numbers).filter divisible_by_three_sum

-- Calculate the probability
theorem probability_sum_divisible_by_3 :
  (successful_pairs.card : ℚ) / (all_pairs numbers).card = 2 / 5 :=
by
  sorry  -- proof

end probability_sum_divisible_by_3_l536_536644


namespace externally_tangent_circles_l536_536338

-- Definitions of the circles and the condition of being externally tangent
def C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 + 2 * p.2 + 1 = 0}
def C2 (m : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 - 6 * p.2 + m = 0}

-- Prove that m = -3 for externally tangent circles
theorem externally_tangent_circles (m : ℝ) :
  (∃x y, (x, y) ∈ C1 ∧ ∃x' y', (x', y') ∈ C2 m) ∧
  (let d := dist (-1, -1) (2, 3) in d = 1 + Real.sqrt (13 - m)) →
  m = -3 :=
by
  sorry

end externally_tangent_circles_l536_536338


namespace divisors_count_of_3b_plus_15_l536_536470

theorem divisors_count_of_3b_plus_15 (a b : ℤ) (h : 4 * b = 10 - 3 * a) : 
  (finset.filter (λ x, (3 * b + 15) % x = 0) (finset.range 9)).card = 4 :=
sorry

end divisors_count_of_3b_plus_15_l536_536470


namespace extreme_points_a_minus_one_max_value_on_interval_l536_536690

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2

theorem extreme_points_a_minus_one :
  (∃ x, (x = 0 ∨ x = 2) ∧ (f (-1) x = 0 ∨ f (-1) x = -4)) :=
sorry

theorem max_value_on_interval (a : ℝ) :
  if a ≥ 0 then
    ∃ x ∈ set.Icc (0 : ℝ) (2 : ℝ), f a x = 12 * a + 8
  else if -1 < a then
    ∃ x ∈ set.Icc (0 : ℝ) (2 : ℝ), f a x = max (f a 0) (f a 2)
  else
    ∃ x ∈ set.Icc (0 : ℝ) (2 : ℝ), f a x = f a 0 :=
sorry

end extreme_points_a_minus_one_max_value_on_interval_l536_536690


namespace prob_eq_l536_536354

theorem prob_eq (y : ℝ) (h : 8^y - 8^(y-1) = 56) : (3 * y)^y = 36 :=
sorry

end prob_eq_l536_536354


namespace projection_is_circumcenter_l536_536100

noncomputable def point_outside_plane (P A B C : Point) : Prop := 
  P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ ¬ colinear P A B ∧ ¬ colinear P A C

def equidistant_from_points (P A B C : Point) : Prop :=
  dist P A = dist P B ∧ dist P B = dist P C

def projection_of_point_on_plane (P A B C P_proj : Point) : Prop :=
  -- Here we should define projection mathematically, but keeping it abstract for now
  sorry

def is_circumcenter (P_proj A B C : Point) : Prop :=
  -- Definition of circumcenter for triangle ABC
  centered P_proj A B C ∧ circumcenter P_proj A B C -- Hypothetical definitions

theorem projection_is_circumcenter
  (P A B C : Point)
  (h1 : point_outside_plane P A B C)
  (h2 : equidistant_from_points P A B C)
  (P_proj : Point)
  (h3 : projection_of_point_on_plane P A B C P_proj) :
  is_circumcenter P_proj A B C :=
sorry

end projection_is_circumcenter_l536_536100


namespace square_area_from_points_l536_536063

theorem square_area_from_points :
  let P1 := (1, 2)
  let P2 := (4, 6)
  let side_length := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let area := side_length^2
  P1.1 = 1 ∧ P1.2 = 2 ∧ P2.1 = 4 ∧ P2.2 = 6 →
  area = 25 :=
by
  sorry

end square_area_from_points_l536_536063


namespace cut_third_cake_into_12_pieces_l536_536582

-- Definitions based on the conditions
def radius : ℝ := 1 -- assume radius is 1 unit for simplicity

def cake_radius (cake : ℕ) : ℝ := radius

def cut_into_pieces (cake: ℕ) (n: ℕ) : list ℝ :=
  list.range n |>.map (λ i, (i : ℝ) * (2 * Real.pi) / n)

-- The theorem we need to prove
theorem cut_third_cake_into_12_pieces (cake1 cake2 cake3 : ℕ) 
  (h1 : cake_radius cake1 = radius) 
  (h2 : cake_radius cake2 = radius) 
  (h3 : cake_radius cake3 = radius) 
  (p1 : cut_into_pieces cake1 3 = [0, 2*Real.pi/3, 4*Real.pi/3]) 
  (p2 : cut_into_pieces cake2 4 = [0, Real.pi/2, Real.pi, 3*Real.pi/2]) : 
  ∃ cuts : list ℝ, cuts = list.range 12 |>.map (λ i, (i : ℝ) * (2 * Real.pi) / 12) := 
sorry

end cut_third_cake_into_12_pieces_l536_536582


namespace prove_true_propositions_l536_536015

variable {Point : Type}
variable [AffineSpace Point]

-- Definitions of lines and planes
variable (m n : Line Point)
variable (α β : Plane Point)

-- Given conditions
variable (perp_line_plane : ∀ (l : Line Point) (p : Plane Point), l ⊥ p → ¬ perpend_lines_exist l p)
variable (parallel_lines : ∀ (p : Plane Point) (l : Line Point), l ∥ p → parallel_planes_exist p l)
variable (parallel_lines_to_one_plane : ∀ (l₁ l₂ : Line Point) (p : Plane Point), l₁ ∥ p ∧ l₂ ∥ p → l₁ ∥ l₂)
variable (perpendicular_lines_to_one_plane : ∀ (l₁ l₂ : Line Point) (p : Plane Point), l₁ ⊥ p ∧ l₂ ⊥ p → l₁ ∥ l₂)

-- The proof problem
theorem prove_true_propositions :
  (perpendicular_lines_to_one_plane m n α → m ∥ n) ∧
  (perp_line_plane m α ∧ perp_line_plane m β → parallel_planes_exist α β) ∧ 
  (perp_line_plane m α → ¬ (parallel_planes_exist α β)) ∧
  (parallel_lines α m ∧ parallel_lines β m → ¬ parallel_planes_exist α β) →
  (parallel_lines_to_one_plane m α n α → ¬ (m ∥ n)) →
  [true, false, false, true] = [true, false, false, true] :=
by
  intros
  sorry

end prove_true_propositions_l536_536015


namespace sum_of_three_positives_eq_2002_l536_536351

theorem sum_of_three_positives_eq_2002 : 
  ∃ (n : ℕ), n = 334000 ∧ (∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ (A B C : ℕ), f A B C ↔ (0 < A ∧ A ≤ B ∧ B ≤ C ∧ A + B + C = 2002))) := by
  sorry

end sum_of_three_positives_eq_2002_l536_536351


namespace initial_garrison_men_l536_536214

theorem initial_garrison_men (M : ℕ) (provisions_days : ℕ) (days_passed : ℕ) (reinforcement : ℕ) (remaining_days : ℕ)
  (initial_provisions : M * provisions_days)
  (provisions_after_12_days : M * (provisions_days - days_passed))
  (provisions_after_reinforcement : (M + reinforcement) * remaining_days)
  (H : M * (provisions_days - days_passed) = (M + reinforcement) * remaining_days) :
  M = 1850 :=
by
  sorry

end initial_garrison_men_l536_536214


namespace union_of_A_and_B_l536_536422

def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4, 5, 7} :=
by sorry

end union_of_A_and_B_l536_536422


namespace area_of_union_square_circle_l536_536588

theorem area_of_union_square_circle (a r : ℝ) (h₁ : a = 20) (h₂ : r = 10) :
    let square_area := a^2 in
    let circle_area := π * r^2 in
    r * 2 ≤ a → square_area = 400 :=
by
  intros
  dsimp [square_area]
  rw [h₁]
  norm_num
  split_ifs
  norm_num

end area_of_union_square_circle_l536_536588


namespace circle_equation_line_through_point_perpendicular_lines_l536_536374

-- (Q1): Prove the equation of the circle passing through the intersection points of the curve and the coordinate axes.
theorem circle_equation (x y: ℝ) (h: y = x^2 - 6 * x + 1) : 
  (∃ x y, (y = x^2 - 6 * x + 1 ∧ (x = 0 ∨ y = 0)) → (x - 3)^2 + (y - 1)^2 = 9) := sorry

-- (Q2): Prove the equation of the line passing through (2, 3) that intersects the circle.
theorem line_through_point (x y: ℝ) (h: (x - 3)^2 + (y - 1)^2 = 9) :
  (∃ k: ℝ, ((y - 3 = k * (x - 2) ∨ x = 2) ∧ 4 * sqrt 2 = 2 * sqrt ((3)^2 - ((abs(3 * k - 1 + 3 - 2 * k))/(sqrt (k^2 + 1)))^2)) → (x = 2 ∨ 3 * x + 4 * y - 18 = 0)) := sorry

-- (Q3): Prove the value of the parameter a for which OA ⟂ OB.
theorem perpendicular_lines (a x1 y1 x2 y2: ℝ) (h: (x1 - 3)^2 + (y1 - 1)^2 = 9 ∧ (x2 - 3)^2 + (y2 - 1)^2 = 9 ∧ y1 = x1 + a ∧ y2 = x2 + a):
  ((OA ⟂ OB) ∧ x1 + x2 = 4 - a ∧ x1 * x2 = (a^2 - 2 * a + 1) / 2) → 
  a = -1 := sorry

end circle_equation_line_through_point_perpendicular_lines_l536_536374


namespace triangle_tangent_l536_536392

/-- In a right triangle ABC with C as the right angle, and given cos A = 3/5, 
    the tangent of angle B is 4/3. -/
theorem triangle_tangent (A B C : ℝ) (x : ℝ) (h1 : A + B = 90) (h2 : cos A = 3 / 5) :
  tan B = 4 / 3 :=
sorry

end triangle_tangent_l536_536392


namespace equalize_after_finite_steps_l536_536473

def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def adjust_balls (piles : List ℕ) : List ℕ :=
  piles.map (λ pile => 
    let third := pile / 3 in
    let left_idx := (piles.indexOf pile + piles.length - 1) % piles.length in
    let right_idx := (piles.indexOf pile + 1) % piles.length in
    piles.update_nth (piles.indexOf pile) (pile - 2 * third)
         .update_nth left_idx (piles.nth! left_idx + third)
         .update_nth right_idx (piles.nth! right_idx + third))

def make_multiple_of_3 (n : ℕ) : ℕ :=
  if n % 3 = 0 then n
  else n + (3 - (n % 3))

def adjust_with_multiple_of_3 (piles : List ℕ) : List ℕ :=
  piles.map make_multiple_of_3

def iteratively_adjust (piles : List ℕ) : List ℕ :=
  let rec aux (p : List ℕ) (steps : ℕ) : List ℕ :=
    if p.all (λ x => x = p.head) then p
    else aux (adjust_with_multiple_of_3 (adjust_balls p)) (steps + 1)
  aux piles 0

theorem equalize_after_finite_steps
  (piles : List ℕ)
  (h : ∀ p ∈ piles, divisible_by_3 p):
  ∃ n : ℕ, ∀ i j, i < piles.length → j < piles.length → (iteratively_adjust piles).nth! i = (iteratively_adjust piles).nth! j :=
begin
  sorry
end

end equalize_after_finite_steps_l536_536473


namespace count_whole_numbers_in_interval_l536_536892

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536892


namespace locus_of_midpoints_l536_536656

theorem locus_of_midpoints 
(triangle_ABC : Triangle)
(S : circle (triangle_ABC.C) (triangle_ABC.A))
(X : Point)
(hX : X ∈ S)
(K : Point)
(hK : midpoint (triangle_ABC.C) X = K) :
  ∃ (ω : circle) (fixed_radius : ℝ), 
  (radius ω = fixed_radius) ∧ 
  (fixed_radius = (1 / 4) * distance (triangle_ABC.A) (triangle_ABC.C)) ∧
  ∀ (BK_mid : Point), 
  midpoint (triangle_ABC.B) K = BK_mid → 
  BK_mid ∈ ω := 
sorry

end locus_of_midpoints_l536_536656


namespace count_whole_numbers_in_interval_l536_536884

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536884


namespace parabola_focus_latus_rectum_l536_536953

theorem parabola_focus_latus_rectum {a : ℝ} (h : latus_rectum y^2 = a * x = line x = -1) : (focus (parabola y^2 = a * x)).coords = (1, 0) :=
sorry

end parabola_focus_latus_rectum_l536_536953


namespace temperature_on_friday_l536_536201

-- Define the conditions
variables {M T W Th F : ℝ}
variable h1 : (M + T + W + Th) / 4 = 48
variable h2 : (T + W + Th + F) / 4 = 40
variable h3 : M = 42

-- Statement to prove the temperature on Friday
theorem temperature_on_friday : F = 10 :=
by
  sorry

end temperature_on_friday_l536_536201


namespace square_area_l536_536074

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (1, 2) ∧ p2 = (4, 6)) :
  let d := distance p1 p2 in
  d^2 = 25 :=
by
  sorry

end square_area_l536_536074


namespace number_of_integers_in_interval_l536_536920

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536920


namespace sum_of_first_1234_terms_l536_536382

-- Define the sequence
def seq : ℕ → ℕ
| 0 := 1
| (n + 1) := if n % (2 + seq n) == 1 then 1 else 2

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ :=
(nat.rec_on n 0 (λ n ih, ih + seq n))

-- Define the given conditions and the correct answer
theorem sum_of_first_1234_terms : sum_seq 1234 = 2419 := 
by sorry

end sum_of_first_1234_terms_l536_536382


namespace domain_of_function_l536_536490

theorem domain_of_function :
  (∀ x, x > -1 ∧ -2 ≤ x ∧ x ≤ 2 → x ∈ set.Ioo (-1) 2) :=
begin
  intros x hx,
  cases hx with h₁ h₂,
  cases h₂ with h₂a h₂b,
  split,
  { exact h₁ },
  { split,
    { linarith },
    { linarith } },
end

end domain_of_function_l536_536490


namespace coin_combinations_count_l536_536736

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536736


namespace count_whole_numbers_in_interval_l536_536902

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536902


namespace oxen_count_b_l536_536194

theorem oxen_count_b 
  (a_oxen : ℕ) (a_months : ℕ)
  (b_months : ℕ) (x : ℕ)
  (c_oxen : ℕ) (c_months : ℕ)
  (total_rent : ℝ) (c_rent : ℝ)
  (h1 : a_oxen * a_months = 70)
  (h2 : c_oxen * c_months = 45)
  (h3 : c_rent / total_rent = 27 / 105)
  (h4 : total_rent = 105) :
  x = 12 :=
by 
  sorry

end oxen_count_b_l536_536194


namespace minimum_value_l536_536274

theorem minimum_value (x : ℝ) (hx : x > 0) : 4 * x^2 + 1 / x^3 ≥ 5 ∧ (4 * x^2 + 1 / x^3 = 5 ↔ x = 1) :=
by {
  sorry
}

end minimum_value_l536_536274


namespace benny_eggs_l536_536605

theorem benny_eggs (dozen_count : ℕ) (eggs_per_dozen : ℕ) (total_eggs : ℕ) 
  (h1 : dozen_count = 7) 
  (h2 : eggs_per_dozen = 12) 
  (h3 : total_eggs = dozen_count * eggs_per_dozen) : 
  total_eggs = 84 := 
by 
  sorry

end benny_eggs_l536_536605


namespace square_area_l536_536071

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (1, 2) ∧ p2 = (4, 6)) :
  let d := distance p1 p2 in
  d^2 = 25 :=
by
  sorry

end square_area_l536_536071


namespace odd_function_property_l536_536206

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^3 + Real.log (1 + x) else x^3 - Real.log (1 - x)

theorem odd_function_property (x : ℝ) (h1 : x < 0) :
  f(x) = x^3 - Real.log (1 - x) :=
by
  -- Statement follows that for x < 0
  sorry

end odd_function_property_l536_536206


namespace compound_interest_rate_13_97_percent_l536_536179

-- Definitions of constants and conditions
def P : ℝ := 780
def A : ℝ := 1300
def n : ℝ := 4
def t : ℝ := 4

-- Compound interest formula
def compound_interest_formula (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

-- The theorem we want to prove
theorem compound_interest_rate_13_97_percent :
  ∃ r : ℝ, compound_interest_formula r ∧ r ≈ 0.1397396 := 
begin
  -- Placeholder for the proof.
  sorry,
end

end compound_interest_rate_13_97_percent_l536_536179


namespace pair_not_equal_to_64_l536_536192

theorem pair_not_equal_to_64 :
  ¬(4 * (9 / 2) = 64) := by
  sorry

end pair_not_equal_to_64_l536_536192


namespace whole_numbers_in_interval_7_4_3pi_l536_536865

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536865


namespace combination_coins_l536_536819

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536819


namespace g_five_steps_l536_536946

noncomputable def g (x : ℚ) : ℚ := -1 / (x ^ 2)

theorem g_five_steps (x : ℚ) : g(g(g(g(g(5))))) = -1 / 23283064365386962890625 := by
  sorry

end g_five_steps_l536_536946


namespace count_whole_numbers_in_interval_l536_536852

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536852


namespace area_of_regular_octagon_l536_536663

theorem area_of_regular_octagon (BDEF_is_rectangle : true) (AB : ℝ) (BC : ℝ) 
    (capture_regular_octagon : true) (AB_eq_1 : AB = 1) (BC_eq_2 : BC = 2)
    (octagon_perimeter_touch : ∀ x, x = 1) : 
    ∃ A : ℝ, A = 11 :=
by
  sorry

end area_of_regular_octagon_l536_536663


namespace points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l536_536190

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem points_in_first_quadrant (x y : ℝ) (h : x > 0 ∧ y > 0) : first_quadrant x y :=
by {
  sorry
}

theorem points_in_fourth_quadrant (x y : ℝ) (h : x > 0 ∧ y < 0) : fourth_quadrant x y :=
by {
  sorry
}

theorem points_in_second_quadrant (x y : ℝ) (h : x < 0 ∧ y > 0) : second_quadrant x y :=
by {
  sorry
}

theorem points_in_third_quadrant (x y : ℝ) (h : x < 0 ∧ y < 0) : third_quadrant x y :=
by {
  sorry
}

end points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l536_536190


namespace ways_to_divide_friends_l536_536836

theorem ways_to_divide_friends : (4 ^ 8 = 65536) := by
  sorry

end ways_to_divide_friends_l536_536836


namespace bianca_points_earned_l536_536556

-- Define the constants and initial conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 17
def not_recycled_bags : ℕ := 8

-- Define a function to calculate the number of recycled bags
def recycled_bags (total: ℕ) (not_recycled: ℕ) : ℕ :=
  total - not_recycled

-- Define a function to calculate the total points earned
def total_points_earned (bags: ℕ) (points_per_bag: ℕ) : ℕ :=
  bags * points_per_bag

-- State the theorem
theorem bianca_points_earned : total_points_earned (recycled_bags total_bags not_recycled_bags) points_per_bag = 45 :=
by
  sorry

end bianca_points_earned_l536_536556


namespace seating_chart_example_l536_536928

def seating_chart_representation (a b : ℕ) : String :=
  s!"{a} columns {b} rows"

theorem seating_chart_example :
  seating_chart_representation 4 3 = "4 columns 3 rows" :=
by
  sorry

end seating_chart_example_l536_536928


namespace sin_45_eq_sqrt2_div_2_l536_536516

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = (Real.sqrt 2) / 2 := 
  sorry

end sin_45_eq_sqrt2_div_2_l536_536516


namespace range_of_m_l536_536310

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |2009 * x + 1| ≥ |m - 1| - 2) → -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l536_536310


namespace encyclopedia_colored_pages_l536_536289

def is_all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 0

def has_red_mark (n : ℕ) : Prop :=
  n % 3 = 2

def colored_pages_count (pages : ℕ) :=
  (finset.range (pages + 1)).filter (λ n, is_all_even_digits n ∧ has_red_mark n)

theorem encyclopedia_colored_pages :
  colored_pages_count 2023 = 44 := 
sorry

end encyclopedia_colored_pages_l536_536289


namespace find_b_l536_536964

theorem find_b (a b c : ℝ) (B : ℝ) (area : ℝ) 
  (h1 : 2 * b = a + c) 
  (h2 : B = π / 6) 
  (h3 : area = 3 / 2) : 
  b = sqrt 3 + 1 := 
by
  sorry

end find_b_l536_536964


namespace probability_first_number_greater_l536_536579

noncomputable def probability_first_greater_second : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_first_number_greater :
  probability_first_greater_second = 7 / 16 :=
sorry

end probability_first_number_greater_l536_536579


namespace find_min_omega_l536_536329

-- Define the function f
def f (ω φ x : ℝ) : ℝ := Math.sin (ω * x + φ)

noncomputable def min_omega (ω φ : ℝ) : ℝ :=
if h : ∃ x₀, ∀ x, f ω φ x₀ ≤ f ω φ x ∧ f ω φ x ≤ f ω φ (x₀ + π / 8) then ω else sorry

theorem find_min_omega (ω φ : ℝ) : 
  (∀ x, f ω φ (π / 32 - x) = f ω φ (π / 32 + x)) ∧ 
  f ω φ (-π / 32) = 0 ∧ 
  ∃ x₀, ∀ x, f ω φ x₀ ≤ f ω φ x ∧ f ω φ x ≤ f ω φ (x₀ + π / 8) → 
  min_omega ω φ = 8 :=
sorry

end find_min_omega_l536_536329


namespace more_birds_than_nests_l536_536557

theorem more_birds_than_nests (birds nests : Nat) (h_birds : birds = 6) (h_nests : nests = 3) : birds - nests = 3 :=
by
  sorry

end more_birds_than_nests_l536_536557


namespace find_a_l536_536423

open Set

variables {α : Type*} [LinearOrderedField α]

def A (a : α) : Set α := {x | x^2 + (a * x) + 1 = 0}
def B : Set α := {1, 2}

theorem find_a (a : α) (h : A a = B) : a = -3 :=
by {
  sorry
}

end find_a_l536_536423


namespace problem_1_problem_2_l536_536317

noncomputable def cos (x : ℝ) : ℝ := sorry
noncomputable def sin (x : ℝ) : ℝ := sorry

theorem problem_1 (x : ℝ) (h1 : cos x = -sqrt 10 / 10) 
                   (h2 : sin x = 3 * sqrt 10 / 10) : 
  sin x + cos x = sqrt 10 / 5 := 
  by sorry

theorem problem_2 (x : ℝ) (h1 : cos x = -sqrt 10 / 10) 
                   (h2 : sin x = 3 * sqrt 10 / 10) : 
  (sin (π / 2 + x) * cos (π / 2 - x)) / (cos (-x) * cos (π - x)) = 3 := 
  by sorry

end problem_1_problem_2_l536_536317


namespace log_probability_l536_536107

noncomputable def geometric_probability (a b : ℝ) (f : ℝ → Prop) : ℝ :=
  let R := set.Icc a b
  let E := {x ∈ R | f x}
  (R.measure / E.measure)

theorem log_probability : 
  geometric_probability 0 4 (λ x, -1 ≤ Real.logBase (1/3) (x + 1/2) ∧ Real.logBase (1/3) (x + 1/2) ≤ 1) = 3/8 :=
by
  sorry

end log_probability_l536_536107


namespace dot_product_range_l536_536355

variable (a b : ℝ)
variable (θ : ℝ)
variable (norm_a : ℝ := 5)
variable (norm_b : ℝ := 13)
variable (cos_θ_range : 0 ≤ Real.cos θ)

theorem dot_product_range : 0 ≤ a * b ∧ a * b ≤ 65 :=
by
  have h1 : a = norm_a := rfl
  have h2 : b = norm_b := rfl
  have h3 : 0 ≤ Real.cos θ := cos_θ_range
  sorry

end dot_product_range_l536_536355


namespace count_three_digit_numbers_l536_536342

-- Define the set of digits to be used
def digits : Set ℕ := {3, 4, 5}

-- Define a three-digit number using the given digits
def is_three_digit_number (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits

-- Define the statement to be proved
theorem count_three_digit_numbers : {n : ℕ | n ≥ 100 ∧ n < 1000 ∧ is_three_digit_number n}.card = 6 :=
by
  sorry

end count_three_digit_numbers_l536_536342


namespace number_of_true_propositions_l536_536324

-- Definitions based on the conditions
def systematic_sampling : Prop :=
  ∀ (sampling_interval_minutes : ℕ), sampling_interval_minutes = 15 → true

def correlation_coefficient_property : Prop :=
  ∀ (corr : ℝ), abs corr ≤ 1

def k_squared_value_relation (k : ℝ) : Prop :=
  k < 0 → false

def regression_line_property : Prop :=
  ∀ (x : ℝ), ∃ (y_hat : ℝ), y_hat = 0.4 * x + 12

-- Main theorem stating the number of true propositions
theorem number_of_true_propositions : ℕ :=
  let p1 := systematic_sampling in
  let p2 := correlation_coefficient_property in
  let p3 := k_squared_value_relation in
  let p4 := regression_line_property in
  if p1 ∧ p2 ∧ ¬p3 ∧ p4 then
    3 
  else 
    sorry

end number_of_true_propositions_l536_536324


namespace highlighter_difference_l536_536963

theorem highlighter_difference :
  ∃ (P : ℕ), 7 + P + (P + 5) = 40 ∧ P - 7 = 7 :=
by
  sorry

end highlighter_difference_l536_536963


namespace box_problem_l536_536966

theorem box_problem 
    (x y : ℕ) 
    (h1 : 10 * x + 20 * y = 18 * (x + y)) 
    (h2 : 10 * x + 20 * (y - 10) = 16 * (x + y - 10)) :
    x + y = 20 :=
sorry

end box_problem_l536_536966


namespace number_of_integers_in_interval_l536_536918

theorem number_of_integers_in_interval (a b : ℝ) (h1 : a = 7 / 4) (h2 : b = 3 * Real.pi) :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, a < x ∧ x < b ↔ 2 ≤ x ∧ x ≤ 9 :=
by
  rw [h1, h2]
  exact ⟨8, by_norm_num, λ x, by norm_num⟩

end number_of_integers_in_interval_l536_536918


namespace original_acid_percentage_zero_l536_536543

theorem original_acid_percentage_zero (a w : ℝ) 
  (h1 : (a + 1) / (a + w + 1) = 1 / 4) 
  (h2 : (a + 2) / (a + w + 2) = 2 / 5) : 
  a / (a + w) = 0 := 
by
  sorry

end original_acid_percentage_zero_l536_536543


namespace coin_combinations_count_l536_536735

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536735


namespace geometric_sequence_inserted_product_l536_536992

theorem geometric_sequence_inserted_product :
  ∃ (a b c : ℝ), a * b * c = 216 ∧
    (∃ (q : ℝ), 
      a = (8/3) * q ∧ 
      b = a * q ∧ 
      c = b * q ∧ 
      (8/3) * q^4 = 27/2) :=
sorry

end geometric_sequence_inserted_product_l536_536992


namespace X_in_Y_l536_536337

def X : set ℕ := {0, 1}
def Y : set (set ℕ) := { x | x ⊆ X }

theorem X_in_Y : X ∈ Y := by
  sorry

end X_in_Y_l536_536337


namespace general_term_of_seq_l536_536693

open Nat

noncomputable def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 3 * 2^n

theorem general_term_of_seq (a : ℕ → ℕ) :
  seq a → ∀ n, a n = (3 * n - 1) * 2^(n-1) :=
by
  sorry

end general_term_of_seq_l536_536693


namespace minimum_r3_value_maximum_r3_value_l536_536294

noncomputable def r3_minimum (pentagon : Fin 5 → ℝ × ℝ) (M : ℝ × ℝ) : ℝ :=
  let distances := Finset.univ.image (λ i, (euclidean_distance (pentagon i) M))
  (distances.sort (≤)).nth_le 2 (by decide)

noncomputable def r3_maximum (pentagon : Fin 5 → ℝ × ℝ) (M : ℝ × ℝ) : ℝ :=
  let distances := Finset.univ.image (λ i, (euclidean_distance (pentagon i) M))
  (distances.sort (≤)).nth_le 2 (by decide)

theorem minimum_r3_value (pentagon : Fin 5 → ℝ × ℝ) (M : ℝ × ℝ) (h : is_regular_pentagon pentagon) :
  ∃ (M : ℝ × ℝ), r3_minimum pentagon M = 0.809 :=
sorry

theorem maximum_r3_value (pentagon : Fin 5 → ℝ × ℝ) (M : ℝ × ℝ) (h : is_regular_pentagon pentagon) :
  ∃ (M : ℝ × ℝ), r3_maximum pentagon M = 1.559 :=
sorry

end minimum_r3_value_maximum_r3_value_l536_536294


namespace part1_part2_part3_l536_536309

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (h1 : ‖a‖ = 1)
variables (h2 : ‖b‖ = 2)
variables (h3 : real.angle (a, b) = real.pi / 3)

-- Part 1
theorem part1 : a ⬝ b = -1 := by
  sorry

-- Part 2
theorem part2 (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 2) (h4 : a ⬝ b = -1) : (a - 3 • b) ⬝ (2 • a + b) = -5 := by
  sorry

-- Part 3
theorem part3 (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 2) (h4 : a ⬝ b = -1) : ‖2 • a - b‖ = 2 * real.sqrt 3 := by
  sorry

end part1_part2_part3_l536_536309


namespace combination_coins_l536_536816

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536816


namespace perimeter_of_parallelogram_ADEF_l536_536388

open Set Classical

variables {A B C D E F : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] 
variables [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables {AB AC BC AD AF DE EF : ℝ}

noncomputable def triangle_ABC (A B C : Type) := 
  ∃ (x y z : ℝ), x = AB ∧ y = AC ∧ z = BC

def isosceles_triangle (A B C : Type) := 
  triangle_ABC A B C ∧ AB = AC

def points_D_E_F (A B C D E F : Type) :=
  ∃ (x y z : Set A B C), x = D ∧ y = E ∧ z = F

def parallel_sides (D E F A B C : Type) := 
  ∃ (x y : Set D E F A C),
  x = ∥ D E ∥ = ∥ A C ∥ ∧ 
  y = ∥ E F ∥ = ∥ A B ∥

theorem perimeter_of_parallelogram_ADEF 
  (h1 : isosceles_triangle A B C)
  (h2 : triangle_ABC A B C)
  (h3 : points_D_E_F A B C D E F)
  (h4 : parallel_sides D E F A B C) :
  perimeter (Set (A D E F)) = 30 := sorry

end perimeter_of_parallelogram_ADEF_l536_536388


namespace polynomial_count_l536_536345

noncomputable def ω : ℂ := exp (2 * real.pi * complex.I / 3)

theorem polynomial_count : 
  ∃ (P : Polynomial ℝ), 
  (∀ (r : ℝ), P.eval (r * (ω^2)) = 0 ↔ (P.eval r = 0)) ∧ 
  (P.degree = 5) ∧ 
  (P.coeff 0 = 2023) ∧ 
  (∀ q : Polynomial ℝ, 
    (∀ (r : ℝ), q.eval (r * (ω^2)) = 0 ↔ (q.eval r = 0)) ∧ 
    (q.degree = 5) ∧ 
    (q.coeff 0 = 2023) → P = q ∨ P = q) := sorry

end polynomial_count_l536_536345


namespace planting_schemes_count_l536_536544

theorem planting_schemes_count :
  ∃ (seeds : Finset String) (plot1 : String), 
  seeds = {"corn", "potatoes", "eggplants", "chili_peppers", "carrots"} ∧ 
  (plot1 = "eggplants" ∨ plot1 = "chili_peppers") → 
  (∃ (remaining_seeds : Finset String), remaining_seeds ⊆ seeds \ {plot1} ∧ 
  remaining_seeds.card = 3 ∧
  (∃ (perm : Finset (List String)), perm = remaining_seeds.permutations ∧ 
  perm.card * 2 = 48)) :=
by
  sorry

end planting_schemes_count_l536_536544


namespace compare_f_log_range_of_t_l536_536313

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem compare_f_log:
  f (Real.log 26 / Real.log 3) < f (Real.log 8 / Real.log 3) ∧
  f (Real.log 8 / Real.log 3) < f (Real.log 3 / Real.log 2) :=
by sorry

theorem range_of_t {t : ℝ} (ht : 0 < t) 
  (h : ∀ x ∈ Icc (2:ℝ) 3, f (t + x^2) + f (1 - x - x^2 - 2^x) > 0) : 
  t < 5 :=
by sorry

end compare_f_log_range_of_t_l536_536313


namespace cost_per_unit_l536_536576

theorem cost_per_unit 
  (units_per_month : ℕ := 400)
  (selling_price_per_unit : ℝ := 440)
  (profit_requirement : ℝ := 40000)
  (C : ℝ) :
  profit_requirement ≤ (units_per_month * selling_price_per_unit) - (units_per_month * C) → C ≤ 340 :=
by
  sorry

end cost_per_unit_l536_536576


namespace prime_roots_sum_product_l536_536106

theorem prime_roots_sum_product (p q : ℕ) (x1 x2 : ℤ)
  (hp: Nat.Prime p) (hq: Nat.Prime q) 
  (h_sum: x1 + x2 = -↑p)
  (h_prod: x1 * x2 = ↑q) : 
  p = 3 ∧ q = 2 :=
sorry

end prime_roots_sum_product_l536_536106


namespace inequality_solution_l536_536463

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) :=
by
  sorry

end inequality_solution_l536_536463


namespace ratio_of_volumes_of_cones_l536_536209

theorem ratio_of_volumes_of_cones (r θ h1 h2 : ℝ) (hθ : 3 * θ + 4 * θ = 2 * π)
    (hr1 : r₁ = 3 * r / 7) (hr2 : r₂ = 4 * r / 7) :
    let V₁ := (1 / 3) * π * r₁^2 * h1
    let V₂ := (1 / 3) * π * r₂^2 * h2
    V₁ / V₂ = (9 : ℝ) / 16 := by
  sorry

end ratio_of_volumes_of_cones_l536_536209


namespace count_whole_numbers_in_interval_l536_536899

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536899


namespace sum_of_solutions_l536_536279

theorem sum_of_solutions : 
  let equation := λ x : ℝ, -10 * x / (x^2 - 4) = 3 * x / (x + 2) - 8 / (x - 2)
  let solutions := {x : ℝ | equation x ∧ x ≠ 2 ∧ x ≠ -2}
  (∑ x in solutions, x) = 13 / 3 := 
sorry

end sum_of_solutions_l536_536279


namespace solve_for_a_l536_536621

theorem solve_for_a : ∀ (a : ℝ), (2 * a - 16 = 9) → (a = 12.5) :=
by
  intro a h
  sorry

end solve_for_a_l536_536621


namespace combinations_of_coins_l536_536771

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def number_of_valid_combinations : ℕ :=
  (List.range 51).countp (λ p, 
  (List.range 11).countp (λ n, 
  (List.range 6).countp (λ d, 
  (List.range 3).countp (λ q, 
  is_valid_combination p n d q)))) 

theorem combinations_of_coins : 
  number_of_valid_combinations = 48 := sorry

end combinations_of_coins_l536_536771


namespace coin_combinations_count_l536_536791

-- Define the types of coins with their respective values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Prove that the number of combinations of coins that sum to 50 equals 10
theorem coin_combinations_count : ∀(p1 p5 p10 p25 : ℕ), 
        p1 * penny + p5 * nickel + p10 * dime + p25 * quarter = 50 →
        p1 ≥ 0 ∧ p5 ≥ 0 ∧ p10 ≥ 0 ∧ p25 ≥ 0 →
        (p1, p5, p10, p25).qunitility → 
        10 := sorry

end coin_combinations_count_l536_536791


namespace coin_combinations_sum_50_l536_536712

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536712


namespace sum_factorial_last_two_digits_l536_536000

theorem sum_factorial_last_two_digits :
  (∑ i in Finset.range 2012, Nat.factorial i) % 100 = 13 :=
by
  sorry

end sum_factorial_last_two_digits_l536_536000


namespace dot_product_value_find_t_l536_536312

noncomputable def vector_dot_product (a b : ℝ^3) : ℝ := 
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def vector_magnitude (a : ℝ^3) : ℝ := 
  real.sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)

theorem dot_product_value (a b : ℝ^3) (theta : ℝ) 
  (h_theta : theta = 2 * real.pi / 3) 
  (h_mag_a : vector_magnitude a = 1) 
  (h_mag_b : vector_magnitude b = 2) : 
  vector_dot_product a b = -1 := 
  sorry

theorem find_t (a b : ℝ^3) (h_dot : vector_dot_product a b = -1) 
  (h_perpendicular : vector_dot_product (2 • a - b) (t • a + b) = 0) : 
  t = 2 := 
  sorry

end dot_product_value_find_t_l536_536312


namespace vector_magnitude_subtraction_l536_536672

variables (a b : EuclideanSpace ℝ (Fin 2))

def angle (u v : EuclideanSpace ℝ (Fin 2)) : ℝ :=  -- Suppose this function calculates the angle between two vectors.
  real.arccos ((inner u v) / ((euclideanNorm u) * (euclideanNorm v)))

noncomputable def magnitude (v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  ∥v∥ -- Euclidean norm of the vector v.

theorem vector_magnitude_subtraction : 
  let a := (2 : ℝ, 0 : ℝ) : EuclideanSpace ℝ (Fin 2)
  let b_has_unit_magnitude := (magnitude b = 1)
  let angle_eq := (angle a b = (π / 3))
  (|a - 2 • b| = 2) :=
by {
  sorry
}

end vector_magnitude_subtraction_l536_536672


namespace A_on_curve_slope_at_A_l536_536320

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x ^ 2

-- Define the point A on the curve f
def A : ℝ × ℝ := (2, 8)

-- Define the condition that A is on the curve f
theorem A_on_curve : A.2 = f A.1 := by
  -- * left as a proof placeholder
  sorry

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

-- State and prove the main theorem
theorem slope_at_A : (deriv f) 2 = 8 := by
  -- * left as a proof placeholder
  sorry

end A_on_curve_slope_at_A_l536_536320


namespace min_b_satisfies_inequality_l536_536275

theorem min_b_satisfies_inequality : ∀ b : ℝ, (b^2 - 12 * b + 32 ≤ 0) → 4 ≤ b :=
begin
  sorry
end

end min_b_satisfies_inequality_l536_536275


namespace x_equals_2_sufficient_not_necessary_l536_536709

noncomputable def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem x_equals_2_sufficient_not_necessary (x : ℝ) :
  let a := (x, 1) in
  let b := (4, x) in
  x = 2 → vectors_parallel a b ∧ ¬ (∀ x, vectors_parallel a b → x = 2) :=
by
  intros
  sorry

end x_equals_2_sufficient_not_necessary_l536_536709


namespace clock_angle_3_45_l536_536533

-- Definitions of the problem conditions
def degree_per_hour : ℝ := 30
def degree_per_minute : ℝ := 6
def minute_hand_position (minute : ℕ) : ℝ := minute * degree_per_minute
def hour_hand_position (hour minute : ℕ) : ℝ :=
  hour * degree_per_hour + minute * 0.5

-- Given time
def hour := 3
def minute := 45

-- Calculation of the angles of the minute hand and the hour hand
def minute_angle := minute_hand_position minute
def hour_angle := hour_hand_position hour minute

-- Calculation of the direct angle between the two hands
def angle_difference := abs (minute_angle - hour_angle)

-- Calculate the smaller angle
def smaller_angle := if angle_difference > 180 then 360 - angle_difference else angle_difference

-- The proof statement
theorem clock_angle_3_45 : smaller_angle = 157.5 :=
by
  sorry

end clock_angle_3_45_l536_536533


namespace general_formula_and_arithmetic_sequence_l536_536696

noncomputable def S_n (n : ℕ) : ℕ := 3 * n ^ 2 - 2 * n
noncomputable def a_n (n : ℕ) : ℕ := S_n n - S_n (n - 1)

theorem general_formula_and_arithmetic_sequence :
  (∀ n : ℕ, a_n n = 6 * n - 5) ∧
  (∀ n : ℕ, (n ≥ 2 → a_n n - a_n (n - 1) = 6) ∧ (a_n 1 = 1)) :=
by
  sorry

end general_formula_and_arithmetic_sequence_l536_536696


namespace intersection_sets_l536_536700

theorem intersection_sets :
  let M := {x : ℝ | (x + 3) * (x - 2) < 0 }
  let N := {x : ℝ | 1 ≤ x ∧ x ≤ 3 }
  M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_sets_l536_536700


namespace combination_coins_l536_536818

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l536_536818


namespace elvin_fixed_monthly_charge_l536_536547

theorem elvin_fixed_monthly_charge :
  ∃ (F C : ℝ), F + C = 40 ∧ F + 2C = 76 ∧ F = 4 :=
by
  let F := 4
  let C := 36
  use [F, C]
  have h1 : (F + C = 40) := by norm_num
  have h2 : (F + 2*C = 76) := by norm_num
  refine ⟨h1, h2, rfl⟩
  sorry -- Complete the proof steps here if needed.

#check elvin_fixed_monthly_charge

end elvin_fixed_monthly_charge_l536_536547


namespace total_students_l536_536198

theorem total_students (girls boys : ℕ) (h1 : girls = 300) (h2 : boys = 8 * (girls / 5)) : girls + boys = 780 := by
  sorry

end total_students_l536_536198


namespace area_of_square_with_adjacent_points_l536_536093

theorem area_of_square_with_adjacent_points (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 6)) :
  let side_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in 
  let area := side_length^2 in 
  area = 25 :=
by
  sorry

end area_of_square_with_adjacent_points_l536_536093


namespace line_circle_intersection_l536_536331

theorem line_circle_intersection (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.1 + A.2 + m = 0) ∧ (B.1 + B.2 + m = 0) ∧
    (A.1^2 + A.2^2 = 2) ∧ (B.1^2 + B.2^2 = 2) ∧
    ‖(A.1, A.2) + (B.1, B.2)‖ ≥ ‖(A.1 - B.1, A.2 - B.2)‖) →
  m ∈ set.Ioc (-2 : ℝ) (-Real.sqrt 2) ∪ set.Ico (Real.sqrt 2) 2 :=
by 
  sorry

end line_circle_intersection_l536_536331


namespace area_of_square_l536_536061

-- We define the points as given in the conditions
def point1 : ℝ × ℝ := (1, 2)
def point2 : ℝ × ℝ := (4, 6)

-- Lean's "def" defines the concept of a square given two adjacent points.
def is_square (p1 p2: ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  in ∃ (l : ℝ), l = d ∧ (l^2 = 25)

-- The theorem assumes the points are adjacent points on a square and proves that their area is 25.
theorem area_of_square :
  is_square point1 point2 :=
by
  -- Insert formal proof here, skipped with 'sorry' for this task
  sorry

end area_of_square_l536_536061


namespace quadratic_root_value_k_l536_536257

theorem quadratic_root_value_k (k : ℝ) :
  (
    ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -4 / 3 ∧
    (∀ x : ℝ, x^2 * k - 8 * x - 18 = 0 ↔ (x = x₁ ∨ x = x₂))
  ) → k = 4.5 :=
by
  sorry

end quadratic_root_value_k_l536_536257


namespace fraction_of_smart_integers_divisible_by_23_l536_536616

def is_smart_integer (n : ℤ) : Prop :=
  even n ∧ 50 < n ∧ n < 200 ∧ (digits 10 n).sum = 12

theorem fraction_of_smart_integers_divisible_by_23 :
  (finset.filter (λ n, is_smart_integer n ∧ n % 23 = 0) (finset.range 200)).card
  = 1 / 8 * (finset.filter is_smart_integer (finset.range 200)).card :=
sorry

end fraction_of_smart_integers_divisible_by_23_l536_536616


namespace coin_combinations_l536_536802

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l536_536802


namespace point_in_fourth_quadrant_l536_536039

noncomputable def z : ℂ := 1 - complex.i

-- Conditions
def conjugate_z := complex.conj z
def division := conjugate_z / z
def square := z ^ 2
def magnitude := complex.abs z

-- Compound expression
def target := division + square + magnitude

-- Lean statement
theorem point_in_fourth_quadrant :
  (target.re > 0) ∧ (target.im < 0) :=
sorry

end point_in_fourth_quadrant_l536_536039


namespace main_theorem_l536_536402

def exists_constant_cq (q : ℕ) (hq : q > 0) (A : Finset ℤ) : Prop :=
  ∃ C_q : ℤ, |(A + q • A)| ≥ (q + 1) * |A| - C_q

theorem main_theorem (q : ℕ) (hq : q > 0) (A : Finset ℤ) : exists_constant_cq q hq A := 
sorry

end main_theorem_l536_536402


namespace count_whole_numbers_in_interval_l536_536867

theorem count_whole_numbers_in_interval :
  let lower_bound := (7 : ℝ) / 4,
      upper_bound := 3 * Real.pi,
      count := Nat.card (Finset.filter (λ n, (lower_bound.ceil ≤ n ∧ n ≤ upper_bound.floor))
                   (Finset.Icc lower_bound.ceil upper_bound.floor))
  in count = 8 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536867


namespace isosceles_triangle_area_l536_536154

theorem isosceles_triangle_area (a b : ℝ) (h : 0 < a ∧ 0 < b) (vertex_angle : angle = π / 2.25) :
  area_isosceles_triangle a b = a^3 * b / (4 * (b^2 - a^2)) :=
sorry

end isosceles_triangle_area_l536_536154


namespace eleven_fifteen_rounded_nearest_tenth_l536_536097

def fraction := 11 / 15

theorem eleven_fifteen_rounded_nearest_tenth : Real.round (11 / 15) 1 = 0.7 :=
by
  sorry

end eleven_fifteen_rounded_nearest_tenth_l536_536097


namespace solve_inequality_l536_536461

theorem solve_inequality (x : ℝ) (h₀ : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ set.Iic (-4) :=
by
  sorry

end solve_inequality_l536_536461


namespace angle_c_with_plane_is_30_degrees_l536_536596

noncomputable theory
open Real

def angle_between_line_and_plane (a b c : ℝ → EuclideanSpace ℝ (Fin 3)) (O : EuclideanSpace ℝ (Fin 3)) : Real := 
 let angle_a_b := (90 : ℝ) * π / 180 -- a and b are perpendicular
 let angle_c_a := (45 : ℝ) * π / 180 -- c forms a 45° angle with a
 let angle_c_b := (60 : ℝ) * π / 180 -- c forms a 60° angle with b
 -- Placeholder for calculating the angle φ, note that additional geometric calculations would be needed here
 (30 : ℝ) * π / 180

theorem angle_c_with_plane_is_30_degrees (a b c : ℝ → EuclideanSpace ℝ (Fin 3)) (O : EuclideanSpace ℝ (Fin 3)) 
(h1 : ∀ t, ∥a t - O∥ = t * ∥a 1 - O∥)
(h2 : ∀ t, ∥b t - O∥ = t * ∥b 1 - O∥)
(h3 : ∀ t, ∥c t - O∥ = t * ∥c 1 - O∥) 
(h4 : inner (a 1 - O) (b 1 - O) = 0)
(h5 : arccos (inner (c 1 - O) (a 1 - O) / (∥c 1 - O∥ * ∥a 1 - O∥)) = (45 : ℝ) * π / 180)
(h6 : arccos (inner (c 1 - O) (b 1 - O) / (∥c 1 - O∥ * ∥b 1 - O∥)) = (60 : ℝ) * π / 180) : 
    angle_between_line_and_plane a b c O = (30 : ℝ) * π / 180 :=
sorry

end angle_c_with_plane_is_30_degrees_l536_536596


namespace angle_B_is_60_or_120_l536_536006

/-- Let O and H denote the circumcenter and orthocenter of triangle ABC, respectively.
If BO = BH, then the possible values of ∠B are 60° and 120°.
Note: ∠B is represented in degrees. 
-/
theorem angle_B_is_60_or_120 (A B C : Point) (O H : Point)
  (hO : is_circumcenter O triangle(A, B, C))
  (hH : is_orthocenter H triangle(A, B, C))
  (h_eq : distance B O = distance B H) :
  ∠B = 60 ∨ ∠B = 120 :=
by sorry

end angle_B_is_60_or_120_l536_536006


namespace min_value_reciprocal_sum_l536_536031

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end min_value_reciprocal_sum_l536_536031


namespace orange_preference_percentage_l536_536124

theorem orange_preference_percentage 
  (red blue green yellow purple orange : ℕ)
  (total : ℕ)
  (h_red : red = 75)
  (h_blue : blue = 80)
  (h_green : green = 50)
  (h_yellow : yellow = 45)
  (h_purple : purple = 60)
  (h_orange : orange = 55)
  (h_total : total = red + blue + green + yellow + purple + orange) :
  (orange * 100) / total = 15 :=
by
sorry

end orange_preference_percentage_l536_536124


namespace count_whole_numbers_in_interval_l536_536847

theorem count_whole_numbers_in_interval :
  let a : ℝ := 7 / 4
  let b : ℝ := 3 * Real.pi
  ∀ (x : ℤ), a < x ∧ (x : ℝ) < b → {n : ℤ | a < n ∧ (n : ℝ) < b}.to_finset.card = 8 := sorry

end count_whole_numbers_in_interval_l536_536847


namespace solve_inequality_l536_536277

theorem solve_inequality (x : ℝ) :
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 :=
  sorry

end solve_inequality_l536_536277


namespace combinations_of_coins_l536_536758

def is_valid_combination (p n d q : ℕ) : Prop :=
  p + 5 * n + 10 * d + 25 * q = 50

def count_combinations : ℕ :=
  (Finset.range 51).sum (λ p, 
    (Finset.range 11).sum (λ n, 
      (Finset.range 6).sum (λ d, 
        (Finset.range 2).sum (λ q, if is_valid_combination p n d q then 1 else 0))))

theorem combinations_of_coins : count_combinations = 46 := 
by sorry

end combinations_of_coins_l536_536758


namespace trigonometric_proof_l536_536243

noncomputable def proof_problem (α β : Real) : Prop :=
  (β = 90 - α) → (Real.sin β = Real.cos α) → 
  (Real.sqrt 3 * Real.sin α + Real.sin β) / Real.sqrt (2 - 2 * Real.cos 100) = 1

-- Statement that incorporates all conditions and concludes the proof problem.
theorem trigonometric_proof :
  proof_problem 20 70 :=
by
  intros h1 h2
  sorry

end trigonometric_proof_l536_536243


namespace sum_of_b_n_seq_l536_536295

noncomputable def a_n (a₁ : ℝ) (q : ℝ) (n : ℕ) := a₁ * q ^ (n - 1)
noncomputable def b_n (a₁ : ℝ) (q : ℝ) (n : ℕ) := Real.log 2 (a_n a₁ q n)
noncomputable def S_n (a₁ : ℝ) (q : ℝ) (n : ℕ) := ∑ i in Finset.range n, b_n a₁ q (i + 1)

theorem sum_of_b_n_seq (a₁ q : ℝ) (n : ℕ) (h₁ : 0 < q) (h₂ : q < 1) (h₃ : a₁ * (q ^ 4) + 2 * (a₁ * q ^ (4 - 2)) * (a₁ * q ^ 4) + (a₁ * q) * (a₁ * q ^ 7) = 25) (h₄ : Real.sqrt ((a₁ * q ^ 2) ^ 2) = 2) :
  (∀ n, a_n a₁ q n = 2 ^ (5 - n)) ∧ (S_n 16 (1/4) n = n * (9 - n) / 2) :=
by 
  sorry

end sum_of_b_n_seq_l536_536295


namespace count_whole_numbers_in_interval_l536_536898

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536898


namespace equation_of_line_l536_536491

def point := (ℝ, ℝ)

-- The definition of the line passing through point (2, 3)
def passes_through (l : ℝ → ℝ → Prop) (P : point) : Prop := l P.1 P.2

-- The condition that the intercepts on the axes are opposite numbers
def opposite_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a = -b ∧ (∀ x y : ℝ, l x y ↔ x/a - y/b = 1)

-- The two potential equations for the line
def line1 (x y : ℝ) : Prop := 3 * x - 2 * y = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem equation_of_line (L : ℝ → ℝ → Prop) :
  passes_through L (2, 3) ∧ opposite_intercepts L ↔ (L = line1 ∨ L = line2) := by
  sorry

end equation_of_line_l536_536491


namespace count_whole_numbers_in_interval_l536_536901

theorem count_whole_numbers_in_interval :
  let a := 7 / 4
  let b := 3 * Real.pi
  ∀ x, a < x ∧ x < b ∧ ∃ n : ℤ, x = n → 8 = count (λ n : ℤ, a < n ∧ n < b) := sorry

end count_whole_numbers_in_interval_l536_536901


namespace diane_to_will_age_ratio_l536_536261

-- Defining the given conditions
variables (W : ℕ) (D : ℕ)
def will_age_3_years_ago := 4
def sum_of_ages_in_5_years := 31
def current_age_of_will := W
def current_age_of_diane := D

-- Stating the theorem we want to prove
theorem diane_to_will_age_ratio :
  (W - 3 = will_age_3_years_ago) ∧ ((W + 5) + (D + 5) = sum_of_ages_in_5_years) → (D.to_rat / W.to_rat = 2) := 
by
  sorry

end diane_to_will_age_ratio_l536_536261


namespace num_triangles_2164_l536_536348

noncomputable def is_valid_triangle (p1 p2 p3 : ℤ × ℤ) : Prop :=
  let det := (fun (x1 y1 x2 y2 x3 y3 : ℤ) => x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
  det p1.1 p1.2 p2.1 p2.2 p3.1 p3.2 ≠ 0

noncomputable def num_valid_triangles : ℕ :=
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5}.to_finset.powerset 3
  points.count (λ t, match t.elems with
    | [p1, p2, p3] => is_valid_triangle p1 p2 p3
    | _ => false
  end)

theorem num_triangles_2164 : num_valid_triangles = 2164 := by
  sorry

end num_triangles_2164_l536_536348


namespace roof_ratio_l536_536151

theorem roof_ratio (L W : ℝ) 
  (h1 : L * W = 784) 
  (h2 : L - W = 42) : 
  L / W = 4 := by 
  sorry

end roof_ratio_l536_536151


namespace xz_less_than_half_l536_536403

theorem xz_less_than_half (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : xy + yz + zx = 1) : x * z < 1 / 2 :=
  sorry

end xz_less_than_half_l536_536403


namespace analytical_expression_of_f_monotonically_increasing_intervals_range_of_perimeter_l536_536704

variables (x C : ℝ)

def a : ℝ × ℝ := (1, Real.sin x)
def b : ℝ × ℝ := (Real.cos (2 * x + Real.pi / 3), Real.sin x)

def f (x : ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2 - 1 / 2 * Real.cos (2 * x))

theorem analytical_expression_of_f :
  f(x) = Real.sin(2 * x + 7 * Real.pi / 6) + 1 / 2 := sorry

theorem monotonically_increasing_intervals :
  ∃ k : ℤ, ∀ x, (k * Real.pi - 5 * Real.pi / 6 ≤ 2 * x + 7 * Real.pi / 6 ∧
    2 * x + 7 * Real.pi / 6 ≤ k * Real.pi - Real.pi / 3) → 
    f(x) < f(x + ε) ∧ ∀ ε > 0 ∧ x + ε < upper_bound := sorry

theorem range_of_perimeter (C : ℝ) (hC : f(C) = 0) (c : ℝ) (hc : c = Real.sqrt 3) :
  2 * Real.sqrt 3 < a + b + c ∧ a + b + c ≤ 3 * Real.sqrt 3 := sorry

end analytical_expression_of_f_monotonically_increasing_intervals_range_of_perimeter_l536_536704


namespace particle_returns_to_origin_l536_536592

-- Let T be the period of SHM
def T (m k : ℝ) : ℝ := 2 * Real.pi * Real.sqrt (m / k)

-- The time for the particle to first return to the origin
def time_to_first_return (m k : ℝ) : ℝ := T m k / 2

theorem particle_returns_to_origin (m k : ℝ) (h : time_to_first_return m k = 1.57) : 
  time_to_first_return m k = 1.57 :=
by
  sorry

end particle_returns_to_origin_l536_536592


namespace compound_interest_rate_approx_l536_536178

-- Assuming that the principal, final amount, number of times compounded per year, and number of years are defined
def principal : ℝ := 780
def final_amount : ℝ := 1300
def times_compounded_per_year : ℕ := 4
def years : ℕ := 4

-- Defining the compound interest rate
noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  ((A / P) ^ (1 / (n * t)) - 1) * n

-- Assertion of the value of the compound interest rate
theorem compound_interest_rate_approx :
  compound_interest_rate principal final_amount times_compounded_per_year years ≈ 0.1396 := by
  sorry

end compound_interest_rate_approx_l536_536178


namespace area_of_square_with_adjacent_points_l536_536088

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def side_length := distance 1 2 4 6
def area_of_square (side : ℝ) : ℝ := side ^ 2

theorem area_of_square_with_adjacent_points :
  area_of_square side_length = 25 :=
by
  unfold side_length
  unfold area_of_square
  sorry

end area_of_square_with_adjacent_points_l536_536088


namespace whole_numbers_in_interval_l536_536907

theorem whole_numbers_in_interval : 
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  ∃ (count : ℕ), count = 8 ∧ ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 9 ↔ n ∈ Set.Icc ⌊lower_bound⌋.succ ⌊upper_bound⌋.pred) :=
by
  let lower_bound := (7 : ℚ) / 4
  let upper_bound := 3 * Real.pi
  existsi 8
  split
  { sorry }
  { sorry }

end whole_numbers_in_interval_l536_536907


namespace coin_combination_l536_536780

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536780


namespace square_area_l536_536072

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (1, 2) ∧ p2 = (4, 6)) :
  let d := distance p1 p2 in
  d^2 = 25 :=
by
  sorry

end square_area_l536_536072


namespace period_of_f_cos_theta_l536_536426

open Real

noncomputable def alpha (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin (2 * x), cos x + sin x)

noncomputable def beta (x : ℝ) : ℝ × ℝ :=
  (1, cos x - sin x)

noncomputable def f (x : ℝ) : ℝ :=
  let (α1, α2) := alpha x
  let (β1, β2) := beta x
  α1 * β1 + α2 * β2

theorem period_of_f :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ T : ℝ, (T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) → T = π) :=
sorry

theorem cos_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ f θ = 1 → cos (θ - π / 6) = sqrt 3 / 2 :=
sorry

end period_of_f_cos_theta_l536_536426


namespace triangle_inequality_1_triangle_inequality_2_triangle_inequality_3_l536_536363

-- Definitions for sides of a triangle and its circumradius
variables (a b c R : ℝ)

-- Condition: Assume a, b, and c are the sides of a triangle, and R is the circumradius

-- Proof Problem 1
theorem triangle_inequality_1 (h1 : a^2 + b^2 + c^2 ≤ 9 * R^2) : a^2 + b^2 + c^2 ≤ 9 * R^2 :=
by
  sorry

-- Proof Problem 2
theorem triangle_inequality_2 (h2 : a + b + c ≤ 3 * sqrt 3 * R) : a + b + c ≤ 3 * sqrt 3 * R :=
by
  sorry

-- Proof Problem 3
theorem triangle_inequality_3 (h3 : (a * b * c)^(1/3) ≤ sqrt 3 * R) : (a * b * c)^(1/3) ≤ sqrt 3 * R :=
by
  sorry

end triangle_inequality_1_triangle_inequality_2_triangle_inequality_3_l536_536363


namespace sufficient_not_necessary_condition_l536_536479

theorem sufficient_not_necessary_condition (m : ℝ) :
  (m = 1) → ∃ (a : ℝ), (f(x) = (m^2 - 4m + 4) * x^2) ∧ (∀ (x : ℝ), f(x) = a * x^2) :=
by
  let f (x : ℝ) := (m^2 - 4m + 4) * x^2
  have : m^2 - 4m + 3 = 0 ↔ (m = 1 ∨ m = 3),  -- roots of the polynomial
  sorry
  /- Further proof involving the translation of 'sufficient but not necessary' condition follows -/

end sufficient_not_necessary_condition_l536_536479


namespace range_of_expression_l536_536703

theorem range_of_expression (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : x ≤ 1) 
  (h3 : x + y - 1 ≥ 0) : 
  3 / 2 ≤ (x + y + 2) / (x + 1) ∧ (x + y + 2) / (x + 1) ≤ 3 :=
by
  sorry

end range_of_expression_l536_536703


namespace fifth_number_selected_l536_536365

-- Define the necessary conditions
def num_students : ℕ := 60
def sample_size : ℕ := 5
def first_selected_number : ℕ := 4
def interval : ℕ := num_students / sample_size

-- Define the proposition to be proved
theorem fifth_number_selected (h1 : 1 ≤ first_selected_number) (h2 : first_selected_number ≤ num_students)
    (h3 : sample_size > 0) (h4 : num_students % sample_size = 0) :
  first_selected_number + 4 * interval = 52 :=
by
  -- Proof omitted
  sorry

end fifth_number_selected_l536_536365


namespace units_digit_of_product_l536_536542

-- Define the three given even composite numbers
def a := 4
def b := 6
def c := 8

-- Define the product of the three numbers
def product := a * b * c

-- State the units digit of the product
theorem units_digit_of_product : product % 10 = 2 :=
by
  -- Proof is skipped here
  sorry

end units_digit_of_product_l536_536542


namespace whole_numbers_in_interval_7_4_3pi_l536_536864

noncomputable def num_whole_numbers_in_interval : ℕ :=
  let lower := (7 : ℝ) / (4 : ℝ)
  let upper := 3 * Real.pi
  Finset.card (Finset.filter (λ x, lower < (x : ℝ) ∧ (x : ℝ) < upper) (Finset.range 10))

theorem whole_numbers_in_interval_7_4_3pi :
  num_whole_numbers_in_interval = 8 := by
-- Proof logic will be added here
sorry

end whole_numbers_in_interval_7_4_3pi_l536_536864


namespace base_representation_l536_536284

theorem base_representation (b : ℕ) (h₁ : b^2 ≤ 125) (h₂ : 125 < b^3) :
  (∀ b, b = 12 → 125 % b % 2 = 1) → b = 12 := 
by
  sorry

end base_representation_l536_536284


namespace max_extra_credit_l536_536046

theorem max_extra_credit (n : ℕ) (n = 200)
  (scores : Fin n → ℕ)
  (extra_credit : ℕ → Prop)
  (mean : ℚ := (∑ i, scores i) / n) :
  ∀ (k : ℕ), k = 199 →
  (∀ i < n, scores i > mean → extra_credit i) →
  k ≤ n - 1 :=
by
  -- Each participant's work is awarded points
  -- Each participant with score exceeding the mean gets extra credit
  -- ∑ i, if scores i > mean then 1 else 0 <= n - 1
  sorry

end max_extra_credit_l536_536046


namespace tile_difference_l536_536172

theorem tile_difference :
  let initial_blue : ℕ := 20
  let initial_green : ℕ := 9
  let added_green : ℕ := 9
  let total_green := initial_green + added_green
  let total_blue := initial_blue
  total_green - total_blue = -2 := by
  sorry

end tile_difference_l536_536172


namespace compound_interest_rate_13_97_percent_l536_536180

-- Definitions of constants and conditions
def P : ℝ := 780
def A : ℝ := 1300
def n : ℝ := 4
def t : ℝ := 4

-- Compound interest formula
def compound_interest_formula (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

-- The theorem we want to prove
theorem compound_interest_rate_13_97_percent :
  ∃ r : ℝ, compound_interest_formula r ∧ r ≈ 0.1397396 := 
begin
  -- Placeholder for the proof.
  sorry,
end

end compound_interest_rate_13_97_percent_l536_536180


namespace min_inv_sum_l536_536021

open Real

theorem min_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) :
  min ((1 / x) + (1 / y)) (1 / 3) :=
sorry

end min_inv_sum_l536_536021


namespace sin_cos_of_angle_l536_536297

theorem sin_cos_of_angle (a : ℝ) (h₀ : a ≠ 0) :
  ∃ (s c : ℝ), (∃ (k : ℝ), s = k * (8 / 17) ∧ c = -k * (15 / 17) ∧ k = if a > 0 then 1 else -1) :=
by
  sorry

end sin_cos_of_angle_l536_536297


namespace count_ways_to_get_50_cents_with_coins_l536_536825

/-- A structure to represent coin counts for pennies, nickels, dimes, and quarters -/
structure CoinCount :=
  (p : ℕ) -- number of pennies
  (n : ℕ) -- number of nickels
  (d : ℕ) -- number of dimes
  (q : ℕ) -- number of quarters

/-- Predicate to represent the total value equation -/
def is_valid_combo (c : CoinCount) : Prop :=
  c.p + 5 * c.n + 10 * c.d + 25 * c.q = 50

/-- Definition to represent the total number of valid combinations -/
def total_combinations (l : list CoinCount) : ℕ :=
  l.filter is_valid_combo |>.length

/- The main theorem we want to prove -/
theorem count_ways_to_get_50_cents_with_coins :
  ∃ l, total_combinations l = 38 :=
sorry

end count_ways_to_get_50_cents_with_coins_l536_536825


namespace smallest_integer_k_l536_536537

theorem smallest_integer_k : 
  ∃ k : ℕ, k > 1 ∧ k % 17 = 1 ∧ k % 11 = 1 ∧ k % 6 = 2 ∧ k = 188 := 
by {
  have solution : 188 % 17 = 1 ∧ 188 % 11 = 1 ∧ 188 % 6 = 2 ∧ 188 > 1,
  { split, norm_num, split, norm_num, split, norm_num, norm_num },
  use 188,
  finish,
}

end smallest_integer_k_l536_536537


namespace largest_f_10_l536_536410

noncomputable def f (x : ℝ) : ℝ := sorry -- placeholder for the polynomial

theorem largest_f_10 :
  (∀ x, ∃ a_n, ∃ a_0, ∀ (i : ℕ), 0 ≤ a_i ∧ f 5 = 25 ∧ f 20 = 1024) → f 10 ≤ 100 :=
by
  sorry

end largest_f_10_l536_536410


namespace square_area_adjacency_l536_536052

-- Definition of points as pairs of integers
def Point := ℤ × ℤ

-- Define the points (1,2) and (4,6)
def P1 : Point := (1, 2)
def P2 : Point := (4, 6)

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement for proving the area of a square given the side length
theorem square_area_adjacency (h : distance P1 P2 = 5) : ∃ area : ℝ, area = 25 :=
by
  use 25
  sorry

end square_area_adjacency_l536_536052


namespace sequence_pairs_count_l536_536368

def is_ascending_pair (a : Fin 1000 → ℕ) (i j : Fin 1000) : Prop :=
  i < j ∧ a i < a j

def is_descending_pair (a : Fin 1000 → ℕ) (i j : Fin 1000) : Prop :=
  i < j ∧ a i > a j

theorem sequence_pairs_count {a : Fin 1000 → ℕ} (h_distinct : Function.Injective a) :
  ∃ k : ℕ, k = 333 ∧ (∃ pairs : Finset (Fin 1000 × Fin 1000),
    (∀ p ∈ pairs, is_ascending_pair a p.1 p.2) ∨ (∀ p ∈ pairs, is_descending_pair a p.1 p.2)) ∧ pairs.card ≥ k :=
sorry

end sequence_pairs_count_l536_536368


namespace max_Sn_at_5_or_6_l536_536153

variable {α : Type*} [LinearOrderedField α]

-- Definitions and assumptions
def first_term (a1 : α) := a1 > 0

def Sn_eq (S_3 S_8 : α) : Prop := S_3 = S_8

theorem max_Sn_at_5_or_6 (a1 S_3 S_8 d : α) (n : ℕ) 
  (h1 : first_term a1)
  (h2 : Sn_eq S_3 S_8)
  (h3 : ∀ k : ℕ, k ∈ {5, 6} → S_3 + 3 * d = 0) : 
  n = 5 ∨ n = 6 :=
by
  sorry

end max_Sn_at_5_or_6_l536_536153


namespace slope_of_tangent_l536_536323

theorem slope_of_tangent {x : ℝ} (h : x = 2) : deriv (λ x, 2 * x^2) x = 8 :=
by
  sorry

end slope_of_tangent_l536_536323


namespace hyperbola_eccentricity_l536_536216

noncomputable def calculate_eccentricity (a b c x0 y0 : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity :
  ∀ (a b c x0 y0 : ℝ),
    (c = 2) →
    (a^2 + b^2 = 4) →
    (x0 = 3) →
    (y0^2 = 24) →
    (5 = x0 + 2) →
    calculate_eccentricity a b c x0 y0 = 2 := 
by 
  intros a b c x0 y0 h1 h2 h3 h4 h5
  sorry

end hyperbola_eccentricity_l536_536216


namespace range_of_x_l536_536040

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1 else 2 ^ x

theorem range_of_x (x : ℝ) :
  f(x) + f (x - 1/2) > 1 ↔ x > -1/4 :=
sorry

end range_of_x_l536_536040


namespace find_x_y_l536_536929

theorem find_x_y (x y : ℝ) (h : (2 * x - 3 * y + 5) ^ 2 + |x - y + 2| = 0) : x = -1 ∧ y = 1 :=
by
  sorry

end find_x_y_l536_536929


namespace F_with_P2019_has_P595_l536_536205

-- Declare the set S of 35 elements
def S : Type := Fin 35

-- Define the set of mappings F from S to itself
def F := S → S

-- Property P(k) definition
def P (k : ℕ) (F : Set F) : Prop :=
  ∀ x y : S, ∃ (f : Fin k → F), (f (Fin.last k)).val (f ⟨k - 1, sorry⟩.val (⋯ (f ⟨1, sorry⟩.val (f ⟨0, sorry⟩.val x)))) = 
                                  (f (Fin.last k)).val (f ⟨k - 1, sorry⟩.val (⋯ (f ⟨1, sorry⟩.val (f ⟨0, sorry⟩.val y)))

-- To check if F with property P(2019) also has property P(595)
theorem F_with_P2019_has_P595 (F : Set F) (h : P 2019 F) : P 595 F :=
  sorry

end F_with_P2019_has_P595_l536_536205


namespace area_of_square_with_adjacent_points_l536_536081

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l536_536081


namespace triangle_proportions_l536_536362

theorem triangle_proportions :
  ∀ (A B C : ℝ) (a b c : ℝ), 
  A = 60 ∧ b = 1 ∧ (1/2) * b * c * Real.sin (A.toRadians) = sqrt 3 → 
  (a + b + c) / (Real.sin (A.toRadians) + Real.sin (B.toRadians) + Real.sin (C.toRadians)) = 2 * sqrt 39 / 3 :=
by
  intros A B C a b c h,
  cases h with hA h1,
  cases h1 with hb hArea,
  sorry

end triangle_proportions_l536_536362


namespace quotient_of_division_l536_536131

theorem quotient_of_division (x q : ℕ) (h1 : 1596 - x = 1345) (h2 : 1596 = 251 * q + 15) : q = 6 := 
by 
-- We assert x = 251 from h1
have hx : x = 251,
{
  linarith,
},
-- Substitute x = 251 into h2 to prove q = 6
have hq : 1596 = 251 * q + 15, from h2,
calc
  q = 6 : sorry

end quotient_of_division_l536_536131


namespace coin_combination_l536_536778

theorem coin_combination (p n d q : ℕ) :
  (p = 1 ∧ n = 5 ∧ d = 10 ∧ q = 25) →
  ∃ (c : ℕ), c = 50 ∧ 
  ∃ (a b c d : ℕ), 
    a * p + b * n + c * d + d * q = 50 ∧ 
    (∑ x in finset.range (a + 1), 
    finset.range (b + 1).card * 
    finset.range (c + 1).card * 
    finset.range (d + 1).card) = 50 := 
by
  sorry

end coin_combination_l536_536778


namespace regular_dodecagon_triangulations_count_l536_536299

theorem regular_dodecagon_triangulations_count :
  ∃ n : ℕ, (n = 4) ∧ ∀ (P : polygon 12) (T : Π (i j : fin 12), P.adjacent i j → {v : fin 12 | T v = true} → card {v : fin 12 | T v = true} % 2 = 1),
  n = 4 := sorry

end regular_dodecagon_triangulations_count_l536_536299


namespace tangent_fixed_point_l536_536303

-- Define the circle C equation and the line equation where point P lies
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2
def line_P (x y : ℝ) : Prop := (x / 3) + (y / 6) = 1

-- Define the fixed point coordinates we are trying to prove
def fixed_point : ℝ × ℝ := (2 / 3, 1 / 3)

-- The main theorem statement
theorem tangent_fixed_point :
  ∀ (x y : ℝ), 
  line_P x y → 
  ∃ a b : ℝ, (circle_C a b → (line_AB_through_P a b x y) → (a, b) = fixed_point) :=
by
  sorry

end tangent_fixed_point_l536_536303


namespace square_area_from_points_l536_536067

theorem square_area_from_points :
  let P1 := (1, 2)
  let P2 := (4, 6)
  let side_length := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let area := side_length^2
  P1.1 = 1 ∧ P1.2 = 2 ∧ P2.1 = 4 ∧ P2.2 = 6 →
  area = 25 :=
by
  sorry

end square_area_from_points_l536_536067


namespace count_whole_numbers_in_interval_l536_536877

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l536_536877


namespace problem_1_problem_2_l536_536560

-- Proof Problem 1
theorem problem_1 :
  (2 + 1/4)^(1/2) - (-0.96)^0 - (3 + 3/8)^(-2/3) + (3/2)^(-2) + ((-32)^(-4))^(-3/4) = 5/2 :=
  sorry

-- Proof Problem 2
theorem problem_2 (a b : ℝ) (h1 : 14^a = 6) (h2 : 14^b = 7) :
  log 42 56 = (3 - 2 * b) / (a + b) :=
  sorry

end problem_1_problem_2_l536_536560


namespace current_population_l536_536370

theorem current_population (p_0 : ℕ) (died_rate fear_rate : ℚ) (initial_population_final: p_0 = 3800)
(died_final : died_rate = 0.10) (fear_final: fear_rate = 0.15) : 
    let died := p_0 * died_rate;
    let remaining := p_0 - died;
    let left := remaining * fear_rate;
    let current := remaining - left in
    current = 2907 := 
by
  -- Using let definitions to avoid using solution steps directly
  rw [initial_population_final, died_final, fear_final]
  let died := p_0 * died_rate;
  let remaining := p_0 - died;
  let left := remaining * fear_rate;
  let current := remaining - left;
  show current = 2907 from sorry

end current_population_l536_536370


namespace count_reaches_five_l536_536622

def g (n : ℕ) : ℕ :=
if n % 2 = 1 then 2 * n + 3 else n / 2

def reaches_five (n : ℕ) : Prop :=
∃ k : ℕ, (iterate g k n = 5)

theorem count_reaches_five : (finset.filter reaches_five (finset.range 51)).card = 5 :=
sorry

end count_reaches_five_l536_536622


namespace part_1_a2009_correct_part_2_bn_correct_l536_536694

noncomputable def sequence_an (n : ℕ) : ℤ :=
  2 * n + 1

def sequence_bn (n : ℕ) : ℤ :=
  4 * n + 1

theorem part_1_a2009_correct :
  sequence_an 2009 = 4019 :=
by norm_num

theorem part_2_bn_correct (n : ℕ) :
  sequence_bn n = 4 * n + 1 :=
by norm_num

end part_1_a2009_correct_part_2_bn_correct_l536_536694


namespace robbery_participants_l536_536193

theorem robbery_participants : 
  ∃ (participants : Finset String), 
    participants = { "Charlie", "James" } ∧ 
    participants.card = 2 ∧ 
    (("Harry" = "Charlie" ∨ "Harry" = "George") + 
     ("James" = "Donald" ∨ "James" = "Tom") + 
     ("Donald" = "Tom" ∨ "Donald" = "Charlie") + 
     ("George" = "Harry" ∨ "George" = "Charlie") + 
     ("Charlie" = "Donald" ∨ "Charlie" = "James") = 1 + 
       (sum fun (p : Finset String) => if (Finset.mem "Charlie" p ∧ ¬Finset.mem "James" p ) then 1 else 
                                      else if Finset.member "James" p then 1 else 0)) :=
begin
  sorry
end

end robbery_participants_l536_536193


namespace parallelogram_in_convex_quadrilateral_l536_536993

theorem parallelogram_in_convex_quadrilateral (A B C D : Type) [convex_quadrilateral A B C D] :
  ∃ (E : Type), is_parallelogram A B C E ∧
  E.1 = A ∧ E.2 = B ∧ E.3 = C ∧ E.4 ∈ convex_hull A B C D :=
sorry

end parallelogram_in_convex_quadrilateral_l536_536993


namespace infinite_spheres_volume_l536_536972

theorem infinite_spheres_volume :
  (∀ (A B : ℝ), A = B / 3 → ∀ (h: ℝ), h = 10 →
  let r₁ := 1 in
  let V₁ := (4 / 3) * real.pi * r₁^3 in
  let k := 1 / 2 in
  let total_volume := V₁ / (1 - k^3) in
  total_volume = 4 * real.pi) :=
begin
  intros A B hAB h h_height r₁ V₁ k total_volume,
  rw [hAB, h],
  rw [show r₁ = 1, by refl,
      show V₁ = (4 / 3) * real.pi * 1^3, by refl,
      show k = 1 / 2, by refl],
  calc (4 / 3 * real.pi * 1^3) / (1 - (1 / 2)^3) 
      = 4 * real.pi : by sorry
end

end infinite_spheres_volume_l536_536972


namespace basketball_not_table_tennis_l536_536970

theorem basketball_not_table_tennis (total_students likes_basketball likes_table_tennis dislikes_all : ℕ) (likes_basketball_not_tt : ℕ) :
  total_students = 30 →
  likes_basketball = 15 →
  likes_table_tennis = 10 →
  dislikes_all = 8 →
  (likes_basketball - 3 = likes_basketball_not_tt) →
  likes_basketball_not_tt = 12 := by
  intros h_total h_basketball h_table_tennis h_dislikes h_eq
  sorry

end basketball_not_table_tennis_l536_536970


namespace area_of_square_l536_536058

-- We define the points as given in the conditions
def point1 : ℝ × ℝ := (1, 2)
def point2 : ℝ × ℝ := (4, 6)

-- Lean's "def" defines the concept of a square given two adjacent points.
def is_square (p1 p2: ℝ × ℝ) : Prop :=
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  in ∃ (l : ℝ), l = d ∧ (l^2 = 25)

-- The theorem assumes the points are adjacent points on a square and proves that their area is 25.
theorem area_of_square :
  is_square point1 point2 :=
by
  -- Insert formal proof here, skipped with 'sorry' for this task
  sorry

end area_of_square_l536_536058


namespace coin_combinations_count_l536_536737

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536737


namespace additional_flour_in_ounces_l536_536997

-- Define the given conditions
def original_flour (x : ℝ) := x = 7
def already_added (y : ℝ) := y = 3.75
def cup_to_ounces (c o : ℝ) := o = c * 8

-- Translate conditions to Lean
theorem additional_flour_in_ounces (x y z o : ℝ) (h1 : original_flour x) (h2 : already_added y) (h3 : cup_to_ounces z o) : o = 82 :=
  by
    -- Define given values based on conditions
    have h4 : x = 7 := h1
    have h5 : y = 3.75 := h2
    have h6 : z = (2 * x) - y := 
      by 
        rw [h4, h5]
        norm_num
    -- Prove the statement
    rw [h3, h6]
    norm_num
    sorry

#check additional_flour_in_ounces

end additional_flour_in_ounces_l536_536997


namespace count_whole_numbers_in_interval_l536_536895

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l536_536895


namespace triangle_area_l536_536682

noncomputable def ellipse_equation(a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 ∧ c / a = real.sqrt 2 / 2 ∧ a > b ∧ b > 0 ∧ c > 0

noncomputable def point_on_ellipse(a b : ℝ) (x y : ℝ) : Prop :=
  x / a ^ 2 + y / b ^ 2 = 1

noncomputable def find_area_triangle(m x1 y1 x2 y2 : ℝ) : ℝ :=
  (3 / 2) * real.sqrt ((x1 + x2) ^ 2 - 4 * x1 * x2)

theorem triangle_area 
  (a b c x1 y1 x2 y2 : ℝ)
  (h1 : ellipse_equation a b c)
  (h2 : point_on_ellipse a b (-2) 0)
  (h3 : (1 / y1) + (1 / y2) = (1 / (6 * y1 / (x1 + 2))) + (1 / (6 * y2 / (x2 + 2))))
  : find_area_triangle (-1) x1 y1 x2 y2 = real.sqrt 10 := 
sorry

end triangle_area_l536_536682


namespace coin_combinations_sum_50_l536_536713

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536713


namespace area_of_square_with_adjacent_points_l536_536079

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l536_536079


namespace coin_combinations_sum_50_l536_536718

/--
Given the values of pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents),
prove that the total number of combinations of these coins that sum to 50 cents is 42.
-/
theorem coin_combinations_sum_50 : 
  ∃ (p n d q : ℕ), 
    (p + 5 * n + 10 * d + 25 * q = 50) → 42 :=
sorry

end coin_combinations_sum_50_l536_536718


namespace provisions_last_for_girls_l536_536160

theorem provisions_last_for_girls (P : ℝ) (G : ℝ) (h1 : P / (50 * G) = P / (250 * (G + 20))) : G = 25 := 
by
  sorry

end provisions_last_for_girls_l536_536160


namespace increasing_geometric_progression_l536_536493

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem increasing_geometric_progression (a : ℝ) (ha : 0 < a)
  (h1 : ∃ b c q : ℝ, b = Int.floor a ∧ c = a - b ∧ a = b + c ∧ c = b * q ∧ a = c * q ∧ 1 < q) : 
  a = golden_ratio :=
sorry

end increasing_geometric_progression_l536_536493


namespace minimum_value_frac_inv_is_one_third_l536_536016

noncomputable def min_value_frac_inv (x y : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ x + y = 12 then 1/x + 1/y else 0

theorem minimum_value_frac_inv_is_one_third (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x + y = 12) :
  min_value_frac_inv x y = 1/3 :=
begin
  -- Proof to be provided
  sorry
end

end minimum_value_frac_inv_is_one_third_l536_536016


namespace no_five_consecutive_interesting_numbers_l536_536623

def E (n : ℕ) : ℕ :=
  (Nat.digits 2 n).count (λ d, d = 1)

def interesting (n : ℕ) : Prop :=
  E n ∣ n

theorem no_five_consecutive_interesting_numbers :
  ¬ ∃ (n : ℕ), interesting n ∧ interesting (n + 1) ∧ interesting (n + 2) ∧ interesting (n + 3) ∧ interesting (n + 4) :=
  sorry

end no_five_consecutive_interesting_numbers_l536_536623


namespace stream_speed_l536_536510

theorem stream_speed (x : ℝ) (hb : ∀ t, t = 48 / (20 + x) → t = 24 / (20 - x)) : x = 20 / 3 :=
by
  have t := hb (48 / (20 + x)) rfl
  sorry

end stream_speed_l536_536510


namespace seq_stabilization_l536_536646

noncomputable def a_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
if h : n > 2017 then
  let s := { p : ℕ × ℕ × ℕ // p.1 + p.2.1 + p.2.2 = n ∧ 1 ≤ p.1 ∧ p.1 ≤ p.2.1 ∧ p.2.1 ≤ p.2.2 ∧ p.2.2 ≤ n-1} in
  let mx := (finset.sup finset.univ (λ (p : s), a p.1 * a p.2.1 * a p.2.2)) in
  mx
else
  a n

theorem seq_stabilization (a : ℕ → ℝ) :
  (∀ n ≤ 2017, a n > 0) →
  ∃ m, m ≤ 2017 ∧ ∃ N, N > 4 * m ∧ ∀ n, n > N → (a_seq a n) * (a_seq a (n - 4*m)) = (a_seq a (n - 2*m))^2 :=
begin
  sorry
end

end seq_stabilization_l536_536646


namespace probability_of_convex_quadrilateral_l536_536631

theorem probability_of_convex_quadrilateral (n : ℕ) (h : n = 8) : 
  let total_chords := Nat.choose n 2,
      total_ways_select_4_chords := Nat.choose total_chords 4,
      ways_select_4_points := Nat.choose n 4
  in (ways_select_4_points : ℚ) / (total_ways_select_4_chords : ℚ) = 2 / 585 :=
by
  -- Additional context and variable unfolding to make the statement explicit
  let total_chords := Nat.choose 8 2,
      total_ways_select_4_chords := Nat.choose total_chords 4,
      ways_select_4_points := Nat.choose 8 4
  have h1 : total_chords = 28 := by sorry, -- Calculation of total chords
  have h2 : total_ways_select_4_chords = 20475 := by sorry, -- Calculation of ways to select 4 chords
  have h3 : ways_select_4_points = 70 := by sorry, -- Calculation of ways to select 4 points
  exact calc
    (ways_select_4_points : ℚ) / (total_ways_select_4_chords : ℚ)
      = (70 : ℚ) / 20475 : by rw [h2, h3]
      ... = 2 / 585 : by norm_num

end probability_of_convex_quadrilateral_l536_536631


namespace coin_combinations_count_l536_536740

-- Definitions for the values of different coins.

def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def total_value := 50

-- Statement of the theorem

theorem coin_combinations_count :
  (∃ (pennies nickels dimes quarters : ℕ),
    pennies * penny_value + nickels * nickel_value +
    dimes * dime_value + quarters * quarter_value = total_value) →
  16 :=
begin
  sorry
end

end coin_combinations_count_l536_536740


namespace range_of_a_l536_536698

theorem range_of_a (a : ℝ) : 
  (set.card {x : ℤ | a^2 - a < (x : ℝ) ∧ (x : ℝ) < 2} = 2) → (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l536_536698


namespace adding_sugar_percentage_l536_536044

theorem adding_sugar_percentage (calories_soft_drink : ℕ) (added_sugar_intake : ℕ) (candy_bars : ℕ) (calories_per_bar : ℕ)
  (h1 : calories_soft_drink = 2500)
  (h2 : added_sugar_intake = 150)
  (h3 : candy_bars = 7)
  (h4 : calories_per_bar = 25)
  (h5 : 100% of recommended intake exceeded) :
  100 * (added_sugar_coke / calories_coke) = 5 :=
by
  sorry

end adding_sugar_percentage_l536_536044


namespace geom_seq_count_l536_536111

def is_geom_seq (a b c : ℕ) : Prop :=
  b * b = a * c

def valid_numbers : Finset ℕ := Finset.range 11 \\ {0}

theorem geom_seq_count : 
  (Finset.filter (λ s : Finset ℕ, ∃ a b c ∈ s, is_geom_seq a b c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c))
    (Finset.powerset_len 3 valid_numbers)).card = 6 :=
by
  sorry

end geom_seq_count_l536_536111


namespace staff_duty_arrangement_l536_536599

theorem staff_duty_arrangement :
  let staff := {1, 2, 3, 4, 5, 6, 7}
  let days := {1, 2, 3, 4, 5, 6, 7}
  let last_5_days := {3, 4, 5, 6, 7}
  let arrangements := finset.permutations staff
  (∃ arr ∈ arrangements, ∀ i ∈ {1, 2}, arr i ≠ A ∧ arr i ≠ B) →
  finset.card arrangements = 2400 :=
by
  sorry

end staff_duty_arrangement_l536_536599
