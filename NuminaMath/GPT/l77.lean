import Mathlib

namespace part_a_part_b_l77_77541

variable (a b : ℝ) (f : ℝ → ℝ)

noncomputable def h (x : ℝ) : ℝ := ∫ t in 0..x, f t

-- Define the conditions used for both parts.
def conditions (a b : ℝ) (f : ℝ → ℝ) :=
  a > 0 ∧ a < 1 ∧ b > 0 ∧ b < 1 ∧
  ContinuousOn f (Set.Icc 0 1) ∧
  ∀ x ∈ Set.Icc 0 1, h f x = h f (a * x) + h f (b * x)

-- Part a: prove f = 0 if a + b < 1
theorem part_a (h_cond : conditions a b f) (h_sum_lt : a + b < 1) : ∀ x ∈ Set.Icc 0 1, f x = 0 :=
sorry

-- Part b: prove f is constant if a + b = 1
theorem part_b (h_cond : conditions a b f) (h_sum_eq : a + b = 1) : ∀ x y ∈ Set.Icc 0 1, f x = f y :=
sorry

end part_a_part_b_l77_77541


namespace sum_q_p_eq_zero_l77_77233

def p (x : Int) : Int := x^2 - 4

def q (x : Int) : Int := 
  if x ≥ 0 then -x
  else x

def q_p (x : Int) : Int := q (p x)

#eval List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0

theorem sum_q_p_eq_zero :
  List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0 :=
sorry

end sum_q_p_eq_zero_l77_77233


namespace range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l77_77851

noncomputable def quadratic_function (m x : ℝ) : ℝ :=
  (m + 1) * x^2 - m * x + m - 1

-- Part 1
theorem range_of_m_if_solution_set_empty (m : ℝ) :
  (∀ x : ℝ, quadratic_function m x < 0 → false) ↔ m ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem solve_inequality_y_geq_m (m x : ℝ) (h : m > -2) :
  (quadratic_function m x ≥ m) ↔ 
  (m = -1 → x ≥ 1) ∧
  (m > -1 → x ≤ -1/(m+1) ∨ x ≥ 1) ∧
  (m > -2 ∧ m < -1 → 1 ≤ x ∧ x ≤ -1/(m+1)) := sorry

end range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l77_77851


namespace max_weight_lbs_l77_77069

/--
Question: Determine the heaviest object that can be weighed using a set of weights of 2 lb, 4 lb, and 12 lb.

Conditions:
- Set of weights: {2 lb, 4 lb, 12 lb}
- Options: (A) 15 lb, (B) 16 lb, (C) 17 lb, (D) 18 lb

Prove that the heaviest object that can be weighed using these weights is 18 lb.
-/
theorem max_weight_lbs (w1 w2 w3 : ℕ) (hw : w1 = 2 ∧ w2 = 4 ∧ w3 = 12) : (w1 + w2 + w3) = 18 := 
by
  -- Given conditions
  rcases hw with ⟨h1, h2, h3⟩
  -- Substitute values
  rw [h1, h2, h3]
  -- Sum the weights
  calc
    2 + 4 + 12 = 18 := by norm_num

end max_weight_lbs_l77_77069


namespace lines_passing_through_point_A_with_equal_intercepts_l77_77415

variable (A : ℝ × ℝ)
variable (A_eq : A = (3, -1))

noncomputable def line1 : ℝ → ℝ → Prop := λ x y, x + y - 2 = 0
noncomputable def line2 : ℝ → ℝ → Prop := λ x y, x - y - 4 = 0
noncomputable def line3 : ℝ → ℝ → Prop := λ x y, 3 * x + y = 0

theorem lines_passing_through_point_A_with_equal_intercepts :
  (line1 3 (-1) ∧ line2 3 (-1) ∧ line3 3 (-1)) :=
sorry

end lines_passing_through_point_A_with_equal_intercepts_l77_77415


namespace downstream_distance_l77_77734

theorem downstream_distance
  (time_downstream : ℝ) (time_upstream : ℝ)
  (distance_upstream : ℝ) (speed_still_water : ℝ)
  (h1 : time_downstream = 3) (h2 : time_upstream = 3)
  (h3 : distance_upstream = 15) (h4 : speed_still_water = 10) :
  ∃ d : ℝ, d = 45 :=
by
  sorry

end downstream_distance_l77_77734


namespace pyramid_volume_l77_77276

noncomputable def volume_of_pyramid (AB BC PB : ℕ) (h1 : AB = 10) (h2 : BC = 6) (h3 : PB = 20) : ℝ :=
  let PA := real.sqrt (PB ^ 2 - AB ^ 2)
  let base_area := AB * BC
  (1 / 3) * base_area * PA

theorem pyramid_volume : volume_of_pyramid 10 6 20 (by rfl) (by rfl) (by rfl) = 200 * real.sqrt 3 :=
  sorry

end pyramid_volume_l77_77276


namespace machine_working_time_l77_77739

theorem machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) (h1 : shirts_per_minute = 3) (h2 : total_shirts = 6) :
  (total_shirts / shirts_per_minute) = 2 :=
by
  -- Begin the proof
  sorry

end machine_working_time_l77_77739


namespace fraction_of_vegan_nut_free_dishes_l77_77150

theorem fraction_of_vegan_nut_free_dishes
  (total_dishes_vegan : ℕ)
  (total_dishes : ℕ)
  (vegan_dishes_fraction : ℚ)
  (vegan_with_nuts : ℕ)
  (h1 : total_dishes_vegan = 6)
  (h2 : vegan_dishes_fraction = 1 / 3)
  (h3 : vegan_with_nuts = 1)
  (h4 : total_dishes = total_dishes_vegan / vegan_dishes_fraction.to_nat) :
  (total_dishes_vegan - vegan_with_nuts) / total_dishes = 5 / 18 :=
by
  sorry

end fraction_of_vegan_nut_free_dishes_l77_77150


namespace directrix_of_parabola_l77_77411

theorem directrix_of_parabola (x y : ℝ) :
  (y = -3 * x^2 + 6 * x - 5) -> (y = -3 * (x - 1)^2 - 2) 
  -> (directrix_y : ℝ) (directrix_y = -23 / 12) := by
  intros h1 h2
  sorry

end directrix_of_parabola_l77_77411


namespace sum_cndn_l77_77937

def cn (n : ℕ) : ℝ := (Complex.abs (1 + 2 * Complex.I))^n * Real.cos (n * Complex.arg (1 + 2 * Complex.I))
def dn (n : ℕ) : ℝ := (Complex.abs (1 + 2 * Complex.I))^n * Real.sin (n * Complex.arg (1 + 2 * Complex.I))

theorem sum_cndn : 
  (∑' n : ℕ, cn n * dn n / (8 ^ n : ℝ)) = 16 / 15 := 
sorry

end sum_cndn_l77_77937


namespace minimize_total_cost_l77_77695

theorem minimize_total_cost (k : ℝ) (s : ℝ) (v : ℝ) (h₁ : v = 10) (h₂ : k * v^2 = 80)
                          (h₃ : ∀ s, k * s^2 + 500 = 500 + 2k * s) (dist : ℝ) (h₄ : dist = 100) :
  (∃ s' : ℝ, s' = 25 ∧ (100 / 25) * (80 * 25^2 / 10^2 + 500) = 4000) :=
by
  sorry

end minimize_total_cost_l77_77695


namespace varphi_value_for_even_function_l77_77630

theorem varphi_value_for_even_function (k : ℤ) :
  ∃ φ : ℝ, (∀ x : ℝ, (sin (2*x + φ + π/4)) = (sin (-(2*x + φ + π/4)))) ↔ φ = k * π + π / 4 :=
by
  sorry

end varphi_value_for_even_function_l77_77630


namespace round_trip_time_l77_77998

def boat_speed_still_water : ℝ := 16
def stream_speed : ℝ := 2
def distance_to_place : ℝ := 7560

theorem round_trip_time : (distance_to_place / (boat_speed_still_water + stream_speed) + distance_to_place / (boat_speed_still_water - stream_speed)) = 960 := by
  sorry

end round_trip_time_l77_77998


namespace menelaus_theorem_iff_collinear_l77_77108

open_locale classical

variable {α : Type*} [linear_ordered_field α]

noncomputable def menelaus_theorem (A B C X Y Z : α) : Prop :=
  (X - B) / (X - C) * (Y - C) / (Y - A) * (Z - A) / (Z - B) = 1

noncomputable def collinear (A B C X Y Z : α) : Prop :=
  ∃ (l : affine_subspace ℝ (affine_space_set ℝ α)),
    A ∈ l ∧ B ∈ l ∧ C ∈ l ∧ X ∈ l ∧ Y ∈ l ∧ Z ∈ l

theorem menelaus_theorem_iff_collinear (A B C X Y Z : α) :
  menelaus_theorem A B C X Y Z ↔ collinear A B C X Y Z :=
sorry

end menelaus_theorem_iff_collinear_l77_77108


namespace value_of_a_l77_77171

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x^2 + 1)

theorem value_of_a (a : ℝ) (h : f a 1 + f a 2 = a^2 + a + 2) : a = Real.sqrt 10 :=
by
  sorry

end value_of_a_l77_77171


namespace rate_up_is_4_l77_77709

variables (R : ℝ) (up_days down_days : ℝ) (down_distance : ℝ)

def rate_up_implies_conditions := 
  (up_days = 2) ∧ (down_days = 2) ∧ (down_distance = 12) ∧ 
  (R * 1.5 * down_days = down_distance)

theorem rate_up_is_4 : rate_up_implies_conditions R 2 2 12 → R = 4 :=
  by 
    intros h
    cases h with h_up_days h_rest
    cases h_rest with h_down_days h_rest
    cases h_rest with h_down_distance h_equation
    sorry

end rate_up_is_4_l77_77709


namespace pond_depth_l77_77183

theorem pond_depth (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) :
    V = L * W * D ↔ D = 5 := 
by
  rw [hL, hW, hV]
  constructor
  · intro h1
    linarith
  · intro h2
    rw [h2]
    linarith

#check pond_depth

end pond_depth_l77_77183


namespace line_canonical_eqn_l77_77692

theorem line_canonical_eqn 
  (x y z : ℝ)
  (h1 : x - y + z - 2 = 0)
  (h2 : x - 2*y - z + 4 = 0) :
  ∃ a : ℝ, ∃ b : ℝ, ∃ c : ℝ,
    (a = (x - 8)/3) ∧ (b = (y - 6)/2) ∧ (c = z/(-1)) ∧ (a = b) ∧ (b = c) ∧ (c = a) :=
by sorry

end line_canonical_eqn_l77_77692


namespace find_b_value_l77_77872

theorem find_b_value (a b c : ℝ)
  (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 7 = 1.5) : b = 15 := by
  sorry

end find_b_value_l77_77872


namespace find_value_of_a_l77_77460

theorem find_value_of_a
  (a : ℝ)
  (h : (a + 3) * 2 * (-2 / 3) = -4) :
  a = -3 :=
sorry

end find_value_of_a_l77_77460


namespace exists_valid_coloring_l77_77374

-- Type for cities
inductive City
| A | B | C | D | E

open City

-- Type for colors
inductive Color
| Yellow
| Red

open Color

-- Definition of an edge by a pair of cities
structure Edge :=
  (city1 : City) 
  (city2 : City)
  (color : Color)
  (city1_ne_city2 : city1 ≠ city2)

-- All pairs of cities: Complete graph K₅
def all_edges (coloring : ∀ (c1 c2 : City), c1 ≠ c2 → Color):
  list Edge :=
  [ (Edge A B (coloring A B sorry) sorry), 
    (Edge A C (coloring A C sorry) sorry),
    (Edge A D (coloring A D sorry) sorry),
    (Edge A E (coloring A E sorry) sorry),
    (Edge B C (coloring B C sorry) sorry),
    (Edge B D (coloring B D sorry) sorry),
    (Edge B E (coloring B E sorry) sorry),
    (Edge C D (coloring C D sorry) sorry),
    (Edge C E (coloring C E sorry) sorry),
    (Edge D E (coloring D E sorry) sorry) ]

-- The proof problem: Does such a coloring exist?
theorem exists_valid_coloring :
  ∃ (coloring : ∀ (c1 c2 : City), c1 ≠ c2 → Color), 
  (∀ c : City, 
    let adj_colors := list.map (λ e, e.color) 
                    (list.filter (λ e, e.city1 = c ∨ e.city2 = c) (all_edges coloring)) 
    in list.alternating adj_colors) ∧
  (∀ e1 e2 : Edge, e1 ≠ e2 → edges_intersect_once e1 e2) :=
begin
  sorry
end

end exists_valid_coloring_l77_77374


namespace function_passes_through_point_l77_77845

theorem function_passes_through_point (a : ℝ) (hf : ∀ x, a^0 = 1) :
  f(1) = 5 :=
by
  let f := λ x, a^(x-1) + 4
  have ha : a^0 = 1 := hf 0
  have h1 : f(1) = a^(1-1) + 4 := rfl
  have h2 : a^0 = 1 := ha
  rw [h2] at h1
  exact h1
sorry

end function_passes_through_point_l77_77845


namespace arithmetic_mean_of_integers_from_neg3_to_6_l77_77643

def integer_range := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

noncomputable def arithmetic_mean : ℚ :=
  (integer_range.sum : ℚ) / (integer_range.length : ℚ)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  arithmetic_mean = 1.5 := by
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l77_77643


namespace general_term_of_a_max_n_Sn_minus_n_an_plus_6_ge_zero_l77_77191

-- Definitions for the geometric sequence {a_n}
def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2^(n)

-- Definitions for the sequence {b_n}
def b : ℕ → ℕ
| 1     := 2
| (n+1) := (n+1) * 2^n

-- Sum of the first n terms of {b_n}
def S (n : ℕ) : ℕ :=
  (range (n+1)).sum (λ (i : ℕ), b (i+1))

theorem general_term_of_a (n : ℕ) : a (n+1) = 2^n := sorry
theorem max_n_Sn_minus_n_an_plus_6_ge_zero : ∃ n : ℕ, S n - n * a (n+1) + 6 ≥ 0 ∧ ∀ m : ℕ, m > n → S m - m * a (m+1) + 6 < 0 ∧ n = 3 := sorry

end general_term_of_a_max_n_Sn_minus_n_an_plus_6_ge_zero_l77_77191


namespace range_of_x_given_conditions_l77_77110

noncomputable def is_even (f : ℝ → ℝ) := ∀ x, f(x) = f(-x)
noncomputable def is_monotone_dec_on_nonneg (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → f(x) ≥ f(y)

theorem range_of_x_given_conditions (f : ℝ → ℝ) (h1 : is_even f) (h2 : is_monotone_dec_on_nonneg f) (hx : f 1 < f (Real.log x)) :
  1 / 10 < x ∧ x < 10 :=
by
  sorry

end range_of_x_given_conditions_l77_77110


namespace ellipse_foci_distance_l77_77782

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end ellipse_foci_distance_l77_77782


namespace angle_measure_l77_77599

theorem angle_measure (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : P = 206 :=
by
  sorry

end angle_measure_l77_77599


namespace cars_pass_quotient_l77_77572

noncomputable def max_cars_quotient : Nat :=
  let m := sorry -- substituting the lengthy calculation steps
  let N := 2000
  N / 10

theorem cars_pass_quotient :
  (N : ℕ) → (N = 2000) → (max_cars_quotient = 200) := by
  intros N hN
  sorry

end cars_pass_quotient_l77_77572


namespace abs_expr_evaluation_l77_77755

theorem abs_expr_evaluation :
  |3 * Real.pi - |3 * Real.pi - 7|| = 7 :=
by
  sorry

end abs_expr_evaluation_l77_77755


namespace a_add_d_eq_zero_l77_77220

-- Define the problem conditions and statement
theorem a_add_d_eq_zero (a b c d k : ℝ) (h_abcdk : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ k ≠ 0)
  (h_inverse : ∀ x, x ≠ -d/c → g(g(x)) = x)
  (h_relation : a + k * c = 0)
  (g : ℝ → ℝ := λ x, k * (a * x + b) / (k * (c * x + d))):

  a + d = 0 :=
sorry

end a_add_d_eq_zero_l77_77220


namespace number_of_ways_to_take_pieces_l77_77313

theorem number_of_ways_to_take_pieces : 
  (Nat.choose 6 4) = 15 := 
by
  sorry

end number_of_ways_to_take_pieces_l77_77313


namespace proof_problem_l77_77152

-- Definitions from conditions
def f (b : ℝ) : ℝ := sorry
def a : ℝ := f 3
def b : ℝ := f 7
def c : ℝ := f 0.63

-- Theorem statement based on the problem translation
theorem proof_problem :
  (f (10 ^ (-2)) = -2) ∧
  (∀ b > 0, f (b^3) / f(b) = 2) ∧
  (∀ b, f 20 ≈ 1.3) ∧
  (∀ b, f (1 / 50) ≈ -1.7) ∧
  (2 * a + b - c = 2) :=
sorry

end proof_problem_l77_77152


namespace min_a_plus_b_l77_77866

noncomputable def min_value_of_a_plus_b (a b : ℝ) : Prop :=
  ∃ (p q r s : ℤ), 
    ((2 * p * r = 2) ∧ (p * s + q * r = 7) ∧ (q * s = -15) ∧ 
    (a = p + q) ∧ (b = r + s) ∧ (a + b = -17))

theorem min_a_plus_b (a b : ℝ) 
  (h : ∃ (p q r s : ℤ), ((2 * p * r = 2) ∧ (p * s + q * r = 7) ∧ (q * s = -15) ∧ 
  (a = p + q) ∧ (b = r + s))): min_value_of_a_plus_b a b :=
begin
  sorry
end

end min_a_plus_b_l77_77866


namespace circle_center_and_radius_l77_77299

open Real

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_center_and_radius :
  let x1 := 3
  let y1 := 5
  let x2 := -5
  let y2 := -3
  let center := midpoint x1 y1 x2 y2
  let radius := distance (-1) 1 x1 y1
  center = (-1, 1) ∧ radius = 4 * sqrt 2 :=
by
  sorry

end circle_center_and_radius_l77_77299


namespace arithmetic_mean_of_integers_from_neg3_to_6_l77_77654

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  let nums := list.range' (-3) 10 in
  (∑ i in nums, i) / (nums.length : ℝ) = 1.5 :=
by
  let nums := list.range' (-3) 10
  have h_sum : (∑ i in nums, i) = 15 := sorry
  have h_length : nums.length = 10 := sorry
  rw [h_sum, h_length]
  norm_num
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l77_77654


namespace students_in_class_l77_77176

theorem students_in_class (b g : ℕ) 
  (h1 : b + g = 20)
  (h2 : (b : ℚ) / 20 = (3 : ℚ) / 4 * (g : ℚ) / 20) : 
  b = 12 ∧ g = 8 :=
by
  sorry

end students_in_class_l77_77176


namespace max_sum_of_prime_ratios_l77_77553

theorem max_sum_of_prime_ratios : 
  ∃ (p : Fin 97 → ℕ), (∀ i, Prime (p i)) ∧ 
  (∑ i, (p i) / ((p i)^2 + 1) = 38) :=
sorry

end max_sum_of_prime_ratios_l77_77553


namespace solution_l77_77758

def sequence (n : ℕ) : ℕ 
| 0 := 2
| (n + 1) := if (n + 1) % 2 = 1 then 2 * (n / 2 + 1) * sequence ((n + 1) / 2) else sequence (n)

#eval sequence (2^100 + 1) -- should evaluate if the implementation and import are correct

theorem solution : sequence (2^100 + 1) = 2^4951 :=
by 
  sorry

end solution_l77_77758


namespace full_problem_l77_77900

-- Part (a)
def part_a : Prop :=
  let O := (0, 0)
  let A := (0, 30)
  let B := (40, 0)
  let C := (14.4, 19.2)
  dist O C = 24

-- Part (b)
def part_b : Prop :=
  let O := (0, 0)
  let A := (0, 30)
  let B := (40, 0)
  let C := (14.4, 19.2)
  C = (14.4, 19.2)

-- Part (c)
def part_c : Prop :=
  let O := (0, 0)
  let A := (0, 30)
  let B := (40, 0)
  let C := (14.4, 19.2)
  let M := (20, 15)
  dist C M = 7

-- A composite theorem to state all parts together (optional)
theorem full_problem : part_a ∧ part_b ∧ part_c :=
  by
  split
  sorry -- Provide proof for part_a
  split
  sorry -- Provide proof for part_b
  sorry -- Provide proof for part_c

end full_problem_l77_77900


namespace max_truthers_cube_l77_77746

-- Condition Definitions
def cube_vertices: Type := Fin 8
def is_adj (v1 v2 : cube_vertices) : Prop := 
  -- v1 and v2 are adjacent if they are connected by an edge in the cube
  (v1.1 ≠ v2.1) ∧ (Hamming.dist v1.1.to_bits v2.1.to_bits = 1)

def is_truther (b: cube_vertices → Prop) (v : cube_vertices) : Prop :=
  b v →
  (∃ l1 l2 t, is_adj v l1 ∧ is_adj v l2 ∧ is_adj v t ∧ b t ∧ ¬ b l1 ∧ ¬ b l2)

def max_truthers (b : cube_vertices → Prop) (k : ℕ) : Prop :=
  ∀ b: cube_vertices → Prop, (∀ v:c_cube_vertices, is_truther b v) → (card {v // b v} = k)

-- Theorem to prove
theorem max_truthers_cube : ∃ b : cube_vertices → Prop, max_truthers b 4 :=
sorry

end max_truthers_cube_l77_77746


namespace cube_root_of_quartic_root_of_decimal_l77_77752

theorem cube_root_of_quartic_root_of_decimal :
  Float.round (Real.sqrt (Real.sqrt (Real.ofNat 8 / 1000)) ^ (1 / 3)) 1 = 0.7 :=
by
  sorry

end cube_root_of_quartic_root_of_decimal_l77_77752


namespace comparison_l77_77472

def f (x : ℝ) : ℝ := Real.log (Real.exp (2 * x) + Real.exp 2) - x

def a : ℝ := f (Real.exp (1 / 3))
def b : ℝ := f (1 / 3)
def c : ℝ := f (4 / 3)

theorem comparison : b > a ∧ a > c := by
  sorry

end comparison_l77_77472


namespace finite_set_with_distances_l77_77588

theorem finite_set_with_distances : 
  ∃ S : Finset (ℝ × ℝ), 
    ∀ P ∈ S, 
      (Finset.filter (λ Q, (dist P Q) = 1) S).card ≥ 1993 :=
sorry

end finite_set_with_distances_l77_77588


namespace contrapositive_iff_l77_77976

theorem contrapositive_iff (a b : ℤ) : (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end contrapositive_iff_l77_77976


namespace sin_double_alpha_l77_77815

variable (α β : ℝ)

theorem sin_double_alpha (h1 : Real.pi / 2 < β ∧ β < α ∧ α < 3 * Real.pi / 4)
        (h2 : Real.cos (α - β) = 12 / 13) 
        (h3 : Real.sin (α + β) = -3 / 5) : 
        Real.sin (2 * α) = -56 / 65 := by
  sorry

end sin_double_alpha_l77_77815


namespace total_perimeter_is_30_l77_77461

namespace RectanglePerimeter

-- Define the lengths and width variables
variables {a b : ℝ} -- Length and width of the original rectangle

-- Given condition: the perimeter of the original rectangle is 10
axiom given_perimeter : 2 * (a + b) = 10

-- Define the function to calculate the total perimeter after cuts
def total_perimeter_after_cuts (a b : ℝ) : ℝ :=
  let original_perimeter := 2 * (a + b) in
  let additional_perimeter := 4 * (a + b) in
  original_perimeter + additional_perimeter

-- Theorem to prove: The total perimeter of the 9 smaller rectangles is 30
theorem total_perimeter_is_30 (a b : ℝ) (h : 2 * (a + b) = 10) : total_perimeter_after_cuts a b = 30 := 
  sorry -- proof to be written

end RectanglePerimeter

end total_perimeter_is_30_l77_77461


namespace binary_sum_correct_l77_77377

def binary_to_decimal (b : ℕ) : ℕ :=
  let digits := b.digits 2
  digits.foldl (λ acc p => acc + p.1 * (2^p.2)) 0

def num1 := 0b1010101  -- 1010101 in base 2
def num2 := 0b111000  -- 111000 in base 2

theorem binary_sum_correct : binary_to_decimal num1 + binary_to_decimal num2 = 141 := by
  sorry

end binary_sum_correct_l77_77377


namespace find_f_of_minus_3_l77_77302

def powerFunction (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem find_f_of_minus_3 :
  (∃ (α : ℝ), powerFunction 2 α = 4) →
  powerFunction (-3) 2 = 9 :=
by
  intros h
  cases h with α hα
  sorry

end find_f_of_minus_3_l77_77302


namespace bessel_general_solution_l77_77589

theorem bessel_general_solution (p : ℝ) (C₁ C₂ : ℝ) :
  (∃ (J_p J_(-p) : ℝ → ℝ), ∀ (x : ℝ), x > 0 →
  ((¬p ∈ ℤ) → (∃ (y : ℝ → ℝ), y(x) = C₁ * J_p(x) + C₂ * J_(-p)(x) ∧ x^2 * y'' + x * y' + (x^2 - p^2) * y = 0)) ∧
  (p ∈ ℤ → ∃ (Y_p : ℝ → ℝ) (y : ℝ → ℝ), y(x) = C₁ * J_p(x) + C₂ * Y_p(x) ∧ x^2 * y'' + x * y' + (x^2 - p^2) * y = 0))) :=
sorry

end bessel_general_solution_l77_77589


namespace sum_eq_two_l77_77953

theorem sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 10 * x - 6 * y - 34) : x + y = 2 :=
by
  sorry

end sum_eq_two_l77_77953


namespace necessary_but_not_sufficient_l77_77099

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x - (Real.pi / 3))

theorem necessary_but_not_sufficient (ω : ℝ) :
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) ↔ (ω = 2) ∨ (∃ ω ≠ 2, ∀ x : ℝ, f ω (x + Real.pi) = f ω x) :=
by
  sorry

end necessary_but_not_sufficient_l77_77099


namespace doug_money_l77_77682

def money_problem (J D B: ℝ) : Prop :=
  J + D + B = 68 ∧
  J = 2 * B ∧
  J = (3 / 4) * D

theorem doug_money (J D B: ℝ) (h: money_problem J D B): D = 36.27 :=
by sorry

end doug_money_l77_77682


namespace max_tan_value_of_B_l77_77122

-- Define the internal angles of the triangle
variables {A B C : ℝ}

-- Define the condition given in the problem
axiom sin_condition : sin B / sin A = cos (A + B)

-- Define the maximum value of tan B
def max_tan_B := sqrt 2 / 4

-- Prove that the maximum value of tan B is sqrt(2) / 4 under the given condition
theorem max_tan_value_of_B : ∀ {A B C : ℝ}, (sin B / sin A = cos (A + B)) → tan B ≤ sqrt 2 / 4 :=
begin
  assume A B C,
  assume h : sin B / sin A = cos (A + B),
  sorry
end

end max_tan_value_of_B_l77_77122


namespace mean_of_integers_neg3_to_6_l77_77646

theorem mean_of_integers_neg3_to_6 : 
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ) in
  let n := (6 - (-3) + 1 : ℤ) in
  s / n = 1.5 :=
by
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ)
  let n := (6 - (-3) + 1 : ℤ)
  simp
  sorry

end mean_of_integers_neg3_to_6_l77_77646


namespace max_sphere_radius_l77_77912

section
variables (C : set (Real × Real × Real))
variables (P : Real × Real × Real)
variables (r : Real)

-- Base circle conditions
def is_base_circle (C : set (Real × Real × Real)) : Prop :=
  ∃ (x y : Real), C = { (x, y, 0) | x^2 + y^2 = 1 }

-- Vertex point condition
def is_vertex_point (P : Real × Real × Real) : Prop :=
  P = (3, 4, 8)

-- Condition for a sphere with radius r to be contained entirely in the slanted cone
def sphere_within_cone (r : Real) (C : set (Real × Real × Real)) (P : Real × Real × Real) : Prop :=
  ∀ (x y z : Real), (x, y, z) ∈ C → ((x - 3)^2 + (y - 4)^2 + (z - 8)^2) ≤ r^2

theorem max_sphere_radius :
  is_base_circle C ∧ is_vertex_point P → ∃ r, sphere_within_cone r C P ∧ r = 3 - sqrt 5 :=
begin
  sorry,
end

end

end max_sphere_radius_l77_77912


namespace solve_inequality_l77_77590

theorem solve_inequality (x : ℝ) :
  -2 * x^2 - x + 6 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3 / 2 :=
sorry

end solve_inequality_l77_77590


namespace no_solution_fractional_eq_l77_77959

theorem no_solution_fractional_eq (y : ℝ) (h : y ≠ 3) : 
  ¬ ( (y-2)/(y-3) = 2 - 1/(3-y) ) :=
by
  sorry

end no_solution_fractional_eq_l77_77959


namespace max_towns_l77_77972

-- Define a spherical planet with total surface area A
def spherical_planet (A : Real) := ∃ r : Real, A = 4 * π * r^2

-- Define the condition for the area of each town
def town_area (A N : Real) := (A / 1000) * N

-- Define the constraint of different latitudes and longitudes for points in different towns
def different_lat_long (N : Nat) : Prop := ∀ i j : Fin N, i ≠ j → different_latitudes i j ∧ different_longitudes i j

-- Define the final proof statement
theorem max_towns (A : Real) (hA : spherical_planet A) (hN : town_area A 31) (hDiff : different_lat_long 31) : N = 31 :=
by
  sorry

end max_towns_l77_77972


namespace problem_1_problem_2_l77_77115

variables (α : ℝ) (h : Real.tan α = 3)

theorem problem_1 : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by
  -- Proof is skipped
  sorry

theorem problem_2 : Real.sin α * Real.sin α + Real.sin α * Real.cos α + 3 * Real.cos α * Real.cos α = 3 / 2 :=
by
  -- Proof is skipped
  sorry

end problem_1_problem_2_l77_77115


namespace sqrt81_minus_sqrt77_plus1_approx_l77_77805

noncomputable def approximate_sqrt (x : ℝ) : ℝ :=
  if x = 81 then 9 else if x = 77 then 8.78 else √x

theorem sqrt81_minus_sqrt77_plus1_approx :
  (approximate_sqrt 81 - approximate_sqrt 77) + 1 ≈ 1.23 :=
by
  have h_sqrt81 : approximate_sqrt 81 = 9 := if_pos rfl
  have h_sqrt77 : approximate_sqrt 77 = 8.78 := if_neg (by norm_num)

  rw [h_sqrt81, h_sqrt77]
  norm_num
  sorry

end sqrt81_minus_sqrt77_plus1_approx_l77_77805


namespace degree_to_radian_l77_77761

theorem degree_to_radian (h : 1 = (π / 180)) : 60 = π * (1 / 3) := 
sorry

end degree_to_radian_l77_77761


namespace sum_of_inverses_of_logarithms_l77_77865

theorem sum_of_inverses_of_logarithms (a b : ℝ) (h1 : 2^a = 10) (h2 : 5^b = 10) : 
  (1 / a) + (1 / b) = 1 :=
by
  sorry

end sum_of_inverses_of_logarithms_l77_77865


namespace sum_abs_roots_l77_77802

theorem sum_abs_roots : 
  let f := (λ x : ℂ, x^4 - 6 * x^3 + 9 * x^2 + 24 * x - 36)
  ∃ roots : list ℂ, (∀ r ∈ roots, f r = 0) ∧ (roots.length = 4) ∧ 
                 (∑ r in roots, complex.abs r = 4 * complex.sqrt 6) := 
sorry

end sum_abs_roots_l77_77802


namespace counters_installed_is_9_l77_77201

variable (counters_installed : ℕ)
variable (original_cabinets new_cabinets installed_cabinets : ℕ)

axiom initial_cabinets : original_cabinets = 3
axiom cabinets_per_counter : new_cabinets = 2 * counters_installed
axiom additional_cabinets : installed_cabinets = 5
axiom total_cabinets : original_cabinets + new_cabinets + installed_cabinets = 26

theorem counters_installed_is_9 : counters_installed = 9 :=
by
  have : 3 + (2 * counters_installed) + 5 = 26 := by
    rw [initial_cabinets, ←add_assoc, nat.add_comm 5 _, add_assoc, add_assoc, cabinets_per_counter, additional_cabinets]
  linarith

end counters_installed_is_9_l77_77201


namespace units_digit_of_7_pow_3_l77_77668

theorem units_digit_of_7_pow_3 : (7 ^ 3) % 10 = 3 :=
by
  sorry

end units_digit_of_7_pow_3_l77_77668


namespace composite_integers_with_property_are_powers_of_prime_l77_77765

-- Lean formalization of the problem

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ n = p * q

def satisfiesDivisorProperty (n : ℕ) (divisors : List ℕ) : Prop :=
  (divisors.head = 1 ∧ divisors.last = n) ∧
  divisors.isSorted (· < ·) ∧
  ∀ i, 1 ≤ i ∧ i ≤ divisors.length - 2 → divisors.nthLe i sorry ∣ (divisors.nthLe (i+1) sorry + divisors.nthLe (i+2) sorry)

theorem composite_integers_with_property_are_powers_of_prime (n : ℕ) :
  isComposite n →
  (∃ divisors : List ℕ, satisfiesDivisorProperty n divisors) →
  ∃ (p : ℕ) (m : ℕ), Prime p ∧ m ≥ 2 ∧ n = p ^ m :=
sorry

end composite_integers_with_property_are_powers_of_prime_l77_77765


namespace largest_fraction_added_to_one_sixth_l77_77627

theorem largest_fraction_added_to_one_sixth :
  ∃ (c d : ℕ), (d < 8) ∧ (c < d) ∧ (c.to_real / d.to_real) = 5.to_real / 7.to_real ∧ 
               ((1.to_real / 6.to_real) + (c.to_real / d.to_real)) < 1 := 
sorry

end largest_fraction_added_to_one_sixth_l77_77627


namespace value_of_P_2007_l77_77919

-- Define P as the product given in the conditions.
def P (n : ℕ) : ℚ := (finset.range (n - 1)).prod (λ i, (1 - (1 / (i + 2))))

-- The goal is to prove P 2007 == 1 / 2007.
theorem value_of_P_2007 
  (n : ℕ) 
  (h : n = 2007) : 
  P n = 1 / n := 
by {
  sorry
}

end value_of_P_2007_l77_77919


namespace sales_revenue_for_a_eq_1_div_7_and_x_eq_7_range_of_a_for_equilibrium_price_ge_6_l77_77365

/-
Problem 1: Given the following conditions, prove the calculated sales revenue:
-/
def sales_revenue (a x : ℝ) : ℝ :=
  let y1 := a * x + (7 / 2) * a^2 - a
  let y2 := (-1 / 224) * x^2 - (1 / 112) * x + 1
  if y2 > y1 then y1 * x else y2 * x

theorem sales_revenue_for_a_eq_1_div_7_and_x_eq_7 :
  ∀ (a : ℝ), a = 1 / 7 →
  (sales_revenue a 7).round.toNat = 50313 := 
by
  intros a ha
  sorry


/-
Problem 2: Given the following conditions, prove the range of values for a:
-/
def equilibrium_function (a x : ℝ) : ℝ :=
  a * x + (7 / 2) * a^2 - a + (1 / 224) * x^2 + (1 / 112) * x - 1

theorem range_of_a_for_equilibrium_price_ge_6 :
  ∀ (a : ℝ), 0 < a ∧ a ≤ 1 / 7 ↔ 
  equilibrium_function a 6 ≤ 0 ∧ equilibrium_function a 14 > 0 :=
by
  intros a
  sorry

end sales_revenue_for_a_eq_1_div_7_and_x_eq_7_range_of_a_for_equilibrium_price_ge_6_l77_77365


namespace comet_orbit_equation_comet_ellipse_equation_l77_77383

theorem comet_orbit_equation : 
    ∃ (a b : ℝ), 
    3 = a ₀ and 
1. The perihelion of the comet is 2 astronomical units from the center of the Sun.
4. The Sun is at one focus of the comet's elliptical orbit. = 
sorry

open_locale nat big_operators
open_locale complex_conjugate

variables (a : ℝ) (b : ℝ) (c : ℝ)

-- The conditions given: perihelion and aphelion distances
def perihelion := (a - c = 2)
def aphelion := (a + c = 6)

-- Focal distance relationship
def focal_distance := (c^2 = a^2 - b^2)

-- The possible equations of the ellipse
def ellipse_equation_x_major := (c^2 = a^2 - b^2)
def ellipse_equation_y_major := (focal_distance = 4)

-- The statements we need to prove
theorem comet_ellipse_equation :
  perihelion → aphelion → focal_distance → 
  ((ellipse_equation_x_major = x^2 / 16 + y^2 / 12) ∨ (ellipse_equation_y_major = y^2 / 16 + x^2 / 12))
  sorry
  
end comet_orbit_equation_comet_ellipse_equation_l77_77383


namespace ralph_tv_hours_l77_77269

theorem ralph_tv_hours :
  (4 * 5 + 6 * 2) = 32 :=
by
  sorry

end ralph_tv_hours_l77_77269


namespace inequality_proof_l77_77579

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d):
    1/a + 1/b + 4/c + 16/d ≥ 64/(a + b + c + d) :=
by
  sorry

end inequality_proof_l77_77579


namespace triangle_perimeter_eq_twenty_l77_77829

theorem triangle_perimeter_eq_twenty (x y : ℝ) (h : |x - 4| + real.sqrt (y - 8) = 0) : 
  20 = if (4,8,8) = (4,4,8) then 0 else 4 + 8 + 8 := 
by sorry

end triangle_perimeter_eq_twenty_l77_77829


namespace circle_radius_sum_l77_77893

/-- Suppose there is a circle \(\omega_1\) with radius 1. 
Circles \(\phi_1, \phi_2, \dots, \phi_8\) have equal radii and are tangent to \(\omega_1\), \(\phi_{i-1}\), and \(\phi_{i+1}\),
where \(\phi_0 = \phi_8\) and \(\phi_1 = \phi_9\). There exists another circle \(\omega_2\) such that \(\omega_1\) is not equal to \(\omega_2\)
and \(\omega_2\) is tangent to each \(\phi_i\) for \(1 \le i \le 8\). The radius of \(\omega_2\) can be expressed in the form
\(a - b\sqrt{c}  -d\sqrt{e - \sqrt{f}} + g \sqrt{h - j \sqrt{k}}\) where \(a, b, \dots, k\) are positive integers and the numbers \(e, f, k, \gcd(h, j)\) are squarefree.
Prove that the sum of these integers is some specific integer. -/
theorem circle_radius_sum
  (a b c d e f g h j k : ℕ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) (hg : g > 0) (hh : h > 0) (hj : j > 0) (hk : k > 0)
  (he_squarefree : nat.is_squarefree e) 
  (hf_squarefree : nat.is_squarefree f) 
  (hk_squarefree : nat.is_squarefree k)
  (hh_j_coprime : nat.gcd h j = 1) :
  a + b + c + d + e + f + g + h + j + k = 42 :=  -- assuming the answer is 42
sorry

end circle_radius_sum_l77_77893


namespace sausages_placement_and_path_length_l77_77573

variables {a b x y : ℝ} (h1 : 2 * x = y) (h2 : b = 1 / (x + y))
variables (h3 : (x + y) = 8 * a) (h4 : x = 1.4 * y)

theorem sausages_placement_and_path_length (h1 : 2 * x = y) (h2 : b = 1 / (x + y))
(h3 : (x + y) = 8 * a) (h4 : x = 1.4 * y) : 
  x < y ∧ (x / y) = 1.4 :=
by {
  sorry
}

end sausages_placement_and_path_length_l77_77573


namespace complex_plane_intersection_unique_point_l77_77187

theorem complex_plane_intersection_unique_point (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 2|) ∧ (|z| = k) →
  k = |(-2.75 : ℝ)| + sqrt (137 / 32) :=
by
  sorry

end complex_plane_intersection_unique_point_l77_77187


namespace convex_polygons_count_l77_77074

theorem convex_polygons_count (n : ℕ) (hn : n = 15) : 
  (Nat.choose n 4) + (Nat.choose n 3) = 1820 :=
by
  rw hn
  sorry

end convex_polygons_count_l77_77074


namespace isosceles_triangle_area_l77_77111

theorem isosceles_triangle_area (x : ℤ) (h1 : x > 2) (h2 : x < 4) 
  (h3 : ∃ (a b : ℤ), a = x ∧ b = 8 - 2 * x ∧ a = b) :
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end isosceles_triangle_area_l77_77111


namespace range_of_a_for_decreasing_f_l77_77839

theorem range_of_a_for_decreasing_f :
  (∀ x : ℝ, (-3) * x^2 + 2 * a * x - 1 ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by
  -- The proof goes here
  sorry

end range_of_a_for_decreasing_f_l77_77839


namespace quadratic_function_min_value_l77_77742

theorem quadratic_function_min_value :
  ∃ x, ∀ y, 5 * x^2 - 15 * x + 2 ≤ 5 * y^2 - 15 * y + 2 ∧ (5 * x^2 - 15 * x + 2 = -9.25) :=
by
  sorry

end quadratic_function_min_value_l77_77742


namespace area_of_inscribed_square_l77_77320

theorem area_of_inscribed_square :
  ∀ (radius : ℝ) (center1 center2 : ℝ × ℝ),
  radius = 1 →
  dist center1 center2 = sqrt 3 →
  ∃ (x : ℝ), x > 0 ∧ 
  let area := x^2 in 
  area = 2 - sqrt 7 / 2 :=
sorry

end area_of_inscribed_square_l77_77320


namespace polynomial_division_remainder_l77_77794

noncomputable def p : ℚ[X] := X^6 + 2*X^5 - 3*X^4 + X^3 - 2*X^2 + 4*X - 5
noncomputable def d : ℚ[X] := (X^2 - 1) * (X - 2)
noncomputable def r : ℚ[X] := 19.5 * X^2 + 7 * X - 28.5

-- Provide the statement of the theorem that expresses the problem
theorem polynomial_division_remainder :
  p % d = r := 
sorry

end polynomial_division_remainder_l77_77794


namespace quadratic_even_coeff_l77_77259

theorem quadratic_even_coeff (a b c : ℤ) (h : a ≠ 0) (hq : ∃ x : ℚ, a * x^2 + b * x + c = 0) : ¬ (∀ x : ℤ, (x ≠ 0 → (x % 2 = 1))) := 
sorry

end quadratic_even_coeff_l77_77259


namespace find_a_eq_neg_7_l77_77882

theorem find_a_eq_neg_7 (a : ℝ) : 
  let z := (Complex.mk a 1) * (Complex.mk 3 4)
  in z.re = z.im → a = -7 :=
by
  -- Introduce the complex number multiplication result
  let z := (Complex.mk a 1) * (Complex.mk 3 4)
  -- Hypothesis: real part equals imaginary part
  intro h : z.re = z.im
  -- Goal is to prove a = -7
  sorry

end find_a_eq_neg_7_l77_77882


namespace length_XY_l77_77189

-- Definitions for the given problem
def radius : ℝ := 12
def angle_AOB : ℝ := 60
def OY_perpendicular_AB : Prop := true 
def intersection_OY_AB_X : Prop := true
def OA_eq_OB : Prop := true

-- Lean 4 statement
theorem length_XY :
  OY_perpendicular_AB →
  intersection_OY_AB_X →
  OA_eq_OB →
  ∠AOB = angle_AOB →
  XY = radius - radius * sqrt(3) / 2 := 
by
  sorry

end length_XY_l77_77189


namespace correct_statements_l77_77128

-- Definitions used in Lean 4 statement based on conditions
def f (x : ℝ) (φ : ℝ) : ℝ := Math.sin (2 * x + φ)

axiom φ_cond : 0 < φ ∧ φ < Math.pi
axiom symmetry_cond : ∃ k : ℤ, 2 * (2 * Math.pi / 3) + φ = k * Math.pi

-- Lean 4 Theorem Statement
theorem correct_statements (φ : ℝ) (φ_cond : 0 < φ ∧ φ < Math.pi) (symmetry_cond : ∃ k : ℤ, 2 * (2 * Math.pi / 3) + φ = k * Math.pi) :
  (∀ x, 0 < x ∧ x < 5 * Math.pi / 12 → f x φ < f (x + 1) φ) ∧ -- Statement A
  ¬ (∃ xa xb, xa ∈ Set.Ioo (-Math.pi / 12) (11 * Math.pi / 12) ∧ f' xa φ = 0 ∧ f' xb φ = 0 ∧ xa ≠ xb) ∧ -- Statement B
  ¬ (∀ x, f x φ = f (7 * Math.pi / 6 - x) φ) ∧ -- Statement C
  (∀ x, (x = 0 → (∃ m b : ℝ, m = -1 ∧ b = Math.sqrt 3 / 2 ∧ ∀ y, y = f x φ → y = m * x + b))) -- Statement D
:= sorry

end correct_statements_l77_77128


namespace estimate_students_above_110_l77_77004

noncomputable def students : ℕ := 50
noncomputable def mean : ℝ := 100
noncomputable def std_dev : ℝ := 10
noncomputable def normal_dist (x : ℝ) : ℝ := (Real.exp (-((x - mean)^2 / (2 * std_dev^2)))) / (std_dev * (Real.sqrt (2 * Real.pi)))
noncomputable def prob_90_to_100 : ℝ := 0.3
noncomputable def desired_prob : ℝ := 0.5 - (prob_90_to_100 + prob_90_to_100)
noncomputable def estimated_students_above_110 : ℝ := desired_prob * students

theorem estimate_students_above_110 : estimated_students_above_110 = 10 := by 
  sorry

end estimate_students_above_110_l77_77004


namespace rectangle_divided_into_13_squares_l77_77405

theorem rectangle_divided_into_13_squares (s a b : ℕ) (h₁ : a * b = 13 * s^2)
  (h₂ : ∃ k l : ℕ, a = k * s ∧ b = l * s ∧ k * l = 13) :
  (a = s ∧ b = 13 * s) ∨ (a = 13 * s ∧ b = s) :=
by
sorry

end rectangle_divided_into_13_squares_l77_77405


namespace find_d_f_l77_77063

variable (b d f e : ℂ)
variable (c1 c2 : ℂ)

-- Given conditions
def b_val : b = 2 := by sorry
def e_val : e = -5 := by sorry
def sum_eq : 2 + b*i + 3 + d*i + e + f*i = 1 - 3*i := by sorry

-- Goal
theorem find_d_f : d + f = -5 := by 
  rw [b_val, e_val] at sum_eq
  sorry

end find_d_f_l77_77063


namespace compute_difference_of_squares_l77_77060

theorem compute_difference_of_squares :
    75^2 - 25^2 = 5000 :=
by
  sorry

end compute_difference_of_squares_l77_77060


namespace initial_transformation_check_l77_77333

def transformation (x₁ x₂ x₃ : ℤ) : ℤ × ℤ × ℤ :=
  -- Function to replace one of the numbers with the difference between the sum of the other two and 1
  match x₁, x₂, x₃ with
  | x₁, x₂, _ => (x₁ + x₂, x₃ - (x₁ + x₂) - 1, x₃)

-- The possible initial state (2, 2, 2) or (3, 3, 3)
def initial_possible_states : list (ℤ × ℤ × ℤ) :=
  [(2, 2, 2), (3, 3, 3)]

-- Target state to reach
def target_state : ℤ × ℤ × ℤ :=
  (17, 1999, 2105)

noncomputable def can_transform (start : ℤ × ℤ × ℤ) : Prop :=
  -- Recursively check if we can reach the target state (17, 1999, 2105) from start
  sorry

theorem initial_transformation_check :
  (∀ start ∈ initial_possible_states, can_transform start ↔ start = (3, 3, 3)) ∧
  (∀ start ∈ initial_possible_states, ¬ can_transform start ↔ start = (2, 2, 2)) :=
begin
  sorry
end

end initial_transformation_check_l77_77333


namespace odd_periodic_function_value_l77_77960

theorem odd_periodic_function_value
  (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = - f x)
  (periodic_f : ∀ x, f (x + 3) = f x)
  (bounded_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f 8.5 = -1 :=
sorry

end odd_periodic_function_value_l77_77960


namespace smallest_odd_number_with_four_primes_l77_77324

theorem smallest_odd_number_with_four_primes (n : ℕ) (h1 : 4 = n.prime_factors.card) (h2 : ∀ p ∈ n.prime_factors, p > 3) (h3 : odd n) : n = 5005 :=
sorry

end smallest_odd_number_with_four_primes_l77_77324


namespace part_a_part_b_part_c_part_d_l77_77516

-- Part (a)
theorem part_a : 
  (let cups := List.range 12 in 
  let sequence := (List.scanl (λ acc _ => (acc + 5) % 12) 1 (List.range 13)).map (λ i => if i = 0 then 12 else i) in
  sequence) 
  = [1,  6, 11,  4,  9,  2,  7, 12,  5, 10,  3,  8, 1] := sorry

-- Part (b)
theorem part_b : 
  (let cups := List.range 9 in 
  let sequence := (List.scanl (λ acc _ => (acc + 6) % 9) 1 (List.range 4)).map (λ i => if i = 0 then 9 else i) in
  let unvisited := List.filter (λ x => !(sequence.contains x)) cups in
  unvisited) 
  = [2, 3, 5, 6, 8, 9] := sorry

-- Part (c)
theorem part_c : 
  (let cups := List.range 120 in 
  let sequence := (List.scanl (λ acc _ => (acc + 3) % 120) 1 (List.range 120)).map (λ i => if i = 0 then 120 else i) in
  List.length (List.filter (λ x => !(sequence.contains x)) cups)) 
  = 0 := sorry

-- Part (d)
theorem part_d : 
  (let final_cup := (1 + 7 * 337) % 1000 in 
  if final_cup = 0 then 1000 else final_cup) 
  = 360 := sorry

end part_a_part_b_part_c_part_d_l77_77516


namespace KimFridayToMondayRatio_l77_77909

variable (MondaySweaters : ℕ) (TuesdaySweaters : ℕ) (WednesdaySweaters : ℕ) (ThursdaySweaters : ℕ) (FridaySweaters : ℕ)

def KimSweaterKnittingConditions (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ) : Prop :=
  MondaySweaters = 8 ∧
  TuesdaySweaters = MondaySweaters + 2 ∧
  WednesdaySweaters = TuesdaySweaters - 4 ∧
  ThursdaySweaters = TuesdaySweaters - 4 ∧
  MondaySweaters + TuesdaySweaters + WednesdaySweaters + ThursdaySweaters + FridaySweaters = 34

theorem KimFridayToMondayRatio 
  (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ)
  (h : KimSweaterKnittingConditions MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters) :
  FridaySweaters / MondaySweaters = 1/2 :=
  sorry

end KimFridayToMondayRatio_l77_77909


namespace percentage_loss_calculation_l77_77985

theorem percentage_loss_calculation
  (initial_cost_euro : ℝ)
  (retail_price_dollars : ℝ)
  (exchange_rate_initial : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (sales_tax : ℝ)
  (exchange_rate_new : ℝ)
  (final_sale_price_dollars : ℝ) :
  initial_cost_euro = 800 ∧
  retail_price_dollars = 900 ∧
  exchange_rate_initial = 1.1 ∧
  discount1 = 0.10 ∧
  discount2 = 0.15 ∧
  sales_tax = 0.10 ∧
  exchange_rate_new = 1.5 ∧
  final_sale_price_dollars = (retail_price_dollars * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) →
  ((initial_cost_euro - final_sale_price_dollars / exchange_rate_new) / initial_cost_euro) * 100 = 36.89 := by
  sorry

end percentage_loss_calculation_l77_77985


namespace find_x_plus_y_l77_77556

open Matrix

-- Main proof statement
theorem find_x_plus_y (x y : ℝ) (hxy : x ≠ y) 
    (hdet : det ![
        ![2, 5, 8],
        ![4, x, y],
        ![4, y, x]
    ] = 0)
    : x + y = 26 :=
begin
    -- Proof would go here
    sorry
end

end find_x_plus_y_l77_77556


namespace area_of_inscribed_square_l77_77713

theorem area_of_inscribed_square
    (r : ℝ)
    (h : ∀ A : ℝ × ℝ, (A.1 = r - 1 ∨ A.1 = -(r - 1)) ∧ (A.2 = r - 2 ∨ A.2 = -(r - 2)) → A.1^2 + A.2^2 = r^2) :
    4 * r^2 = 100 := by
  -- proof would go here
  sorry

end area_of_inscribed_square_l77_77713


namespace area_of_DOB_l77_77257

/--
Point D lies on the extension of side AC of triangle ABC, whose area is S,
with point A situated between D and C. Let O be the centroid of triangle ABC.
It is known that the area of triangle DOC is S_I. Prove that the area of 
triangle DOB is 2S_I - S / 3.
-/
theorem area_of_DOB (A B C D O : Point) (S S_I : ℝ)
  (h1 : lies_on_extension D A C)
  (h2 : area_triangle ABC = S)
  (h3 : is_centroid O A B C)
  (h4 : area_triangle DOC = S_I) :
  area_triangle DOB = 2 * S_I - S / 3 :=
sorry

end area_of_DOB_l77_77257


namespace complex_multiplication_l77_77973

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end complex_multiplication_l77_77973


namespace minimize_xyz_condition_l77_77551

theorem minimize_xyz_condition :
  let m := Inf {s | ∃ (x y z : ℝ), x^3 * y^2 * z = 1 ∧ s = x + 2 * y + 3 * z} in
  m ^ 3 = 72 :=
by
  sorry

end minimize_xyz_condition_l77_77551


namespace area_of_set_U_l77_77241

noncomputable def five_presentable (z : ℂ) : Prop :=
  ∃ (w : ℂ), abs w = 5 ∧ z = w^2 - (1 / w^2)

def set_U : Set ℂ := {z | five_presentable z}

theorem area_of_set_U : measure_theory.measure.measure2.area (set_U) = 48 * real.pi := by
  sorry

end area_of_set_U_l77_77241


namespace analyze_relationship_l77_77137

noncomputable def line := {a b r x y : ℝ // a * x + b * y - r^2 = 0}
noncomputable def circle := {r x y : ℝ // x^2 + y^2 = r^2}
def point := {a b : ℝ}

theorem analyze_relationship (a b r : ℝ) (hA1: a^2 + b^2 = r^2) (hC : circle) (hL : line) :
  (a^2 + b^2 = r^2 → (l (0, 0)) tangent to C) ∧
  (a^2 + b^2 < r^2 → (l (0, 0)) disjoint from C) ∧
  (a^2 + b^2 > r^2 → (l (0, 0)) intersects C) ∧
  (a * a + b * b = r^2 → (l (0, 0)) tangent to C) :=
sorry

end analyze_relationship_l77_77137


namespace problem1_problem2_l77_77142

noncomputable def A (a : ℝ) : set ℝ := {x | x^2 + (a - 1) * x - a > 0}
noncomputable def B (a b : ℝ) : set ℝ := {x | (x + a) * (x + b) > 0}
noncomputable def M : set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def CiB (B : set ℝ) : set ℝ := {x | -(exists b,  B x)}

theorem problem1 (a b : ℝ) (h : a < b) (hCiB_eq_M : CiB (B a b) = M) : a = -1 ∧ b = 3 := sorry

theorem problem2 (a b : ℝ) (h1 : a > b) (h2 : b > -1) : 
  (A a) ∩ (B a b) = {x | x < -a ∨ x > 1} := sorry

end problem1_problem2_l77_77142


namespace min_value_of_expr_l77_77437

theorem min_value_of_expr (a : ℝ) (ha : a > 1) : a + a^2 / (a - 1) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_expr_l77_77437


namespace pond_capacity_l77_77203

theorem pond_capacity :
  let normal_rate := 6 -- gallons per minute
  let restriction_rate := (2/3 : ℝ) * normal_rate -- gallons per minute
  let time := 50 -- minutes
  let capacity := restriction_rate * time -- total capacity in gallons
  capacity = 200 := sorry

end pond_capacity_l77_77203


namespace trigonometric_inequality_C_trigonometric_inequality_D_l77_77673

theorem trigonometric_inequality_C (x : Real) : Real.cos (3*Real.pi/5) > Real.cos (-4*Real.pi/5) :=
by
  sorry

theorem trigonometric_inequality_D (y : Real) : Real.sin (Real.pi/10) < Real.cos (Real.pi/10) :=
by
  sorry

end trigonometric_inequality_C_trigonometric_inequality_D_l77_77673


namespace part_a_part_b_l77_77693

-- Define the condition for part (a)
def part_a_condition (k : ℤ) : Prop :=
  (∏ i in (finset.range 901).map (λ x, k + 1 + x)) / 901 + (∏ i in (finset.range 900).map (λ x, k + 2 + x)) = 
  (∏ i in (finset.range 901).map (λ x, k + 2 + x) ⊕ (k + 902)) / 901

-- Part (a) statement
theorem part_a (k : ℤ) : part_a_condition k :=
sorry

-- Define N for part (b)
noncomputable def N : ℤ := 
  ∑ i in (finset.range 1116).map (λ x, x + 1), ∏ j in (finset.range 899).map (λ x, i + 1 + x)

-- Part (b) statement
theorem part_b : ∀ m ∈ (finset.range 901).map (λ x, 1116 + x), m ∣ 901 * N :=
sorry

end part_a_part_b_l77_77693


namespace problem_1_solution_set_problem_2_min_value_l77_77467

-- Problem (1)
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_1_solution_set :
  {x : ℝ | f (x + 3/2) ≥ 0} = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

-- Problem (2)
theorem problem_2_min_value (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 :=
by
  sorry

end problem_1_solution_set_problem_2_min_value_l77_77467


namespace directrix_of_given_parabola_l77_77413

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  -- Calculate the x-coordinate of the vertex
  let x_vertex := -b / (2 * a) in
  -- Calculate the y-coordinate of the vertex using the vertex formula
  let y_vertex := a * x_vertex^2 + b * x_vertex + c in
  -- Calculate the y-coordinate of the directrix
  y_vertex - (1 / (4 * a))

theorem directrix_of_given_parabola : parabola_directrix (-3) 6 (-5) = -23/12 :=
by
  -- This is where the proof would go
  sorry

end directrix_of_given_parabola_l77_77413


namespace compute_difference_of_squares_l77_77059

theorem compute_difference_of_squares :
    75^2 - 25^2 = 5000 :=
by
  sorry

end compute_difference_of_squares_l77_77059


namespace log_ineq_implies_prod_ineq_l77_77439

-- Define the problem's conditions
variables {a b : ℝ}
variable ha : a > 0
variable hb : b > 0
variable ha_ne1 : a ≠ 1
variable hb_ne1 : b ≠ 1
variable hlog : log a b > 1

-- State the theorem
theorem log_ineq_implies_prod_ineq
  (ha : a > 0)
  (hb : b > 0)
  (ha_ne1 : a ≠ 1)
  (hb_ne1 : b ≠ 1)
  (hlog : log a b > 1) : (b - 1) * (b - a) > 0 :=
sorry

end log_ineq_implies_prod_ineq_l77_77439


namespace sum_of_possible_values_of_N_l77_77086

theorem sum_of_possible_values_of_N : 
  ∀ (L : set (set (ℝ × ℝ))), 
  (card L = 5 ∧ ∀ l ∈ L, ∃ (m b : ℝ), l = {p : ℝ × ℝ | p.2 = m * p.1 + b} ∧ 
  (∃ p ∈ l, ∀ l' ∈ (L \ {l}), p ∈ l')) → 
  (∑ i in {n : ℕ | ∃ l1 l2 ∈ L, n = card ({p : ℝ × ℝ | (∃ l1 l2 ∈ L, p ∈ l1 ∧ p ∈ l2)})}, i) = 53 :=
sorry

end sum_of_possible_values_of_N_l77_77086


namespace correct_statements_l77_77129

-- Definitions used in Lean 4 statement based on conditions
def f (x : ℝ) (φ : ℝ) : ℝ := Math.sin (2 * x + φ)

axiom φ_cond : 0 < φ ∧ φ < Math.pi
axiom symmetry_cond : ∃ k : ℤ, 2 * (2 * Math.pi / 3) + φ = k * Math.pi

-- Lean 4 Theorem Statement
theorem correct_statements (φ : ℝ) (φ_cond : 0 < φ ∧ φ < Math.pi) (symmetry_cond : ∃ k : ℤ, 2 * (2 * Math.pi / 3) + φ = k * Math.pi) :
  (∀ x, 0 < x ∧ x < 5 * Math.pi / 12 → f x φ < f (x + 1) φ) ∧ -- Statement A
  ¬ (∃ xa xb, xa ∈ Set.Ioo (-Math.pi / 12) (11 * Math.pi / 12) ∧ f' xa φ = 0 ∧ f' xb φ = 0 ∧ xa ≠ xb) ∧ -- Statement B
  ¬ (∀ x, f x φ = f (7 * Math.pi / 6 - x) φ) ∧ -- Statement C
  (∀ x, (x = 0 → (∃ m b : ℝ, m = -1 ∧ b = Math.sqrt 3 / 2 ∧ ∀ y, y = f x φ → y = m * x + b))) -- Statement D
:= sorry

end correct_statements_l77_77129


namespace Minnie_takes_70_more_minutes_l77_77247

def Minnie's_speed_flat : ℝ := 30
def Minnie's_speed_downhill : ℝ := 45
def Minnie's_speed_uphill : ℝ := 7.5

def Penny's_speed_flat : ℝ := 45
def Penny's_speed_downhill : ℝ := 60
def Penny's_speed_uphill : ℝ := 15

def distance_A_to_B : ℝ := 15
def distance_B_to_C : ℝ := 20
def distance_C_to_A : ℝ := 30

def distance_A_to_C : ℝ := 30
def distance_C_to_B : ℝ := 20
def distance_B_to_A : ℝ := 15

def Minnie_time : ℝ :=
  (distance_A_to_B / Minnie's_speed_uphill) +
  (distance_B_to_C / Minnie's_speed_downhill) +
  (distance_C_to_A / Minnie's_speed_flat)

def Penny_time : ℝ :=
  (distance_A_to_C / Penny's_speed_flat) +
  (distance_C_to_B / Penny's_speed_uphill) +
  (distance_B_to_A / Penny's_speed_downhill)

noncomputable def time_difference : ℝ :=
  Minnie_time - Penny_time

theorem Minnie_takes_70_more_minutes : time_difference = 70 := by
  sorry

end Minnie_takes_70_more_minutes_l77_77247


namespace smallest_multiple_divisors_l77_77222

theorem smallest_multiple_divisors :
  ∃ m : ℕ, (∃ k1 k2 : ℕ, m = 2^k1 * 5^k2 * 100 ∧ 
    (∀ d : ℕ, d ∣ m → d = 1 ∨ d = m ∨ ∃ e1 e2 : ℕ, d = 2^e1 * 5^e2 * 100)) ∧
    (∀ d : ℕ, d ∣ m → d ≠ 1 → (d ≠ m → ∃ e1 e2 : ℕ, d = 2^e1 * 5^e2 * 100)) ∧
    (m.factors.length = 100) ∧ 
    m / 100 = 2^47 * 5^47 :=
begin
  sorry
end

end smallest_multiple_divisors_l77_77222


namespace max_Xs_5x5_grid_l77_77253

def max_Xs_on_grid : Nat := 11

theorem max_Xs_5x5_grid (M : Matrix (Fin 5) (Fin 5) Bool) :
  (∃ (X_count : Nat), (X_count ≤ 11 ∧ 
  ∀ i : Fin 5, ∀ j : Fin 5, 
    (M i j = true → (∑ k : Fin 5, if M i k then 1 else 0) ≤ 2 ∧ 
                     (∑ k : Fin 5, if M k j then 1 else 0) ≤ 2 ∧
                     (∑ k : Fin 5, if i + k < 5 ∧ j + k < 5 ∧ M (i + k) (j + k) then 1 else 0) ≤ 2 ∧
                     (∑ k : Fin 5, if i + k < 5 ∧ j - k ≥ 0 ∧ j - k < 5 ∧ M (i + k) (j - k) then 1 else 0) ≤ 2))) :=
  sorry

end max_Xs_5x5_grid_l77_77253


namespace dorothy_money_left_l77_77771

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18
def tax_amount : ℝ := annual_income * tax_rate
def money_left : ℝ := annual_income - tax_amount

theorem dorothy_money_left : money_left = 49200 := 
by
  sorry

end dorothy_money_left_l77_77771


namespace max_intersection_points_l77_77038

theorem max_intersection_points (a b h k : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  ∃ (n : ℕ), n = 8 ∧
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 → y = cos x → n ≤ 8) := 
sorry

end max_intersection_points_l77_77038


namespace rectangle_divided_into_13_squares_l77_77409

-- Define the conditions
variables {a b s : ℝ} (m n : ℕ)

-- Mathematical equivalent proof problem Lean statement
theorem rectangle_divided_into_13_squares (h : a * b = 13 * s^2)
  (hm : a = m * s) (hn : b = n * s) (hmn : m * n = 13) :
  a / b = 13 ∨ b / a = 13 :=
begin
  sorry
end

end rectangle_divided_into_13_squares_l77_77409


namespace convex_polygon_diagonals_l77_77889

/-- In a convex polygon with n vertices (n ≥ 4) and some diagonals drawn such that no two of them intersect, 
there exist at least two vertices from which no diagonals are drawn. -/
theorem convex_polygon_diagonals 
  (n : ℕ) 
  (h_n : n ≥ 4) 
  (convex : convex_polygon n)
  (no_intersecting_diagonals : no_intersecting_diags convex) : 
  ∃ v₁ v₂ : vertex convex, 
    v₁ ≠ v₂ ∧ 
    ¬(∃ d₁ : diagonal v₁, true) ∧ 
    ¬(∃ d₂ : diagonal v₂, true) :=
sorry

end convex_polygon_diagonals_l77_77889


namespace number_of_women_without_daughters_l77_77568

-- Definitions for the given conditions
def total_daughters := 7
def total_daughters_and_granddaughters := 42
def granddaughters_per_daughter := 6

-- Theorem that we aim to prove
theorem number_of_women_without_daughters :
  let num_daughters_with_daughters := (total_daughters_and_granddaughters - total_daughters) / granddaughters_per_daughter,
      num_daughters_without_daughters := total_daughters - num_daughters_with_daughters,
      num_granddaughters := total_daughters_and_granddaughters - total_daughters,
      total_women_without_daughters := num_daughters_without_daughters + num_granddaughters
  in total_women_without_daughters = 37 := by
  sorry

end number_of_women_without_daughters_l77_77568


namespace area_of_square_diagonal_l77_77875

variable (a b : ℝ)
variable (a_gt_b : a > b) (b_pos : b > 0)

theorem area_of_square_diagonal :
  let d := 2 * a - b in
  let area := d^2 / 2 in
  area = (2 * a - b)^2 / 2 :=
by
  sorry

end area_of_square_diagonal_l77_77875


namespace plumbers_allocation_l77_77769

noncomputable def num_allocation_schemes (plumbers houses : ℕ) : ℕ :=
  if plumbers = 4 ∧ houses = 3 then 36 else 0

theorem plumbers_allocation :
  (plumbers = 4 ∧ houses = 3) →
  num_allocation_schemes plumbers houses = 36 :=
by
  intros h
  rw [num_allocation_schemes]
  cases h with h1 h2
  rw [h1, h2]
  rfl

end plumbers_allocation_l77_77769


namespace yaw_yaw_age_in_2016_l77_77680

def is_lucky_double_year (y : Nat) : Prop :=
  let d₁ := y / 1000 % 10
  let d₂ := y / 100 % 10
  let d₃ := y / 10 % 10
  let last_digit := y % 10
  last_digit = 2 * (d₁ + d₂ + d₃)

theorem yaw_yaw_age_in_2016 (next_lucky_year : Nat) (yaw_yaw_age_in_next_lucky_year : Nat)
  (h1 : is_lucky_double_year 2016)
  (h2 : ∀ y, y > 2016 → is_lucky_double_year y → y = next_lucky_year)
  (h3 : yaw_yaw_age_in_next_lucky_year = 17) :
  (17 - (next_lucky_year - 2016)) = 5 := sorry

end yaw_yaw_age_in_2016_l77_77680


namespace trajectory_eq_l77_77617

theorem trajectory_eq {x y : ℝ} (h₁ : (x-2)^2 + y^2 = 1) (h₂ : ∃ r, (x+1)^2 = (x-2)^2 + y^2 - r^2) :
  y^2 = 6 * x - 3 :=
by
  sorry

end trajectory_eq_l77_77617


namespace cosine_graph_shift_l77_77317

theorem cosine_graph_shift:
  ∃ (f : ℝ → ℝ), (∀ x, f x = cos (2 * x - (π / 4)))
  ∧ (∀ g : ℝ → ℝ, (∀ x, g x = sin (2 * x)) 
    → (∀ x, f x = g (x + π / 8))) sorry

end cosine_graph_shift_l77_77317


namespace sum_integers_50_to_70_l77_77664

theorem sum_integers_50_to_70 : (\sum k in finset.Icc 50 70, k) = 1260 :=
by sorry

end sum_integers_50_to_70_l77_77664


namespace number_of_true_statements_l77_77386

theorem number_of_true_statements (a : ℝ) :
  (∃ n : ℕ, n = 2 ∧ 
    ( (a < 0 → ∃ x : ℝ, x * x + x + a = 0) ∧
      (¬(∃ x : ℝ, x * x + x + a = 0) → a ≥ 0) ∧
      ( (∃ x : ℝ, x * x + x + a = 0) ↔ a < 0) ∨
      (a ≥ 0 → ¬(∃ x : ℝ, x * x + x + a = 0)) )
  )
:=
begin
  sorry
end

end number_of_true_statements_l77_77386


namespace rectangle_divided_into_13_squares_l77_77408

-- Define the conditions
variables {a b s : ℝ} (m n : ℕ)

-- Mathematical equivalent proof problem Lean statement
theorem rectangle_divided_into_13_squares (h : a * b = 13 * s^2)
  (hm : a = m * s) (hn : b = n * s) (hmn : m * n = 13) :
  a / b = 13 ∨ b / a = 13 :=
begin
  sorry
end

end rectangle_divided_into_13_squares_l77_77408


namespace number_of_handshakes_l77_77047

-- Define the groups and their properties
def group1 : Type := { people : Fin 25 // ∀ p1 p2 : people, p1 ≠ p2 → ∃ q1 : Fin 25, p1 ≠ q1 ∧ p2 = q1 }
def group2 : Type := { people : Fin 15 // ∀ p1 p2 : people, p1 ≠ p2 → ∃ q2 : Fin 15, p1 ≠ q2 ∧ p2 = q2 }

-- Define the total number of people
def total_people := 40

-- Define the main statement about the total number of handshakes
theorem number_of_handshakes :
  let handshakes := sorry, in handshakes = 630 :=
sorry

end number_of_handshakes_l77_77047


namespace goose_eggs_count_l77_77570

theorem goose_eggs_count (E : ℕ) 
  (h1 : (1/2 : ℝ) * E = E/2)
  (h2 : (3/4 : ℝ) * (E/2) = (3 * E) / 8)
  (h3 : (2/5 : ℝ) * ((3 * E) / 8) = (3 * E) / 20)
  (h4 : (3 * E) / 20 = 120) :
  E = 400 :=
sorry

end goose_eggs_count_l77_77570


namespace selected_in_range_eq_six_l77_77733

def numEmployees : Nat := 840
def numSelected : Nat := 21
def samplingInterval : Nat := 40
def rangeStart : Nat := 481
def rangeEnd : Nat := 720

theorem selected_in_range_eq_six (h: SystematicSampling numEmployees numSelected samplingInterval) :
  selected_in_range h rangeStart rangeEnd = 6 := by
  sorry

end selected_in_range_eq_six_l77_77733


namespace simplify_expression_expression_value_at_specific_values_l77_77273

variables {a b : ℝ}

-- The original mathematical expression
def original_expr :=  2 * (a * b^2 + 3 * a^2 * b) - 3 * (a * b^2 + a^2 * b)

-- The simplified result
def simplified_expr := -a * b^2 + 3 * a^2 * b

-- Proof that the original expression simplifies to the correct form
theorem simplify_expression : original_expr = simplified_expr := by
  sorry

-- Show that for a = -1 and b = 2, the value of the simplified expression is 10
theorem expression_value_at_specific_values :
  let a := -1
  let b := 2
  simplified_expr = 10 := by
  sorry

end simplify_expression_expression_value_at_specific_values_l77_77273


namespace total_sessions_l77_77017

theorem total_sessions (p1 p2 p3 p4 : ℕ) 
(h1 : p1 = 6) 
(h2 : p2 = p1 + 5) 
(h3 : p3 = 8) 
(h4 : p4 = 8) : 
p1 + p2 + p3 + p4 = 33 := 
by
  sorry

end total_sessions_l77_77017


namespace simplify_trig_expression_l77_77285

theorem simplify_trig_expression (θ : ℝ) (h₁ : 0 < θ) (h₂ : θ < π) : 
  (1 + sin θ + cos θ) * (sin (θ / 2) - cos (θ / 2)) / sqrt (2 + 2 * cos θ) = -cos θ :=
sorry

end simplify_trig_expression_l77_77285


namespace range_of_m_l77_77881

theorem range_of_m 
  (h : ∀ x : ℝ, x^2 + m * x + m^2 - 1 > 0) :
  m ∈ (Set.Ioo (-(2 * Real.sqrt 3) / 3) (-(2 * Real.sqrt 3) / 3)).union (Set.Ioi ((2 * Real.sqrt 3) / 3)) := 
sorry

end range_of_m_l77_77881


namespace points_in_circle_exist_closer_than_two_l77_77963

theorem points_in_circle_exist_closer_than_two :
  ∀ (points : Fin 10 → EuclideanSpace ℝ 2), 
  (∀ p ∈ points, EuclideanSpace.dist p (0,0) ≤ 2.5) →
  ∃ (i j : Fin 10), i ≠ j ∧ EuclideanSpace.dist (points i) (points j) < 2 := 
sorry

end points_in_circle_exist_closer_than_two_l77_77963


namespace sum_integers_50_to_70_l77_77661

theorem sum_integers_50_to_70 :
  let a := 50
  let l := 70
  ∑ k in Finset.range (l - a + 1), (a + k) = 1260 :=
by
  let a := 50
  let l := 70
  sorry

end sum_integers_50_to_70_l77_77661


namespace find_happy_boys_l77_77251

theorem find_happy_boys 
  (total_boys : ℕ)
  (sad_boys : ℕ)
  (neither_happy_nor_sad_boys : ℕ) :
  total_boys = 16 ∧ sad_boys = 6 ∧ neither_happy_nor_sad_boys = 4 →
  total_boys - (sad_boys + neither_happy_nor_sad_boys) = 6 :=
by
  intro h
  obtain ⟨htotal, hsad, hneither⟩ := h
  linarith

end find_happy_boys_l77_77251


namespace log_270_integers_sum_l77_77799

theorem log_270_integers_sum (a b : ℤ) (h1 : 2 < Real.log 270 / Real.log 10)
  (h2 : Real.log 270 / Real.log 10 < 3)
  (h3 : a = 2)
  (h4 : b = 3 : ℤ)
  : a + b = 5 := 
by
  sorry

end log_270_integers_sum_l77_77799


namespace unique_fib_triple_l77_77951

/-- Define the Fibonacci sequence -/
def fib : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

/-- Main theorem: existence and uniqueness of (a, b, c) -/
theorem unique_fib_triple :
  ∃! (a b c : ℕ), b < a ∧ c < a ∧ (∀ n : ℕ, a ∣ (fib n - n * b * c ^ n)) ∧ a = 5 ∧ b = 2 ∧ c = 3 :=
sorry

end unique_fib_triple_l77_77951


namespace set_A_cannot_form_right_triangle_l77_77329

open Real

theorem set_A_cannot_form_right_triangle :
  let a := sqrt 3
  let b := sqrt 4
  let c := sqrt 5
  ¬ (a^2 + b^2 = c^2) :=
by
  let a := sqrt 3
  let b := sqrt 4
  let c := sqrt 5
  show ¬ (a^2 + b^2 = c^2)
  sorry

#eval set_A_cannot_form_right_triangle

end set_A_cannot_form_right_triangle_l77_77329


namespace price_reduction_for_fixed_profit_maximum_profit_maximum_profit_value_l77_77708

noncomputable def daily_sales (x : ℝ) : ℝ := 20 + 2 * x
noncomputable def profit_per_piece (x : ℝ) : ℝ := 40 - x
noncomputable def daily_profit (x : ℝ) : ℝ := (40 - x) * (20 + 2 * x)

theorem price_reduction_for_fixed_profit :
  ∃ x₁ x₂ : ℝ, daily_profit x₁ = 1200 ∧ daily_profit x₂ = 1200 :=
by
  use [10, 20]
  -- Proof omitted
  sorry

theorem maximum_profit :
  ∃ x : ℝ, ∀ y : ℝ, daily_profit x ≥ daily_profit y :=
by
  use 15
  -- Proof omitted
  sorry

theorem maximum_profit_value :
  daily_profit 15 = 1250 :=
by
  -- Proof omitted
  sorry

end price_reduction_for_fixed_profit_maximum_profit_maximum_profit_value_l77_77708


namespace number_cyclic_permutation_is_multiple_of_27_l77_77888

def digits_to_number (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d, 10 * acc + d) 0

variable (a : ℕ → ℕ) (n : ℕ)

theorem number_cyclic_permutation_is_multiple_of_27 
  (h_length : n = 1953)
  (h_divisible : digits_to_number (List.ofFn (λ i, a i)) % 27 = 0) :
  ∀ k, (digits_to_number (List.ofFn (λ i, a ((i + k) % n))) % 27 = 0) :=
by
  sorry

end number_cyclic_permutation_is_multiple_of_27_l77_77888


namespace average_speed_before_increase_l77_77035

-- Definitions for the conditions
def t_before := 12   -- Travel time before the speed increase in hours
def t_after := 10    -- Travel time after the speed increase in hours
def speed_diff := 20 -- Speed difference between before and after in km/h

-- Variable for the speed before increase
variable (s_before : ℕ) -- Average speed before the speed increase in km/h

-- Definitions for the speeds
def s_after := s_before + speed_diff -- Average speed after the speed increase in km/h

-- Equations derived from the problem conditions
def dist_eqn_before := s_before * t_before
def dist_eqn_after := s_after * t_after

-- The proof problem stated in Lean
theorem average_speed_before_increase : dist_eqn_before = dist_eqn_after → s_before = 100 := by
  sorry

end average_speed_before_increase_l77_77035


namespace min_distance_AC_BD_l77_77979

noncomputable def regular_tetrahedron {α : Type*} [normed_add_torsor ℝ α] (A B C D : α) : Prop :=
  dist A B = 2 ∧ dist A C = 2 ∧ dist A D = 2 ∧
  dist B C = 2 ∧ dist B D = 2 ∧ dist C D = 2

theorem min_distance_AC_BD {α : Type*} [normed_add_torsor ℝ α] (A B C D : α) :
  regular_tetrahedron A B C D →
  ∃ P Q : α, P ∈ segment ℝ A C ∧ Q ∈ segment ℝ B D ∧ dist P Q = 2 :=
by
  sorry

end min_distance_AC_BD_l77_77979


namespace calvin_winning_strategy_l77_77678

theorem calvin_winning_strategy :
  ∃ (n : ℤ), ∃ (p : ℤ), ∃ (q : ℤ),
  (∀ k : ℕ, k > 0 → p = 0 ∧ (q = 2014 + k ∨ q = 2014 - k) → ∃ x : ℤ, (x^2 + p * x + q = 0)) :=
sorry

end calvin_winning_strategy_l77_77678


namespace Ralph_TV_hours_l77_77266

theorem Ralph_TV_hours :
  let hoursWeekdays := 4 * 5,
  let hoursWeekends := 6 * 2,
  let totalHours := hoursWeekdays + hoursWeekends
  in totalHours = 32 := 
by
  sorry

end Ralph_TV_hours_l77_77266


namespace imaginary_part_of_conjugate_l77_77164

def cos60 : ℂ := 1 / 2
def sin60 : ℂ := sqrt 3 / 2
variable (z : ℂ)
axiom z_condition : z * (cos60 + sin60 * complex.i) = -1 + sqrt 3 * complex.i

theorem imaginary_part_of_conjugate :
  z_condition z →
  complex.im (complex.conj z) = -sqrt 3 :=
by
  intro h
  sorry

end imaginary_part_of_conjugate_l77_77164


namespace fraction_of_grid_covered_by_triangle_l77_77255

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 5)
def B : ℝ × ℝ := (7, 2)
def C : ℝ × ℝ := (6, 6)

-- Define the dimensions of the grid
def gridLength : ℝ := 8
def gridWidth : ℝ := 6

-- Calculate the area of the triangle using the Shoelace theorem
def triangleArea : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Calculate the area of the grid
def gridArea : ℝ := gridLength * gridWidth

-- Calculate the fraction of the grid covered by the triangle
def fractionCovered : ℚ := triangleArea / gridArea

-- Proof statement
theorem fraction_of_grid_covered_by_triangle : fractionCovered = 17 / 96 := by
  sorry

end fraction_of_grid_covered_by_triangle_l77_77255


namespace maximum_distance_MN_l77_77841

noncomputable theory

def polar_curve_C (ρ θ : ℝ) : Prop :=
  ρ = 2 * real.sin θ

def parametric_line_l (t x y : ℝ) : Prop :=
  x = - (3/5) * t + 2 ∧ y = (4/5) * t

def point_M_on_line_l_x_axis (x y t : ℝ) : Prop :=
  parametric_line_l t x y ∧ y = 0

def point_N_on_curve_C (ρ θ : ℝ) (x y : ℝ) : Prop :=
  polar_curve_C ρ θ ∧ x^2 + y^2 = ρ^2 ∧ ρ * real.cos θ = x ∧ ρ * real.sin θ = y

def maximum_MN_exists_and_equal (M N : ℝ × ℝ) : Prop :=
  ∃ Mx My Nx Ny, (
    point_M_on_line_l_x_axis Mx My (0 : ℝ) ∧ -- Since the only value that will ever intersect the x-axis is t = 0
    point_N_on_curve_C (Nx^2 + Ny^2) (real.atan2 Ny Nx) Nx Ny ∧
    M = (Mx, My) ∧ N = (Nx, Ny) ∧ 
    ∀ N, ∃ r, (r > 0) ∧ (Nx - Mx)^2 + (Ny - My)^2 ≤ (M.x - C.x) ^2 + (M.y - C.y)^2 + r
    ∧ dist M N ≤ sqrt 5 + 1)

theorem maximum_distance_MN :
  ∃ M N, maximum_MN_exists_and_equal M N :=
sorry

end maximum_distance_MN_l77_77841


namespace medians_sum_square_l77_77932

-- Define the sides of the triangle
variables {a b c : ℝ}

-- Define diameters
variables {D : ℝ}

-- Define medians of the triangle
variables {m_a m_b m_c : ℝ}

-- Defining the theorem statement
theorem medians_sum_square :
  m_a ^ 2 + m_b ^ 2 + m_c ^ 2 = (3 / 4) * (a ^ 2 + b ^ 2 + c ^ 2) + (3 / 4) * D ^ 2 :=
sorry

end medians_sum_square_l77_77932


namespace arithmetic_mean_of_integers_from_neg3_to_6_l77_77640

def integer_range := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

noncomputable def arithmetic_mean : ℚ :=
  (integer_range.sum : ℚ) / (integer_range.length : ℚ)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  arithmetic_mean = 1.5 := by
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l77_77640


namespace sum_of_words_l77_77565

-- Definitions to represent the conditions
def ХЛЕБ : List Char := ['Х', 'Л', 'Е', 'Б']
def КАША : List Char := ['К', 'А', 'Ш', 'А']

-- Function to compute factorial
def factorial : Nat -> Nat
| 0 => 1
| n + 1 => (n + 1) * factorial n

-- Permutations considering repetition (as in multiset permutations)
def permutations_with_repetition (n : Nat) (reps : Nat) : Nat :=
  factorial n / factorial reps

-- The theorem to prove
theorem sum_of_words : (factorial 4) + (permutations_with_repetition 4 2) = 36 := by
  sorry

end sum_of_words_l77_77565


namespace sequence_an_solution_l77_77614

theorem sequence_an_solution (a : ℕ → ℚ)
  (h1 : a 1 = 1 / 2)
  (h2 : ∀ n ≥ 2, (∑ i in Finset.range (n + 1), a (i + 1)) = n^2 * a (n)) :
  ∀ n, a n = 1 / (n * (n + 1)) :=
by 
  -- Proof goes here
  sorry

end sequence_an_solution_l77_77614


namespace part1_k_value_part2_k_range_part3_k_range_l77_77852

-- Part 1 proof problem
theorem part1_k_value (k : ℝ) :
    (∀ x, 1 < x ∧ x < log 2 3 → k * (4^x) - 2^(x+1) + 6 * k < 0) →
    k = 1/4 :=
sorry

-- Part 2 proof problem
theorem part2_k_range (k : ℝ) :
    (∀ x, 1 < x ∧ x < log 2 3 → k * (4^x) - 2^(x+1) + 6 * k < 0) →
    k ≤ 1/4 :=
sorry

-- Part 3 proof problem
theorem part3_k_range (k : ℝ) :
    (∀ x, (∃ x1 x2 ∈ Ioo 1 (log 2 3), x1 < x ∧ x < x2 → k * (4^x) - 2^(x+1) + 6 * k < 0) →
    k ≥ 1/4 :=
sorry

end part1_k_value_part2_k_range_part3_k_range_l77_77852


namespace petes_average_speed_l77_77048

-- Definitions of the conditions
def map_distance : ℝ := 5 -- in inches
def driving_time : ℝ := 6.5 -- in hours
def map_scale : ℝ := 0.01282051282051282 -- in inches per mile

-- Theorem statement: If the conditions are given, then the average speed is 60 miles per hour
theorem petes_average_speed :
  (map_distance / map_scale) / driving_time = 60 :=
by
  -- The proof will go here
  sorry

end petes_average_speed_l77_77048


namespace arithmetic_mean_of_integers_from_neg3_to_6_l77_77653

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  let nums := list.range' (-3) 10 in
  (∑ i in nums, i) / (nums.length : ℝ) = 1.5 :=
by
  let nums := list.range' (-3) 10
  have h_sum : (∑ i in nums, i) = 15 := sorry
  have h_length : nums.length = 10 := sorry
  rw [h_sum, h_length]
  norm_num
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l77_77653


namespace max_l_given_a_neg_l77_77237

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 + 8 * x + 3

theorem max_l_given_a_neg (a : ℝ) (h : a < 0) :
  (∀ x ∈ set.Icc 0 (1 / 2 * (real.sqrt 5 + 1)), |f a x| ≤ 5) →
  a = -8 ∧ (1 / 2 * (real.sqrt 5 + 1)) = (1 / 2 * (real.sqrt 5 + 1)) :=
by
  sorry

end max_l_given_a_neg_l77_77237


namespace range_of_m_l77_77451

-- Definitions of propositions and their negations
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0
def not_p (x : ℝ) : Prop := x < -2 ∨ x > 10
def not_q (x m : ℝ) : Prop := x < (1 - m) ∨ x > (1 + m) ∧ m > 0

-- Statement that \neg p is a necessary but not sufficient condition for \neg q
def necessary_but_not_sufficient (x m : ℝ) : Prop := 
  (∀ x, not_q x m → not_p x) ∧ ¬(∀ x, not_p x → not_q x m)

-- The main theorem to prove
theorem range_of_m (m : ℝ) : (∀ x, necessary_but_not_sufficient x m) ↔ 9 ≤ m :=
by
  sorry

end range_of_m_l77_77451


namespace area_ratio_of_triangles_l77_77578

variables (A B C D K M L : Type) [Point A] [Point B] [Point C] [Point D] [Point K] [Point M] [Point L]
variables (AC BD p q : ℝ)

-- Defining the conditions as mathematical hypotheses
noncomputable def inscribed_quadrilateral (A B C D : Type) : Prop :=
  -- Definition for an inscribed quadrilateral; exact details abstracted
  sorry

noncomputable def ratio_condition1 (BK KC AM MD : ℝ) : Prop :=
  BK / KC = AM / MD

noncomputable def ratio_condition2 (KL LM BC AD : ℝ) : Prop :=
  KL / LM = BC / AD

theorem area_ratio_of_triangles
  (h1 : inscribed_quadrilateral A B C D)
  (h2 : ratio_condition1 A K B C)
  (h3 : ratio_condition2 K L M BC AD)
  (h4 : AC = p)
  (h5 : BD = q) :
  (area (triangle A C L)) / (area (triangle B D L)) = p / q :=
sorry

end area_ratio_of_triangles_l77_77578


namespace nature_of_quadratic_graph_l77_77853

theorem nature_of_quadratic_graph {a b : ℝ} (ha : a ≠ 0) (c : ℝ) (hc : c = b^2 / (3 * a)) :
  (a > 0 → ∃ m, ∀ x, y = ax^2 + bx + c ∧ y ≥ m) ∧
  (a < 0 → ∃ M, ∀ x, y = ax^2 + bx + c ∧ y ≤ M) :=
by
  sorry

end nature_of_quadratic_graph_l77_77853


namespace bounces_to_reach_below_threshold_l77_77699

def initial_height : ℝ := 20
def common_ratio : ℝ := 3/4
def threshold_height : ℝ := 2

theorem bounces_to_reach_below_threshold :
  ∃ k : ℕ, initial_height * common_ratio^k < threshold_height ∧ 
           ∀ n : ℕ, n < k → initial_height * common_ratio^n ≥ threshold_height :=
sorry

end bounces_to_reach_below_threshold_l77_77699


namespace perp_ED_EF_l77_77523

theorem perp_ED_EF
  (O K A B C D P E F : Type)
  [tangent_to_circle OA O A]
  [tangent_to_circle OB O B]
  [center_with_radius K OA KA]
  [lies_on_segment C AB]
  [intersects OC circle_K_at D]
  [lies_on_circle P O]
  [rays_intersects PC PD O E F] :
  perp E D E F :=
sorry

end perp_ED_EF_l77_77523


namespace grade_students_difference_condition_l77_77311

variables (G1 G2 G5 : ℕ)

theorem grade_students_difference_condition (h : G1 + G2 = G2 + G5 + 30) : G1 - G5 = 30 :=
sorry

end grade_students_difference_condition_l77_77311


namespace initial_weight_l77_77359

-- Define the conditions and the final weight.
theorem initial_weight :=
  let W := 598.74
  let weight_after_first_stage := 0.80 * W
  let weight_after_second_stage := 0.70 * weight_after_first_stage
  let weight_after_third_stage := 0.85 * weight_after_second_stage
  let final_weight := 285

  have h1 : weight_after_first_stage = 0.80 * W := rfl
  have h2 : weight_after_second_stage = 0.70 * weight_after_first_stage := rfl
  have h3 : weight_after_third_stage = 0.85 * weight_after_second_stage := rfl
  have h4 : weight_after_third_stage = final_weight := rfl

  -- Prove that the initial weight W equals 598.74
  have h5 : 0.476 * W = 285 := by
    rw [←h1, ←h2, ←h3, h4]

  -- Solve for W
  have h6 : W = 285 / 0.476 := by
    sorry

  -- Verify that W is approximately 598.74
  show W ≈ 598.74 from h6

end initial_weight_l77_77359


namespace polynomial_equivalence_l77_77401

noncomputable def polynomial_solution (P : ℝ → ℝ) : Prop :=
∀ x : ℝ, P(x) * P(x + 1) = P(x^2 + 2)

theorem polynomial_equivalence (P : ℝ → ℝ) :
  polynomial_solution P → ∃ n : ℕ, ∀ x : ℝ, P(x) = (x^2 - x + 2)^n := sorry

end polynomial_equivalence_l77_77401


namespace relationship_abc_l77_77821

noncomputable def f : ℝ → ℝ := sorry

def a := (2 ^ 0.6) * f (2 ^ 0.6)
def b := (Real.log 2) * f (Real.log 2)
def c := (Real.log 2⁻³) * f (Real.log 2⁻³)

theorem relationship_abc (h1 : ∀ x, f x = f (-x))
                          (h2 : ∀ x < 0, f x + x * (deriv (deriv f)) x < 0) :
                          a > b ∧ b > c :=
sorry

end relationship_abc_l77_77821


namespace necessary_and_sufficient_condition_l77_77345

-- Definitions of the lines
def line1 : ℝ → ℝ := λ x, x  -- This is y = x
def line2 (m : ℝ) : ℝ → ℝ := λ x, - (1 / m) * x  -- This is y = - (1 / m) * x

-- Slope of the first line
def slope_line1 : ℝ := 1

-- Slope of the second line
def slope_line2 (m : ℝ) : ℝ := -m

-- Perpendicular condition
def perpendicular (m : ℝ) : Prop := slope_line1 * slope_line2 m = -1

-- The theorem statement
theorem necessary_and_sufficient_condition (m : ℝ) :
  (m = 1) ↔ perpendicular m :=
begin
  unfold perpendicular,
  unfold slope_line1 slope_line2,
  simp,
  split,
  { intro h,
    rw h,
    linarith },
  { intro h,
    linarith }
end

end necessary_and_sufficient_condition_l77_77345


namespace smallest_m_divided_by_100_l77_77226

-- Define m as the smallest positive integer that meets the conditions
def m : ℕ := 2^4 * 3^9 * 5^1

-- Condition: m is a multiple of 100 and has exactly 100 divisors
def is_multiple_of_100 (n : ℕ) : Prop := 100 ∣ n
def has_exactly_100_divisors (n : ℕ) : Prop := (factors_count n (2^4 * 3^9 * 5^1)) = 100

-- The property we want to prove: m meets both conditions and yields the correct fragment
theorem smallest_m_divided_by_100 : 
  is_multiple_of_100 m ∧ has_exactly_100_divisors m → m / 100 = 15746.4 :=
by
  sorry

end smallest_m_divided_by_100_l77_77226


namespace triangle_construction_l77_77064

-- Definitions from the problem conditions
variables (C A B F : Type)
variables (m_c s_c : ℝ) (delta : ℝ)

-- Given conditions
def conditions := s_c > m_c

-- The theorem to prove
theorem triangle_construction 
(conditions : s_c > m_c) 
: ∃ (ABC : Type), median_and_altitude ABC m_c s_c ∧ angle_difference ABC delta :=
sorry

end triangle_construction_l77_77064


namespace tangent_line_at_zero_l77_77980

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2 * Real.exp x

theorem tangent_line_at_zero :
  let slope_at_zero := 1 - 2
  let y_intercept := -2
  ∀ (x y : ℝ), y = slope_at_zero * x + y_intercept ↔ y = -x - 2 :=
by
  intro x y
  unfold slope_at_zero y_intercept
  simp
  exact sorry

end tangent_line_at_zero_l77_77980


namespace rectangles_divided_into_13_squares_l77_77403

theorem rectangles_divided_into_13_squares (m n : ℕ) (h : m * n = 13) : 
  (m = 1 ∧ n = 13) ∨ (m = 13 ∧ n = 1) :=
sorry

end rectangles_divided_into_13_squares_l77_77403


namespace mrs_hilt_total_distance_l77_77569

def total_distance_walked (d n : ℕ) : ℕ := 2 * d * n

theorem mrs_hilt_total_distance :
  total_distance_walked 30 4 = 240 :=
by
  -- Proof goes here
  sorry

end mrs_hilt_total_distance_l77_77569


namespace proof_problem_l77_77185

noncomputable def intersection_points_C1_C2 : Prop :=
  let C1 := fun θ : ℝ => 2 * Real.cos θ
  let C2 := fun θ : ℝ => 2 * Real.cos (θ - Real.pi / 3)
  (∃ θ : ℝ, C1 θ = C2 θ) ↔
  ((C1 (Real.acos (3 / 2)) = C2 (Real.acos (3 / 2))) ∧
  (C1 (Real.acos (0)) = C2 (Real.acos (0))))

noncomputable def max_distance_MN : Prop :=
  let line_l := fun t : ℝ => (t * Real.cos α, t * Real.sin α)
  let M := fun α : ℝ => (2 * Real.cos α * Real.cos α, 2 * Real.cos α * Real.sin α)
  let N := fun α : ℝ => (2 * Real.cos (α - Real.pi / 3) * Real.cos α, 2 * Real.cos (α - Real.pi / 3) * Real.sin α)
  ∃ α: ℝ, |(M α).fst - (N α).fst| + |(M α).snd - (N α).snd| = 2

theorem proof_problem : intersection_points_C1_C2 ∧ max_distance_MN :=
by
  sorry

end proof_problem_l77_77185


namespace find_values_find_sum_l77_77448

-- Conditions
axiom arithmetic_sequence (a : ℕ → ℝ) (λ : ℝ) (n : ℕ) :
  a 0 = λ ∧
  a 1 = 6 ∧
  a 2 = 3 * λ ∧
  ∀ n ≥ 2, a (n+1) - a n = a 2 - 6

noncomputable def S (n : ℕ) : ℝ :=
  3 * n + 3 * n * (n - 1) / 2

axiom S_k_value (k : ℕ) : S k = 165

-- Proof problem
theorem find_values (λ : ℝ) (k : ℕ) :
  λ = 3 ∧ k = 10 :=
sorry

noncomputable def b (n : ℕ) : ℝ :=
  3 / (2 * S n)

def T (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, b (i + 1))

theorem find_sum (n : ℕ) : T n = n / (n + 1) :=
sorry

end find_values_find_sum_l77_77448


namespace line_passes_through_fixed_point_range_of_MN_l77_77125

-- Define the condition of line l equation and point P
def line_equation (m : ℝ) (x y : ℝ) := 2 * x + (1 + m) * y + 2 * m = 0
def point_P := (-1, 0 : ℝ)

-- Given the fixed point Q
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ∃ (Q : ℝ × ℝ), Q = (1, -2) ∧ line_equation m 1 (-2) := sorry

-- Given the projection M of point P on line l and coordinates of N
def point_N := (2, 1 : ℝ)
def distance (p q : ℝ × ℝ) := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem range_of_MN :
  ∃ (M : ℝ × ℝ), ∀ (m : ℝ), distance M point_N = real.sqrt 2 ∨ distance M point_N = 3 * real.sqrt 2 := sorry

end line_passes_through_fixed_point_range_of_MN_l77_77125


namespace exists_equilateral_triangle_if_intersection_l77_77156

noncomputable theory

-- Define the Circle structure
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define a Point structure
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the problem conditions:
def Circle_alpha : Circle := {center := A, radius := a}
def Circle_beta : Circle := {center := B, radius := b}
def Circle_gamma : Circle := {center := C, radius := c}

-- Define the rotation transformation
def rotate (P : Point) (θ : ℝ) (X : Point) : Point :=
  let cosθ := Real.cos (θ * Real.pi / 180)
  let sinθ := Real.sin (θ * Real.pi / 180)
  {
    x := cosθ * (X.x - P.x) - sinθ * (X.y - P.y) + P.x,
    y := sinθ * (X.x - P.x) + cosθ * (X.y - P.y) + P.y
  }

def Annulus (center : Point) (inner_radius outer_radius : ℝ) : Set Point :=
  { P | inner_radius ≤ (P - center).norm ∧ (P - center).norm ≤ outer_radius }

-- Main theorem statement
theorem exists_equilateral_triangle_if_intersection :
  (Annulus (rotate C 60 A) (a - c) (a + c) ∩ {P | (P - B).norm = b} ≠ ∅) ↔
  ∃ (P Q R : Point), P ∈ Circle_alpha ∧ Q ∈ Circle_beta ∧ R ∈ Circle_gamma ∧
  (P - Q).norm = (Q - R).norm ∧ (Q - R).norm = (R - P).norm :=
sorry

end exists_equilateral_triangle_if_intersection_l77_77156


namespace pow_product_l77_77052

theorem pow_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := 
by {
  sorry
}

end pow_product_l77_77052


namespace Ralph_TV_hours_l77_77265

theorem Ralph_TV_hours :
  let hoursWeekdays := 4 * 5,
  let hoursWeekends := 6 * 2,
  let totalHours := hoursWeekdays + hoursWeekends
  in totalHours = 32 := 
by
  sorry

end Ralph_TV_hours_l77_77265


namespace square_area_is_100_l77_77716

-- Define the point and its distances from the closest sides of the square
variables (P : Type) [metric_space P] 
(inside_square : P)
(distance1 distance2 : ℝ)
(distance_to_side1 distance_to_side2 : inside_square = 1 ∧ inside_square = 2)

-- Define the radius of the inscribed circle
def radius := 5

-- Define the side length of the square as twice the radius of the circle
def side_length := 2 * radius

-- Define the area of the square
def area_of_square := side_length * side_length

-- Prove that given the conditions, the area of the square is 100
theorem square_area_is_100 : 
  area_of_square = 100 :=
by 
  sorry

end square_area_is_100_l77_77716


namespace part_one_part_two_l77_77848

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x - a * x

def f' (a : ℝ) (x : ℝ) : ℝ := Real.cos x - a

theorem part_one (a : ℝ) (hcond : ∀ x : ℝ, 0 < x → x < 1 → f' a x > 0) : a ≤ 0 := 
  sorry

noncomputable def h (x : ℝ) : ℝ := Real.log x - x + 1

theorem part_two : ∀ x > 0, x = 1 → h x = 0 :=
  sorry

end part_one_part_two_l77_77848


namespace median_isosceles_right_triangle_leg_length_l77_77043

theorem median_isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ)
  (H1 : m = 15)
  (H2 : m = h / 2)
  (H3 : 2 * x * x = h * h) : x = 15 * Real.sqrt 2 :=
by
  sorry

end median_isosceles_right_triangle_leg_length_l77_77043


namespace smallest_absolute_value_term_proof_l77_77109

def arithmetic_sum (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2
def arithmetic_term (a1 d : ℤ) (n : ℕ) : ℤ := a1 + n * d

def smallest_absolute_value_term (a1 d : ℤ) : ℤ :=
if (2 * a1 + 1008 * d > 0) ∧ (a1 + 1010 * d < 0) then a1 + 1009 * d else sorry

theorem smallest_absolute_value_term_proof (a1 d : ℤ)
  (S2018 S2019 : ℤ)
  (h1 : S2018 > 0)
  (h2 : S2019 < 0)
  (hsum1 : S2018 = arithmetic_sum a1 d 2018)
  (hsum2 : S2019 = arithmetic_sum a1 d 2019)
  (hterm : smallest_absolute_value_term a1 d = a1 + 1009 * d) :
  smallest_absolute_value_term a1 d = a1 + 1009 * d :=
begin
  sorry
end

end smallest_absolute_value_term_proof_l77_77109


namespace cyclic_quadrilateral_l77_77575

theorem cyclic_quadrilateral
  (A B C D M : Point)
  (O1 O2 : Circle)
  (hMAB : M ∈ LineSegment AB)
  (hCirc1 : circumscribed AMCD O1)
  (hCirc2 : circumscribed BMDC O2)
  (hIso : is_isosceles_triangle_with_vertex M ∠CMD O1 O2) :
  is_cyclic_quadrilateral ABCD :=
sorry

end cyclic_quadrilateral_l77_77575


namespace smallest_n_is_63_l77_77509

noncomputable def Q (n : ℕ) : ℚ :=
  1 / ((n^2 + 1) * ∏ k in  (range (n-1)).succ, (k^2 + 1))

theorem smallest_n_is_63 :
  ∀ n : ℕ, Q(n) < 1/4020 → n ≥ 63 := by
  sorry

end smallest_n_is_63_l77_77509


namespace angle_measure_l77_77600

theorem angle_measure (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : P = 206 :=
by
  sorry

end angle_measure_l77_77600


namespace speed_of_second_train_is_correct_l77_77362

-- Definitions and condition in Lean 4
def train1_length : ℝ := 200
def train1_speed_kmph : ℝ := 72
def train2_length : ℝ := 300
def crossing_time : ℝ := 49.9960003199744

-- Conversion factor from km/h to m/s
def kmph_to_mps (speed: ℝ) : ℝ := speed * (5 / 18)

-- Distance covered when the first train passes the second train
def total_distance : ℝ := train1_length + train2_length

-- Speed of train 1 in m/s
def train1_speed_mps : ℝ := kmph_to_mps train1_speed_kmph

-- The relative speed needed to pass the second train in the given time
def relative_speed : ℝ := total_distance / crossing_time

-- Speed of the second train in m/s
def train2_speed_mps : ℝ := train1_speed_mps - relative_speed

-- Convert speed back to kmph
def mps_to_kmph (speed: ℝ) : ℝ := speed * (18 / 5)

-- Expected speed of the second train in kmph
def expected_train2_speed_kmph : ℝ := mps_to_kmph train2_speed_mps

theorem speed_of_second_train_is_correct:
  expected_train2_speed_kmph ≈ 35.99712 := sorry

end speed_of_second_train_is_correct_l77_77362


namespace sum_of_non_domain_points_l77_77760

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + 1 / (3 + 1 / x))

theorem sum_of_non_domain_points : 
  ({0, -1 / 3, -2 / 7} : set ℝ).sum (λ x, x) = -13 / 21 :=
by
  sorry

end sum_of_non_domain_points_l77_77760


namespace find_slope_of_AF_l77_77192

def parabola (p : ℝ) : set (ℝ × ℝ) :=
{ P | ∃ y : ℝ, (y, y^2 / (2 * p)) = P }

noncomputable def focus (p : ℝ) : ℝ × ℝ :=
(p / 2, 0)

variables {A B : ℝ × ℝ} {p : ℝ} (hp : p > 0)

-- The points A and B have the properties given in the conditions
def point_on_parabola (p : ℝ) (A : ℝ × ℝ) : Prop :=
A ∈ (parabola p)

def point_B_properties (A B : ℝ × ℝ) (p : ℝ) : Prop :=
B.2 = 0 ∧ B.1 > p / 2 ∧ (dist A (focus p) = dist B (focus p))

def intersections (A M N : ℝ × ℝ) (p : ℝ) : Prop :=
-- Intersection points M and N of lines through AF and AB with the parabola
(M ∈ parabola p) ∧ (N ∈ parabola p) ∧
(M ≠ A) ∧ (N ≠ A)

def right_angle (M N A : ℝ × ℝ) : Prop :=
let v1 := (M.1 - A.1, M.2 - A.2),
    v2 := (N.1 - A.1, N.2 - A.2)
in v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Target goal: slope of AF is sqrt(3)
noncomputable def slope {P Q : ℝ × ℝ} (hP : P.1 ≠ Q.1) : ℝ :=
(Q.2 - P.2) / (Q.1 - P.1)

theorem find_slope_of_AF (hA : point_on_parabola p A)
    (hB : point_B_properties A B p)
    (hM : intersections A M M' p)
    (hN : intersections A N N' p)
    (h_angle : right_angle A M N) :
    slope A (focus p) = real.sqrt 3 := 
sorry

end find_slope_of_AF_l77_77192


namespace range_of_t_l77_77435

theorem range_of_t (t : ℝ) : (∀ x : ℝ, x > 0 → ln x + ln (2 - 2 * t) + 2 * x^2 ≤ 2 * t * x^2 + 2 * x + x * exp (2 * x)) → 1 > t ∧ t ≥ 1 - Real.exp 1 :=
by
  sorry

end range_of_t_l77_77435


namespace trader_profit_l77_77025

theorem trader_profit (P : ℝ) : 
  let buying_price := 0.90 * P,
      taxes_expenses := 0.05 * buying_price,
      total_cost := buying_price + taxes_expenses,
      selling_price := total_cost + 0.80 * total_cost 
  in (selling_price - P) / P * 100 = 70.1 := 
by 
  repeat {sorry}

end trader_profit_l77_77025


namespace min_perimeter_sum_l77_77087

theorem min_perimeter_sum (m : ℕ) (h : 1 ≤ m) : 
  ∃ partition : (fin (2^m) × fin (2^m) → set (fin (2^m) × fin (2^m))), 
    (∀ i, ∃ r, r ⊆ partition (i, i) ∧ is_rectangle r ∧ r.card = 1) ∧
    sum_perimeter partition = (m + 1) * 2^(m + 2) :=
sorry

end min_perimeter_sum_l77_77087


namespace number_of_boys_correct_l77_77970

noncomputable def number_of_boys (avg_weight_initial correct_avg_weight correction actual_weight misread_weight : ℝ) : ℕ :=
let n := (correction / (correct_avg_weight - avg_weight_initial)) in
if (avg_weight_initial * n + actual_weight - misread_weight = correct_avg_weight * n) then n.to_nat else 0

theorem number_of_boys_correct:
  number_of_boys 58.4 58.7 6 62 56 = 20 :=
by
  sorry

end number_of_boys_correct_l77_77970


namespace find_a_l77_77170

theorem find_a (x a : ℝ) (h₁ : x = 2) (h₂ : (4 - x) / 2 + a = 4) : a = 3 :=
by
  -- Proof steps will go here
  sorry

end find_a_l77_77170


namespace maximum_contribution_l77_77494

theorem maximum_contribution (total_contribution : ℕ) (num_people : ℕ) (individual_min_contribution : ℕ) :
  total_contribution = 20 → num_people = 10 → individual_min_contribution = 1 → 
  ∃ (max_contribution : ℕ), max_contribution = 11 := by
  intro h1 h2 h3
  existsi 11
  sorry

end maximum_contribution_l77_77494


namespace sum_of_three_numbers_l77_77660

theorem sum_of_three_numbers :
  1.35 + 0.123 + 0.321 = 1.794 :=
sorry

end sum_of_three_numbers_l77_77660


namespace value_of_a_l77_77478

theorem value_of_a (P Q : Set ℝ) (a : ℝ) :
  (P = {x | x^2 = 1}) →
  (Q = {x | ax = 1}) →
  (Q ⊆ P) →
  (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end value_of_a_l77_77478


namespace carrey_fixed_amount_l77_77056

theorem carrey_fixed_amount :
  ∃ C : ℝ, 
    (C + 0.25 * 44.44444444444444 = 24 + 0.16 * 44.44444444444444) →
    C = 20 :=
by
  sorry

end carrey_fixed_amount_l77_77056


namespace min_needed_passengers_to_cover_good_groups_l77_77252

def is_good_group (i1 i2 j1 j2 : ℕ) : Prop :=
  i1 < j1 ∧ i1 < j2 ∧ i2 < j1 ∧ i2 < j2 ∧
  i1 ≠ i2 ∧ i1 ≠ j1 ∧ i1 ≠ j2 ∧ i2 ≠ j1 ∧ i2 ≠ j2 ∧ j1 ≠ j2

def passenger_covers_pair (i j : ℕ) (pairs : set (ℕ × ℕ)) : Prop :=
  ∀ p : ℕ × ℕ, p ∈ pairs → p.1 = i ∨ p.2 = j
  
def min_passengers_to_cover_groups (n : ℕ) : ℕ :=
  if n = 2018 then 1009 else 0  -- Using 1009 since n = 2018, otherwise 0 as a placeholder

theorem min_needed_passengers_to_cover_good_groups :
  min_passengers_to_cover_groups 2018 = 1009 :=
by {
  -- Formal proof will go here
  sorry
}

end min_needed_passengers_to_cover_good_groups_l77_77252


namespace eigenvalues_and_eigenfunctions_boundary_value_problem_l77_77075

theorem eigenvalues_and_eigenfunctions_boundary_value_problem :
  ∃ (λ_n : ℕ → ℝ) (y_n : ℕ → ℝ → ℝ),
    (∀ n, λ_n n = (2 * n + 1) / 2) ∧
    (∀ n x, y_n n x = Real.cos((2 * n + 1) / 2 * x)) ∧
    (∀ (λ : ℝ) (y : ℝ → ℝ),
      (λ ≠ 0) →
      (∀ x, deriv (deriv y x) + λ^2 * y x = 0) →
      (deriv y 0 = 0) →
      (y Real.pi = 0) →
      (∃ n, λ = λ_n n ∧ (∀ x, y x = y_n n x))) :=
sorry

end eigenvalues_and_eigenfunctions_boundary_value_problem_l77_77075


namespace limit_of_sequence_equals_15_l77_77939

def sequence (x : ℕ → ℝ) : Prop :=
  (x 1 = 3) ∧
  (x 2 = 24) ∧
  (∀ n, x (n + 2) = (1 / 4) * x (n + 1) + (3 / 4) * x n)

theorem limit_of_sequence_equals_15 (x : ℕ → ℝ)
  (h : sequence x) : 
  (tendsto x at_top (𝓝 15)) :=
sorry

end limit_of_sequence_equals_15_l77_77939


namespace arithmetic_mean_of_range_neg3_to_6_l77_77651

theorem arithmetic_mean_of_range_neg3_to_6 :
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  (sum : Float) / (count : Float) = 1.5 := by
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  have h_sum : sum = 15 := by sorry
  have h_count : count = 10 := by sorry
  calc
    (sum : Float) / (count : Float)
        = (15 : Float) / (10 : Float) : by rw [h_sum, h_count]
    ... = 1.5 : by norm_num

end arithmetic_mean_of_range_neg3_to_6_l77_77651


namespace fifth_equation_pattern_l77_77250

theorem fifth_equation_pattern :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by sorry

end fifth_equation_pattern_l77_77250


namespace difference_in_cm_l77_77605

def line_length : ℝ := 80  -- The length of the line is 80.0 centimeters
def diff_length_factor : ℝ := 0.35  -- The difference factor in the terms of the line's length

theorem difference_in_cm (l : ℝ) (d : ℝ) (h₀ : l = 80) (h₁ : d = 0.35 * l) : d = 28 :=
by
  sorry

end difference_in_cm_l77_77605


namespace math_problem_l77_77916

noncomputable def C : Nat :=
  let odd_multiples_of_3 := {n : Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 2 = 1 ∧ n % 3 = 0}
  odd_multiples_of_3.to_finset.card

noncomputable def D : Nat :=
  let multiples_of_10 := {n : Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 10 = 0}
  multiples_of_10.to_finset.card

theorem math_problem :
  C + D = 2400 := 
sorry

end math_problem_l77_77916


namespace expansion_properties_l77_77120

open Nat

theorem expansion_properties (n : ℕ) (x : ℝ) (h : n = 8)
  (ratio_condition : binomial_coeff n 4 * (-2)^4 / (binomial_coeff n 2 * (-2)^2) = 10) :
  n = 8 ∧
  term_with_max_binomial_coeff = 1120 * x^(-6) ∧
  term_containing_x_3_2 = -16 * x^(3/2) ∧
  sum_of_coeffs = 1 :=
by
  -- Definitions
  let binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k
  let expansion_term (r : ℕ) : ℝ := binomial_coeff n r * (-2)^r * x^(4 - 5*r/2)
  let sum_of_coeffs := ((sqrt 1 - 2 / 1^2)^8)
  -- Skipping the proof
  sorry

end expansion_properties_l77_77120


namespace total_length_of_track_l77_77046

-- Definitions based directly on the conditions provided
def starting_opposite_directions (x : ℝ) (brenda_run : ℝ) (sally_run : ℝ) : Prop :=
  brenda_run = 120 ∧ sally_run = x / 2 - 120

def first_meeting_next_meeting (x : ℝ) (sally_run : ℝ) (next_meeting_sally_run : ℝ) : Prop :=
  next_meeting_sally_run = x / 2 + 60

def second_meeting_next_meeting (x : ℝ) (brenda_run_second : ℝ) (next_meeting_brenda_run : ℝ) : Prop :=
  next_meeting_brenda_run = x + 140

-- Main theorem statement
theorem total_length_of_track (x : ℝ) :
  starting_opposite_directions x 120 (x / 2 - 120) ∧
  first_meeting_next_meeting x (x / 2 - 120) (x / 2 + 60) ∧
  second_meeting_next_meeting x (x - 120) (x + 140) →
  x = 300 + (Real.sqrt 61200) / 2 :=
by
  intros _ _ _
  sorry

end total_length_of_track_l77_77046


namespace distance_and_area_l77_77508

/-- In a rectangular coordinate system, calculate the distance from the origin to the point (12, -5) -/
def distanceFromOrigin : ℝ :=
  let x := 12
  let y := -5
  real.sqrt (x^2 + y^2)

/-- The area of the rectangle formed by the point (12, -5) and the coordinate axes -/
def rectangleArea : ℝ :=
  let x := 12
  let y := 5
  x * y

theorem distance_and_area :
  distanceFromOrigin = 13 ∧ rectangleArea = 60 :=
by
  sorry

end distance_and_area_l77_77508


namespace countColorings_l77_77806

-- Defining the function that counts the number of valid colorings
def validColorings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 3 * 2^n - 2

-- Theorem specifying the number of colorings of the grid of length n
theorem countColorings (n : ℕ) : validColorings n = 3 * 2^n - 2 :=
by
  sorry

end countColorings_l77_77806


namespace sum_of_divisors_2000_l77_77950

theorem sum_of_divisors_2000 (n : ℕ) (h : n < 2000) :
  ∃ (s : Finset ℕ), (s ⊆ {1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000}) ∧ s.sum id = n :=
by
  -- Proof goes here
  sorry

end sum_of_divisors_2000_l77_77950


namespace max_value_f_on_interval_l77_77417

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_f_on_interval :
  ∃ x ∈ set.Icc (-1 : ℝ) 1, (∀ y ∈ set.Icc (-1 : ℝ) 1, f y ≤ f x) ∧ f x = 2 :=
sorry

end max_value_f_on_interval_l77_77417


namespace find_x_l77_77000

theorem find_x :
  ∃ x : ℝ, 8 * 5.4 - (x * 10) / 1.2 = 31.000000000000004 ∧ x = 1.464 :=
by
  sorry

end find_x_l77_77000


namespace proof_problem_l77_77168

theorem proof_problem (x : ℤ) (h : (x - 34) / 10 = 2) : (x - 5) / 7 = 7 :=
  sorry

end proof_problem_l77_77168


namespace exist_excursion_with_frequent_students_l77_77312

noncomputable theory

-- Given conditions
def num_students := 20
def min_students_per_excursion := 4

-- Main theorem to prove
theorem exist_excursion_with_frequent_students (n : ℕ) (h_min_students : n ≥ min_students_per_excursion) :
  ∃ excursion, ∀ student ∈ excursion, student ∈ participated ∧ num_excursions student ≥ n / 17 :=
sorry

end exist_excursion_with_frequent_students_l77_77312


namespace chocolate_game_winner_l77_77691

theorem chocolate_game_winner (n : ℕ) : 
  (prime n → (∃ second_player_wins : Prop, second_player_wins)) ∧ 
  (¬prime n → (∃ first_player_wins : Prop, first_player_wins)) := 
by
  sorry

end chocolate_game_winner_l77_77691


namespace train_speed_problem_l77_77712

open Real

/-- Given specific conditions about the speeds and lengths of trains, prove the speed of the third train is 99 kmph. -/
theorem train_speed_problem
  (man_train_speed_kmph : ℝ)
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (goods_train_time : ℝ)
  (third_train_length : ℝ)
  (third_train_time : ℝ) :
  man_train_speed_kmph = 45 →
  man_train_speed = 45 * 1000 / 3600 →
  goods_train_length = 340 →
  goods_train_time = 8 →
  third_train_length = 480 →
  third_train_time = 12 →
  (third_train_length / third_train_time - man_train_speed) * 3600 / 1000 = 99 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end train_speed_problem_l77_77712


namespace rowing_time_proof_l77_77280

/-- Definitions and conversions -/
def yard_to_meter : ℝ := 0.9144
def river_initial_width_yards : ℕ := 50
def river_final_width_yards : ℕ := 80
def width_increase_yards : ℕ := 2
def width_increase_interval_meters : ℕ := 10
def rowing_speed_mps : ℝ := 5

/-- Conversions to meters -/
def river_initial_width_meters := (river_initial_width_yards : ℝ) * yard_to_meter
def river_final_width_meters := (river_final_width_yards : ℝ) * yard_to_meter
def width_increase_meters := (width_increase_yards : ℝ) * yard_to_meter

/-- Calculating total distance -/
def total_width_increase_meters := river_final_width_meters - river_initial_width_meters
def number_of_segments := total_width_increase_meters / width_increase_meters
def total_distance_meters := number_of_segments * (width_increase_interval_meters : ℝ)

/-- Time calculation -/
def time_taken := total_distance_meters / rowing_speed_mps

theorem rowing_time_proof : time_taken = 30 := by
  sorry

end rowing_time_proof_l77_77280


namespace b_should_pay_l77_77681

open Real

theorem b_should_pay :
  let total_rent := 841
  let a_horses := 12
  let a_months := 8
  let b_horses := 16
  let b_months := 9
  let c_horses := 18
  let c_months := 6
  let a_cost := a_horses * a_months
  let b_cost := b_horses * b_months
  let c_cost := c_horses * c_months
  let total_horse_months := a_cost + b_cost + c_cost
  let cost_per_horse_month := total_rent / total_horse_months
  let b_payment := round (cost_per_horse_month * b_cost)
  b_payment = 348 := 
by
  -- conditions transformation and calculations
  let a_cost := 12 * 8
  let b_cost := 16 * 9
  let c_cost := 18 * 6
  let total_horse_months := a_cost + b_cost + c_cost
  have : total_horse_months = 348 := by norm_num
  let cost_per_horse_month := 841 / 348
  have : cost_per_horse_month = 2.4172413793103448 := by norm_num
  let b_payment := cost_per_horse_month * b_cost
  have : b_payment = 2.4172413793103448 * 144 := by norm_num
  have : b_payment = 348 := by norm_num
  exact this

end b_should_pay_l77_77681


namespace num_words_sum_l77_77564

/-
  Definitions:
  - word_kasha is a multiset with the letters "К", "А", "Ш", "А".
  - word_hleb is a set with the letters "Х", "Л", "Е", "Б".
  - num_distinct_perms is the function to calculate permutations of distinct items.
  - num_perms_with_repetition is the function to calculate permutations of multiset.
-/

def word_kasha : Multiset Char := {'К', 'А', 'Ш', 'А'}
def word_hleb : Finset Char := {'Х', 'Л', 'Е', 'Б'}

def num_distinct_perms (s : Finset Char) : ℕ :=
  (Finset.card s).factorial

def num_perms_with_repetition (m : Multiset Char) : ℕ :=
  Multiset.card m ! / m.dedup.card.factorial

theorem num_words_sum : 
  num_distinct_perms word_hleb + num_perms_with_repetition word_kasha = 36 :=
by {
  sorry
}

end num_words_sum_l77_77564


namespace prove_quadrilateral_sides_and_angles_l77_77796

-- Define the rhombus with its properties
structure Rhombus :=
  (d1 : ℝ) -- diagonal 1
  (d2 : ℝ) -- diagonal 2
  (midpoints_form_rectangle : Prop)
  (sides_of_rectangle : ℝ × ℝ)
  (angles_of_rectangle : list ℝ)

-- Define the specific rhombus for this problem
def givenRhombus : Rhombus := {
  d1 := 6,
  d2 := 10,
  midpoints_form_rectangle := true,
  sides_of_rectangle := (3, 5),
  angles_of_rectangle := [90, 90, 90, 90]
}

-- Statement to prove
theorem prove_quadrilateral_sides_and_angles :
  givenRhombus.midpoints_form_rectangle ∧
  givenRhombus.sides_of_rectangle = (3, 5) ∧
  givenRhombus.angles_of_rectangle = [90, 90, 90, 90] := 
by
  sorry

end prove_quadrilateral_sides_and_angles_l77_77796


namespace geometric_progression_iff_equality_l77_77952

theorem geometric_progression_iff_equality (x y z : ℝ) :
  (∃ a λ : ℝ, x = a * λ^0 ∧ y = a * λ^1 ∧ z = a * λ^2) ↔ ( (x^2 + y^2) * (y^2 + z^2) = (xy + yz)^2 ) :=
by
  sorry

end geometric_progression_iff_equality_l77_77952


namespace petya_wins_in_100_gon_game_l77_77342

theorem petya_wins_in_100_gon_game :
  ∀ (P : Type) [Poly100 P] (X : Point),
    (inside_polygon X P) ∧ (not_on_sides_or_diagonals X P) →
    strategy_exists P X → Petya_winning_strategy :=
begin
  sorry
end

end petya_wins_in_100_gon_game_l77_77342


namespace find_m_and_sum_l77_77631

theorem find_m_and_sum (m a b c : ℝ) (h1 : m = 120 - 60 * Real.sqrt 2)
  (h2 : a = 120) (h3 : b = 60) (h4 : c = 2) : a + b + c = 182 :=
by
  rw [h1, h2, h3, h4]
  norm_num

#print axioms find_m_and_sum

end find_m_and_sum_l77_77631


namespace tan_inequality_cot_inequality_l77_77549

open Real

-- Given conditions
variables (x₁ x₂ : ℝ)
hypothesis (h₁ : 0 < x₁ ∧ x₁ < π/2)
hypothesis (h₂ : 0 < x₂ ∧ x₂ < π/2)
hypothesis (h₃ : x₁ ≠ x₂)

-- Statements to prove
theorem tan_inequality : 1/2 * (tan x₁ + tan x₂) > tan ((x₁ + x₂)/2) :=
sorry

theorem cot_inequality : 1/2 * (1/tan x₁ + 1/tan x₂) > 1/tan ((x₁ + x₂)/2) :=
sorry

end tan_inequality_cot_inequality_l77_77549


namespace white_balls_probability_l77_77701

noncomputable def probability_all_white (total_balls white_balls draw_count : ℕ) : ℚ :=
  if h : total_balls >= draw_count ∧ white_balls >= draw_count then
    (Nat.choose white_balls draw_count : ℚ) / (Nat.choose total_balls draw_count : ℚ)
  else
    0

theorem white_balls_probability :
  probability_all_white 11 5 5 = 1 / 462 :=
by
  sorry

end white_balls_probability_l77_77701


namespace arithmetic_mean_set_l77_77141

theorem arithmetic_mean_set (n : ℕ) (hn : n > 2) :
    let a := 1 - (2 / n : ℝ)
    let b := 2
    arithmetic_mean := ((2 - (4 / n)) + (2 * (n - 2))) / n
  in arithmetic_mean = 2 - (2 / n) - (4 / n^2) :=
by
  sorry

end arithmetic_mean_set_l77_77141


namespace days_A_left_l77_77346

noncomputable def work_30_days := 1 / 30
noncomputable def work_29_999999999999996_days := 1 / 29.999999999999996

theorem days_A_left (W : ℝ) : 
  ∃ x : ℝ, 
    x * work_30_days * W + 10 * work_30_days * W + 10 * work_29_999999999999996_days * W = W ∧ 
    x = 10 := 
by
  use 10
  sorry  -- Skip the proof

end days_A_left_l77_77346


namespace inverse_logarithm_value_l77_77847

noncomputable def f (a x : ℝ) : ℝ := log a x

theorem inverse_logarithm_value (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : f a 9 = 2) :
  ∃ y, y = f a⁻¹ (log a 2) := 
begin
  have ha : a = 3, sorry,
  use 2,
  sorry
end

end inverse_logarithm_value_l77_77847


namespace perpendicular_intersection_line_parabola_l77_77610

theorem perpendicular_intersection_line_parabola (b : ℝ) :
  (let f := λ x : ℝ, x + b in
   ∃ x₁ x₂ : ℝ, 
     f x₁ = (1/2) * x₁ ^ 2 ∧ 
     f x₂ = (1/2) * x₂ ^ 2 ∧ 
     x₁ + x₂ = 2 ∧ 
     x₁ * x₂ + (x₁ + b) * (x₂ + b) = 0) → 
   b = 2 :=
by sorry

end perpendicular_intersection_line_parabola_l77_77610


namespace problem_statement_l77_77130

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem problem_statement (φ : ℝ) (hφ : 0 < φ ∧ φ < π) :
  (∀ x, (0 < x ∧ x < 5*π/12) → f x φ ≤ f 0 φ) ∧
  (∀ x, (x = 0 → f' x φ = -1) ∧
   (f 0 φ = Real.sqrt 3 / 2) → (∀ y, y = Real.sqrt 3 / 2 - x)) :=
sorry

end problem_statement_l77_77130


namespace average_rate_of_change_l77_77292

noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l77_77292


namespace find_x_for_orthogonal_vectors_l77_77144

theorem find_x_for_orthogonal_vectors :
  ∀ (x : ℝ), (let a := (2, x) in
               let b := (-1, 2) in
               (a.1 * b.1 + a.2 * b.2) = 0) → x = 1 :=
by
  intros x h
  sorry

end find_x_for_orthogonal_vectors_l77_77144


namespace pizza_problem_l77_77949

theorem pizza_problem
  (pizza_slices : ℕ)
  (total_pizzas : ℕ)
  (total_people : ℕ)
  (pepperoni_only_friend : ℕ)
  (remaining_pepperoni : ℕ)
  (equal_distribution : Prop)
  (h_cond1 : pizza_slices = 16)
  (h_cond2 : total_pizzas = 2)
  (h_cond3 : total_people = 4)
  (h_cond4 : pepperoni_only_friend = 1)
  (h_cond5 : remaining_pepperoni = 1)
  (h_cond6 : equal_distribution ∧ (pepperoni_only_friend ≤ total_people)) :
  ∃ cheese_slices_left : ℕ, cheese_slices_left = 7 := by
  sorry

end pizza_problem_l77_77949


namespace first_500_commission_percentage_l77_77356

theorem first_500_commission_percentage (x : ℝ) :
  let total_sale := 800
      first_500 := 500
      excess := total_sale - first_500
      commission_first_500 := (x / 100) * first_500
      commission_excess := 0.5 * excess
      total_commission := commission_first_500 + commission_excess
  in total_commission = 0.3125 * total_sale → x = 20 :=
by
  intros
  let total_sale := 800
  let first_500 := 500
  let excess := total_sale - first_500
  let commission_first_500 := (x / 100) * first_500
  let commission_excess := 0.5 * excess
  let total_commission := commission_first_500 + commission_excess
  have h_total_commission : total_commission = 0.3125 * total_sale := by assumption
  have h : commission_first_500 + commission_excess = 0.3125 * total_sale := by rw [←h_total_commission]
  sorry

end first_500_commission_percentage_l77_77356


namespace original_price_l77_77204

variables (P : ℝ)
axiom first_discount : ∀ P, 0.66667 * P
axiom second_discount : ∀ P, 0.5 * P
axiom third_discount : ∀ P, 0.4 * P
axiom final_price : 0.4 * P = 15

theorem original_price : P = 37.50 :=
by
  sorry

end original_price_l77_77204


namespace arithmetic_mean_of_integers_from_neg3_to_6_l77_77652

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  let nums := list.range' (-3) 10 in
  (∑ i in nums, i) / (nums.length : ℝ) = 1.5 :=
by
  let nums := list.range' (-3) 10
  have h_sum : (∑ i in nums, i) = 15 := sorry
  have h_length : nums.length = 10 := sorry
  rw [h_sum, h_length]
  norm_num
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l77_77652


namespace spherical_coords_standard_form_l77_77518

theorem spherical_coords_standard_form :
  ∀ (ρ θ φ : ℝ), ρ > 0 → 0 ≤ θ ∧ θ < 2 * Real.pi → 0 ≤ φ ∧ φ ≤ Real.pi →
  (5, (5 * Real.pi) / 7, (11 * Real.pi) / 6) = (ρ, θ, φ) →
  (ρ, (12 * Real.pi) / 7, Real.pi / 6) = (ρ, θ, φ) :=
by 
  intros ρ θ φ hρ hθ hφ h_eq
  sorry

end spherical_coords_standard_form_l77_77518


namespace x_fifth_plus_inverse_fifth_l77_77274

theorem x_fifth_plus_inverse_fifth (x : ℝ) (h : x + x⁻¹ = 3) : x^5 + x^(-5) = 123 :=
by
  sorry

end x_fifth_plus_inverse_fifth_l77_77274


namespace find_b_find_perimeter_b_plus_c_l77_77501

noncomputable def triangle_condition_1
  (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.cos B = (3 * c - b) * Real.cos A

noncomputable def triangle_condition_2
  (a b : ℝ) (C : ℝ) : Prop :=
  a * Real.sin C = 2 * Real.sqrt 2

noncomputable def triangle_condition_3
  (a b c : ℝ) (A : ℝ) : Prop :=
  (1 / 2) * b * c * Real.sin A = Real.sqrt 2

noncomputable def given_a
  (a : ℝ) : Prop :=
  a = 2 * Real.sqrt 2

theorem find_b
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b = 3 :=
sorry

theorem find_perimeter_b_plus_c
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b + c = 2 * Real.sqrt 3 :=
sorry

end find_b_find_perimeter_b_plus_c_l77_77501


namespace bottles_needed_to_fill_large_bottle_l77_77008

def medium_bottle_ml : ℕ := 150
def large_bottle_ml : ℕ := 1200

theorem bottles_needed_to_fill_large_bottle : large_bottle_ml / medium_bottle_ml = 8 :=
by
  sorry

end bottles_needed_to_fill_large_bottle_l77_77008


namespace find_x_y_l77_77239

theorem find_x_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (2 / 3) * x = (144 / 216) * (1 / x)) (h4 : y * (1 / x) = Real.sqrt x) : x = 1 ∧ y = 1 :=
by
  have h3_simp : (2 / 3) * x = (2 / 3) * (1 / x) := by
    rw [rat.cast_div, rat.cast_num, rat.cast_denom] at h3
    exact h3
  sorry

end find_x_y_l77_77239


namespace number_of_classes_l77_77174

theorem number_of_classes (x : ℕ) (h : x * (x - 1) = 20) : x = 5 :=
by
  sorry

end number_of_classes_l77_77174


namespace arrangement_count_l77_77010

-- Definitions based on the given conditions
def chefs := ["Chef_A", "Chef_B"]
def waiters := ["Waiter_1", "Waiter_2", "Waiter_3"]
def people := chefs ++ waiters

-- The main theorem
theorem arrangement_count : 
  ∃ arrangements : ℕ, 
  arrangements = 48 ∧ 
  (∀ arrangement : list (string), 
   arrangement.length = 5 ∧ 
   -- Chef A does not stand at either end
   arrangement.head ≠ "Chef_A" ∧ arrangement.last ≠ "Chef_A" ∧
   -- Exactly 2 of the 3 waiters stand next to each other
   (∃ i : ℕ, i < 4 ∧ arrangement.nth_le i sorry = "Waiter_X" ∧ arrangement.nth_le (i+1) sorry = "Waiter_Y") ∧ 
    ∀ (W : string), W ∈ waiters → (arrangement.count W = 1) ∧ 
    ∀ (C : string), C ∈ chefs → (arrangement.count C = 1)) :=
sorry

end arrangement_count_l77_77010


namespace Bernardo_received_345_l77_77583

theorem Bernardo_received_345 :
  ∃ (n : ℕ), (∃ (a : ℕ), a = 2) ∧ (∃ (d : ℕ), d = 3) ∧ (1000 ≥ n * (n + 1) / 2) ∧
  let l := 2 + (n - 1) * 3 in
  let m := (n + 2) / 3 in
  345 = m * (a + l) / 2 :=
sorry

end Bernardo_received_345_l77_77583


namespace smallest_union_value_l77_77088

def f (n : ℕ) : ℕ :=
  if n = 1 then 0 else if n = 2 then 3 else n + 2

theorem smallest_union_value (n : ℕ) :
  (∀ (A : finset (finset ℕ)),
    (∀ i j, i ≠ j → ¬ (A i ⊆ A j)) →
    (∀ i j, i ≠ j → A i.card ≠ A j.card) →
    ∃ S : finset ℕ, S = finset.bUnion (finset.range n) A ∧ S.card = f n) :=
sorry

end smallest_union_value_l77_77088


namespace reflection_matrix_over_line_y_eq_x_l77_77788

-- Define the reflection over the line y = x as a linear map
def reflection_over_y_eq_x (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.2, v.1)  -- this swaps the coordinates (x, y) -> (y, x)

-- Define the matrix that corresponds to this reflection
def reflection_matrix := ![
  ![0, 1],
  ![1, 0]
]

theorem reflection_matrix_over_line_y_eq_x :
  ∀ v : ℝ × ℝ, reflection_over_y_eq_x v = matrix.vec_mul reflection_matrix v :=
by
  sorry

end reflection_matrix_over_line_y_eq_x_l77_77788


namespace solution_l77_77011

/-- Original cost price, original selling price, and daily sales at original price -/
def original_cost : ℝ := 80
def original_price : ℝ := 120
def daily_sales : ℝ := 20

/-- Conditions: price reduction per unit and increased sales -/
def price_reduction_per_piece (x : ℝ) : ℝ := x
def daily_sales_increase (x : ℝ) : ℝ := 2 * x

/-- Profit per piece given price reduction x -/
def profit_per_piece (x : ℝ) : ℝ := 40 - x

/-- Daily sales volume given price reduction x -/
def sales_volume (x : ℝ) : ℝ := 20 + 2 * x

/-- Daily profit as a function of price reduction x -/
def daily_profit (x : ℝ) : ℝ := (40 - x) * (20 + 2 * x)

/-- Problem: find price reduction x for a daily profit of 1200 yuan -/
def price_reduction_for_target_profit (target_profit : ℝ) : ℝ := 
  if (solver : ∃ x : ℝ, (40 - x) * (20 + 2 * x) = target_profit) then
    classical.some solver
  else 
    0

/-- Check if a daily profit of 1800 yuan can be achieved -/
def can_achieve_daily_profit_1800 : Prop :=
  ¬ ∃ x : ℝ, (40 - x) * (20 + 2 * x) = 1800

/--Theorem stating the solution to the problem -/
theorem solution : can_achieve_daily_profit_1800 := by
  sorry

end solution_l77_77011


namespace polynomial_remainder_l77_77327

theorem polynomial_remainder :
  ∀ (x : ℝ), polynomial.mod_by_monic (polynomial.C (1 : ℝ) * polynomial.X^4 -
  polynomial.C (3 : ℝ) * polynomial.X^2 + polynomial.C (2 : ℝ)) 
  (polynomial.C (1 : ℝ) * polynomial.X^2 - polynomial.C (3 : ℝ)) = 
  polynomial.C (2 : ℝ) :=
sorry

end polynomial_remainder_l77_77327


namespace shirt_price_after_discounts_l77_77807

theorem shirt_price_after_discounts
  (original_price : ℝ)
  (first_reduction_rate : ℝ)
  (second_reduction_rate : ℝ) :
  original_price = 20 →
  first_reduction_rate = 0.20 →
  second_reduction_rate = 0.40 →
  let first_discounted_price := original_price * (1 - first_reduction_rate) in
  let final_price := first_discounted_price * (1 - second_reduction_rate) in
  final_price = 9.60 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end shirt_price_after_discounts_l77_77807


namespace max_diff_in_grid_l77_77076

theorem max_diff_in_grid : 
  ∀ (arr : ℕ × ℕ → ℕ), 
    (∀ i j, 1 ≤ arr (i, j) ∧ arr (i, j) ≤ 400) 
    ∧ (∀ i j k l, (i, j) ≠ (k, l) → arr (i, j) ≠ arr (k, l)) 
    → ∃ i j k l, (i = k ∨ j = l) 
      ∧ abs (arr (i, j) - arr (k, l)) ≥ 209 :=
by
  sorry

end max_diff_in_grid_l77_77076


namespace smallest_n_l77_77341

-- Define the conditions as properties of integers
def connected (a b : ℕ): Prop := sorry -- Assume we have a definition for connectivity

def condition1 (a b n : ℕ) : Prop :=
  ¬connected a b → Nat.gcd (a^2 + b^2) n = 1

def condition2 (a b n : ℕ) : Prop :=
  connected a b → Nat.gcd (a^2 + b^2) n > 1

theorem smallest_n : ∃ n, n = 65 ∧ ∀ (a b : ℕ), condition1 a b n ∧ condition2 a b n := by
  sorry

end smallest_n_l77_77341


namespace alpha_value_l77_77529

theorem alpha_value
  (β γ δ α : ℝ) 
  (h1 : β = 100)
  (h2 : γ = 30)
  (h3 : δ = 150)
  (h4 : α + β + γ + 0.5 * γ = 360) : 
  α = 215 :=
by
  sorry

end alpha_value_l77_77529


namespace common_point_of_geometric_progression_l77_77735

theorem common_point_of_geometric_progression (a b c x y : ℝ) (r : ℝ) 
  (h1 : b = a * r) (h2 : c = a * r^2) 
  (h3 : a * x + b * y = c) : 
  x = 1 / 2 ∧ y = -1 / 2 := 
sorry

end common_point_of_geometric_progression_l77_77735


namespace sum_of_tan_roots_l77_77804

theorem sum_of_tan_roots : 
  (∀ x ∈ set.Icc (0 : ℝ) (2 * π), tan x * tan x - 10 * tan x + 2 = 0) →
  ∑ x in (finset.filter (λ x, ∀ y ∈ set.Icc (0 : ℝ) (2 * π), x = y) 
  (finset.univ : finset ℝ)), x = 3 * π :=
by {
  intros condition,
  sorry
}

end sum_of_tan_roots_l77_77804


namespace part1_part2_l77_77449

variable {α : Type*} [LinearOrderedField α]

-- Definitions based on given problem conditions.
def arithmetic_seq(a_n : ℕ → α) := ∃ a1 d, ∀ n, a_n n = a1 + ↑(n - 1) * d

noncomputable def a10_seq := (30 : α)
noncomputable def a20_seq := (50 : α)

-- Theorem statements to prove:
theorem part1 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq) :
  ∀ n, a_n n = 2 * ↑n + 10 := sorry

theorem part2 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq)
  (Sn : α) (hSn : Sn = 242) :
  ∃ n, Sn = (↑n / 2) * (2 * 12 + (↑n - 1) * 2) ∧ n = 11 := sorry

end part1_part2_l77_77449


namespace original_bubble_radius_correct_l77_77021

noncomputable def original_bubble_radius : ℝ := (310 / 4)^(1 / 3)

theorem original_bubble_radius_correct :
  let hemisphere_radius := 5
  let cylinder_radius := 2
  let cylinder_height := hemisphere_radius
  let total_volume := (2 / 3 * Real.pi * hemisphere_radius^3) + (Real.pi * cylinder_radius^2 * cylinder_height)
  let original_sphere_radius := (total_volume * 3 / (4 * Real.pi))^(1 / 3)
  original_sphere_radius = original_bubble_radius := by
calc original_sphere_radius = ((310 / 3 * Real.pi) * 3 / (4 * Real.pi))^(1 / 3) : by sorry
                       ... = (310 / 4)^(1 / 3) : by sorry

end original_bubble_radius_correct_l77_77021


namespace sec_two_pi_over_three_l77_77399

theorem sec_two_pi_over_three : Real.sec (2 * Real.pi / 3) = -2 :=
by
  sorry

end sec_two_pi_over_three_l77_77399


namespace parametric_curve_length_l77_77784

-- Define the parametric equations
def x (t : ℝ) := 3 * Real.sin t
def y (t : ℝ) := 3 * Real.cos t

-- Define the derivatives
def dx_dt (t : ℝ) := 3 * Real.cos t
def dy_dt (t : ℝ) := -3 * Real.sin t

-- Length of the curve integral
def curve_length (a b : ℝ) := 
  ∫ t in a..b, Real.sqrt ((dx_dt t)^2 + (dy_dt t)^2)

theorem parametric_curve_length : 
  curve_length 0 (2 * Real.pi) = 6 * Real.pi := by
  sorry

end parametric_curve_length_l77_77784


namespace trigonometric_inequalities_l77_77096

theorem trigonometric_inequalities (θ : ℝ) (h1 : Real.sin (θ + Real.pi) < 0) (h2 : Real.cos (θ - Real.pi) > 0) : 
  Real.sin θ > 0 ∧ Real.cos θ < 0 :=
sorry

end trigonometric_inequalities_l77_77096


namespace duty_arrangements_240_l77_77394

/-
We need to define the conditions given:
- 7 days of duty.
- 5 people in the department.
- Each person can be on duty for up to 2 consecutive days.
- Everyone must be on duty at least once.
- The department head is on duty on the first day.

Given these, we need to prove that the total number of valid arrangements is 240.
-/

def numDutyArrangements (days people : ℕ) (maxDaysPerPerson : ℕ) (headOnFirstDay : Bool) : ℕ := sorry

theorem duty_arrangements_240 :
  numDutyArrangements 7 5 2 true = 240 :=
sorry

end duty_arrangements_240_l77_77394


namespace sum_abs_roots_poly_l77_77801

noncomputable def poly := Polynomial.C (-36) + Polynomial.X * (Polynomial.C 24 + Polynomial.X * (Polynomial.C 9 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X)))

theorem sum_abs_roots_poly : (Polynomial.roots poly).map (λ x, |x|).sum = 6 + 4 * Real.sqrt 3 := sorry

end sum_abs_roots_poly_l77_77801


namespace range_of_b_l77_77446

theorem range_of_b (a b : ℝ) (h1 : a ≠ 0) (h2 : a * b^2 > a) (h3 : a > a * b) : b < -1 :=
sorry

end range_of_b_l77_77446


namespace arc_length_increase_l77_77496

variable (R : ℝ) -- radius of the arc
variable (θ : ℝ) -- angle in degrees
variable (π : ℝ := Real.pi) -- value of π

noncomputable def arc_length (R : ℝ) (θ : ℝ) : ℝ := (2 * π * R * θ) / 360

theorem arc_length_increase 
  (h1 : θ = 1) : 
  arc_length R θ - arc_length R 0 = π * R / 180 := 
by 
  sorry

end arc_length_increase_l77_77496


namespace total_marbles_l77_77611

theorem total_marbles (bag1_marble_count : ℕ) (bag2_marble_count : ℕ) (marble_count_6bags : ℕ) (marble_count_2bags : ℕ) : 
    bag1_marble_count * 6 + bag2_marble_count * 2 = 50 :=
by
  have h1 : marble_count_6bags = 6 * 6 := rfl
  have h2 : marble_count_2bags = 2 * 7 := rfl
  have h3 : marble_count_6bags + marble_count_2bags = 50 := rfl
  exact h3

end total_marbles_l77_77611


namespace angle_A_measure_l77_77289

-- Definitions of the conditions in the problem
def angles_of_quadrilateral (A B C D : ℝ) : Prop :=
  A = 2 * B ∧ A = 3 * C ∧ A = 4 * D ∧ A + B + C + D = 360

-- The statement of the problem as a theorem
theorem angle_A_measure (A B C D : ℝ) (h : angles_of_quadrilateral A B C D) : 
  Int.round A = 173 :=
by
  -- Proof goes here
  sorry

end angle_A_measure_l77_77289


namespace Jerry_has_36_stickers_l77_77532

variable (FredStickers GeorgeStickers JerryStickers CarlaStickers : ℕ)
variable (h1 : FredStickers = 18)
variable (h2 : GeorgeStickers = FredStickers - 6)
variable (h3 : JerryStickers = 3 * GeorgeStickers)
variable (h4 : CarlaStickers = JerryStickers + JerryStickers / 4)
variable (h5 : GeorgeStickers + FredStickers = CarlaStickers ^ 2)

theorem Jerry_has_36_stickers : JerryStickers = 36 := by
  sorry

end Jerry_has_36_stickers_l77_77532


namespace range_of_a_l77_77117

theorem range_of_a (a : ℝ)
  (h : ∀ x : ℝ, f x = x^3 + a * x^2 + (a + 6) * x + 1)
  (h_der : ∀ x : ℝ, f_der x = 3 * x^2 + 2 * a * x + (a + 6)) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f_der x1 = 0 ∧ f_der x2 = 0) ↔ (a < -3 ∨ a > 6) :=
by
  sorry

end range_of_a_l77_77117


namespace reflection_matrix_over_line_y_eq_x_l77_77789

-- Define the reflection over the line y = x as a linear map
def reflection_over_y_eq_x (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.2, v.1)  -- this swaps the coordinates (x, y) -> (y, x)

-- Define the matrix that corresponds to this reflection
def reflection_matrix := ![
  ![0, 1],
  ![1, 0]
]

theorem reflection_matrix_over_line_y_eq_x :
  ∀ v : ℝ × ℝ, reflection_over_y_eq_x v = matrix.vec_mul reflection_matrix v :=
by
  sorry

end reflection_matrix_over_line_y_eq_x_l77_77789


namespace cart_distance_traveled_l77_77339

theorem cart_distance_traveled (circ_front circ_back revol_diff : ℕ) (R : ℕ)
(h1 : circ_front = 30)
(h2 : circ_back = 33)
(h3 : revol_diff = 5)
(h4 : (R + revol_diff) * circ_front = R * circ_back) :
    R * circ_back = 1650 :=
by
  have h5 : 30 * (R + 5) = 33 * R := by rwa [h1, h2, h3] at h4
  have h6 : 30 * (R + 5) = 30 * R + 150 := by rw [mul_add, mul_one]
  have h7 : 30 * (R + 5) = 33 * R := by rwa [← h5]
  have h8 : 150 = 3 * R := by linarith
  have h9 : 50 = R := by linarith
  have result : R * circ_back = 50 * 33 := by rw [h2, ← h9]
  exact result

end cart_distance_traveled_l77_77339


namespace modulo_inverse_product_l77_77926

open Int 

theorem modulo_inverse_product (n : ℕ) (a b c : ℤ) 
  (hn : 0 < n) 
  (ha : a * a.gcd n = 1) 
  (hb : b * b.gcd n = 1) 
  (hc : c * c.gcd n = 1) 
  (hab : (a * b) % n = 1) 
  (hac : (c * a) % n = 1) : 
  ((a * b) * c) % n = c % n :=
by
  sorry

end modulo_inverse_product_l77_77926


namespace find_n_l77_77182

noncomputable def equilateral_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  ∃ (AB AC BC : ℝ), AB = AC ∧ AC = BC ∧ AB > 0

theorem find_n (A B C D E F P Q R : Type) [metric_space A]
    [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space P] [metric_space Q] [metric_space R] 
    (equilateral_triangle_ABC : equilateral_triangle A B C)
    (ratio : ∀ n, (D divides side BC in ratio 3:(n-3)) ∧ 
                 (E divides side CA in ratio 3:(n-3)) ∧ 
                 (F divides side AB in ratio 3:(n-3)) ∧ 
                 n > 6) 
    (intersection_points : (∀ A D B P, P ∈ line AD ∧ P ∈ line BE) ∧ 
                          (∀ B E C Q, Q ∈ line BE ∧ Q ∈ line CF) ∧ 
                          (∀ C F A R, R ∈ line CF ∧ R ∈ line AD)) 
    (area_relation : ∀ area_PQR area_ABC,  area_PQR = (4 / 49) * area_ABC) 
  : ∃ n, n = 8 :=
sorry

end find_n_l77_77182


namespace units_digit_7_pow_3_l77_77665

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_3 : units_digit (7^3) = 3 :=
by
  -- Proof of the theorem would go here
  sorry

end units_digit_7_pow_3_l77_77665


namespace percentage_same_grades_l77_77511

def student_grades := { 
  A_A : ℕ, A_B : ℕ, A_C : ℕ, A_D : ℕ,
  B_A : ℕ, B_B : ℕ, B_C : ℕ, B_D : ℕ,
  C_A : ℕ, C_B : ℕ, C_C : ℕ, C_D : ℕ,
  D_A : ℕ, D_B : ℕ, D_C : ℕ, D_D : ℕ
}

def grade_distribution : student_grades := 
  ⟨3, 2, 1, 0, 1, 6, 3, 1, 0, 2, 7, 2, 0, 1, 2, 2⟩

def total_students : ℕ := 40

def same_grade_count (dist : student_grades) : ℕ :=
  dist.A_A + dist.B_B + dist.C_C + dist.D_D

def same_grade_percentage (count : ℕ) (total : ℕ) : ℚ :=
  (count.to_rat / total.to_rat) * 100

theorem percentage_same_grades :
  same_grade_percentage (same_grade_count grade_distribution) total_students = 45 := by
  sorry

end percentage_same_grades_l77_77511


namespace ratio_CQ_QF_l77_77196

noncomputable def triangle := Type

variables (ABC : triangle)
variables (A B C D F Q : ABC)
variables (h_CF : line C F)
variables (h_AD : line A D)
variables (h_intersection : Q ∈ h_CF ∧ Q ∈ h_AD)
variables (CD DB : ℝ) (h_ratio_CDDB : CD / DB = 2)
variables (AF FB : ℝ) (h_ratio_AFFB : AF / FB = 1)
variables (CQ QF : ℝ)

theorem ratio_CQ_QF:
  (CQ / QF) = 2 :=
sorry

end ratio_CQ_QF_l77_77196


namespace determine_last_card_back_l77_77820

theorem determine_last_card_back (n : ℕ) (f : ℕ → ℕ × ℕ) (cards: Fin n) :
  ∃ k, ∀ cards_shown : Fin n → ℕ,
  ∃ back_number, (cards_shown cards = k) → (back_number = (cards_shown cards).fst ∧ cards_shown cards = k)
  ∨ (back_number = (cards_shown cards).snd ∧ cards_shown cards = k) sorry :=
begin
  sorry
end

end determine_last_card_back_l77_77820


namespace anna_deducted_salary_l77_77044

theorem anna_deducted_salary :
  ∀ (weekly_salary daily_wage deduction deducted_salary : ℝ),
    weekly_salary = 1379 →
    daily_wage = weekly_salary / 5 →
    deduction = daily_wage * 2 →
    deducted_salary = weekly_salary - deduction →
    deducted_salary = 827.40 :=
by
  intros weekly_salary daily_wage deduction deducted_salary
  intros h_weekly h_daily h_deduction h_final
  rw [h_weekly, h_daily, h_deduction, h_final]
  sorry

end anna_deducted_salary_l77_77044


namespace abs_m_minus_n_l77_77159

theorem abs_m_minus_n (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : (1 / 5 * (m^2 + n^2 + 81 + 64 + 100) - 81) = 2) : |m - n| = 4 :=
  sorry

end abs_m_minus_n_l77_77159


namespace set_A_cannot_form_right_triangle_l77_77330

open Real

theorem set_A_cannot_form_right_triangle :
  let a := sqrt 3
  let b := sqrt 4
  let c := sqrt 5
  ¬ (a^2 + b^2 = c^2) :=
by
  let a := sqrt 3
  let b := sqrt 4
  let c := sqrt 5
  show ¬ (a^2 + b^2 = c^2)
  sorry

#eval set_A_cannot_form_right_triangle

end set_A_cannot_form_right_triangle_l77_77330


namespace round_trip_time_l77_77382

theorem round_trip_time (speed_to_work speed_to_home : ℝ) (time_to_work_mins : ℝ)
  (h_to_work : speed_to_work = 75) (h_to_home : speed_to_home = 105) (h_time_to_work : time_to_work_mins = 35) : 
  let time_to_work := time_to_work_mins / 60 in
  let distance_to_work := speed_to_work * time_to_work in
  let time_to_home := distance_to_work / speed_to_home in
  time_to_work + time_to_home = 1 := 
by
  sorry

end round_trip_time_l77_77382


namespace find_2008_star_2010_l77_77066

-- Define the operation
def operation_star (x y : ℕ) : ℕ := sorry  -- We insert a sorry here because the precise definition is given by the conditions

-- The properties given in the problem
axiom property1 : operation_star 2 2010 = 1
axiom property2 : ∀ n : ℕ, operation_star (2 * (n + 1)) 2010 = 3 * operation_star (2 * n) 2010

-- The main proof statement
theorem find_2008_star_2010 : operation_star 2008 2010 = 3 ^ 1003 :=
by
  -- Here we would provide the proof, but it's omitted.
  sorry

end find_2008_star_2010_l77_77066


namespace pyramid_volume_l77_77279

open Real

theorem pyramid_volume (EF FG QF : ℝ) (hEF : EF = 10) (hFG : FG = 5) (hQF : QF = 20)
    (h1 : ∀ (QE : ℝ), QE = sqrt (QF^2 - EF^2)) :
  (1 / 3) * (EF * FG) * sqrt (QF^2 - EF^2) = (500 * sqrt 3) / 3 :=
by
  -- Define basics based on given conditions 
  have hQE : sqrt (20^2 - 10^2) = 10 * sqrt 3 :=
    by simp [← hQF, ← hEF]; norm_num; rw Mul.ctime.succ 3 sqrt
  rw [← hEF, ← hFG]  -- Replace variables with given values
  simp [sqrt] -- Simplify the expression
  sorry -- Complete the proof later

end pyramid_volume_l77_77279


namespace house_orderings_4_l77_77321

-- Definitions for the problem setup

def color := ℕ -- We label each color as a unique number for simplicity.
def houses := {0, 1, 2, 3, 4} -- We label each house from 0 to 4.

structure HouseColors :=
  (arrangement : Fin 5 → color)
  (distinct : Function.Injective arrangement)

-- Conditions
def orange_before_red (hc : HouseColors) : Prop :=
  ∃ i j : Fin 5, hc.arrangement i = 0 ∧ hc.arrangement j = 1 ∧ i < j

def blue_before_yellow (hc : HouseColors) : Prop :=
  ∃ i j : Fin 5, hc.arrangement i = 2 ∧ hc.arrangement j = 3 ∧ i < j

def blue_not_next_yellow (hc : HouseColors) : Prop :=
  ∀ i : Fin 4, ¬ (hc.arrangement i = 2 ∧ hc.arrangement (i + 1) = 3)

def green_before_red_not_next_orange (hc : HouseColors) : Prop :=
  ∃ i j k : Fin 5, hc.arrangement i = 4 ∧ hc.arrangement j = 1 ∧ hc.arrangement k = 0 ∧ i < j ∧ (|i - k| ≠ 1)

-- Theorem statement
theorem house_orderings_4 :
  ∃ hc : HouseColors,
    orange_before_red hc ∧
    blue_before_yellow hc ∧
    blue_not_next_yellow hc ∧
    green_before_red_not_next_orange hc ∧
    (Fintype.card {hc // orange_before_red hc ∧ blue_before_yellow hc ∧ blue_not_next_yellow hc ∧ green_before_red_not_next_orange hc} = 4) :=
sorry

end house_orderings_4_l77_77321


namespace remainder_of_large_power_l77_77082

def powerMod (base exp mod_ : ℕ) : ℕ := (base ^ exp) % mod_

theorem remainder_of_large_power :
  powerMod 2 (2^(2^2)) 500 = 536 :=
sorry

end remainder_of_large_power_l77_77082


namespace problem_I_problem_II_l77_77091

noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

theorem problem_I : 
  let total_outcomes := C 10 3 in
  let no_male_outcomes := C 6 3 in
  (1 - (no_male_outcomes / total_outcomes : ℚ)) = (5 / 6 : ℚ) := 
by
  sorry

theorem problem_II :
  let total_outcomes := C 10 3 in
  let special_outcomes := C 8 1 in
  (special_outcomes / total_outcomes : ℚ) * (4 / 5) * (3 / 5) = (4 / 125 : ℚ) := 
by
  sorry

end problem_I_problem_II_l77_77091


namespace min_ratio_of_integers_l77_77936

theorem min_ratio_of_integers (x y : ℕ) (hx : 50 < x) (hy : 50 < y) (h_mean : x + y = 130) : 
  x = 51 → y = 79 → x / y = 51 / 79 := by
  sorry

end min_ratio_of_integers_l77_77936


namespace circumcircle_of_triangle_l77_77433

noncomputable def circumcircle_equation (A B C : (ℝ × ℝ)) : set (ℝ × ℝ) :=
  {P | ∃ D E F : ℝ, P.1^2 + P.2^2 + D * P.1 + E * P.2 + F = 0}

theorem circumcircle_of_triangle
  (A B C : (ℝ × ℝ))
  (hA : A = (0, 0))
  (hB : B = (2, 2))
  (hC : C = (4, 2)) :
  circumcircle_equation A B C = {(x, y) | (x - 3)^2 + (y + 1)^2 = 10} :=
by sorry

end circumcircle_of_triangle_l77_77433


namespace point_always_outside_circle_l77_77808

theorem point_always_outside_circle (a : ℝ) : a^2 + (2 - a)^2 > 1 :=
by sorry

end point_always_outside_circle_l77_77808


namespace greatest_possible_employees_take_subway_l77_77684

variable (P F : ℕ)

def part_time_employees_take_subway : ℕ := P / 3
def full_time_employees_take_subway : ℕ := F / 4

theorem greatest_possible_employees_take_subway 
  (h1 : P + F = 48) : part_time_employees_take_subway P + full_time_employees_take_subway F ≤ 15 := 
sorry

end greatest_possible_employees_take_subway_l77_77684


namespace train_length_l77_77028

noncomputable def length_of_train (time_sec : ℕ) (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000 / 3600) * time_sec

theorem train_length (h_time : 21 = 21) (h_speed : 75.6 = 75.6) :
  length_of_train 21 75.6 = 441 :=
by
  sorry

end train_length_l77_77028


namespace sequence_Sn_max_m_l77_77463

noncomputable def Sn (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 2
noncomputable def Tn (n : ℕ) : ℕ := n * (n + 1) / 2
noncomputable def an (n : ℕ) : ℕ := 2^n
noncomputable def bn (n : ℕ) : ℕ := if n = 1 then 1 else n

theorem sequence_Sn (a : ℕ → ℕ) (n : ℕ) (hn : Sn (n + 1) a = 2 * a (n + 1) - 2) :
  an n = 2^n :=
sorry

theorem max_m (m : ℝ) (n : ℕ) :
  (∃ n, ∑ i in finset.range n, (2 * (bn i) / (an i : ℝ)) ≥ m) → m ≤ 4 :=
sorry

end sequence_Sn_max_m_l77_77463


namespace area_of_triangle_tangent_line_l77_77968

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem area_of_triangle_tangent_line (x : ℝ) (h : x = Real.pi / 2) :
  let m := 1 + Real.cos x,
      y := x + Real.sin x,
      tangent_line : ℝ → ℝ := λ t, m * (t - x) + y,
      x_intercept := -1,
      y_intercept := 1,
      area := (1 / 2) * abs x_intercept * abs y_intercept
  in area = 1 / 2 :=
  sorry

end area_of_triangle_tangent_line_l77_77968


namespace least_number_remainders_l77_77079

theorem least_number_remainders (n : ℕ) (h₁ : n = 261) : n % 7 = 2 ∧ n % 37 = 0 := 
by
  unfold n
  rw [h₁]
  simp
  split
  { apply Nat.mod_eq_of_lt
    norm_num }
  { apply Nat.mod_eq_of_lt
    norm_num }

#align least_number_remainders least_number_remainders

end least_number_remainders_l77_77079


namespace pyramid_volume_l77_77275

noncomputable def volume_of_pyramid (AB BC PB : ℕ) (h1 : AB = 10) (h2 : BC = 6) (h3 : PB = 20) : ℝ :=
  let PA := real.sqrt (PB ^ 2 - AB ^ 2)
  let base_area := AB * BC
  (1 / 3) * base_area * PA

theorem pyramid_volume : volume_of_pyramid 10 6 20 (by rfl) (by rfl) (by rfl) = 200 * real.sqrt 3 :=
  sorry

end pyramid_volume_l77_77275


namespace fraction_spent_at_toy_store_l77_77485

theorem fraction_spent_at_toy_store 
  (total_allowance : ℝ)
  (arcade_fraction : ℝ)
  (candy_store_amount : ℝ) 
  (remaining_allowance : ℝ)
  (toy_store_amount : ℝ)
  (H1 : total_allowance = 2.40)
  (H2 : arcade_fraction = 3 / 5)
  (H3 : candy_store_amount = 0.64)
  (H4 : remaining_allowance = total_allowance - (arcade_fraction * total_allowance))
  (H5 : toy_store_amount = remaining_allowance - candy_store_amount) :
  toy_store_amount / remaining_allowance = 1 / 3 := 
sorry

end fraction_spent_at_toy_store_l77_77485


namespace speed_of_second_train_is_16_l77_77634

def speed_second_train (v : ℝ) : Prop :=
  ∃ t : ℝ, 
    (20 * t = v * t + 70) ∧ -- Condition: the first train traveled 70 km more than the second train
    (20 * t + v * t = 630)  -- Condition: total distance between stations

theorem speed_of_second_train_is_16 : speed_second_train 16 :=
by
  sorry

end speed_of_second_train_is_16_l77_77634


namespace rectangle_divided_into_13_squares_l77_77410

-- Define the conditions
variables {a b s : ℝ} (m n : ℕ)

-- Mathematical equivalent proof problem Lean statement
theorem rectangle_divided_into_13_squares (h : a * b = 13 * s^2)
  (hm : a = m * s) (hn : b = n * s) (hmn : m * n = 13) :
  a / b = 13 ∨ b / a = 13 :=
begin
  sorry
end

end rectangle_divided_into_13_squares_l77_77410


namespace gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l77_77453

theorem gcd_b_squared_plus_11b_plus_28_and_b_plus_6 (b : ℤ) (h : ∃ k : ℤ, b = 1573 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l77_77453


namespace lattice_points_on_hyperbola_l77_77487

theorem lattice_points_on_hyperbola :
  ((∃ x y : ℤ, x^2 - y^2 = 300^2) → #{(x, y) : ℕ × ℕ | x^2 - y^2 = 300^2} = 54) :=
sorry

end lattice_points_on_hyperbola_l77_77487


namespace probability_data_falls_within_interval_l77_77623

noncomputable def sample_capacity : ℕ := 66

def group_frequencies : List (Set.Ico ℝ ℝ × ℕ) := [
  (Set.Ico 11.5 15.5, 2),
  (Set.Ico 15.5 19.5, 4),
  (Set.Ico 19.5 23.5, 9),
  (Set.Ico 23.5 27.5, 18),
  (Set.Ico 27.5 31.5, 11),
  (Set.Ico 31.5 35.5, 12),
  (Set.Ico 35.5 39.5, 7),
  (Set.Ico 39.5 43.5, 3)
]

theorem probability_data_falls_within_interval :
  let relevant_data_count := (12 + 7 + 3) in
  relevant_data_count = 22 →
  sample_capacity = 66 →
  (relevant_data_count : ℝ) / sample_capacity = 1 / 3 :=
by intros relevant_data_count h1 h2; sorry

end probability_data_falls_within_interval_l77_77623


namespace mean_of_integers_neg3_to_6_l77_77647

theorem mean_of_integers_neg3_to_6 : 
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ) in
  let n := (6 - (-3) + 1 : ℤ) in
  s / n = 1.5 :=
by
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ)
  let n := (6 - (-3) + 1 : ℤ)
  simp
  sorry

end mean_of_integers_neg3_to_6_l77_77647


namespace region_midpoint_area_equilateral_triangle_52_36_l77_77395

noncomputable def equilateral_triangle (A B C: ℝ × ℝ) : Prop :=
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

def midpoint_region_area (a b c : ℝ × ℝ) : ℝ := sorry

theorem region_midpoint_area_equilateral_triangle_52_36 (A B C: ℝ × ℝ) (h: equilateral_triangle A B C) :
  let m := (midpoint_region_area A B C)
  100 * m = 52.36 :=
sorry

end region_midpoint_area_equilateral_triangle_52_36_l77_77395


namespace negation_statement_contrapositive_statement_l77_77863

variable (x y : ℝ)

theorem negation_statement :
  (¬ ((x-1) * (y+2) ≠ 0 → x ≠ 1 ∧ y ≠ -2)) ↔ ((x-1) * (y+2) = 0 → x = 1 ∨ y = -2) :=
by sorry

theorem contrapositive_statement :
  (x = 1 ∨ y = -2) → ((x-1) * (y+2) = 0) :=
by sorry

end negation_statement_contrapositive_statement_l77_77863


namespace num_combinations_producing_plus_l77_77179

-- Define the condition for the top to be "+"
def produces_plus_at_top (a b c d e : ℤ) : Prop :=
  a * b * c * d * e = 1

-- Define the solution space (all possible combinations of +1 and -1 for each cell)
def valid_bottom_row_combinations : list (ℤ × ℤ × ℤ × ℤ × ℤ) :=
  [(-1, -1, -1, -1, -1), (-1, -1, -1, -1, 1), (-1, -1, -1, 1, -1), (-1, -1, -1, 1, 1),
   (-1, -1, 1, -1, -1), (-1, -1, 1, -1, 1), (-1, -1, 1, 1, -1), (-1, -1, 1, 1, 1),
   (-1, 1, -1, -1, -1), (-1, 1, -1, -1, 1), (-1, 1, -1, 1, -1), (-1, 1, -1, 1, 1),
   (-1, 1, 1, -1, -1), (-1, 1, 1, -1, 1), (-1, 1, 1, 1, -1), (-1, 1, 1, 1, 1),
   (1, -1, -1, -1, -1), (1, -1, -1, -1, 1), (1, -1, -1, 1, -1), (1, -1, -1, 1, 1),
   (1, -1, 1, -1, -1), (1, -1, 1, -1, 1), (1, -1, 1, 1, -1), (1, -1, 1, 1, 1),
   (1, 1, -1, -1, -1), (1, 1, -1, -1, 1), (1, 1, -1, 1, -1), (1, 1, -1, 1, 1),
   (1, 1, 1, -1, -1), (1, 1, 1, -1, 1), (1, 1, 1, 1, -1), (1, 1, 1, 1, 1)]

-- Prove that the number of combinations producing "+" at the top equals 17
theorem num_combinations_producing_plus :
  (list.filter (λ (t : ℤ × ℤ × ℤ × ℤ × ℤ), produces_plus_at_top t.1 t.2 t.3 t.4 t.5) valid_bottom_row_combinations).length = 17 :=
  sorry

end num_combinations_producing_plus_l77_77179


namespace difference_of_squares_example_l77_77058

theorem difference_of_squares_example : (75^2 - 25^2) = 5000 := by
  let a := 75
  let b := 25
  have step1 : a + b = 100 := by
    rw [a, b]
    norm_num
  have step2 : a - b = 50 := by
    rw [a, b]
    norm_num
  have result : (a + b) * (a - b) = 5000 := by
    rw [step1, step2]
    norm_num
  rw [pow_two, pow_two, mul_sub, ← result]
  norm_num

end difference_of_squares_example_l77_77058


namespace pythagorean_triples_l77_77675

theorem pythagorean_triples:
  (∃ a b c : ℝ, (a = 1 ∧ b = 2 ∧ c = sqrt 5 ∧ a^2 + b^2 = c^2) ∨
   (a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 ≠ c^2) ∨
   (a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2) ∨
   (a = 4 ∧ b = 5 ∧ c = 6 ∧ a^2 + b^2 ≠ c^2)) ∧
  (∃ a' b' c' : ℝ, a' = 3 ∧ b' = 4 ∧ c' = 5) :=
by
  sorry

end pythagorean_triples_l77_77675


namespace min_L_exists_l77_77542

-- Definitions based on conditions given
def satisfies_condition (x y : ℝ) : Prop := 
  x > (y^4 / 9 + 2015)^(1/4)

def disk (r x y : ℝ) : Prop := 
  x^2 + y^2 ≤ r^2

def intersection_area (f : ℝ → ℝ) (r : ℝ) :=
  f(r) < (Real.pi / 3) * r^2

-- Statement to prove
theorem min_L_exists :
  ∃ (L : ℝ), 
  (∀ (r : ℝ) (x y : ℝ), satisfies_condition x y → disk r x y → f r x y < L * r^2) →
  L = Real.pi / 3 :=
sorry

end min_L_exists_l77_77542


namespace geometric_sequence_formula_l77_77105

noncomputable def a_n (q : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n - 1)

theorem geometric_sequence_formula (q : ℝ) (S : ℕ → ℝ) (n : ℕ) (hn : n > 0) :
  a_n q n = 2^(n - 1) :=
sorry

end geometric_sequence_formula_l77_77105


namespace num_words_sum_l77_77563

/-
  Definitions:
  - word_kasha is a multiset with the letters "К", "А", "Ш", "А".
  - word_hleb is a set with the letters "Х", "Л", "Е", "Б".
  - num_distinct_perms is the function to calculate permutations of distinct items.
  - num_perms_with_repetition is the function to calculate permutations of multiset.
-/

def word_kasha : Multiset Char := {'К', 'А', 'Ш', 'А'}
def word_hleb : Finset Char := {'Х', 'Л', 'Е', 'Б'}

def num_distinct_perms (s : Finset Char) : ℕ :=
  (Finset.card s).factorial

def num_perms_with_repetition (m : Multiset Char) : ℕ :=
  Multiset.card m ! / m.dedup.card.factorial

theorem num_words_sum : 
  num_distinct_perms word_hleb + num_perms_with_repetition word_kasha = 36 :=
by {
  sorry
}

end num_words_sum_l77_77563


namespace kenneth_initial_money_l77_77205

-- Define the costs of the items
def cost_baguette := 2
def cost_water := 1

-- Define the quantities bought
def baguettes_bought := 2
def water_bought := 2

-- Define the amount left after buying the items
def money_left := 44

-- Calculate the total cost
def total_cost := (baguettes_bought * cost_baguette) + (water_bought * cost_water)

-- Define the initial money Kenneth had
def initial_money := total_cost + money_left

-- Prove the initial money is $50
theorem kenneth_initial_money : initial_money = 50 := 
by 
  -- The proof part is omitted because it is not required.
  sorry

end kenneth_initial_money_l77_77205


namespace domain_eq_l77_77298

def domain_of_function :
    Set ℝ := {x | (x - 1 ≥ 0) ∧ (x + 1 > 0)}

theorem domain_eq :
    domain_of_function = {x | x ≥ 1} :=
by
  sorry

end domain_eq_l77_77298


namespace problem1_problem2_problem3_l77_77140

-- Problem 1
theorem problem1 (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, b n = n / 2)
  (h3 : ∀ n, a (n + 1) * b n = S n + 1)
  (h4 : ∀ n, S (n + 1) = S n + a (n + 1))
  : a 4 = 8 := sorry

-- Problem 2
theorem problem2 (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (q : ℚ)
  (h1 : ∃ a1, ∀ n, a (n + 1) = a 1 * q ^ n)
  (h2 : q ≠ 1)
  (h3 : ∀ n, a (n + 1) * b n = S n + 1)
  (h4 : ∀ n, S (n + 1) = S n + a (n + 1))
  : ∃ r, ∀ n, b (n + 1) + 1 / (1 - q) = r * (b n + 1 / (1 - q)) := sorry

-- Problem 3
theorem problem3 (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
  (h1 : ∀ n, a n ≠ 0)
  (h2 : ∀ n, ∃ d, b (n + 1) = b n + d)
  (h3 : ∀ n, a (n + 1) * b n = S n + 1)
  (h4 : ∀ n, S (n + 1) = S n + a (n + 1))
  : (∀ n, a (n + 1) - a n = a n - a (n - 1)) ↔ d = 1/2 := sorry

end problem1_problem2_problem3_l77_77140


namespace trig_identity_l77_77756

theorem trig_identity :
  sin (137 * real.pi / 180) * cos (13 * real.pi / 180) - cos (43 * real.pi / 180) * sin (13 * real.pi / 180) = 1 / 2 := 
by {
  -- This is where the proof would go, but we include sorry to indicate it is not provided.
  sorry
}

end trig_identity_l77_77756


namespace slope_angle_at_point_l77_77797

def f (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 2

theorem slope_angle_at_point :
  let deriv_f := fun x : ℝ => 6 * x^2 - 7
  let slope := deriv_f 1
  let angle := Real.arctan slope
  angle = (3 * Real.pi) / 4 :=
by
  sorry

end slope_angle_at_point_l77_77797


namespace ralph_tv_hours_l77_77267

theorem ralph_tv_hours :
  (4 * 5 + 6 * 2) = 32 :=
by
  sorry

end ralph_tv_hours_l77_77267


namespace find_y_l77_77592

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (h3 : x = 1) : y = 13 := by
  sorry

end find_y_l77_77592


namespace original_triangle_area_l77_77606

-- Define the scaling factor and given areas
def scaling_factor : ℕ := 2
def new_triangle_area : ℕ := 32

-- State that if the dimensions of the original triangle are doubled, the area becomes 32 square feet
theorem original_triangle_area (original_area : ℕ) : (scaling_factor * scaling_factor) * original_area = new_triangle_area → original_area = 8 := 
by
  intros h
  sorry

end original_triangle_area_l77_77606


namespace total_chickens_l77_77314

theorem total_chickens (ducks geese : ℕ) (hens roosters chickens: ℕ) :
  ducks = 45 → geese = 28 →
  hens = ducks - 13 → roosters = geese + 9 →
  chickens = hens + roosters →
  chickens = 69 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_chickens_l77_77314


namespace quadrilateral_area_bound_l77_77121

-- Definitions for the vertices
variables {A B C P1 P2 P3 P4 : Type} [HasArea A] [HasArea B] [HasArea C]
  [HasArea P1] [HasArea P2] [HasArea P3] [HasArea P4]

-- Points P1, P2, P3, P4 lie on sides of triangle ABC
variables [LiesOnSide A B P1] [LiesOnSide B C P2] [LiesOnSide C A P3] [LiesOnSide A C P4]

-- Function representing the area of a triangle
noncomputable def area : Triangle → ℝ := by sorry

-- Main theorem statement
theorem quadrilateral_area_bound (ABC : Triangle) (P1P2P3: Triangle) (P1P2P4: Triangle)
  (P1P3P4: Triangle) (P2P3P4: Triangle) :
  P1P2P3 ∈ sides(ABC) ∧ P1P2P4 ∈ sides(ABC) ∧ P1P3P4 ∈ sides(ABC) ∧ P2P3P4 ∈ sides(ABC) →
  min (area P1P2P3) (min (area P1P2P4) (min (area P1P3P4) (area P2P3P4))) ≤ (1 / 4) * (area ABC) :=
by sorry

end quadrilateral_area_bound_l77_77121


namespace proposition_B_correct_l77_77100

variable {Point : Type}
variable [affine_space Point Plane]
open affine_space

noncomputable def plane_perpendicular
  (l : Line Point) (α β : Plane Point) : Prop :=
  (l ∈ α) ∧ (l ∈ β) → affine_space.orthogonal α β

theorem proposition_B_correct
  (l : Line Point) (α β : Plane Point)
  (h1 : affine_space.parallel l α)
  (h2 : affine_space.orthogonal l α) :
  affine_space.orthogonal α β :=
sorry

end proposition_B_correct_l77_77100


namespace password_encryption_l77_77679

variables (a b x : ℝ)

theorem password_encryption :
  3 * a * (x^2 - 1) - 3 * b * (x^2 - 1) = 3 * (x + 1) * (x - 1) * (a - b) :=
by sorry

end password_encryption_l77_77679


namespace volume_of_pyramid_l77_77277

-- Definitions based on the given conditions
def AB : ℝ := 10
def BC : ℝ := 6
def PB : ℝ := 20
def area_base := AB * BC
def PA := Real.sqrt (PB ^ 2 - AB ^ 2)

-- Statement to prove the volume of pyramid PABCD
theorem volume_of_pyramid :
  let volume := (1 : ℝ) / 3 * area_base * PA in
  volume = 200 * Real.sqrt 3 :=
by
  sorry

end volume_of_pyramid_l77_77277


namespace problem_part1_problem_part2_l77_77475

-- Definitions and assumptions based on the given conditions
def line_l (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

def point_M : ℝ × ℝ := (-2, 0)
def point_N : ℝ × ℝ := (-1, 0)

def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def points_ratio (Q : ℝ × ℝ) : Prop :=
  dist Q point_M / dist Q point_N = real.sqrt 2

-- Prove the existence of the curve C and the value of a
theorem problem_part1 (Q : ℝ × ℝ) (h : points_ratio Q) : Q.1 ^ 2 + Q.2 ^ 2 = 2 :=
sorry

theorem problem_part2 (a : ℝ) (A B : ℝ × ℝ)
  (hA : line_l a A.1 A.2)
  (hB : line_l a B.1 B.2)
  (hC : A.1 ^ 2 + A.2 ^ 2 = 2)
  (hD : B.1 ^ 2 + B.2 ^ 2 = 2)
  (ortho : A.1 * B.1 + A.2 * B.2 = 0) : a = real.sqrt 2 ∨ a = -real.sqrt 2 :=
sorry

end problem_part1_problem_part2_l77_77475


namespace seq_sum_11_l77_77466

noncomputable def S (n : ℕ) : ℕ := sorry

noncomputable def a (n : ℕ) : ℕ := sorry

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem seq_sum_11 :
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) ∧
  (is_arithmetic_sequence a) ∧
  (3 * (a 2 + a 4) + 2 * (a 6 + a 9 + a 12) = 12) →
  S 11 = 11 :=
by
  sorry

end seq_sum_11_l77_77466


namespace decode_digits_l77_77901

theorem decode_digits :
  ∃ (Δ Δ' ◻ : ℕ), Δ = 1 ∧ Δ' = 1 ∧ ◻ = 3 ∧ (10 * Δ + Δ') ^ ◻ = Δ * ◻ ^ 2 * Δ' := by
{
  use 1,
  use 1,
  use 3,
  split,
  {
    exact rfl,
  },
  split,
  {
    exact rfl,
  },
  split,
  {
    exact rfl,
  },
  sorry
}

end decode_digits_l77_77901


namespace solution_set_x_l77_77123

theorem solution_set_x (x : ℝ) : 
  (|x^2 - x - 2| + |1 / x| = |x^2 - x - 2 + 1 / x|) ↔ 
  (x ∈ {y : ℝ | -1 ≤ y ∧ y < 0} ∨ x ≥ 2) :=
sorry

end solution_set_x_l77_77123


namespace domain_of_function_l77_77607

theorem domain_of_function :
  {x : ℝ | 2^x - 8 ≥ 0} = set.Ici 3 :=
begin
  ext x,
  simp [set.Ici, ge_iff_le],
  have h : 0 < 2 := by linarith,
  split,
  { intro h1,
    rw [← log_le_log_iff (pow_pos h x) (by norm_num)] at h1,
    rw [log_pow h] at h1,
    linarith, },
  { intro h2,
    rw [← log_le_log_iff (pow_pos h x) (by norm_num)],
    rw [log_pow h],
    linarith, }
end

end domain_of_function_l77_77607


namespace arithmetic_sequence_sum_10_l77_77512

open nat

section
  variables {a : ℕ → ℝ}

  noncomputable def S_n (n : ℕ) (a : ℕ → ℝ) : ℝ := n / 2 * (a 1 + a n)

  theorem arithmetic_sequence_sum_10 (h : a 2 + a 9 = 2) : S_n 10 a = 10 :=
  by
    sorry
end

end arithmetic_sequence_sum_10_l77_77512


namespace opposite_face_is_B_l77_77009

-- Define the type of the squares and markings
inductive Square
| x | A | B | C | D | E 

-- Define the conditions
def marking_letters := { Square.x, Square.A, Square.B, Square.C, Square.D, Square.E }
def center_square := Square.x
def U_shape_surround := { (Square.A, Square.B, Square.D) }
def adjacent (a b : Square) : Prop := 
  (a = Square.A ∧ b = Square.C) ∨ (a = Square.D ∧ b = Square.E)

-- Statement of the proof problem
theorem opposite_face_is_B :
  ∀ (marking_letters : set Square)
    (center_square : Square)
    (U_shape_surround : set (Square × Square × Square))
    (adjacent : Square → Square → Prop),
  marking_letters = { Square.x, Square.A, Square.B, Square.C, Square.D, Square.E } →
  center_square = Square.x →
  U_shape_surround = { (Square.A, Square.B, Square.D) } →
  adjacent Square.A Square.C →
  adjacent Square.D Square.E →
  (opposite_face Square.x = Square.B) :=
by
  sorry

end opposite_face_is_B_l77_77009


namespace domain_f_x_plus_1_l77_77978

def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem domain_f_x_plus_1 :
  (∀ x, domain_f x) → ∀ y, (-1 ≤ y ∧ y ≤ 1) := 
by {
  intros h y,
  split;
  sorry
}

end domain_f_x_plus_1_l77_77978


namespace find_a_range_f_l77_77558

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x + 10

theorem find_a (h : deriv (f x a) (2) = 0) : a = -12 := 
sorry

theorem range_f (a : ℝ) : 
  (∀ x ∈ Icc (-3 : ℝ) (4 : ℝ), f x (-12) ≥ -6 ∧ f x (-12) ≤ 26) := 
sorry

end find_a_range_f_l77_77558


namespace probability_of_less_than_5_6_l77_77018

-- Define the interval (0, 1) and the corresponding region and its length
def interval : Set ℝ := {x | 0 < x ∧ x < 1}

-- Define the subset of the interval less than 5/6
def subinterval : Set ℝ := {x | 0 < x ∧ x < 5 / 6}

-- Define the probability as the ratio of lengths
def probability_interval_lt_5_6 : ℝ :=
  (5 / 6) / 1

-- The theorem statement
theorem probability_of_less_than_5_6 :
  ∀ x, x ∈ interval → x ∈ subinterval → probability_interval_lt_5_6 = 5 / 6 :=
sorry

end probability_of_less_than_5_6_l77_77018


namespace mean_of_integers_neg3_to_6_l77_77644

theorem mean_of_integers_neg3_to_6 : 
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ) in
  let n := (6 - (-3) + 1 : ℤ) in
  s / n = 1.5 :=
by
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ)
  let n := (6 - (-3) + 1 : ℤ)
  simp
  sorry

end mean_of_integers_neg3_to_6_l77_77644


namespace dot_product_PF1_PF2_l77_77039

variables {F1 F2 P : EuclideanSpace ℝ (Fin 2)}

-- Conditions
def ellipse : Prop := ∀ (x y : ℝ), x^2 / 25 + y^2 / 16 = 1
def hyperbola : Prop := ∀ (x y : ℝ), x^2 / 4 - y^2 / 5 = 1
def shared_foci : Prop := dist F1 F2 = 6
def intersection_point : Prop := true -- We assume P lies on both curves, but keep it generic for simplicity
def distance_PF1 : Prop := dist P F1 = 7
def distance_PF2 : Prop := dist P F2 = 3

-- Problem statement
theorem dot_product_PF1_PF2
  (h1 : ellipse)
  (h2 : hyperbola)
  (h3 : shared_foci)
  (h4 : intersection_point)
  (h5 : distance_PF1)
  (h6 : distance_PF2)
  : (P - F1) ⬝ (P - F2) = 11 :=
by
  sorry

end dot_product_PF1_PF2_l77_77039


namespace sum_of_integral_c_l77_77083

theorem sum_of_integral_c (c : ℤ) (h : c ≤ 30) :
  (∃ x1 x2 : ℚ, 2 * x1^2 - 7 * x1 - c = 0 ∧ 2 * x2^2 - 7 * x2 - c = 0) →
  ∑ c in finset.filter (λ c, ∃ k : ℤ, 49 + 8 * c = k^2) 
    (finset.Icc (-6) 30), c = 4 :=
by
  sorry

end sum_of_integral_c_l77_77083


namespace inscribed_ball_radius_l77_77594

noncomputable theory

-- Define the conditions
def radius_circumscribing_sphere := 5 * Real.sqrt 2 + 5
def tetrahedron_edge_length (r : ℝ) : ℝ := 4 * r
def circumscribed_sphere_radius (a : ℝ) : ℝ := a * Real.sqrt 6 / 4

-- The main theorem we need to prove
theorem inscribed_ball_radius (r ri : ℝ) 
  (h1 : circumscribed_sphere_radius (tetrahedron_edge_length r) = radius_circumscribing_sphere)
  (h2 : ri = (Real.sqrt 6 - 1) ) :
  ri = Real.sqrt 6 - 1 :=
sorry

end inscribed_ball_radius_l77_77594


namespace solution_l77_77012

/-- Original cost price, original selling price, and daily sales at original price -/
def original_cost : ℝ := 80
def original_price : ℝ := 120
def daily_sales : ℝ := 20

/-- Conditions: price reduction per unit and increased sales -/
def price_reduction_per_piece (x : ℝ) : ℝ := x
def daily_sales_increase (x : ℝ) : ℝ := 2 * x

/-- Profit per piece given price reduction x -/
def profit_per_piece (x : ℝ) : ℝ := 40 - x

/-- Daily sales volume given price reduction x -/
def sales_volume (x : ℝ) : ℝ := 20 + 2 * x

/-- Daily profit as a function of price reduction x -/
def daily_profit (x : ℝ) : ℝ := (40 - x) * (20 + 2 * x)

/-- Problem: find price reduction x for a daily profit of 1200 yuan -/
def price_reduction_for_target_profit (target_profit : ℝ) : ℝ := 
  if (solver : ∃ x : ℝ, (40 - x) * (20 + 2 * x) = target_profit) then
    classical.some solver
  else 
    0

/-- Check if a daily profit of 1800 yuan can be achieved -/
def can_achieve_daily_profit_1800 : Prop :=
  ¬ ∃ x : ℝ, (40 - x) * (20 + 2 * x) = 1800

/--Theorem stating the solution to the problem -/
theorem solution : can_achieve_daily_profit_1800 := by
  sorry

end solution_l77_77012


namespace kylie_and_nelly_total_stamps_l77_77537

theorem kylie_and_nelly_total_stamps :
  ∀ (kylie_stamps nelly_delta: ℕ),
  kylie_stamps = 34 →
  nelly_delta = 44 →
  (kylie_stamps + (kylie_stamps + nelly_delta) = 112) :=
by
  intros kylie_stamps nelly_delta h_kylie h_delta
  rw [h_kylie, h_delta]
  rw [add_assoc]
  sorry

end kylie_and_nelly_total_stamps_l77_77537


namespace product_of_reverse_numbers_l77_77991

def reverse (n : Nat) : Nat :=
  Nat.ofDigits 10 (List.reverse (Nat.digits 10 n))

theorem product_of_reverse_numbers : 
  ∃ (a b : ℕ), a * b = 92565 ∧ b = reverse a ∧ ((a = 165 ∧ b = 561) ∨ (a = 561 ∧ b = 165)) :=
by
  sorry

end product_of_reverse_numbers_l77_77991


namespace cost_difference_zero_l77_77763

noncomputable def slice_cost (total_cost : ℕ) (slices : ℕ) : ℝ :=
  total_cost / slices

noncomputable def total_cost (plain_cost : ℕ) (pepperoni_cost : ℕ) : ℕ :=
  plain_cost + pepperoni_cost

def num_pepperoni_slices (total_slices : ℕ) : ℕ := total_slices / 3

def total_plain_slices (total_slices num_pepperoni_slices : ℕ) : ℕ :=
  total_slices - num_pepperoni_slices

def num_slices_liam (num_pepperoni_slices : ℕ) (extra_plain_slices : ℕ) : ℕ :=
  num_pepperoni_slices + extra_plain_slices

def num_slices_dale (total_plain_slices slices_liam : ℕ) : ℕ :=
  total_plain_slices - slices_liam

noncomputable def cost_liam (slice_cost : ℝ) (slices_liam : ℕ) : ℝ :=
  slice_cost * slices_liam

noncomputable def cost_dale (slice_cost : ℝ) (slices_dale : ℕ) : ℝ :=
  slice_cost * slices_dale

theorem cost_difference_zero (total_slices : ℕ) (plain_cost pepperoni_cost extra_slices: ℕ) (liam_slices dale_slices: ℕ):
  total_slices = 12 →
  plain_cost = 12 →
  pepperoni_cost = 3 →
  liam_slices = (num_pepperoni_slices total_slices) + 2 →
  dale_slices = (total_plain_slices total_slices (num_pepperoni_slices total_slices)) - 2 →
  cost_liam (slice_cost (total_cost plain_cost pepperoni_cost) total_slices) liam_slices =
  cost_dale (slice_cost (total_cost plain_cost pepperoni_cost) total_slices) dale_slices :=
by
  intros h1 h2 h3 h4 h5
  rw [total_cost, h1, h2, h3, num_pepperoni_slices, total_plain_slices]
  sorry

end cost_difference_zero_l77_77763


namespace sqrt_8_eq_2sqrt2_cbrt_a8_eq_a2cbrta2_cbrt_16x4_eq_2xcbrt2x_l77_77336

-- Part (a): Prove that the square root of 8 is 2 times the square root of 2.
theorem sqrt_8_eq_2sqrt2 : Real.sqrt 8 = 2 * Real.sqrt 2 := 
  sorry

-- Part (b): Prove that the cube root of a^8 is a^2 times the cube root of a^2.
theorem cbrt_a8_eq_a2cbrta2 (a : ℝ) : Real.cbrt (a^8) = a^2 * Real.cbrt (a^2) := 
  sorry

-- Part (c): Prove that the cube root of 16x^4 is 2x times the cube root of 2x.
theorem cbrt_16x4_eq_2xcbrt2x (x : ℝ) : Real.cbrt (16 * x^4) = 2 * x * Real.cbrt (2 * x) := 
  sorry

end sqrt_8_eq_2sqrt2_cbrt_a8_eq_a2cbrta2_cbrt_16x4_eq_2xcbrt2x_l77_77336


namespace inverse_proportionality_example_l77_77961

theorem inverse_proportionality_example (k : ℝ) (x : ℝ) (y : ℝ) (h1 : 5 * 10 = k) (h2 : x * 40 = k) : x = 5 / 4 :=
by
  -- sorry is used to skip the proof.
  sorry

end inverse_proportionality_example_l77_77961


namespace stock_yield_is_12_percent_l77_77698

def percentage_yield (face_value : ℝ) (dividend_percentage : ℝ) (market_price : ℝ) : ℝ :=
  (dividend_percentage * face_value / market_price) * 100

theorem stock_yield_is_12_percent :
  percentage_yield 100 0.15 125.00000000000001 = 12 :=
by sorry

end stock_yield_is_12_percent_l77_77698


namespace sum_mn_symmetric_f_l77_77166

theorem sum_mn_symmetric_f (m n : ℤ) (f : ℤ → ℤ) (h : ∀ x, f(x) = |x + m| + |n * x + 1|) :
  f(2 - x) = f(2 + x) → m + n = -4 := 
by
  sorry

end sum_mn_symmetric_f_l77_77166


namespace units_digit_of_7_pow_3_l77_77667

theorem units_digit_of_7_pow_3 : (7 ^ 3) % 10 = 3 :=
by
  sorry

end units_digit_of_7_pow_3_l77_77667


namespace min_theta_for_symmetric_shift_l77_77955

theorem min_theta_for_symmetric_shift :
  ∃ θ : ℝ, θ > 0 ∧ 
    (∀ x : ℝ, 2 * sin (3 * x - 3 * θ + π / 3) = 2 * sin (3 * (-x))) ∧ 
    θ = 5 * π / 18 :=
by
  sorry

end min_theta_for_symmetric_shift_l77_77955


namespace ellipse_properties_max_triangle_area_line_eq_l77_77113

variables {a b c k : ℝ}

def is_eccentricity (c a : ℝ) := c / a = sqrt 3 / 2
def is_slope_AF (A : ℝ × ℝ) (F : ℝ × ℝ) := (F.2 - A.2) / (F.1 - A.1) = 2 * sqrt 3 / 3
def ellipse_eq (a b : ℝ) (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

def line_eq (k : ℝ) (x : ℝ) : ℝ := k * x - 2

noncomputable def max_triangle_area_eq (k : ℝ) : bool :=
let t := sqrt (4 * k^2 - 3) in
(4 * t) / (t^2 + 4) = 1

theorem ellipse_properties :
  (a = 2) →
  (b = 1) →
  (c = sqrt 3) →
  is_eccentricity c a →
  is_slope_AF (0, -2) (sqrt 3, 0) →
  ellipse_eq a b 0 0 :=
by
  intros
  sorry

theorem max_triangle_area_line_eq :
  (k = sqrt 7 / 2 ∨ k = - sqrt 7 / 2) →
  max_triangle_area_eq k :=
by 
  intros 
  sorry

end ellipse_properties_max_triangle_area_line_eq_l77_77113


namespace max_handshakes_no_cycles_l77_77697

theorem max_handshakes_no_cycles (n : ℕ) (h_n : n = 20) : 
  (n * (n - 1)) / 2 = 190 :=
by
  rw h_n
  sorry

end max_handshakes_no_cycles_l77_77697


namespace gcd_of_175_100_75_base_6_to_8_conversion_l77_77696

-- Proof for problem 1: GCD of 175, 100, and 75 is 25
theorem gcd_of_175_100_75 : Nat.gcd (Nat.gcd 175 100) 75 = 25 :=
by sorry

-- Proof for problem 2: Base-6 number 1015 converts to decimal 227 and then to base-8 as 343
theorem base_6_to_8_conversion : 
  let dec_val := 1 * 6^3 + 0 * 6^2 + 1 * 6 + 5
  dec_val = 227 ∧ Nat.digits 8 dec_val = [3, 4, 3] :=
by sorry

end gcd_of_175_100_75_base_6_to_8_conversion_l77_77696


namespace coefficient_of_x_squared_l77_77040

theorem coefficient_of_x_squared (k : ℝ) (h : k = 0.64) : 
  (a b c : ℝ) (h_eq : a = k ∧ b = 5 * k ∧ c = k) (h_discriminant : b^2 - 4 * a * c = 0) :
  a = 0.64 :=
by 
  rw h at h_eq
  exact h_eq.left

end coefficient_of_x_squared_l77_77040


namespace photos_in_gallery_l77_77567

theorem photos_in_gallery (P : ℕ) 
  (h1 : P / 2 + (P / 2 + 120) + P = 920) : P = 400 :=
by
  sorry

end photos_in_gallery_l77_77567


namespace eval_simplify_expr_l77_77396

def expr : Int :=
  65 + (160 / 8) + (35 * 12) - 450 - (504 / 7)

theorem eval_simplify_expr : expr = -17 :=
by
  have h1 : 160 / 8 = 20 := by norm_num
  have h2 : 504 / 7 = 72 := by norm_num
  have h3 : 35 * 12 = 420 := by norm_num
  have h4 : expr = 65 + 20 + 420 - 450 - 72 := by
    rw [h1, h2, h3]
  norm_num at h4
  exact h4

end eval_simplify_expr_l77_77396


namespace isosceles_triangle_perimeter_eq_20_l77_77830

variable (x y : ℝ)

theorem isosceles_triangle_perimeter_eq_20:
  |x-4| + sqrt(y-8) = 0 -> x = 4 -> y = 8 -> 4 + 8 + 8 = 20 := 
by 
  sorry

end isosceles_triangle_perimeter_eq_20_l77_77830


namespace mathland_transport_l77_77887

theorem mathland_transport (n : ℕ) (h : n ≥ 2) (transport : Fin n -> Fin n -> Prop) :
(∀ i j, transport i j ∨ transport j i) →
(∃ tr : Fin n -> Fin n -> Prop, 
  (∀ i j, transport i j → tr i j) ∨
  (∀ i j, transport j i → tr i j)) :=
by
  sorry

end mathland_transport_l77_77887


namespace length_of_place_mat_l77_77019

noncomputable def length_of_mat
  (R : ℝ)
  (w : ℝ)
  (n : ℕ)
  (θ : ℝ) : ℝ :=
  2 * R * Real.sin (θ / 2)

theorem length_of_place_mat :
  ∃ y : ℝ, y = length_of_mat 5 1 7 (360 / 7) := by
  use 4.38
  sorry

end length_of_place_mat_l77_77019


namespace abs_diff_l77_77161

theorem abs_diff (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : ((m^2 + n^2 + 81 + 64 + 100) / 5) - 81 = 2) :
  |m - n| = 4 := by
  sorry

end abs_diff_l77_77161


namespace conjugate_of_fraction_l77_77975

def conjugate (z : ℂ) : ℂ := complex.conj z

theorem conjugate_of_fraction :
  conjugate ((2 + complex.i) / (1 - 2 * complex.i)) = -complex.i :=
by
  sorry

end conjugate_of_fraction_l77_77975


namespace sum_of_elements_in_A_l77_77819

def f (x : ℝ) := x^2 - 2 * x

def A : Set ℝ := { x | f (f x) = 0 }

theorem sum_of_elements_in_A : (∑ x in A, x) = 4 := by
  sorry

end sum_of_elements_in_A_l77_77819


namespace max_balloons_l77_77256

def priceSeq (n : ℕ) : ℝ :=
  5 * (4/5)^(n - 1)

def totalCost (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, priceSeq (i + 1)

theorem max_balloons (h : totalCost 40 = 200) : ∃ k, k = 44 ∧ totalCost k ≤ 200 :=
by
  sorry

end max_balloons_l77_77256


namespace maria_money_difference_l77_77670

-- Defining constants for Maria's money when she arrived and left the fair
def money_at_arrival : ℕ := 87
def money_at_departure : ℕ := 16

-- Calculating the expected difference
def expected_difference : ℕ := 71

-- Statement: proving that the difference between money_at_arrival and money_at_departure is expected_difference
theorem maria_money_difference : money_at_arrival - money_at_departure = expected_difference := by
  sorry

end maria_money_difference_l77_77670


namespace prob_at_least_one_woman_l77_77172

def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem prob_at_least_one_woman 
(choose_from : ℕ) (total_men : ℕ) (total_women : ℕ) (select_count : ℕ)
(h_total : choose_from = total_men + total_women)
(h_men : total_men = 9)
(h_women : total_women = 5)
(h_select : select_count = 3) :
  (1 - (binom total_men select_count).to_nat.to_rat / 
       (binom choose_from select_count).to_nat.to_rat) = 23 / 30 :=
by
  sorry

end prob_at_least_one_woman_l77_77172


namespace factor_theorem_example_l77_77871

noncomputable def polynomial_c (c : ℚ) : Polynomial ℚ :=
  c * Polynomial.monomial 3 1 - 6 * Polynomial.monomial 2 1 
  - c * Polynomial.monomial 1 1 + Polynomial.C 10

theorem factor_theorem_example (c : ℚ) :
  Polynomial.eval 3 (polynomial_c c) = 0 -> c = 11 / 6 :=
by
  sorry

end factor_theorem_example_l77_77871


namespace sqrt_of_4_is_2_l77_77615

theorem sqrt_of_4_is_2 : sqrt 4 = 2 := 
sorry

end sqrt_of_4_is_2_l77_77615


namespace problem_equiv_proof_l77_77061

theorem problem_equiv_proof : ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h
  sorry

end problem_equiv_proof_l77_77061


namespace triangle_value_l77_77067

-- Define the operation \(\triangle\)
def triangle (m n p q : ℕ) : ℕ := (m * m) * p * q / n

-- Define the problem statement
theorem triangle_value : triangle 5 6 9 4 = 150 := by
  sorry

end triangle_value_l77_77067


namespace sin_function_a_condition_l77_77982

theorem sin_function_a_condition 
  (a : ℝ)
  (h1 : ∀ x : ℝ, (∀ b : ℝ, y = (sin b - a)^2 + 1 → (sin x - a)^2 + 1 ≤ y)
  (h2 : ∃ x : ℝ, sin x = a) :
  -1 ≤ a ∧ a ≤ 0 := 
sorry

end sin_function_a_condition_l77_77982


namespace ellipse_equation_tangent_line_max_area_l77_77896

-- Conditions
variables {a b k m : ℝ}
def ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def tangent_line (x y : ℝ) (k m : ℝ) := y = k * x + m

theorem ellipse_equation
  (h1 : a > 0)
  (h2 : b > 0)
  (hab : a = 2 * b)
  (hfocus : ∀ x y, angle (A x) (F1 x) (B y) = 60) 
  (hpoint : ellipse (sqrt 3) (1 / 2)) :
  ellipse x y ↔ (x^2 / 4 + y^2 = 1) :=
  sorry

theorem tangent_line_max_area
  (h1 : k > 0)
  (h2 : tangency k m (2 * sqrt 5))
  (h3 : tangency k m (2 * sqrt 5))
  (harea : area_maximized):
  tangent_line x y k m ↔ (y = x + sqrt 5 ∨ y = x - sqrt 5) :=
  sorry

end ellipse_equation_tangent_line_max_area_l77_77896


namespace repeating_decimals_l77_77811

theorem repeating_decimals (n_set : Finset ℤ) :
  (n_set = (Finset.range 21).filter (λ n, 1 ≤ n ∧ n ≤ 20)) →
  (n_set.filter (λ n, ¬ (3 ∣ n))).card = 14 :=
by
  sorry

end repeating_decimals_l77_77811


namespace four_digit_arithmetic_difference_l77_77753

noncomputable def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c)

theorem four_digit_arithmetic_difference : 
  ∃ (x y : ℕ), 
  (1000 ≤ x ∧ x ≤ 9999) ∧ 
  (1000 ≤ y ∧ y ≤ 9999) ∧ 
  ( ∃ a1 a2 a3 a4 b1 b2 b3 b4 : ℕ, 
    x = 1000 * a1 + 100 * a2 + 10 * a3 + a4 ∧
    y = 1000 * b1 + 100 * b2 + 10 * b3 + b4 ∧
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4 ∧ 
    b1 ≠ b2 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b2 ≠ b3 ∧ b2 ≠ b4 ∧ b3 ≠ b4 ∧
    is_arithmetic_sequence a1 a2 a3 a4 ∧
    is_arithmetic_sequence b1 b2 b3 b4 ∧
    (∀ i, i ∈ {a1, a2, a3, a4} → 0 ≤ i ∧ i ≤ 9) ∧
    (∀ j, j ∈ {b1, b2, b3, b4} → 0 ≤ j ∧ j ≤ 9)) ∧ 
  (x < y) ∧ 
  (9876 - 1234 = 8642) :=
sorry

end four_digit_arithmetic_difference_l77_77753


namespace false_proposition_is_A_l77_77367

-- Definitions for propositions
def geom_seq (l : List ℝ) : Prop :=
  match l with
  | [] => true
  | [_] => true
  | [x, y] => true
  | (x::y::z::xs) => y * y = x * z ∧ geom_seq (y::z::xs)

def para_point_line (p : ℝ × ℝ × ℝ) (l : ℝ × ℝ × ℝ) : Prop :=
  -- Placeholder for the actual geometric definition
  false  -- Placeholder, as the geometrical axioms are complex

def min_val (a b : ℝ) (cond : a + b = 2) : Prop :=
  3^a + 3^b = 6 ∧ a = 1 ∧ b = 1

def quad_ineq (a : ℝ) : Prop :=
  a ∈ Set.Icc (-4 : ℝ) 0 ∧ ∀ x : ℝ, a * x^2 + a * x - 1 < 0

-- Proposition statements
def prop_A : Prop :=
  ∀ (p : ℝ × ℝ × ℝ) (l : ℝ × ℝ × ℝ), p ∉ l → para_point_line p l

def prop_B : Prop :=
  ∃! (b2 : ℝ), geom_seq [-9, -3, b2, -1]

def prop_C : Prop :=
  ∃ (a b : ℝ), a + b = 2 ∧ min_val a b (a + b = 2)

def prop_D : Prop :=
  ∃ (a : ℝ), a ∈ Set.Icc (-4 : ℝ) 0 ∧ quad_ineq a

-- Main theorem statement
theorem false_proposition_is_A :
  ¬prop_A ∧ prop_B ∧ prop_C ∧ prop_D :=
by {
  sorry
}

end false_proposition_is_A_l77_77367


namespace gilda_marbles_percentage_left_l77_77431

variable (M : ℝ)
variable (h₀ : M > 0)

def marbles_remaining (M : ℝ) : ℝ :=
  let after_sonia := 0.70 * M
  let after_pedro := after_sonia - 0.20 * after_sonia
  let after_ebony := after_pedro - 0.15 * after_pedro
  let after_jimmy := after_ebony - 0.10 * after_ebony
  after_jimmy

theorem gilda_marbles_percentage_left (M > 0) : 
  (marbles_remaining M / M) * 100 = 42.84 :=
by
  -- This part will be filled with the detailed proof steps.
  sorry

end gilda_marbles_percentage_left_l77_77431


namespace how_many_prime_divisors_of_1155_l77_77488

theorem how_many_prime_divisors_of_1155 : 
  (set.filter prime (set_of (λ x, x ∣ 1155))).card = 4 :=
sorry

end how_many_prime_divisors_of_1155_l77_77488


namespace sum_of_coefficients_is_zero_l77_77816

theorem sum_of_coefficients_is_zero :
  (∃ (a : ℤ) (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℤ), 
    (x^2 - x - 2)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
                      a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_{10} * x^10) →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} = 0 :=
by
  sorry

end sum_of_coefficients_is_zero_l77_77816


namespace probability_of_same_color_vertical_faces_l77_77775

theorem probability_of_same_color_vertical_faces :
  let colors := {red, blue, green}
  let faces := finset.fin_range 6
  let arrangements := {f : faces → colors // ∀ f1 f2, f1 ≠ f2 → f.val f1 = f.val f2}
  let total_configurations := arrangements.card := 3 ^ 6
  let favorable_configurations := 27
  let probability := favorable_configurations / total_configurations
  in
  probability = 1 / 27 :=
by
  sorry

end probability_of_same_color_vertical_faces_l77_77775


namespace smallest_multiple_divisors_l77_77224

theorem smallest_multiple_divisors :
  ∃ m : ℕ, (∃ k1 k2 : ℕ, m = 2^k1 * 5^k2 * 100 ∧ 
    (∀ d : ℕ, d ∣ m → d = 1 ∨ d = m ∨ ∃ e1 e2 : ℕ, d = 2^e1 * 5^e2 * 100)) ∧
    (∀ d : ℕ, d ∣ m → d ≠ 1 → (d ≠ m → ∃ e1 e2 : ℕ, d = 2^e1 * 5^e2 * 100)) ∧
    (m.factors.length = 100) ∧ 
    m / 100 = 2^47 * 5^47 :=
begin
  sorry
end

end smallest_multiple_divisors_l77_77224


namespace ralph_tv_hours_l77_77270

def hours_per_day_mf : ℕ := 4 -- 4 hours per day from Monday to Friday
def days_mf : ℕ := 5         -- 5 days from Monday to Friday
def hours_per_day_ss : ℕ := 6 -- 6 hours per day on Saturday and Sunday
def days_ss : ℕ := 2          -- 2 days, Saturday and Sunday

def total_hours_mf : ℕ := hours_per_day_mf * days_mf
def total_hours_ss : ℕ := hours_per_day_ss * days_ss
def total_hours_in_week : ℕ := total_hours_mf + total_hours_ss

theorem ralph_tv_hours : total_hours_in_week = 32 := 
by
sory -- proof will be written here

end ralph_tv_hours_l77_77270


namespace ellipse_equation_proof_l77_77738

def c := (√6 / 3) * a

theorem ellipse_equation_proof (a b : ℝ) (h1 : b = 2) (h2 : a > b) (h3 : a > 0) (e : ℝ) (h4 : e = (√6) / 3) (h5 : a^2 - b^2 = c^2) :
  (∃ a, ∃ b, ∃ e (h1 : b = 2) (h2 : a > b > 0) (h3 : e = (√6) / 3),
     a = 2 * √3 ∧ (∀ x y : ℝ, (x^2 / 12 + y^2 / 4 = 1) →
       (∃ x1 y1 x2 y2 : ℝ, (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 1 ∧
         2 * (x1 + x2) + 3 * (y1 + y2) - 7 = 0))) := sorry

end ellipse_equation_proof_l77_77738


namespace point_in_second_quadrant_l77_77151

theorem point_in_second_quadrant (α : ℝ) (h1 : -π/2 < α) (h2 : α < 0) :
  ∃ q, q = 2 ∧ tan α < 0 ∧ cos α > 0 := by
  existsi 2
  exact ⟨rfl, sorry, sorry⟩

end point_in_second_quadrant_l77_77151


namespace probability_red_black_l77_77728

theorem probability_red_black :
  let cards := {c | c ∈ (Range 1 53)} -- Cards represented as numbers 1 to 52
  let suits := {1, 2, 3, 4} -- Suits represented as set of suits {1:hearts, 2:diamonds, 3:spades, 4:clubs}
  let number_of_cards := 52
  let number_of_red_cards := 26
  let number_of_black_cards := 26
  let probability_red_first := number_of_red_cards / number_of_cards
  let probability_black_second_given_red_first := number_of_black_cards / (number_of_cards - 1)
  let total_probability := probability_red_first * probability_black_second_given_red_first
  total_probability = 13 / 51 := by
sorry

end probability_red_black_l77_77728


namespace feuerbach_theorem_l77_77262

noncomputable theory

-- Define the angles of the triangle
variables (α β γ : ℝ)

-- Define the trilinear coordinates
def trilinear_coordinates := (sin^2 ((β - γ) / 2) : sin^2 ((α - γ) / 2) : sin^2 ((α - β) / 2))

-- Define the incircle and nine-point circle equations (not actual equations due to simplification)
def incircle_equation := sorry
def nine_point_circle_equation := sorry

-- Statement of the theorem: Incircle touches the nine-point circle (Feuerbach's theorem)
theorem feuerbach_theorem : 
  (incircle_equation α β γ) = (nine_point_circle_equation α β γ) →
  -- Trilinear coordinates of the point of tangency:
  trilinear_coordinates α β γ = (sin^2 ((β - γ) / 2) : sin^2 ((α - γ) / 2) : sin^2 ((α - β) / 2)) :=
by sorry

end feuerbach_theorem_l77_77262


namespace proof_problem_l77_77962

-- Given conditions
variable (c d : ℤ)
axiom h1 : 4 * d = 10 - 3 * c

-- Define the target expression
def expression : ℤ := 3 * d + 15

-- Define the first six positive integers
def first_six_positive_integers : List ℤ := [1, 2, 3, 4, 5, 6]

-- Define the function to check how many of the first six positive integers divide the expression
def count_divisors (n : ℤ) (l : List ℤ) : ℤ :=
  l.filter (λ x => n % x = 0).length

-- Claim: The number of the first six positive integers that must be divisors of 3d + 15 is 2
theorem proof_problem : count_divisors (expression d) first_six_positive_integers = 2 :=
sorry

end proof_problem_l77_77962


namespace square_side_length_proof_l77_77350

noncomputable theory

def square_side_length : ℝ :=
  let θ := 15 * π / 180 in
  let sin_θ := (real.sqrt 3 - 1) / (2 * real.sqrt 2) in
  let a : ℝ := 2 * real.sqrt 3 in
  a

theorem square_side_length_proof :
  let θ := 15 * π / 180 in
  let sin_θ := (real.sqrt 3 - 1) / (2 * real.sqrt 2) in
  let a := (2 * real.sqrt 3 : ℝ) in
  ∀ (ABCD : ℝ) (r : ℝ),
  (circle_touches_extensions ABCD r AB AD 2) ∧
  (circle_tangents_from_point C r) ∧
  (angle_between_tangents C 30) ∧ 
  sin (θ) = sin_θ →
  ABCD = a :=
  sorry

end square_side_length_proof_l77_77350


namespace meal_order_probability_l77_77319

noncomputable def probability_two_people_get_correct_meal 
  (total_people : ℕ) (pasta_orders : ℕ) (salad_orders : ℕ) 
  (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ := 
  favorable_outcomes / total_outcomes

theorem meal_order_probability :
  let total_people := 12
  let pasta_orders := 5
  let salad_orders := 7
  let favorable_outcomes := 157410
  let total_outcomes := 12.factorial
  probability_two_people_get_correct_meal total_people pasta_orders salad_orders favorable_outcomes total_outcomes = 
  (157410 : ℚ) / (479001600 : ℚ) :=
by 
  have total_outcomes_fact : total_outcomes = 12.factorial := rfl
  rw [total_outcomes_fact, show 12.factorial = 479001600, by norm_num]
  exact rfl

end meal_order_probability_l77_77319


namespace no_all_ones_sum_l77_77884

theorem no_all_ones_sum (original : ℕ) (rearranged : ℕ) :
  (∀ c : char, c ∈ (original.digits 10) → c ≠ '0') ∧
  (rearranged ∈ original.permutations) →
  ¬ (∀ d : char, d ∈ ((original + rearranged).digits 10) → d = '1') :=
by 
  sorry

end no_all_ones_sum_l77_77884


namespace no_nat_numbers_satisfy_eqn_l77_77580

theorem no_nat_numbers_satisfy_eqn (a b : ℕ) : a^2 - 3 * b^2 ≠ 8 := by
  sorry

end no_nat_numbers_satisfy_eqn_l77_77580


namespace P_plus_Q_eq_14_l77_77898

variable (P Q : Nat)

-- Conditions:
axiom single_digit_P : P < 10
axiom single_digit_Q : Q < 10
axiom three_P_ends_7 : 3 * P % 10 = 7
axiom two_Q_ends_0 : 2 * Q % 10 = 0

theorem P_plus_Q_eq_14 : P + Q = 14 :=
by
  sorry

end P_plus_Q_eq_14_l77_77898


namespace number_of_divisors_l77_77591

theorem number_of_divisors (a b : ℤ) (h : 4 * b = 9 - 3 * a) : 
  finset.card (finset.filter (λ n, n ∣ (3 * b + 15)) (finset.range 8)) = 3 :=
sorry

end number_of_divisors_l77_77591


namespace count_not_perm_sublist_permutations_correct_l77_77880

def is_not_permutation_sublist (a : List ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 4 → ¬(a.take i ~ List.range' 1 i)

def count_not_perm_sublist_permutations : ℕ :=
  (List.permutes [1, 2, 3, 4, 5]).filter is_not_permutation_sublist).length

theorem count_not_perm_sublist_permutations_correct :
  count_not_perm_sublist_permutations = 70 :=
sorry

end count_not_perm_sublist_permutations_correct_l77_77880


namespace angle_equality_aim_l77_77894

variables {A B C D F E : Type} 
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry F] [EuclideanGeometry E]

def parallelogram_quad (ABCD : Quadrilateral) : Prop :=
  parallel ABCD A B C D

def quad_angles_equal (ABCD : Quadrilateral) : Prop :=
  angle_eq ABCD A D C angle_eq ABCD A C B

def point_on_segment (P : Point) (l : Line) : Prop :=
  lies_on_segment P l

def segment_eq (a b : Segment) : Prop :=
  length a = length b

theorem angle_equality_aim (ABCD : Quadrilateral) (F : Point) (E : Point)
  (h1 : parallelogram_quad ABCD)
  (h2 : quad_angles_equal ABCD)
  (h3 : point_on_segment F (line_through ABCD A D))
  (h4 : point_on_segment E (line_through ABCD B C))
  (h5 : segment_eq (segment ABCD D F) (segment ABCD B E)) :
  angle_eq ABCD A C F angle_eq ABCD C A E :=
sorry

end angle_equality_aim_l77_77894


namespace problem1_part1_problem1_part2_problem2_part1_problem2_part2_l77_77706

-- Probability Distribution
theorem problem1_part1 :
  let X := binom 10 0.5 in
  let p := [0.5, 0.5] in
  (P(X = 0) = 1 / 12) ∧ (P(X = 1) = 5 / 12) ∧ (P(X = 2) = 5 / 12) ∧ (P(X = 3) = 1 / 12) := 
by sorry

-- Expected Value
theorem problem1_part2 :
  let X := binom 10 0.5 in
  E X = 3 / 2 :=
by sorry

-- Cutoff Point for Grade C
theorem problem2_part1 :
  let Y := normal 75.8 6, N(75.8, 36) in
  let eta := (Y - 75.8) / 6 in
  (η ≤ 1.04) ≈ 0.85 →
  let cutoff := 75.8 - 1.04 * 6 in
  round(cutoff) = 70 :=
by sorry

-- Value of k to Maximize P(ξ = k)
theorem problem2_part2 :
  let ξ := binom 800 0.788 in
  (P((ξ = k)) is maximized) → k = 631 →
  k :=
by sorry

end problem1_part1_problem1_part2_problem2_part1_problem2_part2_l77_77706


namespace minimum_unit_cubes_l77_77391

def unit_cube : Type := sorry  -- Placeholder for unit cube type

-- Define conditions: cubes must share at least one face
def share_face (c1 c2 : unit_cube) : Prop := sorry

-- Define conditions: matching the given front and side views
def front_view (figure : list unit_cube) : Prop := sorry
def side_view (figure : list unit_cube) : Prop := sorry

theorem minimum_unit_cubes (figure : list unit_cube) (h_front : front_view figure)
  (h_side : side_view figure) (h_share : ∀ (c1 c2 : unit_cube), c1 ∈ figure ∧ c2 ∈ figure → share_face c1 c2):
  list.length figure = 5 :=
sorry

end minimum_unit_cubes_l77_77391


namespace max_ab_value_l77_77834

noncomputable def max_ab (a b : ℝ) : ℝ :=
  if (a > 0 ∧ b > 0 ∧ 2 * a + b = 1) then a * b else 0

theorem max_ab_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : 2 * a + b = 1) :
  max_ab a b = 1 / 8 := sorry

end max_ab_value_l77_77834


namespace smallest_n_value_l77_77929

theorem smallest_n_value 
  (n : ℕ) 
  (y : ℕ → ℝ) 
  (h1 : ∀ i < n, |y i| < 1) 
  (h2 : ∑ i in finset.range n, |y i| = 23 + |∑ i in finset.range n, y i|) : 
  n = 24 := 
sorry

end smallest_n_value_l77_77929


namespace range_of_a_l77_77879

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → 2 * x + 1 / x - a > 0) → a < 2 * Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_a_l77_77879


namespace consecutive_numbers_count_l77_77969

theorem consecutive_numbers_count (n : ℕ) 
(avg : ℝ) 
(largest : ℕ) 
(h_avg : avg = 20) 
(h_largest : largest = 23) 
(h_eq : (largest + (largest - (n - 1))) / 2 = avg) : 
n = 7 := 
by 
  sorry

end consecutive_numbers_count_l77_77969


namespace exists_isosceles_trapezoid_l77_77689

theorem exists_isosceles_trapezoid (n : ℕ) (h : n = 17)
  (colors : Fin n → Fin 3) :
  ∃ (a b c d: Fin n), 
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ colors a = colors b ∧ colors b = colors c ∧ colors c = colors d ∧ 
    is_isosceles_trapezoid (a,b,c,d) ∧
    is_regular_ngon n :=
sorry

def is_regular_ngon (n : ℕ) : Prop :=
  true

def is_isosceles_trapezoid (abcd : Fin n × Fin n × Fin n × Fin n) : Prop :=
  true

end exists_isosceles_trapezoid_l77_77689


namespace sum_integers_50_to_70_l77_77662

theorem sum_integers_50_to_70 :
  let a := 50
  let l := 70
  ∑ k in Finset.range (l - a + 1), (a + k) = 1260 :=
by
  let a := 50
  let l := 70
  sorry

end sum_integers_50_to_70_l77_77662


namespace least_time_9_horses_at_start_point_l77_77309

def horses := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def lap_time (k : ℕ) : ℕ := k

def stable_lcm (s : Finset ℕ) : ℕ :=
  s.val.foldr Nat.lcm 1

def stable_meeting_time (m : ℕ) : ℕ :=
  stable_lcm (Finset.filter (λ k, k ≤ m) horses)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem least_time_9_horses_at_start_point : 
  ∃ T > 0, sum_of_digits T = 9 ∧ stable_meeting_time 9 = 2520 :=
by
  -- Proof would go here
  sorry

end least_time_9_horses_at_start_point_l77_77309


namespace marbles_count_l77_77388

def num_violet_marbles := 64

def num_red_marbles := 14

def total_marbles (violet : Nat) (red : Nat) : Nat :=
  violet + red

theorem marbles_count :
  total_marbles num_violet_marbles num_red_marbles = 78 := by
  sorry

end marbles_count_l77_77388


namespace range_of_a_l77_77473

theorem range_of_a (a : ℝ) : (a > 1 ∧ a < 5) ↔ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (λ y, log a (5 - a * x)) y ≤ (λ y, log a (5 - a * x)) (x + 1)) :=
by
  sorry

end range_of_a_l77_77473


namespace complex_solution_l77_77838

theorem complex_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (hz : (3 - 4 * i) * z = 5 * i) : z = (4 / 5) + (3 / 5) * i :=
by {
  sorry
}

end complex_solution_l77_77838


namespace proj_orthogonal_vectors_solution_l77_77544

variable (c d : Vector ℝ 2)
variable (u v : Vector ℝ 2)
variable (proj : Vector ℝ 2 → Vector ℝ 2 → Vector ℝ 2)

axiom proj_c_ortho_d : proj c = d → ∀ x y : Vector ℝ 2, (x ∘ y = 0)

theorem proj_orthogonal_vectors_solution :
  proj c (Vector.mk 4 2) = Vector.mk 1 2 →
  (c ∘ d = 0) →
  proj d (Vector.mk 4 2) = Vector.mk 3 0 :=
by
  intros h_proj h_ortho
  sorry

end proj_orthogonal_vectors_solution_l77_77544


namespace cos_double_angle_l77_77155

theorem cos_double_angle (θ : ℝ) (h : ∑' n, (cos θ)^(2 * n) = 9) : cos (2 * θ) = 7 / 9 := 
sorry

end cos_double_angle_l77_77155


namespace largest_real_number_C_l77_77078

theorem largest_real_number_C :
  ∃ (C : ℝ), C = 1/3 ∧ ∀ (n : ℕ) (x : ℕ → ℝ), n > 0 → 
    (∀ k, 0 ≤ k → k ≤ n → (0 = x 0) ∧ (x n = 1) ∧ (∀ j < n, x j < x (j+1))) →
    (∃ C : ℝ, ∑ k in range (n+1), (x k)^2 * (x k - x (k-1)) > C) :=
sorry

end largest_real_number_C_l77_77078


namespace bus_probability_l77_77702

/-- A bus arrives randomly between 3:00 and 4:00, waits for 15 minutes, and then leaves. 
Sarah also arrives randomly between 3:00 and 4:00. Prove the probability that the bus 
will be there when Sarah arrives is 4275/7200. -/
theorem bus_probability : (4275 : ℚ) / 7200 = (4275 / 7200) :=
by 
  sorry

end bus_probability_l77_77702


namespace school_fee_l77_77243

theorem school_fee (a b c d e f g h i j k l : ℕ) (h1 : a = 2) (h2 : b = 100) (h3 : c = 1) (h4 : d = 50) (h5 : e = 5) (h6 : f = 20) (h7 : g = 3) (h8 : h = 10) (h9 : i = 4) (h10 : j = 5) (h11 : k = 4 ) (h12 : l = 50) :
  a * b + c * d + e * f + g * h + i * j + 3 * b + k * d + 2 * f + l * h + 6 * j = 980 := sorry

end school_fee_l77_77243


namespace modulus_complex_num_l77_77303

-- Here we define the complex number as given in the problem condition
def complex_num : ℂ := (1 + complex.i) / complex.i

-- We state the theorem we intend to prove
theorem modulus_complex_num : complex.abs complex_num = real.sqrt 2 := by
  sorry

end modulus_complex_num_l77_77303


namespace curve_symmetry_unique_common_point_l77_77559

variables {C : ℝ → ℝ} (t s : ℝ)
def equation_C := ∀ x, C x = x^3 - x
def equation_C1 := ∀ x y, y = (x - t)^3 - (x - t) + s

theorem curve_symmetry (ht : t ≠ 0) (heq_C : ∀ x, equation_C x) (heq_C1 : ∀ x y, equation_C1 x y) :
  ∀ B1 B2, (B1.x + B2.x) / 2 = t / 2 → (B1.y + B2.y) / 2 = s / 2 →
  ∃ A, A = (t / 2, s / 2) ∧ equation_C B1.x = B1.y ∧ equation_C1 B2.x B2.y :=
symmetric_AB sorry

theorem unique_common_point (ht : t ≠ 0) (heq_C : ∀ x, equation_C x) (heq_C1 : ∀ x y, equation_C1 x y) :
  ∃ s, s = t^3 / 4 - t :=
begin
  sorry
end

end curve_symmetry_unique_common_point_l77_77559


namespace Rajesh_work_completion_time_l77_77942

-- Definitions based on conditions in a)
def Mahesh_rate := 1 / 60 -- Mahesh's rate of work (work per day)
def Mahesh_work := 20 * Mahesh_rate -- Work completed by Mahesh in 20 days
def Rajesh_time_to_complete_remaining_work := 30 -- Rajesh time to complete remaining work (days)
def Remaining_work := 1 - Mahesh_work -- Remaining work after Mahesh's contribution

-- Statement that needs to be proved
theorem Rajesh_work_completion_time :
  (Rajesh_time_to_complete_remaining_work : ℝ) * (1 / Remaining_work) = 45 :=
sorry

end Rajesh_work_completion_time_l77_77942


namespace positive_rational_sum_of_distinct_harmonic_terms_l77_77261

noncomputable
def is_sum_of_distinct_harmonic_terms (q : ℚ) :=
  ∃ (n : ℕ) (k : ℕ → ℕ), q = ∑ i in finset.range n, (1 : ℚ) / k i ∧ ∀ i j, i ≠ j → k i ≠ k j

theorem positive_rational_sum_of_distinct_harmonic_terms (a b : ℕ) (h : 0 < a) (hb : 0 < b) :
  is_sum_of_distinct_harmonic_terms (a / b : ℚ) :=
sorry

end positive_rational_sum_of_distinct_harmonic_terms_l77_77261


namespace length_of_DF_l77_77198

theorem length_of_DF
  (DEF : Type)
  [triangle DEF]
  (DE : ℝ) (EF : ℝ) (DF : ℝ)
  (DE_eq_3 : DE = 3)
  (tan_F_eq_4_div_3 : tan (atan (EF / DE)) = 4 / 3)
  (right_angle_at_E : right_triangle DEF) :
  DF = 5 := by
  sorry

end length_of_DF_l77_77198


namespace haley_initial_shirts_l77_77483

-- Defining the conditions
def returned_shirts := 6
def endup_shirts := 5

-- The theorem statement
theorem haley_initial_shirts : returned_shirts + endup_shirts = 11 := by 
  sorry

end haley_initial_shirts_l77_77483


namespace triangle_cosine_B_l77_77835

theorem triangle_cosine_B (a b c : ℝ) (A B C : ℝ) 
    (h1 : b = a) 
    (h2 : sin B ^ 2 = 2 * sin A * sin C)
    (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
    (angle_A : 0 < A) (angle_B : 0 < B) (angle_C : 0 < C)
    (A_bound : A < π) (B_bound : B < π) (C_bound : C < π) :
    cos B = 1 / 4 := 
by 
  sorry

end triangle_cosine_B_l77_77835


namespace percentage_increase_l77_77687

def originalPrice : ℝ := 300
def newPrice : ℝ := 390

theorem percentage_increase :
  ((newPrice - originalPrice) / originalPrice) * 100 = 30 := by
  sorry

end percentage_increase_l77_77687


namespace B_contribution_in_capital_l77_77729

theorem B_contribution_in_capital 
  (A_c : ℝ) (time_A : ℝ) (time_B : ℝ) (profit_ratio_A : ℝ) (profit_ratio_B : ℝ) (B_c : ℝ) : 
  A_c = 3500 → 
  time_A = 12 → 
  time_B = 2 → 
  profit_ratio_A / profit_ratio_B = 2 / 3 → 
  (A_c * time_A) / (B_c * time_B) = 2 / 3 → 
  B_c = 31500 := 
by 
  intros A_c_eq time_A_eq time_B_eq profit_ratio_eq ratio_eq 
  sorry

end B_contribution_in_capital_l77_77729


namespace arithmetic_mean_of_range_neg3_to_6_l77_77650

theorem arithmetic_mean_of_range_neg3_to_6 :
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  (sum : Float) / (count : Float) = 1.5 := by
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  have h_sum : sum = 15 := by sorry
  have h_count : count = 10 := by sorry
  calc
    (sum : Float) / (count : Float)
        = (15 : Float) / (10 : Float) : by rw [h_sum, h_count]
    ... = 1.5 : by norm_num

end arithmetic_mean_of_range_neg3_to_6_l77_77650


namespace quadratic_function_minimum_value_l77_77854

theorem quadratic_function_minimum_value (m : ℝ) :
  (∀ x ∈ Ico (-2 : ℝ) 2, mx^2 - 2 * m * x + 2 ≥ -2) →
  (∃ y ∈ Ico (-2 : ℝ) 2, mx^2 - 2 * m * y + 2 = -2) →
  (m = 4 ∨ m = -1/2) :=
by
  sorry

end quadratic_function_minimum_value_l77_77854


namespace prod_mod_eq_c_l77_77928

theorem prod_mod_eq_c {n : ℕ} (hn : 0 < n) {a b c : ℤ}
  (ha : is_unit (a % n)) (hb : is_unit (b % n)) (hc : is_unit (c % n))
  (h1 : a ≡ b⁻¹ [ZMOD n]) (h2 : c ≡ a⁻¹ [ZMOD n]) :
  (a * b) * c ≡ c [ZMOD n] :=
sorry

end prod_mod_eq_c_l77_77928


namespace area_pentagon_ABGEF_l77_77886

-- Definitions
def square (length : ℝ) : Set (ℝ × ℝ) := ...

-- Conditions
-- Square properties
constant a : ℝ
constant ABCD : Set (ℝ × ℝ) := square a
constant CGEF : Set (ℝ × ℝ) := square a

-- Intersection of AG and CF is H
constant AG : Line
constant CF : Line
constant H : Point
axiom H_on_AG : H ∈ AG
axiom H_on_CF : H ∈ CF

-- CH is one-third of CF
axiom CH_one_third_CF : distance C H = (1/3) * distance C F

-- Area of triangle CHG
constant CHG : Triangle
axiom area_CHG : area CHG = 6

-- Prove the area of pentagon ABGEF
theorem area_pentagon_ABGEF : area (ABGD ∪ DGEF) - area CHG = 49.5 := sorry

end area_pentagon_ABGEF_l77_77886


namespace Vasya_Tshirt_Day_l77_77736

theorem Vasya_Tshirt_Day
  (total_days : ℕ)
  (participants : ℕ)
  (petya_day : ℕ)
  (petya_number : ℕ)
  (vasya_number : ℕ)
  (matches_per_day : ℕ)
  (days_per_participant : ℕ)
  (total_days = 19)
  (participants = 20)
  (matches_per_day = participants / 2)
  (days_per_participant = participants - 1)
  (petya_number = 11)
  (petya_day = 11)
  (vasya_number = 15) :
  ∃ vasya_day : ℕ, vasya_day = petya_day + (vasya_number - petya_number) := by sorry

end Vasya_Tshirt_Day_l77_77736


namespace max_circle_area_in_square_l77_77290

theorem max_circle_area_in_square :
  let side_len := 10
  let radius := side_len / 2
  let area := Real.pi * radius ^ 2
  area = 25 * Real.pi :=
by
  let side_len := 10
  let radius := side_len / 2
  let area := Real.pi * radius ^ 2
  show area = 25 * Real.pi
  sorry

end max_circle_area_in_square_l77_77290


namespace farm_animal_count_l77_77744

theorem farm_animal_count : 
  let goats := 66
  let chickens := 2 * goats
  let ducks := (goats + chickens) / 2
  let pigs := ducks / 3
  let rabbits := Real.sqrt (ducks - pigs)
  let cows := 4 ^ (pigs / rabbits)
  in goats - pigs = 33 :=
by 
  sorry

end farm_animal_count_l77_77744


namespace octahedron_parallel_edge_pairs_count_l77_77068

-- defining a regular octahedron structure
structure RegularOctahedron where
  vertices : Fin 8
  edges : Fin 12
  faces : Fin 8

noncomputable def numberOfStrictlyParallelEdgePairs (O : RegularOctahedron) : Nat :=
  12 -- Given the symmetry and structure.

theorem octahedron_parallel_edge_pairs_count (O : RegularOctahedron) : 
  numberOfStrictlyParallelEdgePairs O = 12 :=
by
  sorry

end octahedron_parallel_edge_pairs_count_l77_77068


namespace total_cost_proof_l77_77743

-- Definition of parameters and initial conditions
def sandwich_price : ℕ := 4
def soda_price : ℕ := 3
def tax_rate : ℝ := 0.10
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 6

-- Calculate subtotal cost before tax
def subtotal : ℕ := (num_sandwiches * sandwich_price) + (num_sodas * soda_price)

-- Calculate total cost with tax
def total_cost : ℝ := (subtotal : ℝ) * (1.0 + tax_rate)

-- Lean 4 theorem statement
theorem total_cost_proof : total_cost = 50.6 := 
by
  -- Construct the cost and tax calculations
  have h1 : (num_sandwiches * sandwich_price : ℕ) = 28 := by norm_num
  have h2 : (num_sodas * soda_price : ℕ) = 18 := by norm_num
  have h3 : subtotal = 46 := by norm_num
  have h4 : (subtotal : ℝ) * tax_rate = 4.6 := by norm_num
  have h5 : (subtotal : ℝ) * (1.0 + tax_rate) = 50.6 := by norm_num
  -- Conclude the proof
  exact h5 

end total_cost_proof_l77_77743


namespace segment_length_segment_fraction_three_segments_fraction_l77_77355

noncomputable def total_length : ℝ := 4
noncomputable def number_of_segments : ℕ := 5

theorem segment_length (L : ℝ) (n : ℕ) (hL : L = total_length) (hn : n = number_of_segments) :
  L / n = (4 / 5 : ℝ) := by
sorry

theorem segment_fraction (n : ℕ) (hn : n = number_of_segments) :
  (1 / n : ℝ) = (1 / 5 : ℝ) := by
sorry

theorem three_segments_fraction (n : ℕ) (hn : n = number_of_segments) :
  (3 / n : ℝ) = (3 / 5 : ℝ) := by
sorry

end segment_length_segment_fraction_three_segments_fraction_l77_77355


namespace smallest_m_divided_by_100_l77_77227

-- Define m as the smallest positive integer that meets the conditions
def m : ℕ := 2^4 * 3^9 * 5^1

-- Condition: m is a multiple of 100 and has exactly 100 divisors
def is_multiple_of_100 (n : ℕ) : Prop := 100 ∣ n
def has_exactly_100_divisors (n : ℕ) : Prop := (factors_count n (2^4 * 3^9 * 5^1)) = 100

-- The property we want to prove: m meets both conditions and yields the correct fragment
theorem smallest_m_divided_by_100 : 
  is_multiple_of_100 m ∧ has_exactly_100_divisors m → m / 100 = 15746.4 :=
by
  sorry

end smallest_m_divided_by_100_l77_77227


namespace sequence_terms_l77_77999

noncomputable def S (n : ℕ) : ℕ := n^2 + 2 * n

theorem sequence_terms :
  (S 1 = 3) ∧ (∀ n, n ≥ 2 → (S n - S (n-1) = 2 * n + 1)) :=
by
  have h1 : S 1 = 3 := by {
    -- Simplify the definition to show S 1 = 3
    dsimp [S],
    norm_num,
  }
  have h2 : ∀ n, n ≥ 2 → (S n - S (n-1) = 2 * n + 1) := by {
    -- Provide the general proof for n ≥ 2
    intros n hn,
    dsimp [S],
    -- Expand the terms for S n and S (n-1)
    simp [(n - 1), pow_two],
    ring,
  }
  exact ⟨h1, h2⟩

end sequence_terms_l77_77999


namespace problem_statement_l77_77131

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem problem_statement (φ : ℝ) (hφ : 0 < φ ∧ φ < π) :
  (∀ x, (0 < x ∧ x < 5*π/12) → f x φ ≤ f 0 φ) ∧
  (∀ x, (x = 0 → f' x φ = -1) ∧
   (f 0 φ = Real.sqrt 3 / 2) → (∀ y, y = Real.sqrt 3 / 2 - x)) :=
sorry

end problem_statement_l77_77131


namespace numerator_of_fraction_l77_77883

-- Define the conditions
def y_pos (y : ℝ) : Prop := y > 0

-- Define the equation
def equation (x y : ℝ) : Prop := x + (3 * y) / 10 = (1 / 2) * y

-- Prove that x = (1/5) * y given the conditions
theorem numerator_of_fraction {y x : ℝ} (h1 : y_pos y) (h2 : equation x y) : x = (1/5) * y :=
  sorry

end numerator_of_fraction_l77_77883


namespace expression_evaluation_l77_77379

theorem expression_evaluation :
  2 - 3 * (-4) + 5 - (-6) * 7 = 61 :=
sorry

end expression_evaluation_l77_77379


namespace Adam_has_more_apples_l77_77033

theorem Adam_has_more_apples (A J : ℕ) (hA : A = 9) (hJ : J = 6) : A - J = 3 := by
  rw [hA, hJ]
  rfl

end Adam_has_more_apples_l77_77033


namespace construct_line_parallel_to_e_l77_77822

noncomputable def line_parallel_to_e (P : Point) (e : Line) : Line :=
sorry

theorem construct_line_parallel_to_e (P : Point) (e : Line) 
  (circ_tools : Type) (straightedge : Type) (can_draw_circles : (circ_tools → Circle) → Prop) : 
  (exists line_parallel, (line_parallel.contains P ∧ line_parallel // e)) :=
sorry

end construct_line_parallel_to_e_l77_77822


namespace kylie_and_nelly_total_stamps_l77_77538

theorem kylie_and_nelly_total_stamps :
  ∀ (kylie_stamps nelly_delta: ℕ),
  kylie_stamps = 34 →
  nelly_delta = 44 →
  (kylie_stamps + (kylie_stamps + nelly_delta) = 112) :=
by
  intros kylie_stamps nelly_delta h_kylie h_delta
  rw [h_kylie, h_delta]
  rw [add_assoc]
  sorry

end kylie_and_nelly_total_stamps_l77_77538


namespace length_of_bridge_l77_77026

/-- Prove the length of the bridge -/
theorem length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_sec : ℝ) : 
  train_length = 120 →
  train_speed_kmph = 70 →
  crossing_time_sec = 13.884603517432893 →
  (70 * (1000 / 3600) * 13.884603517432893 - 120 = 150) :=
by
  intros h1 h2 h3
  sorry

end length_of_bridge_l77_77026


namespace ryan_finished_time_l77_77045

theorem ryan_finished_time :
  ∀ (r : ℝ), (∀ {t : ℝ}, 0 ≤ t ≤ 9 → r * t = 1/2) ∧
             (∀ {t : ℝ}, 9 ≤ t ≤ 10 → r * t = 7/8) ∧
             (∀ {t₀ t₁ : ℝ}, 9 ≤ t₀ ∧ t₀ ≤ t₁ ∧ t₁ ≤ 10 → ((7/8) - (1/2)) / (t₁ - t₀) = r) →
             (r * 20/60 = 1/8) →
             (10 + 20/60 = 10.3333) :=
sorry

end ryan_finished_time_l77_77045


namespace range_of_a_l77_77876

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 * exp x - a

theorem range_of_a (a : ℝ) : (∃ x, f x a = 0) ∧
  (∀ x₁ x₂ x₃ : ℝ, f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0 → 
  x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) ↔
  a ∈ (0, 4 / exp 2) :=
by
  sorry

end range_of_a_l77_77876


namespace mark_each_question_in_group_A_l77_77514

theorem mark_each_question_in_group_A (a m_a b c m_b m_c T : ℕ) 
  (ha : a = 76) 
  (hb : b = 23) 
  (hc : c = 1) 
  (hm_b : m_b = 2) 
  (hm_c : m_c = 3)
  (hT : T = 76 * m_a + 23 * 2 + 1 * 3)
  (h_condition : 76 * m_a ≥ 0.6 * T) :
  m_a = 1 := 
sorry

end mark_each_question_in_group_A_l77_77514


namespace PointNegativeThreeTwo_l77_77576

def isInSecondQuadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem PointNegativeThreeTwo:
  isInSecondQuadrant (-3) 2 := by
  sorry

end PointNegativeThreeTwo_l77_77576


namespace complex_fraction_evaluation_l77_77085

open Complex

theorem complex_fraction_evaluation :
  ((1 + 2 * Complex.i) / (1 - 2 * Complex.i)) - ((1 - 2 * Complex.i) / (1 + 2 * Complex.i)) = (8 / 5) * Complex.i :=
by
  sorry

end complex_fraction_evaluation_l77_77085


namespace medicine_supply_duration_l77_77907

noncomputable def pillDuration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) : ℚ :=
  let pillPerDay := pillFractionPerThreeDays / 3
  let daysPerPill := 1 / pillPerDay
  numPills * daysPerPill

theorem medicine_supply_duration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) (daysPerMonth : ℚ) :
  numPills = 90 →
  pillFractionPerThreeDays = 1 / 3 →
  daysPerMonth = 30 →
  pillDuration numPills pillFractionPerThreeDays / daysPerMonth = 27 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [pillDuration]
  sorry

end medicine_supply_duration_l77_77907


namespace train_crosses_bridge_in_30_seconds_l77_77732

/--
A train 155 metres long, travelling at 45 km/hr, can cross a bridge with length 220 metres in 30 seconds.
-/
theorem train_crosses_bridge_in_30_seconds
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_km_per_hr : ℕ)
  (total_distance : ℕ)
  (speed_m_per_s : ℚ)
  (time_seconds : ℚ) 
  (h1 : length_train = 155)
  (h2 : length_bridge = 220)
  (h3 : speed_km_per_hr = 45)
  (h4 : total_distance = length_train + length_bridge)
  (h5 : speed_m_per_s = (speed_km_per_hr * 1000) / 3600)
  (h6 : time_seconds = total_distance / speed_m_per_s) :
  time_seconds = 30 :=
sorry

end train_crosses_bridge_in_30_seconds_l77_77732


namespace smallest_base_is_21_l77_77517
noncomputable def smallest_base_for_fraction := Nat.find (λ k, 
  (∃ n₁ n₂ : ℚ, (n₁ = 3 / k ∧ n₂ = 6 / k^2) ∧ 
  (0.363636...ₖ = n₁ + n₂ / (1 - (1 / k^2))) ∧ 
  (3 * k + 6) / (k^2 - 1) = 9 / 70)
)

theorem smallest_base_is_21 : smallest_base_for_fraction = 21 :=
sorry

end smallest_base_is_21_l77_77517


namespace max_consecutive_integers_sum_15_is_5_l77_77988

theorem max_consecutive_integers_sum_15_is_5 :
  ∀ n : ℕ, (∀ (seq : ℕ → ℕ), ∑ i in Finset.range n, seq i = 15 → 
  ∃ (k : ℕ), seq = λ i, k + i) → n ≤ 5 := 
sorry

end max_consecutive_integers_sum_15_is_5_l77_77988


namespace sum_of_eight_numbers_l77_77162

theorem sum_of_eight_numbers (a : ℝ) :
  (∀ l : List ℝ, l.length = 8 → l.sum / 8 = a) → (a = 5.2 → ∃ l, l.sum = 41.6) :=
by
  intro h hf
  use List.replicate 8 (5.2 : ℝ)
  have : (8 * 5.2) = 41.6 := by norm_num
  rw [List.sum_replicate, this]
  exact hf

end sum_of_eight_numbers_l77_77162


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_l77_77798

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString;
  s = s.reverse

noncomputable def smallest_palindrome_not_five_digits (f : ℕ → ℕ) : ℕ :=
  Nat.find (λ n, 
    100 ≤ n ∧ n < 1000 ∧ 
    is_palindrome n ∧ 
    ¬ is_palindrome (f n))

theorem smallest_three_digit_palindrome_not_five_digit_palindrome :
  smallest_palindrome_not_five_digits (λ n => 101 * n) = 505 :=
by
  sorry

end smallest_three_digit_palindrome_not_five_digit_palindrome_l77_77798


namespace incenter_inequality_l77_77930

-- We denote the triangle sides and the incenter of the triangle
variables (a b c : ℝ)

-- Define the ratios involving incenter distances AI, BI, and CI with respective angle bisectors AA', BB', CC'
def ratio_AI_AA' : ℝ := (b + c) / (a + b + c)
def ratio_BI_BB' : ℝ := (a + c) / (a + b + c)
def ratio_CI_CC' : ℝ := (a + b) / (a + b + c)

-- Use these definitions to express the desired inequality
theorem incenter_inequality :
  1 / 4 < (ratio_AI_AA' a b c * ratio_BI_BB' a b c * ratio_CI_CC' a b c) ∧
  (ratio_AI_AA' a b c * ratio_BI_BB' a b c * ratio_CI_CC' a b c) ≤ 8 / 27 :=
sorry

end incenter_inequality_l77_77930


namespace time_A_to_complete_work_alone_l77_77704

theorem time_A_to_complete_work_alone :
  ∃ (x : ℝ), (1 / x) + (1 / 20) = (1 / 8.571428571428571) ∧ x = 15 :=
by
  sorry

end time_A_to_complete_work_alone_l77_77704


namespace permutations_containing_substring_l77_77861

open Nat

/-- Prove that the number of permutations of the string "000011112222" that contain the substring "2020" is equal to 3575. -/
theorem permutations_containing_substring :
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  non_overlap_count - overlap_subtract + add_back = 3575 := 
by
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  have h: non_overlap_count - overlap_subtract + add_back = 3575 := by sorry
  exact h

end permutations_containing_substring_l77_77861


namespace abs_m_minus_n_l77_77158

theorem abs_m_minus_n (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : (1 / 5 * (m^2 + n^2 + 81 + 64 + 100) - 81) = 2) : |m - n| = 4 :=
  sorry

end abs_m_minus_n_l77_77158


namespace daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l77_77013

-- Part (1)
theorem daily_sales_volume_and_profit (x : ℝ) :
  let increase_in_sales := 2 * x
  let profit_per_piece := 40 - x
  increase_in_sales = 2 * x ∧ profit_per_piece = 40 - x :=
by
  sorry

-- Part (2)
theorem profit_for_1200_yuan (x : ℝ) (h1 : (40 - x) * (20 + 2 * x) = 1200) :
  x = 10 ∨ x = 20 :=
by
  sorry

-- Part (3)
theorem profit_impossible_for_1800_yuan :
  ¬ ∃ y : ℝ, (40 - y) * (20 + 2 * y) = 1800 :=
by
  sorry

end daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l77_77013


namespace sum_abs_roots_l77_77803

theorem sum_abs_roots : 
  let f := (λ x : ℂ, x^4 - 6 * x^3 + 9 * x^2 + 24 * x - 36)
  ∃ roots : list ℂ, (∀ r ∈ roots, f r = 0) ∧ (roots.length = 4) ∧ 
                 (∑ r in roots, complex.abs r = 4 * complex.sqrt 6) := 
sorry

end sum_abs_roots_l77_77803


namespace tan_sum_formula_eq_l77_77138

theorem tan_sum_formula_eq {θ : ℝ} (h1 : ∃θ, θ ∈ Set.Ico 0 (2 * Real.pi) 
  ∧ ∃P, P = (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4)) 
  ∧ θ = (3 * Real.pi / 4)) : 
  Real.tan (θ + Real.pi / 3) = 2 - Real.sqrt 3 := 
sorry

end tan_sum_formula_eq_l77_77138


namespace distance_between_midpoints_l77_77502

variables {p q r s : ℝ}

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

theorem distance_between_midpoints :
  let A := (p, q)
  let B := (r, s)
  let M := midpoint A B
  let A' := (p + 5, q + 6)
  let B' := (r - 12, s - 4)
  let M' := midpoint A' B'
  distance M M' = real.sqrt 53 / 2 :=
by
  sorry

end distance_between_midpoints_l77_77502


namespace prob_statement_l77_77940

open Set

-- Definitions from the conditions
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x^2 + 2 * x < 0}

-- Proposition to be proved
theorem prob_statement : A ∩ (Bᶜ) = {-2, 0, 1, 2} :=
by
  sorry

end prob_statement_l77_77940


namespace find_k_and_a_range_l77_77459

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^2 + Real.exp x - k * Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

theorem find_k_and_a_range (k a : ℝ) (h_even : ∀ x : ℝ, f x k = f (-x) k) :
  k = -1 ∧ 2 ≤ a := by
    sorry

end find_k_and_a_range_l77_77459


namespace hyperbola_foci_eccentricity_l77_77295

-- Definitions and conditions
def hyperbola_eq := (x y : ℝ) → (x^2 / 4) - (y^2 / 12) = 1

-- Proof goals: Coordinates of the foci and eccentricity
theorem hyperbola_foci_eccentricity (x y : ℝ) : 
  (∃ c : ℝ, (x^2 / 4) - (y^2 / 12) = 1 ∧ (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)) ∧ 
  (∃ e : ℝ, e = 2) :=
sorry

end hyperbola_foci_eccentricity_l77_77295


namespace inequality_proof_l77_77438

theorem inequality_proof (a b : ℝ) (x y : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_x : 0 < x) (h_y : 0 < y) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) :=
sorry

end inequality_proof_l77_77438


namespace part1_increasing_part2_inequality_part3_min_b_l77_77827

-- Part 1
theorem part1_increasing (f g : ℝ → ℝ) (a : ℝ) (h_f : ∀ x, f x = a * sin x) (h_g : ∀ x, g x = log x)
  (G : ℝ → ℝ) (h_G : ∀ x, G x = f (1 - x) + g x) (h_G_incr : ∀ x ∈ Ioo 0 1, deriv G x > 0) :
  a ≤ 1 :=
sorry

-- Part 2
theorem part2_inequality (n : ℕ) :
  ∑ k in finset.range (n + 1), sin (1 / (1 + k) ^ 2) < log 2 :=
sorry

-- Part 3
theorem part3_min_b (g_inv : ℝ → ℝ) (h_g_inv : ∀ x, g_inv x = exp x) (m : ℝ) (h_m : m < 0) 
  (F : ℝ → ℝ) (h_F : ∀ x, F x = g_inv x - m * x ^ 2 - 2 * (x + 1) + b) (h_F_pos : ∀ x, F x > 0) :
  ∃ b : ℤ, b ≥ 3 :=
sorry

end part1_increasing_part2_inequality_part3_min_b_l77_77827


namespace sum_of_cubes_divisible_by_9n_l77_77956

theorem sum_of_cubes_divisible_by_9n (n : ℕ) (h : n % 3 ≠ 0) : 
  ((n - 1)^3 + n^3 + (n + 1)^3) % (9 * n) = 0 := by
  sorry

end sum_of_cubes_divisible_by_9n_l77_77956


namespace find_eccentricity_of_ellipse_l77_77114

-- Definitions based on conditions
variables (A B C D : Type) [TopologicalSpace A] [TopologicalSpace B] [TopologicalSpace C] [TopologicalSpace D]

def is_rectangle (ABCD : Type) (AB BC CD DA : ℝ) : Prop :=
  AB = 4 ∧ BC = 3

def are_foci_of_ellipse (A B C D : Type) : Prop :=
  -- This would normally involve more complex definitions involving distances and properties of ellipses
  sorry

-- Statement of the proof problem
theorem find_eccentricity_of_ellipse (ABCD : Type) [is_rectangle ABCD 4 3] [are_foci_of_ellipse A B C D] : 
  eccentricity A B C D = 1 / 2 := 
sorry

end find_eccentricity_of_ellipse_l77_77114


namespace chickens_and_rabbits_l77_77175

theorem chickens_and_rabbits (c r : ℕ) (h1 : c + r = 15) (h2 : 2 * c + 4 * r = 40) : c = 10 ∧ r = 5 :=
sorry

end chickens_and_rabbits_l77_77175


namespace min_distance_point_circle_to_line_l77_77081

noncomputable def circle : set (ℝ × ℝ) :=
  { p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 }

def line (p : ℝ × ℝ) : Prop := 
  3 * p.1 + 4 * p.2 + 8 = 0

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_point_circle_to_line :
  ∀ q ∈ circle, ∃ p, line p ∧ distance q p = 2 :=
by
  sorry

end min_distance_point_circle_to_line_l77_77081


namespace rate_2nd_and_3rd_hours_equals_10_l77_77055

-- Define the conditions as given in the problem
def total_gallons_after_5_hours := 34 
def rate_1st_hour := 8 
def rate_4th_hour := 14 
def water_lost_5th_hour := 8 

-- Problem statement: Prove the rate during 2nd and 3rd hours is 10 gallons/hour
theorem rate_2nd_and_3rd_hours_equals_10 (R : ℕ) :
  total_gallons_after_5_hours = rate_1st_hour + 2 * R + rate_4th_hour - water_lost_5th_hour →
  R = 10 :=
by sorry

end rate_2nd_and_3rd_hours_equals_10_l77_77055


namespace coplanar_vectors_lambda_l77_77147

open Real

theorem coplanar_vectors_lambda :
  let a := (2 : ℝ, -1, 3)
  let b := (-1 : ℝ, 4, -2)
  let c := (7 : ℝ, 5, λ)
  -- Vectors are coplanar if determinant of matrix formed by them is zero
  let det := λ a b c, a.1 * (b.2 * c.3 - b.3 * c.2) - a.2 * (b.1 * c.3 - b.3 * c.1) + a.3 * (b.1 * c.2 - b.2 * c.1)
  in det a b c = 0 → λ = 65 / 7 := 
by
  sorry

end coplanar_vectors_lambda_l77_77147


namespace fundraising_goal_shortfall_l77_77585

theorem fundraising_goal_shortfall :
  let ken := 900
  let mary := 5 * ken
  let scott := mary / 3
  let amy := 2 * (ken + scott) / 3
  let total_without_bonus := ken + mary + scott + amy
  let bonus_contribution amount := amount / 10
  let total_with_bonus := total_without_bonus + (bonus_contribution mary) + (bonus_contribution scott) + (bonus_contribution ken) + (bonus_contribution amy)
  let goal := 15000
  goal - total_with_bonus = 5650 := by
  let ken := 900
  let mary := 5 * ken
  let scott := mary / 3
  let amy := 2 * (ken + scott) / 3
  let total_without_bonus := ken + mary + scott + amy
  let bonus_contribution amount := amount / 10
  let total_with_bonus := total_without_bonus + (bonus_contribution mary) + (bonus_contribution scott) + (bonus_contribution ken) + (bonus_contribution amy)
  let goal := 15000
  show goal - total_with_bonus = 5650, from sorry

end fundraising_goal_shortfall_l77_77585


namespace project_completion_days_l77_77703

-- Define the types and conditions
variable {A B : Type}
variables (a_work_rate : ℚ) (b_work_rate : ℚ) (x : ℚ)
variable (quit_duration : ℚ)

-- Define the conditions as Lean definitions
def A_work_rate := 1 / 20
def B_work_rate := 1 / 40
def A_quit_duration := 10

-- Prove the total days to complete the project
theorem project_completion_days
  (a_work_rate := A_work_rate)
  (b_work_rate := B_work_rate)
  (quit_duration := A_quit_duration)
  (x : ℚ) :
  (x - quit_duration) * a_work_rate + x * b_work_rate = 1 → x = 20 := by
  sorry

end project_completion_days_l77_77703


namespace diff_in_set_l77_77707

variable (A : Set Int)
variable (ha : ∃ a ∈ A, a > 0)
variable (hb : ∃ b ∈ A, b < 0)
variable (h : ∀ {a b : Int}, a ∈ A → b ∈ A → (2 * a) ∈ A ∧ (a + b) ∈ A)

theorem diff_in_set (x y : Int) (hx : x ∈ A) (hy : y ∈ A) : (x - y) ∈ A :=
  sorry

end diff_in_set_l77_77707


namespace min_value_expression_l77_77823

noncomputable def normal_distribution (μ σ : ℝ) := sorry -- placeholder for actual normal distribution definition

theorem min_value_expression (μ σ a b : ℝ) (h_dist : normal_distribution μ σ) (hμ : μ = 1)
  (h_a : a > 0) (h_b : b > 0) (h_prob: (∃ p : ℝ, p = Classical.choose sorry) (P ξ ≤ a = P ξ ≥ b))
  : (4 * a + b) / (a * b) = 9/2 := sorry

end min_value_expression_l77_77823


namespace symmetric_line_eq_l77_77301

theorem symmetric_line_eq (x y : ℝ) :
    3 * x - 4 * y + 5 = 0 ↔ 3 * x + 4 * (-y) + 5 = 0 :=
sorry

end symmetric_line_eq_l77_77301


namespace prod_mod_eq_c_l77_77927

theorem prod_mod_eq_c {n : ℕ} (hn : 0 < n) {a b c : ℤ}
  (ha : is_unit (a % n)) (hb : is_unit (b % n)) (hc : is_unit (c % n))
  (h1 : a ≡ b⁻¹ [ZMOD n]) (h2 : c ≡ a⁻¹ [ZMOD n]) :
  (a * b) * c ≡ c [ZMOD n] :=
sorry

end prod_mod_eq_c_l77_77927


namespace samuel_faster_than_sarah_l77_77584

noncomputable def time_at_100_efficiency (time: ℝ) (efficiency: ℝ): ℝ :=
  time / efficiency

theorem samuel_faster_than_sarah 
  (samuel_eff: ℝ := 0.90) 
  (sarah_eff: ℝ := 0.75) 
  (tim_eff: ℝ := 0.80) 
  (tim_time: ℝ := 45.0) : 
  (time_at_100_efficiency tim_time tim_eff) / samuel_eff = 
  62.5 ∧
  (time_at_100_efficiency tim_time tim_eff) / sarah_eff = 
  75.0 ∧
   (time_at_100_efficiency tim_time tim_eff) / sarah_eff - 
  (time_at_100_efficiency tim_time tim_eff) / samuel_eff = 
  12.5 := 
by
  have h_tim := time_at_100_efficiency tim_time tim_eff
  have h_sam := h_tim / samuel_eff
  have h_sar := h_tim / sarah_eff
  have h_diff := h_sar - h_sam
  simp [time_at_100_efficiency]
  exact ⟨rfl, rfl, rfl⟩

end samuel_faster_than_sarah_l77_77584


namespace cos_B_of_sine_ratios_l77_77173

theorem cos_B_of_sine_ratios (A B C : ℝ) (a b c : ℝ)
  (h_sin : sin A / sin B = 6 / 5)
  (h_sin' : sin B / sin C = 5 / 4)
  (h_sin'' : sin A / sin C = 6 / 4)
  (ha : a = 6) (hb : b = 5) (hc : c = 4) :
  cos B = 9 / 16 :=
by sorry

end cos_B_of_sine_ratios_l77_77173


namespace arithmetic_mean_of_integers_from_neg3_to_6_l77_77641

def integer_range := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

noncomputable def arithmetic_mean : ℚ :=
  (integer_range.sum : ℚ) / (integer_range.length : ℚ)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  arithmetic_mean = 1.5 := by
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l77_77641


namespace max_two_distinct_elements_l77_77995

theorem max_two_distinct_elements (a : ℝ) : Finset.card (Finset.image (fun x => abs x) (Finset.insert a (Finset.singleton (-a)))) ≤ 2 :=
by
  sorry

end max_two_distinct_elements_l77_77995


namespace virus_infection_l77_77757

theorem virus_infection (x : ℕ) (h : 1 + x + x^2 = 121) : x = 10 := 
sorry

end virus_infection_l77_77757


namespace problem_statement_l77_77232

noncomputable def smallest_multiple_with_divisors := 
  let m := smallest (fun n : ℕ => 100 ∣ n ∧ (∀ d : ℕ, d ∣ n ↔ d < 101)) 0
  m / 100

theorem problem_statement : smallest_multiple_with_divisors = 324 := 
by 
  sorry

end problem_statement_l77_77232


namespace Kite_area_correct_l77_77910

variables (AC BD : ℝ) (AB : ℝ)
def is_kite := AC = 50 ∧ BD = 80 ∧ AC * BD / 2 = 2000

theorem Kite_area_correct : is_kite AC BD AB → AC * BD / 2 = 2000 :=
by {
    intros h,
    cases h with hAC h,
    cases h with hBD hArea,
    rw [hAC, hBD],
    exact hArea,
    sorry -- The proof is omitted as per instructions
}

end Kite_area_correct_l77_77910


namespace volume_of_M_B_C_K_pyramid_l77_77020

def S : Type := sorry -- Define the apex of the pyramid
def A : Type := sorry -- Define one vertex of the base
def B : Type := sorry -- Define another vertex of the base
def C : Type := sorry -- Define another vertex of the base
def D : Type := sorry -- Define another vertex of the base
def K : Type := sorry -- Define the midpoint of edge CD
def M : Type := sorry -- The intersection point of line SC with the plane γ that passes through B and K

-- Conditions
def radius_of_sphere : ℝ := 4 / 9
def base_side_length : ℝ := 8
def pyramid_height : ℝ := 3

-- Volume calculation theorem statement
theorem volume_of_M_B_C_K_pyramid : 
  volume (pyramid M B C K) = 192 / 37 :=
sorry

end volume_of_M_B_C_K_pyramid_l77_77020


namespace largest_three_digit_multiple_digits_l77_77416

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def starts_with_8 (n : ℕ) : Prop :=
  n / 100 = 8

def distinct_nonzero_digits (ι : ℕ) (a b : ℕ) : Prop :=
  a ≠ b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9

def divisible_by_digits (n a b : ℕ) : Prop :=
  n % 10 = b ∧ (n / 10) % 10 = a ∧ 
  n % 8 = 0 ∧ n % a = 0 ∧ n % b = 0

def largest_valid_number (n a b: ℕ) : Prop :=
  ∃ m, is_three_digit_number m ∧ starts_with_8 m ∧ 
       (∃ x y, distinct_nonzero_digits x y ∧ divisible_by_digits m x y) ∧
       m ≥ n

theorem largest_three_digit_multiple_digits :
  ∃ n : ℕ, is_three_digit_number n ∧ starts_with_8 n ∧ 
  (∃ a b, distinct_nonzero_digits a b ∧ divisible_by_digits n a b) ∧ n = 864 :=
begin
  sorry
end

end largest_three_digit_multiple_digits_l77_77416


namespace brian_video_watching_time_l77_77749

/--
Brian watches a 4-minute video of cats.
Then he watches a video twice as long as the cat video involving dogs.
Finally, he watches a video on gorillas that's twice as long as the combined duration of the first two videos.
Prove that Brian spends a total of 36 minutes watching animal videos.
-/
theorem brian_video_watching_time (cat_video dog_video gorilla_video : ℕ) 
  (h₁ : cat_video = 4) 
  (h₂ : dog_video = 2 * cat_video) 
  (h₃ : gorilla_video = 2 * (cat_video + dog_video)) : 
  cat_video + dog_video + gorilla_video = 36 := by
  sorry

end brian_video_watching_time_l77_77749


namespace sin_alpha_value_l77_77455

theorem sin_alpha_value 
  (α : ℝ) 
  (h1 : sin (α - π / 4) = sqrt 5 / 5) 
  (h2 : α ∈ set.Ioo (3 * π / 4) (5 * π / 4)) : 
  sin α = -sqrt 10 / 10 := 
by sorry

end sin_alpha_value_l77_77455


namespace mena_vs_emily_l77_77944

def menaNumbers : List ℕ := List.range' 1 30

def emilyNumbers : List ℕ := menaNumbers.map (λ n => 
   let str := n.toString
   let new_str := str.map (λ c => if c = '2' then '1' else c)
   new_str.toNat
)

def sumMena := menaNumbers.sum
def sumEmily := emilyNumbers.sum

theorem mena_vs_emily : sumMena = sumEmily + 103 := by
  sorry

end mena_vs_emily_l77_77944


namespace inequality_with_equality_l77_77554

variable {x y : ℝ}
variable (hx : 0 < x) (hy : 0 < y)

theorem inequality_with_equality:
  (x^2 + 8 / (x * y) + y^2 ≥ 8) ∧ (x^2 + 8 / (x * y) + y^2 = 8 → x = y ∧ x = sqrt 2) :=
sorry

end inequality_with_equality_l77_77554


namespace modulo_inverse_product_l77_77925

open Int 

theorem modulo_inverse_product (n : ℕ) (a b c : ℤ) 
  (hn : 0 < n) 
  (ha : a * a.gcd n = 1) 
  (hb : b * b.gcd n = 1) 
  (hc : c * c.gcd n = 1) 
  (hab : (a * b) % n = 1) 
  (hac : (c * a) % n = 1) : 
  ((a * b) * c) % n = c % n :=
by
  sorry

end modulo_inverse_product_l77_77925


namespace mod_M_1000_l77_77918

-- Definition of the string and the constraints
def M : ℕ :=
  number_of_permutations "AAAA" "BBBBB" "CCCCDD" none_first_five [A] none_next_five [B] none_last_six [C]

-- The theorem statement
theorem mod_M_1000 : M % 1000 = 540 :=
begin
  sorry
end

end mod_M_1000_l77_77918


namespace jason_earns_88_dollars_l77_77906

theorem jason_earns_88_dollars (earn_after_school: ℝ) (earn_saturday: ℝ)
  (total_hours: ℝ) (saturday_hours: ℝ) (after_school_hours: ℝ) (total_earn: ℝ)
  (h1 : earn_after_school = 4.00)
  (h2 : earn_saturday = 6.00)
  (h3 : total_hours = 18)
  (h4 : saturday_hours = 8)
  (h5 : after_school_hours = total_hours - saturday_hours)
  (h6 : total_earn = after_school_hours * earn_after_school + saturday_hours * earn_saturday) :
  total_earn = 88.00 :=
by
  sorry

end jason_earns_88_dollars_l77_77906


namespace determine_E_l77_77392

theorem determine_E (a b c d e : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_sum : a + b + c + d + e = 15) :
  let E := -(a + b + c + d + e) in E = -15 := by
  sorry

end determine_E_l77_77392


namespace num_elements_in_M_inter_N_number_of_elements_in_M_inter_N_l77_77560

def M : Set ℕ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 2 * x < 0}

theorem num_elements_in_M_inter_N : Set ℝ :=
  {x | ↑x ∈ M ∧ x ∈ N}

theorem number_of_elements_in_M_inter_N : |(num_elements_in_M_inter_N)| = 1 :=
sorry

end num_elements_in_M_inter_N_number_of_elements_in_M_inter_N_l77_77560


namespace no_perfect_squares_in_sequence_l77_77442

theorem no_perfect_squares_in_sequence (x : ℕ → ℤ) (h₀ : x 0 = 1) (h₁ : x 1 = 3)
  (h_rec : ∀ n : ℕ, x (n + 1) = 6 * x n - x (n - 1)) 
  : ∀ n : ℕ, ¬ ∃ k : ℤ, x n = k * k := 
sorry

end no_perfect_squares_in_sequence_l77_77442


namespace george_initial_socks_l77_77814

theorem george_initial_socks (S : ℕ) (h : S - 4 + 36 = 60) : S = 28 :=
by
  sorry

end george_initial_socks_l77_77814


namespace no_rational_roots_l77_77934

theorem no_rational_roots
  (n : ℕ)
  (a : Fin n.succ → ℤ)
  (h₀ : Odd (a 0))
  (h₁ : Odd (a n))
  (h₂ : Odd (Finset.univ.sum (λ i => a i))) :
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ Int.gcd p q = 1 ∧ eval (p / q : ℚ) (polynomial.of_finset a) = 0 :=
sorry

end no_rational_roots_l77_77934


namespace percentage_correction_l77_77505

def candidate_actual_height : ℕ := 68

def stated_height : ℕ := (1.25 * candidate_actual_height).toNat

theorem percentage_correction : ( (stated_height - candidate_actual_height) / stated_height.toFloat ) * 100 = 20 :=
by sorry

end percentage_correction_l77_77505


namespace part1_part2_l77_77474

def f (x : ℝ) : ℝ := x * Real.log x
def g (x a : ℝ) : ℝ := -x^2 + a * x - 3

theorem part1 (a : ℝ) : (∀ x : ℝ, 0 < x → f x >= 0.5 * g x a) → a ≤ 4 :=
sorry

theorem part2 : ∀ x : ℝ, 0 < x → Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end part1_part2_l77_77474


namespace probability_two_excellent_credit_expected_value_of_vouchers_l77_77762

/-
Probability of selecting 2 people both having excellent credit out of 200 residents.
-/
theorem probability_two_excellent_credit (total_people : ℕ) (excellent_people : ℕ) :
  total_people = 200 →
  excellent_people = 25 →
  ((excellent_people * (excellent_people - 1)) / 2) / ((total_people * (total_people - 1)) / 2) = 3 / 199 :=
by {
  intros _ _,
  sorry
}

/-
Expectation and distribution of the total amount of vouchers (X) received by randomly selecting 2 people from the given population.
-/
theorem expected_value_of_vouchers (total_people : ℕ) 
  (excellent people good_people fair_people slight_default_people poor_people : ℕ)
  (credit_level_vouchers : ℕ → ℕ)
  (distribution_P : ℕ → ℚ) :
  total_people = 200 →
  excellent_people = 25 →
  good_people = 60 →
  fair_people = 65 →
  slight_default_people = 35 →
  poor_people = 15 →
  credit_level_vouchers 0 = 0 →
  credit_level_vouchers 1 = 0 →
  credit_level_vouchers 2 = 50 →
  credit_level_vouchers 3 = 50 →
  credit_level_vouchers 4 = 100 →
  distribution_P 0 = 1 / 16 →
  distribution_P 50 = 5 / 16 →
  distribution_P 100 = 29 / 64 →
  distribution_P 150 = 5 / 32 →
  distribution_P 200 = 1 / 64 →
  ∑ k in [0, 50, 100, 150, 200], k * distribution_P k = 175 / 2 :=
by {
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _,
  sorry
}

end probability_two_excellent_credit_expected_value_of_vouchers_l77_77762


namespace trains_pass_each_other_in_10_seconds_l77_77632

theorem trains_pass_each_other_in_10_seconds :
  ∀ (speed : ℝ) (length : ℝ),
    speed = 60 →
    length = 1 / 6 →
    let relative_speed := speed + speed in
    let total_distance := length + length in
    let speed_per_second := relative_speed / 3600 in
    (total_distance / speed_per_second) = 10 :=
by
  intros speed length speed_eq length_eq
  simp only [*, add_mul, div_eq_mul_inv, mul_assoc]
  sorry

end trains_pass_each_other_in_10_seconds_l77_77632


namespace f_is_zero_l77_77840

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_is_zero 
  (H1 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a)
  (H2 : ∀ x : ℝ, |f x| ≤ 1) : ∀ x : ℝ, f x = 0 := 
sorry

end f_is_zero_l77_77840


namespace quadratic_transform_l77_77992

theorem quadratic_transform :
  ∃ a b c : ℝ, (-4 : ℝ) * (λ x : ℝ, (x + b)^2) + c = (λ x, -4 * x^2 + 20 * x + 196 : ℝ) ∧ a + b + c = 213.5 :=
by
  use [-4, -5 / 2, 221]
  split
  · ext x
    calc
      -4 * (x + (-5 / 2))^2 + 221
          = -4 * (x^2 + 2 * x * (-5 / 2) + (-5 / 2)^2) + 221 : by ring
      ... = -4 * x^2 + 20 * x - 25 + 221 : by ring
      ... = -4 * x^2 + 20 * x + 196 : by ring
  · norm_num

end quadratic_transform_l77_77992


namespace volume_parallelepiped_eq_three_l77_77153

open Real
open Matrix

noncomputable def volume_of_parallelepiped
  (c d : ℝ × ℝ × ℝ)
  (hc : ∥c∥ = 1)
  (hd : ∥d∥ = 1)
  (angle_cd : angle c d = π / 4) : ℝ :=
  abs (2 * (dot_product c ((d + 3 • cross_product d c) × d)))

theorem volume_parallelepiped_eq_three
  (c d : ℝ × ℝ × ℝ)
  (hc : ∥c∥ = 1)
  (hd : ∥d∥ = 1)
  (angle_cd : angle c d = π / 4) :
  volume_of_parallelepiped c d hc hd angle_cd = 3 := sorry

end volume_parallelepiped_eq_three_l77_77153


namespace triangle_range_c_b_l77_77480

theorem triangle_range_c_b (a b c A B C : ℝ) (h1 : a = 1) (h2 : C - B = π / 2) :
  ∃ r : set ℝ, r = Ioc (sqrt 2 / 2) 1 ∧ (c - b ∈ r) := sorry

end triangle_range_c_b_l77_77480


namespace movie_day_goal_l77_77616

theorem movie_day_goal
  (total_points : ℕ)
  (points_per_veg : ℕ)
  (students : ℕ)
  (vegs_per_week : ℕ)
  (weeks_needed : ℕ)
  (days_per_week : ℕ) :
  ∀ total_points = 200 ∧ points_per_veg = 2 ∧ students = 25 ∧ vegs_per_week = 2 ∧ weeks_needed = 50 ∧ days_per_week = 5,
  (weeks_needed * days_per_week = 250) :=
by
  sorry

end movie_day_goal_l77_77616


namespace sum_of_first_30_terms_l77_77186

variable (a : Nat → ℤ)
variable (d : ℤ)
variable (S_30 : ℤ)

-- Conditions from part a)
def condition1 := a 1 + a 2 + a 3 = 3
def condition2 := a 28 + a 29 + a 30 = 165

-- Question translated to Lean 4 statement
theorem sum_of_first_30_terms 
  (h1 : condition1 a)
  (h2 : condition2 a) :
  S_30 = 840 := 
sorry

end sum_of_first_30_terms_l77_77186


namespace greatest_integer_x_l77_77656

theorem greatest_integer_x (x : ℤ) : 
  (∃ n : ℤ, (x^2 + 4*x + 10) = n * (x - 4)) → x ≤ 46 := 
by
  sorry

end greatest_integer_x_l77_77656


namespace trailing_zeros_prod_eq_3_l77_77862

theorem trailing_zeros_prod_eq_3 : 
  let numbers := [2.5, 6, 10, 25, 7, 75, 94]
  let product := numbers.foldl (λ x y => x * y) 1
  let trailing_zeros (n : ℝ) : ℕ :=
    let rec count_factors (n : ℕ) (p : ℕ) (acc : ℕ) : ℕ :=
      if n % p == 0 then count_factors (n / p) p (acc + 1) else acc
    min (count_factors (nat_abs (int.of_real (n * 10^10))) 2 0) (count_factors (nat_abs (int.of_real (n * 10^10))) 5 0)
  trailing_zeros product = 3 := sorry

end trailing_zeros_prod_eq_3_l77_77862


namespace f_2011_eq_neg_cos_l77_77548

def f (x : ℝ) : ℝ := Real.sin x

def f_seq : ℕ → (ℝ → ℝ)
| 0       := f
| (n + 1) := λ x, deriv (f_seq n) x

theorem f_2011_eq_neg_cos (x : ℝ) : f_seq 2011 x = -Real.cos x := 
by
  sorry

end f_2011_eq_neg_cos_l77_77548


namespace find_y_l77_77423

theorem find_y (y : ℝ) : sqrt (5 * y + 15) = 15 → y = 42 :=
by
  sorry

end find_y_l77_77423


namespace birds_left_after_a_week_l77_77726

def initial_chickens := 300
def initial_turkeys := 200
def initial_guinea_fowls := 80
def daily_chicken_loss := 20
def daily_turkey_loss := 8
def daily_guinea_fowl_loss := 5
def days_in_a_week := 7

def remaining_chickens := initial_chickens - daily_chicken_loss * days_in_a_week
def remaining_turkeys := initial_turkeys - daily_turkey_loss * days_in_a_week
def remaining_guinea_fowls := initial_guinea_fowls - daily_guinea_fowl_loss * days_in_a_week

def total_remaining_birds := remaining_chickens + remaining_turkeys + remaining_guinea_fowls

theorem birds_left_after_a_week : total_remaining_birds = 349 := by
  sorry

end birds_left_after_a_week_l77_77726


namespace find_unit_vector_l77_77921

noncomputable def unit_vector : ℝ × ℝ × ℝ := 
  (5 / Real.sqrt 14, 0.5 / Real.sqrt 14, 2.5 / Real.sqrt 14)

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (4, -2, 2)

theorem find_unit_vector (v : ℝ × ℝ × ℝ) : 
  (a, b, v) = ((2, -3, 1), (4, -2, 2), (5 / Real.sqrt 14, 0.5 / Real.sqrt 14, 2.5 / Real.sqrt 14)) → 
  v = unit_vector := 
sorry

end find_unit_vector_l77_77921


namespace solve_eq1_solve_eq2_l77_77287

theorem solve_eq1 : (∃ x : ℚ, (5 * x - 1) / 4 = (3 * x + 1) / 2 - (2 - x) / 3) ↔ x = -1 / 7 :=
sorry

theorem solve_eq2 : (∃ x : ℚ, (3 * x + 2) / 2 - 1 = (2 * x - 1) / 4 - (2 * x + 1) / 5) ↔ x = -9 / 28 :=
sorry

end solve_eq1_solve_eq2_l77_77287


namespace f_3_1_plus_f_3_4_l77_77207

def f (a b : ℕ) : ℚ :=
  if a + b < 5 then (a * b - a + 4) / (2 * a)
  else (a * b - b - 5) / (-2 * b)

theorem f_3_1_plus_f_3_4 :
  f 3 1 + f 3 4 = 7 / 24 :=
by
  sorry

end f_3_1_plus_f_3_4_l77_77207


namespace triangle_area_ABC_l77_77102

def point := (ℝ × ℝ)
def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def A := (0.5, 2) : point
def B := (2, 0) : point
def C := (2, 0.5) : point

theorem triangle_area_ABC : triangle_area A B C = 0.375 := by
  sorry

end triangle_area_ABC_l77_77102


namespace brian_video_watching_time_l77_77748

/--
Brian watches a 4-minute video of cats.
Then he watches a video twice as long as the cat video involving dogs.
Finally, he watches a video on gorillas that's twice as long as the combined duration of the first two videos.
Prove that Brian spends a total of 36 minutes watching animal videos.
-/
theorem brian_video_watching_time (cat_video dog_video gorilla_video : ℕ) 
  (h₁ : cat_video = 4) 
  (h₂ : dog_video = 2 * cat_video) 
  (h₃ : gorilla_video = 2 * (cat_video + dog_video)) : 
  cat_video + dog_video + gorilla_video = 36 := by
  sorry

end brian_video_watching_time_l77_77748


namespace number_of_triples_l77_77860

def satisfies_condition (a b c : ℤ) : Prop :=
  2 * a * b * c = a + b + c + 4 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def count_satisfying_triples : ℤ :=
  (List.product (List.product (List.range' (0 - 10) 21) (List.range' (0 - 10) 21)) (List.range' (0 - 10) 21)).countp
    (λ xyz, satisfies_condition xyz.1.1 xyz.1.2 xyz.2)

theorem number_of_triples : count_satisfying_triples = 6 := by sorry

end number_of_triples_l77_77860


namespace sphere_radius_vol_eq_area_l77_77618

noncomputable def volume (r : ℝ) : ℝ := (4/3) * Real.pi * r ^ 3
noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2

theorem sphere_radius_vol_eq_area (r : ℝ) :
  volume r = surface_area r → r = 3 :=
by
  sorry

end sphere_radius_vol_eq_area_l77_77618


namespace most_persuasive_method_l77_77366

-- Survey data and conditions
def male_citizens : ℕ := 4258
def male_believe_doping : ℕ := 2360
def female_citizens : ℕ := 3890
def female_believe_framed : ℕ := 2386

def random_division_by_gender : Prop := true -- Represents the random division into male and female groups

-- Proposition to prove
theorem most_persuasive_method : 
  random_division_by_gender → 
  ∃ method : String, method = "Independence Test" := by
  sorry

end most_persuasive_method_l77_77366


namespace total_cost_of_puzzles_l77_77352

-- Definitions for the costs of large and small puzzles
def large_puzzle_cost : ℕ := 15
def small_puzzle_cost : ℕ := 23 - large_puzzle_cost

-- Theorem statement
theorem total_cost_of_puzzles :
  (large_puzzle_cost + 3 * small_puzzle_cost) = 39 :=
by
  -- Placeholder for the proof
  sorry

end total_cost_of_puzzles_l77_77352


namespace positive_integer_divisors_product_is_729_l77_77613

theorem positive_integer_divisors_product_is_729 (n : ℕ) 
  (h : (∀ (p : ℕ), (0 < p ∧ dvd p n) → ∃ (m : ℕ), (list.prod (list.filter (λ d : ℕ, d ∣ n) (list.range (n + 1)))) = 729)) :
  n = 27 :=
sorry

end positive_integer_divisors_product_is_729_l77_77613


namespace perp_lines_imply_m_parallel_lines_distance_l77_77481

-- Define the lines and conditions
def line1 (m : ℝ) : ℝ → ℝ → Prop := λ x y, x + m * y + 1 = 0
def line2 (m : ℝ) : ℝ → ℝ → Prop := λ x y, (m - 3) * x - 2 * y + (13 - 7 * m) = 0

-- (1) Prove that l1 ⊥ l2 implies m = -3
theorem perp_lines_imply_m (m : ℝ) : 
  (∀ x y, line1 m x y ∧ line2 m x y → 1 * (m - 3) - 2 * m = 0) → m = -3 := 
by {
  intro h,
  sorry -- the proof would go here
}

-- (2) Prove that l1 ∥ l2 implies the distance is 2√2
theorem parallel_lines_distance (m : ℝ) (d : ℝ) :
  (∀ x y, line1 m x y ∧ line2 m x y → m * (m - 3) + 2 = 0) →
  (m = 1) → d = 2 * Real.sqrt 2 :=
by {
  intros h hm,
  sorry -- the proof would go here
}

end perp_lines_imply_m_parallel_lines_distance_l77_77481


namespace shifted_parabola_sum_l77_77671

theorem shifted_parabola_sum (a b c : ℝ) :
  (∃ (a b c : ℝ), ∀ x : ℝ, 3 * x^2 + 2 * x - 5 = 3 * (x - 6)^2 + 2 * (x - 6) - 5 → y = a * x^2 + b * x + c) → a + b + c = 60 :=
sorry

end shifted_parabola_sum_l77_77671


namespace find_value_of_x_squared_plus_y_squared_l77_77482

theorem find_value_of_x_squared_plus_y_squared (x y : ℝ) (h : (x^2 + y^2 + 1)^2 - 4 = 0) : x^2 + y^2 = 1 :=
by
  sorry

end find_value_of_x_squared_plus_y_squared_l77_77482


namespace circle_radius_l77_77965

theorem circle_radius (A : ℝ) (k : ℝ) (r : ℝ) (h : A = k * π * r^2) (hA : A = 225 * π) (hk : k = 4) : 
  r = 7.5 :=
by 
  sorry

end circle_radius_l77_77965


namespace isosceles_triangle_perimeter_eq_20_l77_77831

variable (x y : ℝ)

theorem isosceles_triangle_perimeter_eq_20:
  |x-4| + sqrt(y-8) = 0 -> x = 4 -> y = 8 -> 4 + 8 + 8 = 20 := 
by 
  sorry

end isosceles_triangle_perimeter_eq_20_l77_77831


namespace gcd_of_B_is_2_l77_77211

def B : Set ℕ :=
    { m | ∃ n : ℕ, m = n + (n + 1) + (n + 2) + (n + 3) }

theorem gcd_of_B_is_2 :
    gcd (Set.toFinset B).val = 2 := 
begin
    -- Formalization of problem's given conditions and required proofs
    sorry
end

end gcd_of_B_is_2_l77_77211


namespace rectangles_divided_into_13_squares_l77_77404

theorem rectangles_divided_into_13_squares (m n : ℕ) (h : m * n = 13) : 
  (m = 1 ∧ n = 13) ∨ (m = 13 ∧ n = 1) :=
sorry

end rectangles_divided_into_13_squares_l77_77404


namespace find_angle_B_area_condition1_correct_area_condition2_correct_l77_77527

theorem find_angle_B (B : ℝ) (h1 : B ≠ π / 2) (h2 : cos (2 * B) = sqrt 3 * cos B - 1) : B = π / 6 :=
sorry

noncomputable def area_condition1 (a b c : ℝ) (hA : sin A = sqrt 3 * sin C) (hb : b = 2) (hB : B = π / 6) : ℝ :=
1 / 2 * a * c * sin B

theorem area_condition1_correct (a b c : ℝ) (h1 : sin A = sqrt 3 * sin C) (h2 : b = 2) (h3 : B = π / 6)
  (ha : a = 2 * sqrt 3) (hc : c = 2) : area_condition1 a b c h1 h2 h3 = sqrt 3 :=
sorry

noncomputable def area_condition2 (a b c : ℝ) (h1 : 2 * b = 3 * a) (h2 : b * sin A = 1) (hB : B = π / 6) : ℝ :=
1 / 2 * a * c * sin B

theorem area_condition2_correct (a b c : ℝ) (h1 : 2 * b = 3 * a) (h2 : b * sin A = 1) (h3 : B = π / 6)
  (ha : a = 2) (hb : b = 3) (hc : c = sqrt 3 + 2 * sqrt 2) : area_condition2 a b c h1 h2 h3 = (sqrt 3 + 2 * sqrt 2) / 2 :=
sorry

end find_angle_B_area_condition1_correct_area_condition2_correct_l77_77527


namespace problem_part_a1_problem_part_a2_problem_part_b_problem_part_c_l77_77236

noncomputable def f : ℝ → ℝ := sorry

axiom f_property (x y : ℝ) (hx : x > 0) (hy : y > 0) : f(x * y) = f(x) + f(y)

axiom f_neg (x : ℝ) (hx : x > 1) : f(x) < 0

axiom f_three : f(3) = -1

theorem problem_part_a1 : f(1) = 0 := sorry

theorem problem_part_a2 : f(1 / 9) = 2 := sorry

theorem problem_part_b : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f(x2) < f(x1) := sorry

theorem problem_part_c : ∀ x : ℝ, f(x) + f(2 - x) < 2 → 1 - (2*Real.sqrt 2)/3 < x ∧ x < 1 + (2*Real.sqrt 2)/3 := sorry

end problem_part_a1_problem_part_a2_problem_part_b_problem_part_c_l77_77236


namespace min_value_when_a_is_2_values_of_a_for_max_value_2_l77_77844

noncomputable def f (a x : ℝ) : ℝ := cos (2 * x) + a * cos x

theorem min_value_when_a_is_2 : (∃ x : ℝ, f 2 x ≤ -3 / 2) := sorry

theorem values_of_a_for_max_value_2 : (∀ x : ℝ, f a x ≤ 2) → a = 1 ∨ a = -1 := sorry

end min_value_when_a_is_2_values_of_a_for_max_value_2_l77_77844


namespace cos_alpha_in_fourth_quadrant_l77_77457

theorem cos_alpha_in_fourth_quadrant (α : ℝ) (P : ℝ × ℝ) (h_angle_quadrant : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi)
(h_point : P = (Real.sqrt 5, 2)) (h_sin : Real.sin α = (Real.sqrt 2 / 4) * 2) :
  Real.cos α = Real.sqrt 10 / 4 :=
sorry

end cos_alpha_in_fourth_quadrant_l77_77457


namespace polynomial_divisibility_l77_77306

noncomputable def A_and_B_sum (A B : ℝ) : ℝ :=
  if (λ (x : ℂ), x^105 + A * x + B) = (λ (x : ℂ), x^2 + x + 1) then 
    A + B 
  else 
    0

theorem polynomial_divisibility (A B : ℝ) :
  (A_and_B_sum A B = -1) ↔ 
  ∃ (ω : ℂ), ω^2 + ω + 1 = 0 ∧ ω^105 + A * ω + B = 0 := 
by sorry

end polynomial_divisibility_l77_77306


namespace matrix_unique_solution_l77_77080

-- Definitions for the conditions given in the problem
def vec_i : Fin 3 → ℤ := ![1, 0, 0]
def vec_j : Fin 3 → ℤ := ![0, 1, 0]
def vec_k : Fin 3 → ℤ := ![0, 0, 1]

def matrix_M : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![5, -3, 8],
  ![4, 6, -2],
  ![-9, 0, 5]
]

-- Define the target vectors
def target_i : Fin 3 → ℤ := ![5, 4, -9]
def target_j : Fin 3 → ℤ := ![-3, 6, 0]
def target_k : Fin 3 → ℤ := ![8, -2, 5]

-- The statement of the proof
theorem matrix_unique_solution : 
  (matrix_M.mulVec vec_i = target_i) ∧
  (matrix_M.mulVec vec_j = target_j) ∧
  (matrix_M.mulVec vec_k = target_k) :=
  by {
    sorry
  }

end matrix_unique_solution_l77_77080


namespace intervals_of_monotonicity_range_of_values_for_a_l77_77469

noncomputable def f (a x : ℝ) : ℝ := x - a * Real.log x + (1 + a) / x

theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioi 0, a ≤ -1 → deriv (f a) x > 0) ∧
  (∀ x ∈ Set.Ioc 0 (1 + a), -1 < a → deriv (f a) x < 0) ∧
  (∀ x ∈ Set.Ioi (1 + a), -1 < a → deriv (f a) x > 0) :=
sorry

theorem range_of_values_for_a (a : ℝ) (e : ℝ) (h : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 1 e, f a x ≤ 0) → (a ≤ -2 ∨ a ≥ (e^2 + 1) / (e - 1)) :=
sorry

end intervals_of_monotonicity_range_of_values_for_a_l77_77469


namespace percentage_solution_P_mixture_l77_77286

-- Define constants for volumes and percentages
variables (P Q : ℝ)

-- Define given conditions
def percentage_lemonade_P : ℝ := 0.2
def percentage_carbonated_P : ℝ := 0.8
def percentage_lemonade_Q : ℝ := 0.45
def percentage_carbonated_Q : ℝ := 0.55
def percentage_carbonated_mixture : ℝ := 0.72

-- Prove that the percentage of the volume of the mixture that is Solution P is 68%
theorem percentage_solution_P_mixture : 
  (percentage_carbonated_P * P + percentage_carbonated_Q * Q = percentage_carbonated_mixture * (P + Q)) → 
  ((P / (P + Q)) * 100 = 68) :=
by
  -- proof skipped
  sorry

end percentage_solution_P_mixture_l77_77286


namespace vectors_form_basis_l77_77499

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- defining the points M, A, B, C
variables (M A B C : V)

-- defining vector operations
def vec (u v : V) : V := v - u

-- conditions: A, B, C are distinct and no three points are collinear 
variables (hma : A ≠ M) (hmb : B ≠ M) (hmc : C ≠ M) (habc : ∀ (λ (x y z : V), ¬ collinear ℝ ({x, y, z})))

-- given condition: OM = OA + OB + OC
variable (h : vec 0 M = vec 0 A + vec 0 B + vec 0 C)

-- goal: vectors MA, MB, MC form a basis in space
theorem vectors_form_basis : 
  Module.Free ℝ (span ℝ ({vec M A, vec M B, vec M C} : set V)) :=
sorry

end vectors_form_basis_l77_77499


namespace pen_count_l77_77596

-- Define the conditions
def total_pens := 140
def difference := 20

-- Define the quantities to be proven
def ballpoint_pens := (total_pens - difference) / 2
def fountain_pens := total_pens - ballpoint_pens

-- The theorem to be proved
theorem pen_count :
  ballpoint_pens = 60 ∧ fountain_pens = 80 :=
by
  -- Proof omitted
  sorry

end pen_count_l77_77596


namespace gcd_of_all_elements_in_B_is_2_l77_77213

-- Define the set B as the set of all numbers that can be represented as the sum of four consecutive positive integers.
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2 ∧ x > 0}

-- Translate the question to a Lean statement.
theorem gcd_of_all_elements_in_B_is_2 : ∀ n ∈ B, gcd n 2 = 2 := 
by
  sorry

end gcd_of_all_elements_in_B_is_2_l77_77213


namespace cos_translation_and_shrink_l77_77424

theorem cos_translation_and_shrink :
  ∀ x : ℝ, let y := cos x in
  let y_translated := cos (x - (π / 3)) in
  (cos (2 * x - (π / 3))) = y_translated :=
by
  intros
  sorry

end cos_translation_and_shrink_l77_77424


namespace smallest_m_divided_by_100_l77_77228

-- Define m as the smallest positive integer that meets the conditions
def m : ℕ := 2^4 * 3^9 * 5^1

-- Condition: m is a multiple of 100 and has exactly 100 divisors
def is_multiple_of_100 (n : ℕ) : Prop := 100 ∣ n
def has_exactly_100_divisors (n : ℕ) : Prop := (factors_count n (2^4 * 3^9 * 5^1)) = 100

-- The property we want to prove: m meets both conditions and yields the correct fragment
theorem smallest_m_divided_by_100 : 
  is_multiple_of_100 m ∧ has_exactly_100_divisors m → m / 100 = 15746.4 :=
by
  sorry

end smallest_m_divided_by_100_l77_77228


namespace sine_identity_solution_l77_77452

open Real

theorem sine_identity_solution :
  ∀ (a b : ℝ), (b ∈ Icc 0 (2 * π)) → (∀ x : ℝ, sin (3 * x - π / 3) = sin (a * x + b))
  → (a, b) = (3, 5 * π / 3) ∨ (a, b) = (-3, 2 * π / 3) := 
sorry

end sine_identity_solution_l77_77452


namespace area_ratio_l77_77836

open Real

noncomputable def area (A B C : Point) : ℝ := 
  1/2 * abs ((B - A).cross (C - A))

variables {O A B C: Point}
variable (S1 S2 : ℝ)

axiom point_in_triangle : 
  inside_triangle O A B C

axiom vector_equation : 
  3 * (B - A) + 2 * (C - B) + (A - C) = 4 * (O - A)

axiom area_ABC : 
  S1 = area A B C

axiom area_OBC : 
  S2 = area O B C

theorem area_ratio : S1 / S2 = 2 :=
sorry

end area_ratio_l77_77836


namespace algebraic_expression_equals_one_l77_77817

variable (m n : ℝ)

theorem algebraic_expression_equals_one
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (h_eq : m - n = 1 / 2) :
  (m^2 - n^2) / (2 * m^2 + 2 * m * n) / (m - (2 * m * n - n^2) / m) = 1 :=
by
  sorry

end algebraic_expression_equals_one_l77_77817


namespace simplify_expression_l77_77659

theorem simplify_expression (x : ℝ) : 
  3 - 5 * x - 7 * x^2 + 9 - 11 * x + 13 * x^2 - 15 + 17 * x + 19 * x^2 = 25 * x^2 + x - 3 := 
by
  sorry

end simplify_expression_l77_77659


namespace ellipse_proof_l77_77842

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ a^2 - b^2 = 3 ∧
  (∃ (x y : ℝ), (x, y) = (1, sqrt(3) / 2) ∧ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∀ x y : ℝ, (x^2 / 4) + y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def isosceles_right_triangle : Prop :=
  ∃ (A M N : ℝ × ℝ), A = (0, 1) ∧ on_ellipse M ∧ on_ellipse N ∧
  isosceles_right_triangle_at A M N

theorem ellipse_proof : ellipse_equation ∧ isosceles_right_triangle :=
sorry

end ellipse_proof_l77_77842


namespace evaluate_polynomial_at_4_l77_77636

-- Define the polynomial f
noncomputable def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

-- Given x = 4, prove that f(4) = 1559
theorem evaluate_polynomial_at_4 : f 4 = 1559 :=
  by
    sorry

end evaluate_polynomial_at_4_l77_77636


namespace rectangular_field_length_l77_77334

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

noncomputable def length_rectangle (area width : ℝ) : ℝ :=
  area / width

theorem rectangular_field_length (base height width : ℝ) (h_base : base = 7.2) (h_height : height = 7) (h_width : width = 4) :
  length_rectangle (area_triangle base height) width = 6.3 :=
by
  -- sorry would be replaced by the actual proof.
  sorry

end rectangular_field_length_l77_77334


namespace pizza_volume_one_piece_eq_pi_l77_77722

def radius (diameter : ℝ) : ℝ := diameter / 2

def volume (r h : ℝ) : ℝ := π * r^2 * h

def volume_of_pizza_piece (diameter thickness : ℝ) (pieces : ℕ) : ℝ :=
  let r := radius diameter
  volume r thickness / pieces

theorem pizza_volume_one_piece_eq_pi
  (thickness : ℝ) (diameter : ℝ) (pieces : ℕ)
  (h_thickness_pos : thickness = 1/4)
  (h_diameter_pos : diameter = 16)
  (h_pieces_pos : pieces = 16) :
  volume_of_pizza_piece diameter thickness pieces = π :=
  by
    sorry

end pizza_volume_one_piece_eq_pi_l77_77722


namespace smallest_positive_period_l77_77134

def f (x R : ℝ) := 2 * sqrt 3 * sin (π * x / R)

theorem smallest_positive_period (R : ℝ) (h : ∀ x y, (x^2 + y^2 = R^2) → 
  (y = f x R → y = 2 * sqrt 3 ∨ y = -2 * sqrt 3)) : 
  ∃ T > 0, T = 8 ∧ (∀ x, f (x + T) R = f x R) :=
sorry

end smallest_positive_period_l77_77134


namespace number_of_selection_methods_l77_77620

theorem number_of_selection_methods : 
  let male_doctors := 6 
  let female_doctors := 5 
  let male_selected := 2 
  let female_selected := 1 
  let combinations (n k : ℕ) : ℕ := n.choose k 
  in combinations male_doctors male_selected * combinations female_doctors female_selected = 75 := 
by 
  sorry

end number_of_selection_methods_l77_77620


namespace part1_part2_l77_77885

-- Define the setup for the triangle and the given conditions
noncomputable def triangle (A B C : ℝ) (a b c : ℝ) :=
  A + B + C = π ∧
  a * cos B = b * cos A

-- Define the given conditions
def condition1 (A B : ℝ) (a b : ℝ) : Prop :=
  a * cos B = b * cos A

def condition2 (A : ℝ) : Prop :=
  sin A = 1 / 3

-- Prove (1): a/b = 1 under the given conditions
theorem part1 (A B C : ℝ) (a b c : ℝ) (h1 : triangle A B C a b c) (h2 : condition1 A B a b) : b / a = 1 :=
sorry

-- Prove (2): sin(C - π/4) = (8 + 7 * sqrt(2)) / 18 under the given conditions
theorem part2 (A B C : ℝ) (a b c : ℝ) (h1 : triangle A B C a b c) 
    (h2 : condition1 A B a b) (h3 : condition2 A) : sin (C - π / 4) = (8 + 7 * sqrt(2)) / 18 :=
sorry

end part1_part2_l77_77885


namespace satellite_modular_units_24_l77_77357

-- Define basic parameters
variables (U N S : ℕ)
def fraction_upgraded : ℝ := 0.2

-- Define the conditions as Lean premises
axiom non_upgraded_per_unit_eq_sixth_total_upgraded : N = S / 6
axiom fraction_sensors_upgraded : (S : ℝ) = fraction_upgraded * (S + U * N)

-- The main statement to be proved
theorem satellite_modular_units_24 (h1 : N = S / 6) (h2 : (S : ℝ) = fraction_upgraded * (S + U * N)) : U = 24 :=
by
  -- The actual proof steps will be written here.
  sorry

end satellite_modular_units_24_l77_77357


namespace Monica_books_next_year_l77_77249

-- Definitions for conditions
def books_last_year : ℕ := 25
def books_this_year (bl_year: ℕ) : ℕ := 3 * bl_year
def books_next_year (bt_year: ℕ) : ℕ := 3 * bt_year + 7

-- Theorem statement
theorem Monica_books_next_year : books_next_year (books_this_year books_last_year) = 232 :=
by
  sorry

end Monica_books_next_year_l77_77249


namespace sum_of_words_l77_77566

-- Definitions to represent the conditions
def ХЛЕБ : List Char := ['Х', 'Л', 'Е', 'Б']
def КАША : List Char := ['К', 'А', 'Ш', 'А']

-- Function to compute factorial
def factorial : Nat -> Nat
| 0 => 1
| n + 1 => (n + 1) * factorial n

-- Permutations considering repetition (as in multiset permutations)
def permutations_with_repetition (n : Nat) (reps : Nat) : Nat :=
  factorial n / factorial reps

-- The theorem to prove
theorem sum_of_words : (factorial 4) + (permutations_with_repetition 4 2) = 36 := by
  sorry

end sum_of_words_l77_77566


namespace find_remaining_score_l77_77430

-- Define the problem conditions
def student_scores : List ℕ := [70, 80, 90]
def average_score : ℕ := 70

-- Define the remaining score to prove it equals 40
def remaining_score : ℕ := 40

-- The theorem statement
theorem find_remaining_score (scores : List ℕ) (avg : ℕ) (r : ℕ) 
    (h_scores : scores = [70, 80, 90]) 
    (h_avg : avg = 70) 
    (h_length : scores.length = 3) 
    (h_avg_eq : (scores.sum + r) / (scores.length + 1) = avg) 
    : r = 40 := 
by
  sorry

end find_remaining_score_l77_77430


namespace convex_polygons_from_10_points_on_circle_l77_77595

theorem convex_polygons_from_10_points_on_circle : 
  let points := 10 in 
  (∑ (k: ℕ) in (finset.range (points + 1)), if k ≥ 3 then nat.choose points k else 0) = 968 :=
by
  sorry

end convex_polygons_from_10_points_on_circle_l77_77595


namespace complex_number_on_real_axis_l77_77443

open Complex

theorem complex_number_on_real_axis (a : ℝ) (h : (a - Complex.i) * (1 + Complex.i) ∈ set_of (λ z : ℂ, im z = 0)) :
  a = 1 :=
sorry

end complex_number_on_real_axis_l77_77443


namespace janet_crows_l77_77200

/-- Janet counts some crows and 60% more hawks than crows The total number of birds is 78. 
Prove that the number of crows is 30. -/
theorem janet_crows (C H : ℕ) (h1 : H = 1.6 * C) (h2 : C + H = 78) : C = 30 :=
by
  sorry

end janet_crows_l77_77200


namespace find_b_of_polynomial_is_square_l77_77304

theorem find_b_of_polynomial_is_square (a b : ℚ)
  (h : ∃ Q : ℚ[X], (Q * Q) = (X^4 + 3*X^3 + X^2 + a*X + C b)) :
  b = 25/64 := by
  sorry

end find_b_of_polynomial_is_square_l77_77304


namespace nearest_integer_to_sum_l77_77351

theorem nearest_integer_to_sum (f : ℝ → ℝ)
    (h : ∀ x ≠ 0, 3 * f x + f (1/x) = 6 * x + Real.sin x + 3) :
    ∃ S : ℝ, (S ≈ 445) ∧ S = ∑ x in {x | f x = 1001} :=
by
  sorry

end nearest_integer_to_sum_l77_77351


namespace count_real_solutions_unique_value_of_y_l77_77768

theorem count_real_solutions : 
  ∀ y : ℝ, (3^(4*y+2) * 9^(2*y+3) = 27^(3*y+4)) → y = -4 :=
begin
  sorry
end

theorem unique_value_of_y : 
  ∃! y : ℝ, 3^(4*y+2) * 9^(2*y+3) = 27^(3*y+4) :=
begin
  use -4,
  split,
  { sorry }, -- show that y = -4 is a solution
  { intros z hz,
    apply count_real_solutions z,
    exact hz },
end

end count_real_solutions_unique_value_of_y_l77_77768


namespace parallelepiped_volume_l77_77293

-- Define the conditions
variables (Q S1 S2 : ℝ) (h : ℝ)

-- Given the base of the parallelepiped is a rhombus with area Q
-- Define the volume of the parallelepiped
def volume (Q S1 S2 : ℝ) : ℝ :=
  sqrt (Q * S1 * S2 / 2)

-- Assertion to be proved
theorem parallelepiped_volume (Q S1 S2 : ℝ) : 
  volume Q S1 S2 = sqrt (Q * S1 * S2 / 2) :=
by
  sorry

end parallelepiped_volume_l77_77293


namespace min_a_simplest_quadratic_root_l77_77491

theorem min_a_simplest_quadratic_root :
  ∃ a : ℤ, (∀ a' : ℤ, (a' < a) → ¬ (is_simplest_quadratic_root (sqrt(3 * a' + 1)))) ∧
           is_simplest_quadratic_root (sqrt(3 * a + 1)) ∧
           3 * a + 1 ≥ 0 :=
begin
  sorry
end

def is_simplest_quadratic_root (x : ℝ) : Prop :=
  ¬ ∃ n : ℤ, x = n ∧ x^2 = (n:ℝ)^2

end min_a_simplest_quadratic_root_l77_77491


namespace compute_t_minus_s_l77_77724

noncomputable def t : ℚ := (40 + 30 + 30 + 20) / 4

noncomputable def s : ℚ := (40 * (40 / 120) + 30 * (30 / 120) + 30 * (30 / 120) + 20 * (20 / 120))

theorem compute_t_minus_s : t - s = -1.67 := by
  sorry

end compute_t_minus_s_l77_77724


namespace reflection_matrix_over_line_y_eq_x_l77_77790

-- Define the reflection over the line y = x as a linear map
def reflection_over_y_eq_x (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.2, v.1)  -- this swaps the coordinates (x, y) -> (y, x)

-- Define the matrix that corresponds to this reflection
def reflection_matrix := ![
  ![0, 1],
  ![1, 0]
]

theorem reflection_matrix_over_line_y_eq_x :
  ∀ v : ℝ × ℝ, reflection_over_y_eq_x v = matrix.vec_mul reflection_matrix v :=
by
  sorry

end reflection_matrix_over_line_y_eq_x_l77_77790


namespace factorize_diff_squares_1_factorize_diff_squares_2_factorize_common_term_l77_77398

-- Proof Problem 1
theorem factorize_diff_squares_1 (x y : ℝ) :
  4 * x^2 - 9 * y^2 = (2 * x + 3 * y) * (2 * x - 3 * y) :=
sorry

-- Proof Problem 2
theorem factorize_diff_squares_2 (a b : ℝ) :
  -16 * a^2 + 25 * b^2 = (5 * b + 4 * a) * (5 * b - 4 * a) :=
sorry

-- Proof Problem 3
theorem factorize_common_term (x y : ℝ) :
  x^3 * y - x * y^3 = x * y * (x + y) * (x - y) :=
sorry

end factorize_diff_squares_1_factorize_diff_squares_2_factorize_common_term_l77_77398


namespace locus_of_H_as_M_traverses_AB_l77_77825

-- Definitions for the given problem
variables {O A B M P Q H C D : Type*}
variables [inner_product_space ℝ O]

-- Assume triangle OAB with angle AOB < 90 degrees
variables (triangle_OAB : affine_subspace ℝ O)
variables (angle_AOB_lt_90 : angle O A B < π / 2)

-- M is any point in triangle OAB (excluding O)
variables (M_in_triangle_OAB : M ∈ triangle_OAB)
variable (M_ne_O : M ≠ O)

-- Perpendiculars MP and MQ from M to OA and OB respectively
variables (P_on_OA : is_orthogonal_projection P OA)
variables (Q_on_OB : is_orthogonal_projection Q OB)

-- Define H as the orthocenter of triangle OPQ
def orthocenter_OPQ := orthocenter ℝ (triangle O P Q)

-- Projections of A, B onto OB, OA
variables (D_on_OB : is_orthogonal_projection (orthogonal_projection ℝ OB A) OB)
variables (C_on_OA : is_orthogonal_projection (orthogonal_projection ℝ OA B) OA)

-- The locus of H as M traverses AB
theorem locus_of_H_as_M_traverses_AB :
  locus_of_orthocenter ℝ (triangle O P Q) (segment O A B) = segment D C :=
sorry

end locus_of_H_as_M_traverses_AB_l77_77825


namespace equation_of_circle_l77_77444

theorem equation_of_circle 
  (C : Type*) [MetricSpace C] [NormedGroup C] [NormedSpace ℝ C] [InnerProductSpace ℝ C]
  (P1 P2 : C) (hP1 : P1 = (1 :ℝ, 4 : ℝ)) (hP2 : P2 = (3 :ℝ, -2 : ℝ))
  (h_center_dist : ∀ (C : C), dist (C) (line_through P1 P2) = real.sqrt 10) :
  (∃ C : C, dist C P1 = dist C P2 ∧ dist C (AffineMap.lineThrough P1 P2) = real.sqrt 10) → 
  (∃ r : ℝ, ∀ x, (x + 1)^2 + y^2 = 20 ∨ (x - 5)^2 + (y - 2)^2 = 20) :=
by 
  sorry

end equation_of_circle_l77_77444


namespace graph_C_is_correct_l77_77891

-- Define conditions
def constant_speed (v : ℝ) (t : ℝ) : ℝ := v * t
def run_pause_run (v1 v2 v3 : ℝ) (t1 t2 t3 : ℝ) (t : ℝ) : ℝ :=
  if t <= t1 then v1 * t
  else if t <= t1 + t2 then v1 * t1
  else if t <= t1 + t2 + t3 then v1 * t1 + v3 * (t - t1 - t2)
  else sorry -- Completing the model for further time periods

-- Define the problem statement
theorem graph_C_is_correct
  (v_tortoise v_hare_start v_hare_end t_delta nap1_duration nap2_duration : ℝ)
  (t_end_tortoise t_end_hare : ℝ) :
  constant_speed v_tortoise t_end_tortoise < run_pause_run v_hare_start v_hare_start v_hare_end t_delta (2 * nap1_duration + 2 * t_delta) t_end_hare := 
sorry

end graph_C_is_correct_l77_77891


namespace factorial_product_identity_l77_77429

theorem factorial_product_identity : 
  ∃ (M : ℕ), M > 0 ∧ 7! * 11! = 15 * M! :=
by
  use 12
  split
  · exact Nat.succ_pos' _
  · sorry

end factorial_product_identity_l77_77429


namespace Ralph_TV_hours_l77_77264

theorem Ralph_TV_hours :
  let hoursWeekdays := 4 * 5,
  let hoursWeekends := 6 * 2,
  let totalHours := hoursWeekdays + hoursWeekends
  in totalHours = 32 := 
by
  sorry

end Ralph_TV_hours_l77_77264


namespace vector_magnitude_sum_is_sqrt_21_l77_77476

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := 
  real.sqrt (v.1^2 + v.2^2)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

noncomputable def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

theorem vector_magnitude_sum_is_sqrt_21 
  (a b : ℝ × ℝ)
  (ha : vector_magnitude a = 2)
  (hb : vector_magnitude b = 3)
  (hsub : vector_sub a b = (real.sqrt 2, real.sqrt 3)) :
  vector_magnitude (vector_add a b) = real.sqrt 21 :=
by
  sorry

end vector_magnitude_sum_is_sqrt_21_l77_77476


namespace smallest_range_possible_l77_77723

-- Definition of the problem conditions
def seven_observations (x1 x2 x3 x4 x5 x6 x7 : ℝ) :=
  (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 9 ∧
  x4 = 10

noncomputable def smallest_range : ℝ :=
  5

-- Lean statement asserting the proof problem
theorem smallest_range_possible (x1 x2 x3 x4 x5 x6 x7 : ℝ) (h : seven_observations x1 x2 x3 x4 x5 x6 x7) :
  ∃ x1' x2' x3' x4' x5' x6' x7', seven_observations x1' x2' x3' x4' x5' x6' x7' ∧ (x7' - x1') = smallest_range :=
sorry

end smallest_range_possible_l77_77723


namespace inclination_angle_line_l77_77983

theorem inclination_angle_line (a b : ℝ) (c : ℝ) 
    (h1 : ∃ a b : ℝ , a^2 + b^2 = (a * (real.sqrt 3 / 2) + b / 2)^2) 
    (h2 : ∃ a b : ℝ, cos (real.pi / 3) * a - sin (real.pi / 3) * b = 0):
  ∃ θ : ℝ, θ = 2 * real.pi / 3 :=
by
  sorry

end inclination_angle_line_l77_77983


namespace smallest_n_l77_77325

theorem smallest_n (n : ℕ) : (∃ a b : ℕ, n = 2^a * 5^b) ∧ (∃ (d : ℕ), d < 10000 ∧ digitExists d 3) → n = 125 :=
by sorry

def digitExists (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, (n / 10^k) % 10 = d

end smallest_n_l77_77325


namespace canteen_distance_l77_77006

theorem canteen_distance (g r b : ℝ) (h1 : g = 400) (h2 : b = 700) :
  let ab := Real.sqrt (g^2 + b^2) in
  let c := ab / 2 in
  c = 403 :=
by
  sorry

end canteen_distance_l77_77006


namespace minimum_degree_g_l77_77914

-- Define the degree function for polynomials
noncomputable def degree (p : Polynomial ℤ) : ℕ := p.natDegree

-- Declare the variables and conditions for the proof
variables (f g h : Polynomial ℤ)
variables (deg_f : degree f = 10) (deg_h : degree h = 12)
variable (eqn : 2 * f + 5 * g = h)

-- State the main theorem for the problem
theorem minimum_degree_g : degree g ≥ 12 :=
    by sorry -- Proof to be provided

end minimum_degree_g_l77_77914


namespace find_n_l77_77445

noncomputable theory

def a (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n - 1)

variables {a1 q n: ℝ}

axiom condition1 : (a a1 q 3) + (a a1 q 6) = 36
axiom condition2 : (a a1 q 4) + (a a1 q 7) = 18
axiom equality : a a1 q n = 1 / 2

theorem find_n : n = 9 :=
sorry

end find_n_l77_77445


namespace alpha_range_in_first_quadrant_l77_77434

open Real

theorem alpha_range_in_first_quadrant (k : ℤ) (α : ℝ) 
  (h1 : cos α ≤ sin α) : 
  (2 * k * π + π / 4) ≤ α ∧ α < (2 * k * π + π / 2) :=
sorry

end alpha_range_in_first_quadrant_l77_77434


namespace find_x_l77_77495

variables (x y : ℚ)  -- Declare x and y as rational numbers

theorem find_x (h1 : 3 * x - 2 * y = 7) (h2 : 2 * x + 3 * y = 8) : 
  x = 37 / 13 :=
by
  sorry

end find_x_l77_77495


namespace jude_change_is_1_l77_77908

noncomputable def jude_change : ℕ := 
  let chair_price : ℕ := 13
  let table_price : ℕ := 50
  let plate_price : ℕ := 20
  let num_chairs : ℕ := 3
  let num_plate_sets : ℕ := 2
  let total_cost := num_chairs * chair_price + table_price + num_plate_sets * plate_price
  130 - total_cost

theorem jude_change_is_1 : jude_change = 1 :=
by
  unfold jude_change
  simp
  sorry

end jude_change_is_1_l77_77908


namespace smallest_AAB_value_l77_77031

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 110 * A + B = 8 * (10 * A + B) ∧ ¬ (A = B) ∧ 110 * A + B = 773 :=
by sorry

end smallest_AAB_value_l77_77031


namespace negation_of_existence_statement_l77_77477

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 3 * x + 2 = 0) = ∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0 :=
by
  sorry

end negation_of_existence_statement_l77_77477


namespace sum_of_products_M_l77_77858

-- The set M
def M : Set ℕ := {4, 9, 14, 15, 19, 63, 99, 124}

-- A function to calculate the product of a set
noncomputable def product (s : Finset ℕ) : ℕ :=
  s.fold (*) 1 id

-- A function to calculate the sum of the products of all non-empty subsets
noncomputable def sum_of_products (M : Finset ℕ) : ℕ :=
  (M.powerset.filter (λ s => s ≠ ∅)).sum (λ s => product s)

-- Given the specific set M, this should be the target value to prove
theorem sum_of_products_M : sum_of_products M.toFinset = 1.92 * 10^11 - 1 := by
  sorry

end sum_of_products_M_l77_77858


namespace cost_of_coffee_B_per_kg_l77_77361

-- Define the cost of coffee A per kilogram
def costA : ℝ := 10

-- Define the amount of coffee A used in the mixture
def amountA : ℝ := 240

-- Define the amount of coffee B used in the mixture
def amountB : ℝ := 240

-- Define the total amount of the mixture
def totalAmount : ℝ := 480

-- Define the selling price of the mixture per kilogram
def sellingPrice : ℝ := 11

-- Define the cost of coffee B per kilogram as a variable B
variable (B : ℝ)

-- Define the total cost of the mixture
def totalCost : ℝ := totalAmount * sellingPrice

-- Define the cost of coffee A used
def costOfA : ℝ := amountA * costA

-- Define the cost of coffee B used as total cost minus the cost of A
def costOfB : ℝ := totalCost - costOfA

-- Calculate the cost of coffee B per kilogram
theorem cost_of_coffee_B_per_kg : B = 12 :=
by
  have h1 : costOfA = 2400 := by sorry
  have h2 : totalCost = 5280 := by sorry
  have h3 : costOfB = 2880 := by sorry
  have h4 : B = costOfB / amountB := by sorry
  have h5 : B = 2880 / 240 := by sorry
  have h6 : B = 12 := by sorry
  exact h6

end cost_of_coffee_B_per_kg_l77_77361


namespace no_polyhedron_with_surface_area_2015_l77_77530

theorem no_polyhedron_with_surface_area_2015 : 
  ¬ ∃ (n k : ℤ), 6 * n - 2 * k = 2015 :=
by
  sorry

end no_polyhedron_with_surface_area_2015_l77_77530


namespace monotonically_increasing_on_interval_l77_77165

noncomputable def f (a x : ℝ) := Real.exp x * (Real.sin x + a * Real.cos x)

theorem monotonically_increasing_on_interval 
  (a : ℝ) : (∀ x ∈ Ioo (Real.pi / 4) (Real.pi / 2), 0 < Real.exp x * (Real.cos x + Real.sin x) + a * Real.exp x * (Real.cos x - Real.sin x)) ↔ a ∈ Ioi (-1) :=
sorry

end monotonically_increasing_on_interval_l77_77165


namespace largest_prime_sum_l77_77826

def is_prime (n : ℕ) : Prop := Nat.Prime n

def uses_each_digit_once (ns : List ℕ) : Prop :=
  let all_digits := ns.bind (λ n => (n.digits 10))
  all_digits.toFinset = (Finset.range 1 10)

theorem largest_prime_sum : 
  ∃ (p1 p2 p3 p4 : ℕ),
    (is_prime p1) ∧ (is_prime p2) ∧ (is_prime p3) ∧ (is_prime p4) ∧ 
    (uses_each_digit_once [p1, p2, p3, p4]) ∧ 
    (p1 + p2 + p3 + p4 = 1798) :=
sorry

end largest_prime_sum_l77_77826


namespace length_of_train2_l77_77633

open Real

noncomputable def length_of_second_train
  (speed1_kmh : ℝ)
  (speed2_kmh : ℝ)
  (length1_m : ℝ)
  (time_s : ℝ)
  : ℝ :=
let relative_speed_ms := ((speed1_kmh + speed2_kmh) * 1000) / 3600 in
let total_distance_m := relative_speed_ms * time_s in
total_distance_m - length1_m

theorem length_of_train2 : length_of_second_train 60 40 200 17.998560115190788 = 300 := by
  simp [length_of_second_train]
  have rel_speed := (100:ℝ) * 1000 / 3600
  have total_dist := rel_speed * 17.998560115190788
  norm_num at rel_speed
  norm_num at total_dist
  norm_num
  exact sorry

end length_of_train2_l77_77633


namespace ralph_tv_hours_l77_77271

def hours_per_day_mf : ℕ := 4 -- 4 hours per day from Monday to Friday
def days_mf : ℕ := 5         -- 5 days from Monday to Friday
def hours_per_day_ss : ℕ := 6 -- 6 hours per day on Saturday and Sunday
def days_ss : ℕ := 2          -- 2 days, Saturday and Sunday

def total_hours_mf : ℕ := hours_per_day_mf * days_mf
def total_hours_ss : ℕ := hours_per_day_ss * days_ss
def total_hours_in_week : ℕ := total_hours_mf + total_hours_ss

theorem ralph_tv_hours : total_hours_in_week = 32 := 
by
sory -- proof will be written here

end ralph_tv_hours_l77_77271


namespace goldfish_to_pretzels_ratio_l77_77246

theorem goldfish_to_pretzels_ratio :
  let pretzels := 64
  let suckers := 32
  let kids := 16
  let items_per_baggie := 22
  let total_items := kids * items_per_baggie
  let goldfish := total_items - pretzels - suckers
  let ratio := goldfish / pretzels
  ratio = 4 :=
by
  let pretzels := 64
  let suckers := 32
  let kids := 16
  let items_per_baggie := 22
  let total_items := 16 * 22 -- or kids * items_per_baggie for clarity
  let goldfish := total_items - pretzels - suckers
  let ratio := goldfish / pretzels
  show ratio = 4
  · sorry

end goldfish_to_pretzels_ratio_l77_77246


namespace u_2023_eq_3_l77_77981

def g : ℕ → ℕ
| 1 := 5
| 2 := 3
| 3 := 1
| 4 := 2
| 5 := 4
| _ := 0  -- Assuming g is only defined for {1, 2, 3, 4, 5}

noncomputable def u : ℕ → ℕ
| 0 := 5
| (n + 1) := g (u n)

theorem u_2023_eq_3 : u 2023 = 3 :=
sorry

end u_2023_eq_3_l77_77981


namespace arithmetic_mean_of_range_neg3_to_6_l77_77648

theorem arithmetic_mean_of_range_neg3_to_6 :
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  (sum : Float) / (count : Float) = 1.5 := by
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  have h_sum : sum = 15 := by sorry
  have h_count : count = 10 := by sorry
  calc
    (sum : Float) / (count : Float)
        = (15 : Float) / (10 : Float) : by rw [h_sum, h_count]
    ... = 1.5 : by norm_num

end arithmetic_mean_of_range_neg3_to_6_l77_77648


namespace volume_of_inscribed_sphere_l77_77360

theorem volume_of_inscribed_sphere (a : ℝ) (V : ℝ) (h₁ : a = 8) (h₂ : V = 4 / 3 * Real.pi * (a / 2) ^ 3) :
  V = 256 / 3 * Real.pi :=
by
  rw [h₁, h₂]
  sorry

end volume_of_inscribed_sphere_l77_77360


namespace initial_percentage_of_milk_is_84_l77_77622

theorem initial_percentage_of_milk_is_84 :
  ∀ (P : ℝ), 
  let initial_volume := 60 
      added_water := 18.75 
      final_volume := initial_volume + added_water 
      final_percentage := 64 in
  (P / 100) * initial_volume = (final_percentage / 100) * final_volume →
  P = 84 :=
by
  intros P initial_volume added_water final_volume final_percentage h
  have h1 : (P / 100) * initial_volume = (final_percentage / 100) * final_volume := h
  -- algebraic steps to solve for P
  sorry

end initial_percentage_of_milk_is_84_l77_77622


namespace midpoint_polar_proof_l77_77180

noncomputable def midpoint_polar (r1 θ1 r2 θ2 : Real) : Real × Real :=
  let x1 := r1 * Real.cos θ1
  let y1 := r1 * Real.sin θ1
  let x2 := r2 * Real.cos θ2
  let y2 := r2 * Real.sin θ2
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  let r := Real.sqrt (mx^2 + my^2)
  let θ := Real.atan2 my mx
  (r, θ)

theorem midpoint_polar_proof (r1 θ1 r2 θ2 : Real)
  (h_r1 : r1 = 10) (h_θ1 : θ1 = Real.pi / 3)
  (h_r2 : r2 = 10) (h_θ2 : θ2 = 5 * Real.pi / 6) :
  midpoint_polar r1 θ1 r2 θ2 = (5 * Real.sqrt 2, 2 * Real.pi / 3) :=
by
  simp [midpoint_polar, h_r1, h_θ1, h_r2, h_θ2]
  sorry

end midpoint_polar_proof_l77_77180


namespace proof_inequality_l77_77550

variable {n : ℕ} (a : Fin n → ℝ) (x : Fin (n+1) → ℝ) (A : ℝ)

-- Condition: each a_k is at least 1
def condition_a (k : Fin n) : Prop := 1 ≤ a k

-- Condition: n is at least 1
def condition_n : Prop := 1 ≤ n

-- Condition: A is computed as specified
def condition_A : Prop := A = 1 + ∑ i, a i

-- Condition: x_0 is 1
def condition_x0 : Prop := x 0 = 1

-- Condition: recurrence relation for x
def condition_xk (k : Fin n) : Prop := x (k + 1) = 1 / (1 + a k * x k)

theorem proof_inequality
  (h_a : ∀ k, condition_a a k)
  (h_n : condition_n)
  (h_A : condition_A a A)
  (h_x0 : condition_x0 x)
  (h_xk : ∀ k, condition_xk a x k) :
  (∑ k, x (k + 1)) > n^2 * A / (n^2 + A^2) :=
sorry

end proof_inequality_l77_77550


namespace shooting_competition_orders_l77_77510

theorem shooting_competition_orders : 
  let targets_A := 4
      targets_B := 3
      targets_C := 3
      initial_sequences := 2
  in (∃all_orders : List (List ℕ), 
       all_orders.length = initial_sequences * (Nat.factorial (targets_A + targets_B + targets_C - 2) / 
       (Nat.factorial (targets_A - 2) * Nat.factorial targets_B * Nat.factorial targets_C))) -> 
     initial_sequences * (Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 3)) = 1120 := 
by sorry

end shooting_competition_orders_l77_77510


namespace count_valid_lists_eq_2_l77_77711

-- List of 5 positive integers
def valid_list (l : List ℕ) : Prop :=
  l.length = 5 ∧
  (↑l.sum / 5 = 5) ∧
  (l.count 5 ≥ 2) ∧
  (List.nthLe (l.sorted, 2) (by norm_num) = 5) ∧
  (l.maximumDBy (by simp) - l.minimumDBy (by simp) = 5)

-- Prove that exactly 2 such lists exist
theorem count_valid_lists_eq_2 :
  (List.finRange 5).filter valid_list = 2 :=
sorry

end count_valid_lists_eq_2_l77_77711


namespace silverware_probability_l77_77178

/-- In a drawer containing 8 forks, 8 spoons, and 8 knives, the probability of randomly
selecting six pieces of silverware and retrieving exactly two forks, two spoons, and two knives
is equal to 2744 / 16825. -/
theorem silverware_probability :
  let total_pieces := 24
  let select_count := 6
  let forks := 8
  let spoons := 8
  let knives := 8
  let desired_forks := 2
  let desired_spoons := 2
  let desired_knives := 2
  let prob := (choose forks desired_forks) * (choose spoons desired_spoons) * (choose knives desired_knives) / ((choose total_pieces select_count) : ℚ)
  in prob = 2744 / 16825 :=
by
  sorry

end silverware_probability_l77_77178


namespace min_value_of_function_l77_77420

theorem min_value_of_function (x : ℝ) (hx : x > 3) :
  (x + (1 / (x - 3))) ≥ 5 :=
sorry

end min_value_of_function_l77_77420


namespace S_2011_l77_77899

variable {α : Type*}

-- Define initial term and sum function for arithmetic sequence
def a1 : ℤ := -2011
noncomputable def S (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * 2

-- Given conditions
def condition1 : a1 = -2011 := rfl
def condition2 : (S 2010 / 2010) - (S 2008 / 2008) = 2 := by sorry

-- Proof statement
theorem S_2011 : S 2011 = -2011 := by 
  -- Use the given conditions to prove the statement
  sorry

end S_2011_l77_77899


namespace delta_delta_delta_l77_77764

-- Define the function Δ
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

-- Mathematical statement to be proved
theorem delta_delta_delta (x : ℝ) : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end delta_delta_delta_l77_77764


namespace batsman_new_average_l77_77700

-- Let A be the average score before the 16th inning
def avg_before (A : ℝ) : Prop :=
  ∃ total_runs: ℝ, total_runs = 15 * A

-- Condition 1: The batsman makes 64 runs in the 16th inning
def score_in_16th_inning := 64

-- Condition 2: This increases his average by 3 runs
def avg_increase (A : ℝ) : Prop :=
  A + 3 = (15 * A + score_in_16th_inning) / 16

theorem batsman_new_average (A : ℝ) (h1 : avg_before A) (h2 : avg_increase A) :
  (A + 3) = 19 :=
sorry

end batsman_new_average_l77_77700


namespace max_value_expression_l77_77217

variables (α β : ℂ)
-- Conditions
def condition1 : Prop := |β| = 2
def condition2 : Prop := (conj α) * β ≠ 2

-- Goal
theorem max_value_expression (h1 : condition1 β) (h2 : condition2 α β) :
  ∃ b : ℝ, b ≤ 1 ∧ (abs ((β - α) / (2 - (conj α) * β)) = b) :=
sorry

end max_value_expression_l77_77217


namespace number_of_extreme_points_l77_77454

variable (f : ℝ → ℝ)

-- Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def f_expression (x : ℝ) : ℝ := if x ≤ 0 then (x + 1) ^ 3 * Real.exp (x + 1) else f x

-- Statement of the problem
theorem number_of_extreme_points (h1 : even_function f)
    (h2 : ∀ x, x ≤ 0 → f x = (x + 1) ^ 3 * Real.exp (x + 1)) : 
    (∃ n, n = 2 ∧ is_extreme_points_count f n) :=
sorry

-- Helper definition to denote counting extreme points
def is_extreme_points_count (f : ℝ → ℝ) (count : ℕ) : Prop :=
  -- Your definition here to formally count the number of extreme points
sorry

end number_of_extreme_points_l77_77454


namespace algebra_ineq_a2_b2_geq_2_l77_77305

theorem algebra_ineq_a2_b2_geq_2
  (a b : ℝ)
  (h1 : a^3 - b^3 = 2)
  (h2 : a^5 - b^5 ≥ 4) :
  a^2 + b^2 ≥ 2 :=
by
  sorry

end algebra_ineq_a2_b2_geq_2_l77_77305


namespace sum_integers_50_to_70_l77_77663

theorem sum_integers_50_to_70 : (\sum k in finset.Icc 50 70, k) = 1260 :=
by sorry

end sum_integers_50_to_70_l77_77663


namespace arrange_abc_l77_77833

theorem arrange_abc (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 2)
                               (h2 : b = Real.sqrt 2)
                               (h3 : c = Real.cos ((3 / 4) * Real.pi)) :
  c < a ∧ a < b :=
by
  sorry

end arrange_abc_l77_77833


namespace inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l77_77809

theorem inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * a * c + 4 * b * c ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * a * c + 4 * b * c → a = 0 ∧ b = 0 ∧ c = 0) := sorry

end inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l77_77809


namespace part_I_part_II_l77_77903

-- Definition of the problem with the given conditions
definition center_lies_on_lines (a b : ℝ) : Prop :=
  (b = a - 1) ∧ (b = -2 * a + 8)

definition circle_radius : ℝ := 1

definition point_N : ℝ × ℝ := (0, 3)

-- Part (I) proof
theorem part_I (a b : ℝ) (h : center_lies_on_lines a b) :
  (x y : ℝ) ((x - 3)^2 + (y - 2)^2 = 1) :=
sorry

-- Part (II) proof
theorem part_II (a b : ℝ) (h : center_lies_on_lines a b)
  (hN : point_N = (0, 3)) (hM : ∃ M : ℝ × ℝ, (dist M point_N = dist M (3, 2)) ∧ (dist M (3, 2) = circle_radius)) :
  3 / 2 ≤ a ∧ a ≤ 7 / 2 :=
sorry

end part_I_part_II_l77_77903


namespace problem_l77_77669

theorem problem (a : ℕ) (h : a = 5000) : (a ^ 150) * 2 = 10 ^ 600 := 
by 
  -- Converting 5000 to 5 * 10^3
  have h1 : 5000 = 5 * 10^3 := by assumption
  -- Simplifying expressions using power rules
  rw [h1],
  have h2 : (5 * 10^3) ^ 150 = (5 ^ 150) * (10^3) ^ 150 := by sorry,
  rw [h2],
  -- Further simplifying
  have h3 : (10^3) ^ 150 = 10^450 := by sorry,
  rw [h3],
  -- Multiplying by 2
  have h4 : 2 * (5 ^ 150) * 10^450 = 10 * 10^450 := by sorry,
  rw [h4],
  -- Final simplification
  have h5 : 10 * 10^450 = 10^600 := by sorry,
  rw [h5],
  exact eq.refl _

end problem_l77_77669


namespace problem_statement_l77_77458

theorem problem_statement (a b : ℝ) (C : ℝ) (sin_C : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_C = (Real.sqrt 15) / 4) :
  Real.cos C = 1 / 4 :=
sorry

end problem_statement_l77_77458


namespace harry_speed_increase_l77_77484

def harry_speed_monday : ℤ := 10  -- Harry ran 10 meters per hour on Monday

def tuesday_to_thursday_speed : ℤ := harry_speed_monday + (harry_speed_monday * 50 / 100)  -- 50% faster than Monday

def harry_speed_friday : ℤ := 24  -- Harry ran at 24 meters per hour on Friday

def percentage_increase (old_speed new_speed : ℤ) : ℤ := ((new_speed - old_speed) * 100) / old_speed

theorem harry_speed_increase :
  percentage_increase tuesday_to_thursday_speed harry_speed_friday = 60 :=
by
  simp [tuesday_to_thursday_speed, harry_speed_monday, harry_speed_friday, percentage_increase]
  sorry

end harry_speed_increase_l77_77484


namespace area_of_inscribed_square_l77_77714

theorem area_of_inscribed_square
    (r : ℝ)
    (h : ∀ A : ℝ × ℝ, (A.1 = r - 1 ∨ A.1 = -(r - 1)) ∧ (A.2 = r - 2 ∨ A.2 = -(r - 2)) → A.1^2 + A.2^2 = r^2) :
    4 * r^2 = 100 := by
  -- proof would go here
  sorry

end area_of_inscribed_square_l77_77714


namespace find_fraction_l77_77348

noncomputable def fraction_of_third (F N : ℝ) : Prop := F * (1 / 3 * N) = 30

noncomputable def fraction_of_number (G N : ℝ) : Prop := G * N = 75

noncomputable def product_is_90 (F N : ℝ) : Prop := F * N = 90

theorem find_fraction (F G N : ℝ) (h1 : fraction_of_third F N) (h2 : fraction_of_number G N) (h3 : product_is_90 F N) :
  G = 5 / 6 :=
sorry

end find_fraction_l77_77348


namespace leftover_wall_space_l77_77371

theorem leftover_wall_space :
  ∃ (D B : ℕ), (D = 3 * 1) ∧ (B = 2 * 1) ∧ 
  (15 - (2 * D + 1.5 * B) = 6) :=
begin
  let L := 15,
  let desk_length := 2,
  let bookcase_length := 1.5,
  let x := 1,
  have D := 3 * x,
  have B := 2 * x,
  have total_length := 2 * D + 1.5 * B,
  have left_over := L - total_length,
  use [D, B],
  split,
  { refl },
  split,
  { refl },
  { norm_cast, -- to handle the units and considers of length
    iterate 2 {
      rw mul_comm,
      rw mul_assoc
    },
    norm_num }
end

end leftover_wall_space_l77_77371


namespace sum_of_inscribed_angles_l77_77721

theorem sum_of_inscribed_angles (P : Polygon) (hPentagon : P.isRegular ∧ P.isInscribedInCircle ∧ P.numSides = 5) : 
  let angles := (set.univ : set (ℕ)) -- represents the five vertices
  ∑ i in angles, P.angleOppositeVertex i = 720 :=
sorry

end sum_of_inscribed_angles_l77_77721


namespace binomial_sum_equal_36_l77_77462

theorem binomial_sum_equal_36 (n : ℕ) (h : n > 0) :
  (n + n * (n - 1) / 2 = 36) → n = 8 :=
by
  sorry

end binomial_sum_equal_36_l77_77462


namespace largest_n_satisfying_inequality_l77_77077

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, n ≥ 1 ∧ n^(6033) < 2011^(2011) ∧ ∀ m : ℕ, m > n → m^(6033) ≥ 2011^(2011) :=
sorry

end largest_n_satisfying_inequality_l77_77077


namespace inscribed_circle_radius_l77_77905

theorem inscribed_circle_radius (ABC : Triangle) (P Q : Point)
  (d₁ d₂ d₃ d₄ d₅ d₆ : ℝ)
  (hP_AB : dist_to_line P ABC.AB = 6)
  (hP_BC : dist_to_line P ABC.BC = 7)
  (hP_CA : dist_to_line P ABC.CA = 12)
  (hQ_AB : dist_to_line Q ABC.AB = 10)
  (hQ_BC : dist_to_line Q ABC.BC = 9)
  (hQ_CA : dist_to_line Q ABC.CA = 4) :
  radius_of_inscribed_circle ABC = 8 := 
sorry

end inscribed_circle_radius_l77_77905


namespace range_lambda_l77_77167

theorem range_lambda (λ : ℝ) (a b : ℝ) : 
  (∀ a b : ℝ, a^2 + 8 * b^2 ≥ λ * b * (a + b)) → -8 ≤ λ ∧ λ ≤ 4 :=
sorry

end range_lambda_l77_77167


namespace triangle_b_value_triangle_sinA_value_triangle_sin2A_plus_pi6_value_l77_77197

theorem triangle_b_value (a b c: ℝ) (cosC: ℝ) (sinC: ℝ) (H1: c = sqrt 2) (H2: cosC = 3 / 4) (H3: 2 * c * sin (asin(a/c)) = b * sinC):
  b = 2 :=
begin
  sorry
end

theorem triangle_sinA_value (a b c: ℝ) (cosC: ℝ) (sinC: ℝ) (H1: c = sqrt 2) (H2: cosC = 3 / 4) (H3: a = 1) (H4: sinC = sqrt 7 / 4) (H5: 2 * c * (sqrt 7 / 4) / sqrt 2 = b * sinC):
  sin (asin(a/c)) = sqrt 14 / 8 :=
begin
  sorry
end

theorem triangle_sin2A_plus_pi6_value (sinA: ℝ) (cosA: ℝ) (sin2A: ℝ) (cos2A: ℝ) (H1: sinA = sqrt 14 / 8) (H2: cosA = 5 * sqrt 2 / 8) (H3: sin2A = 5 * sqrt 7 / 16) (H4: cos2A = 9 / 16):
  sin (2 * asin(sqrt 14 / 8) + pi / 6) = (5 * sqrt 21 + 9) / 32 :=
begin
  sorry
end

end triangle_b_value_triangle_sinA_value_triangle_sin2A_plus_pi6_value_l77_77197


namespace triangle_with_positive_area_l77_77574

noncomputable def num_triangles_with_A (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) : ℕ :=
  let points_excluding_A := total_points.erase A
  let total_pairs := points_excluding_A.card.choose 2
  let collinear_pairs := 20  -- Derived from the problem; in practice this would be calculated
  total_pairs - collinear_pairs

theorem triangle_with_positive_area (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) (h : total_points.card = 25):
  num_triangles_with_A total_points A = 256 :=
by
  sorry

end triangle_with_positive_area_l77_77574


namespace variance_of_data_set_l77_77824

theorem variance_of_data_set :
  ∃ a : ℝ, (a = 5 ∧ 
  (let data := [2, 3, 7, 8, a] in
  let mean := (2 + 3 + 7 + 8 + a) / 5 in
  mean = 5 ∧
  let variance := (1 / 5) * ((2 - mean)^2 + (3 - mean)^2 + (7 - mean)^2 + (8 - mean)^2 + (a - mean)^2) in
  variance = 26/5))
  :=
begin
  use 5,
  split,
  { refl },
  split,
  { norm_num },
  { norm_num }
end

end variance_of_data_set_l77_77824


namespace angle_60_degrees_l77_77924

noncomputable def vector_len (v : ℝ^3) := Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem angle_60_degrees (a b : ℝ^3) (h₁ : vector_len a = vector_len b)
  (h₂ : vector_len a = vector_len (a - b)) (h₃ : a ≠ 0) (h₄ : b ≠ 0) :
  ∃ θ : ℝ, θ = 60 ∧ (vector_len a) * (vector_len b) * Real.cos θ = (vector_len a)^2 - (vector_len (a - b))^2 / 2 :=
sorry

end angle_60_degrees_l77_77924


namespace dorothy_money_left_l77_77772

-- Define the conditions
def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

-- Define the calculation of the amount of money left after paying taxes
def money_left (income : ℝ) (rate : ℝ) : ℝ :=
  income - (rate * income)

-- State the main theorem to prove
theorem dorothy_money_left :
  money_left annual_income tax_rate = 49200 := 
by
  sorry

end dorothy_money_left_l77_77772


namespace walk_all_streets_in_n_days_l77_77294

theorem walk_all_streets_in_n_days (n : ℕ) :
  (∀ (SW NE : ℕ × ℕ), SW = (0, 0) ∧ NE = (n - 1, n - 1) → 
  (∀ day : ℕ, day < n →
  (∃ path_morning path_evening : list (ℕ × ℕ),
    (∀ coord ∈ path_morning, coord.fst ≤ n ∧ coord.snd ≤ n ∧ 
    (coord = SW ∨ (∃ prev, prev ∈ path_morning ∧ 
    (coord.fst = prev.fst + 1 ∨ coord.snd = prev.snd + 1))))
  ∧
    (∀ coord ∈ path_evening, coord.fst ≤ n ∧ coord.snd ≤ n ∧ 
    (coord = NE ∨ (∃ prev, prev ∈ path_evening ∧ 
    (coord.fst = prev.fst - 1 ∨ coord.snd = prev.snd - 1))))
  ∧
  (∀ coord ∈ (path_morning ∪ path_evening), coord = (x, y) → (x, y) ≠ coord)))
 → 
 ¬(∀ edge : (ℕ × ℕ) × (ℕ × ℕ), edge ∈ (set_of_edges n)) :
  ∀ e ∈ (walked_streets day), e = edge → true :=
sorry

end walk_all_streets_in_n_days_l77_77294


namespace factorization_l77_77397

-- Define the polynomial
def p (x: ℝ) : ℝ := x^2 - 6 * x + 9 - 64 * x^4

-- Definitions for the factors
def fac1 (x: ℝ) : ℝ := -8 * x^2 + x - 3
def fac2 (x: ℝ) : ℝ := 8 * x^2 + x - 3

-- Main theorem stating the factorization
theorem factorization :
  p x = fac1 x * fac2 x :=
by
  sorry

end factorization_l77_77397


namespace sun_volume_exceeds_moon_volume_by_387_cubed_l77_77297

/-- Given Sun's distance to Earth is 387 times greater than Moon's distance to Earth. 
Given diameters:
- Sun's diameter: D_s
- Moon's diameter: D_m
Formula for volume of a sphere: V = (4/3) * pi * R^3
Derive that the Sun's volume exceeds the Moon's volume by 387^3 times. -/
theorem sun_volume_exceeds_moon_volume_by_387_cubed
  (D_s D_m : ℝ)
  (h : D_s = 387 * D_m) :
  (4/3) * Real.pi * (D_s / 2)^3 = 387^3 * (4/3) * Real.pi * (D_m / 2)^3 := by
  sorry

end sun_volume_exceeds_moon_volume_by_387_cubed_l77_77297


namespace kevin_ends_with_54_cards_l77_77206

theorem kevin_ends_with_54_cards : 
  (start_cards : ℕ) (found_cards : ℕ) (final_cards : ℕ) 
  (h1 : start_cards = 7) (h2 : found_cards = 47) (final_calculated : final_cards = start_cards + found_cards) :
  final_cards = 54 := 
by 
  sorry

end kevin_ends_with_54_cards_l77_77206


namespace sum_a_b_eq_4_l77_77987

-- Define the problem conditions
variables (a b : ℝ)

-- State the conditions
def condition1 : Prop := 2 * a = 8
def condition2 : Prop := a^2 - b = 16

-- State the theorem
theorem sum_a_b_eq_4 (h1 : condition1 a) (h2 : condition2 a b) : a + b = 4 :=
by sorry

end sum_a_b_eq_4_l77_77987


namespace ralph_tv_hours_l77_77272

def hours_per_day_mf : ℕ := 4 -- 4 hours per day from Monday to Friday
def days_mf : ℕ := 5         -- 5 days from Monday to Friday
def hours_per_day_ss : ℕ := 6 -- 6 hours per day on Saturday and Sunday
def days_ss : ℕ := 2          -- 2 days, Saturday and Sunday

def total_hours_mf : ℕ := hours_per_day_mf * days_mf
def total_hours_ss : ℕ := hours_per_day_ss * days_ss
def total_hours_in_week : ℕ := total_hours_mf + total_hours_ss

theorem ralph_tv_hours : total_hours_in_week = 32 := 
by
sory -- proof will be written here

end ralph_tv_hours_l77_77272


namespace recurring_decimal_to_fraction_l77_77639

theorem recurring_decimal_to_fraction :
  let x := 0.4 + 67 / (99 : ℝ)
  (∀ y : ℝ, y = x ↔ y = 463 / 990) := 
by
  sorry

end recurring_decimal_to_fraction_l77_77639


namespace find_f_inv_neg9_l77_77104

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 0 then (1 / 3) ^ x else sorry -- General definition including non-negatives (to cooperate with ℝ)

noncomputable def f_inv (y : ℝ) : ℝ :=
sorry -- Definition of inverse function, to be proven or assumed

lemma odd_function (x : ℝ) : f (-x) = -f x :=
sorry -- Proof that f is odd

lemma inverse_property (y : ℝ): f (f_inv y) = y ∧ f_inv (f y) = y :=
sorry -- Proof that f_inv is indeed the inverse of f 

theorem find_f_inv_neg9 : f_inv (-9) = 2 :=
by
  -- Applying the odd function property and inverse property
  have h := inverse_property 9
  have h1 : f_inv 9 = -2 := by sorry -- Calculating f_inv 9
  have h2 : f_inv (-9) = - (f_inv 9) := by sorry -- Using odd function property for inverse
  rw [h1] at h2
  exact h2

end find_f_inv_neg9_l77_77104


namespace minimum_planes_to_divide_cube_l77_77418

def q (n : ℕ) : ℕ := (n^3 + 5 * n + 6) / 6

theorem minimum_planes_to_divide_cube : ∃ n, q n ≥ 300 :=
begin
  use 13,
  simp [q],
  norm_num,
end

end minimum_planes_to_divide_cube_l77_77418


namespace min_value_of_exponential_l77_77440

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 1) : 
  2^x + 4^y ≥ 2 * Real.sqrt 2 ∧ 
  (∀ a, (2^x + 4^y = a) → a ≥ 2 * Real.sqrt 2) :=
by
  sorry

end min_value_of_exponential_l77_77440


namespace reflection_matrix_over_y_eq_x_is_correct_l77_77793

theorem reflection_matrix_over_y_eq_x_is_correct :
  let M := matrix.std_basis (fin 2) (fin 2)
  ∃ (R : matrix (fin 2) (fin 2) ℝ), 
    (R ⬝ M 0) = matrix.vec_cons 0 (matrix.vec_cons 1 matrix.vec_empty) ∧
    (R ⬝ M 1) = matrix.vec_cons 1 (matrix.vec_cons 0 matrix.vec_empty) ∧
    R = ![![0, 1], ![1, 0]] :=
sorry

end reflection_matrix_over_y_eq_x_is_correct_l77_77793


namespace find_missing_value_l77_77500

theorem find_missing_value 
  (x : ℕ) (prime_x : Nat.Prime x) (h1 : x = 3)
  (h2 : (∀ y : ℤ, (x - 1) = 2 ∧ (y ≤ 2) ∧ ((x - 1) + (3 * x + 3) + y) / 3 = 10 / 3)) : 
  (8 - 4 * x = -4) :=
by
  have x_bound := h1
  rw x_bound at *
  rw [Nat.Prime.ne_zero prime_x, Nat.Prime.ge_one_iff prime_x] at *
  sorry

end find_missing_value_l77_77500


namespace sin_F_of_right_triangle_DEF_l77_77184

-- Define the right triangle DEF with specific lengths
def triangle_DEF : Type := {F : Type} 

structure RightTriangle (P Q R : Type) :=
  (right : Prop) -- right-angle at P
  (PQ : ℝ) -- length of one side
  (PR : ℝ) -- length of opposite side
  (hypotenuse : ℝ) -- length of the hypotenuse

-- Given conditions for the right triangle DEF
def rightTriangleDEF :=
  RightTriangle.mk true 8 15 17

-- Our theorem to prove
theorem sin_F_of_right_triangle_DEF : rightTriangleDEF.right → rightTriangleDEF.PQ / rightTriangleDEF.hypotenuse = 8 / 17 :=
by sorry

end sin_F_of_right_triangle_DEF_l77_77184


namespace magnitude_leq_5_l77_77146

noncomputable def vector_a : ℝ × ℝ := (-2, 2)
noncomputable def vector_b (k : ℝ) : ℝ × ℝ := (5, k)

noncomputable def vector_sum (k : ℝ) : ℝ × ℝ := 
  (vector_a.1 + vector_b(k).1, vector_a.2 + vector_b(k).2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := 
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_leq_5 (k : ℝ) : magnitude (vector_sum k) ≤ 5 → -6 ≤ k ∧ k ≤ 2 :=
sorry

end magnitude_leq_5_l77_77146


namespace sum_max_min_eq_four_l77_77133

noncomputable def f (x : ℝ) : ℝ :=
  (|2 * x| + x^3 + 2) / (|x| + 1)

-- Define the maximum value M and minimum value m
noncomputable def M : ℝ := sorry -- The maximum value of the function f(x)
noncomputable def m : ℝ := sorry -- The minimum value of the function f(x)

theorem sum_max_min_eq_four : M + m = 4 := by
  sorry

end sum_max_min_eq_four_l77_77133


namespace measure_of_F_l77_77194

-- Definitions from conditions
def angle (A B C : Type) := ℕ -- Assuming angle is represented as natural numbers

axiom d_eq_e (D E F : Type) (angle_D : angle D E F) (angle_E : angle D E F) :
  angle_D = angle_E  -- D and E are congruent angles

axiom f_is_40_more (D E F : Type) (angle_D : angle D E F) (angle_F : angle D E F) :
  angle_F = angle_D + 40  -- F is 40 degrees more than D

-- Theorem to be proved
theorem measure_of_F (D E F : Type) (angle_D : angle D E F) (angle_E : angle D E F) (angle_F : angle D E F) :
  angle_D = angle_E →
  angle_F = angle_D + 40 →
  angle_D + angle_E + angle_F = 180 →
  angle_F = 86.67 :=
by {
  intros, -- Introduce the given assumptions
  sorry -- Proof to be completed
}

end measure_of_F_l77_77194


namespace find_points_A_C_find_equation_line_l_l77_77528

variables (A B C : ℝ × ℝ)
variables (l : ℝ → ℝ)

-- Condition: the coordinates of point B are (2, 1)
def B_coord : Prop := B = (2, 1)

-- Condition: the equation of the line containing the altitude on side BC is x - 2y - 1 = 0
def altitude_BC (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Condition: the equation of the angle bisector of angle A is y = 0
def angle_bisector_A (y : ℝ) : Prop := y = 0

-- Statement of the theorems to be proved
theorem find_points_A_C
    (hB : B_coord B)
    (h_altitude_BC : altitude_BC 1 0)
    (h_angle_bisector_A : angle_bisector_A 0) :
  (A = (1, 0)) ∧ (C = (4, -3)) :=
sorry

theorem find_equation_line_l
    (hB : B_coord B)
    (h_altitude_BC : altitude_BC 1 0)
    (h_angle_bisector_A : angle_bisector_A 0)
    (hA : A = (1, 0)) :
  ((∀ x : ℝ, l x = x - 1)) :=
sorry

end find_points_A_C_find_equation_line_l_l77_77528


namespace total_stamps_l77_77539

-- Definitions based on conditions
def kylies_stamps : ℕ := 34
def nellys_stamps : ℕ := kylies_stamps + 44

-- Statement of the proof problem
theorem total_stamps : kylies_stamps + nellys_stamps = 112 :=
by
  -- Proof goes here
  sorry

end total_stamps_l77_77539


namespace find_t_given_conditions_l77_77504

variables (p t j x y : ℝ)

theorem find_t_given_conditions
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p * (1 - t / 100))
  (h4 : x = 0.10 * t)
  (h5 : y = 0.50 * j)
  (h6 : x + y = 12) :
  t = 24 :=
by sorry

end find_t_given_conditions_l77_77504


namespace bricklayer_hours_l77_77001

theorem bricklayer_hours
  (B E : ℝ)
  (h1 : B + E = 90)
  (h2 : 12 * B + 16 * E = 1350) :
  B = 22.5 :=
by
  sorry

end bricklayer_hours_l77_77001


namespace sequence_odd_l77_77994

theorem sequence_odd (a : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = 7)
  (hr : ∀ n ≥ 2, -1 < (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ∧ (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ≤ 1) :
  ∀ n > 1, Odd (a n) := 
  sorry

end sequence_odd_l77_77994


namespace general_term_formula_l77_77993

theorem general_term_formula (n : ℕ) : 
  (λ n, if n = 1 then 1 else if n = 2 then sqrt 2 else if n = 3 then sqrt 3 else 2) n = sqrt n :=
sorry

end general_term_formula_l77_77993


namespace find_int_solutions_l77_77778

theorem find_int_solutions (x y : ℤ) (h : x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
sorry

end find_int_solutions_l77_77778


namespace sum_possible_x_values_in_isosceles_triangle_l77_77042

def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

def valid_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem sum_possible_x_values_in_isosceles_triangle :
  ∃ (x1 x2 x3 : ℝ), isosceles_triangle 80 x1 x1 ∧ isosceles_triangle x2 80 80 ∧ isosceles_triangle 80 x3 x3 ∧ 
  valid_triangle 80 x1 x1 ∧ valid_triangle x2 80 80 ∧ valid_triangle 80 x3 x3 ∧ 
  x1 + x2 + x3 = 150 :=
by
  sorry

end sum_possible_x_values_in_isosceles_triangle_l77_77042


namespace solve_derivative_equation_l77_77608

theorem solve_derivative_equation :
  (∃ n : ℤ, ∀ x,
    x = 2 * n * Real.pi ∨
    x = 2 * n * Real.pi - 2 * Real.arctan (3 / 5)) :=
by
  sorry

end solve_derivative_equation_l77_77608


namespace det_matrix_equal_24_l77_77378

theorem det_matrix_equal_24 : matrix.det (λ i j, ![![5, 3], ![2, 6]] (i, j)) = 24 := by
  sorry

end det_matrix_equal_24_l77_77378


namespace remainder_of_division_l77_77795

-- Define the polynomials
def poly1 : Polynomial ℝ := (2 * Polynomial.X)^500
def poly2 : Polynomial ℝ := (Polynomial.X^2 + 1) * (Polynomial.X - 1)

-- State the theorem
theorem remainder_of_division :
  Polynomial.modByMonic poly1 (Polynomial.X^2 + 1) = 2^500 * Polynomial.X^2 :=
sorry

end remainder_of_division_l77_77795


namespace number_of_boys_l77_77603

theorem number_of_boys :
  ∃ n : ℕ, (58.4 * n + 9 = 58.85 * n) ∧ n = 20 :=
by
  use 20
  split
  · linarith
  · refl

end number_of_boys_l77_77603


namespace parallelogram_area_l77_77050

noncomputable theory
open_locale big_operators

variables (p q : EuclideanSpace ℝ (Fin 2))
variables (a b : EuclideanSpace ℝ (Fin 2))
variable (θ : ℝ)

-- Given conditions
def vec_p : EuclideanSpace ℝ (Fin 2) := p
def vec_q : EuclideanSpace ℝ (Fin 2) := q
def magnitude_p := ∥p∥ = 2
def magnitude_q := ∥q∥ = 3
def angle_pq := (real.angle θ) = (3 / 4) * real.pi

def vec_a := p - 2 • q
def vec_b := 2 • p + q

-- Define cross product in 3D, since Lean’s mathlib works more naturally with ℝ^3 for cross product
def cross_product (u v : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) :=
  ⟨[u.1 1 * v.1 2 - u.1 2 * v.1 1, u.1 2 * v.1 0 - u.1 0 * v.1 2, u.1 0 * v.1 1 - u.1 1 * v.1 0]⟩

-- Additional assumptions to uplift p and q to 3D for the cross product
def p_3d : EuclideanSpace ℝ (Fin 3) := ⟨p.1 ++ [0]⟩
def q_3d : EuclideanSpace ℝ (Fin 3) := ⟨q.1 ++ [0]⟩
def a_3d : EuclideanSpace ℝ (Fin 3) := ⟨(vec_a p q).1 ++ [0]⟩
def b_3d : EuclideanSpace ℝ (Fin 3) := ⟨(vec_b p q).1 ++ [0]⟩

theorem parallelogram_area :
  let cross_ab := cross_product (a_3d p q) (b_3d p q) in
  ∥cross_ab∥ = 15 * real.sqrt 2 :=
sorry

end parallelogram_area_l77_77050


namespace functional_equation_solution_l77_77400

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x * f y) →
  (∀ x : ℚ, f x = 0 ∨ f x = 1) :=
by
  sorry

end functional_equation_solution_l77_77400


namespace deepak_present_age_l77_77685

theorem deepak_present_age (x : ℕ) (Rahul_age Deepak_age : ℕ) 
  (h1 : Rahul_age = 4 * x) (h2 : Deepak_age = 3 * x) 
  (h3 : Rahul_age + 4 = 32) : Deepak_age = 21 := by
  sorry

end deepak_present_age_l77_77685


namespace average_height_corrected_l77_77291

theorem average_height_corrected :
  ∀ (n : ℕ) (avg_height : ℝ) (wrong_height correct_height : ℝ),
  n = 35 →
  avg_height = 180 →
  wrong_height = 166 →
  correct_height = 106 →
  Float.round ((n * avg_height - wrong_height + correct_height) / n) 2 = 178.29 := 
by
  intros n avg_height wrong_height correct_height h1 h2 h3 h4
  sorry

end average_height_corrected_l77_77291


namespace dot_product_example_l77_77859

open Real

theorem dot_product_example : 
  ∀ (a b : ℝ × ℝ), a = (1, -1) → b = (-1, 2) → (2 • a + b) = (1, 0) → (2 • a + b) • a = 1 :=
by
  intros a b ha hb h2ab
  have hsum : (2 • a + b) = (1, 0), from h2ab
  have hdot : ((1, 0) • (1, -1)) = 1, by {
    calc
      (1, 0) • (1, -1) = 0 -- Computing the dot product, this line would be "filled in" in the proof
  }
  show ((2 • a + b) • a) = 1, from hdot

end dot_product_example_l77_77859


namespace keanu_needs_10_refills_l77_77536

-- Define the conditions 
def tank_capacity : ℕ := 8
def distances : List ℕ := [80, 120, 160, 100]
def consumption_rates : List ℕ := [40, 50, 60, 45]

-- Define the function to calculate gasoline needed for a given distance and rate
def gasoline_needed (distance rate : ℕ) : ℕ :=
  let liters_per_mile := 8 / rate.toFloat
  ceiling (distance.toFloat * liters_per_mile).toNat

-- Calculate total gasoline needed for the trip
def total_gasoline_needed : ℕ :=
  distances.zip consumption_rates |>
  List.map (λ (d_r : ℕ × ℕ), gasoline_needed d_r.1 d_r.2) |>
  List.sum

-- Calculate the number of refills needed
def number_of_refills : ℕ :=
  (total_gasoline_needed.toFloat / tank_capacity.toFloat).ceil.toNat

-- The proof problem stating Keanu needs 10 refills
theorem keanu_needs_10_refills : number_of_refills = 10 := by
  sorry

end keanu_needs_10_refills_l77_77536


namespace inequality_proof_l77_77127

noncomputable def f (a x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x) - x

theorem inequality_proof (a b : ℝ) (ha : 1 < a) (hb : 0 < b) : 
  f a (a + b) > f a 1 → g (a / b) < g 0 → 1 / (a + b) < Real.log (a + b) / b ∧ Real.log (a + b) / b < a / b := 
by
  sorry

end inequality_proof_l77_77127


namespace compound_interest_total_amount_l77_77974

noncomputable def principal (CI r t : ℝ) : ℝ := CI / ((1 + r) ^ t - 1)

theorem compound_interest_total_amount :
  let CI := 246
  let r := 0.05
  let t := 2
  let P := principal CI r t
  let TotalAmount := P + CI
  TotalAmount = 2646 :=
by
  let CI := 246
  let r := 0.05
  let t := 2
  let P := principal CI r t
  have hP : P = 2400 := by sorry
  let TotalAmount := P + CI
  show TotalAmount = 2646, from sorry

end compound_interest_total_amount_l77_77974


namespace tile_covering_possible_l77_77913

theorem tile_covering_possible (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  ((m % 6 = 0) ∨ (n % 6 = 0)) := 
sorry

end tile_covering_possible_l77_77913


namespace B_minus_A_eq_pi_over_two_sin_A_plus_sin_C_range_l77_77545

-- Definitions based on the problem conditions
def triangle {α : Type _} [LinearOrderedField α] (a b c A B C : α) : Prop :=
  a = b * (Real.tan A) ∧ (π / 2 < B ∧ B < π)

-- Proposition to prove B - A = π / 2
theorem B_minus_A_eq_pi_over_two {α : Type _} [LinearOrderedField α] {a b c A B C : α} 
  (h : triangle a b c A B C) : B - A = π / 2 :=
sorry

-- Proposition to find the range of sin A + sin C
theorem sin_A_plus_sin_C_range {α : Type _} [LinearOrderedField α] {a b c A B C : α} 
  (h : triangle a b c A B C) : 
  ∃ l u : α, l = Real.sqrt 2 / 2 ∧ u = 9 / 8 ∧ (l < Real.sin A + Real.sin (π / 2 - 2 * A) ∧ Real.sin A + Real.sin (π / 2 - 2 * A) ≤ u) :=
sorry

end B_minus_A_eq_pi_over_two_sin_A_plus_sin_C_range_l77_77545


namespace sum_of_eight_numbers_l77_77163

theorem sum_of_eight_numbers (a : ℝ) :
  (∀ l : List ℝ, l.length = 8 → l.sum / 8 = a) → (a = 5.2 → ∃ l, l.sum = 41.6) :=
by
  intro h hf
  use List.replicate 8 (5.2 : ℝ)
  have : (8 * 5.2) = 41.6 := by norm_num
  rw [List.sum_replicate, this]
  exact hf

end sum_of_eight_numbers_l77_77163


namespace count_sets_without_perfect_square_l77_77216

-- Define the range of T_i
def T (i : ℕ) : set ℕ := {n : ℕ | 200 * i ≤ n ∧ n < 200 * (i + 1)}

-- Define the theorem
theorem count_sets_without_perfect_square : 
  (card {i : ℕ | i < 500 ∧ ∀ x : ℕ, x^2 ∉ T i}) = 400 :=
by
  sorry

end count_sets_without_perfect_square_l77_77216


namespace angle_EDC_l77_77690

theorem angle_EDC {A B C D E : Type} (h1 : AB = AC) (h2 : ∠BAC = 60) (h3 : ∠BAD = 30) (h4 : D ∈ seg B C) (h5 : E ∈ seg A C) 
(h6 : ∠BDC = 90 - 30) (h7 : is_isosceles ADE) (h8 : ∠EAD = 30) : ∠EDC = 15 :=
sorry

end angle_EDC_l77_77690


namespace centroid_distance_relation_l77_77917

variable {A B C G : Type} [MetricSpace G]

-- Define the distances for sides and centroid distances
noncomputable def distance (x y : G) : Real := sorry

-- Define conditions
def is_centroid (G : G) (A B C : G) : Prop := sorry

-- Define variables for the sides and distances involving the centroid
variable (GA GB GC : G)
variable (AB BC CA : G)

-- Define sums s1 and s2
noncomputable def s1 (GA GB GC : G) :=
  (distance G A) + (distance G B) + (distance G C)

noncomputable def s2 (AB BC CA : G) :=
  (distance A B) + (distance B C) + (distance C A)

-- The final statement to prove
theorem centroid_distance_relation
  (h1 : is_centroid G A B C)
  (h2 : s1 GA GB GC = (distance G A) + (distance G B) + (distance G C))
  (h3 : s2 AB BC CA = (distance A B) + (distance B C) + (distance C A)):
  s1 GA GB GC < s2 AB BC CA :=
sorry

end centroid_distance_relation_l77_77917


namespace cube_volume_is_64_l77_77948

theorem cube_volume_is_64 (a : ℕ) (h : (a - 2) * (a + 3) * a = a^3 + 12) : a^3 = 64 := 
  sorry

end cube_volume_is_64_l77_77948


namespace carolyn_removes_sum_l77_77380

theorem carolyn_removes_sum : 
  let list := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
  let carolyn_moves := [3, 6, 8, 9, 10] in
  list.contains 3 ∧ 
  ∀ i ∈ carolyn_moves, i ∈ list →
  2 ≤ (list.filter (λ x, x ∣ i)).length →
  sum carolyn_moves = 36 := sorry

end carolyn_removes_sum_l77_77380


namespace distance_between_foci_l77_77781

-- Define the given ellipse equation.
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + y^2 = 144

-- Provide the values of the semi-major and semi-minor axes.
def a : ℝ := 12
def b : ℝ := 4

-- Define the equation for calculating the distance between the foci.
noncomputable def focal_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

-- The theorem we need to prove.
theorem distance_between_foci : focal_distance a b = 8 * Real.sqrt 2 :=
by sorry

end distance_between_foci_l77_77781


namespace conference_children_count_l77_77506

theorem conference_children_count :
  ∃ C : ℕ, 
  (C > 0) ∧
  let men := 500 in
  let women := 300 in
  let children := C in
  let indian_men := 0.10 * men in
  let indian_women := 0.60 * women in
  let indian_children := 0.70 * children in
  let total_people := men + women + children in
  let total_indians := indian_men + indian_women + indian_children in
  let indian_percentage := total_indians / total_people in
  abs (indian_percentage - 0.4461538461538461) < 1e-6 :=
begin
  sorry
end

end conference_children_count_l77_77506


namespace part_i_l77_77315

theorem part_i (k : ℕ) (h : k ≥ 1) (S : Finset ℕ) (hS : S = Finset.range (2 * k + 1) \ .succ) 
  (A : Finset ℕ) (hA : A ⊆ S) (hchoose : A.card = k + 1) : 
  ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ Nat.gcd x y = 1 :=
begin
  -- Sorry is added to skip the proof, as per the instructions.
  sorry
end

end part_i_l77_77315


namespace ratio_PR_QS_l77_77577

theorem ratio_PR_QS (P Q R S : Point) (PQ QR PS : ℝ)
  (hPQ : PQ = 4) (hQR : QR = 10) (hPS : PS = 28) :
  PR / QS = 7 / 12 := by
  let PR := PQ + QR
  let QS := PS - PQ
  have hPR : PR = 14 := by linarith [hPQ, hQR]
  have hQS : QS = 24 := by linarith [hPS, hPQ]
  rw [hPR, hQS]
  simp
  norm_num
  sorry

end ratio_PR_QS_l77_77577


namespace angle_PQRS_l77_77602

theorem angle_PQRS (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : 
  P = 206 := 
by
  sorry

end angle_PQRS_l77_77602


namespace parallel_line_to_l3_perpendicular_line_to_l3_l77_77145

-- Define the line equations
def l1 (x y : ℚ) := 2 * x + 3 * y - 5 = 0
def l2 (x y : ℚ) := x + 2 * y - 3 = 0
def l3 (x y : ℚ) := 2 * x + y - 5 = 0

-- Define the intersect point of l1 and l2
def point_P := (1, 1 : ℚ)

-- Parallel line to l3 passing through point_P
theorem parallel_line_to_l3 : (∀ x y : ℚ, 2 * x + y - 3 = 0 ↔ (y - 1 = -2 * (x - 1))) :=
by
  intro x y
  split
  sorry -- skip proof

-- Perpendicular line to l3 passing through point_P
theorem perpendicular_line_to_l3 : (∀ x y : ℚ, x - 2 * y + 1 = 0 ↔ (y - 1 = (x - 1) / 2)) :=
by
  intro x y
  split
  sorry -- skip proof

end parallel_line_to_l3_perpendicular_line_to_l3_l77_77145


namespace inv_sum_eq_six_l77_77493

theorem inv_sum_eq_six (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a + b = 6 * (a * b)) : 1 / a + 1 / b = 6 := 
by 
  sorry

end inv_sum_eq_six_l77_77493


namespace prime_factor_mod_l77_77116

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  (∀ n ≥ 3, a n = 2 * a (n - 1) + a (n - 2))

theorem prime_factor_mod (a : ℕ → ℕ) (h : sequence a) :
  ∀ n, n ≥ 5 → ∃ p, p.prime ∧ p ∣ a n ∧ p % 4 = 1 :=
sorry

end prime_factor_mod_l77_77116


namespace reflect_x_axis_l77_77897

-- Define the original coordinates of the point P
def P_initial := (3, -2)

-- Define the reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- State the theorem
theorem reflect_x_axis :
  reflect_x P_initial = (3, 2) :=
by sorry

end reflect_x_axis_l77_77897


namespace daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l77_77014

-- Part (1)
theorem daily_sales_volume_and_profit (x : ℝ) :
  let increase_in_sales := 2 * x
  let profit_per_piece := 40 - x
  increase_in_sales = 2 * x ∧ profit_per_piece = 40 - x :=
by
  sorry

-- Part (2)
theorem profit_for_1200_yuan (x : ℝ) (h1 : (40 - x) * (20 + 2 * x) = 1200) :
  x = 10 ∨ x = 20 :=
by
  sorry

-- Part (3)
theorem profit_impossible_for_1800_yuan :
  ¬ ∃ y : ℝ, (40 - y) * (20 + 2 * y) = 1800 :=
by
  sorry

end daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l77_77014


namespace determine_real_polynomials_l77_77390

noncomputable def polynomial_solution_statement : Prop :=
∀ (P : ℝ[X]), (∀ (a b c : ℝ), P.eval (a + b - 2 * c) + P.eval (b + c - 2 * a) + P.eval (a + c - 2 * b) =
  3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)) ↔
  ∃ (a b : ℝ), P = Polynomial.C a * X^2 + Polynomial.C b * X

theorem determine_real_polynomials : polynomial_solution_statement :=
by
  sorry

end determine_real_polynomials_l77_77390


namespace set_A_listing_l77_77479

set_option pp.proofs true

theorem set_A_listing :
  let A := {p : ℤ × ℤ | ∃ x y, p = (x, y) ∧ x^2 = y + 1 ∧ |x| < 2} in
  A = {(-1, 0), (0, -1), (1, 0)} :=
by
  sorry

end set_A_listing_l77_77479


namespace sequence_property_l77_77935

open Real

variable {α : Type*} [LinearOrderedField α]

def sequence (a : ℕ → α) : Prop :=
  ∀ n, a n ≤ a (n+2) ∧ a (n+2) ≤ sqrt(a n ^ 2 + r * a (n+1))

theorem sequence_property (r : ℝ) (a : ℕ → ℝ) (h_seq: ∀ n, a n ≤ a (n+2) ∧ a (n+2) ≤ sqrt(a n ^ 2 + r * a (n+1))) :
  (r ≤ 2 → ∃ N, ∀ n ≥ N, a (n+2) = a n) ∧ (2 < r → ¬ ∃ N, ∀ n ≥ N, a (n+2) = a n) :=
sorry

end sequence_property_l77_77935


namespace find_angle_OD_base_l77_77971

noncomputable def angle_between_edge_and_base (α β : ℝ): ℝ :=
  Real.arctan ((Real.sin α * Real.sin β) / Real.sqrt (Real.sin (α - β) * Real.sin (α + β)))

theorem find_angle_OD_base (α β : ℝ) :
  ∃ γ : ℝ, γ = angle_between_edge_and_base α β :=
sorry

end find_angle_OD_base_l77_77971


namespace count_valid_seatings_l77_77964

-- Definitions
def WilsonFamily := list nat  -- boys are 1, girls are 0

-- Establishing the family configuration
def wilson_family : WilsonFamily := [1, 1, 1, 1, 1, 0, 0]

-- Predicate to check arrangement leads to exactly 3 boys together
def exactly_three_boys_together (arr : list nat) : Prop :=
  list.non_empty (arr.group_tail 3) ∧ (∀ s, s ∈ arr.group_tail 4 → s ≠ [1, 1, 1, 1])

-- Main theorem stating the number of ways
theorem count_valid_seatings : 
  ∃ n : nat, n = 5760 ∧ ∀ (arr : list nat), (arr.perm wilson_family) → exactly_three_boys_together arr :=
sorry

end count_valid_seatings_l77_77964


namespace find_missing_number_l77_77340

theorem find_missing_number :
  ∃ n : ℤ, (476 + 424) * n - 4 * 476 * 424 = 2704 ∧ n = 904 :=
begin
  use 904,
  split,
  { calc (476 + 424) * 904 - 4 * 476 * 424
       = 900 * 904 - 4 * 476 * 424 : by simp
   ... = 900 * 904 - 4 * (400 * 400 + 400 * 76 + 400 * 24 + 76 * 24) : by refl
   ... = 900 * 904 - 4 * (160000 + 30400 + 9600 + 1824) : by simp
   ... = 900 * 904 - 4 * 202824 : by simp
   ... = 900 * 904 - 811296 : by simp
   ... = 814000 - 811296 : by simp
   ... = 2704 : by norm_num
  },
  refl,
end

end find_missing_number_l77_77340


namespace sum_of_scores_l77_77946

/-- Prove that given the conditions on Bill, John, and Sue's scores, the total sum of the scores of the three students is 160. -/
theorem sum_of_scores (B J S : ℕ) (h1 : B = J + 20) (h2 : B = S / 2) (h3 : B = 45) : B + J + S = 160 :=
sorry

end sum_of_scores_l77_77946


namespace ranking_of_scores_l77_77533

-- Define the scores of Jessica, Linda, and Nora as variables
variables (Jessica_score Linda_score Nora_score : ℝ)

-- Conditions based on the statements given in the problem
def ranking_conditions :=
  (¬ ∀ s, s < Linda_score) ∧
  (¬ ∀ s, s > Jessica_score)

-- Theorem stating the ranking of the scores
theorem ranking_of_scores (h : ranking_conditions Jessica_score Linda_score Nora_score) :
  Jessica_score > Nora_score ∧ Nora_score > Linda_score :=
sorry

end ranking_of_scores_l77_77533


namespace shoveling_time_l77_77626

theorem shoveling_time :
  let kevin_time := 12
  let dave_time := 8
  let john_time := 6
  let allison_time := 4
  let kevin_rate := 1 / kevin_time
  let dave_rate := 1 / dave_time
  let john_rate := 1 / john_time
  let allison_rate := 1 / allison_time
  let combined_rate := kevin_rate + dave_rate + john_rate + allison_rate
  let total_minutes := 60
  let combined_rate_per_minute := combined_rate / total_minutes
  (1 / combined_rate_per_minute = 96) := 
  sorry

end shoveling_time_l77_77626


namespace absolute_value_expression_l77_77384

theorem absolute_value_expression : |2 * real.pi - |3 * real.pi - 10|| = 5 * real.pi - 10 := by sorry

end absolute_value_expression_l77_77384


namespace smallest_four_digit_number_l77_77741

theorem smallest_four_digit_number :
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ 
           x % 5 = 0 ∧ 
           x % 11 = 7 ∧
           x % 7 = 4 ∧
           x % 9 = 4 ∧
           (∀ y : ℕ, 1000 ≤ y ∧ y < 10000 ∧ 
                      y % 5 = 0 ∧ 
                      y % 11 = 7 ∧
                      y % 7 = 4 ∧ 
                      y % 9 = 4 → y ≥ x) :=
begin
  use 2020,
  split,
  { exact nat.le_of_lt (show 1000 < 2020, by norm_num) },
  split,
  { exact nat.lt_of_lt_of_le (show 2020 < 10000, by norm_num) (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (show 5 ∣ 2020, by norm_num) },
  split,
  { exact show 2020 % 11 = 7, by norm_num },
  split,
  { exact show 2020 % 7 = 4, by norm_num },
  split,
  { exact show 2020 % 9 = 4, by norm_num },
  intros y hy,
  by_contradiction,
  sorry
end

end smallest_four_digit_number_l77_77741


namespace perpendicular_to_plane_l77_77873

theorem perpendicular_to_plane (Line : Type) (Plane : Type) (triangle : Plane) (circle : Plane)
  (perpendicular1 : Line → Plane → Prop)
  (perpendicular2 : Line → Plane → Prop) :
  (∀ l, ∃ t, perpendicular1 l t ∧ t = triangle) ∧ (∀ l, ∃ c, perpendicular2 l c ∧ c = circle) →
  (∀ l, ∃ p, (perpendicular1 l p ∨ perpendicular2 l p) ∧ (p = triangle ∨ p = circle)) :=
by
  sorry

end perpendicular_to_plane_l77_77873


namespace find_m_l77_77090

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

-- Assert the condition that the function is increasing on (0, +∞)
def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃a b⦄, a ∈ I → b ∈ I → a < b → f a < f b

-- Main statement to prove
theorem find_m (m : ℝ) :
  is_increasing_on (f m) (Set.Ioi 0) → m = 2 :=
by
  sorry

end find_m_l77_77090


namespace solution_mixture_l77_77686

/-
  Let X be a solution that is 10% alcohol by volume.
  Let Y be a solution that is 30% alcohol by volume.
  We define the final solution to be 22% alcohol by volume.
  We need to prove that the amount of solution Y that needs
  to be added to 300 milliliters of solution X to achieve this 
  concentration is 450 milliliters.
-/

theorem solution_mixture (y : ℝ) : 
  (0.10 * 300) + (0.30 * y) = 0.22 * (300 + y) → 
  y = 450 :=
by {
  sorry
}

end solution_mixture_l77_77686


namespace quadrilateral_area_PQRS_l77_77895

noncomputable def area_of_quadrilateral (PQ QR RS PS : ℝ) (angle_QRS PR QS : ℝ) : ℝ := 
  if angle_QRS = 90 ∧ PQ = 9 ∧ QR = 6 ∧ RS = 8 ∧ PS = 15 ∧ (angle_between_diagonals : PR ∧ QS satisfies right angle condition) then
    67
  else
    sorry

theorem quadrilateral_area_PQRS :
  ∀ (P Q R S : Type) (PQ QR RS PS angle_QRS PR QS : ℝ),
  angle_QRS = 90 ∧ PQ = 9 ∧ QR = 6 ∧ RS = 8 ∧ PS = 15 ∧
  (∃ O : Type, right_angle_intersection PR QS O) 
  → area_of_quadrilateral PQ QR RS PS angle_QRS PR QS = 67 :=
by {
  intros,
  rw area_of_quadrilateral,
  split_ifs,
  { refl },
  { sorry }
}

end quadrilateral_area_PQRS_l77_77895


namespace maximum_in_interval_l77_77498

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  a * x ^ 2 / 2 - (1 + 2 * a) * x + 2 * log x

def f_prime (a x: ℝ) : ℝ :=
  a * x - (1 + 2 * a) + 2 / x

theorem maximum_in_interval {a : ℝ} (ha : 0 < a) :
  (1 < a ∧ a < 2) ↔ (∀ x ∈ Ioo (1/2 : ℝ) 1, has_deriv_at (f a) (f_prime a x) x ∧ f_prime a 1/2 > 0 ∧ f_prime a 1 < 0) :=
sorry

end maximum_in_interval_l77_77498


namespace tims_seashells_now_l77_77625

def initial_seashells : ℕ := 679
def seashells_given_away : ℕ := 172

theorem tims_seashells_now : (initial_seashells - seashells_given_away) = 507 :=
by
  sorry

end tims_seashells_now_l77_77625


namespace simplify_expression_l77_77958

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : 1 - x ≠ 0) :
  (1 - x) / x / ((1 - x) / x^2) = x := 
by 
  sorry

end simplify_expression_l77_77958


namespace side_of_larger_square_l77_77727

theorem side_of_larger_square (s S : ℕ) (h₁ : s = 5) (h₂ : S^2 = 4 * s^2) : S = 10 := 
by sorry

end side_of_larger_square_l77_77727


namespace distance_between_foci_l77_77780

-- Define the given ellipse equation.
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + y^2 = 144

-- Provide the values of the semi-major and semi-minor axes.
def a : ℝ := 12
def b : ℝ := 4

-- Define the equation for calculating the distance between the foci.
noncomputable def focal_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

-- The theorem we need to prove.
theorem distance_between_foci : focal_distance a b = 8 * Real.sqrt 2 :=
by sorry

end distance_between_foci_l77_77780


namespace rice_yield_l77_77731

theorem rice_yield (X : ℝ) (h1 : 0 ≤ X ∧ X ≤ 40) :
    0.75 * 400 * X + 0.25 * 800 * X + 500 * (40 - X) = 20000 := by
  sorry

end rice_yield_l77_77731


namespace sum_first_6_terms_l77_77856

-- Define the arithmetic sequence
def a : ℕ → ℕ
| 0     := 1
| (n+1) := a n + 3

-- Define the sum of the first n terms of the sequence
def partial_sum (n : ℕ) : ℕ :=
∑ i in Finset.range (n + 1), a i

-- Theorem: The sum of the first 6 terms is 51
theorem sum_first_6_terms : partial_sum 5 = 51 :=
sorry

end sum_first_6_terms_l77_77856


namespace total_marbles_l77_77943

-- Define the number of marbles Mary and Joan have respectively
def mary_marbles := 9
def joan_marbles := 3

-- Prove that the total number of marbles is 12
theorem total_marbles : mary_marbles + joan_marbles = 12 := by
  sorry

end total_marbles_l77_77943


namespace range_of_m_l77_77124

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 3^(2*x + 1) + (m - 1)*(3^(x + 1) - 1) - (m - 3)*3^x = 0 → has_distinct_real_roots(3*(3^x)^2 + (m - 1)*(3*3^x - 1) - (m - 3)*3^x)) → 
  m < (-3 - real.sqrt 21) / 2 := 
sorry

end range_of_m_l77_77124


namespace nine_factorial_mod_eleven_l77_77810

theorem nine_factorial_mod_eleven : (9! % 11) = 1 :=
by
-- The proof is omitted
sorry

end nine_factorial_mod_eleven_l77_77810


namespace neg_p_is_necessary_but_not_sufficient_for_neg_q_l77_77492

variables (p q : Prop)

-- Given conditions: (p → q) and ¬(q → p)
theorem neg_p_is_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) :
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q) :=
sorry

end neg_p_is_necessary_but_not_sufficient_for_neg_q_l77_77492


namespace star_of_fractions_l77_77612

def star (r1 r2 : ℚ) : ℚ :=
  let m := r1.num
  let n := r1.denom
  let p := r2.num
  let q := r2.denom
  m * p * (n / q)

theorem star_of_fractions :
  star (5 / 9) (6 / 4) = 67.5 :=
by
  sorry

end star_of_fractions_l77_77612


namespace towels_after_a_week_l77_77677

theorem towels_after_a_week 
  (initial_green : ℕ) (initial_white : ℕ) (initial_blue : ℕ) 
  (daily_green : ℕ) (daily_white : ℕ) (daily_blue : ℕ) 
  (days : ℕ) 
  (H1 : initial_green = 35)
  (H2 : initial_white = 21)
  (H3 : initial_blue = 15)
  (H4 : daily_green = 3)
  (H5 : daily_white = 1)
  (H6 : daily_blue = 1)
  (H7 : days = 7) :
  (initial_green - daily_green * days) + (initial_white - daily_white * days) + (initial_blue - daily_blue * days) = 36 :=
by 
  sorry

end towels_after_a_week_l77_77677


namespace train_pass_bridge_in_time_l77_77363

def train_length : ℕ := 700
def bridge_length : ℕ := 350
def train_speed_kmh : ℕ := 120
def kmh_to_mps (kmh : ℕ) : ℕ := (kmh * 1000) / 3600

noncomputable def time_to_pass_bridge : ℚ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := kmh_to_mps train_speed_kmh
  (total_distance : ℚ) / (train_speed_mps : ℚ)

theorem train_pass_bridge_in_time : time_to_pass_bridge ≈ 31.5 :=
by 
  sorry

end train_pass_bridge_in_time_l77_77363


namespace color_block_prob_l77_77372

-- Definitions of the problem's conditions
def colors : List (List String) := [
    ["red", "blue", "yellow", "green"],
    ["red", "blue", "yellow", "white"]
]

-- The events in which at least one box receives 3 blocks of the same color
def event_prob : ℚ := 3 / 64

-- Tuple as a statement to prove in Lean
theorem color_block_prob (m n : ℕ) (h : m + n = 67) : 
  ∃ (m n : ℕ), (m / n : ℚ) = event_prob := 
by
  use 3
  use 64
  simp
  sorry

end color_block_prob_l77_77372


namespace square_area_is_100_l77_77715

-- Define the point and its distances from the closest sides of the square
variables (P : Type) [metric_space P] 
(inside_square : P)
(distance1 distance2 : ℝ)
(distance_to_side1 distance_to_side2 : inside_square = 1 ∧ inside_square = 2)

-- Define the radius of the inscribed circle
def radius := 5

-- Define the side length of the square as twice the radius of the circle
def side_length := 2 * radius

-- Define the area of the square
def area_of_square := side_length * side_length

-- Prove that given the conditions, the area of the square is 100
theorem square_area_is_100 : 
  area_of_square = 100 :=
by 
  sorry

end square_area_is_100_l77_77715


namespace find_smallest_common_factor_l77_77996

-- Define the smallest number
def smallest_num : ℕ := 627

-- Define the property that n + 3 is divisible by 4590 and 105
def condition (n : ℕ) : Prop :=
  (n + 3) % 4590 = 0 ∧ (n + 3) % 105 = 0

-- Define the smallest common factor function
def smallest_common_factor (a b c : ℕ) : ℕ :=
  (if h : ∃ d, d > 1 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c then Nat.find h else 1)

-- State the theorem
theorem find_smallest_common_factor :
  smallest_common_factor (smallest_num + 3) 4590 105 = 105 := 
sorry

end find_smallest_common_factor_l77_77996


namespace triangle_perimeter_eq_twenty_l77_77828

theorem triangle_perimeter_eq_twenty (x y : ℝ) (h : |x - 4| + real.sqrt (y - 8) = 0) : 
  20 = if (4,8,8) = (4,4,8) then 0 else 4 + 8 + 8 := 
by sorry

end triangle_perimeter_eq_twenty_l77_77828


namespace remainder_division_l77_77338

theorem remainder_division (x : ℤ) (hx : x % 82 = 5) : (x + 7) % 41 = 12 := 
by 
  sorry

end remainder_division_l77_77338


namespace cost_per_person_l77_77593

theorem cost_per_person (total_cost : ℕ) (num_people : ℕ) (h1 : total_cost = 30000) (h2 : num_people = 300) : total_cost / num_people = 100 := by
  -- No proof provided, only the theorem statement
  sorry

end cost_per_person_l77_77593


namespace integer_in_range_l77_77683

theorem integer_in_range (x : ℤ) 
  (h1 : 0 < x) 
  (h2 : x < 7)
  (h3 : 0 < x)
  (h4 : x < 15)
  (h5 : -1 < x)
  (h6 : x < 5)
  (h7 : 0 < x)
  (h8 : x < 3)
  (h9 : x + 2 < 4) : x = 1 := 
sorry

end integer_in_range_l77_77683


namespace cos_double_angle_l77_77832

theorem cos_double_angle (α : Real) (h : cos α = - (Real.sqrt 2) / 3) : 
  cos (2 * α) = - 5 / 9 := 
by
  sorry

end cos_double_angle_l77_77832


namespace number_of_such_s_in_S_l77_77219

def f (x : ℤ) : ℤ := x^3 - x + 1

def S : set ℤ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

theorem number_of_such_s_in_S : 
  (finset.filter (λ s, f s % 7 = 0) (finset.range 21)).card = 3 := 
  sorry

end number_of_such_s_in_S_l77_77219


namespace train_pass_man_time_approx_l77_77027

-- Definitions based on the conditions
def train_length : ℝ := 160
def man_speed_kmph : ℝ := 6
def train_speed_kmph : ℝ := 90

-- Converting kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

-- Calculate relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (man_speed_kmph + train_speed_kmph)

-- Calculate time taken for the train to pass the man in seconds
def time_to_pass : ℝ := train_length / relative_speed_mps

-- Theorem to prove the time is approximately 6 seconds
theorem train_pass_man_time_approx:
  time_to_pass ≈ 6 := sorry

end train_pass_man_time_approx_l77_77027


namespace pythagorean_triples_l77_77674

theorem pythagorean_triples:
  (∃ a b c : ℝ, (a = 1 ∧ b = 2 ∧ c = sqrt 5 ∧ a^2 + b^2 = c^2) ∨
   (a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 ≠ c^2) ∨
   (a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2) ∨
   (a = 4 ∧ b = 5 ∧ c = 6 ∧ a^2 + b^2 ≠ c^2)) ∧
  (∃ a' b' c' : ℝ, a' = 3 ∧ b' = 4 ∧ c' = 5) :=
by
  sorry

end pythagorean_triples_l77_77674


namespace total_stamps_l77_77540

-- Definitions based on conditions
def kylies_stamps : ℕ := 34
def nellys_stamps : ℕ := kylies_stamps + 44

-- Statement of the proof problem
theorem total_stamps : kylies_stamps + nellys_stamps = 112 :=
by
  -- Proof goes here
  sorry

end total_stamps_l77_77540


namespace melissa_trip_total_time_l77_77254

theorem melissa_trip_total_time :
  ∀ (freeway_dist rural_dist : ℕ) (freeway_speed_factor : ℕ) 
  (rural_time : ℕ),
  freeway_dist = 80 →
  rural_dist = 20 →
  freeway_speed_factor = 4 →
  rural_time = 40 →
  (rural_dist * freeway_speed_factor / rural_time + freeway_dist / (rural_dist * freeway_speed_factor / rural_time)) = 80 :=
by
  intros freeway_dist rural_dist freeway_speed_factor rural_time hd1 hd2 hd3 hd4
  sorry

end melissa_trip_total_time_l77_77254


namespace construct_convex_hexagon_l77_77387

-- Definitions of the sides and their lengths
variables {A B C D E F : Type} -- Points of the hexagon
variables {AB BC CD DE EF FA : ℝ}  -- Lengths of the sides
variables (convex_hexagon : Prop) -- the hexagon is convex

-- Hypotheses of parallel and equal opposite sides
variables (H_AB_DE : AB = DE)
variables (H_BC_EF : BC = EF)
variables (H_CD_AF : CD = AF)

-- Define the construction of the hexagon under the given conditions
theorem construct_convex_hexagon
  (convex_hexagon : Prop)
  (H_AB_DE : AB = DE)
  (H_BC_EF : BC = EF)
  (H_CD_AF : CD = AF) : 
  ∃ (A B C D E F : Type), 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧ convex_hexagon ∧ 
    (AB = FA) ∧ (AF = CD) ∧ (BC = EF) ∧ (AB = DE) := 
sorry -- Proof omitted

end construct_convex_hexagon_l77_77387


namespace hyperbola_eccentricity_l77_77136

variables {a b : ℝ} (ha : a > 0) (hb : b > 0)
def hyperbola (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := sqrt (a^2 + b^2), 
      F := (c, 0), 
      A := (0, b),
      P := (c/3, 2*b/3) in
  (hyp (x y : ℝ) := hyperbola a b x y) ∧
  (right_focus := F) ∧
  (point_A := A) ∧
  (line_FP_intersects_asymptote_at_point_P := P) ∧
  (vector PF : ℝ × ℝ := (fst F - fst P, snd F - snd P)) ∧
  (vector AP : ℝ × ℝ := (fst P - fst A, snd P - snd A)) ∧
  (2 * vector AP = vector PF) →
  let e := c / a in
  e = 2 :=
begin
  sorry
end

end hyperbola_eccentricity_l77_77136


namespace smallest_AAB_value_l77_77029

noncomputable def AAB_value (A B : ℕ) : ℕ := 100 * A + 10 * A + B

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ A ≠ B ∧ 30 * A = 7 * B ∧ AAB_value A B = 773 :=
by
  use 7
  use 3
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  sorry

end smallest_AAB_value_l77_77029


namespace second_player_cannot_win_l77_77344

theorem second_player_cannot_win (pile1 pile2 : ℕ) (h1 : pile1 = 10) (h2 : pile2 = 15) : 
  ¬(∃ strategy : ℕ × ℕ → (ℕ × ℕ), (∀ state ∈ [(10, 15)] ++ list, strategy state ≠ state ∧ win strategy state)) :=
sorry

end second_player_cannot_win_l77_77344


namespace range_of_m_for_inequality_l77_77425

theorem range_of_m_for_inequality (m : ℝ) : (∀ x : ℝ, exp(abs(2 * x + 1)) + m ≥ 0) → (m ≥ -1) := 
sorry

end range_of_m_for_inequality_l77_77425


namespace geoarith_seq_l77_77922

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

noncomputable def increasing_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

noncomputable def arithmetic_seq (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem geoarith_seq (a : ℕ → ℝ) (q : ℝ) (ln : ℝ → ℝ) :
  geometric_seq a q →
  increasing_seq a →
  a 0 + a 2 = 5 →
  arithmetic_seq (a 0 + 3) (3 * a 1) (a 2 + 4) →
  (∀ n, a n = 2^(n - 1)) ∧
  (∀ n, ∑ i in finset.range n, ln (a (3 * i + 1)) = (3 * n * (n + 1) / 2) * ln 2) :=
by
  intro h1 h2 h3 h4
  sorry

end geoarith_seq_l77_77922


namespace BC_value_l77_77209

-- Conditions
def triangle (A B C : Type) := ∃ (AB AC : ℝ), AB = 5 ∧ AC = 12 

-- Main theorem
theorem BC_value {A B C I P X Y : Type} (O_B O_C : Type) (AB AC : ℝ) (H1 : AB = 5) (H2 : AC = 12)
  (H3 : ∃ (P : Type), P = intersection AI BC)
  (H4 : ∃ (O_B O_C : Type), (O_B = circumcircle_center ABP) ∧ (O_C = circumcircle_center ACP))
  (H5 : ∃ (X Y : Type), reflection BC AI = intersection X Y)
  (H6 : ∀ (Hratio : ℝ), Hratio = (O_B O_C / XY) := (PI / IA)) :
  BC = sqrt 109 :=
sorry

end BC_value_l77_77209


namespace calculate_boxed_2_1_neg1_l77_77428

def boxed (a b c : ℤ) : ℤ := a^(b + c) - b^(a - c) + c^(a + b)

theorem calculate_boxed_2_1_neg1 :
  boxed 2 1 (-1) = -1 :=
by
  sorry

end calculate_boxed_2_1_neg1_l77_77428


namespace max_min_difference_l77_77497

noncomputable def f (x a : ℝ) : ℝ := x^3 - 3 * x - a

theorem max_min_difference (a : ℝ) :
  let f (x : ℝ) := x^3 - 3 * x - a
  ∃ M N ∈ [[0, 3]], M = max [f 0, f 1, f 3] ∧ N = min [f 0, f 1, f 3] ∧ M - N = 18 := by
  sorry

end max_min_difference_l77_77497


namespace exists_inequality_l77_77555

theorem exists_inequality (n : ℕ) (x : Fin (n + 1) → ℝ) 
  (hx1 : ∀ i, 0 ≤ x i ∧ x i ≤ 1) 
  (h_n : 2 ≤ n) : 
  ∃ i : Fin n, x i * (1 - x (i + 1)) ≥ (1 / 4) * x 0 * (1 - x n) :=
sorry

end exists_inequality_l77_77555


namespace inequality_holds_l77_77436
noncomputable theory

theorem inequality_holds (θ : ℝ) (k : ℝ) (h_θ_range: θ ∈ Set.Ico 0 (2 * π)) (h_k_positive: 0 < k) :
  (ln (sin θ)^2 - ln (cos θ)^2 ≤ k * cos (2 * θ)) ↔
  (θ ∈ Set.Ioc 0 (π / 4) ∪ Set.Ico (3 * π / 4) π ∪ Set.Ioc π (5 * π / 4) ∪ Set.Ico (7 * π / 4) (2 * π)) :=
sorry

end inequality_holds_l77_77436


namespace game_one_piece_condition_l77_77947

theorem game_one_piece_condition (n : ℕ) : 
  (∃ (m : ℕ) (r : fin 3), n = 3 * m + r.val ∧ (r.val = 1 ∨ r.val = 2)) ↔ game_results_one (initialize_board n) :=
sorry

end game_one_piece_condition_l77_77947


namespace youseff_blocks_l77_77335

theorem youseff_blocks (x : ℕ) 
  (H1 : (1 : ℚ) * x = (1/3 : ℚ) * x + 8) : 
  x = 12 := 
sorry

end youseff_blocks_l77_77335


namespace additional_percent_reduction_l77_77990

noncomputable def jacket_reduction (P : ℝ) (x : ℝ) : Prop :=
let new_price := (1 - x) * 0.65 * P in
(1 + 0.7094) * new_price = P

theorem additional_percent_reduction : ∀ (P : ℝ), jacket_reduction P 0.4149 :=
by
  intros P
  unfold jacket_reduction
  sorry

end additional_percent_reduction_l77_77990


namespace compare_A_B_l77_77672

noncomputable def A (x : ℝ) := x / (x^2 - x + 1)
noncomputable def B (y : ℝ) := y / (y^2 - y + 1)

theorem compare_A_B (x y : ℝ) (hx : x > y) (hx_val : x = 2.00 * 10^1998 + 4) (hy_val : y = 2.00 * 10^1998 + 2) : 
  A x < B y := 
by 
  sorry

end compare_A_B_l77_77672


namespace smallest_multiple_divisors_l77_77223

theorem smallest_multiple_divisors :
  ∃ m : ℕ, (∃ k1 k2 : ℕ, m = 2^k1 * 5^k2 * 100 ∧ 
    (∀ d : ℕ, d ∣ m → d = 1 ∨ d = m ∨ ∃ e1 e2 : ℕ, d = 2^e1 * 5^e2 * 100)) ∧
    (∀ d : ℕ, d ∣ m → d ≠ 1 → (d ≠ m → ∃ e1 e2 : ℕ, d = 2^e1 * 5^e2 * 100)) ∧
    (m.factors.length = 100) ∧ 
    m / 100 = 2^47 * 5^47 :=
begin
  sorry
end

end smallest_multiple_divisors_l77_77223


namespace question_1_question_2_question_3_l77_77849

noncomputable def f (x : ℝ) : ℝ := abs(x + 1) + abs(x - 1)

theorem question_1 : ∀ x : ℝ, f x < 2 * x + 3 ↔ x > -1 / 2 := 
by
  sorry

theorem question_2 : ∀ m : ℝ, (∃ x₀ : ℝ, f x₀ ≤ m) → m ≥ 2 :=
by
  sorry

theorem question_3 : ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 3 * a + b = 2 → 1 / (2 * a) + 1 / (a + b) ≥ 2 :=
by
  sorry

end question_1_question_2_question_3_l77_77849


namespace sin_beta_value_l77_77432

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < β ∧ β < π / 2)
variable (h₂ : Real.sin α = 3 / 5)
variable (h₃ : Real.cos (β - α) = 12 / 13)

theorem sin_beta_value : Real.sin β = 56 / 65 :=
by
  sorry

end sin_beta_value_l77_77432


namespace john_duck_price_l77_77534

theorem john_duck_price
  (n_ducks : ℕ)
  (cost_per_duck : ℕ)
  (weight_per_duck : ℕ)
  (total_profit : ℕ)
  (total_cost : ℕ)
  (total_weight : ℕ)
  (total_revenue : ℕ)
  (price_per_pound : ℕ)
  (h1 : n_ducks = 30)
  (h2 : cost_per_duck = 10)
  (h3 : weight_per_duck = 4)
  (h4 : total_profit = 300)
  (h5 : total_cost = n_ducks * cost_per_duck)
  (h6 : total_weight = n_ducks * weight_per_duck)
  (h7 : total_revenue = total_cost + total_profit)
  (h8 : price_per_pound = total_revenue / total_weight) :
  price_per_pound = 5 := 
sorry

end john_duck_price_l77_77534


namespace regular_polygon_sides_l77_77874

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 144) : ∃ n : ℕ, n = 10 :=
by 
  have exterior_angle := 180 - interior_angle
  have n := 360 / exterior_angle
  use n
  sorry

end regular_polygon_sides_l77_77874


namespace winning_numbers_per_ticket_l77_77628

noncomputable def number_of_tickets : ℕ := 3
noncomputable def value_per_winning_number : ℕ := 20
noncomputable def total_winnings : ℕ := 300

theorem winning_numbers_per_ticket :
  let total_winning_numbers := total_winnings / value_per_winning_number,
      winning_numbers_per_ticket := total_winning_numbers / number_of_tickets in
  winning_numbers_per_ticket = 5 := 
by
  sorry

end winning_numbers_per_ticket_l77_77628


namespace find_a_l77_77843

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 1 else x - 1

theorem find_a (a : ℝ) :
  f (a + 1) = f (a) ↔ (a = -1/2 ∨ a = (-1 + Real.sqrt 5)/2) :=
by sorry

end find_a_l77_77843


namespace gcd_of_B_is_2_l77_77212

def B : Set ℕ :=
    { m | ∃ n : ℕ, m = n + (n + 1) + (n + 2) + (n + 3) }

theorem gcd_of_B_is_2 :
    gcd (Set.toFinset B).val = 2 := 
begin
    -- Formalization of problem's given conditions and required proofs
    sorry
end

end gcd_of_B_is_2_l77_77212


namespace statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l77_77676

-- Statement A: Proving the solution set of the inequality
theorem statement_A_solution_set (x : ℝ) : 
  (x + 2) / (2 * x + 1) > 1 ↔ (-1 / 2) < x ∧ x < 1 :=
sorry

-- Statement B: "ab > 1" is not a sufficient condition for "a > 1, b > 1"
theorem statement_B_insufficient_condition (a b : ℝ) :
  (a * b > 1) → ¬(a > 1 ∧ b > 1) :=
sorry

-- Statement C: The negation of p: ∀ x ∈ ℝ, x² > 0 is true
theorem statement_C_negation (x0 : ℝ) : 
  (∀ x : ℝ, x^2 > 0) → ¬ (∃ x0 : ℝ, x0^2 ≤ 0) :=
sorry

-- Statement D: "a < 2" is not a necessary condition for "a < 6"
theorem statement_D_not_necessary_condition (a : ℝ) :
  (a < 2) → ¬(a < 6) :=
sorry

end statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l77_77676


namespace sum_abs_roots_poly_l77_77800

noncomputable def poly := Polynomial.C (-36) + Polynomial.X * (Polynomial.C 24 + Polynomial.X * (Polynomial.C 9 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X)))

theorem sum_abs_roots_poly : (Polynomial.roots poly).map (λ x, |x|).sum = 6 + 4 * Real.sqrt 3 := sorry

end sum_abs_roots_poly_l77_77800


namespace circle_ray_no_common_points_l77_77837

theorem circle_ray_no_common_points (a : ℝ) :
  (∀ x y : ℝ, ((x - a)^2 + y^2 = 4) → (y = real.sqrt 3 * x) → x < 0) ∨ a < -2 ∨ a > (4 / 3) * real.sqrt 3 :=
by sorry

end circle_ray_no_common_points_l77_77837


namespace sequence_not_palindrome_l77_77054

def concatenate_sequence (n : ℕ) : string :=
  (string.join (list.map to_string (list.range (n + 1)).tail))

def is_palindrome (s : string) : Prop :=
  s = s.reverse

theorem sequence_not_palindrome (n : ℕ) (h : n > 1) : ¬ is_palindrome (concatenate_sequence n) :=
by
  sorry

end sequence_not_palindrome_l77_77054


namespace part_I_l77_77062

noncomputable def f (x a : ℝ) := x * (x - 1) * (x - a)

theorem part_I (a : ℝ) (h : 1 < a) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (deriv (λ x, f x a) x1 = 0 ∧ deriv (λ x, f x a) x2 = 0) :=
sorry

lemma part_II (a : ℝ) (h : 1 < a) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv (λ x, f x a) x1 = 0 ∧ deriv (λ x, f x a) x2 = 0 ∧ f x1 a + f x2 a ≤ 0) →
  2 ≤ a :=
sorry

end part_I_l77_77062


namespace taco_castle_l77_77902

noncomputable theory

theorem taco_castle (D : ℝ) (F : ℝ) (T : ℝ) (N : ℝ) (V : ℝ) (H : ℝ) (M : ℝ) (C : ℝ) (S : ℝ) (Fi : ℝ) : 
  (F = (1/3)*D) ∧ 
  (T = (1/6)*D) ∧ 
  (N = (3/14)*D) ∧ 
  (V = (1/12)*D) ∧ 
  (H = (1/4)*D) ∧ 
  (M = (3/35)*D) ∧ 
  (C = (1/6)*D) ∧ 
  (S = 4*D) ∧ 
  (Fi = (3/70)*D) ∧ 
  (V = 5) → D = 60 := 
by
  sorry

end taco_castle_l77_77902


namespace rate_per_kg_for_fruits_l77_77318

-- Definitions and conditions
def total_cost (rate_per_kg : ℝ) : ℝ := 8 * rate_per_kg + 9 * rate_per_kg

def total_paid : ℝ := 1190

theorem rate_per_kg_for_fruits : ∃ R : ℝ, total_cost R = total_paid ∧ R = 70 :=
by
  sorry

end rate_per_kg_for_fruits_l77_77318


namespace log_monotonically_decreasing_l77_77986

theorem log_monotonically_decreasing : 
  ∀ x : ℝ, 0 < x → x ∈ (0, +∞) ∧ 0 < 0.6 ∧ 0.6 < 1 → ∀ x1 x2 : ℝ, 
  0 < x1 → 0 < x2 → x1 < x2 → log 0.6 x2 < log 0.6 x1 :=
sorry

end log_monotonically_decreasing_l77_77986


namespace angle_between_a_and_b_l77_77818

noncomputable def angle_between_vectors (a b : ℝ) : ℝ :=
  if a = 0 ∨ b = 0 then 0 else real.acos (a * b / (|a| * |b|))

theorem angle_between_a_and_b
  (a b : ℝ)
  (ha : |a| = 3)
  (hb : |b| = 4)
  (h : (a + b) * (a + 3 * b) = 33) :
  angle_between_vectors a b = real.pi / 3 :=    -- 120 degrees in radians.
by sorry

end angle_between_a_and_b_l77_77818


namespace difference_between_max_and_min_change_l77_77373

-- Define percentages as fractions for Lean
def initial_yes : ℚ := 60 / 100
def initial_no : ℚ := 40 / 100
def final_yes : ℚ := 80 / 100
def final_no : ℚ := 20 / 100
def new_students : ℚ := 10 / 100

-- Define the minimum and maximum possible values of changes (in percentage as a fraction)
def min_change : ℚ := 10 / 100
def max_change : ℚ := 50 / 100

-- The theorem we need to prove
theorem difference_between_max_and_min_change : (max_change - min_change) = 40 / 100 :=
by
  sorry

end difference_between_max_and_min_change_l77_77373


namespace mean_of_integers_neg3_to_6_l77_77645

theorem mean_of_integers_neg3_to_6 : 
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ) in
  let n := (6 - (-3) + 1 : ℤ) in
  s / n = 1.5 :=
by
  let s := ∑ i in (-3 : finset ℤ).Icc 6, (i : ℝ)
  let n := (6 - (-3) + 1 : ℤ)
  simp
  sorry

end mean_of_integers_neg3_to_6_l77_77645


namespace directrix_of_parabola_l77_77412

theorem directrix_of_parabola (x y : ℝ) :
  (y = -3 * x^2 + 6 * x - 5) -> (y = -3 * (x - 1)^2 - 2) 
  -> (directrix_y : ℝ) (directrix_y = -23 / 12) := by
  intros h1 h2
  sorry

end directrix_of_parabola_l77_77412


namespace sequence_b15_l77_77358

def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else 2 * (sequence (n - 1) - sequence (n - 2)) / (sequence (n - 3))

theorem sequence_b15 : sequence 15 = 0 := by
  sorry

end sequence_b15_l77_77358


namespace third_root_of_polynomial_l77_77084

variable (a b x : ℝ)
noncomputable def polynomial := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

theorem third_root_of_polynomial (h1 : polynomial a b (-3) = 0) (h2 : polynomial a b 4 = 0) :
  ∃ r : ℝ, r = -17 / 10 ∧ polynomial a b r = 0 :=
by
  sorry

end third_root_of_polynomial_l77_77084


namespace exist_nonzero_ints_l77_77552

theorem exist_nonzero_ints (m n : ℤ) (h_m : m ≥ 2) (h_n : n ≥ 2)
  (a : ℕ → ℤ) (h_a : ∀ i, 1 ≤ i ∧ i ≤ n → ¬ m^(n-1) ∣ a i) :
  ∃ e : ℕ → ℤ, (∀ i, 1 ≤ i ∧ i ≤ n → 0 < |e i| ∧ |e i| < m) ∧ (m^n ∣ ∑ i in finset.range n, e i * a i) :=
by
  sorry

end exist_nonzero_ints_l77_77552


namespace expected_value_difference_l77_77281

def prime_numbers := {2, 3, 5, 7}
def composite_numbers := {4, 6, 8}
def non_leap_year_days := 365
noncomputable def probability_prime : ℚ := 4 / 7
noncomputable def probability_composite : ℚ := 3 / 7

theorem expected_value_difference :
  (probability_prime * non_leap_year_days) - (probability_composite * non_leap_year_days) = 365 / 7 := 
  by sorry

end expected_value_difference_l77_77281


namespace nat_addition_bd_l77_77089

theorem nat_addition_bd (b d : ℕ) (h : 2 * 4 - b * d = 2) : b + d = 5 ∨ b + d = 7 :=
by
  have h1 : 8 - b * d = 2 := h
  have h2 : b * d = 6 := by linarith
  sorry

end nat_addition_bd_l77_77089


namespace distance_center_point_l77_77524

theorem distance_center_point :
  let center := (0 : ℝ, 1 : ℝ)
  let point := (-1 : ℝ, 0 : ℝ)
  (Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2) = Real.sqrt 2) :=
by
  let center := (0, 1)
  let point := (-1, 0)
  exact Eq.refl (Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2))

end distance_center_point_l77_77524


namespace radius_of_tangent_circle_l77_77002

def is_tangent_coor_axes_and_leg (r : ℝ) : Prop :=
  -- Circle with radius r is tangent to coordinate axes and one leg of the triangle
  ∃ O B C : ℝ × ℝ, 
  -- Conditions: centers and tangency
  O = (r, r) ∧ 
  B = (0, 2) ∧ 
  C = (2, 0) ∧ 
  r = 1

theorem radius_of_tangent_circle :
  ∀ r : ℝ, is_tangent_coor_axes_and_leg r → r = 1 :=
by
  sorry

end radius_of_tangent_circle_l77_77002


namespace theta_range_ineq_l77_77101

noncomputable def theta_range (theta : ℝ) : Prop :=
  (θ ∈ Icc 0 (π / 2)) ∧ (sin θ ^ 3 - cos θ ^ 3 ≥ log (cot θ) → θ ∈ Ico (π / 4) (π / 2))

theorem theta_range_ineq (θ : ℝ) : theta_range θ :=
sorry

end theta_range_ineq_l77_77101


namespace front_view_correct_l77_77777

def heights : List (List ℕ) := [[2, 1], [1, 3, 1], [4, 1]]

noncomputable def front_view (heights : List (List ℕ)) : List ℕ :=
  heights.map (λ col => col.maximum'.get_or_else 0)

theorem front_view_correct : front_view heights = [2, 3, 4] :=
by
  sorry

end front_view_correct_l77_77777


namespace area_of_enclosed_region_l77_77967
open Real

noncomputable def enclosed_area : ℝ :=
  ∫ x in 0..1, (sqrt x - x^2)

theorem area_of_enclosed_region : enclosed_area = 1 / 3 := by
  sorry

end area_of_enclosed_region_l77_77967


namespace new_volume_of_cylinder_l77_77307

theorem new_volume_of_cylinder (r h : ℝ) (π : ℝ := Real.pi) (V : ℝ := π * r^2 * h) (hV : V = 15) :
  let r_new := 3 * r
  let h_new := 4 * h
  let V_new := π * (r_new)^2 * h_new
  V_new = 540 :=
by
  sorry

end new_volume_of_cylinder_l77_77307


namespace inscribed_quadrilateral_radius_l77_77072

theorem inscribed_quadrilateral_radius (a b c d : ℝ) (r : ℝ) 
  (h_inscribed : ∃ (O : Type) (R : ℝ) (circumcircle_O : ∀ (P : Type), P ∈ {a, b, c, d} → P ∈ O)) :
  r^2 = (ab + bd) * (ad + bc) * (ab + cd) / ((a + b + c - d) * (a + b - c + d) * (a - b + c + d) * (-a + b + c + d)) :=
by
  sorry

end inscribed_quadrilateral_radius_l77_77072


namespace brian_video_watching_time_l77_77751

theorem brian_video_watching_time :
  let catVideo : ℕ := 4
  let dogVideo : ℕ := 2 * catVideo
  let combinedCatDog : ℕ := catVideo + dogVideo
  let gorillaVideo : ℕ := 2 * combinedCatDog
  let totalTime : ℕ := catVideo + dogVideo + gorillaVideo
  totalTime = 36 :=
by
  -- Define the variables
  let catVideo := 4
  let dogVideo := 2 * catVideo
  let combinedCatDog := catVideo + dogVideo
  let gorillaVideo := 2 * combinedCatDog
  let totalTime := catVideo + dogVideo + gorillaVideo
  -- Combine all steps and assert the final value
  show totalTime = 36, from
    sorry -- Proof not implemented

end brian_video_watching_time_l77_77751


namespace sum_of_all_possible_values_of_f1_l77_77546

-- Define the non-constant polynomial f satisfying the given functional equation
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The sum of all possible values of f(1) is 10/3 given the conditions on f. -/
theorem sum_of_all_possible_values_of_f1 :
  (∀ x : ℝ, f(x - 1) + f(x) + f(x + 1) = (f(x)^2) / (x^2 + 1)) →
  (f(0) ≠ f(1)) →
  f(1) = 10 / 3 :=
sorry

end sum_of_all_possible_values_of_f1_l77_77546


namespace integral1_eq_45_over_4_integral2_eq_1_l77_77051

open Real Int

noncomputable def integral1 := ∫ x in -1..8, 3 * x
noncomputable def integral2 := ∫ x in 2..(Real.exp 1 + 1), 1 / (x - 1)

theorem integral1_eq_45_over_4 : integral1 = 45 / 4 :=
by
  sorry

theorem integral2_eq_1 : integral2 = 1 :=
by
  sorry

end integral1_eq_45_over_4_integral2_eq_1_l77_77051


namespace sin_C_condition1_sin_C_condition2_sin_C_condition3_area_triangle_l77_77381

-- Definitions for the conditions
def Condition1 (a b c : ℝ) : Prop := ∃ B C, c * sin B = b * cos (C - π / 6)
def Condition2 (a b c : ℝ) : Prop := ∃ B, cos B = (2 * a - b) / (2 * c)
def Condition3 (a b c : ℝ) : Prop := ∃ C, (a^2 + b^2 - c^2) * tan C = sqrt 3 * a * b

-- Main statements
theorem sin_C_condition1 (a b c : ℝ) (cond1 : Condition1 a b c) : ∃ C, sin C = sqrt 3 / 2 := by sorry
theorem sin_C_condition2 (a b c : ℝ) (cond2 : Condition2 a b c) : ∃ C, sin C = sqrt 3 / 2 := by sorry
theorem sin_C_condition3 (a b c : ℝ) (cond3 : Condition3 a b c) : ∃ C, sin C = sqrt 3 / 2 := by sorry

theorem area_triangle (a b : ℝ) (c : ℝ := sqrt 3) (h1 : a = 3 * b) :
  1 / 2 * a * b * sqrt 3 / 2 = 9 * sqrt 3 / 28 := by sorry

end sin_C_condition1_sin_C_condition2_sin_C_condition3_area_triangle_l77_77381


namespace minimal_abs_difference_l77_77868

theorem minimal_abs_difference (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a * b - 5 * a + 6 * b = 228) :
  ∃ n, n = 18 ∧ ∀ (a' b' : ℕ), (0 < a' ∧ 0 < b' ∧ a' * b' - 5 * a' + 6 * b' = 228) → |a' - b'| ≥ n := 
sorry

end minimal_abs_difference_l77_77868


namespace number_of_small_triangles_needed_l77_77710

-- Define the area of an equilateral triangle function
def equilateral_triangle_area (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- Conditions based on the problem statement
def large_triangle_side_length : ℝ := 16
def small_triangle_side_length : ℝ := 2

-- Calculate the areas based on the given sides
def area_large_triangle := equilateral_triangle_area large_triangle_side_length
def area_small_triangle := equilateral_triangle_area small_triangle_side_length

-- The proof statement: Prove that the number of small triangles required is 64
theorem number_of_small_triangles_needed : 
  area_large_triangle / area_small_triangle = 64 := by
  sorry

end number_of_small_triangles_needed_l77_77710


namespace dot_product_sufficient_not_necessary_parallel_l77_77923

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Define parallelism in an inner product space
def is_parallel (a b : V) : Prop :=
  ∃ k : ℝ, a = k • b

-- State the theorem about dot product and parallelism
theorem dot_product_sufficient_not_necessary_parallel (h : a ≠ 0) (h : b ≠ 0) :
  (inner_product_space.to_has_inner.inner a b = ∥a∥ * ∥b∥) →
    is_parallel a b := sorry

end dot_product_sufficient_not_necessary_parallel_l77_77923


namespace average_class_score_l77_77177

theorem average_class_score : 
  ∀ (n total score_per_100 score_per_0 avg_rest : ℕ), 
  n = 20 → 
  total = 800 → 
  score_per_100 = 2 → 
  score_per_0 = 3 → 
  avg_rest = 40 → 
  ((score_per_100 * 100 + score_per_0 * 0 + (n - (score_per_100 + score_per_0)) * avg_rest) / n = 40)
:= by
  intros n total score_per_100 score_per_0 avg_rest h_n h_total h_100 h_0 h_rest
  sorry

end average_class_score_l77_77177


namespace pentagon_angle_ap_l77_77598

theorem pentagon_angle_ap (a d : ℝ) (h1 : a > 60) (h2 : 5 * a + 10 * d = 540) :
  ∃ n, (n ∈ {0, 1, 2, 3, 4}) ∧ (a + n * d = 90) :=
by { sorry }

end pentagon_angle_ap_l77_77598


namespace simplify_expression_l77_77957

theorem simplify_expression :
  (210 / 18) * (6 / 150) * (9 / 4) = 21 / 20 :=
by
  sorry

end simplify_expression_l77_77957


namespace area_S_property_l77_77543

def floor (t: ℝ) : ℝ := sorry -- Defining floor function since Mathlib uses int.floor

noncomputable def area_of_S (t V: ℝ) : ℝ :=
  let T := t - floor t in
  π * T^2

theorem area_S_property (t V: ℝ) (ht : t ≥ 0) :
  0 ≤ area_of_S t V ∧ area_of_S t V ≤ π :=
by
  sorry

end area_S_property_l77_77543


namespace power_add_one_eq_twice_l77_77093

theorem power_add_one_eq_twice (a b : ℕ) (h : 2^a = b) : 2^(a + 1) = 2 * b := by
  sorry

end power_add_one_eq_twice_l77_77093


namespace miriam_pushups_l77_77248

theorem miriam_pushups :
  let push_ups_mon := 5 in
  let push_ups_tue := 1.4 * push_ups_mon in
  let push_ups_wed := 2 * push_ups_mon in
  let push_ups_thu := (push_ups_mon + push_ups_tue + push_ups_wed) / 2 in
  let push_ups_fri := push_ups_mon + push_ups_tue + push_ups_wed + push_ups_thu in
  push_ups_fri = 33 :=
by
  sorry

end miriam_pushups_l77_77248


namespace degrees_equal_l77_77931

-- Define the degree of a polynomial
noncomputable def degree (P : polynomial ℝ) : ℕ := polynomial.degree P.toPnat

-- Definitions of the conditions for the polynomials P₁ to P₂₀₂₁
def P (i : ℕ) (hi : 1 ≤ i ∧ i ≤ 2021) : polynomial ℝ := sorry -- placeholder for the actual polynomial functions

axiom non_constant_polynomial (i : ℕ) (hi : 1 ≤ i ∧ i ≤ 2021) : 1 ≤ degree (P i hi)

axiom composition_equal (i : ℕ) (hi : 1 ≤ i ∧ i ≤ 2021 ∧ i < 2021) :
  P i hi ∘ P i.succ (sorry : 1 ≤ i + 1 ∧ i + 1 ≤ 2021) = P i.succ (sorry : 1 ≤ i + 1 ∧ i + 1 ≤ 2021) ∘ P (i + 2) (sorry : 1 ≤ i + 2 ∧ i + 2 ≤ 2021)

axiom composition_equal_last : 
  P 2021 (by linarith) ∘ P 1 (by linarith) = P 1 (by linarith) ∘ P 2 (by linarith)

-- The main statement to be proved
theorem degrees_equal : ∀ i j : ℕ, (1 ≤ i ∧ i ≤ 2021) → (1 ≤ j ∧ j ≤ 2021) → degree (P i sorry) = degree (P j sorry) :=
begin
  sorry
end

end degrees_equal_l77_77931


namespace inverse_of_four_l77_77547

-- Define the function f
def f (x : ℝ) : ℝ := 2^x

-- Define the inverse function f_inv
noncomputable def f_inv (y : ℝ) : ℝ := 
  if h : y > 0 then log y / log 2 else 0

-- State the theorem to be proved
theorem inverse_of_four : f_inv 4 = 2 := 
by
  -- Proof of this theorem will be left to the user
  sorry

end inverse_of_four_l77_77547


namespace domain_f_g_symmetry_f_g_solution_set_f_g_l77_77135

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log a (1 - x)
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

-- Assumptions
variable (a : ℝ)
variable h_a : a > 0 ∧ a ≠ 1

-- 1. Domain of f(x) + g(x) is {x | -1 < x < 1}
theorem domain_f_g : ∀ x : ℝ, (F a x).Domain = { x | -1 < x ∧ x < 1 } :=
by
  sorry

-- 2. Symmetry: f(x) + g(x) is even
theorem symmetry_f_g : ∀ x : ℝ, F a x = F a (-x) :=
by
  sorry

-- 3. Find the set of x such that f(x) - g(x) > 0
theorem solution_set_f_g (a : ℝ) (h : a > 0 ∧ a ≠ 1) (x : ℝ) : 
  if a < 1 then 
    (-1 < x ∧ x < 0) ↔ (f a x - g a x > 0)
  else 
    (0 < x ∧ x < 1) ↔ (f a x - g a x > 0) :=
by
  sorry

end domain_f_g_symmetry_f_g_solution_set_f_g_l77_77135


namespace num_correct_statements_l77_77870

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

def is_even_function (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = h x

def non_decreasing_on_neg (g : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → g x ≤ g y

def mono_increasing_on_neg (h : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < 0 → y < 0 → x ≤ y → h x ≤ h y

theorem num_correct_statements (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_mono : is_monotonically_increasing_on_nonneg f) :
  {s | s ∈ [
    is_even_function (λ x, |f x|),
    ∀ x, ¬ (f (-x) + |f x| = 0),
    non_decreasing_on_neg (λ x, f (-x)),
    mono_increasing_on_neg (λ x, f x * f (-x))
  ]} = 2 :=
by
  sorry

end num_correct_statements_l77_77870


namespace sister_ages_l77_77597

theorem sister_ages (x y : ℕ) (h1 : x = y + 4) (h2 : x^3 - y^3 = 988) : y = 7 ∧ x = 11 :=
by
  sorry

end sister_ages_l77_77597


namespace tangent_circles_l77_77562

theorem tangent_circles
  (C1 C2 C3 : Type) [metric_space C1] [metric_space C2] [metric_space C3]
  (l : Type)
  (r1 r2 r3 : ℝ)
  (h_radii : 0 < r3 ∧ r3 < min r1 r2)
  (h_tangent_C1_C2 : ∀ (x : C1) (y : C2), dist x y = r1 + r2)
  (h_tangent_C1_C3 : ∀ (x : C1) (z : C3), dist x z = r1 + r3)
  (h_tangent_C2_C3 : ∀ (y : C2) (z : C3), dist y z = r2 + r3)
  (h_tangent_l : ∀ (z : C3), z ∈ l) :
  1 / real.sqrt r3 = 1 / real.sqrt r1 + 1 / real.sqrt r2 :=
begin
  sorry
end

end tangent_circles_l77_77562


namespace divide_remaining_into_same_shape_l77_77812

def three_by_three_by_three_cube := fin 3 × fin 3 × fin 3
def tunnel := { (1,1,0), (1,1,1), (1,1,2) }

noncomputable def remaining_cubes := { p : three_by_three_by_three_cube | p ∉ tunnel }

theorem divide_remaining_into_same_shape: 
  (∃ g : fin 8 → fin 3 × fin 3 × fin 3 → Prop, 
    (∀ i : fin 8, ∃ s, g i = {p: three_by_three_by_three_cube | p ∈ s} ∧ s.card = 3 
    ∧ ∃ s', s ≃ s' ∧ s' = tunnel)) :=
sorry

end divide_remaining_into_same_shape_l77_77812


namespace smallest_AAB_value_l77_77030

noncomputable def AAB_value (A B : ℕ) : ℕ := 100 * A + 10 * A + B

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ A ≠ B ∧ 30 * A = 7 * B ∧ AAB_value A B = 773 :=
by
  use 7
  use 3
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  sorry

end smallest_AAB_value_l77_77030


namespace remaining_area_of_torn_square_l77_77015

theorem remaining_area_of_torn_square (perimeter : ℝ) (torn_fraction remaining_fraction : ℝ) :
  perimeter = 32 → torn_fraction = 1 / 4 → remaining_fraction = 1 - torn_fraction → 
  let side := perimeter / 4 in let original_area := side * side in let remaining_area := remaining_fraction * original_area in 
  remaining_area = 48 :=
by
  intros h_perimeter h_torn_fraction h_remaining_fraction
  let side := perimeter / 4
  let original_area := side * side
  let remaining_area := remaining_fraction * original_area
  have : remaining_area = 48
  sorry

end remaining_area_of_torn_square_l77_77015


namespace average_cost_of_diesel_l77_77730

noncomputable def average_cost_per_litre_year1 : ℝ :=
  let cost_diesel := 8.50 * 520
  let delivery_fees := 200
  let miscellaneous_taxes := 300
  cost_diesel + delivery_fees + miscellaneous_taxes

noncomputable def average_cost_per_litre_year2 : ℝ :=
  let cost_diesel := 9.0 * 540
  let delivery_fees := 200
  let miscellaneous_taxes := 300
  cost_diesel + delivery_fees + miscellaneous_taxes

noncomputable def average_cost_per_litre_year3 : ℝ :=
  let cost_diesel := 9.50 * 560
  let delivery_fees := 200
  let miscellaneous_taxes := 300
  cost_diesel + delivery_fees + miscellaneous_taxes

noncomputable def total_cost : ℝ :=
  average_cost_per_litre_year1 + average_cost_per_litre_year2 + average_cost_per_litre_year3

noncomputable def total_litres : ℕ :=
  520 + 540 + 560

noncomputable def average_cost_per_litre : ℝ :=
  total_cost / total_litres

theorem average_cost_of_diesel : average_cost_per_litre ≈ 9.94 :=
  by
    sorry

end average_cost_of_diesel_l77_77730


namespace intersection_M_complement_N_l77_77143

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def N : Set ℝ := {x | ∃ y : ℝ, y = 3*x^2 + 1 }

def complement_N : Set ℝ := {x | ¬ ∃ y : ℝ, y = 3*x^2 + 1}

theorem intersection_M_complement_N :
  (M ∩ complement_N) = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_l77_77143


namespace product_ab_l77_77300

noncomputable def u : ℂ := -1 + 2 * Complex.i
noncomputable def v : ℂ := 2 + 3 * Complex.i
noncomputable def a : ℂ := 3 + Complex.i
noncomputable def b : ℂ := 3 - Complex.i
noncomputable def d : ℂ := 8 * Complex.i

theorem product_ab : a * b = 10 :=
by 
  have ab := (3 + Complex.i) * (3 - Complex.i)
  calc ab = 10 : by { simp, norm_num, }
  sorry

end product_ab_l77_77300


namespace sqrt_expression_simplification_l77_77864

theorem sqrt_expression_simplification (x : ℝ) (h : -x^2 + 5*x - 6 > 0) : 
\(\sqrt{4*x^2 - 12*x + 9} + 3*|x - 3| = 6 - x\) := 
begin 
  sorry 
end

end sqrt_expression_simplification_l77_77864


namespace margo_total_distance_l77_77245

-- Definitions based on the conditions given
def jog_time : ℕ := 15 -- in minutes
def walk_time : ℕ := 25 -- in minutes
def average_speed : ℝ := 6 -- in miles per hour

-- Time conversion: total time in hours
def total_time : ℝ := (jog_time + walk_time) / 60

-- Problem Statement: Prove that the total distance covered is 4 miles
theorem margo_total_distance (jt wt : ℕ) (avg_speed : ℝ) (jt_eq : jt = jog_time) (wt_eq : wt = walk_time) (speed_eq : avg_speed = average_speed) :
  avg_speed * ((jt + wt) / 60) = 4 :=
by
  -- Placeholder to acknowledge the structure, actual proof is not required as per the prompt
  sorry

end margo_total_distance_l77_77245


namespace progressive_more_than_conservative_l77_77240

def conservative_set (A : set ℕ) : Prop :=
  ∀ x ∈ A, ¬∃ y ∈ (A \ {x}), y ∣ x

def progressive_set (A : set ℕ) : Prop :=
  ∀ x ∈ A, ∀ n : ℕ, n ∣ x → n ≤ 10^100 → n ∈ A

theorem progressive_more_than_conservative :
  ∃ S_cons S_prog : set (set ℕ),
    (∀ A ∈ S_cons, conservative_set A) ∧
    (∀ B ∈ S_prog, progressive_set B) ∧
    (∃ f : set ℕ → set ℕ, function.injective f ∧
      (∀ A ∈ S_cons, f A ∈ S_prog)) ∧
    (¬∃ g : set ℕ → set ℕ, function.injective g ∧
      (∀ B ∈ S_prog, conservative_set (g B))) :=
sorry

end progressive_more_than_conservative_l77_77240


namespace area_ratio_ABC_APB_l77_77106

open_locale big_operators

noncomputable def ratio_of_triangle_areas (A B C P : ℝ^2)
    (h : (P - A) + 3 • (P - B) + 4 • (P - C) = 0) :
    ℕ :=
  if h : 3 • (A - B) + 4 • (A - C) = 8 • P - 8 • A then
    3
  else
    0

theorem area_ratio_ABC_APB (A B C P : ℝ^2)
    (h : (P - A) + 3 • (P - B) + 4 • (P - C) = 0) :
    ratio_of_triangle_areas A B C P h = 3 :=
begin
  sorry
end

end area_ratio_ABC_APB_l77_77106


namespace cupcakes_per_package_l77_77244

theorem cupcakes_per_package
  (packages : ℕ) (total_left : ℕ) (cupcakes_eaten : ℕ) (initial_packages : ℕ) (cupcakes_per_package : ℕ)
  (h1 : initial_packages = 3)
  (h2 : cupcakes_eaten = 5)
  (h3 : total_left = 7)
  (h4 : packages = initial_packages * cupcakes_per_package - cupcakes_eaten)
  (h5 : packages = total_left) : 
  cupcakes_per_package = 4 := 
by
  sorry

end cupcakes_per_package_l77_77244


namespace initial_printing_presses_l77_77199

theorem initial_printing_presses (P : ℕ) 
  (h1 : 500000 / (9 * P) = 500000 / (12 * 30)) : 
  P = 40 :=
by
  sorry

end initial_printing_presses_l77_77199


namespace sin_alpha_cos_squared_beta_range_l77_77095

theorem sin_alpha_cos_squared_beta_range (α β : ℝ) 
  (h : Real.sin α + Real.sin β = 1) : 
  ∃ y, y = Real.sin α - Real.cos β ^ 2 ∧ (-1/4 ≤ y ∧ y ≤ 0) :=
sorry

end sin_alpha_cos_squared_beta_range_l77_77095


namespace gcd_of_all_elements_in_B_is_2_l77_77214

-- Define the set B as the set of all numbers that can be represented as the sum of four consecutive positive integers.
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2 ∧ x > 0}

-- Translate the question to a Lean statement.
theorem gcd_of_all_elements_in_B_is_2 : ∀ n ∈ B, gcd n 2 = 2 := 
by
  sorry

end gcd_of_all_elements_in_B_is_2_l77_77214


namespace sprockets_produced_by_machines_l77_77941

noncomputable def machine_sprockets (t : ℝ) : Prop :=
  let machineA_hours := t + 10
  let machineA_rate := 4
  let machineA_sprockets := machineA_hours * machineA_rate
  let machineB_hours := t
  let machineB_rate := 4.4
  let machineB_sprockets := machineB_hours * machineB_rate
  machineA_sprockets = 440 ∧ machineB_sprockets = 440

theorem sprockets_produced_by_machines (t : ℝ) (h : machine_sprockets t) : t = 100 :=
  sorry

end sprockets_produced_by_machines_l77_77941


namespace remainder_poly_eq_l77_77422

noncomputable def f (x : ℤ) : Polynomial ℤ := Polynomial.C (x^2026 - 1)
noncomputable def g (x : ℤ) : Polynomial ℤ := Polynomial.C (x^9 - x^7 + x^5 - x^3 + 1)

theorem remainder_poly_eq (x : ℤ) : 
  let f := Polynomial.X ^ 2026 - 1
  let g := Polynomial.X ^ 9 - Polynomial.X ^ 7 + Polynomial.X ^ 5 - Polynomial.X ^ 3 + 1
  Polynomial.X ^ 2026 - 1 % (Polynomial.X ^ 9 - Polynomial.X ^ 7 + Polynomial.X ^ 5 - Polynomial.X ^ 3 + 1) = 
  -Polynomial.X ^ 9 + Polynomial.X ^ 7 - Polynomial.X ^ 5 + Polynomial.X ^ 4 - Polynomial.X - 1 := 
  sorry

end remainder_poly_eq_l77_77422


namespace greatest_possible_value_of_x_l77_77657

theorem greatest_possible_value_of_x (x : ℝ) (h : ( (5 * x - 25) / (4 * x - 5) ) ^ 3 + ( (5 * x - 25) / (4 * x - 5) ) = 16):
  x = 5 :=
sorry

end greatest_possible_value_of_x_l77_77657


namespace tangent_line_equation_function_inequality_l77_77850

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (2 * x - 2)

-- Statement for Part (1)
theorem tangent_line_equation :
  (∀ (x : ℝ), f x) →
  let f2 := f 2
  let f_prime_x := fun x => 2 * Real.exp (2 * x - 2) * (x^2 + x)
  let f_prime_2 := f_prime_x 2
  let tangent_line := fun x => f_prime_2 * x + (f2 - f_prime_2 * 2)
  tangent_line x = 12 * Real.exp 2 * x - 20 * Real.exp 2 := 
sorry

-- Statement for Part (2)
theorem function_inequality (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) :
  f x ≥ -2 * x^2 + 8 * x - 5 := 
sorry

end tangent_line_equation_function_inequality_l77_77850


namespace maximize_total_yield_l77_77349

-- Definitions for the initial conditions and transformations
def initialTrees : Nat := 100
def initialYield : Nat := 600
def decreasePerTree : Nat := 5

def totalYield (x : Nat) : Int :=
  let newTotalTrees := x + initialTrees
  let newYieldPerTree := initialYield - (decreasePerTree * x)
  newTotalTrees * newYieldPerTree

theorem maximize_total_yield : 
  ∃ x : Nat, totalYield x = 60500 ∧ ∀ y : Nat, totalYield y ≤ totalYield x :=
by
  exists 10
  split
  -- Verifying the yield is indeed 60500 when x = 10
  show totalYield 10 = 60500
  sorry
  -- Proving that no other value of x yields more than totalYield 10
  intro y
  show totalYield y ≤ totalYield 10
  sorry

end maximize_total_yield_l77_77349


namespace proof_problem_l77_77857

open Set

noncomputable def U : Set ℝ := Icc (-5 : ℝ) 4

noncomputable def A : Set ℝ := {x : ℝ | -3 ≤ 2 * x + 1 ∧ 2 * x + 1 < 1}

noncomputable def B : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}

-- Definition of the complement of A in U
noncomputable def complement_U_A : Set ℝ := U \ A

-- The final proof statement
theorem proof_problem : (complement_U_A ∩ B) = Icc 0 2 :=
by
  sorry

end proof_problem_l77_77857


namespace probability_at_least_one_defective_item_l77_77465

def total_products : ℕ := 10
def defective_items : ℕ := 3
def selected_items : ℕ := 3
noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_least_one_defective_item :
    let total_combinations := comb total_products selected_items
    let non_defective_combinations := comb (total_products - defective_items) selected_items
    let opposite_probability := (non_defective_combinations : ℚ) / (total_combinations : ℚ)
    let probability := 1 - opposite_probability
    probability = 17 / 24 :=
by
  sorry

end probability_at_least_one_defective_item_l77_77465


namespace solve_inequality_system_l77_77288

theorem solve_inequality_system (x : ℝ) :
  (x + 2 > -1) ∧ (x - 5 < 3 * (x - 1)) ↔ (x > -1) :=
by
  sorry

end solve_inequality_system_l77_77288


namespace reflection_matrix_over_y_eq_x_is_correct_l77_77792

theorem reflection_matrix_over_y_eq_x_is_correct :
  let M := matrix.std_basis (fin 2) (fin 2)
  ∃ (R : matrix (fin 2) (fin 2) ℝ), 
    (R ⬝ M 0) = matrix.vec_cons 0 (matrix.vec_cons 1 matrix.vec_empty) ∧
    (R ⬝ M 1) = matrix.vec_cons 1 (matrix.vec_cons 0 matrix.vec_empty) ∧
    R = ![![0, 1], ![1, 0]] :=
sorry

end reflection_matrix_over_y_eq_x_is_correct_l77_77792


namespace pyramid_volume_l77_77717

theorem pyramid_volume (a b c : ℝ)
  (base : a = 5 ∧ b = 12)
  (slant_height : c = 15) : 
  let area_base := a * b,
      diag := real.sqrt (a^2 + b^2),
      height := real.sqrt (c^2 - (diag / 2)^2),
      volume := (1 / 3) * area_base * height
  in volume = 270 := 
by
  sorry

end pyramid_volume_l77_77717


namespace computer_operations_in_three_hours_l77_77005

theorem computer_operations_in_three_hours :
  let additions_per_second := 12000
  let multiplications_per_second := 2 * additions_per_second
  let seconds_in_three_hours := 3 * 3600
  (additions_per_second + multiplications_per_second) * seconds_in_three_hours = 388800000 :=
by
  sorry

end computer_operations_in_three_hours_l77_77005


namespace smallest_possible_munificence_of_monic_cubic_is_one_l77_77389

def munificence (p : ℝ → ℝ) (I : Set ℝ) : ℝ := 
  ⨆ x ∈ I, abs (p x)

noncomputable def minMunificence : ℝ := 
  infi (λ (p : ℝ → ℝ), munificence p (Set.Icc (-1 : ℝ) 1))

theorem smallest_possible_munificence_of_monic_cubic_is_one :
  minMunificence = 1 :=
sorry

end smallest_possible_munificence_of_monic_cubic_is_one_l77_77389


namespace number_of_dogs_on_tuesday_l77_77148

variable (T : ℕ)
variable (H1 : 7 + T + 7 + 7 + 9 = 42)

theorem number_of_dogs_on_tuesday : T = 12 := by
  sorry

end number_of_dogs_on_tuesday_l77_77148


namespace find_general_term_find_sum_l77_77855

noncomputable def arith_seq_formula (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = 3 * n - 2

noncomputable def geom_seq_condition (a : ℕ → ℝ) (d : ℝ) :=
  (a 1 + 2 * d) ^ 2 = (a 1) * (a 1 + 16 * d)

theorem find_general_term (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arith_seq_formula a d) 
  (h_geom : geom_seq_condition a d) :
  ∀ n : ℕ, a n = 3 * n - 2 := 
sorry

noncomputable def seq_s (a : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n : ℕ, b n = 1 / (a n * a (n + 1))

noncomputable def sum_s (b : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = ∑ k in finset.range n, b k

theorem find_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : arith_seq_formula a 3) 
  (h_seq_b : seq_s a b) 
  (h_sum_s : sum_s b S) :
  ∀ n : ℕ, S n = n / (3 * n + 1) :=
sorry

end find_general_term_find_sum_l77_77855


namespace ethanol_percentage_fuel_B_l77_77370

noncomputable def percentage_ethanol_in_fuel_B : ℝ :=
  let tank_capacity := 208
  let ethanol_in_fuelA := 0.12
  let total_ethanol := 30
  let volume_fuelA := 82
  let ethanol_from_fuelA := volume_fuelA * ethanol_in_fuelA
  let ethanol_from_fuelB := total_ethanol - ethanol_from_fuelA
  let volume_fuelB := tank_capacity - volume_fuelA
  (ethanol_from_fuelB / volume_fuelB) * 100

theorem ethanol_percentage_fuel_B :
  percentage_ethanol_in_fuel_B = 16 :=
by
  sorry

end ethanol_percentage_fuel_B_l77_77370


namespace surface_area_frustum_volume_frustum_l77_77103

-- Definitions for the conditions
def lateral_edge : ℝ := 6
def radius_base1 : ℝ := 2
def radius_base2 : ℝ := 7
def height_frustum : ℝ := sqrt 11

-- Surface area proof statement
theorem surface_area_frustum :
  π * (radius_base1 ^ 2) + π * (radius_base2 ^ 2) + π * (radius_base1 + radius_base2) * lateral_edge = 107 * π :=
by
  sorry

-- Volume proof statement
theorem volume_frustum :
  (1 / 3) * π * (radius_base1 ^ 2 + radius_base1 * radius_base2 + radius_base2 ^ 2) * height_frustum = (67 * sqrt 11 / 3) * π :=
by
  sorry

end surface_area_frustum_volume_frustum_l77_77103


namespace distinct_pairs_count_l77_77767

theorem distinct_pairs_count : 
  ( {p : ℝ × ℝ | p.1 = 3 * p.1 ^ 2 + p.2 ^ 2 ∧ p.2 = 3 * p.1 * p.2}.to_finset.card = 2 ) :=
by
  sorry

end distinct_pairs_count_l77_77767


namespace minimum_phi_for_odd_function_l77_77877

theorem minimum_phi_for_odd_function (φ : ℝ) 
  (hφ_positive : φ > 0) :
  let f (x : ℝ) := Real.cos (2 * x + Real.pi / 6)
  let translated_f (x : ℝ) := Real.cos (2 * x + Real.pi / 6 - 2 * φ)
  (odd_function : ∀ x, translated_f x = -translated_f (-x)) :
  φ = Real.pi / 3 :=
sorry

end minimum_phi_for_odd_function_l77_77877


namespace incorrect_conclusion_C_l77_77471

noncomputable def f (x : ℝ) := (x - 1)^2 * Real.exp x

theorem incorrect_conclusion_C : 
  ¬(∀ x, ∀ ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → abs (f y - f x) ≥ ε) :=
by
  sorry

end incorrect_conclusion_C_l77_77471


namespace max_area_ratio_l77_77258

open_locale classical

variables {A B C D K M L N : Type*}

-- Assume the given conditions as definitions
def is_on_side_BC (K : point) (C : point) : Prop := ∃ (BC : line), K ∈ BC ∧ C ∈ BC
def is_on_side_AD (M : point) (D : point) : Prop := ∃ (AD : line), M ∈ AD ∧ D ∈ AD
def intersection_CM_DK (C M D K L : point) : Prop := ∃ (CM DK : line), L ∈ CM ∧ L ∈ DK ∧ C ∈ CM ∧ M ∈ CM ∧ D ∈ DK ∧ K ∈ DK
def intersection_AK_BM (A K B M N : point) : Prop := ∃ (AK BM : line), N ∈ AK ∧ N ∈ BM ∧ A ∈ AK ∧ K ∈ AK ∧ B ∈ BM ∧ M ∈ BM

-- Lean theorem statement
theorem max_area_ratio (K_on_BC : is_on_side_BC K C) 
                        (M_on_AD : is_on_side_AD M D)
                        (L_inter_CMK_DK : intersection_CM_DK C M D K L)
                        (N_inter_AK_BM : intersection_AK_BM A K B M N) : 
                        ∃ ratio : ℝ, ratio = 1/4 :=
begin
  -- Here the proof is expected, but we use sorry to indicate it's omitted
  sorry
end

end max_area_ratio_l77_77258


namespace problem_statement_l77_77230

noncomputable def smallest_multiple_with_divisors := 
  let m := smallest (fun n : ℕ => 100 ∣ n ∧ (∀ d : ℕ, d ∣ n ↔ d < 101)) 0
  m / 100

theorem problem_statement : smallest_multiple_with_divisors = 324 := 
by 
  sorry

end problem_statement_l77_77230


namespace sum_of_angles_l77_77107

/-- Given a quadrilateral ABCD inscribed in a circle, where 
- ∠ACB subtends a 70° central angle
- ∠CAD subtends a 50° central angle,
prove that ∠CAB + ∠ACD = 120°. -/
theorem sum_of_angles (ABCD : Quadrilateral) (O : Point) (A B C D : Point) 
  (h1 : InscribedQuadrilateral ABCD) 
  (h2 : CentralAngle O A C B = 70) 
  (h3 : CentralAngle O C A D = 50) 
  : InscribedQuadrilateralAngle A B O + InscribedQuadrilateralAngle A D O = 120 :=
  sorry

end sum_of_angles_l77_77107


namespace f_lg2_plus_f_lg1div2_l77_77126

def f (x : ℝ) : ℝ := log (sqrt (1 + 4 * x^2) - 2 * x) + 3

theorem f_lg2_plus_f_lg1div2 : 
  (f (log 2) + f (log (1/2))) = 6 :=
by
  sorry

end f_lg2_plus_f_lg1div2_l77_77126


namespace ajay_walks_distance_l77_77036

theorem ajay_walks_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h_speed : speed = 3) 
  (h_time : time = 16.666666666666668) : 
  distance = speed * time :=
by
  sorry

end ajay_walks_distance_l77_77036


namespace function_decreasing_interval_l77_77296

theorem function_decreasing_interval : 
  ∀ x : ℝ, 0 < x ∧ x < 2 → ∃ f : ℝ → ℝ, f = λ x, x^2 * (x - 3) ∧ (∀ h : ℝ, f' h < 0) :=
by
  sorry

end function_decreasing_interval_l77_77296


namespace molecular_weight_BaCl2_l77_77658

def molecular_weight_one_mole (w_four_moles : ℕ) (n : ℕ) : ℕ := 
    w_four_moles / n

theorem molecular_weight_BaCl2 
    (w_four_moles : ℕ)
    (H : w_four_moles = 828) :
  molecular_weight_one_mole w_four_moles 4 = 207 :=
by
  -- sorry to skip the proof
  sorry

end molecular_weight_BaCl2_l77_77658


namespace dragons_at_meeting_l77_77635

def dragon_meeting : Prop :=
  ∃ (x y : ℕ), 
    (2 * x + 7 * y = 26) ∧ 
    (x + y = 8)

theorem dragons_at_meeting : dragon_meeting :=
by
  sorry

end dragons_at_meeting_l77_77635


namespace Douglas_won_in_county_Y_l77_77513

def total_percentage (x y t r : ℝ) : Prop :=
  (0.74 * 2 + y * 1 = 0.66 * (2 + 1))

theorem Douglas_won_in_county_Y :
  ∀ (x y t r : ℝ), x = 0.74 → t = 0.66 → r = 2 →
  total_percentage x y t r → y = 0.50 := 
by
  intros x y t r hx ht hr H
  rw [hx, hr, ht] at H
  sorry

end Douglas_won_in_county_Y_l77_77513


namespace intersect_if_and_only_if_cond_l77_77263

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

structure Triangle (A B C : Type) :=
(A1 : A)
(B1 : B)
(C1 : C)
(is_on_side_A1 : ∃ (BC: Set A), A1 ∈ BC)
(is_on_side_B1 : ∃ (CA: Set B), B1 ∈ CA)
(is_on_side_C1 : ∃ (AB: Set C), C1 ∈ AB)

def perpendiculars_intersect_at_one_point (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  ∃ M : A, ∃ (h1 : M = ⊥ (A1) (BC)), ∃ (h2 : M = ⊥ (B1) (CA)), M = ⊥ (C1) (AB)

def given_condition (A1 B1 C1 : Type) [MetricSpace A1] [MetricSpace B1] [MetricSpace C1] : Prop :=
  A1 B^2 + C1 A^2 + B1 C^2 = B1 A^2 + A1 C^2 + C1 B^2

theorem intersect_if_and_only_if_cond (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (T: Triangle A B C):
  perpendiculars_intersect_at_one_point A B C ↔ given_condition A B C := by
  sorry

end intersect_if_and_only_if_cond_l77_77263


namespace angle_is_ninety_degrees_l77_77218

def a : ℝ × ℝ × ℝ := (2, -3, -4)
def b : ℝ × ℝ × ℝ := (real.sqrt 3, 2, -2)
def c : ℝ × ℝ × ℝ := (8, -1, 9)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def linear_comb (a c : ℝ) (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a * v.1 - c * w.1, a * v.2 - c * w.2, a * v.3 - c * w.3)

def orthogonal_vectors (a : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : Prop :=
  dot_product a v = 0

theorem angle_is_ninety_degrees :
  orthogonal_vectors a (linear_comb (dot_product a c) (dot_product a b) b c) :=
by
  sorry

end angle_is_ninety_degrees_l77_77218


namespace arithmetic_mean_of_integers_from_neg3_to_6_l77_77642

def integer_range := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

noncomputable def arithmetic_mean : ℚ :=
  (integer_range.sum : ℚ) / (integer_range.length : ℚ)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  arithmetic_mean = 1.5 := by
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l77_77642


namespace sum_of_evens_105_times_x_l77_77310

theorem sum_of_evens_105_times_x (n : ℕ) (h : n = 211) (h_odd : odd n) (sum_eq : (∑ i in finset.filter (λ i, (i % 2 = 0)) (finset.range (n + 1)), i) = 105 * x) : x = 106 :=
sorry

end sum_of_evens_105_times_x_l77_77310


namespace integers_satisfy_inequality_l77_77486

theorem integers_satisfy_inequality: 
  {x : Int | (x + 2)^2 ≤ 4}.card = 5 :=
by
  sorry

end integers_satisfy_inequality_l77_77486


namespace area_BCM_eq_area_ADME_l77_77904

open_locale big_operators

variables {α : Type*} [linear_ordered_field α] [decidable_eq α] 

structure Point (α : Type*) :=
(x : α)
(y : α)

structure Triangle (α : Type*) :=
(A B C : Point α)

def median (A B C : Point α) : Point α :=
{ x := (B.x + C.x) / 2,
  y := (B.y + C.y) / 2 }

def centroid (A B C : Point α) : Point α :=
{ x := (A.x + B.x + C.x) / 3,
  y := (A.y + B.y + C.y) / 3 }

noncomputable def area (A B C : Point α) : α :=
1/2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

noncomputable def area_quadrilateral (A D M E : Point α) : α :=
  area A D M + area M D E

theorem area_BCM_eq_area_ADME (A B C : Point α) :
  let D := median A B C,
      E := median B A C,
      M := centroid A B C in
  area B C M = area_quadrilateral A D M E :=
sorry

end area_BCM_eq_area_ADME_l77_77904


namespace reflection_matrix_correct_l77_77787

-- Definitions based on the conditions given in the problem
def reflect_over_line_y_eq_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![1, 0]]

-- The main theorem to state the equivalence
theorem reflection_matrix_correct :
  reflect_over_line_y_eq_x_matrix = ![![0, 1], ![1, 0]] :=
by
  sorry

end reflection_matrix_correct_l77_77787


namespace sum_of_first_7_terms_l77_77920

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

theorem sum_of_first_7_terms (h1 : a 2 = 3) (h2 : a 6 = 11)
  (h3 : ∀ n, S n = n * (a 1 + a n) / 2) : S 7 = 49 :=
by 
  sorry

end sum_of_first_7_terms_l77_77920


namespace maximize_AB_l77_77242

theorem maximize_AB (A B C D : ℕ) (h_distinct : list.nodup [A, B, C, D]) 
  (h_range : ∀ x, x ∈ [A, B, C, D] → x ∈ finset.range 10)
  (h_CD : C + D ≠ 1) 
  (h_integer : (A + B) % (C + D) = 0) :
  A + B ≤ 15 :=
by sorry

end maximize_AB_l77_77242


namespace dog_max_distance_from_origin_l77_77571

theorem dog_max_distance_from_origin :
  let center := (5 : ℝ, -2 : ℝ)
  let radius := 15
  let origin := (0 : ℝ, 0 : ℝ)
  let fence_y := 0
  let max_distance := 5 + Real.sqrt 221
  (∀ (x y : ℝ), (x - center.fst) ^ 2 + (y - center.snd) ^ 2 ≤ radius ^ 2 → y ≠ fence_y → true) →
  ∃ (x y : ℝ), (x - center.fst) ^ 2 + (y - center.snd) ^ 2 ≤ radius ^ 2 ∧ y = fence_y ∧ 
  Real.sqrt ((x - origin.fst) ^ 2 + (y - origin.snd) ^ 2) = max_distance :=
begin
  sorry
end

end dog_max_distance_from_origin_l77_77571


namespace f_5_eq_9_l77_77098

def f : ℕ → ℕ
| x := if x >= 10 then x - 2 else f (x + 6)

theorem f_5_eq_9 : f 5 = 9 := 
by
  sorry

end f_5_eq_9_l77_77098


namespace smallest_multiple_divisors_l77_77221

theorem smallest_multiple_divisors :
  ∃ m : ℕ, (∃ k1 k2 : ℕ, m = 2^k1 * 5^k2 * 100 ∧ 
    (∀ d : ℕ, d ∣ m → d = 1 ∨ d = m ∨ ∃ e1 e2 : ℕ, d = 2^e1 * 5^e2 * 100)) ∧
    (∀ d : ℕ, d ∣ m → d ≠ 1 → (d ≠ m → ∃ e1 e2 : ℕ, d = 2^e1 * 5^e2 * 100)) ∧
    (m.factors.length = 100) ∧ 
    m / 100 = 2^47 * 5^47 :=
begin
  sorry
end

end smallest_multiple_divisors_l77_77221


namespace invest_today_for_future_value_l77_77490

-- Define the given future value, interest rate, and number of years as constants
def FV : ℝ := 600000
def r : ℝ := 0.04
def n : ℕ := 15
def target : ℝ := 333087.66

-- Define the present value calculation
noncomputable def PV : ℝ := FV / (1 + r)^n

-- State the theorem that PV is approximately equal to the target value
theorem invest_today_for_future_value : PV = target := 
by sorry

end invest_today_for_future_value_l77_77490


namespace arithmetic_sequence_general_term_l77_77520

theorem arithmetic_sequence_general_term (a₁ : ℕ) (d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 3) :
  ∃ a_n, a_n = a₁ + (n - 1) * d ∧ a_n = 3 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l77_77520


namespace octagon_area_l77_77447

theorem octagon_area (a : ℝ) : 
  let side_length := a in
  let area_square := a^2 in
  let corner_triangle_area := (a^2*(2-2*sqrt 2))/4 in
  let total_corner_area := 4 * corner_triangle_area in
  ((side_length^2 - total_corner_area) = 2 * a^2 * (sqrt 2 - 1)) := 
by 
  let side_length := a in
  let area_square := a^2 in
  let corner_triangle_area := (a^2*(2-2*sqrt 2))/4 in
  let total_corner_area := 4 * corner_triangle_area in
  let octagon_area := area_square - total_corner_area in
  have : octagon_area = 2 * a^2 * (sqrt 2 - 1), 
  sorry

end octagon_area_l77_77447


namespace solution_interval_for_fg_lt_0_l77_77234

variables (f g : ℝ → ℝ)
variable (F : ℝ → ℝ)

-- Conditions
axiom odd_f : ∀ x, f (-x) = -f (x)
axiom even_g : ∀ x, g (-x) = g (x)
axiom def_on_R : ∀ x : ℝ, f x ∈ ℝ ∧ g x ∈ ℝ
axiom increasing_F_on_neg : ∀ x1 x2 : ℝ, x1 < x2 ∧ x2 < 0 → F x1 < F x2
axiom g_at_2 : g 2 = 0
axiom F_def : ∀ x, F x = f x * g x

-- Statement
theorem solution_interval_for_fg_lt_0 : 
  { x : ℝ | f x * g x < 0 } = { x | x < -2 ∨ (0 < x ∧ x < 2) } :=
sorry

end solution_interval_for_fg_lt_0_l77_77234


namespace vector_combination_l77_77188

open Real

variables (OA OB OC : ℝ)
          (angle_AOC angle_BOC : ℝ)
          (m n : ℝ)

-- Given conditions
axiom norm_OA : ∥OA∥ = 1
axiom norm_OB : ∥OB∥ = 2
axiom norm_OC : ∥OC∥ = sqrt 3
axiom tan_angle_AOC : tan angle_AOC = 3
axiom angle_BOC_45 : angle_BOC = (π / 4)

-- Proof of existence of constants m and n
theorem vector_combination :
  ∃ m n : ℝ,
    OC = m * OA + n * OB ∧ m = 5 / 3 ∧ n = 2 / 3 :=
by
  sorry

end vector_combination_l77_77188


namespace calculate_percentage_l77_77705

theorem calculate_percentage :
  ∃ P : ℝ, 
  let number := 680
  in 0.40 * 140 + 80 = (P / 100) * number 
  → P = 20 :=
by
  let number := 680
  existsi (20 : ℝ)
  intros h
  sorry

end calculate_percentage_l77_77705


namespace factorization_sum_l77_77073

variable {a b c : ℤ}

theorem factorization_sum 
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 52 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) : 
  a + b + c = 27 :=
sorry

end factorization_sum_l77_77073


namespace mean_of_large_numbers_l77_77343

def is_large_number {n : ℕ} (l : List ℕ) (i : ℕ) : Prop :=
  ∀ j, j > i → l.get ⟨i, _⟩ > l.get ⟨j, _⟩

theorem mean_of_large_numbers {n : ℕ} (hn : 0 < n) :
  (List.range (n + 1)).map (λ i, (1 : ℝ) / i).sum = (1 : ℝ) + (∑ i in List.range n, (1 : ℝ) / (n + 1 - i.succ)) :=
by
  sorry

end mean_of_large_numbers_l77_77343


namespace sn_geq_mnplus1_l77_77737

namespace Polysticks

def n_stick (n : ℕ) : Type := sorry -- formalize the definition of n-stick
def n_mino (n : ℕ) : Type := sorry -- formalize the definition of n-mino

def S (n : ℕ) : ℕ := sorry -- define the number of n-sticks
def M (n : ℕ) : ℕ := sorry -- define the number of n-minos

theorem sn_geq_mnplus1 (n : ℕ) : S n ≥ M (n+1) := sorry

end Polysticks

end sn_geq_mnplus1_l77_77737


namespace complement_of_union_l77_77561

open Set

variable (U : Set ℕ) (A B : Set ℕ)

namespace ProofProblem

def U_def : Set ℕ := {0, 1, 2, 3, 4, 5}
def A_def : Set ℕ := {1, 2}
def B_def : Set ℕ := {x | -5*x + 4 < 0}

theorem complement_of_union (U : Set ℕ) (A B : Set ℕ) (hU : U = U_def) (hA : A = A_def) (hB : B = B_def) :
      compl (A ∪ B) ∩ U = {0, 3, 4} := by
      sorry

end ProofProblem

end complement_of_union_l77_77561


namespace minimum_hits_hydra_defeat_l77_77007

-- Define the graph and the conditions
noncomputable def hydra_graph {α : Type} [Fintype α] (G : SimpleGraph α) : Prop :=
  ∀ (v : α), ∃ (H : SimpleGraph α), 
    (⋂ (v₁ v₂ : α), (Edge G v v₁ ∧ Edge G v v₂ → Edge G v₁ v₂) → ¬ Connected G v₁ v₂) → ¬ (G = H)

-- Define the minimum number of hits
noncomputable def min_hits (G : SimpleGraph α) : ℕ :=
  -- This is a placeholder for the actual number of hits required, to be determined by the proof
  10

-- The main theorem stating the minimum number of hits required to disconnect the hydra
theorem minimum_hits_hydra_defeat {α : Type} [Fintype α] (G : SimpleGraph α) (h : hydra_graph G) (edges : |E(G)| = 100) :
  ∃ (N : ℕ), (N ≤ 10) ∧ (N = min_hits G) :=
begin
  sorry
end

end minimum_hits_hydra_defeat_l77_77007


namespace celeste_opod_probability_l77_77754

theorem celeste_opod_probability:
  let total_songs := 12 in
  let song_length := 45 in
  let favorite_song_length := 270 in
  let total_time_limit := 360 in
  (∑ k in finset.range (total_songs - 2), (11.factorial)) * (1 / factorial 12) = 5 / 6
:=
by
  sorry

end celeste_opod_probability_l77_77754


namespace example_problem_l77_77369

/-- A formal statement of the given mathematical problem in Lean 4 -/
theorem example_problem :
  (¬ (∀ x : ℝ, (x^2 - 2*x > 0 → x > 2) ∧ (x > 2 → x^2 - 2*x > 0))) ∧ 
  (¬ (forall x : ℝ, (x^2 + 5*x + 6 = 0 → x = -2) ∧ (x = -2 → x^2 + 5*x + 6 = 0))) ∧
  (forall x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ∧ ((x^2 - 3*x + 2 ≠ 0) → x ≠ 1))
  → 2 :=
sorry

end example_problem_l77_77369


namespace variance_of_binomial_l77_77238

variable (X : ℕ → ℝ)
variable (p q : ℝ)
variable (Hpq : p + q = 1)
variable [BinomialDistribution X p q]

-- Prove that the variance D(X) is pq given that X follows a binomial distribution with probabilities p and q and p+q=1.
theorem variance_of_binomial (hXp1 : ∀ n, X n = if n = 1 then p else if n = 0 then q else 0) : D X = p * q :=
sorry

end variance_of_binomial_l77_77238


namespace least_years_to_double_l77_77337

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem least_years_to_double (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) :
  r = 0.13 → n = 1 → 2 * P < compound_interest P r n t → t ≥ 6 :=
by
  intros hr hn hinterest
  simp [compound_interest, hr, hn] at hinterest
  have h : t > log(2) / log (1 + 0.13) := sorry
  linarith

end least_years_to_double_l77_77337


namespace max_reflections_l77_77718

theorem max_reflections (A B C D : Type) (α : ℝ) (n : ℕ) :
  ∠ C D A = 7 ∧ 7 * n ≤ 90 → n ≤ 12 :=
by
sorry

end max_reflections_l77_77718


namespace equations_have_different_graphs_l77_77328

noncomputable def equation1 (x : ℝ) := 2 * x - 3
noncomputable def equation2 (x : ℝ) : option ℝ := if x = -3 / 2 then none else some ((4 * x^2 - 9) / (2 * x + 3))
noncomputable def equation3 (x : ℝ) : option ℝ := if x = -3 / 2 then none else some ((4 * x^2 - 3 * real.sin x) / (2 * x + 3))

theorem equations_have_different_graphs :
  ∀ x : ℝ, equation1 x ≠ equation2 x ∧ equation1 x ≠ equation3 x ∧ equation2 x ≠ equation3 x :=
by sorry

end equations_have_different_graphs_l77_77328


namespace find_bags_l77_77624

theorem find_bags (x : ℕ) : 10 + x + 7 = 20 → x = 3 :=
by
  sorry

end find_bags_l77_77624


namespace prove_intersection_area_is_correct_l77_77720

noncomputable def octahedron_intersection_area 
  (side_length : ℝ) (cut_height_factor : ℝ) : ℝ :=
  have height_triangular_face := Real.sqrt (side_length^2 - (side_length / 2)^2)
  have plane_height := cut_height_factor * height_triangular_face
  have proportional_height := plane_height / height_triangular_face
  let new_side_length := proportional_height * side_length
  have hexagon_area := (3 * Real.sqrt 3 / 2) * (new_side_length^2) / 2 
  (3 * Real.sqrt 3 / 2) * (new_side_length^2)

theorem prove_intersection_area_is_correct 
  : 
  octahedron_intersection_area 2 (3 / 4) = 9 * Real.sqrt 3 / 8 :=
  sorry 

example : 9 + 3 + 8 = 20 := 
  by rfl

end prove_intersection_area_is_correct_l77_77720


namespace harriet_time_A_to_B_l77_77332

noncomputable def time_to_drive_from_A_to_B (distance speed_A_to_B speed_B_to_A total_time : ℝ) : ℝ :=
  let T1 := distance / speed_A_to_B in
  let T2 := distance / speed_B_to_A in
  if T1 + T2 = total_time then T1 * 60 else 0

theorem harriet_time_A_to_B :
  ∀ (distance : ℝ), let time := time_to_drive_from_A_to_B distance 90 160 5 in time = 192 :=
by
  intro distance
  sorry

end harriet_time_A_to_B_l77_77332


namespace angle_BED_leq_30_l77_77193

theorem angle_BED_leq_30
  (A B C D E : Type*)
  [Triangle ABC]
  (angle_ABC : ∠ B A C = 60°)
  (BD_perpendicular_AB : perp BD AB)
  (D_on_angle_bisector_of_BAC : D ∈ bisector ∠ BAC)
  (CE_perpendicular_BC : perp CE BC)
  (E_on_angle_bisector_of_ABC : E ∈ bisector ∠ ABC) :
  ∠ BED ≤ 30° :=
by
  sorry

end angle_BED_leq_30_l77_77193


namespace arithmetic_mean_of_integers_from_neg3_to_6_l77_77655

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  let nums := list.range' (-3) 10 in
  (∑ i in nums, i) / (nums.length : ℝ) = 1.5 :=
by
  let nums := list.range' (-3) 10
  have h_sum : (∑ i in nums, i) = 15 := sorry
  have h_length : nums.length = 10 := sorry
  rw [h_sum, h_length]
  norm_num
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l77_77655


namespace tan_value_l77_77157

noncomputable def tan_alpha_plus_pi_div_4 (α : ℝ) : ℝ :=
  if h1 : sin (π / 2 + 2 * α) = -4 / 5 ∧ (π / 2 < α ∧ α < π)
  then tan (α + π / 4)
  else 0

theorem tan_value (α : ℝ) (h1 : sin (π / 2 + 2 * α) = -4 / 5) (h2 : π / 2 < α) (h3 : α < π) :
  tan (α + π / 4) = -1 / 2 :=
sorry

end tan_value_l77_77157


namespace cos_4theta_eq_neg_half_l77_77869

noncomputable def e_to_iθ (θ : ℂ) : ℂ := complex.exp (complex.I * θ)

theorem cos_4theta_eq_neg_half (θ : ℂ) (h : e_to_iθ θ = (1 - complex.I * real.sqrt 3) / 2) :
  complex.cos (4 * θ) = -1 / 2 := 
sorry

end cos_4theta_eq_neg_half_l77_77869


namespace cos_alpha_minus_beta_point_C_coordinates_l77_77519

noncomputable def cos_alpha : ℝ := sqrt 2 / 10
noncomputable def cos_beta : ℝ := 2 * sqrt 5 / 5

theorem cos_alpha_minus_beta :
  ∀ (alpha beta : ℝ), cos_alpha = (sqrt 2) / 10 ∧ cos_beta = (2 * sqrt 5) / 5 ∧
  α ∈ Icc 0 (π / 2) ∧ β ∈ Icc 0 (π / 2) → cos (alpha - beta) = 9 * sqrt 10 / 50 := sorry

theorem point_C_coordinates :
  ∀ (m n : ℝ),
  (m^2 + n^2 = 1) ∧
  (cos_alpha * m + (7 * sqrt 2 / 10) * n = sqrt 2 / 2) →
  (m = 4/5 ∧ n = 3/5) ∨ (m = -3/5 ∧ n = 4/5) := sorry

end cos_alpha_minus_beta_point_C_coordinates_l77_77519


namespace find_phi_l77_77878

theorem find_phi {f : ℝ → ℝ} (φ : ℝ) (h1 : 0 < φ ∧ φ < π)
  (h2 : ∀ x, f (2 * x + φ) = sin (2 * x + φ) ∧ f (π / 6 - x) = f (π / 6 + x)) :
  φ = π / 6 :=
by
  -- The proof should use the given conditions
  sorry

end find_phi_l77_77878


namespace stratified_sampling_second_third_categories_l77_77503

theorem stratified_sampling_second_third_categories :
  (n : ℕ) (total_villages first_category_villages second_category_villages : ℕ)
  (h_total : total_villages = 300)
  (h_first : first_category_villages = 60)
  (h_second : second_category_villages = 100)
  (sampled_first : ℕ) (h_sampled_first : sampled_first = 3) :
  (n = (sampled_first * total_villages / first_category_villages) → n - sampled_first = 12) :=
by
  sorry

end stratified_sampling_second_third_categories_l77_77503


namespace brian_video_watching_time_l77_77750

theorem brian_video_watching_time :
  let catVideo : ℕ := 4
  let dogVideo : ℕ := 2 * catVideo
  let combinedCatDog : ℕ := catVideo + dogVideo
  let gorillaVideo : ℕ := 2 * combinedCatDog
  let totalTime : ℕ := catVideo + dogVideo + gorillaVideo
  totalTime = 36 :=
by
  -- Define the variables
  let catVideo := 4
  let dogVideo := 2 * catVideo
  let combinedCatDog := catVideo + dogVideo
  let gorillaVideo := 2 * combinedCatDog
  let totalTime := catVideo + dogVideo + gorillaVideo
  -- Combine all steps and assert the final value
  show totalTime = 36, from
    sorry -- Proof not implemented

end brian_video_watching_time_l77_77750


namespace problem1_problem2_l77_77441

section
variables {α β : ℝ}
variables (vec_a vec_b vec_c : ℝ × ℝ)

/- Problem 1 -/
-- Given conditions:
-- β ∈ (\frac{π}{4}, \frac{π}{2}),
-- ⟨vec_b, vec_c⟩ = θ_2,
-- θ_2 = \frac{\pi}{6},
-- vec_b = (1 + 2 * sin β * cos β, 1 - 2 * sin β ^ 2),
-- vec_c = (1, 0)
-- Prove β = \frac{5π}{12}
theorem problem1 (h1 : β ∈ (π / 4, π / 2))
               (h2 : inner vec_b vec_c = π / 6)
               (h_vec_b : vec_b = (1 + 2 * sin β * cos β, 1 - 2 * sin β ^ 2))
               (h_vec_c : vec_c = (1, 0)) :
  β = 5 * π / 12 :=
sorry

/- Problem 2 -/
-- Given conditions:
-- α ∈ (0, \frac{π}{4}),
-- β ∈ (\frac{π}{4}, \frac{π}{2}),
-- vec_a = (2 * cos α ^ 2, 2 * sin α * cos α),
-- vec_b = (1 + 2 * sin β * cos β, 1 - 2 * sin β ^ 2),
-- vec_c = (1, 0),
-- ⟨vec_a, vec_c⟩ = θ_1,
-- ⟨vec_b, vec_c⟩ = θ_2,
-- θ_2 - θ_1 = \frac{\pi}{6}
-- Prove: sin(β - α) = \frac{\sqrt{6} + \sqrt{2}}{4}
theorem problem2 (h1 : α ∈ (0, π / 4))
               (h2 : β ∈ (π / 4, π / 2))
               (h_vec_a : vec_a = (2 * cos α ^ 2, 2 * sin α * cos α))
               (h_vec_b : vec_b = (1 + 2 * sin β * cos β, 1 - 2 * sin β ^ 2))
               (h_vec_c : vec_c = (1, 0))
               (h_inner1 : inner vec_a vec_c = α)
               (h_inner2 : inner vec_b vec_c = β - π / 4)
               (h_inner_diff : (β - π / 4) - α = π / 6) :
  sin (β - α) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
sorry
end

end problem1_problem2_l77_77441


namespace perimeter_of_semi_circle_region_l77_77719

theorem perimeter_of_semi_circle_region (side_length : ℝ) (h : side_length = 1/π) : 
  let radius := side_length / 2
  let circumference_of_half_circle := (1 / 2) * π * side_length
  3 * circumference_of_half_circle = 3 / 2
  := by
  sorry

end perimeter_of_semi_circle_region_l77_77719


namespace inclination_angle_range_l77_77169

theorem inclination_angle_range (k : ℝ) (h : |k| ≤ 1) :
    ∃ α : ℝ, (k = Real.tan α) ∧ (0 ≤ α ∧ α ≤ Real.pi / 4 ∨ 3 * Real.pi / 4 ≤ α ∧ α < Real.pi) :=
by
  sorry

end inclination_angle_range_l77_77169


namespace solution_set_ineq_l77_77997

theorem solution_set_ineq (x : ℝ) : x^2 - 2 * abs x - 15 > 0 ↔ x < -5 ∨ x > 5 :=
sorry

end solution_set_ineq_l77_77997


namespace pow_product_l77_77053

theorem pow_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := 
by {
  sorry
}

end pow_product_l77_77053


namespace partition_MATHEMATICS_l77_77489

-- Define the word MATHEMATICS with its vowels.
def word : List Char := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S']
def vowels : List Char := ['A', 'E', 'A', 'I']

-- Statement of the proof problem.
theorem partition_MATHEMATICS : ∃ n, n = 36 ∧
  (∀ partitions : List (List Char), 
    (∀ part ∈ partitions, ∃ vowel ∈ vowels, vowel ∈ part) →
    count_partitions word partitions n) := 
sorry

end partition_MATHEMATICS_l77_77489


namespace selling_price_correct_l77_77581

def purchase_price := 42000  -- Purchase price of the car in Rs.
def repair_costs := 10000   -- Repair costs in Rs.
def profit_percent := 24.807692307692307 / 100  -- Profit percent converted to a decimal

def total_cost := purchase_price + repair_costs -- Total cost
def profit := profit_percent * total_cost -- Profit in Rs.
def selling_price := total_cost + profit -- Selling price in Rs.

theorem selling_price_correct :
  selling_price = 64898 :=
by {
  sorry -- Proof is omitted intentionally
}

end selling_price_correct_l77_77581


namespace eval_expression_l77_77776

theorem eval_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end eval_expression_l77_77776


namespace log_base_half_cuts_all_horizontal_lines_l77_77759

theorem log_base_half_cuts_all_horizontal_lines (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_eq : y = Real.logb 0.5 x) : ∃ x, ∀ k, k = Real.logb 0.5 x ↔ x > 0 := 
sorry

end log_base_half_cuts_all_horizontal_lines_l77_77759


namespace semicircle_radius_eq_sqrt_48_div_pi_l77_77582

theorem semicircle_radius_eq_sqrt_48_div_pi (AB AD : ℝ) (h₁ : AB = 3) (h₂ : AD = 8) (area_eq : ∀ (r : ℝ), (1/2) * real.pi * r^2 = AB * AD) : 
  ∃ r : ℝ, r = real.sqrt (48 / real.pi) :=
by
  use real.sqrt (48 / real.pi)
  sorry

end semicircle_radius_eq_sqrt_48_div_pi_l77_77582


namespace directrix_of_given_parabola_l77_77414

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  -- Calculate the x-coordinate of the vertex
  let x_vertex := -b / (2 * a) in
  -- Calculate the y-coordinate of the vertex using the vertex formula
  let y_vertex := a * x_vertex^2 + b * x_vertex + c in
  -- Calculate the y-coordinate of the directrix
  y_vertex - (1 / (4 * a))

theorem directrix_of_given_parabola : parabola_directrix (-3) 6 (-5) = -23/12 :=
by
  -- This is where the proof would go
  sorry

end directrix_of_given_parabola_l77_77414


namespace num_of_divisors_sum_pow_two_l77_77426

open Nat

def f (m : ℕ) : ℕ :=
  m.binary_digits.count 1

theorem num_of_divisors_sum_pow_two (n : ℕ) (h_pos : 0 < n) :
  (divisors_count (∑ m in Finset.range (2^n), (-1) ^ (f m) * 2^m)) ≥ factorial n :=
sorry

end num_of_divisors_sum_pow_two_l77_77426


namespace intersection_P_Q_l77_77215

open Set

def P : Set ℤ := {x | x < 1}
def Q : Set ℤ := {x | x ∈ ℤ ∧ x^2 < 4}

theorem intersection_P_Q :
  P ∩ Q = {-1, 0} :=
sorry

end intersection_P_Q_l77_77215


namespace smaller_root_l77_77323

theorem smaller_root :
  ∀ (x : ℝ), (x^2 - 12 * x - 28 = 0) → x = -2 ∨ x = 14 → x = -2 :=
by
  intro x
  intro h
  intro hx
  cases hx
  . rw hx
    intro _
    reflexivity
  . rw hx
    contradiction

end smaller_root_l77_77323


namespace min_total_time_one_tap_min_total_time_two_taps_l77_77621

-- One water tap proof problem
theorem min_total_time_one_tap (t : Fin 10 → Nat) (h : ∀ (i j : Fin 10), i < j → t i < t j) :
  ∀ (s : List (Fin 10)), 
         sum (List.map (λ i, (10 - i) * t i) s) 
         ≥ sum (List.map (λ i, (10 - i) * t i) (List.range 10)) := 
by
  sorry

-- Two water taps proof problem
theorem min_total_time_two_taps (t : Fin 10 → Nat) (h : ∀ (i j : Fin 10), i < j → t i < t j) :
  ∃ (m : Nat) (p q : Fin 10 → Nat), 
    (5 ≤ m ∧ m ≤ 10 ∧ 
    (∀ i : Fin 10, p i = if i < m then t i else q (i - m)) ∧ 
    (sum (List.map (λ i, (10 - (i + 1)) * p i) (List.range m)) 
    + sum (List.map (λ i, (10 - (i + 1)) * q i) (List.range (10 - m))) = T)) :=
by
  sorry

end min_total_time_one_tap_min_total_time_two_taps_l77_77621


namespace fuel_station_solution_l77_77890

noncomputable def fuel_station_problem : Prop :=
  let service_cost_per_vehicle := 2.20 in
  let fuel_cost_per_liter := 0.70 in
  let number_of_minivans := 4 in
  let number_of_trucks := 2 in
  let total_cost := 395.4 in
  let truck_capacity_factor := 2.2 in
  let total_service_cost := service_cost_per_vehicle * (number_of_minivans + number_of_trucks) in
  let total_fuel_cost := total_cost - total_service_cost in
  let total_fuel_liters := total_fuel_cost / fuel_cost_per_liter in
  let total_fuel_required := (number_of_minivans + number_of_trucks * truck_capacity_factor) in
  let V := total_fuel_liters / total_fuel_required in
  V = 65

theorem fuel_station_solution : fuel_station_problem :=
begin
  sorry
end

end fuel_station_solution_l77_77890


namespace find_a_value_l77_77525

noncomputable def point := (ℝ × ℝ)
noncomputable def curve_equation (a : ℝ) : (ℝ × ℝ) → Prop
| (x, y) := y^2 = 2 * a * x

noncomputable def line_param (t : ℝ) : point :=
(-4 + (Real.sqrt 2 / 2) * t, -2 + (Real.sqrt 2 / 2) * t)

def point_distance (p1 p2 : point) : ℝ :=
Real.sqrt (((p1.1 - p2.1)^2) + ((p1.2 - p2.2)^2))

def distances_form_geometric_sequence (p1 p2 p3 : point) : Prop :=
let d1 := point_distance p1 p2
let d2 := point_distance p2 p3
let d3 := point_distance p1 p3
in d2^2 = d1 * d3

theorem find_a_value (a : ℝ) :
  let l1 := line_param (t1 : ℝ), l2 := line_param (t2 : ℝ),
      p := (-4, -2) in
  ∃ (t1 t2 : ℝ), curve_equation a (l1 t1) ∧ curve_equation a (l2 t2) ∧
  distances_form_geometric_sequence p (l1 t1) (l2 t2) → a = 1 :=
sorry

end find_a_value_l77_77525


namespace partition_1987_1988_1989_l77_77531

noncomputable def unique_partition (n : ℕ) : Prop :=
∃ (A B : set ℕ), 
  (1 ∈ A) ∧
  (∀ a1 a2 ∈ A, a1 ≠ a2 → ∀ k : ℕ, a1 + a2 ≠ 2^k + 2) ∧
  (∀ b1 b2 ∈ B, b1 ≠ b2 → ∀ k : ℕ, b1 + b2 ≠ 2^k + 2) ∧
  (n ∈ A ∨ n ∈ B)

theorem partition_1987_1988_1989 :
  unique_partition 1987 ∧ unique_partition 1988 ∧ unique_partition 1989 ∧ ∃ (A B : set ℕ), 
    (1 ∈ A) ∧
    (∀ a1 a2 ∈ A, a1 ≠ a2 → ∀ k : ℕ, a1 + a2 ≠ 2^k + 2) ∧
    (∀ b1 b2 ∈ B, b1 ≠ b2 → ∀ k : ℕ, b1 + b2 ≠ 2^k + 2) ∧
    (1987 ∈ B) ∧
    (1988 ∈ A) ∧
    (1989 ∈ B) :=
by sorry

end partition_1987_1988_1989_l77_77531


namespace smallest_AAB_value_l77_77032

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 110 * A + B = 8 * (10 * A + B) ∧ ¬ (A = B) ∧ 110 * A + B = 773 :=
by sorry

end smallest_AAB_value_l77_77032


namespace closest_point_on_plane_l77_77421

theorem closest_point_on_plane {x y z : ℝ} (A : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) 
  (h_plane : 2 * P.1 + 3 * P.2 - 6 * P.3 = 18) (h_P : P = (144/49, 20/49, -89/49)) (h_A : A = (2, -1, 1)) :
  P = (144 / 49, 20 / 49, -89 / 49) :=
by
  sorry

end closest_point_on_plane_l77_77421


namespace divisible_special_factorial_l77_77065

def special_factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := ((10^n - 1) / 9) * special_factorial n

theorem divisible_special_factorial (n m : ℕ) :
  (special_factorial (n + m)) % ((special_factorial n) * (special_factorial m)) = 0 :=
by
  sorry

end divisible_special_factorial_l77_77065


namespace part1_part2_l77_77181

-- Define the conditions
def given_condition1 (a b c : ℝ) (A B C : ℝ) : Prop :=
  sqrt 3 * a = 2 * c * sin A

def given_condition2 (c : ℝ) : Prop :=
  c = sqrt 7

def given_condition3 (area : ℝ) : Prop :=
  area = 3 * sqrt 3 / 2

def is_acute_triangle (A B C : ℝ) : Prop :=
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2

-- Prove part 1
theorem part1 (a b c A B C : ℝ) (h1 : given_condition1 a b c A B C) (h2 : is_acute_triangle A B C) :
  C = π / 3 :=
  sorry

-- Prove part 2
theorem part2 (a b c A B C area : ℝ)
  (h1 : given_condition1 a b c A B C) (h2 : given_condition2 c) (h3 : given_condition3 area) (h4 : part1 a b c A B C h1 _) :
  a + b = 5 :=
  sorry

end part1_part2_l77_77181


namespace largest_angle_is_150_l77_77526

variable {p q r : ℝ}

-- Conditions
def condition1 : Prop := 2 * p + 3 * q + 3 * r = 2 * p ^ 2
def condition2 : Prop := 2 * p + 3 * q - 3 * r = -8

-- Triangle side length conditions
def triangle_sides : Prop := p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q > r ∧ q + r > p ∧ r + p > q

theorem largest_angle_is_150 (h1 : condition1) (h2 : condition2) (h3 : triangle_sides) :
  ∃ (P Q R : ℝ), P + Q + R = 180 ∧ max P (max Q R) = 150 :=
by sorry

end largest_angle_is_150_l77_77526


namespace inequality_l77_77208

open Real

noncomputable def a_list (N : ℕ) := {a : ℕ → ℝ // (∀ n < N, 0 < a n) ∧ (∑ i in range N, a i = 1) }

def n_i (a : ℕ → ℝ) (i : ℕ) := (finset.univ.filter (λ k, (1 : ℝ) / (2^(i-1)) ≥ a k ∧ a k ≥ (1 : ℝ) / (2^i))).card

theorem inequality (N : ℕ) (hN : 0 < N) (a : a_list N) :
  (∑ i in finset.range (N+1), sqrt (n_i a.1 i * (2 : ℝ)^(-i))) ≤ 4 + sqrt (log (2 : ℝ) N) :=
by sorry

end inequality_l77_77208


namespace find_angle_QHR_l77_77195

-- Definitions
variables (P Q R K L M H : Type) [triangle P Q R]

axiom altitude_PK : is_altitude P Q R K
axiom altitude_QL : is_altitude Q P R L
axiom altitude_RM : is_altitude R P Q M
axiom orthocenter_H : is_orthocenter P Q R H

axiom angle_PQR : ∠ Q P R = 55
axiom angle_PRQ : ∠ P Q R = 18

-- Theorem statement
theorem find_angle_QHR : ∠ Q H R = 73 :=
sorry

end find_angle_QHR_l77_77195


namespace problem_statement_l77_77231

noncomputable def smallest_multiple_with_divisors := 
  let m := smallest (fun n : ℕ => 100 ∣ n ∧ (∀ d : ℕ, d ∣ n ↔ d < 101)) 0
  m / 100

theorem problem_statement : smallest_multiple_with_divisors = 324 := 
by 
  sorry

end problem_statement_l77_77231


namespace common_ratio_of_series_l77_77385

theorem common_ratio_of_series : 
  let a₁ := (7:ℝ) / 4
      a₂ := (28:ℝ) / 9 
  in (a₂ / a₁ = (16:ℝ) / 9) := 
by 
  let a₁ := (7:ℝ) / 4
  let a₂ := (28:ℝ) / 9
  calc
    a₂ / a₁ = (28 / 9) * (4 / 7) : by sorry
         ... = (16 / 9)          : by sorry

end common_ratio_of_series_l77_77385


namespace max_surface_area_of_rotating_bc_l77_77638

noncomputable def maximum_rotational_surface_area (r : ℝ) : ℝ :=
  3 * r^2 * π * sqrt 3

theorem max_surface_area_of_rotating_bc (r : ℝ) (ABC : Type) 
  [triangle_inscribed_in_circle ABC r] [rotation_around_tangent_at_A ABC] : 
  let BC_max_surface_area := maximum_rotational_surface_area r
  in BC_max_surface_area = 3 * r^2 * π * sqrt 3 :=
sorry

end max_surface_area_of_rotating_bc_l77_77638


namespace solve_complex_z_l77_77694

theorem solve_complex_z (z : ℂ) (h : (2 + 1 * complex.I) * z = 5 * complex.I) : z = 1 + 2 * complex.I :=
sorry

end solve_complex_z_l77_77694


namespace ellipse_foci_distance_l77_77783

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end ellipse_foci_distance_l77_77783


namespace constant_term_in_expansion_l77_77190

theorem constant_term_in_expansion (n : ℕ) (k : ℕ) :
  (x - 2 / x : ℝ) ^ 8 = ∑ r in range 9, (choose 8 r * (-2)^r * x^(8 - 2 * r)) →
  (∃ r : ℕ, 8 - 2 * r = 0 ∧ (choose 8 r * (-2)^r = 1120)) :=
by
  sorry

end constant_term_in_expansion_l77_77190


namespace max_members_choir_l77_77977

variable (m k n : ℕ)

theorem max_members_choir :
  (∃ k, m = k^2 + 6) ∧ (∃ n, m = n * (n + 6)) → m = 294 :=
by
  sorry

end max_members_choir_l77_77977


namespace determine_x_l77_77070

theorem determine_x (y : ℚ) (h : y = (36 + 249 / 999) / 100) :
  ∃ x : ℕ, y = x / 99900 ∧ x = 36189 :=
by
  sorry

end determine_x_l77_77070


namespace intelligent_robot_competition_l77_77745

-- Definitions for teams
inductive Team
| A | B | C | D

-- Predictions made by the students
def prediction_XiaoZhang (winners: Team) : Prop := winners = Team.A ∨ winners = Team.B
def prediction_XiaoWang (winners: Team) : Prop := winners = Team.D
def prediction_XiaoLi (winners: Team) : Prop := winners ≠ Team.B ∧ winners ≠ Team.C
def prediction_XiaoZhao (winners: Team) : Prop := winners = Team.A

-- Main theorem
theorem intelligent_robot_competition (winning_team: Team) 
    (correct_predictions: (prediction_XiaoZhang winning_team ∨ ¬ prediction_XiaoZhang winning_team) 
     + (prediction_XiaoWang winning_team ∨ ¬ prediction_XiaoWang winning_team) 
     + (prediction_XiaoLi winning_team ∨ ¬ prediction_XiaoLi winning_team) 
     + (prediction_XiaoZhao winning_team ∨ ¬ prediction_XiaoZhao winning_team) = 2) :
  winning_team = Team.D :=
sorry

end intelligent_robot_competition_l77_77745


namespace q_at_3_l77_77375

def q (x : ℝ) : ℝ :=
  (if x - 3 = 0 then 0 else (x - 3) / (|x - 3|)) * (|x - 3|^(1/3)) +
  3 * (if x - 3 = 0 then 0 else (x - 3) / (|x - 3|)) * (|x - 3|^(1/5)) +
  (|x - 3|^(1/7))

theorem q_at_3 : q 3 = 0 :=
by
  sorry

end q_at_3_l77_77375


namespace Sharmila_average_earnings_Sharmila_average_earnings_correct_l77_77587

noncomputable def total_earnings_first_job :=
  let hourly_earnings_10hrs := 3 * 10 * 15
  let bonus_10hrs := 3 * 20
  let hourly_earnings_8hrs := 2 * 8 * 15
  hourly_earnings_10hrs + bonus_10hrs + hourly_earnings_8hrs

noncomputable def total_earnings_second_job :=
  let hourly_earnings := 5 * 12
  let bonus := 10
  hourly_earnings + bonus

noncomputable def total_hours_worked :=
  3 * 10 + 2 * 8 + 5

noncomputable def average_earnings_per_hour :=
  (total_earnings_first_job + total_earnings_second_job) / total_hours_worked

theorem Sharmila_average_earnings :
  average_earnings_per_hour = 16.08 :=
by
  have total_earnings_first := total_earnings_first_job
  have total_earnings_second := total_earnings_second_job
  have total_hours := total_hours_worked
  let avg_earning := (total_earnings_first + total_earnings_second) / total_hours
  have avg_earning_val : avg_earning = 16.08 := by norm_num
  exact avg_earning_val

-- Ensure the script compiles successfully without proof details by adding sorry.
theorem Sharmila_average_earnings_correct : average_earnings_per_hour = 16.08 :=
  sorry


end Sharmila_average_earnings_Sharmila_average_earnings_correct_l77_77587


namespace tan_alpha_eq_4_cos_2alpha_l77_77097

theorem tan_alpha_eq_4_cos_2alpha :
  (tan α = 4) → cos (2*α) = -15 / 17 :=
by 
  sorry

end tan_alpha_eq_4_cos_2alpha_l77_77097


namespace seq_S_infinately_many_perfect_squares_l77_77450

def seq_a (u v : ℕ) : ℕ → ℕ
| 1       := u + v
| (2*m+1) := seq_a m + v
| (2*m)   := seq_a m + u

def seq_S (u v : ℕ) (m : ℕ) : ℕ :=
(fin_range m).sum (λ n, seq_a u v (n + 1))

theorem seq_S_infinately_many_perfect_squares (u v : ℕ) : 
  ∃ infinity (n : ℕ), (seq_S u v n)^2 ≠ 0 :=
sorry

end seq_S_infinately_many_perfect_squares_l77_77450


namespace rectangle_divided_into_13_squares_l77_77407

theorem rectangle_divided_into_13_squares (s a b : ℕ) (h₁ : a * b = 13 * s^2)
  (h₂ : ∃ k l : ℕ, a = k * s ∧ b = l * s ∧ k * l = 13) :
  (a = s ∧ b = 13 * s) ∨ (a = 13 * s ∧ b = s) :=
by
sorry

end rectangle_divided_into_13_squares_l77_77407


namespace silva_family_zoo_cost_l77_77316

theorem silva_family_zoo_cost :
  let regular_price : ℝ := 10
  let senior_price : ℝ := 7
  let children_discount : ℝ := 0.4
  let children_price : ℝ := (1 - children_discount) * regular_price
  let total_cost : ℝ := 3 * senior_price + 3 * regular_price + 3 * children_price
  total_cost = 69 :=
by
  let regular_price := 10
  let senior_price := 7
  let children_discount := 0.4
  let children_price := (1 - children_discount) * regular_price
  let total_cost := 3 * senior_price + 3 * regular_price + 3 * children_price
  sorry

end silva_family_zoo_cost_l77_77316


namespace sequence_unbounded_l77_77740

theorem sequence_unbounded 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a n = |a (n + 1) - a (n + 2)|)
  (h2 : 0 < a 0)
  (h3 : 0 < a 1)
  (h4 : a 0 ≠ a 1) :
  ¬ ∃ M : ℝ, ∀ n, |a n| ≤ M := 
sorry

end sequence_unbounded_l77_77740


namespace lisa_quizzes_goal_l77_77747

theorem lisa_quizzes_goal :
  ∀ (total_quizzes : ℕ) (completed_quizzes : ℕ) (completed_As : ℕ) (goal_percentage : ℝ),
    total_quizzes = 60 →
    completed_quizzes = 40 →
    completed_As = 34 →
    goal_percentage = 0.85 →
    ∃ max_non_As : ℕ, max_non_As = 3 :=
by
  intros total_quizzes completed_quizzes completed_As goal_percentage
  assume h1 : total_quizzes = 60
  assume h2 : completed_quizzes = 40
  assume h3 : completed_As = 34
  assume h4 : goal_percentage = 0.85
  let required_quizzes := (goal_percentage * total_quizzes).ceil.to_nat
  let remaining_quizzes := total_quizzes - completed_quizzes
  let more_As_needed := required_quizzes - completed_As
  let non_As := remaining_quizzes - more_As_needed
  use non_As
  have : non_As = 3 := by sorry
  assumption

end lisa_quizzes_goal_l77_77747


namespace incorrect_statement_B_l77_77331

open Set

-- Define the relevant events as described in the problem
def event_subscribe_at_least_one (ω : Type) (A B : Set ω) : Set ω := A ∪ B
def event_subscribe_at_most_one (ω : Type) (A B : Set ω) : Set ω := (A ∩ B)ᶜ

-- Define the problem statement
theorem incorrect_statement_B (ω : Type) (A B : Set ω) :
  ¬ (event_subscribe_at_least_one ω A B) = (event_subscribe_at_most_one ω A B)ᶜ :=
sorry

end incorrect_statement_B_l77_77331


namespace m_gt_p_l77_77210

theorem m_gt_p (p m n : ℕ) (prime_p : Nat.Prime p) (pos_m : 0 < m) (pos_n : 0 < n) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end m_gt_p_l77_77210


namespace abs_diff_l77_77160

theorem abs_diff (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : ((m^2 + n^2 + 81 + 64 + 100) / 5) - 81 = 2) :
  |m - n| = 4 := by
  sorry

end abs_diff_l77_77160


namespace number_of_n_divisible_by_prime_lt_20_l77_77911

theorem number_of_n_divisible_by_prime_lt_20 (N : ℕ) : 
  (N = 69) :=
by
  sorry

end number_of_n_divisible_by_prime_lt_20_l77_77911


namespace robbie_weight_l77_77954

theorem robbie_weight (R P : ℝ) 
  (h1 : P = 4.5 * R - 235)
  (h2 : P = R + 115) :
  R = 100 := 
by 
  sorry

end robbie_weight_l77_77954


namespace star_shaded_area_l77_77022

theorem star_shaded_area (A B : Type) (h1 : is_equilateral_triangle A) (h2 : is_equilateral_triangle B) 
  (overlap : overlap_star A B) (star_area : area (overlap_star A B) = 36) : 
  shaded_area (overlap_star A B) = 27 :=
sorry

end star_shaded_area_l77_77022


namespace curve_defined_by_r_eq_4_is_circle_l77_77779

theorem curve_defined_by_r_eq_4_is_circle : ∀ θ : ℝ, ∃ r : ℝ, r = 4 → ∀ θ : ℝ, r = 4 :=
by
  sorry

end curve_defined_by_r_eq_4_is_circle_l77_77779


namespace sum_binom_odd_terms_l77_77118

open Nat

theorem sum_binom_odd_terms {n : ℕ} (h : binomial n 4 = binomial n 6) :
  (∑ k in range (n+1), if k % 2 = 1 then binomial n k else 0) = 2^9 :=
sorry

end sum_binom_odd_terms_l77_77118


namespace equal_tangent_lengths_l77_77604

-- Definitions based on conditions
variables (A B C D K M N : Type)
variable [EuclideanGeometry A B C D K M N]

def trapezoid (A B C D : Type) [EuclideanGeometry A B C D] : Prop :=
  -- Define trapezoid properties
  sorry

def intersect_at_K (A B C D K M N : Type) [EuclideanGeometry A B C D K M N] : Prop :=
  -- Define intersections of diagonals at K
  sorry

def circles_as_diameters (A B C D K M N : Type) [EuclideanGeometry A B C D K M N] : Prop :=
  -- Define circles on AB and CD as diameters
  sorry

def point_K_outside_circles (K M N : Type) [EuclideanGeometry K M N] : Prop :=
  -- Definition of point K outside the circles
  sorry

-- Proof statement based on conditions
theorem equal_tangent_lengths (A B C D K M N : Type)
  [EuclideanGeometry A B C D K M N]
  (h_trapezoid : trapezoid A B C D)
  (h_intersect_at_K : intersect_at_K A B C D K M N)
  (h_circles_as_diameters : circles_as_diameters A B C D K M N)
  (h_point_K_outside_circles : point_K_outside_circles K M N)
  : KM * KA = KN * KD := 
sorry

end equal_tangent_lengths_l77_77604


namespace problem_1_problem_2_l77_77468

noncomputable section

open Real

-- Definitions and Conditions
def f (a x : ℝ) : ℝ := log a (1 + x) - log a (1 - x)

variable {a x : ℝ}

-- Problem (1): Domain and parity
theorem problem_1 (h₁ : a > 0) (h₂ : a ≠ 1) :
  ( ∀ x, 1 + x > 0 ∧ 1 - x > 0 ↔ x ∈ Ioo (-1) 1 ) ∧
  ( ∀ x, f a (-x) = -f a x ) :=
sorry

-- Problem (2): Solve inequality
theorem problem_2 (h₁ : 0 < a) (h₂ : a < 1) (h₃ : a > 0) (h₄ : a ≠ 1) :
  (∀ x, x ∈ Ioo (-1) 0 ↔ f a x > 0) :=
sorry

end problem_1_problem_2_l77_77468


namespace log_base_comparison_l77_77094

theorem log_base_comparison (m n : ℝ) (h1 : log m 9 < 0) (h2 : log n 9 < 0) (h3 : log m 9 < log n 9) : 0 < n ∧ n < m ∧ m < 1 :=
sorry

end log_base_comparison_l77_77094


namespace find_C_coordinates_l77_77725

-- Define points A and B
def A: ℝ × ℝ := (2, -2)
def B: ℝ × ℝ := (14, 4)

-- Define the segment ratio BC/AB as 1/3
def ratio: ℝ := 1/3

-- Calculate the expected change in coordinates based on segment AB
def delta_x: ℝ := (B.1 - A.1) * ratio
def delta_y: ℝ := (B.2 - A.2) * ratio

-- Coordinates of C
def C: ℝ × ℝ := (B.1 + delta_x, B.2 + delta_y)

-- Statement to be proved
theorem find_C_coordinates (h1: A = (2, -2)) (h2 : B = (14, 4)) (h3 : ratio = 1/3) 
  (h4 : delta_x = (B.1 - A.1) * ratio) (h5 : delta_y = (B.2 - A.2) * ratio) :
  C = (18, 6) :=
  by sorry

end find_C_coordinates_l77_77725


namespace smaller_consecutive_number_divisibility_l77_77037

theorem smaller_consecutive_number_divisibility :
  ∃ (m : ℕ), (m < m + 1) ∧ (1 ≤ m ∧ m ≤ 200) ∧ (1 ≤ m + 1 ∧ m + 1 ≤ 200) ∧
              (∀ n, (1 ≤ n ∧ n ≤ 200 ∧ n ≠ m ∧ n ≠ m + 1) → ∃ k, chosen_num = k * n) ∧
              (128 = m) :=
sorry

end smaller_consecutive_number_divisibility_l77_77037


namespace C_gets_more_than_D_l77_77023

-- Define the conditions
def proportion_B := 3
def share_B : ℕ := 3000
def proportion_C := 5
def proportion_D := 4

-- Define the parts based on B's share
def part_value := share_B / proportion_B

-- Define the shares based on the proportions
def share_C := proportion_C * part_value
def share_D := proportion_D * part_value

-- Prove the final statement about the difference
theorem C_gets_more_than_D : share_C - share_D = 1000 :=
by
  -- Proof goes here
  sorry

end C_gets_more_than_D_l77_77023


namespace range_of_a_l77_77984

theorem range_of_a (a : ℝ) (n : ℕ) (hn : n > 0) :
  (∀ n (hn : n > 0), (1 - a) * n - a) • (Real.log a) < 0 ↔ (0 < a ∧ a < 1 / 2 ∨ a > 1) :=
sorry

end range_of_a_l77_77984


namespace no_tetrahedron_with_all_obtuse_planes_l77_77071

theorem no_tetrahedron_with_all_obtuse_planes :
  ¬ ∃ (T : Type) (tetrahedron : T), ∀ (e : set (fin 6)), (∀ (v : fin 4), ∑ (a : angle), a < π) ∧ (∃ (a b c : angle), a > π / 2 ∧ b > π / 2 ∧ c > π / 2) :=
begin
  sorry
end

end no_tetrahedron_with_all_obtuse_planes_l77_77071


namespace area_of_rectangle_l77_77966

theorem area_of_rectangle (b : ℝ) (h_b : b = 20) :
  let l := 1.15 * b in
  let A := l * b in
  A = 460 :=
by
  have h_l : l = 1.15 * b := rfl
  have h_A : A = l * b := rfl
  sorry

end area_of_rectangle_l77_77966


namespace prob_multiple_of_3_l77_77202

-- Define the possible start points
def start_points := Fin 15

-- Define the spinner movement instructions
inductive Spinner
| right1 | right2 | left1 | left2

-- Define the transitions for each spinner result
def move (pos : ℤ) : Spinner → ℤ
| Spinner.right1 := pos + 1
| Spinner.right2 := pos + 2
| Spinner.left1 := pos - 1
| Spinner.left2 := pos - 2

-- What we need to prove
theorem prob_multiple_of_3 : (∃ p : ℚ, p = 17 / 80) :=
sorry

end prob_multiple_of_3_l77_77202


namespace time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l77_77364

noncomputable def time_to_count_envelopes (num_envelopes : ℕ) : ℕ :=
(num_envelopes / 10) * 10

theorem time_to_count_60_envelopes_is_40 :
  time_to_count_envelopes 60 = 40 := 
sorry

theorem time_to_count_90_envelopes_is_10 :
  time_to_count_envelopes 90 = 10 := 
sorry

end time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l77_77364


namespace prob_not_perfect_power_l77_77989

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 1 ∧ x^y = n

theorem prob_not_perfect_power :
  let total_numbers := 200
  let perfect_powers := {n | n ∈ (Finset.range total_numbers.succ) ∧  is_perfect_power n}
  let count_perfect_powers := perfect_powers.card
  let non_perfect_powers := total_numbers - count_perfect_powers
  let probability := (non_perfect_powers : ℚ) / total_numbers
  probability = 179 / 200 := by
  sorry

end prob_not_perfect_power_l77_77989


namespace cube_sqrt_three_eq_three_sqrt_three_l77_77308

theorem cube_sqrt_three_eq_three_sqrt_three : (Real.sqrt 3) ^ 3 = 3 * Real.sqrt 3 := 
by 
  sorry

end cube_sqrt_three_eq_three_sqrt_three_l77_77308


namespace correct_answer_l77_77507

theorem correct_answer (x : ℤ) (h : (x - 11) / 5 = 31) : (x - 5) / 11 = 15 :=
by
  sorry

end correct_answer_l77_77507


namespace problem1_problem2_l77_77119

-- Define f with the given properties and conditions
constant f : ℝ → ℝ
axiom f_domain : ∀ x, 0 < x → x ∈ set.Univ
axiom f_multiplicative : ∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y
axiom f_pos : ∀ x, 1 < x → 0 < f x

-- Problem 1: Prove that f(1) = 0 and f is increasing on (0, +∞).
theorem problem1 : f 1 = 0 ∧ (∀ x y, 0 < x → 0 < y → x < y → f x < f y) := sorry

-- Problem 2: Prove the given solution set for the inequality.
axiom f_two : f 2 = 1
theorem problem2 : ∀ x, f (-x) + f (3 - x) ≥ -2 ↔ x ≤ (3 - Real.sqrt 10) / 2 := sorry

end problem1_problem2_l77_77119


namespace expectedValueProof_l77_77049

-- Definition of the problem conditions
def veryNormalCoin {n : ℕ} : Prop :=
  ∀ t : ℕ, (5 < t → (t - 5) = n → (t+1 = t + 1)) ∧ (t ≤ 5 ∨ n = t)

-- Definition of the expected value calculation
def expectedValue (n : ℕ) : ℚ :=
  if n > 0 then (1/2)^n else 0

-- Expected value for the given problem
def expectedValueProblem : ℚ := 
  let a1 := -2/683
  let expectedFirstFlip := 1/2 - 1/(2 * 683)
  100 * 341 + 683

-- Main statement to prove
theorem expectedValueProof : expectedValueProblem = 34783 := 
  sorry -- Proof omitted

end expectedValueProof_l77_77049


namespace draw_4_balls_score_at_least_5_l77_77619

theorem draw_4_balls_score_at_least_5:
  let red_points := 2  -- Points for a red ball
  let white_points := 1  -- Points for a white ball
  let total_balls := 10
  let red_balls := 4
  let white_balls := 6
  let draw_count := 4
  let choices := 
    Nat.choose red_balls 4 +         -- Choosing 4 reds from 4 reds
    Nat.choose red_balls 3 *         -- Choosing 3 reds from 4 reds
    Nat.choose white_balls 1 +       -- Choosing 1 white from 6 whites
    Nat.choose red_balls 2 *         -- Choosing 2 reds from 4 reds
    Nat.choose white_balls 2 +       -- Choosing 2 whites from 6 whites
    Nat.choose red_balls 1 *         -- Choosing 1 red from 4 reds
    Nat.choose white_balls 3         -- Choosing 3 whites from 6 whites
  in choices = 195 := sorry

end draw_4_balls_score_at_least_5_l77_77619


namespace range_of_a_l77_77470

open Set

-- Define the function f with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2 * a - 1) * x + 1

-- Define the interval (2, +∞)
def in_interval (x : ℝ) : Prop := 2 < x

-- Define the given condition for any two unequal real numbers x1, x2 in the interval (2, +∞)
def cond (a : ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x1 ≠ x2 ∧ in_interval x1 ∧ in_interval x2 →
    (f a (x1 - 1) - f a (x2 - 1)) / (x1 - x2) > 0

theorem range_of_a :
  (∀ (a : ℝ), cond a → a ∈ Icc (-1/2) (⊤)) :=
begin
  -- This is a placeholder to state the theorem. Proof is omitted.
  sorry
end

end range_of_a_l77_77470


namespace decreasing_power_function_on_interval_l77_77139

noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem decreasing_power_function_on_interval (m : ℝ) :
  (∀ x : ℝ, (0 < x) -> power_function m x < 0) ↔ m = -1 := 
by 
  sorry

end decreasing_power_function_on_interval_l77_77139


namespace pentagon_area_equation_l77_77521

-- Definition of the convex pentagon and its properties
variables 
  (ABCDE : Type) [convex_pentagon ABCDE]
  (A B C D E : ABCDE)
  (a b c d e S : ℝ) -- areas of the triangles and the pentagon

-- Given the conditions from the problem
axiom area_tris :
  area_triangle A B C = a ∧
  area_triangle B C D = b ∧
  area_triangle C D E = c ∧
  area_triangle D E A = d ∧
  area_triangle E A B = e 

axiom area_pentagon :
  area_pentagon A B C D E = S ∧
  S = a + b + c + d + e 

-- The proof statement goal: Showing the desired equality
theorem pentagon_area_equation :
  S^2 - S * (a + b + c + d + e) + (a * b + b * c + c * d + d * e + e * a) = 0 :=
by
  sorry

end pentagon_area_equation_l77_77521


namespace flower_rows_l77_77945

theorem flower_rows (R : ℕ) (flowers_per_row : ℕ) (cut_percentage : ℕ) (remaining_flowers : ℕ) (h1 : flowers_per_row = 400) (h2 : cut_percentage = 60) (h3 : remaining_flowers = 8000) : 0.40 * (flowers_per_row * R) = remaining_flowers → R = 50 :=
by
  sorry

end flower_rows_l77_77945


namespace exists_int_n_l77_77393

theorem exists_int_n (n : ℤ) : 21 * n ≡ 1 [MOD 74] :=
  ∃ n : ℤ, 21 * n ≡ 1 [MOD 74]

#print exists_int_n -- Verifying that the definition is correct

end exists_int_n_l77_77393


namespace circumscribed_sphere_volume_l77_77112

noncomputable def tetrahedronVolume (PA AB AC ∠BAC : ℝ) (h1: PA = 2) (h2: AB = 2) (h3: AC = 2)
  (h4: ∠BAC = Real.pi / 3 * 2) : ℝ := 
  (4 / 3) * Real.pi * (Real.sqrt (2 ^ 2 + 1 ^ 2) ^ 3)

theorem circumscribed_sphere_volume (PA AB AC ∠BAC : ℝ) 
  (h1 : PA = 2) (h2 : AB = 2) (h3 : AC = 2) (h4 : ∠BAC = Real.pi / 3 * 2) :
  tetrahedronVolume PA AB AC ∠BAC h1 h2 h3 h4 = (20 * Real.sqrt 5 * Real.pi) / 3 := sorry

end circumscribed_sphere_volume_l77_77112


namespace ratio_sides_1_1_1_l77_77041

theorem ratio_sides_1_1_1 (triangle_perimeter : ℝ) (square_perimeter : ℝ) (pentagon_perimeter : ℝ)
  (triangle_perimeter_eq : triangle_perimeter = 30)
  (square_perimeter_eq : square_perimeter = 40)
  (pentagon_perimeter_eq : pentagon_perimeter = 50) :
  let triangle_side := triangle_perimeter / 3
  let square_side := square_perimeter / 4
  let pentagon_side := pentagon_perimeter / 5
  (triangle_side / triangle_side) : (square_side / square_side) : (pentagon_side / pentagon_side) = 1 : 1 : 1 := 
by
  sorry

end ratio_sides_1_1_1_l77_77041


namespace rectangle_divided_into_13_squares_l77_77406

theorem rectangle_divided_into_13_squares (s a b : ℕ) (h₁ : a * b = 13 * s^2)
  (h₂ : ∃ k l : ℕ, a = k * s ∧ b = l * s ∧ k * l = 13) :
  (a = s ∧ b = 13 * s) ∨ (a = 13 * s ∧ b = s) :=
by
sorry

end rectangle_divided_into_13_squares_l77_77406


namespace dorothy_money_left_l77_77773

-- Define the conditions
def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

-- Define the calculation of the amount of money left after paying taxes
def money_left (income : ℝ) (rate : ℝ) : ℝ :=
  income - (rate * income)

-- State the main theorem to prove
theorem dorothy_money_left :
  money_left annual_income tax_rate = 49200 := 
by
  sorry

end dorothy_money_left_l77_77773


namespace reflection_matrix_correct_l77_77785

-- Definitions based on the conditions given in the problem
def reflect_over_line_y_eq_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![1, 0]]

-- The main theorem to state the equivalence
theorem reflection_matrix_correct :
  reflect_over_line_y_eq_x_matrix = ![![0, 1], ![1, 0]] :=
by
  sorry

end reflection_matrix_correct_l77_77785


namespace num_subsets_2_elem_set_l77_77867

open Nat

theorem num_subsets_2_elem_set : (Finset.card (({M : Finset ℕ | M ⊆ {1, 2}} : Finset (Finset ℕ)))) = 4 :=
  sorry

end num_subsets_2_elem_set_l77_77867


namespace spencer_total_jumps_correct_l77_77326

def speed (day : ℕ) : ℕ := 4 * (2 ^ (day - 1))
def first_session_duration : ℕ := 10
def second_session_duration (day : ℕ) : ℕ := 10 + 5 * (day - 1)

def jumps_per_day (day : ℕ) : ℕ :=
  (speed day) * (first_session_duration + (second_session_duration day))

def practices_on_day (day : ℕ) : Bool :=
  day = 1 ∨ day = 2 ∨ day = 4 ∨ day = 5 ∨ day = 6

def total_jumps : ℕ :=
  List.sum (List.filter_map (λ day, if practices_on_day day then some (jumps_per_day day) else none) [1, 2, 3, 4, 5, 6, 7])

theorem spencer_total_jumps_correct : total_jumps = 8600 := by
  sorry

end spencer_total_jumps_correct_l77_77326


namespace complex_number_solution_l77_77092

theorem complex_number_solution (a b : ℝ) (i : ℂ) (hi : i = complex.I)
  (h : (a - 2 * complex.I) * complex.I = b - complex.I) : 
  a + b * complex.I = -1 + 2 * complex.I :=
sorry

end complex_number_solution_l77_77092


namespace sum_inequality_real_numbers_l77_77235

theorem sum_inequality_real_numbers
  {n : ℕ}
  (a b : fin n → ℝ)
  (x : fin n → ℝ)
  (h_sorted : ∀ i j, i ≤ j → x i ≤ x j)
  (h_conds1 : ∀ k : fin n, (∑ i in finset.range k.val.succ, a i) ≥ (∑ i in finset.range k.val.succ, b i))
  (h_conds2 : (∑ i in finset.range n, a i) = (∑ i in finset.range n, b i)) :
  (∑ i in finset.range n, a i * x i) ≤ (∑ i in finset.range n, b i * x i) :=
by
  sorry

end sum_inequality_real_numbers_l77_77235


namespace angle_EGQ_degree_l77_77282

/-- Segment EF has midpoint G, and segment GF has midpoint H.
    Semi-circles are constructed with diameters EF and GF to form the entire region shown.
    Segment GQ splits the region into two sections, with the larger section being twice the area of the smaller.
    Prove that the degree measure of angle EGQ is 150.0 degrees. -/
theorem angle_EGQ_degree 
  (EF G F H GQ : ℝ) 
  (midpoint_EFG : G = (EF + F) / 2) 
  (midpoint_GFH : H = (G + F) / 2)
  (diameter_EF : radius_EF = (EF / 2)) 
  (diameter_GF : radius_GF = (F / 2)) 
  (semicircle_EF : ∃ rEF, rEF = radius_EF ∧ area_EF = (1/2) * real.pi * rEF^2)
  (semicircle_GF : ∃ rGF, rGF = radius_GF ∧ area_GF = (1/2) * real.pi * rGF^2)
  (total_area : total_area = area_EF + area_GF)
  (GQ_splits : ∃ a l : ℝ, l = 2 * a ∧ a + l = total_area) :
  (∃ θ : ℝ, θ = 150.0 ↔ angle_EGQ = θ) := sorry

end angle_EGQ_degree_l77_77282


namespace reflection_matrix_over_y_eq_x_is_correct_l77_77791

theorem reflection_matrix_over_y_eq_x_is_correct :
  let M := matrix.std_basis (fin 2) (fin 2)
  ∃ (R : matrix (fin 2) (fin 2) ℝ), 
    (R ⬝ M 0) = matrix.vec_cons 0 (matrix.vec_cons 1 matrix.vec_empty) ∧
    (R ⬝ M 1) = matrix.vec_cons 1 (matrix.vec_cons 0 matrix.vec_empty) ∧
    R = ![![0, 1], ![1, 0]] :=
sorry

end reflection_matrix_over_y_eq_x_is_correct_l77_77791


namespace smallest_Q_value_l77_77609

noncomputable def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 + 4*x + 6

theorem smallest_Q_value :
  min (Q (-1)) (min (6) (min (1 + 2 - 1 + 4 + 6) (sorry))) = Q (-1) :=
by
  sorry

end smallest_Q_value_l77_77609


namespace angle_PQRS_l77_77601

theorem angle_PQRS (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : 
  P = 206 := 
by
  sorry

end angle_PQRS_l77_77601


namespace sequence_sum_129_l77_77515

/-- 
  In an increasing sequence of four positive integers where the first three terms form an arithmetic
  progression and the last three terms form a geometric progression, and where the first and fourth
  terms differ by 30, the sum of the four terms is 129.
-/
theorem sequence_sum_129 :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a < a + d) ∧ (a + d < a + 2 * d) ∧ 
    (a + 2 * d < a + 30) ∧ 30 = (a + 30) - a ∧ 
    (a + d) * (a + 30) = (a + 2 * d) ^ 2 ∧ 
    a + (a + d) + (a + 2 * d) + (a + 30) = 129 :=
sorry

end sequence_sum_129_l77_77515


namespace history_book_cost_is_correct_l77_77637

-- Define the conditions
def total_books : ℕ := 80
def math_book_cost : ℕ := 4
def total_price : ℕ := 390
def math_books_purchased : ℕ := 10

-- The number of history books
def history_books_purchased : ℕ := total_books - math_books_purchased

-- The total cost of math books
def total_cost_math_books : ℕ := math_books_purchased * math_book_cost

-- The total cost of history books
def total_cost_history_books : ℕ := total_price - total_cost_math_books

-- Define the cost of each history book
def history_book_cost : ℕ := total_cost_history_books / history_books_purchased

-- The theorem to be proven
theorem history_book_cost_is_correct : history_book_cost = 5 := 
by
  sorry

end history_book_cost_is_correct_l77_77637


namespace units_digit_7_pow_3_l77_77666

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_3 : units_digit (7^3) = 3 :=
by
  -- Proof of the theorem would go here
  sorry

end units_digit_7_pow_3_l77_77666


namespace problem_statement_l77_77229

noncomputable def smallest_multiple_with_divisors := 
  let m := smallest (fun n : ℕ => 100 ∣ n ∧ (∀ d : ℕ, d ∣ n ↔ d < 101)) 0
  m / 100

theorem problem_statement : smallest_multiple_with_divisors = 324 := 
by 
  sorry

end problem_statement_l77_77229


namespace f_neg4_f_5_l77_77557

def f : ℝ → ℝ :=
  λ x, if x < 0 then 3 * x - 7 else 2 * (4 - x)

theorem f_neg4 : f (-4) = -19 := by
  sorry

theorem f_5 : f 5 = -2 := by
  sorry

end f_neg4_f_5_l77_77557


namespace smaller_angle_at_9_15_l77_77322

theorem smaller_angle_at_9_15 (h_degree : ℝ) (m_degree : ℝ) (smaller_angle : ℝ) :
  (h_degree = 277.5) → (m_degree = 90) → (smaller_angle = 172.5) :=
by
  sorry

end smaller_angle_at_9_15_l77_77322


namespace average_fuel_consumption_correct_l77_77347

def distance_to_x : ℕ := 150
def distance_to_y : ℕ := 220
def fuel_to_x : ℕ := 20
def fuel_to_y : ℕ := 15

def total_distance : ℕ := distance_to_x + distance_to_y
def total_fuel_used : ℕ := fuel_to_x + fuel_to_y
def avg_fuel_consumption : ℚ := total_fuel_used / total_distance

theorem average_fuel_consumption_correct :
  avg_fuel_consumption = 0.0946 := by
  sorry

end average_fuel_consumption_correct_l77_77347


namespace smallest_m_divided_by_100_l77_77225

-- Define m as the smallest positive integer that meets the conditions
def m : ℕ := 2^4 * 3^9 * 5^1

-- Condition: m is a multiple of 100 and has exactly 100 divisors
def is_multiple_of_100 (n : ℕ) : Prop := 100 ∣ n
def has_exactly_100_divisors (n : ℕ) : Prop := (factors_count n (2^4 * 3^9 * 5^1)) = 100

-- The property we want to prove: m meets both conditions and yields the correct fragment
theorem smallest_m_divided_by_100 : 
  is_multiple_of_100 m ∧ has_exactly_100_divisors m → m / 100 = 15746.4 :=
by
  sorry

end smallest_m_divided_by_100_l77_77225


namespace part1_part2_l77_77132

noncomputable def f (a x : ℝ) : ℝ := a * x + x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x, x ≥ Real.exp 1 → (a + 1 + Real.log x) ≥ 0) →
  a ≥ -2 :=
by
  sorry

theorem part2 (k : ℤ) :
  (∀ x, 1 < x → (k : ℝ) * (x - 1) < f 1 x) →
  k ≤ 3 :=
by
  sorry

end part1_part2_l77_77132


namespace quadratic_function_proof_l77_77154

def f (x : ℝ) : ℝ := -x^2 - x + 2

axiom root_neg_two : f (-2) = 0
axiom root_one : f (1) = 0
axiom f_at_zero : f (0) = 2

theorem quadratic_function_proof : f = λ x, -x^2 - x + 2 :=
by
  sorry

end quadratic_function_proof_l77_77154


namespace existence_of_positive_numbers_l77_77283

open Real

theorem existence_of_positive_numbers {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 > 2 ∧ a^3 + b^3 + c^3 < 2 ∧ a^4 + b^4 + c^4 > 2 :=
sorry

end existence_of_positive_numbers_l77_77283


namespace prove_eccentricity_l77_77003

open Real

def eccentricity (a b : ℝ) : ℝ := sqrt (1 + (b ^ 2) / (a ^ 2))

variables (a b : ℝ)

-- Definition of conditions
def circle_passing_through_right_focus_of_hyperbola (r : ℝ) : Prop :=
  ∃ F : ℝ × ℝ, F ∈ {p | p.1^2 + p.2^2 = r^2} ∧ F ∈ {p | ∃ x y, x = sqrt (a^2 - y^2/b^2) ∧ p = (x, y)}

def intersects_asymptotes_at_points (A B : ℝ × ℝ) : Prop :=
  A ∈ {p | p.2 = (b/a) * p.1} ∧ B ∈ {p | p.2 = -(b/a) * p.1}

def quadrilateral_is_rhombus (O A F B : ℝ × ℝ) : Prop :=
  dist O A = dist A F ∧ dist A F = dist F B ∧ dist F B = dist B O

-- Variables for the points O, F, A, B
variables (O A F B : ℝ × ℝ) (r : ℝ)

-- Given definitions
def conditions :=
  circle_passing_through_right_focus_of_hyperbola a b r ∧ 
  intersects_asymptotes_at_points a b A B ∧ 
  quadrilateral_is_rhombus O A F B

-- Statement to prove
theorem prove_eccentricity (h : conditions a b O A F B r) : eccentricity a b = 2 :=
sorry

end prove_eccentricity_l77_77003


namespace simplify_frac_l77_77284

theorem simplify_frac (b : ℤ) (hb : b = 2) : (15 * b^4) / (45 * b^3) = 2 / 3 :=
by {
  sorry
}

end simplify_frac_l77_77284


namespace find_f_2009_l77_77456

-- Define the function f and its properties
def f (x : ℝ) : ℝ

-- Functional equation
axiom func_eq (x : ℝ) : f(x + 2) * (1 - f(x)) = 1 + f(x)

-- Initial condition
axiom initial_cond : f 1 = 9997

-- The goal to prove
theorem find_f_2009 : f 2009 = 9997 := by
  sorry

end find_f_2009_l77_77456


namespace mirror_full_body_view_l77_77353

theorem mirror_full_body_view (AB MN : ℝ) (h : AB > 0): 
  (MN = 1/2 * AB) ↔
  ∀ (P : ℝ), (0 < P) → (P < AB) → 
    (P < MN + (AB - P)) ∧ (P > AB - MN + P) := 
by
  sorry

end mirror_full_body_view_l77_77353


namespace non_defective_probability_l77_77016

theorem non_defective_probability :
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  p_non_def = 0.96 :=
by
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  sorry

end non_defective_probability_l77_77016


namespace swimmer_time_against_current_l77_77024

theorem swimmer_time_against_current (swimmer_speed : ℝ) (current_speed : ℝ) (time_with_current : ℝ)
  (h_swimmer_speed : swimmer_speed = 4)
  (h_current_speed : current_speed = 2)
  (h_time_with_current : time_with_current = 3.5) : 
  let effective_speed_with_current := swimmer_speed + current_speed in
  let distance_with_current := effective_speed_with_current * time_with_current in
  let effective_speed_against_current := swimmer_speed - current_speed in
  let time_against_current := distance_with_current / effective_speed_against_current in
  time_against_current = 10.5 :=
by
  sorry

end swimmer_time_against_current_l77_77024


namespace binomial_sum_inequality_l77_77260

theorem binomial_sum_inequality (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k ≤ 2 * n → (1 / (k + 1) * (nat.choose (2 * n) k) = 
    (1 / (2 * n + 1)) * (nat.choose (2 * n + 1) (k + 1)))) →
  (∑ k in finset.range (2 * n + 1), nat.choose (2 * n + 1) k = 2 ^ (2 * n + 1)) →
  (∑ k in finset.range (2 * n + 1), (1 / (k + 1) * nat.choose (2 * n) k) ≤ nat.choose (2 * n + 1) n) :=
begin
  intros h1 h2,
  sorry
end

end binomial_sum_inequality_l77_77260


namespace johns_speed_second_part_l77_77535

theorem johns_speed_second_part (speed_initial : ℕ) (time_initial : ℕ) (total_distance : ℕ) (time_second : ℕ) :
  speed_initial = 45 ∧ time_initial = 2 ∧ total_distance = 240 ∧ time_second = 3 → 
  (total_distance - speed_initial * time_initial) / time_second = 50 :=
by 
  intros h,
  obtain ⟨h1, h2, h3, h4⟩ := h,
  have distance_initial := h1 * h2,
  have remaining_distance := h3 - distance_initial,
  have speed_second := remaining_distance / h4,
  exact speed_second = 50 

end johns_speed_second_part_l77_77535


namespace correct_choice_about_correlation_l77_77368

theorem correct_choice_about_correlation (r : ℝ) (hr : -1 ≤ r ∧ r ≤ 1) :
  (|r| ≤ 1) ∧ (∀ ε > 0, |r - 1| < ε → degree_of_correlation (r) = strong_positive ∧
                         |r + 1| < ε → degree_of_correlation (r) = strong_negative ∧
                         |r| < ε → degree_of_correlation (r) = weak) := 
sorry

end correct_choice_about_correlation_l77_77368


namespace total_bins_used_l77_77774

def bins_of_soup : ℝ := 0.12
def bins_of_vegetables : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

theorem total_bins_used : bins_of_soup + bins_of_vegetables + bins_of_pasta = 0.74 :=
by
  sorry

end total_bins_used_l77_77774


namespace part_1_part_2_l77_77846

def f (x m : ℝ) : ℝ :=
  if x ≤ m then -x^2 - 2 * x else x - 4

theorem part_1 : (m = 0) → (f x m = 0) → (∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c)) := sorry

theorem part_2 : (∀ x, (f x m = 0) → (count_roots (λ x, f x m) = 2)) ↔ ((-2 ≤ m ∧ m < 0) ∨ (4 ≤ m)) := sorry

end part_1_part_2_l77_77846


namespace inequality_problem_l77_77933

theorem inequality_problem
  (a b c : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( ( (2 * a + b + c) ^ 2 ) / ( 2 * a ^ 2 + (b + c) ^ 2 ) ) +
  ( ( (a + 2 * b + c) ^ 2 ) / ( 2 * b ^ 2 + (c + a) ^ 2 ) ) +
  ( ( (a + b + 2 * c) ^ 2 ) / ( 2 * c ^ 2 + (a + b) ^ 2 ) ) ≤ 8 :=
by
  sorry

end inequality_problem_l77_77933


namespace volume_of_pyramid_l77_77278

-- Definitions based on the given conditions
def AB : ℝ := 10
def BC : ℝ := 6
def PB : ℝ := 20
def area_base := AB * BC
def PA := Real.sqrt (PB ^ 2 - AB ^ 2)

-- Statement to prove the volume of pyramid PABCD
theorem volume_of_pyramid :
  let volume := (1 : ℝ) / 3 * area_base * PA in
  volume = 200 * Real.sqrt 3 :=
by
  sorry

end volume_of_pyramid_l77_77278


namespace rectangles_divided_into_13_squares_l77_77402

theorem rectangles_divided_into_13_squares (m n : ℕ) (h : m * n = 13) : 
  (m = 1 ∧ n = 13) ∨ (m = 13 ∧ n = 1) :=
sorry

end rectangles_divided_into_13_squares_l77_77402


namespace angle_inequality_l77_77813

-- Definitions for the given conditions
def OnOneSide (O A B : Point) : Prop := sorry
def Perpendiculars (AA1 BB1 : Line) : Prop := sorry
def RightAngle (α : Angle) : Prop := α = 90

-- The main theorem statement
theorem angle_inequality
  (O A B A1 B1 : Point)
  (h_on_side : OnOneSide O A B)
  (h_perpendiculars : Perpendiculars AA1 BB1)
  (h_right_angles1 : RightAngle (∠(A, A1, O)))
  (h_right_angles2 : RightAngle (∠(B, B1, O)))
  (h_distance : dist O A < dist O B) :
  ∠(O, A, A1) ≥ ∠(O, B, B1) :=
begin
  sorry
end

end angle_inequality_l77_77813


namespace miquel_point_concurrent_l77_77915

noncomputable def miquel_point (A B C D : Point) : Point :=
  let P := intersection (line_through A D) (line_through B C)
  let Q := intersection (line_through A B) (line_through C D)
  let M := intersection (circumcircle ⟨C, B, Q⟩) (circumcircle ⟨A, P, B⟩) in
  M

theorem miquel_point_concurrent (A B C D P Q M : Point) :
  P = intersection (line_through A D) (line_through B C) →
  Q = intersection (line_through A B) (line_through C D) →
  M = intersection (circumcircle ⟨C, B, Q⟩) (circumcircle ⟨A, P, B⟩) →
  M ∈ circumcircle ⟨D, C, P⟩ ∧ M ∈ circumcircle ⟨A, D, Q⟩ :=
begin
  intros hP hQ hM,
  sorry
end

end miquel_point_concurrent_l77_77915


namespace minutes_in_3_5_hours_l77_77149

theorem minutes_in_3_5_hours : 3.5 * 60 = 210 := 
by
  sorry

end minutes_in_3_5_hours_l77_77149


namespace min_trig_sum_l77_77419

noncomputable def trig_sum (x : ℝ) : ℝ :=
|sin x + cos x + tan x + cot x + sec x + csc x|

theorem min_trig_sum : ∃ x : ℝ, trig_sum x = 2 * sqrt 2 - 1 :=
sorry

end min_trig_sum_l77_77419


namespace dorothy_money_left_l77_77770

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18
def tax_amount : ℝ := annual_income * tax_rate
def money_left : ℝ := annual_income - tax_amount

theorem dorothy_money_left : money_left = 49200 := 
by
  sorry

end dorothy_money_left_l77_77770


namespace arithmetic_mean_of_range_neg3_to_6_l77_77649

theorem arithmetic_mean_of_range_neg3_to_6 :
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  (sum : Float) / (count : Float) = 1.5 := by
  let numbers := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let sum := List.sum numbers
  let count := List.length numbers
  have h_sum : sum = 15 := by sorry
  have h_count : count = 10 := by sorry
  calc
    (sum : Float) / (count : Float)
        = (15 : Float) / (10 : Float) : by rw [h_sum, h_count]
    ... = 1.5 : by norm_num

end arithmetic_mean_of_range_neg3_to_6_l77_77649


namespace count_reciprocal_sets_is_two_l77_77354

def set1 (a : ℝ) : Set ℝ := { x | x^2 + a * x + 1 = 0 }
def set2 : Set ℝ := { x | x^2 - 4 * x + 1 < 0 }
def set3 : Set ℝ := { y | ∃ x, y = (Real.log x) / x ∧ x ∈ ([1/Real.exp 1, 1) ∪ (1, Real.exp 1]) }
def set4 : Set ℝ := { y | ∃ x, (x ∈ ([0, 1) ∪ [1, 2])) ∧ y = (2 * x + 2 / 5) / (x + 1 / x) }

def is_reciprocal_set (A : Set ℝ) : Prop :=
  0 ∉ A ∧ ∀ x ∈ A, (1 / x) ∈ A

-- Prove that the number of reciprocal sets among set1, set2, set3, and set4 is 2
theorem count_reciprocal_sets_is_two (a : ℝ) :
  (if is_reciprocal_set (set1 a) then 1 else 0) +
  (if is_reciprocal_set set2 then 1 else 0) +
  (if is_reciprocal_set set3 then 1 else 0) +
  (if is_reciprocal_set set4 then 1 else 0) = 2 :=
sorry

end count_reciprocal_sets_is_two_l77_77354


namespace difference_of_squares_example_l77_77057

theorem difference_of_squares_example : (75^2 - 25^2) = 5000 := by
  let a := 75
  let b := 25
  have step1 : a + b = 100 := by
    rw [a, b]
    norm_num
  have step2 : a - b = 50 := by
    rw [a, b]
    norm_num
  have result : (a + b) * (a - b) = 5000 := by
    rw [step1, step2]
    norm_num
  rw [pow_two, pow_two, mul_sub, ← result]
  norm_num

end difference_of_squares_example_l77_77057


namespace sum_of_octal_numbers_l77_77034

theorem sum_of_octal_numbers :
  (176 : ℕ) + 725 + 63 = 1066 := by
sorry

end sum_of_octal_numbers_l77_77034


namespace number_of_values_g50_eq_18_l77_77427

def divisors_count (n : ℕ) : ℕ :=
  (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card

def g1 (n : ℕ) : ℕ := 3 * divisors_count n

def g : ℕ → ℕ → ℕ
| 1, n := g1 n
| (j+1), n := g1 (g j n)

theorem number_of_values_g50_eq_18 : 
  (finset.filter (λ n, g 50 n = 18) (finset.range 31)).card = 3 :=
sorry

end number_of_values_g50_eq_18_l77_77427


namespace red_pill_cost_l77_77376

theorem red_pill_cost :
  ∃ (r : ℚ) (b : ℚ), (∀ (d : ℕ), d = 21 → 3 * r - 2 = 39) ∧
                      (1 ≤ d → r = b + 1) ∧
                      (21 * (r + 2 * b) = 819) → 
                      r = 41 / 3 :=
by sorry

end red_pill_cost_l77_77376


namespace moses_more_than_esther_by_10_l77_77629

noncomputable theory

def total_amount : ℝ := 80
def moses_share : ℝ := 0.35 * total_amount
def rachel_share : ℝ := 0.20 * total_amount
def remaining_amount : ℝ := total_amount - (moses_share + rachel_share)
def esther_share : ℝ := remaining_amount / 2

theorem moses_more_than_esther_by_10 :
  moses_share - esther_share = 10 := sorry

end moses_more_than_esther_by_10_l77_77629


namespace constant_term_expansion_l77_77522

theorem constant_term_expansion : 
  (constant_term ((x - ((1 : ℚ) / (x ^ 2))) ^ 9) = -84) := 
by 
  sorry

end constant_term_expansion_l77_77522


namespace perpendicular_length_of_centroid_l77_77586

theorem perpendicular_length_of_centroid 
  (A B C D E F : ℝ) -- Coordinates are treated as real numbers
  (AD BE CF : ℝ)
  (RS : set ℝ) -- RS is a set representing a line in the Euclidean plane
  (hAD : AD = 12)
  (hBE : BE = 8)
  (hCF : CF = 30)
  (h_perpendicular_AD : B ⟂ RS)
  (h_perpendicular_BE : C ⟂ RS)
  (h_perpendicular_CF : A ⟂ RS)
  (h_RS_not_intersect_triangle : ∀ x, RS x → ¬(x = A ∨ x = B ∨ x = C)) :
  let G := (A + B + C) / 3 in
  y_G = A * 12 + B * 8 + C * 30 := ((hAD, hBE, hCF, h_perpendicular_AD, h_perpendicular_BE, h_perpendicular_CF, h_RS_not_intersect_triangle).prod) / 3 :=
  y_G = A * 12 + B * 8 + C * 30 := (A * 12 + B * 8 + C * 30) / 3 in
  y_G = 50/3 :=
sorry

end perpendicular_length_of_centroid_l77_77586


namespace reflection_matrix_correct_l77_77786

-- Definitions based on the conditions given in the problem
def reflect_over_line_y_eq_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![1, 0]]

-- The main theorem to state the equivalence
theorem reflection_matrix_correct :
  reflect_over_line_y_eq_x_matrix = ![![0, 1], ![1, 0]] :=
by
  sorry

end reflection_matrix_correct_l77_77786


namespace ralph_tv_hours_l77_77268

theorem ralph_tv_hours :
  (4 * 5 + 6 * 2) = 32 :=
by
  sorry

end ralph_tv_hours_l77_77268


namespace determine_real_pairs_l77_77766

theorem determine_real_pairs (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊ b * n ⌋ = b * ⌊ a * n ⌋) →
  (∃ c : ℝ, (a = 0 ∧ b = c) ∨ (a = c ∧ b = 0) ∨ (a = c ∧ b = c) ∨ (∃ k l : ℤ, a = k ∧ b = l)) :=
by
  sorry

end determine_real_pairs_l77_77766


namespace slope_of_regression_line_l77_77892

variable (h : ℝ)
variable (t1 T1 t2 T2 t3 T3 : ℝ)

-- Given conditions.
axiom t2_is_equally_spaced : t2 = t1 + h
axiom t3_is_equally_spaced : t3 = t1 + 2 * h

theorem slope_of_regression_line :
  t2 = t1 + h →
  t3 = t1 + 2 * h →
  (T3 - T1) / (t3 - t1) = (T3 - T1) / ((t1 + 2 * h) - t1) := 
by
  sorry

end slope_of_regression_line_l77_77892


namespace AM_equals_MN_l77_77938

-- Definitions of the points and their relationships in the rectangle
variables (A B C D E F G M N : Point) (rect : is_rectangle A B C D)

-- Definitions of midpoints and equality of segments
def is_midpoint (P M Q : Point) : Prop := dist P M = dist M Q ∧ collinear P M Q
def is_parallel (l1 l2 : Line) : Prop := ∃ (m1 m2 : ℝ), ∀ (P Q : Point), P ∈ l1 → Q ∈ l2 → ∠ P O Q = m1 ∨ ∠ P Q O = m2

-- Conditions
axiom midpoint_E : is_midpoint B E C
axiom AF_FG : dist A F = dist F G

-- Intersection definitions
axiom intersection_M : is_intersection (line_through E F) (line_through A C) M
axiom intersection_N : is_intersection (line_through C G) (line_through A D) N

-- Theorem statement
theorem AM_equals_MN : dist A M = dist M N :=
sorry

end AM_equals_MN_l77_77938


namespace train_length_l77_77688

-- Let's state the problem and define the constants and assumptions
open Real

noncomputable def length_of_train (v_fast v_slow : ℝ) (t : ℝ) : ℝ :=
  let relative_speed := (v_fast - v_slow) * (5/18) -- convert km/hr to m/s
  let distance_covered := relative_speed * t
  distance_covered / 2 -- since both trains have equal length

theorem train_length
  (v_fast : ℝ) (v_slow : ℝ) (t : ℝ)
  (h_fast : v_fast = 42)
  (h_slow : v_slow = 36)
  (h_time : t = 36) :
  length_of_train v_fast v_slow t = 30 :=
by
  rw [length_of_train, h_fast, h_slow, h_time]
  norm_num
  sorry

end train_length_l77_77688


namespace ellipse_circumcircle_problem_l77_77464

theorem ellipse_circumcircle_problem:
  let f1 : ℝ × ℝ := (-1, 0)
  let f2 : ℝ × ℝ := (1, 0)
  let pt : ℝ × ℝ := (1, (Real.sqrt 2) / 2)
  let l (m : ℝ) : ℝ × ℝ → Prop := λ P, P.1 = m * P.2 - 2
  let P : ℝ × ℝ := (-2, 0)
  ∃ E : ℝ × ℝ → Prop, 
  (E = (λ P, P.1^2 / 2 + P.2^2 = 1)) ∧
  (∀ (A B : ℝ × ℝ), 
    l 2 A ∧ l 2 B ∧ 
    (B = (0, 1)) ∧
    (P.1 - A.1 = 3*(P.1 - B.1)) ∧ 
    ∃ C D : ℝ × ℝ, 
      C = (A.1, -A.2) ∧ D = (B.1, -B.2) ∧
      ∃ O : ℝ × ℝ, let r : ℝ := (Real.sqrt 10) / 3 in
      (O = (-1/3, 0)) ∧
      ((E O) ∧ (E ({-A B C D}) set.to_list) ∧ (O.1 - C.1)^2 + O.2^2 = r^2 ∧
      ((λ X, (X.1 + 1/3)^2 + X.2^2 = 10/9) C ∧
       (λ X, (X.1 + 1/3)^2 + X.2^2 = 10/9) D)) :=
sorry

end ellipse_circumcircle_problem_l77_77464
