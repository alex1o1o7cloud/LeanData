import Mathlib

namespace symmetric_point_on_altitude_l422_422540

-- Definitions for geometric entities and properties

-- Assume A, B, C are points representing the vertices of ΔABC
variable (A B C : Point)

-- Let O be the center of the circumcircle of ΔABC
variable (O : Point)

-- Let D be the midpoint of side BC
variable (D : Point)

-- Let H be the orthocenter of ΔABC
variable (H : Point)

-- Let L be the midpoint of segment AH
variable (L : Point)

-- The theorem stating the required property
theorem symmetric_point_on_altitude :
  -- Statement of the theorem: Proving the point symmetric to O with respect to the midpoints of the medians lies on the altitudes
  ∀ (A B C O D H L : Point), 
    circumcenter O A B C ∧ 
    midpoint D B C ∧ 
    orthocenter H A B C ∧ 
    midpoint L A H → 
    symmetric_point O midpoint_of_medians(A, B, C) lies_on altitudes(A, B, C) := by -- Sorry to skip proof
  sorry

end symmetric_point_on_altitude_l422_422540


namespace base_6_conversion_l422_422185

-- Define the conditions given in the problem
def base_6_to_10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

-- given that 524_6 = 2cd_10 and c, d are base-10 digits, prove that (c * d) / 12 = 3/4
theorem base_6_conversion (c d : ℕ) (h1 : base_6_to_10 5 2 4 = 196) (h2 : 2 * 10 * c + d = 196) :
  (c * d) / 12 = 3 / 4 :=
sorry

end base_6_conversion_l422_422185


namespace general_laborer_daily_wage_l422_422700

theorem general_laborer_daily_wage :
  ∀ (L : ℕ), 
  (∀ (h_operators_count : ℕ), h_operators_count = 30 → 
  (∀ (h_operator_wage : ℕ), h_operator_wage = 129 → 
  (∀ (total_people : ℕ), total_people = 31 →
  (∀ (total_payroll : ℕ), total_payroll = 3952 →
  (∀ (laborer_count : ℕ), laborer_count = 1 →
  (laborer_count + h_operators_count = total_people ∧
   h_operators_count * h_operator_wage + L = total_payroll → L = 82))))))

end general_laborer_daily_wage_l422_422700


namespace lcm_18_20_l422_422736

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_18_20_l422_422736


namespace complex_number_in_first_quadrant_l422_422948

def is_in_first_quadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

theorem complex_number_in_first_quadrant (z : ℂ) (h : 0 < z.re ∧ 0 < z.im) : is_in_first_quadrant z :=
by sorry

end complex_number_in_first_quadrant_l422_422948


namespace parabola_triangle_area_l422_422063

def parabola_p_q_area : Prop :=
  let F := (0 : ℝ, 2 : ℝ)
  let P₀ := (4 * Real.sqrt 2, 4) in -- we choose one point for simplicity
  let P₁ := (-4 * Real.sqrt 2, 4) in
  let Q := (0 : ℝ, -2 : ℝ) in
  let PF₀ := Real.sqrt ((4 * Real.sqrt 2 - 0)^2 + (4 - 2)^2) = 6 in
  let PF₁ := Real.sqrt ((-4 * Real.sqrt 2 - 0)^2 + (4 - 2)^2) = 6 in
  let FQ := Real.sqrt ((0 - 0)^2 + (2 + 2)^2) = 4 in
  let area₀ := (1 / 2) * (abs (0 - 4 * Real.sqrt 2) * 4) in
  let area₁ := (1 / 2) * (abs (0 + 4 * Real.sqrt 2) * 4) in
  area₀ = 8 * Real.sqrt 2 ∧ area₁ = 8 * Real.sqrt 2

theorem parabola_triangle_area : parabola_p_q_area := 
  sorry

end parabola_triangle_area_l422_422063


namespace Aline_wish_fulfilled_iff_even_l422_422516

theorem Aline_wish_fulfilled_iff_even (n : ℕ) (h_positive : n > 0) :
  (∃ (x : Fin 2n → ℝ), (¬ ∀ i j, i ≠ j → x i = x j) ∧
    (∀ (S : Finset (Fin 2n)), S.card = n → S.sum x = (Finset.univ \ S).prod x)) ↔
  Even n :=
by
  sorry

end Aline_wish_fulfilled_iff_even_l422_422516


namespace non_intersecting_pairs_l422_422124

theorem non_intersecting_pairs (n : ℕ) (points : fin (2 * n) → ℝ × ℝ) : 
  ∃ (pairs : list (ℕ × ℕ)), 
    list.length pairs = n ∧ 
    (∀ (i j : ℕ) (h_i_in : i ∈ (pairs.map prod.fst)) (h_j_in : j ∈ (pairs.map prod.snd)), 
      ¬ ∃ (k l : ℕ), 
        k ≠ l ∧ 
        (k, l) ∈ pairs ∧ 
        (dist (points k) (points l) = dist (points (pairs.nth_le i h_i_in).fst) (points (pairs.nth_le i h_i_in).snd) + 
         dist (points (pairs.nth_le j h_j_in).fst) (points (pairs.nth_le j h_j_in).snd))) :=
begin
  sorry
end

end non_intersecting_pairs_l422_422124


namespace total_initial_amounts_l422_422584

theorem total_initial_amounts :
  ∃ (a j t : ℝ), a = 50 ∧ t = 50 ∧ (50 + j + 50 = 187.5) :=
sorry

end total_initial_amounts_l422_422584


namespace arithmetic_sequence_property_l422_422487

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_property
  (h1 : a 3 + 3 * a 8 + a 13 = 120)
  (h2 : a 3 + a 13 = 2 * a 8) :
  a 3 + a 13 - a 8 = 24 := by
  sorry

end arithmetic_sequence_property_l422_422487


namespace minimum_distance_l422_422125

def parametric_line (t : ℝ) : ℝ × ℝ := (1 + t, 7 + t)

def polar_curve (θ : ℝ) : ℝ := -2 * Real.cos θ + 2 * Real.sin θ

theorem minimum_distance :
  ∃ d : ℝ, (∀ p ∈ set.range (λ θ, ⟨polar_curve θ * Real.cos θ, polar_curve θ * Real.sin θ⟩), ∃ t : ℝ, dist p (parametric_line t) = d)
  ∧
  d = sqrt 2 :=
sorry

end minimum_distance_l422_422125


namespace find_point_C_l422_422035

variable (A B C : ℝ × ℝ × ℝ)
variable (vecA vecB vecC : ℝ × ℝ × ℝ)

-- Definitions of points A and B
def point_A := (3, 5, -7)
def point_B := (-1, 3, 3)

-- Definition of vectors
def vec_AB := (vecB.1 - vecA.1, vecB.2 - vecA.2, vecB.3 - vecA.3)
def vec_CB := (vecB.1 - vecC.1, vecB.2 - vecC.2, vecB.3 - vecC.3)

-- Condition
axiom cond : vec_AB = (2 : ℝ) • vec_CB

-- Target statement
theorem find_point_C (vecA vecB : ℝ × ℝ × ℝ) : 
  let vec_AB := (vecB.1 - vecA.1, vecB.2 - vecA.2, vecB.3 - vecA.3);
  let C : ℝ × ℝ × ℝ := (1, 4, -2);
  ∃ vecC, C = vecC ∧ vec_AB = (2 : ℝ) • (vecB.1 - vecC.1, vecB.2 - vecC.2, vecB.3 - vecC.3) :=
sorry

end find_point_C_l422_422035


namespace wire_needed_in_meters_l422_422088

def width_cm : ℕ := 480
def length_cm : ℕ := 360

def perimeter_cm (width length : ℕ) : ℕ := 2 * (width + length)

def perimeter_meter (cm : ℕ) : ℝ := cm / 100.0

theorem wire_needed_in_meters :
  (perimeter_meter (perimeter_cm width_cm length_cm) = 16.8) :=
by
  sorry

end wire_needed_in_meters_l422_422088


namespace solution_set_for_inequality_l422_422781

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + (a - b) * x + 1

theorem solution_set_for_inequality (a b : ℝ) (h1 : 2*a + 4 = -(a-1)) :
  ∀ x : ℝ, (f x a b > f b a b) ↔ ((x ∈ Set.Icc (-2 : ℝ) (2 : ℝ)) ∧ ((x < -1 ∨ 1 < x))) :=
by
  sorry

end solution_set_for_inequality_l422_422781


namespace general_formula_sum_b_l422_422111

-- Define the arithmetic sequence
def arithmetic_sequence (a d: ℕ) (n: ℕ) := a + (n - 1) * d

-- Given conditions
def a1 : ℕ := 1
def d : ℕ := 2
def a (n : ℕ) : ℕ := arithmetic_sequence a1 d n
def b (n : ℕ) : ℕ := 2 ^ a n

-- Formula for the arithmetic sequence
theorem general_formula (n : ℕ) : a n = 2 * n - 1 := 
by sorry

-- Sum of the first n terms of b_n
theorem sum_b (n : ℕ) : (Finset.range n).sum b = (2 / 3) * (4 ^ n - 1) :=
by sorry

end general_formula_sum_b_l422_422111


namespace triangle_inequalities_l422_422470

theorem triangle_inequalities 
  (a b c : ℝ) (A B C : ℝ) (h_a h_b : ℝ) (m_a m_b : ℝ) 
  (h_a_def : h_a = (2 * Real.sqrt((4 * c^2 - (a^2 - b^2 + c^2)^2)/(4 * c^2))) / a)
  (h_b_def : h_b = (2 * Real.sqrt((4 * a^2 - (b^2 - a^2 + c^2)^2)/(4 * a^2))) / b)
  (m_a_def : m_a = 1 / 2 * Real.sqrt(2 * b^2 + 2 * c^2 - a^2))
  (m_b_def : m_b = 1 / 2 * Real.sqrt(2 * a^2 + 2 * c^2 - b^2))
  (law_of_sines : a / Real.sin A = b / Real.sin B) :
  (a ≥ b ↔ Real.sin A ≥ Real.sin B) ∧ 
  (Real.sin A ≥ Real.sin B ↔ m_a ≤ m_b) ∧ 
  (m_a ≤ m_b ↔ h_a ≤ h_b) :=
begin
  sorry
end

end triangle_inequalities_l422_422470


namespace part_a_l422_422522

noncomputable def f : ℝ⁺ → ℝ⁺ := sorry
noncomputable def g : ℝ⁺ → ℝ⁺ := sorry

axiom functional_equation (x y : ℝ⁺) : f (g(x) * y + f(x)) = (y + 2015) * f(x)

theorem part_a (x : ℝ⁺) : f(x) = 2015 * g(x) := sorry

example : ∃ f g : ℝ⁺ → ℝ⁺, (∀ x y : ℝ⁺, f (g(x) * y + f(x)) = (y + 2015) * f(x)) ∧ (∀ x : ℝ⁺, f(x) ≥ 1 ∧ g(x) ≥ 1) :=
begin
  use [λ x, if x < 1 then 2015 else 2015 * x, λ x, (if x < 1 then 2015 else 2015 * x) / 2015],
  split,
  { intros x y,
    by_cases hx: x < 1,
    { rw if_pos hx,
      rw if_pos hx,
      simp,  -- Demonstrates the functional equation holds in this range (details pending). 
      sorry, -- Placeholder for remaining steps
    },
    { rw if_neg hx,
      rw if_neg hx,
      simp, -- Demonstrates the functional equation holds in this range (details pending).
      sorry, -- Placeholder for remaining steps
    }
  },
  { intro x,
    split;
    { by_cases hx: x < 1;
      { rw if_pos hx, simp, linarith,[using hx];
        { rw if_neg hx, simp, linarith,}
    }
  }
end

end part_a_l422_422522


namespace sweet_potatoes_not_yet_sold_l422_422166

theorem sweet_potatoes_not_yet_sold:
  ∀ (total_harvested total_sold_to_Adams total_sold_to_Lenon : ℕ)
  (h_total_harvested : total_harvested = 80)
  (h_sold_to_Adams : total_sold_to_Adams = 20)
  (h_sold_to_Lenon : total_sold_to_Lenon = 15),
  total_harvested - (total_sold_to_Adams + total_sold_to_Lenon) = 45 :=
by
  intro total_harvested total_sold_to_Adams total_sold_to_Lenon
  intro h_total_harvested h_sold_to_Adams h_sold_to_Lenon
  rw [h_total_harvested, h_sold_to_Adams, h_sold_to_Lenon]
  norm_num
  sorry

end sweet_potatoes_not_yet_sold_l422_422166


namespace determine_younger_years_l422_422156

variables (A B C D : ℕ)

-- Conditions from the problem statement
axiom H1 : A + B = B + C + 11
axiom H2 : A + B + D = B + C + D + 8

-- Required proof statement
theorem determine_younger_years : C - (A + D) = -11 - D :=
sorry

end determine_younger_years_l422_422156


namespace chair_cost_l422_422870

theorem chair_cost :
  (∃ (C : ℝ), 3 * C + 50 + 40 = 130 - 4) → 
  (∃ (C : ℝ), C = 12) :=
by
  sorry

end chair_cost_l422_422870


namespace grasshoppers_cannot_be_in_initial_position_after_1999_seconds_l422_422230

theorem grasshoppers_cannot_be_in_initial_position_after_1999_seconds :
  ∀ A B C : ℕ, 
    let initial_order := (A, B, C)
    (∀ n : ℕ, 
      let orders := ([initial_order, (B, C, A), (C, A, B)] : List (ℕ × ℕ × ℕ)), 
      let odd_orders := [(A, C, B), (B, A, C), (C, B, A)] in
      n.mod 2 = 1 →
      ∀ (A' B' C' : ℕ), 
        (A', B', C') ∈ orders ∨ (A', B', C') ∈ odd_orders →
        initial_order ≠ (A', B', C') →
        (A, B, C) ≠ (A', B', C', n = 1999)
sorry

end grasshoppers_cannot_be_in_initial_position_after_1999_seconds_l422_422230


namespace equality_of_costs_l422_422929

theorem equality_of_costs (x : ℕ) :
  (800 + 30 * x = 500 + 35 * x) ↔ x = 60 := by
  sorry

end equality_of_costs_l422_422929


namespace parabola_axis_of_symmetry_a_eq_1_a_value_range_l422_422492

open Real

-- Definitions and conditions
variable (a b c m n : ℝ)
variable (h_parabola1 : (2 * a + 1, m) ∈ {p : ℝ × ℝ | p.2 = a * p.1^2 - 2 * a^2 * p.1 + c })
variable (h_parabola2 : (b, n) ∈ {p : ℝ × ℝ | p.2 = a * p.1^2 - 2 * a^2 * p.1 + c })
variable (h_a_nonzero : a ≠ 0)
variable (h_c_positive : c > 0)

-- Part 1
theorem parabola_axis_of_symmetry_a_eq_1 (ha1 : a = 1) :
    axis_of_symmetry (parabola (λ x : ℝ, x^2 - 2 * x + c)) = 1 ∧ m > c :=
by
  subst ha1
  sorry

-- Part 2
variable (hb_bounds : 2 ≤ b ∧ b ≤ 4)
variable (h_m_c_n : m > c ∧ c > n)

theorem a_value_range (h_b_range : ∀ b ∈ Icc 2 4, m > c ∧ c > n) :
    a > 2 ∨ a < -1 / 2 :=
by
  sorry

end parabola_axis_of_symmetry_a_eq_1_a_value_range_l422_422492


namespace ratio_of_areas_of_concentric_circles_l422_422979

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end ratio_of_areas_of_concentric_circles_l422_422979


namespace find_fourth_vertex_l422_422650

-- Define the coordinates of the given three vertices of the tetrahedron
def v1 := (0, 1, 2) : ℤ × ℤ × ℤ
def v2 := (4, 2, 1) : ℤ × ℤ × ℤ
def v3 := (3, 1, 5) : ℤ × ℤ × ℤ

-- Define the distance formula for ℤ coordinates
def dist (a b : ℤ × ℤ × ℤ) : ℚ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2)

-- Given that the side length is 3√2, so the squared distance is 18 (in ℚ for exact calculations)
def side_length_squared : ℚ := 18

-- State the proof problem
theorem find_fourth_vertex :
  ∃ (v4 : ℤ × ℤ × ℤ),
    dist v1 v4 ^ 2 = side_length_squared ∧
    dist v2 v4 ^ 2 = side_length_squared ∧
    dist v3 v4 ^ 2 = side_length_squared ∧
    v4 = (3, -2, 2) :=
by {
  sorry
}

end find_fourth_vertex_l422_422650


namespace ratio_of_speeds_l422_422262

/-- Jack's marathon time in hours -/
def jack_time : ℝ := 5

/-- Jill's marathon time in hours -/
def jill_time : ℝ := 4.2

/-- Marathon distance in kilometers -/
def marathon_distance : ℝ := 42

/-- Average speed of Jack -/
def jack_speed : ℝ := marathon_distance / jack_time

/-- Average speed of Jill -/
def jill_speed : ℝ := marathon_distance / jill_time

/-- The ratio of Jack's average speed to Jill's average speed is 84 : 100 -/
theorem ratio_of_speeds : jack_speed / jill_speed = 84 / 100 := by
  sorry

end ratio_of_speeds_l422_422262


namespace find_z9_l422_422048

theorem find_z9 (z : ℕ → ℂ) (h1 : ∀ n : ℕ, 0 < n → z n * complex.conj (z (n + 1)) = z (n ^ 2 + n + 1))
                (h2 : ∀ n : ℕ, 0 < n → z n * z (n + 2) = z (n + 1) ^ 2 - 1)
                (h3 : z 2 = 2 + complex.i) :
                z 9 = 9 + complex.i :=
sorry

end find_z9_l422_422048


namespace moon_speed_in_km_per_hour_l422_422265

theorem moon_speed_in_km_per_hour (v_km_per_sec : ℝ) (H : v_km_per_sec = 0.2) : 
  v_km_per_sec * 3600 = 720 := 
by
  rw [H]
  norm_num
  sorry

end moon_speed_in_km_per_hour_l422_422265


namespace no_valid_solution_for_function_l422_422797

theorem no_valid_solution_for_function (x : ℝ) (h : x ≠ 3) : 
  (f x = x + 1) → False := 
by
  let f := λ x, (x^2 - 9) / (x - 3)
  sorry

end no_valid_solution_for_function_l422_422797


namespace range_of_slope_l422_422430

-- Given conditions
def line_through_point (a : ℝ) (x y : ℝ) : Prop :=
  (1 - a) * x + (a + 1) * y - 4 * (a + 1) = 0

def lies_on_function (x y : ℝ) : Prop := 
  y = x + 1 / x

-- We need to prove that the range of slopes is [-3, +∞)
def slope_range (m : ℝ) : Prop :=
  m ≥ -3

theorem range_of_slope (a x y xQ yQ : ℝ) (h1 : line_through_point a x y)
(h2 : lies_on_function xQ yQ) (Q_def : yQ = xQ + 1 / xQ) (P_fixed : (x, y) = (0, 4))
(hQ : Q_def) : slope_range ((yQ - y) / (xQ - x)) := 
sorry

end range_of_slope_l422_422430


namespace max_value_of_y_l422_422777

-- Problem Statement: 
-- Given that \(a, b, c\) are all positive numbers, 
-- prove that the maximum value of \(y = \frac{ab + 2bc}{a^2 + b^2 + c^2}\) is \(\frac{\sqrt{5}}{2}\).

theorem max_value_of_y (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ y, y = (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ∧ y ≤ real.sqrt 5 / 2 :=
by
  sorry

end max_value_of_y_l422_422777


namespace favouring_more_than_one_is_39_l422_422110

def percentage_favouring_more_than_one (x : ℝ) : Prop :=
  let sum_two : ℝ := 8 + 6 + 4 + 2 + 7 + 5 + 3 + 5 + 3 + 2
  let sum_three : ℝ := 1 + 0.5 + 0.3 + 0.8 + 0.2 + 0.1 + 1.5 + 0.7 + 0.3 + 0.4
  let all_five : ℝ := 0.2
  x = sum_two - sum_three - all_five

theorem favouring_more_than_one_is_39 : percentage_favouring_more_than_one 39 := 
by
  sorry

end favouring_more_than_one_is_39_l422_422110


namespace proctoring_arrangements_l422_422485

/-- Consider 4 teachers A, B, C, D each teaching their respective classes a, b, c, d.
    Each teacher must not proctor their own class.
    Prove that there are exactly 9 ways to arrange the proctoring as required. -/
theorem proctoring_arrangements : 
  ∃ (arrangements : Finset ((Fin 4) → (Fin 4))), 
    (∀ (f : (Fin 4) → (Fin 4)), f ∈ arrangements → ∀ i : Fin 4, f i ≠ i) 
    ∧ arrangements.card = 9 :=
sorry

end proctoring_arrangements_l422_422485


namespace positive_area_triangles_5x5_grid_l422_422090

-- Define the range of the grid
def grid_points (n : Nat) : List (Nat × Nat) :=
  (List.range n).bind (λ x, (List.range n).map (λ y, (x + 1, y + 1)))

-- Define the condition for positive area of a triangle
def is_positive_area (p1 p2 p3 : Nat × Nat) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x2 - x1) * (y3 - y1) ≠ (x3 - x1) * (y2 - y1)

-- Count the number of triangles with positive area
def count_positive_area_triangles (n : Nat) : Nat :=
  let points := grid_points n
  points.choose 3 |>.count (λ t, match t with (p1, p2, p3) => is_positive_area p1 p2 p3)

-- Statement of the theorem
theorem positive_area_triangles_5x5_grid : count_positive_area_triangles 5 = 2170 :=
by
  sorry

end positive_area_triangles_5x5_grid_l422_422090


namespace find_a_l422_422792

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then 2^x - 1 else -x^2 - 2*x

theorem find_a (a : ℝ) (h : f(a) = 1) : a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l422_422792


namespace perfect_square_fraction_l422_422154

open Nat

theorem perfect_square_fraction (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
by 
  sorry

end perfect_square_fraction_l422_422154


namespace exponent_identity_l422_422815

variables (x y : ℝ)

theorem exponent_identity (h1 : 10^x = 3) (h2 : 10^y = 4) : 10^(3*x - 2*y) = 27 / 16 := 
by
  sorry

end exponent_identity_l422_422815


namespace first_three_terms_general_term_smallest_m_l422_422514

def seq_a (n : ℕ+) (S : ℕ+ → ℕ) : ℕ :=
  -- sequence a_n defined such that 8 * S_n = (a_n + 2)^2
  sorry

def seq_b (a : ℕ+ → ℕ) (n : ℕ+) : ℕ :=
  -- sequence b_n defined as b_n = 4 / (a_n * a_{n+1})
  sorry

theorem first_three_terms (S : ℕ+ → ℕ) (a : ℕ+ → ℕ) (h : ∀ n : ℕ+, 8 * S n = (a n + 2)^2) :
  a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 10 :=
sorry

theorem general_term (S : ℕ+ → ℕ) (a : ℕ+ → ℕ) (h : ∀ n : ℕ+, 8 * S n = (a n + 2)^2) :
  ∀ n : ℕ+, a n = 4 * n - 2 :=
sorry

theorem smallest_m (S : ℕ+ → ℕ) (a : ℕ+ → ℕ) (b : ℕ+ → ℕ) (T : ℕ+ → ℕ) (h : ∀ n : ℕ+, 8 * S n = (a n + 2)^2)
  (h_b : ∀ n : ℕ+, b n = 4 / (a n * a (n + 1))) (h_T : ∀ n : ℕ+, T n = ∑ i in range (n+1), b i) :
  ∃ m > 0, m = 10 ∧ ∀ n : ℕ+, T n < m / 20 :=
sorry

end first_three_terms_general_term_smallest_m_l422_422514


namespace integral_f1_eq_l422_422780

def f (x : ℝ) : ℝ := 1 / x

def f_K (K : ℝ) (x : ℝ) : ℝ :=
  if (1 / x) ≤ K then K else (1 / x)

def f_1 (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 else 1 / x

theorem integral_f1_eq :
  ∫ x in (1 / 4)..2, f_1 x = 1 + 2 * Real.log 2 :=
by
  sorry

end integral_f1_eq_l422_422780


namespace polar_coordinates_of_center_l422_422494

-- Define the circle in polar coordinates
def circle_eq (ρ θ : ℝ) : Prop := ρ = Real.cos (θ + Real.pi / 3)

-- Define the conversion formulas between polar and rectangular coordinates
def ρ_from_rect (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)
def θ_from_rect (x y : ℝ) : ℝ := Real.atan2 y x

-- Specify the center of the circle in rectangular coordinates
def center_rect : ℝ × ℝ := (1 / 4, -Real.sqrt 3 / 4)

-- Specify the correct polar coordinates of the center
def center_polar : ℝ × ℝ := (1 / 2, -Real.pi / 3)

-- The main theorem
theorem polar_coordinates_of_center : 
  ρ_from_rect center_rect.1 center_rect.2 = center_polar.1 ∧ 
  θ_from_rect center_rect.1 center_rect.2 = center_polar.2 :=
  sorry

end polar_coordinates_of_center_l422_422494


namespace midfielders_to_defenders_ratio_l422_422689

theorem midfielders_to_defenders_ratio 
  (total_players : ℕ) (goalies : ℕ) (defenders : ℕ) (strikers : ℕ)
  (h_total_players : total_players = 40)
  (h_goalies : goalies = 3)
  (h_defenders : defenders = 10)
  (h_strikers : strikers = 7) :
  let midfielders := total_players - goalies - defenders - strikers in
  midfielders / defenders = 2 :=
by
  sorry

end midfielders_to_defenders_ratio_l422_422689


namespace find_n_l422_422239

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 17) (h3 : -150 ≡ n [MOD 17]) : n = 3 :=
sorry

end find_n_l422_422239


namespace land_division_possible_l422_422370

-- Define the basic properties and conditions of the plot
structure Plot :=
  (is_square : Prop)
  (has_center_well : Prop)
  (has_four_trees : Prop)
  (has_four_gates : Prop)

-- Define a section of the plot
structure Section :=
  (contains_tree : Prop)
  (contains_gate : Prop)
  (equal_fence_length : Prop)
  (unrestricted_access_to_well : Prop)

-- Define the property that indicates a valid division of the plot
def valid_division (p : Plot) (sections : List Section) : Prop :=
  sections.length = 4 ∧
  (∀ s ∈ sections, s.contains_tree) ∧
  (∀ s ∈ sections, s.contains_gate) ∧
  (∀ s ∈ sections, s.equal_fence_length) ∧
  (∀ s ∈ sections, s.unrestricted_access_to_well)

-- Define the main theorem to prove
theorem land_division_possible (p : Plot) : 
  p.is_square ∧ p.has_center_well ∧ p.has_four_trees ∧ p.has_four_gates → 
  ∃ sections : List Section, valid_division p sections :=
by
  sorry

end land_division_possible_l422_422370


namespace sqrt_Sn_arithmetic_sum_of_bn_l422_422762

-- Define the arithmetic sequence {aₙ} and its sum Sₙ
def is_arithmetic_sequence {α : Type*} [linear_ordered_field α] (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms {α : Type*} [linear_ordered_field α] (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n, S n = (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2

-- Given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a : ℕ → α) (S : ℕ → α)
variable (d : α)
variable (b : ℕ → α)
variable (T : ℕ → α)

-- Given conditions in Lean
axiom Sn_condition : S 15 = 225
axiom a3_a6_condition : a 3 + a 6 = 16
axiom arithmetic_sequence_condition : is_arithmetic_sequence a d
axiom sum_condition : sum_of_first_n_terms a S

-- Prove that the sequence {sqrt(Sₙ)} is arithmetic
theorem sqrt_Sn_arithmetic : ∀ n, S n = n * n → is_arithmetic_sequence (λ n, (S n) ^ (1/2)) 1 :=
by sorry

-- Define bn and prove its sum
def b (n : ℕ) : α := 2^n * a n

def T (n : ℕ) := (2 * n - 3) * 2^(n + 1) + 6

theorem sum_of_bn (n : ℕ) : T n = ∑ i in finset.range n, b i :=
by sorry

end sqrt_Sn_arithmetic_sum_of_bn_l422_422762


namespace complex_division_result_l422_422883

section
  variable (i : ℂ) (hi : i * i = -1)

  def complex_division : ℂ :=
    4 * i / (Real.sqrt 3 + i)

  theorem complex_division_result : complex_division i hi = 1 + Real.sqrt 3 * i :=
  by 
    -- proof details skipped
    sorry
end

end complex_division_result_l422_422883


namespace smallest_positive_period_monotonically_decreasing_interval_bounded_max_min_l422_422071

noncomputable def f (x : ℝ) : ℝ := 2 * sin^2 (π / 4 + x) - sqrt 3 * cos (2 * x)

theorem smallest_positive_period : ∀ x, f (x + π) = f x := 
sorry

theorem monotonically_decreasing_interval (k : ℤ) : 
  ∀ x, 
    (π / 4 ≤ x) ∧ (x ≤ π / 2) → 
    f (x) ∈ Icc (1 + 2 * sin (2 * (π / 4) - π / 3)) (1 + 2 * sin (2 * (π / 2) - π / 3)) := 
sorry

theorem bounded_max_min (x : ℝ) : 
  (π / 4 ≤ x) ∧ (x ≤ π / 2) → 
  f x ≤ 3 ∧ f x ≥ 2 := 
sorry

end smallest_positive_period_monotonically_decreasing_interval_bounded_max_min_l422_422071


namespace angle_ABC_is_90_radius_of_circles_l422_422025

variable {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variable (AB BC AC : ℝ)
variable (triangle : A → B → C → Prop)
variable (is_tangent_to : Π (x y : A), Prop)
variable (congruent_circles : A → B → Prop)

theorem angle_ABC_is_90 {A B C : Type} [metric_space A] [metric_space B] [metric_space C] 
  (triangle ABC : A → B → C → Prop) 
  (is_tangent_to : Π (x y : A), Prop)
  (congruent_circles : A → B → Prop)
  (h1 : congruent_circles A B)
  (h2 : congruent_circles B C)
  (h3 : is_tangent_to A B)
  (h4 : is_tangent_to B C)
  (h5 : triangle A B C)
  : ∠ABC = 90° :=
sorry

theorem radius_of_circles {A B C : Type} [metric_space α]
  (triangle ABC : A → B → C → Prop)
  (is_tangent_to : Π (x y : A), Prop)
  (congruent_circles : A → B → Prop)
  (h1 : congruent_circles A B)
  (h2 : congruent_circles B C)
  (h3 : is_tangent_to A B)
  (h4 : is_tangent_to B C)
  (h5 : triangle A B C)
  (h6 : AB = 3)
  (h7 : BC = 4)
  (h8 : ∠ABC = 90°)
  : ∀ (r : ℝ), r = 5 / 9 :=
sorry

end angle_ABC_is_90_radius_of_circles_l422_422025


namespace symmetrical_line_eq_l422_422562

theorem symmetrical_line_eq (m b : ℝ) (h : m = 3 ∧ b = -4) :
  ∃ m' b', m' = -m ∧ b' = -b ∧ (∀ x : ℝ, (y = m * x + b) ↔ (y = m' * x + b')) := 
begin
  sorry
end

end symmetrical_line_eq_l422_422562


namespace sum_subset_exists_l422_422580

theorem sum_subset_exists {a : ℕ → ℕ} (h1 : ∑ i in finset.range 101, a i = 200) : 
  ∃ S ⊆ finset.range 101, ∑ i in S, a i = 100 :=
sorry

end sum_subset_exists_l422_422580


namespace alpha_minus_beta_l422_422892

open Real

theorem alpha_minus_beta (α β γ : ℝ) (h₁ : 0 < α ∧ α < β ∧ β < γ ∧ γ < 2 * π)
    (h₂ : ∀ x : ℝ, cos(x + α) + cos(x + β) + cos(x + γ) = 0) :
    α - β = - (2 * π / 3) :=
sorry

end alpha_minus_beta_l422_422892


namespace intersection_set_M_N_l422_422349

noncomputable def set_M : Set ℝ := {x | x^2 ≤ 2 * x}

noncomputable def set_N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (2 - |x|)}

theorem intersection_set_M_N : set_M ∩ set_N = Icc 0 2 \ {2} := by
  sorry

end intersection_set_M_N_l422_422349


namespace curves_intersection_l422_422348

theorem curves_intersection :
  let f1 := λ x : ℝ, 3 * x^2 - 4 * x + 2
  let f2 := λ x : ℝ, x^3 - 2 * x^2 + x + 2
  let x1 := 0
  let x2 := (5 + Real.sqrt 5) / 2
  let x3 := (5 - Real.sqrt 5) / 2
  let y1 := f1 x1
  let y2 := f1 x2
  let y3 := f1 x3
  (f1 x1 = f2 x1 ∧ f1 x2 = f2 x2 ∧ f1 x3 = f2 x3) :=
begin
  sorry,
end

end curves_intersection_l422_422348


namespace sqrt_sum_of_four_terms_of_4_pow_4_l422_422615

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l422_422615


namespace ratio_of_areas_of_concentric_circles_l422_422970

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_concentric_circles_l422_422970


namespace minimum_average_from_digits_no_repeats_l422_422376

theorem minimum_average_from_digits_no_repeats :
  ∃ S : set ℕ, S.card = 6 ∧ S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∧ (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → (x % 10 ≠ y % 10 ∧ x / 10 ≠ y / 10))
  ∧ (∀ s1 s2 s3, s1 ∈ { (d1, d2) | d1 ∈ S ∧ d2 ∈ S }
   → s2 ∈ { (d3, d4) | d3 ∈ S ∧ d4 ∈ S }
   → s3 ∈ { (d5, d6) | d5 ∈ S ∧ d6 ∈ S }
   → (s1 ≠ s2 ∧ s2 ≠ s3 ∧ s3 ≠ s1)
   → (s1.1 * 10 + s1.2 + s2.1 * 10 + s2.2 + s3.1 * 10 + s3.2 +
     (∑ x in S, x) - (s1.1 + s1.2 + s2.1 + s2.2 + s3.1 + s3.2)) / 6 = 16.5) := sorry

end minimum_average_from_digits_no_repeats_l422_422376


namespace curve_C2_eq_correct_polar_distance_correct_l422_422861

variables {θ : ℝ} {x y x' y' : ℝ}

def curve_C1_param (θ : ℝ) : ℝ × ℝ :=
(1 + sqrt 3 * cos θ, sqrt 3 * sin θ)

def curve_C2_eq : Prop :=
(x - 2)^2 + y^2 = 12

def point_M (θ : ℝ) : ℝ × ℝ :=
curve_C1_param θ

def point_P (M : ℝ × ℝ) : ℝ × ℝ :=
(2 * M.1, 2 * M.2)

noncomputable def polar_distance (ρA ρB : ℝ) : ℝ :=
ρB - ρA

theorem curve_C2_eq_correct :
  ∀ θ, let M := point_M θ in let P := point_P M in
  (P.1 - 2)^2 + P.2^2 = 12 := sorry

theorem polar_distance_correct :
  let ρA := 2 in let ρB := 4 in
  polar_distance ρA ρB = 2 := sorry

end curve_C2_eq_correct_polar_distance_correct_l422_422861


namespace profit_percentage_is_sixteen_l422_422257

def profit_percentage (P : ℝ) : ℝ :=
  let bought_price := 0.80 * P
  let sold_price := 1.16 * P
  let profit := sold_price - P
  profit / P * 100

theorem profit_percentage_is_sixteen (P : ℝ) (hP : P > 0) :
  profit_percentage P = 16 := 
by
  sorry

end profit_percentage_is_sixteen_l422_422257


namespace fortune_cookie_problem_l422_422688

theorem fortune_cookie_problem : 
  ∃ n : ℕ, let d := n.digits 10 in 
    (34 + 18 + 73 + n) % 100 ≥ 100 / 6 ∧ 
    list.sorted (<) (list.reverse d) ∧
    ∀ m, let dm := m.digits 10 in 
      (34 + 18 + 73 + m) % 100 ≥ 100 / 6 ∧ 
      list.sorted (<) (list.reverse dm) → n ≤ m :=
begin
  sorry
end

end fortune_cookie_problem_l422_422688


namespace largest_base_for_12_cubed_l422_422244

theorem largest_base_for_12_cubed (b : ℕ) (h_b_gt_1 : b > 1) :
  (∑ d in (Nat.digits b (12^3)), d) ≠ 9 → b ≤ 8 := sorry

end largest_base_for_12_cubed_l422_422244


namespace ellipse_equation_l422_422766

-- Definition of ellipse parameters according to the problem conditions
def semiMajorAxis (a : ℝ) := a > 0
def semiMinorAxis (b : ℝ) := b > 0
def eccentricity (a b : ℝ) := 1/3 = Real.sqrt (1 - (b^2)/(a^2))

-- Given conditions as hypotheses
variables {a b x y : ℝ}
hypothesis a_gt_b : a > b
hypothesis b_gt_0 : b > 0
hypothesis ecc : eccentricity a b

-- Result equation of the ellipse
theorem ellipse_equation : 
  ∃ (a b : ℝ), semiMajorAxis a ∧ semiMinorAxis b ∧ eccentricity a b ∧ 
  ((∃ m : ℝ, a = 3 * m ∧ b = 2 * Real.sqrt 2 * m) → ∀ x y, 
  (x^2 / (a^2) + y^2 / (b^2) = 1 ↔ x^2 / 9 + y^2 / 8 = 1)) :=
begin
  sorry -- Proof omitted
end

end ellipse_equation_l422_422766


namespace frosting_problem_equivalent_l422_422334

/-
Problem:
Cagney can frost a cupcake every 15 seconds.
Lacey can frost a cupcake every 40 seconds.
Mack can frost a cupcake every 25 seconds.
Prove that together they can frost 79 cupcakes in 10 minutes.
-/

def cupcakes_frosted_together_in_10_minutes (rate_cagney rate_lacey rate_mack : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let rate_cagney := 1 / 15
  let rate_lacey := 1 / 40
  let rate_mack := 1 / 25
  let combined_rate := rate_cagney + rate_lacey + rate_mack
  combined_rate * time_seconds

theorem frosting_problem_equivalent:
  cupcakes_frosted_together_in_10_minutes 1 1 1 10 = 79 := by
  sorry

end frosting_problem_equivalent_l422_422334


namespace intersection_non_empty_l422_422435

open Set

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x | 2 * x^2 - 5 * x < 0}

theorem intersection_non_empty (a : ℤ) (h : (M a) ∩ N ≠ ∅) : a = 1 ∨ a = 2 := 
sorry

end intersection_non_empty_l422_422435


namespace g_eq_g_inv_l422_422721

def g (x : ℝ) : ℝ := 3 * x + 4
def g_inv (x : ℝ) : ℝ := (x - 4) / 3

theorem g_eq_g_inv (x : ℝ) : g x = g_inv x → x = -2 := by
    sorry

end g_eq_g_inv_l422_422721


namespace fraction_to_decimal_l422_422712

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fraction_to_decimal_l422_422712


namespace max_gas_tank_capacity_l422_422246

-- Definitions based on conditions
def start_gas : ℕ := 10
def gas_used_store : ℕ := 6
def gas_used_doctor : ℕ := 2
def refill_needed : ℕ := 10

-- Theorem statement based on the equivalence proof problem
theorem max_gas_tank_capacity : 
  start_gas - (gas_used_store + gas_used_doctor) + refill_needed = 12 :=
by
  -- Proof steps go here
  sorry

end max_gas_tank_capacity_l422_422246


namespace solve_sqrt_equation_l422_422014

theorem solve_sqrt_equation (z : ℝ) : (sqrt (5 - 4 * z) = 7) ↔ (z = -11) :=
by
  sorry

end solve_sqrt_equation_l422_422014


namespace range_of_a_l422_422808

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1)
def q : Prop := (a > (1 / 2))

theorem range_of_a (hpq_true: p a ∨ q a) (hpq_false: ¬ (p a ∧ q a)) :
  (0 < a ∧ a ≤ (1 / 2)) ∨ (a ≥ 1) :=
sorry

end range_of_a_l422_422808


namespace area_of_region_in_octagon_l422_422702

noncomputable def area_of_shaded_region (side_length : ℝ) (num_sides : ℕ) : ℝ :=
  let area_of_octagon := 2 * (1 + Real.sqrt 2) * side_length^2
  let radius := side_length / 2
  let area_of_semicircle := (π * radius^2) / 2
  let total_area_semicircles := area_of_semicircle * num_sides
  area_of_octagon - total_area_semicircles

theorem area_of_region_in_octagon : 
  area_of_shaded_region 3 8 = 18 * (1 + Real.sqrt 2) - 9 * π := 
by 
  sorry

end area_of_region_in_octagon_l422_422702


namespace matrix_determinant_l422_422799

theorem matrix_determinant :
  let x := 1
  let y := -1
  det ![![x, -3], ![y, 2]] = -1 := 
by
  -- define x and y
  let x := 1
  let y := -1
  -- define the matrix
  have matrix := ![![x, -3], ![y, 2]]
  -- calculate the determinant
  calc
    det matrix = x * 2 - (-3) * y : by simp [matrix, det]
           ... = 1 * 2 - (-3) * (-1) : by simp [x, y]
           ... = 2 - 3 : by norm_num
           ... = -1 : by norm_num

end matrix_determinant_l422_422799


namespace power_func_passes_point_l422_422104

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_func_passes_point (f : ℝ → ℝ) (h : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) 
  (h_point : f 9 = 1 / 3) : f 25 = 1 / 5 :=
sorry

end power_func_passes_point_l422_422104


namespace cos_angle_sum_l422_422864

-- Define the tetrahedron points A, B, C, D, and O
variables {A B C D O : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]

-- All points are in the same metric space
variables [has_dist A O] [has_dist A B] [has_dist B C] [has_dist A C] 
variables [has_dist D A] [has_dist D B] [has_dist D C] [has_dist A O]

-- Define properties of the tetrahedron as conditions
def tetrahedron {A B C D O : Type} [has_dist A B]
  (h1: dist A B = dist B C) 
  (h2: dist B C = dist C A) 
  (h3: dist D A = dist D B) 
  (h4: dist D B = dist D C) 
  (h5: O = center of circumscribed sphere around A B C D) : Prop :=

-- Define the statement to be proved
theorem cos_angle_sum (A B C D O : Type) [has_dist A B] [has_dist B C] 
  [has_dist A C] [has_dist D B] [has_dist D C] [has_dist A O] :
  dist A B = dist B C →
  dist B C = dist C A →
  dist D A = dist D B →
  dist D B = dist D C →
  O = center of circumscribed sphere around A B C D →
  cos (angle A O B) + cos (angle A O D) ≥ -2 / 3 := by
  sorry

end cos_angle_sum_l422_422864


namespace value_of_f_m_plus_one_depends_on_m_l422_422095

def f (x a : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one_depends_on_m (m a : ℝ) (h : f (-m) a < 0) :
  (∃ m, f (m + 1) a < 0) ∧ (∃ m, f (m + 1) a > 0) :=
by
  sorry

end value_of_f_m_plus_one_depends_on_m_l422_422095


namespace triangle_perimeter_l422_422429

-- Given the hyperbola with equation x^2 - 4y^2 = 4.
def is_hyperbola (x y : ℝ) : Prop := x^2 - 4 * y^2 = 4

-- Let F1 and F2 be the foci of the hyperbola.
axiom F1 F2 : ℝ × ℝ

-- A line passing through F1 intersects the left branch at points A and B.
axiom A B : ℝ × ℝ

-- Given the distance |AB| = 5.
axiom distance_AB : ℝ 
axiom h_distance_AB : distance_AB = 5

-- Condition of the hyperbola at A and B.
axiom h_A_hyperbola : is_hyperbola A.1 A.2
axiom h_B_hyperbola : is_hyperbola B.1 B.2

-- Definition for the distance between two points.
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Prove the perimeter of triangle \(\triangle AF_{2}B\) is 18.
theorem triangle_perimeter :
  dist A F2 + dist B F2 + dist A B = 18 :=
sorry

end triangle_perimeter_l422_422429


namespace smallest_average_l422_422387

noncomputable def smallest_possible_average : ℕ := 165 / 10

theorem smallest_average (s d: Finset ℕ) 
  (h1 : s.card = 3) 
  (h2 : d.card = 3) 
  (h3 : ∀x ∈ s ∪ d, x ∈ (Finset.range 10).erase 0)
  (h4 : (s ∪ d).card = 6)
  (h5 : s ∩ d = ∅) : 
  (∑ x in s, x + ∑ y in d, y) / 6 = smallest_possible_average :=
sorry

end smallest_average_l422_422387


namespace smallest_possible_average_l422_422385

def smallest_average (s : Finset (Fin 10)) : ℕ :=
  (Finset.sum s).toNat / 6

theorem smallest_possible_average : ∃ (s1 s2 : Finset ℕ), (s1.card = 3 ∧ s2.card = 3 ∧ 
 (s1 ∪ s2).card = 6 ∧ ∀ x, x ∈ s1 ∪ s2 → x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  smallest_average (s1 ∪ s2) = 16.5) sorry

end smallest_possible_average_l422_422385


namespace partition_people_l422_422646

-- Define the problem setup:
variables (People : Type) (Country : Type)
variables (countryOf : People → Country) (next : People → People)

-- Conditions:
axiom A1 : ∃ (P : fin 100 → People), (∀ i, countryOf (P i) = countryOf (P (i + 4))) ∧ (∀ i, next (P i) = P ((i + 1) % 100))
axiom A2 : ∀ (P : People) (i j : fin 100), (i ≠ j) → countryOf (P i) ≠ countryOf (P j)
axiom A3 : ∀ (P : People) (i : fin 100), next (next (next (P i))) ≠ P i

-- Proposition to prove:
theorem partition_people : ∃ (G : People → fin 4),
  (∀ i j : People, G i = G j → countryOf i ≠ countryOf j) ∧
  (∀ i, G (next i) ≠ G i) :=
sorry

end partition_people_l422_422646


namespace area_of_triangles_l422_422482

-- Definitions from conditions
variable (A B C D E : Point)
variable (is_isosceles : AB = AC ∧ AB = 17)
variable (BC_eq_16 : BC = 16)
variable (altitude_bisects : BD = DC ∧ BD = 8)
variable (DE_eq_8 : DE = 8)

-- Theorem statement
theorem area_of_triangles :
  (area_of_triangle ABC = 120) ∧ (area_of_triangle AEC = 120) :=
by
  -- Discussion of calculations here
  sorry

end area_of_triangles_l422_422482


namespace exists_unbiased_estimator_iff_l422_422523

variables {θ : ℝ} {Θ₀ : set ℝ}

theorem exists_unbiased_estimator_iff (Hθ : θ ∈ Θ₀) (HΘ₀ : Θ₀ ⊆ set.Icc 0 1) :
  (∃ T, unbiased T θ ∧ (∀ ω, T ω ∈ Θ₀)) ↔ (0 ∈ Θ₀ ∧ 1 ∈ Θ₀) :=
sorry

end exists_unbiased_estimator_iff_l422_422523


namespace intersection_A_B_eq_l422_422078

def A : Set ℝ := { x | (x / (x - 1)) ≥ 0 }

def B : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B_eq :
  (A ∩ B) = { y : ℝ | 1 < y } :=
sorry

end intersection_A_B_eq_l422_422078


namespace num_invitations_eq_63_num_arrangements_eq_504_probability_f_eq_1_over_20_l422_422055

-- (1) Proof Problem
theorem num_invitations_eq_63 :
  let individuals := {A, B, C, D, E, F}
  let n := 6 
  (2^n - 1) = 63 := 
by 
  let individuals := {A, B, C, D, E, F}
  let n := 6
  sorry

-- (2) Proof Problem
theorem num_arrangements_eq_504 :
  let A_does_not_1st_event := true
  let B_does_not_3rd_event := true
  let total_arrangements := 720
  let remove_A_first := 120
  let remove_B_third := 120
  let remove_both := 24
  (total_arrangements - 2 * remove_A_first + remove_both) = 504 := 
by
  let A_does_not_1st_event := true
  let B_does_not_3rd_event := true
  let total_arrangements := 720
  let remove_A_first := 120
  let remove_B_third := 120
  let remove_both := 24
  sorry

-- (3) Proof Problem
theorem probability_f_eq_1_over_20 :
  let total_ways := 720
  let case_DEF_in_same_event := 36
  (case_DEF_in_same_event / total_ways) = 1 / 20 := 
by 
  let total_ways := 720
  let case_DEF_in_same_event := 36
  sorry

end num_invitations_eq_63_num_arrangements_eq_504_probability_f_eq_1_over_20_l422_422055


namespace midpoint_in_xOz_plane_l422_422405

-- Define the points A and B
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def point_A : Point3D := { x := 1, y := -1, z := 1 }
def point_B : Point3D := { x := 3, y := 1, z := 5 }

-- Function to compute the midpoint of two points
def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2,
    z := (p1.z + p2.z) / 2 }

-- The xOz plane condition
def is_in_xOz_plane (p : Point3D) : Prop :=
  p.y = 0

-- Theorem to prove the midpoint lies in the xOz plane
theorem midpoint_in_xOz_plane : is_in_xOz_plane (midpoint point_A point_B) :=
by sorry

end midpoint_in_xOz_plane_l422_422405


namespace min_value_sin_cos_expr_l422_422020

open Real

theorem min_value_sin_cos_expr :
  (∀ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 ≥ 3 / 5) ∧ 
  (∃ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 = 3 / 5) :=
by
  sorry

end min_value_sin_cos_expr_l422_422020


namespace find_m_plus_n_l422_422710

noncomputable def vertices := [(10, 45), (10, 114), (28, 153), (28, 84)]

def midpoint (p1 p2 : ℤ × ℤ) : ℤ × ℤ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def center := midpoint (10, 45) (28, 153)

def slope_through_origin (c : ℤ × ℤ) : ℚ :=
  (c.2 : ℚ) / c.1

theorem find_m_plus_n : (m n : ℕ) (h : m + n = 118) 
  (h_rel_prime : Int.gcd m n = 1) (slope := slope_through_origin center) :
  slope = m / n :=
  sorry

end find_m_plus_n_l422_422710


namespace least_number_to_multiply_for_multiple_of_112_l422_422208

theorem least_number_to_multiply_for_multiple_of_112 (n : ℕ) : 
  (Nat.lcm 72 112) / 72 = 14 := 
sorry

end least_number_to_multiply_for_multiple_of_112_l422_422208


namespace fraction_to_decimal_l422_422714

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 :=
by
  sorry

end fraction_to_decimal_l422_422714


namespace evaluate_expression_l422_422361

-- Definition of the given condition.
def sixty_four_eq_sixteen_squared : Prop := 64 = 16^2

-- The statement to prove that the given expression equals the answer.
theorem evaluate_expression (h : sixty_four_eq_sixteen_squared) : 
  (16^24) / (64^8) = 16^8 :=
by 
  -- h contains the condition that 64 = 16^2, but we provide a proof step later with sorry
  sorry

end evaluate_expression_l422_422361


namespace fraction_to_decimal_l422_422711

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fraction_to_decimal_l422_422711


namespace solve_for_y_l422_422943

theorem solve_for_y :
  ∀ (y : ℝ), (9 * y^2 + 49 * y^2 + 21/2 * y^2 = 1300) → y = 4.34 := 
by sorry

end solve_for_y_l422_422943


namespace number_of_students_per_normal_class_l422_422439

theorem number_of_students_per_normal_class (total_students : ℕ) (percentage_moving : ℕ) (grade_levels : ℕ) (adv_class_size : ℕ) (additional_classes : ℕ) 
  (h1 : total_students = 1590) 
  (h2 : percentage_moving = 40) 
  (h3 : grade_levels = 3) 
  (h4 : adv_class_size = 20) 
  (h5 : additional_classes = 6) : 
  (total_students * percentage_moving / 100 / grade_levels - adv_class_size) / additional_classes = 32 :=
by
  sorry

end number_of_students_per_normal_class_l422_422439


namespace game_score_correct_answers_l422_422907

theorem game_score_correct_answers :
  ∃ x : ℕ, (∃ y : ℕ, x + y = 30 ∧ 7 * x - 12 * y = 77) ∧ x = 23 :=
by
  use 23
  sorry

end game_score_correct_answers_l422_422907


namespace polynomial_use_square_of_binomial_form_l422_422996

theorem polynomial_use_square_of_binomial_form (a b x y : ℝ) :
  (1 + x) * (x + 1) = (x + 1) ^ 2 ∧ 
  (2 * a + b) * (b - 2 * a) = b^2 - 4 * a^2 ∧ 
  (-a + b) * (a - b) = - (a - b)^2 ∧ 
  (x^2 - y) * (y^2 + x) ≠ (x + y)^2 :=
by 
  sorry

end polynomial_use_square_of_binomial_form_l422_422996


namespace triangle_shape_l422_422866

-- Definitions of the conditions
variables {α : Type*}
variables (A B C : α) [Real α] [differentiable_structure α]

-- Given triangle sides opposite to angles A, B, and C
variables (a b c : α)
variables (h: c - a * Real.cos B = (2 * a - b) * Real.cos A)

-- Conclusion: the shape of the triangle is either isosceles or right-angled
theorem triangle_shape :
  (A = π/2) ∨ (sin B = sin A) := 
sorry

end triangle_shape_l422_422866


namespace arithmetic_mean_l422_422168

theorem arithmetic_mean (a b c : ℚ) (h₁ : a = 8 / 12) (h₂ : b = 10 / 12) (h₃ : c = 9 / 12) :
  c = (a + b) / 2 :=
by
  sorry

end arithmetic_mean_l422_422168


namespace minimum_average_from_digits_no_repeats_l422_422375

theorem minimum_average_from_digits_no_repeats :
  ∃ S : set ℕ, S.card = 6 ∧ S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∧ (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → (x % 10 ≠ y % 10 ∧ x / 10 ≠ y / 10))
  ∧ (∀ s1 s2 s3, s1 ∈ { (d1, d2) | d1 ∈ S ∧ d2 ∈ S }
   → s2 ∈ { (d3, d4) | d3 ∈ S ∧ d4 ∈ S }
   → s3 ∈ { (d5, d6) | d5 ∈ S ∧ d6 ∈ S }
   → (s1 ≠ s2 ∧ s2 ≠ s3 ∧ s3 ≠ s1)
   → (s1.1 * 10 + s1.2 + s2.1 * 10 + s2.2 + s3.1 * 10 + s3.2 +
     (∑ x in S, x) - (s1.1 + s1.2 + s2.1 + s2.2 + s3.1 + s3.2)) / 6 = 16.5) := sorry

end minimum_average_from_digits_no_repeats_l422_422375


namespace solution_set_ineq_l422_422618

theorem solution_set_ineq (m : ℝ) (hm : m > 1) :
  {x : ℝ | x^2 + (m-1) * x - m >= 0} = {x : ℝ | x <= -m ∨ x >= 1} :=
sorry

end solution_set_ineq_l422_422618


namespace general_formula_for_an_l422_422788

-- Definitions for the first few terms of the sequence
def a1 : ℚ := 1 / 7
def a2 : ℚ := 3 / 77
def a3 : ℚ := 5 / 777

-- The sequence definition as per the identified pattern
def a_n (n : ℕ) : ℚ := (18 * n - 9) / (7 * (10^n - 1))

-- The theorem to establish that the sequence definition for general n holds given the initial terms 
theorem general_formula_for_an {n : ℕ} :
  (n = 1 → a_n n = a1) ∧
  (n = 2 → a_n n = a2) ∧ 
  (n = 3 → a_n n = a3) ∧ 
  (∀ n > 3, a_n n = (18 * n - 9) / (7 * (10^n - 1))) := 
by
  sorry

end general_formula_for_an_l422_422788


namespace range_of_values_for_a_l422_422804

theorem range_of_values_for_a (a : ℝ) :
  (∃ x ∈ set.Icc 1 2, x^2 + 2 * x - a ≥ 0) → a ≤ 8 :=
by
  sorry

end range_of_values_for_a_l422_422804


namespace more_cookies_than_brownies_l422_422333

-- Define the initial quantities
def initial_cookies : ℕ := 60
def initial_brownies : ℕ := 10

-- Define the actions each day as a list of tuples
def daily_actions : list (ℕ × ℕ × ℕ × ℕ) :=
  [(2, 1, 10, 0), (4, 2, 0, 4), (3, 1, 5, 2),
   (5, 1, 0, 0), (4, 3, 8, 0), (3, 2, 0, 1), (2, 1, 0, 5)]

-- Sum up the total effect on cookies and brownies
def final_cookies : ℕ := daily_actions.foldl (λ c d, c - d.1 + d.3) initial_cookies
def final_brownies : ℕ := daily_actions.foldl (λ b d, b - d.2 + d.4) initial_brownies

theorem more_cookies_than_brownies : final_cookies = final_brownies + 49 :=
  sorry

end more_cookies_than_brownies_l422_422333


namespace indeterminate_turnips_l422_422543

theorem indeterminate_turnips 
(h_sandy_carrots : 8 ≠ ∀ n, Sandy_grow_turnips) 
(h_mary_carrots : 6 ≠ ∀ n,  Mary_grow_turnips) 
(h_total_carrots : 14 ≠ ∀ n) : 
∀ t : ℕ, (Sandy_grow_turnips t) → t = 8 :=
by
  sorry

end indeterminate_turnips_l422_422543


namespace polygon_sides_from_interior_angles_l422_422955

theorem polygon_sides_from_interior_angles (S : ℕ) (h : S = 1260) : S = (9 - 2) * 180 :=
by
  sorry

end polygon_sides_from_interior_angles_l422_422955


namespace equal_mondays_fridays_l422_422307

theorem equal_mondays_fridays (days_in_month : ℕ) (days_in_month = 30) : 
  (∃ num_days_eq : ℕ, num_days_eq = 3) :=
begin
  sorry,
end

end equal_mondays_fridays_l422_422307


namespace smallest_palindrome_base2_base4_l422_422342

def is_palindrome_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := (Nat.digits b n)
  digits = digits.reverse

theorem smallest_palindrome_base2_base4 : 
  ∃ (x : ℕ), x > 15 ∧ is_palindrome_base x 2 ∧ is_palindrome_base x 4 ∧ x = 17 :=
by
  sorry

end smallest_palindrome_base2_base4_l422_422342


namespace area_of_triangle_formed_by_lines_l422_422240

def line1 (x : ℝ) : ℝ := 5
def line2 (x : ℝ) : ℝ := 1 + x
def line3 (x : ℝ) : ℝ := 1 - x

theorem area_of_triangle_formed_by_lines :
  let A := (4, 5)
  let B := (-4, 5)
  let C := (0, 1)
  (1 / 2) * abs (4 * 5 + (-4) * 1 + 0 * 5 - (5 * (-4) + 1 * 4 + 5 * 0)) = 16 := by
  sorry

end area_of_triangle_formed_by_lines_l422_422240


namespace students_in_each_normal_class_l422_422441

theorem students_in_each_normal_class
  (initial_students : ℕ)
  (percent_to_move : ℚ)
  (grade_levels : ℕ)
  (students_in_advanced_class : ℕ)
  (num_of_normal_classes : ℕ)
  (h_initial_students : initial_students = 1590)
  (h_percent_to_move : percent_to_move = 0.4)
  (h_grade_levels : grade_levels = 3)
  (h_students_in_advanced_class : students_in_advanced_class = 20)
  (h_num_of_normal_classes : num_of_normal_classes = 6) :
  let students_moving := (initial_students : ℚ) * percent_to_move,
      students_per_grade := students_moving / grade_levels,
      students_remaining := students_per_grade - (students_in_advanced_class : ℚ),
      students_per_normal_class := students_remaining / (num_of_normal_classes : ℚ)
  in students_per_normal_class = 32 := 
by
  sorry

end students_in_each_normal_class_l422_422441


namespace part1_part2_part3_l422_422423

noncomputable def f (x : ℝ) (a b : ℝ) := (a * Real.log x / x) + b
noncomputable def tangent_line (x y : ℝ) := x - y - 1 = 0
noncomputable def g (x c : ℝ) := Real.log c x - x

variables {a b c : ℝ}

/- Part 1 -/
theorem part1 (h : tangent_line 1 0) (h1 : ∀ x, (0 : ℝ) < x → f 1 a b = 0) : a = 1 ∧ b = 0 := sorry

/- Part 2 -/
noncomputable def f_max := (fun x : ℝ => (Real.log x / x) : ℝ → ℝ)
theorem part2 (h : ∀ x, (0 : ℝ) < x → f_max x <= 1 / Real.exp 1) : f_max (Real.exp 1) = 1 / Real.exp 1 := sorry

/- Part 3 -/
theorem part3 (h1 : ∃ x, (0 : ℝ) < x ∧ x ≠ 1 ∧ g x c = 0) : c ≤ Real.exp (1 / Real.exp 1) := sorry

end part1_part2_part3_l422_422423


namespace Karen_wall_paint_area_l422_422271

theorem Karen_wall_paint_area :
  let height_wall := 10
  let width_wall := 15
  let height_window := 3
  let width_window := 5
  let height_door := 2
  let width_door := 6
  let area_wall := height_wall * width_wall
  let area_window := height_window * width_window
  let area_door := height_door * width_door
  let area_to_paint := area_wall - area_window - area_door
  area_to_paint = 123 := by
{
  sorry
}

end Karen_wall_paint_area_l422_422271


namespace x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l422_422821

theorem x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842
  (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l422_422821


namespace cos_angle_PMC_eq_zero_l422_422347

noncomputable theory
open_locale real

variables {A B C P M : Type} [real_inner_product_space ℝ M] [real_inner_product_space ℝ P]
variables (triangle_ABC : triangle ℝ A B C)
variables (M : M) (P : M)
variables (is_midpoint : midpoint ℝ B C M)
variables (perpendicular_AP_BC : ∃ (P : M), P = A ∧ is_perpendicular (AP : line ℝ) (BC : line ℝ) ∧ (intersection_point AP BC M))

theorem cos_angle_PMC_eq_zero
  (h_triangle : triangle_ABC)
  (h_midpoint : is_midpoint)
  (h_perpendicular : perpendicular_AP_BC) :
  ∃ (cos_angle_PMC : ℝ), cos_angle_PMC = 0 := sorry

end cos_angle_PMC_eq_zero_l422_422347


namespace find_k_m_n_sum_eq_39_l422_422056

theorem find_k_m_n_sum_eq_39 (t : ℝ) (k m n : ℕ) (h1 : (1 + real.sin t) * (1 + real.cos t) = 9 / 4)
  (h2 : (1 - real.sin t) * (1 - real.cos t) = m / n - real.sqrt k) 
  (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) 
  (hrel_prime : nat.coprime m n) : 
  k + m + n = 39 :=
sorry

end find_k_m_n_sum_eq_39_l422_422056


namespace find_t_l422_422272

theorem find_t (s t : ℤ) (h1 : 12 * s + 7 * t = 173) (h2 : s = t - 3) : t = 11 :=
by
  sorry

end find_t_l422_422272


namespace common_ratio_of_sequence_l422_422666

theorem common_ratio_of_sequence 
  (a1 a2 a3 a4 : ℤ)
  (h1 : a1 = 25)
  (h2 : a2 = -50)
  (h3 : a3 = 100)
  (h4 : a4 = -200)
  (is_geometric : ∀ (i : ℕ), a1 * (-2) ^ i = if i = 0 then a1 else if i = 1 then a2 else if i = 2 then a3 else a4) : 
  (-50 / 25 = -2) ∧ (100 / -50 = -2) ∧ (-200 / 100 = -2) :=
by 
  sorry

end common_ratio_of_sequence_l422_422666


namespace area_grazing_horse_approximate_area_grazing_horse_l422_422630

def area_of_grazing (r : ℝ) : ℝ := (1/4 * Real.pi * r^2)

theorem area_grazing_horse : 
  area_of_grazing 16 = 64 * Real.pi := 
by 
  -- Direct calculation
  sorry

theorem approximate_area_grazing_horse : 
  abs (area_of_grazing 16 - 201.06) < 0.001 := 
by 
  -- Numerical approximation
  sorry

end area_grazing_horse_approximate_area_grazing_horse_l422_422630


namespace largest_prime_factor_147_l422_422594

theorem largest_prime_factor_147 : 
  ∃ p : ℕ, p.prime ∧ p ∈ prime_factors 147 ∧ ∀ q ∈ prime_factors 147, q ≤ p := 
sorry

end largest_prime_factor_147_l422_422594


namespace sasha_added_num_l422_422328

theorem sasha_added_num (a b c : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a / b = 5 * (a + c) / (b * c)) : c = 6 ∨ c = -20 := 
sorry

end sasha_added_num_l422_422328


namespace find_days_l422_422466

theorem find_days
  (wages1 : ℕ) (workers1 : ℕ) (days1 : ℕ)
  (wages2 : ℕ) (workers2 : ℕ) (days2 : ℕ)
  (h1 : wages1 = 9450) (h2 : workers1 = 15) (h3 : wages2 = 9975)
  (h4 : workers2 = 19) (h5 : days2 = 5) :
  days1 = 6 := 
by
  -- Insert proof here
  sorry

end find_days_l422_422466


namespace triangle_equalities_l422_422176

variables {a b c r R S p : ℝ}

-- Definition of the semiperimeter
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Prove the given equalities under the given conditions
theorem triangle_equalities (h1 : S = p * r)
                            (h2 : S = (a * b * c) / (4 * R))
                            (h3 : p = semiperimeter a b c) :
  (a * b * c = 4 * p * r * R) ∧ (a * b + b * c + c * a = r^2 + p^2 + 4 * r * R) :=
by
  sorry

end triangle_equalities_l422_422176


namespace convert_vulgar_fraction_l422_422632

theorem convert_vulgar_fraction (h : (35 : ℚ) / 100 = 7 / 20) : (0.35 : ℚ) = 7 / 20 := 
by 
  have h1 : (0.35 : ℚ) = 35 / 100 := by norm_num
  rw h1
  exact h

end convert_vulgar_fraction_l422_422632


namespace tom_initial_books_l422_422964

theorem tom_initial_books (B : ℕ) (h1 : B - 4 + 38 = 39) : B = 5 :=
by
  sorry

end tom_initial_books_l422_422964


namespace distinct_values_f_in_interval_l422_422205

noncomputable def f (x : ℝ) : ℤ :=
  ⌊x⌋ + ⌊2 * x⌋ + ⌊(5 * x) / 3⌋ + ⌊3 * x⌋ + ⌊4 * x⌋

theorem distinct_values_f_in_interval : 
  ∃ n : ℕ, n = 734 ∧ 
    ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 100 → 
      f x = f y → x = y :=
sorry

end distinct_values_f_in_interval_l422_422205


namespace interest_rate_calc_l422_422259

theorem interest_rate_calc
  (P : ℝ) (A : ℝ) (T : ℝ) (SI : ℝ := A - P)
  (R : ℝ := (SI * 100) / (P * T))
  (hP : P = 750)
  (hA : A = 950)
  (hT : T = 5) :
  R = 5.33 :=
by
  sorry

end interest_rate_calc_l422_422259


namespace find_angle_BAE_l422_422759

open Real EuclideanGeometry

variables (ABC : Triangle) (E D : Point)
variables (hE : E ∈ Interior (Triangle.vertices ABC))
variables (hD : Collinear [ABC.B, E, D] ∧ Between (ABC.B, E, D))
variables (h1 : Angle D C E = 10) -- angle DCE = 10 degrees
variables (h2 : Angle D B C = 30) -- angle DBC = 30 degrees
variables (h3 : Angle E C B = 20) -- angle ECB = 20 degrees
variables (h4 : Angle A B D = 40) -- angle ABD = 40 degrees

-- Prove that angle BAE = 60 degrees
theorem find_angle_BAE : Angle B A E = 60 :=
by sorry

end find_angle_BAE_l422_422759


namespace tangent_line_at_zero_decreasing_intervals_l422_422426

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem tangent_line_at_zero :
  let t : ℝ × ℝ := (0, f 0)
  (∀ x : ℝ, (9 * x - f x - 2 = 0) → t.snd = -2) := by
  sorry

theorem decreasing_intervals :
  ∀ x : ℝ, (-3 * x^2 + 6 * x + 9 < 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

end tangent_line_at_zero_decreasing_intervals_l422_422426


namespace adam_and_simon_65_miles_apart_l422_422327

theorem adam_and_simon_65_miles_apart :
  ∀ (x : ℝ), (x = 5) →
  let adam_distance := 5 * x in
  let simon_distance := 12 * x in
  ∃ d : ℝ, d = 65 ∧ sqrt (adam_distance^2 + simon_distance^2) = d :=
by
  intros x hx
  rw hx
  let adam_distance := 5 * 5
  let simon_distance := 12 * 5
  use 65
  split
  { refl }
  rw [pow_two, pow_two]
  simp [adam_distance, simon_distance]
  sorry -- proof omitted

end adam_and_simon_65_miles_apart_l422_422327


namespace martha_crayons_total_l422_422897

theorem martha_crayons_total :
  let initial_crayons := 18 in
  let lost_crayons := initial_crayons / 2 in
  let remaining_crayons := initial_crayons - lost_crayons in
  let crayons_after_first_purchase := remaining_crayons + 20 in
  let crayons_after_contest_win := crayons_after_first_purchase + 15 in
  let final_crayons := crayons_after_contest_win + 25 in
  final_crayons = 69 := by
  sorry

end martha_crayons_total_l422_422897


namespace problem_solution_l422_422764

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : ℝ) : Prop :=
  eccentricity = 1 / 3 → 
  let e := real.sqrt (a^2 - b^2) / a in
  e = eccentricity → (a = 3 ∧ b = 2 * real.sqrt 2 → 
  (∀ A1 A2 B : ℝ × ℝ,
    A1 = (-3, 0) ∧
    A2 = (3, 0) ∧
    B = (0, 2 * real.sqrt 2) →
    let BA1 := (B.1 - A1.1, B.2 - A1.2) in
    let BA2 := (B.1 - A2.1, B.2 - A2.2) in
    BA1.1 * BA2.1 + BA1.2 * BA2.2 = -1 →
    (∀ x y : ℝ, (x / 3)^2 + (y / (2 * real.sqrt 2))^2 = 1)))

theorem problem_solution : ellipse_equation 3 (2 * real.sqrt 2) (by norm_num) (by norm_num) (1/3) :=
sorry

end problem_solution_l422_422764


namespace neg_univ_prop_l422_422214

-- Translate the original mathematical statement to a Lean 4 statement.
theorem neg_univ_prop :
  (¬(∀ x : ℝ, x^2 ≠ x)) ↔ (∃ x : ℝ, x^2 = x) :=
by
  sorry

end neg_univ_prop_l422_422214


namespace combined_work_days_l422_422254

def rate_a (W : ℝ) : ℝ := W / 14
def rate_b (W : ℝ) : ℝ := W / 10.5
def combined_rate (W D : ℝ) : ℝ := W / D

theorem combined_work_days (W : ℝ) : (1 / 14 + 1 / 10.5) = 1 / 6 :=
by
  sorry

end combined_work_days_l422_422254


namespace Keiko_speed_is_pi_div_3_l422_422587

noncomputable def Keiko_avg_speed {r : ℝ} (v : ℝ → ℝ) (pi : ℝ) : ℝ :=
let C1 := 2 * pi * (r + 6) - 2 * pi * r
let t1 := 36
let v1 := C1 / t1

let C2 := 2 * pi * (r + 8) - 2 * pi * r
let t2 := 48
let v2 := C2 / t2

if v r = v1 ∧ v r = v2 then (v1 + v2) / 2 else 0

theorem Keiko_speed_is_pi_div_3 (pi : ℝ) (r : ℝ) (v : ℝ → ℝ) :
  v r = π / 3 ∧ (forall t1 t2 C1 C2, C1 / t1 = π / 3 ∧ C2 / t2 = π / 3 → 
  (C1/t1 + C2/t2)/2 = π / 3) :=
sorry

end Keiko_speed_is_pi_div_3_l422_422587


namespace number_of_space_diagonals_l422_422656

theorem number_of_space_diagonals
  (V E F T Q : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 42)
  (hT : T = 30)
  (hQ : Q = 12):
  (V * (V - 1) / 2 - E - 2 * Q) = 341 :=
by
  sorry

end number_of_space_diagonals_l422_422656


namespace complex_number_proof_l422_422553

-- Define the conditions and prove the proposition
theorem complex_number_proof (a b : ℝ) (i : ℂ) (hi : i = complex.I) (h : (a + 2 * i) / i = b - i) : a + b = 3 :=
by
  sorry -- Proof to be completed

end complex_number_proof_l422_422553


namespace sqrt_sum_4_pow_4_eq_32_l422_422608

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l422_422608


namespace decreasing_intervals_l422_422206

def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem decreasing_intervals :
  ∀ (k : ℤ), ∀ (x : ℝ), (Real.pi / 12 + k * Real.pi) ≤ x ∧ x ≤ (7 * Real.pi / 12 + k * Real.pi) → 
  g x = g (2 * x + Real.pi / 3) :=
begin
  sorry
end

end decreasing_intervals_l422_422206


namespace find_f_sum_l422_422570

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom f_at_one : f 1 = 9

theorem find_f_sum :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end find_f_sum_l422_422570


namespace sqrt_four_four_summed_l422_422605

theorem sqrt_four_four_summed :
  sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := by
  sorry

end sqrt_four_four_summed_l422_422605


namespace event_occurs_with_high_probability_l422_422455

theorem event_occurs_with_high_probability 
    (ε : ℝ) (hε : ε > 0) :
    ∀ (n : ℕ), ∃ (k : ℕ), k ≥ n → (P (λ ω, ∃ m ≥ k, A m) = 1) :=
sorry

end event_occurs_with_high_probability_l422_422455


namespace sqrt_sum_of_powers_l422_422596

theorem sqrt_sum_of_powers : sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  have h : 4^4 = 256 := by
    calc
      4^4 = 4 * 4 * 4 * 4 := by rw [show 4^4 = 4 * 4 * 4 * 4 from rfl]
      ... = 16 * 16       := by rw [mul_assoc, mul_assoc]
      ... = 256           := by norm_num
  calc
    sqrt (4^4 + 4^4 + 4^4 + 4^4)
      = sqrt (256 + 256 + 256 + 256) := by rw [h, h, h, h]
      ... = sqrt 1024                 := by norm_num
      ... = 32                        := by norm_num

end sqrt_sum_of_powers_l422_422596


namespace function_odd_and_decreasing_l422_422001

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Definition of a decreasing function on the interval (0, +∞)
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x1 x2 : ℝ⦄, x1 ∈ I → x2 ∈ I → x1 < x2 → f x1 ≥ f x2

-- Define the function y = 1 / x
def f (x : ℝ) : ℝ := 1 / x

-- Define the interval (0, +∞)
def interval : Set ℝ := { x : ℝ | 0 < x }

-- Lean statement to prove the function y = 1 / x is odd and decreasing on (0, +∞)
theorem function_odd_and_decreasing :
  is_odd_function f ∧ is_decreasing_on f interval :=
by
  sorry

end function_odd_and_decreasing_l422_422001


namespace locus_centers_tangent_circles_l422_422558

theorem locus_centers_tangent_circles (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (3 - r)^2) →
  a^2 - 12 * a + 4 * b^2 = 0 :=
by
  sorry

end locus_centers_tangent_circles_l422_422558


namespace equal_area_division_l422_422236

theorem equal_area_division (A B C D E : Point) (h : LineSegment B C)
(h_divide : ∃ (D E : Point), Collinear B D C ∧ Collinear D E C ∧ Collinear E C B ∧
SegmentsEqual (LineSegment B D) (LineSegment D E) ∧ SegmentsEqual (LineSegment D E) (LineSegment E C)) 
(h_lines : LinesIntersectingAtVertices A D ∧ LinesIntersectingAtVertices A E) :
  AreaOfTriangle (Triangle A B D) = AreaOfTriangle (Triangle A D E) ∧
  AreaOfTriangle (Triangle A D E) = AreaOfTriangle (Triangle A E C) :=
sorry

end equal_area_division_l422_422236


namespace find_a8_a12_sum_l422_422773

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a8_a12_sum
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end find_a8_a12_sum_l422_422773


namespace find_2a_plus_b_l422_422189

open Function

noncomputable def f (a b : ℝ) (x : ℝ) := 2 * a * x - 3 * b
noncomputable def g (x : ℝ) := 5 * x + 4
noncomputable def h (a b : ℝ) (x : ℝ) := g (f a b x)
noncomputable def h_inv (x : ℝ) := 2 * x - 9

theorem find_2a_plus_b (a b : ℝ) (h_comp_inv_eq_id : ∀ x, h a b (h_inv x) = x) :
  2 * a + b = 1 / 15 := 
sorry

end find_2a_plus_b_l422_422189


namespace min_value_f_a_eq_4_f_increasing_a_in_0_1_l422_422751

open Real

noncomputable def f (x : ℝ) (a : ℝ) := x + a / (x + 1)

-- Problem (1) minimum value of f(x) when a = 4
theorem min_value_f_a_eq_4 : ∃ x ∈ Ici (0 : ℝ), f x 4 = 3 :=
by sorry

-- Problem (2) monotonicity and minimum value of f(x) when a ∈ (0, 1)
theorem f_increasing_a_in_0_1 (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x1 x2 ∈ Ici (0 : ℝ), x1 < x2 → f x1 a < f x2 a) ∧
  (∃ x ∈ Ici (0 : ℝ), f x a = a) :=
by sorry

end min_value_f_a_eq_4_f_increasing_a_in_0_1_l422_422751


namespace total_measurable_weights_l422_422985

-- Define the weights
def weights : List ℕ := [1, 2, 6, 26]

-- statement that a total of 28 different weights can be measured
theorem total_measurable_weights : 
    (card (Finset.filter (λ x => ∃ l1 l2 : List ℕ, List.sublist l1 weights ∧ List.sublist l2 weights ∧ x = (l1.sum - l2.sum) ∧ l1.sum ≠ l2.sum) (Finset.range 36))) = 28 :=
sorry

end total_measurable_weights_l422_422985


namespace largest_good_number_is_99_number_of_absolute_good_numbers_is_39_l422_422825

-- Definition of good number
def is_good_number (N a b : ℕ) : Prop :=
  10 ≤ N ∧ N ≤ 99 ∧ N = a * b + a + b

-- Definition of absolute good number
def is_absolute_good_number (N a b : ℕ) : Prop :=
  is_good_number N a b ∧ (ab_div_sum_eq_3 a b ∨ ab_div_sum_eq_4 a b)

def ab_div_sum_eq_3 (a b : ℕ) : Prop :=
  a * b = 3 * (a + b)

def ab_div_sum_eq_4 (a b : ℕ) : Prop :=
  a * b = 4 * (a + b)

-- The largest good number is 99
theorem largest_good_number_is_99 : ∃ N, (∃ a b, is_good_number N a b) ∧ N = 99 := 
  sorry

-- The number of absolute good numbers is 39
theorem number_of_absolute_good_numbers_is_39 : ∃ n, (n = 39 ∧ 
  ∃ N a b, is_absolute_good_number N a b) := 
  sorry

end largest_good_number_is_99_number_of_absolute_good_numbers_is_39_l422_422825


namespace laborer_saved_money_l422_422556

theorem laborer_saved_money (
  avg_expenditure_first_6_months : ℕ := 75,
  expenditure_next_4_months_per_mo : ℕ := 60,
  monthly_income : ℕ := 72,
  debt_cleared_off : ℕ := 18,
  total_expenditure_first_6_months : ℕ := 450,
  total_income_first_6_months : ℕ := 432,
  total_expenditure_next_4_months : ℕ := 240,
  total_income_next_4_months : ℕ := 288,
) : ∃ saved_money : ℕ, saved_money = 30 :=
by
  let debt := total_expenditure_first_6_months - total_income_first_6_months
  let total_after_4_months := total_income_next_4_months - (total_expenditure_next_4_months + debt)
  use total_after_4_months
  sorry

end laborer_saved_money_l422_422556


namespace general_term_b_sum_first_n_terms_l422_422862

def seq_a : ℕ → ℝ
| 0       => 0   -- we use a placeholder for a0 to align indices (a₁ = 1)
| 1       => 1
| (n + 2) => (1 + 1 / (n + 1)) * seq_a (n + 1) + (n + 2) / 2 ^ (n + 1)

def seq_b (n : ℕ) : ℝ := seq_a (n + 1) / (n + 1)

theorem general_term_b (n : ℕ) (h : 0 < n) : seq_b n = 2 - 1 / 2 ^ n := 
sorry

def sum_a (n : ℕ) : ℝ := ∑ i in finset.range n, seq_a (i + 1)

theorem sum_first_n_terms (n : ℕ) : sum_a n = n * (n + 1) + (n + 2) / 2 ^ (n - 1) - 4 :=
sorry

end general_term_b_sum_first_n_terms_l422_422862


namespace a_2023_eq_5_2022_l422_422057

noncomputable def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ := (finset.range n).sum a

def a_seq : ℕ → ℕ
| 0     := 1
| (n+1) := 4 * S_n a_seq (n + 1) + 1

theorem a_2023_eq_5_2022 : a_seq 2023 = 5 ^ 2022 :=
by
  sorry

end a_2023_eq_5_2022_l422_422057


namespace sqrt_four_four_summed_l422_422602

theorem sqrt_four_four_summed :
  sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := by
  sorry

end sqrt_four_four_summed_l422_422602


namespace isosceles_triangle_side_l422_422832

theorem isosceles_triangle_side (a : ℝ) : 
  (10 - a = 7 ∨ 10 - a = 6) ↔ (a = 3 ∨ a = 4) := 
by sorry

end isosceles_triangle_side_l422_422832


namespace carriage_problem_l422_422116

theorem carriage_problem (x : ℕ) : 
  3 * (x - 2) = 2 * x + 9 := 
sorry

end carriage_problem_l422_422116


namespace circumcircle_radius_locus_of_centers_l422_422967

variables {P Q A B C O1 O2 O3 : Type*}
variables [metric_space P] [metric_space Q]
variables [metric_space O1] [metric_space O2]
variables [metric_space A] [metric_space B] [metric_space C]

-- Define the distance between the centers of circles O1 and O2
def distance (O1 O2 : Type*) : ℝ := d

-- Define A, B, C being on the respective circles and intersections
def point_on_first_circle (A : Type*) : Prop := ∀ (A : Type*), A ≠ P ∧ A ≠ Q
def intersect_point (A B : Type*) (O2 : Type*) : Prop := ¬ (A = P ∧ B = Q)

-- Define P and Q being the intersection points of the two circles
def intersect_circles (O1 O2 P Q : Type*) : Prop := 
eq (distance O1 O2) d ∧ eq (point_on_first_circle A) true ∧ eq (intersect_point B C O2) true

-- Question reconverted into a proof structure
theorem circumcircle_radius (O1 O2 O3 : Type*) [metric_space O3] : 
(intersect_circles O1 O2 P Q) → 
(O3 = O1 + O2) → 
(distance A B = d) :=
sorry

-- Movement of point A and locus description for centers O3 of ∆ABC
theorem locus_of_centers (O1 O2 O3 : Type*) [metric_space O3] :
(intersect_circles O1 O2 P Q) → 
(point_on_first_circle A) → 
∃ locus, (O3.moves_as A.moves O1 O2) :=
sorry

end circumcircle_radius_locus_of_centers_l422_422967


namespace trajectory_ellipse_l422_422535

noncomputable def trajectory_of_P (P : ℝ × ℝ) : Prop := 
  let M := (-3, 0) in
  let N := (3, 0) in
  let PM := ∥P - M∥ in
  let PN := ∥P - N∥ in
  PM = 10 - PN

theorem trajectory_ellipse :
  ∀ P : ℝ × ℝ, trajectory_of_P P ↔ (P.1^2) / 25 + (P.2^2) / 16 = 1 := 
by
  intro P
  sorry

end trajectory_ellipse_l422_422535


namespace hypotenuse_length_2a_l422_422910

theorem hypotenuse_length_2a (a : ℝ) :
  let A := (a, -a^2)
  let B := (-a, -a^2)
  let O := (0, 0)
  in A.2 = -(A.1 ^ 2) ∧ B.2 = -(B.1 ^ 2) ∧
     (A.1, A.2) ≠ O ∧ (B.1, B.2) ≠ O ∧ (O.1, O.2) = (0, 0) →
     let OA_sq := (A.1 ^ 2 + A.2 ^ 2)
     let AB_sq := ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)
     in real.sqrt AB_sq = 2 * real.sqrt OA_sq := by
  sorry

end hypotenuse_length_2a_l422_422910


namespace find_digit_to_make_divisible_by_seven_l422_422995

/-- 
  Given a number formed by concatenating 2023 digits of 6 with 2023 digits of 5.
  In a three-digit number 6*5, find the digit * to make this number divisible by 7.
  i.e., We must find the digit x such that the number 600 + 10x + 5 is divisible by 7.
-/
theorem find_digit_to_make_divisible_by_seven :
  ∃ x : ℕ, x < 10 ∧ (600 + 10 * x + 5) % 7 = 0 :=
sorry

end find_digit_to_make_divisible_by_seven_l422_422995


namespace ternary_minimum_value_1_over_a_plus_1_over_b_l422_422830

noncomputable def minimum_value_1_over_a_plus_1_over_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
(h3 : ∃ x y, (2 * a * x - b * y + 2 = 0)
∧ (x^2 + y^2 + 2 * x - 4 * y + 1 = 0)
∧ x = -1 ∧ y = 2) : ℝ :=
  ∃ H : a + b = 1, 4

theorem ternary_minimum_value_1_over_a_plus_1_over_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ x y, (2 * a * x - b * y + 2 = 0) 
  ∧ (x^2 + y^2 + 2 * x - 4 * y + 1 = 0) 
  ∧ x = -1 ∧ y = 2) : minimum_value_1_over_a_plus_1_over_b a b h1 h2 h3 = 4 :=
sorry

end ternary_minimum_value_1_over_a_plus_1_over_b_l422_422830


namespace part1_part2_l422_422396

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions
axiom H1 : ∀ n, S n = ∑ i in finset.range n, a i
axiom H2 : ∀ n, 4 * S n = (a n) ^ 2 + 2 * (a n) - 3
axiom H3 : ∀ n, a n > 0

-- Proof goal for part 1
theorem part1 : ∀ n, a n = 2 * n + 1 :=
sorry

-- Definitions for part 2
def b (n : ℕ) : ℝ := 1 / ((a n) ^ 2 - 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Proof goal for part 2
theorem part2 : ∀ n, T n = n / (4 * n + 4) :=
sorry

end part1_part2_l422_422396


namespace graph_does_not_pass_second_quadrant_l422_422451

noncomputable def y_function (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) : 
  ∀ x y : ℝ, (y = y_function a b x) → ¬(x < 0 ∧ y > 0) := by
  sorry

end graph_does_not_pass_second_quadrant_l422_422451


namespace find_k_l422_422082

open Real

def vector (α β : ℝ) : ℝ × ℝ := (α, β)

def is_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem find_k
  (a b : ℝ × ℝ)
  (ha : a = (1, 2))
  (hb : b = (-3, 2)) :
  is_parallel (k * a.1 + b.1, k * a.2 + b.2) (a.1 - 3 * b.1, a.2 - 3 * b.2) →
  k = -1/3 :=
begin
  sorry
end

end find_k_l422_422082


namespace part_I_part_II_l422_422795

variable (k m : ℝ)

def f (x : ℝ) : ℝ := k * x^2 + (3 + k) * x + 3

def g (x : ℝ) : ℝ := f x - m * x

theorem part_I {k : ℝ} (hk : k ≠ 0) (h_f_2 : f k 2 = 3) : 
  f k = -x^2 + 2x + 3 := 
by
  sorry

theorem part_II {k : ℝ} (hk : k ≠ 0) (h_f_2 : f k 2 = 3) (h_g_monotone : ∀ x ∈ set.Icc (-2:ℝ) (2:ℝ), ∀ y ∈ set.Icc (-2:ℝ) (2:ℝ), x ≤ y → g k m x ≤ g k m y) : 
  m ∈ set.Ici (6:ℝ) ∪ set.Iic (-2:ℝ) := 
by
  sorry

end part_I_part_II_l422_422795


namespace pen_defectiveness_l422_422835

theorem pen_defectiveness (N D : ℕ) (h1 : N + D = 16)
  (h2 : (N / 16) * ((N - 1) / 15) = 0.65) : D = 3 :=
sorry

end pen_defectiveness_l422_422835


namespace sqrt_sum_4_pow_4_eq_32_l422_422610

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l422_422610


namespace find_angle_A_l422_422478

-- Conditions
def is_triangle (A B C : ℝ) : Prop := A + B + C = 180
def B_is_two_C (B C : ℝ) : Prop := B = 2 * C
def B_is_80 (B : ℝ) : Prop := B = 80

-- Theorem statement
theorem find_angle_A (A B C : ℝ) (h₁ : is_triangle A B C) (h₂ : B_is_two_C B C) (h₃ : B_is_80 B) : A = 60 := by
  sorry

end find_angle_A_l422_422478


namespace probability_at_least_4_girls_l422_422869

theorem probability_at_least_4_girls (h : ∀ i ∈ Finset.range 6, 0.5) : 
  (Finset.sum (Finset.range 3) 
    (λ k, Classical.choose (Nat.choose 6 (4 + k)))) / (2^6) = 11 / 32 :=
by
  sorry

end probability_at_least_4_girls_l422_422869


namespace all_elements_rational_l422_422873

open Set

def finite_set_in_interval (n : ℕ) : Set ℝ :=
  {x | ∃ i, i ∈ Finset.range (n + 1) ∧ (x = 0 ∨ x = 1 ∨ 0 < x ∧ x < 1)}

def unique_distance_condition (S : Set ℝ) : Prop :=
  ∀ d, d ≠ 1 → ∃ x_i x_j x_k x_l, x_i ∈ S ∧ x_j ∈ S ∧ x_k ∈ S ∧ x_l ∈ S ∧ 
        abs (x_i - x_j) = d ∧ abs (x_k - x_l) = d ∧ (x_i = x_k → x_j ≠ x_l)

theorem all_elements_rational
  (n : ℕ)
  (S : Set ℝ)
  (hS1 : ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1)
  (hS2 : 0 ∈ S)
  (hS3 : 1 ∈ S)
  (hS4 : unique_distance_condition S) :
  ∀ x ∈ S, ∃ q : ℚ, (x : ℝ) = q := 
sorry

end all_elements_rational_l422_422873


namespace ratio_of_smaller_circle_to_larger_circle_l422_422975

section circles

variables {Q : Type} (C1 C2 : ℝ) (angle1 : ℝ) (angle2 : ℝ)

def ratio_of_areas (C1 C2 : ℝ) : ℝ := (C1 / C2)^2

theorem ratio_of_smaller_circle_to_larger_circle
  (h1 : angle1 = 60)
  (h2 : angle2 = 48)
  (h3 : (angle1 / 360) * C1 = (angle2 / 360) * C2) :
  ratio_of_areas C1 C2 = 16 / 25 :=
by
  sorry

end circles

end ratio_of_smaller_circle_to_larger_circle_l422_422975


namespace distance_from_center_to_line_l422_422432

-- Define the conditions 
def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin θ + 2 * ρ * Real.cos θ = 1

-- Define the assertion that we want to prove
theorem distance_from_center_to_line (ρ θ : ℝ) 
  (h_circle: circle_polar_eq ρ θ) 
  (h_line: line_polar_eq ρ θ) : 
  ∃ d : ℝ, d = (Real.sqrt 5) / 5 := 
sorry

end distance_from_center_to_line_l422_422432


namespace find_x_solution_l422_422906

theorem find_x_solution :
  ∃ x : ℝ, (7 / 4) * x = 63 ∧ x = 36 :=
by
  use 36
  split
  · norm_num
  · norm_num
  sorry

end find_x_solution_l422_422906


namespace least_number_to_multiply_l422_422211

theorem least_number_to_multiply (x : ℕ) :
  (72 * x) % 112 = 0 → x = 14 :=
by 
  sorry

end least_number_to_multiply_l422_422211


namespace number_of_valid_dropped_class_combinations_is_12_l422_422504

-- Definitions of classes and their durations
def duration (class : String) : Real :=
  if class = "A1" ∨ class = "A2" ∨ class = "A3" then 2 else
  if class = "B1" ∨ class = "B2" then 1.5 else
  if class = "C" then 2.5 else 0

-- Total hours before dropping any classes
def total_hours : Real :=
  3 * duration "A1" + 2 * duration "B1" + duration "C"

-- Helper function to calculate the total hours after dropping two classes
def total_hours_after_drop (drop1 drop2 : String) : Real :=
  total_hours - duration drop1 - duration drop2

-- The set of all classes
def classes : List String := ["A1", "A2", "A3", "B1", "B2", "C"]

-- Valid combinations of dropped classes according to the conditions specified
def valid_combinations : List (String × String) :=
  List.filter (λ (d : String × String),
    let (drop1, drop2) := d in
    drop1 ≠ drop2 ∧
    ¬((drop1 = "B1" ∧ drop2 = "B2") ∨ (drop1 = "B2" ∧ drop2 = "B1")) ∧
    ¬(drop1 = "C" ∧ drop2 ∉ ["A1", "A2", "A3"]) ∧
    ¬(drop2 = "C" ∧ drop1 ∉ ["A1", "A2", "A3"])
  ) (classes.product classes)

-- The formal Lean statement of the problem
theorem number_of_valid_dropped_class_combinations_is_12 :
  valid_combinations.length = 12 := by
    -- Proof would go here
    sorry

end number_of_valid_dropped_class_combinations_is_12_l422_422504


namespace max_height_of_table_l422_422129

noncomputable def max_table_height (a b c : ℝ) (h : ℝ) : Prop :=
  let s := (a + b + c) / 2
  let area := real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h_a := (2 * area) / a
  let h_b := (2 * area) / b
  let h_c := (2 * area) / c
  h <= (h_a * h_c) / (h_a + h_c)

theorem max_height_of_table :
  ∀ (a b c : ℝ) (h : ℝ), a = 25 → b = 32 → c = 29 →
  max_table_height a b c h →
  h = 84 * real.sqrt 1547 / 57 ∧ 84 + 1547 + 57 = 1688 :=
by {
  intros,
  sorry
}

end max_height_of_table_l422_422129


namespace max_product_at_n_12_l422_422564

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n-1)

noncomputable def product_first_n_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  ∏ i in Finset.range n, geometric_sequence a₁ q (i + 1)

theorem max_product_at_n_12 :
  ∀ (n : ℕ), (product_first_n_terms 1536 (-1/2) 12) ≥ (product_first_n_terms 1536 (-1/2) n) :=
begin
  -- proof goes here
  sorry
end

end max_product_at_n_12_l422_422564


namespace chlorine_moles_l422_422740

theorem chlorine_moles (methane_used chlorine_used chloromethane_formed : ℕ)
  (h_combined_methane : methane_used = 3)
  (h_formed_chloromethane : chloromethane_formed = 3)
  (balanced_eq : methane_used = chloromethane_formed) :
  chlorine_used = 3 :=
by
  have h : chlorine_used = methane_used := by sorry
  rw [h_combined_methane] at h
  exact h

end chlorine_moles_l422_422740


namespace max_regions_quadratic_trinomials_l422_422499

theorem max_regions_quadratic_trinomials (a b c : Fin 100 → ℝ) :
  ∃ R, (∀ (n : ℕ), n ≤ 100 → R = n^2 + 1) → R = 10001 := 
  sorry

end max_regions_quadratic_trinomials_l422_422499


namespace percent_game_of_thrones_altered_l422_422028

def votes_game_of_thrones : ℕ := 10
def votes_twilight : ℕ := 12
def votes_art_of_deal : ℕ := 20

def altered_votes_art_of_deal : ℕ := votes_art_of_deal - (votes_art_of_deal * 80 / 100)
def altered_votes_twilight : ℕ := votes_twilight / 2
def total_altered_votes : ℕ := altered_votes_art_of_deal + altered_votes_twilight + votes_game_of_thrones

theorem percent_game_of_thrones_altered :
  ((votes_game_of_thrones * 100) / total_altered_votes) = 50 := by
  sorry

end percent_game_of_thrones_altered_l422_422028


namespace false_propositions_l422_422698

theorem false_propositions : (Prop1 false) ∧ (Prop2 false) ∧ (Prop3 true) ∧ (Prop4 false) :=
begin
  -- Definitions of the propositions according to the conditions
  let Prop1 := ∀ (l1 l2 l3 : Line), intersect l3 l1 → intersect l3 l2 → ¬Parallel l1 l2 → EqualCorrespondingAngles l1 l2 l3,
  let Prop2 := ∀ (p : Point) (l : Line), ¬(∃! m : Line, through p m ∧ Parallel m l)
  let Prop3 := ∀ (angle1 angle2 : Angle), angle1 = 40° ∧ Parallel (side1 angle1) (side1 angle2) ∧ Parallel (side2 angle1) (side2 angle2) → (angle2 = 40° ∨ angle2 = 140°)
  let Prop4 := ∀ (a b c : Line), Perpendicular b c ∧ Perpendicular a c → Parallel b a,
  
  sorry
end

end false_propositions_l422_422698


namespace cheetah_speed_l422_422659

theorem cheetah_speed (v : ℝ) : 
  let deer_speed := 50 in let deer_head_start := 2 / 60 in let time_to_catch := 1 / 60 in
  (deer_speed * (deer_head_start + time_to_catch)) = (time_to_catch * v) → v = 150 :=
by
  let deer_speed := 50
  let deer_head_start := 2 / 60
  let time_to_catch := 1 / 60
  intro h
  sorry

end cheetah_speed_l422_422659


namespace hexagon_cosines_identity_l422_422641

theorem hexagon_cosines_identity (ABCDEF : Type) [InscribedHexagon ABCDEF] 
  (hAB : length(AB) = 5) (hBC : length(BC) = 5) (hCD : length(CD) = 5) 
  (hDE : length(DE) = 5) (hEF : length(EF) = 5) (hFA : length(FA) = 5)
  (hAC : length(AC) = 2) :
  (1 - cos (angle B)) * (1 - cos (angle ACE)) = 2.25 :=
by 
  sorry

end hexagon_cosines_identity_l422_422641


namespace number_of_students_per_normal_class_l422_422438

theorem number_of_students_per_normal_class (total_students : ℕ) (percentage_moving : ℕ) (grade_levels : ℕ) (adv_class_size : ℕ) (additional_classes : ℕ) 
  (h1 : total_students = 1590) 
  (h2 : percentage_moving = 40) 
  (h3 : grade_levels = 3) 
  (h4 : adv_class_size = 20) 
  (h5 : additional_classes = 6) : 
  (total_students * percentage_moving / 100 / grade_levels - adv_class_size) / additional_classes = 32 :=
by
  sorry

end number_of_students_per_normal_class_l422_422438


namespace stratified_sampling_sum_l422_422304

theorem stratified_sampling_sum :
  let grains := 40
  let vegetable_oils := 10
  let animal_foods := 30
  let fruits_and_vegetables := 20
  let sample_size := 20
  let total_food_types := grains + vegetable_oils + animal_foods + fruits_and_vegetables
  let sampling_fraction := sample_size / total_food_types
  let number_drawn := sampling_fraction * (vegetable_oils + fruits_and_vegetables)
  number_drawn = 6 :=
by
  sorry

end stratified_sampling_sum_l422_422304


namespace martin_savings_correct_l422_422477

theorem martin_savings_correct :
  let coat_original := 100
  let pants_original := 50
  let coat_discount := 0.30
  let pants_discount := 0.60
  let total_original := coat_original + pants_original
  let coat_saving := coat_original * coat_discount
  let pants_saving := pants_original * pants_discount
  let total_saving := coat_saving + pants_saving
  let percent_saved := (total_saving / total_original) * 100
  in percent_saved = 40 := 
by {
  sorry
}

end martin_savings_correct_l422_422477


namespace range_of_slope_angle_l422_422467

theorem range_of_slope_angle (a b : ℝ) (h1 : 0 < a^2 + b^2) :
  (∃ p1 p2 p3 : ℝ × ℝ, (p1 ∈ {(x, y) | (x - 3) ^ 2 + (y - √3) ^ 2 = 24} ∧
                   p2 ∈ {(x, y) | (x - 3) ^ 2 + (y - √3) ^ 2 = 24} ∧
                   p3 ∈ {(x, y) | (x - 3) ^ 2 + (y - √3) ^ 2 = 24} ∧
                   ∀ p, p ∈ {p1, p2, p3} → distance_to_line p a b 0 = √6)) →
  (let θ := real.arctan (-a / b) in
   (0 ≤ θ ∧ θ ≤ 5*real.pi/12) ∨ (11*real.pi/12 ≤ θ ∧ θ < real.pi)) :=
sorry

end range_of_slope_angle_l422_422467


namespace side_length_of_square_l422_422928

theorem side_length_of_square {ABCDEF : Type} [Hexagon ABCDEF]
  (P Q R S : ABCDEF → ABCDEF → Type)
  (hPQRS : square PQRS)
  (hP_on_AB : P (side AB))
  (hQ_on_CD : Q (side CD))
  (hR_on_EF : R (side EF))
  (hAB : AB = 50)
  (hEF : EF = 35 * (√3 - 1)) :
  let side_length := 25 * √3 - 17
  sqrt PQRS.side_length = side_length :=
sorry

end side_length_of_square_l422_422928


namespace increasing_interval_and_symmetry_axis_find_m_from_max_l422_422424

noncomputable def f (x m : ℝ) := sin x ^ 2 + 2 * sqrt 3 * sin x * cos x + 3 * cos x ^ 2 + m

-- Proof Problem (I)
theorem increasing_interval_and_symmetry_axis (m : ℝ) : 
  ( ∀ k : ℤ, ∃ I : set ℝ, I = [-Real.pi/3 + k * pi, Real.pi/6 + k * pi] ∧ 
    (∀ x ∈ I, order (f x m) ≤ order (f x m)) ∧
    ∃ k' : ℤ, I = [k' * Real.pi + Real.pi / 6, k' * Real.pi + Real.pi / 6 + pi / 2]) :=
sorry

-- Proof Problem (II)
theorem find_m_from_max (m : ℝ) : 
  (∀ x ∈ [0, Real.pi / 3], f x m ≤ 9) ∧ (∃ max_val : ℝ, max_val = 9) → 
  m = 5 := 
sorry

end increasing_interval_and_symmetry_axis_find_m_from_max_l422_422424


namespace tangent_line_equation_l422_422561

theorem tangent_line_equation 
    (h_perpendicular : ∃ m1 m2 : ℝ, m1 * m2 = -1 ∧ (∀ y, x + m1 * y = 4) ∧ (x + 4 * y = 4)) 
    (h_tangent : ∀ x : ℝ, y = 2 * x ^ 2 ∧ (∀ y', y' = 4 * x)) :
    ∃ a b c : ℝ, (4 * a - b - c = 0) ∧ (∀ (t : ℝ), a * t + b * (2 * t ^ 2) = 1) :=
sorry

end tangent_line_equation_l422_422561


namespace categorize_numbers_l422_422363

def number1 := -(-7)
def number2 := -2.6
def number3 := -(Real.pi / 3)
def number4 := abs (-2)
def number5 := Real.sqrt 8
def number6 := -((1/2)^2)
def number7 := (-(1/3)^3)
def number8 := 0.3030030003 -- Eventually this should be more generalized

theorem categorize_numbers :
  ({number1, number4} = {7, 2}) ∧
  ({number2, number6, number7} = {-2.6, -(1/4), -(1/27)}) ∧
  (irrational number3 ∧ irrational number5 ∧ irrational number8) :=
by
  sorry

end categorize_numbers_l422_422363


namespace distinct_real_roots_find_p_l422_422066

theorem distinct_real_roots (p : ℝ) : 
  let f := (fun x => (x - 3) * (x - 2) - p^2)
  let Δ := 1 + 4 * p ^ 2 
  0 < Δ :=
by sorry

theorem find_p (x1 x2 p : ℝ) : 
  (x1 + x2 = 5) → 
  (x1 * x2 = 6 - p^2) → 
  (x1^2 + x2^2 = 3 * x1 * x2) → 
  (p = 1 ∨ p = -1) :=
by sorry

end distinct_real_roots_find_p_l422_422066


namespace cube_root_eval_l422_422730

noncomputable def cube_root_nested (N : ℝ) : ℝ := (N * (N * (N * (N)))) ^ (1/81)

theorem cube_root_eval (N : ℝ) (h : N > 1) : 
  cube_root_nested N = N ^ (40 / 81) := 
sorry

end cube_root_eval_l422_422730


namespace time_to_walk_l422_422961

variable (v l r w : ℝ)
variable (h1 : l = 15 * (v + r))
variable (h2 : l = 30 * (v + w))
variable (h3 : l = 20 * r)

theorem time_to_walk (h1 : l = 15 * (v + r)) (h2 : l = 30 * (v + w)) (h3 : l = 20 * r) : l / w = 60 := 
by sorry

end time_to_walk_l422_422961


namespace possible_values_of_N_l422_422117

theorem possible_values_of_N :
  ∃ N : ℕ, (∃ A x B : ℕ, N = 1000 * A + 100 * x + B ∧ 
                 10000 * A + 1000 * x + B = 9 * N
                 ∧ B ∈ {25, 50, 75} ∧ (x = 2 ∧ B = 25 ∨ x = 4 ∧ B = 50 ∨ x = 6 ∧ B = 75))
  → N ∈ {225, 450, 675} :=
begin
  sorry
end

end possible_values_of_N_l422_422117


namespace set_non_neg_even_set_primes_up_to_10_eq_sol_set_l422_422009

noncomputable def non_neg_even (x : ℕ) : Prop := x % 2 = 0 ∧ x ≤ 10
def primes_up_to_10 (x : ℕ) : Prop := Nat.Prime x ∧ x ≤ 10
def eq_sol (x : ℤ) : Prop := x^2 + 2*x - 15 = 0

theorem set_non_neg_even :
  {x : ℕ | non_neg_even x} = {0, 2, 4, 6, 8, 10} := by
  sorry

theorem set_primes_up_to_10 :
  {x : ℕ | primes_up_to_10 x} = {2, 3, 5, 7} := by
  sorry

theorem eq_sol_set :
  {x : ℤ | eq_sol x} = {-5, 3} := by
  sorry

end set_non_neg_even_set_primes_up_to_10_eq_sol_set_l422_422009


namespace area_ratio_of_squares_l422_422578

variables (A B C D E F G H : Point)
variables (r s : ℝ) -- s is the side length of the square ABCD

-- Circle and Square Geometry Definitions
noncomputable def square_inscribed_in_circle (A B C D : Point) (r : ℝ) : Prop := sorry
noncomputable def smaller_arc (A B G H : Point) : Prop := sorry
noncomputable def forms_square (E F G H : Point) : Prop := sorry

-- Area Calculation Definitions
noncomputable def area_of_square (s : ℝ) : ℝ := s * s
noncomputable def area_ratio (area1 area2 : ℝ) : ℝ := area1 / area2

-- Main Statement
theorem area_ratio_of_squares 
 (h1 : square_inscribed_in_circle A B C D r) 
 (h2 : smaller_arc A B G H)
 (h3 : forms_square E F G H)
 (s_eq : ∀ {s}, s = distance_without_diagonals ABCD)
 : area_ratio (area_of_square s) (area_of_square s) = 1 :=
begin
  sorry
end

end area_ratio_of_squares_l422_422578


namespace sum_of_first_seven_terms_l422_422752

section
def max (a b : ℕ) := if a ≥ b then a else b

def a_seq : ℕ → ℤ
| 0       := 0 -- Not used in this problem, but required for natural number indexing.
| 1       := -5
| 2       := -2
| 3       := 1
| 4       := 4
| 5       := 8
| 6       := 16
| n + 1  := max (a_seq n + 3) (2 * a_seq n)

noncomputable def S (n : ℕ) : ℤ :=
(nat.iterate (λ t, t + a_seq (n - t)) 0 n 1) / (n + 1)

theorem sum_of_first_seven_terms :
  S 7 = 54 :=
sorry
end

end sum_of_first_seven_terms_l422_422752


namespace isosceles_tetrahedron_OI_length_l422_422506

theorem isosceles_tetrahedron_OI_length (A B C D O I : Type)
  (h_AB : AB = 1300)
  (h_CD : CD = 1300)
  (h_BC : BC = 1400)
  (h_AD : AD = 1400)
  (h_CA : CA = 1500)
  (h_BD : BD = 1500)
  (h_circumcenter : is_circumcenter O ABCD)
  (h_incenter : is_incenter I ABCD) :
  ∃ n : ℕ, n = 1 ∧ (OI ABCD O I) = 0 := 
by 
  sorry

end isosceles_tetrahedron_OI_length_l422_422506


namespace incircle_radius_of_triangle_l422_422965

theorem incircle_radius_of_triangle
  (A B C : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (h_right_angle : ∠ BCA = π / 2)
  (h_A_45 : ∠ BAC = π / 4)
  (h_AC_8 : dist A C = 8) :
  r_incircle (A, B, C) = 4 - 2 * sqrt 2 :=
sorry

end incircle_radius_of_triangle_l422_422965


namespace find_five_digit_number_l422_422228

theorem find_five_digit_number
  (x y : ℕ)
  (h1 : 10 * y + x - (10000 * x + y) = 34767)
  (h2 : 10 * y + x + (10000 * x + y) = 86937) :
  10000 * x + y = 26035 := by
  sorry

end find_five_digit_number_l422_422228


namespace correct_average_is_121_point_5_l422_422474

theorem correct_average_is_121_point_5 :
  (list_length : ℕ) = 20 → 
  (incorrect_average : ℝ) = 120 → 
  (wrong_readings: ℕ → ℝ) = λ i, if i = 0 then 215 else if i = 1 then 180 else if i = 2 then 273 else if i = 3 then 94 else if i = 4 then 156 else 0 →
  (correct_readings: ℕ → ℝ) = λ i, if i = 0 then 205 else if i = 1 then 170 else if i = 2 then 263 else if i = 3 then 84 else if i = 4 then 166 else 0 →
  (((list_length * incorrect_average) + (sum (λ i, (wrong_readings i - correct_readings i)) [0, 1, 2, 3, 4])) / list_length) = 121.5 :=
begin
  intros list_length incorrect_average wrong_readings correct_readings,
  sorry
end

end correct_average_is_121_point_5_l422_422474


namespace parallel_planes_conditions_l422_422512

-- Definitions of the geometric relationships
variables {Plane Line : Type} 

-- Assumptions
variables (α β γ : Plane) (a b : Line)
variables (a_subset_alpha : a ⊆ α) (b_subset_beta : b ⊆ β)
variables (a_parallel_beta : a ∥ β) (b_parallel_alpha : b ∥ α)
variables (alpha_parallel_gamma : α ∥ γ) (beta_parallel_gamma : β ∥ γ)
variables (alpha_perpendicular_gamma : α ⟂ γ) (beta_perpendicular_gamma : β ⟂ γ)
variables (a_perpendicular_alpha : a ⟂ α) (b_perpendicular_beta : b ⟂ β)
variables (a_parallel_b : a ∥ b)

-- Proof statement
theorem parallel_planes_conditions (h1 : α ∥ γ) (h2 : β ∥ γ) (h3 : a ⟂ α) (h4 : b ⟂ β) (h5 : a ∥ b) : α ∥ β :=
sorry

end parallel_planes_conditions_l422_422512


namespace part1_part2_l422_422496

noncomputable theory

open Real

variables (A B C a b c : ℝ) [fact (b = 3)] [fact (B ≠ π / 2)]

/-- Part 1: Given the equation (a - b * cos(C)) * sin(B) + sqrt(3) * b * cos(B) * cos(C) = 0 
    and b = 3 and a = 5, prove c = 7 -/
theorem part1 (h1 : (a - b * cos C) * sin B + sqrt 3 * b * cos B * cos C = 0)
  (h2 : a = 5) : c = 7 := sorry

/-- Part 2: Given the equation (a - b * cos(C)) * sin(B) + sqrt(3) * b * cos(B) * cos(C) = 0
    and b = 3, prove the area of the triangle ABC is 6 * sqrt(3) -/
theorem part2 (h1 : (a - b * cos C) * sin B + sqrt 3 * b * cos B * cos C = 0)
  (median_on_ab : (7 / 2)^2 = (1 / 4) * (2 * b^2 + 2 * a^2 - 4 * a * b * cos C)) :
  (1 / 2) * a * b * (sin C) = 6 * sqrt 3 := sorry

end part1_part2_l422_422496


namespace problem1_problem2_l422_422398

-- Define the problem setting with given conditions
variables {a b c : ℝ} {A B C : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variables (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
variables (h_triangle : A + B + C = π)
variables (h1 : b * tan B = sqrt 3 * (a * cos C + c * cos A))
variables (h2 : b = 2 * sqrt 3) (h_area : 1 / 2 * a * c * sin B = 3 * sqrt 3)

-- Lean 4 statement for Problem 1
theorem problem1 : B = π / 3 :=
  sorry

-- Lean 4 statement for Problem 2
theorem problem2 : a + c = 4 * sqrt 3 :=
  sorry

end problem1_problem2_l422_422398


namespace proof_problem_l422_422180

noncomputable def problem (A B C D E : Type*) (angle : Type*) 
  (intersect : Segment BD ∩ Segment AE = Segment C)
  (eq1 : AB = BC)
  (eq2 : BC = CD)
  (eq3 : CD = CE)
  (eq4 : ∠A = (5/2) * ∠B) : Prop :=
∠D = 52.5°

theorem proof_problem (A B C D E : Type*) (angle : Type*) 
  (intersect : Segment BD ∩ Segment AE = Segment C)
  (AB BC CD CE : ℝ)
  (eq1 : AB = BC)
  (eq2 : BC = CD)
  (eq3 : CD = CE)
  (angleA angleB angleD : ℝ)
  (eq4 : angleA = (5/2) * angleB) : ∠D = 52.5 :=
sorry

end proof_problem_l422_422180


namespace total_tomato_seeds_l422_422528

theorem total_tomato_seeds (mike_morning mike_afternoon : ℕ) 
  (ted_morning : mike_morning = 50) 
  (ted_afternoon : mike_afternoon = 60) 
  (ted_morning_eq : 2 * mike_morning = 100) 
  (ted_afternoon_eq : mike_afternoon - 20 = 40)
  (total_seeds : mike_morning + mike_afternoon + (2 * mike_morning) + (mike_afternoon - 20) = 250) : 
  (50 + 60 + 100 + 40 = 250) :=
sorry

end total_tomato_seeds_l422_422528


namespace sum_of_real_solutions_l422_422743

theorem sum_of_real_solutions :
  let f := λ x : ℝ, (x - 3) / (x^2 + 5 * x + 2)
  let g := λ x : ℝ, (x - 6) / (x^2 - 8 * x)
  (∃ x1 x2, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2) →
  (∃ x1 x2, x1 ≠ x2 ∧ x1 + x2 = -26 / 5) :=
by
  sorry

end sum_of_real_solutions_l422_422743


namespace sqrt_sum_4_pow_4_eq_32_l422_422609

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l422_422609


namespace number_of_students_per_normal_class_l422_422437

theorem number_of_students_per_normal_class (total_students : ℕ) (percentage_moving : ℕ) (grade_levels : ℕ) (adv_class_size : ℕ) (additional_classes : ℕ) 
  (h1 : total_students = 1590) 
  (h2 : percentage_moving = 40) 
  (h3 : grade_levels = 3) 
  (h4 : adv_class_size = 20) 
  (h5 : additional_classes = 6) : 
  (total_students * percentage_moving / 100 / grade_levels - adv_class_size) / additional_classes = 32 :=
by
  sorry

end number_of_students_per_normal_class_l422_422437


namespace geometric_sequence_log_sum_l422_422841

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 3 * a 7 = 4) :
  (∑ i in Finset.range 9, Real.log (a i) / Real.log 2) = 9 :=
sorry

end geometric_sequence_log_sum_l422_422841


namespace trajectory_of_Q_l422_422060

/-- Given the following conditions: 
 * C is the center of the circle with equation (x+1)^2 + y^2 = 8
 * P is a moving point on the circle
 * Q is a point on the radius CP
 * A is a point at (1, 0)
 * M is a point on AP such that 𝑀𝑄⃗ ⃗ ⋅ 𝐴𝑃⃗ ⃗ = 0 and 𝐴𝑃⃗ ⃗ = 2𝐴𝑀⃗ ⃗
 
  We aim to prove:
  1. The equation of the trajectory of Q: (x^2 / 2) + y^2 = 1
  2. The range of slope k for a line tangent to the circle x^2 + y^2 = 1
  given that ℝ is the set of real numbers and
  3/4 ≤ (OF ⃗ )⋅(OH⃗ ) ≤ 4/5 where F and H are the intersection points with the trajectory of Q.
-/
theorem trajectory_of_Q 
: ∀ (C P Q A M O F H : ℝ → ℝ × ℝ), 
(C (x,y) = (x + 1)^2 + y^2 = 8) → 
(P (x,y) ∈ circle) → 
(Q ∈ radius (C P)) → 
(A = (1,0)) → 
((M ∈ line_segment (A P)) ∧ ((MQ • AP = 0) ∧ (AP = 2 * AM))) →
(((x * x / 2) + y * y = 1) ∧ 
∀ (k : ℝ), 
(Tangent k (circle (0, 0) 1)) → 
(Intersection l (trajectory (Q)) = {F, H}) → 
(3/4 ≤ (OF • OH) ≤ 4/5) → 
(-√2/2 ≤ k ≤ -√3/3 ∨ √3/3 ≤ k ≤ √2/2)).
Proof
by sorry

end trajectory_of_Q_l422_422060


namespace sum_planar_angles_convex_polyhedral_angle_lt_2pi_l422_422590

theorem sum_planar_angles_convex_polyhedral_angle_lt_2pi
  (n : ℕ)
  (P_n : Type) 
  [ConvexPolyhedralAngle P_n] 
  (angles : P_n → ℝ) 
  (sum_angles : ∀ (s : finset P_n), ∑ x in s, angles x) :
  sum_angles (finset.univ : finset P_n) < 2 * real.pi := 
sorry

end sum_planar_angles_convex_polyhedral_angle_lt_2pi_l422_422590


namespace range_of_func_y_l422_422806

-- Definitions based on conditions
def set_A : set ℝ := { x | 2^x ≤ (1 / 4) ^ (x - 2) }

def func_y (x : ℝ) : ℝ := (1 / 2) ^ x

-- Statement of the theorem
theorem range_of_func_y : (∀ x ∈ set_A, func_y x ∈ set.Ici (2 ^ (-4 / 3))) :=
sorry

end range_of_func_y_l422_422806


namespace decorations_count_l422_422718

-- Define the conditions as Lean definitions
def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def large_cauldron := 1
def budget_more_decorations := 20
def left_to_put_up := 10

-- Define the total decorations
def decorations_already_up := plastic_skulls + broomsticks + spiderwebs + pumpkins + large_cauldron
def additional_decorations := budget_more_decorations + left_to_put_up
def total_decorations := decorations_already_up + additional_decorations

-- Prove the total number of decorations will be 83
theorem decorations_count : total_decorations = 83 := by 
  sorry

end decorations_count_l422_422718


namespace probability_of_prime_ball_is_one_half_l422_422231

-- Define the set of all balls
def balls : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

-- Define a function to check primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers from the list of balls
def prime_balls : Set ℕ := {n ∈ balls | is_prime n}

-- Calculate the probability that a randomly chosen ball has a prime number
def probability_prime_ball : ℚ :=
  Set.card prime_balls / Set.card balls

-- Theorem statement that asserts the probability is 1/2
theorem probability_of_prime_ball_is_one_half : probability_prime_ball = 1 / 2 :=
by
  sorry

end probability_of_prime_ball_is_one_half_l422_422231


namespace costs_equal_at_60_guests_l422_422932

theorem costs_equal_at_60_guests :
  ∀ (x : ℕ),
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by
  intro x
  split
  · intro h
    have : 800 - 500 = 35 * x - 30 * x,
    calc
      800 + 30 * x = 500 + 35 * x : h
    rw [add_comm, this]
    sorry
  · intro hx
    rw hx
    rfl

end costs_equal_at_60_guests_l422_422932


namespace triangle_is_right_triangle_l422_422497

variable {α : Type*} [linear_ordered_field α]
  {a b c : α} {A B C : α}

-- Conditions in the problem
def sides_opposite_angles (a b c A B C : α) : Prop :=
  (A + B + C = π) ∧ (a > 0) ∧ (b > 0) ∧ (c > 0)

def tan_formula (a b c A B : α) : Prop :=
  (cos B ≠ 0) ∧ (a ≠ 0) ∧ (sqrt 3 * c / a * cos B = tan A + tan B)

def triangle_condition (a b c : α) : Prop :=
  b - c = sqrt 3 * a / 3

-- Proof problem statement
theorem triangle_is_right_triangle
  (A B C a b c : α)
  (h1 : sides_opposite_angles a b c A B C)
  (h2 : tan_formula a b c A B)
  (h3 : triangle_condition a b c) :
  B = π / 2 ∨ C = π / 2 :=
sorry

end triangle_is_right_triangle_l422_422497


namespace pyramid_base_sidelength_l422_422194

theorem pyramid_base_sidelength (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120) (hh : h = 24) (area_eq : A = 1/2 * s * h) : s = 10 := by
  sorry

end pyramid_base_sidelength_l422_422194


namespace sqrt_fraction_eq_value_l422_422270

theorem sqrt_fraction_eq_value (x : ℝ) (h : (sqrt x + sqrt 243) / sqrt 75 = 2.4) : x = 27 :=
  sorry

end sqrt_fraction_eq_value_l422_422270


namespace number_of_digits_of_n_l422_422145

theorem number_of_digits_of_n :
  ∃ n : ℕ,
    (n > 0) ∧ 
    (15 ∣ n) ∧ 
    (∃ m : ℕ, n^2 = m^4) ∧ 
    (∃ k : ℕ, n^4 = k^2) ∧ 
    Nat.digits 10 n = 5 :=
by
  sorry

end number_of_digits_of_n_l422_422145


namespace locus_H_interior_OXY_l422_422957

variable {α : Type} [LinearOrderedField α]

structure Triangle (α : Type) :=
(O A B : α × α)
(acute_angle_O : true) -- Assuming we have an acute angle at O for the structure definition

def foot_perpendicular (O A B : α × α) (M : α × α) : (α × α) :=
sorry -- Foot of the perpendicular from M to OA (details skipped)

def orthocenter (P Q O : α × α) : α × α :=
sorry -- Orthocenter of triangle OPQ (details skipped)

def locus_of_H (O A B : α × α) : set (α × α) :=
{ H | ∃ (M : α × α) (inside_triangle_OAB : true), let P := foot_perpendicular O A M, 
                                                 let Q := foot_perpendicular O B M in
                                                 orthocenter P Q O = H }

theorem locus_H_interior_OXY (O A B : α × α) (tri : Triangle α) :
  let X := foot_perpendicular O A B in
  let Y := foot_perpendicular O B A in
  locus_of_H O A B = { H | H ∈ interior (Triangle.mk O X Y) } :=
sorry

end locus_H_interior_OXY_l422_422957


namespace polar_line_circle_intersection_l422_422858

noncomputable def line_eq (a : ℝ) (ρ θ : ℝ) : Prop := ρ * Real.cos (θ + π / 6) = a
noncomputable def circle_eq (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

def exactly_one_common_point (a : ℝ) : Prop :=
  ∀ (ρ : ℝ) (θ : ℝ), line_eq a ρ θ ∧ circle_eq ρ θ → 
  a = -3 ∨ a = 1

theorem polar_line_circle_intersection (a : ℝ) :
  (∀ (ρ θ : ℝ), line_eq a ρ θ ∧ circle_eq ρ θ → 
  ∃! (ρ : ℝ) (θ : ℝ), line_eq a ρ θ ∧ circle_eq ρ θ) →
  exactly_one_common_point a :=
sorry

end polar_line_circle_intersection_l422_422858


namespace part_a_cannot_be_achieved_part_b_cannot_be_achieved_l422_422635

-- Define the grid and conditions for both parts (a) and (b)
open Set

def grid : Type := ℤ × ℤ
def is_chip (g : grid → Prop) (c : grid) : Prop := g c

def initial_chips_a : grid → Prop
| (0, 0) := true
| (0, 1) := true
| (1, 0) := true
| (1, 1) := true
| (2, 0) := true
| (2, 1) := true
| _      := false

def initial_chip_b : grid → Prop
| (0, 0) := true
| _      := false

def free_all_cells (g : grid → Prop) : Prop :=
  ∀ c, g c = false

theorem part_a_cannot_be_achieved : ¬ (∃ g : grid → Prop, ∀ c, initial_chips_a c → g c) ∧ free_all_cells g :=
by sorry

theorem part_b_cannot_be_achieved : ¬ (∃ g : grid → Prop, ∀ c, initial_chip_b c → g c) ∧ free_all_cells g :=
by sorry

end part_a_cannot_be_achieved_part_b_cannot_be_achieved_l422_422635


namespace sum_of_roots_zero_l422_422753

theorem sum_of_roots_zero :
  let roots := {x : ℝ | x^2 - 7*|x| + 6 = 0}
  ∑ x in roots, x = 0 :=
begin
  sorry
end

end sum_of_roots_zero_l422_422753


namespace factors_of_f_l422_422171

-- Define the polynomial f(x) = x^4 + 16
def f (x : ℝ) := x^4 + 16

-- Define the potential factors g(x) and h(x)
def g (x : ℝ) := x^2 - 4x + 4
def h (x : ℝ) := x^2 + 4x + 4

-- State that g(x) and h(x) are factors of f(x)
theorem factors_of_f : ∃ (p q : ℝ → ℝ), (p = g ∨ p = h) ∧ (q = g ∨ q = h) ∧ f = λ x, p x * q x :=
by sorry

end factors_of_f_l422_422171


namespace number_of_valid_triangles_is_27_l422_422091

-- Condition Definitions
def is_valid_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + b + c < 15

def count_valid_triangles : ℕ :=
  (Finset.range 15).sum (λ a, (Finset.range 15).sum (λ b, (Finset.range 15).count (λ c, is_valid_triangle a b c)))

theorem number_of_valid_triangles_is_27 : count_valid_triangles = 27 :=
  sorry

end number_of_valid_triangles_is_27_l422_422091


namespace negation_of_proposition_l422_422215

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℤ, 2 * x_0 + x_0 + 1 ≤ 0) ↔ ∀ x : ℤ, 2 * x + x + 1 > 0 :=
by sorry

end negation_of_proposition_l422_422215


namespace valid_pairings_count_l422_422840

-- Define a structure representing a seating arrangement around a circle
structure Person := (id : ℕ)
def people : List Person := List.range 12 |>.map (λ i => ⟨i + 1⟩)

-- Define the neighborhood relationship based on the given conditions
def knows (p1 p2 : Person) : Prop :=
  | p1.id % 12 + 1 == p2.id         -- adjacent right
  || p2.id % 12 + 1 == p1.id        -- adjacent left
  || (p1.id + 6) % 12 == p2.id      -- directly across
  || (p1.id + 2) % 12 + 1 == p2.id  -- two places to the right

-- Define what constitutes a valid pairing
def valid_pairing (pairs : List (Person × Person)) : Prop :=
  pairs.length = 6
  ∧ (∀ (p_i p_j : Person × Person), p_i ∈ pairs → p_j ∈ pairs → (p_i ≠ p_j → p_i.1 ≠ p_j.1 ∧ p_i.2 ≠ p_j.2))
  ∧ (∀ (p : Person × Person), p ∈ pairs → knows p.1 p.2)

-- The theorem to prove the number of valid pairings
theorem valid_pairings_count :
  ∃ n, n = 15 ∧ n = List.filter valid_pairing (List.powerset (List.cartesianProduct people people)).length := 
  sorry

end valid_pairings_count_l422_422840


namespace customers_left_l422_422695

theorem customers_left (initial_customers left_customers incoming_customers final_customers : ℕ)
                       (h1 : initial_customers = 19)
                       (h2 : incoming_customers = 36)
                       (h3 : final_customers = 41) :
                       initial_customers - left_customers + incoming_customers = final_customers → left_customers = 14 :=
by
  intros h
  rw [h1, h2, h3] at h
  linarith

end customers_left_l422_422695


namespace quadratic_non_real_roots_b_range_l422_422097

theorem quadratic_non_real_roots_b_range (b : ℝ) :
  let f := λ x : ℝ, x^2 + b * x + 16
  -- Discriminant condition for non-real roots:
  -- b^2 - 64 < 0
  (b^2 < 64) -> b ∈ Set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_b_range_l422_422097


namespace find_BC_length_l422_422399

-- Define the given data and problem conditions
variables (A B C I P Q : Type)
variables (AB AC BC : ℝ)
variables [h1 : noncomputable] (incenter : A × B × C → I)
variables [h2 : midpoint : A × B → P]
variables [h3 : midpoint : A × C → Q]
variables [h4 : ∠ P I Q + ∠ B I C = 180)

-- Main statement to prove the length of BC
theorem find_BC_length (hAB : AB = 20) (hAC : AC = 14) : BC = 34 / 3 :=
sorry

end find_BC_length_l422_422399


namespace triangle_area_l422_422266

theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (h_perimeter : perimeter = 40) (h_inradius : inradius = 2.5) : 
  (inradius * (perimeter / 2)) = 50 :=
by
  -- Lean 4 statement code
  sorry

end triangle_area_l422_422266


namespace rise_in_water_level_calculation_l422_422317

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width
noncomputable def rise_in_water_level (V A : ℝ) : ℝ := V / A

theorem rise_in_water_level_calculation :
  let r := 10
  let length := 30
  let width := 25
  let V := volume_of_sphere r
  let A := area_of_rectangle length width
  V / A ≈ 5.59 :=
by
  let r := 10
  let length := 30
  let width := 25
  let V := volume_of_sphere r
  let A := area_of_rectangle length width
  sorry

end rise_in_water_level_calculation_l422_422317


namespace IncorrectStatements_l422_422113

def StatementA : Prop :=
  "Axioms in mathematics are propositions that are assumed to be true without proof."

def StatementB : Prop :=
  "A valid mathematical proof may consist of several different logical sequences leading to the same conclusion."

def StatementC : Prop :=
  "All terms and concepts within a proof must be explicitly defined before they can be used in the argumentation."

def StatementD : Prop :=
  "A valid conclusion can still be technically drawn from an invalid or false set of premises in formal logic."

def StatementE : Prop :=
  "A direct proof method requires having more than one contradicting statements to reach a conclusion."

theorem IncorrectStatements : ¬StatementD ∧ ¬StatementE :=
  by
  sorry

end IncorrectStatements_l422_422113


namespace D_score_is_2_l422_422372

-- Definitions of conditions

-- Students and their scores
universe u
constant A B C D : Type u
constant score : Type u → ℕ

-- The quiz consists of 5 questions
constant Q : ℕ
axiom Q_def : Q = 5

-- Scores of students A, B, and C
constant score_A score_B score_C : ℕ

-- B's score is higher than C's score by 1 point
axiom score_B_higher_C : score_B = score_C + 1

-- B and C differ only on the 4th question, and B's answer is correct (mark)
constant answer : Type u → ℕ → bool
axiom answer_diff_B_C :
  (∀ q, q ≠ 4 → answer B q = answer C q) ∧ answer B 4 = true

-- A and D have contrary answers for the 1st, 2nd, 3rd, and 5th questions
axiom contrary_answers_A_D :
  ∀ q, q ∈ [1, 2, 3, 5] → (answer A q = !answer D q) 

-- Combined score of A and D for questions 1, 2, 3, and 5 is 4
axiom combined_score_A_D :
  score A + score D = 4 ∧ 
  (answer A 4 = false ∧ answer D 4 = false)

-- The desired proof problem: Prove that D's score is 2
theorem D_score_is_2 : score D = 2 := sorry

end D_score_is_2_l422_422372


namespace total_cost_of_apples_and_bananas_l422_422222

variable (a b : ℝ)

theorem total_cost_of_apples_and_bananas (a b : ℝ) : 2 * a + 3 * b = 2 * a + 3 * b :=
by
  sorry

end total_cost_of_apples_and_bananas_l422_422222


namespace a_n_general_term_T_n_sum_l422_422949

def a_n (n : ℕ) : ℝ := 2 * n - 1

def b_n : ℕ → ℝ
| 1     := 1
| (n+1) := b_n n / 2

def a_n_mul_b_n (n : ℕ) : ℝ := a_n n * b_n n

def T_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, a_n_mul_b_n (k + 1))

theorem a_n_general_term (n : ℕ) : a_n n = 2 * n - 1 :=
by sorry

theorem T_n_sum (n : ℕ) : T_n n = 6 - (2 * n + 3) / (2 ^ (n - 1)) :=
by sorry

end a_n_general_term_T_n_sum_l422_422949


namespace ratio_of_areas_of_concentric_circles_l422_422969

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_concentric_circles_l422_422969


namespace keychain_arrangements_l422_422848

theorem keychain_arrangements :
  let H := "House"
  let C := "Car"
  let O := "Office"
  let B := "Bike"
  let Key1 := "Key1"
  let Key2 := "Key2"
  let keys := ["House", "Car", "Office", "Bike", "Key1", "Key2"]
  -- Grouping keys as pairs
  let pairHC := [H, C]
  let pairOB := [O, B]
  let remainingKeys := ["Key1", "Key2"]
  -- Considering rotational and reflective symmetries
  countingArrangements(keys, pairHC, pairOB, remainingKeys) = 24 :=
by
  sorry

-- Function to count the number of distinct arrangements (considering rotation and reflection)
def countingArrangements (keys : List String) (pairHC : List String) (pairOB : List String) (remainingKeys : List String) : Nat :=
  -- Logic to count arrangements goes here
  sorry

end keychain_arrangements_l422_422848


namespace least_number_to_multiply_for_multiple_of_112_l422_422209

theorem least_number_to_multiply_for_multiple_of_112 (n : ℕ) : 
  (Nat.lcm 72 112) / 72 = 14 := 
sorry

end least_number_to_multiply_for_multiple_of_112_l422_422209


namespace omega_value_a_value_g_properties_l422_422524

noncomputable def f (x : ℝ) (ω : ℝ) (a : ℝ) : ℝ := 
  sin (2 * ω * x + (π / 3)) + (sqrt 3 / 2) + a

noncomputable def g (x : ℝ) (ω : ℝ) (a : ℝ) : ℝ := 
  f x ω a - a

theorem omega_value (ω a : ℝ) (h : ω > 0) :
  (∀ x, fractional x ∧ first_highest f x x = π / 6) → ω = 1 / 2 :=
by
  sorry

theorem a_value (ω a : ℝ) (h : ω = 1 / 2) 
  (h_min : ∀ x, (-π / 3 ≤ x ∧ x ≤ 5 * π / 6) → 
    min (f x ω a) = sqrt 3) :
  a = (sqrt 3 + 1) / 2 :=
by
  sorry

theorem g_properties :
  (∀ (x : ℝ), g x = sin (x + π / 3) + (sqrt 3 / 2)) ∧
  (axis of symmetry for g is x = π / 6 + k * π) ∧
  (center of symmetry for g is (-π / 3 + k * π, sqrt 3 / 2)) :=
by
  sorry

end omega_value_a_value_g_properties_l422_422524


namespace function_even_not_odd_not_both_l422_422392

def f (x : ℝ) : ℝ := x^2 - 2

theorem function_even_not_odd_not_both (x : ℝ) (h : x ∈ Ioc (-5:ℝ) 5) :
  (∀ x : ℝ, f (-x) = f x) ∧ ¬(∀ x : ℝ, f (-x) = -f x) :=
by {
  have h_even : ∀ x : ℝ, f (-x) = f x := 
    λ x, by simp [f, (*.)],
  have h_not_odd : ¬(∀ x : ℝ, f (-x) = -f x) := 
    λ h, false.elim $ h 1 (by norm_num),
  exact ⟨h_even, h_not_odd⟩,
}

end function_even_not_odd_not_both_l422_422392


namespace path_count_from_minus5_minus5_to_5_5_l422_422275

theorem path_count_from_minus5_minus5_to_5_5 :
  let paths := 4252 in
  (∀ x y : ℤ, -3 ≤ x → x ≤ 3 → -3 ≤ y → y ≤ 3 → false) →
  let step_path (start: ℤ × ℤ) :=
    | (x, y) => (x+1, y) or (x, y+1) in
  (∀ n: ℕ, start end: ℤ × ℤ, n = 20 → start = (-5,-5) → end = (5,5) →
    (∃ p : ℕ → ℤ × ℤ, p 0 = start ∧ p 20 = end ∧ (∀ i < 20, p (i+1) = step_path (p i) ∧ (∀ x y : ℤ, let (x, y) := p i in (-3 ≤ x ∧ x ≤ 3 ∧ -3 ≤ y ∧ y ≤ 3) → false)))
  :=
  sorry  

end path_count_from_minus5_minus5_to_5_5_l422_422275


namespace race_distance_1803_l422_422109

noncomputable def minRaceDistance (A B : Point) (wall : Line) :=
  let dist_A_to_wall := 400
  let dist_wall_to_B := 600
  let length_wall := 1500
  sqrt (length_wall^2 + (dist_A_to_wall + dist_wall_to_B)^2)

theorem race_distance_1803 (A B : Point) (wall : Line) :
  dist A wall = 400 ∧ dist wall B = 600 ∧ length wall = 1500 →
  minRaceDistance A B wall = 1803 :=
  by
  sorry

end race_distance_1803_l422_422109


namespace exists_x_eq_l422_422150

noncomputable def R (P Q : ℝ[X]) : ℝ[X] :=
  P.comp (X - C 1) - Q.comp (X + C 1)

theorem exists_x_eq : ∀ (P Q : ℝ[X]),
  Polynomial.degree P = 2014 →
  Polynomial.degree Q = 2014 →
  P.leadingCoeff = 1 →
  Q.leadingCoeff = 1 →
  (∀ x : ℝ, P.eval x ≠ Q.eval x) →
  ∃ x : ℝ, P.eval (x - 1) = Q.eval (x + 1) :=
sorry

end exists_x_eq_l422_422150


namespace minimum_distance_PQ_l422_422703

-- Define the conditions in the problem
def cube_side_length : ℝ := 100
def speed_P : ℝ := 3
def speed_Q : ℝ := 2
def initial_P_position : ℝ × ℝ × ℝ := (0, 0, 0)
def initial_Q_position : ℝ × ℝ × ℝ := (0, 0, 100)
def P_position (t : ℝ) : ℝ × ℝ × ℝ := (3 * t, 3 * t, 3 * t)
def Q_position (t : ℝ) : ℝ × ℝ × ℝ := (2 * t, 0, 100)

-- Define the distance function
def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

-- Define the distance between P and Q as a function of time
def distance_PQ (t : ℝ) : ℝ :=
  distance (P_position t) (Q_position t)

-- Statement to prove the minimum distance
theorem minimum_distance_PQ :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ (cube_side_length / speed_P) ∧
    distance_PQ t = 72.5333 :=
sorry

end minimum_distance_PQ_l422_422703


namespace common_ratio_of_sequence_l422_422665

theorem common_ratio_of_sequence 
  (a1 a2 a3 a4 : ℤ)
  (h1 : a1 = 25)
  (h2 : a2 = -50)
  (h3 : a3 = 100)
  (h4 : a4 = -200)
  (is_geometric : ∀ (i : ℕ), a1 * (-2) ^ i = if i = 0 then a1 else if i = 1 then a2 else if i = 2 then a3 else a4) : 
  (-50 / 25 = -2) ∧ (100 / -50 = -2) ∧ (-200 / 100 = -2) :=
by 
  sorry

end common_ratio_of_sequence_l422_422665


namespace total_amount_paid_l422_422500

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_tshirts : ℕ := 6

theorem total_amount_paid : 
  (number_of_tshirts : ℝ) * (original_price * discount_rate) = 60 := by
  sorry

end total_amount_paid_l422_422500


namespace number_of_distinct_real_roots_of_transformed_equation_l422_422829

theorem number_of_distinct_real_roots_of_transformed_equation
  (a b c : ℝ)
  (x1 x2 : ℝ)
  (h1 : (x1^3 + a*x1^2 + b*x1 + c) = x1)
  (h2 : (3*x1^2 + 2*a*x1 + b) = 0)
  (h3 : (3*x2^2 + 2*a*x2 + b) = 0) :
  (complex.roots (polynomial.C 3 * (polynomial.C a * ((polynomial.monomial 3 1 + polynomial.C a * (polynomial.monomial 2 1) + polynomial.C b * (polynomial.monomial 1 1) + polynomial.C c)^2) + polynomial.C a * (polynomial.monomial 2 1) + polynomial.C b) = 0)).to_finset.card = 3 :=
sorry

end number_of_distinct_real_roots_of_transformed_equation_l422_422829


namespace costs_equal_at_60_guests_l422_422931

theorem costs_equal_at_60_guests :
  ∀ (x : ℕ),
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by
  intro x
  split
  · intro h
    have : 800 - 500 = 35 * x - 30 * x,
    calc
      800 + 30 * x = 500 + 35 * x : h
    rw [add_comm, this]
    sorry
  · intro hx
    rw hx
    rfl

end costs_equal_at_60_guests_l422_422931


namespace incorrect_transformation_D_l422_422998

theorem incorrect_transformation_D (x y m : ℝ) (hxy: x = y) : m = 0 → ¬ (x / m = y / m) :=
by
  intro hm
  simp [hm]
  -- Lean's simp tactic simplifies known equalities
  -- The simp tactic will handle the contradiction case directly when m = 0.
  sorry

end incorrect_transformation_D_l422_422998


namespace slower_bus_pass_time_l422_422232

-- Definitions based on the conditions
def length_bus : ℝ := 3125
def speed_fast_kmph : ℝ := 40
def speed_slow_kmph : ℝ := 35

-- Conversion from km/hr to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)
def speed_fast_mps : ℝ := kmph_to_mps speed_fast_kmph
def speed_slow_mps : ℝ := kmph_to_mps speed_slow_kmph

-- Relative speed when buses are moving in opposite directions
def relative_speed : ℝ := speed_fast_mps + speed_slow_mps

-- Time taken for the slower bus to pass the driver of the faster one
def time_to_pass (distance : ℝ) (relative_speed : ℝ) : ℝ := distance / relative_speed

-- Proof statement
theorem slower_bus_pass_time :
  time_to_pass length_bus relative_speed ≈ 150 := by
  sorry

end slower_bus_pass_time_l422_422232


namespace ratio_area_APCQ_BDEFQ_l422_422484

-- Define regular hexagon ABCDEF
variable (A B C D E F : Point)
variable (P : Point)
variable (Q : Point)

-- Define geometric conditions
variable (h_regular_hexagon : regular_hexagon A B C D E F)
variable (h_midpoint_P : midpoint P A B)
variable (h_midpoint_Q : midpoint Q D E)

-- The theorem to prove
theorem ratio_area_APCQ_BDEFQ : 
  area_ratio (quadrilateral A P C Q) (region B D E F Q) = 2 / 3 :=
sorry

end ratio_area_APCQ_BDEFQ_l422_422484


namespace lattice_points_on_hyperbola_l422_422812

theorem lattice_points_on_hyperbola :
  (set_of (λ (p : ℤ × ℤ), p.1 ^ 2 - p.2 ^ 2 = 1500 ^ 2)).finite.card = 90 :=
sorry

end lattice_points_on_hyperbola_l422_422812


namespace inequality_proof_l422_422539

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a ^ 2 + 1)) + (b / Real.sqrt (b ^ 2 + 1)) + (c / Real.sqrt (c ^ 2 + 1)) ≤ (3 / 2) :=
by
  sorry

end inequality_proof_l422_422539


namespace value_of_b_l422_422942

theorem value_of_b (b x : ℝ) (h1 : 2 * x + 7 = 3) (h2 : b * x - 10 = -2) : b = -4 :=
by
  sorry

end value_of_b_l422_422942


namespace find_Q_coordinates_l422_422683

noncomputable def Q_coordinates : ℝ × ℝ :=
  (-1 - 20 * real.sqrt 6, 121)

theorem find_Q_coordinates (V : ℝ × ℝ) (F : ℝ × ℝ) (Q_in_second_quadrant : Prop) (QF_distance : ℝ) :
  V = (-1, 1) →
  F = (-1, 2) →
  QF_distance = 121 →
  Q_in_second_quadrant →
  (∃ Q : ℝ × ℝ, Q = Q_coordinates) :=
by
  intros hV hF hQF h2Q
  use Q_coordinates
  sorry

end find_Q_coordinates_l422_422683


namespace fishing_trip_count_l422_422544

theorem fishing_trip_count (n: ℕ):
    let shelly_per_trip := 5 - 2 in
    let sam_per_trip := shelly_per_trip - 1 in
    let total_per_trip := shelly_per_trip + sam_per_trip in
    total_per_trip * n = 25 -> n = 5 := 
by
  intro h
  sorry

end fishing_trip_count_l422_422544


namespace evaluate_expression_l422_422337

theorem evaluate_expression :
  12 - 5 * 3^2 + 8 / 2 - 7 + 4^2 = -20 :=
by
  sorry

end evaluate_expression_l422_422337


namespace minimum_average_from_digits_no_repeats_l422_422377

theorem minimum_average_from_digits_no_repeats :
  ∃ S : set ℕ, S.card = 6 ∧ S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∧ (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → (x % 10 ≠ y % 10 ∧ x / 10 ≠ y / 10))
  ∧ (∀ s1 s2 s3, s1 ∈ { (d1, d2) | d1 ∈ S ∧ d2 ∈ S }
   → s2 ∈ { (d3, d4) | d3 ∈ S ∧ d4 ∈ S }
   → s3 ∈ { (d5, d6) | d5 ∈ S ∧ d6 ∈ S }
   → (s1 ≠ s2 ∧ s2 ≠ s3 ∧ s3 ≠ s1)
   → (s1.1 * 10 + s1.2 + s2.1 * 10 + s2.2 + s3.1 * 10 + s3.2 +
     (∑ x in S, x) - (s1.1 + s1.2 + s2.1 + s2.2 + s3.1 + s3.2)) / 6 = 16.5) := sorry

end minimum_average_from_digits_no_repeats_l422_422377


namespace average_speed_is_six_l422_422694

-- Define lengths of each segment in kilometers
def swim_length : ℝ := 1
def bike_length : ℝ := 2
def run_length : ℝ := 2
def walk_length : ℝ := 1

-- Define speeds for each segment in kilometers per hour
def swim_speed : ℝ := 2
def bike_speed : ℝ := 25
def run_speed : ℝ := 12
def walk_speed : ℝ := 4

-- Define the total distance
def total_distance : ℝ := swim_length + bike_length + run_length + walk_length

-- Define the time for each segment
def swim_time : ℝ := swim_length / swim_speed
def bike_time : ℝ := bike_length / bike_speed
def run_time : ℝ := run_length / run_speed
def walk_time : ℝ := walk_length / walk_speed

-- Define the total time
def total_time : ℝ := swim_time + bike_time + run_time + walk_time

-- Define the average speed
def average_speed : ℝ := total_distance / total_time

-- The statement to be proven
theorem average_speed_is_six : average_speed = 6 := by
  sorry

end average_speed_is_six_l422_422694


namespace area_ratio_of_concentric_circles_l422_422982

noncomputable theory

-- Define the given conditions
def C1 (r1 : ℝ) : ℝ := 2 * Real.pi * r1
def C2 (r2 : ℝ) : ℝ := 2 * Real.pi * r2
def arc_length (angle : ℝ) (circumference : ℝ) : ℝ := (angle / 360) * circumference

-- Lean statement for the math proof problem
theorem area_ratio_of_concentric_circles 
  (r1 r2 : ℝ) (h₁ : arc_length 60 (C1 r1) = arc_length 48 (C2 r2)) : 
  (Real.pi * r1^2) / (Real.pi * r2^2) = 16 / 25 :=
by
  sorry  -- Proof omitted

end area_ratio_of_concentric_circles_l422_422982


namespace sufficient_but_not_necessary_condition_l422_422081

variables (α β γ : Plane)
variables (a b c : Line)
variables (θ : ℝ)

-- Conditions
def planes_intersection : Prop :=
  α ∩ β = a ∧ β ∩ γ = b ∧ γ ∩ α = c

def angle_between_planes : Prop :=
  ∀ (π1 π2 : Plane), π1 ≠ π2 → angle π1 π2 = θ

def angle_greater_than_pi_over_3 : Prop :=
  θ > Real.pi / 3

def lines_intersect_at_single_point : Prop :=
  ∃ (P : Point), P ∈ a ∧ P ∈ b ∧ P ∈ c

-- Statement to prove
theorem sufficient_but_not_necessary_condition
  (h1 : planes_intersection α β γ a b c)
  (h2 : angle_between_planes α β γ θ)
  (h3 : lines_intersect_at_single_point a b c) :
  angle_greater_than_pi_over_3 θ →
  ∃ (P : Point), lines_intersect_at_single_point a b c ∧ ¬ (angle_greater_than_pi_over_3 θ → ¬ (lines_intersect_at_single_point a b c)) :=
sorry

end sufficient_but_not_necessary_condition_l422_422081


namespace solve_fraction_equation_l422_422550

theorem solve_fraction_equation
  (x : ℝ)
  (h1 : 3 * x^2 + 6 * x - 4 ≠ 0)
  (h2 : 3 * x - 2 ≠ 0)
  (h3 : (6 * x + 2) / (3 * x^2 + 6 * x - 4) = 3 * x / (3 * x - 2)) :
  x = sqrt 3 / 3 ∨ x = -sqrt 3 / 3 :=
by
  sorry

end solve_fraction_equation_l422_422550


namespace avg_of_multiplied_numbers_is_84_l422_422264

theorem avg_of_multiplied_numbers_is_84
  (avg_nat : ℕ → ℕ → ℕ := λ n sum, sum / n)
  (n : ℕ) (avg : ℕ) (multiplier : ℕ) (new_avg : ℕ)
  (h_avg : avg_nat 10 (10 * 7) = 7)
  (h_mult : multiplier = 12)
  (h_new_avg : new_avg = avg_nat 10 ((10 * 7) * multiplier)) :
  new_avg = 84 :=
by
  sorry

end avg_of_multiplied_numbers_is_84_l422_422264


namespace sequence_general_formula_T_2014_U_n_l422_422416

-- 1. Prove the general formula for the sequence {a_n}
theorem sequence_general_formula (S_n a_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, S_n n = 1/2 - 1/2 * a_n n) →
  (∃ n, a_n n = (1/3)^n) :=
sorry

-- 2. Prove T_{2014}
theorem T_2014 : ∃ Tₙ, Tₙ 2014 = -4028 / 2015 :=
sorry

-- 3. Prove U_n
theorem U_n (a_n : ℕ → ℝ) (f : ℝ → ℝ) (c_n : ℕ → ℝ) (U_n : ℕ → ℝ) (n : ℕ)
  (h_a : ∀ n, a_n n = (1/3)^n) 
  (h_f : ∀ x : ℝ, f x = Real.log x / Real.log 3) 
  (h_c : ∀ n, c_n n = a_n n * f (a_n n)) :
  (U_n n = -3/4 + 3/4 * (1/3)^n + 3/2 * n * (1/3)^(n + 1)) :=
sorry

end sequence_general_formula_T_2014_U_n_l422_422416


namespace club_limit_l422_422223

theorem club_limit (n : ℕ) (h : ∀ (A B : Finset ℕ), A.card = 4 ∧ B.card = 4 ∧ coe (A ∩ B).card ≤ 2 → coe (A ∩ B).card ≤ 2) : n ≤ 12 :=
sorry

end club_limit_l422_422223


namespace Marty_combinations_l422_422526

theorem Marty_combinations :
  let colors := 5
  let methods := 4
  let patterns := 3
  colors * methods * patterns = 60 :=
by
  sorry

end Marty_combinations_l422_422526


namespace number_of_pairs_satisfying_l422_422410

theorem number_of_pairs_satisfying (h1 : 2 ^ 2013 < 5 ^ 867) (h2 : 5 ^ 867 < 2 ^ 2014) :
  ∃ k, k = 279 ∧ ∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2012 ∧ 5 ^ n < 2 ^ m ∧ 2 ^ (m + 2) < 5 ^ (n + 1) → 
  ∃ (count : ℕ), count = 279 :=
by
  sorry

end number_of_pairs_satisfying_l422_422410


namespace conic_section_is_parabola_l422_422356

-- Define the equation |y-3| = sqrt((x+4)^2 + y^2)
def equation (x y : ℝ) : Prop := |y - 3| = Real.sqrt ((x + 4) ^ 2 + y ^ 2)

-- The main theorem stating the conic section type is a parabola
theorem conic_section_is_parabola : ∀ x y : ℝ, equation x y → false := sorry

end conic_section_is_parabola_l422_422356


namespace sqrt_sum_of_powers_l422_422599

theorem sqrt_sum_of_powers : sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  have h : 4^4 = 256 := by
    calc
      4^4 = 4 * 4 * 4 * 4 := by rw [show 4^4 = 4 * 4 * 4 * 4 from rfl]
      ... = 16 * 16       := by rw [mul_assoc, mul_assoc]
      ... = 256           := by norm_num
  calc
    sqrt (4^4 + 4^4 + 4^4 + 4^4)
      = sqrt (256 + 256 + 256 + 256) := by rw [h, h, h, h]
      ... = sqrt 1024                 := by norm_num
      ... = 32                        := by norm_num

end sqrt_sum_of_powers_l422_422599


namespace train_crossing_time_l422_422322

-- Define the speed of the train in km/hr
def speed_kmh : ℕ := 60

-- Define the length of the train in meters
def length_m : ℕ := 100

-- Convert the speed from km/hr to m/s
def speed_ms : ℚ := (60 * 1000) / 3600

-- Define the expected time in seconds
def expected_time : ℚ := 6

-- The theorem we want to prove: The time it takes for the train to cross the pole is 6 seconds
theorem train_crossing_time (s_kmh : ℕ) (l_m : ℕ) : s_kmh = speed_kmh → l_m = length_m → (l_m / speed_ms) = expected_time :=
by
  intros hskh hlm
  rw [←hskh, ←hlm]
  sorry

end train_crossing_time_l422_422322


namespace smallest_average_l422_422389

noncomputable def smallest_possible_average : ℕ := 165 / 10

theorem smallest_average (s d: Finset ℕ) 
  (h1 : s.card = 3) 
  (h2 : d.card = 3) 
  (h3 : ∀x ∈ s ∪ d, x ∈ (Finset.range 10).erase 0)
  (h4 : (s ∪ d).card = 6)
  (h5 : s ∩ d = ∅) : 
  (∑ x in s, x + ∑ y in d, y) / 6 = smallest_possible_average :=
sorry

end smallest_average_l422_422389


namespace melanie_statistical_study_l422_422847

theorem melanie_statistical_study:
  let days := List.range' 1 32  -- days from 1 to 31
  let occurrences := days.map (λ d, if d <= 30 then 12 else 8)  -- number of occurrences for each day
  let dataset := List.join (days.zip occurrences).map (λ (d, count), List.replicate count d)
  let n := dataset.length
  let median_position := (n + 1) / 2
  let median := dataset.get! (median_position - 1)
  let mean := (dataset.sum : Float) / n
  let modes := days.filter (λ d, d <= 30)
  let mode_median := (modes.sum : Float) / modes.length
  (mode_median < mean ∧ mean < median) := sorry

end melanie_statistical_study_l422_422847


namespace calculate_f_log2_9_l422_422045

noncomputable def f : ℝ → ℝ
| x@(0 < x ∧ x ≤ 1) := 2^x
| x := /- handle non-0<x<=1 cases -/ sorry

theorem calculate_f_log2_9 {f : ℝ → ℝ} :
  (∀ x : ℝ, f(x + 1) = 1 / f(x)) →
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f(x) = 2^x) →
  f(log 2 9) = 8 / 9 :=
by
  intros h1 h2 
  sorry

end calculate_f_log2_9_l422_422045


namespace geom_seq_common_ratio_l422_422673

theorem geom_seq_common_ratio:
  ∃ r : ℝ, 
  r = -2 ∧ 
  (∀ n : ℕ, n = 0 → n = 3 →
  let a : ℕ → ℝ := λ n, if n = 0 then 25 else
                            if n = 1 then -50 else
                            if n = 2 then 100 else
                            if n = 3 then -200 else 0 in
  (a n = a 0 * r ^ n)) :=
by
  sorry

end geom_seq_common_ratio_l422_422673


namespace lcm_of_18_and_20_l422_422737

theorem lcm_of_18_and_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_of_18_and_20_l422_422737


namespace dogs_not_eat_either_l422_422473

-- Definitions for our conditions
variable (dogs_total : ℕ) (dogs_watermelon : ℕ) (dogs_salmon : ℕ) (dogs_both : ℕ)

-- Specific values of our conditions
def dogs_total_value : ℕ := 60
def dogs_watermelon_value : ℕ := 9
def dogs_salmon_value : ℕ := 48
def dogs_both_value : ℕ := 5

-- The theorem we need to prove
theorem dogs_not_eat_either : 
    dogs_total = dogs_total_value → 
    dogs_watermelon = dogs_watermelon_value → 
    dogs_salmon = dogs_salmon_value → 
    dogs_both = dogs_both_value → 
    (dogs_total - (dogs_watermelon + dogs_salmon - dogs_both) = 8) :=
by
  intros
  sorry

end dogs_not_eat_either_l422_422473


namespace rational_coefficient_terms_count_l422_422358

theorem rational_coefficient_terms_count :
  ∃ n, n = 61 ∧ ∀ k, 0 ≤ k ∧ k ≤ 1200 ∧ (∃ m, k = 20 * m) →
  (2 ^ (k / 4) * 3 ^ ((1200 - k) / 5) : ℚ) ∈ ℚ :=
by
  -- Step to avoid proof
  sorry

end rational_coefficient_terms_count_l422_422358


namespace find_angle_BMC_l422_422507

variables {A B C D E M O' O'' : Point}
variables {triangle_ABC : triangle A B C}
variables [RightTriangle triangle_ABC C]
variables [Median AD triangle_ABC A D M]
variables [Median BE triangle_ABC B E M]
variables [Centroid M triangle_ABC]
variables [TangentCircles (circumcircle A E M) (circumcircle C D M)]

theorem find_angle_BMC : ∠ B M C = 90 :=
sorry

end find_angle_BMC_l422_422507


namespace find_fraction_value_l422_422882

variable (a b : ℝ)

theorem find_fraction_value (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 := 
  sorry

end find_fraction_value_l422_422882


namespace transformed_function_eq_l422_422075

def initial_function (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

def shifted_function (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 3) - Real.pi / 3)

def result_function (x : ℝ) : ℝ := Real.sin (x / 2 + Real.pi / 3)

theorem transformed_function_eq :
  (∀ x : ℝ, shifted_function (x / 2) = result_function x) :=
by
  intro x
  unfold shifted_function result_function
  sorry

end transformed_function_eq_l422_422075


namespace equilibrium_proof_l422_422638

noncomputable def equilibrium_constant (Γ_eq B_eq : ℝ) : ℝ :=
(Γ_eq ^ 3) / (B_eq ^ 3)

theorem equilibrium_proof (Γ_eq B_eq : ℝ) (K_c : ℝ) (B_initial : ℝ) (Γ_initial : ℝ)
  (hΓ : Γ_eq = 0.25) (hB : B_eq = 0.15) (hKc : K_c = 4.63) 
  (ratio : Γ_eq = B_eq + B_initial) (hΓ_initial : Γ_initial = 0) :
  equilibrium_constant Γ_eq B_eq = K_c ∧ 
  B_initial = 0.4 ∧ 
  Γ_initial = 0 := 
by
  sorry

end equilibrium_proof_l422_422638


namespace no_nat_nums_satisfy_gcd_lcm_condition_l422_422868

theorem no_nat_nums_satisfy_gcd_lcm_condition :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y = x + y + 2021 := 
sorry

end no_nat_nums_satisfy_gcd_lcm_condition_l422_422868


namespace trapezoid_PB_length_l422_422877

theorem trapezoid_PB_length
  (A B C D P : Type)
  (AD BC AP PD : ℝ)
  (h1 : AD = 24 * √2)
  (h2 : AP = 8 * √2)
  (h3 : PD = 16 * √2)
  (h4 : ∀{x : ℝ}, is_trapezoid A B C D x)
  (h5 : ∠ B A C = π / 4)
  (h6 : ∀{x : ℝ}, diagonal_length A B C D x = 20 * √2) :
  ∀{y : ℝ}, line_segment_length P B y = 8 * √2 := sorry

end trapezoid_PB_length_l422_422877


namespace function_in_second_quadrant_l422_422856

theorem function_in_second_quadrant (k : ℝ) : (∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → (k / x₁ < k / x₂)) → (∀ x : ℝ, x < 0 → (k > 0)) :=
sorry

end function_in_second_quadrant_l422_422856


namespace sum_seven_numbers_l422_422855

theorem sum_seven_numbers (
  p q r s t u v : ℕ
  (h1 : p + q + r = 35)
  (h2 : q + r + s = 35)
  (h3 : r + s + t = 35)
  (h4 : s + t + u = 35)
  (h5 : t + u + v = 35)
  (h6 : q + u = 15)
) : p + q + r + s + t + u + v = 90 :=
sorry

end sum_seven_numbers_l422_422855


namespace concert_tickets_l422_422963

theorem concert_tickets (A C : ℕ) (h1 : C = 3 * A) (h2 : 7 * A + 3 * C = 6000) : A + C = 1500 :=
by {
  -- Proof omitted
  sorry
}

end concert_tickets_l422_422963


namespace min_value_of_linear_expression_l422_422935

theorem min_value_of_linear_expression {x y : ℝ} (h1 : 2 * x - y ≥ 0) (h2 : x + y - 3 ≥ 0) (h3 : y - x ≥ 0) :
  ∃ z, z = 2 * x + y ∧ z = 4 := by
  sorry

end min_value_of_linear_expression_l422_422935


namespace john_travel_remaining_money_l422_422132

-- Defining the parameters based on the problem conditions
def initial_amount : ℝ := 1600
def spent_first_day (X : ℝ) := X
def exchange_rate : ℝ := 0.85
def spent_second_day_eur : ℝ := 150
def spent_in_usd (X Y : ℝ) := X + Y
def money_left (initial spent : ℝ) := initial - spent
def final_condition (Y : ℝ) := spent_in_usd Y 176.47 - 600 = 500

-- Translating the problem conditions to Lean definitions
def john_trip (X : ℝ) : Prop :=
  let spent_usd := spent_in_usd X (spent_second_day_eur / exchange_rate)
  money_left initial_amount spent_usd = final_condition X

-- Stating the theorem to be proved
theorem john_travel_remaining_money : ∃ (X : ℝ), john_trip X :=
sorry

end john_travel_remaining_money_l422_422132


namespace packaging_cost_per_bar_calculation_l422_422919

def cost_of_packaging_per_bar (cost_of_chocolates_per_bar : ℝ) (number_of_bars : ℕ) (selling_price : ℝ) (profit : ℝ) : ℝ :=
  let total_cost_of_chocolates := cost_of_chocolates_per_bar * number_of_bars
  let total_cost_including_packaging := selling_price - profit
  let total_packaging_cost := total_cost_including_packaging - total_cost_of_chocolates
  total_packaging_cost / number_of_bars

theorem packaging_cost_per_bar_calculation :
  cost_of_packaging_per_bar 5 5 90 55 = 2 :=
by
  unfold cost_of_packaging_per_bar
  simp
  sorry

end packaging_cost_per_bar_calculation_l422_422919


namespace hyperbola_eccentricity_l422_422734

theorem hyperbola_eccentricity :
  let c := sqrt 3
  let P := (sqrt 2, -sqrt 2)
  let e := sqrt 3
  ∃ (a b : ℝ), (a^2 + b^2 = 3) ∧ 
               (2 / a^2 - 2 / b^2 = 1) ∧
               (e = c / a) := 
by
  sorry

end hyperbola_eccentricity_l422_422734


namespace fibonacci_pythagorean_l422_422192

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Define what it means to be a Pythagorean hypotenuse
def is_pythagorean_hypotenuse (c : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = c^2

-- The main theorem statement we want to prove
theorem fibonacci_pythagorean (k : ℕ) (h : k ≥ 2) : 
  is_pythagorean_hypotenuse (fibonacci (2*k + 1)) :=
sorry

end fibonacci_pythagorean_l422_422192


namespace solve_linear_system_l422_422182

theorem solve_linear_system :
  ∃ (x y : ℚ), (4 * x - 3 * y = 2) ∧ (6 * x + 5 * y = 1) ∧ (x = 13 / 38) ∧ (y = -4 / 19) :=
by
  sorry

end solve_linear_system_l422_422182


namespace a_sequence_arithmetic_sum_of_bn_l422_422863

   noncomputable def a (n : ℕ) : ℕ := 1 + n

   def S (n : ℕ) : ℕ := n * (n + 1) / 2

   def b (n : ℕ) : ℚ := 1 / S n

   def T (n : ℕ) : ℚ := (Finset.range n).sum b

   theorem a_sequence_arithmetic (n : ℕ) (a_n_positive : ∀ n, a n > 0)
     (a₁_is_one : a 0 = 1) :
     (a (n+1)) - a n = 1 := by
     sorry

   theorem sum_of_bn (n : ℕ) :
     T n = 2 * n / (n + 1) := by
     sorry
   
end a_sequence_arithmetic_sum_of_bn_l422_422863


namespace snowballs_made_by_brother_l422_422130

/-- Janet makes 50 snowballs and her brother makes the remaining snowballs. Janet made 25% of the total snowballs. 
    Prove that her brother made 150 snowballs. -/
theorem snowballs_made_by_brother (total_snowballs : ℕ) (janet_snowballs : ℕ) (fraction_janet : ℚ)
  (h1 : janet_snowballs = 50) (h2 : fraction_janet = 25 / 100) (h3 : janet_snowballs = fraction_janet * total_snowballs) :
  total_snowballs - janet_snowballs = 150 :=
by
  sorry

end snowballs_made_by_brother_l422_422130


namespace sum_square_ends_same_digit_l422_422546

theorem sum_square_ends_same_digit {a b : ℤ} (h : (a + b) % 10 = 0) :
  (a^2 % 10) = (b^2 % 10) :=
by
  sorry

end sum_square_ends_same_digit_l422_422546


namespace enclosing_polygon_sides_l422_422313

theorem enclosing_polygon_sides (m : ℕ) (h_m : m = 12) (n : ℕ) 
  (h_conditions : ∀ (i : ℕ), i < m → Polygon (n) is around Polygon (12)) :
  n = 6 :=
sorry

end enclosing_polygon_sides_l422_422313


namespace graph_shift_left_by_pi_over_12_l422_422585

theorem graph_shift_left_by_pi_over_12 :
  ∀ x : ℝ, 2 * sin (3 * (x + (π / 12))) = 2 * sin (3 * x + (π / 4)) := by
  intro x
  -- Skipping the proof
  sorry

end graph_shift_left_by_pi_over_12_l422_422585


namespace problem_solution_l422_422881

noncomputable def bowtie (a b : ℝ) : ℝ :=
  a ^ 2 + Real.sqrt (b ^ 2 + Real.sqrt (b ^ 2 + Real.sqrt (b ^ 2 + Real.sqrt (b ^ 2 + Real.sqrt (b ^ 2 + Real.sqrt (b ^ 2 + Real.sqrt (b ^ 2 + Real.sqrt (b ^ 2 + Real.sqrt (b ^ 2 + 0))))))))))

theorem problem_solution (y : ℝ) : bowtie 3 y = 18 → y = 6 * Real.sqrt 2 ∨ y = -6 * Real.sqrt 2 :=
by
  sorry

end problem_solution_l422_422881


namespace polygon_sides_from_interior_angles_l422_422954

theorem polygon_sides_from_interior_angles (S : ℕ) (h : S = 1260) : S = (9 - 2) * 180 :=
by
  sorry

end polygon_sides_from_interior_angles_l422_422954


namespace probability_tenth_ball_black_l422_422278

theorem probability_tenth_ball_black :
  let total_balls := 30
  let black_balls := 4
  let red_balls := 7
  let yellow_balls := 5
  let green_balls := 6
  let white_balls := 8
  (black_balls / total_balls) = 4 / 30 :=
by sorry

end probability_tenth_ball_black_l422_422278


namespace sqrt_sum_of_powers_l422_422598

theorem sqrt_sum_of_powers : sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  have h : 4^4 = 256 := by
    calc
      4^4 = 4 * 4 * 4 * 4 := by rw [show 4^4 = 4 * 4 * 4 * 4 from rfl]
      ... = 16 * 16       := by rw [mul_assoc, mul_assoc]
      ... = 256           := by norm_num
  calc
    sqrt (4^4 + 4^4 + 4^4 + 4^4)
      = sqrt (256 + 256 + 256 + 256) := by rw [h, h, h, h]
      ... = sqrt 1024                 := by norm_num
      ... = 32                        := by norm_num

end sqrt_sum_of_powers_l422_422598


namespace largest_good_number_number_of_absolute_good_numbers_l422_422826

-- Conditions for a good number
def is_good_number (N a b : ℕ) : Prop :=
  N = ab + a + b

-- Conditions for an absolute good number (case when ab/(a+b) = 3 or 4)
def is_absolute_good_number (N a b : ℕ) : Prop :=
  is_good_number N a b ∧ (ab = 3 * (a + b) ∨ ab = 4 * (a + b))

-- Prove that the largest good number is 99
theorem largest_good_number :
  ∃ N, (∃ a b, is_good_number N a b) ∧ 10 ≤ N ∧ N ≤ 99 ∧ ∀ M, (∃ a b, is_good_number M a b) ∧ 10 ≤ M ∧ M ≤ 99 → M ≤ N :=
  sorry

-- Prove that the number of absolute good numbers is 39
theorem number_of_absolute_good_numbers :
  ∃ count, count = 39 ∧ count = (∑ N in (finset.Icc 10 99), (∃ a b, is_absolute_good_number N a b)) :=
  sorry

end largest_good_number_number_of_absolute_good_numbers_l422_422826


namespace sequence_2023rd_letter_is_A_l422_422592

def sequence_pattern : List Char := ['A', 'B', 'C', 'D', 'E', 'D', 'C', 'B', 'A', 'A', 'B', 'C', 'E', 'D', 'C', 'B', 'A']

theorem sequence_2023rd_letter_is_A : 
  ∃ (n : ℕ) (S : List Char), S = sequence_pattern ∧ 2023 = n * S.length → S.nth ((2023 - 1) % S.length) = some 'A' :=
by
  let S := sequence_pattern
  have h : 2023 % S.length = 0,
  sorry
  exact ⟨119, S, rfl, by norm_num [S, List.nth_le, h]⟩

end sequence_2023rd_letter_is_A_l422_422592


namespace length_of_BC_l422_422092

theorem length_of_BC
  (ABC DEF : Triangle)
  (h_congruent : ABC ≅ DEF)
  (EF : ℝ)
  (h_EF : EF = 25) :
  (BC : ℝ) = 25 :=
sorry

end length_of_BC_l422_422092


namespace maximum_volume_pyramid_l422_422758

-- Define the points on the sphere and their geometric properties.
variable {O S A B C : Point} 
variable {r : ℝ}

-- Sphere conditions
def sphere_radius (O : Point) (r : ℝ) : Prop := r = 3 * Real.sqrt 2

-- Points on the surface of the sphere
def on_sphere_surface (S A B C : Point) (O : Point) (r : ℝ) : Prop :=
  dist O S = r ∧ dist O A = r ∧ dist O B = r ∧ dist O C = r

-- Angle and distance conditions
def angle_ABC_90 (A B C : Point) : Prop := ∠ABC = Real.pi / 2
def distances_AB_BC (A B C : Point) : Prop := dist A B = 4 * Real.sqrt 2 ∧ dist B C = 4 * Real.sqrt 2

-- Main proof obligation
theorem maximum_volume_pyramid
  (O S A B C : Point)
  (r : ℝ)
  (h : ℝ)
  (V : ℝ)
  (hsphere : sphere_radius O r)
  (hsurf : on_sphere_surface S A B C O r)
  (hangle : angle_ABC_90 A B C)
  (hdist : distances_AB_BC A B C)
  : V = (64 * Real.sqrt 2) / 3 :=
by
  sorry

end maximum_volume_pyramid_l422_422758


namespace max_area_ACDB_l422_422904

-- Definitions for semicircle, points on semicircle, etc.
noncomputable def radius := sorry 
noncomputable def Point := sorry

-- Assume existence of points A, B, C, D on the semicircle
variables (A B C D : Point)
-- Assume conditions described in the problem
axiom semicircle : ∀ (O : Point) (R : ℝ), true

-- Define the area function for quadrilateral ACDB
noncomputable def area_ACDB : ℝ := sorry

-- Theorem statement
theorem max_area_ACDB :
  -- condition that C and D divide the semicircle into three equal parts
  divides_into_three_equal_parts C D →
  -- maximum area condition
  area_ACDB = (3 * real.sqrt 3 / 4) * radius^2 :=
sorry

end max_area_ACDB_l422_422904


namespace tangent_line_equiv_l422_422464

variable {a b : ℝ}

theorem tangent_line_equiv {a b : ℝ} 
  (curve : ∀ x:ℝ, y:ℝ, y = x^2 + a * x + b)
  (tangent : ∀ x:ℝ, y:ℝ, x - y + 1 = 0)
  (tangent_point : curve 0 b)
  (derivative : ∀ x:ℝ, y:ℝ, (derivative_curve : Deriv (λ x, x^2 + a * x + b) = 1) ) :
  a = 1 ∧ b = 1 := 
by 
  sorry

end tangent_line_equiv_l422_422464


namespace f_tan_squared_l422_422187

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 ∨ x = 1 then 0 else (λ y, if y = x / (x - 1) then (1 : ℝ) / x else 0) x
  
theorem f_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π / 2) :
  f (tan t ^ 2) = -cos (2 * t) / sin t ^ 2 := by
  sorry

end f_tan_squared_l422_422187


namespace lcm_of_18_and_20_l422_422738

theorem lcm_of_18_and_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_of_18_and_20_l422_422738


namespace problem_solution_l422_422925

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem problem_solution :
  let interval_start := 10^(2014!) * Real.pi
  let interval_end := 10^(2014!+2018) * Real.pi
  let count_solutions := 3 * (10^(2014!-1)) * (10^2018 - 1)
  sum_of_digits count_solutions = 18162 :=
by
  let interval_start := 10^(2014!) * Real.pi
  let interval_end := 10^(2014!+2018) * Real.pi
  let count_solutions := 3 * (10^(2014!-1)) * (10^2018 - 1)
  sorry

end problem_solution_l422_422925


namespace volume_of_P3_l422_422760

structure PolyhedronSequence (P : ℕ → ℝ) : Prop :=
(volume_P0 : P 0 = 1)
(volume_reduction : ∀ (i : ℕ), 
  ∃ k : ℕ → ℝ,
    (∀ j < 4^{i}, P (i + 1) = P i + k j * (1 / 4)))

theorem volume_of_P3 : 
  ∀ P : ℕ → ℝ,
    PolyhedronSequence P →
    P 3 = 5 / 2 :=
by
  intros P sequence
  cases sequence with h0 h1
  rw h0
  -- Further elaboration would be needed to complete the proof, replacing "sorry" with a correct step when ready.
  sorry

end volume_of_P3_l422_422760


namespace female_employees_sampled_l422_422654

theorem female_employees_sampled
  (T : ℕ) -- Total number of employees
  (M : ℕ) -- Number of male employees
  (F : ℕ) -- Number of female employees
  (S_m : ℕ) -- Number of sampled male employees
  (H_T : T = 140)
  (H_M : M = 80)
  (H_F : F = 60)
  (H_Sm : S_m = 16) :
  ∃ S_f : ℕ, S_f = 12 :=
by
  sorry

end female_employees_sampled_l422_422654


namespace sqrt_sum_of_powers_l422_422597

theorem sqrt_sum_of_powers : sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  have h : 4^4 = 256 := by
    calc
      4^4 = 4 * 4 * 4 * 4 := by rw [show 4^4 = 4 * 4 * 4 * 4 from rfl]
      ... = 16 * 16       := by rw [mul_assoc, mul_assoc]
      ... = 256           := by norm_num
  calc
    sqrt (4^4 + 4^4 + 4^4 + 4^4)
      = sqrt (256 + 256 + 256 + 256) := by rw [h, h, h, h]
      ... = sqrt 1024                 := by norm_num
      ... = 32                        := by norm_num

end sqrt_sum_of_powers_l422_422597


namespace sum_not_integer_l422_422541

open BigOperators

theorem sum_not_integer : 
  ¬ ∃ (s : ℤ), ∑ m in finset.Ico 2 1986, ∑ n in finset.Ico (m + 1) 1986, (1 : ℚ) / (m * n : ℚ) = s :=
sorry

end sum_not_integer_l422_422541


namespace tan_double_angle_l422_422038

theorem tan_double_angle (α β : ℝ) (h1 : Real.tan (α + β) = 7) (h2 : Real.tan (α - β) = 1) : 
  Real.tan (2 * α) = -4/3 :=
by
  sorry

end tan_double_angle_l422_422038


namespace line_not_thru_second_quadrant_l422_422217

theorem line_not_thru_second_quadrant (a : ℝ) : 
  (a ≥ 2 → ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ (a-2) * y = (3a-1) * x - 4)) ∧
  (a < 2 → (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ (a-2) * y = (3a-1) * x - 4)) := 
by
  sorry

end line_not_thru_second_quadrant_l422_422217


namespace min_x1_l422_422793

def f (x : ℝ) : ℝ :=
  if x > 0 then 3^x - 1 else x^2 + 1

theorem min_x1 (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 ≤ 0) (h_eq : f x1 = f x2) : x1 ≥ Real.log 2 / Real.log 3 := by
  sorry

end min_x1_l422_422793


namespace polygon_with_interior_sum_1260_eq_nonagon_l422_422952

theorem polygon_with_interior_sum_1260_eq_nonagon :
  ∃ n : ℕ, (n-2) * 180 = 1260 ∧ n = 9 := by
  sorry

end polygon_with_interior_sum_1260_eq_nonagon_l422_422952


namespace simplify_polynomial_l422_422547

theorem simplify_polynomial :
  (6 * p ^ 4 + 2 * p ^ 3 - 8 * p + 9) + (-3 * p ^ 3 + 7 * p ^ 2 - 5 * p - 1) = 
  6 * p ^ 4 - p ^ 3 + 7 * p ^ 2 - 13 * p + 8 :=
by
  sorry

end simplify_polynomial_l422_422547


namespace total_decorations_l422_422717

theorem total_decorations 
  (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) (pumpkins : ℕ) 
  (cauldron : ℕ) (budget_decorations : ℕ) (left_decorations : ℕ)
  (h_skulls : skulls = 12)
  (h_broomsticks : broomsticks = 4)
  (h_spiderwebs : spiderwebs = 12)
  (h_pumpkins : pumpkins = 2 * spiderwebs)
  (h_cauldron : cauldron = 1)
  (h_budget_decorations : budget_decorations = 20)
  (h_left_decorations : left_decorations = 10) : 
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_decorations + left_decorations = 83 := 
by 
  sorry

end total_decorations_l422_422717


namespace inequality_proof_l422_422538

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a ^ 2 + 1)) + (b / Real.sqrt (b ^ 2 + 1)) + (c / Real.sqrt (c ^ 2 + 1)) ≤ (3 / 2) :=
by
  sorry

end inequality_proof_l422_422538


namespace mean_median_difference_l422_422579

def drops : List ℝ := [180, 150, 210, 195, 165, 230]

def mean (l : List ℝ) : ℝ := l.sum / l.length

def median (l : List ℝ) : ℝ := 
  let sorted := l.qsort (· < ·)
  let len := l.length
  if len % 2 = 0 then
    (sorted.get! (len / 2 - 1) + sorted.get! (len / 2)) / 2
  else
    sorted.get! (len / 2)

def positive_difference (a b : ℝ) : ℝ := abs (a - b)

theorem mean_median_difference :
  positive_difference (mean drops) (median drops) = 0.83 := by
  sorry

end mean_median_difference_l422_422579


namespace length_AC_correct_l422_422118

noncomputable def length_AC : ℝ := 
  let AB := 15.0
  let DC := 24.0
  let AD := 9.0
  let BD := Real.sqrt (AB^2 - AD^2)
  let BC := Real.sqrt (DC^2 - BD^2)
  let AE := AD + BC
  Real.sqrt (AE^2 + BD^2)

theorem length_AC_correct : round (length_AC * 10) / 10 = 30.7 :=
by
  have h_AB : AB = 15 := rfl
  have h_DC : DC = 24 := rfl
  have h_AD : AD = 9 := rfl
  have h_BD : BD = Real.sqrt (AB^2 - AD^2) := rfl
  have h_BC : BC = Real.sqrt (DC^2 - BD^2) := rfl
  have h_AE : AE = AD + BC := rfl
  have h_AC : length_AC = Real.sqrt (AE^2 + BD^2) := rfl
  sorry

end length_AC_correct_l422_422118


namespace work_together_l422_422628

variable (W : ℝ) -- 'W' denotes the total work
variable (a_days b_days c_days : ℝ)

-- Conditions provided in the problem
axiom a_work : a_days = 18
axiom b_work : b_days = 6
axiom c_work : c_days = 12

-- The statement to be proved
theorem work_together :
  (W / a_days + W / b_days + W / c_days) * (36 / 11) = W := by
  sorry

end work_together_l422_422628


namespace trains_meeting_point_l422_422235

-- Definitions
def speed_A_kmph := 72
def speed_B_kmph := 54
def time_A_seconds := 7
def time_B_seconds := 9
def distance_km := 162

-- Proof statement
theorem trains_meeting_point :
  let speed_A_mps := speed_A_kmph * 1000 / 3600 in
  let speed_B_mps := speed_B_kmph * 1000 / 3600 in
  let length_A := speed_A_mps * time_A_seconds in
  let length_B := speed_B_mps * time_B_seconds in
  let relative_speed := speed_A_mps + speed_B_mps in
  let distance_m := distance_km * 1000 in
  let time_to_meet := distance_m / relative_speed in
  let distance_A_m := speed_A_mps * time_to_meet in
  let meeting_point_km := distance_A_m / 1000 in
  length_A = 140 ∧ length_B = 135 ∧ meeting_point_km ≈ 92.5714 := 
by
  sorry

end trains_meeting_point_l422_422235


namespace parking_cost_l422_422179

def sally_savings : ℕ := 28
def park_entry_cost : ℕ := 55
def meal_pass_cost : ℕ := 25
def one_way_distance : ℕ := 165
def miles_per_gallon : ℕ := 30
def gas_cost_per_gallon : ℕ := 3
def additional_savings_needed : ℕ := 95

theorem parking_cost :
  let total_distance := 2 * one_way_distance,
      total_gallons := total_distance / miles_per_gallon,
      gas_cost := total_gallons * gas_cost_per_gallon,
      total_cost_without_parking := gas_cost + park_entry_cost + meal_pass_cost,
      total_amount_needed := sally_savings + additional_savings_needed,
      parking_cost := total_amount_needed - total_cost_without_parking
  in parking_cost = 10 :=
by
  let total_distance := 2 * one_way_distance
  let total_gallons := total_distance / miles_per_gallon
  let gas_cost := total_gallons * gas_cost_per_gallon
  let total_cost_without_parking := gas_cost + park_entry_cost + meal_pass_cost
  let total_amount_needed := sally_savings + additional_savings_needed
  let parking_cost := total_amount_needed - total_cost_without_parking
  show parking_cost = 10
  sorry

end parking_cost_l422_422179


namespace garden_perimeter_l422_422311

theorem garden_perimeter (L B : ℕ) (hL : L = 100) (hB : B = 200) : 
  2 * (L + B) = 600 := by
sorry

end garden_perimeter_l422_422311


namespace num_C_atoms_in_compound_l422_422293

def num_H_atoms := 6
def num_O_atoms := 1
def molecular_weight := 58
def atomic_weight_C := 12
def atomic_weight_H := 1
def atomic_weight_O := 16

theorem num_C_atoms_in_compound : 
  ∃ (num_C_atoms : ℕ), 
    molecular_weight = (num_C_atoms * atomic_weight_C) + (num_H_atoms * atomic_weight_H) + (num_O_atoms * atomic_weight_O) ∧ 
    num_C_atoms = 3 :=
by
  -- To be proven
  sorry

end num_C_atoms_in_compound_l422_422293


namespace factorial_expression_value_l422_422705

theorem factorial_expression_value :
  (13.factorial - 11.factorial + 9.factorial) / 10.factorial = 17051 / 10 :=
by
  sorry

end factorial_expression_value_l422_422705


namespace donut_selection_count_l422_422173

theorem donut_selection_count :
  let glazed := 0
  let chocolate := 0
  let powdered := 0
  let jelly := 0
  let donuts : ℕ := glazed + chocolate + powdered + jelly
  (finset.card (finset.range 10).powerset // assume elements are non-negative integers
  (finset.range 10).powerset.filter (λ s, 
                                     donut_types.val.sum = donuts + 6 = 84

-- where donut_types, val.glazed, val.chocolate, val.powdered, val.jelly 
 "test_proof.sorry")

end donut_selection_count_l422_422173


namespace calculate_interest_2520K_320days_l422_422937

variables (principal : ℝ) (days : ℕ) (interest_rate : ℝ)

def calculate_net_interest : ℝ :=
  let base_interest := (principal * (days : ℝ)) / 10000 in
  base_interest - (base_interest / 8)

theorem calculate_interest_2520K_320days :
  calculate_net_interest 2520 320 3.15 = 70.56 :=
by
  unfold calculate_net_interest
  -- The proofs go here 
  sorry

end calculate_interest_2520K_320days_l422_422937


namespace lcm_18_20_l422_422735

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_18_20_l422_422735


namespace opposite_of_8_is_neg8_l422_422572

theorem opposite_of_8_is_neg8 : ∃ y : ℤ, 8 + y = 0 ∧ y = -8 := by
  use -8
  split
  ·
    sorry

  ·
    rfl

end opposite_of_8_is_neg8_l422_422572


namespace tanC_over_tanA_plus_tanC_over_tanB_l422_422844

theorem tanC_over_tanA_plus_tanC_over_tanB {a b c : ℝ} (A B C : ℝ) (h : a / b + b / a = 6 * Real.cos C) (acute_triangle : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) :
  (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 4 :=
sorry -- Proof not required

end tanC_over_tanA_plus_tanC_over_tanB_l422_422844


namespace minimize_cost_l422_422662

noncomputable def shipping_cost (x : ℝ) : ℝ := 5 * x
noncomputable def storage_cost (x : ℝ) : ℝ := 20 / x
noncomputable def total_cost (x : ℝ) : ℝ := shipping_cost x + storage_cost x

theorem minimize_cost : ∃ x : ℝ, x = 2 ∧ total_cost x = 20 :=
by
  use 2
  unfold total_cost
  unfold shipping_cost
  unfold storage_cost
  sorry

end minimize_cost_l422_422662


namespace smallest_possible_average_l422_422384

def smallest_average (s : Finset (Fin 10)) : ℕ :=
  (Finset.sum s).toNat / 6

theorem smallest_possible_average : ∃ (s1 s2 : Finset ℕ), (s1.card = 3 ∧ s2.card = 3 ∧ 
 (s1 ∪ s2).card = 6 ∧ ∀ x, x ∈ s1 ∪ s2 → x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  smallest_average (s1 ∪ s2) = 16.5) sorry

end smallest_possible_average_l422_422384


namespace complex_division_result_l422_422202

theorem complex_division_result : (1 + 2 * Complex.i) / (2 - Complex.i) = Complex.i := 
by
  sorry

end complex_division_result_l422_422202


namespace perfect_square_solution_l422_422019

theorem perfect_square_solution (x : ℤ) : 
  ∃ k : ℤ, x^2 - 14 * x - 256 = k^2 ↔ x = 15 ∨ x = -1 :=
by
  sorry

end perfect_square_solution_l422_422019


namespace x_sq_minus_y_sq_l422_422891

variable (a : ℤ)
variable (n : ℤ)
def x : ℤ := a^n - a^(-n)
def y : ℤ := a^n + a^(-n)

theorem x_sq_minus_y_sq (a : ℕ) (n : ℕ) (ha : a > 0) (hn : n > 0) :
    (x a n)^2 - (y a n)^2 = -4 := by
    sorry

end x_sq_minus_y_sq_l422_422891


namespace karlson_max_candies_l422_422959

theorem karlson_max_candies:
  let candies_eaten (n : ℕ) : ℕ :=
    if n = 0 then 0 else n * (n - 1) / 2
  in candies_eaten 31 = 465 :=
by
  sorry

end karlson_max_candies_l422_422959


namespace incorrect_transformation_D_l422_422999

theorem incorrect_transformation_D (x y m : ℝ) (hxy: x = y) : m = 0 → ¬ (x / m = y / m) :=
by
  intro hm
  simp [hm]
  -- Lean's simp tactic simplifies known equalities
  -- The simp tactic will handle the contradiction case directly when m = 0.
  sorry

end incorrect_transformation_D_l422_422999


namespace count_no_valid_n_l422_422086

theorem count_no_valid_n :
  (finset.count (λ n : ℤ, 5 ≤ n ∧ n ≤ 12 ∧ ∃ m : ℤ, (2 * n^2 + 3 * n + 2) = m^2)
                (finset.Icc 5 12).val) = 0 :=
by
  -- The proof goes here, but we will skip this part
  sorry

end count_no_valid_n_l422_422086


namespace heptagon_exterior_angles_sum_l422_422951

-- Defining a polygon and its properties of exterior angles
def is_polygon (n : ℕ) : Prop := n ≥ 3

def sum_of_exterior_angles (n : ℕ) : ℝ :=
  if is_polygon n then 360 else 0

-- Specific case for a heptagon with 7 sides
def is_heptagon : Prop := is_polygon 7

-- Theorem that needs to be proved
theorem heptagon_exterior_angles_sum : is_heptagon → sum_of_exterior_angles 7 = 360 :=
by
  intro h,
  unfold is_heptagon at h,
  unfold sum_of_exterior_angles,
  simp [is_polygon, h],
  sorry

end heptagon_exterior_angles_sum_l422_422951


namespace g_of_2_l422_422945

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : x * g y = 2 * y * g x 
axiom g_of_10 : g 10 = 5

theorem g_of_2 : g 2 = 2 :=
by
    sorry

end g_of_2_l422_422945


namespace distinct_grids_count_l422_422699

-- Define the grid and its labeling
def Grid (m n : Nat) := Array (Array Nat)

-- Define the condition that each number in the grid is between 1 and 2^10
def isValidNumber (x : Nat) : Prop := 1 ≤ x ∧ x ≤ 2^10

-- Define the condition for the sum of a row being divisible by 2^n
def rowSumDivisibleBy2Pow (grid : Grid 11 11) (n : Nat) : Prop := 
  let rowSum := (List.range 11).map (λ i => grid[n][i]).sum
  2^n ∣ rowSum

-- Define the condition for the sum of a column being divisible by 2^n
def colSumDivisibleBy2Pow (grid : Grid 11 11) (n : Nat) : Prop := 
  let colSum := (List.range 11).map (λ i => grid[i][n]).sum
  2^n ∣ colSum

-- Define the main function to check all conditions for the grid
def isValidGrid (grid : Grid 11 11) : Prop :=
  (List.range 11).all (λ n => rowSumDivisibleBy2Pow grid n ∧ colSumDivisibleBy2Pow grid n) ∧
  (grid.flatten.all isValidNumber)

/-- The number of possible distinct 11 × 11 grids satisfying the conditions is 2^1110 -/
theorem distinct_grids_count : ∑ grid in (validGrids 11 11), isValidGrid grid = 2^1110 := 
by
  sorry

end distinct_grids_count_l422_422699


namespace area_of_triangle_PF1F2_is_sqrt_15_l422_422857

noncomputable def area_triangle_PF1F2 
    (a b : ℝ)
    (h1 : a > b) 
    (h2 : b > 0)
    (P U S T V : ℝ × ℝ)
    (PU PS PV PT : ℝ)
    (h3 : PU = 1 ∧ PS = 2 ∧ PV = 3 ∧ PT = 6) 
    (h4 : ((U.1 - P.1)^2 + (U.2 - P.2)^2 = PU^2) ∧ 
          ((S.1 - P.1)^2 + (S.2 - P.2)^2 = PS^2) ∧ 
          ((V.1 - P.1)^2 + (V.2 - P.2)^2 = PV^2) ∧
          ((T.1 - P.1)^2 + (T.2 - P.2)^2 = PT^2))
    (h5 : P.1 = (|T.1 - S.1| / 2) ∧ P.2 = (|V.2 - U.2| / 2))
    (h6 : (P.1, P.2) = (2, 1)) :
    ℝ := 
begin
    let c := sqrt (a^2 - b^2),
    let F1F2 := 2 * c,
    (1 / 2) * F1F2 * P.2
end

theorem area_of_triangle_PF1F2_is_sqrt_15 
    (a b : ℝ)
    (h1 : a = sqrt 20) 
    (h2 : b = sqrt 5)
    (P : ℝ × ℝ)
    (h3 : P.1 = 2 ∧ P.2 = 1) :
    area_triangle_PF1F2 a b h1 h2 P U S T V PU PS PV PT h3 h4 h5 h6 = sqrt 15 :=
sorry

end area_of_triangle_PF1F2_is_sqrt_15_l422_422857


namespace milly_folds_count_l422_422529

theorem milly_folds_count (mixing_time baking_time total_minutes fold_time rest_time : ℕ) 
  (h : total_minutes = 360)
  (h_mixing_time : mixing_time = 10)
  (h_baking_time : baking_time = 30)
  (h_fold_time : fold_time = 5)
  (h_rest_time : rest_time = 75) : 
  (total_minutes - (mixing_time + baking_time)) / (fold_time + rest_time) = 4 := 
by
  sorry

end milly_folds_count_l422_422529


namespace total_area_correct_l422_422351

noncomputable def total_area_of_triangles (r : ℝ) : ℝ :=
  let Q := (0, 15 : ℝ)
  let A := (3, 15 : ℝ)
  let B := (15, 0 : ℝ)
  let C := (0, r)
  let area_QCA := (3 * (15 - r)) / 2
  let AC := real.sqrt (9 + (15 - r) ^ 2)
  let BC := real.sqrt (225 + r ^ 2)
  let area_ABC := (AC * BC) / 2
  area_QCA + area_ABC

theorem total_area_correct (r : ℝ) :
  total_area_of_triangles r =
  (45 / 2) - (3 * r / 2) + (1 / 2) * real.sqrt (9 + (15 - r) ^ 2) * real.sqrt (225 + r ^ 2) :=
by
  sorry

end total_area_correct_l422_422351


namespace locus_of_foci_l422_422031

open_locale classical

variables {P : Type} [metric_space P] [finite_dimensional ℝ P]
variables (d e1 e2 : affine_subspace ℝ P)
variables {M F : P}

def is_directrix (d : affine_subspace ℝ P) (p : parabola ℝ P) : Prop := sorry
def is_focus (F : P) (p : parabola ℝ P) : Prop := sorry
def is_tangent (e : affine_subspace ℝ P) (p : parabola ℝ P) : Prop := sorry
def symmetric_line (d : affine_subspace ℝ P) (p : parabola ℝ P) : affine_subspace ℝ P := sorry

axiom criterion (d e1 e2 : affine_subspace ℝ P) (M : P) :
  e1 ⊆ d ∧ e2 ⊆ d ∧ e1 ≠ e2 ∧ ∀ x ∈ e1, y ∈ e2, x ≠ y → dist M x = dist M y

theorem locus_of_foci {d e1 e2 : affine_subspace ℝ P} {M : P} (hd : is_directrix d) 
  (he1 : is_tangent e1) (he2 : is_tangent e2)
  (hyp : criterion d e1 e2 M) :
  ∃ f : affine_subspace ℝ P, ∀ (F : P), is_focus F → F ∈ f ∧ symmetric f d :=
begin
  sorry
end

end locus_of_foci_l422_422031


namespace sqrt_sum_of_four_terms_of_4_pow_4_l422_422614

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l422_422614


namespace kids_from_lawrence_county_go_to_camp_l422_422728

theorem kids_from_lawrence_county_go_to_camp : 
  (1201565 - 590796 = 610769) := 
by
  sorry

end kids_from_lawrence_county_go_to_camp_l422_422728


namespace find_n_for_polygon_l422_422687

def regular_polygon_inscribed {R : ℝ} (hR : R ≠ 0) : ℕ :=
  let A := 6 * R^2
  let formula_area (n : ℕ) := 2 * n * R^2 * Real.sin (360 / n * Real.pi / 180)
  ∃ n : ℕ, formula_area n = A ∧ n = 12

theorem find_n_for_polygon {R : ℝ} (hR : R ≠ 0) :
  regular_polygon_inscribed hR = 12 :=
sorry

end find_n_for_polygon_l422_422687


namespace inequality_positive_real_numbers_l422_422536

theorem inequality_positive_real_numbers
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a * b + b * c + c * a = 1) :
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ (3 / 2) :=
  sorry

end inequality_positive_real_numbers_l422_422536


namespace sum_of_coefficients_l422_422034

theorem sum_of_coefficients :
  let f := (1 + x)^6 in
  let a := f.coeff 0 in
  let a_1 := f.coeff 1 in
  let a_2 := f.coeff 2 in
  let a_3 := f.coeff 3 in
  let a_4 := f.coeff 4 in
  let a_5 := f.coeff 5 in
  let a_6 := f.coeff 6 in
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 63 :=
by
  sorry

end sum_of_coefficients_l422_422034


namespace cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l422_422649

section price_calculations

variables {x : ℕ} (hx : x > 20)

-- Definitions based on the problem statement.
def suit_price : ℕ := 400
def tie_price : ℕ := 80

def option1_cost (x : ℕ) : ℕ :=
  20 * suit_price + tie_price * (x - 20)

def option2_cost (x : ℕ) : ℕ :=
  (20 * suit_price + tie_price * x) * 9 / 10

def option1_final_cost := option1_cost 30
def option2_final_cost := option2_cost 30

def optimal_cost : ℕ := 20 * suit_price + tie_price * 10 * 9 / 10

-- Proof obligations
theorem cost_option1_eq : option1_cost x = 80 * x + 6400 :=
by sorry

theorem cost_option2_eq : option2_cost x = 72 * x + 7200 :=
by sorry

theorem option1_final_cost_eq : option1_final_cost = 8800 :=
by sorry

theorem option2_final_cost_eq : option2_final_cost = 9360 :=
by sorry

theorem option1_more_cost_effective : option1_final_cost < option2_final_cost :=
by sorry

theorem optimal_cost_eq : optimal_cost = 8720 :=
by sorry

end price_calculations

end cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l422_422649


namespace dice_probability_l422_422745

-- Define the conditions for the dice roll problem
def diceRollCondition : Prop :=
  ∃ a b c : ℕ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c)

-- Formalize the probability statement
theorem dice_probability (h : diceRollCondition) : 
  let P := 1 / 3 in 
  P = 1 / 3 :=
by
  sorry

end dice_probability_l422_422745


namespace roots_of_quadratic_are_1_and_minus2_l422_422121

noncomputable def roots_of_quadratic 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (h1 : a + b + c = 0) 
  (h2 : 4 * a - 2 * b + c = 0) : set ℝ :=
  {x : ℝ | a * x^2 + b * x + c = 0}

theorem roots_of_quadratic_are_1_and_minus2 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (h1 : a + b + c = 0) 
  (h2 : 4 * a - 2 * b + c = 0) : 
  roots_of_quadratic a b c ha h1 h2 = {1, -2} :=
by 
  sorry

end roots_of_quadratic_are_1_and_minus2_l422_422121


namespace math_teacher_daughter_age_l422_422163

theorem math_teacher_daughter_age (P : ℤ → ℤ) (a p : ℤ) [hp : Fact (Nat.Prime (Int.natAbs p))] :
  (∀ x, polynomial.integerCoeff P x) → -- P(x) has integer coefficients
  P a = a → -- P(a) = a
  P 0 = p → -- P(0) = p
  p > a → -- p > a
  a = 1 := sorry

end math_teacher_daughter_age_l422_422163


namespace marble_count_l422_422276

noncomputable def total_marbles (blue red white: ℕ) : ℕ := blue + red + white

theorem marble_count (W : ℕ) (h_prob : (9 + W) / (6 + 9 + W : ℝ) = 0.7) : 
  total_marbles 6 9 W = 20 :=
by
  sorry

end marble_count_l422_422276


namespace original_price_of_petrol_l422_422312

noncomputable def original_price (P : ℝ) : Prop :=
  let new_price := 0.9 * P in
  let gallons_more := 5 in
  let amount := 280 in
  amount / new_price = amount / P + gallons_more

theorem original_price_of_petrol (P : ℝ) : original_price P → P ≈ 6.22 :=
by
  sorry

end original_price_of_petrol_l422_422312


namespace hernandez_state_tax_l422_422899

-- Conditions as definitions
def monthsResident := 9
def totalMonths := 12
def taxableIncome := 42500
def taxRate := 0.04

-- Compute the proportion of the year Mr. Hernandez was a resident
def residentProportion := (monthsResident : ℝ) / totalMonths

-- Compute the prorated taxable income
def proratedIncome := taxableIncome * residentProportion

-- Statement to prove the tax amount
theorem hernandez_state_tax : proratedIncome * taxRate = 1275 := by
  sorry

end hernandez_state_tax_l422_422899


namespace maximum_value_inequality_l422_422155

theorem maximum_value_inequality (a b c : ℝ) 
  (h₁ : a + b + c = 2) 
  (h₂ : a ≥ -1/2) 
  (h₃ : b ≥ -1) 
  (h₄ : c ≥ -2) : 
  sqrt (4*a + 2) + sqrt (4*b + 4) + sqrt (4*c + 8) ≤ sqrt 66 :=
by
  sorry

end maximum_value_inequality_l422_422155


namespace kmph_to_mps_l422_422283

theorem kmph_to_mps (s : ℝ) (h : s = 0.975) : s * (1000 / 3600) = 0.2708 := by
  -- We include the assumption s = 0.975 as part of the problem condition.
  -- Import Mathlib to gain access to real number arithmetic.
  -- sorry is added to indicate a place where the proof should go.
  sorry

end kmph_to_mps_l422_422283


namespace area_ratio_of_concentric_circles_l422_422981

noncomputable theory

-- Define the given conditions
def C1 (r1 : ℝ) : ℝ := 2 * Real.pi * r1
def C2 (r2 : ℝ) : ℝ := 2 * Real.pi * r2
def arc_length (angle : ℝ) (circumference : ℝ) : ℝ := (angle / 360) * circumference

-- Lean statement for the math proof problem
theorem area_ratio_of_concentric_circles 
  (r1 r2 : ℝ) (h₁ : arc_length 60 (C1 r1) = arc_length 48 (C2 r2)) : 
  (Real.pi * r1^2) / (Real.pi * r2^2) = 16 / 25 :=
by
  sorry  -- Proof omitted

end area_ratio_of_concentric_circles_l422_422981


namespace range_of_a_l422_422462

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x ^ 2 + 2 * x + 1)

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, a * x ^ 2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l422_422462


namespace part1_values_a_b_part2i_b_geq_part2ii_range_a_l422_422054

variables (a b x : ℝ)

def f (x : ℝ) (a : ℝ) (b : ℝ) := x^2 - 2 * a * x + b
def g (x : ℝ) (a : ℝ) := x - a
def h (x : ℝ) (a : ℝ) (b : ℝ) := (f x a b + g x a - abs (f x a b - g x a)) / 2

-- Part (1)
theorem part1_values_a_b :
  (f (-3) a b = a) ∧ (f a -a^2+b = -3) → a = -2 ∧ b = 1 := sorry

-- Part (2)(i)
theorem part2i_b_geq :
  (∀ x, h x a b = g x a) → b - a^2 ≥ 1/4 := sorry

-- Part (2)(ii)
theorem part2ii_range_a :
  (∀ x, (∃ x, h x a b = a) ∧ unique_roots (h x a b)) → a < -(1:ℝ) + sqrt 5 / 2 := sorry

end part1_values_a_b_part2i_b_geq_part2ii_range_a_l422_422054


namespace sqrt_sum_of_four_terms_of_4_pow_4_l422_422612

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l422_422612


namespace express_in_scientific_notation_l422_422269

theorem express_in_scientific_notation : (0.000021 : ℝ) = 2.1 * 10^(-5) := 
sorry

end express_in_scientific_notation_l422_422269


namespace both_savings_2000_l422_422219

variables (P1 P2 : Type) (I1 I2 E1 E2 : ℕ)
variable (x : ℕ)

def income_ratio : Prop := I1 / I2 = 5 / 4
def expenditure_ratio : Prop := E1 / E2 = 3 / 2
def income_P1 : Prop := I1 = 5000
def expenditure_common_multiple : Prop := E1 = 3 * x ∧ E2 = 2 * x
def savings (I E : ℕ) : ℕ := I - E

theorem both_savings_2000
  (ir : income_ratio P1 P2 I1 I2)
  (er : expenditure_ratio P1 P2 E1 E2)
  (ip1 : income_P1 P1 I1)
  (ecm : expenditure_common_multiple P1 P2 E1 E2 x)
  (equal_savings : savings I1 E1 = savings I2 E2) :
  savings I1 E1 = 2000 ∧ savings I2 E2 = 2000 :=
sorry

end both_savings_2000_l422_422219


namespace quadratic_eq_l422_422065

noncomputable def roots (r s : ℝ): Prop := r + s = 12 ∧ r * s = 27 ∧ (r = 2 * s ∨ s = 2 * r)

theorem quadratic_eq (r s : ℝ) (h : roots r s) : 
   Polynomial.C 1 * (X^2 - Polynomial.C (r + s) * X + Polynomial.C (r * s)) = X ^ 2 - 12 * X + 27 := 
sorry

end quadratic_eq_l422_422065


namespace complex_division_result_l422_422203

theorem complex_division_result : (1 + 2 * Complex.i) / (2 - Complex.i) = Complex.i := 
by
  sorry

end complex_division_result_l422_422203


namespace fourth_division_students_l422_422481

-- Define the total number of students
def total_students : ℕ := 500

-- Define the percentage of students in each division for each subject
def percentage_math_first : ℚ := 0.15
def percentage_math_second : ℚ := 0.40
def percentage_math_third : ℚ := 0.35
def percentage_math_fourth : ℚ := 0.10

def percentage_english_first : ℚ := 0.20
def percentage_english_second : ℚ := 0.45
def percentage_english_third : ℚ := 0.25
def percentage_english_fourth : ℚ := 0.10

def percentage_science_first : ℚ := 0.12
def percentage_science_second : ℚ := 0.48
def percentage_science_third : ℚ := 0.35
def percentage_science_fourth : ℚ := 0.05

def percentage_social_first : ℚ := 0.25
def percentage_social_second : ℚ := 0.30
def percentage_social_third : ℚ := 0.35
def percentage_social_fourth : ℚ := 0.10

def percentage_language_first : ℚ := 0.18
def percentage_language_second : ℚ := 0.42
def percentage_language_third : ℚ := 0.30
def percentage_language_fourth : ℚ := 0.10

-- Calculate the number of students in each division for each subject
def students_math_fourth : ℕ := (percentage_math_fourth * total_students).toNat
def students_english_fourth : ℕ := (percentage_english_fourth * total_students).toNat
def students_science_fourth : ℕ := (percentage_science_fourth * total_students).toNat
def students_social_fourth : ℕ := (percentage_social_fourth * total_students).toNat
def students_language_fourth : ℕ := (percentage_language_fourth * total_students).toNat

-- Calculate the total number of students in the Fourth Division across all subjects
def total_students_fourth : ℕ :=
  students_math_fourth +
  students_english_fourth +
  students_science_fourth +
  students_social_fourth +
  students_language_fourth

-- The theorem to be proved
theorem fourth_division_students : total_students_fourth = 225 := by
  sorry

end fourth_division_students_l422_422481


namespace complex_div_simplification_l422_422201

theorem complex_div_simplification : (1 + 2*Complex.i) / (2 - Complex.i) = Complex.i :=
by
  sorry

end complex_div_simplification_l422_422201


namespace volume_of_prism_l422_422616

variable (x y z : ℝ)
variable (h1 : x * y = 15)
variable (h2 : y * z = 10)
variable (h3 : x * z = 6)

theorem volume_of_prism : x * y * z = 30 :=
by
  sorry

end volume_of_prism_l422_422616


namespace problem_statement_l422_422779

-- Definitions of f and g based on given conditions
variables {f g : ℝ → ℝ} (hf : ∀ x : ℝ, f (-x) = -f x) (hg : ∀ x : ℝ, g (-x) = g x)
          (hdf : ∀ x : ℝ, x > 0 → deriv f x > 0) (hdg : ∀ x : ℝ, x > 0 → deriv g (-x) > 0)

theorem problem_statement :
  ∀ x : ℝ, x < 0 → deriv f x > 0 ∧ deriv g (-x) < 0 :=
by
  sorry

end problem_statement_l422_422779


namespace invertible_product_labels_l422_422566

theorem invertible_product_labels :
  let domain5 := {-6, -5, -4, -3, -2, 0, 1, 3},
      function3 (x : ℝ) := x^3 - 2,
      function4 (x : ℝ) := 5 / x,
      function5 : finset ℝ → finset ℝ := λ xs, xs.image (λ x, _) /- details omitted -/,
      function6 (x : ℝ) := -x - 2
  in (bijective function3) 
     ∧ (bijective function4)
     ∧ (∀ x ∈ domain5, ∀ y ∈ domain5, x ≠ y → function5({x}) ≠ function5({y}))
     ∧ (bijective function6) 
     → 3 * 4 * 5 * 6 = 360 :=
by { sorry }

end invertible_product_labels_l422_422566


namespace chinese_team_arrangements_l422_422486

noncomputable def numArrangements : ℕ := 6

theorem chinese_team_arrangements :
  let swimmers := {A, B, C, D}
  let styles := {backstroke, breaststroke, butterfly, freestyle}
  let constraints := (A ∈ {backstroke, freestyle} ∧ B ∈ {butterfly, freestyle}) 
                      ∧ (C ∈ styles ∧ D ∈ styles) 
                      ∧ (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  in
  constraints → numArrangements = 6 := 
by 
  sorry -- Proof to be done

end chinese_team_arrangements_l422_422486


namespace simple_interest_rate_l422_422258

def principal : ℕ := 600
def amount : ℕ := 950
def time : ℕ := 5
def expected_rate : ℚ := 11.67

theorem simple_interest_rate (P A T : ℕ) (R : ℚ) :
  P = principal → A = amount → T = time → R = expected_rate →
  (A = P + P * R * T / 100) :=
by
  intros hP hA hT hR
  sorry

end simple_interest_rate_l422_422258


namespace julie_reading_ratio_l422_422502

theorem julie_reading_ratio
  (total_pages : ℕ)
  (read_yesterday : ℕ)
  (read_twice_yesterday : ℕ)
  (read_tomorrow : ℕ)
  (remaining_pages : ℕ)
  (total_pages = 120)
  (read_yesterday = 12)
  (read_twice_yesterday = 2 * read_yesterday)
  (read_tomorrow = 42)
  (remaining_pages = total_pages - (read_yesterday + read_twice_yesterday)) :
  read_tomorrow / remaining_pages = 1 / 2 :=
by
  sorry

end julie_reading_ratio_l422_422502


namespace new_boarders_joined_l422_422267

def original_ratio_b_to_d : ℚ := 5 / 12
def new_ratio_b_to_d : ℚ := 1 / 2
def initial_boarders : ℕ := 150
def initial_day_students : ℕ := (initial_boarders * 12) / 5

theorem new_boarders_joined :
  ∃ (x : ℕ), (150 + x) / initial_day_students = 1 / 2 ∧
             (150 + x) = 360 / 2 ∧
             x = 30 := 
by {
  use 30,
  split,
  {
    -- Proof that the new ratio holds
    calc
      (150 + 30) / initial_day_students = 180 / 360 : by { rw initial_day_students, simp }
      ... = 1/2 : by norm_num,
  },
  {
    -- Proof that the total number of boarders is as described
    calc
      150 + 30 = 180 : by norm_num,
  },
  {
    -- Proof that x equals 30
    norm_num,
  },
}

end new_boarders_joined_l422_422267


namespace sum_of_special_indices_l422_422887

noncomputable def sequence (n : ℕ) : ℝ :=
if n = 1 then 0.202
else if n = 2 then (0.2021:ℚ)^(sequence 1)
else if n % 2 = 1 then (0.20201.rep [0201].concat.replicate ((n + 2) // 2)):ℚ)^(sequence (n - 1))
else (0.202011.rep [02011].concat.replicate ((n + 2) // 2)):ℚ)^(sequence (n - 1))

theorem sum_of_special_indices :
  (∑ k in (finset.range 1011).filter (λ k, k.mod 2 = 1 ∧ sequence k < sequence (k + 1)) \ 
       (∑ k in (finset.range 1011).filter (λ k, k.mod 2 = 0 ∧ sequence k > sequence (k + 1))),
  k) = 257130 :=
sorry

end sum_of_special_indices_l422_422887


namespace ellipse_equation_line_HN_fixed_point_l422_422414

def point := ℝ × ℝ

-- Given points A and B
def A : point := (0, -2)
def B : point := (3/2, -1)
def P : point := (1, -2)

-- Equation of ellipse
def ellipse (x y : ℝ) (m n : ℝ) := m*x^2 + n*y^2 = 1

-- Conditions for ellipse passing through A and B
def passes_through_A (m n : ℝ) := ellipse A.1 A.2 m n
def passes_through_B (m n : ℝ) := ellipse B.1 B.2 m n

-- Prove the equation of the ellipse
theorem ellipse_equation : ∃ m n, passes_through_A m n ∧ passes_through_B m n ∧ m > 0 ∧ n > 0 ∧ m ≠ n ∧ (m = 1/3 ∧ n = 1/4) :=
sorry

-- Now for the line HN passing through a fixed point
-- Definition of line HN passing through fixed point (0, -2)
def fixed_point : point := (0, -2)

def line (p1 p2 : point) (x y : ℝ) := 
  ∃ k : ℝ, (y - p1.2) = k * (x - p1.1) ∧ (y - p2.2) = k * (x - p2.1)

def line_HN (H N : point) := line H N

theorem line_HN_fixed_point :
  ∀ (m n : ℝ) (M N T H : point), 
    ellipse P.1 P.2 m n ∧ passes_through_A m n ∧ passes_through_B m n  ∧ 
    (H.1 + M.1) / 2 = T.1 ∧ (H.2 + M.2) / 2 = T.2 ∧
    (T.1 = (3 * M.2 + 6) / 2 ∧ T.2 = M.2) ∧
    line_HN H N fixed_point.1 fixed_point.2 :=
sorry

end ellipse_equation_line_HN_fixed_point_l422_422414


namespace hexagon_inequality_l422_422895

/-- Let ABCDEF be a convex hexagon, and let A₁, B₁, C₁, D₁, E₁, F₁ be midpoints of the sides AB, BC, CD, DE, EF, FA respectively.
Denote by p the perimeter of hexagon ABCDEF and by p₁ the perimeter of hexagon A₁B₁C₁D₁E₁F₁. Suppose that all inner angles
of hexagon A₁B₁C₁D₁E₁F₁ are equal. Prove that p ≥ (2 * sqrt(3) / 3) * p₁. -/
theorem hexagon_inequality (A B C D E F A₁ B₁ C₁ D₁ E₁ F₁ : Point)
                          (h_midpoints : (midpoint A B = A₁) ∧ (midpoint B C = B₁) ∧ (midpoint C D = C₁) ∧
                                         (midpoint D E = D₁) ∧ (midpoint E F = E₁) ∧ (midpoint F A = F₁))
                          (h_convex : is_convex_hexagon A B C D E F)
                          (h_equal_angles : ∀ i j k, angle_equal (angle (A₁ i) (B₁ j) (C₁ k)))
                          {p p₁ : ℝ} (h_perimeters : p = hexagon_perimeter A B C D E F ∧
                                                    p₁ = hexagon_perimeter A₁ B₁ C₁ D₁ E₁ F₁) :
  p ≥ (2 * Real.sqrt 3 / 3) * p₁ :=
begin
  sorry
end

end hexagon_inequality_l422_422895


namespace wyatt_total_envelopes_l422_422623

theorem wyatt_total_envelopes :
  let b := 10
  let y := b - 4
  let t := b + y
  t = 16 :=
by
  let b := 10
  let y := b - 4
  let t := b + y
  sorry

end wyatt_total_envelopes_l422_422623


namespace profit_function_max_profit_at_break_even_bounds_l422_422292

open Real

-- Definitions for parts of the problem
def fixed_cost : ℝ := 5000
def direct_cost (x : ℝ) : ℝ := 0.025 * x + 0.5
def revenue (x : ℝ) : ℝ := 5 * x - x^2
def profit (x : ℝ) : ℝ := revenue x - direct_cost x

-- Conditions from the problem
def condition_fixed_cost := fixed_cost = 5000
def condition_direct_cost (x : ℝ) := direct_cost x = 0.025 * x + 0.5
def condition_revenue (x : ℝ) := revenue x = 5 * x - x^2

-- Correct answers derived
theorem profit_function (x : ℝ) (h0 : 0 ≤ x) (h5 : x ≤ 5) :
  profit x = 5 * x - x^2 - (0.025 * x + 0.5) := sorry

theorem max_profit_at : ∃ x : ℝ, x = 2.375 ∧ profit 2.375 = 10.78125 := sorry

theorem break_even_bounds (x : ℝ) :
  5x - x^2 - (0.025 * x + 0.5) ≥ 0 ↔ (0.1 ≤ x ∧ x ≤ 5) ∨ (5 < x ∧ x < 48) := sorry

end profit_function_max_profit_at_break_even_bounds_l422_422292


namespace probability_of_all_girls_chosen_is_1_over_11_l422_422291

-- Defining parameters and conditions
def total_members : ℕ := 12
def boys : ℕ := 6
def girls : ℕ := 6
def chosen_members : ℕ := 3

-- Number of combinations to choose 3 members from 12
def total_combinations : ℕ := Nat.choose total_members chosen_members

-- Number of combinations to choose 3 girls from 6
def girl_combinations : ℕ := Nat.choose girls chosen_members

-- Probability is defined as the ratio of these combinations
def probability_all_girls_chosen : ℚ := girl_combinations / total_combinations

-- Proof Statement
theorem probability_of_all_girls_chosen_is_1_over_11 : probability_all_girls_chosen = 1 / 11 := by
  sorry -- Proof to be completed

end probability_of_all_girls_chosen_is_1_over_11_l422_422291


namespace c_finish_work_in_6_days_l422_422828

theorem c_finish_work_in_6_days (a b c : ℝ) (ha : a = 1/36) (hb : b = 1/18) (habc : a + b + c = 1/4) : c = 1/6 :=
by
  sorry

end c_finish_work_in_6_days_l422_422828


namespace ratio_of_smaller_circle_to_larger_circle_l422_422972

section circles

variables {Q : Type} (C1 C2 : ℝ) (angle1 : ℝ) (angle2 : ℝ)

def ratio_of_areas (C1 C2 : ℝ) : ℝ := (C1 / C2)^2

theorem ratio_of_smaller_circle_to_larger_circle
  (h1 : angle1 = 60)
  (h2 : angle2 = 48)
  (h3 : (angle1 / 360) * C1 = (angle2 / 360) * C2) :
  ratio_of_areas C1 C2 = 16 / 25 :=
by
  sorry

end circles

end ratio_of_smaller_circle_to_larger_circle_l422_422972


namespace minimize_norm_l422_422776

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Given conditions
variables (a b : ℝ × ℝ)
axiom cond1 : vector_magnitude a = 2
axiom cond2 : vector_magnitude b = 1
axiom cond3 : dot_product a b = 2 * 1 * real.cos (real.pi / 3) -- cos(60°) = 1/2

-- Goal: finding the value of x that minimizes the expression
theorem minimize_norm (x : ℝ) :
  ∃ x, ∀ y, vector_magnitude (a.1 - x * b.1, a.2 - x * b.2) ≤ vector_magnitude (a.1 - y * b.1, a.2 - y * b.2) := 
sorry

end minimize_norm_l422_422776


namespace ratio_a6_b6_l422_422761

-- Definitions for sequences and sums
variable {α : Type*} [LinearOrderedField α] 
variable (a b : ℕ → α) 
variable (S T : ℕ → α)

-- Main theorem stating the problem
theorem ratio_a6_b6 (h : ∀ n, S n / T n = (2 * n - 5) / (4 * n + 3)) :
    a 6 / b 6 = 17 / 47 :=
sorry

end ratio_a6_b6_l422_422761


namespace cos_sum_l422_422391

theorem cos_sum (α β : ℝ)
  (hα_range : 0 < α ∧ α < (π / 2))
  (hβ_range : (π / 2) < β ∧ β < π)
  (hcosα : Real.cos α = 3 / 5)
  (hsinβ : Real.sin β = √2 / 10) :
  Real.cos (α + β) = -√2 / 2 :=
by
  sorry

end cos_sum_l422_422391


namespace problem_solution_l422_422924

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem problem_solution :
  let interval_start := 10^(2014!) * Real.pi
  let interval_end := 10^(2014!+2018) * Real.pi
  let count_solutions := 3 * (10^(2014!-1)) * (10^2018 - 1)
  sum_of_digits count_solutions = 18162 :=
by
  let interval_start := 10^(2014!) * Real.pi
  let interval_end := 10^(2014!+2018) * Real.pi
  let count_solutions := 3 * (10^(2014!-1)) * (10^2018 - 1)
  sorry

end problem_solution_l422_422924


namespace vanessa_deleted_30_files_l422_422237

-- Define the initial conditions
def original_files : Nat := 16 + 48
def files_left : Nat := 34

-- Define the number of files deleted
def files_deleted : Nat := original_files - files_left

-- The theorem to prove the number of files deleted
theorem vanessa_deleted_30_files : files_deleted = 30 := by
  sorry

end vanessa_deleted_30_files_l422_422237


namespace find_sin_theta_l422_422880

variables {G : Type*} [inner_product_space ℝ G]

theorem find_sin_theta
  {a b c : G}
  (norm_a : ‖a‖ = 1) 
  (norm_b : ‖b‖ = 5) 
  (norm_c : ‖c‖ = 3) 
  (cross_eq : a × (a × b) = c) : 
  real.sin (real.angle a b) = 3 / 5 :=
by
  sorry

end find_sin_theta_l422_422880


namespace shortest_path_on_tetrahedron_eq_diameter_circumscribed_circle_l422_422402

/-- 
  Given an equilateral tetrahedron, the shortest path on its surface between two points 
  is equivalent to the diameter of the circumscribed circle of one of its equilateral faces.
-/
theorem shortest_path_on_tetrahedron_eq_diameter_circumscribed_circle
  (a : ℝ)
  (is_tetrahedron : ∀ {p : E}, p ∈ vertices_of_tetrahedron → ∃ t : triangle E, t ∈ faces_of_tetrahedron)
  (equilateral_faces : ∀ {t : triangle E}, t ∈ faces_of_tetrahedron → is_equilateral t)
  (K M : E) (K_on_surface : K ∈ surface_of_tetrahedron) (M_on_surface : M ∈ surface_of_tetrahedron)
  : shortest_path_on_surface K M = diameter_circumscribed_circle_of_face :=
sorry

end shortest_path_on_tetrahedron_eq_diameter_circumscribed_circle_l422_422402


namespace smallest_cubes_to_fill_box_l422_422629

theorem smallest_cubes_to_fill_box
  (L W D : ℕ)
  (hL : L = 30)
  (hW : W = 48)
  (hD : D = 12) :
  ∃ (n : ℕ), n = (L * W * D) / ((Nat.gcd (Nat.gcd L W) D) ^ 3) ∧ n = 80 := 
by
  sorry

end smallest_cubes_to_fill_box_l422_422629


namespace omega_range_l422_422143

-- Preliminary definitions and assumptions
def function_has_zeros (f : ℝ → ℝ) (a b : ℝ) (k : ℕ) : Prop := 
  ∃ xs : Fin k → ℝ, ∀ i : Fin k, a ≤ xs i ∧ xs i ≤ b ∧ f (xs i) = 0

theorem omega_range (ω : ℝ) (hω : 0 < ω) (φ : ℝ → ℝ) :
  (∀ (φ : ℝ), 
    function_has_zeros 
      (λ x => (sin (ω * x + φ) - 1/2)) 
      0 
      (2 * real.pi) 
      3
  ∧ (∀ (φ : ℝ), 
    function_has_zeros 
      (λ x => (sin (ω * x + φ) - 1/2)) 
      0 
      (2 * real.pi) 
      4)
  → (5/3 ≤ ω ∧ ω < 2)) := 
sorry

end omega_range_l422_422143


namespace initial_salty_cookies_count_l422_422172

-- Define initial conditions
def initial_sweet_cookies : ℕ := 9
def sweet_cookies_ate : ℕ := 36
def salty_cookies_left : ℕ := 3
def salty_cookies_ate : ℕ := 3

-- Theorem to prove the initial salty cookies count
theorem initial_salty_cookies_count (initial_salty_cookies : ℕ) 
    (initial_sweet_cookies : initial_sweet_cookies = 9) 
    (sweet_cookies_ate : sweet_cookies_ate = 36)
    (salty_cookies_ate : salty_cookies_ate = 3) 
    (salty_cookies_left : salty_cookies_left = 3) : 
    initial_salty_cookies = 6 := 
sorry

end initial_salty_cookies_count_l422_422172


namespace range_of_a_for_maximum_l422_422794

noncomputable def f (x : ℝ) (a : ℝ) := 3 * Real.log x - x^2 + (a - 1/2) * x

theorem range_of_a_for_maximum (a : ℝ) : 
  (∀ x ∈ set.Ioo (1 : ℝ) 3, has_deriv_at (λ x, 3 * Real.log x - x^2 + (a - 1/2) * x) (3 / x - 2 * x + (a - 1/2)) x ∧ 
    (∀ x ∈ set.Ioo (1 : ℝ) 3, 3 / x - 2 * x + (a - 1/2) = 0 → ∃ y ∈ set.Ioo (1 : ℝ) 3, (3 / y - 2 * y + (a - 1/2)) * (2 / y^2 - 2) < 0)) 
  → -1/2 < a ∧ a < 11/2 :=
begin
  sorry    
end

end range_of_a_for_maximum_l422_422794


namespace triangle_angle_B_condition_l422_422966

variable (A B C M D : Type)
variable [LinearOrder B]
variables {angle_B angle_C : ℝ}
variable {M is_midpoint : (BC : Type)}
variable (perpendicular_bisector_intersects_circumcircle : Prop)
variable (triangle_ABC : Triangle ℝ)
variable (circumcircle_ADCB_order : Prop)
variable (angle_ADM angle_DAC : ℝ)

-- Definitions and assumptions
def angle_B_greater_than_angle_C := angle_B > angle_C
def midpoint_of_BC := is_midpoint M BC
def perpendicular_bisector_condition := perpendicular_bisector_intersects_circumcircle
def circumcircle_condition := circumcircle_ADCB_order
def angle_ADM_given := angle_ADM = 68
def angle_DAC_given := angle_DAC = 64

-- Goal
theorem triangle_angle_B_condition :
  angle_B_greater_than_angle_C angle_B angle_C ∧
  midpoint_of_BC BC M ∧
  perpendicular_bisector_condition ∧
  circumcircle_condition ∧
  angle_ADM_given angle_ADM ∧
  angle_DAC_given angle_DAC →
  angle_B = 86 :=
by sorry

end triangle_angle_B_condition_l422_422966


namespace prove_m_plus_n_l422_422874

noncomputable def m_plus_n (a b : ℝ) (m n : ℕ) [rel_prime : Nat.Coprime m n] : ℕ :=
if h : (8^a + 2^(b+7)) * (2^(a+3) + 8^(b-2)) = 4^(a+b+2) ∧ ab = (m / n) then m + n else 0

theorem prove_m_plus_n (a b : ℝ) (m n : ℕ) [rel_prime : Nat.Coprime m n] : 
  (8^a + 2^(b+7)) * (2^(a+3) + 8^(b-2)) = 4^(a+b+2) → 
  m_plus_n a b m n = 121 :=
by
  sorry

end prove_m_plus_n_l422_422874


namespace cuboids_non_overlap_face_l422_422657

-- Define the setup and helper definitions

-- A cuboid is defined by its 3D extents within the cube
structure Cuboid :=
(x_min x_max y_min y_max z_min z_max : ℝ)
(valid_x : x_min < x_max)
(valid_y : y_min < y_max)
(valid_z : z_min < z_max)

-- A cube is defined as a set of cuboids satisfying the provided conditions
structure CubePartition :=
(cuboids : set Cuboid)
(partition_property : ∀ (c₁ c₂ : Cuboid), c₁ ∈ cuboids → c₂ ∈ cuboids → c₁ ≠ c₂ →
                   ((c₁.x_min < c₂.x_max ∧ c₁.x_max > c₂.x_min) ∨
                    (c₁.y_min < c₂.y_max ∧ c₁.y_max > c₂.y_min) ∨
                    (c₁.z_min < c₂.z_max ∧ c₁.z_max > c₂.z_min)))

-- The main theorem to prove
theorem cuboids_non_overlap_face (cube_partition : CubePartition) :
  ∀ (c₁ c₂ c₃ : Cuboid), c₁ ∈ cube_partition.cuboids → c₂ ∈ cube_partition.cuboids → c₃ ∈ cube_partition.cuboids →
  ∃ f : (Cuboid → ℝ × ℝ), (∀ {c_ij}, c_ij ∈ [c₁, c₂, c₃] → f c_ij = f c_ij) ∧
  ∀ {i j}, i ≠ j → f c₁.i ∩ f c₂.j = ∅ := 
by sorry

end cuboids_non_overlap_face_l422_422657


namespace geometric_series_sum_eq_l422_422885

theorem geometric_series_sum_eq :
  let a := (5 : ℚ)
  let r := (-1/2 : ℚ)
  (∑' n : ℕ, a * r^n) = (10 / 3 : ℚ) :=
by
  sorry

end geometric_series_sum_eq_l422_422885


namespace area_ratio_of_concentric_circles_l422_422980

noncomputable theory

-- Define the given conditions
def C1 (r1 : ℝ) : ℝ := 2 * Real.pi * r1
def C2 (r2 : ℝ) : ℝ := 2 * Real.pi * r2
def arc_length (angle : ℝ) (circumference : ℝ) : ℝ := (angle / 360) * circumference

-- Lean statement for the math proof problem
theorem area_ratio_of_concentric_circles 
  (r1 r2 : ℝ) (h₁ : arc_length 60 (C1 r1) = arc_length 48 (C2 r2)) : 
  (Real.pi * r1^2) / (Real.pi * r2^2) = 16 / 25 :=
by
  sorry  -- Proof omitted

end area_ratio_of_concentric_circles_l422_422980


namespace equivalent_root_expression_l422_422331

variable (x : ℝ)
variable (hx : 0 < x)

theorem equivalent_root_expression : (x * real.sqrt (real.cbrt x))^(1/4) = x^(1/3) :=
sorry

end equivalent_root_expression_l422_422331


namespace solved_zk_problem_l422_422849

noncomputable def parallelogram_zk_calc (EF FG k HE GH : ℝ) : Prop :=
  ∃ z k : ℝ,
    EF = 5 * z + 5 ∧ 
    FG = 4 * k ^ 2 ∧ 
    GH = 40 ∧ 
    HE = k + 20 ∧ 
    EF = GH ∧ 
    FG = HE ∧ 
    z = 7 ∧ 
    k = (1 + Real.sqrt 321) / 8 ∧ 
    z * k = (7 + 7 * Real.sqrt 321) / 8

theorem solved_zk_problem :
  parallelogram_zk_calc (5 * 7 + 5) (4 * ((1 + Real.sqrt 321) / 8) ^ 2) 
    ((1 + Real.sqrt 321) / 8) 
    ((1 + Real.sqrt 321) / 8 + 20) 
    40 :=
begin
  sorry
end

end solved_zk_problem_l422_422849


namespace max_dot_product_between_ellipses_l422_422450

noncomputable def ellipse1 (x y : ℝ) : Prop := (x^2 / 25 + y^2 / 9 = 1)
noncomputable def ellipse2 (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 9 = 1)

theorem max_dot_product_between_ellipses :
  ∀ (M N : ℝ × ℝ),
    ellipse1 M.1 M.2 →
    ellipse2 N.1 N.2 →
    ∃ θ φ : ℝ,
      M = (5 * Real.cos θ, 3 * Real.sin θ) ∧
      N = (3 * Real.cos φ, 3 * Real.sin φ) ∧
      (15 * Real.cos θ * Real.cos φ + 9 * Real.sin θ * Real.sin φ ≤ 15) :=
by
  sorry

end max_dot_product_between_ellipses_l422_422450


namespace total_decorations_l422_422716

theorem total_decorations 
  (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) (pumpkins : ℕ) 
  (cauldron : ℕ) (budget_decorations : ℕ) (left_decorations : ℕ)
  (h_skulls : skulls = 12)
  (h_broomsticks : broomsticks = 4)
  (h_spiderwebs : spiderwebs = 12)
  (h_pumpkins : pumpkins = 2 * spiderwebs)
  (h_cauldron : cauldron = 1)
  (h_budget_decorations : budget_decorations = 20)
  (h_left_decorations : left_decorations = 10) : 
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_decorations + left_decorations = 83 := 
by 
  sorry

end total_decorations_l422_422716


namespace limit_sequence_l422_422340

open Real

noncomputable def a_n (n : ℕ) : ℝ := (n * (n + 1) / 2) / (n - n^2 + 3)

theorem limit_sequence : 
  tendsto (λ (n : ℕ), a_n n) at_top (𝓝 (-1 / 2)) :=
by
  sorry

end limit_sequence_l422_422340


namespace sum_of_roots_l422_422367

theorem sum_of_roots :
  let f := (3 : ℝ) * (x : ℝ)^3 - 9 * x^2 + 5 * x - 15
  let g := (4 : ℝ) * x^3 + 2 * x^2 - 6 * x + 24
  (3 * (-9) / 3 + 2 * (-1) / 4) = 2.5 :=
by
  let S₁ := -(coeff (3 : ℝ) * (-9)) / 3
  let S₂ := -(coeff (4 : ℝ) * (2)) / 4
  have : S₁ = 3 := rfl
  have : S₂ = -0.5 := rfl
  have : S₁ + S₂ = 2.5 := by norm_num
  exact this


end sum_of_roots_l422_422367


namespace solve_for_y_l422_422990

theorem solve_for_y :
  ∀ y : ℤ, (2 ^ 5 * y) / (8 ^ 2 * 3 ^ 5) = 0.16666666666666666 → y = 81 :=
by
  intro y h
  sorry

end solve_for_y_l422_422990


namespace cost_of_each_ruler_l422_422106
-- Import the necessary library

-- Define the conditions and statement
theorem cost_of_each_ruler (students : ℕ) (rulers_each : ℕ) (cost_per_ruler : ℕ) (total_cost : ℕ) 
  (cond1 : students = 42)
  (cond2 : students / 2 < 42 / 2)
  (cond3 : cost_per_ruler > rulers_each)
  (cond4 : students * rulers_each * cost_per_ruler = 2310) : 
  cost_per_ruler = 11 :=
sorry

end cost_of_each_ruler_l422_422106


namespace length_of_CB_l422_422468

-- Definitions for the conditions
variables {A B C D E : Type*} [linear_ordered_field F]
variables (CD DA CE CB : F)
variables (h1 : CD = 6)
variables (h2 : DA = 8)
variables (h3 : CE = 9)

-- Definition for the theorem
theorem length_of_CB : CB = 21 :=
by
  -- apply the ratio similar triangles theorem and given conditions
  have h_sim :  ∃ (factor : F), CB = factor * CE := sorry, 
  have h_ratio : factor = (CD + DA) / CD := sorry,
  have h_calc : factor * CE = 21 := sorry,
  exact h_calc

end length_of_CB_l422_422468


namespace percentage_increase_second_year_l422_422576

theorem percentage_increase_second_year 
  (initial_population : ℝ)
  (first_year_increase : ℝ) 
  (population_after_2_years : ℝ) 
  (final_population : ℝ)
  (H_initial_population : initial_population = 800)
  (H_first_year_increase : first_year_increase = 0.22)
  (H_population_after_2_years : final_population = 1220) :
  ∃ P : ℝ, P = 25 := 
by
  -- Define the population after the first year
  let population_after_first_year := initial_population * (1 + first_year_increase)
  -- Define the equation relating populations and solve for P
  let second_year_increase := (final_population / population_after_first_year - 1) * 100
  -- Show P equals 25
  use second_year_increase
  sorry

end percentage_increase_second_year_l422_422576


namespace find_c_l422_422277

theorem find_c 
  (a b c : ℝ) 
  (h_vertex : ∀ x y, y = a * x^2 + b * x + c → 
    (∃ k l, l = b / (2 * a) ∧ k = a * l^2 + b * l + c ∧ k = 3 ∧ l = -2))
  (h_pass : ∀ x y, y = a * x^2 + b * x + c → 
    (x = 2 ∧ y = 7)) : c = 4 :=
by sorry

end find_c_l422_422277


namespace area_of_square_is_74_l422_422723

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def adjacent_vertices_1 : ℝ × ℝ := (-2, -1)
def adjacent_vertices_2 : ℝ × ℝ := (3, 6)

def side_length : ℝ :=
  distance adjacent_vertices_1 adjacent_vertices_2

def area_of_square (s : ℝ) : ℝ :=
  s^2

theorem area_of_square_is_74 : area_of_square side_length = 74 := by
  sorry

end area_of_square_is_74_l422_422723


namespace fraction_of_house_painted_l422_422453

theorem fraction_of_house_painted (total_time : ℝ) (paint_time : ℝ) (house : ℝ) (h1 : total_time = 60) (h2 : paint_time = 15) (h3 : house = 1) : 
  (paint_time / total_time) * house = 1 / 4 :=
by
  sorry

end fraction_of_house_painted_l422_422453


namespace sum_of_distinct_residues_l422_422744

theorem sum_of_distinct_residues (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  ∑ k in {0, 1, 2, 3, 4, 5, 6, 8, 10}, (k : ℕ) = 39 :=
by
  sorry

end sum_of_distinct_residues_l422_422744


namespace circle_problem_l422_422067

theorem circle_problem :
  (∃ (m : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + m = 0 → (2^2 + (-2)^2 + 2 * 2 - 4 * (-2) + m = 0 ∧
  ∃ (C : ℝ × ℝ) (r : ℝ), C = (-1, 2) ∧ r = 5 ∧
  (∀ l : ℝ → ℝ, l = -4 / 3 →
  (∀ (P Q : ℝ × ℝ), (∃ (b : ℝ), |PQ| = 4 * sqrt 6 →
  (l = C) ∧ (x - C.1)^2 + (y - C.2)^2 = r^2 → 
  ((4 * x + 3 * y - 7 = 0) ∨ (4 * x + 3 * y + 3 = 0)))
    )
  )
)) :=
sorry

end circle_problem_l422_422067


namespace ratio_of_place_values_l422_422123

def thousands_place_value : ℝ := 1000
def tenths_place_value : ℝ := 0.1

theorem ratio_of_place_values : thousands_place_value / tenths_place_value = 10000 := by
  sorry

end ratio_of_place_values_l422_422123


namespace height_of_parabolic_arch_l422_422684

theorem height_of_parabolic_arch (a : ℝ) (x : ℝ) (k : ℝ) (h : ℝ) (s : ℝ) :
  k = 20 →
  s = 30 →
  a = - 4 / 45 →
  x = 3 →
  k = h →
  y = a * x^2 + k →
  h = 20 → 
  y = 19.2 :=
by
  -- Given the conditions, we'll prove using provided Lean constructs
  sorry

end height_of_parabolic_arch_l422_422684


namespace percentage_mike_has_l422_422165
-- Definitions and conditions
variables (phone_cost : ℝ) (additional_needed : ℝ)
def amount_mike_has := phone_cost - additional_needed

-- Main statement
theorem percentage_mike_has (phone_cost : ℝ) (additional_needed : ℝ) (h1 : phone_cost = 1300) (h2 : additional_needed = 780) : 
  (amount_mike_has phone_cost additional_needed) * 100 / phone_cost = 40 :=
by
  sorry

end percentage_mike_has_l422_422165


namespace shonda_kids_calculation_l422_422920

def number_of_kids (B E P F A : Nat) : Nat :=
  let T := B * E
  let total_people := T / P
  total_people - (F + A + 1)

theorem shonda_kids_calculation :
  (number_of_kids 15 12 9 10 7) = 2 :=
by
  unfold number_of_kids
  exact rfl

end shonda_kids_calculation_l422_422920


namespace algebraic_expression_value_l422_422465

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 2 * x^2 + 3 * x - 7 = -6 :=
by sorry

end algebraic_expression_value_l422_422465


namespace quadratic_has_real_roots_l422_422801

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) := 
by 
  sorry

end quadratic_has_real_roots_l422_422801


namespace consecutive_odd_numbers_first_l422_422633

theorem consecutive_odd_numbers_first :
  ∃ x : ℤ, 11 * x = 3 * (x + 4) + 4 * (x + 2) + 16 ∧ x = 9 :=
by 
  sorry

end consecutive_odd_numbers_first_l422_422633


namespace correct_statements_count_l422_422947

-- Define the conditions based on the problem
def statement1_valid : Prop := False  -- INPUT a+2; is not a valid input statement
def statement2_valid : Prop := True   -- x = x - 5; is a valid assignment statement
def statement3_valid : Prop := False  -- PRINT M=2; is not a valid output statement

-- Define the number of correct statements
def num_correct_statements : ℕ := 
  (if statement1_valid then 1 else 0) + 
  (if statement2_valid then 1 else 0) + 
  (if statement3_valid then 1 else 0)

-- The theorem proving that the number of correct statements is exactly 1
theorem correct_statements_count : num_correct_statements = 1 :=
by 
  -- We know by the conditions that only the second statement is valid
  -- Thus, the total count should be 1
  sorry

end correct_statements_count_l422_422947


namespace unique_5_digit_number_l422_422252

theorem unique_5_digit_number (N : ℕ) 
  (digits : Finset ℕ) 
  (h1 : digits.card = 5)
  (h2 : ∀ d ∈ digits, d ≠ 0 ∧ d < 10)
  (h3 : N = (digits.sum * 1332)) :
  N = 35964 := 
sorry

end unique_5_digit_number_l422_422252


namespace exists_x_eq_l422_422149

noncomputable def R (P Q : ℝ[X]) : ℝ[X] :=
  P.comp (X - C 1) - Q.comp (X + C 1)

theorem exists_x_eq : ∀ (P Q : ℝ[X]),
  Polynomial.degree P = 2014 →
  Polynomial.degree Q = 2014 →
  P.leadingCoeff = 1 →
  Q.leadingCoeff = 1 →
  (∀ x : ℝ, P.eval x ≠ Q.eval x) →
  ∃ x : ℝ, P.eval (x - 1) = Q.eval (x + 1) :=
sorry

end exists_x_eq_l422_422149


namespace integral_value_l422_422787

theorem integral_value (a : ℝ) (h : -35 * a^3 = -280) : ∫ x in a..2 * Real.exp 1, 1 / x = 1 := by
  sorry

end integral_value_l422_422787


namespace five_books_permutation_l422_422175

theorem five_books_permutation : ∃ (n : ℕ), n = 5! ∧ n = 120 := 
by {
  use (5!),
  split,
  { refl },
  { norm_num }
}
sorry

end five_books_permutation_l422_422175


namespace balcony_more_than_orchestra_l422_422631

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 340) 
  (h2 : 12 * x + 8 * y = 3320) : 
  y - x = 40 := 
sorry

end balcony_more_than_orchestra_l422_422631


namespace percent_game_of_thrones_altered_l422_422027

def votes_game_of_thrones : ℕ := 10
def votes_twilight : ℕ := 12
def votes_art_of_deal : ℕ := 20

def altered_votes_art_of_deal : ℕ := votes_art_of_deal - (votes_art_of_deal * 80 / 100)
def altered_votes_twilight : ℕ := votes_twilight / 2
def total_altered_votes : ℕ := altered_votes_art_of_deal + altered_votes_twilight + votes_game_of_thrones

theorem percent_game_of_thrones_altered :
  ((votes_game_of_thrones * 100) / total_altered_votes) = 50 := by
  sorry

end percent_game_of_thrones_altered_l422_422027


namespace bus_distance_l422_422583

theorem bus_distance (w r : ℝ) (h1 : w = 0.17) (h2 : r = w + 3.67) : r = 3.84 :=
by
  sorry

end bus_distance_l422_422583


namespace perpendiculars_intersect_symmetric_point_l422_422285

theorem perpendiculars_intersect_symmetric_point
  (ABC : Triangle)
  (O : Point)
  (circle : Circle)
  (A1 A2 B1 B2 C1 C2 : Point)
  (H_circle_intersections : circle.intersects_sides ABC A1 A2 B1 B2 C1 C2)
  (M : Point)
  (H_perpendiculars_intersect_at_M : ∀ {P Q R : Point},
    (P ∈ {A1, B1, C1}) →
    (Q ∈ {B1, C1, A1}) →
    (R ∈ {C1, A1, B1}) →
    meet_perpendiculars.at_single_point P Q R M) :
  ∃ M', meet_perpendiculars.at_single_point A2 B2 C2 M' ∧ M' = symmetric_point M O :=
sorry

end perpendiculars_intersect_symmetric_point_l422_422285


namespace not_even_or_odd_l422_422722

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem not_even_or_odd : ¬(∀ x : ℝ, f (-x) = f x) ∧ ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry

end not_even_or_odd_l422_422722


namespace range_of_f_in_interval_l422_422796

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x - 9

theorem range_of_f_in_interval :
  set.range (λ (x : {x : ℝ // 1 < x ∧ x < 4}), f x) = set.Ico (-18 : ℝ) (-14 : ℝ) :=
by sorry

end range_of_f_in_interval_l422_422796


namespace angle_A_greater_than_45_degrees_angle_A_range_l422_422479

variable {A B C : Type} [isTriangle A B C]

-- Given conditions
def is_acute (A B C : Type) [isTriangle A B C] : Prop := 
  ∀ ⦃α β γ : ℝ⦄, α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90

def intersect_in_one_point (A B C : Type) [isTriangle A B C] : Prop :=
  -- This would express the specific intersection of angle bisector AD, median BM, and altitude CH at one point.
  sorry

-- Problem statement
theorem angle_A_greater_than_45_degrees 
  [hAcute : is_acute A B C] 
  [hIntersect : intersect_in_one_point A B C] : 
  let α := angle A in
  α > 45 := 
sorry

-- Extended problem statement to include the range of angle A
theorem angle_A_range 
  [hAcute : is_acute A B C] 
  [hIntersect : intersect_in_one_point A B C] : 
  let α := angle A in
  51.8333 < α ∧ α < 90 :=
sorry

end angle_A_greater_than_45_degrees_angle_A_range_l422_422479


namespace smallest_angle_of_convex_20_sided_polygon_l422_422346

theorem smallest_angle_of_convex_20_sided_polygon :
  ∀ (a d : ℕ), 
    let n := 20 in
    let average_angle := (n - 2) * 180 / n in
    average_angle = 162 ∧ 
    (∀ k : ℕ, k ≤ n → a + (k - 1) * d < 180) ∧ 
    d < 18 / 19 ∧ 
    a + 19 * d < 180
  → a = 143 :=
by
  sorry

end smallest_angle_of_convex_20_sided_polygon_l422_422346


namespace geom_seq_common_ratio_l422_422674

theorem geom_seq_common_ratio:
  ∃ r : ℝ, 
  r = -2 ∧ 
  (∀ n : ℕ, n = 0 → n = 3 →
  let a : ℕ → ℝ := λ n, if n = 0 then 25 else
                            if n = 1 then -50 else
                            if n = 2 then 100 else
                            if n = 3 then -200 else 0 in
  (a n = a 0 * r ^ n)) :=
by
  sorry

end geom_seq_common_ratio_l422_422674


namespace max_subset_size_l422_422511

theorem max_subset_size (T : Set ℕ) (hT₁ : ∀ x ∈ T, x ≤ 2499) (hT₂ : ∀ x y ∈ T, x ≠ y → |x - y| ≠ 5 ∧ |x - y| ≠ 6) :
  T.card ≤ 1250 := 
sorry

end max_subset_size_l422_422511


namespace solve_inequality_l422_422013

theorem solve_inequality (x : ℝ) (h1: 3 * x - 8 ≠ 0) :
  5 ≤ x / (3 * x - 8) ∧ x / (3 * x - 8) < 10 ↔ (8 / 3) < x ∧ x ≤ (20 / 7) := 
sorry

end solve_inequality_l422_422013


namespace find_AP_l422_422119

-- Definitions of the geometrical constructs and their properties
variables (A B C D : Point)
variables (W X Y Z : Point)
variables (P : Point)

-- Given conditions
def is_square (A B C D : Point) (side_length : ℝ) : Prop := sorry
def is_rectangle (W X Y Z : Point) (ZY_length XY_length : ℝ) : Prop := sorry
def are_perpendicular (line1 line2 : Line) : Prop := sorry
def area_of_rectangle (W X Y Z : Point) : ℝ := 12 * 8
def shaded_area (shaded_rectangle : Rectangle) : ℝ := (area_of_rectangle W X Y Z) / 3

-- The main theorem we need to prove
theorem find_AP (A B C D W X Y Z P : Point) :
    is_square A B C D 8 →
    is_rectangle W X Y Z 12 8 →
    are_perpendicular (line_through A D) (line_through W X) →
    shaded_area (mk_rectangle P D C X) = (area_of_rectangle W X Y Z) / 3 →
    distance A P = 4 :=
sorry

end find_AP_l422_422119


namespace probability_at_least_half_girls_l422_422164

noncomputable def binomial (n k : ℕ) : ℚ := (nat.choose n k : ℚ)

theorem probability_at_least_half_girls :
  let p_girl := 0.52
  let p_boy := 0.48
  let n := 7
  let p_4 := binomial n 4 * (p_girl)^4 * (p_boy)^3
  let p_5 := binomial n 5 * (p_girl)^5 * (p_boy)^2
  let p_6 := binomial n 6 * (p_girl)^6 * (p_boy)^1
  let p_7 := binomial n 7 * (p_girl)^7 * (p_boy)^0
  p_4 + p_5 + p_6 + p_7 ≈ 0.98872 := sorry

end probability_at_least_half_girls_l422_422164


namespace students_in_each_normal_class_l422_422440

theorem students_in_each_normal_class
  (initial_students : ℕ)
  (percent_to_move : ℚ)
  (grade_levels : ℕ)
  (students_in_advanced_class : ℕ)
  (num_of_normal_classes : ℕ)
  (h_initial_students : initial_students = 1590)
  (h_percent_to_move : percent_to_move = 0.4)
  (h_grade_levels : grade_levels = 3)
  (h_students_in_advanced_class : students_in_advanced_class = 20)
  (h_num_of_normal_classes : num_of_normal_classes = 6) :
  let students_moving := (initial_students : ℚ) * percent_to_move,
      students_per_grade := students_moving / grade_levels,
      students_remaining := students_per_grade - (students_in_advanced_class : ℚ),
      students_per_normal_class := students_remaining / (num_of_normal_classes : ℚ)
  in students_per_normal_class = 32 := 
by
  sorry

end students_in_each_normal_class_l422_422440


namespace solution_l422_422408

noncomputable def given_conditions (θ : ℝ) : Prop := 
  let a := (3, 1)
  let b := (Real.sin θ, Real.cos θ)
  (a.1 : ℝ) / b.1 = a.2 / b.2 

theorem solution (θ : ℝ) (h: given_conditions θ) :
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 5 / 2 :=
by
  sorry

end solution_l422_422408


namespace number_of_pairs_satisfying_conditions_l422_422089

theorem number_of_pairs_satisfying_conditions :
  ∃ (count : ℕ), count = 46 ∧ 
  count = (List.sum (List.map (λ m, List.length (List.filter (λ n, n < 30 - m^2 ∧ n % 2 = 0) (List.range 30)))) (List.range 1 6)) := 
by
  sorry

end number_of_pairs_satisfying_conditions_l422_422089


namespace fixed_point_exists_l422_422233

variables {α : Type*} [ComplexModule α]

def exists_fixed_point (a : α) (ρ : ℝ) (φ θ : ℝ) : Prop :=
  ∃ (P : α), ∀ (t : ℝ),
    abs (P - exp (θ + t) * I) = abs (P - (a + ρ * exp (φ + t) * I))

theorem fixed_point_exists (a : α) (ρ : ℝ) (φ θ : ℝ) :
  exists_fixed_point a ρ φ θ :=
  sorry

end fixed_point_exists_l422_422233


namespace prime_factors_of_M_unique_l422_422355

theorem prime_factors_of_M_unique (M : ℝ) (h : log 3 (log 5 (log 7 (log 11 M))) = 8) :
  ∃! p : ℕ, p.prime ∧ p ∣ nat.factorize_nat M := sorry

end prime_factors_of_M_unique_l422_422355


namespace find_k_l422_422495

variable (c : ℝ) (k : ℝ)
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a (n + 1) = c * a n

def sum_sequence (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k (c_ne_zero : c ≠ 0)
  (h_geo : geometric_sequence a c)
  (h_sum : sum_sequence S k)
  (h_a1 : a 1 = 3 + k)
  (h_a2 : a 2 = S 2 - S 1)
  (h_a3 : a 3 = S 3 - S 2) :
  k = -1 :=
sorry

end find_k_l422_422495


namespace susan_gate_probability_l422_422554

theorem susan_gate_probability : 
  let p := 41
  let q := 70 in
  p + q = 111 ∧ (123 : ℚ) / 210 = (41 : ℚ) / 70 :=
by
  sorry

end susan_gate_probability_l422_422554


namespace unique_function_l422_422046

variables {f : ℝ → ℝ} {t : ℝ}

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(x) + f(-x) = 0

def decreasing_function (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ) (t : ℝ), t > 0 → f(x + t) < f(x)

theorem unique_function :
  (odd_function f) ∧ (decreasing_function f) → (∀ x : ℝ, f(x) = -3x) :=
begin
  intros h x,
  sorry
end

end unique_function_l422_422046


namespace complex_number_solution_l422_422643

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i * z = 1) : z = -i :=
by sorry

end complex_number_solution_l422_422643


namespace magnitude_of_z_l422_422033

noncomputable def i := Complex.I

def z := (2 - i) / (1 + i) - Complex.I^(2016 : ℕ)

theorem magnitude_of_z :
  Complex.abs z = (Real.sqrt 10) / 2 :=
by
  sorry

end magnitude_of_z_l422_422033


namespace roots_polynomial_expression_l422_422884

theorem roots_polynomial_expression (p q r : ℝ) (hRoots : Polynomial.eval (x : ℝ) (x^3 - 8*x^2 + 9*x - 3) = 0) 
  (hVieta1 : p + q + r = 8) (hVieta2 : p*q + p*r + q*r = 9) (hVieta3 : p*q*r = 3) :
  (p / (q*r + 1) + q / (p*r + 1) + r / (p*q + 1) = 83 / 43) := 
sorry

end roots_polynomial_expression_l422_422884


namespace geometric_sequence_sum_l422_422709

theorem geometric_sequence_sum (a : ℕ → ℕ) (r : ℕ) (n : ℕ)
  (h1 : a 2 + a 4 = 20) (h2 : a 3 + a 5 = 40)
  (h3 : ∀ i, a i = a * r ^ (i - 1)) :
  let S_n := (finset.range n).sum (λ i, a * r ^ i) in
  S_n = 2^(n+1) - 2 :=
by
  sorry

end geometric_sequence_sum_l422_422709


namespace can_juice_ounces_l422_422279

def total_cost : ℝ := 84
def unit_cost_per_ounce : ℝ := 7.0
def number_of_ounces : ℝ := total_cost / unit_cost_per_ounce

theorem can_juice_ounces : number_of_ounces = 12 :=
by {
  unfold number_of_ounces,
  rw [total_cost, unit_cost_per_ounce],
  norm_num,
  sorry
}

end can_juice_ounces_l422_422279


namespace solution_set_f_range_of_a_l422_422798

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x - 1|

-- Define the function g(x)
def g (x a : ℝ) : ℝ := x^2 - 2*x + |a^2 - 3|

-- Prove the solution set for the inequality f(x) ≤ 7.
theorem solution_set_f (x : ℝ) : f x ≤ 7 ↔ -2 ≤ x ∧ x ≤ 5 :=
by
  sorry

-- Prove the range of values for a such that minimum value of g(x) is not less than the minimum value of f(x).
theorem range_of_a (a x : ℝ) : (∀ x, g x a ≥ 1) ↔ (a ≤ -sqrt 5 ∨ (a ≥ -1 ∧ a ≤ 1) ∨ a ≥ sqrt 5) :=
by
  sorry

end solution_set_f_range_of_a_l422_422798


namespace geom_seq_common_ratio_l422_422672

theorem geom_seq_common_ratio:
  ∃ r : ℝ, 
  r = -2 ∧ 
  (∀ n : ℕ, n = 0 → n = 3 →
  let a : ℕ → ℝ := λ n, if n = 0 then 25 else
                            if n = 1 then -50 else
                            if n = 2 then 100 else
                            if n = 3 then -200 else 0 in
  (a n = a 0 * r ^ n)) :=
by
  sorry

end geom_seq_common_ratio_l422_422672


namespace largest_n_exists_largest_n_l422_422395

def s (n : ℕ) : ℕ := n.digits.sum

theorem largest_n (n : ℕ) (h : n = s(n)^2 + 2 * s(n) - 2) : n ≤ 397 :=
sorry

theorem exists_largest_n : ∃ n, n = s(n)^2 + 2 * s(n) - 2 ∧ n = 397 :=
sorry

end largest_n_exists_largest_n_l422_422395


namespace mutually_exclusive_and_complementary_events_l422_422373

def genuine_products : ℕ := 6
def defective_products : ℕ := 3
def total_products : ℕ := genuine_products + defective_products
def selected_products : ℕ := 3

def at_least_two_defective (selected : List ℕ) : Prop :=
  selected.filter (λ x => x = 1).length ≥ 2

def at_most_one_defective (selected : List ℕ) : Prop :=
  selected.filter (λ x => x = 1).length ≤ 1

theorem mutually_exclusive_and_complementary_events :
  ∀ (selection : List ℕ), selection.length = selected_products → 
  (at_least_two_defective selection ∨ at_most_one_defective selection) ∧
  (at_least_two_defective selection → ¬at_most_one_defective selection) ∧
  (at_most_one_defective selection → ¬at_least_two_defective selection) :=
  sorry

end mutually_exclusive_and_complementary_events_l422_422373


namespace quadratic_real_roots_range_l422_422802

theorem quadratic_real_roots_range (m : ℝ) : 
(m - 1) * x^2 - 2 * x + 1 = 0 → (m ≤ 2 ∧ m ≠ 1) :=
begin
  sorry
end

end quadratic_real_roots_range_l422_422802


namespace smallest_possible_average_l422_422379

theorem smallest_possible_average :
  ∃ (S : set ℝ), S = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  (∃ (A B C D E F : ℝ), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ E ∈ S ∧ F ∈ S ∧ 
  {A, B, C, D, E, F}.card = 6 ∧ 
  (∀ x ∈ {A, B, C, D, E, F}, ∃ n (d : ℝ), (0 ≤ n ∧ n ≤ 9) ∧ d ∈ {x}) ∧ 
  (A < 10 ∧ B < 10 ∧ C < 10) ∧ 
  (D ≥ 10 ∧ E ≥ 10 ∧ F ≥ 10)) ∧ 
  (∀ (set1 : finset ℝ), 
    set1 = {A, B, C, D, E, F} → 
    let sum1 := set1.sum in 
    sum1 = 99 / 6 → 
    sum1 = 16.5) :=
sorry

end smallest_possible_average_l422_422379


namespace polynomial_remainder_degrees_l422_422993

-- Define the divisor polynomial
def divisor := 3 * (x^3) - 4 * (x^2) + x - 5

-- Define a polynomial division remainder property
def valid_remainder_degrees :=
  {d : ℕ | d < degree divisor}

theorem polynomial_remainder_degrees :
  valid_remainder_degrees = {0, 1, 2} :=
by
  sorry

end polynomial_remainder_degrees_l422_422993


namespace common_ratio_of_sequence_l422_422667

theorem common_ratio_of_sequence 
  (a1 a2 a3 a4 : ℤ)
  (h1 : a1 = 25)
  (h2 : a2 = -50)
  (h3 : a3 = 100)
  (h4 : a4 = -200)
  (is_geometric : ∀ (i : ℕ), a1 * (-2) ^ i = if i = 0 then a1 else if i = 1 then a2 else if i = 2 then a3 else a4) : 
  (-50 / 25 = -2) ∧ (100 / -50 = -2) ∧ (-200 / 100 = -2) :=
by 
  sorry

end common_ratio_of_sequence_l422_422667


namespace range_of_a_l422_422461

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → -1 < a ∧ a < 3 :=
sorry

end range_of_a_l422_422461


namespace smallest_possible_average_l422_422383

def smallest_average (s : Finset (Fin 10)) : ℕ :=
  (Finset.sum s).toNat / 6

theorem smallest_possible_average : ∃ (s1 s2 : Finset ℕ), (s1.card = 3 ∧ s2.card = 3 ∧ 
 (s1 ∪ s2).card = 6 ∧ ∀ x, x ∈ s1 ∪ s2 → x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  smallest_average (s1 ∪ s2) = 16.5) sorry

end smallest_possible_average_l422_422383


namespace rectangle_from_intersections_l422_422352

theorem rectangle_from_intersections :
  (∀ (P1 P2 P3 P4 : ℝ × ℝ),
    (P1.1 * P1.2 = 20 ∧ P2.1 * P2.2 = 20 ∧ P3.1 * P3.2 = 20 ∧ P4.1 * P4.2 = 20) ∧
    (P1.1^2 + P1.2^2 = 41 ∧ P2.1^2 + P2.2^2 = 41 ∧ P3.1^2 + P3.2^2 = 41 ∧ P4.1^2 + P4.2^2 = 41) →
  (let d := λ (P Q : ℝ × ℝ), (P.1 - Q.1)^2 + (P.2 - Q.2)^2 in
  d P1 P2 = d P3 P4 ∧ d P2 P3 = d P4 P1 ∧ d P1 P3 = d P2 P4)) → 
  ∃ (P1 P2 P3 P4 : ℝ × ℝ),
    (P1.1 * P1.2 = 20 ∧ P2.1 * P2.2 = 20 ∧ P3.1 * P3.2 = 20 ∧ P4.1 * P4.2 = 20) ∧
    (P1.1^2 + P1.2^2 = 41 ∧ P2.1^2 + P2.2^2 = 41 ∧ P3.1^2 + P3.2^2 = 41 ∧ P4.1^2 + P4.2^2 = 41) →
  true := by
  admit

end rectangle_from_intersections_l422_422352


namespace difference_in_students_and_guinea_pigs_l422_422360

def num_students (classrooms : ℕ) (students_per_classroom : ℕ) : ℕ := classrooms * students_per_classroom
def num_guinea_pigs (classrooms : ℕ) (guinea_pigs_per_classroom : ℕ) : ℕ := classrooms * guinea_pigs_per_classroom
def difference_students_guinea_pigs (students : ℕ) (guinea_pigs : ℕ) : ℕ := students - guinea_pigs

theorem difference_in_students_and_guinea_pigs :
  ∀ (classrooms : ℕ) (students_per_classroom : ℕ) (guinea_pigs_per_classroom : ℕ),
  classrooms = 6 →
  students_per_classroom = 24 →
  guinea_pigs_per_classroom = 3 →
  difference_students_guinea_pigs (num_students classrooms students_per_classroom) (num_guinea_pigs classrooms guinea_pigs_per_classroom) = 126 :=
by
  intros
  sorry

end difference_in_students_and_guinea_pigs_l422_422360


namespace range_of_a_l422_422076

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) : 
  (∀ t : ℝ, t ≥ 1 → f (2 * t - 1) a ≥ 2 * f t a - 3) ↔ a < 2 := 
by 
  sorry

end range_of_a_l422_422076


namespace pyramid_side_length_l422_422196

noncomputable def side_length_of_square_base (area_of_lateral_face : ℝ) (slant_height : ℝ) : ℝ :=
  2 * area_of_lateral_face / slant_height

theorem pyramid_side_length 
  (area_of_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_of_lateral_face = 120)
  (h2 : slant_height = 24) :
  side_length_of_square_base area_of_lateral_face slant_height = 10 :=
by
  -- Skipping the proof details.
  sorry

end pyramid_side_length_l422_422196


namespace highest_price_day_weekly_profit_cost_effective_method_l422_422280

-- Define the conditions
def standard_price : ℕ := 10
def cost_price : ℕ := 8

def relative_prices : List ℤ := [+1, -2, +3, -1, +2, +5, -4]
def quantities : List ℕ := [20, 35, 10, 30, 15, 5, 50]

def promo1_price (n : ℕ) : ℕ :=
  if n <= 5 then 12 * n else 12 * 5 + (n - 5) * (12 * 8 / 10)
def promo2_price (n : ℕ) : ℕ := 10 * n

-- Define the proofs
theorem highest_price_day : 
  (relative_prices.zipWith (fun rp q => standard_price + rp) quantities).max = 15 :=
by sorry

theorem weekly_profit : 
  let price_difference := relative_prices.zipWith (fun rp q => rp * q) quantities
      sales_profit := (standard_price - cost_price) * quantities.sum
  in price_difference.sum + sales_profit = 135 :=
by sorry

theorem cost_effective_method : 
  min (promo1_price 35) (promo2_price 35) = promo1_price 35 :=
by sorry

end highest_price_day_weekly_profit_cost_effective_method_l422_422280


namespace polynomial_remainder_degrees_l422_422994

-- Define the divisor polynomial
def divisor := 3 * (x^3) - 4 * (x^2) + x - 5

-- Define a polynomial division remainder property
def valid_remainder_degrees :=
  {d : ℕ | d < degree divisor}

theorem polynomial_remainder_degrees :
  valid_remainder_degrees = {0, 1, 2} :=
by
  sorry

end polynomial_remainder_degrees_l422_422994


namespace max_white_dominos_l422_422739

def max_dominos (board : Type) [fintype board] (marked_cells : finset board) (size_domino : finset board → Prop) :=
  ∀ (d : finset board), size_domino d → d.card = 2 ∧ d.subset marked_cells ∈ {}

theorem max_white_dominos {board : Type} [fintype board] (cells : finset board) (marked_cells : finset board) :
  (∀ (d : finset board), d ∈ marked_cells → d.card = 2 ∧ d.card = 2 → ∀ (b : board), max_dominos board d) →
  marked_cells.card = 16 → 
  max_dominos cells = 16 := 
  by 
  sorry

end max_white_dominos_l422_422739


namespace sin2θ_value_l422_422043

theorem sin2θ_value (θ : Real) (h1 : Real.sin θ = 4/5) (h2 : Real.sin θ - Real.cos θ > 1) : Real.sin (2*θ) = -24/25 := 
by 
  sorry

end sin2θ_value_l422_422043


namespace C_younger_than_A_l422_422268

variables (A B C : ℕ)

-- Original Condition
axiom age_condition : A + B = B + C + 17

-- Lean Statement to Prove
theorem C_younger_than_A (A B C : ℕ) (h : A + B = B + C + 17) : C + 17 = A :=
by {
  -- Proof would go here but is omitted.
  sorry
}

end C_younger_than_A_l422_422268


namespace problem_inequality_l422_422912

theorem problem_inequality (a b c d : ℝ) (h1 : d ≥ 0) (h2 : a + b = 2) (h3 : c + d = 2) :
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 :=
by sorry

end problem_inequality_l422_422912


namespace number_of_pairs_l422_422369

def remainder (p q : ℝ) : ℝ :=
  let r := p % q in if r < 0 then r + q else r

def r1 (a b : ℕ) : ℝ := remainder (a * Real.sqrt 2 + b * Real.sqrt 3) (Real.sqrt 2)
def r2 (a b : ℕ) : ℝ := remainder (a * Real.sqrt 2 + b * Real.sqrt 3) (Real.sqrt 3)

theorem number_of_pairs :
  (finset.card (finset.filter (λ (ab : ℕ × ℕ), ab.1 ≤ 20 ∧ ab.2 ≤ 20 ∧ r1 ab.1 ab.2 + r2 ab.1 ab.2 = Real.sqrt 2)
  ((finset.range 21).product (finset.range 21)))) = 16 :=
  sorry

end number_of_pairs_l422_422369


namespace two_class_students_l422_422472

-- Define the types of students and total sum variables
variables (H M E HM HE ME HME : ℕ)
variable (Total_Students : ℕ)

-- Given conditions
axiom condition1 : Total_Students = 68
axiom condition2 : H = 19
axiom condition3 : M = 14
axiom condition4 : E = 26
axiom condition5 : HME = 3

-- Inclusion-Exclusion principle formula application
def exactly_two_classes : Prop := 
  Total_Students = H + M + E - (HM + HE + ME) + HME

-- Theorem to prove the number of students registered for exactly two classes is 6
theorem two_class_students : H + M + E - 2 * HME + HME - (HM + HE + ME) = 6 := by
  sorry

end two_class_students_l422_422472


namespace phone_earning_is_10_l422_422810

-- Define the conditions as variables
variables (P : ℝ) 
          (earned_per_laptop : ℝ) 
          (phones_monday : ℝ) 
          (phones_tuesday : ℝ) 
          (laptops_wednesday : ℝ) 
          (laptops_thursday : ℝ) 
          (total_earned : ℝ)

-- Assume the given conditions
axiom phone_earning : P
axiom laptop_earning : earned_per_laptop = 20
axiom monday_phones : phones_monday = 3
axiom tuesday_phones : phones_tuesday = 5
axiom wednesday_laptops : laptops_wednesday = 2
axiom thursday_laptops : laptops_thursday = 4
axiom total_income : total_earned = 200

-- The theorem to prove
theorem phone_earning_is_10 :
  ∀ P earned_per_laptop phones_monday phones_tuesday laptops_wednesday laptops_thursday total_earned,
    (earned_per_laptop = 20) →
    (phones_monday = 3) →
    (phones_tuesday = 5) →
    (laptops_wednesday = 2) →
    (laptops_thursday = 4) →
    (total_earned = 200) →
    (phones_monday * P + phones_tuesday * P + laptops_wednesday * earned_per_laptop + laptops_thursday * earned_per_laptop = total_earned) →
    (P = 10) :=
by {
  intros P earned_per_laptop phones_monday phones_tuesday laptops_wednesday laptops_thursday total_earned,
  intros hp1 hp2 hp3 hp4 ht income_eq,
  simp at *,
  sorry
}

end phone_earning_is_10_l422_422810


namespace smallest_whole_number_for_inequality_l422_422241

theorem smallest_whole_number_for_inequality:
  ∃ (x : ℕ), (2 : ℝ) / 5 + (x : ℝ) / 9 > 1 ∧ ∀ (y : ℕ), (2 : ℝ) / 5 + (y : ℝ) / 9 > 1 → x ≤ y :=
by
  sorry

end smallest_whole_number_for_inequality_l422_422241


namespace f_strictly_increasing_intervals_g_range_on_interval_l422_422073

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * sin x ^ 2
noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x)

-- Proof for the first part: intervals where f(x) is strictly increasing
theorem f_strictly_increasing_intervals (k : ℤ) :
  ∀ x, -π/6 + k*π ≤ x ∧ x ≤ π/3 + k*π → f x ⋖ f (x + 1) := sorry

-- Proof for the second part: range of g(x) on the given interval
theorem g_range_on_interval : 
  let I := Set.Icc (-π/6 : ℝ) (π/3 : ℝ) in
  Set.range (g ∘ (fun x => x ∈ I)) = Set.Icc (-√3 : ℝ) 2 := sorry

end f_strictly_increasing_intervals_g_range_on_interval_l422_422073


namespace circle_equation_unique_l422_422941

theorem circle_equation_unique {F D E : ℝ} : 
  (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 4 ∧ y = 2) → x^2 + y^2 + D * x + E * y + F = 0) → 
  (x^2 + y^2 - 8 * x + 6 * y = 0) :=
by 
  sorry

end circle_equation_unique_l422_422941


namespace cards_problem_l422_422530

-- Definitions of the cards and their arrangement
def cards : List ℕ := [1, 3, 4, 6, 7, 8]
def missing_numbers : List ℕ := [2, 5, 9]

-- Function to check no three consecutive numbers are in ascending or descending order
def no_three_consec (ls : List ℕ) : Prop :=
  ∀ (a b c : ℕ), a < b → b < c → b - a = 1 → c - b = 1 → False ∧
                a > b → b > c → a - b = 1 → b - c = 1 → False

-- Assume that cards A, B, and C are not visible
variables (A B C : ℕ)

-- Ensure that A, B, and C are among the missing numbers
axiom A_in_missing : A ∈ missing_numbers
axiom B_in_missing : B ∈ missing_numbers
axiom C_in_missing : C ∈ missing_numbers

-- Ensuring no three consecutive cards are in ascending or descending order
axiom no_three_consec_cards : no_three_consec (cards ++ [A, B, C])

-- The final proof problem
theorem cards_problem : A = 5 ∧ B = 2 ∧ C = 9 :=
by
  sorry

end cards_problem_l422_422530


namespace derivative_at_x0_l422_422774

variable {f : ℝ → ℝ} -- declaration of a differentiable function

-- Define the differentiability of f and their properties
def differentiable_f := differentiable ℝ f

-- Define the property that f(x) is odd
def odd_f := ∀ x : ℝ, f (-x) = -f x

-- Define a specific point x_0 in ℝ and a parameter k
variable (x_0 : ℝ) (k : ℝ)

-- State the condition f'(-x₀) = k
def condition_at_minus_x0 := f' (-x_0) = k

-- The main theorem we aim to prove: f'(x₀) = k
theorem derivative_at_x0 (df : differentiable_f) (of : odd_f) (cond : condition_at_minus_x0) (h_k : k ≠ 0) : f' x_0 = k :=
by {
  sorry,
}

end derivative_at_x0_l422_422774


namespace club_officers_selection_l422_422290

theorem club_officers_selection :
  let total_members := 12 in
  let eligible_treasurers := 5 in
  let ways_to_select_officers := total_members * (total_members - 1) * eligible_treasurers * (total_members - 2) in
  ways_to_select_officers = 6600 :=
by
  let total_members := 12
  let eligible_treasurers := 5
  let ways_to_select_officers := total_members * (total_members - 1) * eligible_treasurers * (total_members - 2)
  show ways_to_select_officers = 6600, from sorry

end club_officers_selection_l422_422290


namespace polynomial_evaluation_sum_l422_422135

theorem polynomial_evaluation_sum :
  ∃ (q_1 q_2 : ℤ[X]), -- ∃ indicates the existence of polynomials
  (∀ i, monic q_i ∧ (∀ r s: ℤ[X], q_i ≠ r * s ∨ is_unit r ∨ is_unit s)) ∧
  (X^5 - 3*X^2 - X - 1 = q_1 * q_2) ∧
  (q_1.eval 2 + q_2.eval 2 = 6) :=
sorry

end polynomial_evaluation_sum_l422_422135


namespace greatest_distance_sum_l422_422140

-- Definitions of points and the context of the problem
variables {A B C D E : ℝ × ℝ} -- Points in ℝ^2 representing vertices and intersection points
variables {DE : set (ℝ × ℝ)} -- Line segment DE 

-- Assumptions
-- Let D be the intersection of the angle bisector of ∠BAC with side BC
def angle_bisector_intersection_1 (A B C D : ℝ × ℝ) : Prop := 
  is_on_angle_bisector A C B D -- D is on the angle bisector of ∠BAC

-- Let E be the intersection of the angle bisector of ∠ABC with side CA
def angle_bisector_intersection_2 (A B E C : ℝ × ℝ) : Prop := 
  is_on_angle_bisector B C A E -- E is on the angle bisector of ∠ABC

-- Statement to be proven: The greatest distance from any point on line DE to the sides of the triangle 
-- equals the sum of the other two distances
theorem greatest_distance_sum 
  (hD : angle_bisector_intersection_1 A B C D)
  (hE : angle_bisector_intersection_2 A B E C)
  (P: ℝ × ℝ)
  (hP_on_DE : P ∈ DE) :
  ∃ d₁ d₂ d₃ : ℝ, is_distance_from P (A, B, C) d₁ d₂ d₃ ∧ 
  max d₁ d₂ d₃ = d₁ + d₂ ∨ max d₁ d₂ d₃ = d₁ + d₃ ∨ max d₁ d₂ d₃ = d₂ + d₃ := 
sorry

end greatest_distance_sum_l422_422140


namespace maximum_sum_each_side_equals_22_l422_422732

theorem maximum_sum_each_side_equals_22 (A B C D : ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ 10)
  → (∀ S, S = A ∨ S = B ∨ S = C ∨ S = D ∧ A + B + C + D = 33)
  → (A + B + C + D + 55) / 4 = 22 :=
by
  sorry

end maximum_sum_each_side_equals_22_l422_422732


namespace journey_total_time_l422_422326

-- Define the total journey distance and speeds for each half of the journey.
def total_journey_distance : ℝ := 448
def speed_first_half : ℝ := 21
def speed_second_half : ℝ := 24

-- Define the proof statement to show that the total time is 20 hours.
theorem journey_total_time : 
  let distance_half := total_journey_distance / 2 in
  let time_first_half := distance_half / speed_first_half in
  let time_second_half := distance_half / speed_second_half in
  time_first_half + time_second_half = 20 := 
by {
  sorry -- proof goes here
}

end journey_total_time_l422_422326


namespace polygon_with_interior_sum_1260_eq_nonagon_l422_422953

theorem polygon_with_interior_sum_1260_eq_nonagon :
  ∃ n : ℕ, (n-2) * 180 = 1260 ∧ n = 9 := by
  sorry

end polygon_with_interior_sum_1260_eq_nonagon_l422_422953


namespace foot_of_altitude_proof_l422_422521

open Complex

noncomputable def foot_of_altitude (a b c : ℂ) : ℂ :=
  1 / 2 * (a + b + c - (conj a) * b * c)

theorem foot_of_altitude_proof 
  (a b c : ℂ) 
  (ha : abs a = 1) 
  (hb : abs b = 1) 
  (hc : abs c = 1) : 
  foot_of_altitude a b c = 1 / 2 * (a + b + c - conj a * b * c) := 
by
  sorry

end foot_of_altitude_proof_l422_422521


namespace lcm_prod_eq_factorial_l422_422911

theorem lcm_prod_eq_factorial (n : ℕ) (h : n > 0) :
    (∏ k in Finset.range (n + 1).filter (λ k, k > 0), Nat.lcm (Finset.range (⌊n / k⌋ + 1)).filter (λ x, x > 0)) = n! := 
by 
  sorry

end lcm_prod_eq_factorial_l422_422911


namespace S_k_plus_1_l422_422879

theorem S_k_plus_1 (k : ℕ) (hk : k ≥ 3) :
  let S_k := (∑ i in finset.range (2 * k - (k + 2) + 1), (1 : ℝ) / (k + 2 + i)) 
  let S_k_plus_1 := (∑ i in finset.range (2 * k + 1 - (k + 3) + 1), (1 : ℝ) / (k + 3 + i)) 
  S_k_plus_1 = S_k + 1 / (2 * k) + 1 / (2 * k + 1) - 1 / (k + 2) :=
by
  let S_k := (∑ i in finset.range (2 * k - (k + 2) + 1), (1 : ℝ) / (k + 2 + i))
  let S_k_plus_1 := (∑ i in finset.range (2 * k + 1 - (k + 3) + 1), (1 : ℝ) / (k + 3 + i))
  sorry

end S_k_plus_1_l422_422879


namespace largest_good_number_number_of_absolute_good_numbers_l422_422827

-- Conditions for a good number
def is_good_number (N a b : ℕ) : Prop :=
  N = ab + a + b

-- Conditions for an absolute good number (case when ab/(a+b) = 3 or 4)
def is_absolute_good_number (N a b : ℕ) : Prop :=
  is_good_number N a b ∧ (ab = 3 * (a + b) ∨ ab = 4 * (a + b))

-- Prove that the largest good number is 99
theorem largest_good_number :
  ∃ N, (∃ a b, is_good_number N a b) ∧ 10 ≤ N ∧ N ≤ 99 ∧ ∀ M, (∃ a b, is_good_number M a b) ∧ 10 ≤ M ∧ M ≤ 99 → M ≤ N :=
  sorry

-- Prove that the number of absolute good numbers is 39
theorem number_of_absolute_good_numbers :
  ∃ count, count = 39 ∧ count = (∑ N in (finset.Icc 10 99), (∃ a b, is_absolute_good_number N a b)) :=
  sorry

end largest_good_number_number_of_absolute_good_numbers_l422_422827


namespace minimum_distance_sum_l422_422412

noncomputable def parabola := { P : ℝ × ℝ // P.2^2 = 8 * P.1 }
def focus : ℝ × ℝ := (2, 0)
def point_A : ℝ × ℝ := (5, 2)

def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem minimum_distance_sum (P : parabola) :
  distance ⟨P.val.1, P.val.2⟩ point_A + distance ⟨P.val.1, P.val.2⟩ focus ≥ 7 :=
sorry

end minimum_distance_sum_l422_422412


namespace triangle_CD_sqrt_mn_l422_422498

variable {A B C D : Type} [Real A] [Real B] [Real C] [Real D]

theorem triangle_CD_sqrt_mn
  (ABC : Triangle A B C)
  (A_ne_BC : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (external_angle_bisector : bisector (external_angle_of_triangle ABC C))
  (D_on_extension_BA : lies_between B A (extension D (line_segment B A)))
  (BD_minus_BC_eq_m : BD - BC = m)
  (AC_plus_AD_eq_n : AC + AD = n) :
  CD = Real.sqrt (m * n) := 
sorry

end triangle_CD_sqrt_mn_l422_422498


namespace increasing_interval_of_f_l422_422724

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem increasing_interval_of_f :
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 4) → MonotoneOn f (Set.Icc (-3 * Real.pi / 4) (Real.pi / 4)) :=
by
  sorry

end increasing_interval_of_f_l422_422724


namespace perimeter_of_T_shape_l422_422936

-- Define conditions as hypotheses
variables (A : ℝ) (n : ℕ) (a : ℝ)

-- Given hypotheses
axiom area_of_figure : A = 144
axiom number_of_squares : n = 6
axiom area_per_square : A / n = a * a

-- Target to prove the perimeter of the figure
theorem perimeter_of_T_shape (A : ℝ) (n : ℕ) (a : ℝ) 
    (h1 : A = 144)
    (h2 : n = 6)
    (h3 : A / n = a * a) :
    7 * a = 14 * real.sqrt 6 :=
sorry

end perimeter_of_T_shape_l422_422936


namespace percentage_altered_votes_got_is_50_l422_422029

def original_votes_got := 10
def original_votes_twilight := 12
def original_votes_art_of_deal := 20

def votes_after_tampering :=
  original_votes_got +
  (original_votes_twilight / 2) +
  (original_votes_art_of_deal * 0.2)

def percentage_votes_got :=
  (original_votes_got / votes_after_tampering) * 100

theorem percentage_altered_votes_got_is_50 :
  percentage_votes_got = 50 := by
  sorry

end percentage_altered_votes_got_is_50_l422_422029


namespace scenario_1_scenario_2_scenario_3_scenario_4_l422_422234

-- Definitions based on conditions
def prob_A_hit : ℚ := 2 / 3
def prob_B_hit : ℚ := 3 / 4

-- Scenario 1: Prove that the probability of A shooting 3 times and missing at least once is 19/27
theorem scenario_1 : 
  (1 - (prob_A_hit ^ 3)) = 19 / 27 :=
by sorry

-- Scenario 2: Prove that the probability of A hitting the target exactly 2 times and B hitting the target exactly 1 time after each shooting twice is 1/6
theorem scenario_2 : 
  (2 * ((prob_A_hit ^ 2) * (1 - prob_A_hit)) * (2 * (prob_B_hit * (1 - prob_B_hit)))) = 1 / 6 :=
by sorry

-- Scenario 3: Prove that the probability of A missing the target and B hitting the target 2 times after each shooting twice is 1/16
theorem scenario_3 :
  ((1 - prob_A_hit) ^ 2) * (prob_B_hit ^ 2) = 1 / 16 :=
by sorry

-- Scenario 4: Prove that the probability that both A and B hit the target once after each shooting twice is 1/6
theorem scenario_4 : 
  (2 * (prob_A_hit * (1 - prob_A_hit)) * 2 * (prob_B_hit * (1 - prob_B_hit))) = 1 / 6 :=
by sorry

end scenario_1_scenario_2_scenario_3_scenario_4_l422_422234


namespace exists_integer_a_l422_422934

theorem exists_integer_a
  (n : ℕ) (k : ℕ) (h_k : k ≥ 2)
  (x : Fin n → ℝ) (h_x : ∑ i, x i ^ 2 = 1) :
  ∃ (a : Fin n → ℤ), (|a i| ≤ (k - 1) ∧
   |∑ i, a i * (x i : ℝ)| ≤ ((k - 1) * real.sqrt n) / (k ^ n - 1)) :=
begin
  sorry
end

end exists_integer_a_l422_422934


namespace ellipse_equation_l422_422768

-- Definition of ellipse parameters according to the problem conditions
def semiMajorAxis (a : ℝ) := a > 0
def semiMinorAxis (b : ℝ) := b > 0
def eccentricity (a b : ℝ) := 1/3 = Real.sqrt (1 - (b^2)/(a^2))

-- Given conditions as hypotheses
variables {a b x y : ℝ}
hypothesis a_gt_b : a > b
hypothesis b_gt_0 : b > 0
hypothesis ecc : eccentricity a b

-- Result equation of the ellipse
theorem ellipse_equation : 
  ∃ (a b : ℝ), semiMajorAxis a ∧ semiMinorAxis b ∧ eccentricity a b ∧ 
  ((∃ m : ℝ, a = 3 * m ∧ b = 2 * Real.sqrt 2 * m) → ∀ x y, 
  (x^2 / (a^2) + y^2 / (b^2) = 1 ↔ x^2 / 9 + y^2 / 8 = 1)) :=
begin
  sorry -- Proof omitted
end

end ellipse_equation_l422_422768


namespace common_ratio_of_sequence_l422_422664

theorem common_ratio_of_sequence 
  (a1 a2 a3 a4 : ℤ)
  (h1 : a1 = 25)
  (h2 : a2 = -50)
  (h3 : a3 = 100)
  (h4 : a4 = -200)
  (is_geometric : ∀ (i : ℕ), a1 * (-2) ^ i = if i = 0 then a1 else if i = 1 then a2 else if i = 2 then a3 else a4) : 
  (-50 / 25 = -2) ∧ (100 / -50 = -2) ∧ (-200 / 100 = -2) :=
by 
  sorry

end common_ratio_of_sequence_l422_422664


namespace find_a_l422_422789

def f (x : ℝ) : ℝ :=
  if x >= 0 then 2^x - 1 else -x^2 - 2*x

theorem find_a (a : ℝ) : f(a) = 1 ↔ a = 1 ∨ a = -1 := by
  sorry

end find_a_l422_422789


namespace spies_denounced_each_other_l422_422582

theorem spies_denounced_each_other :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 10 ∧ 
  (∀ (u v : ℕ), (u, v) ∈ pairs → (v, u) ∈ pairs) :=
sorry

end spies_denounced_each_other_l422_422582


namespace ellipse_equation_l422_422767

-- Definition of ellipse parameters according to the problem conditions
def semiMajorAxis (a : ℝ) := a > 0
def semiMinorAxis (b : ℝ) := b > 0
def eccentricity (a b : ℝ) := 1/3 = Real.sqrt (1 - (b^2)/(a^2))

-- Given conditions as hypotheses
variables {a b x y : ℝ}
hypothesis a_gt_b : a > b
hypothesis b_gt_0 : b > 0
hypothesis ecc : eccentricity a b

-- Result equation of the ellipse
theorem ellipse_equation : 
  ∃ (a b : ℝ), semiMajorAxis a ∧ semiMinorAxis b ∧ eccentricity a b ∧ 
  ((∃ m : ℝ, a = 3 * m ∧ b = 2 * Real.sqrt 2 * m) → ∀ x y, 
  (x^2 / (a^2) + y^2 / (b^2) = 1 ↔ x^2 / 9 + y^2 / 8 = 1)) :=
begin
  sorry -- Proof omitted
end

end ellipse_equation_l422_422767


namespace bc_length_l422_422533

-- Given conditions
variables (O A M B C : Point) (r : ℝ) (alpha : ℝ)
hypothesis (is_radius : dist O A = r)
hypothesis (on_radius : dist A M = (dist M O - r))
hypothesis (points_on_circle : dist O B = r ∧ dist O C = r)
hypothesis (angle_AMB_OMC : ∠ AMB = alpha ∧ ∠ OMC = alpha)
hypothesis (radius_value : r = 8)
hypothesis (cos_alpha : cos alpha = 3 / 4)

-- The goal is to prove BC = 12
theorem bc_length : dist B C = 12 :=
by
  sorry

end bc_length_l422_422533


namespace relationship_OA_OB_l422_422428

def hyperbola_center (a b : ℝ) (x y : ℝ) : Prop := 
  x * x / (a * a) - y * y / (b * b) = 1

variables (a b : ℝ) -- semi-major axis a and semi-minor axis b
variables (F1 F2 O P I A B : ℝ × ℝ) -- Points in the plane
variables (e : ℝ) -- eccentricity of the hyperbola

def is_foci (O F1 F2 : ℝ × ℝ) (e : ℝ) : Prop := 
  ∃ (x₁ x₂ : ℝ), O = (0, 0) ∧ F1 = (x₁, 0) ∧ F2 = (-x₂, 0) ∧ |x₁| = |x₂| ∧ |x₁| = e * sqrt (a * a + b * b)

def incircle_tangent_points(I A B : ℝ × ℝ) (r : ℝ) : Prop :=
  dist I A = r ∧ dist I B = r ∧ I.2 = r

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1)/2, (P.2 + Q.2)/2)

def are_midpoints (O A B : ℝ × ℝ) : Prop :=
  I = midpoint F1 F2 ∧ O = (0, 0)

theorem relationship_OA_OB 
  (h1 : hyperbola_center a b O.1 O.2)
  (h2 : is_foci O F1 F2 e)
  (h3 : incircle_tangent_points I A B r)
  (h4 : are_midpoints O A B) :
  dist O A = dist O B :=
sorry

end relationship_OA_OB_l422_422428


namespace tangent_segment_theorem_distance_segments_equal_l422_422853

theorem tangent_segment_theorem 
  (A B C D E F G H I K L: ℝ)
  (h_tangents : tangent A I K = tangent A K I)
  (h_AB : tangent A B = tangent A I) 
  (h_BD : tangent B D = tangent B H) 
  (h_GE : tangent G E = tangent C E) 
  (h_EF : tangent E F = tangent D E) : 
  BC = DE :=
begin
  sorry
end

theorem distance_segments_equal 
  (A G C K D L F H B J I M T: ℝ)
  (h_parallel_AG_CK : parallel AG CK)
  (h_parallel_DL_FH : parallel DL FH)
  (h_perpendicular_AG_MT : perpendicular AG MT)
  (h_perpendicular_CK_MT : perpendicular CK MT)
  (h_perpendicular_DL_MT : perpendicular DL MT)
  (h_perpendicular_FH_MT : perpendicular FH MT): 
  distance AG CK = distance DL FH :=
begin
  sorry
end

end tangent_segment_theorem_distance_segments_equal_l422_422853


namespace maria_sister_drank_l422_422160

-- Define the conditions
def initial_bottles : ℝ := 45.0
def maria_drank : ℝ := 14.0
def remaining_bottles : ℝ := 23.0

-- Define the problem statement to prove the number of bottles Maria's sister drank
theorem maria_sister_drank (initial_bottles maria_drank remaining_bottles : ℝ) : 
    (initial_bottles - maria_drank) - remaining_bottles = 8.0 :=
by
  sorry

end maria_sister_drank_l422_422160


namespace equation_of_line_perpendicular_l422_422302

theorem equation_of_line_perpendicular 
  (P : ℝ × ℝ) (hx : P.1 = -1) (hy : P.2 = 2)
  (a b c : ℝ) (h_line : 2 * a - 3 * b + 4 = 0)
  (l : ℝ → ℝ) (h_perpendicular : ∀ x, l x = -(3/2) * x)
  (h_passing : l (-1) = 2)
  : a * 3 + b * 2 - 1 = 0 :=
sorry

end equation_of_line_perpendicular_l422_422302


namespace cos_value_correct_l422_422037

noncomputable def cos_value (α : ℝ) (h : sin (π / 6 + α) = 1 / 3) : ℝ :=
  cos (2 * π / 3 - 2 * α)

theorem cos_value_correct (α : ℝ) (h : sin (π / 6 + α) = 1 / 3) :
  cos_value α h = -7 / 9 :=
sorry

end cos_value_correct_l422_422037


namespace sum_of_valid_c_l422_422875

noncomputable def f (n : ℕ) : ℕ :=
if n % 3 = 0 then n / 3 else 4 * n - 10

def iter5_f (n : ℕ) : ℕ := f (f (f (f (f n))))

theorem sum_of_valid_c : 
  let valid_c := {c : ℕ | iter5_f c = 2} in 
  ∑ c in valid_c, c = 235 :=
sorry

end sum_of_valid_c_l422_422875


namespace incorrect_statement_b_l422_422621

/--
Given:
1. For random variables ξ and η satisfying η = 2ξ + 3, the variance D(η) = 4D(ξ).
2. In regression analysis, the smaller the value of R², the worse the model fit and the larger the sum of squared residuals.
3. If residual points are uniformly distributed within a horizontal band, a narrower band indicates higher prediction accuracy.
4. The regression line always passes through the center of the sample points.

Prove that option B is incorrect.
-/
theorem incorrect_statement_b (ξ η : ℝ) 
  (cond1 : η = 2 * ξ + 3) 
  (cond2 : ∀ (R² : ℝ), R² < 1 → worse_fit R² ∧ larger_residual_sum R²)
  (cond3 : ∀ (band_width : ℝ), narrower_band band_width → higher_accuracy band_width)
  (cond4 : regression_line_passes_center) : 
  incorrect_statement B := 
sorry

end incorrect_statement_b_l422_422621


namespace min_path_triangle_l422_422834

/--
In triangle XYZ, given that:
1. ∠XYZ = 30°
2. XY = 12
3. XZ = 8
4. Points P and Q lie on XY and XZ respectively,
prove that the minimum possible value of YP + PQ + QZ is equal to √(208 + 96√3).
-/
theorem min_path_triangle 
(α β γ : ℝ) 
(h1 : α + β + γ = π / 6) 
(h2 : XY = 12) 
(h3 : XZ = 8) 
(P Q : ℝ) 
(hP : P ∈ segment XY) 
(hQ : Q ∈ segment XZ) : 
  ∃ YP PQ QZ : ℝ, YP + PQ + QZ = √(208 + 96√3) :=
sorry

end min_path_triangle_l422_422834


namespace intersection_M_N_eq_A_l422_422894

def M : Set ℝ := {x : ℝ | x^2 - 3 * x - 10 < 0}
def N : Set ℤ := {x : ℤ | abs x < 2}
def A : Set ℤ := {-1, 0, 1}

theorem intersection_M_N_eq_A : (M ∩ N : Set ℤ) = A := by
  sorry

end intersection_M_N_eq_A_l422_422894


namespace number_of_memorable_telephone_numbers_l422_422343

-- Definitions for readability
def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def is_memorable (d : ℕ → ℕ) : Prop :=
  (d 0 = d 4 ∧ d 1 = d 5 ∧ d 2 = d 6 ∧ d 3 = d 7) ∨
  (d 0 = d 5 ∧ d 1 = d 6 ∧ d 2 = d 7 ∧ d 3 = d 8)

-- Theorem: Number of memorable telephone numbers satisfying the conditions
theorem number_of_memorable_telephone_numbers :
  (finset.univ.image (λ d : fin (9), d)).filter is_memorable).card = 199990 := by
  sorry

end number_of_memorable_telephone_numbers_l422_422343


namespace solve_inequality_l422_422551

theorem solve_inequality : 
  {x : ℝ} → -3 * x^2 + 5 * x + 4 < 0 → x ∈ set.Ioo (-4/3 : ℝ) (1 : ℝ) := 
by
  intros x h
  sorry

end solve_inequality_l422_422551


namespace binary_product_correct_l422_422021

theorem binary_product_correct :
  let x := nat.of_digits 2 [1, 1, 0, 1, 1]
  let y := nat.of_digits 2 [1, 1, 1, 1]
  let product := nat.of_digits 2 [1, 0, 1, 1, 1, 1, 0, 1]
  x * y = product := by sorry

end binary_product_correct_l422_422021


namespace area_ratio_of_concentric_circles_l422_422983

noncomputable theory

-- Define the given conditions
def C1 (r1 : ℝ) : ℝ := 2 * Real.pi * r1
def C2 (r2 : ℝ) : ℝ := 2 * Real.pi * r2
def arc_length (angle : ℝ) (circumference : ℝ) : ℝ := (angle / 360) * circumference

-- Lean statement for the math proof problem
theorem area_ratio_of_concentric_circles 
  (r1 r2 : ℝ) (h₁ : arc_length 60 (C1 r1) = arc_length 48 (C2 r2)) : 
  (Real.pi * r1^2) / (Real.pi * r2^2) = 16 / 25 :=
by
  sorry  -- Proof omitted

end area_ratio_of_concentric_circles_l422_422983


namespace prove_a_leq_neg_2sqrt2_l422_422460

def f (x a : ℝ) : ℝ := Real.log x + (1 / 2) * x^2 + a * x

noncomputable def extreme_points (a : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), x1^2 + a * x1 + 1 = 0 ∧ x2^2 + a * x2 + 1 = 0 ∧ x1 ≠ x2

theorem prove_a_leq_neg_2sqrt2 (a : ℝ) (h_extrema : extreme_points a) (h_sum : f h_extrema.fst a + f h_extrema.snd a ≤ -5) : 
  a ≤ -2 * Real.sqrt 2 := sorry

end prove_a_leq_neg_2sqrt2_l422_422460


namespace rocket_engine_jet_speed_l422_422122

-- Definitions based on conditions
def v (v0 : ℝ) (m1 m2 : ℝ) := v0 * Real.log ((m1 + m2) / m1)

variables (m1 m2 : ℝ) (v : ℝ := 10)
#check Real.log -- checking if Real.log is properly imported
axiom m1_two_m2 : m1 = 2 * m2
axiom log_two : Real.log 2 = 0.7
axiom log_three : Real.log 3 = 1.1

theorem rocket_engine_jet_speed
  (v : ℝ := 10)
  (v0 : ℝ)
  (m1 m2 : ℝ)
  (h_m1 : m1 = 2 * m2)
  (h_v : v = v0 * Real.log ((m1 + m2) / m1))
  (h_log_two : Real.log 2 = 0.7)
  (h_log_three : Real.log 3 = 1.1) : v0 = 25 :=
by sorry

end rocket_engine_jet_speed_l422_422122


namespace smallest_b_l422_422917

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : a + b = 7) (h4 : 2 + a ≤ b) : b = 9 / 2 :=
by
  sorry

end smallest_b_l422_422917


namespace total_garbage_collected_l422_422350

def Daliah := 17.5
def Dewei := Daliah - 2
def Zane := 4 * Dewei
def Bela := Zane + 3.75

theorem total_garbage_collected :
  Daliah + Dewei + Zane + Bela = 160.75 :=
by
  sorry

end total_garbage_collected_l422_422350


namespace cone_prism_ratio_l422_422314

noncomputable def cone_prism_volume_ratio (w h : ℝ) : ℝ :=
  let V_cone := (1 / 3) * π * w^2 * h
  let V_prism := 2 * w^2 * h
  V_cone / V_prism

theorem cone_prism_ratio (w h : ℝ) (hw : w ≠ 0) (hh : h ≠ 0) : cone_prism_volume_ratio w h = (π / 6) :=
by
  have V_cone := (1 / 3) * π * w^2 * h
  have V_prism := 2 * w^2 * h
  have ratio := V_cone / V_prism
  calc ratio = (1 / 3) * π * w^2 * h / (2 * w^2 * h) : by sorry
            ... = π / 6 : by sorry

end cone_prism_ratio_l422_422314


namespace number_of_impossible_d_vals_is_infinite_l422_422575

theorem number_of_impossible_d_vals_is_infinite
  (t_1 t_2 s d : ℕ)
  (h1 : 2 * t_1 + t_2 - 4 * s = 4041)
  (h2 : t_1 = s + 2 * d)
  (h3 : t_2 = s + d)
  (h4 : 4 * s > 0) :
  ∀ n : ℕ, n ≠ 808 * 5 ↔ ∃ d, d > 0 ∧ d ≠ n :=
sorry

end number_of_impossible_d_vals_is_infinite_l422_422575


namespace ellipse_properties_l422_422400

noncomputable def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def is_focus_on_directrix (focus : ℝ × ℝ) : Prop := focus = (-1, 0)

def on_line (A : ℝ × ℝ) (m : ℝ) (c : ℝ) : Prop := A.2 = m * A.1 + c

theorem ellipse_properties :
  (∃ (focus : ℝ × ℝ), is_focus_on_directrix focus) ∧
  (∃ (P : ℝ × ℝ), P = (1, 3/2) ∧ ellipse_equation P.1 P.2) ∧
  (∀ (A B : ℝ × ℝ) (m : ℝ), m = 1/2 ∧ on_line A m (k ≠ 1) ∧ on_line B m (k ≠ 1) →
    let k1 := (A.2 - 3/2) / (A.1 - 1)
        k2 := (B.2 - 3/2) / (B.1 - 1)
    in k1 + k2 = 0) :=
begin
  sorry
end

end ellipse_properties_l422_422400


namespace x_add_inv_ge_two_x_add_inv_eq_two_iff_l422_422545

theorem x_add_inv_ge_two {x : ℝ} (h : 0 < x) : x + (1 / x) ≥ 2 :=
sorry

theorem x_add_inv_eq_two_iff {x : ℝ} (h : 0 < x) : (x + (1 / x) = 2) ↔ (x = 1) :=
sorry

end x_add_inv_ge_two_x_add_inv_eq_two_iff_l422_422545


namespace students_in_each_normal_class_l422_422444

theorem students_in_each_normal_class
  (total_students : ℕ)
  (percentage_moving : ℝ)
  (grades : ℕ)
  (adv_class_size : ℕ)
  (num_normal_classes : ℕ)
  (h1 : total_students = 1590)
  (h2 : percentage_moving = 0.4)
  (h3 : grades = 3)
  (h4 : adv_class_size = 20)
  (h5 : num_normal_classes = 6) :
  ((total_students * percentage_moving).toNat / grades - adv_class_size) / num_normal_classes = 32 := 
by sorry

end students_in_each_normal_class_l422_422444


namespace intersection_points_in_plane_l422_422198

-- Define the cones with parallel axes and equal angles
def cone1 (a1 b1 c1 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a1)^2 + (y - b1)^2 = k^2 * (z - c1)^2

def cone2 (a2 b2 c2 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a2)^2 + (y - b2)^2 = k^2 * (z - c2)^2

-- Given conditions
variable (a1 b1 c1 a2 b2 c2 k : ℝ)

-- The theorem to be proven
theorem intersection_points_in_plane (x y z : ℝ) 
  (h1 : cone1 a1 b1 c1 k x y z) (h2 : cone2 a2 b2 c2 k x y z) : 
  ∃ (A B C D : ℝ), A * x + B * y + C * z + D = 0 :=
by
  sorry

end intersection_points_in_plane_l422_422198


namespace common_ratio_of_geometric_seq_l422_422676

theorem common_ratio_of_geometric_seq (a b c d : ℤ) (h1 : a = 25)
    (h2 : b = -50) (h3 : c = 100) (h4 : d = -200)
    (h_geo_1 : b = a * -2)
    (h_geo_2 : c = b * -2)
    (h_geo_3 : d = c * -2) : 
    let r := (-2 : ℤ) in r = -2 := 
by 
  sorry

end common_ratio_of_geometric_seq_l422_422676


namespace problem_21_sum_correct_l422_422489

theorem problem_21_sum_correct (A B C D E : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
    (h_eq : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 21 :=
sorry

end problem_21_sum_correct_l422_422489


namespace min_value_of_sum_of_squares_l422_422769

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x + 2 * y + z = 1) : 
    x^2 + y^2 + z^2 ≥ (1 / 6) := 
  sorry

noncomputable def min_val_xy2z (x y z : ℝ) (h : x + 2 * y + z = 1) : ℝ :=
  if h_sq : x^2 + y^2 + z^2 = 1 / 6 then (x^2 + y^2 + z^2) else if x = 1 / 6 ∧ z = 1 / 6 ∧ y = 1 / 3 then 1 / 6 else (1 / 6)

example (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + y^2 + z^2 = min_val_xy2z x y z h :=
  sorry

end min_value_of_sum_of_squares_l422_422769


namespace gingerbread_to_bagels_l422_422115

theorem gingerbread_to_bagels (gingerbread drying_rings bagels : ℕ) 
  (h1 : gingerbread = 1 → drying_rings = 6) 
  (h2 : drying_rings = 9 → bagels = 4) 
  (h3 : gingerbread = 3) : bagels = 8 :=
by
  sorry

end gingerbread_to_bagels_l422_422115


namespace constant_term_expansion_l422_422593

theorem constant_term_expansion (x : ℝ) :
  let term := ∑ k in finset.range (8 + 1), (nat.choose 8 k) * (9 * x)^k * ((2 / (3 * x))^(8 - k))
  ∃ k, 2 * k = 8 ∧ term = 90720 := by
sor

end constant_term_expansion_l422_422593


namespace trig_eq_num_solutions_sum_digits_l422_422922

def trig_eq (x : ℝ) : ℝ :=
  24 * sin (2 * x) + 7 * cos (2 * x) - 36 * sin x - 48 * cos x + 35

def interval_start := 10 ^ (factorial 2014) * Real.pi
def interval_end := 10 ^ (factorial 2014 + 2018) * Real.pi

def N := 3 / 2 * (10 ^ (factorial 2014 + 2018) - 10 ^ (factorial 2014))

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem trig_eq_num_solutions_sum_digits :
  sum_digits (nat_abs (floor (N - (N / 10 ^ 2018)))) = 18162 :=
sorry

end trig_eq_num_solutions_sum_digits_l422_422922


namespace all_programming_languages_basic_statements_l422_422225

theorem all_programming_languages_basic_statements :
  ∀ (P : Type), P = "programming_language" → 
    (∃ S, S = ["input statement", "output statement", "assignment statement", "conditional statement", "loop statement"]) :=
by
  intros P hP
  use ["input statement", "output statement", "assignment statement", "conditional statement", "loop statement"]
  simp
  sorry

end all_programming_languages_basic_statements_l422_422225


namespace parabola_tangent_xsum_l422_422157

theorem parabola_tangent_xsum
  (p : ℝ) (hp : p > 0) 
  (X_A X_B X_M : ℝ) 
  (hxM_line : ∃ y, y = -2 * p ∧ y = -2 * p)
  (hxA_tangent : ∃ y, y = (X_A / p) * (X_A - X_M) - 2 * p)
  (hxB_tangent : ∃ y, y = (X_B / p) * (X_B - X_M) - 2 * p) :
  2 * X_M = X_A + X_B :=
by
  sorry

end parabola_tangent_xsum_l422_422157


namespace bird_counts_l422_422308

theorem bird_counts :
  ∀ (num_cages_1 num_cages_2 num_cages_empty parrot_per_cage parakeet_per_cage canary_per_cage cockatiel_per_cage lovebird_per_cage finch_per_cage total_cages : ℕ),
    num_cages_1 = 7 →
    num_cages_2 = 6 →
    num_cages_empty = 2 →
    parrot_per_cage = 3 →
    parakeet_per_cage = 5 →
    canary_per_cage = 4 →
    cockatiel_per_cage = 2 →
    lovebird_per_cage = 3 →
    finch_per_cage = 1 →
    total_cages = 15 →
    (num_cages_1 * parrot_per_cage = 21) ∧
    (num_cages_1 * parakeet_per_cage = 35) ∧
    (num_cages_1 * canary_per_cage = 28) ∧
    (num_cages_2 * cockatiel_per_cage = 12) ∧
    (num_cages_2 * lovebird_per_cage = 18) ∧
    (num_cages_2 * finch_per_cage = 6) :=
by
  intros
  sorry

end bird_counts_l422_422308


namespace expectation_2X_minus_1_variance_2X_minus_1_l422_422559

noncomputable def distribution_X : ℕ → ℝ
| 0 := 0.3
| 1 := 0.4
| 2 := 0.3
| _ := 0

theorem expectation_2X_minus_1 :
    (∀x, distribution_X x ≥ 0) → (∑ x, distribution_X x = 1) →
    (∑ x, x * distribution_X x = 1) →
    (∑ x, (2 * x - 1) * distribution_X x = 1) := 
by
  intros h1 h2 h3
  sorry

theorem variance_2X_minus_1 :
    (∀x, distribution_X x ≥ 0) → (∑ x, distribution_X x = 1) →
    (∑ x, x^2 * distribution_X x = 1.6) →
    (1.6 - (1)^2 = 0.6) → 
    4 * 0.6 = 2.4 := 
by
  intros h1 h2 h3 h4
  sorry

end expectation_2X_minus_1_variance_2X_minus_1_l422_422559


namespace range_of_eccentricity_l422_422785

theorem range_of_eccentricity (x y : ℝ) (C : x^2 / 3 + y^2 / 2 = 1) (x_1 y_1 : ℝ)
    (Hp : x_1^2 / 3 + y_1^2 / 2 = 1) (H : Real) (λ : ℝ) [λ ≥ 1] :
    let PH := |x_1 - 3|, HQ := λ * PH in
    let ex := (3 * (1 + λ) - x_1) / λ in
    let ey := y_1 in
    let e := sqrt(1 - 2 / (3 * λ^2)) in
    (e >= sqrt(1 - 2 / 3) / sqrt(3) ∧ e < 1) := sorry

end range_of_eccentricity_l422_422785


namespace value_of_a_plus_b_l422_422956

theorem value_of_a_plus_b (a b c : ℤ) 
    (h1 : a + b + c = 11)
    (h2 : a + b - c = 19)
    : a + b = 15 := 
by
    -- Mathematical details skipped
    sorry

end value_of_a_plus_b_l422_422956


namespace kai_marbles_over_200_l422_422503

theorem kai_marbles_over_200 (marbles_on_day : Nat → Nat)
  (h_initial : marbles_on_day 0 = 4)
  (h_growth : ∀ n, marbles_on_day (n + 1) = 3 * marbles_on_day n) :
  ∃ k, marbles_on_day k > 200 ∧ k = 4 := by
  sorry

end kai_marbles_over_200_l422_422503


namespace one_element_in_A_inter_B_range_m_l422_422077

theorem one_element_in_A_inter_B_range_m (m : ℝ) :
  let A := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = -x^2 + m * x - 1}
  let B := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = 3 - x ∧ 0 ≤ x ∧ x ≤ 3}
  (∃! p, p ∈ A ∧ p ∈ B) → (m = 3 ∨ m > 10 / 3) :=
by
  sorry

end one_element_in_A_inter_B_range_m_l422_422077


namespace length_of_AB_l422_422127

noncomputable def AB_CD_sum_240 (AB CD : ℝ) (h : ℝ) : Prop :=
  AB + CD = 240

noncomputable def ratio_of_areas (AB CD : ℝ) : Prop :=
  AB / CD = 5 / 3

theorem length_of_AB (AB CD : ℝ) (h : ℝ) (h_ratio : ratio_of_areas AB CD) (h_sum : AB_CD_sum_240 AB CD h) : AB = 150 :=
by
  unfold ratio_of_areas at h_ratio
  unfold AB_CD_sum_240 at h_sum
  sorry

end length_of_AB_l422_422127


namespace units_digit_R_12445_l422_422151

def units_digit (n : ℕ) : ℕ :=
  n % 10

noncomputable def R (n : ℕ) : ℚ :=
  let a := 3 + 2 * Real.sqrt 2
  let b := 3 - 2 * Real.sqrt 2
  (a^n + b^n) / 2

theorem units_digit_R_12445 :
  units_digit (R 12445).natAbs = 3 :=
by <|
  sorry

end units_digit_R_12445_l422_422151


namespace sqrt_four_four_summed_l422_422603

theorem sqrt_four_four_summed :
  sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := by
  sorry

end sqrt_four_four_summed_l422_422603


namespace who_did_not_win_l422_422371

-- defining the participants
variables (A B C D : Prop)

-- given conditions
def condition1 : Prop := A → B
def condition2 : Prop := B → C
def condition3 : Prop := ¬D → ¬C
def exactly_one_did_not_win : Prop := (¬A ∧ B ∧ C ∧ D) ∨ (A ∧ ¬B ∧ C ∧ D) ∨ (A ∧ B ∧ ¬C ∧ D) ∨ (A ∧ B ∧ C ∧ ¬D)
def all_statements_true : Prop := condition1 ∧ condition2 ∧ condition3

-- the theorem stating the one who did not win the prize
theorem who_did_not_win (h1 : all_statements_true) (h2 : exactly_one_did_not_win) : ¬A := 
by  
  sorry

end who_did_not_win_l422_422371


namespace parrots_count_l422_422622

theorem parrots_count (p r : ℕ) : 2 * p + 4 * r = 26 → p + r = 10 → p = 7 := by
  intros h1 h2
  sorry

end parrots_count_l422_422622


namespace concurrency_of_lines_l422_422637

theorem concurrency_of_lines
  (k₁ k₂ k : Circle)
  (O₁ O₂ O : Point)
  (C A B : Point)
  (ℓ : Line)
  (h1 : k₁.center = O₁)
  (h2 : k₂.center = O₂)
  (h3 : k.center = O)
  (h4 : tangent k₁ k₂ C)
  (h5 : tangent k₁ k C)
  (h6 : common_tangent ℓ k₁ k₂ C)
  (h7 : diameter k A B)
  (h8 : perpendicular ℓ (line_through A B))
  (h9 : same_side ℓ O₁ A)
  : concurrent (line_through A O₂) (line_through B O₁) ℓ :=
sorry

end concurrency_of_lines_l422_422637


namespace timothy_per_acre_l422_422281

-- Define the conditions
def mixturePercentage : ℝ := 0.05
def totalMixtureFor15Acres : ℝ := 600
def acresPlanted : ℝ := 15
def poundsPerAcre : ℝ := totalMixtureFor15Acres / acresPlanted

-- State the theorem to prove the number of pounds of timothy needed per acre
theorem timothy_per_acre : mixturePercentage * poundsPerAcre = 2 := by
  sorry

end timothy_per_acre_l422_422281


namespace proof_equivalent_problem_l422_422814

theorem proof_equivalent_problem (x : ℝ)
  (h : x - sqrt (x^2 - 4) + 1 / (x + sqrt (x^2 - 4)) = 10) :
  x^2 - sqrt (x^4 - 4) + 1 / (x^2 - sqrt (x^4 - 4)) = 225 / 16 :=
by
  sorry

end proof_equivalent_problem_l422_422814


namespace snowboard_price_after_discounts_l422_422901

theorem snowboard_price_after_discounts
  (original_price : ℝ) (friday_discount_rate : ℝ) (monday_discount_rate : ℝ) 
  (sales_tax_rate : ℝ) (price_after_all_adjustments : ℝ) :
  original_price = 200 →
  friday_discount_rate = 0.40 →
  monday_discount_rate = 0.20 →
  sales_tax_rate = 0.05 →
  price_after_all_adjustments = 100.80 :=
by
  intros
  sorry

end snowboard_price_after_discounts_l422_422901


namespace fractional_part_of_cake_eaten_l422_422248

theorem fractional_part_of_cake_eaten :
  let total_eaten := 1 / 3 + 1 / 3^2 + 1 / 3^3 + 1 / 3^4
  in total_eaten = 40 / 81 :=
by
  sorry

end fractional_part_of_cake_eaten_l422_422248


namespace quadratic_non_real_roots_b_range_l422_422098

theorem quadratic_non_real_roots_b_range (b : ℝ) :
  let f := λ x : ℝ, x^2 + b * x + 16
  -- Discriminant condition for non-real roots:
  -- b^2 - 64 < 0
  (b^2 < 64) -> b ∈ Set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_b_range_l422_422098


namespace tangent_line_perpendicular_l422_422823

theorem tangent_line_perpendicular (m : ℝ) :
  (∀ x : ℝ, y = 2 * x^2) →
  (∀ x : ℝ, (4 * x - y + m = 0) ∧ (x + 4 * y - 8 = 0) → 
  (16 + 8 * m = 0)) →
  m = -2 :=
by
  sorry

end tangent_line_perpendicular_l422_422823


namespace find_matching_sum_l422_422345

-- Define the functions f and g according to the problem conditions
def f (i j : ℕ) : ℕ := 15 * (i - 1) + j
def g (i j : ℕ) : ℕ := 12 * (j - 1) + i

-- Define the condition of matching numbers and summing them within board constraints
def matching_sum : ℕ :=
  (Finset.filter
    (λ (pair : ℕ × ℕ), f pair.fst pair.snd = g pair.fst pair.snd)
    (Finset.product (Finset.range 13) (Finset.range 16)))
  .val
  .sum
  (λ (pair : ℕ × ℕ), f pair.fst pair.snd)

theorem find_matching_sum : matching_sum = 913 := 
sorry

end find_matching_sum_l422_422345


namespace articles_for_z_men_l422_422819

-- The necessary conditions and given values
def articles_produced (men hours days : ℕ) := men * hours * days

theorem articles_for_z_men (x z : ℕ) (H : articles_produced x x x = x^2) :
  articles_produced z z z = z^3 / x := by
  sorry

end articles_for_z_men_l422_422819


namespace train_crosses_bridge_in_30_seconds_l422_422321

def length_of_train : ℝ := 120 -- meters
def speed_of_train_kmh : ℝ := 45 -- km/hr
def length_of_bridge : ℝ := 255 -- meters

def speed_of_train_ms : ℝ :=
  (speed_of_train_kmh * 1000) / 3600

def total_distance : ℝ :=
  length_of_train + length_of_bridge

def time_to_cross_bridge : ℝ :=
  total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_30_seconds :
  time_to_cross_bridge = 30 := by
  sorry

end train_crosses_bridge_in_30_seconds_l422_422321


namespace quadratic_real_roots_range_l422_422803

theorem quadratic_real_roots_range (m : ℝ) : 
(m - 1) * x^2 - 2 * x + 1 = 0 → (m ≤ 2 ∧ m ≠ 1) :=
begin
  sorry
end

end quadratic_real_roots_range_l422_422803


namespace longer_side_length_l422_422112

-- Definitions and conditions
def height (x : ℝ) := 3 * x
def width (x : ℝ) := x
def border := 3
def panes := 12
def ratio := 3
def perimeter := 108
def total_length (x : ℝ) := 2 * (4 * (width x) + 5 * border) + 2 * (3 * (height x) + 4 * border)

theorem longer_side_length : ∃ x : ℝ, total_length x = perimeter → 3 * x + 4 * border > 4 * x + 5 * border ∧ 3 * x + 4 * border = 30.684 :=
by
  sorry

end longer_side_length_l422_422112


namespace tan_beta_minus_2alpha_l422_422058

theorem tan_beta_minus_2alpha
  (α β : ℝ)
  (h1 : Real.tan α = 1/2)
  (h2 : Real.tan (α - β) = -1/3) :
  Real.tan (β - 2 * α) = -1/7 := 
sorry

end tan_beta_minus_2alpha_l422_422058


namespace failed_both_l422_422845

-- Defining the conditions based on the problem statement
def failed_hindi : ℝ := 0.34
def failed_english : ℝ := 0.44
def passed_both : ℝ := 0.44

-- Defining a proposition to represent the problem and its solution
theorem failed_both (x : ℝ) (h1 : x = failed_hindi + failed_english - (1 - passed_both)) : 
  x = 0.22 :=
by
  sorry

end failed_both_l422_422845


namespace median_equation_altitude_equation_area_triangle_l422_422114

variable (A B C : Point)
variable (x y : ℝ)

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def line_through (P Q : Point) : ℝ × ℝ × ℝ :=
  let k := (Q.y - P.y) / (Q.x - P.x)
  let b := P.y - k * P.x
  (-k, 1, -b)

theorem median_equation :
  let A : Point := ⟨-2, 1⟩
  let B : Point := ⟨2, 1⟩
  let C : Point := ⟨4, -3⟩
  let D := midpoint A C
  line_through B D = (2, -1, 3) :=
sorry

theorem altitude_equation :
  let A : Point := ⟨-2, 1⟩
  let B : Point := ⟨2, 1⟩
  let C : Point := ⟨4, -3⟩
  let k_BC := (C.y - B.y) / (C.x - B.x)
  let k_AH := -1 / k_BC
  let line := line_through A ⟨A.x + 1, A.y + k_AH⟩
  line = (1, -2, 4) :=
sorry

theorem area_triangle :
  let A : Point := ⟨-2, 1⟩
  let B : Point := ⟨2, 1⟩
  let C : Point := ⟨4, -3⟩
  let BC_length := Math.sqrt ((C.x - B.x)^2 + (C.y - B.y)^2)
  let line_BC := (C.y - B.y) / (C.x - B.x)
  let d := ((C.y - B.y) * A.x + (B.x - C.x) * A.y) / Math.sqrt((C.y - B.y)^2 + (C.x - B.x)^2)
  let area := 1 / 2 * BC_length * d
  area = 8 :=
sorry

end median_equation_altitude_equation_area_triangle_l422_422114


namespace right_triangle_point_probability_l422_422047

noncomputable def calc_probability_of_point_in_triangle (radius: ℝ) : ℝ :=
  let s_circle := π * radius^2
  let base := 1
  let height := 1
  let c := 120
  let s_triangle := 3 * (1 / 2) * base * height * real.sin (real.to_radians c)
  s_triangle / s_circle

theorem right_triangle_point_probability : calc_probability_of_point_in_triangle 1 = 3 * real.sqrt 3 / (4 * π) :=
by
  sorry

end right_triangle_point_probability_l422_422047


namespace largest_five_digit_integer_prod_1440_l422_422989

theorem largest_five_digit_integer_prod_1440 :
  ∃ (n : ℕ), (digits_product n = 1440) ∧ (is_five_digit_number n) ∧ (∀ m, (digits_product m = 1440) ∧ (is_five_digit_number m) → n ≥ m) := 
sorry

-- Definitions
def digits_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

def is_five_digit_number (n : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000

end largest_five_digit_integer_prod_1440_l422_422989


namespace jimmy_earnings_l422_422131

theorem jimmy_earnings : 
  let price15 := 15
  let price20 := 20
  let discount := 5
  let sale_price15 := price15 - discount
  let sale_price20 := price20 - discount
  let num_low_worth := 4
  let num_high_worth := 1
  num_low_worth * sale_price15 + num_high_worth * sale_price20 = 55 :=
by
  sorry

end jimmy_earnings_l422_422131


namespace train_length_proof_l422_422693

-- Define the conditions given in the problem
def speed_km_hr : ℝ := 108
def time_sec : ℝ := 1.4998800095992322
def speed_m_s : ℝ := speed_km_hr * (1000 / 3600)
def length_of_train : ℝ := speed_m_s * time_sec

-- The theorem to prove
theorem train_length_proof :
  length_of_train = 44.996400287976966 :=
by
  -- Proof omitted
  sorry

end train_length_proof_l422_422693


namespace graph_of_abs_f_is_D_l422_422589

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then sqrt(4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

theorem graph_of_abs_f_is_D :
  ∀ x : ℝ, |f x| = 
  (if -3 ≤ x ∧ x < 0 then x + 2
   else if 0 ≤ x ∧ x ≤ 2 then abs (sqrt(4 - (x - 2)^2) - 2)
   else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
   else 0) := sorry

end graph_of_abs_f_is_D_l422_422589


namespace additional_machines_needed_l422_422261

theorem additional_machines_needed
  (machines : ℕ)
  (days : ℕ)
  (one_fourth_less_days : ℕ)
  (machine_days_total : ℕ)
  (machines_needed : ℕ)
  (additional_machines : ℕ) 
  (h1 : machines = 15) 
  (h2 : days = 36)
  (h3 : one_fourth_less_days = 27)
  (h4 : machine_days_total = machines * days)
  (h5 : machines_needed = machine_days_total / one_fourth_less_days) :
  additional_machines = machines_needed - machines → additional_machines = 5 :=
by
  admit -- sorry

end additional_machines_needed_l422_422261


namespace ratio_of_triangle_areas_eq_one_l422_422636

/-- Given:
1. Triangle ABC is acute with vertices A, B, C.
2. Point M is on side AB.
3. Point D is inside triangle ABC.
4. Circles ωA and ωB are the circumcircles of triangles AMD and BMD, respectively.
5. Side AC intersects ωA again at point P (P ≠ A).
6. Side BC intersects ωB again at point Q (Q ≠ B).
7. Ray PD intersects ωB again at point R (R ≠ D).
8. Ray QD intersects ωA again at point S (S ≠ D).

Prove: The ratio of the areas of triangles ACR and BCS is 1. -/
theorem ratio_of_triangle_areas_eq_one 
  {A B C M D P Q R S : Point}
  (hABC : acute_triangle A B C)
  (hM : M ∈ segment A B)
  (hD : in_interior_triangle A B C D)
  (hωA : circle A M D P)
  (hωB : circle B M D Q)
  (hAP : P ∈ intersection (line_segment A C) (circle A M D))
  (hBQ : Q ∈ intersection (line_segment B C) (circle B M D))
  (hPD : R ∈ ray P D)
  (hQD : S ∈ ray Q D) :
  area (triangle A C R) / area (triangle B C S) = 1 :=
sorry

end ratio_of_triangle_areas_eq_one_l422_422636


namespace multiplication_addition_l422_422987

theorem multiplication_addition :
  23 * 37 + 16 = 867 :=
by
  sorry

end multiplication_addition_l422_422987


namespace common_ratio_of_geometric_seq_l422_422678

theorem common_ratio_of_geometric_seq (a b c d : ℤ) (h1 : a = 25)
    (h2 : b = -50) (h3 : c = 100) (h4 : d = -200)
    (h_geo_1 : b = a * -2)
    (h_geo_2 : c = b * -2)
    (h_geo_3 : d = c * -2) : 
    let r := (-2 : ℤ) in r = -2 := 
by 
  sorry

end common_ratio_of_geometric_seq_l422_422678


namespace LaShawns_collection_has_one_and_a_half_times_more_than_Kymbrea_l422_422871

theorem LaShawns_collection_has_one_and_a_half_times_more_than_Kymbrea's (x : ℕ) :
  25 + 5 * x = 1.5 * (40 + 3 * x) → x = 70 :=
by
  sorry

end LaShawns_collection_has_one_and_a_half_times_more_than_Kymbrea_l422_422871


namespace max_marks_l422_422238

theorem max_marks (M : ℝ) (h : 0.92 * M = 460) : M = 500 :=
by
  sorry

end max_marks_l422_422238


namespace f_2012_l422_422782

noncomputable def problem_statement (f : ℤ → ℤ) : Prop :=
  (∀ x, f(x + 1) = -f(-x + 1)) ∧ (∀ x, f(x - 1) = f(-x - 1)) ∧ (f(0) = 2)

theorem f_2012 (f : ℤ → ℤ) (h: problem_statement f) : f 2012 = -2 :=
by 
  sorry

end f_2012_l422_422782


namespace exists_non_decreasing_subsequences_l422_422177

theorem exists_non_decreasing_subsequences {a b c : ℕ → ℕ} : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end exists_non_decreasing_subsequences_l422_422177


namespace reward_scheme_l422_422655

theorem reward_scheme
    (h1 : ∀ (a b : ℝ), y = a * log 4 80000 + b → y = 10000)
    (h2 : ∀ (a b : ℝ), y = a * log 4 640000 + b → y = 40000) :
    ∃ x : ℝ, (y = 80000 → x = 1024000)
by
  sorry

end reward_scheme_l422_422655


namespace sum_binomial_mod_prime_l422_422778

theorem sum_binomial_mod_prime (p n : ℕ) (hp : Prime p) (hn : n ≥ p) : 
  ∑ k in Finset.range (n / p + 1), (-1 : ℤ) ^ k * (Nat.choose n (p * k) : ℤ) ≡ 0 [MOD p] :=
sorry

end sum_binomial_mod_prime_l422_422778


namespace find_f_2023_l422_422663

noncomputable def f : ℤ → ℤ := sorry

theorem find_f_2023 (h1 : ∀ x : ℤ, f (x+2) + f x = 3) (h2 : f 1 = 0) : f 2023 = 3 := sorry

end find_f_2023_l422_422663


namespace pentagon_area_ratio_l422_422878

-- Define the convex pentagon with given conditions
variable {F G H I J : Type}
variable [HasScalar ℝ (Point ℝ)]

-- Points and configuration properties
variable (FG_par_IJ : ∀ (A B C D : Point ℝ), A = F ∧ B = G ∧ C = I ∧ D = J → ∥A - C∥ = ∥B - D∥)
variable (GH_par_FI : ∀ (A B C D : Point ℝ), A = G ∧ B = H ∧ C = F ∧ D = I → ∥A - C∥ = ∥B - D∥)
variable (GI_par_HJ : ∀ (A B C D : Point ℝ), A = G ∧ B = I ∧ C = H ∧ D = J → ∥A - C∥ = ∥B - D∥)
variable (angle_FGH : angle F G H = 100)
variable (length_FG : ∥F - G∥ = 4)
variable (length_GH : ∥G - H∥ = 7)
variable (length_HJ : ∥H - J∥ = 21)

-- Translate the mathematical ratio and the required proof
theorem pentagon_area_ratio (x y : ℕ) (h_rel_prime : nat.coprime x y) (h_ratio: ((x:ℝ) / y = 200 / 491.04)) : 
  x + y = 346 := 
sorry

end pentagon_area_ratio_l422_422878


namespace find_eccentricity_l422_422051

def ellipse_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity_of_ellipse (a c : ℝ) : ℝ :=
  c / a

theorem find_eccentricity (a b : ℝ) (c := a * (sqrt 2) / 3) (h : 9 * c^2 = 2 * a^2) : 
  ∃ e : ℝ, e = eccentricity_of_ellipse a c ∧ e = sqrt 2 / 3 :=
by
  sorry

end find_eccentricity_l422_422051


namespace find_t_l422_422169

-- Define my hours worked and hourly rate
def my_hours_worked (t : ℝ) : ℝ := t - 4
def my_hourly_rate (t : ℝ) : ℝ := 3t - 2

-- Define Bella's hours worked and hourly rate
def bella_hours_worked (t : ℝ) : ℝ := 4t - 10
def bella_hourly_rate (t : ℝ) : ℝ := 2t - 9

-- Define my earnings and Bella's earnings
def my_earnings (t : ℝ) : ℝ := my_hours_worked t * my_hourly_rate t
def bella_earnings (t : ℝ) : ℝ := bella_hours_worked t * bella_hourly_rate t

-- The objective is to prove t = 5 given the condition that my earnings = Bella's earnings
theorem find_t (t : ℝ): 
  my_earnings t = bella_earnings t → t = 5 :=
by
  intro h
  -- The proof goes here
  sorry

end find_t_l422_422169


namespace opposite_of_8_is_neg8_l422_422571

theorem opposite_of_8_is_neg8 : ∃ y : ℤ, 8 + y = 0 ∧ y = -8 := by
  use -8
  split
  ·
    sorry

  ·
    rfl

end opposite_of_8_is_neg8_l422_422571


namespace midpoint_quadrilateral_inequality_l422_422520

theorem midpoint_quadrilateral_inequality 
  (A B C D E F G H : ℝ) 
  (S_ABCD : ℝ)
  (midpoints_A : E = (A + B) / 2)
  (midpoints_B : F = (B + C) / 2)
  (midpoints_C : G = (C + D) / 2)
  (midpoints_D : H = (D + A) / 2)
  (EG : ℝ)
  (HF : ℝ) :
  S_ABCD ≤ EG * HF ∧ EG * HF ≤ (B + D) * (A + C) / 4 := by
  sorry

end midpoint_quadrilateral_inequality_l422_422520


namespace number_of_subsets_of_A_l422_422434

-- Define the conditions
def A : Set ℝ := { x | (x^2 - 4) / Real.sqrt x = 0 }

-- State the theorem
theorem number_of_subsets_of_A : set.finite A → set.card A = 1 → 2^1 = 2 := by
  intros
  sorry

end number_of_subsets_of_A_l422_422434


namespace semicircle_circumference_eq_51_42_l422_422574

/-- Given a rectangle with length 24 cm and breadth 16 cm, if the perimeter of a square is equal
to the perimeter of this rectangle, then the circumference of a semicircle with diameter equal
to the side of this square is 51.42 cm (rounded to two decimal places). -/
theorem semicircle_circumference_eq_51_42 :
  let length := 24
  let breadth := 16
  let perimeter_rectangle := 2 * (length + breadth)
  ∃ π : ℝ, (π = Real.pi) →
  let side_square := perimeter_rectangle / 4
  let diameter := side_square
  let circumference_semicircle := (π * diameter) / 2 + diameter
  Real.floor (circumference_semicircle * 100) / 100 = 51.42 :=
by
  sorry

end semicircle_circumference_eq_51_42_l422_422574


namespace ben_eggs_left_l422_422332

def initial_eggs : ℕ := 50
def day1_morning : ℕ := 5
def day1_afternoon : ℕ := 4
def day2_morning : ℕ := 8
def day2_evening : ℕ := 3
def day3_afternoon : ℕ := 6
def day3_night : ℕ := 2

theorem ben_eggs_left : initial_eggs - (day1_morning + day1_afternoon + day2_morning + day2_evening + day3_afternoon + day3_night) = 22 := 
by
  sorry

end ben_eggs_left_l422_422332


namespace matrix_det_5_7_2_3_l422_422336

theorem matrix_det_5_7_2_3 : det (matrix.of (vector 5 [7]) (vector 2 [3])) = 1 :=
by
  sorry

end matrix_det_5_7_2_3_l422_422336


namespace odd_function_values_decreasing_function_inequality_solution_l422_422068

noncomputable def f (x : ℝ) : ℝ := (-2^x + 1) / (2^(x+1) + 2)

theorem odd_function_values (a b : ℝ) (h_a : a = 1) (h_b : b = 2) :
  f(x) = (-2^x + a) / (2^(x+1) + b) →
  (∀ x : ℝ, f(-x) = -f(x)) :=
by 
  sorry

theorem decreasing_function (x1 x2 : ℝ) (h : x1 < x2) :
  f(x1) > f(x2) :=
by
  sorry

theorem inequality_solution (x : ℝ) :
  f(x) > -1/6 ↔ x < 1 :=
by
  sorry

end odd_function_values_decreasing_function_inequality_solution_l422_422068


namespace sufficient_not_necessary_l422_422640

theorem sufficient_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by
  sorry

end sufficient_not_necessary_l422_422640


namespace inequality_proof_l422_422039

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : -b > 0) (h3 : a > -b) (h4 : c < 0) : 
  a * (1 - c) > b * (c - 1) :=
sorry

end inequality_proof_l422_422039


namespace value_of_P_value_of_Q_l422_422096

/-- Define conditions for P and Q based on the problem description --/
def simplify_numerator : ℕ := 8 * (∑ k in finset.range 2000, k^3)
def simplify_denominator : ℕ := ∑ k in finset.range 2000, k^3

theorem value_of_P :
  (simplify_numerator / simplify_denominator: ℝ)^(1/3) = 2 := sorry

theorem value_of_Q (P : ℕ) (hP : P = 2) :
  ∃ Q : ℕ, ∀ x : ℕ, (x - P) * (x - 2 * Q) - 1 = 0 → Q = 1 :=
sorry

end value_of_P_value_of_Q_l422_422096


namespace sum_2015_l422_422805

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≠ 0 → a n * a (n + 1) = (-1 : ℤ) ^ n

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range(n + 1), a i

theorem sum_2015 (a : ℕ → ℤ) (h : sequence a) : S a 2015 = -1 :=
  sorry

end sum_2015_l422_422805


namespace smallest_positive_angle_terminal_side_l422_422595

/-- 
  Given an angle θ, which is -2015° and the fact that a full rotation is 360°, 
  prove that the smallest positive angle that has the same terminal side 
  as θ is 145°.
--/
theorem smallest_positive_angle_terminal_side (θ : ℝ) (hθ : θ = -2015) : 
  ∃ k : ℤ, 0 < θ + k * 360 ∧ θ + k * 360 = 145 :=
by 
  have h : θ + 6 * 360 = 145,
  { 
    calc
      θ + 6 * 360 = -2015 + 6 * 360 : by rw [hθ]
              ... = -2015 + 2160     : by norm_num
              ... = 145            : by norm_num },
  use 6,
  split,
  { norm_num, },
  { exact h, }

end smallest_positive_angle_terminal_side_l422_422595


namespace raul_money_left_l422_422913

theorem raul_money_left (initial_money : ℕ) (cost_per_comic : ℕ) (number_of_comics : ℕ) (money_left : ℕ)
  (h1 : initial_money = 87)
  (h2 : cost_per_comic = 4)
  (h3 : number_of_comics = 8)
  (h4 : money_left = initial_money - (number_of_comics * cost_per_comic)) :
  money_left = 55 :=
by 
  rw [h1, h2, h3] at h4
  exact h4

end raul_money_left_l422_422913


namespace largest_good_number_is_99_number_of_absolute_good_numbers_is_39_l422_422824

-- Definition of good number
def is_good_number (N a b : ℕ) : Prop :=
  10 ≤ N ∧ N ≤ 99 ∧ N = a * b + a + b

-- Definition of absolute good number
def is_absolute_good_number (N a b : ℕ) : Prop :=
  is_good_number N a b ∧ (ab_div_sum_eq_3 a b ∨ ab_div_sum_eq_4 a b)

def ab_div_sum_eq_3 (a b : ℕ) : Prop :=
  a * b = 3 * (a + b)

def ab_div_sum_eq_4 (a b : ℕ) : Prop :=
  a * b = 4 * (a + b)

-- The largest good number is 99
theorem largest_good_number_is_99 : ∃ N, (∃ a b, is_good_number N a b) ∧ N = 99 := 
  sorry

-- The number of absolute good numbers is 39
theorem number_of_absolute_good_numbers_is_39 : ∃ n, (n = 39 ∧ 
  ∃ N a b, is_absolute_good_number N a b) := 
  sorry

end largest_good_number_is_99_number_of_absolute_good_numbers_is_39_l422_422824


namespace find_x_squared_inversely_l422_422186

theorem find_x_squared_inversely 
  (y : ℝ) (x : ℝ) (k : ℝ) 
  (h1 : x^2 * y^3 = k) 
  (h2 : x = 10) 
  (h3 : y = 2) :
  let k := (10^2) * (2^3),
      x := sqrt (k / (4^3))
  in x^2 = 12.5 :=
by
  sorry

end find_x_squared_inversely_l422_422186


namespace circumcircle_radius_is_two_l422_422865

-- Definitions for the problem setup
variables {A B C L M N : Type}
variables [triangle ABC]

-- Conditions given in the problem
def angle_bisectors (A B C L M N : Type) : Prop := angle_bisector A L ∧ angle_bisector B M ∧ angle_bisector C N
def angle_condition (A N M L C : Type) : Prop := ∠ANM = ∠ALC

-- Given sides of triangle LMN
variables {side1 side2 : ℝ}
def given_sides_lmn : Prop := side1 = 3 ∧ side2 = 4

-- Main theorem statement: radius of the circumcircle of triangle LMN
theorem circumcircle_radius_is_two (A B C L M N : Type) (side1 side2 : ℝ) 
  (h1 : angle_bisectors A B C L M N) (h2 : angle_condition A N M L C) (h3 : given_sides_lmn) : 
  ∃ (R : ℝ), R = 2 :=
sorry

end circumcircle_radius_is_two_l422_422865


namespace correct_operation_l422_422245

theorem correct_operation :
  (∀ (a : ℤ), 2 * a - a ≠ 1) ∧
  (∀ (a : ℤ), (a^2)^4 ≠ a^6) ∧
  (∀ (a b : ℤ), (a * b)^2 ≠ a * b^2) ∧
  (∀ (a : ℤ), a^3 * a^2 = a^5) :=
by
  sorry

end correct_operation_l422_422245


namespace equilateral_triangle_bound_l422_422041

theorem equilateral_triangle_bound (n k : ℕ) (h_n_gt_3 : n > 3) 
  (h_k_triangles : ∃ T : Finset (Finset (ℝ × ℝ)), T.card = k ∧ ∀ t ∈ T, 
  ∃ a b c : (ℝ × ℝ), t = {a, b, c} ∧ dist a b = 1 ∧ dist b c = 1 ∧ dist c a = 1) :
  k < (2 * n) / 3 :=
by
  sorry

end equilateral_triangle_bound_l422_422041


namespace positive_even_multiples_of_3_perfect_squares_l422_422000

theorem positive_even_multiples_of_3_perfect_squares (n : ℕ) :
  let count := cardinality {x : ℕ | (0 < x) ∧ (x % 3 = 0) ∧ (x % 2 = 0) ∧ (x < 1500) ∧ (∃ k : ℕ, x = 36 * k^2)} in
  count = 6 :=
by
  sorry

end positive_even_multiples_of_3_perfect_squares_l422_422000


namespace servings_have_correct_fat_amount_l422_422534

noncomputable def grams_of_fat_per_serving : ℕ :=
let cream := 3 / 6 in
let cheese := 2 / 6 in
let butter := 1 / 6 in
let fat_cream := cream * 88 in
let fat_cheese := cheese * 110 in
let fat_butter := butter * 184 in
let total_fat := fat_cream + fat_cheese + fat_butter in
total_fat / 4

theorem servings_have_correct_fat_amount : 
    grams_of_fat_per_serving = 27.84 := 
by
  -- Since from the condition:
  -- cream ratio: 3/6,
  -- cheese ratio: 2/6,
  -- butter ratio: 1/6,
  -- per cup fat content: cream (88 grams), cheese (110 grams), butter (184 grams).
  -- Total fat computation per ingredient:
  -- cream_fat = (1/2) * 88 = 44,
  -- cheese_fat = (1/3) * 110 = 36.67,
  -- butter_fat = (1/6) * 184 = 30.67,
  -- Total fat = 111.34.
  -- Since the recipe serves 4, 
  -- per serving fat amount = 111.34 / 4 = 27.84 (approximately).
  sorry

end servings_have_correct_fat_amount_l422_422534


namespace least_number_subtracted_l422_422242

theorem least_number_subtracted (n k : ℕ) (h₁ : n = 123457) (h₂ : k = 79) : ∃ r, n % k = r ∧ r = 33 :=
by
  sorry

end least_number_subtracted_l422_422242


namespace train_crossing_time_l422_422323

-- Define the speed of the train in km/hr
def speed_kmh : ℕ := 60

-- Define the length of the train in meters
def length_m : ℕ := 100

-- Convert the speed from km/hr to m/s
def speed_ms : ℚ := (60 * 1000) / 3600

-- Define the expected time in seconds
def expected_time : ℚ := 6

-- The theorem we want to prove: The time it takes for the train to cross the pole is 6 seconds
theorem train_crossing_time (s_kmh : ℕ) (l_m : ℕ) : s_kmh = speed_kmh → l_m = length_m → (l_m / speed_ms) = expected_time :=
by
  intros hskh hlm
  rw [←hskh, ←hlm]
  sorry

end train_crossing_time_l422_422323


namespace angle_4_measure_l422_422525

theorem angle_4_measure (p q : Line) (h_parallel : p.parallel q)
  (m1 m2 : ℝ) (h_angle1 : m1 = (1 / 5) * m2) :
  ∀ (m4 : ℝ), m4 = m1 → ∃ x : ℝ, (m2 + m4 = 180) ∧ (x = 30) :=
by
  intro m4 h_m4
  use 30
  sorry

end angle_4_measure_l422_422525


namespace find_r1_plus_s1_l422_422136

theorem find_r1_plus_s1 :
  ∃ r s : Polynomial ℤ,
    r.Monic ∧ s.Monic ∧
    degree r > 0 ∧ degree s > 0 ∧
    (X ^ 8 - 50 * X ^ 4 + 1 = r * s) ∧
    (r.eval 1 + s.eval 1 = 4) :=
sorry

end find_r1_plus_s1_l422_422136


namespace arc_triangle_area_ratio_l422_422652

theorem arc_triangle_area_ratio 
  {r : ℝ} (h_circle_radius : r = 3)
  (h_arcs_into_triangle : ∃ side_length : ℝ, side_length = 2 * r * π / 3)
  (h_square_side : ∃ side_length : ℝ, side_length = 2 * r) :
  ∃ ratio : ℝ, ratio = (√3 * π / 9) :=
by
  -- Proof omitted
  sorry

end arc_triangle_area_ratio_l422_422652


namespace clay_blocks_needed_l422_422325

noncomputable def volume_cylinder (r h : ℝ) := real.pi * r^2 * h

def volume_block (l w h : ℝ) := l * w * h

theorem clay_blocks_needed :
  ∀ (r h : ℝ) (l w : ℝ),
    -- Given the dimensions of the cylinder
    r = 3 ∧ h = 10 →
    -- And the dimensions of the block
    l = 8 ∧ w = 3 ∧ 2 = 2 →
    -- Prove that 6 whole blocks are needed
    let num_blocks := (volume_cylinder r h) / (volume_block l w 2) in
    nat.ceil num_blocks = 6 :=
begin
  sorry
end

end clay_blocks_needed_l422_422325


namespace max_sum_expr_l422_422394

-- Define the sequence a_k with the specified conditions
noncomputable def a_k (k : ℕ) : ℝ := sorry    -- Placeholder for defining the sequence a_k

-- Conditions for the sequence
axiom a_k_bounds : ∀ k, 1 ≤ k ∧ k ≤ 2020 → 0 ≤ a_k k ∧ a_k k ≤ 1
axiom a_k_2021 : a_k 2021 = a_k 1
axiom a_k_2022 : a_k 2022 = a_k 2

-- Define the summation
def sum_expr (n : ℕ) : ℝ :=
  ∑ k in finset.range n,
    (a_k (k + 1) - a_k (k + 2) * a_k (k + 3))

-- The main theorem stating the maximum sum is 673
theorem max_sum_expr : sum_expr 2020 = 673 := sorry

end max_sum_expr_l422_422394


namespace find_mn_l422_422783

theorem find_mn
  (AB BC : ℝ) -- Lengths of AB and BC
  (m n : ℝ)   -- Coefficients of the quadratic equation
  (h_perimeter : 2 * (AB + BC) = 12)
  (h_area : AB * BC = 5)
  (h_roots_sum : AB + BC = -m)
  (h_roots_product : AB * BC = n) :
  m * n = -30 :=
by
  sorry

end find_mn_l422_422783


namespace geometric_sequence_common_ratio_l422_422669

theorem geometric_sequence_common_ratio 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 25) 
  (h2 : a2 = -50) 
  (h3 : a3 = 100) 
  (h4 : a4 = -200)
  (h_geometric : a2 / a1 = a3 / a2 ∧ a3 / a2 = a4 / a3) : 
  a2 / a1 = -2 :=
by 
  have r1 : a2 / a1 = -2, sorry
  -- additional steps to complete proof here
  exact r1

end geometric_sequence_common_ratio_l422_422669


namespace fractional_part_of_cake_eaten_l422_422249

theorem fractional_part_of_cake_eaten :
  let total_eaten := 1 / 3 + 1 / 3^2 + 1 / 3^3 + 1 / 3^4
  in total_eaten = 40 / 81 :=
by
  sorry

end fractional_part_of_cake_eaten_l422_422249


namespace length_in_scientific_notation_l422_422084

theorem length_in_scientific_notation : (161000 : ℝ) = 1.61 * 10^5 := 
by 
  -- Placeholder proof
  sorry

end length_in_scientific_notation_l422_422084


namespace triangle_area_arithmetic_progression_l422_422324

open Classical

variables (a d : ℝ) (d_pos : d > 0) (a_pos : a > 0)

theorem triangle_area_arithmetic_progression :
  let b := 2 * a - d in
  let h := 2 * a + d in
  (1 / 2) * b * h = 2 * a^2 - (d^2) / 2 :=
by
  let b := 2 * a - d
  let h := 2 * a + d
  sorry

end triangle_area_arithmetic_progression_l422_422324


namespace length_segment_DM_l422_422852

-- Define the context and given data
variables (A B C D M K : Type)
variables [metric_space A B C D M K]
variables (angleCAB angleCBA angleKDA angleKAD : ℝ)
variables (AC AB AD DM : ℝ)
variables (midpoint : A → B → C)

-- Conditions
def conditions : Prop :=
  (angleCAB = 2 * angleCBA) ∧
  (AD = ⟂ (A, B)) ∧
  (midpoint A B M) ∧
  (AC = 2)

-- Theorem statement
theorem length_segment_DM (h : conditions) : DM = 1 :=
by
  sorry

end length_segment_DM_l422_422852


namespace compute_pairs_a_b_l422_422508

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem compute_pairs_a_b (a b : ℝ) (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -b) :
  ((∀ x, f (f x a b) a b = -1 / x) ↔ (a = -1 ∧ b = 1)) :=
sorry

end compute_pairs_a_b_l422_422508


namespace ABCD_is_parallelogram_l422_422108

variables {A B C D K L M N P Q : Type} [AddGroup V] [VectorSpace ℝ V]

-- Define points
variables {A B C D K L M N P Q : V}

-- Define that K, L, M, and N are midpoints of sides
def midpoint (x y : V) : V := (x + y) / 2

-- Conditions
def K_midpoint : K = midpoint A B := sorry
def L_midpoint : L = midpoint B C := sorry
def M_midpoint : M = midpoint C D := sorry
def N_midpoint : N = midpoint D A := sorry

-- Intersection points
def intersection_A_L_C_K (P : V) : Prop := sorry
def intersection_A_M_C_N (Q : V) : Prop := sorry

-- Given parallelogram condition
def APCQ_parallelogram : is_parallelogram A P C Q := sorry

-- The main theorem to prove ABCD is a parallelogram
theorem ABCD_is_parallelogram (h1: K_midpoint)
                             (h2: L_midpoint)
                             (h3: M_midpoint)
                             (h4: N_midpoint)
                             (h5: intersection_A_L_C_K P)
                             (h6: intersection_A_M_C_N Q)
                             (h7: APCQ_parallelogram) :
  is_parallelogram A B C D := sorry

end ABCD_is_parallelogram_l422_422108


namespace problem_statement_l422_422419

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a₁ d : ℤ, ∀ n : ℕ, a n = a₁ + n * d

theorem problem_statement :
  ∃ a : ℕ → ℤ, is_arithmetic_sequence a ∧
  a 2 = 7 ∧
  a 4 + a 6 = 26 ∧
  (∀ n : ℕ, a (n + 1) = 2 * n + 1) ∧
  ∃ S : ℕ → ℤ, (S n = n^2 + 2 * n) ∧
  ∃ b : ℕ → ℚ, (∀ n : ℕ, b n = 1 / (a n ^ 2 - 1)) ∧
  ∃ T : ℕ → ℚ, (T n = (n / 4) * (1 / (n + 1))) :=
sorry

end problem_statement_l422_422419


namespace mrs_randall_total_teaching_years_l422_422167

def years_teaching_third_grade : ℕ := 18
def years_teaching_second_grade : ℕ := 8

theorem mrs_randall_total_teaching_years : years_teaching_third_grade + years_teaching_second_grade = 26 :=
by
  sorry

end mrs_randall_total_teaching_years_l422_422167


namespace min_distance_eq_sqrt2_div2_l422_422748

open Real

def vec := ℝ × ℝ

def dot_product (u v : vec) : ℝ := u.1 * v.1 + u.2 * v.2

def distance_point_to_line (x y : ℝ) (a b c : ℝ) : ℝ := abs (a * x + b * y + c) / sqrt (a ^ 2 + b ^ 2)

noncomputable def minimum_distance (m n : ℝ) (a b c : vec) : ℝ := 
  let c := (m, n)
  let vec_a := a
  let vec_b := b
  let vec_c := c
  if dot_product ((vec_a.1 - vec_c.1, vec_a.2 - vec_c.2))  ((vec_b.1 - vec_c.1, vec_b.2 - vec_c.2)) = 0 then
    let center := ((1 / 2 : ℝ), (1 / 2 : ℝ))
    let radius := sqrt 2 / 2
    distance_point_to_line center.1 center.2 1 1 1 - radius
  else 0

theorem min_distance_eq_sqrt2_div2 :
  ∀ (m n : ℝ), dot_product ((1 - m, 0 - n)) ((0 - m, 1 - n)) = 0 → minimum_distance m n (1, 0) (0, 1) = sqrt 2 / 2 := 
by
  intros m n h
  simp only [minimum_distance]
  rw [if_pos]
  simp
  sorry

end min_distance_eq_sqrt2_div2_l422_422748


namespace stock_decrease_to_original_l422_422319

/-
Theorem: Given a stock investment that increased by 30% in 2006 and by an additional 20% in 2007,
a decrease of 35.9% is needed in 2008 to bring the stock's value back to its original price at
the beginning of 2006.
-/
theorem stock_decrease_to_original (x : ℝ) (h1 : x > 0) :
  let x_2006 := 1.30 * x,
      x_2007 := 1.20 * x_2006,
      final_value := x_2007 * (1 - 0.359)
  in final_value = x :=
by
  sorry

end stock_decrease_to_original_l422_422319


namespace students_in_each_normal_class_l422_422442

theorem students_in_each_normal_class
  (initial_students : ℕ)
  (percent_to_move : ℚ)
  (grade_levels : ℕ)
  (students_in_advanced_class : ℕ)
  (num_of_normal_classes : ℕ)
  (h_initial_students : initial_students = 1590)
  (h_percent_to_move : percent_to_move = 0.4)
  (h_grade_levels : grade_levels = 3)
  (h_students_in_advanced_class : students_in_advanced_class = 20)
  (h_num_of_normal_classes : num_of_normal_classes = 6) :
  let students_moving := (initial_students : ℚ) * percent_to_move,
      students_per_grade := students_moving / grade_levels,
      students_remaining := students_per_grade - (students_in_advanced_class : ℚ),
      students_per_normal_class := students_remaining / (num_of_normal_classes : ℚ)
  in students_per_normal_class = 32 := 
by
  sorry

end students_in_each_normal_class_l422_422442


namespace pure_imaginary_square_l422_422463

variable {a : ℝ}
variable {i : ℂ}

theorem pure_imaginary_square (h : (1 + a * i)^2.im = 0) : a = 1 ∨ a = -1 := 
by sorry

end pure_imaginary_square_l422_422463


namespace infinite_integer_solutions_l422_422893

theorem infinite_integer_solutions 
  (a b c k D x0 y0 : ℤ) 
  (hD_pos : D = b^2 - 4 * a * c) 
  (hD_non_square : (∀ n : ℤ, D ≠ n^2)) 
  (hk_nonzero : k ≠ 0) 
  (h_initial_sol : a * x0^2 + b * x0 * y0 + c * y0^2 = k) :
  ∃ (X Y : ℤ), a * X^2 + b * X * Y + c * Y^2 = k ∧
  (∀ (m : ℕ), ∃ (Xm Ym : ℤ), a * Xm^2 + b * Xm * Ym + c * Ym^2 = k ∧
  (Xm, Ym) ≠ (x0, y0)) :=
sorry

end infinite_integer_solutions_l422_422893


namespace solve_nat_numbers_equation_l422_422921

theorem solve_nat_numbers_equation (n k l m : ℕ) (h_l : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n = 2) ∧ (k = 1) ∧ (l = 2) ∧ (m = 3) := 
by
  sorry

end solve_nat_numbers_equation_l422_422921


namespace probability_of_prime_is_0_3_l422_422260

noncomputable def count_primes : ℕ :=
15

noncomputable def total_numbers : ℕ :=
50

def probability_prime : ℝ :=
(count_primes : ℝ) / (total_numbers : ℝ)

theorem probability_of_prime_is_0_3 :
  probability_prime = 0.3 :=
by
  sorry

end probability_of_prime_is_0_3_l422_422260


namespace points_concyclic_ABDC_l422_422854

variables {A B C D K N M O : Type} [acute_triangle_ABC A B C O] 
          (K_on_BC : on_segment K B C) (not_midpoint_K : ¬ midpoint K B C)
          (D_on_AK_ext : on_extended_segment D A K) (BD_intersects_AC : intersects (line B D) (line A C N))
          (CD_intersects_AB : intersects (line C D) (line A B M)) (OK_perp_MN : perpendicular (line O K) (line M N))

-- statement to prove
theorem points_concyclic_ABDC : concyclic A B D C := 
sorry

end points_concyclic_ABDC_l422_422854


namespace taxi_ride_distance_l422_422102

theorem taxi_ride_distance (initial_fare additional_fare total_fare : ℝ) 
  (initial_distance : ℝ) (additional_distance increment_distance : ℝ) :
  initial_fare = 1.0 →
  additional_fare = 0.45 →
  initial_distance = 1/5 →
  increment_distance = 1/5 →
  total_fare = 7.3 →
  additional_distance = (total_fare - initial_fare) / additional_fare →
  (initial_distance + additional_distance * increment_distance) = 3 := 
by sorry

end taxi_ride_distance_l422_422102


namespace ratio_of_areas_of_concentric_circles_l422_422968

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_concentric_circles_l422_422968


namespace smallest_average_l422_422388

noncomputable def smallest_possible_average : ℕ := 165 / 10

theorem smallest_average (s d: Finset ℕ) 
  (h1 : s.card = 3) 
  (h2 : d.card = 3) 
  (h3 : ∀x ∈ s ∪ d, x ∈ (Finset.range 10).erase 0)
  (h4 : (s ∪ d).card = 6)
  (h5 : s ∩ d = ∅) : 
  (∑ x in s, x + ∑ y in d, y) / 6 = smallest_possible_average :=
sorry

end smallest_average_l422_422388


namespace divisor_count_of_fourth_power_l422_422452

theorem divisor_count_of_fourth_power (n : ℕ) (h : n > 0) :
  let y := n^4 in 
  let d := (factors_multiset y).to_finset.card in
  d = 405 :=
sorry

end divisor_count_of_fourth_power_l422_422452


namespace find_a_l422_422790

def f (x : ℝ) : ℝ :=
  if x >= 0 then 2^x - 1 else -x^2 - 2*x

theorem find_a (a : ℝ) : f(a) = 1 ↔ a = 1 ∨ a = -1 := by
  sorry

end find_a_l422_422790


namespace no_real_b_for_line_through_vertex_l422_422746

theorem no_real_b_for_line_through_vertex :
  ¬ ∃ b : ℝ, let vertex := b^2 + 1 in vertex = b :=
by
  sorry

end no_real_b_for_line_through_vertex_l422_422746


namespace squares_total_l422_422005

def number_of_squares (figure : Type) : ℕ := sorry

theorem squares_total (figure : Type) : number_of_squares figure = 38 := sorry

end squares_total_l422_422005


namespace domain_of_function_l422_422560

noncomputable def function_domain := {x : ℝ | 1 + 1 / x > 0 ∧ 1 - x^2 ≥ 0}

theorem domain_of_function : function_domain = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l422_422560


namespace find_a_and_b_minimum_value_of_polynomial_l422_422425

noncomputable def polynomial_has_maximum (x y a b : ℝ) : Prop :=
  y = a * x ^ 3 + b * x ^ 2 ∧ x = 1 ∧ y = 3

noncomputable def polynomial_minimum_value (y : ℝ) : Prop :=
  y = 0

theorem find_a_and_b (a b x y : ℝ) (h : polynomial_has_maximum x y a b) :
  a = -6 ∧ b = 9 :=
by sorry

theorem minimum_value_of_polynomial (a b y : ℝ) (h : a = -6 ∧ b = 9) :
  polynomial_minimum_value y :=
by sorry

end find_a_and_b_minimum_value_of_polynomial_l422_422425


namespace largest_N_l422_422727

theorem largest_N (N : ℕ) : 
  (∀ k l : ℕ, k ∈ set.Icc 80 99 → l ∈ set.Icc 80 99 → k ≠ l → (reciprocal k! - reciprocal l!).abs ≠ 0) →
  (∀ k : ℕ, k ∈ set.Icc 80 99 → has_decimal_sequence_length k! N → False) →
  (∃ n : ℕ, n = 155) :=
by
  sorry

end largest_N_l422_422727


namespace inverse_log_function_l422_422207

noncomputable def inverse_function (a : ℝ) (y : ℝ) : ℝ :=
  a^(y - 2) - 1

theorem inverse_log_function (a : ℝ) (x : ℝ) (hx : x > -1) :
  ∃ y, y = 2 + log a (x + 1) ∧ (∀ z, z = 2 + log a (x + 1) → y = a^(z - 2) - 1) :=
by
  sorry

end inverse_log_function_l422_422207


namespace part_I_a_part_I_b_part_I_c_part_II_part_III_l422_422072

namespace ProofProblems

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- Problem 1: f(2) + f(1/2) = 1
theorem part_I_a : f 2 + f (1 / 2) = 1 :=
by sorry

-- Problem 1: f(3) + f(1/3) = 1
theorem part_I_b : f 3 + f (1 / 3) = 1 :=
by sorry

-- Problem 1: f(4) + f(1/4) = 1
theorem part_I_c : f 4 + f (1 / 4) = 1 :=
by sorry

-- Problem 2: For all x ≠ 0, f(x) + f(1/x) = 1
theorem part_II (x : ℝ) (hx : x ≠ 0) : f x + f (1 / x) = 1 :=
by sorry

-- Problem 3: Sum f(1) + f(2) + ... + f(2011) + f(1/2011) + ... + f(1) = 2011
theorem part_III : 
  let S := (List.range 2011).map (λ n, f (n + 1)) ++ (List.range 2011).map (λ n, f (1 / (n + 1)))
  S.sum = 2011 :=
by sorry

end ProofProblems

end part_I_a_part_I_b_part_I_c_part_II_part_III_l422_422072


namespace remainder_3n_mod_7_l422_422101

theorem remainder_3n_mod_7 (n : ℤ) (k : ℤ) (h : n = 7*k + 1) :
  (3 * n) % 7 = 3 := by
  sorry

end remainder_3n_mod_7_l422_422101


namespace find_angle_BCA_l422_422867

open Real

variables (A B C M : Type) [EuclideanGeometry A B C]
variables (angle : A → A → MeasureTheory.Measure ℝ)

def is_median (A B C M : Type) [EuclideanGeometry A B C] : Prop :=
  distance A M = distance M C

theorem find_angle_BCA :
  is_median A B C M →
  angle B A C = 30 →
  angle B M C = 45 →
  angle B C A = 105 :=
by
sorry

end find_angle_BCA_l422_422867


namespace sum_valid_set_eq_95004_l422_422148

-- Define the set of non-negative integers n satisfying the given floor condition
def valid_set : Set ℤ := { n | n ≥ 0 ∧ (n / 27 = n / 28) }

-- Define the sum of all elements in the valid set
noncomputable def A : ℤ := ∑ n in valid_set, n

-- The main theorem which we need to prove
theorem sum_valid_set_eq_95004 : A = 95004 := 
by
  sorry

end sum_valid_set_eq_95004_l422_422148


namespace nth_monomial_pattern_l422_422491

variables (n : ℕ)
def nth_monomial (n : ℕ) : ℤ := (-1)^(n + 1) * x^(2 * n - 1)

theorem nth_monomial_pattern (n : ℕ) : 
  nth_monomial n = (-1)^(n + 1) * x^(2 * n - 1) :=
sorry

end nth_monomial_pattern_l422_422491


namespace class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l422_422483

theorem class_contribution_Miss_Evans :
  let total_contribution : ℝ := 90
  let class_funds_Evans : ℝ := 14
  let num_students_Evans : ℕ := 19
  let individual_contribution_Evans : ℝ := (total_contribution - class_funds_Evans) / num_students_Evans
  individual_contribution_Evans = 4 := 
sorry

theorem class_contribution_Mr_Smith :
  let total_contribution : ℝ := 90
  let class_funds_Smith : ℝ := 20
  let num_students_Smith : ℕ := 15
  let individual_contribution_Smith : ℝ := (total_contribution - class_funds_Smith) / num_students_Smith
  individual_contribution_Smith = 4.67 := 
sorry

theorem class_contribution_Mrs_Johnson :
  let total_contribution : ℝ := 90
  let class_funds_Johnson : ℝ := 30
  let num_students_Johnson : ℕ := 25
  let individual_contribution_Johnson : ℝ := (total_contribution - class_funds_Johnson) / num_students_Johnson
  individual_contribution_Johnson = 2.40 := 
sorry

end class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l422_422483


namespace angle_ECD_l422_422933

noncomputable def latitude_C : Float := 0  
noncomputable def longitude_C : Float := 9 
noncomputable def latitude_D : Float := 47.92 
noncomputable def longitude_D : Float := 106.92 

noncomputable def radius_Earth : Float := 6371 

noncomputable def convert_to_cartesian (lat : Float) (lon : Float) : (Float × Float × Float) :=
  let deg_to_rad := Float.pi / 180
  let lat_rad := lat * deg_to_rad
  let lon_rad := lon * deg_to_rad
  let x := radius_Earth * Float.cos lat_rad * Float.cos lon_rad
  let y := radius_Earth * Float.cos lat_rad * Float.sin lon_rad
  let z := radius_Earth * Float.sin lat_rad
  (x, y, z)

noncomputable def vector_EC : (Float × Float × Float) := convert_to_cartesian latitude_C longitude_C
noncomputable def vector_ED : (Float × Float × Float) := convert_to_cartesian latitude_D longitude_D

noncomputable def dot_product (v1 v2 : (Float × Float × Float)) : Float :=
  let (x1, y1, z1) := v1
  let (x2, y2, z2) := v2
  x1 * x2 + y1 * y2 + z1 * z2

noncomputable def magnitude (v : (Float × Float × Float)) : Float :=
  let (x, y, z) := v
  Float.sqrt (x*x + y*y + z*z)

noncomputable def angle_between (v1 v2 : (Float × Float × Float)) : Float :=
  let cos_theta := dot_product v1 v2 / (magnitude v1 * magnitude v2)
  Float.acos cos_theta * (180 / Float.pi)

theorem angle_ECD :
  ∃ θ : Float, θ = angle_between vector_EC vector_ED := by
  -- Proof is skipped
  sorry

end angle_ECD_l422_422933


namespace largest_last_digit_l422_422563

theorem largest_last_digit (s : List Nat) (h1 : s.head = 2) (h2 : ∀ (i : Nat), i < 2006 → (s[i] * 10 + s[i+1]) % 23 = 0 ∨ (s[i] * 10 + s[i+1]) % 37 = 0) : 
(s[2006] = 9) :=
sorry

end largest_last_digit_l422_422563


namespace cake_eaten_after_four_trips_l422_422250

-- Define the fraction of the cake eaten on each trip
def fraction_eaten (n : Nat) : ℚ :=
  (1 / 3) ^ n

-- Define the total cake eaten after four trips
def total_eaten_after_four_trips : ℚ :=
  fraction_eaten 1 + fraction_eaten 2 + fraction_eaten 3 + fraction_eaten 4

-- The mathematical statement we want to prove
theorem cake_eaten_after_four_trips : total_eaten_after_four_trips = 40 / 81 := 
by
  sorry

end cake_eaten_after_four_trips_l422_422250


namespace quadratic_has_real_roots_l422_422800

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) := 
by 
  sorry

end quadratic_has_real_roots_l422_422800


namespace books_arrangement_l422_422448

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def binom (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem books_arrangement (h_highlight : Bool) : 
  (if h_highlight then 21 * (factorial 7 / (factorial 2 * factorial 2)) else
   factorial 7 / (factorial 2 * factorial 2)) = 
  (if h_highlight then 26460 else 1260) := 
sorry

end books_arrangement_l422_422448


namespace ratio_of_larger_to_smaller_l422_422221

theorem ratio_of_larger_to_smaller 
    (x y : ℝ) 
    (hx : x > 0) 
    (hy : y > 0) 
    (h : x + y = 7 * (x - y)) : 
    x / y = 4 / 3 := 
by 
    sorry

end ratio_of_larger_to_smaller_l422_422221


namespace area_three_layers_l422_422962

def total_area_rugs : ℝ := 200
def floor_covered_area : ℝ := 140
def exactly_two_layers_area : ℝ := 24

theorem area_three_layers : (2 * (200 - 140 - 24) / 2 = 2 * 18) := 
by admit -- since we're instructed to skip the proof

end area_three_layers_l422_422962


namespace smallest_n_for_Qn_l422_422843

theorem smallest_n_for_Qn (n : ℕ) : 
  (∃ n : ℕ, 1 / (n * (2 * n + 1)) < 1 / 2023 ∧ ∀ m < n, 1 / (m * (2 * m + 1)) ≥ 1 / 2023) ↔ n = 32 := by
sorry

end smallest_n_for_Qn_l422_422843


namespace perpendicular_lines_l422_422896

def line1 (m : ℝ) (x y : ℝ) := m * x - 3 * y - 1 = 0
def line2 (m : ℝ) (x y : ℝ) := (3 * m - 2) * x - m * y + 2 = 0

theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, line1 m x y) →
  (∀ x y : ℝ, line2 m x y) →
  (∀ x y : ℝ, (m / 3) * ((3 * m - 2) / m) = -1) →
  m = 0 ∨ m = -1/3 :=
by
  intros
  sorry

end perpendicular_lines_l422_422896


namespace find_a_l422_422791

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then 2^x - 1 else -x^2 - 2*x

theorem find_a (a : ℝ) (h : f(a) = 1) : a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l422_422791


namespace f_integer_iff_conditions_integer_l422_422393

theorem f_integer_iff_conditions_integer
  (f : ℤ → ℤ)
  (a b c : ℤ)
  (h_def : ∀ x : ℤ, f x = a * x^2 + b * x + c) :
  (∀ n : ℤ, ∃ i : ℤ, f n = i) ↔ (2 * a ∈ ℤ ∧ (a + b) ∈ ℤ ∧ c ∈ ℤ) :=
sorry

end f_integer_iff_conditions_integer_l422_422393


namespace sum_of_unit_vectors_square_l422_422771

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions of unit vectors and angle between them
variables (m n : V)
hypothesis h1 : ∥m∥ = 1
hypothesis h2 : ∥n∥ = 1
hypothesis h3 : ⟪m, n⟫ = (1 / 2 : ℝ)

-- The theorem statement
theorem sum_of_unit_vectors_square : 
  (∥m + n∥ ^ 2 = 3) :=
sorry

end sum_of_unit_vectors_square_l422_422771


namespace percent_enclosed_by_hexagons_l422_422216

theorem percent_enclosed_by_hexagons (a : ℝ) (h1 : a > 0) :
  let large_square_area := (4 * a)^2 in
  let small_square_area := a^2 in
  let hexagons_area := 10 * small_square_area in
  hexagons_area / large_square_area * 100 = 62.5 :=
by
  let large_square_area := (4 * a)^2
  let small_square_area := a^2
  let hexagons_area := 10 * small_square_area
  have area_fraction : hexagons_area / large_square_area = 10 / 16 := by
    sorry
  have percent_conversion : (10 / 16) * 100 = 62.5 := by
    sorry
  exact percent_conversion

end percent_enclosed_by_hexagons_l422_422216


namespace isabella_score_sixth_test_l422_422134

def test_scores : Type := { s : ℕ // 91 ≤ s ∧ s ≤ 100 }

variables {a b c d e f g : test_scores}

theorem isabella_score_sixth_test (h_diff : a.val ≠ b.val ∧ a.val ≠ c.val ∧ a.val ≠ d.val ∧ a.val ≠ e.val ∧ a.val ≠ f.val ∧ a.val ≠ g.val ∧ 
                                   b.val ≠ c.val ∧ b.val ≠ d.val ∧ b.val ≠ e.val ∧ b.val ≠ f.val ∧ b.val ≠ g.val ∧ 
                                   c.val ≠ d.val ∧ c.val ≠ e.val ∧ c.val ≠ f.val ∧ c.val ≠ g.val ∧ 
                                   d.val ≠ e.val ∧ d.val ≠ f.val ∧ d.val ≠ g.val ∧ 
                                   e.val ≠ f.val ∧ e.val ≠ g.val ∧ 
                                   f.val ≠ g.val)
  (h_avg_integral : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7}, ∃ (avg : ℤ), 
    avg * n = (finset.image (λ i : fin n, (list.nth_le [a.val, b.val, c.val, d.val, e.val, f.val, g.val] i i.2)).sum id))
  (h_g_eq_95 : g.val = 95) :
  f.val = 100 := sorry

end isabella_score_sixth_test_l422_422134


namespace correct_option_l422_422518

variable (p q : Prop)

/-- If only one of p and q is true, then p or q is a true proposition. -/
theorem correct_option (h : (p ∧ ¬ q) ∨ (¬ p ∧ q)) : p ∨ q :=
by sorry

end correct_option_l422_422518


namespace A_inter_B_A_union_B_complementU_A_union_complementU_B_complementU_A_inter_complementU_B_l422_422079

variables (U : Set ℝ) (A B : Set ℝ)

-- Definitions
def A : Set ℝ := {x | x < 1 ∨ x > 2}
def B : Set ℝ := {x | x < -3 ∨ x ≥ 1}
def U : Set ℝ := {x | true}

-- Statements to prove
theorem A_inter_B : A ∩ B = {x | x < -3 ∨ x > 2} :=
by sorry

theorem A_union_B : A ∪ B = U :=
by sorry

theorem complementU_A_union_complementU_B : (U \ A) ∪ (U \ B) = {x | -3 ≤ x ∧ x ≤ 2} :=
by sorry

theorem complementU_A_inter_complementU_B : (U \ A) ∩ (U \ B) = ∅ :=
by sorry

end A_inter_B_A_union_B_complementU_A_union_complementU_B_complementU_A_inter_complementU_B_l422_422079


namespace room_area_cm_squared_l422_422704

noncomputable def length_feet : ℕ := 14
noncomputable def length_inches : ℕ := 8
noncomputable def width_feet : ℕ := 10
noncomputable def width_inches : ℕ := 5
noncomputable def cm_per_foot : ℝ := 30.48
noncomputable def cm_per_inch : ℝ := 2.54

theorem room_area_cm_squared :
  let length_cm := length_feet * cm_per_foot + length_inches * cm_per_inch
  let width_cm := width_feet * cm_per_foot + width_inches * cm_per_inch in
  length_cm * width_cm = 141935.4 :=
by
  -- Proof skipped
  sorry

end room_area_cm_squared_l422_422704


namespace fraction_of_science_liking_students_l422_422837

open Real

theorem fraction_of_science_liking_students (total_students math_fraction english_fraction no_fav_students math_students english_students fav_students remaining_students science_students fraction_science) :
  total_students = 30 ∧
  math_fraction = 1/5 ∧
  english_fraction = 1/3 ∧
  no_fav_students = 12 ∧
  math_students = total_students * math_fraction ∧
  english_students = total_students * english_fraction ∧
  fav_students = total_students - no_fav_students ∧
  remaining_students = fav_students - (math_students + english_students) ∧
  science_students = remaining_students ∧
  fraction_science = science_students / remaining_students →
  fraction_science = 1 :=
by
  sorry

end fraction_of_science_liking_students_l422_422837


namespace ratio_of_areas_of_concentric_circles_l422_422977

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end ratio_of_areas_of_concentric_circles_l422_422977


namespace option_D_is_correct_l422_422620

noncomputable def correct_operation : Prop := 
  (∀ x : ℝ, x + x ≠ 2 * x^2) ∧
  (∀ y : ℝ, 2 * y^3 + 3 * y^2 ≠ 5 * y^5) ∧
  (∀ x : ℝ, 2 * x - x ≠ 1) ∧
  (∀ x y : ℝ, 4 * x^3 * y^2 - (-2)^2 * x^3 * y^2 = 0)

theorem option_D_is_correct : correct_operation :=
by {
  -- We'll complete the proofs later
  sorry
}

end option_D_is_correct_l422_422620


namespace abs_quadratic_inequality_solution_l422_422002

theorem abs_quadratic_inequality_solution (x : ℝ) :
  |x^2 - 4 * x + 3| ≤ 3 ↔ 0 ≤ x ∧ x ≤ 4 :=
by sorry

end abs_quadratic_inequality_solution_l422_422002


namespace average_speed_calculation_l422_422648

-- Define the conditions for each hour of travel.
def first_hour_speed := 80 -- km/h
def first_hour_time := 1 -- hour

def second_hour_speed := 120 -- km/h
def second_hour_time := 1 -- hour
def rest_break := 0.25 -- 15 minutes in hours

def third_hour_speed := 60 -- km/h
def third_hour_time := 1 -- hour

-- Define the problem statement in Lean.
theorem average_speed_calculation :
  let total_distance := (first_hour_speed * first_hour_time) + (second_hour_speed * second_hour_time) + (third_hour_speed * third_hour_time)
  let total_time := first_hour_time + second_hour_time + rest_break + third_hour_time
  total_distance / total_time = 80 := by
  sorry

end average_speed_calculation_l422_422648


namespace intersection_angle_bisectors_on_side_l422_422886

open EuclideanGeometry

variable {P : Type} [MetricSpace P] {A B C D W : P}

-- Define chordal quadrilateral and distance conditions
noncomputable def chordalQuadrilateralCondition (A B C D : P) : Prop :=
  distance A B + distance C D = distance B C

-- Define the problem stating that W is the intersection of angle bisectors
theorem intersection_angle_bisectors_on_side (h : chordalQuadrilateralCondition A B C D) :
  ∃ W : P, (angleBisectorPoint W A B D ∧ angleBisectorPoint W C D A) ∧ lies_on_segment W B C :=
sorry

end intersection_angle_bisectors_on_side_l422_422886


namespace fly_movement_l422_422229

-- Define the grid size and properties
def grid_size : ℕ := 9

-- Define initial positions of flies (as a list of coordinates)
def initial_positions : list (ℕ × ℕ) := [(1, 2), (3, 1), (5, 6), (x₄, y₄), (x₅, y₅), (x₆, y₆), (x₇, y₇), (x₈, y₈), (x₉, y₉)]

-- Define the new positions for the three flies that move
def new_positions : list (ℕ × ℕ) := [(1, 3), (4, 1), (4, 7)]

-- Define the theorem asserting the movement and placing conditions
theorem fly_movement (x₄ y₄ x₅ y₅ x₆ y₆ x₇ y₇ x₈ y₈ x₉ y₉ : ℕ) :
    (∀ i j, i ≠ j → initial_positions.nth i ≠ initial_positions.nth j) →
    (∀ (i : ℕ) (hi : i < grid_size) (j : ℕ) (hj : j < grid_size), 
        (initial_positions.nth i = some (x₄, y₄) ∨ initial_positions.nth i = some (x₅, y₅) ∨ initial_positions.nth i = some (x₆, y₆)) ↔ 
        (initial_positions.nth j = some (x₇, y₇) ∨ initial_positions.nth j = some (x₈, y₈) ∨ initial_positions.nth j = some (x₉, y₉))) →
    (∀ (i : ℕ) (hi : i < grid_size) (x y : ℕ) (hxy : (x, y) ∈ new_positions),
        (initial_positions.nth i = some (x₄, y₄) ∨ initial_positions.nth i = some (x₅, y₅) ∨ initial_positions.nth i = some (x₆, y₆))) →
    (∀ (i j : ℕ) (hi : i < 3) (hj : j < 3), i ≠ j → new_positions.nth i ≠ new_positions.nth j) →
    true :=
by sorry

end fly_movement_l422_422229


namespace integral_ln_x_eval_l422_422007

-- Define the integral of ln x from 1 to e
theorem integral_ln_x_eval : ∫ x in 1..Real.exp 1, Real.log x = 1 :=
by
  sorry

end integral_ln_x_eval_l422_422007


namespace proof_problem_l422_422409

-- Conditions
def imag_unit (i : ℂ) : Prop := i * i = -1
def condition1 (z : ℂ) (i : ℂ) : Prop := (z - 2) * i = -3 - i
def condition2 (z : ℂ) (w : ℂ) : Prop := z = w → ∃ x : ℝ, (x + complex.I) / z = w → -3 < x ∧ x < 1 / 3

-- The main theorem that needs to be proved
theorem proof_problem (i z : ℂ) (x : ℝ) : 
  imag_unit i → 
  condition1 z i → 
  condition2 z (1 + 3 * complex.I) :=
by
  sorry

end proof_problem_l422_422409


namespace unique_symmetric_solutions_l422_422647

theorem unique_symmetric_solutions (a b α β : ℝ) (h_mul : α * β = a) (h_add : α + β = b) :
  ∀ (x y : ℝ), x * y = a ∧ x + y = b → (x = α ∧ y = β) ∨ (x = β ∧ y = α) :=
by
  sorry

end unique_symmetric_solutions_l422_422647


namespace johns_age_l422_422191

theorem johns_age :
  ∃ J : ℝ, J - 10 = (1 / 3) * (J + 15) ∧ J = 22.5 := 
by 
  use 22.5
  split
  -- Proof step to show the equation holds with J = 22.5
  sorry
  -- Proof step to show J is indeed 22.5
  refl

end johns_age_l422_422191


namespace p_minus_q_l422_422146

-- Define the given equation as a predicate.
def eqn (x : ℝ) : Prop := (3*x - 9) / (x*x + 3*x - 18) = x + 3

-- Define the values p and q as distinct solutions.
def p_and_q (p q : ℝ) : Prop := eqn p ∧ eqn q ∧ p ≠ q ∧ p > q

theorem p_minus_q {p q : ℝ} (h : p_and_q p q) : p - q = 2 := sorry

end p_minus_q_l422_422146


namespace tim_investment_amount_l422_422903

variables (T : ℝ) (im_rate_tim im_rate_lana : ℝ) (initial_lana : ℝ) (extra_interest : ℝ)

def interest_earned (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * ((1 + r) ^ n - 1)

def total_interest_tim_2Y : ℝ :=
  interest_earned T im_rate_tim 2

def total_interest_lana_2Y : ℝ :=
  interest_earned initial_lana im_rate_lana 2

theorem tim_investment_amount :
  im_rate_tim = 0.10 →
  im_rate_lana = 0.05 →
  initial_lana = 800 →
  extra_interest = 44.000000000000114 →
  total_interest_tim_2Y - total_interest_lana_2Y = extra_interest →
  T = 600.0000000000005 :=
by
  intros h_im_rate_tim h_im_rate_lana h_initial_lana h_extra_interest h_interest_diff
  sorry

end tim_investment_amount_l422_422903


namespace min_quotient_l422_422510

theorem min_quotient {a b : ℕ} (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 400 ≤ b) (h₄ : b ≤ 800) (h₅ : a + b ≤ 950) : a / b = 1 / 8 := 
by
  sorry

end min_quotient_l422_422510


namespace shaded_area_isosceles_right_triangle_l422_422330

/-- Given an isosceles right triangle with legs of length 10 partitioned into 25 congruent smaller triangles, where 15 of these triangles are shaded, we prove that the area of the shaded region is 30. -/
theorem shaded_area_isosceles_right_triangle :
  ∀ (A B C : Type) (isosceles_right_triangle : A) (congruent_smaller_triangles : B) (shaded_triangles : C) 
    [number_of_smaller_triangles : DecidableEq B] [number_of_shaded_triangles : DecidableEq C],
    (IsoscelesRightTriangle.hasLegsOfLength isosceles_right_triangle 10 10) → 
    (IsoscelesRightTriangle.partitionInto isosceles_right_triangle congruent_smaller_triangles 25) → 
    (ShadedRegion.includes shaded_triangles congruent_smaller_triangles 15) →
    (Area.ofShadedRegion shaded_triangles 30) := 
by
  intro A B C isosceles_right_triangle congruent_smaller_triangles shaded_triangles _ _
  intro h1 h2 h3
  have : Area.ofShadedRegion shaded_triangles 30 := sorry
  exact this

end shaded_area_isosceles_right_triangle_l422_422330


namespace value_of_expression_l422_422093

theorem value_of_expression
  (a b x y : ℝ)
  (h1 : a + b = 0)
  (h2 : x * y = 1) : 
  2 * (a + b) + (7 / 4) * (x * y) = 7 / 4 := 
sorry

end value_of_expression_l422_422093


namespace ratio_of_smaller_circle_to_larger_circle_l422_422973

section circles

variables {Q : Type} (C1 C2 : ℝ) (angle1 : ℝ) (angle2 : ℝ)

def ratio_of_areas (C1 C2 : ℝ) : ℝ := (C1 / C2)^2

theorem ratio_of_smaller_circle_to_larger_circle
  (h1 : angle1 = 60)
  (h2 : angle2 = 48)
  (h3 : (angle1 / 360) * C1 = (angle2 / 360) * C2) :
  ratio_of_areas C1 C2 = 16 / 25 :=
by
  sorry

end circles

end ratio_of_smaller_circle_to_larger_circle_l422_422973


namespace possible_degrees_remainder_l422_422991

theorem possible_degrees_remainder (p : Polynomial ℝ) : 
  ∀ q : Polynomial ℝ, degree q = 3 → (∀ r : Polynomial ℝ, r.degree < 3) := 
sorry

end possible_degrees_remainder_l422_422991


namespace alpha_value_l422_422770

theorem alpha_value (α : ℝ) (h1 : -π/2 < α ∧ α < π/2) (h2 : sin α + cos α = sqrt 2 / 2) : α = -π/12 :=
sorry

end alpha_value_l422_422770


namespace f_positive_when_a_1_f_negative_solution_sets_l422_422750

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

end f_positive_when_a_1_f_negative_solution_sets_l422_422750


namespace stephen_total_distance_l422_422184

theorem stephen_total_distance :
  let mountain_height := 40000
  let ascent_fraction := 3 / 4
  let descent_fraction := 2 / 3
  let extra_distance_fraction := 0.10
  let normal_trips := 8
  let harsh_trips := 2
  let ascent_distance := ascent_fraction * mountain_height
  let descent_distance := descent_fraction * ascent_distance
  let normal_trip_distance := ascent_distance + descent_distance
  let harsh_trip_extra_distance := extra_distance_fraction * ascent_distance
  let harsh_trip_distance := ascent_distance + harsh_trip_extra_distance + descent_distance
  let total_normal_distance := normal_trip_distance * normal_trips
  let total_harsh_distance := harsh_trip_distance * harsh_trips
  let total_distance := total_normal_distance + total_harsh_distance
  total_distance = 506000 :=
by
  sorry

end stephen_total_distance_l422_422184


namespace complex_div_simplification_l422_422200

theorem complex_div_simplification : (1 + 2*Complex.i) / (2 - Complex.i) = Complex.i :=
by
  sorry

end complex_div_simplification_l422_422200


namespace sqrt_sum_of_powers_l422_422600

theorem sqrt_sum_of_powers : sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  have h : 4^4 = 256 := by
    calc
      4^4 = 4 * 4 * 4 * 4 := by rw [show 4^4 = 4 * 4 * 4 * 4 from rfl]
      ... = 16 * 16       := by rw [mul_assoc, mul_assoc]
      ... = 256           := by norm_num
  calc
    sqrt (4^4 + 4^4 + 4^4 + 4^4)
      = sqrt (256 + 256 + 256 + 256) := by rw [h, h, h, h]
      ... = sqrt 1024                 := by norm_num
      ... = 32                        := by norm_num

end sqrt_sum_of_powers_l422_422600


namespace probability_one_solve_l422_422174

variables {p1 p2 : ℝ}

theorem probability_one_solve (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) := 
sorry

end probability_one_solve_l422_422174


namespace shaded_area_percentage_l422_422243

theorem shaded_area_percentage :
  let area_square := 6 * 6 in
  let area_first_rectangle := 2 * 2 in
  let area_second_rectangle := 1 * 6 in
  let area_third_rectangle := 1 * 6 in
  let total_shaded_area := area_first_rectangle + area_second_rectangle + area_third_rectangle in
  (total_shaded_area.toFloat / area_square * 100) = 44.44 :=
by 
  sorry

end shaded_area_percentage_l422_422243


namespace ratio_of_shaded_area_l422_422026

-- Definitions
variable (S : Type) [Field S]
variable (square_area shaded_area : S) -- Areas of the square and the shaded regions.
variable (PX XQ : S) -- Lengths such that PX = 3 * XQ.

-- Conditions
axiom condition1 : PX = 3 * XQ
axiom condition2 : shaded_area / square_area = 0.375

-- Goal
theorem ratio_of_shaded_area (PX XQ square_area shaded_area : S) [Field S] 
  (condition1 : PX = 3 * XQ)
  (condition2 : shaded_area / square_area = 0.375) : shaded_area / square_area = 0.375 := 
  by
  sorry

end ratio_of_shaded_area_l422_422026


namespace same_terminal_side_as_30_degrees_l422_422950

def angles_with_same_terminal_side (theta : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = theta + k * 360}

theorem same_terminal_side_as_30_degrees :
  angles_with_same_terminal_side 30 = {α | ∃ k : ℤ, α = 30 + k * 360} :=
sorry

end same_terminal_side_as_30_degrees_l422_422950


namespace triangle_perimeter_AEC_l422_422318

/--
Let A, B, C, D be vertices of a square with side length 2, arranged in that order.
Let folding the paper such that vertex C meets edge AB at point C', where AC' = 1/4.
Let E be the intersection point of the edges BC and AD after the fold.
We aim to determine the perimeter of the triangle AEC', which is given to be (sqrt 65 + 1) / 4.
-/
theorem triangle_perimeter_AEC' :
  let A := (0, 2)
  let B := (0, 0)
  let C := (2, 0)
  let D := (2, 2)
  let C' := (1/4, 0)
  let E := (0, 2)
  let AC' := (0 - 1/4)^2 + (2 - 0)^2 = 1/16 + 4 → AC' = 1/4
  let EC' := (0 - 1/4)^2 + (2 - 0)^2 = sqrt (65 / 16) → EC' = sqrt(65) / 4
  let AE := 0
  AE + EC' + AC' = (sqrt 65 + 1) / 4 :=
sorry

end triangle_perimeter_AEC_l422_422318


namespace range_of_alpha_l422_422909

variable {x : ℝ}

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + 2

theorem range_of_alpha (x : ℝ) (α : ℝ) (h : α = Real.arctan (3*x^2 - 1)) :
  α ∈ Set.Ico 0 (Real.pi / 2) ∪ Set.Ico (3 * Real.pi / 4) Real.pi :=
sorry

end range_of_alpha_l422_422909


namespace sin_beta_is_neg_three_fifths_l422_422407

variable (α β : ℝ)

-- Given conditions
variable (h1 : cos (α - β) * cos α + sin (α - β) * sin α = -4/5)
variable (h2 : π < β ∧ β < 3 * π / 2) -- indicating β is in the third quadrant

-- Prove sin β = -3/5
theorem sin_beta_is_neg_three_fifths (h1 : cos (α - β) * cos α + sin (α - β) * sin α = -4/5)
    (h2 : π < β ∧ β < 3 * π / 2) : sin β = -3/5 := 
sorry

end sin_beta_is_neg_three_fifths_l422_422407


namespace lambda_sum_leq_n_l422_422889

theorem lambda_sum_leq_n 
  (λ : ℕ → ℝ)
  (n : ℕ)
  (h0 : n > 0)
  (g : ℝ → ℝ)
  (h1 : ∀ θ : ℝ, g θ ≥ -1)
  (h2 : ∀ θ : ℝ, g θ = ∑ i in finset.range n, λ i * real.cos ((i + 1) * θ)) :
  finset.sum (finset.range n) λ ≤ n :=
begin
  sorry
end

end lambda_sum_leq_n_l422_422889


namespace sufficient_not_necessary_l422_422639

theorem sufficient_not_necessary (x : ℝ) : (x > 0) → (x ≠ 0) ∧ (x ≠ 0 → (x > 0 ∨ x < 0)) :=
by
  intro h
  split
  · exact ne_of_gt h
  · intro h1
    sorry

end sufficient_not_necessary_l422_422639


namespace minimum_cubes_required_l422_422297

-- Define the conditions
def front_view_cubes : ℕ := 2
def side_view_cubes : ℕ := 3
def min_cubes : ℕ := 3

-- Statement of the proof problem
theorem minimum_cubes_required
  (front_view : ℕ)
  (side_view : ℕ)
  : front_view = front_view_cubes →
    side_view = side_view_cubes →
    ∃ cubes : ℕ, cubes = min_cubes ∧ cubes >= front_view ∧ cubes >= side_view :=
by
  intros h1 h2
  existsi min_cubes
  split
  · exact rfl
  · split
    · linarith only [h1, h2]
    · linarith only [h1, h2]
  · sorry

-- Ensure Lean can build the code
#check minimum_cubes_required

end minimum_cubes_required_l422_422297


namespace set_Y_numbers_l422_422181
open Set

def X := {p : ℕ | 10 < p ∧ p < 100 ∧ Nat.Prime p}

def range (s : Set ℕ) : ℕ :=
  s.sup id - s.inf id

theorem set_Y_numbers (Y : Set ℕ) 
  (hX : range X = 86) 
  (hXY : range (X ∪ Y) = 91) : 
  ∀ y ∈ Y, y ≤ 6 ∨ y ≥ 102 :=
by
  sorry

end set_Y_numbers_l422_422181


namespace max_min_values_l422_422213

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values : 
  (∀ x ∈ Icc (0 : ℝ) (3 : ℝ), f 2 ≤ f x ∧ f x ≤ f 0) ∧ 
  (f 2 = -15) ∧ 
  (f 0 = 5) := sorry

end max_min_values_l422_422213


namespace induction_step_term_l422_422984

theorem induction_step_term (k : ℕ) :
  (∑ i in Finset.range (2 * (k + 1) + 1), i) - (∑ i in Finset.range (2 * k + 1), i) = (2 * k + 1) + (2 * k + 2) :=
by
  sorry

end induction_step_term_l422_422984


namespace difference_place_values_l422_422988

def place_value (digit : Char) (position : String) : Real :=
  match digit, position with
  | '1', "hundreds" => 100
  | '1', "tenths" => 0.1
  | _, _ => 0 -- for any other cases (not required in this problem)

theorem difference_place_values :
  (place_value '1' "hundreds" - place_value '1' "tenths" = 99.9) :=
by
  sorry

end difference_place_values_l422_422988


namespace find_AP_l422_422120

-- Definitions of the geometrical constructs and their properties
variables (A B C D : Point)
variables (W X Y Z : Point)
variables (P : Point)

-- Given conditions
def is_square (A B C D : Point) (side_length : ℝ) : Prop := sorry
def is_rectangle (W X Y Z : Point) (ZY_length XY_length : ℝ) : Prop := sorry
def are_perpendicular (line1 line2 : Line) : Prop := sorry
def area_of_rectangle (W X Y Z : Point) : ℝ := 12 * 8
def shaded_area (shaded_rectangle : Rectangle) : ℝ := (area_of_rectangle W X Y Z) / 3

-- The main theorem we need to prove
theorem find_AP (A B C D W X Y Z P : Point) :
    is_square A B C D 8 →
    is_rectangle W X Y Z 12 8 →
    are_perpendicular (line_through A D) (line_through W X) →
    shaded_area (mk_rectangle P D C X) = (area_of_rectangle W X Y Z) / 3 →
    distance A P = 4 :=
sorry

end find_AP_l422_422120


namespace sampleCandy_l422_422836

-- Define the percentage of customers who are caught
def P_caught := 0.22

-- Define the percentage of customers who sample candy and are not caught
def P_notCaught_of_Sample (x : ℝ) : ℝ := 0.08 * x

-- Using the condition that the sum of caught and not caught equals the total percentage
theorem sampleCandy (x : ℝ) (h₁ : P_caught = 0.22) (h₂ : P_notCaught_of_Sample x + P_caught = x) :
  x = 23.91 / 100 :=
by
  sorry

end sampleCandy_l422_422836


namespace f_2_eq_4_f_half_eq_quarter_f_f_neg1_eq_1_f_a_eq_3_l422_422070

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2 * x

theorem f_2_eq_4 : f(2) = 4 := sorry

theorem f_half_eq_quarter : f(1/2) = 1/4 := sorry

theorem f_f_neg1_eq_1 : f(f(-1)) = 1 := sorry

theorem f_a_eq_3 (a : ℝ) : f(a) = 3 → (a = sqrt 3 ∨ a = 3 / 2) := sorry

end f_2_eq_4_f_half_eq_quarter_f_f_neg1_eq_1_f_a_eq_3_l422_422070


namespace find_c_for_parallel_and_intersecting_lines_l422_422003

theorem find_c_for_parallel_and_intersecting_lines :
  ∃ (c : ℝ) (d : ℝ), (3 * 2 - 4 * (-3) = c) ∧ (6 * 2 + d * (-3) = 2 * c) ∧ (6 * d = -24) ∧ (d = -8) ∧ (c = 18) :=
begin
  use 18, 
  use -8,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { apply eq.symm,
    norm_num },
  split,
  { refl },
  { refl }
end

end find_c_for_parallel_and_intersecting_lines_l422_422003


namespace correct_statements_l422_422052

-- Define the conditions: X ~ B(n, 1/2)
def binomial_distribution (n : ℕ) := true -- placeholder definition for binomial distribution

-- Define the mean and variance of X ~ B(n, 1/2)
def expected_value (n : ℕ) := n / 2
def variance (n : ℕ) := n / 4

-- Formalize the correct statements
theorem correct_statements (n : ℕ) (h_binom : binomial_distribution n) :
  (n = 6 → ¬ (P (X ≤ 1) = 3 / 32)) ∧
  (expected_value n + 2 * variance n = n) ∧
  (n = 11 → (∃ k, (k = 5 ∨ k = 6) ∧ (P (X = k) is maximized))) ∧
  (0.96 ≤ 1 - (25 / n) → n ≥ 625) :=
by {
  sorry
}

end correct_statements_l422_422052


namespace not_divisible_by_4_iff_exists_squares_sum_divisible_l422_422509

theorem not_divisible_by_4_iff_exists_squares_sum_divisible (n : ℕ) (hn : 0 < n) :
  (¬ 4 ∣ n) ↔ ∃ a b : ℤ, n ∣ (a^2 + b^2 + 1) := 
by
  sorry

end not_divisible_by_4_iff_exists_squares_sum_divisible_l422_422509


namespace students_in_each_normal_class_l422_422445

theorem students_in_each_normal_class
  (total_students : ℕ)
  (percentage_moving : ℝ)
  (grades : ℕ)
  (adv_class_size : ℕ)
  (num_normal_classes : ℕ)
  (h1 : total_students = 1590)
  (h2 : percentage_moving = 0.4)
  (h3 : grades = 3)
  (h4 : adv_class_size = 20)
  (h5 : num_normal_classes = 6) :
  ((total_students * percentage_moving).toNat / grades - adv_class_size) / num_normal_classes = 32 := 
by sorry

end students_in_each_normal_class_l422_422445


namespace min_value_of_reciprocal_sum_l422_422775

theorem min_value_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) :
  ∃ z, (z = 3 + 2 * Real.sqrt 2) ∧ (∀ z', (z' = 1 / x + 1 / y) → z ≤ z') :=
sorry

end min_value_of_reciprocal_sum_l422_422775


namespace monotonically_increasing_interval_l422_422074

def f (x : ℝ) : ℝ := Real.sin ((1/2) * x - (Real.pi / 3))

def g (x : ℝ) : ℝ := Real.sin (x - (7 * Real.pi / 12))

theorem monotonically_increasing_interval :
  ∀ x, (x ∈ Set.Icc (Real.pi / 12) (13 * Real.pi / 12)) →
  (Real.cos (x - (7 * Real.pi / 12)) > 0) :=
sorry

end monotonically_increasing_interval_l422_422074


namespace probability_equals_0_647_l422_422310

noncomputable def rectangle_length : ℝ := 12
noncomputable def rectangle_width : ℝ := 10
noncomputable def circle_radius : ℝ := 1
noncomputable def small_rectangle_length : ℝ := 4
noncomputable def small_rectangle_width : ℝ := 2
noncomputable def sheet_diameter : ℝ := 1.5
noncomputable def allowable_length : ℝ := rectangle_length - sheet_diameter
noncomputable def allowable_width : ℝ := rectangle_width - sheet_diameter
noncomputable def allowable_area : ℝ := allowable_length * allowable_width
noncomputable def white_circle_area_affected : ℝ := 4 * (π * (circle_radius + sheet_diameter / 2)^2)
noncomputable def white_rectangle_area : ℕ := 1
noncomputable def expanded_white_area : ℝ := white_rectangle_area * (small_rectangle_length + sheet_diameter) * (small_rectangle_width + sheet_diameter)
noncomputable def total_white_area_affected : ℝ := white_circle_area_affected + expanded_white_area
noncomputable def probability_cover_white_area : ℝ := total_white_area_affected / allowable_area

theorem probability_equals_0_647 :
  probability_cover_white_area = 0.647 := by
  sorry

end probability_equals_0_647_l422_422310


namespace sum_of_digits_nine_ab_l422_422128

noncomputable def sum_digits_base_10 (n : ℕ) : ℕ :=
-- Function to compute the sum of digits of a number in base 10
sorry

def a : ℕ := 6 * ((10^1500 - 1) / 9)

def b : ℕ := 3 * ((10^1500 - 1) / 9)

def nine_ab : ℕ := 9 * a * b

theorem sum_of_digits_nine_ab :
  sum_digits_base_10 nine_ab = 13501 :=
sorry

end sum_of_digits_nine_ab_l422_422128


namespace sqrt_sum_4_pow_4_eq_32_l422_422606

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l422_422606


namespace least_number_to_multiply_l422_422210

theorem least_number_to_multiply (x : ℕ) :
  (72 * x) % 112 = 0 → x = 14 :=
by 
  sorry

end least_number_to_multiply_l422_422210


namespace family_of_subsets_l422_422011

theorem family_of_subsets (n : ℕ) (𝓕 : set (set (fin n))) :
    (∀ X : set (fin n), X ≠ ∅ → (set.count (λ A, A ∈ 𝓕 ∧ (A ∩ X).card.even) 𝓕 = 𝓕.count / 2)) →
    𝓕 = ∅ ∨ 𝓕 = set.univ :=
by 
    sorry

end family_of_subsets_l422_422011


namespace area_TQSR_is_four_l422_422642

-- Define the dimensions of the rectangle and the point T.
def PQRS : Type := {P Q R S T : ℝ × ℝ // 
  P = (0, 3) ∧ 
  Q = (6, 3) ∧ 
  R = (6, 0) ∧ 
  S = (0, 0) ∧ 
  T = (2, 3) ∧ 
  P.1 - R.1 = 6 ∧ 
  P.2 - R.2 = 3 ∧ 
  (R.2 + 2) = 3
}

-- Define that TQSR forms a trapezoid.
def is_trapezoid (Q R S T : ℝ × ℝ) : Prop :=
  Q.2 = T.2 ∧ S.2 = R.2 ∧ Q.1 = R.1 ∧ (Q.1 - T.1) = 4

-- Define the area of a trapezoid.
def trapezoid_area (b1 b2 h : ℝ) : ℝ := (b1 + b2) * h / 2

-- Prove the area of trapezoid TQSR is 4 square units.
theorem area_TQSR_is_four (r : PQRS) : 
  is_trapezoid r.val.Q r.val.R r.val.S r.val.T →
  trapezoid_area 6 2 1 = 4 :=
  by
    sorry

end area_TQSR_is_four_l422_422642


namespace ab_squared_non_positive_l422_422816

theorem ab_squared_non_positive (a b : ℝ) (h : 7 * a + 9 * |b| = 0) : a * b^2 ≤ 0 :=
sorry

end ab_squared_non_positive_l422_422816


namespace compute_polynomial_l422_422227

-- Definitions for the given conditions
def allowed_operations (u v : ℤ) : ℤ := u * v + v

-- Main theorem: for any polynomial with integer coefficients, the value can be computed using the given computer operations
theorem compute_polynomial (c : ℤ) (P : ℤ[X]) :
    ∃ (steps : ℕ) (vals : vector ℤ (steps + 3)),
      (vals.head ∈ {c, 1, -1}) ∧
      (∀ i < steps, vals.nth i = allowed_operations (vals.nth i) (vals.nth (i % 3))) ∧ 
      (P.eval c = vals.last) :=
    sorry

end compute_polynomial_l422_422227


namespace probability_ab_lt_zero_l422_422519

def A := {0, 1, -3, 6, -8, -10, 5, 12, -13}
def B := {-1, 2, -4, 7, 6, -9, 8, -11, 10}

theorem probability_ab_lt_zero :
  let positive_count_A := Finset.card (Finset.filter (λ x, 0 < x) A) in
  let negative_count_A := Finset.card (Finset.filter (λ x, x < 0) A) in
  let positive_count_B := Finset.card (Finset.filter (λ x, 0 < x) B) in
  let negative_count_B := Finset.card (Finset.filter (λ x, x < 0) B) in
  let zero_count_A := Finset.card (Finset.filter (λ x, x = 0) A) in
  let valid_elements_A := Finset.card A - zero_count_A in
  let total_possible_products := valid_elements_A * Finset.card B in
  let positive_A_negative_B := positive_count_A * negative_count_B in
  let negative_A_positive_B := negative_count_A * positive_count_B in
  let negative_products := positive_A_negative_B + negative_A_positive_B in
  (negative_products : ℚ) / total_possible_products = 4 / 9 :=
by sorry

end probability_ab_lt_zero_l422_422519


namespace cos_plus_pi_half_odd_l422_422725

theorem cos_plus_pi_half_odd (x : ℝ) : cos (x + (π / 2)) = -cos (x + (π / 2)) := 
by
  sorry

end cos_plus_pi_half_odd_l422_422725


namespace sum_coefficients_l422_422552

theorem sum_coefficients (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f(x + 3) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f(x) = a * x^2 + b * x + c) →
  a + b + c = 3 :=
by
  sorry

end sum_coefficients_l422_422552


namespace juice_difference_is_eight_l422_422946

-- Defining the initial conditions
def initial_large_barrel : ℕ := 10
def initial_small_barrel : ℕ := 8
def poured_juice : ℕ := 3

-- Defining the final amounts
def final_large_barrel : ℕ := initial_large_barrel + poured_juice
def final_small_barrel : ℕ := initial_small_barrel - poured_juice

-- The statement we need to prove
theorem juice_difference_is_eight :
  final_large_barrel - final_small_barrel = 8 :=
by
  -- Skipping the proof
  sorry

end juice_difference_is_eight_l422_422946


namespace factorize_expression_1_factorize_expression_2_l422_422900

theorem factorize_expression_1 (m : ℤ) : 
  m^3 - 2 * m^2 - 4 * m + 8 = (m - 2)^2 * (m + 2) := 
sorry

theorem factorize_expression_2 (x y : ℤ) : 
  x^2 - 2 * x * y + y^2 - 9 = (x - y + 3) * (x - y - 3) :=
sorry

end factorize_expression_1_factorize_expression_2_l422_422900


namespace compute_expression_l422_422888

noncomputable def f (x : ℝ) := x^2 + 1
noncomputable def g (x : ℝ) := 2^x
noncomputable def f_inv (y : ℝ) := Real.sqrt (y - 1)
noncomputable def g_inv (y : ℝ) := Real.log y / Real.log 2

theorem compute_expression : f (g_inv (f_inv (f_inv (g (f 3))))) = 7.25 :=
by
  sorry

end compute_expression_l422_422888


namespace intersect_at_four_points_l422_422354

theorem intersect_at_four_points (a : ℝ) : 
  (∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = a^2) ∧ (p.2 = p.1^2 - a - 1) ∧ 
                 ∃ q : ℝ × ℝ, (q.1 ≠ p.1 ∧ q.2 ≠ p.2) ∧ (q.1^2 + q.2^2 = a^2) ∧ (q.2 = q.1^2 - a - 1) ∧ 
                 ∃ r : ℝ × ℝ, (r.1 ≠ p.1 ∧ r.1 ≠ q.1 ∧ r.2 ≠ p.2 ∧ r.2 ≠ q.2) ∧ (r.1^2 + r.2^2 = a^2) ∧ (r.2 = r.1^2 - a - 1) ∧
                 ∃ s : ℝ × ℝ, (s.1 ≠ p.1 ∧ s.1 ≠ q.1 ∧ s.1 ≠ r.1 ∧ s.2 ≠ p.2 ∧ s.2 ≠ q.2 ∧ s.2 ≠ r.2) ∧ (s.1^2 + s.2^2 = a^2) ∧ (s.2 = s.1^2 - a - 1))
  ↔ a > -1/2 := 
by 
  sorry

end intersect_at_four_points_l422_422354


namespace find_a_l422_422565

theorem find_a (a : ℕ) (h₁ : a ≠ 0) (h₂ : (a : ℚ) / (a + 37) = 0.875) : a = 259 :=
by
sorrry

end find_a_l422_422565


namespace trader_profit_percentage_l422_422256

theorem trader_profit_percentage
  (P : ℝ)
  (h1 : P > 0)
  (buy_price : ℝ := 0.80 * P)
  (sell_price : ℝ := 1.60 * P) :
  (sell_price - P) / P * 100 = 60 := 
by sorry

end trader_profit_percentage_l422_422256


namespace angle_incenter_triangle_l422_422137

-- Define the type for angles
structure Triangle :=
  (A B C : Type)

-- Define the type for points
structure Point :=
  (x y z : Type)

-- Define the problem conditions and theorem in Lean
theorem angle_incenter_triangle 
  (ABC : Triangle)
  (H I_b I_c L : Point)
  (altitude : H → ABC)
  (incenter_ABH : I_b → ABC)
  (incenter_ACH : I_c → ABC)
  (incircle_touch : L → ABC)
  : ∠ LI_b I_c = 90 :=
  sorry

end angle_incenter_triangle_l422_422137


namespace barbed_wire_cost_l422_422555

theorem barbed_wire_cost
  (A : ℕ)          -- Area of the square field (sq m)
  (cost_per_meter : ℕ)  -- Cost per meter for the barbed wire (Rs)
  (gate_width : ℕ)      -- Width of each gate (m)
  (num_gates : ℕ)       -- Number of gates
  (side_length : ℕ)     -- Side length of the square field (m)
  (perimeter : ℕ)       -- Perimeter of the square field (m)
  (total_length : ℕ)    -- Total length of the barbed wire needed (m)
  (total_cost : ℕ)      -- Total cost of drawing the barbed wire (Rs)
  (h1 : A = 3136)       -- Given: Area = 3136 sq m
  (h2 : cost_per_meter = 1)  -- Given: Cost per meter = 1 Rs/m
  (h3 : gate_width = 1)      -- Given: Width of each gate = 1 m
  (h4 : num_gates = 2)       -- Given: Number of gates = 2
  (h5 : side_length * side_length = A)  -- Side length calculated from the area
  (h6 : perimeter = 4 * side_length)    -- Perimeter of the square field
  (h7 : total_length = perimeter - (num_gates * gate_width))  -- Actual barbed wire length after gates
  (h8 : total_cost = total_length * cost_per_meter)           -- Total cost calculation
  : total_cost = 222 :=      -- The result we need to prove
sorry

end barbed_wire_cost_l422_422555


namespace percentage_of_class_taking_lunch_l422_422218

theorem percentage_of_class_taking_lunch 
  (total_students : ℕ)
  (boys_ratio : ℕ := 6)
  (girls_ratio : ℕ := 4)
  (boys_percentage_lunch : ℝ := 0.60)
  (girls_percentage_lunch : ℝ := 0.40) :
  total_students = 100 →
  (6 / (6 + 4) * 100) = 60 →
  (4 / (6 + 4) * 100) = 40 →
  (boys_percentage_lunch * 60 + girls_percentage_lunch * 40) = 52 →
  ℝ :=
    by
      intros
      sorry

end percentage_of_class_taking_lunch_l422_422218


namespace divisibility_of_n_l422_422890

def n : ℕ := (2^4 - 1) * (3^6 - 1) * (5^10 - 1) * (7^12 - 1)

theorem divisibility_of_n : 
    (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) := 
by 
  sorry

end divisibility_of_n_l422_422890


namespace sum_of_m_values_l422_422940

theorem sum_of_m_values : 
  (∑ p in ({(1, 15), (-1, -15), (3, 5), (-3, -5), (15, 1), (-15, -1), (5, 3), (-5, -3)} : Finset (ℤ × ℤ)), (p.1 + p.2)) = 48 :=
by 
  -- Proof omitted
  sorry

end sum_of_m_values_l422_422940


namespace verify_equation_holds_l422_422253

noncomputable def verify_equation (m n : ℝ) : Prop :=
  1.55 * Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2)) 
  - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2)) 
  = 2 * Real.sqrt (3 * m - n)

theorem verify_equation_holds (m n : ℝ) (h : 9 * m^2 - n^2 ≥ 0) : verify_equation m n :=
by
  -- Proof goes here. 
  -- Implement the proof as per the solution steps sketched in the problem statement.
  sorry

end verify_equation_holds_l422_422253


namespace ratio_CB_CA_of_right_triangle_is_4_3_l422_422022

-- Definitions for the conditions
variables {A B C X O M O_prime : Point}

-- Right triangle with A at the origin and C at (1,1)
def is_right_triangle (A C B : Point) : Prop :=
  (angle A C B = 90)

-- Distance between points
def dist (p q : Point) : Real :=
  sqrt ((p.x - q.x) ^ 2 + (p.y - q.y) ^ 2)

-- Line segment division into equal parts
def divides_equally (A B O : Point) : Prop :=
  dist A O = dist O B

-- The incenter of the triangle
def incenter (A C B X : Point) : Prop :=
  angle A X B = 135

-- Prove the ratio
theorem ratio_CB_CA_of_right_triangle_is_4_3 
(hyp_tr : is_right_triangle A C B)
(mid_pt_div : divides_equally A B O)
(angle_right : angle (midpoint A B) X C = 90)
: dist C B / dist C A = 4 / 3 := sorry

end ratio_CB_CA_of_right_triangle_is_4_3_l422_422022


namespace factor_difference_of_squares_l422_422577

theorem factor_difference_of_squares (a : ℝ) : a^2 - 16 = (a - 4) * (a + 4) := 
sorry

end factor_difference_of_squares_l422_422577


namespace exists_row_or_column_with_same_pluses_l422_422397

-- Setting up the problem conditions
variables {n : ℕ} (table : ℕ → ℕ → bool)  -- Assume: true represents plus, false represents minus

-- Conditions: The table is 2n x 2n, the number of pluses equals the number of minuses
def squareTable (table : ℕ → ℕ → bool) (n : ℕ) := 
  ∀ i j, 0 ≤ i < 2 * n ∧ 0 ≤ j < 2 * n 

def equalNumPlusesMinuses (table : ℕ → ℕ → bool) (n : ℕ) :=
  (∀ i j, 0 ≤ i < 2 * n ∧ 0 ≤ j < 2 * n) → 
  ∑ i in Finset.range (2 * n), ∑ j in Finset.range (2 * n), if table i j then 1 else 0 = 2 * n^2 ∧
  ∑ i in Finset.range (2 * n), ∑ j in Finset.range (2 * n), if ¬ table i j then 1 else 0 = 2 * n^2

-- Theorem: Prove that there exists either a pair of rows or a pair of columns with the same number of pluses
theorem exists_row_or_column_with_same_pluses (table: ℕ → ℕ → bool) (n : ℕ)
  (h1 : squareTable table n)
  (h2 : equalNumPlusesMinuses table n) :
  ∃ i j, i ≠ j ∧ (∀ k, table i k = table j k) ∨ (∀ k, table k i = table k j) :=
sorry

end exists_row_or_column_with_same_pluses_l422_422397


namespace empirical_regression_probability_A_champion_l422_422715

-- Definition of given data for regression problem
noncomputable def data_x : List ℕ := [1, 2, 3, 4, 5, 6]
noncomputable def data_y : List ℕ := [50, 78, 124, 121, 137, 352]
noncomputable def data_u : List ℤ := [ln 50, ln 78, ln 124, ln 121, ln 137, ln 352]
noncomputable def sum_u : ℤ := 28.5
noncomputable def sum_xu : ℤ := 106.05

-- Empirical regression calculation
theorem empirical_regression :
  let x_avg := (1 + 2 + 3 + 4 + 5 + 6) / 6
      u_avg := 28.5 / 6
      b := (∑ i in List.range 6, data_x.get i * data_u.get i - 6 * x_avg * u_avg) / ((∑ i in List.range 6, (data_x.get i)^2) - 6 * x_avg^2)
      intercept := u_avg - b * x_avg
      a := Real.exp intercept
  in y = a * Real.exp(b * x) :=
sorry

-- Definition of given probabilities for competition problem
def P_A_wins_against_B : ℚ := 1/2
def P_A_wins_against_C : ℚ := 1/3
def P_B_wins_against_C : ℚ := 3/5

-- Probability calculation for company A becoming the champion
theorem probability_A_champion :
  let P := 
      P_A_wins_against_B * P_A_wins_against_C + 
      P_A_wins_against_B * (1 - P_A_wins_against_C) * P_B_wins_against_C * P_A_wins_against_B +
      (1 - P_A_wins_against_B) * (1 - P_B_wins_against_C) * P_A_wins_against_C * P_A_wins_against_B
  in P = 3/10 :=
sorry

end empirical_regression_probability_A_champion_l422_422715


namespace binomial_coeff_sum_l422_422220

open Nat BigOperators

noncomputable def sum_of_coefficients : ℚ :=
  let binomial := (1 / x - x / 2) ^ 6
  test_coeff := eval 1 binomial
  test_coeff

noncomputable def coeff_of_x_squared : ℚ :=
  let general_term r := (nat.choose 6 r) * (- (1 / 2)) ^ r * x ^ (2 * r - 6)
  let r := 4
  nat.choose 6 r * (1 / 16)

theorem binomial_coeff_sum :
  sum_of_coefficients = (1 / 64) ∧ coeff_of_x_squared = (15 / 16) :=
by
  sorry

end binomial_coeff_sum_l422_422220


namespace solve_x_l422_422548

theorem solve_x :
  ∀ x : ℚ, 5 * x - 3 * x = 420 - 10 * (x + 2) ↔ x = 100 / 3 :=
by
  intro x
  split
  sorry

end solve_x_l422_422548


namespace Annika_three_times_Hans_in_future_l422_422850

theorem Annika_three_times_Hans_in_future
  (hans_age_now : Nat)
  (annika_age_now : Nat)
  (x : Nat)
  (hans_future_age : Nat)
  (annika_future_age : Nat)
  (H1 : hans_age_now = 8)
  (H2 : annika_age_now = 32)
  (H3 : hans_future_age = hans_age_now + x)
  (H4 : annika_future_age = annika_age_now + x)
  (H5 : annika_future_age = 3 * hans_future_age) :
  x = 4 := 
  by
  sorry

end Annika_three_times_Hans_in_future_l422_422850


namespace prove_a_le_minus_2sqrt2_l422_422457

theorem prove_a_le_minus_2sqrt2 
  (f : ℝ → ℝ)
  (a x1 x2 : ℝ)
  (h1 : f = (λ x, Real.log x + (1/2) * x^2 + a * x))
  (h2 : f' = (λ x, (1 / x) + x + a))
  (h3 : f' x1 = 0 ∧ f' x2 = 0)
  (h4 : Real.log x1 + (1/2) * x1^2 + a * x1
          + Real.log x2 + (1/2) * x2^2 + a * x2 ≤ -5)
  (h5 : x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ x1 * x2 = 1 ∧ -a > 0) :
  a ≤ -2 * Real.sqrt 2 := 
sorry

end prove_a_le_minus_2sqrt2_l422_422457


namespace max_sin_sum_l422_422044

noncomputable theory

open Real

theorem max_sin_sum (n : ℕ) (a : ℝ)
  (h1 : 0 ≤ a) (h2 : a ≤ n)
  (h3 : ∃ (x : ℕ → ℝ), (∀i, 1 ≤ i ∧ i ≤ n → sin^2 (x i) = a) ) :
  ∃ x : ℕ → ℝ, (∀ i, (1 ≤ i ∧ i ≤ n) → sin^2 (x i) = a) ∧
    (∑ i in range n, sin (2 * x i)).abs = 2 * real.sqrt (a * (n - a)) :=
sorry

end max_sin_sum_l422_422044


namespace bisect_angle_BFC_l422_422591

open EuclideanGeometry

-- Definitions for the problem context
variables {k : circle} {A B C D E F : point}
variables (hA_out : A ∉ k) 
          (h_secant : Secant.line_through A k) 
          (h_tangentD : Tangent.line_through A k D) 
          (h_tangentE : Tangent.line_through A k E)
          (h_intersectBC : Secant.intersect_points k A B C)
          (h_midpointF : midpoint D E F)

-- The main theorem to prove
theorem bisect_angle_BFC : is_angle_bisector DE (∠ B F C) :=
by
  sorry

end bisect_angle_BFC_l422_422591


namespace max_digit_sum_in_24_hour_format_l422_422661

theorem max_digit_sum_in_24_hour_format : 
  ∃ t : ℕ × ℕ, (0 ≤ t.fst ∧ t.fst < 24 ∧ 0 ≤ t.snd ∧ t.snd < 60 ∧ (t.fst / 10 + t.fst % 10 + t.snd / 10 + t.snd % 10 = 24)) :=
sorry

end max_digit_sum_in_24_hour_format_l422_422661


namespace probability_of_drawing_red_ball_from_5red_2white_3yellow_fairness_of_game_for_4red_6white_probability_of_drawing_white_ball_after_one_removed_l422_422226

section

-- Problem (1)
theorem probability_of_drawing_red_ball_from_5red_2white_3yellow
  (red white yellow : ℕ)
  (h_red : red = 5) (h_white : white = 2) (h_yellow : yellow = 3) :
  (red + white + yellow = 10) → 
  (red / (red + white + yellow) = 1 / 2) :=
by
  sorry

-- Problem (2)
theorem fairness_of_game_for_4red_6white
  (red white : ℕ)
  (h_red : red = 4) (h_white : white = 6) :
  (red + white = 10) → 
  ((red / (red + white) ≠ white / (red + white)) → ¬ fair) :=
by
  sorry

-- Problem (3)
theorem probability_of_drawing_white_ball_after_one_removed
  (initial_red initial_white remaining_white : ℕ)
  (h_initial_red : initial_red = 4) 
  (h_initial_white : initial_white = 6) 
  (remaining_white = initial_white - 1) :
  (initial_red + initial_white - 1 = 9) → 
  (remaining_white / (initial_red + initial_white - 1) = 5 / 9) :=
by
  sorry

end

end probability_of_drawing_red_ball_from_5red_2white_3yellow_fairness_of_game_for_4red_6white_probability_of_drawing_white_ball_after_one_removed_l422_422226


namespace fraction_of_girls_l422_422838

theorem fraction_of_girls (G T B : ℕ) (Fraction : ℚ)
  (h1 : Fraction * G = (1/3 : ℚ) * T)
  (h2 : (B : ℚ) / G = 1/2) :
  Fraction = 1/2 := by
  sorry

end fraction_of_girls_l422_422838


namespace ganpat_paint_time_l422_422811

theorem ganpat_paint_time (H_rate G_rate : ℝ) (together_time H_time : ℝ) (h₁ : H_time = 3)
  (h₂ : together_time = 2) (h₃ : H_rate = 1 / H_time) (h₄ : G_rate = 1 / G_time)
  (h₅ : 1/H_time + 1/G_rate = 1/together_time) : G_time = 3 := 
by 
  sorry

end ganpat_paint_time_l422_422811


namespace last_number_is_odd_l422_422567

/-- The integers from 1 to 2018 are written on the board.
 An operation consists of choosing any two of them, erasing them,
and writing the absolute value of their difference on the board. -/
theorem last_number_is_odd : 
    (∀ (S : Multiset ℕ), 
      (S = Multiset.range' 1 2018) → 
      (∀ (a b : ℕ), a ∈ S → b ∈ S → S.erase a.erase b.add (abs (a - b)) = S) → 
      S.card = 1 → S.head = (some odd S.head)) :=
by
  sorry

end last_number_is_odd_l422_422567


namespace limit_sequence_l422_422341

open Real

noncomputable def a_n (n : ℕ) : ℝ := (n * (n + 1) / 2) / (n - n^2 + 3)

theorem limit_sequence : 
  tendsto (λ (n : ℕ), a_n n) at_top (𝓝 (-1 / 2)) :=
by
  sorry

end limit_sequence_l422_422341


namespace raul_money_left_l422_422916

theorem raul_money_left (initial_amount comics_cost comics_count: ℕ) (h1: initial_amount = 87) (h2: comics_cost = 4) (h3: comics_count = 8):
  initial_amount - (comics_cost * comics_count) = 55 :=
by
  rw [h1, h2, h3]
  norm_num

end raul_money_left_l422_422916


namespace geometric_sequence_common_ratio_l422_422671

theorem geometric_sequence_common_ratio 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 25) 
  (h2 : a2 = -50) 
  (h3 : a3 = 100) 
  (h4 : a4 = -200)
  (h_geometric : a2 / a1 = a3 / a2 ∧ a3 / a2 = a4 / a3) : 
  a2 / a1 = -2 :=
by 
  have r1 : a2 / a1 = -2, sorry
  -- additional steps to complete proof here
  exact r1

end geometric_sequence_common_ratio_l422_422671


namespace sum_of_valid_z_values_l422_422726

variable (z : ℕ)

-- Define the conditions from the problem statement.
def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0
def single_digit (z : ℕ) : Prop := z < 10

-- Main statement to prove
theorem sum_of_valid_z_values :
  (∀ z, single_digit z → is_divisible_by_6 (36 * 1000 + z * 100 + 52) → z ∈ {2, 5, 8}) →
  (2 + 5 + 8) = 15 :=
by
  sorry

end sum_of_valid_z_values_l422_422726


namespace ratio_of_areas_of_concentric_circles_l422_422978

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end ratio_of_areas_of_concentric_circles_l422_422978


namespace stickers_distribution_l422_422083

theorem stickers_distribution (stickers sheets : ℕ) (h_stickers : stickers = 10) (h_sheets: sheets = 4) :
  ∃ d : ℕ, d = 34 ∧ 
  (number_of_ways_to_distribute_stickers stickers sheets d) where
  number_of_ways_to_distribute_stickers : ℕ → ℕ → ℕ → Prop := sorry

end stickers_distribution_l422_422083


namespace impossible_to_achieve_target_l422_422581

def initial_matchsticks := (1, 0, 0, 0)  -- Initial matchsticks at vertices (A, B, C, D)
def target_matchsticks := (1, 9, 8, 9)   -- Target matchsticks at vertices (A, B, C, D)

def S (a1 a2 a3 a4 : ℕ) : ℤ := a1 - a2 + a3 - a4

theorem impossible_to_achieve_target : 
  ¬∃ (f : ℕ × ℕ × ℕ × ℕ → ℕ × ℕ × ℕ × ℕ), 
    (f initial_matchsticks = target_matchsticks) ∧ 
    (∀ (a1 a2 a3 a4 : ℕ) k, 
      f (a1, a2, a3, a4) = (a1 - k, a2 + k, a3, a4 + k) ∨ 
      f (a1, a2, a3, a4) = (a1, a2 - k, a3 + k, a4 + k) ∨ 
      f (a1, a2, a3, a4) = (a1 + k, a2 - k, a3 - k, a4) ∨ 
      f (a1, a2, a3, a4) = (a1 - k, a2, a3 + k, a4 - k)) := sorry

end impossible_to_achieve_target_l422_422581


namespace prove_a_leq_neg_2sqrt2_l422_422459

def f (x a : ℝ) : ℝ := Real.log x + (1 / 2) * x^2 + a * x

noncomputable def extreme_points (a : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), x1^2 + a * x1 + 1 = 0 ∧ x2^2 + a * x2 + 1 = 0 ∧ x1 ≠ x2

theorem prove_a_leq_neg_2sqrt2 (a : ℝ) (h_extrema : extreme_points a) (h_sum : f h_extrema.fst a + f h_extrema.snd a ≤ -5) : 
  a ≤ -2 * Real.sqrt 2 := sorry

end prove_a_leq_neg_2sqrt2_l422_422459


namespace squares_total_l422_422006

def number_of_squares (figure : Type) : ℕ := sorry

theorem squares_total (figure : Type) : number_of_squares figure = 38 := sorry

end squares_total_l422_422006


namespace cake_eaten_after_four_trips_l422_422251

-- Define the fraction of the cake eaten on each trip
def fraction_eaten (n : Nat) : ℚ :=
  (1 / 3) ^ n

-- Define the total cake eaten after four trips
def total_eaten_after_four_trips : ℚ :=
  fraction_eaten 1 + fraction_eaten 2 + fraction_eaten 3 + fraction_eaten 4

-- The mathematical statement we want to prove
theorem cake_eaten_after_four_trips : total_eaten_after_four_trips = 40 / 81 := 
by
  sorry

end cake_eaten_after_four_trips_l422_422251


namespace range_of_k_l422_422404

open Real

theorem range_of_k (k : ℝ) :
  (∃ x1 x2 ∈ Icc 1 2, abs (exp x1 - exp x2) > k * abs (log x1 - log x2)) ↔ k < 2 * exp 2 := 
sorry

end range_of_k_l422_422404


namespace find_p_l422_422103

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def hyperbola_focus : ℝ × ℝ :=
  (2, 0)

theorem find_p (p : ℝ) (h : p > 0) (hp : parabola_focus p = hyperbola_focus) : p = 4 :=
by
  sorry

end find_p_l422_422103


namespace raul_money_left_l422_422915

theorem raul_money_left (initial_amount comics_cost comics_count: ℕ) (h1: initial_amount = 87) (h2: comics_cost = 4) (h3: comics_count = 8):
  initial_amount - (comics_cost * comics_count) = 55 :=
by
  rw [h1, h2, h3]
  norm_num

end raul_money_left_l422_422915


namespace optimal_strategy_l422_422296

-- Define the players
structure Players where
  F : Type -- Father
  M : Type -- Mother
  S : Type -- Son
  prob : Players → Players → ℝ -- The probability function

-- Given conditions
axiom weakest_player : ∀ (P : Players), P.F < P.M
axiom strongest_player : ∀ (P : Players), P.S > P.M ∧ P.S > P.F
axiom no_ties : ∀ (A B : Players), (A ≠ B) → (P.prob A B ≠ 1/2)
axiom winner_plays_next : ∀ (A B C : Players), -- self-explanatory
axiom championship_condition : ∀ (A : Players), -- another axiom related to victory condition
axiom father_chooses_first : ∃ (A B : Players), A = P.F ∨ B = P.F
axiom constant_probability : ∀ (A B : Players), P.prob A B = P.prob A B

-- Theorem: Father's optimal strategy
theorem optimal_strategy (P: Players) : 
  P.prob P.F P.S < P.prob P.M P.S ∧ P.prob P.F P.M < P.prob P.S P.M →
  P.prob P.F P.M > P.prob P.F P.S 
  := by 
    sorry

end optimal_strategy_l422_422296


namespace sqrt_sum_4_pow_4_eq_32_l422_422607

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l422_422607


namespace sum_of_c_for_eight_solutions_l422_422190

def g (x : ℝ) : ℝ := ((x - 6) * (x - 4) * (x - 2) * x * (x + 2) * (x + 4) * (x + 6)) / 1005 - 2.5

theorem sum_of_c_for_eight_solutions :
  (∑ c in {c : ℤ | ∃ (x : ℝ) (hx : g x = c), (∃ unique (y₁ y₂ : ℝ) (h₁ : g y₁ = c) (h₂ : g y₂ = c), y₁ ≠ y₂)}, (c : ℤ)) = -5 := 
  sorry

end sum_of_c_for_eight_solutions_l422_422190


namespace min_sqrt_eq_sum_sqrt_implies_param_l422_422012

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem min_sqrt_eq_sum_sqrt_implies_param (a b c : ℝ) (r s t : ℝ)
    (h1 : 0 < a ∧ a ≤ 1)
    (h2 : 0 < b ∧ b ≤ 1)
    (h3 : 0 < c ∧ c ≤ 1)
    (h4 : min (sqrt ((a * b + 1) / (a * b * c))) (min (sqrt ((b * c + 1) / (a * b * c))) (sqrt ((a * c + 1) / (a * b * c)))) 
          = (sqrt ((1 - a) / a) + sqrt ((1 - b) / b) + sqrt ((1 - c) / c))) :
    ∃ r, a = 1 / (1 + r^2) ∧ b = 1 / (1 + (1 / r^2)) ∧ c = (r + 1 / r)^2 / (1 + (r + 1 / r)^2) :=
sorry

end min_sqrt_eq_sum_sqrt_implies_param_l422_422012


namespace jewelry_showroom_payment_l422_422681

noncomputable def total_cost_to_restock (necklaces_needed rings_needed bangles_needed : ℕ) : ℝ :=
  let necklace_cost := necklaces_needed * 5
  let discounted_necklace_cost := if necklaces_needed >= 15 then necklace_cost * 0.85
                                 else if necklaces_needed >= 10 then necklace_cost * 0.90
                                 else necklace_cost
  let ring_cost := rings_needed * 8
  let discounted_ring_cost := if rings_needed >= 30 then ring_cost * 0.93
                             else if rings_needed >= 20 then ring_cost * 0.95
                             else ring_cost
  let bangle_cost := bangles_needed * 6
  let discounted_bangle_cost := if bangles_needed >= 25 then bangle_cost * 0.90
                               else if bangles_needed >= 15 then bangle_cost * 0.92
                               else bangle_cost
  let total_cost := discounted_necklace_cost + discounted_ring_cost + discounted_bangle_cost
  total_cost * 1.02

theorem jewelry_showroom_payment : 
  total_cost_to_restock (20 - 8) (40 - 25) (30 - 17) = 257.04 :=
by
  sorry

end jewelry_showroom_payment_l422_422681


namespace sqrt_sum_of_four_terms_of_4_pow_4_l422_422613

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l422_422613


namespace factor_polynomial_l422_422362

theorem factor_polynomial :
  (x^2 + 5 * x + 4) * (x^2 + 11 * x + 30) + (x^2 + 8 * x - 10) =
  (x^2 + 8 * x + 7) * (x^2 + 8 * x + 19) := by
  sorry

end factor_polynomial_l422_422362


namespace fraction_to_decimal_l422_422713

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 :=
by
  sorry

end fraction_to_decimal_l422_422713


namespace find_n_l422_422018

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 :=
by
  sorry

end find_n_l422_422018


namespace starting_lineups_possible_l422_422339

theorem starting_lineups_possible :
  (C(12, 2) = 66) := by
  sorry

end starting_lineups_possible_l422_422339


namespace exists_unique_digit_count_l422_422625

noncomputable def number_of_digits_in_base (k : ℤ) (b : ℝ) : ℤ :=
  (⌊k / Real.log10 b⌋ + 1)

theorem exists_unique_digit_count (n : ℤ) (hn : n > 1) :
  ∃! k, (n = number_of_digits_in_base k (2 : ℝ)) ∨ (n = number_of_digits_in_base k (5 : ℝ)) := 
sorry

end exists_unique_digit_count_l422_422625


namespace common_ratio_of_geometric_seq_l422_422677

theorem common_ratio_of_geometric_seq (a b c d : ℤ) (h1 : a = 25)
    (h2 : b = -50) (h3 : c = 100) (h4 : d = -200)
    (h_geo_1 : b = a * -2)
    (h_geo_2 : c = b * -2)
    (h_geo_3 : d = c * -2) : 
    let r := (-2 : ℤ) in r = -2 := 
by 
  sorry

end common_ratio_of_geometric_seq_l422_422677


namespace students_in_each_normal_class_l422_422443

theorem students_in_each_normal_class
  (total_students : ℕ)
  (percentage_moving : ℝ)
  (grades : ℕ)
  (adv_class_size : ℕ)
  (num_normal_classes : ℕ)
  (h1 : total_students = 1590)
  (h2 : percentage_moving = 0.4)
  (h3 : grades = 3)
  (h4 : adv_class_size = 20)
  (h5 : num_normal_classes = 6) :
  ((total_students * percentage_moving).toNat / grades - adv_class_size) / num_normal_classes = 32 := 
by sorry

end students_in_each_normal_class_l422_422443


namespace vectors_perpendicular_l422_422032

theorem vectors_perpendicular : 
  ∀ (a b : ℝ × ℝ), a = (2, 0) → b = (1, 1) → 
  let c := (a.1 - b.1, a.2 - b.2) in 
  c.1 * b.1 + c.2 * b.2 = 0 :=
by
  intros a b ha hb
  rw [ha, hb]
  let c := (a.1 - b.1, a.2 - b.2)
  rw [sub_eq_add_neg, sub_eq_add_neg, sub_eq_add_neg, sub_eq_add_neg]

  -- We skip the actual proof here, just ensuring the theorem statement.
  sorry

end vectors_perpendicular_l422_422032


namespace bond_value_at_6_years_l422_422908

-- Given conditions as definitions in Lean 4
def principal := 200
def rate : ℚ := 0.25
def time1 : ℚ := 4
def value1 := 400
def time2 : ℚ := 2

-- Value of the bond at the end of 4 years
def value_after_4_years := principal + principal * rate * time1

-- Prove that the value of the bond at the end of 6 years is $500
theorem bond_value_at_6_years : 
  let interest_for_2_years := principal * rate * time2 in
  let value_after_6_years := value_after_4_years + interest_for_2_years in
  value_after_6_years = 500 := 
by
  sorry

end bond_value_at_6_years_l422_422908


namespace arithmetic_sequence_general_term_and_sum_l422_422772

theorem arithmetic_sequence_general_term_and_sum 
  (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a4 : a 4 = 8) (h_a3_a7 : a 3 + a 7 = 20) 
  (b : ℕ → ℝ) (h_b : ∀ n, b n = 1 / (a n * a (n + 1))) 
  (S : ℕ → ℝ) (h_sum_s : ∀ n, S n = ∑ k in Finset.range n, b k) :
  (∀ n, a n = 2 * n) ∧ (∀ n, S n = n / (4 * (n + 1))) :=
by 
  -- Proof will be provided here later.
  sorry

end arithmetic_sequence_general_term_and_sum_l422_422772


namespace num_isosceles_points_l422_422476

-- Definitions to handle coordinates on the grid
structure Point where
  x : ℤ
  y : ℤ

def point_D : Point := { x := 2, y := 2 }
def point_E : Point := { x := 5, y := 2 }

-- Definition of distance function on a grid
def distance (p1 p2 : Point) : ℕ :=
  abs (p2.x - p1.x) + abs (p2.y - p1.y)

-- Proving the number of points that make △DEF isosceles
theorem num_isosceles_points : 
  (finset.univ.filter (λ F : Point, distance point_D F = distance F point_E ∨ distance point_D F = distance point_D point_E ∨ distance F point_E = distance F point_E)).card = 2 := 
begin
  sorry
end

end num_isosceles_points_l422_422476


namespace max_area_and_length_l422_422284

def material_cost (x y : ℝ) : ℝ :=
  900 * x + 400 * y + 200 * x * y

def area (x y : ℝ) : ℝ := x * y

theorem max_area_and_length (x y : ℝ) (h₁ : material_cost x y ≤ 32000) :
  ∃ (S : ℝ) (x : ℝ), S = 100 ∧ x = 20 / 3 :=
sorry

end max_area_and_length_l422_422284


namespace dot_product_self_l422_422513

variable (v : ℝ × ℝ × ℝ) (h : v = (3, 4, 12))

theorem dot_product_self : (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) = 169 :=
by
  rw [h]
  calc
    (3 : ℝ) ^ 2 + (4 : ℝ) ^ 2 + (12 : ℝ) ^ 2 = 9 + 16 + 144 := by norm_num
    ...                              = 169 := by norm_num

end dot_product_self_l422_422513


namespace complex_conjugate_product_l422_422784

noncomputable def complex_number : ℂ := (√3 + complex.I) / (1 - √3 * complex.I)
def conjugate_z : ℂ := complex.conj complex_number

theorem complex_conjugate_product : complex_number * conjugate_z = 1 := by
  sorry

end complex_conjugate_product_l422_422784


namespace possible_values_of_b_l422_422099

theorem possible_values_of_b (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → x.im ≠ 0) ↔ (b ∈ set.Ioo (-8 : ℝ) 8) := 
by sorry

end possible_values_of_b_l422_422099


namespace limit_S_over_V_l422_422138

noncomputable theory

open Real

-- Define the curve C
def curve (x : ℝ) : ℝ := 1 / x

-- Define tangent line at A(1, 1)
def tangent_linear_equation (x y : ℝ) : Prop := x + y = 2

-- Define point P on the curve C
def point_P (t : ℝ) (ht : t > 0) : ℝ × ℝ := (t, 1 / t)

-- Define parallel line m passing through point P
def parallel_line_m (t x y : ℝ) (ht : t > 0) : Prop := y = -x + t + 1 / t

-- Define intersection point Q of line m and curve C
def point_Q (t : ℝ) (ht : t > 0) : ℝ × ℝ := (1 / t, t)

-- Define the area S
def area_S (t : ℝ) (ht : t > 0) : ℝ := 2 * abs (log t)

-- Define the volume V
def volume_V (t : ℝ) (ht : t > 0) : ℝ := π * abs (t - 1 / t)

-- Final limit statement
theorem limit_S_over_V : tendsto (fun (t : ℝ) => 2 * abs (log t) / (π * abs (t - 1 / t))) (𝓝[<] 1) (𝓝 (2 / π)) := 
sorry

end limit_S_over_V_l422_422138


namespace remainder_196c_2008_mod_97_l422_422152

theorem remainder_196c_2008_mod_97 (c : ℤ) : ((196 * c) ^ 2008) % 97 = 44 := by
  sorry

end remainder_196c_2008_mod_97_l422_422152


namespace dartboard_probability_l422_422905

noncomputable def inner_radius : ℝ := 4
noncomputable def outer_radius : ℝ := 8
noncomputable def num_regions : ℕ := 3
noncomputable def point_values_inner : List ℕ := [3, 4, 5]
noncomputable def point_values_outer : List ℕ := [4, 5, 3]

def area_circle (radius : ℝ) : ℝ := π * radius^2
def area_ring (outer_radius inner_radius : ℝ) : ℝ := π * (outer_radius^2 - inner_radius^2)

def probabilities_of_multiples_of_10 : ℝ :=
  let inner_area := area_circle inner_radius
  let outer_area := area_ring outer_radius inner_radius
  /-(Inner region probabilities)-/
  let prob_inner_5 := (inner_area / 3) / inner_area
  /-(Outer region probabilities: 4 or 5)-/
  let prob_outer_4_5 := (2 * outer_area / 3) / outer_area
  let prob_5_4 := prob_inner_5 * prob_outer_4_5
  2 * prob_5_4  -- Double for symmetry

theorem dartboard_probability : probabilities_of_multiples_of_10 = 1 / 16 := by
  sorry

end dartboard_probability_l422_422905


namespace smallest_possible_average_l422_422381

theorem smallest_possible_average :
  ∃ (S : set ℝ), S = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  (∃ (A B C D E F : ℝ), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ E ∈ S ∧ F ∈ S ∧ 
  {A, B, C, D, E, F}.card = 6 ∧ 
  (∀ x ∈ {A, B, C, D, E, F}, ∃ n (d : ℝ), (0 ≤ n ∧ n ≤ 9) ∧ d ∈ {x}) ∧ 
  (A < 10 ∧ B < 10 ∧ C < 10) ∧ 
  (D ≥ 10 ∧ E ≥ 10 ∧ F ≥ 10)) ∧ 
  (∀ (set1 : finset ℝ), 
    set1 = {A, B, C, D, E, F} → 
    let sum1 := set1.sum in 
    sum1 = 99 / 6 → 
    sum1 = 16.5) :=
sorry

end smallest_possible_average_l422_422381


namespace girls_left_to_play_kho_kho_l422_422289

theorem girls_left_to_play_kho_kho (B G x : ℕ) 
  (h_eq : B = G)
  (h_twice : B = 2 * (G - x))
  (h_total : B + G = 32) :
  x = 8 :=
by sorry

end girls_left_to_play_kho_kho_l422_422289


namespace minimum_valid_seating_l422_422653

def circular_table := { chairs : ℕ // chairs = 100 }
def valid_seating (N : ℕ) (seated : set ℕ) : Prop :=
  (∀ n ∈ seated, n ∈ finset.range 100) ∧
  (∀ n, n ∈ seated → (n + 1) % 100 ∈ seated ∨ (n + 99) % 100 ∈ seated) ∧
  (∀ n, (n ∉ seated ∧ (n + 1) ∉ seated ∧ (n + 2) ∉ seated) → false)

theorem minimum_valid_seating {N : ℕ} {seated : set ℕ} :
  (circular_table → valid_seating N seated) → 
  (∃ N = 25, valid_seating N seated) :=
sorry

end minimum_valid_seating_l422_422653


namespace domino_coverings_l422_422756

/-- 
Proof Problem: The number of ways to cover a 13x2 rectangular playing field using 2x1 and 3x1 
dominoes, with the conditions that no gaps, no overlapping, no domino extending beyond the
playing field, and all dominoes oriented the same way, is equal to 257.
-/
theorem domino_coverings (n m : ℕ) (h1 : n = 13) (h2 : m = 2) :
  let total_coverings := 257 in
  -- demonstrate the number of valid domino coverings
  ∃ cover_count, cover_count = total_coverings :=
begin
  let cover_count := 257,
  use cover_count,
  sorry
end

end domino_coverings_l422_422756


namespace range_of_a_l422_422831

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) = false → (a < -2 ∨ 2 < a) :=
by
  intros h,
  sorry

end range_of_a_l422_422831


namespace determine_p_q_l422_422807

noncomputable def A (p q : ℝ) : set ℝ := {x | x^2 + p * x + q = 0 }
def B : set ℝ := {x | x^2 - 3 * x + 2 = 0}

theorem determine_p_q (p q : ℝ):
  (A p q ∩ B = A p q) ↔ 
  (p^2 < 4 * q) ∨ (p = -2 ∧ q = 1) ∨ (p = -4 ∧ q = 4) ∨ (p = -3 ∧ q = 2) :=
sorry

end determine_p_q_l422_422807


namespace solution_set_of_derivative_positive_l422_422094

def f (x : ℝ) := 2 * x - 4 * Real.log x

theorem solution_set_of_derivative_positive :
  { x : ℝ | 2 - 4 / x > 0 } = Ioi 2 := 
sorry

end solution_set_of_derivative_positive_l422_422094


namespace ratio_of_smaller_circle_to_larger_circle_l422_422974

section circles

variables {Q : Type} (C1 C2 : ℝ) (angle1 : ℝ) (angle2 : ℝ)

def ratio_of_areas (C1 C2 : ℝ) : ℝ := (C1 / C2)^2

theorem ratio_of_smaller_circle_to_larger_circle
  (h1 : angle1 = 60)
  (h2 : angle2 = 48)
  (h3 : (angle1 / 360) * C1 = (angle2 / 360) * C2) :
  ratio_of_areas C1 C2 = 16 / 25 :=
by
  sorry

end circles

end ratio_of_smaller_circle_to_larger_circle_l422_422974


namespace probability_snow_once_first_week_l422_422532

theorem probability_snow_once_first_week :
  let p_first_two_days := (3 / 4) * (3 / 4)
  let p_next_three_days := (1 / 2) * (1 / 2) * (1 / 2)
  let p_last_two_days := (2 / 3) * (2 / 3)
  let p_no_snow := p_first_two_days * p_next_three_days * p_last_two_days
  let p_at_least_once := 1 - p_no_snow
  p_at_least_once = 31 / 32 :=
by
  sorry

end probability_snow_once_first_week_l422_422532


namespace quadratic_coefficients_l422_422023

theorem quadratic_coefficients :
  ∀ (x : ℝ), ∃ a b c : ℝ, (a ≠ 0) ∧ a * x^2 + b * x + c = 0 ∧ a = 1 ∧ b = -5 ∧ c = 1 :=
by
  intros x
  use [1, -5, 1]
  split
  sorry

end quadratic_coefficients_l422_422023


namespace equation_of_line_l422_422755

theorem equation_of_line {x y : ℝ} (b : ℝ) (h1 : ∀ x y, (3 * x + 4 * y - 7 = 0) → (y = -3/4 * x))
  (h2 : (1 / 2) * |b| * |(4 / 3) * b| = 24) : 
  ∃ b : ℝ, ∀ x, y = -3/4 * x + b := 
sorry

end equation_of_line_l422_422755


namespace percentage_selected_B_is_7_l422_422471

-- Define the conditions
def candidates_A := 8000
def candidates_B := 8000
def selected_percentage_A := 0.06
def selected_increase_B := 80

-- The function to calculate the selected candidates in state A and state B
def selected_A := selected_percentage_A * candidates_A
def selected_B := selected_A + selected_increase_B
def selected_percentage_B := (selected_B / candidates_B) * 100

-- The theorem
theorem percentage_selected_B_is_7 :
  selected_percentage_B = 7 := by
  -- Proof steps are not required, hence we skip them
  sorry

end percentage_selected_B_is_7_l422_422471


namespace log_base_conversion_l422_422817

theorem log_base_conversion :
  ∀ (a b : ℝ), 16 = a → log b a = 4 / (log b 10) :=
by
  assume a b h
  rw [h]
  -- Here, we need to prove that log base b of 16 is 4 divided by log base b of 10
  sorry

end log_base_conversion_l422_422817


namespace incenter_collinear_l422_422294

open EuclideanGeometry

noncomputable def incenter (A B C : Point) : Circle := sorry
noncomputable def isConvex (ABC : Triangle) : Prop := sorry
noncomputable def isInscribed (A B C D : Point) (Γ : Circle) : Prop := sorry
noncomputable def isTangent (Γ δ : Circle) (P : Point) : Prop := sorry
noncomputable def intersects (δ : Circle) (BC : Segment) : Prop := sorry
noncomputable def points_collinear (P Q R : Point) : Prop := sorry
noncomputable def lies_on_line (P : Point) (l : Line) : Prop := sorry

theorem incenter_collinear : ∀ (A B C D P Q : Point) (Γ δ : Circle),
  isConvex (Quadrilateral.mk A B C D) →
  isInscribed A B C D Γ →
  isTangent δ Γ P →
  isTangent δ Γ Q →
  intersects δ (Segment.mk B C) →
  (lies_on_line (incenter A B C) (Line.mk P Q)) ∧ 
  (lies_on_line (incenter D B C) (Line.mk P Q)) :=
begin
  sorry
end

end incenter_collinear_l422_422294


namespace pyramid_side_length_l422_422197

noncomputable def side_length_of_square_base (area_of_lateral_face : ℝ) (slant_height : ℝ) : ℝ :=
  2 * area_of_lateral_face / slant_height

theorem pyramid_side_length 
  (area_of_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_of_lateral_face = 120)
  (h2 : slant_height = 24) :
  side_length_of_square_base area_of_lateral_face slant_height = 10 :=
by
  -- Skipping the proof details.
  sorry

end pyramid_side_length_l422_422197


namespace complex_div_simplification_l422_422199

theorem complex_div_simplification : (1 + 2*Complex.i) / (2 - Complex.i) = Complex.i :=
by
  sorry

end complex_div_simplification_l422_422199


namespace distinct_integer_sums_count_l422_422707

def is_special_fraction (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 20

noncomputable def special_fractions : list ℚ :=
  (list.range 19).map (λ n, ((n + 1) : ℚ) / ((20 - (n + 1)) : ℚ))

noncomputable def special_fraction_sums : list ℚ :=
  list.bind special_fractions (λ x, special_fractions.map (λ y, x + y))

noncomputable def distinct_integer_sums : list ℤ :=
  special_fraction_sums.filter_map (λ q, if q.den = 1 then some q.num else none)

theorem distinct_integer_sums_count : distinct_integer_sums.length = 13 :=
sorry

end distinct_integer_sums_count_l422_422707


namespace find_f_2023_4_l422_422315

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.sqrt x
| (n + 1), x => 4 / (2 - f n x)

theorem find_f_2023_4 : f 2023 4 = -2 := sorry

end find_f_2023_4_l422_422315


namespace smallest_average_l422_422386

noncomputable def smallest_possible_average : ℕ := 165 / 10

theorem smallest_average (s d: Finset ℕ) 
  (h1 : s.card = 3) 
  (h2 : d.card = 3) 
  (h3 : ∀x ∈ s ∪ d, x ∈ (Finset.range 10).erase 0)
  (h4 : (s ∪ d).card = 6)
  (h5 : s ∩ d = ∅) : 
  (∑ x in s, x + ∑ y in d, y) / 6 = smallest_possible_average :=
sorry

end smallest_average_l422_422386


namespace greatest_integer_b_l422_422017

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 12 ≠ 0) ↔ b = 6 := 
by
  sorry

end greatest_integer_b_l422_422017


namespace competition_participants_l422_422839

theorem competition_participants (N : ℕ)
  (h1 : (1 / 12) * N = 18) :
  N = 216 := 
by
  sorry

end competition_participants_l422_422839


namespace tetrahedron_volume_formula_l422_422634

theorem tetrahedron_volume_formula
  (S₁ S₂ S₃ S₄ R₁ R₂ R₃ R₄ l₁ l₂ l₃ l₄ : ℝ) :
  ∃ V : ℝ,
  V = (1 / 3) * sqrt ((1 / 2) * (S₁^2 * (l₁^2 - R₁^2) + S₂^2 * (l₂^2 - R₂^2) + S₃^2 * (l₃^2 - R₃^2) + S₄^2 * (l₄^2 - R₄^2))) :=
sorry

end tetrahedron_volume_formula_l422_422634


namespace goods_train_speed_l422_422298

theorem goods_train_speed (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ) :
  length_train = 280.04 →
  length_platform = 240 →
  time_seconds = 26 →
  speed_kmph = (length_train + length_platform) / time_seconds * 3.6 →
  speed_kmph = 72 :=
by
  intros h_train h_platform h_time h_speed
  rw [h_train, h_platform, h_time] at h_speed
  sorry

end goods_train_speed_l422_422298


namespace minimum_value_quadratic_l422_422568

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem minimum_value_quadratic :
  ∀ x : ℝ, quadratic x ≥ 1 :=
by
  sorry

end minimum_value_quadratic_l422_422568


namespace sqrt_four_four_summed_l422_422604

theorem sqrt_four_four_summed :
  sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := by
  sorry

end sqrt_four_four_summed_l422_422604


namespace f_f_of_5_equals_5_l422_422420

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then log x / log (1/3) else (1/3) ^ x

theorem f_f_of_5_equals_5 : f (f 5) = 5 :=
sorry

end f_f_of_5_equals_5_l422_422420


namespace circle_symmetric_eq_l422_422053

theorem circle_symmetric_eq :
  ∀ (x y : ℝ), ((x + 1)^2 + (y - 1)^2 = 1) →
              ((y + 1, x - 1) symmetric with respect to line x - y - 1 = 0) →
              ((x - 2)^2 + (y + 2)^2 = 1) := by
  sorry

end circle_symmetric_eq_l422_422053


namespace arithmetic_sequence_max_value_l422_422480

theorem arithmetic_sequence_max_value 
  (a_n : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a_n n = a_1 + (n - 1) * d)
  (h_a4 : a_n 4 = 2) : 
  ∃ a2 a6 : ℝ, a2 = a_n 2 ∧ a6 = a_n 6 ∧ a2 + a6 = 4 ∧ a2 * a6 ≤ 4 :=
by
  set a2 := a_n 2 with h_a2
  set a6 := a_n 6 with h_a6
  have h1 : a2 + a6 = 4 := sorry
  have h2 : a2 * a6 ≤ 4 := sorry
  use [a2, a6]
  split
  · exact h_a2
  split
  · exact h_a6
  split
  · exact h1
  · exact h2
  sorry

end arithmetic_sequence_max_value_l422_422480


namespace dress_designs_count_l422_422660

theorem dress_designs_count :
  let colors := 5 in
  let patterns := 6 in
  colors * patterns = 30 := by
  sorry

end dress_designs_count_l422_422660


namespace expand_expression_l422_422008

theorem expand_expression (x : ℝ) : (x - 1) * (4 * x + 5) = 4 * x^2 + x - 5 := 
by
  -- Proof omitted
  sorry

end expand_expression_l422_422008


namespace calculate_expression_l422_422706

theorem calculate_expression :
  -real.sqrt 4 + abs (real.sqrt 2 - 2) - 202 * 3^0 = -real.sqrt 2 - 1 :=
by {
  sorry
}

end calculate_expression_l422_422706


namespace data_transmission_time_l422_422729

def packet_size : ℕ := 256
def num_packets : ℕ := 100
def transmission_rate : ℕ := 200
def total_data : ℕ := num_packets * packet_size
def transmission_time_in_seconds : ℚ := total_data / transmission_rate
def transmission_time_in_minutes : ℚ := transmission_time_in_seconds / 60

theorem data_transmission_time :
  transmission_time_in_minutes = 2 :=
  sorry

end data_transmission_time_l422_422729


namespace approximate_roots_l422_422733

noncomputable def f (x : ℝ) : ℝ := 0.3 * x^3 - 2 * x^2 - 0.2 * x + 0.5

theorem approximate_roots : 
  ∃ x₁ x₂ x₃ : ℝ, 
    (f x₁ = 0 ∧ |x₁ + 0.4| < 0.1) ∧ 
    (f x₂ = 0 ∧ |x₂ - 0.5| < 0.1) ∧ 
    (f x₃ = 0 ∧ |x₃ - 2.6| < 0.1) :=
by
  sorry

end approximate_roots_l422_422733


namespace three_digit_cubes_divisible_by_8_and_9_l422_422447

theorem three_digit_cubes_divisible_by_8_and_9 : 
  ∃! n : ℕ, (216 ≤ n^3 ∧ n^3 ≤ 999) ∧ (n % 6 = 0) :=
sorry

end three_digit_cubes_divisible_by_8_and_9_l422_422447


namespace find_b_l422_422368

theorem find_b (x b : ℕ)
  (h1 : [x, x + 2, x + b, x + 7, x + 32].sorted)
  (h2 : ((x + (x + 2) + (x + b) + (x + 7) + (x + 32)) / 5) = (x + b + 5)) :
  b = 4 :=
sorry

end find_b_l422_422368


namespace ratio_of_areas_of_concentric_circles_l422_422971

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_concentric_circles_l422_422971


namespace problem_solution_l422_422765

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : ℝ) : Prop :=
  eccentricity = 1 / 3 → 
  let e := real.sqrt (a^2 - b^2) / a in
  e = eccentricity → (a = 3 ∧ b = 2 * real.sqrt 2 → 
  (∀ A1 A2 B : ℝ × ℝ,
    A1 = (-3, 0) ∧
    A2 = (3, 0) ∧
    B = (0, 2 * real.sqrt 2) →
    let BA1 := (B.1 - A1.1, B.2 - A1.2) in
    let BA2 := (B.1 - A2.1, B.2 - A2.2) in
    BA1.1 * BA2.1 + BA1.2 * BA2.2 = -1 →
    (∀ x y : ℝ, (x / 3)^2 + (y / (2 * real.sqrt 2))^2 = 1)))

theorem problem_solution : ellipse_equation 3 (2 * real.sqrt 2) (by norm_num) (by norm_num) (1/3) :=
sorry

end problem_solution_l422_422765


namespace k_at_4_l422_422144

/- Defining the cubic polynomial h(x) = x^3 - 3x + 2 -/
def h (x : ℝ) : ℝ := x^3 - 3 * x + 2

/- Let k be a cubic polynomial such that k(0) = 2, and the roots of k are the squares of the roots of h -/
noncomputable def k (x : ℝ) : ℝ := 
  let a := Classical.some (exists_root_of_cubic_eq_zero h)
  let b := Classical.some (exists_root_of_cubic_eq_zero (λ x, if x ≠ a then h(x) / (x - a) else 0))
  let c := Classical.some (exists_root_of_cubic_eq_zero (λ x, if x ≠ a ∧ x ≠ b then h(x) / ((x - a) * (x - b)) else 0))
  let D := 1 / 2
  D * (x - a^2) * (x - b^2) * (x - c^2)

/- Proving that k(4) = 0 -/
theorem k_at_4 : k 4 = 0 :=
sorry

end k_at_4_l422_422144


namespace standard_equation_of_ellipse_area_triangle_ABM_max_area_triangle_ABM_l422_422413

open Real

axiom ellipse_properties
  (a b : ℝ)
  (a_gt_b_gt_0 : a > b ∧ b > 0)
  (point_on_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  : (∀ e : ℝ, e = sqrt 3 / 2)
  → point_on_ellipse (sqrt 2) (sqrt 2 / 2)
  → a = 2 ∧ b = 1

theorem standard_equation_of_ellipse :
  ∃ a b : ℝ, a = 2 ∧ b = 1 ∧ ellipse_properties a b :=
begin
  sorry
end

theorem area_triangle_ABM (k : ℝ) (k_ne_zero : k ≠ 0) :
  let x1 := 2 / sqrt (1 + 4 * k^2),
      y1 := 2 * k / sqrt (1 + 4 * k^2),
      x2 := -2 / sqrt (1 + 4 * k^2),
      y2 := -2 * k / sqrt (1 + 4 * k^2),
      M := (1, 1) in
  sqrt (1 + k^2) ≠ 0
  → 2 / sqrt (1 + 4 * k^2) ≠ 0
  → ∃ A B : ℝ × ℝ, A = (x1, y1) ∧ B = (x2, y2)
  → let area := (2 * abs (k - 1)) / sqrt (1 + 4 * k^2) in
    area = 2 * |k - 1| / sqrt (1 + 4 * k^2)
:= begin
  sorry
end

theorem max_area_triangle_ABM :
  let f (k : ℝ) := (4 * (k - 1)^2) / (1 + 4 * k^2) in
  ∃ k : ℝ, k = -1 / 4 ∧ f k = 5 → sqrt f k = sqrt 5
:= begin
  sorry
end

end standard_equation_of_ellipse_area_triangle_ABM_max_area_triangle_ABM_l422_422413


namespace cylinder_lateral_surface_area_l422_422557

theorem cylinder_lateral_surface_area (S : ℝ) 
  (hS : 0 < S) 
  (unf_square : ∃ h : ℝ, h = 2 * π * (real.sqrt (S / π))) :
  let r := real.sqrt (S / π) in
  let l := 2 * π * r in
  l * l = 4 * π * S :=
by
  sorry

end cylinder_lateral_surface_area_l422_422557


namespace prod_eval_l422_422731

open BigOperators

noncomputable def product_expr : ℕ → ℝ
| k => (real.sqrt ((2 * k - 1) * (2 * k + 1))) / ((2 * k)^2)

theorem prod_eval :
  ∏ k in finset.range 50, product_expr (k + 1) = 0.171 :=
sorry

end prod_eval_l422_422731


namespace geom_seq_common_ratio_l422_422675

theorem geom_seq_common_ratio:
  ∃ r : ℝ, 
  r = -2 ∧ 
  (∀ n : ℕ, n = 0 → n = 3 →
  let a : ℕ → ℝ := λ n, if n = 0 then 25 else
                            if n = 1 then -50 else
                            if n = 2 then 100 else
                            if n = 3 then -200 else 0 in
  (a n = a 0 * r ^ n)) :=
by
  sorry

end geom_seq_common_ratio_l422_422675


namespace smallest_possible_average_l422_422382

def smallest_average (s : Finset (Fin 10)) : ℕ :=
  (Finset.sum s).toNat / 6

theorem smallest_possible_average : ∃ (s1 s2 : Finset ℕ), (s1.card = 3 ∧ s2.card = 3 ∧ 
 (s1 ∪ s2).card = 6 ∧ ∀ x, x ∈ s1 ∪ s2 → x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  smallest_average (s1 ∪ s2) = 16.5) sorry

end smallest_possible_average_l422_422382


namespace complex_division_result_l422_422204

theorem complex_division_result : (1 + 2 * Complex.i) / (2 - Complex.i) = Complex.i := 
by
  sorry

end complex_division_result_l422_422204


namespace sqrt_sum_of_four_terms_of_4_pow_4_l422_422611

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l422_422611


namespace sum_of_common_divisors_l422_422742

theorem sum_of_common_divisors (a b c d e : ℕ) (h₁ : a = 45) (h₂ : b = 90) (h₃ : c = 15) (h₄ : d = 135) (h₅ : e = 180) :
  ∑ x in {1, 3, 5, 15}, x ∈ ({1, 3, 5, 9, 15, 45} ∩ {1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90} ∩ {1, 3, 5, 9, 15, 27, 45, 135} ∩ {1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90, 180}) = 24 :=
sorry

end sum_of_common_divisors_l422_422742


namespace range_of_b_l422_422418

theorem range_of_b {a b c d : ℝ} 
  (h1 : a = b - d) 
  (h2 : c = b + d) 
  (h3 : a^2 + b^2 + c^2 = 21)
  (h4 : ∀ x y z, x + y > z → x > 0 → y > 0 → z > 0) : 
  sqrt 6 < b ∧ b ≤ sqrt 7 :=
by
  sorry

end range_of_b_l422_422418


namespace product_of_regular_15gon_l422_422918

-- Definitions and conditions
def Q (n : ℕ) : ℂ := 3 + exp (2 * π * complex.I * n / 15)

theorem product_of_regular_15gon : 
  let Q1 := Q 1 in
  let Q8 := Q 8 in
  Q1.re = 1 ∧ Q1.im = 0 ∧ Q8.re = 5 ∧ Q8.im = 0 →
  (∏ k in finset.range 15, Q k) = 14348906 :=
by 
  intros _ Q1_re Q1_im Q8_re Q8_im
  sorry

end product_of_regular_15gon_l422_422918


namespace total_amount_paid_l422_422501

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_tshirts : ℕ := 6

theorem total_amount_paid : 
  (number_of_tshirts : ℝ) * (original_price * discount_rate) = 60 := by
  sorry

end total_amount_paid_l422_422501


namespace number_of_irrational_numbers_in_list_l422_422329

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem number_of_irrational_numbers_in_list :
  let numbers := [Real.pi / 2, 0, 2, 22 / 7, -Real.sqrt 5, 3.14, 
                  5.2020020002] in 
  list.count is_irrational numbers = 3 := 
by {
  sorry
}

end number_of_irrational_numbers_in_list_l422_422329


namespace max_volume_cylinder_l422_422309

theorem max_volume_cylinder (x : ℝ) (h1 : x > 0) (h2 : x < 10) : 
  (∀ x, 0 < x ∧ x < 10 → ∃ max_v, max_v = (4 * (10^3) * Real.pi) / 27) ∧ 
  ∃ x, x = 20/3 := 
by
  sorry

end max_volume_cylinder_l422_422309


namespace kite_ratio_l422_422300
noncomputable section

-- Definitions based on the conditions in the problem
def is_kite (ABCD : Type) [quadrilateral ABCD] : Prop :=
  ∃ (A B C D : Point), 
    diagonals_perpendicular ABCD ∧ 
    (∠ B = 90°) ∧ (∠ D = 90°)

def tangency_points (ABCD : Type) [kite ABCD] (M N : Point) : Prop :=
  ∃ (incircle : circle ABCD),
    M ∈ (incircle ⊙ AB) ∧ 
    N ∈ (incircle ⊙ BC)

def similar_kite (ABCD AB'C'D' : Type) : Prop :=
  similar ABCD AB'C'D'

-- Main Lean statement
theorem kite_ratio (ABCD AB'C'D' : Type) [kite ABCD] [kite AB'C'D']
  {M N N' : Point} 
  (h: tangency_points ABCD M N) 
  (ω: circle)
  (h_ω: circle_centers_tangent ABCD ω) 
  (h_similar: similar_kite ABCD AB'C'D')
  (h_parallel: line_parallel MN' AC):
  ratio AB BC = (1 + sqrt 5) / 2 :=
by
  sorry

end kite_ratio_l422_422300


namespace complement_A_eq_l422_422080

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1}

theorem complement_A_eq :
  U \ A = {0, 2} :=
by
  sorry

end complement_A_eq_l422_422080


namespace distinct_units_digits_of_squares_mod_6_l422_422085

theorem distinct_units_digits_of_squares_mod_6 : 
  ∃ (s : Finset ℕ), s = {0, 1, 4, 3} ∧ s.card = 4 :=
by
  sorry

end distinct_units_digits_of_squares_mod_6_l422_422085


namespace range_of_m_l422_422040

open Real

def f (x m : ℝ) : ℝ := x^3 - 3 * x + m

theorem range_of_m (a b c m : ℝ)
  (ha : a ∈ Icc 0 2)
  (hb : b ∈ Icc 0 2)
  (hc : c ∈ Icc 0 2)
  (triangle : ∃ (x y z : ℝ), x = f a m ∧ y = f b m ∧ z = f c m ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
  m > 6 := 
sorry

end range_of_m_l422_422040


namespace necessary_and_sufficient_condition_for_tangency_l422_422569

-- Given conditions
variables (ρ θ D E : ℝ)

-- Definition of the circle in polar coordinates and the condition for tangency with the radial axis
def circle_eq : Prop := ρ = D * Real.cos θ + E * Real.sin θ

-- Statement of the proof problem
theorem necessary_and_sufficient_condition_for_tangency :
  (circle_eq ρ θ D E) → (D = 0 ∧ E ≠ 0) :=
sorry

end necessary_and_sufficient_condition_for_tangency_l422_422569


namespace probability_no_same_number_of_wins_l422_422842

-- Define the conditions of the problem
def teams : ℕ := 10
def games : ℕ := teams * (teams - 1) / 2
def outcomes : ℕ := 2 ^ games
def permutations : ℕ := Nat.factorial teams

-- Statement of the theorem
theorem probability_no_same_number_of_wins :
  (1 : ℚ) / (outcomes : ℚ) * (permutations : ℚ) = (10.factorial : ℚ) / 2 ^ 45 :=
sorry

end probability_no_same_number_of_wins_l422_422842


namespace min_distance_focus_parabola_l422_422411

open Real

theorem min_distance_focus_parabola : 
  ∀ (P : ℝ × ℝ) (F : ℝ × ℝ), (P.1 = (P.2)^2 / 4) ∧ F = (1, 0) → 
  min (λ P, dist P F) = 1 :=
begin
  sorry
end

end min_distance_focus_parabola_l422_422411


namespace decorations_count_l422_422719

-- Define the conditions as Lean definitions
def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def large_cauldron := 1
def budget_more_decorations := 20
def left_to_put_up := 10

-- Define the total decorations
def decorations_already_up := plastic_skulls + broomsticks + spiderwebs + pumpkins + large_cauldron
def additional_decorations := budget_more_decorations + left_to_put_up
def total_decorations := decorations_already_up + additional_decorations

-- Prove the total number of decorations will be 83
theorem decorations_count : total_decorations = 83 := by 
  sorry

end decorations_count_l422_422719


namespace smallest_possible_average_l422_422378

theorem smallest_possible_average :
  ∃ (S : set ℝ), S = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  (∃ (A B C D E F : ℝ), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ E ∈ S ∧ F ∈ S ∧ 
  {A, B, C, D, E, F}.card = 6 ∧ 
  (∀ x ∈ {A, B, C, D, E, F}, ∃ n (d : ℝ), (0 ≤ n ∧ n ≤ 9) ∧ d ∈ {x}) ∧ 
  (A < 10 ∧ B < 10 ∧ C < 10) ∧ 
  (D ≥ 10 ∧ E ≥ 10 ∧ F ≥ 10)) ∧ 
  (∀ (set1 : finset ℝ), 
    set1 = {A, B, C, D, E, F} → 
    let sum1 := set1.sum in 
    sum1 = 99 / 6 → 
    sum1 = 16.5) :=
sorry

end smallest_possible_average_l422_422378


namespace patternACannotFormCube_l422_422403

inductive Pattern
| A
| B
| C
| D

open Pattern

def canFormCube : Pattern → Prop
| A := False   -- Given that Pattern A cannot be folded to form a cube
| B := False   -- Given that Pattern B cannot be folded to form a cube
| C := True    -- Given that Pattern C can be folded to form a cube
| D := False   -- Given that Pattern D cannot be folded to form a cube

theorem patternACannotFormCube : canFormCube A = False := by
  sorry

end patternACannotFormCube_l422_422403


namespace length_of_AB_l422_422431

open Real

noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (sqrt 3 / 2) * t)
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, (∃ t : ℝ, line t = A) ∧ (∃ θ : ℝ, curve θ = A) ∧
                 (∃ t : ℝ, line t = B) ∧ (∃ θ : ℝ, curve θ = B) ∧
                 dist A B = 1 :=
by
  sorry

end length_of_AB_l422_422431


namespace systematic_sampling_fourth_student_l422_422288

/-- A class with 54 students, and four tickets for the Shanghai World Expo will be 
distributed using systematic sampling. Given that the students with numbers 3, 29, 
and 42 have already been selected, the student number of the fourth selected 
student is 16. -/
theorem systematic_sampling_fourth_student (n : ℕ) (students : set ℕ) (tickets : finset ℕ)
  (H1 : n = 54)
  (H2 : students = {1, 2, 3, ..., 54})
  (H3 : tickets = {3, 29, 42, _}) :
  _ = 16 :=
by
  sorry

end systematic_sampling_fourth_student_l422_422288


namespace inequality_may_not_hold_l422_422036

variable (a b : ℝ)

theorem inequality_may_not_hold (h : 1 / a < 1 / b ∧ 1 / b < 0) :
  ∃ b a : ℝ, 1 / a < 1 / b ∧ 1 / b < 0 ∧ ¬ (log (-b) (-a) ≥ 0) := by
  sorry

end inequality_may_not_hold_l422_422036


namespace find_beta_l422_422754

variable {α β : ℝ}

theorem find_beta (h1 : sin α + sin (α + β) + cos (α + β) = √3) (h2 : β ∈ Icc (π/4) π) : β = π/4 :=
sorry

end find_beta_l422_422754


namespace sequence_sum_l422_422757

theorem sequence_sum (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end sequence_sum_l422_422757


namespace hyperbola_properties_l422_422427

open Real

def is_asymptote (y x : ℝ) : Prop :=
  y = (1/2) * x ∨ y = -(1/2) * x

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_properties :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) →
  ∀ (a b c : ℝ), 
  (a = 2) →
  (b = 1) →
  (c = sqrt (a^2 + b^2)) →
  (∀ y x : ℝ, (is_asymptote y x)) ∧ (eccentricity a (sqrt (a^2 + b^2)) = sqrt 5 / 2) :=
by
  intros x y h a b c ha hb hc
  sorry

end hyperbola_properties_l422_422427


namespace shaded_areas_sum_l422_422958

theorem shaded_areas_sum (s : ℝ) (r : ℝ) (a b c : ℕ) 
  (h_eq_triangle : s = 10) 
  (h_diameter : r = s / 2) 
  (h_shaded_areas : 2 * (π * r^2 / 8 - (r^2 * real.sqrt 3 / 4)) = (a * π - b * real.sqrt c)) : 
  a + b + c = 78 :=
by
  sorry

end shaded_areas_sum_l422_422958


namespace collinear_A_B_D_find_k_l422_422159

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (A B C D : V)
variables (k : ℝ)

-- Conditions for part (1)
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom a_b_not_collinear : ¬(∃ (c : ℝ), b = c • a)

axiom AB : vector AB = a + b
axiom BC : vector BC = 2 • a + 8 • b
axiom CD : vector CD = 3 • (a - b)

-- Proof for part (1)
theorem collinear_A_B_D : collinear ℝ ({A, B, D} : set V) :=
sorry

-- Conditions for part (2)
axiom collinear_opposite_dir : collinear ℝ ({k • a + b, a + k • b} : set V) ∧ k < 0

-- Proof for part (2)
theorem find_k : k = -1 :=
sorry

end collinear_A_B_D_find_k_l422_422159


namespace vehicle_y_speed_l422_422986

theorem vehicle_y_speed (V_y : ℝ) : 
  let x_speed := 36
  let x_distance_in_5_hours := x_speed * 5
  let y_catch_up_distance := 22
  let y_ahead_distance := 23
  let y_total_distance := x_distance_in_5_hours + y_catch_up_distance + y_ahead_distance
  V_y = y_total_distance / 5 ↔ V_y = 45 :=
by
  sorry

end vehicle_y_speed_l422_422986


namespace number_of_smaller_cubes_in_larger_cube_l422_422682

-- Defining the conditions
def volume_large_cube : ℝ := 125
def volume_small_cube : ℝ := 1
def surface_area_difference : ℝ := 600

-- Translating the question into a math proof problem
theorem number_of_smaller_cubes_in_larger_cube : 
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3) ^ 2) = surface_area_difference :=
by
  sorry

end number_of_smaller_cubes_in_larger_cube_l422_422682


namespace locus_is_hyperbola_l422_422353

def locus_of_points (M : Type) [has_coordinates M (ℝ × ℝ)] (F : M) (L : ℝ → Prop) :=
  { p : M | ∃ (x y : ℝ), p = ⟨x, y⟩ ∧ (sqrt ((x - 4.5) ^ 2 + y ^ 2) = 3 * |x - 0.5|) }

def hyperbola_equation (M : Type) [has_coordinates M (ℝ × ℝ)] :=
  { p : M | ∃ (x y : ℝ), p = ⟨x, y⟩ ∧ (x ^ 2 / 2.25 - y ^ 2 / 18 = 1) }

theorem locus_is_hyperbola (M : Type) [has_coordinates M (ℝ × ℝ)] (p : M) :
  locus_of_points M (⟨4.5, 0⟩ : M) (λ x, x = 0.5) p ↔ hyperbola_equation M p :=
by
  sorry

end locus_is_hyperbola_l422_422353


namespace sum_first_n_terms_l422_422059

variable {a : ℕ → ℕ}  -- Define the arithmetic sequence
variable {S : ℕ → ℕ}  -- Define the sum sequence
variable (d : ℕ)  -- Define the common difference of the arithmetic sequence

-- Conditions given in the problem
axiom (h1 : a 1 = 1)  -- a_1 = 1
axiom (h2 : S 2 = a 3)  -- S_2 = a_3

-- Theorem to be proven
theorem sum_first_n_terms (n : ℕ) (h1 : a 1 = 1) (h2 : S 2 = a 3)
  (h3 : ∀ m : ℕ, a (m + 1) = a m + d)
  (h4 : S n = n * (a 1 + a n) / 2)
  (h5 : ∀ k : ℕ, S k = (k * (a 1 + (a 1 + (k - 1) * d))) / 2) :
  S n = (n * n + n) / 2 := 
sorry

end sum_first_n_terms_l422_422059


namespace ratio_of_areas_of_concentric_circles_l422_422976

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end ratio_of_areas_of_concentric_circles_l422_422976


namespace cubic_polynomials_integer_roots_l422_422364

theorem cubic_polynomials_integer_roots (a b : ℤ) :
  (∀ α1 α2 α3 : ℤ, α1 + α2 + α3 = 0 ∧ α1 * α2 + α2 * α3 + α3 * α1 = a ∧ α1 * α2 * α3 = -b) →
  (∀ β1 β2 β3 : ℤ, β1 + β2 + β3 = 0 ∧ β1 * β2 + β2 * β3 + β3 * β1 = b ∧ β1 * β2 * β3 = -a) →
  a = 0 ∧ b = 0 :=
by
  sorry

end cubic_polynomials_integer_roots_l422_422364


namespace sqrt_four_four_summed_l422_422601

theorem sqrt_four_four_summed :
  sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := by
  sorry

end sqrt_four_four_summed_l422_422601


namespace f_property_l422_422142

def f (x : ℝ) : ℝ := 1 / (3^x + real.sqrt 3)

theorem f_property (x₁ x₂ : ℝ) (h : x₁ + x₂ = 1) : 
  f x₁ + f x₂ = real.sqrt 3 / 3 :=
sorry

end f_property_l422_422142


namespace somu_present_age_l422_422263

theorem somu_present_age (S F : ℕ) (h1 : S = (1 / 3) * F)
    (h2 : S - 5 = (1 / 5) * (F - 5)) : S = 10 := by
  sorry

end somu_present_age_l422_422263


namespace coeff_x4_expression_l422_422366

theorem coeff_x4_expression : 
  let expr := 2 * (x^2 - 2 * x^4 + 3 * x) + 4 * (2 * x + 3 * x^4 - x^2 + 2 * x^5 - 2 * x^4) - 6 * (2 + x - 5 * x^4 - 2 * x^3 + x^5)
  in (coeff expr 4) = 30 := 
by
  sorry

end coeff_x4_expression_l422_422366


namespace quilt_shaded_fraction_l422_422301

-- Define the conditions
def total_unit_squares : ℕ := 16
def fully_shaded_squares : ℕ := 4
def half_shaded_squares (total: ℕ) : ℕ := total
def quarter_shaded_squares : ℕ := 4

-- Define the sums for the shaded areas
def fully_shaded_area (num_squares : ℕ) : ℕ := num_squares
def half_shaded_contribution (num_squares : ℕ) : ℕ := num_squares * (1 / 2 : ℚ)
def quarter_shaded_contribution (num_squares : ℕ) : ℕ := num_squares * (1 / 2 : ℚ)

-- Define the total shaded area
def total_shaded_area : ℚ := fully_shaded_area fully_shaded_squares +
                              (half_shaded_contribution (half_shaded_squares 8) + quarter_shaded_contribution quarter_shaded_squares)

-- Define the sought fraction
def shaded_fraction := total_shaded_area / total_unit_squares

-- The theorem stating the proof goal
theorem quilt_shaded_fraction : shaded_fraction = (5 / 8 : ℚ) :=
by sorry

end quilt_shaded_fraction_l422_422301


namespace A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l422_422686

-- Definitions of events
def A : Prop := sorry -- event that the part is of the first grade
def B : Prop := sorry -- event that the part is of the second grade
def C : Prop := sorry -- event that the part is of the third grade

-- Mathematically equivalent proof problems
theorem A_or_B : A ∨ B ↔ (A ∨ B) :=
by sorry

theorem not_A_or_C : ¬(A ∨ C) ↔ B :=
by sorry

theorem A_and_C : (A ∧ C) ↔ false :=
by sorry

theorem A_and_B_or_C : ((A ∧ B) ∨ C) ↔ C :=
by sorry

end A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l422_422686


namespace geometric_sequence_common_ratio_l422_422668

theorem geometric_sequence_common_ratio 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 25) 
  (h2 : a2 = -50) 
  (h3 : a3 = 100) 
  (h4 : a4 = -200)
  (h_geometric : a2 / a1 = a3 / a2 ∧ a3 / a2 = a4 / a3) : 
  a2 / a1 = -2 :=
by 
  have r1 : a2 / a1 = -2, sorry
  -- additional steps to complete proof here
  exact r1

end geometric_sequence_common_ratio_l422_422668


namespace standard_deviation_sqrt_2_l422_422433

def sample_values : List ℝ := [6, 7, 8, 9, 10]

theorem standard_deviation_sqrt_2
  (avg : ℝ) 
  (h_avg : avg = 8) 
  (h_sum : (6 + 7 + 8 + 9 + 10) / 5 = avg) :
  let variance := (1 / 5) * ((6 - avg)^2 + (7 - avg)^2 + (8 - avg)^2 + (9 - avg)^2 + (10 - avg)^2)
  in sqrt variance = sqrt 2 :=
by
  sorry

end standard_deviation_sqrt_2_l422_422433


namespace percentage_altered_votes_got_is_50_l422_422030

def original_votes_got := 10
def original_votes_twilight := 12
def original_votes_art_of_deal := 20

def votes_after_tampering :=
  original_votes_got +
  (original_votes_twilight / 2) +
  (original_votes_art_of_deal * 0.2)

def percentage_votes_got :=
  (original_votes_got / votes_after_tampering) * 100

theorem percentage_altered_votes_got_is_50 :
  percentage_votes_got = 50 := by
  sorry

end percentage_altered_votes_got_is_50_l422_422030


namespace possible_number_of_points_l422_422493

def valid_points_config (lines points : ℕ) : Prop :=
(lines = 3) ∧ (∀ l, l ∈ {1, 2, 3} → (2 ≤ points - 2 * (lines - 1)) ∧ (points - 2 * (lines - 1) ≤ points))

theorem possible_number_of_points (n : ℕ) : valid_points_config 3 n → n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 :=
sorry

end possible_number_of_points_l422_422493


namespace number_of_true_propositions_l422_422697

-- Definitions of the four propositions
def proposition1 : Prop :=
  ∀ (l₁ l₂ : ℝ^3), (l₁ ∈ plane₁ ∧ l₂ ∈ plane₂ ∧ plane₁ ≠ plane₂) → skew l₁ l₂

def proposition2 : Prop :=
  ∃! (l : ℝ^3), perpendicular_to_skew_lines l l₁ l₂

def proposition3 : Prop :=
  ∀ (l₁ l₂ : ℝ^3), (intersects_skew_lines l₁ l₂) → skew l₁ l₂

def proposition4 : Prop :=
  ∀ (a b c : ℝ^3), (skew a b ∧ skew b c) → skew a c

-- The problem statement
theorem number_of_true_propositions : (¬ proposition1 ∧ ¬ proposition2 ∧ ¬ proposition3 ∧ ¬ proposition4) → (number_of_true_propositions = 0) :=
sorry

end number_of_true_propositions_l422_422697


namespace number_of_classes_l422_422282

-- Definitions as per the conditions
def books_per_month : ℕ := 7
def months_per_year : ℕ := 12
def total_books_per_year : ℕ := 84
def books_per_year_per_student : ℕ := books_per_month * months_per_year

-- Main theorem to be proved
theorem number_of_classes (s : ℕ) (h1 : books_per_year_per_student = books_per_month * months_per_year)
    (h2 : total_books_per_year = 84)
    (h3 : total_books_per_year = 84)
    (h4 : books_per_year_per_student * s * 1 = total_books_per_year)
    (h5 : s > 0):
  1 = 1 :=
begin
  -- not including proof steps as per the instructions
  sorry
end

end number_of_classes_l422_422282


namespace speed_of_X_l422_422685

theorem speed_of_X (t1 t2 Vx : ℝ) (h1 : t2 - t1 = 3) 
  (h2 : 3 * Vx + Vx * t1 = 60 * t1 + 30)
  (h3 : 3 * Vx + Vx * t2 + 30 = 60 * t2) : Vx = 60 :=
by sorry

end speed_of_X_l422_422685


namespace fred_earnings_correct_l422_422872

structure Person :=
  (initial_amount : ℕ)
  (final_amount : ℕ)

def fred := Person.mk 19 40 -- Fred
def fred_earnings := fred.final_amount - fred.initial_amount

theorem fred_earnings_correct : fred_earnings = 21 := by
  unfold fred_earnings
  unfold fred
  simp
  norm_num
  sorry

end fred_earnings_correct_l422_422872


namespace expression_with_8_factors_l422_422822

theorem expression_with_8_factors {a b : ℕ} (ha : nat.prime a) (hb : nat.prime b) (hodd_a : a % 2 = 1) (hodd_b : b % 2 = 1) (h : a < b) : 
  ∃ E, E = a^3 * b ∧
  ∀ n, n.dvd E → ∃ m k, E = n * m ∧ m * k = n * n ∧ m ∈ finset.range (E + 1) ∧ k ∈ finset.range (E + 1) :=
sorry

end expression_with_8_factors_l422_422822


namespace triangle_inequality_and_similarity_l422_422515

-- Define the proof problem
theorem triangle_inequality_and_similarity
  (a b c : ℝ) -- Side lengths of the first triangle
  (S : ℝ) -- Area of the first triangle
  (α β γ : ℝ) -- Angles of the second triangle
  (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) -- Side lengths are positive
  (h₂ : 0 < S) -- Area is positive
  (h₃ : 0 < α ∧ α < π) -- Angles are within appropriate range
  (h₄ : 0 < β ∧ β < π)
  (h₅ : 0 < γ ∧ γ < π)
  (h₆ : α + β + γ = π) -- Sum of angles of the second triangle equals π
  : a^2 * Real.cot α + b^2 * Real.cot β + c^2 * Real.cot γ ≥ 4 * S := sorry

end triangle_inequality_and_similarity_l422_422515


namespace price_of_turban_l422_422305

-- Definitions based on conditions
def total_yearly_payment : ℝ := 90
def months_worked : ℝ := 9
def amount_paid : ℝ := 65
def prorated_payment : ℝ := (3 / 4) * total_yearly_payment

-- Theorem stating that the price of the turban is 2.5
theorem price_of_turban : 
  ∃ T : ℝ, prorated_payment = T + amount_paid ∧ T = 2.5 :=
by
  sorry

end price_of_turban_l422_422305


namespace calculate_expression_l422_422338

theorem calculate_expression : ((-3: ℤ) ^ 3 + (5: ℤ) ^ 2 - ((-2: ℤ) ^ 2)) = -6 := by
  sorry

end calculate_expression_l422_422338


namespace sum_of_digits_of_valid_n_eq_seven_l422_422517

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_valid_n (n : ℕ) : Prop :=
  (500 < n) ∧ (Nat.gcd 70 (n + 150) = 35) ∧ (Nat.gcd (n + 70) 150 = 50)

theorem sum_of_digits_of_valid_n_eq_seven :
  ∃ n : ℕ, is_valid_n n ∧ sum_of_digits n = 7 := by
  sorry

end sum_of_digits_of_valid_n_eq_seven_l422_422517


namespace clock_angle_at_3_30_l422_422335

theorem clock_angle_at_3_30 
    (deg_per_hour: Real := 30)
    (full_circle_deg: Real := 360)
    (hours_on_clock: Real := 12)
    (hour_hand_extra_deg: Real := 30 / 2)
    (hour_hand_deg: Real := 3 * deg_per_hour + hour_hand_extra_deg)
    (minute_hand_deg: Real := 6 * deg_per_hour) : 
    hour_hand_deg = 105 ∧ minute_hand_deg = 180 ∧ (minute_hand_deg - hour_hand_deg) = 75 := 
sorry

-- The problem specifies to write the theorem statement only, without the proof steps.

end clock_angle_at_3_30_l422_422335


namespace correct_equation_l422_422287

variable (x : ℝ) (h1 : x > 0)

def length_pipeline : ℝ := 3000
def efficiency_increase : ℝ := 0.2
def days_ahead : ℝ := 10

theorem correct_equation :
  (length_pipeline / x) - (length_pipeline / ((1 + efficiency_increase) * x)) = days_ahead :=
by
  sorry

end correct_equation_l422_422287


namespace least_xy_l422_422188

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : xy = 108 := by
  sorry

end least_xy_l422_422188


namespace unique_solution_non_unique_solution_l422_422024

-- Definitions of the problem conditions
section conditions
variable (a b x : ℝ)

-- Definition of problem equation
def problem_eq (a b x : ℝ) : Prop := 
  (x ≠ 2) ∧ (x ≠ 3) ∧
  ((x - a) / (x - 2) + (x - b) / (x - 3) = 2)

-- Proposition for unique solution
theorem unique_solution :
  ∀ a b : ℝ, (a + b ≠ 5) ∧ (a ≠ 2) ∧ (b ≠ 3) →
  ∃ x : ℝ, problem_eq a b x := sorry

-- Proposition for non-unique solution
theorem non_unique_solution :
  ∃ x : ℝ, problem_eq 2 3 x := sorry

end conditions

end unique_solution_non_unique_solution_l422_422024


namespace circle_properties_l422_422015

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x = 0

theorem circle_properties (x y : ℝ) :
  circle_center x y ↔ ((x - 2)^2 + y^2 = 2^2) ∧ ((2, 0) = (2, 0)) :=
by
  sorry

end circle_properties_l422_422015


namespace determine_f_zero_l422_422141

variable (f : ℝ → ℝ)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * (f x) * y

theorem determine_f_zero (h1: functional_equation f)
    (h2 : f 2 = 4) : f 0 = 0 := 
sorry

end determine_f_zero_l422_422141


namespace possible_values_of_b_l422_422100

theorem possible_values_of_b (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → x.im ≠ 0) ↔ (b ∈ set.Ioo (-8 : ℝ) 8) := 
by sorry

end possible_values_of_b_l422_422100


namespace angle_DAE_is_45_l422_422701

theorem angle_DAE_is_45 (A B C D E : Type)
  [HasAngle A B C] [HasAngle B C D] [HasAngle C D E] [HasAngle D E A] [HasAngle A B E] [IsoscelesRightTriangle A B C]
  (h1 : angle A B C = 90)
  (h2 : IsSquare B C D E)
  (h3 : common_side BC (triangle A B C) (square B C D E)) : angle D A E = 45 :=
by
  sorry

end angle_DAE_is_45_l422_422701


namespace bead_necklace_l422_422627

-- Define the range of beads
def beadLabels : List ℕ := List.range' 290 (2023 - 290 + 1)

-- Define a function to check if three numbers can form the sides of a triangle
def formsTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem statement to be proved
theorem bead_necklace (L : List ℕ) (hL : ∀ (i : ℕ), L.perm beadLabels) :
  ∃ (i : ℕ), i < L.length - 2 ∧ formsTriangle (L.nthLe i sorry) (L.nthLe (i + 1) sorry) (L.nthLe (i + 2) sorry) :=
sorry

end bead_necklace_l422_422627


namespace prove_dihedral_angles_minus_Omega_l422_422542

def tetrahedron.dihedral_angles_sum (α : ℕ → ℕ → ℝ) (i j k l : ℕ) :=
  [α i j, α i k, α i l, α j k, α j l, α k l].sum

def tetrahedron.Omega (α : ℕ → ℕ → ℝ) (i j k l : ℕ) :=
  λ v, α v j + α v k + α v l - π

def sum_dihedral_angles_minus_sum_Omega_eq_4pi (α : ℕ → ℕ → ℝ) (i j k l : ℕ) : Prop :=
  (tetrahedron.dihedral_angles_sum α i j k l 
    - (Ω α i j k l 0 + Ω α i j k l 1 + Ω α i j k l 2 + Ω α i j k l 3)) = 4 * π

theorem prove_dihedral_angles_minus_Omega (α : ℕ → ℕ → ℝ)
  (i j k l : ℕ)
  (h1: ∀ v, tetrahedron.Omega α i j k l v = α v j + α v k + α v l - π)
  (h2: ∀ (α_eq_twice : ∀ i j, α i j = α j i)):
  sum_dihedral_angles_minus_sum_Omega_eq_4pi α i j k l := sorry

end prove_dihedral_angles_minus_Omega_l422_422542


namespace final_pens_count_l422_422626

def pens_after_miking (initial_pens : ℕ) (mike_pens : ℕ) : ℕ :=
  initial_pens + mike_pens

def pens_after_cindy (current_pens : ℕ) (increase_percent : ℚ) : ℕ :=
  current_pens + (current_pens * increase_percent).toNat

def pens_after_giving (current_pens : ℕ) (giving_percent : ℚ) : ℕ :=
  current_pens - (current_pens * giving_percent).toNat

theorem final_pens_count : 
  let initial_pens := 5 in
  let mike_pens := 20 in
  let cindy_increase := (3 / 2 : ℚ) in -- 150% as a rational number
  let giving_percent := (3 / 10 : ℚ) in -- 30% as a rational number
  pens_after_giving (pens_after_cindy (pens_after_miking initial_pens mike_pens) cindy_increase) giving_percent = 44 :=
by 
  sorry

end final_pens_count_l422_422626


namespace slope_of_line_6x_minus_4y_eq_16_l422_422741

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  if b ≠ 0 then -a / b else 0

theorem slope_of_line_6x_minus_4y_eq_16 :
  slope_of_line 6 (-4) (-16) = 3 / 2 :=
by
  -- skipping the proof
  sorry

end slope_of_line_6x_minus_4y_eq_16_l422_422741


namespace speed_of_faster_train_l422_422588

noncomputable def speed_of_slower_train : ℝ := 36
noncomputable def length_of_each_train : ℝ := 70
noncomputable def time_to_pass : ℝ := 36

theorem speed_of_faster_train : 
  ∃ (V_f : ℝ), 
    (V_f - speed_of_slower_train) * (1000 / 3600) = 140 / time_to_pass ∧ 
    V_f = 50 :=
by {
  sorry
}

end speed_of_faster_train_l422_422588


namespace library_books_total_l422_422505

-- Definitions for the conditions
def books_purchased_last_year : Nat := 50
def books_purchased_this_year : Nat := 3 * books_purchased_last_year
def books_before_last_year : Nat := 100

-- The library's current number of books
def total_books_now : Nat :=
  books_before_last_year + books_purchased_last_year + books_purchased_this_year

-- The proof statement
theorem library_books_total : total_books_now = 300 :=
by
  -- Placeholder for actual proof
  sorry

end library_books_total_l422_422505


namespace centroid_of_prism_proof_l422_422016

noncomputable def centroid_of_prism := 
  let V := (∫ x in 0..1, ∫ y in 0..1, ∫ z in 0..(4 - x - y), 1, volume) 
  (1 / V, 
   ∫ x in 0..1, ∫ y in 0..1, ∫ z in 0..(4 - x - y), x * volume / V, 
   ∫ x in 0..1, ∫ y in 0..1, ∫ z in 0..(4 - x - y), y * volume / V, 
   ∫ x in 0..1, ∫ y in 0..1, ∫ z in 0..(4 - x - y), z * volume / V)  

theorem centroid_of_prism_proof : 
  centroid_of_prism = (17 / 36, 17 / 36, 55 / 36) := sorry

end centroid_of_prism_proof_l422_422016


namespace prove_sqrt_simplified_expr_l422_422549

def sqrt_simplified_expr (x : ℝ) : Prop :=
  x = (sqrt (7^2 + 24^2)) / (sqrt (49 + 16)) → x = (25 * sqrt 65) / 65

theorem prove_sqrt_simplified_expr (x : ℝ) : sqrt_simplified_expr x :=
  by
    sorry

end prove_sqrt_simplified_expr_l422_422549


namespace part1_part2_part3_triangle_area_l422_422436

noncomputable def vector_m (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin x, cos x)

noncomputable def vector_n (x : ℝ) : ℝ × ℝ :=
  (cos x, cos x)

noncomputable def f (x : ℝ) : ℝ :=
  vector_m x.1 * vector_n x.1 + vector_m x.2 * vector_n x.2

theorem part1 (x : ℝ) : 
  f x = sqrt 3 / 2 * sin (2 * x) + 1 / 2 * cos (2 * x) + 1 / 2 :=
sorry

theorem part2 (x : ℝ): 
  ∃ k : ℤ, -π / 3 + k * π ≤ x ∧ x ≤ π / 6 + k * π :=
sorry

variables (a b c A : ℝ)

-- Conditions for triangle ABC
axiom side_a : a = 1
axiom sides_sum : b + c = 2
axiom f_A : f A = 1

theorem part3 : A = π / 3 :=
sorry

theorem triangle_area : 
  let b := sqrt (2 + cos (π / 3)) / 2,
  let c := 2 - b in
  1 / 2 * b * c * sqrt 3 / 2 = sqrt 3 / 4 :=
sorry

end part1_part2_part3_triangle_area_l422_422436


namespace intersection_M_N_l422_422105

-- Define sets M and N
def M := { x : ℝ | ∃ t : ℝ, x = 2^(-t) }
def N := { y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 1 } :=
by sorry

end intersection_M_N_l422_422105


namespace count_digits_2_and_3_in_range_l422_422087

theorem count_digits_2_and_3_in_range :
  let count := (λ (s : Set ℕ), s.filter (λ n, 
    let d1 := (n / 1000) % 10 in
    let d2 := (n / 100) % 10 in
    let d3 := (n / 10) % 10 in
    let d4 := n % 10 in
    d1 = 1 ∧ 
    ((d2 = 2 ∧ d3 = 3) ∨ (d2 = 3 ∧ d3 = 2) ∨
     (d2 = 2 ∧ d4 = 3) ∨ (d2 = 3 ∧ d4 = 2) ∨
     (d3 = 2 ∧ d4 = 3) ∨ (d3 = 3 ∧ d4 = 2)))) {n | 1000 ≤ n ∧ n < 2000}
  in count = 108 := 
by {
  sorry 
}

end count_digits_2_and_3_in_range_l422_422087


namespace smallest_possible_average_l422_422380

theorem smallest_possible_average :
  ∃ (S : set ℝ), S = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  (∃ (A B C D E F : ℝ), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ E ∈ S ∧ F ∈ S ∧ 
  {A, B, C, D, E, F}.card = 6 ∧ 
  (∀ x ∈ {A, B, C, D, E, F}, ∃ n (d : ℝ), (0 ≤ n ∧ n ≤ 9) ∧ d ∈ {x}) ∧ 
  (A < 10 ∧ B < 10 ∧ C < 10) ∧ 
  (D ≥ 10 ∧ E ≥ 10 ∧ F ≥ 10)) ∧ 
  (∀ (set1 : finset ℝ), 
    set1 = {A, B, C, D, E, F} → 
    let sum1 := set1.sum in 
    sum1 = 99 / 6 → 
    sum1 = 16.5) :=
sorry

end smallest_possible_average_l422_422380


namespace count_divisibles_l422_422813

theorem count_divisibles (h1 : Finset.range' 1 60 → ℕ) (h2 : ∀ n, n ∈ h1 -> (n % 3 = 0 ∨ n % 5 = 0 ∨ n % 7 = 0)) :
  (Finset.filter (λ x, (x % 3 = 0) ∨ (x % 5 = 0) ∨ (x % 7 = 0)) (Finset.range' 1 60)).card = 33 :=
by
  sorry

end count_divisibles_l422_422813


namespace three_same_colored_balls_l422_422960

theorem three_same_colored_balls (balls : ℕ) (color_count : ℕ) (balls_per_color : ℕ) (h1 : balls = 60) (h2 : color_count = balls / balls_per_color) (h3 : balls_per_color = 6) :
  ∃ n, n = 21 ∧ (∀ picks : ℕ, picks ≥ n → ∃ c, ∃ k ≥ 3, k ≤ balls_per_color ∧ (c < color_count) ∧ (picks / c = k)) :=
sorry

end three_same_colored_balls_l422_422960


namespace AMC10_paths_count_l422_422107

/-- The number of different paths to spell "AMC10" starting from the central 'A' and moving to adjacent letters (including diagonals) is equal to 4096. -/
theorem AMC10_paths_count : 
  (number_of_paths "AMC10" (start := 'A') (moves := [adjacent, diagonal])) = 4096 :=
sorry

end AMC10_paths_count_l422_422107


namespace equality_of_costs_l422_422930

theorem equality_of_costs (x : ℕ) :
  (800 + 30 * x = 500 + 35 * x) ↔ x = 60 := by
  sorry

end equality_of_costs_l422_422930


namespace Gerald_needs_to_average_5_chores_per_month_l422_422390

def spending_per_month := 100
def season_length := 4
def cost_per_chore := 10
def total_spending := spending_per_month * season_length
def months_not_playing := 12 - season_length
def amount_to_save_per_month := total_spending / months_not_playing
def chores_per_month := amount_to_save_per_month / cost_per_chore

theorem Gerald_needs_to_average_5_chores_per_month :
  chores_per_month = 5 := by
  sorry

end Gerald_needs_to_average_5_chores_per_month_l422_422390


namespace greatest_possible_integer_l422_422527

noncomputable def max_integer : ℕ :=
  let m := 142 in
  if m < 150 ∧ (∃ k : ℤ, m = 9 * k - 2) ∧ (∃ j : ℤ, m = 11 * j - 4) then m else 0

theorem greatest_possible_integer :
  ∃ m : ℕ, m = 142 ∧ m < 150 ∧
  (∃ k : ℤ, m = 9 * k - 2) ∧ (∃ j : ℤ, m = 11 * j - 4) :=
by
  use max_integer
  sorry

end greatest_possible_integer_l422_422527


namespace pyramid_base_sidelength_l422_422195

theorem pyramid_base_sidelength (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120) (hh : h = 24) (area_eq : A = 1/2 * s * h) : s = 10 := by
  sorry

end pyramid_base_sidelength_l422_422195


namespace moles_of_H2_required_l422_422446

theorem moles_of_H2_required 
  (moles_C : ℕ) 
  (moles_O2 : ℕ) 
  (moles_CH4 : ℕ) 
  (moles_CO2 : ℕ) 
  (balanced_reaction_1 : ℕ → ℕ → ℕ → Prop)
  (balanced_reaction_2 : ℕ → ℕ → ℕ → ℕ → Prop)
  (H_balanced : balanced_reaction_2 2 4 2 1)
  (H_form_CO2 : balanced_reaction_1 1 1 1) :
  moles_C = 2 ∧ moles_O2 = 1 ∧ moles_CH4 = 2 ∧ moles_CO2 = 1 → (∃ moles_H2, moles_H2 = 4) :=
by sorry

end moles_of_H2_required_l422_422446


namespace max_min_sum_is_4_l422_422415

noncomputable def f (x : ℝ) : ℝ := 2 + x^2 * cos (π / 2 + x)

theorem max_min_sum_is_4 (a : ℝ) (h : a > 0) :
  let M := Real.sup (Set.image f (Set.Icc (-a) a))
  let m := Real.inf (Set.image f (Set.Icc (-a) a))
  M + m = 4 :=
by
  sorry

end max_min_sum_is_4_l422_422415


namespace perimeter_triangle_ABC_l422_422586

-- Given triangle ABC and its circumcircle Ω
variable (A B C X Y E F : Point)
variable (Ω : Circle)
variable (XY : Segment XY) (AC : Segment AC) (AB : Segment AB)
variable (AF : Segment AF) (EC : Segment EC) (FB : Segment FB)

-- Conditions of the problem
axiom circumcircle (TriangleHasCircumcircle : triangle A B C ∧ circumcircle Ω (triangle_circumcircle Ω A B C))
axiom chord_intersects (chordIntersects : chord XY Ω ∧ intersects XY AC E ∧ intersects XY AB F ∧ lies_between E X F)
axiom bisects_arc (bisectsArc : bisects A XY)
axiom given_lengths (givenLengths : EC.length = 7 ∧ FB.length = 10 ∧ AF.length = 8 ∧ (Y.length - X.length = 2))

-- Question and required proof statement
theorem perimeter_triangle_ABC : 
  let EC_length := 7
      FB_length := 10
      AF_length := 8
      YF_XE_diff := 2
  in (perimeter_tr A B C EC_length FB_length AF_length YF_XE_diff) = 51 :=
sorry

end perimeter_triangle_ABC_l422_422586


namespace find_f_sqrt2_l422_422720

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x, x > 0 → (∃ y, f y = x ∨ y = x)

axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_at_8 : f 8 = 3

-- Define the problem statement
theorem find_f_sqrt2 : f (Real.sqrt 2) = 1 / 2 := sorry

end find_f_sqrt2_l422_422720


namespace total_consumption_in_a_day_l422_422133

/-- Jorge and Giuliana's daily consumption of croissants, cakes, and pizzas -/
variables (jorgeCroissants giulianaCroissants : ℕ) 
          (jorgeCakes giulianaCakes : ℕ)
          (jorgePizzas giulianaPizzas : ℕ)

-- Given conditions
def jorge_croissants := 7
def giuliana_croissants := 7
def jorge_cakes := 18
def giuliana_cakes := 18
def jorge_pizzas := 30
def giuliana_pizzas := 30

-- Total consumption for Jorge and Giuliana
def total_croissants := jorge_croissants + giuliana_croissants
def total_cakes := jorge_cakes + giuliana_cakes
def total_pizzas := jorge_pizzas + giuliana_pizzas

-- Total daily consumption
def total_consumed := total_croissants + total_cakes + total_pizzas

-- Theorem statement
theorem total_consumption_in_a_day :
  total_consumed = 110 :=
by 
  -- Placeholder for the proof
  sorry

end total_consumption_in_a_day_l422_422133


namespace max_correct_answers_l422_422170

theorem max_correct_answers (a b c : ℕ) :
  a + b + c = 50 ∧ 4 * a - c = 99 ∧ b = 50 - a - c ∧ 50 - a - c ≥ 0 →
  a ≤ 29 := by
  sorry

end max_correct_answers_l422_422170


namespace sharon_trip_distance_l422_422004

-- Define the variables and constants.
def total_distance := x : Real
def usual_time := 200 : Real
def snowstorm_time := 340 : Real
def speed_reduction := 15 / 60 : Real -- 15 miles per hour converted to miles per minute.

-- Define the given conditions.
def usual_speed := total_distance / usual_time
def distance_before_snowstorm := total_distance / 4
def new_speed := usual_speed - speed_reduction

-- The total travel time is 340 minutes considering the condition changes.
theorem sharon_trip_distance :
  (distance_before_snowstorm / usual_speed) + ((3 * distance_before_snowstorm) / new_speed) = snowstorm_time ->
  total_distance = 104 :=
begin
  sorry
end

end sharon_trip_distance_l422_422004


namespace seokgi_jumped_furthest_l422_422624

noncomputable def yooseung_jump : ℝ := 15 / 8
def shinyoung_jump : ℝ := 2
noncomputable def seokgi_jump : ℝ := 17 / 8

theorem seokgi_jumped_furthest :
  yooseung_jump < seokgi_jump ∧ shinyoung_jump < seokgi_jump :=
by
  sorry

end seokgi_jumped_furthest_l422_422624


namespace dealer_profit_percentage_l422_422658

-- Given conditions
def cost_price : ℝ := 1
def marked_price : ℝ := cost_price * 2
def discount_rate : ℝ := 0.10
def selling_price_discounted : ℝ := marked_price * (1 - discount_rate)
def articles_deal_ratio : ℝ := 15 / 20
def special_deal_price_per_article : ℝ := cost_price * articles_deal_ratio

-- Desired proof statement
theorem dealer_profit_percentage : 
  let profit := selling_price_discounted - cost_price in
  let profit_percentage := (profit / cost_price) * 100 in
  profit_percentage = 80 :=
by
  sorry

end dealer_profit_percentage_l422_422658


namespace p_neither_necessary_nor_sufficient_l422_422042

def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x : ℝ) : Prop := x ≠ 0
def r (y : ℝ) : Prop := y ≠ -1

theorem p_neither_necessary_nor_sufficient (x y : ℝ) (h1: p x y) (h2: q x) (h3: r y) :
  ¬(p x y → q x) ∧ ¬(q x → p x y) := 
by 
  sorry

end p_neither_necessary_nor_sufficient_l422_422042


namespace coefficient_of_x_squared_in_expansion_rational_terms_in_expansion_l422_422786

theorem coefficient_of_x_squared_in_expansion : 
  let expr := (λ x : ℝ, (root 3 x - (1 / (2 * root 3 x))) ^ 10) in
  (coeff (series.expr x) 2) = (45 / 4) :=
sorry

theorem rational_terms_in_expansion :
  let expr := (λ x : ℝ, (root 3 x - (1 / (2 * root 3 x))) ^ 10) in
  (rational_terms (series.expr x)) = 
    [(45 / 4) * x ^ 2, (-63 / 8) * x ^ 0, (45 / 256) * x ^ (-2)] :=
sorry

end coefficient_of_x_squared_in_expansion_rational_terms_in_expansion_l422_422786


namespace minnie_takes_more_time_l422_422898

-- Definitions
def speed_Minnie_flat : ℝ := 25
def speed_Minnie_downhill : ℝ := 35
def speed_Minnie_uphill : ℝ := 10

def speed_Penny_flat : ℝ := 35
def speed_Penny_downhill : ℝ := 45
def speed_Penny_uphill : ℝ := 15

def distance_X_to_Y : ℝ := 15
def distance_Y_to_Z : ℝ := 20
def distance_Z_to_X : ℝ := 25

-- Calculate Minnie's total time in hours
def total_time_Minnie : ℝ :=
  (distance_X_to_Y / speed_Minnie_uphill) +
  (distance_Y_to_Z / speed_Minnie_downhill) +
  (distance_Z_to_X / speed_Minnie_flat)

-- Calculate Penny's total time in hours
def total_time_Penny : ℝ :=
  (distance_Z_to_X / speed_Penny_flat) +
  (distance_Y_to_Z / speed_Penny_uphill) +
  (distance_X_to_Y / speed_Penny_downhill)

-- Convert hours to minutes
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- Calculate the difference in time in minutes
def time_difference_minutes : ℝ :=
  hours_to_minutes (total_time_Minnie - total_time_Penny)

-- The proof statement
theorem minnie_takes_more_time : time_difference_minutes = 414.29 := by
  sorry

end minnie_takes_more_time_l422_422898


namespace new_boys_joined_l422_422274

-- Definitions based on conditions
variables (initial_people : ℕ) (initial_girls_ratio : ℝ) (final_girls_ratio : ℝ)
variables (number_of_new_boys : ℕ)

-- Constants from the problem
def initial_people := 20
def initial_girls_ratio := 0.40
def final_girls_ratio := 0.32

-- Mathematical formulation of the problem
def problem_statement := number_of_new_boys = 5

-- Proof statement that we will prove
theorem new_boys_joined (number_of_new_boys : ℕ) : 
  (∀ initial_people = 20, initial_girls_ratio = 0.40, final_girls_ratio = 0.32,
  let initial_girls := initial_girls_ratio * initial_people,
      total_people_after := initial_people + number_of_new_boys,
      new_girls_ratio := initial_girls / total_people_after in
  new_girls_ratio = final_girls_ratio → number_of_new_boys = 5) :=
by
  sorry

end new_boys_joined_l422_422274


namespace locus_of_points_l422_422406

noncomputable def locus_points (A B : Point) (d : ℝ) : Set Point :=
  let M : Point := (x, y)
  let AM_sq := (x - A.x) ^ 2 + (y - A.y) ^ 2
  let BM_sq := (x - B.x) ^ 2 + (y - B.y) ^ 2
  M ∈ { M | AM_sq + BM_sq = d }

theorem locus_of_points (A B : Point) (d : ℝ) (hA : A = (0, 0)) (hB : B = (b, 0)) :
  d > b^2 / 2 →
  ∃ c r : ℝ, r > 0 ∧ locus_points A B d = { (x, y) | (x - c)^2 + y^2 = r^2 } := sorry

end locus_of_points_l422_422406


namespace bacteria_growth_correct_l422_422359

-- Define the conditions
def divisions_per_hour : ℕ := 3 * 60 // 20  -- Number of divisions in 3 hours
def initial_bacteria : ℕ := 1  -- Starting with 1 bacterium
def divisions : ℕ := 9  -- Calculated from (3 * 60) / 20
def expected_bacteria_after_3_hours : ℕ := 512  -- Expected number of bacteria after 3 hours

-- Formulate the theorem 
theorem bacteria_growth_correct : (initial_bacteria * 2 ^ divisions) = expected_bacteria_after_3_hours := by
  -- Define the proof environment
  sorry

end bacteria_growth_correct_l422_422359


namespace range_of_m_for_second_quadrant_l422_422061

theorem range_of_m_for_second_quadrant (m : ℝ) :
  (P : ℝ × ℝ) → P = (1 + m, 3) → P.fst < 0 → m < -1 :=
by
  intro P hP hQ
  sorry

end range_of_m_for_second_quadrant_l422_422061


namespace ball_more_than_bat_l422_422938

theorem ball_more_than_bat :
  ∃ x y : ℕ, (2 * x + 3 * y = 1300) ∧ (3 * x + 2 * y = 1200) ∧ (y - x = 100) :=
by
  sorry

end ball_more_than_bat_l422_422938


namespace solve_for_nabla_l422_422449

theorem solve_for_nabla : ∃ (nabla : ℤ), 4 * -3 = nabla - 3 ∧ nabla = -9 := by
  have h1 : 4 * -3 = -12 := by linarith
  use -9
  split
  · exact h1
  · rfl

end solve_for_nabla_l422_422449


namespace inequality_positive_real_numbers_l422_422537

theorem inequality_positive_real_numbers
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a * b + b * c + c * a = 1) :
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ (3 / 2) :=
  sorry

end inequality_positive_real_numbers_l422_422537


namespace determine_rm_l422_422212

theorem determine_rm : 
  ∃ (r m : ℝ), (∀ t : ℝ, ∃ x y : ℝ, ⟨x, y⟩ = ⟨r, -2⟩ + t • ⟨3, m⟩ ∧ y = 2 * x - 8) ∧ r = 3 ∧ m = 6 :=
by
  sorry

end determine_rm_l422_422212


namespace rearranged_sum_not_9999_l422_422645

theorem rearranged_sum_not_9999 :
  ∀ n : ℕ, let s := String.repeat '9' 1999 in n + (rearrange_digits n) ≠ s.toNat :=
by
  sorry

end rearranged_sum_not_9999_l422_422645


namespace problem_solution_l422_422763

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : ℝ) : Prop :=
  eccentricity = 1 / 3 → 
  let e := real.sqrt (a^2 - b^2) / a in
  e = eccentricity → (a = 3 ∧ b = 2 * real.sqrt 2 → 
  (∀ A1 A2 B : ℝ × ℝ,
    A1 = (-3, 0) ∧
    A2 = (3, 0) ∧
    B = (0, 2 * real.sqrt 2) →
    let BA1 := (B.1 - A1.1, B.2 - A1.2) in
    let BA2 := (B.1 - A2.1, B.2 - A2.2) in
    BA1.1 * BA2.1 + BA1.2 * BA2.2 = -1 →
    (∀ x y : ℝ, (x / 3)^2 + (y / (2 * real.sqrt 2))^2 = 1)))

theorem problem_solution : ellipse_equation 3 (2 * real.sqrt 2) (by norm_num) (by norm_num) (1/3) :=
sorry

end problem_solution_l422_422763


namespace senior_high_sample_count_l422_422306

theorem senior_high_sample_count 
  (total_students : ℕ)
  (junior_high_students : ℕ)
  (senior_high_students : ℕ)
  (total_sampled_students : ℕ)
  (H1 : total_students = 1800)
  (H2 : junior_high_students = 1200)
  (H3 : senior_high_students = 600)
  (H4 : total_sampled_students = 180) :
  (senior_high_students * total_sampled_students / total_students) = 60 := 
sorry

end senior_high_sample_count_l422_422306


namespace factorize_ax_squared_minus_9a_l422_422010

theorem factorize_ax_squared_minus_9a (a x : ℝ) : 
  a * x^2 - 9 * a = a * (x - 3) * (x + 3) :=
sorry

end factorize_ax_squared_minus_9a_l422_422010


namespace surface_area_of_circumscribed_sphere_of_tetrahedron_l422_422320

-- Define points and verify perpendicular edges
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

noncomputable def A : Point3D := ⟨0, 0, 0⟩
noncomputable def B : Point3D := ⟨0, 4, 0⟩
noncomputable def C : Point3D := ⟨4, 4, 0⟩
noncomputable def D : Point3D := ⟨0, 0, 2⟩

def distance (p1 p2 : Point3D) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2).sqrt

def surface_area_of_circumscribed_sphere (radius : ℝ) : ℝ :=
  4 * Real.pi * radius^2

theorem surface_area_of_circumscribed_sphere_of_tetrahedron :
  surface_area_of_circumscribed_sphere (distance A D / 2) = 36 * Real.pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_of_tetrahedron_l422_422320


namespace subtracted_number_l422_422183

theorem subtracted_number:
  ∃ x: ℝ, 4 * 5.0 - x = 13 ∧ x = 7 :=
by
  use 7
  split
  repeat { sorry }

end subtracted_number_l422_422183


namespace exam_sequences_l422_422680

-- Definition of the conditions
def chinese_first (seq : List ℕ) : Prop := seq.nth_le 0 (by linarith) = 1

def not_adjacent (seq : List ℕ) : Prop :=
  ∀ i, (i < seq.length - 1) → (seq.nth_le i (by linarith) = 2 → seq.nth_le (i + 1) (by linarith) ≠ 3)
  ∧ (seq.nth_le i (by linarith) = 3 → seq.nth_le (i + 1) (by linarith) ≠ 2)

-- Proof statement for the number of possible exam sequences
theorem exam_sequences (seq : List ℕ) : chinese_first seq ∧ not_adjacent seq →
  seq.length = 6 ∧ (∃ perms, perms.length = 72) :=
begin
  sorry
end

end exam_sequences_l422_422680


namespace upper_bound_y_l422_422833

theorem upper_bound_y 
  (U : ℤ) 
  (x y : ℤ)
  (h1 : 3 < x ∧ x < 6) 
  (h2 : 6 < y ∧ y < U) 
  (h3 : y - x = 4) : 
  U = 10 := 
sorry

end upper_bound_y_l422_422833


namespace math_problem_l422_422147

-- Definitions for the conditions
variables {p q r x : ℝ}
def condition1 : Prop := p < q
def condition2 : Prop := ∀ x, (x > 2 ∨ (3 ≤ x ∧ x ≤ 5)) ↔ ( ( (x - p)*(x - q) / (x - r) ) ≤ 0 )

-- Statement of the theorem
theorem math_problem : condition1 → condition2 → p + q + 2 * r = 12 :=
by 
  intros h1 h2 
  sorry

end math_problem_l422_422147


namespace raul_money_left_l422_422914

theorem raul_money_left (initial_money : ℕ) (cost_per_comic : ℕ) (number_of_comics : ℕ) (money_left : ℕ)
  (h1 : initial_money = 87)
  (h2 : cost_per_comic = 4)
  (h3 : number_of_comics = 8)
  (h4 : money_left = initial_money - (number_of_comics * cost_per_comic)) :
  money_left = 55 :=
by 
  rw [h1, h2, h3] at h4
  exact h4

end raul_money_left_l422_422914


namespace sphere_radius_l422_422417

theorem sphere_radius (R : ℝ) (h : 4 * Real.pi * R^2 = 4 * Real.pi) : R = 1 :=
sorry

end sphere_radius_l422_422417


namespace common_ratio_of_geometric_seq_l422_422679

theorem common_ratio_of_geometric_seq (a b c d : ℤ) (h1 : a = 25)
    (h2 : b = -50) (h3 : c = 100) (h4 : d = -200)
    (h_geo_1 : b = a * -2)
    (h_geo_2 : c = b * -2)
    (h_geo_3 : d = c * -2) : 
    let r := (-2 : ℤ) in r = -2 := 
by 
  sorry

end common_ratio_of_geometric_seq_l422_422679


namespace trader_excess_donations_l422_422692

-- Define the conditions
def profit : ℤ := 1200
def allocation_percentage : ℤ := 60
def family_donation : ℤ := 250
def friends_donation : ℤ := (20 * family_donation) / 100 + family_donation
def total_family_friends_donation : ℤ := family_donation + friends_donation
def local_association_donation : ℤ := 15 * total_family_friends_donation / 10
def total_donations : ℤ := family_donation + friends_donation + local_association_donation
def allocated_amount : ℤ := allocation_percentage * profit / 100

-- Theorem statement (Question)
theorem trader_excess_donations : total_donations - allocated_amount = 655 :=
by
  sorry

end trader_excess_donations_l422_422692


namespace sum_of_cubes_l422_422454

variable {p q r : ℂ} -- Assume complex roots for generality

-- The polynomial is x^3 - x^2 + x - 2 = 0 with roots p, q, r
def polynomial := (λ x : ℂ, x^3 - x^2 + x - 2)

theorem sum_of_cubes (h1 : polynomial p = 0)
                     (h2 : polynomial q = 0)
                     (h3 : polynomial r = 0)
                     (h4 : p ≠ q)
                     (h5 : q ≠ r)
                     (h6 : r ≠ p) :
                     p^3 + q^3 + r^3 = 4 := by
  sorry

end sum_of_cubes_l422_422454


namespace trig_eq_num_solutions_sum_digits_l422_422923

def trig_eq (x : ℝ) : ℝ :=
  24 * sin (2 * x) + 7 * cos (2 * x) - 36 * sin x - 48 * cos x + 35

def interval_start := 10 ^ (factorial 2014) * Real.pi
def interval_end := 10 ^ (factorial 2014 + 2018) * Real.pi

def N := 3 / 2 * (10 ^ (factorial 2014 + 2018) - 10 ^ (factorial 2014))

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem trig_eq_num_solutions_sum_digits :
  sum_digits (nat_abs (floor (N - (N / 10 ^ 2018)))) = 18162 :=
sorry

end trig_eq_num_solutions_sum_digits_l422_422923


namespace sum_first_2024_terms_l422_422064

theorem sum_first_2024_terms : 
  let a (n : ℕ) := (2 * n - 1) * Real.sin (n * Real.pi / 2)
  let S (n : ℕ) := (∑ i in Finset.range (n + 1), a i)
  in S 2024 = -2024 :=
by
  let a (n : ℕ) := (2 * n - 1) * Real.sin (n * Real.pi / 2)
  let S (n : ℕ) := (∑ i in Finset.range (n + 1), a i)
  show S 2024 = -2024
  sorry

end sum_first_2024_terms_l422_422064


namespace triangle_stick_sum_l422_422049

theorem triangle_stick_sum : 
  (∑ n in Finset.Ico 5 16, n) = 110 := by
  sorry

end triangle_stick_sum_l422_422049


namespace opposite_of_abs_neg2023_l422_422573

def abs_opposite (x : Int) : Int := -|x|

theorem opposite_of_abs_neg2023 : abs_opposite (-2023) = -2023 :=
by
  unfold abs_opposite
  have h : |-2023| = 2023 := abs_neg_of_nat 2023 (by norm_num)
  rw h
  norm_num
  sorry

end opposite_of_abs_neg2023_l422_422573


namespace y_intercept_tangent_line_l422_422069

-- Define the function f and its derivative
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a / Real.exp x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a / Real.exp x

theorem y_intercept_tangent_line (a x₀ : ℝ) (h1 : f' 0 a = 0) 
  (h2 : f' x₀ 1 = (Real.sqrt 2) / 2) : 
  (f x₀ 1) - ( (Real.sqrt 2) / 2 ) * x₀ = (3 * Real.sqrt 2) / 2 - (Real.sqrt 2) / 4 * Real.ln 2 :=
by 
  sorry

end y_intercept_tangent_line_l422_422069


namespace acbd_is_parallelogram_l422_422809

-- Declaration of circles and points
def Circle (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ) : Prop :=
  ∀ (P : EuclideanSpace ℝ (Fin 2)), EuclideanDistance P center = radius

noncomputable def unit_circle (center : EuclideanSpace ℝ (Fin 2)) : Prop :=
  Circle center 1

-- Assume the centers of the circles
variables {O1 O2 O3 O4 A B M N C D : EuclideanSpace ℝ (Fin 2)}

-- Hypotheses based on the problem conditions
axiom h1 : unit_circle O1
axiom h2 : unit_circle O2
axiom h3 : unit_circle O3
axiom h4 : unit_circle O4
axiom intersects : ∀ (P : EuclideanSpace ℝ (Fin 2)), (h1 P → h2 P ↔ (P = A ∨ P = B))
axiom point_on_ω1 : h1 M
axiom point_on_ω2 : h2 N
axiom point_on_ω3 : h3 M
axiom point_on_ω4 : h4 N
axiom second_intersection_1 : h3 C ∧ C ≠ M ∧ h1 C
axiom second_intersection_2 : h4 D ∧ D ≠ N ∧ h2 D

-- The goal statement in Lean
theorem acbd_is_parallelogram : 
  let vect := λ (P Q : EuclideanSpace ℝ (Fin 2)), Q -ᵥ P in
  vect A C = vect D B :=
begin
  sorry
end

end acbd_is_parallelogram_l422_422809


namespace trajectory_proof_l422_422859
open Real

-- Definitions from conditions
def parametric_equation_line_l (t : ℝ) : ℝ × ℝ :=
  (-1/2 * t, 2 + (sqrt 3 / 2) * t)

def rect_circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

def rect_trajectory_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + y^2 = 16

def point_P := (0, 2 : ℝ × ℝ)

-- Problem statement
theorem trajectory_proof :
  (∀ x y : ℝ, (∃ t : ℝ, (x, y) = parametric_equation_line_l t) → rect_trajectory_equation x y) →
  (∀ points_A_B : (ℝ × ℝ) × (ℝ × ℝ), (prod.fst points_A_B, prod.snd points_A_B) ∈ set_of (λ points, (parametric_equation_line_l (prod.fst points), parametric_equation_line_l (prod.snd points))= points_A_B) →
  |prod.fst point_P - prod.fst (prod.fst points_A_B)| + |prod.snd point_P - prod.snd (prod.fst points_A_B)| + |prod.fst point_P - prod.fst (prod.snd points_A_B)| + |prod.snd point_P - prod.snd (prod.snd points_A_B)| = 4 + 2*sqrt 3) :=
by
  sorry

end trajectory_proof_l422_422859


namespace train_passes_jogger_in_l422_422255

-- define the given conditions in Lean
def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def head_start_m : ℝ := 240
def train_length_m : ℝ := 210

-- Define speeds in m/s
def kmh_to_ms (speed : ℝ) : ℝ := (speed * 1000) / 3600
def jogger_speed_ms : ℝ := kmh_to_ms jogger_speed_kmh
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Relative speed in m/s
def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms

-- Total distance that needs to be covered by the train to pass the jogger
def total_distance_m : ℝ := head_start_m + train_length_m

-- Compute the time in seconds
def time_to_pass_jogger : ℝ := total_distance_m / relative_speed_ms

-- Theorem stating the specific time to pass the jogger
theorem train_passes_jogger_in : time_to_pass_jogger = 45 := by
  sorry

end train_passes_jogger_in_l422_422255


namespace sum_even_coeffs_equals_l422_422876

theorem sum_even_coeffs_equals (n : ℕ) :
  let g (x : ℝ) := (1 - x + x^2)^n
  ∃ b : ℕ → ℝ, (∀ k, (g x = ∑ i in range(2*n + 1), b i * x^i)) →
  (∑ k in range(n + 1), b(2 * k)) = (1 + 3^n) / 2 :=
by
  sorry

end sum_even_coeffs_equals_l422_422876


namespace line_cartesian_eq_curve_cartesian_eq_intersection_dist_diff_l422_422860

noncomputable def parametric_line : ℝ × ℝ → ℝ → ℝ × ℝ
| (a, b) t := (-2 + 1/2 * t, sqrt 3 / 2 * t)

def polar_curve (rho theta : ℝ) : Prop :=
  rho * sin theta ^ 2 + 4 * cos theta = 0

noncomputable def cartesian_line (x y : ℝ) : Prop :=
  sqrt 3 * x - y + 2 * sqrt 3 = 0

def cartesian_curve (x y : ℝ) : Prop :=
  y ^ 2 = -4 * x

theorem line_cartesian_eq :
  ∀ t : ℝ, parametric_line (0, 0) t = (x, y) → cartesian_line x y :=
sorry

theorem curve_cartesian_eq :
  ∀ rho theta : ℝ, polar_curve rho theta → cartesian_curve (rho * cos theta) (rho * sin theta) :=
sorry

theorem intersection_dist_diff :
  ∀ t1 t2 : ℝ, 
  (A ≡ (-2 + 1/2 * t1, sqrt 3 / 2 * t1)) ∧ (B ≡ (-2 + 1/2 * t2, sqrt 3 / 2 * t2)) → 
  (cartesian_curve (-2 + 1/2 * t1) (sqrt 3 / 2 * t1)) ∧ (cartesian_curve (-2 + 1/2 * t2) (sqrt 3 / 2 * t2)) →
  |(1 / distance (-2, 0) A) - (1 / distance (-2, 0) B)| = 1/4 :=
sorry

end line_cartesian_eq_curve_cartesian_eq_intersection_dist_diff_l422_422860


namespace mary_total_nickels_l422_422162

-- Define the initial number of nickels Mary had
def mary_initial_nickels : ℕ := 7

-- Define the number of nickels her dad gave her
def mary_received_nickels : ℕ := 5

-- The goal is to prove the total number of nickels Mary has now is 12
theorem mary_total_nickels : mary_initial_nickels + mary_received_nickels = 12 :=
by
  sorry

end mary_total_nickels_l422_422162


namespace shaded_perimeter_is_10_l422_422927

-- Define the problem conditions in Lean
def is_shaded (i j : ℕ) : Prop :=
  (i = 1 ∧ j = 2) ∨ 
  (i = 2 ∧ j = 1) ∨ 
  (i = 2 ∧ j = 3) ∨ 
  (i = 3 ∧ j = 2)

-- Define the statement to be proved
theorem shaded_perimeter_is_10 : (
  let perimeter := 10 in
  ∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 3) ∧ (1 ≤ j ∧ j ≤ 3) →
  if is_shaded i j then
    (perimeter = 10) -- This is the condition we are proving
  else
    true -- For non-shaded cells, the condition is trivially true
) :=
sorry -- Proof omitted for this conversion task

end shaded_perimeter_is_10_l422_422927


namespace part_a_part_b_l422_422644

-- Part (a)
theorem part_a (n : ℕ) (a b : ℝ) : 
  a^(n+1) + b^(n+1) = (a + b) * (a^n + b^n) - a * b * (a^(n - 1) + b^(n - 1)) :=
by sorry

-- Part (b)
theorem part_b {a b : ℝ} (h1 : a + b = 1) (h2: a * b = -1) : 
  a^10 + b^10 = 123 :=
by sorry

end part_a_part_b_l422_422644


namespace lateral_surface_area_of_cone_l422_422062

-- Define the problem data as conditions
def base_radius : ℝ := 3
def volume : ℝ := 12 * Real.pi

-- Define the lateral surface area using the given conditions and prove it
theorem lateral_surface_area_of_cone : 
  (∃ S : ℝ, S = 15 * Real.pi) :=
by
  -- Assuming the base radius r and volume V of the cone, we need to show the lateral surface area S
  let r := base_radius
  let V := volume

  -- From given r = 3 and V = 12 * π, follow the steps to solve for S
  let h := 4
  let l := Real.sqrt (r ^ 2 + h ^ 2)
  let S := Real.pi * r * l

  -- Conclude with the result
  use S = 15 * Real.pi

  sorry

end lateral_surface_area_of_cone_l422_422062


namespace common_ratio_of_geometric_sequence_l422_422490

theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_a1a5 : a 1 * a 5 = 4)
  (h_a4 : a 4 = 1) :
  (∀ n, a n = (a 1) * (q ^ (n - 1))) → q = 1 / 2 := 
begin
  sorry
end

end common_ratio_of_geometric_sequence_l422_422490


namespace isosceles_right_triangle_contains_probability_l422_422846

noncomputable def isosceles_right_triangle_probability : ℝ :=
  let leg_length := 2
  let triangle_area := (leg_length * leg_length) / 2
  let distance_radius := 1
  let quarter_circle_area := (Real.pi * (distance_radius * distance_radius)) / 4
  quarter_circle_area / triangle_area

theorem isosceles_right_triangle_contains_probability :
  isosceles_right_triangle_probability = (Real.pi / 8) :=
by
  sorry

end isosceles_right_triangle_contains_probability_l422_422846


namespace log_function_domain_l422_422939

theorem log_function_domain :
  { x : ℝ | x^2 - 2 * x - 3 > 0 } = { x | x > 3 } ∪ { x | x < -1 } :=
by {
  sorry
}

end log_function_domain_l422_422939


namespace max_value_of_f_l422_422456

noncomputable def f (x : ℝ) := x^3 - 3 * x + 1

theorem max_value_of_f (h: ∃ x, f x = -1) : ∃ y, f y = 3 :=
by
  -- We'll later prove this with appropriate mathematical steps using Lean tactics
  sorry

end max_value_of_f_l422_422456


namespace find_a_l422_422421

-- Define the quadratic function
def f (a x : ℝ) : ℝ := -x^2 + 2 * a * x + (1 - a)

-- Theorem stating the conditions and the conclusion
theorem find_a :
  (∀ x ∈ set.Icc 0 1, f a x ≤ 2) → (∃ x ∈ set.Icc 0 1, f a x = 2) → (a = 2 ∨ a = -1) :=
sorry

end find_a_l422_422421


namespace least_number_to_subtract_l422_422617

theorem least_number_to_subtract (n : ℕ) (h : n = 13294) : ∃ k : ℕ, n - 1 = k * 97 :=
by
  sorry

end least_number_to_subtract_l422_422617


namespace Sn_nine_l422_422158

variables {a_1 d : ℤ} (a_4 a_6 : ℤ) (S_n : ℤ → ℤ)
          (a_n : ℤ → ℤ)

-- Conditions:
def sum_of_roots : Prop := a_4 + a_6 = 18
def arithmetic_seq : Prop := a_4 = a_1 + 3 * d ∧ a_6 = a_1 + 5 * d
def Sn (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- The proof statement to be generated:
theorem Sn_nine : sum_of_roots a_4 a_6 ∧ arithmetic_seq a_4 a_6 a_1 d → S_n 9 = 81 :=
by
  intros
  sorry

end Sn_nine_l422_422158


namespace genetically_modified_microorganisms_percentage_l422_422651

-- Define the given percentages for different categories
def microphotonics : ℝ := 13
def home_electronics : ℝ := 24
def food_additives : ℝ := 15
def industrial_lubricants : ℝ := 8

-- Define the angle of basic astrophysics in degrees
def basic_astrophysics_angle : ℝ := 39.6

-- Define the total circle degrees and total budget percentage
def total_degrees : ℝ := 360
def total_percentage : ℝ := 100

-- Define the percentage of the budget allocated to basic astrophysics based on the angle
def basic_astrophysics_percentage : ℝ := basic_astrophysics_angle / total_degrees * total_percentage

-- Calculate the total known percentage
def total_known_percentage : ℝ := microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics_percentage

-- Define the proof statement
theorem genetically_modified_microorganisms_percentage : total_percentage - total_known_percentage = 29 := by
  -- Here we would place the detailed proof steps
  sorry

end genetically_modified_microorganisms_percentage_l422_422651


namespace minimum_average_from_digits_no_repeats_l422_422374

theorem minimum_average_from_digits_no_repeats :
  ∃ S : set ℕ, S.card = 6 ∧ S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∧ (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → (x % 10 ≠ y % 10 ∧ x / 10 ≠ y / 10))
  ∧ (∀ s1 s2 s3, s1 ∈ { (d1, d2) | d1 ∈ S ∧ d2 ∈ S }
   → s2 ∈ { (d3, d4) | d3 ∈ S ∧ d4 ∈ S }
   → s3 ∈ { (d5, d6) | d5 ∈ S ∧ d6 ∈ S }
   → (s1 ≠ s2 ∧ s2 ≠ s3 ∧ s3 ≠ s1)
   → (s1.1 * 10 + s1.2 + s2.1 * 10 + s2.2 + s3.1 * 10 + s3.2 +
     (∑ x in S, x) - (s1.1 + s1.2 + s2.1 + s2.2 + s3.1 + s3.2)) / 6 = 16.5) := sorry

end minimum_average_from_digits_no_repeats_l422_422374


namespace lines_intersect_at_common_point_l422_422531

theorem lines_intersect_at_common_point 
  (A B C M N L K : Type) [Triangle ABC] 
  (rect_1 : rectangle ABMN) 
  (rect_2 : rectangle LBCK) 
  (acute_ABC : acute_triangle ABC) 
  (congruent_rectangles : congruent_rectangles ABMN LBCK)
  (AB_eq_LB : AB = LB) :
  ∃ P : Type, collinear A L P ∧ collinear C M P ∧ collinear N K P :=
sorry

end lines_intersect_at_common_point_l422_422531


namespace sequence_perfect_square_l422_422316

theorem sequence_perfect_square (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) :
  ∃! n, ∃ k, a n = k ^ 2 :=
by
  sorry

end sequence_perfect_square_l422_422316


namespace average_comparison_l422_422691

variables {a r : ℝ} (a_pos : a ≠ 0)

noncomputable def average_student (x y z : ℝ) : ℝ := 
  let avg_xy := (x + y) / 2 in
  (avg_xy + z) / 2

noncomputable def true_average (x y z : ℝ) : ℝ := 
  (x + y + z) / 3

theorem average_comparison (x y z : ℝ) (a_pos : a ≠ 0) : 
  x = a → y = a * r → z = a * r^2 → 
  let A := true_average x y z in
  let B := average_student x y z in
  (B < A ∨ B > A) :=
begin
  intros hx hy hz,
  let A := true_average x y z,
  let B := average_student x y z,
  have hA : A = (a + a * r + a * r^2) / 3 := by { subst hx, subst hy, subst hz, refl },
  have hB : B = (a + a * r + 2 * a * r^2) / 4 := by { subst hx, subst hy, subst hz, refl },
  suffices : (B - A ≠ 0),
  { exact (lt_or_gt_of_ne this) },
  calc B - A = ((a + a * r + 2* a * r^2) / 4) - ((a + a * r + a * r^2) / 3) : by { subst hA, subst hB }
        ... = ((3 * (a + a * r + 2 * a * r^2) - 4 * (a + a * r + a * r^2)) / 12) 
        : by { field_simp [div_sub_div_same, show (4:ℝ) * (3:ℝ) = 12, by norm_num], ring }
        ... = (ar^2 - a) / 12 
        : by { ring },
  by_contradiction h,
  have : (ar^2 - a) / 12 = 0 := by simp [h], 
  have : ar^2 = a := by { field_simp at this, linarith },
  have : r^2 = 1 := by { field_simp at this, exact (eq_div_iff a_pos a_pos).mpr this },
  have := pow_one r ^ 2, 
  exact zero_ne_one (by linarith),
  sorry,
end

end average_comparison_l422_422691


namespace ratio_of_squares_of_chords_eq_ratio_of_projections_l422_422747

theorem ratio_of_squares_of_chords_eq_ratio_of_projections
  (O A B C D B1 C1 : Point)
  (hcircle : Circle O)
  (hA : OnCircumference A hcircle)
  (hAB : IsChord A B hcircle)
  (hAC : IsChord A C hcircle)
  (hD : DiametricallyOpposite A D hcircle)
  (hB1 : Projection B A D B1)
  (hC1 : Projection C A D C1):
  (AB : ℝ) -> (AC : ℝ) -> (AB1 : ℝ) -> (AC1 : ℝ) -> 
  ∃ (AB AC AB1 AC1 : ℝ),
  AB^2 / AC^2 = AB1 / AC1 := by
  sorry

end ratio_of_squares_of_chords_eq_ratio_of_projections_l422_422747


namespace distinct_patterns_4x4_shading_three_squares_l422_422902

-- Define the 4x4 grid
def grid := fin 4 × fin 4

-- Function to count distinct patterns with given symmetry considerations
noncomputable def distinctPatterns (n : ℕ) (symmetry : (grid → grid) → Prop) : ℕ :=
sorry -- Placeholder for the actual function definition considering symmetry

-- The problem statement translated into a Lean theorem
theorem distinct_patterns_4x4_shading_three_squares : distinctPatterns 3 (λ f, (is_rotation f ∨ is_reflection f)) = 11 :=
sorry -- Proof to be filled in

end distinct_patterns_4x4_shading_three_squares_l422_422902


namespace integers_exist_for_eqns_l422_422139

theorem integers_exist_for_eqns (a b c : ℤ) :
  ∃ (p1 q1 r1 p2 q2 r2 : ℤ), 
    a = q1 * r2 - q2 * r1 ∧ 
    b = r1 * p2 - r2 * p1 ∧ 
    c = p1 * q2 - p2 * q1 :=
  sorry

end integers_exist_for_eqns_l422_422139


namespace sqrt_prime_not_sum_of_rationals_and_other_sqrts_l422_422153

theorem sqrt_prime_not_sum_of_rationals_and_other_sqrts 
  (p : ℕ → ℕ) 
  (hp : ∀ i, nat.prime (p i)) 
  (h_distinct: ∀ j k, j ≠ k → p j ≠ p k) :
  ¬ ∃ (a : ℚ) (b : ℚ → ℚ) (s : finset ℕ), 
    sqrt (p (nat.succ s.sup.id)) = 
    a + ∑ i in s, b i * sqrt (p i) := 
sorry

end sqrt_prime_not_sum_of_rationals_and_other_sqrts_l422_422153


namespace star_area_ratio_l422_422286

-- Definitions based on conditions from step a
def circle_radius : ℝ := 3
def circle_area : ℝ := Real.pi * circle_radius ^ 2
def sector_area : ℝ := (1 / 6) * circle_area
def triangle_area : ℝ := (1 / 2) * circle_radius * circle_radius
def star_segment_area : ℝ := sector_area - triangle_area
def star_area : ℝ := 6 * star_segment_area
def area_ratio : ℝ := star_area / circle_area

-- Theorem statement
theorem star_area_ratio (r : ℝ) (h1 : r = circle_radius) (h2 : star_area = 9 * Real.pi - 27) : 
  area_ratio = (Real.pi - 3) / Real.pi := 
sorry

end star_area_ratio_l422_422286


namespace inequality_solution_l422_422926

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15)
  ↔ (x ∈ set.Ioo 1 2 ∪ set.Ioo 3 6 ∪ set.Ioo 8 10) :=
sorry

end inequality_solution_l422_422926


namespace memorable_numbers_count_l422_422344

def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

def is_memorable (d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ d₉ : ℕ) : Prop :=
  (d₁ = d₄ ∧ d₂ = d₅ ∧ d₃ = d₆) ∨ (d₁ = d₇ ∧ d₂ = d₈ ∧ d₃ = d₉)

theorem memorable_numbers_count :
  (finset.univ.filter (λ n : fin (10^9),
    let d₁ := n / 10^8 % 10,
        d₂ := n / 10^7 % 10,
        d₃ := n / 10^6 % 10,
        d₄ := n / 10^5 % 10,
        d₅ := n / 10^4 % 10,
        d₆ := n / 10^3 % 10,
        d₇ := n / 10^2 % 10,
        d₈ := n / 10 % 10,
        d₉ := n % 10
    in is_digit d₁ ∧ is_digit d₂ ∧ is_digit d₃ ∧ is_digit d₄ ∧ is_digit d₅ ∧ is_digit d₆ ∧ is_digit d₇ ∧ is_digit d₈ ∧ is_digit d₉ ∧ is_memorable d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ d₉)).card = 1999000 :=
sorry


end memorable_numbers_count_l422_422344


namespace married_fraction_l422_422475

variable (total_people : ℕ) (fraction_women : ℚ) (max_unmarried_women : ℕ)
variable (fraction_married : ℚ)

theorem married_fraction (h1 : total_people = 80)
                         (h2 : fraction_women = 1/4)
                         (h3 : max_unmarried_women = 20)
                         : fraction_married = 3/4 :=
by
  sorry

end married_fraction_l422_422475


namespace true_converses_of_propositions_l422_422851

/-- If four points are not coplanar, then any three of these points are not collinear -/
def not_coplanar_implies_not_collinear (A B C D : Point) : 
    ¬Coplanar A B C D → ¬Collinear A B C := sorry

/-- If two lines have no common points, then these two lines are skew lines -/
def no_common_points_implies_skew (l1 l2 : Line) : 
    ¬∃ P : Point, lies_on P l1 ∧ lies_on P l2 → Skew l1 l2 := sorry

theorem true_converses_of_propositions (A B C D : Point) (l1 l2 : Line) :
    (¬Collinear A B C → ¬Coplanar A B C D) ∧ (Skew l1 l2 → ¬∃ P : Point, lies_on P l1 ∧ lies_on P l2) := sorry

end true_converses_of_propositions_l422_422851


namespace line_polar_equation_intersection_distance_l422_422126

noncomputable def polar_equation_of_line (t : ℝ) : ℝ × ℝ :=
  let x := t
  let y := Real.sqrt 3 * t
  (x, y)

def polar_curve (ρ θ : ℝ) : ℝ :=
  4 * ρ ^ 2 * Real.cos (2 * θ) - 4 * ρ * Real.sin θ + 3

theorem line_polar_equation (t : ℝ) :
  ∃ θ, θ = π / 3 ∧ ∀ (ρ : ℝ), polar_equation_of_line t = (ρ * Real.cos θ, ρ * Real.sin θ) :=
begin
  sorry
end

theorem intersection_distance :
  ∀ θ, θ = π / 3 → ∃ ρ1 ρ2 : ℝ, 
  (polar_curve ρ1 θ = 0 ∧ polar_curve ρ2 θ = 0) ∧ |ρ1 - ρ2| = 3 :=
begin
  sorry
end

end line_polar_equation_intersection_distance_l422_422126


namespace find_N_l422_422365

noncomputable def satisfies_conditions (N : ℕ) : Prop :=
  ∃ x : ℕ, (N + 25 = (x + 5) ^ 2) ∧ (∀ p : ℕ, prime p → p ∣ N → p = 2 ∨ p = 5)

theorem find_N (N : ℕ) : satisfies_conditions N ↔ N = 200 ∨ N = 2000 :=
by
  sorry

end find_N_l422_422365


namespace standard_eq_ellipse_fixed_point_exists_l422_422401

-- Definitions for the given conditions
def a : ℝ := 2
def b : ℝ := sqrt 3
def e : ℝ := 1 / 2
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Proof goal 1: Standard equation of the ellipse
theorem standard_eq_ellipse (x y : ℝ) : ellipse_eq x y ↔ x^2 / 4 + y^2 / 3 = 1 := by
  sorry

-- Coordinates for point P and line l
variable (t : ℝ)
def P : ℝ × ℝ := (1, t)
def is_perpendicular (l1 l2 : ℝ × ℝ → Prop) := ∃ k1 k2 : ℝ, (∀ p, l1 p = p.1 - k1 * p.2 = 0) ∧ (∀ p, l2 p = p.1 - k2 * p.2 = 0) ∧ k1 * k2 = -1
def line_l (x y : ℝ) : Prop := y = - (x - 1) / k

-- Proof goal 2: Fixed point coordinates
theorem fixed_point_exists : ∀ t, ∃ (x_fp y_fp : ℝ), (line_l x_fp y_fp) :=
  by
  use (1/4, 0)
  sorry

end standard_eq_ellipse_fixed_point_exists_l422_422401


namespace mark_brought_in_4_times_more_cans_l422_422303

theorem mark_brought_in_4_times_more_cans (M J R : ℕ) (h1 : M = 100) 
  (h2 : J = 2 * R + 5) (h3 : M + J + R = 135) : M / J = 4 :=
by sorry

end mark_brought_in_4_times_more_cans_l422_422303


namespace arithmetic_sequence_sum_l422_422488

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l422_422488


namespace possible_degrees_remainder_l422_422992

theorem possible_degrees_remainder (p : Polynomial ℝ) : 
  ∀ q : Polynomial ℝ, degree q = 3 → (∀ r : Polynomial ℝ, r.degree < 3) := 
sorry

end possible_degrees_remainder_l422_422992


namespace age_contradiction_l422_422178

-- Given the age ratios and future age of Sandy
def current_ages (x : ℕ) : ℕ × ℕ × ℕ := (4 * x, 3 * x, 5 * x)
def sandy_age_after_6_years (age_sandy_current : ℕ) : ℕ := age_sandy_current + 6

-- Given conditions
def ratio_condition (x : ℕ) (age_sandy age_molly age_danny : ℕ) : Prop :=
  current_ages x = (age_sandy, age_molly, age_danny)

def sandy_age_condition (age_sandy_current : ℕ) : Prop :=
  sandy_age_after_6_years age_sandy_current = 30

def age_sum_condition (age_molly age_danny : ℕ) : Prop :=
  age_molly + age_danny = (age_molly + 4) + (age_danny + 4)

-- Main theorem
theorem age_contradiction : ∃ x age_sandy age_molly age_danny, 
  ratio_condition x age_sandy age_molly age_danny ∧
  sandy_age_condition age_sandy ∧
  (¬ age_sum_condition age_molly age_danny) := 
by
  -- Omitting the proof; the focus is on setting up the statement only
  sorry

end age_contradiction_l422_422178


namespace probability_hyperbola_check_l422_422295

def roll_die := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

theorem probability_hyperbola_check :
  let outcomes := { (m, n) | m ∈ roll_die ∧ n ∈ roll_die }
  let on_hyperbola := { (m, n) | m ∈ roll_die ∧ n ∈ roll_die ∧ n = 1 / m }
  let probability := on_hyperbola.card.to_real / outcomes.card.to_real
  in probability = 1 / 6 :=
by
  sorry

end probability_hyperbola_check_l422_422295


namespace tan_theta_geq_one_range_of_f_theta_l422_422050

variable (ABC : Type) [Triangle ABC]
variable (AB AC : Vector ABC)
variable (π : Real)
variable (θ : π / 4 ≤ θ ∧ θ < π / 2)
variable (sin cos tan : Real → Real)

-- Conditions
theorem tan_theta_geq_one 
  (area_triangle : (1 / 2) * (Vector.magnitude AB) * (Vector.magnitude AC) * sin θ = 2)
  (dot_product_cond : 0 < Vector.dot_product AB AC ∧ Vector.dot_product AB AC ≤ 4) :
  1 ≤ tan θ :=
sorry

-- Conditions for function range
theorem range_of_f_theta 
  (H : tan θ ≥ 1) :
  2 ≤ 2 * sin^2(π / 4 + θ) - sqrt 3 * cos (2 * θ) ∧ 2 * sin^2(π / 4 + θ) - sqrt 3 * cos (2 * θ) ≤ 3 :=
sorry

end tan_theta_geq_one_range_of_f_theta_l422_422050


namespace armistice_day_is_wednesday_l422_422193

-- Define the starting date
def start_day : Nat := 5 -- 5 represents Friday if we consider 0 = Sunday

-- Define the number of days after which armistice was signed
def days_after : Nat := 2253

-- Define the target day (Wednesday = 3)
def expected_day : Nat := 3

-- Define the function to calculate the day of the week after a number of days
def day_after_n_days (start_day : Nat) (n : Nat) : Nat :=
  (start_day + n) % 7

-- Define the theorem to prove the equivalent mathematical problem
theorem armistice_day_is_wednesday : day_after_n_days start_day days_after = expected_day := by
  sorry

end armistice_day_is_wednesday_l422_422193


namespace student_travel_fraction_by_bus_l422_422690

theorem student_travel_fraction_by_bus:
  let total_distance : ℝ := 90
    , distance_by_foot := total_distance / 5
    , distance_by_car := 12
    , distance_by_bus := total_distance - distance_by_foot - distance_by_car
    , fraction_by_bus := distance_by_bus / total_distance in
  fraction_by_bus = (2 : ℝ) / 3 :=
by
  sorry

end student_travel_fraction_by_bus_l422_422690


namespace paint_required_for_1000_smaller_statues_l422_422820

theorem paint_required_for_1000_smaller_statues :
  ∀ (paint6ft : ℝ) (height6ft height2ft : ℝ) (num_statues : ℕ),
    paint6ft = 1 → height6ft = 6 → height2ft = 2 → num_statues = 1000 →
    let volume_ratio := (height2ft / height6ft) ^ 3 in
    paint6ft * volume_ratio * num_statues = 37 :=
begin
  sorry
end

end paint_required_for_1000_smaller_statues_l422_422820


namespace determine_n_eq_1_l422_422357

theorem determine_n_eq_1 :
  ∃ n : ℝ, (∀ x : ℝ, (x = 2 → (x^3 - 3*x^2 + n = 2*x^3 - 6*x^2 + 5*n))) → n = 1 :=
by
  sorry

end determine_n_eq_1_l422_422357


namespace coin_genuine_or_counterfeit_l422_422696

theorem coin_genuine_or_counterfeit (coins : ℕ) (counterfeit_coins : ℕ) (specified_coin_is_counterfeit : Bool)
    (weight_diff : ℤ) (balance_scale : ℤ) (even_weight_diff : coins = 101 ∧ counterfeit_coins = 50 ∧ weight_diff % 2 = 0 → 
    ¬ specified_coin_is_counterfeit ∧ coins = 101 ∧ counterfeit_coins = 50 ∧ weight_diff % 2 = 1 → specified_coin_is_counterfeit) :
  ∃ specified_coin_is_genuine, (specified_coin_is_genuine ↔ even_weight_diff) := by
  sorry

end coin_genuine_or_counterfeit_l422_422696


namespace volume_of_polyhedron_l422_422708

theorem volume_of_polyhedron (s : ℝ) : 
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  volume = (Real.sqrt 3 / 2) * s^3 :=
by
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  show volume = (Real.sqrt 3 / 2) * s^3
  sorry

end volume_of_polyhedron_l422_422708


namespace arithmetic_sequence_15th_term_l422_422944

theorem arithmetic_sequence_15th_term (a1 a2 a3 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 3) (h2 : a2 = 14) (h3 : a3 = 25) (h4 : d = a2 - a1) (h5 : a2 - a1 = a3 - a2) (h6 : n = 15) :
  a1 + (n - 1) * d = 157 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_15th_term_l422_422944


namespace incorrect_statement_proof_l422_422997

def SequentialStructureExists : Prop := ∀ (alg : Algorithm), alg.hasSequentialStructure
def LoopStructureDef : Prop := ∀ (alg : Algorithm), alg.loopStructure → alg.selectionStructure
def StatementIncorrect : Prop := (IncorrectStatement() = "C")

theorem incorrect_statement_proof
  (H1 : SequentialStructureExists)
  (H2 : LoopStructureDef)
  (H3 : StatementIncorrect) : False := 
sorry

end incorrect_statement_proof_l422_422997


namespace find_f_log2_3_l422_422422

noncomputable def f : ℝ → ℝ
| x => if x ≥ 4 then (1/2)^x else f (x + 1)

theorem find_f_log2_3 :
  f (Real.log 3 / Real.log 2) = 1 / 24 :=
by
  -- Sorry to skip the proof.
  sorry

end find_f_log2_3_l422_422422


namespace geometric_sequence_common_ratio_l422_422670

theorem geometric_sequence_common_ratio 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 25) 
  (h2 : a2 = -50) 
  (h3 : a3 = 100) 
  (h4 : a4 = -200)
  (h_geometric : a2 / a1 = a3 / a2 ∧ a3 / a2 = a4 / a3) : 
  a2 / a1 = -2 :=
by 
  have r1 : a2 / a1 = -2, sorry
  -- additional steps to complete proof here
  exact r1

end geometric_sequence_common_ratio_l422_422670


namespace system_of_linear_equations_with_integer_solutions_l422_422247

theorem system_of_linear_equations_with_integer_solutions :
  ∃ (x y : ℤ), (x + y = 5) ∧ (2 * x + y = 7) := by
  -- We need to find integers x and y that satisfy the system of equations
  use 2, 3
  -- Show that these satisfy the first equation
  split
  rfl
  -- Show that these satisfy the second equation
  rfl

end system_of_linear_equations_with_integer_solutions_l422_422247


namespace martin_wastes_time_l422_422161

def t1 := 75
def t2 := 4 * t1
def t3 := 25
def t4 := 40
def t5 := 75
def t6 := 45
def t7 := 5
def t8 := 5
def t9 := 20.5
def t10 := 40.75

def total_time_wasted := t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10

theorem martin_wastes_time : total_time_wasted = 632.25 := 
by
  sorry -- Proof not required as per instructions

end martin_wastes_time_l422_422161


namespace normal_vector_proof_l422_422818

-- Define the 3D vector type
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a specific normal vector n
def n : Vector3D := ⟨1, -2, 2⟩

-- Define the vector v we need to prove is a normal vector of the same plane
def v : Vector3D := ⟨2, -4, 4⟩

-- Define the statement (without the proof)
theorem normal_vector_proof : v = ⟨2 * n.x, 2 * n.y, 2 * n.z⟩ :=
by
  sorry

end normal_vector_proof_l422_422818


namespace gun_drawing_line_creates_line_opened_folding_fan_creates_surface_l422_422299

def Point {α : Type*} := α
def Line {α : Type*} := set α
def Surface {α : Type*} := set (set α)

def GunDrawingLine (p: Point ℝ) : (p → Line ℝ) := 
λ p, {x | x = p}

def OpenedFoldingFan (l: Line ℝ) : (l → Surface ℝ) :=
λ l, {s | s = l}

theorem gun_drawing_line_creates_line (p: Point ℝ):
  GunDrawingLine p = {x | x = p} := sorry

theorem opened_folding_fan_creates_surface (l: Line ℝ):
  OpenedFoldingFan l = {s | s = l} := sorry

end gun_drawing_line_creates_line_opened_folding_fan_creates_surface_l422_422299


namespace triangle_side_a_calculation_l422_422469

theorem triangle_side_a_calculation
  (A : ℝ) (AB : ℝ) (area : ℝ) (a : ℝ)
  (hA : A = π / 3)
  (hAB : AB = 4)
  (harea : area = 2 * sqrt 3) :
  a = 2 * sqrt 3 :=
sorry

end triangle_side_a_calculation_l422_422469


namespace relationship_between_a_b_c_l422_422749

variables (a b c : ℝ)

noncomputable def a_def : ℝ := Real.log 0.2 / Real.log 3
noncomputable def b_def : ℝ := 3 ^ 0.2
noncomputable def c_def : ℝ := 0.3 ^ 0.2

theorem relationship_between_a_b_c (ha : a = a_def) (hb : b = b_def) (hc : c = c_def) :
  b > c ∧ c > a :=
by {
  sorry
}

end relationship_between_a_b_c_l422_422749


namespace tin_amount_new_alloy_l422_422273

-- We will define the given conditions
def alloy_a_mass : ℝ := 120
def alloy_b_mass : ℝ := 180
def ratio_lead_tin_a : ℝ := 2 / 3
def ratio_tin_copper_b : ℝ := 3 / 8

-- Calculation of amount of tin in alloy A
def tin_in_a (m : ℝ) : ℝ :=
  (3 / 5) * m

-- Calculation of amount of tin in alloy B
def tin_in_b (m : ℝ) : ℝ :=
  (3 / 8) * m

-- Total amount of tin in the new alloy
def total_tin (a_tin : ℝ) (b_tin : ℝ) : ℝ :=
  a_tin + b_tin

-- Now, we pose the proposition that we need to prove given the conditions
theorem tin_amount_new_alloy :
  total_tin (tin_in_a alloy_a_mass) (tin_in_b alloy_b_mass) = 139.5 :=
by
  sorry

end tin_amount_new_alloy_l422_422273


namespace graph_equation_l422_422619

theorem graph_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end graph_equation_l422_422619


namespace prove_a_le_minus_2sqrt2_l422_422458

theorem prove_a_le_minus_2sqrt2 
  (f : ℝ → ℝ)
  (a x1 x2 : ℝ)
  (h1 : f = (λ x, Real.log x + (1/2) * x^2 + a * x))
  (h2 : f' = (λ x, (1 / x) + x + a))
  (h3 : f' x1 = 0 ∧ f' x2 = 0)
  (h4 : Real.log x1 + (1/2) * x1^2 + a * x1
          + Real.log x2 + (1/2) * x2^2 + a * x2 ≤ -5)
  (h5 : x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ x1 * x2 = 1 ∧ -a > 0) :
  a ≤ -2 * Real.sqrt 2 := 
sorry

end prove_a_le_minus_2sqrt2_l422_422458


namespace pencils_remaining_in_drawer_l422_422224

theorem pencils_remaining_in_drawer :
  ∀ (initial_pencils : ℕ) (taken_pencils : ℕ),
  initial_pencils = 34 → taken_pencils = 22 → (initial_pencils - taken_pencils) = 12 :=
begin
  intros initial_pencils taken_pencils h_initial h_taken,
  rw h_initial,
  rw h_taken,
  norm_num,
end

end pencils_remaining_in_drawer_l422_422224
