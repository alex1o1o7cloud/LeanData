import Mathlib

namespace distance_to_fourth_buoy_l178_178941

theorem distance_to_fourth_buoy
  (buoy_interval_distance : ℕ)
  (total_distance_to_third_buoy : ℕ)
  (h : total_distance_to_third_buoy = buoy_interval_distance * 3) :
  (buoy_interval_distance * 4 = 96) :=
by
  sorry

end distance_to_fourth_buoy_l178_178941


namespace cells_after_3_hours_l178_178580

noncomputable def cell_division_problem (t : ℕ) : ℕ :=
  2 ^ (t * 2)

theorem cells_after_3_hours : cell_division_problem 3 = 64 := by
  sorry

end cells_after_3_hours_l178_178580


namespace probability_at_least_one_expired_l178_178096

theorem probability_at_least_one_expired (total_bottles : ℕ) (expired_bottles : ℕ)
  (selection_size : ℕ) (prob_both_unexpired : ℚ) :
  total_bottles = 30 →
  expired_bottles = 3 →
  selection_size = 2 →
  prob_both_unexpired = 351 / 435 →
  (1 - prob_both_unexpired) = 28 / 145 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end probability_at_least_one_expired_l178_178096


namespace shifted_parabola_eq_l178_178014

-- Definitions
def original_parabola (x y : ℝ) : Prop := y = 3 * x^2

def shifted_origin (x' y' x y : ℝ) : Prop :=
  (x' = x + 1) ∧ (y' = y + 1)

-- Target statement
theorem shifted_parabola_eq : ∀ (x y x' y' : ℝ),
  original_parabola x y →
  shifted_origin x' y' x y →
  y' = 3*(x' - 1)*(x' - 1) + 1 → 
  y = 3*(x + 1)*(x + 1) - 1 :=
by
  intros x y x' y' h_orig h_shifted h_new_eq
  sorry

end shifted_parabola_eq_l178_178014


namespace Sheelas_monthly_income_l178_178745

theorem Sheelas_monthly_income (I : ℝ) (h : 0.32 * I = 3800) : I = 11875 :=
by
  sorry

end Sheelas_monthly_income_l178_178745


namespace right_triangle_x_value_l178_178628

theorem right_triangle_x_value (x Δ : ℕ) (h₁ : x > 0) (h₂ : Δ > 0) :
  ((x + 2 * Δ)^2 = x^2 + (x + Δ)^2) → 
  x = (Δ * (-1 + 2 * Real.sqrt 7)) / 2 := 
sorry

end right_triangle_x_value_l178_178628


namespace even_func_min_value_l178_178304

theorem even_func_min_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_neq_a : a ≠ 1) (h_neq_b : b ≠ 1) (h_even : ∀ x : ℝ, a^x + b^x = a^(-x) + b^(-x)) :
  ab = 1 → (∃ y : ℝ, y = (1 / a + 4 / b) ∧ y = 4) :=
by
  sorry

end even_func_min_value_l178_178304


namespace quadratic_complete_square_l178_178346

theorem quadratic_complete_square (c r s k : ℝ) (h1 : 8 * k^2 - 6 * k + 16 = c * (k + r)^2 + s) 
  (h2 : c = 8) 
  (h3 : r = -3 / 8) 
  (h4 : s = 119 / 8) : 
  s / r = -119 / 3 := 
by 
  sorry

end quadratic_complete_square_l178_178346


namespace stratified_sampling_females_l178_178785

theorem stratified_sampling_females :
  let total_employees := 200
  let male_employees := 120
  let female_employees := 80
  let sample_size := 20
  number_of_female_in_sample = (female_employees / total_employees) * sample_size := by
  sorry

end stratified_sampling_females_l178_178785


namespace points_with_tangent_length_six_l178_178554

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 4 = 0

-- Define the property of a point having a tangent of length 6 to the circle
def tangent_length_six (h k cx cy r : ℝ) : Prop :=
  (cx - h)^2 + (cy - k)^2 - r^2 = 36

-- Main theorem statement
theorem points_with_tangent_length_six : 
  (∀ x1 y1 : ℝ, (x1 = -4 ∧ y1 = 6) ∨ (x1 = 5 ∧ y1 = -3) → 
    (∃ r1 : ℝ, tangent_length_six x1 y1 (-1) 0 3) ∧ 
    (∃ r2 : ℝ, tangent_length_six x1 y1 2 3 3)) :=
  by 
  sorry

end points_with_tangent_length_six_l178_178554


namespace medical_team_formation_l178_178347

theorem medical_team_formation (m f : ℕ) (h_m : m = 5) (h_f : f = 4) :
  (m + f).choose 3 - m.choose 3 - f.choose 3 = 70 :=
by
  sorry

end medical_team_formation_l178_178347


namespace colorings_without_two_corners_l178_178903

def valid_colorings (n: ℕ) (exclude_cells : Finset (Fin n × Fin n)) : ℕ := sorry

theorem colorings_without_two_corners :
  valid_colorings 5 ∅ = 120 →
  valid_colorings 5 {(0, 0)} = 96 →
  valid_colorings 5 {(0, 0), (4, 4)} = 78 :=
by {
  sorry
}

end colorings_without_two_corners_l178_178903


namespace rectangle_area_l178_178529

theorem rectangle_area (P W : ℝ) (hP : P = 52) (hW : W = 11) :
  ∃ A L : ℝ, (2 * L + 2 * W = P) ∧ (A = L * W) ∧ (A = 165) :=
by
  sorry

end rectangle_area_l178_178529


namespace LCM_is_4199_l178_178563

theorem LCM_is_4199 :
  let beats_of_cymbals := 13
  let beats_of_triangle := 17
  let beats_of_tambourine := 19
  Nat.lcm (Nat.lcm beats_of_cymbals beats_of_triangle) beats_of_tambourine = 4199 := 
by 
  sorry 

end LCM_is_4199_l178_178563


namespace sqrt_meaningful_range_l178_178981

theorem sqrt_meaningful_range (x : ℝ) (h : 3 * x - 5 ≥ 0) : x ≥ 5 / 3 :=
sorry

end sqrt_meaningful_range_l178_178981


namespace chick_hits_at_least_five_l178_178776

theorem chick_hits_at_least_five (x y z : ℕ) (h1 : 9 * x + 5 * y + 2 * z = 61) (h2 : x + y + z = 10) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : x ≥ 5 :=
sorry

end chick_hits_at_least_five_l178_178776


namespace remainder_31_l178_178980

theorem remainder_31 (x : ℤ) (h : x % 62 = 7) : (x + 11) % 31 = 18 := by
  sorry

end remainder_31_l178_178980


namespace number_of_black_ribbons_l178_178706

theorem number_of_black_ribbons (total_ribbons : ℕ)
  (yellow_fraction : ℚ) (purple_fraction : ℚ) (orange_fraction : ℚ) (black_fraction : ℚ)
  (silver_ribbons : ℕ) (H1 : yellow_fraction = 1/4)
  (H2 : purple_fraction = 1/3)
  (H3 : orange_fraction = 1/6)
  (H4 : black_fraction = 1/12)
  (H5 : silver_ribbons = 40)
  (H6 : (yellow_fraction + purple_fraction + orange_fraction + black_fraction) + (silver_ribbons / total_ribbons) = 1) :
  (black_fraction * total_ribbons).toNat = 20 := 
by
  sorry -- Proof goes here

end number_of_black_ribbons_l178_178706


namespace find_p_range_l178_178442

theorem find_p_range (p : ℝ) (A : ℝ → ℝ) :
  (A = fun x => abs x * x^2 + (p + 2) * x + 1) →
  (∀ x, 0 < x → A x ≠ 0) →
  (-4 < p ∧ p < 0) :=
by
  intro hA h_no_pos_roots
  sorry

end find_p_range_l178_178442


namespace minimum_value_of_3a_plus_b_l178_178824

theorem minimum_value_of_3a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 2) : 
  3 * a + b ≥ (7 + 2 * Real.sqrt 6) / 2 :=
sorry

end minimum_value_of_3a_plus_b_l178_178824


namespace min_students_changed_l178_178593

-- Define the initial percentage of "Yes" and "No" at the beginning of the year
def initial_yes_percentage : ℝ := 0.40
def initial_no_percentage : ℝ := 0.60

-- Define the final percentage of "Yes" and "No" at the end of the year
def final_yes_percentage : ℝ := 0.80
def final_no_percentage : ℝ := 0.20

-- Define the minimum possible percentage of students that changed their mind
def min_changed_percentage : ℝ := 0.40

-- Prove that the minimum possible percentage of students that changed their mind is 40%
theorem min_students_changed :
  (final_yes_percentage - initial_yes_percentage = min_changed_percentage) ∧
  (initial_yes_percentage = final_yes_percentage - min_changed_percentage) ∧
  (initial_no_percentage - min_changed_percentage = final_no_percentage) :=
by
  sorry

end min_students_changed_l178_178593


namespace min_sum_of_factors_l178_178997

theorem min_sum_of_factors (a b : ℤ) (h1 : a * b = 72) : a + b ≥ -73 :=
sorry

end min_sum_of_factors_l178_178997


namespace smallest_non_factor_product_of_100_l178_178369

/-- Let a and b be distinct positive integers that are factors of 100. 
    The smallest value of their product which is not a factor of 100 is 8. -/
theorem smallest_non_factor_product_of_100 (a b : ℕ) (hab : a ≠ b) (ha : a ∣ 100) (hb : b ∣ 100) (hprod : ¬ (a * b ∣ 100)) : a * b = 8 :=
sorry

end smallest_non_factor_product_of_100_l178_178369


namespace cubes_sum_l178_178214

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l178_178214


namespace problem_1_problem_2_l178_178295

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * a * x^2 + 2 * x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a
noncomputable def h' (x : ℝ) (a : ℝ) : ℝ := (1 / x) - a * x - 2
noncomputable def G (x : ℝ) : ℝ := ((1 / x) - 1) ^ 2 - 1

theorem problem_1 (a : ℝ): 
  (∃ x : ℝ, 0 < x ∧ h' x a < 0) ↔ a > -1 :=
by sorry

theorem problem_2 (a : ℝ):
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → h' x a ≤ 0) ↔ a ≥ -(7 / 16) :=
by sorry

end problem_1_problem_2_l178_178295


namespace calc_expression_l178_178265

theorem calc_expression : 
  |1 - Real.sqrt 2| - Real.sqrt 8 + (Real.sqrt 2 - 1)^0 = -Real.sqrt 2 :=
by
  sorry

end calc_expression_l178_178265


namespace wall_width_l178_178244

theorem wall_width (w h l : ℝ)
  (h_eq_6w : h = 6 * w)
  (l_eq_7h : l = 7 * h)
  (V_eq : w * h * l = 86436) :
  w = 7 :=
by
  sorry

end wall_width_l178_178244


namespace tetrahedron_edge_length_l178_178123

-- Define the problem specifications
def mutuallyTangent (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop) :=
  a = b ∧ a = c ∧ a = d ∧ b = c ∧ b = d ∧ c = d

noncomputable def tetrahedronEdgeLength (r : ℝ) : ℝ :=
  2 + 2 * Real.sqrt 6

-- Proof goal: edge length of tetrahedron containing four mutually tangent balls each of radius 1
theorem tetrahedron_edge_length (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop)
  (h1 : r = 1)
  (h2 : mutuallyTangent r a b c d)
  : tetrahedronEdgeLength r = 2 + 2 * Real.sqrt 6 :=
sorry

end tetrahedron_edge_length_l178_178123


namespace lines_in_4_by_4_grid_l178_178664

/--
In a 4-by-4 grid of lattice points, the number of different lines that pass through at least two points is 30.
-/
theorem lines_in_4_by_4_grid : 
  ∃ lines : ℕ, lines = 30 ∧ (∀ pts : fin 4 × fin 4, ∃ l : Set (fin 4 × fin 4), 
  ∀ p1 p2 : fin 4 × fin 4, p1 ∈ pts → p2 ∈ pts → p1 ≠ p2 → p1 ∈ l ∧ p2 ∈ l) := 
sorry

end lines_in_4_by_4_grid_l178_178664


namespace lines_in_4_by_4_grid_l178_178663

/--
In a 4-by-4 grid of lattice points, the number of different lines that pass through at least two points is 30.
-/
theorem lines_in_4_by_4_grid : 
  ∃ lines : ℕ, lines = 30 ∧ (∀ pts : fin 4 × fin 4, ∃ l : Set (fin 4 × fin 4), 
  ∀ p1 p2 : fin 4 × fin 4, p1 ∈ pts → p2 ∈ pts → p1 ≠ p2 → p1 ∈ l ∧ p2 ∈ l) := 
sorry

end lines_in_4_by_4_grid_l178_178663


namespace num_quarters_l178_178755

theorem num_quarters (n q : ℕ) (avg_initial avg_new : ℕ) 
  (h1 : avg_initial = 10) 
  (h2 : avg_new = 12) 
  (h3 : avg_initial * n + 10 = avg_new * (n + 1)) :
  q = 1 :=
by {
  sorry
}

end num_quarters_l178_178755


namespace most_reasonable_plan_l178_178154

-- Defining the conditions as a type
inductive SurveyPlans
| A -- Surveying students in the second grade of School B
| C -- Randomly surveying 150 teachers
| B -- Surveying 600 students randomly selected from School C
| D -- Randomly surveying 150 students from each of the four schools

-- Define the main theorem asserting that the most reasonable plan is Option D
theorem most_reasonable_plan : SurveyPlans.D = SurveyPlans.D :=
by
  sorry

end most_reasonable_plan_l178_178154


namespace tan_alpha_second_quadrant_complex_expression_l178_178440

theorem tan_alpha_second_quadrant (α : Real) (h1 : sin α = 3 / 5) (h2 : π / 2 < α ∧ α < π) :
  tan α = -3 / 4 :=
sorry

theorem complex_expression (α : Real) (h1 : sin α = 3 / 5) (h2 : π / 2 < α ∧ α < π)
  (h3 : cos α = -4 / 5) :
  (2 * sin α + 3 * cos α) / (cos α - sin α) = 6 / 7 :=
sorry

end tan_alpha_second_quadrant_complex_expression_l178_178440


namespace range_of_d_l178_178033

variable {S : ℕ → ℝ} -- S is the sum of the series
variable {a : ℕ → ℝ} -- a is the arithmetic sequence

theorem range_of_d (d : ℝ) (h1 : a 3 = 12) (h2 : S 12 > 0) (h3 : S 13 < 0) :
  -24 / 7 < d ∧ d < -3 := sorry

end range_of_d_l178_178033


namespace distance_between_cities_l178_178510

theorem distance_between_cities
  (S : ℤ)
  (h1 : ∀ x : ℤ, 0 ≤ x ∧ x ≤ S → (gcd x (S - x) = 1 ∨ gcd x (S - x) = 3 ∨ gcd x (S - x) = 13)) :
  S = 39 := 
sorry

end distance_between_cities_l178_178510


namespace simplified_expression_at_3_l178_178746

noncomputable def simplify_and_evaluate (x : ℝ) : ℝ :=
  (3 * x ^ 2 + 8 * x - 6) - (2 * x ^ 2 + 4 * x - 15)

theorem simplified_expression_at_3 : simplify_and_evaluate 3 = 30 :=
by
  sorry

end simplified_expression_at_3_l178_178746


namespace stephan_cannot_afford_laptop_l178_178874

noncomputable def initial_laptop_price : ℝ := sorry

theorem stephan_cannot_afford_laptop (P₀ : ℝ) (h_rate : 0 < 0.06) (h₁ : initial_laptop_price = P₀) : 
  56358 < P₀ * (1.06)^2 :=
by 
  sorry

end stephan_cannot_afford_laptop_l178_178874


namespace power_of_integer_is_two_l178_178149

-- Definitions based on conditions
def is_power_of_integer (n : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ), n = m^k

-- Given conditions translated to Lean definitions
def g : ℕ := 14
def n : ℕ := 3150 * g

-- The proof problem statement in Lean
theorem power_of_integer_is_two (h : g = 14) : is_power_of_integer n :=
sorry

end power_of_integer_is_two_l178_178149


namespace probability_real_cos_sin_l178_178884

def rational_set : Finset ℚ := {q | ∃ n d : ℤ, 0 ≤ n ∧ n < 3 * d ∧ 1 ≤ d ∧ d ≤ 7 ∧ q = (n : ℚ) / d}.to_finset

def special_rationals : Finset ℚ := rational_set.filter (λ q, q ∈ set.Ico (0 : ℚ) 3)

def a_b_possible_pairs : Finset (ℚ × ℚ) := special_rationals.product special_rationals

def is_real_cos_sin_expression (a b : ℚ) : Bool :=
  let x := Real.cos (a * Real.pi)
  let y := Real.sin (b * Real.pi)
  ((4 * x^3 * y - 4 * x * y^3) = 0).to_bool

theorem probability_real_cos_sin :
  ∃ p : ℚ, ∀ (a b : ℚ), (a, b) ∈ a_b_possible_pairs →
    (is_real_cos_sin_expression a b = true ↔ p = (Rational.count (λ (a, b) : ℚ × ℚ, is_real_cos_sin_expression a b = true) / rational_set.card ^ 2)) :=
begin
  sorry
end

end probability_real_cos_sin_l178_178884


namespace sum_of_cubes_l178_178232

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178232


namespace pentagon_area_inequality_l178_178343

-- Definitions for the problem
structure Point :=
(x y : ℝ)

structure Triangle :=
(A B C : Point)

noncomputable def area (T : Triangle) : ℝ :=
  1 / 2 * abs ((T.B.x - T.A.x) * (T.C.y - T.A.y) - (T.C.x - T.A.x) * (T.B.y - T.A.y))

structure Pentagon :=
(A B C D E : Point)

noncomputable def pentagon_area (P : Pentagon) : ℝ :=
  area ⟨P.A, P.B, P.C⟩ + area ⟨P.A, P.C, P.D⟩ + area ⟨P.A, P.D, P.E⟩ -
  area ⟨P.E, P.B, P.C⟩

-- Given conditions
variables (A B C D E F : Point)
variables (P : Pentagon) 
-- P is a convex pentagon with points A, B, C, D, E in order 

-- Intersection point of AD and EC is F 
axiom intersect_diagonals (AD EC : Triangle) : AD.C = F ∧ EC.B = F

-- Theorem statement
theorem pentagon_area_inequality :
  let AED := Triangle.mk A E D
  let EDC := Triangle.mk E D C
  let EAB := Triangle.mk E A B
  let DCB := Triangle.mk D C B
  area AED + area EDC + area EAB + area DCB > pentagon_area P :=
  sorry

end pentagon_area_inequality_l178_178343


namespace domain_of_f_l178_178895

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (4 * x - 3))

theorem domain_of_f :
  {x : ℝ | 4 * x - 3 > 0 ∧ Real.log (4 * x - 3) ≠ 0} = 
  {x : ℝ | x ∈ Set.Ioo (3 / 4) 1 ∪ Set.Ioi 1} :=
by
  sorry

end domain_of_f_l178_178895


namespace general_term_seq_l178_178958

open Nat

-- Definition of the sequence given conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n, n ≥ 1 → (n - 1) * a (n + 1) - n * a n + 1 = 0

-- To prove that the general term is a_n = n
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end general_term_seq_l178_178958


namespace max_weight_of_crates_on_trip_l178_178259

def max_crates : ℕ := 5
def min_crate_weight : ℕ := 150

theorem max_weight_of_crates_on_trip : max_crates * min_crate_weight = 750 := by
  sorry

end max_weight_of_crates_on_trip_l178_178259


namespace min_f_l178_178166

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x then (x + 1) * Real.log x
else 2 * x + 3

noncomputable def f' (x : ℝ) : ℝ :=
if 0 < x then Real.log x + (x + 1) / x
else 2

theorem min_f'_for_x_pos : ∃ (c : ℝ), c = 2 ∧ ∀ x > 0, f' x ≥ c := 
  sorry

end min_f_l178_178166


namespace find_k_min_value_quadratic_zero_l178_178809

theorem find_k_min_value_quadratic_zero (x y k : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 5 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 10 * x - 6 * y + 9 = 0) ↔ k = 1 :=
by
  sorry

end find_k_min_value_quadratic_zero_l178_178809


namespace alcohol_water_ratio_l178_178368

theorem alcohol_water_ratio (a b : ℚ) (h₀ : a > 0) (h₁ : b > 0) :
  (3 * a / (a + 2) + 8 / (4 + b)) / (6 / (a + 2) + 2 * b / (4 + b)) = (3 * a + 8) / (6 + 2 * b) :=
by
  sorry

end alcohol_water_ratio_l178_178368


namespace line_through_A_area_1_l178_178398

def line_equation : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (y = k * (x + 2) + 2) ↔ 
    (x + 2 * y - 2 = 0 ∨ 2 * x + y + 2 = 0) ∧ 
    (2 * (k * 0 + 2) * (-2 - 2 / k) = 2)

theorem line_through_A_area_1 : line_equation :=
by
  sorry

end line_through_A_area_1_l178_178398


namespace max_e_of_conditions_l178_178613

theorem max_e_of_conditions (a b c d e : ℝ) 
  (h1 : a + b + c + d + e = 8) 
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ (16 / 5) :=
by 
  sorry

end max_e_of_conditions_l178_178613


namespace value_of_fraction_l178_178463

variables (w x y : ℝ)

theorem value_of_fraction (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 :=
sorry

end value_of_fraction_l178_178463


namespace middle_guards_hours_l178_178253

theorem middle_guards_hours (h1 : ∀ (first_guard_hours last_guard_hours total_night_hours : ℕ),
  first_guard_hours = 3 ∧ last_guard_hours = 2 ∧ total_night_hours = 9) : ∃ middle_guard_hours : ℕ,
  let remaining_hours := 9 - (3 + 2) in
  let middle_guard_hours := remaining_hours / 2 in
  middle_guard_hours = 2 := by
  sorry

end middle_guards_hours_l178_178253


namespace students_in_both_clubs_l178_178902

theorem students_in_both_clubs :
  ∀ (total_students drama_club science_club either_club both_club : ℕ),
  total_students = 300 →
  drama_club = 100 →
  science_club = 140 →
  either_club = 220 →
  (drama_club + science_club - both_club = either_club) →
  both_club = 20 :=
by
  intros total_students drama_club science_club either_club both_club
  intros h1 h2 h3 h4 h5
  sorry

end students_in_both_clubs_l178_178902


namespace correct_statements_conclusion_l178_178775

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

def mutually_exclusive (A B : Set Ω) : Prop := (A ∩ B) = ∅

def independence (A B : Set Ω) : Prop := μ(A ∩ B) = μ(A) * μ(B)

def normal_dist (μ σ : ℝ) (X : Ω → ℝ) : Prop := ∀ᵐ ω ∂μ, X ω ~ Gaussian μ σ

def calc_percentile (data : List ℝ) (p : ℝ) : ℝ := sorry

theorem correct_statements_conclusion
    (A B : Set Ω)
    (A_event : measurable_set A)
    (B_event : measurable_set B)
    (X : Ω → ℝ)
    (μ : ℝ) (σ : ℝ)
    (data: List ℝ) :
    (mutually_exclusive A B → ¬ (μ(A) + μ(B) = 1)) ∧
    (independence A B → (0 < μ A) → (0 < μ B) → 
       (μ (B ∩ A) / μ A = μ B) → (μ (A ∩ B) / μ B = μ A)) ∧
    (normal_dist 2 σ X → (μ (set_of (λ (ω : Ω), X ω ≤ 3)) = 0.6) →
       (μ (set_of (λ (ω : Ω), X ω ≤ 1)) = 0.4)) ∧
    (calc_percentile [2,3,4,5,6] 0.6 ≠ 4) :=
by
  sorry

end correct_statements_conclusion_l178_178775


namespace problem_statement_l178_178441

variable {a b c : ℝ}

theorem problem_statement (h : a < b) (hc : c < 0) : ¬ (a * c < b * c) :=
by sorry

end problem_statement_l178_178441


namespace noah_total_watts_used_l178_178170

theorem noah_total_watts_used :
  let bedroom_watts_per_hour := 6
  let office_watts_per_hour := 3 * bedroom_watts_per_hour
  let living_room_watts_per_hour := 4 * bedroom_watts_per_hour
  let hours_on := 2
  let bedroom_total := bedroom_watts_per_hour * hours_on
  let office_total := office_watts_per_hour * hours_on
  let living_room_total := living_room_watts_per_hour * hours_on
  bedroom_total + office_total + living_room_total = 96 :=
by
  -- Define the given conditions as variables
  let bedroom_watts_per_hour := 6
  let office_watts_per_hour := 3 * bedroom_watts_per_hour
  let living_room_watts_per_hour := 4 * bedroom_watts_per_hour
  let hours_on := 2
  
  -- Calculate watts used over two hours
  let bedroom_total := bedroom_watts_per_hour * hours_on
  let office_total := office_watts_per_hour * hours_on
  let living_room_total := living_room_watts_per_hour * hours_on
  
  -- Sum up the totals
  have h1 : bedroom_total = 12 := rfl
  have h2 : office_total = 36 := rfl
  have h3 : living_room_total = 48 := rfl
  have sum_totals : 12 + 36 + 48 = 96 := by norm_num

  -- Conclusion
  show bedroom_total + office_total + living_room_total = 96 from sum_totals

end noah_total_watts_used_l178_178170


namespace probability_A_l178_178530

variable (A B : Prop)
variable (P : Prop → ℝ)

axiom prob_B : P B = 0.4
axiom prob_A_and_B : P (A ∧ B) = 0.15
axiom prob_notA_and_notB : P (¬ A ∧ ¬ B) = 0.5499999999999999

theorem probability_A : P A = 0.20 :=
by sorry

end probability_A_l178_178530


namespace yard_length_l178_178466

theorem yard_length (n : ℕ) (d : ℕ) (k : ℕ) (h : k = n - 1) (hd : d = 5) (hn : n = 51) : (k * d) = 250 := 
by
  sorry

end yard_length_l178_178466


namespace find_number_l178_178759

theorem find_number (n : ℕ) (h : Nat.factorial 4 / Nat.factorial (4 - n) = 24) : n = 3 :=
by
  sorry

end find_number_l178_178759


namespace jenny_profit_l178_178475

-- Definitions for the conditions
def cost_per_pan : ℝ := 10.0
def pans_sold : ℕ := 20
def selling_price_per_pan : ℝ := 25.0

-- Definition for the profit calculation based on the given conditions
def total_revenue : ℝ := pans_sold * selling_price_per_pan
def total_cost : ℝ := pans_sold * cost_per_pan
def profit : ℝ := total_revenue - total_cost

-- The actual theorem statement
theorem jenny_profit : profit = 300.0 := by
  sorry

end jenny_profit_l178_178475


namespace age_weight_not_proportional_l178_178185

theorem age_weight_not_proportional (age weight : ℕ) : ¬(∃ k, ∀ (a w : ℕ), w = k * a → age / weight = k) :=
by
  sorry

end age_weight_not_proportional_l178_178185


namespace distance_between_cities_l178_178513

theorem distance_between_cities (S : ℕ) 
  (h : ∀ x : ℕ, x ≤ S → gcd x (S - x) ∈ {1, 3, 13}) : S = 39 :=
sorry

end distance_between_cities_l178_178513


namespace radius_of_sphere_in_truncated_cone_l178_178408

-- Definitions based on conditions
def radius_top_base := 5
def radius_bottom_base := 24

-- Theorem statement (without proof)
theorem radius_of_sphere_in_truncated_cone :
    (∃ (R_s : ℝ),
      (R_s = Real.sqrt 180.5) ∧
      ∀ (h : ℝ),
      (h^2 + (radius_bottom_base - radius_top_base)^2 = (h + R_s)^2 - R_s^2)) :=
sorry

end radius_of_sphere_in_truncated_cone_l178_178408


namespace find_m_n_sum_l178_178864

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem find_m_n_sum (x₀ : ℝ) (m n : ℤ) 
  (hmn_adj : n = m + 1) 
  (hx₀_zero : f x₀ = 0) 
  (hx₀_interval : (m : ℝ) < x₀ ∧ x₀ < (n : ℝ)) :
  m + n = 1 :=
sorry

end find_m_n_sum_l178_178864


namespace polynomial_remainder_l178_178953

theorem polynomial_remainder (x : ℝ) :
  (x^4 + 3 * x^2 - 4) % (x^2 + 2) = x^2 - 4 :=
sorry

end polynomial_remainder_l178_178953


namespace isosceles_right_triangle_area_l178_178204

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (A : ℝ)
  (h_def : h = 6 * Real.sqrt 2)
  (rel_leg_hypotenuse : h = Real.sqrt 2 * l)
  (area_def : A = 1 / 2 * l * l) :
  A = 18 :=
by
  sorry

end isosceles_right_triangle_area_l178_178204


namespace toys_ratio_l178_178858

-- Definitions of given conditions
variables (rabbits : ℕ) (toys_monday toys_wednesday toys_friday toys_saturday total_toys : ℕ)
variables (h_rabbits : rabbits = 16)
variables (h_toys_monday : toys_monday = 6)
variables (h_toys_friday : toys_friday = 4 * toys_monday)
variables (h_toys_saturday : toys_saturday = toys_wednesday / 2)
variables (h_total_toys : total_toys = rabbits * 3)

-- Define the Lean theorem to state the problem conditions and prove the ratio
theorem toys_ratio (h : toys_monday + toys_wednesday + toys_friday + toys_saturday = total_toys) :
  (if (2 * toys_wednesday = 12) then 2 else 1) = 2 :=
by 
  sorry

end toys_ratio_l178_178858


namespace jane_emily_total_accessories_l178_178158

def total_accessories : ℕ :=
  let jane_dresses := 4 * 10
  let emily_dresses := 3 * 8
  let jane_ribbons := 3 * jane_dresses
  let jane_buttons := 2 * jane_dresses
  let jane_lace_trims := 1 * jane_dresses
  let jane_beads := 4 * jane_dresses
  let emily_ribbons := 2 * emily_dresses
  let emily_buttons := 3 * emily_dresses
  let emily_lace_trims := 2 * emily_dresses
  let emily_beads := 5 * emily_dresses
  let emily_bows := 1 * emily_dresses
  jane_ribbons + jane_buttons + jane_lace_trims + jane_beads +
  emily_ribbons + emily_buttons + emily_lace_trims + emily_beads + emily_bows 

theorem jane_emily_total_accessories : total_accessories = 712 := 
by
  sorry

end jane_emily_total_accessories_l178_178158


namespace percentage_change_in_receipts_l178_178572

theorem percentage_change_in_receipts
  (P S : ℝ) -- Original price and sales
  (hP : P > 0)
  (hS : S > 0)
  (new_P : ℝ := 0.70 * P) -- Price after 30% reduction
  (new_S : ℝ := 1.50 * S) -- Sales after 50% increase
  :
  (new_P * new_S - P * S) / (P * S) * 100 = 5 :=
by
  sorry

end percentage_change_in_receipts_l178_178572


namespace simon_stamps_received_l178_178183

theorem simon_stamps_received (initial_stamps total_stamps received_stamps : ℕ) (h1 : initial_stamps = 34) (h2 : total_stamps = 61) : received_stamps = 27 :=
by
  sorry

end simon_stamps_received_l178_178183


namespace homer_second_try_points_l178_178971

theorem homer_second_try_points (x : ℕ) :
  400 + x + 2 * x = 1390 → x = 330 :=
by
  sorry

end homer_second_try_points_l178_178971


namespace arithmetic_sequence_value_l178_178471

theorem arithmetic_sequence_value (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_cond : a 3 + a 9 = 15 - a 6) : a 6 = 5 :=
sorry

end arithmetic_sequence_value_l178_178471


namespace closing_price_l178_178940

theorem closing_price (opening_price : ℝ) (percent_increase : ℝ) (closing_price : ℝ) 
  (h₀ : opening_price = 6) (h₁ : percent_increase = 0.3333) : closing_price = 8 :=
by
  sorry

end closing_price_l178_178940


namespace middle_guards_hours_l178_178252

def total_hours := 9
def hours_first_guard := 3
def hours_last_guard := 2
def remaining_hours := total_hours - hours_first_guard - hours_last_guard
def num_middle_guards := 2

theorem middle_guards_hours : remaining_hours / num_middle_guards = 2 := by
  sorry

end middle_guards_hours_l178_178252


namespace man_speed_in_still_water_l178_178073

theorem man_speed_in_still_water (V_m V_s : ℝ) 
  (h1 : V_m + V_s = 8)
  (h2 : V_m - V_s = 6) : 
  V_m = 7 := 
by
  sorry

end man_speed_in_still_water_l178_178073


namespace number_of_children_at_matinee_l178_178115

-- Definitions of constants based on conditions
def children_ticket_price : ℝ := 4.50
def adult_ticket_price : ℝ := 6.75
def total_receipts : ℝ := 405
def additional_children : ℕ := 20

-- Variables for number of adults and children
variable (A C : ℕ)

-- Assertions based on conditions
axiom H1 : C = A + additional_children
axiom H2 : children_ticket_price * (C : ℝ) + adult_ticket_price * (A : ℝ) = total_receipts

-- Theorem statement: Prove that the number of children is 48
theorem number_of_children_at_matinee : C = 48 :=
by
  sorry

end number_of_children_at_matinee_l178_178115


namespace road_distance_l178_178507

theorem road_distance 
  (S : ℕ) 
  (h : ∀ x : ℕ, x ≤ S → ∃ d : ℕ, d ∈ {1, 3, 13} ∧ d = Nat.gcd x (S - x)) : 
  S = 39 :=
by
  sorry

end road_distance_l178_178507


namespace functions_are_computable_l178_178800

def f1 : ℕ → ℕ := λ n => 0
def f2 : ℕ → ℕ := λ n => n + 1
def f3 : ℕ → ℕ := λ n => max 0 (n - 1)
def f4 : ℕ → ℕ := λ n => n % 2
def f5 : ℕ → ℕ := λ n => n * 2
def f6 : ℕ × ℕ → ℕ := λ (m, n) => if m ≤ n then 1 else 0

theorem functions_are_computable :
  (Computable f1) ∧
  (Computable f2) ∧
  (Computable f3) ∧
  (Computable f4) ∧
  (Computable f5) ∧
  (Computable f6) := by
  sorry

end functions_are_computable_l178_178800


namespace arithmetic_sequence_y_solve_l178_178351

theorem arithmetic_sequence_y_solve (y : ℝ) (h : y > 0) (arithmetic : ∀ a b c : ℝ, b = (a + c) / 2 → a, b, c are in arithmetic sequence):
  y^2 = (2^2 + 5^2) / 2 →
  y = Real.sqrt 14.5 :=
by
  assume h_y : y > 0,
  assume h_seq : y^2 = (2^2 + 5^2) / 2,
  sorry

end arithmetic_sequence_y_solve_l178_178351


namespace time_to_cross_tree_l178_178575

def train_length : ℕ := 600
def platform_length : ℕ := 450
def time_to_pass_platform : ℕ := 105

-- Definition of the condition that leads to the speed of the train
def speed_of_train : ℚ := (train_length + platform_length) / time_to_pass_platform

-- Statement to prove the time to cross the tree
theorem time_to_cross_tree :
  (train_length : ℚ) / speed_of_train = 60 :=
by
  sorry

end time_to_cross_tree_l178_178575


namespace circle_radius_through_focus_and_tangent_l178_178784

-- Define the given conditions of the problem
def ellipse_eq (x y : ℝ) : Prop := x^2 + 4 * y^2 = 16

-- State the problem as a theorem
theorem circle_radius_through_focus_and_tangent
  (x y : ℝ) (h : ellipse_eq x y) (r : ℝ) :
  r = 4 - 2 * Real.sqrt 3 :=
sorry

end circle_radius_through_focus_and_tangent_l178_178784


namespace system1_solution_exists_system2_solution_exists_l178_178754

-- System (1)
theorem system1_solution_exists (x y : ℝ) (h1 : y = 2 * x - 5) (h2 : 3 * x + 4 * y = 2) : 
  x = 2 ∧ y = -1 :=
by
  sorry

-- System (2)
theorem system2_solution_exists (x y : ℝ) (h1 : 3 * x - y = 8) (h2 : (y - 1) / 3 = (x + 5) / 5) : 
  x = 5 ∧ y = 7 :=
by
  sorry

end system1_solution_exists_system2_solution_exists_l178_178754


namespace angle_A_in_triangle_find_b_c_given_a_and_A_l178_178715

theorem angle_A_in_triangle (A B C : ℝ) (a b c : ℝ)
  (h1 : 2 * Real.cos (2 * A) + 4 * Real.cos (B + C) + 3 = 0) :
  A = π / 3 :=
by
  sorry

theorem find_b_c_given_a_and_A (b c : ℝ)
  (A : ℝ)
  (a : ℝ := Real.sqrt 3)
  (h1 : 2 * b * Real.cos A + Real.sqrt (0 - c^2 + 6 * c - 9) = a)
  (h2 : b + c = 3)
  (h3 : A = π / 3) :
  (b = 2 ∧ c = 1) ∨ (b = 1 ∧ c = 2) :=
by
  sorry

end angle_A_in_triangle_find_b_c_given_a_and_A_l178_178715


namespace geom_sequence_general_formula_l178_178827

theorem geom_sequence_general_formula :
  ∃ (a : ℕ → ℝ) (a₁ q : ℝ), 
  (∀ n, a n = a₁ * q ^ n ∧ abs (q) < 1 ∧ ∑' i, a i = 3 ∧ ∑' i, (a i)^2 = (9 / 2)) →
  (∀ n, a n = 2 * ((1 / 3) ^ (n - 1))) :=
by sorry

end geom_sequence_general_formula_l178_178827


namespace problem_l178_178177

-- Define the values in the grid
def grid : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ := (4, 3, 1, 1, 6, 2, 3)

-- Define the variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def condition_1 := (A = 3) ∧ (B = 2) ∧ (C = 4)
def condition_2 := (4 + A + 1 + B + C + 3 = 9)
def condition_3 := (A + 1 + 6 = 9)
def condition_4 := (1 + A + 6 = 9)
def condition_5 := (B + 2 + C + 5 = 9)

-- Define that the sum of the red cells is equal to any row
def sum_of_red_cells := (A + B + C = 9)

-- The final goal to prove
theorem problem : condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 ∧ sum_of_red_cells := 
by {
  refine ⟨_, _, _, _, _, _⟩;
  sorry   -- proofs for each condition
}

end problem_l178_178177


namespace perfect_square_trinomial_l178_178302

theorem perfect_square_trinomial (m : ℝ) : (∃ b : ℝ, (x^2 - 6 * x + m) = (x + b) ^ 2) → m = 9 :=
by
  sorry

end perfect_square_trinomial_l178_178302


namespace min_sum_of_factors_l178_178998

theorem min_sum_of_factors (a b : ℤ) (h1 : a * b = 72) : a + b ≥ -73 :=
sorry

end min_sum_of_factors_l178_178998


namespace class_size_l178_178270

def S : ℝ := 30

theorem class_size (total percent_dogs_videogames percent_dogs_movies number_students_prefer_dogs : ℝ)
  (h1 : percent_dogs_videogames = 0.5)
  (h2 : percent_dogs_movies = 0.1)
  (h3 : number_students_prefer_dogs = 18)
  (h4 : total * (percent_dogs_videogames + percent_dogs_movies) = number_students_prefer_dogs) :
  total = S :=
by
  sorry

end class_size_l178_178270


namespace find_price_of_turban_l178_178242

-- Define the main variables and conditions
def price_of_turban (T : ℝ) : Prop :=
  ((3 / 4) * 90 + T = 60 + T) → T = 30

-- State the theorem with the given conditions and aim to find T
theorem find_price_of_turban (T : ℝ) (h1 : 90 + T = 120) :  price_of_turban T :=
by
  intros
  sorry


end find_price_of_turban_l178_178242


namespace x_minus_q_eq_three_l178_178697

theorem x_minus_q_eq_three (x q : ℝ) (h1 : |x - 3| = q) (h2 : x > 3) : x - q = 3 :=
by 
  sorry

end x_minus_q_eq_three_l178_178697


namespace cos_sin_identity_l178_178102

theorem cos_sin_identity : 
  (Real.cos (14 * Real.pi / 180) * Real.cos (59 * Real.pi / 180) + 
   Real.sin (14 * Real.pi / 180) * Real.sin (121 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end cos_sin_identity_l178_178102


namespace percentage_rent_this_year_l178_178719

variables (E : ℝ)

-- Define the conditions from the problem
def rent_last_year (E : ℝ) : ℝ := 0.20 * E
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 1.4375 * rent_last_year E

-- The main statement to prove
theorem percentage_rent_this_year : 
  0.2875 * E = (25 / 100) * (earnings_this_year E) :=
by sorry

end percentage_rent_this_year_l178_178719


namespace range_of_a_l178_178452

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, y = (a * (Real.cos x)^2 - 3) * (Real.sin x) ∧ y ≥ -3) 
  → a ∈ Set.Icc (-3/2 : ℝ) 12 :=
sorry

end range_of_a_l178_178452


namespace no_integer_solutions_l178_178947

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
  -x^2 + 4 * y * z + 3 * z^2 = 36 ∧
  x^2 + 2 * x * y + 9 * z^2 = 121 → false :=
by
  sorry

end no_integer_solutions_l178_178947


namespace equation_of_line_passing_through_A_equation_of_circle_l178_178988

variable {α β γ : ℝ}
variable {a b c u v w : ℝ}
variable (A : ℝ × ℝ × ℝ) -- Barycentric coordinates of point A

-- Statement for the equation of a line passing through point A in barycentric coordinates
theorem equation_of_line_passing_through_A (A : ℝ × ℝ × ℝ) : 
  ∃ (u v w : ℝ), u * α + v * β + w * γ = 0 := by
  sorry

-- Statement for the equation of a circle in barycentric coordinates
theorem equation_of_circle {u v w : ℝ} :
  -a^2 * β * γ - b^2 * γ * α - c^2 * α * β +
  (u * α + v * β + w * γ) * (α + β + γ) = 0 := by
  sorry

end equation_of_line_passing_through_A_equation_of_circle_l178_178988


namespace hands_per_student_l178_178765

theorem hands_per_student (hands_without_peter : ℕ) (total_students : ℕ) (hands_peter : ℕ) 
  (h1 : hands_without_peter = 20) 
  (h2 : total_students = 11) 
  (h3 : hands_peter = 2) : 
  (hands_without_peter + hands_peter) / total_students = 2 :=
by
  sorry

end hands_per_student_l178_178765


namespace merchant_marked_price_percent_l178_178928

theorem merchant_marked_price_percent (L : ℝ) (hL : L = 100) (purchase_price : ℝ) (h1 : purchase_price = L * 0.70) (x : ℝ)
  (selling_price : ℝ) (h2 : selling_price = x * 0.75) :
  (selling_price - purchase_price) / selling_price = 0.30 → x = 133.33 :=
by
  sorry

end merchant_marked_price_percent_l178_178928


namespace hyperbola_eccentricity_l178_178287

-- Definitions translated from conditions
noncomputable def parabola_focus : ℝ × ℝ := (0, -Real.sqrt 5)
noncomputable def a : ℝ := 2
noncomputable def c : ℝ := Real.sqrt 5

-- Eccentricity formula for the hyperbola
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- Statement to be proved
theorem hyperbola_eccentricity :
  eccentricity c a = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_eccentricity_l178_178287


namespace min_initial_questionnaires_l178_178461

theorem min_initial_questionnaires 
(N : ℕ) 
(h1 : 0.60 * (N:ℝ) + 0.60 * (N:ℝ) * 0.80 + 0.60 * (N:ℝ) * (0.80^2) ≥ 750) : 
  N ≥ 513 := sorry

end min_initial_questionnaires_l178_178461


namespace count_distinct_lines_l178_178668

-- Define a 4-by-4 grid of lattice points
def grid_points := finset (ℕ × ℕ)

-- The set of all points in a 4-by-4 grid
def four_by_four_grid : grid_points :=
  {(0, 0), (0, 1), (0, 2), (0, 3),
   (1, 0), (1, 1), (1, 2), (1, 3),
   (2, 0), (2, 1), (2, 2), (2, 3),
   (3, 0), (3, 1), (3, 2), (3, 3)}.to_finset

-- A line passing through at least two points
def line (p1 p2 : ℕ × ℕ) : set (ℕ × ℕ) :=
  {p : ℕ × ℕ | ∃ λ : ℚ, ∃ b : ℚ, (p.2 : ℚ) = λ * (p.1 : ℚ) + b}

noncomputable theory

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 50. -/
theorem count_distinct_lines (grid : grid_points) (h : grid = four_by_four_grid) :
  ∃ n, n = 50 :=
by
  sorry

end count_distinct_lines_l178_178668


namespace problem_ab_plus_a_plus_b_l178_178029

noncomputable def polynomial := fun x : ℝ => x^4 - 6 * x - 2

theorem problem_ab_plus_a_plus_b :
  ∀ (a b : ℝ), polynomial a = 0 → polynomial b = 0 → (a * b + a + b) = 4 :=
by
  intros a b ha hb
  sorry

end problem_ab_plus_a_plus_b_l178_178029


namespace relay_race_order_count_l178_178985

-- Definitions based on the given conditions
def team_members : List String := ["Sam", "Priya", "Jordan", "Luis"]
def first_runner := "Sam"
def last_runner := "Jordan"

-- Theorem stating the number of different possible orders
theorem relay_race_order_count {team_members first_runner last_runner} :
  (team_members = ["Sam", "Priya", "Jordan", "Luis"]) →
  (first_runner = "Sam") →
  (last_runner = "Jordan") →
  (2 = 2) :=
by
  intros _ _ _
  sorry

end relay_race_order_count_l178_178985


namespace solve_n_minus_m_l178_178030

theorem solve_n_minus_m :
  ∃ m n, 
    (m ≡ 4 [MOD 7]) ∧ 100 ≤ m ∧ m < 1000 ∧ 
    (n ≡ 4 [MOD 7]) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
    n - m = 903 :=
by
  sorry

end solve_n_minus_m_l178_178030


namespace list_price_is_35_l178_178794

-- Define the conditions in Lean
variable (x : ℝ)

def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * (alice_selling_price x)

def bob_selling_price (x : ℝ) : ℝ := x - 20
def bob_commission (x : ℝ) : ℝ := 0.20 * (bob_selling_price x)

-- Define the theorem to be proven
theorem list_price_is_35 (x : ℝ) 
  (h : alice_commission x = bob_commission x) : x = 35 :=
by sorry

end list_price_is_35_l178_178794


namespace find_a_l178_178617

theorem find_a (a : ℝ) (h : (∃ x : ℝ, (a - 3) * x ^ |a - 2| + 4 = 0) ∧ |a-2| = 1) : a = 1 :=
sorry

end find_a_l178_178617


namespace largest_divisor_is_15_l178_178774

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def largest_divisor (n : ℕ) : ℕ :=
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)

theorem largest_divisor_is_15 : ∀ (n : ℕ), n > 0 → is_even n → 15 ∣ largest_divisor n ∧ (∀ m, m ∣ largest_divisor n → m ≤ 15) :=
by
  intros n pos even
  sorry

end largest_divisor_is_15_l178_178774


namespace geometric_sequence_first_term_l178_178473

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

theorem geometric_sequence_first_term (a_1 q : ℝ)
  (h1 : a_n a_1 q 2 * a_n a_1 q 3 * a_n a_1 q 4 = 27)
  (h2 : a_n a_1 q 6 = 27) 
  (h3 : a_1 > 0) : a_1 = 1 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_first_term_l178_178473


namespace car_bus_washing_inconsistency_l178_178860

theorem car_bus_washing_inconsistency :
  ∀ (C B : ℕ), 
    C % 2 = 0 →
    B % 2 = 1 →
    7 * C + 18 * B = 309 →
    3 + 8 + 5 + C + B = 15 →
    false :=
by
  sorry

end car_bus_washing_inconsistency_l178_178860


namespace jerry_books_vs_action_figures_l178_178323

-- Define the initial conditions as constants
def initial_books : ℕ := 7
def initial_action_figures : ℕ := 3
def added_action_figures : ℕ := 2

-- Define the total number of action figures after adding
def total_action_figures : ℕ := initial_action_figures + added_action_figures

-- The theorem we need to prove
theorem jerry_books_vs_action_figures : initial_books - total_action_figures = 2 :=
by
  -- Proof placeholder
  sorry

end jerry_books_vs_action_figures_l178_178323


namespace covariance_eq_integral_l178_178780

noncomputable def gauss_bivariate_density (a b r : ℝ) (φ : ℝ × ℝ → ℝ) : Prop :=
  ∀ x1 x2, φ (x1, x2) = (1 / (2 * π * (1 - r ^ 2).sqrt)) * 
  exp(-(x1^2 + x2^2 - 2 * r * x1 * x2) / (2 * (1 - r ^ 2)))

theorem covariance_eq_integral
  (ξ η : Type)
  [Gaussian ξ]
  [Gaussian η]
  (φ : ℝ × ℝ → ℝ)
  (f g : ℝ → ℝ)
  (f' g' : ℝ → ℝ)
  (a b ρ : ℝ)
  (h1 : expect ξ = 0) (h2 : expect η = 0)
  (h3 : var ξ = 1) (h4 : var η = 1)
  (ρ_nonneg : ρ ≥ 0)
  (density_condition : gauss_bivariate_density a b ρ φ) :
  covariance (f ∘ ξ) (g ∘ η) = ∫ r in 0..ρ, E (f' (ξ r) * g' (η r)) := 
sorry

end covariance_eq_integral_l178_178780


namespace obtuse_triangle_has_two_acute_angles_l178_178838

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- A theorem to prove that an obtuse triangle has exactly 2 acute angles 
theorem obtuse_triangle_has_two_acute_angles (A B C : ℝ) (h : is_obtuse_triangle A B C) : 
  (A > 0 ∧ A < 90 → B > 0 ∧ B < 90 → C > 0 ∧ C < 90) ∧
  (A > 0 ∧ A < 90 ∧ B > 0 ∧ B < 90) ∨
  (A > 0 ∧ A < 90 ∧ C > 0 ∧ C < 90) ∨
  (B > 0 ∧ B < 90 ∧ C > 0 ∧ C < 90) :=
sorry

end obtuse_triangle_has_two_acute_angles_l178_178838


namespace xiao_ming_speed_difference_l178_178361

noncomputable def distance_school : ℝ := 9.3
noncomputable def time_cycling : ℝ := 0.6
noncomputable def distance_park : ℝ := 0.9
noncomputable def time_walking : ℝ := 0.2

noncomputable def cycling_speed : ℝ := distance_school / time_cycling
noncomputable def walking_speed : ℝ := distance_park / time_walking
noncomputable def speed_difference : ℝ := cycling_speed - walking_speed

theorem xiao_ming_speed_difference : speed_difference = 11 := by
  sorry

end xiao_ming_speed_difference_l178_178361


namespace volume_Q4_l178_178826

noncomputable def tetrahedron_sequence (n : ℕ) : ℝ :=
  -- Define the sequence recursively
  match n with
  | 0       => 1
  | (n + 1) => tetrahedron_sequence n + (4^n * (1 / 27)^(n + 1))

theorem volume_Q4 : tetrahedron_sequence 4 = 1.173832 :=
by
  sorry

end volume_Q4_l178_178826


namespace number_of_lines_at_least_two_points_4_by_4_grid_l178_178660

-- Definition of 4-by-4 grid
def grid : Type := (Fin 4) × (Fin 4)

-- Definition of a line passing through at least two points in this grid
def line_through_at_least_two_points (points : List grid) : Prop := 
  points.length ≥ 2
  ∧ ∃ m b, ∀ (x y : Fin 4 × Fin 4), (x ∈ points ∧ y ∈ points) → (y.snd : ℕ) = m * (x.fst : ℕ) + b

-- Defining the total number of points choosing 2 out of 16
def total_points : Nat := Nat.choose 16 2

-- Defining the overcount of vertical, horizontal,
-- major diagonals, and secondary diagonals lines
def overcount : Nat := 8 + 2 + 4

-- Total distinct count of lines passing through at least two points
def correct_answer : Nat := total_points - overcount

-- Main theorem stating that the total count is 106
theorem number_of_lines_at_least_two_points_4_by_4_grid : correct_answer = 106 := 
by
  sorry

end number_of_lines_at_least_two_points_4_by_4_grid_l178_178660


namespace count_distinct_lines_l178_178669

-- Define a 4-by-4 grid of lattice points
def grid_points := finset (ℕ × ℕ)

-- The set of all points in a 4-by-4 grid
def four_by_four_grid : grid_points :=
  {(0, 0), (0, 1), (0, 2), (0, 3),
   (1, 0), (1, 1), (1, 2), (1, 3),
   (2, 0), (2, 1), (2, 2), (2, 3),
   (3, 0), (3, 1), (3, 2), (3, 3)}.to_finset

-- A line passing through at least two points
def line (p1 p2 : ℕ × ℕ) : set (ℕ × ℕ) :=
  {p : ℕ × ℕ | ∃ λ : ℚ, ∃ b : ℚ, (p.2 : ℚ) = λ * (p.1 : ℚ) + b}

noncomputable theory

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 50. -/
theorem count_distinct_lines (grid : grid_points) (h : grid = four_by_four_grid) :
  ∃ n, n = 50 :=
by
  sorry

end count_distinct_lines_l178_178669


namespace average_temperature_MTWT_l178_178189

theorem average_temperature_MTWT (T_TWTF : ℝ) (T_M : ℝ) (T_F : ℝ) (T_MTWT : ℝ) :
    T_TWTF = 40 →
    T_M = 42 →
    T_F = 10 →
    T_MTWT = ((4 * T_TWTF - T_F + T_M) / 4) →
    T_MTWT = 48 := 
by
  intros hT_TWTF hT_M hT_F hT_MTWT
  rw [hT_TWTF, hT_M, hT_F] at hT_MTWT
  norm_num at hT_MTWT
  exact hT_MTWT

end average_temperature_MTWT_l178_178189


namespace total_income_l178_178255

theorem total_income (I : ℝ) 
  (h1 : I * 0.225 = 40000) : 
  I = 177777.78 :=
by
  sorry

end total_income_l178_178255


namespace midpoint_pentagon_inequality_l178_178532

noncomputable def pentagon_area_midpoints (T : ℝ) : ℝ := sorry

theorem midpoint_pentagon_inequality {T t : ℝ} 
  (h1 : t = pentagon_area_midpoints T)
  (h2 : 0 < T) : 
  (3/4) * T > t ∧ t > (1/2) * T :=
  sorry

end midpoint_pentagon_inequality_l178_178532


namespace distance_between_cities_l178_178522

variable {x S : ℕ}

/-- The distance between city A and city B is 39 kilometers -/
theorem distance_between_cities (S : ℕ) (h1 : ∀ x : ℕ, 0 ≤ x → x ≤ S → (GCD x (S - x) = 1 ∨ GCD x (S - x) = 3 ∨ GCD x (S - x) = 13)) :
  S = 39 :=
sorry

end distance_between_cities_l178_178522


namespace guilty_prob_l178_178317

-- Defining suspects
inductive Suspect
| A
| B
| C

open Suspect

-- Constants for the problem
def looks_alike (x y : Suspect) : Prop :=
(x = A ∧ y = B) ∨ (x = B ∧ y = A)

def timid (x : Suspect) : Prop :=
x = A ∨ x = B

def bold (x : Suspect) : Prop :=
x = C

def alibi_dover (x : Suspect) : Prop :=
x = A ∨ x = B

def needs_accomplice (x : Suspect) : Prop :=
timid x

def works_alone (x : Suspect) : Prop :=
bold x

def in_bar_during_robbery (x : Suspect) : Prop :=
x = A ∨ x = B

-- Theorem to be proved
theorem guilty_prob :
  ∃ x : Suspect, (x = B) ∧ ∀ y : Suspect, y ≠ B → 
    ((y = A ∧ timid y ∧ needs_accomplice y ∧ in_bar_during_robbery y) ∨
    (y = C ∧ bold y ∧ works_alone y)) :=
by
  sorry

end guilty_prob_l178_178317


namespace exist_students_with_comparable_scores_l178_178606

theorem exist_students_with_comparable_scores :
  ∃ (A B : ℕ) (a1 a2 a3 b1 b2 b3 : ℕ), 
    A ≠ B ∧ A < 49 ∧ B < 49 ∧
    (0 ≤ a1 ∧ a1 ≤ 7) ∧ (0 ≤ a2 ∧ a2 ≤ 7) ∧ (0 ≤ a3 ∧ a3 ≤ 7) ∧ 
    (0 ≤ b1 ∧ b1 ≤ 7) ∧ (0 ≤ b2 ∧ b2 ≤ 7) ∧ (0 ≤ b3 ∧ b3 ≤ 7) ∧ 
    (a1 ≥ b1) ∧ (a2 ≥ b2) ∧ (a3 ≥ b3) := 
sorry

end exist_students_with_comparable_scores_l178_178606


namespace arithmetic_progression_probability_l178_178124

theorem arithmetic_progression_probability (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_outcomes = 6^4 ∧ favorable_outcomes = 3 →
  favorable_outcomes / total_outcomes = 1 / 432 :=
by
  sorry

end arithmetic_progression_probability_l178_178124


namespace CALI_area_is_180_l178_178747

-- all the conditions used in Lean definitions
def is_square (s : ℕ) : Prop := (s > 0)

def are_midpoints (T O W N B E R K : ℕ) : Prop := 
  (T = (B + E) / 2) ∧ (O = (E + R) / 2) ∧ (W = (R + K) / 2) ∧ (N = (K + B) / 2)

def is_parallel (CA BO : ℕ) : Prop :=
  CA = BO 

-- the condition indicates the length of each side of the square BERK is 10
def side_length_of_BERK : ℕ := 10

-- definition of lengths and condition
def BERK_lengths (BERK_side_length : ℕ) (BERK_diag_length : ℕ): Prop :=
  BERK_side_length = side_length_of_BERK ∧ BERK_diag_length = BERK_side_length * (2^(1/2))

def CALI_area_of_length (length: ℕ): ℕ := length^2

theorem CALI_area_is_180 
(BERK_side_length BERK_diag_length : ℕ)
(CALI_length : ℕ)
(T O W N B E R K CA BO : ℕ)
(h1 : is_square BERK_side_length)
(h2 : are_midpoints T O W N B E R K)
(h3 : is_parallel CA BO)
(h4 : BERK_lengths BERK_side_length BERK_diag_length)
(h5 : CA = CA)
: CALI_area_of_length 15 = 180 :=
sorry

end CALI_area_is_180_l178_178747


namespace remainder_when_divided_by_x_minus_2_l178_178817

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + x^2 + 4

theorem remainder_when_divided_by_x_minus_2 : f 2 = 56 :=
by
  -- Proof steps will go here.
  sorry

end remainder_when_divided_by_x_minus_2_l178_178817


namespace find_m_n_sum_l178_178001

theorem find_m_n_sum (n m : ℝ) (d : ℝ) 
(h1 : ∀ x y, 2*x + y + n = 0) 
(h2 : ∀ x y, 4*x + m*y - 4 = 0) 
(hd : d = (3/5) * Real.sqrt 5) 
: m + n = -3 ∨ m + n = 3 :=
sorry

end find_m_n_sum_l178_178001


namespace right_triangle_and_mod_inverse_l178_178422

theorem right_triangle_and_mod_inverse (a b c m : ℕ) (h1 : a = 48) (h2 : b = 55) (h3 : c = 73) (h4 : m = 4273) 
  (h5 : a^2 + b^2 = c^2) : ∃ x : ℕ, (480 * x) % m = 1 ∧ x = 1643 :=
by
  sorry

end right_triangle_and_mod_inverse_l178_178422


namespace sum_of_cubes_l178_178237

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l178_178237


namespace sum_of_cubes_l178_178234

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l178_178234


namespace negation_if_proposition_l178_178527

variable (a b : Prop)

theorem negation_if_proposition (a b : Prop) : ¬ (a → b) = a ∧ ¬b := 
sorry

end negation_if_proposition_l178_178527


namespace roots_of_quadratic_l178_178012

theorem roots_of_quadratic (a b c : ℝ) (h1 : a ≠ 0) (h2 : a + b + c = 0) (h3 : a - b + c = 0) : 
  (a * 1 ^2 + b * 1 + c = 0) ∧ (a * (-1) ^2 + b * (-1) + c = 0) :=
sorry

end roots_of_quadratic_l178_178012


namespace sum_of_first_7_terms_l178_178241

theorem sum_of_first_7_terms (a d : ℤ) (h3 : a + 2 * d = 2) (h4 : a + 3 * d = 5) (h5 : a + 4 * d = 8) :
  (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) + (a + 6 * d)) = 35 :=
by
  sorry

end sum_of_first_7_terms_l178_178241


namespace lines_in_4_by_4_grid_l178_178636

theorem lines_in_4_by_4_grid : 
  (count_lines_passing_through_at_least_two_points (4, 4) = 62) :=
sorry

def count_lines_passing_through_at_least_two_points (m n : ℕ) : ℕ :=
  let total_pairs := (m * n) * ((m * n) - 1) / 2
  let overcount_lines := (6 - 1) * 10 + (3 - 1) * 4
  total_pairs - overcount_lines

end lines_in_4_by_4_grid_l178_178636


namespace students_not_playing_either_game_l178_178901

theorem students_not_playing_either_game
  (total_students : ℕ) -- There are 20 students in the class
  (play_basketball : ℕ) -- Half of them play basketball
  (play_volleyball : ℕ) -- Two-fifths of them play volleyball
  (play_both : ℕ) -- One-tenth of them play both basketball and volleyball
  (h_total : total_students = 20)
  (h_basketball : play_basketball = 10)
  (h_volleyball : play_volleyball = 8)
  (h_both : play_both = 2) :
  total_students - (play_basketball + play_volleyball - play_both) = 4 := by
  sorry

end students_not_playing_either_game_l178_178901


namespace lines_in_4_by_4_grid_l178_178682

theorem lines_in_4_by_4_grid : 
  let n := 4 in
  number_of_lines_at_least_two_points (grid_of_lattice_points n) = 96 :=
by sorry

end lines_in_4_by_4_grid_l178_178682


namespace C_payment_l178_178382

-- Defining the problem conditions
def A_work_rate : ℚ := 1 / 6
def B_work_rate : ℚ := 1 / 8
def A_B_work_rate : ℚ := A_work_rate + B_work_rate
def A_B_C_work_rate : ℚ := 1 / 3
def C_work_rate : ℚ := A_B_C_work_rate - A_B_work_rate
def total_payment : ℚ := 3600

-- The proof goal
theorem C_payment : (C_work_rate / A_B_C_work_rate) * total_payment = 450 :=
by
  rw [C_work_rate, A_work_rate, B_work_rate, A_B_work_rate, A_B_C_work_rate, total_payment]
  sorry

end C_payment_l178_178382


namespace broken_crayons_l178_178495

theorem broken_crayons (total new used : Nat) (h1 : total = 14) (h2 : new = 2) (h3 : used = 4) :
  total = new + used + 8 :=
by
  -- Proof omitted
  sorry

end broken_crayons_l178_178495


namespace sum_of_cubes_l178_178222

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178222


namespace sum_of_cubes_l178_178230

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178230


namespace minimum_value_expression_l178_178863

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (4 * z / (2 * x + y)) + (4 * x / (y + 2 * z)) + (y / (x + z)) ≥ 3 :=
by 
  sorry

end minimum_value_expression_l178_178863


namespace cannot_afford_laptop_l178_178872

theorem cannot_afford_laptop (P_0 : ℝ) : 56358 < P_0 * (1.06)^2 :=
by
  sorry

end cannot_afford_laptop_l178_178872


namespace range_of_a_l178_178103

def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, operation (x - a) (x + 1) < 1) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l178_178103


namespace lines_in_4x4_grid_l178_178644

theorem lines_in_4x4_grid :
  let n := 4
  let total_points := n * n
  let choose_two_points := total_points.choose 2
  let horizontal_and_vertical_lines := n + n
  let diagonal_lines := 6 -- based on detailed breakdown
  let adjustment_for_lines_through_four_points := 8 * 3
  let adjustment_for_lines_through_three_points := 4 * 2
  let initial_line_count := choose_two_points
  let adjusted_line_count := initial_line_count - adjustment_for_lines_through_four_points - adjustment_for_lines_through_three_points
  in adjusted_line_count = 88 := 
by {
  exact 88 // Placeholder proof statement
  sorry
}

end lines_in_4x4_grid_l178_178644


namespace value_of_expression_when_x_is_3_l178_178914

theorem value_of_expression_when_x_is_3 :
  (3^6 - 6*3 = 711) :=
by
  sorry

end value_of_expression_when_x_is_3_l178_178914


namespace probability_at_least_one_white_ball_stall_owner_monthly_earning_l178_178855

noncomputable def prob_at_least_one_white_ball : ℚ :=
1 - (3 / 10)

theorem probability_at_least_one_white_ball : prob_at_least_one_white_ball = 9 / 10 :=
sorry

noncomputable def expected_monthly_earnings (daily_draws : ℕ) (days_in_month : ℕ) : ℤ :=
(days_in_month * (90 * 1 - 10 * 5))

theorem stall_owner_monthly_earning (daily_draws : ℕ) (days_in_month : ℕ) :
  daily_draws = 100 → days_in_month = 30 →
  expected_monthly_earnings daily_draws days_in_month = 1200 :=
sorry

end probability_at_least_one_white_ball_stall_owner_monthly_earning_l178_178855


namespace boiling_point_water_standard_l178_178371

def boiling_point_water_celsius : ℝ := 100

theorem boiling_point_water_standard (bp_f : ℝ := 212) (ice_melting_c : ℝ := 0) (ice_melting_f : ℝ := 32) (pot_temp_c : ℝ := 55) (pot_temp_f : ℝ := 131) : boiling_point_water_celsius = 100 :=
by 
  -- Assuming standard atmospheric conditions, the boiling point of water in Celsius is 100.
  sorry

end boiling_point_water_standard_l178_178371


namespace find_a9_l178_178156

variable (a : ℕ → ℝ)

theorem find_a9 (h1 : a 4 - a 2 = -2) (h2 : a 7 = -3) : a 9 = -5 :=
sorry

end find_a9_l178_178156


namespace smallest_integer_inequality_l178_178818

theorem smallest_integer_inequality (x y z : ℝ) : 
  (x^3 + y^3 + z^3)^2 ≤ 3 * (x^6 + y^6 + z^6) ∧ 
  (∃ n : ℤ, (0 < n ∧ n < 3) → ∀ x y z : ℝ, ¬(x^3 + y^3 + z^3)^2 ≤ n * (x^6 + y^6 + z^6)) :=
by
  sorry

end smallest_integer_inequality_l178_178818


namespace find_three_digit_number_l178_178916

theorem find_three_digit_number (A B C D : ℕ) 
  (h1 : A + C = 5) 
  (h2 : B = 3)
  (h3 : A * 100 + B * 10 + C + 124 = D * 111) 
  (h4 : A ≠ B ∧ A ≠ C ∧ B ≠ C) : 
  A * 100 + B * 10 + C = 431 := 
by 
  sorry

end find_three_digit_number_l178_178916


namespace car_mileage_proof_l178_178802

noncomputable def car_average_mpg 
  (odometer_start: ℝ) (odometer_end: ℝ) 
  (fuel1: ℝ) (fuel2: ℝ) (odometer2: ℝ) 
  (fuel3: ℝ) (odometer3: ℝ) (final_fuel: ℝ) 
  (final_odometer: ℝ): ℝ :=
  (odometer_end - odometer_start) / 
  ((fuel1 + fuel2 + fuel3 + final_fuel): ℝ)

theorem car_mileage_proof:
  car_average_mpg 56200 57150 6 14 56600 10 56880 20 57150 = 19 :=
by
  sorry

end car_mileage_proof_l178_178802


namespace option_d_is_deductive_reasoning_l178_178796

-- Define the conditions of the problem
def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ c q : ℤ, c * q ≠ 0 ∧ ∀ n : ℕ, a n = c * q ^ n

-- Define the specific sequence {-2^n}
def a (n : ℕ) : ℤ := -2^n

-- State the proof problem
theorem option_d_is_deductive_reasoning :
  is_geometric_sequence a :=
sorry

end option_d_is_deductive_reasoning_l178_178796


namespace sum_of_possible_radii_l178_178925

-- Define the geometric and algebraic conditions of the problem
noncomputable def circleTangentSum (r : ℝ) : Prop :=
  let center_C := (r, r)
  let center_other := (3, 3)
  let radius_other := 2
  (∃ r : ℝ, (r > 0) ∧ ((center_C.1 - center_other.1)^2 + (center_C.2 - center_other.2)^2 = (r + radius_other)^2))

-- Define the theorem statement
theorem sum_of_possible_radii : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ circleTangentSum r1 ∧ circleTangentSum r2 ∧ r1 + r2 = 16 :=
sorry

end sum_of_possible_radii_l178_178925


namespace largest_integer_x_l178_178909

theorem largest_integer_x (x : ℤ) : (8:ℚ)/11 > (x:ℚ)/15 → x ≤ 10 :=
by
  intro h
  sorry

end largest_integer_x_l178_178909


namespace geometric_sequence_sixth_term_l178_178583

noncomputable def a : ℕ := 3
noncomputable def r : ℕ := 3

theorem geometric_sequence_sixth_term :
  (a : ℕ) = 3 →
  (a * r^4 = 243) →
  (r = 3) →
  (a * r^5 = 729) :=
by
  intros ha hr ht
  have ha : a = 3 := ha
  have hr : a * r^4 = 243 := hr
  have ht : r = 3 := ht
  rw [ha, ht]
  norm_num
  sorry

end geometric_sequence_sixth_term_l178_178583


namespace infinite_n_dividing_a_pow_n_plus_1_l178_178116

theorem infinite_n_dividing_a_pow_n_plus_1 (a : ℕ) (h1 : 1 < a) (h2 : a % 2 = 0) :
  ∃ (S : Set ℕ), S.Infinite ∧ ∀ n ∈ S, n ∣ a^n + 1 := 
sorry

end infinite_n_dividing_a_pow_n_plus_1_l178_178116


namespace sum_of_cubes_l178_178226

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l178_178226


namespace area_of_right_triangle_from_roots_l178_178309

theorem area_of_right_triangle_from_roots :
  ∀ (a b : ℝ), (a^2 - 7*a + 12 = 0) → (b^2 - 7*b + 12 = 0) →
  (∃ (area : ℝ), (area = 6) ∨ (area = (3 * real.sqrt 7) / 2)) :=
by
  intros a b ha hb
  sorry

end area_of_right_triangle_from_roots_l178_178309


namespace pages_share_units_digit_l178_178393

def units_digit (n : Nat) : Nat :=
  n % 10

theorem pages_share_units_digit :
  (∃ (x_set : Finset ℕ), (∀ (x : ℕ), x ∈ x_set ↔ (1 ≤ x ∧ x ≤ 63 ∧ units_digit x = units_digit (64 - x))) ∧ x_set.card = 13) :=
by
  sorry

end pages_share_units_digit_l178_178393


namespace candies_left_after_carlos_ate_l178_178564

def num_red_candies : ℕ := 50
def num_yellow_candies : ℕ := 3 * num_red_candies - 35
def num_blue_candies : ℕ := (2 * num_yellow_candies) / 3
def num_green_candies : ℕ := 20
def num_purple_candies : ℕ := num_green_candies / 2
def num_silver_candies : ℕ := 10
def num_candies_eaten_by_carlos : ℕ := num_yellow_candies + num_green_candies / 2

def total_candies : ℕ := num_red_candies + num_yellow_candies + num_blue_candies + num_green_candies + num_purple_candies + num_silver_candies
def candies_remaining : ℕ := total_candies - num_candies_eaten_by_carlos

theorem candies_left_after_carlos_ate : candies_remaining = 156 := by
  sorry

end candies_left_after_carlos_ate_l178_178564


namespace lines_in_4x4_grid_l178_178675

theorem lines_in_4x4_grid : 
  let grid_points := finset.univ.product finset.univ
  let total_points := 16
  let pairs_of_points := total_points.choose 2
  let horizontal_lines := 4
  let vertical_lines := 4
  let diagonal_lines := 2
  let lines_through_four_points := horizontal_lines + vertical_lines + diagonal_lines
  let correction := lines_through_four_points * (4.choose 2 - 1)
  let number_of_lines := pairs_of_points - correction
  in number_of_lines = 70 := 
by {
  sorry
}

end lines_in_4x4_grid_l178_178675


namespace line_through_intersection_perpendicular_l178_178944

theorem line_through_intersection_perpendicular (x y : ℝ) :
  (2 * x - 3 * y + 10 = 0) ∧ (3 * x + 4 * y - 2 = 0) →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a = 2) ∧ (b = 3) ∧ (c = -2) ∧ (3 * a + 2 * b = 0)) :=
by
  sorry

end line_through_intersection_perpendicular_l178_178944


namespace urn_contains_specific_balls_after_operations_l178_178799

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5
def final_red_balls : ℕ := 10
def final_blue_balls : ℕ := 6
def target_probability : ℚ := 16 / 115

noncomputable def urn_proba_result : ℚ := sorry

theorem urn_contains_specific_balls_after_operations :
  urn_proba_result = target_probability := sorry

end urn_contains_specific_balls_after_operations_l178_178799


namespace odd_function_value_at_neg_two_l178_178448

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then 2 * x - 3 else - (2 * (-x) - 3)

theorem odd_function_value_at_neg_two :
  (∀ x, f (-x) = -f x) → f (-2) = -1 :=
by
  intro odd_f
  sorry

end odd_function_value_at_neg_two_l178_178448


namespace equation_of_line_l178_178046

theorem equation_of_line (x_intercept slope : ℝ)
  (hx : x_intercept = 2) (hm : slope = 1) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -1 ∧ c = -2 ∧ (∀ x y : ℝ, y = slope * (x - x_intercept) ↔ a * x + b * y + c = 0) := sorry

end equation_of_line_l178_178046


namespace total_enemies_l178_178710

theorem total_enemies (E : ℕ) (h : 8 * (E - 2) = 40) : E = 7 := sorry

end total_enemies_l178_178710


namespace lines_in_4_by_4_grid_l178_178637

theorem lines_in_4_by_4_grid : 
  (count_lines_passing_through_at_least_two_points (4, 4) = 62) :=
sorry

def count_lines_passing_through_at_least_two_points (m n : ℕ) : ℕ :=
  let total_pairs := (m * n) * ((m * n) - 1) / 2
  let overcount_lines := (6 - 1) * 10 + (3 - 1) * 4
  total_pairs - overcount_lines

end lines_in_4_by_4_grid_l178_178637


namespace neg_cube_squared_l178_178079

theorem neg_cube_squared (x : ℝ) : (-x^3) ^ 2 = x ^ 6 :=
by
  sorry

end neg_cube_squared_l178_178079


namespace shiela_used_seven_colors_l178_178936

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ)
  (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : (total_blocks / blocks_per_color) = 7 :=
by
  sorry

end shiela_used_seven_colors_l178_178936


namespace train_crossing_time_l178_178074

-- Condition definitions
def length_train : ℝ := 100
def length_bridge : ℝ := 150
def speed_kmph : ℝ := 54
def speed_mps : ℝ := 15

-- Given the conditions, prove the time to cross the bridge is 16.67 seconds
theorem train_crossing_time :
  (100 + 150) / (54 * 1000 / 3600) = 16.67 := by sorry

end train_crossing_time_l178_178074


namespace value_of_g_at_neg3_l178_178843

def g (x : ℚ) : ℚ := (6 * x + 2) / (x - 2)

theorem value_of_g_at_neg3 : g (-3) = 16 / 5 := by
  sorry

end value_of_g_at_neg3_l178_178843


namespace distance_to_x_axis_l178_178701

theorem distance_to_x_axis (x y : ℤ) (h : (x, y) = (-3, 5)) : |y| = 5 := by
  -- coordinates of point A are (-3, 5)
  sorry

end distance_to_x_axis_l178_178701


namespace gas_price_l178_178402

theorem gas_price (x : ℝ) (h1 : 10 * (x + 0.30) = 12 * x) : x + 0.30 = 1.80 := by
  sorry

end gas_price_l178_178402


namespace perimeter_hypotenuse_ratios_l178_178443

variable {x y : Real}
variable (h_pos_x : x > 0) (h_pos_y : y > 0)

theorem perimeter_hypotenuse_ratios
    (h_sides : (3 * x + 3 * y = (3 * x + 3 * y)) ∨ 
               (4 * x = (4 * x)) ∨
               (4 * y = (4 * y)))
    : 
    (∃ p : Real, p = 7 * (x + y) / (3 * (x + y)) ∨
                 p = 32 * y / (100 / 7 * y) ∨
                 p = 224 / 25 * y / 4 * y ∨ 
                 p = 7 / 3 ∨ 
                 p = 56 / 25) := by sorry

end perimeter_hypotenuse_ratios_l178_178443


namespace total_watermelons_l178_178993

/-- Proof statement: Jason grew 37 watermelons and Sandy grew 11 watermelons. 
    Prove that they grew a total of 48 watermelons. -/
theorem total_watermelons (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ) 
                         (h1 : jason_watermelons = 37) (h2 : sandy_watermelons = 11) :
  total_watermelons = 48 :=
by
  sorry

end total_watermelons_l178_178993


namespace problem_i31_problem_i32_problem_i33_problem_i34_l178_178010

-- Problem I3.1
theorem problem_i31 (a : ℝ) :
  a = 1.8 * 5.0865 + 1 - 0.0865 * 1.8 → a = 10 :=
by sorry

-- Problem I3.2
theorem problem_i32 (a b : ℕ) (oh ok : ℕ) (OABC : Prop) :
  oh = ok ∧ oh = a ∧ ok = a ∧ OABC ∧ (b = AC) → b = 10 :=
by sorry

-- Problem I3.3
theorem problem_i33 (b c : ℕ) :
  b = 10 → c = (10 - 2) :=
by sorry

-- Problem I3.4
theorem problem_i34 (c d : ℕ) :
  c = 30 → d = 3 * c → d = 90 :=
by sorry

end problem_i31_problem_i32_problem_i33_problem_i34_l178_178010


namespace number_division_l178_178585

theorem number_division (N x : ℕ) 
  (h1 : (N - 5) / x = 7) 
  (h2 : (N - 34) / 10 = 2)
  : x = 7 := 
by
  sorry

end number_division_l178_178585


namespace molecular_weight_calculation_l178_178556

-- Define the atomic weights of each element
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms of each element in the compound
def num_atoms_C : ℕ := 7
def num_atoms_H : ℕ := 6
def num_atoms_O : ℕ := 2

-- The molecular weight calculation
def molecular_weight : ℝ :=
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_H * atomic_weight_H) +
  (num_atoms_O * atomic_weight_O)

theorem molecular_weight_calculation : molecular_weight = 122.118 :=
by
  -- Proof
  sorry

end molecular_weight_calculation_l178_178556


namespace lines_in_4_by_4_grid_l178_178681

theorem lines_in_4_by_4_grid : 
  let n := 4 in
  number_of_lines_at_least_two_points (grid_of_lattice_points n) = 96 :=
by sorry

end lines_in_4_by_4_grid_l178_178681


namespace find_sequence_index_l178_178018

theorem find_sequence_index (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) - 3 = a n)
  (h₃ : ∃ n, a n = 2023) : ∃ n, a n = 2023 ∧ n = 675 := 
by 
  sorry

end find_sequence_index_l178_178018


namespace defective_units_l178_178474

-- Conditions given in the problem
variable (D : ℝ) (h1 : 0.05 * D = 0.35)

-- The percent of the units produced that are defective is 7%
theorem defective_units (h1 : 0.05 * D = 0.35) : D = 7 := sorry

end defective_units_l178_178474


namespace different_lines_through_two_points_in_4_by_4_grid_l178_178634

theorem different_lines_through_two_points_in_4_by_4_grid : 
  let points := fin 4 × fin 4 in
  let number_of_lines := 
    (nat.choose 16 2) - 
    (8 * (4 - 1)) - 
    (2 * (4 - 1)) in
  number_of_lines = 90 :=
by
  sorry

end different_lines_through_two_points_in_4_by_4_grid_l178_178634


namespace compute_complex_power_l178_178942

noncomputable def complex_number := Complex.exp (Complex.I * 125 * Real.pi / 180)

theorem compute_complex_power :
  (complex_number ^ 28) = Complex.ofReal (-Real.cos (40 * Real.pi / 180)) + Complex.I * Real.sin (40 * Real.pi / 180) :=
by
  sorry

end compute_complex_power_l178_178942


namespace tan_inequality_solution_l178_178499

variable (x : ℝ)
variable (k : ℤ)

theorem tan_inequality_solution (hx : Real.tan (2 * x - Real.pi / 4) ≤ 1) :
  ∃ k : ℤ,
  (k * Real.pi / 2 - Real.pi / 8 < x) ∧ (x ≤ k * Real.pi / 2 + Real.pi / 4) :=
sorry

end tan_inequality_solution_l178_178499


namespace Bing_max_games_l178_178716

/-- 
  Jia, Yi, and Bing play table tennis with the following rules: each game is played between two 
  people, and the loser gives way to the third person. If Jia played 10 games and Yi played 
  7 games, then Bing can play at most 13 games; and can win at most 10 games.
-/
theorem Bing_max_games 
  (games_played_Jia : ℕ)
  (games_played_Yi : ℕ)
  (games_played_Bing : ℕ)
  (games_won_Bing  : ℕ)
  (hJia : games_played_Jia = 10)
  (hYi : games_played_Yi = 7) :
  (games_played_Bing ≤ 13) ∧ (games_won_Bing ≤ 10) := 
sorry

end Bing_max_games_l178_178716


namespace ratio_of_sides_l178_178279

open Real

variable (s y x : ℝ)

-- Assuming the rectangles and squares conditions
def condition1 := 4 * (x * y) + s * s = 9 * (s * s)
def condition2 := x = 2 * s
def condition3 := y = s

-- Stating the theorem
theorem ratio_of_sides (h1 : condition1 s y x) (h2 : condition2 s x) (h3 : condition3 s y) :
  x / y = 2 := by
  sorry

end ratio_of_sides_l178_178279


namespace whole_process_time_is_9_l178_178269

variable (BleachingTime : ℕ)
variable (DyeingTime : ℕ)

-- Conditions
axiom bleachingTime_is_3 : BleachingTime = 3
axiom dyeingTime_is_twice_bleachingTime : DyeingTime = 2 * BleachingTime

-- Question and Proof Problem
theorem whole_process_time_is_9 (BleachingTime : ℕ) (DyeingTime : ℕ)
  (h1 : BleachingTime = 3) (h2 : DyeingTime = 2 * BleachingTime) : 
  (BleachingTime + DyeingTime) = 9 :=
  by
  sorry

end whole_process_time_is_9_l178_178269


namespace g_inverse_sum_l178_178723

-- Define the function g and its inverse
def g (x : ℝ) : ℝ := x ^ 3
noncomputable def g_inv (y : ℝ) : ℝ := y ^ (1/3 : ℝ)

-- State the theorem to be proved
theorem g_inverse_sum : g_inv 8 + g_inv (-64) = -2 := by 
  sorry

end g_inverse_sum_l178_178723


namespace last_digit_fib_mod_12_l178_178891

noncomputable def F : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => (F n + F (n + 1)) % 12

theorem last_digit_fib_mod_12 : ∃ N, ∀ n < N, (∃ k, F k % 12 = n) ∧ ∀ m > N, F m % 12 ≠ 11 :=
sorry

end last_digit_fib_mod_12_l178_178891


namespace f_value_at_3_l178_178198

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_at_3 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 4 * x^2) : f 3 = -4 / 3 :=
by
  sorry

end f_value_at_3_l178_178198


namespace solve_inequality_l178_178112

theorem solve_inequality (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ x ∈ Set.Ioo (-2) (3) :=
sorry

end solve_inequality_l178_178112


namespace friends_division_l178_178300

def num_ways_to_divide (total_friends teams : ℕ) : ℕ :=
  4^8 - (Nat.choose 4 1) * 3^8 + (Nat.choose 4 2) * 2^8 - (Nat.choose 4 3) * 1^8

theorem friends_division (total_friends teams : ℕ) (h_friends : total_friends = 8) (h_teams : teams = 4) :
  num_ways_to_divide total_friends teams = 39824 := by
  sorry

end friends_division_l178_178300


namespace guests_accommodation_l178_178854

open Nat

theorem guests_accommodation :
  let guests := 15
  let rooms := 4
  (4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4 = 4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4) :=
by
  sorry

end guests_accommodation_l178_178854


namespace no_real_roots_in_interval_l178_178328

variable {a b c : ℝ}

theorem no_real_roots_in_interval (ha : 0 < a) (h : 12 * a + 5 * b + 2 * c > 0) :
  ¬ ∃ α β, (2 < α ∧ α < 3) ∧ (2 < β ∧ β < 3) ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0 := by
  sorry

end no_real_roots_in_interval_l178_178328


namespace yards_in_a_mile_l178_178828

def mile_eq_furlongs : Prop := 1 = 5 * 1
def furlong_eq_rods : Prop := 1 = 50 * 1
def rod_eq_yards : Prop := 1 = 5 * 1

theorem yards_in_a_mile (h1 : mile_eq_furlongs) (h2 : furlong_eq_rods) (h3 : rod_eq_yards) :
  1 * (5 * (50 * 5)) = 1250 :=
by
-- Given conditions, translate them:
-- h1 : 1 mile = 5 furlongs -> 1 * 1 = 5 * 1
-- h2 : 1 furlong = 50 rods -> 1 * 1 = 50 * 1
-- h3 : 1 rod = 5 yards -> 1 * 1 = 5 * 1
-- Prove that the number of yards in one mile is 1250
sorry

end yards_in_a_mile_l178_178828


namespace solve_equation_l178_178887

theorem solve_equation (x : ℝ) (h : x ≠ 1) (h_eq : x / (x - 1) = (x - 3) / (2 * x - 2)) : x = -3 :=
by
  sorry

end solve_equation_l178_178887


namespace find_f_two_l178_178524

-- The function f is defined on (0, +∞) and takes positive values
noncomputable def f : ℝ → ℝ := sorry

-- The given condition that areas of triangle AOB and trapezoid ABH_BH_A are equal
axiom equalAreas (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) : 
  (1 / 2) * |x1 * f x2 - x2 * f x1| = (1 / 2) * (x2 - x1) * (f x1 + f x2)

-- The specific given value
axiom f_one : f 1 = 4

-- The theorem we need to prove
theorem find_f_two : f 2 = 2 :=
sorry

end find_f_two_l178_178524


namespace find_y_l178_178977

theorem find_y (x y : ℝ) (h1 : 2 * x - 3 * y = 24) (h2 : x + 2 * y = 15) : y = 6 / 7 :=
by sorry

end find_y_l178_178977


namespace find_f_of_7_over_2_l178_178453

variable (f : ℝ → ℝ)

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f (x - 2)
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = 3^x

theorem find_f_of_7_over_2 : f (7 / 2) = -Real.sqrt 3 :=
by
  sorry

end find_f_of_7_over_2_l178_178453


namespace custom_mul_2021_1999_l178_178600

axiom custom_mul : ℕ → ℕ → ℕ

axiom custom_mul_id1 : ∀ (A : ℕ), custom_mul A A = 0
axiom custom_mul_id2 : ∀ (A B C : ℕ), custom_mul A (custom_mul B C) = custom_mul A B + C

theorem custom_mul_2021_1999 : custom_mul 2021 1999 = 22 := by
  sorry

end custom_mul_2021_1999_l178_178600


namespace determine_g_l178_178597

noncomputable def g (x : ℝ) : ℝ := -4 * x^4 + 6 * x^3 - 9 * x^2 + 10 * x - 8

theorem determine_g (x : ℝ) : 
  4 * x^4 + 5 * x^2 - 2 * x + 7 + g x = 6 * x^3 - 4 * x^2 + 8 * x - 1 := by
  sorry

end determine_g_l178_178597


namespace smallest_number_plus_3_divisible_by_18_70_100_21_l178_178571

/-- 
The smallest number such that when increased by 3 is divisible by 18, 70, 100, and 21.
-/
theorem smallest_number_plus_3_divisible_by_18_70_100_21 : 
  ∃ n : ℕ, (∃ k : ℕ, n + 3 = k * 18) ∧ (∃ l : ℕ, n + 3 = l * 70) ∧ (∃ m : ℕ, n + 3 = m * 100) ∧ (∃ o : ℕ, n + 3 = o * 21) ∧ n = 6297 :=
sorry

end smallest_number_plus_3_divisible_by_18_70_100_21_l178_178571


namespace log_expression_correct_l178_178752

-- The problem involves logarithms and exponentials
theorem log_expression_correct : 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 50) + (Real.log 25) + Real.exp (Real.log 3) = 5 := 
  by 
    sorry

end log_expression_correct_l178_178752


namespace siblings_water_intake_l178_178540

theorem siblings_water_intake (Theo_water : ℕ) (Mason_water : ℕ) (Roxy_water : ℕ) : 
  Theo_water = 8 → Mason_water = 7 → Roxy_water = 9 → 
  (7 * Theo_water + 7 * Mason_water + 7 * Roxy_water = 168) :=
begin
  intros hTheo hMason hRoxy,
  rw [hTheo, hMason, hRoxy],
  norm_num,
end

end siblings_water_intake_l178_178540


namespace simplify_expr_l178_178349

def expr (y : ℝ) := y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8)

theorem simplify_expr (y : ℝ) : expr y = 4 * y^3 - 6 * y^2 + 15 * y - 48 :=
by
  sorry

end simplify_expr_l178_178349


namespace complement_A_correct_l178_178138

def A : Set ℝ := {x | 1 - (8 / (x - 2)) < 0}

def complement_A : Set ℝ := {x | x ≤ 2 ∨ x ≥ 10}

theorem complement_A_correct : (Aᶜ = complement_A) :=
by {
  -- Placeholder for the necessary proof
  sorry
}

end complement_A_correct_l178_178138


namespace tangent_line_at_2_number_of_zeros_l178_178136

noncomputable def f (x : ℝ) := 3 * Real.log x + (1/2) * x^2 - 4 * x + 1

theorem tangent_line_at_2 :
  let x := 2
  ∃ k b : ℝ, (∀ y : ℝ, y = k * x + b) ∧ (k = -1/2) ∧ (b = 3 * Real.log 2 - 5) ∧ (∀ x y : ℝ, (y - (3 * Real.log 2 - 5) = -1/2 * (x - 2)) ↔ (x + 2 * y - 6 * Real.log 2 + 8 = 0)) :=
by
  sorry

noncomputable def g (x : ℝ) (m : ℝ) := f x - m

theorem number_of_zeros (m : ℝ) :
  let g := g
  (m > -5/2 ∨ m < 3 * Real.log 3 - 13/2 → ∃ x : ℝ, g x = 0) ∧ 
  (m = -5/2 ∨ m = 3 * Real.log 3 - 13/2 → ∃ x y : ℝ, g x = 0 ∧ g y = 0 ∧ x ≠ y) ∧
  (3 * Real.log 3 - 13/2 < m ∧ m < -5/2 → ∃ x y z : ℝ, g x = 0 ∧ g y = 0 ∧ g z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) :=
by
  sorry

end tangent_line_at_2_number_of_zeros_l178_178136


namespace solve_for_y_l178_178350

-- Define the variables and conditions
variable (y : ℝ)
variable (h_pos : y > 0)
variable (h_seq : (4 + y^2 = 2 * y^2 ∧ y^2 + 25 = 2 * y^2))

-- State the theorem
theorem solve_for_y : y = Real.sqrt 14.5 :=
by sorry

end solve_for_y_l178_178350


namespace range_of_a_l178_178139

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := {x | abs (x - 2) ≤ a}
def set_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

lemma disjoint_sets (A B : Set ℝ) : A ∩ B = ∅ :=
  sorry

theorem range_of_a (h : set_A a ∩ set_B = ∅) : a < 1 :=
  by
  sorry

end range_of_a_l178_178139


namespace proof_inequality_l178_178023

variable {a b c : ℝ}

theorem proof_inequality (h : a * b < 0) : a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := by
  sorry

end proof_inequality_l178_178023


namespace num_first_graders_in_class_l178_178058

def numKindergartners := 14
def numSecondGraders := 4
def totalStudents := 42

def numFirstGraders : Nat := totalStudents - (numKindergartners + numSecondGraders)

theorem num_first_graders_in_class :
  numFirstGraders = 24 :=
by
  sorry

end num_first_graders_in_class_l178_178058


namespace elberta_money_l178_178629

theorem elberta_money (GrannySmith Anjou Elberta : ℝ)
  (h_granny : GrannySmith = 100)
  (h_anjou : Anjou = 1 / 4 * GrannySmith)
  (h_elberta : Elberta = Anjou + 5) : Elberta = 30 := by
  sorry

end elberta_money_l178_178629


namespace range_of_a_l178_178898

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 0 → x^2 + 2 * x - 3 + a ≤ 0) ↔ a ≤ -12 :=
by
  sorry

end range_of_a_l178_178898


namespace fred_change_received_l178_178035

theorem fred_change_received :
  let ticket_price := 5.92
  let ticket_count := 2
  let borrowed_movie_price := 6.79
  let amount_paid := 20.00
  let total_cost := (ticket_price * ticket_count) + borrowed_movie_price
  let change := amount_paid - total_cost
  change = 1.37 :=
by
  sorry

end fred_change_received_l178_178035


namespace find_real_number_x_l178_178839

theorem find_real_number_x 
    (x : ℝ) 
    (i : ℂ) 
    (h_imaginary_unit : i*i = -1) 
    (h_equation : (1 - 2*i)*(x + i) = 4 - 3*i) : 
    x = 2 := 
by
  sorry

end find_real_number_x_l178_178839


namespace tangent_product_value_l178_178009

theorem tangent_product_value (A B : ℝ) (hA : A = 20) (hB : B = 25) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
sorry

end tangent_product_value_l178_178009


namespace pensioners_painting_conditions_l178_178740

def boardCondition (A Z : ℕ) : Prop :=
(∀ x y, (∃ i j, i ≤ 1 ∧ j ≤ 1 ∧ (x + 3 = A) ∧ (i ≤ 2 ∧ j ≤ 4 ∨ i ≤ 4 ∧ j ≤ 2) → x + 2 * y = Z))

theorem pensioners_painting_conditions (A Z : ℕ) :
  (boardCondition A Z) ↔ (A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8) :=
sorry

end pensioners_painting_conditions_l178_178740


namespace at_least_one_solves_l178_178492

-- Given probabilities
def pA : ℝ := 0.8
def pB : ℝ := 0.6

-- Probability that at least one solves the problem
def prob_at_least_one_solves : ℝ := 1 - ((1 - pA) * (1 - pB))

-- Statement: Prove that the probability that at least one solves the problem is 0.92
theorem at_least_one_solves : prob_at_least_one_solves = 0.92 :=
by
  -- Proof steps would go here
  sorry

end at_least_one_solves_l178_178492


namespace arithmetic_sequence_properties_l178_178163

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) :
  (∀ n, a n = -2 * n + 15) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 7 → ∑ i in finset.range n, abs (a (i + 1)) = -n^2 + 14*n) ∧
  (∀ n, n ≥ 8 → ∑ i in finset.range n, abs (a (i + 1)) = n^2 - 14*n + 98) :=
by
  sorry  -- sorry to skip the proof

end arithmetic_sequence_properties_l178_178163


namespace tangents_intersect_on_AP_l178_178465

noncomputable def geometric_setup (A B C D E P M N : Point) (w : Circle) : Prop :=
  ∃ ABC_triangle_tangent : Triangle ABC,
  ∃ is_on_AB_D : D ∈ LineSegment AB,
  ∃ is_on_AC_E : E ∈ LineSegment AC,
  ∃ is_parallel_DE_BC : DE ∥ BC,
  ∃ is_intersection_P : P = intersection_point (Line BE) (Line CD),
  ∃ second_intersection_M : M = second_intersection (Circle_through APD) (Circle_through BCD),
  ∃ second_intersection_N : N = second_intersection (Circle_through APE) (Circle_through BCE),
  ∃ circle_through_MN : w = Circle_through M N,
  ∃ is_tangent_w_BC : tangent w Line BC,
  True

theorem tangents_intersect_on_AP (A B C D E P M N : Point) (w : Circle) :
  geometric_setup A B C D E P M N w →
  intersects (Line (tangent w M)) (Line (tangent w N)) ∈ Line AP :=
by
  intro h
  sorry

end tangents_intersect_on_AP_l178_178465


namespace missed_bus_time_l178_178548

theorem missed_bus_time (T: ℕ) (speed_ratio: ℚ) (T_slow: ℕ) (missed_time: ℕ) : 
  T = 16 → speed_ratio = 4/5 → T_slow = (5/4) * T → missed_time = T_slow - T → missed_time = 4 :=
by
  sorry

end missed_bus_time_l178_178548


namespace roots_of_cubic_eq_l178_178420

theorem roots_of_cubic_eq (r s t a b c d : ℂ) (h1 : a ≠ 0) (h2 : r ≠ 0) (h3 : s ≠ 0) 
  (h4 : t ≠ 0) (hrst : ∀ x : ℂ, a * x ^ 3 + b * x ^ 2 + c * x + d = 0 → (x = r ∨ x = s ∨ x = t) ∧ (x = r <-> r + s + t - x = -b / a)) 
  (Vieta1 : r + s + t = -b / a) (Vieta2 : r * s + r * t + s * t = c / a) (Vieta3 : r * s * t = -d / a) :
  (1 / r ^ 3 + 1 / s ^ 3 + 1 / t ^ 3 = c ^ 3 / d ^ 3) := 
by sorry

end roots_of_cubic_eq_l178_178420


namespace equal_books_for_students_l178_178059

-- Define the conditions
def num_girls : ℕ := 15
def num_boys : ℕ := 10
def total_books : ℕ := 375
def books_for_girls : ℕ := 225
def books_for_boys : ℕ := total_books - books_for_girls -- Calculate books for boys

-- Define the theorem
theorem equal_books_for_students :
  books_for_girls / num_girls = 15 ∧ books_for_boys / num_boys = 15 :=
by
  sorry

end equal_books_for_students_l178_178059


namespace rectangle_area_l178_178788

theorem rectangle_area (area_square : ℝ) 
  (width_rectangle : ℝ) (length_rectangle : ℝ)
  (h1 : area_square = 16)
  (h2 : width_rectangle^2 = area_square)
  (h3 : length_rectangle = 3 * width_rectangle) :
  width_rectangle * length_rectangle = 48 := by sorry

end rectangle_area_l178_178788


namespace exists_divisor_c_of_f_l178_178031

theorem exists_divisor_c_of_f (f : ℕ → ℕ) 
  (h₁ : ∀ n, f n ≥ 2)
  (h₂ : ∀ m n, f (m + n) ∣ (f m + f n)) :
  ∃ c > 1, ∀ n, c ∣ f n :=
sorry

end exists_divisor_c_of_f_l178_178031


namespace math_problem_l178_178890

theorem math_problem (x y : ℤ) (a b : ℤ) (h1 : x - 5 = 7 * a) (h2 : y + 7 = 7 * b) (h3 : (x ^ 2 + y ^ 3) % 11 = 0) : 
  ((y - x) / 13) = 13 :=
sorry

end math_problem_l178_178890


namespace extrema_f_unique_solution_F_l178_178625

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * x^2 - m * Real.log x
noncomputable def F (x : ℝ) (m : ℝ) : ℝ := - (1 / 2) * x^2 + (m + 1) * x - m * Real.log x

theorem extrema_f (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x ≠ y → f x m ≠ f y m) ∧
  (m > 0 → ∃ x₀ > 0, ∀ x > 0, f x₀ m ≤ f x m) :=
sorry

theorem unique_solution_F (m : ℝ) (h : m ≥ 1) :
  ∃ x₀ > 0, ∀ x > 0, F x₀ m = 0 ∧ (F x m = 0 → x = x₀) :=
sorry

end extrema_f_unique_solution_F_l178_178625


namespace minimize_transportation_cost_l178_178075

noncomputable def transportation_minimization
  (m n : ℕ)
  (a : Fin m → ℝ)
  (b : Fin n → ℝ)
  (c : Matrix (Fin m) (Fin n) ℝ)
  (h_balanced : ∑ i, a i = ∑ j, b j) : ℝ :=
  let x : Matrix (Fin m) (Fin n) ℝ := sorry in
  if h_feasible : (∀ i, ∑ j, x i j ≤ a i) ∧ (∀ j, ∑ i, x i j ≥ b j) ∧ (∀ i j, 0 ≤ x i j)
  then ∑ i, ∑ j, c i j * x i j
  else 0    -- This represents the minimum cost when conditions are met, 0 otherwise (eligible for further optimization proofs).

theorem minimize_transportation_cost
  (m n : ℕ)
  (a : Fin m → ℝ)
  (b : Fin n → ℝ)
  (c : Matrix (Fin m) (Fin n) ℝ)
  (h_balanced : ∑ i, a i = ∑ j, b j) :
  ∃ x : Matrix (Fin m) (Fin n) ℝ, 
    (∀ i, ∑ j, x i j ≤ a i) ∧ 
    (∀ j, ∑ i, x i j ≥ b j) ∧ 
    (∀ i j, 0 ≤ x i j) ∧
    (∑ i, ∑ j, c i j * x i j = transportation_minimization m n a b c h_balanced) :=
sorry

end minimize_transportation_cost_l178_178075


namespace triangle_ab_length_l178_178022

/-- In triangle ABC, point N lies on side AB such that AN = 3NB; the median AM intersects CN at point O.
Given AM = 7 cm, CN = 7 cm, and ∠NOM = 60°, prove that AB = 4√7 cm. -/
theorem triangle_ab_length (A B C N M O : Point) (x : Real)
(hN : x > 0)  -- NB is x, thus x > 0
(h_AN_3NB : dist A N = 3 * dist N B)
(h_AM : dist A M = 7)
(h_CN : dist C N = 7)
(h_nom_60 : ∠ N O M = 60) :
  dist A B = 4 * Real.sqrt 7 :=
sorry

end triangle_ab_length_l178_178022


namespace lines_in_4x4_grid_l178_178645

theorem lines_in_4x4_grid :
  let n := 4
  let total_points := n * n
  let choose_two_points := total_points.choose 2
  let horizontal_and_vertical_lines := n + n
  let diagonal_lines := 6 -- based on detailed breakdown
  let adjustment_for_lines_through_four_points := 8 * 3
  let adjustment_for_lines_through_three_points := 4 * 2
  let initial_line_count := choose_two_points
  let adjusted_line_count := initial_line_count - adjustment_for_lines_through_four_points - adjustment_for_lines_through_three_points
  in adjusted_line_count = 88 := 
by {
  exact 88 // Placeholder proof statement
  sorry
}

end lines_in_4x4_grid_l178_178645


namespace gunny_bag_capacity_in_tons_l178_178036

def ton_to_pounds := 2200
def pound_to_ounces := 16
def packets := 1760
def packet_weight_pounds := 16
def packet_weight_ounces := 4

theorem gunny_bag_capacity_in_tons :
  ((packets * (packet_weight_pounds + (packet_weight_ounces / pound_to_ounces))) / ton_to_pounds) = 13 :=
sorry

end gunny_bag_capacity_in_tons_l178_178036


namespace factorial_sum_simplify_l178_178803

theorem factorial_sum_simplify :
  7 * (Nat.factorial 7) + 5 * (Nat.factorial 5) + 3 * (Nat.factorial 3) + (Nat.factorial 3) = 35904 :=
by
  sorry

end factorial_sum_simplify_l178_178803


namespace probability_at_least_three_aces_l178_178375

open Nat

noncomputable def combination (n k : ℕ) : ℕ :=
  n.choose k

theorem probability_at_least_three_aces :
  (combination 4 3 * combination 48 2 + combination 4 4 * combination 48 1) / combination 52 5 = (combination 4 3 * combination 48 2 + combination 4 4 * combination 48 1 : ℚ) / combination 52 5 :=
by
  sorry

end probability_at_least_three_aces_l178_178375


namespace principal_trebled_after_5_years_l178_178533

theorem principal_trebled_after_5_years (P R: ℝ) (n: ℝ) :
  (P * R * 10 / 100 = 700) →
  ((P * R * n + 3 * P * R * (10 - n)) / 100 = 1400) →
  n = 5 :=
by
  intros h1 h2
  sorry

end principal_trebled_after_5_years_l178_178533


namespace grilled_cheese_sandwiches_l178_178160

-- Define the number of ham sandwiches Joan makes
def ham_sandwiches := 8

-- Define the cheese requirements for each type of sandwich
def cheddar_for_ham := 1
def swiss_for_ham := 1
def cheddar_for_grilled := 2
def gouda_for_grilled := 1

-- Total cheese used
def total_cheddar := 40
def total_swiss := 20
def total_gouda := 30

-- Prove the number of grilled cheese sandwiches Joan makes
theorem grilled_cheese_sandwiches (ham_sandwiches : ℕ) (cheddar_for_ham : ℕ) (swiss_for_ham : ℕ)
                                  (cheddar_for_grilled : ℕ) (gouda_for_grilled : ℕ)
                                  (total_cheddar : ℕ) (total_swiss : ℕ) (total_gouda : ℕ) :
    (total_cheddar - ham_sandwiches * cheddar_for_ham) / cheddar_for_grilled = 16 :=
by
  sorry

end grilled_cheese_sandwiches_l178_178160


namespace problem_solution_l178_178284

theorem problem_solution :
  ∀ (x y : ℚ), 
  4 * x + y = 20 ∧ x + 2 * y = 17 → 
  5 * x^2 + 18 * x * y + 5 * y^2 = 696 + 5/7 := 
by 
  sorry

end problem_solution_l178_178284


namespace angle_ABC_measure_l178_178707

theorem angle_ABC_measure
  (CBD : ℝ)
  (ABC ABD : ℝ)
  (h1 : CBD = 90)
  (h2 : ABC + ABD + CBD = 270)
  (h3 : ABD = 100) : 
  ABC = 80 :=
by
  -- Given:
  -- CBD = 90
  -- ABC + ABD + CBD = 270
  -- ABD = 100
  sorry

end angle_ABC_measure_l178_178707


namespace attendance_rate_comparison_l178_178081

theorem attendance_rate_comparison (attendees_A total_A attendees_B total_B : ℕ) 
  (hA : (attendees_A / total_A: ℚ) > (attendees_B / total_B: ℚ)) : 
  (attendees_A > attendees_B) → false :=
by
  sorry

end attendance_rate_comparison_l178_178081


namespace infinite_n_perfect_squares_l178_178731

-- Define the condition that k is a positive natural number and k >= 2
variable (k : ℕ) (hk : 2 ≤ k) 

-- Define the statement asserting the existence of infinitely many n such that both kn + 1 and (k+1)n + 1 are perfect squares
theorem infinite_n_perfect_squares : ∀ k : ℕ, (2 ≤ k) → ∃ n : ℕ, ∀ m : ℕ, (2 ≤ k) → k * n + 1 = m * m ∧ (k + 1) * n + 1 = (m + k) * (m + k) := 
by
  sorry

end infinite_n_perfect_squares_l178_178731


namespace count_distribution_schemes_l178_178881

theorem count_distribution_schemes :
  let total_pieces := 7
  let pieces_A_B := 2 + 2
  let remaining_pieces := total_pieces - pieces_A_B
  let communities := 5

  -- Number of ways to distribute 7 pieces of equipment such that communities A and B receive at least 2 pieces each
  let ways_one_community := 5
  let ways_two_communities := 20  -- 2 * (choose 5 2)
  let ways_three_communities := 10  -- (choose 5 3)

  ways_one_community + ways_two_communities + ways_three_communities = 35 :=
by
  -- The actual proof steps are omitted here.
  sorry

end count_distribution_schemes_l178_178881


namespace tiles_needed_l178_178380

-- Definitions of the given conditions
def side_length_smaller_tile : ℝ := 0.3
def number_smaller_tiles : ℕ := 500
def side_length_larger_tile : ℝ := 0.5

-- Statement to prove the required number of larger tiles
theorem tiles_needed (x : ℕ) :
  side_length_larger_tile * side_length_larger_tile * x =
  side_length_smaller_tile * side_length_smaller_tile * number_smaller_tiles →
  x = 180 :=
by
  sorry

end tiles_needed_l178_178380


namespace range_of_b_l178_178292

open Real

noncomputable def f (x b : ℝ) := exp x * (x - b)
noncomputable def f'' (x b : ℝ) := exp x * (x - b + 2)

theorem range_of_b (b : ℝ) :
  (∃ x ∈ Icc (1/2) 2, f(x, b) + x * f''(x, b) > 0) → b < 8/3 :=
by
  sorry

end range_of_b_l178_178292


namespace part1_domain_of_f_part2_inequality_l178_178333

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (abs (x + 1) + abs (x - 1) - 4)

theorem part1_domain_of_f : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by 
  sorry

theorem part2_inequality (a b : ℝ) (h_a : -2 < a) (h_a' : a < 2) (h_b : -2 < b) (h_b' : b < 2) 
  : 2 * abs (a + b) < abs (4 + a * b) :=
by 
  sorry

end part1_domain_of_f_part2_inequality_l178_178333


namespace ellipse_equation_l178_178132

def major_axis_length (a : ℝ) := 2 * a = 8
def eccentricity (c a : ℝ) := c / a = 3 / 4

theorem ellipse_equation (a b c x y : ℝ) (h1 : major_axis_length a)
    (h2 : eccentricity c a) (h3 : b^2 = a^2 - c^2) :
    (x^2 / 16 + y^2 / 7 = 1 ∨ x^2 / 7 + y^2 / 16 = 1) :=
by
  sorry

end ellipse_equation_l178_178132


namespace ants_need_more_hours_l178_178405

theorem ants_need_more_hours (initial_sugar : ℕ) (removal_rate : ℕ) (hours_spent : ℕ) : 
  initial_sugar = 24 ∧ removal_rate = 4 ∧ hours_spent = 3 → 
  (initial_sugar - removal_rate * hours_spent) / removal_rate = 3 :=
by
  intro h
  sorry

end ants_need_more_hours_l178_178405


namespace algebraic_expression_value_l178_178865

noncomputable def algebraic_expression (a b c d : ℝ) : ℝ :=
  a^5 / ((a - b) * (a - c) * (a - d)) +
  b^5 / ((b - a) * (b - c) * (b - d)) +
  c^5 / ((c - a) * (c - b) * (c - d)) +
  d^5 / ((d - a) * (d - b) * (d - c))

theorem algebraic_expression_value {a b c d : ℝ} 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b + c + d = 3) 
  (h_sum_sq : a^2 + b^2 + c^2 + d^2 = 45) : 
  algebraic_expression a b c d = -9 :=
by
  sorry

end algebraic_expression_value_l178_178865


namespace prove_all_perfect_squares_l178_178825

noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k^2 = n

noncomputable def all_distinct (l : List ℕ) : Prop :=
l.Nodup

noncomputable def pairwise_products_are_perfect_squares (l : List ℕ) : Prop :=
∀ i j, i < l.length → j < l.length → i ≠ j → is_perfect_square (l.nthLe i sorry * l.nthLe j sorry)

theorem prove_all_perfect_squares :
  ∀ l : List ℕ, l.length = 25 →
  (∀ x ∈ l, x ≤ 1000 ∧ 0 < x) →
  all_distinct l →
  pairwise_products_are_perfect_squares l →
  ∀ x ∈ l, is_perfect_square x := 
by
  intros l h1 h2 h3 h4
  sorry

end prove_all_perfect_squares_l178_178825


namespace find_valid_n_l178_178807

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_valid_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : is_square (n^2 + (n + 1)^2)) :
  n = 3 ∨ n = 20 ∨ n = 119 :=
by
  sorry

end find_valid_n_l178_178807


namespace find_x0_l178_178966

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := 3 * x - 5

theorem find_x0 :
  (∃ x0 : ℝ, f (g x0) = 1) → (∃ x0 : ℝ, x0 = 4/3) :=
by
  sorry

end find_x0_l178_178966


namespace min_value_problem_l178_178131

theorem min_value_problem 
  (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ (min_val : ℝ), min_val = 2 * x + 3 * y^2 ∧ min_val = 8 / 9 :=
by
  sorry

end min_value_problem_l178_178131


namespace total_votes_is_240_l178_178931

variable {x : ℕ} -- Total number of votes (natural number)
variable {S : ℤ} -- Score (integer)

-- Given conditions
axiom score_condition : S = 120
axiom votes_condition : 3 * x / 4 - x / 4 = S

theorem total_votes_is_240 : x = 240 :=
by
  -- Proof should go here
  sorry

end total_votes_is_240_l178_178931


namespace right_triangle_OAB_condition_l178_178445

theorem right_triangle_OAB_condition
  (a b : ℝ)
  (h1: a ≠ 0) 
  (h2: b ≠ 0) :
  (b - a^3) * (b - a^3 - 1/a) = 0 :=
sorry

end right_triangle_OAB_condition_l178_178445


namespace find_length_AC_l178_178609

-- Geometry setup
variables {A B C P T Q M : ℝ} [ordered_field ℝ]

-- Points coordinates (simplistic 1D abstraction for the sake of Lean formalization)
-- Assuming without loss of generality certain coordinates for simplification
axiom A : ℝ
axiom B : ℝ 
axiom C : ℝ 
axiom P : ℝ 
axiom T : ℝ 
axiom Q : ℝ 
axiom M : ℝ 

-- Conditions from problem
axiom median_AM : (A + M) / 2 = A -- Assume A is coordinate at 0 for simplicity
axiom parallel_PT_AC : True -- Line PT is parallel to AC (It means same slope, but True suffices here)
axiom PQ_length : P - Q = 3
axiom QT_length : Q - T = 5

-- Proof that AC = 11 given conditions
theorem find_length_AC (h : (PQ_length) ∧ (QT_length) ∧ (parallel_PT_AC) ∧ (median_AM)) : C = 11 :=
begin
  sorry -- Proof goes here
end

end find_length_AC_l178_178609


namespace side_length_of_base_l178_178186

-- Given conditions
def lateral_face_area := 90 -- Area of one lateral face in square meters
def slant_height := 20 -- Slant height in meters

-- The theorem statement
theorem side_length_of_base 
  (s : ℝ)
  (h : ℝ := slant_height)
  (a : ℝ := lateral_face_area)
  (h_area : 2 * a = s * h) :
  s = 9 := 
sorry

end side_length_of_base_l178_178186


namespace factor_expression_1_factor_expression_2_l178_178038

theorem factor_expression_1 (a b c : ℝ) : a^2 + 2 * a * b + b^2 + a * c + b * c = (a + b) * (a + b + c) :=
  sorry

theorem factor_expression_2 (a x y : ℝ) : 4 * a^2 - x^2 + 4 * x * y - 4 * y^2 = (2 * a + x - 2 * y) * (2 * a - x + 2 * y) :=
  sorry

end factor_expression_1_factor_expression_2_l178_178038


namespace lines_in_4_by_4_grid_l178_178684

theorem lines_in_4_by_4_grid : 
  let n := 4 in
  number_of_lines_at_least_two_points (grid_of_lattice_points n) = 96 :=
by sorry

end lines_in_4_by_4_grid_l178_178684


namespace distance_from_minus_one_is_four_or_minus_six_l178_178049

theorem distance_from_minus_one_is_four_or_minus_six :
  {x : ℝ | abs (x + 1) = 5} = {-6, 4} :=
sorry

end distance_from_minus_one_is_four_or_minus_six_l178_178049


namespace restore_triangle_Nagel_point_l178_178494

-- Define the variables and types involved
variables {Point : Type}

-- Assume a structure to capture the properties of a triangle
structure Triangle (Point : Type) :=
(A B C : Point)

-- Define the given conditions
variables (N B E : Point)

-- Statement of the main Lean theorem to reconstruct the triangle ABC
theorem restore_triangle_Nagel_point 
    (N B E : Point) :
    ∃ (ABC : Triangle Point), 
      (ABC).B = B ∧
      -- Additional properties of the triangle to be stated here
      sorry
    :=
sorry

end restore_triangle_Nagel_point_l178_178494


namespace y_coordinate_of_P_l178_178045

theorem y_coordinate_of_P (x y : ℝ) (h1 : |y| = 1/2 * |x|) (h2 : |x| = 12) :
  y = 6 ∨ y = -6 :=
sorry

end y_coordinate_of_P_l178_178045


namespace general_formula_for_sequence_l178_178335

theorem general_formula_for_sequence :
  ∀ (a : ℕ → ℕ), (a 0 = 1) → (a 1 = 1) →
  (∀ n, 2 ≤ n → a n = 2 * a (n - 1) - a (n - 2)) →
  ∀ n, a n = (2^n - 1)^2 :=
by
  sorry

end general_formula_for_sequence_l178_178335


namespace number_of_lines_at_least_two_points_4_by_4_grid_l178_178658

-- Definition of 4-by-4 grid
def grid : Type := (Fin 4) × (Fin 4)

-- Definition of a line passing through at least two points in this grid
def line_through_at_least_two_points (points : List grid) : Prop := 
  points.length ≥ 2
  ∧ ∃ m b, ∀ (x y : Fin 4 × Fin 4), (x ∈ points ∧ y ∈ points) → (y.snd : ℕ) = m * (x.fst : ℕ) + b

-- Defining the total number of points choosing 2 out of 16
def total_points : Nat := Nat.choose 16 2

-- Defining the overcount of vertical, horizontal,
-- major diagonals, and secondary diagonals lines
def overcount : Nat := 8 + 2 + 4

-- Total distinct count of lines passing through at least two points
def correct_answer : Nat := total_points - overcount

-- Main theorem stating that the total count is 106
theorem number_of_lines_at_least_two_points_4_by_4_grid : correct_answer = 106 := 
by
  sorry

end number_of_lines_at_least_two_points_4_by_4_grid_l178_178658


namespace simplify_expression_l178_178919

theorem simplify_expression :
  ((0.3 * 0.2) / (0.4 * 0.5)) - (0.1 * 0.6) = 0.24 :=
by
  sorry

end simplify_expression_l178_178919


namespace find_other_endpoint_l178_178525

theorem find_other_endpoint (x1 y1 x_m y_m x y : ℝ) 
  (h1 : (x_m, y_m) = (3, 7))
  (h2 : (x1, y1) = (0, 11)) :
  (x, y) = (6, 3) ↔ (x_m = (x1 + x) / 2 ∧ y_m = (y1 + y) / 2) :=
by
  simp at h1 h2
  simp
  sorry

end find_other_endpoint_l178_178525


namespace siblings_water_intake_l178_178539

theorem siblings_water_intake 
  (theo_daily : ℕ := 8) 
  (mason_daily : ℕ := 7) 
  (roxy_daily : ℕ := 9) 
  (days_in_week : ℕ := 7) 
  : (theo_daily + mason_daily + roxy_daily) * days_in_week = 168 := 
by 
  sorry

end siblings_water_intake_l178_178539


namespace sixth_term_of_geometric_seq_l178_178584

-- conditions
def is_geometric_sequence (seq : ℕ → ℕ) := 
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = seq n * r

def first_term (seq : ℕ → ℕ) := seq 1 = 3
def fifth_term (seq : ℕ → ℕ) := seq 5 = 243

-- question to be proved
theorem sixth_term_of_geometric_seq (seq : ℕ → ℕ) 
  (h_geom : is_geometric_sequence seq) 
  (h_first : first_term seq) 
  (h_fifth : fifth_term seq) : 
  seq 6 = 729 :=
sorry

end sixth_term_of_geometric_seq_l178_178584


namespace inequality_transform_l178_178614

theorem inequality_transform {a b c d e : ℝ} (hab : a > b) (hb0 : b > 0) 
  (hcd : c < d) (hd0 : d < 0) (he : e < 0) : 
  e / (a - c)^2 > e / (b - d)^2 :=
by 
  sorry

end inequality_transform_l178_178614


namespace total_distance_is_810_l178_178249

-- Define conditions as constants
constant first_day_distance : ℕ := 100
constant second_day_distance : ℕ := 3 * first_day_distance
constant third_day_distance : ℕ := second_day_distance + 110

-- The total distance traveled in three days
constant total_distance : ℕ := first_day_distance + second_day_distance + third_day_distance

-- The theorem to prove
theorem total_distance_is_810 : total_distance = 810 := 
  by
    -- The proof steps would go here, but we insert sorry to indicate the proof is not provided
    sorry

end total_distance_is_810_l178_178249


namespace suraj_average_increase_l178_178043

theorem suraj_average_increase
  (A : ℝ)
  (h1 : 9 * A + 200 = 10 * 128) :
  128 - A = 8 :=
by
  sorry

end suraj_average_increase_l178_178043


namespace geometric_sequence_sum_l178_178329

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 0 + a 1 = 1) (h2 : a 1 + a 2 = 2) : a 5 + a 6 = 32 :=
sorry

end geometric_sequence_sum_l178_178329


namespace maximum_value_l178_178430

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem maximum_value : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ f 1 :=
by
  intros x hx
  sorry

end maximum_value_l178_178430


namespace number_of_children_l178_178062

theorem number_of_children (total_people : ℕ) (num_adults num_children : ℕ)
  (h1 : total_people = 42)
  (h2 : num_children = 2 * num_adults)
  (h3 : num_adults + num_children = total_people) :
  num_children = 28 :=
by
  sorry

end number_of_children_l178_178062


namespace part1_part2_l178_178447

open Set

variable {m x : ℝ}

def A (m : ℝ) : Set ℝ := { x | x^2 - (m+1)*x + m = 0 }
def B (m : ℝ) : Set ℝ := { x | x * m - 1 = 0 }

theorem part1 (h : A m ⊆ B m) : m = 1 :=
by
  sorry

theorem part2 (h : B m ⊂ A m) : m = 0 ∨ m = -1 :=
by
  sorry

end part1_part2_l178_178447


namespace seventeen_divides_9x_plus_5y_l178_178332

theorem seventeen_divides_9x_plus_5y (x y : ℤ) (h : 17 ∣ (2 * x + 3 * y)) : 17 ∣ (9 * x + 5 * y) :=
sorry

end seventeen_divides_9x_plus_5y_l178_178332


namespace find_function_f_l178_178334

-- Define the problem in Lean 4
theorem find_function_f (f : ℝ → ℝ) : 
  (f 0 = 1) → 
  ((∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2)) → 
  (∀ x : ℝ, f x = x + 1) :=
  by
    intros h₁ h₂
    sorry

end find_function_f_l178_178334


namespace find_a_l178_178194

open Real

theorem find_a :
  ∃ a : ℝ, (1/5) * (0.5 + a + 1 + 1.4 + 1.5) = 0.28 * 3 + 0.16 := by
  use 0.6
  sorry

end find_a_l178_178194


namespace turtle_population_2002_l178_178359

theorem turtle_population_2002 (k : ℝ) (y : ℝ)
  (h1 : 58 + k * 92 = y)
  (h2 : 179 - 92 = k * y) 
  : y = 123 :=
by
  sorry

end turtle_population_2002_l178_178359


namespace min_real_roots_l178_178484

open Polynomial

theorem min_real_roots {g : Polynomial ℝ} (degree_g : g.degree = 504)
  (h : ∃ s : Fin 504 → ℂ, ∀ i, g.eval s i = 0 ∧ (∃! j, ∀ k, |s i| = |s j| → j = k) ∧ (Finset.image (λ i, |s i|) Finset.univ).card = 252)
  : ∃ r : ℂ → ℕ, ∃ m ≤ 252, real_roots_count g = 126 :=
sorry

end min_real_roots_l178_178484


namespace mean_of_two_remaining_numbers_l178_178893

theorem mean_of_two_remaining_numbers (a b c: ℝ) (h1: (a + b + c + 100) / 4 = 90) (h2: a = 70) : (b + c) / 2 = 95 := by
  sorry

end mean_of_two_remaining_numbers_l178_178893


namespace number_of_lines_in_4_by_4_grid_l178_178655

/-- A 4-by-4 grid of lattice points -/
def lattice_points_4x4 : set (ℕ × ℕ) :=
  {(i, j) | i < 4 ∧ j < 4}

/-- A line in the Euclidean plane -/
def is_line (p1 p2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | ∃ λ : ℝ, p = (λ * (p2.1 - p1.1) + p1.1, λ * (p2.2 - p1.2) + p1.2)}

noncomputable def count_lines_through_points (points : set (ℕ × ℕ)) : ℕ :=
  /- counting logic to be implemented -/
  sorry

theorem number_of_lines_in_4_by_4_grid : count_lines_through_points lattice_points_4x4 = 70 :=
  sorry

end number_of_lines_in_4_by_4_grid_l178_178655


namespace number_of_lines_in_4_by_4_grid_l178_178653

/-- A 4-by-4 grid of lattice points -/
def lattice_points_4x4 : set (ℕ × ℕ) :=
  {(i, j) | i < 4 ∧ j < 4}

/-- A line in the Euclidean plane -/
def is_line (p1 p2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | ∃ λ : ℝ, p = (λ * (p2.1 - p1.1) + p1.1, λ * (p2.2 - p1.2) + p1.2)}

noncomputable def count_lines_through_points (points : set (ℕ × ℕ)) : ℕ :=
  /- counting logic to be implemented -/
  sorry

theorem number_of_lines_in_4_by_4_grid : count_lines_through_points lattice_points_4x4 = 70 :=
  sorry

end number_of_lines_in_4_by_4_grid_l178_178653


namespace kelly_peanut_weight_l178_178859

-- Define the total weight of snacks and the weight of raisins
def total_snacks_weight : ℝ := 0.5
def raisins_weight : ℝ := 0.4

-- Define the weight of peanuts as the remaining part
def peanuts_weight : ℝ := total_snacks_weight - raisins_weight

-- Theorem stating Kelly bought 0.1 pounds of peanuts
theorem kelly_peanut_weight : peanuts_weight = 0.1 :=
by
  -- proof would go here
  sorry

end kelly_peanut_weight_l178_178859


namespace percentage_increase_is_2_l178_178934

def alan_price := 2000
def john_price := 2040
def percentage_increase (alan_price : ℕ) (john_price : ℕ) : ℕ := (john_price - alan_price) * 100 / alan_price

theorem percentage_increase_is_2 (alan_price john_price : ℕ) (h₁ : alan_price = 2000) (h₂ : john_price = 2040) :
  percentage_increase alan_price john_price = 2 := by
  rw [h₁, h₂]
  sorry

end percentage_increase_is_2_l178_178934


namespace find_sample_size_l178_178396

def ratio_A : ℕ := 2
def ratio_B : ℕ := 3
def ratio_C : ℕ := 5
def total_ratio : ℕ := ratio_A + ratio_B + ratio_C
def num_B_selected : ℕ := 24

theorem find_sample_size : ∃ n : ℕ, num_B_selected * total_ratio = ratio_B * n :=
by
  sorry

end find_sample_size_l178_178396


namespace largest_digit_divisible_by_6_l178_178212

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N = 8 ∧ (45670 + N) % 6 = 0 :=
sorry

end largest_digit_divisible_by_6_l178_178212


namespace fraction_of_Jeff_money_l178_178389

/-- Emma, Daya, Jeff, and Brenda's monetary amounts and relationships -/
def Emma_money : ℤ := 8
def Daya_money : ℤ := (Emma_money + (Emma_money / 4)) -- 25% more than Emma
def Brenda_money : ℤ := 8
def Jeff_money : ℤ := (Brenda_money - 4) -- 4 less than Brenda

/-- The fraction of money Jeff has compared to Daya -/
theorem fraction_of_Jeff_money :
  (Jeff_money : ℚ) / (Daya_money : ℚ) = 2 / 5 :=
by
  sorry

end fraction_of_Jeff_money_l178_178389


namespace complex_exp_l178_178975

theorem complex_exp {i : ℂ} (h : i^2 = -1) : (1 + i)^30 + (1 - i)^30 = 0 := by
  sorry

end complex_exp_l178_178975


namespace value_of_a_l178_178619

theorem value_of_a (a : ℝ) (h : (a - 3) * x ^ |a - 2| + 4 = 0) : a = 1 :=
by
  sorry

end value_of_a_l178_178619


namespace total_books_l178_178630

theorem total_books (hbooks : ℕ) (fbooks : ℕ) (gbooks : ℕ)
  (Harry_books : hbooks = 50)
  (Flora_books : fbooks = 2 * hbooks)
  (Gary_books : gbooks = hbooks / 2) :
  hbooks + fbooks + gbooks = 175 := by
  sorry

end total_books_l178_178630


namespace sharon_prob_discard_card_l178_178089

theorem sharon_prob_discard_card {cards : Fin 49 → (Fin 7 × Fin 7)}
  (h_unique : ∀ i j, i ≠ j → cards i ≠ cards j)
  (h_colors : ∀ c : Fin 7, ∃ i, (cards i).1 = c)
  (h_numbers : ∀ n : Fin 7, ∃ i, (cards i).2 = n)
  (h_selection : ∀ (s : Finset (Fin 49)), s.card = 8 →
                 ∃ t : Finset (Fin 49), t ⊆ s ∧ t.card = 7 ∧
                 ∀ c : Fin 7, ∃ i ∈ t, (cards i).1 = c ∧
                 ∀ n : Fin 7, ∃ i ∈ t, (cards i).2 = n):
  let p := 4
  let q := 9
  Nat.gcd p q = 1 ∧ (p : ℚ) / q = 4 / 9 ∧ p + q = 13 :=
by
  sorry

end sharon_prob_discard_card_l178_178089


namespace at_least_one_has_two_distinct_roots_l178_178698

theorem at_least_one_has_two_distinct_roots
  (p q1 q2 : ℝ)
  (h : p = q1 + q2 + 1) :
  (1 - 4 * q1 > 0) ∨ ((q1 + q2 + 1) ^ 2 - 4 * q2 > 0) :=
by sorry

end at_least_one_has_two_distinct_roots_l178_178698


namespace frog_jump_probability_l178_178084

open scoped ProbabilityTheory

/-
  Given:
  - The frog makes 4 jumps in a pond.
  - The first three jumps are 1 meter each.
  - The last jump can be either 1 meter or 2 meters long, each with equal probability.
  - The directions of all jumps are chosen independently at random.

  Prove that:
  - The probability that the frog’s final position is no more than 1.5 meters from its starting position is 1/6.
-/
theorem frog_jump_probability :
  let u v w : ℝ^2 := sorry -- random unit vectors representing the first three jumps
  let z1 z2 : ℝ^2 := sorry -- random unit vectors representing the last jump of length 1 or 2 meters
  let final_position_with_z1 := u + v + w + z1
  let final_position_with_z2 := u + v + w + z2
  let within_1_5_meters (p : ℝ^2) := ∥p∥ ≤ 1.5
  let probability_within_1_5_meters := 
  (1/2) * P (within_1_5_meters final_position_with_z1) +
  (1/2) * P (within_1_5_meters final_position_with_z2)
  in
  probability_within_1_5_meters = 1/6 :=
sorry

end frog_jump_probability_l178_178084


namespace find_triangle_side_value_find_triangle_tan_value_l178_178316

noncomputable def triangle_side_value (A B C : ℝ) (a b c : ℝ) : Prop :=
  C = 2 * Real.pi / 3 ∧
  c = 5 ∧
  a = Real.sqrt 5 * b * Real.sin A ∧
  b = 2 * Real.sqrt 15 / 3

noncomputable def triangle_tan_value (B : ℝ) : Prop :=
  Real.tan (B + Real.pi / 4) = 3

theorem find_triangle_side_value (A B C a b c : ℝ) :
  triangle_side_value A B C a b c := by sorry

theorem find_triangle_tan_value (B : ℝ) :
  triangle_tan_value B := by sorry

end find_triangle_side_value_find_triangle_tan_value_l178_178316


namespace degree_to_radian_conversion_l178_178595

theorem degree_to_radian_conversion : (1440 * (Real.pi / 180) = 8 * Real.pi) := 
by
  sorry

end degree_to_radian_conversion_l178_178595


namespace problem_statement_l178_178450

variable (x y z a b c : ℝ)

-- Conditions
def condition1 := x / a + y / b + z / c = 5
def condition2 := a / x + b / y + c / z = 0

-- Proof statement
theorem problem_statement (h1 : condition1 x y z a b c) (h2 : condition2 x y z a b c) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := 
sorry

end problem_statement_l178_178450


namespace quadratic_solution_difference_l178_178844

theorem quadratic_solution_difference : 
  ∃ a b : ℝ, (a^2 - 12 * a + 20 = 0) ∧ (b^2 - 12 * b + 20 = 0) ∧ (a > b) ∧ (a - b = 8) :=
by
  sorry

end quadratic_solution_difference_l178_178844


namespace math_problem_l178_178869

noncomputable def sqrt180 : ℝ := Real.sqrt 180
noncomputable def two_thirds_sqrt180 : ℝ := (2 / 3) * sqrt180
noncomputable def forty_percent_300_cubed : ℝ := (0.4 * 300)^3
noncomputable def forty_percent_180 : ℝ := 0.4 * 180
noncomputable def one_third_less_forty_percent_180 : ℝ := forty_percent_180 - (1 / 3) * forty_percent_180

theorem math_problem : 
  (two_thirds_sqrt180 * forty_percent_300_cubed) - one_third_less_forty_percent_180 = 15454377.6 :=
  by
    have h1 : sqrt180 = Real.sqrt 180 := rfl
    have h2 : two_thirds_sqrt180 = (2 / 3) * sqrt180 := rfl
    have h3 : forty_percent_300_cubed = (0.4 * 300)^3 := rfl
    have h4 : forty_percent_180 = 0.4 * 180 := rfl
    have h5 : one_third_less_forty_percent_180 = forty_percent_180 - (1 / 3) * forty_percent_180 := rfl
    sorry

end math_problem_l178_178869


namespace students_that_do_not_like_either_sport_l178_178082

def total_students : ℕ := 30
def students_like_basketball : ℕ := 15
def students_like_table_tennis : ℕ := 10
def students_like_both : ℕ := 3

theorem students_that_do_not_like_either_sport : (total_students - (students_like_basketball + students_like_table_tennis - students_like_both)) = 8 := 
by
  sorry

end students_that_do_not_like_either_sport_l178_178082


namespace total_obstacle_course_time_l178_178025

-- Definitions for the given conditions
def first_part_time : Nat := 7 * 60 + 23
def second_part_time : Nat := 73
def third_part_time : Nat := 5 * 60 + 58

-- State the main theorem
theorem total_obstacle_course_time :
  first_part_time + second_part_time + third_part_time = 874 :=
by
  sorry

end total_obstacle_course_time_l178_178025


namespace rational_sign_product_l178_178612

theorem rational_sign_product (a b c : ℚ) (h : |a| / a + |b| / b + |c| / c = 1) : abc / |abc| = -1 := 
by
  -- Proof to be provided
  sorry

end rational_sign_product_l178_178612


namespace number_of_employees_is_five_l178_178757

theorem number_of_employees_is_five
  (rudy_speed : ℕ)
  (joyce_speed : ℕ)
  (gladys_speed : ℕ)
  (lisa_speed : ℕ)
  (mike_speed : ℕ)
  (average_speed : ℕ)
  (h1 : rudy_speed = 64)
  (h2 : joyce_speed = 76)
  (h3 : gladys_speed = 91)
  (h4 : lisa_speed = 80)
  (h5 : mike_speed = 89)
  (h6 : average_speed = 80) :
  (rudy_speed + joyce_speed + gladys_speed + lisa_speed + mike_speed) / average_speed = 5 :=
by
  sorry

end number_of_employees_is_five_l178_178757


namespace binary_to_decimal_110_eq_6_l178_178268

theorem binary_to_decimal_110_eq_6 : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) :=
by
  sorry

end binary_to_decimal_110_eq_6_l178_178268


namespace three_wheels_possible_two_wheels_not_possible_l178_178040

-- Define the conditions as hypotheses
def wheels_spokes (total_spokes_visible : ℕ) (max_spokes_per_wheel : ℕ) (wheels : ℕ) : Prop :=
  total_spokes_visible >= wheels * max_spokes_per_wheel ∧ wheels ≥ 1

-- Prove if a) three wheels is a possible solution
theorem three_wheels_possible : ∃ wheels, wheels = 3 ∧ wheels_spokes 7 3 wheels := by
  sorry

-- Prove if b) two wheels is not a possible solution
theorem two_wheels_not_possible : ¬ ∃ wheels, wheels = 2 ∧ wheels_spokes 7 3 wheels := by
  sorry

end three_wheels_possible_two_wheels_not_possible_l178_178040


namespace book_purchasing_methods_l178_178907

theorem book_purchasing_methods :
  ∃ (A B C D : ℕ),
  A + B + C + D = 10 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
  3 * A + 5 * B + 7 * C + 11 * D = 70 ∧
  (∃ N : ℕ, N = 4) :=
by sorry

end book_purchasing_methods_l178_178907


namespace fixed_real_root_l178_178624

theorem fixed_real_root (k x : ℝ) (h : x^2 + (k + 3) * x + (k + 2) = 0) : x = -1 :=
sorry

end fixed_real_root_l178_178624


namespace restore_original_price_l178_178088

def price_after_increases (p : ℝ) : ℝ :=
  let p1 := p * 1.10
  let p2 := p1 * 1.10
  let p3 := p2 * 1.05
  p3

theorem restore_original_price (p : ℝ) (h : p = 1) : 
  ∃ x : ℝ, x = 22 ∧ (price_after_increases p) * (1 - x / 100) = 1 := 
by 
  sorry

end restore_original_price_l178_178088


namespace right_triangle_area_l178_178308

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end right_triangle_area_l178_178308


namespace previous_salary_is_40_l178_178857

-- Define the conditions
def new_salary : ℕ := 80
def percentage_increase : ℕ := 100

-- Proven goal: John's previous salary before the raise
def previous_salary : ℕ := new_salary / 2

theorem previous_salary_is_40 : previous_salary = 40 := 
by
  -- Proof steps would go here
  sorry

end previous_salary_is_40_l178_178857


namespace max_min_values_l178_178048

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

theorem max_min_values : 
  ∃ (max_val min_val : ℝ), 
    max_val = 7 ∧ min_val = -20 ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max_val) ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min_val ≤ f x) := 
by
  sorry

end max_min_values_l178_178048


namespace miles_driven_on_Monday_l178_178932

def miles_Tuesday : ℕ := 18
def miles_Wednesday : ℕ := 21
def avg_miles_per_day : ℕ := 17

theorem miles_driven_on_Monday (miles_Monday : ℕ) :
  (miles_Monday + miles_Tuesday + miles_Wednesday) / 3 = avg_miles_per_day →
  miles_Monday = 12 :=
by
  intro h
  sorry

end miles_driven_on_Monday_l178_178932


namespace different_lines_through_two_points_in_4_by_4_grid_l178_178633

theorem different_lines_through_two_points_in_4_by_4_grid : 
  let points := fin 4 × fin 4 in
  let number_of_lines := 
    (nat.choose 16 2) - 
    (8 * (4 - 1)) - 
    (2 * (4 - 1)) in
  number_of_lines = 90 :=
by
  sorry

end different_lines_through_two_points_in_4_by_4_grid_l178_178633


namespace point_on_line_l178_178013

theorem point_on_line (A B C x₀ y₀ : ℝ) :
  (A * x₀ + B * y₀ + C = 0) ↔ (A * (x₀ - x₀) + B * (y₀ - y₀) = 0) :=
by 
  sorry

end point_on_line_l178_178013


namespace find_S15_l178_178611

-- Define the arithmetic progression series
variable {S : ℕ → ℕ}

-- Given conditions
axiom S5 : S 5 = 3
axiom S10 : S 10 = 12

-- We need to prove the final statement
theorem find_S15 : S 15 = 39 := 
by
  sorry

end find_S15_l178_178611


namespace eval_expression_at_neg_one_l178_178239

variable (x : ℤ)

theorem eval_expression_at_neg_one : x = -1 → 3 * x ^ 2 + 2 * x - 1 = 0 := by
  intro h
  rw [h]
  show 3 * (-1) ^ 2 + 2 * (-1) - 1 = 0
  sorry

end eval_expression_at_neg_one_l178_178239


namespace lines_in_4_by_4_grid_l178_178662

/--
In a 4-by-4 grid of lattice points, the number of different lines that pass through at least two points is 30.
-/
theorem lines_in_4_by_4_grid : 
  ∃ lines : ℕ, lines = 30 ∧ (∀ pts : fin 4 × fin 4, ∃ l : Set (fin 4 × fin 4), 
  ∀ p1 p2 : fin 4 × fin 4, p1 ∈ pts → p2 ∈ pts → p1 ≠ p2 → p1 ∈ l ∧ p2 ∈ l) := 
sorry

end lines_in_4_by_4_grid_l178_178662


namespace chantel_bracelets_final_count_l178_178433

-- Definitions for conditions
def bracelets_made_days (days : ℕ) (bracelets_per_day : ℕ) : ℕ :=
  days * bracelets_per_day

def initial_bracelets (days1 : ℕ) (bracelets_per_day1 : ℕ) : ℕ :=
  bracelets_made_days days1 bracelets_per_day1

def after_giving_away1 (initial_count : ℕ) (given_away1 : ℕ) : ℕ :=
  initial_count - given_away1

def additional_bracelets (days2 : ℕ) (bracelets_per_day2 : ℕ) : ℕ :=
  bracelets_made_days days2 bracelets_per_day2

def final_count (remaining_after_giving1 : ℕ) (additional_made : ℕ) (given_away2 : ℕ) : ℕ :=
  remaining_after_giving1 + additional_made - given_away2

-- Main theorem statement
theorem chantel_bracelets_final_count :
  ∀ (days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 : ℕ),
  days1 = 5 →
  bracelets_per_day1 = 2 →
  given_away1 = 3 →
  days2 = 4 →
  bracelets_per_day2 = 3 →
  given_away2 = 6 →
  final_count (after_giving_away1 (initial_bracelets days1 bracelets_per_day1) given_away1)
              (additional_bracelets days2 bracelets_per_day2)
              given_away2 = 13 :=
by
  intros days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 hdays1 hbracelets_per_day1 hgiven_away1 hdays2 hbracelets_per_day2 hgiven_away2
  -- Proof is not required, so we use sorry
  sorry

end chantel_bracelets_final_count_l178_178433


namespace number_of_lines_at_least_two_points_4_by_4_grid_l178_178659

-- Definition of 4-by-4 grid
def grid : Type := (Fin 4) × (Fin 4)

-- Definition of a line passing through at least two points in this grid
def line_through_at_least_two_points (points : List grid) : Prop := 
  points.length ≥ 2
  ∧ ∃ m b, ∀ (x y : Fin 4 × Fin 4), (x ∈ points ∧ y ∈ points) → (y.snd : ℕ) = m * (x.fst : ℕ) + b

-- Defining the total number of points choosing 2 out of 16
def total_points : Nat := Nat.choose 16 2

-- Defining the overcount of vertical, horizontal,
-- major diagonals, and secondary diagonals lines
def overcount : Nat := 8 + 2 + 4

-- Total distinct count of lines passing through at least two points
def correct_answer : Nat := total_points - overcount

-- Main theorem stating that the total count is 106
theorem number_of_lines_at_least_two_points_4_by_4_grid : correct_answer = 106 := 
by
  sorry

end number_of_lines_at_least_two_points_4_by_4_grid_l178_178659


namespace area_smallest_region_enclosed_l178_178104

theorem area_smallest_region_enclosed {x y : ℝ} (circle_eq : x^2 + y^2 = 9) (abs_line_eq : y = |x|) :
  ∃ area, area = (9 * Real.pi) / 4 :=
by
  sorry

end area_smallest_region_enclosed_l178_178104


namespace find_constants_l178_178109

theorem find_constants (A B C : ℝ) (hA : A = 7) (hB : B = -9) (hC : C = 5) :
  (∀ (x : ℝ), x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → 
    ( -2 * x ^ 2 + 5 * x - 7) / (x ^ 3 - x) = A / x + (B * x + C) / (x ^ 2 - 1) ) :=
by
  intros x hx
  rw [hA, hB, hC]
  sorry

end find_constants_l178_178109


namespace original_number_of_motorcycles_l178_178904

theorem original_number_of_motorcycles (x y : ℕ) 
  (h1 : x + 2 * y = 42) 
  (h2 : x > y) 
  (h3 : 2 * (x - 3) + 4 * y = 3 * (x + y - 3)) : x = 16 := 
sorry

end original_number_of_motorcycles_l178_178904


namespace solve_abs_eq_l178_178750

theorem solve_abs_eq (x : ℝ) : (|x - 3| = 5 - x) ↔ (x = 4) :=
by
  sorry

end solve_abs_eq_l178_178750


namespace distance_between_cities_l178_178515

theorem distance_between_cities (S : ℕ) 
  (h1 : ∀ x, 0 ≤ x ∧ x ≤ S → x.gcd (S - x) ∈ {1, 3, 13}) : 
  S = 39 :=
sorry

end distance_between_cities_l178_178515


namespace least_k_for_168_l178_178846

theorem least_k_for_168 (k : ℕ) :
  (k^3 % 168 = 0) ↔ k ≥ 42 :=
sorry

end least_k_for_168_l178_178846


namespace cubes_sum_l178_178218

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l178_178218


namespace deck_length_is_30_l178_178168

theorem deck_length_is_30
  (x : ℕ)
  (h1 : ∀ a : ℕ, a = 40 * x)
  (h2 : ∀ b : ℕ, b = 3 * a + 1 * a ∧ b = 4800) :
  x = 30 := by
  sorry

end deck_length_is_30_l178_178168


namespace lines_in_4_by_4_grid_l178_178676

-- Definition for the grid and the number of lattice points.
def grid : Nat := 16

-- Theorem stating that the number of different lines passing through at least two points in a 4-by-4 grid of lattice points.
theorem lines_in_4_by_4_grid : 
  (number_of_lines : Nat) → number_of_lines = 40 ↔ grid = 16 := 
by
  -- Calculating number of lines passing through at least two points in a 4-by-4 grid.
  sorry -- proof skipped

end lines_in_4_by_4_grid_l178_178676


namespace percentage_increase_of_base_l178_178764

theorem percentage_increase_of_base
  (h b : ℝ) -- Original height and base
  (h_new : ℝ) -- New height
  (b_new : ℝ) -- New base
  (A_original A_new : ℝ) -- Original and new areas
  (p : ℝ) -- Percentage increase in the base
  (h_new_def : h_new = 0.60 * h)
  (b_new_def : b_new = b * (1 + p / 100))
  (A_original_def : A_original = 0.5 * b * h)
  (A_new_def : A_new = 0.5 * b_new * h_new)
  (area_decrease : A_new = 0.84 * A_original) :
  p = 40 := by
  sorry

end percentage_increase_of_base_l178_178764


namespace spherical_ball_radius_l178_178792

noncomputable def largest_spherical_ball_radius (inner_radius outer_radius : ℝ) (center : ℝ × ℝ × ℝ) (table_z : ℝ) : ℝ :=
  let r := 4
  r

theorem spherical_ball_radius
  (inner_radius outer_radius : ℝ)
  (center : ℝ × ℝ × ℝ)
  (table_z : ℝ)
  (h1 : inner_radius = 3)
  (h2 : outer_radius = 5)
  (h3 : center = (4,0,1))
  (h4 : table_z = 0) :
  largest_spherical_ball_radius inner_radius outer_radius center table_z = 4 :=
by sorry

end spherical_ball_radius_l178_178792


namespace lines_in_4_by_4_grid_l178_178661

/--
In a 4-by-4 grid of lattice points, the number of different lines that pass through at least two points is 30.
-/
theorem lines_in_4_by_4_grid : 
  ∃ lines : ℕ, lines = 30 ∧ (∀ pts : fin 4 × fin 4, ∃ l : Set (fin 4 × fin 4), 
  ∀ p1 p2 : fin 4 × fin 4, p1 ∈ pts → p2 ∈ pts → p1 ≠ p2 → p1 ∈ l ∧ p2 ∈ l) := 
sorry

end lines_in_4_by_4_grid_l178_178661


namespace smart_charging_piles_eq_l178_178469

theorem smart_charging_piles_eq (x : ℝ) :
  301 * (1 + x) ^ 2 = 500 :=
by sorry

end smart_charging_piles_eq_l178_178469


namespace UBA_Capital_bought_8_SUVs_l178_178570

noncomputable def UBA_Capital_SUVs : ℕ := 
  let T := 9  -- Number of Toyotas
  let H := 1  -- Number of Hondas
  let SUV_Toyota := 9 / 10 * T  -- 90% of Toyotas are SUVs
  let SUV_Honda := 1 / 10 * H   -- 10% of Hondas are SUVs
  SUV_Toyota + SUV_Honda  -- Total number of SUVs

theorem UBA_Capital_bought_8_SUVs : UBA_Capital_SUVs = 8 := by
  sorry

end UBA_Capital_bought_8_SUVs_l178_178570


namespace octahedron_has_constant_perimeter_cross_sections_l178_178810

structure Octahedron :=
(edge_length : ℝ)

def all_cross_sections_same_perimeter (oct : Octahedron) :=
  ∀ (face1 face2 : ℝ), (face1 = face2)

theorem octahedron_has_constant_perimeter_cross_sections (oct : Octahedron) :
  all_cross_sections_same_perimeter oct :=
  sorry

end octahedron_has_constant_perimeter_cross_sections_l178_178810


namespace total_length_of_ropes_l178_178544

theorem total_length_of_ropes (L : ℝ) 
  (h1 : (L - 12 = 4 * (L - 42))) : 
  2 * L = 104 := 
by
  sorry

end total_length_of_ropes_l178_178544


namespace area_of_right_triangle_with_given_sides_l178_178312

theorem area_of_right_triangle_with_given_sides :
  let f : (ℝ → ℝ) := fun x => x^2 - 7 * x + 12
  let a := 3
  let b := 4
  let c := sqrt 7
  let hypotenuse := max a b
  let leg := min a b
in (hypotenuse = 4 ∧ leg = 3 ∧ (f(3) = 0) ∧ (f(4) = 0) → 
   (∃ (area : ℝ), (area = 6 ∨ area = (3 * sqrt 7) / 2))) :=
by
  intros
  sorry

end area_of_right_triangle_with_given_sides_l178_178312


namespace lines_in_4_by_4_grid_l178_178679

-- Definition for the grid and the number of lattice points.
def grid : Nat := 16

-- Theorem stating that the number of different lines passing through at least two points in a 4-by-4 grid of lattice points.
theorem lines_in_4_by_4_grid : 
  (number_of_lines : Nat) → number_of_lines = 40 ↔ grid = 16 := 
by
  -- Calculating number of lines passing through at least two points in a 4-by-4 grid.
  sorry -- proof skipped

end lines_in_4_by_4_grid_l178_178679


namespace max_f_when_a_minus_1_range_of_a_l178_178832

noncomputable section

-- Definitions of the functions given in the problem
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := x * f a x
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2 * a - 1) * x + (a - 1)

-- Statement (1): Proving the maximum value of f(x) when a = -1
theorem max_f_when_a_minus_1 : 
  (∀ x : ℝ, f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement (2): Proving the range of a when g(x) ≤ h(x) for x ≥ 1
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → g a x ≤ h a x) → (1 ≤ a) :=
sorry

end max_f_when_a_minus_1_range_of_a_l178_178832


namespace circle_division_parts_l178_178908

-- Define the number of parts a circle is divided into by the chords.
noncomputable def numberOfParts (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24

-- Prove that the number of parts is given by the defined function.
theorem circle_division_parts (n : ℕ) : numberOfParts n = (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24 := by
  sorry

end circle_division_parts_l178_178908


namespace fraction_evaluation_l178_178918

theorem fraction_evaluation :
  (2 + 3 * 6) / (23 + 6) = 20 / 29 := by
  -- Proof can be filled in here
  sorry

end fraction_evaluation_l178_178918


namespace gcd_a_b_l178_178026

-- Define a and b
def a : ℕ := 333333
def b : ℕ := 9999999

-- Prove that gcd(a, b) = 3
theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l178_178026


namespace range_of_a_l178_178982

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 :=
by {
  sorry
}

end range_of_a_l178_178982


namespace B_current_age_l178_178254

theorem B_current_age (A B : ℕ) (h1 : A = B + 15) (h2 : A - 5 = 2 * (B - 5)) : B = 20 :=
by sorry

end B_current_age_l178_178254


namespace total_sum_is_750_l178_178490

-- Define the individual numbers
def joyce_number : ℕ := 30

def xavier_number (joyce : ℕ) : ℕ :=
  4 * joyce

def coraline_number (xavier : ℕ) : ℕ :=
  xavier + 50

def jayden_number (coraline : ℕ) : ℕ :=
  coraline - 40

def mickey_number (jayden : ℕ) : ℕ :=
  jayden + 20

def yvonne_number (xavier joyce : ℕ) : ℕ :=
  xavier + joyce

-- Prove the total sum is 750
theorem total_sum_is_750 :
  joyce_number + xavier_number joyce_number + coraline_number (xavier_number joyce_number) +
  jayden_number (coraline_number (xavier_number joyce_number)) +
  mickey_number (jayden_number (coraline_number (xavier_number joyce_number))) +
  yvonne_number (xavier_number joyce_number) joyce_number = 750 :=
by {
  -- Proof omitted for brevity
  sorry
}

end total_sum_is_750_l178_178490


namespace find_possible_values_of_A_and_Z_l178_178739

-- Defining the conditions
def contains_A_gold_cells (board : ℕ → ℕ → ℕ) (A: ℕ) : Prop :=
∀ (i j : ℕ), i + 2 < 2016 ∧ j + 2 < 2016 → 
  (∑ 0 ≤ k < 3, ∑ 0 ≤ l < 3, board (i + k) (j + l)) = A

def contains_Z_gold_cells (board : ℕ → ℕ → ℕ) (Z: ℕ) : Prop :=
  (∀ (i j : ℕ), i + 1 < 2016 ∧ j + 3 < 2016 → 
  (∑ 0 ≤ k < 2, ∑ 0 ≤ l < 4, board (i + k) (j + l)) = Z) ∧
  (∀ (i j : ℕ), i + 3 < 2016 ∧ j + 1 < 2016 → 
  (∑ 0 ≤ k < 4, ∑ 0 ≤ l < 2, board (i + k) (j + l)) = Z)

-- The theorem statement
theorem find_possible_values_of_A_and_Z (A Z : ℕ) :
  (∃ (board : ℕ → ℕ → ℕ),
    contains_A_gold_cells board A ∧ contains_Z_gold_cells board Z) ↔ 
    (A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8) := sorry

end find_possible_values_of_A_and_Z_l178_178739


namespace job_planned_completion_days_l178_178390

noncomputable def initial_days_planned (W D : ℝ) := 6 * (W / D) = (W - 3 * (W / D)) / 3

theorem job_planned_completion_days (W : ℝ ) : 
  ∃ D : ℝ, initial_days_planned W D ∧ D = 6 := 
sorry

end job_planned_completion_days_l178_178390


namespace sum_of_cubes_l178_178220

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178220


namespace total_weight_of_4_moles_of_ba_cl2_l178_178555

-- Conditions
def atomic_weight_ba : ℝ := 137.33
def atomic_weight_cl : ℝ := 35.45
def moles_ba_cl2 : ℝ := 4

-- Molecular weight of BaCl2
def molecular_weight_ba_cl2 : ℝ := 
  atomic_weight_ba + 2 * atomic_weight_cl

-- Total weight of 4 moles of BaCl2
def total_weight : ℝ := 
  molecular_weight_ba_cl2 * moles_ba_cl2

-- Theorem stating the total weight of 4 moles of BaCl2
theorem total_weight_of_4_moles_of_ba_cl2 :
  total_weight = 832.92 :=
sorry

end total_weight_of_4_moles_of_ba_cl2_l178_178555


namespace incenter_circumcenter_identity_l178_178546

noncomputable def triangle : Type := sorry
noncomputable def incenter (t : triangle) : Type := sorry
noncomputable def circumcenter (t : triangle) : Type := sorry
noncomputable def inradius (t : triangle) : ℝ := sorry
noncomputable def circumradius (t : triangle) : ℝ := sorry
noncomputable def distance (A B : Type) : ℝ := sorry

theorem incenter_circumcenter_identity (t : triangle) (I O : Type)
  (hI : I = incenter t) (hO : O = circumcenter t)
  (r : ℝ) (h_r : r = inradius t)
  (R : ℝ) (h_R : R = circumradius t) :
  distance I O ^ 2 = R ^ 2 - 2 * R * r :=
sorry

end incenter_circumcenter_identity_l178_178546


namespace cubes_sum_l178_178217

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l178_178217


namespace xy_minimization_l178_178960

theorem xy_minimization (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : (1 / (x : ℝ)) + 1 / (3 * y) = 1 / 11) : x * y = 176 ∧ x + y = 30 :=
by
  sorry

end xy_minimization_l178_178960


namespace count_distinct_lines_l178_178667

-- Define a 4-by-4 grid of lattice points
def grid_points := finset (ℕ × ℕ)

-- The set of all points in a 4-by-4 grid
def four_by_four_grid : grid_points :=
  {(0, 0), (0, 1), (0, 2), (0, 3),
   (1, 0), (1, 1), (1, 2), (1, 3),
   (2, 0), (2, 1), (2, 2), (2, 3),
   (3, 0), (3, 1), (3, 2), (3, 3)}.to_finset

-- A line passing through at least two points
def line (p1 p2 : ℕ × ℕ) : set (ℕ × ℕ) :=
  {p : ℕ × ℕ | ∃ λ : ℚ, ∃ b : ℚ, (p.2 : ℚ) = λ * (p.1 : ℚ) + b}

noncomputable theory

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 50. -/
theorem count_distinct_lines (grid : grid_points) (h : grid = four_by_four_grid) :
  ∃ n, n = 50 :=
by
  sorry

end count_distinct_lines_l178_178667


namespace inverse_value_of_f_l178_178957

theorem inverse_value_of_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2^x - 2) : f⁻¹ 2 = 3 :=
sorry

end inverse_value_of_f_l178_178957


namespace total_lemonade_poured_l178_178425

def lemonade_poured (first: ℝ) (second: ℝ) (third: ℝ) := first + second + third

theorem total_lemonade_poured :
  lemonade_poured 0.25 0.4166666666666667 0.25 = 0.917 :=
by
  sorry

end total_lemonade_poured_l178_178425


namespace sqrt_of_16_l178_178503

theorem sqrt_of_16 : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_of_16_l178_178503


namespace alcohol_water_ratio_mixtures_l178_178703

theorem alcohol_water_ratio_mixtures :
  let a_alcohol_ratio := (2 : ℚ) / 3,
      a_water_ratio := (1 : ℚ) / 3,
      b_alcohol_ratio := (4 : ℚ) / 7,
      b_water_ratio := (3 : ℚ) / 7,
      volume_a := 5,
      volume_b := 14,
      total_alcohol := a_alcohol_ratio * volume_a + b_alcohol_ratio * volume_b,
      total_water := a_water_ratio * volume_a + b_water_ratio * volume_b
  in total_alcohol / total_water = 34 / 23 :=
by
  sorry

end alcohol_water_ratio_mixtures_l178_178703


namespace x_in_interval_l178_178085

theorem x_in_interval (x : ℝ) (h : x = (1 / x) * (-x) + 2) : 0 < x ∧ x ≤ 2 :=
by
  -- Place the proof here
  sorry

end x_in_interval_l178_178085


namespace sum_of_cubes_l178_178219

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178219


namespace manager_salary_l178_178894

theorem manager_salary (n : ℕ) (avg_salary : ℕ) (increment : ℕ) (new_avg_salary : ℕ) (new_total_salary : ℕ) (old_total_salary : ℕ) :
  n = 20 →
  avg_salary = 1500 →
  increment = 1000 →
  new_avg_salary = avg_salary + increment →
  old_total_salary = n * avg_salary →
  new_total_salary = (n + 1) * new_avg_salary →
  (new_total_salary - old_total_salary) = 22500 :=
by
  intros h_n h_avg_salary h_increment h_new_avg_salary h_old_total_salary h_new_total_salary
  sorry

end manager_salary_l178_178894


namespace number_of_boats_l178_178905

theorem number_of_boats (total_people : ℕ) (people_per_boat : ℕ)
  (h1 : total_people = 15) (h2 : people_per_boat = 3) : total_people / people_per_boat = 5 :=
by {
  -- proof steps here
  sorry
}

end number_of_boats_l178_178905


namespace smallest_number_of_multiplications_is_two_l178_178711

theorem smallest_number_of_multiplications_is_two : ∃ (ops : List (ℕ → ℕ → ℕ)),
  (∀ op ∈ ops, op = (+) ∨ op = (-) ∨ op = (*)) ∧ ops.length = 61 ∧
  (Σ' (n < 62), (ops.nth n).get_or_else (+) (n + 1) (n + 2)) = 2023 ∧
  ops.count ((*) (·)) = 2 := 
sorry

end smallest_number_of_multiplications_is_two_l178_178711


namespace integer_ratio_condition_l178_178165

variable (x y : ℝ)

theorem integer_ratio_condition 
  (h : 3 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 6)
  (h_int : ∃ t : ℤ, x = t * y) :
  ∃ t : ℤ, t = -2 :=
by
  sorry

end integer_ratio_condition_l178_178165


namespace sum_of_cubes_l178_178228

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l178_178228


namespace minimum_g_value_l178_178946

noncomputable def g (x : ℝ) := (9 * x^2 + 18 * x + 20) / (4 * (2 + x))

theorem minimum_g_value :
  ∀ x ≥ (1 : ℝ), g x = (47 / 16) := sorry

end minimum_g_value_l178_178946


namespace max_k_value_l178_178127

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k_value
    (h : ∀ (x : ℝ), 1 < x → f x > k * (x - 1)) :
    k = 3 := sorry

end max_k_value_l178_178127


namespace cost_of_later_purchase_l178_178191

-- Define the costs of bats and balls as constants.
def cost_of_bat : ℕ := 500
def cost_of_ball : ℕ := 100

-- Define the quantities involved in the later purchase.
def bats_purchased_later : ℕ := 3
def balls_purchased_later : ℕ := 5

-- Define the expected total cost for the later purchase.
def expected_total_cost_later : ℕ := 2000

-- The theorem to be proved: the cost of the later purchase of bats and balls is $2000.
theorem cost_of_later_purchase :
  bats_purchased_later * cost_of_bat + balls_purchased_later * cost_of_ball = expected_total_cost_later :=
sorry

end cost_of_later_purchase_l178_178191


namespace solve_abs_eq_l178_178748

theorem solve_abs_eq (x : ℝ) : (|x - 3| = 5 - x) ↔ (x = 4) := 
by
  sorry

end solve_abs_eq_l178_178748


namespace means_imply_sum_of_squares_l178_178352

noncomputable def arithmetic_mean (x y z : ℝ) : ℝ :=
(x + y + z) / 3

noncomputable def geometric_mean (x y z : ℝ) : ℝ :=
(x * y * z) ^ (1/3)

noncomputable def harmonic_mean (x y z : ℝ) : ℝ :=
3 / ((1/x) + (1/y) + (1/z))

theorem means_imply_sum_of_squares (x y z : ℝ) :
  arithmetic_mean x y z = 10 →
  geometric_mean x y z = 6 →
  harmonic_mean x y z = 4 →
  x^2 + y^2 + z^2 = 576 :=
by
  -- Proof is omitted for now
  exact sorry

end means_imply_sum_of_squares_l178_178352


namespace find_radii_l178_178526

theorem find_radii (r R : ℝ) (h₁ : R - r = 2) (h₂ : R + r = 16) : r = 7 ∧ R = 9 := by
  sorry

end find_radii_l178_178526


namespace max_quarters_in_wallet_l178_178190

theorem max_quarters_in_wallet:
  ∃ (q n : ℕ), 
    (30 * n) + 50 = 31 * (n + 1) ∧ 
    q = 22 :=
by
  sorry

end max_quarters_in_wallet_l178_178190


namespace sum_of_cubes_l178_178238

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l178_178238


namespace solve_abs_eq_l178_178751

theorem solve_abs_eq (x : ℝ) : (|x - 3| = 5 - x) ↔ (x = 4) :=
by
  sorry

end solve_abs_eq_l178_178751


namespace polygon_area_l178_178714

-- Definitions and conditions
def side_length (n : ℕ) (p : ℕ) := p / n
def rectangle_area (s : ℕ) := 2 * s * s
def total_area (r : ℕ) (area : ℕ) := r * area

-- Theorem statement with conditions and conclusion
theorem polygon_area (n r p : ℕ) (h1 : n = 24) (h2 : r = 4) (h3 : p = 48) :
  total_area r (rectangle_area (side_length n p)) = 32 := by
  sorry

end polygon_area_l178_178714


namespace c_investment_l178_178261

theorem c_investment 
  (A_investment B_investment : ℝ)
  (C_share total_profit : ℝ)
  (hA : A_investment = 8000)
  (hB : B_investment = 4000)
  (hC_share : C_share = 36000)
  (h_profit : total_profit = 252000) :
  ∃ (x : ℝ), (x / 4000) / (2 + 1 + x / 4000) = (36000 / 252000) ∧ x = 2000 :=
by
  sorry

end c_investment_l178_178261


namespace find_b_l178_178488

theorem find_b (p : ℕ) (hp : Nat.Prime p) :
  (∃ b : ℕ, b = (p + 1) ^ 2 ∨ b = 4 * p ∧ ∀ (x1 x2 : ℤ), x1 * x2 = p * b ∧ x1 + x2 = b) → 
  (∃ b : ℕ, b = (p + 1) ^ 2 ∨ b = 4 * p) :=
by
  sorry

end find_b_l178_178488


namespace draw_balls_equiv_l178_178387

noncomputable def number_of_ways_to_draw_balls (n : ℕ) (k : ℕ) (ball1 : ℕ) (ball2 : ℕ) : ℕ :=
  if n = 15 ∧ k = 4 ∧ ball1 = 1 ∧ ball2 = 15 then
    4 * (Nat.choose 14 3 * Nat.factorial 3) * 2
  else
    0

theorem draw_balls_equiv : number_of_ways_to_draw_balls 15 4 1 15 = 17472 :=
by
  dsimp [number_of_ways_to_draw_balls]
  rw [Nat.choose, Nat.factorial]
  norm_num
  sorry

end draw_balls_equiv_l178_178387


namespace rowing_speed_upstream_l178_178400

theorem rowing_speed_upstream (V_m V_down : ℝ) (h_Vm : V_m = 35) (h_Vdown : V_down = 40) : V_m - (V_down - V_m) = 30 :=
by
  sorry

end rowing_speed_upstream_l178_178400


namespace probability_sunflower_seed_l178_178937

theorem probability_sunflower_seed :
  ∀ (sunflower_seeds green_bean_seeds pumpkin_seeds : ℕ),
  sunflower_seeds = 2 →
  green_bean_seeds = 3 →
  pumpkin_seeds = 4 →
  (sunflower_seeds + green_bean_seeds + pumpkin_seeds = 9) →
  (sunflower_seeds : ℚ) / (sunflower_seeds + green_bean_seeds + pumpkin_seeds) = 2 / 9 := 
by 
  intros sunflower_seeds green_bean_seeds pumpkin_seeds h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h1, h2, h3]
  sorry -- Proof omitted as per instructions.

end probability_sunflower_seed_l178_178937


namespace complex_prod_eq_l178_178733

theorem complex_prod_eq (x y z : ℂ) (h1 : x * y + 6 * y = -24) (h2 : y * z + 6 * z = -24) (h3 : z * x + 6 * x = -24) :
  x * y * z = 144 :=
by
  sorry

end complex_prod_eq_l178_178733


namespace least_cost_flower_bed_divisdes_l178_178885

theorem least_cost_flower_bed_divisdes:
  let Region1 := 5 * 2
  let Region2 := 3 * 5
  let Region3 := 2 * 4
  let Region4 := 5 * 4
  let Region5 := 5 * 3
  let Cost_Dahlias := 2.70
  let Cost_Cannas := 2.20
  let Cost_Begonias := 1.70
  let Cost_Freesias := 3.20
  let total_cost := 
    Region1 * Cost_Dahlias + 
    Region2 * Cost_Cannas + 
    Region3 * Cost_Freesias + 
    Region4 * Cost_Begonias + 
    Region5 * Cost_Cannas
  total_cost = 152.60 :=
by
  sorry

end least_cost_flower_bed_divisdes_l178_178885


namespace count_lines_in_4x4_grid_l178_178693

theorem count_lines_in_4x4_grid : 
  let grid_points : Fin 4 × Fin 4 := 
  ∃! lines : set (set (Fin 4 × Fin 4)), 
  ∀ line ∈ lines, ∃ (p1 p2 : Fin 4 × Fin 4), p1 ≠ p2 ∧ p1 ∈ line ∧ p2 ∈ line ∧ (grid_points ⊆ line ⊆ grid_points) :=
  lines = 84 :=
sorry

end count_lines_in_4x4_grid_l178_178693


namespace oil_flow_relationship_l178_178705

theorem oil_flow_relationship (t : ℝ) (Q : ℝ) (initial_quantity : ℝ) (flow_rate : ℝ)
  (h_initial : initial_quantity = 20) (h_flow : flow_rate = 0.2) :
  Q = initial_quantity - flow_rate * t :=
by
  -- proof to be filled in
  sorry

end oil_flow_relationship_l178_178705


namespace contrapositive_mul_non_zero_l178_178192

variables (a b : ℝ)

theorem contrapositive_mul_non_zero (h : a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :
  (a = 0 ∨ b = 0) → a * b = 0 :=
by
  sorry

end contrapositive_mul_non_zero_l178_178192


namespace no_hamiltonian_path_l178_178318

-- Definitions derived from conditions:
def airports := (fin 2014)
def divides_country_equally (a b : airports) : Prop :=
  -- Placeholder for the actual condition that the line passing through a and b
  -- divides the country into two parts with exactly 1006 airports each.
  sorry

def G : simple_graph airports := {
  adj := λ a b, divides_country_equally a b,
  sym := sorry,  -- symmetry: if a is connected to b, then b is connected to a
  loopless := sorry  -- no loops: no vertex is connected to itself
}

-- The main theorem we need to prove:
theorem no_hamiltonian_path (G : simple_graph airports)
  (hG : ∀ a b : airports, G.adj a b ↔ divides_country_equally a b) :
  ¬(∃ p : list airports, p.nodup ∧ p.length = 2014 ∧ ∀ v ∈ p, v.degree G = 1 ∨ p = [p.head, ..., v, ..., p.last]) :=
sorry

end no_hamiltonian_path_l178_178318


namespace ants_harvest_remaining_sugar_l178_178407

-- Define the initial conditions
def ants_removal_rate : ℕ := 4
def initial_sugar_amount : ℕ := 24
def hours_passed : ℕ := 3

-- Calculate the correct answer
def remaining_sugar (initial : ℕ) (rate : ℕ) (hours : ℕ) : ℕ :=
  initial - (rate * hours)

def additional_hours_needed (remaining_sugar : ℕ) (rate : ℕ) : ℕ :=
  remaining_sugar / rate

-- The specification of the proof problem
theorem ants_harvest_remaining_sugar :
  additional_hours_needed (remaining_sugar initial_sugar_amount ants_removal_rate hours_passed) ants_removal_rate = 3 :=
by
  -- Proof omitted
  sorry

end ants_harvest_remaining_sugar_l178_178407


namespace cubes_sum_l178_178216

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l178_178216


namespace add_fraction_l178_178501

theorem add_fraction (x : ℚ) (h : x - 7/3 = 3/2) : x + 7/3 = 37/6 :=
by
  sorry

end add_fraction_l178_178501


namespace inequality_solution_set_l178_178830

theorem inequality_solution_set 
  (c : ℝ) (a : ℝ) (b : ℝ) (h : c > 0) (hb : b = (5 / 2) * c) (ha : a = - (3 / 2) * c) :
  ∀ x : ℝ, (a * x^2 + b * x + c ≥ 0) ↔ (- (1 / 3) ≤ x ∧ x ≤ 2) :=
sorry

end inequality_solution_set_l178_178830


namespace limit_integral_cos_div_x_zero_l178_178815

theorem limit_integral_cos_div_x_zero :
  tendsto (λ x : ℝ, (∫ t in 0..x^2, cos t) / x) (nhds 0) (nhds 0) :=
sorry

end limit_integral_cos_div_x_zero_l178_178815


namespace solve_system_l178_178500

theorem solve_system :
  ∃ x y : ℝ, (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧ 
              (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0) ∧ 
              (x = -3 ∧ y = -1) :=
  sorry

end solve_system_l178_178500


namespace sum_of_cubes_l178_178221

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178221


namespace chantel_bracelets_final_count_l178_178434

-- Definitions for conditions
def bracelets_made_days (days : ℕ) (bracelets_per_day : ℕ) : ℕ :=
  days * bracelets_per_day

def initial_bracelets (days1 : ℕ) (bracelets_per_day1 : ℕ) : ℕ :=
  bracelets_made_days days1 bracelets_per_day1

def after_giving_away1 (initial_count : ℕ) (given_away1 : ℕ) : ℕ :=
  initial_count - given_away1

def additional_bracelets (days2 : ℕ) (bracelets_per_day2 : ℕ) : ℕ :=
  bracelets_made_days days2 bracelets_per_day2

def final_count (remaining_after_giving1 : ℕ) (additional_made : ℕ) (given_away2 : ℕ) : ℕ :=
  remaining_after_giving1 + additional_made - given_away2

-- Main theorem statement
theorem chantel_bracelets_final_count :
  ∀ (days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 : ℕ),
  days1 = 5 →
  bracelets_per_day1 = 2 →
  given_away1 = 3 →
  days2 = 4 →
  bracelets_per_day2 = 3 →
  given_away2 = 6 →
  final_count (after_giving_away1 (initial_bracelets days1 bracelets_per_day1) given_away1)
              (additional_bracelets days2 bracelets_per_day2)
              given_away2 = 13 :=
by
  intros days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 hdays1 hbracelets_per_day1 hgiven_away1 hdays2 hbracelets_per_day2 hgiven_away2
  -- Proof is not required, so we use sorry
  sorry

end chantel_bracelets_final_count_l178_178434


namespace notebook_and_pencil_cost_l178_178211

theorem notebook_and_pencil_cost :
  ∃ (x y : ℝ), 6 * x + 4 * y = 9.2 ∧ 3 * x + y = 3.8 ∧ x + y = 1.8 :=
by
  sorry

end notebook_and_pencil_cost_l178_178211


namespace shaded_area_eq_l178_178472

noncomputable def diameter_AB : ℝ := 6
noncomputable def diameter_BC : ℝ := 6
noncomputable def diameter_CD : ℝ := 6
noncomputable def diameter_DE : ℝ := 6
noncomputable def diameter_EF : ℝ := 6
noncomputable def diameter_FG : ℝ := 6
noncomputable def diameter_AG : ℝ := 6 * 6 -- 36

noncomputable def area_small_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

noncomputable def area_large_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

theorem shaded_area_eq :
  area_large_semicircle diameter_AG + area_small_semicircle diameter_AB = 166.5 * Real.pi :=
  sorry

end shaded_area_eq_l178_178472


namespace number_of_lines_at_least_two_points_4_by_4_grid_l178_178656

-- Definition of 4-by-4 grid
def grid : Type := (Fin 4) × (Fin 4)

-- Definition of a line passing through at least two points in this grid
def line_through_at_least_two_points (points : List grid) : Prop := 
  points.length ≥ 2
  ∧ ∃ m b, ∀ (x y : Fin 4 × Fin 4), (x ∈ points ∧ y ∈ points) → (y.snd : ℕ) = m * (x.fst : ℕ) + b

-- Defining the total number of points choosing 2 out of 16
def total_points : Nat := Nat.choose 16 2

-- Defining the overcount of vertical, horizontal,
-- major diagonals, and secondary diagonals lines
def overcount : Nat := 8 + 2 + 4

-- Total distinct count of lines passing through at least two points
def correct_answer : Nat := total_points - overcount

-- Main theorem stating that the total count is 106
theorem number_of_lines_at_least_two_points_4_by_4_grid : correct_answer = 106 := 
by
  sorry

end number_of_lines_at_least_two_points_4_by_4_grid_l178_178656


namespace count_three_digit_multiples_of_24_l178_178143

-- We define that the range of three-digit integers is from 100 to 999
def range_start := 100
def range_end := 999

-- We define that the number we are looking for must be a multiple of both 6 and 8, which has an LCM of 24
def lcm_6_8 := Nat.lcm 6 8

-- We state the problem of counting the number of multiples of 24 within the range 100 to 999
theorem count_three_digit_multiples_of_24 : 
  let multiples := finset.Ico range_start range_end
  multiples.filter (λ n, n % lcm_6_8 = 0).card = 37 :=
by
  sorry

end count_three_digit_multiples_of_24_l178_178143


namespace number_of_lines_at_least_two_points_4_by_4_grid_l178_178657

-- Definition of 4-by-4 grid
def grid : Type := (Fin 4) × (Fin 4)

-- Definition of a line passing through at least two points in this grid
def line_through_at_least_two_points (points : List grid) : Prop := 
  points.length ≥ 2
  ∧ ∃ m b, ∀ (x y : Fin 4 × Fin 4), (x ∈ points ∧ y ∈ points) → (y.snd : ℕ) = m * (x.fst : ℕ) + b

-- Defining the total number of points choosing 2 out of 16
def total_points : Nat := Nat.choose 16 2

-- Defining the overcount of vertical, horizontal,
-- major diagonals, and secondary diagonals lines
def overcount : Nat := 8 + 2 + 4

-- Total distinct count of lines passing through at least two points
def correct_answer : Nat := total_points - overcount

-- Main theorem stating that the total count is 106
theorem number_of_lines_at_least_two_points_4_by_4_grid : correct_answer = 106 := 
by
  sorry

end number_of_lines_at_least_two_points_4_by_4_grid_l178_178657


namespace wrench_force_l178_178523

def force_inversely_proportional (f1 f2 : ℝ) (L1 L2 : ℝ) : Prop :=
  f1 * L1 = f2 * L2

theorem wrench_force
  (f1 : ℝ) (L1 : ℝ) (f2 : ℝ) (L2 : ℝ)
  (h1 : L1 = 12) (h2 : f1 = 450) (h3 : L2 = 18) (h_prop : force_inversely_proportional f1 f2 L1 L2) :
  f2 = 300 :=
by
  sorry

end wrench_force_l178_178523


namespace single_equivalent_discount_l178_178545

theorem single_equivalent_discount :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  ∃ (k : ℝ), (1 - k) = (1 - discount1) * (1 - discount2) * (1 - discount3) ∧ k = 0.27325 :=
by
  sorry

end single_equivalent_discount_l178_178545


namespace estimated_white_balls_l178_178464

noncomputable def estimate_white_balls (total_balls draws white_draws : ℕ) : ℕ :=
  total_balls * white_draws / draws

theorem estimated_white_balls (total_balls draws white_draws : ℕ) (h1 : total_balls = 20)
  (h2 : draws = 100) (h3 : white_draws = 40) :
  estimate_white_balls total_balls draws white_draws = 8 := by
  sorry

end estimated_white_balls_l178_178464


namespace distance_between_cities_l178_178517

-- Conditions Initialization
variable (A B : ℝ)
variable (S : ℕ)
variable (x : ℕ)
variable (gcd_values : ℕ)

-- Conditions:
-- 1. The distance between cities A and B is an integer.
-- 2. Markers are placed every kilometer.
-- 3. At each marker, one side has distance to city A, and the other has distance to city B.
-- 4. The GCDs observed were only 1, 3, and 13.

-- Given these conditions, we prove the distance is exactly 39 kilometers.
theorem distance_between_cities (h1 : gcd_values = 1 ∨ gcd_values = 3 ∨ gcd_values = 13)
    (h2 : S = Nat.lcm 1 (Nat.lcm 3 13)) :
    S = 39 := by 
  sorry

end distance_between_cities_l178_178517


namespace sum_of_A_B_in_B_l178_178862

def A : Set ℤ := { x | ∃ k : ℤ, x = 2 * k }
def B : Set ℤ := { x | ∃ k : ℤ, x = 2 * k + 1 }
def C : Set ℤ := { x | ∃ k : ℤ, x = 4 * k + 1 }

theorem sum_of_A_B_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end sum_of_A_B_in_B_l178_178862


namespace line_equation_perpendicular_l178_178896

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem line_equation_perpendicular (c : ℝ) :
  (∃ k : ℝ, x - 2 * y + k = 0) ∧ is_perpendicular 2 1 1 (-2) → x - 2 * y - 3 = 0 := by
  sorry

end line_equation_perpendicular_l178_178896


namespace find_real_number_x_l178_178840

theorem find_real_number_x (x : ℝ) (i : ℂ) (hx : i = complex.I) (h : (1 - 2 * (complex.I)) * (x + complex.I) = 4 - 3 * (complex.I)) : x = 2 :=
by sorry

end find_real_number_x_l178_178840


namespace joan_seashells_total_l178_178717

-- Definitions
def original_seashells : ℕ := 70
def additional_seashells : ℕ := 27
def total_seashells : ℕ := original_seashells + additional_seashells

-- Proof Statement
theorem joan_seashells_total : total_seashells = 97 := by
  sorry

end joan_seashells_total_l178_178717


namespace line_intersects_circle_l178_178948

theorem line_intersects_circle 
  (k : ℝ)
  (x y : ℝ)
  (h_line : x = 0 ∨ y = -2)
  (h_circle : (x - 1)^2 + (y + 2)^2 = 16) :
  (-2 - -2)^2 < 16 := by
  sorry

end line_intersects_circle_l178_178948


namespace angel_vowels_written_l178_178098

theorem angel_vowels_written (num_vowels : ℕ) (times_written : ℕ) (h1 : num_vowels = 5) (h2 : times_written = 4) : num_vowels * times_written = 20 := by
  sorry

end angel_vowels_written_l178_178098


namespace right_triangle_area_l178_178313

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end right_triangle_area_l178_178313


namespace factors_and_multiple_of_20_l178_178713

-- Define the relevant numbers
def a := 20
def b := 5
def c := 4

-- Given condition: the equation 20 / 5 = 4
def condition : Prop := a / b = c

-- Factors and multiples relationships to prove
def are_factors : Prop := a % b = 0 ∧ a % c = 0
def is_multiple : Prop := b * c = a

-- The main statement combining everything
theorem factors_and_multiple_of_20 (h : condition) : are_factors ∧ is_multiple :=
sorry

end factors_and_multiple_of_20_l178_178713


namespace next_meeting_time_at_B_l178_178246

-- Definitions of conditions
def perimeter := 800 -- Perimeter of the block in meters
def t1 := 1 -- They meet for the first time after 1 minute
def AB := 100 -- Length of side AB in meters
def BC := 300 -- Length of side BC in meters
def CD := 100 -- Length of side CD in meters
def DA := 300 -- Length of side DA in meters

-- Main theorem statement
theorem next_meeting_time_at_B :
  ∃ t : ℕ, t = 9 ∧ (∃ m1 m2 : ℕ, ((t = m1 * m2 + 1) ∧ m2 = 800 / (t1 * (AB + BC + CD + DA))) ∧ m1 = 9) :=
sorry

end next_meeting_time_at_B_l178_178246


namespace probability_two_different_color_chips_l178_178061

theorem probability_two_different_color_chips : 
  let blue_chips := 4
  let yellow_chips := 5
  let green_chips := 3
  let total_chips := blue_chips + yellow_chips + green_chips
  let prob_diff_color := 
    ((blue_chips / total_chips) * ((yellow_chips + green_chips) / (total_chips - 1))) + 
    ((yellow_chips / total_chips) * ((blue_chips + green_chips) / (total_chips - 1))) + 
    ((green_chips / total_chips) * ((blue_chips + yellow_chips) / (total_chips - 1)))
  in
  prob_diff_color = 47 / 66 :=
by
  sorry

end probability_two_different_color_chips_l178_178061


namespace injective_g_restricted_to_interval_l178_178574

def g (x : ℝ) : ℝ := (x + 3) ^ 2 - 10

theorem injective_g_restricted_to_interval :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (-3) → x2 ∈ Set.Ici (-3) → g x1 = g x2 → x1 = x2) :=
sorry

end injective_g_restricted_to_interval_l178_178574


namespace complex_div_conjugate_l178_178276

theorem complex_div_conjugate (a b : ℂ) (h1 : a = 2 - I) (h2 : b = 1 + 2 * I) :
    a / b = -I := by
  sorry

end complex_div_conjugate_l178_178276


namespace area_of_equilateral_triangle_l178_178065

theorem area_of_equilateral_triangle
  (A B C D E : Type) 
  (side_length : ℝ) 
  (medians_perpendicular : Prop) 
  (BD CE : ℝ)
  (inscribed_circle : Prop)
  (equilateral_triangle : A = B ∧ B = C) 
  (s : side_length = 18) 
  (BD_len : BD = 15) 
  (CE_len : CE = 9) 
  : ∃ area, area = 81 * Real.sqrt 3
  :=
by {
  sorry
}

end area_of_equilateral_triangle_l178_178065


namespace solve_r_l178_178421

-- Define E(a, b, c) as given
def E (a b c : ℕ) : ℕ := a * b^c

-- Lean 4 statement for the proof
theorem solve_r (r : ℕ) (r_pos : 0 < r) : E r r 3 = 625 → r = 5 :=
by
  intro h
  sorry

end solve_r_l178_178421


namespace selection_schemes_l178_178821

theorem selection_schemes (people : Finset ℕ) (A B C : ℕ) (h_people : people.card = 5) 
(h_A_B_individuals : A ∈ people ∧ B ∈ people) (h_A_B_C_exclusion : A ≠ C ∧ B ≠ C) :
  ∃ (number_of_schemes : ℕ), number_of_schemes = 36 :=
by
  sorry

end selection_schemes_l178_178821


namespace Rayden_more_birds_l178_178345

theorem Rayden_more_birds (dLily gLily : ℕ) (h1 : dLily = 20) (h2 : gLily = 10) (h3 : ∀ x, x = 3 * dLily → ∀ y, y = 4 * gLily → x - dLily + y - gLily = 70) :
  let dRayden := 3 * dLily,
      gRayden := 4 * gLily in
  dRayden - dLily + gRayden - gLily = 70 :=
begin
  intro dRayden,
  intro gRayden,
  cases h1,
  cases h2,
  cases h3,
  sorry
end

end Rayden_more_birds_l178_178345


namespace household_member_count_l178_178767

variable (M : ℕ) -- the number of members in the household

-- Conditions
def slices_per_breakfast := 3
def slices_per_snack := 2
def slices_per_member_daily := slices_per_breakfast + slices_per_snack
def slices_per_loaf := 12
def loaves_last_days := 3
def loaves_given := 5
def total_slices := slices_per_loaf * loaves_given
def daily_consumption := total_slices / loaves_last_days

-- Proof statement
theorem household_member_count : daily_consumption = slices_per_member_daily * M → M = 4 :=
by
  sorry

end household_member_count_l178_178767


namespace abs_expression_value_l178_178728

theorem abs_expression_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs (abs x - 2 * x) - abs x) - x) = 6069 :=
by sorry

end abs_expression_value_l178_178728


namespace tangent_line_tangent_value_at_one_l178_178355
noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

theorem tangent_line_tangent_value_at_one
  (f : ℝ → ℝ)
  (hf1 : f 1 = 3 - 1 / 2)
  (hf'1 : deriv f 1 = 1 / 2)
  (tangent_eq : ∀ x, f 1 + deriv f 1 * (x - 1) = 1 / 2 * x + 2) :
  f 1 + deriv f 1 = 3 :=
by sorry

end tangent_line_tangent_value_at_one_l178_178355


namespace inverse_sum_l178_178725

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum : g⁻¹ 8 + g⁻¹ (-64) = -2 :=
by
  -- proof steps will go here
  sorry

end inverse_sum_l178_178725


namespace rectangles_in_grid_l178_178403

-- Define a function that calculates the number of rectangles formed
def number_of_rectangles (n m : ℕ) : ℕ :=
  ((m + 2) * (m + 1) * (n + 2) * (n + 1)) / 4

-- Prove that the number_of_rectangles function correctly calculates the number of rectangles given n and m 
theorem rectangles_in_grid (n m : ℕ) :
  number_of_rectangles n m = ((m + 2) * (m + 1) * (n + 2) * (n + 1)) / 4 := 
by
  sorry

end rectangles_in_grid_l178_178403


namespace right_triangle_area_l178_178307

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end right_triangle_area_l178_178307


namespace algebraic_expression_evaluation_l178_178285

theorem algebraic_expression_evaluation (m : ℝ) (h : m^2 - m - 3 = 0) : m^2 - m - 2 = 1 := 
by
  sorry

end algebraic_expression_evaluation_l178_178285


namespace boys_number_is_60_l178_178385

-- Definitions based on the conditions
variables (x y : ℕ)

def sum_boys_girls (x y : ℕ) : Prop := 
  x + y = 150

def girls_percentage (x y : ℕ) : Prop := 
  y = (x * 150) / 100

-- Prove that the number of boys equals 60
theorem boys_number_is_60 (x y : ℕ) 
  (h1 : sum_boys_girls x y) 
  (h2 : girls_percentage x y) : 
  x = 60 := by
  sorry

end boys_number_is_60_l178_178385


namespace largest_constant_inequality_equality_condition_l178_178423

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆) ^ 2 ≥
    3 * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂)) :=
sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₄ = x₂ + x₅) ∧ (x₂ + x₅ = x₃ + x₆) :=
sorry

end largest_constant_inequality_equality_condition_l178_178423


namespace solve_x_perpendicular_l178_178297

def vec_a : ℝ × ℝ := (1, 3)
def vec_b (x : ℝ) : ℝ × ℝ := (3, x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem solve_x_perpendicular (x : ℝ) (h : perpendicular vec_a (vec_b x)) : x = -1 :=
by {
  sorry
}

end solve_x_perpendicular_l178_178297


namespace geometric_series_sum_l178_178536

theorem geometric_series_sum : 
    ∑' n : ℕ, (1 : ℝ) * (-1 / 2) ^ n = 2 / 3 :=
by
    sorry

end geometric_series_sum_l178_178536


namespace original_speed_of_Person_A_l178_178770

variable (v_A v_B : ℝ)

-- Define the conditions
def condition1 : Prop := v_B = 2 * v_A
def condition2 : Prop := v_A + 10 = 4 * (v_B - 5)

-- Define the theorem to prove
theorem original_speed_of_Person_A (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_B) : v_A = 18 := 
by
  sorry

end original_speed_of_Person_A_l178_178770


namespace find_pairs_l178_178732

theorem find_pairs (a b : ℕ) (q r : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : 0 ≤ r) (h5 : r < a + b)
  (h6 : q^2 + r = 1977) :
  (a, b) = (50, 37) ∨ (a, b) = (50, 7) ∨ (a, b) = (37, 50) ∨ (a, b) = (7, 50) :=
  sorry

end find_pairs_l178_178732


namespace distinct_lines_count_in_4x4_grid_l178_178688

theorem distinct_lines_count_in_4x4_grid :
  let grid_points := (finRange 4).product (finRange 4)
  let lines := {line : Set (ℕ × ℕ) | ∃ (a b : ℤ), ∀ p ∈ line, a * (p.1:ℤ) + b * (p.2:ℤ) = 1}
  let distinct_lines := {line ∈ lines | ∃ (p1 p2 : ℕ × ℕ), p1 ∈ grid_points ∧ p2 ∈ grid_points ∧ p1 ≠ p2 ∧ line = {p | this line passes through p}}
  lines.card = 50 :=
by
  sorry

end distinct_lines_count_in_4x4_grid_l178_178688


namespace num_lines_passing_through_4x4_grid_l178_178648

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 66. -/
theorem num_lines_passing_through_4x4_grid : 
  let p := 4 * 4 in
  let total_point_pairs := p * (p - 1) / 2 in
  let horizontal_lines_count := 4 in
  let vertical_lines_count := 4 in
  let diagonal_lines_4_count := 2 in
  let diagonal_lines_3_count := 2 in
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count in
  (total_point_pairs - overcount_correction) = 66 :=
by
  let p := 4 * 4
  let total_point_pairs := p * (p - 1) / 2
  let horizontal_lines_count := 4
  let vertical_lines_count := 4
  let diagonal_lines_4_count := 2
  let diagonal_lines_3_count := 2
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count
  have h_correct_count : total_point_pairs - overcount_correction = 66, from sorry
  exact h_correct_count

end num_lines_passing_through_4x4_grid_l178_178648


namespace base_length_of_parallelogram_l178_178206

theorem base_length_of_parallelogram (A h : ℝ) (hA : A = 44) (hh : h = 11) :
  ∃ b : ℝ, b = 4 ∧ A = b * h :=
by
  sorry

end base_length_of_parallelogram_l178_178206


namespace find_digits_l178_178897

theorem find_digits (a b : ℕ) (h1 : (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9)) :
  (∃ (c : ℕ), 10000 * a + 6790 + b = 72 * c) ↔ (a = 3 ∧ b = 2) :=
by
  sorry

end find_digits_l178_178897


namespace distance_between_cities_is_39_l178_178520

noncomputable def city_distance : ℕ :=
  ∃ S : ℕ, (∀ x : ℕ, (0 ≤ x ∧ x ≤ S) → gcd x (S - x) ∈ {1, 3, 13}) ∧
             (S % (Nat.lcm (Nat.lcm 1 3) 13) = 0) ∧ S = 39

theorem distance_between_cities_is_39 :
  city_distance := 
by
  sorry

end distance_between_cities_is_39_l178_178520


namespace range_of_k_l178_178485

def P (x k : ℝ) : Prop := x^2 + k*x + 1 > 0
def Q (x k : ℝ) : Prop := k*x^2 + x + 2 < 0

theorem range_of_k (k : ℝ) : (¬ (P 2 k ∧ Q 2 k)) ↔ k ∈ (Set.Iic (-5/2) ∪ Set.Ici (-1)) := 
by
  sorry

end range_of_k_l178_178485


namespace count_lines_in_4x4_grid_l178_178692

theorem count_lines_in_4x4_grid : 
  let grid_points : Fin 4 × Fin 4 := 
  ∃! lines : set (set (Fin 4 × Fin 4)), 
  ∀ line ∈ lines, ∃ (p1 p2 : Fin 4 × Fin 4), p1 ≠ p2 ∧ p1 ∈ line ∧ p2 ∈ line ∧ (grid_points ⊆ line ⊆ grid_points) :=
  lines = 84 :=
sorry

end count_lines_in_4x4_grid_l178_178692


namespace pages_with_same_units_digit_l178_178392

theorem pages_with_same_units_digit :
  {x : ℕ | 1 ≤ x ∧ x ≤ 63 ∧ (x % 10) = (64 - x) % 10}.to_finset.card = 6 :=
by
  sorry

end pages_with_same_units_digit_l178_178392


namespace complex_pow_diff_zero_l178_178099

theorem complex_pow_diff_zero {i : ℂ} (h : i^2 = -1) : (2 + i)^(12) - (2 - i)^(12) = 0 := by
  sorry

end complex_pow_diff_zero_l178_178099


namespace range_of_m_l178_178847

theorem range_of_m (m : ℝ) (h : 2 * m + 3 < 4) : m < 1 / 2 :=
by
  sorry

end range_of_m_l178_178847


namespace students_not_in_any_subject_l178_178709

theorem students_not_in_any_subject (total_students mathematics_students chemistry_students biology_students
  mathematics_chemistry_students chemistry_biology_students mathematics_biology_students all_three_students: ℕ)
  (h_total: total_students = 120) 
  (h_m: mathematics_students = 70)
  (h_c: chemistry_students = 50)
  (h_b: biology_students = 40)
  (h_mc: mathematics_chemistry_students = 30)
  (h_cb: chemistry_biology_students = 20)
  (h_mb: mathematics_biology_students = 10)
  (h_all: all_three_students = 5) :
  total_students - ((mathematics_students - mathematics_chemistry_students - mathematics_biology_students + all_three_students) +
    (chemistry_students - chemistry_biology_students - mathematics_chemistry_students + all_three_students) +
    (biology_students - chemistry_biology_students - mathematics_biology_students + all_three_students) +
    (mathematics_chemistry_students + chemistry_biology_students + mathematics_biology_students - 2 * all_three_students)) = 20 :=
by sorry

end students_not_in_any_subject_l178_178709


namespace probability_proof_l178_178994

noncomputable def probability_greater_than_ten : ℚ :=
  let outcomes : List (ℤ × ℤ) := [(-1, -1), (-1, 1), (1, -1), (1, 1), (1, 2), (2, 1), (2, 2)]
  let valid_positions (card_start: ℤ) (spin_outcomes: List (ℤ × ℤ)) : List (ℤ × ℤ) :=
    spin_outcomes.filter (λ (x1, x2) => card_start + x1 + x2 > 10) 
  let valid_probabilities (starting_points : List ℤ) : ℚ :=
    (1 / 12) * starting_points.map (λ start => 
      (valid_positions start outcomes).length / outcomes.length
    ).sum
  valid_probabilities [8, 9, 10] + 
  valid_probabilities [11, 12]

theorem probability_proof : probability_greater_than_ten = 23 / 54 := by
  sorry

end probability_proof_l178_178994


namespace range_of_x_l178_178819

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3) :=
by
  sorry

end range_of_x_l178_178819


namespace value_of_expression_l178_178537

theorem value_of_expression : 2 - (-5) = 7 :=
by
  sorry

end value_of_expression_l178_178537


namespace compound_interest_rate_l178_178557

theorem compound_interest_rate :
  ∀ (P A : ℝ) (t n : ℕ) (r : ℝ),
  P = 12000 →
  A = 21500 →
  t = 5 →
  n = 1 →
  A = P * (1 + r / n) ^ (n * t) →
  r = 0.121898 :=
by
  intros P A t n r hP hA ht hn hCompound
  sorry

end compound_interest_rate_l178_178557


namespace number_of_lines_in_4_by_4_grid_l178_178651

/-- A 4-by-4 grid of lattice points -/
def lattice_points_4x4 : set (ℕ × ℕ) :=
  {(i, j) | i < 4 ∧ j < 4}

/-- A line in the Euclidean plane -/
def is_line (p1 p2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | ∃ λ : ℝ, p = (λ * (p2.1 - p1.1) + p1.1, λ * (p2.2 - p1.2) + p1.2)}

noncomputable def count_lines_through_points (points : set (ℕ × ℕ)) : ℕ :=
  /- counting logic to be implemented -/
  sorry

theorem number_of_lines_in_4_by_4_grid : count_lines_through_points lattice_points_4x4 = 70 :=
  sorry

end number_of_lines_in_4_by_4_grid_l178_178651


namespace count_lines_in_4x4_grid_l178_178695

theorem count_lines_in_4x4_grid : 
  let grid_points : Fin 4 × Fin 4 := 
  ∃! lines : set (set (Fin 4 × Fin 4)), 
  ∀ line ∈ lines, ∃ (p1 p2 : Fin 4 × Fin 4), p1 ≠ p2 ∧ p1 ∈ line ∧ p2 ∈ line ∧ (grid_points ⊆ line ⊆ grid_points) :=
  lines = 84 :=
sorry

end count_lines_in_4x4_grid_l178_178695


namespace tim_investment_l178_178769

noncomputable def initial_investment_required 
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / ((1 + r / n) ^ (n * t))

theorem tim_investment :
  initial_investment_required 100000 0.10 2 3 = 74622 :=
by
  sorry

end tim_investment_l178_178769


namespace find_other_number_l178_178568

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 61) (h_a : A = 210) : B = 671 :=
by
  sorry

end find_other_number_l178_178568


namespace largest_divisor_of_polynomial_l178_178771

theorem largest_divisor_of_polynomial (n : ℕ) (h : n % 2 = 0) : 
  105 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) :=
sorry

end largest_divisor_of_polynomial_l178_178771


namespace hyperbola_aux_lines_l178_178626

theorem hyperbola_aux_lines (a : ℝ) (h_a_positive : a > 0)
  (h_hyperbola_eqn : ∀ x y, (x^2 / a^2) - (y^2 / 16) = 1)
  (h_asymptotes : ∀ x y, y = 4/3 * x ∨ y = -4/3 * x) : 
  ∀ x, (x = 9/5 ∨ x = -9/5) := sorry

end hyperbola_aux_lines_l178_178626


namespace atomic_weight_of_nitrogen_l178_178205

-- Definitions from conditions
def molecular_weight := 53.0
def hydrogen_weight := 1.008
def chlorine_weight := 35.45
def hydrogen_atoms := 4
def chlorine_atoms := 1

-- The proof goal
theorem atomic_weight_of_nitrogen : 
  53.0 - (4.0 * 1.008) - 35.45 = 13.518 :=
by
  sorry

end atomic_weight_of_nitrogen_l178_178205


namespace johannes_sells_48_kg_l178_178479

-- Define Johannes' earnings
def earnings_wednesday : ℕ := 30
def earnings_friday : ℕ := 24
def earnings_today : ℕ := 42

-- Price per kilogram of cabbage
def price_per_kg : ℕ := 2

-- Prove that the total kilograms of cabbage sold is 48
theorem johannes_sells_48_kg :
  ((earnings_wednesday + earnings_friday + earnings_today) / price_per_kg) = 48 := by
  sorry

end johannes_sells_48_kg_l178_178479


namespace water_heaters_price_l178_178837

/-- 
  Suppose Oleg plans to sell 5000 units of water heaters. 
  The variable cost of producing and selling one water heater is 800 rubles,
  and the total fixed costs are 1,000,000 rubles. 
  Oleg wants his revenues to exceed expenses by 1,500,000 rubles.
  At what price should Oleg sell the water heaters to meet his target profit?
-/
theorem water_heaters_price
  (n : ℕ) (c_v C_f p_r : ℕ) 
  (h_n : n = 5000) 
  (h_c_v : c_v = 800) 
  (h_C_f : C_f = 1000000) 
  (h_p_r : p_r = 1500000) :
  ∃ p : ℕ, let total_variable_costs := n * c_v,
               total_expenses := C_f + total_variable_costs,
               required_revenue := total_expenses + p_r,
               p := required_revenue / n
           in p = 1300 :=
by
  use 1300
  let total_variable_costs := n * c_v
  let total_expenses := C_f + total_variable_costs
  let required_revenue := total_expenses + p_r
  let p := required_revenue / n
  sorry

end water_heaters_price_l178_178837


namespace arithmetic_sequence_a3_l178_178987

theorem arithmetic_sequence_a3 (a : ℕ → ℤ) (h1 : a 1 = 4) (h10 : a 10 = 22) (d : ℤ) (hd : ∀ n, a n = a 1 + (n - 1) * d) :
  a 3 = 8 :=
by
  -- Skipping the proof
  sorry

end arithmetic_sequence_a3_l178_178987


namespace question1_perpendicular_question2_parallel_l178_178607

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

noncomputable def vector_k_a_plus_2_b (k : ℝ) (a b : Vector2D) : Vector2D :=
  ⟨k * a.x + 2 * b.x, k * a.y + 2 * b.y⟩

noncomputable def vector_2_a_minus_4_b (a b : Vector2D) : Vector2D :=
  ⟨2 * a.x - 4 * b.x, 2 * a.y - 4 * b.y⟩

def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

def parallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

def opposite_direction (v1 v2 : Vector2D) : Prop :=
  parallel v1 v2 ∧ v1.x * v2.x + v1.y * v2.y < 0

noncomputable def vector_a : Vector2D := ⟨1, 1⟩
noncomputable def vector_b : Vector2D := ⟨2, 3⟩

theorem question1_perpendicular (k : ℝ) : 
  perpendicular (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b) ↔ 
  k = -21 / 4 :=
sorry

theorem question2_parallel (k : ℝ) :
  (parallel (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b) ∧
  opposite_direction (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b)) ↔ 
  k = -1 / 2 :=
sorry

end question1_perpendicular_question2_parallel_l178_178607


namespace initial_storks_count_l178_178366

-- Definitions based on the conditions provided
def initialBirds : ℕ := 3
def additionalStorks : ℕ := 6
def totalBirdsAndStorks : ℕ := 13

-- The mathematical statement to be proved
theorem initial_storks_count (S : ℕ) (h : initialBirds + S + additionalStorks = totalBirdsAndStorks) : S = 4 :=
by
  sorry

end initial_storks_count_l178_178366


namespace pencils_in_all_l178_178107

/-- Eugene's initial number of pencils -/
def initial_pencils : ℕ := 51

/-- Pencils Eugene gets from Joyce -/
def additional_pencils : ℕ := 6

/-- Total number of pencils Eugene has in all -/
def total_pencils : ℕ :=
  initial_pencils + additional_pencils

/-- Proof that Eugene has 57 pencils in all -/
theorem pencils_in_all : total_pencils = 57 := by
  sorry

end pencils_in_all_l178_178107


namespace plane_speed_ratio_train_l178_178437

def distance (speed time : ℝ) := speed * time

theorem plane_speed_ratio_train (x y z : ℝ)
  (h_train : distance x 20 = distance y 10)
  (h_wait_time : z > 5)
  (h_plane_meet_train : distance y (8/9) = distance x (z + 8/9)) :
  y = 10 * x :=
by {
  sorry
}

end plane_speed_ratio_train_l178_178437


namespace arrange_2022_l178_178321

def digits := [2, 0, 2, 2]

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

theorem arrange_2022 : 
  let valid_numbers := {n : ℕ | (digits.permutations.map (λ d, d.foldl (λ acc x, acc * 10 + x) 0)).to_finset ∋ n ∧ is_multiple_of_2 n} in
  valid_numbers.to_finset.card = 4 :=
by sorry

end arrange_2022_l178_178321


namespace find_x_range_l178_178627

theorem find_x_range {x : ℝ} : 
  (∀ (m : ℝ), abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 ) →
  ( ( -1 + Real.sqrt 7 ) / 2 < x ∧ x < ( 1 + Real.sqrt 3 ) / 2 ) :=
by
  intros h
  sorry

end find_x_range_l178_178627


namespace minimum_jellybeans_l178_178095

theorem minimum_jellybeans (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n = 164 :=
by sorry

end minimum_jellybeans_l178_178095


namespace sum_of_cubes_l178_178223

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178223


namespace sin_alpha_value_l178_178130

open Real

theorem sin_alpha_value (α β : ℝ) 
  (h1 : cos (α - β) = 3 / 5) 
  (h2 : sin β = -5 / 13) 
  (h3 : 0 < α ∧ α < π / 2) 
  (h4 : -π / 2 < β ∧ β < 0) 
  : sin α = 33 / 65 :=
sorry

end sin_alpha_value_l178_178130


namespace certain_event_is_A_l178_178240

def isCertainEvent (event : Prop) : Prop := event

axiom event_A : Prop
axiom event_B : Prop
axiom event_C : Prop
axiom event_D : Prop

axiom event_A_is_certain : isCertainEvent event_A
axiom event_B_is_not_certain : ¬ isCertainEvent event_B
axiom event_C_is_impossible : ¬ event_C
axiom event_D_is_not_certain : ¬ isCertainEvent event_D

theorem certain_event_is_A : isCertainEvent event_A := by
  exact event_A_is_certain

end certain_event_is_A_l178_178240


namespace awareness_not_related_to_education_level_l178_178424

def low_education : ℕ := 35 + 35 + 80 + 40 + 60 + 150
def high_education : ℕ := 55 + 64 + 6 + 110 + 140 + 25

def a : ℕ := 150
def b : ℕ := 125
def c : ℕ := 250
def d : ℕ := 275
def n : ℕ := 800

-- K^2 calculation
def K2 : ℚ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Critical value for 95% confidence
def critical_value_95 : ℚ := 3.841

theorem awareness_not_related_to_education_level : K2 < critical_value_95 :=
by
  -- proof to be added here
  sorry

end awareness_not_related_to_education_level_l178_178424


namespace instantaneous_velocity_at_3_l178_178460

open Real

variable (s : ℝ → ℝ)
variable (t : ℝ)

def velocity (s : ℝ → ℝ) (t : ℝ) : ℝ := deriv s t

theorem instantaneous_velocity_at_3 (h : ∀ t, s t = 3 * t^2) : velocity s 3 = 18 :=
by
  -- Omitted code to establish the proof
  sorry

end instantaneous_velocity_at_3_l178_178460


namespace function_properties_l178_178620

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 / x

theorem function_properties : 
  (∀ x : ℝ, x ≠ 0 → f (1 / x) + 2 * f x = 3 * x) ∧ 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x : ℝ, 0 < x → ∀ y : ℝ, x < y → f x < f y) := by
  -- Proof of the theorem would go here
  sorry

end function_properties_l178_178620


namespace triangular_region_area_l178_178260

theorem triangular_region_area :
  let x_intercept := 4
  let y_intercept := 6
  let area := (1 / 2) * x_intercept * y_intercept
  area = 12 :=
by
  sorry

end triangular_region_area_l178_178260


namespace factor_expression_l178_178811

theorem factor_expression (x y z : ℝ) :
  ((x^3 - y^3)^3 + (y^3 - z^3)^3 + (z^3 - x^3)^3) / 
  ((x - y)^3 + (y - z)^3 + (z - x)^3) = 
  ((x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2)) :=
by {
  sorry  -- The proof goes here
}

end factor_expression_l178_178811


namespace sin_alpha_third_quadrant_l178_178841

theorem sin_alpha_third_quadrant 
  (α : ℝ) 
  (hcos : Real.cos α = -3 / 5) 
  (hquad : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.sin α = -4 / 5 := 
sorry

end sin_alpha_third_quadrant_l178_178841


namespace max_intersections_pentagon_triangle_max_intersections_pentagon_quadrilateral_l178_178360

-- Define the pentagon and various other polygons
inductive PolygonType
| pentagon
| triangle
| quadrilateral

-- Define a function that calculates the maximum number of intersections
def max_intersections (K L : PolygonType) : ℕ :=
  match K, L with
  | PolygonType.pentagon, PolygonType.triangle => 10
  | PolygonType.pentagon, PolygonType.quadrilateral => 16
  | _, _ => 0  -- We only care about the cases specified in our problem

-- Theorem a): When L is a triangle, the intersections should be 10
theorem max_intersections_pentagon_triangle : max_intersections PolygonType.pentagon PolygonType.triangle = 10 :=
  by 
  -- provide proof here, but currently it is skipped with sorry
  sorry

-- Theorem b): When L is a quadrilateral, the intersections should be 16
theorem max_intersections_pentagon_quadrilateral : max_intersections PolygonType.pentagon PolygonType.quadrilateral = 16 :=
  by
  -- provide proof here, but currently it is skipped with sorry
  sorry

end max_intersections_pentagon_triangle_max_intersections_pentagon_quadrilateral_l178_178360


namespace probability_no_own_dress_l178_178397

open Finset

noncomputable def derangements (n : ℕ) : Finset (Perm (Fin n)) :=
  filter (λ σ : Perm (Fin n), ∀ i, σ i ≠ i) univ

theorem probability_no_own_dress :
  let daughters := 3 in
  let total_permutations := univ.card (Perm (Fin daughters)) in
  let derangements_count := (derangements daughters).card in
  let probability := (derangements_count : ℚ) / total_permutations in
  probability = 1 / 3 :=
by
  sorry

end probability_no_own_dress_l178_178397


namespace while_loop_output_correct_do_while_loop_output_correct_l178_178070

def while_loop (a : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (7 - i)).map (λ n => (i + n, a + n + 1))

def do_while_loop (x : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (10 - i + 1)).map (λ n => (i + n, x + (n + 1) * 10))

theorem while_loop_output_correct : while_loop 2 1 = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)] := 
sorry

theorem do_while_loop_output_correct : do_while_loop 100 1 = [(1, 110), (2, 120), (3, 130), (4, 140), (5, 150), (6, 160), (7, 170), (8, 180), (9, 190), (10, 200)] :=
sorry

end while_loop_output_correct_do_while_loop_output_correct_l178_178070


namespace rank_of_A_eq_k_l178_178487

open Matrix Complex

variables {n : ℕ} {k : ℂ} {A : Matrix (Fin n) (Fin n) ℂ}

-- Conditions provided in the problem
def positive_integer (n : ℕ) := n > 0

def trace_nonzero (A : Matrix (Fin n) (Fin n) ℂ) := A.trace ≠ 0

def rank_condition (A : Matrix (Fin n) (Fin n) ℂ) (k : ℂ) :=
  rank A + rank (A.trace • (1 : Matrix (Fin n) (Fin n) ℂ) - k • A) = n

-- Proving the main theorem
theorem rank_of_A_eq_k (hn : positive_integer n) (hk : k ∈ ℂ) 
  (hA : A ∈ Matrix (Fin n) (Fin n) ℂ) (htrace : trace_nonzero A)
  (hrank : rank_condition A k) : rank A = k :=
sorry

end rank_of_A_eq_k_l178_178487


namespace gloria_turtle_time_l178_178140

theorem gloria_turtle_time (g_time : ℕ) (george_time : ℕ) (gloria_time : ℕ) 
  (h1 : g_time = 6) 
  (h2 : george_time = g_time - 2)
  (h3 : gloria_time = 2 * george_time) : 
  gloria_time = 8 :=
sorry

end gloria_turtle_time_l178_178140


namespace sum_of_cubes_l178_178227

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l178_178227


namespace lines_in_4_by_4_grid_l178_178678

-- Definition for the grid and the number of lattice points.
def grid : Nat := 16

-- Theorem stating that the number of different lines passing through at least two points in a 4-by-4 grid of lattice points.
theorem lines_in_4_by_4_grid : 
  (number_of_lines : Nat) → number_of_lines = 40 ↔ grid = 16 := 
by
  -- Calculating number of lines passing through at least two points in a 4-by-4 grid.
  sorry -- proof skipped

end lines_in_4_by_4_grid_l178_178678


namespace geometric_sum_four_terms_l178_178289

/-- 
Given that the sequence {a_n} is a geometric sequence with the sum of its 
first n terms denoted as S_n, if S_4=1 and S_8=4, prove that a_{13}+a_{14}+a_{15}+a_{16}=27 
-/ 
theorem geometric_sum_four_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), S (n + 1) = a (n + 1) + S n) 
  (h2 : S 4 = 1) 
  (h3 : S 8 = 4) 
  : (a 13) + (a 14) + (a 15) + (a 16) = 27 := 
sorry

end geometric_sum_four_terms_l178_178289


namespace regular_polygon_of_45_deg_l178_178256

def is_regular_polygon (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 2 ∧ 360 % k = 0 ∧ n = 360 / k

def regular_polygon_is_octagon (angle : ℕ) : Prop :=
  is_regular_polygon 8 ∧ angle = 45

theorem regular_polygon_of_45_deg : regular_polygon_is_octagon 45 :=
  sorry

end regular_polygon_of_45_deg_l178_178256


namespace provisions_last_60_days_l178_178582

/-
A garrison of 1000 men has provisions for a certain number of days.
At the end of 15 days, a reinforcement of 1250 arrives, and it is now found that the provisions will last only for 20 days more.
Prove that the provisions were supposed to last initially for 60 days.
-/

def initial_provisions (D : ℕ) : Prop :=
  let initial_garrison := 1000
  let reinforcement_garrison := 1250
  let days_spent := 15
  let remaining_days := 20
  initial_garrison * (D - days_spent) = (initial_garrison + reinforcement_garrison) * remaining_days

theorem provisions_last_60_days (D : ℕ) : initial_provisions D → D = 60 := by
  sorry

end provisions_last_60_days_l178_178582


namespace distinct_values_for_T_l178_178155

-- Define the conditions given in the problem:
def distinct_digits (n : ℕ) : Prop :=
  n / 1000 ≠ (n / 100 % 10) ∧ n / 1000 ≠ (n / 10 % 10) ∧ n / 1000 ≠ (n % 10) ∧
  (n / 100 % 10) ≠ (n / 10 % 10) ∧ (n / 100 % 10) ≠ (n % 10) ∧
  (n / 10 % 10) ≠ (n % 10)

def Psum (P S T : ℕ) : Prop := P + S = T

-- Main theorem statement:
theorem distinct_values_for_T : ∀ (P S T : ℕ),
  distinct_digits P ∧ distinct_digits S ∧ distinct_digits T ∧
  Psum P S T → 
  (∃ (values : Finset ℕ), values.card = 7 ∧ ∀ val ∈ values, val = T) :=
by
  sorry

end distinct_values_for_T_l178_178155


namespace vectors_orthogonal_dot_product_l178_178949

theorem vectors_orthogonal_dot_product (y : ℤ) :
  (3 * -2) + (4 * y) + (-1 * 5) = 0 → y = 11 / 4 :=
by
  sorry

end vectors_orthogonal_dot_product_l178_178949


namespace polynomial_remainder_l178_178558

theorem polynomial_remainder (x : ℤ) :
  let dividend := 3*x^3 - 2*x^2 - 23*x + 60
  let divisor := x - 4
  let quotient := 3*x^2 + 10*x + 17
  let remainder := 128
  dividend = divisor * quotient + remainder :=
by 
  -- proof steps would go here, but we use "sorry" as instructed
  sorry

end polynomial_remainder_l178_178558


namespace factor_polynomial_l178_178196

def p (x y z : ℝ) : ℝ := x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2)

theorem factor_polynomial (x y z : ℝ) : 
  p x y z = (x - y) * (y - z) * (z - x) * -(x * y + x * z + y * z) :=
by 
  simp [p]
  sorry

end factor_polynomial_l178_178196


namespace value_of_fraction_l178_178842

theorem value_of_fraction (x y : ℝ) (h : 1 / x - 1 / y = 2) : (x + x * y - y) / (x - x * y - y) = 1 / 3 :=
by
  sorry

end value_of_fraction_l178_178842


namespace different_lines_through_two_points_in_4_by_4_grid_l178_178632

theorem different_lines_through_two_points_in_4_by_4_grid : 
  let points := fin 4 × fin 4 in
  let number_of_lines := 
    (nat.choose 16 2) - 
    (8 * (4 - 1)) - 
    (2 * (4 - 1)) in
  number_of_lines = 90 :=
by
  sorry

end different_lines_through_two_points_in_4_by_4_grid_l178_178632


namespace num_arrangements_l178_178599

theorem num_arrangements : 
  ∃ (teachers students : Finset ℕ), 
  teachers.card = 2 ∧
  students.card = 4 ∧ 
  (∃ (groupA groupB : Finset ℕ), 
    groupA.card = 3 ∧ 
    groupB.card = 3 ∧ 
    groupA ∩ groupB = ∅ ∧ 
    teachers ∩ students = ∅ ∧
    (teachers ∪ students = groupA ∪ groupB) ∧ 
    (card groupA.choose 1 * card students.choose 2 = 12)) :=
sorry

end num_arrangements_l178_178599


namespace price_difference_l178_178083

theorem price_difference (P : ℝ) :
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  difference = 0.24 * P := by
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  sorry

end price_difference_l178_178083


namespace pictures_per_album_l178_178039

-- Definitions based on the conditions
def phone_pics := 35
def camera_pics := 5
def total_pics := phone_pics + camera_pics
def albums := 5 

-- Statement that needs to be proven
theorem pictures_per_album : total_pics / albums = 8 := by
  sorry

end pictures_per_album_l178_178039


namespace max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2_l178_178410

theorem max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2 (n : ℕ) (hn : n > 0) :
  ∃ m, m = Nat.gcd (15 * n + 4) (9 * n + 2) ∧ m ≤ 2 :=
by
  sorry

end max_gcd_of_15n_plus_4_and_9n_plus_2_eq_2_l178_178410


namespace book_arrangement_l178_178696

theorem book_arrangement :
  let total_books := 6
  let identical_books := 3
  let unique_arrangements := Nat.factorial total_books / Nat.factorial identical_books
  unique_arrangements = 120 := by
  sorry

end book_arrangement_l178_178696


namespace quad_root_l178_178122

theorem quad_root (m : ℝ) (β : ℝ) (root_condition : ∃ α : ℝ, α = -5 ∧ (α + β) * (α * β) = x^2 + m * x - 10) : β = 2 :=
by
  sorry

end quad_root_l178_178122


namespace constant_value_AP_AQ_l178_178282

noncomputable def ellipse_trajectory (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def circle_O (x y : ℝ) : Prop :=
  (x^2 + y^2) = 12 / 7

theorem constant_value_AP_AQ (x y : ℝ) (h : circle_O x y) :
  ∃ (P Q : ℝ × ℝ), ellipse_trajectory (P.1) (P.2) ∧ ellipse_trajectory (Q.1) (Q.2) ∧ 
  ((P.1 - x) * (Q.1 - x) + (P.2 - y) * (Q.2 - y)) = - (12 / 7) :=
sorry

end constant_value_AP_AQ_l178_178282


namespace find_a_value_l178_178193

-- Given values
def month_code : List ℝ := [1, 2, 3, 4, 5]
def prices (a : ℝ) : List ℝ := [0.5, a, 1, 1.4, 1.5]

-- Linear regression equation parameters
def lin_reg_slope : ℝ := 0.28
def lin_reg_intercept : ℝ := 0.16

-- Average function
def average (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Proof statement
theorem find_a_value (a : ℝ) (h : (average (prices a)) = lin_reg_slope * (average month_code) + lin_reg_intercept) : a = 0.6 :=
  sorry

end find_a_value_l178_178193


namespace minutes_per_mile_l178_178878

-- Define the total distance Peter needs to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def walked_distance : ℝ := 1.0

-- Define the remaining time Peter needs to walk to reach the grocery store
def remaining_time : ℝ := 30.0

-- Define the remaining distance Peter needs to walk
def remaining_distance : ℝ := total_distance - walked_distance

-- The desired statement to prove: it takes Peter 20 minutes to walk one mile
theorem minutes_per_mile : remaining_distance / remaining_time = 1.0 / 20.0 := by
  sorry

end minutes_per_mile_l178_178878


namespace division_result_l178_178372

theorem division_result :
  3486 / 189 = 18.444444444444443 := by
  sorry

end division_result_l178_178372


namespace concert_ticket_cost_l178_178395

theorem concert_ticket_cost :
  ∀ (x : ℝ), 
    (12 * x - 2 * 0.05 * x = 476) → 
    x = 40 :=
by
  intros x h
  sorry

end concert_ticket_cost_l178_178395


namespace nine_chapters_compensation_difference_l178_178990

noncomputable def pig_consumes (x : ℝ) := x
noncomputable def sheep_consumes (x : ℝ) := 2 * x
noncomputable def horse_consumes (x : ℝ) := 4 * x
noncomputable def cow_consumes (x : ℝ) := 8 * x

theorem nine_chapters_compensation_difference :
  ∃ (x : ℝ), 
    cow_consumes x + horse_consumes x + sheep_consumes x + pig_consumes x = 9 ∧
    (horse_consumes x - pig_consumes x) = 9 / 5 :=
by
  sorry

end nine_chapters_compensation_difference_l178_178990


namespace percentage_discount_on_pencils_l178_178157

-- Establish the given conditions
variable (cucumbers pencils price_per_cucumber price_per_pencil total_spent : ℕ)
variable (h1 : cucumbers = 100)
variable (h2 : price_per_cucumber = 20)
variable (h3 : price_per_pencil = 20)
variable (h4 : total_spent = 2800)
variable (h5 : cucumbers = 2 * pencils)

-- Propose the statement to be proved
theorem percentage_discount_on_pencils : 20 * pencils * price_per_pencil = 20 * (total_spent - cucumbers * price_per_cucumber) ∧ pencils = 50 ∧ ((total_spent - cucumbers * price_per_cucumber) * 100 = 80 * pencils * price_per_pencil) :=
by
  sorry

end percentage_discount_on_pencils_l178_178157


namespace total_hangers_l178_178704

def pink_hangers : ℕ := 7
def green_hangers : ℕ := 4
def blue_hangers : ℕ := green_hangers - 1
def yellow_hangers : ℕ := blue_hangers - 1

theorem total_hangers :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 := by
  sorry

end total_hangers_l178_178704


namespace episodes_relationship_l178_178870

variable (x y z : ℕ)

theorem episodes_relationship 
  (h1 : x * z = 50) 
  (h2 : y * z = 75) : 
  y = (3 / 2) * x ∧ z = 50 / x := 
by
  sorry

end episodes_relationship_l178_178870


namespace subset_A_inter_B_eq_A_l178_178834

variable {x : ℝ}
def A (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2 * k}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem subset_A_inter_B_eq_A (k : ℝ) : (A k ∩ B = A k) ↔ (k ≤ 3 / 2) := 
sorry

end subset_A_inter_B_eq_A_l178_178834


namespace find_constant_g_l178_178078

open Real

theorem find_constant_g
  (g : ℝ → ℝ)
  (hg_diff : ∀ x ∈ Icc 0 π, differentiable_at ℝ g x)
  (hg_cont_diff : continuous_on (deriv g) (Icc 0 π))
  (f := λ x => g x * sin x)
  (h_eq : ∫ x in 0..π, (f x)^2 = ∫ x in 0..π, (deriv f x)^2) :
  ∃ c : ℝ, ∀ x ∈ Icc 0 π, g x = c :=
begin
  -- proof placeholder
  sorry
end

end find_constant_g_l178_178078


namespace intersection_A_complement_B_l178_178080

def set_A : Set ℝ := {x | 1 < x ∧ x < 4}
def set_B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_Complement_B : Set ℝ := {x | x < -1 ∨ x > 3}
def set_Intersection : Set ℝ := {x | set_A x ∧ set_Complement_B x}

theorem intersection_A_complement_B : set_Intersection = {x | 3 < x ∧ x < 4} := by
  sorry

end intersection_A_complement_B_l178_178080


namespace find_probability_l178_178621

noncomputable def probability_distribution (X : ℕ → ℝ) := ∀ k, X k = 1 / (2^k)

theorem find_probability (X : ℕ → ℝ) (h : probability_distribution X) :
  X 3 + X 4 = 3 / 16 :=
by
  sorry

end find_probability_l178_178621


namespace average_of_numbers_l178_178187

theorem average_of_numbers (a b c d e : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) (h₄ : d = 11) (h₅ : e = 12) :
  (a + b + c + d + e) / 5 = 10 :=
by
  sorry

end average_of_numbers_l178_178187


namespace find_a_n_plus_b_n_l178_178137

noncomputable def a (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else if n = 2 then 3 
  else sorry -- Placeholder for proper recursive implementation

noncomputable def b (n : ℕ) : ℕ := 
  if n = 1 then 5
  else sorry -- Placeholder for proper recursive implementation

theorem find_a_n_plus_b_n (n : ℕ) (i j k l : ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : b 1 = 5) 
  (h4 : i + j = k + l) (h5 : a i + b j = a k + b l) : a n + b n = 4 * n + 2 := 
by
  sorry

end find_a_n_plus_b_n_l178_178137


namespace rectangle_width_of_square_l178_178502

theorem rectangle_width_of_square (side_length_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (h1 : side_length_square = 3) (h2 : length_rectangle = 3)
  (h3 : (side_length_square ^ 2) = length_rectangle * width_rectangle) : width_rectangle = 3 :=
by
  sorry

end rectangle_width_of_square_l178_178502


namespace typing_speed_ratio_l178_178379

theorem typing_speed_ratio (T M : ℝ) 
  (h1 : T + M = 12) 
  (h2 : T + 1.25 * M = 14) : 
  M / T = 2 :=
by 
  -- The proof will go here
  sorry

end typing_speed_ratio_l178_178379


namespace problem_statement_l178_178129

theorem problem_statement {a b c d : ℝ} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (4 / (b - c)) + (9 / (c - d)) ≥ (36 / (a - d)) :=
by
  sorry -- proof is omitted according to the instructions

end problem_statement_l178_178129


namespace angle_B_area_of_triangle_l178_178969

/-
Given a triangle ABC with angle A, B, C and sides a, b, c opposite to these angles respectively.
Consider the conditions:
- A = π/6
- b = (4 + 2 * sqrt 3) * a * cos B
- b = 1

Prove:
1. B = 5 * π / 12
2. The area of triangle ABC = 1 / 4
-/

namespace TriangleProof

open Real

def triangle_conditions (A B C a b c : ℝ) : Prop :=
  A = π / 6 ∧
  b = (4 + 2 * sqrt 3) * a * cos B ∧
  b = 1

theorem angle_B (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  B = 5 * π / 12 :=
sorry

theorem area_of_triangle (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  1 / 2 * b * c * sin A = 1 / 4 :=
sorry

end TriangleProof

end angle_B_area_of_triangle_l178_178969


namespace ratio_five_to_one_l178_178559

theorem ratio_five_to_one (x : ℕ) (h : 5 * 12 = x) : x = 60 :=
by
  sorry

end ratio_five_to_one_l178_178559


namespace change_combinations_l178_178972

def isValidCombination (nickels dimes quarters : ℕ) : Prop :=
  nickels * 5 + dimes * 10 + quarters * 25 = 50 ∧ quarters ≤ 1

theorem change_combinations : {n // ∃ (combinations : ℕ) (nickels dimes quarters : ℕ), 
  n = combinations ∧ isValidCombination nickels dimes quarters ∧ 
  ((nickels, dimes, quarters) = (10, 0, 0) ∨
   (nickels, dimes, quarters) = (8, 1, 0) ∨
   (nickels, dimes, quarters) = (6, 2, 0) ∨
   (nickels, dimes, quarters) = (4, 3, 0) ∨
   (nickels, dimes, quarters) = (2, 4, 0) ∨
   (nickels, dimes, quarters) = (0, 5, 0) ∨
   (nickels, dimes, quarters) = (5, 0, 1) ∨
   (nickels, dimes, quarters) = (3, 1, 1) ∨
   (nickels, dimes, quarters) = (1, 2, 1))}
  :=
  ⟨9, sorry⟩

end change_combinations_l178_178972


namespace hexagon_area_correct_l178_178414

structure Point where
  x : ℝ
  y : ℝ

def hexagon : List Point := [
  { x := 0, y := 0 },
  { x := 2, y := 4 },
  { x := 6, y := 4 },
  { x := 8, y := 0 },
  { x := 6, y := -4 },
  { x := 2, y := -4 }
]

def area_of_hexagon (hex : List Point) : ℝ :=
  -- Assume a function that calculates the area of a polygon given a list of vertices
  sorry

theorem hexagon_area_correct : area_of_hexagon hexagon = 16 :=
  sorry

end hexagon_area_correct_l178_178414


namespace different_lines_through_two_points_in_4_by_4_grid_l178_178631

theorem different_lines_through_two_points_in_4_by_4_grid : 
  let points := fin 4 × fin 4 in
  let number_of_lines := 
    (nat.choose 16 2) - 
    (8 * (4 - 1)) - 
    (2 * (4 - 1)) in
  number_of_lines = 90 :=
by
  sorry

end different_lines_through_two_points_in_4_by_4_grid_l178_178631


namespace isosceles_right_triangle_area_l178_178200

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = 6 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 18 := by 
  sorry

end isosceles_right_triangle_area_l178_178200


namespace wendy_albums_used_l178_178549

def total_pictures : ℕ := 45
def pictures_in_one_album : ℕ := 27
def pictures_per_album : ℕ := 2

theorem wendy_albums_used :
  let remaining_pictures := total_pictures - pictures_in_one_album
  let albums_used := remaining_pictures / pictures_per_album
  albums_used = 9 :=
by
  sorry

end wendy_albums_used_l178_178549


namespace convex_polygon_angles_eq_nine_l178_178357

theorem convex_polygon_angles_eq_nine (n : ℕ) (a : ℕ → ℝ) (d : ℝ)
  (h1 : a (n - 1) = 180)
  (h2 : ∀ k, a (k + 1) - a k = d)
  (h3 : d = 10) :
  n = 9 :=
by
  sorry

end convex_polygon_angles_eq_nine_l178_178357


namespace min_value_of_a_plus_b_l178_178280

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / (a + 1)) + (2 / (1 + b)) = 1) : 
  a + b ≥ 2 * Real.sqrt 2 + 1 :=
sorry

end min_value_of_a_plus_b_l178_178280


namespace prime_sum_eq_14_l178_178967

theorem prime_sum_eq_14 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 := 
sorry

end prime_sum_eq_14_l178_178967


namespace road_distance_l178_178508

theorem road_distance 
  (S : ℕ) 
  (h : ∀ x : ℕ, x ≤ S → ∃ d : ℕ, d ∈ {1, 3, 13} ∧ d = Nat.gcd x (S - x)) : 
  S = 39 :=
by
  sorry

end road_distance_l178_178508


namespace moles_CO2_required_l178_178110

theorem moles_CO2_required
  (moles_MgO : ℕ) 
  (moles_MgCO3 : ℕ) 
  (balanced_equation : ∀ (MgO CO2 MgCO3 : ℕ), MgO + CO2 = MgCO3) 
  (reaction_produces : moles_MgO = 3 ∧ moles_MgCO3 = 3) :
  3 = 3 :=
by
  sorry

end moles_CO2_required_l178_178110


namespace max_value_of_expression_l178_178911

theorem max_value_of_expression :
  ∀ r : ℝ, -3 * r^2 + 30 * r + 8 ≤ 83 :=
by
  -- Proof needed
  sorry

end max_value_of_expression_l178_178911


namespace melinda_probability_correct_l178_178489

def probability_two_digit_between_20_and_30 : ℚ :=
  11 / 36

theorem melinda_probability_correct :
  probability_two_digit_between_20_and_30 = 11 / 36 :=
by
  sorry

end melinda_probability_correct_l178_178489


namespace find_set_A_find_range_a_l178_178290

-- Define the universal set and the complement condition for A
def universal_set : Set ℝ := {x | true}
def complement_A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 > 0}

-- Define the set A
def set_A : Set ℝ := {x | -1/2 ≤ x ∧ x ≤ 2}

-- Define the set C
def set_C (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

-- Define the proof problem for part (1)
theorem find_set_A : { x | -1 / 2 ≤ x ∧ x ≤ 2 } = { x | ¬ (2 * x^2 - 3 * x - 2 > 0) } :=
by
  sorry

-- Define the proof problem for part (2)
theorem find_range_a (a : ℝ) (C_ne_empty : (set_C a).Nonempty) (sufficient_not_necessary : ∀ x, x ∈ set_C a → x ∈ set_A → x ∈ set_A) :
  a ∈ Set.Icc (-1/8 : ℝ) 0 ∪ Set.Icc 1 (4/3 : ℝ) :=
by
  sorry

end find_set_A_find_range_a_l178_178290


namespace distance_between_cities_l178_178521

variable {x S : ℕ}

/-- The distance between city A and city B is 39 kilometers -/
theorem distance_between_cities (S : ℕ) (h1 : ∀ x : ℕ, 0 ≤ x → x ≤ S → (GCD x (S - x) = 1 ∨ GCD x (S - x) = 3 ∨ GCD x (S - x) = 13)) :
  S = 39 :=
sorry

end distance_between_cities_l178_178521


namespace selling_price_correct_l178_178836

/-- Define the total number of units to be sold -/
def total_units : ℕ := 5000

/-- Define the variable cost per unit -/
def variable_cost_per_unit : ℕ := 800

/-- Define the total fixed costs -/
def fixed_costs : ℕ := 1000000

/-- Define the desired profit -/
def desired_profit : ℕ := 1500000

/-- The selling price p must be calculated such that revenues exceed expenses by the desired profit -/
theorem selling_price_correct : 
  ∃ p : ℤ, p = 1300 ∧ (total_units * p) - (fixed_costs + (total_units * variable_cost_per_unit)) = desired_profit :=
by
  sorry

end selling_price_correct_l178_178836


namespace value_of_expression_l178_178921

theorem value_of_expression :
  (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2000 :=
by {
  sorry
}

end value_of_expression_l178_178921


namespace problem_1_problem_2_l178_178135

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.logBase 9 (9^x + 1) + k * x

theorem problem_1 (k : ℝ) : 
  (∀ x, f x k = f (-x) k) → k = -1/2 := 
by
  intro h
  -- Proof must be filled by the user
  sorry

theorem problem_2 (a x : ℝ) (h_a : a > 0) : 
  (f x (-1/2) - Real.logBase 9 (a + 1/a) > 0) ↔ 
  (a ≠ 1 → (a > 1 → (x > Real.logBase 3 a ∨ x < Real.logBase 3 (1/a))) ∧ 
           (a < 1 → (x > Real.logBase 3 (1/a) ∨ x < Real.logBase 3 a))) ∧ 
  (a = 1 → x ≠ 0) := 
by
  intro ha
  -- Proof must be filled by the user
  sorry

end problem_1_problem_2_l178_178135


namespace discount_amount_correct_l178_178531

noncomputable def cost_price : ℕ := 180
noncomputable def markup_percentage : ℝ := 0.45
noncomputable def profit_percentage : ℝ := 0.20

theorem discount_amount_correct : 
  let markup := cost_price * markup_percentage
  let mp := cost_price + markup
  let profit := cost_price * profit_percentage
  let sp := cost_price + profit
  let discount_amount := mp - sp
  discount_amount = 45 :=
by
  sorry

end discount_amount_correct_l178_178531


namespace distance_between_cities_l178_178509

theorem distance_between_cities
  (S : ℤ)
  (h1 : ∀ x : ℤ, 0 ≤ x ∧ x ≤ S → (gcd x (S - x) = 1 ∨ gcd x (S - x) = 3 ∨ gcd x (S - x) = 13)) :
  S = 39 := 
sorry

end distance_between_cities_l178_178509


namespace cyclic_quadrilateral_l178_178720

theorem cyclic_quadrilateral (T : ℕ) (S : ℕ) (AB BC CD DA : ℕ) (M N : ℝ × ℝ) (AC BD PQ MN : ℝ) (m n : ℕ) :
  T = 2378 → 
  S = 2 + 3 + 7 + 8 → 
  AB = S - 11 → 
  BC = 2 → 
  CD = 3 → 
  DA = 10 → 
  AC * BD = 47 → 
  PQ / MN = 1/2 → 
  m + n = 3 :=
by
  sorry

end cyclic_quadrilateral_l178_178720


namespace remainder_when_n_plus_2947_divided_by_7_l178_178889

theorem remainder_when_n_plus_2947_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (n + 2947) % 7 = 3 :=
by
  sorry

end remainder_when_n_plus_2947_divided_by_7_l178_178889


namespace city_distance_GCD_l178_178511

open Int

theorem city_distance_GCD :
  ∃ S : ℤ, (∀ (x : ℤ), 0 ≤ x ∧ x ≤ S → ∃ d ∈ {1, 3, 13}, d = gcd x (S - x)) ∧ S = 39 :=
by
  sorry

end city_distance_GCD_l178_178511


namespace distinct_four_digit_integers_l178_178051

open Nat

theorem distinct_four_digit_integers (count_digs_18 : ℕ) :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (∃ d1 d2 d3 d4 : ℕ,
      d1 * d2 * d3 * d4 = 18 ∧
      d1 > 0 ∧ d1 < 10 ∧
      d2 > 0 ∧ d2 < 10 ∧
      d3 > 0 ∧ d3 < 10 ∧
      d4 > 0 ∧ d4 < 10 ∧
      n = d1 * 1000 + d2 * 100 + d3 * 10 + d4)) →
  count_digs_18 = 24 :=
sorry

end distinct_four_digit_integers_l178_178051


namespace solution_fraction_l178_178579

-- Conditions and definition of x
def initial_quantity : ℝ := 1
def concentration_70 : ℝ := 0.70
def concentration_25 : ℝ := 0.25
def concentration_new : ℝ := 0.35

-- Definition of the fraction of the solution replaced
def x (fraction : ℝ) : Prop :=
  concentration_70 * initial_quantity - concentration_70 * fraction + concentration_25 * fraction = concentration_new * initial_quantity

-- The theorem we need to prove
theorem solution_fraction : ∃ (fraction : ℝ), x fraction ∧ fraction = 7 / 9 :=
by
  use 7 / 9
  simp [x]
  sorry  -- Proof steps would be filled here

end solution_fraction_l178_178579


namespace cost_per_revision_l178_178900

theorem cost_per_revision
  (x : ℝ)
  (initial_cost : ℝ)
  (revised_once : ℝ)
  (revised_twice : ℝ)
  (total_pages : ℝ)
  (total_cost : ℝ)
  (cost_per_page_first_time : ℝ) :
  initial_cost = cost_per_page_first_time * total_pages →
  revised_once * x + revised_twice * (2 * x) + initial_cost = total_cost →
  revised_once + revised_twice + (total_pages - (revised_once + revised_twice)) = total_pages →
  total_pages = 200 →
  initial_cost = 1000 →
  cost_per_page_first_time = 5 →
  revised_once = 80 →
  revised_twice = 20 →
  total_cost = 1360 →
  x = 3 :=
by
  intros h_initial h_total_cost h_tot_pages h_tot_pages_200 h_initial_1000 h_cost_5 h_revised_once h_revised_twice h_given_cost
  -- Proof steps to be filled
  sorry

end cost_per_revision_l178_178900


namespace tan_alpha_plus_cot_alpha_l178_178438

theorem tan_alpha_plus_cot_alpha (α : Real) (h : Real.sin (2 * α) = 3 / 4) : 
  Real.tan α + 1 / Real.tan α = 8 / 3 :=
  sorry

end tan_alpha_plus_cot_alpha_l178_178438


namespace distinct_lines_count_in_4x4_grid_l178_178689

theorem distinct_lines_count_in_4x4_grid :
  let grid_points := (finRange 4).product (finRange 4)
  let lines := {line : Set (ℕ × ℕ) | ∃ (a b : ℤ), ∀ p ∈ line, a * (p.1:ℤ) + b * (p.2:ℤ) = 1}
  let distinct_lines := {line ∈ lines | ∃ (p1 p2 : ℕ × ℕ), p1 ∈ grid_points ∧ p2 ∈ grid_points ∧ p1 ≠ p2 ∧ line = {p | this line passes through p}}
  lines.card = 50 :=
by
  sorry

end distinct_lines_count_in_4x4_grid_l178_178689


namespace ratio_of_prices_l178_178798

-- Define the problem
theorem ratio_of_prices (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = CP + 0.2 * CP) 
  (h2 : SP2 = CP - 0.2 * CP) : 
  SP2 / SP1 = 2 / 3 :=
by
  -- proof
  sorry

end ratio_of_prices_l178_178798


namespace good_horse_catchup_l178_178781

theorem good_horse_catchup 
  (x : ℕ) 
  (good_horse_speed : ℕ) (slow_horse_speed : ℕ) (head_start_days : ℕ) 
  (H1 : good_horse_speed = 240)
  (H2 : slow_horse_speed = 150)
  (H3 : head_start_days = 12) :
  good_horse_speed * x - slow_horse_speed * x = slow_horse_speed * head_start_days :=
by
  sorry

end good_horse_catchup_l178_178781


namespace determine_continuous_function_l178_178866

open Real

theorem determine_continuous_function (f : ℝ → ℝ) 
  (h_continuous : Continuous f)
  (h_initial : f 0 = 1)
  (h_inequality : ∀ x y : ℝ, f (x + y) ≥ f x * f y) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = exp (k * x) :=
sorry

end determine_continuous_function_l178_178866


namespace fn_has_two_distinct_real_roots_l178_178161

def f (x : ℝ) : ℝ := x^2 + 2018 * x + 1

def f_iter (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
| 0 := id
| 1 := f
| (k+2) := f ∘ f_iter f (k+1)

theorem fn_has_two_distinct_real_roots (n : ℕ) (hn : 0 < n) : 
  ∃ (x y : ℝ), x ≠ y ∧ f_iter f n x = 0 ∧ f_iter f n y = 0 :=
sorry

end fn_has_two_distinct_real_roots_l178_178161


namespace totalTilesUsed_l178_178737

-- Define the dining room dimensions
def diningRoomLength : ℕ := 18
def diningRoomWidth : ℕ := 15

-- Define the border width
def borderWidth : ℕ := 2

-- Define tile dimensions
def tile1x1 : ℕ := 1
def tile2x2 : ℕ := 2

-- Calculate the number of tiles used along the length and width for the border
def borderTileCountLength : ℕ := 2 * 2 * (diningRoomLength - 2 * borderWidth)
def borderTileCountWidth : ℕ := 2 * 2 * (diningRoomWidth - 2 * borderWidth)

-- Total number of one-foot by one-foot tiles for the border
def totalBorderTileCount : ℕ := borderTileCountLength + borderTileCountWidth

-- Calculate the inner area dimensions
def innerLength : ℕ := diningRoomLength - 2 * borderWidth
def innerWidth : ℕ := diningRoomWidth - 2 * borderWidth
def innerArea : ℕ := innerLength * innerWidth

-- Number of two-foot by two-foot tiles needed
def tile2x2Count : ℕ := (innerArea + tile2x2 * tile2x2 - 1) / (tile2x2 * tile2x2) -- Ensures rounding up without floating point arithmetic

-- Prove that the total number of tiles used is 139
theorem totalTilesUsed : totalBorderTileCount + tile2x2Count = 139 := by
  sorry

end totalTilesUsed_l178_178737


namespace moles_of_NaCl_formed_l178_178601

-- Define the balanced chemical reaction and quantities
def chemical_reaction :=
  "NaOH + HCl → NaCl + H2O"

-- Define the initial moles of sodium hydroxide (NaOH) and hydrochloric acid (HCl)
def moles_NaOH : ℕ := 2
def moles_HCl : ℕ := 2

-- The stoichiometry from the balanced equation: 1 mole NaOH reacts with 1 mole HCl to produce 1 mole NaCl.
def stoichiometry_NaOH_to_NaCl : ℕ := 1
def stoichiometry_HCl_to_NaCl : ℕ := 1

-- Given the initial conditions, prove that 2 moles of NaCl are formed.
theorem moles_of_NaCl_formed :
  (moles_NaOH = 2) → (moles_HCl = 2) → 2 = 2 :=
by 
  sorry

end moles_of_NaCl_formed_l178_178601


namespace find_b_l178_178283

theorem find_b (b : ℚ) : (-4 : ℚ) * (45 / 4) = -45 → (-4 + 45 / 4) = -b → b = -29 / 4 := by
  intros h1 h2
  sorry

end find_b_l178_178283


namespace lines_intersect_first_quadrant_l178_178277

theorem lines_intersect_first_quadrant (k : ℝ) :
  (∃ (x y : ℝ), 2 * x + 7 * y = 14 ∧ k * x - y = k + 1 ∧ x > 0 ∧ y > 0) ↔ k > 0 :=
by
  sorry

end lines_intersect_first_quadrant_l178_178277


namespace distribute_items_l178_178467

theorem distribute_items : 
  (Nat.choose (5 + 3 - 1) (3 - 1)) * (Nat.choose (3 + 3 - 1) (3 - 1)) * (Nat.choose (2 + 3 - 1) (3 - 1)) = 1260 :=
by
  sorry

end distribute_items_l178_178467


namespace Eithan_savings_account_l178_178066

variable (initial_amount wife_firstson_share firstson_remaining firstson_secondson_share 
          secondson_remaining secondson_thirdson_share thirdson_remaining 
          charity_donation remaining_after_charity tax_rate final_remaining : ℝ)

theorem Eithan_savings_account:
  initial_amount = 5000 →
  wife_firstson_share = initial_amount * (2/5) →
  firstson_remaining = initial_amount - wife_firstson_share →
  firstson_secondson_share = firstson_remaining * (3/10) →
  secondson_remaining = firstson_remaining - firstson_secondson_share →
  thirdson_remaining = secondson_remaining * (1-0.30) →
  charity_donation = 200 →
  remaining_after_charity = thirdson_remaining - charity_donation →
  tax_rate = 0.05 →
  final_remaining = remaining_after_charity * (1 - tax_rate) →
  final_remaining = 927.2 := 
  by
    intros
    sorry

end Eithan_savings_account_l178_178066


namespace johannes_cabbage_sales_l178_178477

def price_per_kg : ℝ := 2
def earnings_wednesday : ℝ := 30
def earnings_friday : ℝ := 24
def earnings_today : ℝ := 42

theorem johannes_cabbage_sales :
  (earnings_wednesday / price_per_kg) + (earnings_friday / price_per_kg) + (earnings_today / price_per_kg) = 48 := by
  sorry

end johannes_cabbage_sales_l178_178477


namespace expressions_same_type_l178_178097

def same_type_as (e1 e2 : ℕ × ℕ) : Prop :=
  e1 = e2

def exp_of_expr (a_exp b_exp : ℕ) : ℕ × ℕ :=
  (a_exp, b_exp)

def exp_3a2b := exp_of_expr 2 1
def exp_neg_ba2 := exp_of_expr 2 1

theorem expressions_same_type :
  same_type_as exp_neg_ba2 exp_3a2b :=
by
  sorry

end expressions_same_type_l178_178097


namespace max_k_value_l178_178011

noncomputable def max_k (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : ℝ :=
  let k := (-1 + Real.sqrt 7) / 2
  k

theorem max_k_value (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end max_k_value_l178_178011


namespace find_abs_of_y_l178_178386

theorem find_abs_of_y (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 20 * x^3 - 15 * x = 3) : 
  |20 * y^3 - 15 * y| = 4 := 
sorry

end find_abs_of_y_l178_178386


namespace servant_cash_received_l178_178459

theorem servant_cash_received (salary_cash : ℕ) (turban_value : ℕ) (months_worked : ℕ) (total_months : ℕ)
  (h_salary_cash : salary_cash = 90) (h_turban_value : turban_value = 70) (h_months_worked : months_worked = 9)
  (h_total_months : total_months = 12) : 
  salary_cash * months_worked / total_months + (turban_value * months_worked / total_months) - turban_value = 50 := by
sorry

end servant_cash_received_l178_178459


namespace find_other_root_l178_178120

theorem find_other_root (m : ℝ) (α : ℝ) :
  (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C m * Polynomial.X - Polynomial.C 10 = 0) →
  (α = -5) →
  ∃ β : ℝ, (α + β = -m) ∧ (α * β = -10) :=
by 
  sorry

end find_other_root_l178_178120


namespace value_of_fraction_l178_178150

variable (m n : ℚ)

theorem value_of_fraction (h₁ : 3 * m + 2 * n = 0) (h₂ : m ≠ 0 ∧ n ≠ 0) :
  (m / n - n / m) = 5 / 6 := 
sorry

end value_of_fraction_l178_178150


namespace tan_alpha_ratio_expression_l178_178439

variable (α : Real)
variable (h1 : Real.sin α = 3/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha {α : Real}
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  : Real.tan α = -3/4 := sorry

theorem ratio_expression {α : Real}
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  : (2 * Real.sin α + 3 * Real.cos α) / (Real.cos α - Real.sin α) = 6/7 := sorry

end tan_alpha_ratio_expression_l178_178439


namespace cost_formula_l178_178195

-- Definitions based on conditions
def base_cost : ℕ := 15
def additional_cost_per_pound : ℕ := 5
def environmental_fee : ℕ := 2

-- Definition of cost function
def cost (P : ℕ) : ℕ := base_cost + additional_cost_per_pound * (P - 1) + environmental_fee

-- Theorem stating the formula for the cost C
theorem cost_formula (P : ℕ) (h : 1 ≤ P) : cost P = 12 + 5 * P :=
by
  -- Proof would go here
  sorry

end cost_formula_l178_178195


namespace lottery_probability_l178_178986

theorem lottery_probability (total_tickets winners non_winners people : ℕ)
  (h_total : total_tickets = 10) (h_winners : winners = 3) (h_non_winners : non_winners = 7) (h_people : people = 5) :
  1 - (Nat.choose non_winners people : ℚ) / (Nat.choose total_tickets people : ℚ) = 77 / 84 := 
by
  sorry

end lottery_probability_l178_178986


namespace hyperbola_proof_l178_178615

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 4 = 1

def hyperbola_conditions (origin : ℝ × ℝ) (eccentricity : ℝ) (radius : ℝ) (focus : ℝ × ℝ) : Prop :=
  origin = (0, 0) ∧
  focus.1 = 0 ∧
  eccentricity = Real.sqrt 5 / 2 ∧
  radius = 2

theorem hyperbola_proof :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), hyperbola_conditions (0, 0) (Real.sqrt 5 / 2) 2 (0, c) → 
    C x y ↔ hyperbola_equation x y) :=
by
  sorry

end hyperbola_proof_l178_178615


namespace waiter_total_customers_l178_178264

def numCustomers (T : ℕ) (totalTips : ℕ) (tipPerCustomer : ℕ) (numNoTipCustomers : ℕ) : ℕ :=
  T + numNoTipCustomers

theorem waiter_total_customers
  (T : ℕ)
  (h1 : 3 * T = 6)
  (numNoTipCustomers : ℕ := 5)
  (total := numCustomers T 6 3 numNoTipCustomers) :
  total = 7 := by
  sorry

end waiter_total_customers_l178_178264


namespace Q_current_age_l178_178175

-- Definitions for the current ages of P and Q
variable (P Q : ℕ)

-- Conditions
-- 1. P + Q = 100
-- 2. P = 3 * (Q - (P - Q))  (from P is thrice as old as Q was when P was as old as Q is now)

axiom age_sum : P + Q = 100
axiom age_relation : P = 3 * (Q - (P - Q))

theorem Q_current_age : Q = 40 :=
by
  sorry

end Q_current_age_l178_178175


namespace a2_a3_equals_20_l178_178296

-- Sequence definition
def a_n (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

-- Proof that a_2 * a_3 = 20
theorem a2_a3_equals_20 :
  a_n 2 * a_n 3 = 20 :=
by
  sorry

end a2_a3_equals_20_l178_178296


namespace num_solutions_x_squared_minus_y_squared_eq_2001_l178_178331

theorem num_solutions_x_squared_minus_y_squared_eq_2001 
  (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2 - y^2 = 2001 ↔ (x, y) = (1001, 1000) ∨ (x, y) = (335, 332) := sorry

end num_solutions_x_squared_minus_y_squared_eq_2001_l178_178331


namespace remainder_17_pow_77_mod_7_l178_178912

theorem remainder_17_pow_77_mod_7 : (17^77) % 7 = 5 := 
by sorry

end remainder_17_pow_77_mod_7_l178_178912


namespace man_speed_in_still_water_l178_178072

noncomputable def speed_of_man_in_still_water (vm vs : ℝ) : Prop :=
  -- Condition 1: v_m + v_s = 8
  vm + vs = 8 ∧
  -- Condition 2: v_m - v_s = 5
  vm - vs = 5

-- Proving the speed of the man in still water is 6.5 km/h
theorem man_speed_in_still_water : ∃ (v_m : ℝ), (∃ v_s : ℝ, speed_of_man_in_still_water v_m v_s) ∧ v_m = 6.5 :=
by
  sorry

end man_speed_in_still_water_l178_178072


namespace fraction_of_l178_178551

theorem fraction_of (a b : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) : (a / b) = 3/5 :=
by sorry

end fraction_of_l178_178551


namespace radian_measure_of_negative_150_degree_l178_178052

theorem radian_measure_of_negative_150_degree  : (-150 : ℝ) * (Real.pi / 180) = - (5 * Real.pi / 6) := by
  sorry

end radian_measure_of_negative_150_degree_l178_178052


namespace max_value_y_l178_178301

theorem max_value_y (x : ℝ) (h : x < -1) : x + 1/(x + 1) ≤ -3 :=
by sorry

end max_value_y_l178_178301


namespace angle_C_in_triangle_l178_178021

theorem angle_C_in_triangle (A B C : ℝ)
  (hA : A = 60)
  (hAC : C = 2 * B)
  (hSum : A + B + C = 180) : C = 80 :=
sorry

end angle_C_in_triangle_l178_178021


namespace lines_in_4_by_4_grid_l178_178677

-- Definition for the grid and the number of lattice points.
def grid : Nat := 16

-- Theorem stating that the number of different lines passing through at least two points in a 4-by-4 grid of lattice points.
theorem lines_in_4_by_4_grid : 
  (number_of_lines : Nat) → number_of_lines = 40 ↔ grid = 16 := 
by
  -- Calculating number of lines passing through at least two points in a 4-by-4 grid.
  sorry -- proof skipped

end lines_in_4_by_4_grid_l178_178677


namespace hyperbola_with_foci_condition_l178_178147

theorem hyperbola_with_foci_condition (k : ℝ) :
  ( ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 → ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 ∧ (k + 3 > 0 ∧ k + 2 < 0) ) ↔ (-3 < k ∧ k < -2) :=
sorry

end hyperbola_with_foci_condition_l178_178147


namespace max_sum_seq_l178_178019

theorem max_sum_seq (a : ℕ → ℝ) (h1 : a 1 = 0)
  (h2 : abs (a 2) = abs (a 1 - 1)) 
  (h3 : abs (a 3) = abs (a 2 - 1)) 
  (h4 : abs (a 4) = abs (a 3 - 1)) 
  : ∃ M, (∀ (b : ℕ → ℝ), b 1 = 0 → abs (b 2) = abs (b 1 - 1) → abs (b 3) = abs (b 2 - 1) → abs (b 4) = abs (b 3 - 1) → (b 1 + b 2 + b 3 + b 4) ≤ M) 
    ∧ (a 1 + a 2 + a 3 + a 4 = M) :=
  sorry

end max_sum_seq_l178_178019


namespace find_x_l178_178113

-- Define the percentages and multipliers as constants
def percent_47 := 47.0 / 100.0
def percent_36 := 36.0 / 100.0

-- Define the given quantities
def quantity1 := 1442.0
def quantity2 := 1412.0

-- Calculate the percentages of the quantities
def part1 := percent_47 * quantity1
def part2 := percent_36 * quantity2

-- Calculate the expression
def expression := (part1 - part2) + 63.0

-- Define the value of x given
def x := 232.42

-- Theorem stating the proof problem
theorem find_x : expression = x := by
  -- proof goes here
  sorry

end find_x_l178_178113


namespace complement_supplement_measure_l178_178506

theorem complement_supplement_measure (x : ℝ) (h : 180 - x = 3 * (90 - x)) : 
  (180 - x = 135) ∧ (90 - x = 45) :=
by {
  sorry
}

end complement_supplement_measure_l178_178506


namespace last_digit_two_power_2015_l178_178172

/-- The last digit of powers of 2 cycles through 2, 4, 8, 6. Therefore, the last digit of 2^2015 is the same as 2^3, which is 8. -/
theorem last_digit_two_power_2015 : (2^2015) % 10 = 8 :=
by sorry

end last_digit_two_power_2015_l178_178172


namespace first_dimension_length_l178_178930

-- Definitions for conditions
def tank_surface_area (x : ℝ) : ℝ := 14 * x + 20
def cost_per_sqft : ℝ := 20
def total_cost (x : ℝ) : ℝ := (tank_surface_area x) * cost_per_sqft

-- The theorem we need to prove
theorem first_dimension_length : ∃ x : ℝ, total_cost x = 1520 ∧ x = 4 := by 
  sorry

end first_dimension_length_l178_178930


namespace solution_is_correct_l178_178076

noncomputable def solve_system_of_inequalities : Prop :=
  ∃ x y : ℝ, 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧ 
    (x = -1/3) ∧ 
    (y = 2/3)

theorem solution_is_correct : solve_system_of_inequalities :=
sorry

end solution_is_correct_l178_178076


namespace jillian_apartment_size_l178_178592

theorem jillian_apartment_size :
  ∃ (s : ℝ), (1.20 * s = 720) ∧ s = 600 := by
sorry

end jillian_apartment_size_l178_178592


namespace martha_profit_l178_178337

theorem martha_profit :
  let loaves_baked := 60
  let cost_per_loaf := 1
  let morning_price := 3
  let afternoon_price := 3 * 0.75
  let evening_price := 2
  let morning_loaves := loaves_baked / 3
  let afternoon_loaves := (loaves_baked - morning_loaves) / 2
  let evening_loaves := loaves_baked - morning_loaves - afternoon_loaves
  let morning_revenue := morning_loaves * morning_price
  let afternoon_revenue := afternoon_loaves * afternoon_price
  let evening_revenue := evening_loaves * evening_price
  let total_revenue := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost := loaves_baked * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 85 := 
by
  sorry

end martha_profit_l178_178337


namespace arithmetic_sequence_problem_l178_178831

variable (n : ℕ) (a S : ℕ → ℕ)

theorem arithmetic_sequence_problem
  (h1 : a 2 + a 8 = 82)
  (h2 : S 41 = S 9)
  (hSn : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n, a n = 51 - 2 * n) ∧ (∀ n, S n ≤ 625) := sorry

end arithmetic_sequence_problem_l178_178831


namespace incorrect_calculation_l178_178376

theorem incorrect_calculation :
    (5 / 8 + (-7 / 12) ≠ -1 / 24) :=
by
  sorry

end incorrect_calculation_l178_178376


namespace isosceles_right_triangle_area_l178_178203

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (A : ℝ)
  (h_def : h = 6 * Real.sqrt 2)
  (rel_leg_hypotenuse : h = Real.sqrt 2 * l)
  (area_def : A = 1 / 2 * l * l) :
  A = 18 :=
by
  sorry

end isosceles_right_triangle_area_l178_178203


namespace find_k_l178_178726

noncomputable def geometric_series_sum (k : ℝ) (h : k > 1) : ℝ :=
  ∑' n, ((7 * n - 2) / k ^ n)

theorem find_k (k : ℝ) (h : k > 1)
  (series_sum : geometric_series_sum k h = 18 / 5) :
  k = 3.42 :=
by
  sorry

end find_k_l178_178726


namespace sqrt_neg_square_real_l178_178267

theorem sqrt_neg_square_real : ∃! (x : ℝ), -(x + 2) ^ 2 = 0 := by
  sorry

end sqrt_neg_square_real_l178_178267


namespace smallest_t_for_sin_theta_circle_l178_178763

theorem smallest_t_for_sin_theta_circle : 
  ∃ t, (∀ θ, 0 ≤ θ ∧ θ ≤ t → (let r := Real.sin θ in (r * Real.cos θ, r * Real.sin θ))) = (λ θ, (Real.cos θ, Real.sin θ)) ∧ 
        (∀ t', (∀ θ, 0 ≤ θ ∧ θ ≤ t' → (let r := Real.sin θ in (r * Real.cos θ, r * Real.sin θ))) = (λ θ, (Real.cos θ, Real.sin θ)) → t' ≥ t)) ∧ t = Real.pi := 
    by sorry

end smallest_t_for_sin_theta_circle_l178_178763


namespace kekai_ratio_l178_178024

/-
Kekai sells 5 shirts at $1 each,
5 pairs of pants at $3 each,
and he has $10 left after giving some money to his parents.
Our goal is to prove the ratio of the money Kekai gives to his parents
to the total money he earns from selling his clothes is 1:2.
-/

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def shirt_price : ℕ := 1
def pants_price : ℕ := 3
def money_left : ℕ := 10

def total_earnings : ℕ := (shirts_sold * shirt_price) + (pants_sold * pants_price)
def money_given_to_parents : ℕ := total_earnings - money_left
def ratio (a b : ℕ) := (a / Nat.gcd a b, b / Nat.gcd a b)

theorem kekai_ratio : ratio money_given_to_parents total_earnings = (1, 2) :=
  by
    sorry

end kekai_ratio_l178_178024


namespace factorization_l178_178134

open Polynomial

noncomputable def expr (a b c : ℤ) : ℤ := a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3)

noncomputable def p (a b c : ℤ) : ℤ := a * b^3 + a * c^3 + a * b * c^2 + b^2 * c^2

theorem factorization (a b c : ℤ) : 
  expr a b c = (a - b) * (b - c) * (c - a) * p a b c :=
by sorry

end factorization_l178_178134


namespace committee_selection_l178_178468

-- Definitions based on the conditions
def num_people := 12
def num_women := 7
def num_men := 5
def committee_size := 5
def min_women := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Required number of ways to form the committee
def num_ways_5_person_committee_with_at_least_2_women : ℕ :=
  binom num_women min_women * binom (num_people - min_women) (committee_size - min_women)

-- Statement to be proven
theorem committee_selection : num_ways_5_person_committee_with_at_least_2_women = 2520 :=
by
  sorry

end committee_selection_l178_178468


namespace proj_w_v_is_v_l178_178603

noncomputable def proj_w_v (v w : ℝ × ℝ) : ℝ × ℝ :=
  let c := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (c * w.1, c * w.2)

def v : ℝ × ℝ := (-3, 2)
def w : ℝ × ℝ := (4, -2)

theorem proj_w_v_is_v : proj_w_v v w = v := 
  sorry

end proj_w_v_is_v_l178_178603


namespace divide_subtract_multiply_l178_178037

theorem divide_subtract_multiply :
  (-5) / ((1/4) - (1/3)) * 12 = 720 := by
  sorry

end divide_subtract_multiply_l178_178037


namespace find_a_l178_178622

open Set

theorem find_a (A : Set ℝ) (B : Set ℝ) (f : ℝ → ℝ) (a : ℝ)
  (hA : A = Ici 0) 
  (hB : B = univ)
  (hf : ∀ x ∈ A, f x = 2^x - 1) 
  (ha_in_A : a ∈ A) 
  (ha_f_eq_3 : f a = 3) :
  a = 2 := 
by
  sorry

end find_a_l178_178622


namespace toothpicks_in_stage_200_l178_178761

def initial_toothpicks : ℕ := 6
def toothpicks_per_stage : ℕ := 5
def stage_number : ℕ := 200

theorem toothpicks_in_stage_200 :
  initial_toothpicks + (stage_number - 1) * toothpicks_per_stage = 1001 := by
  sorry

end toothpicks_in_stage_200_l178_178761


namespace johannes_sells_48_kg_l178_178478

-- Define Johannes' earnings
def earnings_wednesday : ℕ := 30
def earnings_friday : ℕ := 24
def earnings_today : ℕ := 42

-- Price per kilogram of cabbage
def price_per_kg : ℕ := 2

-- Prove that the total kilograms of cabbage sold is 48
theorem johannes_sells_48_kg :
  ((earnings_wednesday + earnings_friday + earnings_today) / price_per_kg) = 48 := by
  sorry

end johannes_sells_48_kg_l178_178478


namespace least_number_remainder_l178_178213

theorem least_number_remainder (n : ℕ) :
  (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4) → n = 256 :=
by
  sorry

end least_number_remainder_l178_178213


namespace theo_cookie_price_l178_178820

theorem theo_cookie_price :
  (∃ (dough_amount total_earnings per_cookie_earnings_carla per_cookie_earnings_theo : ℕ) 
     (cookies_carla cookies_theo : ℝ), 
  dough_amount = 120 ∧ 
  cookies_carla = 20 ∧ 
  per_cookie_earnings_carla = 50 ∧ 
  cookies_theo = 15 ∧ 
  total_earnings = cookies_carla * per_cookie_earnings_carla ∧ 
  per_cookie_earnings_theo = total_earnings / cookies_theo ∧ 
  per_cookie_earnings_theo = 67) :=
sorry

end theo_cookie_price_l178_178820


namespace arithmetic_sequence_properties_l178_178162

variables {a : ℕ → ℤ} {S T : ℕ → ℤ}

theorem arithmetic_sequence_properties 
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1)) / 2 * (a 2 - a 1)) -- Sum of first n terms of arithmetic sequence
  (h₄ : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1)) -- General term formula of arithmetic sequence
  : (∀ n, a n = -2 * n + 15) ∧
    ( (∀ n, 1 ≤ n ∧ n ≤ 7 → T n = -n^2 + 14 * n) ∧ 
      (∀ n, n ≥ 8 → T n = n^2 - 14 * n + 98)) :=
by
sorry

end arithmetic_sequence_properties_l178_178162


namespace asia_fraction_correct_l178_178146

-- Define the problem conditions
def fraction_NA (P : ℕ) : ℚ := 1/3 * P
def fraction_Europe (P : ℕ) : ℚ := 1/8 * P
def fraction_Africa (P : ℕ) : ℚ := 1/5 * P
def others : ℕ := 42
def total_passengers : ℕ := 240

-- Define the target fraction for Asia
def fraction_Asia (P: ℕ) : ℚ := 17 / 120

-- Theorem: the fraction of the passengers from Asia equals 17/120
theorem asia_fraction_correct : ∀ (P : ℕ), 
  P = total_passengers →
  fraction_NA P + fraction_Europe P + fraction_Africa P + fraction_Asia P * P + others = P →
  fraction_Asia P = 17 / 120 := 
by sorry

end asia_fraction_correct_l178_178146


namespace hectors_sibling_product_l178_178142

theorem hectors_sibling_product (sisters : Nat) (brothers : Nat) (helen : Nat -> Prop): 
  (helen 4) → (helen 7) → (helen 5) → (helen 6) →
  (sisters + 1 = 5) → (brothers + 1 = 7) → ((sisters * brothers) = 30) :=
by
  sorry

end hectors_sibling_product_l178_178142


namespace problem_a_l178_178383

theorem problem_a (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  Int.floor (5 * x) + Int.floor (5 * y) ≥ Int.floor (3 * x + y) + Int.floor (3 * y + x) :=
sorry

end problem_a_l178_178383


namespace find_circle_radius_l178_178816

noncomputable def circle_radius (x y : ℝ) : ℝ :=
  (x - 1) ^ 2 + (y + 2) ^ 2

theorem find_circle_radius :
  (∀ x y : ℝ, 25 * x^2 - 50 * x + 25 * y^2 + 100 * y + 125 = 0 → circle_radius x y = 0) → radius = 0 :=
sorry

end find_circle_radius_l178_178816


namespace select_student_D_l178_178367

-- Define the scores and variances based on the conditions
def avg_A : ℝ := 96
def avg_B : ℝ := 94
def avg_C : ℝ := 93
def avg_D : ℝ := 96

def var_A : ℝ := 1.2
def var_B : ℝ := 1.2
def var_C : ℝ := 0.6
def var_D : ℝ := 0.4

-- Proof statement in Lean 4
theorem select_student_D (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ) 
                         (h_avg_A : avg_A = 96)
                         (h_avg_B : avg_B = 94)
                         (h_avg_C : avg_C = 93)
                         (h_avg_D : avg_D = 96)
                         (h_var_A : var_A = 1.2)
                         (h_var_B : var_B = 1.2)
                         (h_var_C : var_C = 0.6)
                         (h_var_D : var_D = 0.4) 
                         (h_D_highest_avg : avg_D = max avg_A (max avg_B (max avg_C avg_D)))
                         (h_D_lowest_var : var_D = min (min (min var_A var_B) var_C) var_D) :
  avg_D = 96 ∧ var_D = 0.4 := 
by 
  -- As we're not asked to prove, we put sorry here to indicate the proof step is omitted.
  sorry

end select_student_D_l178_178367


namespace exists_two_people_with_property_l178_178493

theorem exists_two_people_with_property (n : ℕ) (P : Fin (2 * n + 2) → Fin (2 * n + 2) → Prop) :
  ∃ A B : Fin (2 * n + 2), 
    A ≠ B ∧
    (∃ S : Finset (Fin (2 * n + 2)), 
      S.card = n ∧
      ∀ C ∈ S, (P C A ∧ P C B) ∨ (¬P C A ∧ ¬P C B)) :=
sorry

end exists_two_people_with_property_l178_178493


namespace smallest_sum_of_distinct_integers_l178_178458

theorem smallest_sum_of_distinct_integers (x y : ℕ) (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  x + y = 98 :=
sorry

end smallest_sum_of_distinct_integers_l178_178458


namespace lines_in_4_by_4_grid_l178_178683

theorem lines_in_4_by_4_grid : 
  let n := 4 in
  number_of_lines_at_least_two_points (grid_of_lattice_points n) = 96 :=
by sorry

end lines_in_4_by_4_grid_l178_178683


namespace imaginary_unit_problem_l178_178446

theorem imaginary_unit_problem (i : ℂ) (h : i^2 = -1) :
  ( (1 + i) / i )^2014 = 2^(1007 : ℤ) * i :=
by sorry

end imaginary_unit_problem_l178_178446


namespace compute_cos_2_sum_zero_l178_178730

theorem compute_cos_2_sum_zero (x y z : ℝ)
  (h1 : Real.cos (x + Real.pi / 4) + Real.cos (y + Real.pi / 4) + Real.cos (z + Real.pi / 4) = 0)
  (h2 : Real.sin (x + Real.pi / 4) + Real.sin (y + Real.pi / 4) + Real.sin (z + Real.pi / 4) = 0) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 :=
by
  sorry

end compute_cos_2_sum_zero_l178_178730


namespace sum_x_y_z_l178_178482

theorem sum_x_y_z (a b : ℝ) (x y z : ℕ) 
  (h_a : a^2 = 16 / 44) 
  (h_b : b^2 = (2 + Real.sqrt 5)^2 / 11) 
  (h_a_neg : a < 0) 
  (h_b_pos : b > 0) 
  (h_expr : (a + b)^3 = x * Real.sqrt y / z) : 
  x + y + z = 181 := 
sorry

end sum_x_y_z_l178_178482


namespace colored_sectors_overlap_l178_178271

/--
Given two disks each divided into 1985 equal sectors, with 200 sectors on each disk colored arbitrarily,
and one disk is rotated by angles that are multiples of 360 degrees / 1985, 
prove that there are at least 80 positions where no more than 20 colored sectors coincide.
-/
theorem colored_sectors_overlap :
  ∀ (disks : ℕ → ℕ) (sectors_colored : ℕ),
  disks 1 = 1985 → disks 2 = 1985 →
  sectors_colored = 200 →
  ∃ (p : ℕ), p ≥ 80 ∧ (∀ (i : ℕ), (i < p → sectors_colored ≤ 20)) := 
sorry

end colored_sectors_overlap_l178_178271


namespace max_interval_length_l178_178955

def m (x : ℝ) : ℝ := x^2 - 3 * x + 4
def n (x : ℝ) : ℝ := 2 * x - 3

def are_close_functions (m n : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |m x - n x| ≤ 1

theorem max_interval_length
  (h : are_close_functions m n 2 3) :
  3 - 2 = 1 :=
sorry

end max_interval_length_l178_178955


namespace alice_bob_meet_l178_178262

theorem alice_bob_meet (t : ℝ) 
(h1 : ∀ s : ℝ, s = 30 * t) 
(h2 : ∀ b : ℝ, b = 29.5 * 60 ∨ b = 30.5 * 60)
(h3 : ∀ a : ℝ, a = 30 * t)
(h4 : ∀ a b : ℝ, a = b):
(t = 59 ∨ t = 61) :=
by
  sorry

end alice_bob_meet_l178_178262


namespace probability_remainder_is_4_5_l178_178938

def probability_remainder_1 (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1)

theorem probability_remainder_is_4_5 : 
  ∀ N, N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1) → (number_of_successful_outcomes / total_outcomes = 4 / 5) :=
sorry

end probability_remainder_is_4_5_l178_178938


namespace no_non_similar_triangles_with_geometric_angles_l178_178004

theorem no_non_similar_triangles_with_geometric_angles :
  ¬∃ (a r : ℕ), a > 0 ∧ r > 0 ∧ a ≠ a * r ∧ a ≠ a * r * r ∧ a * r ≠ a * r * r ∧
  a + a * r + a * r * r = 180 :=
by
  sorry

end no_non_similar_triangles_with_geometric_angles_l178_178004


namespace complete_the_square_l178_178547

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 11 ∧ a = -4) ↔ (x ^ 2 - 8 * x + 5 = 0) :=
by
  sorry

end complete_the_square_l178_178547


namespace initial_number_of_persons_l178_178505

/-- The average weight of some persons increases by 3 kg when a new person comes in place of one of them weighing 65 kg. 
    The weight of the new person might be 89 kg.
    Prove that the number of persons initially was 8.
-/
theorem initial_number_of_persons (n : ℕ) (h1 : (89 - 65 = 3 * n)) : n = 8 := by
  sorry

end initial_number_of_persons_l178_178505


namespace average_chem_math_l178_178779

theorem average_chem_math (P C M : ℕ) (h : P + C + M = P + 180) : (C + M) / 2 = 90 :=
  sorry

end average_chem_math_l178_178779


namespace inverse_sum_l178_178724

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum : g⁻¹ 8 + g⁻¹ (-64) = -2 :=
by
  -- proof steps will go here
  sorry

end inverse_sum_l178_178724


namespace route_a_faster_by_8_minutes_l178_178738

theorem route_a_faster_by_8_minutes :
  let route_a_distance := 8 -- miles
  let route_a_speed := 40 -- miles per hour
  let route_b_distance := 9 -- miles
  let route_b_speed := 45 -- miles per hour
  let route_b_stop := 8 -- minutes
  let time_route_a := route_a_distance / route_a_speed * 60 -- time in minutes
  let time_route_b := (route_b_distance / route_b_speed) * 60 + route_b_stop -- time in minutes
  time_route_b - time_route_a = 8 :=
by
  sorry

end route_a_faster_by_8_minutes_l178_178738


namespace geometric_sequence_a4_l178_178054

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * q)
    (h_a2 : a 2 = 1)
    (h_q : q = 2) : 
    a 4 = 4 :=
by
  -- Skip the proof as instructed
  sorry

end geometric_sequence_a4_l178_178054


namespace part1_part2_l178_178118

open Complex

def equation (a z : ℂ) : Prop := z^2 - (a + I) * z - (I + 2) = 0

theorem part1 (m : ℝ) (a : ℝ) : equation a m → a = 1 := by
  sorry

theorem part2 (a : ℝ) : ¬ ∃ n : ℝ, equation a (n * I) := by
  sorry

end part1_part2_l178_178118


namespace arithmetic_sequence_sum_l178_178959

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (k : ℕ) :
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * n) →
  S (k + 2) - S k = 24 →
  k = 5 :=
by
  intros a1 ha hS hSk
  sorry

end arithmetic_sequence_sum_l178_178959


namespace option_b_correct_l178_178591

theorem option_b_correct : (-(-2)) = abs (-2) := by
  sorry

end option_b_correct_l178_178591


namespace sum_of_cubes_l178_178236

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l178_178236


namespace probability_three_red_balls_l178_178247

theorem probability_three_red_balls :
  (7 / 21) * (6 / 20) * (5 / 19) = 1 / 38 := by
  sorry

end probability_three_red_balls_l178_178247


namespace maxValue_a1_l178_178319

variable (a_1 q : ℝ)

def isGeometricSequence (a_1 q : ℝ) : Prop :=
  a_1 ≥ 1 ∧ a_1 * q ≤ 2 ∧ a_1 * q^2 ≥ 3

theorem maxValue_a1 (h : isGeometricSequence a_1 q) : a_1 ≤ 4 / 3 := 
sorry

end maxValue_a1_l178_178319


namespace min_sum_ab_l178_178999

theorem min_sum_ab (a b : ℤ) (hab : a * b = 72) : a + b ≥ -17 := by
  sorry

end min_sum_ab_l178_178999


namespace hilt_books_difference_l178_178169

noncomputable def original_price : ℝ := 11
noncomputable def discount_rate : ℝ := 0.20
noncomputable def discount_price (price : ℝ) (rate : ℝ) : ℝ := price * (1 - rate)
noncomputable def quantity : ℕ := 15
noncomputable def sale_price : ℝ := 25
noncomputable def tax_rate : ℝ := 0.10
noncomputable def price_with_tax (price : ℝ) (rate : ℝ) : ℝ := price * (1 + rate)

noncomputable def total_cost : ℝ := discount_price original_price discount_rate * quantity
noncomputable def total_revenue : ℝ := price_with_tax sale_price tax_rate * quantity
noncomputable def profit : ℝ := total_revenue - total_cost

theorem hilt_books_difference : profit = 280.50 :=
by
  sorry

end hilt_books_difference_l178_178169


namespace alpha_plus_beta_l178_178965

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem alpha_plus_beta (α β : ℝ) (hα : 0 ≤ α) (hαβ : α < Real.pi) (hβ : 0 ≤ β) (hββ : β < Real.pi)
  (hα_neq_β : α ≠ β) (hf_α : f α = 1 / 2) (hf_β : f β = 1 / 2) : α + β = (7 * Real.pi) / 6 :=
by
  sorry

end alpha_plus_beta_l178_178965


namespace intersection_of_M_and_N_l178_178455

-- Definitions from conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof problem statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
sorry

end intersection_of_M_and_N_l178_178455


namespace max_squares_overlap_l178_178394

-- Definitions based on conditions.
def side_length_checkerboard_square : ℝ := 0.75
def side_length_card : ℝ := 2
def minimum_overlap : ℝ := 0.25

-- Main theorem to prove.
theorem max_squares_overlap :
  ∃ max_overlap_squares : ℕ, max_overlap_squares = 9 :=
by
  sorry

end max_squares_overlap_l178_178394


namespace x_intercept_correct_l178_178951

noncomputable def x_intercept_of_line : ℝ × ℝ :=
if h : (-4 : ℝ) ≠ 0 then (24 / (-4), 0) else (0, 0)

theorem x_intercept_correct : x_intercept_of_line = (-6, 0) := by
  -- proof will be given here
  sorry

end x_intercept_correct_l178_178951


namespace sum_of_cubes_l178_178225

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l178_178225


namespace simplify_expression_l178_178497

theorem simplify_expression (x : ℝ) : (3 * x + 8) + (50 * x + 25) = 53 * x + 33 := 
by sorry

end simplify_expression_l178_178497


namespace estate_area_correct_l178_178929

-- Define the basic parameters given in the problem
def scale : ℝ := 500  -- 500 miles per inch
def width_on_map : ℝ := 5  -- 5 inches
def height_on_map : ℝ := 3  -- 3 inches

-- Define actual dimensions based on the scale
def actual_width : ℝ := width_on_map * scale  -- actual width in miles
def actual_height : ℝ := height_on_map * scale  -- actual height in miles

-- Define the expected actual area of the estate
def actual_area : ℝ := 3750000  -- actual area in square miles

-- The main theorem to prove
theorem estate_area_correct :
  (actual_width * actual_height) = actual_area := by
  sorry

end estate_area_correct_l178_178929


namespace tan_theta_correct_l178_178008

noncomputable def tan_theta : Real :=
  let θ : Real := sorry
  if h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4) then
    if h : Real.sin θ + Real.cos θ = 17 / 13 then
      Real.tan θ
    else
      0
  else
    0

theorem tan_theta_correct {θ : Real} (h_θ : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 := sorry

end tan_theta_correct_l178_178008


namespace gambler_win_percentage_l178_178251

theorem gambler_win_percentage :
  ∀ (T W play_extra : ℕ) (P_win_extra P_week P_current P_required : ℚ),
    T = 40 →
    P_win_extra = 0.80 →
    play_extra = 40 →
    P_week = 0.60 →
    P_required = 48 →
    (W + P_win_extra * play_extra = P_required) →
    (P_current = (W : ℚ) / T * 100) →
    P_current = 40 :=
by
  intros T W play_extra P_win_extra P_week P_current P_required h1 h2 h3 h4 h5 h6 h7
  sorry

end gambler_win_percentage_l178_178251


namespace gumball_cost_l178_178736

theorem gumball_cost (n : ℕ) (T : ℕ) (h₁ : n = 4) (h₂ : T = 32) : T / n = 8 := by
  sorry

end gumball_cost_l178_178736


namespace find_second_speed_l178_178699

theorem find_second_speed (d t_b : ℝ) (v1 : ℝ) (t_m t_a : ℤ): 
  d = 13.5 ∧ v1 = 5 ∧ t_m = 12 ∧ t_a = 15 →
  (t_b = (d / v1) - (t_m / 60)) →
  (t2 = t_b - (t_a / 60)) →
  v = d / t2 →
  v = 6 :=
by
  sorry

end find_second_speed_l178_178699


namespace sum_of_arithmetic_sequence_l178_178470

variable (S : ℕ → ℝ)

def arithmetic_seq_property (S : ℕ → ℝ) : Prop :=
  S 4 = 4 ∧ S 8 = 12

theorem sum_of_arithmetic_sequence (h : arithmetic_seq_property S) : S 12 = 24 :=
by
  sorry

end sum_of_arithmetic_sequence_l178_178470


namespace smallest_positive_angle_same_terminal_side_l178_178055

theorem smallest_positive_angle_same_terminal_side : 
  ∃ k : ℤ, (∃ α : ℝ, α > 0 ∧ α = -660 + k * 360) ∧ (∀ β : ℝ, β > 0 ∧ β = -660 + k * 360 → β ≥ α) :=
sorry

end smallest_positive_angle_same_terminal_side_l178_178055


namespace budget_for_bulbs_l178_178071

theorem budget_for_bulbs (num_crocus_bulbs : ℕ) (cost_per_crocus : ℝ) (budget : ℝ)
  (h1 : num_crocus_bulbs = 22)
  (h2 : cost_per_crocus = 0.35)
  (h3 : budget = num_crocus_bulbs * cost_per_crocus) :
  budget = 7.70 :=
sorry

end budget_for_bulbs_l178_178071


namespace sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l178_178822

variable {α : ℝ} (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)

theorem sin_and_tan (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3/5 ∧ Real.tan α = 3/4 :=
sorry

theorem sin_add_pi_over_4_and_tan_2alpha (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)
  (h_sin : Real.sin α = -3/5) (h_tan : Real.tan α = 3/4) :
  Real.sin (α + π/4) = -7 * Real.sqrt 2 / 10 ∧ Real.tan (2 * α) = 24/7 :=
sorry

end sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l178_178822


namespace john_spending_l178_178325

open Nat Real

noncomputable def cost_of_silver (silver_ounce: Real) (silver_price: Real) : Real :=
  silver_ounce * silver_price

noncomputable def quantity_of_gold (silver_ounce: Real): Real :=
  2 * silver_ounce

noncomputable def cost_per_ounce_gold (silver_price: Real) (multiplier: Real): Real :=
  silver_price * multiplier

noncomputable def cost_of_gold (gold_ounce: Real) (gold_price: Real) : Real :=
  gold_ounce * gold_price

noncomputable def total_cost (cost_silver: Real) (cost_gold: Real): Real :=
  cost_silver + cost_gold

theorem john_spending :
  let silver_ounce := 1.5
  let silver_price := 20
  let gold_multiplier := 50
  let cost_silver := cost_of_silver silver_ounce silver_price
  let gold_ounce := quantity_of_gold silver_ounce
  let gold_price := cost_per_ounce_gold silver_price gold_multiplier
  let cost_gold := cost_of_gold gold_ounce gold_price
  let total := total_cost cost_silver cost_gold
  total = 3030 :=
by
  sorry

end john_spending_l178_178325


namespace TV_cost_is_1700_l178_178327

def hourlyRate : ℝ := 10
def workHoursPerWeek : ℝ := 30
def weeksPerMonth : ℝ := 4
def additionalHours : ℝ := 50

def weeklyEarnings : ℝ := hourlyRate * workHoursPerWeek
def monthlyEarnings : ℝ := weeklyEarnings * weeksPerMonth
def additionalEarnings : ℝ := hourlyRate * additionalHours

def TVCost : ℝ := monthlyEarnings + additionalEarnings

theorem TV_cost_is_1700 : TVCost = 1700 := sorry

end TV_cost_is_1700_l178_178327


namespace original_number_of_friends_l178_178401

theorem original_number_of_friends (F : ℕ) (h₁ : 5000 / F - 125 = 5000 / (F + 8)) : F = 16 :=
sorry

end original_number_of_friends_l178_178401


namespace geometric_series_problem_l178_178028

noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_problem
  (c d : ℝ)
  (h : geometric_series_sum (c/d) (1/d) = 6) :
  geometric_series_sum (c/(c + 2 * d)) (1/(c + 2 * d)) = 3 / 4 := by
  sorry

end geometric_series_problem_l178_178028


namespace sum_of_cubes_l178_178224

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l178_178224


namespace running_laps_l178_178577

theorem running_laps (A B : ℕ)
  (h_ratio : ∀ t : ℕ, (A * t) = 5 * (B * t) / 3)
  (h_start : A = 5 ∧ B = 3 ∧ ∀ t : ℕ, (A * t) - (B * t) = 4) :
  (B * 2 = 6) ∧ (A * 2 = 10) :=
by
  sorry

end running_laps_l178_178577


namespace subsets_of_A_value_of_a_l178_178996

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 - a*x + 2 = 0}

theorem subsets_of_A : 
  (A = {1, 2} ∧ (∀ S, S ⊆ A → S = ∅ ∨ S = {1} ∨ S = {2} ∨ S = {1, 2}))  :=
by
  sorry

theorem value_of_a (a : ℝ) (B_non_empty : B a ≠ ∅) (B_subset_A : ∀ x, x ∈ B a → x ∈ A): 
  a = 3 :=
by
  sorry

end subsets_of_A_value_of_a_l178_178996


namespace probability_of_three_specific_suits_l178_178976

noncomputable def probability_at_least_one_from_each_of_three_suits : ℚ :=
  1 - (1 / 4) ^ 5

theorem probability_of_three_specific_suits (hearts clubs diamonds : ℕ) :
  hearts = 0 ∧ clubs = 0 ∧ diamonds = 0 → 
  probability_at_least_one_from_each_of_three_suits = 1023 / 1024 := 
by 
  sorry

end probability_of_three_specific_suits_l178_178976


namespace exists_good_set_l178_178486

variable (M : Set ℕ) [DecidableEq M] [Fintype M]
variable (f : Finset ℕ → ℕ)

theorem exists_good_set :
  ∃ T : Finset ℕ, T.card = 10 ∧ (∀ k ∈ T, f (T.erase k) ≠ k) := by
  sorry

end exists_good_set_l178_178486


namespace area_of_triangle_l178_178768

theorem area_of_triangle (base : ℝ) (height : ℝ) (h_base : base = 3.6) (h_height : height = 2.5 * base) : 
  (base * height) / 2 = 16.2 :=
by {
  sorry
}

end area_of_triangle_l178_178768


namespace find_the_number_l178_178278

variable (x : ℕ)

theorem find_the_number (h : 43 + 3 * x = 58) : x = 5 :=
by 
  sorry

end find_the_number_l178_178278


namespace new_cost_relation_l178_178409

def original_cost (k t b : ℝ) : ℝ :=
  k * (t * b)^4

def new_cost (k t b : ℝ) : ℝ :=
  k * ((2 * b) * (0.75 * t))^4

theorem new_cost_relation (k t b : ℝ) (C : ℝ) 
  (hC : C = original_cost k t b) :
  new_cost k t b = 25.63 * C := sorry

end new_cost_relation_l178_178409


namespace solve_quadratic_inequality_l178_178973

theorem solve_quadratic_inequality (a x : ℝ) (h : a < 1) : 
  x^2 - (a + 1) * x + a < 0 ↔ (a < x ∧ x < 1) :=
by
  sorry

end solve_quadratic_inequality_l178_178973


namespace find_smaller_number_l178_178535

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number_l178_178535


namespace inequality_true_l178_178145

theorem inequality_true (a b : ℝ) (h : a > b) : (2 * a - 1) > (2 * b - 1) :=
by {
  sorry
}

end inequality_true_l178_178145


namespace ticket_price_l178_178340

theorem ticket_price (Olivia_money : ℕ) (Nigel_money : ℕ) (left_money : ℕ) (total_tickets : ℕ)
  (h1 : Olivia_money = 112)
  (h2 : Nigel_money = 139)
  (h3 : left_money = 83)
  (h4 : total_tickets = 6) :
  (Olivia_money + Nigel_money - left_money) / total_tickets = 28 :=
by
  sorry

end ticket_price_l178_178340


namespace value_of_v_3_l178_178594

-- Defining the polynomial
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Given evaluation point
def eval_point : ℝ := -2

-- Horner's method intermediate value v_3
def v_3_using_horner_method (x : ℝ) : ℝ :=
  let V0 := 1
  let V1 := x * V0 - 5
  let V2 := x * V1 + 6
  let V3 := x * V2 -- x^3 term is zero
  V3

-- Statement to prove
theorem value_of_v_3 :
  v_3_using_horner_method eval_point = -40 :=
by 
  -- Proof to be completed later
  sorry

end value_of_v_3_l178_178594


namespace find_f_1002_l178_178053

noncomputable def f : ℕ → ℝ :=
  sorry

theorem find_f_1002 (f : ℕ → ℝ) 
  (h : ∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) :
  f 1002 = 21 :=
sorry

end find_f_1002_l178_178053


namespace second_consecutive_odd_integer_l178_178388

theorem second_consecutive_odd_integer (x : ℤ) 
  (h1 : ∃ x, x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) 
  (h2 : (x + 2) + (x + 4) = x + 17) : 
  (x + 2) = 13 :=
by
  sorry

end second_consecutive_odd_integer_l178_178388


namespace probability_at_most_two_visitors_l178_178797

theorem probability_at_most_two_visitors (p : ℚ) (h : p = 3 / 5) :
  (1 - p ^ 3) = 98 / 125 :=
by
  -- Definition of independent probabilities
  have h1 : p ^ 3 = (3 / 5) ^ 3, by rw h
  -- Compute the left side
  sorry

end probability_at_most_two_visitors_l178_178797


namespace find_y_value_l178_178760

noncomputable def y_value (y : ℝ) :=
  (3 * y)^2 + (7 * y)^2 + (1 / 2) * (3 * y) * (7 * y) = 1200

theorem find_y_value (y : ℝ) (hy : y_value y) : y = 10 :=
by
  sorry

end find_y_value_l178_178760


namespace father_age_when_rachel_is_25_l178_178344

-- Definitions for Rachel's age, Grandfather's age, Mother's age, and Father's age
def rachel_age : ℕ := 12
def grandfather_age : ℕ := 7 * rachel_age
def mother_age : ℕ := grandfather_age / 2
def father_age : ℕ := mother_age + 5
def years_until_rachel_is_25 : ℕ := 25 - rachel_age
def fathers_age_when_rachel_is_25 : ℕ := father_age + years_until_rachel_is_25

-- Theorem to prove that Rachel's father will be 60 years old when Rachel is 25 years old
theorem father_age_when_rachel_is_25 : fathers_age_when_rachel_is_25 = 60 := by
  sorry

end father_age_when_rachel_is_25_l178_178344


namespace find_p_l178_178047

noncomputable def f (p : ℝ) : ℝ := 2 * p - 20

theorem find_p : (f ∘ f ∘ f) p = 6 → p = 18.25 := by
  sorry

end find_p_l178_178047


namespace burger_cost_l178_178588

theorem burger_cost 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 440)
  (h2 : 3 * b + 2 * s = 330) : b = 110 := 
by 
  sorry

end burger_cost_l178_178588


namespace percent_defective_units_shipped_l178_178017

variable (P : Real)
variable (h1 : 0.07 * P = d)
variable (h2 : 0.0035 * P = s)

theorem percent_defective_units_shipped (h1 : 0.07 * P = d) (h2 : 0.0035 * P = s) : 
  (s / d) * 100 = 5 := sorry

end percent_defective_units_shipped_l178_178017


namespace line_form_x_eq_ky_add_b_perpendicular_y_line_form_x_eq_ky_add_b_perpendicular_x_l178_178377

theorem line_form_x_eq_ky_add_b_perpendicular_y {k b : ℝ} : 
  ¬ ∃ c : ℝ, x = c ∧ ∀ y : ℝ, x = k*y + b :=
sorry

theorem line_form_x_eq_ky_add_b_perpendicular_x {b : ℝ} : 
  ∃ k : ℝ, k = 0 ∧ ∀ y : ℝ, x = k*y + b :=
sorry

end line_form_x_eq_ky_add_b_perpendicular_y_line_form_x_eq_ky_add_b_perpendicular_x_l178_178377


namespace total_volume_of_pyramids_l178_178777

theorem total_volume_of_pyramids :
  let base := 40
  let height_base := 20
  let height_pyramid := 30
  let area_base := (1 / 2) * base * height_base
  let volume_pyramid := (1 / 3) * area_base * height_pyramid
  3 * volume_pyramid = 12000 :=
by 
  sorry

end total_volume_of_pyramids_l178_178777


namespace max_chocolate_bars_l178_178718

-- Definitions
def john_money := 2450
def chocolate_bar_cost := 220

-- Theorem statement
theorem max_chocolate_bars : ∃ (x : ℕ), x = 11 ∧ chocolate_bar_cost * x ≤ john_money ∧ (chocolate_bar_cost * (x + 1) > john_money) := 
by 
  -- This is to indicate we're acknowledging that the proof is left as an exercise
  sorry

end max_chocolate_bars_l178_178718


namespace opposite_of_neg5_l178_178050

-- Define the concept of the opposite of a number
def opposite (x : Int) : Int :=
  -x

-- The proof problem: Prove that the opposite of -5 is 5
theorem opposite_of_neg5 : opposite (-5) = 5 :=
by
  sorry

end opposite_of_neg5_l178_178050


namespace total_wheels_l178_178565

theorem total_wheels (bicycles tricycles : ℕ) (wheels_per_bicycle wheels_per_tricycle : ℕ) 
  (h1 : bicycles = 50) (h2 : tricycles = 20) (h3 : wheels_per_bicycle = 2) (h4 : wheels_per_tricycle = 3) : 
  (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160) :=
by
  sorry

end total_wheels_l178_178565


namespace other_acute_angle_right_triangle_l178_178708

theorem other_acute_angle_right_triangle (A : ℝ) (B : ℝ) (C : ℝ) (h₁ : A + B = 90) (h₂ : B = 54) : A = 36 :=
by
  sorry

end other_acute_angle_right_triangle_l178_178708


namespace g_g_g_g_3_l178_178596

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end g_g_g_g_3_l178_178596


namespace find_valid_primes_and_integers_l178_178813

def is_prime (p : ℕ) : Prop := Nat.Prime p

def valid_pair (p x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 2 * p ∧ x^(p-1) ∣ (p-1)^x + 1

theorem find_valid_primes_and_integers (p x : ℕ) (hp : is_prime p) 
  (hx : valid_pair p x) : 
  (p = 2 ∧ x = 1) ∨ 
  (p = 2 ∧ x = 2) ∨ 
  (p = 3 ∧ x = 1) ∨ 
  (p = 3 ∧ x = 3) ∨
  (x = 1) :=
sorry

end find_valid_primes_and_integers_l178_178813


namespace angle_between_hands_at_3_27_l178_178067

noncomputable def minute_hand_angle (m : ℕ) : ℝ :=
  (m / 60.0) * 360.0

noncomputable def hour_hand_angle (h : ℕ) (m : ℕ) : ℝ :=
  ((h + m / 60.0) / 12.0) * 360.0

theorem angle_between_hands_at_3_27 : 
  minute_hand_angle 27 - hour_hand_angle 3 27 = 58.5 :=
by
  rw [minute_hand_angle, hour_hand_angle]
  simp
  sorry

end angle_between_hands_at_3_27_l178_178067


namespace lines_in_4_by_4_grid_l178_178640

theorem lines_in_4_by_4_grid : 
  (count_lines_passing_through_at_least_two_points (4, 4) = 62) :=
sorry

def count_lines_passing_through_at_least_two_points (m n : ℕ) : ℕ :=
  let total_pairs := (m * n) * ((m * n) - 1) / 2
  let overcount_lines := (6 - 1) * 10 + (3 - 1) * 4
  total_pairs - overcount_lines

end lines_in_4_by_4_grid_l178_178640


namespace roots_sum_squares_l178_178266

theorem roots_sum_squares (a b c : ℝ) (h₁ : Polynomial.eval a (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0)
  (h₂ : Polynomial.eval b (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0)
  (h₃ : Polynomial.eval c (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0) :
  a^2 + b^2 + c^2 = -26 / 9 :=
sorry

end roots_sum_squares_l178_178266


namespace find_a7_a8_l178_178016

variable {R : Type*} [LinearOrderedField R]
variable {a : ℕ → R}

-- Conditions
def cond1 : a 1 + a 2 = 40 := sorry
def cond2 : a 3 + a 4 = 60 := sorry

-- Goal 
theorem find_a7_a8 : a 7 + a 8 = 135 := 
by 
  -- provide the actual proof here
  sorry

end find_a7_a8_l178_178016


namespace total_cost_correct_l178_178208

-- Define the conditions
def uber_cost : ℤ := 22
def lyft_additional_cost : ℤ := 3
def taxi_additional_cost : ℤ := 4
def tip_percentage : ℚ := 0.20

-- Define the variables for cost of Lyft and Taxi based on the problem
def lyft_cost : ℤ := uber_cost - lyft_additional_cost
def taxi_cost : ℤ := lyft_cost - taxi_additional_cost

-- Calculate the tip
def tip : ℚ := taxi_cost * tip_percentage

-- Final total cost including the tip
def total_cost : ℚ := taxi_cost + tip

-- The theorem to prove
theorem total_cost_correct :
  total_cost = 18 := by
  sorry

end total_cost_correct_l178_178208


namespace chantel_final_bracelets_count_l178_178432

def bracelets_made_in_first_5_days : ℕ := 5 * 2

def bracelets_after_giving_away_at_school : ℕ := bracelets_made_in_first_5_days - 3

def bracelets_made_in_next_4_days : ℕ := 4 * 3

def total_bracelets_before_soccer_giveaway : ℕ := bracelets_after_giving_away_at_school + bracelets_made_in_next_4_days

def bracelets_after_giving_away_at_soccer : ℕ := total_bracelets_before_soccer_giveaway - 6

theorem chantel_final_bracelets_count : bracelets_after_giving_away_at_soccer = 13 :=
sorry

end chantel_final_bracelets_count_l178_178432


namespace remainder_div_l178_178435

theorem remainder_div (P Q R D Q' R' : ℕ) (h₁ : P = Q * D + R) (h₂ : Q = (D - 1) * Q' + R') (h₃ : D > 1) :
  P % (D * (D - 1)) = D * R' + R := by sorry

end remainder_div_l178_178435


namespace find_number_l178_178303

theorem find_number (x : ℝ) (h : 0.35 * x = 0.50 * x - 24) : x = 160 :=
by
  sorry

end find_number_l178_178303


namespace area_of_right_triangle_with_given_sides_l178_178311

theorem area_of_right_triangle_with_given_sides :
  let f : (ℝ → ℝ) := fun x => x^2 - 7 * x + 12
  let a := 3
  let b := 4
  let c := sqrt 7
  let hypotenuse := max a b
  let leg := min a b
in (hypotenuse = 4 ∧ leg = 3 ∧ (f(3) = 0) ∧ (f(4) = 0) → 
   (∃ (area : ℝ), (area = 6 ∨ area = (3 * sqrt 7) / 2))) :=
by
  intros
  sorry

end area_of_right_triangle_with_given_sides_l178_178311


namespace remainder_of_1234567_div_257_l178_178415

theorem remainder_of_1234567_div_257 : 1234567 % 257 = 123 := by
  sorry

end remainder_of_1234567_div_257_l178_178415


namespace quad_root_l178_178121

theorem quad_root (m : ℝ) (β : ℝ) (root_condition : ∃ α : ℝ, α = -5 ∧ (α + β) * (α * β) = x^2 + m * x - 10) : β = 2 :=
by
  sorry

end quad_root_l178_178121


namespace shaded_region_area_l178_178179

variables (a b : ℕ) 
variable (A : Type) 

def AD := 5
def CD := 2
def semi_major_axis := 6
def semi_minor_axis := 4

noncomputable def area_ellipse := Real.pi * semi_major_axis * semi_minor_axis
noncomputable def area_rectangle := AD * CD
noncomputable def area_shaded_region := area_ellipse - area_rectangle

theorem shaded_region_area : area_shaded_region = 24 * Real.pi - 10 :=
by {
  sorry
}

end shaded_region_area_l178_178179


namespace average_length_of_strings_l178_178341

theorem average_length_of_strings (l1 l2 l3 : ℝ) (hl1 : l1 = 2) (hl2 : l2 = 5) (hl3 : l3 = 3) : 
  (l1 + l2 + l3) / 3 = 10 / 3 :=
by
  rw [hl1, hl2, hl3]
  change (2 + 5 + 3) / 3 = 10 / 3
  sorry

end average_length_of_strings_l178_178341


namespace fraction_identity_l178_178419

theorem fraction_identity :
  ( (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) = (432 / 1105) ) :=
by
  sorry

end fraction_identity_l178_178419


namespace polynomial_sum_l178_178330

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def j (x : ℝ) : ℝ := x^2 - x - 3

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = -3 * x^2 + 11 * x - 15 := by
  sorry

end polynomial_sum_l178_178330


namespace quadrilateral_segments_l178_178258

theorem quadrilateral_segments {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a + b + c + d = 2) (h6 : 1/4 < a) (h7 : a < 1) (h8 : 1/4 < b) (h9 : b < 1)
  (h10 : 1/4 < c) (h11 : c < 1) (h12 : 1/4 < d) (h13 : d < 1) : 
  (a + b > d) ∧ (a + c > d) ∧ (a + d > c) ∧ (b + c > d) ∧ 
  (b + d > c) ∧ (c + d > a) ∧ (a + b + c > d) ∧ (a + b + d > c) ∧
  (a + c + d > b) ∧ (b + c + d > a) :=
sorry

end quadrilateral_segments_l178_178258


namespace simplify_expression_l178_178374

theorem simplify_expression : (245^2 - 225^2) / 20 = 470 := by
  sorry

end simplify_expression_l178_178374


namespace base9_4318_is_base10_3176_l178_178943

def base9_to_base10 (n : Nat) : Nat :=
  let d₀ := (n % 10) * 9^0
  let d₁ := ((n / 10) % 10) * 9^1
  let d₂ := ((n / 100) % 10) * 9^2
  let d₃ := ((n / 1000) % 10) * 9^3
  d₀ + d₁ + d₂ + d₃

theorem base9_4318_is_base10_3176 :
  base9_to_base10 4318 = 3176 :=
by
  sorry

end base9_4318_is_base10_3176_l178_178943


namespace lines_in_4_by_4_grid_l178_178665

/--
In a 4-by-4 grid of lattice points, the number of different lines that pass through at least two points is 30.
-/
theorem lines_in_4_by_4_grid : 
  ∃ lines : ℕ, lines = 30 ∧ (∀ pts : fin 4 × fin 4, ∃ l : Set (fin 4 × fin 4), 
  ∀ p1 p2 : fin 4 × fin 4, p1 ∈ pts → p2 ∈ pts → p1 ≠ p2 → p1 ∈ l ∧ p2 ∈ l) := 
sorry

end lines_in_4_by_4_grid_l178_178665


namespace Shara_shells_total_l178_178181

def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

theorem Shara_shells_total : (initial_shells + (shells_per_day * days) + shells_fourth_day) = 41 := by
  sorry

end Shara_shells_total_l178_178181


namespace cannot_afford_laptop_l178_178871

theorem cannot_afford_laptop (P_0 : ℝ) : 56358 < P_0 * (1.06)^2 :=
by
  sorry

end cannot_afford_laptop_l178_178871


namespace distance_between_cities_l178_178516

theorem distance_between_cities (S : ℕ) 
  (h1 : ∀ x, 0 ≤ x ∧ x ≤ S → x.gcd (S - x) ∈ {1, 3, 13}) : 
  S = 39 :=
sorry

end distance_between_cities_l178_178516


namespace sum_of_cubes_l178_178235

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l178_178235


namespace parabola_focus_distance_l178_178133

noncomputable def parabolic_distance (x y : ℝ) : ℝ :=
  x + x / 2

theorem parabola_focus_distance : 
  (∃ y : ℝ, (1 : ℝ) = (1 / 2) * y^2) → 
  parabolic_distance 1 y = 3 / 2 :=
by 
  intros hy
  obtain ⟨y, hy⟩ := hy
  unfold parabolic_distance
  have hx : 1 = (1 / 2) * y^2 := hy
  sorry

end parabola_focus_distance_l178_178133


namespace base_of_parallelogram_l178_178848

theorem base_of_parallelogram (Area Height : ℕ) (h1 : Area = 44) (h2 : Height = 11) : (Area / Height) = 4 :=
by
  sorry

end base_of_parallelogram_l178_178848


namespace triangle_problem_l178_178480

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : a = b * Real.tan (π / 6))
  (h2 : B > π / 2)
  (h3 : A = π / 6)
  (h4 : A + B + C = π) : 
  (B = 2 * π / 3) ∧ ((sin A + sin C) ∈ Ioo (sqrt 2 / 2) (9 / 8)) :=
by
  sorry

end triangle_problem_l178_178480


namespace time_after_2016_hours_l178_178365

theorem time_after_2016_hours (current_time : ℕ) (h : current_time = 7) : 
  (current_time + 2016) % 12 = 7 :=
by
  rw h
  exact Nat.mod_eq_of_lt (by
    norm_num)
  sorry

end time_after_2016_hours_l178_178365


namespace crossing_time_proof_l178_178391

/-
  Problem:
  Given:
  1. length_train: 600 (length of the train in meters)
  2. time_signal_post: 40 (time taken to cross the signal post in seconds)
  3. time_bridge_minutes: 20 (time taken to cross the bridge in minutes)

  Prove:
  t_cross_bridge: the time it takes to cross the bridge and the full length of the train is 1240 seconds
-/

def length_train : ℕ := 600
def time_signal_post : ℕ := 40
def time_bridge_minutes : ℕ := 20

-- Converting time to cross the bridge from minutes to seconds
def time_bridge_seconds : ℕ := time_bridge_minutes * 60

-- Finding the speed
def speed_train : ℕ := length_train / time_signal_post

-- Finding the length of the bridge
def length_bridge : ℕ := speed_train * time_bridge_seconds

-- Finding the total distance covered
def total_distance : ℕ := length_train + length_bridge

-- Given distance and speed, find the time to cross
def time_to_cross : ℕ := total_distance / speed_train

theorem crossing_time_proof : time_to_cross = 1240 := by
  sorry

end crossing_time_proof_l178_178391


namespace percentage_loss_l178_178399

theorem percentage_loss (CP SP : ℝ) (hCP : CP = 1400) (hSP : SP = 1148) : 
  (CP - SP) / CP * 100 = 18 := by 
  sorry

end percentage_loss_l178_178399


namespace parabola_shifted_left_and_down_l178_178152

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  3 * (x - 4) ^ 2 + 3

-- Define the transformation (shift 4 units to the left and 4 units down)
def transformed_parabola (x : ℝ) : ℝ :=
  initial_parabola (x + 4) - 4

-- Prove that after transformation the given parabola becomes y = 3x^2 - 1
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 3 * x ^ 2 - 1 := 
by 
  sorry

end parabola_shifted_left_and_down_l178_178152


namespace relations_of_sets_l178_178151

open Set

theorem relations_of_sets {A B : Set ℝ} (h : ∃ x ∈ A, x ∉ B) : 
  ¬(A ⊆ B) ∧ ((A ∩ B ≠ ∅) ∨ (B ⊆ A) ∨ (A ∩ B = ∅)) := sorry

end relations_of_sets_l178_178151


namespace log_b_243_values_l178_178006

theorem log_b_243_values : 
  ∃! (s : Finset ℕ), (∀ b ∈ s, ∃ n : ℕ, b^n = 243) ∧ s.card = 2 :=
by 
  sorry

end log_b_243_values_l178_178006


namespace distinct_lines_count_in_4x4_grid_l178_178686

theorem distinct_lines_count_in_4x4_grid :
  let grid_points := (finRange 4).product (finRange 4)
  let lines := {line : Set (ℕ × ℕ) | ∃ (a b : ℤ), ∀ p ∈ line, a * (p.1:ℤ) + b * (p.2:ℤ) = 1}
  let distinct_lines := {line ∈ lines | ∃ (p1 p2 : ℕ × ℕ), p1 ∈ grid_points ∧ p2 ∈ grid_points ∧ p1 ≠ p2 ∧ line = {p | this line passes through p}}
  lines.card = 50 :=
by
  sorry

end distinct_lines_count_in_4x4_grid_l178_178686


namespace math_proof_problem_l178_178867

variable (a d e : ℝ)

theorem math_proof_problem (h1 : a < 0) (h2 : a < d) (h3 : d < e) :
  (a * d < a * e) ∧ (a + d < d + e) ∧ (e / a < 1) :=
by {
  sorry
}

end math_proof_problem_l178_178867


namespace lines_in_4x4_grid_l178_178672

theorem lines_in_4x4_grid : 
  let grid_points := finset.univ.product finset.univ
  let total_points := 16
  let pairs_of_points := total_points.choose 2
  let horizontal_lines := 4
  let vertical_lines := 4
  let diagonal_lines := 2
  let lines_through_four_points := horizontal_lines + vertical_lines + diagonal_lines
  let correction := lines_through_four_points * (4.choose 2 - 1)
  let number_of_lines := pairs_of_points - correction
  in number_of_lines = 70 := 
by {
  sorry
}

end lines_in_4x4_grid_l178_178672


namespace count_lines_in_4x4_grid_l178_178691

theorem count_lines_in_4x4_grid : 
  let grid_points : Fin 4 × Fin 4 := 
  ∃! lines : set (set (Fin 4 × Fin 4)), 
  ∀ line ∈ lines, ∃ (p1 p2 : Fin 4 × Fin 4), p1 ≠ p2 ∧ p1 ∈ line ∧ p2 ∈ line ∧ (grid_points ⊆ line ⊆ grid_points) :=
  lines = 84 :=
sorry

end count_lines_in_4x4_grid_l178_178691


namespace minimum_of_f_value_of_a_l178_178833

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_of_f :
  f (1 / Real.exp 1) = - 1 / Real.exp 1 :=
by sorry

noncomputable def F (x : ℝ) (a : ℝ) : ℝ := (f x - a) / x

theorem value_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 1 (Real.exp 1), F x a ≥ 3 / 2) →
  a = - Real.sqrt (Real.exp 1) :=
by sorry

end minimum_of_f_value_of_a_l178_178833


namespace remainder_of_division_l178_178417

def dividend := 1234567
def divisor := 257

theorem remainder_of_division : dividend % divisor = 774 :=
by
  sorry

end remainder_of_division_l178_178417


namespace R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l178_178888

theorem R_H_nonneg_def (H : ℝ) (s t : ℝ) (hH : 0 < H ∧ H ≤ 1) :
  (1 / 2) * (|t| ^ (2 * H) + |s| ^ (2 * H) - |t - s| ^ (2 * H)) ≥ 0 := sorry

theorem R_K_nonneg_def (K : ℝ) (s t : ℝ) (hK : 0 < K ∧ K ≤ 2) :
  (1 / 2 ^ K) * (|t + s| ^ K - |t - s| ^ K) ≥ 0 := sorry

theorem R_HK_nonneg_def (H K : ℝ) (s t : ℝ) (hHK : 0 < H ∧ H ≤ 1 ∧ 0 < K ∧ K ≤ 1) :
  (1 / 2 ^ K) * ( (|t| ^ (2 * H) + |s| ^ (2 * H)) ^ K - |t - s| ^ (2 * H * K) ) ≥ 0 := sorry

end R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l178_178888


namespace sum_of_three_numbers_l178_178906

theorem sum_of_three_numbers :
  ((3 : ℝ) / 8) + 0.125 + 9.51 = 10.01 :=
sorry

end sum_of_three_numbers_l178_178906


namespace probability_sum_not_less_than_15_l178_178924

open ProbTheory

noncomputable def probability_not_less_than_15 : ℚ :=
let prob_individual : ℚ := 1 / 8 in
let prob_pair_7_8    := prob_individual * prob_individual in
let prob_pair_8_8    := prob_individual * prob_individual in
prob_pair_7_8 + prob_pair_7_8 + prob_pair_8_8

theorem probability_sum_not_less_than_15 :
  probability_not_less_than_15 = 3 / 64 :=
by
  sorry

end probability_sum_not_less_than_15_l178_178924


namespace sequence_a_n_a31_l178_178454

theorem sequence_a_n_a31 (a : ℕ → ℤ) 
  (h_initial : a 1 = 2)
  (h_recurrence : ∀ n : ℕ, a n + a (n + 1) + n^2 = 0) :
  a 31 = -463 :=
sorry

end sequence_a_n_a31_l178_178454


namespace find_a_b_minimum_value_l178_178294

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2

/-- Given the function y = f(x) = ax^3 + bx^2, when x = 1, it has a maximum value of 3 -/
def condition1 (a b : ℝ) : Prop :=
  f 1 a b = 3 ∧ (3 * a + 2 * b = 0)

/-- Find the values of the real numbers a and b -/
theorem find_a_b : ∃ (a b : ℝ), condition1 a b :=
sorry

/-- Find the minimum value of the function -/
theorem minimum_value : ∀ (a b : ℝ), condition1 a b → (∃ x_min, ∀ x, f x a b ≥ f x_min a b) :=
sorry

end find_a_b_minimum_value_l178_178294


namespace first_four_cards_all_red_l178_178789

noncomputable def probability_first_four_red_cards : ℚ :=
  (26 / 52) * (25 / 51) * (24 / 50) * (23 / 49)

theorem first_four_cards_all_red :
  probability_first_four_red_cards = 276 / 9801 :=
by
  -- The proof itself is not required; we are only stating it.
  sorry

end first_four_cards_all_red_l178_178789


namespace XiaoMing_team_award_l178_178257

def points (x : ℕ) : ℕ := 2 * x + (8 - x)

theorem XiaoMing_team_award (x : ℕ) : 2 * x + (8 - x) ≥ 12 := 
by 
  sorry

end XiaoMing_team_award_l178_178257


namespace range_of_3x_minus_y_l178_178956

-- Defining the conditions in Lean
variable (x y : ℝ)

-- Condition 1: -1 ≤ x + y ≤ 1
def cond1 : Prop := -1 ≤ x + y ∧ x + y ≤ 1

-- Condition 2: 1 ≤ x - y ≤ 3
def cond2 : Prop := 1 ≤ x - y ∧ x - y ≤ 3

-- The theorem statement to prove that the range of 3x - y is [1, 7]
theorem range_of_3x_minus_y (h1 : cond1 x y) (h2 : cond2 x y) : 1 ≤ 3 * x - y ∧ 3 * x - y ≤ 7 := by
  sorry

end range_of_3x_minus_y_l178_178956


namespace area_of_fifteen_sided_figure_l178_178581

noncomputable def figure_area : ℝ :=
  let full_squares : ℝ := 6
  let num_triangles : ℝ := 10
  let triangles_to_rectangles : ℝ := num_triangles / 2
  let triangles_area : ℝ := triangles_to_rectangles
  full_squares + triangles_area

theorem area_of_fifteen_sided_figure :
  figure_area = 11 := by
  sorry

end area_of_fifteen_sided_figure_l178_178581


namespace ratio_lions_l178_178743

variable (Safari_Lions : Nat)
variable (Safari_Snakes : Nat)
variable (Safari_Giraffes : Nat)
variable (Savanna_Lions_Ratio : ℕ)
variable (Savanna_Snakes : Nat)
variable (Savanna_Giraffes : Nat)
variable (Savanna_Total : Nat)

-- Conditions
def conditions := 
  (Safari_Lions = 100) ∧
  (Safari_Snakes = Safari_Lions / 2) ∧
  (Safari_Giraffes = Safari_Snakes - 10) ∧
  (Savanna_Lions_Ratio * Safari_Lions + Savanna_Snakes + Savanna_Giraffes = Savanna_Total) ∧
  (Savanna_Snakes = 3 * Safari_Snakes) ∧
  (Savanna_Giraffes = Safari_Giraffes + 20) ∧
  (Savanna_Total = 410)

-- Theorem to prove
theorem ratio_lions : conditions Safari_Lions Safari_Snakes Safari_Giraffes Savanna_Lions_Ratio Savanna_Snakes Savanna_Giraffes Savanna_Total → Savanna_Lions_Ratio = 2 := by
  sorry

end ratio_lions_l178_178743


namespace less_than_subtraction_l178_178553

-- Define the numbers as real numbers
def a : ℝ := 47.2
def b : ℝ := 0.5

-- Theorem statement
theorem less_than_subtraction : a - b = 46.7 :=
by
  sorry

end less_than_subtraction_l178_178553


namespace max_movies_watched_l178_178970

-- Conditions given in the problem
def movie_duration : Nat := 90
def tuesday_minutes : Nat := 4 * 60 + 30
def tuesday_movies : Nat := tuesday_minutes / movie_duration
def wednesday_movies : Nat := 2 * tuesday_movies

-- Problem statement: Total movies watched in two days
theorem max_movies_watched : 
  tuesday_movies + wednesday_movies = 9 := 
by
  -- We add the placeholder for the proof here
  sorry

end max_movies_watched_l178_178970


namespace sarah_books_check_out_l178_178180

theorem sarah_books_check_out
  (words_per_minute : ℕ)
  (words_per_page : ℕ)
  (pages_per_book : ℕ)
  (reading_hours : ℕ)
  (number_of_books : ℕ) :
  words_per_minute = 40 →
  words_per_page = 100 →
  pages_per_book = 80 →
  reading_hours = 20 →
  number_of_books = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sarah_books_check_out_l178_178180


namespace allen_mother_age_l178_178590

variable (A M : ℕ)

theorem allen_mother_age (h1 : A = M - 25) (h2 : (A + 3) + (M + 3) = 41) : M = 30 :=
by
  sorry

end allen_mother_age_l178_178590


namespace num_lines_passing_through_4x4_grid_l178_178649

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 66. -/
theorem num_lines_passing_through_4x4_grid : 
  let p := 4 * 4 in
  let total_point_pairs := p * (p - 1) / 2 in
  let horizontal_lines_count := 4 in
  let vertical_lines_count := 4 in
  let diagonal_lines_4_count := 2 in
  let diagonal_lines_3_count := 2 in
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count in
  (total_point_pairs - overcount_correction) = 66 :=
by
  let p := 4 * 4
  let total_point_pairs := p * (p - 1) / 2
  let horizontal_lines_count := 4
  let vertical_lines_count := 4
  let diagonal_lines_4_count := 2
  let diagonal_lines_3_count := 2
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count
  have h_correct_count : total_point_pairs - overcount_correction = 66, from sorry
  exact h_correct_count

end num_lines_passing_through_4x4_grid_l178_178649


namespace total_cost_proof_l178_178209

def uber_cost : ℤ := 22
def lyft_cost : ℤ := uber_cost - 3
def taxi_cost : ℤ := lyft_cost - 4
def tip : ℤ := (taxi_cost * 20) / 100
def total_cost : ℤ := taxi_cost + tip

theorem total_cost_proof :
  total_cost = 18 :=
by
  sorry

end total_cost_proof_l178_178209


namespace pages_copied_l178_178322

-- Define the assumptions
def cost_per_pages (cent_per_pages: ℕ) : Prop := 
  5 * cent_per_pages = 7 * 1

def total_cents (dollars: ℕ) (cents: ℕ) : Prop :=
  cents = dollars * 100

-- The problem to prove
theorem pages_copied (dollars: ℕ) (cents: ℕ) (cent_per_pages: ℕ) : 
  cost_per_pages cent_per_pages → total_cents dollars cents → dollars = 35 → cents = 3500 → 
  3500 * (5/7 : ℚ) = 2500 :=
by
  sorry

end pages_copied_l178_178322


namespace number_of_points_l178_178868

theorem number_of_points (x y : ℕ) (h : y = (2 * x + 2018) / (x - 1)) 
  (h2 : x > y) (h3 : 0 < x) (h4 : 0 < y) : 
  ∃! (x y : ℕ), y = (2 * x + 2018) / (x - 1) ∧ x > y ∧ 0 < x ∧ 0 < y :=
sorry

end number_of_points_l178_178868


namespace no_non_similar_triangles_with_geometric_angles_l178_178002

theorem no_non_similar_triangles_with_geometric_angles :
  ¬ ∃ (a r : ℤ), 0 < a ∧ 0 < r ∧ a ≠ ar ∧ a ≠ ar^2 ∧ ar ≠ ar^2 ∧
  a + ar + ar^2 = 180 :=
sorry

end no_non_similar_triangles_with_geometric_angles_l178_178002


namespace carol_first_six_l178_178935

noncomputable def prob_six : ℚ := 1 / 6
noncomputable def not_six : ℚ := 5 / 6
noncomputable def first_term : ℚ := not_six * not_six * prob_six
noncomputable def common_ratio : ℚ := (not_six) ^ 4
noncomputable def carol_prob_first_six : ℚ := first_term / (1 - common_ratio)

theorem carol_first_six : carol_prob_first_six = 125 / 671 := sorry

end carol_first_six_l178_178935


namespace lg_sum_eq_lg_double_diff_l178_178978

theorem lg_sum_eq_lg_double_diff (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_harmonic : 2 / y = 1 / x + 1 / z) : 
  Real.log (x + z) + Real.log (x - 2 * y + z) = 2 * Real.log (x - z) := 
by
  sorry

end lg_sum_eq_lg_double_diff_l178_178978


namespace range_of_a_l178_178528

theorem range_of_a (a : ℝ) :
  (0 + 0 + a) * (2 - 1 + a) < 0 ↔ (-1 < a ∧ a < 0) :=
by sorry

end range_of_a_l178_178528


namespace smallest_n_satisfying_conditions_l178_178274

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), (n > 0) ∧ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (n * α^2 + a * α + b = 0) ∧ (n * β^2 + a * β + b = 0)
 ) ∧ (∀ (m : ℕ), m > 0 ∧ m < n → ¬ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (m * α^2 + a * α + b = 0) ∧ (m * β^2 + a * β + b = 0))) := 
sorry

end smallest_n_satisfying_conditions_l178_178274


namespace boat_distance_along_stream_l178_178989

-- Define the conditions
def boat_speed_still_water := 15 -- km/hr
def distance_against_stream_one_hour := 9 -- km

-- Define the speed of the stream
def stream_speed := boat_speed_still_water - distance_against_stream_one_hour -- km/hr

-- Define the effective speed along the stream
def effective_speed_along_stream := boat_speed_still_water + stream_speed -- km/hr

-- Define the proof statement
theorem boat_distance_along_stream : effective_speed_along_stream = 21 :=
by
  -- Given conditions and definitions, the steps are assumed logically correct
  sorry

end boat_distance_along_stream_l178_178989


namespace num_lines_passing_through_4x4_grid_l178_178646

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 66. -/
theorem num_lines_passing_through_4x4_grid : 
  let p := 4 * 4 in
  let total_point_pairs := p * (p - 1) / 2 in
  let horizontal_lines_count := 4 in
  let vertical_lines_count := 4 in
  let diagonal_lines_4_count := 2 in
  let diagonal_lines_3_count := 2 in
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count in
  (total_point_pairs - overcount_correction) = 66 :=
by
  let p := 4 * 4
  let total_point_pairs := p * (p - 1) / 2
  let horizontal_lines_count := 4
  let vertical_lines_count := 4
  let diagonal_lines_4_count := 2
  let diagonal_lines_3_count := 2
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count
  have h_correct_count : total_point_pairs - overcount_correction = 66, from sorry
  exact h_correct_count

end num_lines_passing_through_4x4_grid_l178_178646


namespace no_non_similar_triangles_with_geometric_angles_l178_178003

theorem no_non_similar_triangles_with_geometric_angles :
  ¬ ∃ (a r : ℤ), 0 < a ∧ 0 < r ∧ a ≠ ar ∧ a ≠ ar^2 ∧ ar ≠ ar^2 ∧
  a + ar + ar^2 = 180 :=
sorry

end no_non_similar_triangles_with_geometric_angles_l178_178003


namespace trapezoid_height_l178_178356

-- We are given the lengths of the sides of the trapezoid
def length_parallel1 : ℝ := 25
def length_parallel2 : ℝ := 4
def length_non_parallel1 : ℝ := 20
def length_non_parallel2 : ℝ := 13

-- We need to prove that the height of the trapezoid is 12 cm
theorem trapezoid_height (h : ℝ) :
  (h^2 + (20^2 - 16^2) = 144 ∧ h = 12) :=
sorry

end trapezoid_height_l178_178356


namespace correct_statements_l178_178984

def studentsPopulation : Nat := 70000
def sampleSize : Nat := 1000
def isSamplePopulation (s : Nat) (p : Nat) : Prop := s < p
def averageSampleEqualsPopulation (sampleAvg populationAvg : ℕ) : Prop := sampleAvg = populationAvg
def isPopulation (p : Nat) : Prop := p = studentsPopulation

theorem correct_statements (p s : ℕ) (h1 : isSamplePopulation s p) (h2 : isPopulation p) 
  (h4 : s = sampleSize) : 
  (isSamplePopulation s p ∧ ¬averageSampleEqualsPopulation 1 1 ∧ isPopulation p ∧ s = sampleSize) := 
by
  sorry

end correct_statements_l178_178984


namespace find_y_l178_178164

theorem find_y (a b y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : y > 0)
  (h4 : (2 * a)^(4 * b) = a^b * y^(3 * b)) : y = 2^(4 / 3) * a :=
by
  sorry

end find_y_l178_178164


namespace average_percentage_l178_178243

theorem average_percentage (num_students1 num_students2 : Nat) (avg1 avg2 avg : Nat) :
  num_students1 = 15 ->
  avg1 = 73 ->
  num_students2 = 10 ->
  avg2 = 88 ->
  (num_students1 * avg1 + num_students2 * avg2) / (num_students1 + num_students2) = avg ->
  avg = 79 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_percentage_l178_178243


namespace basketball_team_points_l178_178378

theorem basketball_team_points (total_points : ℕ) (number_of_players : ℕ) (points_per_player : ℕ) 
  (h1 : total_points = 18) (h2 : number_of_players = 9) : points_per_player = 2 :=
by {
  sorry -- Proof goes here
}

end basketball_team_points_l178_178378


namespace carpet_coverage_percentage_l178_178578

variable (l w : ℝ) (floor_area carpet_area : ℝ)

theorem carpet_coverage_percentage 
  (h_carpet_area: carpet_area = l * w) 
  (h_floor_area: floor_area = 180) 
  (hl : l = 4) 
  (hw : w = 9) : 
  carpet_area / floor_area * 100 = 20 := by
  sorry

end carpet_coverage_percentage_l178_178578


namespace tangent_line_intersects_circle_l178_178605

-- Definitions and conditions
def curve (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + a*x + 1 - 2*a)
def tangent_line_at_P (a : ℝ) (x : ℝ) : ℝ := (1 - a) * x + (1 - 2*a)
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16
def fixed_point : ℝ × ℝ := (-2, -1)

-- Theorem statement
theorem tangent_line_intersects_circle (a : ℝ) :
  ∃ x y, (fixed_point = (x, y) ∧ curve a 0 = 1 - 2*a ∧ tangent_line_at_P a x = y) → circle x y := 
sorry

end tangent_line_intersects_circle_l178_178605


namespace jeff_can_store_songs_l178_178159

def gbToMb (gb : ℕ) : ℕ := gb * 1000

def newAppsStorage : ℕ :=
  5 * 450 + 5 * 300 + 5 * 150

def newPhotosStorage : ℕ :=
  300 * 4 + 50 * 8

def newVideosStorage : ℕ :=
  15 * 400 + 30 * 200

def newPDFsStorage : ℕ :=
  25 * 20

def totalNewStorage : ℕ :=
  newAppsStorage + newPhotosStorage + newVideosStorage + newPDFsStorage

def existingStorage : ℕ :=
  gbToMb 7

def totalUsedStorage : ℕ :=
  existingStorage + totalNewStorage

def totalStorage : ℕ :=
  gbToMb 32

def remainingStorage : ℕ :=
  totalStorage - totalUsedStorage

def numSongs (storage : ℕ) (avgSongSize : ℕ) : ℕ :=
  storage / avgSongSize

theorem jeff_can_store_songs : 
  numSongs remainingStorage 20 = 320 :=
by
  sorry

end jeff_can_store_songs_l178_178159


namespace sum_of_reciprocals_l178_178042

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + y = 3 * x * y) (h2 : x - y = 2) : (1/x + 1/y) = 4/3 :=
by
  -- Proof omitted
  sorry

end sum_of_reciprocals_l178_178042


namespace train_speed_proof_l178_178922

noncomputable def train_length : ℝ := 620
noncomputable def crossing_time : ℝ := 30.99752019838413
noncomputable def man_speed_kmh : ℝ := 8

noncomputable def man_speed_ms : ℝ := man_speed_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := train_length / crossing_time
noncomputable def train_speed_ms : ℝ := relative_speed + man_speed_ms
noncomputable def train_speed_kmh : ℝ := train_speed_ms * (3600 / 1000)

theorem train_speed_proof : abs (train_speed_kmh - 80) < 0.0001 := by
  sorry

end train_speed_proof_l178_178922


namespace eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l178_178144

theorem eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five:
  (0.85 * 40) - (4 / 5 * 25) = 14 :=
by
  sorry

end eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l178_178144


namespace second_player_cannot_prevent_first_l178_178876

noncomputable def player_choice (set_x2_coeff_to_zero : Prop) (first_player_sets : Prop) (second_player_cannot_prevent : Prop) : Prop :=
  ∀ (b : ℝ) (c : ℝ), (set_x2_coeff_to_zero ∧ first_player_sets ∧ second_player_cannot_prevent) → 
  (∀ x : ℝ, x^3 + b * x + c = 0 → ∃! x : ℝ, x^3 + b * x + c = 0)

theorem second_player_cannot_prevent_first (b c : ℝ) :
  player_choice (set_x2_coeff_to_zero := true)
                (first_player_sets := true)
                (second_player_cannot_prevent := true) :=
sorry

end second_player_cannot_prevent_first_l178_178876


namespace total_wages_l178_178576

theorem total_wages (A_days B_days : ℝ) (A_wages : ℝ) (W : ℝ) 
  (h1 : A_days = 10)
  (h2 : B_days = 15)
  (h3 : A_wages = 2100) :
  W = 3500 :=
by sorry

end total_wages_l178_178576


namespace proof_problem_l178_178481

noncomputable def a : ℝ := 2 - 0.5
noncomputable def b : ℝ := Real.log (Real.pi) / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 4

theorem proof_problem : b > a ∧ a > c := 
by
sorry

end proof_problem_l178_178481


namespace spadesuit_problem_l178_178806

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := abs (a - b)

-- Theorem statement
theorem spadesuit_problem : spadesuit (spadesuit 2 3) (spadesuit 6 (spadesuit 9 4)) = 0 := 
sorry

end spadesuit_problem_l178_178806


namespace area_difference_l178_178411

noncomputable def speed_ratio_A_B : ℚ := 3 / 2
noncomputable def side_length : ℝ := 100
noncomputable def perimeter : ℝ := 4 * side_length

noncomputable def distance_A := (3 / 5) * perimeter
noncomputable def distance_B := perimeter - distance_A

noncomputable def EC := distance_A - 2 * side_length
noncomputable def DE := distance_B - side_length

noncomputable def area_ADE := 0.5 * DE * side_length
noncomputable def area_BCE := 0.5 * EC * side_length

theorem area_difference :
  (area_ADE - area_BCE) = 1000 :=
by
  sorry

end area_difference_l178_178411


namespace peter_work_days_l178_178569

variable (W M P : ℝ)
variable (h1 : M + P = W / 20) -- Combined rate of Matt and Peter
variable (h2 : 12 * (W / 20) + 14 * P = W) -- Work done by Matt and Peter for 12 days + Peter's remaining work

theorem peter_work_days :
  P = W / 35 :=
by
  sorry

end peter_work_days_l178_178569


namespace solve_for_x_l178_178886

theorem solve_for_x (x : ℚ) (h : (x - 75) / 4 = (5 - 3 * x) / 7) : x = 545 / 19 :=
sorry

end solve_for_x_l178_178886


namespace find_prime_triplet_l178_178882

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / 2 → (m ∣ n) → False

theorem find_prime_triplet :
  ∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ 
  (p + q = r) ∧ 
  (∃ k : ℕ, (r - p) * (q - p) - 27 * p = k * k) ∧ 
  (p = 2 ∧ q = 29 ∧ r = 31) := by
  sorry

end find_prime_triplet_l178_178882


namespace temperature_problem_l178_178801

theorem temperature_problem (N : ℤ) (M L : ℤ) :
  M = L + N →
  (M - 10) - (L + 6) = 4 ∨ (M - 10) - (L + 6) = -4 →
  (N - 16 = 4 ∨ 16 - N = 4) →
  ((N = 20 ∨ N = 12) → 20 * 12 = 240) :=
by
   sorry

end temperature_problem_l178_178801


namespace right_triangle_area_l178_178314

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end right_triangle_area_l178_178314


namespace alpha_centauri_boards_l178_178741

-- Definitions representing the given conditions.
def valid_3x3 (gold_cells : ℕ → ℕ → Bool) (A : ℕ) : Prop :=
  ∀ i j, (Σ k l, gold_cells (i + k) (j + l) ∧ 0 ≤ k < 3 ∧ 0 ≤ l < 3) = A

def valid_2x4_or_4x2 (gold_cells : ℕ → ℕ → Bool) (Z : ℕ) : Prop :=
  ∀ i j, (Σ k l, gold_cells (i + k) (j + l) ∧ 0 ≤ k < 2 ∧ 0 ≤ l < 4) = Z ∧
         (Σ k l, gold_cells (i + k) (j + l) ∧ 0 ≤ k < 4 ∧ 0 ≤ l < 2) = Z

-- The theorem to be proved.
theorem alpha_centauri_boards (gold_cells : ℕ → ℕ → Bool) (A Z : ℕ) :
  valid_3x3 gold_cells A ∧ valid_2x4_or_4x2 gold_cells Z →
  (A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8) :=
sorry

end alpha_centauri_boards_l178_178741


namespace correct_operation_l178_178561

theorem correct_operation (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by sorry

end correct_operation_l178_178561


namespace smallest_five_digit_int_equiv_mod_l178_178913

theorem smallest_five_digit_int_equiv_mod (n : ℕ) (h1 : 10000 ≤ n) (h2 : n % 9 = 4) : n = 10003 := 
sorry

end smallest_five_digit_int_equiv_mod_l178_178913


namespace potatoes_left_l178_178734

def p_initial : ℕ := 8
def p_eaten : ℕ := 3
def p_left : ℕ := p_initial - p_eaten

theorem potatoes_left : p_left = 5 := by
  sorry

end potatoes_left_l178_178734


namespace geometric_sequence_condition_l178_178363

-- Given the sum of the first n terms of the sequence {a_n} is S_n = 2^n + c,
-- we need to prove that the sequence {a_n} is a geometric sequence if and only if c = -1.
theorem geometric_sequence_condition (c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n, S n = 2^n + c) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (∃ q, ∀ n ≥ 1, a n = a 1 * q ^ (n - 1)) ↔ (c = -1) :=
by
  -- Proof skipped
  sorry

end geometric_sequence_condition_l178_178363


namespace time_per_step_l178_178805

def apply_and_dry_time (total_time steps : ℕ) : ℕ :=
  total_time / steps

theorem time_per_step : apply_and_dry_time 120 6 = 20 := by
  -- Proof omitted
  sorry

end time_per_step_l178_178805


namespace f_1_eq_2_f_6_plus_f_7_eq_15_f_2012_eq_3849_l178_178729

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_properties (n : ℕ+) : f (f n) = 3 * n

axiom f_increasing (n : ℕ+) : f (n + 1) > f n

-- Proof for f(1)
theorem f_1_eq_2 : f 1 = 2 := 
by
sorry

-- Proof for f(6) + f(7)
theorem f_6_plus_f_7_eq_15 : f 6 + f 7 = 15 := 
by
sorry

-- Proof for f(2012)
theorem f_2012_eq_3849 : f 2012 = 3849 := 
by
sorry

end f_1_eq_2_f_6_plus_f_7_eq_15_f_2012_eq_3849_l178_178729


namespace find_a_l178_178616

theorem find_a (a : ℝ) (h : (∃ x : ℝ, (a - 3) * x ^ |a - 2| + 4 = 0) ∧ |a-2| = 1) : a = 1 :=
sorry

end find_a_l178_178616


namespace lines_in_4x4_grid_l178_178641

theorem lines_in_4x4_grid :
  let n := 4
  let total_points := n * n
  let choose_two_points := total_points.choose 2
  let horizontal_and_vertical_lines := n + n
  let diagonal_lines := 6 -- based on detailed breakdown
  let adjustment_for_lines_through_four_points := 8 * 3
  let adjustment_for_lines_through_three_points := 4 * 2
  let initial_line_count := choose_two_points
  let adjusted_line_count := initial_line_count - adjustment_for_lines_through_four_points - adjustment_for_lines_through_three_points
  in adjusted_line_count = 88 := 
by {
  exact 88 // Placeholder proof statement
  sorry
}

end lines_in_4x4_grid_l178_178641


namespace min_distance_between_tracks_l178_178883

noncomputable def min_distance : ℝ :=
  (Real.sqrt 163 - 6) / 3

theorem min_distance_between_tracks :
  let RationalManTrack := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let IrrationalManTrack := {p : ℝ × ℝ | (p.1 - 2)^2 / 9 + p.2^2 / 25 = 1}
  ∀ pA ∈ RationalManTrack, ∀ pB ∈ IrrationalManTrack,
  dist pA pB = min_distance :=
sorry

end min_distance_between_tracks_l178_178883


namespace intersection_A_B_l178_178962

-- Define the sets A and B based on given conditions
def A : Set ℝ := { x | x^2 ≤ 1 }
def B : Set ℝ := { x | (x - 2) / x ≤ 0 }

-- State the proof problem
theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_A_B_l178_178962


namespace students_side_by_side_with_A_and_B_l178_178541

theorem students_side_by_side_with_A_and_B (total students_from_club_A students_from_club_B: ℕ) 
    (h1 : total = 100)
    (h2 : students_from_club_A = 62)
    (h3 : students_from_club_B = 54) :
  ∃ p q r : ℕ, p + q + r = 100 ∧ p + q = 62 ∧ p + r = 54 ∧ p = 16 :=
by
  sorry

end students_side_by_side_with_A_and_B_l178_178541


namespace difference_of_roots_of_quadratic_l178_178814

theorem difference_of_roots_of_quadratic :
  (∃ (r1 r2 : ℝ), 3 * r1 ^ 2 + 4 * r1 - 15 = 0 ∧
                  3 * r2 ^ 2 + 4 * r2 - 15 = 0 ∧
                  r1 + r2 = -4 / 3 ∧
                  r1 * r2 = -5 ∧
                  r1 - r2 = 14 / 3) :=
sorry

end difference_of_roots_of_quadratic_l178_178814


namespace sum_of_cubes_l178_178231

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178231


namespace custom_op_3_7_l178_178148

-- Define the custom operation (a # b)
def custom_op (a b : ℕ) : ℕ := a * b - b + b^2

-- State the theorem that proves the result
theorem custom_op_3_7 : custom_op 3 7 = 63 := by
  sorry

end custom_op_3_7_l178_178148


namespace larger_page_number_l178_178560

theorem larger_page_number (x : ℕ) (h1 : (x + (x + 1) = 125)) : (x + 1 = 63) :=
by
  sorry

end larger_page_number_l178_178560


namespace max_large_sculptures_l178_178950

theorem max_large_sculptures (x y : ℕ) (h1 : 1 * x = x) 
  (h2 : 3 * y = y + y + y) 
  (h3 : ∃ n, n = (x + y) / 2) 
  (h4 : x + 3 * y + (x + y) / 2 ≤ 30) 
  (h5 : x > y) : 
  y ≤ 4 := 
sorry

end max_large_sculptures_l178_178950


namespace lines_in_4x4_grid_l178_178643

theorem lines_in_4x4_grid :
  let n := 4
  let total_points := n * n
  let choose_two_points := total_points.choose 2
  let horizontal_and_vertical_lines := n + n
  let diagonal_lines := 6 -- based on detailed breakdown
  let adjustment_for_lines_through_four_points := 8 * 3
  let adjustment_for_lines_through_three_points := 4 * 2
  let initial_line_count := choose_two_points
  let adjusted_line_count := initial_line_count - adjustment_for_lines_through_four_points - adjustment_for_lines_through_three_points
  in adjusted_line_count = 88 := 
by {
  exact 88 // Placeholder proof statement
  sorry
}

end lines_in_4x4_grid_l178_178643


namespace amount_of_pizza_needed_l178_178064

theorem amount_of_pizza_needed :
  (1 / 2 + 1 / 3 + 1 / 6) = 1 := by
  sorry

end amount_of_pizza_needed_l178_178064


namespace total_marbles_l178_178735

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3
def Peter_marbles : ℕ := 7

theorem total_marbles : Mary_marbles + Joan_marbles + Peter_marbles = 19 := by
  sorry

end total_marbles_l178_178735


namespace cubic_function_decreasing_l178_178354

-- Define the given function
def f (a x : ℝ) : ℝ := a * x^3 - 1

-- Define the condition that the function is decreasing on ℝ
def is_decreasing_on_R (a : ℝ) : Prop :=
  ∀ x : ℝ, 3 * a * x^2 ≤ 0 

-- Main theorem and its statement
theorem cubic_function_decreasing (a : ℝ) (h : is_decreasing_on_R a) : a < 0 :=
sorry

end cubic_function_decreasing_l178_178354


namespace coefficient_x3_in_binomial_expansion_l178_178428

-- Define the binomial expansion term
def binomial_general_term (n k : ℕ) (a b : ℤ) : ℤ :=
  Nat.choose n k * a^(n-k) * b^k

-- Prove the coefficient of the term containing x^3 in the expansion of (x-4)^5 is 160
theorem coefficient_x3_in_binomial_expansion :
  binomial_general_term 5 2 1 (-4) = 160 := by
  sorry

end coefficient_x3_in_binomial_expansion_l178_178428


namespace intersection_A_B_l178_178961

-- Definitions of sets A and B
def A : Set ℕ := {2, 3, 5, 7}
def B : Set ℕ := {1, 2, 3, 5, 8}

-- Prove that the intersection of sets A and B is {2, 3, 5}
theorem intersection_A_B :
  A ∩ B = {2, 3, 5} :=
sorry

end intersection_A_B_l178_178961


namespace part1_part2_part3_l178_178305

-- Part 1
def harmonic_fraction (num denom : ℚ) : Prop :=
  ∃ a b : ℚ, num = a - 2 * b ∧ denom = a^2 - b^2 ∧ ¬(∃ x : ℚ, a - 2 * b = (a - b) * x)

theorem part1 (a b : ℚ) (h : harmonic_fraction (a - 2 * b) (a^2 - b^2)) : true :=
  by sorry

-- Part 2
theorem part2 (a : ℕ) (h : harmonic_fraction (x - 1) (x^2 + a * x + 4)) : a = 4 ∨ a = 5 :=
  by sorry

-- Part 3
theorem part3 (a b : ℚ) :
  (4 * a^2 / (a * b^2 - b^3) - a / b * 4 / b) = (4 * a / (ab - b^2)) :=
  by sorry

end part1_part2_part3_l178_178305


namespace fraction_Renz_Miles_l178_178338

-- Given definitions and conditions
def Mitch_macarons : ℕ := 20
def Joshua_diff : ℕ := 6
def kids : ℕ := 68
def macarons_per_kid : ℕ := 2
def total_macarons_given : ℕ := kids * macarons_per_kid
def Joshua_macarons : ℕ := Mitch_macarons + Joshua_diff
def Miles_macarons : ℕ := 2 * Joshua_macarons
def Mitch_Joshua_Miles_macarons : ℕ := Mitch_macarons + Joshua_macarons + Miles_macarons
def Renz_macarons : ℕ := total_macarons_given - Mitch_Joshua_Miles_macarons

-- The theorem to prove
theorem fraction_Renz_Miles : (Renz_macarons : ℚ) / (Miles_macarons : ℚ) = 19 / 26 :=
by
  sorry

end fraction_Renz_Miles_l178_178338


namespace find_A_l178_178968

variable (U A CU_A : Set ℕ)

axiom U_is_universal : U = {1, 3, 5, 7, 9}
axiom CU_A_is_complement : CU_A = {5, 7}

theorem find_A (h1 : U = {1, 3, 5, 7, 9}) (h2 : CU_A = {5, 7}) : 
  A = {1, 3, 9} :=
by
  sorry

end find_A_l178_178968


namespace lines_in_4x4_grid_l178_178642

theorem lines_in_4x4_grid :
  let n := 4
  let total_points := n * n
  let choose_two_points := total_points.choose 2
  let horizontal_and_vertical_lines := n + n
  let diagonal_lines := 6 -- based on detailed breakdown
  let adjustment_for_lines_through_four_points := 8 * 3
  let adjustment_for_lines_through_three_points := 4 * 2
  let initial_line_count := choose_two_points
  let adjusted_line_count := initial_line_count - adjustment_for_lines_through_four_points - adjustment_for_lines_through_three_points
  in adjusted_line_count = 88 := 
by {
  exact 88 // Placeholder proof statement
  sorry
}

end lines_in_4x4_grid_l178_178642


namespace distinct_lines_count_in_4x4_grid_l178_178687

theorem distinct_lines_count_in_4x4_grid :
  let grid_points := (finRange 4).product (finRange 4)
  let lines := {line : Set (ℕ × ℕ) | ∃ (a b : ℤ), ∀ p ∈ line, a * (p.1:ℤ) + b * (p.2:ℤ) = 1}
  let distinct_lines := {line ∈ lines | ∃ (p1 p2 : ℕ × ℕ), p1 ∈ grid_points ∧ p2 ∈ grid_points ∧ p1 ≠ p2 ∧ line = {p | this line passes through p}}
  lines.card = 50 :=
by
  sorry

end distinct_lines_count_in_4x4_grid_l178_178687


namespace total_cost_correct_l178_178207

-- Define the conditions
def uber_cost : ℤ := 22
def lyft_additional_cost : ℤ := 3
def taxi_additional_cost : ℤ := 4
def tip_percentage : ℚ := 0.20

-- Define the variables for cost of Lyft and Taxi based on the problem
def lyft_cost : ℤ := uber_cost - lyft_additional_cost
def taxi_cost : ℤ := lyft_cost - taxi_additional_cost

-- Calculate the tip
def tip : ℚ := taxi_cost * tip_percentage

-- Final total cost including the tip
def total_cost : ℚ := taxi_cost + tip

-- The theorem to prove
theorem total_cost_correct :
  total_cost = 18 := by
  sorry

end total_cost_correct_l178_178207


namespace min_value_frac_sum_l178_178027

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 2) : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 3 * b = 2 ∧ (2 / a + 4 / b) = 14) :=
by
  sorry

end min_value_frac_sum_l178_178027


namespace area_XMY_l178_178853

-- Definitions
structure Triangle :=
(area : ℝ)

def ratio (a b : ℝ) : Prop := ∃ k : ℝ, (a = k * b)

-- Given conditions
variables {XYZ XMY YZ MY : ℝ}
variables (h1 : ratio XYZ 35)
variables (h2 : ratio (XM / MY) (5 / 2))

-- Theorem to prove
theorem area_XMY (hYZ_ratio : YZ = XM + MY) (hshared_height : true) : XMY = 10 :=
by
  sorry

end area_XMY_l178_178853


namespace sum_of_digits_in_rectangle_l178_178178

theorem sum_of_digits_in_rectangle :
  ∃ A B C : ℕ,
    (4 + A + 1 + B = 12) ∧
    (4 + A + 1 + B = 6 + 6) ∧
    (C + 1 + 6 + C = 11) ∧
    (1 + B + 2 + C = 9) ∧
    (A + 8 + 8 = 8) ∧
    (A + 8 + B + 2 + C = 13) ∧
    (9 = 4 + A + 1 + B) ∧
    (B + 2 + C = 9) ∧    
    B = 5 ∧ A = 2 ∧ C = 6 :=
sorry

end sum_of_digits_in_rectangle_l178_178178


namespace joann_lollipops_l178_178324

theorem joann_lollipops : 
  ∃ (a : ℚ), 
  (7 * a  + 3 * (1 + 2 + 3 + 4 + 5 + 6) = 150) ∧ 
  (a_4 = a + 9) ∧ 
  (a_4 = 150 / 7) :=
by
  sorry

end joann_lollipops_l178_178324


namespace trapezium_distance_parallel_sides_l178_178272

theorem trapezium_distance_parallel_sides (a b A : ℝ) (h : ℝ) (h1 : a = 20) (h2 : b = 18) (h3 : A = 380) :
  A = (1 / 2) * (a + b) * h → h = 20 :=
by
  intro h4
  rw [h1, h2, h3] at h4
  sorry

end trapezium_distance_parallel_sides_l178_178272


namespace four_digit_number_divisibility_l178_178808

theorem four_digit_number_divisibility : ∃ x : ℕ, 
  (let n := 1000 + x * 100 + 50 + x; 
   ∃ k₁ k₂ : ℤ, (n = 36 * k₁) ∧ ((10 * 5 + x) = 4 * k₂) ∧ ((2 * x + 6) % 9 = 0)) :=
sorry

end four_digit_number_divisibility_l178_178808


namespace sally_pens_initial_count_l178_178496

theorem sally_pens_initial_count :
  ∃ P : ℕ, (P - (7 * 44)) / 2 = 17 ∧ P = 342 :=
by 
  sorry

end sally_pens_initial_count_l178_178496


namespace non_red_fraction_l178_178786

-- Define the conditions
def cube_edge : ℕ := 4
def num_cubes : ℕ := 64
def num_red_cubes : ℕ := 48
def num_white_cubes : ℕ := 12
def num_blue_cubes : ℕ := 4
def total_surface_area : ℕ := 6 * (cube_edge * cube_edge)

-- Define the non-red surface area exposed
def white_cube_exposed_area : ℕ := 12
def blue_cube_exposed_area : ℕ := 0

-- Calculating non-red area
def non_red_surface_area : ℕ := white_cube_exposed_area + blue_cube_exposed_area

-- The theorem to prove
theorem non_red_fraction (cube_edge : ℕ) (num_cubes : ℕ) (num_red_cubes : ℕ) 
  (num_white_cubes : ℕ) (num_blue_cubes : ℕ) (total_surface_area : ℕ) 
  (non_red_surface_area : ℕ) : 
  (non_red_surface_area : ℚ) / (total_surface_area : ℚ) = 1 / 8 :=
by 
  sorry

end non_red_fraction_l178_178786


namespace lines_in_4_by_4_grid_l178_178680

-- Definition for the grid and the number of lattice points.
def grid : Nat := 16

-- Theorem stating that the number of different lines passing through at least two points in a 4-by-4 grid of lattice points.
theorem lines_in_4_by_4_grid : 
  (number_of_lines : Nat) → number_of_lines = 40 ↔ grid = 16 := 
by
  -- Calculating number of lines passing through at least two points in a 4-by-4 grid.
  sorry -- proof skipped

end lines_in_4_by_4_grid_l178_178680


namespace solve_inequalities_l178_178077

-- Conditions from the problem
def condition1 (x y : ℝ) : Prop := 13 * x ^ 2 - 4 * x * y + 4 * y ^ 2 ≤ 2
def condition2 (x y : ℝ) : Prop := 2 * x - 4 * y ≤ -3

-- Given answers from the solution
def solution_x : ℝ := -1/3
def solution_y : ℝ := 2/3

-- Translate the proof problem in Lean
theorem solve_inequalities : condition1 solution_x solution_y ∧ condition2 solution_x solution_y :=
by
  -- Here you will provide the proof.
  sorry

end solve_inequalities_l178_178077


namespace print_papers_in_time_l178_178856

theorem print_papers_in_time :
  ∃ (n : ℕ), 35 * 15 * n = 500000 * 21 * n := by
  sorry

end print_papers_in_time_l178_178856


namespace find_unique_n_k_l178_178108

theorem find_unique_n_k (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
    (n+1)^n = 2 * n^k + 3 * n + 1 ↔ (n = 3 ∧ k = 3) := by
  sorry

end find_unique_n_k_l178_178108


namespace euro_operation_example_l178_178567

def euro_operation (x y : ℕ) : ℕ := 3 * x * y - x - y

theorem euro_operation_example : euro_operation 6 (euro_operation 4 2) = 300 := by
  sorry

end euro_operation_example_l178_178567


namespace find_pairs_l178_178812

theorem find_pairs :
  { (m, n) : ℕ × ℕ | (m > 0) ∧ (n > 0) ∧ (m^2 - n ∣ m + n^2)
      ∧ (n^2 - m ∣ n + m^2) } = { (2, 2), (3, 3), (1, 2), (2, 1), (3, 2), (2, 3) } :=
sorry

end find_pairs_l178_178812


namespace find_principal_amount_l178_178091

variables (P R : ℝ)

theorem find_principal_amount (h : (4 * P * (R + 2) / 100) - (4 * P * R / 100) = 56) : P = 700 :=
sorry

end find_principal_amount_l178_178091


namespace sum_of_arithmetic_seq_minimum_value_n_equals_5_l178_178362

variable {a : ℕ → ℝ} -- Define a sequence of real numbers
variable {S : ℕ → ℝ} -- Define the sum function for the sequence

-- Assume conditions
axiom a3_a8_neg : a 3 + a 8 < 0
axiom S11_pos : S 11 > 0

-- Prove the minimum value of S_n occurs at n = 5
theorem sum_of_arithmetic_seq_minimum_value_n_equals_5 :
  ∃ n, (∀ m < 5, S m ≥ S n) ∧ (∀ m > 5, S m > S n) ∧ n = 5 :=
sorry

end sum_of_arithmetic_seq_minimum_value_n_equals_5_l178_178362


namespace avg_weight_b_c_l178_178756

theorem avg_weight_b_c
  (a b c : ℝ)
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : b = 31) :
  (b + c) / 2 = 43 := 
by {
  sorry
}

end avg_weight_b_c_l178_178756


namespace order_of_abcd_l178_178979

-- Define the rational numbers a, b, c, d
variables {a b c d : ℚ}

-- State the conditions as assumptions
axiom h1 : a + b = c + d
axiom h2 : a + d < b + c
axiom h3 : c < d

-- The goal is to prove the correct order of a, b, c, d
theorem order_of_abcd (a b c d : ℚ) (h1 : a + b = c + d) (h2 : a + d < b + c) (h3 : c < d) :
  b > d ∧ d > c ∧ c > a :=
sorry

end order_of_abcd_l178_178979


namespace custom_op_eval_l178_178974

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b - a^2 * b

theorem custom_op_eval :
  custom_op 3 4 = -4 :=
by
  sorry

end custom_op_eval_l178_178974


namespace regular_triangular_prism_properties_l178_178128

-- Regular triangular pyramid defined
structure RegularTriangularPyramid (height : ℝ) (base_side : ℝ)

-- Regular triangular prism defined
structure RegularTriangularPrism (height : ℝ) (base_side : ℝ) (lateral_area : ℝ)

-- Given data
def pyramid := RegularTriangularPyramid 15 12
def prism_lateral_area := 120

-- Statement of the problem
theorem regular_triangular_prism_properties (h_prism : ℝ) (ratio_lateral_area : ℚ) :
  (h_prism = 10 ∨ h_prism = 5) ∧ (ratio_lateral_area = 1/9 ∨ ratio_lateral_area = 4/9) :=
sorry

end regular_triangular_prism_properties_l178_178128


namespace sum_of_cubes_l178_178233

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178233


namespace proof_problem_l178_178167

open Set Real

noncomputable def f (x : ℝ) : ℝ := sin x
noncomputable def g (x : ℝ) : ℝ := cos x
def U : Set ℝ := univ
def M : Set ℝ := {x | f x ≠ 0}
def N : Set ℝ := {x | g x ≠ 0}
def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem proof_problem :
  {x : ℝ | f x * g x = 0} = (C_U M) ∪ (C_U N) :=
by
  sorry

end proof_problem_l178_178167


namespace average_of_three_strings_l178_178342

variable (len1 len2 len3 : ℝ)
variable (h1 : len1 = 2)
variable (h2 : len2 = 5)
variable (h3 : len3 = 3)

def average_length (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem average_of_three_strings : average_length len1 len2 len3 = 10 / 3 := by
  rw [←h1, ←h2, ←h3]
  rw [average_length]
  norm_num
  sorry

end average_of_three_strings_l178_178342


namespace distance_between_cities_is_39_l178_178519

noncomputable def city_distance : ℕ :=
  ∃ S : ℕ, (∀ x : ℕ, (0 ≤ x ∧ x ≤ S) → gcd x (S - x) ∈ {1, 3, 13}) ∧
             (S % (Nat.lcm (Nat.lcm 1 3) 13) = 0) ∧ S = 39

theorem distance_between_cities_is_39 :
  city_distance := 
by
  sorry

end distance_between_cities_is_39_l178_178519


namespace philip_farm_animal_count_l178_178880

def number_of_cows : ℕ := 20

def number_of_ducks : ℕ := number_of_cows * 3 / 2

def total_cows_and_ducks : ℕ := number_of_cows + number_of_ducks

def number_of_pigs : ℕ := total_cows_and_ducks / 5

def total_animals : ℕ := total_cows_and_ducks + number_of_pigs

theorem philip_farm_animal_count : total_animals = 60 := by
  sorry

end philip_farm_animal_count_l178_178880


namespace remainder_of_x50_div_by_x_sub_1_cubed_l178_178111

theorem remainder_of_x50_div_by_x_sub_1_cubed :
  (x^50 % (x-1)^3) = (1225*x^2 - 2500*x + 1276) :=
sorry

end remainder_of_x50_div_by_x_sub_1_cubed_l178_178111


namespace rectangle_integer_sides_noncongruent_count_l178_178087

theorem rectangle_integer_sides_noncongruent_count (h w : ℕ) :
  (2 * (w + h) = 72 ∧ w ≠ h) ∨ ((w = h) ∧ 2 * (w + h) = 72) →
  (∃ (count : ℕ), count = 18) :=
by
  sorry

end rectangle_integer_sides_noncongruent_count_l178_178087


namespace recurring_decimal_difference_fraction_l178_178413

noncomputable def recurring_decimal_seventy_three := 73 / 99
noncomputable def decimal_seventy_three := 73 / 100

theorem recurring_decimal_difference_fraction :
  recurring_decimal_seventy_three - decimal_seventy_three = 73 / 9900 := sorry

end recurring_decimal_difference_fraction_l178_178413


namespace num_lines_passing_through_4x4_grid_l178_178650

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 66. -/
theorem num_lines_passing_through_4x4_grid : 
  let p := 4 * 4 in
  let total_point_pairs := p * (p - 1) / 2 in
  let horizontal_lines_count := 4 in
  let vertical_lines_count := 4 in
  let diagonal_lines_4_count := 2 in
  let diagonal_lines_3_count := 2 in
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count in
  (total_point_pairs - overcount_correction) = 66 :=
by
  let p := 4 * 4
  let total_point_pairs := p * (p - 1) / 2
  let horizontal_lines_count := 4
  let vertical_lines_count := 4
  let diagonal_lines_4_count := 2
  let diagonal_lines_3_count := 2
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count
  have h_correct_count : total_point_pairs - overcount_correction = 66, from sorry
  exact h_correct_count

end num_lines_passing_through_4x4_grid_l178_178650


namespace distinct_remainders_count_l178_178299

theorem distinct_remainders_count {N : ℕ} (hN : N = 420) :
  ∃ (count : ℕ), (∀ n : ℕ, n ≥ 1 ∧ n ≤ N → ((n % 5 ≠ n % 6) ∧ (n % 5 ≠ n % 7) ∧ (n % 6 ≠ n % 7))) →
  count = 386 :=
by {
  sorry
}

end distinct_remainders_count_l178_178299


namespace find_a1_in_arithmetic_sequence_l178_178015

theorem find_a1_in_arithmetic_sequence (d n a_n : ℤ) (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10) :
  ∃ a1 : ℤ, a1 = -38 :=
by
  sorry

end find_a1_in_arithmetic_sequence_l178_178015


namespace sum_series_eq_one_l178_178945

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, (2^n + 1) / (3^(2^n) + 1)

theorem sum_series_eq_one : sum_series = 1 := 
by 
  sorry

end sum_series_eq_one_l178_178945


namespace average_marks_all_students_l178_178384

theorem average_marks_all_students
  (n1 n2 : ℕ)
  (avg1 avg2 : ℕ)
  (h1 : avg1 = 40)
  (h2 : avg2 = 80)
  (h3 : n1 = 30)
  (h4 : n2 = 50) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 65 :=
by
  sorry

end average_marks_all_students_l178_178384


namespace distinct_lines_count_in_4x4_grid_l178_178690

theorem distinct_lines_count_in_4x4_grid :
  let grid_points := (finRange 4).product (finRange 4)
  let lines := {line : Set (ℕ × ℕ) | ∃ (a b : ℤ), ∀ p ∈ line, a * (p.1:ℤ) + b * (p.2:ℤ) = 1}
  let distinct_lines := {line ∈ lines | ∃ (p1 p2 : ℕ × ℕ), p1 ∈ grid_points ∧ p2 ∈ grid_points ∧ p1 ≠ p2 ∧ line = {p | this line passes through p}}
  lines.card = 50 :=
by
  sorry

end distinct_lines_count_in_4x4_grid_l178_178690


namespace total_pages_is_360_l178_178245

-- Definitions from conditions
variable (A B : ℕ) -- Rates of printer A and printer B in pages per minute.
variable (total_pages : ℕ) -- Total number of pages of the task.

-- Given conditions
axiom h1 : 24 * (A + B) = total_pages -- Condition from both printers working together.
axiom h2 : 60 * A = total_pages -- Condition from printer A alone.
axiom h3 : B = A + 3 -- Condition of printer B printing 3 more pages per minute.

-- Goal: Prove the total number of pages is 360
theorem total_pages_is_360 : total_pages = 360 := 
by 
  sorry

end total_pages_is_360_l178_178245


namespace max_principals_and_assistant_principals_l178_178336

theorem max_principals_and_assistant_principals : 
  ∀ (years term_principal term_assistant), (years = 10) ∧ (term_principal = 3) ∧ (term_assistant = 2) 
  → ∃ n, n = 9 :=
by
  sorry

end max_principals_and_assistant_principals_l178_178336


namespace equilateral_triangle_on_parallel_lines_l178_178370

theorem equilateral_triangle_on_parallel_lines 
  (l1 l2 l3 : ℝ → Prop)
  (h_parallel_12 : ∀ x y, l1 x → l2 y → ∀ z, l1 z → l2 z)
  (h_parallel_23 : ∀ x y, l2 x → l3 y → ∀ z, l2 z → l3 z) 
  (h_parallel_13 : ∀ x y, l1 x → l3 y → ∀ z, l1 z → l3 z) 
  (A : ℝ) (hA : l1 A)
  (B : ℝ) (hB : l2 B)
  (C : ℝ) (hC : l3 C):
  ∃ A B C : ℝ, l1 A ∧ l2 B ∧ l3 C ∧ (dist A B = dist B C ∧ dist B C = dist C A) :=
by
  sorry

end equilateral_triangle_on_parallel_lines_l178_178370


namespace solve_cubic_equation_l178_178498

variable (t : ℝ)

theorem solve_cubic_equation (x : ℝ) :
  x^3 - 2 * t * x^2 + t^3 = 0 ↔ 
  x = t ∨ x = t * (1 + Real.sqrt 5) / 2 ∨ x = t * (1 - Real.sqrt 5) / 2 :=
sorry

end solve_cubic_equation_l178_178498


namespace smallest_positive_period_of_f_cos_2x0_l178_178451

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sin x * Real.cos x + 2 * (Real.sqrt 3) * (Real.cos x)^2 - Real.sqrt 3

theorem smallest_positive_period_of_f :
  (∃ p > 0, ∀ x, f x = f (x + p)) ∧
  (∀ q > 0, (∀ x, f x = f (x + q)) -> q ≥ Real.pi) :=
sorry

theorem cos_2x0 (x0 : ℝ) (h0 : x0 ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h1 : f (x0 - Real.pi / 12) = 6 / 5) :
  Real.cos (2 * x0) = (3 - 4 * Real.sqrt 3) / 10 :=
sorry

end smallest_positive_period_of_f_cos_2x0_l178_178451


namespace expand_and_simplify_l178_178426

theorem expand_and_simplify :
  (x : ℝ) → (x^2 - 3 * x + 3) * (x^2 + 3 * x + 3) = x^4 - 3 * x^2 + 9 :=
by 
  sorry

end expand_and_simplify_l178_178426


namespace solve_for_m_l178_178306

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 4

theorem solve_for_m (m : ℤ) (h : ∃ x : ℝ, 2^x + x = 4 ∧ m ≤ x ∧ x ≤ m + 1) : m = 1 :=
by
  sorry

end solve_for_m_l178_178306


namespace vector_collinear_l178_178623

open Real

theorem vector_collinear 
  (m : ℝ × ℝ) (n : ℝ × ℝ) 
  (h_m : m = (0, -2)) 
  (h_n : n = (sqrt 3, 1)) : 
  ∃ k : ℝ, 2 • m + n = k • (-1, sqrt 3) :=
by
  -- Define the given vectors
  let m := (0 : ℝ, -2)
  let n := (sqrt 3, 1)

  -- Prove the existence of a scalar k such that  
  -- 2 • m + n = k • (-1, sqrt 3)
  sorry

end vector_collinear_l178_178623


namespace trapezoid_area_l178_178742

theorem trapezoid_area (EF GH EG FH : ℝ) (h : ℝ)
  (h_EF : EF = 60) (h_GH : GH = 30) (h_EG : EG = 25) (h_FH : FH = 18) (h_alt : h = 15) :
  (1 / 2 * (EF + GH) * h) = 675 :=
by
  rw [h_EF, h_GH, h_alt]
  sorry

end trapezoid_area_l178_178742


namespace plants_per_row_l178_178491

-- Define the conditions from the problem
def rows : ℕ := 7
def extra_plants : ℕ := 15
def total_plants : ℕ := 141

-- Define the problem statement to prove
theorem plants_per_row :
  ∃ x : ℕ, rows * x + extra_plants = total_plants ∧ x = 18 :=
by
  sorry

end plants_per_row_l178_178491


namespace curve_self_intersection_l178_178263

def curve_crosses_itself_at_point (x y : ℝ) : Prop :=
∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ (t₁^2 - 4 = x) ∧ (t₁^3 - 6 * t₁ + 7 = y) ∧ (t₂^2 - 4 = x) ∧ (t₂^3 - 6 * t₂ + 7 = y)

theorem curve_self_intersection : curve_crosses_itself_at_point 2 7 :=
sorry

end curve_self_intersection_l178_178263


namespace num_five_digit_palindromes_with_even_middle_l178_178298

theorem num_five_digit_palindromes_with_even_middle :
  (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ ∃ c', c = 2 * c' ∧ 0 ≤ c' ∧ c' ≤ 4 ∧ 10000 * a + 1000 * b + 100 * c + 10 * b + a ≤ 99999) →
  9 * 10 * 5 = 450 :=
by
  sorry

end num_five_digit_palindromes_with_even_middle_l178_178298


namespace complement_of_A_in_U_l178_178456

open Set

variable (U : Set ℤ := { -2, -1, 0, 1, 2 })
variable (A : Set ℤ := { x | 0 < Int.natAbs x ∧ Int.natAbs x < 2 })

theorem complement_of_A_in_U :
  U \ A = { -2, 0, 2 } :=
by
  sorry

end complement_of_A_in_U_l178_178456


namespace lines_in_4_by_4_grid_l178_178638

theorem lines_in_4_by_4_grid : 
  (count_lines_passing_through_at_least_two_points (4, 4) = 62) :=
sorry

def count_lines_passing_through_at_least_two_points (m n : ℕ) : ℕ :=
  let total_pairs := (m * n) * ((m * n) - 1) / 2
  let overcount_lines := (6 - 1) * 10 + (3 - 1) * 4
  total_pairs - overcount_lines

end lines_in_4_by_4_grid_l178_178638


namespace city_distance_GCD_l178_178512

open Int

theorem city_distance_GCD :
  ∃ S : ℤ, (∀ (x : ℤ), 0 ≤ x ∧ x ≤ S → ∃ d ∈ {1, 3, 13}, d = gcd x (S - x)) ∧ S = 39 :=
by
  sorry

end city_distance_GCD_l178_178512


namespace distance_center_to_line_l178_178758

noncomputable def circle_center : ℝ × ℝ :=
  let b := 2
  let c := -4
  (1, -2)

noncomputable def distance_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / (Real.sqrt (a^2 + b^2))

theorem distance_center_to_line : distance_point_to_line circle_center 3 4 5 = 0 :=
by
  sorry

end distance_center_to_line_l178_178758


namespace eleonora_age_l178_178573

-- Definitions
def age_eleonora (e m : ℕ) : Prop :=
m - e = 3 * (2 * e - m) ∧ 3 * e + (m + 2 * e) = 100

-- Theorem stating that Eleonora's age is 15
theorem eleonora_age (e m : ℕ) (h : age_eleonora e m) : e = 15 :=
sorry

end eleonora_age_l178_178573


namespace dvds_still_fit_in_book_l178_178923

def total_capacity : ℕ := 126
def dvds_already_in_book : ℕ := 81

theorem dvds_still_fit_in_book : (total_capacity - dvds_already_in_book = 45) :=
by
  sorry

end dvds_still_fit_in_book_l178_178923


namespace money_taken_l178_178381

def total_people : ℕ := 6
def cost_per_soda : ℝ := 0.5
def cost_per_pizza : ℝ := 1.0

theorem money_taken (total_people cost_per_soda cost_per_pizza : ℕ × ℝ × ℝ ) :
  total_people * cost_per_soda + total_people * cost_per_pizza = 9 := by
  sorry

end money_taken_l178_178381


namespace product_469160_9999_l178_178101

theorem product_469160_9999 :
  469160 * 9999 = 4690696840 :=
by
  sorry

end product_469160_9999_l178_178101


namespace largest_divisor_of_polynomial_l178_178772

theorem largest_divisor_of_polynomial (n : ℕ) (h : n % 2 = 0) : 
  105 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) :=
sorry

end largest_divisor_of_polynomial_l178_178772


namespace train_speed_is_correct_l178_178093

-- Definitions of the given conditions.
def train_length : ℕ := 250
def bridge_length : ℕ := 150
def time_taken : ℕ := 20

-- Definition of the total distance covered by the train.
def total_distance : ℕ := train_length + bridge_length

-- The speed calculation.
def speed : ℕ := total_distance / time_taken

-- The theorem that we need to prove.
theorem train_speed_is_correct : speed = 20 := by
  -- proof steps go here
  sorry

end train_speed_is_correct_l178_178093


namespace lines_in_4x4_grid_l178_178673

theorem lines_in_4x4_grid : 
  let grid_points := finset.univ.product finset.univ
  let total_points := 16
  let pairs_of_points := total_points.choose 2
  let horizontal_lines := 4
  let vertical_lines := 4
  let diagonal_lines := 2
  let lines_through_four_points := horizontal_lines + vertical_lines + diagonal_lines
  let correction := lines_through_four_points * (4.choose 2 - 1)
  let number_of_lines := pairs_of_points - correction
  in number_of_lines = 70 := 
by {
  sorry
}

end lines_in_4x4_grid_l178_178673


namespace glorias_turtle_time_l178_178141

theorem glorias_turtle_time :
  ∀ (t_G t_{Ge} t_{Gl} : ℕ), 
    t_G = 6 →
    t_{Ge} = t_G - 2 →
    t_{Gl} = 2 * t_{Ge} →
    t_{Gl} = 8 := by
  intros t_G t_{Ge} t_{Gl} hG hGe hGl
  rw [hG] at hGe
  rw [hGe, hG] at hGl
  rw [hGe]
  exact hGl
  sorry -- this is a placeholder indicating that there's no need to complete the proof steps

end glorias_turtle_time_l178_178141


namespace function_has_local_minimum_at_zero_l178_178954

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * (x - 1))

def is_local_minimum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, abs (y - x) < ε → f x ≤ f y

theorem function_has_local_minimum_at_zero :
  -4 < 0 ∧ 0 < 1 ∧ is_local_minimum f 0 := 
sorry

end function_has_local_minimum_at_zero_l178_178954


namespace B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l178_178983

namespace GoGame

-- Define the players: A, B, C
inductive Player
| A
| B
| C

open Player

-- Define the probabilities as given
def P_A_beats_B : ℝ := 0.4
def P_B_beats_C : ℝ := 0.5
def P_C_beats_A : ℝ := 0.6

-- Define the game rounds and logic
def probability_B_winning_four_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
(1 - P_A_beats_B)^2 * P_B_beats_C^2

def probability_C_winning_three_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
  P_A_beats_B * P_C_beats_A^2 * P_B_beats_C + 
  (1 - P_A_beats_B) * P_B_beats_C^2 * P_C_beats_A

-- Proof statements
theorem B_wins_four_rounds_prob_is_0_09 : 
  probability_B_winning_four_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.09 :=
by
  sorry

theorem C_wins_three_rounds_prob_is_0_162 : 
  probability_C_winning_three_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.162 :=
by
  sorry

end GoGame

end B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l178_178983


namespace find_fifth_number_l178_178504

def avg_sum_9_numbers := 936
def sum_first_5_numbers := 495
def sum_last_5_numbers := 500

theorem find_fifth_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 : ℝ)
  (h1 : A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 = avg_sum_9_numbers)
  (h2 : A1 + A2 + A3 + A4 + A5 = sum_first_5_numbers)
  (h3 : A5 + A6 + A7 + A8 + A9 = sum_last_5_numbers) :
  A5 = 29.5 :=
sorry

end find_fifth_number_l178_178504


namespace johannes_cabbage_sales_l178_178476

def price_per_kg : ℝ := 2
def earnings_wednesday : ℝ := 30
def earnings_friday : ℝ := 24
def earnings_today : ℝ := 42

theorem johannes_cabbage_sales :
  (earnings_wednesday / price_per_kg) + (earnings_friday / price_per_kg) + (earnings_today / price_per_kg) = 48 := by
  sorry

end johannes_cabbage_sales_l178_178476


namespace total_cost_proof_l178_178210

def uber_cost : ℤ := 22
def lyft_cost : ℤ := uber_cost - 3
def taxi_cost : ℤ := lyft_cost - 4
def tip : ℤ := (taxi_cost * 20) / 100
def total_cost : ℤ := taxi_cost + tip

theorem total_cost_proof :
  total_cost = 18 :=
by
  sorry

end total_cost_proof_l178_178210


namespace game_spinner_probability_l178_178926

theorem game_spinner_probability (P_A P_B P_D P_C : ℚ) (h₁ : P_A = 1/4) (h₂ : P_B = 1/3) (h₃ : P_D = 1/6) (h₄ : P_A + P_B + P_C + P_D = 1) :
  P_C = 1/4 :=
by
  sorry

end game_spinner_probability_l178_178926


namespace lines_in_4x4_grid_l178_178671

theorem lines_in_4x4_grid : 
  let grid_points := finset.univ.product finset.univ
  let total_points := 16
  let pairs_of_points := total_points.choose 2
  let horizontal_lines := 4
  let vertical_lines := 4
  let diagonal_lines := 2
  let lines_through_four_points := horizontal_lines + vertical_lines + diagonal_lines
  let correction := lines_through_four_points * (4.choose 2 - 1)
  let number_of_lines := pairs_of_points - correction
  in number_of_lines = 70 := 
by {
  sorry
}

end lines_in_4x4_grid_l178_178671


namespace count_lines_in_4x4_grid_l178_178694

theorem count_lines_in_4x4_grid : 
  let grid_points : Fin 4 × Fin 4 := 
  ∃! lines : set (set (Fin 4 × Fin 4)), 
  ∀ line ∈ lines, ∃ (p1 p2 : Fin 4 × Fin 4), p1 ≠ p2 ∧ p1 ∈ line ∧ p2 ∈ line ∧ (grid_points ⊆ line ⊆ grid_points) :=
  lines = 84 :=
sorry

end count_lines_in_4x4_grid_l178_178694


namespace new_total_lines_l178_178174

-- Definitions and conditions
variable (L : ℕ)
def increased_lines : ℕ := L + 60
def percentage_increase := (60 : ℚ) / L = 1 / 3

-- Theorem statement
theorem new_total_lines : percentage_increase L → increased_lines L = 240 :=
by
  sorry

end new_total_lines_l178_178174


namespace appropriate_chart_for_milk_powder_l178_178315

-- Define the chart requirements and the correctness condition
def ChartType := String
def pie : ChartType := "pie"
def line : ChartType := "line"
def bar : ChartType := "bar"

-- The condition we need for our proof
def representsPercentagesWell (chart: ChartType) : Prop :=
  chart = pie

-- The main theorem statement
theorem appropriate_chart_for_milk_powder : representsPercentagesWell pie :=
by
  sorry

end appropriate_chart_for_milk_powder_l178_178315


namespace total_distance_of_ship_l178_178248

-- Define the conditions
def first_day_distance : ℕ := 100
def second_day_distance := 3 * first_day_distance
def third_day_distance := second_day_distance + 110
def total_distance := first_day_distance + second_day_distance + third_day_distance

-- Theorem stating that given the conditions the total distance traveled is 810 miles
theorem total_distance_of_ship :
  total_distance = 810 := by
  sorry

end total_distance_of_ship_l178_178248


namespace cost_of_27_lilies_l178_178939

theorem cost_of_27_lilies
  (cost_18 : ℕ)
  (price_ratio : ℕ → ℕ → Prop)
  (h_cost_18 : cost_18 = 30)
  (h_price_ratio : ∀ n m c : ℕ, price_ratio n m ↔ c = n * 5 / 3 ∧ m = c * 3 / 5) :
  ∃ c : ℕ, price_ratio 27 c ∧ c = 45 := 
by
  sorry

end cost_of_27_lilies_l178_178939


namespace book_arrangement_count_l178_178007

theorem book_arrangement_count :
  let n := 6
  let identical_pairs := 2
  let total_arrangements_if_unique := n.factorial
  let ident_pair_correction := (identical_pairs.factorial * identical_pairs.factorial)
  (total_arrangements_if_unique / ident_pair_correction) = 180 := by
  sorry

end book_arrangement_count_l178_178007


namespace find_a_l178_178364

variable (a b c : ℚ)

theorem find_a (h1 : a + b + c = 150) (h2 : a - 3 = b + 4) (h3 : b + 4 = 4 * c) : 
  a = 631 / 9 :=
by
  sorry

end find_a_l178_178364


namespace willie_stickers_l178_178917

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (final_stickers : ℕ) 
  (h1 : initial_stickers = 124) 
  (h2 : given_stickers = 43) 
  (h3 : final_stickers = initial_stickers - given_stickers) :
  final_stickers = 81 :=
sorry

end willie_stickers_l178_178917


namespace max_value_of_expression_l178_178721

noncomputable def max_value_expr (a b c : ℝ) : ℝ :=
  a + b^2 + c^3

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  max_value_expr a b c ≤ 8 :=
  sorry

end max_value_of_expression_l178_178721


namespace round_robin_points_change_l178_178783

theorem round_robin_points_change (n : ℕ) (athletes : Finset ℕ) (tournament1_scores tournament2_scores : ℕ → ℚ) :
  Finset.card athletes = 2 * n →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) ≥ n) →
  (∀ a ∈ athletes, abs (tournament2_scores a - tournament1_scores a) = n) :=
by
  sorry

end round_robin_points_change_l178_178783


namespace classroom_has_total_books_l178_178850

-- Definitions for the conditions
def num_children : Nat := 10
def books_per_child : Nat := 7
def additional_books : Nat := 8

-- Total number of books the children have
def total_books_from_children : Nat := num_children * books_per_child

-- The expected total number of books in the classroom
def total_books : Nat := total_books_from_children + additional_books

-- The main theorem to be proven
theorem classroom_has_total_books : total_books = 78 :=
by
  sorry

end classroom_has_total_books_l178_178850


namespace different_lines_through_two_points_in_4_by_4_grid_l178_178635

theorem different_lines_through_two_points_in_4_by_4_grid : 
  let points := fin 4 × fin 4 in
  let number_of_lines := 
    (nat.choose 16 2) - 
    (8 * (4 - 1)) - 
    (2 * (4 - 1)) in
  number_of_lines = 90 :=
by
  sorry

end different_lines_through_two_points_in_4_by_4_grid_l178_178635


namespace rectangular_field_perimeter_l178_178586

theorem rectangular_field_perimeter
  (a b : ℝ)
  (diag_eq : a^2 + b^2 = 1156)
  (area_eq : a * b = 240)
  (side_relation : a = 2 * b) :
  2 * (a + b) = 91.2 :=
by
  sorry

end rectangular_field_perimeter_l178_178586


namespace lines_in_4_by_4_grid_l178_178639

theorem lines_in_4_by_4_grid : 
  (count_lines_passing_through_at_least_two_points (4, 4) = 62) :=
sorry

def count_lines_passing_through_at_least_two_points (m n : ℕ) : ℕ :=
  let total_pairs := (m * n) * ((m * n) - 1) / 2
  let overcount_lines := (6 - 1) * 10 + (3 - 1) * 4
  total_pairs - overcount_lines

end lines_in_4_by_4_grid_l178_178639


namespace smallest_n_13n_congruent_456_mod_5_l178_178373

theorem smallest_n_13n_congruent_456_mod_5 : ∃ n : ℕ, (n > 0) ∧ (13 * n ≡ 456 [MOD 5]) ∧ (∀ m : ℕ, (m > 0 ∧ 13 * m ≡ 456 [MOD 5]) → n ≤ m) :=
by
  sorry

end smallest_n_13n_congruent_456_mod_5_l178_178373


namespace percentage_reduction_l178_178587

theorem percentage_reduction (original reduced : ℝ) (h_original : original = 253.25) (h_reduced : reduced = 195) : 
  ((original - reduced) / original) * 100 = 22.99 :=
by
  sorry

end percentage_reduction_l178_178587


namespace Turan_2_l178_178032

noncomputable theory

open Classical

variable {V : Type*} (G : SimpleGraph V) [Fintype V] [DecidableEq V]

def no_K3 (G : SimpleGraph V) := ∀ (a b c : V), G.Adj a b → G.Adj b c → G.Adj c a → False

theorem Turan_2 (G : SimpleGraph V) [NoK3 : no_K3 G] : G.edgeFinset.card ≤ Fintype.card V * Fintype.card V / 4 := sorry

end Turan_2_l178_178032


namespace ratio_of_areas_l178_178877

theorem ratio_of_areas (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
    let S₁ := (1 - p * q * r) * (1 - p * q * r)
    let S₂ := (1 + p + p * q) * (1 + q + q * r) * (1 + r + r * p)
    S₁ / S₂ = (S₁ / S₂) := sorry

end ratio_of_areas_l178_178877


namespace part1_part2_l178_178782

def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci (n+1) + fibonacci n

theorem part1 (m n : ℕ) (h : divides m n) : divides (fibonacci m) (fibonacci n) :=
sorry

theorem part2 (m n : ℕ) : Nat.gcd (fibonacci m) (fibonacci n) = fibonacci (Nat.gcd m n) :=
sorry

end part1_part2_l178_178782


namespace height_at_15_inches_l178_178086

-- Define the conditions
def parabolic_eq (a x : ℝ) : ℝ := a * x^2 + 24
noncomputable def a : ℝ := -2 / 75
def x : ℝ := 15
def expected_y : ℝ := 18

-- Lean 4 statement
theorem height_at_15_inches :
  parabolic_eq a x = expected_y :=
by
  sorry

end height_at_15_inches_l178_178086


namespace triangle_area_is_correct_l178_178202

-- Define the given conditions
def is_isosceles_right_triangle (h : ℝ) (l : ℝ) : Prop :=
  h = l * sqrt 2

def triangle_hypotenuse := 6 * sqrt 2  -- Given hypotenuse

-- Prove that the area of the triangle is 18 square units
theorem triangle_area_is_correct : 
  ∃ (l : ℝ), is_isosceles_right_triangle triangle_hypotenuse l ∧ (1/2) * l^2 = 18 :=
by
  sorry

end triangle_area_is_correct_l178_178202


namespace no_integer_points_on_circle_l178_178117

theorem no_integer_points_on_circle : 
  ∀ x : ℤ, ¬ ((x - 3)^2 + (x + 1 + 2)^2 ≤ 64) := by
  sorry

end no_integer_points_on_circle_l178_178117


namespace sum_of_longest_altitudes_l178_178598

theorem sum_of_longest_altitudes (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) (h₄ : a^2 + b^2 = c^2) :
  a + b = 14 :=
by
  sorry

end sum_of_longest_altitudes_l178_178598


namespace number_of_lines_in_4_by_4_grid_l178_178652

/-- A 4-by-4 grid of lattice points -/
def lattice_points_4x4 : set (ℕ × ℕ) :=
  {(i, j) | i < 4 ∧ j < 4}

/-- A line in the Euclidean plane -/
def is_line (p1 p2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | ∃ λ : ℝ, p = (λ * (p2.1 - p1.1) + p1.1, λ * (p2.2 - p1.2) + p1.2)}

noncomputable def count_lines_through_points (points : set (ℕ × ℕ)) : ℕ :=
  /- counting logic to be implemented -/
  sorry

theorem number_of_lines_in_4_by_4_grid : count_lines_through_points lattice_points_4x4 = 70 :=
  sorry

end number_of_lines_in_4_by_4_grid_l178_178652


namespace ants_need_more_hours_l178_178404

theorem ants_need_more_hours (initial_sugar : ℕ) (removal_rate : ℕ) (hours_spent : ℕ) : 
  initial_sugar = 24 ∧ removal_rate = 4 ∧ hours_spent = 3 → 
  (initial_sugar - removal_rate * hours_spent) / removal_rate = 3 :=
by
  intro h
  sorry

end ants_need_more_hours_l178_178404


namespace sum_of_digits_l178_178056

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 4 + 258 = 7 * 100 + b * 10 + 2) (h2 : (7 * 100 + b * 10 + 2) % 3 = 0) :
  a + b = 4 :=
sorry

end sum_of_digits_l178_178056


namespace remainder_of_division_l178_178418

def dividend := 1234567
def divisor := 257

theorem remainder_of_division : dividend % divisor = 774 :=
by
  sorry

end remainder_of_division_l178_178418


namespace smallest_n_is_60_l178_178068

def smallest_n (n : ℕ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ (24 ∣ n^2) ∧ (450 ∣ n^3) ∧ ∀ m : ℕ, 24 ∣ m^2 → 450 ∣ m^3 → m ≥ n

theorem smallest_n_is_60 : smallest_n 60 :=
  sorry

end smallest_n_is_60_l178_178068


namespace find_phi_increasing_intervals_l178_178291

open Real

-- Defining the symmetry condition
noncomputable def symmetric_phi (x_sym : ℝ) (k : ℤ) (phi : ℝ): Prop :=
  2 * x_sym + phi = k * π + π / 2

-- Finding the value of phi given the conditions
theorem find_phi (x_sym : ℝ) (phi : ℝ) (k : ℤ) 
  (h_sym: symmetric_phi x_sym k phi) (h_phi_bound : -π < phi ∧ phi < 0)
  (h_xsym: x_sym = π / 8) :
  phi = -3 * π / 4 :=
by
  sorry

-- Defining the function and its increasing intervals
noncomputable def f (x : ℝ) (phi : ℝ) : ℝ := sin (2 * x + phi)

-- Finding the increasing intervals of f on the interval [0, π]
theorem increasing_intervals (phi : ℝ) 
  (h_phi: phi = -3 * π / 4) :
  ∀ x, (0 ≤ x ∧ x ≤ π) → 
    (π / 8 ≤ x ∧ x ≤ 5 * π / 8) :=
by
  sorry

end find_phi_increasing_intervals_l178_178291


namespace arithmetic_expression_result_l178_178100

theorem arithmetic_expression_result :
  (24 / (8 + 2 - 5)) * 7 = 33.6 :=
by
  sorry

end arithmetic_expression_result_l178_178100


namespace max_ab_l178_178829

theorem max_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 4 * b = 8) :
  ab ≤ 4 :=
sorry

end max_ab_l178_178829


namespace sine_shift_l178_178823

variable (m : ℝ)

theorem sine_shift (h : Real.sin 5.1 = m) : Real.sin 365.1 = m :=
by
  sorry

end sine_shift_l178_178823


namespace wrongly_recorded_height_l178_178353

theorem wrongly_recorded_height 
  (avg_incorrect : ℕ → ℕ → ℕ)
  (avg_correct : ℕ → ℕ → ℕ)
  (boy_count : ℕ)
  (incorrect_avg_height : ℕ) 
  (correct_avg_height : ℕ) 
  (actual_height : ℕ) 
  (correct_total_height : ℕ) 
  (incorrect_total_height: ℕ)
  (x : ℕ) :
  avg_incorrect boy_count incorrect_avg_height = incorrect_total_height →
  avg_correct boy_count correct_avg_height = correct_total_height →
  incorrect_total_height - x + actual_height = correct_total_height →
  x = 176 := 
by 
  intros h1 h2 h3
  sorry

end wrongly_recorded_height_l178_178353


namespace nine_chapters_compensation_difference_l178_178991

noncomputable def pig_consumes (x : ℝ) := x
noncomputable def sheep_consumes (x : ℝ) := 2 * x
noncomputable def horse_consumes (x : ℝ) := 4 * x
noncomputable def cow_consumes (x : ℝ) := 8 * x

theorem nine_chapters_compensation_difference :
  ∃ (x : ℝ), 
    cow_consumes x + horse_consumes x + sheep_consumes x + pig_consumes x = 9 ∧
    (horse_consumes x - pig_consumes x) = 9 / 5 :=
by
  sorry

end nine_chapters_compensation_difference_l178_178991


namespace number_of_balloons_Allan_bought_l178_178589

theorem number_of_balloons_Allan_bought 
  (initial_balloons final_balloons : ℕ) 
  (h1 : initial_balloons = 5) 
  (h2 : final_balloons = 8) : 
  final_balloons - initial_balloons = 3 := 
  by 
  sorry

end number_of_balloons_Allan_bought_l178_178589


namespace find_other_root_l178_178119

theorem find_other_root (m : ℝ) (α : ℝ) :
  (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C m * Polynomial.X - Polynomial.C 10 = 0) →
  (α = -5) →
  ∃ β : ℝ, (α + β = -m) ∧ (α * β = -10) :=
by 
  sorry

end find_other_root_l178_178119


namespace tan_ratio_l178_178610

-- Definitions of the problem conditions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to the angles

-- The given equation condition
axiom h : a * Real.cos B - b * Real.cos A = (4 / 5) * c

-- The goal is to prove the value of tan(A) / tan(B)
theorem tan_ratio (A B C : ℝ) (a b c : ℝ) (h : a * Real.cos B - b * Real.cos A = (4 / 5) * c) :
  Real.tan A / Real.tan B = 9 :=
sorry

end tan_ratio_l178_178610


namespace find_value_of_k_l178_178927

noncomputable def line_parallel_and_point_condition (k : ℝ) :=
  ∃ (m : ℝ), m = -5/4 ∧ (22 - (-8)) / (k - 3) = m

theorem find_value_of_k : ∃ k : ℝ, line_parallel_and_point_condition k ∧ k = -21 :=
by
  sorry

end find_value_of_k_l178_178927


namespace area_of_fig_eq_2_l178_178892

noncomputable def area_of_fig : ℝ :=
  - ∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem area_of_fig_eq_2 : area_of_fig = 2 :=
by
  sorry

end area_of_fig_eq_2_l178_178892


namespace arithmetic_expression_l178_178427

theorem arithmetic_expression :
  (30 / (10 + 2 - 5) + 4) * 7 = 58 :=
by
  sorry

end arithmetic_expression_l178_178427


namespace prob_join_provincial_team_expected_value_ξ_l178_178849

-- Define the probability of ranking in the top 20
def prob_top_20 : ℝ := 1 / 4

-- Define the condition that a student can join the provincial team
def join_provincial_team (competitions : ℕ → Prop) (n : ℕ) : Prop :=
  ∃ (i j : ℕ), i < j ∧ j < n ∧ competitions i ∧ competitions j

-- Statement for the first part of the problem
theorem prob_join_provincial_team :
  ∀ (competitions : ℕ → Prop),
    (∀ i, ProbabilityTheory.Independent (λ j, competitions j)) →
    (∀ i, ProbabilityTheory.prob (competitions i) = prob_top_20) →
    (ProbabilityTheory.prob (λ w, join_provincial_team competitions 5 w) = 67 / 256) := sorry

-- Define the random variable ξ as the number of competitions participated
def ξ (competitions : ℕ → Prop) : ℕ := if join_provincial_team competitions 5 then 5 else 2

-- Statement for the second part of the problem
theorem expected_value_ξ :
  ∀ (competitions : ℕ → Prop),
    (∀ i, ProbabilityTheory.Independent (λ j, competitions j)) →
    (∀ i, ProbabilityTheory.prob (competitions i) = prob_top_20) →
    (MeasureTheory.conditionalExpectation (ProbabilityTheory.Measure_m0_measurable_space prob_measurable) (ξ competitions) 
    = (4 / 3)) := sorry

end prob_join_provincial_team_expected_value_ξ_l178_178849


namespace trajectory_of_moving_circle_l178_178457

-- Define the two given circles C1 and C2
def C1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 2)^2 + p.2^2 = 1}
def C2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 81}

-- Define a moving circle P with center P_center and radius r
structure Circle (α : Type) := 
(center : α × α) 
(radius : ℝ)

def isExternallyTangentTo (P : Circle ℝ) (C : Set (ℝ × ℝ)) :=
  ∃ k ∈ C, (P.center.1 - k.1)^2 + (P.center.2 - k.2)^2 = (P.radius + 1)^2

def isInternallyTangentTo (P : Circle ℝ) (C : Set (ℝ × ℝ)) :=
  ∃ k ∈ C, (P.center.1 - k.1)^2 + (P.center.2 - k.2)^2 = (9 - P.radius)^2

-- Formulate the problem statement
theorem trajectory_of_moving_circle :
  ∀ P : Circle ℝ, 
  isExternallyTangentTo P C1 → 
  isInternallyTangentTo P C2 → 
  (P.center.1^2 / 25 + P.center.2^2 / 21 = 1) := 
sorry

end trajectory_of_moving_circle_l178_178457


namespace aaron_and_carson_scoops_l178_178094

def initial_savings (a c : ℕ) : Prop :=
  a = 150 ∧ c = 150

def total_savings (t a c : ℕ) : Prop :=
  t = a + c

def restaurant_expense (r t : ℕ) : Prop :=
  r = 3 * t / 4

def service_charge_inclusive (r sc : ℕ) : Prop :=
  r = sc * 115 / 100

def remaining_money (t r rm : ℕ) : Prop :=
  rm = t - r

def money_left (al cl : ℕ) : Prop :=
  al = 4 ∧ cl = 4

def ice_cream_scoop_cost (s : ℕ) : Prop :=
  s = 4

def total_scoops (rm ml s scoop_total : ℕ) : Prop :=
  scoop_total = (rm - (ml - 4 - 4)) / s

theorem aaron_and_carson_scoops :
  ∃ a c t r sc rm al cl s scoop_total, initial_savings a c ∧
  total_savings t a c ∧
  restaurant_expense r t ∧
  service_charge_inclusive r sc ∧
  remaining_money t r rm ∧
  money_left al cl ∧
  ice_cream_scoop_cost s ∧
  total_scoops rm (al + cl) s scoop_total ∧
  scoop_total = 16 :=
sorry

end aaron_and_carson_scoops_l178_178094


namespace count_distinct_lines_l178_178666

-- Define a 4-by-4 grid of lattice points
def grid_points := finset (ℕ × ℕ)

-- The set of all points in a 4-by-4 grid
def four_by_four_grid : grid_points :=
  {(0, 0), (0, 1), (0, 2), (0, 3),
   (1, 0), (1, 1), (1, 2), (1, 3),
   (2, 0), (2, 1), (2, 2), (2, 3),
   (3, 0), (3, 1), (3, 2), (3, 3)}.to_finset

-- A line passing through at least two points
def line (p1 p2 : ℕ × ℕ) : set (ℕ × ℕ) :=
  {p : ℕ × ℕ | ∃ λ : ℚ, ∃ b : ℚ, (p.2 : ℚ) = λ * (p.1 : ℚ) + b}

noncomputable theory

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 50. -/
theorem count_distinct_lines (grid : grid_points) (h : grid = four_by_four_grid) :
  ∃ n, n = 50 :=
by
  sorry

end count_distinct_lines_l178_178666


namespace intersection_of_A_and_B_l178_178861

def setA : Set ℝ := { x | x ≤ 4 }
def setB : Set ℝ := { x | x ≥ 1/2 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | 1/2 ≤ x ∧ x ≤ 4 } := by
  sorry

end intersection_of_A_and_B_l178_178861


namespace numbers_difference_l178_178543

theorem numbers_difference (A B C : ℝ) (h1 : B = 10) (h2 : B - A = C - B) (h3 : A * B = 85) (h4 : B * C = 115) : 
  B - A = 1.5 ∧ C - B = 1.5 :=
by
  sorry

end numbers_difference_l178_178543


namespace power_of_same_base_power_of_different_base_l178_178550

theorem power_of_same_base (a n : ℕ) (h : ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m) :
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ a^n = (a^k)^m :=
  sorry

theorem power_of_different_base (a n : ℕ) : ∃ (b m : ℕ), a^n = b^m :=
  sorry

end power_of_same_base_power_of_different_base_l178_178550


namespace complete_circle_formed_l178_178762

theorem complete_circle_formed (t : ℝ) : (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ k : ℤ, θ = k * π) → t = π :=
by
  sorry

end complete_circle_formed_l178_178762


namespace sqrt_expr_is_integer_l178_178436

theorem sqrt_expr_is_integer (x : ℤ) (n : ℤ) (h : n^2 = x^2 - x + 1) : x = 0 ∨ x = 1 := by
  sorry

end sqrt_expr_is_integer_l178_178436


namespace value_of_a_l178_178618

theorem value_of_a (a : ℝ) (h : (a - 3) * x ^ |a - 2| + 4 = 0) : a = 1 :=
by
  sorry

end value_of_a_l178_178618


namespace Shara_shells_total_l178_178182

def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

theorem Shara_shells_total : (initial_shells + (shells_per_day * days) + shells_fourth_day) = 41 := by
  sorry

end Shara_shells_total_l178_178182


namespace noah_total_wattage_l178_178171

def bedroom_wattage := 6
def office_wattage := 3 * bedroom_wattage
def living_room_wattage := 4 * bedroom_wattage
def hours_on := 2

theorem noah_total_wattage : 
  bedroom_wattage * hours_on + 
  office_wattage * hours_on + 
  living_room_wattage * hours_on = 96 := by
  sorry

end noah_total_wattage_l178_178171


namespace isosceles_right_triangle_area_l178_178199

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = 6 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 18 := by 
  sorry

end isosceles_right_triangle_area_l178_178199


namespace minute_hand_angle_45min_l178_178286

theorem minute_hand_angle_45min
  (duration : ℝ)
  (h1 : duration = 45) :
  (-(3 / 4) * 2 * Real.pi = - (3 * Real.pi / 2)) :=
by
  sorry

end minute_hand_angle_45min_l178_178286


namespace percentage_of_mothers_l178_178852

open Real

-- Define the constants based on the conditions provided.
def P : ℝ := sorry -- Total number of parents surveyed
def M : ℝ := sorry -- Number of mothers
def F : ℝ := sorry -- Number of fathers

-- The equations derived from the conditions.
axiom condition1 : M + F = P
axiom condition2 : (1/8)*M + (1/4)*F = 17.5/100 * P

-- The proof goal: to show the percentage of mothers.
theorem percentage_of_mothers :
  M / P = 3 / 5 :=
by
  -- Proof goes here
  sorry

end percentage_of_mothers_l178_178852


namespace number_of_first_grade_students_l178_178712

noncomputable def sampling_ratio (total_students : ℕ) (sampled_students : ℕ) : ℚ :=
  sampled_students / total_students

noncomputable def num_first_grade_selected (first_grade_students : ℕ) (ratio : ℚ) : ℚ :=
  ratio * first_grade_students

theorem number_of_first_grade_students
  (total_students : ℕ)
  (sampled_students : ℕ)
  (first_grade_students : ℕ)
  (h_total : total_students = 2400)
  (h_sampled : sampled_students = 100)
  (h_first_grade : first_grade_students = 840)
  : num_first_grade_selected first_grade_students (sampling_ratio total_students sampled_students) = 35 := by
  sorry

end number_of_first_grade_students_l178_178712


namespace specialCollectionAtEndOfMonth_l178_178787

noncomputable def specialCollectionBooksEndOfMonth (initialBooks loanedBooks returnedPercentage : ℕ) :=
  initialBooks - (loanedBooks - loanedBooks * returnedPercentage / 100)

theorem specialCollectionAtEndOfMonth :
  specialCollectionBooksEndOfMonth 150 80 65 = 122 :=
by
  sorry

end specialCollectionAtEndOfMonth_l178_178787


namespace distance_between_cities_l178_178514

theorem distance_between_cities (S : ℕ) 
  (h : ∀ x : ℕ, x ≤ S → gcd x (S - x) ∈ {1, 3, 13}) : S = 39 :=
sorry

end distance_between_cities_l178_178514


namespace find_y_value_l178_178126

-- Define the linear relationship
def linear_eq (k b x : ℝ) : ℝ := k * x + b

-- Given conditions
variables (k b : ℝ)
axiom h1 : linear_eq k b 0 = -1
axiom h2 : linear_eq k b (1/2) = 2

-- Prove that the value of y when x = -1/2 is -4
theorem find_y_value : linear_eq k b (-1/2) = -4 :=
by sorry

end find_y_value_l178_178126


namespace sum_of_triangulars_iff_sum_of_squares_l178_178348

-- Definitions of triangular numbers and sums of squares
def isTriangular (n : ℕ) : Prop := ∃ k, n = k * (k + 1) / 2
def isSumOfTwoTriangulars (m : ℕ) : Prop := ∃ x y, m = (x * (x + 1) / 2) + (y * (y + 1) / 2)
def isSumOfTwoSquares (n : ℕ) : Prop := ∃ a b, n = a * a + b * b

-- Main theorem statement
theorem sum_of_triangulars_iff_sum_of_squares (m : ℕ) (h_pos : 0 < m) : 
  isSumOfTwoTriangulars m ↔ isSumOfTwoSquares (4 * m + 1) :=
sorry

end sum_of_triangulars_iff_sum_of_squares_l178_178348


namespace find_f_neg_2017_l178_178288

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x
axiom log_function : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem find_f_neg_2017 : f (-2017) = 1 := by
  sorry

end find_f_neg_2017_l178_178288


namespace find_b_l178_178727

def p (x : ℝ) : ℝ := 2 * x - 3
def q (x : ℝ) (b : ℝ) : ℝ := 5 * x - b

theorem find_b (b : ℝ) (h : p (q 3 b) = 13) : b = 7 :=
by sorry

end find_b_l178_178727


namespace heaviest_object_can_be_weighed_is_40_number_of_different_weights_is_40_l178_178444

def weights : List ℕ := [1, 3, 9, 27]

theorem heaviest_object_can_be_weighed_is_40 : 
  List.sum weights = 40 :=
by
  sorry

theorem number_of_different_weights_is_40 :
  List.range (List.sum weights) = List.range 40 :=
by
  sorry

end heaviest_object_can_be_weighed_is_40_number_of_different_weights_is_40_l178_178444


namespace total_fruits_consumed_l178_178173

def starting_cherries : ℝ := 16.5
def remaining_cherries : ℝ := 6.3

def starting_strawberries : ℝ := 10.7
def remaining_strawberries : ℝ := 8.4

def starting_blueberries : ℝ := 20.2
def remaining_blueberries : ℝ := 15.5

theorem total_fruits_consumed 
  (sc : ℝ := starting_cherries)
  (rc : ℝ := remaining_cherries)
  (ss : ℝ := starting_strawberries)
  (rs : ℝ := remaining_strawberries)
  (sb : ℝ := starting_blueberries)
  (rb : ℝ := remaining_blueberries) :
  (sc - rc) + (ss - rs) + (sb - rb) = 17.2 := by
  sorry

end total_fruits_consumed_l178_178173


namespace value_three_std_devs_less_than_mean_l178_178851

-- Define the given conditions as constants.
def mean : ℝ := 16.2
def std_dev : ℝ := 2.3

-- Translate the question into a proof statement.
theorem value_three_std_devs_less_than_mean : mean - 3 * std_dev = 9.3 :=
by sorry

end value_three_std_devs_less_than_mean_l178_178851


namespace relationship_x1_x2_l178_178293

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem relationship_x1_x2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f x1 = f x2) :
  x1 > 2 - x2 :=
begin
  sorry
end

end relationship_x1_x2_l178_178293


namespace less_than_subtraction_l178_178552

-- Define the numbers as real numbers
def a : ℝ := 47.2
def b : ℝ := 0.5

-- Theorem statement
theorem less_than_subtraction : a - b = 46.7 :=
by
  sorry

end less_than_subtraction_l178_178552


namespace solve_abs_eq_l178_178749

theorem solve_abs_eq (x : ℝ) : (|x - 3| = 5 - x) ↔ (x = 4) := 
by
  sorry

end solve_abs_eq_l178_178749


namespace smallest_b_greater_than_l178_178778

theorem smallest_b_greater_than (a b : ℤ) (h₁ : 9 < a) (h₂ : a < 21) (h₃ : 10 / b ≥ 2 / 3) (h₄ : b < 31) : 14 < b :=
sorry

end smallest_b_greater_than_l178_178778


namespace younger_by_17_l178_178057

variables (A B C : ℕ)

-- Given condition
axiom age_condition : A + B = B + C + 17

-- To show
theorem younger_by_17 : A - C = 17 :=
by
  sorry

end younger_by_17_l178_178057


namespace student_correct_answers_l178_178092

variable (C I : ℕ)

theorem student_correct_answers :
  C + I = 100 ∧ C - 2 * I = 76 → C = 92 :=
by
  intros h
  sorry

end student_correct_answers_l178_178092


namespace find_x_plus_y_l178_178835

-- Define the vectors
def vector_a : ℝ × ℝ := (1, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_c (y : ℝ) : ℝ × ℝ := (-1, y)

-- Define the conditions
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v2.1 = k * v1.1 ∧ v2.2 = k * v1.2

-- State the theorem
theorem find_x_plus_y (x y : ℝ)
  (h1 : perpendicular vector_a (vector_b x))
  (h2 : parallel vector_a (vector_c y)) :
  x + y = 1 :=
sorry

end find_x_plus_y_l178_178835


namespace cookies_per_person_l178_178700

theorem cookies_per_person (cookies_per_bag : ℕ) (bags : ℕ) (damaged_cookies_per_bag : ℕ) (people : ℕ) (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_each : ℕ) :
  (cookies_per_bag = 738) →
  (bags = 295) →
  (damaged_cookies_per_bag = 13) →
  (people = 125) →
  (total_cookies = cookies_per_bag * bags) →
  (remaining_cookies = total_cookies - (damaged_cookies_per_bag * bags)) →
  (cookies_each = remaining_cookies / people) →
  cookies_each = 1711 :=
by
  sorry 

end cookies_per_person_l178_178700


namespace sum_of_f10_values_l178_178197

noncomputable def f : ℕ → ℝ := sorry

axiom f_cond1 : f 1 = 4

axiom f_cond2 : ∀ (m n : ℕ), m ≥ n → f (m + n) + f (m - n) = (f (2 * m) + f (2 * n)) / 2

theorem sum_of_f10_values : f 10 = 400 :=
sorry

end sum_of_f10_values_l178_178197


namespace isosceles_triangle_vertex_angle_l178_178320

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (h_iso : A = B ∨ A = C ∨ B = C) (h_sum : A + B + C = 180) (h_one_angle : A = 50 ∨ B = 50 ∨ C = 50) :
  A = 80 ∨ B = 80 ∨ C = 80 ∨ A = 50 ∨ B = 50 ∨ C = 50 :=
by
  sorry

end isosceles_triangle_vertex_angle_l178_178320


namespace min_value_of_sum_l178_178125

theorem min_value_of_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2 * a + b) : a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_sum_l178_178125


namespace poles_needed_l178_178566

theorem poles_needed (L W : ℕ) (dist : ℕ)
  (hL : L = 90) (hW : W = 40) (hdist : dist = 5) :
  (2 * (L + W)) / dist = 52 :=
by 
  sorry

end poles_needed_l178_178566


namespace acute_triangle_tangent_difference_range_l178_178020

theorem acute_triangle_tangent_difference_range {A B C a b c : ℝ} 
    (h1 : a^2 + b^2 > c^2) (h2 : b^2 + c^2 > a^2) (h3 : c^2 + a^2 > b^2)
    (hb2_minus_ha2_eq_ac : b^2 - a^2 = a * c) :
    1 < (1 / Real.tan A - 1 / Real.tan B) ∧ (1 / Real.tan A - 1 / Real.tan B) < (2 * Real.sqrt 3 / 3) :=
by
  sorry

end acute_triangle_tangent_difference_range_l178_178020


namespace num_lines_passing_through_4x4_grid_l178_178647

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 66. -/
theorem num_lines_passing_through_4x4_grid : 
  let p := 4 * 4 in
  let total_point_pairs := p * (p - 1) / 2 in
  let horizontal_lines_count := 4 in
  let vertical_lines_count := 4 in
  let diagonal_lines_4_count := 2 in
  let diagonal_lines_3_count := 2 in
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count in
  (total_point_pairs - overcount_correction) = 66 :=
by
  let p := 4 * 4
  let total_point_pairs := p * (p - 1) / 2
  let horizontal_lines_count := 4
  let vertical_lines_count := 4
  let diagonal_lines_4_count := 2
  let diagonal_lines_3_count := 2
  let overcount_correction := 
    5 * (horizontal_lines_count + vertical_lines_count + diagonal_lines_4_count) + 
    2 * diagonal_lines_3_count
  have h_correct_count : total_point_pairs - overcount_correction = 66, from sorry
  exact h_correct_count

end num_lines_passing_through_4x4_grid_l178_178647


namespace maximum_value_of_f_l178_178114

theorem maximum_value_of_f (x : ℝ) (h : x^4 + 36 ≤ 13 * x^2) : 
  ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), (x^4 + 36 ≤ 13 * x^2) → (x^3 - 3 * x ≤ m) :=
sorry

end maximum_value_of_f_l178_178114


namespace missed_number_l178_178791

/-
  A student finds the sum \(1 + 2 + 3 + \cdots\) as his patience runs out. 
  He found the sum as 575. When the teacher declared the result wrong, 
  the student realized that he missed a number.
  Prove that the number he missed is 20.
-/

theorem missed_number (n : ℕ) (S_incorrect S_correct S_missed : ℕ) 
  (h1 : S_incorrect = 575)
  (h2 : S_correct = n * (n + 1) / 2)
  (h3 : S_correct = 595)
  (h4 : S_missed = S_correct - S_incorrect) :
  S_missed = 20 :=
sorry

end missed_number_l178_178791


namespace value_of_expression_when_x_is_3_l178_178915

theorem value_of_expression_when_x_is_3 :
  (3^6 - 6*3 = 711) :=
by
  sorry

end value_of_expression_when_x_is_3_l178_178915


namespace system_solution_xz_y2_l178_178604

theorem system_solution_xz_y2 (x y z : ℝ) (k : ℝ)
  (h : (x + 2 * k * y + 4 * z = 0) ∧
       (4 * x + k * y - 3 * z = 0) ∧
       (3 * x + 5 * y - 2 * z = 0) ∧
       x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ k = 95 / 12) :
  (x * z) / (y ^ 2) = 10 :=
by sorry

end system_solution_xz_y2_l178_178604


namespace thompson_class_average_l178_178339

theorem thompson_class_average
  (n : ℕ) (initial_avg : ℚ) (final_avg : ℚ) (bridget_index : ℕ) (first_n_score_sum : ℚ)
  (total_students : ℕ) (final_score_sum : ℚ)
  (h1 : n = 17) -- Number of students initially graded
  (h2 : initial_avg = 76) -- Average score of the first 17 students
  (h3 : final_avg = 78) -- Average score after adding Bridget's test
  (h4 : bridget_index = 18) -- Total number of students
  (h5 : total_students = 18) -- Total number of students
  (h6 : first_n_score_sum = n * initial_avg) -- Total score of the first 17 students
  (h7 : final_score_sum = total_students * final_avg) -- Total score of the 18 students):
  -- Bridget's score
  (bridgets_score : ℚ) :
  bridgets_score = final_score_sum - first_n_score_sum :=
sorry

end thompson_class_average_l178_178339


namespace neg_p_l178_178000

theorem neg_p (p : ∀ x : ℝ, x^2 ≥ 0) : ∃ x : ℝ, x^2 < 0 := 
sorry

end neg_p_l178_178000


namespace unique_integer_solution_l178_178041

theorem unique_integer_solution (x y z : ℤ) (h : 2 * x^2 + 3 * y^2 = z^2) : x = 0 ∧ y = 0 ∧ z = 0 :=
by {
  sorry
}

end unique_integer_solution_l178_178041


namespace cubes_sum_l178_178215

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l178_178215


namespace necessary_but_not_sufficient_l178_178412

variable (x : ℝ)

theorem necessary_but_not_sufficient (h : x > 2) : x > 1 ∧ ¬ (x > 1 → x > 2) :=
by
  sorry

end necessary_but_not_sufficient_l178_178412


namespace solve_for_k_l178_178462

theorem solve_for_k (t k : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : t = 105) : k = 221 :=
by
  sorry

end solve_for_k_l178_178462


namespace Philip_total_animals_l178_178879

-- Total number of animals computation
def total_animals (cows ducks pigs : Nat) : Nat :=
  cows + ducks + pigs

-- Number of ducks computation
def number_of_ducks (cows : Nat) : Nat :=
  cows + cows / 2 -- 50% more ducks than cows

-- Number of pigs computation
def number_of_pigs (total_ducks_cows : Nat) : Nat :=
  total_ducks_cows / 5 -- one-fifth of total ducks and cows

theorem Philip_total_animals :
  let cows := 20 in
  let ducks := number_of_ducks cows in
  let total_ducks_cows := cows + ducks in
  let pigs := number_of_pigs total_ducks_cows in
  total_animals cows ducks pigs = 60 :=
by
  sorry

end Philip_total_animals_l178_178879


namespace closest_point_on_line_l178_178602

open Real

theorem closest_point_on_line (x y : ℝ) (h_line : y = 4 * x - 3) (h_closest : ∀ p : ℝ × ℝ, (p.snd - -1)^2 + (p.fst - 2)^2 ≥ (y - -1)^2 + (x - 2)^2) :
  x = 10 / 17 ∧ y = 31 / 17 :=
sorry

end closest_point_on_line_l178_178602


namespace johns_weekly_earnings_increase_l178_178326

def combined_percentage_increase (initial final : ℕ) : ℕ :=
  ((final - initial) * 100) / initial

theorem johns_weekly_earnings_increase :
  combined_percentage_increase 40 60 = 50 :=
by
  sorry

end johns_weekly_earnings_increase_l178_178326


namespace angle_A_is_equilateral_l178_178153

namespace TriangleProof

variables {A B C : ℝ} {a b c : ℝ}

-- Given condition (a+b+c)(a-b-c) + 3bc = 0
def condition1 (a b c : ℝ) : Prop := (a + b + c) * (a - b - c) + 3 * b * c = 0

-- Given condition a = 2c * cos B
def condition2 (a c B : ℝ) : Prop := a = 2 * c * Real.cos B

-- Prove that if (a+b+c)(a-b-c) + 3bc = 0, then A = π / 3
theorem angle_A (h1 : condition1 a b c) : A = Real.pi / 3 :=
sorry

-- Prove that if a = 2c * cos B and A = π / 3, then ∆ ABC is an equilateral triangle
theorem is_equilateral (h2 : condition2 a c B) (hA : A = Real.pi / 3) : 
  b = c ∧ a = b ∧ B = C :=
sorry

end TriangleProof

end angle_A_is_equilateral_l178_178153


namespace lines_in_4x4_grid_l178_178674

theorem lines_in_4x4_grid : 
  let grid_points := finset.univ.product finset.univ
  let total_points := 16
  let pairs_of_points := total_points.choose 2
  let horizontal_lines := 4
  let vertical_lines := 4
  let diagonal_lines := 2
  let lines_through_four_points := horizontal_lines + vertical_lines + diagonal_lines
  let correction := lines_through_four_points * (4.choose 2 - 1)
  let number_of_lines := pairs_of_points - correction
  in number_of_lines = 70 := 
by {
  sorry
}

end lines_in_4x4_grid_l178_178674


namespace distance_between_cities_l178_178518

-- Conditions Initialization
variable (A B : ℝ)
variable (S : ℕ)
variable (x : ℕ)
variable (gcd_values : ℕ)

-- Conditions:
-- 1. The distance between cities A and B is an integer.
-- 2. Markers are placed every kilometer.
-- 3. At each marker, one side has distance to city A, and the other has distance to city B.
-- 4. The GCDs observed were only 1, 3, and 13.

-- Given these conditions, we prove the distance is exactly 39 kilometers.
theorem distance_between_cities (h1 : gcd_values = 1 ∨ gcd_values = 3 ∨ gcd_values = 13)
    (h2 : S = Nat.lcm 1 (Nat.lcm 3 13)) :
    S = 39 := by 
  sorry

end distance_between_cities_l178_178518


namespace warehouse_can_release_100kg_l178_178538

theorem warehouse_can_release_100kg (a b c d : ℕ) : 
  24 * a + 23 * b + 17 * c + 16 * d = 100 → True :=
by
  sorry

end warehouse_can_release_100kg_l178_178538


namespace radioactive_ball_identification_l178_178964

/-- Given 11 balls where exactly 2 are radioactive and each test can detect the presence of 
    radioactive balls in a group, prove that fewer than 7 tests are insufficient to guarantee 
    identification of the 2 radioactive balls, but 7 tests are always sufficient. -/
theorem radioactive_ball_identification (total_balls radioactive_balls : ℕ) (tests : ℕ) 
  (test_group : set ℕ → Prop) :
  total_balls = 11 → radioactive_balls = 2 → 
  (∀ (S : set ℕ), test_group S ↔ ∃ (x : ℕ), x ∈ S ∧ is_radioactive x) →
  (tests < 7 → ¬identify_radioactive_balls total_balls radioactive_balls test_group tests) ∧ 
  (tests = 7 → identify_radioactive_balls total_balls radioactive_balls test_group tests) := 
by
  intros htotal hradioactive htest
  sorry

end radioactive_ball_identification_l178_178964


namespace length_of_room_l178_178542

theorem length_of_room (area_in_sq_inches : ℕ) (length_of_side_in_feet : ℕ) (h1 : area_in_sq_inches = 14400)
  (h2 : length_of_side_in_feet * length_of_side_in_feet = area_in_sq_inches / 144) : length_of_side_in_feet = 10 :=
  by
  sorry

end length_of_room_l178_178542


namespace equivalent_statements_l178_178069

-- Definitions
variables (P Q : Prop)

-- Original statement
def original_statement := P → Q

-- Statements
def statement_I := P → Q
def statement_II := Q → P
def statement_III := ¬ Q → ¬ P
def statement_IV := ¬ P ∨ Q

-- Proof problem
theorem equivalent_statements : 
  (statement_III P Q ∧ statement_IV P Q) ↔ original_statement P Q :=
sorry

end equivalent_statements_l178_178069


namespace possible_values_of_expression_l178_178963

theorem possible_values_of_expression (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  ∃ (vals : Finset ℤ), vals = {6, 2, 0, -2, -6} ∧
  (∃ val ∈ vals, val = (if p > 0 then 1 else -1) + 
                         (if q > 0 then 1 else -1) + 
                         (if r > 0 then 1 else -1) + 
                         (if s > 0 then 1 else -1) + 
                         (if (p * q * r) > 0 then 1 else -1) + 
                         (if (p * r * s) > 0 then 1 else -1)) :=
by
  sorry

end possible_values_of_expression_l178_178963


namespace smallest_possible_value_of_n_l178_178184

theorem smallest_possible_value_of_n 
  {a b c m n : ℕ} 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (hc_pos : c > 0) 
  (h_ordering : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c = 3010) 
  (h_factorial : a.factorial * b.factorial * c.factorial = m * 10^n) 
  (h_m_not_div_10 : ¬ (10 ∣ m)) 
  : n = 746 := 
sorry

end smallest_possible_value_of_n_l178_178184


namespace lines_in_4_by_4_grid_l178_178685

theorem lines_in_4_by_4_grid : 
  let n := 4 in
  number_of_lines_at_least_two_points (grid_of_lattice_points n) = 96 :=
by sorry

end lines_in_4_by_4_grid_l178_178685


namespace number_of_correct_propositions_is_zero_l178_178795

-- Defining the propositions as functions
def proposition1 (f : ℝ → ℝ) (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
  (increasing_neg : ∀ x < 0, f x ≤ f (x + 1)) : Prop :=
  ∀ x1 x2, x1 ≤ x2 → f x1 ≤ f x2

def proposition2 (a b : ℝ) (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0) : Prop :=
  b^2 < 8 * a ∧ (a > 0 ∨ (a = 0 ∧ b = 0))

def proposition3 : Prop :=
  ∀ x, (x ≥ 1 → (x^2 - 2 * x - 3) ≥ (x^2 - 2 * (x + 1) - 3))

-- The main theorem to prove
theorem number_of_correct_propositions_is_zero :
  ∀ (f : ℝ → ℝ)
    (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
    (increasing_neg : ∀ x < 0, f x ≤ f (x + 1))
    (a b : ℝ)
    (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0),
    (¬ proposition1 f increasing_pos increasing_neg ∧
     ¬ proposition2 a b no_intersection ∧
     ¬ proposition3) :=
by
  sorry

end number_of_correct_propositions_is_zero_l178_178795


namespace initial_average_runs_l178_178188

theorem initial_average_runs (A : ℝ) (h : 10 * A + 65 = 11 * (A + 3)) : A = 32 :=
  by sorry

end initial_average_runs_l178_178188


namespace count_distinct_lines_l178_178670

-- Define a 4-by-4 grid of lattice points
def grid_points := finset (ℕ × ℕ)

-- The set of all points in a 4-by-4 grid
def four_by_four_grid : grid_points :=
  {(0, 0), (0, 1), (0, 2), (0, 3),
   (1, 0), (1, 1), (1, 2), (1, 3),
   (2, 0), (2, 1), (2, 2), (2, 3),
   (3, 0), (3, 1), (3, 2), (3, 3)}.to_finset

-- A line passing through at least two points
def line (p1 p2 : ℕ × ℕ) : set (ℕ × ℕ) :=
  {p : ℕ × ℕ | ∃ λ : ℚ, ∃ b : ℚ, (p.2 : ℚ) = λ * (p.1 : ℚ) + b}

noncomputable theory

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid of lattice points is 50. -/
theorem count_distinct_lines (grid : grid_points) (h : grid = four_by_four_grid) :
  ∃ n, n = 50 :=
by
  sorry

end count_distinct_lines_l178_178670


namespace find_digits_l178_178176

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 7
noncomputable def C : ℕ := 1

def row_sums_equal (A B C : ℕ) : Prop :=
  (4 + 2 + 3 = 9) ∧
  (A + 6 + 1 = 9) ∧
  (1 + 2 + 6 = 9) ∧
  (B + 2 + C = 9)

def column_sums_equal (A B C : ℕ) : Prop :=
  (4 + 1 + B = 12) ∧
  (2 + A + 2 = 6 + 3 + 1) ∧
  (3 + 1 + 6 + C = 12)

def red_cells_sum_equals_row_sum (A B C : ℕ) : Prop :=
  (A + B + C = 9)

theorem find_digits :
  ∃ (A B C : ℕ),
    row_sums_equal A B C ∧
    column_sums_equal A B C ∧
    red_cells_sum_equals_row_sum A B C ∧
    100 * A + 10 * B + C = 371 :=
by
  exact ⟨3, 7, 1, ⟨rfl, rfl, rfl, rfl⟩⟩

end find_digits_l178_178176


namespace remainder_of_1234567_div_257_l178_178416

theorem remainder_of_1234567_div_257 : 1234567 % 257 = 123 := by
  sorry

end remainder_of_1234567_div_257_l178_178416


namespace speed_of_stream_l178_178920

theorem speed_of_stream (vs : ℝ) (h : ∀ (d : ℝ), d / (57 - vs) = 2 * (d / (57 + vs))) : vs = 19 :=
by
  sorry

end speed_of_stream_l178_178920


namespace scientific_notation_l178_178875

variables (n : ℕ) (h : n = 505000)

theorem scientific_notation : n = 505000 → "5.05 * 10^5" = "scientific notation of 505000" :=
by
  intro h
  sorry

end scientific_notation_l178_178875


namespace number_of_lines_in_4_by_4_grid_l178_178654

/-- A 4-by-4 grid of lattice points -/
def lattice_points_4x4 : set (ℕ × ℕ) :=
  {(i, j) | i < 4 ∧ j < 4}

/-- A line in the Euclidean plane -/
def is_line (p1 p2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | ∃ λ : ℝ, p = (λ * (p2.1 - p1.1) + p1.1, λ * (p2.2 - p1.2) + p1.2)}

noncomputable def count_lines_through_points (points : set (ℕ × ℕ)) : ℕ :=
  /- counting logic to be implemented -/
  sorry

theorem number_of_lines_in_4_by_4_grid : count_lines_through_points lattice_points_4x4 = 70 :=
  sorry

end number_of_lines_in_4_by_4_grid_l178_178654


namespace josh_total_candies_l178_178995

def josh_initial_candies (initial_candies given_siblings : ℕ) : Prop :=
  ∃ (remaining_1 best_friend josh_eats share_others : ℕ),
    (remaining_1 = initial_candies - given_siblings) ∧
    (best_friend = remaining_1 / 2) ∧
    (josh_eats = 16) ∧
    (share_others = 19) ∧
    (remaining_1 = 2 * (josh_eats + share_others))

theorem josh_total_candies : josh_initial_candies 100 30 :=
by
  sorry

end josh_total_candies_l178_178995


namespace apples_for_pies_l178_178766

-- Define the conditions
def apples_per_pie : ℝ := 4.0
def number_of_pies : ℝ := 126.0

-- Define the expected answer
def number_of_apples : ℝ := number_of_pies * apples_per_pie

-- State the theorem to prove the question == answer given the conditions
theorem apples_for_pies : number_of_apples = 504 :=
by
  -- This is where the proof would go. Currently skipped.
  sorry

end apples_for_pies_l178_178766


namespace area_of_rectangle_given_conditions_l178_178090

-- Defining the conditions given in the problem
variables (s d r a : ℝ)

-- Given conditions for the problem
def is_square_inscribed_in_circle (s d : ℝ) := 
  d = s * Real.sqrt 2 ∧ 
  d = 4

def is_circle_inscribed_in_rectangle (r : ℝ) :=
  r = 2

def rectangle_dimensions (length width : ℝ) :=
  length = 2 * width ∧ 
  width = 2

-- The theorem we want to prove
theorem area_of_rectangle_given_conditions :
  ∀ (s d r length width : ℝ),
  is_square_inscribed_in_circle s d →
  is_circle_inscribed_in_rectangle r →
  rectangle_dimensions length width →
  a = length * width →
  a = 8 :=
by
  intros s d r length width h1 h2 h3 h4
  sorry

end area_of_rectangle_given_conditions_l178_178090


namespace real_part_of_solution_is_8_l178_178753

theorem real_part_of_solution_is_8 :
  ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ ((a + b * complex.i) ^ 3 + 2 * (a + b * complex.i) ^ 2 * complex.i - 
    2 * (a + b * complex.i) * complex.i - 8 = 1624 * complex.i) ∧ (a = 8) :=
by
  sorry

end real_part_of_solution_is_8_l178_178753


namespace chantel_final_bracelets_count_l178_178431

def bracelets_made_in_first_5_days : ℕ := 5 * 2

def bracelets_after_giving_away_at_school : ℕ := bracelets_made_in_first_5_days - 3

def bracelets_made_in_next_4_days : ℕ := 4 * 3

def total_bracelets_before_soccer_giveaway : ℕ := bracelets_after_giving_away_at_school + bracelets_made_in_next_4_days

def bracelets_after_giving_away_at_soccer : ℕ := total_bracelets_before_soccer_giveaway - 6

theorem chantel_final_bracelets_count : bracelets_after_giving_away_at_soccer = 13 :=
sorry

end chantel_final_bracelets_count_l178_178431


namespace find_certain_number_l178_178793

theorem find_certain_number (x : ℤ) (h : x + 34 - 53 = 28) : x = 47 :=
by {
  sorry
}

end find_certain_number_l178_178793


namespace area_of_right_triangle_from_roots_l178_178310

theorem area_of_right_triangle_from_roots :
  ∀ (a b : ℝ), (a^2 - 7*a + 12 = 0) → (b^2 - 7*b + 12 = 0) →
  (∃ (area : ℝ), (area = 6) ∨ (area = (3 * real.sqrt 7) / 2)) :=
by
  intros a b ha hb
  sorry

end area_of_right_triangle_from_roots_l178_178310


namespace grandfather_time_difference_l178_178044

-- Definitions based on the conditions
def treadmill_days : ℕ := 4
def miles_per_day : ℕ := 2
def monday_speed : ℕ := 6
def tuesday_speed : ℕ := 3
def wednesday_speed : ℕ := 4
def thursday_speed : ℕ := 3
def walk_speed : ℕ := 3

-- The theorem statement
theorem grandfather_time_difference :
  let monday_time := (miles_per_day : ℚ) / monday_speed
  let tuesday_time := (miles_per_day : ℚ) / tuesday_speed
  let wednesday_time := (miles_per_day : ℚ) / wednesday_speed
  let thursday_time := (miles_per_day : ℚ) / thursday_speed
  let actual_total_time := monday_time + tuesday_time + wednesday_time + thursday_time
  let walk_total_time := (treadmill_days * miles_per_day : ℚ) / walk_speed
  (walk_total_time - actual_total_time) * 60 = 80 := sorry

end grandfather_time_difference_l178_178044


namespace angle_between_strips_l178_178790

theorem angle_between_strips (w : ℝ) (a : ℝ) (angle : ℝ) (h_w : w = 1) (h_area : a = 2) :
  ∃ θ : ℝ, θ = 30 ∧ angle = θ :=
by
  sorry

end angle_between_strips_l178_178790


namespace distance_qr_eq_b_l178_178899

theorem distance_qr_eq_b
  (a b c : ℝ)
  (hP : b = c * Real.cosh (a / c))
  (hQ : ∃ Q : ℝ × ℝ, Q = (0, c) ∧ Q.2 = c * Real.cosh (Q.1 / c))
  : QR = b := by
  sorry

end distance_qr_eq_b_l178_178899


namespace lcm_24_150_is_600_l178_178273

noncomputable def lcm_24_150 : ℕ :=
  let a := 24
  let b := 150
  have h₁ : a = 2^3 * 3 := by sorry
  have h₂ : b = 2 * 3 * 5^2 := by sorry
  Nat.lcm a b

theorem lcm_24_150_is_600 : lcm_24_150 = 600 := by
  -- Use provided primes conditions to derive the result
  sorry

end lcm_24_150_is_600_l178_178273


namespace inequality_1_inequality_2_l178_178483

noncomputable def f (x : ℝ) : ℝ := |x - 2| - 3
noncomputable def g (x : ℝ) : ℝ := |x + 3|

theorem inequality_1 (x : ℝ) : f x < g x ↔ x > -2 := 
by sorry

theorem inequality_2 (a : ℝ) : (∀ x : ℝ, f x < g x + a) ↔ a > 2 := 
by sorry

end inequality_1_inequality_2_l178_178483


namespace distance_to_town_l178_178744

theorem distance_to_town (fuel_efficiency : ℝ) (fuel_used : ℝ) (distance : ℝ) : 
  fuel_efficiency = 70 / 10 → 
  fuel_used = 20 → 
  distance = fuel_efficiency * fuel_used → 
  distance = 140 :=
by
  intros
  sorry

end distance_to_town_l178_178744


namespace temp_drop_of_8_deg_is_neg_8_l178_178845

theorem temp_drop_of_8_deg_is_neg_8 (rise_3_deg : ℤ) (h : rise_3_deg = 3) : ∀ drop_8_deg, drop_8_deg = -8 :=
by
  intros
  sorry

end temp_drop_of_8_deg_is_neg_8_l178_178845


namespace triangle_area_is_correct_l178_178201

-- Define the given conditions
def is_isosceles_right_triangle (h : ℝ) (l : ℝ) : Prop :=
  h = l * sqrt 2

def triangle_hypotenuse := 6 * sqrt 2  -- Given hypotenuse

-- Prove that the area of the triangle is 18 square units
theorem triangle_area_is_correct : 
  ∃ (l : ℝ), is_isosceles_right_triangle triangle_hypotenuse l ∧ (1/2) * l^2 = 18 :=
by
  sorry

end triangle_area_is_correct_l178_178201


namespace ratio_of_marbles_l178_178106

noncomputable def marble_ratio : ℕ :=
  let initial_marbles := 40
  let marbles_after_breakfast := initial_marbles - 3
  let marbles_after_lunch := marbles_after_breakfast - 5
  let marbles_after_moms_gift := marbles_after_lunch + 12
  let final_marbles := 54
  let marbles_given_back_by_Susie := final_marbles - marbles_after_moms_gift
  marbles_given_back_by_Susie / 5

theorem ratio_of_marbles : marble_ratio = 2 := by
  -- proof steps would go here
  sorry

end ratio_of_marbles_l178_178106


namespace correct_operation_l178_178562

theorem correct_operation (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by sorry

end correct_operation_l178_178562


namespace sum_terms_a1_a17_l178_178449

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 :=
sorry

end sum_terms_a1_a17_l178_178449


namespace solution_set_of_inequality_l178_178534

-- Define the conditions and theorem
theorem solution_set_of_inequality (x : ℝ) (hx : x ≠ 0) : (1 / x < x) ↔ ((-1 < x ∧ x < 0) ∨ (1 < x)) :=
by sorry

end solution_set_of_inequality_l178_178534


namespace stephan_cannot_afford_laptop_l178_178873

noncomputable def initial_laptop_price : ℝ := sorry

theorem stephan_cannot_afford_laptop (P₀ : ℝ) (h_rate : 0 < 0.06) (h₁ : initial_laptop_price = P₀) : 
  56358 < P₀ * (1.06)^2 :=
by 
  sorry

end stephan_cannot_afford_laptop_l178_178873


namespace max_n_value_l178_178608

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 1/(a - b) + 1/(b - c) ≥ n / (a - c)) :
  n ≤ 4 := 
sorry

end max_n_value_l178_178608


namespace ellipse_equation_dot_product_constant_l178_178281

-- Given circles F₁ and F₂, and Circle O
noncomputable def circle_F1 (r : ℝ) : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x + 1)^2 + y^2 = r^2 }

noncomputable def circle_F2 (r : ℝ) : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x - 1)^2 + y^2 = (4 - r)^2 }

def circle_O : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in x^2 + y^2 = (12 / 7) }

-- Define the ellipse E
noncomputable def ellipse_E : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x^2) / 4 + (y^2) / 3 = 1 }

-- Define the proof problem for part (1)
theorem ellipse_equation (1 ≤ r ∧ r ≤ 3) :
  ∀ (p : ℝ × ℝ), p ∈ circle_F1 r → p ∈ circle_F2 r → p ∈ ellipse_E :=
sorry

-- Define the vectors and their dot product
noncomputable def vector_dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the proof problem for part (2)
theorem dot_product_constant (A P Q : ℝ × ℝ) (r : ℝ) (hA : A ∈ circle_O)
(hPA : P ∈ ellipse_E) (hQA : Q ∈ ellipse_E) :
  vector_dot_product (⟨P.1 - A.1, P.2 - A.2⟩) (⟨Q.1 - A.1, Q.2 - A.2⟩) = - (12 / 7) :=
sorry

end ellipse_equation_dot_product_constant_l178_178281


namespace circle_diameter_l178_178702

theorem circle_diameter (r : ℝ) (h : r = 4) : 2 * r = 8 := sorry

end circle_diameter_l178_178702


namespace largest_divisor_is_15_l178_178773

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def largest_divisor (n : ℕ) : ℕ :=
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)

theorem largest_divisor_is_15 : ∀ (n : ℕ), n > 0 → is_even n → 15 ∣ largest_divisor n ∧ (∀ m, m ∣ largest_divisor n → m ≤ 15) :=
by
  intros n pos even
  sorry

end largest_divisor_is_15_l178_178773


namespace pulled_pork_sandwiches_l178_178992

/-
  Jack uses 3 cups of ketchup, 1 cup of vinegar, and 1 cup of honey.
  Each burger takes 1/4 cup of sauce.
  Each pulled pork sandwich takes 1/6 cup of sauce.
  Jack makes 8 burgers.
  Prove that Jack can make exactly 18 pulled pork sandwiches.
-/
theorem pulled_pork_sandwiches :
  (3 + 1 + 1) - (8 * (1/4)) = 3 -> 
  3 / (1/6) = 18 :=
sorry

end pulled_pork_sandwiches_l178_178992


namespace sum_of_cubes_l178_178229

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l178_178229


namespace A_share_of_profit_l178_178933

theorem A_share_of_profit
  (A_investment : ℤ) (B_investment : ℤ) (C_investment : ℤ)
  (A_profit_share : ℚ) (B_profit_share : ℚ) (C_profit_share : ℚ)
  (total_profit : ℤ) :
  A_investment = 6300 ∧ B_investment = 4200 ∧ C_investment = 10500 ∧
  A_profit_share = 0.45 ∧ B_profit_share = 0.3 ∧ C_profit_share = 0.25 ∧ 
  total_profit = 12200 →
  A_profit_share * total_profit = 5490 :=
by sorry

end A_share_of_profit_l178_178933


namespace no_non_similar_triangles_with_geometric_angles_l178_178005

theorem no_non_similar_triangles_with_geometric_angles :
  ¬∃ (a r : ℕ), a > 0 ∧ r > 0 ∧ a ≠ a * r ∧ a ≠ a * r * r ∧ a * r ≠ a * r * r ∧
  a + a * r + a * r * r = 180 :=
by
  sorry

end no_non_similar_triangles_with_geometric_angles_l178_178005


namespace g_inverse_sum_l178_178722

-- Define the function g and its inverse
def g (x : ℝ) : ℝ := x ^ 3
noncomputable def g_inv (y : ℝ) : ℝ := y ^ (1/3 : ℝ)

-- State the theorem to be proved
theorem g_inverse_sum : g_inv 8 + g_inv (-64) = -2 := by 
  sorry

end g_inverse_sum_l178_178722


namespace mass_percentage_C_in_CaCO3_is_correct_l178_178429

structure Element where
  name : String
  molar_mass : ℚ

def Ca : Element := ⟨"Ca", 40.08⟩
def C : Element := ⟨"C", 12.01⟩
def O : Element := ⟨"O", 16.00⟩

def molar_mass_CaCO3 : ℚ :=
  Ca.molar_mass + C.molar_mass + 3 * O.molar_mass

def mass_percentage_C_in_CaCO3 : ℚ :=
  (C.molar_mass / molar_mass_CaCO3) * 100

theorem mass_percentage_C_in_CaCO3_is_correct :
  mass_percentage_C_in_CaCO3 = 12.01 :=
by
  sorry

end mass_percentage_C_in_CaCO3_is_correct_l178_178429


namespace negation_proof_l178_178358

theorem negation_proof :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) → (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end negation_proof_l178_178358


namespace Nell_has_123_more_baseball_cards_than_Ace_cards_l178_178034

def Nell_cards_diff (baseball_cards_new : ℕ) (ace_cards_new : ℕ) : ℕ :=
  baseball_cards_new - ace_cards_new

theorem Nell_has_123_more_baseball_cards_than_Ace_cards:
  (Nell_cards_diff 178 55) = 123 :=
by
  -- proof here
  sorry

end Nell_has_123_more_baseball_cards_than_Ace_cards_l178_178034


namespace find_ordered_pair_l178_178952

theorem find_ordered_pair (x y : ℤ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x - y = (x - 2) + (y - 2))
  : (x, y) = (5, 2) := 
sorry

end find_ordered_pair_l178_178952


namespace ants_harvest_remaining_sugar_l178_178406

-- Define the initial conditions
def ants_removal_rate : ℕ := 4
def initial_sugar_amount : ℕ := 24
def hours_passed : ℕ := 3

-- Calculate the correct answer
def remaining_sugar (initial : ℕ) (rate : ℕ) (hours : ℕ) : ℕ :=
  initial - (rate * hours)

def additional_hours_needed (remaining_sugar : ℕ) (rate : ℕ) : ℕ :=
  remaining_sugar / rate

-- The specification of the proof problem
theorem ants_harvest_remaining_sugar :
  additional_hours_needed (remaining_sugar initial_sugar_amount ants_removal_rate hours_passed) ants_removal_rate = 3 :=
by
  -- Proof omitted
  sorry

end ants_harvest_remaining_sugar_l178_178406


namespace largest_n_satisfying_ineq_l178_178910
  
theorem largest_n_satisfying_ineq : ∃ n : ℕ, (n < 10) ∧ ∀ m : ℕ, (m < 10) → m ≤ n ∧ (n < 10) ∧ (m < 10) → n = 9 :=
by
  sorry

end largest_n_satisfying_ineq_l178_178910


namespace probability_of_cold_given_rhinitis_l178_178105

/-- Define the events A and B as propositions --/
def A : Prop := sorry -- A represents having rhinitis
def B : Prop := sorry -- B represents having a cold

/-- Define the given probabilities as assumptions --/
axiom P_A : ℝ -- P(A) = 0.8
axiom P_A_and_B : ℝ -- P(A ∩ B) = 0.6

/-- Adding the conditions --/
axiom P_A_val : P_A = 0.8
axiom P_A_and_B_val : P_A_and_B = 0.6

/-- Define the conditional probability --/
noncomputable def P_B_given_A : ℝ := P_A_and_B / P_A

/-- The main theorem which states the problem --/
theorem probability_of_cold_given_rhinitis : P_B_given_A = 0.75 :=
by 
  sorry

end probability_of_cold_given_rhinitis_l178_178105


namespace card_pair_probability_sum_l178_178250

theorem card_pair_probability_sum (cards : Finset (ℕ × ℕ)) :
  (∀ n ∈ (Finset.range 1 21), cards.card = 4 * 20) →
  (cards.card = 80) →
  (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
   ∀ n ∈ {a, b, c}, ∃ (s ∈ cards), s = (n, n) ∧ (cards.erase s).card = 4 * 20 - 2) →
  let remaining_cards := 74 in
  let total_pairs := 102 + 3 in
  let total_ways := 2701 in
  total_pairs / total_ways = 105 / 2701 ∧ 105 + 2701 = 2806 :=
begin
  intros,
  sorry
end

end card_pair_probability_sum_l178_178250


namespace payment_methods_20_yuan_l178_178063

theorem payment_methods_20_yuan :
  let ten_yuan_note := 10
  let five_yuan_note := 5
  let one_yuan_note := 1
  ∃ (methods : Nat), 
    methods = 9 ∧ 
    ∃ (num_10 num_5 num_1 : Nat),
      (num_10 * ten_yuan_note + num_5 * five_yuan_note + num_1 * one_yuan_note = 20) →
      methods = 9 :=
sorry

end payment_methods_20_yuan_l178_178063


namespace marbles_count_l178_178804

theorem marbles_count (initial_marble: ℕ) (bought_marble: ℕ) (final_marble: ℕ) 
  (h1: initial_marble = 53) (h2: bought_marble = 134) : 
  final_marble = initial_marble + bought_marble -> final_marble = 187 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

-- sorry is omitted as proof is given.

end marbles_count_l178_178804


namespace kangaroo_fraction_sum_l178_178060

theorem kangaroo_fraction_sum (G P : ℕ) (hG : 1 ≤ G) (hP : 1 ≤ P) (hTotal : G + P = 2016) : 
  (G * (P / G) + P * (G / P) = 2016) :=
by
  sorry

end kangaroo_fraction_sum_l178_178060


namespace arithmetic_series_sum_l178_178275

variable (a₁ aₙ d S : ℝ)
variable (n : ℕ)

-- Defining the conditions (a₁, aₙ, d, and the formula for arithmetic series sum)
def first_term : a₁ = 10 := sorry
def last_term : aₙ = 70 := sorry
def common_diff : d = 1 / 7 := sorry

-- Equation to find number of terms (n)
def find_n : 70 = 10 + (n - 1) * (1 / 7) := sorry

-- Formula for the sum of an arithmetic series
def series_sum : S = (n * (10 + 70)) / 2 := sorry

-- The proof problem statement
theorem arithmetic_series_sum : 
  a₁ = 10 → 
  aₙ = 70 → 
  d = 1 / 7 → 
  (70 = 10 + (n - 1) * (1 / 7)) → 
  S = (n * (10 + 70)) / 2 → 
  S = 16840 := by 
  intros h1 h2 h3 h4 h5 
  -- proof steps would go here
  sorry

end arithmetic_series_sum_l178_178275
