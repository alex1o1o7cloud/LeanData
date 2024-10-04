import Mathlib

namespace range_sin_cos2_eq_l67_67112

theorem range_sin_cos2_eq : set.range (λ x : ℝ, sin x + cos (2 * x)) = set.Icc (-2 : ℝ) (9 / 8 : ℝ) := 
sorry

end range_sin_cos2_eq_l67_67112


namespace max_teams_with_most_wins_l67_67391

/-- In a round-robin tournament with 8 teams, where each team plays exactly once 
against every other team and each game results in a win for one team and a loss 
for the other, the maximum number of teams that can tie for the most wins is 5. -/
theorem max_teams_with_most_wins (n : ℕ) (teams : Fin n) : 
  n = 8 → (∀ t₁ t₂: Fin n, t₁ ≠ t₂ → (wins t₁ t₂ ∨ wins t₂ t₁))
  → (∃ k : ℕ, (k ≤ n) ∧ (maximum_possible_tied_teams k) ∧ k = 5) :=
begin
  -- Problem conditions and definitions will be used here
  sorry
end

end max_teams_with_most_wins_l67_67391


namespace sin_240_eq_neg_sqrt3_div_2_l67_67679

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67679


namespace sufficient_but_not_necessary_condition_l67_67511

noncomputable def lines_perpendicular {a : ℝ} : Prop :=
  let line1_slope := -1 / a
  let line2_slope := -a / (a + 2)
  line1_slope * line2_slope = -1

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = -3) → lines_perpendicular :=
sorry

end sufficient_but_not_necessary_condition_l67_67511


namespace ironing_pants_each_day_l67_67245

-- Given conditions:
def minutes_ironing_shirt := 5 -- minutes per day
def days_per_week := 5 -- days per week
def total_minutes_ironing_4_weeks := 160 -- minutes over 4 weeks

-- Target statement to prove:
theorem ironing_pants_each_day : 
  (total_minutes_ironing_4_weeks / 4 - minutes_ironing_shirt * days_per_week) /
  days_per_week = 3 :=
by 
sorry

end ironing_pants_each_day_l67_67245


namespace ceil_minus_x_of_fractional_part_half_l67_67271

theorem ceil_minus_x_of_fractional_part_half (x : ℝ) (hx : x - ⌊x⌋ = 1 / 2) : ⌈x⌉ - x = 1 / 2 :=
by
 sorry

end ceil_minus_x_of_fractional_part_half_l67_67271


namespace fraction_simplification_l67_67833

noncomputable theory

def x := 3
def y := 4

theorem fraction_simplification :
  (x / (y + 1)) / (y / (x + 2)) = 3 / 4 := by
  sorry

end fraction_simplification_l67_67833


namespace monotonicity_of_f_range_of_a_when_two_extreme_points_minimum_value_of_m_l67_67344

-- Definitions based on given conditions.
def f (x a : ℝ) : ℝ := (3 - x) * Real.exp x / x + a / x
def f' (x a : ℝ) : ℝ := (-x^2 + 3 * x - 3) * Real.exp x / x^2 - a / x^2
def h (x a : ℝ) : ℝ := (-x^2 + 3 * x - 3) * Real.exp x - a
noncomputable def h' (x : ℝ) : ℝ := (-x^2 + x) * Real.exp x

-- Goal: to prove that given these definitions, certain statements hold.
theorem monotonicity_of_f (a : ℝ) (h₀ : a > -3/4) :
  ∀ x > 0, f' x a < 0 :=
sorry

theorem range_of_a_when_two_extreme_points :
  ∀ a, ∃ x1 x2, (0 < x1 ∧ x1 < x2 ∧ h 0 a < 0 ∧ h 1 a > 0) ↔ (-3 < a ∧ a < -Real.exp 1) :=
sorry

theorem minimum_value_of_m (a : ℝ) (h₀ : -3 < a ∧ a < -Real.exp 1) :
  ∃ m (h₁ : ∃ x, f x a < m), ∀ m', m' < m → ¬∃ x, f x a < m' :=
sorry

end monotonicity_of_f_range_of_a_when_two_extreme_points_minimum_value_of_m_l67_67344


namespace unique_n_specified_by_logs_l67_67270

theorem unique_n_specified_by_logs :
  ∃! n : ℕ, (0 < n) ∧ (log 3 (log 27 n) = log 9 (log 9 n)) ∧ (n = 11) :=
by
  sorry

end unique_n_specified_by_logs_l67_67270


namespace isosceles_triangle_of_equal_inscribed_radii_l67_67017

theorem isosceles_triangle_of_equal_inscribed_radii 
  {A B C A1 C1 O1 O2 : Type*}
  (hAA1 : is_angle_bisector A A1 B C)
  (hCC1 : is_angle_bisector C C1 A B)
  (hO1_inscribed : is_center_of_inscribed_circle O1 A A1 C)
  (hO2_inscribed : is_center_of_inscribed_circle O2 C C1 A)
  (h_equal_radii : radius_of_inscribed_circle O1 A A1 C = radius_of_inscribed_circle O2 C C1 A) :
  is_isosceles A B C := 
sorry

end isosceles_triangle_of_equal_inscribed_radii_l67_67017


namespace largest_n_cube_condition_l67_67734

theorem largest_n_cube_condition :
  ∃ n : ℕ, (n^3 + 4 * n^2 - 15 * n - 18 = k^3) ∧ ∀ m : ℕ, (m^3 + 4 * m^2 - 15 * m - 18 = k^3 → m ≤ n) → n = 19 :=
by
  sorry

end largest_n_cube_condition_l67_67734


namespace hyperbola_eccentricity_l67_67326

variables {a b : ℝ} (hyp1 : a > 0) (hyp2 : b > 0)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the focus
def right_focus (c : ℝ) : Prop :=
  c^2 = a^2 + b^2

-- Define the intersection points with the line through origin
def intersect_points (x1 y1 : ℝ) : Prop :=
  hyperbola x1 y1 ∧ hyperbola (-x1) (-y1)

-- Define the dot product condition
def dot_product_zero (c x1 y1 : ℝ) : Prop :=
  (c - x1) * (c + x1) + (-y1) * y1 = 0

-- Define the area condition
def area_condition (c x1 y1 : ℝ) : Prop :=
  c * abs y1 * 0.5 = ab

-- Statement of the theorem
theorem hyperbola_eccentricity : 
  ∃ (c : ℝ), 
    right_focus c ∧
    ∃ (x1 y1 : ℝ), 
      intersect_points x1 y1 ∧
      dot_product_zero c x1 y1 ∧ 
      area_condition c x1 y1 ∧ 
      sqrt (c^2 / a^2) = sqrt 2 :=
sorry

end hyperbola_eccentricity_l67_67326


namespace area_of_fig_abcd_l67_67994

theorem area_of_fig_abcd
  (r : ℝ) (θ : ℝ)
  (hr : r = 15)
  (hθ : θ = real.pi / 4) :
  2 * (1 / 2 * r^2 * θ) = 225 * real.pi / 4 :=
begin
  sorry
end

end area_of_fig_abcd_l67_67994


namespace relatively_prime_example_l67_67787

theorem relatively_prime_example :
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd d c = 1 ∧ Nat.gcd e c = 1 :=
by
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  sorry

end relatively_prime_example_l67_67787


namespace triangle_sin_identity_triangle_max_perimeter_l67_67317

-- Problem statement for the first proof problem
theorem triangle_sin_identity (A B C : ℝ) (a b c : ℝ) 
    (h1 : b^2 + c^2 = a^2 + b * c)
    (h2 : A + B + C = π) (h3 : a = 2 * b * cos A) 
    (h4 : ∀ {X Y : ℝ}, 2 * sin (X + Y) cos (X - Y) = sin (2 * X) + sin (2 * Y)) :
    2 * sin B * cos C - sin (B - C) = sqrt(3) / 2 := 
sorry

-- Problem statement for the second proof problem
theorem triangle_max_perimeter (A B C : ℝ) (a b c : ℝ)
    (h1 : b^2 + c^2 = a^2 + b * c)
    (h2 : a = 2)
    (h3 : ∀ {X Y : ℝ}, sin X + sin Y = 2 * sin ((X + Y) / 2) * cos ((X - Y) / 2)) :
    ∃ P : ℝ, P = a + b + c ∧ P ≤ 6 :=
sorry

end triangle_sin_identity_triangle_max_perimeter_l67_67317


namespace negation_of_proposition_l67_67531

theorem negation_of_proposition (m : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∃ x : ℝ, x^2 + 2*x + m > 0) :=
sorry

end negation_of_proposition_l67_67531


namespace student_ticket_cost_l67_67190

theorem student_ticket_cost (cost_per_student_ticket : ℝ) :
  (12 * cost_per_student_ticket + 4 * 3 = 24) → cost_per_student_ticket = 1 :=
by
  intros h
  -- We should provide a complete proof here, but for illustration, we use sorry.
  sorry

end student_ticket_cost_l67_67190


namespace problem_false_statements_l67_67742

noncomputable def statement_I : Prop :=
  ∀ x : ℝ, ⌊x + Real.pi⌋ = ⌊x⌋ + 3

noncomputable def statement_II : Prop :=
  ∀ x : ℝ, ⌊x + Real.sqrt 2⌋ = ⌊x⌋ + ⌊Real.sqrt 2⌋

noncomputable def statement_III : Prop :=
  ∀ x : ℝ, ⌊x * Real.pi⌋ = ⌊x⌋ * ⌊Real.pi⌋

theorem problem_false_statements : ¬(statement_I ∨ statement_II ∨ statement_III) := 
by
  sorry

end problem_false_statements_l67_67742


namespace sin_240_eq_neg_sqrt3_div_2_l67_67678

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67678


namespace number_of_dryers_used_l67_67244

-- Definitions based on the given conditions:
def washer_cost : ℝ := 4
def dryer_cost_per_10min : ℝ := 0.25
def loads_of_laundry : ℝ := 2
def dryer_time_per_dryer : ℝ := 40
def total_spent : ℝ := 11

-- Derived values based on the conditions:
def cost_per_dryer_40min : ℝ := (dryer_time_per_dryer / 10) * dryer_cost_per_10min
def cost_of_washing : ℝ := loads_of_laundry * washer_cost
def total_spent_on_drying : ℝ := total_spent - cost_of_washing

-- Theorem statement
theorem number_of_dryers_used :
  total_spent_on_drying / cost_per_dryer_40min = 3 :=
by
  sorry

end number_of_dryers_used_l67_67244


namespace ratio_area_EAB_ABCD_l67_67010

/-- Given an isosceles trapezoid ABCD with shorter base length AB = 10, longer base length CD = 20, 
and non-parallel sides AD and BC extended to meet at E, prove that the ratio of the area of triangle EAB 
to the area of trapezoid ABCD is 1/3. -/
theorem ratio_area_EAB_ABCD (AB CD : ℝ) (hAB : AB = 10) (hCD : CD = 20) :
  ∃ (E : Type*) (is_isosceles_trapezoid: isosceles_trapezoid ABCD) (meet_at_E: meet (AD ∪ BC) E),
  (area (triangle EAB) / area (trapezoid ABCD) = 1 / 3) :=
by
  sorry

end ratio_area_EAB_ABCD_l67_67010


namespace not_hyperbola_locus_l67_67103

def dist (p1 p2 : ℝ × ℝ) : ℝ := ( (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 ).sqrt

def F1 : ℝ × ℝ := (0, 4)
def F2 : ℝ × ℝ := (0, -4)

theorem not_hyperbola_locus :
  ¬ ∃ (M : ℝ × ℝ), dist M F1 - dist M F2 = 8 :=
sorry

end not_hyperbola_locus_l67_67103


namespace nine_point_passes_through_third_midpoint_l67_67499

-- Definitions related to triangles, circumcircles, and nine-point circles might be available in Mathlib
-- We will define some necessary concepts using the information at hand

-- Given that ∆ABC and ∆XYZ share a circumcircle, and the nine-point circle of ∆ABC intersects midpoints of XY and XZ
variable {Point : Type*} [AffineSpace Point ℝ]

-- Assume existence of nine-point circle and its properties
variable (A B C X Y Z : Point)
variable (circumcircle : Circle Point) (ninePointCircleABC : Circle Point)
variable (M N P : Point) -- midpoints of XY, XZ, YZ respectively

-- Hypotheses that define the conditions
axiom commonCircumcircle : Circumcircle ∆ABC = circumcircle
axiom passesThroughMidpoints : 
  ninePointCircleABC.passes_through_midpoint (midpoint X Y) ∧ 
  ninePointCircleABC.passes_through_midpoint (midpoint X Z)

-- Theorem to prove that γ passes through the midpoint of YZ
theorem nine_point_passes_through_third_midpoint :
  ninePointCircleABC.passes_through_midpoint (midpoint Y Z) := 
sorry -- Proof omitted

end nine_point_passes_through_third_midpoint_l67_67499


namespace smaller_rectangle_area_l67_67567

-- Define the conditions
def large_rectangle_length : ℝ := 40
def large_rectangle_width : ℝ := 20
def smaller_rectangle_length : ℝ := large_rectangle_length / 2
def smaller_rectangle_width : ℝ := large_rectangle_width / 2

-- Define what we want to prove
theorem smaller_rectangle_area : 
  (smaller_rectangle_length * smaller_rectangle_width = 200) :=
by
  sorry

end smaller_rectangle_area_l67_67567


namespace oliver_did_not_wash_109_items_l67_67056

theorem oliver_did_not_wash_109_items :
  let total_items := 39 + 47 + 25 + 18,
      items_washed := 20,
      items_not_washed := total_items - items_washed
  in items_not_washed = 109 := by
  have total_items : ℕ := 39 + 47 + 25 + 18
  have items_washed : ℕ := 20
  have items_not_washed : ℕ := total_items - items_washed
  show items_not_washed = 109
  -- proof goes here
  sorry

end oliver_did_not_wash_109_items_l67_67056


namespace num_non_empty_subsets_P_l67_67877

def is_pos_nat (n : ℕ) : Prop :=
  n > 0

def is_pair_in_set (x y : ℕ) : Prop :=
  is_pos_nat x ∧ is_pos_nat y ∧ x + y = 5

def P : Set (ℕ × ℕ) :=
  { p | is_pair_in_set p.1 p.2 }

def num_non_empty_subsets (s : Set α) : ℕ :=
  2 ^ s.to_finset.card - 1

theorem num_non_empty_subsets_P : num_non_empty_subsets P = 15 :=
by
  sorry

end num_non_empty_subsets_P_l67_67877


namespace concurrency_of_AA1_BB1_CC1_l67_67058

variable {Point : Type} [EuclideanGeometry Point]

-- Define that we have an arbitrary triangle ABC with vertices A, B, and C.
variables {A B C C₁ A₁ B₁ : Point} 

-- Define that triangles ABC₁, A₁BC, and AB₁C are equilateral.
def equilateral_triangle (A B C : Point) : Prop :=
∃ r : ℝ, r > 0 ∧ dist A B = r ∧ dist B C = r ∧ dist C A = r

-- Assume the conditions given in the problem statement.
axiom ABC_eq : equilateral_triangle A B C₁
axiom A1BC_eq : equilateral_triangle A₁ B C
axiom AB1C_eq : equilateral_triangle A B₁ C

-- Prove that the lines AA₁, BB₁, and CC₁ intersect at a single point.
theorem concurrency_of_AA1_BB1_CC1 :
  ∃ (Q : Point), (colinear {A, A₁, Q} ∧ colinear {B, B₁, Q} ∧ colinear {C, C₁, Q}) := 
sorry

end concurrency_of_AA1_BB1_CC1_l67_67058


namespace determinant_identity_l67_67328

variable (x y z w : ℝ)
variable (h1 : x * w - y * z = -3)

theorem determinant_identity :
  (x + z) * w - (y + w) * z = -3 :=
by sorry

end determinant_identity_l67_67328


namespace a_minus_b_is_6_l67_67425

-- Defining the problem
def largest_a : ℕ := Nat.find_greatest (λ n => n^3 < 999) 999
def smallest_b : ℕ := Nat.find (λ n => n^5 > 99) 0

-- The theorem to prove
theorem a_minus_b_is_6 : largest_a - smallest_b = 6 := by
  sorry

end a_minus_b_is_6_l67_67425


namespace derivative_cos_pi_div_2_l67_67094

theorem derivative_cos_pi_div_2 : deriv (λ x : ℝ, cos x) (π / 2) = -1 :=
by sorry

end derivative_cos_pi_div_2_l67_67094


namespace log_lt_implies_cube_lt_not_necessary_log_lt_for_cube_lt_l67_67183

-- Define the conditions and main theorem.
theorem log_lt_implies_cube_lt {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : 
  (log a < log b) → (a^3 < b^3) :=
begin
  -- Proof should go here, skipping using sorry
  sorry,
end

-- Define a counterexample conditionally invalidating the necessity of log a < log b for a^3 < b^3.
theorem not_necessary_log_lt_for_cube_lt {a b : ℝ} (ha : a < b) (ha_neg : a < 0) (hb_neg : b < 0) :
  (a^3 < b^3) ∧ ¬(log a < log b) :=
begin
  -- Proof should go here, skipping using sorry
  sorry,
end

end log_lt_implies_cube_lt_not_necessary_log_lt_for_cube_lt_l67_67183


namespace avg_age_new_students_l67_67090

theorem avg_age_new_students :
  ∀ (O A_old A_new_avg : ℕ) (A_new : ℕ),
    O = 12 ∧ A_old = 40 ∧ A_new_avg = (A_old - 4) ∧ A_new_avg = 36 →
    A_new * 12 = (24 * A_new_avg) - (O * A_old) →
    A_new = 32 :=
by
  intros O A_old A_new_avg A_new
  intro h
  rcases h with ⟨hO, hA_old, hA_new_avg, h36⟩
  sorry

end avg_age_new_students_l67_67090


namespace polygon_dot_product_probability_l67_67314

theorem polygon_dot_product_probability:
  (P : ℚ) (P = 2 / 3) :
  ∀ (n : ℕ) (h : n = 2017)
    (A : Fin n → ℂ)
    (h_reg : ∀ i j, i ≠ j → ∥A i - A j∥ = 2 * sin (π * (i - j) / n)) 
    (i j : Fin n) (h_ij : i ≠ j), 
    (A i) • (A j) > 1 / 2 :=
sorry

end polygon_dot_product_probability_l67_67314


namespace finitely_countably_additive_measure_not_extendable_l67_67747

noncomputable def A (n1 : ℕ) (ns : List ℕ) : Set ℝ :=
  (Ioc 0 (1 / n1)) ∪ ⋃ i in ns, Ioc (1 / (i + 1)) (1 / i)

noncomputable def B (ns : List ℕ) : Set ℝ :=
  ⋃ i in ns, Ioc (1 / (i + 1)) (1 / i)

noncomputable def mu : Set ℝ → ℝ 
| A n1 ns => List.sum (List.map (λ i => (-1)^(i + 1) / i) ns) + ∑' n, if n ≥ n1 then (-1)^(n + 1) / n else 0
| B ns    => List.sum (List.map (λ i => (-1)^(i + 1) / i) ns)

theorem finitely_countably_additive_measure_not_extendable :
  ∃ (μ : Set ℝ → ℝ) (A : Set ℝ), 
    (∀ A B : Set ℝ, μ (A ∪ B) = μ A + μ B - μ (A ∩ B)) ∧
    (∀ A : Set ℝ, μ A = μ (Ioc 0 (1 / n1)) + μ (⋃ n ≥ n1, Ioc (1 / (n + 1)) (1 / n))) ∧
    (∃ S : Set (Set ℝ), ∀ s ∈ S, μ s = List.sum (List.map (λ i, (-1)^(i + 1) / i) (Finset.toList s)) + ∑' n, if n ≥ n1 then (-1)^(n + 1) / n else 0) ∧
    ¬(∃ σ : Set (Set ℝ), σ ⊇ A ∧ is_countably_additive μ) :=
begin
  use mu,
  sorry
end

end finitely_countably_additive_measure_not_extendable_l67_67747


namespace partition_triangle_l67_67690

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l67_67690


namespace sin_240_deg_l67_67608

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67608


namespace sum_first_five_terms_arithmetic_l67_67881

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → α} (a_is_arithmetic : is_arithmetic_sequence a)
          (h1 : a 2 + a 6 = 10)
          (h2 : a 4 * a 8 = 45)

theorem sum_first_five_terms_arithmetic (a : ℕ → α) (a_is_arithmetic : is_arithmetic_sequence a)
  (h1 : a 2 + a 6 = 10) (h2 : a 4 * a 8 = 45) : 
  (∑ i in Finset.range 5, a i) = 20 := 
by
  sorry  

end sum_first_five_terms_arithmetic_l67_67881


namespace count_real_z10_l67_67972

theorem count_real_z10 (z : ℂ) (h : z ^ 30 = 1) : 
  (↑((({z : ℂ | z ^ 30 = 1}.to_finset).filter (λ x, x ^ 10 = 1)).card) + 
  ↑((({z : ℂ | z ^ 30 = 1}.to_finset).filter (λ x, x ^ 10 = -1)).card)) = 16 := 
sorry

end count_real_z10_l67_67972


namespace domain_of_function_l67_67099

theorem domain_of_function :
  ∀ x : ℝ, (x + 1 ≥ 0) ∧ (6 - 3x > 0) ↔ (-1 ≤ x ∧ x < 2) :=
by
  intros x
  split
  -- => direction
  { intro h
    cases h with hx1 hx2
    split
    { linarith }
    { linarith } }
  -- <= direction
  { intro h
    cases h with hx1 hx2
    split
    { linarith }
    { linarith } }

end domain_of_function_l67_67099


namespace find_function_f_l67_67288

-- Define the problem condition
def func_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x - f(y)) = f(f(y)) + x * f(x) + x^2

-- State the desired result
theorem find_function_f :
  ∃! f : ℝ → ℝ, func_eq f ∧ (∀ x : ℝ, f(x) = 1 - (x^2 / 2)) :=
sorry

end find_function_f_l67_67288


namespace probability_sum_greater_than_7_l67_67993

/-- Let Ω be the sample space for the event of tossing two dice. Each die has 6 faces,
numbered 1 to 6. The event of the sum of the numbers on two dice being greater than 7
has probability 5/12. -/
theorem probability_sum_greater_than_7 :
  let Ω := {(i, j) | i ∈ {1, 2, 3, 4, 5, 6}, j ∈ {1, 2, 3, 4, 5, 6}} in
  let E := {(i, j) ∈ Ω | i + j > 7} in
  (finset.card E : ℚ) / (finset.card Ω : ℚ) = 5 / 12 :=
by sorry

end probability_sum_greater_than_7_l67_67993


namespace sum_of_fractions_l67_67935

def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 3 * x - (5/12)

theorem sum_of_fractions (n : ℕ) (h : n = 2016) :
  ∑ k in Finset.range n, f((k + 1 : ℝ) / 2017) = 2016 := 
sorry

end sum_of_fractions_l67_67935


namespace smallest_phi_abs_l67_67350

noncomputable def shifted_function (x : ℝ) (φ : ℝ) : ℝ :=
  3 * sin (3 * (x - π / 4) + φ)

theorem smallest_phi_abs {φ : ℝ} :
  (∀ x, shifted_function (π / 3) φ = 0) →
  (∃ k : ℤ, φ = k * π - π / 4) →
  abs φ = π / 4 :=
by
  sorry

end smallest_phi_abs_l67_67350


namespace factorable_iff_m_eq_2_l67_67269

theorem factorable_iff_m_eq_2 (m : ℤ) :
  (∃ (A B C D : ℤ), (x y : ℤ) -> (x^2 + 2*x*y + 2*x + m*y + 2*m = (x + A*y + B) * (x + C*y + D))) ↔ m = 2 :=
sorry

end factorable_iff_m_eq_2_l67_67269


namespace multiples_of_10_not_3_or_8_l67_67820

theorem multiples_of_10_not_3_or_8 (p : ℕ → Prop) :
  {n : ℕ | 1 ≤ n ∧ n ≤ 300 ∧ n % 10 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0}.card = 15 :=
by
  sorry

end multiples_of_10_not_3_or_8_l67_67820


namespace problem_conditions_l67_67889

def G (m : ℕ) : ℕ := m % 10

theorem problem_conditions (a b c : ℕ) (non_neg_m : ∀ m : ℕ, 0 ≤ m) :
  -- Condition ①
  ¬ (G (a - b) = G a - G b) ∧
  -- Condition ②
  (a - b = 10 * c → G a = G b) ∧
  -- Condition ③
  (G (a * b * c) = G (G a * G b * G c)) ∧
  -- Condition ④
  ¬ (G (3^2015) = 9) :=
by sorry

end problem_conditions_l67_67889


namespace range_of_f_eq_R_l67_67797

noncomputable def u (a x : ℝ) : ℝ := x^2 - a * x - a

theorem range_of_f_eq_R (a : ℝ) : 
  (∀ (f : ℝ → ℝ), (∀ x, f x = real.log (x^2 - a * x - a)) → (set.range f = set.univ ↔ (a ∈ set.Iic (-4) ∪ set.Ici 0))) :=
  sorry

end range_of_f_eq_R_l67_67797


namespace divide_triangle_l67_67711

/-- 
  Given a triangle with the total sum of numbers as 63,
  we want to prove that it can be divided into three parts
  where each part's sum is 21.
-/
theorem divide_triangle (total_sum : ℕ) (H : total_sum = 63) :
  ∃ (part1 part2 part3 : ℕ), 
    (part1 + part2 + part3 = total_sum) ∧ 
    part1 = 21 ∧ 
    part2 = 21 ∧ 
    part3 = 21 :=
by 
  use 21, 21, 21
  split
  . exact H.symm ▸ rfl
  . split; rfl
  . split; rfl
  . rfl

end divide_triangle_l67_67711


namespace amusement_park_weekly_revenue_l67_67237

def ticket_price : ℕ := 3
def visitors_mon_to_fri_per_day : ℕ := 100
def visitors_saturday : ℕ := 200
def visitors_sunday : ℕ := 300

theorem amusement_park_weekly_revenue : 
  let total_visitors_weekdays := visitors_mon_to_fri_per_day * 5
  let total_visitors_weekend := visitors_saturday + visitors_sunday
  let total_visitors := total_visitors_weekdays + total_visitors_weekend
  let total_revenue := total_visitors * ticket_price
  total_revenue = 3000 := by
  sorry

end amusement_park_weekly_revenue_l67_67237


namespace min_abs_sum_l67_67150

theorem min_abs_sum : ∃ x : ℝ, (|x + 1| + |x + 2| + |x + 6|) = 5 :=
sorry

end min_abs_sum_l67_67150


namespace count_real_z10_l67_67970

theorem count_real_z10 (z : ℂ) (h : z ^ 30 = 1) : 
  (↑((({z : ℂ | z ^ 30 = 1}.to_finset).filter (λ x, x ^ 10 = 1)).card) + 
  ↑((({z : ℂ | z ^ 30 = 1}.to_finset).filter (λ x, x ^ 10 = -1)).card)) = 16 := 
sorry

end count_real_z10_l67_67970


namespace car_speed_5_hours_l67_67199

variable (T : ℝ)
variable (S : ℝ)

theorem car_speed_5_hours (h1 : T > 0) (h2 : 2 * T = S * 5.0) : S = 2 * T / 5.0 :=
sorry

end car_speed_5_hours_l67_67199


namespace area_of_ABCM_is_zero_l67_67261

-- Define the regular 8-sided polygon and the properties
def Polygon (V : Type) [add_group V] := list V -- Simplified placeholder for the polygon vertices 
def is_regular_8_sided_polygon (polygon : list (ℝ × ℝ)) : Prop :=
  list.length polygon = 8 ∧
  ∀ i, i < 8 → ((polygon.nth i).get_or_else (0, 0)).1 = 5 ∧ ((polygon.nth i).get_or_else (0, 0)).2 = 0 
-- Note: This is a simplification; actual coordinates need more precise conditions

-- Define the intersection of diagonals A-E and D-G
def intersect (A E D G : ℝ × ℝ) : ℝ × ℝ := (0, 0)  -- Simplification for intersection point M

-- Given the above, define ABM and determine its area
def area_of_quadrilateral (A B C M : ℝ × ℝ) : ℝ :=
  0  -- Placeholder function to calculate the area

-- Now we state the problem in Lean, proving the area of ABCM is 0 given the conditions
theorem area_of_ABCM_is_zero (A B C D E G M : ℝ × ℝ) (polygon : list (ℝ × ℝ)) :
  is_regular_8_sided_polygon polygon →
  intersect A E D G = M →
  area_of_quadrilateral A B C M = 0 :=
begin
  intros hpolygon hintersect,
  sorry -- Proof is omitted as per instructions
end

end area_of_ABCM_is_zero_l67_67261


namespace volume_ratio_tetrahedron_octahedron_l67_67964

noncomputable def ratio_volumes (a : ℝ) (V_tet V_oct : ℝ) : Prop :=
  ∀ (a : ℝ) (V_tet V_oct : ℝ), 
    V_tet = (a ^ 3) / (6 * sqrt 2) → 
      V_oct = (2 * V_tet) → 
        (V_tet / V_oct = 1 / 2)

theorem volume_ratio_tetrahedron_octahedron (a : ℝ) (V_tet V_oct : ℝ) 
  (h1 : V_tet = a^3 / (6 * sqrt 2)) 
  (h2 : V_oct = 2 * V_tet) : 
  V_tet / V_oct = 1 / 2 :=
by 
  sorry

end volume_ratio_tetrahedron_octahedron_l67_67964


namespace smaller_rectangle_area_l67_67563

theorem smaller_rectangle_area
  (L : ℕ) (W : ℕ) (h₁ : L = 40) (h₂ : W = 20) :
  let l := L / 2;
      w := W / 2 in
  l * w = 200 :=
by
  sorry

end smaller_rectangle_area_l67_67563


namespace solve_inequality_l67_67530

noncomputable def solution_set (x : ℝ) : set ℝ := {y | y < -1 ∨ y > 2}

theorem solve_inequality (x : ℝ) : 
  solution_set x = {y | y ∈ (-∞:ℝ, -1) ∪ (2, +∞:ℝ)} :=
sorry

end solve_inequality_l67_67530


namespace sin_240_eq_neg_sqrt3_div_2_l67_67671

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67671


namespace range_of_f_l67_67144

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : set.range (λ x, if x = -2 then 0 else f x) = { y : ℝ | y ≠ 1 } :=
by
  sorry

end range_of_f_l67_67144


namespace tan_equation_solution_l67_67515

noncomputable def find_solutions :=
  {x : ℝ // ∃ k : ℤ, x = k * Real.pi ∨ x = (arctan (sqrt (3 / 5)) + k * Real.pi) ∨ x = (- arctan (sqrt (3 / 5))) + k * Real.pi}

theorem tan_equation_solution :
  ∀ x, 8.4743 * Real.tan (2 * x) - 4 * Real.tan (3 * x) = (Real.tan (3 * x))^2 * Real.tan (2 * x) → x ∈ find_solutions :=
sorry

end tan_equation_solution_l67_67515


namespace find_bridge_length_l67_67224

noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmh : ℝ := 44
noncomputable def crossing_time : ℝ := 45
noncomputable def train_speed_ms : ℝ := train_speed_kmh / 3.6
noncomputable def total_distance : ℝ := train_speed_ms * crossing_time
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem find_bridge_length : bridge_length = 299.9 :=
by
  -- Calculate the train speed in meters per second
  have h1 : train_speed_ms = 44 / 3.6 := rfl
  -- Compute the total distance covered by the train in 45 seconds
  have h2 : total_distance = (44 / 3.6) * 45 := rfl
  -- Derive the length of the bridge
  have h3 : bridge_length = (44 / 3.6) * 45 - 250 := rfl
  -- Verify the final answer
  have h4 : bridge_length = 299.9 := calc
    bridge_length = (44 / 3.6) * 45 - 250 : by rw [h3]
    ... = 299.9 : by norm_num
  exact h4

#eval find_bridge_length

end find_bridge_length_l67_67224


namespace sea_star_collection_l67_67813

theorem sea_star_collection (S : ℕ) (initial_seashells : ℕ) (initial_snails : ℕ) (lost_sea_creatures : ℕ) (remaining_items : ℕ) :
  initial_seashells = 21 →
  initial_snails = 29 →
  lost_sea_creatures = 25 →
  remaining_items = 59 →
  S + initial_seashells + initial_snails = remaining_items + lost_sea_creatures →
  S = 34 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end sea_star_collection_l67_67813


namespace locations_definitely_entered_summer_l67_67105

/- Definitions for the conditions given in the problem -/
structure TemperatureData (median mode mean variance : ℝ)

def locationA_data : TemperatureData := 
{ median := 24, mode := 22, mean := 0, variance := 0 }

def locationB_data : TemperatureData := 
{ median := 27, mode := 0, mean := 24, variance := 0 }

def locationC_data : TemperatureData := 
{ median := 0, mode := 0, mean := 26, variance := 10.8 }

def entered_summer (temps : list ℝ) : Prop := 
  (temps.length = 5) ∧ (∀ t ∈ temps, t ≥ 22) ∧ (temps.sum / 5 ≥ 22)

/- The main theorem to prove -/
theorem locations_definitely_entered_summer :
  entered_summer [22, 22, 24, 25, 26] ∧ 
  ¬entered_summer [19, 20, 27, 27, 27] ∧
  entered_summer [22, 25, 25, 26, 32] :=
by
  sorry

end locations_definitely_entered_summer_l67_67105


namespace Bill_initial_money_l67_67303

theorem Bill_initial_money (joint_money : ℕ) (pizza_cost : ℕ) (num_pizzas : ℕ) (final_bill_amount : ℕ) (initial_joint_money_eq : joint_money = 42) (pizza_cost_eq : pizza_cost = 11) (num_pizzas_eq : num_pizzas = 3) (final_bill_amount_eq : final_bill_amount = 39) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end Bill_initial_money_l67_67303


namespace optimal_insulation_layer_cost_l67_67011

theorem optimal_insulation_layer_cost :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 10 → (∃ (k : ℝ), C(x) = k / (3 * x + 5) ∧ C(0) = 8 ∧ 
     C(x) = 40 / (3 * x + 5))) →
  (f(x) = (600 / (3 * x + 5) + 8 * x) ∧
   (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 10 → f(x) = ((15*40) / (3 * x + 5) + 8 * x))) →
  (∃ (x : ℕ), x = 10 / 3 ∧ f(x) < f(y) ∀ (y : ℝ), y ≠ x) :=
sorry

end optimal_insulation_layer_cost_l67_67011


namespace arithmetic_sequence_sum_l67_67885

noncomputable def S (n : ℕ) (a_1 d : ℝ) : ℝ := n * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (
  a_2 a_4 a_6 a_8 a_1 d : ℝ,
  h1 : a_2 + a_6 = 10,
  h2 : a_4 * a_8 = 45,
  h3 : a_2 = a_1 + d,
  h4 : a_4 = a_1 + 3 * d,
  h5 : a_6 = a_1 + 5 * d,
  h6 : a_8 = a_1 + 7 * d,
  h7 : 2 * a_4 = 10,
  h8 : a_4 = 5,
  h9 : 5 * a_8 = 45,
  h10 : a_8 = 9,
  h11 : d = 1,
  h12 : a_1 = 2
) : S 5 a_1 d = 20 := sorry

end arithmetic_sequence_sum_l67_67885


namespace solution_points_satisfy_equation_l67_67175

theorem solution_points_satisfy_equation (x y : ℝ) :
  x^2 * (y + y^2) = y^3 + x^4 → (y = x ∨ y = -x ∨ y = x^2) := sorry

end solution_points_satisfy_equation_l67_67175


namespace sum_of_digits_of_x_l67_67558

-- Define what it means for a number to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

-- Define the conditions
def conditions (x : ℕ) : Prop :=
  is_palindrome x ∧ (100 ≤ x ∧ x ≤ 999) ∧ is_palindrome (x + 50) ∧ (1000 ≤ x + 50 ∧ x + 50 ≤ 1049)

-- Define the function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Prove the statement
theorem sum_of_digits_of_x : ∃ x : ℕ, conditions x ∧ sum_of_digits x = 15 :=
by
  sorry

end sum_of_digits_of_x_l67_67558


namespace kevin_food_spending_l67_67072

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end kevin_food_spending_l67_67072


namespace symmetric_function_value_l67_67943

theorem symmetric_function_value (ϕ : ℝ) :
  (∃ ϕ, (∀ x : ℝ, sin(-x + ϕ) = -sin(x + ϕ))) → (ϕ = π) := by
sorry

end symmetric_function_value_l67_67943


namespace train_length_is_140_l67_67225

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let distance := speed_ms * time_s
  distance - bridge_length_m

theorem train_length_is_140 :
  train_length 45 30 235 = 140 := by
  sorry

end train_length_is_140_l67_67225


namespace corridor_half_coverage_possible_l67_67205

theorem corridor_half_coverage_possible (L : ℝ) (runners : List (ℝ × ℝ)) :
  (∀ r ∈ runners, r.1 ≤ r.2) → 
  (Σ l r ∈ runners, l.1 = corridor_left_end ∧ l.2 = corridor_right_end) = runners →
  (∃ subset_runners : List (ℝ × ℝ), ∀ r1 r2 ∈ subset_runners, r1 ≠ r2 → ¬(r1.1 < r2.2 ∧ r2.1 < r1.2) ∧ 
    (Σ l r ∈ subset_runners, l.2 - l.1 >= L / 2)) :=
by { sorry }

end corridor_half_coverage_possible_l67_67205


namespace vector_magnitude_positive_l67_67829

variable {V : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)

-- Given: 
-- a is any non-zero vector
-- b is a unit vector
theorem vector_magnitude_positive (ha : a ≠ 0) (hb : ‖b‖ = 1) : ‖a‖ > 0 := 
sorry

end vector_magnitude_positive_l67_67829


namespace sum_of_coordinates_l67_67060

-- Definitions of points and their coordinates
def pointC (x : ℝ) : ℝ × ℝ := (x, 8)
def pointD (x : ℝ) : ℝ × ℝ := (x, -8)

-- The goal is to prove that the sum of the four coordinate values of points C and D is 2x
theorem sum_of_coordinates (x : ℝ) :
  (pointC x).1 + (pointC x).2 + (pointD x).1 + (pointD x).2 = 2 * x :=
by
  sorry

end sum_of_coordinates_l67_67060


namespace pa_length_max_min_values_l67_67548

noncomputable def parametric_eq_curve_c := ∀ θ : ℝ, (2 * Real.cos θ, 3 * Real.sin θ)
noncomputable def general_eq_line_l := ∀ x y: ℝ, 2*x + y - 6 = 0
noncomputable def distance_to_line_l (P: ℝ × ℝ) := 
  let (x, y) := P in
  (abs (4 * Real.cos x + 3 * Real.sin y - 6)) / (Real.sqrt 5)
noncomputable def pa_length := ∀ θ: ℝ, (abs (5 * Real.sin (θ + Real.atan (4 / 3)) - 6)) * (2 * (Real.sqrt 5) / 5)

theorem pa_length_max_min_values: ∀ θ: ℝ,
  ∃ max_val min_val : ℝ,
  max_val = (2 * ℝ.sqrt 5)/5 * abs (5*ℝ.sin (θ + Real.atan (4/3) - 6)) = 22*sqrt(5)/5 ∧
  min_val = (2 * ℝ.sqrt 5)/5 * abs (5*ℝ.sin (θ + Real.atan (4/3) + 6)) = 2*ℝ.sqrt 5/5 := 
by
  sorry

end pa_length_max_min_values_l67_67548


namespace final_percentage_of_water_in_mixture_l67_67902

theorem final_percentage_of_water_in_mixture (original_milk liters_pure_milk : ℕ) (water_percentage : ℝ) :
  water_percentage = 5 → original_milk = 10 → liters_pure_milk = 15 →
  let total_mixture_volume := original_milk + liters_pure_milk in
  let total_water := (water_percentage / 100 * original_milk) in
  (total_water / total_mixture_volume) * 100 = 2 :=
  by
  intros h1 h2 h3
  let total_mixture_volume := original_milk + liters_pure_milk
  let total_water := (water_percentage / 100 * original_milk)
  have total_mixture_volume_eq : total_mixture_volume = 25 := by sorry
  have total_water_eq : total_water = 0.5 := by sorry
  have percentage_water_eq : (total_water / total_mixture_volume) * 100 = 2 := by sorry
  exact percentage_water_eq

end final_percentage_of_water_in_mixture_l67_67902


namespace cos_theta_example_l67_67836

variables (a b : ℝ × ℝ) (θ : ℝ)

def cos_theta (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

theorem cos_theta_example :
  let a := (2, -1)
  let b := (1, 3)
  cos_theta a b = -(Real.sqrt 2) / 10 :=
by
  sorry

end cos_theta_example_l67_67836


namespace total_price_of_property_l67_67954

theorem total_price_of_property (price_per_sq_ft: ℝ) (house_size barn_size: ℝ) (house_price barn_price total_price: ℝ) :
  price_per_sq_ft = 98 ∧ house_size = 2400 ∧ barn_size = 1000 → 
  house_price = price_per_sq_ft * house_size ∧
  barn_price = price_per_sq_ft * barn_size ∧
  total_price = house_price + barn_price →
  total_price = 333200 :=
by
  sorry

end total_price_of_property_l67_67954


namespace functional_equation_solution_l67_67289

variable (f : ℝ → ℝ)

-- Declare the conditions as hypotheses
axiom cond1 : ∀ x : ℝ, 0 < x → 0 < f x
axiom cond2 : f 1 = 1
axiom cond3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

-- State the theorem to be proved
theorem functional_equation_solution : ∀ x : ℝ, f x = x :=
sorry

end functional_equation_solution_l67_67289


namespace find_m_l67_67381

theorem find_m (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 2 * y - 4 = 0) →
  (∀ x y : ℝ, x - 2 * y + m = 0) →
  (m = 7 ∨ m = -3) :=
by
  sorry

end find_m_l67_67381


namespace card_subsets_l67_67474

theorem card_subsets (A : Finset ℕ) (hA_card : A.card = 3) : (A.powerset.card = 8) :=
sorry

end card_subsets_l67_67474


namespace outstanding_students_and_books_l67_67930

-- Definitions based on conditions
def numOutstandingStudents : Nat := sorry

def totalBooks (x : Nat) : Nat := 
  if x = 8 then
    3 * x + 7
  else
    sorry

-- Problem statement as Lean theorem
theorem outstanding_students_and_books : 
  ∃ (x : Nat) (books : Nat), 
    (3 * x + 7 = books ∧ books = 31) ∧ (5 * x - 9 = books - 16) ∧ (x = 8) := 
by
  use 8
  use 31
  split
  case left =>
    split
    case left => sorry
    case right => sorry
  case right => sorry

end outstanding_students_and_books_l67_67930


namespace pats_family_people_count_l67_67911

theorem pats_family_people_count 
  (cookies : ℕ) 
  (candy : ℕ) 
  (brownies : ℕ) 
  (pieces_per_person : ℕ) 
  (h_cookies : cookies = 42) 
  (h_candy    : candy = 63) 
  (h_brownies : brownies = 21) 
  (h_pieces_per_person : pieces_per_person = 18) : 
  (cookies + candy + brownies) / pieces_per_person = 7 :=
by 
  rw [h_cookies, h_candy, h_brownies, h_pieces_per_person]
  sorry

end pats_family_people_count_l67_67911


namespace sin_240_l67_67626

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67626


namespace rational_numbers_covering_l67_67916

theorem rational_numbers_covering :
  ∃ (covering : ℕ → set.Icc (0 : ℝ) (1 : ℝ)) (length_sum : ℝ),
    (∀ n, covering n ⊆ set.Icc (0 : ℝ) (1 : ℝ)) ∧
    (∀ n, ∃ r : ℚ, r ∈ set.Icc (0 : ℝ) (1 : ℝ) ∧ r ∈ covering n) ∧
    (∀ r : ℚ, r ∈ set.Icc (0 : ℝ) (1 : ℝ) → ∃ n, r ∈ covering n) ∧
    (length_sum = ∑' n, |covering n|) ∧
    length_sum ≤ (1 / 1000) :=
begin
  sorry
end

end rational_numbers_covering_l67_67916


namespace problem_inequality_l67_67408

variable {n : ℕ}
variable (S_n : Finset (Fin n)) (f : Finset (Fin n) → ℝ)

axiom pos_f : ∀ A : Finset (Fin n), 0 < f A
axiom cond_f : ∀ (A : Finset (Fin n)) (x y : Fin n), x ≠ y → f (A ∪ {x}) * f (A ∪ {y}) ≤ f (A ∪ {x, y}) * f A

theorem problem_inequality (A B : Finset (Fin n)) : f A * f B ≤ f (A ∪ B) * f (A ∩ B) := sorry

end problem_inequality_l67_67408


namespace max_value_and_period_l67_67361

def vector_product (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 * b.1, a.2 * b.2)

def m : ℝ × ℝ := (1, 1/2)

def n : ℝ × ℝ := (0, 1)

def f (x : ℝ) : ℝ := 1/2 * (2 * (1/2 * Real.sin (x / 2) + 1) - 2) + 1
-- Simplified as f(x) = 1/2 * Real.sin (x / 2) + 1

theorem max_value_and_period : 
  (∀ x : ℝ,  f x <= 3/2) ∧ (∃ T : ℝ, T = 4*π ∧ (∀ x : ℝ, f (x + T) = f x)) :=
by 
  sorry

end max_value_and_period_l67_67361


namespace arccos_cos_eq_l67_67597

theorem arccos_cos_eq :
  Real.arccos (Real.cos 11) = 0.7168 := by
  sorry

end arccos_cos_eq_l67_67597


namespace matrix_rank_at_least_two_l67_67407

open Matrix

-- Define the problem statement
theorem matrix_rank_at_least_two 
  {m n : ℕ} 
  (A : Matrix (Fin m) (Fin n) ℚ)
  (prime_count: ℕ)
  (h1 : ∀ i j, 1 ≤ list.countp (λ p, p.prime) (abs (A i j)))
  (h2 : list.countp (λ p, p.prime) (abs (A i j)) ≥ m + n) :
  rank A ≥ 2 := 
sorry

end matrix_rank_at_least_two_l67_67407


namespace work_done_by_6_men_and_11_women_l67_67831

-- Definitions based on conditions
def work_completed_by_men (men : ℕ) (days : ℕ) : ℚ := men / (8 * days)
def work_completed_by_women (women : ℕ) (days : ℕ) : ℚ := women / (12 * days)
def combined_work_rate (men : ℕ) (women : ℕ) (days : ℕ) : ℚ := 
  work_completed_by_men men days + work_completed_by_women women days

-- Problem statement
theorem work_done_by_6_men_and_11_women :
  combined_work_rate 6 11 12 = 1 := by
  sorry

end work_done_by_6_men_and_11_women_l67_67831


namespace length_prod_AD_CD_l67_67990

-- Definitions and assumptions
open EuclideanGeometry

noncomputable def point (P A B C : Point) : Prop := 
  equidistant P A B ∧
  ∠APB = 2 * ∠ACB ∧
  intersect AC BP D ∧
  length PB = 3 ∧
  length PD = 2

-- Theorem statement
theorem length_prod_AD_CD {A B C P D : Point} (h : point P A B C) : 
  length AD * length CD = 5 :=
sorry

end length_prod_AD_CD_l67_67990


namespace count_real_z10_of_z30_eq_1_l67_67974

theorem count_real_z10_of_z30_eq_1 :
  ∃ S : Finset ℂ, S.card = 30 ∧ (∀ z ∈ S, z^30 = 1) ∧ (Finset.filter (λ z : ℂ, z^10 ∈ ℝ) S).card = 10 := 
by {
  sorry -- proof is not required/required to fill in
}

end count_real_z10_of_z30_eq_1_l67_67974


namespace john_total_cost_l67_67867

def base_price : ℝ := 450
def discount_percentage : ℝ := 0.15
def marble_percentage_increase : ℝ := 0.70
def glass_percentage_increase : ℝ := 0.35
def shipping_and_insurance : ℝ := 75
def tax_percentage : ℝ := 0.12

theorem john_total_cost (base_price discount_percentage marble_percentage_increase glass_percentage_increase shipping_and_insurance tax_percentage : ℝ) : 
  base_price = 450 ∧ 
  discount_percentage = 0.15 ∧ 
  marble_percentage_increase = 0.70 ∧ 
  glass_percentage_increase = 0.35 ∧ 
  shipping_and_insurance = 75 ∧ 
  tax_percentage = 0.12 →
  let discounted_price := base_price * (1 - discount_percentage) in
  let marble_cost := discounted_price * marble_percentage_increase in
  let price_with_marble := discounted_price + marble_cost in
  let glass_cost := price_with_marble * glass_percentage_increase in
  let price_with_glass := price_with_marble + glass_cost in
  let tax := price_with_glass * tax_percentage in
  let price_with_tax := price_with_glass + tax in
  let total_price := price_with_tax + shipping_and_insurance in
  total_price = 1058.18 :=
by {
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  cases h6 with h7 h8,
  cases h8 with h9 h10,
  cases h10 with h11 h12,
  have h_discounted_price : discounted_price = 450 * (1 - 0.15) := by { rw h1, rw h3, norm_num },
  have h_marble_cost : marble_cost = discounted_price * 0.70 := by { rw ← h_discounted_price, rw h4, norm_num },
  have h_price_with_marble : price_with_marble = discounted_price + discounted_price * 0.70 := by { rw h_marble_cost, norm_num },
  have h_glass_cost : glass_cost = price_with_marble * 0.35 := by { rw h_price_with_marble, rw h5, norm_num },
  have h_price_with_glass : price_with_glass = price_with_marble + price_with_marble * 0.35 := by { rw h_glass_cost, norm_num },
  have h_tax : tax = price_with_glass * 0.12 := by { rw h_price_with_glass, rw h6, norm_num },
  have h_price_with_tax : price_with_tax = price_with_glass + price_with_glass * 0.12 := by { rw h_tax, norm_num },
  have h_total_price : total_price = price_with_tax + 75 := by { rw h_price_with_tax, rw h7, rw h8, norm_num },
  exact h_total_price,
}

end john_total_cost_l67_67867


namespace count_real_z10_of_z30_eq_1_l67_67973

theorem count_real_z10_of_z30_eq_1 :
  ∃ S : Finset ℂ, S.card = 30 ∧ (∀ z ∈ S, z^30 = 1) ∧ (Finset.filter (λ z : ℂ, z^10 ∈ ℝ) S).card = 10 := 
by {
  sorry -- proof is not required/required to fill in
}

end count_real_z10_of_z30_eq_1_l67_67973


namespace ball_arrangements_l67_67123

theorem ball_arrangements : 
  let red : ℕ := 6
  let green : ℕ := 3
  let total_balls : ℕ := red + green
  let selected_balls : ℕ := 4
  (finset.card (finset.univ.powerset.filter (λ s, s.card = selected_balls))) = 15 :=
by 
  sorry

end ball_arrangements_l67_67123


namespace solve_equation_1_solve_equation_2_l67_67079

theorem solve_equation_1 (x : ℝ) : (x + 2) ^ 2 = 3 * (x + 2) ↔ x = -2 ∨ x = 1 := by
  sorry

theorem solve_equation_2 (x : ℝ) : x ^ 2 - 8 * x + 3 = 0 ↔ x = 4 + Real.sqrt 13 ∨ x = 4 - Real.sqrt 13 := by
  sorry

end solve_equation_1_solve_equation_2_l67_67079


namespace annual_interest_rate_is_15_l67_67584

-- Define principal amount P and time periods T1, T2
def P : ℕ := 640
def T1 : ℝ := 3.5
def T2 : ℝ := 5.0

-- Define difference in interests
def difference_in_interests : ℝ := 144

-- Define simple interest formula
def simple_interest (P : ℕ) (T : ℝ) (r : ℝ) : ℝ := P * T * r / 100

-- Define the goal: prove that the annual interest rate r is 15%
theorem annual_interest_rate_is_15 (r : ℝ) :
  simple_interest P T2 r - simple_interest P T1 r = difference_in_interests → r = 15 := by
  sorry

end annual_interest_rate_is_15_l67_67584


namespace smaller_rectangle_area_l67_67562

theorem smaller_rectangle_area
  (L : ℕ) (W : ℕ) (h₁ : L = 40) (h₂ : W = 20) :
  let l := L / 2;
      w := W / 2 in
  l * w = 200 :=
by
  sorry

end smaller_rectangle_area_l67_67562


namespace Q_on_angle_bisector_of_BAC_R_on_circumcircle_of_ABC_l67_67528

open Real EuclideanGeometry

variables {A B C E D P Q K L R : Point}

-- Conditions
axiom conditions :
  ∃ (ABC : Triangle) (E D P Q K L R : Point),
  E ∈ lineSegment A B ∧ D ∈ lineSegment A C ∧
  dist B E = dist C D ∧
  is_intersection P (line B E) (line C D) ∧
  second_intersection Q (circumcircle E P B) (circumcircle D P C) ∧
  midpoint K B E ∧ midpoint L C D ∧
  is_perpendicular_bisector R K Q ∧ is_perpendicular_bisector R L Q

-- Questions translated into Lean statements
theorem Q_on_angle_bisector_of_BAC :
  (Q ∈ angleBisector A B C) :=
sorry

theorem R_on_circumcircle_of_ABC :
  (R ∈ circumcircle A B C) :=
sorry

end Q_on_angle_bisector_of_BAC_R_on_circumcircle_of_ABC_l67_67528


namespace checkered_triangle_division_l67_67694

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l67_67694


namespace probability_Z_l67_67221

variable (p_X p_Y p_Z p_W : ℚ)

def conditions :=
  (p_X = 1/4) ∧ (p_Y = 1/3) ∧ (p_W = 1/6) ∧ (p_X + p_Y + p_Z + p_W = 1)

theorem probability_Z (h : conditions p_X p_Y p_Z p_W) : p_Z = 1/4 :=
by
  obtain ⟨hX, hY, hW, hSum⟩ := h
  sorry

end probability_Z_l67_67221


namespace floor_sums_equal_l67_67077

theorem floor_sums_equal {n : ℕ} (h : 1 < n) :
  (∑ k in (finset.range (n - 1)).map (nat.cast_add 2), nat.floor (n ^ (1 / k))) =
  (∑ k in (finset.range (n - 1)).map (nat.cast_add 2), nat.floor (real.log n / real.log k)) :=
sorry

end floor_sums_equal_l67_67077


namespace rhombus_locus_of_rectangle_l67_67423

def rectangle (K : Type) [euclidean_space K] :=
  ∃ (A B C D : K), -- Points A, B, C, and D forming the vertices of the rectangle
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    dist A B = dist C D ∧ dist B C = dist D A ∧
    dist A C = dist B D -- Opposite sides are equal

def is_center {K : Type} [euclidean_space K] (A B C D O : K) :=
  dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D -- Center O is equidistant to all vertices

def is_locus {K : Type} [euclidean_space K] (A B C D O M : K) :=
  dist A M ≥ dist O M ∧ dist B M ≥ dist O M ∧ dist C M ≥ dist O M ∧ dist D M ≥ dist O M

def is_rhombus_locus {K : Type} [euclidean_space K] (A B C D O : K) :=
  ∃ M : K, is_locus A B C D O M

theorem rhombus_locus_of_rectangle {K : Type} [euclidean_space K] :
  ∀ (A B C D O : K), rectangle A B C D → is_center A B C D O →
    is_rhombus_locus A B C D O :=
sorry -- Proof to be filled in

end rhombus_locus_of_rectangle_l67_67423


namespace marble_probability_l67_67537

theorem marble_probability :
  let total_marbles := 20
  let red_marbles := 12
  let blue_marbles := 8
  let draws := 3
  let total_ways := Nat.choose 20 3
  let favorable_ways := Nat.choose 12 2 * Nat.choose 8 1
  favorable_ways / ↑total_ways = 44 / 95 :=
by
  let total_marbles := 20
  let red_marbles := 12
  let blue_marbles := 8
  let draws := 3
  let total_ways := Nat.choose 20 3
  let favorable_ways := Nat.choose 12 2 * Nat.choose 8 1
  have h1 : total_ways = 1140 := by sorry
  have h2 : favorable_ways = 528 := by sorry
  calc
    528 / 1140 = 44 / 95 : by sorry

end marble_probability_l67_67537


namespace sum_c_d_eq_neg11_l67_67082

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x + 6) / (x^2 + c * x + d)

theorem sum_c_d_eq_neg11 (c d : ℝ) 
    (h₀ : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = 3 ∨ x = -4)) :
    c + d = -11 := 
sorry

end sum_c_d_eq_neg11_l67_67082


namespace only_odd_digit_square_l67_67304

def odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d % 2 = 1

theorem only_odd_digit_square (n : ℕ) : n^2 = n → odd_digits n → n = 1 ∨ n = 9 :=
by
  intros
  sorry

end only_odd_digit_square_l67_67304


namespace sin_240_eq_neg_sqrt3_div_2_l67_67632

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67632


namespace multiples_of_10_not_3_or_8_l67_67816

noncomputable def count_integers_between_1_and_300_that_are_multiples_of_10_not_3_or_8 : Nat :=
  Nat.card (Finset.filter (λ n, n % 10 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0) (Finset.range (300 + 1)))

theorem multiples_of_10_not_3_or_8 :
  count_integers_between_1_and_300_that_are_multiples_of_10_not_3_or_8 = 15 :=
by
  sorry

end multiples_of_10_not_3_or_8_l67_67816


namespace distance_from_circle_center_to_line_l67_67532

noncomputable def polar_circle_center : ℝ × ℝ := (1, 0)

noncomputable def distance_from_point_to_line (point : ℝ × ℝ) (line : ℝ → Prop) : ℝ := 
  let (x, y) := point in 
  let distance := |x - 4| in
  distance

theorem distance_from_circle_center_to_line :
  let circle_center := polar_circle_center
  let line := λ x, x = 4
  distance_from_point_to_line circle_center line = 3 := 
by
  sorry

end distance_from_circle_center_to_line_l67_67532


namespace range_of_quadratic_function_l67_67294

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 + x

theorem range_of_quadratic_function : 
  set.range (λ x : {x : ℝ // -1 ≤ x ∧ x ≤ 3}, quadratic_function x) = set.Icc (-1/4 : ℝ) 12 :=
sorry

end range_of_quadratic_function_l67_67294


namespace initial_mean_corrected_observations_l67_67947

theorem initial_mean_corrected_observations:
  ∃ M : ℝ, 
  (∀ (Sum_initial Sum_corrected : ℝ), 
    Sum_initial = 50 * M ∧ 
    Sum_corrected = Sum_initial + (48 - 23) → 
    Sum_corrected / 50 = 41.5) →
  M = 41 :=
by
  sorry

end initial_mean_corrected_observations_l67_67947


namespace max_subsequences_of_n_n1_n2_l67_67505

theorem max_subsequences_of_n_n1_n2 (seq : List ℕ) (h_len : seq.length = 2007) :
  ∃ k, k = 669 ∧ (List.natSubsequencesOfForm seq n (n + 1) (n + 2)).length ≤ k ^ 3 :=
by
  sorry

end max_subsequences_of_n_n1_n2_l67_67505


namespace minimum_value_function_equality_holds_at_two_thirds_l67_67897

noncomputable def f (x : ℝ) : ℝ := 4 / x + 1 / (1 - x)

theorem minimum_value_function (x : ℝ) (hx : 0 < x ∧ x < 1) : f x ≥ 9 := sorry

theorem equality_holds_at_two_thirds : f (2 / 3) = 9 := sorry

end minimum_value_function_equality_holds_at_two_thirds_l67_67897


namespace divide_triangle_l67_67715

/-- 
  Given a triangle with the total sum of numbers as 63,
  we want to prove that it can be divided into three parts
  where each part's sum is 21.
-/
theorem divide_triangle (total_sum : ℕ) (H : total_sum = 63) :
  ∃ (part1 part2 part3 : ℕ), 
    (part1 + part2 + part3 = total_sum) ∧ 
    part1 = 21 ∧ 
    part2 = 21 ∧ 
    part3 = 21 :=
by 
  use 21, 21, 21
  split
  . exact H.symm ▸ rfl
  . split; rfl
  . split; rfl
  . rfl

end divide_triangle_l67_67715


namespace amount_of_water_added_l67_67158

noncomputable def initial_sugar_water : ℝ := 300
noncomputable def initial_sugar_concentration : ℝ := 0.08
noncomputable def final_sugar_concentration : ℝ := 0.05
noncomputable def added_water : ℝ := 180

theorem amount_of_water_added
  (initial_sugar_water = 300)
  (initial_sugar_concentration = 0.08)
  (final_sugar_concentration = 0.05)
  (added_water = 180) :
  let initial_sugar := initial_sugar_concentration * initial_sugar_water,
      final_solution := initial_sugar_water + added_water,
      final_sugar_concentration' := initial_sugar / final_solution in
  final_sugar_concentration' = final_sugar_concentration :=
by
  sorry

end amount_of_water_added_l67_67158


namespace circumradius_relation_l67_67396

-- Definitions

variables {A B C D E F : Type} [Triangle A B C] [OrthicTriangle A B C D E F]
variables (R R' : ℝ)

-- Main statement

theorem circumradius_relation (h1 : Circumradius (triangle A B C) = R)
    (h2 : Circumradius (orthicTriangle A B C) = R') : R = 2 * R' :=    
sorry

end circumradius_relation_l67_67396


namespace product_inequality_l67_67765

variable (x1 x2 x3 x4 y1 y2 : ℝ)

theorem product_inequality (h1 : y2 ≥ y1) 
                          (h2 : y1 ≥ x1)
                          (h3 : x1 ≥ x3)
                          (h4 : x3 ≥ x2)
                          (h5 : x2 ≥ x1)
                          (h6 : x1 ≥ 2)
                          (h7 : x1 + x2 + x3 + x4 ≥ y1 + y2) : 
                          x1 * x2 * x3 * x4 ≥ y1 * y2 :=
  sorry

end product_inequality_l67_67765


namespace multiples_of_10_not_3_or_8_l67_67815

noncomputable def count_integers_between_1_and_300_that_are_multiples_of_10_not_3_or_8 : Nat :=
  Nat.card (Finset.filter (λ n, n % 10 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0) (Finset.range (300 + 1)))

theorem multiples_of_10_not_3_or_8 :
  count_integers_between_1_and_300_that_are_multiples_of_10_not_3_or_8 = 15 :=
by
  sorry

end multiples_of_10_not_3_or_8_l67_67815


namespace intersecting_lines_l67_67102

theorem intersecting_lines (a b : ℝ) (h1 : 3 = (1 / 3) * 6 + a) (h2 : 6 = (1 / 3) * 3 + b) : a + b = 6 :=
sorry

end intersecting_lines_l67_67102


namespace sin_240_deg_l67_67658

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67658


namespace length_PQ_equals_m_n_plus_one_l67_67030

theorem length_PQ_equals_m_n_plus_one :
  ∃ (m n : ℕ), 
    let R : ℝ × ℝ := (10, 8),
    let P := (46/11 : ℝ, 207/22 : ℝ),
    let Q := (174/11 : ℝ, 145/22 : ℝ),
    R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) ∧
    m.gcd n = 1 ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (m / n)^2 ∧
    m + n = 156 := sorry

end length_PQ_equals_m_n_plus_one_l67_67030


namespace builder_installed_windows_l67_67556

theorem builder_installed_windows (total_windows needed installed_hours_per_window rest_installation_hours : ℕ)
  (h1 : total_windows = 10)
  (h2 : installed_hours_per_window = 5)
  (h3 : rest_installation_hours = 20) : 
  ∃ (already_installed : ℕ), already_installed = 6 :=
by
  have windows_left := rest_installation_hours / installed_hours_per_window
  have already_installed := total_windows - windows_left
  use already_installed
  have h4 : windows_left = 4
    from by { sorry }
  have h5 : already_installed = 6
    from by { sorry }
  exact h5

end builder_installed_windows_l67_67556


namespace arithmetic_sequence_sum_l67_67883

noncomputable def S (n : ℕ) (a_1 d : ℝ) : ℝ := n * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (
  a_2 a_4 a_6 a_8 a_1 d : ℝ,
  h1 : a_2 + a_6 = 10,
  h2 : a_4 * a_8 = 45,
  h3 : a_2 = a_1 + d,
  h4 : a_4 = a_1 + 3 * d,
  h5 : a_6 = a_1 + 5 * d,
  h6 : a_8 = a_1 + 7 * d,
  h7 : 2 * a_4 = 10,
  h8 : a_4 = 5,
  h9 : 5 * a_8 = 45,
  h10 : a_8 = 9,
  h11 : d = 1,
  h12 : a_1 = 2
) : S 5 a_1 d = 20 := sorry

end arithmetic_sequence_sum_l67_67883


namespace coprime_with_others_l67_67792

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end coprime_with_others_l67_67792


namespace ball_arrangement_count_l67_67124

theorem ball_arrangement_count : 
  let red_balls := 6
  let green_balls := 3
  let selected_balls := 4
  (number_of_arrangements red_balls green_balls selected_balls) = 15 :=
by
  sorry

def number_of_arrangements (red_balls : ℕ) (green_balls : ℕ) (selected_balls : ℕ) : ℕ :=
  let choose := λ n k : ℕ, nat.choose n k 
  let case1 := choose (red_balls) (selected_balls)  -- 4 Red Balls - 1 way
  let case2 := choose (red_balls) 3 * choose (green_balls) 1  -- 3 Red Balls and 1 Green Ball
  let case3 := choose (red_balls) 2 * choose (green_balls) 2  -- 2 Red Balls and 2 Green Balls
  let case4 := choose (red_balls) 1 * choose (green_balls) 3  -- 1 Red Ball and 3 Green Balls
  case1 + case2 + case3 + case4

end ball_arrangement_count_l67_67124


namespace sin_240_deg_l67_67663

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67663


namespace square_floor_tiling_total_number_of_tiles_l67_67575

theorem square_floor_tiling (s : ℕ) (h : (2 * s - 1 : ℝ) / (s ^ 2 : ℝ) = 0.41) : s = 4 :=
by
  sorry

theorem total_number_of_tiles : 4^2 = 16 := 
by
  norm_num

end square_floor_tiling_total_number_of_tiles_l67_67575


namespace infinite_superabundant_numbers_l67_67894

def is_superabundant (σ : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ k, 1 ≤ k → k < m → (σ m) / m > (σ k) / k

theorem infinite_superabundant_numbers (σ : ℕ → ℕ) (hσ : ∀ n, σ n = ∑ d in (finset.divisors n), d) :
  ∃∞ m, m ≥ 1 ∧ is_superabundant σ m := by
sorry

end infinite_superabundant_numbers_l67_67894


namespace sin_240_deg_l67_67602

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67602


namespace extrema_range_l67_67799

noncomputable def hasExtrema (a : ℝ) : Prop :=
  (4 * a^2 + 12 * a > 0)

theorem extrema_range (a : ℝ) : hasExtrema a ↔ (a < -3 ∨ a > 0) := sorry

end extrema_range_l67_67799


namespace sin_240_l67_67620

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67620


namespace exam_paper_max_marks_l67_67518

/-- A candidate appearing for an examination has to secure 40% marks to pass paper i.
    The candidate secured 40 marks and failed by 20 marks.
    Prove that the maximum mark for paper i is 150. -/
theorem exam_paper_max_marks (p : ℝ) (s f : ℝ) (M : ℝ) (h1 : p = 0.40) (h2 : s = 40) (h3 : f = 20) (h4 : p * M = s + f) :
  M = 150 :=
sorry

end exam_paper_max_marks_l67_67518


namespace brown_eyed_brunettes_count_l67_67387

-- Definitions of conditions
variables (total_students blue_eyed_blondes brunettes brown_eyed_students : ℕ)
variable (brown_eyed_brunettes : ℕ)

-- Initial conditions
axiom h1 : total_students = 60
axiom h2 : blue_eyed_blondes = 18
axiom h3 : brunettes = 40
axiom h4 : brown_eyed_students = 24

-- Proof objective
theorem brown_eyed_brunettes_count :
  brown_eyed_brunettes = 24 - (24 - (20 - (20 - 18))) := sorry

end brown_eyed_brunettes_count_l67_67387


namespace planes_parallel_or_coincident_l67_67335

-- Define the normal vectors of the planes
def normal_vector_alpha : ℝ × ℝ × ℝ := (1/2, -1, 1/3)
def normal_vector_beta : ℝ × ℝ × ℝ := (-3, 6, -2)

-- Statement to prove that the planes are either parallel or coincident
theorem planes_parallel_or_coincident (n m : ℝ × ℝ × ℝ) (h₁ : n = normal_vector_alpha) (h₂ : m = normal_vector_beta) :
  ∃ k : ℝ, m = k • n :=
sorry

end planes_parallel_or_coincident_l67_67335


namespace solve_sin_cos_l67_67414

def int_part (x : ℝ) : ℤ := ⌊x⌋

theorem solve_sin_cos (x : ℝ) :
  int_part (Real.sin x + Real.cos x) = 1 ↔ ∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (Real.pi / 2) + 2 * Real.pi * n :=
by
  sorry

end solve_sin_cos_l67_67414


namespace non_congruent_triangles_in_grid_l67_67368

-- Define the points
def point1 := (0, 0)
def point2 := (0.5, 0)
def point3 := (1, 0)
def point4 := (0, 0.5)
def point5 := (0.5, 0.5)
def point6 := (1, 0.5)
def point7 := (0, 1)
def point8 := (0.5, 1)
def point9 := (1, 1)

-- Define the set of points
def points := {point1, point2, point3, point4, point5, point6, point7, point8, point9}

-- Define a function that determines whether two triangles are congruent
def congruent (a b c d e f : ℝ×ℝ) : Prop :=
  -- define what it means for two triangles to be congruent
  sorry

-- Define a function that counts the number of unique non-congruent triangles
def count_non_congruent_triangles (pts : set (ℝ×ℝ)) : ℕ :=
  -- define the logic for counting unique non-congruent triangles
  sorry

-- Main theorem
theorem non_congruent_triangles_in_grid : 
  count_non_congruent_triangles points = 3 :=
sorry

end non_congruent_triangles_in_grid_l67_67368


namespace sum_first_five_terms_arithmetic_l67_67880

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → α} (a_is_arithmetic : is_arithmetic_sequence a)
          (h1 : a 2 + a 6 = 10)
          (h2 : a 4 * a 8 = 45)

theorem sum_first_five_terms_arithmetic (a : ℕ → α) (a_is_arithmetic : is_arithmetic_sequence a)
  (h1 : a 2 + a 6 = 10) (h2 : a 4 * a 8 = 45) : 
  (∑ i in Finset.range 5, a i) = 20 := 
by
  sorry  

end sum_first_five_terms_arithmetic_l67_67880


namespace hyperbola_ksq_l67_67554

theorem hyperbola_ksq :
  ∃ k : ℝ, 
    (∃ a b : ℝ,
      0 < a ∧ 0 < b ∧
      (∀ (x y : ℝ), (y^2 / a^2) - (x^2 / b^2) = 1 ↔ 
        ((x, y) = (4, 3) ∨ (x, y) = (0, 2) ∨ (x, y) = (2, k)))
    ) → k^2 = 17 / 4 :=
begin
  sorry
end

end hyperbola_ksq_l67_67554


namespace b_l67_67246

def initial_marbles : Nat := 24
def lost_through_hole : Nat := 4
def given_away : Nat := 2 * lost_through_hole
def eaten_by_dog : Nat := lost_through_hole / 2

theorem b {m : Nat} (h₁ : m = initial_marbles - lost_through_hole)
  (h₂ : m - given_away = m₁)
  (h₃ : m₁ - eaten_by_dog = 10) :
  m₁ - eaten_by_dog = 10 := sorry

end b_l67_67246


namespace tissue_magnification_l67_67582

-- Define the known diameters
def magnified_diameter : ℝ := 1
def actual_diameter : ℝ := 0.001

-- Define the magnification factor
def magnification_factor (magnified_diameter actual_diameter : ℝ) : ℝ :=
  magnified_diameter / actual_diameter

-- The theorem statement asserting the magnification factor is 1000 given the conditions
theorem tissue_magnification :
  magnification_factor magnified_diameter actual_diameter = 1000 :=
by
  sorry

end tissue_magnification_l67_67582


namespace digital_earth_correct_statements_l67_67467

def digital_earth_provides_digital_spatial_laboratory : Prop :=
  "Digital Earth provides a digital spatial laboratory for scientists to study global issues, conducive to sustainable development research."

def digital_earth_government_decision_making_reliance : Prop :=
  "Government decision-making can fully rely on digital Earth."

def digital_earth_provides_basis_urban_management : Prop :=
  "Digital Earth provides a basis for urban management."

def digital_earth_development_predictability : Prop :=
  "The development of digital Earth is predictable."

theorem digital_earth_correct_statements
  (h1 : digital_earth_provides_digital_spatial_laboratory)
  (h3 : digital_earth_provides_basis_urban_management) :
  (∀ h2 h4, ¬ (digital_earth_government_decision_making_reliance ∧ digital_earth_development_predictability)) → (true) :=
by
  sorry

end digital_earth_correct_statements_l67_67467


namespace daisy_marks_three_points_l67_67046

noncomputable def mario_luigi_midpoints_count (Γ : Type) [Field Γ] [LinearOrder Γ] [MetricSpace Γ] [ProperSpace Γ] 
  (circle : Set Γ) (S : Γ) (mario_speed_ratio : Γ) (luigi_speed: Γ) (time_period: Γ)
  (daisy_positions: Finset Γ) : Prop := 
  S ∈ circle ∧ 
  mario_speed_ratio = 3 ∧ 
  luigi_speed * time_period = 2 * π ∧
  (∀ t ∈ Icc (0 : Γ) time_period, let mario_pos := exp_map_circle (mario_speed_ratio * luigi_speed) t in
                                   let luigi_pos := exp_map_circle luigi_speed t in
                                   let daisy_pos := (mario_pos + luigi_pos) / 2 in
                                   daisy_positions = set_of {daisy_pos}) ∧ 
  daisy_positions.card = 3
  -- where exp_map_circle speed t := some function to get the exponential map on the unit circle

theorem daisy_marks_three_points :
  ∃ (Γ : Type)
  (circle : Set Γ)
  (S : Γ)
  (luigi_speed : Γ),
  mario_luigi_midpoints_count Γ circle S 3 luigi_speed 6 { -- assuming 6 as the time period
  (0, sqrt 2 / 2),
  (0, -sqrt 2 / 2),
  (0, 0) } := 
sorry

end daisy_marks_three_points_l67_67046


namespace roots_of_unity_reals_l67_67977

theorem roots_of_unity_reals (S : Finset ℂ) (h : ∀ z ∈ S, z ^ 30 = 1) (h_card : S.card = 30) : 
  (S.filter (λ z, z ^ 10 ∈ Set ℝ)).card = 10 := 
sorry

end roots_of_unity_reals_l67_67977


namespace coefficient_x4_in_expansion_l67_67722

theorem coefficient_x4_in_expansion :
  let f : Polynomial ℚ := x * (x - 2 / x) ^ 7
  (coeff f 4) = 84 :=
by
  sorry

end coefficient_x4_in_expansion_l67_67722


namespace cistern_capacity_is_correct_l67_67133

noncomputable theory

def cistern_capacity (C : ℝ) : Prop :=
  let rate_A := C / (15 / 2)
  let rate_B := C / 5
  let rate_C := 14
  let net_rate := rate_A + rate_B - rate_C
  let time_to_empty := 60
  net_rate * time_to_empty = C

theorem cistern_capacity_is_correct :
  ∃ C : ℝ, C ≈ 237.74 ∧ cistern_capacity C :=
by
  sorry

end cistern_capacity_is_correct_l67_67133


namespace cos_angle_between_vectors_l67_67834

theorem cos_angle_between_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (1, 3)
  let dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  let magnitude (x : ℝ × ℝ) : ℝ := Real.sqrt (x.1 ^ 2 + x.2 ^ 2)
  let cos_theta := dot_product a b / (magnitude a * magnitude b)
  cos_theta = -Real.sqrt 2 / 10 :=
by
  sorry

end cos_angle_between_vectors_l67_67834


namespace normal_distribution_property_l67_67512

variable (X : ℝ → ℝ)
variable σ² : ℝ

noncomputable def is_normal_distribution (X : ℝ → ℝ) (μ σ² : ℝ) :=
  ∀ x : ℝ, P(X ≤ x) = by sorry

theorem normal_distribution_property :
  is_normal_distribution X 4 σ² → (0.85 : ℝ) = P(X ≤ 6) → (0.35 : ℝ) = P(2 < X ≤ 4) :=
by
  intro h1 h2
  sorry

end normal_distribution_property_l67_67512


namespace num_real_z10_l67_67967

theorem num_real_z10 (z : ℂ) (h : z^30 = 1) : (∃ n : ℕ, z = exp (2 * π * I * n / 30)) → ∃ n, z^10 ∈ ℝ :=
by sorry -- Here, we need to show that there are exactly 20 such complex numbers.

end num_real_z10_l67_67967


namespace sum_arithmetic_series_l67_67151

theorem sum_arithmetic_series :
  let a1 := 1000
  let an := 5000
  let d := 4
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 3003000 := by
    sorry

end sum_arithmetic_series_l67_67151


namespace smallest_side_of_triangle_l67_67445

theorem smallest_side_of_triangle (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) : 
  a > c ∧ b > c :=
by
  sorry

end smallest_side_of_triangle_l67_67445


namespace chessboard_parallelogram_l67_67057

theorem chessboard_parallelogram (n : ℕ) (h : n ≥ 2) (f : ℕ → ℕ → Prop) :
  (∀ i j : ℕ, f i j → i < n ∧ j < n) →
  (∃ k : ℕ, k ∈ {0,1} ∧ ∑ i in (finset.range n), (finset.card (finset.filter (λ j, f i j) (finset.range n))) = 2 * n) →
  (∑ i in (finset.range n), (finset.card (finset.filter (λ j, f i j) (finset.range n))) = 2 * n) →
  ∃ i1 i2 j1 j2 i3 i4 j3 j4, f i1 j1 ∧ f i2 j2 ∧ f i3 j3 ∧ f i4 j4 ∧
    i1 ≠ i2 ∧ j1 ≠ j2 ∧ i3 ≠ i4 ∧ j3 ≠ j4 ∧
    (i1 - i2) = (i3 - i4) ∧ (j1 - j2) = (j3 - j4) :=
by
  sorry

end chessboard_parallelogram_l67_67057


namespace triangle_parts_sum_eq_l67_67705

-- Define the total sum of numbers in the triangle
def total_sum : ℕ := 63

-- Define the required sum for each part
def required_sum : ℕ := 21

-- Define the possible parts that sum to the required sum
def part1 : list ℕ := [10, 6, 5]  -- this sums to 21
def part2 : list ℕ := [10, 5, 4, 1, 1]  -- this sums to 21

def part_sum (part : list ℕ) : ℕ := part.sum

-- The main theorem
theorem triangle_parts_sum_eq : 
  part_sum part1 = required_sum ∧ 
  part_sum part2 = required_sum ∧ 
  part_sum (list.drop (list.length part1 + list.length part2) [10, 6, 5, 10, 5, 4, 1, 1]) = required_sum :=
by
  sorry

end triangle_parts_sum_eq_l67_67705


namespace sqrt_difference_inequality_l67_67153

theorem sqrt_difference_inequality (a : ℝ) (h : a ≥ 2) :
  sqrt (a + 1) - sqrt a < sqrt (a - 1) - sqrt (a - 2) :=
by
  sorry

end sqrt_difference_inequality_l67_67153


namespace area_of_rhombus_area_enclosed_by_graph_l67_67139

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem area_of_rhombus (d₁ d₂ : ℝ) (h₁ : d₁ = 10) (h₂ : d₂ = 4) : 
  ∃ area, area = (1 / 2) * d₁ * d₂ ∧ area = 20 := by
  use (1 / 2) * d₁ * d₂
  split
  · exact rfl
  · rw [h₁, h₂]
    norm_num
    exact rfl

theorem area_enclosed_by_graph : 
  ∃ area, (∀ (x y : ℝ), abs (2 * x) + abs (5 * y) ≤ 10) → area = 20 := by
  have d₁ : ℝ := 10
  have d₂ : ℝ := 4
  let area := (1 / 2) * d₁ * d₂
  use area
  intro _
  apply area_of_rhombus
  { exact rfl }
  { exact rfl }

end area_of_rhombus_area_enclosed_by_graph_l67_67139


namespace problem_inequality_l67_67444

variable (a b : ℝ)

theorem problem_inequality (h_pos : 0 < a) (h_pos' : 0 < b) (h_sum : a + b = 1) :
  (1 / a^2 - a^3) * (1 / b^2 - b^3) ≥ (31 / 8)^2 := 
  sorry

end problem_inequality_l67_67444


namespace barn_painting_total_area_l67_67194

theorem barn_painting_total_area :
  let width := 12
  let length := 15
  let height := 5
  let divider_width := 12
  let divider_height := 5

  let external_wall_area := 2 * (width * height + length * height)
  let dividing_wall_area := 2 * (divider_width * divider_height)
  let ceiling_area := width * length
  let total_area := 2 * external_wall_area + dividing_wall_area + ceiling_area

  total_area = 840 := by
    sorry

end barn_painting_total_area_l67_67194


namespace equation_true_when_n_eq_2_l67_67170

theorem equation_true_when_n_eq_2 : (2 ^ (2 / 2)) = 2 :=
by
  sorry

end equation_true_when_n_eq_2_l67_67170


namespace range_of_f_l67_67146

noncomputable def f (x : ℝ) : ℝ := if x = -2 then 0 else (x^2 + 5 * x + 6) / (x + 2)

theorem range_of_f :
  set.range f = { y : ℝ | y ≠ 1 } :=
begin
  sorry
end

end range_of_f_l67_67146


namespace common_divisors_of_90_and_75_l67_67366

def is_divisor (n d : ℤ) : Prop :=
  d ≠ 0 ∧ n % d = 0

def common_divisors_count (a b : ℤ) : ℕ :=
  (List.filter (λ d, is_divisor a d ∧ is_divisor b d) (List.range (Int.natAbs a + 1))).length

theorem common_divisors_of_90_and_75 : common_divisors_count 90 75 = 8 := sorry

end common_divisors_of_90_and_75_l67_67366


namespace mark_total_payment_l67_67052

def labor_rate_radiator : ℝ := 75
def labor_hours_radiator : ℝ := 2
def part_cost : ℝ := 150
def labor_rate_cleaning : ℝ := 60
def labor_hours_cleaning : ℝ := 1
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.08

theorem mark_total_payment : 
  let labor_cost_radiator := labor_rate_radiator * labor_hours_radiator in
  let discounted_labor_cost_radiator := labor_cost_radiator * (1 - discount_rate) in
  let cleaning_labor_cost := labor_rate_cleaning * labor_hours_cleaning in
  let total_cost_before_tax := discounted_labor_cost_radiator + part_cost + cleaning_labor_cost in
  let tax := total_cost_before_tax * tax_rate in
  let total_cost_after_tax := total_cost_before_tax + tax in
  total_cost_after_tax = 372.60 :=
by
  sorry

end mark_total_payment_l67_67052


namespace explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l67_67941

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x + (a - 1)

-- Proof needed for the first question:
theorem explicit_formula_is_even (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) → a = 2 ∧ ∀ x : ℝ, f x a = x^2 + 1 :=
by sorry

-- Proof needed for the second question:
theorem tangent_line_at_1 (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 :=
by sorry

-- The tangent line equation at x = 1 in the required form
theorem tangent_line_equation (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 → (f 1 - deriv f 1 * 1 + deriv f 1 * x = 2 * x) :=
by sorry

end explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l67_67941


namespace maynard_unfilled_holes_l67_67901

theorem maynard_unfilled_holes (total_holes : ℕ) (filled_percentage : ℝ) (unfilled_holes : ℕ) :
  total_holes = 8 → filled_percentage = 0.75 → unfilled_holes = total_holes - nat.floor (filled_percentage * total_holes) → unfilled_holes = 2 :=
  by
    intros h1 h2 h3
    rw [h1, h2] at h3
    simp at h3
    exact h3


end maynard_unfilled_holes_l67_67901


namespace red_pigment_contribution_l67_67200

-- Definitions related to the problem
def sky_blue_paint_blue_fraction := 0.10
def sky_blue_paint_red_fraction := 0.90
def green_paint_blue_fraction := 0.70
def brown_paint_blue_fraction := 0.40
def total_brown_paint_weight := 10.0

-- Equation representing the blue pigment balance
def blue_pigment_balance (x y: ℝ) : Prop :=
  sky_blue_paint_blue_fraction * x + green_paint_blue_fraction * y = brown_paint_blue_fraction * total_brown_paint_weight

-- Equation representing the total weight balance
def weight_balance (x y: ℝ) : Prop :=
  x + y = total_brown_paint_weight

-- Main theorem
theorem red_pigment_contribution (x y: ℝ) (h1 : weight_balance x y) (h2 : blue_pigment_balance x y) : 
  sky_blue_paint_red_fraction * x = 4.5 :=
sorry

end red_pigment_contribution_l67_67200


namespace sin_240_eq_neg_sqrt3_div_2_l67_67633

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67633


namespace roots_of_unity_reals_l67_67980

theorem roots_of_unity_reals (S : Finset ℂ) (h : ∀ z ∈ S, z ^ 30 = 1) (h_card : S.card = 30) : 
  (S.filter (λ z, z ^ 10 ∈ Set ℝ)).card = 10 := 
sorry

end roots_of_unity_reals_l67_67980


namespace solve_system_1_solve_system_2_solve_system_3_solve_system_4_l67_67461

-- System 1
theorem solve_system_1 (x y : ℝ) (h1 : x = y + 1) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
  sorry

-- System 2
theorem solve_system_2 (x y : ℝ) (h1 : 3 * x + y = 8) (h2 : x - y = 4) : x = 3 ∧ y = -1 :=
by
  sorry

-- System 3
theorem solve_system_3 (x y : ℝ) (h1 : 5 * x + 3 * y = 2) (h2 : 3 * x + 2 * y = 1) : x = 1 ∧ y = -1 :=
by
  sorry

-- System 4
theorem solve_system_4 (x y z : ℝ) (h1 : x + y = 3) (h2 : y + z = -2) (h3 : z + x = 9) : x = 7 ∧ y = -4 ∧ z = 2 :=
by
  sorry

end solve_system_1_solve_system_2_solve_system_3_solve_system_4_l67_67461


namespace angle_EFG_deg_l67_67188

noncomputable def m_angle_EFG (E F G H : Type) (square_side : ℝ) (octagon_side : ℝ)
  (octagon_internal_angle : ℝ) (square_internal_angle : ℝ)
  (same_side: square_side = octagon_side)
  (oct_int_angle_correct: octagon_internal_angle = 135)
  (sq_int_angle_correct: square_internal_angle = 90)
  : ℝ :=
let m_angle_EFH := octagon_internal_angle - square_internal_angle in
let isosceles_base_angle := (180 - m_angle_EFH) / 2 in
isosceles_base_angle

theorem angle_EFG_deg (E F G H : Type) (square_side : ℝ) (octagon_side : ℝ)
  (octagon_internal_angle : ℝ) (square_internal_angle : ℝ)
  (same_side: square_side = octagon_side)
  (oct_int_angle_correct: octagon_internal_angle = 135)
  (sq_int_angle_correct: square_internal_angle = 90)
  : m_angle_EFG E F G H square_side octagon_side octagon_internal_angle square_internal_angle same_side oct_int_angle_correct sq_int_angle_correct = 67.5 :=
sorry

end angle_EFG_deg_l67_67188


namespace count_real_z10_l67_67969

theorem count_real_z10 (z : ℂ) (h : z ^ 30 = 1) : 
  (↑((({z : ℂ | z ^ 30 = 1}.to_finset).filter (λ x, x ^ 10 = 1)).card) + 
  ↑((({z : ℂ | z ^ 30 = 1}.to_finset).filter (λ x, x ^ 10 = -1)).card)) = 16 := 
sorry

end count_real_z10_l67_67969


namespace inequality_system_solution_l67_67080

theorem inequality_system_solution (x : ℤ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (x = 3 ∨ x = 4) := sorry

end inequality_system_solution_l67_67080


namespace divide_triangle_l67_67712

/-- 
  Given a triangle with the total sum of numbers as 63,
  we want to prove that it can be divided into three parts
  where each part's sum is 21.
-/
theorem divide_triangle (total_sum : ℕ) (H : total_sum = 63) :
  ∃ (part1 part2 part3 : ℕ), 
    (part1 + part2 + part3 = total_sum) ∧ 
    part1 = 21 ∧ 
    part2 = 21 ∧ 
    part3 = 21 :=
by 
  use 21, 21, 21
  split
  . exact H.symm ▸ rfl
  . split; rfl
  . split; rfl
  . rfl

end divide_triangle_l67_67712


namespace farmer_cows_more_than_goats_l67_67208

-- Definitions of the variables
variables (C P G x : ℕ)

-- Conditions given in the problem
def twice_as_many_pigs_as_cows : Prop := P = 2 * C
def more_cows_than_goats : Prop := C = G + x
def goats_count : Prop := G = 11
def total_animals : Prop := C + P + G = 56

-- The theorem to prove
theorem farmer_cows_more_than_goats
  (h1 : twice_as_many_pigs_as_cows C P)
  (h2 : more_cows_than_goats C G x)
  (h3 : goats_count G)
  (h4 : total_animals C P G) :
  C - G = 4 :=
sorry

end farmer_cows_more_than_goats_l67_67208


namespace kevin_food_spending_l67_67073

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end kevin_food_spending_l67_67073


namespace triangle_parts_sum_eq_l67_67709

-- Define the total sum of numbers in the triangle
def total_sum : ℕ := 63

-- Define the required sum for each part
def required_sum : ℕ := 21

-- Define the possible parts that sum to the required sum
def part1 : list ℕ := [10, 6, 5]  -- this sums to 21
def part2 : list ℕ := [10, 5, 4, 1, 1]  -- this sums to 21

def part_sum (part : list ℕ) : ℕ := part.sum

-- The main theorem
theorem triangle_parts_sum_eq : 
  part_sum part1 = required_sum ∧ 
  part_sum part2 = required_sum ∧ 
  part_sum (list.drop (list.length part1 + list.length part2) [10, 6, 5, 10, 5, 4, 1, 1]) = required_sum :=
by
  sorry

end triangle_parts_sum_eq_l67_67709


namespace Danny_more_wrappers_than_caps_l67_67265

-- Define the conditions
def bottle_caps_park := 11
def wrappers_park := 28

-- State the theorem representing the problem
theorem Danny_more_wrappers_than_caps:
  wrappers_park - bottle_caps_park = 17 :=
by
  sorry

end Danny_more_wrappers_than_caps_l67_67265


namespace bowling_team_scores_l67_67553

theorem bowling_team_scores : 
  ∀ (A B C : ℕ), 
  C = 162 → 
  B = 3 * C → 
  A + B + C = 810 → 
  A / B = 1 / 3 := 
by 
  intros A B C h1 h2 h3 
  sorry

end bowling_team_scores_l67_67553


namespace largest_four_digit_number_l67_67293

theorem largest_four_digit_number (a b c d : ℕ) (h : {a, b, c, d} = {9, 4, 1, 5}) : 
  let n := list.max [a * 1000 + b * 100 + c * 10 + d,
                     a * 1000 + b * 100 + d * 10 + c,
                     a * 1000 + c * 100 + b * 10 + d,
                     a * 1000 + c * 100 + d * 10 + b,
                     a * 1000 + d * 100 + b * 10 + c,
                     a * 1000 + d * 100 + c * 10 + b,
                     b * 1000 + a * 100 + c * 10 + d,
                     b * 1000 + a * 100 + d * 10 + c,
                     b * 1000 + c * 100 + a * 10 + d,
                     b * 1000 + c * 100 + d * 10 + a,
                     b * 1000 + d * 100 + a * 10 + c,
                     b * 1000 + d * 100 + c * 10 + a,
                     c * 1000 + a * 100 + b * 10 + d,
                     c * 1000 + a * 100 + d * 10 + b,
                     c * 1000 + b * 100 + a * 10 + d,
                     c * 1000 + b * 100 + d * 10 + a,
                     c * 1000 + d * 100 + a * 10 + b,
                     c * 1000 + d * 100 + b * 10 + a,
                     d * 1000 + a * 100 + b * 10 + c,
                     d * 1000 + a * 100 + c * 10 + b,
                     d * 1000 + b * 100 + a * 10 + c,
                     d * 1000 + b * 100 + c * 10 + a,
                     d * 1000 + c * 100 + a * 10 + b,
                     d * 1000 + c * 100 + b * 10 + a] in
  n = 9541 := 
begin
  sorry
end

end largest_four_digit_number_l67_67293


namespace students_in_school_at_least_225_l67_67003

-- Conditions as definitions
def students_in_band := 85
def students_in_sports := 200
def students_in_both := 60
def students_in_either := 225

-- The proof statement
theorem students_in_school_at_least_225 :
  students_in_band + students_in_sports - students_in_both = students_in_either :=
by
  -- This statement will just assert the correctness as per given information in the problem
  sorry

end students_in_school_at_least_225_l67_67003


namespace min_operations_at_least_l67_67529

noncomputable def min_operations (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, (k + 1).ceil_div (k + 1))

theorem min_operations_at_least : ∀ (n : ℕ), 0 < n →
  let bound := (Finset.range n).sum (λ k, (n.ceil_div (k + 1)))
  in min_operations n = bound :=
λ n hn, sorry

end min_operations_at_least_l67_67529


namespace volume_of_pyramid_is_correct_l67_67395

structure RectangularPrism :=
  (AB BC CG : ℝ)
  (AB_pos : AB = 3)
  (BC_pos : BC = 1)
  (CG_pos : CG = 2)

noncomputable def volume_pyramid (prism : RectangularPrism) : ℝ :=
  let base_area := prism.BC * (Real.sqrt (prism.AB ^ 2 + prism.CG ^ 2)) in
  (1 / 3) * base_area * prism.CG / 2

theorem volume_of_pyramid_is_correct (prism : RectangularPrism) :
  volume_pyramid prism = 4 / 3 :=
by
  -- Skipped proof step
  sorry

end volume_of_pyramid_is_correct_l67_67395


namespace kevin_food_expenditure_l67_67076

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end kevin_food_expenditure_l67_67076


namespace solve_quadratic_eq_l67_67960

theorem solve_quadratic_eq (x : ℝ) : x^2 = 6 * x ↔ (x = 0 ∨ x = 6) := by
  sorry

end solve_quadratic_eq_l67_67960


namespace num_subsets_l67_67811

def P : Set ℕ := {0, 1, 2}
def Q : Set ℕ := {x | x > 0}

theorem num_subsets :
  let intersection := {x | x ∈ P ∧ x ∈ Q} in
  fintype.card (Set.powerset intersection) = 4 :=
  by
    let intersection := {x | x ∈ P ∧ x ∈ Q}
    haveI : fintype intersection := sorry
    exact fintype.card_eq.mpr sorry

end num_subsets_l67_67811


namespace number_of_divisors_f2010_l67_67744

-- Define the function f(n) as per the problem statement
def f (n: ℕ) : ℕ := 2 ^ n

-- Define the problem statement to prove the number of divisors of f(2010)
theorem number_of_divisors_f2010 :
  (nat.divisors (f 2010)).card = 2011 :=
by {
  -- We'll skip the proof using sorry
  sorry
}

end number_of_divisors_f2010_l67_67744


namespace tangent_perpendicular_line_l67_67962

theorem tangent_perpendicular_line (f g : ℝ → ℝ) (a : ℝ) (h₀ : a ≠ 0) 
  (h₁ : ∀ x, f x = real.exp x) 
  (h₂ : ∀ x, g x = a * x^2 - a) 
  (h₃ : ∀ x₀, deriv g x₀ = deriv (\ x => a * x^2 - a) x₀)
  (h₄ : (∃ x₀, g x₀ = x₀ + 1 ∧ deriv g x₀ = deriv (\ x => x + 1) x₀)) : 
  ∃ L : ℝ → ℝ, (∀ x₀, g x₀ = x₀ + 1 ∧ L = (λ x, - (x + 1)) ∧ L = (λ x, x + y + 1)) :=
sorry

end tangent_perpendicular_line_l67_67962


namespace count_lattice_points_l67_67541

noncomputable def distance (p1 p2 : ℤ × ℤ) : ℤ :=
  Int.natAbs (p2.1 - p1.1) + Int.natAbs (p2.2 - p1.2)

noncomputable def valid_path_distance (A B P : ℤ × ℤ) : Prop :=
  distance A P + distance P B ≤ 24

noncomputable def is_lattice_point (x y : ℤ) : Prop := true

theorem count_lattice_points :
  (let count := (finset.Icc (-10, -8) (10, 8)).filter (λ P, valid_path_distance (-4, 3) (4, -3) P).card in
  count = 273) :=
by
  sorry

end count_lattice_points_l67_67541


namespace find_common_real_root_l67_67865

theorem find_common_real_root :
  ∃ (m a : ℝ), (a^2 + m * a + 2 = 0) ∧ (a^2 + 2 * a + m = 0) ∧ m = -3 ∧ a = 1 :=
by
  -- Skipping the proof
  sorry

end find_common_real_root_l67_67865


namespace smaller_rectangle_area_l67_67561

theorem smaller_rectangle_area
  (L : ℕ) (W : ℕ) (h₁ : L = 40) (h₂ : W = 20) :
  let l := L / 2;
      w := W / 2 in
  l * w = 200 :=
by
  sorry

end smaller_rectangle_area_l67_67561


namespace a_beats_b_by_l67_67517

noncomputable def speed (distance : ℝ) (time : ℝ) := distance / time

noncomputable def distance_run (speed : ℝ) (time : ℝ) := speed * time

theorem a_beats_b_by : by what distance does A beat B :=
  let distance_a := 160
  let time_a := 28
  let distance_b := 160
  let time_b := 32

  -- Calculate speeds
  let speed_a := speed distance_a time_a
  let speed_b := speed distance_b time_b

  -- Calculate distances run by A and B in 32 seconds
  let distance_a_in_32 := distance_run speed_a time_b
  let distance_b_in_32 := distance_run speed_b time_b

  -- Since B runs 160 meters in 32 seconds, we calculate the distance A beats B
  have h1 : distance_b_in_32 = 160 := by rfl
  have h2 : distance_a_in_32 - distance_b_in_32 = 22.848 := by 
    sorry

  show 22.848 meters, from h2

end a_beats_b_by_l67_67517


namespace place_three_digit_left_two_digit_l67_67186

-- Definitions based on conditions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Theorem statement
theorem place_three_digit_left_two_digit (x y : ℕ) (hx : is_three_digit x) (hy : is_two_digit y) : 
  (placing x to left of y) = 100 * x + y :=
sorry

end place_three_digit_left_two_digit_l67_67186


namespace tangent_line_at_x_is_2_l67_67775

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - 3 * Real.log x

theorem tangent_line_at_x_is_2 :
  ∃ x₀ : ℝ, (x₀ > 0) ∧ ((1/2) * x₀ - (3 / x₀) = -1/2) ∧ x₀ = 2 :=
by
  sorry

end tangent_line_at_x_is_2_l67_67775


namespace image_of_3_4_preimage_of_1_neg6_l67_67768

noncomputable def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

theorem image_of_3_4 : f 3 4 = (7, 12) := by
  -- It is by definition
  rfl

theorem preimage_of_1_neg6 : (∃ x y : ℝ, f x y = (1, -6)) := by
  exists (-2)
  exists 3
  -- Verification for the first pair
  rfl
  exists 3
  exists (-2)
  -- Verification for the second pair
  rfl

end image_of_3_4_preimage_of_1_neg6_l67_67768


namespace minimum_distance_l67_67048

noncomputable def z1 : ℂ := -3 - √3 * I
noncomputable def z2 : ℂ := √3 + I
noncomputable def z (θ : ℝ) : ℂ := √3 * sin θ + I * (√3 * cos θ + 2)

theorem minimum_distance (θ : ℝ) : 
  min (|z θ - z1| + |z θ - z2|) = 2 * (√3 + 1) :=
sorry

end minimum_distance_l67_67048


namespace total_distance_100_l67_67156

-- Definitions for the problem conditions:
def initial_velocity : ℕ := 40
def common_difference : ℕ := 10
def total_time (v₀ : ℕ) (d : ℕ) : ℕ := (v₀ / d) + 1  -- The total time until the car stops
def distance_traveled (v₀ : ℕ) (d : ℕ) : ℕ :=
  (v₀ * total_time v₀ d) - (d * total_time v₀ d * (total_time v₀ d - 1)) / 2

-- Statement to prove:
theorem total_distance_100 : distance_traveled initial_velocity common_difference = 100 := by
  sorry

end total_distance_100_l67_67156


namespace sin_240_l67_67621

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67621


namespace initial_puppies_count_l67_67217

theorem initial_puppies_count (sold_puppies cages puppies_per_cage total_cages : ℕ) 
  (h1 : sold_puppies = 3) 
  (h2 : puppies_per_cage = 5) 
  (h3 : total_cages = 3) 
  (puppies_in_cages : cages * puppies_per_cage = 15) :
  sold_puppies + puppies_in_cages = 18 :=
by
  sorry

end initial_puppies_count_l67_67217


namespace greatest_difference_between_units_digits_l67_67483

-- Define the conditions
def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0

-- Given number with variable units digit, simplified as x range (0-9)
def possible_units_digits : Finset ℕ :=
  {0, 3, 6, 9}

-- The main theorem
theorem greatest_difference_between_units_digits :
  ∀ x ∈ possible_units_digits, ∀ y ∈ possible_units_digits, 
  is_multiple_of_three (8 + 4 + x) ∧ is_multiple_of_three (8 + 4 + y) → (9 - 0) = 9 :=
begin
  intros x hx y hy,
  -- Initial hypothesis: digits are multiples of three
  intros h1 h2,
  rw [h1, h2],
  -- Conclusion
  exact rfl,
end

end greatest_difference_between_units_digits_l67_67483


namespace sin_240_eq_neg_sqrt3_over_2_l67_67647

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67647


namespace circle_area_l67_67576

theorem circle_area (side_length : ℝ) (h_side_length : side_length = 10) : 
  let radius := side_length / 2 in
  let area := Real.pi * radius^2 in
  area = 25 * Real.pi :=
by
  rw [h_side_length]
  let radius := 10 / 2
  have h_radius : radius = 5 := by norm_num
  let area := Real.pi * radius^2
  have h_area : area = 25 * Real.pi := by norm_num
  exact h_area

#check circle_area

end circle_area_l67_67576


namespace ordering_abc_l67_67751

noncomputable def a : ℝ := 2^Real.sin (Real.pi / 5)
noncomputable def b : ℝ := Real.logb (Real.pi / 5) (Real.pi / 4)
noncomputable def c : ℝ := Real.log 2 (Real.sin (Real.pi / 5))

theorem ordering_abc : a > b ∧ b > c := by
  sorry

end ordering_abc_l67_67751


namespace horizontal_distance_traveled_l67_67215

theorem horizontal_distance_traveled :
  (∃ xP xQ : ℝ, (xP ^ 2 - 2 * xP - 8 = 8) ∧ (xQ ^ 2 - 2 * xQ - 8 = -8) ∧ (∀ x1 x2 : ℝ, x1 = xP → x2 = xQ → (abs (x1 - x2)) = abs (sqrt 17 - 1))) := sorry

end horizontal_distance_traveled_l67_67215


namespace jake_bitcoins_l67_67019

theorem jake_bitcoins :
  let initial_bitcoins := 80
  let after_first_donation := initial_bitcoins - 20
  let after_giving_brother := after_first_donation / 2
  let after_tripling := after_giving_brother * 3
  let final_bitcoins := after_tripling - 10
  final_bitcoins = 80 :=
by
  let initial_bitcoins := 80
  let after_first_donation := initial_bitcoins - 20
  let after_giving_brother := after_first_donation / 2
  let after_tripling := after_giving_brother * 3
  let final_bitcoins := after_tripling - 10
  show final_bitcoins = 80, by sorry

end jake_bitcoins_l67_67019


namespace divide_triangle_l67_67714

/-- 
  Given a triangle with the total sum of numbers as 63,
  we want to prove that it can be divided into three parts
  where each part's sum is 21.
-/
theorem divide_triangle (total_sum : ℕ) (H : total_sum = 63) :
  ∃ (part1 part2 part3 : ℕ), 
    (part1 + part2 + part3 = total_sum) ∧ 
    part1 = 21 ∧ 
    part2 = 21 ∧ 
    part3 = 21 :=
by 
  use 21, 21, 21
  split
  . exact H.symm ▸ rfl
  . split; rfl
  . split; rfl
  . rfl

end divide_triangle_l67_67714


namespace checkered_triangle_division_l67_67696

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l67_67696


namespace total_jump_rope_time_l67_67595

theorem total_jump_rope_time :
  let cindy := 12
  let betsy := cindy / 2
  let tina := betsy * 3
  let sarah := cindy + tina
  in cindy + betsy + tina + sarah = 66 := by
  sorry

end total_jump_rope_time_l67_67595


namespace base_for_784_as_CDEC_l67_67508

theorem base_for_784_as_CDEC : 
  ∃ (b : ℕ), 
  (b^3 ≤ 784 ∧ 784 < b^4) ∧ 
  (∃ C D : ℕ, C ≠ D ∧ 784 = (C * b^3 + D * b^2 + C * b + C) ∧ 
  b = 6) :=
sorry

end base_for_784_as_CDEC_l67_67508


namespace mask_distribution_l67_67494

theorem mask_distribution (x : ℕ) (total_masks_3 : ℕ) (total_masks_4 : ℕ)
    (h1 : total_masks_3 = 3 * x + 20)
    (h2 : total_masks_4 = 4 * x - 25) :
    3 * x + 20 = 4 * x - 25 :=
by
  sorry

end mask_distribution_l67_67494


namespace complement_A_union_B_range_of_a_l67_67808

-- Definitions and conditions
def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 10}
def B : set ℝ := {x | 2 * x - 6 ≥ 0}
def C (a : ℝ) : set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem 1: To prove the complement of A ∪ B in the set of real numbers
theorem complement_A_union_B :
  (∁ (A ∪ B)) = {x | x < -1 ∨ x > 10} := by sorry

-- Theorem 2: To prove the range of values for a such that C ⊆ A
theorem range_of_a (a : ℝ) :
  (C a ⊆ A) → (-1 ≤ a ∧ a ≤ 9) := by sorry

end complement_A_union_B_range_of_a_l67_67808


namespace sin_240_l67_67622

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67622


namespace correct_propositions_l67_67749

-- Definitions and conditions
variable (a b c : Line)
variable (α β : Plane)

-- Correct propositions
-- Proposition (1): If $a \parallel c$ and $b \parallel c$, then $a \parallel b$
def prop1 : Prop := (a ∥ c ∧ b ∥ c) → a ∥ b 

-- Proposition (5): If $a \nsubseteq \alpha$, $b \parallel \alpha$, and $a \parallel b$, then $a \parallel \alpha$
def prop5 : Prop := (¬(a ⊆ α) ∧ b ∥ α ∧ a ∥ b) → a ∥ α 

-- Theorem statement to be proved
theorem correct_propositions : prop1 a b c ∧ prop5 a b c α := 
by
  sorry

end correct_propositions_l67_67749


namespace relatively_prime_example_l67_67786

theorem relatively_prime_example :
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd d c = 1 ∧ Nat.gcd e c = 1 :=
by
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  sorry

end relatively_prime_example_l67_67786


namespace relatively_prime_example_l67_67785

theorem relatively_prime_example :
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd d c = 1 ∧ Nat.gcd e c = 1 :=
by
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  sorry

end relatively_prime_example_l67_67785


namespace sin_240_eq_neg_sqrt3_div_2_l67_67618

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67618


namespace sets_intersection_union_a_value_l67_67429

def A := {x | ∃ (n : ℕ), x = n ∧ 1 ≤ n ∧ n < 6}
def B := {x | (x = 1) ∨ (x = 2)}
def C (a : ℝ) := {a, a^2 + 1}

theorem sets_intersection_union : 
  (A ∩ B = {1, 2}) ∧ (A ∪ B = {1, 2, 3, 4, 5}) := 
by
  sorry

theorem a_value (a : ℝ) (h₁ : B ⊆ C a) (h₂ : C a ⊆ B) : 
  a = 1 :=
by
  sorry

end sets_intersection_union_a_value_l67_67429


namespace relatively_prime_number_exists_l67_67777

def gcd (a b : ℕ) : ℕ := a.gcd b

def is_relatively_prime_to_all (n : ℕ) (lst : List ℕ) : Prop :=
  ∀ m ∈ lst, m ≠ n → gcd n m = 1

def given_numbers : List ℕ := [20172017, 20172018, 20172019, 20172020, 20172021]

theorem relatively_prime_number_exists :
  ∃ n ∈ given_numbers, is_relatively_prime_to_all n given_numbers := 
begin
  use 20172019,
  split,
  { -- Show 20172019 is in the list
    simp },
  { -- Prove 20172019 is relatively prime to all other numbers in the list
    intros m h1 h2,
    -- Further proof goes here
    sorry
  }
end

end relatively_prime_number_exists_l67_67777


namespace arccos_cos_11_eq_l67_67600

theorem arccos_cos_11_eq: Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end arccos_cos_11_eq_l67_67600


namespace count_triangles_in_figure_l67_67371

noncomputable def triangles_in_figure : ℕ := 53

theorem count_triangles_in_figure : triangles_in_figure = 53 := 
by sorry

end count_triangles_in_figure_l67_67371


namespace tangent_line_at_1_minimum_value_at_interval_range_of_a_l67_67802

open Real

noncomputable def f (a x : ℝ) := log x + (1/2) * a * x^2 - (a + 1) * x

theorem tangent_line_at_1 (a : ℝ) :
  (a = 1) → (
  let f1 := f a 1 in
  let tangent_eqn := - (3 / 2) in
  f1 = tangent_eqn
) :=
sorry

theorem minimum_value_at_interval (a : ℝ) :
  (a > 0) → (
  let min_val := -2 in
  let interval := [1, exp 1] in
  (∀ x ∈ interval, f a x ≥ min_val) → a = 2
) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 ∈ (0 : ℝ, +∞), x1 < x2 → f a x1 + x1 < f a x2 + x2) ↔ (0 ≤ a ∧ a ≤ 4) :=
sorry

end tangent_line_at_1_minimum_value_at_interval_range_of_a_l67_67802


namespace arithmetic_geometric_mean_inequality_l67_67449

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end arithmetic_geometric_mean_inequality_l67_67449


namespace distance_to_x_axis_l67_67936

def P : ℝ × ℝ := (3, -5)

theorem distance_to_x_axis : real.dist (P.snd) 0 = 5 := by
  -- P.snd extracts the y-coordinate from point P
  -- |y| = |-5| = 5
  rw [real.dist_eq, abs_neg]
  norm_num

end distance_to_x_axis_l67_67936


namespace ratio_GD_GE_eq_AC2_AB2_l67_67201

-- Define the problem conditions
variables {A B C D E F G : Point}
variable {circle : Circle A}
variable {triangle_ABC : Triangle A B C}
variable [circle_passes_through_BC : circle.passes_through B ∧ circle.passes_through C]
variable [D_on_AB : lies_on D (line_through A B)]
variable [E_on_AC : lies_on E (line_through A C)]
variable [AF_is_median : is_median A F (triangle_ABC)]
variable [G_on_DE : lies_on G (line_through D E)]

-- The proof statement
theorem ratio_GD_GE_eq_AC2_AB2 
  (h_circle : circle.passes_through B ∧ circle.passes_through C)
  (h_D : lies_on D (line_through A B))
  (h_E : lies_on E (line_through A C))
  (h_median : is_median A F (triangle_ABC))
  (h_G : lies_on G (line_through D E)) :
  (GD / GE) = (AC ^ 2 / AB ^ 2) :=
sorry

end ratio_GD_GE_eq_AC2_AB2_l67_67201


namespace miles_driven_after_pie_before_gas_l67_67866

-- Definitions from the conditions
def distance_to_grandma : ℕ := 78
def miles_driven_before_pie : ℕ := 35
def miles_left : ℕ := 25

-- Theorem statement that needs to be proved
theorem miles_driven_after_pie_before_gas (total_distance : ℕ) (miles_before_pie : ℕ) (miles_remaining : ℕ) :
  total_distance = 78 →
  miles_before_pie = 35 →
  miles_remaining = 25 →
  (total_distance - miles_remaining - miles_before_pie = 18) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end miles_driven_after_pie_before_gas_l67_67866


namespace binomials_not_coprime_l67_67914

theorem binomials_not_coprime (n k m : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) :
  ¬ Nat.coprime (Nat.choose n k) (Nat.choose n m) := by
  sorry

end binomials_not_coprime_l67_67914


namespace maximum_area_of_triangle_ABC_l67_67393

noncomputable def PA : ℝ := 3
noncomputable def PB : ℝ := 4
noncomputable def PC : ℝ := 2
noncomputable def BC : ℝ := 6

theorem maximum_area_of_triangle_ABC : 
  ∀ (A B C P : EuclideanSpace ℝ (Fin 2)), 
  dist P A = PA → dist P B = PB → dist P C = PC → dist B C = BC → 
  collinear {B, C, P} →
  ∃ (max_area : ℝ), max_area = 9 := 
by
  sorry

end maximum_area_of_triangle_ABC_l67_67393


namespace OC_eq_DF_EF_tangent_to_parabola_l67_67806

-- Definitions based on conditions
def is_on_parabola (x y : ℝ) : Prop := x ^ 2 = 4 * y
def tangent_slope (x₁ : ℝ) : ℝ := x₁ / 2
def tangent_line (x₁ y₁ x : ℝ) : ℝ := (tangent_slope x₁) * (x - x₁) + y₁
def point_C (x₁ : ℝ) : ℝ × ℝ := (x₁ / 2, 0)
def point_H (a : ℝ) : ℝ × ℝ := (a, -1)
def point_D (a : ℝ) : ℝ × ℝ := (a, 0)
def hf_perpendicular_slope (x₁ : ℝ) : ℝ := -2 / x₁
def point_F (a x₁ : ℝ) : ℝ × ℝ := (a - x₁ / 2, 0)
def point_E (a x₁ y₁ : ℝ) : ℝ × ℝ := (a, tangent_line x₁ y₁ a)

-- Statement 1: Prove |OC| = |DF|
theorem OC_eq_DF (a x₁ y₁ : ℝ) (h₁ : is_on_parabola x₁ y₁) (h₂ : x₁ ≠ 0) :
  abs (point_C x₁).fst = abs (point_F a x₁).fst :=
by
  sorry

-- Statement 2: Prove EF is tangent to the parabola
theorem EF_tangent_to_parabola (a x₁ y₁ : ℝ) (h₁ : is_on_parabola x₁ y₁) (h₂ : x₁ ≠ 0) :
  let E := point_E a x₁ y₁;
  let F := point_F a x₁;
  is_tangent_to_parabola (EF_slope E F) :=
by
  sorry

-- Auxiliary Definitions
def is_tangent_to_parabola (slope : ℝ) : Prop :=
  ∃ x y : ℝ, is_on_parabola x y ∧ x ^ 2 - 4 * slope * x + 4 * slope ^ 2 = 0 ∧ discriminant_slope_eq_0 slope

def EF_slope (E F : ℝ × ℝ) : ℝ := (E.snd - F.snd) / (E.fst - F.fst)

def discriminant_slope_eq_0 (slope : ℝ) : Prop :=
  16 * slope ^ 2 - 16 * slope ^ 2 = 0

end OC_eq_DF_EF_tangent_to_parabola_l67_67806


namespace cauchy_schwarz_inequality_l67_67888

theorem cauchy_schwarz_inequality
  {n : ℕ}
  {a b : Fin n → ℝ}
  (ha : ∀ i, 0 < a i)
  (hb : ∀ i, 0 < b i) :
  (∑ i, (a i) ^ 2 / (b i)) ≥ ((∑ i, a i) ^ 2 / (∑ i, b i)) :=
  sorry

end cauchy_schwarz_inequality_l67_67888


namespace polynomial_abs_sum_eq_81_l67_67826

theorem polynomial_abs_sum_eq_81 
  (a a_1 a_2 a_3 a_4 : ℝ) 
  (h : (1 - 2 * x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4)
  (ha : a > 0) 
  (ha_2 : a_2 > 0) 
  (ha_4 : a_4 > 0) 
  (ha_1 : a_1 < 0) 
  (ha_3 : a_3 < 0): 
  |a| + |a_1| + |a_2| + |a_3| + |a_4| = 81 := 
by 
  sorry

end polynomial_abs_sum_eq_81_l67_67826


namespace max_true_statements_l67_67892

theorem max_true_statements (x : ℝ) :
  (∀ x, -- given the conditions
    (0 < x^2 ∧ x^2 < 1) →
    (x^2 > 1) →
    (-1 < x ∧ x < 0) →
    (0 < x ∧ x < 1) →
    (0 < x - x^2 ∧ x - x^2 < 1)) →
  -- Prove the maximum number of these statements that can be true is 3
  (∃ (count : ℕ), count = 3) :=
sorry

end max_true_statements_l67_67892


namespace greatest_sum_l67_67120

theorem greatest_sum (sora_cards : list ℕ) (heesu_cards : list ℕ) (jiyeon_cards : list ℕ) :
  sora_cards = [4, 6] →
  heesu_cards = [7, 5] →
  jiyeon_cards = [3, 8] →
  sum heesu_cards > sum sora_cards ∧ sum heesu_cards > sum jiyeon_cards :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  simp
  sorry

end greatest_sum_l67_67120


namespace unique_intersection_points_l67_67944

theorem unique_intersection_points :
  let f1 (x : ℝ) := log 2 x
  let f2 (x : ℝ) := 1 / (log 2 x)
  let f3 (x : ℝ) := - (log 2 x)
  let f4 (x : ℝ) := -1 / (log 2 x)
  ∃ (count : ℕ), count = 3 ∧
    ∀ (x : ℝ), 0 < x → (f1 x = f2 x ∨ f1 x = f3 x ∨ f1 x = f4 x ∨ 
                         f2 x = f3 x ∨ f2 x = f4 x ∨ f3 x = f4 x) → count = 3 :=
by
  sorry

end unique_intersection_points_l67_67944


namespace Thomas_speed_greater_than_Jeremiah_l67_67988

-- Define constants
def Thomas_passes_kilometers_per_hour := 5
def Jeremiah_passes_kilometers_per_hour := 6

-- Define speeds (in meters per hour)
def Thomas_speed := Thomas_passes_kilometers_per_hour * 1000
def Jeremiah_speed := Jeremiah_passes_kilometers_per_hour * 1000

-- Define hypothetical additional distances
def Thomas_hypothetical_additional_distance := 600 * 2
def Jeremiah_hypothetical_additional_distance := 50 * 2

-- Define effective distances traveled
def Thomas_effective_distance := Thomas_speed + Thomas_hypothetical_additional_distance
def Jeremiah_effective_distance := Jeremiah_speed + Jeremiah_hypothetical_additional_distance

-- Theorem to prove
theorem Thomas_speed_greater_than_Jeremiah : Thomas_effective_distance > Jeremiah_effective_distance := by
  -- Placeholder for the proof
  sorry

end Thomas_speed_greater_than_Jeremiah_l67_67988


namespace kevin_food_expense_l67_67069

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end kevin_food_expense_l67_67069


namespace count_paths_avoiding_3_2_l67_67552

/-- There is a grid of size 6x5. Starting from point A (0,0) to point B (6,5),
    calculate the number of different 11-unit paths without passing through (3,2). -/
theorem count_paths_avoiding_3_2 : 
  let A := (0, 0)
      B := (6, 5)
      avoid := (3, 2)
      total_paths := Nat.choose 11 5
      paths_to_avoid := (Nat.choose 5 3) * (Nat.choose 5 2)
  in total_paths - paths_to_avoid = 362 :=
by
  have A := (0, 0)
  have B := (6, 5)
  have avoid := (3, 2)
  have total_paths := Nat.choose 11 5
  have paths_to_avoid := (Nat.choose 5 3) * (Nat.choose 5 2)
  exact Nat.sub_eq_of_eq_add (by norm_num; exact rfl)

end count_paths_avoiding_3_2_l67_67552


namespace find_c_l67_67770

theorem find_c (c : ℝ) : (∀ x : ℝ, x^2 + x < c ↔ x ∈ set.Ioc (-2) 1) → c = 2 :=
by sorry

end find_c_l67_67770


namespace exam_students_l67_67093

noncomputable def totalStudents (N : ℕ) (T : ℕ) := T = 70 * N
noncomputable def marksOfExcludedStudents := 5 * 50
noncomputable def remainingStudents (N : ℕ) := N - 5
noncomputable def remainingMarksCondition (N T : ℕ) := (T - marksOfExcludedStudents) / remainingStudents N = 90

theorem exam_students (N : ℕ) (T : ℕ) 
  (h1 : totalStudents N T) 
  (h2 : remainingMarksCondition N T) : 
  N = 10 :=
by 
  sorry

end exam_students_l67_67093


namespace max_distance_between_circle_centers_l67_67845

theorem max_distance_between_circle_centers :
  let rect_width := 20
  let rect_height := 16
  let circle_diameter := 8
  let horiz_distance := rect_width - circle_diameter
  let vert_distance := rect_height - circle_diameter
  let max_distance := Real.sqrt (horiz_distance ^ 2 + vert_distance ^ 2)
  max_distance = 4 * Real.sqrt 13 :=
by
  sorry

end max_distance_between_circle_centers_l67_67845


namespace findCorrectAnswer_l67_67258

-- Definitions
variable (x : ℕ)
def mistakenCalculation : Prop := 3 * x = 90
def correctAnswer : ℕ := x - 30

-- Theorem statement
theorem findCorrectAnswer (h : mistakenCalculation x) : correctAnswer x = 0 :=
sorry

end findCorrectAnswer_l67_67258


namespace sin_240_l67_67623

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67623


namespace find_number_l67_67096

theorem find_number (x : ℤ) (h : 2 * x - 8 = -12) : x = -2 :=
by
  sorry

end find_number_l67_67096


namespace sum_possible_values_sum_of_all_possible_values_l67_67827

theorem sum_possible_values (A B C : ℕ) (hA : A ≤ 9) (hB : B ≤ 9) (hC : C ≤ 9) (hdiv : 9 ∣ (14 + A + B + C)) :
    A + B + C = 4 ∨ A + B + C = 13 ∨ A + B + C = 22 :=
sorry

theorem sum_of_all_possible_values :
  ∑ (x : ℕ) in ({4, 13, 22} : Finset ℕ), x = 39 :=
sorry

end sum_possible_values_sum_of_all_possible_values_l67_67827


namespace grocer_purchased_bananas_l67_67168

theorem grocer_purchased_bananas :
  ∃ (P : ℕ), 
    (P / 3) * (1 / 2) = PurchasePrice / 4 ∧ 
    (P / 4) * 1 = SellingPrice / 1 ∧ 
    SellingPrice - PurchasePrice = 11 ∧ 
    P = 792 := 
begin
    sorry
end

end grocer_purchased_bananas_l67_67168


namespace find_function_l67_67311

noncomputable def satisfies_conditions (f : ℝ → ℝ) :=
  (∀ x y : ℝ, f(x) + f(y) + 1 ≥ f(x + y) ∧ f(x + y) ≥ f(x) + f(y)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f(0) ≥ f(x)) ∧
  (-f(-1) = 1 ∧ f(1) = 1)

theorem find_function (f : ℝ → ℝ) (h : satisfies_conditions f) : 
  ∀ x : ℝ, f(x) = floor x := 
sorry

end find_function_l67_67311


namespace number_of_consecutive_zeros_l67_67411

-- Define the conditions based on the problem description
def productFactorialEndZeros : Nat :=
  let products := ∏ i in (Finset.range 50).map (λ i => i + 1), i!
  -- Count factors of 5 in the factorial decomposition
  (List.range 50).sum (λ n => (n + 1).factorial.factorCount 5)

theorem number_of_consecutive_zeros (N : Nat) :
  (productFactorialEndZeros % 100) = 12 := by 
  sorry

end number_of_consecutive_zeros_l67_67411


namespace max_sum_of_sequence_l67_67898

def sequence (n : ℕ) : ℤ := -n^2 + 9 * n + 10

def sum_of_sequence (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).sum sequence

theorem max_sum_of_sequence :
  ∃ n : ℕ, (n = 9 ∨ n = 10) ∧ ∀ m, sum_of_sequence n ≥ sum_of_sequence m :=
sorry

end max_sum_of_sequence_l67_67898


namespace rows_count_mod_pascals_triangle_l67_67260

-- Define the modified Pascal's triangle function that counts the required rows.
def modified_pascals_triangle_satisfying_rows (n : ℕ) : ℕ := sorry

-- Statement of the problem
theorem rows_count_mod_pascals_triangle :
  modified_pascals_triangle_satisfying_rows 30 = 4 :=
sorry

end rows_count_mod_pascals_triangle_l67_67260


namespace area_of_abs_2x_plus_abs_5y_eq_10_l67_67137

theorem area_of_abs_2x_plus_abs_5y_eq_10 : ∀ (x y : ℝ), |2 * x| + |5 * y| = 10 → ∃ A : ℝ, A = 20 :=
by
  intros x y h
  use 20
  sorry

end area_of_abs_2x_plus_abs_5y_eq_10_l67_67137


namespace sqrt_two_lt_xn_lt_sqrt_two_plus_one_over_n_l67_67756

noncomputable def x : ℕ → ℝ
| 1     := 2
| (n+1) := x n / 2 + 1 / x n

theorem sqrt_two_lt_xn_lt_sqrt_two_plus_one_over_n (n : ℕ) (hn : n ≥ 1) :
  sqrt 2 < x n ∧ x n < sqrt 2 + 1 / n := 
sorry

end sqrt_two_lt_xn_lt_sqrt_two_plus_one_over_n_l67_67756


namespace unique_reversible_six_digit_number_exists_l67_67729

theorem unique_reversible_six_digit_number_exists :
  ∃! (N : ℤ), 100000 ≤ N ∧ N < 1000000 ∧
  ∃ (f e d c b a : ℤ), 
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧ 
  9 * N = 100000 * f + 10000 * e + 1000 * d + 100 * c + 10 * b + a := 
sorry

end unique_reversible_six_digit_number_exists_l67_67729


namespace extreme_points_inequality_l67_67795

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 + a * Real.log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - 4 * x + 2

theorem extreme_points_inequality (a : ℝ) (h_a : 0 < a ∧ a < 1) (x0 : ℝ)
  (h_ext : 4 * x0^2 - 4 * x0 + a = 0) (h_min : ∃ x1, x0 + x1 = 1 ∧ x0 < x1 ∧ x1 < 1) :
  g x0 a > 1 / 2 - Real.log 2 :=
sorry

end extreme_points_inequality_l67_67795


namespace perpendicular_simson_lines_of_diametrically_opposite_points_l67_67915

theorem perpendicular_simson_lines_of_diametrically_opposite_points 
  {ABC : Triangle} 
  {circumcircle : Circle} 
  {P₁ P₂ : Point} 
  (hP1P2 : P₁ ≠ P₂ ∧ P₁ ⬝ P₂ ∈ circumcircle) 
  {A₁ B₁ A₂ B₂ : Point} 
  (hA₁ : A₁ = foot P₁ BC) 
  (hB₁ : B₁ = foot P₁ AC) 
  (hA₂ : A₂ = foot P₂ BC) 
  (hB₂ : B₂ = foot P₂ AC) :
  perpendicular (simson_line_of P₁ ABC) (simson_line_of P₂ ABC) ∧ 
  intersection_point (simson_line_of P₁ ABC) (simson_line_of P₂ ABC) ∈ nine_point_circle ABC := by
    sorry

end perpendicular_simson_lines_of_diametrically_opposite_points_l67_67915


namespace count_triangles_in_figure_l67_67370

noncomputable def triangles_in_figure : ℕ := 53

theorem count_triangles_in_figure : triangles_in_figure = 53 := 
by sorry

end count_triangles_in_figure_l67_67370


namespace vertex_angle_isosceles_triangle_l67_67392

theorem vertex_angle_isosceles_triangle (α : ℝ) (β : ℝ) (sum_of_angles : α + α + β = 180) (base_angle : α = 50) :
  β = 80 :=
by
  sorry

end vertex_angle_isosceles_triangle_l67_67392


namespace find_c_l67_67869

theorem find_c (c : ℕ) (h₀ : c > 0) : (∃ k m : ℕ,  1 ≤ k ∧ 2 ≤ m ∧ ∃ x : ℕ, (a_k c)^2 + c^3 = x^m) ↔ ∃ ℓ : ℕ, ℓ ≥ 2 ∧ c = ℓ^2 - 1 :=
by {
  sorry
}

def a_k (c : ℕ) : ℕ → ℕ
| 1     := c
| (n+1) := (a_k n)^2 + (a_k n) + c^3

end find_c_l67_67869


namespace range_of_k_l67_67838

theorem range_of_k (k : ℝ) (h : (∀ x y : ℝ, x^2 + k * y^2 = 2 → (∀ (a b : ℝ), (a ≠ 0) → (b = 0 → ∃ c : ℝ, foci_y c a b)))) : 0 < k ∧ k < 1 :=
sorry

end range_of_k_l67_67838


namespace sin_cos_fraction_l67_67752

variable (α : ℝ)
hypothesis h : tan (π - α) = 1 / 3

theorem sin_cos_fraction (h : tan (π - α) = 1 / 3) : 
  (sin α + cos α) / (sin α - cos α) = -1 / 2 :=
sorry

end sin_cos_fraction_l67_67752


namespace product_of_nonreal_roots_l67_67739

theorem product_of_nonreal_roots (x : ℂ) :
  (x^4 - 6*x^3 + 15*x^2 - 20*x = 4019) →
  (∏ (z ∈ {root : ℂ | x^4 - 6*x^3 + 15*x^2 - 20*x = 4019 ∧ Im root ≠ 0}) = 4 + √4035) :=
begin
  sorry
end

end product_of_nonreal_roots_l67_67739


namespace age_problem_l67_67519

theorem age_problem (c b a : ℕ) (h1 : b = 2 * c) (h2 : a = b + 2) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end age_problem_l67_67519


namespace dirichlet_propositions_l67_67939

noncomputable def dirichlet_func (x : ℝ) : ℝ :=
  if x.is_rational then 1 else 0

theorem dirichlet_propositions :
  let f := dirichlet_func in
  (∀ x : ℝ, f (f x) = 1) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ (x : ℝ) (T : ℝ), T ≠ 0 ∧ T.is_rational → f (x + T) = f x) ∧
  (∃ (x1 x2 x3 : ℝ), 
    x1 = -Real.sqrt 3 / 3 ∧ 
    x2 = 0 ∧ 
    x3 = Real.sqrt 3 / 3 ∧ 
    f x1 = 0 ∧ 
    f x2 = 1 ∧ 
    f x3 = 0 ∧ 
    Equilateral (x1, f x1) (x2, f x2) (x3, f x3)) :=
by sorry

end dirichlet_propositions_l67_67939


namespace minimal_pos_period_tan_l67_67106

theorem minimal_pos_period_tan : 
  ∀ x : ℝ, ∃ T > 0, ∀ y : ℝ, y = (λ x, Real.tan(2 * x + Real.pi / 3)) (x + T) := 
begin
  sorry
end

end minimal_pos_period_tan_l67_67106


namespace triangle_cosine_law_example_l67_67321

theorem triangle_cosine_law_example 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (BC AB AC : ℝ) (hBC : BC = 6) (hAB : AB = 4) (hcosB : real.cos (real.pi / 3) = 1 / 3) :
  AC = 6 :=
by
  -- Placeholder for the actual proof
  sorry

end triangle_cosine_law_example_l67_67321


namespace parallelogram_product_constant_l67_67129

open EuclideanGeometry -- Open the Euclidean geometry namespace

noncomputable def parallelogram (A B C D : Point) : Prop :=
  collinear(A, B, D) = collinear(A, C, B) ∧ -- AB parallel to CD
  collinear(A, D, C) = collinear(A, B, D)   -- AD parallel to BC

/-- Given a parallelogram ABCD, and a line through C intersecting the extensions of AB and AD at points K and M. Prove that the product BK * DM is constant (i.e., BK * DM = BC * CD). -/
theorem parallelogram_product_constant
  (A B C D K M : Point)
  (hparallelogram : parallelogram A B C D)
  (hK : collinear A B K) -- K lies on the extension of AB
  (hM : collinear A D M) -- M lies on the extension of AD
  (hline : ¬ collinear K C M) -- K, C, and M are not collinear (the line through C is not degenerate)
  (hintersectK : between A B K) -- K is between A and the extension of B
  (hintersectM : between A D M) -- M is between A and the extension of D) :
  :
  (dist B K) * (dist D M) = (dist B C) * (dist C D) :=
sorry -- Placeholder for the proof

end parallelogram_product_constant_l67_67129


namespace bullet_trains_cross_time_l67_67526

noncomputable def time_to_cross
  (length1 length2 : ℝ)
  (speed1_kph speed2_kph : ℝ) : ℝ :=
let relative_speed := (speed1_kph + speed2_kph) * 1000 / 3600 in
let combined_length := length1 + length2 in
combined_length / relative_speed

theorem bullet_trains_cross_time
  (length1 length2 : ℝ)
  (speed1_kph speed2_kph : ℝ)
  (h_length1 : length1 = 140)
  (h_length2 : length2 = 170)
  (h_speed1 : speed1_kph = 60)
  (h_speed2 : speed2_kph = 40) :
  abs (time_to_cross length1 length2 speed1_kph speed2_kph - 11.16) < 1e-2 :=
by
  rw [h_length1, h_length2, h_speed1, h_speed2]
  unfold time_to_cross
  norm_num
  -- You would provide the details of the calculation if needed
  sorry

end bullet_trains_cross_time_l67_67526


namespace count_arrangements_eq_48_l67_67986

-- Definitions from the conditions
def num_red_people : ℕ := 2
def num_yellow_people : ℕ := 2
def num_blue_people : ℕ := 1

-- The total number of people
def total_people : ℕ := num_red_people + num_yellow_people + num_blue_people

-- The number of valid arrangements is 48
theorem count_arrangements_eq_48 (h1 : num_red_people = 2) (h2 : num_yellow_people = 2) (h3 : num_blue_people = 1) (h4 : total_people = 5):
  rearrangement_count num_red_people num_yellow_people num_blue_people 5 = 48 := sorry

end count_arrangements_eq_48_l67_67986


namespace solve_equation_l67_67733

/-- 
Given the expression y = (x^2 + 2x - 8) / (x + 2) and the equation y = 3x - 4,
prove that the only solution for x is 1, provided that x ≠ -2.
-/
theorem solve_equation : ∀ x : ℝ, x ≠ -2 → (↑(x^2 + 2 * x - 8) / (x + 2) = 3 * x - 4) ↔ x = 1 :=
by intros x h_y;
   sorry

end solve_equation_l67_67733


namespace total_cost_l67_67053

variables (p e n : ℕ) -- represent the costs of pencil, eraser, and notebook in cents

-- Given conditions
def conditions : Prop :=
  9 * p + 7 * e + 4 * n = 220 ∧
  p > n ∧ n > e ∧ e > 0

-- Prove the total cost
theorem total_cost (h : conditions p e n) : p + n + e = 26 :=
sorry

end total_cost_l67_67053


namespace task_completion_time_l67_67534

def work_done_by_one_man_per_day (w_fraction : ℝ) (m_men : ℕ) (d_days : ℕ) : ℝ :=
  w_fraction / (m_men * d_days)

def days_required_by_men (w_fraction : ℝ) (m_men : ℕ) (d_days : ℕ) (n_men : ℕ) : ℝ :=
  (1 : ℝ) / (n_men * (work_done_by_one_man_per_day w_fraction m_men d_days))

theorem task_completion_time :
  ∀ (w_fraction : ℝ) (m_men : ℕ) (d_days : ℕ) (n_men : ℕ),
  w_fraction = 1 / 3 → m_men = 10 → d_days = 3 → n_men = 35 →
  abs ((days_required_by_men w_fraction m_men d_days n_men) - 0.9) < 0.05 :=
by
  intros w_fraction m_men d_days n_men
  intro hw_fraction hm_men hd_days hn_men
  sorry

end task_completion_time_l67_67534


namespace solve_quadratics_l67_67560

theorem solve_quadratics (p q u v : ℤ)
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q)
  (h2 : u ≠ 0 ∧ v ≠ 0 ∧ u ≠ v)
  (h3 : p + q = -u)
  (h4 : pq = -v)
  (h5 : u + v = -p)
  (h6 : uv = -q) :
  p = -1 ∧ q = 2 ∧ u = -1 ∧ v = 2 :=
by {
  sorry
}

end solve_quadratics_l67_67560


namespace ways_to_divide_friends_l67_67374

theorem ways_to_divide_friends : 
  let n := 6
  let teams := 3
  n ∈ ℕ ∧ teams ∈ ℕ ∧ n = 6 ∧ teams = 3 →
  (let ways := teams^n)  
  ways = 729 := 
begin
  intros n teams _ _,
  have ways_eq_3_pow_6 : ways = 3^6 := by rw n; rw teams,
  rw ways_eq_3_pow_6,
  norm_num,
end

end ways_to_divide_friends_l67_67374


namespace number_of_rationals_l67_67737

theorem number_of_rationals : 
  ∃ count, 
    (∀ m n : ℕ, 0 < m ∧ m < n ∧ m * n = nat.factorial 25 ∧ nat.gcd m n = 1) → count = 256 :=
sorry

end number_of_rationals_l67_67737


namespace distinct_values_of_exponentiation_l67_67590

theorem distinct_values_of_exponentiation : 
  (∃ (values : Finset ℕ), values = {evaluate ((3 : ℕ) ^ (3 ^ (3 ^ 3))), 
                                 evaluate ((3 ^ 3) ^ 3), 
                                 ((3 ^ (3 ^ (3 ^ 3)))), 
                                 ((3 ^ (3 ^ 3)) ^ 3),
                                 (3 ^ ((3 ^ 3) ^ 3))} ∧ 
    values.card = 1) := 
sorry

end distinct_values_of_exponentiation_l67_67590


namespace area_of_quadrilateral_ABCD_62p5_sqrt_3_l67_67063

theorem area_of_quadrilateral_ABCD_62p5_sqrt_3
  (A B C D E : Type)
  (AC CD AE : ℝ)
  (angle_ABC angle_ACD : ℝ) :
  angle_ABC = 90 ∧ angle_ACD = 60 ∧ AC = 25 ∧ CD = 10 ∧ AE = 15 → 
  (area_of_quadrilateral A B C D = 62.5 * Real.sqrt 3) :=
sorry

end area_of_quadrilateral_ABCD_62p5_sqrt_3_l67_67063


namespace even_a_given_perfect_square_l67_67874

theorem even_a_given_perfect_square (a n : ℤ) (h1 : a > 2) (h2 : n > 1) (h3 : ∃ k : ℤ, a^n - 2^n = k^2) : Even a :=
sorry

end even_a_given_perfect_square_l67_67874


namespace triangle_third_side_l67_67007

theorem triangle_third_side (DE DF : ℝ) (E F : ℝ) (EF : ℝ) 
    (h₁ : DE = 7) 
    (h₂ : DF = 21) 
    (h₃ : E = 3 * F) : EF = 14 * Real.sqrt 2 :=
sorry

end triangle_third_side_l67_67007


namespace optimal_school_location_l67_67385

variable (d : ℝ) (students_A students_B : ℕ)

def is_minimal_total_distance_location (location : string) : Prop :=
  location = "Srednie Boltay"

theorem optimal_school_location :
  students_A = 50 →
  students_B = 100 →
  is_minimal_total_distance_location "Srednie Boltay" :=
by
  intros hA hB
  sorry

end optimal_school_location_l67_67385


namespace contractor_initial_hire_l67_67546

theorem contractor_initial_hire :
  ∃ (P : ℕ), 
    (∀ (total_work : ℝ), 
      (P * 20 = (1/4) * total_work) ∧ 
      ((P - 2) * 75 = (3/4) * total_work)) → 
    P = 10 :=
by
  sorry

end contractor_initial_hire_l67_67546


namespace plane_equation_through_A_perpendicular_to_BC_l67_67513

-- Definitions of the points
def A : ℝ × ℝ × ℝ := (5, -1, 2)
def B : ℝ × ℝ × ℝ := (2, -4, 3)
def C : ℝ × ℝ × ℝ := (4, -1, 3)

-- Define the vector BC
def BC : ℝ × ℝ × ℝ := (C.1 - B.1, C.2 - B.2, C.3 - B.3)

-- The normal vector of the plane is the same as BC (since plane is perpendicular to BC)
def normal_vector : ℝ × ℝ × ℝ := BC

-- Define the equation of the plane passing through point A and perpendicular to vector BC
theorem plane_equation_through_A_perpendicular_to_BC :
  ∃ (a b c d : ℝ), (a, b, c) = normal_vector ∧ (a * 5 + b * (-1) + c * 2 + d = 0) ∧ (∀ (x y z : ℝ), a * x + b * y + c * z + d = 0 ↔ 2 * x + 3 * y - 7 = 0) := sorry

end plane_equation_through_A_perpendicular_to_BC_l67_67513


namespace molly_ate_11_suckers_l67_67923

/-- 
Sienna gave Bailey half of her suckers.
Jen ate 11 suckers and gave the rest to Molly.
Molly ate some suckers and gave the rest to Harmony.
Harmony kept 3 suckers and passed the remainder to Taylor.
Taylor ate one and gave the last 5 suckers to Callie.
How many suckers did Molly eat?
-/
theorem molly_ate_11_suckers
  (sienna_bailey_suckers : ℕ)
  (jen_ate : ℕ)
  (jens_remainder_to_molly : ℕ)
  (molly_remainder_to_harmony : ℕ) 
  (harmony_kept : ℕ) 
  (harmony_remainder_to_taylor : ℕ)
  (taylor_ate : ℕ)
  (taylor_remainder_to_callie : ℕ)
  (jen_condition : jen_ate = 11)
  (harmony_condition : harmony_kept = 3)
  (taylor_condition : taylor_ate = 1)
  (taylor_final_suckers : taylor_remainder_to_callie = 5) :
  molly_ate = 11 :=
by sorry

end molly_ate_11_suckers_l67_67923


namespace range_of_f_l67_67041

open Nat

def f (n : ℕ) (k : ℕ) : ℕ :=
  floor ((n + (n : ℝ)^(1 / (k : ℝ)))^(1 / (k : ℝ))) + n

theorem range_of_f (k : ℕ) (hk : k > 0) :
  (∃ (n : ℕ), f n k = m) ↔ m ∈ (ℕ \ {t | ∃ t : ℕ, t = t^k}) :=
sorry

end range_of_f_l67_67041


namespace sin_240_eq_neg_sqrt3_div_2_l67_67639

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67639


namespace find_f_50_l67_67942

variable {f : ℝ → ℝ}
variable (hf : ∀ x y : ℝ, 0 < x ∧ 0 < y → x * f(y) - y * f(x) = x * y * f(x / y))

theorem find_f_50 : f 50 = 0 :=
by
  sorry

end find_f_50_l67_67942


namespace partition_triangle_l67_67688

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l67_67688


namespace cake_angle_between_adjacent_pieces_l67_67252

theorem cake_angle_between_adjacent_pieces 
  (total_angle : ℝ := 360)
  (total_pieces : ℕ := 10)
  (eaten_pieces : ℕ := 1)
  (angle_per_piece := total_angle / total_pieces)
  (remaining_pieces := total_pieces - eaten_pieces)
  (new_angle_per_piece := total_angle / remaining_pieces) :
  (new_angle_per_piece - angle_per_piece = 4) := 
by
  sorry

end cake_angle_between_adjacent_pieces_l67_67252


namespace divisible_by_20_ordered_triplets_l67_67743

theorem divisible_by_20_ordered_triplets :
  let valid_triplets := {triplet : ℕ × ℕ × ℕ // triplet.1 > 0 ∧ triplet.2 > 0 ∧ triplet.3 > 0 ∧ triplet.1 < 10 ∧ triplet.2 < 10 ∧ triplet.3 < 10 ∧ (triplet.1 * triplet.2 * triplet.3) % 20 = 0}
  in Fintype.card valid_triplets = 72 :=
by
  sorry

end divisible_by_20_ordered_triplets_l67_67743


namespace sin_240_eq_neg_sqrt3_div_2_l67_67638

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67638


namespace max_value_of_seq_diff_l67_67409

theorem max_value_of_seq_diff :
  ∀ (a : Fin 2017 → ℝ),
    a 0 = a 2016 →
    (∀ i : Fin 2015, |a i + a (i+2) - 2 * a (i+1)| ≤ 1) →
    ∃ b : ℝ, b = 508032 ∧ ∀ i j, 1 ≤ i → i < j → j ≤ 2017 → |a i - a j| ≤ b :=
  sorry

end max_value_of_seq_diff_l67_67409


namespace real_roots_for_all_a_b_l67_67457

theorem real_roots_for_all_a_b (a b : ℝ) : ∃ x : ℝ, (x^2 / (x^2 - a^2) + x^2 / (x^2 - b^2) = 4) :=
sorry

end real_roots_for_all_a_b_l67_67457


namespace seeds_per_row_top_bed_l67_67362

/-- 
Grace’s garden configuration:
- 2 large beds on top, each with 4 rows.
- Each of the 2 medium beds on the bottom has 3 rows, each row with 20 seeds.
- The total number of seeds in all the beds is 320.
Prove that the number of seeds sown per row in the top bed is 25.
-/
theorem seeds_per_row_top_bed :
  let large_bed_rows := 2 * 4,
      medium_bed_rows := 2 * 3,
      total_seeds := 320,
      seeds_per_medium_row := 20 in
  ∃ seeds_per_top_row : ℕ,
    8 * seeds_per_top_row = total_seeds - seeds_per_medium_row * medium_bed_rows ∧ 
    seeds_per_top_row = 25 :=
begin
  sorry
end

end seeds_per_row_top_bed_l67_67362


namespace probability_of_multiple_of_45_l67_67522

-- Definitions for conditions
def singleDigitMultiplesOf3 : Finset ℕ := {3, 6, 9}
def primesLessThan20 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Function to check if a number is a multiple of 45
def isMultipleOf45 (n : ℕ) : Prop := 45 ∣ n

-- Probability calculation
def probability : ℕ := 
  let favorableOutcomes := (singleDigitMultiplesOf3.product primesLessThan20).filter (λ p => isMultipleOf45 (p.1 * p.2))
  favorableOutcomes.card / (singleDigitMultiplesOf3.card * primesLessThan20.card)

theorem probability_of_multiple_of_45 : 
  probability = 1 / 24 := 
  by
  -- Proceed with the proof
  sorry

end probability_of_multiple_of_45_l67_67522


namespace new_students_average_age_l67_67089

theorem new_students_average_age :
  let O := 12 in
  let A_O := 40 in
  let N := 12 in
  let new_avg := A_O - 4 in
  let total_age_before := O * A_O in
  let total_age_after := (O + N) * new_avg in
  ∃ A_N : ℕ, total_age_before + N * A_N = total_age_after ∧ A_N = 32 :=
by
  let O := 12
  let A_O := 40
  let N := 12
  let new_avg := A_O - 4
  let total_age_before := O * A_O
  let total_age_after := (O + N) * new_avg
  use 32
  split
  · sorry
  · rfl

end new_students_average_age_l67_67089


namespace sin_240_l67_67625

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67625


namespace bus_problem_l67_67533

theorem bus_problem (x : ℕ)
  (h1 : 28 + 82 - x = 30) :
  82 - x = 2 :=
by {
  sorry
}

end bus_problem_l67_67533


namespace smallest_N_l67_67507

theorem smallest_N (N : ℕ) (h : 7 * N = 999999) : N = 142857 :=
sorry

end smallest_N_l67_67507


namespace sin_240_eq_neg_sqrt3_div_2_l67_67680

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67680


namespace sin_45_eq_sqrt2_div_2_l67_67592

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt2_div_2_l67_67592


namespace correct_word_l67_67401

-- Declare the sentence as a constant structure 
constant options : list string := ["ones", "one", "that", "those"]

-- Define the previous context that air in the city is being referred to
constant in_city : string := "air in the city"

-- Define the function that chooses the correct replacement word
def correct_choice (previous_context : string) (candidates : list string) : string :=
  if previous_context = "air in the city" then
    "that" -- Assume the logic is correct; this is purely structural
  else
    "invalid"

-- Statement to prove
theorem correct_word : correct_choice in_city options = "that" :=
by
  sorry

end correct_word_l67_67401


namespace find_k_range_l67_67900

open Real

theorem find_k_range (f : ℝ → ℝ) (mono : Monotonic f)
    (add_f : ∀ x y : ℝ, f (x + y) = f x + f y) (f_one : f 1 = -1) :
    (∀ t : ℝ, 0 < t → f (k * log2 t) + f (log2 t - (log2 t) ^ 2 - 2) > 0)
    ↔ -1 - 2 * sqrt 2 < k ∧ k < -1 + 2 * sqrt 2 := sorry

end find_k_range_l67_67900


namespace prove_correct_line_eq_l67_67100

def line_intersection (l1 l2 : ℝ × ℝ × ℝ) : ℝ × ℝ :=
let (a1, b1, c1) := l1 in
let (a2, b2, c2) := l2 in
let det := a1 * b2 - a2 * b1 in
((b1 * c2 - b2 * c1) / det, ((c1 * a2) - (a1 * c2)) / det)

noncomputable def satisfies_perpendicularity (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
let (a1, b1, c1) := l1 in
let (a2, b2, c2) := l2 in
a1 * a2 + b1 * b2 = 0

noncomputable def is_solution (eq_intercept : ℝ × ℝ → ℝ × ℝ × ℝ) : Prop :=
eq_intercept (line_intersection (3, 1, 0) (1, 1, -2)) = (1, -2, 7) 

theorem prove_correct_line_eq :
  satisfies_perpendicularity (2, 1, 3) (1, -2, 7) ∧ is_solution (λ p, (1, -2, -7)) :=
by
  sorry

end prove_correct_line_eq_l67_67100


namespace fish_population_estimate_l67_67104

theorem fish_population_estimate 
  (marked_initial : ℕ) (total_second_catch : ℕ) (marked_second_catch : ℕ) : 
  marked_initial = 30 → 
  total_second_catch = 50 → 
  marked_second_catch = 2 → 
  ∃ (x : ℕ), x = 750 :=
by
  intros
  use 750
  sorry

end fish_population_estimate_l67_67104


namespace area_of_rhombus_area_enclosed_by_graph_l67_67140

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem area_of_rhombus (d₁ d₂ : ℝ) (h₁ : d₁ = 10) (h₂ : d₂ = 4) : 
  ∃ area, area = (1 / 2) * d₁ * d₂ ∧ area = 20 := by
  use (1 / 2) * d₁ * d₂
  split
  · exact rfl
  · rw [h₁, h₂]
    norm_num
    exact rfl

theorem area_enclosed_by_graph : 
  ∃ area, (∀ (x y : ℝ), abs (2 * x) + abs (5 * y) ≤ 10) → area = 20 := by
  have d₁ : ℝ := 10
  have d₂ : ℝ := 4
  let area := (1 / 2) * d₁ * d₂
  use area
  intro _
  apply area_of_rhombus
  { exact rfl }
  { exact rfl }

end area_of_rhombus_area_enclosed_by_graph_l67_67140


namespace binary_1010_to_decimal_l67_67685

theorem binary_1010_to_decimal :
  binary_to_decimal [1, 0, 1, 0] = 10 :=
sorry

end binary_1010_to_decimal_l67_67685


namespace find_inner_circle_radius_of_trapezoid_l67_67027

noncomputable def radius_of_inner_circle (k m n p : ℤ) : ℝ :=
  (-k + m * Real.sqrt n) / p

def is_equivalent (a b : ℝ) : Prop := a = b

theorem find_inner_circle_radius_of_trapezoid :
  ∃ (r : ℝ), is_equivalent r (radius_of_inner_circle 123 104 3 29) :=
by
  let r := radius_of_inner_circle 123 104 3 29
  have h1 :  (4^2 + (Real.sqrt (r^2 + 8 * r))^2 = (r + 4)^2) := sorry
  have h2 :  (3^2 + (Real.sqrt (r^2 + 6 * r))^2 = (r + 3)^2) := sorry
  have height_eq : Real.sqrt 13 = (Real.sqrt (r^2 + 6 * r) + Real.sqrt (r^2 + 8 * r)) := sorry
  use r
  exact sorry

end find_inner_circle_radius_of_trapezoid_l67_67027


namespace second_train_further_l67_67995

-- Define the speeds of the two trains
def speed_train1 : ℝ := 50
def speed_train2 : ℝ := 60

-- Define the total distance between points A and B
def total_distance : ℝ := 1100

-- Define the distances traveled by the two trains when they meet
def distance_train1 (t: ℝ) : ℝ := speed_train1 * t
def distance_train2 (t: ℝ) : ℝ := speed_train2 * t

-- Define the meeting condition
def meeting_condition (t: ℝ) : Prop := distance_train1 t + distance_train2 t = total_distance

-- Prove the distance difference
theorem second_train_further (t: ℝ) (h: meeting_condition t) : distance_train2 t - distance_train1 t = 100 :=
sorry

end second_train_further_l67_67995


namespace sin_240_deg_l67_67609

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67609


namespace kevin_food_expenditure_l67_67074

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end kevin_food_expenditure_l67_67074


namespace triangle_parts_sum_eq_l67_67706

-- Define the total sum of numbers in the triangle
def total_sum : ℕ := 63

-- Define the required sum for each part
def required_sum : ℕ := 21

-- Define the possible parts that sum to the required sum
def part1 : list ℕ := [10, 6, 5]  -- this sums to 21
def part2 : list ℕ := [10, 5, 4, 1, 1]  -- this sums to 21

def part_sum (part : list ℕ) : ℕ := part.sum

-- The main theorem
theorem triangle_parts_sum_eq : 
  part_sum part1 = required_sum ∧ 
  part_sum part2 = required_sum ∧ 
  part_sum (list.drop (list.length part1 + list.length part2) [10, 6, 5, 10, 5, 4, 1, 1]) = required_sum :=
by
  sorry

end triangle_parts_sum_eq_l67_67706


namespace triangle_parts_sum_eq_l67_67707

-- Define the total sum of numbers in the triangle
def total_sum : ℕ := 63

-- Define the required sum for each part
def required_sum : ℕ := 21

-- Define the possible parts that sum to the required sum
def part1 : list ℕ := [10, 6, 5]  -- this sums to 21
def part2 : list ℕ := [10, 5, 4, 1, 1]  -- this sums to 21

def part_sum (part : list ℕ) : ℕ := part.sum

-- The main theorem
theorem triangle_parts_sum_eq : 
  part_sum part1 = required_sum ∧ 
  part_sum part2 = required_sum ∧ 
  part_sum (list.drop (list.length part1 + list.length part2) [10, 6, 5, 10, 5, 4, 1, 1]) = required_sum :=
by
  sorry

end triangle_parts_sum_eq_l67_67707


namespace smaller_rectangle_area_l67_67568

-- Define the conditions
def large_rectangle_length : ℝ := 40
def large_rectangle_width : ℝ := 20
def smaller_rectangle_length : ℝ := large_rectangle_length / 2
def smaller_rectangle_width : ℝ := large_rectangle_width / 2

-- Define what we want to prove
theorem smaller_rectangle_area : 
  (smaller_rectangle_length * smaller_rectangle_width = 200) :=
by
  sorry

end smaller_rectangle_area_l67_67568


namespace exists_monochromatic_triangle_l67_67982

theorem exists_monochromatic_triangle :
  ∀ (points : Finset ℕ), points.card = 6 → 
  (∀ (x y : ℕ), x ∈ points → y ∈ points → x ≠ y → (coloring : (x, y) → ℤ)) → 
  (∃ (triangle : Finset (Finset ℕ)), triangle.card = 3 ∧ 
  (∀ (x y : ℕ), (x ∈ triangle) → (y ∈ triangle) → (x, y) ∈ coloring (x, y) ∨ coloring (x, y) ∧ 
  (0 ≤ coloring (x, y) ∧ coloring (x, y) ≤ 1)) :=
by
  sorry

end exists_monochromatic_triangle_l67_67982


namespace exp_function_not_increasing_l67_67953

open Real

theorem exp_function_not_increasing (a : ℝ) (x : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : a < 1) :
  ¬(∀ x₁ x₂ : ℝ, x₁ < x₂ → a^x₁ < a^x₂) := by
  sorry

end exp_function_not_increasing_l67_67953


namespace good_deed_done_by_C_l67_67985

def did_good (A B C : Prop) := 
  (¬A ∧ ¬B ∧ C) ∨ (¬A ∧ B ∧ ¬C) ∨ (A ∧ ¬B ∧ ¬C)

def statement_A (B : Prop) := B
def statement_B (B : Prop) := ¬B
def statement_C (C : Prop) := ¬C

theorem good_deed_done_by_C (A B C : Prop)
  (h_deed : (did_good A B C))
  (h_statement : (statement_A B ∧ ¬statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ ¬statement_B B ∧ statement_C C)) :
  C :=
by 
  sorry

end good_deed_done_by_C_l67_67985


namespace relatively_prime_number_exists_l67_67780

def gcd (a b : ℕ) : ℕ := a.gcd b

def is_relatively_prime_to_all (n : ℕ) (lst : List ℕ) : Prop :=
  ∀ m ∈ lst, m ≠ n → gcd n m = 1

def given_numbers : List ℕ := [20172017, 20172018, 20172019, 20172020, 20172021]

theorem relatively_prime_number_exists :
  ∃ n ∈ given_numbers, is_relatively_prime_to_all n given_numbers := 
begin
  use 20172019,
  split,
  { -- Show 20172019 is in the list
    simp },
  { -- Prove 20172019 is relatively prime to all other numbers in the list
    intros m h1 h2,
    -- Further proof goes here
    sorry
  }
end

end relatively_prime_number_exists_l67_67780


namespace first_no_buses_time_first_unable_dispatch_time_min_dispatch_interval_min_buses_added_l67_67196

noncomputable def dispatch_conditions (minute_elapsed : ℕ) : Prop :=
  let S := minute_elapsed / 6 + 1 in
  let y := minute_elapsed / 8 - 1 in
  S == y + 15

noncomputable def no_buses_parking_lot (t : ℕ) : Prop :=
  let minute_elapsed := t - 360 in
  let S := minute_elapsed / 6 + 1 in
  8 * (S - 15) > 6 * (S - 1) - 3

theorem first_no_buses_time : no_buses_parking_lot  690 := 
sorry

noncomputable def unable_on_time_dispatch (t : ℕ) : Prop :=
  let minute_elapsed := t - 360 in
  let S := minute_elapsed / 6 + 1 in
  8 * (S - 16) > 6 * (S - 1) - 3

theorem first_unable_dispatch_time : unable_on_time_dispatch 714 :=
sorry

noncomputable def dispatch_interval (a : ℕ) : Prop :=
  let time_elapsed := 840 in
  let S := time_elapsed / a + 1 in
  let y := time_elapsed / 8 - 1 in
  a * (S - 1) >= 840 ∧ 8 * (y - 1) <= 840 - 3

theorem min_dispatch_interval : dispatch_interval 8 :=
sorry

noncomputable def buses_needed (m : ℕ) : Prop :=
  let time_duration := 840 in
  (time_duration / 6 + 1) = 141 - (15 + m) ∧ (time_duration - 3) / 8 = 104

theorem min_buses_added : buses_needed 22 :=
sorry

end first_no_buses_time_first_unable_dispatch_time_min_dispatch_interval_min_buses_added_l67_67196


namespace loan_amount_l67_67064

theorem loan_amount (R T SI : ℕ) (hR : R = 7) (hT : T = 7) (hSI : SI = 735) : 
  ∃ P : ℕ, P = 1500 := 
by 
  sorry

end loan_amount_l67_67064


namespace checkered_triangle_division_l67_67703

-- Define the conditions as assumptions
variable (T : Set ℕ) 
variable (sum_T : Nat) (h_sumT : sum_T = 63)
variable (part1 part2 part3 : Set ℕ)
variable (sum_part1 sum_part2 sum_part3 : Nat)
variable (h_part1 : sum part1 = 21)
variable (h_part2 : sum part2 = 21)
variable (h_part3 : sum part3 = 21)

-- Define the goal as a theorem
theorem checkered_triangle_division : 
  (∃ part1 part2 part3 : Set ℕ, sum part1 = 21 ∧ sum part2 = 21 ∧ sum part3 = 21 ∧ Disjoint part1 (part2 ∪ part3) ∧ Disjoint part2 part3 ∧ T = part1 ∪ part2 ∪ part3) :=
sorry

end checkered_triangle_division_l67_67703


namespace unique_function_satisfying_conditions_l67_67164

noncomputable def X := Set.Ici (1 : ℝ)

theorem unique_function_satisfying_conditions :
  ∀ f : X → ℝ, 
  (∀ x ∈ X, f x ≤ 2 * x + 2) ∧ 
  (∀ x ∈ X, x * f (x + 1) = (f x)^2 - 1) → 
  (∀ x ∈ X, f x = x + 1) :=
by
  intro f
  intro h
  sorry

end unique_function_satisfying_conditions_l67_67164


namespace Laura_more_than_200_paperclips_on_Friday_l67_67868

theorem Laura_more_than_200_paperclips_on_Friday:
  ∀ (n : ℕ), (n = 4 ∨ n = 0 ∨ n ≥ 1 ∧ (n - 1 = 0 ∨ n = 1) → 4 * 3 ^ n > 200) :=
by
  sorry

end Laura_more_than_200_paperclips_on_Friday_l67_67868


namespace problem_l67_67331

theorem problem
  (a b c d e : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (- (a * b))^2009 - (c + d)^2010 - e^2011 = 0 := 
by
  sorry

end problem_l67_67331


namespace weight_of_apples_l67_67496

-- Definitions based on conditions
def total_weight : ℕ := 10
def weight_orange : ℕ := 1
def weight_grape : ℕ := 3
def weight_strawberry : ℕ := 3

-- Prove that the weight of apples is 3 kilograms
theorem weight_of_apples : (total_weight - (weight_orange + weight_grape + weight_strawberry)) = 3 :=
by
  sorry

end weight_of_apples_l67_67496


namespace clerks_needed_eq_84_l67_67203

def forms_processed_per_hour : ℕ := 25
def type_a_forms_count : ℕ := 3000
def type_b_forms_count : ℕ := 4000
def type_a_form_time_minutes : ℕ := 3
def type_b_form_time_minutes : ℕ := 4
def working_hours_per_day : ℕ := 5
def total_minutes_in_an_hour : ℕ := 60
def forms_time_needed (count : ℕ) (time_per_form : ℕ) : ℕ := count * time_per_form
def total_forms_time_needed : ℕ := forms_time_needed type_a_forms_count type_a_form_time_minutes +
                                    forms_time_needed type_b_forms_count type_b_form_time_minutes
def total_hours_needed : ℕ := total_forms_time_needed / total_minutes_in_an_hour
def clerk_hours_needed : ℕ := total_hours_needed / working_hours_per_day
def required_clerks : ℕ := Nat.ceil (clerk_hours_needed)

theorem clerks_needed_eq_84 :
  required_clerks = 84 :=
by
  sorry

end clerks_needed_eq_84_l67_67203


namespace cos_double_angle_zero_l67_67376

variable (θ : ℝ)

-- Conditions
def tan_eq_one : Prop := Real.tan θ = 1

-- Objective
theorem cos_double_angle_zero (h : tan_eq_one θ) : Real.cos (2 * θ) = 0 :=
sorry

end cos_double_angle_zero_l67_67376


namespace cos_theta_example_l67_67837

variables (a b : ℝ × ℝ) (θ : ℝ)

def cos_theta (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

theorem cos_theta_example :
  let a := (2, -1)
  let b := (1, 3)
  cos_theta a b = -(Real.sqrt 2) / 10 :=
by
  sorry

end cos_theta_example_l67_67837


namespace value_of_a_plus_b_l67_67034

theorem value_of_a_plus_b (a b : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b) 
  (hg : ∀ x, g x = -3 * x + 2)
  (hgf : ∀ x, g (f x) = -2 * x - 3) :
  a + b = 7 / 3 :=
by
  sorry

end value_of_a_plus_b_l67_67034


namespace double_decker_bus_total_capacity_l67_67002

-- Define conditions for the lower floor seating
def lower_floor_left_seats : Nat := 15
def lower_floor_right_seats : Nat := 12
def lower_floor_priority_seats : Nat := 4

-- Each seat on the left and right side of the lower floor holds 2 people
def lower_floor_left_capacity : Nat := lower_floor_left_seats * 2
def lower_floor_right_capacity : Nat := lower_floor_right_seats * 2
def lower_floor_priority_capacity : Nat := lower_floor_priority_seats * 1

-- Define conditions for the upper floor seating
def upper_floor_left_seats : Nat := 20
def upper_floor_right_seats : Nat := 20
def upper_floor_back_capacity : Nat := 15

-- Each seat on the left and right side of the upper floor holds 3 people
def upper_floor_left_capacity : Nat := upper_floor_left_seats * 3
def upper_floor_right_capacity : Nat := upper_floor_right_seats * 3

-- Total capacity of lower and upper floors
def lower_floor_total_capacity : Nat := lower_floor_left_capacity + lower_floor_right_capacity + lower_floor_priority_capacity
def upper_floor_total_capacity : Nat := upper_floor_left_capacity + upper_floor_right_capacity + upper_floor_back_capacity

-- Assert the total capacity
def bus_total_capacity : Nat := lower_floor_total_capacity + upper_floor_total_capacity

-- Prove that the total bus capacity is 193 people
theorem double_decker_bus_total_capacity : bus_total_capacity = 193 := by
  sorry

end double_decker_bus_total_capacity_l67_67002


namespace find_unit_prices_and_purchasing_plans_l67_67544

theorem find_unit_prices_and_purchasing_plans :
  (∃ x y : ℕ, (20 * x + 10 * y = 1100) ∧ (25 * x + 20 * y = 1750) ∧ (x = 30) ∧ (y = 50)) ∧
  (∃ plans : Finset ℕ, (∀ m ∈ plans, 25 ≤ m ∧ m ≤ 100 / 3) ∧ (plans.card = 9)) :=
by
  existsi 30, 50
  split
  · norm_num
  · existsi (Finset.range' 25 (100 / 3 - 25 + 1)).card
    split
    ·
      intros m hm
      norm_num at hm
      split
      · exact le_of_lt_succ hm.left
      · exact lt_of_lt_of_le hm.right (by norm_num)
    · have h : (Finset.range' 25 (100 / 3 - 25 + 1)).card = 9 := by norm_num
      exact h

end find_unit_prices_and_purchasing_plans_l67_67544


namespace area_of_abs_2x_plus_abs_5y_eq_10_l67_67138

theorem area_of_abs_2x_plus_abs_5y_eq_10 : ∀ (x y : ℝ), |2 * x| + |5 * y| = 10 → ∃ A : ℝ, A = 20 :=
by
  intros x y h
  use 20
  sorry

end area_of_abs_2x_plus_abs_5y_eq_10_l67_67138


namespace triangle_ineq_proof_l67_67843

noncomputable def triangle_inequality
  (a b c h_a h_b h_c Δ : ℝ) 
  (h_area_a: Δ = 1/2 * a * h_a)
  (h_area_b: Δ = 1/2 * b * h_b)
  (h_area_c: Δ = 1/2 * c * h_c) : Prop :=
  a * h_b + b * h_c + c * h_a ≥ 6 * Δ

theorem triangle_ineq_proof
  (a b c h_a h_b h_c Δ : ℝ) 
  (h_area_a: Δ = 1/2 * a * h_a)
  (h_area_b: Δ = 1/2 * b * h_b)
  (h_area_c: Δ = 1/2 * c * h_c) : triangle_inequality a b c h_a h_b h_c Δ :=
  by
    sorry

end triangle_ineq_proof_l67_67843


namespace prove_a_plus_b_l67_67899

theorem prove_a_plus_b (a b k : ℝ) (h_ext : ∀ x, x = 0 → (2 * a * x + b = 0))
  (h_tangent : ∀ f (h1 : f 1 = a * 1^2 + b * 1 + k) (h2 : f'(1) = 2 * a * 1 + b), 
   (f'(1) = 2) ∨ (f'(1) = -1/2))
  (h_k : k > 0) :
  a + b = 1 :=
by 
  sorry

end prove_a_plus_b_l67_67899


namespace part1_tangent_line_eq_part2_zeros_distance_l67_67347

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  sin (Real.pi * x) - 3 * (x - 1) * Real.log (x + 1) - m

theorem part1_tangent_line_eq (m : ℝ) : (m = 0) →
  let f := f x 0 in
  ∀ x, ∃ y, y = Real.pi + 3 * x :=
begin
  sorry
end

theorem part2_zeros_distance (m : ℝ) (x1 x2 : ℝ) :
  (0 ≤ x1 ∧ x1 ≤ 1) ∧ (0 ≤ x2 ∧ x2 ≤ 1) →
  (f x1 m = 0) ∧ (f x2 m = 0) → 
  |x1 - x2| ≤ 1 - (2 * m) / (Real.pi + 3) :=
begin
  sorry
end

end part1_tangent_line_eq_part2_zeros_distance_l67_67347


namespace sin_240_eq_neg_sqrt3_over_2_l67_67650

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67650


namespace time_to_drain_tank_due_to_leak_l67_67559

noncomputable def timeToDrain (P L : ℝ) : ℝ := (1 : ℝ) / L

theorem time_to_drain_tank_due_to_leak (P L : ℝ)
  (hP : P = 0.5)
  (hL : P - L = 5/11) :
  timeToDrain P L = 22 :=
by
  -- to state what needs to be proved here
  sorry

end time_to_drain_tank_due_to_leak_l67_67559


namespace kevin_food_expenditure_l67_67075

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end kevin_food_expenditure_l67_67075


namespace range_f_l67_67719

-- Define the operation a * b.
def star (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f.
def f (x : ℝ) : ℝ := 2^x * 2^(-x)

-- State the theorem to prove the range of f(x).
theorem range_f :
  (∀ y : ℝ, (∃ x : ℝ, f(x) = y) ↔ 0 < y ∧ y ≤ 1) :=
sorry

end range_f_l67_67719


namespace find_ratio_l67_67029

noncomputable theory

variables (O : ℝ × ℝ × ℝ := (0, 0, 0))
variables (a b c : ℝ) (A B C : ℝ × ℝ × ℝ)
variables (alpha beta gamma : ℝ)
variables (s t u : ℝ := 1, u = 1)

def passes_through {x y z : ℝ} (A : ℝ × ℝ × ℝ) (B : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) : Prop :=
  ∀ (x y z : ℝ), A = (alpha, 0, 0) ∧ B = (0, beta, 0) ∧ C = (0, 0, gamma)

theorem find_ratio (p q r : ℝ) (A_intersects_x : A = (alpha, 0, 0))
  (B_intersects_y : B = (0, beta, 0))
  (C_intersects_z : C = (0, 0, gamma))
  (plane_eq : (a, b, c))
  (sphere_center : ((p + 1), (q + 1), (r + 1)) =
    (0, 0, 0))
  (equidist_OA : (p + 1)^2 +(q + 1)^2 + (r + 1)^2 = ((p + 1) - alpha)^2 + (q + 1)^2 + (r + 1)^2)
  (equidist_OB : (p + 1)^2 + (q + 1)^2 + (r + 1)^2 = (p + 1)^2 + ((q + 1) - beta)^2 + (r + 1)^2)
  (equidist_OC : (p + 1)^2 + (q + 1)^2 + (r + 1)^2 = (p + 1)^2 + (q + 1)^2 + ((r + 1) - gamma)^2) :
  (a / p + b / q + c / r = 2) :=
sorry

end find_ratio_l67_67029


namespace infinite_sum_set_l67_67180

open Real

theorem infinite_sum_set (A : ℝ) (hA : 0 < A) (n : ℕ) (hn : 0 < n)
  (x : ℕ → ℝ) (hxpos : ∀ i, 0 < x i)
  (hxs : tsum x = A) :
  (n > 1 → (set.range (λ k, tsum (λ i, (x i)^n)) = set.Ioc 0 (A^n))) ∧ 
  (n = 1 → (set.range (λ k, tsum (λ i, (x i)^1)) = {A})) := 
by sorry  -- proof goes here

end infinite_sum_set_l67_67180


namespace line_in_plane_if_two_points_in_plane_l67_67481

theorem line_in_plane_if_two_points_in_plane (A B : Point) (l : Line) (α : Plane)
    (hA_l : A ∈ l) (hB_l : B ∈ l) (hA_α : A ∈ α) (hB_α : B ∈ α) :
    l ⊆ α :=
sorry

end line_in_plane_if_two_points_in_plane_l67_67481


namespace roots_of_unity_reals_l67_67978

theorem roots_of_unity_reals (S : Finset ℂ) (h : ∀ z ∈ S, z ^ 30 = 1) (h_card : S.card = 30) : 
  (S.filter (λ z, z ^ 10 ∈ Set ℝ)).card = 10 := 
sorry

end roots_of_unity_reals_l67_67978


namespace class_mean_score_l67_67384

theorem class_mean_score:
  ∀ (n: ℕ) (m: ℕ) (a b: ℕ),
  n + m = 50 →
  n * a = 3400 →
  m * b = 750 →
  a = 85 →
  b = 75 →
  (n * a + m * b) / (n + m) = 83 :=
by
  intros n m a b h1 h2 h3 h4 h5
  sorry

end class_mean_score_l67_67384


namespace coin_flip_probability_l67_67377

theorem coin_flip_probability :
  let P : (ℕ → Bool) → ℕ → ℚ := λ seq n, if seq n then (1/2 : ℚ) else (1/2 : ℚ)
  let E : (ℕ → Bool) → Prop := λ seq, seq 0 ∧ ¬ (seq 1 ∨ seq 2 ∨ seq 3 ∨ seq 4)
  (∑' (seq : ℕ → Bool), if E seq then (∏ n, P seq n) else 0) = 1/32 := sorry

end coin_flip_probability_l67_67377


namespace number_of_continents_collected_l67_67247

-- Definitions of the given conditions
def books_per_continent : ℕ := 122
def total_books : ℕ := 488

-- The mathematical statement to be proved
theorem number_of_continents_collected :
  total_books / books_per_continent = 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_continents_collected_l67_67247


namespace coefficient_x5_in_expansion_l67_67142

theorem coefficient_x5_in_expansion :
  ∀ (x : ℝ), (∃ c : ℝ, (x + 3)^9 = c * x^5 + _) → c = 10206 :=
by
  sorry

end coefficient_x5_in_expansion_l67_67142


namespace max_marked_nodes_l67_67232

-- Define the problem assumptions
def grid_size : ℕ := 6
def total_vertices : ℕ := (grid_size + 1) * (grid_size + 1)
def corner_vertices : ℕ := 4

-- State the theorem
theorem max_marked_nodes : ∃ (marked_vertices : ℕ),
  marked_vertices = total_vertices - corner_vertices ∧
  ∀ v, -- for all vertices v
    v ∈ marked_vertices → -- if v is a marked vertex
    ∃ (painted_adj : ℕ) (unpainted_adj : ℕ), -- there exists painted and unpainted adjacent cells
    painted_adj = unpainted_adj := -- such that painted and unpainted cells are equal in number
  marked_vertices = 45 := 
sorry

end max_marked_nodes_l67_67232


namespace geometric_sequence_problem_l67_67014

variable {α : Type*} [LinearOrder α] [Field α]

def is_geometric_sequence (a : ℕ → α) :=
  ∀ n : ℕ, a (n + 1) * a (n - 1) = a n ^ 2

theorem geometric_sequence_problem (a : ℕ → α) (r : α) (h1 : a 1 = 1) (h2 : is_geometric_sequence a) (h3 : a 3 * a 5 = 4 * (a 4 - 1)) : 
  a 7 = 4 :=
by
  sorry

end geometric_sequence_problem_l67_67014


namespace polygon_interior_angle_sum_1440_regular_l67_67878

theorem polygon_interior_angle_sum_1440_regular
  {Q : Type} 
  (n : ℕ) 
  (interior_angle : Q → ℝ) 
  (exterior_angle : Q → ℝ)
  (h₁ : ∀ v : Q, interior_angle v = 4 * exterior_angle v) 
  (h₂ : ∑ v in finset.univ, exterior_angle v = 360)
  (vertices : finset Q) 
  (h₃ : ∑ v in vertices, interior_angle v = 4 * ∑ v in vertices, exterior_angle v) :
∑ v in vertices, interior_angle v = 1440 ∧ 
(∀ u v : Q, exterior_angle u = exterior_angle v) :=
sorry

end polygon_interior_angle_sum_1440_regular_l67_67878


namespace P_greater_than_2_l67_67305

-- Definition of normal distribution
noncomputable def normal_dist (mean variance : ℝ) (X : ℝ → ℝ) := sorry

-- Definition of the probability function
noncomputable def P (X : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- Given conditions
variables (σ : ℝ) (X : ℝ → ℝ)
axiom X_normal : normal_dist 0 (σ^2) X
axiom P_minus2_to_0 : P X (-2) 0 = 0.4

-- Theorem statement
theorem P_greater_than_2 : P X 2 ∞ = 0.1 := sorry

end P_greater_than_2_l67_67305


namespace simplify_fraction_l67_67925

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 :=
by
  sorry

end simplify_fraction_l67_67925


namespace exists_int_x_for_a_eq_7_l67_67174

theorem exists_int_x_for_a_eq_7 :
  ∃ x : ℤ, (∏ i in finset.range (a + 1), (1 + 1 / (x + i))) = a - x
  ↔ a = 7 :=
sorry

end exists_int_x_for_a_eq_7_l67_67174


namespace sum_of_squares_powers_l67_67291

theorem sum_of_squares_powers (n : ℕ) (h₁ : n > 1) (k : ℕ) (hk : k > 0) (p : ℕ) [hp : Fact (Nat.prime p)] :
  (∑ i in Finset.range (n + 1), i ^ 2) - 1 = p ^ k ↔ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 7 :=
by
  sorry

end sum_of_squares_powers_l67_67291


namespace probability_non_defective_pens_l67_67850

theorem probability_non_defective_pens 
  (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) 
  (h_total : total_pens = 12) 
  (h_defective : defective_pens = 4) 
  (h_selected : selected_pens = 2) : 
  (8 / 12 : ℚ) * (7 / 11 : ℚ) = 14 / 33 :=
by
  rw [←nat.cast_add_one defective_pens, ←nat.cast_add_one (total_pens - defective_pens)],
  norm_num,
  rw [mul_comm, mul_div_assoc, ←cast_eq_of_rat_eq, ←cast_eq_of_rat_eq],
  field_simp,
  norm_num,
  sorry

end probability_non_defective_pens_l67_67850


namespace multiples_of_10_not_3_or_8_l67_67819

theorem multiples_of_10_not_3_or_8 (p : ℕ → Prop) :
  {n : ℕ | 1 ≤ n ∧ n ≤ 300 ∧ n % 10 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0}.card = 15 :=
by
  sorry

end multiples_of_10_not_3_or_8_l67_67819


namespace shopper_saves_more_l67_67109

-- Definitions and conditions
def cover_price : ℝ := 30
def percent_discount : ℝ := 0.25
def dollar_discount : ℝ := 5
def first_discounted_price : ℝ := cover_price * (1 - percent_discount)
def second_discounted_price : ℝ := first_discounted_price - dollar_discount
def first_dollar_discounted_price : ℝ := cover_price - dollar_discount
def second_percent_discounted_price : ℝ := first_dollar_discounted_price * (1 - percent_discount)

def additional_savings : ℝ := second_percent_discounted_price - second_discounted_price

-- Theorem stating the shopper saves 125 cents more with 25% first
theorem shopper_saves_more : additional_savings = 1.25 := by
  sorry

end shopper_saves_more_l67_67109


namespace cos_2beta_value_l67_67306

theorem cos_2beta_value (α β : ℝ) 
  (h1 : Real.sin (α - β) = 3/5) 
  (h2 : Real.cos (α + β) = -3/5) 
  (h3 : α - β ∈ Set.Ioo (π/2) π) 
  (h4 : α + β ∈ Set.Ioo (π/2) π) : 
  Real.cos (2 * β) = 24/25 := 
sorry

end cos_2beta_value_l67_67306


namespace min_sum_of_distances_l67_67912

-- Define a regular tetrahedron and a point in space
noncomputable def is_center_of_tetrahedron (A B C D P : EuclideanSpace ℝ (Fin 3))
    := ∀ Q : EuclideanSpace ℝ (Fin 3),
          ∑ v in {A, B, C, D}, (EuclideanSpace.dist P v) ≤ ∑ v in {A, B, C, D}, (EuclideanSpace.dist Q v)

theorem min_sum_of_distances (A B C D P : EuclideanSpace ℝ (Fin 3)) 
  (h_tetra : is_regular_tetrahedron A B C D) :
  (∑ v in {A, B, C, D}, EuclideanSpace.dist P v = ∑ v in {A, B, C, D}, EuclideanSpace.dist (centroid A B C D) v)
  ↔ is_center_of_tetrahedron A B C D P := 
begin
  sorry,
end

end min_sum_of_distances_l67_67912


namespace length_of_OC_l67_67417

open Real

-- Definitions for the geometrical entities and conditions
noncomputable def circle_center := (0 : ℝ, 0 : ℝ)  -- Center O at origin
noncomputable def circle_radius := 1

variable (B : ℝ × ℝ) (l : ℝ × ℝ → ℝ × ℝ) (A : ℝ × ℝ) (C : ℝ × ℝ)

-- Assume B is on the circle and l is the tangent at B
axiom hb_circle : B.1^2 + B.2^2 = circle_radius^2
axiom hl_tangent : dot_product (B.1 - circle_center.1, B.2 - circle_center.2) (l B) = 0

-- Define angle conditions and C as the foot of the perpendicular
axiom ha_angle : ∠(A, circle_center, B) = π / 3
axiom hc_foot_perp : perpendicular_line (B, C) (A, circle_center)

-- Proof statement that OC = 1/2
theorem length_of_OC
  (OB : ℝ := distance circle_center B)
  (OC : ℝ := distance circle_center C) : OC = 1 / 2 :=
by
  sorry

end length_of_OC_l67_67417


namespace sin_240_eq_neg_sqrt3_over_2_l67_67648

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67648


namespace h_n_mul_h_2023_eq_l67_67502

variable (h : ℕ → ℝ)
variable (k : ℝ)
variable (n : ℕ)

axiom pos_nat (m : ℕ) : m > 0

axiom h_rule : ∀ m n, pos_nat m → pos_nat n → h (m + n) = h m * h n

axiom h_one : h 1 = k

theorem h_n_mul_h_2023_eq : h n * h 2023 = k ^ (n + 2023) :=
sorry

end h_n_mul_h_2023_eq_l67_67502


namespace fruit_seller_loss_percentage_l67_67210

-- Definitions for the given problem
def selling_price_loss : ℝ := 13
def selling_price_profit : ℝ := 19.93
def profit_percentage : ℝ := 0.15

-- Function to calculate the cost price
def cost_price (sp_profit : ℝ) (profit_perc : ℝ) : ℝ :=
  sp_profit / (1 + profit_perc)

-- Function to calculate the loss percentage
def loss_percentage (cp : ℝ) (sp_loss : ℝ) : ℝ :=
  ((cp - sp_loss) / cp) * 100

-- The main theorem to prove the percentage loss
theorem fruit_seller_loss_percentage : 
  loss_percentage (cost_price selling_price_profit profit_percentage) selling_price_loss = 25 := 
by
  sorry

end fruit_seller_loss_percentage_l67_67210


namespace sin_240_eq_neg_sqrt3_div_2_l67_67630

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67630


namespace incenter_on_tangent_l67_67179

noncomputable def circumcenter_of_triangle (A B C : Point) : Point := sorry
noncomputable def midpoint_of_arc (S : Circle) (A B : Point) (exclude: Point) : Point := sorry
noncomputable def tangent_circle (center : Point) (line : Line) : Circle := sorry
noncomputable def incenter_of_triangle (A B C : Point) : Point := sorry
noncomputable def external_common_tangent (C1 C2 : Circle) : Line := sorry

theorem incenter_on_tangent (A B C : Point) (S : Circle) :
  let A0 := midpoint_of_arc S B C A
  let C0 := midpoint_of_arc S A B C
  let S1 := tangent_circle A0 (line_through B C)
  let S2 := tangent_circle C0 (line_through A B)
  let I := incenter_of_triangle A B C
  (lies_on I (external_common_tangent S1 S2)) :=
begin
  -- Proof goes here
  sorry
end

end incenter_on_tangent_l67_67179


namespace polygon_sides_l67_67841

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
by
  sorry

end polygon_sides_l67_67841


namespace can_determine_counterfeit_coin_l67_67486

/-- 
Given 101 coins where 50 are counterfeit and each counterfeit coin 
differs by 1 gram from the genuine ones, prove that Petya can 
determine if a given coin is counterfeit with a single weighing 
using a balance scale.
-/
theorem can_determine_counterfeit_coin :
  ∃ (coins : Fin 101 → ℤ), 
    (∃ i : Fin 101, (1 ≤ i ∧ i ≤ 50 → coins i = 1) ∧ (51 ≤ i ∧ i ≤ 101 → coins i = 0)) →
    (∃ (b : ℤ), (0 < b → b ∣ 1) ∧ (¬(0 < b → b ∣ 1) → coins 101 = b)) :=
by
  sorry

end can_determine_counterfeit_coin_l67_67486


namespace apples_in_pile_l67_67490

/-- Assuming an initial pile of 8 apples and adding 5 more apples, there should be 13 apples in total. -/
theorem apples_in_pile (initial_apples added_apples : ℕ) (h1 : initial_apples = 8) (h2 : added_apples = 5) :
  initial_apples + added_apples = 13 :=
by
  sorry

end apples_in_pile_l67_67490


namespace smaller_rectangle_area_l67_67566

-- Define the lengths and widths of the rectangles
def bigRectangleLength : ℕ := 40
def bigRectangleWidth : ℕ := 20
def smallRectangleLength : ℕ := bigRectangleLength / 2
def smallRectangleWidth : ℕ := bigRectangleWidth / 2

-- Define the area of the rectangles
def area (length width : ℕ) : ℕ := length * width

-- Prove the area of the smaller rectangle
theorem smaller_rectangle_area : area smallRectangleLength smallRectangleWidth = 200 :=
by
  -- Skip the proof
  sorry

end smaller_rectangle_area_l67_67566


namespace isosceles_trapezoid_angle_relationships_l67_67456

noncomputable theory
open_locale classical

variables {α : Type*} [euclidean_space α] {A B C D : α}
variables (AB BC CD AD : ℝ)
variables (angle_A angle_B angle_C angle_D : ℝ)

structure IsoscelesTrapezoid (A B C D : α) : Prop :=
  (AB_eq_BC : AB = BC)
  (BC_eq_CD : BC = CD)
  (BC_parallel_AD : parallel BC AD)

def angle_relationships (A B C D : α) (angle_A angle_B angle_C angle_D : ℝ) : Prop :=
  angle_ACB = angle_ADB ∧
  angle_BAC = angle_BDC ∧
  angle_CAD = angle_CBD ∧
  angle_CBD = angle_BAC ∧
  angle_CBD = angle_BDC ∧
  angle_BAC = angle_BDC ∧
  angle_BAC = (angle_BAD / 2) ∧
  angle_BDC = (angle_ADC / 2)

theorem isosceles_trapezoid_angle_relationships
  (A B C D : α) (AB BC CD AD : ℝ) (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : IsoscelesTrapezoid A B C D) :
  angle_relationships A B C D angle_A angle_B angle_C angle_D :=
sorry

end isosceles_trapezoid_angle_relationships_l67_67456


namespace smaller_rectangle_area_l67_67569

-- Define the conditions
def large_rectangle_length : ℝ := 40
def large_rectangle_width : ℝ := 20
def smaller_rectangle_length : ℝ := large_rectangle_length / 2
def smaller_rectangle_width : ℝ := large_rectangle_width / 2

-- Define what we want to prove
theorem smaller_rectangle_area : 
  (smaller_rectangle_length * smaller_rectangle_width = 200) :=
by
  sorry

end smaller_rectangle_area_l67_67569


namespace train_cross_signal_time_approx_l67_67536

/-- Given conditions: -/
def TrainLength : ℤ := 300
def PlatformLength : ℤ := 200
def TimeToCrossPlatform : ℤ := 30

/-- Prove the time to cross the signal pole is approximately 18 seconds -/
theorem train_cross_signal_time_approx :
  let speed := (TrainLength + PlatformLength) / TimeToCrossPlatform in
  let timeToCrossSignalPole := TrainLength / speed in
  timeToCrossSignalPole ≈ 18 := by
    sorry

end train_cross_signal_time_approx_l67_67536


namespace max_triangles_formed_by_4_points_l67_67861

theorem max_triangles_formed_by_4_points (points : Finset Point) 
    (h_points_card : points.card = 4) 
    (h_no_three_collinear : ∀ (a b c : Point), a ∈ points → b ∈ points → c ∈ points → ¬ collinear {a, b, c}) : 
    ∃ (triangles : Finset (Finset Point)), triangles.card = 4 ∧ (∀ t ∈ triangles, t.card = 3) :=
by
  sorry

end max_triangles_formed_by_4_points_l67_67861


namespace mutually_exclusive_shots_proof_l67_67216

/-- Definition of a mutually exclusive event to the event "at most one shot is successful". -/
def mutual_exclusive_at_most_one_shot_successful (both_shots_successful at_most_one_shot_successful : Prop) : Prop :=
  (at_most_one_shot_successful ↔ ¬both_shots_successful)

variable (both_shots_successful : Prop)
variable (at_most_one_shot_successful : Prop)

/-- Given two basketball shots, prove that "both shots are successful" is a mutually exclusive event to "at most one shot is successful". -/
theorem mutually_exclusive_shots_proof : mutual_exclusive_at_most_one_shot_successful both_shots_successful at_most_one_shot_successful :=
  sorry

end mutually_exclusive_shots_proof_l67_67216


namespace nonneg_int_repr_l67_67913

theorem nonneg_int_repr (n : ℕ) : ∃ (a b c : ℕ), (0 < a ∧ a < b ∧ b < c) ∧ n = a^2 + b^2 - c^2 :=
sorry

end nonneg_int_repr_l67_67913


namespace same_color_difference_perfect_square_l67_67276

theorem same_color_difference_perfect_square :
  (∃ (f : ℤ → ℕ) (a b : ℤ), f a = f b ∧ a ≠ b ∧ ∃ (k : ℤ), a - b = k * k) :=
sorry

end same_color_difference_perfect_square_l67_67276


namespace find_second_equation_value_l67_67776

theorem find_second_equation_value:
  (∃ x y : ℝ, 2 * x + y = 26 ∧ (x + y) / 3 = 4) →
  (∃ x y : ℝ, 2 * x + y = 26 ∧ x + 2 * y = 10) :=
by
  sorry

end find_second_equation_value_l67_67776


namespace triangle_area_proof_l67_67141

noncomputable def line_1 (x : ℝ) : ℝ := 2 * x + 1
noncomputable def line_2 (x : ℝ) : ℝ := -3 * x + 16

def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * abs ((x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2))

theorem triangle_area_proof :
  let v1 : ℝ × ℝ := (3, 7) in
  let v2 : ℝ × ℝ := (0, 1) in
  let v3 : ℝ × ℝ := (0, 16) in
  triangle_area v1.1 v1.2 v2.1 v2.2 v3.1 v3.2 = 22.5 :=
by
  sorry

end triangle_area_proof_l67_67141


namespace distance_between_lines_l67_67906

noncomputable def point := ℤ

noncomputable def distance (p1 p2 : point) : ℤ := abs (p1 - p2)

noncomputable def A := 0
noncomputable def B := 18
noncomputable def C := 32

noncomputable def d_A := 12
noncomputable def d_B := 15
noncomputable def d_C := 20

theorem distance_between_lines (l m : point) (h1 : distance A B = 18) (h2 : distance B C = 14)
  (h3 : distance A m = d_A) (h4 : distance B m = d_B) (h5 : distance C m = d_C) : distance l m = 12 :=
sorry

end distance_between_lines_l67_67906


namespace shortest_distance_from_parabola_to_line_l67_67327

open Real

noncomputable def parabola_point (M : ℝ × ℝ) : Prop :=
  M.snd^2 = 6 * M.fst

noncomputable def distance_to_line (M : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * M.fst + b * M.snd + c) / sqrt (a^2 + b^2)

theorem shortest_distance_from_parabola_to_line (M : ℝ × ℝ) (h : parabola_point M) :
  distance_to_line M 3 (-4) 12 = 3 :=
by
  sorry

end shortest_distance_from_parabola_to_line_l67_67327


namespace range_abs_sum_l67_67955

theorem range_abs_sum (x : ℝ) : ∃ y, y = |x + 2| + |x + 3| ∧ y ∈ set.Ici 1 :=
by
  sorry

end range_abs_sum_l67_67955


namespace modulus_of_z_l67_67774

def z : ℂ := Complex.I * (2 - Complex.I)
def modulus := Complex.abs z

theorem modulus_of_z : modulus = Real.sqrt 5 :=
by
  sorry

end modulus_of_z_l67_67774


namespace union_M_N_l67_67323

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3x = 0}

theorem union_M_N : M ∪ N = ({0, -3, 4} : Set ℝ) := by
  sorry

end union_M_N_l67_67323


namespace factorize_3a_squared_minus_6a_plus_3_l67_67728

theorem factorize_3a_squared_minus_6a_plus_3 (a : ℝ) : 
  3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 :=
by 
  sorry

end factorize_3a_squared_minus_6a_plus_3_l67_67728


namespace inequality_proof_l67_67024

-- Define the conditions a_i are real numbers and 0 ≤ a_i ≤ π/2.
variables {n : ℕ} (a : Fin n → ℝ)

-- Define the bounds of a_i
def a_bounds (a : Fin n → ℝ) : Prop := ∀ i, 0 ≤ a i ∧ a i ≤ π / 2

-- Define the proof problem
theorem inequality_proof (h : a_bounds a) :
  (1 / n * ∑ i, 1 / (1 + Real.sin (a i))) * (1 + ∏ i, (Real.sin (a i)) ^ (1 / n : ℝ)) ≤ 1 :=
sorry

end inequality_proof_l67_67024


namespace sin_240_deg_l67_67605

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67605


namespace carla_wins_probability_l67_67594

-- Define the probabilities for specific conditions
def prob_derek_rolls_less_than_7 : ℚ := 5 / 6
def prob_emily_rolls_less_than_7 : ℚ := 2 / 3
def prob_combined_rolls_lt_10 : ℚ := 4 / 5

-- Define the total probability based on the given conditions
def total_probability : ℚ :=
  (prob_derek_rolls_less_than_7 * prob_emily_rolls_less_than_7) * prob_combined_rolls_lt_10

-- Main theorem asserting the probability of the event
theorem carla_wins_probability :
  total_probability = 8 / 27 :=
by
  -- Assuming the correct answer provided, we state the theorem
  sorry

end carla_wins_probability_l67_67594


namespace bulbs_arrangement_correct_l67_67119

noncomputable def bulbs_arrangement : ℕ :=
  let blueRedArrangements := Nat.choose 15 8
  let whiteArrangement := Nat.choose 16 12
  blueRedArrangements * whiteArrangement

theorem bulbs_arrangement_correct : bulbs_arrangement = 11711700 := 
by
  let blueRedArrangements := Nat.choose 15 8
  have h1: blueRedArrangements = 6435 := by sorry
  let whiteArrangement := Nat.choose 16 12
  have h2: whiteArrangement = 1820 := by sorry
  rw [bulbs_arrangement, h1, h2]
  norm_num

end bulbs_arrangement_correct_l67_67119


namespace bisect_angle_M_T_K_N_l67_67240
-- Import necessary library

-- Definitions based on conditions
variables {O₁ O₂ T M N A B C D K : Type*}
variables [Points : Set O₁] [Points : Set O₂]
variables (is_tangent : Tangent O₁ O₂ T)

variables (on_circ_O₁_M : OnCircle O₁ M)
variables (on_circ_O₁_N : OnCircle O₁ N)
variables (different_from_T_M : M ≠ T)
variables (different_from_T_N : N ≠ T)

variables (on_chord_O₂_A : OnLine A B)
variables (on_chord_O₂_M : M ∈ A ⊓ B)
variables (on_chord_O₂_C : OnLine C D)
variables (on_chord_O₂_N : N ∈ C ⊓ D)

variables (common_point_K : Exists ∃ K, LiesOnIntersection K (Line A C) (Line B D) (Line M N))

-- Statement to be proved
theorem bisect_angle_M_T_K_N (h : ∀ T M N A B C D K, 
  is_tangent O₁ O₂ T →
  OnCircle O₁ M → 
  OnCircle O₁ N → 
  M ≠ T → 
  N ≠ T → 
  OnLine A B → 
  M ∈ A ⊓ B → 
  OnLine C D → 
  N ∈ C ⊓ D → 
  (∃ K, LiesOnIntersection K (Line AC) (Line BD) (Line MN)) → 
  bisects TK ∠MTN) :
  TK bisects ∠MTN := by sorry

-- This completes the statement of the proof without the proof steps.

end bisect_angle_M_T_K_N_l67_67240


namespace find_number_of_possible_Rs_l67_67061

-- Define the conditions and question
noncomputable def P : ℝ × ℝ := (-5, 0)
noncomputable def Q : ℝ × ℝ := (5, 0)
noncomputable def dist_PQ : ℝ := 10
noncomputable def area_triangle : ℝ := 20
noncomputable def dist_PR : ℝ := 10

-- Define the function to check for number of possible Rs
noncomputable def num_possible_Rs (P Q : ℝ × ℝ) (dist_PQ area_triangle dist_PR : ℝ) : ℕ :=
  have h_R : ℝ := 4
  let possible_Rs := [
    (√84 - 5, 4), (-√84 - 5, 4),
    (√84 - 5, -4), (-√84 - 5, -4)
  ]
  possible_Rs.length

-- Lean theorem statement
theorem find_number_of_possible_Rs :
  num_possible_Rs P Q dist_PQ area_triangle dist_PR = 4 :=
by sorry

end find_number_of_possible_Rs_l67_67061


namespace num_solutions_ffx_eq6_l67_67896

def f (x : ℝ) : ℝ :=
if x < 0 then -x + 4 else 3 * x - 6

theorem num_solutions_ffx_eq6 : ∃ S : set ℝ, (∀ x ∈ S, f (f x) = 6) ∧ finset.card S = 2 := by
sorry

end num_solutions_ffx_eq6_l67_67896


namespace find_number_of_flowers_l67_67255
open Nat

theorem find_number_of_flowers (F : ℕ) (h_candles : choose 4 2 = 6) (h_groupings : 6 * choose F 8 = 54) : F = 9 :=
sorry

end find_number_of_flowers_l67_67255


namespace find_ratio_l67_67872

-- Define the triangle with given side lengths
structure Triangle :=
  (A B C : Point)
  (AB BC CA : ℝ)
  (hAB : AB = 7)
  (hBC : BC = 8)
  (hCA : CA = 9)

-- Define the circumcenter, incenter, and circumcircle of the triangle
structure TriangleProperties :=
  (O I : Point) -- circumcenter and incenter
  (Γ : Circle)  -- circumcircle
  (hO_circumcenter : O.is_circumcenter ABC)
  (hI_incenter : I.is_incenter ABC)
  (hΓ_circumcircle : Γ = Circle.circumcircle ABC)

-- Define midpoint of major arc, intersection point D, and reflection point E
structure SpecialPoints (ABC : Triangle) (props : TriangleProperties ABC) :=
  (M : Point) -- midpoint of major arc BAC
  (D : Point) -- intersection with circumcircle of ∆IMO and Γ
  (E : Point) -- reflection of D over IO
  (hM_major_arc : M.is_midpoint_major_arc props.Γ ABC.A ABC.B ABC.C)
  (hD_intersection : D = Intersection.circumcircle_intersects_circle_B_not_M 
    (TrianglePropertiesΓ.props) props.I props.O M)
  (hE_reflection : E = Reflection.point_over_line D props.I props.O)

-- The main proof structure: Find the integer closest to 1000 * BE / CE
theorem find_ratio (ABC : Triangle) (props : TriangleProperties ABC) (points : SpecialPoints ABC props) :
  ∃ k : ℤ, k = 467 ∧ abs ((1000 * (distance points.E ABC.B) / (distance points.E ABC.C)) - k) < 1 :=
begin
  sorry
end

end find_ratio_l67_67872


namespace probability_diff_colors_l67_67309

theorem probability_diff_colors :
  let total_ways := Nat.choose 6 2 in
  let ways_diff_colors := (Nat.choose 3 1) * (Nat.choose 3 1) in
  (ways_diff_colors : ℚ) / total_ways = 3 / 5 :=
by
  sorry

end probability_diff_colors_l67_67309


namespace sin_240_eq_neg_sqrt3_div_2_l67_67681

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67681


namespace golden_ticket_problem_l67_67254

open Real

/-- The golden ratio -/
noncomputable def φ := (1 + sqrt 5) / 2

/-- Assume the proportions and the resulting area -/
theorem golden_ticket_problem
  (a b : ℝ)
  (h : 0 + b * φ = 
        φ - (5 + sqrt 5) / (8 * φ)) :
  b / a = -4 / 3 :=
  sorry

end golden_ticket_problem_l67_67254


namespace age_difference_l67_67118

variable (A B C : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 13) : (A + B) - (B + C) = 13 := by
  sorry

end age_difference_l67_67118


namespace line_pq_through_centroid_l67_67870

-- Define the conditions
variables {A B C X Y Z U V W P Q A' B' C' : Type}
variable [affine_space ℝ A]
variables [affine_space ℝ B]
variables [affine_space ℝ C]
variables (ABXY BCZW CAUV : set X)
variables (XZU YWV : set ℝ)
variables (P Q : ℝ) (G : ℝ)

-- Conditions
def fixed_triangle (A B C : Type) [triangle A B C] : Prop := true

def similar_isosceles_trapezoids (ABXY BCZW CAUV : set X) 
  (similar : ∀ ABXY BCZW CAUV, isosceles_trapezoid ∧ similar_trapezoid) : Prop :=
  true

def circumcircles_meet_at_two_points (XZU YWV : set ℝ) (P Q : ℝ) : Prop := 
  meet_at_two_points XZU YWV P Q

-- Prove line PQ passes through the centroid G
theorem line_pq_through_centroid (A B C : Type) 
  [triangle A B C] (ABXY BCZW CAUV : set X) 
  (XZU YWV : set ℝ) (P Q G : ℝ) :
  fixed_triangle A B C →
  similar_isosceles_trapezoids ABXY BCZW CAUV →
  circumcircles_meet_at_two_points XZU YWV P Q →
  line_through P Q G :=
sorry

end line_pq_through_centroid_l67_67870


namespace multiples_of_10_not_3_or_8_l67_67818

theorem multiples_of_10_not_3_or_8 (p : ℕ → Prop) :
  {n : ℕ | 1 ≤ n ∧ n ≤ 300 ∧ n % 10 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0}.card = 15 :=
by
  sorry

end multiples_of_10_not_3_or_8_l67_67818


namespace base_seven_for_ABBA_l67_67152

theorem base_seven_for_ABBA (A B : ℕ) (h₀ : A ≠ B) : ∃ (b : ℕ), (b = 7) ∧ let n := 600 in
  n = A * b^3 + B * b^2 + B * b + A ∧ (b^3 ≤ n) ∧ (n < b^4) :=
begin
  sorry
end

end base_seven_for_ABBA_l67_67152


namespace paint_rooms_l67_67404

theorem paint_rooms (R B Y : ℕ) (hR : R = 5) (hB : B = 5) (hY : Y = 5) (h_room_paint : ∀ (r b y : ℕ), r + b + y = 4 ∧ r ≥ 1 ∧ b ≥ 1 ∧ y ≥ 1) : 
  ∃ (rooms : ℕ), rooms = 3 ∧ (R + B + Y) / 4 ≥ rooms :=
by
  split; try { exact 3 };
  simp [hR, hB, hY];
  norm_num;
  sorry

end paint_rooms_l67_67404


namespace sin_240_deg_l67_67657

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67657


namespace pascal_50_5th_element_is_22050_l67_67998

def pascal_fifth_element (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_50_5th_element_is_22050 :
  pascal_fifth_element 50 4 = 22050 :=
by
  -- Calculation steps would go here
  sorry

end pascal_50_5th_element_is_22050_l67_67998


namespace multiples_of_10_not_3_or_8_l67_67817

noncomputable def count_integers_between_1_and_300_that_are_multiples_of_10_not_3_or_8 : Nat :=
  Nat.card (Finset.filter (λ n, n % 10 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0) (Finset.range (300 + 1)))

theorem multiples_of_10_not_3_or_8 :
  count_integers_between_1_and_300_that_are_multiples_of_10_not_3_or_8 = 15 :=
by
  sorry

end multiples_of_10_not_3_or_8_l67_67817


namespace smallest_non_six_digit_palindrome_l67_67740

-- Definition of a four-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.reverse = digits

-- Definition of a six-digit number
def is_six_digit (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000

-- Definition of a non-palindrome
def not_palindrome (n : ℕ) : Prop :=
  ¬ is_palindrome n

-- Find the smallest four-digit palindrome whose product with 103 is not a six-digit palindrome
theorem smallest_non_six_digit_palindrome :
  ∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ is_palindrome n ∧ not_palindrome (103 * n)
  ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ is_palindrome m ∧ not_palindrome (103 * m) → n ≤ m) :=
  sorry

end smallest_non_six_digit_palindrome_l67_67740


namespace perimeter_of_contained_l67_67571

variable (P Q : Set ℝ^3)

def is_contained (P Q : Set ℝ^3) : Prop :=
  ∀ x ∈ P, x ∈ Q

def perimeter (parallelepiped : Set ℝ^3) : ℝ :=
  -- (Assume perimeter is defined accordingly)

theorem perimeter_of_contained (h : is_contained P Q) : perimeter P ≤ perimeter Q := 
sorry

end perimeter_of_contained_l67_67571


namespace martinez_taller_than_chiquita_l67_67436

theorem martinez_taller_than_chiquita :
  ∀ (M C : ℤ), C = 5 ∧ M + C = 12 → (M - C) = 2 :=
begin
  sorry
end

end martinez_taller_than_chiquita_l67_67436


namespace manufacturer_cost_price_correct_l67_67169

noncomputable def manufacturer_cost_price (C : ℝ) : Prop :=
  let wholesaler_selling_price := 1.18 * C in
  let retailer_selling_price := wholesaler_selling_price * 1.20 in
  let retailer_cost_price := 30.09 / 1.25 in
  retailer_selling_price = retailer_cost_price

theorem manufacturer_cost_price_correct (C : ℝ) : manufacturer_cost_price C → C ≈ 17 :=
by
  -- Proof goes here
  sorry

end manufacturer_cost_price_correct_l67_67169


namespace probability_non_defective_pens_l67_67849

theorem probability_non_defective_pens 
  (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) 
  (h_total : total_pens = 12) 
  (h_defective : defective_pens = 4) 
  (h_selected : selected_pens = 2) : 
  (8 / 12 : ℚ) * (7 / 11 : ℚ) = 14 / 33 :=
by
  rw [←nat.cast_add_one defective_pens, ←nat.cast_add_one (total_pens - defective_pens)],
  norm_num,
  rw [mul_comm, mul_div_assoc, ←cast_eq_of_rat_eq, ←cast_eq_of_rat_eq],
  field_simp,
  norm_num,
  sorry

end probability_non_defective_pens_l67_67849


namespace complement_intersection_l67_67357

open Set -- Open namespace for set operations

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3, 4}

-- Define set B
def B : Set ℕ := {3, 4, 5, 6}

-- Define the intersection A ∩ B
def A_inter_B : Set ℕ := A ∩ B

-- Define the complement C_I(S) as I \ S, where S is a subset of I
def complement (S : Set ℕ) : Set ℕ := I \ S

-- Prove that the complement of A ∩ B in I is {1, 2, 5, 6}
theorem complement_intersection : complement A_inter_B = {1, 2, 5, 6} :=
by
  sorry -- Proof to be provided

end complement_intersection_l67_67357


namespace ratio_area_circle_to_rectangle_l67_67421

section hexagon

variables (s : ℝ) -- side length of the regular hexagon

def regular_hexagon (A B C D E F : ℝ × ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ 
  dist D E = s ∧ dist E F = s ∧ dist F A = s ∧
  -- additional regularity conditions can be defined if needed

def length_BD : ℝ := s * Real.sqrt 3

def area_ABDE : ℝ := s^2 * Real.sqrt 3

def inradius_triangle_BDF : ℝ := s / 2

def area_circle_P (r : ℝ) : ℝ := π * r^2

def ratio_areas : ℝ :=
  let radius := inradius_triangle_BDF s in
  let area_circle_P := area_circle_P radius in
  area_circle_P / area_ABDE s

theorem ratio_area_circle_to_rectangle (A B C D E F : ℝ × ℝ) :
  regular_hexagon s A B C D E F →
  ratio_areas s = (π * Real.sqrt 3) / 12 :=
by
  intro h_hex
  dsimp [ratio_areas, inradius_triangle_BDF, area_circle_P, area_ABDE]
  sorry

end hexagon

end ratio_area_circle_to_rectangle_l67_67421


namespace max_ab_bc_cd_da_l67_67952

theorem max_ab_bc_cd_da (a b c d : ℕ) (h : {a, b, c, d} = {1, 3, 5, 7}) :
  ab + bc + cd + da ≤ 64 :=
sorry

end max_ab_bc_cd_da_l67_67952


namespace corners_different_colors_l67_67219

-- Define the problem parameters
variables {l m n : ℕ}
def corner_positions : set (ℕ × ℕ × ℕ) := {(0, 0, 0), (2*l, 0, 0), (0, 2*m, 0), (0, 0, 2*n),
                                           (2*l, 2*m, 0), (2*l, 0, 2*n), (0, 2*m, 2*n), (2*l, 2*m, 2*n)}

-- Define the coloring function
def is_valid_coloring (coloring : (ℕ × ℕ × ℕ) → ℕ) : Prop :=
  ∀ x y z x' y' z', ((x, y, z) ∈ corner_positions ∧ (x', y', z') ∈ corner_positions ∧
  (x = x' ∧ y = y') ∨ (x = x' ∧ z = z') ∨ (y = y' ∧ z = z')) → coloring (x, y, z) ≠ coloring (x', y', z')

-- Prove that all corner cubes are painted in different colors
theorem corners_different_colors (coloring : (ℕ × ℕ × ℕ) → ℕ) (color_range : fin 8) :
  is_valid_coloring coloring → ∀ p1 p2 ∈ corner_positions, p1 ≠ p2 → coloring p1 ≠ coloring p2 :=
sorry

end corners_different_colors_l67_67219


namespace monotonic_decreasing_intervals_l67_67945

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem monotonic_decreasing_intervals : 
  (∀ x : ℝ, (0 < x ∧ x < 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) ∧
  (∀ x : ℝ, (1 < x ∧ x < Real.exp 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) :=
by
  sorry

end monotonic_decreasing_intervals_l67_67945


namespace neg_real_root_condition_l67_67108

theorem neg_real_root_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (0 < a ∧ a ≤ 1) ∨ (a < 0) :=
by
  sorry

end neg_real_root_condition_l67_67108


namespace travel_same_direction_time_l67_67455

variable (A B : Type) [MetricSpace A] (downstream_speed upstream_speed : ℝ)
  (H_A_downstream_speed : downstream_speed = 8)
  (H_A_upstream_speed : upstream_speed = 4)
  (H_B_downstream_speed : downstream_speed = 8)
  (H_B_upstream_speed : upstream_speed = 4)
  (H_equal_travel_time : (∃ x : ℝ, x * downstream_speed + (3 - x) * upstream_speed = 3)
                      ∧ (∃ x : ℝ, x * upstream_speed + (3 - x) * downstream_speed = 3))

theorem travel_same_direction_time (A_α_downstream B_β_upstream A_α_upstream B_β_downstream : ℝ)
  (H_travel_time : (∃ x : ℝ, x = 1) ∧ (A_α_upstream = 3 - A_α_downstream) ∧ (B_β_downstream = 3 - B_β_upstream)) :
  A_α_downstream = 1 → A_α_upstream = 3 - 1 → B_β_downstream = 1 → B_β_upstream = 3 - 1 → ∃ t, t = 1 :=
by
  sorry

end travel_same_direction_time_l67_67455


namespace no_such_triangle_exists_l67_67399

theorem no_such_triangle_exists (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : b = 0.25 * (a + b + c)) :
  ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end no_such_triangle_exists_l67_67399


namespace squirrels_more_than_nuts_l67_67182

theorem squirrels_more_than_nuts 
  (squirrels : ℕ) 
  (nuts : ℕ) 
  (h_squirrels : squirrels = 4) 
  (h_nuts : nuts = 2) 
  : squirrels - nuts = 2 :=
by
  sorry

end squirrels_more_than_nuts_l67_67182


namespace tile_position_l67_67005

theorem tile_position
    (tile_1_1 : (ℕ × ℕ))
    (tiles_1_3 : finset (ℕ × ℕ × ℕ))
    (H_tiling_16 : tiles_1_3.card = 16)
    (H_tiling_cover : ∀ tile ∈ tiles_1_3, covers_3 tile)
    (H_tiling_1_1 : covers_1 tile_1_1)
    (H_no_overlap : no_overlap tiles_1_3 tile_1_1)
    (H_square : ∀ x y, x ∈ [1..7] ∧ y ∈ [1..7]) :
    tile_1_1 ∈ [(1, 1), (4, 4), (7, 7)] ∨ tile_1_1.1 = 1 ∨ tile_1_1.1 = 7 ∨ tile_1_1.2 = 1 ∨ tile_1_1.2 = 7 :=
  sorry

end tile_position_l67_67005


namespace sphere_volume_is_correct_l67_67754
open Real

-- Define the edge length of the cube
def cube_edge_length : ℝ := 2

-- Define the cube's space diagonal
def cube_space_diagonal : ℝ := cube_edge_length * sqrt 3

-- Define the radius of the sphere
def sphere_radius : ℝ := cube_space_diagonal / 2

-- Define the volume of the sphere
def sphere_volume : ℝ := (4 / 3) * π * (sphere_radius ^ 3)

theorem sphere_volume_is_correct :
  sphere_volume = 4 * sqrt 3 * π :=
sorry

end sphere_volume_is_correct_l67_67754


namespace sin_240_eq_neg_sqrt3_div_2_l67_67666

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67666


namespace checkered_triangle_division_l67_67702

-- Define the conditions as assumptions
variable (T : Set ℕ) 
variable (sum_T : Nat) (h_sumT : sum_T = 63)
variable (part1 part2 part3 : Set ℕ)
variable (sum_part1 sum_part2 sum_part3 : Nat)
variable (h_part1 : sum part1 = 21)
variable (h_part2 : sum part2 = 21)
variable (h_part3 : sum part3 = 21)

-- Define the goal as a theorem
theorem checkered_triangle_division : 
  (∃ part1 part2 part3 : Set ℕ, sum part1 = 21 ∧ sum part2 = 21 ∧ sum part3 = 21 ∧ Disjoint part1 (part2 ∪ part3) ∧ Disjoint part2 part3 ∧ T = part1 ∪ part2 ∪ part3) :=
sorry

end checkered_triangle_division_l67_67702


namespace problem1_problem2_problem3_problem4_l67_67250

theorem problem1 : -15 + (-23) - 26 - (-15) = -49 := 
by sorry

theorem problem2 : (- (1 / 2) + (2 / 3) - (1 / 4)) * (-24) = 2 := 
by sorry

theorem problem3 : -24 / (-6) * (- (1 / 4)) = -1 := 
by sorry

theorem problem4 : -1 ^ 2024 - (-2) ^ 3 - 3 ^ 2 + 2 / (2 / 3 * (3 / 2)) = 5 / 2 := 
by sorry

end problem1_problem2_problem3_problem4_l67_67250


namespace calc_power_of_256_l67_67593

theorem calc_power_of_256 : (256 : ℕ) ^ (3 / 4 : ℚ) = 64 :=
by
  -- We'll use the following conditions
  -- 1. 256 = 2^8
  -- 2. The power rule (a^m)^n = a^(m * n)
  have h1 : (256 : ℕ) = 2 ^ 8 := by norm_num
  have h2 : (256 : ℕ) ^ (3 / 4 : ℚ) = (2 ^ 8) ^ (3 / 4 : ℚ) := by rw h1
  have h3 : (2 ^ 8 : ℕ) ^ (3 / 4 : ℚ) = 2 ^ (8 * (3 / 4 : ℚ)) := by simp [pow_mul]
  have h4 : (8 * (3 / 4 : ℚ)) = 6 := by norm_num
  rw [h2, h3, h4]
  norm_num

end calc_power_of_256_l67_67593


namespace min_time_to_same_side_l67_67241

def side_length : ℕ := 50
def speed_A : ℕ := 5
def speed_B : ℕ := 3

def time_to_same_side (side_length speed_A speed_B : ℕ) : ℕ :=
  30

theorem min_time_to_same_side :
  time_to_same_side side_length speed_A speed_B = 30 :=
by
  -- The proof goes here
  sorry

end min_time_to_same_side_l67_67241


namespace partition_triangle_l67_67686

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l67_67686


namespace no_real_solutions_l67_67730

theorem no_real_solutions (x : ℝ) : (x + 2 + complex.i) * (x + 3 + 2 * complex.i) * (x + 4 + complex.i) ∉ ℝ := 
sorry

end no_real_solutions_l67_67730


namespace count_triangles_in_figure_l67_67373

def rectangle_sim (r w l : ℕ) : Prop := 
  (number_of_small_right_triangles r w l = 24) ∧
  (number_of_isosceles_triangles r w l = 6) ∧
  (number_of_half_length_isosceles_triangles r w l = 8) ∧
  (number_of_large_right_triangles r w l = 12) ∧
  (number_of_full_width_isosceles_triangles r w l = 3)

theorem count_triangles_in_figure (r w l : ℕ) (H : rectangle_sim r w l) : 
  total_number_of_triangles r w l = 53 :=
sorry

end count_triangles_in_figure_l67_67373


namespace isosceles_triangle_perimeter_22_l67_67963

theorem isosceles_triangle_perimeter_22 :
  ∃ a b c: ℝ, a = 4 ∧ b = 9 ∧ c = 9 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ (a + b + c = 22) :=
by {
  -- Definitions of the sides
  let a := 4,
  let b := 9,
  let c := 9,
  -- Conditions of the triangle
  have h1 : a + b > c := by norm_num,
  have h2 : a + c > b := by norm_num,
  have h3 : b + c > a := by norm_num,
  -- Perimeter calculation
  use [a, b, c],
  split,
  norm_num,
  split,
  norm_num,
  split,
  norm_num,
  split,
  exact h1,
  split,
  exact h2,
  split,
  exact h3,
  norm_num,
  sorry -- This adds the conclusion and the parts left unproven
}

end isosceles_triangle_perimeter_22_l67_67963


namespace equilateral_triangle_to_square_l67_67364

theorem equilateral_triangle_to_square :
  ∃ (parts : List (Set ℝ × ℝ)), 
    ∀ (triangle : Set (ℝ × ℝ)),
      (is_equilateral_triangle triangle) →
      (is_decomposable_into_parts triangle parts) →
      (can_form_square parts) :=
sorry

end equilateral_triangle_to_square_l67_67364


namespace arithmetic_sequence_length_correct_l67_67822

noncomputable def arithmetic_sequence_length (a d T_n : ℕ) : ℕ :=
  let n := (T_n - a) / d + 1 in n

theorem arithmetic_sequence_length_correct : 
  arithmetic_sequence_length 2 4 2014 = 504 :=
by
  sorry

end arithmetic_sequence_length_correct_l67_67822


namespace part_a_part_b_l67_67177

noncomputable def juca_marbles : ℕ :=
B where
  2 * 3 = 2 ∧
  3 * 4 = 3 ∧
  4 * 5 = 4 ∧
  6 * 7 = 6 ∧
  B < 800

theorem part_a : ∀ (B : ℕ), (B < 800 ∧ B % 3 = 2 ∧ B % 4 = 3 ∧ B % 5 = 4 ∧ B % 7 = 6) → B % 20 = 19 :=
by
  sorry

theorem part_b : ∃ (B : ℕ), (B < 800 ∧ B % 3 = 2 ∧ B % 4 = 3 ∧ B % 5 = 4 ∧ B % 7 = 6) → B = 419 :=
by
  sorry

end part_a_part_b_l67_67177


namespace sin_240_deg_l67_67655

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67655


namespace sum_of_squares_mod_11_l67_67549

def cube_points (n : ℕ) : List (ℕ × ℕ × ℕ) :=
  List.product (List.product (List.range (n+1)) (List.range (n+1))) (List.range (n+1))

def vector_length_squared (x y z : ℕ) : ℕ :=
  x^2 + y^2 + z^2

def sum_of_squares_in_cube (n : ℕ) : ℕ :=
  (cube_points n).foldr (λ (p : ℕ × ℕ × ℕ) (acc : ℕ), acc + (vector_length_squared p.1 p.2.1 p.2.2)) 0

theorem sum_of_squares_mod_11 (n : ℕ) (h : n = 1000) :
  (sum_of_squares_in_cube n) % 11 = 0 :=
by
  sorry

end sum_of_squares_mod_11_l67_67549


namespace inequality_of_f_l67_67796

def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 5

theorem inequality_of_f (x₁ x₂ : ℝ) (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ :=
by
  -- sorry placeholder for the actual proof
  sorry

end inequality_of_f_l67_67796


namespace sin_240_eq_neg_sqrt3_div_2_l67_67643

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67643


namespace sin_240_eq_neg_sqrt3_div_2_l67_67644

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67644


namespace radius_of_circle_l67_67907

theorem radius_of_circle (α : ℝ) (a b : ℝ) (hα1 : α < 90) (hα2 : α > 0) (ha : a > 0) (hb : b > 0) :
  let R := (a + b - 2 * Real.sqrt (a * b) * Real.cos α) / (2 * Real.sin α) in
  R = (a + b - 2 * Real.sqrt (a * b) * Real.cos α) / (2 * Real.sin α) :=
by
  sorry

end radius_of_circle_l67_67907


namespace find_number_l67_67390

def correct_answer (N : ℚ) : ℚ := 5 / 16 * N
def incorrect_answer (N : ℚ) : ℚ := 5 / 6 * N
def condition (N : ℚ) : Prop := incorrect_answer N = correct_answer N + 150

theorem find_number (N : ℚ) (h : condition N) : N = 288 / 5 := by
  sorry

end find_number_l67_67390


namespace area_triangle_difference_is_54_l67_67928

structure Point where
  x : ℝ
  y : ℝ

structure Square where
  A B C D: Point
  side_length: ℝ

def E (B: Point) : Point := { x := B.x - 12, y := 0 }
def F (B C: Point) : Point := { x := B.x, y := (B.y + C.y) / 2 }
def G (C: Point) : Point := { x := C.x - 12, y := C.y }

def area_triangle (P Q R: Point) : ℝ :=
  1 / 2 * abs((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

theorem area_triangle_difference_is_54 :
  ∀ (A B C D : Point),
  let E := E B in
  let F := F B C in
  let G := G C in
  let area_EFG := area_triangle E F G in
  let area_AFD := area_triangle A F D in
  area_EFG - area_AFD = 54 :=
by
  sorry

end area_triangle_difference_is_54_l67_67928


namespace income_on_fourth_day_l67_67197

-- Define the incomes and the average income
def incomes : Fin 5 → ℝ
| ⟨0, _⟩ => 45
| ⟨1, _⟩ => 50
| ⟨2, _⟩ => 60
| ⟨3, _⟩ => x
| ⟨4, _⟩ => 70

def average_income := 58

-- Define the theorem to prove the income on the fourth day
theorem income_on_fourth_day (x : ℝ) (h : (∑ i in (Finset.range 5), incomes i) / 5 = average_income) : x = 65 :=
by
  sorry

end income_on_fourth_day_l67_67197


namespace imaginary_axis_length_l67_67352

theorem imaginary_axis_length (a b : ℝ) (h1 : a > 0)
    (h2 : b > 0)
    (point M : ℝ × ℝ)
    (F1 F2 : ℝ × ℝ)
    (h3 : dist M F1 = 10) 
    (h4 : dist M F2 = 4) 
    (eccentricity : ℝ)
    (h5 : eccentricity = 2)
    (h6 : ∃ x y, x ∈ ℝ ∧ y ∈ ℝ ∧ (x / a)^2 - (y / b)^2 = 1) :
    2 * b = 6 * Real.sqrt 3 :=
by
  sorry -- Proof to be filled in later

end imaginary_axis_length_l67_67352


namespace polynomial_divisible_exists_l67_67026

theorem polynomial_divisible_exists (p : Polynomial ℤ) (a : ℕ → ℤ) (k : ℕ) 
  (h_inc : ∀ i j, i < j → a i < a j) (h_nonzero : ∀ i, i < k → p.eval (a i) ≠ 0) :
  ∃ a_0 : ℤ, ∀ i, i < k → p.eval (a i) ∣ p.eval a_0 := 
by
  sorry

end polynomial_divisible_exists_l67_67026


namespace find_s_l_l67_67946

theorem find_s_l :
  ∃ s l : ℝ, ∀ t : ℝ, 
  (-8 + l * t, s + -6 * t) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p.snd = 3 / 4 * x + 2 ∧ p.fst = x} ∧ 
  (s = -4 ∧ l = -8) :=
by
  sorry

end find_s_l_l67_67946


namespace probability_even_three_digit_number_l67_67745

-- Define the total set of digits
def digits : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the property of being even
def is_even (n : ℕ) := n % 2 = 0

-- Define the probability calculation
theorem probability_even_three_digit_number : 
  (∑ n in (digits.powerset.filter (λ s, s.card = 3)), 
    if is_even (s.to_list.permutations.map (λ l, 100 * l.head + 10 * l.tail.head + l.tail.tail.head))
    then 1 
    else 0) / 
  (∑ n in (digits.powerset.filter (λ s, s.card = 3)), 
    s.to_list.permutations.length) = 2 / 5 :=
sorry

end probability_even_three_digit_number_l67_67745


namespace students_neither_cs_nor_robotics_l67_67233

theorem students_neither_cs_nor_robotics
  (total_students : ℕ)
  (cs_students : ℕ)
  (robotics_students : ℕ)
  (both_cs_and_robotics : ℕ)
  (H1 : total_students = 150)
  (H2 : cs_students = 90)
  (H3 : robotics_students = 70)
  (H4 : both_cs_and_robotics = 20) :
  (total_students - (cs_students + robotics_students - both_cs_and_robotics)) = 10 :=
by
  sorry

end students_neither_cs_nor_robotics_l67_67233


namespace sin_240_eq_neg_sqrt3_div_2_l67_67673

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67673


namespace sin_240_eq_neg_sqrt3_div_2_l67_67676

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67676


namespace angle_EHG_half_BAC_l67_67022

noncomputable theory
open_locale classical

variables {A B C E F G H : Type}
variables [ABC : triangle A B C] [ab_lt_bc : AB < BC]
variables [E_in_AC : point_on_line E A C] [F_in_AB : point_on_line F A B]
variables [BF_eq_BC : distance B F = distance B C] [CE_eq_BC : distance C E = distance B C]
variables [G_intersection_BE_CF : intersection G (line B E) (line C F)]
variables [H_parallel_AC : parallel_line_through H G (line A C)] [HG_eq_AF : distance H G = distance A F]
variables [H_C_opposite_halfplanes_BG : opposite_halfplanes H C (line B G)]

theorem angle_EHG_half_BAC : angle E H G = angle A B C / 2 :=
sorry

end angle_EHG_half_BAC_l67_67022


namespace exists_x_lt_0_l67_67875

variable {X : Type} [OrderedSemiring X] [LinearOrder X] [DenselyOrdered X] [OrderTopology X]
variable (f : X → X)

theorem exists_x_lt_0 (hf1 : ∀ (x y : X), x < y → f x < f y)
    (hf2 : ∀ (x y : X), f ((2 * x * y) / (x + y)) ≥ (f x + f y) / 2) :
    ∃ x : X, f x < 0 := sorry

end exists_x_lt_0_l67_67875


namespace amusement_park_weekly_revenue_l67_67238

def ticket_price : ℕ := 3
def visitors_mon_to_fri_per_day : ℕ := 100
def visitors_saturday : ℕ := 200
def visitors_sunday : ℕ := 300

theorem amusement_park_weekly_revenue : 
  let total_visitors_weekdays := visitors_mon_to_fri_per_day * 5
  let total_visitors_weekend := visitors_saturday + visitors_sunday
  let total_visitors := total_visitors_weekdays + total_visitors_weekend
  let total_revenue := total_visitors * ticket_price
  total_revenue = 3000 := by
  sorry

end amusement_park_weekly_revenue_l67_67238


namespace problem_1_min_value_problem_2_exists_m_l67_67351

open Real

noncomputable def f (x m : ℝ) := log x + m / (2 * x)
noncomputable def g (x m : ℝ) := x - 2 * m

theorem problem_1_min_value (m : ℝ) (x : ℝ) (hx : x > 0) (hm : m = 1) : 
  f x 1 = 1 - log 2 := sorry

theorem problem_2_exists_m (x : ℝ) (hx : x ∈ Icc (1 / exp 1) 1) : 
  ∃ (m : ℝ), (4 / 5 < m < 1) ∧ ∀ x ∈ Icc (1 / exp 1) 1, f x m > g x m + 1  := sorry

end problem_1_min_value_problem_2_exists_m_l67_67351


namespace math_problem_mod_1001_l67_67268

theorem math_problem_mod_1001 :
  (2^6 * 3^10 * 5^12 - 75^4 * (26^2 - 1)^2 + 3^10 - 50^6 + 5^12) % 1001 = 400 := by
  sorry

end math_problem_mod_1001_l67_67268


namespace part1_part2_l67_67766

variables (A B C a b c : Real)

-- Part 1: Prove that A = π / 3 given the initial condition
theorem part1 (h1 : (a - b + c) / c = b / (a + b - c)) : 
  A = π / 3 :=
sorry

-- Part 2: Prove ΔABC is a right triangle given A = π / 3 and b - c = √3 / 3 * a
theorem part2 (h1 : A = π / 3) (h2 : b - c = sqrt 3 / 3 * a) : 
  ∃ B : Real, B = π / 2 :=
sorry

end part1_part2_l67_67766


namespace cans_of_red_paint_l67_67910

theorem cans_of_red_paint (ratio_red_white : ℕ × ℕ)
  (total_cans : ℕ)
  (red_cans_needed : ℕ) 
  (h_ratio : ratio_red_white = (5, 3))
  (h_total : total_cans = 50)
  (h_red_cans : red_cans_needed = 31) :
  red_cans_needed = (5 * total_cans) / (5 + 3) :=
by
  rw [h_ratio, h_total, h_red_cans]
  sorry

end cans_of_red_paint_l67_67910


namespace arccos_cos_eq_l67_67598

theorem arccos_cos_eq :
  Real.arccos (Real.cos 11) = 0.7168 := by
  sorry

end arccos_cos_eq_l67_67598


namespace count_valid_n_leq_1000_num_positive_integers_l67_67736

theorem count_valid_n_leq_1000
    (n : ℕ)
    (h1 : 1 ≤ n)
    (h2 : n ≤ 1000) :
  (∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ = n) → 
  (∃ m : ℕ, n = 6 * m ∨ n = 6 * m + 1 ∨ n = 6 * m + 2 ∨ n = 6 * m + 3) :=
sorry

theorem num_positive_integers (h : 1 ≤ (1000 : ℕ)) :
  (card {n : ℕ | 1 ≤ n ∧ n ≤ 1000 ∧ ∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ = n} : ℕ) = 667 :=
sorry

end count_valid_n_leq_1000_num_positive_integers_l67_67736


namespace find_value_of_expression_l67_67333

section problem

variables {f : ℝ → ℝ} {a b : ℝ}

-- Conditions
def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def piecewise_defined_f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 2 then x + a / x else if 2 ≤ x ∧ x ≤ 3 then b * x - 3 else 0

-- a and b satisfy the simultaneous equations derived from the piecewise function and periodicity
axiom periodicity : periodic f 2
axiom piecewise_f : ∀ x, f x = piecewise_defined_f x
axiom special_condition : f (7/2) = f (-7/2)

-- Equations derived via solutions
axiom eq1 : a - 15 * b = -36
axiom eq2 : a - 3 * b = -4

-- Question: Find the value of 15b - 2a
theorem find_value_of_expression : 15 * b - 2 * a = 32 :=
  sorry

end problem

end find_value_of_expression_l67_67333


namespace least_triangle_perimeter_l67_67959

theorem least_triangle_perimeter :
  ∃ c : ℕ, (c + 24 > 37) ∧ (c + 37 > 24) ∧ (24 + 37 > c) ∧ (24 + 37 + c) = 75 :=
by
  -- Define the known sides
  let a := 24
  let b := 37
  -- The third side must satisfy the triangle inequalities
  exists 14
  dsimp
  split
  · linarith
  split
  · linarith
  split
  · linarith
  rfl

end least_triangle_perimeter_l67_67959


namespace megans_candy_l67_67434

variable (M : ℕ)

theorem megans_candy (h1 : M * 3 + 10 = 25) : M = 5 :=
by sorry

end megans_candy_l67_67434


namespace smaller_rectangle_area_l67_67564

-- Define the lengths and widths of the rectangles
def bigRectangleLength : ℕ := 40
def bigRectangleWidth : ℕ := 20
def smallRectangleLength : ℕ := bigRectangleLength / 2
def smallRectangleWidth : ℕ := bigRectangleWidth / 2

-- Define the area of the rectangles
def area (length width : ℕ) : ℕ := length * width

-- Prove the area of the smaller rectangle
theorem smaller_rectangle_area : area smallRectangleLength smallRectangleWidth = 200 :=
by
  -- Skip the proof
  sorry

end smaller_rectangle_area_l67_67564


namespace count_real_z10_of_z30_eq_1_l67_67976

theorem count_real_z10_of_z30_eq_1 :
  ∃ S : Finset ℂ, S.card = 30 ∧ (∀ z ∈ S, z^30 = 1) ∧ (Finset.filter (λ z : ℂ, z^10 ∈ ℝ) S).card = 10 := 
by {
  sorry -- proof is not required/required to fill in
}

end count_real_z10_of_z30_eq_1_l67_67976


namespace range_of_f_l67_67145

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : set.range (λ x, if x = -2 then 0 else f x) = { y : ℝ | y ≠ 1 } :=
by
  sorry

end range_of_f_l67_67145


namespace reconstruct_diagonals_l67_67204

-- Definitions
def isConvexPolygon (P : Type) :=
  ∀ (x y z : P), convex (triangle x y z)

def noIntersectingDiagonals (P : Type) (D : set (P × P)) :=
  ∀ (d1 d2 ∈ D), d1 ≠ d2 → ¬intersect d1 d2

variables {P : Type} [finite P] -- Assuming P is a finite type representing the vertices
variables (triangles : P → ℕ) -- Number of triangles adjoining each vertex

-- Main statement: we can reconstruct all erased diagonals.
theorem reconstruct_diagonals (convex : isConvexPolygon P) 
                             (no_intersect : noIntersectingDiagonals P (diagonals P triangles)) :
  ∃ (D : set (P × P)), -- There exists a set of diagonals
    (∀ d ∈ D, valid_diagonal d) -- such that each diagonal is valid
    ∧ reconstructible_from_labels P triangles D := begin
  sorry
end

end reconstruct_diagonals_l67_67204


namespace stratified_sampling_A_l67_67001

theorem stratified_sampling_A (A B C total_units : ℕ) (propA : A = 400) (propB : B = 300) (propC : C = 200) (units : total_units = 90) :
  let total_families := A + B + C
  let nA := (A * total_units) / total_families
  nA = 40 :=
by
  -- prove the theorem here
  sorry

end stratified_sampling_A_l67_67001


namespace inscribed_square_and_circle_in_right_triangle_l67_67855

variables (DE EF DF : ℝ) (h s : ℝ)
-- DE represents side DE of the triangle DEF.
-- EF represents side EF of the triangle DEF.
-- DF represents side DF of the triangle DEF.
-- h is the altitude from F to DE.
-- s is the side length of the inscribed square.

noncomputable def side_length_square :=
  sqrt (DE * EF)

noncomputable def radius_circle (s : ℝ) : ℝ :=
  s / 2

theorem inscribed_square_and_circle_in_right_triangle 
  (DE EF DF : ℝ)
  (h s : ℝ)
  (h₁ : DE = 5) 
  (h₂ : EF = 12) 
  (h₃ : DF = 13)
  : s = 780 / 169 ∧ radius_circle s = 390 / 169 :=
sorry

end inscribed_square_and_circle_in_right_triangle_l67_67855


namespace tulips_count_l67_67239

/-- Define the tulips for eyes and smile -/
def red_tulips (eyes_tulips : ℕ) (smile_tulips : ℕ) : ℕ :=
  (2 * eyes_tulips) + smile_tulips

/-- Define the tulips for the yellow background -/
def yellow_tulips (smile_tulips : ℕ) : ℕ :=
  9 * smile_tulips

/-- Calculate the total tulips needed -/
def total_tulips (eyes_tulips : ℕ) (smile_tulips : ℕ) : ℕ :=
  red_tulips eyes_tulips smile_tulips + yellow_tulips smile_tulips

theorem tulips_count : total_tulips 8 18 = 196 :=
by
  /- By computing each portion -/
  show total_tulips 8 18 = 196 from
  have h1 : red_tulips 8 18 = 34 := by rfl
  have h2 : yellow_tulips 18 = 162 := by rfl
  have h3 : red_tulips 8 18 + yellow_tulips 18 = 196 := by
    rw [h1, h2]
    exact rfl
  exact h3

end tulips_count_l67_67239


namespace equivalent_proposition_l67_67163

theorem equivalent_proposition (H : Prop) (P : Prop) (Q : Prop) (hpq : H → P → ¬ Q) : (H → ¬ Q → ¬ P) :=
by
  intro h nq np
  sorry

end equivalent_proposition_l67_67163


namespace sin_240_eq_neg_sqrt3_div_2_l67_67636

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67636


namespace mask_production_total_in_july_l67_67984

theorem mask_production_total_in_july :
  (let A_july := 3000 * 2^4,
       B_july := 5000 * 3^4,
       C_july := 8000 * 4^4 in
    A_july + B_july + C_july = 2501000) := by
  sorry

end mask_production_total_in_july_l67_67984


namespace negation_all_swans_white_l67_67950

variables {α : Type} (swan white : α → Prop)

theorem negation_all_swans_white :
  (¬ ∀ x, swan x → white x) ↔ (∃ x, swan x ∧ ¬ white x) :=
by {
  sorry
}

end negation_all_swans_white_l67_67950


namespace ring_width_eq_disk_radius_l67_67389

theorem ring_width_eq_disk_radius
  (r R1 R2 : ℝ)
  (h1 : R2 = 3 * r)
  (h2 : 7 * π * r^2 = π * (R1^2 - R2^2)) :
  R1 - R2 = r :=
by {
  sorry
}

end ring_width_eq_disk_radius_l67_67389


namespace proof_problem_l67_67477

theorem proof_problem
  (n k m: ℤ)
  (h1 : n = 7 * k + 4)
  (h2 : -20 ≤ n ∧ n ≤ 100)
  (h3 : 7 * m - 3 = 7 * k + 4)
  (h4 : m = k + 1) :
  (∃ (kmin kmax: ℤ), kmin = -3 ∧ kmax = 13 ∧
  (M_card : ∃ (s : Set ℤ), s = {n | ∃ k, -3 ≤ k ∧ k ≤ 13 ∧ n = 7 * k + 4} ∧ s.card = 34)) ∧
  (nmin nmax : ℤ), nmin = -17 ∧ nmax = 96 :=
sorry

end proof_problem_l67_67477


namespace data_set_mode_is_3_l67_67948

open List

def mode (l : List ℕ) : ℕ :=
  let frequencies := l.foldl (λ freq_map x => 
    freq_map.insert x (freq_map.findD x 0 + 1)) 
    (Std.Map.empty ℕ ℕ)
  frequencies.toList.maximumBy (λ a b => a.2 < b.2).1

theorem data_set_mode_is_3 : mode [0, 1, 2, 2, 3, 1, 3, 3] = 3 := 
by
  sorry

end data_set_mode_is_3_l67_67948


namespace maximize_angle_AXB_x_coord_l67_67059

def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (3, 8)

theorem maximize_angle_AXB_x_coord :
  ∃ x : ℝ, (∃ X : ℝ × ℝ, X = (x, 0)) ∧
  maximize_angle_AXB A B (x, 0) ∧ x = 5 * Real.sqrt 2 - 3 :=
sorry

end maximize_angle_AXB_x_coord_l67_67059


namespace disjoint_subsets_equal_sum_l67_67031

-- Given: S is a 68-element subset of the set {1, 2, ..., 2015}
-- Prove: There exist three mutually disjoint non-empty subsets A, B, C of S such that |A| = |B| = |C| and ∑_{a ∈ A} a = ∑_{b ∈ B} b = ∑_{c ∈ C} c

open Finset

noncomputable def S : Finset ℕ := Finset.range 2016 -- Fixed this range to achieve {1, ..., 2015}

theorem disjoint_subsets_equal_sum 
  (S : Finset ℕ) (hS : S.card = 68) (hS_sub : S ⊆ Finset.range 2016) : 
  ∃ (A B C : Finset ℕ), 
  A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ disjoint A B ∧ disjoint B C ∧ disjoint A C ∧ A.card = B.card ∧ B.card = C.card ∧ (A.sum id = B.sum id) ∧ (B.sum id = C.sum id) :=
sorry

end disjoint_subsets_equal_sum_l67_67031


namespace evaluate_expression_at_minus_two_l67_67283

theorem evaluate_expression_at_minus_two :
  ∀ (y : ℤ), y = -2 → y^3 - y^2 + 2*y + 4 = -12 :=
by
  intros y h
  rw h
  have : (-2) ^ 3 - (-2) ^ 2 + 2 * (-2) + 4 = -12 := by
    calc (-2) ^ 3 - (-2) ^ 2 + 2 * (-2) + 4
        = -8 - 4 - 4 + 4 : by norm_num
      ... = -12 : by norm_num
  exact this

end evaluate_expression_at_minus_two_l67_67283


namespace remainder_4x_mod_7_l67_67383

theorem remainder_4x_mod_7 (x : ℤ) (k : ℤ) (h : x = 7 * k + 5) : (4 * x) % 7 = 6 :=
by
  sorry

end remainder_4x_mod_7_l67_67383


namespace sin_240_eq_neg_sqrt3_div_2_l67_67617

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67617


namespace length_of_train_l67_67167

theorem length_of_train
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_mps : ℝ := speed_kmph * (1000 / 3600))
  (total_distance : ℝ := train_speed_mps * crossing_time)
  (train_length : ℝ := total_distance - platform_length)
  (h_speed : speed_kmph = 72)
  (h_platform : platform_length = 260)
  (h_time : crossing_time = 26)
  : train_length = 260 := by
  sorry

end length_of_train_l67_67167


namespace average_wage_per_day_l67_67516

theorem average_wage_per_day :
  let num_male := 20
  let num_female := 15
  let num_child := 5
  let wage_male := 35
  let wage_female := 20
  let wage_child := 8
  let total_wages := (num_male * wage_male) + (num_female * wage_female) + (num_child * wage_child)
  let total_workers := num_male + num_female + num_child
  total_wages / total_workers = 26 := by
  sorry

end average_wage_per_day_l67_67516


namespace find_smallest_n_l67_67684

noncomputable def a : ℕ → ℕ
| 0     := 0  -- conventionally using index starting from 1
| 1     := 3
| (n+2) := (a (n + 1)) ^ 2 - 2 * (n + 1) * (a (n + 1)) + 2

def S (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), a k

theorem find_smallest_n : ∃ n : ℕ, 2 ^ n > S n ∧ n = 6 :=
  sorry

end find_smallest_n_l67_67684


namespace coprime_with_others_l67_67791

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end coprime_with_others_l67_67791


namespace smallest_period_sin_2alpha_l67_67345

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - cos (2 * x)

-- (1) Proving the smallest positive period of the function f(x) is π.
theorem smallest_period (x : ℝ) : (∃ T > 0, ∀ x : ℝ, f(x + T) = f(x)) ∧ (∀ T' > 0, (∀ x : ℝ, f(x + T') = f(x)) → T' >= π) :=
sorry

-- (2) Given α in the interval (0, π/3) and f(α) = 6/5, proving sin(2α) = (3 * sqrt 3 + 4) / 10.
theorem sin_2alpha (α : ℝ) (h₁ : α ∈ Ioo 0 (π / 3)) (h₂ : f(α) = 6 / 5) : sin (2 * α) = (3 * sqrt 3 + 4) / 10 :=
sorry

end smallest_period_sin_2alpha_l67_67345


namespace sum_of_lengths_l67_67918

noncomputable def length_PkQk (k : ℕ) (h : 1 ≤ k ∧ k ≤ 99) : ℝ :=
  (Real.sqrt 74) * (100 - k) / 100

theorem sum_of_lengths (s : Finset ℕ) :
  (∀ k ∈ s, 1 ≤ k ∧ k ≤ 99) →
  (s.sum (λ k, length_PkQk k (by simp [Mem])) * 2 - Real.sqrt 74) = 989 * (Real.sqrt 74) :=
by
  intro h
  sorry

end sum_of_lengths_l67_67918


namespace relationship_between_k_and_c_l67_67015

-- Define the functions and given conditions
def y1 (x : ℝ) (c : ℝ) : ℝ := x^2 + 2*x + c
def y2 (x : ℝ) (k : ℝ) : ℝ := k*x + 2

-- Define the vertex of y1
def vertex_y1 (c : ℝ) : ℝ × ℝ := (-1, c - 1)

-- State the main theorem
theorem relationship_between_k_and_c (k c : ℝ) (hk : k ≠ 0) :
  y2 (vertex_y1 c).1 k = (vertex_y1 c).2 → c + k = 3 :=
by
  sorry

end relationship_between_k_and_c_l67_67015


namespace num_real_z10_l67_67965

theorem num_real_z10 (z : ℂ) (h : z^30 = 1) : (∃ n : ℕ, z = exp (2 * π * I * n / 30)) → ∃ n, z^10 ∈ ℝ :=
by sorry -- Here, we need to show that there are exactly 20 such complex numbers.

end num_real_z10_l67_67965


namespace sin_cos_product_ratio_expression_l67_67325

variable {x : ℝ}

theorem sin_cos_product (h1 : 0 < x) (h2 : x < π) (h3 : sin x + cos x = 7/13) :
  sin x * cos x = -60/169 :=
sorry

theorem ratio_expression (h1 : 0 < x) (h2 : x < π) (h3 : sin x + cos x = 7/13) :
  (5 * sin x + 4 * cos x) / (15 * sin x - 7 * cos x) = 8 / 43 :=
sorry

end sin_cos_product_ratio_expression_l67_67325


namespace hyperbola_foci_on_x_axis_l67_67341

noncomputable def range_of_m (m : ℝ) : Prop :=
  (∃ x y : ℝ, (x^2 / m) + (y^2 / (m - 4)) = 1) ∧ m > 0 ∧ m - 4 < 0

theorem hyperbola_foci_on_x_axis (m : ℝ) :
  range_of_m m → 0 < m ∧ m < 4 :=
begin
  -- Formal proof goes here, but it is omitted as per instructions
  sorry
end

end hyperbola_foci_on_x_axis_l67_67341


namespace round_to_nearest_0_01_l67_67501

theorem round_to_nearest_0_01 (x : ℝ) : x = 3.8963 → Real.round x 2 = 3.90 :=
by
  intro h
  sorry

end round_to_nearest_0_01_l67_67501


namespace cartesian_equation_of_curve_PA_plus_PB_l67_67313

-- Define the problem of converting the polar equation to Cartesian
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the Cartesian equation of the curve
def curve_cartesian (x y : ℝ) : Prop := x = y^2 / 2

-- Define the parametric equation of the line with given α
def line_parametric (α t : ℝ) : ℝ × ℝ := (1/2 + t * Real.cos α, t * Real.sin α)

-- Fixed point P
def P : ℝ × ℝ := (1/2, 0)

-- Define the distance function
def distance (P A : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)

-- Statement for the proof of Question 1
theorem cartesian_equation_of_curve (θ : ℝ) (h : 0 < θ ∧ θ < Real.pi)
  (h_polar : ∀ ρ, ρ = 2 * Real.cos θ / Real.sin θ^2) :
  ∃ ρ x y, polar_to_cartesian ρ θ = (x, y) ∧ curve_cartesian x y := 
sorry

-- Statement for the proof of Question 2
theorem PA_plus_PB (α : ℝ) (hα : α = Real.pi / 3)
  (A B : ℝ × ℝ) (t1 t2 : ℝ) (hA : line_parametric α t1 = A) (hB : line_parametric α t2 = B)
  (h_curve_A : curve_cartesian A.1 A.2) (h_curve_B : curve_cartesian B.1 B.2)
  (h_t1t2 : t1 + t2 = 4 / 3 ∧ t1 * t2 = -4 / 3) :
  distance P A + distance P B = 8 / 3 := 
sorry

end cartesian_equation_of_curve_PA_plus_PB_l67_67313


namespace simplify_expr_l67_67750

theorem simplify_expr (a b : ℝ) (h₁ : a + b = 0) (h₂ : a ≠ b) : (1 - a) + (1 - b) = 2 := by
  sorry

end simplify_expr_l67_67750


namespace mike_rita_combined_bars_l67_67540

theorem mike_rita_combined_bars (total_bars : ℕ) (num_people : ℕ) (mike_rita_bars : ℕ) :
  total_bars = 12 → num_people = 3 → mike_rita_bars = 8 :=
by
  assume h1 : total_bars = 12,
  assume h2 : num_people = 3,
  sorry

end mike_rita_combined_bars_l67_67540


namespace eight_digit_number_divisible_by_101_l67_67062

theorem eight_digit_number_divisible_by_101 
  (N a b : ℕ) 
  (hn1 : N = 10 * a + b)
  (hn2 : 10^7 * b + a < 10^8)
  (hN : 101 ∣ N) 
  (hN_digits : 10^7 ≤ N < 10^8) : 
  101 ∣ (10^7 * b + a) := 
by
  sorry

end eight_digit_number_divisible_by_101_l67_67062


namespace digits_consecutive_digits_divisors_l67_67097

section part_a

variables {a b c d e f : ℕ}

/-- Assuming the digits a, b, c, d, e, and f are distinct and chosen from {1, 2, ..., 9},
prove that at least two of them are consecutive. -/
theorem digits_consecutive (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                                      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                                      c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                                      d ≠ e ∧ d ≠ f ∧
                                      e ≠ f)
  (h_range : a ∈ finset.range 10 ∧ b ∈ finset.range 10 ∧ c ∈ finset.range 10 ∧
             d ∈ finset.range 10 ∧ e ∈ finset.range 10 ∧ f ∈ finset.range 10) :
  ∃ x y ∈ {a, b, c, d, e, f}, |x - y| = 1 :=
sorry

end part_a

section part_b

variables {a b c d e f : ℕ}

/-- Determine the possible values of the positive integer x that divides any 6-digit number
formed by a, b, c, d, e, and f. -/
theorem digits_divisors (h_consecutive : ∃ x y ∈ {a, b, c, d, e, f}, |x - y| = 1)
  (h_range : a ∈ finset.range 10 ∧ b ∈ finset.range 10 ∧ c ∈ finset.range 10 ∧
             d ∈ finset.range 10 ∧ e ∈ finset.range 10 ∧ f ∈ finset.range 10) :
  ∀ x, x > 0 → divides x (100000 * c + 10000 * d + 1000 * e + 100 * f + 10 * a + b) ∧
            divides x (100000 * c + 10000 * d + 1000 * e + 100 * f + 10 * b + a) →
            x ∈ {1, 3, 9} :=
sorry

end part_b

end digits_consecutive_digits_divisors_l67_67097


namespace problem_statement_l67_67249

theorem problem_statement : |1 - real.sqrt (4 / 3)| + (real.sqrt 3 - 1 / 2)^0 = 2 * real.sqrt 3 / 3 := 
by
  sorry

end problem_statement_l67_67249


namespace prism_volume_eq_l67_67933

noncomputable def prism_volume (a α β : ℝ) : ℝ :=
  1 / 2 * a^3 * Real.sin α * Real.sin β * (Real.cos β)^2

theorem prism_volume_eq (a α β : ℝ) :
  (baseIsoscelesTrapezoid α ∧ legEqualsShorterBase ∧ diagonalPrismEquals a ∧ angleDiagonalBasePlane β) →
  prism_volume a α β = (1 / 2 * a^3 * Real.sin α * Real.sin β * (Real.cos β)^2) :=
  by
  sorry

end prism_volume_eq_l67_67933


namespace percentage_increase_first_year_l67_67111

variables (P : ℝ) (pop_initial pop_final : ℝ) (increase_2nd_year : ℝ)

def population_growth (P : ℝ) : ℝ :=
  1200 * (1 + P / 100) * (1 + increase_2nd_year / 100)

theorem percentage_increase_first_year
  (h1 : pop_initial = 1200)
  (h2 : increase_2nd_year = 30)
  (h3 : pop_final = 1950) : 
  P = 25 := 
by
  sorry

end percentage_increase_first_year_l67_67111


namespace range_of_cos_B_l67_67853

-- Definitions based on the conditions
variables {α : Type*} [linear_ordered_field α]

-- Acute Triangle condition
def is_acute_triangle (a b c : α) := 
  a^2 + b^2 > c^2 ∧
  a^2 + c^2 > b^2 ∧
  b^2 + c^2 > a^2

-- Centroid condition
def is_centroid (G : α) (a b c : α) := 
  (2 * G = (a^2 + b^2 + c^2))

-- Proof objective
theorem range_of_cos_B
  {a b c : α}
  (h_acute : is_acute_triangle a b c)
  (h_centroid : is_centroid (AG : α) a b c)
  (h_length_AG_BC : AG = BC) :
  0 < cos B ∧ cos B < sqrt 3 / 3 :=
sorry

end range_of_cos_B_l67_67853


namespace sin_240_eq_neg_sqrt3_over_2_l67_67649

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67649


namespace cosine_value_parallel_vectors_l67_67360

theorem cosine_value_parallel_vectors (α : ℝ) (h1 : ∃ (a : ℝ × ℝ) (b : ℝ × ℝ), a = (Real.cos (Real.pi / 3 + α), 1) ∧ b = (1, 4) ∧ a.1 * b.2 - a.2 * b.1 = 0) : 
  Real.cos (Real.pi / 3 - 2 * α) = 7 / 8 := by
  sorry

end cosine_value_parallel_vectors_l67_67360


namespace tan_five_pi_over_four_l67_67287

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l67_67287


namespace solve_inequality_l67_67262

theorem solve_inequality (x : ℝ) : (x^2 + 5 * x - 14 < 0) ↔ (-7 < x ∧ x < 2) :=
sorry

end solve_inequality_l67_67262


namespace can_identify_counterfeit_coin_l67_67489

theorem can_identify_counterfeit_coin (coins : Fin 101 → ℤ) :
  (∃ n : Fin 101, (∑ i, if i ≠ n then coins i else 0 = 50 ∧
                   ∑ i, if i = n then coins i else 0 = 1)) →
  (∑ i, coins i = 0 ∨ ∑ i, coins i % 2 = 0) :=
sorry

end can_identify_counterfeit_coin_l67_67489


namespace intervals_monotonic_decrease_l67_67723

noncomputable def f (x : ℝ) : ℝ := tan (π / 4 - x)

theorem intervals_monotonic_decrease :
  ∀ (k : ℤ), ∀ x, (k * π - π / 4) < x ∧ x < (k * π + 3 * π / 4) → 
  ∃ k : ℤ, f x = -tan (x - π / 4) ∧ (k * π - π / 4) < x ∧ x < (k * π + 3 * π / 4) :=
sorry

end intervals_monotonic_decrease_l67_67723


namespace ramsey_six_vertices_monochromatic_quadrilateral_l67_67926

theorem ramsey_six_vertices_monochromatic_quadrilateral :
  ∀ (V : Type) (E : V → V → Prop), (∀ x y : V, x ≠ y → E x y ∨ ¬ E x y) →
  ∃ (u v w x : V), u ≠ v ∧ v ≠ w ∧ w ≠ x ∧ x ≠ u ∧ (E u v = E v w ∧ E v w = E w x ∧ E w x = E x u) :=
by sorry

end ramsey_six_vertices_monochromatic_quadrilateral_l67_67926


namespace mango_salsa_count_proof_l67_67572

noncomputable def mango_salsa_dishes 
  (total_dishes : ℕ) 
  (fresh_mango_ratio : ℚ) 
  (removable_mango_dishes : ℕ) 
  (mango_jelly_dishes : ℕ) 
  (edible_dishes : ℕ) : ℕ :=
  let fresh_mango_dishes := (total_dishes * fresh_mango_ratio).toNat in
  let non_removable_mango_dishes := fresh_mango_dishes - removable_mango_dishes in
  let total_non_edible_dishes := total_dishes - edible_dishes in
  let non_mango_salsa_dishes := non_removable_mango_dishes + mango_jelly_dishes in
  total_non_edible_dishes - non_mango_salsa_dishes

theorem mango_salsa_count_proof :
  mango_salsa_dishes 36 (1/6) 2 1 28 = 3 :=
sorry

end mango_salsa_count_proof_l67_67572


namespace range_of_a_l67_67830

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x + 5| > a) → a < 8 := by
  sorry

end range_of_a_l67_67830


namespace can_determine_counterfeit_coin_l67_67487

/-- 
Given 101 coins where 50 are counterfeit and each counterfeit coin 
differs by 1 gram from the genuine ones, prove that Petya can 
determine if a given coin is counterfeit with a single weighing 
using a balance scale.
-/
theorem can_determine_counterfeit_coin :
  ∃ (coins : Fin 101 → ℤ), 
    (∃ i : Fin 101, (1 ≤ i ∧ i ≤ 50 → coins i = 1) ∧ (51 ≤ i ∧ i ≤ 101 → coins i = 0)) →
    (∃ (b : ℤ), (0 < b → b ∣ 1) ∧ (¬(0 < b → b ∣ 1) → coins 101 = b)) :=
by
  sorry

end can_determine_counterfeit_coin_l67_67487


namespace relatively_prime_number_exists_l67_67784

theorem relatively_prime_number_exists :
  -- Given numbers
  (let a := 20172017 in
   let b := 20172018 in
   let c := 20172019 in
   let d := 20172020 in
   let e := 20172021 in
   -- Number c is relatively prime to all other given numbers
   nat.gcd c a = 1 ∧
   nat.gcd c b = 1 ∧
   nat.gcd c d = 1 ∧
   nat.gcd c e = 1) :=
by {
  -- Proof omitted
  sorry
}

end relatively_prime_number_exists_l67_67784


namespace B_more_than_C_l67_67231

variables (A B C : ℕ)
noncomputable def total_subscription : ℕ := 50000
noncomputable def total_profit : ℕ := 35000
noncomputable def A_profit : ℕ := 14700
noncomputable def A_subscr : ℕ := B + 4000

theorem B_more_than_C (B_subscr C_subscr : ℕ) (h1 : A_subscr + B_subscr + C_subscr = total_subscription)
    (h2 : 14700 * 50000 = 35000 * A_subscr) :
    B_subscr - C_subscr = 5000 :=
sorry

end B_more_than_C_l67_67231


namespace ball_arrangement_count_l67_67125

theorem ball_arrangement_count : 
  let red_balls := 6
  let green_balls := 3
  let selected_balls := 4
  (number_of_arrangements red_balls green_balls selected_balls) = 15 :=
by
  sorry

def number_of_arrangements (red_balls : ℕ) (green_balls : ℕ) (selected_balls : ℕ) : ℕ :=
  let choose := λ n k : ℕ, nat.choose n k 
  let case1 := choose (red_balls) (selected_balls)  -- 4 Red Balls - 1 way
  let case2 := choose (red_balls) 3 * choose (green_balls) 1  -- 3 Red Balls and 1 Green Ball
  let case3 := choose (red_balls) 2 * choose (green_balls) 2  -- 2 Red Balls and 2 Green Balls
  let case4 := choose (red_balls) 1 * choose (green_balls) 3  -- 1 Red Ball and 3 Green Balls
  case1 + case2 + case3 + case4

end ball_arrangement_count_l67_67125


namespace find_AC_l67_67917

-- Define the geometric context
variables {A B C D : Type} -- Representing the points
variable [fact (A ≠ B)]     -- Ensuring that each point is distinct
variable [fact (B ≠ C)]
variable [fact (C ≠ D)]
variable [fact (D ≠ A)]
variable (circleABCD : is_inscribed A B C D)

-- Given angles and sides
def angle_BAC := 50
def angle_ADB := 55
def side_AD := 5
def side_BC := 7

-- Prove that AC is approximately 6.4
theorem find_AC :
  ∃ AC : ℝ, AC ≈ 6.4 :=
by
  sorry

end find_AC_l67_67917


namespace probability_of_average_three_is_one_fourth_l67_67746

/-- The set of numbers is {1, 2, 3, 4} -/
def set_of_numbers : Finset ℕ := {1, 2, 3, 4}

/-- The property we are interested in: the average of three numbers being exactly 3 -/
def has_average_three (s : Finset ℕ) : Prop :=
  s.card = 3 ∧ (s.sum / 3) = 3

/-- The number of combinations of selecting 3 different numbers from a set of 4 -/
def total_combinations : ℕ := set_of_numbers.card.choose 3

/-- The number of favorable combinations -/
def favorable_combinations : ℕ :=
  if has_average_three {2, 3, 4} then 1 else 0

/-- Probability calculation -/
def probability : ℚ := favorable_combinations / total_combinations

/-- The proof problem statement -/
theorem probability_of_average_three_is_one_fourth :
  probability = 1 / 4 :=
by
  sorry

end probability_of_average_three_is_one_fourth_l67_67746


namespace correct_propositions_l67_67758

variables (m n : Line) (α β : Plane)

-- Assume the given conditions
axiom perp_m_alpha : Perp m α
axiom perp_n_beta : Perp n β
axiom perp_m_n : Perp m n
axiom parallel_m_alpha : Parallel m α
axiom parallel_n_beta : Parallel n β
axiom parallel_m_n : Parallel m n

-- Define the propositions
def prop1 := Perp m α ∧ Perp n β ∧ Perp m n → Perp α β
def prop4 := Perp m α ∧ Parallel n β ∧ Parallel m n → Perp α β

-- Prove propositions 1 and 4 are correct
theorem correct_propositions : prop1 ∧ prop4 :=
by {
    -- Proposition 1
    apply and.intro,
    {
        intros h1 h2 h3,
        exact Perp α β, -- Prop1 conclusion
        sorry
    },
    -- Proposition 4
    {
        intros h4 h5 h6,
        exact Perp α β, -- Prop4 conclusion
        sorry
    }
}

end correct_propositions_l67_67758


namespace derivative_of_f_l67_67346

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * real.log x + x^2

-- Define the statement that we need to prove
theorem derivative_of_f :
  ∀ x : ℝ, 0 < x → deriv f x = (3 / x) + 2 * x :=
by
  intro x hx
  sorry

end derivative_of_f_l67_67346


namespace area_BCD_l67_67263

-- Coordinates of points A, B, and C
def A := (1 : ℝ, 3 : ℝ)
def B := (4 : ℝ, 6 : ℝ)
def C := (2 : ℝ, 8 : ℝ)

-- Midpoint D of segment AB
def D := (5 / 2, 9 / 2)

-- Function to compute the area of triangle given points
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

-- Theorem statement: The area of triangle BCD is 3
theorem area_BCD : triangle_area B C D = 3 := 
by
  sorry

end area_BCD_l67_67263


namespace range_of_4a_minus_2b_l67_67839

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b)
  (h2 : a - b ≤ 2)
  (h3 : 2 ≤ a + b)
  (h4 : a + b ≤ 4) : 
  5 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 10 :=
by
  sorry

end range_of_4a_minus_2b_l67_67839


namespace ratio_girls_to_boys_l67_67983

theorem ratio_girls_to_boys (g b : ℕ) (h1 : g = b + 4) (h2 : g + b = 28) :
  g / gcd g b = 4 ∧ b / gcd g b = 3 :=
by
  sorry

end ratio_girls_to_boys_l67_67983


namespace triangle_area_example_l67_67503

def point := (ℝ × ℝ)

def area_triangle (A B C : point) : ℝ :=
1 / 2 * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

theorem triangle_area_example : area_triangle (3, 0) (6, 8) (3, 9) = 13.5 := 
by
sory

end triangle_area_example_l67_67503


namespace sin_240_eq_neg_sqrt3_div_2_l67_67642

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67642


namespace can_draw_continuously_without_retracing_l67_67535

-- Define the vertices and edges of the graph
structure Graph where
  vertices : Type
  edges : vertices → vertices → Prop
  symmetric : Symmetric edges
  connected : ∀ v1 v2 : vertices, ∃ p : List vertices, p.Head = v1 ∧ p.Last = v2 ∧ List.Pairwise edges p

-- Define an example graph corresponding to the figure
structure ExampleGraph where
  A B C D E : Graph.vertices
  edges : Graph.edges

-- Specify degree calculation for vertices
def degree (g : Graph) (v : g.vertices) : Nat :=
  Finset.card {e | g.edges v e}

-- Define the properties for an Eulerian path
def eulerian_path_exists (g : Graph) : Prop :=
  let odd_degree_vertices := {v : g.vertices | degree g v % 2 = 1}
  Finset.card odd_degree_vertices = 2 ∨ Finset.card odd_degree_vertices = 0

-- Given a specific graph instance
def example_graph : Graph :=
{ vertices := ExampleGraph,
  edges := ExampleGraph.edges,
  symmetric := by sorry,
  connected := by sorry }

-- Statement of the proof problem
theorem can_draw_continuously_without_retracing (g : Graph) :
  eulerian_path_exists g :=
by sorry

end can_draw_continuously_without_retracing_l67_67535


namespace tv_purchase_price_correct_l67_67574

theorem tv_purchase_price_correct (x : ℝ) (h : (1.4 * x * 0.8 - x) = 270) : x = 2250 :=
by
  sorry

end tv_purchase_price_correct_l67_67574


namespace regular_polygon_interior_angle_ratio_l67_67114

theorem regular_polygon_interior_angle_ratio (r k : ℕ) (h1 : 180 - 360 / r = (5 : ℚ) / (3 : ℚ) * (180 - 360 / k)) (h2 : r = 2 * k) :
  r = 8 ∧ k = 4 :=
sorry

end regular_polygon_interior_angle_ratio_l67_67114


namespace find_f_cos_l67_67753

variable {x : ℝ} {n : ℤ}

def f (x : ℝ) : ℝ := sin ((4 * n + 1) * x)

theorem find_f_cos :
  f (cos x) = cos ((4 * n + 1) * x) :=
sorry

end find_f_cos_l67_67753


namespace max_diagonals_no_common_vertex_l67_67193

theorem max_diagonals_no_common_vertex (grid_size : ℕ) (cells : ℕ) (vertex_count : ℕ) (odd_indices : ℕ) 
  (diagonals_per_cell : ℕ) : grid_size = 15 → cells = 15 * 15 → vertex_count = (grid_size + 1) * (grid_size + 1) → 
  odd_indices = (grid_size + 1) / 2 → diagonals_per_cell = 2 →
  ∃ max_diagonals : ℕ, max_diagonals = 128 :=
by
  intros h1 h2 h3 h4 h5
  let max_diagonals := 64 * 2
  exists_claim max_diagonals
  sorry

end max_diagonals_no_common_vertex_l67_67193


namespace independent_set_probability_bound_l67_67299

open Nat

variables {n k : ℕ} [fact (2 ≤ k)] [fact (k ≤ n)] (p : ℝ)
noncomputable def q := 1 - p

-- Mathematical equivalent proof statement
theorem independent_set_probability_bound :
  ∀ (G : Type) [fintype G] [infinite G] [probability_space G],
  P [α (G) ≥ k] ≤ (binom n k) * (q ^ (binom k 2)) := by
  sorry

end independent_set_probability_bound_l67_67299


namespace even_integers_between_fractions_l67_67367

theorem even_integers_between_fractions:
  let a := 21 / 5
  let b := 43 / 3
  (5 : ℝ) ≤ a ∧ a ≤ (14 : ℝ) ∧ ∀ x : ℝ, (5 ≤ x ∧ x ≤ 14) → x ∈ {6, 8, 10, 12, 14} :=
  sorry

end even_integers_between_fractions_l67_67367


namespace total_earnings_correct_l67_67235

-- Define the conditions as initial parameters

def ticket_price : ℕ := 3
def weekday_visitors_per_day : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300

def total_weekday_visitors : ℕ := 5 * weekday_visitors_per_day
def total_weekend_visitors : ℕ := saturday_visitors + sunday_visitors
def total_visitors : ℕ := total_weekday_visitors + total_weekend_visitors

def total_earnings := total_visitors * ticket_price

-- Prove that the total earnings of the amusement park in a week is $3000
theorem total_earnings_correct : total_earnings = 3000 :=
by
  sorry

end total_earnings_correct_l67_67235


namespace infinite_series_sum_l67_67256

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * n - 1) / 3 ^ (n + 1)) = 2 :=
by
  sorry

end infinite_series_sum_l67_67256


namespace base_conversion_min_sum_l67_67475

theorem base_conversion_min_sum (a b : ℕ) (h1 : 3 * a + 6 = 6 * b + 3) (h2 : 6 < a) (h3 : 6 < b) : a + b = 20 :=
sorry

end base_conversion_min_sum_l67_67475


namespace possible_to_form_four_groups_l67_67078

def place_matchsticks : Prop :=
  ∃ (configuration : matrix (fin 4) (fin 4) bool), 
  (∀ i j, 
    (i < 4 ∧ j < 4 → configuration i j = tt) ∧ 
    (∃ m n, 
      (0 <= m ∧ m < 4 ∧ 0 <= n ∧ n < 4) ∧ 
      (count_matchsticks configuration = 11) ∧ 
      (surrounds_four_groups configuration)))

noncomputable def count_matchsticks (configuration : matrix (fin 4) (fin 4) bool) : ℕ := sorry

noncomputable def surrounds_four_groups (configuration : matrix (fin 4) (fin 4) bool) : Prop := sorry

theorem possible_to_form_four_groups : place_matchsticks :=
sorry

end possible_to_form_four_groups_l67_67078


namespace twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l67_67132

theorem twenty_five_percent_less_than_80_is_twenty_five_percent_more_of (n : ℝ) (h : 1.25 * n = 80 - 0.25 * 80) : n = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l67_67132


namespace sin_240_eq_neg_sqrt3_div_2_l67_67637

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67637


namespace C_will_answer_yes_l67_67127

-- Define the possible types of islanders
inductive Type
| knight
| liar

open Type

-- Define the islanders
constant A : Type
constant B : Type
constant C : Type

-- A makes a statement: "B and C are of the same type."
constant A_statement : (B = knight ↔ C = knight) ∨ (B = liar ↔ C = liar)

-- Define the function that represents C's answer
def C_answer (A B C : Type) : String :=
if A = knight then
  if ((C = knight ∧ B = knight) ∨ (C = liar ∧ B = liar)) then "Yes" else "No"
else
  if ((C = knight ∧ B = liar) ∨ (C = liar ∧ B = knight)) then "Yes" else "No"

-- The theorem stating that C will answer "Yes"
theorem C_will_answer_yes : C_answer A B C = "Yes" :=
sorry

end C_will_answer_yes_l67_67127


namespace prove_y_l67_67272

theorem prove_y (x y : ℝ) (h1 : 3 * x^2 - 4 * x + 7 * y + 3 = 0) (h2 : 3 * x - 5 * y + 6 = 0) :
  25 * y^2 - 39 * y + 69 = 0 := sorry

end prove_y_l67_67272


namespace checkered_triangle_division_l67_67698

-- Define the conditions as assumptions
variable (T : Set ℕ) 
variable (sum_T : Nat) (h_sumT : sum_T = 63)
variable (part1 part2 part3 : Set ℕ)
variable (sum_part1 sum_part2 sum_part3 : Nat)
variable (h_part1 : sum part1 = 21)
variable (h_part2 : sum part2 = 21)
variable (h_part3 : sum part3 = 21)

-- Define the goal as a theorem
theorem checkered_triangle_division : 
  (∃ part1 part2 part3 : Set ℕ, sum part1 = 21 ∧ sum part2 = 21 ∧ sum part3 = 21 ∧ Disjoint part1 (part2 ∪ part3) ∧ Disjoint part2 part3 ∧ T = part1 ∪ part2 ∪ part3) :=
sorry

end checkered_triangle_division_l67_67698


namespace range_of_a_l67_67279

theorem range_of_a 
  (a : ℝ):
  (∀ x : ℝ, |x + 2| + |x - 1| > a^2 - 2 * a) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l67_67279


namespace car_total_distance_l67_67154

-- Define the arithmetic sequence (s_n) where a = 40 and d = -10
def car_travel (n : ℕ) : ℕ := if n > 0 then 40 - 10 * (n - 1) else 0

-- Define the sum of the first 'k' terms of the arithmetic sequence
noncomputable def sum_car_travel (k : ℕ) : ℕ :=
  ∑ i in Finset.range k, car_travel (i + 1)

-- Main theorem statement
theorem car_total_distance : sum_car_travel 4 = 100 :=
by
  sorry

end car_total_distance_l67_67154


namespace arithmetic_sequence_problem_l67_67957

theorem arithmetic_sequence_problem :
  ∃ (f : ℕ → ℤ) (a_n b_n S_n : ℕ → ℤ),
    (∀ (x : ℕ), f(x) = x^2 - 4*x + 2) ∧
    (∀ (x : ℕ), a_1 = f(x + 1) ∧ a_2 = 0 ∧ a_3 = f(x - 1)) ∧
    ∃ x, (x = 1 ∨ x = 3) → 
      (x = 1 → a_n = 2*n - 4) ∧ (x = 3 → a_n = 4 - 2*n) ∧
      (a_n = 2*n - 4 → 
        (∀ n, b_n = a_{n+1} + a_{n+2} + a_{n+3} + a_{n+4}) ∧
        (b_n = 8*n + 4) ∧
        (S_n = ∑ k in finset.range n, 1 / (b_k * b_{k+1}) = n / (48 * (2*n+3))) 
      ) :=
sorry

end arithmetic_sequence_problem_l67_67957


namespace line_through_points_l67_67211

theorem line_through_points (x y : ℝ) :
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 3) ↔ 2 * x + y - 3 = 0 :=
by
  intros hx hy
  sorry

end line_through_points_l67_67211


namespace number_of_tangent_segments_l67_67981

def mutually_external_circles (n : ℕ) : Prop :=
  2017 = n ∧
  ∀ i j, i ≠ j → (¬tangent i j) ∧ (∃ c : set ℕ, ∀ (k ∈ c), k ≠ i ∧ k ≠ j → ¬three_common_tangent i j k)

def tangent_segments (s : ℕ) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → 
  ∀ t1 t2, ¬intersect t1 t2 → 
  s = 3 * (2017 - 1)

theorem number_of_tangent_segments :
  ∃ s, tangent_segments s ∧ s = 6048 :=
by {
  sorry
}

end number_of_tangent_segments_l67_67981


namespace sin_240_deg_l67_67601

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67601


namespace right_triangle_ratio_l67_67292

theorem right_triangle_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∃ (x y : ℝ), 5 * (x * y) = x^2 + y^2 ∧ 5 * (a^2 + b^2) = (x + y)^2 ∧ 
    ((x - y)^2 < x^2 + y^2 ∧ x^2 + y^2 < (x + y)^2)):
  (1/2 < a / b) ∧ (a / b < 2) := by
  sorry

end right_triangle_ratio_l67_67292


namespace square_side_length_leq_half_l67_67083

theorem square_side_length_leq_half
    (l : ℝ)
    (h_square_inside_unit : l ≤ 1)
    (h_no_center_contain : ∀ (x y : ℝ), x^2 + y^2 > (l/2)^2 → (0.5 ≤ x ∨ 0.5 ≤ y)) :
    l ≤ 0.5 := 
sorry

end square_side_length_leq_half_l67_67083


namespace smallest_a_l67_67949

theorem smallest_a (a : ℕ) (hdiv : 21 ∣ a) (hdivcount : a.count_divisors = 105) : a = 254016 :=
sorry

end smallest_a_l67_67949


namespace minimum_x2y3z_l67_67903

theorem minimum_x2y3z (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : x^3 + y^3 + z^3 - 3 * x * y * z = 607) : 
  x + 2 * y + 3 * z ≥ 1215 :=
sorry

end minimum_x2y3z_l67_67903


namespace path_traveled_by_point_A_in_rectangle_l67_67451

noncomputable def total_distance_traveled_by_point_A (AB BC : ℝ) : ℝ :=
  (4 + real.sqrt (AB^2 + BC^2)) * real.pi

theorem path_traveled_by_point_A_in_rectangle :
  let AB := 3
  let BC := 8
  total_distance_traveled_by_point_A AB BC = (4 + real.sqrt (3^2 + 8^2)) * real.pi :=
by 
  sorry

end path_traveled_by_point_A_in_rectangle_l67_67451


namespace arithmetic_sequence_n_value_l67_67858

theorem arithmetic_sequence_n_value
  (a : ℕ → ℚ)
  (h1 : a 1 = 1 / 3)
  (h2 : a 2 + a 5 = 4)
  (h3 : a n = 33)
  : n = 50 :=
sorry

end arithmetic_sequence_n_value_l67_67858


namespace seokgi_money_l67_67919

open Classical

variable (S Y : ℕ)

theorem seokgi_money (h1 : ∃ S, S + 2000 < S + Y + 2000)
                     (h2 : ∃ Y, Y + 1500 < S + Y + 1500)
                     (h3 : 3500 + (S + Y + 2000) = (S + Y) + 3500)
                     (boat_price1: ∀ S, S + 2000 = S + 2000)
                     (boat_price2: ∀ Y, Y + 1500 = Y + 1500) :
  S = 5000 :=
by sorry

end seokgi_money_l67_67919


namespace not_divisible_l67_67251

theorem not_divisible (n : ℕ) : ¬ ((4^n - 1) ∣ (5^n - 1)) :=
by
  sorry

end not_divisible_l67_67251


namespace angle_ABC_is_30_degrees_l67_67375

theorem angle_ABC_is_30_degrees (angle_CBD angle_ABD : ℝ) (sum_angles_at_B : ℝ) 
  (h1 : angle_CBD = 90) (h2 : angle_ABD = 60) 
  (h3 : angle_CBD + angle_ABD + (sum_angles_at_B - (angle_CBD + angle_ABD)) = 180) : 
  (sum_angles_at_B - (angle_CBD + angle_ABD)) = 30 :=
by simp [h1, h2] at h3; exact h3

end angle_ABC_is_30_degrees_l67_67375


namespace trapezoidal_field_perimeter_l67_67226

-- Definitions derived from the conditions
def length_of_longer_parallel_side : ℕ := 15
def length_of_shorter_parallel_side : ℕ := 9
def total_perimeter_of_rectangle : ℕ := 52

-- Correct Answer
def correct_perimeter_of_trapezoidal_field : ℕ := 46

-- Theorem statement
theorem trapezoidal_field_perimeter 
  (a b w : ℕ)
  (h1 : a = length_of_longer_parallel_side)
  (h2 : b = length_of_shorter_parallel_side)
  (h3 : 2 * (a + w) = total_perimeter_of_rectangle)
  (h4 : w = 11) -- from the solution calculation
  : a + b + 2 * w = correct_perimeter_of_trapezoidal_field :=
by
  sorry

end trapezoidal_field_perimeter_l67_67226


namespace sequence_relation_sum_of_sequence_T_l67_67315

-- Condition definitions
def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 1 / 2 else 1 / 2 ^ n

def sequence_S (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i, sequence_a (i + 1))

theorem sequence_relation : ∀ (n : ℕ), n ≥ 2 → 
  2 * sequence_S n = sequence_S (n - 1) + 1 :=
sorry

noncomputable def sequence_b (n : ℕ) : ℚ :=
  real.log (sequence_a n) / real.log (1 / 2)

noncomputable def T (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i, 1 / (sequence_b i * sequence_b (i + 1)))

-- Proof goal
theorem sum_of_sequence_T (n : ℕ) : 
  T n = n / (n + 1) :=
sorry

end sequence_relation_sum_of_sequence_T_l67_67315


namespace third_consecutive_even_l67_67482

theorem third_consecutive_even {a b c d : ℕ} (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h_sum : a + b + c + d = 52) : c = 14 :=
by
  sorry

end third_consecutive_even_l67_67482


namespace partition_triangle_l67_67689

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l67_67689


namespace sin_240_eq_neg_sqrt3_over_2_l67_67653

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67653


namespace sum_is_ten_l67_67842

variable (x y : ℝ) (S : ℝ)

-- Conditions
def condition1 : Prop := x + y = S
def condition2 : Prop := x = 25 / y
def condition3 : Prop := x^2 + y^2 = 50

-- Theorem
theorem sum_is_ten (h1 : condition1 x y S) (h2 : condition2 x y) (h3 : condition3 x y) : S = 10 :=
sorry

end sum_is_ten_l67_67842


namespace probability_of_at_least_65_cents_l67_67463

-- Define the possible coin states: heads or tails
inductive Coin
| heads
| tails

-- Define the values of each coin
def coin_value : Coin → ℕ 
| Coin.heads   := 1
| Coin.tails   := 0

-- Specific value of each coin in cents:
def penny_value : Coin → ℕ
| Coin.heads   := 1
| Coin.tails   := 0

def nickel_value : Coin → ℕ
| Coin.heads   := 5
| Coin.tails   := 0

def dime_value : Coin → ℕ
| Coin.heads   := 10
| Coin.tails   := 0

def half_dollar_value : Coin → ℕ
| Coin.heads   := 50
| Coin.tails   := 0

-- Define the probability calculation
def probability_at_least_65_cents (penny nickel dime half_dollar : Coin) : ℚ :=
  let total_heads_value := penny_value penny + nickel_value nickel + dime_value dime + half_dollar_value half_dollar in
  if total_heads_value >= 65 then 1 else 0

def num_success_cases : ℚ := 5
def total_cases : ℚ := 16

theorem probability_of_at_least_65_cents :
  let probability := num_success_cases / total_cases in
  probability = (5 : ℚ) / 16 :=
by
  sorry

end probability_of_at_least_65_cents_l67_67463


namespace arithmetic_sequence_length_correct_l67_67821

noncomputable def arithmetic_sequence_length (a d T_n : ℕ) : ℕ :=
  let n := (T_n - a) / d + 1 in n

theorem arithmetic_sequence_length_correct : 
  arithmetic_sequence_length 2 4 2014 = 504 :=
by
  sorry

end arithmetic_sequence_length_correct_l67_67821


namespace seq_an_value_l67_67755

theorem seq_an_value (a : ℕ → ℝ) (h : ∀ n : ℕ, 0 < n → (∑ i in range n, (2^(i-1) * a i.succ)) = (n^2 + 1) / 3) :
  ∀ n : ℕ, a n.succ = if n = 0 then 2 / 3 else (2*n + 1) / (3 * 2^n) := 
by
  sorry

end seq_an_value_l67_67755


namespace real_solutions_to_equation_l67_67731

theorem real_solutions_to_equation :
  {x : ℝ | x = 1 + real.sqrt 2 ∨ x = 1 - real.sqrt 2} ⊆
  {x : ℝ | x^4 + (2 - x)^4 = 34} :=
by
  sorry

end real_solutions_to_equation_l67_67731


namespace math_problem_l67_67187

-- Define the conditions
def power_3_5_plus_9720 : ℕ := 3^5 + 9720
def sqrt_289_minus_div : ℝ := (Real.sqrt 289) - (845 / 169.1)

-- State the proof problem
theorem math_problem :
  (power_3_5_plus_9720 * sqrt_289_minus_div = 119556) :=
by
  sorry

end math_problem_l67_67187


namespace odd_with_period_pi_l67_67470

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + π / 4) ^ 2 + Real.cos (x - π / 4) ^ 2 - 1

theorem odd_with_period_pi : 
  Function.Odd f ∧ ∀ x, f (x + π) = f x :=
sorry

end odd_with_period_pi_l67_67470


namespace divide_triangle_l67_67713

/-- 
  Given a triangle with the total sum of numbers as 63,
  we want to prove that it can be divided into three parts
  where each part's sum is 21.
-/
theorem divide_triangle (total_sum : ℕ) (H : total_sum = 63) :
  ∃ (part1 part2 part3 : ℕ), 
    (part1 + part2 + part3 = total_sum) ∧ 
    part1 = 21 ∧ 
    part2 = 21 ∧ 
    part3 = 21 :=
by 
  use 21, 21, 21
  split
  . exact H.symm ▸ rfl
  . split; rfl
  . split; rfl
  . rfl

end divide_triangle_l67_67713


namespace find_d_l67_67320

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) :
  d = 180 :=
sorry

end find_d_l67_67320


namespace sin_240_deg_l67_67606

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67606


namespace sequence_a8_l67_67398

theorem sequence_a8 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a n + a (n + 1)) 
  (h2 : a 7 = 120) : 
  a 8 = 194 :=
sorry

end sequence_a8_l67_67398


namespace part1_part2_l67_67809

-- Definitions based on conditions
def U := {0, 1, 2, 3}
def A (m : ℝ) : Set ℝ := { x | x ∈ U ∧ (x^2 + m * x = 0) }

-- Statement for part (1)
theorem part1 (m : ℝ) (h : U \ A m = {1, 2}) : m = -3 := sorry

-- Statement for part (2)
theorem part2 (m : ℝ) (h : ∃ x, A m = {x}) : m = 0 := sorry

end part1_part2_l67_67809


namespace equal_projections_l67_67856

noncomputable theory

variable {V : Type*} [inner_product_space ℝ V]

-- Define the vectors and their properties
variables (O A B C : V)
variables (x : ℝ)
variables (h : ℝ)
variable (Pi : affine_subspace ℝ V)

-- State the conditions for the problem
def equal_length_vectors : Prop := 
  dist O A = x ∧ dist O B = x ∧ dist O C = x

def non_collinear_points : Prop := 
  ¬ collinear ℝ ({O, A, B, C} : set V)

def plane_through_points : Prop := 
  Pi = affine_span ℝ ({A, B, C} : set V)

-- Formalize the projection equality condition
theorem equal_projections :
  equal_length_vectors O A B C x → non_collinear_points O A B C →
  (∃ (Pi : affine_subspace ℝ V), plane_through_points A B C Pi ∧
   (proj_length O A Pi = proj_length O B Pi ∧ proj_length O B Pi = proj_length O C Pi)) :=
sorry

end equal_projections_l67_67856


namespace socks_probability_l67_67814
-- First, import the necessary modules

-- Define the problem statement
theorem socks_probability :
  let colors := {red, blue, green, yellow, purple}
  let total_socks := 10
  let choose_socks := 4
  let pairs_combinations := (Finset.powersetLen 3 colors).card
  let ways_to_choose_pair := 3
  let remaining_combinations := 4
  let total_ways := (total_socks.choose choose_socks)
  let favorable_outcomes := pairs_combinations * ways_to_choose_pair * remaining_combinations
  in total_ways = 210 ∧ favorable_outcomes = 120 ∧ favorable_outcomes / total_ways = (4 / 7 : ℚ) :=
by
  sorry

end socks_probability_l67_67814


namespace smallest_enclosing_sphere_radius_l67_67278

-- Define a structure for a sphere
structure Sphere (α : Type*) [NormedField α] [NormedGroup α] :=
  (center : Point α)
  (radius : α)

-- Define the Point structure as a 3D point
structure Point (α : Type*) :=
  (x : α)
  (y : α)
  (z : α)

-- Define the conditions
def eight_tangent_spheres (α : Type*) [NormedField α] [NormedGroup α] : Prop :=
  ∀ (i j k : Sign), let center := (match i with | Sign.pos => 1 | Sign.neg => -1 end,
                               match j with | Sign.pos => 1 | Sign.neg => -1 end,
                               match k with | Sign.pos => 1 | Sign.neg => -1 end) in
  Sphere.mk center 1 -- Spheres of radius 1 tangent to coordinate planes

-- Define the theorem to prove the required radius
theorem smallest_enclosing_sphere_radius (α : Type*) [NormedField α] [NormedGroup α] :
  eight_tangent_spheres α → ∃ r : α, r = (1 + Real.sqrt 3) :=
by
  sorry

end smallest_enclosing_sphere_radius_l67_67278


namespace find_number_of_boys_l67_67386

noncomputable def number_of_boys (B G : ℕ) : Prop :=
  (B : ℚ) / (G : ℚ) = 7.5 / 15.4 ∧ G = B + 174

theorem find_number_of_boys : ∃ B G : ℕ, number_of_boys B G ∧ B = 165 := 
by 
  sorry

end find_number_of_boys_l67_67386


namespace pure_imaginary_a_value_l67_67340

theorem pure_imaginary_a_value (a : ℝ) :
  (let complex_number := (1 - complex.i) * (1 + a * complex.i)
  in complex.re complex_number = 0 ∧ complex_number ≠ 0) → a = -1 :=
sorry

end pure_imaginary_a_value_l67_67340


namespace center_of_symmetry_of_tangent_transformation_l67_67934

noncomputable def center_of_symmetry (k : ℤ) : ℝ × ℝ :=
  ((k * Real.pi / 4) - (Real.pi / 6), 0)

theorem center_of_symmetry_of_tangent_transformation :
  ∀ (k : ℤ), ∃ (x y : ℝ), (x, y) = center_of_symmetry k :=
by
  intro k
  use [(k * Real.pi / 4) - (Real.pi / 6), 0]
  rw center_of_symmetry
  simp
  sorry

end center_of_symmetry_of_tangent_transformation_l67_67934


namespace extrema_of_f_l67_67732

noncomputable def f (x : ℝ) : ℝ := (x / 8) + (2 / x)

theorem extrema_of_f :
  ∀ x, x ∈ Set.Ioo (-5 : ℝ) (10 : ℝ) →
    ((x = -4 → f x = -1) ∧ (x = 4 → f x = 1)) := 
begin
  -- Conditions for the function f
  intros x hx,
  have dom := Set.mem_Ioo.1 hx,
  split;
  { intro h,
    rw h,
    -- Evaluations
    sorry },
  sorry
end

end extrema_of_f_l67_67732


namespace sin_240_eq_neg_sqrt3_div_2_l67_67668

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67668


namespace prime_divisor_form_l67_67427

theorem prime_divisor_form {p q : ℕ} (hp : Nat.Prime p) (hpgt2 : p > 2) (hq : Nat.Prime q) (hq_dvd : q ∣ 2^p - 1) : 
  ∃ k : ℕ, q = 2 * k * p + 1 := 
sorry

end prime_divisor_form_l67_67427


namespace incircle_identity_proof_l67_67462

open Real

variable {a b c p r α : ℝ}
variable {A B C P : Point}  -- Assuming Point is a predefined type in the geometry library

-- Assuming the existence of various geometric objects and functions, e.g., angles, tangents, etc.
theorem incircle_identity_proof:
  (in_triangle A B C ∧ incircle_touch BC P ∧ ∠APB = α ∧ p = (a + b + c) / 2) →
  1 / (p - b) + 1 / (p - c) = 2 / (r * tan α) :=
  by 
    sorry

end incircle_identity_proof_l67_67462


namespace group_division_ways_l67_67493

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem group_division_ways : 
  choose 30 10 * choose 20 10 * choose 10 10 = Nat.factorial 30 / (Nat.factorial 10 * Nat.factorial 10 * Nat.factorial 10) := 
by
  sorry

end group_division_ways_l67_67493


namespace maximum_c_value_l67_67085

noncomputable def max_possible_c (a b c x y z : ℝ) : ℝ :=
  if a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
     a^x + b^y + c^z = 4 ∧
     x * a^x + y * b^y + z * c^z = 6 ∧
     x^2 * a^x + y^2 * b^y + z^2 * c^z = 9
  then c
  else 0

theorem maximum_c_value (a b c x y z : ℝ) :
  a ≥ 1 → b ≥ 1 → c ≥ 1 → x > 0 → y > 0 → z > 0 →
  a^x + b^y + c^z = 4 →
  x * a^x + y * b^y + z * c^z = 6 →
  x^2 * a^x + y^2 * b^y + z^2 * c^z = 9 →
  max_possible_c a b c x y z = c → c = real.cbrt 4 :=
by
  sorry

end maximum_c_value_l67_67085


namespace common_ratio_sum_first_n_terms_l67_67430

-- Define the conditions and variables
variables {α : Type*} [field α] (a1 q : α)
noncomputable def a_n (n : ℕ) : α := a1 * q^n

noncomputable def S_n (n : ℕ) : α := a1 * (1 - q^n) / (1 - q)

-- Hypotheses
axiom H1 : S_n 1 + (S_n 2) = 2 * S_n 3
axiom H2 : a1 - a_n 2 = 3

-- Theorems to be proven
theorem common_ratio (q_nonzero : q ≠ 0) (a1_nonzero : a1 ≠ 0) :
  q = - (1 / 2) := sorry

theorem sum_first_n_terms (a1 : α) (q : α) (n : ℕ) (H1 : S_n 1 + (S_n 2) = 2 * S_n 3) 
  (H2 : a1 - (a_n 2) = 3) :
  S_n n = (8 / 3) * (1 - (- 1 / 2)^n) := sorry

end common_ratio_sum_first_n_terms_l67_67430


namespace count_triangles_in_figure_l67_67372

def rectangle_sim (r w l : ℕ) : Prop := 
  (number_of_small_right_triangles r w l = 24) ∧
  (number_of_isosceles_triangles r w l = 6) ∧
  (number_of_half_length_isosceles_triangles r w l = 8) ∧
  (number_of_large_right_triangles r w l = 12) ∧
  (number_of_full_width_isosceles_triangles r w l = 3)

theorem count_triangles_in_figure (r w l : ℕ) (H : rectangle_sim r w l) : 
  total_number_of_triangles r w l = 53 :=
sorry

end count_triangles_in_figure_l67_67372


namespace journey_distances_l67_67176

theorem journey_distances (d : ℝ) (d1 d2 d3 d4 : ℝ) : 
  (d = d1 + d2 + d3 + d4) ∧ 
  (d1 = d / 3) ∧ 
  (d2 = d1 / 2) ∧ 
  (d3 = d1) ∧ 
  (d4 = 100) → 
  Ivan_tsarevich_correct (d = 600) ∧ (d1 + d2 + d3 + d4 = 600) ∧ (d4 = 100) :=
by sorry

def Ivan_tsarevich_correct (d : ℝ) : Prop :=
  d4 = 100

end journey_distances_l67_67176


namespace parallelepiped_volume_l67_67828

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions: a and b are unit vectors and the angle between them is π/4
def is_unit_vector (v : EuclideanSpace ℝ (Fin 3)) := ∥v∥ = 1
def angle_is_pi_over_four : ℝ := real.angle a b = real.pi / 4

-- To prove: volume of the parallelepiped is 1/2
theorem parallelepiped_volume : 
  is_unit_vector a ∧ is_unit_vector b ∧ angle_is_pi_over_four →
  abs (a • ((a + (b × a)) × b)) = 1 / 2 :=
begin
  sorry,
end

end parallelepiped_volume_l67_67828


namespace maximum_non_collinear_checkers_l67_67506

theorem maximum_non_collinear_checkers :
  ∀ (board : matrix (fin 6) (fin 6) bool),
    (∀ i j, board i j = tt → i ≠ j) → -- existence of distinct checkers
    (∀ i j k, board i j = tt ∧ board i k = tt → j ≠ k) → -- no two checkers on same row
    (∀ i j k, board j i = tt ∧ board k i = tt → j ≠ k) → -- no two checkers on same column
    (∀ i j k l m n, board i j = tt ∧ board k l = tt ∧ board m n = tt → 
      ¬(i - k) * (j - l) = (i - m) * (j - n)) → -- non-collinearity in any direction
    ∃ (count : ℕ), count = 12 :=
begin
  sorry
end

end maximum_non_collinear_checkers_l67_67506


namespace relatively_prime_number_exists_l67_67781

theorem relatively_prime_number_exists :
  -- Given numbers
  (let a := 20172017 in
   let b := 20172018 in
   let c := 20172019 in
   let d := 20172020 in
   let e := 20172021 in
   -- Number c is relatively prime to all other given numbers
   nat.gcd c a = 1 ∧
   nat.gcd c b = 1 ∧
   nat.gcd c d = 1 ∧
   nat.gcd c e = 1) :=
by {
  -- Proof omitted
  sorry
}

end relatively_prime_number_exists_l67_67781


namespace sin_240_eq_neg_sqrt3_div_2_l67_67615

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67615


namespace constant_term_of_Harriet_l67_67432

noncomputable def constant_term_of_polynomials : ℤ :=
by 
  let c := ∃ (p q : Polynomial ℤ), 
    p.degree = 3 ∧ p.leadingCoeff = 1 ∧ q.degree = 3 ∧ q.leadingCoeff = 1 ∧
    p.coeff 2 = 0 ∧ q.coeff 2 = 0 ∧ p.coeff 0 = q.coeff 0 ∧ p.coeff 0 > 0 ∧
    p * q = Polynomial.Coeff! 6 + Polynomial.Coeff! 5 * Polynomial.X + 4 * Polynomial.X^3 + 9 * Polynomial.X + Polynomial.Coeff! 16
  exact mathlib.sqrt 16 sorry

theorem constant_term_of_Harriet's_polynomial : constant_term_of_polynomials = 4 := 
sorry

end constant_term_of_Harriet_l67_67432


namespace hyperbola_eccentricity_l67_67804

variable {a b c e : ℝ}
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def parabola (c x y : ℝ) : Prop := y^2 = 4 * c * x

theorem hyperbola_eccentricity 
  (ha : a > 0) (hb : b > 0) (hc : c = Real.sqrt (a^2 + b^2))
  (h_intersect_A : hyperbola a b c (2 * c))
  (h_eccentricity_eq : e = (c / a))
  (h_distance_AB : |Complex.norm (Complex.mk c (2 * c) - Complex.mk 0 0)| = 4 * c) :
  e = Real.sqrt 2 + 1 :=
  by
    sorry

end hyperbola_eccentricity_l67_67804


namespace mn_parallel_bc_l67_67023

-- Definitions for the geometric entities and their properties
variables {A B C E F K L X Y M N : Type} 
variables [DistinguishedPointOnTriangleSide A B C E]
variables [DistinguishedPointOnTriangleSide B C E]
variables [SegmentIntersectWithLine A B K]
variables [SegmentIntersectWithLine A C L]
variables [ParallelLines E K A C]
variables [ParallelLines F L A B]
variables [IncircleTouchPoint (Triangle B E K) A B]
variables [IncircleTouchPoint (Triangle C F L) A C]
variables [LineIntersection A C E X M]
variables [LineIntersection A B F Y N]

-- Given conditions
axiom ax1 : DistinguishedPointOnTriangleSide A B C E
axiom ax2 : DistinguishedPointOnTriangleSide B C E
axiom ax3 : SegmentIntersectWithLine A B K
axiom ax4 : SegmentIntersectWithLine A C L
axiom ax5 : ParallelLines E K A C
axiom ax6 : ParallelLines F L A B
axiom ax7 : IncircleTouchPoint (Triangle B E K) A B X
axiom ax8 : IncircleTouchPoint (Triangle C F L) A C Y
axiom ax9 : LineIntersection A C E X M
axiom ax10 : LineIntersection A B F Y N
axiom ax11 : Eq (Dist A X) (Dist A Y)

-- The required proof statement
theorem mn_parallel_bc : ParallelLines M N B C :=
by
  sorry

end mn_parallel_bc_l67_67023


namespace intersection_of_A_and_B_l67_67759

def A := {x : ℝ | |x - 2| ≤ 1}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}
def C := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l67_67759


namespace standard_ellipse_max_MN_PF_l67_67860

-- Definitions of constants and conditions
def a : ℝ := sqrt(3) * b
def b : ℝ := sqrt(2)
def c : ℝ := 2

-- Ellipse equation and derived parameters
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 6 + (y^2) / 2 = 1
def is_focus (f : ℝ × ℝ) : Prop := f.1 = c ∧ f.2 = 0

-- To prove that the standard form of the ellipse equation is correct
theorem standard_ellipse :
  (a = sqrt(3) * b) ∧ (c^2 = a^2 - b^2) ∧ a > b ∧ b > 0 → ellipse_eq x y :=
sorry

-- To prove the maximum value of |MN|/|PF|is sqrt(3)
theorem max_MN_PF :
  ∃ (x y z : ℝ), is_focus (2, 0) ∧ (y = x + 2) → (z / (x + y)) ≤ sqrt(3) :=
sorry

end standard_ellipse_max_MN_PF_l67_67860


namespace angle_bao_eq_angle_cah_l67_67424

-- Define the geometric entities and the orthocenter and circumcenter properties
variables {A B C H O : Type}
  [real_trigonometry : HasRealTrigonometry Type]
  [has_angle : HasAngle Type]
  [triangle ABC : IsTriangle A B C]
  [orthocenter H : IsOrthocenter A B C H]
  [circumcenter O : IsCircumcenter A B C O]

-- Define the angles
variables (alpha beta gamma : real_trigonometry.angle)

-- The theorem to be proved
theorem angle_bao_eq_angle_cah : angle (B, A, O) = angle (C, A, H) :=
begin
  sorry
end

end angle_bao_eq_angle_cah_l67_67424


namespace total_distance_is_32_l67_67403

def first_museum_distance : ℕ := 5
def second_museum_distance : ℕ := 15
def cultural_center_distance : ℕ := 10

def traffic_adjustment : ℕ := 5
def bus_adjustment : ℕ := -2
def bicycle_adjustment : ℕ := -1

noncomputable def adjusted_second_museum_distance := second_museum_distance + traffic_adjustment
noncomputable def adjusted_cultural_center_distance := cultural_center_distance + bus_adjustment
noncomputable def adjusted_first_museum_distance := first_museum_distance + bicycle_adjustment

theorem total_distance_is_32 : adjusted_second_museum_distance + adjusted_cultural_center_distance + adjusted_first_museum_distance = 32 := by
  sorry

end total_distance_is_32_l67_67403


namespace convex_polygon_equally_divided_max_side_convex_polygon_equally_divided_min_side_impossible_l67_67864

theorem convex_polygon_equally_divided_max_side (P : Type*) [convex_polygon P] (L : ℝ) :
  ∃ X X', divides_polygon_along_line P X X' (L : ℝ) ∧
    equal_perimeters (P X X').1 (P X X').2 ∧
    equal_longest_sides (P X X').1 (P X X').2 := sorry

theorem convex_polygon_equally_divided_min_side_impossible (P : Type*) [convex_polygon P] (L : ℝ) :
  ¬∃ X X', divides_polygon_along_line P X X' (L : ℝ) ∧
    equal_perimeters (P X X').1 (P X X').2 ∧
    equal_shortest_sides (P X X').1 (P X X').2 := sorry

end convex_polygon_equally_divided_max_side_convex_polygon_equally_divided_min_side_impossible_l67_67864


namespace distance_A_B_360_l67_67195

noncomputable def distance_between_A_and_B (time_total : ℝ) (velocity_stream : ℝ) (speed_boat_still : ℝ) : ℝ := 
  let D := (time_total * (speed_boat_still + velocity_stream) * (speed_boat_still - velocity_stream)) / 
           (speed_boat_still + velocity_stream + (speed_boat_still / 2))
  in D

theorem distance_A_B_360 :
  distance_between_A_and_B 38 4 14 = 360 := by
  sorry

end distance_A_B_360_l67_67195


namespace max_coloring_number_l67_67143

open Nat

def is_coloring_valid (coloring : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ k ∈ (range (n + 1)), ∃ p1 p2 : ℕ, p1 < p2 ∧ p1 ≤ 14 ∧ p2 ≤ 14 ∧
    ((coloring p1 ∧ coloring p2 ∧ (p2 - p1 = k)) ∨ (¬coloring p1 ∧ ¬coloring p2 ∧ (p2 - p1 = k)))

theorem max_coloring_number : ∃ n, is_coloring_valid (fun x => x % 2 = 0) n ∧
  ∀ m > n, ¬is_coloring_valid (fun x => x % 2 = 0) m :=
begin
  use 11,
  split,
  { sorry },
  { sorry }
end

end max_coloring_number_l67_67143


namespace sum_of_odd_powered_coeffs_l67_67307

noncomputable def f (m n : ℕ) (x : ℝ) : ℝ :=
  (1 + x)^m + (1 + 2 * x)^n

theorem sum_of_odd_powered_coeffs (h1 : f 5 3 = (1 + 5)() ^ 5 + (1 + 2 * (3))() ^ 3) 
  (h2 : (binom m 1 . nat.cast + 2 * (binom n_over_1 () . nat.cast)  . nat.cast = 11 ) 
  (h3 : m + 2n = 11) :
  (sum_of_odd_powered_coeffs f x = 30 := 
  sorry
) 

end sum_of_odd_powered_coeffs_l67_67307


namespace intersection_point_slope_of_line_l67_67359

-- Definitions of the lines
def l1 (x y : ℝ) : Prop := x - y = 1
def l2 (x y : ℝ) : Prop := x + y = 3

-- Prove the intersection point of lines l1 and l2
theorem intersection_point : ∃ (P : ℝ × ℝ), l1 P.1 P.2 ∧ l2 P.1 P.2 ∧ P = (2, 1) := by
  use (2, 1)
  split
  · show l1 2 1
    sorry
  split
  · show l2 2 1
    sorry
  · rfl

-- Prove the slope k of the line passing through the intersection point P, given the area condition
theorem slope_of_line (k : ℝ) : (y - 1 = k * (x - 2)) → (1/2 * abs ((1 - 2*k) * (2 - 1/k))) = 4 → 
  (k = -1/2) ∨ (k = (3/2 + sqrt 2)) := by
  intro h_eqn h_area
  sorry

end intersection_point_slope_of_line_l67_67359


namespace lemon_juice_percentage_l67_67550

-- Definitions based on conditions
def total_volume : ℝ := 50
def orange_juice_volume : ℝ := 20
def grapefruit_percent : ℝ := 0.25
def grapefruit_juice_volume : ℝ := grapefruit_percent * total_volume
def lemon_juice_volume : ℝ := total_volume - orange_juice_volume - grapefruit_juice_volume

-- Question: What percentage of the drink is lemon juice?
def lemon_percent : ℝ := (lemon_juice_volume / total_volume) * 100

-- Mathematically equivalent proof problem to show percentage of lemon juice is 35%
theorem lemon_juice_percentage : lemon_percent = 35 := by
  sorry

end lemon_juice_percentage_l67_67550


namespace jack_paid_20_l67_67018

-- Define the conditions
def numberOfSandwiches : Nat := 3
def costPerSandwich : Nat := 5
def changeReceived : Nat := 5

-- Define the total cost
def totalCost : Nat := numberOfSandwiches * costPerSandwich

-- Define the amount paid
def amountPaid : Nat := totalCost + changeReceived

-- Prove that the amount paid is 20
theorem jack_paid_20 : amountPaid = 20 := by
  -- You may assume the steps and calculations here, only providing the statement
  sorry

end jack_paid_20_l67_67018


namespace inequality_a_inequality_b_l67_67191

theorem inequality_a : 
  (1 / Real.logBase 2 3) + (1 / Real.logBase 4 3) < 2 :=
sorry

theorem inequality_b (π : ℝ) (b : ℝ) (hπ : π > 1) : 
  (1 / Real.logBase 2 π) + (1 / Real.logBase b π) > 2 :=
sorry

end inequality_a_inequality_b_l67_67191


namespace fraction_of_females_l67_67442

variable (participants_last_year males_last_year females_last_year males_this_year females_this_year participants_this_year : ℕ)

-- The conditions
def conditions :=
  males_last_year = 20 ∧
  participants_this_year = (110 * (participants_last_year/100)) ∧
  males_this_year = (105 * males_last_year / 100) ∧
  females_this_year = (120 * females_last_year / 100) ∧
  participants_last_year = males_last_year + females_last_year ∧
  participants_this_year = males_this_year + females_this_year

-- The proof statement
theorem fraction_of_females (h : conditions males_last_year females_last_year males_this_year females_this_year participants_last_year participants_this_year) :
  (females_this_year : ℚ) / (participants_this_year : ℚ) = 4 / 11 :=
  sorry

end fraction_of_females_l67_67442


namespace sin_240_deg_l67_67659

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67659


namespace set_intersection_l67_67412

def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1, 2}

theorem set_intersection :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end set_intersection_l67_67412


namespace PQRS_are_concyclic_l67_67047

variables {A B C D P Q R S : Type*}

-- Assume some relevant variables and structures to represent points and convexity
variable [∀ {a b c d: Type*}, ConvexQuadrilateral a b c d]

def PR_intersects_QS (PR QS : Type*) : Prop := sorry  -- assuming an appropriate definition

def point_on_segment (P AB : Type*) : Prop := sorry  -- assuming an appropriate definition

def perpendicular_diagonals (ABCD PR QS : Type*) : Prop := sorry  -- assuming an appropriate definition

noncomputable def concyclic_points (P Q R S : Type*) : Prop := sorry  -- assuming an appropriate definition

theorem PQRS_are_concyclic
  (ABCD: Type*)
  [h: ConvexQuadrilateral ABCD]
  (P Q R S: Type*)
  (hP: point_on_segment P (AB : PairOfPointsStructure ABCD))
  (hQ: point_on_segment Q (BC : PairOfPointsStructure ABCD))
  (hR: point_on_segment R (CD : PairOfPointsStructure ABCD))
  (hS: point_on_segment S (DA : PairOfPointsStructure ABCD))
  (hIntersect: PR_intersects_QS (PR : LineSegmentStructure P R) (QS: LineSegmentStructure Q S))
  (hPerpendicular: perpendicular_diagonals ABCD (PR : LineSegmentStructure P R) (QS: LineSegmentStructure Q S)) :
  concyclic_points P Q R S := sorry

end PQRS_are_concyclic_l67_67047


namespace total_cost_price_is_correct_l67_67520

noncomputable theory

def cost_price_bicycle (Cb : ℝ) : ℝ := (320.75 / 2.15625)
def cost_price_helmet (Ch : ℝ) : ℝ := (121 / 1.716)
def cost_price_gloves (Cg : ℝ) : ℝ := (86.40 / 2.1)

def total_cost_price := cost_price_bicycle 148.75 + cost_price_helmet 70.51 + cost_price_gloves 41.14

theorem total_cost_price_is_correct : total_cost_price = 260.40 :=
by
  -- Here the proof script goes
  sorry

end total_cost_price_is_correct_l67_67520


namespace find_a_l67_67717

-- Define the sequence \( G_n \)
def G : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n+3) := (G (n+2)) + (G (n+1)) + 1
| _      := 0 -- This case won't be used but needed for non-negative domain

-- Assuming (a, b, c) form an increasing geometric sequence such that \( G_a, G_b, G_c \) are in geometric sequence,
-- and given that \( a + b + c = 2100 \), we need to prove \( a = 698 \).
theorem find_a :
    ∃ (a b c : ℕ), (G a) * (G c) = (G b)^2 ∧ a + b + c = 2100 ∧ a = 698 :=
by
    sorry

end find_a_l67_67717


namespace total_area_of_map_l67_67229

def level1_area : ℕ := 40 * 20
def level2_area : ℕ := 15 * 15
def level3_area : ℕ := (25 * 12) / 2

def total_area : ℕ := level1_area + level2_area + level3_area

theorem total_area_of_map : total_area = 1175 := by
  -- Proof to be completed
  sorry

end total_area_of_map_l67_67229


namespace length_of_rectangle_is_64_l67_67087

-- Define the given conditions
def perimeter_square : ℝ := 256
def width_rectangle : ℝ := 32
def area_square : ℝ := (perimeter_square / 4) ^ 2
def area_rectangle : ℝ := area_square / 2

-- Prove the length of the rectangle is 64 cm
theorem length_of_rectangle_is_64 : (area_rectangle / width_rectangle) = 64 := by
  sorry

end length_of_rectangle_is_64_l67_67087


namespace sum_divisible_by_4_l67_67318

def is_fundamental_term (a : List (List Int)) (c : List (Fin n)) : Int :=
  List.prod (List.of_fn (λ i => a[i][c.nth_le i i.is_lt]))

def sum_fundamental_terms (n : ℕ) (a : List (List Int)) : Int :=
  List.sum (List.of_fn (λ p : Fin n -> Fin n => is_fundamental_term a (List.of_fn p)))

theorem sum_divisible_by_4 {n : ℕ} (a : List (List Int)) (h : n ≥ 4) (h2 : ∀ i j, a[i][j] = 1 ∨ a[i][j] = -1) :
  (∃ k : ℤ, sum_fundamental_terms n a = 4 * k) :=
sorry

end sum_divisible_by_4_l67_67318


namespace negative_integer_reciprocal_of_d_l67_67579

def a : ℚ := 3
def b : ℚ := |1 / 3|
def c : ℚ := -2
def d : ℚ := -1 / 2

theorem negative_integer_reciprocal_of_d (h : d ≠ 0) : ∃ k : ℤ, (d⁻¹ : ℚ) = ↑k ∧ k < 0 :=
by
  sorry

end negative_integer_reciprocal_of_d_l67_67579


namespace sin_240_deg_l67_67662

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67662


namespace intersection_sum_l67_67394

-- Definitions for points and their coordinates
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Definitions for midpoints D and E
def D : ℝ × ℝ := ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 )
def E : ℝ × ℝ := ( (A.1 + C.1) / 2, (A.2 + C.2) / 2 )

-- Definitions for lines AE and CD
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
def line_AE (x : ℝ) : ℝ := slope A E * x + A.2 - slope A E * A.1
def line_CD (x : ℝ) : ℝ := 0 -- vertical line, not used directly in y-value computation

-- Point of intersection of AE and CD
def F_x : ℝ := 0
def F_y : ℝ := line_AE 0

-- Sum of coordinates of point F
def sum_F : ℝ := F_x + F_y

-- Prove that the sum of the coordinates of F is 5
theorem intersection_sum :
  sum_F = 5 :=
sorry -- Proof step skipped for now

end intersection_sum_l67_67394


namespace ball_arrangements_l67_67122

theorem ball_arrangements : 
  let red : ℕ := 6
  let green : ℕ := 3
  let total_balls : ℕ := red + green
  let selected_balls : ℕ := 4
  (finset.card (finset.univ.powerset.filter (λ s, s.card = selected_balls))) = 15 :=
by 
  sorry

end ball_arrangements_l67_67122


namespace probability_of_four_requests_in_four_hours_l67_67583

-- Define the Poisson mass function
def poisson_pmf (λ t m : ℝ) : ℝ :=
  (λ * t) ^ m / (Nat.factorial m) * Real.exp (-(λ * t))

-- Define the specific values for this problem
def λ : ℝ := 2
def t : ℝ := 4
def m : ℝ := 4

-- Define the expected probability
def expected_probability : ℝ := 0.0572

-- The main theorem stating the exact probability matches the expected value approximately
theorem probability_of_four_requests_in_four_hours : 
  abs (poisson_pmf λ t m - expected_probability) < 0.0001 :=
by
  sorry

end probability_of_four_requests_in_four_hours_l67_67583


namespace students_who_won_first_prize_l67_67951

theorem students_who_won_first_prize :
  ∃ x : ℤ, 30 ≤ x ∧ x ≤ 55 ∧ (x % 3 = 2) ∧ (x % 5 = 4) ∧ (x % 7 = 2) ∧ x = 44 :=
by
  sorry

end students_who_won_first_prize_l67_67951


namespace n_parallel_α_l67_67358

variables {Point : Type} [euclidean_space Point]

-- Definitions and conditions
def line (P : Type) [euclidean_space P] := P → P → Prop
def plane (P : Type) [euclidean_space P] := P → Prop

variable (l m n : Point → Point → Prop)
variable (α : Point → Prop)

-- Mutually perpendicular skew lines condition
def mutually_perpendicular_skew (l m n : Point → Point → Prop) := 
  ∀ (p q r s t u : Point), l p q → m r s → n t u → ⟪p - q, r - s⟫ = 0 ∧ ⟪r - s, t - u⟫ = 0 ∧ ⟪t - u, p - q⟫ = 0

-- Plane containing line l and perpendicular to line m condition
def plane_contains_and_perpendicular (α : Point → Prop) (l m : Point → Point → Prop) := 
  ∀ (p q r s : Point), l p q → m r s → α p ∧ α q ∧ ⟪p - q, r - s⟫ = 0

-- Statement to be proved
theorem n_parallel_α
  (h_skew : mutually_perpendicular_skew l m n)
  (h_plane : plane_contains_and_perpendicular α l m)
  : ∀ (p q r s t u : Point), n p q → α r ∧ α s → m t u → ⟪p - q, r - s⟫ = 0 := 
sorry

end n_parallel_α_l67_67358


namespace triangle_right_angled_l67_67958

theorem triangle_right_angled
  (a b c : ℝ) (r : ℝ)
  (triangle_side_lengths : a > 0 ∧ b > 0 ∧ c > 0)
  (incircle_diameter_ap : ∃ d : ℝ, b = a + d ∧ c = a + 2 * d ∧ 2 * r = a + 3 * d) :
  ∃ t : triangle, t.is_right_angled :=
sorry

end triangle_right_angled_l67_67958


namespace part1_part2_part3_l67_67800

def f (x : ℚ) : ℚ := (3 - x^2) / (1 + x^2)

theorem part1 :
  f 3 = -3/5 ∧
  f 4 = -13/17 ∧
  f (1/3) = 13/5 ∧
  f (1/4) = 47/17 :=
by
  split;
  { sorry }

theorem part2 :
  ∀ x : ℚ, f x + f (1/x) = 2 :=
by
  intro x
  sorry

theorem part3 :
  f 1 + ∑ k in Finset.range 2015 + 1, (f (k + 1) + f (1 / (k + 1))) = 4029 :=
by
  sorry

end part1_part2_part3_l67_67800


namespace range_of_a_l67_67763

theorem range_of_a
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg_x : ∀ x, x ≤ 0 → f x = 2 * x + x^2)
  (h_three_solutions : ∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 2 * a^2 + a ∧ f x2 = 2 * a^2 + a ∧ f x3 = 2 * a^2 + a) :
  -1 < a ∧ a < 1/2 :=
sorry

end range_of_a_l67_67763


namespace original_class_strength_l67_67465

theorem original_class_strength 
  (orig_avg_age : ℕ) (new_students_num : ℕ) (new_avg_age : ℕ) 
  (avg_age_decrease : ℕ) (orig_strength : ℕ) :
  orig_avg_age = 40 →
  new_students_num = 12 →
  new_avg_age = 32 →
  avg_age_decrease = 4 →
  (orig_strength + new_students_num) * (orig_avg_age - avg_age_decrease) = orig_strength * orig_avg_age + new_students_num * new_avg_age →
  orig_strength = 12 := 
by
  intros
  sorry

end original_class_strength_l67_67465


namespace sin_240_eq_neg_sqrt3_div_2_l67_67667

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67667


namespace partition_triangle_l67_67687

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l67_67687


namespace emily_chairs_count_l67_67280

theorem emily_chairs_count 
  (C : ℕ) 
  (T : ℕ) 
  (time_per_furniture : ℕ)
  (total_time : ℕ) 
  (hT : T = 2) 
  (h_time : time_per_furniture = 8) 
  (h_total : 8 * C + 8 * T = 48) : 
  C = 4 := by
    sorry

end emily_chairs_count_l67_67280


namespace total_weight_of_rings_l67_67021

theorem total_weight_of_rings :
  let orange_ring := 0.08333333333333333
  let purple_ring := 0.3333333333333333
  let white_ring := 0.4166666666666667
  orange_ring + purple_ring + white_ring = 0.8333333333333333 :=
by
  let orange_ring := 0.08333333333333333
  let purple_ring := 0.3333333333333333
  let white_ring := 0.4166666666666667
  sorry

end total_weight_of_rings_l67_67021


namespace sin_240_eq_neg_sqrt3_div_2_l67_67645

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67645


namespace number_of_birds_is_122_l67_67275

-- Defining the variables
variables (b m i : ℕ)

-- Define the conditions as part of an axiom
axiom heads_count : b + m + i = 300
axiom legs_count : 2 * b + 4 * m + 6 * i = 1112

-- We aim to prove the number of birds is 122
theorem number_of_birds_is_122 (h1 : b + m + i = 300) (h2 : 2 * b + 4 * m + 6 * i = 1112) : b = 122 := by
  sorry

end number_of_birds_is_122_l67_67275


namespace prime_digital_root_probability_l67_67825

def digital_root (n : ℕ) : ℕ :=
  if n == 0 then 0 else 1 + (n - 1) % 9

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

theorem prime_digital_root_probability :
  (finset.range 1000).filter (λ n, is_prime (digital_root n)).card.toFloat / 1000 = 0.444 :=
  sorry

end prime_digital_root_probability_l67_67825


namespace dodecagon_product_one_l67_67220

noncomputable def Q (n : ℕ) := (x_n : ℝ, y_n : ℝ)

def is_regular_dodecagon (Q : ℕ → ℂ) : Prop :=
  ∃ (x : ℕ → ℝ) (y : ℕ → ℝ),
    Q 1 = ⟨1, 0⟩ ∧
    Q 7 = ⟨-1, 0⟩ ∧
    ∀ n, Q n = (x n, y n)

theorem dodecagon_product_one (Q : ℕ → ℂ) :
  is_regular_dodecagon Q → ∑ n in range 12, (Q n) = 1 := 
by 
  sorry

end dodecagon_product_one_l67_67220


namespace solution_set_of_f_gt_zero_l67_67035

theorem solution_set_of_f_gt_zero (b : ℝ) :
  (∀ x, f x = x^2 + b * x + 1) ∧ (f (-1) = f 3) →
  { x : ℝ | f x > 0 } = { x : ℝ | x ≠ 1 } :=
by
  sorry

end solution_set_of_f_gt_zero_l67_67035


namespace cos_2beta_correct_l67_67760

open Real

theorem cos_2beta_correct (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
    (h3 : tan α = 1 / 7) (h4 : cos (α + β) = 2 * sqrt 5 / 5) :
    cos (2 * β) = 4 / 5 := 
  sorry

end cos_2beta_correct_l67_67760


namespace open_door_within_time_l67_67989

-- Define the initial conditions
def device := ℕ → ℕ

-- Constraint: Each device has 5 toggle switches ("0" or "1") and a three-digit display.
def valid_configuration (d : device) (k : ℕ) : Prop :=
  d k < 32 ∧ d k <= 999

def system_configuration (A B : device) (k : ℕ) : Prop :=
  A k = B k

-- Constraint: The devices can be synchronized to display the same number simultaneously to open the door.
def open_door (A B : device) : Prop :=
  ∃ k, system_configuration A B k

-- The main theorem: Devices A and B can be synchronized within the given time constraints to open the door.
theorem open_door_within_time (A B : device) (notebook : ℕ) : 
  (∀ k, valid_configuration A k ∧ valid_configuration B k) →
  open_door A B :=
by sorry

end open_door_within_time_l67_67989


namespace sin_240_deg_l67_67603

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67603


namespace problem_l67_67330

theorem problem
  (a b c d e : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (- (a * b))^2009 - (c + d)^2010 - e^2011 = 0 := 
by
  sorry

end problem_l67_67330


namespace polar_coordinates_of_point_l67_67264

/--
Given a point (-1, -sqrt(3)) in rectangular coordinates,
prove that its polar coordinates are (2, 4 * pi / 3).
--/
theorem polar_coordinates_of_point : 
  ∃ (r θ : ℝ), (-1 = r * real.cos θ) ∧ (-real.sqrt 3 = r * real.sin θ) ∧ (r > 0) ∧ (0 ≤ θ ∧ θ < 2 * real.pi) ∧ (r = 2) ∧ (θ = 4 * real.pi / 3) :=
by
  sorry

end polar_coordinates_of_point_l67_67264


namespace num_five_digit_numbers_condition_met_l67_67036

theorem num_five_digit_numbers_condition_met :
  let num_vals : ℕ := 1900 * 4 in
  let n_vals := (100:ℕ) * 50 + 49 in
  num_vals = 7600 :=
by
  let num_vals := 1900 * 4
  let expected_num_vals := 7600
  -- We must verify and check our calculation accurately
  have h1 : num_vals = expected_num_vals := by
    calc
      num_vals = 1900 * 4      : rfl
      ...     = 7600          : by norm_num
  exact h1

end num_five_digit_numbers_condition_met_l67_67036


namespace find_AB_l67_67859

variables (A B C M B' C' E D : Type)
variables (triangle_ABC : Triangle A B C)
variables (is_median : median A M (Triangle A B C))
variables (is_reflection : reflection A M triangle_ABC (Triangle A B' C'))
variables (AE EC BD : ℝ)
variables (AE_val : AE = 8)
variables (EC_val : EC = 16)
variables (BD_val : BD = 14)

noncomputable def AB : ℝ :=
  2 * real.sqrt 93

theorem find_AB (A B C M B' C' E D : Type) 
  (triangle_ABC : Triangle A B C)
  (is_median : median A M (Triangle A B C))
  (is_reflection : reflection A M triangle_ABC (Triangle A B' C'))
  (AE EC BD : ℝ)
  (AE_val : AE = 8)
  (EC_val : EC = 16)
  (BD_val : BD = 14) :
  AB AE EC BD = 2 * real.sqrt 93 := 
sorry

end find_AB_l67_67859


namespace arithmetic_sequence_length_l67_67823

theorem arithmetic_sequence_length :
  ∃ (n : ℕ), let a := 2 in
             let d := 4 in
             let l := 2014 in
             l = a + (n - 1) * d ∧ n = 504 :=
by
  use 504
  simp
  sorry

end arithmetic_sequence_length_l67_67823


namespace kevin_food_spending_l67_67071

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end kevin_food_spending_l67_67071


namespace largest_unorderable_dumplings_l67_67388

theorem largest_unorderable_dumplings : 
  ∀ (a b c : ℕ), 43 ≠ 6 * a + 9 * b + 20 * c :=
by sorry

end largest_unorderable_dumplings_l67_67388


namespace proof_problem_l67_67761

noncomputable def conditions : Prop :=
  ∃ (w : ℝ), w < 0 ∧ |w| < 1 ∧ ∀ x : ℝ, f x = Real.sin (w * x + (↑Real.pi / 4))

def question_1 (w : ℝ) : Prop :=
  w = (-1 / 2) →
  ((minimal_positive_period (λ x, Real.sin (w * x + (↑Real.pi / 4))) = 4 * Real.pi) ∧
   (∀ (k : ℤ), center_of_symmetry (λ x, Real.sin (w * x + (↑Real.pi / 4))) k = (2 * k * Real.pi - (↑Real.pi / 2), 0)) ∧
   (∀ (k : ℤ), axis_of_symmetry (λ x, Real.sin (w * x + (↑Real.pi / 4))) k = -2 * k * Real.pi - (↑Real.pi / 2)))

def question_2 (w : ℝ) : Prop :=
  ∀ (x : ℝ), (Real.sin (w * x + (↑Real.pi / 4))).monotonic_decreasing_on (↑Real.pi / 2, ↑Real.pi) →
  w ∈ Ico (-3 / 4 : ℝ) 0

theorem proof_problem : conditions → question_1 (-1 / 2) ∧ ∀ (w : ℝ), question_2 w :=
by
  sorry

end proof_problem_l67_67761


namespace checkered_triangle_division_l67_67695

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l67_67695


namespace angle_BAC_is_120_l67_67873

noncomputable def circumcenter (A B C : Point) : Point := sorry

noncomputable def midpoint (A B : Point) : Point := sorry

noncomputable def is_perpendicular (l1 l2 : Line) : Prop := sorry

variables {A B C O : Point} {l : Line}

-- Assumptions
axiom circumcenter_def : O = circumcenter A B C
axiom midpoint_BC_def : let M := midpoint B C in M ∈ l
axiom perpendicular_bisector_def : is_perpendicular l (bisector_angle A B C)
axiom midpoint_AO_def : let M := midpoint (A, O) in M ∈ l

-- Proof statement
theorem angle_BAC_is_120 : angle A B C = 120 :=
sorry

end angle_BAC_is_120_l67_67873


namespace cos_angle_between_vectors_l67_67835

theorem cos_angle_between_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (1, 3)
  let dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  let magnitude (x : ℝ × ℝ) : ℝ := Real.sqrt (x.1 ^ 2 + x.2 ^ 2)
  let cos_theta := dot_product a b / (magnitude a * magnitude b)
  cos_theta = -Real.sqrt 2 / 10 :=
by
  sorry

end cos_angle_between_vectors_l67_67835


namespace sin_240_eq_neg_sqrt3_over_2_l67_67654

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67654


namespace right_triangle_BC_length_l67_67397

variables {A B C : ℝ}

theorem right_triangle_BC_length (h1 : ∃ (ABC : Type), triangle ABC ∧ right_angle A ∧ tan B = 4 / 3 ∧ AB = 3) : BC = 5 :=
sorry

end right_triangle_BC_length_l67_67397


namespace cost_of_eraser_pencil_l67_67852

-- Define the cost of regular and short pencils
def cost_regular_pencil : ℝ := 0.5
def cost_short_pencil : ℝ := 0.4

-- Define the quantities sold
def quantity_eraser_pencils : ℕ := 200
def quantity_regular_pencils : ℕ := 40
def quantity_short_pencils : ℕ := 35

-- Define the total revenue
def total_revenue : ℝ := 194

-- Problem statement: Prove that the cost of a pencil with an eraser is 0.8
theorem cost_of_eraser_pencil (P : ℝ)
  (h : 200 * P + 40 * cost_regular_pencil + 35 * cost_short_pencil = total_revenue) :
  P = 0.8 := by
  sorry

end cost_of_eraser_pencil_l67_67852


namespace kevin_food_expense_l67_67070

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end kevin_food_expense_l67_67070


namespace count_real_z10_of_z30_eq_1_l67_67975

theorem count_real_z10_of_z30_eq_1 :
  ∃ S : Finset ℂ, S.card = 30 ∧ (∀ z ∈ S, z^30 = 1) ∧ (Finset.filter (λ z : ℂ, z^10 ∈ ℝ) S).card = 10 := 
by {
  sorry -- proof is not required/required to fill in
}

end count_real_z10_of_z30_eq_1_l67_67975


namespace rohit_distance_from_start_l67_67172

noncomputable def rohit_final_position : ℕ × ℕ :=
  let start := (0, 0)
  let p1 := (start.1, start.2 - 25)       -- Moves 25 meters south.
  let p2 := (p1.1 + 20, p1.2)           -- Turns left (east) and moves 20 meters.
  let p3 := (p2.1, p2.2 + 25)           -- Turns left (north) and moves 25 meters.
  let result := (p3.1 + 15, p3.2)       -- Turns right (east) and moves 15 meters.
  result

theorem rohit_distance_from_start :
  rohit_final_position = (35, 0) :=
sorry

end rohit_distance_from_start_l67_67172


namespace points_concyclic_l67_67586

open EuclideanGeometry

-- Define the conditions as hypotheses
variables (ω : Circle)
variables (A B C D E P Q T : Point)
variables (A' : Point)
variables (hA : A ∈ ω) (hB : B ∈ ω) (hC : C ∈ ω) (hD : D ∈ ω) (hE : E ∈ ω) 
variables (h_seq : cyclic_order5 ω A B C D E)
variables (hAB_BD : dist A B = dist B D)
variables (hBC_CE : dist B C = dist C E)
variables (hP : intersects (line AC) (line BE) P)
variables (hQ : is_parallel (line BE) (line AQ))
variables (hQ_int_DE : intersects (line AQ) (line (extension D E)) Q)
variables (hCircle : CircleThrough A P Q T)

-- Reflection condition
def reflection_condition (A B C : Point) := refl A C = refl C A

-- Goal: Prove the points are concyclic
theorem points_concyclic :
  concyclic ω A' B P T :=
sorry

end points_concyclic_l67_67586


namespace angle_APB_of_extended_pentagon_l67_67922

theorem angle_APB_of_extended_pentagon
  (ABCDE : Π (A B C D E : Type), A = B = C = D = E)
  (regular : regular_pentagon ABCDE)
  (extend_AB_BC : extended_sides_meet_at_P ABCDE A B C P) :
  ∠APB = 36 :=
begin
  sorry
end

end angle_APB_of_extended_pentagon_l67_67922


namespace base_conversion_addition_l67_67285

theorem base_conversion_addition :
  (214 % 8 / 32 % 5 + 343 % 9 / 133 % 4) = 9134 / 527 :=
by sorry

end base_conversion_addition_l67_67285


namespace reducible_fraction_a_l67_67300

def gcd_reducible_fraction_a (n : ℤ) : Prop :=
  Nat.gcd (n^2 + 2 * n + 4) (n^2 + n + 3) > 1

theorem reducible_fraction_a (n : ℤ) (k : ℤ) : gcd_reducible_fraction_a n ↔ n = 3 * k - 1 := 
by
  sorry

end reducible_fraction_a_l67_67300


namespace number_of_students_surveyed_l67_67006

noncomputable def M : ℕ := 60
noncomputable def N : ℕ := 90
noncomputable def B : ℕ := M / 3

theorem number_of_students_surveyed : M + B + N = 170 := by
  rw [M, N, B]
  norm_num
  sorry

end number_of_students_surveyed_l67_67006


namespace transform_f_g_even_and_periodic_g_monotonicity_and_range_l67_67794

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * sin x ^ 2 + 2 * sqrt 3 * sin x * cos x - 1

def g (x : ℝ) : ℝ :=
  (1/2) * abs (f (x + π/12)) + (1/2) * abs (f (x + π/3))

-- State the properties we need to prove

-- 1. Prove the transformation of f(x)
theorem transform_f (x : ℝ) : f(x) = 2 * sin (2 * x - π/6) :=
sorry

-- 2. Prove that g(x) is even and has the smallest positive period π/4
theorem g_even_and_periodic : 
  (∀ x : ℝ, g(-x) = g(x)) ∧ (∀ x : ℝ, g(x + π/4) = g(x)) ∧ 
  (∀ T > 0, T < π/4 → ¬∀ x : ℝ, g(x + T) = g(x)) :=
sorry

-- 3. Prove the intervals of monotonicity and the range of g(x)
theorem g_monotonicity_and_range :
  (∀ k : ℤ, 
    (∀ x : ℝ, (k*π/4 ≤ x ∧ x ≤ k*π/4 + π/8 → g(deriv x) > 0)) ∧
    (∀ x : ℝ, (k*π/4 + π/8 ≤ x ∧ x ≤ k*π/4 + π/4 → g(deriv x) < 0))) ∧
  (range g = Icc 1 (sqrt 2)) :=
sorry

end transform_f_g_even_and_periodic_g_monotonicity_and_range_l67_67794


namespace max_val_of_a_l67_67840

theorem max_val_of_a(
  f : ℝ → ℝ,
  h : ∀ x : ℝ, f x = x^2 + |2 * x - 6|
) : ∃ a : ℝ, (∀ x : ℝ, f x ≥ a) ∧ (∀ b : ℝ, (∀ x : ℝ, f x ≥ b) → a ≥ b) ∧ a = 5 :=
sorry

end max_val_of_a_l67_67840


namespace power_function_passing_through_point_l67_67803

theorem power_function_passing_through_point (a : ℝ): 
  (∃ (f : ℝ → ℝ), (∀ x, f(x) = x^a) ∧ f(3) = 27 → f(2) = 8 :=
by {
  sorry
}

end power_function_passing_through_point_l67_67803


namespace daisy_marks_three_points_l67_67045

noncomputable def mario_luigi_midpoints_count (Γ : Type) [Field Γ] [LinearOrder Γ] [MetricSpace Γ] [ProperSpace Γ] 
  (circle : Set Γ) (S : Γ) (mario_speed_ratio : Γ) (luigi_speed: Γ) (time_period: Γ)
  (daisy_positions: Finset Γ) : Prop := 
  S ∈ circle ∧ 
  mario_speed_ratio = 3 ∧ 
  luigi_speed * time_period = 2 * π ∧
  (∀ t ∈ Icc (0 : Γ) time_period, let mario_pos := exp_map_circle (mario_speed_ratio * luigi_speed) t in
                                   let luigi_pos := exp_map_circle luigi_speed t in
                                   let daisy_pos := (mario_pos + luigi_pos) / 2 in
                                   daisy_positions = set_of {daisy_pos}) ∧ 
  daisy_positions.card = 3
  -- where exp_map_circle speed t := some function to get the exponential map on the unit circle

theorem daisy_marks_three_points :
  ∃ (Γ : Type)
  (circle : Set Γ)
  (S : Γ)
  (luigi_speed : Γ),
  mario_luigi_midpoints_count Γ circle S 3 luigi_speed 6 { -- assuming 6 as the time period
  (0, sqrt 2 / 2),
  (0, -sqrt 2 / 2),
  (0, 0) } := 
sorry

end daisy_marks_three_points_l67_67045


namespace sequence_sum_of_every_third_term_l67_67573

theorem sequence_sum_of_every_third_term :
  ∃ (a : ℕ), 
  let seq_sum := 1500 * a + (1500 * 1499) / 2,
      every_third_term_sum := (500 * a) + 3 * (499 * 500) / 2
  in 
  seq_sum = 3000 →
  every_third_term_sum = 500 :=
begin
  sorry
end

end sequence_sum_of_every_third_term_l67_67573


namespace students_in_both_classes_study_Japanese_fraction_l67_67242

variable (J S : ℕ)
variable (S_eq_to_2J : S = 2 * J)

variable (senior_study_Japanese : ℚ)
variable (junior_study_Japanese : ℚ)
variable (total_students_Japanese : ℚ)
variable (total_students : ℚ)

def fraction_study_Japanese (J S : ℕ) 
  (S_eq_to_2J : S = 2 * J)
  (senior_study_Japanese : ℚ)
  (junior_study_Japanese : ℚ)
  (total_students_Japanese : ℚ)
  (total_students : ℚ) : Prop :=
  total_students_Japanese / total_students = 1 / 3

# Check initial conditions
theorem students_in_both_classes_study_Japanese_fraction (J S : ℕ)  
  (S_eq_to_2J : S = 2 * J)
  (senior_study_Japanese : senior_study_Japanese = (3/8) * ↑S)
  (junior_study_Japanese : junior_study_Japanese = (1/4) * ↑J)
  (total_students_Japanese : total_students_Japanese = senior_study_Japanese + junior_study_Japanese)
  (total_students : total_students = J + S) :
  fraction_study_Japanese J S S_eq_to_2J senior_study_Japanese junior_study_Japanese total_students_Japanese total_students :=
sorry

end students_in_both_classes_study_Japanese_fraction_l67_67242


namespace tetrahedron_volume_eq_l67_67578

noncomputable def volume_of_tetrahedron (W M N O : ℝ^3) : Real :=
  1 / 6 * abs (W - M).cross_product(W - N).dot_product(W - O)

theorem tetrahedron_volume_eq (W M N O : ℝ^3) (U : Set ℝ^3) :
  is_unit_cube U →
  W = ⟨0, 0, 0⟩ ∧ U.contains W ∧
  M = midpoint (W + uvec 1 0 0) (W + uvec 0 1 0) ∧ (U.contains M) ∧
  N = midpoint (W + uvec 1 0 0) (W + uvec 0 0 1) ∧ (U.contains N) ∧
  O = midpoint (W + uvec 0 1 0) (W + uvec 0 0 1) ∧ (U.contains O) →
  volume_of_tetrahedron W M N O = √2 / 48 :=
by
  -- proof is omitted (sorry)
  sorry

end tetrahedron_volume_eq_l67_67578


namespace correct_conclusions_l67_67356

-- Definitions for given conditions
variables {x y a : ℝ}

-- Conditions
def system_of_equations := 3 * x + 2 * y = 8 + a ∧ 2 * x + 3 * y = 3 * a
def opposite_numbers := x = -y

-- Prove the given conclusions
theorem correct_conclusions (h : system_of_equations) :
  (opposite_numbers → a = -2) ∧
  (∀ a, (∀ x y, system_of_equations → x - y = 8 - 2 * a)) ∧
  (∀ a, (∀ x y, system_of_equations → 7 * x + 3 * y = 24)) ∧
  (∀ x y, (system_of_equations → x = -3 / 7 * y + 24 / 7)) :=
sorry

end correct_conclusions_l67_67356


namespace minimum_value_f_on_interval_l67_67337

def f (x : ℝ) (m : ℝ) := 2 * x^3 - 6 * x^2 + m

theorem minimum_value_f_on_interval :
  ∃ (m : ℝ), (∀ x ∈ (set.Icc (-2 : ℝ) (2 : ℝ)), f x m ≤ 3)
  → (m = 3 → (∀ y : ℝ, y ∈ (set.Icc (-2 : ℝ) (2 : ℝ)) → f y m ≥ -37))
:= sorry

end minimum_value_f_on_interval_l67_67337


namespace probability_no_defective_pencils_l67_67524

theorem probability_no_defective_pencils (total_pencils : ℕ) (defective_pencils : ℕ) (purchased_pencils : ℕ) :
  total_pencils = 7 → defective_pencils = 2 → purchased_pencils = 3 →
  (∃ (probability : ℚ), probability = (2 : ℚ) / 7) :=
by
  intros h1 h2 h3
  use (2 : ℚ) / 7
  sorry

end probability_no_defective_pencils_l67_67524


namespace decrease_of_negative_distance_l67_67379

theorem decrease_of_negative_distance (x : Int) (increase : Int → Int) (decrease : Int → Int) :
  (increase 30 = 30) → (decrease 5 = -5) → (decrease 5 = -5) :=
by
  intros
  sorry

end decrease_of_negative_distance_l67_67379


namespace sum_eq_13_5_l67_67189

def a_n (n: ℕ) := (1 / (2^n : ℝ))
def b_n (n: ℕ) := (1 / (3^n : ℝ))
/-...other definitions for c_n, ..., z_n...-/
def z_n (n: ℕ) := (1 / (27^n : ℝ))

def S := { (i1, i2, ..., i26) | i1 ≥ 1 ∧ ∀ j, 2 ≤ j ∧ j ≤ 26 → (i_j ≥ 0 ∧ i_j ∈ ℤ)}

noncomputable def sum :=
  ∑' (x : { (i1, i2, ..., i26) | (i1, i2, ..., i26) ∈ S }),
    a_n x.i1 * b_n x.i2 -- * c_n x.i3 * ...* z_n x.i26

theorem sum_eq_13_5 : sum = 13.5 := by
  sorry

end sum_eq_13_5_l67_67189


namespace percent_additional_discount_l67_67452

def additional_discount_percentage
  (P_full : ℝ) (d1 : ℝ) (P_three_years : ℝ) : ℝ :=
  let P_after_first_discount := P_full * (1 - d1) in
  let additional_discount_amount := P_after_first_discount - P_three_years in
  (additional_discount_amount / P_after_first_discount) * 100

theorem percent_additional_discount
  (P_full : ℝ)
  (d1 : ℝ)
  (P_three_years : ℝ)
  (h1 : P_full = 85)
  (h2 : d1 = 0.20)
  (h3 : P_three_years = 51) :
  additional_discount_percentage P_full d1 P_three_years = 25 := by
  sorry

end percent_additional_discount_l67_67452


namespace sin_240_deg_l67_67604

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67604


namespace S_n_formula_l67_67032

def S_n (n : ℕ) : ℚ := ∑ k in Finset.range n, 1 / ((k+1 : ℕ) * (k+2 : ℕ))

theorem S_n_formula (n : ℕ) : S_n n = n / (n + 1) := 
  sorry

end S_n_formula_l67_67032


namespace equilateral_triangle_infinite_construction_l67_67355

theorem equilateral_triangle_infinite_construction (a b c : ℝ) 
  (h_triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) 
  (S : ℝ := (a + b + c) / 2)
  (h_infinite_construction : ∀ n : ℕ, 
    let S_n := S / (2 ^ n) in 
    S_n > |a - b| ∧ S_n > |b - c| ∧ S_n > |c - a|) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_infinite_construction_l67_67355


namespace height_prediction_at_age_10_l67_67214

theorem height_prediction_at_age_10 :
  ∀ (x : ℝ), x = 10 → (y = 7.19 * x + 73.93) → y ≈ 145.83 := sorry

end height_prediction_at_age_10_l67_67214


namespace distance_from_center_to_line_l67_67098

-- Define the circle and its center
def is_circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def center : (ℝ × ℝ) := (1, 0)

-- Define the line equation y = tan(30°) * x
def is_line (x y : ℝ) : Prop := y = (1 / Real.sqrt 3) * x

-- Function to compute the distance from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (abs (A * p.1 + B * p.2 + C)) / Real.sqrt (A^2 + B^2)

-- The main theorem to be proven:
theorem distance_from_center_to_line : 
  distance_point_to_line center (1 / Real.sqrt 3) (-1) 0 = 1 / 2 :=
  sorry

end distance_from_center_to_line_l67_67098


namespace induction_step_l67_67996

theorem induction_step
  (x y : ℝ)
  (k : ℕ)
  (base : ∀ n, ∃ m, (n = 2 * m - 1) → (x^n + y^n) = (x + y) * m) :
  (x^(2 * k + 1) + y^(2 * k + 1)) = (x + y) * (k + 1) :=
by
  sorry

end induction_step_l67_67996


namespace relatively_prime_number_exists_l67_67783

theorem relatively_prime_number_exists :
  -- Given numbers
  (let a := 20172017 in
   let b := 20172018 in
   let c := 20172019 in
   let d := 20172020 in
   let e := 20172021 in
   -- Number c is relatively prime to all other given numbers
   nat.gcd c a = 1 ∧
   nat.gcd c b = 1 ∧
   nat.gcd c d = 1 ∧
   nat.gcd c e = 1) :=
by {
  -- Proof omitted
  sorry
}

end relatively_prime_number_exists_l67_67783


namespace zero_in_interval_l67_67485

def f (x : ℝ) : ℝ := (1 / x) - 6 + 2 * x

theorem zero_in_interval : ∃ (c : ℝ), c ∈ Ioo 2 3 ∧ f c = 0 := by
  sorry

end zero_in_interval_l67_67485


namespace sum_of_roots_l67_67426

def f (x : ℝ) : ℝ := x^3 + x + 1

theorem sum_of_roots (a b c : ℝ) (h : a = 4096 ∧ b = 16 ∧ c = -9 ∧ (∀ z : ℝ, f (4*z/4) = 4096*z^3 + 16*z + 1)) :
  ∑ z in {z : ℝ | 4096*z^3 + 16*z - 9 = 0}.toFinset, z = 0 := by
  sorry

end sum_of_roots_l67_67426


namespace triangle_parts_sum_eq_l67_67708

-- Define the total sum of numbers in the triangle
def total_sum : ℕ := 63

-- Define the required sum for each part
def required_sum : ℕ := 21

-- Define the possible parts that sum to the required sum
def part1 : list ℕ := [10, 6, 5]  -- this sums to 21
def part2 : list ℕ := [10, 5, 4, 1, 1]  -- this sums to 21

def part_sum (part : list ℕ) : ℕ := part.sum

-- The main theorem
theorem triangle_parts_sum_eq : 
  part_sum part1 = required_sum ∧ 
  part_sum part2 = required_sum ∧ 
  part_sum (list.drop (list.length part1 + list.length part2) [10, 6, 5, 10, 5, 4, 1, 1]) = required_sum :=
by
  sorry

end triangle_parts_sum_eq_l67_67708


namespace common_divisors_of_90_and_75_l67_67365

def is_divisor (n d : ℤ) : Prop :=
  d ≠ 0 ∧ n % d = 0

def common_divisors_count (a b : ℤ) : ℕ :=
  (List.filter (λ d, is_divisor a d ∧ is_divisor b d) (List.range (Int.natAbs a + 1))).length

theorem common_divisors_of_90_and_75 : common_divisors_count 90 75 = 8 := sorry

end common_divisors_of_90_and_75_l67_67365


namespace contest_route_count_l67_67587

def grid : List (List String) :=
[["C"],
 ["C", "O", "C"],
 ["C", "O", "N", "O", "C"],
 ["C", "O", "N", "T", "N", "O", "C"],
 ["C", "O", "N", "T", "E", "T", "N", "O", "C"],
 ["C", "O", "N", "T", "E", "S", "E", "T", "N", "O", "C"],
 ["C", "O", "N", "T", "E", "S", "T", "S", "E", "T", "N", "O", "C"]]

-- Horizontal or vertical adjacency condition.
def adjacent (cell1 cell2 : (Nat, Nat)) : Bool :=
  let (x1, y1) := cell1
  let (x2, y2) := cell2
  (abs (x1 - x2) = 0 && abs (y1 - y2) = 1) || (abs (x1 - x2) = 1 && abs (y1 - y2) = 0)

-- Route calculation based on the adjacency and grid structure.
def count_routes (grid : List (List String)) : Nat :=
  2^6 + 2^6 - 1

theorem contest_route_count : count_routes grid = 127 := by
  sorry

end contest_route_count_l67_67587


namespace exists_x0_for_which_f_lt_g_l67_67049

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin x ^ 2017
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2017 + 2017 ^ x

theorem exists_x0_for_which_f_lt_g :
  ∃ x0 : ℝ, ∀ x : ℝ, x > x0 → f x < g x :=
sorry

end exists_x0_for_which_f_lt_g_l67_67049


namespace arithmetic_sequence_sum_l67_67886

noncomputable def S (n : ℕ) (a_1 d : ℝ) : ℝ := n * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (
  a_2 a_4 a_6 a_8 a_1 d : ℝ,
  h1 : a_2 + a_6 = 10,
  h2 : a_4 * a_8 = 45,
  h3 : a_2 = a_1 + d,
  h4 : a_4 = a_1 + 3 * d,
  h5 : a_6 = a_1 + 5 * d,
  h6 : a_8 = a_1 + 7 * d,
  h7 : 2 * a_4 = 10,
  h8 : a_4 = 5,
  h9 : 5 * a_8 = 45,
  h10 : a_8 = 9,
  h11 : d = 1,
  h12 : a_1 = 2
) : S 5 a_1 d = 20 := sorry

end arithmetic_sequence_sum_l67_67886


namespace curve_crosses_itself_and_point_of_crossing_l67_67585

-- Define the function for x and y
def x (t : ℝ) : ℝ := t^2 + 1
def y (t : ℝ) : ℝ := t^4 - 9 * t^2 + 6

-- Definition of the curve crossing itself and the point of crossing
theorem curve_crosses_itself_and_point_of_crossing :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁ = 10 ∧ y t₁ = 6) :=
by
  sorry

end curve_crosses_itself_and_point_of_crossing_l67_67585


namespace sin_240_l67_67619

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67619


namespace simplify_eval_expression_l67_67924

variables (a b : ℝ)

theorem simplify_eval_expression :
  a = Real.sqrt 3 →
  b = Real.sqrt 3 - 1 →
  ((3 * a) / (2 * a - b) - 1) / ((a + b) / (4 * a^2 - b^2)) = 3 * Real.sqrt 3 - 1 :=
by
  sorry

end simplify_eval_expression_l67_67924


namespace range_of_a_l67_67343

noncomputable def has_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, (f x > f y) ∧ ∃ z w : ℝ, (f z < f w)

theorem range_of_a (a : ℝ) :
  let f (x : ℝ) := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1 in
  has_extrema f → a ∈ Iio (-1) ∪ Ioi 2 :=
sorry

end range_of_a_l67_67343


namespace sin_240_eq_neg_sqrt3_div_2_l67_67664

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67664


namespace yuna_class_total_students_l67_67581

theorem yuna_class_total_students (K M KM : ℕ) (h1 : K = 28) (h2 : M = 27) (h3 : KM = 22) (h4 : K + M - KM = 33) : K + M - KM = 33 :=
by
  rw [h1, h2, h3]
  exact h4

end yuna_class_total_students_l67_67581


namespace sin_240_l67_67624

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67624


namespace noah_garden_larger_by_75_l67_67431

-- Define the dimensions of Liam's garden
def length_liam : ℕ := 30
def width_liam : ℕ := 50

-- Define the dimensions of Noah's garden
def length_noah : ℕ := 35
def width_noah : ℕ := 45

-- Define the areas of the gardens
def area_liam : ℕ := length_liam * width_liam
def area_noah : ℕ := length_noah * width_noah

theorem noah_garden_larger_by_75 :
  area_noah - area_liam = 75 :=
by
  -- The proof goes here
  sorry

end noah_garden_larger_by_75_l67_67431


namespace sin_240_eq_neg_sqrt3_div_2_l67_67674

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67674


namespace remainder_of_3_pow_19_div_10_l67_67527

def w : ℕ := 3 ^ 19

theorem remainder_of_3_pow_19_div_10 : w % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_div_10_l67_67527


namespace coprime_with_others_l67_67789

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end coprime_with_others_l67_67789


namespace sin_240_eq_neg_sqrt3_over_2_l67_67651

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67651


namespace tire_usage_is_25714_l67_67198

-- Definitions based on conditions
def car_has_six_tires : Prop := (4 + 2 = 6)
def used_equally_over_miles (total_miles : ℕ) (number_of_tires : ℕ) : Prop := 
  (total_miles * 4) / number_of_tires = 25714

-- Theorem statement based on proof
theorem tire_usage_is_25714 (miles_driven : ℕ) (num_tires : ℕ) 
  (h1 : car_has_six_tires) 
  (h2 : miles_driven = 45000)
  (h3 : num_tires = 7) :
  used_equally_over_miles miles_driven num_tires :=
by
  sorry

end tire_usage_is_25714_l67_67198


namespace arccos_cos_11_eq_l67_67599

theorem arccos_cos_11_eq: Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end arccos_cos_11_eq_l67_67599


namespace trisha_cookies_count_l67_67302

def art_cookie_radius : ℝ := 2
def trisha_cookie_side : ℝ := 4
def art_batches_cookies : ℕ := 18

def art_cookie_area : ℝ := Real.pi * art_cookie_radius ^ 2
def total_dough_area : ℝ := art_batches_cookies * art_cookie_area
def trisha_cookie_area : ℝ := trisha_cookie_side ^ 2

theorem trisha_cookies_count : total_dough_area / trisha_cookie_area = 14 := 
by
  -- Skipping the proof as instructed, this should be the place to prove the statement
  sorry

end trisha_cookies_count_l67_67302


namespace incorrect_average_l67_67466

theorem incorrect_average (S : ℕ) 
  (h_correct_average : (S + 86) / 10 = 26)
  (h_incorrect_sum : S + 26) : 
  (S + 26) / 10 = 20 :=
by
  sorry

end incorrect_average_l67_67466


namespace distinct_solution_pairs_l67_67735

theorem distinct_solution_pairs : 
  {p : ℝ × ℝ | (2 * p.1 ^ 2 - p.1 = p.2 ^ 2) ∧ (p.2 = 4 * p.1 * p.2)}.finite.toFinset.card = 2 := 
by
  sorry

end distinct_solution_pairs_l67_67735


namespace sin_240_eq_neg_sqrt3_div_2_l67_67669

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67669


namespace find_c_l67_67349

variables (a b c : ℝ)

noncomputable def f (x : ℝ) : ℝ := x^3 + a * x^2 + b

theorem find_c (h1 : b = c - a) 
              (h2 : (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f(x₁) = 0 ∧ f(x₂) = 0 ∧ f(x₃) = 0))
              (h3 : ∀ x : ℝ, ((x < -3) ∨ (1 < x ∧ x < 3/2) ∨ (3/2 < x)) → f(x) = 0)
              : c = 1 := 
sorry

end find_c_l67_67349


namespace average_weight_of_children_l67_67773

theorem average_weight_of_children :
  let ages := [3, 4, 5, 6, 7]
  let regression_equation (x : ℕ) := 3 * x + 5
  let average l := (l.foldr (· + ·) 0) / l.length
  average (List.map regression_equation ages) = 20 :=
by
  sorry

end average_weight_of_children_l67_67773


namespace girl_students_not_playing_soccer_l67_67013

theorem girl_students_not_playing_soccer (total_students boys total_soccer_players : ℕ) (soccer_players_are_boys_percent : ℚ) (H1 : total_students = 420) (H2 : boys = 296) (H3 : total_soccer_players = 250) (H4 : soccer_players_are_boys_percent = 0.86) : 
  let boys_playing_soccer := soccer_players_are_boys_percent * total_soccer_players,
      boys_not_playing_soccer := boys - boys_playing_soccer,
      students_not_playing_soccer := total_students - total_soccer_players,
      girls_not_playing_soccer := students_not_playing_soccer - boys_not_playing_soccer in
  girls_not_playing_soccer = 89 := 
by 
  let boys_playing_soccer := soccer_players_are_boys_percent * total_soccer_players
  let boys_not_playing_soccer := boys - boys_playing_soccer
  let students_not_playing_soccer := total_students - total_soccer_players
  let girls_not_playing_soccer := students_not_playing_soccer - boys_not_playing_soccer
  show girls_not_playing_soccer = 89 from sorry

end girl_students_not_playing_soccer_l67_67013


namespace triangle_sides_ratio_l67_67295

theorem triangle_sides_ratio
  (triangle_ABC : Triangle)
  (median_split : ∃ (B M K L : Point) (x : ℝ), B ≠ M ∧ B ≠ K ∧ B ≠ L ∧ BK = x ∧ KL = x ∧ LM = x ∧ BM = B + 2 * K + L ∧ inscribed_circle_splits_median B M K L)
  (side_ratio : ∃ (a b : ℝ), b = 2 * a) :
  ∃ (ratio : Ratio), ratio = (5 : ℕ, 10 : ℕ, 13 : ℕ) := by
  sorry

end triangle_sides_ratio_l67_67295


namespace triangle_parts_sum_eq_l67_67704

-- Define the total sum of numbers in the triangle
def total_sum : ℕ := 63

-- Define the required sum for each part
def required_sum : ℕ := 21

-- Define the possible parts that sum to the required sum
def part1 : list ℕ := [10, 6, 5]  -- this sums to 21
def part2 : list ℕ := [10, 5, 4, 1, 1]  -- this sums to 21

def part_sum (part : list ℕ) : ℕ := part.sum

-- The main theorem
theorem triangle_parts_sum_eq : 
  part_sum part1 = required_sum ∧ 
  part_sum part2 = required_sum ∧ 
  part_sum (list.drop (list.length part1 + list.length part2) [10, 6, 5, 10, 5, 4, 1, 1]) = required_sum :=
by
  sorry

end triangle_parts_sum_eq_l67_67704


namespace smallest_lcm_of_quadruplets_l67_67121

theorem smallest_lcm_of_quadruplets :
  ∀ (a b c d : ℕ), ∃ n : ℕ,
  (gcd ?m_1 ?m_2 ?m_3 ?m_4 = 36) ∧
  (∃ quad_count = 36000, quadruplets_count quad_count ∧ (lcm ?m_1 ?m_2 ?m_3 ?m_4 = n)) ∧
  (n = 38880) := 
sorry

end smallest_lcm_of_quadruplets_l67_67121


namespace inverse_graph_passes_through_l67_67929

-- Definitions based on the problem's conditions
def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x

def passes_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop := f p.1 = p.2

-- Function y = f^-1(x) - x passes through the point (-1, 2)
theorem inverse_graph_passes_through
  (f : ℝ → ℝ)
  (hinv : has_inverse f)
  (hgraph : passes_through (λ x, x - f(x)) (1, 2)) :
  passes_through (λ x, (classical.some hinv) x - x) (-1, 2) :=
sorry

end inverse_graph_passes_through_l67_67929


namespace sequence_product_fourth_power_l67_67277

theorem sequence_product_fourth_power (a : ℕ) : 
  let a_1 := a in
  let a_2 := a * a^2 in
  let a_3 := a_2 * a^2 in
  let a_4 := a_3 * a^2 in
  let a_5 := a_4 * a^2 in
  let a_6 := a_5 * a^2 in
  let a_7 := a_6 * a^2 in
  a_1 * a_3 * a_5 * a_7 = (a_4)^4 := 
by 
  sorry -- Proof placeholder

end sequence_product_fourth_power_l67_67277


namespace calculate_pi_approx_l67_67213

-- Conditions:
def side_length_square : ℝ := 1
def beans_in_square : ℕ := 5001
def beans_in_circle : ℕ := 3938

-- Given that the conditions hold, we need to prove the calculated value of pi is approximately 3.15 rounded to three significant figures.
theorem calculate_pi_approx (side_length_square = 1) (beans_in_square = 5001) (beans_in_circle = 3938) : 
  abs ((4 * (beans_in_circle.to_rat / beans_in_square.to_rat)).to_real - 3.15) < 0.005 := 
by
  sorry

end calculate_pi_approx_l67_67213


namespace sum_of_values_abs_eq_23_l67_67523

theorem sum_of_values_abs_eq_23 (x : ℝ) : 
  (| x - 5 | = 23) → (x = 28 ∨ x = -18) :=
begin
  sorry
end

end sum_of_values_abs_eq_23_l67_67523


namespace checkered_triangle_division_l67_67701

-- Define the conditions as assumptions
variable (T : Set ℕ) 
variable (sum_T : Nat) (h_sumT : sum_T = 63)
variable (part1 part2 part3 : Set ℕ)
variable (sum_part1 sum_part2 sum_part3 : Nat)
variable (h_part1 : sum part1 = 21)
variable (h_part2 : sum part2 = 21)
variable (h_part3 : sum part3 = 21)

-- Define the goal as a theorem
theorem checkered_triangle_division : 
  (∃ part1 part2 part3 : Set ℕ, sum part1 = 21 ∧ sum part2 = 21 ∧ sum part3 = 21 ∧ Disjoint part1 (part2 ∪ part3) ∧ Disjoint part2 part3 ∧ T = part1 ∪ part2 ∪ part3) :=
sorry

end checkered_triangle_division_l67_67701


namespace remainder_of_floor_division_l67_67257

theorem remainder_of_floor_division : 
  ∀ (m : ℤ), (3^123 : ℤ) ≡ 1 [MOD 2] →
            (3^8 : ℤ) ≡ 1 [MOD 16] →
            (5 * 13 : ℤ) ≡ 1 [MOD 16] →
            m ≡ 9 * 13 [MOD 16] →
            m ≡ 5 [MOD 16] 
:= by 
  sorry

end remainder_of_floor_division_l67_67257


namespace find_2a_plus_b_l67_67040

theorem find_2a_plus_b (a b : ℝ) (h1 : 0 < a ∧ a < π / 2) (h2 : 0 < b ∧ b < π / 2)
  (h3 : 5 * (Real.sin a)^2 + 3 * (Real.sin b)^2 = 2)
  (h4 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 3) :
  2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l67_67040


namespace initial_workers_number_l67_67081

-- Define the initial problem
variables {W : ℕ} -- Number of initial workers
variables (Work1 : ℕ := W * 8) -- Work done for the first hole
variables (Work2 : ℕ := (W + 65) * 6) -- Work done for the second hole
variables (Depth1 : ℕ := 30) -- Depth of the first hole
variables (Depth2 : ℕ := 55) -- Depth of the second hole

-- Expressing the conditions and question
theorem initial_workers_number : 8 * W * 55 = 30 * (W + 65) * 6 → W = 45 :=
by
  sorry

end initial_workers_number_l67_67081


namespace sin_240_eq_neg_sqrt3_div_2_l67_67631

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67631


namespace no_square_number_divisible_by_six_in_range_l67_67286

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ (x : ℕ), (x ^ 2) % 6 = 0 ∧ 39 < x ^ 2 ∧ x ^ 2 < 120 :=
by
  sorry

end no_square_number_divisible_by_six_in_range_l67_67286


namespace checkered_triangle_division_l67_67693

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l67_67693


namespace find_difference_l67_67961

theorem find_difference (x y : ℚ) (h₁ : x + y = 520) (h₂ : x / y = 3 / 4) : y - x = 520 / 7 :=
by
  sorry

end find_difference_l67_67961


namespace seashells_solution_l67_67453

variable (s t m : ℕ)

def seashells_problem := 
    (s = 18) ∧ 
    (t = 65) → 
    (m = t - s) → 
    m = 47

theorem seashells_solution (s t m : ℕ) : seashells_problem s t m := by
  intros h1 h2 h3
  rw [h2, h3]
  exact Nat.sub_self_add 18 47 

end seashells_solution_l67_67453


namespace carla_time_l67_67464

-- Define the variables and constants
variables (C : ℝ)

-- Conditions
def sylvia_work_rate := 1 / 45
def carla_work_rate := 1 / C
def combined_work_rate := 1 / 18

-- Theorem statement
theorem carla_time (h : sylvia_work_rate + carla_work_rate = combined_work_rate) : C = 30 :=
sorry

end carla_time_l67_67464


namespace june_ride_time_l67_67405

theorem june_ride_time
  (distance_june_julia : ℝ)
  (time_june_julia : ℝ)
  (distance_to_bernard : ℝ)
  (rest_time : ℝ) : 
  distance_june_julia = 1.2 →
  time_june_julia = 4.8 →
  distance_to_bernard = 4.5 →
  rest_time = 2 →
  let speed := distance_june_julia / time_june_julia in
  let riding_time := distance_to_bernard / speed in
  let total_time := riding_time + rest_time in
  total_time = 20 := 
by
  sorry

end june_ride_time_l67_67405


namespace tetrahedron_edge_assignment_possible_l67_67448

theorem tetrahedron_edge_assignment_possible 
(s S a b : ℝ) 
(hs : s ≥ 0) (hS : S ≥ 0) (ha : a ≥ 0) (hb : b ≥ 0) :
  ∃ (e₁ e₂ e₃ e₄ e₅ e₆ : ℝ),
    e₁ ≥ 0 ∧ e₂ ≥ 0 ∧ e₃ ≥ 0 ∧ e₄ ≥ 0 ∧ e₅ ≥ 0 ∧ e₆ ≥ 0 ∧
    (e₁ + e₂ + e₃ = s) ∧ (e₁ + e₄ + e₅ = S) ∧
    (e₂ + e₄ + e₆ = a) ∧ (e₃ + e₅ + e₆ = b) := by
  sorry

end tetrahedron_edge_assignment_possible_l67_67448


namespace monochromatic_prob_l67_67725

-- Define the vertices of the pentagon
def vertices := Finset.range 5

-- Define the edges of the pentagon including diagonals
def pentagon_edges : Finset (Finset ℕ) :=
  let sides : Finset (Finset ℕ) := Finset.filter (λ s, s.card = 2) (vertices.powerset),
  let diagonals : Finset (Finset ℕ) := Finset.filter (λ s, s.card = 2 ∧ (s.sum ≠ 1 ∨ s.sum ≠ 4) ∨ (s.sum ≠ 2 ∨ s.sum ≠ 3))
  sides ∪ diagonals

-- The event which checks if there is a monochromatic triangle
def monochromatic_triangle (colors : Finset (Finset ℕ) → Bool) : Prop :=
  ∃ t ∈ pentagon_edges.triples, (colors t ∧ ∀ e ∈ t, colors e = colors t) ∨ (¬colors t ∧ ∀ e ∈ t, colors e = colors t)

-- Given conditions in terms of probability
noncomputable def color_distribution : Distribution (Finset (Finset ℕ) → Bool) :=
  probability.uniform (Finset.image (λ c : (Finset (Finset ℕ)) → Bool, c) (Finset.powerset pentagon_edges))

-- Problem statement to be proved
theorem monochromatic_prob : 
  Pr[monochromatic_triangle] = 253 / 256 := 
begin
  sorry
end

end monochromatic_prob_l67_67725


namespace length_of_congruent_sides_of_isosceles_triangle_l67_67301

noncomputable def side_length_square : Real := 2
noncomputable def area_square := side_length_square ^ 2
noncomputable def total_area_triangles := (1 / 2) * area_square
noncomputable def area_one_triangle := total_area_triangles / 4
noncomputable def base_one_triangle := side_length_square 
noncomputable def height_one_triangle := (2 * area_one_triangle) / base_one_triangle

theorem length_of_congruent_sides_of_isosceles_triangle :
  let hypotenuse := Math.sqrt (1^2 + (height_one_triangle / 2)^2) in
  hypotenuse = (Real.sqrt 5) / 2 :=
by
  calc hypotenuse = Math.sqrt (1 + (1 / 4)) : sorry
                ... = Math.sqrt (5 / 4)      : sorry
                ... = Real.sqrt 5 / 2        : sorry

end length_of_congruent_sides_of_isosceles_triangle_l67_67301


namespace area_of_triangle_CDE_l67_67008

-- Definitions based on the given conditions
def isosceles_right_triangle (A B C : Type) (rightAngle : C = 90 ∧ isosceles A B) : Prop := 
  ∃ (area : ℝ), area = 18

def trisect_angle (A B C D E : Type) (trisected : trisect A C B intersects A B D E) : Prop := 
  ∃ (trisect : trisect_intersect A C B A B D E), trisect

-- The theorem to prove the area of triangle CDE is 4.5
theorem area_of_triangle_CDE :
  ∀ (A B C D E : Type),
  isosceles_right_triangle A B C (A rfl) →
  trisect_angle A B C D E (D ∧ E rfl) →
  area A C D E = 4.5 :=
by
  sorry

end area_of_triangle_CDE_l67_67008


namespace number_of_correct_statements_l67_67382

variable {a : ℕ → ℝ} (a1 : ℝ) (q : ℝ) (c k : ℝ)

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ a1 q, a 0 = a1 ∧ ∀ n, a (n+1) = a n * q

theorem number_of_correct_statements :
  is_geometric a →
  let statements : ℕ → Prop :=
    λ i, match i with
    | 1 => is_geometric (λ n, (a n)^2) ∧ is_geometric (λ n, a (n*2))
    | 2 => ∀ n, ∃ k, (Real.log (a n) - Real.log (a 0)) = k * n
    | 3 => is_geometric (λ n, 1 / (a n)) ∧ is_geometric (λ n, |a n|)
    | 4 => is_geometric (λ n, c * a n) ∧ (k ≠ 0 → (¬ is_geometric (λ n, a n + k) ∧ ¬ is_geometric (λ n, a n - k)))
    | _ => false
    end
  in ∃ count : ℕ, count = 2 ∧ (count = Nat.card (Finset.filter statements (Finset.range 4)))
  :=
by 
  sorry

end number_of_correct_statements_l67_67382


namespace cone_volume_ratio_l67_67148

def volume (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

theorem cone_volume_ratio :
  let r_C := 15.6
  let h_C := 29.5
  let r_D := 29.5
  let h_D := 15.6
  volume r_C h_C / volume r_D h_D = 156 / 295 :=
  by
    sorry

end cone_volume_ratio_l67_67148


namespace tetrahedron_edge_assignment_possible_l67_67447

theorem tetrahedron_edge_assignment_possible 
(s S a b : ℝ) 
(hs : s ≥ 0) (hS : S ≥ 0) (ha : a ≥ 0) (hb : b ≥ 0) :
  ∃ (e₁ e₂ e₃ e₄ e₅ e₆ : ℝ),
    e₁ ≥ 0 ∧ e₂ ≥ 0 ∧ e₃ ≥ 0 ∧ e₄ ≥ 0 ∧ e₅ ≥ 0 ∧ e₆ ≥ 0 ∧
    (e₁ + e₂ + e₃ = s) ∧ (e₁ + e₄ + e₅ = S) ∧
    (e₂ + e₄ + e₆ = a) ∧ (e₃ + e₅ + e₆ = b) := by
  sorry

end tetrahedron_edge_assignment_possible_l67_67447


namespace sin_240_eq_neg_sqrt3_div_2_l67_67675

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67675


namespace equilateral_triangle_minimized_perimeter_ratio_l67_67757

theorem equilateral_triangle_minimized_perimeter_ratio :
  ∀ (A B C K M P : ℝ) (length : ℝ) (h_equilateral : A = B ∧ B = C ∧ C = A)
    (h_midpoint_K : K = (A + B) / 2) (h_ratio_M : BM / MC = 1 / 3)
    (h_perimeter_minimized : minimized_perimeter_triangle PKM),
  AP / PC = 2 / 3 := 
sorry

end equilateral_triangle_minimized_perimeter_ratio_l67_67757


namespace billy_ate_9_apples_on_wednesday_l67_67588

/-- Define the problem conditions -/
def apples (day : String) : Nat :=
  match day with
  | "Monday" => 2
  | "Tuesday" => 2 * apples "Monday"
  | "Friday" => apples "Monday" / 2
  | "Thursday" => 4 * apples "Friday"
  | _ => 0  -- For other days, we'll define later

/-- Define total apples eaten -/
def total_apples : Nat := 20

/-- Define sum of known apples excluding Wednesday -/
def known_sum : Nat :=
  apples "Monday" + apples "Tuesday" + apples "Friday" + apples "Thursday"

/-- Calculate apples eaten on Wednesday -/
def wednesday_apples : Nat := total_apples - known_sum

theorem billy_ate_9_apples_on_wednesday : wednesday_apples = 9 :=
  by
  sorry  -- Proof skipped

end billy_ate_9_apples_on_wednesday_l67_67588


namespace daisy_marked_point_count_l67_67043

noncomputable def princess_daisy_marked_points (Γ : Type) [Inhabited Γ] (circular_path : Γ → Prop)
  (S : Γ) (start_position : circular_path S) 
  (luigi_speed mario_speed : ℝ) (valid_speed : luigi_speed = 1 ∧ mario_speed = 3) 
  (t : ℝ) (time_bound : 0 ≤ t ∧ t ≤ 1) : ℕ :=
let luigi_position := t % 1,
    mario_position := luigi_speed * t % 1,
    midpoint := (luigi_position + mario_position) / 2 in
if midpoint ≠ S then 1 else 0

theorem daisy_marked_point_count (Γ : Type) [Inhabited Γ] (circular_path : Γ → Prop)
  (S : Γ) (start_position : circular_path S)
  (luigi_speed mario_speed : ℝ) (valid_speed : luigi_speed = 1 ∧ mario_speed = 3) 
  (t : ℝ) (time_bound : 0 ≤ t ∧ t ≤ 1) : 
  princess_daisy_marked_points Γ circular_path S start_position luigi_speed mario_speed valid_speed t time_bound = 1 :=
sorry

end daisy_marked_point_count_l67_67043


namespace triangle_side_sum_l67_67410

variable {V : Type*} [inner_product_space ℝ V]

def centroid (d e f : V) : V := (d + e + f) / 3

def squared_distance (u v : V) : ℝ := ∥u - v∥^2

theorem triangle_side_sum (d e f G : V) (hG : G = centroid d e f)
  (h_sum_squares : squared_distance G d + squared_distance G e + squared_distance G f = 84) :
  squared_distance d e + squared_distance d f + squared_distance e f = 252 :=
by
  unfold squared_distance
  unfold centroid at hG
  sorry

end triangle_side_sum_l67_67410


namespace set_intersection_example_l67_67810

theorem set_intersection_example :
  let M := {x : ℝ | -1 < x ∧ x < 1}
  let N := {x : ℝ | 0 ≤ x}
  {x : ℝ | -1 < x ∧ x < 1} ∩ {x : ℝ | 0 ≤ x} = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end set_intersection_example_l67_67810


namespace count_real_z10_l67_67971

theorem count_real_z10 (z : ℂ) (h : z ^ 30 = 1) : 
  (↑((({z : ℂ | z ^ 30 = 1}.to_finset).filter (λ x, x ^ 10 = 1)).card) + 
  ↑((({z : ℂ | z ^ 30 = 1}.to_finset).filter (λ x, x ^ 10 = -1)).card)) = 16 := 
sorry

end count_real_z10_l67_67971


namespace general_term_formula_l67_67332

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 1      := 3
| (n + 1):= 1 / 2 * sequence n + 1

theorem general_term_formula :
  ∀ n : ℕ, sequence n = (2^n + 1) / (2^(n-1)) := 
sorry

end general_term_formula_l67_67332


namespace prob1_prob2_prob3_l67_67721

-- (1)
theorem prob1 (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2) : ∃ d, ∀ n, a (3 * n) = a (3 * n + 3) - d :=
sorry

-- (2)
theorem prob2 (a : ℕ → ℤ) (d : ℤ) (h_a5 : a 5 = 6) (h_d : d ≠ 0) 
  (hg : ∃ q, a 3 / q = a 5 / q ∧ a (nat.succ (3 + nat.succ (5 + q))) = a 5 * q) :
  ∃ n₁ ∈ {6, 8, 11}, True :=
sorry

-- (3)
theorem prob3 (a : ℕ → ℝ) (q : ℝ) (h_q : q ≠ 1) 
  (harith : ∀ᶠ n in Filter.atTop, ∃ d : ℝ, ∃ m : ℕ, a (n + m) = a n + d * m) : q = -1 :=
sorry

end prob1_prob2_prob3_l67_67721


namespace sin_240_eq_neg_sqrt3_div_2_l67_67677

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67677


namespace octal_to_decimal_conversion_l67_67207

theorem octal_to_decimal_conversion : 
  let d8 := 8
  let f := fun (x: Nat) (y: Nat) => x * d8 ^ y
  7 * d8^0 + 6 * d8^1 + 3 * d8^2 = 247 := 
by
  let d8 := 8
  let f := fun (x: Nat) (y: Nat) => x * d8 ^ y
  sorry

end octal_to_decimal_conversion_l67_67207


namespace diamond_eq_l67_67539

noncomputable def diamond_op (a b : ℝ) (k : ℝ) : ℝ := sorry

theorem diamond_eq (x : ℝ) :
  let k := 2
  let a := 2023
  let b := 7
  let c := x
  (diamond_op a (diamond_op b c k) k = 150) ∧ 
  (∀ a b c, diamond_op a (diamond_op b c k) k = k * (diamond_op a b k) * c) ∧
  (∀ a, diamond_op a a k = k) →
  x = 150 / 2023 :=
sorry

end diamond_eq_l67_67539


namespace william_washing_time_l67_67161

theorem william_washing_time :
  let time_windows := 4
  let time_car_body := 7
  let time_tires := 4
  let time_waxing := 9
  let cars_washed := 2
  let suv_multiplier := 2
  let normal_car_time := time_windows + time_car_body + time_tires + time_waxing
  let total_normal_cars_time := cars_washed * normal_car_time
  let suv_time := suv_multiplier * normal_car_time
  let total_time := total_normal_cars_time + suv_time
  in total_time = 96 :=
by
  sorry

end william_washing_time_l67_67161


namespace sum_of_possible_values_of_c_l67_67718

noncomputable def g (c x : ℝ) : ℝ := c / (3 * x - 5)

theorem sum_of_possible_values_of_c (c : ℝ) (h : g c 3 = (g c)⁻¹ (c + 3)) : c = 20/3 ∨ c = -4 :=
by
  have g3 : g c 3 = c / 4 := by
    sorry
  have hinv : (g c)⁻¹ (c + 3) = (4 * (c + 3)) / (3 * (c + 3) + 5) := by
    sorry
  have h' : g c (c / 4) = c + 3 := by
    sorry
  have h'' : c / (3 * (c / 4) - 5) = c + 3 := by
    sorry
  have h''' : c / (3c / 4 - 5) = c + 3 := by
    sorry
  have quad_eq : 3 * c^2 - 8 * c - 60 = 0 := by
    sorry
  have roots : c = (20 / 3) ∨ c = -4 := by
    sorry
  show c = 20/3 ∨ c = -4, from roots

end sum_of_possible_values_of_c_l67_67718


namespace find_f_10_l67_67793

theorem find_f_10 :
  (∃ f : ℕ → ℕ, (∀ x : ℕ, f(3 * x + 1) = x^2 + 3 * x + 2) ∧ f(10) = 20) :=
begin
  sorry
end

end find_f_10_l67_67793


namespace numberOfZeros_l67_67267

noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem numberOfZeros :
  ∃ x ∈ Set.Ioo 1 (Real.exp Real.pi), g x = 0 ∧ ∀ y ∈ Set.Ioo 1 (Real.exp Real.pi), g y = 0 → y = x := 
sorry

end numberOfZeros_l67_67267


namespace smallest_n_for_constant_term_l67_67296

theorem smallest_n_for_constant_term :
  ∃ (n : ℕ), (n > 0) ∧ ((∃ (r : ℕ), 2 * n = 5 * r) ∧ (∀ (m : ℕ), m > 0 → (∃ (r' : ℕ), 2 * m = 5 * r') → n ≤ m)) ∧ n = 5 :=
by
  sorry

end smallest_n_for_constant_term_l67_67296


namespace partition_triangle_l67_67691

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l67_67691


namespace ratio_a_b_range_for_b_l67_67016

theorem ratio_a_b (A B C a b c : ℝ)
  (h : (cos B - 2 * cos A) / cos C = (2 * a - b) / c) : a / b = 2 :=
sorry

theorem range_for_b (A B C a b : ℝ) (h1 : A > π / 2)
  (h2 : c = 3) (h3 : (cos B - 2 * cos A) / cos C = (2 * a - b) / c) : 
  (sqrt 3 < b) ∧ (b < 3) :=
sorry

end ratio_a_b_range_for_b_l67_67016


namespace count_perfect_cubes_and_squares_in_range_l67_67369

theorem count_perfect_cubes_and_squares_in_range : 
  let a := 2^10 + 1,
      b := 2^20 + 1
  in (finset.range (11) ∩ finset.icc 4 (b ^ 6)).card = 7 :=
by
  -- Define the problem constants
  let a := 2^10 + 1
  let b := 2^20 + 1

  -- Add the proof here
  sorry

end count_perfect_cubes_and_squares_in_range_l67_67369


namespace remaining_leaves_after_summer_l67_67402

theorem remaining_leaves_after_summer :
  let branches := 100
  let twigs_per_branch := 150
  let total_twigs := branches * twigs_per_branch
  let percentage_twigs_sprouting_leaves := (0.20, 0.30, 0.50)
  let leaves_per_twig_spring := (3, 4, 5)
  let leaves_increase_summer := 2
  let leaves_eaten_percentage := 0.10
  let twig_counts_spring := (percentage_twigs_sprouting_leaves.1 * total_twigs,
                            percentage_twigs_sprouting_leaves.2 * total_twigs,
                            percentage_twigs_sprouting_leaves.3 * total_twigs)
  let leaves_spring := (leaves_per_twig_spring.1 * twig_counts_spring.1 +
                      leaves_per_twig_spring.2 * twig_counts_spring.2 +
                      leaves_per_twig_spring.3 * twig_counts_spring.3)
  let leaves_summer := ((leaves_per_twig_spring.1 + leaves_increase_summer) * twig_counts_spring.1 +
                      (leaves_per_twig_spring.2 + leaves_increase_summer) * twig_counts_spring.2 +
                      (leaves_per_twig_spring.3 + leaves_increase_summer) * twig_counts_spring.3)
  let leaves_eaten := leaves_eaten_percentage * leaves_summer
  let remaining_leaves := leaves_summer - leaves_eaten
  remaining_leaves = 85050 :=
by
  sorry

end remaining_leaves_after_summer_l67_67402


namespace lines_not_triangle_l67_67484

theorem lines_not_triangle (m : ℝ) : 
  (m = 1/3 ∨ m = 1 ∨ m = -1) ↔ ¬ ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ), 
  (x₁ + y₁ = 2) ∧ (m * x₂ + y₂ = 0) ∧ (x₃ - y₃ = 4) ∧ 
  ((x₁, y₁) ≠ (x₂, y₂)) ∧ ((x₂, y₂) ≠ (x₃, y₃)) ∧ ((x₃, y₃) ≠ (x₁, y₁)) :=
begin
  sorry
end

end lines_not_triangle_l67_67484


namespace bacteria_instantaneous_speed_and_intervals_l67_67509

noncomputable def b (t : ℝ) : ℝ := 105 + 104 * t - 103 * t^2

theorem bacteria_instantaneous_speed_and_intervals :
  let b' (t : ℝ) : ℝ := -2000 * t + 10000 in
  b' 5 = 0 ∧
  b' 10 = -10000 ∧
  (∀ t, 0 < t ∧ t < 5 → b' t > 0) ∧
  (∀ t, t > 5 → b' t < 0) :=
by
  sorry

end bacteria_instantaneous_speed_and_intervals_l67_67509


namespace sum_first_five_terms_arithmetic_l67_67882

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → α} (a_is_arithmetic : is_arithmetic_sequence a)
          (h1 : a 2 + a 6 = 10)
          (h2 : a 4 * a 8 = 45)

theorem sum_first_five_terms_arithmetic (a : ℕ → α) (a_is_arithmetic : is_arithmetic_sequence a)
  (h1 : a 2 + a 6 = 10) (h2 : a 4 * a 8 = 45) : 
  (∑ i in Finset.range 5, a i) = 20 := 
by
  sorry  

end sum_first_five_terms_arithmetic_l67_67882


namespace total_earnings_correct_l67_67236

-- Define the conditions as initial parameters

def ticket_price : ℕ := 3
def weekday_visitors_per_day : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300

def total_weekday_visitors : ℕ := 5 * weekday_visitors_per_day
def total_weekend_visitors : ℕ := saturday_visitors + sunday_visitors
def total_visitors : ℕ := total_weekday_visitors + total_weekend_visitors

def total_earnings := total_visitors * ticket_price

-- Prove that the total earnings of the amusement park in a week is $3000
theorem total_earnings_correct : total_earnings = 3000 :=
by
  sorry

end total_earnings_correct_l67_67236


namespace sin_240_eq_neg_sqrt3_div_2_l67_67610

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67610


namespace power_sub_inverse_eq_zero_l67_67762

theorem power_sub_inverse_eq_zero {x : ℂ} (h : x - 1/x = -complex.I * real.sqrt 6) : 
  x ^ 4374 - 1 / (x ^ 4374) = 0 := 
sorry

end power_sub_inverse_eq_zero_l67_67762


namespace max_dot_product_l67_67767

theorem max_dot_product (x : ℝ) (y : ℝ) (hx : y = x^2) :
  let OP := (x, y)
  let OA := (1, -1)
  max_value (OP.1 * OA.1 + OP.2 * OA.2) = 1/4 :=
sorry

end max_dot_product_l67_67767


namespace trigonometric_simplification_l67_67458

theorem trigonometric_simplification :
  (tan 10 * (π / 180) + tan 30 * (π / 180) + tan 50 * (π / 180) + tan 70 * (π / 180)) / cos (30 * (π / 180))
  = 4 * sin (40 * (π / 180)) + 1 :=
by
  sorry

end trigonometric_simplification_l67_67458


namespace cube_root_sum_is_integer_l67_67171

theorem cube_root_sum_is_integer :
  let a := (2 + (10 / 9) * Real.sqrt 3)^(1/3)
  let b := (2 - (10 / 9) * Real.sqrt 3)^(1/3)
  a + b = 2 := by
  sorry

end cube_root_sum_is_integer_l67_67171


namespace common_term_AP_GP_l67_67940

theorem common_term_AP_GP :
  ∀ (x : ℚ),
  let a1 := 2*x - 3 in
  let a2 := 5*x - 11 in
  let g1 := x + 1 in
  let g2 := 2*x + 3 in
  (∃ t : ℚ, (∃ n : ℕ, n ≥ 1 ∧ a1 + (n - 1) * (a2 - a1) = t) ∧
             (∃ m : ℕ, m ≥ 1 ∧ g1 * (g2 / g1)^(m - 1) = t)) → t = 37 / 3 := 
by
  sorry

end common_term_AP_GP_l67_67940


namespace bridge_length_correct_l67_67555

def walking_speed_kmph := 8 -- km/hr
def time_crossing_bridge_min := 15 -- minutes

def bridge_length_m : ℝ :=
  (walking_speed_kmph / 60) * time_crossing_bridge_min * 1000 -- in meters

theorem bridge_length_correct :
  bridge_length_m = 2000 :=
by
  sorry

end bridge_length_correct_l67_67555


namespace proper_subsets_intersection_even_l67_67039

variable {X : Type} [Fintype X] {𝒜 : Set (Set X)}

open Set

theorem proper_subsets_intersection_even (n : ℕ) (hn : 2 ≤ n) (hX : Fintype.card X = n)
  (h𝒜 : ∀ (S : Set X), S ⊂ univ → (∃ T ∈ 𝒜, T ∩ S ≠ ∅)) :
  𝒜 = 𝒫 (univ : Set X) - ∅ :=
sorry

end proper_subsets_intersection_even_l67_67039


namespace total_onions_grown_l67_67055

theorem total_onions_grown :
  let onions_per_day_nancy := 3
  let days_worked_nancy := 4
  let onions_per_day_dan := 4
  let days_worked_dan := 6
  let onions_per_day_mike := 5
  let days_worked_mike := 5
  let onions_per_day_sasha := 6
  let days_worked_sasha := 4
  let onions_per_day_becky := 2
  let days_worked_becky := 6

  let total_onions_nancy := onions_per_day_nancy * days_worked_nancy
  let total_onions_dan := onions_per_day_dan * days_worked_dan
  let total_onions_mike := onions_per_day_mike * days_worked_mike
  let total_onions_sasha := onions_per_day_sasha * days_worked_sasha
  let total_onions_becky := onions_per_day_becky * days_worked_becky

  let total_onions := total_onions_nancy + total_onions_dan + total_onions_mike + total_onions_sasha + total_onions_becky

  total_onions = 97 :=
by
  -- proof goes here
  sorry

end total_onions_grown_l67_67055


namespace charlie_golden_delicious_bags_l67_67724

theorem charlie_golden_delicious_bags :
  ∀ (total_bags fruit_bags macintosh_bags cortland_bags golden_delicious_bags : ℝ),
  total_bags = 0.67 →
  macintosh_bags = 0.17 →
  cortland_bags = 0.33 →
  total_bags = golden_delicious_bags + macintosh_bags + cortland_bags →
  golden_delicious_bags = 0.17 := by
  intros total_bags fruit_bags macintosh_bags cortland_bags golden_delicious_bags
  intros h_total h_macintosh h_cortland h_sum
  sorry

end charlie_golden_delicious_bags_l67_67724


namespace relatively_prime_number_exists_l67_67779

def gcd (a b : ℕ) : ℕ := a.gcd b

def is_relatively_prime_to_all (n : ℕ) (lst : List ℕ) : Prop :=
  ∀ m ∈ lst, m ≠ n → gcd n m = 1

def given_numbers : List ℕ := [20172017, 20172018, 20172019, 20172020, 20172021]

theorem relatively_prime_number_exists :
  ∃ n ∈ given_numbers, is_relatively_prime_to_all n given_numbers := 
begin
  use 20172019,
  split,
  { -- Show 20172019 is in the list
    simp },
  { -- Prove 20172019 is relatively prime to all other numbers in the list
    intros m h1 h2,
    -- Further proof goes here
    sorry
  }
end

end relatively_prime_number_exists_l67_67779


namespace equilateral_triangle_height_l67_67863

theorem equilateral_triangle_height (a b c d : ℝ)
  (habc : ∃ (A B C : ℝ → ℝ) (M : ℝ → ℝ), 
            ∀ (x y : ℝ), A y = B x ∧ B y = C x ∧ C y = A x ∧
            dist M (A 0) = b ∧ dist M (B 0) = c ∧ dist M (C 0) = d)
  (triangle_equilateral : ∀ (x y : ℝ), dist (A 0) (B 0) = a ∧
                                   dist (B 0) (C 0) = a ∧
                                   dist (C 0) (A 0) = a)
  : a * sqrt(3) / 2 = b + c + d :=
sorry

end equilateral_triangle_height_l67_67863


namespace simplify_division_l67_67459

theorem simplify_division :
  (2 * 10^12) / (4 * 10^5 - 1 * 10^4) = 5.1282 * 10^6 :=
by
  -- problem statement
  sorry

end simplify_division_l67_67459


namespace sin_240_eq_neg_sqrt3_div_2_l67_67635

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67635


namespace correct_calculation_for_A_l67_67159

theorem correct_calculation_for_A (x : ℝ) : (-2 * x) ^ 3 = -8 * x ^ 3 :=
by
  sorry

end correct_calculation_for_A_l67_67159


namespace average_chapters_per_book_l67_67437

theorem average_chapters_per_book (total_chapters total_books : ℝ) (h_total_chapters : total_chapters = 17.0) (h_total_books : total_books = 4.0) : (total_chapters / total_books) = 4.25 :=
by {
  rw [h_total_chapters, h_total_books],
  norm_num
}

end average_chapters_per_book_l67_67437


namespace distinct_fractions_5_l67_67909

theorem distinct_fractions_5 (x : ℝ) :
  ∃ (A : set (ℝ → ℝ)), A = {λ x, (2 * x + 2) / (x + 1), λ x, (2 * x + 1) / (1 * x + 2), 
  λ x, (1 * x + 2) / (2 * x + 1), λ x, (x + 1) / (2 * (x + 1)), λ x, 1} ∧ A.card = 5 :=
by {
  sorry
}

end distinct_fractions_5_l67_67909


namespace sin_240_deg_l67_67607

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l67_67607


namespace sum_valid_divisors_l67_67281

def isValid (x : ℕ) : Prop :=
  x ≥ 18 ∧ (420 / x) ≥ 12 ∧ 420 % x = 0

def sumOfValidXs : ℕ :=
  (Nat.divisors 420).filter isValid |>.sum

theorem sum_valid_divisors : sumOfValidXs = 134 := by
  sorry

end sum_valid_divisors_l67_67281


namespace coprime_with_others_l67_67790

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end coprime_with_others_l67_67790


namespace checkered_triangle_division_l67_67700

-- Define the conditions as assumptions
variable (T : Set ℕ) 
variable (sum_T : Nat) (h_sumT : sum_T = 63)
variable (part1 part2 part3 : Set ℕ)
variable (sum_part1 sum_part2 sum_part3 : Nat)
variable (h_part1 : sum part1 = 21)
variable (h_part2 : sum part2 = 21)
variable (h_part3 : sum part3 = 21)

-- Define the goal as a theorem
theorem checkered_triangle_division : 
  (∃ part1 part2 part3 : Set ℕ, sum part1 = 21 ∧ sum part2 = 21 ∧ sum part3 = 21 ∧ Disjoint part1 (part2 ∪ part3) ∧ Disjoint part2 part3 ∧ T = part1 ∪ part2 ∪ part3) :=
sorry

end checkered_triangle_division_l67_67700


namespace identical_function_for_x_ge_0_l67_67510

theorem identical_function_for_x_ge_0 :
  (∀ (x : ℝ), x ≥ 0 → (sqrt x) ^ 2 = x) ∧
  (∀ (x : ℝ), x ≥ 0 → sqrt (x ^ 2) ≠ x ∨ x < 0) ∧
  (∀ (x : ℝ), x ≥ 0 → real.cbrt (x ^ 3) ≠ x ∨ x < 0) ∧
  (∀ (x : ℝ), x ≥ 0 → x ≠ 0 → (x^2 / x) ≠ x ∨ x < 0) :=
by sorry

end identical_function_for_x_ge_0_l67_67510


namespace a_minus_b_range_l67_67324

noncomputable def range_of_a_minus_b (a b : ℝ) : Set ℝ :=
  {x | -2 < a ∧ a < 1 ∧ 0 < b ∧ b < 4 ∧ x = a - b}

theorem a_minus_b_range (a b : ℝ) (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) :
  ∃ x, range_of_a_minus_b a b x ∧ (-6 < x ∧ x < 1) :=
by
  sorry

end a_minus_b_range_l67_67324


namespace find_c_l67_67353

variables {V : Type} [AddCommGroup V] [Module ℝ V]

def a : V := (1, -2)
def b : V := (2, -4)
def c : V := (-7, 14)

axiom a_parallel_b : a ∥ b

theorem find_c (h : 3 • a + 2 • b + c = 0) : c = (-7, 14) :=
by sorry

end find_c_l67_67353


namespace two_quadrilaterals_form_triangle_and_pentagon_two_quadrilaterals_form_triangle_quadrilateral_and_pentagon_l67_67274

theorem two_quadrilaterals_form_triangle_and_pentagon (A B C D E F G H : Point) (grid : set Point) 
  (h_ABC_quadrilateral : quadrilateral A B C D) 
  (h_EFGH_quadrilateral : quadrilateral E F G H) 
  (h_A_in_grid : A ∈ grid) (h_B_in_grid : B ∈ grid)
  (h_C_in_grid : C ∈ grid) (h_D_in_grid : D ∈ grid)
  (h_E_in_grid : E ∈ grid) (h_F_in_grid : F ∈ grid)
  (h_G_in_grid : G ∈ grid) (h_H_in_grid : H ∈ grid) :
  (∃ T : set Point, is_triangle T ∧ T ⊆ {A, B, C, D, E, F, G, H}) ∧
  (∃ P : set Point, is_pentagon P ∧ P ⊆ {A, B, C, D, E, F, G, H}) := 
sorry

theorem two_quadrilaterals_form_triangle_quadrilateral_and_pentagon (I J K L M N O P : Point) (grid : set Point) 
  (h_IJKL_quadrilateral : quadrilateral I J K L) 
  (h_MNOP_quadrilateral : quadrilateral M N O P) 
  (h_I_in_grid : I ∈ grid) (h_J_in_grid : J ∈ grid)
  (h_K_in_grid : K ∈ grid) (h_L_in_grid : L ∈ grid)
  (h_M_in_grid : M ∈ grid) (h_N_in_grid : N ∈ grid)
  (h_O_in_grid : O ∈ grid) (h_P_in_grid : P ∈ grid) :
  (∃ T : set Point, is_triangle T ∧ T ⊆ {I, J, K, L, M, N, O, P}) ∧
  (∃ Q : set Point, is_quadrilateral Q ∧ Q ⊆ {I, J, K, L, M, N, O, P}) ∧
  (∃ P : set Point, is_pentagon P ∧ P ⊆ {I, J, K, L, M, N, O, P}) := 
sorry

end two_quadrilaterals_form_triangle_and_pentagon_two_quadrilaterals_form_triangle_quadrilateral_and_pentagon_l67_67274


namespace maximum_knights_possible_l67_67206

/-- Define the type to represent counts. -/
inductive Count
| knight : Count -- A knight who always tells the truth
| liar : Count   -- A liar who always lies

/-- Define a function to represent the grid of counties.
    The index to the grid is a pair of (row, column) from 0 to 4. -/
def grid := (Fin 5) × (Fin 5) → Count

/-- Define the neighbor relation on the grid. -/
def neighbors (g : grid) (r c : Fin 5) : List (Count) :=
  match r, c with
  | ⟨0, _⟩, _     => [g (⟨r.val, sorry⟩ , ⟨c.val+1,% 5⟩), g (⟨r.val+1,% 5⟩, ⟨c.val,% 5⟩)]
  | _, ⟨0, _⟩     => [g (⟨r.val-1,% 5⟩, ⟨c.val,% 5⟩), g (⟨r.val,% 5⟩, ⟨c.val+1,% 5⟩)]
  | ⟨4, _⟩, _     => [g (⟨r.val-1,% 5⟩, ⟨c.val,% 5⟩), g (⟨r.val,% 5⟩, ⟨c.val-1,% 5⟩)]
  | _, ⟨4, _⟩     => [g (⟨r.val,% 5⟩, ⟨c.val-1,% 5⟩), g (⟨r.val+1,% 5⟩, ⟨c.val,% 5⟩)]
  | _             => [g (⟨r.val+1,% 5⟩, ⟨c.val,% 5⟩), g (⟨r.val-1,% 5⟩, ⟨c.val,% 5⟩),
                      g (⟨r.val,% 5⟩, ⟨c.val+1,% 5⟩), g (⟨r.val,% 5⟩, ⟨c.val-1,% 5⟩)]

/-- Define a function to count the knights in a list. -/
def count_knights (l : List Count) : Nat :=
  l.countp (λ c, c = Count.knight)

theorem maximum_knights_possible (g : grid) :
  (∀ r c, g (r, c) = Count.knight → count_knights (neighbors g r c) = 2 ∨ 
           g (r, c) = Count.liar → count_knights (neighbors g r c) ≠ 2) →
  (∃ (knights : Nat), knights = 8 ∧
                      knights = (List.join $ List.map (λ r, List.map (g r) $ Fin.finRange 5) $ Fin.finRange 5).countp (λ x, x = Count.knight)) :=
sorry

end maximum_knights_possible_l67_67206


namespace arctan_sum_identity_l67_67862

/-!
# Triangle identity proof

Given a triangle ABC with ∠C = 2π/3,
we want to prove that:
arctan(a / (b + c)) + arctan(b / (a + c)) = π/4
-/

def triangle_angle_C_is_2π_over_3 (A B C : Type*) [triangle A B C] : Prop :=
  angle B C A = 2 * Real.pi / 3

theorem arctan_sum_identity (A B C : Type*) [triangle A B C]
  (h : triangle_angle_C_is_2π_over_3 A B C) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = π / 4 :=
by sorry

end arctan_sum_identity_l67_67862


namespace hearty_buys_red_packages_l67_67363

-- Define the conditions
def packages_of_blue := 3
def beads_per_package := 40
def total_beads := 320

-- Calculate the number of blue beads
def blue_beads := packages_of_blue * beads_per_package

-- Calculate the number of red beads
def red_beads := total_beads - blue_beads

-- Prove that the number of red packages is 5
theorem hearty_buys_red_packages : (red_beads / beads_per_package) = 5 := by
  sorry

end hearty_buys_red_packages_l67_67363


namespace operation_value_l67_67895

theorem operation_value (p q : ℝ)
  (h1 : 1 * 2 = p * 1^2 + q + 1 = 869)
  (h2 : 2 * 3 = p * 2^3 + q + 1 = 883) :
  2 * 9 = p * 2^9 + q + 1 = 1891 :=
begin
  sorry
end

end operation_value_l67_67895


namespace roots_of_star_eq_zero_l67_67720

def star (a b : ℝ) : ℝ := a * b^2 - a * b - 1

theorem roots_of_star_eq_zero : 
  ∀ x : ℝ, star 1 x = (x^2 - x - 1) →
  (∃ a b : ℝ, (a ≠ b) ∧ (a^2 - a - 1 = 0) ∧ (b^2 - b - 1 = 0)) :=
by
  intros x h
  rw [star, h]
  sorry

end roots_of_star_eq_zero_l67_67720


namespace min_value_expr_l67_67891

open Real

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (let a := (x + 1/y)^2 in
   let b := (y + 1/x)^2 in
   a * (a - 5) + b * (b - 5)) ≥ -12.5 :=
sorry

end min_value_expr_l67_67891


namespace count_valid_pairs_l67_67570

theorem count_valid_pairs : 
  ∃ (n : ℕ), 
  (∀ (a b : ℕ), (b > a) → 
    (a * b = 3 * (a-4) * (b-4) + 12) → 
    (a, b) ∈ (λ (t : ℕ × ℕ), (t.1 > 0) ∧ (t.2 > 0)) 
  ) ∧ n = 3 :=
begin
  sorry
end

end count_valid_pairs_l67_67570


namespace num_real_z10_l67_67966

theorem num_real_z10 (z : ℂ) (h : z^30 = 1) : (∃ n : ℕ, z = exp (2 * π * I * n / 30)) → ∃ n, z^10 ∈ ℝ :=
by sorry -- Here, we need to show that there are exactly 20 such complex numbers.

end num_real_z10_l67_67966


namespace round_to_nearest_hundredth_l67_67067

theorem round_to_nearest_hundredth (x : ℝ) (hx : x = 24.7396) : Real.round (100 * x) / 100 = 24.74 :=
by
    sorry

end round_to_nearest_hundredth_l67_67067


namespace william_washing_time_l67_67162

theorem william_washing_time :
  let time_windows := 4
  let time_car_body := 7
  let time_tires := 4
  let time_waxing := 9
  let cars_washed := 2
  let suv_multiplier := 2
  let normal_car_time := time_windows + time_car_body + time_tires + time_waxing
  let total_normal_cars_time := cars_washed * normal_car_time
  let suv_time := suv_multiplier * normal_car_time
  let total_time := total_normal_cars_time + suv_time
  in total_time = 96 :=
by
  sorry

end william_washing_time_l67_67162


namespace sin_240_eq_neg_sqrt3_div_2_l67_67613

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67613


namespace rhombus_locus_of_rectangle_l67_67422

def rectangle (K : Type) [euclidean_space K] :=
  ∃ (A B C D : K), -- Points A, B, C, and D forming the vertices of the rectangle
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    dist A B = dist C D ∧ dist B C = dist D A ∧
    dist A C = dist B D -- Opposite sides are equal

def is_center {K : Type} [euclidean_space K] (A B C D O : K) :=
  dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D -- Center O is equidistant to all vertices

def is_locus {K : Type} [euclidean_space K] (A B C D O M : K) :=
  dist A M ≥ dist O M ∧ dist B M ≥ dist O M ∧ dist C M ≥ dist O M ∧ dist D M ≥ dist O M

def is_rhombus_locus {K : Type} [euclidean_space K] (A B C D O : K) :=
  ∃ M : K, is_locus A B C D O M

theorem rhombus_locus_of_rectangle {K : Type} [euclidean_space K] :
  ∀ (A B C D O : K), rectangle A B C D → is_center A B C D O →
    is_rhombus_locus A B C D O :=
sorry -- Proof to be filled in

end rhombus_locus_of_rectangle_l67_67422


namespace solution1_solution2_l67_67460

-- Define the first equation
def equation1 (x : ℝ) : Prop := 3 * x - 5 = 10

-- Define the second equation
def equation2 (x : ℝ) : Prop := 2 * x + 4 * (2 * x - 3) = 6 - 2 * (x + 1)

-- Prove the solutions
theorem solution1 : ∃ x : ℝ, equation1 x ∧ x = 5 :=
by
  use 5
  split
  · change 3 * 5 - 5 = 10
    norm_num
  · rfl

theorem solution2 : ∃ x : ℝ, equation2 x ∧ x = 4 / 3 :=
by
  use 4 / 3
  split
  · change 2 * (4 / 3) + 4 * (2 * (4 / 3) - 3) = 6 - 2 * (4 / 3 + 1)
    norm_num
  · rfl

end solution1_solution2_l67_67460


namespace smaller_rectangle_area_l67_67565

-- Define the lengths and widths of the rectangles
def bigRectangleLength : ℕ := 40
def bigRectangleWidth : ℕ := 20
def smallRectangleLength : ℕ := bigRectangleLength / 2
def smallRectangleWidth : ℕ := bigRectangleWidth / 2

-- Define the area of the rectangles
def area (length width : ℕ) : ℕ := length * width

-- Prove the area of the smaller rectangle
theorem smaller_rectangle_area : area smallRectangleLength smallRectangleWidth = 200 :=
by
  -- Skip the proof
  sorry

end smaller_rectangle_area_l67_67565


namespace f_2010_eq_neg_sin_x_l67_67419

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => λ x => Real.sin x
| (n + 1) => λ x => (f n)' x

theorem f_2010_eq_neg_sin_x : ∀ x, f 2010 x = - Real.sin x := 
by 
  sorry

end f_2010_eq_neg_sin_x_l67_67419


namespace presentations_required_periods_l67_67202

theorem presentations_required_periods :
  ∀ (num_students : ℕ) (students_per_group : ℕ) (group_presentation_time : ℕ) 
    (individual_presentation_time : ℕ) (indiv_presentation_questions : ℕ)
    (group_presentation_questions : ℕ) (period_length : ℕ),
    num_students = 32 →
    students_per_group = 4 →
    group_presentation_time = 12 →
    individual_presentation_time = 5 →
    indiv_presentation_questions = 3 →
    group_presentation_questions = 0 → -- assuming the 12 minutes includes Q&A
    period_length = 40 →
    (let individual_presentations := num_students - students_per_group in
     let total_individual_time := individual_presentations * (individual_presentation_time + indiv_presentation_questions) in
     let total_group_time := students_per_group * group_presentation_time in
     let total_time := total_individual_time + total_group_time in
     let periods_needed := (total_time + period_length - 1) / period_length in
     periods_needed = 7) :=
by
  intros num_students students_per_group group_presentation_time individual_presentation_time
         indiv_presentation_questions group_presentation_questions period_length
         h1 h2 h3 h4 h5 h6 h7
  let individual_presentations := num_students - students_per_group
  let total_individual_time := individual_presentations * (individual_presentation_time + indiv_presentation_questions)
  let total_group_time := students_per_group * group_presentation_time
  let total_time := total_individual_time + total_group_time
  let periods_needed := (total_time + period_length - 1) / period_length
  have : num_students = 32 := h1
  have : students_per_group = 4 := h2
  have : group_presentation_time = 12 := h3
  have : individual_presentation_time = 5 := h4
  have : indiv_presentation_questions = 3 := h5
  have : group_presentation_questions = 0 := h6
  have : period_length = 40 := h7
  have : individual_presentations = 28 := sorry
  have : total_individual_time = 224 := sorry
  have : total_group_time = 48 := sorry
  have : total_time = 272 := sorry
  have : periods_needed = 7 := sorry
  exact (Eq.refl 7)

end presentations_required_periods_l67_67202


namespace tangent_circumcircle_l67_67178

open EuclideanGeometry

variables {A B C P Q O A' S : Point}

-- Define conditions given in the problem
def conditions (hABC : Triangle A B C) (hP : P ∈ segment A B) (hQ : Q ∈ segment A C)
  (hPQ_parallel_BC : parallel PQ BC) (hBQ_CP_intersect_O : intersects BQ CP O)
  (hA'_symmetric_A : symmetric_with_respect_to A' A BC)
  (hAO'_intersection_S : intersects A' O ((circumcircle A P Q) S)) : Prop := sorry

-- Define the statement that needs to be proved
theorem tangent_circumcircle (hABC : Triangle A B C) (hP : P ∈ segment A B) (hQ : Q ∈ segment A C)
  (hPQ_parallel_BC : parallel PQ BC) (hBQ_CP_intersect_O : intersects BQ CP O)
  (hA'_symmetric_A : symmetric_with_respect_to A' A BC)
  (hAO'_intersection_S : intersects A' O ((circumcircle A P Q) S)) :
  tangential_circles (circumcircle B S C) (circumcircle A P Q) := 
sorry

end tangent_circumcircle_l67_67178


namespace circle_area_of_polar_eq_l67_67471

/-- The graph of the polar equation r = -2 * cos θ + 6 * sin θ is a circle.
    We aim to prove that the area of this circle is 10 * π. -/
theorem circle_area_of_polar_eq:
  (∀ θ : ℝ, let r := -2 * cos θ + 6 * sin θ,
            (∃ x y : ℝ, x = r * cos θ ∧ y = r * sin θ ∧ (x + 1)^2 + (y - 3)^2 = 10)) ->
  (let radius := real.sqrt 10 in π * radius^2 = 10 * π) :=
by
  sorry

end circle_area_of_polar_eq_l67_67471


namespace relatively_prime_example_l67_67788

theorem relatively_prime_example :
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd d c = 1 ∧ Nat.gcd e c = 1 :=
by
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  sorry

end relatively_prime_example_l67_67788


namespace all_faces_multiple_of_3_l67_67469

def polyhedron (P : Type) := 
  ∃ (coloring : P → bool)
    (edges : P → ℕ)
    (adjacent : P → P → Prop),
    (∀ F₁ F₂, adjacent F₁ F₂ → coloring F₁ ≠ coloring F₂)
  ∧ (∃ F_w, (∀ F, F ≠ F_w → edges F % 3 = 0) ∧ edges F_w % 3 ≠ 0)

theorem all_faces_multiple_of_3 (P : Type) (F_w : P) 
  (coloring : P → bool) (edges : P → ℕ) (adjacent : P → P → Prop):
  (∀ F₁ F₂, adjacent F₁ F₂ → coloring F₁ ≠ coloring F₂) →
  (∀ F, F ≠ F_w → edges F % 3 = 0) →
  edges F_w % 3 = 0 :=
sorry

end all_faces_multiple_of_3_l67_67469


namespace solve_triangle_problem_l67_67844

noncomputable def triangle_problem (A B C a b c : ℝ) (m n : ℝ × ℝ) :=
  c^2 = a^2 + b^2 - ab ∧
  cos C = 1 / 2 ∧
  C = π / 3 ∧
  tan A - tan B = sqrt 3 / 3 * (1 + tan A * tan B) ∧
  m = (sin A, 1) ∧ 
  n = (3, cos (2 * A))

theorem solve_triangle_problem :
  ∀ A B C a b c (m n : ℝ × ℝ),
    triangle_problem A B C a b c m n →
    B = π / 4 ∧ m.1 * n.1 + m.2 * n.2 = 17 / 8 :=
sorry

end solve_triangle_problem_l67_67844


namespace team_selection_l67_67212

theorem team_selection (boys girls : ℕ) (choose_boys choose_girls : ℕ) 
  (boy_count girl_count : ℕ) (h1 : boy_count = 10) (h2 : girl_count = 12) 
  (h3 : choose_boys = 5) (h4 : choose_girls = 3) :
    (Nat.choose boy_count choose_boys) * (Nat.choose girl_count choose_girls) = 55440 :=
by
  rw [h1, h2, h3, h4]
  sorry

end team_selection_l67_67212


namespace saleswoman_commission_percentage_l67_67741

noncomputable def commission_percentage (sale1 sale2 sale3 : ℝ) : ℝ :=
  let commission (amount : ℝ) : ℝ :=
    if amount ≤ 500 then 0.20 * amount
    else if amount ≤ 1000 then 0.20 * 500 + 0.50 * (amount - 500)
    else 0.20 * 500 + 0.50 * 500 + 0.30 * (amount - 1000)
  let total_commission := commission sale1 + commission sale2 + commission sale3
  let total_amount := sale1 + sale2 + sale3
  (total_commission / total_amount) * 100

theorem saleswoman_commission_percentage :
  commission_percentage 800 1500 2500 ≈ 32.29 :=
by
  sorry

end saleswoman_commission_percentage_l67_67741


namespace composite_solid_volume_l67_67228

noncomputable def volume_truncated_cone (R r h : ℝ) : ℝ :=
  (π * h / 3) * (R^2 + R * r + r^2)

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem composite_solid_volume :
  let R := 10
  let r := 5
  let h1 := 8
  let h2 := 4
  volume_truncated_cone R r h1 + volume_cylinder r h2 = 566.\overline{6} * π := 
by 
  sorry

end composite_solid_volume_l67_67228


namespace points_collinear_in_regular_hexagon_l67_67038

theorem points_collinear_in_regular_hexagon
  (A B C D E F P Q : Type)
  (s : ℝ)
  (hexagon : regular_hexagon A B C D E F s)
  (P_on_BD : on_diagonal P B D)
  (Q_on_DF : on_diagonal Q D F)
  (BP_eq_s : distance B P = s)
  (DQ_eq_s : distance D Q = s) :
  collinear C P Q :=
sorry

end points_collinear_in_regular_hexagon_l67_67038


namespace sin_240_deg_l67_67656

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67656


namespace mr_problem_l67_67095

-- Define point and grid
structure Point :=
  (x : ℕ)
  (y : ℕ)

def is_neighbor (p1 p2 : Point) : Prop :=
  (p1.x = p2.x ∧ |p1.y - p2.y| = 1) ∨ (p1.y = p2.y ∧ |p1.x - p2.x| = 1)

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt (real.of_nat (p1.x - p2.x) ^ 2 + real.of_nat (p1.y - p2.y) ^ 2)

def growing_path (p : list Point) : Prop :=
  ∀ i, i < p.length - 1 → distance (p.nth_le i (by linarith)) (p.nth_le (i + 1) (by linarith)) < 
    distance (p.nth_le (i + 1) (by linarith)) (p.nth_le (i + 2) (by linarith))

-- Maximum length of a growing path
def max_growing_path_length (points : list Point) : ℕ := sorry

-- Number of growing paths of maximum length
def num_max_growing_paths (points : list Point) : ℕ := sorry

-- Main theorem
theorem mr_problem : max_growing_path_length [list of points in 3x3 grid] = 3 ∧ 
                     num_max_growing_paths [list of points in 3x3 grid] = 12 ∧ 
                     (max_growing_path_length [list of points in 3x3 grid] * 
                      num_max_growing_paths [list of points in 3x3 grid]) = 36 := sorry

end mr_problem_l67_67095


namespace smallest_t_of_triangle_inequality_l67_67115

theorem smallest_t_of_triangle_inequality :
  ∃ (t : ℕ), 7 + t > 11.5 ∧ 7 + 11.5 > t ∧ 11.5 + t > 7 ∧ ∀ (t' : ℕ), (7 + t' > 11.5 ∧ 7 + 11.5 > t' ∧ 11.5 + t' > 7) → t ≤ t' :=
begin
  sorry
end

end smallest_t_of_triangle_inequality_l67_67115


namespace total_journey_time_l67_67851

def speed_of_river : ℝ := 2
def distance_upstream : ℝ := 32
def speed_in_still_water : ℝ := 6

def effective_speed_upstream : ℝ := speed_in_still_water - speed_of_river
def effective_speed_downstream : ℝ := speed_in_still_water + speed_of_river

def time_upstream : ℝ := distance_upstream / effective_speed_upstream
def time_downstream : ℝ := distance_upstream / effective_speed_downstream

def total_time : ℝ := time_upstream + time_downstream

theorem total_journey_time : total_time = 12 := by
  calc
    total_time = time_upstream + time_downstream      : rfl
            ...= (distance_upstream / effective_speed_upstream) 
             + (distance_upstream / effective_speed_downstream) : rfl
            ...= (32 / 4) + (32 / 8)                           : rfl
            ...= 8 + 4                                        : rfl
            ...= 12                                           : rfl

end total_journey_time_l67_67851


namespace correct_sequence_configuration_l67_67771

-- Define the sequence a_n and sum S_n
variable {a : ℕ → ℝ}
def S (n : ℕ) : ℝ := ∑ i in finset.range n, a i

-- Define the conditions in Lean 4
axiom h₁ : ∀ k > 2022, |S k| > |S (k + 1)|
axiom h₂ : ∀ n ≥ 2023, ∃ q, 0 < q < 1 ∧ a (n+1) = a n * q
axiom h₃ : ∃ d, ∀ m < 2022, a (m + 1) = a m + d

-- Prove that the correct configuration is that the first 2022 terms
-- form an arithmetic progression and the terms starting from
-- the 2022th term form a geometric progression (i.e., option C is correct)
theorem correct_sequence_configuration : 
  (∀ m < 2022, ∃ d, a (m + 1) = a m + d) ∧
  (∀ n ≥ 2023, ∃ q, 0 < q < 1 ∧ a (n+1) = a n * q) :=
sorry

end correct_sequence_configuration_l67_67771


namespace calculate_bankers_discount_l67_67932

noncomputable def bankers_discount (S r t BG : ℝ) : ℝ :=
  let TD := S / (1 + r)^t
  let BD := S * r * t
  BD

theorem calculate_bankers_discount :
  ∀ (S : ℝ), 
    (let r := 0.12 
     let t := 5 
     let BG := 150 
     let TD := S / (1 + r)^t 
     let BD := S * r * t 
     150 = BD - TD) →
    bankers_discount S 0.12 5 150 ≈ 1576.05 :=
by
  intro S h
  dsimp [bankers_discount]
  -- Skipping the proof part
  sorry

end calculate_bankers_discount_l67_67932


namespace min_roots_in_interval_l67_67551

theorem min_roots_in_interval :
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f(2 + x) = f(2 - x)) → (∀ x : ℝ, f(9 + x) = f(9 - x)) → 
  f 0 = 0 → ∃ n : ℕ, n ≥ 250 ∧ ∀ m ≤ 250, ∃ x : ℝ, -1000 ≤ x ∧ x ≤ 1000 ∧ f x = 0 :=
by
  intros f h1 h2 h3
  sorry

end min_roots_in_interval_l67_67551


namespace conic_section_is_hyperbola_l67_67682

theorem conic_section_is_hyperbola : 
  ∀ (x y : ℝ), x^2 + 2 * x - 8 * y^2 = 0 → (∃ a b h k : ℝ, (x + 1)^2 / a^2 - (y - 0)^2 / b^2 = 1) := 
by 
  intros x y h_eq;
  sorry

end conic_section_is_hyperbola_l67_67682


namespace intersection_A_B_l67_67322

-- Definitions
def A := {0, 1, 2, 3, 4, 5}
def B := {x : ℕ | -1 < x ∧ x < 5}

-- Proof statement
theorem intersection_A_B : A ∩ B = {0, 1, 2, 3, 4} := 
by sorry

end intersection_A_B_l67_67322


namespace fans_attended_show_l67_67479

-- Definitions from the conditions
def total_seats : ℕ := 60000
def sold_percentage : ℝ := 0.75
def fans_stayed_home : ℕ := 5000

-- The proof statement
theorem fans_attended_show :
  let sold_seats := sold_percentage * total_seats
  let fans_attended := sold_seats - fans_stayed_home
  fans_attended = 40000 :=
by
  -- Auto-generated proof placeholder.
  sorry

end fans_attended_show_l67_67479


namespace dihedral_angle_SC_l67_67316

noncomputable def α := by sorry
noncomputable def β := by sorry
noncomputable def θ := by sorry
noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan x

theorem dihedral_angle_SC (α β θ : ℝ) (h1 : θ < α) (h2 : α < π / 2)
  (h3 : 0 < β) (h4 : β < π / 2) (h5 : ∠ ASB = π / 2) :
  θ = π - Real.arccos (cot α * cot β) := by
  sorry

end dihedral_angle_SC_l67_67316


namespace binary_addition_l67_67136

theorem binary_addition (a b : ℕ) :
  (a = (2^0 + 2^2 + 2^4 + 2^6)) → (b = (2^0 + 2^3 + 2^6)) →
  (a + b = 158) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end binary_addition_l67_67136


namespace fraction_sum_exists_and_not_coprime_l67_67921

theorem fraction_sum_exists_and_not_coprime (n : ℕ) (hn : n > 5) :
  (∃ (x : Fin n → ℕ), (∑ i, (1 : ℚ) / x i = 1997 / 1998) ∧ ∃ i j, i ≠ j ∧ gcd (x i) (x j) > 1) :=
by
  sorry

end fraction_sum_exists_and_not_coprime_l67_67921


namespace sin_240_eq_neg_sqrt3_div_2_l67_67612

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67612


namespace distinct_keychain_arrangements_l67_67009

theorem distinct_keychain_arrangements : 
  ∃ arrangements : ℕ, 
    (arrangements = 6) ∧
    (∀ keys : Finset ℕ, keys.card = 6) ∧
    (∀ HK CK : ℕ, HK ≠ CK) ∧
    (∀ (HK CK OK : ℕ) (arr arrangement : Finset ℕ), 
      ((HK ∈ arr ∧ HK ∈ arrangement) ∧
       (CK ∈ arr ∧ CK ∈ arrangement) ∧
       (OK ∈ arr ∧ OK ∈ arrangement) ∧
       (HK = CK + 1 ∨ HK = CK - 1) ∧
       (HK = OK + 3 ∨ HK = OK - 3) ∧
       -- Considering rotations and reflections as same
       (arrangements = ((keys.erase HK).erase CK).erase OK.card))) :=
sorry

end distinct_keychain_arrangements_l67_67009


namespace obtuse_triangle_third_vertex_l67_67500

theorem obtuse_triangle_third_vertex
  (a b : ℝ × ℝ)
  (ha : a = (4, 3))
  (hb : b = (0, 0))
  (area : ℝ)
  (harea : area = 24) :
  ∃ c : ℝ × ℝ, c.2 = 0 ∧ c.1 < 0 ∧ let base := (b.1 - c.1).abs in (1/2) * base * a.2 = area :=
begin
  use (-16, 0),
  split,
  { refl },
  split,
  { linarith },
  {
    intro base,
    calc
      (1 / 2) * base * a.2 = (1 / 2) * (0 - (-16)).abs * 3 : by simp [ha]
                      ... = (1 / 2) * 16 * 3 : by simp
                      ... = 24 : by norm_num,
    sorry
  }
end

end obtuse_triangle_third_vertex_l67_67500


namespace race_length_l67_67450

theorem race_length (members : ℕ) (member_distance : ℕ) (ralph_multiplier : ℕ) 
    (h1 : members = 4) (h2 : member_distance = 3) (h3 : ralph_multiplier = 2) : 
    members * member_distance + ralph_multiplier * member_distance = 18 :=
by
  -- Start the proof with sorry to denote missing steps.
  sorry

end race_length_l67_67450


namespace inequality_solution_set_l67_67116

theorem inequality_solution_set (x : ℝ) : |x - 5| + |x + 3| ≤ 10 ↔ -4 ≤ x ∧ x ≤ 6 :=
by
  sorry

end inequality_solution_set_l67_67116


namespace hibiscus_flower_ratio_l67_67433

theorem hibiscus_flower_ratio (x : ℕ) 
  (h1 : 2 + x + 4 * x = 22) : x / 2 = 2 := 
sorry

end hibiscus_flower_ratio_l67_67433


namespace part1_part2_l67_67308

-- Part (1)
theorem part1 (x y : ℝ) (hx : x = 2 + real.sqrt 3) (hy : y = 2 - real.sqrt 3) : x * y = 1 :=
by
  sorry

-- Part (2)
theorem part2 (x y : ℝ) (hx : x = 2 + real.sqrt 3) (hy : y = 2 - real.sqrt 3) : x^3 * y + x^2 = 14 + 8 * real.sqrt 3 :=
by
  sorry

end part1_part2_l67_67308


namespace limit_log_limit_exp_l67_67521

-- Define the conditions
variable (x : ℝ)
variable (hx : x > 0)

-- Define the first theorem
theorem limit_log (x : ℝ) (hx : x > 0) : 
  tendsto (fun n : ℕ => n * log (1 + x / n)) atTop (𝓝 x) := sorry

-- Define the second theorem
theorem limit_exp (x : ℝ) (hx : x > 0) : 
  tendsto (fun n : ℕ => (1 + x / n)^n) atTop (𝓝 (exp x)) := sorry

end limit_log_limit_exp_l67_67521


namespace money_digit_sum_equivalence_l67_67491

theorem money_digit_sum_equivalence :
  ∃ (k : ℕ) (n : ℕ) (pounds shillings pence : ℕ), 
  (1 ≤ k ∧ k ≤ 9) ∧
  (sum_of_digits pounds = k * (n + 2)) ∧
  (sum_of_digits (pounds * 240 + shillings * 12 + pence) = sum_of_digits pounds + sum_of_digits shillings + sum_of_digits pence) ∧
  (k = 4 → (pounds = 44444) ∧ (shillings = 4) ∧ (pence = 4) ∧ (pounds * 240 + shillings * 12 + pence = 10666612)) :=
by sorry

end money_digit_sum_equivalence_l67_67491


namespace evaluate_expression_l67_67284

-- Define our variables and their constraints
variables {a b c : ℝ}

theorem evaluate_expression (h : a ≠ b) : 
  (c * (a ^ (-6) - b ^ (-6))) / (a ^ (-3) - b ^ (-3)) = c * (a ^ (-6) + a ^ (-3) * b ^ (-3) + b ^ (-6)) := 
by sorry

end evaluate_expression_l67_67284


namespace tens_digit_of_8_pow_306_l67_67504

theorem tens_digit_of_8_pow_306 : ∀ n,  n % 6 = 0 -> (∃ m, 8 ^ n % 100 = m ∧ m / 10 % 10 = 6) :=
by
  intro n hn
  -- The corresponding exponent in the cycle of last two digits of 8^k in 68, 44, 52, 16, 28, 24
  have hcycle : 8^6 % 100 = 64 := by -- The sixth power of 8 mod cycle length (6)
    norm_num [pow_succ]
  have hmod : (306 % 6 = 0) := by -- This is given as the precursor condition
    rfl
  use (8 ^ 12 % 100); 
  split;
  apply hcycle;
  sorry

end tens_digit_of_8_pow_306_l67_67504


namespace rectangular_box_unique_solution_l67_67218

theorem rectangular_box_unique_solution :
  ∃ (a b c : ℕ), even a ∧ even b ∧ even c ∧ 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ 
  (2 * (a * b + a * c + b * c) = 4 * (a + b + c)) ∧ (a, b, c) = (2, 2, 2) :=
by
  sorry

end rectangular_box_unique_solution_l67_67218


namespace smallest_k_for_table_l67_67876

-- Define the problem's main proof goal
theorem smallest_k_for_table (n : ℕ) (h_pos : 0 < n) :
  ∃ k : ℕ, (∀ (coloring : fin (2 * n) → fin k → fin n),
    ∃ (r1 r2 c1 c2 : fin (2 * n)),
      r1 ≠ r2 ∧ c1 ≠ c2 ∧ coloring r1 c1 = coloring r1 c2 ∧ coloring r2 c1 = coloring r2 c2) ∧
  ∀ k' : ℕ,
    (∀ (coloring : fin (2 * n) → fin k' → fin n),
      ∃ (r1 r2 c1 c2 : fin (2 * n)),
        r1 ≠ r2 ∧ c1 ≠ c2 ∧ coloring r1 c1 = coloring r1 c2 ∧ coloring r2 c1 = coloring r2 c2) →
    k' ≥ 2 * n ^ 2 - n + 1 :=
begin
  sorry
end

end smallest_k_for_table_l67_67876


namespace tourist_grouping_l67_67128

noncomputable def numberOfGroupings : ℕ := 762

theorem tourist_grouping:
  ∃ (A B C : Type) (t : Finset (A ⊕ B ⊕ C)),
    (A ∪ B ∪ C = {0, ..., 7}) ∧
    (∀ x ∈ A ∪ B ∪ C, x ∈ t) ∧
    (∃ g : A ∪ B ∪ C, g = ∅) ∧
    (∀ g₁ g₂ : A ∪ B ∪ C, (g₁ = A ∨ g₁ = B ∨ g₁ = C) → (g₁ ≠ ∅ → g₂ ≠ ∅)) ∧
    (∑ s in Finset.filter (λ x, x ≠ ∅) {A, B, C}, A.card + B.card) = numberOfGroupings :=
by
  sorry

end tourist_grouping_l67_67128


namespace lesser_number_is_14_l67_67117

theorem lesser_number_is_14 (x y : ℕ) (h₀ : x + y = 60) (h₁ : 4 * y - x = 10) : y = 14 :=
by 
  sorry

end lesser_number_is_14_l67_67117


namespace probability_no_defective_pens_l67_67848

theorem probability_no_defective_pens
  (total_pens : ℕ) (defective_pens : ℕ) (non_defective_pens : ℕ) (prob_first_non_defective : ℚ) (prob_second_non_defective : ℚ) :
  total_pens = 12 →
  defective_pens = 4 →
  non_defective_pens = total_pens - defective_pens →
  prob_first_non_defective = non_defective_pens / total_pens →
  prob_second_non_defective = (non_defective_pens - 1) / (total_pens - 1) →
  prob_first_non_defective * prob_second_non_defective = 14 / 33 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end probability_no_defective_pens_l67_67848


namespace boats_seating_problem_l67_67227

theorem boats_seating_problem 
  (total_boats : ℕ) (total_people : ℕ) 
  (big_boat_seats : ℕ) (small_boat_seats : ℕ) 
  (b s : ℕ) 
  (h1 : total_boats = 12) 
  (h2 : total_people = 58) 
  (h3 : big_boat_seats = 6) 
  (h4 : small_boat_seats = 4) 
  (h5 : b + s = 12) 
  (h6 : b * 6 + s * 4 = 58) 
  : b = 5 ∧ s = 7 :=
sorry

end boats_seating_problem_l67_67227


namespace relatively_prime_number_exists_l67_67778

def gcd (a b : ℕ) : ℕ := a.gcd b

def is_relatively_prime_to_all (n : ℕ) (lst : List ℕ) : Prop :=
  ∀ m ∈ lst, m ≠ n → gcd n m = 1

def given_numbers : List ℕ := [20172017, 20172018, 20172019, 20172020, 20172021]

theorem relatively_prime_number_exists :
  ∃ n ∈ given_numbers, is_relatively_prime_to_all n given_numbers := 
begin
  use 20172019,
  split,
  { -- Show 20172019 is in the list
    simp },
  { -- Prove 20172019 is relatively prime to all other numbers in the list
    intros m h1 h2,
    -- Further proof goes here
    sorry
  }
end

end relatively_prime_number_exists_l67_67778


namespace area_square_AD_l67_67134

-- Define the two right triangles ABC and ACD
variables (A B C D : Type) 
variables [point : A → A → Type]
variables [rtriangle1 : right_triangle (point A B) (point B C) (point A C)]
variables [rtriangle2 : right_triangle (point A C) (point C D) (point A D)]

-- Define the side lengths
variables (BC AC CD AD : ℝ)

-- Given conditions
axiom area_square_BC : BC * BC = 25
axiom area_square_AC : AC * AC = 36
axiom area_square_CD : CD * CD = 16
axiom common_side : AC = AC

-- Prove the area of the square on side AD
theorem area_square_AD : AD * AD = 52 := sorry

end area_square_AD_l67_67134


namespace problem_statement_l67_67920

variables {n : ℕ} {r : ℝ}

-- Define the side lengths based on given conditions
def a_n (r : ℝ) (n : ℕ) : ℝ := 2 * r * Real.sin (Real.pi / n)
def A_n (r : ℝ) (n : ℕ) : ℝ := 2 * r * Real.tan (Real.pi / n)

-- The circumscribed 2n-sided polygon
def A_2n (r : ℝ) (n : ℕ) : ℝ := 2 * r * Real.tan (Real.pi / (2 * n))

theorem problem_statement (r : ℝ) (n : ℕ) :
  (1 / A_2n r n) = (1 / A_n r n) + (1 / a_n r n) :=
sorry

end problem_statement_l67_67920


namespace baseball_card_decrease_l67_67538

theorem baseball_card_decrease :
  let initial_value := 100 in
  let first_year_value := initial_value * (1 - 0.60) in
  let second_year_value := first_year_value * (1 - 0.30) in
  let third_year_value := second_year_value * (1 - 0.20) in
  let fourth_year_value := third_year_value * (1 - 0.10) in
  let total_decrease := initial_value - fourth_year_value in
  let total_percent_decrease := (total_decrease / initial_value) * 100 in
  total_percent_decrease = 79.84 :=
by
  sorry

end baseball_card_decrease_l67_67538


namespace sin_240_eq_neg_sqrt3_div_2_l67_67634

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67634


namespace car_total_distance_l67_67155

-- Define the arithmetic sequence (s_n) where a = 40 and d = -10
def car_travel (n : ℕ) : ℕ := if n > 0 then 40 - 10 * (n - 1) else 0

-- Define the sum of the first 'k' terms of the arithmetic sequence
noncomputable def sum_car_travel (k : ℕ) : ℕ :=
  ∑ i in Finset.range k, car_travel (i + 1)

-- Main theorem statement
theorem car_total_distance : sum_car_travel 4 = 100 :=
by
  sorry

end car_total_distance_l67_67155


namespace minimum_value_l67_67890

theorem minimum_value {p q r s t u v w : ℝ} 
  (h : {p, q, r, s, t, u, v, w} ⊆ {-8, -6, -4, -1, 3, 5, 7, 14}) 
  (distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ 
            q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧ 
            r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ 
            s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ 
            t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ 
            u ≠ v ∧ u ≠ w ∧ 
            v ≠ w) 
  (sum_elements : p + q + r + s + t + u + v + w = 20) : 
  3 * (p + q + r + s)^2 + (t + u + v + w)^2 = 300 := 
sorry

end minimum_value_l67_67890


namespace min_sum_a_b_l67_67329

theorem min_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hlog : Real.log10 a + Real.log10 b = Real.log10 (a + b)) :
  a + b ≥ 4 := 
by
  sorry

end min_sum_a_b_l67_67329


namespace maximized_total_profit_at_eight_l67_67051

def c (x : ℝ) := 100 + 13 * x

def p (x : ℝ) := 800 / (x + 2) - 3

def profit (x : ℝ) := x * (p x) - c x

theorem maximized_total_profit_at_eight : 
  ∃ x, x = 8 ∧ ∀ y, profit y ≤ profit x := 
by
  sorry

end maximized_total_profit_at_eight_l67_67051


namespace find_smallest_magnitude_w3z3_l67_67037

noncomputable def smallest_magnitude_w3z3 (w z : ℂ) (h1 : |w + z| = 2) (h2 : |w^2 + z^2| = 18) : ℝ :=
  |(w + z) * (w^2 - w * z + z^2)|

theorem find_smallest_magnitude_w3z3 :
  ∀ w z : ℂ, |w + z| = 2 → |w^2 + z^2| = 18 → smallest_magnitude_w3z3 w z = 50 :=
by
  intros
  sorry

end find_smallest_magnitude_w3z3_l67_67037


namespace temperature_storage_range_l67_67543

-- Define the conditions
def central_temperature : ℝ := 20
def variation : ℝ := 2

-- Define the temperature range
def temperature_range := Icc (central_temperature - variation) (central_temperature + variation)

-- THM: The temperature range for storing the drug is [18, 22].
theorem temperature_storage_range : temperature_range = Icc 18 22 := 
by 
  sorry

end temperature_storage_range_l67_67543


namespace relatively_prime_number_exists_l67_67782

theorem relatively_prime_number_exists :
  -- Given numbers
  (let a := 20172017 in
   let b := 20172018 in
   let c := 20172019 in
   let d := 20172020 in
   let e := 20172021 in
   -- Number c is relatively prime to all other given numbers
   nat.gcd c a = 1 ∧
   nat.gcd c b = 1 ∧
   nat.gcd c d = 1 ∧
   nat.gcd c e = 1) :=
by {
  -- Proof omitted
  sorry
}

end relatively_prime_number_exists_l67_67782


namespace find_xy_plus_yz_plus_xz_l67_67042

noncomputable theory

variables {x y z : ℝ}

-- Given conditions as definitions
def condition1 : Prop := x > 0 ∧ y > 0 ∧ z > 0
def condition2 : Prop := x^2 + x*y + y^2 = 27
def condition3 : Prop := y^2 + y*z + z^2 = 16
def condition4 : Prop := z^2 + z*x + x^2 = 43

-- The theorem to prove
theorem find_xy_plus_yz_plus_xz (hx : condition1) (h1 : condition2) (h2 : condition3) (h3 : condition4) :
  x*y + y*z + z*x = 24 :=
sorry

end find_xy_plus_yz_plus_xz_l67_67042


namespace radishes_per_row_l67_67209

theorem radishes_per_row 
  (bean_seedlings : ℕ) (beans_per_row : ℕ) 
  (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (rows_per_bed : ℕ) (plant_beds : ℕ)
  (h1 : bean_seedlings = 64) (h2 : beans_per_row = 8)
  (h3 : pumpkin_seeds = 84) (h4 : pumpkins_per_row = 7)
  (h5 : radishes = 48) (h6 : rows_per_bed = 2) (h7 : plant_beds = 14) : 
  (radishes / ((plant_beds * rows_per_bed) - (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row))) = 6 := 
by sorry

end radishes_per_row_l67_67209


namespace compare_inequalities_l67_67748

theorem compare_inequalities (a b c π : ℝ) (h1 : a > π) (h2 : π > b) (h3 : b > 1) (h4 : 1 > c) (h5 : c > 0) 
  (x := a^(1 / π)) (y := Real.log b / Real.log π) (z := Real.log π / Real.log c) : x > y ∧ y > z := 
sorry

end compare_inequalities_l67_67748


namespace array_no_duplicate_rows_l67_67927

theorem array_no_duplicate_rows {A : matrix (fin n) (fin n) α} (h : ∀ i j : fin n, i ≠ j → A i ≠ A j) : 
  ∃ c : fin n, ∀ i j : fin n, i ≠ j → (A i).erase c ≠ (A j).erase c :=
by
  sorry

end array_no_duplicate_rows_l67_67927


namespace num_real_solutions_eq_2_l67_67801

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 4 * x + 6 else (Real.log x / Real.log 8).abs

theorem num_real_solutions_eq_2 : 
  (∃ a b c : ℝ, a = -2 ∧ b = 64 ∧ c = 1 / 64 ∧ 
    (f a = 2) ∧ (f b = 2) ∧ (f c = 2)) ∧
    (∀ x : ℝ, f x = 2 → x = -2 ∨ x = 64 ∨ x = 1 / 64) :=
by
  sorry

end num_real_solutions_eq_2_l67_67801


namespace sin_240_eq_neg_sqrt3_div_2_l67_67611

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67611


namespace delores_money_left_l67_67266

theorem delores_money_left : 
  ∀ (initial_amount computer_price printer_price : ℕ), 
    initial_amount = 450 → 
    computer_price = 400 → 
    printer_price = 40 → 
    initial_amount - (computer_price + printer_price) = 10 :=
by
  intros initial_amount computer_price printer_price h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end delores_money_left_l67_67266


namespace tom_catches_jerry_after_time_l67_67495

-- Let's define the lengths of the loops
def smaller_loop_length : ℝ := m
def larger_loop_length : ℝ := (4 / 3) * m

-- Defining the speeds
def tom_speed : ℝ := smaller_loop_length / 15
def jerry_speed : ℝ := larger_loop_length / 20

-- Calculating when Tom catches Jerry 
theorem tom_catches_jerry_after_time (m : ℝ) : 
  let relative_speed := tom_speed - jerry_speed,
      initial_separation := larger_loop_length in
  (initial_separation / relative_speed) = 80 :=
by 
  -- so we know the lengths of the loops
  let smaller_loop_length := m
  let larger_loop_length := (4 / 3) * smaller_loop_length
  -- defining their speeds
  have tom_speed_def : tom_speed = smaller_loop_length / 15 := rfl
  have jerry_speed_def : jerry_speed = larger_loop_length / 20 := rfl
  -- relative speed
  have relative_speed_def : relative_speed = (smaller_loop_length / 15) - (larger_loop_length / 20) := by sorry
  -- initial separation
  have initial_separation_def : initial_separation = larger_loop_length := rfl
  -- the time calculation based on initial separation and relative speed
  have time_calc : (initial_separation / relative_speed) = 80 := by sorry
  exact time_calc

end tom_catches_jerry_after_time_l67_67495


namespace sum_of_recorded_products_l67_67131

theorem sum_of_recorded_products (n : ℕ) (hn : n = 25) : 
  ∑ xy in (splits n), xy = 300 :=
by {
  sorry -- the proof details go here
}

end sum_of_recorded_products_l67_67131


namespace correct_number_is_650_l67_67514

theorem correct_number_is_650 
  (n : ℕ) 
  (h : n - 152 = 346): 
  n + 152 = 650 :=
by
  sorry

end correct_number_is_650_l67_67514


namespace log_eq_satisfies_3_l67_67166

theorem log_eq_satisfies_3 (x : ℝ) :
  7.3092 * (log 9 x)^2 = log 3 x * log 3 ((sqrt (2 * x + 1)) - 1) ↔ x = 3 :=
by sorry

end log_eq_satisfies_3_l67_67166


namespace parabola_line_intersection_ratio_l67_67334

variable {p : ℝ} (h : p > 0)

def focus := (p / 2, 0) : ℝ × ℝ
def parabola (x y : ℝ) := y^2 = 2 * p * x
def line (x y : ℝ) := √3 * x - y - (√3 * p / 2) = 0

theorem parabola_line_intersection_ratio (h_intersects_in_first_and_fourth : ∀ x y, parabola p x y → line p x y → y ≠ 0) :
  let O := (0, 0)
  let F := focus p
  ∃ (A B : ℝ × ℝ), parabola p A.fst A.snd ∧ line p A.fst A.snd ∧ A.snd > 0 ∧
                   parabola p B.fst B.snd ∧ line p B.fst B.snd ∧ B.snd < 0 ∧
                   (1 - (p / 2, 0) - A)^2 / (F - B)^2 = 9 := sorry

end parabola_line_intersection_ratio_l67_67334


namespace minimum_time_for_xiang_qing_fried_eggs_l67_67165

-- Define the time taken for each individual step
def wash_scallions_time : ℕ := 1
def beat_eggs_time : ℕ := 1 / 2
def mix_egg_scallions_time : ℕ := 1
def wash_pan_time : ℕ := 1 / 2
def heat_pan_time : ℕ := 1 / 2
def heat_oil_time : ℕ := 1 / 2
def cook_dish_time : ℕ := 2

-- Define the total minimum time required
def minimum_time : ℕ := 5

-- The main theorem stating that the minimum time required is 5 minutes
theorem minimum_time_for_xiang_qing_fried_eggs :
  wash_scallions_time + beat_eggs_time + mix_egg_scallions_time + wash_pan_time + heat_pan_time + heat_oil_time + cook_dish_time = minimum_time := 
by sorry

end minimum_time_for_xiang_qing_fried_eggs_l67_67165


namespace find_value_of_a_l67_67832

theorem find_value_of_a
  (p q : ℝ)
  (h1 : 144^p = 10)
  (h2 : 1728^q = 5)
  (a : ℝ)
  (h3 : a = 12^(2*p - 3*q)) :
  a = 2 :=
sorry

end find_value_of_a_l67_67832


namespace solve_floor_trig_eq_l67_67415

-- Define the floor function
def floor (x : ℝ) : ℤ := by 
  sorry

-- Define the condition and theorem
theorem solve_floor_trig_eq (x : ℝ) (n : ℤ) : 
  floor (Real.sin x + Real.cos x) = 1 ↔ (∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (2 * Real.pi * n + Real.pi / 2)) := 
  by 
  sorry

end solve_floor_trig_eq_l67_67415


namespace exists_sum_of_divisibles_l67_67065

theorem exists_sum_of_divisibles : ∃ (a b: ℕ), a + b = 316 ∧ (13 ∣ a) ∧ (11 ∣ b) :=
by
  existsi 52
  existsi 264
  sorry

end exists_sum_of_divisibles_l67_67065


namespace A_pow_101_l67_67406

def A : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 1],
  ![1, 0, 0],
  ![0, 1, 0]
]

theorem A_pow_101 :
  A ^ 101 = ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] := by
  sorry

end A_pow_101_l67_67406


namespace two_8_sided_dice_probability_l67_67991

theorem two_8_sided_dice_probability :
  let outcomes_total : ℕ := 8 * 8, 
      blue_die_even_count : ℕ := 4, 
      yellow_die_prime_count : ℕ := 4,
      successful_outcomes : ℕ := blue_die_even_count * yellow_die_prime_count,
      probability : ℚ := successful_outcomes / outcomes_total
  in probability = 1 / 4 := 
by
  sorry

end two_8_sided_dice_probability_l67_67991


namespace evaluate_magnitude_l67_67282

def complex_magnitude (z : ℂ) : ℝ :=
  complex.abs z

theorem evaluate_magnitude :
  complex_magnitude (complex.mk (2/3) (-5/4)) = 17 / 12 :=
by
  sorry

end evaluate_magnitude_l67_67282


namespace probability_no_defective_pens_l67_67847

theorem probability_no_defective_pens
  (total_pens : ℕ) (defective_pens : ℕ) (non_defective_pens : ℕ) (prob_first_non_defective : ℚ) (prob_second_non_defective : ℚ) :
  total_pens = 12 →
  defective_pens = 4 →
  non_defective_pens = total_pens - defective_pens →
  prob_first_non_defective = non_defective_pens / total_pens →
  prob_second_non_defective = (non_defective_pens - 1) / (total_pens - 1) →
  prob_first_non_defective * prob_second_non_defective = 14 / 33 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end probability_no_defective_pens_l67_67847


namespace milk_sold_in_ounces_l67_67438

theorem milk_sold_in_ounces:
  let monday_morning_milk : ℕ := 60 * 250 + 40 * 300 + 50 * 350 in
  let monday_evening_milk : ℕ := 0.5 * 100 * 400 + 0.25 * 100 * 500 + 0.25 * 100 * 450 in
  let tuesday_morning_milk : ℕ := 0.4 * 60 * 300 + 0.3 * 60 * 350 + 0.3 * 60 * 400 in
  let tuesday_evening_milk : ℕ := 0.25 * 200 * 450 + 0.35 * 200 * 500 + 0.4 * 200 * 550 in
  let total_monday_milk : ℕ := monday_morning_milk + monday_evening_milk in
  let total_tuesday_milk : ℕ := tuesday_morning_milk + tuesday_evening_milk in
  let total_milk_bought : ℕ := total_monday_milk + total_tuesday_milk in
  let remaining_milk : ℕ := 84000 in
  let milk_sold_ml : ℕ := total_milk_bought - remaining_milk in
  let milk_sold_oz : ℕ := milk_sold_ml / 30 in 
  milk_sold_oz = 4215 :=
begin
  sorry
end

end milk_sold_in_ounces_l67_67438


namespace sin_240_eq_neg_sqrt3_div_2_l67_67670

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67670


namespace sequence_inequality_l67_67956

theorem sequence_inequality (a : ℕ → ℕ) (h₁ : a 1 = 1007) (h₂ : ∀ i, a (i + 1) ≥ a i + 1) :
  1 / 2016 > ∑ i in finset.range 2016, 1 / ((a (i + 1))^2 + (a (i + 2))^2) :=
by sorry

end sequence_inequality_l67_67956


namespace kevin_food_expense_l67_67068

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end kevin_food_expense_l67_67068


namespace num_colorings_correct_l67_67192

open Finset

def num_colorings_6_points : ℕ :=
  let total_partitions := 
    1 + -- All points in one group
    6 + -- 5-1 pattern
    15 + -- 4-2 pattern
    15 + -- 4-1-1 pattern
    10 + -- 3-3 pattern
    60 + -- 3-2-1 pattern
    20 + -- 3-1-1-1 pattern
    15 + -- 2-2-2 pattern
    45 + -- 2-2-1-1 pattern
    15 + -- 2-1-1-1-1 pattern
    1 -- 1-1-1-1-1-1 pattern
  total_partitions

theorem num_colorings_correct : num_colorings_6_points = 203 :=
by
  sorry

end num_colorings_correct_l67_67192


namespace calc_value_l67_67683

noncomputable def sequence (n : ℕ) : ℤ :=
  match n with
  | 0       => 2
  | (n + 1) => 1 - (sequence n)

def sum_first_n_terms (n : ℕ) : ℤ :=
  (List.range (n + 1)).sum (λ i, sequence i)

theorem calc_value : sum_first_n_terms 2006 - 2 * sum_first_n_terms 2007 + sum_first_n_terms 2008 = -3 :=
  sorry

end calc_value_l67_67683


namespace sine_angle_between_lateral_edge_and_base_l67_67498

-- Define the problem-related parameters and conditions
variables (a b : ℝ)
-- a and b are the side lengths of the bases of the frustum

-- Define the main statement
theorem sine_angle_between_lateral_edge_and_base (a b : ℝ) :
  ((sin (atan2 b (2 * (a + b) / sqrt (3 * a * b)))) = 1 / 3) :=
begin
  sorry
end

end sine_angle_between_lateral_edge_and_base_l67_67498


namespace checkered_triangle_division_l67_67692

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l67_67692


namespace round_robin_teams_l67_67904

theorem round_robin_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 := 
by
  sorry

end round_robin_teams_l67_67904


namespace fraction_multiplication_squared_l67_67591

theorem fraction_multiplication_squared :
  (\left(\frac{4.0}{5.0}\right)^2 * \left(\frac{3.0}{7.0}\right)^2 * \left(\frac{2.0}{3.0}\right)^2 = (\frac{64.0}{1225.0})) :=
by
  sorry

end fraction_multiplication_squared_l67_67591


namespace first_method_of_exhaustion_l67_67854

-- Define the names
inductive Names where
  | ZuChongzhi
  | LiuHui
  | ZhangHeng
  | YangHui
  deriving DecidableEq

-- Statement of the problem
def method_of_exhaustion_author : Names :=
  Names.LiuHui

-- Main theorem to state the result
theorem first_method_of_exhaustion : method_of_exhaustion_author = Names.LiuHui :=
by 
  sorry

end first_method_of_exhaustion_l67_67854


namespace side_length_of_square_in_right_triangle_l67_67066

theorem side_length_of_square_in_right_triangle (P Q R : Type) [metric_space P]
  [inhabited P] (hpqrt : PQR) (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12)
  (h_tri : ∃ c, c = sqrt (a^2 + b^2)) :
  ∃ s : ℝ, s = 45/8 := by
sorry

end side_length_of_square_in_right_triangle_l67_67066


namespace sin_240_eq_neg_sqrt3_div_2_l67_67629

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67629


namespace pascal_fifth_number_l67_67999

def binom (n k : Nat) : Nat := Nat.choose n k

theorem pascal_fifth_number (n r : Nat) (h1 : n = 50) (h2 : r = 4) : binom n r = 230150 := by
  sorry

end pascal_fifth_number_l67_67999


namespace relatively_prime_solutions_l67_67290

theorem relatively_prime_solutions  (x y : ℤ) (h_rel_prime : gcd x y = 1) : 
  2 * (x^3 - x) = 5 * (y^3 - y) ↔ 
  (x = 0 ∧ (y = 1 ∨ y = -1)) ∨ 
  (x = 1 ∧ y = 0) ∨
  (x = -1 ∧ y = 0) ∨
  (x = 4 ∧ (y = 3 ∨ y = -3)) ∨ 
  (x = -4 ∧ (y = -3 ∨ y = 3)) ∨
  (x = 1 ∧ y = -1) ∨
  (x = -1 ∧ y = 1) ∨
  (x = 0 ∧ y = 0) :=
by sorry

end relatively_prime_solutions_l67_67290


namespace sin_240_eq_neg_sqrt3_div_2_l67_67616

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67616


namespace prob_heads_greater_than_tails_6_tosses_l67_67130

-- We define our coin toss space
def coin_tosses : List Bool := [true, false]  -- true represents heads, false represents tails

-- Function to calculate the number of heads
def num_heads (l : List Bool) : Nat :=
  l.countp id  -- counting the number of true values in the list

-- Define conditions for more heads than tails in 6 tosses
def heads_greater_than_tails : List Bool -> Prop :=
  fun l => l.length = 6 ∧ num_heads l > (l.length - num_heads l)

-- Calculate the probability
noncomputable def prob_heads_greater_than_tails : Real :=
  let total_outcomes := 2^6
  let favorable_outcomes := List.length (List.filter heads_greater_than_tails (List.replicateM 6 coin_tosses))
  favorable_outcomes / total_outcomes

-- Proof statement
theorem prob_heads_greater_than_tails_6_tosses :
  prob_heads_greater_than_tails = 11 / 32 := by sorry

end prob_heads_greater_than_tails_6_tosses_l67_67130


namespace door_solution_l67_67184

def door_problem (x : ℝ) : Prop :=
  let w := x - 4
  let h := x - 2
  let diagonal := x
  (diagonal ^ 2 - (h) ^ 2 = (w) ^ 2)

theorem door_solution (x : ℝ) : door_problem x :=
  sorry

end door_solution_l67_67184


namespace sister_gave_correct_l67_67298

-- Definitions based on the given conditions
def initial_candies : ℕ := 23
def eaten_candies : ℕ := 7
def final_candies : ℕ := 37

-- The number of pieces Robin's sister gave her
def sister_gave : ℕ :=  final_candies - (initial_candies - eaten_candies)

-- The proof statement
theorem sister_gave_correct : sister_gave = 21 := by
  tidy
  have h1 : initial_candies - eaten_candies = 16 := by
    exact Nat.sub_eq 23 7
  have h2 : final_candies - 16 = 21 := by
    exact Nat.sub_eq 37 16
  simp [sister_gave, h1, h2]
  rfl

end sister_gave_correct_l67_67298


namespace find_n_solution_l67_67050

def product_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.prod

theorem find_n_solution : ∃ n : ℕ, n > 0 ∧ n^2 - 17 * n + 56 = product_of_digits n ∧ n = 4 := 
by
  sorry

end find_n_solution_l67_67050


namespace max_possible_value_e_l67_67418

def b (n : ℕ) : ℕ := (7^n - 1) / 6

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n+1))

theorem max_possible_value_e (n : ℕ) : e n = 1 := by
  sorry

end max_possible_value_e_l67_67418


namespace roots_of_unity_reals_l67_67979

theorem roots_of_unity_reals (S : Finset ℂ) (h : ∀ z ∈ S, z ^ 30 = 1) (h_card : S.card = 30) : 
  (S.filter (λ z, z ^ 10 ∈ Set ℝ)).card = 10 := 
sorry

end roots_of_unity_reals_l67_67979


namespace trainee_teacher_arrangements_l67_67222

-- Define the main theorem
theorem trainee_teacher_arrangements :
  let n := 5
  let classes := 3
  let arrangements := 50
  (∃ (C : Fin classes → Finset (Fin n)) (h : ∀ i, C i ≠ ∅) (hXiaoLi : 0 ∈ C 0),
    (∀ i j, i ≠ j → C i ∩ C j = ∅) ∧ (Finset.univ = ⋃ i, C i)) ↔ (n = 5 ∧ classes = 3 ∧ arrangements = 50) := 
by 
  sorry

end trainee_teacher_arrangements_l67_67222


namespace same_units_digit_pages_l67_67557

theorem same_units_digit_pages (n : ℕ) (h₁ : n = 75) :
  (∑ i in finset.filter (λ x : ℕ, (x % 10) = ((76 - x) % 10)) (finset.range (n+1)), 1) = 15 :=
by sorry

end same_units_digit_pages_l67_67557


namespace range_a_l67_67348

noncomputable theory
open Real

def f (x : ℝ) (a : ℝ) : ℝ := a * ln (x + 1) - x^2

theorem range_a (a : ℝ) (p q : ℝ) (hp : p ∈ Ioo 0 1) (hq : q ∈ Ioo 0 1) (hpq : p ≠ q) : 
  a ≥ 18 → (f (p + 1) a - f (q + 1) a) / (p - q) > 2 :=
sorry

end range_a_l67_67348


namespace equilateral_MNP_l67_67440

theorem equilateral_MNP
  (A B C M N P : Point)
  (h1 : is_triangle A B C)
  (h2 : is_isosceles B M C 120)
  (h3 : is_isosceles C N A 120)
  (h4 : is_isosceles A P B 120)
  : is_equilateral M N P :=
  sorry

end equilateral_MNP_l67_67440


namespace orthogonal_projections_coincide_l67_67428

variable (A : Type) [Point A]
variable (α : Plane)
variable (l : Line α)
variable (B : Point α)

def is_orthogonal_projection (A : Point) (α : Plane) (B : Point) : Prop := 
  -- A is such a point that B is its orthogonal projection onto α 
  sorry

def is_on_line (p : Point) (l : Line α) : Prop :=
  -- p is on the line l 
  sorry

theorem orthogonal_projections_coincide
  (A : Point) (α : Plane) (l : Line α) (B : Point)
  (h1 : is_orthogonal_projection A α B)
  (h2 : is_on_line B l) :
  ∀ P Q : Point, is_orthogonal_projection P l → is_orthogonal_projection Q l → P = Q := 
by
  sorry

end orthogonal_projections_coincide_l67_67428


namespace correct_mean_25_values_l67_67525

theorem correct_mean_25_values (incorrect_mean : ℝ) (wrong_value correct_value : ℝ) (count : ℕ) 
  (incorrect_mean_eq : incorrect_mean = 190) 
  (wrong_value_eq : wrong_value = 130) 
  (correct_value_eq : correct_value = 165) 
  (count_eq : count = 25) : 
  let incorrect_total_sum := incorrect_mean * count,
      corrected_total_sum := incorrect_total_sum - wrong_value + correct_value,
      correct_mean := corrected_total_sum / count 
  in correct_mean = 191.4 :=
by 
  sorry

end correct_mean_25_values_l67_67525


namespace find_a_l67_67812

noncomputable def U := ℝ
def M (a : ℝ) := { x : ℝ | x + a ≥ 0 }
def N := { x : ℝ | log x / log 2 - 1 < 1 }
def complement_N := { x : ℝ | x ≤ 1 ∨ x ≥ 3 }

theorem find_a (a : ℝ) (h : M a ∩ complement_N = {x : ℝ | x = 1 ∨ x ≥ 3}) : a = -1 :=
  sorry

end find_a_l67_67812


namespace pen_cost_l67_67020

variable (joshua_has : ℤ := 500) -- Joshua has 5 dollars (500 cents)
variable (borrowed : ℤ := 68) -- Joshua borrowed 68 cents
variable (needed_more : ℤ := 32) -- Joshua needs 32 more cents
variable (pen_cost_in_cents : ℤ := 600) -- The pen costs 600 cents (6 dollars)

/-- Prove that the cost of the pen is $6.00 given the conditions. -/
theorem pen_cost (joshua_has + borrowed + needed_more = pen_cost_in_cents) : 
  joshua_has + borrowed + needed_more = 600 := 
sorry

end pen_cost_l67_67020


namespace simplify_fraction_l67_67726

variable {x y : ℝ}

theorem simplify_fraction (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
  sorry

end simplify_fraction_l67_67726


namespace ellipse_eqn_product_of_slopes_sum_of_squares_l67_67857

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2)/2 + y^2 = 1

-- Given conditions
variable (x y a b : ℝ)
-- Ellipse data
axiom eccentricity : a > b > 0 ∧ (c = sqrt(a^2 - b^2)) ∧ (c = 1) ∧ (e = c/a) ∧ (e = sqrt(2)/2)
axiom on_circle : ∀ (x y : ℝ), x^2 + y^2 = 1 → (x, y) = (sqrt(2), 0)

-- Definition of points A, B, M on the ellipse
variables (x1 y1 x2 y2 x y : ℝ)
axiom on_ellipse_A : ellipse_eq x1 y1
axiom on_ellipse_B : ellipse_eq x2 y2
axiom on_ellipse_M : ellipse_eq x y
axiom relation_M : x = x1 * (cos θ) + x2 * (sin θ) ∧ y = y1 * (cos θ) + y2 * (sin θ)

-- Part (1): Equation of ellipse
theorem ellipse_eqn : ellipse_eq x y := sorry

-- Part (2)(i): Product of slopes
theorem product_of_slopes (x1 y1 x2 y2 : ℝ) : (y1 / x1) * (y2 / x2) = -1/2 := sorry

-- Part (2)(ii): Sum of squares
theorem sum_of_squares (x1 y1 x2 y2 : ℝ) : (x1^2 + y1^2) + (x2^2 + y2^2) = 3 := sorry

end ellipse_eqn_product_of_slopes_sum_of_squares_l67_67857


namespace min_value_arithmetic_seq_l67_67769

theorem min_value_arithmetic_seq (a : ℕ → ℝ) (h_arith_seq : ∀ n, a n ≤ a (n + 1)) (h_pos : ∀ n, a n > 0) (h_cond : a 1 + a 2017 = 2) :
  ∃ (min_value : ℝ), min_value = 2 ∧ (∀ (x y : ℝ), x + y = 2 → x > 0 → y > 0 → x + y / (x * y) = 2) :=
  sorry

end min_value_arithmetic_seq_l67_67769


namespace max_three_element_subsets_l67_67223

variable (S : Set ℕ)
variable {s1 s2 : Set ℕ}

-- Definitions from conditions
def is_seven_elements (s : Set ℕ) : Prop := s.card = 7
def is_three_element_subset (s : Set ℕ) (t : Set ℕ) : Prop := t ⊆ s ∧ t.card = 3
def has_exactly_one_common_element (t1 t2 : Set ℕ) : Prop := (t1 ∩ t2).card = 1

-- Main theorem to prove
theorem max_three_element_subsets (S : Set ℕ)
  (hS : is_seven_elements S)
  (h_intersect : ∀ (t1 t2 : Set ℕ), t1 ≠ t2 → is_three_element_subset S t1 → is_three_element_subset S t2 → has_exactly_one_common_element t1 t2) :
  ∃ (l : List (Set ℕ)), l.length = 7 ∧ ∀ t ∈ l, is_three_element_subset S t ∧ ∀ t1 t2 ∈ l, t1 ≠ t2 → has_exactly_one_common_element t1 t2 :=
  sorry

end max_three_element_subsets_l67_67223


namespace polar_equation_of_line_through_point_l67_67738

-- Definitions
def point := (2 : ℝ, real.pi / 4)

def line_parallel_to_polar_axis_through_polar (pt : ℝ × ℝ) : Prop :=
  ∃ (ρ θ : ℝ), pt = (ρ, θ) ∧ ρ * real.sin θ = √2

-- Theorem
theorem polar_equation_of_line_through_point :
  line_parallel_to_polar_axis_through_polar point :=
sorry

end polar_equation_of_line_through_point_l67_67738


namespace sin_240_eq_neg_sqrt3_div_2_l67_67672

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67672


namespace billy_ate_9_apples_on_wednesday_l67_67589

/-- Define the problem conditions -/
def apples (day : String) : Nat :=
  match day with
  | "Monday" => 2
  | "Tuesday" => 2 * apples "Monday"
  | "Friday" => apples "Monday" / 2
  | "Thursday" => 4 * apples "Friday"
  | _ => 0  -- For other days, we'll define later

/-- Define total apples eaten -/
def total_apples : Nat := 20

/-- Define sum of known apples excluding Wednesday -/
def known_sum : Nat :=
  apples "Monday" + apples "Tuesday" + apples "Friday" + apples "Thursday"

/-- Calculate apples eaten on Wednesday -/
def wednesday_apples : Nat := total_apples - known_sum

theorem billy_ate_9_apples_on_wednesday : wednesday_apples = 9 :=
  by
  sorry  -- Proof skipped

end billy_ate_9_apples_on_wednesday_l67_67589


namespace num_valid_arrangements_l67_67443

-- Define the types of files 
inductive File
  | A | B | C | D | E

open File

-- Define the type of drawers
abbreviation Drawer := Fin 7

-- Conditions where each file is placed in a drawer
def is_valid_arrangement (arrangement : File → Drawer) : Prop :=
  let adjacent (x y : Drawer) := abs (x.val - y.val) = 1
  adjacent (arrangement A) (arrangement B) ∧ 
  adjacent (arrangement C) (arrangement D)

-- Goal: Prove the number of valid arrangements
theorem num_valid_arrangements :
  {a : File → Drawer // is_valid_arrangement a}.to_finset.card = 240 :=
by
  sorry

end num_valid_arrangements_l67_67443


namespace sin_squared_minus_cos_squared_value_l67_67476

noncomputable def sin_squared_minus_cos_squared : Real :=
  (Real.sin (Real.pi / 12))^2 - (Real.cos (Real.pi / 12))^2

theorem sin_squared_minus_cos_squared_value :
  sin_squared_minus_cos_squared = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_squared_minus_cos_squared_value_l67_67476


namespace inf_11_lambda_9_mu_l67_67846

noncomputable def minimum_value_lambda_mu : ℝ :=
  let O := (0, 0) : ℝ × ℝ
  let A := (1, 2) : ℝ × ℝ
  let B := (3, 0) : ℝ × ℝ
  let circle := { p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 1 }
  let OP (P : ℝ × ℝ) (λ μ : ℝ) := λ • A + μ • B
  inf { 11 * λ + 9 * μ | P ∈ circle ∧ ∃ λ μ, OP P λ μ = P }

theorem inf_11_lambda_9_mu :
  minimum_value_lambda_mu = 12 :=
sorry

end inf_11_lambda_9_mu_l67_67846


namespace age_of_B_l67_67092

theorem age_of_B (A B C : ℕ) 
  (h1 : (A + B + C) / 3 = 22)
  (h2 : (A + B) / 2 = 18)
  (h3 : (B + C) / 2 = 25) : 
  B = 20 := 
by
  sorry

end age_of_B_l67_67092


namespace equal_degree_l67_67243

open GraphTheory

def ChessTournament (V : Type) [Fintype V] (E : V → V → Prop) :=
∀ x y : V, x ≠ y → 
  (∃ C D : V, C ≠ x ∧ C ≠ y ∧ D ≠ x ∧ D ≠ y ∧
    E x C ∧ E y C ∧ E x D ∧ E y D) ∧ -- Condition about contestants not playing
  ∀ a b c d : V, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
    E a b + E a c + E a d + E b c + E b d + E c d ≠ 5 -- No four contestants playing exactly five games

theorem equal_degree {V : Type} [Fintype V] (E : V → V → Prop) (h : ChessTournament V E) : 
  ∃ d : ℕ, ∀ v : V, degree v E = d :=
sorry

end equal_degree_l67_67243


namespace two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l67_67273

theorem two_pow_2023_add_three_pow_2023_mod_seven_not_zero : (2^2023 + 3^2023) % 7 ≠ 0 := 
by sorry

end two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l67_67273


namespace sin_240_eq_neg_sqrt3_div_2_l67_67614

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67614


namespace range_of_a_l67_67798

def f (x : ℝ) : ℝ := Real.exp (|x|) + x^2

theorem range_of_a (a : ℝ) : 
  (f (3 * a - 2) > f (a - 1)) ↔ (a < 1/2 ∨ a > 3/4) :=
by
  sorry

end range_of_a_l67_67798


namespace shift_cos_graph_l67_67454

theorem shift_cos_graph (x : ℝ) : (cos (2 * (x - π / 12)) = cos (2 * x - π / 6)) :=
by sorry

end shift_cos_graph_l67_67454


namespace checkered_triangle_division_l67_67697

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l67_67697


namespace score_order_l67_67004

variable (A B C D : ℕ)

theorem score_order
  (h1 : A + C = B + D)
  (h2 : B > D)
  (h3 : C > A + B) :
  C > B ∧ B > A ∧ A > D :=
by 
  sorry

end score_order_l67_67004


namespace negation_of_forall_ge_zero_l67_67473

theorem negation_of_forall_ge_zero :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
sorry

end negation_of_forall_ge_zero_l67_67473


namespace math_problem_proof_l67_67468

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

variables (a b c : ℝ) (h : a ≠ 0)
variables (y_val : ℕ → ℝ) 
(h0 : y_val (-3) = 15) 
(h1 : y_val (-1) = 3) 
(h2 : y_val 0 = 0)
(h3 : y_val 1 = -1)
(h4 : y_val 2 = 0)
(h5 : y_val 4 = 8)
(h6 : y_val n = 3)
(h7 : ∀ x, quadratic_function a b c x = y_val x)

theorem math_problem_proof :
  (∀ x, x = 1 ↔ (quadratic_function a b c x = -1) → 
    (by simp only [quadratic_function])
  ) ∧ 
  (m = 8 ∧ n = 3) ∧ 
  (∃ x₁ x₂, quadratic_function a b c x₁ = 0 ∧ quadratic_function a b c x₂ = 0 ∧ x₁ = 0 ∧ x₂ = 2) :=
begin
  sorry
end

end math_problem_proof_l67_67468


namespace smallest_y_value_l67_67149

theorem smallest_y_value : 
  ∀ y : ℝ, (3 * y^2 + 15 * y - 90 = y * (y + 20)) → y ≥ -6 :=
by
  sorry

end smallest_y_value_l67_67149


namespace count_even_digits_in_base_8_of_512_l67_67248

def is_even (n : ℕ) : Prop := n % 2 = 0

def base_8_representation_of_512 : list ℕ := [1, 0, 0, 0]  -- Base-8 representation of 512

theorem count_even_digits_in_base_8_of_512 :
  (list.countp is_even base_8_representation_of_512) = 3 :=
by
  sorry

end count_even_digits_in_base_8_of_512_l67_67248


namespace correct_solution_statement_l67_67160

-- Definitions based on the conditions in the problem
def condition_A (volume_naoh_sol water: ℝ) (mass_frac_naoh: ℝ) (density_naoh_sol: ℝ) : Prop :=
  let mixed_density := (density_naoh_sol + 1.0) / 2 -- Assuming density of water is 1 g/cm³.
  let total_mass := mixed_density * (volume_naoh_sol + water)
  let solute_mass_frac := (mass_frac_naoh * density_naoh_sol * volume_naoh_sol) / total_mass
  solute_mass_frac = 20.0 / 100.0

def condition_B (required_mass_naoh : ℝ) (prepared_mass_naoh: ℝ) : Prop :=
  required_mass_naoh = 0.25 * 0.5 * 40.0 ∧ prepared_mass_naoh = 4.8

def condition_C : Prop :=
  "viewing the scale mark from above" = "higher concentration"

def condition_D : Prop :=
  "Pass Cl₂ through the solution" = "removes FeSO₄ from Fe₂(SO₄)₃"

-- Combining conditions into a main theorem statement
theorem correct_solution_statement :
  condition_A 0.5 0.5 40 = false ∧
  condition_B 5.0 4.8 = false ∧
  condition_C = true ∧
  condition_D = false :=
by sorry

end correct_solution_statement_l67_67160


namespace divide_triangle_l67_67710

/-- 
  Given a triangle with the total sum of numbers as 63,
  we want to prove that it can be divided into three parts
  where each part's sum is 21.
-/
theorem divide_triangle (total_sum : ℕ) (H : total_sum = 63) :
  ∃ (part1 part2 part3 : ℕ), 
    (part1 + part2 + part3 = total_sum) ∧ 
    part1 = 21 ∧ 
    part2 = 21 ∧ 
    part3 = 21 :=
by 
  use 21, 21, 21
  split
  . exact H.symm ▸ rfl
  . split; rfl
  . split; rfl
  . rfl

end divide_triangle_l67_67710


namespace measure_of_angle_RPQ_l67_67012

theorem measure_of_angle_RPQ (P R S Q : Type) [DecidableEq Q]
  (hPQ_EQ_PR : PQ = PR) (hAngle_RSQ : ∠RSQ = 3 * x)
  (hAngle_RPQ : ∠RPQ = 2 * x) (hBisector : QP bisects ∠SQR) :
  ∠RPQ = 72 :=
by
  sorry

end measure_of_angle_RPQ_l67_67012


namespace minimal_area_proof_l67_67319

noncomputable def minimal_area_triangle (A X Y O : Point) (h1 : inside_angle A X Y O) : Triangle :=
  let M := point_on_extension A O (dist_eq A O)
  let B := intersect (parallel_line M Y) A X 
  let C := intersect (line_segment B O) A Y
  Triangle.mk A B C

theorem minimal_area_proof (A X Y O : Point) (h1 : inside_angle A X Y O) :
  ∃ A B C, minimal_area_triangle A X Y O = Triangle.mk A B C ∧ (∀ (B₁ C₁ : Point), line_through O (B₁, C₁) → area (Triangle.mk A B C) ≤ area (Triangle.mk A B₁ C₁)) :=
sorry

end minimal_area_proof_l67_67319


namespace problem1_l67_67185

theorem problem1 :
  ∃ (sin_45 : ℝ) (a : ℝ) (b : ℤ), 
  sin_45 = (√2 / 2) ∧ a = 4 ∧ b = -1 → 
  (√2 * sin_45 - a - b = -2) :=
sorry

end problem1_l67_67185


namespace solve_sin_cos_l67_67413

def int_part (x : ℝ) : ℤ := ⌊x⌋

theorem solve_sin_cos (x : ℝ) :
  int_part (Real.sin x + Real.cos x) = 1 ↔ ∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (Real.pi / 2) + 2 * Real.pi * n :=
by
  sorry

end solve_sin_cos_l67_67413


namespace convex_quadrilateral_diagonals_l67_67446

theorem convex_quadrilateral_diagonals {a b c d e f : ℝ} (h : convex_quadrilateral a b c d e f) (k : ℝ) (h_k : k = a + b + c + d) :
  (k / 2) < (e + f) ∧ (e + f) < k :=
sorry

-- Definitions for the context of the problem.
structure convex_quadrilateral (a b c d e f : ℝ) : Prop :=
-- Assuming the existence of a definition that ensures the quadrilateral with given sides and diagonals is convex.
-- Relevant conditions of convexity of the quadrilateral should be included here.

end convex_quadrilateral_diagonals_l67_67446


namespace sin_240_eq_neg_sqrt3_div_2_l67_67641

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67641


namespace cylinder_problem_l67_67716

noncomputable def cylinder_volume (radius height : ℝ) : ℝ :=
  π * radius^2 * height

theorem cylinder_problem (h r : ℝ)
  (height_C_eq_radius_D : h = r)
  (radius_C_eq_height_D : r = 3 * h)
  (volume_relation : cylinder_volume r h = 3 * cylinder_volume h r) :
  ∃ M : ℝ, cylinder_volume r h = M * π * h^3 :=
begin
  use 9,
  sorry,
end

end cylinder_problem_l67_67716


namespace sum_first_five_terms_arithmetic_l67_67879

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → α} (a_is_arithmetic : is_arithmetic_sequence a)
          (h1 : a 2 + a 6 = 10)
          (h2 : a 4 * a 8 = 45)

theorem sum_first_five_terms_arithmetic (a : ℕ → α) (a_is_arithmetic : is_arithmetic_sequence a)
  (h1 : a 2 + a 6 = 10) (h2 : a 4 * a 8 = 45) : 
  (∑ i in Finset.range 5, a i) = 20 := 
by
  sorry  

end sum_first_five_terms_arithmetic_l67_67879


namespace sin_240_l67_67627

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l67_67627


namespace reciprocal_roots_iff_l67_67807

-- Define the quadratic equation
def quadratic_eq (k : ℝ) : Poly :=
  (5 : ℝ) * X^2 + (7 : ℝ) * X + k

-- Define the condition for roots to be reciprocals
def roots_reciprocal (r s : ℝ) : Prop :=
  r * s = 1

-- Define Vieta's formula for this specific quadratic equation
def vieta_product_of_roots (k : ℝ) (r s : ℝ) : Prop :=
  r * s = k / 5

-- The main theorem which states the condition under which roots are reciprocal
theorem reciprocal_roots_iff (k : ℝ) (r s : ℝ) :
  (roots_reciprocal r s) ↔ (vieta_product_of_roots k r s) :=
by
  sorry

end reciprocal_roots_iff_l67_67807


namespace range_of_m_l67_67336

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ∀ y : ℝ, (2 ≤ x ∧ x ≤ 3) → (3 ≤ y ∧ y ≤ 6) → m * x^2 - x * y + y^2 ≥ 0) ↔ (m ≥ 0) :=
by
  sorry

end range_of_m_l67_67336


namespace angle_AKD_eq_angle_B_l67_67908

def is_isosceles (A B C : Point) : Prop :=
  dist A B = dist B C

noncomputable def point_relation (A B C D K : Point) : Prop :=
  (dist A D / dist D C = 2) ∧ (∠ A K D / ∠ D K C = 2)

theorem angle_AKD_eq_angle_B (A B C D K : Point) 
  (h_iso : is_isosceles A B C) 
  (h_on_AC : D ∈ line_segment A C) 
  (h_on_BD : K ∈ line_segment B D) 
  (h_relation : point_relation A B C D K) :
  ∠ A K D = ∠ B :=
sorry

end angle_AKD_eq_angle_B_l67_67908


namespace third_side_length_l67_67380

noncomputable def calc_third_side (a b : ℕ) (hypotenuse : Bool) : ℝ :=
if hypotenuse then
  Real.sqrt (a^2 + b^2)
else
  Real.sqrt (abs (a^2 - b^2))

theorem third_side_length (a b : ℕ) (h_right_triangle : (a = 8 ∧ b = 15)) :
  calc_third_side a b true = 17 ∨ calc_third_side 15 8 false = Real.sqrt 161 :=
by {
  sorry
}

end third_side_length_l67_67380


namespace sin_240_eq_neg_sqrt3_over_2_l67_67652

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67652


namespace area_of_triangle_l67_67339

theorem area_of_triangle (s1 s2 s3 : ℕ) (h1 : s1^2 = 36) (h2 : s2^2 = 64) (h3 : s3^2 = 100) (h4 : s1^2 + s2^2 = s3^2) :
  (1 / 2 : ℚ) * s1 * s2 = 24 := by
  sorry

end area_of_triangle_l67_67339


namespace function_property_l67_67234

/-- Define the functions under consideration. --/
def f₁ (x : ℝ) : ℝ := x^3
def f₂ (x : ℝ) : ℝ := cos (2 * x)
def f₃ (x : ℝ) : ℝ := sin (3 * x)
def f₄ (x : ℝ) : ℝ := tan (2 * x + π / 4)

/-- Prove that f₃ is both an odd function and a periodic function. --/
theorem function_property : 
  (∃ T : ℝ, T > 0 ∧ ∀ x, f₃ (x + T) = f₃ x) ∧ 
  (∀ x, f₃ (-x) = -f₃ x) :=
sorry

end function_property_l67_67234


namespace wire_length_approx_is_correct_l67_67230

noncomputable def S : ℝ := 5.999999999999998
noncomputable def L : ℝ := (5 / 2) * S
noncomputable def W : ℝ := S + L

theorem wire_length_approx_is_correct : abs (W - 21) < 1e-16 := by
  sorry

end wire_length_approx_is_correct_l67_67230


namespace number_of_squares_center_55_40_l67_67439

/-- 
Given that the center of a square is at (55, 40) 
and all vertices have natural number coordinates,
prove that the number of such squares is 1560.
-/
theorem number_of_squares_center_55_40
    (center_x : ℕ := 55)
    (center_y : ℕ := 40) : 
    ∃ n : ℕ, n = 1560 ∧ 
      all squares with center (center_x, center_y) and vertices having natural number coordinates = n :=
sorry

end number_of_squares_center_55_40_l67_67439


namespace probability_X_gt_4_l67_67338

open ProbabilityTheory

theorem probability_X_gt_4 (X : ℝ → Measure ℝ) (σ : ℝ) (hX : X = Normal 2 σ) (h : ∫ x in Ioo 0 4, pdf (Normal 2 σ) x = 0.8) :
  ∫ x in Ioi 4, pdf (Normal 2 σ) x = 0.1 :=
sorry

end probability_X_gt_4_l67_67338


namespace cosine_theta_between_planes_l67_67887

open Real

-- Define vectors representing the normal vectors of the planes
def n1 := (3: ℝ, -1, 1)
def n2 := (4: ℝ, 2, -1)

-- Define the dot product of two 3-dimensional vectors
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Define the magnitude of a 3-dimensional vector
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)

-- Prove that cos theta = 9 / sqrt 231
noncomputable def cos_theta : ℝ :=
  dot_product n1 n2 / (magnitude n1 * magnitude n2)

theorem cosine_theta_between_planes :
  cos_theta = 9 / sqrt 231 :=
sorry

end cosine_theta_between_planes_l67_67887


namespace pascal_50_5th_element_is_22050_l67_67997

def pascal_fifth_element (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_50_5th_element_is_22050 :
  pascal_fifth_element 50 4 = 22050 :=
by
  -- Calculation steps would go here
  sorry

end pascal_50_5th_element_is_22050_l67_67997


namespace positive_integer_solutions_count_l67_67937

theorem positive_integer_solutions_count :
  (∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ (4 / m + 2 / n = 1)) → 4 := sorry

end positive_integer_solutions_count_l67_67937


namespace express_in_scientific_notation_l67_67905

theorem express_in_scientific_notation :
  (2370000 : ℝ) = 2.37 * 10^6 := 
by
  -- proof omitted
  sorry

end express_in_scientific_notation_l67_67905


namespace triangle_inequality_cubed_l67_67764

theorem triangle_inequality_cubed
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a^3 / c^3) + (b^3 / c^3) + (3 * a * b / c^2) > 1 := 
sorry

end triangle_inequality_cubed_l67_67764


namespace sin_240_eq_neg_sqrt3_div_2_l67_67628

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67628


namespace area_of_smaller_circle_l67_67497

noncomputable def smaller_circle_area (PA AB : ℝ) : ℝ :=
  let radius_smaller_circle := (PA * PA - (2 * AB * AB / 8)).sqrt
  π * radius_smaller_circle ^ 2

theorem area_of_smaller_circle (PA AB : ℝ)
  (hPA : PA = 5)
  (hAB : AB = 5) :
  smaller_circle_area PA AB = 12.5 * π :=
by
  sorry

end area_of_smaller_circle_l67_67497


namespace price_verification_l67_67173

noncomputable def price_on_hot_day : ℚ :=
  let P : ℚ := 225 / 172
  1.25 * P

theorem price_verification :
  (32 * 7 * (225 / 172) + 32 * 3 * (1.25 * (225 / 172)) - (32 * 10 * 0.75)) = 210 :=
sorry

end price_verification_l67_67173


namespace exterior_angle_sum_l67_67259

theorem exterior_angle_sum (n : ℕ) (h_n : 3 ≤ n) :
  let polygon_exterior_angle_sum := 360
  let triangle_exterior_angle_sum := 0
  (polygon_exterior_angle_sum + triangle_exterior_angle_sum = 360) :=
by sorry

end exterior_angle_sum_l67_67259


namespace smallest_base_number_l67_67580

def base6_to_dec (a b c: Nat) : Nat :=
  a * 6^2 + b * 6 + c

def base4_to_dec (a b c d: Nat) : Nat :=
  a * 4^3 + b * 4^2 + c * 4 + d

def base2_to_dec (a b c d e f: Nat) : Nat :=
  a * 2^5 + b * 2^4 + c * 2^3 + d * 2^2 + e * 2 + f

theorem smallest_base_number:
  let n1 := base6_to_dec 2 1 0 in
  let n2 := base4_to_dec 1 0 0 0 in
  let n3 := base2_to_dec 1 1 1 1 1 1 in
  n3 < n1 ∧ n3 < n2 :=
by
  sorry

end smallest_base_number_l67_67580


namespace convex_polygon_tiling_exists_l67_67893

noncomputable def convex_polygon := sorry  -- Assuming this is defined elsewhere

theorem convex_polygon_tiling_exists (K : convex_polygon) :
  ∃ (P : fin 6 → convex_polygon), 
    (∀ i, ∃ p : Point, p ∈ boundary_points_of (P i) ∧ p ∈ boundary_points_of K) ∧
    (∀ i j, i ≠ j → disjoint (interior_of (P i)) (interior_of (P j))) :=
sorry

end convex_polygon_tiling_exists_l67_67893


namespace arithmetic_sequence_sum_l67_67884

noncomputable def S (n : ℕ) (a_1 d : ℝ) : ℝ := n * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (
  a_2 a_4 a_6 a_8 a_1 d : ℝ,
  h1 : a_2 + a_6 = 10,
  h2 : a_4 * a_8 = 45,
  h3 : a_2 = a_1 + d,
  h4 : a_4 = a_1 + 3 * d,
  h5 : a_6 = a_1 + 5 * d,
  h6 : a_8 = a_1 + 7 * d,
  h7 : 2 * a_4 = 10,
  h8 : a_4 = 5,
  h9 : 5 * a_8 = 45,
  h10 : a_8 = 9,
  h11 : d = 1,
  h12 : a_1 = 2
) : S 5 a_1 d = 20 := sorry

end arithmetic_sequence_sum_l67_67884


namespace find_line_equation_l67_67492

noncomputable def y_line (m b x : ℝ) : ℝ := m * x + b
noncomputable def quadratic_y (x : ℝ) : ℝ := x ^ 2 + 8 * x + 7

noncomputable def equation_of_the_line : Prop :=
  ∃ (m b k : ℝ),
    (quadratic_y k = y_line m b k + 6 ∨ quadratic_y k = y_line m b k - 6) ∧
    (y_line m b 2 = 7) ∧ 
    b ≠ 0 ∧
    y_line 19.5 (-32) = y_line m b

theorem find_line_equation : equation_of_the_line :=
sorry

end find_line_equation_l67_67492


namespace daisy_marked_point_count_l67_67044

noncomputable def princess_daisy_marked_points (Γ : Type) [Inhabited Γ] (circular_path : Γ → Prop)
  (S : Γ) (start_position : circular_path S) 
  (luigi_speed mario_speed : ℝ) (valid_speed : luigi_speed = 1 ∧ mario_speed = 3) 
  (t : ℝ) (time_bound : 0 ≤ t ∧ t ≤ 1) : ℕ :=
let luigi_position := t % 1,
    mario_position := luigi_speed * t % 1,
    midpoint := (luigi_position + mario_position) / 2 in
if midpoint ≠ S then 1 else 0

theorem daisy_marked_point_count (Γ : Type) [Inhabited Γ] (circular_path : Γ → Prop)
  (S : Γ) (start_position : circular_path S)
  (luigi_speed mario_speed : ℝ) (valid_speed : luigi_speed = 1 ∧ mario_speed = 3) 
  (t : ℝ) (time_bound : 0 ≤ t ∧ t ≤ 1) : 
  princess_daisy_marked_points Γ circular_path S start_position luigi_speed mario_speed valid_speed t time_bound = 1 :=
sorry

end daisy_marked_point_count_l67_67044


namespace part1_part2_l67_67354

theorem part1 (a : ℝ) (h1 : (∃ x : ℝ, |x - 2| < a) -> (∃ y : ℝ, y = (3/2))) 
(h2 : (∃ x : ℝ, ¬ (|x - 2| < a)) -> (∃ y : ℝ, y = (-1/2))) : a = 1 :=
sorry

theorem part2 (a b : ℝ) (h : a + b = 1) : 
  (min 
    (by have hb_pos := (abs_pos_of_ne_zero.mpr (by decide)),
        have ha_pos := (abs_pos_of_ne_zero.mpr (by decide)),
        exact (1 / (3 * |b|) + |b| / a))
    (by have hb_neg := (abs_neg_of_ne_zero.mpr (by decide))
        exact (2 * real.sqrt 3 - 1) / 3)) :=
sorry

end part1_part2_l67_67354


namespace ratio_of_length_to_height_l67_67472

theorem ratio_of_length_to_height
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (vol_eq : 129024 = w * h * l)
  (w_eq : w = 8) :
  l / h = 7 := 
sorry

end ratio_of_length_to_height_l67_67472


namespace find_number_l67_67297

theorem find_number (x : ℝ) (h : x = 11) : (3 / x) * (∏ n in finset.range 118 \{0, 1, 2}, 1 + 1 / (n+3)) = 11 := 
by
  rw h
  simp
  sorry

end find_number_l67_67297


namespace general_formula_S_sum_T_n_l67_67772

-- Define the sequence {a_n}
def a (n : ℕ) : ℝ := 
  if n = 1 then 1 else (sqrt n) - (sqrt (n - 1))

-- Define the partial sum S_n
def S (n : ℕ) : ℝ :=
  finset.sum (finset.range n) a

-- Define the sequence {b_n}
def b (n : ℕ) : ℝ := 
  (-1:ℝ)^n / a n

-- Define the sum of the first n terms of {b_n} as T_n
def T (n : ℕ) : ℝ :=
  finset.sum (finset.range n) b

-- Prove the general formula for S_n
theorem general_formula_S (n : ℕ) : S n = sqrt n :=
by
  sorry

-- Prove the sum of the first n terms T_n for the sequence {b_n}
theorem sum_T_n (n : ℕ) : T n = (-1:ℝ)^n * sqrt n :=
by
  sorry

end general_formula_S_sum_T_n_l67_67772


namespace geometric_sequence_max_product_l67_67312

theorem geometric_sequence_max_product (a : ℕ → ℝ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 2 + a 4 = 5) :
  ∃ (n : ℕ), a 1 * a 2 * ... * a n ≤ 64 := 
sorry

end geometric_sequence_max_product_l67_67312


namespace factorization_correct_l67_67938

theorem factorization_correct (c d : ℤ) (h : 25 * x^2 - 160 * x - 144 = (5 * x + c) * (5 * x + d)) : c + 2 * d = -2 := 
sorry

end factorization_correct_l67_67938


namespace proof1_proof2_l67_67342

noncomputable def complex_expr1 (x : ℂ) : ℂ :=
(x)^6 + (conj x)^6

theorem proof1 : complex_expr1 (⟨-1 / 2, sqrt 3 / 2⟩) = 2 := by
  sorry

noncomputable def complex_expr2 (z : ℂ) (n : ℕ) : ℂ :=
if n % 2 = 1 then (z^4)^n + (conj z^4)^n else 0

theorem proof2 (hn : Odd n) : complex_expr2 (⟨1 / sqrt 2, 1 / sqrt 2⟩) n = -2 := by
  sorry

end proof1_proof2_l67_67342


namespace simplify_expression_l67_67033

theorem simplify_expression (a b c : ℝ) (h : a + b + c = 3) : 
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) →
  (1 / (b^2 + c^2 - 3 * a^2) + 1 / (a^2 + c^2 - 3 * b^2) + 1 / (a^2 + b^2 - 3 * c^2) = -3) :=
by
  intros
  sorry

end simplify_expression_l67_67033


namespace cockroach_optimal_initial_positions_l67_67992

-- Define a convex polygon and positions on the perimeter
structure ConvexPolygon (α : Type) :=
  (vertices : list α)
  (convex : Prop) -- This property indicates the convexity of the polygon

-- Cockroach structure defined on the vertices of the polygon
structure CockroachPosition (α : Type) :=
  (start_position : α)

noncomputable def max_min_distance_condition 
  {α : Type} -- Type representing points on the perimeter
  (P : ConvexPolygon α) -- Convex polygon with perimeter P
  (c1 c2 : CockroachPosition α) -- Positions of the cockroaches
  : Prop :=
  -- Positions divide the perimeter into equal halves
  sorry

-- Lean statement ensuring correct initial positions
theorem cockroach_optimal_initial_positions 
  {α : Type} 
  (P : ConvexPolygon α)
  (c1 c2 : CockroachPosition α)
  (h_same_direction_same_speed : Prop) -- Cockroaches move in the same direction at the same speed
  (init_position_cond : max_min_distance_condition P c1 c2)
  : ∃ c1_init c2_init, c1_init = c1.start_position ∧ c2_init = c2.start_position ∧ init_position_cond :=
sorry

end cockroach_optimal_initial_positions_l67_67992


namespace new_students_average_age_l67_67088

theorem new_students_average_age :
  let O := 12 in
  let A_O := 40 in
  let N := 12 in
  let new_avg := A_O - 4 in
  let total_age_before := O * A_O in
  let total_age_after := (O + N) * new_avg in
  ∃ A_N : ℕ, total_age_before + N * A_N = total_age_after ∧ A_N = 32 :=
by
  let O := 12
  let A_O := 40
  let N := 12
  let new_avg := A_O - 4
  let total_age_before := O * A_O
  let total_age_after := (O + N) * new_avg
  use 32
  split
  · sorry
  · rfl

end new_students_average_age_l67_67088


namespace total_distance_100_l67_67157

-- Definitions for the problem conditions:
def initial_velocity : ℕ := 40
def common_difference : ℕ := 10
def total_time (v₀ : ℕ) (d : ℕ) : ℕ := (v₀ / d) + 1  -- The total time until the car stops
def distance_traveled (v₀ : ℕ) (d : ℕ) : ℕ :=
  (v₀ * total_time v₀ d) - (d * total_time v₀ d * (total_time v₀ d - 1)) / 2

-- Statement to prove:
theorem total_distance_100 : distance_traveled initial_velocity common_difference = 100 := by
  sorry

end total_distance_100_l67_67157


namespace part1_number_of_students_part2_probability_distribution_and_expected_value_part3_probability_at_least_one_prize_l67_67086

-- Part 1: Number of students participating from each class
theorem part1_number_of_students :
  let students_in_class := [30, 40, 20, 10]
  let total_students := sum students_in_class
  let sampling_ratio := 10 / total_students
  let participating_students := map (λ x, x * sampling_ratio) students_in_class
  participating_students = [3, 4, 2, 1] :=
by 
  sorry

-- Part 2: Probability distribution and expected value of X
open probability_theory

theorem part2_probability_distribution_and_expected_value :
  let X := λ (correct_answers : ℕ), if correct_answers ≤ 4 then correct_answers else 0
  let p_X := function.update (function.raise_to_fun ℕ ennreal) 4 0
  p_X 1 = 1 / 30 ∧
  p_X 2 = 3 / 10 ∧
  p_X 3 = 1 / 2 ∧
  p_X 4 = 1 / 6 ∧
  let E_X := ∑ i, i * p_X i
  E_X = 2.8 :=
by
  sorry

-- Part 3: Probability that at least one student from Class 1 will receive a prize
theorem part3_probability_at_least_one_prize :
  let p_correct := 1 / 3
  let p_incorrect := 2 / 3
  let p_prize := ∑ k in finset.range 5, (choose 4 k) * (p_correct ^ k) * (p_incorrect ^ (4 - k))
  let binom_dist := function.update (binomial 3 p_prize) 3 0
  let p_at_least_one := 1 - binom_dist 0
  p_at_least_one = 217 / 729 :=
by
  sorry

end part1_number_of_students_part2_probability_distribution_and_expected_value_part3_probability_at_least_one_prize_l67_67086


namespace chess_team_boys_count_l67_67545

theorem chess_team_boys_count (J S B : ℕ) 
  (h1 : J + S + B = 32) 
  (h2 : (1 / 3 : ℚ) * J + (1 / 2 : ℚ) * S + B = 18) : 
  B = 4 :=
by
  sorry

end chess_team_boys_count_l67_67545


namespace card_average_2023_l67_67542

noncomputable def n_finder : ℕ :=
  let n := 3034 in
  if (2 * n + 1) = 3 * 2023 then
    n
  else
    0

theorem card_average_2023 (n : ℕ) (h1 : ∑ i in range (n + 1), i = n * (n + 1) / 2)
    (h2 : ∑ i in range (n + 1), i^2 = n * (n + 1) * (2 * n + 1) / 6)
    (h3 : 2 * n + 1 = 3 * 2023) : n = 3034 :=
by
  sorry

end card_average_2023_l67_67542


namespace smallest_product_set_l67_67478

theorem smallest_product_set : 
  let s := { -7, -5, -1, 1, 3 } in
  ∃ a b ∈ s, (∀ x y ∈ s, (x * y) ≥ (a * b)) ∧ (a * b = -21) :=
by
  sorry

end smallest_product_set_l67_67478


namespace checkered_triangle_division_l67_67699

-- Define the conditions as assumptions
variable (T : Set ℕ) 
variable (sum_T : Nat) (h_sumT : sum_T = 63)
variable (part1 part2 part3 : Set ℕ)
variable (sum_part1 sum_part2 sum_part3 : Nat)
variable (h_part1 : sum part1 = 21)
variable (h_part2 : sum part2 = 21)
variable (h_part3 : sum part3 = 21)

-- Define the goal as a theorem
theorem checkered_triangle_division : 
  (∃ part1 part2 part3 : Set ℕ, sum part1 = 21 ∧ sum part2 = 21 ∧ sum part3 = 21 ∧ Disjoint part1 (part2 ∪ part3) ∧ Disjoint part2 part3 ∧ T = part1 ∪ part2 ∪ part3) :=
sorry

end checkered_triangle_division_l67_67699


namespace solve_floor_trig_eq_l67_67416

-- Define the floor function
def floor (x : ℝ) : ℤ := by 
  sorry

-- Define the condition and theorem
theorem solve_floor_trig_eq (x : ℝ) (n : ℤ) : 
  floor (Real.sin x + Real.cos x) = 1 ↔ (∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (2 * Real.pi * n + Real.pi / 2)) := 
  by 
  sorry

end solve_floor_trig_eq_l67_67416


namespace addie_initial_stamps_l67_67987

-- Define the initial number of stamps Parker had.
def initial_stamps_Parker := 18

-- Define the fraction of Addie's stamps that she adds to Parker's scrapbook.
def fraction_added := 1 / 4

-- Define the final number of stamps Parker has.
def final_stamps_Parker := 36

-- Define the number of stamps Addie added to Parker's scrapbook.
def stamps_added := final_stamps_Parker - initial_stamps_Parker

-- Prove that given these conditions, Addie initially had 72 stamps.
theorem addie_initial_stamps : ∃ (x : ℕ), x / 4 = stamps_added ∧ x = 72 :=
by {
  use 72,
  split,
  {
    -- First part: x / 4 = stamps_added
    show 72 / 4 = stamps_added,
    simp [stamps_added, initial_stamps_Parker, final_stamps_Parker],
  },
  {
    -- Second part: x = 72
    refl,
  }
}

end addie_initial_stamps_l67_67987


namespace problem_statement_l67_67871

/- Definitions for the given problem -/
variables {A B C D O X Y : Point}
variables {angleB angleC : Real}
variable {a b c d : Nat}
variable (gcd_condition : Nat.gcd a b = 1)
variables (not_divisible_by_square : ∀ p : Nat, nat.prime p → ¬ (p ^ 2 ∣ c))

/- Conditions given in the problem -/
def triangle (A B C : Point) := ∃ (a b c : Real), a + b + c = 180°
def excircle (A B C : Point) (D : Point) := ∃ (circle : Circle), touches circle BC D
def circumcenter (ABC : triangle) (O : Point) := ∃ (circumcircle : Circle), center circumcircle = O
def incircle_altitude_intersect (A B C : Point) (X Y : Point) := altitude A intersects incircle at X and Y
def collinear (A O D : Point) := line through A O D

def ratio_ao_ax (A O X : Point) := ∃ (a b c d : Real), ∃ (gcd_condition : Nat.gcd a b = 1), 
  ∃ (not_divisible_by_square : ∀ p : Nat, nat.prime p → ¬ (p ^ 2 ∣ c)),
  (AO / AX = (a + b * Real.sqrt c) / d)
  
theorem problem_statement :
  (triangle A B C) ∧
  (angleB - angleC = 30) ∧
  (excircle A B C D) ∧
  (circumcenter (triangle A B C) O) ∧
  (incircle_altitude_intersect A B C X Y) ∧
  (collinear A O D) ∧
  (ratio_ao_ax A O X (12, 10, 3, 39))
→ 
  12 + 10 + 3 + 39 = 64 :=
begin
  sorry
end

end problem_statement_l67_67871


namespace points_collinear_l67_67596

variables {Γ Γ' : Type*} [metric_space Γ] [metric_space Γ']
variables {A B C D E F P Q : Type*} [point A] [point B] [point C] [point D] [point E] [point F] [point P] [point Q]
variables (h₁ : A ∈ Γ) (h₂ : A ∈ Γ') (h₃ : B ∈ Γ) (h₄ : B ∈ Γ') -- Intersections: A and B belong to both circles.
          (h₅ : tangent_line Γ A C) -- Tangent to Γ at A intersects Γ' at C
          (h₆ : tangent_line Γ' A D) -- Tangent to Γ' at A intersects Γ at D
          (h₇ : intersects CD Γ E) -- Line segment CD intersects Γ at E
          (h₈ : intersects CD Γ' F) -- Line segment CD intersects Γ' at F
          (h₉ : perpendicular E AC P) -- Perpendicular from E to AC intersects Γ' at P
          (h₁₀ : perpendicular F AD Q) -- Perpendicular from F to AD intersects Γ at Q
          (h₁₁ : same_side A P CD) -- A, P, Q are on the same side of CD
          (h₁₂ : same_side A Q CD)

theorem points_collinear : collinear A P Q :=
sorry

end points_collinear_l67_67596


namespace sin_240_eq_neg_sqrt3_over_2_l67_67646

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l67_67646


namespace isabella_haircut_l67_67400

theorem isabella_haircut (original_length current_length : ℕ) 
  (h_ori : original_length = 18) 
  (h_cur : current_length = 9) : 
  original_length - current_length = 9 := 
by 
  rw [h_ori, h_cur]
  sorry

end isabella_haircut_l67_67400


namespace shifted_parabola_l67_67931

theorem shifted_parabola (x : ℝ) : 
  let y := -3 * x^2 in
  let y1 := -3 * (x - 5)^2 in
  let y2 := y1 + 2 in
  y2 = -3 * (x - 5)^2 + 2 :=
by
  -- Proof will be added here
  sorry

end shifted_parabola_l67_67931


namespace sin_240_deg_l67_67661

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67661


namespace squirrel_journey_time_l67_67577

theorem squirrel_journey_time : 
  (let distance := 2
  let speed_to_tree := 3
  let speed_return := 2
  let time_to_tree := distance / speed_to_tree
  let time_return := distance / speed_return
  let total_time := (time_to_tree + time_return) * 60
  total_time = 100) :=
by
  sorry

end squirrel_journey_time_l67_67577


namespace num_real_z10_l67_67968

theorem num_real_z10 (z : ℂ) (h : z^30 = 1) : (∃ n : ℕ, z = exp (2 * π * I * n / 30)) → ∃ n, z^10 ∈ ℝ :=
by sorry -- Here, we need to show that there are exactly 20 such complex numbers.

end num_real_z10_l67_67968


namespace carly_practice_time_l67_67253

-- conditions
def practice_time_butterfly_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def practice_time_backstroke_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def total_weekly_practice (butterfly_hours : ℕ) (backstroke_hours : ℕ) : ℕ :=
  butterfly_hours + backstroke_hours

def monthly_practice (weekly_hours : ℕ) (weeks_per_month : ℕ) : ℕ :=
  weekly_hours * weeks_per_month

-- Proof Problem Statement
theorem carly_practice_time :
  practice_time_butterfly_weekly 3 4 + practice_time_backstroke_weekly 2 6 * 4 = 96 :=
by
  sorry

end carly_practice_time_l67_67253


namespace angle_RIS_acute_l67_67028

theorem angle_RIS_acute (ABC : Type*) [euclidean_geometry ABC]
  (A B C I K L M R S : ABC) :
  incenter I A B C →
  incircle A B C K L M →
  line_through B parallel_to line_through M K →
  intersects line_through B MK line_through L M at R →
  intersects line_through B MK line_through L K at S →
  angle_at_point_lt IR SI 90 :=
sorry

end angle_RIS_acute_l67_67028


namespace min_value_P_l67_67107

-- Define the polynomial P
def P (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

-- Theorem statement: The minimum value of P(x, y) is -18
theorem min_value_P : ∃ (x y : ℝ), P x y = -18 := by
  sorry

end min_value_P_l67_67107


namespace avg_age_new_students_l67_67091

theorem avg_age_new_students :
  ∀ (O A_old A_new_avg : ℕ) (A_new : ℕ),
    O = 12 ∧ A_old = 40 ∧ A_new_avg = (A_old - 4) ∧ A_new_avg = 36 →
    A_new * 12 = (24 * A_new_avg) - (O * A_old) →
    A_new = 32 :=
by
  intros O A_old A_new_avg A_new
  intro h
  rcases h with ⟨hO, hA_old, hA_new_avg, h36⟩
  sorry

end avg_age_new_students_l67_67091


namespace number_of_orderings_l67_67135

def house_orderings (houses : list string) : Prop :=
  ∃ green blue red pink, houses = [green, blue, red, pink] ∧
  houses.all_distinct ∧
  (∃ i j, i < j ∧ houses[i] = "green" ∧ houses[j] = "blue") ∧ 
  (∃ k l, k < l ∧ houses[k] = "red" ∧ houses[l] = "pink") ∧
  (¬ ∃ m, (houses[m] = "green" ∧ houses[m+1] = "pink") ∨ (houses[m] = "pink" ∧ houses[m+1] = "green"))

theorem number_of_orderings (houses : list string) :
  house_orderings houses → houses.card = 2 := by
  sorry

end number_of_orderings_l67_67135


namespace soda_cost_l67_67054

theorem soda_cost (b s : ℕ) 
  (h₁ : 3 * b + 2 * s = 450) 
  (h₂ : 2 * b + 3 * s = 480) : 
  s = 108 := 
by
  sorry

end soda_cost_l67_67054


namespace farmer_damage_comparison_l67_67441

noncomputable def volume_of_cheese_gnawed (shape: String) (vertices: List ℝ) : ℝ := by
  sorry

theorem farmer_damage_comparison 
    (pentagonal_prism_vertices: List ℝ) 
    (quadrilateral_pyramid_vertices: List ℝ)
    (condition_1: String = "non-regular pentagonal prism")
    (condition_2: String = "regular quadrilateral pyramid")
    (condition_3: ∀ b, b > 2 → b/2 = height)
    (condition_4: ∀ v, v ∈ pentagonal_prism_vertices ∨ v ∈ quadrilateral_pyramid_vertices →
                   ∃ r, r = 1)
    (condition_5: (nonoverlap: Σ s, s ∈ vertices → ∀ another, another ≠ s → ∀ p, p ∈ s.vertices → ¬ (p ∈ another))
    (condition_6: ∀ edge length, edge length > 2) : 
    volume_of_cheese_gnawed "pentagonal_prism" pentagonal_prism_vertices = 
    4.5 * volume_of_cheese_gnawed "quadrilateral_pyramid" quadrilateral_pyramid_vertices := 
by
  sorry

end farmer_damage_comparison_l67_67441


namespace range_of_x_for_sqrt_function_l67_67113

theorem range_of_x_for_sqrt_function (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (-x + 3)) ↔ (x ≤ 3) :=
by
  sorry

end range_of_x_for_sqrt_function_l67_67113


namespace polygon_n_is_9_l67_67547

theorem polygon_n_is_9 (n : ℕ) (h1 : ∀ i ∈ finset.range n, (interior_angle : ℝ) = 140) : n = 9 :=
sorry

end polygon_n_is_9_l67_67547


namespace fans_attended_show_l67_67480

-- Definitions from the conditions
def total_seats : ℕ := 60000
def sold_percentage : ℝ := 0.75
def fans_stayed_home : ℕ := 5000

-- The proof statement
theorem fans_attended_show :
  let sold_seats := sold_percentage * total_seats
  let fans_attended := sold_seats - fans_stayed_home
  fans_attended = 40000 :=
by
  -- Auto-generated proof placeholder.
  sorry

end fans_attended_show_l67_67480


namespace Mikey_leaves_l67_67435

theorem Mikey_leaves (initial_leaves : ℕ) (leaves_blew_away : ℕ) 
  (h1 : initial_leaves = 356) 
  (h2 : leaves_blew_away = 244) : 
  initial_leaves - leaves_blew_away = 112 :=
by
  -- proof steps would go here
  sorry

end Mikey_leaves_l67_67435


namespace sin_240_eq_neg_sqrt3_div_2_l67_67640

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67640


namespace determine_fake_coin_l67_67181

theorem determine_fake_coin (N : ℕ) : 
  (∃ (n : ℕ), N = 2 * n + 2) ↔ (∃ (n : ℕ), N = 2 * n + 2) := by 
  sorry

end determine_fake_coin_l67_67181


namespace bus_speed_excluding_stoppages_l67_67727

theorem bus_speed_excluding_stoppages (u : ℝ) (hu : u = 6) (t : ℝ) (ht : t = 1/2) : 
  2 * u = 12 :=
by
  intros
  rw [hu, ht]
  sorry

end bus_speed_excluding_stoppages_l67_67727


namespace arithmetic_sequence_length_l67_67824

theorem arithmetic_sequence_length :
  ∃ (n : ℕ), let a := 2 in
             let d := 4 in
             let l := 2014 in
             l = a + (n - 1) * d ∧ n = 504 :=
by
  use 504
  simp
  sorry

end arithmetic_sequence_length_l67_67824


namespace multiple_of_six_and_nine_l67_67084

-- Definitions: x is a multiple of 6, y is a multiple of 9.
def is_multiple_of_six (x : ℤ) : Prop := ∃ m : ℤ, x = 6 * m
def is_multiple_of_nine (y : ℤ) : Prop := ∃ n : ℤ, y = 9 * n

-- Assertions: Given the conditions, prove the following.
theorem multiple_of_six_and_nine (x y : ℤ)
  (hx : is_multiple_of_six x) (hy : is_multiple_of_nine y) :
  ((∃ k : ℤ, x - y = 3 * k) ∧
   (∃ m n : ℤ, x = 6 * m ∧ y = 9 * n ∧ (2 * m - 3 * n) % 3 ≠ 0)) :=
by
  sorry

end multiple_of_six_and_nine_l67_67084


namespace smallest_of_three_l67_67126

noncomputable def A : ℕ := 38 + 18
noncomputable def B : ℕ := A - 26
noncomputable def C : ℕ := B / 3

theorem smallest_of_three : C < A ∧ C < B := by
  sorry

end smallest_of_three_l67_67126


namespace common_rational_root_l67_67101

theorem common_rational_root (a b c d e f g : ℚ) (p : ℚ) :
  (48 * p^4 + a * p^3 + b * p^2 + c * p + 16 = 0) ∧
  (16 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 48 = 0) ∧
  (∃ m n : ℤ, p = m / n ∧ Int.gcd m n = 1 ∧ n ≠ 1 ∧ p < 0 ∧ n > 0) →
  p = -1/2 :=
by
  sorry

end common_rational_root_l67_67101


namespace sin_240_eq_neg_sqrt3_div_2_l67_67665

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l67_67665


namespace can_identify_counterfeit_coin_l67_67488

theorem can_identify_counterfeit_coin (coins : Fin 101 → ℤ) :
  (∃ n : Fin 101, (∑ i, if i ≠ n then coins i else 0 = 50 ∧
                   ∑ i, if i = n then coins i else 0 = 1)) →
  (∑ i, coins i = 0 ∨ ∑ i, coins i % 2 = 0) :=
sorry

end can_identify_counterfeit_coin_l67_67488


namespace complex_conjugate_solution_l67_67310

theorem complex_conjugate_solution (z : ℂ) (h : complex.I * z = 2 - complex.I) : complex.conj z = -1 + 2 * complex.I :=
by
  sorry

end complex_conjugate_solution_l67_67310


namespace range_of_f_l67_67147

noncomputable def f (x : ℝ) : ℝ := if x = -2 then 0 else (x^2 + 5 * x + 6) / (x + 2)

theorem range_of_f :
  set.range f = { y : ℝ | y ≠ 1 } :=
begin
  sorry
end

end range_of_f_l67_67147


namespace bananas_to_pears_ratio_l67_67110

theorem bananas_to_pears_ratio (B P : ℕ) (hP : P = 50) (h1 : B + 10 = 160) (h2: ∃ k : ℕ, B = k * P) : B / P = 3 :=
by
  -- proof steps would go here
  sorry

end bananas_to_pears_ratio_l67_67110


namespace find_m_l67_67805

noncomputable def value_of_m (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a = b) : ℝ :=
  8

theorem find_m (m a b : ℝ) (h1 : a > 0) (h2 : b > 0) (hyperbolic : x^2 / a^2 - y^2 / b^2 = 1)
    (eccentricity : ecc_hyperbola = 3) (circle : x^2 + y^2 - 6*y + m = 0)
    (tangent : intersects_asymptote_tangent): m = 8 :=
begin
  sorry
end

end find_m_l67_67805


namespace complex_division_example_l67_67420

theorem complex_division_example : (2 * complex.I) / (1 + complex.I) = 1 + complex.I := by
  sorry

end complex_division_example_l67_67420


namespace c_S_power_of_2_l67_67025

variables (m : ℕ) (S : String)

-- condition: m > 1
def is_valid_m (m : ℕ) : Prop := m > 1

-- function c(S)
def c (S : String) : ℕ := sorry  -- actual implementation is skipped

-- function to check if a number represented by a string is divisible by m
def is_divisible_by (n m : ℕ) : Prop := n % m = 0

-- Property that c(S) can take only powers of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem c_S_power_of_2 (m : ℕ) (S : String) (h1 : is_valid_m m) :
  is_power_of_two (c S) :=
sorry

end c_S_power_of_2_l67_67025


namespace number_of_squares_with_side_lengths_greater_than_5_l67_67000

-- Define the grid size and the conditions for the squares
def grid_size : ℕ := 15

-- Define the function to calculate the number of squares with integer side lengths greater than 5
def number_of_squares (n : ℕ) : ℕ :=
  let number_of_directly_aligned_squares := (List.sum (List.map (λ k => (n - k + 1) * (n - k + 1)) (List.range (15 - 6 + 1)).map (λ i => i + 6)))
  let number_of_diagonally_aligned_squares := 4 -- Since there are only 4 valid 10-length diagonal squares satisfying the problem's conditions
  number_of_directly_aligned_squares + number_of_diagonally_aligned_squares

-- Define the theorem for the specific grid size
theorem number_of_squares_with_side_lengths_greater_than_5 : number_of_squares grid_size = 389 := by
  -- It is only the statement required, the implementation proof is omitted with sorry.
  sorry

end number_of_squares_with_side_lengths_greater_than_5_l67_67000


namespace sin_240_deg_l67_67660

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l67_67660


namespace scatter_plot_linear_residuals_sum_corr_l67_67378

theorem scatter_plot_linear_residuals_sum_corr 
  (points : List (ℝ × ℝ))
  (h_line : ∀ (x y : ℝ), (x, y) ∈ points → ∃ (a b : ℝ), y = a * x + b) :
  (∀ (p : ℝ × ℝ), p ∈ points → residual p 0) ∧
  (sum_of_squares_of_residuals points = 0) ∧
  (correlation_coefficient points = 1 ∨ correlation_coefficient points = -1) :=
by
  sorry

end scatter_plot_linear_residuals_sum_corr_l67_67378
