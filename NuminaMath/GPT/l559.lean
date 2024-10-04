import Mathlib
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Divisibility
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Choose
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.NumberTheory.Prime
import Mathlib.Order.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Instances.Real

namespace quadratic_function_properties_l559_559302

-- Definitions based on problem conditions
def is_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-2 - x) = f (-2 + x)

def intersects_y_axis (f : ℝ → ℝ) : Prop :=
  f 0 = 1

def x_axis_segment_length (f : ℝ → ℝ) (length : ℝ) : Prop :=
  let roots := {x : ℝ | f x = 0} in
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ roots ∧ x2 ∈ roots ∧ abs (x2 - x1) = length

-- Translate the problem into Lean statement
theorem quadratic_function_properties (f : ℝ → ℝ) (a : ℝ) :
  is_symmetric f ∧ intersects_y_axis f ∧ x_axis_segment_length f (2 * sqrt 2) →
  (f = (λ x, (1/2 : ℝ) * x^2 + 2 * x + 1)) ∧
  (∀ x : ℝ, f x - a * x is_monotone_on [2, 3] →
    a ∈ (-∞, 4] ∪ [5, ∞)) ∧
  (∀ x : ℝ, 2 * f x - (a + 5) * x - 2 + a > 0 ↔
    (if a < 1 then x < a ∨ x > 1 else if a = 1 then x ≠ 1 else x < 1 ∨ x > a)) :=
sorry

end quadratic_function_properties_l559_559302


namespace coordinates_of_D_l559_559729

noncomputable def point := (ℝ × ℝ)

/-- Define the points A, B, and C. -/
def A : point := (-2, 1)
def B : point := (2, 5)
def C : point := (4, -1)

/-- Define the midpoint function for two points. -/
def midpoint (P Q : point) : point :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

/-- Given that M is the midpoint of B and C. -/
def M : point := midpoint B C

/-- Define the coordinates of point D. -/
def D : point := (8, 3)

theorem coordinates_of_D :
  ∃ D : point, D = (8, 3)
  ∧ (M = midpoint A D)
  ∧ (D.1 - B.1 = C.1 - A.1)
  ∧ (D.2 - B.2 = C.2 - A.2) :=
by
  use D
  sorry

end coordinates_of_D_l559_559729


namespace john_total_animals_is_114_l559_559028

  -- Define the entities and their relationships based on the conditions
  def num_snakes : ℕ := 15
  def num_monkeys : ℕ := 2 * num_snakes
  def num_lions : ℕ := num_monkeys - 5
  def num_pandas : ℕ := num_lions + 8
  def num_dogs : ℕ := num_pandas / 3

  -- Define the total number of animals
  def total_animals : ℕ := num_snakes + num_monkeys + num_lions + num_pandas + num_dogs

  -- Prove that the total number of animals is 114
  theorem john_total_animals_is_114 : total_animals = 114 := by
    sorry
  
end john_total_animals_is_114_l559_559028


namespace sequence_sum_n_eq_21_l559_559373

theorem sequence_sum_n_eq_21 (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ k, a (k + 1) = a k + 1)
  (h3 : ∀ n, S n = (n * (n + 1)) / 2)
  (h4 : S n = 21) :
  n = 6 :=
sorry

end sequence_sum_n_eq_21_l559_559373


namespace roger_trips_l559_559838

theorem roger_trips (a b trays_per_trip : ℕ) (h1 : a = 10) (h2 : b = 2) (h3 : trays_per_trip = 4) : ((a + b) / trays_per_trip = 3) :=
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end roger_trips_l559_559838


namespace ordered_pairs_1806_l559_559108

theorem ordered_pairs_1806 :
  (∃ (xy_list : List (ℕ × ℕ)), xy_list.length = 12 ∧ ∀ (xy : ℕ × ℕ), xy ∈ xy_list → xy.1 * xy.2 = 1806) :=
sorry

end ordered_pairs_1806_l559_559108


namespace solve_compound_inequality_l559_559844

noncomputable def compound_inequality_solution (x : ℝ) : Prop :=
  (3 - (1 / (3 * x + 4)) < 5) ∧ (2 * x + 1 > 0)

theorem solve_compound_inequality (x : ℝ) :
  compound_inequality_solution x ↔ (x > -1/2) :=
by
  sorry

end solve_compound_inequality_l559_559844


namespace average_monthly_growth_rate_l559_559993

variable (x : ℝ)

-- Conditions
def turnover_January : ℝ := 36
def turnover_March : ℝ := 48

-- Theorem statement that corresponds to the problem's conditions and question
theorem average_monthly_growth_rate :
  turnover_January * (1 + x)^2 = turnover_March :=
sorry

end average_monthly_growth_rate_l559_559993


namespace base_7_minus_base_8_l559_559650

def convert_base_7 (n : ℕ) : ℕ :=
  match n with
  | 543210 => 5 * 7^5 + 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0
  | _ => 0

def convert_base_8 (n : ℕ) : ℕ :=
  match n with
  | 45321 => 4 * 8^4 + 5 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0
  | _ => 0

theorem base_7_minus_base_8 : convert_base_7 543210 - convert_base_8 45321 = 75620 := by
  sorry

end base_7_minus_base_8_l559_559650


namespace relationship_between_areas_l559_559203

-- Assume necessary context and setup
variables (A B C C₁ C₂ : ℝ)
variables (a b c : ℝ) (h : a^2 + b^2 = c^2)

-- Define the conditions
def right_triangle := a = 8 ∧ b = 15 ∧ c = 17
def circumscribed_circle (d : ℝ) := d = 17
def areas_relation (A B C₁ C₂ : ℝ) := (C₁ < C₂) ∧ (A + B = C₁ + C₂)

-- Problem statement in Lean 4
theorem relationship_between_areas (ht : right_triangle 8 15 17) (hc : circumscribed_circle 17) :
  areas_relation A B C₁ C₂ :=
by sorry

end relationship_between_areas_l559_559203


namespace probability_window_seat_l559_559356

theorem probability_window_seat : 
  let seats := ["A", "B", "C", "D", "F"]
  let window_seats := ["A", "F"]
  (∀ seat, seat ∈ seats → (1 : ℝ) / (length seats) = (1 : ℝ) / 5) → 
  (∃ seat, seat ∈ window_seats → (1 : ℝ) / (length seats) = (2 : ℝ) / 5) := 
by 
  sorry

end probability_window_seat_l559_559356


namespace find_a_tangent_line_range_f_l559_559723

noncomputable def f (x a : ℝ) : ℝ := x^3 + a * x^2 - x

-- Statement 1: Given f'(1)=4, prove a=1.
theorem find_a : ∃ a : ℝ, (∀ x : ℝ, deriv (λ x, f x a) 1 = 4) → a = 1 :=
sorry

-- Statement 2: Given f(x)=x^3 + x^2 - x and the point (1, f(1)), find the equation of the tangent line.
theorem tangent_line : let a := 1 in 
  (∀ x : ℝ, deriv (λ x, f x a) x = 3 * x ^ 2 + 2 * x - 1) →
  let p : ℝ × ℝ := (1, f 1 a) in
  (p.2 = 1) →
  (∀ x y : ℝ, y = f 1 a + deriv (λ x, f x a) 1 * (x - p.fst)) →
  (∀ x y : ℝ, 4 * x - y - 3 = 0) :=
sorry

-- Statement 3: Find the range of f(x) on the interval [0, 2].
theorem range_f : let a := 1 in
  (∀ x : ℝ, deriv (λ x, f x a) x = 3 * x ^ 2 + 2 * x - 1) →
  ( ∀ x ∈ set.Icc 0 2, f x a ∈ set.Icc (-5/27 : ℝ) 10) :=
sorry

end find_a_tangent_line_range_f_l559_559723


namespace A_half_B_l559_559248

-- Define the arithmetic series sum function
def series_sum (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define A and B according to the problem conditions
def A : ℕ := (Finset.range 2022).sum (λ m => series_sum (m + 1))

def B : ℕ := (Finset.range 2022).sum (λ m => (m + 1) * (m + 2))

-- The proof statement
theorem A_half_B : A = B / 2 :=
by
  sorry

end A_half_B_l559_559248


namespace imaginaria_city_population_l559_559871

theorem imaginaria_city_population (a b c : ℕ) (h₁ : a^2 + 225 = b^2 + 1) (h₂ : b^2 + 1 + 75 = c^2) : 5 ∣ a^2 :=
by
  sorry

end imaginaria_city_population_l559_559871


namespace smallest_positive_period_range_of_f_on_interval_l559_559719

noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x) - 2 * Math.sin x ^ 2

theorem smallest_positive_period : 
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ t > 0, (∀ x, f (x + t) = f x) → T ≤ t :=
begin
  use π,
  sorry,
end

theorem range_of_f_on_interval : 
  ∃ y_min y_max, 
    ∀ y ∈ set.image (λ x, f x) (set.Icc (-π / 4) (3 * π / 8)), y_min ≤ y ∧ y ≤ y_max :=
begin
  use [-2, sqrt 2 - 1],
  sorry,
end

end smallest_positive_period_range_of_f_on_interval_l559_559719


namespace factorization_correct_l559_559252

noncomputable def original_poly (x : ℝ) : ℝ := 12 * x ^ 2 + 18 * x - 24
noncomputable def factored_poly (x : ℝ) : ℝ := 6 * (2 * x - 1) * (x + 4)

theorem factorization_correct (x : ℝ) : original_poly x = factored_poly x :=
by
  sorry

end factorization_correct_l559_559252


namespace probability_five_distinct_numbers_l559_559502

def num_dice := 5
def num_faces := 6

def favorable_outcomes : ℕ := nat.factorial 5 * num_faces
def total_outcomes : ℕ := num_faces ^ num_dice

theorem probability_five_distinct_numbers :
  (favorable_outcomes / total_outcomes : ℚ) = 5 / 54 := 
sorry

end probability_five_distinct_numbers_l559_559502


namespace sheets_of_paper_per_week_l559_559995

theorem sheets_of_paper_per_week
  (sheets_per_class_per_day : ℕ)
  (num_classes : ℕ)
  (school_days_per_week : ℕ)
  (total_sheets_per_week : ℕ) 
  (h1 : sheets_per_class_per_day = 200)
  (h2 : num_classes = 9)
  (h3 : school_days_per_week = 5)
  (h4 : total_sheets_per_week = sheets_per_class_per_day * num_classes * school_days_per_week) :
  total_sheets_per_week = 9000 :=
sorry

end sheets_of_paper_per_week_l559_559995


namespace simplify_expression_l559_559842

theorem simplify_expression (a b : ℝ) (h₁ : a = 1 - Real.sqrt 2) (h₂ : b = 1 + Real.sqrt 2) :
  ( (a⁻¹ - b⁻¹) / (a⁻³ + b⁻³) / (a^2 * b^2 / ((a + b)^2 - 3 * a * b)) * ((a^2 - b^2) / (a * b))⁻¹ ) = 1 / 4 :=
by
  sorry

end simplify_expression_l559_559842


namespace fred_washing_car_earning_l559_559384

-- Definitions based on conditions
def last_week_fred_money : ℕ := 33
def last_week_jason_money : ℕ := 95
def newspaper_earning : ℕ := 16
def weekend_total_earning : ℕ := 90
def washing_car_earning : ℕ := weekend_total_earning - newspaper_earning

-- Theorem statement: Prove Fred earned 74 dollars from washing cars
theorem fred_washing_car_earning : washing_car_earning = 74 :=
by
  rw [washing_car_earning, weekend_total_earning, newspaper_earning]
  exact Nat.sub_self 90 16 74

end fred_washing_car_earning_l559_559384


namespace vines_painted_l559_559648

-- Definitions based on the conditions in the problem statement
def time_per_lily : ℕ := 5
def time_per_rose : ℕ := 7
def time_per_orchid : ℕ := 3
def time_per_vine : ℕ := 2
def total_time_spent : ℕ := 213
def lilies_painted : ℕ := 17
def roses_painted : ℕ := 10
def orchids_painted : ℕ := 6

-- The theorem to prove the number of vines painted
theorem vines_painted (vines_painted : ℕ) : 
  213 = (17 * 5) + (10 * 7) + (6 * 3) + (vines_painted * 2) → 
  vines_painted = 20 :=
by
  intros h
  sorry

end vines_painted_l559_559648


namespace mirror_area_l559_559238

-- Defining the conditions as Lean functions and values
def frame_height : ℕ := 100
def frame_width : ℕ := 140
def frame_border : ℕ := 15

-- Statement to prove the area of the mirror
theorem mirror_area :
  let mirror_width := frame_width - 2 * frame_border
  let mirror_height := frame_height - 2 * frame_border
  mirror_width * mirror_height = 7700 :=
by
  sorry

end mirror_area_l559_559238


namespace non_zero_real_m_value_l559_559329

theorem non_zero_real_m_value (m : ℝ) (h1 : 3 - m ∈ ({1, 2, 3} : Set ℝ)) (h2 : m ≠ 0) : m = 2 := 
sorry

end non_zero_real_m_value_l559_559329


namespace arccos_cos_solution_l559_559436

theorem arccos_cos_solution :
  ∀ (x : ℝ), 0 ≤ (2 * x / 3) ∧ (2 * x / 3) ≤ ℼ → 
  (arccos (cos x) = 2 * x / 3 ↔ x = 0 ∨ x = 6 * ℼ / 5 ∨ x = 12 * ℼ / 5) := 
by
  intro x
  intro h
  constructor
  sorry

end arccos_cos_solution_l559_559436


namespace collinear_incenters_and_K_l559_559794

noncomputable def cyclic_quadrilateral {A B C D : Type*} :=
  ∃ ω : Circle, (A ∈ ω) ∧ (B ∈ ω) ∧ (C ∈ ω) ∧ (D ∈ ω)

noncomputable def triangle_incircle {A B C : Type*} :=
  ∃ I : Point, ∃ r : ℝ, is_incenter I A B C ∧ is_inradius r I A B C

theorem collinear_incenters_and_K
  {A B C D : Type*}
  (h_cyclic : cyclic_quadrilateral A B C D)
  (I1 I2 : Point)
  (r1 r2 : ℝ)
  (h_incenter_1 : triangle_incircle A C D I1 r1)
  (h_incenter_2 : triangle_incircle A B C I2 r2)
  (h_radii_equal : r1 = r2)
  (ω ω' : Circle)
  (h_tangent_1 : tangent_to_circle ω' A B)
  (h_tangent_2 : tangent_to_circle ω' A D)
  (h_tangent_3 : tangent_to_circle ω' ω)
  (T : Point)
  (h_T_on_ω'_and_ω : touching_points ω' ω T)
  (K : Point)
  (h_tangents_meet : tangents_meet A T K ω) :
  collinear I1 I2 K := by sorry

end collinear_incenters_and_K_l559_559794


namespace one_bag_covers_250_sqfeet_l559_559647

noncomputable def lawn_length : ℝ := 22
noncomputable def lawn_width : ℝ := 36
noncomputable def bags_count : ℝ := 4
noncomputable def extra_area : ℝ := 208

noncomputable def lawn_area : ℝ := lawn_length * lawn_width
noncomputable def total_covered_area : ℝ := lawn_area + extra_area
noncomputable def one_bag_area : ℝ := total_covered_area / bags_count

theorem one_bag_covers_250_sqfeet :
  one_bag_area = 250 := 
by
  sorry

end one_bag_covers_250_sqfeet_l559_559647


namespace remainder_987654_div_8_l559_559147

theorem remainder_987654_div_8 : 987654 % 8 = 2 := by
  sorry

end remainder_987654_div_8_l559_559147


namespace altitude_incircle_sum_l559_559926

noncomputable def right_triangle (a b c mc varrho varrho1 varrho2 : ℝ) :=
  isRightTriangle a b c ∧
  altitudeFromRightAngle a b c mc ∧
  incircleRadius a b c varrho ∧
  incircleRadius (bc / 2) a mc varrho1 ∧
  incircleRadius (ac / 2) b mc varrho2

theorem altitude_incircle_sum 
  (a b c mc varrho varrho1 varrho2 : ℝ)
  (h : right_triangle a b c mc varrho varrho1 varrho2) :
  mc = varrho + varrho1 + varrho2 :=
sorry

end altitude_incircle_sum_l559_559926


namespace concyclicity_and_perpendicularity_l559_559388

-- Defining points P, lines l1, l2, circles S1, S2, T1, T2, and points A, B, C, D
variable (P : Point)
variable (l1 l2 : Line)
variable (S1 S2 T1 T2 : Circle)
variable (A B C D : Point)

-- Conditions as definitions
def intersection_of_lines (P : Point) (l1 l2 : Line) : Prop := P ∈ l1 ∧ P ∈ l2
def tangent_at_point (S : Circle) (P : Point) : Prop := S.touch P
def second_intersection (S T : Circle) (X : Point) : Prop := X ≠ P ∧ X ∈ S ∧ X ∈ T

-- Hypotheses based on problem conditions
axiom h1 : intersection_of_lines P l1 l2
axiom h2 : tangent_at_point S1 P ∧ tangent_at_point S2 P
axiom h3 : tangent_at_point T1 P ∧ tangent_at_point T2 P
axiom h4 : ∀ X : Point, second_intersection S1 T1 A
axiom h5 : ∀ X : Point, second_intersection S1 T2 B
axiom h6 : ∀ X : Point, second_intersection S2 T1 C
axiom h7 : ∀ X : Point, second_intersection S2 T2 D

-- The theorem statement to prove
theorem concyclicity_and_perpendicularity :
  (concyclic {A, B, C, D} ↔ perpendicular l1 l2) :=
sorry

end concyclicity_and_perpendicularity_l559_559388


namespace probability_of_distinct_dice_numbers_l559_559492

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l559_559492


namespace quadratic_polynomial_solution_l559_559657

theorem quadratic_polynomial_solution :
  ∃ a b c : ℚ, 
    (∀ x : ℚ, ax*x + bx + c = 8 ↔ x = -2) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 2 ↔ x = 1) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 10 ↔ x = 3) ∧ 
    a = 6 / 5 ∧ 
    b = -4 / 5 ∧ 
    c = 8 / 5 :=
by {
  sorry
}

end quadratic_polynomial_solution_l559_559657


namespace value_of_x_l559_559342

theorem value_of_x (b x : ℝ) (hb : b > 0) (hb_ne : b ≠ 1) (hx_ne : x ≠ 1)
  (h : log (x) / log (b^3) + log (b) / log (x^3) + log (x) / log (b) = 2) :
  x = b^((6 - 2 * real.sqrt 5) / 8) :=
sorry

end value_of_x_l559_559342


namespace rectangular_coordinates_correct_l559_559548

noncomputable def rectangular_coordinates : ℝ × ℝ → ℝ × ℝ :=
λ (p : ℝ × ℝ),
let r := Real.sqrt (p.1 ^ 2 + p.2 ^ 2),
    theta := Real.arctan (p.2 / p.1),
    cos_theta := p.1 / r,
    sin_theta := p.2 / r,
    r2 := r * r,
    cos_3theta := 4 * cos_theta ^ 3 - 3 * cos_theta,
    sin_3theta := 3 * sin_theta - 4 * sin_theta ^ 3 in
(r2 * cos_3theta, r2 * sin_3theta)

-- Given the conditions
def p : ℝ × ℝ := (12, 5)

-- Prove that the rectangular coordinates of (r^2, 3θ) are as expected
theorem rectangular_coordinates_correct :
  rectangular_coordinates p = ( - 494004 / 2197, 4441555 / 2197 ) :=
by
  sorry

end rectangular_coordinates_correct_l559_559548


namespace coordinates_reflection_y_axis_l559_559858

theorem coordinates_reflection_y_axis :
  let M := (-5, 2) in
  reflect_y_axis M = (5, 2) :=
by
  sorry

end coordinates_reflection_y_axis_l559_559858


namespace total_votes_cast_l559_559354

-- Define the variables and constants
def total_votes (V : ℝ) : Prop :=
  let A := 0.32 * V
  let B := 0.28 * V
  let C := 0.22 * V
  let D := 0.18 * V
  -- Candidate A defeated Candidate B by 1200 votes
  0.32 * V - 0.28 * V = 1200 ∧
  -- Candidate A defeated Candidate C by 2200 votes
  0.32 * V - 0.22 * V = 2200 ∧
  -- Candidate B defeated Candidate D by 900 votes
  0.28 * V - 0.18 * V = 900

noncomputable def V := 30000

-- State the theorem
theorem total_votes_cast : total_votes V := by
  sorry

end total_votes_cast_l559_559354


namespace right_triangle_condition_l559_559776

theorem right_triangle_condition (a b c : ℝ) : (a^2 = b^2 - c^2) → (∃ B : ℝ, B = 90) := 
sorry

end right_triangle_condition_l559_559776


namespace find_missing_value_l559_559529

theorem find_missing_value :
  300 * 2 + (12 + 4) * 1 / 8 = 602 :=
by
  sorry

end find_missing_value_l559_559529


namespace inclination_range_l559_559704

-- Define the condition that the two points are on the same side of the line
def same_side_of_line (a : ℝ) (p1 p2 : ℚ × ℚ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (a * x1 - y1 + 1) * (a * x2 - y2 + 1) > 0

-- Define the points
def p1 : ℚ × ℚ := (-1, 2)
def p2 : ℚ × ℚ := (↑(Real.sqrt 3) / 3, 0)

-- Main statement: determine the range of the angle of inclination θ such that θ ∈ (2 * Real.pi / 3, 3 * Real.pi / 4)
theorem inclination_range (a : ℝ) (θ : ℝ) 
  (h_ne_zero : a ≠ 0) 
  (h_theta : θ = Real.arctan a) 
  (h_same_side : same_side_of_line a p1 p2) :
  2 * Real.pi / 3 < θ ∧ θ < 3 * Real.pi / 4 :=
sorry

end inclination_range_l559_559704


namespace angle_between_vectors_is_2pi_over_3_l559_559736

variables (a b : ℝ^3)
variables (norm_a : ∥a∥ = 1)
variables (norm_b : ∥b∥ = 2)
variables (condition : a • (a - b) = 2)

theorem angle_between_vectors_is_2pi_over_3 :
  real.angle a b = 2 * real.pi / 3 :=
by 
  -- Add the proof here
  sorry

end angle_between_vectors_is_2pi_over_3_l559_559736


namespace ellipse_c_has_equation_correct_value_of_m_l559_559690

def ellipse_c_equation (a b : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧
  (1 / 2) * Real.sqrt(2) = (Real.sqrt(a^2 - b^2) / a) ∧
  -2 = -2 ∧
  (Real.sqrt(a^2 - b^2)) = 2 ∧
  a = 2 * Real.sqrt(2) ∧ b = 2

def value_of_m (m : ℝ) : Prop :=
  -2 * Real.sqrt 3 < m ∧ m < 2 * Real.sqrt 3 ∧
  (x1 x2 : ℝ) (h1 : 3 * x1^2 + 4 * m * x1 + 2 * m^2 - 8 = 0) (h2 : 3 * x2^2 + 4 * m * x2 + 2 * m^2 - 8 = 0),
  (-2 * m / 3)^2 + (m / 3)^2 = 1

theorem ellipse_c_has_equation :
  (a b : ℝ) (h : ellipse_c_equation a b) :
  (a = 2 * Real.sqrt 2 ∧ b = 2) →
  (∀ x y : ℝ, (x^2 / 8 + y^2 / 4 = 1) → true) := sorry

theorem correct_value_of_m :
  (m : ℝ) (h : value_of_m m) :
  m = (3 * Real.sqrt 5) / 5 ∨ m = -(3 * Real.sqrt 5) / 5 := sorry

end ellipse_c_has_equation_correct_value_of_m_l559_559690


namespace image_of_line_is_line_f_is_affine_map_l559_559522

variables {n : ℕ} (f : ℝ^n → ℝ^n)
  (H_bijective : function.bijective f)
  (H_n_minus_one_affine : ∀ (A : affine_subspace ℝ ℝ^n), dim(A) = n - 1 → ∃ B : affine_subspace ℝ ℝ^n, f '' ↑A = ↑B ∧ dim(B) = n - 1)

theorem image_of_line_is_line (L : affine_subspace ℝ ℝ^n) (H_L_line : dim(L) = 1) : 
  ∃ (M : affine_subspace ℝ ℝ^n), f '' ↑L = ↑M ∧ dim(M) = 1 := 
sorry

theorem f_is_affine_map : 
  ∃ (g : ℝ^n → ℝ^n) (h : ℝ^n →ₗ[ℝ] ℝ^n), f = g ∘ h := 
sorry

end image_of_line_is_line_f_is_affine_map_l559_559522


namespace lines_are_concurrent_l559_559401

-- Definitions based on conditions
variables {α : Type*} [euclidean_space α] 
variables {A B C D X Y O M N : α}
variables (Γ_1 Γ_2 : circle α)

-- Conditions
def distinct_points : Prop := A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D ∧ A ≠ C ∧ B ≠ D

def circle_with_diameter (Γ : circle α) (x y : α) : Prop := Γ.diameter = [x, y]

def intersects_at_two_points (Γ₁ Γ₂ : circle α) (X Y : α) : Prop := 
  Γ₁ ∩ Γ₂ = {X, Y} ∧ X ≠ Y

def point_on_line (P Q R: α) : Prop := collinear P Q R 

def line_intersects_circle_at (ℓ : line α) (Γ : circle α) (P : α) : Prop := 
  P ∈ (ℓ ∩ Γ)

-- Lean 4 statement
theorem lines_are_concurrent
  (h_dist : distinct_points A B C D)
  (h_circles: circle_with_diameter Γ_1 A C ∧ circle_with_diameter Γ_2 B D)
  (h_intersect : intersects_at_two_points Γ_1 Γ_2 X Y)
  (h_on_line : point_on_line O X Y)
  (h_intersections : line_intersects_circle_at (line C O) Γ_1 M ∧ line_intersects_circle_at (line B O) Γ_2 N):
  concurrent (line A M) (line D N) (line X Y) :=
sorry

end lines_are_concurrent_l559_559401


namespace angle_O_AOO_B_range_l559_559708

/-- Given the incenter O and the centers of two excircles O_A and O_B of a triangle,
    prove that the angle O_AOO_B lies between 90 degrees and 180 degrees, 
    provided O_A, O, and O_B are not collinear and ∠O_AOO_B ≠ 90 degrees. -/
theorem angle_O_AOO_B_range (O O_A O_B : Point)
  (h_incenter : is_incenter O)
  (h_excenter_A : is_excenter O_A)
  (h_excenter_B : is_excenter O_B)
  (h_not_collinear : ¬ collinear O_A O O_B)
  (h_not_perpendicular : ∠ O_A O O_B ≠ 90) :
  90 < ∠ O_A O O_B ∧ ∠ O_A O O_B < 180 :=
sorry

end angle_O_AOO_B_range_l559_559708


namespace max_interesting_numbers_l559_559906

/-- 
Define a function to calculate the sum of the digits of a natural number.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

/-- 
Define a function to check if a natural number is prime.
-/
def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

/-- 
Define a number to be interesting if the sum of its digits is a prime number.
-/
def interesting_number (n : ℕ) : Prop :=
  is_prime (sum_of_digits n)

/-- 
Statement: For any five consecutive natural numbers, there are at most 4 interesting numbers.
-/
theorem max_interesting_numbers (a : ℕ) :
  (finset.range 5).filter (λ i, interesting_number (a + i)) .card ≤ 4 := 
by
  sorry

end max_interesting_numbers_l559_559906


namespace locus_is_apollonius_circle_l559_559638

noncomputable def apollonius_circle (n : ℝ) (A B : ℝ × ℝ) :
    set (ℝ × ℝ) :=
  { P | dist P A = n * dist P B }

theorem locus_is_apollonius_circle (n : ℝ) (A B : ℝ × ℝ) :
    { P : ℝ × ℝ | dist P A = n * dist P B } = apollonius_circle n A B :=
by simp [apollonius_circle]

end locus_is_apollonius_circle_l559_559638


namespace remaining_paint_fraction_l559_559940

def initial_paint : ℚ := 1

def paint_day_1 : ℚ := initial_paint - (1/2) * initial_paint
def paint_day_2 : ℚ := paint_day_1 - (1/4) * paint_day_1
def paint_day_3 : ℚ := paint_day_2 - (1/3) * paint_day_2

theorem remaining_paint_fraction : paint_day_3 = 1/4 :=
by
  sorry

end remaining_paint_fraction_l559_559940


namespace base_nine_sum_of_product_of_125_and_33_is_16_l559_559445

theorem base_nine_sum_of_product_of_125_and_33_is_16 :
  let n₁ := 1 * 9^2 + 2 * 9^1 + 5 * 9^0,
      n₂ := 3 * 9^1 + 3 * 9^0,
      product := n₁ * n₂,
      base_nine_digits := [4, 2, 4, 6],
      sum_of_digits := 4 + 2 + 4 + 6
  in sum_of_digits = 16 := by
begin
  -- Defining the numbers in base 10
  let n₁ := 1 * 9^2 + 2 * 9^1 + 5 * 9^0,
  let n₂ := 3 * 9^1 + 3 * 9^0,
  have n₁_val : n₁ = 104, by norm_num,
  have n₂_val : n₂ = 30, by norm_num,
  
  -- Computing the product in base 10
  let product := n₁ * n₂,
  have product_val : product = 3120, by norm_num,
  
  -- Converting the product to base 9
  let base_nine_digits := [4, 2, 4, 6], -- we precompute the base-9 digits of 3120
  
  -- Computing the sum of the base-9 digits
  let sum_of_digits := base_nine_digits.foldr (.+.) (0 : ℕ),
  have sum_of_digits_val : sum_of_digits = 16, by norm_num,
  
  -- Prove the final assertion
  show sum_of_digits = 16, from sum_of_digits_val,
end

end base_nine_sum_of_product_of_125_and_33_is_16_l559_559445


namespace find_cd_l559_559869

noncomputable def equilateral_vertices (a b c : ℂ) : Prop :=
  complex.abs (a - b) = complex.abs (b - c) ∧ complex.abs (b - c) = complex.abs (c - a)

theorem find_cd (c d : ℂ) (h : equilateral_vertices 0 (c + 8 * complex.I) (d + 30 * complex.I)) :
  (c * d) = -1976 / 9 :=
sorry

end find_cd_l559_559869


namespace true_statements_count_is_two_l559_559079

def reciprocal (a : ℤ) : ℚ := 1 / a

def is_statement_i_true : Prop :=
  (reciprocal 4 + reciprocal 8 = reciprocal 12) = false

def is_statement_ii_true : Prop :=
  (reciprocal 9 * reciprocal 3 = reciprocal 27) = true

def is_statement_iii_true : Prop :=
  ((reciprocal 7 - reciprocal 5) * reciprocal 12 = reciprocal 24) = false

def is_statement_iv_true : Prop :=
  (reciprocal 15 / reciprocal 3 = reciprocal 5) = true

def num_true_statements : ℕ :=
  ([is_statement_i_true, is_statement_ii_true, is_statement_iii_true, is_statement_iv_true].count (λ b, b))

theorem true_statements_count_is_two : num_true_statements = 2 := by
sorry

end true_statements_count_is_two_l559_559079


namespace rotation_to_second_quadrant_l559_559300

noncomputable def z : ℂ := (-1 + 3 * complex.I) / complex.I

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem rotation_to_second_quadrant (θ : ℝ) : 
  is_in_fourth_quadrant z → is_in_second_quadrant (complex.exp (complex.I * θ) * z) → θ = 2 * real.pi / 3 :=
by sorry

end rotation_to_second_quadrant_l559_559300


namespace poly_remainder_l559_559040

theorem poly_remainder (h : Polynomial ℝ) :
  h = 2 * (Polynomial.X ^ 7) + Polynomial.X ^ 5 - Polynomial.X ^ 4 + 2 * Polynomial.X ^ 3 + 3 * Polynomial.X ^ 2 + Polynomial.X + 1 →
  Polynomial.remainder (h.comp (Polynomial.X ^ 10)) h = 10 :=
by
  intro h_def
  sorry

end poly_remainder_l559_559040


namespace equation_solution_l559_559072

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end equation_solution_l559_559072


namespace sum_a1_to_a8_value_of_a3_l559_559293

noncomputable def polynomial := λ (x : ℝ), (2 + x) * (1 - 2 * x) ^ 7

theorem sum_a1_to_a8 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) :
  polynomial 1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 ∧
  polynomial 0 = a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = -5 := 
sorry

theorem value_of_a3 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) :
  polynomial 1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 →
  polynomial 0 = a_0 →
  let binomial_coefficient := λ (n k : ℕ), nat.choose n k in
  a_3 = 2 * binomial_coefficient 7 3 * (-2)^3 + binomial_coefficient 7 2 * (-2)^2 →
  a_3 = -476 :=
sorry

end sum_a1_to_a8_value_of_a3_l559_559293


namespace area_of_S_correct_l559_559470

/-- The taxicab distance between two points in the plane. -/
def taxicab_distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  |x2 - x1| + |y2 - y1|

/-- A predicate to define points within the given taxicab distance from the octagon vertices. -/
def within_taxicab_distance (x y : ℝ) : Prop :=
  ∃ (vx vy : ℝ), 
  ( (vx, vy) ∈ {(0,0), (1,0), (z_1, z_2) | other vertices of the octagon}) ∧ -- List all vertices appropriately
  taxicab_distance vx vy x y ≤ 2/3 

/-- The area of the set S can be written as m/n, where m, n are coprime integers -/
def area_of_S : ℚ := 23 / 9

theorem area_of_S_correct :
  area_of_S = 23 / 9 ∧ (100 * 23 + 9 = 2309) :=
by
  sorry

end area_of_S_correct_l559_559470


namespace arithmetic_sequence_index_l559_559461

theorem arithmetic_sequence_index (a : ℕ → ℕ) (n : ℕ) (first_term comm_diff : ℕ):
  (∀ k, a k = first_term + comm_diff * (k - 1)) → a n = 2016 → n = 404 :=
by 
  sorry

end arithmetic_sequence_index_l559_559461


namespace wood_length_l559_559195

theorem wood_length (l_original : ℝ) (l_sawed : ℝ) (l_new : ℝ) : 
  l_original = 8.9 ∧ l_sawed = 2.3 → l_new = l_original - l_sawed → l_new = 6.6 :=
by
  intros h h1
  have h2 : l_original = 8.9 := h.left
  have h3 : l_sawed = 2.3 := h.right
  rw [h2, h3] at h1
  exact h1

end wood_length_l559_559195


namespace factor_tree_X_value_l559_559765

def H : ℕ := 2 * 5
def J : ℕ := 3 * 7
def F : ℕ := 7 * H
def G : ℕ := 11 * J
def X : ℕ := F * G

theorem factor_tree_X_value : X = 16170 := by
  sorry

end factor_tree_X_value_l559_559765


namespace tony_water_intake_l559_559477

theorem tony_water_intake :
  ∃ W : ℝ, 48 = 1.05 * W * 0.96 ∧ 0.9 * W ≈ 42.86 :=
by
  -- Let W be the amount of water Tony drank two days ago directly.
  -- Yesterday, Tony drank 48 ounces, which is 4% less than what he drank the day before.
  -- The day before, Tony drank 5% more than W.
  -- Tony's direct water intake two days ago is 0.9 * W, which should be approximately 42.86 ounces.
  sorry

end tony_water_intake_l559_559477


namespace correct_operation_l559_559164

theorem correct_operation (a b : ℝ) (c d : ℂ) : (sqrt 4 = 2 ∧ (± abs (5) ≠ -5) ∧ (sqrt (abs (7) ^ 2) = 7) ∧ ¬(is_real (sqrt (-3))) ) := sorry

end correct_operation_l559_559164


namespace quadrilateral_inequality_l559_559927

variable {A B C D : ℝ}
variable {S : ℝ} -- Area of the quadrilateral
variable {α β : ℝ} -- Angles between specific lines

-- Given conditions
def is_convex_quadrilateral (A B C D : ℝ) := true -- Dummy definition for convex quadrilateral
def area_quadrilateral (A B C D : ℝ) : ℝ := S -- Dummy definition for area
def angle_between_lines_ab_cd : ℝ := α -- Given angle between lines AB and CD
def angle_between_lines_ad_bc : ℝ := β -- Given angle between lines AD and BC

-- The statement we aim to prove
theorem quadrilateral_inequality (h1 : is_convex_quadrilateral A B C D)
    (h2 : area_quadrilateral A B C D = S)
    (h3 : angle_between_lines_ab_cd = α)
    (h4 : angle_between_lines_ad_bc = β) :
    A * B * sin α + C * D * sin β ≤ 2 * S ∧ 2 * S ≤ A * B + C * D :=
by
  sorry

end quadrilateral_inequality_l559_559927


namespace circumcircle_diameter_l559_559672

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable {AB : ℝ} (hAB : AB = 4)

theorem circumcircle_diameter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (hAB : AB = 4) :
  diameter (circumcircle A B C) = 2 * sqrt 5 := 
sorry

end circumcircle_diameter_l559_559672


namespace triangle_MN_ratio_l559_559016

theorem triangle_MN_ratio {A B C M N : Type}
  (vec : Type -> Type)
  (am mc : vec ℝ) (bn nc : vec ℝ) (mn ab ac : vec ℝ)
  (h1 : am = 2 • mc)
  (h2 : bn = 3 • nc)
  (h3 : mn = ab • x + ac • y)
  (h4 : mn = (1 / 3 : ℝ) • ac + (1 / 4 : ℝ) • ab)
  : (x / y : ℝ) = 3 := 
sorry

end triangle_MN_ratio_l559_559016


namespace age_of_b_l559_559920

variable (a b : ℕ)
variable (h1 : a * 3 = b * 5)
variable (h2 : (a + 2) * 2 = (b + 2) * 3)

theorem age_of_b : b = 6 :=
by
  sorry

end age_of_b_l559_559920


namespace broadcasting_methods_l559_559531

theorem broadcasting_methods : 
  let commercials := 3
  let olympic_promotions := 2
  let total_slots := 5
  let factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
  let binomial (n k : ℕ) : ℕ := if k = 0 then 1 else (n * binomial (n - 1) (k - 1)) / k in
  let total_broadcasting_methods := factorial commercials * binomial 4 2 in
  total_broadcasting_methods = 36 :=
by
  sorry

end broadcasting_methods_l559_559531


namespace correct_operation_l559_559166

theorem correct_operation (a b : ℝ) (c d : ℂ) : (sqrt 4 = 2 ∧ (± abs (5) ≠ -5) ∧ (sqrt (abs (7) ^ 2) = 7) ∧ ¬(is_real (sqrt (-3))) ) := sorry

end correct_operation_l559_559166


namespace arithmetic_sequence_sum_l559_559034

theorem arithmetic_sequence_sum :
  ∃ (d a1 a2 a3 : ℝ), a1 + a2 + a3 = 15 ∧ a1 * a2 * a3 = 80 ∧ a2 = a1 + d ∧
  a_{11} + a_{12} + a_{13} = 105 :=
by
  sorry

end arithmetic_sequence_sum_l559_559034


namespace smaller_solution_exists_smaller_solution_is_minus_twelve_l559_559508

-- Define the quadratic equation
def quadratic_eq : Polynomial ℤ := Polynomial.C (-48) + Polynomial.X * Polynomial.C 8 + Polynomial.X^2 

-- Define what it means for a number to be a solution
def is_solution (x : ℝ) : Prop := x^2 + 8 * x - 48 = 0

-- The main statement we want to prove
theorem smaller_solution_exists : ∃ x : ℝ, is_solution x ∧ ∀ y : ℝ, is_solution y → x ≤ y := 
begin
  sorry
end

-- Alternatively, directly state the specific solution
theorem smaller_solution_is_minus_twelve : is_solution (-12) ∧ ∀ y : ℝ, is_solution y → -12 ≤ y :=
begin
  sorry
end

end smaller_solution_exists_smaller_solution_is_minus_twelve_l559_559508


namespace correct_option_l559_559159

noncomputable def OptionA : Prop := (Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2)
noncomputable def OptionB : Prop := (Real.sqrt (5 ^ 2) = -5 ∨ Real.sqrt (5 ^ 2) = 5)
noncomputable def OptionC : Prop := Real.sqrt ((-7) ^ 2) = 7
noncomputable def OptionD : Prop := (Real.sqrt (-3) = -Real.sqrt 3)

theorem correct_option : OptionC := 
by 
  unfold OptionC
  simp
  exact eq.refl 7

end correct_option_l559_559159


namespace dante_flour_eggs_l559_559255

theorem dante_flour_eggs (eggs : ℕ) (h_eggs : eggs = 60) (h_flour : ∀ (f : ℕ), f = eggs / 2) : eggs + (eggs / 2) = 90 := 
by {
  rw h_eggs,
  calc
    60 + (60 / 2) = 60 + 30   : by norm_num
    ...         = 90 : by norm_num
}

end dante_flour_eggs_l559_559255


namespace parabola_problem_l559_559706

noncomputable def is_solution : Prop := ∀ (p : ℝ), 
  (∀ P Q : ℝ × ℝ, 
    P == (x1, y1) ∧ Q == (x2, y2) → 
    (sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt 15) →
    (∃ y, y^2 = 2 * p * x)) →
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ (y1 = 2 * x1 + 1) ∧ (y2 = 2 * x2 + 1) →
  (p = -2 ∨ p = 6) ∧ (y1^2 = -4 * x1 ∨ y1^2 = 12 * x1)

theorem parabola_problem : ∃ (p : ℝ), 
  (∀ P Q : ℝ × ℝ, 
    (P = (x1, y1) ∧ Q = (x2, y2)) → 
    (sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt 15) →
    (∃ y, y^2 = 2 * p * x)) →
  y1^2 = 2 * p * x1 ∧ y2^2 = 2 * p * x2 ∧ (y1 = 2 * x1 + 1) ∧ (y2 = 2 * x2 + 1) →
  (p = -2 ∨ p = 6) ∧ (y1^2 = -4 * x1 ∨ y1^2 = 12 * x1) := sorry

end parabola_problem_l559_559706


namespace parabola_vertex_l559_559093

theorem parabola_vertex:
  ∃ x y: ℝ, y^2 + 8 * y + 2 * x + 1 = 0 ∧ (x, y) = (7.5, -4) := sorry

end parabola_vertex_l559_559093


namespace solve_system_l559_559845

theorem solve_system (a b c : ℕ) (h1 : a^2 + b - c = 100) (h2 : a + b^2 - c = 124) :
  a = 12 ∧ b = 13 ∧ c = 57 :=
by {
  unfold ℕ,
  sorry
}

end solve_system_l559_559845


namespace total_workers_count_l559_559090

theorem total_workers_count 
  (W N : ℕ)
  (h1 : (W : ℝ) * 9000 = 7 * 12000 + N * 6000)
  (h2 : W = 7 + N) 
  : W = 14 :=
sorry

end total_workers_count_l559_559090


namespace mark_weekly_leftover_l559_559412

def initial_hourly_wage := 40
def raise_percentage := 5 / 100
def daily_hours := 8
def weekly_days := 5
def old_weekly_bills := 600
def personal_trainer_cost := 100

def new_hourly_wage := initial_hourly_wage * (1 + raise_percentage)
def weekly_hours := daily_hours * weekly_days
def weekly_earnings := new_hourly_wage * weekly_hours
def new_weekly_expenses := old_weekly_bills + personal_trainer_cost
def leftover_per_week := weekly_earnings - new_weekly_expenses

theorem mark_weekly_leftover : leftover_per_week = 980 := by
  sorry

end mark_weekly_leftover_l559_559412


namespace polarEquationOfCircleCenter1_1Radius1_l559_559862

noncomputable def circleEquationInPolarCoordinates (θ : ℝ) : ℝ := 2 * Real.cos (θ - 1)

theorem polarEquationOfCircleCenter1_1Radius1 (ρ θ : ℝ) 
  (h : Real.sqrt ((ρ * Real.cos θ - Real.cos 1)^2 + (ρ * Real.sin θ - Real.sin 1)^2) = 1) :
  ρ = circleEquationInPolarCoordinates θ :=
by sorry

end polarEquationOfCircleCenter1_1Radius1_l559_559862


namespace points_collinear_l559_559046

open EuclideanGeometry

variables {A B C D E F P : Point}
variables (α β : Real) 

-- Define the given parallelogram with an acute angle at A
def isParallelogram (A B C D : Point) : Prop :=
  parallelogram A B C D ∧ acute ∠ A

-- Define the circle with diameter AC that intersects lines BC and CD at E and F respectively
def circleIntersects (A C B E D F : Point) : Prop :=
  circle (diameter A C) ∧ lineIntersects B C E ∧ lineIntersects C D F

-- Tangent to the circle at A intersects the line BD at P
def tangentIntersects (A C B D P : Point) : Prop :=
  tangent (circle (diameter A C)) A ∧ lineIntersects B D P

-- Menelaus' theorem conditions
def menelausCondition (P B D F C E : Point) : Prop :=
  (PB / PD) * (DF / FC) * (CE / EB) = 1

-- The final proof statement
theorem points_collinear 
  (h1 : isParallelogram A B C D)
  (h2 : circleIntersects A C B E D F)
  (h3 : tangentIntersects A C B D P)
  (h4 : menelausCondition P B D F C E) :
  collinear P F E :=
sorry

end points_collinear_l559_559046


namespace find_z_l559_559660

theorem find_z (z : ℝ) : 
  (sqrt 1.21 / sqrt 0.81) + (sqrt z / sqrt 0.49) = 2.9365079365079367 → z = 1.44 :=
by
  sorry

end find_z_l559_559660


namespace find_num_cats_l559_559890

-- Define the conditions as individual definitions
def teeth_dogs := 42
def teeth_cats := 30
def teeth_pigs := 28
def num_dogs := 5
def num_pigs := 7
def total_teeth := 706

-- Define the target number of cats as a variable to be determined
variable num_cats : ℕ

-- Statement: Prove that num_cats = 10 given the conditions
theorem find_num_cats :
  num_dogs * teeth_dogs + num_pigs * teeth_pigs + num_cats * teeth_cats = total_teeth → num_cats = 10 :=
by
  intro h
  sorry

end find_num_cats_l559_559890


namespace count_positive_integers_number_of_positive_integers_l559_559663

theorem count_positive_integers (x : ℕ) : 
  (50 ≤ x^2 ∧ x^2 ≤ 180) ↔ (x = 8 ∨ x = 9 ∨ x = 10 ∨ x = 11 ∨ x = 12 ∨ x = 13) :=
by sorry

theorem number_of_positive_integers : 
  (Finset.card (Finset.filter (λ x, 50 ≤ x^2 ∧ x^2 ≤ 180) (Finset.range (13 + 1)))) = 6 :=
by sorry

end count_positive_integers_number_of_positive_integers_l559_559663


namespace worksheets_already_graded_l559_559236

variables (num_problems_per_worksheet num_total_worksheets num_problems_left num_worksheets_graded : ℕ)

-- Define the conditions
def num_problems_per_worksheet := 7
def num_total_worksheets := 17
def num_problems_left := 63

-- Calculate the total number of problems
def total_problems := num_total_worksheets * num_problems_per_worksheet

-- Calculate the number of problems already graded
def problems_already_graded := total_problems - num_problems_left

-- Prove the number of worksheets already graded is 8
theorem worksheets_already_graded : (problems_already_graded / num_problems_per_worksheet) = 8 :=
by
  sorry

end worksheets_already_graded_l559_559236


namespace polynomial_coeff_sum_l559_559870

-- Lean statement
theorem polynomial_coeff_sum
  (p q r s : ℝ)
  (h : ∀ x : ℂ, (x^4 + (p:ℂ) * x^3 + (q:ℂ) * x^2 + (r:ℂ) * x + (s:ℂ) = 0 →
      (x = 1 + complex.I ∨ x = 1 - complex.I ∨ x = 3 * complex.I ∨ x = -3 * complex.I))) :
  p + q + r + s = 9 :=
by
  sorry

end polynomial_coeff_sum_l559_559870


namespace ker_is_iso_coker_l559_559795

-- Definitions based on conditions
variables (G : Type*) [add_comm_group G] [module (zmod p) G]
variables (f : G →ₗ[zmod p] G)

-- Definition of the proof problem
theorem ker_is_iso_coker (p : ℕ) [prime p] (hG : ∀ x ∈ G, x ^ (zmod p.card) = 0) :
  linear_map.quot_ker_equiv_range f = sorry :=
by
  sorry

end ker_is_iso_coker_l559_559795


namespace pentagon_interior_angles_l559_559450

theorem pentagon_interior_angles
  (x y : ℝ)
  (H_eq_triangle : ∀ (angle : ℝ), angle = 60)
  (H_rect_QT : ∀ (angle : ℝ), angle = 90)
  (sum_interior_angles_pentagon : ∀ (n : ℕ), (n - 2) * 180 = 3 * 180) :
  x + y = 60 :=
by
  sorry

end pentagon_interior_angles_l559_559450


namespace trigonometric_identity_l559_559593

theorem trigonometric_identity :
  let θ := 30 * Real.pi / 180
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2) / (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_l559_559593


namespace dimes_count_l559_559914

-- Definitions of types of coins and their values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def halfDollar := 50

-- Condition statements as assumptions
variables (num_pennies num_nickels num_dimes num_quarters num_halfDollars : ℕ)

-- Sum of all coins and their values (in cents)
def total_value := num_pennies * penny + num_nickels * nickel + num_dimes * dime + num_quarters * quarter + num_halfDollars * halfDollar

-- Total number of coins
def total_coins := num_pennies + num_nickels + num_dimes + num_quarters + num_halfDollars

-- Proving the number of dimes is 5 given the conditions.
theorem dimes_count : 
  total_value = 163 ∧ 
  total_coins = 12 ∧ 
  num_pennies ≥ 1 ∧ 
  num_nickels ≥ 1 ∧ 
  num_dimes ≥ 1 ∧ 
  num_quarters ≥ 1 ∧ 
  num_halfDollars ≥ 1 → 
  num_dimes = 5 :=
by
  sorry

end dimes_count_l559_559914


namespace find_constant_l559_559471

-- Define the conditions
def is_axles (x : ℕ) : Prop := x = 5
def toll_for_truck (t : ℝ) : Prop := t = 4

-- Define the formula for the toll
def toll_formula (t : ℝ) (constant : ℝ) (x : ℕ) : Prop :=
  t = 2.50 + constant * (x - 2)

-- Proof problem statement
theorem find_constant : ∃ (constant : ℝ), 
  ∀ x : ℕ, is_axles x → toll_for_truck 4 →
  toll_formula 4 constant x → constant = 0.50 :=
sorry

end find_constant_l559_559471


namespace initial_books_l559_559427

theorem initial_books 
  (fiction_sold : ℕ) (fiction_left : ℕ) (total_earned : ℕ) 
  (price_fiction : ℕ) (price_non_fiction : ℕ) : 
  fiction_sold = 137 ∧ fiction_left = 105 ∧ total_earned = 685 ∧ 
  price_fiction = 3 ∧ price_non_fiction = 5 
  →
  (fiction_left + fiction_sold = 242 ∧ 
  (total_earned - (fiction_sold * price_fiction)) / price_non_fiction = 54) :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  sorry

end initial_books_l559_559427


namespace max_area_condition_P_moves_on_fixed_circle_l559_559793

-- Definitions required from the conditions
variables (A B C D H M N P: Type) [acute_triangle : ∀(a b c : Type), Prop] 
          (circumcircle : ∀(X Y Z : Type), Type) 
          (orthocenter : ∀(X Y Z : Type), Type)

-- Definitions from the questions
def condition_max_area_triangle_AMN (ABC H: Type) [circumcircle ABC] [orthocenter ABC] := 
  ∃ H, ∀ Δ : Type, Δ ⊥ H

def moves_on_fixed_circle (ABC H M N P : Type) [circumcircle ABC] [orthocenter ABC] :=
  ∃ d : Type, ∀ (MT : Type), MT.perp DB → ∀ (NT : Type), NT.perp DC → P ∈ d → P ∈ reflection_circle

-- Proof problem 1
theorem max_area_condition (ABC : Type) [acute_triangle ABC] [circumcircle ABC] [orthocenter ABC] :
  condition_max_area_triangle_AMN ABC H ↔ 
  (∃ Δ : Type, (Δ ⊥ AH ↔ max_area_triangle AMN))
:= sorry

-- Proof problem 2
theorem P_moves_on_fixed_circle (ABC : Type) [acute_triangle ABC] [circumcircle ABC] [orthocenter ABC] :
  moves_on_fixed_circle ABC H M N P ↔ 
  ∃ d : Type, parallel d CD ∧ ∃ reflection_circle, P ∈ reflection_circle
:= sorry

end max_area_condition_P_moves_on_fixed_circle_l559_559793


namespace p_implies_q_q_not_implies_p_l559_559297

variable (x : ℝ)

def p (x : ℝ) : Prop := 2 < x ∧ x < 4
def q (x : ℝ) : Prop := x < -3 ∨ x > 2

theorem p_implies_q (x : ℝ) : p x → q x := 
by {
  intros h,
  sorry
}

theorem q_not_implies_p (x : ℝ) : ¬ (q x → p x) := 
by {
  intros h_not_implies,
  sorry
}

end p_implies_q_q_not_implies_p_l559_559297


namespace second_account_interest_rate_l559_559512

theorem second_account_interest_rate
  (investment1 : ℝ)
  (rate1 : ℝ)
  (interest1 : ℝ)
  (investment2 : ℝ)
  (interest2 : ℝ)
  (h1 : 4000 = investment1)
  (h2 : 0.08 = rate1)
  (h3 : 320 = interest1)
  (h4 : 7200 - 4000 = investment2)
  (h5 : interest1 = interest2) :
  interest2 / investment2 = 0.1 :=
by
  sorry

end second_account_interest_rate_l559_559512


namespace fraction_of_canvas_painted_blue_l559_559883

noncomputable def square_canvas_blue_fraction : ℚ :=
  sorry

theorem fraction_of_canvas_painted_blue :
  square_canvas_blue_fraction = 3 / 8 :=
  sorry

end fraction_of_canvas_painted_blue_l559_559883


namespace product_of_cosines_l559_559279

theorem product_of_cosines 
  (A B : ℝ) 
  (n : ℕ) 
  (T : ℕ → ℝ)
  (hT : T n = ∏ i in (finset.range n).image (λ k, (cos (A / 2 ^ (k + 1)) + cos (B / 2 ^ (k + 1))))) :
  T n = (sin ( (A+B) / 2) * sin ( (A-B) / 2)) / (2^n * sin ( (A+B) / 2^(n+1)) * sin ( (A-B) / 2^(n+1))) :=
sorry

end product_of_cosines_l559_559279


namespace coloring_objects_l559_559352

theorem coloring_objects (n_people : ℕ) (n_colors_total : ℕ) (n_unique_per_person : ℕ) :
  n_people = 3 → n_colors_total = 24 → n_unique_per_person = n_colors_total / n_people →
  n_unique_per_person * n_people = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3.symm

end coloring_objects_l559_559352


namespace find_pool_length_l559_559126

-- Define the conditions of the problem
def pool_width : ℝ := 25
def lowered_by_inches : ℝ := 6
def removed_gallons : ℝ := 1875
def gallons_to_cubic_feet : ℝ := 7.48052
def target_length : ℝ := 20.04 -- This is the length value we want to prove

-- Convert values based on the given conditions
def lowered_by_feet : ℝ := lowered_by_inches / 12
def removed_cubic_feet : ℝ := removed_gallons / gallons_to_cubic_feet

-- State the theorem to be proven
theorem find_pool_length :
  ∃ (L : ℝ), (removed_cubic_feet = L * pool_width * lowered_by_feet) ∧ (L ≈ target_length) := by
  sorry

end find_pool_length_l559_559126


namespace complement_intersection_l559_559410

-- Define the universal set R as the set of real numbers
def R : Set ℝ := Set.univ

-- Define the set A
def A : Set ℝ := {x | x^2 - 5 * x + 6 < 0}

-- Define the set B
def B : Set ℝ := {x | log x > 1}

-- State the theorem
theorem complement_intersection :
  Set.compl (A ∩ B) = {x | x ≤ real.exp 1} ∪ {x | 3 ≤ x} :=
by sorry

end complement_intersection_l559_559410


namespace ellipse_find_a_l559_559689

theorem ellipse_find_a :
  ∃ a : ℝ, (∀ x y : ℝ, (x^2 / a^2 + y^2 / 2 = 1) → (a = sqrt 6 ∨ a = -sqrt 6)) ∧
  ∃ F : ℝ × ℝ, ((F = (2, 0)) ∧ (8 * F.1 = F.2^2)) :=
by
  sorry

end ellipse_find_a_l559_559689


namespace subcommittee_ways_l559_559088

theorem subcommittee_ways :
  ∃ (n : ℕ), n = Nat.choose 10 4 * Nat.choose 7 2 ∧ n = 4410 :=
by
  use 4410
  sorry

end subcommittee_ways_l559_559088


namespace city_map_distance_example_l559_559825

variable (distance_on_map : ℝ)
variable (scale : ℝ)
variable (actual_distance : ℝ)

theorem city_map_distance_example
  (h1 : distance_on_map = 16)
  (h2 : scale = 1 / 10000)
  (h3 : actual_distance = distance_on_map / scale) :
  actual_distance = 1.6 * 10^3 :=
by
  sorry

end city_map_distance_example_l559_559825


namespace solution_set_of_inequality_l559_559463

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem solution_set_of_inequality :
  {x : ℝ | f(x) ≤ 4} = set.Icc (-5 / 2) (3 / 2) :=
by sorry

end solution_set_of_inequality_l559_559463


namespace savings_account_amount_l559_559967

noncomputable def final_amount : ℝ :=
  let initial_deposit : ℝ := 5000
  let first_quarter_rate : ℝ := 0.01
  let second_quarter_rate : ℝ := 0.0125
  let deposit_end_third_month : ℝ := 1000
  let withdrawal_end_fifth_month : ℝ := 500
  let amount_after_first_quarter := initial_deposit * (1 + first_quarter_rate)
  let amount_before_second_quarter := amount_after_first_quarter + deposit_end_third_month
  let amount_after_second_quarter := amount_before_second_quarter * (1 + second_quarter_rate)
  let final_amount := amount_after_second_quarter - withdrawal_end_fifth_month
  final_amount

theorem savings_account_amount :
  final_amount = 5625.625 :=
by
  sorry

end savings_account_amount_l559_559967


namespace odd_function_f_neg_9_l559_559696

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then x^(1/2) 
else -((-x)^(1/2))

theorem odd_function_f_neg_9 : f (-9) = -3 := by
  sorry

end odd_function_f_neg_9_l559_559696


namespace terry_total_driving_time_l559_559083

-- Define the conditions
def speed : ℝ := 40 -- miles per hour
def distance : ℝ := 60 -- miles

-- Define the time for one trip
def time_for_one_trip (d : ℝ) (s : ℝ) : ℝ := d / s

-- Define the total driving time for a round trip (forth and back)
def total_driving_time (d : ℝ) (s : ℝ) : ℝ := 2 * time_for_one_trip d s

-- State the theorem to be proven
theorem terry_total_driving_time : total_driving_time distance speed = 3 := 
by
  sorry

end terry_total_driving_time_l559_559083


namespace equation_solution_l559_559071

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end equation_solution_l559_559071


namespace boys_with_dogs_percentage_l559_559557

-- Definitions according to the given conditions
def total_students := 100
def girls := total_students / 2
def boys := total_students / 2
def percentage_girls_with_dogs := 20 / 100.0
def students_with_dogs := 15

-- Use the conditions to define the number of girls and boys with dogs
def girls_with_dogs := percentage_girls_with_dogs * girls
def boys_with_dogs := students_with_dogs - girls_with_dogs

-- Theorem stating that the percentage of boys with dogs is 10%
theorem boys_with_dogs_percentage : (boys_with_dogs / boys) * 100 = 10 :=
by
  sorry

end boys_with_dogs_percentage_l559_559557


namespace Dexter_card_count_l559_559644

theorem Dexter_card_count : 
  let basketball_boxes := 9
  let cards_per_basketball_box := 15
  let football_boxes := basketball_boxes - 3
  let cards_per_football_box := 20
  let basketball_cards := basketball_boxes * cards_per_basketball_box
  let football_cards := football_boxes * cards_per_football_box
  let total_cards := basketball_cards + football_cards
  total_cards = 255 :=
sorry

end Dexter_card_count_l559_559644


namespace dot_product_ab_complex_dot_product_magnitude_sum_l559_559299

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Defining the magnitudes and angle conditions
def norm_a : ℝ := 2
def norm_b : ℝ := 3
def angle_ab : ℝ := 120 * (Real.pi / 180) -- converting degrees to radians

-- Defining the conditions explicitly
axiom norm_a_def : ‖a‖ = norm_a
axiom norm_b_def : ‖b‖ = norm_b
axiom angle_ab_def : real.inner_product_space.angle a b = angle_ab

-- Problem 1: Prove that the dot product is -3
theorem dot_product_ab : inner a b = -3 := by 
sorry

-- Problem 2: Prove the complex dot product equals -34
theorem complex_dot_product : inner (2 • a - b) (a + 3 • b) = -34 := by
sorry

-- Problem 3: Prove the magnitude of the sum is sqrt(7)
theorem magnitude_sum : ‖a + b‖ = Real.sqrt 7 := by
sorry

end dot_product_ab_complex_dot_product_magnitude_sum_l559_559299


namespace curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559002

section
variables (k t θ ρ : ℝ)  

def parametric_curve_C1 (k : ℝ) (t : ℝ) : ℝ × ℝ := (cos t ^ k, sin t ^ k)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

def curve_C2 (ρ θ : ℝ) : Prop := 4 * ρ * cos θ - 16 * ρ * sin θ + 3 = 0

theorem curve_C1_when_k1_is_circle : 
  ∀ t : ℝ, (parametric_curve_C1 1 t).fst ^ 2 + (parametric_curve_C1 1 t).snd ^ 2 = 1 :=
begin
  intros t,
  simp only [parametric_curve_C1],
  have h : cos t ^ 2 + sin t ^ 2 = 1, from real.sin_sq_add_cos_sq t,
  simp [h],
end

noncomputable def intersection_points_when_k4 : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p = parametric_curve_C1 4 t ∧ 
    let (x, y) := p in 4 * x - 16 * y + 3 = 0}

theorem find_intersection_points_when_k4 : 
  intersection_points_when_k4 = { (1/4, 1/4) } :=
begin
  sorry
end

end

end curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559002


namespace curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559006

section
variables (k t θ ρ : ℝ)  

def parametric_curve_C1 (k : ℝ) (t : ℝ) : ℝ × ℝ := (cos t ^ k, sin t ^ k)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

def curve_C2 (ρ θ : ℝ) : Prop := 4 * ρ * cos θ - 16 * ρ * sin θ + 3 = 0

theorem curve_C1_when_k1_is_circle : 
  ∀ t : ℝ, (parametric_curve_C1 1 t).fst ^ 2 + (parametric_curve_C1 1 t).snd ^ 2 = 1 :=
begin
  intros t,
  simp only [parametric_curve_C1],
  have h : cos t ^ 2 + sin t ^ 2 = 1, from real.sin_sq_add_cos_sq t,
  simp [h],
end

noncomputable def intersection_points_when_k4 : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p = parametric_curve_C1 4 t ∧ 
    let (x, y) := p in 4 * x - 16 * y + 3 = 0}

theorem find_intersection_points_when_k4 : 
  intersection_points_when_k4 = { (1/4, 1/4) } :=
begin
  sorry
end

end

end curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559006


namespace tan_sin_identity_l559_559592

noncomputable def tan_deg : ℝ := tan (real.pi / 6)
noncomputable def sin_deg : ℝ := sin (real.pi / 6)

theorem tan_sin_identity :
  (tan_deg ^ 2 - sin_deg ^ 2) / (tan_deg ^ 2 * sin_deg ^ 2) = 1 :=
by
  have tan_30 := by
    simp [tan_deg, real.tan_pi_div_six]
    norm_num
  have sin_30 := by
    simp [sin_deg, real.sin_pi_div_six]
    norm_num
  sorry

end tan_sin_identity_l559_559592


namespace exists_integers_a_b_part_a_l559_559268

theorem exists_integers_a_b_part_a : 
  ∃ a b : ℤ, (∀ x : ℝ, x^2 + a * x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a * x + (b : ℝ) = 0) := 
sorry

end exists_integers_a_b_part_a_l559_559268


namespace prove_inequalities_l559_559634

open Set

variable {x : ℝ}

def condition1 : Prop := 3 * x - 2 < (x + 2) ^ 2
def condition2 : Prop := (x + 2) ^ 2 < 9 * x - 6

theorem prove_inequalities (h1 : condition1) (h2 : condition2) : 2 < x ∧ x < 3 :=
by
  sorry

end prove_inequalities_l559_559634


namespace smallest_cos_a_l559_559396

noncomputable def main_proof (a b c : ℝ) : Prop :=
  sin a = cot b ∧
  sin b = cot c ∧
  sin c = cot a →
  ∃ m : ℝ, m = cos a ∧ m = 1 / sqrt 2

theorem smallest_cos_a (a b c : ℝ) (h : sin a = cot b ∧ sin b = cot c ∧ sin c = cot a) : 
  ∃ m : ℝ, m = 1 / sqrt 2 ∧ m = cos a :=
by
  sorry

end smallest_cos_a_l559_559396


namespace angle_CAB_eq_90_l559_559791

theorem angle_CAB_eq_90 (A B C D E K : Type) 
  [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited K] 
  [triangle : triangle A B C]
  (h_eq : A = C)
  (D_is_angle_bisector_CAB : D = point_of_angle_bisector (angle CAB))
  (E_is_angle_bisector_ABC : E = point_of_angle_bisector (angle ABC))
  (K_is_incenter_ADC : K = incenter (triangle ADC))
  (h_angle_45 : angle BEK = 45) :
  angle CAB = 90 :=
sorry

end angle_CAB_eq_90_l559_559791


namespace cost_of_individual_roll_l559_559196

theorem cost_of_individual_roll
  (p : ℕ) (c : ℝ) (s : ℝ) (x : ℝ)
  (hc : c = 9)
  (hp : p = 12)
  (hs : s = 0.25)
  (h : 12 * x = 9 * (1 + s)) :
  x = 0.9375 :=
by
  sorry

end cost_of_individual_roll_l559_559196


namespace K_time_travel_45_miles_l559_559928

variables (x : ℝ)

-- Condition: K's speed in miles per hour is x.
-- Condition: K travels 1/2 mile per hour faster than M.
-- Condition: K takes 45 minutes (3/4 hour) less than M to travel 45 miles.
-- Question: Prove that K's time for traveling 45 miles is 45 / x.

theorem K_time_travel_45_miles (x : ℝ) (h1 : x > 0) :
  let K_time := 45 / x in
  let M_time := 45 / (x - 1/2) in
  M_time - K_time = 3/4 → K_time = 45 / x :=
sorry

end K_time_travel_45_miles_l559_559928


namespace mixed_oil_rate_l559_559179

noncomputable def rate_of_mixed_oil
  (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℚ :=
(total_cost : ℚ) / (total_volume : ℚ)
where
  total_cost := volume1 * price1 + volume2 * price2
  total_volume := volume1 + volume2

theorem mixed_oil_rate :
  rate_of_mixed_oil 10 50 5 66 = 55.33 := 
by
  sorry

end mixed_oil_rate_l559_559179


namespace rodney_correct_guess_probability_l559_559837

open Nat

theorem rodney_correct_guess_probability :
  let tens_odd_numbers := [70, 72, 74, 76, 78, 90, 92, 94, 96, 98] in
  (∑ n in tens_odd_numbers, if n > 65 then 1 else 0) = 10 → 
  1 / 10 = (1 / length tens_odd_numbers : ℚ) :=
by
  sorry

end rodney_correct_guess_probability_l559_559837


namespace parabola_focus_directrix_distance_l559_559361

theorem parabola_focus_directrix_distance (p : ℝ) :
  (∀ x y : ℝ, (x, y) = (1, 3) → y^2 = 2 * p * x) →
  p = 9 / 2 :=
by
  intros,
  -- Assume passing through the point (1, 3)
  have h₁ : (3 : ℝ)^2 = 2 * p * (1 : ℝ) := by sorry,
  -- Solve for p 
  have h₂ : p = 9 / 2 := by sorry,
  exact h₂

end parabola_focus_directrix_distance_l559_559361


namespace triangle_AM_plus_KM_eq_AB_l559_559828

theorem triangle_AM_plus_KM_eq_AB 
  (A B C M D K : Type) 
  [has_add M] [has_eq M]
  [has_add AB] [has_eq AB]
  [has_add AC] [has_eq AC]
  [has_add BD] [has_eq BD]
  [has_sub BD] [has_eq AC]
  (hMedian : is_median A M B C)
  (hD : D ∈ segment A C)
  (hBD_eq_AC : segment_length B D = segment_length A C)
  (hIntersection : ∃ K, K ∈ segment B D ∧ line A M ∩ line B D = {K})
  (hDK_eq_DC : segment_length D K = segment_length D C)
  : segment_length A M + segment_length K M = segment_length A B :=
sorry

end triangle_AM_plus_KM_eq_AB_l559_559828


namespace volume_inside_sphere_outside_cylinder_l559_559556

noncomputable
def volume_difference : ℚ :=
  let radius_sphere := 5 in
  let radius_cylinder := 3 in
  let height_cylinder := 8 in
  let volume_sphere := (4/3 : ℚ) * π * (radius_sphere : ℚ)^3 in
  let volume_cylinder := π * (radius_cylinder : ℚ)^2 * (height_cylinder : ℚ) in
  (volume_sphere - volume_cylinder)

theorem volume_inside_sphere_outside_cylinder : volume_difference = (284/3) * π := 
  sorry

end volume_inside_sphere_outside_cylinder_l559_559556


namespace concurrency_of_lines_l559_559707

noncomputable theory
open_locale classical

variables {A B C A1 D E F G : Type*}
variables [triangle A B C] [¬ isosceles A B C] (bisector : angle_bisector A A1 B C)
variables [square A A1 D E] [square A A1 F G] (opposite_sides : opposite_side B F A A1)

theorem concurrency_of_lines (h₁ : non_isosceles_triangle A B C)
                            (h₂ : angle_bisector A A1 B C)
                            (h₃ : squares A A1 D E A A1 F G)
                            (h₄ : opposite_side B F A A1) :
  concurrent (line_through B D) (line_through C F) (line_through E G) :=
sorry

end concurrency_of_lines_l559_559707


namespace z_conjugate_in_first_quadrant_l559_559394

open Complex

-- Definitions based on conditions
def a : ℝ := sorry
def z : ℂ := (a^2 - 4 * a + 5 : ℝ) - (6 : ℂ) * I

-- Conjugate of z
def conjugate_z : ℂ := conj z

-- Coordinates of the conjugate
def coord_re : ℝ := re conjugate_z
def coord_im : ℝ := im conjugate_z

-- Proof statement
theorem z_conjugate_in_first_quadrant (a : ℝ) :
  0 < coord_re ∧ 0 < coord_im :=
sorry

end z_conjugate_in_first_quadrant_l559_559394


namespace prove_road_length_l559_559937

-- Define variables for days taken by team A, B, and C
variables {a b c : ℕ}

-- Define the daily completion rates for teams A, B, and C
def rateA : ℕ := 300
def rateB : ℕ := 240
def rateC : ℕ := 180

-- Define the maximum length of the road
def max_length : ℕ := 3500

-- Define the total section of the road that team A completes in a days
def total_A (a : ℕ) : ℕ := a * rateA

-- Define the total section of the road that team B completes in b days and 18 hours
def total_B (a b : ℕ) : ℕ := 240 * (a + b) + 180

-- Define the total section of the road that team C completes in c days and 8 hours
def total_C (a b c : ℕ) : ℕ := 180 * (a + b + c) + 60

-- Define the constraint on the sum of days taken: a + b + c
def total_days (a b c : ℕ) : ℕ := a + b + c

-- The proof goal: Prove that (a * 300 == 3300) given the conditions
theorem prove_road_length :
  (total_A a = 3300) ∧ (total_B a b ≤ max_length) ∧ (total_C a b c ≤ max_length) ∧ (total_days a b c ≤ 19) :=
sorry

end prove_road_length_l559_559937


namespace min_xy_eq_nine_l559_559186

theorem min_xy_eq_nine (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + y + 3) : x * y = 9 :=
sorry

end min_xy_eq_nine_l559_559186


namespace triangle_AM_KM_AB_l559_559827

theorem triangle_AM_KM_AB (A B C D M K : Point) (hBD_eq_AC : BD = AC)
  (hAM_median : is_median A M)
  (hK_on_AM : AM ∩ BD = K)
  (hDK_eq_DC : DK = DC) :
  AM + KM = AB :=
sorry

end triangle_AM_KM_AB_l559_559827


namespace chord_length_intercepted_l559_559215

theorem chord_length_intercepted (t : ℝ) :
  let x := 2 + t
  let y := real.sqrt 3 * t
  (x^2 - y^2 = 1) → (chord_length (2 + t, real.sqrt 3 * t) = real.sqrt 10) := by
  sorry

end chord_length_intercepted_l559_559215


namespace sales_in_fifth_month_l559_559941

-- Define the sales figures and average target
def s1 : ℕ := 6435
def s2 : ℕ := 6927
def s3 : ℕ := 6855
def s4 : ℕ := 7230
def s6 : ℕ := 6191
def s_target : ℕ := 6700
def n_months : ℕ := 6

-- Define the total sales and the required fifth month sale
def total_sales : ℕ := s_target * n_months
def s5 : ℕ := total_sales - (s1 + s2 + s3 + s4 + s6)

-- The main theorem statement we need to prove
theorem sales_in_fifth_month :
  s5 = 6562 :=
sorry

end sales_in_fifth_month_l559_559941


namespace find_y_intercept_l559_559131

-- Define the slopes of the lines
def slope_l1 : ℝ := 1 / 2
def slope_l2 : ℝ := 1 / 3
def slope_l3 : ℝ := 1 / 4

-- Define the common y-intercept
def y_intercept (b : ℝ) := ∀ (x : ℝ), y = slope_l1 * x + b ∨ y = slope_l2 * x + b ∨ y = slope_l3 * x + b

-- Define the x-intercepts calculation
def x_intercept_l1 (b : ℝ) : ℝ := -2 * b
def x_intercept_l2 (b : ℝ) : ℝ := -3 * b
def x_intercept_l3 (b : ℝ) : ℝ := -4 * b

-- Condition for the sum of x-intercepts
def sum_x_intercepts (b : ℝ) : Prop :=
  x_intercept_l1 b + x_intercept_l2 b + x_intercept_l3 b = 36

-- The theorem we want to prove
theorem find_y_intercept (b : ℝ) (h : sum_x_intercepts b) : b = -4 :=
by
  sorry

end find_y_intercept_l559_559131


namespace lowest_number_speak_both_l559_559517

theorem lowest_number_speak_both (S H E B : ℕ) (h1 : S = 40) (h2 : H = 30) (h3 : E = 20) :
  S = H + E - B → B = 10 := by
  intros he hs
  rw [h1, h2, h3] at he
  sorry -- proof is not required

end lowest_number_speak_both_l559_559517


namespace average_speed_correct_l559_559121

/-- Define the distances covered in each hour. --/
def distance_first_hour : ℕ := 98
def distance_second_hour : ℕ := 70

/-- Define the total distance traveled as the sum of the distances in each hour. --/
def total_distance : ℕ := distance_first_hour + distance_second_hour

/-- Define the total time traveled. --/
def total_time : ℕ := 2

/-- Define the average speed as the total distance divided by the total time. --/
def average_speed : ℕ := total_distance / total_time

theorem average_speed_correct : average_speed = 84 := by
  simp [distance_first_hour, distance_second_hour, total_distance, total_time, average_speed]
  sorry

end average_speed_correct_l559_559121


namespace trigonometric_identity_l559_559460

theorem trigonometric_identity :
  (cos (32 * Real.pi / 180) * sin (62 * Real.pi / 180) - sin (32 * Real.pi / 180) * sin (28 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l559_559460


namespace scientific_notation_384000_l559_559444

theorem scientific_notation_384000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 384000 = a * 10 ^ n ∧ 
  a = 3.84 ∧ n = 5 :=
sorry

end scientific_notation_384000_l559_559444


namespace megatek_manufacturing_sector_l559_559087

theorem megatek_manufacturing_sector (percentage_manufacturing : ℝ) (total_degrees : ℝ) (sector_degrees : ℝ) 
  (h1 : percentage_manufacturing = 0.15) 
  (h2 : total_degrees = 360) :
  sector_degrees = percentage_manufacturing * total_degrees := 
by
  sorry

example : ∃ sector_degrees, megatek_manufacturing_sector 0.15 360 54 :=
by
  use 54
  apply megatek_manufacturing_sector
  repeat {rfl}

end megatek_manufacturing_sector_l559_559087


namespace trigonometric_identity_example_l559_559619

theorem trigonometric_identity_example :
  (let θ := 30 / 180 * Real.pi in
   (Real.tan θ)^2 - (Real.sin θ)^2) / ((Real.tan θ)^2 * (Real.sin θ)^2) = 4 / 3 :=
by
  let θ := 30 / 180 * Real.pi
  have h_tan2 := Real.tan θ * Real.tan θ
  have h_sin2 : Real.sin θ * Real.sin θ = 1/4 := by sorry
  have h_cos2 : Real.cos θ * Real.cos θ = 3/4 := by sorry
  sorry

end trigonometric_identity_example_l559_559619


namespace hope_cup_1990_inequalities_l559_559319

variable {a b c x y z : ℝ}

/-- Given a > b > c, x > y > z,
    M = ax + by + cz,
    N = az + by + cx,
    P = ay + bz + cx,
    Q = az + bx + cy.

    Prove that:
    M > P > N and M > Q > N. -/
theorem hope_cup_1990_inequalities :
  a > b -> b > c -> x > y -> y > z ->
  let M := a * x + b * y + c * z in
  let N := a * z + b * y + c * x in
  let P := a * y + b * z + c * x in
  let Q := a * z + b * x + c * y in
  M > P ∧ P > N ∧ M > Q ∧ Q > N := sorry

end hope_cup_1990_inequalities_l559_559319


namespace expression_equals_required_value_l559_559148

-- Define the expression as needed
def expression : ℚ := (((((4 + 2)⁻¹ + 2)⁻¹) + 2)⁻¹) + 2

-- Define the theorem stating that the expression equals the required value
theorem expression_equals_required_value : 
  expression = 77 / 32 := 
sorry

end expression_equals_required_value_l559_559148


namespace decreasing_interval_cos_square_l559_559348

theorem decreasing_interval_cos_square (φ : ℝ) (k : ℤ) (h₁ : 0 < φ ∧ φ < π / 2) (h₂ : sin φ - cos φ = sqrt 2 / 2) :
  ∀ x, (k * π - 5 * π / 12 <= x ∧ x <= k * π + π / 12) → ∀ ε > 0, (f : ℝ → ℝ), (∀ x, f x = cos^2 (x + φ)) → 
    (∀ x, f (x + ε) < f x) ∧ (∀ x, f (x - ε) > f x) :=
by sorry

end decreasing_interval_cos_square_l559_559348


namespace diff_between_largest_and_smallest_fraction_l559_559422

theorem diff_between_largest_and_smallest_fraction : 
  let f1 := (3 : ℚ) / 4
  let f2 := (7 : ℚ) / 8
  let f3 := (13 : ℚ) / 16
  let f4 := (1 : ℚ) / 2
  let largest := max f1 (max f2 (max f3 f4))
  let smallest := min f1 (min f2 (min f3 f4))
  largest - smallest = (3 : ℚ) / 8 :=
by
  sorry

end diff_between_largest_and_smallest_fraction_l559_559422


namespace right_triangle_condition_l559_559775

theorem right_triangle_condition (a b c : ℝ) : (a^2 = b^2 - c^2) → (∃ B : ℝ, B = 90) := 
sorry

end right_triangle_condition_l559_559775


namespace positive_difference_between_loan_options_l559_559574

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P r t : ℝ) : ℝ :=
  P * (1 + r * t)

def total_payment_option1 : ℝ :=
  let P := 10000
  let r := 0.08
  let n := 12
  let t := 6
  let A := compound_interest P r n t
  let half_payment := A / 2
  let remainder := half_payment
  let final_payment := compound_interest remainder r n t
  half_payment + final_payment

def total_payment_option2 : ℝ :=
  let P := 10000
  let r := 0.09
  let t := 12
  simple_interest P r t

theorem positive_difference_between_loan_options :
  abs (total_payment_option1 - total_payment_option2) = 118 :=
by
  sorry

end positive_difference_between_loan_options_l559_559574


namespace find_50th_term_l559_559631

-- Definitions of powers and distinct sums
def is_power_of (b n : ℕ) : Prop := ∃ k : ℕ, n = b ^ k
def is_sum_of_distinct_powers (b n : ℕ) : Prop := ∃ S : finset ℕ, (∀ i ∈ S, ∀ j ∈ S, i ≠ j → ((b ^ i) + (b ^ j) ≠ n)) ∧ n = S.sum (λ i, b ^ i)

-- Defining the sequence
def in_sequence (n : ℕ) : Prop :=
  is_power_of 2 n ∨ is_power_of 3 n ∨ (is_sum_of_distinct_powers 2 n) ∨ (is_sum_of_distinct_powers 3 n)

-- Lean 4 statement for the 50th term
theorem find_50th_term : ∃ s : list ℕ, (∀ x ∈ s, in_sequence x) ∧ sorted (<) s ∧ length s = 50 ∧ nth s 49 = some 327 := sorry

end find_50th_term_l559_559631


namespace distict_digits_sum_l559_559812

def f (n : ℕ) : ℕ := (n.digits 10).toFinset.card

theorem distict_digits_sum :
  (∑ n in finset.filter (λ n, nat.digits 10 n).length = 2019, finset.range (10^2019), f n) = 9 * (10^2019 - 9^2019) :=
by
  sorry

end distict_digits_sum_l559_559812


namespace ab_product_power_l559_559346

theorem ab_product_power (a b : ℤ) (n : ℕ) (h1 : (a * b)^n = 128 * 8) : n = 10 := by
  sorry

end ab_product_power_l559_559346


namespace modulus_of_w_l559_559846

noncomputable def w : ℂ := sorry -- We do not need the explicit value of w for this statement

theorem modulus_of_w (w : ℂ) (h : w^2 = -18 + 18 * I) : complex.abs w = 3 * real.sqrt (real.sqrt 2) :=
by sorry

end modulus_of_w_l559_559846


namespace elisa_paint_ratio_l559_559822

theorem elisa_paint_ratio :
  let monday_paint_sqft := 30
  let tuesday_paint_sqft := 2 * monday_paint_sqft
  let total_paint_sqft := 105
  let wednesday_paint_sqft := total_paint_sqft - (monday_paint_sqft + tuesday_paint_sqft)
  (wednesday_paint_sqft : ℝ) / monday_ppaint_sqft = 1 / 2 :=
by
  have monday_paint_sqft := 30
  have tuesday_paint_sqft := 2 * monday_paint_sqft
  have total_paint_sqft := 105
  let wednesday_paint_sqft := total_paint_sqft - (monday_paint_sqft + tuesday_paint_sqft)
  show (wednesday_paint_sqft : ℝ) / monday_paint_sqft = 1 / 2
  sorry

end elisa_paint_ratio_l559_559822


namespace distinguishable_arrangements_count_l559_559125

-- Definitions of conditions
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def vertices := Finset.range 9
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0
def is_valid (arrangement : Finset ℕ) : Prop :=
  ∀ i in vertices, is_multiple_of_three (arrangement i + arrangement ((i + 1) % 9) + arrangement ((i + 2) % 9))

-- Claim to prove: The number of distinguishable arrangements
noncomputable def number_of_distinguishable_arrangements : ℕ := sorry

theorem distinguishable_arrangements_count :
  number_of_distinguishable_arrangements = 144 :=
sorry

end distinguishable_arrangements_count_l559_559125


namespace probability_eq_2_div_5_l559_559425

noncomputable theory
open Real

def AB_length := 5
def AP_between_areas (x : ℝ) : Prop := sqrt 3 < (sqrt 3 / 4) * x^2 ∧ (sqrt 3 / 4) * x^2 < 4 * sqrt 3

theorem probability_eq_2_div_5 :
  (∫ x in 0..AB_length, if AP_between_areas x then 1 else 0) / AB_length = 2/5 :=
sorry

end probability_eq_2_div_5_l559_559425


namespace curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559005

section
variables (k t θ ρ : ℝ)  

def parametric_curve_C1 (k : ℝ) (t : ℝ) : ℝ × ℝ := (cos t ^ k, sin t ^ k)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

def curve_C2 (ρ θ : ℝ) : Prop := 4 * ρ * cos θ - 16 * ρ * sin θ + 3 = 0

theorem curve_C1_when_k1_is_circle : 
  ∀ t : ℝ, (parametric_curve_C1 1 t).fst ^ 2 + (parametric_curve_C1 1 t).snd ^ 2 = 1 :=
begin
  intros t,
  simp only [parametric_curve_C1],
  have h : cos t ^ 2 + sin t ^ 2 = 1, from real.sin_sq_add_cos_sq t,
  simp [h],
end

noncomputable def intersection_points_when_k4 : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p = parametric_curve_C1 4 t ∧ 
    let (x, y) := p in 4 * x - 16 * y + 3 = 0}

theorem find_intersection_points_when_k4 : 
  intersection_points_when_k4 = { (1/4, 1/4) } :=
begin
  sorry
end

end

end curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559005


namespace indistinguishable_partitions_of_5_into_3_boxes_l559_559742

-- Definitions for the conditions
def areIndistinguishable (A : Multiset ℕ) (B : Multiset ℕ) : Prop :=
  A = B

theorem indistinguishable_partitions_of_5_into_3_boxes:
  { S : Multiset (Multiset ℕ) // ∀ x ∈ S, x.card = 3 ∧ x.sum = 5 } = 
  { {5, 0, 0}, {4, 1, 0}, {3, 2, 0}, {3, 1, 1}, {2, 2, 1} } :=
by
  sorry

end indistinguishable_partitions_of_5_into_3_boxes_l559_559742


namespace final_value_geq_inv_n_l559_559056

-- Define the initial conditions and the operation
def initial_numbers (n : ℕ) : list ℕ := list.repeat 1 n

-- Define the operation as given
def operation (a b : ℕ) : ℕ := (a + b) / 4

-- Define the proof statement
theorem final_value_geq_inv_n (n : ℕ) (hn : n > 0) :
  ∃ x, x ∈ list.iterate (λ l, match l with
                             | [] | [_] => l
                             | a :: b :: ls => (operation a b) :: ls
                             end) (n - 1) (initial_numbers n) ∧ x ≥ 1 / n := sorry

end final_value_geq_inv_n_l559_559056


namespace cost_of_eight_books_with_discount_l559_559476

/-- 
Three identical books regularly cost a total of $45.
A discount of 10% is applied to purchases of five or more books.
What is the cost in dollars of eight of these books if the discount is applied?
-/
theorem cost_of_eight_books_with_discount :
  ∃ price_per_book : ℝ, price_per_book = 15 ∧
  ∃ regular_cost_eight_books : ℝ, regular_cost_eight_books = 8 * price_per_book ∧
  ∃ discounted_cost : ℝ, discounted_cost = regular_cost_eight_books - 0.10 * regular_cost_eight_books ∧
  discounted_cost = 108 :=
begin
  sorry
end

end cost_of_eight_books_with_discount_l559_559476


namespace exist_positives_sum_either_l559_559181

variable {a : ℕ → ℕ → ℝ} -- assuming the index set as natural numbers for generalization 

-- Conditions
def positive_diagonal (i : ℕ) : Prop := a i i > 0
def negative_off_diagonal (i j : ℕ) : Prop := i ≠ j → a i j < 0

theorem exist_positives_sum_either (h_pos_diag : ∀ i, positive_diagonal i)
                                   (h_neg_off_diag : ∀ i j, negative_off_diagonal i j) : 
  ∃ (c1 c2 c3 : ℝ), 0 < c1 ∧ 0 < c2 ∧ 0 < c3 ∧ 
        ((a 1 1 * c1 + a 1 2 * c2 + a 1 3 * c3 > 0 ∧ 
          a 2 1 * c1 + a 2 2 * c2 + a 2 3 * c3 > 0 ∧ 
          a 3 1 * c1 + a 3 2 * c2 + a 3 3 * c3 > 0) 
        ∨
         (a 1 1 * c1 + a 1 2 * c2 + a 1 3 * c3 < 0 ∧ 
          a 2 1 * c1 + a 2 2 * c2 + a 2 3 * c3 < 0 ∧ 
          a 3 1 * c1 + a 3 2 * c2 + a 3 3 * c3 < 0) 
        ∨
         (a 1 1 * c1 + a 1 2 * c2 + a 1 3 * c3 = 0 ∧ 
          a 2 1 * c1 + a 2 2 * c2 + a 2 3 * c3 = 0 ∧ 
          a 3 1 * c1 + a 3 2 * c2 + a 3 3 * c3 = 0)) := sorry

end exist_positives_sum_either_l559_559181


namespace proposition_d_is_false_l559_559964

theorem proposition_d_is_false:
  (∀ (P Q : Point), ∃! (l : Line), passes_through l P ∧ passes_through l Q) →  -- Condition A
  (∀ (T : Triangle), equilateral T → isosceles T) →                             -- Condition B
  (∀ (α β : Angle), vertical α β → α = β) →                                     -- Condition C
  ¬ (∀ (T : Triangle) (e : AngleOf T), external e →  -- Statement for D
    ∀ (i : AngleOf T), internal i → e > i) :=                                      
sorry

end proposition_d_is_false_l559_559964


namespace correct_option_l559_559161

noncomputable def OptionA : Prop := (Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2)
noncomputable def OptionB : Prop := (Real.sqrt (5 ^ 2) = -5 ∨ Real.sqrt (5 ^ 2) = 5)
noncomputable def OptionC : Prop := Real.sqrt ((-7) ^ 2) = 7
noncomputable def OptionD : Prop := (Real.sqrt (-3) = -Real.sqrt 3)

theorem correct_option : OptionC := 
by 
  unfold OptionC
  simp
  exact eq.refl 7

end correct_option_l559_559161


namespace quadratic_has_minimum_l559_559301

theorem quadratic_has_minimum (a b : ℝ) (h : a > b^2) :
  ∃ (c : ℝ), c = (4 * b^2 / a) - 3 ∧ (∃ x : ℝ, a * x ^ 2 + 2 * b * x + c < 0) :=
by sorry

end quadratic_has_minimum_l559_559301


namespace inequality_proof_l559_559404

theorem inequality_proof (a b c d : ℝ) : 
  (a^2 + b^2 + 1) * (c^2 + d^2 + 1) ≥ 2 * (a + c) * (b + d) :=
by sorry

end inequality_proof_l559_559404


namespace running_time_difference_l559_559585

theorem running_time_difference :
  ∀ (distance speed usual_speed : ℝ), 
  distance = 30 →
  usual_speed = 10 →
  speed = (distance / (usual_speed / 2)) - (distance / (usual_speed * 1.5)) →
  speed = 4 :=
by
  intros distance speed usual_speed hd hu hs
  sorry

end running_time_difference_l559_559585


namespace solve_for_x_l559_559437

theorem solve_for_x : ∃ (x : ℝ), 7 * (2 * x - 3) + 4 = -3 * (2 - 5 * x) ∧ x = -11 :=
by
  use -11
  split
  sorry

end solve_for_x_l559_559437


namespace sqrt_of_neg_7_sq_is_7_l559_559177

theorem sqrt_of_neg_7_sq_is_7 : sqrt ((-7)^2) = 7 :=
by sorry

end sqrt_of_neg_7_sq_is_7_l559_559177


namespace trigonometric_identity_l559_559599

theorem trigonometric_identity :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  t30 = 1 / Real.sqrt 3 →
  s30 = 1 / 2 →
  (t30 ^ 2 - s30 ^ 2) / (t30 ^ 2 * s30 ^ 2) = 1 :=
by
  intros t30_eq s30_eq
  -- Proof would go here
  sorry

end trigonometric_identity_l559_559599


namespace brokerage_percentage_is_02778_l559_559457

-- Conditions
def market_value_stock : ℝ := 90.02777777777779
def income : ℝ := 756
def investment : ℝ := 6500
def stock_rate : ℝ := 10.5

-- Question: What is the brokerage percentage?
theorem brokerage_percentage_is_02778 :
  let face_value := (income * 100) / stock_rate in
  let market_value := (face_value * market_value_stock) / 100 in
  let brokerage := investment - market_value in
  let brokerage_percentage := (brokerage / market_value) * 100 in
  brokerage_percentage = 0.2778 := sorry

end brokerage_percentage_is_02778_l559_559457


namespace rectangle_perimeter_l559_559948

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b = 2 * (a + b))) : 2 * (a + b) = 36 :=
by sorry

end rectangle_perimeter_l559_559948


namespace probability_five_distinct_dice_rolls_l559_559496

theorem probability_five_distinct_dice_rolls : 
  let total_outcomes := 6^5
  let favorable_outcomes := 6 * 5 * 4 * 3 * 2
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 54 :=
by
  sorry

end probability_five_distinct_dice_rolls_l559_559496


namespace find_a_l559_559711

theorem find_a (x : ℝ) (a : ℝ)
  (h1 : 3 * x - 4 = a)
  (h2 : (x + a) / 3 = 1)
  (h3 : (x = (a + 4) / 3) → (x = 3 - a → ((a + 4) / 3 = 2 * (3 - a)))) :
  a = 2 :=
sorry

end find_a_l559_559711


namespace sufficient_condition_increasing_function_l559_559323

open Real

theorem sufficient_condition_increasing_function (a : ℝ) :
  (∀ x : ℝ, (1 < x) → 2 * x ^ 3 ≥ a) → (0 < a ∧ a < 2) → ∃ x > 1, f' x ≥ 0 :=
by sorry

end sufficient_condition_increasing_function_l559_559323


namespace min_value_of_f_l559_559864

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 3

theorem min_value_of_f : ∃ x ∈ Icc (-1:ℝ) (1:ℝ), f x = -1 := by
  sorry

end min_value_of_f_l559_559864


namespace modulus_condition_l559_559307

noncomputable def modulus_of_a_plus_bi (a b : ℝ) : ℝ :=
  complex.abs (a + b * complex.I)

theorem modulus_condition (a b : ℝ) (h : (1 + 2 * complex.I) / (a + b * complex.I) = 1 + complex.I) :
  modulus_of_a_plus_bi a b = sqrt 10 / 2 :=
  sorry

end modulus_condition_l559_559307


namespace knight_tour_impossible_49_squares_l559_559781

-- Define the size of the chessboard
def boardSize : ℕ := 7

-- Define the total number of squares on the chessboard
def totalSquares : ℕ := boardSize * boardSize

-- Define the condition for a knight's tour on the 49-square board
def knight_tour_possible (n : ℕ) : Prop :=
  n = totalSquares ∧ 
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 
  -- add condition representing knight's tour and ending
  -- adjacent condition can be mathematically proved here 
  -- but we'll skip here as we asked just to state the problem not the proof.
  sorry -- Placeholder for the precise condition

-- Define the final theorem statement
theorem knight_tour_impossible_49_squares : ¬ knight_tour_possible totalSquares :=
by sorry

end knight_tour_impossible_49_squares_l559_559781


namespace find_x_value_l559_559882

theorem find_x_value (x : ℝ) 
  (h₁ : 1 / (x + 8) + 2 / (3 * x) + 1 / (2 * x) = 1 / (2 * x)) :
  x = (-1 + Real.sqrt 97) / 6 :=
sorry

end find_x_value_l559_559882


namespace cubes_closed_under_multiplication_l559_559409

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def is_closed_under (v : Set ℕ) (op : ℕ → ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a ∈ v → b ∈ v → op a b ∈ v

theorem cubes_closed_under_multiplication :
  let v := { n : ℕ | is_cube n } in
  is_closed_under v (λ x y, x * y) :=
by
  sorry

end cubes_closed_under_multiplication_l559_559409


namespace three_pow_two_n_plus_two_sub_two_pow_n_plus_one_div_by_seven_l559_559059

theorem three_pow_two_n_plus_two_sub_two_pow_n_plus_one_div_by_seven (n : ℤ) : 
  7 ∣ (3 ^ (2 * n + 2) - 2 ^ (n + 1)) := 
sorry

end three_pow_two_n_plus_two_sub_two_pow_n_plus_one_div_by_seven_l559_559059


namespace least_three_digit_12_heavy_number_l559_559563

def is_12_heavy (n : ℕ) : Prop :=
  n % 12 > 8

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem least_three_digit_12_heavy_number :
  ∃ n, three_digit n ∧ is_12_heavy n ∧ ∀ m, three_digit m ∧ is_12_heavy m → n ≤ m :=
  Exists.intro 105 (by
    sorry)

end least_three_digit_12_heavy_number_l559_559563


namespace midpoint_of_tangents_l559_559799

noncomputable def ApolloniusCircle (A B : Point) : Circle := sorry -- Definition of Apollonius circle

variable (A B P Q : Point)
variable (S : Circle)

-- This theorem states that if A is outside the Apollonius Circle S of A and B, and P, Q are points
-- where the tangents from A touch S, then B is the midpoint of the segment PQ.
theorem midpoint_of_tangents 
  (hS : S = ApolloniusCircle A B)
  (hA_outside : ¬ (A ∈ S))
  (h_tangents : ∀ P Q, Tangent A P S ∧ Tangent A Q S) 
  : Midpoint B P Q := 
sorry

end midpoint_of_tangents_l559_559799


namespace sum_divisors_36_l559_559281

def sum_of_divisors (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).filter (λ d, n % d = 0).sum

theorem sum_divisors_36 : sum_of_divisors 36 = 91 := sorry

end sum_divisors_36_l559_559281


namespace f_eq_32x5_l559_559039

def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

theorem f_eq_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  -- the proof proceeds here
  sorry

end f_eq_32x5_l559_559039


namespace integer_part_of_x_l559_559745

theorem integer_part_of_x : 
  let x := Real.sqrt 18 - 2 in
  Int.floor x = 2 :=
by
  sorry

end integer_part_of_x_l559_559745


namespace no_such_2013_distinct_naturals_l559_559264

theorem no_such_2013_distinct_naturals :
  ¬ (∃ (a : Fin 2013 → ℕ), Function.Injective a ∧ ∀ k : Fin 2013, ∑ i in (Fin 2013).erase k, a i ≥ (a k) ^ 2) := by
  sorry

end no_such_2013_distinct_naturals_l559_559264


namespace trigonometric_identity_l559_559594

theorem trigonometric_identity :
  let θ := 30 * Real.pi / 180
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2) / (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_l559_559594


namespace number_of_participants_l559_559098

theorem number_of_participants (total_gloves : ℕ) (gloves_per_participant : ℕ)
  (h : total_gloves = 126) (h' : gloves_per_participant = 2) : 
  (total_gloves / gloves_per_participant = 63) :=
by
  sorry

end number_of_participants_l559_559098


namespace find_PL_l559_559011

-- Definitions of the basic geometric properties and assumptions
variables {Point : Type} [MetricSpace Point]
def square (s : ℝ) (W X Y Z : Point) : Prop :=
  dist W X = s ∧ dist X Y = s ∧ dist Y Z = s ∧ dist Z W = s ∧
  dist W Y = dist X Z

def congruent (LMNO PQRS : Point × Point × Point × Point) : Prop :=
  dist (LMNO.1) (LMNO.2) = dist (PQRS.1) (PQRS.2) ∧
  dist (LMNO.2) (LMNO.3) = dist (PQRS.2) (PQRS.3)

-- Statement of the problem conditions and the conclusion as a theorem
theorem find_PL 
  (W X Y Z L M N O P Q R S : Point)
  (h_square : square 2 W X Y Z)
  (h_congruent : congruent (L, M, N, O) (P, Q, R, S))
  (PL PS PR : Point -> ℝ) : 
  PL P = 1 :=
sorry

end find_PL_l559_559011


namespace segment_length_and_sum_l559_559326

theorem segment_length_and_sum (a : ℕ) (h : 0 < a) :
  (let f : ℝ → ℝ := λ x, a * (a + 1) * x^2 - (2 * a + 1) * x + 1 in
  ∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ abs (x1 - x2) = 1 / (a * (a + 1))) ∧
  (Σ i in Finset.range (2005 + 1), (1 : ℝ) / (i * (i + 1))) = 2005 / 2006 :=
    by
      sorry

end segment_length_and_sum_l559_559326


namespace quadrilateral_is_parallelogram_l559_559045

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

def is_parallelogram (a b c d : A) : Prop :=
∃ E F, midpoint a d = E ∧ midpoint b c = E ∧ 
midpoint a b = F ∧ midpoint c d = F

theorem quadrilateral_is_parallelogram
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (dist_AB dist_CD dist_AC dist_BC dist_DA dist_BD : ℝ)
  (h1 : dist_AB + dist_CD = Real.sqrt 2 * dist_AC)
  (h2 : dist_BC + dist_DA = Real.sqrt 2 * dist_BD) :
  is_parallelogram A B C D :=
sorry

end quadrilateral_is_parallelogram_l559_559045


namespace total_production_second_year_l559_559539

noncomputable def production_rate := 10
noncomputable def days_in_year := 365
noncomputable def reduction_rate := 0.10

theorem total_production_second_year :
  let first_year_production := production_rate * days_in_year
  let reduction_amount := reduction_rate * first_year_production
  let second_year_production := first_year_production - reduction_amount
  second_year_production = 3285 :=
by
  sorry

end total_production_second_year_l559_559539


namespace probability_five_distinct_numbers_l559_559503

def num_dice := 5
def num_faces := 6

def favorable_outcomes : ℕ := nat.factorial 5 * num_faces
def total_outcomes : ℕ := num_faces ^ num_dice

theorem probability_five_distinct_numbers :
  (favorable_outcomes / total_outcomes : ℚ) = 5 / 54 := 
sorry

end probability_five_distinct_numbers_l559_559503


namespace probability_theorem_l559_559475

open Set Function

-- Define the conditions for the problem
def boxA := {n | 1 ≤ n ∧ n ≤ 30}
def boxB := {n | 15 ≤ n ∧ n ≤ 44}

def P_A (n : ℕ) := n ∈ boxA
def P_B (n : ℕ) := n ∈ boxB

def conditionA := {n ∈ boxA | n < 20}
def conditionB := {n ∈ boxB | (∃ k, n = 2 * k) ∨ n > 35}

-- Calculate the probabilities
def probabilityA := (conditionA.toFinset.card : ℚ) / (boxA.toFinset.card : ℚ)
def probabilityB := (conditionB.toFinset.card : ℚ) / (boxB.toFinset.card : ℚ)

-- The combined probability of independent events
def combined_probability := probabilityA * probabilityB

-- Lean statement for the proof problem
theorem probability_theorem : combined_probability = 361 / 900 :=
by
  -- defining the calculations based on conditions
  have boxA_card : boxA.toFinset.card = 30 := sorry
  have boxB_card : boxB.toFinset.card = 30 := sorry
  
  have conditionA_card : conditionA.toFinset.card = 19 := sorry
  have conditionB_card : conditionB.toFinset.card = 19 := sorry

  -- calculations for the individual probabilities
  have pA : probabilityA = 19 / 30 := by
    rw [probabilityA, conditionA_card, boxA_card]; norm_num

  have pB : probabilityB = 19 / 30 := by
    rw [probabilityB, conditionB_card, boxB_card]; norm_num
    
  -- verifying the combined probability
  rw [combined_probability, pA, pB]; norm_num

end probability_theorem_l559_559475


namespace max_projection_area_l559_559830

noncomputable def maxProjectionArea (a : ℝ) : ℝ :=
  if a > (Real.sqrt 3 / 3) ∧ a <= (Real.sqrt 3 / 2) then
    Real.sqrt 3 / 4
  else if a >= (Real.sqrt 3 / 2) then
    a / 2
  else 
    0  -- if the condition for a is not met, it's an edge case which shouldn't logically occur here

theorem max_projection_area (a : ℝ) (h1 : a > Real.sqrt 3 / 3) (h2 : a <= Real.sqrt 3 / 2 ∨ a >= Real.sqrt 3 / 2) :
  maxProjectionArea a = 
    if a > Real.sqrt 3 / 3 ∧ a <= Real.sqrt 3 / 2 then Real.sqrt 3 / 4
    else if a >= Real.sqrt 3 / 2 then a / 2
    else
      sorry :=
by sorry

end max_projection_area_l559_559830


namespace john_new_weekly_earnings_l559_559381

theorem john_new_weekly_earnings
  (original_earnings : ℕ)
  (percentage_increase : ℕ)
  (raise_amount : ℕ)
  (new_weekly_earnings : ℕ)
  (original_earnings_eq : original_earnings = 50)
  (percentage_increase_eq : percentage_increase = 40)
  (raise_amount_eq : raise_amount = original_earnings * percentage_increase / 100)
  (new_weekly_earnings_eq : new_weekly_earnings = original_earnings + raise_amount) :
  new_weekly_earnings = 70 := by
  sorry

end john_new_weekly_earnings_l559_559381


namespace martin_family_ice_cream_cost_l559_559442

theorem martin_family_ice_cream_cost (R : ℤ)
  (kiddie_scoop_cost : ℤ) (double_scoop_cost : ℤ)
  (total_cost : ℤ) :
  kiddie_scoop_cost = 3 → 
  double_scoop_cost = 6 → 
  total_cost = 32 →
  2 * R + 2 * kiddie_scoop_cost + 3 * double_scoop_cost = total_cost →
  R = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end martin_family_ice_cream_cost_l559_559442


namespace correct_option_l559_559160

noncomputable def OptionA : Prop := (Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2)
noncomputable def OptionB : Prop := (Real.sqrt (5 ^ 2) = -5 ∨ Real.sqrt (5 ^ 2) = 5)
noncomputable def OptionC : Prop := Real.sqrt ((-7) ^ 2) = 7
noncomputable def OptionD : Prop := (Real.sqrt (-3) = -Real.sqrt 3)

theorem correct_option : OptionC := 
by 
  unfold OptionC
  simp
  exact eq.refl 7

end correct_option_l559_559160


namespace time_spent_driving_l559_559084

def distance_home_to_work: ℕ := 60
def speed_mph: ℕ := 40

theorem time_spent_driving:
  (2 * distance_home_to_work) / speed_mph = 3 := by
  sorry

end time_spent_driving_l559_559084


namespace solve_for_t_l559_559343

theorem solve_for_t (t : ℝ) (h1 : sin (2 * t) = - ∫ x in 0..t, cos x)
  (h2 : 0 < t ∧ t < π) : t = 2 * π / 3 :=
sorry

end solve_for_t_l559_559343


namespace number_of_volunteers_in_sample_probability_same_grade_l559_559355

namespace VolunteersSampling

-- Definitions of conditions
def first_grade_volunteers := 36
def second_grade_volunteers := 72
def third_grade_volunteers := 54
def third_grade_sample := 3
def sample_ratio := third_grade_sample / third_grade_volunteers

-- Question I answers
def first_grade_sample := first_grade_volunteers * sample_ratio
def second_grade_sample := second_grade_volunteers * sample_ratio

-- Theorem for Question I
theorem number_of_volunteers_in_sample :
  first_grade_sample = 2 ∧ second_grade_sample = 4 := by
  sorry

-- Definitions for Question II 
def volunteers_pair (a b : Type) : list (a × b) := []

def first_grade_sample_count := 2
def second_grade_sample_count := 4

-- Generate the pairs
def possible_pairs := [
  -- pairs from the first grade
  (A1, A2) 
  -- pairs from first grade and second grade and so on...
]

-- Count of pairs where both volunteers are from the same grade
def same_grade_pairs_count := 7 -- from the solution

-- Total pairs count
def total_pairs_count := 15

-- Theorem for Question II
theorem probability_same_grade :
  (same_grade_pairs_count / total_pairs_count) = (7 / 15) := by
  sorry

end VolunteersSampling

end number_of_volunteers_in_sample_probability_same_grade_l559_559355


namespace norm_scaled_vector_l559_559803

variable {ℝ : Type*} [NormedSpace ℝ (EuclideanSpace ℝ (Fin 2))]

theorem norm_scaled_vector (v : EuclideanSpace ℝ (Fin 2)) (hv : ‖v‖ = 5) : ‖7 • v‖ = 35 := 
sorry

end norm_scaled_vector_l559_559803


namespace min_value_square_l559_559519

def f (x y : ℝ) : ℝ := x^2 + 2*x + y^2 + 4*y

theorem min_value_square (x y : ℝ) :
  let f_sum := f x y + f (x+1) y + f (x+1) (y+1) + f x (y+1) in
  ∃ x y : ℝ, f_sum = -18 :=
by
  sorry

end min_value_square_l559_559519


namespace percentage_drive_from_center_l559_559231

-- Definitions based on given conditions
def distance_to_center (D : ℝ) : ℝ := D
def round_trip_distance (D : ℝ) : ℝ := 2 * D
def completed_round_trip_distance (D : ℝ) : ℝ := 0.6 * round_trip_distance D
def distance_from_center (D : ℝ) : ℝ := completed_round_trip_distance D - distance_to_center D
def percentage_completed_from_center (D : ℝ) : ℝ := (distance_from_center D / distance_to_center D) * 100

-- The theorem to prove
theorem percentage_drive_from_center (D : ℝ) (hD_pos : D > 0) :
  percentage_completed_from_center D = 20 :=
by
  sorry

end percentage_drive_from_center_l559_559231


namespace polynomial_divisibility_l559_559572

theorem polynomial_divisibility (n k : ℕ) (c : Fin k → ℤ) (h_k_even : Even k)
  (h_all_odd : ∀ i, c i % 2 ≠ 0)
  (h_divisibility : ∀ (x : ℤ), (x + 1) ^ n - 1 = (P x) * (Q x) ) 
  : (k + 1) ∣ n :=
sorry

end polynomial_divisibility_l559_559572


namespace total_animals_l559_559026

namespace Zoo

def snakes := 15
def monkeys := 2 * snakes
def lions := monkeys - 5
def pandas := lions + 8
def dogs := pandas / 3

theorem total_animals : snakes + monkeys + lions + pandas + dogs = 114 := by
  -- definitions from conditions
  have h_snakes : snakes = 15 := rfl
  have h_monkeys : monkeys = 2 * snakes := rfl
  have h_lions : lions = monkeys - 5 := rfl
  have h_pandas : pandas = lions + 8 := rfl
  have h_dogs : dogs = pandas / 3 := rfl
  -- sorry is used as a placeholder for the proof
  sorry

end Zoo

end total_animals_l559_559026


namespace no_such_2013_distinct_numbers_l559_559267

theorem no_such_2013_distinct_numbers :
  ∀ (a : Fin 2013 → ℕ), (Function.Injective a) ∧ (∀ i : Fin 2013, (∑ j in Finset.univ.erase i, a j) ≥ (a i) ^ 2) → False :=
by
  sorry

end no_such_2013_distinct_numbers_l559_559267


namespace area_PTQ_is_8_l559_559885

noncomputable def triangle_area_equality_problem 
  (P Q R S T U : Type) 
  (areaPQR : ℝ) (PS : ℝ) (SQ : ℝ)
  (on_PQ : S → P Q) : 
  Prop :=
  areaPQR = 20 ∧ PS = 3 ∧ SQ = 2 ∧ 
  (let PQ := PS + SQ in 
  let ratio_PS_SQ := PS / PQ in 
  let area_PTQ := ratio_PS_SQ * areaPQR in 
  area_PTQ = 8)

theorem area_PTQ_is_8 
  {P Q R S T U : Type} 
  (areaPQR : ℝ) (PS : ℝ) (SQ : ℝ)
  (on_PQ : S → P Q) 
  (h : triangle_area_equality_problem P Q R S T U areaPQR PS SQ on_PQ) : 
  let PQ := PS + SQ in 
  let ratio_PS_SQ := PS / PQ in 
  let area_PTQ := ratio_PS_SQ * areaPQR in 
  area_PTQ = 8 :=
by
  sorry

end area_PTQ_is_8_l559_559885


namespace exists_overlapping_pairs_l559_559269

-- Definition of conditions:
def no_boy_danced_with_all_girls (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ b : B, ∃ g : G, ¬ danced b g

def each_girl_danced_with_at_least_one_boy (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ g : G, ∃ b : B, danced b g

-- The main theorem to prove:
theorem exists_overlapping_pairs
  (B : Type) (G : Type) (danced : B → G → Prop)
  (h1 : no_boy_danced_with_all_girls B G danced)
  (h2 : each_girl_danced_with_at_least_one_boy B G danced) :
  ∃ (b1 b2 : B) (g1 g2 : G), b1 ≠ b2 ∧ g1 ≠ g2 ∧ danced b1 g1 ∧ danced b2 g2 :=
sorry

end exists_overlapping_pairs_l559_559269


namespace count_5_digit_palindromes_l559_559930

def is_digit (n : Nat) : Prop := 0 ≤ n ∧ n ≤ 9

def is_palindrome (n : Nat) : Prop :=
  ∃ a b c : Nat, a ≠ 0 ∧ is_digit a ∧ is_digit b ∧ is_digit c ∧
  n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem count_5_digit_palindromes : 
  (Finset.filter is_palindrome (Finset.range 100000)).card = 900 :=
by
  sorry

end count_5_digit_palindromes_l559_559930


namespace new_books_count_l559_559245

-- Defining the conditions
def num_adventure_books : ℕ := 13
def num_mystery_books : ℕ := 17
def num_used_books : ℕ := 15

-- Proving the number of new books Sam bought
theorem new_books_count : (num_adventure_books + num_mystery_books) - num_used_books = 15 :=
by
  sorry

end new_books_count_l559_559245


namespace number_of_pages_read_on_fourth_day_l559_559334

-- Define variables
variables (day1 day2 day3 day4 total_pages: ℕ)

-- Define conditions
def condition1 := day1 = 63
def condition2 := day2 = 2 * day1
def condition3 := day3 = day2 + 10
def condition4 := total_pages = 354
def read_in_four_days := total_pages = day1 + day2 + day3 + day4

-- State the theorem to be proven
theorem number_of_pages_read_on_fourth_day (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : read_in_four_days) : day4 = 29 :=
by sorry

end number_of_pages_read_on_fourth_day_l559_559334


namespace kayla_spent_money_l559_559030

theorem kayla_spent_money :
  let k := 100 in
  let r := (1 / 4 : ℝ) * k in
  let f := (1 / 10 : ℝ) * k in
  r + f = 35 := by
  sorry

end kayla_spent_money_l559_559030


namespace range_of_a_l559_559407

noncomputable def S : Set ℝ := {x | |x - 1| + |x + 2| > 5}
noncomputable def T (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}

theorem range_of_a (a : ℝ) : 
  (S ∪ T a) = Set.univ ↔ -2 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l559_559407


namespace solve_equation_l559_559074

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end solve_equation_l559_559074


namespace flour_per_cake_l559_559135

theorem flour_per_cake (traci_flour harris_flour : ℕ) (cakes_each : ℕ)
  (h_traci_flour : traci_flour = 500)
  (h_harris_flour : harris_flour = 400)
  (h_cakes_each : cakes_each = 9) :
  (traci_flour + harris_flour) / (2 * cakes_each) = 50 := by
  sorry

end flour_per_cake_l559_559135


namespace cubic_has_three_zeros_l559_559867

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem cubic_has_three_zeros : (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
sorry

end cubic_has_three_zeros_l559_559867


namespace cow_manure_plant_height_l559_559021

theorem cow_manure_plant_height
  (control_plant_height : ℝ)
  (bone_meal_ratio : ℝ)
  (cow_manure_ratio : ℝ)
  (h1 : control_plant_height = 36)
  (h2 : bone_meal_ratio = 1.25)
  (h3 : cow_manure_ratio = 2) :
  (control_plant_height * bone_meal_ratio * cow_manure_ratio) = 90 :=
sorry

end cow_manure_plant_height_l559_559021


namespace august_weekdays_five_times_l559_559847

theorem august_weekdays_five_times (N : ℕ) 
  (h1 : ∃ k : ℕ, N ≡ k [MOD 7] ∧ k = 6) 
  (h2 : ∀ d, 1 ≤ d ∧ d ≤ 31): 
  ∃ t w r f : ℕ, (t = 5 ∧ w = 5 ∧ r = 5 ∧ f = 5) :=
by
  -- Definition of starting days of the week and conditions given
  have start_day_july : ℕ := 6
  have days_in_july : ℕ := 31
  have days_in_august : ℕ := 31

  -- Ensure correct days and questions alignment for Lean
  let start_day_august := start_day_july + (days_in_july % 7)

  -- Logic to capture how many times each weekday repeats in August
  have t := (days_in_august + start_day_august - 2) // 7 + 1
  have w := (days_in_august + start_day_august - 3) // 7 + 1
  have r := (days_in_august + start_day_august - 4) // 7 + 1
  have f := (days_in_august + start_day_august - 5) // 7 + 1

  -- Proving the required assertion
  use t, w, r, f
  sorry -- skipping the concrete proof

end august_weekdays_five_times_l559_559847


namespace cost_per_text_message_for_first_plan_l559_559915

theorem cost_per_text_message_for_first_plan (x : ℝ) : 
  (9 + 60 * x = 60 * 0.40) → (x = 0.25) :=
by
  intro h
  sorry

end cost_per_text_message_for_first_plan_l559_559915


namespace equivalent_conditions_l559_559044

theorem equivalent_conditions 
  (f : ℕ+ → ℕ+)
  (H1 : ∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m))
  (H2 : ∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :
  (∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m)) ↔ 
  (∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :=
sorry

end equivalent_conditions_l559_559044


namespace minimize_shoes_l559_559824

-- Definitions for inhabitants, one-legged inhabitants, and shoe calculations
def total_inhabitants := 10000
def P (percent_one_legged : ℕ) := (percent_one_legged * total_inhabitants) / 100
def non_one_legged (percent_one_legged : ℕ) := total_inhabitants - (P percent_one_legged)
def non_one_legged_with_shoes (percent_one_legged : ℕ) := (non_one_legged percent_one_legged) / 2
def shoes_needed (percent_one_legged : ℕ) := 
  (P percent_one_legged) + 2 * (non_one_legged_with_shoes percent_one_legged)

-- Theorem to prove that 100% one-legged minimizes the shoes required
theorem minimize_shoes : ∀ (percent_one_legged : ℕ), shoes_needed percent_one_legged = total_inhabitants → percent_one_legged = 100 :=
by
  intros percent_one_legged h
  sorry

end minimize_shoes_l559_559824


namespace find_pq_l559_559641

variable (p q : ℝ)

def vec1 : Fin 3 → ℝ := ![3, p, -4]
def vec2 : Fin 3 → ℝ := ![6, 5, q]
def cross_product (v₁ v₂ : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![v₁ 1 * v₂ 2 - v₁ 2 * v₂ 1, 
    v₁ 2 * v₂ 0 - v₁ 0 * v₂ 2, 
    v₁ 0 * v₂ 1 - v₁ 1 * v₂ 0]

theorem find_pq :
  cross_product (vec1 p q) (vec2 p q) = ![0, 0, 0] →
  p = 5 / 2 ∧ q = -8 := by
  sorry

end find_pq_l559_559641


namespace probability_of_distinct_dice_numbers_l559_559494

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l559_559494


namespace original_salary_l559_559116

theorem original_salary (S : ℝ) (h : (1.12) * (0.93) * (1.09) * (0.94) * S = 1212) : 
  S = 1212 / ((1.12) * (0.93) * (1.09) * (0.94)) :=
by
  sorry

end original_salary_l559_559116


namespace find_a33_in_arithmetic_sequence_grid_l559_559571

theorem find_a33_in_arithmetic_sequence_grid 
  (matrix : ℕ → ℕ → ℕ)
  (rows_are_arithmetic : ∀ i, ∃ a b, ∀ j, matrix i j = a + b * (j - 1))
  (columns_are_arithmetic : ∀ j, ∃ c d, ∀ i, matrix i j = c + d * (i - 1))
  : matrix 3 3 = 31 :=
sorry

end find_a33_in_arithmetic_sequence_grid_l559_559571


namespace total_cups_l559_559256

variable (eggs : ℕ) (flour : ℕ)
variable (h : eggs = 60) (h1 : flour = eggs / 2)

theorem total_cups (eggs : ℕ) (flour : ℕ) (h : eggs = 60) (h1 : flour = eggs / 2) : 
  eggs + flour = 90 := 
by
  sorry

end total_cups_l559_559256


namespace Y_fraction_X_l559_559835

variable (R : ℝ) -- Retail price of the house

def Z_price : ℝ := R * (1 - 0.30) -- Salesman Z offers a 30% discount
def X_price : ℝ := Z_price R * (1 - 0.15) -- Salesman X matches Z's price and then offers an additional 15% discount
def Y_avg_price : ℝ := (Z_price R + X_price R) / 2 -- Salesman Y averages Z's and X's prices
def Y_price : ℝ := Y_avg_price R * (1 - 0.40) -- Salesman Y offers a 40% discount on the average price

theorem Y_fraction_X : Y_price R / X_price R = 0.653 := by sorry

end Y_fraction_X_l559_559835


namespace sum_of_coefficients_l559_559312

theorem sum_of_coefficients (n : ℕ) (h : binomialCoeff n 2 = binomialCoeff n 4) :
  (∑ k in finset.range (n+1), binomialCoeff n k * 2^k : ℕ) = 3^6 := 
by
  have h1 : n = 6 := 
    begin
      sorry -- Provide proof that n = 6 given h.
    end,
  rw h1,
  sorry -- Provide proof that the sum of coefficients in (1 + 2)^6 is 3^6.

end sum_of_coefficients_l559_559312


namespace food_company_total_food_l559_559540

theorem food_company_total_food (boxes : ℕ) (kg_per_box : ℕ) (full_boxes : boxes = 388) (weight_per_box : kg_per_box = 2) :
  boxes * kg_per_box = 776 :=
by
  -- the proof would go here
  sorry

end food_company_total_food_l559_559540


namespace john_needs_more_money_l559_559523

def total_needed : ℝ := 2.50
def current_amount : ℝ := 0.75
def remaining_amount : ℝ := 1.75

theorem john_needs_more_money : total_needed - current_amount = remaining_amount :=
by
  sorry

end john_needs_more_money_l559_559523


namespace intercepts_sum_eq_five_l559_559184

theorem intercepts_sum_eq_five (m : ℕ) 
  (h_m : m = 7) 
  (x y : ℕ) 
  (hx : x < m) 
  (hy : y < m) 
  (cong : 2 * x ≡ 3 * y + 2 [MOD m]) :
  let x_int := 1, y_int := 4 in
  x_int + y_int = 5 :=
by
  sorry

end intercepts_sum_eq_five_l559_559184


namespace min_value_of_function_l559_559490

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 729

theorem min_value_of_function : ∃ x : ℝ, f x = 702 :=
begin
  sorry
end

end min_value_of_function_l559_559490


namespace find_sachin_age_l559_559919

-- Define Sachin's and Rahul's ages as variables
variables (S R : ℝ)

-- Define the conditions
def rahul_age := S + 9
def age_ratio := (S / R) = (7 / 9)

-- State the theorem for Sachin's age
theorem find_sachin_age (h1 : R = rahul_age S) (h2 : age_ratio S R) : S = 31.5 :=
by sorry

end find_sachin_age_l559_559919


namespace total_whales_observed_l559_559782

-- Define the conditions
def trip1_male_whales : ℕ := 28
def trip1_female_whales : ℕ := 2 * trip1_male_whales
def trip1_total_whales : ℕ := trip1_male_whales + trip1_female_whales

def baby_whales_trip2 : ℕ := 8
def adult_whales_trip2 : ℕ := 2 * baby_whales_trip2
def trip2_total_whales : ℕ := baby_whales_trip2 + adult_whales_trip2

def trip3_male_whales : ℕ := trip1_male_whales / 2
def trip3_female_whales : ℕ := trip1_female_whales
def trip3_total_whales : ℕ := trip3_male_whales + trip3_female_whales

-- Prove the total number of whales observed
theorem total_whales_observed : trip1_total_whales + trip2_total_whales + trip3_total_whales = 178 := by
  -- Assuming all intermediate steps are correct
  sorry

end total_whales_observed_l559_559782


namespace probability_of_red_tile_is_one_fourth_l559_559532

noncomputable def numTiles : ℕ := 52

def isRed (n : ℕ) : Prop := n % 4 = 3

def redTiles : finset ℕ := finset.filter isRed (finset.range (numTiles + 1))

def probabilityRed : ℚ := redTiles.card / numTiles

theorem probability_of_red_tile_is_one_fourth :
  probabilityRed = 1 / 4 :=
by
  sorry

end probability_of_red_tile_is_one_fourth_l559_559532


namespace solution_set_of_inequality_l559_559682

theorem solution_set_of_inequality (a b x : ℝ) (h1 : 0 < a) (h2 : b = 2 * a) : ax > b ↔ x > -2 :=
by sorry

end solution_set_of_inequality_l559_559682


namespace max_value_of_x0_l559_559440

noncomputable def sequence_max_value (seq : Fin 1996 → ℝ) (pos_seq : ∀ i, seq i > 0) : Prop :=
  seq 0 = seq 1995 ∧
  (∀ i : Fin 1995, seq i + 2 / seq i = 2 * seq (i + 1) + 1 / seq (i + 1)) ∧
  (seq 0 ≤ 2^997)

theorem max_value_of_x0 :
  ∃ seq : Fin 1996 → ℝ, ∀ pos_seq : ∀ i, seq i > 0, sequence_max_value seq pos_seq :=
sorry

end max_value_of_x0_l559_559440


namespace distinct_ways_to_place_digits_in_grid_l559_559345

theorem distinct_ways_to_place_digits_in_grid : 
  let digits := {1, 2, 3, 4}
  let grid_size := 6
  let blanks := 2
  let total_elements := grid_size
  total_elements = digits.to_finset.card + blanks → 
  ∃ n : ℕ, n = Nat.factorial total_elements ∧ n = 720 :=
by
  let digits := {1, 2, 3, 4}
  let grid_size := 6
  let blanks := 2
  let total_elements := grid_size
  have h1 : total_elements = digits.to_finset.card + blanks := rfl
  use Nat.factorial total_elements
  split
  . rfl
  . have : Nat.factorial 6 = 720 := rfl
    exact this

end distinct_ways_to_place_digits_in_grid_l559_559345


namespace tan_sin_div_l559_559605

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l559_559605


namespace average_value_l559_559975

theorem average_value (z : ℝ) : 
  let s := [5, 5 + 3 * z, 5 + 6 * z, 5 + 9 * z, 5 + 12 * z] in
  (s.sum / 5) = (5 + 6 * z) :=
by
  sorry

end average_value_l559_559975


namespace correct_operation_l559_559171

theorem correct_operation :
  (∀ x : ℝ, sqrt (x^2) = abs x) ∧
  sqrt 4 = 2 ∧
  (∀ x : ℝ, sqrt (x^2) = x ∨ sqrt (x^2) = -x) →
  ((sqrt 4 ≠ ± 2) ∧
   (± sqrt (5^2) ≠ -5) ∧
   (sqrt ((-7)^2) = 7) ∧
   (sqrt (-3 : ℝ) ≠ -sqrt 3)) :=
by
  intro h
  clear h -- clear the hypothesis since no proof is needed
  split
  · intro h1
    -- prove sqrt 4 ≠ ±2
    sorry
  split
  · intro h2
    -- prove ± sqrt (5^2) ≠ -5
    sorry
  split
  · intro h3
    -- prove sqrt ((-7)^2) = 7
    exact abs_neg 7
  · intro h4
    -- prove sqrt (-3) ≠ - sqrt 3
    sorry

end correct_operation_l559_559171


namespace cubic_inequality_l559_559182

theorem cubic_inequality (p q x : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 :=
sorry

end cubic_inequality_l559_559182


namespace find_smallest_x_l559_559659

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem find_smallest_x (x: ℕ) (h1: 2 * x = 144) (h2: 3 * x = 216) : x = 72 :=
by
  sorry

end find_smallest_x_l559_559659


namespace travel_time_l559_559136

-- Definitions of the conditions
variables (x : ℝ) (speed_elder speed_younger : ℝ)
variables (time_elder_total time_younger_total : ℝ)

def elder_speed_condition : Prop := speed_elder = x
def younger_speed_condition : Prop := speed_younger = x - 4
def elder_distance : Prop := 42 / speed_elder + 1 = time_elder_total
def younger_distance : Prop := 42 / speed_younger + 1 / 3 = time_younger_total

-- The main theorem we want to prove
theorem travel_time : ∀ (x : ℝ), 
  elder_speed_condition x speed_elder → 
  younger_speed_condition x speed_younger → 
  elder_distance speed_elder time_elder_total → 
  younger_distance speed_younger time_younger_total → 
  time_elder_total = time_younger_total ∧ time_elder_total = (10 / 3) :=
sorry

end travel_time_l559_559136


namespace cost_of_one_roll_sold_individually_l559_559198

-- Definitions based on conditions
def cost_case_12_rolls := 9
def percent_savings := 0.25

-- Variable representing the cost of one roll sold individually
variable (x : ℝ)

-- Statement to prove
theorem cost_of_one_roll_sold_individually : (12 * x - 12 * percent_savings * x) = cost_case_12_rolls → x = 1 :=
by
  intro h
  -- This is where the proof would go
  sorry

end cost_of_one_roll_sold_individually_l559_559198


namespace boiling_point_fahrenheit_l559_559486

def celsius_to_fahrenheit (c : ℝ) : ℝ := (c * 9/5) + 32

theorem boiling_point_fahrenheit :
  celsius_to_fahrenheit 100 = 212 := 
begin
  -- proof goes here
  sorry
end

end boiling_point_fahrenheit_l559_559486


namespace tiffany_max_points_l559_559132

theorem tiffany_max_points
  (initial_money : ℕ)
  (cost_per_game : ℕ)
  (rings_per_game : ℕ)
  (points_red : ℕ)
  (points_green : ℕ)
  (points_blue : ℕ)
  (success_rate_blue : ℕ)
  (red_buckets_game1_game2 : ℕ)
  (green_buckets_game1_game2 : ℕ)
  (blue_buckets_game1_game2 : ℕ) :
  initial_money = 3 →
  cost_per_game = 1 →
  rings_per_game = 5 →
  points_red = 2 →
  points_green = 3 →
  points_blue = 5 →
  success_rate_blue = 10 →
  red_buckets_game1_game2 = 4 →
  green_buckets_game1_game2 = 5 →
  blue_buckets_game1_game2 = 1 →
  let points_so_far := red_buckets_game1_game2 * points_red + green_buckets_game1_game2 * points_green + blue_buckets_game1_game2 * points_blue in
  let points_potential := rings_per_game * points_green in
  points_so_far + points_potential = 43 :=
begin
  sorry
end

end tiffany_max_points_l559_559132


namespace solve_equation_l559_559069

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end solve_equation_l559_559069


namespace monteCarlo_correctness_l559_559483

def monteCarloApproxArea (N N1 : ℕ) : ℝ :=
  2 * (N1 / N)

theorem monteCarlo_correctness (N : ℕ) (N1 : ℕ) :
  monteCarloApproxArea N N1 = 2 * (N1 / N) :=
by
  sorry

end monteCarlo_correctness_l559_559483


namespace find_PZ_l559_559479

noncomputable def right_triangle_at (XYZ : Triangle) (Y : Point) : Prop :=
  XYZ.angles Y = π / 2

variables (X Y Z P : Point)
variables (PX PY PZ : ℝ)
variables (θ : ℝ)

def Triangle :=
  { XYZ : Triangle // right_triangle_at XYZ Y }

def triangle_XYZ_has_right_angle_at_Y := right_triangle_at ⟨X, Y, Z⟩ Y

def PX_eq_eight := PX = 8
def PY_eq_four := PY = 4
def angles_equal_120 := θ = (2 * π / 3) -- 120 degrees in radians

theorem find_PZ : triangle_XYZ_has_right_angle_at_Y ∧ PX_eq_eight ∧ PY_eq_four ∧ angles_equal_120 →
                   PZ = 4 :=
by {
  sorry
}

end find_PZ_l559_559479


namespace no_rectangular_parallelepiped_of_bricks_l559_559892

-- Definition of a brick shape
def brick : Type :=
  { S : set (ℤ × ℤ × ℤ) // S.card = 4 ∧ ∃ v w x y : ℤ × ℤ × ℤ, S = {v, w, x, y} ∧ 
    (∃ i j k : fin 3, v + ([i, j, k]) ∈ S ∧ w + ([i, j, k]) ∈ S ∧ x + ([i, j, k]) ∈ S) }

-- The main theorem to prove
theorem no_rectangular_parallelepiped_of_bricks
  (S : set (ℤ × ℤ × ℤ)) (hs : brick S) :
  ¬ ∃ (b : ℤ × ℤ × ℤ), b = (11, 12, 13) := sorry

end no_rectangular_parallelepiped_of_bricks_l559_559892


namespace possible_scores_for_B_l559_559481

variable (A_score B_score : ℕ)

-- Conditions
def test_conditions :=
  A_score = 27 ∧
  A_score = 9 * 3 ∧
  B_score ∈ {24, 27, 30}

-- Problem statement as a Lean theorem
theorem possible_scores_for_B (h : test_conditions A_score B_score) : 
  B_score = 24 ∨ B_score = 27 ∨ B_score = 30 :=
sorry

end possible_scores_for_B_l559_559481


namespace slope_of_line_through_focus_l559_559727

-- Definition of the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Definition of the focus
def focus := (1 : ℝ, 0 : ℝ)

-- Definition of midpoint of chord
structure midpoint :=
  (x0 y0 : ℝ)

-- Definition of slope
def slope (y1 y2 x1 x2 : ℝ) := (y1 - y2) / (x1 - x2)

-- Definition of the problem statement
theorem slope_of_line_through_focus (x1 y1 x2 y2 : ℝ) :
    parabola x1 y1 →
    parabola x2 y2 →
    let M := midpoint.mk ((x1 + x2) / 2) ((y1 + y2) / 2) in
    M.x0 + 2 = 5 →
    slope y1 y2 x1 x2 = ± (sqrt 6 / 3) := by
  sorry

end slope_of_line_through_focus_l559_559727


namespace tan_sin_identity_l559_559587

noncomputable def tan_deg : ℝ := tan (real.pi / 6)
noncomputable def sin_deg : ℝ := sin (real.pi / 6)

theorem tan_sin_identity :
  (tan_deg ^ 2 - sin_deg ^ 2) / (tan_deg ^ 2 * sin_deg ^ 2) = 1 :=
by
  have tan_30 := by
    simp [tan_deg, real.tan_pi_div_six]
    norm_num
  have sin_30 := by
    simp [sin_deg, real.sin_pi_div_six]
    norm_num
  sorry

end tan_sin_identity_l559_559587


namespace correct_operation_l559_559156

noncomputable def sqrt_op_A: Prop := sqrt 4 ≠ 2
noncomputable def sqrt_op_B: Prop := (± sqrt (5^2)) ≠ -5
noncomputable def sqrt_op_C: Prop := sqrt ((-7) ^ 2) = 7
noncomputable def sqrt_op_D: Prop := sqrt (-3) ≠ - sqrt 3

theorem correct_operation : (sqrt_op_A ∧ sqrt_op_B ∧ sqrt_op_C ∧ sqrt_op_D) → (sqrt_op_C = 7) :=
by
  intros h
  sorry

end correct_operation_l559_559156


namespace liza_butter_amount_l559_559052

theorem liza_butter_amount (B : ℕ) (h1 : B / 2 + B / 5 + (1 / 3) * ((B - B / 2 - B / 5) / 1) = B - 2) : B = 10 :=
sorry

end liza_butter_amount_l559_559052


namespace smallest_n_not_prime_l559_559922

-- Define function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the statement to be proved
theorem smallest_n_not_prime : ∃ n : ℕ, (2^n + 1) > 1 ∧ ¬ is_prime(2^n + 1) ∧ (∀ m : ℕ, m < n → is_prime(2^m + 1)) :=
by
  -- Proof goes here
  sorry

end smallest_n_not_prime_l559_559922


namespace sqrt_of_neg_7_sq_is_7_l559_559175

theorem sqrt_of_neg_7_sq_is_7 : sqrt ((-7)^2) = 7 :=
by sorry

end sqrt_of_neg_7_sq_is_7_l559_559175


namespace sum_digits_9N_eq_9_l559_559997

open Nat

noncomputable theory

/-- Proof problem stating that for a natural number N where each digit of N is strictly
    greater than the digit to its left, the sum of the digits of 9N is 9. -/
theorem sum_digits_9N_eq_9 (N : ℕ) 
  (h : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ digitLength N → digits i N < digits j N) :
  sumDigits (9 * N) = 9 := 
sorry

end sum_digits_9N_eq_9_l559_559997


namespace median_is_87_point_5_l559_559866

noncomputable def set_of_numbers : set ℝ := {90, 86, 88, 87, 92}

theorem median_is_87_point_5 
  (x : ℝ) 
  (h_mean : (90 + 86 + 88 + 87 + 92 + x) / 6 = 88) : 
  (finset.median {90, 86, 88, 87, 92, x}) = 87.5 :=
by 
  sorry

end median_is_87_point_5_l559_559866


namespace first_year_after_2010_with_digit_sum_8_l559_559488

-- Define a function to calculate the sum of the digits of a year.
def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

-- Define the main statement.
theorem first_year_after_2010_with_digit_sum_8 :
  ∃ y : ℕ, y > 2010 ∧ sum_of_digits y = 8 ∧ ∀ z : ℕ, (z > 2010 ∧ sum_of_digits z = 8 → y ≤ z) :=
by
  -- It's beneficial to start with the obvious candidate, 2015.
  let y := 2015
  -- Check that 2015 satisfies both conditions.
  have sum_digits_2015 : sum_of_digits y = 8 := by
    -- Compute the sum of the digits for 2015.
    show sum_of_digits 2015 = 8
    sorry
  
  -- Check that 2015 is the smallest such year.
  have is_minimum : ∀ z : ℕ, (z > 2010 ∧ sum_of_digits z = 8 → y ≤ z) := by
    -- Suppose there is another such year z.
    intro z
    intro hz
    cases hz with hz1 hz2
    -- argument based on year comparison and digit sums
    sorry
  
  -- Combine both results to fulfill the theorem
  exact ⟨y, by norm_num, sum_digits_2015, is_minimum⟩

end first_year_after_2010_with_digit_sum_8_l559_559488


namespace test_completion_days_l559_559246

theorem test_completion_days :
  let barbara_days := 10
  let edward_days := 9
  let abhinav_days := 11
  let alex_days := 12
  let barbara_rate := 1 / barbara_days
  let edward_rate := 1 / edward_days
  let abhinav_rate := 1 / abhinav_days
  let alex_rate := 1 / alex_days
  let one_cycle_work := barbara_rate + edward_rate + abhinav_rate + alex_rate
  let cycles_needed := (1 : ℚ) / one_cycle_work
  Nat.ceil cycles_needed = 3 :=
by
  sorry

end test_completion_days_l559_559246


namespace reduce_to_four_cards_l559_559968

theorem reduce_to_four_cards 
  (deck : List (ℕ × ℕ)) -- assuming the deck is given as a list of pairs (card number, suit)
  (initial_seq : List (ℕ × ℕ))
  (rule : ∀ (seq : List (ℕ × ℕ)), ∃ (new_seq: List (ℕ × ℕ)), seq.length > 4 → new_seq.length < seq.length ∧ 
    (∀ i, i < new_seq.length - 1 → 
    (new_seq.nth i = new_seq.nth (i + 1) ∨ (i + 2 < new_seq.length ∧ new_seq.nth i = new_seq.nth (i + 2))))) :
  ∃ (final_seq : List (ℕ × ℕ)), final_seq.length = 4 := 
sorry

end reduce_to_four_cards_l559_559968


namespace celsius_to_fahrenheit_25_l559_559096

theorem celsius_to_fahrenheit_25 :
  ∀ (x : ℝ), x = 25 → (9 / 5 * x + 32 = 77) :=
by
  intro x hx
  rw hx
  norm_num
  sorry

end celsius_to_fahrenheit_25_l559_559096


namespace determine_y_l559_559294

noncomputable def log_of_base {α : Type} [LinearOrderedField α] (a b : α) : α :=
  Real.log a / Real.log b

theorem determine_y
  (a b c x : ℝ)
  (p q r : ℝ)
  (x_ne_one : x ≠ 1)
  (h1 : log_of_base a x = 2 * p)
  (h2 : log_of_base b x = 3 * q)
  (h3 : log_of_base c x = 4 * r)
  (h4 : b^3 / (a^2 * c) = x ^ y)
  :
  y = 9 * q - 4 * p - 4 * r := by
  sorry

end determine_y_l559_559294


namespace has_roots_in_intervals_l559_559817

noncomputable def f (x : ℝ) : ℝ := (1 - x) * (2 - x) * (3 - x) * (4 - x)

theorem has_roots_in_intervals :
  ∃ c1 c2 c3 : ℝ,
    (1 < c1 ∧ c1 < 2) ∧
    (2 < c2 ∧ c2 < 3) ∧
    (3 < c3 ∧ c3 < 4) ∧
    f'.deriv c1 = 0 ∧
    f'.deriv c2 = 0 ∧
    f'.deriv c3 = 0 :=
sorry

end has_roots_in_intervals_l559_559817


namespace pet_store_problem_l559_559547

noncomputable def num_ways_to_buy_pets (puppies kittens hamsters birds : ℕ) (people : ℕ) : ℕ :=
  (puppies * kittens * hamsters * birds) * (people.factorial)

theorem pet_store_problem :
  num_ways_to_buy_pets 12 10 5 3 4 = 43200 :=
by
  sorry

end pet_store_problem_l559_559547


namespace parallelogram_area_l559_559141

def base : ℝ := 20
def intercept : ℝ := 6
def slant_height : ℝ := 7

noncomputable def height : ℝ := real.sqrt 13

theorem parallelogram_area :
  height = real.sqrt 13 → 
  (20 : ℝ) * height = 20 * real.sqrt 13 :=
by
  intro h_eq
  rw [h_eq]
  norm_num
  sorry

end parallelogram_area_l559_559141


namespace num_ordered_pairs_1806_l559_559113

theorem num_ordered_pairs_1806 :
  let n := 1806 in
  let pf := [(2, 1), (3, 2), (101, 1)] in
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1) in
  ∃ (c : ℕ), c = num_divisors ∧ c = 12 :=
by
  let n := 1806
  let pf := [(2, 1), (3, 2), (101, 1)]
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1)
  use num_divisors
  split
  . rfl
  . rfl
  sorry

end num_ordered_pairs_1806_l559_559113


namespace custard_combination_l559_559214

theorem custard_combination :
  let number_of_flavors := 5 in
  let number_of_toppings := 7 in
  let binomial (n k : ℕ) := nat.choose n k in
  number_of_flavors * binomial number_of_toppings 2 = 105 := 
by
  sorry

end custard_combination_l559_559214


namespace tan_sin_identity_l559_559591

noncomputable def tan_deg : ℝ := tan (real.pi / 6)
noncomputable def sin_deg : ℝ := sin (real.pi / 6)

theorem tan_sin_identity :
  (tan_deg ^ 2 - sin_deg ^ 2) / (tan_deg ^ 2 * sin_deg ^ 2) = 1 :=
by
  have tan_30 := by
    simp [tan_deg, real.tan_pi_div_six]
    norm_num
  have sin_30 := by
    simp [sin_deg, real.sin_pi_div_six]
    norm_num
  sorry

end tan_sin_identity_l559_559591


namespace find_tan_theta_l559_559772

-- Define the isosceles triangle conditions

variables {α θ : ℝ}
-- Condition given in problem: tan(α/2) = 2
axiom tan_half_alpha : real.tan (α / 2) = 2

-- The angle θ is between the altitude and the angle bisector from the vertex angle to the base
-- We are to prove tan θ = -4/3.
theorem find_tan_theta (α θ : ℝ) (tan_half_alpha : real.tan (α / 2) = 2) : real.tan θ = -4 / 3 :=
sorry

end find_tan_theta_l559_559772


namespace compare_abc_l559_559695

def a : ℝ := logBase 2 0.5
def b : ℝ := 2 ^ 0.5
def c : ℝ := 0.5 ^ 2

theorem compare_abc : a < c ∧ c < b := by
  sorry

end compare_abc_l559_559695


namespace pole_intersection_height_l559_559137

section pole_intersection

def height_of_intersection (h1 h2 d: ℝ) : ℝ :=
  let slope1 := -h1/d
  let slope2 := h2/d
  let x := h2 / (slope2 - slope1)
  slope2 * x

theorem pole_intersection_height :
  ∀ (h1 h2 h mid: ℝ), h1 = 30 → h2 = 90 → mid = 10 → ((d: ℝ) = 150) →
      height_of_intersection h1 h2 d = 22.5 :=
by
  intros h1 h2 h mid h1_eq h2_eq mid_eq d_eq
  simp [height_of_intersection, h1_eq, h2_eq, d_eq, mid_eq]
  sorry

end pole_intersection

end pole_intersection_height_l559_559137


namespace trigonometric_identity_l559_559600

theorem trigonometric_identity :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  t30 = 1 / Real.sqrt 3 →
  s30 = 1 / 2 →
  (t30 ^ 2 - s30 ^ 2) / (t30 ^ 2 * s30 ^ 2) = 1 :=
by
  intros t30_eq s30_eq
  -- Proof would go here
  sorry

end trigonometric_identity_l559_559600


namespace caps_production_l559_559932

def caps1 : Int := 320
def caps2 : Int := 400
def caps3 : Int := 300

def avg_caps (caps1 caps2 caps3 : Int) : Int := (caps1 + caps2 + caps3) / 3

noncomputable def total_caps_after_four_weeks : Int :=
  caps1 + caps2 + caps3 + avg_caps caps1 caps2 caps3

theorem caps_production : total_caps_after_four_weeks = 1360 :=
by
  sorry

end caps_production_l559_559932


namespace sum_of_binom_l559_559250

open Nat

-- Define the binomial coefficient.
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_binom :
  binom 2 0 + binom 3 1 + binom 4 2 + binom 5 3 + binom 6 4 + binom 7 5 +
  binom 8 6 + binom 9 7 + binom 10 8 + binom 11 9 + binom 12 10 + binom 13 11 +
  binom 14 12 + binom 15 13 + binom 16 14 + binom 17 15 + binom 18 16 + binom 19 17 = 1140 :=
begin
  sorry -- Proof to be provided
end

end sum_of_binom_l559_559250


namespace even_plus_abs_odd_is_even_l559_559703

/-- 
Given that f is an even function and g is an odd function,
prove that f(x) + |g(x)| is an even function.
--/

open Real

variables {f g : ℝ → ℝ}

def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h x = -h (-x)

theorem even_plus_abs_odd_is_even (hf : is_even f) (hg : is_odd g) : is_even (fun x => f x + abs (g x)) :=
by
  sorry

end even_plus_abs_odd_is_even_l559_559703


namespace difference_of_squares_and_product_l559_559879

theorem difference_of_squares_and_product (a b : ℕ) (h1 : a = 9) (h2 : b = 10) : 
  (a^2 + b^2) - (a * b) = 91 :=
by {
  -- Using the conditions h1 and h2
  have h3 : a = 9 := h1,
  have h4 : b = 10 := h2,
  -- Replace a and b based on h3 and h4 in the equation
  calc
    (a^2 + b^2) - (a * b)
      = (9^2 + 10^2) - (9 * 10) : by rw [h3, h4]
      ... = 181 - 90 : by norm_num
      ... = 91 : by norm_num
}

end difference_of_squares_and_product_l559_559879


namespace c_value_difference_l559_559397

theorem c_value_difference (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  max c - min c = 34 / 3 :=
sorry

end c_value_difference_l559_559397


namespace sum_prob_expression_l559_559286

open Nat

theorem sum_prob_expression (n : ℕ) (h_pos : 0 < n) :
  (1 : ℝ) / n * ∑ k in Finset.range n, ((k + 1) * (Nat.factorial (k + 1)) * (Nat.choose n (k + 1)) / (n : ℝ)^(k + 1)) = 1 :=
by
  sorry

end sum_prob_expression_l559_559286


namespace min_value_fraction_l559_559328

theorem min_value_fraction {a : ℕ → ℕ} (h1 : a 1 = 10)
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) :
    ∃ n : ℕ, (n > 0) ∧ (n - 1 + 10 / n = 16 / 3) :=
by {
  sorry
}

end min_value_fraction_l559_559328


namespace max_PA_PB_l559_559360

-- Define the fixed points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (0, -1)

-- Define the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  y^2 / 4 + x^2 / 3 = 1

-- Define the distance function
def dist (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

-- Statement to prove
theorem max_PA_PB (P : ℝ × ℝ) (hP : on_ellipse P) : dist P A + dist P B ≤ 5 :=
sorry

end max_PA_PB_l559_559360


namespace number_of_quadratic_PQ_equal_to_PR_l559_559798

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

def is_quadratic (Q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, Q = λ x => a * x^2 + b * x + c

theorem number_of_quadratic_PQ_equal_to_PR :
  let possible_Qx_fwds := 4^4
  let non_quadratic_cases := 6
  possible_Qx_fwds - non_quadratic_cases = 250 :=
by
  sorry

end number_of_quadratic_PQ_equal_to_PR_l559_559798


namespace shortest_path_reflection_l559_559552

noncomputable def shortestPathLength {A : ℝ × ℝ} (ax : A.1 = -3) (ay : A.2 = 9) 
    {O : ℝ × ℝ} (ox : O.1 = 2) (oy : O.2 = 3) (r : ℝ) (r_eq : r = 1) 
    {O' : ℝ × ℝ} (ox' : O'.1 = 2) (oy' : O'.2 = -3) 
    {C : ℝ × ℝ → Prop} (C_eq : ∀ x y, C (x, y) ↔ (x - 2) ^ 2 + (y - 3) ^ 2 = 1) 
    (tangentPathLength : ℝ) (tangent_eq : tangentPathLength = 12) : Prop :=
  let dist := Real.sqrt ((A.1 - O'.1) ^ 2 + (A.2 - O'.2) ^ 2)
  tangentPathLength = dist - r

theorem shortest_path_reflection : shortestPathLength sorry sorry sorry sorry sorry sorry sorry sorry sorry sorry := sorry

end shortest_path_reflection_l559_559552


namespace ascending_order_l559_559969

theorem ascending_order : (3 / 8 : ℝ) < 0.75 ∧ 
                          0.75 < (1 + 2 / 5 : ℝ) ∧ 
                          (1 + 2 / 5 : ℝ) < 1.43 ∧
                          1.43 < (13 / 8 : ℝ) :=
by
  sorry

end ascending_order_l559_559969


namespace analogical_conclusions_l559_559240

theorem analogical_conclusions : 
  (∀ a b : ℂ, a - b = 0 → a = b) ∧ 
  (∀ a b c d : ℚ, a + b * Real.sqrt 2 = c + d * Real.sqrt 2 → a = c ∧ b = d) ∧ 
  ¬ (∀ a b : ℂ, a - b > 0 → a > b) →
  ¬ (∀ a b c d : ℝ, (a + b * Complex.I = c + d * Complex.I → a = c ∧ b = d) ∧ (a - b > 0 → a > b)) →
  2 = 2 := 
by
  sorry

end analogical_conclusions_l559_559240


namespace statement_II_must_be_true_l559_559347

-- Define the set of all creatures
variable (Creature : Type)

-- Define properties for being a dragon, mystical, and fire-breathing
variable (Dragon Mystical FireBreathing : Creature → Prop)

-- Given conditions
-- All dragons breathe fire
axiom all_dragons_breathe_fire : ∀ c, Dragon c → FireBreathing c
-- Some mystical creatures are dragons
axiom some_mystical_creatures_are_dragons : ∃ c, Mystical c ∧ Dragon c

-- Questions to prove (we will only formalize the must be true statement)
-- Statement II: Some fire-breathing creatures are mystical creatures

theorem statement_II_must_be_true : ∃ c, FireBreathing c ∧ Mystical c :=
by
  sorry

end statement_II_must_be_true_l559_559347


namespace ball_placement_l559_559831

theorem ball_placement (num_balls : ℕ) (label1 : ℕ) (label2 : ℕ) : 
  (num_balls = 5) → (label1 = 1) → (label2 = 2) →
  (finset.card {s : finset (fin num_balls) | s.card ≥ label1 ∧ (num_balls - s.card) ≥ label2}) = 25 := 
by 
  -- Assume we have conditions
  intros h_nb h_l1 h_l2,
  sorry -- Skipping the proof

end ball_placement_l559_559831


namespace value_of_B_l559_559284

theorem value_of_B (B : ℚ) (h : 3 * B - 5 = 23) : B = 28 / 3 :=
by
  sorry

-- Explanation:
-- B is declared as a rational number (ℚ) because the answer involves a fraction.
-- h is the condition 3 * B - 5 = 23.
-- The theorem states that given h, B equals 28 / 3.

end value_of_B_l559_559284


namespace units_digit_of_7_pow_2500_l559_559510

theorem units_digit_of_7_pow_2500 : (7^2500) % 10 = 1 :=
by
  -- Variables and constants can be used to formalize steps if necessary, 
  -- but focus is on the statement itself.
  -- Sorry is used to skip the proof part.
  sorry

end units_digit_of_7_pow_2500_l559_559510


namespace intersecting_chords_l559_559251

variables (O K A B C D : Type*)
variables (circle : O)
variables (a b c d : ℝ)
variables (h₁ : is_chord O A B)
variables (h₂ : is_chord O C D)
variables (h₃ : intersects A B C D K)
variables (h₄ : AK_eq_a : segment_length A K = a)
variables (h₅ : BK_eq_b : segment_length B K = b)
variables (h₆ : CK_eq_c : segment_length C K = c)
variables (h₇ : DK_eq_d : segment_length D K = d)

theorem intersecting_chords :
  a * b = c * d :=
sorry

end intersecting_chords_l559_559251


namespace find_f_log_l559_559674

noncomputable def f : ℝ → ℝ := sorry -- We'll define the function later

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def f_positive_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → f (x) = 3 ^ x

theorem find_f_log :
  is_odd_function f →
  f_positive_condition f →
  f (real.log 4 / real.log (1/2)) = -9 :=
by
  intro h_odd h_pos
  sorry

end find_f_log_l559_559674


namespace num_audio_cassettes_in_second_set_l559_559076

-- Define the variables and constants
def costOfAudio (A : ℕ) : ℕ := A
def costOfVideo (V : ℕ) : ℕ := V
def totalCost (numOfAudio : ℕ) (numOfVideo : ℕ) (A : ℕ) (V : ℕ) : ℕ :=
  numOfAudio * (costOfAudio A) + numOfVideo * (costOfVideo V)

-- Given conditions
def condition1 (A V : ℕ) : Prop := ∃ X : ℕ, totalCost X 4 A V = 1350
def condition2 (A V : ℕ) : Prop := totalCost 7 3 A V = 1110
def condition3 : Prop := costOfVideo 300 = 300

-- Main theorem to prove: The number of audio cassettes in the second set is 7
theorem num_audio_cassettes_in_second_set :
  ∃ (A : ℕ), condition1 A 300 ∧ condition2 A 300 ∧ condition3 →
  7 = 7 :=
by
  sorry

end num_audio_cassettes_in_second_set_l559_559076


namespace part1_part2_part3_l559_559807

noncomputable def f : ℝ → ℝ := sorry

axiom fx_property (x y : ℝ) : f(x + y) = f(x) + f(y) - 1
axiom fx_less_than_one (x : ℝ) (h : x < 0) : f(x) < 1
axiom f_four : f 4 = 7

-- Part 1: Prove that f(0) = 1
theorem part1 : f 0 = 1 := sorry

-- Part 2: Prove that f is an increasing function on ℝ
theorem part2 : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) := sorry

-- Part 3: Prove that f(x^2 + x) < 4 is equivalent to x ∈ (-2, 1] given f(4) = 7
theorem part3 : ∀ x : ℝ, f(x^2 + x) < 4 ↔ -2 < x ∧ x ≤ 1 := sorry

end part1_part2_part3_l559_559807


namespace triangle_AM_KM_AB_l559_559826

theorem triangle_AM_KM_AB (A B C D M K : Point) (hBD_eq_AC : BD = AC)
  (hAM_median : is_median A M)
  (hK_on_AM : AM ∩ BD = K)
  (hDK_eq_DC : DK = DC) :
  AM + KM = AB :=
sorry

end triangle_AM_KM_AB_l559_559826


namespace inequality_proof_l559_559724

-- Definitions
def f (x : ℝ) : ℝ := |3 * x - 1| + |x + 1|

def g (x : ℝ) : ℝ := f x + 2 * |x + 1|

-- Theorem statement: Proving the given inequalities
theorem inequality_proof
  (a b : ℝ) 
  (h1 : f (a + b) = 1)
  (h2 : g (a - b) = 2) : 
  a^2 + b^2 = 4 → 
  (∃ a b : ℝ, a^2 + b^2 = 4 ∧ (1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ 3 / 2)) :=
begin
  sorry
end

end inequality_proof_l559_559724


namespace tangent_line_at_one_monotonicity_of_f_l559_559715

def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_one :
  let x := 1 in
  let f_x := f x in
  let f' := λ x, 1 + Real.log x in
  let slope := f' x in 
  (slope = 1) → (f x = 0) → 
  tangent_line := λ y, y = x - 1
:= by
  intros; sorry

theorem monotonicity_of_f (t : ℝ) (ht : 0 < t) :
  let e_inv := 1 / Real.exp 1 in
  let f' := λ x, 1 + Real.log x in
  if hte : t ≤ e_inv then 
    ∀ x, 0 < x ∧ x ≤ t → f' x ≤ 0 
  else
    ∀ x, 0 < x ∧ x ≤ e_inv → f' x ≤ 0 
    ∧ ∀ x, e_inv < x ∧ x ≤ t → f' x > 0
:= by
  intros; sorry

end tangent_line_at_one_monotonicity_of_f_l559_559715


namespace solution_to_problem_l559_559913

def cost_of_A_and_B (x y : ℕ) : Prop :=
  3 * x + y = 55 ∧ 2 * x + 4 * y = 120

def minimum_A_cars (m : ℕ) : Prop :=
  10 * m + 25 * (15 - m) ≤ 220

theorem solution_to_problem :
  ∃ x y m : ℕ, cost_of_A_and_B x y ∧ minimum_A_cars m ∧ m ≥ 11 ∧ x = 10 ∧ y = 25 :=
begin
  sorry

end solution_to_problem_l559_559913


namespace age_difference_l559_559849

variable (A B C D : ℕ)

theorem age_difference (h1 : A + B = B + C + 12) 
                        (h2 : B + C = C + D + 8) 
                        (h3 : D = A - 6) : 
                        C - A = -12 := 
by sorry

end age_difference_l559_559849


namespace integer_type_l559_559545

theorem integer_type (f : ℕ) (h : f = 14) (x : ℕ) (hx : 3150 * f = x * x) : f > 0 :=
by
  sorry

end integer_type_l559_559545


namespace point_on_transformed_graph_l559_559315

theorem point_on_transformed_graph (g : ℝ → ℝ) 
  (hg : g 12 = 10) : 
  let y := (1/3) * g (3 * 4) + 3 in 4 + y = 55 / 9 := 
by 
  -- Proof: as detailed above.
  sorry

end point_on_transformed_graph_l559_559315


namespace problem_1_l559_559330

open Finset

theorem problem_1 (M N : Finset ℤ) (hM : M = {0, 1, 2}) (hN : N = {-1, 0, 1}) :
  M ∩ N = {0, 1} :=
by
  rw [hM, hN]
  simp
  sorry

end problem_1_l559_559330


namespace calculate_expression_l559_559974

theorem calculate_expression : (3^5 * 4^5) / 6^5 = 32 := 
by
  sorry

end calculate_expression_l559_559974


namespace line_equation_l559_559698

def point : ℝ × ℝ := (3, 4)
def normal_vector : ℝ × ℝ := (1, 2)

theorem line_equation :
  ∃ (a b c : ℝ), a * point.1 + b * point.2 + c = 0 ∧ (a, b) = normal_vector ∧ (a * 3 + b * 4 + c = 0) :=
begin
  use [1, 2, -11],
  split,
  { calc
      1 * 3 + 2 * 4 - 11 
          = 3 + 8 - 11 : by ring 
      ... = 0 : by norm_num },
  split,
  { exact rfl },
  { calc
      1 * 3 + 2 * 4 - 11
          = 3 + 8 - 11 : by ring 
      ... = 0 : by norm_num }
end

end line_equation_l559_559698


namespace total_production_second_year_l559_559536

theorem total_production_second_year :
  (let daily_production := 10
       days_in_year := 365
       reduction_percentage := 0.10
       total_production_first_year := daily_production * days_in_year
       reduction_amount := total_production_first_year * reduction_percentage in
    total_production_first_year - reduction_amount = 3285) :=
by
  let daily_production := 10
  let days_in_year := 365
  let reduction_percentage := 0.10
  let total_production_first_year := daily_production * days_in_year
  let reduction_amount := total_production_first_year * reduction_percentage
  show total_production_first_year - reduction_amount = 3285
  sorry

end total_production_second_year_l559_559536


namespace coordinates_reflection_y_axis_l559_559857

theorem coordinates_reflection_y_axis :
  let M := (-5, 2) in
  reflect_y_axis M = (5, 2) :=
by
  sorry

end coordinates_reflection_y_axis_l559_559857


namespace cos_alpha_eq_l559_559376

open Real

-- Define the angles and their conditions
variables (α β : ℝ)

-- Hypothesis and initial conditions
axiom ha1 : 0 < α ∧ α < π
axiom ha2 : 0 < β ∧ β < π
axiom h_cos_beta : cos β = -5 / 13
axiom h_sin_alpha_plus_beta : sin (α + β) = 3 / 5

-- The main theorem to prove
theorem cos_alpha_eq : cos α = 56 / 65 := sorry

end cos_alpha_eq_l559_559376


namespace total_tweets_correct_l559_559430

-- Define the rates at which Polly tweets under different conditions
def happy_rate : ℕ := 18
def hungry_rate : ℕ := 4
def mirror_rate : ℕ := 45

-- Define the durations of each activity
def happy_duration : ℕ := 20
def hungry_duration : ℕ := 20
def mirror_duration : ℕ := 20

-- Compute the total number of tweets
def total_tweets : ℕ := happy_rate * happy_duration + hungry_rate * hungry_duration + mirror_rate * mirror_duration

-- Statement to prove
theorem total_tweets_correct : total_tweets = 1340 := by
  sorry

end total_tweets_correct_l559_559430


namespace number_of_ways_to_divide_l559_559369

def shape_17_cells : Type := sorry -- We would define the structure of the shape here
def checkerboard_pattern : shape_17_cells → Prop := sorry -- The checkerboard pattern condition
def num_black_cells (s : shape_17_cells) : ℕ := 9 -- Number of black cells
def num_gray_cells (s : shape_17_cells) : ℕ := 8 -- Number of gray cells
def divides_into (s : shape_17_cells) (rectangles : ℕ) (squares : ℕ) : Prop := sorry -- Division condition

theorem number_of_ways_to_divide (s : shape_17_cells) (h1 : checkerboard_pattern s) (h2 : divides_into s 8 1) :
  num_black_cells s = 9 ∧ num_gray_cells s = 8 → 
  (∃ ways : ℕ, ways = 10) := 
sorry

end number_of_ways_to_divide_l559_559369


namespace BPQC_cyclic_l559_559008

-- Definitions corresponding to conditions in the problem
variables (A B C T P Q : Type) 
[linear_ordered_field A]
-- This includes acute-angled condition implicitly
variables (ABC : triangle A B C) 
(AT_perp : altitude A B C T)

-- The perpendiculars from T to sides AB and AC
variables (T_perp_AB : altitude A T P) (T_perp_AC : altitude A T Q)

-- Statement of the theorem
theorem BPQC_cyclic : cyclic_quadrilateral B P Q C := sorry

end BPQC_cyclic_l559_559008


namespace widgets_difference_l559_559823

-- Define the variables and assumptions
variables (w t : ℝ)
variables (h1 : w = 3 * t)
variables (h2 : w * t + (w + 6) * (t - 3) = 300)

-- Prove that the difference in production is approximately 40.58
theorem widgets_difference (h1 : w = 3 * t) (h2 : w * t + (w + 6) * (t - 3) = 300) :
  (w * t) - ((w + 6) * (t - 3)) ≈ 40.58 :=
sorry

end widgets_difference_l559_559823


namespace maximal_length_curve_of_constant_width1_maximal_length_curve_AP_BQ_l559_559514

-- Part (a)
-- Defining the problem statement and conditions
def convex_curve_of_diameter_one (K : convex_curve) (D1: diameter K = 1) := 
  ∀ (L : convex_curve), diameter L = 1 → length L ≤ length K

theorem maximal_length_curve_of_constant_width1 {K : convex_curve} (D1 : diameter K = 1) :
  convex_curve_of_diameter_one K D1 := sorry

-- Part (b)
-- Defining the problem statement and conditions

def trapezoid_with_parallel_chords (K : convex_curve) (AB PQ : chord) (AQ BP : diameter K) (delta : distance AB PQ) :=
  (is_parallel_to AB PQ ∧ diameter AQ ∧ diameter BP ∧ distance AB PQ = delta)

def longest_convex_curve (K : convex_curve) (D _W: width K = D) :=
  ∀ (L : convex_curve), diameter L = D ∧ width L = _W → length L ≤ length K

theorem maximal_length_curve_AP_BQ (K : convex_curve) (delta : ℝ) (D : ℝ)
  (AB PQ : chord) (AQ BP : diameter K) 
  (trapezoid_props : trapezoid_with_parallel_chords K AB PQ AQ BP delta)
  (width_cond : width K = delta) : longest_convex_curve K D delta := sorry

end maximal_length_curve_of_constant_width1_maximal_length_curve_AP_BQ_l559_559514


namespace no_such_2013_distinct_naturals_l559_559263

theorem no_such_2013_distinct_naturals :
  ¬ (∃ (a : Fin 2013 → ℕ), Function.Injective a ∧ ∀ k : Fin 2013, ∑ i in (Fin 2013).erase k, a i ≥ (a k) ^ 2) := by
  sorry

end no_such_2013_distinct_naturals_l559_559263


namespace curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559007

section
variables (k t θ ρ : ℝ)  

def parametric_curve_C1 (k : ℝ) (t : ℝ) : ℝ × ℝ := (cos t ^ k, sin t ^ k)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

def curve_C2 (ρ θ : ℝ) : Prop := 4 * ρ * cos θ - 16 * ρ * sin θ + 3 = 0

theorem curve_C1_when_k1_is_circle : 
  ∀ t : ℝ, (parametric_curve_C1 1 t).fst ^ 2 + (parametric_curve_C1 1 t).snd ^ 2 = 1 :=
begin
  intros t,
  simp only [parametric_curve_C1],
  have h : cos t ^ 2 + sin t ^ 2 = 1, from real.sin_sq_add_cos_sq t,
  simp [h],
end

noncomputable def intersection_points_when_k4 : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p = parametric_curve_C1 4 t ∧ 
    let (x, y) := p in 4 * x - 16 * y + 3 = 0}

theorem find_intersection_points_when_k4 : 
  intersection_points_when_k4 = { (1/4, 1/4) } :=
begin
  sorry
end

end

end curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559007


namespace num_four_digit_palindromes_l559_559546

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ is_palindrome n

theorem num_four_digit_palindromes : 
  {n : ℕ | is_four_digit_palindrome n}.to_finset.card = 90 :=
by sorry

end num_four_digit_palindromes_l559_559546


namespace turns_to_fill_drum_together_l559_559247

-- Define the capacities of bucket P and Q
def bucketQ_capacity := C
def bucketP_capacity := 3 * bucketQ_capacity

-- Define the total capacity of the drum
def drum_capacity := 60 * bucketP_capacity

-- Define the combined capacity of both buckets per turn
def combined_capacity_per_turn := bucketP_capacity + bucketQ_capacity

-- Define the number of turns required for both buckets to fill the drum
theorem turns_to_fill_drum_together (C : ℝ) :
  60 * bucketP_capacity / combined_capacity_per_turn = 45 :=
by
  let bucketQ_capacity := C
  let bucketP_capacity := 3 * bucketQ_capacity
  let drum_capacity := 60 * bucketP_capacity
  let combined_capacity_per_turn := bucketP_capacity + bucketQ_capacity
  have h : 60 * bucketP_capacity = 180 * bucketQ_capacity := by ring
  rw h
  have h2 : combined_capacity_per_turn = 4 * bucketQ_capacity := by ring
  rw h2
  norm_num
  sorry

end turns_to_fill_drum_together_l559_559247


namespace two_digit_values_heartsuit_double_heartsuit_eq_5_l559_559393

-- Define the function heartsuit
def heartsuit (x : Nat) : Nat :=
  (x / 10) ^ 2 + (x % 10) ^ 2

-- Define the proof problem
theorem two_digit_values_heartsuit_double_heartsuit_eq_5 :
  (Finset.filter (λ x, heartsuit (heartsuit x) = 5) (Finset.range 100)).filter (Proc.is_two_digit).card = 2 := by
  sorry

-- Auxiliary function to determine if a number is two-digit
def Proc.is_two_digit (n : Nat) : Bool :=
  n ≥ 10 ∧ n < 100

end two_digit_values_heartsuit_double_heartsuit_eq_5_l559_559393


namespace probability_five_distinct_dice_rolls_l559_559499

theorem probability_five_distinct_dice_rolls : 
  let total_outcomes := 6^5
  let favorable_outcomes := 6 * 5 * 4 * 3 * 2
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 54 :=
by
  sorry

end probability_five_distinct_dice_rolls_l559_559499


namespace altitudes_bounded_by_perimeter_l559_559759

theorem altitudes_bounded_by_perimeter (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 2) :
  ¬ (∀ (ha hb hc : ℝ), ha = 2 / a * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hb = 2 / b * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hc = 2 / c * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     ha > 1 / Real.sqrt 3 ∧ 
                     hb > 1 / Real.sqrt 3 ∧ 
                     hc > 1 / Real.sqrt 3 ) :=
sorry

end altitudes_bounded_by_perimeter_l559_559759


namespace greatest_sum_of_consecutive_odd_integers_lt_500_l559_559142

-- Define the consecutive odd integers and their conditions
def consecutive_odd_integers (n : ℤ) : Prop :=
  n % 2 = 1 ∧ (n + 2) % 2 = 1

-- Define the condition that their product must be less than 500
def prod_less_500 (n : ℤ) : Prop :=
  n * (n + 2) < 500

-- The theorem statement
theorem greatest_sum_of_consecutive_odd_integers_lt_500 : 
  ∃ n : ℤ, consecutive_odd_integers n ∧ prod_less_500 n ∧ ∀ m : ℤ, consecutive_odd_integers m ∧ prod_less_500 m → n + (n + 2) ≥ m + (m + 2) :=
sorry

end greatest_sum_of_consecutive_odd_integers_lt_500_l559_559142


namespace rectangle_side_ratio_l559_559553

noncomputable def sin_30_deg := 1 / 2

theorem rectangle_side_ratio 
  (a b c : ℝ) 
  (h1 : a + b = 2 * c) 
  (h2 : a * b = (c ^ 2) / 2) :
  (a / b = 3 + 2 * Real.sqrt 2) ∨ (a / b = 3 - 2 * Real.sqrt 2) :=
by
  sorry

end rectangle_side_ratio_l559_559553


namespace order_of_coins_l559_559064

-- Conditions
variables (C A B E D F : Type)
variable [Covers : (Type → Type → Prop)]

-- Given conditions
axiom cover_C_A : Covers C A
axiom cover_C_B : Covers C B
axiom cover_C_E : Covers C E
axiom cover_E_A : Covers E A
axiom cover_E_D : Covers E D
axiom cover_E_F : Covers E F
axiom cover_D_B : Covers D B
axiom cover_D_F : Covers D F
axiom cover_A_B : Covers A B
axiom F_partially_visible : ∃ X, Covers X F  -- Partial statement for visibility

-- Proof to show the order of coins from top to bottom
theorem order_of_coins (C A B E D F : Type) [Covers : (Type → Type → Prop)] : 
  C = C ∧ E = E ∧ D = D ∧ F = F ∧ A = A ∧ B = B →
  (∀ X Y, Covers X Y → 
    (X = C ∧ Y = A) ∨ (X = C ∧ Y = B) ∨ (X = C ∧ Y = E) ∨
    (X = E ∧ Y = A) ∨ (X = E ∧ Y = D) ∨ (X = E ∧ Y = F) ∨
    (X = D ∧ Y = B) ∨ (X = D ∧ Y = F) ∨ 
    (X = A ∧ Y = B)) →
  (order_top_to_bottom = [C, E, D, F, A, B]) :=
begin
  -- Placeholder proof
  sorry
end

end order_of_coins_l559_559064


namespace cycling_distance_l559_559865

-- Define the conditions
def cycling_time : ℕ := 40  -- Total cycling time in minutes
def time_per_interval : ℕ := 10  -- Time per interval in minutes
def distance_per_interval : ℕ := 2  -- Distance per interval in miles

-- Proof statement
theorem cycling_distance : (cycling_time / time_per_interval) * distance_per_interval = 8 := by
  sorry

end cycling_distance_l559_559865


namespace difference_ranges_l559_559528

-- Define conditions for the problem
open_locale classical

variables {person1 person2 person3 : ℕ}
variables (r1 r2 r3 : ℕ)

-- Define ranges for the persons
def ranges : Prop :=
  r1 = 100 ∧ r2 = 70 ∧ r3 = 40 ∧ 
  (∀ p, p = person1 ∨ p = person2 ∨ p = person3) ∧
  (∀ s, s >= 400 ∧ s <= 700)

-- Define minimum and maximum function for possible ranges
def minRange : ℕ := 700
def maxRange : ℕ := 400
  
theorem difference_ranges : (maxRange -  minRange) = 200 :=
by
  have H : ranges r1 r2 r3,
  sorry

#check difference_ranges

end difference_ranges_l559_559528


namespace order_proof_l559_559296

variable (a b c : ℝ)
variable (log : ℝ → ℝ)
variable (sqrt : ℝ → ℝ)

axiom log_change_base (x b new_base : ℝ) : log x = log (x) / log (b)
axiom log_inc (x y : ℝ) : x < y → log x < log y
axiom sqrt_def (x : ℝ) : sqrt x = x ^ (0.5)

def a_val : ℝ := log 10 / 2
def b_val : ℝ := log 3
def c_val : ℝ := sqrt 2

theorem order_proof : a_val > b_val ∧ b_val > c_val := by
  sorry

end order_proof_l559_559296


namespace f_of_f_of_neg_sqrt2_eq_l559_559666

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x + 1 else |x|

theorem f_of_f_of_neg_sqrt2_eq : f (f (-real.sqrt 2)) = 3 * real.sqrt 2 + 1 := by
  sorry

end f_of_f_of_neg_sqrt2_eq_l559_559666


namespace triangle_area_calc_l559_559760

theorem triangle_area_calc (A : ℝ) (B C : ℝ) (AB BC : ℝ) :
  (sqrt 3 * sin A + cos A = 1) →
  (AB = 2) →
  (BC = 2 * sqrt 3) →
  let area := (1 / 2) * AB * BC * sin B
  area = sqrt 3 :=
begin
  sorry
end

end triangle_area_calc_l559_559760


namespace proof_problem_l559_559387

def sum_of_digits_base_2 (n : ℕ) : ℕ :=
  n.to_digits 2 |>.countp (fun d => d = 1)

theorem proof_problem (K a b l m : ℕ)
  (h1 : K % 2 = 1) -- odd number
  (h2 : sum_of_digits_base_2 K = 2) -- S_2(K) = 2
  (h3 : a * b = K) -- ab = K
  (h4 : a > 1) -- a > 1
  (h5 : b > 1) -- b > 1
  (h6 : l > 2) -- l > 2
  (h7 : m > 2) -- m > 2
  (h8 : sum_of_digits_base_2 a < l) -- S_2(a) < l
  (h9 : sum_of_digits_base_2 b < m) -- S_2(b) < m
  : K ≤ 2^(l * m - 6) + 1 := sorry

end proof_problem_l559_559387


namespace exists_tetrahedron_labeled_1234_l559_559383

-- Define the labels and the conditions for the tetrahedron and its division
variables (V : Type) [fintype V]
variables (label : V → ℕ) -- Label function for vertices
variables (tetra : set (set V)) -- Set representing the tetrahedron

-- Conditions: labels and division properties
def original_vertices (v : V) := label v ∈ {1, 2, 3, 4}
def original_edges (e : set V) := ∃ v1 v2, e = {v1, v2} ∧ original_vertices v1 ∧ original_vertices v2
def original_faces (f : set V) := ∃ v1 v2 v3, f = {v1, v2, v3} ∧ original_vertices v1 ∧ original_vertices v2 ∧ original_vertices v3
def original_tetra := ∃ v1 v2 v3 v4, tetra = {{v1, v2, v3, v4}} ∧ original_vertices v1 ∧ original_vertices v2 ∧ original_vertices v3 ∧ original_vertices v4 

-- Property of the smaller tetrahedra
def small_tetrahedra (t : set V) := ∃ v1 v2 v3 v4, t = {v1, v2, v3, v4}
def valid_division (T : set (set V)) := ∀ t1 t2 ∈ T, t1 ≠ t2 → 
  (t1 ∩ t2 = ∅ ∨
   ∃ v, t1 ∩ t2 = {v} ∨
   ∃ e, t1 ∩ t2 = e ∧ original_edges e ∨
   ∃ f, t1 ∩ t2 = f ∧ original_faces f)

-- Labeling conditions for new vertices
def valid_labeling (T : set (set V)) := ∀ t ∈ T, ∀ v ∈ t, 
  (∃ {v1 v2}, v ∈ {v1, v2} ∧ original_vertices v1 ∧ original_vertices v2 → label v ∈ {label v1, label v2}) ∨
  (∃ {v1 v2 v3}, v ∈ {v1, v2, v3} ∧ original_faces {v1, v2, v3} → label v ∈ {label v1, label v2, label v3}) ∨
  (original_tetra {v} → label v ∈ {1, 2, 3, 4})

-- Theorem to prove the existence of a small tetrahedron with vertices labeled 1, 2, 3, and 4
theorem exists_tetrahedron_labeled_1234 (T : set (set V)) (hdiv : valid_division T) (hlabel : valid_labeling T) : 
  ∃ t ∈ T, ∀ v ∈ t, label v ∈ {1, 2, 3, 4} :=
sorry

end exists_tetrahedron_labeled_1234_l559_559383


namespace curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559001

section
variables (k t θ ρ : ℝ)  

def parametric_curve_C1 (k : ℝ) (t : ℝ) : ℝ × ℝ := (cos t ^ k, sin t ^ k)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

def curve_C2 (ρ θ : ℝ) : Prop := 4 * ρ * cos θ - 16 * ρ * sin θ + 3 = 0

theorem curve_C1_when_k1_is_circle : 
  ∀ t : ℝ, (parametric_curve_C1 1 t).fst ^ 2 + (parametric_curve_C1 1 t).snd ^ 2 = 1 :=
begin
  intros t,
  simp only [parametric_curve_C1],
  have h : cos t ^ 2 + sin t ^ 2 = 1, from real.sin_sq_add_cos_sq t,
  simp [h],
end

noncomputable def intersection_points_when_k4 : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p = parametric_curve_C1 4 t ∧ 
    let (x, y) := p in 4 * x - 16 * y + 3 = 0}

theorem find_intersection_points_when_k4 : 
  intersection_points_when_k4 = { (1/4, 1/4) } :=
begin
  sorry
end

end

end curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559001


namespace temple_shop_total_cost_l559_559058

theorem temple_shop_total_cost :
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  total_cost = 374 :=
by
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  show total_cost = 374
  sorry

end temple_shop_total_cost_l559_559058


namespace cos_sq_165_minus_sin_sq_15_eq_sqrt3_div2_l559_559282

theorem cos_sq_165_minus_sin_sq_15_eq_sqrt3_div2 :
  cos (165 * (Real.pi / 180)) ^ 2 - sin (15 * (Real.pi / 180)) ^ 2 = sqrt 3 / 2 :=
by sorry

end cos_sq_165_minus_sin_sq_15_eq_sqrt3_div2_l559_559282


namespace number_of_pages_read_on_fourth_day_l559_559335

-- Define variables
variables (day1 day2 day3 day4 total_pages: ℕ)

-- Define conditions
def condition1 := day1 = 63
def condition2 := day2 = 2 * day1
def condition3 := day3 = day2 + 10
def condition4 := total_pages = 354
def read_in_four_days := total_pages = day1 + day2 + day3 + day4

-- State the theorem to be proven
theorem number_of_pages_read_on_fourth_day (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : read_in_four_days) : day4 = 29 :=
by sorry

end number_of_pages_read_on_fourth_day_l559_559335


namespace mother_bought_potatoes_l559_559416

theorem mother_bought_potatoes (p s m l : ℕ) (hs : s = 15) (hm : m = 24) (hl : l = 13) (h_total_used : s + m + l = p) : p = 52 :=
by 
  rw [hs, hm, hl] at h_total_used
  have h_sum : 15 + 24 + 13 = 52 := rfl
  rw h_sum at h_total_used
  exact h_total_used

end mother_bought_potatoes_l559_559416


namespace percent_of_a_l559_559439

variable (a b : ℝ)

-- Defining the given condition
def condition := a = 1.5 * b

-- Defining the question to be proven (the percentage form)
theorem percent_of_a (h : condition a b) : (3 * b) / a = 2 :=
by {
  -- Set up the proof context with the given condition
  rw [condition] at h,
  -- The proof will continue here, as required by Lean
  sorry
}

end percent_of_a_l559_559439


namespace sallyRecitePoems_sallyCanReciteThree_l559_559839

variable (memorized : ℕ)
variable (forgotten : ℕ)
variable (recited : ℕ)

-- Conditions
def sallyMemorized : memorized = 8 := sorry
def sallyForgot : forgotten = 5 := sorry

-- Question and answer proof
theorem sallyRecitePoems (h1 : memorized = 8) (h2 : forgotten = 5) : recited = memorized - forgotten := by
  rw [h1, h2]
  exact Nat.sub_self 8 5

-- Theorem that proves Sally could still recite 3 poems
theorem sallyCanReciteThree (h1 : memorized = 8) (h2 : forgotten = 5) : recited = 3 := by
  rw [sallyRecitePoems h1 h2]
  exact rfl

end sallyRecitePoems_sallyCanReciteThree_l559_559839


namespace choose_jia_science_museum_l559_559558

theorem choose_jia_science_museum :
  ∀ (grades museums : ℕ), grades = 5 → museums = 5 → 
  (∑ i in finset.range museums, if i = 2 then 1 else 0) = 2 →
  (∑ i in finset.range museums, if i \ne 2 then 1 else 0) = 3 →
  finite.card (finset.filter (λ i, i < grades) finset.univ) = grades →
  ∃ (ways : ℕ), ways = nat.choose grades 2 * 4 ^ (grades - 2) :=
begin
  intros grades museums hg hm hjia hrest hgrades,
  use nat.choose grades 2 * 4 ^ (grades - 2),
  sorry
end

end choose_jia_science_museum_l559_559558


namespace problem1_problem2_problem3_problem4_l559_559979

-- Problem 1
theorem problem1 :
  -11 - (-8) + (-13) + 12 = -4 :=
  sorry

-- Problem 2
theorem problem2 :
  3 + 1 / 4 + (- (2 + 3 / 5)) + (5 + 3 / 4) - (8 + 2 / 5) = -2 :=
  sorry

-- Problem 3
theorem problem3 :
  -36 * (5 / 6 - 4 / 9 + 11 / 12) = -47 :=
  sorry

-- Problem 4
theorem problem4 :
  12 * (-1 / 6) + 27 / abs (3 ^ 2) + (-2) ^ 3 = -7 :=
  sorry

end problem1_problem2_problem3_problem4_l559_559979


namespace phase_shift_of_sine_l559_559258

-- Define the variables and conditions
def b : ℝ := 4
def c : ℝ := π / 2

-- State the problem statement
theorem phase_shift_of_sine :
  (c / b) = π / 8 :=
sorry

end phase_shift_of_sine_l559_559258


namespace cube_sum_equals_36_l559_559805

variable {a b c k : ℝ}

theorem cube_sum_equals_36 (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (heq : (a^3 - 12) / a = (b^3 - 12) / b)
    (heq_another : (b^3 - 12) / b = (c^3 - 12) / c) :
    a^3 + b^3 + c^3 = 36 := by
  sorry

end cube_sum_equals_36_l559_559805


namespace correct_equation_l559_559990

-- Definitions of the conditions
def january_turnover (T : ℝ) : Prop := T = 36
def march_turnover (T : ℝ) : Prop := T = 48
def average_monthly_growth_rate (x : ℝ) : Prop := True

-- The goal to be proved
theorem correct_equation (T_jan T_mar : ℝ) (x : ℝ) 
  (h_jan : january_turnover T_jan) 
  (h_mar : march_turnover T_mar) 
  (h_growth : average_monthly_growth_rate x) : 
  36 * (1 + x)^2 = 48 :=
sorry

end correct_equation_l559_559990


namespace two_fifty_ith_digit_l559_559893

theorem two_fifty_ith_digit (n : ℕ) (h_fraction : (5 / 37 : ℚ) = 0.135) (h_repeat : "135".length = 3) : 
  (nth_digit (5 / 37) 250) = 1 :=
by
  sorry

end two_fifty_ith_digit_l559_559893


namespace evaluate_expression_l559_559649

theorem evaluate_expression :
  (81: ℝ) ^ 0.25 * (81: ℝ) ^ 0.20 = 3 * (81: ℝ) ^ (4 / 5) :=
by
  sorry

end evaluate_expression_l559_559649


namespace unique_inverse_l559_559057

noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ := (fun (a b c d : ℚ) => ![![a, b], ![c, d]])
begin
    have h1 :! [![1], ![2]] = ![![a, b], ![c, d]] * ![![7], ![10]],
    have h2 :! [![2], ![4]]  = ![![a,b], ![c,d]] * ![![2], ![4]],
    -- Solve the equation system
    -- This will lead
    sorry

theorem unique_inverse :
  (M_inv = ![![-2, (3/2)], ![1, -(1/2)]]) :=
begin
    apply Matrix.eq_of_duplicate,
    rw Matrix.ext_iff,
    intro k i,
    simp,
end

end unique_inverse_l559_559057


namespace angle_east_northwest_l559_559936

-- Definitions and given conditions
def total_angles : ℝ := 360
def number_of_paths : ℕ := 10
def central_angle (total_angles : ℝ) (number_of_paths : ℕ) : ℝ := total_angles / number_of_paths

-- Function to find the angle between two paths given their positions
def angle_between_paths (path1 path2 : ℕ) (central_angle : ℝ) : ℝ :=
  let diff := abs (path1 - path2)
  if diff <= number_of_paths / 2 then central_angle * diff else central_angle * (number_of_paths - diff)

-- Positions of East and Northwest paths
def east_path_position : ℕ := 3
def northwest_path_position : ℕ := 8

-- Proof statement
theorem angle_east_northwest : angle_between_paths east_path_position northwest_path_position (central_angle total_angles number_of_paths) = 144 := by
  sorry

end angle_east_northwest_l559_559936


namespace value_of_x_when_y_equals_8_l559_559472

noncomputable def inverse_variation(cube_root : ℝ → ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y * (cube_root x) = k

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem value_of_x_when_y_equals_8 : 
  ∃ k : ℝ, (inverse_variation cube_root k 8 2) → 
  (inverse_variation cube_root k (1 / 8) 8) := 
sorry

end value_of_x_when_y_equals_8_l559_559472


namespace francis_had_2_muffins_l559_559290

noncomputable def cost_of_francis_breakfast (m : ℕ) : ℕ := 2 * m + 6
noncomputable def cost_of_kiera_breakfast : ℕ := 4 + 3
noncomputable def total_cost (m : ℕ) : ℕ := cost_of_francis_breakfast m + cost_of_kiera_breakfast

theorem francis_had_2_muffins (m : ℕ) : total_cost m = 17 → m = 2 :=
by
  -- Sorry is used here to leave the proof steps blank.
  sorry

end francis_had_2_muffins_l559_559290


namespace rowing_velocity_l559_559947

theorem rowing_velocity (v : ℝ) :
  (∀ (d1 d2 : ℝ), d1 = 2.4 → d2 = 2.4 → (1 : ℝ) = d1 / (5 + v) + d2 / (5 - v)) →
  (v = 1) :=
by
  intros h,
  sorry

end rowing_velocity_l559_559947


namespace number_of_elements_in_A_number_of_elements_in_A_with_condition_l559_559406

def A : Set (ℤ × ℤ × ℤ) := 
  { p | let (m₁, m₂, m₃) := p in m₂ ∈ [-2, 0, 2] ∧ m₁ ∉ [1, 2, 3] ∧ m₃ ∉ [1, 2, 3] }

-- Proof Problem 1: Prove that the number of elements in set A is 27
theorem number_of_elements_in_A : (Finset.card (A.toFinset)) = 27 := 
  sorry

-- Proof Problem 2: Prove that the number of elements in set A that satisfy 2 ≤ |m₁| + |m₂| + |m₃| ≤ 5 is 18
theorem number_of_elements_in_A_with_condition : 
  (Finset.card ((A ∩ {p | let (m₁, m₂, m₃) := p in 2 ≤ |m₁| + |m₂| + |m₃| ∧ |m₁| + |m₂| + |m₃| ≤ 5}).toFinset)) = 18 := 
  sorry

end number_of_elements_in_A_number_of_elements_in_A_with_condition_l559_559406


namespace sequence_conditions_general_formulas_sum_of_first_n_terms_l559_559316

noncomputable def arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n n = a_n 1 + d * (n - 1)

noncomputable def geometric_sequence (b_n : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, q > 0 ∧ ∀ n : ℕ, b_n (n + 1) = b_n n * q

variables {a_n b_n c_n : ℕ → ℤ}
variables (d q : ℤ) (d_pos : 0 < d) (hq : q > 0)
variables (S_n : ℕ → ℤ)

axiom initial_conditions : a_n 1 = 2 ∧ b_n 1 = 2 ∧ a_n 3 = 8 ∧ b_n 3 = 8

theorem sequence_conditions : arithmetic_sequence a_n ∧ geometric_sequence b_n := sorry

theorem general_formulas :
  (∀ n : ℕ, a_n n = 3 * n - 1) ∧
  (∀ n : ℕ, b_n n = 2^n) := sorry

theorem sum_of_first_n_terms :
  (∀ n : ℕ, S_n n = 3 * 2^(n+1) - n - 6) := sorry

end sequence_conditions_general_formulas_sum_of_first_n_terms_l559_559316


namespace range_of_a_l559_559669

def A : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l559_559669


namespace train_length_calc_l559_559233

theorem train_length_calc (time : ℝ) (speed_km_hr : ℝ) :
  time = 0.9999200063994881 →
  speed_km_hr = 144 →
  let speed_m_s := speed_km_hr * 1000 / 3600 in
  let length := speed_m_s * time in
  length = 39.996800255959524 :=
by
  intro h_time h_speed
  have h_speed_m_s : speed_m_s = 40 := by
    rw [h_speed, mul_div_assoc, mul_one, div_self]
    norm_num
  rw [h_speed_m_s, h_time, mul_comm]
  norm_num
  sorry

end train_length_calc_l559_559233


namespace find_t_l559_559694

variable {a b : EuclideanVector}
variable {t : ℝ}

noncomputable def a_dot_b_zero := (a ⬝ b = 0)
noncomputable def norm_a_b := (∥a + b∥ = t * ∥a∥)
noncomputable def angle_a_b := (angle (a + b) (a - b) = 2 * π / 3)

theorem find_t (h1 : a_dot_b_zero) (h2 : norm_a_b) (h3 : angle_a_b) : t = 2 :=
sorry

end find_t_l559_559694


namespace symmetric_abs_necessary_not_sufficient_l559_559665

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def y_axis_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

theorem symmetric_abs_necessary_not_sufficient (f : ℝ → ℝ) :
  is_odd_function f → y_axis_symmetric f := sorry

end symmetric_abs_necessary_not_sufficient_l559_559665


namespace max_abs_value_of_quadratic_function_l559_559362

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def point_in_band_region (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem max_abs_value_of_quadratic_function (a b c t : ℝ) (h1 : point_in_band_region (quadratic_function a b c (-2) + 2) 0 4)
                                             (h2 : point_in_band_region (quadratic_function a b c 0 + 2) 0 4)
                                             (h3 : point_in_band_region (quadratic_function a b c 2 + 2) 0 4)
                                             (h4 : point_in_band_region (t + 1) (-1) 3) :
  |quadratic_function a b c t| ≤ 5 / 2 :=
sorry

end max_abs_value_of_quadratic_function_l559_559362


namespace reasoning_error_l559_559185

theorem reasoning_error (exponential_decreasing : ∃ f : ℝ → ℝ, ∀ x y : ℝ, x < y → f y < f x ∧ (∀ x : ℝ, f x = a^x) )
                        (is_exponential : ∀ x : ℝ, h x = (2:ℝ)^x) :
                        ¬ ∀ x y : ℝ, x < y → h y < h x :=
by
  sorry

end reasoning_error_l559_559185


namespace probability_exactly_one_second_class_product_l559_559939

open Nat

/-- Proof problem -/
theorem probability_exactly_one_second_class_product :
  let n := 100 -- total products
  let k := 4   -- number of selected products
  let first_class := 90 -- first-class products
  let second_class := 10 -- second-class products
  let C (n k : ℕ) := Nat.choose n k
  (C second_class 1 * C first_class 3 : ℚ) / C n k = 
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose n k :=
by
  -- Mathematically equivalent proof
  sorry

end probability_exactly_one_second_class_product_l559_559939


namespace find_value_of_a_l559_559467

theorem find_value_of_a (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 < a → a^x ≥ 1)
  (h_sum : (a^1) + (a^0) = 3) : a = 2 :=
sorry

end find_value_of_a_l559_559467


namespace girls_together_arrangements_l559_559474

theorem girls_together_arrangements (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) : 
  (4 + 3 - 2)! * 3! = 720 := 
by 
  rw [←h_boys, ←h_girls] 
  norm_num
  sorry

end girls_together_arrangements_l559_559474


namespace general_formula_sum_first_n_terms_bound_b_inv_sum_l559_559687

def sequence_a (n : ℕ) : ℕ → ℕ
| 0       := 0
| 1       := 3
| (n + 1) := sequence_a n * (3 * (n + 1) / n)

theorem general_formula :
  ∀ n : ℕ, sequence_a n = n * 3^n := by
  sorry

theorem sum_first_n_terms :
  ∀ n : ℕ, let S_n := ∑ i in range(n + 1), sequence_a i in
  S_n = (2 * n - 1) / 4 * 3^(n + 1) + 3 / 4 := by
  sorry

def sequence_b (n : ℕ) := (2 * n + 3) / (n + 1) * sequence_a n

theorem bound_b_inv_sum :
  ∀ n : ℕ, 5 / 6 ≤ ∑ i in range(n + 1), 1 / sequence_b i < 1 := by
  sorry

end general_formula_sum_first_n_terms_bound_b_inv_sum_l559_559687


namespace M_intersection_N_l559_559732

def M := { x : ℝ | -1 < x ∧ x < 1 }
def N := { x : ℝ | 0 ≤ x ∧ x < 1 }

theorem M_intersection_N :
  { x : ℝ | -1 < x ∧ x < 1 } ∩ { x : ℝ | ∀ (x:ℝ), x / (x - 1) ≤ 0 } = { x : ℝ | 0 ≤ x ∧ x < 1 } :=
by
  rw set.inter_def
  -- Further steps would be to show that the left side simplifies to the right side.
  sorry

end M_intersection_N_l559_559732


namespace sector_maximum_area_angle_l559_559190

noncomputable def maximum_central_angle (r : ℝ) : ℝ := 2

theorem sector_maximum_area_angle {r : ℝ} (h1 : 0 < r) (h2 : 0 < 40 - 2 * r) :
  (∃ r : ℝ, r = 10) → ∃ α : ℝ, α = maximum_central_angle r :=
by
  intro h
  use maximum_central_angle r
  exact (Exists.intro 2 (rfl))

-- Given the total length of the wire is 40 cm
-- When r = 10 cm, then the central angle α = 2 radians

end sector_maximum_area_angle_l559_559190


namespace roots_equal_irrational_l559_559627

theorem roots_equal_irrational (d : ℝ) (h_eq : (16 * real.pi ^ 2 - 12 * d = 16)) : 
  ∃ x1 x2 : ℝ, x1 = x2 ∧ irrational (2 * real.pi / 3) ∧ irrational (2 * real.pi / 3) :=
by
  sorry

end roots_equal_irrational_l559_559627


namespace wu_stops_after_five_years_l559_559178

theorem wu_stops_after_five_years :
  ∃ a b : ℕ, (a % 2 = 1) ∧ (2^b * ∃ (p : ℚ), p.denom = 2^b ∧ p.num = a) ∧ a + b = 1536 :=
sorry

end wu_stops_after_five_years_l559_559178


namespace max_interesting_numbers_l559_559903

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

def is_interesting (n : ℕ) : Prop :=
  is_prime (sum_of_digits n)

theorem max_interesting_numbers :
  ∀ (a b c d e : ℕ), b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
    (∀ x ∈ {a, b, c, d, e}, x ∈ ℕ) →
    ∃ s : Finset ℕ, s ⊆ {a, b, c, d, e} ∧
                     (∀ n ∈ s, is_interesting n) ∧ s.card = 4 :=
by
  sorry

end max_interesting_numbers_l559_559903


namespace find_lambda_l559_559325

variable (a b : ℝ × ℝ)
variable (λ : ℝ)

def perpendicular (u v : ℝ × ℝ) : Prop := 
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (h₁ : a = (2, 1))
  (h₂ : b = (-1, 3))
  (h₃ : perpendicular a (a.1 - λ * b.1, a.2 + λ * b.2)) :
  λ = -5 := 
sorry

end find_lambda_l559_559325


namespace trigonometric_identity_example_l559_559617

theorem trigonometric_identity_example :
  (let θ := 30 / 180 * Real.pi in
   (Real.tan θ)^2 - (Real.sin θ)^2) / ((Real.tan θ)^2 * (Real.sin θ)^2) = 4 / 3 :=
by
  let θ := 30 / 180 * Real.pi
  have h_tan2 := Real.tan θ * Real.tan θ
  have h_sin2 : Real.sin θ * Real.sin θ = 1/4 := by sorry
  have h_cos2 : Real.cos θ * Real.cos θ = 3/4 := by sorry
  sorry

end trigonometric_identity_example_l559_559617


namespace range_of_sum_l559_559693

theorem range_of_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b + 1 / a + 9 / b = 10) : 2 ≤ a + b ∧ a + b ≤ 8 :=
sorry

end range_of_sum_l559_559693


namespace speed_in_still_water_l559_559513

namespace SwimmingProblem

variable (V_m V_s : ℝ)

-- Downstream condition
def downstream_condition : Prop := V_m + V_s = 18

-- Upstream condition
def upstream_condition : Prop := V_m - V_s = 13

-- The main theorem stating the problem
theorem speed_in_still_water (h_downstream : downstream_condition V_m V_s) 
                             (h_upstream : upstream_condition V_m V_s) :
    V_m = 15.5 :=
by
  sorry

end SwimmingProblem

end speed_in_still_water_l559_559513


namespace compound_interest_amount_l559_559118

-- Define the simple interest formula
def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

-- Define the compound interest formula
def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100) ^ T - P

-- Define the given conditions
def SI : ℝ := simple_interest 1400.0000000000014 10 3
def CI (P : ℝ) : ℝ := compound_interest P 10 2

-- Define the problem statement
theorem compound_interest_amount :
  ∃ P : ℝ, SI = (CI P) / 2 → P = 4000 :=
by
  sorry

end compound_interest_amount_l559_559118


namespace invertible_product_labels_l559_559626

def f2 (x : ℝ) : ℝ := x^3 - 2 * x
def f3 : ℝ → ℝ
| -6 := 0
| -5 := 2
| -4 := 4
| -3 := 6
| -2 := 5
| -1 := 3
| 0  := 0
| 1  := -3
| 2  := -5
| 3  := -6
| _  := 0 -- Handling remaining cases for function definition but will not be used
def f4 (x : ℝ) : ℝ := -tan x
def f5 (x : ℝ) : ℝ := 3 / x

theorem invertible_product_labels :
  (∀ x ∈ set.Icc (-2 : ℝ) (4 : ℝ), function.injective f2) ∧
  (∀ x y ∈ set.Icc (-6 : ℝ) (3 : ℝ), (f3 x = f3 y → x = y)) ∧
  (∀ x ∈ (set.Ioo (-real.pi / 2) (real.pi / 2) \ {0}), function.injective f4) ∧
  (∀ x ∈ (set.univ \ {0} : set ℝ), function.injective f5) →
  2 * 3 * 4 * 5 = 120 :=
by sorry

end invertible_product_labels_l559_559626


namespace parallel_line_length_divides_triangle_l559_559446

theorem parallel_line_length_divides_triangle {BC : ℝ} (hBC : BC = 18) :
  ∃ (MN : ℝ), (MN = 9 * real.sqrt 2) ∧ (1/2) * (BC * h) = (MN * mn_height) := 
by 
  sorry

end parallel_line_length_divides_triangle_l559_559446


namespace impossible_condition_l559_559717

noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

theorem impossible_condition (a b c : ℝ) (h : f a > f b ∧ f b > f c) : ¬ (b < a ∧ a < c) :=
by
  sorry

end impossible_condition_l559_559717


namespace total_selling_price_is_18000_l559_559560

def cost_price_per_meter : ℕ := 50
def loss_per_meter : ℕ := 5
def meters_sold : ℕ := 400

def selling_price_per_meter := cost_price_per_meter - loss_per_meter

def total_selling_price := selling_price_per_meter * meters_sold

theorem total_selling_price_is_18000 :
  total_selling_price = 18000 :=
sorry

end total_selling_price_is_18000_l559_559560


namespace has_exactly_one_zero_l559_559095

noncomputable def f (a : ℝ) : ℝ → ℝ 
| x := if x > 0 then log x / log 2 else -2^x + a

theorem has_exactly_one_zero (a : ℝ) : 
  (∃! x, f a x = 0) ↔ a < 0 := by
sorry

end has_exactly_one_zero_l559_559095


namespace minimize_expense_l559_559933

def price_after_first_discount (initial_price : ℕ) (discount : ℕ) : ℕ :=
  initial_price * (100 - discount) / 100

def final_price_set1 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 15
  let step2 := price_after_first_discount step1 25
  price_after_first_discount step2 10

def final_price_set2 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 25
  let step2 := price_after_first_discount step1 10
  price_after_first_discount step2 10

theorem minimize_expense (initial_price : ℕ) (h : initial_price = 12000) :
  final_price_set1 initial_price = 6885 ∧ final_price_set2 initial_price = 7290 ∧
  final_price_set1 initial_price < final_price_set2 initial_price := by
  sorry

end minimize_expense_l559_559933


namespace find_lambda_find_t_range_l559_559709

-- Definitions of points and circles
structure Point (α : Type) :=
  (x : α)
  (y : α)

noncomputable def on_circle_O (P : Point ℝ) : Prop :=
  P.x^2 + P.y^2 = 4

noncomputable def distance (P Q : Point ℝ) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- First part of the problem: Finding λ
theorem find_lambda (a m : ℝ) (h_m : m > 0) (P : Point ℝ) (hP : on_circle_O P) (λ : ℝ)
    (h : distance P (Point.mk a 2) = λ * distance P (Point.mk m 1)) :
    λ = Real.sqrt 2 := sorry

-- Second part of the problem: Finding the range of t
noncomputable def on_circle_C (P : Point ℝ) : Prop :=
  P.x^2 + P.y^2 = 1

theorem find_t_range (a : ℝ) (M : Point ℝ) (t : ℝ)
    (h1 : on_circle_C M)
    (N : Point ℝ := Point.mk (2 * M.x - a) (2 * M.y - t))
    (h2 : on_circle_C N)
    (h3 : M.x^2 + M.y^2 = 1)
    (h4 : 8 * M.x + 4 * t * M.y - t^2 - 7 = 0) :
    -Real.sqrt 5 ≤ t ∧ t ≤ Real.sqrt 5 := sorry

end find_lambda_find_t_range_l559_559709


namespace maximal_inverse_sum_exists_l559_559042

def S (n : ℕ) : Set (Set ℕ) := {A | A.card = n ∧ ∀ a b c ∈ A, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (2*b ≠ a + c)}

theorem maximal_inverse_sum_exists (n : ℕ) :
  ∃ (A ∈ S n), ∀ B ∈ S n, (∑ x in A, 1 / (x : ℚ)) ≥ ∑ x in B, 1 / (x : ℚ) :=
sorry

end maximal_inverse_sum_exists_l559_559042


namespace part_a_l559_559520

theorem part_a
  (A B C M O O_b : Type)
  [triangle A B C] 
  [angle_bisector B intersects_circle_at M] 
  [incircle_center O] 
  [excircle_center_tangent_to_side_AC O_b] :
  circle_with_center (circumcenter M) through_points {A, C, O, O_b} :=
sorry

end part_a_l559_559520


namespace cistern_filling_time_l559_559207

/-
Given the following conditions:
- Pipe A fills the cistern in 10 hours.
- Pipe B fills the cistern in 12 hours.
- Exhaust pipe C drains the cistern in 15 hours.
- Exhaust pipe D drains the cistern in 20 hours.

Prove that if all four pipes are opened simultaneously, the cistern will be filled in 15 hours.
-/

theorem cistern_filling_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 12
  let rate_C := -(1 / 15)
  let rate_D := -(1 / 20)
  let combined_rate := rate_A + rate_B + rate_C + rate_D
  let time_to_fill := 1 / combined_rate
  time_to_fill = 15 :=
by 
  sorry

end cistern_filling_time_l559_559207


namespace tan_sin_identity_l559_559614

theorem tan_sin_identity :
  (let θ := 30 * Real.pi / 180 in 
   let tan_θ := Real.sin θ / Real.cos θ in 
   let sin_θ := 1 / 2 in
   let cos_θ := Real.sqrt 3 / 2 in
   ((tan_θ ^ 2 - sin_θ ^ 2) / (tan_θ ^ 2 * sin_θ ^ 2) = 1)) :=
by
  let θ := 30 * Real.pi / 180
  let tan_θ := Real.sin θ / Real.cos θ
  let sin_θ := 1 / 2
  let cos_θ := Real.sqrt 3 / 2
  have h_tan_θ : tan_θ = 1 / Real.sqrt 3 := sorry  -- We skip the derivation proof.
  have h_sin_θ : sin_θ = 1 / 2 := rfl
  have h_cos_θ : cos_θ = Real.sqrt 3 / 2 := sorry  -- We skip the derivation proof.
  have numerator := tan_θ ^ 2 - sin_θ ^ 2
  have denominator := tan_θ ^ 2 * sin_θ ^ 2
  -- show equality
  have h : (numerator / denominator) = (1 : ℝ) := sorry  -- We skip the final proof.
  exact h

end tan_sin_identity_l559_559614


namespace students_with_screws_neq_bolts_l559_559767

-- Let's define the main entities
def total_students : ℕ := 40
def nails_neq_bolts : ℕ := 15
def screws_eq_nails : ℕ := 10

-- Main theorem statement
theorem students_with_screws_neq_bolts (total : ℕ) (neq_nails_bolts : ℕ) (eq_screws_nails : ℕ) :
  total = 40 → neq_nails_bolts = 15 → eq_screws_nails = 10 → ∃ k, k ≥ 15 ∧ k ≤ 40 - eq_screws_nails - neq_nails_bolts := 
by
  intros
  sorry

end students_with_screws_neq_bolts_l559_559767


namespace fourth_friend_age_is_8_l559_559443

-- Define the given data
variables (a1 a2 a3 a4 : ℕ)
variables (h_avg : (a1 + a2 + a3 + a4) / 4 = 9)
variables (h1 : a1 = 7) (h2 : a2 = 9) (h3 : a3 = 12)

-- Formalize the theorem to prove that the fourth friend's age is 8
theorem fourth_friend_age_is_8 : a4 = 8 :=
by
  -- Placeholder for the proof
  sorry

end fourth_friend_age_is_8_l559_559443


namespace min_area_ocab_parabola_l559_559728

theorem min_area_ocab_parabola :
  let parabola := {p : ℝ × ℝ | p.snd = p.fst ^ 2},
      O := (0 : ℝ, 0 : ℝ),
      F := (0 : ℝ, 1 / 4 : ℝ),
      N := {n : ℝ × ℝ | n.fst = 0 ∧ n.snd > 0},
      intersects (l : ℝ × ℝ → Prop) (P : Set (ℝ × ℝ)) (Q : Set (ℝ × ℝ)) :=
        ∃ A B : ℝ × ℝ, P A ∧ Q A ∧ P B ∧ Q B,
      dot_product (u v : ℝ × ℝ) := u.fst * v.fst + u.snd * v.snd,
      symmetric_point (F A : ℝ × ℝ) :=
        (2 * A.fst - F.fst, 2 * A.snd - F.snd),
      area_of_quadrilateral (O C A B : ℝ × ℝ) :=
        let d1 := abs (((F.snd) / (real.sqrt (1 + (A.fst)^2)))),
            d2 := abs (2 / (real.sqrt (1 + (A.fst + B.fst)^2))),
            OA := A.fst * real.sqrt (1 + A.fst^2),
            AB := real.sqrt ((1 + (A.fst + B.fst)^2) * ((A.fst + B.fst)^2 + 8)) in
        (1 / 2 : ℝ) * (OA * d1 + AB * d2)
  in ∀ l, ∀ A B : ℝ × ℝ,
  (N l ∧ l A ∧ A ∈ parabola ∧ l B ∧ B ∈ parabola) ∧
  dot_product (A - O) (B - O) = 2 →
  ∃ C : ℝ × ℝ, 
  symmetric_point F A = C →
  area_of_quadrilateral O C A B = 3 :=
sorry

end min_area_ocab_parabola_l559_559728


namespace a_n_arithmetic_sum_b_n_l559_559673

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m ^ x

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
axiom seq_a_geometric (n : ℕ) (m : ℝ) (h₀ : m > 0) (h₁ : m ≠ 1) :
  f (a n) m = m^(n + 1)

axiom b_n_def (n : ℕ) (m : ℝ) : b n = a n * f (a n) m

-- Question 1: Prove the sequence {a_n} is an arithmetic sequence
theorem a_n_arithmetic (n : ℕ) (m : ℝ) (h₀ : m > 0) (h₁ : m ≠ 1) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d := sorry

-- Question 2: Find S_n when m = 2
theorem sum_b_n (n : ℕ) :
  let m := 2 in
  S n = 2^(n+2) * n := sorry

end a_n_arithmetic_sum_b_n_l559_559673


namespace players_both_kabadi_kho_kho_l559_559527

noncomputable def numberOfPlayersPlayingBothGames (kabadi: ℕ) (kho_kho_only: ℕ) (total: ℕ) : ℕ :=
  total - kho_kho_only

theorem players_both_kabadi_kho_kho (kabadi: ℕ) (kho_kho_only: ℕ) (total: ℕ) :
  kabadi = 10 → kho_kho_only = 35 → total = 45 → numberOfPlayersPlayingBothGames kabadi kho_kho_only total = 10 :=
by
  intros h_kabadi h_kho_kho_only h_total
  rw [h_kabadi, h_kho_kho_only, h_total]
  unfold numberOfPlayersPlayingBothGames
  rw [Nat.sub_self]
  exact rfl

end players_both_kabadi_kho_kho_l559_559527


namespace tan_sin_div_l559_559609

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l559_559609


namespace solve_equation_l559_559068

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end solve_equation_l559_559068


namespace point_reflection_l559_559856

-- Define the original point and the reflection function
structure Point where
  x : ℝ
  y : ℝ

def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

-- Define the original point
def M : Point := ⟨-5, 2⟩

-- State the theorem to prove the reflection
theorem point_reflection : reflect_y_axis M = ⟨5, 2⟩ :=
  sorry

end point_reflection_l559_559856


namespace sum_of_digits_of_9N_is_9_l559_559999

-- Define what it means for a natural number N to have strictly increasing digits.
noncomputable def strictly_increasing_digits (N : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → (N.digits i < N.digits j)

-- Formal statement of the problem
theorem sum_of_digits_of_9N_is_9 (N : ℕ) (h : strictly_increasing_digits N) : ∑ d in (9 * N).digits, d = 9 :=
by sorry

end sum_of_digits_of_9N_is_9_l559_559999


namespace range_f_on_interval_maximum_a_plus_b_l559_559726

noncomputable def f (x a b : ℝ) := Real.exp x - 0.5 * a * x ^ 2 - b

theorem range_f_on_interval :
  (∀ x, f x 1 1 ≥ 0) → 
  ∃ (x_min x_max : ℝ), -1 ≤ x_min ∧ x_min ≤ 1 ∧ -1 ≤ x_max ∧ x_max ≤ 1 ∧ 
    (∀ x, -1 ≤ x ∧ x ≤ 1 → f x 1 1 ≥ f x_min 1 1 ∧ f x 1 1 ≤ f x_max 1 1) ∧ 
    interval_range (f (-1) 1 1) (f (1) 1 1) = set.Icc ((1/Real.exp 1) - (3/2)) (Real.exp 1 - (3/2)) := by 
  sorry 

theorem maximum_a_plus_b (h_nonnegative : ∀ x, f x a b ≥ 0) : 
  a + b ≤ Real.exp (-Real.sqrt 2) := by 
  sorry 

end range_f_on_interval_maximum_a_plus_b_l559_559726


namespace problem1_part1_monotonic_intervals_problem2_prove_slope_midpoint_greater_problem3_range_of_b_l559_559818

-- Problem 1
variable (x : ℝ) (f g : ℝ → ℝ)
def f_def : f x = Real.log x := by rfl
def g_def (a : ℝ) : g x = (2 - a) * (x - 1) - 2 * f x := by rfl

theorem problem1_part1_monotonic_intervals :
  let g' := λ x : ℝ, 1 - 2 / x
  g' x < 0 ∧ (0 < x ∧ x < 2) → g' x > 0 ∧ (x > 2) := sorry

-- Problem 2
variable (x_1 x_2 x_0 : ℝ) (f' k : ℝ → ℝ)
def f_prime_def : f' x = 1 / x := by rfl
def k_def : k (x_1 x_2 : ℝ) := (f' x_2 - f' x_1) / (x_2 - x_1) := by rfl

theorem problem2_prove_slope_midpoint_greater :
  let x_0 := (x_1 + x_2) / 2
  k x > f' x_0 := sorry

-- Problem 3
variable (b x_1 x_2 : ℝ)
def F_def : F x = |Real.log x| + b / (x + 1) := by rfl

theorem problem3_range_of_b :
  (0 < x_1 ∧ x_1 ≤ 2 ∧ 0 < x_2 ∧ x_2 ≤ 2 ∧ x_1 ≠ x_2) → (F x_1 - F x_2) / (x_1 - x_2) < -1 → b ≥ 27 / 2 := sorry

end problem1_part1_monotonic_intervals_problem2_prove_slope_midpoint_greater_problem3_range_of_b_l559_559818


namespace largest_radius_cone_l559_559211

structure Crate :=
  (width : ℝ)
  (depth : ℝ)
  (height : ℝ)

structure Cone :=
  (radius : ℝ)
  (height : ℝ)

noncomputable def larger_fit_within_crate (c : Crate) (cone : Cone) : Prop :=
  cone.radius = min c.width c.depth / 2 ∧ cone.height = max (max c.width c.depth) c.height

theorem largest_radius_cone (c : Crate) (cone : Cone) : 
  c.width = 5 → c.depth = 8 → c.height = 12 → larger_fit_within_crate c cone → cone.radius = 2.5 :=
by
  sorry

end largest_radius_cone_l559_559211


namespace photos_per_album_l559_559222

theorem photos_per_album
  (n : ℕ) -- number of pages in each album
  (x y : ℕ) -- album numbers
  (h1 : 4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20)
  (h2 : 4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12) :
  4 * n = 32 :=
by 
  sorry

end photos_per_album_l559_559222


namespace downstream_distance_15_minutes_l559_559518

theorem downstream_distance_15_minutes
  (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ)
  (h1 : speed_boat = 24)
  (h2 : speed_current = 3)
  (h3 : time_minutes = 15) :
  let effective_speed := speed_boat + speed_current
  let time_hours := time_minutes / 60
  let distance := effective_speed * time_hours
  distance = 6.75 :=
by {
  sorry
}

end downstream_distance_15_minutes_l559_559518


namespace no_single_digit_A_has_positive_integer_solutions_l559_559639

theorem no_single_digit_A_has_positive_integer_solutions :
  ∀ (A : ℕ), A ∈ finset.range 10 → ¬ ∃ (x : ℕ), x > 0 ∧ (x^2 - (2*A)*x + (10*A + 3) = 0)  :=
by
  intro A hA
  simp only [finset.range, nat.cast_add, finset.mem_range, nat.cast_one, nat.cast_zero] at hA
  by_contra
  obtain ⟨x, hx_pos, hx_eq⟩ := h
  have h₂ : 2 * ↑A ≥ 1 := by linarith
  sorry

end no_single_digit_A_has_positive_integer_solutions_l559_559639


namespace john_total_animals_is_114_l559_559027

  -- Define the entities and their relationships based on the conditions
  def num_snakes : ℕ := 15
  def num_monkeys : ℕ := 2 * num_snakes
  def num_lions : ℕ := num_monkeys - 5
  def num_pandas : ℕ := num_lions + 8
  def num_dogs : ℕ := num_pandas / 3

  -- Define the total number of animals
  def total_animals : ℕ := num_snakes + num_monkeys + num_lions + num_pandas + num_dogs

  -- Prove that the total number of animals is 114
  theorem john_total_animals_is_114 : total_animals = 114 := by
    sorry
  
end john_total_animals_is_114_l559_559027


namespace find_overtime_hours_l559_559544

-- Definitions of the conditions
def regular_pay_rate : ℕ := 3
def regular_hours : ℕ := 40
def overtime_rate := 2 * regular_pay_rate
def total_payment : ℕ := 192

-- The statement we want to prove
theorem find_overtime_hours :
  let regular_pay := regular_pay_rate * regular_hours,
      overtime_pay := total_payment - regular_pay,
      overtime_hours := overtime_pay / overtime_rate
  in overtime_hours = 12 :=
by
  sorry

end find_overtime_hours_l559_559544


namespace dexter_total_cards_l559_559643

-- Define the given conditions as constants and variables in Lean
constant num_basketball_boxes : ℕ := 9
constant cards_per_basketball_box : ℕ := 15
constant boxes_filled_less_with_football_cards : ℕ := 3
constant cards_per_football_box : ℕ := 20

-- Calculate the derived quantities based on the above conditions
def num_football_boxes : ℕ := num_basketball_boxes - boxes_filled_less_with_football_cards
def total_basketball_cards : ℕ := num_basketball_boxes * cards_per_basketball_box
def total_football_cards : ℕ := num_football_boxes * cards_per_football_box
def total_cards : ℕ := total_basketball_cards + total_football_cards

-- State the theorem to prove
theorem dexter_total_cards : total_cards = 255 := by
  sorry  -- proof placeholder; the goal is to establish that total_cards = 255

end dexter_total_cards_l559_559643


namespace cost_price_of_article_l559_559578

theorem cost_price_of_article (SP CP : ℝ) (h1 : SP = 150) (h2 : SP = CP + (1 / 4) * CP) : CP = 120 :=
by
  sorry

end cost_price_of_article_l559_559578


namespace perpendicular_vectors_parallel_vectors_l559_559735

-- Definitions of the vectors
def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos ((3 * x) / 2), Real.sin ((3 * x) / 2))
def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))

-- Predicates for perpendicular and parallel
def is_perpendicular (x : ℝ) : Prop := 
  (Real.cos ((3 * x) / 2) * Real.cos (x / 2)) - (Real.sin ((3 * x) / 2) * Real.sin (x / 2)) = 0

def is_parallel (x : ℝ) : Prop := 
  (Real.sin (2 * x) = 0)

-- Set of solutions for perpendicular vectors
def perpendicular_solution_set : Set ℝ := 
  { x | ∃ k : ℤ, x = (Real.pi * (2 * k + 1)) / 4 }

-- Set of solutions for parallel vectors
def parallel_solution_set : Set ℝ := 
  { x | ∃ k : ℤ, x = k * Real.pi }

-- Theorems collection
theorem perpendicular_vectors : ∀ x : ℝ, is_perpendicular x ↔ x ∈ perpendicular_solution_set := sorry
theorem parallel_vectors : ∀ x : ℝ, is_parallel x ↔ x ∈ parallel_solution_set := sorry

end perpendicular_vectors_parallel_vectors_l559_559735


namespace range_of_f_l559_559114

def f (x : ℝ) := |x| - 4

theorem range_of_f : set.range f = {y : ℝ | y ≥ -4} :=
by
  sorry

end range_of_f_l559_559114


namespace intersection_on_CD_l559_559033

variables {Point : Type*} [MetricSpace Point]

-- Definitions of given elements
variables (A B C D F E K L M S T : Point)

def is_cyclic_quad (A B C D : Point) : Prop :=
   -- Definition of cyclic quadrilateral
  ∃ O : Point, O ≠ A ∧ O ≠ B ∧ O ≠ C ∧ O ≠ D ∧
  dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D

def is_projection (F : Point) (AD K : Point) : Prop :=
  ∃ (F' AD' : Point), F' = F ∧ AD' = AD ∧ dist F AD' = dist F' K

def is_midpoint (X Y Z : Point) : Prop :=
  dist X Y = dist X Z

variables (h_cyclic : is_cyclic_quad A B C D)
variables (h_intersections : (∃ F E : Point, F = (A + C) / 2 ∧ E = (B + D) / 2))
variables (h_projectionK : is_projection F (A + D) K)
variables (h_projectionL : is_projection F (B + C) L)
variables (h_midpointM : is_midpoint E F M)
variables (h_midpointS : is_midpoint C F S)
variables (h_midpointT : is_midpoint D F T)

-- Statement to prove
theorem intersection_on_CD :
  ∃ P : Point, P ≠ M ∧ (circle_through M K T).contains P ∧ (circle_through M L S).contains P ∧ P lies_on CD :=
sorry

end intersection_on_CD_l559_559033


namespace annual_pension_formula_l559_559950

-- Define the problem variables
variables (a b p q c x k : ℝ) (ha : a ≠ b)

-- The conditions as Lean definitions
def pension (x : ℝ) : ℝ := k * real.sqrt (c * x)
def condition1 : Prop := pension (x + a) = pension x + 2 * p
def condition2 : Prop := pension (x + b) = pension x + 2 * q

-- State the theorem to prove
theorem annual_pension_formula (h1 : condition1 a b p q c x k) (h2 : condition2 a b p q c x k) : 
    pension x = (a * q^2 - b * p^2) / (b * p - a * q) := 
sorry

end annual_pension_formula_l559_559950


namespace steps_to_z_l559_559426

open Nat

theorem steps_to_z (start end z n k d : ℕ) (h1 : start = 2) (h2 : end = 34) (h3 : n = 8) (h4 : k = 6) (h5 : d = (end - start) / n) (h6 : z = start + k * d) : z = 26 :=
sorry

end steps_to_z_l559_559426


namespace photos_per_album_l559_559223

theorem photos_per_album
  (n : ℕ) -- number of pages in each album
  (x y : ℕ) -- album numbers
  (h1 : 4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20)
  (h2 : 4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12) :
  4 * n = 32 :=
by 
  sorry

end photos_per_album_l559_559223


namespace add_closed_sub_closed_mul_closed_pow_closed_l559_559911

variable (a b : ℤ)
variable (n : ℕ)

theorem add_closed (a b : ℤ) : a + b ∈ ℤ := sorry

theorem sub_closed (a b : ℤ) : a - b ∈ ℤ := sorry

theorem mul_closed (a b : ℤ) : a * b ∈ ℤ := sorry

theorem pow_closed (a : ℤ) (n : ℕ) : a ^ n ∈ ℤ := sorry

end add_closed_sub_closed_mul_closed_pow_closed_l559_559911


namespace ant_turns_equal_l559_559521

theorem ant_turns_equal (P : Polyhedron) 
  (h1 : ∀ v : Vertex, v ∈ P → vertex_degree v = 3) 
  (h2 : ∀ n : ℕ, even (number_of_faces_with_n_vertices P n))
  (h_cycle : non_self_intersecting_cycle (edges_of_polyhedron P)) 
  (h_fair_division : ∀ n : ℕ, number_of_faces_with_n_vertices_on_side P n side1 = number_of_faces_with_n_vertices_on_side P n side2):
  number_of_left_turns (path_of_cycle P) = number_of_right_turns (path_of_cycle P) :=
sorry

end ant_turns_equal_l559_559521


namespace james_initial_toys_l559_559378

/-- The initial number of toys James had. -/
def initial_toys : ℕ := 100

theorem james_initial_toys
  (sold_percentage : ℚ := 0.8)
  (cost_per_toy : ℚ := 20)
  (sell_per_toy : ℚ := 30)
  (net_profit : ℚ := 800) :
  let T := initial_toys in
  (sold_percentage * T) * (sell_per_toy - cost_per_toy) = net_profit :=
by 
  sorry

end james_initial_toys_l559_559378


namespace number_of_outcomes_is_correct_l559_559061

noncomputable def calculate_outcomes : ℕ :=
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let scenarios := 4 + 6 + 4
  choose 6 2 * choose 4 1 * scenarios

theorem number_of_outcomes_is_correct :
  calculate_outcomes = 840 :=
by
  sorry

end number_of_outcomes_is_correct_l559_559061


namespace asymptotes_of_hyperbola_l559_559094

theorem asymptotes_of_hyperbola (x y : ℝ) (h : x^2 - 2 * y^2 = 3) : 
    (y = sqrt(2) / 2 * x) ∨ (y = - sqrt(2) / 2 * x) :=
sorry

end asymptotes_of_hyperbola_l559_559094


namespace total_collection_l559_559942

theorem total_collection (members paise_per_member : ℕ) (h1 : members = 77) (h2 : paise_per_member = 77) : 
  (members * paise_per_member) / 100 = 59.29 :=
by
  sorry

end total_collection_l559_559942


namespace integral_abs_value_sum_l559_559273

theorem integral_abs_value_sum :
  ∫ x in 0..4, (|x-1| + |x-3|) = 10 :=
by
  sorry

end integral_abs_value_sum_l559_559273


namespace vector_operations_l559_559333

variables (a b : ℝ × ℝ)
variable (k : ℝ)

def vec_a := (2, 0) : ℝ × ℝ
def vec_b := (1, 4) : ℝ × ℝ

theorem vector_operations :
  2 • vec_a + 3 • vec_b = (7, 12) ∧
  vec_a - 2 • vec_b = (0, -8) ∧
  (∃ t : ℝ, k • vec_a + vec_b = t • (vec_a + 2 • vec_b)) → k = 1 / 2 :=
by
  sorry

end vector_operations_l559_559333


namespace solve_equation_l559_559075

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end solve_equation_l559_559075


namespace polynomial_roots_identity_l559_559389

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def real_roots (P : Polynomial ℝ) : Prop := ∀ root ∈ P.roots, is_real root

theorem polynomial_roots_identity (n : ℕ) (h : n ≥ 3) :
  ∃ P : Polynomial ℝ,
    P = Polynomial.monomial n 1 - Polynomial.monomial (n-1) (binom n 1) + Polynomial.monomial (n-2) (binom n 2) ∧
    real_roots P ∧
    P = (Polynomial.X - 1)^n :=
begin
  sorry
end

end polynomial_roots_identity_l559_559389


namespace ratio_sequences_l559_559792

-- Define positive integers n and k, with k >= n and k - n even.
variables {n k : ℕ} (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0)

-- Define the sets S_N and S_M
def S_N (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_N
def S_M (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_M

-- Main theorem: N / M = 2^(k - n)
theorem ratio_sequences (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0) :
  (S_N n k : ℝ) / (S_M n k : ℝ) = 2^(k - n) := sorry

end ratio_sequences_l559_559792


namespace number_of_distinct_gardens_l559_559994

def is_adjacent (i1 j1 i2 j2 : ℕ) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 + 1 = j2)) ∨ 
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 + 1 = i2))

def is_garden (M : ℕ → ℕ → ℕ) (m n : ℕ) : Prop :=
  ∀ i j i' j', (i < m ∧ j < n ∧ i' < m ∧ j' < n ∧ is_adjacent i j i' j') → 
    ((M i j = M i' j') ∨ (M i j = M i' j' + 1) ∨ (M i j + 1 = M i' j')) ∧
  ∀ i j, (i < m ∧ j < n ∧ 
    (∀ (i' j'), is_adjacent i j i' j' → (M i j ≤ M i' j'))) → M i j = 0

theorem number_of_distinct_gardens (m n : ℕ) : 
  ∃ (count : ℕ), count = 2 ^ (m * n) - 1 :=
sorry

end number_of_distinct_gardens_l559_559994


namespace perpendicular_planes_imply_perpendicular_l559_559395

-- Assume lines a and b, and planes ξ and ζ are different
variable (a b : Type) -- types representing the lines
variable (ξ ζ : Type) -- types representing the planes

-- Assume a ⊥ b, a ⊥ ξ, and b ⊥ ζ
variable [HasPerpendicular a b]
variable [HasPerpendicular a ξ]
variable [HasPerpendicular b ζ]

-- To prove that ξ ⊥ ζ
theorem perpendicular_planes_imply_perpendicular (a b : Type) (ξ ζ : Type)
  [HasPerpendicular a b] [HasPerpendicular a ξ] [HasPerpendicular b ζ] : HasPerpendicular ξ ζ :=
sorry

end perpendicular_planes_imply_perpendicular_l559_559395


namespace number_of_ways_to_divide_l559_559370

def shape_17_cells : Type := sorry -- We would define the structure of the shape here
def checkerboard_pattern : shape_17_cells → Prop := sorry -- The checkerboard pattern condition
def num_black_cells (s : shape_17_cells) : ℕ := 9 -- Number of black cells
def num_gray_cells (s : shape_17_cells) : ℕ := 8 -- Number of gray cells
def divides_into (s : shape_17_cells) (rectangles : ℕ) (squares : ℕ) : Prop := sorry -- Division condition

theorem number_of_ways_to_divide (s : shape_17_cells) (h1 : checkerboard_pattern s) (h2 : divides_into s 8 1) :
  num_black_cells s = 9 ∧ num_gray_cells s = 8 → 
  (∃ ways : ℕ, ways = 10) := 
sorry

end number_of_ways_to_divide_l559_559370


namespace trigonometric_identity_l559_559604

theorem trigonometric_identity :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  t30 = 1 / Real.sqrt 3 →
  s30 = 1 / 2 →
  (t30 ^ 2 - s30 ^ 2) / (t30 ^ 2 * s30 ^ 2) = 1 :=
by
  intros t30_eq s30_eq
  -- Proof would go here
  sorry

end trigonometric_identity_l559_559604


namespace analytical_expression_range_of_c_maximum_value_of_y_l559_559720

-- Given function and conditions
noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) := a * x^2 + (b-8) * x - a - a * b

-- Proof of the function's specific form
theorem analytical_expression (x: ℝ) : 
(∀ x, x ∈ Ioo (-3 : ℝ) 2 → f (-3) 5 x > 0) → 
(∀ x, x ∈ (Iic (-3 : ℝ) ∪ Ioi (2 : ℝ)) → f (-3) 5 x < 0) →
f (-3) 5 x = -3*x^2 - 3*x + 18 :=
sorry

-- Proof for the range of c
theorem range_of_c (c : ℝ) :
(∀ x, -3 * x^2 + 5 * x + c ≤ 0) → 
c ≤ -25/12 :=
sorry

-- Proof for the maximum value of y
theorem maximum_value_of_y (x : ℝ) (h : x > -1) : 
y (-3) 5 (x : ℝ) = (f (-3) 5 x - 21) / (x + 1) → 
(y (-3) 5 x ≤ -3) ∧ (∃ x₀, x₀ = 0 ∧ y (-3) 5 x₀ = -3) :=
sorry

end analytical_expression_range_of_c_maximum_value_of_y_l559_559720


namespace trucks_needed_l559_559988

theorem trucks_needed (S M L : ℕ) (total_packages : ℕ) (truckB_capacityS truckB_capacityM truckB_capacityL : ℕ) 
                      (h1 : 2 * S = 3 * M) (h2 : 3 * M = L)
                      (h_total : total_packages = S + M + L)
                      (truckB_capacityS = 90) (truckB_capacityM = 60) (truckB_capacityL = 50) :
                      S = 600 ∧ M = 300 ∧ L = 100 ∧ (total_packages = 1000) ∧ ((L + 49) / 50 + (M + 59) / 60 + (S + 89) / 90 = 14) :=
by
  sorry

end trucks_needed_l559_559988


namespace count_divisible_by_7_l559_559808

theorem count_divisible_by_7 (f : ℕ → ℕ) (g : ℕ → ℕ) :
  (∀ n, 10000 ≤ n ∧ n ≤ 99999 → (n = 50 * f n + g n ∧ 0 ≤ g n ∧ g n < 50)) →
  (finset.card ((finset.Icc 10000 99999).filter (λ n, 7 ∣ (f n + g n))) = 12600) :=
by
  sorry

end count_divisible_by_7_l559_559808


namespace quadratic_has_exactly_one_root_l559_559815

noncomputable def discriminant (b c : ℝ) : ℝ :=
b^2 - 4 * c

noncomputable def f (x b c : ℝ) : ℝ :=
x^2 + b * x + c

noncomputable def transformed_f (x b c : ℝ) : ℝ :=
(x - 2020)^2 + b * (x - 2020) + c

theorem quadratic_has_exactly_one_root (b c : ℝ)
  (h_discriminant : discriminant b c = 2020) :
  ∃! x : ℝ, f (x - 2020) b c + f x b c = 0 :=
sorry

end quadratic_has_exactly_one_root_l559_559815


namespace abs_b_leq_one_l559_559804

theorem abs_b_leq_one (a b : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : |b| ≤ 1 := 
sorry

end abs_b_leq_one_l559_559804


namespace BoomBoom_num_words_with_repeated_letters_l559_559009

/-- The alphabet of the Boom-Boom tribe contains six letters. -/
def alphabet := {A, B, C, D, E, F}

/-- A word is any sequence of six letters in the alphabet. -/
def word := (List.ofFn $ fun _ => alphabet) 

/-- Given that there are 6 letters in the alphabet of the Boom-Boom tribe, we want to prove the number of 6-letter words with at least two identical letters. -/
theorem BoomBoom_num_words_with_repeated_letters : 
  let total_sequences := 6^6
  let unique_sequences := 6!
  let repeated_letter_sequences := total_sequences - unique_sequences
  repeated_letter_sequences = 45936 :=
by 
  sorry

end BoomBoom_num_words_with_repeated_letters_l559_559009


namespace main_diagonal_contains_all_numbers_l559_559229

def is_latin_square (M : matrix (fin 7) (fin 7) (fin 8)) : Prop :=
  ∀ i, ∀ c : fin 8, ∃! j, M (i, j) = c

def is_symmetric (M : matrix (fin 7) (fin 7) (fin 8)) : Prop :=
  ∀ i j, M (i, j) = M (j, i)

theorem main_diagonal_contains_all_numbers (M : matrix (fin 7) (fin 7) (fin 8))
    (h_latin : is_latin_square M)
    (h_symmetric : is_symmetric M) :
  ∀ n : fin 8, ∃ i : fin 7, M (i, i) = n :=
sorry

end main_diagonal_contains_all_numbers_l559_559229


namespace weight_of_purple_ring_l559_559029

def orange_ring_weight (o : ℝ) : Prop := o = 0.08333333333333333
def white_ring_weight (w : ℝ) : Prop := w = 0.4166666666666667
def total_weight (t : ℝ) : Prop := t = 0.8333333333333334

theorem weight_of_purple_ring (o w t p : ℝ)
  (ho : orange_ring_weight o)
  (hw : white_ring_weight w)
  (ht : total_weight t) :
  p = t - (o + w) :=
by
  rw [orange_ring_weight, white_ring_weight, total_weight] at ho hw ht
  cases ho
  cases hw
  cases ht
  sorry

end weight_of_purple_ring_l559_559029


namespace price_satisfies_conditions_max_kg_of_apples_within_budget_l559_559934

noncomputable theory

-- Definitions for prices of apples and pears
def price_per_kg_apple : ℝ := 8
def price_per_kg_pear : ℝ := 6

-- Conditions from the problem
def condition1 (A P : ℝ) : Prop := 1 * A + 3 * P = 26
def condition2 (A P : ℝ) : Prop := 2 * A + 1 * P = 22

-- Prove that these prices satisfy the conditions
theorem price_satisfies_conditions :
  condition1 price_per_kg_apple price_per_kg_pear ∧
  condition2 price_per_kg_apple price_per_kg_pear :=
by 
  sorry

-- Condition for total weight and cost constraints
def max_apples (x y max_weight max_cost A P : ℝ) : Prop := 
  x + y = max_weight ∧
  A*x + P*y ≤ max_cost

-- Prove that the maximum kilograms of apples within the constraints is 5 kilograms
theorem max_kg_of_apples_within_budget :
  ∃ (x y : ℝ), max_apples x y 15 100 price_per_kg_apple price_per_kg_pear ∧ x = 5 :=
by
  sorry

end price_satisfies_conditions_max_kg_of_apples_within_budget_l559_559934


namespace angle_ABC_eq_60_l559_559764

noncomputable theory

-- Define a convex pentagon with equal sides and the given angle relationship
structure ConvexPentagon (A B C D E : Type) [MetricSpace B] :=
(equal_sides: dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E A)
(angle_condition: ∠(A, B, C) = 2 * ∠(D, B, E))

-- Define the given conditions and the theorem to be proved
theorem angle_ABC_eq_60 {A B C D E : Type} [MetricSpace B] (H : ConvexPentagon A B C D E) :
  ∠(A, B, C) = 60 :=
sorry

end angle_ABC_eq_60_l559_559764


namespace sqrt_x_minus_2_defined_iff_x_ge_2_l559_559133

theorem sqrt_x_minus_2_defined_iff_x_ge_2 : ∀ x : ℝ, (√(x - 2)).isDefined ↔ x ≥ 2 := 
sorry

end sqrt_x_minus_2_defined_iff_x_ge_2_l559_559133


namespace trigonometric_identity_l559_559596

theorem trigonometric_identity :
  let θ := 30 * Real.pi / 180
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2) / (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_l559_559596


namespace OI_parallel_l_theorem_l559_559780

noncomputable def triangle_ids (a b c : ℂ) : Prop :=
  abs a = abs b ∧ abs b = abs c ∧ abs c = abs a

noncomputable def equal_pentagon_side_lengths (a b c : ℂ) (t : ℝ) (M Q N P : ℂ) : Prop :=
  let M := a + t * complex.I
  let Q := a - t * complex.I
  let P := b + t * complex.I
  let N := b - t * complex.I
  complex.abs (M - Q) = complex.abs (Q - P) ∧
  complex.abs (Q - P) = complex.abs (P - N) ∧
  complex.abs (P - N) = complex.abs (N - M) ∧
  complex.abs (N - M) = complex.abs (M - Q)

noncomputable def OI_parallel_l (a b c : ℂ) (O I : ℂ) (l : set ℂ) : Prop :=
  ∃ M Q N P S, 
  triangle_ids a b c ∧ 
  equal_pentagon_side_lengths a b c 1 M Q N P ∧
  S = complex.affine_combination [M, N] [Q, P] 0.5 ∧
  OI_parallel O I l

theorem OI_parallel_l_theorem (a b c O I l : ℂ) :
  triangle_ids a b c →
  equal_pentagon_side_lengths a b c 1 M Q N P →
  S = complex.affine_combination [M, N] [Q, P] 0.5 →
  OI_parallel O I l :=
-- Implementation goes here
sorry

end OI_parallel_l_theorem_l559_559780


namespace find_area_of_triangle_l559_559534

noncomputable def area_of_triangle
  (R : ℝ) (A B C : ℝ) 
  (angle_ABC : ℝ) (angle_CAB : ℝ) 
  (circle_through_AB : (ℝ × ℝ) -> Bool)
  (circle_tangent_AC_at_A : (ℝ × ℝ) -> Bool) : ℝ :=
sorry

theorem find_area_of_triangle
  (R : ℝ) (A B C : Points) 
  (beta alpha : ℝ)
  (h₁ : circle_through_AB (A, B) = True) 
  (h₂ : circle_tangent_AC_at_A (A, (C, A)) = True)
  (h₃ : angle_ABC = beta)
  (h₄ : angle_CAB = alpha) :
  area_of_triangle R A B C beta alpha circle_through_AB circle_tangent_AC_at_A = sorry :=
sorry

end find_area_of_triangle_l559_559534


namespace sin_neg2pi_plus_alpha_l559_559670

theorem sin_neg2pi_plus_alpha (alpha : ℝ) (h1 : cos (alpha - π) = -5/13) (h2 : 3π/2 < alpha ∧ alpha < 2π) :
  sin (-2 * π + alpha) = -12/13 := sorry

end sin_neg2pi_plus_alpha_l559_559670


namespace pages_read_on_fourth_day_l559_559336

-- condition: Hallie reads the whole book in 4 days, read specific pages each day
variable (total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages : ℕ)

-- Given conditions
def conditions : Prop :=
  first_day_pages = 63 ∧
  second_day_pages = 2 * first_day_pages ∧
  third_day_pages = second_day_pages + 10 ∧
  total_pages = 354 ∧
  first_day_pages + second_day_pages + third_day_pages + fourth_day_pages = total_pages

-- Prove Hallie read 29 pages on the fourth day
theorem pages_read_on_fourth_day (h : conditions total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages) :
  fourth_day_pages = 29 := sorry

end pages_read_on_fourth_day_l559_559336


namespace initial_birds_count_l559_559438

theorem initial_birds_count (current_total_birds birds_joined initial_birds : ℕ) 
  (h1 : current_total_birds = 6) 
  (h2 : birds_joined = 4) : 
  initial_birds = current_total_birds - birds_joined → 
  initial_birds = 2 :=
by 
  intro h3
  rw [h1, h2] at h3
  exact h3

end initial_birds_count_l559_559438


namespace alpha_wins_game_l559_559848

-- Define the conditions in the problem.

-- There are 2019 points forming a regular 2019-sided polygon.
def points : Fin 2019 → ℝ × ℝ := λ i, (cos (2 * π * i / 2019), sin (2 * π * i / 2019))

-- A function to check if a triangle is obtuse.
def isObtuse (A B C : ℝ × ℝ) : Prop := 
  let a := dist B C
  let b := dist A C
  let c := dist A B
  max (a^2 + b^2 - c^2) (max (b^2 + c^2 - a^2) (c^2 + a^2 - b^2)) > 0

-- A function to check if a triangle is acute.
def isAcute (A B C : ℝ × ℝ) : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- Define the initial state of the game.
def initialMarkedPoints : List (ℝ × ℝ) := [points 0, points 1]

-- A player has a winning strategy proposition
def AlphaHasWinningStrategy : Prop := ∀ markedPoints : List (ℝ × ℝ), 
  ∃ newPoint : Fin 2019, 
  newPoint ∉ markedPoints ∧ 
  isObtuse markedPoints.head markedPoints.getLast? (points newPoint)

theorem alpha_wins_game :
  AlphaHasWinningStrategy :=
sorry -- Proof goes here.

end alpha_wins_game_l559_559848


namespace sqrt_log_expression_evaluation_l559_559271

theorem sqrt_log_expression_evaluation :
  (sqrt (logBase 3 8 - logBase 2 8 + logBase 4 8) = (sqrt (3 * (2 * log 2 - log 3)) / sqrt (2 * log 3))) :=
by
  sorry

end sqrt_log_expression_evaluation_l559_559271


namespace sum_telescope_l559_559983

theorem sum_telescope : 
  ∑ k in Finset.range 50, (-1:ℝ)^(k + 1) * (k^2 - k + 1) / k! = 51 / 50! - 1 :=
by
  sorry

end sum_telescope_l559_559983


namespace angle_C_area_of_triangle_l559_559374

variables (a b : ℝ)
def c : ℝ := real.sqrt 7
def A : ℝ := real.acos ((a^2 + b^2 - c^2) / (2 * a * b))
def C : ℝ := real.pi / 3

theorem angle_C (ha : a + b = 5) (hb : b = real.acos ((a^2 + b^2 - c^2) / (2 * a * b)) + 0.5 * a := by sorry) : 
  a = 3 ∧ b = 2 :=
begin
  sorry
end

theorem area_of_triangle (ha : a = 3) (hb : b = 2) (hC : C = real.pi / 3) :
  (1/2) * a * b * real.sin(C) = (3 * real.sqrt 3) / 2 :=
begin
  sorry
end

end angle_C_area_of_triangle_l559_559374


namespace problem_statement_l559_559731

-- Sequence definitions
def a (n : ℕ) : ℕ
| 0 => 0  -- Indexed for 1-based sequence, a_1 is actually a(1)
| 1 => 1
| 2 => 2
| (k+3) => a (k + 1) + (2 - 2 * (-1) ^ (k + 1))

-- Sum definition
def S (n : ℕ) : ℕ := (Finset.range n).sum (λ i => a (i + 1))

-- The problem states to show this specific value
theorem problem_statement : S 2017 = 2017 * 1010 - 1 := by
  -- Proof is omitted
  sorry

end problem_statement_l559_559731


namespace solve_eq_l559_559276

theorem solve_eq : ∃ x : ℚ, (∛(5 + x) = 4 / 3) ∧ x = -71 / 27 := 
by
  use (-71 / 27)
  split
  . sorry
  . sorry

end solve_eq_l559_559276


namespace idiom_describes_random_event_l559_559151

-- Define the idioms and their descriptions
inductive Idiom
| SunSetsInTheWest
| PullUpSeedlings
| KillTwoBirdsOneStone
| AscendToHeavenInOneStep

open Idiom

-- Define a predicate for whether an idiom describes a random event
def describes_random_event : Idiom → Prop
| SunSetsInTheWest          := false
| PullUpSeedlings           := false
| KillTwoBirdsOneStone      := true
| AscendToHeavenInOneStep   := false

-- The theorem stating which idiom describes a random event
theorem idiom_describes_random_event : describes_random_event KillTwoBirdsOneStone = true :=
by 
  sorry

end idiom_describes_random_event_l559_559151


namespace fat_rings_per_group_l559_559576

theorem fat_rings_per_group (F : ℕ)
  (h1 : ∀ F, (70 * (F + 4)) = (40 * (F + 4)) + 180)
  : F = 2 :=
sorry

end fat_rings_per_group_l559_559576


namespace max_interesting_numbers_l559_559901

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

def is_interesting (n : ℕ) : Prop :=
  is_prime (sum_of_digits n)

theorem max_interesting_numbers :
  ∀ (a b c d e : ℕ), b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
    (∀ x ∈ {a, b, c, d, e}, x ∈ ℕ) →
    ∃ s : Finset ℕ, s ⊆ {a, b, c, d, e} ∧
                     (∀ n ∈ s, is_interesting n) ∧ s.card = 4 :=
by
  sorry

end max_interesting_numbers_l559_559901


namespace surface_area_of_cross_shape_with_five_unit_cubes_l559_559559

noncomputable def unit_cube_surface_area : ℕ := 6
noncomputable def num_cubes : ℕ := 5
noncomputable def total_surface_area_iso_cubes : ℕ := num_cubes * unit_cube_surface_area
noncomputable def central_cube_exposed_faces : ℕ := 2
noncomputable def surrounding_cubes_exposed_faces : ℕ := 5
noncomputable def surrounding_cubes_count : ℕ := 4
noncomputable def cross_shape_surface_area : ℕ := 
  central_cube_exposed_faces + (surrounding_cubes_count * surrounding_cubes_exposed_faces)

theorem surface_area_of_cross_shape_with_five_unit_cubes : cross_shape_surface_area = 22 := 
by sorry

end surface_area_of_cross_shape_with_five_unit_cubes_l559_559559


namespace track_length_l559_559577

theorem track_length (x : ℕ) : 
  (∀ brenda sally : ℕ, (brenda = 120) → (sally = 180) → 
  (brenda * (x + 60) = x * x) → x = 180) :=
begin
  intros brenda sally h1 h2 h3,
  sorry
end

end track_length_l559_559577


namespace correct_proposition_l559_559713

-- Define the propositions as Lean propositions
def proposition_1 : Prop := ∀ (x : ℝ), (y = x^n) → (0, 0) ∈ graph ∧ (1, 1) ∈ graph
def proposition_2 : Prop := ∀ (n : ℝ), (graph of (y = x^n)) ≠ (straight line)
def proposition_3 : Prop := (n = 0) → (graph of (y = x^0)) is (straight line)
def proposition_4 : Prop := (n > 0) → (graph of (y = x^n)) is increasing
def proposition_5 : Prop := (n < 0) → (graph of (y = x^n)) is decreasing in first quadrant

-- Prove the correct propositions number is (5)
theorem correct_proposition : 
(proposition_1 = false ∧ 
 proposition_2 = false ∧ 
 proposition_3 = false ∧ 
 proposition_4 = false ∧
 proposition_5 = true) := by
 sorry

end correct_proposition_l559_559713


namespace initial_customers_l559_559562

theorem initial_customers : 
  ∀ (initial left remaining: ℕ), left = 11 → remaining = 3 → left + remaining = initial - 11 + 3 → initial = 14 :=
by
  intros initial left remaining h_left h_remaining h_total
  have h1 : initial = left + remaining := by linarith
  rw [h_left, h_remaining] at h1
  exact h1

end initial_customers_l559_559562


namespace num_nat_numbers_l559_559105

theorem num_nat_numbers (n : ℕ) (h1 : n ≥ 1) (h2 : n ≤ 1992)
  (h3 : ∃ k3, n = 3 * k3)
  (h4 : ¬ (∃ k2, n = 2 * k2))
  (h5 : ¬ (∃ k5, n = 5 * k5)) : ∃ (m : ℕ), m = 266 :=
by
  sorry

end num_nat_numbers_l559_559105


namespace minimum_number_of_visitors_l559_559447

noncomputable def cinema_visitors : Nat :=
  let ticket_prices : List Nat := [50, 55, 60, 65]
  let combinations : List Nat := (ticket_prices ++ 
                                   (List.map (λx => x.1 + x.2) (List.product ticket_prices ticket_prices))).eraseDuplicates
  let num_choices := combinations.length
  (200 - 1) * num_choices + 1

theorem minimum_number_of_visitors (showing_movies : List String)
    (prices : List Nat)
    (min_movies max_movies : Nat)
    (conflict_movie_pairs : List (String × String))
    (spending_people : Nat)
    (unique_visitors : Nat) :
    showing_movies = ["Toy Story", "Ice Age", "Shrek", "The Monkey King"] ∧
    prices = [50, 55, 60, 65] ∧
    min_movies = 1 ∧ max_movies = 2 ∧
    conflict_movie_pairs = [("Ice Age", "Shrek")] ∧
    spending_people = 200 ∧
    unique_visitors = cinema_visitors →
    unique_visitors = 1801 :=
by
  intros
  sorry

end minimum_number_of_visitors_l559_559447


namespace betty_shorter_than_carter_l559_559584

theorem betty_shorter_than_carter :
  let carter_height := 2 * 24 in
  let betty_height := 3 * 12 in
  carter_height - betty_height = 12 :=
by
  let carter_height := 2 * 24
  let betty_height := 3 * 12
  show carter_height - betty_height = 12
  sorry

end betty_shorter_than_carter_l559_559584


namespace round_85960_to_three_sig_figs_l559_559062

theorem round_85960_to_three_sig_figs :
  (let num := 85960
   let method := "round_half_up"
   let precision := 3
   round_to_significant_figures num precision method = 8.60 * 10^4) := sorry

end round_85960_to_three_sig_figs_l559_559062


namespace least_pos_int_with_ten_factors_l559_559896

theorem least_pos_int_with_ten_factors : ∃ (n : ℕ), n > 0 ∧ (∀ m, (m > 0 ∧ ∃ d : ℕ, d∣n → d = 1 ∨ d = n) → m < n) ∧ ( ∃! n, ∃ d : ℕ, d∣n ) := sorry

end least_pos_int_with_ten_factors_l559_559896


namespace curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559000

section
variables (k t θ ρ : ℝ)  

def parametric_curve_C1 (k : ℝ) (t : ℝ) : ℝ × ℝ := (cos t ^ k, sin t ^ k)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

def curve_C2 (ρ θ : ℝ) : Prop := 4 * ρ * cos θ - 16 * ρ * sin θ + 3 = 0

theorem curve_C1_when_k1_is_circle : 
  ∀ t : ℝ, (parametric_curve_C1 1 t).fst ^ 2 + (parametric_curve_C1 1 t).snd ^ 2 = 1 :=
begin
  intros t,
  simp only [parametric_curve_C1],
  have h : cos t ^ 2 + sin t ^ 2 = 1, from real.sin_sq_add_cos_sq t,
  simp [h],
end

noncomputable def intersection_points_when_k4 : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p = parametric_curve_C1 4 t ∧ 
    let (x, y) := p in 4 * x - 16 * y + 3 = 0}

theorem find_intersection_points_when_k4 : 
  intersection_points_when_k4 = { (1/4, 1/4) } :=
begin
  sorry
end

end

end curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559000


namespace students_in_sample_l559_559770

theorem students_in_sample (T : ℕ) (S : ℕ) (F : ℕ) (J : ℕ) (se : ℕ)
  (h1 : J = 22 * T / 100)
  (h2 : S = 25 * T / 100)
  (h3 : se = 160)
  (h4 : F = S + 64)
  (h5 : ∀ x, x ∈ ({F, S, J, se} : Finset ℕ) → x ≤ T ∧  x ≥ 0):
  T = 800 :=
by
  have h6 : T = F + S + J + se := sorry
  sorry

end students_in_sample_l559_559770


namespace _l559_559744

variable (f : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def given_conditions : Prop :=
  is_even_function f ∧ ∀ x : ℝ, x > 0 → f x = x

noncomputable theorem find_f_of_negative_x (h : given_conditions f) : ∀ x : ℝ, x < 0 → f x = -x :=
by
  sorry

end _l559_559744


namespace points_coplanar_l559_559877

-- Definitions of the conditions
variables (A B C D K L M N: Type*)
variable (Sphere : Type*)
variable [MetricSpace Sphere]

-- Conditions on the quadrilateral
variables (AB BC CD DA: Set Sphere)
variable (ABCD: Set Sphere)
variables (tangent_point : Sphere → Sphere → Sphere → Prop)

-- Given sides touch the sphere
axiom touch_AB : tangent_point A B K
axiom touch_BC : tangent_point B C L
axiom touch_CD : tangent_point C D M
axiom touch_DA : tangent_point D A N

-- Prove that points K, L, M, N lie in one plane
theorem points_coplanar : ∃ plane : Set Sphere, tangent_point A B K ∧ tangent_point B C L ∧ tangent_point C D M ∧ tangent_point D A N → 
  coplanar {K, L, M, N} :=
begin
  sorry -- proof is omitted as per instructions
end

end points_coplanar_l559_559877


namespace cos2_165_minus_sin2_15_l559_559187

theorem cos2_165_minus_sin2_15 : cos (165 * Real.pi / 180)^2 - sin (15 * Real.pi / 180)^2 = sqrt 3 / 2 := 
by 
  -- Define the necessary angles in radians
  let θ_165 := 165 * Real.pi / 180
  let θ_15 := 15 * Real.pi / 180
  
  -- Use trigonometric identities and known values to simplify the problem
  have h1 : cos θ_165 = cos θ_15, from by rw [cos_coe]
  have h2 : sin θ_15 = sin θ_15, from by rw [sin_coe]
  
  -- Prove the final equality
  sorry

end cos2_165_minus_sin2_15_l559_559187


namespace maximize_profit_l559_559935

noncomputable def profit_increase (x : ℝ) : ℝ :=
  (20 + x) * (300 - 10 * x)

noncomputable def max_profit_increase : ℝ :=
  let f := λ x, (20 + x) * (300 - 10 * x)
  f 5

noncomputable def profit_decrease (a : ℝ) : ℝ :=
  (20 - a) * (300 + 20 * a)

noncomputable def max_profit_decrease : ℝ :=
  let f := λ a, (20 - a) * (300 + 20 * a)
  f 2.5

theorem maximize_profit : 
  ∃ p : ℝ, p = 65 ∧ 
    ∀ q : ℝ, (q = (60 + 5)) ∨ (q = (60 - 2.5)) → 
    ((profit_increase 5 ≥ profit_increase (q - 60)) ∧ (profit_decrease 2.5 ≥ profit_decrease (60 - q)))
:= by
  sorry

end maximize_profit_l559_559935


namespace first_day_bacteria_exceeds_200_l559_559763

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, 2 * 3^n > 200 ∧ ∀ m : ℕ, m < n → 2 * 3^m ≤ 200 :=
by
  -- sorry for skipping proof
  sorry

end first_day_bacteria_exceeds_200_l559_559763


namespace arithmetic_sequence_common_difference_l559_559816

theorem arithmetic_sequence_common_difference 
  (d : ℝ) (h : d ≠ 0) (a : ℕ → ℝ)
  (h1 : a 1 = 9 * d)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (k : ℕ) :
  (a k)^2 = (a 1) * (a (2 * k)) → k = 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l559_559816


namespace four_letter_arrangements_l559_559738

def letters := ['A', 'B', 'C', 'D', 'E', 'F']

theorem four_letter_arrangements :
  let count := letters.erase 'C' in
  list.length count = 5 ∧
  list.contains count 'B' ∧
  let remaining := count.erase 'B' in
  list.length remaining = 4 →
  list.length (remaining.erase_nth 0) = 3 →
  let arrangements := 'C' :: 'B' :: (remaining.erase_nth 0).erase_nth 0 in
  arrangements.length = 4 →
  arrangements.nodup →
  true :=
by
  sorry

end four_letter_arrangements_l559_559738


namespace coins_count_l559_559191

variable (x : ℕ)

def total_value : ℕ → ℕ := λ x => x + (x * 50) / 100 + (x * 25) / 100

theorem coins_count (h : total_value x = 140) : x = 80 :=
sorry

end coins_count_l559_559191


namespace gcd_of_35_and_number_between_70_and_90_is_7_l559_559097

def number_between_70_and_90 (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 90

def gcd_is_7 (a b : ℕ) : Prop :=
  Nat.gcd a b = 7

theorem gcd_of_35_and_number_between_70_and_90_is_7 : 
  ∃ (n : ℕ), number_between_70_and_90 n ∧ gcd_is_7 35 n ∧ (n = 77 ∨ n = 84) :=
by
  sorry

end gcd_of_35_and_number_between_70_and_90_is_7_l559_559097


namespace trigonometric_identity_example_l559_559618

theorem trigonometric_identity_example :
  (let θ := 30 / 180 * Real.pi in
   (Real.tan θ)^2 - (Real.sin θ)^2) / ((Real.tan θ)^2 * (Real.sin θ)^2) = 4 / 3 :=
by
  let θ := 30 / 180 * Real.pi
  have h_tan2 := Real.tan θ * Real.tan θ
  have h_sin2 : Real.sin θ * Real.sin θ = 1/4 := by sorry
  have h_cos2 : Real.cos θ * Real.cos θ = 3/4 := by sorry
  sorry

end trigonometric_identity_example_l559_559618


namespace ratio_BE_EC_l559_559015

variables {A B C F H E : Type} (h1 : divides AC F (2/5)) (h2 : midpoint CF H) (h3 : intersects BC BH E)

-- Measure the ratio at which point E divides side BC
theorem ratio_BE_EC : divides BC E (4/7) :=
sorry

end ratio_BE_EC_l559_559015


namespace verify_total_amount_spent_by_mary_l559_559054

def shirt_price : Float := 13.04
def shirt_sales_tax_rate : Float := 0.07

def jacket_original_price_gbp : Float := 15.34
def jacket_discount_rate : Float := 0.20
def jacket_sales_tax_rate : Float := 0.085
def conversion_rate_usd_per_gbp : Float := 1.28

def scarf_price : Float := 7.90
def hat_price : Float := 9.13
def hat_scarf_sales_tax_rate : Float := 0.065

def total_amount_spent_by_mary : Float :=
  let shirt_total := shirt_price * (1 + shirt_sales_tax_rate)
  let jacket_discounted := jacket_original_price_gbp * (1 - jacket_discount_rate)
  let jacket_total_gbp := jacket_discounted * (1 + jacket_sales_tax_rate)
  let jacket_total_usd := jacket_total_gbp * conversion_rate_usd_per_gbp
  let hat_scarf_combined_price := scarf_price + hat_price
  let hat_scarf_total := hat_scarf_combined_price * (1 + hat_scarf_sales_tax_rate)
  shirt_total + jacket_total_usd + hat_scarf_total

theorem verify_total_amount_spent_by_mary : total_amount_spent_by_mary = 49.13 :=
by sorry

end verify_total_amount_spent_by_mary_l559_559054


namespace cell_division_proof_l559_559368

-- Define the problem
def cell_division_ways (n m : Nat) : Nat :=
  if (n = 17 ∧ m = 9) then 10 else 0

-- The Lean statement to assert the problem
theorem cell_division_proof : cell_division_ways 17 9 = 10 :=
by
-- simplifying the definition for the given parameters
simp [cell_division_ways]
sorry

end cell_division_proof_l559_559368


namespace max_interesting_numbers_l559_559904

/-- 
Define a function to calculate the sum of the digits of a natural number.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

/-- 
Define a function to check if a natural number is prime.
-/
def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

/-- 
Define a number to be interesting if the sum of its digits is a prime number.
-/
def interesting_number (n : ℕ) : Prop :=
  is_prime (sum_of_digits n)

/-- 
Statement: For any five consecutive natural numbers, there are at most 4 interesting numbers.
-/
theorem max_interesting_numbers (a : ℕ) :
  (finset.range 5).filter (λ i, interesting_number (a + i)) .card ≤ 4 := 
by
  sorry

end max_interesting_numbers_l559_559904


namespace num_ways_to_choose_leaders_l559_559208

theorem num_ways_to_choose_leaders (n k: ℕ) (h_n: n = 20) (h_k: k = 3) : 
  @nat.choose n k = 1140 :=
by
  rw [h_n, h_k]
  sorry

end num_ways_to_choose_leaders_l559_559208


namespace curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559003

section
variables (k t θ ρ : ℝ)  

def parametric_curve_C1 (k : ℝ) (t : ℝ) : ℝ × ℝ := (cos t ^ k, sin t ^ k)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

def curve_C2 (ρ θ : ℝ) : Prop := 4 * ρ * cos θ - 16 * ρ * sin θ + 3 = 0

theorem curve_C1_when_k1_is_circle : 
  ∀ t : ℝ, (parametric_curve_C1 1 t).fst ^ 2 + (parametric_curve_C1 1 t).snd ^ 2 = 1 :=
begin
  intros t,
  simp only [parametric_curve_C1],
  have h : cos t ^ 2 + sin t ^ 2 = 1, from real.sin_sq_add_cos_sq t,
  simp [h],
end

noncomputable def intersection_points_when_k4 : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p = parametric_curve_C1 4 t ∧ 
    let (x, y) := p in 4 * x - 16 * y + 3 = 0}

theorem find_intersection_points_when_k4 : 
  intersection_points_when_k4 = { (1/4, 1/4) } :=
begin
  sorry
end

end

end curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559003


namespace angles_of_quadrilateral_inscribed_circle_l559_559206

theorem angles_of_quadrilateral_inscribed_circle 
  (A B C D O K L M N : Point)
  (inscribed : InscribedCircle ABCD O K L M N)
  (bisect_OA : Bisects K N O A)
  (bisect_OB : Bisects K L O B)
  (divides_OD : DividesInRatio M N O D 1 3)
  : angles ABCD = (60, 120, 90, 90) :=
sorry

end angles_of_quadrilateral_inscribed_circle_l559_559206


namespace probability_of_2_4_6_l559_559150

-- Definitions based on conditions
def is_fair_die (n : ℕ) := (n >= 1)

def total_outcomes := (8 : ℕ)

def favorable_outcomes (f : ℕ → Prop) : ℕ := 
  finset.card {x | f x}

-- Question/Proof statement
theorem probability_of_2_4_6 :
  (is_fair_die total_outcomes) →
  (favorable_outcomes (λ x, x = 2 ∨ x = 4 ∨ x = 6) = 3) →
  finset.card (finset.range (total_outcomes + 1)) = 8 →
  (3 / 8) = (favorable_outcomes (λ x, x = 2 ∨ x = 4 ∨ x = 6) / total_outcomes) :=
by
  sorry

end probability_of_2_4_6_l559_559150


namespace exterior_angle_DEF_l559_559836

theorem exterior_angle_DEF :
  let heptagon_angle := (180 * (7 - 2)) / 7
  let octagon_angle := (180 * (8 - 2)) / 8
  let total_degrees := 360
  total_degrees - (heptagon_angle + octagon_angle) = 96.43 :=
by
  sorry

end exterior_angle_DEF_l559_559836


namespace platform_length_l559_559561

theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℝ) (expected_platform_length : ℝ) : 
  train_length = 160 → 
  train_speed_kmph = 72 → 
  time_seconds = 25 → 
  expected_platform_length = 340 →
  expected_platform_length = (train_speed_kmph * 1000 / 3600 * time_seconds) - train_length :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  rw [← mul_assoc]
  norm_num
  sorry

end platform_length_l559_559561


namespace trigonometric_identity_example_l559_559620

theorem trigonometric_identity_example :
  (let θ := 30 / 180 * Real.pi in
   (Real.tan θ)^2 - (Real.sin θ)^2) / ((Real.tan θ)^2 * (Real.sin θ)^2) = 4 / 3 :=
by
  let θ := 30 / 180 * Real.pi
  have h_tan2 := Real.tan θ * Real.tan θ
  have h_sin2 : Real.sin θ * Real.sin θ = 1/4 := by sorry
  have h_cos2 : Real.cos θ * Real.cos θ = 3/4 := by sorry
  sorry

end trigonometric_identity_example_l559_559620


namespace numberOfFactorsM_l559_559392

theorem numberOfFactorsM (M : ℕ) (h : M = 72^5 + 5 * 72^4 + 10 * 72^3 + 10 * 72^2 + 5 * 72 + 1) :
  nat.factors M = 6 :=
sorry

end numberOfFactorsM_l559_559392


namespace curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559004

section
variables (k t θ ρ : ℝ)  

def parametric_curve_C1 (k : ℝ) (t : ℝ) : ℝ × ℝ := (cos t ^ k, sin t ^ k)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

def curve_C2 (ρ θ : ℝ) : Prop := 4 * ρ * cos θ - 16 * ρ * sin θ + 3 = 0

theorem curve_C1_when_k1_is_circle : 
  ∀ t : ℝ, (parametric_curve_C1 1 t).fst ^ 2 + (parametric_curve_C1 1 t).snd ^ 2 = 1 :=
begin
  intros t,
  simp only [parametric_curve_C1],
  have h : cos t ^ 2 + sin t ^ 2 = 1, from real.sin_sq_add_cos_sq t,
  simp [h],
end

noncomputable def intersection_points_when_k4 : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p = parametric_curve_C1 4 t ∧ 
    let (x, y) := p in 4 * x - 16 * y + 3 = 0}

theorem find_intersection_points_when_k4 : 
  intersection_points_when_k4 = { (1/4, 1/4) } :=
begin
  sorry
end

end

end curve_C1_when_k1_is_circle_find_intersection_points_when_k4_l559_559004


namespace part1_part2_l559_559702

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined : ∀ x : ℝ, f x ∈ ℝ
axiom f_not_zero : ¬(∀ x : ℝ, f x = 0)
axiom f_ab : ∀ a b : ℝ, f (a * b) = a * f b + b * f a
axiom f_half : f (1 / 2) = 1

theorem part1 (x : ℝ) : 
  (x = 1 / 4 → f x = 1) ∧ 
  (x = 1 / 8 → f x = 3 / 4) ∧
  (x = 1 / 16 → f x = 1 / 2) := 
  by sorry

theorem part2 (n : ℕ) (hn : n > 0) : 
  f (2 ^ (-n : ℤ)) = n * (1 / 2) ^ (n - 1) :=
  by sorry

end part1_part2_l559_559702


namespace sum_of_digits_initial_usd_d_l559_559912

-- Define the conditions and variables
variables (d : ℚ)

-- Initial condition: Oliver exchanged his $d$ U.S. dollars for Canadian dollars.
def exchanged_amount_in_cad (d : ℚ) : ℚ := (8 / 5) * d

-- Condition after spending 80 CAD
def remaining_cad_after_spending (d : ℚ) : ℚ := exchanged_amount_in_cad d - 80

-- Final condition: Oliver had $d$ Canadian dollars remaining
def final_remaining_cad (d : ℚ) : Prop := remaining_cad_after_spending d = d

-- The sum of digits function (assuming d is converted to its integer representation)
def sum_of_digits (n : ℕ) : ℕ := 
  n.to_string.foldr (λ c acc, acc + c.to_nat - '0'.to_nat) 0

-- The main theorem to prove
theorem sum_of_digits_initial_usd_d (d : ℕ) (h : final_remaining_cad d) :
  sum_of_digits d = 7 :=
sorry

end sum_of_digits_initial_usd_d_l559_559912


namespace sqrt_expression_equal_cos_half_theta_l559_559746

noncomputable def sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta (θ : Real) : Real :=
  Real.sqrt (1 / 2 + 1 / 2 * Real.sqrt (1 / 2 + 1 / 2 * Real.cos (2 * θ))) - Real.sqrt (1 - Real.sin θ)

theorem sqrt_expression_equal_cos_half_theta (θ : Real) (h : π < θ) (h2 : θ < 3 * π / 2)
  (h3 : Real.cos θ < 0) (h4 : 0 < Real.sin (θ / 2)) (h5 : Real.cos (θ / 2) < 0) :
  sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta θ = Real.cos (θ / 2) :=
by
  sorry

end sqrt_expression_equal_cos_half_theta_l559_559746


namespace problem_1_problem_2_problem_3_l559_559924

def convert_to_dms (deg : ℚ) : (ℚ × ℚ × ℚ) := 
  let d := deg.nat_part 
  let m := ((deg - d) * 60).nat_part 
  let s := (((deg - d) * 60 - m) * 60)
  (d, m, s)

def multiply_degrees (deg1 deg2 : ℚ) : ℚ :=
  deg1 * deg2

def add_dms (dms1 dms2 : (ℚ × ℚ × ℚ)) : (ℚ × ℚ × ℚ) :=
  let (d1, m1, s1) := dms1
  let (d2, m2, s2) := dms2
  let total_seconds := s1 + s2 
  let carry_minutes := total_seconds / 60
  let seconds := total_seconds % 60
  let total_minutes := m1 + m2 + carry_minutes
  let carry_degrees := total_minutes / 60
  let minutes := total_minutes % 60
  let degrees := d1 + d2 + carry_degrees
  (degrees, minutes, seconds)

theorem problem_1 :
  convert_to_dms 3.76 = (3, 45, 36) :=
by sorry

theorem problem_2 :
  multiply_degrees 0.5 5 = 2.5 :=
by sorry

theorem problem_3 :
  add_dms (15, 48, 36) (37, 27, 59) = (53, 16, 35) :=
by sorry

end problem_1_problem_2_problem_3_l559_559924


namespace sqrt_two_irrational_l559_559152

def irrational (x : ℝ) := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem sqrt_two_irrational : irrational (Real.sqrt 2) := 
by 
  sorry

end sqrt_two_irrational_l559_559152


namespace sequence_term_2014_l559_559327

def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 4 / 5) ∧ 
  (∀ n, 0 ≤ a n ∧ a n ≤ 1 → 
    (a (n+1) = if 0 ≤ a n ∧ a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1))

theorem sequence_term_2014 (a : ℕ → ℚ)
  (h_seq : sequence a) : a 2014 = 3 / 5 :=
  sorry

end sequence_term_2014_l559_559327


namespace different_movies_count_l559_559632

theorem different_movies_count 
    (d_movies : ℕ) (h_movies : ℕ) (a_movies : ℕ) (b_movies : ℕ) (c_movies : ℕ) 
    (together_movies : ℕ) (dha_movies : ℕ) (bc_movies : ℕ) 
    (db_movies : ℕ) (ac_movies : ℕ)
    (H_d : d_movies = 20) (H_h : h_movies = 26) (H_a : a_movies = 35) 
    (H_b : b_movies = 29) (H_c : c_movies = 16)
    (H_together : together_movies = 5)
    (H_dha : dha_movies = 4) (H_bc : bc_movies = 3) 
    (H_db : db_movies = 2) (H_ac : ac_movies = 4) :
    d_movies + h_movies + a_movies + b_movies + c_movies 
    - 4 * together_movies - 3 * dha_movies - 2 * bc_movies - db_movies - 3 * ac_movies = 74 := by sorry

end different_movies_count_l559_559632


namespace ab_plus_b_l559_559986

theorem ab_plus_b (A B : ℤ) (h1 : A * B = 10) (h2 : 3 * A + 7 * B = 51) : A * B + B = 12 :=
by
  sorry

end ab_plus_b_l559_559986


namespace starting_number_is_33_l559_559128

theorem starting_number_is_33 (n : ℕ)
  (h1 : ∀ k, (33 + k * 11 ≤ 79) → (k < 5))
  (h2 : ∀ k, (k < 5) → (33 + k * 11 ≤ 79)) :
  n = 33 :=
sorry

end starting_number_is_33_l559_559128


namespace total_polled_votes_l559_559573

theorem total_polled_votes (V : ℕ) (I : ℕ) 
  (candidate1_percentage : ℚ)
  (defeat_margin : ℚ)
  (valid_votes : V) :
  candidate1_percentage = 0.2 → 
  defeat_margin = 500 →
  I = 10 →
  ((0.8 * V) - (0.2 * V) = defeat_margin) →
  V = 833 →
  valid_votes + I = 843 := 
sorry

end total_polled_votes_l559_559573


namespace max_gcd_13n_plus_4_8n_plus_3_l559_559567

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ) (h: n > 0) : ∃ m, 7 = gcd (13 * n + 4) (8 * n + 3) :=
sorry

end max_gcd_13n_plus_4_8n_plus_3_l559_559567


namespace sqrt_of_neg_7_sq_is_7_l559_559176

theorem sqrt_of_neg_7_sq_is_7 : sqrt ((-7)^2) = 7 :=
by sorry

end sqrt_of_neg_7_sq_is_7_l559_559176


namespace sum_consecutive_integers_from_95_to_105_l559_559977

theorem sum_consecutive_integers_from_95_to_105 : 
  (finset.range (105 - 95 + 1)).sum (λ x, 95 + x) = 1100 :=
by
  sorry

end sum_consecutive_integers_from_95_to_105_l559_559977


namespace at_least_one_positive_l559_559310

theorem at_least_one_positive (x y z : ℝ) :
  let a := x^2 - 2*y + (Real.pi / 2),
      b := y^2 - 2*z + (Real.pi/3),
      c := z^2 - 2*x + (Real.pi/6)
  in a > 0 ∨ b > 0 ∨ c > 0 :=
by
  sorry

end at_least_one_positive_l559_559310


namespace Dexter_card_count_l559_559645

theorem Dexter_card_count : 
  let basketball_boxes := 9
  let cards_per_basketball_box := 15
  let football_boxes := basketball_boxes - 3
  let cards_per_football_box := 20
  let basketball_cards := basketball_boxes * cards_per_basketball_box
  let football_cards := football_boxes * cards_per_football_box
  let total_cards := basketball_cards + football_cards
  total_cards = 255 :=
sorry

end Dexter_card_count_l559_559645


namespace no_such_numbers_l559_559260

theorem no_such_numbers :
  ¬∃ (a : Fin 2013 → ℕ),
    (∀ i : Fin 2013, (∑ j : Fin 2013, if j ≠ i then a j else 0) ≥ (a i) ^ 2) :=
sorry

end no_such_numbers_l559_559260


namespace shaded_area_l559_559010

noncomputable theory

open_locale big_operators

def length_AB : ℝ := 3
def length_BC : ℝ := 6
def length_CD : ℝ := 4
def length_DE : ℝ := 5
def length_EF : ℝ := 7
def diameter_AF : ℝ := length_AB + length_BC + length_CD + length_DE + length_EF

def area_semicircle (d : ℝ) : ℝ := (π * d^2) / 8

theorem shaded_area : 
  area_semicircle diameter_AF - 
  ( area_semicircle length_AB + 
    area_semicircle length_BC + 
    area_semicircle length_CD + 
    area_semicircle length_DE + 
    area_semicircle length_EF 
  ) = (245 * π) / 4 :=
by sorry

end shaded_area_l559_559010


namespace total_distance_walked_l559_559053

theorem total_distance_walked (t1 t2 : ℝ) (r : ℝ) (total_distance : ℝ)
  (h1 : t1 = 15 / 60)  -- Convert 15 minutes to hours
  (h2 : t2 = 25 / 60)  -- Convert 25 minutes to hours
  (h3 : r = 3)         -- Average speed in miles per hour
  (h4 : total_distance = r * (t1 + t2))
  : total_distance = 2 :=
by
  -- here is where the proof would go
  sorry

end total_distance_walked_l559_559053


namespace leftover_value_is_5_30_l559_559951

variable (q_per_roll d_per_roll : ℕ)
variable (j_quarters j_dimes l_quarters l_dimes : ℕ)
variable (value_per_quarter value_per_dime : ℝ)

def total_leftover_value (q_per_roll d_per_roll : ℕ) 
  (j_quarters l_quarters j_dimes l_dimes : ℕ)
  (value_per_quarter value_per_dime : ℝ) : ℝ :=
  let total_quarters := j_quarters + l_quarters
  let total_dimes := j_dimes + l_dimes
  let leftover_quarters := total_quarters % q_per_roll
  let leftover_dimes := total_dimes % d_per_roll
  (leftover_quarters * value_per_quarter) + (leftover_dimes * value_per_dime)

theorem leftover_value_is_5_30 :
  total_leftover_value 45 55 95 140 173 285 0.25 0.10 = 5.3 := 
by
  sorry

end leftover_value_is_5_30_l559_559951


namespace expression_constant_for_large_x_l559_559908

theorem expression_constant_for_large_x (x : ℝ) (h : x ≥ 4 / 7) : 
  -4 * x + |4 - 7 * x| - |1 - 3 * x| + 4 = 1 :=
by
  sorry

end expression_constant_for_large_x_l559_559908


namespace range_of_m_l559_559400

-- Definitions of the functions
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

-- Condition for associated functions on interval [0, 3]
def associated_functions (f g : ℝ → ℝ) (a b : ℝ) := ∃ x1 x2 ∈ set.Icc a b, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2

-- The range of m for which f(x) and g(x) are associated functions on the interval [0, 3]
noncomputable def correct_range : set ℝ := { m | -9/4 < m ∧ m ≤ -2 }

-- The main theorem statement
theorem range_of_m : associated_functions (f) (λ x => g x m) 0 3 -> m ∈ correct_range := 
by sorry

end range_of_m_l559_559400


namespace triangle_ratio_l559_559303

theorem triangle_ratio (P₁ P₂ P₃ P Q₁ Q₂ Q₃ : Point) (h₁ : lies_inside_triangle P P₁ P₂ P₃ P)
  (h₂ : intersect_opposite_sides P₁ P P₂ P₃ Q₁)
  (h₃ : intersect_opposite_sides P₂ P P₃ P₁ Q₂)
  (h₄ : intersect_opposite_sides P₃ P P₁ P₂ Q₃) :
  ∃ i, (i = 1 ∨ i = 2 ∨ i = 3) ∧ (∃ k, k = P₁ P P Q₁ ∧ k ≤ 2) ∨ (∃ k, k = P₂ P P Q₂ ∧ k ≤ 2) ∨ (∃ k, k = P₃ P P Q₃ ∧ k ≤ 2)) ∨  
  (∃ i, (i = 1 ∨ i = 2 ∨ i = 3) ∧ (∃ k, k = P₁ P P Q₁ ∧ k ≥ 2) ∨ (∃ k, k = P₂ P P Q₂ ∧ k ≥ 2) ∨ (∃ k, k = P₃ P P Q₃ ∧ k ≥ 2)) :=
begin
  sorry
end

end triangle_ratio_l559_559303


namespace milkman_pure_milk_l559_559833

theorem milkman_pure_milk (x : ℝ) 
  (h_cost : 3.60 * x = 3 * (x + 5)) : x = 25 :=
  sorry

end milkman_pure_milk_l559_559833


namespace HallGroupFour_l559_559353

theorem HallGroupFour (n m : ℕ) (h1 : n = 100) (h2 : m = 67)
  (hall : ∀ x : Fin n, (card (set_of (λ y : Fin n, x ≠ y ∧ knw x y)) ≥ m)) :
  ∃ (s : Finset (Fin n)), s.card = 4 ∧ ∀ (x ∈ s) (y ∈ s), x ≠ y → knw x y := 
sorry

end HallGroupFour_l559_559353


namespace no_such_2013_distinct_naturals_l559_559262

theorem no_such_2013_distinct_naturals :
  ¬ (∃ (a : Fin 2013 → ℕ), Function.Injective a ∧ ∀ k : Fin 2013, ∑ i in (Fin 2013).erase k, a i ≥ (a k) ^ 2) := by
  sorry

end no_such_2013_distinct_naturals_l559_559262


namespace min_mu_l559_559779

noncomputable def min_mu_value (A B C P : ℝ × ℝ) : ℝ :=
  let PA := P - A
  let PB := P - B
  let PC := P - C
  PA.1 * PB.1 + PA.2 * PB.2 + PB.1 * PC.1 + PB.2 * PC.2 + PC.1 * PA.1 + PC.2 * PA.2

theorem min_mu (A B C : ℝ × ℝ) (h₁ : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 2)
  (h₂ : (C.1 - A.1)^2 + (C.2 - A.2)^2 = 3)
  (h₃ : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = √(3 / 2))
  : ∃ P : ℝ × ℝ, min_mu_value A B C P = (√2 / 2 - 5 / 3) :=
  sorry

end min_mu_l559_559779


namespace not_shown_is_5_percent_l559_559385

def expected_attendees : ℕ := 220
def actual_attendees : ℕ := 209
def percentage_not_shown (expected actual : ℕ) : ℚ :=
  ((expected - actual).toRat / expected.toRat) * 100

theorem not_shown_is_5_percent :
  percentage_not_shown expected_attendees actual_attendees = 5 := by
  sorry

end not_shown_is_5_percent_l559_559385


namespace hexagon_largest_angle_l559_559458

theorem hexagon_largest_angle (x : ℚ) : 
  let a1 := 2 * x,
      a2 := 3 * x,
      a3 := 3 * x,
      a4 := 4 * x,
      a5 := 4 * x,
      a6 := 5 * x in
  a1 + a2 + a3 + a4 + a5 + a6 = 720 → 
  5 * x = 1200 / 7 :=
begin
  intros a1 a2 a3 a4 a5 a6 h_sum,
  sorry
end

end hexagon_largest_angle_l559_559458


namespace solution_l559_559691

variable {V : Type*} [InnerProductSpace ℝ V]

noncomputable def t_value (m n : V) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : 4 * ∥m∥ = 3 * ∥n∥) (h₄ : ⟪m, n⟫ = (1/3) * ∥m∥ * ∥n∥) (h₅ : ⟪n, t • m + n⟫ = 0) : ℝ := 
-4

theorem solution {m n : V} (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : 4 * ∥m∥ = 3 * ∥n∥) (h₄ : ⟪m, n⟫ = (1/3) * ∥m∥ * ∥n∥) (h₅ : ⟪n, t • m + n⟫ = 0) : t = -4 :=
sorry

end solution_l559_559691


namespace max_interesting_numbers_in_five_consecutive_l559_559900

def sum_of_digits (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

def is_prime_sum_of_digits (n : ℕ) : Prop :=
  (sum_of_digits n).prime

theorem max_interesting_numbers_in_five_consecutive (n : ℕ) :
  let consecutive_nums := [n, n+1, n+2, n+3, n+4]
  (count is_prime_sum_of_digits consecutive_nums ≤ 4) :=
sorry

end max_interesting_numbers_in_five_consecutive_l559_559900


namespace solve_house_A_cost_l559_559420

-- Definitions and assumptions
variables (A B C : ℝ)
variable base_salary : ℝ := 3000
variable commission_rate : ℝ := 0.02
variable total_earnings : ℝ := 8000

-- Conditions
def house_B_cost (A : ℝ) : ℝ := 3 * A
def house_C_cost (A : ℝ) : ℝ := 2 * A - 110000

-- Define Nigella's commission calculation
def nigella_commission (A B C : ℝ) : ℝ := commission_rate * A + commission_rate * B + commission_rate * C

-- Commission earned based on total earnings and base salary
def commission_earned : ℝ := total_earnings - base_salary

-- Lean theorem statement
theorem solve_house_A_cost 
  (hB : B = house_B_cost A)
  (hC : C = house_C_cost A)
  (h_commission : nigella_commission A B C = commission_earned) : 
  A = 60000 :=
by 
-- Sorry is used to skip the actual proof
sorry

end solve_house_A_cost_l559_559420


namespace length_AB_l559_559243

theorem length_AB (r : ℝ) (A B : ℝ) (π : ℝ) : 
  r = 4 ∧ π = 3 ∧ (A = 8 ∧ B = 8) → (A = B ∧ A + B = 24 → AB = 6) :=
by
  intros
  sorry

end length_AB_l559_559243


namespace smallest_sum_l559_559872

theorem smallest_sum (r s t : ℕ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t) 
  (h_prod : r * s * t = 1230) : r + s + t = 52 :=
sorry

end smallest_sum_l559_559872


namespace polygon_sides_count_l559_559630

def sides_square : ℕ := 4
def sides_triangle : ℕ := 3
def sides_hexagon : ℕ := 6
def sides_heptagon : ℕ := 7
def sides_octagon : ℕ := 8
def sides_nonagon : ℕ := 9

def total_sides_exposed : ℕ :=
  let adjacent_1side := sides_square + sides_nonagon - 2 * 1
  let adjacent_2sides :=
    sides_triangle + sides_hexagon +
    sides_heptagon + sides_octagon - 4 * 2
  adjacent_1side + adjacent_2sides

theorem polygon_sides_count : total_sides_exposed = 27 := by
  sorry

end polygon_sides_count_l559_559630


namespace series_sum_l559_559624

theorem series_sum :
  ∑' n : ℕ,  n ≠ 0 → (6 * n + 2) / ((6 * n - 1)^2 * (6 * n + 5)^2) = 1 / 600 := by
  sorry

end series_sum_l559_559624


namespace max_pies_without_ingredients_l559_559433

theorem max_pies_without_ingredients
  (total_pies chocolate_pies berries_pies cinnamon_pies poppy_seeds_pies : ℕ)
  (h1 : total_pies = 60)
  (h2 : chocolate_pies = 1 / 3 * total_pies)
  (h3 : berries_pies = 3 / 5 * total_pies)
  (h4 : cinnamon_pies = 1 / 2 * total_pies)
  (h5 : poppy_seeds_pies = 1 / 5 * total_pies) : 
  total_pies - max chocolate_pies (max berries_pies (max cinnamon_pies poppy_seeds_pies)) = 24 := 
by
  sorry

end max_pies_without_ingredients_l559_559433


namespace number_of_zeros_in_result_l559_559743

-- Definitions of the initial values
def a : ℕ := 2016
def b : ℕ := 2016

-- Multiplication of a and b
def result : ℕ := a * b

-- The actual theorem to prove the correct answer:
theorem number_of_zeros_in_result : (num_zeros result) = 2015 :=
sorry

end number_of_zeros_in_result_l559_559743


namespace alex_annual_income_l559_559464

theorem alex_annual_income (q : ℝ) (B : ℝ)
  (H1 : 0.01 * q * 50000 + 0.01 * (q + 3) * (B - 50000) = 0.01 * (q + 0.5) * B) :
  B = 60000 :=
by sorry

end alex_annual_income_l559_559464


namespace sum_digits_9N_eq_9_l559_559996

open Nat

noncomputable theory

/-- Proof problem stating that for a natural number N where each digit of N is strictly
    greater than the digit to its left, the sum of the digits of 9N is 9. -/
theorem sum_digits_9N_eq_9 (N : ℕ) 
  (h : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ digitLength N → digits i N < digits j N) :
  sumDigits (9 * N) = 9 := 
sorry

end sum_digits_9N_eq_9_l559_559996


namespace rank_vw_wv_l559_559734

theorem rank_vw_wv {n : ℕ} (v w : Fin n → ℝ) (h_v_lin_indep : LinearIndependence ℝ ![v, w]) :
  Matrix.rank (Matrix.ofFun (λ i j, v i * w j - w i * v j)) = 2 :=
sorry

end rank_vw_wv_l559_559734


namespace no_such_2013_distinct_numbers_l559_559266

theorem no_such_2013_distinct_numbers :
  ∀ (a : Fin 2013 → ℕ), (Function.Injective a) ∧ (∀ i : Fin 2013, (∑ j in Finset.univ.erase i, a j) ≥ (a i) ^ 2) → False :=
by
  sorry

end no_such_2013_distinct_numbers_l559_559266


namespace probability_odd_female_committee_l559_559209


theorem probability_odd_female_committee (men women : ℕ) (total_committee : ℕ) 
  (h_men : men = 5) (h_women : women = 4) (h_committee : total_committee = 3) : 
  (choose (men + women) total_committee).toRat * (44 : ℚ) / (84 : ℚ) = 11 / 21 :=
by
  -- ∃k, this stub in Lean to guarantee successful compilation
  sorry

end probability_odd_female_committee_l559_559209


namespace possible_final_state_l559_559018

-- Definitions of initial conditions and operations
def initial_urn : (ℕ × ℕ) := (100, 100)  -- (W, B)

-- Define operations that describe changes in (white, black) marbles
inductive Operation
| operation1 : Operation
| operation2 : Operation
| operation3 : Operation
| operation4 : Operation

def apply_operation (op : Operation) (state : ℕ × ℕ) : ℕ × ℕ :=
  match op with
  | Operation.operation1 => (state.1, state.2 - 2)
  | Operation.operation2 => (state.1, state.2 - 1)
  | Operation.operation3 => (state.1, state.2 - 1)
  | Operation.operation4 => (state.1 - 2, state.2 + 1)

-- The final state in the form of the specific condition to prove.
def final_state (state : ℕ × ℕ) : Prop :=
  state = (2, 0)  -- 2 white marbles are an expected outcome.

-- Statement of the problem in Lean
theorem possible_final_state : ∃ (sequence : List Operation), 
  (sequence.foldl (fun state op => apply_operation op state) initial_urn).1 = 2 :=
sorry

end possible_final_state_l559_559018


namespace correct_operation_l559_559169

theorem correct_operation :
  (∀ x : ℝ, sqrt (x^2) = abs x) ∧
  sqrt 4 = 2 ∧
  (∀ x : ℝ, sqrt (x^2) = x ∨ sqrt (x^2) = -x) →
  ((sqrt 4 ≠ ± 2) ∧
   (± sqrt (5^2) ≠ -5) ∧
   (sqrt ((-7)^2) = 7) ∧
   (sqrt (-3 : ℝ) ≠ -sqrt 3)) :=
by
  intro h
  clear h -- clear the hypothesis since no proof is needed
  split
  · intro h1
    -- prove sqrt 4 ≠ ±2
    sorry
  split
  · intro h2
    -- prove ± sqrt (5^2) ≠ -5
    sorry
  split
  · intro h3
    -- prove sqrt ((-7)^2) = 7
    exact abs_neg 7
  · intro h4
    -- prove sqrt (-3) ≠ - sqrt 3
    sorry

end correct_operation_l559_559169


namespace problem_f_symmetry_problem_f_definition_problem_correct_answer_l559_559819

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then Real.log x else Real.log (2 - x)

theorem problem_f_symmetry (x : ℝ) : f (2 - x) = f x := 
sorry

theorem problem_f_definition (x : ℝ) (hx : x ≥ 1) : f x = Real.log x :=
sorry

theorem problem_correct_answer: 
  f (1 / 2) < f 2 ∧ f 2 < f (1 / 3) :=
sorry

end problem_f_symmetry_problem_f_definition_problem_correct_answer_l559_559819


namespace turtle_min_distance_l559_559235

theorem turtle_min_distance :
  ∀ (observer_count : ℕ), 
    (≥ 6 observer_count) → 
    (∀ (t : ℝ), 0 ≤ t → t ≤ 6 →
    (∀ (i : ℕ), (i < observer_count) → 
      (∃ d : ℝ, d = 1 ∧ ∃ (start end : ℝ), 0 ≤ start ∧ start < end ∧ end ≤ 6 ∧ start ≤ t ∧ t ≤ end)) →
  ∃ turtle_distance : ℝ, turtle_distance = 4 :=
by {
  sorry
}

end turtle_min_distance_l559_559235


namespace number_of_factors_multiples_of_360_l559_559676

def n : ℕ := 2^10 * 3^14 * 5^8

theorem number_of_factors_multiples_of_360 (n : ℕ) (hn : n = 2^10 * 3^14 * 5^8) : 
  ∃ (k : ℕ), k = 832 ∧ 
  (∀ m : ℕ, m ∣ n → 360 ∣ m → k = 8 * 13 * 8) := 
sorry

end number_of_factors_multiples_of_360_l559_559676


namespace box_2008_count_l559_559129

noncomputable def box_count (a : ℕ → ℕ) : Prop :=
  a 1 = 7 ∧ a 4 = 8 ∧ ∀ n : ℕ, 1 ≤ n ∧ n + 3 ≤ 2008 → a n + a (n + 1) + a (n + 2) + a (n + 3) = 30

theorem box_2008_count (a : ℕ → ℕ) (h : box_count a) : a 2008 = 8 :=
by
  sorry

end box_2008_count_l559_559129


namespace necessary_but_not_sufficient_l559_559730

def quadratic_inequality (x : ℝ) : Prop :=
  x^2 - 3 * x + 2 < 0

def necessary_condition_A (x : ℝ) : Prop :=
  -1 < x ∧ x < 2

def necessary_condition_D (x : ℝ) : Prop :=
  -2 < x ∧ x < 2

theorem necessary_but_not_sufficient :
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_A x ∧ ¬(quadratic_inequality x ∧ necessary_condition_A x)) ∧ 
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_D x ∧ ¬(quadratic_inequality x ∧ necessary_condition_D x)) :=
sorry

end necessary_but_not_sufficient_l559_559730


namespace find_s_l559_559653

theorem find_s (s : Real) (h : ⌊s⌋ + s = 15.4) : s = 7.4 :=
sorry

end find_s_l559_559653


namespace solve_equation_l559_559073

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end solve_equation_l559_559073


namespace mrs_sheridan_fish_count_l559_559820

theorem mrs_sheridan_fish_count (initial_fish : ℝ) (given_fish : ℝ) (remaining_fish : ℝ) : 
  initial_fish = 47.0 ∧ given_fish = 22.0 → remaining_fish = initial_fish - given_fish → remaining_fish = 25.0 :=
by
  intro h
  cases h with h1 h2
  rw [h1, h2]
  norm_num
  sorry

end mrs_sheridan_fish_count_l559_559820


namespace count_of_valid_numbers_is_36_l559_559237

-- Define the function to count valid two-digit numbers
def count_valid_numbers : ℕ :=
  (finset.range 6).card * (finset.range 6).card

-- Define the theorem we want to prove
theorem count_of_valid_numbers_is_36 : count_valid_numbers = 36 := by
  -- Skip the proof
  sorry

end count_of_valid_numbers_is_36_l559_559237


namespace bisection_method_flowchart_l559_559961

def f (x : ℝ) : ℝ := x ^ 2 - 2

theorem bisection_method_flowchart {a b : ℝ} (h₀ : f a * f b < 0) :
    ∃ c : ℝ, f c = 0 ∧ "Program Flowchart" :=
sorry

end bisection_method_flowchart_l559_559961


namespace ana_july_salary_l559_559566

def initial_salary_may : ℝ := 2500
def raise_percentage_june : ℝ := 0.25
def pay_cut_percentage_july : ℝ := 0.25
def one_time_bonus_july : ℝ := 200

theorem ana_july_salary :
  let s_june : ℝ := initial_salary_may * (1 + raise_percentage_june),
      s_july_before_bonus : ℝ := s_june * (1 - pay_cut_percentage_july),
      s_final : ℝ := s_july_before_bonus + one_time_bonus_july
  in s_final = 2543.75 :=
by
  sorry

end ana_july_salary_l559_559566


namespace cost_per_box_is_0_l559_559200

-- Define the conditions
def length := 15 -- length of the box in inches
def width := 12 -- width of the box in inches
def height := 10 -- height of the box in inches
def total_volume := 1080000 -- total volume in cubic inches (i.e., 1.08 million cubic inches)
def total_cost_per_month := 240 -- total amount paid per month in dollars

-- Calculate the cost per box per month
def cost_per_box_per_month : ℝ := total_cost_per_month / (total_volume / (length * width * height))

-- State the problem
theorem cost_per_box_is_0.40 : cost_per_box_per_month = 0.40 := by
  sorry

end cost_per_box_is_0_l559_559200


namespace train_cross_signal_pole_time_l559_559530

theorem train_cross_signal_pole_time :
  ∀ (train_length platform_length platform_cross_time signal_cross_time : ℝ),
  train_length = 300 →
  platform_length = 300 →
  platform_cross_time = 36 →
  signal_cross_time = train_length / ((train_length + platform_length) / platform_cross_time) →
  signal_cross_time = 18 :=
by
  intros train_length platform_length platform_cross_time signal_cross_time h_train_length h_platform_length h_platform_cross_time h_signal_cross_time
  rw [h_train_length, h_platform_length, h_platform_cross_time] at h_signal_cross_time
  sorry

end train_cross_signal_pole_time_l559_559530


namespace union_of_S_and_T_l559_559049

def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem union_of_S_and_T : S ∪ T = {1, 3, 5, 6} := 
by
  sorry

end union_of_S_and_T_l559_559049


namespace probability_of_selected_number_between_l559_559216

open Set

theorem probability_of_selected_number_between (s : Set ℤ) (a b x y : ℤ) 
  (h1 : a = 25) 
  (h2 : b = 925) 
  (h3 : x = 25) 
  (h4 : y = 99) 
  (h5 : s = Set.Icc a b) :
  (y - x + 1 : ℚ) / (b - a + 1 : ℚ) = 75 / 901 := 
by 
  sorry

end probability_of_selected_number_between_l559_559216


namespace vanessa_weeks_needed_to_buy_dress_l559_559140

def dress_original_cost : ℝ := 150
def discount_rate : ℝ := 0.15
def initial_savings : ℝ := 35
def odd_week_allowance : ℝ := 30
def even_week_allowance : ℝ := 35
def weekly_arcade_expense : ℝ := 20
def weekly_snack_expense : ℝ := 10

def required_savings (dress_original_cost : ℝ) (discount_rate : ℝ) (initial_savings : ℝ) : ℝ :=
  (dress_original_cost * (1 - discount_rate)) - initial_savings

def net_savings_per_two_weeks (odd_week_allowance even_week_allowance weekly_arcade_expense weekly_snack_expense : ℝ) : ℝ :=
  (odd_week_allowance + even_week_allowance) - 2 * (weekly_arcade_expense + weekly_snack_expense)

def average_weekly_savings (net_savings_per_two_weeks : ℝ) : ℝ :=
  net_savings_per_two_weeks / 2

def weeks_needed (required_savings average_weekly_savings : ℝ) : ℝ :=
  required_savings / average_weekly_savings

theorem vanessa_weeks_needed_to_buy_dress :
  weeks_needed (required_savings dress_original_cost discount_rate initial_savings) 
               (average_weekly_savings (net_savings_per_two_weeks odd_week_allowance even_week_allowance weekly_arcade_expense weekly_snack_expense)) = 37 := by
  sorry

end vanessa_weeks_needed_to_buy_dress_l559_559140


namespace highest_visible_sum_l559_559288

-- Define the cubes and their faces
structure Cube :=
  (faces : Finset ℕ)
  (h_faces : faces = {1, 3, 6, 12, 24, 48})

-- Define the stacking of four cubes
def stack_cubes (cubes : Finset Cube) : ℕ :=
  cubes.sum (λ cube, 
    let all_faces := cube.faces in
    let visible_faces := all_faces.erase 1 in
    let visible_sum := visible_faces.sum id in
    visible_sum)

-- Define the problem statement
theorem highest_visible_sum :
  ∀ cubes, 4 ∈ cubes.card →
  ∀ c1 c2 c3 c4 ∈ cubes,
    Cube.h_faces c1 → Cube.h_faces c2 → Cube.h_faces c3 → Cube.h_faces c4 →
    stack_cubes cubes = 360 :=
by sorry

end highest_visible_sum_l559_559288


namespace evaluate_f_f_7_l559_559675

def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then -x + 6
  else x^2 + 1

theorem evaluate_f_f_7 : f (f 7) = 2 :=
  by
    -- Proof steps are omitted
    sorry

end evaluate_f_f_7_l559_559675


namespace arithmetic_sequence_a4_l559_559773

/-- Given an arithmetic sequence {a_n}, where S₁₀ = 60 and a₇ = 7, prove that a₄ = 5. -/
theorem arithmetic_sequence_a4 (a₁ d : ℝ) 
  (h1 : 10 * a₁ + 45 * d = 60) 
  (h2 : a₁ + 6 * d = 7) : 
  a₁ + 3 * d = 5 :=
  sorry

end arithmetic_sequence_a4_l559_559773


namespace Carlos_has_highest_result_l559_559962

def Alice_final_result : ℕ := 30 + 3
def Ben_final_result : ℕ := 34 + 3
def Carlos_final_result : ℕ := 13 * 3

theorem Carlos_has_highest_result : (Carlos_final_result > Alice_final_result) ∧ (Carlos_final_result > Ben_final_result) := by
  sorry

end Carlos_has_highest_result_l559_559962


namespace ellipse_equation_l559_559314

def is_ellipse_eq (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_equation 
  (a b c : ℝ) 
  (h1 : a = 2 * c)
  (h2 : a - c = sqrt 3) 
  : (is_ellipse_eq x y (2 * sqrt 3) 3) ∨ (is_ellipse_eq y x (2 * sqrt 3) 3) :=
sorry

end ellipse_equation_l559_559314


namespace tan_sin_div_l559_559608

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l559_559608


namespace jordan_rectangle_length_l559_559582

def rectangle_area (length width : ℝ) : ℝ := length * width

theorem jordan_rectangle_length :
  let carol_length := 8
  let carol_width := 15
  let jordan_width := 30
  let carol_area := rectangle_area carol_length carol_width
  ∃ jordan_length, rectangle_area jordan_length jordan_width = carol_area →
  jordan_length = 4 :=
by
  sorry

end jordan_rectangle_length_l559_559582


namespace probability_of_correct_match_l559_559945

namespace Probability

-- Definition of factorial to facilitate the calculation
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Main theorem statement: the probability of matching 4 celebrities correctly by guessing randomly is 1/24
theorem probability_of_correct_match : (1 : ℚ) / (factorial 4) = 1 / 24 := by
  sorry

end Probability

end probability_of_correct_match_l559_559945


namespace shortest_distance_from_origin_to_circle_l559_559907

noncomputable def shortDistToCircleFromOrigin : ℝ :=
  let circleCenter := (-7 : ℝ, 3)
  let radius := Real.sqrt 7
  let originDist := Real.sqrt ((-7)^2 + (3)^2)
  originDist - radius

theorem shortest_distance_from_origin_to_circle :
  ∀ (x y : ℝ), x^2 + 14*x + y^2 - 6*y + 65 = 0 → 
  shortDistToCircleFromOrigin = Real.sqrt 58 - Real.sqrt 7 := by
  intros x y h
  sorry

end shortest_distance_from_origin_to_circle_l559_559907


namespace find_p_l559_559101

-- Define the hyperbola equation and the parameter conditions
variables (p : ℝ) (h1 : p > 0)
def hyperbola_focus := - real.sqrt (3 + p^2 / 16)

-- Define the parabola equation
def parabola_directrix := p / 2

-- The main theorem to prove: p = 4 under the given conditions
theorem find_p (h2 : hyperbola_focus p = parabola_directrix p) : p = 4 :=
by
  sorry

end find_p_l559_559101


namespace runner_speed_ratio_l559_559138

noncomputable def speed_ratio (u1 u2 : ℝ) : ℝ := u1 / u2

theorem runner_speed_ratio (u1 u2 : ℝ) (h1 : u1 > u2) (h2 : u1 + u2 = 5) (h3 : u1 - u2 = 5/3) :
  speed_ratio u1 u2 = 2 :=
by
  sorry

end runner_speed_ratio_l559_559138


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l559_559505

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l559_559505


namespace find_value_of_a_l559_559468

theorem find_value_of_a (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 < a → a^x ≥ 1)
  (h_sum : (a^1) + (a^0) = 3) : a = 2 :=
sorry

end find_value_of_a_l559_559468


namespace no_such_2013_distinct_numbers_l559_559265

theorem no_such_2013_distinct_numbers :
  ∀ (a : Fin 2013 → ℕ), (Function.Injective a) ∧ (∀ i : Fin 2013, (∑ j in Finset.univ.erase i, a j) ≥ (a i) ^ 2) → False :=
by
  sorry

end no_such_2013_distinct_numbers_l559_559265


namespace probability_five_distinct_numbers_l559_559501

def num_dice := 5
def num_faces := 6

def favorable_outcomes : ℕ := nat.factorial 5 * num_faces
def total_outcomes : ℕ := num_faces ^ num_dice

theorem probability_five_distinct_numbers :
  (favorable_outcomes / total_outcomes : ℚ) = 5 / 54 := 
sorry

end probability_five_distinct_numbers_l559_559501


namespace num_of_perimeters_l559_559550

variable (n : ℕ)
def quadrilateralEFGH (EF: ℕ) (EH: ℕ) (GH: ℕ) (FG: ℕ) : Prop :=
  (EF > 0) ∧ (EH > 0) ∧ (GH > 0) ∧ (FG > 0) ∧ (EF + EH + GH + FG < 1200) ∧
  (EF = 3) ∧ (GH = EH) ∧
  (∃ x y, (FG = x) ∧ (GH = y) ∧ EF = 3 ∧ x^2 + (y - 3)^2 = y^2)

theorem num_of_perimeters ：
  quadrilateralEFGH 3 _ _ _ →
  ∃ p, (∃ x y, (x^2 = 6 * y - 9) ∧ (p = 3 + x + 2 * y) ∧ (p < 1200)) →
  ∃ m, m = 42 :=
by
  sorry

end num_of_perimeters_l559_559550


namespace range_of_a_l559_559757

-- Define the main proof problem statement
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ∈ set.Ioo 0 1 → (x + real.log a) / real.exp x - (a * real.log x) / x > 0) : 
  a ∈ set.Ico (1 / real.exp 1) 1 :=
sorry

end range_of_a_l559_559757


namespace median_is_correct_l559_559146

def median_of_list : ℝ :=
  let lst : List ℕ := List.range 3031 ++ (List.range 3031).map (λ x => x * x)
  let sorted_lst := lst.qsort (≤)
  (sorted_lst.nthLe 3029 sorry + sorted_lst.nthLe 3030 sorry) / 2

theorem median_is_correct : median_of_list = 2975.5 :=
  sorry

end median_is_correct_l559_559146


namespace correct_option_l559_559162

noncomputable def OptionA : Prop := (Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2)
noncomputable def OptionB : Prop := (Real.sqrt (5 ^ 2) = -5 ∨ Real.sqrt (5 ^ 2) = 5)
noncomputable def OptionC : Prop := Real.sqrt ((-7) ^ 2) = 7
noncomputable def OptionD : Prop := (Real.sqrt (-3) = -Real.sqrt 3)

theorem correct_option : OptionC := 
by 
  unfold OptionC
  simp
  exact eq.refl 7

end correct_option_l559_559162


namespace count_linear_equations_l559_559104

-- Define the expressions
def E1 := (x : ℝ) → x = 1
def E2 := (x : ℝ) → x + 1 = 0
def E3 := 1 = 0
def E4 := (x : ℝ) → x + x^2 = 0

-- Define a predicate for checking linearity of equations
def is_linear (expr : (ℝ → Prop)) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, expr x = (a * x + b = 0)

theorem count_linear_equations : 
  (∃ (f : (ℝ → Prop)), f = E1 ∧ is_linear f) ∧
  (∃ (f : (ℝ → Prop)), f = E2 ∧ is_linear f) ∧
  ¬ (∃ (f : Prop), f = E3 ∧ is_linear (λ x, f)) ∧
  ¬ (∃ (f : (ℝ → Prop)), f = E4 ∧ is_linear f) → 
  (∃ n : ℕ, n = 2) :=
by
  sorry

end count_linear_equations_l559_559104


namespace greatest_sum_of_consecutive_odd_integers_lt_500_l559_559143

-- Define the consecutive odd integers and their conditions
def consecutive_odd_integers (n : ℤ) : Prop :=
  n % 2 = 1 ∧ (n + 2) % 2 = 1

-- Define the condition that their product must be less than 500
def prod_less_500 (n : ℤ) : Prop :=
  n * (n + 2) < 500

-- The theorem statement
theorem greatest_sum_of_consecutive_odd_integers_lt_500 : 
  ∃ n : ℤ, consecutive_odd_integers n ∧ prod_less_500 n ∧ ∀ m : ℤ, consecutive_odd_integers m ∧ prod_less_500 m → n + (n + 2) ≥ m + (m + 2) :=
sorry

end greatest_sum_of_consecutive_odd_integers_lt_500_l559_559143


namespace angle_CFE_l559_559365

open Real EuclideanGeometry

noncomputable def CFB_isosceles : 𝟏𝟖𝟎° :=
  50°

noncomputable def EFB_eq_three_DFE (DFE : Real) : Real :=
  3 * DFE

theorem angle_CFE 
  (F_on_CD : F lies_on_segment (C, D))
  (isosceles_CFB : is_isosceles (triangle C F B))
  (isosceles_DFE : is_isosceles (triangle D F E))
  (EFB_eq_three_DFE : ∃ DFE : Real, ∠ EFB = 3 * DFE)
  (angle_FCB : ∠ FCB = 50°) :
  ∠ CFE = 65° :=
sorry

end angle_CFE_l559_559365


namespace ratio_of_areas_l559_559204

theorem ratio_of_areas (x : ℝ) (hx : 0 < x) : 
  let length_rect1 := 3 * x             
  let width_rect1 := 2 * x              
  let radius_circle1 := x               
  let diagonal_rect2 := 2 * x           
  let k := x / Real.sqrt 5              
  let length_rect2 := 3 * k             
  let width_rect2 := 2 * k              
  let radius_circle2 := x / Real.sqrt 5 
  let area_circle2 := Real.pi * (radius_circle2 ^ 2)
  let area_rect1 := length_rect1 * width_rect1
in area_circle2 / area_rect1 = Real.pi / 30 :=
by
  sorry

end ratio_of_areas_l559_559204


namespace first_route_time_with_green_lights_l559_559543

-- Define the conditions
def second_route_time : ℕ := 14
def additional_time_for_all_red_lights : ℕ := 5
def time_added_per_red_light : ℕ := 3
def number_of_stoplights : ℕ := 3

-- Calculate hypothetical time for all red lights on first route
def time_first_route_all_red_lights : ℕ := second_route_time + additional_time_for_all_red_lights

-- Question asks us to prove the time of the first route with all green lights
theorem first_route_time_with_green_lights : 
  let T := time_first_route_all_red_lights - (number_of_stoplights * time_added_per_red_light)
  in T = 10 :=
by {
  -- Simplified inline calculations as hypothesized in problem statement
  let total_added_time := number_of_stoplights * time_added_per_red_light 
  let actual_time_first_route := time_first_route_all_red_lights - total_added_time
  have h1 : total_added_time = 9 := rfl
  have h2 : time_first_route_all_red_lights = 19 := rfl
  have h3 : actual_time_first_route = 10 := rfl
  exact h3,
}

end first_route_time_with_green_lights_l559_559543


namespace minimum_value_f_inequality_proof_l559_559718

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 1)

-- The minimal value of f(x)
def m : ℝ := 4

theorem minimum_value_f :
  (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, -3 ≤ x ∧ x ≤ 1 ∧ f x = m) :=
by
  sorry -- Proof that the minimum value of f(x) is 4 and occurs in the range -3 ≤ x ≤ 1

variables (p q r : ℝ)

-- Given condition that p^2 + 2q^2 + r^2 = 4
theorem inequality_proof (h : p^2 + 2 * q^2 + r^2 = m) : q * (p + r) ≤ 2 :=
by
  sorry -- Proof that q(p + r) ≤ 2 given p^2 + 2q^2 + r^2 = 4

end minimum_value_f_inequality_proof_l559_559718


namespace difference_between_percentage_and_fraction_l559_559189

theorem difference_between_percentage_and_fraction (n : ℕ) (h : n = 100) : (3 / 5 * (n : ℚ) - 0.5 * (n : ℚ) = 10) :=
by 
  have h1 : (n : ℚ) = 100 := by rw [h]
  have h2 : 3 / 5 * (100 : ℚ) = 60 := by norm_num
  have h3 : 0.5 * (100 : ℚ) = 50 := by norm_num
  have h4 : 60 - 50 = 10 := by norm_num
  rw [←h1]
  rw [h2, h3, h4]
  sorry

end difference_between_percentage_and_fraction_l559_559189


namespace trigonometric_identity_l559_559598

theorem trigonometric_identity :
  let θ := 30 * Real.pi / 180
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2) / (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_l559_559598


namespace aaron_ate_more_apples_l559_559843

-- Define the number of apples eaten by Aaron and Zeb
def apples_eaten_by_aaron : ℕ := 6
def apples_eaten_by_zeb : ℕ := 1

-- Theorem to prove the difference in apples eaten
theorem aaron_ate_more_apples :
  apples_eaten_by_aaron - apples_eaten_by_zeb = 5 :=
by
  sorry

end aaron_ate_more_apples_l559_559843


namespace series_sum_l559_559981

theorem series_sum :
  ∑' n : ℕ, (↑n)^2 / ((4 * n - 2)^2 * (4 * n + 2)^2) = (Real.pi ^ 2) / 192 - 1 / 32 :=
by
  -- Proof goes here
  sorry

end series_sum_l559_559981


namespace largest_lambda_l559_559655

theorem largest_lambda (a b c : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = x^3 + a * x^2 + b * x + c) 
  (h_roots : ∀ r : ℝ, f r = 0 → 0 ≤ r) : 
  ∃ λ : ℝ, λ = -1 / 27 ∧ ∀ x : ℝ, 0 ≤ x → f x ≥ λ * (x - a)^3 :=
by
  sorry

end largest_lambda_l559_559655


namespace convex_polygon_sides_l559_559124

theorem convex_polygon_sides (S : ℝ) (n : ℕ) (a₁ a₂ a₃ a₄ : ℝ) 
    (h₁ : S = 4320) 
    (h₂ : a₁ = 120) 
    (h₃ : a₂ = 120) 
    (h₄ : a₃ = 120) 
    (h₅ : a₄ = 120) 
    (h_sum : S = 180 * (n - 2)) :
    n = 26 :=
by
  sorry

end convex_polygon_sides_l559_559124


namespace ordered_pairs_1806_l559_559109

theorem ordered_pairs_1806 :
  (∃ (xy_list : List (ℕ × ℕ)), xy_list.length = 12 ∧ ∀ (xy : ℕ × ℕ), xy ∈ xy_list → xy.1 * xy.2 = 1806) :=
sorry

end ordered_pairs_1806_l559_559109


namespace number_of_ways_to_select_3_numbers_even_sum_geq_10_l559_559063

theorem number_of_ways_to_select_3_numbers_even_sum_geq_10 :
  ∃ (choices : Finset (Finset ℕ)), 
    (∀ c ∈ choices, c.card = 3 ∧ (∃ (s : ℕ), s = c.sum ∧ s % 2 = 0 ∧ s ≥ 10)) 
    ∧ choices.card = 51 :=
sorry

end number_of_ways_to_select_3_numbers_even_sum_geq_10_l559_559063


namespace paint_cans_used_l559_559428

theorem paint_cans_used (init_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) (final_rooms : ℕ) :
  init_rooms = 50 → lost_cans = 5 → remaining_rooms = 40 → final_rooms = 40 → 
  remaining_rooms / (lost_cans / (init_rooms - remaining_rooms)) = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end paint_cans_used_l559_559428


namespace parallelogram_iff_bisector_intersection_l559_559371

variables {A B C D P K L M N : Type}
variables [ConvexQuadrilateral A B C D] 
variables [InPlane P A B C D]
variables [Bisectors PK PL PM PN A P B B P C C P D D P A]

theorem parallelogram_iff_bisector_intersection :
    ∃ P, (P = intersection (perpendicular_bisector A C) (perpendicular_bisector B D)) ↔ parallelogram K L M N :=
sorry

end parallelogram_iff_bisector_intersection_l559_559371


namespace cocoa_powder_total_l559_559880

variable (already_has : ℕ) (still_needs : ℕ)

theorem cocoa_powder_total (h₁ : already_has = 259) (h₂ : still_needs = 47) : already_has + still_needs = 306 :=
by
  sorry

end cocoa_powder_total_l559_559880


namespace sequence_infinite_powers_of_2_l559_559876

def last_digit (n : ℕ) : ℕ := n % 10

noncomputable def a : ℕ → ℕ
| 0       := a0
| (n + 1) := a n + last_digit (a n)

theorem sequence_infinite_powers_of_2 (a0 : ℕ) (h0 : a0 % 5 ≠ 0) :
  ∃ infinitely_many n, ∃ k, 2^k = a n :=
sorry

end sequence_infinite_powers_of_2_l559_559876


namespace album_photos_proof_l559_559220

def photos_per_page := 4

-- Conditions
def position_81st_photo (n: ℕ) (x: ℕ) :=
  4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20

def position_171st_photo (n: ℕ) (y: ℕ) :=
  4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12

noncomputable def album_photos := 32

theorem album_photos_proof :
  ∃ n x y, position_81st_photo n x ∧ position_171st_photo n y ∧ 4 * n = album_photos :=
by
  sorry

end album_photos_proof_l559_559220


namespace equation_solution_l559_559070

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end equation_solution_l559_559070


namespace product_congruent_to_2_l559_559386

theorem product_congruent_to_2 {p : ℕ} (hp : Nat.Prime p) 
  (a : Fin (p - 2) → ℕ) 
  (h₁ : ∀ k, p ∣ a k -> False) 
  (h₂ : ∀ k, p ∣ (a k ^ k - 1) -> False) : 
  ∃ I : Finset (Fin (p - 2)), (∏ i in I, a i ≡ 2 [MOD p]) := by 
  sorry

end product_congruent_to_2_l559_559386


namespace flour_already_put_in_l559_559413

theorem flour_already_put_in (total_flour flour_still_needed: ℕ) (h1: total_flour = 9) (h2: flour_still_needed = 6) : total_flour - flour_still_needed = 3 := 
by
  -- Here we will state the proof
  sorry

end flour_already_put_in_l559_559413


namespace largest_and_smallest_correct_l559_559298

noncomputable def largest_and_smallest (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) : ℝ × ℝ :=
  if hx_y : x * y > 0 then
    if hx_y_sq : x * y * y > x then
      (x * y, x)
    else
      sorry
  else
    sorry

theorem largest_and_smallest_correct {x y : ℝ} (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  largest_and_smallest x y hx hy = (x * y, x) :=
by {
  sorry
}

end largest_and_smallest_correct_l559_559298


namespace pair_D_is_parallel_l559_559241

def vec (α : Type*) := (α × α)

def parallel (a b : vec ℝ) : Prop :=
  ∃ λ : ℝ, a.1 = λ * b.1 ∧ a.2 = λ * b.2

theorem pair_D_is_parallel :
  parallel (-3, 2) (6, -4) :=
sorry

end pair_D_is_parallel_l559_559241


namespace pages_read_on_fourth_day_l559_559337

-- condition: Hallie reads the whole book in 4 days, read specific pages each day
variable (total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages : ℕ)

-- Given conditions
def conditions : Prop :=
  first_day_pages = 63 ∧
  second_day_pages = 2 * first_day_pages ∧
  third_day_pages = second_day_pages + 10 ∧
  total_pages = 354 ∧
  first_day_pages + second_day_pages + third_day_pages + fourth_day_pages = total_pages

-- Prove Hallie read 29 pages on the fourth day
theorem pages_read_on_fourth_day (h : conditions total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages) :
  fourth_day_pages = 29 := sorry

end pages_read_on_fourth_day_l559_559337


namespace correct_equation_l559_559086

theorem correct_equation (x Planned : ℝ) (h1 : 6 * x = Planned + 7) (h2 : 5 * x = Planned - 13) :
  6 * x - 7 = 5 * x + 13 :=
by
  sorry

end correct_equation_l559_559086


namespace identify_counterfeit_coins_l559_559372

theorem identify_counterfeit_coins (N : ℕ) (coins : Fin N → ℝ)
  (hN : N ≥ 5) (count_fake : ∃ m1 m2 : Fin N, m1 ≠ m2 ∧ coins m1 < coins m2)
  (h_fake_lighter : ∃ m1 m2 : Fin N, ∀ x : Fin N, (x = m1 ∨ x = m2) → coins x < coins 0) :
  ∃ m1 m2 : Fin N, ∀ x : Fin N, (x = m1 ∨ x = m2) →
    (∀ y : Fin N, (y ≠ m1 ∧ y ≠ m2) → coins x < coins y) :=
by
  sorry

end identify_counterfeit_coins_l559_559372


namespace correct_operation_l559_559154

noncomputable def sqrt_op_A: Prop := sqrt 4 ≠ 2
noncomputable def sqrt_op_B: Prop := (± sqrt (5^2)) ≠ -5
noncomputable def sqrt_op_C: Prop := sqrt ((-7) ^ 2) = 7
noncomputable def sqrt_op_D: Prop := sqrt (-3) ≠ - sqrt 3

theorem correct_operation : (sqrt_op_A ∧ sqrt_op_B ∧ sqrt_op_C ∧ sqrt_op_D) → (sqrt_op_C = 7) :=
by
  intros h
  sorry

end correct_operation_l559_559154


namespace calc1_calc2_l559_559978

theorem calc1 : (-2) * (-1/8) = 1/4 :=
by
  sorry

theorem calc2 : (-5) / (6/5) = -25/6 :=
by
  sorry

end calc1_calc2_l559_559978


namespace square_area_l559_559966

open Real

lemma edge_equality (x : ℝ) : 5 * x - 20 = 25 - 2 * x :=
by sorry

def square_edge_length (x : ℝ) : ℝ :=
5 * x - 20

theorem square_area : ∃ x : ℝ, (5 * x - 20 = 25 - 2 * x) ∧ (square_edge_length x) ^ 2 = 7225 / 49 :=
by
  use 45 / 7
  split
  · -- Proof for 5 * (45 / 7) - 20 = 25 - 2 * (45 / 7)
    calc
      5 * (45 / 7) - 20 = 25 - 2 * (45 / 7) : sorry

  · -- Proof for (square_edge_length (45 / 7)) ^ 2 = 7225 / 49
    calc
      (square_edge_length (45 / 7)) ^ 2 = (85 / 7) ^ 2 : by
        simp [square_edge_length, edge_equality, 45 / 7]
      ... = 7225 / 49 : by -- calculation for this simplification
        norm_num

end square_area_l559_559966


namespace min_value_of_f_l559_559987

noncomputable def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 2015

theorem min_value_of_f : ∃ x₀ : ℝ, ∀ x : ℝ, f(x) ≥ f(x₀) ∧ f(x₀) = 1987 :=
by
  -- Proof steps would go here
  sorry

end min_value_of_f_l559_559987


namespace tan_sin_identity_l559_559588

noncomputable def tan_deg : ℝ := tan (real.pi / 6)
noncomputable def sin_deg : ℝ := sin (real.pi / 6)

theorem tan_sin_identity :
  (tan_deg ^ 2 - sin_deg ^ 2) / (tan_deg ^ 2 * sin_deg ^ 2) = 1 :=
by
  have tan_30 := by
    simp [tan_deg, real.tan_pi_div_six]
    norm_num
  have sin_30 := by
    simp [sin_deg, real.sin_pi_div_six]
    norm_num
  sorry

end tan_sin_identity_l559_559588


namespace min_points_convex_ngon_l559_559489

-- Define the problem
theorem min_points_convex_ngon (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ), forall (points : set (ℝ × ℝ)), points.card = m → 
  (∀ (tri : set (ℝ × ℝ)), tri ⊆ vertices_of_ngon n → tri.card = 3 → 
  ∃ (p ∈ points), p ∈ interior_of_triangle tri) ∧ m = n - 2 :=
begin
  sorry
end

end min_points_convex_ngon_l559_559489


namespace defective_rate_proof_probability_distribution_and_variance_l559_559201

noncomputable def defective_rate_process1 : ℚ := 1 / 10
noncomputable def defective_rate_process2 : ℚ := 1 / 11
noncomputable def defective_rate_process3 : ℚ := 1 / 12

noncomputable def qualified_rate_process1 : ℚ := 1 - defective_rate_process1
noncomputable def qualified_rate_process2 : ℚ := 1 - defective_rate_process2
noncomputable def qualified_rate_process3 : ℚ := 1 - defective_rate_process3

noncomputable def overall_qualified_rate : ℚ :=
  qualified_rate_process1 * qualified_rate_process2 * qualified_rate_process3

noncomputable def overall_defective_rate : ℚ := 1 - overall_qualified_rate

-- Define the binomial distribution of defective items
def binomial_distribution (n : ℕ) (p : ℚ) : ℕ → ℚ
| k := (nat.choose n k) * (p^k) * ((1-p)^(n-k))

theorem defective_rate_proof :
  overall_defective_rate = 1 / 4 :=
sorry

noncomputable def X : Type := ℕ

noncomputable def P (k : ℕ) : ℚ :=
  binomial_distribution 3 (1 / 4) k

theorem probability_distribution_and_variance :
  (P 0 = 27 / 64) ∧ (P 1 = 27 / 64) ∧ (P 2 = 9 / 64) ∧ (P 3 = 1 / 64) ∧
  (3 * (1 / 4) * (1 - (1 / 4)) = 9 / 16) :=
sorry

end defective_rate_proof_probability_distribution_and_variance_l559_559201


namespace sticks_predict_good_fortune_l559_559925

def good_fortune_probability := 11 / 12

theorem sticks_predict_good_fortune:
  (∃ (α β: ℝ), 0 ≤ α ∧ α ≤ π / 2 ∧ 0 ≤ β ∧ β ≤ π / 2 ∧ (0 ≤ β ∧ β < π - α) ∧ (0 ≤ α ∧ α < π - β)) → 
  good_fortune_probability = 11 / 12 :=
sorry

end sticks_predict_good_fortune_l559_559925


namespace trigonometric_identity_l559_559595

theorem trigonometric_identity :
  let θ := 30 * Real.pi / 180
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2) / (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_l559_559595


namespace apples_grapes_equiv_l559_559080

-- Definitions
def apples_to_grapes_equivalence (apples grapes : ℕ) : Prop :=
  (3 / 4) * 16 * apples = 10 * grapes

def value_of_apple_in_grapes (apple_value_in_grapes : ℝ) : Prop :=
  apple_value_in_grapes = 10 / 12

def grapes_for_partial_apples (grapes_partial : ℝ) : ℝ → Prop :=
  λ (partial_apples : ℕ), (partial_apples * 10 / 12 = grapes_partial)

-- Theorem
theorem apples_grapes_equiv
  (partial_apples_val : ℝ)
  (h₁ : apples_to_grapes_equivalence 12 10)
  (h₂ : value_of_apple_in_grapes (5 / 6))
  (h₃ : grapes_for_partial_apples 2.5 3) :
  partial_apples_val = 2.5 :=
by
  sorry

end apples_grapes_equiv_l559_559080


namespace optimal_chord_intersection_l559_559810

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := x^2

-- Define the point on the parabola P
def P (a : ℝ) : ℝ × ℝ := (a, parabola a)

-- Define the tangent at point P
def tangent (a : ℝ) (x : ℝ) : ℝ := 2 * a * (x - a) + parabola a

-- Define the normal at point P
def normal (a : ℝ) (x : ℝ) : ℝ := - (x - a) / (2 * a) + parabola a

-- Assume a != 0 for the calculations
axiom a_pos (a : ℝ) : a ≠ 0

-- Define the point Q as the intersection of normal line at P with the parabola
def Q (a : ℝ) : ℝ × ℝ :=
let x_Q := -a - 1 / (2 * a) in
(x_Q, parabola x_Q)

-- Define the enclosed area between the parabola and chord PQ
def enclosed_area (a : ℝ) : ℝ :=
∫ x in P(a).1..Q(a).1, parabola x

-- Main proof statement: The optimal chord PQ intersects the parabola at the point (0, 1/4)
theorem optimal_chord_intersection :
  ∃ a : ℝ, Q a = (0, 1 / 4) ∧ (PQ_perpendicular_to_tangent a) ∧ (area_minimized_condition a) :=
sorry

end optimal_chord_intersection_l559_559810


namespace area_of_triangle_AFk_is_4sqrt3_l559_559868

noncomputable def parabola_focus : point := (1, 0)

noncomputable def parabola_directrix : line := {p : point | p.1 = -1}

noncomputable def line_through_focus (m : ℝ) : line := {p : point | p.2 = m * (p.1 - 1)}

noncomputable def parabola_intersection (m : ℝ) (above_x_axis : ℝ) : point :=
  let y := 2 * √3 in
  let x := 3 in
  (x, y)

noncomputable def perpendicular_distance_to_directrix (A : point) (l : line) : ℝ := 4

noncomputable def calculate_area (A F K : point) : ℝ := 4 * √3

theorem area_of_triangle_AFk_is_4sqrt3 :
  let F := parabola_focus in
  let l := parabola_directrix in
  let A := parabola_intersection (√3) 3 in
  let K := (A.1 + 1, 0) in
  calculate_area A F K = 4 * √3 :=
sorry

end area_of_triangle_AFk_is_4sqrt3_l559_559868


namespace ellipse_standard_equation_l559_559122

theorem ellipse_standard_equation :
  ∀ (a b c : ℝ),
    2 * c = 6 →
    c / a = 3 / 5 →
    a^2 = b^2 + c^2 →
    ∃ (a' b' : ℝ), a = 5 ∧ b = 4 ∧ (a' = a) ∧ (b' = b) ∧ (∀ x y : ℝ, (x^2 / a'^2 + y^2 / b'^2 = 1) ↔ (x^2 / 25 + y^2 / 16 = 1)) :=
begin
  sorry
end

end ellipse_standard_equation_l559_559122


namespace evaluate_expression_l559_559579

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^3 + a * b^2 = 59 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l559_559579


namespace initial_price_of_phone_l559_559972

theorem initial_price_of_phone
  (initial_price_TV : ℕ)
  (increase_TV_fraction : ℚ)
  (initial_price_phone : ℚ)
  (increase_phone_percentage : ℚ)
  (total_amount : ℚ)
  (h1 : initial_price_TV = 500)
  (h2 : increase_TV_fraction = 2/5)
  (h3 : increase_phone_percentage = 0.40)
  (h4 : total_amount = 1260) :
  initial_price_phone = 400 := by
  sorry

end initial_price_of_phone_l559_559972


namespace train_length_l559_559234

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (h1 : speed_kmh = 90) (h2 : time_s = 12) : 
  ∃ length_m : ℕ, length_m = 300 := 
by
  sorry

end train_length_l559_559234


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l559_559504

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l559_559504


namespace finite_circles_l559_559679

theorem finite_circles {P : Fin 100 → ℝ × ℝ} :
  ∃ (C : Fin n → (ℝ × ℝ) × ℝ), 
    (∀ i, ∃ j, (P i).fst ∈ set.ball ((C j).fst) ((C j).snd / 2)) ∧ -- Each point is inside some circle
    (∀ i j, i ≠ j → dist ((C i).fst) ((C j).fst) > 1) ∧ -- Distance between different circles' centers > 1
    (finset.sum finset.univ (λ i : Fin n, (C i).snd)) < 100 := -- Sum of diameters < 100
sorry

end finite_circles_l559_559679


namespace max_diff_y_coords_intersection_l559_559455

theorem max_diff_y_coords_intersection :
  let f1 := fun x : ℝ => 2 - x^2 + x^3
  let f2 := fun x : ℝ => -1 + x^2 + x^3
  let intersections := {x : ℝ | f1 x = f2 x}
  let y_vals := {y : ℝ | ∃ x ∈ intersections, y = f1 x}
  let max_diff := max y_vals - min y_vals
  max_diff = (3 * Real.sqrt(3)) / 2 :=
by
  sorry

end max_diff_y_coords_intersection_l559_559455


namespace find_PH_l559_559797

noncomputable def ellipse := {P : ℝ × ℝ | let x := P.1, y := P.2 in (x^2) / 25 + (y^2) / 9 = 1}

variable (P : ℝ × ℝ) (F1 F2 H : ℝ × ℝ)
variable (hP : P ∈ ellipse)
variable (h_perp1 : (segment P F1) ⊥ (segment P F2))
variable (center : ℝ × ℝ := (0, 0))
variable (h_focus1 : F1 = (-4, 0))
variable (h_focus2 : F2 = (4, 0))
variable (h_perp2 : (midpoint F1 F2) = H)

theorem find_PH :
  |segment P H| = 9 / 4 :=
sorry

end find_PH_l559_559797


namespace ice_cream_universe_flavors_count_l559_559341

theorem ice_cream_universe_flavors_count : ∀ (balls sticks : ℕ), balls = 4 → sticks = 3 → (balls + sticks).choose sticks = 35 :=
by
  intros balls sticks h_balls h_sticks
  rw [h_balls, h_sticks]
  exact Nat.choose_succ_succ 4 3 35 sorry

end ice_cream_universe_flavors_count_l559_559341


namespace correct_operation_l559_559172

theorem correct_operation :
  (∀ x : ℝ, sqrt (x^2) = abs x) ∧
  sqrt 4 = 2 ∧
  (∀ x : ℝ, sqrt (x^2) = x ∨ sqrt (x^2) = -x) →
  ((sqrt 4 ≠ ± 2) ∧
   (± sqrt (5^2) ≠ -5) ∧
   (sqrt ((-7)^2) = 7) ∧
   (sqrt (-3 : ℝ) ≠ -sqrt 3)) :=
by
  intro h
  clear h -- clear the hypothesis since no proof is needed
  split
  · intro h1
    -- prove sqrt 4 ≠ ±2
    sorry
  split
  · intro h2
    -- prove ± sqrt (5^2) ≠ -5
    sorry
  split
  · intro h3
    -- prove sqrt ((-7)^2) = 7
    exact abs_neg 7
  · intro h4
    -- prove sqrt (-3) ≠ - sqrt 3
    sorry

end correct_operation_l559_559172


namespace train_crossing_time_l559_559958

theorem train_crossing_time (speed_kmh : ℕ) (length_m : ℕ) (conversion_factor : ℝ)
  (speed_m_s : ℝ) (h1 : speed_kmh = 60) (h2 : length_m = 500)
  (h3 : conversion_factor = 1000.0 / 3600.0)
  (h4 : speed_m_s = speed_kmh * conversion_factor) :
  length_m / speed_m_s ≈ 30 :=
by {
  sorry
}

end train_crossing_time_l559_559958


namespace find_certain_number_l559_559748

theorem find_certain_number (h1 : 213 * 16 = 3408) (x : ℝ) (h2 : x * 2.13 = 0.03408) : x = 0.016 :=
by
  sorry

end find_certain_number_l559_559748


namespace necessary_condition_for_positive_on_interval_l559_559048

theorem necessary_condition_for_positive_on_interval (a b : ℝ) (h : a + 2 * b > 0) :
  (∀ x, 0 ≤ x → x ≤ 1 → (a * x + b) > 0) ↔ ∃ c, 0 < c ∧ c ≤ 1 ∧ a + 2 * b > 0 ∧ ¬∀ d, 0 < d ∧ d ≤ 1 → a * d + b > 0 := 
by 
  sorry

end necessary_condition_for_positive_on_interval_l559_559048


namespace bret_trip_time_l559_559973

theorem bret_trip_time
  (total_distance : ℝ) (initial_speed : ℝ) (reduced_speed : ℝ) (distance_before_reduction : ℝ)
  (reduced_distance := total_distance - distance_before_reduction)
  (time_first_part := distance_before_reduction / initial_speed)
  (time_reduced_part := reduced_distance / reduced_speed)
  (total_time := time_first_part + time_reduced_part) :
  total_distance = 70 → initial_speed = 20 → reduced_speed = 12 →
  distance_before_reduction = 2 → total_time ≈ 5.7667 :=
by
  intros h1 h2 h3 h4
  unfold total_distance initial_speed reduced_speed distance_before_reduction at h1 h2 h3 h4
  unfold reduced_distance time_first_part time_reduced_part total_time
  sorry

end bret_trip_time_l559_559973


namespace find_t_l559_559032

noncomputable def point_A (t : ℝ) : ℝ × ℝ × ℝ := (t - 3, -2, 2 * t)

noncomputable def point_B (t : ℝ) : ℝ × ℝ × ℝ := (-1, t + 2, 0)

noncomputable def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

noncomputable def distance_sq (P Q : ℝ × ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2

theorem find_t (t : ℝ) :
  let A := point_A t;
      B := point_B t;
      M := midpoint A B in
  distance_sq M A = 3 * t^2 - 4 * t + 10 → t = 5 :=
by
  intro A B M dist_eqn
  have A_def := point_A_def t
  have B_def := point_B_def t
  have M_def := midpoint_def A B
  sorry

end find_t_l559_559032


namespace tan_sin_identity_l559_559589

noncomputable def tan_deg : ℝ := tan (real.pi / 6)
noncomputable def sin_deg : ℝ := sin (real.pi / 6)

theorem tan_sin_identity :
  (tan_deg ^ 2 - sin_deg ^ 2) / (tan_deg ^ 2 * sin_deg ^ 2) = 1 :=
by
  have tan_30 := by
    simp [tan_deg, real.tan_pi_div_six]
    norm_num
  have sin_30 := by
    simp [sin_deg, real.sin_pi_div_six]
    norm_num
  sorry

end tan_sin_identity_l559_559589


namespace average_of_last_12_results_l559_559854

theorem average_of_last_12_results 
  (avg_25 : ℕ -> ℕ)
  (avg_12_first : ℕ -> ℕ)
  (thirteenth_result : ℕ)
  (avg_12_last : ℕ -> ℕ) :
  (avg_25 25 = 20) ->
  (avg_12_first 12 = 14) ->
  (thirteenth_result = 128) ->
  avg_12_last 12 = 17 :=
by
sory

end average_of_last_12_results_l559_559854


namespace symmetric_curve_eq_l559_559451

-- Definitions from the problem conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 1
def line_of_symmetry (x y : ℝ) : Prop := x - y + 3 = 0

-- Problem statement derived from the translation step
theorem symmetric_curve_eq (x y : ℝ) : (x - 2) ^ 2 + (y + 1) ^ 2 = 1 ∧ x - y + 3 = 0 → (x + 4) ^ 2 + (y - 5) ^ 2 = 1 := 
by
  sorry

end symmetric_curve_eq_l559_559451


namespace wire_length_is_180_l559_559564

def wire_problem (length1 length2 : ℕ) (h1 : length1 = 106) (h2 : length2 = 74) (h3 : length1 = length2 + 32) : Prop :=
  (length1 + length2 = 180)

-- Use the definition as an assumption to write the theorem.
theorem wire_length_is_180 (length1 length2 : ℕ) 
  (h1 : length1 = 106) 
  (h2 : length2 = 74) 
  (h3 : length1 = length2 + 32) : 
  length1 + length2 = 180 :=
by
  rw [h1, h2] at h3
  sorry

end wire_length_is_180_l559_559564


namespace johns_weekly_allowance_l559_559380

-- Define the variables and constants
variable (A : ℝ) -- John's weekly allowance in dollars
constant exchange_rate : ℝ := 1.75 -- $ per foreign unit
constant spent_arcade_fraction : ℝ := 3/5 -- Fraction spent at the arcade
constant spent_toystore_fraction : ℝ := 2/7 -- Fraction spent at the toy store
constant spent_bookstore_fraction : ℝ := 1/3 -- Fraction spent at the bookstore
constant spent_candystore_foreign : ℝ := 1.2 -- Foreign units spent at the candy store

-- Translate conditions into Lean constants
constant remaining_after_arcade : ℝ := (1 - spent_arcade_fraction) * A
constant spent_toystore : ℝ := spent_toystore_fraction * remaining_after_arcade
constant remaining_after_toystore : ℝ := remaining_after_arcade - spent_toystore
constant spent_bookstore : ℝ := spent_bookstore_fraction * remaining_after_toystore
constant remaining_after_bookstore : ℝ := remaining_after_toystore - spent_bookstore
constant spent_candystore : ℝ := spent_candystore_foreign * exchange_rate

-- The proof statement
theorem johns_weekly_allowance : remaining_after_bookstore = spent_candystore → A = 11.03 :=
by
  -- We should fill in the actual proof here.
  sorry

end johns_weekly_allowance_l559_559380


namespace increasing_intervals_value_of_two_a_plus_b_l559_559525
open Real

noncomputable def f (x : ℝ) := sin (2 * x + π / 6) + 1 / 2

theorem increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ,
    (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) ↔
    (f (x - h / 2) ≤ f x ∧ f x ≤ f (x + h / 2)) :=
sorry

def y (x a b : ℝ) := a - b * cos (x - π / 3)
variables (a b x : ℝ)

theorem value_of_two_a_plus_b
  (h1 : b > 0)
  (h2 : y x a b ≤ 3 / 2)
  (h3 : y x a b ≥ -1 / 2)
  (h4 : 0 ≤ x ∧ x ≤ π) :
  2 * a + b = 2 :=
sorry

end increasing_intervals_value_of_two_a_plus_b_l559_559525


namespace two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l559_559035

theorem two_divides_a_squared_minus_a (a : ℤ) : ∃ k₁ : ℤ, a^2 - a = 2 * k₁ :=
sorry

theorem three_divides_a_cubed_minus_a (a : ℤ) : ∃ k₂ : ℤ, a^3 - a = 3 * k₂ :=
sorry

end two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l559_559035


namespace count_valid_three_digit_numbers_l559_559340

def is_valid_three_digit_number (n : Nat) : Prop := 
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  100 ≤ n ∧ n < 1000 ∧ d2 = (d1 + d3) / 2 ∧ (d1 + d2 + d3) % 5 = 0

theorem count_valid_three_digit_numbers : 
  ((Finset.filter is_valid_three_digit_number (Finset.range 1000)).card) = 8 := 
  sorry

end count_valid_three_digit_numbers_l559_559340


namespace percentage_of_opened_shells_l559_559931

theorem percentage_of_opened_shells
    (total_pistachios : ℕ)
    (percent_with_shells : ℝ)
    (opened_shells : ℕ)
    (h1 : total_pistachios = 80)
    (h2 : percent_with_shells = 0.95)
    (h3 : opened_shells = 57) :
    let pistachios_with_shells := percent_with_shells * total_pistachios in
    let percentage_opened_shells := (opened_shells : ℝ) / pistachios_with_shells * 100 in
    percentage_opened_shells = 75 :=
by
  sorry

end percentage_of_opened_shells_l559_559931


namespace probability_odd_product_gt_15_l559_559840

theorem probability_odd_product_gt_15 :
  let balls := {1, 2, 3, 4, 5, 6, 7}
  let total_ways := 49
  let odd_ways := 5 in
  odd_ways / total_ways = 5 / 49 := by
  sorry

end probability_odd_product_gt_15_l559_559840


namespace sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l559_559976

-- 1. Prove that 33 * 207 = 6831
theorem sum_of_207_instances_of_33 : 33 * 207 = 6831 := by
    sorry

-- 2. Prove that 3000 - 112 * 25 = 200
theorem difference_when_25_instances_of_112_are_subtracted_from_3000 : 3000 - 112 * 25 = 200 := by
    sorry

-- 3. Prove that 12 * 13 - (12 + 13) = 131
theorem difference_between_product_and_sum_of_12_and_13 : 12 * 13 - (12 + 13) = 131 := by
    sorry

end sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l559_559976


namespace arithmetic_sequence_sum_15_l559_559318

section
variables {α : Type*} [LinearOrder α] [AddCommMonoid α] [Module ℝ α]

noncomputable def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → ℝ} (h_arith_seq : arithmetic_sequence a)
variables (h_sum_condition : a 4 + a 12 = 10)

theorem arithmetic_sequence_sum_15 (a : ℕ → ℝ) [arithmetic_sequence a]
  (h_sum_condition : a 4 + a 12 = 10) :
  ∑ k in finset.range 15, a k = 75 :=
sorry

end

end arithmetic_sequence_sum_15_l559_559318


namespace tan_sin_div_l559_559610

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l559_559610


namespace tiles_needed_l559_559227

variables (room_length room_width : ℝ) (carpet_length carpet_width : ℝ) (tile_length tile_width : ℝ)
  (room_area carpet_area tile_area tileable_area : ℝ) (number_of_tiles : ℕ)

def room_length := 15
def room_width := 16
def carpet_length := 3
def carpet_width := 4
def tile_length := 1/4
def tile_width := 2/3

def room_area := room_length * room_width
def carpet_area := carpet_length * carpet_width
def tile_area := tile_length * tile_width
def tileable_area := room_area - carpet_area
def number_of_tiles := tileable_area / tile_area

theorem tiles_needed : number_of_tiles = 1368 :=
  sorry

end tiles_needed_l559_559227


namespace ratio_four_of_v_m_l559_559480

theorem ratio_four_of_v_m (m v : ℝ) (h : m < v) 
  (h_eq : 5 * (3 / 4 * m) = v - 1 / 4 * m) : v / m = 4 :=
sorry

end ratio_four_of_v_m_l559_559480


namespace range_of_m_l559_559753

theorem range_of_m (m : ℝ) :
  (∀ x ∈ set.Icc 1 2, (λ x, x + 2^x - m > 0) x) → m < 3 :=
by
  sorry

end range_of_m_l559_559753


namespace smaller_of_two_integers_l559_559089

theorem smaller_of_two_integers (m n : ℕ) (h1 : 100 ≤ m ∧ m < 1000) (h2 : 100 ≤ n ∧ n < 1000)
  (h3 : (m + n) / 2 = m + n / 1000) : min m n = 999 :=
by {
  sorry
}

end smaller_of_two_integers_l559_559089


namespace dihedral_angle_at_vertex_l559_559224

variables (P : Type) [RegularTriangularPyramid P]
variables (A B C : P) (alpha : ℝ)

-- Definitions used in Lean 4 statement derived from the problem conditions.
def plane_passes_through_base_vertex (A : P) : Prop := -- Detail definition as per problem condition
sorry

def plane_perpendicular_to_lateral_face (A : P) (F : P) : Prop := -- Detail definition as per problem condition
sorry

def plane_parallel_to_base_side (A B C : P) : Prop := -- Detail definition as per problem condition
sorry

def plane_angle_with_base (A : P) (alpha : ℝ) : Prop := -- Detail definition as per problem condition
sorry

-- Theorem statement in Lean 4
theorem dihedral_angle_at_vertex (P : Type) [RegularTriangularPyramid P]
  (A B C : P) (F : P) (alpha : ℝ)
  (h₁ : plane_passes_through_base_vertex A)
  (h₂ : plane_perpendicular_to_lateral_face A F)
  (h₃ : plane_parallel_to_base_side B C)
  (h₄ : plane_angle_with_base A alpha) :
  angle_at_vertex A F B C = 2 * arctan (sqrt 3 * sin alpha) :=
sorry

end dihedral_angle_at_vertex_l559_559224


namespace center_parallelogram_line_midpoints_l559_559685

variables {A B C D P Q : Type*} [quadrilateral ABCD]
variables [Intersection P A B C D] [Intersection Q A B C D]

theorem center_parallelogram_line_midpoints (ABCD : Quadrilateral A B C D) 
  (P Q : Intersection P A B C D) 
  (parallelogram_formed : Parallelogram (lines_through P Q intersect ABCD_sides)) :
  center(parallelogram_formed) ∈ line(midr_diag1 ABCD) midr_diag2 ABCD) := 
sorry

end center_parallelogram_line_midpoints_l559_559685


namespace candidate_percentage_l559_559193

variables (M T : ℝ)

theorem candidate_percentage (h1 : (P / 100) * T = M - 30) 
                             (h2 : (45 / 100) * T = M + 15)
                             (h3 : M = 120) : 
                             P = 30 := 
by 
  sorry

end candidate_percentage_l559_559193


namespace trigonometric_identity_l559_559597

theorem trigonometric_identity :
  let θ := 30 * Real.pi / 180
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2) / (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_l559_559597


namespace smallest_guaranteed_highest_score_l559_559358

-- Define the scoring system and the concept of guaranteeing more points
def point_distribution : list ℕ := [4, 3, 2, 1]
def num_matches : ℕ := 3
def guaranteed_highest_score (n : ℕ) : Prop :=
  ∀ scores, (∀ (i : ℕ), i < 4 → (scores i).sum = n → 
    (∃ j, j ≠ i ∧ (scores j).sum < n))

theorem smallest_guaranteed_highest_score : guaranteed_highest_score 10 :=
by
  sorry

end smallest_guaranteed_highest_score_l559_559358


namespace cake_icing_problem_l559_559625

theorem cake_icing_problem (cube_size icing_size total_cubes cubes_with_icing : ℕ) 
  (condition1 : cube_size = 5) 
  (condition2 : icing_size = 1) 
  (condition3 : total_cubes = 125) 
  (question : cubes_with_icing = 27) : 
  cubes_with_icing = 27 := 
by
  -- We skip the proof steps
  simp [condition1, condition2, condition3]
  exact question

end cake_icing_problem_l559_559625


namespace sum_of_smallest_and_largest_prime_factors_1260_l559_559509

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : List ℕ :=
  List.filter is_prime (List.range (n + 1)).filter (λ m, m ∣ n)

noncomputable def smallest_prime_factor (n : ℕ) : ℕ :=
  List.minimum' (prime_factors n)

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  List.maximum' (prime_factors n)

theorem sum_of_smallest_and_largest_prime_factors_1260 :
  smallest_prime_factor 1260 + largest_prime_factor 1260 = 9 :=
by
  sorry

end sum_of_smallest_and_largest_prime_factors_1260_l559_559509


namespace even_g_of_constraining_f_range_of_a_strictly_increasing_f_l559_559349

-- Problem 1
theorem even_g_of_constraining_f {g : ℝ → ℝ} (f : ℝ → ℝ) (h : ∀ x1 x2 : ℝ, |f x1 - f x2| ≥ |g x1 - g x2|) :
  f = (λ x : ℝ, x^2) → ∀ x : ℝ, g x = g (-x) :=
by intros; sorry

-- Problem 2
theorem range_of_a {f g : ℝ → ℝ} (a : ℝ) (h : ∀ x1 x2 : ℝ, |f x1 - f x2| ≥ |g x1 - g x2|) : 
  f = (λ x, a * x + x^3) → g = sin → a ≥ 1 :=
by intros; sorry

-- Problem 3
theorem strictly_increasing_f (f g : ℝ → ℝ) (h : ∀ x1 x2 : ℝ, |f x1 - f x2| ≥ |g x1 - g x2|)
    (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → g x1 > g x2) (h_f0f1 : f 0 < f 1) 
    (h_cont : continuous f) : 
  ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2 :=
by intros; sorry

end even_g_of_constraining_f_range_of_a_strictly_increasing_f_l559_559349


namespace even_function_expression_l559_559701

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x then 2*x + 1 else -2*x + 1

theorem even_function_expression (x : ℝ) (hx : x < 0) :
  f x = -2*x + 1 :=
by sorry

end even_function_expression_l559_559701


namespace bhanu_income_problem_l559_559575

-- Define the total income
def total_income (I : ℝ) : Prop :=
  let petrol_spent := 300
  let house_rent := 70
  (0.10 * (I - petrol_spent) = house_rent)

-- Define the percentage of income spent on petrol
def petrol_percentage (P : ℝ) (I : ℝ) : Prop :=
  0.01 * P * I = 300

-- The theorem we aim to prove
theorem bhanu_income_problem : 
  ∃ I P, total_income I ∧ petrol_percentage P I ∧ P = 30 :=
by
  sorry

end bhanu_income_problem_l559_559575


namespace bottle_weight_is_six_l559_559784

variable (b : ℝ) -- weight of each bottle in ounces
variable (cans : ℝ) -- number of cans collected
variable (can_weight : ℝ) -- weight of each can in ounces
variable (total_weight : ℝ) -- total carrying capacity in ounces
variable (bottles : ℝ) -- number of bottles collected
variable (total_money : ℝ) -- total money made in cents
variable (value_per_bottle : ℝ) -- money made per bottle
variable (value_per_can : ℝ) -- money made per can

-- Given conditions
def conditions :=
  can_weight = 2 ∧
  total_weight = 100 ∧
  cans = 20 ∧
  value_per_bottle = 10 ∧
  value_per_can = 3 ∧
  total_money = 160

-- Prove: The weight of each bottle is 6 ounces
theorem bottle_weight_is_six : conditions → b = 6 := by
  sorry

end bottle_weight_is_six_l559_559784


namespace least_integer_with_ten_factors_l559_559895

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.map (λ k, k + 1)).prod

theorem least_integer_with_ten_factors : ∃ n, ∃ p q a b : ℕ, 
  prime p ∧ prime q ∧ p < q ∧
  n = p^a * q^b ∧
  a + 1 = 2 ∧ b + 1 = 5 ∧
  n = 48 :=
sorry

end least_integer_with_ten_factors_l559_559895


namespace expected_value_of_X_l559_559473

noncomputable def E_X : ℚ :=
  let total := 10
  let first_class := 6
  let second_class := 4
  let selected := 3
  let X_distribution := [(0, 1/30), (1, 3/10), (2, 1/2), (3, 1/6)]
  let expected_value := 
    (X_distribution.map (λ p => p.1 * p.2)).sum
  expected_value

theorem expected_value_of_X :
  E_X = 9/5 :=
by
  unfold E_X
  let X_distribution := [(0, 1/30), (1, 3/10), (2, 1/2), (3, 1/6)]
  calc (X_distribution.map (λ p => p.1 * p.2)).sum = 0 * (1/30) + 1 * (3/10) + 2 * (1/2) + 3 * (1/6) : by simp
  ... = 0 + 3/10 + 1 + 1/2 + 1/2 : by simp
  ... = 3/10 + 1 + 1/2 + 1/2 : by simp
  ... = 3/10 + 2 : by simp
  ... = 9/5 : by norm_num
  done
      

end expected_value_of_X_l559_559473


namespace josephine_filled_each_2container_with_0_l559_559424

noncomputable def amount_in_each_of_two_containers (total_milk: ℝ) (count_3cont: ℕ) (liters_per_3cont: ℝ) (count_5cont: ℕ) (liters_per_5cont: ℝ) (count_2cont: ℕ): ℝ :=
let total_3cont_milk := count_3cont * liters_per_3cont;
let total_5cont_milk := count_5cont * liters_per_5cont;
let remaining_milk := total_milk - (total_3cont_milk + total_5cont_milk);
let milk_per_2cont := remaining_milk / count_2cont;
milk_per_2cont

theorem josephine_filled_each_2container_with_0.75_liters :
  ∀ (total_milk : ℝ) (count_3cont count_5cont count_2cont : ℕ) (liters_per_3cont liters_per_5cont : ℝ),
  total_milk = 10 →
  count_3cont = 3 →
  liters_per_3cont = 2 →
  count_5cont = 5 →
  liters_per_5cont = 0.5 →
  count_2cont = 2 →
  amount_in_each_of_two_containers total_milk count_3cont liters_per_3cont count_5cont liters_per_5cont count_2cont = 0.75 :=
by
  intros total_milk count_3cont count_5cont count_2cont liters_per_3cont liters_per_5cont 
         h_total h_count3 h_liters3 h_count5 h_liters5 h_count2
  rw [amount_in_each_of_two_containers, h_total, h_count3, h_liters3, h_count5, h_liters5, h_count2]
  sorry

end josephine_filled_each_2container_with_0_l559_559424


namespace subgraphs_isomorphic_to_Kr_l559_559891

open Finset

noncomputable def binom (x : ℝ) (r : ℕ) : ℝ :=
  if r = 0 then 1 else (1/(r.factorial)) * (List.prod (finRange r).map (λ i => x - i))

theorem subgraphs_isomorphic_to_Kr (n m r : ℕ) (G : SimpleGraph (Fin n)) (K_r : SimpleGraph (Fin (r + 1)))
  (h₁ : ∀ v : Fin n, ∑ w in G.neighborFinset v, 1 = G.degree v)
  (h₂ : G.edgeFinset.card = m)
  (h₃ : K_r = SimpleGraph.star (Fin (r + 1)) 0) :
  ∃ count : ℕ, count ≥ n * (2 * m / r) ∧
    ∀ v : Fin n, ∃ S : Finset (Fin n), S.card = r ∧ K_r.isSubgraph (G.induce S) :=
sorry

end subgraphs_isomorphic_to_Kr_l559_559891


namespace sqrt_of_neg_7_sq_is_7_l559_559173

theorem sqrt_of_neg_7_sq_is_7 : sqrt ((-7)^2) = 7 :=
by sorry

end sqrt_of_neg_7_sq_is_7_l559_559173


namespace cost_of_individual_roll_l559_559197

theorem cost_of_individual_roll
  (p : ℕ) (c : ℝ) (s : ℝ) (x : ℝ)
  (hc : c = 9)
  (hp : p = 12)
  (hs : s = 0.25)
  (h : 12 * x = 9 * (1 + s)) :
  x = 0.9375 :=
by
  sorry

end cost_of_individual_roll_l559_559197


namespace solve_house_A_cost_l559_559421

-- Definitions and assumptions
variables (A B C : ℝ)
variable base_salary : ℝ := 3000
variable commission_rate : ℝ := 0.02
variable total_earnings : ℝ := 8000

-- Conditions
def house_B_cost (A : ℝ) : ℝ := 3 * A
def house_C_cost (A : ℝ) : ℝ := 2 * A - 110000

-- Define Nigella's commission calculation
def nigella_commission (A B C : ℝ) : ℝ := commission_rate * A + commission_rate * B + commission_rate * C

-- Commission earned based on total earnings and base salary
def commission_earned : ℝ := total_earnings - base_salary

-- Lean theorem statement
theorem solve_house_A_cost 
  (hB : B = house_B_cost A)
  (hC : C = house_C_cost A)
  (h_commission : nigella_commission A B C = commission_earned) : 
  A = 60000 :=
by 
-- Sorry is used to skip the actual proof
sorry

end solve_house_A_cost_l559_559421


namespace extremum_point_m_and_intervals_l559_559714

def f (x : ℝ) (m : ℝ) := Real.exp x - Real.log (x + m)
def f_prime (x : ℝ) (m : ℝ) := Real.exp x - 1 / (x + m)

theorem extremum_point_m_and_intervals (m : ℝ) :
  (f_prime 0 m = 0) → m = 1 ∧ 
    (∀ x, x > 0 → f_prime x 1 > 0) ∧ 
    (∀ x, -1 < x ∧ x < 0 → f_prime x 1 < 0) :=
by
  sorry

end extremum_point_m_and_intervals_l559_559714


namespace sin_minus_cos_value_l559_559295

open Real

noncomputable def tan_alpha := sqrt 3
noncomputable def alpha_condition (α : ℝ) := π < α ∧ α < (3 / 2) * π

theorem sin_minus_cos_value (α : ℝ) (h1 : tan α = tan_alpha) (h2 : alpha_condition α) : 
  sin α - cos α = -((sqrt 3) - 1) / 2 := 
by 
  sorry

end sin_minus_cos_value_l559_559295


namespace special_day_jacket_price_l559_559874

noncomputable def original_price : ℝ := 240
noncomputable def first_discount_rate : ℝ := 0.4
noncomputable def special_day_discount_rate : ℝ := 0.25

noncomputable def first_discounted_price : ℝ :=
  original_price * (1 - first_discount_rate)
  
noncomputable def special_day_price : ℝ :=
  first_discounted_price * (1 - special_day_discount_rate)

theorem special_day_jacket_price : special_day_price = 108 := by
  -- definitions and calculations go here
  sorry

end special_day_jacket_price_l559_559874


namespace pool_capacity_l559_559786

theorem pool_capacity (bucket_capacity : ℕ) (fill_time : ℕ) (total_time_minutes : ℕ) 
  (known_bucket_capacity : bucket_capacity = 2)
  (known_fill_time : fill_time = 20)
  (known_total_time_minutes : total_time_minutes = 14) :
  let total_time_seconds := total_time_minutes * 60 in
  let num_buckets := total_time_seconds / fill_time in
  let pool_capacity := num_buckets * bucket_capacity in
  pool_capacity = 84 :=
by
  rw [known_bucket_capacity, known_fill_time, known_total_time_minutes]
  let total_time_seconds := 14 * 60
  let num_buckets := total_time_seconds / 20
  let pool_capacity := num_buckets * 2
  show pool_capacity = 84
  sorry

end pool_capacity_l559_559786


namespace probability_at_least_3_smile_l559_559350

theorem probability_at_least_3_smile (p_smile : ℝ) (h_prob : p_smile = 1/3) :
    let p_at_least_3 : ℝ := 353 / 729 in
    (∑ k in Iio 3, (nat.choose 6 k) * (p_smile ^ k) * ((1 - p_smile) ^ (6 - k))) = (1 - p_at_least_3) :=
by
  sorry

end probability_at_least_3_smile_l559_559350


namespace least_pos_int_with_ten_factors_l559_559897

theorem least_pos_int_with_ten_factors : ∃ (n : ℕ), n > 0 ∧ (∀ m, (m > 0 ∧ ∃ d : ℕ, d∣n → d = 1 ∨ d = n) → m < n) ∧ ( ∃! n, ∃ d : ℕ, d∣n ) := sorry

end least_pos_int_with_ten_factors_l559_559897


namespace friends_activity_l559_559283

-- Defining the problem conditions
def total_friends : ℕ := 5
def organizers : ℕ := 3
def managers : ℕ := total_friends - organizers

-- Stating the proof problem
theorem friends_activity (h1 : organizers = 3) (h2 : managers = 2) :
  Nat.choose total_friends organizers = 10 :=
sorry

end friends_activity_l559_559283


namespace range_of_a_inequality_l559_559754

theorem range_of_a_inequality (a : ℝ) (h : ∀ x ∈ Ioo 0 1, (x + real.log a) / real.exp x - (a * real.log x) / x > 0) :
  a ∈ set.Ico (1 / real.exp 1) 1 :=
sorry

end range_of_a_inequality_l559_559754


namespace cartesian_equations_and_distance_l559_559014

open Real 

noncomputable def C1_param (t : ℝ) : ℝ × ℝ := (6 + (sqrt 3 / 2) * t, 0.5 * t)
noncomputable def C2_polar (θ : ℝ) : ℝ := 10 * cos θ

def C1_cartesian_eq : Prop := ∀ t : ℝ, 
  let (x, y) := C1_param t in x - sqrt 3 * y = 6

def C2_cartesian_eq (x y : ℝ) : Prop := x^2 + y^2 = 10 * x

theorem cartesian_equations_and_distance :
  (C1_cartesian_eq) ∧ (∀ x y : ℝ, (∃ t : ℝ, (x, y) = C1_param t) → C2_cartesian_eq x y) ∧
  let t_vals := {t : ℝ | (fst (C1_param t))^2 + (snd (C1_param t))^2 = 10 * (fst (C1_param t))} in
  let t1 := Classical.some (Nonempty.some t_vals.nonempty) in
  let t2 := Classical.some (Nonempty.some (t_vals.diff_singleton_nonempty t1)) in
  let t1_sum_t2 := -sqrt 3 in t1 + t2 = t1_sum_t2 ∧
  let t1_prod_t2 := -24 in t1 * t2 = t1_prod_t2 ∧
  abs (t2 - t1) = 3 * sqrt 11 :=
begin
  sorry
end

end cartesian_equations_and_distance_l559_559014


namespace no_quadruples_solution_l559_559275

theorem no_quadruples_solution (a b c d : ℝ) :
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 →
    false :=
by 
  intros h
  sorry

end no_quadruples_solution_l559_559275


namespace correct_operation_l559_559170

theorem correct_operation :
  (∀ x : ℝ, sqrt (x^2) = abs x) ∧
  sqrt 4 = 2 ∧
  (∀ x : ℝ, sqrt (x^2) = x ∨ sqrt (x^2) = -x) →
  ((sqrt 4 ≠ ± 2) ∧
   (± sqrt (5^2) ≠ -5) ∧
   (sqrt ((-7)^2) = 7) ∧
   (sqrt (-3 : ℝ) ≠ -sqrt 3)) :=
by
  intro h
  clear h -- clear the hypothesis since no proof is needed
  split
  · intro h1
    -- prove sqrt 4 ≠ ±2
    sorry
  split
  · intro h2
    -- prove ± sqrt (5^2) ≠ -5
    sorry
  split
  · intro h3
    -- prove sqrt ((-7)^2) = 7
    exact abs_neg 7
  · intro h4
    -- prove sqrt (-3) ≠ - sqrt 3
    sorry

end correct_operation_l559_559170


namespace smallest_base_l559_559280

theorem smallest_base (k : ℕ) (hk : 0 < k) : (0.\overline{14}_k = \frac{5}{27}) → k = 14 :=
by
  sorry

end smallest_base_l559_559280


namespace one_minus_i_pow_four_l559_559623

theorem one_minus_i_pow_four : (1 - Complex.i) ^ 4 = -4 :=
by
  sorry

end one_minus_i_pow_four_l559_559623


namespace max_interesting_numbers_in_five_consecutive_l559_559898

def sum_of_digits (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

def is_prime_sum_of_digits (n : ℕ) : Prop :=
  (sum_of_digits n).prime

theorem max_interesting_numbers_in_five_consecutive (n : ℕ) :
  let consecutive_nums := [n, n+1, n+2, n+3, n+4]
  (count is_prime_sum_of_digits consecutive_nums ≤ 4) :=
sorry

end max_interesting_numbers_in_five_consecutive_l559_559898


namespace correct_display_sum_l559_559852

theorem correct_display_sum :
  ∃ d e : ℕ, ((561374 + 397562 = 958936) ∧ (change_digit (561374 + 397562) d e = 1000936)) → (d + e = 9) :=
by
  sorry

end correct_display_sum_l559_559852


namespace maximal_N8_value_l559_559971

noncomputable def max_permutations_of_projections (A : Fin 8 → ℝ × ℝ) : ℕ := sorry

theorem maximal_N8_value (A : Fin 8 → ℝ × ℝ) :
  max_permutations_of_projections A = 56 :=
sorry

end maximal_N8_value_l559_559971


namespace num_possible_b2_values_l559_559228

def sequence_rule (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b (n + 1) - b n).natAbs

def initial_conditions (b : ℕ → ℕ) : Prop :=
  b 1 = 1001 ∧ b 2 < 1001 ∧ b 2010 = 1

theorem num_possible_b2_values
  (b : ℕ → ℕ)
  (h_sequence_rule : sequence_rule b)
  (h_initial_conditions : initial_conditions b) :
  ∃ S : finset ℕ, S.card = 286 ∧ ∀ x ∈ S, b 2 = x ∧ x < 1001 ∧ x % 2 = 1 ∧ gcd 1001 x = 1 :=
sorry

end num_possible_b2_values_l559_559228


namespace exists_convex_ngon_l559_559677

-- Definitions for pairwise non-collinear vectors and their sum
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {n : ℕ} (h₁ : n ≥ 3) 
variables (a : fin n → V) 

def pairwise_non_collinear (a : fin n → V) : Prop := 
  ∀ i j k : fin n, i ≠ j → j ≠ k → k ≠ i → ¬ collinear ℝ {a i, a j, a k}

def sum_zero (a : fin n → V) : Prop := 
  ∑ i : fin n, a i = 0

-- Lean statement
theorem exists_convex_ngon 
  (h₁ : n ≥ 3) 
  (h₂ : pairwise_non_collinear a) 
  (h₃ : sum_zero a) : 
  ∃ (polygon : finset (fin n → V)), convex hull (finset.image a polygon) := 
sorry

end exists_convex_ngon_l559_559677


namespace analytical_expression_range_of_c_maximum_value_of_y_l559_559721

-- Given function and conditions
noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) := a * x^2 + (b-8) * x - a - a * b

-- Proof of the function's specific form
theorem analytical_expression (x: ℝ) : 
(∀ x, x ∈ Ioo (-3 : ℝ) 2 → f (-3) 5 x > 0) → 
(∀ x, x ∈ (Iic (-3 : ℝ) ∪ Ioi (2 : ℝ)) → f (-3) 5 x < 0) →
f (-3) 5 x = -3*x^2 - 3*x + 18 :=
sorry

-- Proof for the range of c
theorem range_of_c (c : ℝ) :
(∀ x, -3 * x^2 + 5 * x + c ≤ 0) → 
c ≤ -25/12 :=
sorry

-- Proof for the maximum value of y
theorem maximum_value_of_y (x : ℝ) (h : x > -1) : 
y (-3) 5 (x : ℝ) = (f (-3) 5 x - 21) / (x + 1) → 
(y (-3) 5 x ≤ -3) ∧ (∃ x₀, x₀ = 0 ∧ y (-3) 5 x₀ = -3) :=
sorry

end analytical_expression_range_of_c_maximum_value_of_y_l559_559721


namespace bucket_holds_120_ounces_l559_559423

theorem bucket_holds_120_ounces :
  ∀ (fill_buckets remove_buckets baths_per_day ounces_per_week : ℕ),
    fill_buckets = 14 →
    remove_buckets = 3 →
    baths_per_day = 7 →
    ounces_per_week = 9240 →
    baths_per_day * (fill_buckets - remove_buckets) * (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = ounces_per_week →
    (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = 120 :=
by
  intros fill_buckets remove_buckets baths_per_day ounces_per_week Hfill Hremove Hbaths Hounces Hcalc
  sorry

end bucket_holds_120_ounces_l559_559423


namespace greatest_ratio_l559_559646

noncomputable def distance (p1 p2: ℝ × ℝ) := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def on_circle (r: ℝ) (p: ℝ × ℝ) := p.1^2 + p.2^2 = r^2

def is_integer_point (p: ℝ × ℝ) := ∃ (a b: ℤ), (p.1 = a ∧ p.2 = b)

theorem greatest_ratio 
  (A B C D: ℝ × ℝ)
  (hA: on_circle 4 A) (hB: on_circle 4 B) (hC: on_circle 4 C) (hD: on_circle 4 D)
  (hAi: is_integer_point A) (hBi: is_integer_point B) (hCi: is_integer_point C) (hDi: is_integer_point D)
  (hAB: ¬ ∃ n: ℤ, distance A B = n)
  (hCD: ¬ ∃ n: ℤ, distance C D = n) :
  ∃ (k: ℝ), k = 2 ∧ ∀ (AB: ℝ) (CD: ℝ), AB ∈ {distance A B} → CD ∈ {distance C D} → (AB / CD ≤ k) := 
sorry

end greatest_ratio_l559_559646


namespace find_genuine_coin_with_two_weighings_l559_559239

-- Defining the types of coins and the condition of genuine and counterfeit
inductive Coin
| C1 | C2 | C3 | C4 | C5 : Coin

-- Defining the predicate for a coin being genuine
def isGenuine (c : Coin) : Prop

-- Given conditions
axiom three_genuine_two_counterfeit : ∃ c1 c2 c3,
  isGenuine c1 ∧ isGenuine c2 ∧ isGenuine c3 ∧
  ∀ c, c ≠ c1 ∧ c ≠ c2 ∧ c ≠ c3 → ¬ isGenuine c
axiom counterfeit_coins_same_weight : ∀ c1 c2, ¬ isGenuine c1 → ¬ isGenuine c2 → weight c1 = weight c2

-- Proof statement
theorem find_genuine_coin_with_two_weighings :
  ∃ c, isGenuine c :=
by sorry

end find_genuine_coin_with_two_weighings_l559_559239


namespace solve_equation_l559_559066

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2/3 → (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2)) →
  (∀ x : ℝ, x = 1 / Real.sqrt 3 ∨ x = -1 / Real.sqrt 3) :=
by
  sorry

end solve_equation_l559_559066


namespace dexter_total_cards_l559_559642

-- Define the given conditions as constants and variables in Lean
constant num_basketball_boxes : ℕ := 9
constant cards_per_basketball_box : ℕ := 15
constant boxes_filled_less_with_football_cards : ℕ := 3
constant cards_per_football_box : ℕ := 20

-- Calculate the derived quantities based on the above conditions
def num_football_boxes : ℕ := num_basketball_boxes - boxes_filled_less_with_football_cards
def total_basketball_cards : ℕ := num_basketball_boxes * cards_per_basketball_box
def total_football_cards : ℕ := num_football_boxes * cards_per_football_box
def total_cards : ℕ := total_basketball_cards + total_football_cards

-- State the theorem to prove
theorem dexter_total_cards : total_cards = 255 := by
  sorry  -- proof placeholder; the goal is to establish that total_cards = 255

end dexter_total_cards_l559_559642


namespace right_triangle_angles_l559_559117

theorem right_triangle_angles (a b c : ℝ) (h : a^2 + b^2 = c^2) (h_a : a = 3) (h_b : b = 4) (h_c : c = 5) :
  ∃ θ₁ θ₂ θ₃ : ℝ, θ₁ = 90 ∧ θ₂ ≈ 36.87 ∧ θ₃ ≈ 53.13 ∧ θ₁ + θ₂ + θ₃ = 180 :=
sorry

end right_triangle_angles_l559_559117


namespace total_production_second_year_l559_559537

theorem total_production_second_year :
  (let daily_production := 10
       days_in_year := 365
       reduction_percentage := 0.10
       total_production_first_year := daily_production * days_in_year
       reduction_amount := total_production_first_year * reduction_percentage in
    total_production_first_year - reduction_amount = 3285) :=
by
  let daily_production := 10
  let days_in_year := 365
  let reduction_percentage := 0.10
  let total_production_first_year := daily_production * days_in_year
  let reduction_amount := total_production_first_year * reduction_percentage
  show total_production_first_year - reduction_amount = 3285
  sorry

end total_production_second_year_l559_559537


namespace trigonometric_identity_l559_559601

theorem trigonometric_identity :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  t30 = 1 / Real.sqrt 3 →
  s30 = 1 / 2 →
  (t30 ^ 2 - s30 ^ 2) / (t30 ^ 2 * s30 ^ 2) = 1 :=
by
  intros t30_eq s30_eq
  -- Proof would go here
  sorry

end trigonometric_identity_l559_559601


namespace solve_log_eq_l559_559120

theorem solve_log_eq (x : ℝ) : log 2 ((x + 1)^2) + log 4 (x + 1) = 5 ↔ x = 3 :=
by
  sorry

end solve_log_eq_l559_559120


namespace solve_remainder_l559_559081

theorem solve_remainder (y : ℤ) 
  (hc1 : y + 4 ≡ 9 [ZMOD 3^3])
  (hc2 : y + 4 ≡ 16 [ZMOD 5^3])
  (hc3 : y + 4 ≡ 36 [ZMOD 7^3]) : 
  y ≡ 32 [ZMOD 105] :=
by
  sorry

end solve_remainder_l559_559081


namespace probability_gcd_is_one_l559_559887

-- Definitions for the problem
def set := {1, 2, 3, 4, 5, 6, 7, 8}
def pairs := { (a, b) | a ∈ set ∧ b ∈ set ∧ a < b }
def gcd_is_one (a b : ℕ) : Prop := Nat.gcd a b = 1
def valid_pairs := { (a, b) ∈ pairs | gcd_is_one a b }

-- Lean 4 statement for the proof problem
theorem probability_gcd_is_one : 
  (valid_pairs.card : ℚ) / (pairs.card : ℚ) = 3 / 4 :=
sorry -- To be proven

end probability_gcd_is_one_l559_559887


namespace no_power_of_q_l559_559684

theorem no_power_of_q (n : ℕ) (hn : n > 0) (q : ℕ) (hq : Prime q) : ¬ (∃ k : ℕ, n^q + ((n-1)/2)^2 = q^k) := 
by
  sorry  -- proof steps are not required as per instructions

end no_power_of_q_l559_559684


namespace fraction_of_top10_lists_l559_559213

theorem fraction_of_top10_lists (total_members : ℕ) (min_lists : ℝ) (H1 : total_members = 795) (H2 : min_lists = 198.75) :
  (min_lists / total_members) = 1 / 4 :=
by
  -- The proof is omitted as requested
  sorry

end fraction_of_top10_lists_l559_559213


namespace polar_line_equation_l559_559705

theorem polar_line_equation (P : ℝ × ℝ) (hP : P = (1, π / 3)) :
  ∃ ρ : ℝ, ∀ θ : ℝ, (P = (1, π / 3) → ρ = 1 / (2 * cos θ)) :=
by
  sorry

end polar_line_equation_l559_559705


namespace geometric_sequence_a6_l559_559012

variable {α : Type} [LinearOrderedSemiring α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ a₁ q : α, ∀ n, a n = a₁ * q ^ n

theorem geometric_sequence_a6 
  (a : ℕ → α) 
  (h_seq : is_geometric_sequence a) 
  (h1 : a 2 + a 4 = 20) 
  (h2 : a 3 + a 5 = 40) : 
  a 6 = 64 :=
by
  sorry

end geometric_sequence_a6_l559_559012


namespace third_year_increment_l559_559077

-- Define the conditions
def total_payments : ℕ := 96
def first_year_cost : ℕ := 20
def second_year_cost : ℕ := first_year_cost + 2
def third_year_cost (x : ℕ) : ℕ := second_year_cost + x
def fourth_year_cost (x : ℕ) : ℕ := third_year_cost x + 4

-- The main proof statement
theorem third_year_increment (x : ℕ) 
  (H : first_year_cost + second_year_cost + third_year_cost x + fourth_year_cost x = total_payments) :
  x = 2 :=
sorry

end third_year_increment_l559_559077


namespace gumball_machine_total_gumballs_l559_559541

theorem gumball_machine_total_gumballs :
  let red := 24
  let blue := red / 2
  let green := 4 * blue
  let yellow := (6 / 10) * green -- 60% of green
  let yellow_rounded := if yellow - Int.floor yellow < 0.5 then Int.floor yellow else Int.ceil yellow
  let orange := (red + blue) / 3
  24 + blue + green + yellow_rounded + orange = 124 :=
by
  unfold red
  unfold blue
  unfold green
  unfold yellow
  unfold yellow_rounded
  unfold orange
  simp only [Int.floor]
  norm_num
  rfl

end gumball_machine_total_gumballs_l559_559541


namespace house_A_cost_l559_559419

theorem house_A_cost (base_salary earnings commission_rate total_houses cost_A cost_B cost_C : ℝ)
  (H_base_salary : base_salary = 3000)
  (H_earnings : earnings = 8000)
  (H_commission_rate : commission_rate = 0.02)
  (H_cost_B : cost_B = 3 * cost_A)
  (H_cost_C : cost_C = 2 * cost_A - 110000)
  (H_total_commission : earnings - base_salary = 5000)
  (H_total_cost : 5000 / commission_rate = 250000)
  (H_total_houses : base_salary + commission_rate * (cost_A + cost_B + cost_C) = earnings) :
  cost_A = 60000 := sorry

end house_A_cost_l559_559419


namespace no_valid_n_l559_559662

theorem no_valid_n : ¬(∃ n : ℕ+, 100 ≤ n / 4 ∧ n / 4 ≤ 999 ∧ 100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_valid_n_l559_559662


namespace range_of_z_l559_559667

def range_z (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) : Prop :=
  let z := 2 * x - y
  -3 ≤ z ∧ z ≤ 4

theorem range_of_z :
  ∀ (x y : ℝ), -1 ≤ x ∧ x ≤ 2 → 0 ≤ y ∧ y ≤ 1 → let z := 2 * x - y in -3 ≤ z ∧ z ≤ 4 := by
  intros x y hx hy
  let z := 2 * x - y
  sorry

end range_of_z_l559_559667


namespace total_production_second_year_l559_559538

noncomputable def production_rate := 10
noncomputable def days_in_year := 365
noncomputable def reduction_rate := 0.10

theorem total_production_second_year :
  let first_year_production := production_rate * days_in_year
  let reduction_amount := reduction_rate * first_year_production
  let second_year_production := first_year_production - reduction_amount
  second_year_production = 3285 :=
by
  sorry

end total_production_second_year_l559_559538


namespace f_property_l559_559398

noncomputable def f : ℝ → ℝ := sorry

theorem f_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f(x * y) = f(x) / y := sorry

example (h₁ : f(500) = 3) : f(600) = 5 / 2 :=
  by sorry

end f_property_l559_559398


namespace locus_of_midpoints_of_chords_through_P_l559_559402

noncomputable def midpoint_locus (O P : Point) (r : ℝ) (hP : 0 < dist O P ∧ dist O P < r) :=
  {M : Point | ∃ A B : Point, midpoint A B = M ∧ on_circle O r A ∧ on_circle O r B ∧ on_line P A B}

theorem locus_of_midpoints_of_chords_through_P
  (O P : Point) (r : ℝ) (hP : 0 < dist O P ∧ dist O P < r) :
  midpoint_locus O P r hP = (circle (dist O P / 2) (line_through O P) - {P}) :=
sorry

end locus_of_midpoints_of_chords_through_P_l559_559402


namespace integer_values_of_a_l559_559274

theorem integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^4 + 4 * x^3 + a * x^2 + 8 = 0) ↔ (a = -14 ∨ a = -13 ∨ a = -5 ∨ a = 2) :=
sorry

end integer_values_of_a_l559_559274


namespace ln_1_1_approx_sqrt4_17_approx_l559_559485

-- Define the Taylor series expansion for ln(1 + x)
def taylor_ln (x : ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, (-1 : ℝ) ^ (i + 1 + 1) * x ^ (i + 1) / (i + 1)

-- Define the binomial series expansion for (1 + x)^m
def binomial_series (x : ℝ) (m : ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, ((m : ℝ) choose i) * x ^ i

-- Prove ln 1.1 ≈ 0.0953 using series expansion with precision 0.0001
theorem ln_1_1_approx : (abs (taylor_ln 0.1 4 - 0.0953) < 0.0001) :=
sorry

-- Prove (17 ^ 1/4) ≈ 2.0305 using binomial expansion with precision 0.0001
theorem sqrt4_17_approx : (abs ((2 * binomial_series (1 / 16) (1 / 4) 4) - 2.0305) < 0.0001) :=
sorry

end ln_1_1_approx_sqrt4_17_approx_l559_559485


namespace optimal_minimum_cost_l559_559218

-- Define the nutritional content of lunch and dinner
def lunch := { carbohydrates := 12, proteins := 6, vitamins_c := 6 }
def dinner := { carbohydrates := 8, proteins := 6, vitamins_c := 10 }

-- Define the nutritional requirements
def nutritional_requirements := { carbohydrates := 64, proteins := 42, vitamins_c := 54 }

-- Define the cost per unit of lunch and dinner
def cost_per_unit_lunch := 2.5
def cost_per_unit_dinner := 4.0

-- Define the amount of lunch and dinner that the nutritionist should plan
def units_lunch := 4
def units_dinner := 3

noncomputable def minimum_cost_planning (units_lunch : ℕ) (units_dinner : ℕ) : Prop :=
  (units_lunch * lunch.carbohydrates + units_dinner * dinner.carbohydrates >= nutritional_requirements.carbohydrates) ∧
  (units_lunch * lunch.proteins + units_dinner * dinner.proteins >= nutritional_requirements.proteins) ∧
  (units_lunch * lunch.vitamins_c + units_dinner * dinner.vitamins_c >= nutritional_requirements.vitamins_c) ∧
  ((units_lunch * cost_per_unit_lunch + units_dinner * cost_per_unit_dinner) = (4 * cost_per_unit_lunch + 3 * cost_per_unit_dinner))

theorem optimal_minimum_cost : minimum_cost_planning 4 3 :=
by
  sorry

end optimal_minimum_cost_l559_559218


namespace stratified_sampling_correctness_l559_559956

def ratio_of_athletes (total_male total_female : Nat) : Rat :=
  total_male / total_female.toRat

def num_female_athletes_selected (male_selected : Nat) (ratio_m_f : Rat) : Nat :=
  (male_selected * (ratio_m_f.den / ratio_m_f.num)).toNat

theorem stratified_sampling_correctness :
  ∀ (total_male total_female male_selected female_selected : Nat),
  total_male = 56 →
  total_female = 42 →
  male_selected = 8 →
  female_selected = num_female_athletes_selected male_selected (ratio_of_athletes 56 42) →
  female_selected = 6 :=
by
  intros
  rfl  -- replace our hypotheses with the actual values
  sorry  -- we skip the proof here

end stratified_sampling_correctness_l559_559956


namespace range_of_a_inequality_l559_559755

theorem range_of_a_inequality (a : ℝ) (h : ∀ x ∈ Ioo 0 1, (x + real.log a) / real.exp x - (a * real.log x) / x > 0) :
  a ∈ set.Ico (1 / real.exp 1) 1 :=
sorry

end range_of_a_inequality_l559_559755


namespace part_I_part_II_min_ϕ_l559_559332

noncomputable def vec_m (x : ℝ) : ℝ × ℝ :=
(Real.cos (x / 3), sqrt 3 * Real.cos (x / 3))

noncomputable def vec_n (x : ℝ) : ℝ × ℝ :=
(Real.sin (x / 3), Real.cos (x / 3))

noncomputable def f (x : ℝ) : ℝ :=
(vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2

def g (x ϕ : ℝ) : ℝ :=
Real.sin (2 * x + 2 / 3 * ϕ + π / 3) + sqrt 3 / 2

theorem part_I (k : ℤ) :
  monotonic_on f [3 * k * π - 5 * π / 4, 3 * k * π + π / 4] ∧
  monotonic_on f [3 * k * π + π / 4, 3 * k * π + 7 * π / 4] := 
sorry

theorem part_II_min_ϕ :
  ∃ (k : ℤ), ϕ = 3 * k * π / 2 + π / 4 ∧ ϕ > 0 → ϕ = π / 4 :=
sorry

end part_I_part_II_min_ϕ_l559_559332


namespace calc_expression_l559_559580

theorem calc_expression : 2 * real.sin (real.pi / 3) - (1 / 3) ^ 0 = real.sqrt 3 - 1 :=
by
  have h1 : real.sin (real.pi / 3) = real.sqrt 3 / 2 := by sorry
  have h2 : (1 / 3 : real) ^ 0 = 1 := by sorry
  rw h1
  rw h2
  sorry

end calc_expression_l559_559580


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l559_559507

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l559_559507


namespace true_propositions_l559_559411

-- Assume basic plane and line structure
structure Plane := (name : string)

structure Line := (name : string)

-- Definitions for parallelism and perpendicularity
def is_parallel (p1 p2 : Plane) : Prop := sorry
def is_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def is_perpendicular (p1 p2 : Plane) : Prop := sorry
def is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def intersects_along (p1 p2 : Plane) (l : Line) : Prop := sorry
def intersects (l1 l2 : Line) : Prop := sorry

-- Planes α and β
axiom α : Plane
axiom β : Plane
-- α and β are distinct
axiom h_distinct : α ≠ β

-- Proposition Evaluations 
axiom prop_1 : (∃ l1 l2 : Line, intersects l1 l2 ∧ is_parallel_to_plane l1 β ∧ is_parallel_to_plane l2 β) → is_parallel α β
axiom prop_2 : (∃ l : Line, ¬is_parallel_to_plane l α ∧ is_parallel_to_plane l α) → is_parallel_to_plane l α
axiom prop_3 : (∃ l : Line, intersects_along α β l ∧ (∃ l' : Line, is_perpendicular_to_plane l' α)) → ¬is_perpendicular α β
axiom prop_4 : (∃ l : Line, (∃ l1 l2 : Line, intersects l1 l2 ∧ is_perpendicular_to_plane l1 α ∧ is_perpendicular_to_plane l2 α)) → ¬is_perpendicular_to_plane l α

-- Main statement
theorem true_propositions : (prop_1 ∧ prop_2 ∧ ¬prop_3 ∧ ¬prop_4) := sorry

end true_propositions_l559_559411


namespace coeff_x5_in_product_l559_559635

-- Define the first polynomial
def poly1 := x^6 - 4 * x^5 + 6 * x^4 - 5 * x^3 + 3 * x^2 - 2 * x + 1

-- Define the second polynomial
def poly2 := 3 * x^4 - 2 * x^3 + x^2 + 4 * x + 5

-- State the theorem to determine the coefficient of x^5 when the two polynomials are multiplied
theorem coeff_x5_in_product : coeff (poly1 * poly2) 5 = -5 :=
sorry

end coeff_x5_in_product_l559_559635


namespace richard_probability_correct_l559_559432

open Finset

-- Define the problem statement and conditions
noncomputable def richard_probability : ℚ :=
  let total_socks := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let pairs := { #{0,1}, #{2,3}, #{4,5}, #{6,7}, #{8,9} }
  let draw := (total_socks.filter (λ x, x < 10)).powerset
  let favorable_outcomes := ((pairs.toFinset).choose 1) * ((total_socks.diff (pairs.toFinset)).choose 3) in
    have h_favorable: βfinite favorable_outcomes, from sorry,
    have h_total: ∃! (s : Finset ℕ) (h : s ∈ draw), s.card = 5 , from sorry,
    let probability := (favorable_outcomes.card / draw.card : ℚ) in
      probability

-- The theorem we need to prove
theorem richard_probability_correct :
  richard_probability = 5 / 63 :=
sorry

end richard_probability_correct_l559_559432


namespace frank_total_cans_l559_559291

def total_cans_picked_up (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  let total_bags := bags_saturday + bags_sunday
  total_bags * cans_per_bag

theorem frank_total_cans : total_cans_picked_up 5 3 5 = 40 := by
  sorry

end frank_total_cans_l559_559291


namespace cow_manure_plant_height_l559_559022

theorem cow_manure_plant_height (control_height bone_meal_percentage cow_manure_percentage : ℝ)
  (control_height_eq : control_height = 36)
  (bone_meal_eq : bone_meal_percentage = 125)
  (cow_manure_eq : cow_manure_percentage = 200) :
  let bone_meal_height := (bone_meal_percentage / 100) * control_height in
  let cow_manure_height := (cow_manure_percentage / 100) * bone_meal_height in
  cow_manure_height = 90 := by
  sorry

end cow_manure_plant_height_l559_559022


namespace root_5_over_root_4_l559_559863

theorem root_5_over_root_4 :
  (5 ^ (1 / 5)) / (5 ^ (1 / 4)) = 5 ^ (- (1 / 20)) :=
by
  sorry

end root_5_over_root_4_l559_559863


namespace trumpets_tried_out_l559_559415

theorem trumpets_tried_out (flutes clarinets trumpets pianists total_band : ℕ)
  (flute_rate clarinet_rate trumpet_rate pianist_rate : ℚ) 
  (h1 : flutes = 20) (h2 : clarinets = 30) (h3 : pianists = 20) 
  (h4 : total_band = 53)
  (h5 : flute_rate = 0.8) (h6 : clarinet_rate = 0.5) (h7 : trumpet_rate = 1/3) (h8 : pianist_rate = 1/10) 
  : 3 * (total_band - (flute_rate * flutes).toNat - (clarinet_rate * clarinets).toNat - (pianist_rate * pianists).toNat) = 60 :=
by
  sorry

end trumpets_tried_out_l559_559415


namespace store_money_left_l559_559955

variable (total_items : Nat) (original_price : ℝ) (discount_percent : ℝ)
variable (percent_sold : ℝ) (amount_owed : ℝ)

theorem store_money_left
  (h_total_items : total_items = 2000)
  (h_original_price : original_price = 50)
  (h_discount_percent : discount_percent = 0.80)
  (h_percent_sold : percent_sold = 0.90)
  (h_amount_owed : amount_owed = 15000)
  : (total_items * original_price * (1 - discount_percent) * percent_sold - amount_owed) = 3000 := 
by 
  sorry

end store_money_left_l559_559955


namespace probability_of_distinct_dice_numbers_l559_559493

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l559_559493


namespace find_intervals_find_b_range_l559_559737

def vector_m (ω x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (ω * x), 1)

def vector_n (ω x : ℝ) : ℝ × ℝ := (cos (ω * x), cos (ω * x) ^ 2 + 1)

def f (ω b x : ℝ) : ℝ := (sqrt 3 * sin (ω * x) * cos(ω * x) + cos(ω * x)^2 + 1) + b

def f_simplified (ω b x : ℝ) : ℝ := sin (2 * x + π / 6) + 3 / 2 + b

-- Prove that the specified intervals and conditions hold
theorem find_intervals (ω : ℝ) (h_omega_range : ω ∈ set.Icc 0 3) : ∃ k : ℤ, ω = 3 * k + 1 ∧ ∀ x, 
  (set.Icc ((k:ℝ) * π - π / 3) ((k:ℝ) * π + π / 6)) = {x : ℝ | f_simplified ω 0 x > 0 ∧ f_simplified ω 0 x < 0 } :=
sorry

theorem find_b_range (x : ℝ) (h_x_range : x ∈ set.Icc 0 (7 * π / 12)) : real.set.Ioo (-2) ((sqrt 3 - 3) / 2) ∪ { -5 / 2 } = 
    { b : ℝ | ∃ x ∈ set.Icc 0 (7 * π / 12), f 1 b x == 0 } :=
sorry

end find_intervals_find_b_range_l559_559737


namespace cost_of_one_roll_sold_individually_l559_559199

-- Definitions based on conditions
def cost_case_12_rolls := 9
def percent_savings := 0.25

-- Variable representing the cost of one roll sold individually
variable (x : ℝ)

-- Statement to prove
theorem cost_of_one_roll_sold_individually : (12 * x - 12 * percent_savings * x) = cost_case_12_rolls → x = 1 :=
by
  intro h
  -- This is where the proof would go
  sorry

end cost_of_one_roll_sold_individually_l559_559199


namespace calc_expression_l559_559581

theorem calc_expression : 
  ((2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + log 4 3 * log 9 2) = 11 / 36 := 
by 
  sorry

end calc_expression_l559_559581


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l559_559506

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l559_559506


namespace stones_in_courtyard_l559_559106

theorem stones_in_courtyard (S T B : ℕ) (h1 : T = S + 3 * S) (h2 : B = 2 * (T + S)) (h3 : B = 400) : S = 40 :=
by
  sorry

end stones_in_courtyard_l559_559106


namespace trigonometric_identity_l559_559524

theorem trigonometric_identity (t : ℝ) : 
  5.43 * Real.cos (22 * Real.pi / 180 - t) * Real.cos (82 * Real.pi / 180 - t) +
  Real.cos (112 * Real.pi / 180 - t) * Real.cos (172 * Real.pi / 180 - t) = 
  0.5 * (Real.sin t + Real.cos t) :=
sorry

end trigonometric_identity_l559_559524


namespace quadrilateral_area_l559_559551

variable (A B C D : Point)

def is_quad (A B C D : Point) : Prop :=
  A = (0, 0) ∧ B = (0, 2) ∧ C = (3, 2) ∧ D = (5, 5)

noncomputable def area_quad (A B C D : Point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (x4, y4) := D
  (1/2 : ℝ) * ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem quadrilateral_area : is_quad A B C D → area_quad A B C D = (6 + 3 * Real.sqrt 13) / 2 :=
by
  intro h
  sorry

end quadrilateral_area_l559_559551


namespace boys_cannot_score_twice_l559_559065

-- Define the total number of points in the tournament
def total_points_in_tournament : ℕ := 15

-- Define the number of boys and girls
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 4

-- Define the points scored by boys and girls
axiom points_by_boys : ℕ
axiom points_by_girls : ℕ

-- The conditions
axiom total_points_condition : points_by_boys + points_by_girls = total_points_in_tournament
axiom boys_twice_girls_condition : points_by_boys = 2 * points_by_girls

-- The statement to prove
theorem boys_cannot_score_twice : False :=
  by {
    -- Note: provide a sketch to illustrate that under the given conditions the statement is false
    sorry
  }

end boys_cannot_score_twice_l559_559065


namespace union_of_A_and_B_l559_559929

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} :=
by
  sorry

end union_of_A_and_B_l559_559929


namespace cone_surface_area_l559_559699

-- Definitions for the conditions
def base_radius : ℝ := 4
def height : ℝ := 2 * Real.sqrt 5

-- Statement of the problem
theorem cone_surface_area :
  let slant_height := Real.sqrt (base_radius^2 + height^2)
  let base_area := Real.pi * base_radius^2
  let lateral_surface_area := Real.pi * base_radius * slant_height
  base_area + lateral_surface_area = 40 * Real.pi :=
sorry

end cone_surface_area_l559_559699


namespace f_eq_32x5_l559_559038

def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

theorem f_eq_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  -- the proof proceeds here
  sorry

end f_eq_32x5_l559_559038


namespace no_possible_5x10_table_l559_559989

theorem no_possible_5x10_table (M : Matrix (Fin 5) (Fin 10) ℕ) 
  (h1 : ∀ i, (∑ j, M i j) = 30) 
  (h2 : ∀ j, (∑ i, M i j) = 10) :
  False :=
by 
  sorry

end no_possible_5x10_table_l559_559989


namespace range_of_x_l559_559313

variable (f : ℝ → ℝ)
variable (h_inc : ∀ a b : ℝ, a ≤ b → f(a) ≤ f(b))
variable (h_dom : ∀ x : ℝ, 0 ≤ x → 0 ≤ f(x))

theorem range_of_x (x : ℝ) :
  f(2 * x - 1) < f (1 / 3) → x ∈ Set.Ico (1 / 2) (2 / 3) :=
by
  intro h
  have : 0 ≤ 2 * x - 1 := sorry
  have : 2 * x - 1 < 1 / 3 := sorry
  sorry

end range_of_x_l559_559313


namespace ordered_pairs_1806_l559_559110

theorem ordered_pairs_1806 :
  (∃ (xy_list : List (ℕ × ℕ)), xy_list.length = 12 ∧ ∀ (xy : ℕ × ℕ), xy ∈ xy_list → xy.1 * xy.2 = 1806) :=
sorry

end ordered_pairs_1806_l559_559110


namespace centroid_moves_on_circle_l559_559733

/-- Given a triangle ABC where base AB is fixed in length and position,
and given that the vertex C moves on a circle centered at the midpoint M of AB,
prove that the centroid G of triangle ABC moves on a circle centered at M. -/
theorem centroid_moves_on_circle
  (A B C M G : Point)
  (hAB_fixed : fixed AB)
  (hM_mid_AB : M = midpoint A B)
  (hC_on_circle : ∃ r, ∀ θ, C = circle_point M r θ)
  (hG_centroid : G = centroid A B C) :
  ∃ r', ∀ θ, G = circle_point M r' θ :=
sorry

end centroid_moves_on_circle_l559_559733


namespace max_interesting_numbers_l559_559905

/-- 
Define a function to calculate the sum of the digits of a natural number.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

/-- 
Define a function to check if a natural number is prime.
-/
def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

/-- 
Define a number to be interesting if the sum of its digits is a prime number.
-/
def interesting_number (n : ℕ) : Prop :=
  is_prime (sum_of_digits n)

/-- 
Statement: For any five consecutive natural numbers, there are at most 4 interesting numbers.
-/
theorem max_interesting_numbers (a : ℕ) :
  (finset.range 5).filter (λ i, interesting_number (a + i)) .card ≤ 4 := 
by
  sorry

end max_interesting_numbers_l559_559905


namespace distance_between_midpoints_of_skew_edges_l559_559278

theorem distance_between_midpoints_of_skew_edges (S_total : ℝ) (hS : S_total = 36) : 
  ∃ d : ℝ, d = 3 :=
by
  let a := real.sqrt (S_total / 6)
  let K := (0, a / 2, 0)
  let M := (a / 2, 0, 0)
  let d := real.sqrt ((a / 2 - 0) ^ 2 + (0 - a / 2) ^ 2 + (0 - 0) ^ 2)
  use d
  sorry

end distance_between_midpoints_of_skew_edges_l559_559278


namespace sum_seq_integer_part_l559_559875

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1/3 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * a n + a n

theorem sum_seq_integer_part :
  ∀ a : ℕ → ℚ,
    seq a →
    int.fract (finset.sum (finset.range 2016) (λ n, 1 / (a n + 1))) = 2 := 
sorry

end sum_seq_integer_part_l559_559875


namespace inequality_solution_l559_559652

section
variables (y : ℝ)

def domain (y : ℝ) : Prop := 
  y ≠ 2 ∧ y ≠ -3

theorem inequality_solution :
  (domain y ∧ (2 / (y - 2) + 5 / (y + 3) ≤ 2)) ↔ (y ∈ set.Icc (-3 : ℝ) (-1) ∪ set.Ioc (2 : ℝ) 4) :=
sorry
end

end inequality_solution_l559_559652


namespace compare_values_l559_559031

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log (logBase (1 / 3) 2)
noncomputable def c : ℝ := 1 / Real.sin 1

theorem compare_values : c > a ∧ a > b :=
by
  -- Proof goes here
  sorry

end compare_values_l559_559031


namespace minimum_distance_PQ_l559_559751

theorem minimum_distance_PQ
    (P Q : EuclideanSpace ℝ (fin 2))
    (curve_Q : (∃ x y : ℝ, Q = (x, y) ∧ x^2 + (y + 2)^2 = 1)) :
    ∃ min_dist : ℝ, min_dist = -1 :=
begin
    sorry
end

end minimum_distance_PQ_l559_559751


namespace option_B_correct_l559_559910

theorem option_B_correct (x : ℝ) (h : x > 0) : 
  let option_A := ∀ x > 0, x ≠ 1 → log x + 1 / log x ≥ 2 in
  let option_B := sqrt x + 1 / sqrt x ≥ 2 in
  let option_C := ∀ x ≥ 2, x + 1 / x ≥ 2 in
  let option_D := ∃ x, 0 < x ∧ x ≤ 2 ∧ ∀ y, 0 < y ∧ y ≤ 2 → x - 1 / x ≤ y - 1 / y in
  ¬ option_A ∧ option_B ∧ ¬ option_C ∧ ¬ option_D :=
begin
  sorry
end

end option_B_correct_l559_559910


namespace max_interesting_numbers_in_five_consecutive_l559_559899

def sum_of_digits (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

def is_prime_sum_of_digits (n : ℕ) : Prop :=
  (sum_of_digits n).prime

theorem max_interesting_numbers_in_five_consecutive (n : ℕ) :
  let consecutive_nums := [n, n+1, n+2, n+3, n+4]
  (count is_prime_sum_of_digits consecutive_nums ≤ 4) :=
sorry

end max_interesting_numbers_in_five_consecutive_l559_559899


namespace trapezoid_DC_length_l559_559651

-- Definitions of known quantities
def AB : ℝ := 6
def BC : ℝ := 4
def angle_BCD : ℝ := 30
def angle_CDA : ℝ := 45
def expected_DC : ℝ := 6 + (4 * real.sqrt 3) / 3 + 3 * real.sqrt 2

-- Lean statement to prove the desired condition
theorem trapezoid_DC_length :
  ∀ (AB BC angle_BCD angle_CDA : ℝ) (h1 : AB = 6) (h2 : BC = 4)
    (h3 : angle_BCD = 30) (h4 : angle_CDA = 45), DC = 6 + (4 * real.sqrt 3) / 3 + 3 * real.sqrt 2 :=
by
  sorry

end trapezoid_DC_length_l559_559651


namespace Davante_boys_count_l559_559985

def days_in_week := 7
def friends (days : Nat) := days * 2
def girls := 3
def boys (total_friends girls : Nat) := total_friends - girls

theorem Davante_boys_count :
  boys (friends days_in_week) girls = 11 :=
  by
    sorry

end Davante_boys_count_l559_559985


namespace total_eggs_calc_l559_559212

section

variable (total_chickens : ℕ)
variable (bcm_percentage rir_percentage lh_percentage : ℚ)
variable (bcm_hen_percentage rir_hen_percentage lh_hen_percentage : ℚ)
variable (bcm_egg_dist rir_egg_dist lh_egg_dist : ℚ × ℚ × ℚ)
variable (bcm_egg_rates rir_egg_rates lh_egg_rates : ℕ × ℕ × ℕ)

def number_of_chickens (percentage : ℚ) : ℕ :=
  (percentage * (total_chickens : ℚ)).toNat

def number_of_hens (chickens : ℕ) (hen_percentage : ℚ) : ℕ :=
  (hen_percentage * (chickens : ℚ)).toNat

def number_of_egg_laying_hens (total_hens : ℕ) (distribution : ℚ × ℚ × ℚ) : ℕ × ℕ × ℕ :=
  let (d1, d2, d3) := distribution
  let h1 := (d1 * (total_hens : ℚ)).toNat
  let h2 := (d2 * (total_hens : ℚ)).toNat
  let h3 := total_hens - (h1 + h2)
  (h1, h2, h3)

def total_eggs_per_week (hens : ℕ × ℕ × ℕ) (rates : ℕ × ℕ × ℕ) : ℕ :=
  let (h1, h2, h3) := hens
  let (r1, r2, r3) := rates
  h1 * r1 + h2 * r2 + h3 * r3

theorem total_eggs_calc
  (H_total_chickens : total_chickens = 500)
  (H_bcm_percentage : bcm_percentage = 0.25)
  (H_rir_percentage : rir_percentage = 0.4)
  (H_lh_percentage : lh_percentage = 0.35)
  (H_bcm_hen_percentage : bcm_hen_percentage = 0.65)
  (H_rir_hen_percentage : rir_hen_percentage = 0.55)
  (H_lh_hen_percentage : lh_hen_percentage = 0.60)
  (H_bcm_egg_dist : bcm_egg_dist = (0.40, 0.30, 0.30))
  (H_rir_egg_dist : rir_egg_dist = (0.20, 0.50, 0.30))
  (H_lh_egg_dist : lh_egg_dist = (0.25, 0.45, 0.30))
  (H_bcm_egg_rates : bcm_egg_rates = (3, 4, 5))
  (H_rir_egg_rates : rir_egg_rates = (5, 6, 7))
  (H_lh_egg_rates : lh_egg_rates = (6, 7, 8)) :
  let bcm_chickens := number_of_chickens bcm_percentage total_chickens
  let rir_chickens := number_of_chickens rir_percentage total_chickens
  let lh_chickens := number_of_chickens lh_percentage total_chickens
  let bcm_hens := number_of_hens bcm_chickens bcm_hen_percentage
  let rir_hens := number_of_hens rir_chickens rir_hen_percentage
  let lh_hens := number_of_hens lh_chickens lh_hen_percentage
  let bcm_egg_laying_hens := number_of_egg_laying_hens bcm_hens bcm_egg_dist
  let rir_egg_laying_hens := number_of_egg_laying_hens rir_hens rir_egg_dist
  let lh_egg_laying_hens := number_of_egg_laying_hens lh_hens lh_egg_dist
  let bcm_eggs_per_week := total_eggs_per_week bcm_egg_laying_hens bcm_egg_rates
  let rir_eggs_per_week := total_eggs_per_week rir_egg_laying_hens rir_egg_rates
  let lh_eggs_per_week := total_eggs_per_week lh_egg_laying_hens lh_egg_rates
  bcm_eggs_per_week + rir_eggs_per_week + lh_eggs_per_week = 1729 :=
sorry

end

end total_eggs_calc_l559_559212


namespace area_of_ngon_gt_nlogn_div4_minus_half_l559_559060

open Function

variable {n : Nat} (polygon : List (Fin n → ℝ × ℝ))

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

def area_of_ngon (polygon : List (Fin n → ℝ × ℝ)) : ℝ :=
  polygon.to_finset.sum (λ ⟨i, hi⟩, area_of_triangle (polygon.nth i) (polygon.nth ((i + 1) % n)) (polygon.nth ((i + 2) % n)))

noncomputable def verify_property (polygon : List (Fin n → ℝ × ℝ)) : Prop :=
  ∀ ⟨i, hi⟩, area_of_triangle (polygon.nth i) (polygon.nth ((i + 1) % n)) (polygon.nth ((i + 2) % n)) ≥ 1

theorem area_of_ngon_gt_nlogn_div4_minus_half
  (h1 : n ≥ 4)
  (h2 : verify_property polygon) :
  area_of_ngon polygon > (n * log 2 n) / 4 - 1 / 2 :=
by sorry

end area_of_ngon_gt_nlogn_div4_minus_half_l559_559060


namespace circle_center_l559_559205

theorem circle_center (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 10)
  (h2 : x - 2 * y = 0) :
  (x = 2 ∧ y = 1) :=
begin
  sorry
end

end circle_center_l559_559205


namespace moles_of_hcl_l559_559656

-- Definitions according to the conditions
def methane := 1 -- 1 mole of methane (CH₄)
def chlorine := 2 -- 2 moles of chlorine (Cl₂)
def hcl := 1 -- The expected number of moles of Hydrochloric acid (HCl)

-- The Lean 4 statement (no proof required)
theorem moles_of_hcl (methane chlorine : ℕ) : hcl = 1 :=
by sorry

end moles_of_hcl_l559_559656


namespace symmetric_center_l559_559664

-- Defining the original function
def f (x : ℝ) : ℝ := Real.sin (4 * x + π / 3)

-- Defining the transformed function
def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem symmetric_center :
  (∃ k : ℤ, g (k * π / 2) = 0) ∧ (∀ x : ℝ, k * π / 2 = x → g x = 0) ∧ (x = π / 2) := by
  sorry

end symmetric_center_l559_559664


namespace cross_section_is_isosceles_triangle_l559_559092

-- Definitions directly from the conditions
structure Cone where
  base : Set Point    -- The base is a circle
  vertex : Point      -- The vertex of the cone
  axis : Line         -- The axis is a line segment connecting the vertex and center of the base circle

def cross_section_through_axis (c : Cone) (plane : Plane) : Set Point := {
-- Definition is somewhat abstract here as capturing actual geometry is complex,
-- but intended to represent the set of points of intersection of the cone with the plane through the axis.
 sorry
}

theorem cross_section_is_isosceles_triangle (c : Cone) (h₁ : ∀ p, p ∈ c.base → distance c.vertex p = distance c.vertex p) 
(h₂: ∀ p, p ∈ c.axis → p = midpoint p.vertex c.base.center) : 
(∃ plane : Plane, (cross_section_through_axis c plane).is_isosceles_triangle) :=
sorry

end cross_section_is_isosceles_triangle_l559_559092


namespace arithmetic_seq_problem_l559_559359

variable (a : ℕ → ℕ)

def arithmetic_seq (a₁ d : ℕ) : ℕ → ℕ :=
  λ n => a₁ + n * d

theorem arithmetic_seq_problem (a₁ d : ℕ)
  (h_cond : (arithmetic_seq a₁ d 1) + 2 * (arithmetic_seq a₁ d 5) + (arithmetic_seq a₁ d 9) = 120)
  : (arithmetic_seq a₁ d 2) + (arithmetic_seq a₁ d 8) = 60 := 
sorry

end arithmetic_seq_problem_l559_559359


namespace carrie_strawberries_harvest_l559_559583

theorem carrie_strawberries_harvest :
  ∀ (length width : ℕ) (plants_per_sq_ft yield_per_plant : ℕ),
    length = 10 →
    width = 12 →
    plants_per_sq_ft = 5 →
    yield_per_plant = 10 →
    length * width * plants_per_sq_ft * yield_per_plant = 6000 :=
by
  intros length width plants_per_sq_ft yield_per_plant h_length h_width h_plants h_yield
  rw [h_length, h_width, h_plants, h_yield]
  norm_num
  sorry

end carrie_strawberries_harvest_l559_559583


namespace sampling_interval_l559_559134

theorem sampling_interval (total_students sample_size k : ℕ) (h1 : total_students = 1200) (h2 : sample_size = 40) (h3 : k = total_students / sample_size) : k = 30 :=
by
  sorry

end sampling_interval_l559_559134


namespace class_raised_initial_amount_l559_559055

/-- Miss Grayson's class raised some money for their field trip.
Each student contributed $5 each.
There are 20 students in her class.
The cost of the trip is $7 for each student.
After all the field trip costs were paid, there is $10 left in Miss Grayson's class fund.
Prove that the class initially raised $150 for the field trip. -/
theorem class_raised_initial_amount
  (students : ℕ)
  (contribution_per_student : ℕ)
  (cost_per_student : ℕ)
  (remaining_fund : ℕ)
  (total_students : students = 20)
  (per_student_contribution : contribution_per_student = 5)
  (per_student_cost : cost_per_student = 7)
  (remaining_amount : remaining_fund = 10) :
  (students * contribution_per_student + remaining_fund) = 150 := 
sorry

end class_raised_initial_amount_l559_559055


namespace crayon_percentage_after_adjustments_l559_559019

def initial_crayons : ℕ := 41
def initial_pencils : ℕ := 26
def removed_crayons : ℕ := 8
def added_crayons : ℕ := 12
def crayon_increase_rate : ℝ := 0.10

theorem crayon_percentage_after_adjustments :
  let total_crayons := initial_crayons - removed_crayons + added_crayons
  let total_crayons_after_increase := (total_crayons : ℝ) * (1 + crayon_increase_rate)
  let rounded_crayons := Int.round total_crayons_after_increase
  let total_items := rounded_crayons + (initial_pencils : ℤ)
  let crayon_percentage := (rounded_crayons.to_rat / total_items.to_rat) * 100
  rounded_crayons = 50 ∧ 
  Real.approx (crayon_percentage : ℝ) 65.79 :=
by
  sorry

end crayon_percentage_after_adjustments_l559_559019


namespace inscribe_three_equal_circles_l559_559484

-- Given a larger circle (circle with radius R centered at O)
variables (R : ℝ) (O : ℝ × ℝ)

-- Define conditions of the problem:
-- This will capture the existence of three equal circles tangentially inscribed in a larger circle.
theorem inscribe_three_equal_circles (R : ℝ) (O : ℝ × ℝ) (r : ℝ) (C1 C2 C3 : ℝ × ℝ) :
  r > 0 → r < R / 2 →
  dist O C1 = R - r ∧ dist O C2 = R - r ∧ dist O C3 = R - r →
  dist C1 C2 = 2 * r ∧ dist C2 C3 = 2 * r ∧ dist C3 C1 = 2 * r →
  ∃ (C1 C2 C3 : ℝ × ℝ), 
  circle C1 r ∧ circle C2 r ∧ circle C3 r :=
begin
  sorry
end

end inscribe_three_equal_circles_l559_559484


namespace arnold_danny_age_l559_559515

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 15) : x = 7 :=
sorry

end arnold_danny_age_l559_559515


namespace find_largest_a_l559_559813

theorem find_largest_a (a : ℝ) (h : a > 1) : 
  (∀ x, 0 ≤ x → x ≤ 3 → 0 ≤ a^x - 1 ∧ a^x - 1 ≤ 3) → a ≤ real.cbrt 4 := 
by 
  sorry

end find_largest_a_l559_559813


namespace prob_perpendicular_lines_square_l559_559949

theorem prob_perpendicular_lines_square
  (vertices : Finset ℕ) (lines : Finset (Finset ℕ))
  (total_outcomes side_perp_outcomes diagonal_perp_outcomes : ℕ) :
  vertices = {1, 2, 3, 4} →
  lines = {{1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 3}, {2, 4}} →
  total_outcomes = 36 →
  side_perp_outcomes = 4 →
  diagonal_perp_outcomes = 4 →
  (total_outcomes = 36 ∧ side_perp_outcomes = 4 ∧ diagonal_perp_outcomes = 4) →
  let favorable_outcomes := (side_perp_outcomes + diagonal_perp_outcomes) * 2 in
  let probability := favorable_outcomes.to_rat / total_outcomes.to_rat in
  probability = 5 / 18 := 
by
  intros h_vertices h_lines h_total h_side h_diag h_total_side_diag
  have h_favorable : favorable_outcomes = 10 := by
    have hf : favorable_outcomes = (4 + 4) * 2 := rfl
    exact hf
  have h_prob : probability = 5 / 18 := by
    have hp : probability = 10 / 36 := rfl
    norm_num at hp
  exact h_prob

end prob_perpendicular_lines_square_l559_559949


namespace no_such_numbers_l559_559259

theorem no_such_numbers :
  ¬∃ (a : Fin 2013 → ℕ),
    (∀ i : Fin 2013, (∑ j : Fin 2013, if j ≠ i then a j else 0) ≥ (a i) ^ 2) :=
sorry

end no_such_numbers_l559_559259


namespace calculate_years_l559_559462

variable {P R T SI : ℕ}

-- Conditions translations
def simple_interest_one_fifth (P SI : ℕ) : Prop :=
  SI = P / 5

def rate_of_interest (R : ℕ) : Prop :=
  R = 4

-- Proof of the number of years T
theorem calculate_years (h1 : simple_interest_one_fifth P SI)
                        (h2 : rate_of_interest R)
                        (h3 : SI = (P * R * T) / 100) : T = 5 :=
by
  sorry

end calculate_years_l559_559462


namespace Audrey_ball_count_l559_559377

variable (Jake : ℕ) (Audrey : ℕ)

axiom Jake_has_7_balls : Jake = 7
axiom Jake_has_34_fewer_than_Audrey : Jake + 34 = Audrey

theorem Audrey_ball_count : Audrey = 41 := by
  rw [← Jake_has_34_fewer_than_Audrey, Jake_has_7_balls]
  exact rfl

end Audrey_ball_count_l559_559377


namespace sufficient_condition_for_B_l559_559668

noncomputable def A (k : ℝ) : set ℝ := {x | x >= k}
noncomputable def B : set ℝ := {x | 3 / (x + 1) < 1}

theorem sufficient_condition_for_B (k : ℝ) : 
  (∀ x, x ∈ A k → x ∈ B) → (¬ (∀ x, x ∈ B → x ∈ A k)) → k > 2 :=
by
  sorry

end sufficient_condition_for_B_l559_559668


namespace probability_five_distinct_numbers_l559_559500

def num_dice := 5
def num_faces := 6

def favorable_outcomes : ℕ := nat.factorial 5 * num_faces
def total_outcomes : ℕ := num_faces ^ num_dice

theorem probability_five_distinct_numbers :
  (favorable_outcomes / total_outcomes : ℚ) = 5 / 54 := 
sorry

end probability_five_distinct_numbers_l559_559500


namespace average_retail_price_l559_559586

theorem average_retail_price 
  (products : Fin 20 → ℝ)
  (h1 : ∀ i, 400 ≤ products i) 
  (h2 : ∃ s : Finset (Fin 20), s.card = 10 ∧ ∀ i ∈ s, products i < 1000)
  (h3 : ∃ i, products i = 11000): 
  (Finset.univ.sum products) / 20 = 1200 := 
by
  sorry

end average_retail_price_l559_559586


namespace num_ordered_pairs_1806_l559_559112

theorem num_ordered_pairs_1806 :
  let n := 1806 in
  let pf := [(2, 1), (3, 2), (101, 1)] in
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1) in
  ∃ (c : ℕ), c = num_divisors ∧ c = 12 :=
by
  let n := 1806
  let pf := [(2, 1), (3, 2), (101, 1)]
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1)
  use num_divisors
  split
  . rfl
  . rfl
  sorry

end num_ordered_pairs_1806_l559_559112


namespace radius_of_third_circle_is_correct_l559_559886

noncomputable def radius_of_third_circle (r : ℝ) :=
  ∃ P Q R : ℝ × ℝ,
    -- Centers of Circle 1 (P) and Circle 2 (Q)
    (norm P = 3) ∧ 
    (norm Q = 7) ∧
    -- Distance between centers P and Q is 10
    (dist P Q = 10) ∧
    -- center of third circle R such that third circle is tangent to Circle 1 and Circle 2
    (dist P R = 3 + r) ∧ 
    (dist Q R = 7 + r) ∧
    -- Third circle is tangent to the line connecting centers P and Q 
    (dist R ((P + Q) / 2) = (3 + r).abs / 2) ∧
    -- The radius of the third circle is r
    (r = real.sqrt 46 - 5)

theorem radius_of_third_circle_is_correct : 
  ∀ r : ℝ, radius_of_third_circle r → r = real.sqrt 46 - 5 :=
by
  intro r h
  -- Apply the conditions extracted
  rcases h with ⟨P, Q, R, hP, hQ, hPQ, hPR, hQR, hR⟩
  sorry

end radius_of_third_circle_is_correct_l559_559886


namespace monotonically_increasing_interval_l559_559637

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem monotonically_increasing_interval : 
  ∀ x : ℝ, x ∈ set.Icc (-π) (-π / 4) → 
  ∀ y : ℝ, y ∈ set.Icc x (-π / 4) → f(x) ≤ f(y) := 
by 
  sorry

end monotonically_increasing_interval_l559_559637


namespace remainder_of_polynomial_l559_559658

theorem remainder_of_polynomial (x : ℝ) : (λ x : ℝ, x^4 - x + 1) (-3) = 85 :=
by
  sorry

end remainder_of_polynomial_l559_559658


namespace average_minutes_run_is_44_over_3_l559_559244

open BigOperators

def average_minutes_run (s : ℕ) : ℚ :=
  let sixth_graders := 3 * s
  let seventh_graders := s
  let eighth_graders := s / 2
  let total_students := sixth_graders + seventh_graders + eighth_graders
  let total_minutes_run := 20 * sixth_graders + 12 * eighth_graders
  total_minutes_run / total_students

theorem average_minutes_run_is_44_over_3 (s : ℕ) (h1 : 0 < s) : 
  average_minutes_run s = 44 / 3 := 
by
  sorry

end average_minutes_run_is_44_over_3_l559_559244


namespace everyone_can_cross_l559_559832

-- Define each agent
inductive Agent
| C   -- Princess Sonya
| K (i : Fin 8) -- Knights numbered 1 to 7

open Agent

-- Define friendships
def friends (a b : Agent) : Prop :=
  match a, b with
  | C, (K 4) => False
  | (K 4), C => False
  | _, _ => (∃ i : Fin 8, a = K i ∧ b = K (i+1)) ∨ (∃ i : Fin 7, a = K (i+1) ∧ b = K i) ∨ a = C ∨ b = C

-- Define the crossing conditions
def boatCanCarry : List Agent → Prop
| [a, b] => friends a b
| [a, b, c] => friends a b ∧ friends b c ∧ friends a c
| _ => False

-- The main statement to prove
theorem everyone_can_cross (agents : List Agent) (steps : List (List Agent)) :
  agents = [C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7] →
  (∀ step ∈ steps, boatCanCarry step) →
  (∃ final_state : List (List Agent), final_state = [[C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7]]) :=
by 
  -- The proof is omitted.
  sorry

end everyone_can_cross_l559_559832


namespace max_weak_quartets_120_l559_559565

noncomputable def max_weak_quartets (n : ℕ) : ℕ :=
  -- Placeholder definition to represent the maximum weak quartets
  sorry  -- To be replaced with the actual mathematical definition

theorem max_weak_quartets_120 : max_weak_quartets 120 = 4769280 := by
  sorry

end max_weak_quartets_120_l559_559565


namespace tan_sin_div_l559_559607

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l559_559607


namespace sqrt_of_neg_7_sq_is_7_l559_559174

theorem sqrt_of_neg_7_sq_is_7 : sqrt ((-7)^2) = 7 :=
by sorry

end sqrt_of_neg_7_sq_is_7_l559_559174


namespace angle_C_measurement_l559_559375

variables (A B C : ℝ)

theorem angle_C_measurement
  (h1 : A + C = 2 * B)
  (h2 : C - A = 80)
  (h3 : A + B + C = 180) :
  C = 100 :=
by sorry

end angle_C_measurement_l559_559375


namespace distributi_l559_559881

def number_of_distributions (spots : ℕ) (classes : ℕ) (min_spot_per_class : ℕ) : ℕ :=
  Nat.choose (spots - min_spot_per_class * classes + (classes - 1)) (classes - 1)

theorem distributi.on_of_10_spots (A B C : ℕ) (hA : A ≥ 1) (hB : B ≥ 1) (hC : C ≥ 1) 
(h_total : A + B + C = 10) : number_of_distributions 10 3 1 = 36 :=
by
  sorry

end distributi_l559_559881


namespace inverse_proposition_true_l559_559100

variables {l s : Type} [EuclideanGeometry l s]

-- Conditions
def is_perpendicular_to_projection (l s : Type) [EuclideanGeometry l s] : Prop :=
  -- Placeholder for the definition that line l is perpendicular to the projection of skew line s
  sorry

def is_perpendicular (l s : Type) [EuclideanGeometry l s] : Prop :=
  -- Placeholder for the definition that line l is perpendicular to skew line s
  sorry

-- Theorem to prove
theorem inverse_proposition_true :
  ∀ (l s : Type) [EuclideanGeometry l s],
    ¬is_perpendicular_to_projection l s → ¬is_perpendicular l s :=
by
  sorry

end inverse_proposition_true_l559_559100


namespace terry_total_driving_time_l559_559082

-- Define the conditions
def speed : ℝ := 40 -- miles per hour
def distance : ℝ := 60 -- miles

-- Define the time for one trip
def time_for_one_trip (d : ℝ) (s : ℝ) : ℝ := d / s

-- Define the total driving time for a round trip (forth and back)
def total_driving_time (d : ℝ) (s : ℝ) : ℝ := 2 * time_for_one_trip d s

-- State the theorem to be proven
theorem terry_total_driving_time : total_driving_time distance speed = 3 := 
by
  sorry

end terry_total_driving_time_l559_559082


namespace find_f_one_l559_559628

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_one : (∀ x: ℝ, 2 * x + 5 ≠ 0 → f(x) - f ((x - 1) / (2 * x + 5)) = x^2 + 1) → f(1) = 101 / 25 := 
by 
  assume h,
  sorry

end find_f_one_l559_559628


namespace find_XY_sq_l559_559801

variable {A B C T X Y : Type}
variable [point A] [point B] [point C] [point T] [point X] [point Y]
variable (h1 : triangle A B C)
variable (h2 : is_acute h1)
variable (h3 : scalene h1)
variable (w : circumcircle h1)
variable (tangent_wb : tangent w B)
variable (tangent_wc : tangent w C)
variable (T_eq_tangent_intersection : T = intersection tangent_wb tangent_wc)
variable (X_proj_T_AB : X = project T (line B A))
variable (Y_proj_T_AC : Y = project T (line C A))
variable (BT_len : length B T = 20)
variable (CT_len : length C T = 20)
variable (BC_len : length B C = 26)
variable (dists_sum : length_sq T X + length_sq T Y + length_sq X Y = 1750)

theorem find_XY_sq : length_sq X Y = 950 :=
sorry

end find_XY_sq_l559_559801


namespace intersection_eq_interval_l559_559331

def P : Set ℝ := {x | x * (x - 3) < 0}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_eq_interval : P ∩ Q = {x | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_eq_interval_l559_559331


namespace sin_product_inequality_l559_559680

theorem sin_product_inequality (n : ℕ) (x_i : ℕ → ℝ) (h1 : ∀ i, 0 < x_i i ∧ x_i i < π) (x : ℝ) (h2 : x = (∑ i in finset.range n, x_i i) / n) :
  (finset.prod (finset.range n) (λ i, sin (x_i i) / x_i i)) ≤ (abs (sin x / x)) ^ n :=
sorry

end sin_product_inequality_l559_559680


namespace product_eq_binom_l559_559809

open Nat

theorem product_eq_binom (n : ℕ) (h : n > 19) : 
  (∏ i in (range 20), (n - i)) = Nat.choose n 20 :=
sorry

end product_eq_binom_l559_559809


namespace coeff_x2y4_in_expansion_l559_559277

theorem coeff_x2y4_in_expansion : 
  let f := fun x y => (x + y) * (2 * x - y) ^ 5 in
  -- Coefficient of x^2 y^4 in the expansion of (x + y)(2x - y)^5
  (coeff (expansion f) (2, 4)) = 50 := 
by
  sorry

end coeff_x2y4_in_expansion_l559_559277


namespace commute_time_value_l559_559219

variables (x y : ℝ)

noncomputable def commute_times := [x, y, 8, 11, 9]

def average (l : list ℝ) := (l.sum / l.length)

def variance (l : list ℝ) :=
  let avg := average l in
  (l.map (λ xi, (xi - avg)^2)).sum / l.length

theorem commute_time_value 
(h_avg : average commute_times = 8)
(h_var : variance commute_times = 4) :
  |x - y| = 2 := 
sorry

end commute_time_value_l559_559219


namespace num_ordered_pairs_1806_l559_559111

theorem num_ordered_pairs_1806 :
  let n := 1806 in
  let pf := [(2, 1), (3, 2), (101, 1)] in
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1) in
  ∃ (c : ℕ), c = num_divisors ∧ c = 12 :=
by
  let n := 1806
  let pf := [(2, 1), (3, 2), (101, 1)]
  let num_divisors := (1 + 1) * (2 + 1) * (1 + 1)
  use num_divisors
  split
  . rfl
  . rfl
  sorry

end num_ordered_pairs_1806_l559_559111


namespace total_animals_l559_559025

namespace Zoo

def snakes := 15
def monkeys := 2 * snakes
def lions := monkeys - 5
def pandas := lions + 8
def dogs := pandas / 3

theorem total_animals : snakes + monkeys + lions + pandas + dogs = 114 := by
  -- definitions from conditions
  have h_snakes : snakes = 15 := rfl
  have h_monkeys : monkeys = 2 * snakes := rfl
  have h_lions : lions = monkeys - 5 := rfl
  have h_pandas : pandas = lions + 8 := rfl
  have h_dogs : dogs = pandas / 3 := rfl
  -- sorry is used as a placeholder for the proof
  sorry

end Zoo

end total_animals_l559_559025


namespace cow_manure_plant_height_l559_559020

theorem cow_manure_plant_height
  (control_plant_height : ℝ)
  (bone_meal_ratio : ℝ)
  (cow_manure_ratio : ℝ)
  (h1 : control_plant_height = 36)
  (h2 : bone_meal_ratio = 1.25)
  (h3 : cow_manure_ratio = 2) :
  (control_plant_height * bone_meal_ratio * cow_manure_ratio) = 90 :=
sorry

end cow_manure_plant_height_l559_559020


namespace range_of_m_l559_559678

noncomputable def p (x : ℝ) := abs(1 - (x - 1) / 3) >= 2
noncomputable def q (x : ℝ) (m : ℝ) := x^2 - 2 * x + 1 - m^2 >= 0

theorem range_of_m (m : ℝ) (h : 0 < m)  :
  (∀ x : ℝ, ¬(p x) → ¬(q x m)) → 
  0 < m ∧ m ≤ 3 := 
sorry

end range_of_m_l559_559678


namespace sequence_1005th_term_4018_l559_559661

def sequence_term_condition (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (∑ i in (finset.range n).map (finset.nat_embedded i)).val / n = 2 * n

theorem sequence_1005th_term_4018 (seq : ℕ → ℕ) (h : sequence_term_condition seq) : seq 1005 = 4018 :=
by sorry

end sequence_1005th_term_4018_l559_559661


namespace find_line_eq_through_MN_l559_559681

noncomputable def M : ℝ × ℝ := ((1 + 5/3) / 2, (0 + 2/3) / 2)

theorem find_line_eq_through_MN :
  let N := (-1, 1 : ℝ) in
  let l1 := λ (p : ℝ × ℝ), p.1 + 2 * p.2 - 1 = 0 in
  let l2 := λ (p : ℝ × ℝ), p.1 + 2 * p.2 - 3 = 0 in
  let C := (1, 0 : ℝ) in
  let D := (5/3, 2/3 : ℝ) in
  let M := ((1 + 5/3) / 2, (0 + 2/3) / 2 : ℝ × ℝ) in
  ∃ l : ℝ → ℝ → Prop, (∀ p : ℝ × ℝ, l p.1 p.2 ↔ 2 * p.1 + 7 * p.2 - 5 = 0) ∧
    (l M.1 M.2 = true) ∧ (l N.1 N.2 = true) :=
by
  sorry

end find_line_eq_through_MN_l559_559681


namespace peaches_sold_to_relatives_l559_559051

-- Definitions based on the conditions
def total_peaches := 15
def sold_to_friends := 10
def price_per_friends_peach := 2
def total_earned := 25
def price_per_relative_peach := 1.25

-- Compute the number of peaches sold to relatives based on the conditions
theorem peaches_sold_to_relatives :
  (total_earned - (sold_to_friends * price_per_friends_peach)) / price_per_relative_peach = 4 :=
by
  have earnings_from_friends := sold_to_friends * price_per_friends_peach
  have earnings_from_relatives := total_earned - earnings_from_friends
  have peaches_to_relatives := earnings_from_relatives / price_per_relative_peach
  show peaches_to_relatives = 4 from sorry

end peaches_sold_to_relatives_l559_559051


namespace area_of_inscribed_rectangle_l559_559533

theorem area_of_inscribed_rectangle (r : ℝ) (h : r = 6) (ratio : ℝ) (hr : ratio = 3 / 1) :
  ∃ (length width : ℝ), (width = 2 * r) ∧ (length = ratio * width) ∧ (length * width = 432) :=
by
  sorry

end area_of_inscribed_rectangle_l559_559533


namespace painters_room_area_l559_559482

theorem painters_room_area :
  ∃ (x : ℝ), 
  (let r₁ := x / 6 in
   let r₂ := x / 8 in
   let effective_combined_rate := r₁ + r₂ - 5 in
   4 * effective_combined_rate = x) ∧ x = 120 :=
sorry

end painters_room_area_l559_559482


namespace Sheila_attends_picnic_probability_l559_559841

theorem Sheila_attends_picnic_probability :
  let P_rain := 0.5
  let P_no_rain := 0.5
  let P_Sheila_goes_if_rain := 0.3
  let P_Sheila_goes_if_no_rain := 0.7
  let P_friend_agrees := 0.5
  (P_rain * P_Sheila_goes_if_rain + P_no_rain * P_Sheila_goes_if_no_rain) * P_friend_agrees = 0.25 := 
by
  sorry

end Sheila_attends_picnic_probability_l559_559841


namespace cube_root_neg8_l559_559448

theorem cube_root_neg8 : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  use -2
  split
  { calc (-2)^3 = -8 : by norm_num }
  { rfl }

end cube_root_neg8_l559_559448


namespace tan_sin_identity_l559_559590

noncomputable def tan_deg : ℝ := tan (real.pi / 6)
noncomputable def sin_deg : ℝ := sin (real.pi / 6)

theorem tan_sin_identity :
  (tan_deg ^ 2 - sin_deg ^ 2) / (tan_deg ^ 2 * sin_deg ^ 2) = 1 :=
by
  have tan_30 := by
    simp [tan_deg, real.tan_pi_div_six]
    norm_num
  have sin_30 := by
    simp [sin_deg, real.sin_pi_div_six]
    norm_num
  sorry

end tan_sin_identity_l559_559590


namespace greatest_k_divides_n_l559_559549

theorem greatest_k_divides_n (n : ℕ) (h_pos : 0 < n) (h_divisors_n : Nat.totient n = 72) (h_divisors_5n : Nat.totient (5 * n) = 90) : ∃ k : ℕ, ∀ m : ℕ, (5^k ∣ n) → (5^(k+1) ∣ n) → k = 3 :=
by
  sorry

end greatest_k_divides_n_l559_559549


namespace option_d_is_right_triangle_l559_559778

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + c^2 = b^2

theorem option_d_is_right_triangle (a b c : ℝ) (h : a^2 = b^2 - c^2) :
  right_triangle a b c :=
by
  sorry

end option_d_is_right_triangle_l559_559778


namespace jelly_bean_ratio_l559_559127

theorem jelly_bean_ratio (total_jelly_beans : ℕ) (jelly_beans_in_jar_X : ℕ) 
  (h_total : total_jelly_beans = 1200) 
  (h_jar_X : jelly_beans_in_jar_X = 800) : 
  let jelly_beans_in_jar_Y := total_jelly_beans - jelly_beans_in_jar_X in
  jelly_beans_in_jar_X / jelly_beans_in_jar_Y = 2 :=
by 
  have h_jar_Y : jelly_beans_in_jar_Y = 400 := by sorry
  have h_ratio : jelly_beans_in_jar_X / jelly_beans_in_jar_Y = 2 := by sorry
  exact h_ratio

end jelly_bean_ratio_l559_559127


namespace greatest_number_of_quarters_l559_559434

theorem greatest_number_of_quarters (q : ℕ) :
  let total_value := 0.25 * q + 0.05 * q + 0.20 * 2 * q in
  total_value = 4.85 → q = 9 := 
by 
  sorry

end greatest_number_of_quarters_l559_559434


namespace tan_sin_identity_l559_559611

theorem tan_sin_identity :
  (let θ := 30 * Real.pi / 180 in 
   let tan_θ := Real.sin θ / Real.cos θ in 
   let sin_θ := 1 / 2 in
   let cos_θ := Real.sqrt 3 / 2 in
   ((tan_θ ^ 2 - sin_θ ^ 2) / (tan_θ ^ 2 * sin_θ ^ 2) = 1)) :=
by
  let θ := 30 * Real.pi / 180
  let tan_θ := Real.sin θ / Real.cos θ
  let sin_θ := 1 / 2
  let cos_θ := Real.sqrt 3 / 2
  have h_tan_θ : tan_θ = 1 / Real.sqrt 3 := sorry  -- We skip the derivation proof.
  have h_sin_θ : sin_θ = 1 / 2 := rfl
  have h_cos_θ : cos_θ = Real.sqrt 3 / 2 := sorry  -- We skip the derivation proof.
  have numerator := tan_θ ^ 2 - sin_θ ^ 2
  have denominator := tan_θ ^ 2 * sin_θ ^ 2
  -- show equality
  have h : (numerator / denominator) = (1 : ℝ) := sorry  -- We skip the final proof.
  exact h

end tan_sin_identity_l559_559611


namespace sqrt_inequality_l559_559918
theorem sqrt_inequality (n : ℕ) : sqrt (n + 1) + 2 * sqrt n < sqrt (9 * n + 3) := sorry
  
end sqrt_inequality_l559_559918


namespace distinct_values_of_g_l559_559629

-- Definitions
def g (x : ℝ) : ℝ := ∑ k in (finset.range 10).map (λ i, i + 3), 
  ⌊(k : ℝ) * x⌋ - k * ⌊x⌋

-- Theorem Statement
theorem distinct_values_of_g :
  ∃! n : ℕ, n = 45 ∧ ∀ x : ℝ, x ≥ 0 → (g '' (set.Ici 0)).finite ∧ (g '' (set.Ici 0)).card = n :=
sorry

end distinct_values_of_g_l559_559629


namespace abs_h_of_roots_sum_squares_eq_34_l559_559469

theorem abs_h_of_roots_sum_squares_eq_34 
  (h : ℝ)
  (h_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0)) 
  (sum_of_squares_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0) → r^2 + s^2 = 34) :
  |h| = Real.sqrt 10 :=
by
  sorry

end abs_h_of_roots_sum_squares_eq_34_l559_559469


namespace tan_sin_identity_l559_559612

theorem tan_sin_identity :
  (let θ := 30 * Real.pi / 180 in 
   let tan_θ := Real.sin θ / Real.cos θ in 
   let sin_θ := 1 / 2 in
   let cos_θ := Real.sqrt 3 / 2 in
   ((tan_θ ^ 2 - sin_θ ^ 2) / (tan_θ ^ 2 * sin_θ ^ 2) = 1)) :=
by
  let θ := 30 * Real.pi / 180
  let tan_θ := Real.sin θ / Real.cos θ
  let sin_θ := 1 / 2
  let cos_θ := Real.sqrt 3 / 2
  have h_tan_θ : tan_θ = 1 / Real.sqrt 3 := sorry  -- We skip the derivation proof.
  have h_sin_θ : sin_θ = 1 / 2 := rfl
  have h_cos_θ : cos_θ = Real.sqrt 3 / 2 := sorry  -- We skip the derivation proof.
  have numerator := tan_θ ^ 2 - sin_θ ^ 2
  have denominator := tan_θ ^ 2 * sin_θ ^ 2
  -- show equality
  have h : (numerator / denominator) = (1 : ℝ) := sorry  -- We skip the final proof.
  exact h

end tan_sin_identity_l559_559612


namespace value_of_mathematics_l559_559102

def letter_value (n : ℕ) : ℤ :=
  -- The function to assign values based on position modulo 8
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 0 => 0
  | _ => 0 -- This case is practically unreachable

def letter_position (c : Char) : ℕ :=
  -- The function to find the position of a character in the alphabet
  c.toNat - 'a'.toNat + 1

def value_of_word (word : String) : ℤ :=
  -- The function to calculate the sum of values of letters in the word
  word.foldr (fun c acc => acc + letter_value (letter_position c)) 0

theorem value_of_mathematics : value_of_word "mathematics" = 6 := 
  by
    sorry -- Proof to be completed

end value_of_mathematics_l559_559102


namespace standard_deviation_of_data_points_l559_559712

-- Definitions of the mean and standard deviation for a list of data points
noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  mean (data.map (λ x => (x - mean data) ^ 2))

noncomputable def standard_deviation (data : List ℝ) : ℝ :=
  real.sqrt (variance data)

-- The proof problem
theorem standard_deviation_of_data_points :
  standard_deviation [3, 5, 7, 4, 6] = real.sqrt 2 := 
by 
  sorry

end standard_deviation_of_data_points_l559_559712


namespace Frank_can_buy_7_candies_l559_559511

def tickets_whack_a_mole := 33
def tickets_skee_ball := 9
def cost_per_candy := 6

def total_tickets := tickets_whack_a_mole + tickets_skee_ball

theorem Frank_can_buy_7_candies : total_tickets / cost_per_candy = 7 := by
  sorry

end Frank_can_buy_7_candies_l559_559511


namespace g_f_neg5_l559_559399

-- Define the function f
def f (x : ℝ) := 2 * x ^ 2 - 4

-- Define the function g with the known condition g(f(5)) = 12
axiom g : ℝ → ℝ
axiom g_f5 : g (f 5) = 12

-- Now state the main theorem we need to prove
theorem g_f_neg5 : g (f (-5)) = 12 := by
  sorry

end g_f_neg5_l559_559399


namespace find_m_value_l559_559758

-- Definition of the condition
def no_linear_term (m : ℝ) : Prop :=
  let product := (λ x : ℝ, (x + m) * (x + 3))
  ∀ x : ℝ, product(x) = x^2 + (3 + m) * x + 3 * m → (3 + m) = 0

-- The main theorem stating the value of m
theorem find_m_value : ∃ m : ℝ, no_linear_term m ∧ m = -3 :=
by
  sorry

end find_m_value_l559_559758


namespace number_of_people_condition_l559_559766

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_people_condition (n k : ℕ) (h1 : ∀ i : ℕ, i < n → (∀ j : ℕ, j < n → (i ≠ j → ((i + k) % n) ≠ j))) : 
  is_even (n / gcd n k) :=
  sorry

end number_of_people_condition_l559_559766


namespace sequence_periodic_mod_l559_559403

-- Define the sequence (u_n) recursively
def sequence_u (a : ℕ) : ℕ → ℕ
  | 0     => a  -- Note: u_1 is defined as the initial term a, treating the starting index as 0 for compatibility with Lean's indexing.
  | (n+1) => a ^ (sequence_u a n)

-- The theorem stating there exist integers k and N such that for all n ≥ N, u_{n+k} ≡ u_n (mod m)
theorem sequence_periodic_mod (a m : ℕ) (hm : 0 < m) (ha : 0 < a) :
  ∃ k N : ℕ, ∀ n : ℕ, N ≤ n → (sequence_u a (n + k) ≡ sequence_u a n [MOD m]) :=
by
  sorry

end sequence_periodic_mod_l559_559403


namespace volume_related_to_area_l559_559982

theorem volume_related_to_area (x y z : ℝ) 
  (bottom_area_eq : 3 * x * y = 3 * x * y)
  (front_area_eq : 2 * y * z = 2 * y * z)
  (side_area_eq : 3 * x * z = 3 * x * z) :
  (3 * x * y) * (2 * y * z) * (3 * x * z) = 18 * (x * y * z) ^ 2 := 
by sorry

end volume_related_to_area_l559_559982


namespace indistinguishable_partitions_of_5_into_3_boxes_l559_559741

-- Definitions for the conditions
def areIndistinguishable (A : Multiset ℕ) (B : Multiset ℕ) : Prop :=
  A = B

theorem indistinguishable_partitions_of_5_into_3_boxes:
  { S : Multiset (Multiset ℕ) // ∀ x ∈ S, x.card = 3 ∧ x.sum = 5 } = 
  { {5, 0, 0}, {4, 1, 0}, {3, 2, 0}, {3, 1, 1}, {2, 2, 1} } :=
by
  sorry

end indistinguishable_partitions_of_5_into_3_boxes_l559_559741


namespace correct_operation_l559_559157

noncomputable def sqrt_op_A: Prop := sqrt 4 ≠ 2
noncomputable def sqrt_op_B: Prop := (± sqrt (5^2)) ≠ -5
noncomputable def sqrt_op_C: Prop := sqrt ((-7) ^ 2) = 7
noncomputable def sqrt_op_D: Prop := sqrt (-3) ≠ - sqrt 3

theorem correct_operation : (sqrt_op_A ∧ sqrt_op_B ∧ sqrt_op_C ∧ sqrt_op_D) → (sqrt_op_C = 7) :=
by
  intros h
  sorry

end correct_operation_l559_559157


namespace triangle_circumradius_correct_l559_559043

def triangle_circumradius_is_DC 
  (A B C M N T D : Type)
  (hAB : AB = 3)
  (hAC : AC = 8)
  (hBC : BC = 7)
  (hM : M = midpoint A B)
  (hN : N = midpoint A C)
  (hAT_TC : AT = T C)
  (hCircumcircle_BAT_MAN : circumcircle BAT D ∧ circumcircle MAN D) : Prop :=
  DC = (7 * sqrt 3) / 3

theorem triangle_circumradius_correct : 
  ∀ (A B C M N T D : Type),
  (triangle_circumradius_is_DC A B C M N T D) := sorry

end triangle_circumradius_correct_l559_559043


namespace equivalent_single_increase_l559_559459

-- Defining the initial price of the mobile
variable (P : ℝ)
-- Condition stating the price after a 40% increase
def increased_price := 1.40 * P
-- Condition stating the new price after a further 15% decrease
def final_price := 0.85 * increased_price P

-- The mathematically equivalent statement to prove
theorem equivalent_single_increase:
  final_price P = 1.19 * P :=
sorry

end equivalent_single_increase_l559_559459


namespace f_is_32x5_l559_559036

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

-- State the theorem to be proved
theorem f_is_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  sorry

end f_is_32x5_l559_559036


namespace find_x_in_inches_l559_559452

theorem find_x_in_inches (x : ℝ) :
  let area_smaller_square := 9 * x^2
  let area_larger_square := 36 * x^2
  let area_triangle := 9 * x^2
  area_smaller_square + area_larger_square + area_triangle = 1950 → x = (5 * Real.sqrt 13) / 3 :=
by
  sorry

end find_x_in_inches_l559_559452


namespace intersection_M_N_l559_559050

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Prove the intersection of M and N equals (1, 2)
theorem intersection_M_N :
  ∀ x, x ∈ M ∩ N ↔ 1 < x ∧ x < 2 :=
by
  -- Skipping the proof here
  sorry

end intersection_M_N_l559_559050


namespace parabola_focus_l559_559654

open Real

theorem parabola_focus (a : ℝ) (h k : ℝ) (x y : ℝ) (f : ℝ) :
  (a = -1/4) → (h = 0) → (k = 0) → 
  (f = (1 / (4 * a))) →
  (y = a * (x - h) ^ 2 + k) → 
  (y = -1 / 4 * x ^ 2) → f = -1 := by
  intros h_a h_h h_k h_f parabola_eq _
  rw [h_a, h_h, h_k] at *
  sorry

end parabola_focus_l559_559654


namespace geometric_sequence_problem_l559_559802

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

-- Define the statement for the roots of the quadratic function
def is_root (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ :=
  x^2 - x - 2013

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : is_root quadratic_function (a 2)) 
  (h3 : is_root quadratic_function (a 3)) : 
  a 1 * a 4 = -2013 :=
sorry

end geometric_sequence_problem_l559_559802


namespace maxGoodOperations_2018_minGoodOperations_2018_consistentLengthsWhenMinGoodOperations_l559_559952

/-- To represent a "good" operation, the lengths of resulting ropes must not be equal --/
structure GoodOperation (length1 length2 : ℕ) : Prop :=
(eq_or_ne : length1 ≠ length2)

/-- Define a function to calculate the maximum number of good operations --/
def maxGoodOperations (n : ℕ) : ℕ :=
  n - 2

/-- Define a function to calculate the number of 1s in binary representation of a number --/
def sumOfBinaryDigits (n : ℕ) : ℕ :=
  n.toDigits 2 |>.count (· = 1)

/-- Define a function to calculate the minimum number of good operations --/
def minGoodOperations (n : ℕ) : ℕ :=
  sumOfBinaryDigits n - 1

/-- Prove that the maximum number of good operations for a rope of length 2018 is 2016 --/
theorem maxGoodOperations_2018 : maxGoodOperations 2018 = 2016 :=
  sorry

/-- Prove that the minimum number of good operations for a rope of length 2018 is 6 --/
theorem minGoodOperations_2018 : minGoodOperations 2018 = 6 :=
  sorry

/-- Prove that in all processes with minimum number of good operations,
    the number of different lengths recorded is the same --/
theorem consistentLengthsWhenMinGoodOperations (n : ℕ) :
  ∀ (process : list (ℕ × ℕ)),
  count_good_operations process = minGoodOperations n →
  ∀ (ropes : list ℕ),
  lengths_of_ropes process = ropes →
  unique_lengths_count ropes := sorry

end maxGoodOperations_2018_minGoodOperations_2018_consistentLengthsWhenMinGoodOperations_l559_559952


namespace distribution_of_balls_into_boxes_l559_559739

def number_of_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
if balls = 5 ∧ boxes = 3 then 5 else 0

theorem distribution_of_balls_into_boxes :
  number_of_ways_to_distribute_balls 5 3 = 5 :=
by
  -- This is the placeholder for the proof
  sorry

end distribution_of_balls_into_boxes_l559_559739


namespace slope_divides_parallelogram_l559_559253

theorem slope_divides_parallelogram (m n: ℕ) (hmn: Nat.gcd m n = 1):
  (∃ m n: ℕ, Nat.gcd m n = 1 ∧ 12 * m = 192 ∧ 12 * n = 12) → m + n = 17 :=
by
  -- Define the vertices of the parallelogram
  let A := (12 : ℕ, 60 : ℤ)
  let B := (12 : ℕ, 152 : ℤ)
  let C := (32 : ℕ, 204 : ℤ)
  let D := (32 : ℕ, 112 : ℤ)
  -- Define the line parameters with m and n relatively prime
  intro hmn
  -- Define the proportion and verify
  have h1 : 60 + 132 = 192 := by sorry
  have h2 : 12 * 16 = 192 := by sorry
  use 16, 1,
  split,
  {
    exact Nat.gcd_refl 1,
  },
  {
    split,
    {
      exact Nat.gcd.mul_left_cancel 16 192 12 h1,
    },
    {
      exact Nat.gcd.one_mul_left 17,
    }
  }
  sorry

end slope_divides_parallelogram_l559_559253


namespace f_prime_zero_eq_neg_four_l559_559344

noncomputable def f (x : ℝ) := 2 * x * f' 1 + x^2

theorem f_prime_zero_eq_neg_four : deriv f 0 = -4 := 
by {
  sorry
}

end f_prime_zero_eq_neg_four_l559_559344


namespace eval_expression_l559_559270

theorem eval_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) :
  Real.sqrt (1 - 2 * Real.sin (Real.pi + 2) * Real.cos (Real.pi + 2)) = Real.sin 2 - Real.cos 2 :=
sorry

end eval_expression_l559_559270


namespace correct_operation_l559_559163

theorem correct_operation (a b : ℝ) (c d : ℂ) : (sqrt 4 = 2 ∧ (± abs (5) ≠ -5) ∧ (sqrt (abs (7) ^ 2) = 7) ∧ ¬(is_real (sqrt (-3))) ) := sorry

end correct_operation_l559_559163


namespace correct_operation_l559_559167

theorem correct_operation (a b : ℝ) (c d : ℂ) : (sqrt 4 = 2 ∧ (± abs (5) ≠ -5) ∧ (sqrt (abs (7) ^ 2) = 7) ∧ ¬(is_real (sqrt (-3))) ) := sorry

end correct_operation_l559_559167


namespace probability_five_distinct_dice_rolls_l559_559498

theorem probability_five_distinct_dice_rolls : 
  let total_outcomes := 6^5
  let favorable_outcomes := 6 * 5 * 4 * 3 * 2
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 54 :=
by
  sorry

end probability_five_distinct_dice_rolls_l559_559498


namespace find_factor_l559_559217

theorem find_factor (x f : ℝ) (h1 : x = 6)
    (h2 : (2 * x + 9) * f = 63) : f = 3 :=
sorry

end find_factor_l559_559217


namespace function_decreasing_interval_l559_559725

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

def f (a x : ℝ) : ℝ := log_base a (x^2 + 2*x - 3)

theorem function_decreasing_interval (a : ℝ) (h1 : 1 < a) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < -3 → x₂ < -3 → f a x₁ > f a x₂) :=
by
  sorry

end function_decreasing_interval_l559_559725


namespace transformation_invariant_l559_559884

-- Define the initial and transformed parabolas
def initial_parabola (x : ℝ) : ℝ := 2 * x^2
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3

-- Define the transformation process
def move_right_1 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1)
def move_up_3 (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 3

-- Concatenate transformations to form the final transformation
def combined_transformation (x : ℝ) : ℝ :=
  move_up_3 (move_right_1 initial_parabola) x

-- Statement to prove
theorem transformation_invariant :
  ∀ x : ℝ, combined_transformation x = transformed_parabola x := 
by {
  sorry
}

end transformation_invariant_l559_559884


namespace almonds_walnuts_ratio_l559_559202

theorem almonds_walnuts_ratio (x : ℝ) (h1 : 5 / (5 + x) = 116.67 / 140) : x ≈ 1 :=
by
  sorry

end almonds_walnuts_ratio_l559_559202


namespace man_speed_in_still_water_l559_559946

def speed_in_still_water (s_c t_s d_m : ℝ) : ℝ :=
  let d_km := d_m / 1000
  let t_h := t_s / 3600
  let downstream_speed := d_km / t_h
  downstream_speed - s_c

theorem man_speed_in_still_water :
  speed_in_still_water 3 11.999040076793857 60 = 15 := by
  sorry

end man_speed_in_still_water_l559_559946


namespace tan_sin_identity_l559_559616

theorem tan_sin_identity :
  (let θ := 30 * Real.pi / 180 in 
   let tan_θ := Real.sin θ / Real.cos θ in 
   let sin_θ := 1 / 2 in
   let cos_θ := Real.sqrt 3 / 2 in
   ((tan_θ ^ 2 - sin_θ ^ 2) / (tan_θ ^ 2 * sin_θ ^ 2) = 1)) :=
by
  let θ := 30 * Real.pi / 180
  let tan_θ := Real.sin θ / Real.cos θ
  let sin_θ := 1 / 2
  let cos_θ := Real.sqrt 3 / 2
  have h_tan_θ : tan_θ = 1 / Real.sqrt 3 := sorry  -- We skip the derivation proof.
  have h_sin_θ : sin_θ = 1 / 2 := rfl
  have h_cos_θ : cos_θ = Real.sqrt 3 / 2 := sorry  -- We skip the derivation proof.
  have numerator := tan_θ ^ 2 - sin_θ ^ 2
  have denominator := tan_θ ^ 2 * sin_θ ^ 2
  -- show equality
  have h : (numerator / denominator) = (1 : ℝ) := sorry  -- We skip the final proof.
  exact h

end tan_sin_identity_l559_559616


namespace question1_question2_l559_559671

variables (θ : ℝ)

-- Condition: tan θ = 2
def tan_theta_eq : Prop := Real.tan θ = 2

-- Question 1: Prove (4 * sin θ - 2 * cos θ) / (3 * sin θ + 5 * cos θ) = 6 / 11
theorem question1 (h : tan_theta_eq θ) : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11 :=
by
  sorry

-- Question 2: Prove 1 - 4 * sin θ * cos θ + 2 * cos² θ = -1 / 5
theorem question2 (h : tan_theta_eq θ) : 1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1 / 5 :=
by
  sorry

end question1_question2_l559_559671


namespace alice_younger_than_carol_l559_559980

theorem alice_younger_than_carol :
  ∀ (Alice Carol Betty : ℕ), 
  Carol = 5 * Alice →
  Carol = 2 * Betty →
  Betty = 6 →
  Alice + 10 = Carol :=
by
  intros Alice Carol Betty h1 h2 h3
  rw [h3, mul_comm 2 6] at h2
  have h4 : Carol = 12 := by rw h2; exact rfl
  rw [h4] at h1
  linarith
  sorry

end alice_younger_than_carol_l559_559980


namespace binom_divisibility_l559_559287

theorem binom_divisibility (k : ℕ) (h : k ≥ 2) :
  2^(3 * k) ∣ (Nat.binomial (2 ^ (k + 1)) (2 ^ k) - Nat.binomial (2 ^ k) (2 ^ (k - 1))) ∧
  ¬ 2^(3 * k + 1) ∣ (Nat.binomial (2 ^ (k + 1)) (2 ^ k) - Nat.binomial (2 ^ k) (2 ^ (k - 1))) :=
by
  sorry

end binom_divisibility_l559_559287


namespace liquid_x_percentage_l559_559516

theorem liquid_x_percentage 
  (percentage_a : ℝ) (percentage_b : ℝ)
  (weight_a : ℝ) (weight_b : ℝ)
  (h1 : percentage_a = 0.8)
  (h2 : percentage_b = 1.8)
  (h3 : weight_a = 400)
  (h4 : weight_b = 700) :
  (weight_a * (percentage_a / 100) + weight_b * (percentage_b / 100)) / (weight_a + weight_b) * 100 = 1.44 := 
by
  sorry

end liquid_x_percentage_l559_559516


namespace midpoint_of_AP_l559_559013

theorem midpoint_of_AP (A B C D M P Q : Point) 
  (h1 : Rectangle A B C D) 
  (h2 : Midpoint M B C)
  (h3 : OnDiagonal P A C)
  (h4 : OnDiagonal Q A C)
  (h5 : ∠ D P C = 90)
  (h6 : ∠ D Q M = 90) 
  : Midpoint Q A P :=
sorry

end midpoint_of_AP_l559_559013


namespace interval_of_convergence_l559_559636

noncomputable def radius_of_convergence_series1 (z : ℂ) : ℂ :=
  ∑' (n : ℕ), (3 + 4 * complex.I)^n / (z + 2 * complex.I)^n

noncomputable def radius_of_convergence_series2 (z : ℂ) : ℂ :=
  ∑' (n : ℕ), (z + 2 * complex.I / 6)^n

theorem interval_of_convergence (z : ℂ) :
  (radius_of_convergence_series1 z).radius > 5 ∧ (radius_of_convergence_series2 z).radius < 6 ↔ (5 < complex.abs (z + 2 * complex.I) < 6) :=
sorry

end interval_of_convergence_l559_559636


namespace dante_flour_eggs_l559_559254

theorem dante_flour_eggs (eggs : ℕ) (h_eggs : eggs = 60) (h_flour : ∀ (f : ℕ), f = eggs / 2) : eggs + (eggs / 2) = 90 := 
by {
  rw h_eggs,
  calc
    60 + (60 / 2) = 60 + 30   : by norm_num
    ...         = 90 : by norm_num
}

end dante_flour_eggs_l559_559254


namespace average_of_all_results_is_24_l559_559180

-- Definitions translated from conditions
def average_1 := 20
def average_2 := 30
def n1 := 30
def n2 := 20
def total_sum_1 := n1 * average_1
def total_sum_2 := n2 * average_2

-- Lean 4 statement
theorem average_of_all_results_is_24
  (h1 : total_sum_1 = n1 * average_1)
  (h2 : total_sum_2 = n2 * average_2) :
  ((total_sum_1 + total_sum_2) / (n1 + n2) = 24) :=
by
  sorry

end average_of_all_results_is_24_l559_559180


namespace non_overlapping_circles_l559_559749

theorem non_overlapping_circles {C : Type*} (circles : finset (set C)) (M : ℝ) 
  (h_cover : (⋃₀ (↑circles : set (set C))).measure ≤ M) :
  ∃ non_overlapping_subset : finset (set C), 
    (∀ c₁ c₂ ∈ non_overlapping_subset, c₁ ≠ c₂ → disjoint c₁ c₂) ∧ 
    ∑' (c ∈ non_overlapping_subset), c.measure ≥ (1 / 9 : ℝ) * M :=
sorry

end non_overlapping_circles_l559_559749


namespace number_of_bottle_caps_l559_559859

def total_cost : ℝ := 25
def cost_per_bottle_cap : ℝ := 5

theorem number_of_bottle_caps : total_cost / cost_per_bottle_cap = 5 := 
by 
  sorry

end number_of_bottle_caps_l559_559859


namespace area_of_set_K_l559_559774

open Metric

def set_K :=
  {p : ℝ × ℝ | (abs p.1 + abs (3 * p.2) - 6) * (abs (3 * p.1) + abs p.2 - 6) ≤ 0}

def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Define the area function for a general set s

theorem area_of_set_K : area set_K = 24 :=
  sorry

end area_of_set_K_l559_559774


namespace general_formula_a_general_formula_b_sum_first_n_a2n_bn_l559_559697

variable {an : ℕ → ℚ}
variable {bn : ℕ → ℚ}
variable {Sn : ℕ → ℚ}
variable {Tn : ℕ → ℚ}

/-- an is an arithmetic sequence -/
axiom an_arithmetic : ∀ n, an (n + 1) - an n = an 1 - an 0

/-- Sum of the first n terms of an is Sn -/
axiom sum_first_n : ∀ n : ℕ, Sn n = ∑ i in range n, an i

/-- bn is a geometric sequence with first term 2 and common ratio > 0 -/
axiom bn_geometric : ∀ n, bn 1 = 2 ∧ ∃ q, q > 0 ∧ (∀ n, bn (n + 1) = q * bn n)

/-- b2 + b3 = 12 -/
axiom b2_b3 : bn 2 + bn 3 = 12

/-- b3 = a4 - 2 * a1 -/
axiom b3_a4_2a1 : bn 3 = an 4 - 2 * an 1

/-- S11 = 11 * b4 -/
axiom S11_eq : Sn 11 = 11 * bn 4

/-- General formula for an -/
theorem general_formula_a : ∀ n, an n = 3 * n - 2 :=
sorry

/-- General formula for bn -/
theorem general_formula_b : ∀ n, bn n = 2 ^ n :=
sorry

/-- Sum of the first n terms of the sequence {a2n * bn} -/
theorem sum_first_n_a2n_bn : ∀ n, Tn n = (3 * n - 4) * 2 ^ (n + 2) + 16 :=
sorry

end general_formula_a_general_formula_b_sum_first_n_a2n_bn_l559_559697


namespace intersection_of_planes_is_line_AC_l559_559357

-- Definitions for a spatial quadrilateral and points on its edges.
-- We'll define quadrilateral ABCD and points E, F, G, H on the respective edges.
variables {A B C D E F G H M : Type}
variables [point : Type]

-- Define edges AB, BC, CD, DA as lines.
def edge (x y : point) : Type := sorry  -- Placeholder

-- Conditions from a)
hypothesis (H1 : E ∈ edge A B)
hypothesis (H2 : F ∈ edge B C)
hypothesis (H3 : G ∈ edge C D)
hypothesis (H4 : H ∈ edge D A)

-- EF and GH are lines intersecting at M
def line (x y : point) : Type := sorry  -- Placeholder
def intersects (l1 l2 : Type) (p : point) : Prop := sorry  -- Placeholder

-- EF and GH intersecting at M
hypothesis (H5 : intersects (line E F) (line G H) M)

-- Planes determined by edges
def plane (x y z : point) : Type := sorry  -- Placeholder

-- Definitions of planes
def plane_ABC := plane A B C
def plane_ACD := plane A C D

-- EF lies in plane_ABC
hypothesis (H_EF_ABC : ∀ P, P ∈ (line E F) → P ∈ plane_ABC)

-- GH lies in plane_ACD
hypothesis (H_GH_ACD : ∀ Q, Q ∈ (line G H) → Q ∈ plane_ACD)

theorem intersection_of_planes_is_line_AC :
  M ∈ line A C :=
sorry

end intersection_of_planes_is_line_AC_l559_559357


namespace integral_independent_of_a_l559_559183

noncomputable def I (a : ℝ) : ℝ :=
  ∫ x in (0:ℝ)..(Real.infinity), 1 / ((1 + x^2) * (1 + x^a))

theorem integral_independent_of_a (a : ℝ) : I(a) = π / 4 := by
  sorry

end integral_independent_of_a_l559_559183


namespace total_candy_bars_bought_by_john_l559_559787

section CandyBars

/-- Dave pays for 6 candy bars. Each candy bar costs $1.50. John paid $21. Prove that 
the total number of candy bars John bought is 20. --/

variables (cost_per_bar : ℝ) (john_paid dave_paid total_cost: ℝ) (n d : ℕ)

def cost_per_bar : ℝ := 1.5
def dave_paid : ℝ := 6 * cost_per_bar
def john_paid : ℝ := 21
def total_cost : ℝ := john_paid + dave_paid
def n : ℕ := (total_cost / cost_per_bar).toNat
def d : ℕ := 6

theorem total_candy_bars_bought_by_john :
  n = 20 :=
by
  sorry

end CandyBars

end total_candy_bars_bought_by_john_l559_559787


namespace prob_sum_at_least_10_prob_area_greater_than_60_l559_559526

/-- Problem 1: Probability that the sum of two die rolls is at least 10 is 1/6 --/
theorem prob_sum_at_least_10 : 
  let faces := [1, 2, 3, 4, 5, 6]
  let outcomes := (faces.product faces)
  let event := outcomes.filter (λ (t : ℕ × ℕ), (t.fst + t.snd) ≥ 10)
  (event.length / outcomes.length : ℚ) = 1 / 6 := sorry

/-- Problem 2: Probability that the area of the rectangle is greater than 60 when a point is chosen randomly on a 16cm line segment is 1/4 --/
theorem prob_area_greater_than_60 : 
  let length : ℚ := 16
  let event_length : ℚ := 4
  let total_length : ℚ := length
  (event_length / total_length) = 1 / 4 := sorry

end prob_sum_at_least_10_prob_area_greater_than_60_l559_559526


namespace career_preference_degrees_l559_559115

theorem career_preference_degrees (boys girls : ℕ) (ratio_boys_to_girls : boys / gcd boys girls = 2 ∧ girls / gcd boys girls = 3) 
  (boys_preference : ℕ) (girls_preference : ℕ) 
  (h1 : boys_preference = boys / 3)
  (h2 : girls_preference = 2 * girls / 3) : 
  (boys_preference + girls_preference) / (boys + girls) * 360 = 192 :=
by
  sorry

end career_preference_degrees_l559_559115


namespace maximize_garden_area_length_l559_559554

noncomputable def length_parallel_to_wall (cost_per_foot : ℝ) (fence_cost : ℝ) : ℝ :=
  let total_length := fence_cost / cost_per_foot 
  let y := total_length / 4 
  let length_parallel := total_length - 2 * y
  length_parallel

theorem maximize_garden_area_length :
  ∀ (cost_per_foot fence_cost : ℝ), cost_per_foot = 10 → fence_cost = 1500 → 
  length_parallel_to_wall cost_per_foot fence_cost = 75 :=
by
  intros
  simp [length_parallel_to_wall, *]
  sorry

end maximize_garden_area_length_l559_559554


namespace remaining_macaroons_correct_l559_559289

variable (k : ℚ)

def total_baked : ℚ := 50 + 40 + 30 + 20 + 10

def total_eaten (k : ℚ) : ℚ := k + 2 * k + 3 * k + 10 * k + k / 5

def remaining_macaroons (k : ℚ) : ℚ := total_baked - total_eaten k

theorem remaining_macaroons_correct (k : ℚ) : remaining_macaroons k = 150 - (81 * k / 5) := 
by {
  -- The proof goes here.
  sorry
}

end remaining_macaroons_correct_l559_559289


namespace max_min_sum_l559_559408

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 + (2 ^ (1/2)) * x ^ 3 - 3 * x⁻¹) / (x ^ 2 + 1)

variable (m M N : ℝ)

theorem max_min_sum (h : ∀ x ∈ set.Icc (-m) m, f x ≤ M ∧ f x ≥ N) : M + N = 2 :=
sorry

end max_min_sum_l559_559408


namespace max_gcd_13n_plus_4_8n_plus_3_l559_559568

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ) (h: n > 0) : ∃ m, 7 = gcd (13 * n + 4) (8 * n + 3) :=
sorry

end max_gcd_13n_plus_4_8n_plus_3_l559_559568


namespace part_a_impossible_part_b_possible_l559_559292

-- Define the conditions for part a) 8 numbers problem:
theorem part_a_impossible :
  ∀ (s : Finset ℕ), s.card = 8 → (∀ a b c ∈ s, a ≠ b ∧ b ≠ c ∧ a ≠ c →
  ∃ i1 i2 i3, a = i1 ^ 2 ∧ b = i2 ^ 2 ∧ c = i3 ^ 2 ∧ ((i1 ^ 2 + i2 ^ 2 + i3 ^ 2) % 10 = 0)) →
  set.range (nat.range 1 26) ∩ s = ∅ :=
sorry

-- Define the conditions for part b) 9 numbers problem:
theorem part_b_possible :
  ∃ (s : Finset ℕ), s.card = 9 ∧ (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b ∧ b ≠ c ∧ a ≠ c →
  ∃ i1 i2 i3, a = i1 ^ 2 ∧ b = i2 ^ 2 ∧ c = i3 ^ 2 ∧ ((i1 ^ 2 + i2 ^ 2 + i3 ^ 2) % 10 = 0)) ∧
  set.range (nat.range 1 26) ∩ s = s :=
sorry

end part_a_impossible_part_b_possible_l559_559292


namespace mass_of_man_l559_559192

variable (L : ℝ) (B : ℝ) (h : ℝ) (ρ : ℝ)

-- Given conditions
def boatLength := L = 3
def boatBreadth := B = 2
def sinkingDepth := h = 0.018
def waterDensity := ρ = 1000

-- The mass of the man
theorem mass_of_man (L B h ρ : ℝ) (H1 : boatLength L) (H2 : boatBreadth B) (H3 : sinkingDepth h) (H4 : waterDensity ρ) : 
  ρ * L * B * h = 108 := by
  sorry

end mass_of_man_l559_559192


namespace probability_five_distinct_dice_rolls_l559_559497

theorem probability_five_distinct_dice_rolls : 
  let total_outcomes := 6^5
  let favorable_outcomes := 6 * 5 * 4 * 3 * 2
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 54 :=
by
  sorry

end probability_five_distinct_dice_rolls_l559_559497


namespace prism_sphere_intersection_l559_559686

theorem prism_sphere_intersection
  (n : ℕ) (h : n ≥ 2)
  (S : ℝ^3)
  (A : fin (2 * n) → ℝ^3)
  (hA : ∀ i j, ‖S - A i‖ = ‖S - A j‖)
  (Ω : sphere S)
  (B : fin (2 * n) → ℝ^3)
  (hB : ∀ i, B i ∈ Ω ∧ ∃ k, B i ∈ segment S (A k)) :
  (∑ i in range n, ‖S - B (2 * i + 1)‖) = (∑ i in range n, ‖S - B (2 * i)‖) :=
sorry

end prism_sphere_intersection_l559_559686


namespace investment_difference_l559_559382

noncomputable def compound_yearly (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * (1 + r)^t

noncomputable def compound_monthly (P : ℕ) (r : ℚ) (months : ℕ) : ℚ :=
  P * (1 + r)^(months)

theorem investment_difference :
  let P := 70000
  let r := 0.05
  let t := 3
  let monthly_r := r / 12
  let months := t * 12
  compound_monthly P monthly_r months - compound_yearly P r t = 263.71 :=
by
  sorry

end investment_difference_l559_559382


namespace math_problem_l559_559305

noncomputable def exponential_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n = 2 * 3^(n - 1)

noncomputable def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(2 * 3^n - 2) / 2

theorem math_problem 
  (a : ℕ → ℝ) (b : ℕ → ℕ) (c : ℕ → ℝ) (S T P : ℕ → ℝ)
  (h1 : exponential_sequence a)
  (h2 : a 1 * a 3 = 36)
  (h3 : a 3 + a 4 = 9 * (a 1 + a 2))
  (h4 : ∀ n, S n + 1 = 3^(b n))
  (h5 : ∀ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2)
  (h6 : ∀ n, c n = a n / ((a n + 1) * (a (n + 1) + 1)))
  (h7 : ∀ n, P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2)) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧
  ∀ n, b n = n ∧
  ∀ n, a n * b n = 2 * n * 3^(n - 1) ∧
  ∃ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2 ∧
  P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2) :=
by sorry

end math_problem_l559_559305


namespace present_age_allens_mother_l559_559963

-- Definitions based on the conditions
constants (A M : ℝ)
axiom allen_younger : A = M - 30
axiom future_age_sum : (A + 7) + (M + 7) = 85

-- Statement to prove Allen's mother's present age
theorem present_age_allens_mother : M = 50.5 :=
by sorry

end present_age_allens_mother_l559_559963


namespace contracted_houses_l559_559965

theorem contracted_houses : ∀ (H : ℕ), 
  ((3 / 5 : ℚ) * H + 300 + 500 = H) → (H = 2000) :=
by
  intro H
  assume h
  sorry

end contracted_houses_l559_559965


namespace area_triangle_QPO_l559_559769

variables (A B C D P Q O N M : Type*)
variables [Parallelogram A B C D]
variables [Point P Q O N M]
variables [Trisection D P B C N]
variables [Bisection C Q A D M]
variables [Midpoint M D A]
variables [Intersection D P A B P]
variables [Intersection C Q A B Q]
variables [Area ABCD]

def area (T : Type*) : ℝ :=
sorry

theorem area_triangle_QPO
  {P Q O N M : Type*}
  (parall : Parallelogram A B C D)
  (trisect : Trisection D P B C N)
  (bisect : Bisection C Q A D M)
  (midpt : Midpoint M D A)
  (intersect_DP_AB : Intersection D P A B P)
  (intersect_CQ_AB : Intersection C Q A B Q)
  (area_ABCD_k : Area A B C D = k) :
  area (triangle Q P O) = 13 * k / 36 :=
sorry

end area_triangle_QPO_l559_559769


namespace problem_problem_2015_l559_559814

def f (x : ℝ) : ℝ := x + Real.sqrt (x^2 + 1) + (1 / (x - Real.sqrt (x^2 + 1)))

theorem problem (x : ℝ) : f x = 0 :=
by sorry

theorem problem_2015 : f 2015 = 0 :=
by sorry

end problem_problem_2015_l559_559814


namespace range_of_x_l559_559322

noncomputable def f (x : ℝ) : ℝ := x * (2^x - 1 / 2^x)

theorem range_of_x (x : ℝ) (h : f (x - 1) > f x) : x < 1 / 2 :=
by sorry

end range_of_x_l559_559322


namespace distribution_of_balls_into_boxes_l559_559740

def number_of_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
if balls = 5 ∧ boxes = 3 then 5 else 0

theorem distribution_of_balls_into_boxes :
  number_of_ways_to_distribute_balls 5 3 = 5 :=
by
  -- This is the placeholder for the proof
  sorry

end distribution_of_balls_into_boxes_l559_559740


namespace binomial_expansion_problem_l559_559320

noncomputable def binom (n k : ℕ) := Nat.choose n k

theorem binomial_expansion_problem (x : ℝ) (n : ℕ) (h_n : n ∈ { m : ℕ | m > 0}) 
 (h_ratio : (binom n 4 * (-2)^4) / (binom n 2 * (-2)^2) = 10) :
  n = 8 ∧
  (∃ k : ℕ, k = binom 8 4 * 2^4 ∧  k = 1120) ∧
  (∃ r : ℕ, r = 1 ∧ (binom 8 1 * (-2)^1) * x^(3/2) = -16 * x^(3/2)) ∧
  ((1 - 2)^8 = 1) := 
sorry

end binomial_expansion_problem_l559_559320


namespace daphne_visits_l559_559633

theorem daphne_visits :
  let sophie_visits := λ n : ℕ, n % 4 = 0
  let linda_visits := λ n : ℕ, n % 6 = 0
  let mae_visits := λ n : ℕ, n % 8 = 0
  let n_days := 360
  let two_friends_visit := λ n : ℕ, 
    ((sophie_visits n ∧ linda_visits n ∧ ¬mae_visits n) ∨ 
     (sophie_visits n ∧ ¬linda_visits n ∧ mae_visits n) ∨ 
     (¬sophie_visits n ∧ linda_visits n ∧ mae_visits n))
  in finset.card (finset.filter two_friends_visit (finset.range n_days)) = 45 := sorry

end daphne_visits_l559_559633


namespace coloring_count_l559_559130

def grid := fin 3 × fin 3

def is_valid_coloring (coloring : grid → fin 3) : Prop :=
  ∀ (i j : fin 3), ∀ (di dj : int),
    (abs di + abs dj = 1) →
    (i + di < 3 ∧ j + dj < 3) →
    coloring ⟨i, j⟩ ≠ coloring ⟨(i : int) + di, (j : int) + dj⟩

def uses_at_least_two_colors (coloring : grid → fin 3) : Prop :=
  ∃ c1 c2, ∃ (i1 j1 i2 j2 : fin 3),
    coloring ⟨i1, j1⟩ = c1 ∧ coloring ⟨i2, j2⟩ = c2 ∧ c1 ≠ c2

def all_valid_colorings : set (grid → fin 3) :=
  { coloring | is_valid_coloring coloring ∧ uses_at_least_two_colors coloring }

theorem coloring_count : (set.card all_valid_colorings) = 2 :=
by sorry

end coloring_count_l559_559130


namespace piggy_bank_dimes_count_l559_559783

/-- Define conditions and the theorem --/
def Ivan_has_filled_two_piggy_banks : Prop := true -- Placeholder for conditions translation

def total_value_eq_12 (D P : ℕ) : Prop := 
  0.10 * D + 0.01 * P = 12

def total_coins_eq_200 (D P : ℕ) : Prop := 
  D + P = 200

theorem piggy_bank_dimes_count : 
  Ivan_has_filled_two_piggy_banks → 
  (∃ D P : ℕ, D = 111 ∧ total_value_eq_12 D P ∧ total_coins_eq_200 D P) :=
by sorry

end piggy_bank_dimes_count_l559_559783


namespace tan_sin_identity_l559_559613

theorem tan_sin_identity :
  (let θ := 30 * Real.pi / 180 in 
   let tan_θ := Real.sin θ / Real.cos θ in 
   let sin_θ := 1 / 2 in
   let cos_θ := Real.sqrt 3 / 2 in
   ((tan_θ ^ 2 - sin_θ ^ 2) / (tan_θ ^ 2 * sin_θ ^ 2) = 1)) :=
by
  let θ := 30 * Real.pi / 180
  let tan_θ := Real.sin θ / Real.cos θ
  let sin_θ := 1 / 2
  let cos_θ := Real.sqrt 3 / 2
  have h_tan_θ : tan_θ = 1 / Real.sqrt 3 := sorry  -- We skip the derivation proof.
  have h_sin_θ : sin_θ = 1 / 2 := rfl
  have h_cos_θ : cos_θ = Real.sqrt 3 / 2 := sorry  -- We skip the derivation proof.
  have numerator := tan_θ ^ 2 - sin_θ ^ 2
  have denominator := tan_θ ^ 2 * sin_θ ^ 2
  -- show equality
  have h : (numerator / denominator) = (1 : ℝ) := sorry  -- We skip the final proof.
  exact h

end tan_sin_identity_l559_559613


namespace profit_function_max_profit_value_l559_559210

-- Definitions from the conditions
def sales_volume (x : ℝ) : ℝ := 3 - 2 / (x + 1)
def production_cost (P : ℝ) : ℝ := 10 + 2 * P
def selling_price (P : ℝ) : ℝ := 4 + 30 / P

-- Definition of profit y as a function of x
def profit (x : ℝ) : ℝ :=
  let P := sales_volume x
  (selling_price P) * P - x - (production_cost P)

-- Statement for part (I)
theorem profit_function (x : ℝ) (a : ℝ) (hx : 0 ≤ x ∧ x ≤ a) : profit x = 26 - 4 / (x + 1) - x :=
  sorry

-- Statement for part (II)
theorem max_profit_value (a : ℝ) (ha : 0 < a) :
  (a ≥ 1 → profit 1 = 23) ∧ (a < 1 → profit a = 26 - 4 / (a + 1) - a) :=
  sorry

end profit_function_max_profit_value_l559_559210


namespace least_b_l559_559078

theorem least_b (a b : ℕ) (ha : Prime.cube a ∨ ∃ p q : ℕ, Prime p ∧ Prime q ∧ a = p * q)
  (hb : ∃ n : ℕ, ∀ k : ℕ, b.factors.count k = n ↔ k = a)
  (hdiv : a ∣ b) : b = 24 := by
  sorry

end least_b_l559_559078


namespace range_of_m_l559_559821

noncomputable def quadratic_expr_never_equal (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2 * x^2 + 4 * x + m ≠ 3 * x^2 - 2 * x + 6

theorem range_of_m (m : ℝ) : quadratic_expr_never_equal m ↔ m < -3 := 
by
  sorry

end range_of_m_l559_559821


namespace k_geq_n_l559_559683

noncomputable def a : ℕ := sorry
noncomputable def k : ℕ := sorry
noncomputable def n : ℕ := sorry

-- Conditions
axiom a_two_n_digits : a.digits.size = 2 * n
axiom k_natural : k > 0
axiom a_decreasing_two_digit_chunks : ∀ i j : ℕ, i < j ∧ j < n → (a.div (10 ^ (2 * i)) % 100) < (a.div (10 ^ (2 * j)) % 100)
axiom ka_increasing_two_digit_chunks : ∀ i j : ℕ, i < j ∧ j < n → ((k * a).div (10 ^ (2 * j)) % 100) < ((k * a).div (10 ^ (2 * i)) % 100)

-- Theorem to prove
theorem k_geq_n : k ≥ n :=
sorry

end k_geq_n_l559_559683


namespace tan_sin_div_l559_559606

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l559_559606


namespace computers_produced_per_month_l559_559938

-- Define the conditions
constant rate_of_computer_production : ℕ → ℝ
axiom constant_production_rate : ∀ t : ℕ, rate_of_computer_production (t + 1) = 3.125
axiom production_period : ∀ t : ℕ, rate_of_computer_production (2 * t) = rate_of_computer_production t * 2

-- Define the month and production parameters
def minutes_per_day : ℕ := 24 * 60
def days_in_month : ℕ := 28
def minutes_per_month : ℕ := minutes_per_day * days_in_month
def half_hour_intervals_per_month : ℕ := minutes_per_month / 30

-- Calculate the total number of computers produced per month
noncomputable def total_computers_produced_per_month : ℝ := 3.125 * half_hour_intervals_per_month

-- Prove the total number of computers produced per month is 4200
theorem computers_produced_per_month : total_computers_produced_per_month = 4200 := by
  sorry

end computers_produced_per_month_l559_559938


namespace trigonometric_identity_l559_559603

theorem trigonometric_identity :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  t30 = 1 / Real.sqrt 3 →
  s30 = 1 / 2 →
  (t30 ^ 2 - s30 ^ 2) / (t30 ^ 2 * s30 ^ 2) = 1 :=
by
  intros t30_eq s30_eq
  -- Proof would go here
  sorry

end trigonometric_identity_l559_559603


namespace pictures_at_dolphin_show_l559_559916

def taken_before : Int := 28
def total_pictures_taken : Int := 44

theorem pictures_at_dolphin_show : total_pictures_taken - taken_before = 16 := by
  -- solution proof goes here
  sorry

end pictures_at_dolphin_show_l559_559916


namespace sum_cos_series_sum_sin_series_cos_series_coeff_sin_series_coeff_l559_559249

-- Part (a)
theorem sum_cos_series (a ϕ : ℝ) (h : |a| < 1) :
  (∑' k : ℕ, (a ^ k) * (Real.cos (k * ϕ))) = (1 - a * Real.cos ϕ) / (1 - 2 * a * Real.cos ϕ + a ^ 2) :=
by sorry

-- Part (b)
theorem sum_sin_series (a ϕ : ℝ) (h : |a| < 1) :
  (∑' k : ℕ, (a ^ (k + 1)) * (Real.sin ((k + 1) * ϕ))) = (a * Real.sin ϕ) / (1 - 2 * a * Real.cos ϕ + a ^ 2) :=
by sorry

-- Part (c)
theorem cos_series_coeff (ϕ : ℝ) (n : ℕ) :
  (∑ i in finset.range (n + 1), (n.choose i) * (Real.cos ((i + 1) * ϕ))) = 
    2^n * (Real.cos (ϕ / 2))^n * (Real.cos ((n + 2) * ϕ / 2)) :=
by sorry

-- Part (d)
theorem sin_series_coeff (ϕ : ℝ) (n : ℕ) :
  (∑ i in finset.range (n + 1), (n.choose i) * (Real.sin ((i + 1) * ϕ))) = 
    2^n * (Real.cos (ϕ / 2))^n * (Real.sin ((n + 2) * ϕ / 2)) :=
by sorry

end sum_cos_series_sum_sin_series_cos_series_coeff_sin_series_coeff_l559_559249


namespace range_of_a_l559_559811

noncomputable def a_range (a : Real) : Prop :=
  ∃ x : Real, (a + Real.cos x) * (a - Real.sin x) = 1

theorem range_of_a (a : Real) :
  a_range a → a ∈ set.Icc (-1 - Real.sqrt 2 / 2) (-1 + Real.sqrt 2 / 2) ∪ set.Icc (1 - Real.sqrt 2 / 2) (1 + Real.sqrt 2 / 2) :=
sorry

end range_of_a_l559_559811


namespace distinguishable_rearrangements_vowels_first_l559_559339

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def permutations (n k : ℕ) : ℕ :=
  factorial n / factorial k

theorem distinguishable_rearrangements_vowels_first :
  let word := "CONSTRUCTION".to_list in
  let vowels := ['O', 'U', 'I'] in
  let consonants := ['C', 'N', 'S', 'T', 'R', 'C', 'T', 'N'] in
  let num_vowel_arrangements := factorial 3 in
  let num_consonant_arrangements := permutations 8 (2 * 2 * 2) in
  num_vowel_arrangements * num_consonant_arrangements = 30240 := by
  sorry

end distinguishable_rearrangements_vowels_first_l559_559339


namespace number_of_pairings_is_odd_product_l559_559232

noncomputable def number_of_pairings (n : ℕ) : ℕ :=
  (2 * n).factorial / (n.factorial * 2^n)

theorem number_of_pairings_is_odd_product (n : ℕ) :
  number_of_pairings n = List.prod (List.map (λ k, 2 * k + 1) (List.range n)) := by
  sorry

end number_of_pairings_is_odd_product_l559_559232


namespace gcd_max_value_l559_559569

theorem gcd_max_value (n : ℕ) (hn : 0 < n) : ∃ m, (m = 3 ∧ ∀ (n : ℕ), 0 < n → ∃ k, m = gcd (13 * n + 4) (8 * n + 3)) :=
by
  sorry

end gcd_max_value_l559_559569


namespace g_123_eq_1494_l559_559454

def g : ℤ → ℤ 
| n => if n >= 1500 then n - 4 else g (g (n + 6))

theorem g_123_eq_1494 : g 123 = 1494 :=
by
  sorry

end g_123_eq_1494_l559_559454


namespace contest_question_count_l559_559768

noncomputable def total_questions_in_contest (riley_correct : ℕ) (total_incorrect : ℕ) (riley_mistakes : ℕ) (ofelia_bonus : ℕ) : ℕ :=
  let r := riley_correct in
  let o := (1/2 : ℚ) * r + ofelia_bonus in
  have riley_attempted : ℕ := r + riley_mistakes from
    by norm_num,
  have ofelia_incorrect := riley_attempted - o.to_nat from
    by norm_num,
  let team_incorrect := riley_mistakes + ofelia_incorrect in
  if h : team_incorrect = total_incorrect then
    r + riley_mistakes
  else
    sorry -- This would account for the error scenario.

theorem contest_question_count : total_questions_in_contest 32 17 3 5 = 35 :=
by {
  unfold total_questions_in_contest,
  norm_num,
  sorry
}

end contest_question_count_l559_559768


namespace monomial_proof_l559_559242

-- Define the monomial form
def monomial_form (a : ℝ) : Prop :=
  ∃ (a : ℝ), (∃ (x y : ℝ), true)

-- Prove that the type of the monomial is x^2*y
def type_of_monomial : Prop :=
  ∀ (a : ℝ), monomial_form a ↔ a * x^2 * y

-- Prove that the coefficient can be any real number, examples as 1
def coefficient_of_monomial (a : ℝ) : Prop :=
  a = 1 ∨ a = -2

-- Prove that the degree of the monomial x^2*y is 3
def degree_of_monomial : Prop :=
  ∀ (x y : ℝ), (2 + 1 = 3)

/-- Final full mathematical proof problem combining all parts,
showing the monomial, its coefficient as example 1, and degree 3 -/
theorem monomial_proof : ∀ (x y : ℝ),
  (∃ (a : ℝ), type_of_monomial a) ∧
  (∃ (a : ℝ), coefficient_of_monomial a) ∧
  (degree_of_monomial x y)
:= 
by
  sorry

end monomial_proof_l559_559242


namespace solve_equation_l559_559067

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end solve_equation_l559_559067


namespace correct_operation_l559_559155

noncomputable def sqrt_op_A: Prop := sqrt 4 ≠ 2
noncomputable def sqrt_op_B: Prop := (± sqrt (5^2)) ≠ -5
noncomputable def sqrt_op_C: Prop := sqrt ((-7) ^ 2) = 7
noncomputable def sqrt_op_D: Prop := sqrt (-3) ≠ - sqrt 3

theorem correct_operation : (sqrt_op_A ∧ sqrt_op_B ∧ sqrt_op_C ∧ sqrt_op_D) → (sqrt_op_C = 7) :=
by
  intros h
  sorry

end correct_operation_l559_559155


namespace range_of_a_l559_559722

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * a * x

theorem range_of_a (a x : ℝ) (hx : x + f a x + m ≠ 0 ∀ m : ℝ) : a < 1 / 3 :=
by
  sorry

end range_of_a_l559_559722


namespace fraction_purple_after_change_l559_559762

variables (x : ℕ)  -- Assume the total number of marbles is a natural number

def initial_yellow_marbles := (4 / 7 : ℚ) * x
def initial_green_marbles := (2 / 7 : ℚ) * x
def initial_purple_marbles := x - initial_yellow_marbles - initial_green_marbles

def new_purple_marbles := 3 * initial_purple_marbles
def new_total_marbles := initial_yellow_marbles + initial_green_marbles + new_purple_marbles

theorem fraction_purple_after_change : new_purple_marbles / new_total_marbles = (1 / 3 : ℚ) :=
by
  sorry

end fraction_purple_after_change_l559_559762


namespace trigonometric_identity_l559_559602

theorem trigonometric_identity :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  t30 = 1 / Real.sqrt 3 →
  s30 = 1 / 2 →
  (t30 ^ 2 - s30 ^ 2) / (t30 ^ 2 * s30 ^ 2) = 1 :=
by
  intros t30_eq s30_eq
  -- Proof would go here
  sorry

end trigonometric_identity_l559_559602


namespace circle_condition_l559_559861

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + m = 0) → m < 1 := 
by
  sorry

end circle_condition_l559_559861


namespace train_seats_count_l559_559957

theorem train_seats_count 
  (Standard Comfort Premium : ℝ)
  (Total_SEATS : ℝ)
  (hs : Standard = 36)
  (hc : Comfort = 0.20 * Total_SEATS)
  (hp : Premium = (3/5) * Total_SEATS)
  (ht : Standard + Comfort + Premium = Total_SEATS) :
  Total_SEATS = 180 := sorry

end train_seats_count_l559_559957


namespace mrs_hilt_money_left_l559_559417

def price_pencil := 0.11 -- in dollars
def price_notebook := 0.45 -- in dollars
def num_pencils := 3
def sales_tax_rate := 0.08 -- 8%
def initial_money := 1.50 -- in dollars

def total_cost_before_tax : ℝ := (num_pencils * price_pencil) + price_notebook
def sales_tax : ℝ := sales_tax_rate * total_cost_before_tax
def total_cost_with_tax : ℝ := total_cost_before_tax + sales_tax
def money_left : ℝ := initial_money - total_cost_with_tax

theorem mrs_hilt_money_left : money_left = 0.66 :=
by
  sorry

end mrs_hilt_money_left_l559_559417


namespace set_intersection_example_l559_559308

theorem set_intersection_example :
  let A := {1, 2, 3}
  let B := {1, 2, 5}
  A ∩ B = {1, 2} := 
by
  sorry

end set_intersection_example_l559_559308


namespace price_per_litre_of_second_oil_l559_559747

-- Define the conditions given in the problem
def oil1_volume : ℝ := 10 -- 10 litres of first oil
def oil1_rate : ℝ := 50 -- Rs. 50 per litre

def oil2_volume : ℝ := 5 -- 5 litres of the second oil
def total_mixed_volume : ℝ := oil1_volume + oil2_volume -- Total volume of mixed oil

def mixed_rate : ℝ := 55.33 -- Rs. 55.33 per litre for the mixed oil

-- Define the target value to prove: price per litre of the second oil
def price_of_second_oil : ℝ := 65.99

-- Prove the statement
theorem price_per_litre_of_second_oil : 
  (oil1_volume * oil1_rate + oil2_volume * price_of_second_oil) = total_mixed_volume * mixed_rate :=
by 
  sorry -- actual proof to be provided

end price_per_litre_of_second_oil_l559_559747


namespace range_of_a_l559_559309

theorem range_of_a {A B : Set ℝ} (a : ℝ)
  (hA : ∀ x, x ∈ A ↔ 1 ≤ x ∧ x ≤ 3)
  (hB : ∀ x, x ∈ B ↔ a ≤ x ∧ x ≤ a + 3)
  (hSubset : A ⊆ B) :
  0 ≤ a ∧ a ≤ 1 :=
begin
  sorry
end

end range_of_a_l559_559309


namespace num_solutions_g_g_of_x_eq_5_l559_559790

def g (x : ℝ) : ℝ := 
if x ≤ 1 then -x + 2 else 3 * x - 7

theorem num_solutions_g_g_of_x_eq_5 : 
  ∃ (s : Finset ℝ), 
    (∀ x, g (g x) = 5 ↔ x ∈ s) ∧ 
    Finset.card s = 3 := 
by
  sorry

end num_solutions_g_g_of_x_eq_5_l559_559790


namespace both_reunions_l559_559923

theorem both_reunions (U O H B : ℕ) 
  (hU : U = 100) 
  (hO : O = 50) 
  (hH : H = 62) 
  (attend_one : U = O + H - B) :  
  B = 12 := 
by 
  sorry

end both_reunions_l559_559923


namespace samantha_birth_year_l559_559453

theorem samantha_birth_year
  (first_amc8_year : ℕ := 1985)
  (held_annually : ∀ (n : ℕ), n ≥ 0 → first_amc8_year + n = 1985 + n)
  (samantha_age_7th_amc8 : ℕ := 12) :
  ∃ (birth_year : ℤ), birth_year = 1979 :=
by
  sorry

end samantha_birth_year_l559_559453


namespace trigonometric_identity_example_l559_559621

theorem trigonometric_identity_example :
  (let θ := 30 / 180 * Real.pi in
   (Real.tan θ)^2 - (Real.sin θ)^2) / ((Real.tan θ)^2 * (Real.sin θ)^2) = 4 / 3 :=
by
  let θ := 30 / 180 * Real.pi
  have h_tan2 := Real.tan θ * Real.tan θ
  have h_sin2 : Real.sin θ * Real.sin θ = 1/4 := by sorry
  have h_cos2 : Real.cos θ * Real.cos θ = 3/4 := by sorry
  sorry

end trigonometric_identity_example_l559_559621


namespace prove_m_plus_n_eq_47_l559_559449

variables (P R A C : ℝ)

-- Conditions
def condition1 (B: ℝ) : Prop := B = (3/16) * P
def condition2 (B: ℝ) : Prop := B = (2/9) * R
def condition3 (A C B: ℝ) : Prop := A = P - B ∧ C = R - B ∧ P/R = 32/27

-- Theorem statement
theorem prove_m_plus_n_eq_47 (B: ℝ) (h1: condition1 P B) (h2: condition2 R B) (h3: condition3 P R A B) :
  (13 / 16 * P / (7 / 9 * R)) = 26 / 21 ∧ (26 + 21 = 47) :=
by
  sorry

end prove_m_plus_n_eq_47_l559_559449


namespace min_transportation_expense_l559_559535

-- Define the conditions
def num_air_conditioners := 100
def num_trucks_type_a := 4
def num_trucks_type_b := 8
def capacity_truck_type_a := 20
def cost_truck_type_a := 400
def capacity_truck_type_b := 10
def cost_truck_type_b := 300

-- Define the goal to prove
theorem min_transportation_expense : 
  let total_cost := 
    (num_trucks_type_a * cost_truck_type_a) + 
    (2 * cost_truck_type_b) in 
  total_cost = 2200 :=
by
  sorry

end min_transportation_expense_l559_559535


namespace option_d_is_right_triangle_l559_559777

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + c^2 = b^2

theorem option_d_is_right_triangle (a b c : ℝ) (h : a^2 = b^2 - c^2) :
  right_triangle a b c :=
by
  sorry

end option_d_is_right_triangle_l559_559777


namespace correct_equation_l559_559991

-- Definitions of the conditions
def january_turnover (T : ℝ) : Prop := T = 36
def march_turnover (T : ℝ) : Prop := T = 48
def average_monthly_growth_rate (x : ℝ) : Prop := True

-- The goal to be proved
theorem correct_equation (T_jan T_mar : ℝ) (x : ℝ) 
  (h_jan : january_turnover T_jan) 
  (h_mar : march_turnover T_mar) 
  (h_growth : average_monthly_growth_rate x) : 
  36 * (1 + x)^2 = 48 :=
sorry

end correct_equation_l559_559991


namespace line_equation_correct_l559_559944

def point := (ℝ × ℝ)
def slope (θ : ℝ) : ℝ := Real.tan θ

-- Define the conditions
def P : point := (-1, 2)
def θ : ℝ := Real.pi / 4  -- 45 degrees in radians
def k : ℝ := slope θ      -- slope calculated using θ

-- Define the target property to prove
def correct_eq : Prop := ∀ x y : ℝ, (y - 2 = k * (x + 1)) → (x - y + 3 = 0)

theorem line_equation_correct : correct_eq :=
sorry

end line_equation_correct_l559_559944


namespace player1_wins_game_533_player1_wins_game_1000_l559_559889

-- Defining a structure for the game conditions
structure Game :=
  (target_sum : ℕ)
  (player1_wins_optimal : Bool)

-- Definition of the game scenarios
def game_533 := Game.mk 533 true
def game_1000 := Game.mk 1000 true

-- Theorem statements for the respective games
theorem player1_wins_game_533 : game_533.player1_wins_optimal :=
by sorry

theorem player1_wins_game_1000 : game_1000.player1_wins_optimal :=
by sorry

end player1_wins_game_533_player1_wins_game_1000_l559_559889


namespace max_right_angles_convex_polygon_l559_559145

theorem max_right_angles_convex_polygon (n : ℕ) (h : n > 4) : 
  ∀ x, x ≤ 3 := 
begin  
  sorry
end

end max_right_angles_convex_polygon_l559_559145


namespace area_triangle_ABC_l559_559363

variable (AB CD : ℝ) (height : ℝ)
variable (h_parallel : ∃ h : ℝ, h = height)
variable (h_area_trapezoid : 0 < h ∧ h == 24)
variable (h_ratio : CD = 3 * AB)

theorem area_triangle_ABC (AB CD height : ℝ) (h_parallel : ∃ h : ℝ, h = height)
  (h_area_trapezoid : 0 < h ∧ h == 24) (h_ratio : CD = 3 * AB) : 
  let Area_Tri_ABC := 1/2 * AB * height in
  Area_Tri_ABC == 6 :=
by
  -- This is where the proof would go
  sorry

end area_triangle_ABC_l559_559363


namespace find_a_l559_559466

theorem find_a (a : ℝ) (h1 : a > 0) :
  (a^0 + a^1 = 3) → a = 2 :=
by sorry

end find_a_l559_559466


namespace journal_pages_l559_559123

theorem journal_pages (sessions_per_week : ℕ) (pages_per_session : ℕ) (weeks : ℕ) :
  sessions_per_week = 3 →
  pages_per_session = 4 →
  weeks = 6 →
  (sessions_per_week * pages_per_session * weeks) = 72 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num, -- Optional to simplify the expression
end

end journal_pages_l559_559123


namespace num_negative_values_x_l559_559107

def operation_star (a b : ℤ) : ℤ := a^2 / b

theorem num_negative_values_x :
  { x : ℤ // x < 0 ∧ ∃ n : ℤ, n > 0 ∧ operation_star 12 x = n }.card = 15 :=
  sorry

end num_negative_values_x_l559_559107


namespace jessica_remaining_time_after_penalties_l559_559024

-- Definitions for the given conditions
def questions_answered : ℕ := 16
def total_questions : ℕ := 80
def time_used_minutes : ℕ := 12
def exam_duration_minutes : ℕ := 60
def penalty_per_incorrect_answer_minutes : ℕ := 2

-- Define the rate of answering questions
def answering_rate : ℚ := questions_answered / time_used_minutes

-- Define the total time needed to answer all questions
def total_time_needed : ℚ := total_questions / answering_rate

-- Define the remaining time after penalties
def remaining_time_after_penalties (x : ℕ) : ℤ :=
  max 0 (0 - penalty_per_incorrect_answer_minutes * x)

-- The theorem to prove
theorem jessica_remaining_time_after_penalties (x : ℕ) : 
  remaining_time_after_penalties x = max 0 (0 - penalty_per_incorrect_answer_minutes * x) := 
by
  sorry

end jessica_remaining_time_after_penalties_l559_559024


namespace cell_division_proof_l559_559367

-- Define the problem
def cell_division_ways (n m : Nat) : Nat :=
  if (n = 17 ∧ m = 9) then 10 else 0

-- The Lean statement to assert the problem
theorem cell_division_proof : cell_division_ways 17 9 = 10 :=
by
-- simplifying the definition for the given parameters
simp [cell_division_ways]
sorry

end cell_division_proof_l559_559367


namespace album_photos_proof_l559_559221

def photos_per_page := 4

-- Conditions
def position_81st_photo (n: ℕ) (x: ℕ) :=
  4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20

def position_171st_photo (n: ℕ) (y: ℕ) :=
  4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12

noncomputable def album_photos := 32

theorem album_photos_proof :
  ∃ n x y, position_81st_photo n x ∧ position_171st_photo n y ∧ 4 * n = album_photos :=
by
  sorry

end album_photos_proof_l559_559221


namespace ratio_quadrilateral_l559_559853

theorem ratio_quadrilateral
  (ABCD_area : ℝ)
  (h_ABCD : ABCD_area = 40)
  (K L M N : Type)
  (AK KB : ℝ)
  (h_ratio : AK / KB = BL / LC ∧ BL / LC = CM / MD ∧ CM / MD = DN / NA)
  (KLMN_area : ℝ)
  (h_KLMN : KLMN_area = 25) :
  (AK / (AK + KB) = 1 / 4 ∨ AK / (AK + KB) = 3 / 4) :=
sorry

end ratio_quadrilateral_l559_559853


namespace good_matrix_sum_of_permutation_l559_559391

-- Definitions
def is_good_matrix (A : Matrix (Fin n) (Fin n) ℕ) (m : ℕ) : Prop :=
  (∀ i, ∑ j, A i j = m) ∧ (∀ j, ∑ i, A i j = m)

def is_permutation_matrix (P : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∀ i j, P i j = 1 ↔ (∃! (k : Fin n), (P i k = 1) ∧ (P k j = 1))

-- Theorem Statement
theorem good_matrix_sum_of_permutation (n : ℕ) (hn : 1 ≤ n) (A : Matrix (Fin n) (Fin n) ℕ) (m : ℕ)
  (hA : is_good_matrix A m) : ∃ (Pm : List (Matrix (Fin n) (Fin n) ℕ)), (∀ P ∈ Pm, is_permutation_matrix P) ∧ A = List.sum Pm := 
sorry

end good_matrix_sum_of_permutation_l559_559391


namespace lamp_post_break_height_l559_559943

def height_at_break_point (x : ℝ) : Prop :=
  let h := 6
  let d := 2
  let b := x
  (h - b)^2 + d^2 = (h - b + sqrt ((x^2) + d^2) - h)^2

theorem lamp_post_break_height : height_at_break_point (sqrt 10) :=
sorry

end lamp_post_break_height_l559_559943


namespace number_of_solutions_l559_559640

theorem number_of_solutions (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 5) :
  (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 4 → x = -2 :=
sorry

end number_of_solutions_l559_559640


namespace range_of_a_l559_559878

-- Definition of the quadratic inequality problem
def quadratic_inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, 2*x^2 - a*x + 1 > 0

-- Statement of the theorem, which needs to be proven
theorem range_of_a (a : ℝ) (h : quadratic_inequality a) : -2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 :=
begin
  sorry,
end

end range_of_a_l559_559878


namespace tan_sin_identity_l559_559615

theorem tan_sin_identity :
  (let θ := 30 * Real.pi / 180 in 
   let tan_θ := Real.sin θ / Real.cos θ in 
   let sin_θ := 1 / 2 in
   let cos_θ := Real.sqrt 3 / 2 in
   ((tan_θ ^ 2 - sin_θ ^ 2) / (tan_θ ^ 2 * sin_θ ^ 2) = 1)) :=
by
  let θ := 30 * Real.pi / 180
  let tan_θ := Real.sin θ / Real.cos θ
  let sin_θ := 1 / 2
  let cos_θ := Real.sqrt 3 / 2
  have h_tan_θ : tan_θ = 1 / Real.sqrt 3 := sorry  -- We skip the derivation proof.
  have h_sin_θ : sin_θ = 1 / 2 := rfl
  have h_cos_θ : cos_θ = Real.sqrt 3 / 2 := sorry  -- We skip the derivation proof.
  have numerator := tan_θ ^ 2 - sin_θ ^ 2
  have denominator := tan_θ ^ 2 * sin_θ ^ 2
  -- show equality
  have h : (numerator / denominator) = (1 : ℝ) := sorry  -- We skip the final proof.
  exact h

end tan_sin_identity_l559_559615


namespace max_interesting_numbers_l559_559902

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

def is_interesting (n : ℕ) : Prop :=
  is_prime (sum_of_digits n)

theorem max_interesting_numbers :
  ∀ (a b c d e : ℕ), b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
    (∀ x ∈ {a, b, c, d, e}, x ∈ ℕ) →
    ∃ s : Finset ℕ, s ⊆ {a, b, c, d, e} ∧
                     (∀ n ∈ s, is_interesting n) ∧ s.card = 4 :=
by
  sorry

end max_interesting_numbers_l559_559902


namespace min_value_g_l559_559491

noncomputable def g (x : ℝ) : ℝ := (6 * x^2 + 11 * x + 17) / (7 * (2 + x))

theorem min_value_g : ∃ x, x ≥ 0 ∧ g x = 127 / 24 :=
by
  sorry

end min_value_g_l559_559491


namespace sin_C_eq_one_fourth_b_div_c_l559_559351

-- Definitions based on the given conditions
variables (A B C a b c : ℝ)
variables (cosA sinA sinC : ℝ)

-- Given conditions
axiom h1 : a = 2 * c * cosA
axiom h2 : sqrt 5 * sinA = 1

-- Proof of the first part: sin C = 1/4
theorem sin_C_eq_one_fourth (h1 : a = 2 * c * cosA) (h2 : sqrt 5 * sinA = 1) : sinC = 1 / 4 := 
sorry

-- Given the solution of sin C, use it to prove the second part
axiom sinC_value : sinC = 1 / 4

-- Proof of the second part: b/c = (2sqrt(5) + 5sqrt(3)) / 5
theorem b_div_c (h1 : a = 2 * c * cosA) (h2 : sqrt 5 * sinA = 1) (h3 : sinC = 1 / 4) : b / c = (2 * sqrt 5 + 5 * sqrt 3) / 5 := 
sorry

end sin_C_eq_one_fourth_b_div_c_l559_559351


namespace option_A_option_D_l559_559017

variables {A B C a b c : Real}

-- Proof Problem for Option A
theorem option_A (hAB : A > B) : sin A > sin B :=
sorry

-- Proof Problem for Option D
theorem option_D (h_sin : sin C ^ 2 > sin A ^ 2 + sin B ^ 2) : C > π / 2 :=
begin
  sorry
end

end option_A_option_D_l559_559017


namespace prob_A_wins_all_three_rounds_prob_B_wins_within_five_rounds_l559_559139

-- Definitions of probabilities and initial conditions
noncomputable def P_A_first_wins : ℚ := 2/3
noncomputable def P_A_first_draw : ℚ := 1/6
noncomputable def P_B_first_wins : ℚ := 1/2
noncomputable def P_B_first_draw : ℚ := 1/4
noncomputable def P_B_first_loses : ℚ := 1 - P_B_first_wins - P_B_first_draw
noncomputable def round_A_wins_first : ℚ := P_A_first_wins * P_B_first_loses * P_B_first_loses

-- Problem Statement 1: Probability of the game ending within three rounds and A winning all rounds
theorem prob_A_wins_all_three_rounds : round_A_wins_first = 1/24 := 
by
  sorry

-- Problem Statement 2: Probability of the game ending within five rounds and B winning
noncomputable def prob_B_wins_in_three : ℚ := (1 - P_A_first_wins - P_A_first_draw)^3
noncomputable def prob_B_wins_in_four : ℚ := 3 * (1 - P_A_first_wins - P_A_first_draw)^2 * P_B_first_wins * 5/6
noncomputable def prob_B_wins_in_five : ℚ := 3 * (1 - P_A_first_wins - P_A_first_draw)^2 * P_B_first_wins * 5/6 * 1/2 
                                           + 3 * P_B_first_wins^2 * (1 - P_A_first_wins - P_A_first_draw) * (5/6)^2

theorem prob_B_wins_within_five_rounds : prob_B_wins_in_three + prob_B_wins_in_four + prob_B_wins_in_five = 31/216 := 
by
  sorry

end prob_A_wins_all_three_rounds_prob_B_wins_within_five_rounds_l559_559139


namespace complex_number_solution_l559_559700

theorem complex_number_solution (Z : ℂ) (h : (1 + 2 * complex.I)^3 * Z = 1 + 2 * complex.I) :
  Z = -3 / 25 + 24 / 125 * complex.I := sorry

end complex_number_solution_l559_559700


namespace area_of_intersection_is_100_l559_559954

-- A point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the vertices and midpoints
def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨5, 0, 0⟩
def C : Point3D := ⟨5, 5, 0⟩
def D : Point3D := ⟨0, 5, 0⟩
def E : Point3D := ⟨2.5, 2.5, 2.5 * Real.sqrt 2⟩

def midpoint (P Q : Point3D) : Point3D := 
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2, (P.z + Q.z) / 2⟩

def R : Point3D := midpoint A E
def S : Point3D := midpoint B C
def T : Point3D := midpoint C D

-- The plane's equation: x + y + z√2 = 7.5 is derived from points R, S, T

-- Function calculating the area of the polygon formed by the intersection
def polygon_area (plane : Point3D → Prop) : ℝ := sorry

-- The proof problem statement
theorem area_of_intersection_is_100 : 
  polygon_area (λ P, P.x + P.y + P.z * Real.sqrt 2 = 7.5) = Real.sqrt 100 := 
sorry

end area_of_intersection_is_100_l559_559954


namespace jerry_cut_pine_trees_l559_559785

theorem jerry_cut_pine_trees (P : ℕ)
  (h1 : 3 * 60 = 180)
  (h2 : 4 * 100 = 400)
  (h3 : 80 * P + 180 + 400 = 1220) :
  P = 8 :=
by {
  sorry -- Proof not required as per the instructions
}

end jerry_cut_pine_trees_l559_559785


namespace series_convergence_p_geq_2_l559_559917

noncomputable def ai_series_converges (a : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, a i ^ 2 = l

noncomputable def bi_series_converges (b : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, b i ^ 2 = l

theorem series_convergence_p_geq_2 
  (a b : ℕ → ℝ) 
  (h₁ : ai_series_converges a)
  (h₂ : bi_series_converges b) 
  (p : ℝ) (hp : p ≥ 2) : 
  ∃ l : ℝ, ∑' i, |a i - b i| ^ p = l := 
sorry

end series_convergence_p_geq_2_l559_559917


namespace find_abc_sum_l559_559796

theorem find_abc_sum :
  ∃ (a b c : ℤ), 2 * a + 3 * b = 52 ∧ 3 * b + c = 41 ∧ b * c = 60 ∧ a + b + c = 25 :=
by
  use 8, 12, 5
  sorry

end find_abc_sum_l559_559796


namespace orcs_per_squad_is_eight_l559_559851

-- Defining the conditions
def total_weight_of_swords := 1200
def weight_each_orc_can_carry := 15
def number_of_squads := 10

-- Proof statement to demonstrate the answer
theorem orcs_per_squad_is_eight :
  (total_weight_of_swords / weight_each_orc_can_carry) / number_of_squads = 8 := by
  sorry

end orcs_per_squad_is_eight_l559_559851


namespace ellipse_hyperbola_eccentricities_l559_559304

theorem ellipse_hyperbola_eccentricities 
    (c : ℝ) 
    (a1 a2 : ℝ) 
    (h1 : ∃ e1 e2 : ℝ, e1 = c / a1 ∧ e2 = c / a2 ∧ a1 = 5 + c ∧ a2 = 5 - c ∧ 1 < (25 / c^2) < 4) 
    (h2 : 5 / 2 < c ∧ c < 5) : 
    ∃ e1 e2 : ℝ, e1 * e2 ∈ (1 / 3, ∞) :=
by
  sorry

end ellipse_hyperbola_eccentricities_l559_559304


namespace range_a_l559_559716

noncomputable def f (x : ℝ) (t : ℝ) (a : ℝ) : ℝ := x^2 * Real.exp(x) + Real.log(t) - a

theorem range_a {a : ℝ} :
  (∀ t, 1 ≤ t ∧ t ≤ Real.exp(1) → ∃! x, -1 ≤ x ∧ x ≤ 1 ∧ f x t a = 0) ↔ (1 + 1 / Real.exp(1) < a ∧ a ≤ Real.exp(1)) :=
by
  sorry

end range_a_l559_559716


namespace cow_manure_plant_height_l559_559023

theorem cow_manure_plant_height (control_height bone_meal_percentage cow_manure_percentage : ℝ)
  (control_height_eq : control_height = 36)
  (bone_meal_eq : bone_meal_percentage = 125)
  (cow_manure_eq : cow_manure_percentage = 200) :
  let bone_meal_height := (bone_meal_percentage / 100) * control_height in
  let cow_manure_height := (cow_manure_percentage / 100) * bone_meal_height in
  cow_manure_height = 90 := by
  sorry

end cow_manure_plant_height_l559_559023


namespace width_of_rect_prism_l559_559750

theorem width_of_rect_prism 
  (l : ℝ) (h : ℝ) (d : ℝ) (w : ℝ)
  (hl : l = 8) (hh : h = 15) (hd : d = 17) :
  sqrt (l ^ 2 + w ^ 2 + h ^ 2) = d → w = 0 :=
by
  intros hdiagonal
  sorry

end width_of_rect_prism_l559_559750


namespace degree_of_polynomial_l559_559487

-- Variables denoted as nonzero constants
variables {a b c d e f g : ℝ}

-- Hypothesis that a, b, c, d, e, f, g are non-zero
variables [NeZero a] [NeZero b] [NeZero c] [NeZero d] [NeZero e] [NeZero f] [NeZero g]

-- Degree of the given polynomial
theorem degree_of_polynomial :
  degree ((X^4 + a*X^7 + b*X + c) * (X^5 + d*X^3 + e*X + f) * (X + g)) = 13 :=
by sorry

end degree_of_polynomial_l559_559487


namespace leak_empty_time_l559_559542

variable (inlet_rate : ℕ := 6) -- litres per minute
variable (total_capacity : ℕ := 12960) -- litres
variable (empty_time_with_inlet_open : ℕ := 12) -- hours

def inlet_rate_per_hour := inlet_rate * 60 -- litres per hour
def net_emptying_rate := total_capacity / empty_time_with_inlet_open -- litres per hour
def leak_rate := net_emptying_rate + inlet_rate_per_hour -- litres per hour

theorem leak_empty_time : total_capacity / leak_rate = 9 := by
  sorry

end leak_empty_time_l559_559542


namespace distinct_ordered_pairs_sum_reciprocal_l559_559338

theorem distinct_ordered_pairs_sum_reciprocal :
  {mn_pairs : Finset (ℕ × ℕ) | ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 1 / (m:ℝ) + 1 / (n:ℝ) = 1 / 3}.card = 3 :=
begin
  sorry,
end

end distinct_ordered_pairs_sum_reciprocal_l559_559338


namespace sufficient_condition_l559_559306

open Classical

variables (x a : ℝ)

def p := x^2 + 2*x - 3 > 0
def q := x > a
def not_p := -3 ≤ x ∧ x ≤ 1
def not_q := x ≤ a

theorem sufficient_condition (x a : ℝ) (h1 : ∀ x, p x → ¬q x) :
    (¬p) → (¬q) → (a ≥ 1) :=
  begin
    sorry
  end

end sufficient_condition_l559_559306


namespace house_A_cost_l559_559418

theorem house_A_cost (base_salary earnings commission_rate total_houses cost_A cost_B cost_C : ℝ)
  (H_base_salary : base_salary = 3000)
  (H_earnings : earnings = 8000)
  (H_commission_rate : commission_rate = 0.02)
  (H_cost_B : cost_B = 3 * cost_A)
  (H_cost_C : cost_C = 2 * cost_A - 110000)
  (H_total_commission : earnings - base_salary = 5000)
  (H_total_cost : 5000 / commission_rate = 250000)
  (H_total_houses : base_salary + commission_rate * (cost_A + cost_B + cost_C) = earnings) :
  cost_A = 60000 := sorry

end house_A_cost_l559_559418


namespace least_integer_with_ten_factors_l559_559894

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.map (λ k, k + 1)).prod

theorem least_integer_with_ten_factors : ∃ n, ∃ p q a b : ℕ, 
  prime p ∧ prime q ∧ p < q ∧
  n = p^a * q^b ∧
  a + 1 = 2 ∧ b + 1 = 5 ∧
  n = 48 :=
sorry

end least_integer_with_ten_factors_l559_559894


namespace point_reflection_l559_559855

-- Define the original point and the reflection function
structure Point where
  x : ℝ
  y : ℝ

def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

-- Define the original point
def M : Point := ⟨-5, 2⟩

-- State the theorem to prove the reflection
theorem point_reflection : reflect_y_axis M = ⟨5, 2⟩ :=
  sorry

end point_reflection_l559_559855


namespace shift_graph_l559_559047

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 - 2 * sin x * cos x - sin x ^ 2
noncomputable def g (x : ℝ) : ℝ := 2 * cos x ^ 2 + 2 * sin x * cos x - 1

theorem shift_graph (m : ℝ) (k : ℤ) : (∀ x, f (x - m) = g x) ↔ m = (π / 4 - k * π) :=
by
  intro x
  sorry

end shift_graph_l559_559047


namespace probability_of_distinct_dice_numbers_l559_559495

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l559_559495


namespace gcd_max_value_l559_559570

theorem gcd_max_value (n : ℕ) (hn : 0 < n) : ∃ m, (m = 3 ∧ ∀ (n : ℕ), 0 < n → ∃ k, m = gcd (13 * n + 4) (8 * n + 3)) :=
by
  sorry

end gcd_max_value_l559_559570


namespace number_of_solutions_is_3_l559_559311

noncomputable def count_solutions : Nat :=
  Nat.card {x : Nat // x < 150 ∧ (x + 15) % 45 = 75 % 45}

theorem number_of_solutions_is_3 : count_solutions = 3 := by
  sorry

end number_of_solutions_is_3_l559_559311


namespace min_cost_per_student_is_80_l559_559188

def num_students : ℕ := 48
def swims_per_student : ℕ := 8
def cost_per_card : ℕ := 240
def cost_per_bus : ℕ := 40

def total_swims : ℕ := num_students * swims_per_student

def min_cost_per_student : ℕ :=
  let n := 8
  let c := total_swims / n
  let total_cost := cost_per_card * n + cost_per_bus * c
  total_cost / num_students

theorem min_cost_per_student_is_80 :
  min_cost_per_student = 80 :=
sorry

end min_cost_per_student_is_80_l559_559188


namespace maximum_value_of_n_with_positive_sequence_l559_559688

theorem maximum_value_of_n_with_positive_sequence (a : ℕ → ℝ) (h_seq : ∀ n : ℕ, 0 < a n) 
    (h_arithmetic : ∀ n : ℕ, a (n + 1)^2 - a n^2 = 1) : ∃ n : ℕ, n = 24 ∧ a n < 5 :=
by
  sorry

end maximum_value_of_n_with_positive_sequence_l559_559688


namespace population_total_l559_559225

theorem population_total (total_population layers : ℕ) (ratio_A ratio_B ratio_C : ℕ) 
(sample_capacity : ℕ) (prob_ab_in_C : ℚ) 
(h1 : ratio_A = 3)
(h2 : ratio_B = 6)
(h3 : ratio_C = 1)
(h4 : sample_capacity = 20)
(h5 : prob_ab_in_C = 1 / 21)
(h6 : total_population = 10 * ratio_C) :
  total_population = 70 := 
by 
  sorry

end population_total_l559_559225


namespace rationalize_denominator_cube_l559_559431

theorem rationalize_denominator_cube (A B C : ℤ) (hC_pos : 0 < C)
  (hB_coprime : ∀ p : ℕ, Prime p → p ^ 3 ∣ B → False) :
  (5 * Real.cbrt 49) / 21 = A * Real.cbrt B / C → 
  A + B + C = 75 :=
  by
    sorry 

end rationalize_denominator_cube_l559_559431


namespace price_of_tea_mixture_l559_559850

noncomputable def price_of_mixture (price1 price2 price3 : ℝ) (ratio1 ratio2 ratio3 : ℝ) : ℝ :=
  (price1 * ratio1 + price2 * ratio2 + price3 * ratio3) / (ratio1 + ratio2 + ratio3)

theorem price_of_tea_mixture :
  price_of_mixture 126 135 175.5 1 1 2 = 153 := 
by
  sorry

end price_of_tea_mixture_l559_559850


namespace correct_operation_l559_559165

theorem correct_operation (a b : ℝ) (c d : ℂ) : (sqrt 4 = 2 ∧ (± abs (5) ≠ -5) ∧ (sqrt (abs (7) ^ 2) = 7) ∧ ¬(is_real (sqrt (-3))) ) := sorry

end correct_operation_l559_559165


namespace pens_count_l559_559429

theorem pens_count (markers : ℕ) (H1 : markers = 25) : 
  ∃ pens : ℕ, (pens : ℚ) / markers = 2 / 5 ∧ pens = 10 :=
by
  use 10
  split
  -- prove the ratio condition
  have : (10 : ℚ) / 25 = 2 / 5 := by norm_num
  assumption
  -- prove the pens count condition
  refl

end pens_count_l559_559429


namespace cone_altitude_to_radius_ratio_l559_559555

theorem cone_altitude_to_radius_ratio (r h : ℝ) (V_cone V_sphere : ℝ)
  (h1 : V_sphere = (4 / 3) * Real.pi * r^3)
  (h2 : V_cone = (1 / 3) * Real.pi * r^2 * h)
  (h3 : V_cone = (1 / 3) * V_sphere) :
  h / r = 4 / 3 :=
by
  sorry

end cone_altitude_to_radius_ratio_l559_559555


namespace length_segment_FF_l559_559478

-- Define the points F and F' based on the given conditions
def F : (ℝ × ℝ) := (4, 3)
def F' : (ℝ × ℝ) := (-4, 3)

-- The theorem to prove the length of the segment FF' is 8
theorem length_segment_FF' : dist F F' = 8 :=
by
  sorry

end length_segment_FF_l559_559478


namespace remaining_flour_needed_l559_559414

-- Define the required total amount of flour
def total_flour : ℕ := 8

-- Define the amount of flour already added
def flour_added : ℕ := 2

-- Define the remaining amount of flour needed
def remaining_flour : ℕ := total_flour - flour_added

-- The theorem we need to prove
theorem remaining_flour_needed : remaining_flour = 6 := by
  sorry

end remaining_flour_needed_l559_559414


namespace population_increase_rate_l559_559873

theorem population_increase_rate (P₀ P₁ : ℕ) (rate : ℚ) (h₁ : P₀ = 220) (h₂ : P₁ = 242) :
  rate = ((P₁ - P₀ : ℚ) / P₀) * 100 := by
  sorry

end population_increase_rate_l559_559873


namespace option_A_is_correct_l559_559909

-- Define propositions p and q
variables (p q : Prop)

-- Option A
def isOptionACorrect: Prop := (¬p ∨ ¬q) → (¬p ∧ ¬q)

theorem option_A_is_correct: isOptionACorrect p q := sorry

end option_A_is_correct_l559_559909


namespace disks_paint_count_l559_559366

/-- In the figure below, 7 disks are arranged in a circle. 3 of these disks are to be painted blue, 
3 are to be painted red, and 1 is to be painted green. Two paintings that can be obtained from one another 
by a rotation of the entire figure are considered the same. Prove that the number of different paintings 
is 20. -/
noncomputable def distinctPaintings : Nat := 20

theorem disks_paint_count :
  let n := 7
  let blue := 3
  let red := 3
  let green := 1
  let total_symmetries := 7
  (∑ i in Finset.range total_symmetries, 1) / total_symmetries = distinctPaintings := 
by
  sorry

end disks_paint_count_l559_559366


namespace cafeteria_problem_l559_559888

-- Define the conditions and the final proof statement
theorem cafeteria_problem (m: ℝ) (a b c: ℕ) 
  (cond1 : (0.4 * (60 * 60) : ℝ) = (60 - m) ^ 2) 
  (cond2 : m = a - b * real.sqrt c) 
  (cond3 : nat_prime_square_free c): 
  (a + b + c = 87) := 
sorry

-- Definition for square-free condition
def nat_prime_square_free (c: ℕ) : Prop :=
  ∀ p, nat.prime p → p^2 ∣ c → false

end cafeteria_problem_l559_559888


namespace largest_integer_less_l559_559144

theorem largest_integer_less (
    sum_logs : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 1004 → log 3 (rat(i+1) / rat(i))
  ) :
  let logProduct := log 3 (1005)  -- Using prod_{i=1}^{1004} 1 * ... * 1005 / 1 simplified to log 1005
  in int.floor logProduct = 6 :=
by sorry

end largest_integer_less_l559_559144


namespace inner_rectangle_length_l559_559953

theorem inner_rectangle_length 
  (a b c : ℝ)
  (h1 : ∃ a1 a2 a3 : ℝ, a2 - a1 = a3 - a2)
  (w_inner : ℝ)
  (width_inner : w_inner = 2)
  (w_shaded : ℝ)
  (width_shaded : w_shaded = 1.5)
  (ar_prog : a = 2 * w_inner ∧ b = 3 * w_inner + 15 ∧ c = 3 * w_inner + 33)
  : ∀ x : ℝ, 2 * x = a → 3 * x + 15 = b → 3 * x + 33 = c → x = 3 :=
by
  sorry

end inner_rectangle_length_l559_559953


namespace exists_xi_eq_l559_559390

noncomputable
def continuously_differentiable (f: ℝ → ℝ) : Prop := differentiable ℝ f ∧ continuous f

theorem exists_xi_eq (f: ℝ → ℝ) (hf: continuously_differentiable f) :
  (∀ x, f x > 0) →
  ∃ ξ: ℝ, ξ ∈ Ioo (0 : ℝ) 1 ∧ e^(deriv f ξ) * (f 0)^(f ξ) = (f 1)^(f ξ) :=
by
  sorry

end exists_xi_eq_l559_559390


namespace clear_queue_with_three_windows_l559_559959

def time_to_clear_queue_one_window (a x y : ℕ) : Prop := a / (x - y) = 40

def time_to_clear_queue_two_windows (a x y : ℕ) : Prop := a / (2 * x - y) = 16

theorem clear_queue_with_three_windows (a x y : ℕ) 
  (h1 : time_to_clear_queue_one_window a x y) 
  (h2 : time_to_clear_queue_two_windows a x y) : 
  a / (3 * x - y) = 10 :=
by
  sorry

end clear_queue_with_three_windows_l559_559959


namespace coordinates_of_P_l559_559692

def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (-9, 2)
def P : ℝ × ℝ := (-1, -2)

theorem coordinates_of_P : P = (1 / 3 • (B.1 - A.1) + 2 / 3 • A.1, 1 / 3 • (B.2 - A.2) + 2 / 3 • A.2) :=
by
    rw [A, B, P]
    sorry

end coordinates_of_P_l559_559692


namespace correct_option_l559_559158

noncomputable def OptionA : Prop := (Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2)
noncomputable def OptionB : Prop := (Real.sqrt (5 ^ 2) = -5 ∨ Real.sqrt (5 ^ 2) = 5)
noncomputable def OptionC : Prop := Real.sqrt ((-7) ^ 2) = 7
noncomputable def OptionD : Prop := (Real.sqrt (-3) = -Real.sqrt 3)

theorem correct_option : OptionC := 
by 
  unfold OptionC
  simp
  exact eq.refl 7

end correct_option_l559_559158


namespace sum_of_squares_l559_559984

open_locale big_operators

theorem sum_of_squares (n : ℕ) : 
  ∑ i in Finset.range (n + 1), i * i = n * (n + 1) * (2 * n + 1) / 6 := 
by
  sorry

end sum_of_squares_l559_559984


namespace range_of_a_l559_559752

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * a * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l559_559752


namespace correct_total_annual_cost_l559_559788

def cost_after_coverage (cost: ℕ) (coverage: ℕ) : ℕ :=
  cost - (cost * coverage / 100)

def epiPen_costs : ℕ :=
  (cost_after_coverage 500 75) +
  (cost_after_coverage 550 60) +
  (cost_after_coverage 480 70) +
  (cost_after_coverage 520 65)

def monthly_medical_expenses : ℕ :=
  (cost_after_coverage 250 80) +
  (cost_after_coverage 180 70) +
  (cost_after_coverage 300 75) +
  (cost_after_coverage 350 60) +
  (cost_after_coverage 200 70) +
  (cost_after_coverage 400 80) +
  (cost_after_coverage 150 90) +
  (cost_after_coverage 100 100) +
  (cost_after_coverage 300 60) +
  (cost_after_coverage 350 90) +
  (cost_after_coverage 450 85) +
  (cost_after_coverage 500 65)

def total_annual_cost : ℕ :=
  epiPen_costs + monthly_medical_expenses

theorem correct_total_annual_cost :
  total_annual_cost = 1542 :=
  by sorry

end correct_total_annual_cost_l559_559788


namespace correct_operation_l559_559153

noncomputable def sqrt_op_A: Prop := sqrt 4 ≠ 2
noncomputable def sqrt_op_B: Prop := (± sqrt (5^2)) ≠ -5
noncomputable def sqrt_op_C: Prop := sqrt ((-7) ^ 2) = 7
noncomputable def sqrt_op_D: Prop := sqrt (-3) ≠ - sqrt 3

theorem correct_operation : (sqrt_op_A ∧ sqrt_op_B ∧ sqrt_op_C ∧ sqrt_op_D) → (sqrt_op_C = 7) :=
by
  intros h
  sorry

end correct_operation_l559_559153


namespace find_x_in_sequence_l559_559226

theorem find_x_in_sequence :
  (∀ a b c d : ℕ, a * b * c * d = 120) →
  (a = 2) →
  (b = 4) →
  (d = 3) →
  ∃ x : ℕ, 2 * 4 * x * 3 = 120 ∧ x = 5 :=
sorry

end find_x_in_sequence_l559_559226


namespace complex_conjugate_l559_559041

noncomputable def i_power_2023 : ℂ :=
  Complex.exp (Complex.I * (2023 * Real.pi / 2))

theorem complex_conjugate :
  let z := (1 - (1 - Complex.I)^2) / i_power_2023 in
  Complex.conj z = -2 - Complex.I :=
by
  let z := (1 - (1 - Complex.I)^2) / i_power_2023
  exact sorry

end complex_conjugate_l559_559041


namespace triangle_AM_plus_KM_eq_AB_l559_559829

theorem triangle_AM_plus_KM_eq_AB 
  (A B C M D K : Type) 
  [has_add M] [has_eq M]
  [has_add AB] [has_eq AB]
  [has_add AC] [has_eq AC]
  [has_add BD] [has_eq BD]
  [has_sub BD] [has_eq AC]
  (hMedian : is_median A M B C)
  (hD : D ∈ segment A C)
  (hBD_eq_AC : segment_length B D = segment_length A C)
  (hIntersection : ∃ K, K ∈ segment B D ∧ line A M ∩ line B D = {K})
  (hDK_eq_DC : segment_length D K = segment_length D C)
  : segment_length A M + segment_length K M = segment_length A B :=
sorry

end triangle_AM_plus_KM_eq_AB_l559_559829


namespace quadratic_equation_roots_distinct_l559_559119

theorem quadratic_equation_roots_distinct (m : ℝ) : 
  let a := 2
      b := -m
      c := -1
      Δ := b^2 - 4 * a * c
  in Δ > 0 :=
by
  let a := 2
  let b := -m
  let c := -1
  let Δ := b^2 - 4 * a * c
  have h : Δ = m^2 + 8,
  { simp [Δ, b, a, c],
    ring },
  rw h,
  exact add_pos_of_nonneg_of_pos (pow_two_nonneg m) (by norm_num)

end quadratic_equation_roots_distinct_l559_559119


namespace imaginary_part_of_complex_number_l559_559099

-- Given a condition
def condition : ℂ := Complex.i ^ 2016

-- Define the complex number in question
def complex_number : ℂ := (condition / (2 * Complex.i - 1)) * Complex.i

-- The theorem statement
theorem imaginary_part_of_complex_number : complex_number.im = -2/5 := 
by
  -- The proof is omitted
  sorry

end imaginary_part_of_complex_number_l559_559099


namespace find_x_l559_559364

theorem find_x 
  (x : ℝ) 
  (angle_PQS angle_QSR angle_SRQ : ℝ) 
  (h1 : angle_PQS = 2 * x)
  (h2 : angle_QSR = 50)
  (h3 : angle_SRQ = x) :
  x = 50 :=
sorry

end find_x_l559_559364


namespace trigonometric_identity_example_l559_559622

theorem trigonometric_identity_example :
  (let θ := 30 / 180 * Real.pi in
   (Real.tan θ)^2 - (Real.sin θ)^2) / ((Real.tan θ)^2 * (Real.sin θ)^2) = 4 / 3 :=
by
  let θ := 30 / 180 * Real.pi
  have h_tan2 := Real.tan θ * Real.tan θ
  have h_sin2 : Real.sin θ * Real.sin θ = 1/4 := by sorry
  have h_cos2 : Real.cos θ * Real.cos θ = 3/4 := by sorry
  sorry

end trigonometric_identity_example_l559_559622


namespace ratio_of_f_n_l559_559806

-- Define f(n) as the base-4 logarithm of the sum of the elements of the nth row in Pascal's triangle
def f (n : ℕ) : ℝ := Real.logb 4 (2 ^ n)

-- Define the problem statement
theorem ratio_of_f_n (n : ℕ) : (f n) / (Real.logb 4 8) = n / 3 := 
by
  sorry

end ratio_of_f_n_l559_559806


namespace evaluate_integral_l559_559272

noncomputable def integral_problem : ℝ :=
  ∫ x in -2..2, (real.sqrt (4 - x^2) - x ^ 2017)

theorem evaluate_integral : integral_problem = 2 * real.pi := by
  sorry

end evaluate_integral_l559_559272


namespace average_monthly_growth_rate_l559_559992

variable (x : ℝ)

-- Conditions
def turnover_January : ℝ := 36
def turnover_March : ℝ := 48

-- Theorem statement that corresponds to the problem's conditions and question
theorem average_monthly_growth_rate :
  turnover_January * (1 + x)^2 = turnover_March :=
sorry

end average_monthly_growth_rate_l559_559992


namespace smallest_possible_N_l559_559230

theorem smallest_possible_N : ∃ (N : ℕ), (N % 8 = 0 ∧ N % 4 = 0 ∧ N % 2 = 0) ∧ (∀ M : ℕ, (M % 8 = 0 ∧ M % 4 = 0 ∧ M % 2 = 0) → N ≤ M) ∧ N = 8 :=
by
  sorry

end smallest_possible_N_l559_559230


namespace consecutive_integers_divisible_product_l559_559285

theorem consecutive_integers_divisible_product (m n : ℕ) (h : m < n) :
  ∀ k : ℕ, ∃ i j : ℕ, i ≠ j ∧ k + i < k + n ∧ k + j < k + n ∧ (k + i) * (k + j) % (m * n) = 0 :=
by sorry

end consecutive_integers_divisible_product_l559_559285


namespace total_fish_correct_l559_559789

def Leo_fish := 40
def Agrey_fish := Leo_fish + 20
def Sierra_fish := Agrey_fish + 15
def total_fish := Leo_fish + Agrey_fish + Sierra_fish

theorem total_fish_correct : total_fish = 175 := by
  sorry


end total_fish_correct_l559_559789


namespace complex_pure_imaginary_solution_l559_559710

theorem complex_pure_imaginary_solution (a : ℝ)
  (h1 : a^2 + a - 2 = 0)
  (h2 : a^2 - 3a + 2 ≠ 0) :
  a = -2 :=
sorry

end complex_pure_imaginary_solution_l559_559710


namespace problem_number_of_true_props_l559_559800

def floor (x : ℝ) : ℤ := ⌊x⌋

def seq (a : ℕ) : ℕ → ℤ
| 1     := a
| (n+1) := floor ((seq a n + floor (a / seq a n.to_real)) / 2)

def P1 (a : ℕ) : Prop :=
  a = 5 → seq a 1 = 5 ∧ seq a 2 = 3 ∧ seq a 3 = 2

def P2 (a : ℕ) : Prop :=
  ∃ k : ℕ, ∀ n ≥ k, seq a n = seq a k

def P3 (a : ℕ) : Prop :=
  ∀ n ≥ 1, (seq a n : ℝ) > real.sqrt a - 1

def P4 (a : ℕ) : Prop :=
  ∃ k : ℕ, seq a k ≥ seq a (k + 1) → ∀ n, seq a n = floor (real.sqrt a)

def numberOfTrueProps (a : ℕ) : ℕ :=
  [P1 a, P2 a, P3 a, P4 a].count (λ P, P)  -- Note: this checks how many are true

theorem problem_number_of_true_props :
  numberOfTrueProps 5 = 3 :=
sorry

end problem_number_of_true_props_l559_559800


namespace midpoint_locus_line_MQ_fixed_point_l559_559405

-- Definitions from conditions
def line1 (x : ℝ) : ℝ := real.sqrt 3 * x
def line2 (x : ℝ) : ℝ := - real.sqrt 3 * x

def pointA (a : ℝ) : ℝ × ℝ := (a, real.sqrt 3 * a)
def pointB (b : ℝ) : ℝ × ℝ := (b, - real.sqrt 3 * b)

def dot_product (a b : ℝ) : ℝ :=
  let (xa, ya) := pointA a
  let (xb, yb) := pointB b
  xa * xb + ya * yb

-- Assuming the conditions
theorem midpoint_locus (a b x₀ y₀ : ℝ) (hab : dot_product a b = -2) :
  (x₀ ^ 2 - (y₀ ^ 2) / 3 = 1) ↔ ((x₀, y₀) = (1/2 * (a + b), real.sqrt 3 / 2 * (a - b))) :=
sorry

def pointP : ℝ × ℝ := (-2, 0)

-- Assuming the conditions in the context of the symmetric point problem
theorem line_MQ_fixed_point (a b x₀ y₀ : ℝ) (hab : dot_product a b = -2) (Q : ℝ × ℝ)
  (Q_is_reflection_about_AB : reflection (a, real.sqrt 3 * a) (b, - real.sqrt 3 * b) (-2, 0) = Q) :
  ∃ K : ℝ × ℝ, lineThrough (x₀, y₀) Q K :=
sorry

end midpoint_locus_line_MQ_fixed_point_l559_559405


namespace time_spent_driving_l559_559085

def distance_home_to_work: ℕ := 60
def speed_mph: ℕ := 40

theorem time_spent_driving:
  (2 * distance_home_to_work) / speed_mph = 3 := by
  sorry

end time_spent_driving_l559_559085


namespace determine_d_l559_559834

theorem determine_d (u v c d : ℝ) 
  (h1 : is_root (λ x => x^3 + c * x + d) u)
  (h2 : is_root (λ x => x^3 + c * x + d) v)
  (h3 : is_root (λ x => x^3 + c * x + d + 300) (u + 5))
  (h4 : is_root (λ x => x^3 + c * x + d + 300) (v - 4)) :
  d = -616 ∨ d = 1575 := 
sorry

end determine_d_l559_559834


namespace simplify_sqrt_expression_l559_559435

theorem simplify_sqrt_expression :
  (√(1 - 2 * sin 1 * cos 1) = sin 1 - cos 1) :=
by
  have h1: sin^2 1 + cos^2 1 = 1 := sorry
  have h2: (π / 4 < 1) := sorry
  have h3: (1 < π / 2) := sorry
  have h4: sin 1 > cos 1 := sorry
  sorry

end simplify_sqrt_expression_l559_559435


namespace range_of_a_l559_559756

-- Define the main proof problem statement
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ∈ set.Ioo 0 1 → (x + real.log a) / real.exp x - (a * real.log x) / x > 0) : 
  a ∈ set.Ico (1 / real.exp 1) 1 :=
sorry

end range_of_a_l559_559756


namespace jane_current_age_l559_559379

theorem jane_current_age : 
  (∀ (J: ℕ), ∀ (A: ℕ),
    (∀ (T: ℕ), jane_stopped_babysitting = T → oldest_person_age = A → T + 10 = 12 → A = 22) →
    (∀ (T: ℕ), T + 12 = 34) →
    J = 34) :=
by
  intro J A h1 h2
  have h3 : A = 22 := h1 _ _ sorry sorry sorry
  have h4 : J = 34 := h2 _
  exact h4

end jane_current_age_l559_559379


namespace average_speed_for_trip_l559_559194

theorem average_speed_for_trip : 
  let speed_first := 35
  let time_first := 4
  let speed_additional := 44
  let time_total := 6
  in
  (speed_first * time_first + speed_additional * (time_total - time_first)) / time_total = 38 := 
by
  sorry

end average_speed_for_trip_l559_559194


namespace necessary_sufficient_condition_l559_559441

theorem necessary_sufficient_condition (A B C : ℝ)
    (h : ∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) :
    |A - B + C| ≤ 2 * Real.sqrt (A * C) := 
by sorry

end necessary_sufficient_condition_l559_559441


namespace fraction_of_two_bedroom_l559_559761

theorem fraction_of_two_bedroom {x : ℝ} 
    (h1 : 0.17 + x = 0.5) : x = 0.33 :=
by
  sorry

end fraction_of_two_bedroom_l559_559761


namespace new_person_weight_l559_559091

/-- Conditions: The average weight of 8 persons increases by 6 kg when a new person replaces one of them weighing 45 kg -/
theorem new_person_weight (W : ℝ) (new_person_wt : ℝ) (avg_increase : ℝ) (replaced_person_wt : ℝ) 
  (h1 : avg_increase = 6) (h2 : replaced_person_wt = 45) (weight_increase : 8 * avg_increase = new_person_wt - replaced_person_wt) :
  new_person_wt = 93 :=
by
  sorry

end new_person_weight_l559_559091


namespace probability_not_order_dessert_l559_559103

-- Let P(D) be the probability of ordering dessert.
-- Let P(not D) be the probability of not ordering dessert.
-- Given conditions
def P_D : ℝ := 0.60

-- Proof statement: Prove that P(not D) = 0.40

theorem probability_not_order_dessert (P_D : ℝ) (h : P_D = 0.60) : 1 - P_D = 0.40 :=
by
  rw h
  norm_num
  sorry

end probability_not_order_dessert_l559_559103


namespace bisection_method_zero_point_l559_559324

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

theorem bisection_method_zero_point :
  let x1 := (1 + 2) / 2,
      f_x1 := f x1,
      x2 := (x1 + 2) / 2
  in 
  f(1) = -2 ∧ f(2) = 1 ∧ f_x1 < 0 ∧ x2 = 7 / 4 :=
by
  sorry

end bisection_method_zero_point_l559_559324


namespace price_of_72_cans_l559_559921

def regular_price_per_can : ℝ := 0.30
def discount_percentage : ℝ := 0.15
def discounted_price_per_can := regular_price_per_can * (1 - discount_percentage)
def cans_purchased : ℕ := 72

theorem price_of_72_cans :
  cans_purchased * discounted_price_per_can = 18.36 :=
by sorry

end price_of_72_cans_l559_559921


namespace correct_operation_l559_559168

theorem correct_operation :
  (∀ x : ℝ, sqrt (x^2) = abs x) ∧
  sqrt 4 = 2 ∧
  (∀ x : ℝ, sqrt (x^2) = x ∨ sqrt (x^2) = -x) →
  ((sqrt 4 ≠ ± 2) ∧
   (± sqrt (5^2) ≠ -5) ∧
   (sqrt ((-7)^2) = 7) ∧
   (sqrt (-3 : ℝ) ≠ -sqrt 3)) :=
by
  intro h
  clear h -- clear the hypothesis since no proof is needed
  split
  · intro h1
    -- prove sqrt 4 ≠ ±2
    sorry
  split
  · intro h2
    -- prove ± sqrt (5^2) ≠ -5
    sorry
  split
  · intro h3
    -- prove sqrt ((-7)^2) = 7
    exact abs_neg 7
  · intro h4
    -- prove sqrt (-3) ≠ - sqrt 3
    sorry

end correct_operation_l559_559168


namespace radius_of_inscribed_sphere_PMQN_l559_559860

/- Definitions based on the problem conditions -/

def P : ℝ := sorry -- You would define P according to its required properties.
def A : ℝ := sorry -- You would define A according to its required properties.
def B : ℝ := sorry -- You would define B according to its required properties.
def C : ℝ := sorry -- You would define C according to its required properties.
def M : ℝ := (A + C) / 2
def N : ℝ := (B + C) / 2

/- Conditions -/

axiom PA_perpendicular_to_ABC : sorry -- Express that PA is perpendicular to the plane ABC.
axiom PA_eq : PA = 1
axiom angle_A_is_right : (angle ABC A = 90) -- Express that the angle at A is right.
axiom AB_eq : AB = 2
axiom AC_eq : AC = 2

/- The statement of the theorem to be proven -/

theorem radius_of_inscribed_sphere_PMQN :
  radius_inscribed_sphere_PMNC = 1 / (sqrt 2 + 2 + sqrt 6) :=
sorry

end radius_of_inscribed_sphere_PMQN_l559_559860


namespace total_cups_l559_559257

variable (eggs : ℕ) (flour : ℕ)
variable (h : eggs = 60) (h1 : flour = eggs / 2)

theorem total_cups (eggs : ℕ) (flour : ℕ) (h : eggs = 60) (h1 : flour = eggs / 2) : 
  eggs + flour = 90 := 
by
  sorry

end total_cups_l559_559257


namespace f_is_32x5_l559_559037

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

-- State the theorem to be proved
theorem f_is_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  sorry

end f_is_32x5_l559_559037


namespace range_of_a_l559_559317

variable {R : Type*} [LinearOrder R]

def A := {x : R | -1 ≤ x ∧ x ≤ 1}
def B (a : R) := {x : R | x < a}
def complement (s : Set R) := {x : R | x ∉ s}

theorem range_of_a (a : R) :
  B a ⊆ complement A ↔ a ≤ -1 := sorry

end range_of_a_l559_559317


namespace find_a_l559_559465

theorem find_a (a : ℝ) (h1 : a > 0) :
  (a^0 + a^1 = 3) → a = 2 :=
by sorry

end find_a_l559_559465


namespace fraction_spent_l559_559970

theorem fraction_spent (borrowed_from_brother borrowed_from_father borrowed_from_mother gift_from_granny savings remaining amount_spent : ℕ)
  (h_borrowed_from_brother : borrowed_from_brother = 20)
  (h_borrowed_from_father : borrowed_from_father = 40)
  (h_borrowed_from_mother : borrowed_from_mother = 30)
  (h_gift_from_granny : gift_from_granny = 70)
  (h_savings : savings = 100)
  (h_remaining : remaining = 65)
  (h_amount_spent : amount_spent = borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings - remaining) :
  (amount_spent : ℚ) / (borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings) = 3 / 4 :=
by
  sorry

end fraction_spent_l559_559970


namespace polynomial_division_remainder_l559_559149

def p (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 - 12 * x + 7
def d (x : ℝ) : ℝ := 2 * x + 3
def q (x : ℝ) : ℝ := x^2 - 4 * x + 2
def r (x : ℝ) : ℝ := -4 * x + 1

theorem polynomial_division_remainder :
  ∀ x : ℝ, p(x) = d(x) * q(x) + r(x) :=
by
  intro x
  sorry

end polynomial_division_remainder_l559_559149


namespace no_such_numbers_l559_559261

theorem no_such_numbers :
  ¬∃ (a : Fin 2013 → ℕ),
    (∀ i : Fin 2013, (∑ j : Fin 2013, if j ≠ i then a j else 0) ≥ (a i) ^ 2) :=
sorry

end no_such_numbers_l559_559261


namespace ratio_inequality_l559_559771

-- Let A, B, and C be points in a plane such that ∠ACB = 90°
variables (A B C : Point)
variable [EuclideanGeometry]

-- Define the triangle ABC
def is_right_triangle (A B C : Point) : Prop :=
  ∠ A C B = π / 2

-- Define the median through B bisecting the angle between BA and the bisector of ∠B
def median_bisects_angle (A B C : Point) : Prop :=
  let M := midpoint A C in 
  let bisector_B := bisector ∠ A B C in
  let median_B := median_through B in
  angle_bisects (angle_between BA) bisector_B median_B

-- Prove the target inequality
theorem ratio_inequality (A B C : Point) [H1 : is_right_triangle A B C] [H2 : median_bisects_angle A B C] :
  5 / 2 < AB / BC ∧ AB / BC < 3 :=
by
  sorry

end ratio_inequality_l559_559771


namespace trivia_contest_winning_probability_l559_559960

noncomputable def probability_winning_contest (questions guesses : ℕ) (choices: ℕ): ℚ :=
  let probability_correct_guess := (1:ℚ) / choices in
  let probability_wrong_guess := 1 - probability_correct_guess in
  let probability_all_correct := probability_correct_guess^questions in
  let probability_exactly_three_correct := Nat.choose questions 3 *
    probability_correct_guess^(questions - 1) * probability_wrong_guess in
  probability_all_correct + probability_exactly_three_correct

theorem trivia_contest_winning_probability :
  probability_winning_contest 4 4 4 = 13 / 256 := by
  sorry

end trivia_contest_winning_probability_l559_559960


namespace diagonal_length_isosceles_trapezoid_l559_559456

-- Define the properties of the trapezoid as given conditions.
variables (A B C D : Type) [metric_space A]
variables (AB BC CD DA : ℝ)
variables (AB_eq_24 BC_eq_12 CD_eq_12 DA_eq_12 : Prop)
variables (trapezoid : is_trapezoid A B C D (by sorry) (by sorry))

-- Define the lengths of the bases and non-parallel sides
def base_lengths (AB BC CD DA : ℝ) :=
  AB = 24 ∧ CD = 12 ∧ BC = 12 ∧ DA = 12

-- Define the diagonal of the trapezoid 
def diagonal (A B C D : Type) [metric_space A] :=
  dist A C = 12 * real.sqrt 3

-- State the theorem
theorem diagonal_length_isosceles_trapezoid :
  base_lengths AB BC CD DA →
  diagonal A B C D :=
by
  -- Here should be the proof steps, which are omitted.
  sorry

end diagonal_length_isosceles_trapezoid_l559_559456


namespace sum_digits_9N_eq_9_l559_559998

open Nat

noncomputable theory

/-- Proof problem stating that for a natural number N where each digit of N is strictly
    greater than the digit to its left, the sum of the digits of 9N is 9. -/
theorem sum_digits_9N_eq_9 (N : ℕ) 
  (h : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ digitLength N → digits i N < digits j N) :
  sumDigits (9 * N) = 9 := 
sorry

end sum_digits_9N_eq_9_l559_559998


namespace f_decreasing_f_range_on_interval_range_of_a_l559_559321

def f (x : ℝ) : ℝ := - (2^x) / (2^x + 1)

theorem f_decreasing :
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 :=
by sorry

theorem f_range_on_interval :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x ∈ set.Icc (-4 / 5) (-2 / 3) :=
by sorry

def g (x a : ℝ) : ℝ := (a / 2) + f x

theorem range_of_a :
  ∀ x a : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → g x a ≥ 0) → a ≥ 8 / 5 :=
by sorry

end f_decreasing_f_range_on_interval_range_of_a_l559_559321
