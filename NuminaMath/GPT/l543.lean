import Mathlib

namespace leopard_arrangement_l543_543962

theorem leopard_arrangement : ∃ (f : Fin 9 → ℕ), 
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  let shortest_three := {f 0, f 1, f 2} in
  let middle_six := {f 3, f 4, f 5, f 6, f 7, f 8} in
  shortest_three = {0, 1, 2} ∧
  ∃ (perm : Finset.perm (Fin 9)),
    Finset.card shortest_three * Finset.card middle_six = 3! * 6! ∧ 
    3! * 6! = 4320 :=
by sorry

end leopard_arrangement_l543_543962


namespace max_difference_total_excess_shortfall_total_profit_l543_543245

-- Definitions of the weight differences and number of boxes, according to the conditions.
def differences := [-2, -1.5, -1, 0, 2, 2.5, 3]
def box_counts := [3, 4, 2, 2, 2, 6, 1]
def cost_price := 6
def selling_price := 8
def standard_weight := 30
def num_boxes := 20

-- Problem (1): Proving the maximum difference in weight between any two boxes.
theorem max_difference : ∃ (d1 d2 : ℝ), d1 ≠ d2 ∧ d1 ∈ differences ∧ d2 ∈ differences ∧ abs(d1 - d2) = 5 := by
  sorry

-- Problem (2): Proving the total excess or shortfall in weight compared to the standard weight.
theorem total_excess_shortfall : 
  (list.zip_with (λ d n => d * n) differences box_counts).sum = 8 := by
  sorry

-- Problem (3): Proving the total profit made from selling all the apples.
theorem total_profit :
  2 * ((standard_weight * num_boxes) + 
      (list.zip_with (λ d n => d * n) differences box_counts).sum) = 1216 := by
  sorry

end max_difference_total_excess_shortfall_total_profit_l543_543245


namespace quadrilateral_area_bounds_l543_543174

variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (S : ℝ) (a : ℝ) (α β : ℝ)
variable (isCyclicQuadrilateral : A × B × C × D → Prop)

-- Conditions
axiom cyclic_quadrilateral (ABCD : A × B × C × D) : isCyclicQuadrilateral ABCD
axiom diagonal_length (AC : Type) : AC = a
axiom angles_with_sides (AC AB AD : Type) : AC ∠ AB = α ∧ AC ∠ AD = β 
-- Proving the area S
theorem quadrilateral_area_bounds
  (ABCD : A × B × C × D)
  [isCyclicQuadrilateral ABCD]
  (AC : Type)
  [diagonal_length AC]
  [angles_with_sides AC AB AD] :
  ∀ S : ℝ, (a^2 * sin (α + β) * sin β) / (2 * sin α) ≤ S ∧ S ≤ (a^2 * sin (α + β) * sin α) / (2 * sin β) :=
begin
  sorry
end

end quadrilateral_area_bounds_l543_543174


namespace correct_propositions_count_l543_543974

noncomputable def integral_val : ℝ := ∫ x in 0..2, sqrt (4 - x^2)

def geometric (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n, a (n + 1) = r * a n

def proposition_p (a : ℕ → ℝ) :=
  (geometric a) ∧ (a 3 * a 6 = integral_val) ∧ (real.logBase π (a 4) + real.logBase π (a 5) = sqrt 2 / 2)

def proposition_q : Prop := ∀ x : ℝ, sin x ≠ 1

def neg_prop_q : Prop := ∃ x : ℝ, sin x = 1

theorem correct_propositions_count (a : ℕ → ℝ) :
  (¬ proposition_p a ∨ ¬ proposition_q) ∨
  (proposition_p a ∧ proposition_q) ∨
  (¬ proposition_p a ∧ proposition_q) ∨
  (proposition_p a ∧ ¬ proposition_q) →
  (¬ proposition_p a ∨ ¬ proposition_q) ∨ (¬ proposition_p a ∧ proposition_q) :=
sorry

end correct_propositions_count_l543_543974


namespace base_12_addition_l543_543315

theorem base_12_addition (A B: ℕ) (hA: A = 10) (hB: B = 11) : 
  8 * 12^2 + A * 12 + 2 + (3 * 12^2 + B * 12 + 7) = 1 * 12^3 + 0 * 12^2 + 9 * 12 + 9 := 
by
  sorry

end base_12_addition_l543_543315


namespace distance_between_tangent_and_parallel_line_l543_543864

noncomputable def distance_between_lines (A B C D : ℝ) (x₁ y₁ : ℝ) :=
  abs (A * x₁ + B * y₁ + C) / real.sqrt (A * A + B * B)

theorem distance_between_tangent_and_parallel_line :
  ∀ M : ℝ × ℝ,
  ∀ C : ℝ × ℝ × ℝ,
  ∀ l1 : ℝ × ℝ × ℝ,
  ∀ l_slope a l2 : ℝ × ℝ × ℝ,
  M.1 = 0 ∧ M.2 = 2 →
  (C.1 = 4 ∧ C.2 = -1 ∧ C.3 = 25) →
  (l1.1 = 4 ∧ l1.2 = a ∧ l1.3 = 2) →
  (l_slope.1 = 0 ∧ l_slope.2 = 2 ∧ l_slope.3 = (4.0 - (-4.0 / 3))) →
  (l2.1 = 4 ∧ l2.2 = -3 ∧ l2.3 = 6) →
  distance_between_lines l2.1 l2.2 l2.3 M.1 M.2 = 4 / 5 := by
    intros M C l1 l_slope a l2 hM hC hl1 hl_slope hl2
    sorry

end distance_between_tangent_and_parallel_line_l543_543864


namespace valid_parameterizations_of_line_l543_543609

theorem valid_parameterizations_of_line :
  let L := λ (x : ℝ), (4 / 3) * x - (20 / 3)
  let is_valid (p d : ℝ × ℝ) : Prop :=
    (d.1 ≠ 0 ∧ d.2 ≠ 0 ∧ 
     (∃ k : ℝ, d = (3 * k, 4 * k)) ∧ 
     p.2 = L p.1)
  (is_valid (5, 0) (-3, -4)) ∧
  (is_valid (20, 4) (9, 12)) ∧
  ¬ (is_valid (3, -7) (4 / 3, 1)) ∧
  ¬ (is_valid (17 / 4, -1) (1, 4 / 3)) ∧
  ¬ (is_valid (0, -20 / 3) (18, -24)).
sorry

end valid_parameterizations_of_line_l543_543609


namespace transformed_variance_l543_543489

variable {x : ℕ → ℝ}

-- Assume the variance of the sequence x is 3
def variance (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (1 / n) * (finset.sum (finset.range n) (λ i, (x i - finset.sum (finset.range n) (λ j, x j) / n) ^ 2))

axiom variance_seq : variance x 2009 = 3

-- Prove that the variance of the transformed sequence is 27
theorem transformed_variance : variance (λ i, 3 * (x i - 2)) 2009 = 27 := by
  sorry

end transformed_variance_l543_543489


namespace slope_of_line_l543_543868

variables {a b c : ℝ}
variables (x y : ℝ)

noncomputable def hyperbola (x y : ℝ) (a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
noncomputable def focus_position (a b : ℝ) := (sqrt (a^2 + b^2), 0)
noncomputable def asymptote_slope (a b : ℝ) := (b / a)
noncomputable def point_distance (x1 y1 x2 y2 : ℝ) := sqrt ((x2 - x1)^2 + (y2 - y1)^2)
noncomputable def slope (x1 y1 x2 y2 : ℝ) := (y2 - y1) / (x2 - x1)
noncomputable def midpoint (x1 y1 x2 y2 : ℝ) := ((x1 + x2) / 2, (y1 + y2) / 2)

theorem slope_of_line (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  let F := focus_position a b in
  let A := foot_of_perpendicular_to_asymptote F (a, b) in
  let B := intersection_with_hyperbola A in
  let h₂ : vector_equality (F, B, A) in
  slope (fst F) (snd F) (fst A) (snd A) = 1 :=
sorry

end slope_of_line_l543_543868


namespace distance_h_eq_OK_cos_theta_l543_543176

-- Let O1 and O2 be circles that intersect at points P and Q
-- Let A, B lie on O1 and C, D lie on O2
-- Let M and N be the midpoints of AD and BC, respectively
-- Let O be the midpoint of O1O2
-- Let K be the midpoint of PQ

variables {O1 O2 P Q A B C D : Type} 
variables [Inhabited O1] [Inhabited O2] [Inhabited P] [Inhabited Q]
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Define the geometric points and properties
variables (midpoint : Type → Type → Type)
variables (lies_on : Type → Type → Prop)
variables (angle : Type → Type → Type → ℝ)
constants
  (APB : Prop) (CPD : Prop)
  (M N O K : Type) (θ h OK : ℝ)

-- Define the conditions
axioms 
  (AP_lies_on_O1: lies_on A O1)
  (CP_lies_on_O1: lies_on C O1)
  (BP_lies_on_O2: lies_on B O2)
  (DP_lies_on_O2: lies_on D O2)
  (M_is_midpoint_AD: midpoint M (A × D))
  (N_is_midpoint_BC: midpoint N (B × C))
  (O_is_midpoint_O1O2: midpoint O (O1 × O2))
  (K_is_midpoint_PQ: midpoint K (P × Q))
  (acute_angle_theta: 0 < θ ∧ θ < π / 2)
  (distance_O_to_MN: ∃ h, h = by sorry)

-- The final proof statement
theorem distance_h_eq_OK_cos_theta : h = OK * cos θ := by sorry

end distance_h_eq_OK_cos_theta_l543_543176


namespace parabola_properties_l543_543236

open Real

-- Given conditions
def vertex_origin : Prop := (0, 0) ∈ C
def focus_on_x_axis : Prop := ∃ α ∈ ℝ, α ≠ 0 ∧ ∃ p ∈ ℝ, p = (α, 0)
def passes_through_P : Prop := (2, 2) ∈ C

-- Main theorem
theorem parabola_properties (vertex_origin : Prop)
    (focus_on_x_axis : Prop)
    (passes_through_P : Prop)
    : ∃ (eq_C : ℝ → ℝ → Prop) (focus_C : ℝ × ℝ),
    (∀ x y, eq_C x y ↔ y ^ 2 = 2 * x ) ∧ focus_C = (1/2, 0) ∧
    ∀ M N, (M ∈ eq_C → N ∈ eq_C → l M N = 0) → |MN| = 2 * sqrt 6 := sorry


end parabola_properties_l543_543236


namespace maximizeSum_l543_543075

open BigOperators

-- Define the arithmetic sequence and conditions
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable d a₁ : ℝ

-- Define properties of the sequence and sum
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) := ∀ n, a (n + 1) = a n + d

def sumOfFirstNTerms (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

-- Given conditions
axiom h1 : a 0 + a 1 + a 2 = 156
axiom h2 : a 1 + a 2 + a 3 = 147
axiom h3 : isArithmeticSeq a d
axiom h4 : S = sumOfFirstNTerms a

-- Prove that n = 19 maximizes S_n
theorem maximizeSum : ∃ n, S n = - (3 / 2) * n^2 + (113 / 2) * n ∧ n = 19 :=
by
  sorry

end maximizeSum_l543_543075


namespace cos_double_angle_l543_543379

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l543_543379


namespace handshakes_six_couples_l543_543992

/-- At a gathering of six couples, each person shakes hands with everyone else 
    except for their spouse and the first new person they meet. 
    Prove the total number of handshakes exchanged is 54. -/
theorem handshakes_six_couples : 
  let total_people := 12
  let shakes_per_person := 9
  in (total_people * shakes_per_person) / 2 = 54 :=
by
  let total_people := 12
  let shakes_per_person := 9
  calc
    (total_people * shakes_per_person) / 2 = (12 * 9) / 2 := rfl
    ... = 108 / 2 := rfl
    ... = 54 := rfl

end handshakes_six_couples_l543_543992


namespace evaluate_fraction_l543_543788

-- Define the custom operations x@y and x#y
def op_at (x y : ℝ) : ℝ := x * y - y^2
def op_hash (x y : ℝ) : ℝ := x + y - x * y^2 + x^2

-- State the proof goal
theorem evaluate_fraction : (op_at 7 3) / (op_hash 7 3) = -3 :=
by
  -- Calculations to prove the theorem
  sorry

end evaluate_fraction_l543_543788


namespace sum_common_seq_l543_543002

-- Define the arithmetic sequences
def seq1 (n : ℕ) : ℕ := 2 * n - 1
def seq2 (n : ℕ) : ℕ := 3 * n - 2

-- Define the common term sequence
def common_seq (a : ℕ) : Prop :=
  ∃ (n1 n2 : ℕ), seq1 n1 = a ∧ seq2 n2 = a

-- Prove the sum of the first n terms of the sequence common_seq is 3n^2 - 2n
theorem sum_common_seq (n : ℕ) : ∃ S : ℕ, 
  S = (3 * n ^ 2 - 2 * n) ∧ 
  ∀ (a i : ℕ), (a = 1 + 6 * (i - 1)) → i ∈ (fin n) → common_seq a :=
sorry

end sum_common_seq_l543_543002


namespace feet_of_perpendiculars_form_equilateral_l543_543200

-- Define an equilateral triangle
structure EquilateralTriangle (A B C : Type) :=
  (is_equilateral : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
                    ∀ (a b c : Type), a ≠ b ∧ b ≠ c ∧ c ≠ a → a = b ∧ b = c ∧ c = a)

-- Define the property about feet of perpendiculars forming an equilateral triangle
theorem feet_of_perpendiculars_form_equilateral
  {A B C A1 B1 C1 : Type}
  (hABC : EquilateralTriangle A B C)
  (hA1 : ∃ P, foot_of_perpendicular A B C P = A1)
  (hB1 : ∃ P, foot_of_perpendicular B C A P = B1)
  (hC1 : ∃ P, foot_of_perpendicular C A B P = C1) :
  EquilateralTriangle A1 B1 C1 := 
sorry

end feet_of_perpendiculars_form_equilateral_l543_543200


namespace candies_equal_in_time_l543_543575

theorem candies_equal_in_time (children : ℕ) (candies : ℕ → ℕ) (initial_even : ∀ i, (candies i) % 2 = 0) :
  ∃ k : ℕ, ∀ i j, 
  (distribute_candies (candies i) k) = (distribute_candies (candies j) k) :=
  sorry

def distribute_candies (candies : ℕ) (rounds : ℕ) : ℕ :=
  -- Function to distribute candies based on the problem's description
  sorry

end candies_equal_in_time_l543_543575


namespace relation_among_abc_l543_543852

noncomputable def f (x : ℝ) : ℝ :=
  ((1 / 3) ^ |x|) + 2

def a : ℝ := f (-real.log 4 / real.log 3)
def b : ℝ := f (real.log 5 / real.log 2)
def c : ℝ := f 0

theorem relation_among_abc : b < a ∧ a < c :=
by
  -- Since the logarithmic values have already been compared in the solution
  have h1 : 0 < real.log 4 / real.log 3,
  { sorry },
  have h2 : real.log 4 / real.log 3 < real.log 5 / real.log 2,
  { sorry },
  have h3 : 0 < real.log 5 / real.log 2,
  { sorry },
  -- From the given piecewise decreasing nature of the function
  have fa : a = f (real.log 4 / real.log 3),
  { sorry },
  have fb : b < f (real.log 4 / real.log 3),
  { sorry },
  have fc : a < c,
  { sorry },
  exact ⟨fb, fc⟩

end relation_among_abc_l543_543852


namespace eggs_not_eaten_is_6_l543_543981

noncomputable def eggs_not_eaten_each_week 
  (trays_purchased : ℕ) 
  (eggs_per_tray : ℕ) 
  (eggs_morning : ℕ) 
  (days_in_week : ℕ) 
  (eggs_night : ℕ) : ℕ :=
  let total_eggs := trays_purchased * eggs_per_tray
  let eggs_eaten_son_daughter := eggs_morning * days_in_week
  let eggs_eaten_rhea_husband := eggs_night * days_in_week
  let eggs_eaten_total := eggs_eaten_son_daughter + eggs_eaten_rhea_husband
  total_eggs - eggs_eaten_total

theorem eggs_not_eaten_is_6 
  (trays_purchased : ℕ := 2) 
  (eggs_per_tray : ℕ := 24) 
  (eggs_morning : ℕ := 2) 
  (days_in_week : ℕ := 7) 
  (eggs_night : ℕ := 4) : 
  eggs_not_eaten_each_week trays_purchased eggs_per_tray eggs_morning days_in_week eggs_night = 6 :=
by
  -- Here should be proof steps, but we use sorry to skip it as per instruction
  sorry

end eggs_not_eaten_is_6_l543_543981


namespace radius_of_wheel_l543_543238

theorem radius_of_wheel (D : ℝ) (n : ℕ) (r : ℝ) (π_val : ℝ) 
  (cond1 : D = 915.2) (cond2 : n = 650) (approx_pi : π_val ≈ 3.14159) :
  r ≈ D / (2 * π_val * ↑n) :=
by
  sorry

end radius_of_wheel_l543_543238


namespace count_legs_l543_543470

theorem count_legs (total_animals : ℕ) (ducks : ℕ) (legs_duck : ℕ) (legs_horse : ℕ) :
  total_animals = 11 -> ducks = 7 -> legs_duck = 2 -> legs_horse = 4 -> 
  2 * ducks + 4 * (total_animals - ducks) = 30 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end count_legs_l543_543470


namespace terminal_side_of_alpha_second_quadrant_l543_543116

theorem terminal_side_of_alpha_second_quadrant 
  (α : ℝ) 
  (h1 : tan α < 0) 
  (h2 : cos α < 0) : 
  quadrant_of_angle α = 2 := 
sorry

end terminal_side_of_alpha_second_quadrant_l543_543116


namespace negation_equiv_l543_543465

open Nat

theorem negation_equiv (P : Prop) :
  (¬ (∃ n : ℕ, (n! * n!) > (2^n))) ↔ (∀ n : ℕ, (n! * n!) ≤ (2^n)) :=
by
  sorry

end negation_equiv_l543_543465


namespace area_of_quadrilateral_PF1QF2_l543_543837

theorem area_of_quadrilateral_PF1QF2 (x y : ℝ) (F1 F2 P Q : ℝ×ℝ) 
  (h1 : ∀ p : ℝ×ℝ, p ∈ set_of (λ q, q.1^2/16 + q.2^2/4 = 1))
  (h2 : F1 = (4, 0) ∧ F2 = (-4, 0)) 
  (h3 : Q = (-P.1, -P.2))
  (h4 : dist P Q = dist F1 F2) :
  let a := 8 in
  let c := 4 in
  let b_sq := a^2 - c^2 in
  let m := |dist P F1| in
  let n := |dist P F2| in
  m * n = 8 :=
by sorry

end area_of_quadrilateral_PF1QF2_l543_543837


namespace intersection_points_range_l543_543337

theorem intersection_points_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ a = x₁^3 - 3 * x₁ ∧
  a = x₂^3 - 3 * x₂ ∧ a = x₃^3 - 3 * x₃) ↔ (-2 < a ∧ a < 2) :=
sorry

end intersection_points_range_l543_543337


namespace roots_of_polynomial_l543_543032

theorem roots_of_polynomial :
  {x : ℝ | x^10 - 5*x^8 + 4*x^6 - 64*x^4 + 320*x^2 - 256 = 0} = {x | x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2} :=
by
  sorry

end roots_of_polynomial_l543_543032


namespace cyclist_club_member_count_l543_543123

-- Define the set of valid digits.
def valid_digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 9}

-- Define the problem statement
theorem cyclist_club_member_count : valid_digits.card ^ 3 = 512 :=
by
  -- Placeholder for the proof
  sorry

end cyclist_club_member_count_l543_543123


namespace minimum_value_of_f_l543_543048

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + x - 2 * Real.log x

theorem minimum_value_of_f : ∃ x : ℝ, f(x) = 3/2 ∧ ∀ y : ℝ, f y ≥ f x := by
  let x_min := 1
  have h_derivative : ∀ x : ℝ, (1/2) * x^2 + x - 2 * Real.log x ≥ 3/2 := sorry
  use x_min
  split
  { 
    -- Prove that f(1) = 3/2
    -- This should be calculated
    have h_fx_min := calc
      f 1 = (1/2) * 1^2 + 1 - 2 * Real.log 1 : by ring
      ... = 1/2 + 1 - 2 * 0 : by { congr, norm_num }
      ... = 3/2 : by norm_num
    exact h_fx_min
  }
  {
    -- Prove that f(x) >= 3/2 for all x
    exact h_derivative
  }

end minimum_value_of_f_l543_543048


namespace Liam_savings_after_trip_and_bills_l543_543546

theorem Liam_savings_after_trip_and_bills :
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  total_savings - bills_cost - trip_cost = 1500 := by
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  sorry

end Liam_savings_after_trip_and_bills_l543_543546


namespace findPrincipal_l543_543052

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

# Check for noncomputable
noncomputable def simpleInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T

theorem findPrincipal :
  SI = 500 →
  R = 0.05 →
  T = 1 →
  simpleInterest P R T = 500 →
  P = 10000 := by
  intros hSI hR hT hSimpleInterest
  sorry

end findPrincipal_l543_543052


namespace fn_2011_equals_sin_l543_543168

noncomputable def fn : ℕ → (ℝ → ℝ)
| 0       := λ x, Real.cos x
| (n + 1) := λ x, (fn n x).deriv

theorem fn_2011_equals_sin (x : ℝ) : fn 2011 x = Real.sin x :=
by sorry

end fn_2011_equals_sin_l543_543168


namespace B_participated_in_street_dance_l543_543139

-- Define the clubs
inductive Club
| street_dance
| animation
| instrumental

open Club

-- Define the participation of students
structure Participation :=
  (A B C : Club → Prop)

-- Define the conditions
variables (p : Participation)
def condition1 := ∃ n : ℕ, 
  (λ num_clubs, (num_clubs A) > (num_clubs B) ∧ ¬ num_clubs A animation)
def condition2 := ¬ p.B instrumental
def condition3 := ∀ x, x x x ↔ x x x

-- The theorem to prove
theorem B_participated_in_street_dance 
  (h1 : condition1 p)
  (h2 : condition2 p)
  (h3 : condition3 p):
  p.B street_dance :=
sorry

end B_participated_in_street_dance_l543_543139


namespace find_x_plus_y_l543_543074

variables {x y : ℚ}

def a : ℚ × ℚ × ℚ := (3, -2, 4)
def b : ℚ × ℚ × ℚ := (1, x, y)
def parallel (u v : ℚ × ℚ × ℚ) : Prop := ∃ (λ : ℚ), b = (λ * u.1, λ * u.2, λ * u.3)

theorem find_x_plus_y (h : parallel a b) : x + y = 7 := sorry

end find_x_plus_y_l543_543074


namespace problem_solution_l543_543077

variable (f : ℝ → ℝ)

noncomputable def example_problem :=
∀ x : ℝ, differentiable ℝ f x

theorem problem_solution
  (h_diff : ∀ x : ℝ, differentiable ℝ f x)
  (h_deriv : ∀ x : ℝ, deriv f x < f x) :
  f 1 < ℯ * f 0 ∧ f 2014 < (ℯ ^ 2014) * f 0 :=
sorry

end problem_solution_l543_543077


namespace cos_double_angle_l543_543399

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l543_543399


namespace number_of_valid_arithmetic_sequences_l543_543330

theorem number_of_valid_arithmetic_sequences : 
  ∃ S : Finset (Finset ℕ), 
  S.card = 16 ∧ 
  ∀ s ∈ S, s.card = 3 ∧ 
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ s = {a, b, c} ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 
  (b - a = c - b) ∧ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) := 
sorry

end number_of_valid_arithmetic_sequences_l543_543330


namespace constant_remainder_l543_543363

open Polynomial

noncomputable def find_a (f g : Polynomial ℝ) : ℝ :=
  let q := f /ₘ g
  let r := f %ₘ g
  if r.degree = 0 then r.coeff 0 else 0

theorem constant_remainder (a : ℝ) :
  find_a (10 * X ^ 3 - 7 * X ^ 2 + a * X + 10) (2 * X ^ 2 - 5 * X + 2) = 10 - (45 - a) := sorry

end constant_remainder_l543_543363


namespace circle_center_coordinates_l543_543596

theorem circle_center_coordinates :
  let p1 := (2, -3)
  let p2 := (8, 9)
  let midpoint (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  midpoint (2 : ℝ) (-3) 8 9 = (5, 3) :=
by
  sorry

end circle_center_coordinates_l543_543596


namespace green_other_side_l543_543282
open Classical

def numBlueBlueCards : Nat := 4
def numBlueGreenCards : Nat := 2
def numGreenGreenCards : Nat := 2

noncomputable def totalGreenSides : Nat := (2 * numBlueGreenCards) + (2 * numGreenGreenCards)
noncomputable def greenGreenSides : Nat := 2 * numGreenGreenCards

theorem green_other_side (h : totalGreenSides = 6 ∧ greenGreenSides = 4) :
  probability_that_other_side_is_green : ℚ :=
  by
    unfold totalGreenSides at h
    unfold greenGreenSides at h
    have h1 : totalGreenSides = 6 := h.1
    have h2 : greenGreenSides = 4 := h.2
    
    sorry
  
#check green_other_side

end green_other_side_l543_543282


namespace solve_first_equation_solve_second_equation_l543_543208

theorem solve_first_equation (x : ℝ) : (8 * x = -2 * (x + 5)) → (x = -1) :=
by
  intro h
  sorry

theorem solve_second_equation (x : ℝ) : ((x - 1) / 4 = (5 * x - 7) / 6 + 1) → (x = -1 / 7) :=
by
  intro h
  sorry

end solve_first_equation_solve_second_equation_l543_543208


namespace find_a_l543_543446

-- Define the function f
def f (a x : ℝ) := Real.log 2 (a / (x^2 + 1))

-- Prove the given condition
theorem find_a :
  (∃ (a : ℝ), ∀ y ∈ Set.Icc (-1 : ℝ) (⊤ : ℝ), ∃ x ∈ Set.univ, f a x = y) →
  a = 2 :=
begin
  -- The proof is omitted (as instructed).
  sorry
end

end find_a_l543_543446


namespace angle_between_refracted_rays_l543_543252

noncomputable def snell_law (n1 n2 : ℝ) (θ1 θ2 : ℝ) := 
  n1 * sin θ1 = n2 * sin θ2

noncomputable def angle_of_refraction_first_ray := (30 : ℝ)
noncomputable def refractive_index := (1.6 : ℝ)

noncomputable def compute_angle_between_refracted_rays (β γ : ℝ) : ℝ :=
  β + γ

theorem angle_between_refracted_rays : 
  compute_angle_between_refracted_rays angle_of_refraction_first_ray (Real.arcsin (Real.sin (36.87 * (Real.pi / 180.0)) / refractive_index)) = 52.0 := 
sorry

end angle_between_refracted_rays_l543_543252


namespace probability_A_off_at_A2_probability_A_B_not_same_stop_l543_543916

-- Given stops A0, A1, A2, A3, A4, A5
inductive Stop
| A0 | A1 | A2 | A3 | A4 | A5

open Stop

-- Passengers A and B can get off at any stop A_i for i ∈ {1, 2, 3, 4, 5}
def stops : List Stop := [A1, A2, A3, A4, A5]

-- Define probability as 1 divided by the length of the list of stops
def probability_of_getting_off_at (s : Stop) : ℚ :=
  if s ∈ stops then 1 / stops.length else 0

-- Event A: Passenger A gets off at stop A2
noncomputable def P_A : ℚ := probability_of_getting_off_at A2

-- Event B: Passengers A and B do not get off at the same stop
noncomputable def P_B : ℚ :=
  1 - (1 / (stops.length * stops.length))

-- Proof statement
theorem probability_A_off_at_A2 : P_A = 1 / 5 := 
  by sorry

theorem probability_A_B_not_same_stop : P_B = 4 / 5 := 
  by sorry

end probability_A_off_at_A2_probability_A_B_not_same_stop_l543_543916


namespace ratio_germany_higher_future_cost_policy_uncertain_l543_543696

-- Define the ratios in Russia and Germany
def insurance_ratio_russia : ℝ := 1500000 / 23000
def insurance_ratio_germany : ℝ := 3000000 / 80

-- Define the statement that the ratio in Germany is significantly higher than in Russia
theorem ratio_germany_higher (threshold : ℝ) : insurance_ratio_germany > insurance_ratio_russia + threshold := sorry

-- Define future cost of insurance policy influenced by demand and supply increase
theorem future_cost_policy_uncertain (demand_increase supply_increase : ℝ) : 
  ∃ (future_cost : ℝ), (future_cost = demand_increase + supply_increase) ∨ (future_cost = demand_increase - supply_increase) := sorry

end ratio_germany_higher_future_cost_policy_uncertain_l543_543696


namespace find_150th_letter_l543_543669

def repeating_sequence : String := "ABCD"

def position := 150

theorem find_150th_letter :
  repeating_sequence[(position % 4) - 1] = 'B' := 
sorry

end find_150th_letter_l543_543669


namespace range_of_ab_l543_543849

theorem range_of_ab (a b : ℝ) 
  (h1: ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → (2 * a * x - b * y + 2 = 0)) : 
  ab ≤ 0 :=
sorry

end range_of_ab_l543_543849


namespace convergent_fraction_solution_l543_543175

theorem convergent_fraction_solution (d a₀ a₁ ... aₖ₋₁ : ℤ) (k : ℕ)
  (h_period : ∃ n, n ≠ 0 ∧ ([a₀; a₁, ..., aₖ₋₁, 2 * a₀, a₁, ...] = [a₀; a₁, ..., aₖ₋₁].cycle n)) :
  ∃ pₖ₋₁ qₖ₋₁ : ℤ, (pₖ₋₁^2 - d * qₖ₋₁^2 = (-1)^k) :=
by
  sorry

end convergent_fraction_solution_l543_543175


namespace fraction_product_simplified_l543_543769

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end fraction_product_simplified_l543_543769


namespace sidney_thursday_jacks_l543_543204

open Nat

-- Define the number of jumping jacks Sidney did on each day
def monday_jacks := 20
def tuesday_jacks := 36
def wednesday_jacks := 40

-- Define the total number of jumping jacks done by Sidney
-- on Monday, Tuesday, and Wednesday
def sidney_mon_wed_jacks := monday_jacks + tuesday_jacks + wednesday_jacks

-- Define the total number of jumping jacks done by Brooke
def brooke_jacks := 438

-- Define the relationship between Brooke's and Sidney's total jumping jacks
def sidney_total_jacks := brooke_jacks / 3

-- Prove the number of jumping jacks Sidney did on Thursday
theorem sidney_thursday_jacks :
  sidney_total_jacks - sidney_mon_wed_jacks = 50 :=
by
  sorry

end sidney_thursday_jacks_l543_543204


namespace find_a_l543_543166

def f(x : ℚ) : ℚ := x / 3 + 2
def g(x : ℚ) : ℚ := 5 - 2 * x

theorem find_a (a : ℚ) (h : f (g a) = 4) : a = -1 / 2 :=
by
  sorry

end find_a_l543_543166


namespace sum_common_seq_l543_543003

-- Define the arithmetic sequences
def seq1 (n : ℕ) : ℕ := 2 * n - 1
def seq2 (n : ℕ) : ℕ := 3 * n - 2

-- Define the common term sequence
def common_seq (a : ℕ) : Prop :=
  ∃ (n1 n2 : ℕ), seq1 n1 = a ∧ seq2 n2 = a

-- Prove the sum of the first n terms of the sequence common_seq is 3n^2 - 2n
theorem sum_common_seq (n : ℕ) : ∃ S : ℕ, 
  S = (3 * n ^ 2 - 2 * n) ∧ 
  ∀ (a i : ℕ), (a = 1 + 6 * (i - 1)) → i ∈ (fin n) → common_seq a :=
sorry

end sum_common_seq_l543_543003


namespace find_k_carboxylic_l543_543921

def is_k_carboxylic (num k : ℕ) : Prop :=
  ∃ (seq : List ℕ), seq.length = k ∧ (∀ n ∈ seq, n > 9 ∧ ∃ d, n = List.repeat d (Nat.digits 10 n).length / List.dedup (Nat.digits 10 n).length = 1) ∧ seq.sum = num

theorem find_k_carboxylic : ∃ k : ℕ, is_k_carboxylic 8002 k ∧ ∀ m : ℕ, is_k_carboxylic 8002 m → k ≤ m :=
begin
  use 14,
  split,
  { sorry },  -- Proof that 8002 is 14-carboxylic
  { intros m h,
    sorry }  -- Proof that there is no smaller k
end

end find_k_carboxylic_l543_543921


namespace shortest_distance_to_parabola_l543_543819

-- Definitions for given conditions
def point : ℝ × ℝ := (10, 8)

def parabola_y (x : ℝ) : ℝ := x^2 / 4

-- Main theorem statement
theorem shortest_distance_to_parabola : 
  let dist (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) in
  let candidate_points := [(4, parabola_y 4), (16, parabola_y 16)] in
  ((dist point (4, parabola_y 4)) ≤ (dist point (16, parabola_y 16))) :=
by sorry

end shortest_distance_to_parabola_l543_543819


namespace miles_per_book_l543_543557

theorem miles_per_book (total_miles : ℝ) (books_read : ℝ) (miles_per_book : ℝ) : 
  total_miles = 6760 ∧ books_read = 15 → miles_per_book = 450.67 := 
by
  sorry

end miles_per_book_l543_543557


namespace tan_A_tan_B_eq_4_l543_543505

theorem tan_A_tan_B_eq_4
  (A B C H F : Point)
  (triangle_ABC : Triangle A B C)
  (H_is_orthocenter : isOrthocenter H triangle_ABC)
  (HF_eq_8 : dist H F = 8)
  (HC_eq_24 : dist H C = 24)
  (CF_eq_32 : dist C F = 32)
  (angle_AFH_eq_90 : angle A F H = 90)
  (angle_AHF_eq_B : angle A H F = angle A B C) :
  tan (angle A B C) * tan (angle B A C) = 4 :=
sorry

end tan_A_tan_B_eq_4_l543_543505


namespace largest_power_of_2_factor_80_l543_543011

noncomputable def largest_power_of_2_factor (N : ℝ) : ℕ :=
  let p := ∑ k in (Finset.range 8).map (Nat.succ).toFinset, 2 * (k : ℝ) * Real.log k
  let n := Real.exp p
  if h : (nat.pow 2 80) ∣ Int.floor n then
    80
  else
    0 -- This case will never occur in the context of the given problem, used for completeness

theorem largest_power_of_2_factor_80 :
  largest_power_of_2_factor (Real.exp (∑ k in (Finset.range 8).map (Nat.succ).toFinset, 2 * (k : ℝ) * Real.log k)) = 80 :=
sorry

end largest_power_of_2_factor_80_l543_543011


namespace find_grazing_months_l543_543268

def oxen_months_A := 10 * 7
def oxen_months_B := 12 * 5
def total_rent := 175
def rent_C := 45

def proportion_equation (x : ℕ) : Prop :=
  45 / 175 = (15 * x) / (oxen_months_A + oxen_months_B + 15 * x)

theorem find_grazing_months (x : ℕ) (h : proportion_equation x) : x = 3 :=
by
  -- We will need to involve some calculations leading to x = 3
  sorry

end find_grazing_months_l543_543268


namespace fraction_integer_condition_special_integers_l543_543261

theorem fraction_integer_condition (p : ℕ) (h : (p + 2) % (p + 1) = 0) : p = 2 :=
by
  sorry

theorem special_integers (N : ℕ) (h1 : ∀ q : ℕ, N = 2 ^ p * 3 ^ q ∧ (2 * p + 1) * (2 * q + 1) = 3 * (p + 1) * (q + 1)) : 
  N = 144 ∨ N = 324 :=
by
  sorry

end fraction_integer_condition_special_integers_l543_543261


namespace max_n_plus_m_l543_543228

theorem max_n_plus_m : 
  ∀ (m n : ℝ), (∀ x ∈ set.Icc m n, 3 ≤ 2^(2 * x) - 2^(x + 2) + 7 ∧ 2^(2 * x) - 2^(x + 2) + 7 ≤ 7) → n + m = 3 :=
begin
  intros m n h,
  sorry
end

end max_n_plus_m_l543_543228


namespace ellipse_equation_l543_543447

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : ℝ)
  (h3 : eccentricity = 1 / 2) 
  (h4 : (1 : ℝ)^2 / a^2 + (3/2)^2 / b^2 = 1) :
  (a = 2) ∧ (b = sqrt(3)) ∧ (eccentricity = 1 / 2) ∧ (∀ (m n : ℝ), 
  (3 * m^2 + 4 * n^2 = 12) → (|sqrt((m - 1)^2 + (n - 1)^2)| = 2)) := sorry

end ellipse_equation_l543_543447


namespace shampoo_time_l543_543558

-- Time spent hosing off the dog
def t_h : ℕ := 10

-- Number of times the dog is shampooed
def n_s : ℕ := 3

-- Total time spent cleaning the dog
def t_total : ℕ := 55

-- Time to shampoo the dog once
def t_s (t_h n_s t_total : ℕ) : ℕ := 
  (t_total - t_h) / n_s

theorem shampoo_time {t_h n_s t_total : ℕ} (h : t_h = 10) (n : n_s = 3) (tt : t_total = 55)
  : t_s t_h n_s t_total = 15 :=
by
  rw [h, n, tt]
  rfl
  sorry -- further intermediate steps would follow here if constructing the full proof

end shampoo_time_l543_543558


namespace probability_of_two_pair_is_correct_l543_543652

def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes : ℕ := binomial_coefficient 52 5

def successful_two_pair_outcomes : ℕ :=
  13 * binomial_coefficient 4 2 * 12 * binomial_coefficient 4 2 * 11 * binomial_coefficient 4 1

def probability_two_pair : ℚ := successful_two_pair_outcomes / total_outcomes

theorem probability_of_two_pair_is_correct :
  probability_two_pair = 342 / 7205 :=
by {
  -- This step would involve expanding the left-hand side and simplifying, but it is omitted here.
  sorry
}

end probability_of_two_pair_is_correct_l543_543652


namespace AB_is_not_shortest_side_l543_543965

variable (A B C : Point)

-- Definitions and conditions
variable (median_a : Line) (median_b : Line)
variable (is_perpendicular : Perpendicular median_a median_b)

-- The proof statement
theorem AB_is_not_shortest_side (h : perpendicular median_a median_b) : 
( ¬ shortest_side A B C A B) := sorry

end AB_is_not_shortest_side_l543_543965


namespace find_denomination_of_bills_l543_543648

variables 
  (bills_13 : ℕ)  -- Denomination of the bills Tim has 13 of
  (bills_5 : ℕ := 5)  -- Denomination of the bills Tim has 11 of, which are $5 bills
  (bills_1 : ℕ := 1)  -- Denomination of the bills Tim has 17 of, which are $1 bills
  (total_amt : ℕ := 128)  -- Total amount Tim needs to pay
  (num_bills_13 : ℕ := 13)  -- Number of bills of unknown denomination
  (num_bills_5 : ℕ := 11)  -- Number of $5 bills
  (num_bills_1 : ℕ := 17)  -- Number of $1 bills
  (min_bills : ℕ := 16)  -- Minimum number of bills to be used

theorem find_denomination_of_bills : 
  num_bills_13 * bills_13 + num_bills_5 * bills_5 + num_bills_1 * bills_1 = total_amt →
  num_bills_13 + num_bills_5 + num_bills_1 ≥ min_bills → 
  bills_13 = 4 :=
by
  intros h1 h2
  sorry

end find_denomination_of_bills_l543_543648


namespace value_of_expression_at_midpoint_l543_543544

-- Define the points D and E
def D : ℝ × ℝ := (10, 9)
def E : ℝ × ℝ := (4, 6)

-- Define the midpoint F of D and E
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def F : ℝ × ℝ := midpoint D E

-- Define the expression to be evaluated at point F
def expression (P : ℝ × ℝ) : ℝ := 3 * P.1 - 2 * P.2

-- State the theorem
theorem value_of_expression_at_midpoint : expression F = 6 :=
sorry

end value_of_expression_at_midpoint_l543_543544


namespace common_ratio_of_progression_bounds_l543_543241

theorem common_ratio_of_progression_bounds (a : ℝ) (q : ℝ) (h_a_pos : 0 < a) (h_q_ge_one : 1 ≤ q)
(h_triangle_ineq1 : a + a * q > a * q^2) (h_triangle_ineq2 : a + a * q^2 > a * q) (h_triangle_ineq3 : a * q + a * q^2 > a) :
  (sqrt 5 - 1) / 2 ≤ q ∧ q ≤ (sqrt 5 + 1) / 2 :=
by sorry

end common_ratio_of_progression_bounds_l543_543241


namespace total_salmon_now_l543_543328

def initial_salmon : ℕ := 500

def increase_factor : ℕ := 10

theorem total_salmon_now : initial_salmon * increase_factor = 5000 := by
  sorry

end total_salmon_now_l543_543328


namespace length_of_room_l543_543158

theorem length_of_room (Area Width Length : ℝ) (h1 : Area = 10) (h2 : Width = 2) (h3 : Area = Length * Width) : Length = 5 :=
by
  sorry

end length_of_room_l543_543158


namespace line_passing_thru_point_l543_543887

variable (k : ℝ) (O A B C : Point ℝ)
variable (V_OA V_OB V_OC : Vector ℝ)

def collinear (A B C : Point ℝ) : Prop :=
-- Define collinear property (determinant of matrix formed by points is 0)

def vector (O A : Point ℝ) : Vector ℝ := -- Define vector as difference of points

theorem line_passing_thru_point :
  (V_OA = (k, 12)) ∧ (V_OB = (4, 5)) ∧ (V_OC = (10, k)) ∧ collinear A B C ∧ k < 0 →
  ∃ eq : Line ℝ, eq = (2, 1, -3) :=
sorry

end line_passing_thru_point_l543_543887


namespace question_correct_statements_l543_543134

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (f : ℝ → ℝ) : ∀ x y : ℝ, f (x + y) = f x + f y
axiom periodicity (f : ℝ → ℝ) : f 2 = 0

theorem question_correct_statements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ -- ensuring the function is periodic
  (∀ x : ℝ, f x = -f (-x)) ∧ -- ensuring the function is odd
  (∀ x : ℝ, f (x+2) = -f (-x)) :=  -- ensuring symmetry about point (1,0)
by
  -- We'll prove this using the conditions given and properties derived from it
  sorry 

end question_correct_statements_l543_543134


namespace equilateral_triangle_segments_l543_543710

theorem equilateral_triangle_segments (a x y : ℝ)
  (h1 : ∃ ABC : set ℝ, equilateral ABC ∧ side_length ABC = a)
  (h2 : ∃ O : ℝ, is_center O ABC)
  (h3 : ∃ A1 B1 : ℝ, line_through O A1 B1)
  (h4 : cuts_segments A1 B1 AC x)
  (h5 : cuts_segments B1 A1 BC y)
  (h6 : x = segment_len AB1 A)
  (h7 : y = segment_len BA1 B):
  3 * x * y - 2 * a * (x + y) + a^2 = 0 := 
sorry

end equilateral_triangle_segments_l543_543710


namespace triangle_isosceles_l543_543155

theorem triangle_isosceles
  (A B C : ℝ) -- Angles of the triangle, A, B, and C
  (h1 : A = 2 * C) -- Condition 1: Angle A equals twice angle C
  (h2 : B = 2 * C) -- Condition 2: Angle B equals twice angle C
  (h3 : A + B + C = 180) -- Sum of angles in a triangle equals 180 degrees
  : A = B := -- Conclusion: with the conditions above, angles A and B are equal
by
  sorry

end triangle_isosceles_l543_543155


namespace lines_parallel_if_perpendicular_to_plane_l543_543467

axiom line : Type
axiom plane : Type

-- Definitions of perpendicular and parallel
axiom perp : line → plane → Prop
axiom parallel : line → line → Prop

variables (a b : line) (α : plane)

theorem lines_parallel_if_perpendicular_to_plane (h1 : perp a α) (h2 : perp b α) : parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_plane_l543_543467


namespace negation_of_implication_l543_543232

theorem negation_of_implication (a b : ℝ) :
  (a > b → 2^a > 2^b) → (a ≤ b → 2^a ≤ 2^b) :=
by
  intro h ha_le_b
  sorry

end negation_of_implication_l543_543232


namespace cos_double_angle_l543_543380

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l543_543380


namespace part1_part2_part3_l543_543856

-- Define the sequence a_n and its partial sums S_n under given conditions
noncomputable def a_seq (n : ℕ) : ℝ := if n = 1 then 1 else 2 * S (n-1).to_nat + 1
noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum a_seq

-- (1) Proving a_2 = 3
theorem part1 (a_2 : ℝ) (h1 : a_seq 1 = 1) (h2 : ∀ n, a_seq (n + 1) = 2 * real.sqrt (S n) + 1) :
  a_2 = 3 :=
sorry

-- (2) Proving the general formula a_n = 2n - 1
theorem part2 (a_n : ℕ → ℝ) (h1 : a_seq 1 = 1) (h2 : ∀ n, a_seq (n + 1) = 2 * real.sqrt (S n) + 1) :
  ∀ n, a_n n = 2 * n - 1 :=
sorry

-- (3) Proving non-existence of positive integer k such that a_k, S_{2k-1}, a_{4k} form a geometric sequence
theorem part3 (h1 : a_seq 1 = 1) (h2 : ∀ n, a_seq (n + 1) = 2 * real.sqrt (S n) + 1) :
  ¬ ∃ k : ℕ, k > 0 ∧ ∃ (a : ℝ), a_seq k * a_seq (4 * k) = real.sqrt (S (2 * k - 1)) :=
sorry

end part1_part2_part3_l543_543856


namespace correct_conclusions_for_ellipse_l543_543825

-- Definitions for conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
    x^2 / a^2 + y^2 / b^2 = 1

def foci (a b : ℝ) : Set (ℝ × ℝ) :=
    let c := sqrt (a^2 - b^2)
    {(-c, 0), (c, 0)}

def point_on_ellipse (a b m n : ℝ) : Prop :=
    ellipse a b m n

def angles (P F1 F2 : (ℝ × ℝ)) (alpha beta : ℝ) : Prop :=
    let angle1 := real.atan2 (P.1 - F1.1) (P.2 - F1.2)
    let angle2 := real.atan2 (P.1 - F2.1) (P.2 - F2.2)
    abs (angle1 - angle2) = alpha ∧ 
    abs (angle2 - angle1) = beta

-- Lean definitions that encapsulate the (questions, conditions, correct answers) tuple.
theorem correct_conclusions_for_ellipse (a b m n alpha beta : ℝ)
    (P F1 F2 : (ℝ × ℝ))
    (hEllipse : ellipse a b m n)
    (hFoci : foci a b)
    (hPoint : point_on_ellipse a b m n)
    (hAngles : angles P F1 F2 alpha beta) :

  -- (2) The eccentricity of the ellipse
  let e := sqrt (1 - (b^2 / a^2))
  e = (sin (alpha + beta)) / (sin alpha + sin beta) ∧

  -- (4) There exists a fixed circle condition
  ∃ r1 r2 : ℝ, r1 = 2 * a ∧ r2 = PF ∧ 
    let circle1 := circle P r1
    let circle2 := circle F1 r2
    circle1.radius = circle2.radius ∧

  -- (5) Given inequality relation
  (1 / m^2) + (1 / n^2) ≥ (1 / a + 1 / b)^2 := sorry

end correct_conclusions_for_ellipse_l543_543825


namespace hyperbola_eccentricity_l543_543450

noncomputable theory
open Classical

theorem hyperbola_eccentricity
  (a b p : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hp : 0 < p)
  (h_hyperbola_focus : (p / 2) = Real.sqrt (a^2 + b^2))
  (h_cd_ab : (2 * b * Real.sqrt (a^2 + b^2)) / a = Real.sqrt 2 * (2 * b^2) / a) :
  ∃ e : ℝ, e = Real.sqrt 2 :=
by
  use Real.sqrt 2
  sorry

end hyperbola_eccentricity_l543_543450


namespace circumscribed_triangle_l543_543066

/-- Given an acute triangle ABC, with circles centered at A and C passing through point B, which intersect again at point F.
These circles intersect the circumcircle ω of triangle ABC at points D and E, respectively.
Segment BF intersects the circle ω at point O. Prove that O is the circumcenter of triangle DEF. -/
theorem circumscribed_triangle (A B C D E F O : Type*) 
  [∀ T, is_triangle A B C] [circle_center A B] [circle_center C B] 
  [circle_point (A, B) F] [circle_point (C, B) F]
  [circle_point (A B C) D] [circle_point (A B C) E] [segment_intersect (B F) O] :
  is_circumcenter (D E F) O :=
sorry

end circumscribed_triangle_l543_543066


namespace sum_first_9_terms_l543_543924

noncomputable def sum_of_first_n_terms (a1 d : Int) (n : Int) : Int :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_9_terms (a1 d : ℤ) 
  (h1 : a1 + (a1 + 3 * d) + (a1 + 6 * d) = 39)
  (h2 : (a1 + 2 * d) + (a1 + 5 * d) + (a1 + 8 * d) = 27) :
  sum_of_first_n_terms a1 d 9 = 99 := by
  sorry

end sum_first_9_terms_l543_543924


namespace cost_price_per_meter_l543_543701

theorem cost_price_per_meter (selling_price : ℝ) (total_meters : ℕ) (profit_per_meter : ℝ)
  (h1 : selling_price = 8925)
  (h2 : total_meters = 85)
  (h3 : profit_per_meter = 5) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 100 := by
  sorry

end cost_price_per_meter_l543_543701


namespace math_problem_l543_543691

def option_a : Prop := ¬ linear_equation (x^2 - 2*x - 3 = 0)
def option_b : Prop := linear_equation (x + 1 = 0)
def option_c : Prop := ¬ equation (3*x + 2)
def option_d : Prop := linear_equation_two_var (2*x + y = 5)

theorem math_problem :
  option_b ∧
  option_a ∧
  option_c ∧
  option_d ≠ option_b :=
by
  sorry

# Additional definitions used in the theorem statement
def linear_equation (e : Prop) : Prop := -- define condition for linear equation in one variable
  sorry

def linear_equation_two_var (e : Prop) : Prop := -- define condition for linear equation in two variables
  sorry

def equation (e : Prop) : Prop := -- define condition for an equation (general)
  sorry

end math_problem_l543_543691


namespace probability_multiple_of_3_or_4_l543_543624

-- Given the numbers 1 through 30 are written on 30 cards one number per card,
-- and Sara picks one of the 30 cards at random,
-- the probability that the number on her card is a multiple of 3 or 4 is 1/2.

-- Define the set of numbers from 1 to 30
def numbers := finset.range 30 \ {0}

-- Define what it means to be a multiple of 3 or 4 within the given range
def is_multiple_of_3_or_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 4 = 0

-- Define the set of multiples of 3 or 4 within the given range
def multiples_of_3_or_4 := numbers.filter is_multiple_of_3_or_4

-- The probability calculation
theorem probability_multiple_of_3_or_4 : 
  (multiples_of_3_or_4.card : ℚ) / numbers.card = 1 / 2 :=
begin
  -- The set multiples_of_3_or_4 contains 15 elements
  have h_multiples_card : multiples_of_3_or_4.card = 15, sorry,
  -- The set numbers contains 30 elements
  have h_numbers_card : numbers.card = 30, sorry,
  -- Therefore, the probability is 15/30 = 1/2
  rw [h_multiples_card, h_numbers_card],
  norm_num,
end

end probability_multiple_of_3_or_4_l543_543624


namespace pascal_row_20_fifth_sixth_sum_l543_543348

-- Conditions from the problem
def pascal_element (n k : ℕ) : ℕ := Nat.choose n k

-- Question translated to a Lean theorem
theorem pascal_row_20_fifth_sixth_sum :
  pascal_element 20 4 + pascal_element 20 5 = 20349 :=
by
  sorry

end pascal_row_20_fifth_sixth_sum_l543_543348


namespace magic_square_x_value_l543_543497

theorem magic_square_x_value 
  (a b c d e f g h : ℤ) 
  (h1 : x + b + c = d + e + c)
  (h2 : x + f + e = a + b + d)
  (h3 : x + e + c = a + g + 19)
  (h4 : b + f + e = a + g + 96) 
  (h5 : 19 = b)
  (h6 : 96 = c)
  (h7 : 1 = f)
  (h8 : a + d + x = b + c + f) : 
    x = 200 :=
by
  sorry

end magic_square_x_value_l543_543497


namespace find_curve_C1_eq_and_calc_PA_PB_l543_543927

open Real

noncomputable def curve_C1_eq : Prop := 
  ∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ↔ ∃ θ : ℝ, x = 2 * cos θ ∧ y = 2 * sin θ

noncomputable def line_l_eq : ∀ t : ℝ, Prop :=
  λ t, let x := t * cos (π / 3) in
       let y := sqrt 3 + t * sin (π / 3) in
       True

noncomputable def intersect_points_eq (A B : ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ, 
    let x1 := t1 * cos (π / 3) in
    let y1 := sqrt 3 + t1 * sin (π / 3) in
    let x2 := t2 * cos (π / 3) in
    let y2 := sqrt 3 + t2 * sin (π / 3) in
    A = (x1, y1) ∧ B = (x2, y2)

noncomputable def calc_P (A B : ℝ × ℝ) : ℝ :=
  let P : ℝ × ℝ := (0, sqrt 3) in
  let PA := dist P A in
  let PB := dist P B in
  1 / PA + 1 / PB

theorem find_curve_C1_eq_and_calc_PA_PB :
  (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ↔ ∃ θ : ℝ, x = 2 * cos θ ∧ y = 2 * sin θ) →
  (∀ t : ℝ, ∃ A B : ℝ × ℝ, intersect_points_eq A B t) →
  ∀ A B : ℝ × ℝ, intersect_points_eq (0, sqrt 3) calc_P (0, sqrt 3) A B = 3 / 2 :=
sorry

end find_curve_C1_eq_and_calc_PA_PB_l543_543927


namespace length_PF_l543_543189

theorem length_PF (P F : ℝ × ℝ)
  (parabola_eq : ∀ (x y : ℝ), y^2 = 8 * (x + 2))
  (focus_eq : F = (0, 0))
  (line_through_focus : ∀ (x : ℝ), x ≠ F.1 → P.2 = 0 ∧ y = sqrt(3) * x)
  (intersects : ∃ (A B : ℝ × ℝ), ∀ (x y : ℝ), (y ≠ 0) → (line_through_focus x y ∧ parabola_eq x y)) :
  dist P F = 16 / 3 := sorry

end length_PF_l543_543189


namespace domain_f_domain_f_equiv_l543_543222

noncomputable def f (x : ℝ) := Real.log (3 * x - 1)

theorem domain_f : 
  {x : ℝ | 3 * x - 1 > 0} = {x : ℝ | x > 1 / 3} :=
by sorry

theorem domain_f_equiv : 
  {x : ℝ | ∃ y, f y = f x } = Ioi (1 / 3 : ℝ) :=
by sorry

end domain_f_domain_f_equiv_l543_543222


namespace bobbie_letters_to_remove_l543_543983

-- Definitions of the conditions
def samanthaLastNameLength := 7
def bobbieLastNameLength := samanthaLastNameLength + 3
def jamieLastNameLength := 4
def targetBobbieLastNameLength := 2 * jamieLastNameLength

-- Question: How many letters does Bobbie need to take off to have a last name twice the length of Jamie's?
theorem bobbie_letters_to_remove : 
  bobbieLastNameLength - targetBobbieLastNameLength = 2 := by 
  sorry

end bobbie_letters_to_remove_l543_543983


namespace periodic_function_of_2011_l543_543170

noncomputable def f : ℕ → (ℝ → ℝ)
| 0     := λ x, Real.cos x
| (n+1) := λ x, f n x.deriv

theorem periodic_function_of_2011 :
  ∀ x, f 2011 x = Real.sin x := by
sorry

end periodic_function_of_2011_l543_543170


namespace cos_of_double_angles_l543_543407

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l543_543407


namespace number_of_positive_solutions_l543_543234

theorem number_of_positive_solutions (x y z : ℕ) (h_cond : x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 12) :
    ∃ (n : ℕ), n = 55 :=
by 
  sorry

end number_of_positive_solutions_l543_543234


namespace greatest_power_of_3_dividing_30_factorial_l543_543269

theorem greatest_power_of_3_dividing_30_factorial : 
  let w := (Nat.factorial 30)
  in ∃ k : ℕ, (3^k ∣ w) ∧ ∀ m : ℕ, (3^m ∣ w) → m ≤ 14 := 
by
  let w := (Nat.factorial 30)
  use 14
  split
  . sorry -- Proof that 3^14 divides w
  . intros m hDiv
    sorry -- Proof that m <= 14 if 3^m divides w

end greatest_power_of_3_dividing_30_factorial_l543_543269


namespace factors_of_48_multiples_of_8_l543_543124

theorem factors_of_48_multiples_of_8 : 
  ∃ count : ℕ, count = 4 ∧ (∀ d ∈ {d | d ∣ 48 ∧ (∃ k, d = 8 * k)}, true) :=
by {
  sorry  -- This is a placeholder for the actual proof
}

end factors_of_48_multiples_of_8_l543_543124


namespace convert_10212_base3_to_decimal_l543_543785

def convert_to_decimal (n : ℕ) (b : ℕ) : ℕ :=
  List.foldl (λ acc (d : ℕ), acc * b + d) 0 (n.digits b)

theorem convert_10212_base3_to_decimal :
  convert_to_decimal 10212 3 = 104 :=
by
  sorry

end convert_10212_base3_to_decimal_l543_543785


namespace six_over_seven_letters_probability_zero_l543_543646

theorem six_over_seven_letters_probability_zero :
  ∀ (letters : Fin 7 → Fin 7), (∑ i : Fin 7, if letters i = i then 1 else 0 = 6) → false :=
by
  sorry

end six_over_seven_letters_probability_zero_l543_543646


namespace max_perimeter_of_eight_pieces_l543_543554

noncomputable
def triangle_base_length : ℝ := 10

noncomputable
def triangle_height : ℝ := 12

noncomputable
def segment_length : ℝ := triangle_base_length / 8

noncomputable
def perimeter (k : ℕ) : ℝ :=
  if 0 ≤ k ∧ k < 8 then
    segment_length
    + Real.sqrt (triangle_height^2 + k^2 * segment_length^2)
    + Real.sqrt (triangle_height^2 + (k+1)^2 * segment_length^2)
  else 0

noncomputable
def max_perimeter : ℝ := Real.max (List.map perimeter [0, 1, 2, 3, 4, 5, 6, 7])

-- The theorem statement
theorem max_perimeter_of_eight_pieces : max_perimeter = 30.53 :=
by sorry

end max_perimeter_of_eight_pieces_l543_543554


namespace inequality_not_hold_l543_543475

variable {a b : ℝ}
variable (h : a < b < 0)

theorem inequality_not_hold (h : a < b < 0) : ¬ (1 / (a - b) > 1 / a) :=
sorry

end inequality_not_hold_l543_543475


namespace find_max_theta_l543_543054

noncomputable def max_theta (θ : ℝ) : ℝ :=
  if θ < Real.pi
    ∧ ∏ k in Finset.range 11, Real.cos (2^k * θ) ≠ 0
    ∧ ∏ k in Finset.range 11, (1 + 1 / Real.cos (2^k * θ)) = 1
  then θ else 0

theorem find_max_theta : max_theta (2046 * Real.pi / 2047) = 2046 * Real.pi / 2047 :=
by
  sorry

end find_max_theta_l543_543054


namespace even_number_rolls_probability_l543_543726

theorem even_number_rolls_probability :
  let prob_even := 1 / 2
  in let prob_at_least_six := (Nat.choose 8 6 * prob_even^6 * (1 - prob_even)^2) +
                              (Nat.choose 8 7 * prob_even^7 * (1 - prob_even)) +
                              prob_even^8
  in prob_at_least_six = 121 / 256 :=
by
  sorry

end even_number_rolls_probability_l543_543726


namespace find_150th_letter_in_pattern_l543_543664

theorem find_150th_letter_in_pattern : 
  (let sequence := "ABCD";
   sequence.length = 4 → 
   sequence[(150 % 4)] = 'B') :=
by
  sorry

end find_150th_letter_in_pattern_l543_543664


namespace probability_is_half_l543_543615

-- Define the set of numbers from 1 to 30
def numbers : Finset ℕ := (Finset.range 30).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define the set of multiples of 3 from 1 to 30
def multiples_of_3 : Finset ℕ := numbers.filter (λ n, n % 3 = 0)

-- Define the set of multiples of 4 from 1 to 30
def multiples_of_4 : Finset ℕ := numbers.filter (λ n, n % 4 = 0)

-- Define the set of multiples of 12 from 1 to 30 (multiples of both 3 and 4)
def multiples_of_12 : Finset ℕ := numbers.filter (λ n, n % 12 = 0)

-- Calculate the probability using the principle of inclusion-exclusion
def favorable_outcomes : ℕ := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card

-- Total number of outcomes
def total_outcomes : ℕ := numbers.card

-- Calculate the probability
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 1/2
theorem probability_is_half : probability = 1 / 2 := by
  sorry

end probability_is_half_l543_543615


namespace total_cantaloupes_l543_543831

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by sorry

end total_cantaloupes_l543_543831


namespace cosine_identity_l543_543392

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l543_543392


namespace smallest_initial_number_sum_of_digits_l543_543762

theorem smallest_initial_number_sum_of_digits : ∃ (N : ℕ), 
  (0 ≤ N ∧ N < 1000) ∧ 
  ∃ (k : ℕ), 16 * N + 700 + 50 * k < 1000 ∧ 
  (N = 16) ∧ 
  (Nat.digits 10 N).sum = 7 := 
by
  sorry

end smallest_initial_number_sum_of_digits_l543_543762


namespace terminal_side_of_angle_l543_543114

theorem terminal_side_of_angle (α : ℝ) (h1 : Real.Tan α < 0) (h2 : Real.Cos α < 0) : 
  ∃ β : ℝ, β = α ∧ Real.Tan β < 0 ∧ Real.Cos β < 0 ∧ terminal_quadrant β = quadrant.II :=
sorry

end terminal_side_of_angle_l543_543114


namespace number_of_girls_in_school_l543_543498

theorem number_of_girls_in_school (total_students : ℕ) (sample_size : ℕ) (girl_boy_diff : ℕ) 
  (stratified : sample_size = 250) (total_students_condition : total_students = 1750) 
  (girl_boy_diff_condition : girl_boy_diff = 20) :
  ∃ (total_girls : ℕ), total_girls = 805 :=
by 
  use 805
  sorry

end number_of_girls_in_school_l543_543498


namespace intersection_A_B_l543_543438

-- Definitions based on the conditions
def A : Set ℝ := {x | x ≥ 0}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

-- Proof statement
theorem intersection_A_B :
  (A ∩ (B : Set ℝ)) = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_A_B_l543_543438


namespace find_symmetric_circle_l543_543452

section
variable {R : Type} [Real R]

-- Define the first circle C1
def C1 (x y : R) : Prop := (x + 1) ^ 2 + (y - 1) ^ 2 = 1

-- Define the line of symmetry
def line_symmetry (x y : R) : Prop := x - y - 1 = 0

-- Define the second circle C2
def C2 (x y : R) : Prop := (x - 2) ^ 2 + (y + 2) ^ 2 = 1

-- Define the symmetric point transformation
def symmetric_point (x y : R) : R × R := (y + 1, x - 1)

-- The main theorem stating the equation of the symmetric circle
theorem find_symmetric_circle (x y : R) (hx : C1 (y + 1) (x - 1)) : C2 x y :=
sorry
end

end find_symmetric_circle_l543_543452


namespace jessica_final_balance_l543_543263

theorem jessica_final_balance (B : ℝ) (withdrawn : ℝ := 400) (decrease_fraction : ℝ := 2/5) (deposit_fraction : ℝ := 1/4) :
  let remaining_balance := B - withdrawn
  in let new_balance := remaining_balance + deposit_fraction * remaining_balance
  in (remaining_balance = B * (1 - decrease_fraction)) →
     new_balance = 750 :=
  by
    intros H
    sorry

end jessica_final_balance_l543_543263


namespace clown_mobiles_count_l543_543600

theorem clown_mobiles_count (C T : ℕ) (hC : C = 28) (hT : T = 140) : ∃ n : ℕ, n = T / C ∧ n = 5 :=
by
  use T / C
  split
  · exact Nat.div_eq_of_eq_mul (by norm_num; exact Eq.symm (by norm_num [hC, hT]))
  · exact Nat.div_eq_of_eq_mul (by norm_num; exact Eq.symm (by norm_num [hC, hT]))

end clown_mobiles_count_l543_543600


namespace hyperbola_eccentricity_proof_l543_543590

noncomputable def hyperbola_eccentricity_asymptote_tangent_circle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∀ (x y : ℝ),
    ((x^2 / a^2) - (y^2 / b^2) = 1) → -- Hyperbola equation
    ((x - 2)^2 + y^2 = 3) →            -- Circle equation
    (abs ((2 * b) / (real.sqrt (a^2 + b^2)) - real.sqrt 3) = 0) →    -- Tangency condition
    (real.eccentricity a b = 2)        -- Eccentricity is 2

theorem hyperbola_eccentricity_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (tangent_condition : ∀ (x y : ℝ),
    ((x^2 / a^2) - (y^2 / b^2) = 1) →  -- Hyperbola equation
    ((x - 2)^2 + y^2 = 3) →            -- Circle equation
    (abs ((2 * b) / (real.sqrt (a^2 + b^2)) - real.sqrt 3) = 0)) : 
  hyperbola_eccentricity_asymptote_tangent_circle a b ha hb := 
begin
  sorry
end

end hyperbola_eccentricity_proof_l543_543590


namespace total_books_received_l543_543187

theorem total_books_received (books1 books2 : ℕ) (h1 : books1 = 54) (h2 : books2 = 23) : books1 + books2 = 77 := by
  rw [h1, h2]
  exact rfl

end total_books_received_l543_543187


namespace total_lemons_produced_l543_543734

def normal_tree_lemons_per_year := 60
def production_factor := 1.5
def grove_rows := 50
def grove_cols := 30
def years := 5

theorem total_lemons_produced :
  let engineered_tree_lemons_per_year := normal_tree_lemons_per_year * production_factor in
  let total_trees := grove_rows * grove_cols in
  let lemons_per_year := total_trees * engineered_tree_lemons_per_year in
  let total_lemons := lemons_per_year * years in
  total_lemons = 675000 :=
by
  sorry

end total_lemons_produced_l543_543734


namespace Jenna_total_profit_l543_543517

noncomputable theory
open_locale classical

def SupplierA_cost : ℝ := 3000 * 3.50
def SupplierB_cost : ℝ := 2000 * 4.00
def total_widget_cost : ℝ := SupplierA_cost + SupplierB_cost
def total_shipping_fee : ℝ := (3000 + 2000) * 0.25
def rent : ℝ := 10000
def salaries : ℝ := 4 * 2500
def total_expenses_excluding_widgets : ℝ := rent + salaries
def total_expenses : ℝ := total_widget_cost + total_shipping_fee + total_expenses_excluding_widgets
def total_revenue : ℝ := (3000 + 2000) * 8
def profit_before_taxes : ℝ := total_revenue - total_expenses
def tax_rate : ℝ := 0.25
def taxes : ℝ := tax_rate * profit_before_taxes
def total_profit_after_taxes : ℝ := profit_before_taxes - taxes

theorem Jenna_total_profit : total_profit_after_taxes = 187.50 := 
by
  sorry

end Jenna_total_profit_l543_543517


namespace cylinder_volume_from_square_l543_543274

theorem cylinder_volume_from_square 
  (side_length : ℝ) 
  (h1 : side_length = 1) : 
  let r := side_length / (2 * Real.pi),
      h := side_length in 
  π * r^2 * h = 1 / (4 * Real.pi) := by 
    sorry

end cylinder_volume_from_square_l543_543274


namespace exists_special_triangle_l543_543036

theorem exists_special_triangle (paint : ℝ × ℝ → fin 3) : 
  ∃ (A B C : ℝ × ℝ), 
    (paint A = paint B ∧ paint B = paint C) ∧ 
    (∃ (circumradius : ℝ), circumradius = 2008) ∧ 
    (∃ (α β γ : ℝ), α + β + γ = π ∧ 
                     (α = 2 * β ∨ α = 3 * β ∨ β = 2 * γ ∨ β = 3 * γ ∨ γ = 2 * α ∨ γ = 3 * α)) :=
begin
  -- The proof will start here
  sorry
end

end exists_special_triangle_l543_543036


namespace find_150th_letter_l543_543668

def repeating_sequence : String := "ABCD"

def position := 150

theorem find_150th_letter :
  repeating_sequence[(position % 4) - 1] = 'B' := 
sorry

end find_150th_letter_l543_543668


namespace compute_complex_power_l543_543020

theorem compute_complex_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 :=
by
  sorry

end compute_complex_power_l543_543020


namespace johns_weekly_earnings_increase_l543_543518

noncomputable def percentageIncrease (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem johns_weekly_earnings_increase :
  percentageIncrease 30 40 = 33.33 :=
by
  sorry

end johns_weekly_earnings_increase_l543_543518


namespace fractions_product_simplified_l543_543775

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end fractions_product_simplified_l543_543775


namespace circumcenter_on_line_AM_l543_543565

open EuclideanGeometry

-- Define points and their positions
variables {P : Type*} [MetricSpace P] [InnerProductSpace ℝ P] (A M B C : P)
variables (hAinside : ∃ X Y : P, ∃ α β : ℝ, 0 < α ∧ 0 < β ∧ 
                        Angle A X M = α ∧ Angle A Y M = β)

-- Define the ray reflection conditions
variables (h_refl : ∃ R S : P, is_ray A R ∧ is_ray R S ∧ is_ray S A ∧
                    reflects R B ∧ reflects S C ∧ Angle A R B = Angle B C S)

-- Define the reflection and angle equality
variables (h_reflect : ∀ X Y Z : P, Angle_of_reflection X Y = Angle_of_reflection Y Z)

-- Define the circumcenter of triangle
axiom center_of_circumcircle_tri (x y z : P) : ∃ c : P, c = circumcenter x y z

-- Mathematical statement to prove equivalence
theorem circumcenter_on_line_AM : ∃ O : P, O = circumcenter B C M ∧ lies_on_line O A M :=
by {
  sorry
}

end circumcenter_on_line_AM_l543_543565


namespace x_values_l543_543356

noncomputable def find_x_values : ℝ → Prop :=
  λ x, x > 4 ∧ x > (5 / 2) ∧ x - (5 / 2) > 1 ∧ x - 4 > 1 ∧ (
    ∃ c a b, (c = log x (x - (5 / 2)) ∨ c = log (x - (5 / 2)) (x - 4) ∨ c = log (x - 4) x) ∧ 
    ((c = a * b ∧ a * b * c = 1) ∧ (c = 1 ∨ c = -1)) ∧ (x = (9 / 2) ∨ x = 2 + sqrt 5)
  )

theorem x_values : ∃ x, find_x_values x :=
by
  sorry

end x_values_l543_543356


namespace icosahedron_faces_meet_l543_543894

/-- An icosahedron is a polyhedron with 20 faces, each of which is an equilateral triangle. -/
structure Icosahedron where
  faces : ℕ
  face_shape : faces = 20
  face_type : String
  (h1 : face_type = "equilateral triangle")

/-- At each vertex of an icosahedron, five faces meet. -/
theorem icosahedron_faces_meet (I : Icosahedron) : ∀ v : ℕ, v = 12 → 5 :=
by
  sorry

end icosahedron_faces_meet_l543_543894


namespace sum_of_cubes_mod_11_l543_543683

theorem sum_of_cubes_mod_11 : 
  (∑ i in Finset.range 10, (i + 1)^3) % 11 = 0 := 
by
  sorry

end sum_of_cubes_mod_11_l543_543683


namespace good_sequences_product_good_l543_543601

-- Define what it means for a sequence to be good
def is_good_sequence (a : ℕ → ℝ) : Prop :=
  ∀ k, ∀ n, (derivative^k (λ i, a i)) n > 0

-- Definition of the derivative of a sequence
def derivative (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a (n + 1) - a n

-- k-th derivative of a sequence
def k_derivative (k : ℕ) (a : ℕ → ℝ) : ℕ → ℝ :=
  nat.iterate (derivative) k a

-- Product sequence
def product_sequence (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n * b n

-- Theorem stating if a and b are good sequences, then their product is a good sequence
theorem good_sequences_product_good
  (a b : ℕ → ℝ) (h_a : is_good_sequence a) (h_b : is_good_sequence b) :
  is_good_sequence (product_sequence a b) :=
sorry

end good_sequences_product_good_l543_543601


namespace solution_interval_l543_543063

theorem solution_interval {f : ℝ → ℝ} 
    (h_monotonic : ∀ a b, 0 < a → 0 < b → a < b → f(a) ≤ f(b))
    (h_infinite_domain : ∀ x, 0 < x → f(f(x) - Real.log x / Real.log 2) = 3) :
    let x := f(x)
    let f(x) := Real.log x / Real.log 2 + 2
in (1 < x ∧ x < 2) :=
begin
  sorry
end

end solution_interval_l543_543063


namespace solve_equation_l543_543809

noncomputable def roots_of_equation := {x : Real | 
  1 / (x^2 + 8 * x - 12) + 1 / (x^2 + 5 * x - 12) + 1 / (x^2 - 10 * x - 12) = 0 }

theorem solve_equation :
  roots_of_equation = {sqrt 12, -sqrt 12, 4, 3} :=
sorry

end solve_equation_l543_543809


namespace project_completion_l543_543513

def people := ℕ
def days := ℕ

variables (team_1 team_2 : people) (days_half : days) (total_days : days)

theorem project_completion
  (h1 : team_1 = 12)
  (h2 : days_half = 15)
  (h3 : team_2 = team_1 + 8)
  (h4 : total_days = 33) :
  (2 * team_1 * days_half = total_days * team_2) :=
by sorry

end project_completion_l543_543513


namespace probability_leftmost_blue_off_rightmost_red_on_l543_543199

noncomputable def probability_specific_arrangement : ℚ :=
  let total_ways_arrange_colors := combinatorial.choose 6 3 in
  let total_ways_choose_on := combinatorial.choose 6 3 in
  let ways_arrange_colors_given_restrictions := combinatorial.choose 4 2 in
  let ways_choose_on_given_restrictions := combinatorial.choose 4 2 in
  (ways_arrange_colors_given_restrictions * ways_choose_on_given_restrictions : ℚ) / (total_ways_arrange_colors * total_ways_choose_on : ℚ)

theorem probability_leftmost_blue_off_rightmost_red_on :
  probability_specific_arrangement = 9 / 100 :=
begin
  -- The proof will be placed here
  sorry
end

end probability_leftmost_blue_off_rightmost_red_on_l543_543199


namespace side_length_square_l543_543993

theorem side_length_square (s : ℝ) (h1 : ∃ (s : ℝ), (s > 0)) (h2 : 6 * s^2 = 3456) : s = 24 :=
sorry

end side_length_square_l543_543993


namespace Carol_extra_invitations_l543_543778

theorem Carol_extra_invitations (packs_initial invitations_per_pack friends_invited : ℕ)
  (h1 : invitations_per_pack = 5)
  (h2 : packs_initial = 3)
  (h3 : friends_invited = 23) :
  ∃ (extra_invitations : ℕ), 
  let total_initial_invitations := packs_initial * invitations_per_pack,
      additional_invitations_needed := friends_invited - total_initial_invitations,
      packs_to_buy := (additional_invitations_needed + invitations_per_pack - 1) / invitations_per_pack,
      total_invitations := total_initial_invitations + packs_to_buy * invitations_per_pack,
      extra_invitations := total_invitations - friends_invited in
  extra_invitations = 2 := 
by
{
  sorry
}

end Carol_extra_invitations_l543_543778


namespace find_a_prove_f_pos_l543_543454

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.log x + (1 / 2) * x

theorem find_a (a x0 : ℝ) (hx0 : x0 > 0) (h_tangent : (x0 - a) * Real.log x0 + (1 / 2) * x0 = (1 / 2) * x0 ∧ Real.log x0 - a / x0 + 3 / 2 = 1 / 2) :
  a = 1 :=
sorry

theorem prove_f_pos (a : ℝ) (h_range : 1 / (2 * Real.exp 1) < a ∧ a < 2 * Real.sqrt (Real.exp 1)) (x : ℝ) (hx : x > 0) :
  f x a > 0 :=
sorry

end find_a_prove_f_pos_l543_543454


namespace fractions_product_simplified_l543_543776

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end fractions_product_simplified_l543_543776


namespace symmetric_function_value_l543_543133

variables {ℝ : Type*}

-- Defining the function f and the interval of interest
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x * (1 - x) else x * (x + 1)

theorem symmetric_function_value (x : ℝ) (hx : x ∈ set.Iic (0 : ℝ)) :
  f x = x * (x + 1) :=
sorry

end symmetric_function_value_l543_543133


namespace find_b_from_lengths_l543_543797

theorem find_b_from_lengths
  (EA EB EC ED b : ℝ)
  (hEA : EA = 2)
  (hEB : EB = real.sqrt 11)
  (hEC : EC = 4)
  (hED : ED = b) :
  b = 3 :=
sorry

end find_b_from_lengths_l543_543797


namespace combined_income_is_16800_l543_543801

-- Given conditions
def ErnieOldIncome : ℕ := 6000
def ErnieCurrentIncome : ℕ := (4 * ErnieOldIncome) / 5
def JackCurrentIncome : ℕ := 2 * ErnieOldIncome

-- Proof that their combined income is $16800
theorem combined_income_is_16800 : ErnieCurrentIncome + JackCurrentIncome = 16800 := by
  sorry

end combined_income_is_16800_l543_543801


namespace eval_98_times_98_l543_543804

theorem eval_98_times_98 : (98 * 98 = 9604) := by
  have h1 : 98 = 100 - 2 := by ring
  have h2 : 98 * 98 = (100 - 2) * (100 - 2) := by rw h1
  have h3 : (100 - 2) * (100 - 2) = 100^2 - 2*100*2 + 2^2 := by ring
  have h4 : 100^2 - 2*100*2 + 2^2 = 10000 - 400 + 4 := by norm_num
  have h5 : 10000 - 400 + 4 = 9604 := by norm_num
  rw [←h3, ←h4, ←h2],
  exact h5

end eval_98_times_98_l543_543804


namespace count_irreducible_fractions_l543_543122

theorem count_irreducible_fractions :
  let m := 2015 in
  let a_min := m * m in
  let a_max := m * m + m in
  (∑ n in (filter (λ n, n.gcd m = 1) (Ico (1 : ℕ) m)), 1) = 1440 := 
by
  simp only [Nat.gcd_rec],
  sorry

end count_irreducible_fractions_l543_543122


namespace th150th_letter_is_B_l543_543656

def pattern := "ABCD".data

def nth_letter_in_pattern (n : ℕ) : Char :=
  let len := pattern.length
  pattern.get n % len

theorem th150th_letter_is_B :
  nth_letter_in_pattern 150 = 'B' :=
by {
  -- This proof is placed here as a placeholder
  sorry
}

end th150th_letter_is_B_l543_543656


namespace martha_age_future_comparison_l543_543963

theorem martha_age_future_comparison :
  ∃ x : ℕ, 32 = 2 * (10 + x) ∧ x = 6 :=
by
    use 6
    split
    . 
      linarith
    . 
      rfl

end martha_age_future_comparison_l543_543963


namespace diameter_perpendicular_to_chord_bisects_l543_543570

-- Definitions based on the conditions
variables (O A B M : Type)
variables [metric_space O] -- using metric_space for geometrical constraints
variables [is_circenter O A] [is_circenter O B] -- OA and OB being radii implies circle with centre O
variables (OM_perpendicular_AB : is_perpendicular (distance O M) (distance A B)) -- OM ⊥ AB

-- Statement to prove that M is the midpoint of AB
theorem diameter_perpendicular_to_chord_bisects :
  is_midpoint M A B := sorry

end diameter_perpendicular_to_chord_bisects_l543_543570


namespace minor_arc_KB_measure_l543_543142

theorem minor_arc_KB_measure
  (Q : Type*)
  [∀ x : Q, x ∈ Q → Prop]
  (K T B : Q)
  (hK : K ∈ Q)
  (hT : T ∈ Q)
  (hB : B ∈ Q)
  (hKAT : ∃ A : Q, angle A K T = 72)
  (hKBT : angle K B T = 40) :
  arc_measure Q K B = 80 := 
sorry

end minor_arc_KB_measure_l543_543142


namespace max_mn_on_parabola_l543_543633

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end max_mn_on_parabola_l543_543633


namespace train_length_is_150_l543_543702

noncomputable def train_length_crossing_post (t_post : ℕ := 10) : ℕ := 10
noncomputable def train_length_crossing_platform (length_platform : ℕ := 150) (t_platform : ℕ := 20) : ℕ := 20
def train_constant_speed (L v : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) : Prop :=
  v = L / t_post ∧ v = (L + length_platform) / t_platform

theorem train_length_is_150 (L : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) (H : train_constant_speed L v t_post t_platform length_platform) : 
  L = 150 :=
by
  sorry

end train_length_is_150_l543_543702


namespace total_lemons_produced_l543_543735

def normal_tree_lemons_per_year := 60
def production_factor := 1.5
def grove_rows := 50
def grove_cols := 30
def years := 5

theorem total_lemons_produced :
  let engineered_tree_lemons_per_year := normal_tree_lemons_per_year * production_factor in
  let total_trees := grove_rows * grove_cols in
  let lemons_per_year := total_trees * engineered_tree_lemons_per_year in
  let total_lemons := lemons_per_year * years in
  total_lemons = 675000 :=
by
  sorry

end total_lemons_produced_l543_543735


namespace lisa_total_distance_l543_543958

-- Definitions for distances and counts of trips
def plane_distance : ℝ := 256.0
def train_distance : ℝ := 120.5
def bus_distance : ℝ := 35.2

def plane_trips : ℕ := 32
def train_trips : ℕ := 16
def bus_trips : ℕ := 42

-- Definition of total distance traveled
def total_distance_traveled : ℝ :=
  (plane_distance * plane_trips)
  + (train_distance * train_trips)
  + (bus_distance * bus_trips)

-- The statement to be proven
theorem lisa_total_distance :
  total_distance_traveled = 11598.4 := by
  sorry

end lisa_total_distance_l543_543958


namespace sum_of_squares_of_pairwise_distances_leq_nsqR2_l543_543509

-- Definitions for the problem
variables (n : ℕ) (R : ℝ) (points : fin n → EuclideanSpace ℝ (fin 2))

-- Condition: All points are within a circle of radius R
def points_within_circle : Prop := ∀ (i : fin n), dist (points i) (0 : EuclideanSpace ℝ (fin 2)) ≤ R

-- Theorem: The sum of the squares of the pairwise distances between points does not exceed n^2 * R^2
theorem sum_of_squares_of_pairwise_distances_leq_nsqR2 (h : points_within_circle n R points) :
  (∑ i j in finset.range n, if h : j > i then (dist (points i) (points j)) ^ 2 else 0) ≤ n^2 * R^2 := 
sorry

end sum_of_squares_of_pairwise_distances_leq_nsqR2_l543_543509


namespace each_candle_burns_exactly_4_hours_l543_543731

-- Definition of the sets of candles being lit each day
def Candles : Type := Fin 7 

-- The sequence of candles lit each evening
def evening_candles (n : Fin 7) : Set Candles := 
  match n with
  |  ⟨0, _⟩ => {0}
  |  ⟨1, _⟩ => {1, 2}
  |  ⟨2, _⟩ => {3, 4, 5}
  |  ⟨3, _⟩ => {6, 0, 1, 2}
  |  ⟨4, _⟩ => {3, 4, 5, 6, 0}
  |  ⟨5, _⟩ => {1, 2, 3, 4, 5, 6}
  |  ⟨6, _⟩ => {0, 1, 2, 3, 4, 5, 6}

-- Prove that each candle burns for exactly 4 hours
theorem each_candle_burns_exactly_4_hours :
  ∀ (c : Candles), ∑ n in Finset.range 7, if c ∈ evening_candles n then 1 else 0 = 4 := 
by 
  sorry

end each_candle_burns_exactly_4_hours_l543_543731


namespace find_n_minus_m_l543_543125

theorem find_n_minus_m (n m : ℕ) (x y : ℕ) (h1 : 2 * 8^n * 16^n = 2^15)
    (h2 : (m * x + y) * (2 * x - y) does not contain xy term) :
    n - m = 0 := 
sorry

end find_n_minus_m_l543_543125


namespace salon_extra_cans_l543_543743

theorem salon_extra_cans
  (customers_per_day : ℕ)
  (cans_per_customer : ℕ)
  (cans_bought_per_day : ℕ)
  (H1 : customers_per_day = 14)
  (H2 : cans_per_customer = 2)
  (H3 : cans_bought_per_day = 33) :
  let cans_used_per_day := customers_per_day * cans_per_customer in
  let extra_cans := cans_bought_per_day - cans_used_per_day in
  extra_cans = 5 :=
by
  sorry

end salon_extra_cans_l543_543743


namespace find_k_for_infinite_solutions_l543_543367

noncomputable def has_infinitely_many_solutions (k : ℝ) : Prop :=
  ∀ x : ℝ, 5 * (3 * x - k) = 3 * (5 * x + 15)

theorem find_k_for_infinite_solutions :
  has_infinitely_many_solutions (-9) :=
by
  sorry

end find_k_for_infinite_solutions_l543_543367


namespace necessary_but_not_sufficient_condition_l543_543954

-- Define the mathematical structures and propositions
variables {α β  : Type*} -- types for planes
variables {a b : Type*} -- types for lines

-- Define the conditions and hypothesis
variable [plane α] 
variable [plane β]
variable [line a] 
variable [line b]

-- Assume lines a and b lie in plane α
variable (a_sub_α : a ⊆ α)
variable (b_sub_α : b ⊆ α)

-- Assume lines a and b are parallel to plane β
variable (a_parallel_β : a ∥ β)
variable (b_parallel_β : b ∥ β)

-- Define the theorem statement
theorem necessary_but_not_sufficient_condition 
  (h : (a ∥ β) ∧ (b ∥ β)) :
  (α ∥ β) ↔ (a ∥ β) ∧ (b ∥ β):= 
sorry

end necessary_but_not_sufficient_condition_l543_543954


namespace max_m_plus_n_l543_543639

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end max_m_plus_n_l543_543639


namespace probability_correct_match_l543_543290

/-- 
A contest organizer uses a game where six historical figures are paired with quotes incorrectly list next to their portraits. 
Participants should guess which quote belongs to which historical figure. What is the probability that a participant guessing 
at random will match all six correctly?
-/
theorem probability_correct_match : 
  (1 / Nat.factorial 6 : ℚ) = 1 / 720 :=
by
  simp [Nat.factorial]
  sorry

end probability_correct_match_l543_543290


namespace calculate_yield_l543_543918

-- Define the conditions
def x := 6
def x_pos := 3
def x_tot := 3 * x
def nuts_x_pos := x + x_pos
def nuts_x := x
def nuts_x_neg := x - x_pos
def yield_x_pos := 60
def yield_x := 120
def avg_yield := 100

-- Calculate yields
def nuts_x_pos_yield : ℕ := nuts_x_pos * yield_x_pos
def nuts_x_yield : ℕ := nuts_x * yield_x
noncomputable def total_yield (yield_x_neg : ℕ) : ℕ :=
  nuts_x_pos_yield + nuts_x_yield + nuts_x_neg * yield_x_neg

-- Equation combining all
lemma yield_per_tree : (total_yield Y) / x_tot = avg_yield := sorry

-- Prove Y = 180
theorem calculate_yield : (x = 6 → ((nuts_x_neg * 180 = 540) ∧ rate = 180)) := sorry

end calculate_yield_l543_543918


namespace steps_taken_l543_543323

noncomputable def andrewSpeed : ℝ := 1 -- Let Andrew's speed be represented by 1 feet per minute
noncomputable def benSpeed : ℝ := 3 * andrewSpeed -- Ben's speed is 3 times Andrew's speed
noncomputable def totalDistance : ℝ := 21120 -- Distance between the houses in feet
noncomputable def andrewStep : ℝ := 3 -- Each step of Andrew covers 3 feet

theorem steps_taken : (totalDistance / (andrewSpeed + benSpeed)) * andrewSpeed / andrewStep = 1760 := by
  sorry -- proof to be filled in later

end steps_taken_l543_543323


namespace smallest_whole_number_larger_than_sum_l543_543822

-- Definitions based on conditions
def mixed_numbers := [4 + 1/2, 6 + 1/3, 8 + 1/4, 10 + 1/5]

-- Main theorem statement
theorem smallest_whole_number_larger_than_sum : 
  let sum := mixed_numbers.sum in
  let smallest_whole_number := sum.ceil in
  smallest_whole_number = 30 :=
by
  sorry

end smallest_whole_number_larger_than_sum_l543_543822


namespace john_has_leftover_bulbs_l543_543934

-- Definitions of the problem statements
def initial_bulbs : ℕ := 40
def used_bulbs : ℕ := 16
def remaining_bulbs_after_use : ℕ := initial_bulbs - used_bulbs
def given_to_friend : ℕ := remaining_bulbs_after_use / 2

-- Statement to prove
theorem john_has_leftover_bulbs :
  remaining_bulbs_after_use - given_to_friend = 12 :=
by
  sorry

end john_has_leftover_bulbs_l543_543934


namespace probability_of_rolling_divisor_of_12_l543_543725

def is_divisor (a b : ℕ) : Prop := b % a = 0

def outcomes_on_die := {1, 2, 3, 4, 5, 6, 7, 8}

def divisors_of_12 := {d ∈ outcomes_on_die | is_divisor d 12}

def probability_of_divisor :=
  (divisors_of_12.to_finset.card / outcomes_on_die.to_finset.card : ℚ)

theorem probability_of_rolling_divisor_of_12 :
  probability_of_divisor = 5 / 8 :=
by 
  sorry

end probability_of_rolling_divisor_of_12_l543_543725


namespace sin_beta_value_l543_543059

theorem sin_beta_value 
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : cos (α - β) = -5 / 13)
  (h4 : sin α = 4 / 5) :
  sin β = -56 / 65 := 
sorry

end sin_beta_value_l543_543059


namespace max_m_n_value_l543_543636

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end max_m_n_value_l543_543636


namespace no_negative_roots_l543_543975

theorem no_negative_roots (x : ℝ) : 4 * x^4 - 7 * x^3 - 20 * x^2 - 13 * x + 25 ≠ 0 ∨ x ≥ 0 := 
sorry

end no_negative_roots_l543_543975


namespace cos_double_angle_l543_543406

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l543_543406


namespace wall_building_days_l543_543128

theorem wall_building_days (p1 p2 l1 l2 d1 : ℕ)
  (h1: p1 = 18)
  (h2: l1 = 140)
  (h3: d1 = 42)
  (h4: p2 = 30)
  (h5: l2 = 100) :
  let total_work := p1 * d1
  let work_100m := (total_work * l2) / l1
  let days_needed := work_100m / p2
  days_needed = 18 :=
by
  -- Definitions for conditional values
  rw [h1, h2, h3, h4, h5]
  -- Calculate total work
  let total_work:= 18 * 42
  -- Calculate work for a 100 m wall
  let work_100m := (total_work * 100) / 140
  -- Calculate days required
  let days_needed := work_100m / 30
  -- Prove that days_needed is 18
  sorry

end wall_building_days_l543_543128


namespace largest_term_S_8_div_a_8_l543_543859

variable {d a_1 : ℝ}
noncomputable def S (n : ℕ) : ℝ := (d / 2) * n^2 + (a_1 - d / 2) * n

axiom Sn_conditions : S 15 > 0 ∧ S 16 < 0

theorem largest_term_S_8_div_a_8 :
  ∃ a : ℕ → ℝ, a 8 > 0 ∧ ∀ n, n ≤ 15 → a n = a_1 + (n - 1) * d → 
  ∀ m, m ≤ 15 → S m / (a m) ≤ S 8 / (a 8) :=
sorry

end largest_term_S_8_div_a_8_l543_543859


namespace chase_travel_time_l543_543777

/-- Cameron drives at twice the speed of his brother, Chase. -/
def cameron_speed (chase_speed : ℝ) : ℝ := 2 * chase_speed

/-- Danielle drives at three times the speed of Cameron. -/
def danielle_speed (chase_speed : ℝ) : ℝ := 3 * cameron_speed chase_speed

/-- It takes Danielle 30 minutes (0.5 hours) to travel from Granville to Salisbury. -/
def danielle_travel_time : ℝ := 0.5

/-- The distance from Granville to Salisbury is given by Danielle's speed multiplied by her travel time. -/
def distance (chase_speed : ℝ) : ℝ := danielle_speed chase_speed * danielle_travel_time

/-- We are to prove that the travel time for Chase to cover this distance is 180 minutes. -/
theorem chase_travel_time (chase_speed : ℝ) :
  let T := (distance chase_speed) / chase_speed in
  T * 60 = 180 :=
by
  sorry

end chase_travel_time_l543_543777


namespace complex_number_properties_l543_543909

theorem complex_number_properties (z : ℂ) (h : z * (conj z + 2 * (1 : ℂ) * I) = 8 + 6 * I) :
  (z.re = 3) ∧ (z.im = 1) ∧ (z * (conj z) ≠ real.sqrt 10) ∧ (0 < z.re ∧ 0 < z.im) :=
by
  sorry

end complex_number_properties_l543_543909


namespace solve_system_of_equations_l543_543998

variables {a1 a2 a3 a4 : ℝ}

theorem solve_system_of_equations (h_distinct: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :
  ∃ (x1 x2 x3 x4 : ℝ),
    (|a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1) ∧
    (|a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1) ∧
    (|a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1) ∧
    (|a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) ∧
    (x1 = 1 / (a1 - a4)) ∧ (x2 = 0) ∧ (x3 = 0) ∧ (x4 = 1 / (a1 - a4)) :=
sorry

end solve_system_of_equations_l543_543998


namespace exists_constants_for_symmetric_sum_l543_543512

theorem exists_constants_for_symmetric_sum (a b c : ℝ) (h : c ≠ 0) :
  ∀ n : ℕ, n > 0 →
    (∑ i in finset.range (n + 1), i ^ 2 + ∑ i in finset.range n, (n - i) ^ 2) =
    a * n * (b * n ^ 2 + c) :=
sorry

end exists_constants_for_symmetric_sum_l543_543512


namespace Triangle_inequality_l543_543913

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {s : ℝ}

def m_a (a b c : ℝ) : ℝ := 1 / 2 * √(2 * b^2 + 2 * c^2 - a^2)

def t_a (a b c : ℝ) (A : ℝ) : ℝ := (2 * b * c / (b + c)) * Real.cos (A / 2)

def h_a (c B : ℝ) : ℝ := c * Real.sin B

def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

variable (h_cos_A_half_le_1 : Real.cos (A / 2) ≤ 1)
variable (h_triangle_inequality : c * Real.sin B = c * Real.sin B)

theorem Triangle_inequality (a b c A B C : ℝ) (h_cos_A_half_le_1 : Real.cos (A / 2) ≤ 1)
  (h_triangle_inequality : (c * Real.sin B) ≤ (c * Real.sin B)) :
  m_a a b c ≥ t_a a b c A ∧ t_a a b c A ≥ h_a c B :=
by
  sorry

end Triangle_inequality_l543_543913


namespace M_eq_ineq_sqrt3_ab_l543_543541

def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

def M : Set ℝ := { x | f x ≤ 6 }

theorem M_eq : M = { x | -3 ≤ x ∧ x ≤ 3 } := 
sorry

theorem ineq_sqrt3_ab (a b : ℝ) (ha : a ∈ {x | -3 ≤ x ∧ x ≤ 3}) (hb : b ∈ {x | -3 ≤ x ∧ x ≤ 3}) : 
  √3 * |a + b| ≤ |a * b + 3| := 
sorry

end M_eq_ineq_sqrt3_ab_l543_543541


namespace vector_subtraction_l543_543545

theorem vector_subtraction (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) :=
begin
  intros ha hb,
  rw [ha, hb],
  simp,
  sorry
end

end vector_subtraction_l543_543545


namespace number_of_ordered_pairs_l543_543939

theorem number_of_ordered_pairs :
  let U := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  in
  let is_valid_pair (A B : set ℕ) :=
    A ∪ B = U ∧ A ∩ B = ∅ ∧ 
    (A.nonempty ∧ B.nonempty) ∧ 
    (A.card ∉ A) ∧ 
    (B.card ∉ B)
  in
  let N := ∑ n in finset.range 14, if n = 6 then 0 else (finset.card (finset.powerset_len (n - 1) (U.erase (n + 1))) : ℕ)
  in N = 6476 :=
by {
  let U := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  let is_valid_pair := λ (A B : set ℕ), A ∪ B = U ∧ A ∩ B = ∅ ∧ A.nonempty ∧ B.nonempty ∧ A.card ∉ A ∧ B.card ∉ B,
  sorry
}

end number_of_ordered_pairs_l543_543939


namespace midline_length_of_trapezoid_l543_543594

theorem midline_length_of_trapezoid (h : ℝ) (l : ℝ) (d : ℝ) :
  h = 2 ∧ l = 4 ∧ d = l → 
  let midline_length := 3 * Real.sqrt 3 in 
  midline_length = 3 * Real.sqrt 3 :=
by
  intro H
  cases H with H_h H_rest
  cases H_rest with H_l H_d
  let midline_length : ℝ := 3 * Real.sqrt 3
  show midline_length = 3 * Real.sqrt 3
  sorry

end midline_length_of_trapezoid_l543_543594


namespace intervals_of_monotonicity_ext_property_l543_543542

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x^2 + x + 1)

theorem intervals_of_monotonicity (a : ℝ) (ha : a > 0) :
  (if a = 1 / 2 then Ioo (-∞ : ℝ) (+∞ : ℝ)
  else if 0 < a ∧ a < 1 / 2 then Ioo (-∞ : ℝ) (- 1 / a) ∪ Ioo (-2 : ℝ) (+∞ : ℝ)
  else Ioo (-∞ : ℝ) (-2) ∪ Ioo (- 1 / a) (+∞ : ℝ))
  ∧
  (if 0 < a ∧ a < 1 / 2 then Ioc (- 1 / a) (-2 : ℝ)
  else if a > 1 / 2 then Ioc (-2 : ℝ) (- 1 / a)
  else Ioc (-∞ : ℝ) (+∞ : ℝ)) := sorry

theorem ext_property (θ : ℝ) (h : θ ∈ Icc 0 (π / 2)) (a : ℝ) (hx : (fun x => f a x) 1 = 0) :
  | f a (Real.cos θ) - f a (Real.sin θ) | < 2 := sorry

end intervals_of_monotonicity_ext_property_l543_543542


namespace product_of_real_parts_l543_543223

noncomputable def complex_solutions_product : ℂ :=
  let discriminant := complex.i
  let a := 1
  let b := 2
  let x1 := -b/(2*a) + (complex.sqrt ((b^2 - 4*a*discriminant)/(4*a^2)))
  let x2 := -b/(2*a) - (complex.sqrt ((b^2 - 4*a*discriminant)/(4*a^2)))
  (x1.re * x2.re)

theorem product_of_real_parts : 
  (complex_solutions_product) = (1 - real.sqrt 2) / 2 :=
sorry

end product_of_real_parts_l543_543223


namespace satisfies_differential_eqn_l543_543201

noncomputable def y (x : ℝ) : ℝ := 5 * Real.exp (-2 * x) + (1 / 3) * Real.exp x

theorem satisfies_differential_eqn : ∀ x : ℝ, (deriv y x) + 2 * y x = Real.exp x :=
by
  -- The proof is to be provided
  sorry

end satisfies_differential_eqn_l543_543201


namespace Liam_savings_after_trip_and_bills_l543_543547

theorem Liam_savings_after_trip_and_bills :
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  total_savings - bills_cost - trip_cost = 1500 := by
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  sorry

end Liam_savings_after_trip_and_bills_l543_543547


namespace ellipse_constant_sum_ellipse_inverse_sum_bounds_triangle_AOB_area_bounds_l543_543072

-- Ellipse definition and known data
variables {a b : ℝ}
variables (A B : ℝ)
variables {x y : ℝ}
variable (O : (x, y))

-- Conditions
def ellipse (a b x y: ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1

def orthogonal (OA OB : ℝ) : Prop :=
  OA ⟂ OB

def distances (OA OB : ℝ) (x y : ℝ) : Prop :=
  OA = sqrt(x^2 + y^2) ∧ OB = sqrt((a-x)^2 + (b-y)^2)

-- Questions to prove
theorem ellipse_constant_sum (a b OA OB : ℝ) (hx : ellipse a b x y) (ho : orthogonal OA OB) (hdist : distances OA OB x y) : 
  (1 / OA^2) + (1 / OB^2) = (1 / a^2) + (1 / b^2) :=
sorry

theorem ellipse_inverse_sum_bounds (a b OA OB : ℝ) (hx : ellipse a b x y) (ho : orthogonal OA OB) (hdist : distances OA OB x y) : 
  (a + b) / (a * b) ≤ (1 / OA) + (1 / OB) ∧ (1 / OA) + (1 / OB) ≤ sqrt(2 * (a^2 + b^2)) / (a * b) :=
sorry

theorem triangle_AOB_area_bounds (a b OA OB : ℝ) (hx : ellipse a b x y) (ho : orthogonal OA OB) (hdist : distances OA OB x y) :
  a^2 * b^2 / (a^2 + b^2) ≤ (1 / 2) * OA * OB ∧ (1 / 2) * OA * OB ≤ (1 / 2) * a * b :=
sorry

end ellipse_constant_sum_ellipse_inverse_sum_bounds_triangle_AOB_area_bounds_l543_543072


namespace parallelepiped_volume_l543_543898

open Real
open EuclideanSpace

variables {a b : ℝ^3}
variables (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : angle a b = π / 4)

theorem parallelepiped_volume :
  abs (a • ((b + 2 * b × a) × b)) = 1 :=
sorry

end parallelepiped_volume_l543_543898


namespace polynomial_not_product_of_three_l543_543571

theorem polynomial_not_product_of_three (P : ℤ[X]) : 
  (P = (X^2 - 12*X + 11)^4 + 23) → ¬ ∃ P₁ P₂ P₃ : ℤ[X], 
    (P = P₁ * P₂ * P₃) ∧ (¬ constant_polynomial P₁) ∧ (¬ constant_polynomial P₂) ∧ 
    (¬ constant_polynomial P₃) := by
  sorry

end polynomial_not_product_of_three_l543_543571


namespace angle_B_l543_543156

variable (A B C A' B' C': Point)
variable (ΔABC: Triangle A B C)

axiom angle_bisector_A_intersects_perpendicular_bisector_AB 
  (hA: ∃ A', is_angle_bisector A ΔABC ∧ intersects_at_perpendicular_bisector A' A B):
  True

axiom angle_bisector_B_intersects_median_BC 
  (hB: ∃ B', is_angle_bisector B ΔABC ∧ intersects_at_median B' B C):
  True

axiom angle_bisector_C_intersects_perpendicular_bisector_CA 
  (hC: ∃ C', is_angle_bisector C ΔABC ∧ intersects_at_perpendicular_bisector C' C A):
  True

axiom A'_B'_C'_distinct 
  (hDistinct: A' ≠ B' ∧ A' ≠ C' ∧ B' ≠ C'):
  True

theorem angle_B'A'C'_proof 
    (hA : ∃ A', is_angle_bisector A ΔABC ∧ intersects_at_perpendicular_bisector A' A B)
    (hB : ∃ B', is_angle_bisector B ΔABC ∧ intersects_at_median B' B C)
    (hC : ∃ C', is_angle_bisector C ΔABC ∧ intersects_at_perpendicular_bisector C' C A)
    (hDistinct : A' ≠ B' ∧ A' ≠ C' ∧ B' ≠ C'):
    angle A' (line_segment A' B') C' = (π / 2) - (1 / 2 * angle A B C) :=
begin
  sorry
end

end angle_B_l543_543156


namespace determine_points_on_line_l543_543794

def pointA : ℝ × ℝ := (2, 5)
def pointB : ℝ × ℝ := (1, 2.2)
def line_eq (x y : ℝ) : ℝ := 3 * x - 5 * y + 8

theorem determine_points_on_line :
  (line_eq pointA.1 pointA.2 ≠ 0) ∧ (line_eq pointB.1 pointB.2 = 0) :=
by
  sorry

end determine_points_on_line_l543_543794


namespace find_angle_l543_543758

theorem find_angle (x : Real) : 
  (x - (1 / 2) * (180 - x) = -18 - 24/60 - 36/3600) -> 
  x = 47 + 43/60 + 36/3600 :=
by
  sorry

end find_angle_l543_543758


namespace evaluate_expression_l543_543126

def star (A B : ℝ) : ℝ := (A + B) / 2
def hash (A B : ℝ) : ℝ := A * B

theorem evaluate_expression : hash (star 3 4) (star 5 7) = 21 := by
  sorry

end evaluate_expression_l543_543126


namespace union_A_B_eq_R_l543_543883

def set_A (x : ℝ) : Prop := 4 * x - 3 > 0
def set_B (x : ℝ) : Prop := x - 6 < 0

theorem union_A_B_eq_R : { x : ℝ | set_A x } ∪ { x : ℝ | set_B x } = (set.univ : set ℝ) :=
by
  sorry

end union_A_B_eq_R_l543_543883


namespace find_integer_divisible_by_18_and_square_root_in_range_l543_543042

theorem find_integer_divisible_by_18_and_square_root_in_range :
  ∃ x : ℕ, 28 < Real.sqrt x ∧ Real.sqrt x < 28.2 ∧ 18 ∣ x ∧ x = 792 :=
by
  sorry

end find_integer_divisible_by_18_and_square_root_in_range_l543_543042


namespace pie_distribution_l543_543277

theorem pie_distribution :
  let initial_pie := (8 / 9 : ℚ)
  let additional_pie := initial_pie * (10 / 100)
  let total_pie := initial_pie + additional_pie
  let pie_per_employee := total_pie / 4
  pie_per_employee = (11 / 45) :=
by
  let initial_pie := (8 / 9 : ℚ)
  let additional_pie := initial_pie * (10 / 100)
  let total_pie := initial_pie + additional_pie
  let pie_per_employee := total_pie / 4
  calc
    pie_per_employee = (8 / 9 + (8 / 9) * (10 / 100)) / 4 : by sorry
    ... = (11 / 45) : by sorry

end pie_distribution_l543_543277


namespace letter_150_in_pattern_l543_543674

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l543_543674


namespace kostya_verifies_median_l543_543949

-- Define the conditions
variables (n : ℕ) (hn : n > 1) 
def isMedianDevice (coins : ℕ → ℕ) (M : ℕ) : Bool := sorry -- A function representing the median device
def coinsOfDifferentWeights : Set ℕ := sorry  -- Set of coins of different weights

-- Define the statement
theorem kostya_verifies_median (M : ℕ) (hMed : M ∈ coinsOfDifferentWeights):   
  ∃ (verify : (ℕ → ℕ) → ℕ → Bool), 
  (∀ (c : ℕ → ℕ) (m : ℕ), m ∈ coinsOfDifferentWeights → 
    isMedianDevice (c.median) m = verify c m) ∧
  ∀ (c : ℕ → ℕ) (m : ℕ), 
    (m = M → 
      ∃ (attempts ≤ n + 2), 
      verify c m = true) :=
sorry

end kostya_verifies_median_l543_543949


namespace net_change_salary_l543_543313

/-- Given an initial salary S and a series of percentage changes:
    20% increase, 10% decrease, 15% increase, and 5% decrease,
    prove that the net change in salary is 17.99%. -/
theorem net_change_salary (S : ℝ) :
  (1.20 * 0.90 * 1.15 * 0.95 - 1) * S = 0.1799 * S :=
sorry

end net_change_salary_l543_543313


namespace problem1_problem2_exists_largest_k_real_problem3_exists_largest_k_int_l543_543988

-- Problem 1: Prove the inequality for all real numbers x, y
theorem problem1 (x y : ℝ) : x^2 + y^2 + 1 > x * (y + 1) :=
sorry

-- Problem 2: Prove the largest k = sqrt(2) for the inequality with reals
theorem problem2_exists_largest_k_real : ∃ (k : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 1 ≥ k * x * (y + 1)) ∧ k = Real.sqrt 2 :=
sorry

-- Problem 3: Prove the largest k = 3/2 for the inequality with integers
theorem problem3_exists_largest_k_int : ∃ (k : ℝ), (∀ (m n : ℤ), m^2 + n^2 + 1 ≥ k * m * (n + 1)) ∧ k = 3 / 2 :=
sorry

end problem1_problem2_exists_largest_k_real_problem3_exists_largest_k_int_l543_543988


namespace complex_fourth_power_l543_543018

theorem complex_fourth_power (i : ℂ) (hi : i^2 = -1) : (1 - i)^4 = -4 := 
sorry

end complex_fourth_power_l543_543018


namespace translation_symmetric_y_axis_phi_l543_543227

theorem translation_symmetric_y_axis_phi :
  ∀ (f : ℝ → ℝ) (φ : ℝ),
    (∀ x : ℝ, f x = Real.sin (2 * x + π / 6)) →
    (0 < φ ∧ φ ≤ π / 2) →
    (∀ x, Real.sin (2 * (x + φ) + π / 6) = Real.sin (2 * (-x + φ) + π / 6)) →
    φ = π / 6 :=
by
  intros f φ f_def φ_bounds symmetry
  sorry

end translation_symmetric_y_axis_phi_l543_543227


namespace cos_double_angle_proof_l543_543416

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l543_543416


namespace limit_of_function_l543_543781

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2 * Real.pi) ^ 2 / Real.tan (Real.cos x - 1)

-- Define the limit point "2 * Real.pi"
def limit_point : ℝ := 2 * Real.pi

-- Statement of the limit problem
theorem limit_of_function : Filter.Tendsto f (nhds limit_point) (nhds (-2)) := by
  sorry

end limit_of_function_l543_543781


namespace horizontal_asymptote_l543_543905

def numerator (x : ℝ) : ℝ :=
  15 * x^4 + 3 * x^3 + 7 * x^2 + 6 * x + 2

def denominator (x : ℝ) : ℝ :=
  5 * x^4 + x^3 + 4 * x^2 + 2 * x + 1

noncomputable def rational_function (x : ℝ) : ℝ :=
  numerator x / denominator x

theorem horizontal_asymptote :
  ∃ y : ℝ, (∀ x : ℝ, x ≠ 0 → rational_function x = y) ↔ y = 3 :=
by
  sorry

end horizontal_asymptote_l543_543905


namespace find_solution_l543_543823

theorem find_solution (x : ℝ) (t : ℝ) (ht : t > 0) (h : 4^x = t) : 4^x * |4^x - 2| = 3 → x = Real.logb 4 3 :=
by
  sorry

end find_solution_l543_543823


namespace smallest_positive_b_l543_543820

theorem smallest_positive_b (b : ℤ) :
  b % 5 = 1 ∧ b % 4 = 2 ∧ b % 7 = 3 → b = 86 :=
by
  sorry

end smallest_positive_b_l543_543820


namespace binomial_sum_identity_l543_543704

-- Given conditions about omega
def omega : ℂ := (Real.cos (2 * Real.pi / 3) : ℂ) + complex.I * (Real.sin (2 * Real.pi / 3) : ℂ)
lemma omega_cubic_root_unity : omega ^ 3 = 1 := by sorry
lemma omega_sum_zero : 1 + omega + omega ^ 2 = 0 := by sorry

-- Main theorem to prove
theorem binomial_sum_identity (n : ℕ) : 
  1 + ∑ k in finset.range (nat.ceil (n / 3.0)), nat.choose n (3 * k) = 
  1 / 3 * (2 ^ n + 2 * Real.cos (n * Real.pi / 3)) :=
sorry

end binomial_sum_identity_l543_543704


namespace smallest_valid_number_l543_543014

-- Define a predicate that checks if the number is divisible by all elements in a list
def is_divisible_by_all (n : ℕ) (l : list ℕ) : Prop :=
  ∀ d ∈ l, d ∣ n

-- Define a predicate that checks if the number is not divisible by any element in a list
def is_not_divisible_by_any (n : ℕ) (l : list ℕ) : Prop :=
  ∀ d ∈ l, ¬ (d ∣ n)

-- Define the conditions given in the problem
def problem_conditions (n : ℕ) : Prop :=
  is_divisible_by_all n [1, 2, 4, 7, 8] ∧ 
  is_not_divisible_by_any n [3, 5, 6, 9] ∧
  10000 ≤ n ∧ n < 100000

-- Define the proposition to prove that 14728 is the smallest valid number
theorem smallest_valid_number : ∀ n : ℕ, problem_conditions n → 14728 ≤ n :=
by sorry

end smallest_valid_number_l543_543014


namespace sum_of_values_l543_543531

def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_values (z₁ z₂ : ℝ) (h₁ : f (3 * z₁) = 10) (h₂ : f (3 * z₂) = 10) :
  z₁ + z₂ = - (2 / 9) :=
by
  sorry

end sum_of_values_l543_543531


namespace total_stickers_l543_543712

-- Definitions for the given conditions
def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22

-- The theorem to be proven
theorem total_stickers : stickers_per_page * number_of_pages = 220 := by
  sorry

end total_stickers_l543_543712


namespace probability_is_half_l543_543616

-- Define the set of numbers from 1 to 30
def numbers : Finset ℕ := (Finset.range 30).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define the set of multiples of 3 from 1 to 30
def multiples_of_3 : Finset ℕ := numbers.filter (λ n, n % 3 = 0)

-- Define the set of multiples of 4 from 1 to 30
def multiples_of_4 : Finset ℕ := numbers.filter (λ n, n % 4 = 0)

-- Define the set of multiples of 12 from 1 to 30 (multiples of both 3 and 4)
def multiples_of_12 : Finset ℕ := numbers.filter (λ n, n % 12 = 0)

-- Calculate the probability using the principle of inclusion-exclusion
def favorable_outcomes : ℕ := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card

-- Total number of outcomes
def total_outcomes : ℕ := numbers.card

-- Calculate the probability
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 1/2
theorem probability_is_half : probability = 1 / 2 := by
  sorry

end probability_is_half_l543_543616


namespace min_slope_tangent_l543_543242

noncomputable def f (x b a : ℝ) := x^2 + 2 * Real.log x - b * x + a

def slope_of_tangent (b a : ℝ) : ℝ := 2 / b + b

theorem min_slope_tangent (b a : ℝ) (hb : b > 0) :
  slope_of_tangent b a ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_slope_tangent_l543_543242


namespace perpendicular_centroid_orthocenter_l543_543925
open EuclideanGeometry

noncomputable def centroid (A B C : Point) : Point :=
∃ P, ∀ Q, dist(A, Q) + dist(B, Q) + dist(C, Q) ≥ 3 * dist(P, Q)

noncomputable def orthocenter (A B C : Point) : Point :=
∃ R, is_orthogonal_to (R - A) (B - C) ∧ 
      is_orthogonal_to (R - B) (A - C) ∧ 
      is_orthogonal_to (R - C) (A - B)

theorem perpendicular_centroid_orthocenter 
  (A B C D M P Q R S : Point) 
  (quad: convex A B C D)
  (intersect: mid AC M ∧ mid BD M)
  (centroid_AMD: centroid A M D = P)
  (centroid_CMB: centroid C M B = Q)
  (orthocenter_DMC: orthocenter D M C = R)
  (orthocenter_MAB: orthocenter M A B = S):
  is_perpendicular PQ RS :=
sorry

end perpendicular_centroid_orthocenter_l543_543925


namespace find_angle_DSG_l543_543154

noncomputable def angleDGO : ℝ := 60
noncomputable def angleDOG : ℝ := 60
noncomputable def bisects (a b : ℝ) : Prop := b = a / 2

theorem find_angle_DSG (angleDGO_eq_angleDOG : angleDGO = angleDOG) 
                       (angleDOG_is_60 : angleDOG = 60)
                       (DS_bisects_DOG : bisects angleDOG 30) 
                       : 90 = 90 :=
begin
  -- Let α = ∠DSG
  let α : ℝ := angleDOG / 2,
  -- Prove that α = 90 degrees
  have α_eq_90 : α = 90 := sorry,
  exact α_eq_90
end

end find_angle_DSG_l543_543154


namespace middle_segments_ratio_sum_one_l543_543188

theorem middle_segments_ratio_sum_one
  (ABC : Triangle)
  (P : Point)
  (a b c a' b' c' : ℝ)
  (hP_in_ABC : P ∈ interior abc)
  (parallel_a : Line_through P parallel_to side(ABC, A, B))
  (parallel_b : Line_through P parallel_to side(ABC, B, C))
  (parallel_c: Line_through P parallel_to side(ABC, C, A))
  (a_split : ABC.side(A, B) = a' + other_segments_a)
  (a_ratio : a' / side_length(ABC, A, B) = some_ratio_a)
  (b_split : ABC.side(B, C) = b' + other_segments_b)
  (b_ratio: b' / side_length(ABC, B, C) = some_ratio_b)
  (c_split : ABC.side(C, A) = c' + other_segments_c)
  (c_ratio: c' / side_length(ABC, C, A) = some_ratio_c) :
  a' / a + b' / b + c' / c = 1 := 
sorry

end middle_segments_ratio_sum_one_l543_543188


namespace total_students_l543_543915

theorem total_students (T : ℕ)
  (A_cond : (2/9 : ℚ) * T = (a_real : ℚ))
  (B_cond : (1/3 : ℚ) * T = (b_real : ℚ))
  (C_cond : (2/9 : ℚ) * T = (c_real : ℚ))
  (D_cond : (1/9 : ℚ) * T = (d_real : ℚ))
  (E_cond : 15 = e_real) :
  (2/9 : ℚ) * T + (1/3 : ℚ) * T + (2/9 : ℚ) * T + (1/9 : ℚ) * T + 15 = T → T = 135 :=
by
  sorry

end total_students_l543_543915


namespace exist_eight_P_points_l543_543858

-- Consider a triangle ABC and its circumcircle Ω.
variables {A B C : Point} (ABC_circumcircle : Circle)

-- Let's say the lines PA, PB, PC through point P intersect Ω at A1, B1, C1 other than A, B, C
variables {P A1 B1 C1 : Point}

-- Definition for congruency of triangles A1B1C1 and ABC
def are_congruent (Δ₁ Δ₂ : Triangle) : Prop :=
Δ₁ ≈ Δ₂ -- Assume ≈ denotes triangular congruence

-- Condition: A1, B1, C1 form a triangle congruent to ABC
constants (PA_line : Line) (PB_line : Line) (PC_line : Line)
h1: intersects PA_line ABC_circumcircle A1 ∧ A1 ≠ A
h2: intersects PB_line ABC_circumcircle B1 ∧ B1 ≠ B
h3: intersects PC_line ABC_circumcircle C1 ∧ C1 ≠ C
h4: are_congruent ⟨A1, B1, C1⟩ ⟨A, B, C⟩

-- Theorem: Proving there exist exactly 8 such points P
theorem exist_eight_P_points : ∃ (P1 P2 P3 P4 P5 P6 P7 P8 : Point), 
  (∀ (i j : ℕ), i ≠ j → P_i ≠ P_j) ∧
  (∀ (n : ℕ), n < 8 →
    ∃ (A1 B1 C1 : Point),
      intersects (line_through P_n A) ABC_circumcircle A1 ∧ A1 ≠ A ∧
      intersects (line_through P_n B) ABC_circumcircle B1 ∧ B1 ≠ B ∧
      intersects (line_through P_n C) ABC_circumcircle C1 ∧ C1 ≠ C ∧
      are_congruent ⟨A1, B1, C1⟩ ⟨A, B, C⟩) :=
sorry

end exist_eight_P_points_l543_543858


namespace equal_bills_for_telephone_services_l543_543254

theorem equal_bills_for_telephone_services:
  ∀ (x : ℕ), (11 + 0.25 * x = 12 + 0.20 * x) → (x = 20) :=
by
  intro x
  intros h
  sorry

end equal_bills_for_telephone_services_l543_543254


namespace th150th_letter_is_B_l543_543655

def pattern := "ABCD".data

def nth_letter_in_pattern (n : ℕ) : Char :=
  let len := pattern.length
  pattern.get n % len

theorem th150th_letter_is_B :
  nth_letter_in_pattern 150 = 'B' :=
by {
  -- This proof is placed here as a placeholder
  sorry
}

end th150th_letter_is_B_l543_543655


namespace orthocenter_complex_number_l543_543534

theorem orthocenter_complex_number (O : ℂ) (A B C H : ℂ)
  (h_origin : O = 0)
  (h_on_unit_circle_A : |A| = 1)
  (h_on_unit_circle_B : |B| = 1)
  (h_on_unit_circle_C : |C| = 1)
  (h_H_is_orthocenter : H = A + B + C) :
  H = A + B + C :=
by
  sorry

end orthocenter_complex_number_l543_543534


namespace largest_n_satisfying_conditions_l543_543812

noncomputable def is_diff_of_consecutive_cubes (n : ℤ) : Prop :=
  ∃ m : ℤ, n^2 = (m + 1)^3 - m^3

noncomputable def is_perfect_square_offset (n : ℤ) : Prop :=
  ∃ k : ℤ, 2 * n + 99 = k^2

theorem largest_n_satisfying_conditions :
  ∃ n : ℤ, n = 4513 ∧ is_diff_of_consecutive_cubes n ∧ is_perfect_square_offset n :=
by
  use 4513
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end largest_n_satisfying_conditions_l543_543812


namespace find_point_on_graph_of_double_eqn_l543_543088

def graph_point_condition (g : ℝ → ℝ) (cond_point : ℝ × ℝ) (x_val : ℝ) (y_val : ℝ) : Prop :=
  cond_point = (3, 5) ∧ g 3 = 5 ∧ (2 * y_val = 4 * g(3 * x_val) + 6) ∧ (x_val = 1) ∧ (y_val = 13)

theorem find_point_on_graph_of_double_eqn (g : ℝ → ℝ) :
  graph_point_condition g (3, 5) 1 13 → 1 + 13 = 14 := 
by
  intros
  exact rfl

end find_point_on_graph_of_double_eqn_l543_543088


namespace yearly_return_of_1500_investment_l543_543266

theorem yearly_return_of_1500_investment 
  (combined_return_percent : ℝ)
  (total_investment : ℕ)
  (return_500 : ℕ)
  (investment_500 : ℕ)
  (investment_1500 : ℕ) :
  combined_return_percent = 0.085 →
  total_investment = (investment_500 + investment_1500) →
  return_500 = (investment_500 * 7 / 100) →
  investment_500 = 500 →
  investment_1500 = 1500 →
  total_investment = 2000 →
  (return_500 + investment_1500 * combined_return_percent * 100) = (combined_return_percent * total_investment * 100) →
  ((investment_1500 * (9 : ℝ)) / 100) + return_500 = 0.085 * total_investment →
  (investment_1500 * 7 / 100) = investment_1500 →
  (investment_1500 / investment_1500) = (13500 / 1500) →
  (9 : ℝ) = 9 :=
sorry

end yearly_return_of_1500_investment_l543_543266


namespace solve_matrix_expression_l543_543111

theorem solve_matrix_expression (x : ℝ) (a b c d : ℝ) 
  (h_rule : (matrix (fin 2) (fin 2) ℝ) → ℝ)
  (h_rule_eq : h_rule ![![a, c], ![d, b]] = a * b - c * d)
  (h_matrix_eq : h_rule ![![3 * x, x + 1], ![2 * x, x + 2]] = 6) :
  x = -2 + real.sqrt 10 ∨ x = -2 - real.sqrt 10 :=
by {
  have h_formula : (3 * x) * (x + 2) - (x + 1) * (2 * x) = x^2 + 4 * x,
  { sorry },
  
  have h_equation : x^2 + 4 * x = 6,
  { rw [←h_rule_eq, h_formula], exact h_matrix_eq },

  suffices h_quadratic : x^2 + 4 * x - 6 = 0,
  { sorry },

  -- Use quadratic formula
  assumption
}

end solve_matrix_expression_l543_543111


namespace seventeen_divides_odd_exponentiation_diff_l543_543081

theorem seventeen_divides_odd_exponentiation_diff
  (x : ℤ)
  (y z w : ℤ) 
  (h_y_odd : y % 2 = 1) 
  (h_z_odd : z % 2 = 1)
  (h_w_odd : w % 2 = 1) :
  17 ∣ x ^ (y ^ (y ^ w)) - x ^ (y ^ 2) :=
by
  sorry

end seventeen_divides_odd_exponentiation_diff_l543_543081


namespace problem1_problem2_l543_543060

-- Define vectors and cosine differences
variables {α β : ℝ}
def a := (Real.cos α, Real.sin α)
def b := (Real.cos β, Real.sin β)
def cos_diff := Real.cos(α - β)

-- Problem conditions
axiom h1 : |(a.1 - b.1, a.2 - b.2)| = 2 * Real.sqrt 5 / 5
axiom h2 : 0 < α ∧ α < Real.pi / 2
axiom h3 : -Real.pi / 2 < β ∧ β < 0
axiom h4 : Real.cos β = 12 / 13

-- Statements to prove
theorem problem1 : cos_diff = 3 / 5 := sorry
theorem problem2 : Real.sin α = 33 / 65 := sorry

end problem1_problem2_l543_543060


namespace polynomial_q_correct_l543_543078

noncomputable def polynomial_q (x : ℝ) : ℝ :=
  -x^6 + 12*x^5 + 9*x^4 + 14*x^3 - 5*x^2 + 17*x + 1

noncomputable def polynomial_rhs (x : ℝ) : ℝ :=
  x^6 + 12*x^5 + 13*x^4 + 14*x^3 + 17*x + 3

noncomputable def polynomial_2 (x : ℝ) : ℝ :=
  2*x^6 + 4*x^4 + 5*x^2 + 2

theorem polynomial_q_correct (x : ℝ) : 
  polynomial_q x = polynomial_rhs x - polynomial_2 x := 
by
  sorry

end polynomial_q_correct_l543_543078


namespace magnitude_of_sum_l543_543085

-- Define the vectors and conditions
variables (a b : ℝ → ℝ → ℝ)
          (angle_ab : Real.Angle)
          (norm_a norm_b : ℝ)

-- Assume the given conditions
axiom h1 : angle_ab = Real.pi * (2 / 3)
axiom h2 : ∀ x y, ∥a x y∥ = 1
axiom h3 : ∀ x y, ∥b x y∥ = 1

-- Define the dot product of two vectors
noncomputable def dot_product (a b : ℝ → ℝ → ℝ) : ℝ :=
  (∥a 1 0∥ * ∥b 1 0∥ * Real.cos angle_ab)

-- Define the sum of the vectors
noncomputable def a_plus_2b (a b : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ :=
  λ x y, a x y + 2 * b x y

-- Define the norm of the sum of the vectors
noncomputable def norm_a_plus_2b (a b : ℝ → ℝ → ℝ) : ℝ :=
  Real.sqrt (∥a 1 0∥ ^ 2 + 4 * dot_product a b + (2 * ∥b 1 0∥) ^ 2)

-- The main statement to be proved
theorem magnitude_of_sum : norm_a_plus_2b a b = Real.sqrt 3 :=
sorry

end magnitude_of_sum_l543_543085


namespace angle_bisector_intersection_l543_543138

theorem angle_bisector_intersection (A B C P : Type) 
  [Triangle A B C]
  (h1 : Angle A C B = 36)
  (h2 : Bisector (Angle A B C) A B P)
  (h3 : Bisector (Angle C A B) C A P) : 
  Angle A P B = 108 := by
  sorry

end angle_bisector_intersection_l543_543138


namespace trig_identity_proof_l543_543514

theorem trig_identity_proof 
  (α p q : ℝ)
  (hp : p ≠ 0) (hq : q ≠ 0)
  (tangent : Real.tan α = p / q) :
  Real.sin (2 * α) = 2 * p * q / (p^2 + q^2) ∧
  Real.cos (2 * α) = (q^2 - p^2) / (q^2 + p^2) ∧
  Real.tan (2 * α) = (2 * p * q) / (q^2 - p^2) :=
by
  sorry

end trig_identity_proof_l543_543514


namespace ratio_of_areas_l543_543980

-- Definitions of perimeter in Lean terms
def P_A : ℕ := 16
def P_B : ℕ := 32

-- Ratio of the area of region A to region C
theorem ratio_of_areas (s_A s_C : ℕ) (h₀ : 4 * s_A = P_A)
  (h₁ : 4 * s_C = 12) : s_A^2 / s_C^2 = 1 / 9 :=
by 
  sorry

end ratio_of_areas_l543_543980


namespace Tammy_earnings_3_weeks_l543_543586

theorem Tammy_earnings_3_weeks
  (trees : ℕ)
  (oranges_per_tree_per_day : ℕ)
  (oranges_per_pack : ℕ)
  (price_per_pack : ℕ)
  (weeks : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  oranges_per_pack = 6 →
  price_per_pack = 2 →
  weeks = 3 →
  (trees * oranges_per_tree_per_day * weeks * 7) / oranges_per_pack * price_per_pack = 840 :=
by
  intro ht ht12 h6 h2 h3
  -- proof to be filled in here
  sorry

end Tammy_earnings_3_weeks_l543_543586


namespace symmetric_group_abelian_iff_l543_543989

open Function

-- Define the symmetric group S_E
def symmetric_group (E : Type*) := Equiv.Perm E

-- Define what it means for a group to be abelian
def is_abelian (G : Type*) [group G] : Prop :=
  ∀ (a b : G), a * b = b * a

-- Theorem stating symmetric group S_E is abelian if and only if |E| ≤ 2
theorem symmetric_group_abelian_iff (E : Type*) [fintype E] :
  (is_abelian (symmetric_group E)) ↔ (fintype.card E ≤ 2) :=
sorry

end symmetric_group_abelian_iff_l543_543989


namespace sine_expression_equals_one_l543_543535

noncomputable def d : ℝ := 2 * Real.pi / 13

theorem sine_expression_equals_one :
  (sin (4 * d) * sin (8 * d) * sin (12 * d) * sin (16 * d) * sin (20 * d)) /
  (sin (2 * d) * sin (4 * d) * sin (6 * d) * sin (8 * d) * sin (10 * d)) = 1 :=
  sorry

end sine_expression_equals_one_l543_543535


namespace value_of_f_neg_10_l543_543106

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.log (x) / Real.log (2) else f (x + 3)

theorem value_of_f_neg_10 : f (-10) = 1 :=
by
  sorry

end value_of_f_neg_10_l543_543106


namespace problem_l543_543461

open Real

def p (a : ℝ) := ∀ x ∈ Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) := ∃ x₀ : ℝ, x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem problem (a : ℝ) : (¬ p a ∨ ¬ q a) = False → (a ≤ -2 ∨ a = 1) := 
by
  sorry

end problem_l543_543461


namespace find_k_l543_543885

theorem find_k 
  (t k r : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : r = 3 * t)
  (h3 : r = 150) : 
  k = 122 := 
sorry

end find_k_l543_543885


namespace probability_multiple_of_3_or_4_l543_543626

theorem probability_multiple_of_3_or_4 :
  let numbers := Finset.range 30
  let multiples_of_3 := {n ∈ numbers | n % 3 = 0}
  let multiples_of_4 := {n ∈ numbers | n % 4 = 0}
  let multiples_of_12 := {n ∈ numbers | n % 12 = 0}
  let favorable_count := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card
  let probability := (favorable_count : ℚ) / numbers.card
  probability = (1 / 2 : ℚ) :=
by
  sorry

end probability_multiple_of_3_or_4_l543_543626


namespace find_angle_degree_l543_543357

-- Define the angle
variable {x : ℝ}

-- Define the conditions
def complement (x : ℝ) : ℝ := 90 - x
def supplement (x : ℝ) : ℝ := 180 - x

-- Define the given condition
def condition (x : ℝ) : Prop := complement x = (1/3) * (supplement x)

-- The theorem statement
theorem find_angle_degree (x : ℝ) (h : condition x) : x = 45 :=
by
  sorry

end find_angle_degree_l543_543357


namespace negate_existential_l543_543612

theorem negate_existential :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + 4 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0 :=
by
  sorry

end negate_existential_l543_543612


namespace ted_grandfather_time_saved_l543_543213

theorem ted_grandfather_time_saved 
  (d : ℝ) (t_1 : ℝ) (t_2 : ℝ) (t_3 : ℝ) (t_4 : ℝ)
  (r_1 : ℝ) (r_2 : ℝ) (r_3 : ℝ) (r_4 : ℝ)
  (days : ℕ)
  (h_d : d = 1.5)
  (h_r : r_1 = 6 ∧ r_2 = 3 ∧ r_3 = 4.5 ∧ r_4 = 2)
  (h_days : days = 4) :
  let time_actual := (d / r_1) + (d / r_2) + (d / r_3) + (d / r_4),
      time_constant := days * (d / 4.5)
   in (time_actual - time_constant) * 60 = 30 := by
  sorry

end ted_grandfather_time_saved_l543_543213


namespace twenty_five_percent_less_than_eighty_is_forty_eight_l543_543251

theorem twenty_five_percent_less_than_eighty_is_forty_eight:
  (n : ℝ) (h : 80 - 0.25 * 80 = 60) (h1 : 60 = (5 / 4) * n) : n = 48 :=
by {
  sorry
}

end twenty_five_percent_less_than_eighty_is_forty_eight_l543_543251


namespace total_borders_length_is_15_l543_543306

def garden : ℕ × ℕ := (6, 7)
def num_beds : ℕ := 5
def total_length_of_borders (length width : ℕ) : ℕ := 15

theorem total_borders_length_is_15 :
  ∃ a b : ℕ, 
  garden = (a, b) ∧ 
  num_beds = 5 ∧ 
  total_length_of_borders a b = 15 :=
by
  use (6, 7)
  rw [garden]
  rw [num_beds]
  exact ⟨rfl, rfl, sorry⟩

end total_borders_length_is_15_l543_543306


namespace count_factorable_integers_l543_543359

def is_factorable_as_linear_factors (n : ℕ) : Prop :=
  ∃ a b : ℤ, (a + b = 3 ∧ a * b = -n)

theorem count_factorable_integers :
  (∃ N : ℕ, N = 2000) → 44 = (Finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 2000 ∧ is_factorable_as_linear_factors n) (Finset.range 2001)).card :=
by
  intro hN
  sorry

end count_factorable_integers_l543_543359


namespace range_of_a_l543_543875

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else a^x + 2 * a + 2

theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ y ∈ Set.range (f a), y ≥ 3) ↔ (a ∈ Set.Ici (1/2) ∪ Set.Ioi 1) :=
sorry

end range_of_a_l543_543875


namespace total_cantaloupes_l543_543830

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by sorry

end total_cantaloupes_l543_543830


namespace grasshopper_proof_l543_543709

def grasshopper_avoid_M (n : ℕ) (a : Fin n → ℕ) (M : Fin (n - 1) → ℕ) : Prop :=
  let s := ∑ i: Fin n, a i
  ∀ perm : Fin n → ℕ, (∀ i j, i ≠ j → perm i ≠ perm j) →
  (∀ i, ∃ k, perm i < perm k ∧ (∑ j in Fin.range (nat.pred i), perm (Fin.castSucc j)) ≠ M k)

theorem grasshopper_proof (n : ℕ) (a : Fin n → ℕ) (M : Fin (n - 1) → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) (h_not_in : (∑ i, a i) ∉ (Set.range M)) : grasshopper_avoid_M n a M :=
sorry

end grasshopper_proof_l543_543709


namespace district_B_high_schools_l543_543150

theorem district_B_high_schools :
  ∀ (total_schools public_schools parochial_schools private_schools districtA_schools districtB_private_schools: ℕ),
  total_schools = 50 ∧ 
  public_schools = 25 ∧ 
  parochial_schools = 16 ∧ 
  private_schools = 9 ∧ 
  districtA_schools = 18 ∧ 
  districtB_private_schools = 2 ∧ 
  (∃ districtC_schools, 
     districtC_schools = public_schools / 3 + parochial_schools / 3 + private_schools / 3) →
  ∃ districtB_schools, 
    districtB_schools = total_schools - districtA_schools - (public_schools / 3 + parochial_schools / 3 + private_schools / 3) ∧ 
    districtB_schools = 5 := by
  sorry

end district_B_high_schools_l543_543150


namespace cos_double_angle_l543_543394

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l543_543394


namespace factorize_expression_l543_543806

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by sorry

end factorize_expression_l543_543806


namespace arithmetic_sequence_sum_l543_543067

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = -2)
  (h3 : list.sum (list.of_fn (λ k, a (3 * (k + 1))) 33) = -82) :
  list.sum (list.of_fn (λ k, a (1 + 3 * k)) 33) = 50 :=
sorry

end arithmetic_sequence_sum_l543_543067


namespace probability_N18_mod7_is_one_l543_543050

-- Definitions of the conditions
def is_divisible_by (m n : Nat) := ∃ k, n = m * k
def in_range (N : Nat) : Prop := 1 ≤ N ∧ N ≤ 2019

-- The main proof statement
theorem probability_N18_mod7_is_one :
  (∑ N in Finset.range 2020, if (N % 7 = 0) then 0 else 1) = 1722 :=
sorry

end probability_N18_mod7_is_one_l543_543050


namespace arrange_leopards_correct_l543_543960

-- Definitions for conditions
def num_shortest : ℕ := 3
def total_leopards : ℕ := 9
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Calculation of total ways to arrange given conditions
def arrange_leopards (num_shortest : ℕ) (total_leopards : ℕ) : ℕ :=
  let choose2short := (num_shortest * (num_shortest - 1)) / 2
  let arrange2short := 2 * factorial (total_leopards - num_shortest)
  choose2short * arrange2short * factorial (total_leopards - num_shortest)

theorem arrange_leopards_correct :
  arrange_leopards num_shortest total_leopards = 30240 := by
  sorry

end arrange_leopards_correct_l543_543960


namespace possible_value_of_phi_l543_543576

noncomputable def f (ϕ : ℝ) (x : ℝ) : ℝ :=  sin (2 * x + ϕ)

theorem possible_value_of_phi {ϕ : ℝ} (h1 : abs ϕ ≤ π / 2)
  (h2 : ∀ x, f ϕ (x + π / 6) = f ϕ (-x - π / 6)) :
  ϕ = π / 6 :=
begin
  sorry
end

end possible_value_of_phi_l543_543576


namespace minimum_value_of_x_is_4_l543_543903

-- Given conditions
variable {x : ℝ} (hx_pos : 0 < x) (h : log x ≥ log 2 + 1/2 * log x)

-- The minimum value of x is 4
theorem minimum_value_of_x_is_4 : x ≥ 4 :=
by
  sorry

end minimum_value_of_x_is_4_l543_543903


namespace triangle_area_l543_543845

theorem triangle_area (a b c : ℝ) 
  (h : |a - sqrt 8| + sqrt (b^2 - 5) + (c - sqrt 3)^2 = 0) :
  ∃ (area : ℝ), (a = sqrt 8 ∧ b = sqrt 5 ∧ c = sqrt 3) ∧ 
    (a + b > c ∧ a + c > b ∧ b + c > a) ∧ a^2 = b^2 + c^2 ∧ 
    area = (sqrt 5 * sqrt 3) / 2 :=
  sorry

end triangle_area_l543_543845


namespace cone_plane_distance_l543_543229

theorem cone_plane_distance (H α : ℝ) : 
  (x = 2 * H * (Real.sin (α / 4)) ^ 2) :=
sorry

end cone_plane_distance_l543_543229


namespace area_ABC_l543_543941

noncomputable def condition (P A B : Point ℝ) (m : ℝ) (h : m > 0) : Prop :=
  3 * dist P A + 4 * P = m * B

noncomputable def area_of_triangle (A B C : Point ℝ) : ℝ := sorry

theorem area_ABC (P A B C : Point ℝ)  (m : ℝ) 
  (h1 : m > 0) 
  (h2 : condition P A B m) 
  (h3 : area_of_triangle A B P = 8) : 
  area_of_triangle A B C = 14 := sorry

end area_ABC_l543_543941


namespace log_fraction_simplification_l543_543039

theorem log_fraction_simplification :
  (log 3 4) / (log 9 8) = 4 / 3 :=
by
  -- Primary conditions from the problem
  have h1 : 4 = 2^2 := by simp,
  have h2 : 8 = 2^3 := by simp,
  have h3 : 9 = 3^2 := by simp,

  -- Logarithm transformation with properties
  have log_prop := log_div (log_base_pow (3^2) (2^3)),
  have log_base_trans := log_base_pow 3 (2^2),

  -- Final simplification using law of logarithms and arithmetic
  calc
    (log 3 4) / (log 9 8) 
      = (2 * (log 3 2)) / ((3/2) * (log 3 2)) : by rw [log_base_trans, log_prop]
  ... = 4 / 3 : by sorry


end log_fraction_simplification_l543_543039


namespace multiply_and_simplify_fractions_l543_543772

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end multiply_and_simplify_fractions_l543_543772


namespace probability_is_half_l543_543614

-- Define the set of numbers from 1 to 30
def numbers : Finset ℕ := (Finset.range 30).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define the set of multiples of 3 from 1 to 30
def multiples_of_3 : Finset ℕ := numbers.filter (λ n, n % 3 = 0)

-- Define the set of multiples of 4 from 1 to 30
def multiples_of_4 : Finset ℕ := numbers.filter (λ n, n % 4 = 0)

-- Define the set of multiples of 12 from 1 to 30 (multiples of both 3 and 4)
def multiples_of_12 : Finset ℕ := numbers.filter (λ n, n % 12 = 0)

-- Calculate the probability using the principle of inclusion-exclusion
def favorable_outcomes : ℕ := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card

-- Total number of outcomes
def total_outcomes : ℕ := numbers.card

-- Calculate the probability
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 1/2
theorem probability_is_half : probability = 1 / 2 := by
  sorry

end probability_is_half_l543_543614


namespace number_of_members_l543_543699

theorem number_of_members (n : ℕ) (h1 : n * n = 8649) : n = 93 := 
begin
  sorry
end

end number_of_members_l543_543699


namespace number_of_boys_is_correct_l543_543151

-- Define the conditions
def total_students : ℕ := 420
def students_playing_soccer : ℕ := 250
def percent_soccer_players_boys : ℝ := 0.82
def girls_not_playing_soccer : ℕ := 63

-- The statement we need to prove
theorem number_of_boys_is_correct :
  let total_boys := 
    let boys_playing_soccer := (percent_soccer_players_boys * students_playing_soccer).toNat
    let girls_playing_soccer := students_playing_soccer - boys_playing_soccer
    let total_girls := girls_not_playing_soccer + girls_playing_soccer
    total_students - total_girls
  in total_boys = 312 :=
by
  sorry

end number_of_boys_is_correct_l543_543151


namespace boxes_in_carton_l543_543284

variable (c b : ℕ)

-- Define the conditions
def case_contains_cartons : Prop := True
def box_contains_paper_clips : Prop := true
def total_paper_clips_2_cases : 2 * (200 * b * c) = 400

-- State the theorem to prove the relation between b and c
theorem boxes_in_carton : total_paper_clips_2_cases c b → b * c = 1 :=
by
  sorry

end boxes_in_carton_l543_543284


namespace parallelepiped_volume_l543_543901

open Real EuclideanSpace

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (angle_a_b : Real := pi / 4)
variables (unit_a : ∥a∥ = 1)
variables (unit_b : ∥b∥ = 1)
variable (angle_condition : angle_between a b = angle_a_b)

theorem parallelepiped_volume :
  abs (a • ((b + 2 * (b × a)) × b)) = 1 :=
sorry

end parallelepiped_volume_l543_543901


namespace table_filling_l543_543798

noncomputable def numWaysToFillTable : ℕ :=
  2^25

theorem table_filling (n m : ℕ) (table : matrix (fin n) (fin m) (ℤ)) :
  n = 6 → m = 6 →
  (∀ i, (∏ j, table i j) > 0) →
  (∀ j, (∏ i, table i j) > 0) →
  numWaysToFillTable = 2^25 :=
begin
  intros n_eq m_eq row_prod_pos col_prod_pos,
  sorry
end

end table_filling_l543_543798


namespace equation_no_solution_at_5_l543_543096

theorem equation_no_solution_at_5 :
  ∀ (some_expr : ℝ), ¬(1 / (5 + 5) + some_expr = 1 / (5 - 5)) :=
by
  intro some_expr
  sorry

end equation_no_solution_at_5_l543_543096


namespace enhancing_integer_count_l543_543341

def is_enhancing (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  (∀ (i : ℕ), i < digits.length - 1 → digits.get i < digits.get (i + 1)) ∨
  (∀ (i : ℕ), i < digits.length - 1 → digits.get i > digits.get (i + 1))

noncomputable def count_enhancing_integers : ℕ :=
  let one_digit := 9 in
  let multi_digit := 2 * (2^9 - 1) in
  one_digit + multi_digit

theorem enhancing_integer_count : count_enhancing_integers = 1031 :=
  sorry

end enhancing_integer_count_l543_543341


namespace divisible_by_4_count_l543_543945

theorem divisible_by_4_count : 
  let a_n (n : ℕ) := list.join (list.map (λ x, integer.to_digits 10 x) (list.range (n + 1))),
      count_divisible_by_4 := list.countp (λ k, (k % 100 // 10) * 10 + (k % 10) % 4 = 0) (list.range 101) in
  ∀ k, 1 ≤ k → k ≤ 100 → count_divisible_by_4 = 20 :=
by
  sorry

end divisible_by_4_count_l543_543945


namespace max_number_of_cubes_l543_543345

structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Cube :=
(A1 A2 F1 F2 E H J : Point)
(A1_A2_on_lower_face : A1.z = 0 ∧ A2.z = 0)
(F1_F2_on_upper_face : F1.z = 1 ∧ F2.z = 1)
(A1_A2_F1_F2_not_coplanar : ¬∃(l : ℝ) (m : ℝ) (n : ℝ), (λ p : Point, l * p.x + m * p.y + n * p.z = 0 ∧ l * (A1.x) + m * (A1.y) + n * (A1.z) = 0 ∧ l * (A2.x) + m * (A2.y) + n * (A2.z) = 0 ∧ l * (F1.x) + m * (F1.y) + n * (F1.z) = 0 ∧ l * (F2.x) + m * (F2.y) + n * (F2.z) = 0))
(E_on_front_face : E.y = 0)
(H_on_back_face : H.y = 1)
(J_on_right_face : J.x = 1)

theorem max_number_of_cubes (c : Cube) : 2 := sorry

end max_number_of_cubes_l543_543345


namespace cos_of_double_angles_l543_543413

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l543_543413


namespace boaster_guarantee_distinct_balls_l543_543870

noncomputable def canGuaranteeDistinctBallCounts (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)) : Prop :=
  ∀ i j : Fin 2018, i ≠ j → boxes i ≠ boxes j

theorem boaster_guarantee_distinct_balls :
  ∃ (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)),
  canGuaranteeDistinctBallCounts boxes pairs :=
sorry

end boaster_guarantee_distinct_balls_l543_543870


namespace number_of_valid_permutations_l543_543324

def is_permutation (l : List ℕ) : Prop :=
  l ~ [1, 2, 3, 4, 5, 6]

def valid_sequence (a : ℕ → ℕ) : Prop :=
  a 1 ≠ 1 ∧ a 3 ≠ 3 ∧ a 5 ≠ 5 ∧ a 1 < a 3 ∧ a 3 < a 5

noncomputable def count_valid_permutations : ℕ :=
  (List.permutations [1, 2, 3, 4, 5, 6]).filter (λ l, valid_sequence (λ i, l.nth_le (i - 1) (by simp [Nat.succ_le_succ_iff, List.length_perm])), sorry).length

theorem number_of_valid_permutations : count_valid_permutations = 36 :=
sorry

end number_of_valid_permutations_l543_543324


namespace math_problem_l543_543432

-- Define the given ellipse and conditions
def ellipse (a b : ℝ) (h : a > b) : set (ℝ × ℝ) :=
  {p | let ⟨x, y⟩ := p in x^2 / a^2 + y^2 / b^2 = 1}

def eccentricity_condition (a b : ℝ) : Prop :=
  sqrt(1 - b^2 / a^2) = 1 / 2

def min_dot_product_condition (a b : ℝ) (c : ℝ) : Prop :=
  let e := (x^2 /a^2 + y^2 /b^2 = 1) ∧ 
            (∀ P, P ∈ e → 
                    let ⟨x, y⟩ := P
                    (-c-x, -y)•(c-x, -y) = 2)

-- Problem statement
theorem math_problem (a b c : ℝ) (h1 :  a > b) (h2 : eccentricity_condition a b) (h3 : min_dot_product_condition a b c) : 
  (a = 2 ∧ b = sqrt 3 ∧ c = 1 ∧ 
   {e | let ⟨x, y⟩ := e in x^2 / 4 + y^2 / 3 = 1} ∧ 
   ∃ M N : ℝ × ℝ, M = (-sqrt 6 / 3, 0) ∧ N = (sqrt 6 / 3, 0) ∧ 
   ∀ E : ℝ × ℝ, tangent_to_triangle_sides E → dist E M + dist E N = constant_value) :=
begin
  sorry
end

end math_problem_l543_543432


namespace binomial_p_value_l543_543448

theorem binomial_p_value (n : ℕ) (p : ℝ) (ξ : ℝ) (Eξ : ℝ) (Dξ : ℝ) :
  (ξ ∼ B(n, p)) → (Eξ = n * p) → (Eξ = 7) → (Dξ = n * p * (1 - p)) → (Dξ = 6) → p = 1 / 7 :=
by
  intros hξ hEξ1 hEξ2 hDξ1 hDξ2
  sorry

end binomial_p_value_l543_543448


namespace lines_parallel_l543_543928

-- Definitions for lines and perpendicularity (assuming lines are represented as sets of points in 2D space)
def line (p1 p2 : Real × Real) : set (Real × Real) := 
    {q : Real × Real | ∃ t : Real, q = (p1.1 + t * (p2.1 - p1.1), p1.2 + t * (p2.2 - p1.2))}

def perpendicular (l1 l2 : set (Real × Real)) : Prop := ∃ (p1 p2 p3 : Real × Real), line p1 p2 = l1 ∧ line p2 p3 = l2 ∧ (p3.2 - p2.2) * (p2.1 - p1.1) + (p3.1 - p2.1) * (p2.2 - p1.2) = 0

def parallel (l1 l2 : set (Real × Real)) : Prop := ∃ (p1 p2 p3 : Real × Real), line p1 p2 = l1 ∧ line p3 (p3 + (p2 - p1)) = l2

-- Assume we have three lines l, m, n represented as sets of points in a plane
variables (l m n : set (Real × Real))

-- The given Lean theorem statement to prove the lines l and n are parallel provided the conditions
theorem lines_parallel (P : Real × Real) 
    (hlm : perpendicular l m)
    (hnm : perpendicular n m) : 
    parallel l n :=
sorry

end lines_parallel_l543_543928


namespace solve_for_x_l543_543582

theorem solve_for_x (x : ℚ) : 
  x = 48 / (7 - 3/8 + 4/9) → x = 3456 / 509 := 
by 
  assume h : x = 48 / (7 - 3/8 + 4/9)
  show x = 3456 / 509
  sorry

end solve_for_x_l543_543582


namespace correct_equation_is_D_l543_543689

theorem correct_equation_is_D :
  (∀ a b c d : Prop, (a = (sqrt 16 = 4 ∨ sqrt 16 = -4)) → 
                     (b = (sqrt 16 = 4)) → 
                     (c = (sqrt ((-4)^2) = 4)) → 
                     (d = (∛ (-27) = -3)) → 
                     d) :=
by
  intros a b c d hA hB hC hD
  exact hD
  sorry

end correct_equation_is_D_l543_543689


namespace smallest_positive_period_pi_not_smallest_positive_period_pi_for_f2_answer_is_C_l543_543320

def f1 (x : ℝ) : ℝ := Real.cos x * Real.sin x
def f2 (x : ℝ) : ℝ := Real.cos x + Real.sin x
def f3 (x : ℝ) : ℝ := Real.sin x / Real.cos x
def f4 (x : ℝ) : ℝ := 2 * (Real.sin x)^2

theorem smallest_positive_period_pi :
  (∀ x : ℝ, f1 (x + π) = f1 x) ∧ (∀ x : ℝ, f3 (x + π) = f3 x) ∧ (∀ x : ℝ, f4 (x + π) = f4 x) :=
by
  split
  · sorry
  · split
    · sorry
    · sorry

theorem not_smallest_positive_period_pi_for_f2 :
  ¬ (∀ x : ℝ, f2 (x + π) = f2 x) :=
by
  sorry

theorem answer_is_C :
  (∀ x : ℝ, f1 (x + π) = f1 x) ∧ (∀ x : ℝ, f3 (x + π) = f3 x) ∧ (∀ x : ℝ, f4 (x + π) = f4 x) ∧ ¬ (∀ x : ℝ, f2 (x + π) = f2 x) :=
by
  split
  · exact smallest_positive_period_pi.1
  · split
    · exact smallest_positive_period_pi.2.1
    · split
      · exact smallest_positive_period_pi.2.2
      · exact not_smallest_positive_period_pi_for_f2

end smallest_positive_period_pi_not_smallest_positive_period_pi_for_f2_answer_is_C_l543_543320


namespace daves_apps_count_l543_543786

theorem daves_apps_count (x : ℕ) : 
  let initial_apps : ℕ := 21
  let added_apps : ℕ := 89
  let total_apps : ℕ := initial_apps + added_apps
  let deleted_apps : ℕ := x
  let more_added_apps : ℕ := x + 3
  total_apps - deleted_apps + more_added_apps = 113 :=
by
  sorry

end daves_apps_count_l543_543786


namespace collinear_and_bisect_angle_l543_543526

noncomputable def t_value : ℝ :=
  (-39 * Real.sqrt 70 - 70 * Real.sqrt 22) / (11 * Real.sqrt 22 - 53 * Real.sqrt 70)

def a : ℝ × ℝ × ℝ := (5, -3, -6)
def c : ℝ × ℝ × ℝ := (-3, -2, 3)

def b : ℝ × ℝ × ℝ := (5 - 8 * t_value, -3 + t_value, -6 + 9 * t_value)

theorem collinear_and_bisect_angle :
  ∃ b : ℝ × ℝ × ℝ, (∃ k : ℝ, b = (k • a ∧ ∃ l : ℝ, b = l • c)) ∧ 
  ((a.fst * b.fst + a.snd * b.snd + a.trd * b.trd) / (Real.sqrt (a.fst^2 + a.snd^2 + a.trd^2) * Real.sqrt (b.fst^2 + b.snd^2 + b.trd^2)))
  = ((b.fst * c.fst + b.snd * c.snd + b.trd * c.trd) / (Real.sqrt (b.fst^2 + b.snd^2 + b.trd^2) * Real.sqrt (c.fst^2 + c.snd^2 + c.trd^2))) :=
begin
  use b,
  sorry
end

end collinear_and_bisect_angle_l543_543526


namespace probability_multiple_of_3_or_4_l543_543622

-- Given the numbers 1 through 30 are written on 30 cards one number per card,
-- and Sara picks one of the 30 cards at random,
-- the probability that the number on her card is a multiple of 3 or 4 is 1/2.

-- Define the set of numbers from 1 to 30
def numbers := finset.range 30 \ {0}

-- Define what it means to be a multiple of 3 or 4 within the given range
def is_multiple_of_3_or_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 4 = 0

-- Define the set of multiples of 3 or 4 within the given range
def multiples_of_3_or_4 := numbers.filter is_multiple_of_3_or_4

-- The probability calculation
theorem probability_multiple_of_3_or_4 : 
  (multiples_of_3_or_4.card : ℚ) / numbers.card = 1 / 2 :=
begin
  -- The set multiples_of_3_or_4 contains 15 elements
  have h_multiples_card : multiples_of_3_or_4.card = 15, sorry,
  -- The set numbers contains 30 elements
  have h_numbers_card : numbers.card = 30, sorry,
  -- Therefore, the probability is 15/30 = 1/2
  rw [h_multiples_card, h_numbers_card],
  norm_num,
end

end probability_multiple_of_3_or_4_l543_543622


namespace unique_zero_point_l543_543876

theorem unique_zero_point (a : ℝ) (f : ℝ → ℝ) (x0 : ℝ) (h₀ : f = λ x, a * x^3 - 3 * x^2 + 1)
  (h₁ : ∃! y, f y = 0) (h₂ : x0 < 0) : a > 3 / 2 := by
  sorry

end unique_zero_point_l543_543876


namespace cos_of_double_angles_l543_543409

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l543_543409


namespace infinite_sum_equals_one_third_l543_543331

noncomputable def series_sum := ∑' n : ℕ, (n^2 + 2 * n - 2) / (n + 3)!

theorem infinite_sum_equals_one_third : series_sum = 1 / 3 := 
by
  sorry

end infinite_sum_equals_one_third_l543_543331


namespace cos_double_angle_proof_l543_543417

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l543_543417


namespace probability_of_x_in_0_2_l543_543196

-- Condition: Random selection implies uniform distribution
noncomputable def interval_length (a b : ℝ) : ℝ := b - a

def interval_minus1_to_3 := interval_length (-1) 3
def interval_0_to_2 := interval_length 0 2

theorem probability_of_x_in_0_2 :
  let x_randomly_selected : Prop := (uniformly_random_from_interval : (-1, 3))
  interval_minus1_to_3 = 4 ∧ interval_0_to_2 = 2 →
  probability (x ∈ [0, 2]) = 1/2 := 
sorry

end probability_of_x_in_0_2_l543_543196


namespace problem1_1_problem2_1_l543_543369

noncomputable theory
open scoped Classical

-- Problem 1
def bread_weight_distribution := (1000 : ℝ, 50 ^ 2 : ℝ)

def average_weight_distribution (n : ℕ) :
  (ℝ × ℝ) := (1000, 50 ^ 2 / n)

theorem problem1_1 :
  ∀ (n : ℕ) (Y : ℝ), (n = 25) →
  let dist := average_weight_distribution n in
  let μ := dist.1 in
  let σ := dist.2.sqrt in
  P (Y < 980) = 0.02275 :=
by sorry

-- Problem 2
def box1_distribution := (6, 2) -- total_loaves, black_loaves
def box2_distribution := (8, 3) -- total_loaves, black_loaves

def probability_distribution : ℝ × ℝ × ℝ :=
  (53 / 140, 449 / 840, 73 / 840)

def expectation (prob : ℝ × ℝ × ℝ) (values : ℕ × ℕ × ℕ := (0, 1, 2)) :
  ℝ := ∑ i in [values.1, values.2, values.3], i * prob.i
  
theorem problem2_1 :
  ∀ (distribution : ℝ × ℝ × ℝ), (distribution = probability_distribution) →
  let E := expectation distribution in
  E = 17 / 24 :=
by sorry

end problem1_1_problem2_1_l543_543369


namespace max_m_plus_n_l543_543638

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end max_m_plus_n_l543_543638


namespace simplify_expression_evaluate_expression_l543_543991

theorem simplify_expression (x : ℕ) (hx : x = 8) : 
  (2 * x / (x + 1) - (2 * x + 4) / (x ^ 2 - 1) / (x + 2) / (x ^ 2 - 2 * x + 1)) = (2 / (x + 1)) :=
by {
  sorry
}

theorem evaluate_expression : 
  simplify_expression 8 rfl = 2 / 9 :=
by {
  sorry
}

end simplify_expression_evaluate_expression_l543_543991


namespace smallest_perimeter_consecutive_integers_triangle_l543_543033

theorem smallest_perimeter_consecutive_integers_triangle :
  ∃ (a b c : ℕ), 
    1 < a ∧ a + 1 = b ∧ b + 1 = c ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    a + b + c = 12 :=
by
  -- proof placeholder
  sorry

end smallest_perimeter_consecutive_integers_triangle_l543_543033


namespace find_x_l543_543212

noncomputable def f (x : ℝ) : ℝ := real.root 4 ((2 * x + 6) / 5)

theorem find_x : ∀ x : ℝ, f (3 * x) = 3 * f x → x = -40 / 13 := 
by
  intros x h
  sorry

end find_x_l543_543212


namespace euler_criterion_l543_543540

theorem euler_criterion (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (hp_gt_two : p > 2) (ha : 1 ≤ a ∧ a ≤ p - 1) : 
  (∃ b : ℕ, b^2 % p = a % p) ↔ a^((p - 1) / 2) % p = 1 :=
sorry

end euler_criterion_l543_543540


namespace find_n_eq_l543_543162

def C_n (n : ℕ) : ℝ :=
  2048 * (1 - 1 / (2^n))

def D_n (n : ℕ) : ℝ :=
  64 * (1 - 1 / ((-2)^n))

theorem find_n_eq (n : ℕ) (h : 1 ≤ n) : C_n n = D_n n → n = 6 := by
  sorry

end find_n_eq_l543_543162


namespace marked_price_of_each_article_l543_543739

noncomputable def marked_price_each (total_price : ℝ) (discount_rate : ℝ) (number_of_articles : ℕ) : ℝ :=
  let marked_price := total_price / (1 - discount_rate) in
  marked_price / number_of_articles

theorem marked_price_of_each_article :
  marked_price_each 50 0.40 2 = 41.67 :=
by
  sorry

end marked_price_of_each_article_l543_543739


namespace max_abc_l543_543530

theorem max_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_eq : (a * b) + c = (a + c) * (b + c)) : abc ≤ ↑(1/27) =
begin
  sorry
end

end max_abc_l543_543530


namespace work_done_resistive_force_l543_543283

noncomputable def mass : ℝ := 0.01  -- 10 grams converted to kilograms
noncomputable def v1 : ℝ := 400.0  -- initial speed in m/s
noncomputable def v2 : ℝ := 100.0  -- final speed in m/s

noncomputable def kinetic_energy (m v : ℝ) : ℝ := 0.5 * m * v^2

theorem work_done_resistive_force :
  let KE1 := kinetic_energy mass v1
  let KE2 := kinetic_energy mass v2
  KE1 - KE2 = 750 :=
by
  sorry

end work_done_resistive_force_l543_543283


namespace tammy_earnings_after_3_weeks_l543_543587

noncomputable def oranges_picked_per_day (num_trees : ℕ) (oranges_per_tree : ℕ) : ℕ :=
  num_trees * oranges_per_tree

noncomputable def packs_sold_per_day (oranges_per_day : ℕ) (oranges_per_pack : ℕ) : ℕ :=
  oranges_per_day / oranges_per_pack

noncomputable def total_packs_sold_in_weeks (packs_per_day : ℕ) (days_in_week : ℕ) (num_weeks : ℕ) : ℕ :=
  packs_per_day * days_in_week * num_weeks

noncomputable def money_earned (total_packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  total_packs * price_per_pack

theorem tammy_earnings_after_3_weeks :
  let num_trees := 10
  let oranges_per_tree := 12
  let oranges_per_pack := 6
  let price_per_pack := 2
  let days_in_week := 7
  let num_weeks := 3
  oranges_picked_per_day num_trees oranges_per_tree /
  oranges_per_pack *
  days_in_week *
  num_weeks *
  price_per_pack = 840 :=
by {
  sorry
}

end tammy_earnings_after_3_weeks_l543_543587


namespace curve_equation_l543_543444

-- Defining the conditions
def is_circle_center_origin_radius_1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def is_right_half_unit_circle (x y : ℝ) : Prop :=
  x = sqrt (1 - y^2)

def is_lower_half_unit_circle (x y : ℝ) : Prop :=
  y = - sqrt (1 - x^2)

-- Stating the equivalent proof problem
theorem curve_equation 
  (x y : ℝ)
  (h1 : is_circle_center_origin_radius_1 x y) : 
  (x - sqrt (1 - y^2)) * (y + sqrt (1 - x^2)) = 0 :=
sorry

end curve_equation_l543_543444


namespace series1_converges_l543_543510

noncomputable def series1 := ∑' n : ℕ, (n^2 + n + 1) / (4 * n^4 + 5 * n^3 + 6 * n^2 + n + 2)

-- We assume the convergence of the p-series
noncomputable def p_series_converges := summable (λ n : ℕ, 1 / (n ^ 2))

-- We prove the given series converges
theorem series1_converges (h : p_series_converges) : summable series1 := 
sorry

end series1_converges_l543_543510


namespace bernardo_wins_l543_543765

/-- 
Bernardo and Silvia play the following game. An integer between 0 and 999 inclusive is selected
and given to Bernardo. Whenever Bernardo receives a number, he doubles it and passes the result 
to Silvia. Whenever Silvia receives a number, she adds 50 to it and passes the result back. 
The winner is the last person who produces a number less than 1000. The smallest initial number 
that results in a win for Bernardo is 16, and the sum of the digits of 16 is 7.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem bernardo_wins (N : ℕ) (h : 16 ≤ N ∧ N ≤ 18) : sum_of_digits 16 = 7 :=
by
  sorry

end bernardo_wins_l543_543765


namespace cos_double_angle_proof_l543_543420

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l543_543420


namespace max_m_plus_n_l543_543637

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end max_m_plus_n_l543_543637


namespace perpendicular_condition_l543_543889

theorem perpendicular_condition (x : ℝ) : (x - 1, 2) ⬝ (2, 1) = 0 ↔ x = 0 :=
by
  sorry

end perpendicular_condition_l543_543889


namespace max_red_stamps_l543_543185

-- Definitions based on the problem conditions
def price_red := 1.1
def num_blue := 80
def price_blue := 0.8
def num_yellow := 7
def price_yellow := 2
def total_earnings := 100

-- Total earnings equations for each type of stamp
def earnings_blue := num_blue * price_blue
def earnings_yellow := num_yellow * price_yellow

-- Prove the number of red stamps needed to reach the total earnings
theorem max_red_stamps : ∃ R : ℕ, (price_red * R + earnings_blue + earnings_yellow = total_earnings) ∧ (R = 20) :=
by
  sorry -- Proof will be filled in here

end max_red_stamps_l543_543185


namespace can_reach_14_from_458_l543_543511

theorem can_reach_14_from_458 : 
  ∃ (f : ℕ → ℕ), f 0 = 458 ∧ f (n + 1) = (2 * f n ∨ f (n + 1) = f n / 10) ∧ ∃ n, f n = 14 :=
sorry

end can_reach_14_from_458_l543_543511


namespace range_of_a_l543_543884

noncomputable def A := { x : ℝ | 0 < x ∧ x < 2 }
noncomputable def B (a : ℝ) := { x : ℝ | 0 < x ∧ x < (2 / a) }

theorem range_of_a (a : ℝ) (h : 0 < a) : (A ∩ (B a)) = A → 0 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l543_543884


namespace probability_is_half_l543_543613

-- Define the set of numbers from 1 to 30
def numbers : Finset ℕ := (Finset.range 30).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define the set of multiples of 3 from 1 to 30
def multiples_of_3 : Finset ℕ := numbers.filter (λ n, n % 3 = 0)

-- Define the set of multiples of 4 from 1 to 30
def multiples_of_4 : Finset ℕ := numbers.filter (λ n, n % 4 = 0)

-- Define the set of multiples of 12 from 1 to 30 (multiples of both 3 and 4)
def multiples_of_12 : Finset ℕ := numbers.filter (λ n, n % 12 = 0)

-- Calculate the probability using the principle of inclusion-exclusion
def favorable_outcomes : ℕ := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card

-- Total number of outcomes
def total_outcomes : ℕ := numbers.card

-- Calculate the probability
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 1/2
theorem probability_is_half : probability = 1 / 2 := by
  sorry

end probability_is_half_l543_543613


namespace calc_30_exp_l543_543010

theorem calc_30_exp :
  30 * 30 ^ 10 = 30 ^ 11 :=
by sorry

end calc_30_exp_l543_543010


namespace range_of_x_l543_543848

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : sqrt(1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
  (Real.pi / 4) ≤ x ∧ x ≤ 5 * (Real.pi / 4) :=
by 
  sorry

end range_of_x_l543_543848


namespace distributive_property_l543_543332

theorem distributive_property (a b : ℝ) : 3 * a * (2 * a - b) = 6 * a^2 - 3 * a * b :=
by
  sorry

end distributive_property_l543_543332


namespace frustum_volume_ratio_l543_543747

theorem frustum_volume_ratio 
(base_edge : ℕ) (original_height : ℕ) (scale_factor : ℚ) :
  base_edge = 36 → 
  original_height = 15 → 
  scale_factor = 1/3 → 
  let original_volume := (1 / 3) * (base_edge ^ 2) * original_height in 
  let smaller_base_edge := base_edge * scale_factor in 
  let smaller_height := original_height * scale_factor in 
  let smaller_volume := (1 / 3) * (smaller_base_edge ^ 2) * smaller_height in 
  ((original_volume - smaller_volume) / original_volume) = 26 / 27 :=
by
  sorry

end frustum_volume_ratio_l543_543747


namespace cubes_set_closed_under_multiplication_l543_543956

-- Define the set of cubes of positive integers
def cubes_set : Set ℕ :=
  { n | ∃ m : ℕ, m > 0 ∧ n = m^3 }

-- Specify the question and answer (i.e., statement of closure under multiplication)
theorem cubes_set_closed_under_multiplication : ∀ a b ∈ cubes_set, a * b ∈ cubes_set :=
by
  sorry

end cubes_set_closed_under_multiplication_l543_543956


namespace proposition_false_n4_l543_543855

variable {P : ℕ → Prop}

theorem proposition_false_n4
  (h_ind : ∀ (k : ℕ), k ≠ 0 → P k → P (k + 1))
  (h_false_5 : P 5 = False) :
  P 4 = False :=
sorry

end proposition_false_n4_l543_543855


namespace vectors_dot_product_sum_l543_543494

-- Define the vectors and their magnitudes
variables (AB BC CA : ℝ³)

-- Conditions in the problem
axiom H1 : ‖AB‖ = 2
axiom H2 : ‖BC‖ = 3
axiom H3 : ‖CA‖ = 4
axiom H4 : AB + BC + CA = 0

-- The theorem to prove
theorem vectors_dot_product_sum :
  AB • BC + BC • CA + CA • AB = -29 / 2 :=
by {
  sorry
}

end vectors_dot_product_sum_l543_543494


namespace sin_triple_eq_sin_x_nontrivial_solutions_l543_543207

theorem sin_triple_eq_sin_x_nontrivial_solutions (k : ℝ) :
  (∃ x : ℝ, sin (3 * x) = k * sin x ∧ sin x ≠ 0) ↔ (-1 ≤ k ∧ k < 3) :=
sorry

end sin_triple_eq_sin_x_nontrivial_solutions_l543_543207


namespace number_of_solutions_l543_543890

theorem number_of_solutions : 
  let S := { x : ℕ | 1 ≤ x ∧ x ≤ 150 ∧ 
             ¬ (x ∣= (y^2 ∧ y ≤ 10) ∨ (3 ∣ x ∧ x ≤ 150)) } 
  in S.card = 93 :=
begin
  sorry
end

end number_of_solutions_l543_543890


namespace g_is_odd_l543_543931

def g (x : ℝ) : ℝ := log (x^3 + sqrt (1 + x^6))

theorem g_is_odd (x : ℝ) : g (-x) = -g (x) :=
by
  sorry

end g_is_odd_l543_543931


namespace parallelepiped_volume_l543_543900

open Real EuclideanSpace

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (angle_a_b : Real := pi / 4)
variables (unit_a : ∥a∥ = 1)
variables (unit_b : ∥b∥ = 1)
variable (angle_condition : angle_between a b = angle_a_b)

theorem parallelepiped_volume :
  abs (a • ((b + 2 * (b × a)) × b)) = 1 :=
sorry

end parallelepiped_volume_l543_543900


namespace math_proof_problem_l543_543107

def f (ω x : ℝ) : ℝ := sqrt 3 * sin (ω * x) + cos (ω * x)
def g (f : ℝ → ℝ) (φ x : ℝ) : ℝ := f (x + φ)
def even_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

theorem math_proof_problem (ω x₁ x₂ φ m : ℝ) (h_pos_ω : ω > 0)
  (h_abs_diff_f : |f ω x₁ - f ω x₂| = 4) (h_abs_diff_x : |x₁ - x₂| = π / 2) :
  ω = 2 ∧ 
  (∀ φ, 0 < φ ∧ φ < π / 2 → even_function (g (f ω) φ)) → φ = π / 6 ∧
  (∃ x ∈ Ioo (π / 16) (2 * π), f ω x = sqrt 3) ∧
  (∀ m, (∀ x ∈ Icc (-π / 12) m, -2 ≤ f ω x) → m ∈ Icc (2 * π / 3) ⊤) :=
begin
  sorry
end

end math_proof_problem_l543_543107


namespace probability_three_different_suits_is_169_over_425_l543_543318

def card_deck := Finset (Fin 52)
def suits : Finset (Fin 4) := {0, 1, 2, 3}
def cards_of_suit (s : Fin 4) : Finset (Fin 52) := sorry -- Assume we have a function to give cards of each suit

noncomputable def probability_three_different_suits : ℚ :=
  let total_cards := (52 : ℕ)
  let first_prob := 1
  let second_prob := (39 : 51 : ℚ)
  let third_prob := (26 : 50 : ℚ)
  first_prob * second_prob * third_prob

theorem probability_three_different_suits_is_169_over_425 :
  probability_three_different_suits = (169 : 425 : ℚ) := 
by {
  sorry
}

end probability_three_different_suits_is_169_over_425_l543_543318


namespace best_trip_representation_l543_543985

structure TripConditions where
  initial_walk_moderate : Prop
  main_road_speed_up : Prop
  bird_watching : Prop
  return_same_route : Prop
  coffee_stop : Prop
  final_walk_moderate : Prop

theorem best_trip_representation (conds : TripConditions) : 
  conds.initial_walk_moderate →
  conds.main_road_speed_up →
  conds.bird_watching →
  conds.return_same_route →
  conds.coffee_stop →
  conds.final_walk_moderate →
  True := 
by 
  intros 
  exact True.intro

end best_trip_representation_l543_543985


namespace sum_of_legs_of_larger_triangle_is_correct_l543_543253

noncomputable def similar_right_triangles_sum_of_legs 
    (area_smaller : ℝ)
    (area_larger : ℝ)
    (hypotenuse_smaller : ℝ)
    (ratio_legs : ℝ)
    (scale_factor : ℝ) : ℝ :=
  let a := scale_factor * hypotenuse_smaller * sqrt ratio_legs
  let b := scale_factor * hypotenuse_smaller / sqrt ratio_legs
  in a + b

theorem sum_of_legs_of_larger_triangle_is_correct :
  ∀ (area_smaller area_larger hypotenuse_smaller : ℝ),
    area_smaller = 12 →
    area_larger = 192 →
    hypotenuse_smaller = 10 →
    ratio_legs = 2 →
    similar_right_triangles_sum_of_legs area_smaller area_larger hypotenuse_smaller ratio_legs (sqrt (area_larger / area_smaller)) = 24 * sqrt 3 :=
sorry

end sum_of_legs_of_larger_triangle_is_correct_l543_543253


namespace find_a_n_find_S_n_find_b_n_find_T_n_l543_543095

def a_n (n : ℕ) := 2 * n + 1
def S_n (n : ℕ) := n^2 + 2 * n
def b_n (n : ℕ) := 4 * (real.cbrt 5)^(n - 2)
def T_n (n : ℕ) := (4 / real.cbrt 5) * (1 - (real.cbrt 5)^n) / (1 - real.cbrt 5)

axiom a_4 : a_n 4 = 9
axiom a_8_minus_a_3 : a_n 8 - a_n 3 = 10

axiom b_2 : b_n 2 = a_n 3 - 3
axiom b_5 : b_n 5 = 6 * a_n 2 + 2

-- Lean statements for the proof goals
theorem find_a_n : ∀ n, a_n n = 2 * n + 1 := sorry
theorem find_S_n : ∀ n, S_n n = n^2 + 2 * n := sorry
theorem find_b_n : ∀ n, b_n n = 4 * (real.cbrt 5)^(n - 2) := sorry
theorem find_T_n : ∀ n, T_n n = (4 / real.cbrt 5) * (1 - (real.cbrt 5)^n) / (1 - real.cbrt 5) := sorry

end find_a_n_find_S_n_find_b_n_find_T_n_l543_543095


namespace BC_eq_l543_543024

-- Definitions of the conditions
variables (p : ℝ) (h_pos : p > 0)
variables (A F B C : ℝ×ℝ)
-- A is on the parabola axis
-- parabola equation is y^2 = 2px
variable h_parabola : ∀ (P : ℝ×ℝ), (P.2) ^ 2 = 2 * p * (P.1)
-- Point A is where line l intersects the axis
variable h_axis_intersection : A.2 = 0
-- Given distances |AF| = 6
variable h_AF : dist A F = 6
-- Given vector relationship A-F = 2(B-F)
variable h_vector_rel : A - F = 2 * (B - F)

-- The distance |BC|
noncomputable def BC : ℝ := dist B C

-- The theorem to be proved
theorem BC_eq : BC = 9 / 2 :=
by
  sorry

end BC_eq_l543_543024


namespace mitch_family_milk_drank_l543_543492

variable (regular_milk : ℝ) (soy_milk : ℝ)
variable (total_milk : ℝ)

-- Conditions
def regular_milk := 0.5
def soy_milk := 0.1

-- Statement to prove
theorem mitch_family_milk_drank :
  regular_milk + soy_milk = 0.6 :=
by
  sorry

end mitch_family_milk_drank_l543_543492


namespace boat_speed_in_still_water_l543_543267

-- Definitions based on the conditions
def downstream_distance : ℝ := 10
def downstream_time : ℝ := 3
def upstream_distance : ℝ := 10
def upstream_time : ℝ := 6

-- Using the given speeds to calculate B
def downstream_speed : ℝ := downstream_distance / downstream_time
def upstream_speed : ℝ := upstream_distance / upstream_time

def B : ℝ := (downstream_speed + upstream_speed) / 2

-- The proof problem
theorem boat_speed_in_still_water : B = 2.5 := by
  sorry

end boat_speed_in_still_water_l543_543267


namespace line_intersects_circle_midpoint_trajectory_is_circle_m_range_exists_l543_543424

noncomputable def circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5
noncomputable def line (m x y : ℝ) : Prop := mx - y + 1 + 2 * m = 0

-- Prove that line l always intersects circle C at two distinct points
theorem line_intersects_circle (m x₁ y₁ x₂ y₂ : ℝ) (h₁ : circle x₁ y₁) (h₂ : circle x₂ y₂) (h_line₁ : line m x₁ y₁) (h_line₂ : line m x₂ y₂) (h_diff : (x₁, y₁) ≠ (x₂, y₂)) :
  ∃ A B, circle (A.1) (A.2) ∧ circle (B.1) (B.2) ∧ line m (A.1) (A.2) ∧ line m (B.1) (B.2) ∧ A ≠ B := sorry

-- Equation of the trajectory of the midpoint M of chord AB
noncomputable def midpoint_trajectory (x y : ℝ) : Prop := (x + 2)^2 + (y - 1 / 2)^2 = 1 / 4

theorem midpoint_trajectory_is_circle :
  ∀ (x y : ℝ), midpoint_trajectory x y → ∃ r : ℝ, (x + 2)^2 + (y - 1 / 2)^2 = r ∧ r = 1 / 4 := sorry

-- Determine if there exists real number m such that there are four points on circle C at a distance of 4√5/5 from line l
theorem m_range_exists (m : ℝ) :
  (∃ (P1 P2 P3 P4 : ℝ × ℝ), circle (P1.1) (P1.2) ∧ circle (P2.1) (P2.2) ∧ circle (P3.1) (P3.2) ∧ circle (P4.1) (P4.2) ∧ line m (P1.1) (P1.2) ∧ line m (P2.1) (P2.2) ∧ line m (P3.1) (P3.2) ∧ line m (P4.1) (P4.2) ∧ abs (m * P1.1 - P1.2 + 1 + 2 * m) / sqrt(1 + m^2) = 4 * sqrt(5) / 5 )
  ↔ (m > 2 ∨ m < -2) := sorry

end line_intersects_circle_midpoint_trajectory_is_circle_m_range_exists_l543_543424


namespace quadrilateral_area_proof_l543_543742

noncomputable def cyclic_quadrilateral_area (R : ℝ) (A B φ : ℝ) : ℝ :=
  2 * R^2 * (Real.sin A) * (Real.sin B) * (Real.sin φ)

theorem quadrilateral_area_proof (R : ℝ) (A B φ : ℝ) :
  ∃ (S : ℝ), S = cyclic_quadrilateral_area R A B φ :=
begin
  use 2 * R^2 * (Real.sin A) * (Real.sin B) * (Real.sin φ),
  rw cyclic_quadrilateral_area,
  sorry,
end

end quadrilateral_area_proof_l543_543742


namespace suitable_for_lottery_method_B_l543_543000

def total_items_A : Nat := 3000
def samples_A : Nat := 600

def total_items_B (n: Nat) : Nat := 2 * 15
def samples_B : Nat := 6

def total_items_C : Nat := 2 * 15
def samples_C : Nat := 6

def total_items_D : Nat := 3000
def samples_D : Nat := 10

def is_lottery_suitable (total_items : Nat) (samples : Nat) (different_factories : Bool) : Bool :=
  total_items <= 30 && samples <= total_items && !different_factories

theorem suitable_for_lottery_method_B : 
  is_lottery_suitable (total_items_B 2) samples_B false = true :=
  sorry

end suitable_for_lottery_method_B_l543_543000


namespace microorganism_colored_area_correct_l543_543198

theorem microorganism_colored_area_correct :
  let a := 30
  let b := 40
  let c := 50
  let speed := 1 / 6
  let movement_time := 60   -- time in seconds
  let radius := 10          -- distance microorganisms can travel
  let area_triangle := 1 / 2 * a * b -- area of the right triangle
  let perimeter := a + b + c -- perimeter of the triangle
  let area_strips := radius * perimeter -- area covered by strips along the sides
  let area_sectors := 100 * Real.pi -- combined area of circular sectors
  let total_area := area_triangle + area_strips + area_sectors -- total covered area

  total_area ≈ 2114 := by
  let a := 30
  let b := 40
  let c := 50
  let speed := 1 / 6
  let movement_time := 60
  let distance := speed * movement_time
  have r : radius = distance := by sorry

  have h1 : a^2 + b^2 = c^2 := by sorry
  have area_triangle_correct : 1 / 2 * a * b = 600 := by sorry
  have perimeter_correct : a + b + c = 120 := by sorry
  have strips_correct : radius * 120 = 1200 := by sorry
  have sectors_correct : 100 * Real.pi ≈ 314 := by sorry
  have total_area_correct : 600 + 1200 + 314 = 2114 := by sorry

  rfl -- the final proof step

end microorganism_colored_area_correct_l543_543198


namespace probability_multiple_of_3_or_4_l543_543627

theorem probability_multiple_of_3_or_4 :
  let numbers := Finset.range 30
  let multiples_of_3 := {n ∈ numbers | n % 3 = 0}
  let multiples_of_4 := {n ∈ numbers | n % 4 = 0}
  let multiples_of_12 := {n ∈ numbers | n % 12 = 0}
  let favorable_count := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card
  let probability := (favorable_count : ℚ) / numbers.card
  probability = (1 / 2 : ℚ) :=
by
  sorry

end probability_multiple_of_3_or_4_l543_543627


namespace dave_tickets_final_l543_543761

theorem dave_tickets_final (initial_tickets : ℕ) (candy_bar_cost : ℕ)
    (beanie_cost : ℕ) (racing_game_win : ℕ) (claw_machine_win : ℕ) :
    initial_tickets = 11 →
    candy_bar_cost = 3 →
    beanie_cost = 5 →
    racing_game_win = 10 →
    claw_machine_win = 7 →
    (initial_tickets - candy_bar_cost - beanie_cost + racing_game_win + claw_machine_win) * 2 = 40 :=
by
  intros h_initial h_candy h_beanie h_racing h_claw
  rw [h_initial, h_candy, h_beanie, h_racing, h_claw]
  norm_num
  sorry

end dave_tickets_final_l543_543761


namespace mail_in_rebate_amount_l543_543584

def initial_cost : ℝ := 150
def cashback_percentage : ℝ := 10 / 100
def final_cost_after_cashback_and_rebate : ℝ := 110

def cashback_amount : ℝ := initial_cost * cashback_percentage
def cost_after_cashback : ℝ := initial_cost - cashback_amount
def rebate_amount : ℝ := cost_after_cashback - final_cost_after_cashback_and_rebate

theorem mail_in_rebate_amount :
  rebate_amount = 25 := by
  sorry

end mail_in_rebate_amount_l543_543584


namespace curve_equation_min_distance_l543_543854

theorem curve_equation 
  (A B C : ℝ × ℝ)
  (h_AB_len : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
  (h_mid_C : C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_A_xaxis : A.2 = 0)
  (h_B_yaxis : B.1 = 0) :
  C.1^2 + C.2^2 = 1 := 
sorry

theorem min_distance 
  (a b x y : ℝ)
  (h_curve_eq : x^2 + y^2 = 1)
  (h_inter_line : sqrt 2 * a * x + b * y = 1)
  (h_right_angle_triangle : (x, y) ≠ (0, 0) ∧ ∃ D, (D.1^2 + D.2^2 = 1 ∧ sqrt 2 * a * D.1 + b * D.2 = 1 ∧ (x, y) ≠ D)) :
  ∃ d, d = sqrt 2 - 1 ∧ ∀ u v, sqrt ((a - u)^2 + (b - v - 1)^2) >= d := 
sorry

end curve_equation_min_distance_l543_543854


namespace min_distance_PQ_l543_543178

noncomputable def minimum_distance_between_P_and_Q : ℝ :=
  let P : ℝ × ℝ := (0, 0) in
  let d := (1 / (real.sqrt 2)) in d

theorem min_distance_PQ : 
  ∀ (P : ℝ × ℝ), P = (0, 0) → minimum_distance_between_P_and_Q = (real.sqrt 2 / 2) := by
  intros P hP
  rw [hP]
  simp [minimum_distance_between_P_and_Q]
  sorry

end min_distance_PQ_l543_543178


namespace parallelepiped_volume_l543_543899

open Real
open EuclideanSpace

variables {a b : ℝ^3}
variables (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : angle a b = π / 4)

theorem parallelepiped_volume :
  abs (a • ((b + 2 * b × a) × b)) = 1 :=
sorry

end parallelepiped_volume_l543_543899


namespace part1_part2_l543_543110

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (x k : ℝ) : ℝ := k * x ^ 3 - x - 2

theorem part1 (k : ℝ) : (¬ MonotonicOn (λ x, g x k) (Set.Ioo 1 2)) ↔ (1/12 < k ∧ k < 1/3) :=
by sorry

theorem part2 (k : ℝ) : (∀ x, 1 ≤ x → f x ≥ g x k + x + 2) → k ≤ -Real.exp 1 :=
by sorry

end part1_part2_l543_543110


namespace point_A_inside_circle_O_l543_543867

-- Definitions based on conditions in the problem
def radius := 5 -- in cm
def distance_to_center := 4 -- in cm

-- The theorem to be proven
theorem point_A_inside_circle_O (r d : ℝ) (hr : r = 5) (hd : d = 4) (h : r > d) : true :=
by {
  sorry
}

end point_A_inside_circle_O_l543_543867


namespace complement_of_A_in_U_l543_543957

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x | x^2 - 6*x + 5 = 0}

theorem complement_of_A_in_U :
  (\U \ A) = {2, 3, 4} :=
    sorry

end complement_of_A_in_U_l543_543957


namespace number_of_valid_sets_l543_543235

open Set

variable {α : Type} (a b : α)

def is_valid_set (M : Set α) : Prop := M ∪ {a} = {a, b}

theorem number_of_valid_sets (a b : α) : (∃! M : Set α, is_valid_set a b M) := 
sorry

end number_of_valid_sets_l543_543235


namespace red_marble_difference_l543_543650

theorem red_marble_difference :
  ∃ (total1 total2 white1 white2 red1 red2 : ℕ),
    -- Conditions
    total1 = 140 ∧
    total2 = 72 ∧
    white1 = (3 * total1) / 10 ∧
    white2 = total2 / 4 ∧
    white1 + white2 = 60 ∧
    red1 = total1 - white1 ∧
    red2 = total2 - white2 ∧
    -- Question -> Answer
    red1 - red2 = 44 :=
begin
  -- Adjusted problem statement with conditions and the required result to prove
  use [140, 72, 42, 18, 98, 54],
  simp only [true_and, eq_self_iff_true, add_comm],
end

end red_marble_difference_l543_543650


namespace x_squared_plus_y_squared_l543_543129

theorem x_squared_plus_y_squared (x y : ℝ) (h₁ : x - y = 18) (h₂ : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end x_squared_plus_y_squared_l543_543129


namespace find_purchase_price_l543_543237

noncomputable def purchase_price (a : ℝ) : ℝ := a
def retail_price : ℝ := 1100
def discount_rate : ℝ := 0.8
def profit_rate : ℝ := 0.1

theorem find_purchase_price (a : ℝ) (h : purchase_price a * (1 + profit_rate) = retail_price * discount_rate) : a = 800 := by
  sorry

end find_purchase_price_l543_543237


namespace part_I_part_II_l543_543453

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 1|

-- Part (Ⅰ): Solve the inequality f(x) ≤ 2
theorem part_I (x : ℝ) : f(x) ≤ 2 ↔ 0 ≤ x ∧ x ≤ (2 / 3) := sorry

-- Part (Ⅱ): Find the range of m given the condition
theorem part_II (m : ℝ) (h : ∀ x : ℝ, ∃ a ∈ Set.Icc (-2 : ℝ) 1, f(x) ≥ f(a) + m) : m ≤ 0 := sorry

end part_I_part_II_l543_543453


namespace problem1_problem2_l543_543012

-- Problem 1
theorem problem1 :
  (sqrt 7 - sqrt 13) * (sqrt 7 + sqrt 13) + (sqrt 3 + 1)^2 - (sqrt 6 * sqrt 3) / sqrt 2 + abs (-sqrt 3) = -3 + 3 * sqrt 3 :=
by sorry

-- Problem 2
theorem problem2 (a : ℝ) (h : a < 0) :
  sqrt (4 - (a + 1/a)^2) - sqrt (4 + (a - 1/a)^2) = -2 :=
by sorry

end problem1_problem2_l543_543012


namespace binomial_coefficient_largest_l543_543132

theorem binomial_coefficient_largest (n : ℕ) :
  (let poly := 18 * (x : ℝ) ^ 2 - (17 / (2 * x : ℝ))) in 
  (∀ k, k = 4 → binomial_coefficient n k) < binomial_coefficient n 8 :=
  n = 8 :=
sorry

end binomial_coefficient_largest_l543_543132


namespace find_constant_a_l543_543459

noncomputable def f (a t : ℝ) : ℝ := (t - 2)^2 - 4 - a

theorem find_constant_a :
  (∃ (a : ℝ),
    (∀ (t : ℝ), -1 ≤ t ∧ t ≤ 1 → |f a t| ≤ 4) ∧ 
    (∃ (t : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ |f a t| = 4)) →
  a = 1 :=
sorry

end find_constant_a_l543_543459


namespace octahedron_volume_l543_543430

theorem octahedron_volume (r : ℝ) (h1 : r = 2)
  (h2 : ∀ P Q R: Plane, (P ⊥ Q) ∧ (Q ⊥ R) ∧ (R ⊥ P)) :
  volume_of_octahedron r = (32 / 3) := 
sorry

end octahedron_volume_l543_543430


namespace cos_double_angle_l543_543403

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l543_543403


namespace time_difference_correct_l543_543264

def timePerMileBoy (totalTime : ℕ) (distance : ℕ) : ℕ :=
  totalTime / distance

def timePerMileAdult (totalTime : ℕ) (distance : ℕ) : ℕ :=
  totalTime / distance

def timeDifference (timeBoy : ℕ) (timeAdult : ℕ) : ℕ :=
  timeAdult - timeBoy

theorem time_difference_correct :
  timeDifference (timePerMileBoy 135 18) (timePerMileAdult 150 6) = 17.5 :=
by
  sorry

end time_difference_correct_l543_543264


namespace fillings_count_is_seven_l543_543826

-- Declare the ingredients as a set
def ingredients := {'rice', 'meat', 'eggs'}

-- Define the number of different combinations of fillings
def number_of_fillings := 
  (ingredients.to_finset.powerset.erase ∅).card
  
-- State the theorem to prove
theorem fillings_count_is_seven : number_of_fillings = 7 := by
  sorry

end fillings_count_is_seven_l543_543826


namespace mrs_hilt_apple_pies_l543_543556

-- Given definitions
def total_pies := 30 * 5
def pecan_pies := 16

-- The number of apple pies
def apple_pies := total_pies - pecan_pies

-- The proof statement
theorem mrs_hilt_apple_pies : apple_pies = 134 :=
by
  sorry -- Proof step to be filled

end mrs_hilt_apple_pies_l543_543556


namespace Mary_age_is_10_l543_543977

-- Define the parameters for the ages of Rahul and Mary
variables (Rahul Mary : ℕ)

-- Conditions provided in the problem
def condition1 := Rahul = Mary + 30
def condition2 := Rahul + 20 = 2 * (Mary + 20)

-- Stating the theorem to be proved
theorem Mary_age_is_10 (Rahul Mary : ℕ) 
  (h1 : Rahul = Mary + 30) 
  (h2 : Rahul + 20 = 2 * (Mary + 20)) : 
  Mary = 10 :=
by 
  sorry

end Mary_age_is_10_l543_543977


namespace self_tangent_curves_l543_543136

-- Definitions of the given curves
def curve1 (x y : ℝ) : Prop := x^2 - y^2 = 1
def curve2 (x y : ℝ) : Prop := y = x^2 - |x|
def curve3 (x y : ℝ) : Prop := y = 3 * sin x + 4 * cos x
def curve4 (x y : ℝ) : Prop := |x| + 1 = sqrt (4 - y^2)

-- Statement of the proof problem: which curves have self-tangents
theorem self_tangent_curves :
  (∃ (x1 x2 y : ℝ), curve1 x1 y ∧ curve1 x2 y ∧ x1 ≠ x2) = false ∧
  (∃ (x y : ℝ), curve2 x y ∧ (∃ y', curve2 (-x) y' ∧ y = y')) ∧
  (∃ (x y : ℝ), curve3 x y ∧ (∃ x', curve3 x' y ∧ x ≠ x')) ∧
  (∃ (x1 x2 y : ℝ), curve4 x1 y ∧ curve4 x2 y ∧ x1 ≠ x2) = false :=
by
  sorry

end self_tangent_curves_l543_543136


namespace problem_statement_l543_543832

theorem problem_statement (n : ℕ) (h₀ : 3 ≤ n)
  (a b : ℕ → ℝ) 
  (h₁: ∑ i in Finset.range n, a i = ∑ i in Finset.range n, b i)
  (h₂: 0 < a 1 ∧ a 1 = a 2 ∧ ∀ i < n-2, a i + a (i + 1) = a (i + 2))
  (h₃: 0 < b 1 ∧ b 1 ≤ b 2 ∧ ∀ i < n-2, b i + b (i + 1) ≤ b (i + 2)) :
  a (n-2) + a (n-1) ≤ b (n-2) + b (n-1) :=
sorry

end problem_statement_l543_543832


namespace cos_of_double_angles_l543_543408

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l543_543408


namespace cos_double_angle_l543_543404

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l543_543404


namespace min_value_ST_is_11_18_l543_543161

open EuclideanGeometry

noncomputable def minimum_ST (A B C D M S T O : Point)
    (hRhom : rhombus A B C D)
    (hAC : length A C = 20)
    (hBD : length B D = 24)
    (hMBC : lies_on_segment M B C)
    (hPerpS : foot_point M S A C)
    (hPerpT : foot_point M T B D)
    (hOACBD : midpoint_intersection A C B D O) : Real :=
  5 * Real.sqrt 5

theorem min_value_ST_is_11_18 (A B C D M S T O : Point)
    (hRhom : rhombus A B C D)
    (hAC : length A C = 20)
    (hBD : length B D = 24)
    (hMBC : lies_on_segment M B C)
    (hPerpS : foot_point M S A C)
    (hPerpT : foot_point M T B D)
    (hOACBD : midpoint_intersection A C B D O) :
  minimum_ST A B C D M S T O hRhom hAC hBD hMBC hPerpS hPerpT hOACBD = 5 * Real.sqrt 5 := 
sorry

end min_value_ST_is_11_18_l543_543161


namespace train_speed_in_km_hr_l543_543312

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 9.679225661947045
noncomputable def distance_covered : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := distance_covered / crossing_time
noncomputable def speed_km_hr : ℝ := speed_m_s * 3.6

theorem train_speed_in_km_hr : speed_km_hr = 90.0216 := by
  sorry

end train_speed_in_km_hr_l543_543312


namespace Diamond_example_l543_543629

def Diamond (a b : ℕ) : ℕ := a^2 * b^2 - b + 2

theorem Diamond_example : Diamond 3 4 = 142 := 
by
  calc
    Diamond 3 4 = 3^2 * 4^2 - 4 + 2 : rfl
    ... = 9 * 16 - 4 + 2 : by rw [pow_two, pow_two]
    ... = 144 - 4 + 2 : rfl
    ... = 140 + 2 : by rw [← add_assoc, sub_add_cancel]
    ... = 142 : add_comm

end Diamond_example_l543_543629


namespace express_w_l543_543436

theorem express_w (w a b c : ℝ) (x y z : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ w ∧ b ≠ w ∧ c ≠ w)
  (h1 : x + y + z = 1)
  (h2 : x * a^2 + y * b^2 + z * c^2 = w^2)
  (h3 : x * a^3 + y * b^3 + z * c^3 = w^3)
  (h4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  w = - (a * b * c) / (a * b + b * c + c * a) :=
sorry

end express_w_l543_543436


namespace area_of_triangle_ABC_l543_543506

theorem area_of_triangle_ABC
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_C_eq : Real.sin C = Real.sqrt 3 / 3)
  (sin_CBA_eq : Real.sin C + Real.sin (B - A) = Real.sin (2 * A))
  (a_minus_b_eq : a - b = 3 - Real.sqrt 6)
  (c_eq : c = Real.sqrt 3) :
  1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 2 / 2 := sorry

end area_of_triangle_ABC_l543_543506


namespace question1_extremum_at_1_question2_extremum_in_interval_question3_sequence_condition_l543_543102

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + 1/x + a * x

-- Condition 1: Given function specification
variable (a : ℝ)

-- Condition 2 & related proof question
theorem question1_extremum_at_1 : (∀ x > 0, f x a) f(1, a) = 1 := sorry

-- Condition 3 & related proof question
theorem question2_extremum_in_interval (t : ℝ) (a : ℝ) :
  (∃ c ∈ Set.Ioo 2 3, Deriv f c a = 0) ↔  a ∈ Set.Ioo (-1/4) (-2/9) := sorry

-- Condition 4 & related proof question
theorem question3_sequence_condition (x : ℕ → ℝ) : (∀ n : ℕ, 0 < x n) →
  (∀ n : ℕ, Real.log (x n) + 1/(x (n+1)) < 1) → x 1 ≤ 1 := sorry

end question1_extremum_at_1_question2_extremum_in_interval_question3_sequence_condition_l543_543102


namespace min_value_and_inequality_l543_543108

theorem min_value_and_inequality (x y : ℝ) (m : ℝ):
  (∃ x : ℝ, f x = (|2 * x - 1| + |x - 3|)) ∧ 
  (∀ x y : ℝ, f x > m * (|y + 1| - |y - 1|)) →
  (f 0 >= (|2 * (1 / 2) - 1| + |(1 / 2) - 3|) = (5 / 2)) ∧
  (m ∈ Ioo (-5/4) (5/4)) :=
by
  sorry

end min_value_and_inequality_l543_543108


namespace intersection_and_circle_properties_l543_543468

theorem intersection_and_circle_properties :
  let l1 := λ x : ℝ, 2 * x + 3,
      l2 := λ x : ℝ, x + 2,
      C := (-1, 1 : ℝ),
      circle_eq := λ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 1 in
  (l1 C.1 = C.2) ∧
  (l2 C.1 = C.2) ∧
  (circle_eq C.1 C.2) ∧
  (∃ t : ℝ, (t = 1 ∨ t = -1) ∧ 
    let line := λ x y : ℝ, x + y + t = 0,
        height := |t| / sqrt(2),
        s_triangle_max := 1 / 2 in
    height = sqrt(2) / 2 ∧
    (∀ A B : (ℝ × ℝ), (line A.1 A.2 = 0 ∧ line B.1 B.2 = 0 ∧ (circle_eq A.1 A.2 ∧ circle_eq B.1 B.2)) → 
      (∃ S : ℝ, S ∈ (set_of (λ S_max, S_max ≤ s_triangle_max))))) := 
by
  sorry

end intersection_and_circle_properties_l543_543468


namespace vector_problem_l543_543214

variables (a b : ℝ × ℝ)
noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
let dot_product := a.1 * b.1 + a.2 * b.2 in
let magnitude_a := Real.sqrt (a.1^2 + a.2^2) in
let magnitude_b := Real.sqrt (b.1^2 + b.2^2) in
Real.arccos (dot_product / (magnitude_a * magnitude_b))

theorem vector_problem :
  angle_between (2, 0) b = Real.pi / 3 → 
  Real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 1 → 
  Real.sqrt ((2 + 2 * b.1) ^ 2 + (2 * b.2) ^ 2) = 2 * Real.sqrt 3 :=
begin
  sorry
end

end vector_problem_l543_543214


namespace marbles_around_perimeter_l543_543308

/-- A square is made by arranging 12 marbles of the same size without gaps in both the width and length. -/
def marbles_per_side : ℕ := 12

/-- Each side of the square has 12 marbles and we need to find the marbles around the perimeter without double-counting corners. -/
theorem marbles_around_perimeter (marbles_per_side = 12) : marbles_per_side * 4 - 4 = 44 :=
by
  sorry

end marbles_around_perimeter_l543_543308


namespace number_of_ordered_triples_l543_543049

theorem number_of_ordered_triples :
  ∃ n, (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.lcm a b = 12 ∧ Nat.gcd b c = 6 ∧ Nat.lcm c a = 24) ∧ n = 4 :=
sorry

end number_of_ordered_triples_l543_543049


namespace regular_polygon_circumradius_inradius_inequality_l543_543569

theorem regular_polygon_circumradius_inradius_inequality (n : ℕ) (h : n ≥ 3) (r R : ℝ) 
  (relation : r / R = Real.cos ( Real.pi / n)) : 
  R ≤ 2 * r :=
by 
  have cos_ineq : Real.cos ( Real.pi / n) ≥ 1/2 := sorry,
  have ineq : r / R ≥ 1/2,
  { rw relation,
    exact cos_ineq, },
  linarith [ineq]

end regular_polygon_circumradius_inradius_inequality_l543_543569


namespace find_b10_l543_543943

def seq (b : ℕ → ℕ) :=
  (b 1 = 2)
  ∧ (∀ m n, b (m + n) = b m + b n + 2 * m * n)

theorem find_b10 (b : ℕ → ℕ) (h : seq b) : b 10 = 110 :=
by 
  -- Proof omitted, as requested.
  sorry

end find_b10_l543_543943


namespace cos_double_angle_l543_543383

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l543_543383


namespace trigonometric_identity_l543_543440

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.cos (π / 6 - x) = - Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 :=
by
  sorry

end trigonometric_identity_l543_543440


namespace avg_rate_change_l543_543217

def f (x : ℝ) : ℝ := x^2 + x

theorem avg_rate_change : (f 2 - f 1) / (2 - 1) = 4 := by
  -- here the proof steps should follow
  sorry

end avg_rate_change_l543_543217


namespace a_gives_b_head_start_l543_543703

theorem a_gives_b_head_start (Va Vb L H : ℝ) 
    (h1 : Va = (20 / 19) * Vb)
    (h2 : L / Va = (L - H) / Vb) : 
    H = (1 / 20) * L := sorry

end a_gives_b_head_start_l543_543703


namespace binarySeq_equinum_subsets_l543_543194

open Set

def binarySeq := ℕ →₀ {0, 1}

def toSubset (s : binarySeq) : Set ℕ := {n | s n = 1}

theorem binarySeq_equinum_subsets :
  bijective toSubset :=
by 
  sorry

end binarySeq_equinum_subsets_l543_543194


namespace derivative_y_1_l543_543047

noncomputable def y : ℝ → ℝ := λ x => exp(x) / x

theorem derivative_y_1 : (deriv y) x = exp(x) * (x - 1) / (x ^ 2) := by
  sorry

end derivative_y_1_l543_543047


namespace aftershave_alcohol_concentration_l543_543233

def initial_volume : ℝ := 12
def initial_concentration : ℝ := 0.60
def desired_concentration : ℝ := 0.40
def water_added : ℝ := 6
def final_volume : ℝ := initial_volume + water_added

theorem aftershave_alcohol_concentration :
  initial_concentration * initial_volume = desired_concentration * final_volume :=
by
  sorry

end aftershave_alcohol_concentration_l543_543233


namespace eval_floor_abs_value_l543_543353

def floor_abs_value (x : ℝ) : ℤ := Int.floor (Real.abs x)

theorem eval_floor_abs_value : floor_abs_value (-58.6) = 58 :=
by
  sorry

end eval_floor_abs_value_l543_543353


namespace find_prob_A_l543_543090

variable (P : String → ℝ)
variable (A B : String)

-- Conditions
axiom prob_complement_twice : P B = 2 * P A
axiom prob_sum_to_one : P A + P B = 1

-- Statement to be proved
theorem find_prob_A : P A = 1 / 3 :=
by
  -- Proof to be filled in
  sorry

end find_prob_A_l543_543090


namespace quadratic_factorization_value_of_a_l543_543485

theorem quadratic_factorization_value_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + a = 0 ↔ 2 * (x - 2)^2 = 4) → a = 4 :=
by
  intro h
  sorry

end quadratic_factorization_value_of_a_l543_543485


namespace solve_price_of_pizza_l543_543800

noncomputable def price_of_pizza
  (total_goal : ℝ)
  (amount_needed : ℝ)
  (price_potato_fries : ℝ)
  (price_soda : ℝ)
  (num_pizzas : ℕ)
  (num_potato_fries : ℕ)
  (num_sodas : ℕ)
  (total_revenue : ℝ) : ℝ :=
  let P := (total_revenue - (num_potato_fries * price_potato_fries + num_sodas * price_soda)) / num_pizzas in
  P

theorem solve_price_of_pizza :
  price_of_pizza 500 258 0.30 2 15 40 25 242 = 12 :=
by
  sorry

end solve_price_of_pizza_l543_543800


namespace choose_two_from_four_l543_543159

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Definition of combination
def combination (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem statement
theorem choose_two_from_four : combination 4 2 = 6 :=
by
  -- Proof omitted
  sorry

end choose_two_from_four_l543_543159


namespace num_rose_bushes_approximation_l543_543720

noncomputable def num_rose_bushes (radius spacing : ℝ) : ℝ :=
  2 * real.pi * radius / spacing

theorem num_rose_bushes_approximation :
  num_rose_bushes 15 0.75 ≈ 126 := 
by 
  sorry

end num_rose_bushes_approximation_l543_543720


namespace hundred_squared_plus_two_hundred_one_is_composite_l543_543157

theorem hundred_squared_plus_two_hundred_one_is_composite : 
    ¬ Prime (100^2 + 201) :=
by {
  sorry
}

end hundred_squared_plus_two_hundred_one_is_composite_l543_543157


namespace cos_double_angle_l543_543382

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l543_543382


namespace coloring_not_always_possible_l543_543799

-- Define a convex polyhedron as having faces that are polygons with an even number of sides
structure ConvexPolyhedron where
  faces : List (List ℕ)
  all_faces_even_sides : ∀ f ∈ faces, (f.length % 2 = 0)

-- Define a function that checks if an edge coloring is possible
def isColoringPossible (M : ConvexPolyhedron) (coloring : List (ℕ × ℕ) → Bool) : Prop :=
  ∀ f ∈ M.faces, 
    let edges := f.length / 2 in
    (coloring (List.zip (List.range edges) (List.repeat edges 2)) = true)

-- We want to prove that such a coloring is not always possible
theorem coloring_not_always_possible : ∃ (M : ConvexPolyhedron), ∀ coloring, ¬ isColoringPossible M coloring := sorry

end coloring_not_always_possible_l543_543799


namespace monthly_income_P_l543_543708

theorem monthly_income_P (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : (P + R) / 2 = 5200) :
  P = 4000 := 
sorry

end monthly_income_P_l543_543708


namespace function_relationship_l543_543086

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ {x y}, x ∈ s → y ∈ s → x < y → f y ≤ f x

-- The main statement we want to prove
theorem function_relationship (f : ℝ → ℝ) 
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on f (Set.Ici 0)) :
  f 1 > f (-10) :=
by sorry

end function_relationship_l543_543086


namespace find_a_and_b_l543_543543

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x + 1))

theorem find_a_and_b
  (a b : ℝ)
  (h1 : a < b)
  (h2 : f a = f ((- (b + 1)) / (b + 2)))
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) :
  a = - 2 / 5 ∧ b = - 1 / 3 :=
sorry

end find_a_and_b_l543_543543


namespace T_5_3_l543_543029

def T (x y : ℕ) : ℕ := 4 * x + 5 * y + x * y

theorem T_5_3 : T 5 3 = 50 :=
by
  sorry

end T_5_3_l543_543029


namespace centroid_parallel_BC_l543_543521

-- All necessary definitions and the proof problem without completing the proof itself.

variable (ABC ABC_1 BCA_1 CAB_1 : Type) [triangle : Triangle ABC] 
variable [equilateral_triangle_ABC1 : EquilateralTriangle ABC_1]
variable [equilateral_triangle_BCA1 : EquilateralTriangle BCA_1]
variable [equilateral_triangle_CAB1 : EquilateralTriangle CAB_1]

variable (P : Point) (Q : Point) (R : Point)
variable [hP : IntersectCircumcircles ABC_1 CAB_1 P]
variable [hQ : OnCircumcircleAndParallel CAB_1 P Q BA_1]
variable [hR : OnCircumcircleAndParallel ABC_1 P R CA_1]

theorem centroid_parallel_BC (centroid_ABC : Point) (centroid_PQR : Point) :
  LineConnectingCentroidsParallel centroid_ABC centroid_PQR BC :=
by
  -- Proof would go here
  sorry

end centroid_parallel_BC_l543_543521


namespace squares_nailed_with_2n_minus_2_nails_l543_543560

variable (n : Nat)
variable (rectangular_table : Type)
variable (square : Type)
variable (color : square → Nat)

-- Conditions
axiom rectangular_table_exists : ∃ t : rectangular_table, True
axiom squares_equal_size : ∀ (s1 s2 : square), size s1 = size s2
axiom sides_parallel : ∀ (s : square), sides_parallel_to_table s
axiom n_diff_colors : ∀ s : square, 1 ≤ color s ∧ color s ≤ n
axiom n_squares_intersection : 
  ∀ (squares : List square),
  length squares = n → 
  ∃ (s1 s2 : square), s1 ∈ squares → s2 ∈ squares → s1 ≠ s2 → 
  can_be_nailed_together_with_one_nail s1 s2

-- Statement to be proved
theorem squares_nailed_with_2n_minus_2_nails :
  ∃ (squares : List square), 
  (∀ s : square, color s = k → s ∈ squares) →
  ∃ (positions : List point), length positions ≤ 2*n - 2 ∧ can_be_nailed squares positions :=
sorry

end squares_nailed_with_2n_minus_2_nails_l543_543560


namespace students_taking_history_but_not_statistics_l543_543144

-- Definitions based on conditions
def T : Nat := 150
def H : Nat := 58
def S : Nat := 42
def H_union_S : Nat := 95

-- Statement to prove
theorem students_taking_history_but_not_statistics : H - (H + S - H_union_S) = 53 :=
by
  sorry

end students_taking_history_but_not_statistics_l543_543144


namespace expected_value_replacement_seeds_is_200_l543_543286

noncomputable def germination_prob : ℝ := 0.9
noncomputable def num_seeds_sown : ℕ := 1000
noncomputable def replacement_factor : ℕ := 2

def prob_distribution : distribution := binomial num_seeds_sown (1 - germination_prob)

def expected_num_replacement_seeds : ℝ :=
2 * (num_seeds_sown * (1 - germination_prob))

theorem expected_value_replacement_seeds_is_200 :
  expected_num_replacement_seeds = 200 :=
by
  sorry

end expected_value_replacement_seeds_is_200_l543_543286


namespace circle_intersect_tangent_ratio_l543_543334

open EuclideanGeometry

theorem circle_intersect_tangent_ratio
    (O1 O2 B C A E F H G D : Point)
    (h1 : Circle O1)
    (h2 : Circle O2)
    (hBC_diameter : Diameter O1 B C)
    (h_tangent_C : TangentLine O1 C A)
    (hAB_line : Line A B E)
    (hC_tangent_meets_F : ExtendedLine C E h2 F)
    (hH_on_AF : Segment A F H)
    (hHE_line_intersects_G : ExtendedLine H E O1 G)
    (hBG_line_intersects_the_extended_AC : ExtendedLine B G (ExtendedLine A C) D) :
    \(\frac{\dist A H}{\dist H F} = \frac{\dist A C}{\dist C D}\) :=
sorry

end circle_intersect_tangent_ratio_l543_543334


namespace gof_def_final_composition_l543_543907

variable (x : ℝ)

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := (x - 3) / 2

-- Definition of gof (composition of g and f)
def gof (x : ℝ) : ℝ := g (f x)

-- Hypothesis: gof(x) = x
theorem gof_def (x : ℝ) : gof x = x := by
  unfold gof g f
  rw [mul_add, mul_two, sub_eq_add_neg, add_comm (2 * x), add_right_neg, div_two_self]

-- Main statement to prove
theorem final_composition (x : ℝ) : 
  f (f x) = 4 * x + 9 := by
  sorry

end gof_def_final_composition_l543_543907


namespace probability_multiple_of_3_or_4_l543_543617

theorem probability_multiple_of_3_or_4 :
  let numbers := {n | 1 ≤ n ∧ n ≤ 30},
      multiples_of_3 := {n | n ∈ numbers ∧ n % 3 = 0},
      multiples_of_4 := {n | n ∈ numbers ∧ n % 4 = 0},
      multiples_of_12 := {n | n ∈ numbers ∧ n % 12 = 0},
      favorable_outcomes := multiples_of_3 ∪ multiples_of_4,
      double_counted_outcomes := multiples_of_12,
      total_favorable_outcomes := set.card favorable_outcomes - set.card double_counted_outcomes,
      total_outcomes := set.card numbers in
  total_favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_multiple_of_3_or_4_l543_543617


namespace terminal_side_of_angle_l543_543113

theorem terminal_side_of_angle (α : ℝ) (h1 : Real.Tan α < 0) (h2 : Real.Cos α < 0) : 
  ∃ β : ℝ, β = α ∧ Real.Tan β < 0 ∧ Real.Cos β < 0 ∧ terminal_quadrant β = quadrant.II :=
sorry

end terminal_side_of_angle_l543_543113


namespace inequality_proof_l543_543421

theorem inequality_proof (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b := 
by
  sorry

end inequality_proof_l543_543421


namespace unique_angle_l543_543893

noncomputable def sin_cos_angles (x : ℝ) : Prop :=
  0 ≤ x ∧ x < 360 ∧ sin (x * real.pi / 180) = -0.5 ∧ cos (x * real.pi / 180) = 0.5

theorem unique_angle (x : ℝ) (h : sin_cos_angles x) : x = 330 :=
sorry

end unique_angle_l543_543893


namespace sum_of_medians_is_63_l543_543149

def scores_A := [12, 15, 24, 25, 31, 36, 37, 39, 44, 49, 50]
def scores_B := [13, 14, 16, 23, 26, 27, 28, 33, 38, 39, 51]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem sum_of_medians_is_63 :
  median scores_A + median scores_B = 63 :=
by
  sorry

end sum_of_medians_is_63_l543_543149


namespace part1_part2_l543_543250

noncomputable theory

section Proof

-- Part 1
def proof1 (μ σ : ℝ) : ℝ :=
  let p_outside := 1 - 0.9974
  let X := Binomial 10 p_outside
  1 - (0.9974 ^ 10)

theorem part1 (μ σ : ℝ) : proof1 μ σ = 0.0257 :=
by sorry

-- Part 2
def possible_Y_distribution : pmf ℕ :=
  pmf.of_finite (λ y, match y with
    | 0 => 1 / 42
    | 1 => 5 / 21
    | 2 => 10 / 21
    | 3 => 5 / 21
    | 4 => 1 / 42
    | _ => 0
  end)

def expectation_Y : ℝ :=
  pmf.expected_value possible_Y_distribution (λ y, y)

theorem part2 : expectation_Y = 2 :=
by sorry

end Proof

end part1_part2_l543_543250


namespace findMonthlyIncome_l543_543978

-- Variables and conditions
variable (I : ℝ) -- Raja's monthly income
variable (saving : ℝ) (r1 r2 r3 r4 r5 : ℝ) -- savings and monthly percentages

-- Conditions
def condition1 : r1 = 0.45 := by sorry
def condition2 : r2 = 0.12 := by sorry
def condition3 : r3 = 0.08 := by sorry
def condition4 : r4 = 0.15 := by sorry
def condition5 : r5 = 0.10 := by sorry
def conditionSaving : saving = 5000 := by sorry

-- Define the main equation
def mainEquation (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) : Prop :=
  (r1 * I) + (r2 * I) + (r3 * I) + (r4 * I) + (r5 * I) + saving = I

-- Main theorem to prove
theorem findMonthlyIncome (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) 
  (h1 : r1 = 0.45) (h2 : r2 = 0.12) (h3 : r3 = 0.08) (h4 : r4 = 0.15) (h5 : r5 = 0.10) (hSaving : saving = 5000) :
  mainEquation I r1 r2 r3 r4 r5 saving → I = 50000 :=
  by sorry

end findMonthlyIncome_l543_543978


namespace original_weight_calculation_l543_543700

-- Conditions
variable (postProcessingWeight : ℝ) (originalWeight : ℝ)
variable (lostPercentage : ℝ)

-- Problem Statement
theorem original_weight_calculation
  (h1 : postProcessingWeight = 240)
  (h2 : lostPercentage = 0.40) :
  originalWeight = 400 :=
sorry

end original_weight_calculation_l543_543700


namespace fraction_zero_implies_x_eq_two_l543_543488

theorem fraction_zero_implies_x_eq_two (x : ℝ) (h : (x^2 - 4) / (x + 2) = 0) : x = 2 :=
sorry

end fraction_zero_implies_x_eq_two_l543_543488


namespace problem_lean_statement_l543_543933

theorem problem_lean_statement : 
  let jo_sum := (120 * 121) / 2,
      kate_sum := 12 * (25 * (240 / 2)),
      difference := abs(kate_sum - jo_sum)
  in difference = 28740 :=
by
  let jo_sum := (120 * 121) / 2
  let kate_sum := 12 * (25 * (240 / 2))
  let difference := abs(kate_sum - jo_sum)
  have h1 : jo_sum = 7260 := by calc
    jo_sum = (120 * 121) / 2 : rfl
    ... = 7260 : by norm_num,
  have h2 : kate_sum = 36000 := by calc
    kate_sum = 12 * (25 * 120) : by rw [mul_div_cancel 240 2, mul_comm 25 240]
    ... = 36000 : by norm_num,
  have h3 : difference = abs(36000 - 7260) := by rw [h1,h2],
  have h4 : abs(36000 - 7260) = 28740 := by norm_num,
  exact h3.trans h4

end problem_lean_statement_l543_543933


namespace equal_diagonals_n_gon_l543_543906

theorem equal_diagonals_n_gon (n : ℕ) (F : Type) [polygon F n] (h1 : 4 ≤ n) (h2 : ∀ d1 d2 : diagonal F, d1 = d2) :
  n = 4 ∨ n = 5 := 
sorry

end equal_diagonals_n_gon_l543_543906


namespace simplify_problem_1_simplify_problem_2_l543_543580

-- Problem 1: Statement of Simplification Proof
theorem simplify_problem_1 :
  (- (99 + (71 / 72)) * 36 = - (3599 + 1 / 2)) :=
by sorry

-- Problem 2: Statement of Simplification Proof
theorem simplify_problem_2 :
  (-3 * (1 / 4) - 2.5 * (-2.45) + (7 / 2) * (1 / 4) = 6 + 1 / 4) :=
by sorry

end simplify_problem_1_simplify_problem_2_l543_543580


namespace parallel_tangents_a3_plus_b2_plus_d_eq_seven_l543_543071

theorem parallel_tangents_a3_plus_b2_plus_d_eq_seven:
  ∃ (a b d : ℝ),
  (1, 1).snd = a * (1:ℝ)^3 + b * (1:ℝ)^2 + d ∧
  (-1, -3).snd = a * (-1:ℝ)^3 + b * (-1:ℝ)^2 + d ∧
  (3 * a * (1:ℝ)^2 + 2 * b * 1 = 3 * a * (-1:ℝ)^2 + 2 * b * -1) ∧
  a^3 + b^2 + d = 7 := 
sorry

end parallel_tangents_a3_plus_b2_plus_d_eq_seven_l543_543071


namespace right_triangle_side_length_l543_543230

theorem right_triangle_side_length (c a b : ℕ) (hc : c = 13) (ha : a = 12) (hypotenuse_eq : c ^ 2 = a ^ 2 + b ^ 2) : b = 5 :=
sorry

end right_triangle_side_length_l543_543230


namespace range_of_k_l543_543097

noncomputable theory

open Real

variables (k x : ℝ)

def quadratic_equation (k x : ℝ) : ℝ :=
  x^2 - (2*k + (1/2)*k^2)*x + k^2

def discriminant (k : ℝ) : ℝ :=
  (2*k + (1/2)*k^2)^2 - 4 * k^2

def has_unequal_real_roots (k : ℝ) : Prop :=
  k > 0 ∧ discriminant k > 0

def in_interval (x k : ℝ) : Prop :=
  k - 1 ≤ x ∧ x ≤ k + 1

theorem range_of_k (h : ∃ x, |x - k| = (sqrt 2 / 2) * k * sqrt x ∧ in_interval x k ∧ has_unequal_real_roots k) :
  0 < k ∧ k ≤ 1 :=
sorry

end range_of_k_l543_543097


namespace fernanda_savings_before_payments_l543_543006

open Real

theorem fernanda_savings_before_payments (aryan_debt kyro_debt aryan_payment kyro_payment total_savings before_savings : ℝ) 
  (h1: aryan_debt = 1200)
  (h2: aryan_debt = 2 * kyro_debt)
  (h3: aryan_payment = 0.6 * aryan_debt)
  (h4: kyro_payment = 0.8 * kyro_debt)
  (h5: total_savings = before_savings + aryan_payment + kyro_payment)
  (h6: total_savings = 1500) :
  before_savings = 300 :=
by
  sorry

end fernanda_savings_before_payments_l543_543006


namespace tan_alpha_minus_beta_l543_543843

theorem tan_alpha_minus_beta
  (α β : ℝ)
  (tan_alpha : Real.tan α = 2)
  (tan_beta : Real.tan β = -7) :
  Real.tan (α - β) = -9 / 13 :=
by sorry

end tan_alpha_minus_beta_l543_543843


namespace shortest_distance_ln_curve_l543_543239

open Real

theorem shortest_distance_ln_curve (x : ℝ) (h : x > 0) :
  let curve := λ x, log x in
  let line := λ x, x + 2 in
  (∀ x, x > 0 → dist (curve x) (line x) ≥ dist 0 (line 1)) ∧
  dist 0 (line 1) = 3 * sqrt 2 / 2
:= sorry

end shortest_distance_ln_curve_l543_543239


namespace phil_final_coins_l543_543971

-- Define the initial number of coins and computing the subsequent counts year by year
def initial_coins : ℕ := 1000
def coins_after_year_1 : ℕ := initial_coins * 4
def coins_after_year_2 : ℕ := coins_after_year_1 + 7 * 52
def coins_after_year_3 : ℕ := coins_after_year_2 + 3 * 182
def coins_after_year_4 : ℕ := coins_after_year_3 + 2 * 52
def coins_after_year_5 : ℕ := coins_after_year_4 - Nat.floor ((coins_after_year_4 : ℝ) * 0.4)
def coins_after_year_6 : ℕ := coins_after_year_5 + 5 * 91
def coins_after_year_7 : ℕ := coins_after_year_6 + 20 * 12
def coins_after_year_8 : ℕ := coins_after_year_7 + 10 * 52
def coins_after_year_9 : ℕ := coins_after_year_8 - Nat.floor (coins_after_year_8 / 3)

theorem phil_final_coins :
  coins_after_year_9 = 2816 := by
  sorry

end phil_final_coins_l543_543971


namespace exists_triangle_l543_543027

-- Definitions for altitude, median, and circumradius
variables (h_a m_a R : ℝ)

-- Assumption that median is greater than altitude
axiom ma_gt_ha : m_a > h_a

-- Definition of a triangle and its properties
structure Triangle := 
  (A B C : ℝ × ℝ)

def altitude (ABC : Triangle) : ℝ :=
  let (A, B, C) := (ABC.A, ABC.B, ABC.C) in
  -- logic to compute the altitude

def median (ABC : Triangle) : ℝ :=
  let (A, B, C) := (ABC.A, ABC.B, ABC.C) in
  -- logic to compute the median

def circumradius (ABC : Triangle) : ℝ :=
  let (A, B, C) := (ABC.A, ABC.B, ABC.C) in
  -- logic to compute the circumradius

-- Proposition to show the existence of the triangle satisfying given conditions
theorem exists_triangle (h_a m_a R : ℝ) (ma_gt_ha : m_a > h_a) : 
  ∃ (ABC : Triangle), altitude ABC = h_a ∧ median ABC = m_a ∧ circumradius ABC = R :=
sorry

end exists_triangle_l543_543027


namespace coefficient_of_x_in_exponent_is_3_l543_543480

theorem coefficient_of_x_in_exponent_is_3
  (some_number : ℝ)
  (h : ∀ x : ℝ, 4 ^ (2 * x + 2) = 16 ^ (some_number * x - 1))
  (hx : ∀ x : ℝ, x = 1) :
  some_number = 3 :=
by
  -- The statement is skipped with sorry since no actual proof steps are to be provided.
  sorry

end coefficient_of_x_in_exponent_is_3_l543_543480


namespace microorganism_colored_area_correct_l543_543197

theorem microorganism_colored_area_correct :
  let a := 30
  let b := 40
  let c := 50
  let speed := 1 / 6
  let movement_time := 60   -- time in seconds
  let radius := 10          -- distance microorganisms can travel
  let area_triangle := 1 / 2 * a * b -- area of the right triangle
  let perimeter := a + b + c -- perimeter of the triangle
  let area_strips := radius * perimeter -- area covered by strips along the sides
  let area_sectors := 100 * Real.pi -- combined area of circular sectors
  let total_area := area_triangle + area_strips + area_sectors -- total covered area

  total_area ≈ 2114 := by
  let a := 30
  let b := 40
  let c := 50
  let speed := 1 / 6
  let movement_time := 60
  let distance := speed * movement_time
  have r : radius = distance := by sorry

  have h1 : a^2 + b^2 = c^2 := by sorry
  have area_triangle_correct : 1 / 2 * a * b = 600 := by sorry
  have perimeter_correct : a + b + c = 120 := by sorry
  have strips_correct : radius * 120 = 1200 := by sorry
  have sectors_correct : 100 * Real.pi ≈ 314 := by sorry
  have total_area_correct : 600 + 1200 + 314 = 2114 := by sorry

  rfl -- the final proof step

end microorganism_colored_area_correct_l543_543197


namespace number_of_girls_l543_543143

-- Defining the problem conditions
variables (B G : ℕ)
axioms 
  (h1 : B = 8 * (G / 5))
  (h2 : B + G = 520)

-- Stating the goal
theorem number_of_girls : G = 200 :=
by sorry

end number_of_girls_l543_543143


namespace arithmetic_sequence_properties_l543_543179

noncomputable def S_n (n : ℕ) (a_n : ℕ → ℝ) (c : ℝ) : ℝ := (1/3) * n * a_n n + a_n n - c

def a_seq : ℕ → ℝ
| 0     := 0
| 1     := 3 -- a_1 = 3 calculated from initial problem
| (n+1) := a_seq n + 3 -- From a_n = 3n, we have the pattern

def general_term_formula : ℕ → ℝ := λ n, 3 * n

theorem arithmetic_sequence_properties (c : ℝ) (hS : ∀ n : ℕ, S_n n a_seq c = if (n = 0) then 0 else S_n n a_seq c) :
  (∀ n : ℕ, a_seq n = 3 * n) ∧
  (∀ n : ℕ, ∑ k in finset.range n, 1 / (a_seq k * a_seq (k+1)) < 1/9)
:= sorry

end arithmetic_sequence_properties_l543_543179


namespace compute_g_1986_l543_543948

noncomputable def g : ℕ → ℤ
| 0     := 0
| 1     := 2
| (n+1) := 2 - 2 * g n

theorem compute_g_1986 : g 1986 = -2 := sorry

end compute_g_1986_l543_543948


namespace coefficient_of_term_l543_543221

-- Define the term
def term : ℝ := - (2 * x^3 * y * z^2) / 5

-- State the theorem
theorem coefficient_of_term (x y z : ℝ) : 
  term = (-2 / 5) * x^3 * y * z^2 := 
sorry

end coefficient_of_term_l543_543221


namespace area_of_quadrilateral_l543_543841
noncomputable def c := sqrt (16 - 4) -- √12 = 2√3

theorem area_of_quadrilateral (a b : ℝ) (P Q F1 F2 : ℝ×ℝ) :
  let e : set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} in
  let ellipse_symmetric := (P ∈ e) ∧ (Q ∈ e) ∧ (P.1 = -Q.1) ∧ (P.2 = -Q.2) ∧ dist (P, Q) = 2 * c in
  let F1F2 := 2 * c in
  let mn := 8 in
  (dist (P, F1) + dist (P, F2) = 2 * a) →
  (dist (P, F1)^2 + dist (P, F2)^2 = F1F2^2 / 4 * (a^2 - b^2)) →
  (mn = 8) →
  area_of_quadrilateral P F1 Q F2 = 8 :=
by
  sorry

end area_of_quadrilateral_l543_543841


namespace smallest_positive_period_of_f_l543_543347

noncomputable def f (x : ℝ) : ℝ := 1 - 3 * Real.sin (x + Real.pi / 4) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x :=
by
  intros
  -- Proof is omitted
  sorry

end smallest_positive_period_of_f_l543_543347


namespace domain_of_reciprocal_shifted_function_l543_543595

def domain_of_function (x : ℝ) : Prop :=
  x ≠ 1

theorem domain_of_reciprocal_shifted_function : 
  ∀ x : ℝ, (∃ y : ℝ, y = 1 / (x - 1)) ↔ domain_of_function x :=
by 
  sorry

end domain_of_reciprocal_shifted_function_l543_543595


namespace purchase_costs_max_balls_l543_543285

def soccer_and_basketball_costs (x y : ℕ) : Prop :=
  4 * x + 7 * y = 740 ∧ 7 * x + 5 * y = 860 ∧ x = 80 ∧ y = 60

def max_soccer_balls (m : ℕ) : Prop :=
  ∀ m : ℕ, 80 * m + 60 * (50 - m) ≤ 3600 → m ≤ 30

theorem purchase_costs : ∃ x y : ℕ, soccer_and_basketball_costs x y := by
  use 80, 60
  split
  sorry -- Proof of 4 * 80 + 7 * 60 = 740
  split
  sorry -- Proof of 7 * 80 + 5 * 60 = 860
  split
  refl
  refl

theorem max_balls : ∃ m : ℕ, max_soccer_balls m := by
  use 30
  intros n h
  sorry -- Proof that for any n, 80 * n + 60 * (50 - n) ≤ 3600 implies n ≤ 30

end purchase_costs_max_balls_l543_543285


namespace correct_statements_in_given_problem_l543_543692

theorem correct_statements_in_given_problem :
  (let A := {1, 2} in 
   let B := {(1, 2)} in 
   A ≠ B /\
   let f : ℝ → ℝ := λ x, sqrt (3 - 2*x - x^2) in 
   ((∀ x : ℝ, x ∈ Icc (-3) (-1) → 
     ∀ y : ℝ, y ∈ Icc (-3) (-1) → x ≤ y → f x ≤ f y) /\
   (∀ a b : ℝ, log x (18) 9 = a → 18^b = 5 → log (36) (45) = (a + b) / (2 - a)) /\
   (∀ (f : ℝ → ℝ), 
     (∀ x : ℝ, x > 0 → f x = 2 * x^2 + 1 / x - 1) → 
     ∀ x : ℝ, x < 0 → f x = - (2 * x^2 + 1 / x - 1))) :=
  true
    sorry

end correct_statements_in_given_problem_l543_543692


namespace solution_set_of_inequality_l543_543476

theorem solution_set_of_inequality (a : ℝ) (a_gt_1 : a > 1) :
  {x : ℝ | |x| + a > 1} = set.univ := by
  sorry

end solution_set_of_inequality_l543_543476


namespace th150th_letter_is_B_l543_543659

def pattern := "ABCD".data

def nth_letter_in_pattern (n : ℕ) : Char :=
  let len := pattern.length
  pattern.get n % len

theorem th150th_letter_is_B :
  nth_letter_in_pattern 150 = 'B' :=
by {
  -- This proof is placed here as a placeholder
  sorry
}

end th150th_letter_is_B_l543_543659


namespace totalLemonProductionIn5Years_l543_543736

-- Definition of a normal lemon tree's production rate
def normalLemonProduction : ℕ := 60

-- Definition of the percentage increase for Jim's lemon trees (50%)
def percentageIncrease : ℕ := 50

-- Calculate Jim's lemon tree production per year
def jimLemonProduction : ℕ := normalLemonProduction * (100 + percentageIncrease) / 100

-- Calculate the total number of trees in Jim's grove
def treesInGrove : ℕ := 50 * 30

-- Calculate the total lemon production by Jim's grove in one year
def annualLemonProduction : ℕ := treesInGrove * jimLemonProduction

-- Calculate the total lemon production by Jim's grove in 5 years
def fiveYearLemonProduction : ℕ := 5 * annualLemonProduction

-- Theorem: Prove that the total lemon production in 5 years is 675000
theorem totalLemonProductionIn5Years : fiveYearLemonProduction = 675000 := by
  -- Proof needs to be filled in
  sorry

end totalLemonProductionIn5Years_l543_543736


namespace extreme_values_range_of_a_l543_543880

noncomputable def f (x : ℝ) := x^2 * Real.exp x
noncomputable def y (x : ℝ) (a : ℝ) := f x - a * x

theorem extreme_values :
  ∃ x_max x_min,
    (x_max = -2 ∧ f x_max = 4 / Real.exp 2) ∧
    (x_min = 0 ∧ f x_min = 0) := sorry

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y x₁ a = 0 ∧ y x₂ a = 0) ↔
  -1 / Real.exp 1 < a ∧ a < 0 := sorry

end extreme_values_range_of_a_l543_543880


namespace probability_of_not_first_class_product_l543_543979

theorem probability_of_not_first_class_product
  (P_A : ℝ) (hP_A : P_A = 0.7) :
  let P_not_A := 1 - P_A in
  P_not_A = 0.3 :=
by
  sorry

end probability_of_not_first_class_product_l543_543979


namespace closest_to_2010_l543_543754

theorem closest_to_2010 :
  let A := 2008 * 2012
  let B := 1000 * Real.pi
  let C := 58 * 42
  let D := (48.3 ^ 2 - 2 * 8.3 * 48.3 + 8.3 ^ 2)
  abs (2010 - D) < abs (2010 - A) ∧
  abs (2010 - D) < abs (2010 - B) ∧
  abs (2010 - D) < abs (2010 - C) :=
by
  sorry

end closest_to_2010_l543_543754


namespace product_difference_is_multiple_of_2001_l543_543976

-- Define product of the first 1000 positive even integers
def product_even_1000 : ℕ := ∏ i in (Finset.range 1000).map (λ x, 2 * (x + 1)), id

-- Define product of the first 1000 positive odd integers
def product_odd_1000 : ℕ := ∏ i in (Finset.range 1000).map (λ x, 2 * (x + 1) - 1), id

-- Prove that the difference of the two products is a multiple of 2001
theorem product_difference_is_multiple_of_2001 : (product_even_1000 - product_odd_1000) % 2001 = 0 :=
by sorry

end product_difference_is_multiple_of_2001_l543_543976


namespace totalLemonProductionIn5Years_l543_543737

-- Definition of a normal lemon tree's production rate
def normalLemonProduction : ℕ := 60

-- Definition of the percentage increase for Jim's lemon trees (50%)
def percentageIncrease : ℕ := 50

-- Calculate Jim's lemon tree production per year
def jimLemonProduction : ℕ := normalLemonProduction * (100 + percentageIncrease) / 100

-- Calculate the total number of trees in Jim's grove
def treesInGrove : ℕ := 50 * 30

-- Calculate the total lemon production by Jim's grove in one year
def annualLemonProduction : ℕ := treesInGrove * jimLemonProduction

-- Calculate the total lemon production by Jim's grove in 5 years
def fiveYearLemonProduction : ℕ := 5 * annualLemonProduction

-- Theorem: Prove that the total lemon production in 5 years is 675000
theorem totalLemonProductionIn5Years : fiveYearLemonProduction = 675000 := by
  -- Proof needs to be filled in
  sorry

end totalLemonProductionIn5Years_l543_543737


namespace ratio_of_red_to_blue_beads_l543_543329

theorem ratio_of_red_to_blue_beads (red_beads blue_beads : ℕ) (h1 : red_beads = 30) (h2 : blue_beads = 20) :
    (red_beads / Nat.gcd red_beads blue_beads) = 3 ∧ (blue_beads / Nat.gcd red_beads blue_beads) = 2 := 
by 
    -- Proof will go here
    sorry

end ratio_of_red_to_blue_beads_l543_543329


namespace area_of_quadrilateral_PF1QF2_l543_543836

theorem area_of_quadrilateral_PF1QF2 (x y : ℝ) (F1 F2 P Q : ℝ×ℝ) 
  (h1 : ∀ p : ℝ×ℝ, p ∈ set_of (λ q, q.1^2/16 + q.2^2/4 = 1))
  (h2 : F1 = (4, 0) ∧ F2 = (-4, 0)) 
  (h3 : Q = (-P.1, -P.2))
  (h4 : dist P Q = dist F1 F2) :
  let a := 8 in
  let c := 4 in
  let b_sq := a^2 - c^2 in
  let m := |dist P F1| in
  let n := |dist P F2| in
  m * n = 8 :=
by sorry

end area_of_quadrilateral_PF1QF2_l543_543836


namespace wheat_harvest_prob_l543_543140

theorem wheat_harvest_prob :
  let P (A1 A2 B : Prop) := ℝ,
      P_A1 := 0.8,
      P_A2 := 0.2,
      P_B_given_A1 := 0.6,
      P_B_given_A2 := 0.2 in
  P (B) = P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 → P (B) = 0.52 :=
by
  sorry

end wheat_harvest_prob_l543_543140


namespace inequality_proof_l543_543904

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (2 * x) + 1 / (2 * y) + 1 / (2 * z)) > 
  (1 / (y + z) + 1 / (z + x) + 1 / (x + y)) :=
  by
    let a := y + z
    let b := z + x
    let c := x + y
    have x_def : x = (a + c - b) / 2 := sorry
    have y_def : y = (a + b - c) / 2 := sorry
    have z_def : z = (b + c - a) / 2 := sorry
    sorry

end inequality_proof_l543_543904


namespace compute_h_of_2_l543_543953

def f (x : ℝ) : ℝ := 2 * x^2 + 5
def g (x : ℝ) : ℝ := Real.sqrt (f x + 3) - 2
def h (x : ℝ) : ℝ := f (g x)

theorem compute_h_of_2 : h 2 = 13 := by
  sorry

end compute_h_of_2_l543_543953


namespace gcd_of_XY_is_6_l543_543483

theorem gcd_of_XY_is_6 (X Y : ℕ) (h1 : Nat.lcm X Y = 180)
  (h2 : X * 6 = Y * 5) : Nat.gcd X Y = 6 :=
sorry

end gcd_of_XY_is_6_l543_543483


namespace range_of_f_l543_543818

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 - 2 * x + 2)

theorem range_of_f : set.range f = {y : ℝ | -1/2 ≤ y} :=
by
  sorry

end range_of_f_l543_543818


namespace cos_angle_PQR_l543_543922

/-- Definition of the angles in tetrahedron PQRS with given conditions and the goal to prove. -/
theorem cos_angle_PQR
  (PQ PR PS QR RS SQ : ℝ)
  (a b α : ℝ)
  (h1 : ∠ PSR = 90)
  (h2 : ∠ PRQ = 90)
  (h3 : ∠ QRS = 90)
  (ha : a = sin (∠ PQS))
  (hb : b = sin (∠ PRS))
  (hPR : PR = 2 * PQ)
  (hα : α = cos (∠ PSQ)) :
  cos (∠ PQR) = 2 * b^2 * α := 
sorry

end cos_angle_PQR_l543_543922


namespace trajectory_equation_line_equation_l543_543091

theorem trajectory_equation (x y : ℝ) : (let k_PA := y / (x + real.sqrt 2),
                                             k_PB := y / (x - real.sqrt 2)
                                         in k_PA * k_PB = -1/2) → (x^2 / 2 + y^2 = 1) :=
by
  intro h
  sorry

theorem line_equation (k : ℝ) : (∀ x y: ℝ, (x * 2 + real.sqrt 2 = 0) → (k * x + 1 = y) →
    (x^2 / 2 + y^2 = 1) → (abs(MN) = 4 * real.sqrt(2) / 3)) →
    (k = 1 ∨ k = -1) :=
by
  intro h
  sorry

end trajectory_equation_line_equation_l543_543091


namespace incorrect_propositions_l543_543246

/--
Given the following propositions:
1. Three points determine a plane.
2. A rectangle is a plane figure.
3. Three lines intersecting in pairs determine a plane.
4. Two intersecting planes divide the space into four regions.

Prove that propositions 1 and 3 are incorrect.
-/
theorem incorrect_propositions :
  (∃ x: Prop, x = "Three points determine a plane" ∧ x = "incorrect") ∧
  (∃ y: Prop, y = "Three lines intersecting in pairs determine a plane" ∧ y = "incorrect") :=
by
  sorry

end incorrect_propositions_l543_543246


namespace duration_approx_231_l543_543360

noncomputable def proof_duration : ℝ :=
    let A : ℝ := 1120
    let P : ℝ := 979.0209790209791
    let r : ℝ := 0.06
    let expected_t : ℝ := 2.31 
    let t : ℝ := Real.log (A / P) / Real.log (1 + r)
    t

theorem duration_approx_231 : Real.abs (proof_duration - 2.31) < 0.01 := by
  -- Statement that the real duration is approximately 2.31
  sorry

end duration_approx_231_l543_543360


namespace find_150th_letter_in_pattern_l543_543665

theorem find_150th_letter_in_pattern : 
  (let sequence := "ABCD";
   sequence.length = 4 → 
   sequence[(150 % 4)] = 'B') :=
by
  sorry

end find_150th_letter_in_pattern_l543_543665


namespace cos_double_angle_l543_543393

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l543_543393


namespace cosine_angle_between_a_and_2a_minus_b_l543_543862

noncomputable def vector_a : EuclideanSpace ℝ (Fin 3) := sorry
noncomputable def vector_b : EuclideanSpace ℝ (Fin 3) := sorry

def norm_eq (v : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry

axiom non_zero_vec_a : vector_a ≠ 0
axiom non_zero_vec_b : vector_b ≠ 0
axiom norm_a_eq_b_and_a_plus_b : norm_eq vector_a = norm_eq vector_b ∧ norm_eq vector_b = norm_eq (vector_a + vector_b)

theorem cosine_angle_between_a_and_2a_minus_b :
  ∀ {a b : EuclideanSpace ℝ (Fin 3)}, 
  vector_a ≠ 0 → vector_b ≠ 0 →
  norm_eq vector_a = norm_eq vector_b ∧ norm_eq vector_b = norm_eq (vector_a + vector_b) →
  let angle_cosine := (inner_product vector_a (2 • vector_a - vector_b) / (norm_eq vector_a * norm_eq (2 • vector_a - vector_b))) 
  in angle_cosine = 5 * sqrt 7 / 14 :=
sorry

end cosine_angle_between_a_and_2a_minus_b_l543_543862


namespace distance_between_planes_l543_543030

open Real

theorem distance_between_planes :
  let n := ⟨2, 4, -2⟩
  let p1 := (2, 4, -2, 10)
  let p2 := (1, 2, -1, -3)
  (parallel_planes p1 p2 n) →
  distance_between_planes p1 p2 = (sqrt 6) / 6 :=
by
  intros
  sorry

end distance_between_planes_l543_543030


namespace monotonic_intervals_lambda_range_l543_543109

noncomputable theory

/-
 Statement (I):
-/
def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + log x

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  if a = 1 / 2 then ∀ x, 0 < x → 0 < f' a x
  else if a > 1 / 2 then ((∀ x, 0 < x ∧ x < 1 / (2 * a) → 0 < f' a x) ∧ (∀ x, 1 < x → 0 < f' a x) ∧ (∀ x, (1 / (2 * a)) < x ∧ x < 1 → f' a x ≤ 0))
  else 0 < a ∧ a < 1 / 2 → ((∀ x, 0 < x ∧ x < 1 → 0 < f' a x) ∧ (∀ x, (1 / (2 * a)) < x → 0 < f' a x) ∧ (∀ x, 1 < x ∧ x < (1 / (2 * a)) → f' a x ≤ 0)) :=
sorry

/-
 Statement (II):
-/
def g (a x : ℝ) : ℝ := a * x^2 - x + log x

theorem lambda_range (a lambda : ℝ) (h1 : 0 < a ∧ a < 1 / 4)
  (h2 : ∀ (x1 x2 : ℝ), x1 ≠ x2 ∧ (2 * a * x1^2 - x1 + 1 = 0 ∧ 2 * a * x2^2 - x2 + 1 = 0) → g a x1 + g a x2 < lambda * (x1 + x2)) :
  λ > 1 / exp 2 - 1 / 2 :=
sorry

end monotonic_intervals_lambda_range_l543_543109


namespace basketball_shot_probability_l543_543278

-- Define the conditions related to the problem
def prob_success_shot : ℚ := 2 / 3
def num_attempts : ℕ := 3
def num_success_shots : ℕ := 2

-- Use the conditions to express the probability of exactly 2 successful shots out of 3 attempts
theorem basketball_shot_probability :
  (nat.choose num_attempts num_success_shots * prob_success_shot ^ num_success_shots * (1 - prob_success_shot) ^ (num_attempts - num_success_shots) = 4 / 9) :=
by
  sorry

end basketball_shot_probability_l543_543278


namespace cos_alpha_l543_543087

theorem cos_alpha (x y : ℝ) (h_coord : (x, y) = (1, 2)) :
  let hypotenuse := real.sqrt (x^2 + y^2) in
  let cos_alpha := x / hypotenuse in
  cos_alpha = (real.sqrt 5) / 5 :=
by
  let hypotenuse := real.sqrt (x^2 + y^2)
  let cos_alpha := x / hypotenuse
  have h1 : hypotenuse = real.sqrt 5, by
    -- Proof of the hypotenuse calculation
    sorry
  have h2 : cos_alpha = 1 / (real.sqrt 5), by
    -- Proof of cos_alpha calculation
    sorry
  have h3 : (1 : ℝ) / (real.sqrt 5) = (real.sqrt 5) / 5, by
    -- Rationalizing the denominator
    sorry
  exact eq.trans h2 h3

end cos_alpha_l543_543087


namespace probability_red_or_blue_l543_543259

theorem probability_red_or_blue :
  let red := 5
  let blue := 3
  let green := 4
  let yellow := 6
  let total := red + blue + green + yellow
  let favorable := red + blue
  let prob := (favorable : ℚ) / total
  abs ((prob - 0.4444) : ℚ) < 0.001 :=
by
  let red := 5
  let blue := 3
  let green := 4
  let yellow := 6
  let total := red + blue + green + yellow
  let favorable := red + blue
  let prob := (favorable : ℚ) / total
  have : (prob - 0.4444).abs < 0.001 := sorry
  exact this

end probability_red_or_blue_l543_543259


namespace no_same_meaning_l543_543923

-- Define the alphabet of the tribe UYU
inductive UYU : Type
| У : UYU
| Ы : UYU

open UYU

-- Define what it means for words to be equivalent
def equiv_word (w1 w2 : list UYU) : Prop :=
  -- The equivalence relation based on the conditions: (put it simply for now)
  sorry

-- Example words
def word1 : list UYU := [У, Ы, Ы]
def word2 : list UYU := [Ы, У, У]

-- Prove that word1 and word2 do not have the same meaning
theorem no_same_meaning : ¬ equiv_word word1 word2 :=
sorry

end no_same_meaning_l543_543923


namespace exist_x_y_l543_543790

theorem exist_x_y (b : ℝ) :
  (∃ x y : ℝ, x * y = b^(2 * b) ∧ ∀ a b, log b (x^(log b y) * y^(log b x)) = 5 * b^5) ↔
  (0 < b ∧ b <= (2/5)^(1/3)) :=
by sorry

end exist_x_y_l543_543790


namespace correct_number_of_propositions_l543_543946

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_eq {x y : ℝ} (hx : -1 < x) (hy : x < 1) (hx' : -1 < y) (hy' : y < 1) : 
  f x - f y = f ((x - y) / (1 - x * y))

axiom f_pos {x : ℝ} (hx : -1 < x) (hx' : x < 0) : 0 < f x

def proposition_1 : Prop := f 0 = 0
def proposition_2 : Prop := ∀ x : ℝ, x ∈ (-1, 1) → f x = f (-x)
def proposition_3 : Prop := ∃! x : ℝ, x ∈ (-1, 1) ∧ f x = 0
def proposition_4 : Prop := f (1 / 2) + f (1 / 3) < f (1 / 4)

def number_of_true_propositions : ℕ :=
  [proposition_1, proposition_2, proposition_3, proposition_4].count (λ p, p)

theorem correct_number_of_propositions : number_of_true_propositions = 3 :=
sorry

end correct_number_of_propositions_l543_543946


namespace num_two_digit_palindromes_l543_543473

theorem num_two_digit_palindromes : 
  let is_palindrome (n : ℕ) : Prop := (n / 10) = (n % 10)
  ∃ n : ℕ, 10 ≤ n ∧ n < 90 ∧ is_palindrome n →
  ∃ count : ℕ, count = 9 := 
sorry

end num_two_digit_palindromes_l543_543473


namespace minimum_cost_solution_l543_543289

def Tank := {capacity: ℕ, cost: ℕ, available: ℕ}

def oil_problem (A B C : Tank) (max_tanks total_oil : ℕ) : Prop :=
  ∃ (xA xB xC : ℕ),
    xA + xB + xC ≤ max_tanks ∧
    xA ≤ A.available ∧
    xB ≤ B.available ∧
    xC ≤ C.available ∧
    xA * A.capacity + xB * B.capacity + xC * C.capacity = total_oil ∧
    xA * A.cost + xB * B.cost + xC * C.cost = 1540 ∧
    (xA = 9 ∧ xB = 8 ∧ xC = 0)

def tank_A : Tank := {capacity := 50, cost := 100, available := 10}
def tank_B : Tank := {capacity := 35, cost := 80, available := 15}
def tank_C : Tank := {capacity := 20, cost := 60, available := 20}

theorem minimum_cost_solution :
  oil_problem tank_A tank_B tank_C 20 728 :=
by sorry

end minimum_cost_solution_l543_543289


namespace tangent_sum_identity_l543_543873

noncomputable def arccos (x : ℝ) : ℝ := sorry
noncomputable def tg (x : ℝ) : ℝ := sorry

theorem tangent_sum_identity (a b : ℝ) (h_b_nonzero : b ≠ 0) (h_valid_input : -1 ≤ 2 * a / b ∧ 2 * a / b ≤ 1) : 
    let x := 7 * Real.pi / 4 in
    let y := (1 / 2) * arccos (2 * a / b) in 
    tg (x + y) + tg (x - y) = -b / a :=
by
  let α := arccos (2 * a / b)
  have hα : y = α / 2 := sorry
  have hx : x = 7 * Real.pi / 4 := sorry
  have hy : y = α / 2 := sorry
  sorry

end tangent_sum_identity_l543_543873


namespace find_p_q_l543_543173

variables (p q : ℝ)

def complex_conjugate_roots (a b c : ℂ) : Prop :=
  ∃ x y : ℝ, a = 1 ∧ b = -(x + y * complex.I) ∧ c = x * x + y * y

theorem find_p_q :
  complex_conjugate_roots 1 (6 + complex.I * p) (13 + complex.I * q) →
  (p = 0 ∧ q = 0) :=
begin
  sorry
end

end find_p_q_l543_543173


namespace circular_permutations_count_l543_543220

noncomputable def mobius (n : ℕ) : ℤ := 
  if n = 1 then 1 
  else if ∃ p : ℕ, p.prime ∧ p * p ∣ n then 0 
  else (-1)^(Finset.card (Finset.filter Nat.prime (Finset.powerset (Finset.range (n+1)))))

noncomputable def circular_permutations (n r : ℕ) : ℕ :=
  (1 / n : ℚ) * (∑ d in Finset.divisors n, mobius d * r ^ (n / d))

theorem circular_permutations_count (n r : ℕ) : 
  circular_permutations n r = (1 / n : ℚ) * (∑ d in Finset.divisors n, mobius d * r ^ (n / d)) :=
sorry

end circular_permutations_count_l543_543220


namespace number_of_adults_l543_543007

theorem number_of_adults (total_bill : ℕ) (cost_per_meal : ℕ) (num_children : ℕ) (total_cost_children : ℕ) 
  (remaining_cost_for_adults : ℕ) (num_adults : ℕ) 
  (H1 : total_bill = 56)
  (H2 : cost_per_meal = 8)
  (H3 : num_children = 5)
  (H4 : total_cost_children = num_children * cost_per_meal)
  (H5 : remaining_cost_for_adults = total_bill - total_cost_children)
  (H6 : num_adults = remaining_cost_for_adults / cost_per_meal) :
  num_adults = 2 :=
by
  sorry

end number_of_adults_l543_543007


namespace count_arithmetic_three_digit_numbers_l543_543929

def is_arithmetic_three_digit_number (n : Nat) : Prop :=
  ∃ (a b c : Nat), n = 100 * a + 10 * b + c ∧
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  b - a = c - b

theorem count_arithmetic_three_digit_numbers : 
  { n : Nat // is_arithmetic_three_digit_number n }.toFinset.card = 45 :=
by 
  sorry

end count_arithmetic_three_digit_numbers_l543_543929


namespace arith_seq_infinite_digit_9_count_special_numbers_less_than_2010_pow_2010_l543_543023

-- Define the arithmetic sequence terms
def arith_seq_term (n : ℕ) : ℤ := 13 * n - 5

-- Proof problem for part (a)
theorem arith_seq_infinite_digit_9 : 
  ∀ m : ℕ, ∃ n : ℕ, arith_seq_term n = 10^m - 1 :=
sorry

-- Proof problem for part (b)
theorem count_special_numbers_less_than_2010_pow_2010 :
  let upper_bound := 2010 ^ 2010 in 
  ∃ count : ℕ, count = (λ m : ℕ, (10^m - 1 < upper_bound) ∧ (∃ k : ℕ, m = 3 + 12*k)).to_finset.card :=
sorry

end arith_seq_infinite_digit_9_count_special_numbers_less_than_2010_pow_2010_l543_543023


namespace train_length_l543_543750

theorem train_length 
    (speed_kmh : ℝ) (time_seconds : ℝ)
    (h1 : speed_kmh = 100)
    (h2 : time_seconds = 3.6) :
    ∃ length : ℝ, length ≈ 100 :=
by
    let speed_ms := speed_kmh * (1000 / 3600)
    let length := speed_ms * time_seconds
    have h3 : length ≈ 100, 
        from sorry 
    exact ⟨length, h3⟩

end train_length_l543_543750


namespace math_proof_problem_l543_543098

-- Definitions of the propositions
def Prop1 : Prop := ∀ (L : Type) (P Q : L -> Prop), (∃ R, parallel_to P R ∧ parallel_to Q R) → parallel_to P Q
def Prop2 : Prop := ∀ (P Q R : Type), parallel_to P R ∧ parallel_to Q R → parallel_to P Q
def Prop3 : Prop := ∀ (L : Type) (P Q : L -> Prop), (∃ R, perpendicular_to P R ∧ perpendicular_to Q R) → parallel_to P Q
def Prop4 : Prop := ∀ (L : Type) (l₁ l₂ l₃ : L), (angle_with l₁ l₃ = angle_with l₂ l₃) → parallel_to l₁ l₂

-- Correct propositions
def correct_propositions : Prop :=
  Prop2 ∧ Prop3 ∧ ¬Prop1 ∧ ¬Prop4

-- Proof statement
theorem math_proof_problem : Prop :=
  correct_propositions

end math_proof_problem_l543_543098


namespace cos_double_angle_l543_543398

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l543_543398


namespace correct_calculation_l543_543265

theorem correct_calculation (a : ℝ) : (-a)^10 / (-a)^3 = -a^7 :=
by sorry

end correct_calculation_l543_543265


namespace range_of_F_l543_543135

theorem range_of_F (f : ℝ → ℝ)
  (h : set.range f = set.Icc (1 / 2 : ℝ) 3) :
  set.range (λ x, f x + 1 / f x) = set.Icc 2 (10 / 3) := 
sorry

end range_of_F_l543_543135


namespace find_m_l543_543429

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem find_m (m : ℝ) :
  (∀ (x : ℝ), x ∈ set.Icc (1/3) 3 → f (Real.log x / Real.log 3 + m) ≥ 3) →
  f (Real.log (1 : ℝ) / Real.log 3 + m) = 3 →
  m = -1 ∨ m = 3 :=
sorry

end find_m_l543_543429


namespace smallest_common_multiple_five_digit_l543_543688

def is_multiple (a b : ℕ) : Prop := ∃ k, a = k * b

def smallest_five_digit_multiple_of_3_and_5 (x : ℕ) : Prop :=
  is_multiple x 3 ∧ is_multiple x 5 ∧ 10000 ≤ x ∧ x ≤ 99999 ∧ (∀ y, (10000 ≤ y ∧ y ≤ 99999 ∧ is_multiple y 3 ∧ is_multiple y 5) → x ≤ y)

theorem smallest_common_multiple_five_digit : smallest_five_digit_multiple_of_3_and_5 10005 :=
sorry

end smallest_common_multiple_five_digit_l543_543688


namespace sequence_sum_example_l543_543857

theorem sequence_sum_example (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 2) 
  (h2 : ∀ n, a (n + 1) = a n + 2^(n-1) + 1)
  (hS : ∀ n, S n = ∑ i in finset.range (n + 1), a (i + 1)) :
  S 10 = 1078 := by
  sorry

end sequence_sum_example_l543_543857


namespace cos_double_angle_sum_l543_543377

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l543_543377


namespace parallelepiped_volume_l543_543902

open Real EuclideanSpace

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (angle_a_b : Real := pi / 4)
variables (unit_a : ∥a∥ = 1)
variables (unit_b : ∥b∥ = 1)
variable (angle_condition : angle_between a b = angle_a_b)

theorem parallelepiped_volume :
  abs (a • ((b + 2 * (b × a)) × b)) = 1 :=
sorry

end parallelepiped_volume_l543_543902


namespace arithmetic_geometric_sum_l543_543910

theorem arithmetic_geometric_sum (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h_arith : 2 * b = a + c) (h_geom : a^2 = b * c) 
  (h_sum : a + 3 * b + c = 10) : a = -4 :=
by
  sorry

end arithmetic_geometric_sum_l543_543910


namespace sampling_method_is_systematic_l543_543724

-- Definition of the conditions
def factory_produces_product := True  -- Assuming the factory is always producing
def uses_conveyor_belt := True  -- Assuming the conveyor belt is always in use
def samples_taken_every_10_minutes := True  -- Sampling at specific intervals

-- Definition corresponding to the systematic sampling
def systematic_sampling := True

-- Theorem: Prove that given the conditions, the sampling method is systematic sampling.
theorem sampling_method_is_systematic :
  factory_produces_product → uses_conveyor_belt → samples_taken_every_10_minutes → systematic_sampling :=
by
  intros _ _ _
  trivial

end sampling_method_is_systematic_l543_543724


namespace merge_coins_n_ge_3_merge_coins_n_eq_2_l543_543244

-- For Part 1
theorem merge_coins_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), m = 1 ∨ m = 2 :=
sorry

-- For Part 2
theorem merge_coins_n_eq_2 (r s : ℕ) :
  ∃ (k : ℕ), r + s = 2^k * Nat.gcd r s :=
sorry

end merge_coins_n_ge_3_merge_coins_n_eq_2_l543_543244


namespace largest_divisor_of_exp_and_linear_combination_l543_543813

theorem largest_divisor_of_exp_and_linear_combination :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ x = 18 :=
by
  sorry

end largest_divisor_of_exp_and_linear_combination_l543_543813


namespace solution_set_inequality_l543_543642

theorem solution_set_inequality (x : ℝ) : (x-3) * (x-1) > 0 → (x < 1 ∨ x > 3) :=
by sorry

end solution_set_inequality_l543_543642


namespace g_neither_even_nor_odd_l543_543930

noncomputable def g (x : ℝ) : ℝ := 5 ^ (x ^ 2 - 4) - | x - 1 |

theorem g_neither_even_nor_odd : 
  (∀ x : ℝ, g (-x) ≠ g x) ∧ (∀ x : ℝ, g (-x) ≠ -g x) :=
by
  unfold g
  sorry

end g_neither_even_nor_odd_l543_543930


namespace find_x_l543_543466

def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, -1)
def scalar_multiply (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def collinear (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x :
  ∃ x : ℝ, collinear (vector_add (scalar_multiply 3 a) b) (c x) ∧ x = 4 :=
by
  sorry

end find_x_l543_543466


namespace tammy_earnings_after_3_weeks_l543_543588

noncomputable def oranges_picked_per_day (num_trees : ℕ) (oranges_per_tree : ℕ) : ℕ :=
  num_trees * oranges_per_tree

noncomputable def packs_sold_per_day (oranges_per_day : ℕ) (oranges_per_pack : ℕ) : ℕ :=
  oranges_per_day / oranges_per_pack

noncomputable def total_packs_sold_in_weeks (packs_per_day : ℕ) (days_in_week : ℕ) (num_weeks : ℕ) : ℕ :=
  packs_per_day * days_in_week * num_weeks

noncomputable def money_earned (total_packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  total_packs * price_per_pack

theorem tammy_earnings_after_3_weeks :
  let num_trees := 10
  let oranges_per_tree := 12
  let oranges_per_pack := 6
  let price_per_pack := 2
  let days_in_week := 7
  let num_weeks := 3
  oranges_picked_per_day num_trees oranges_per_tree /
  oranges_per_pack *
  days_in_week *
  num_weeks *
  price_per_pack = 840 :=
by {
  sorry
}

end tammy_earnings_after_3_weeks_l543_543588


namespace eccentricity_of_ellipse_l543_543434

theorem eccentricity_of_ellipse 
  (a b c m n : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : m > 0) 
  (h4 : n > 0) 
  (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 -> (m^2 + n^2 > x^2 + y^2))
  (hyperbola_eq : ∀ x y : ℝ, x^2 / m^2 - y^2 / n^2 = 1 -> (m^2 + n^2 > x^2 - y^2))
  (same_foci: ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 <= 1 → x^2 / m^2 - y^2 / n^2 = 1)
  (geometric_mean : c^2 = a * m)
  (arithmetic_mean : 2 * n^2 = 2 * m^2 + c^2) : 
  (c / a = 1 / 2) :=
sorry

end eccentricity_of_ellipse_l543_543434


namespace line_intersects_circle_midpoint_trajectory_is_circle_m_range_exists_l543_543423

noncomputable def circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5
noncomputable def line (m x y : ℝ) : Prop := mx - y + 1 + 2 * m = 0

-- Prove that line l always intersects circle C at two distinct points
theorem line_intersects_circle (m x₁ y₁ x₂ y₂ : ℝ) (h₁ : circle x₁ y₁) (h₂ : circle x₂ y₂) (h_line₁ : line m x₁ y₁) (h_line₂ : line m x₂ y₂) (h_diff : (x₁, y₁) ≠ (x₂, y₂)) :
  ∃ A B, circle (A.1) (A.2) ∧ circle (B.1) (B.2) ∧ line m (A.1) (A.2) ∧ line m (B.1) (B.2) ∧ A ≠ B := sorry

-- Equation of the trajectory of the midpoint M of chord AB
noncomputable def midpoint_trajectory (x y : ℝ) : Prop := (x + 2)^2 + (y - 1 / 2)^2 = 1 / 4

theorem midpoint_trajectory_is_circle :
  ∀ (x y : ℝ), midpoint_trajectory x y → ∃ r : ℝ, (x + 2)^2 + (y - 1 / 2)^2 = r ∧ r = 1 / 4 := sorry

-- Determine if there exists real number m such that there are four points on circle C at a distance of 4√5/5 from line l
theorem m_range_exists (m : ℝ) :
  (∃ (P1 P2 P3 P4 : ℝ × ℝ), circle (P1.1) (P1.2) ∧ circle (P2.1) (P2.2) ∧ circle (P3.1) (P3.2) ∧ circle (P4.1) (P4.2) ∧ line m (P1.1) (P1.2) ∧ line m (P2.1) (P2.2) ∧ line m (P3.1) (P3.2) ∧ line m (P4.1) (P4.2) ∧ abs (m * P1.1 - P1.2 + 1 + 2 * m) / sqrt(1 + m^2) = 4 * sqrt(5) / 5 )
  ↔ (m > 2 ∨ m < -2) := sorry

end line_intersects_circle_midpoint_trajectory_is_circle_m_range_exists_l543_543423


namespace power_function_value_at_two_l543_543083

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_value_at_two (a : ℝ) (h : f (1/2) a = 8) : f 2 a = 1 / 8 := by
  sorry

end power_function_value_at_two_l543_543083


namespace area_of_quadrilateral_PF1QF2_l543_543838

theorem area_of_quadrilateral_PF1QF2 (x y : ℝ) (F1 F2 P Q : ℝ×ℝ) 
  (h1 : ∀ p : ℝ×ℝ, p ∈ set_of (λ q, q.1^2/16 + q.2^2/4 = 1))
  (h2 : F1 = (4, 0) ∧ F2 = (-4, 0)) 
  (h3 : Q = (-P.1, -P.2))
  (h4 : dist P Q = dist F1 F2) :
  let a := 8 in
  let c := 4 in
  let b_sq := a^2 - c^2 in
  let m := |dist P F1| in
  let n := |dist P F2| in
  m * n = 8 :=
by sorry

end area_of_quadrilateral_PF1QF2_l543_543838


namespace projection_of_AB_onto_CD_l543_543469

def vector2D := (ℝ × ℝ)

-- Define the vectors AB and CD.
def AB : vector2D := (4, -3)
def CD : vector2D := (-5, -12)

-- Define the dot product of two vectors.
def dot_product (v1 v2 : vector2D) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the magnitude of a vector.
def magnitude (v : vector2D) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the projection of vector v1 in the direction of vector v2.
def projection (v1 v2 : vector2D) : ℝ := (dot_product v1 v2) / (magnitude v2)

-- State the theorem with the correct answer.
theorem projection_of_AB_onto_CD :
  projection AB CD = 16 / 13 :=
by
sorry

end projection_of_AB_onto_CD_l543_543469


namespace max_m_n_value_l543_543635

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end max_m_n_value_l543_543635


namespace quadrilateral_area_is_8_l543_543833

noncomputable section
open Real

def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def origin_symmetric (P Q : ℝ × ℝ) : Prop := P.1 = -Q.1 ∧ P.2 = -Q.2

def distance (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_quadrilateral (P Q F1 F2 : ℝ × ℝ) : Prop :=
  ∃ a b c d, a = P ∧ b = F1 ∧ c = Q ∧ d = F2

def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1*B.2 + B.1*C.2 + C.1*D.2 + D.1*A.2 - (B.1*A.2 + C.1*B.2 + D.1*C.2 + A.1*D.2))

theorem quadrilateral_area_is_8 (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  origin_symmetric P Q →
  distance P Q = distance f1 f2 →
  is_quadrilateral P Q f1 f2 →
  area_of_quadrilateral P f1 Q f2 = 8 := 
by
  sorry

end quadrilateral_area_is_8_l543_543833


namespace smallest_rational_l543_543001

theorem smallest_rational {a b c d : ℚ}
  (h1 : a = -2)
  (h2 : b = -1)
  (h3 : c = 2)
  (h4 : d = 0) : ∃ x ∈ {a, b, c, d}, ∀ y ∈ {a, b, c, d}, x ≤ y ∧ x = -2 :=
by {
  sorry
}

end smallest_rational_l543_543001


namespace prob_of_king_or_queen_top_l543_543499

/-- A standard deck comprises 52 cards, with 13 ranks and 4 suits, each rank having one card per suit. -/
def standard_deck : Set (String × String) :=
Set.prod { "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King" }
          { "Hearts", "Diamonds", "Clubs", "Spades" }

/-- There are four cards of rank King and four of rank Queen in the standard deck. -/
def count_kings_and_queens : Nat := 
4 + 4

/-- The total number of cards in a standard deck is 52. -/
def total_cards : Nat := 52

/-- The probability that the top card is either a King or a Queen is 2/13. -/
theorem prob_of_king_or_queen_top :
  (count_kings_and_queens / total_cards : ℚ) = (2 / 13 : ℚ) :=
sorry

end prob_of_king_or_queen_top_l543_543499


namespace students_taking_history_but_not_statistics_l543_543706

theorem students_taking_history_but_not_statistics :
  ∀ (total_students history_students statistics_students history_or_statistics_both : ℕ),
    total_students = 90 →
    history_students = 36 →
    statistics_students = 32 →
    history_or_statistics_both = 57 →
    history_students - (history_students + statistics_students - history_or_statistics_both) = 25 :=
by intros; sorry

end students_taking_history_but_not_statistics_l543_543706


namespace line_inclination_angle_and_slope_l543_543605

theorem line_inclination_angle_and_slope :
  ∀ θ : ℝ, ∀ m : ℝ,
    (line_equation : ∀ x y : ℝ, y = x - 2 → (θ = 45 ∧ m = 1)) :=
begin
  sorry
end

end line_inclination_angle_and_slope_l543_543605


namespace find_150th_letter_in_pattern_l543_543662

theorem find_150th_letter_in_pattern : 
  (let sequence := "ABCD";
   sequence.length = 4 → 
   sequence[(150 % 4)] = 'B') :=
by
  sorry

end find_150th_letter_in_pattern_l543_543662


namespace total_plums_l543_543753

def alyssa_plums : Nat := 17
def jason_plums : Nat := 10

theorem total_plums : alyssa_plums + jason_plums = 27 := 
by
  -- proof goes here
  sorry

end total_plums_l543_543753


namespace sum_of_solutions_l543_543536

def g (x : ℝ) : ℝ := 3 * x - 2

def g_inv (y : ℝ) : ℝ := (y + 2) / 3

theorem sum_of_solutions : ∑ x in {x : ℝ | g_inv x = g (2 * x)}, x = 8 / 17 :=
by
  sorry

end sum_of_solutions_l543_543536


namespace sum_common_seq_first_n_l543_543004

def seq1 (n : ℕ) := 2 * n - 1
def seq2 (n : ℕ) := 3 * n - 2

def common_seq (n : ℕ) := 6 * n - 5

def sum_first_n_terms (a : ℕ) (d : ℕ) (n : ℕ) := 
  n * (2 * a + (n - 1) * d) / 2

theorem sum_common_seq_first_n (n : ℕ) : 
  sum_first_n_terms 1 6 n = 3 * n^2 - 2 * n := 
by sorry

end sum_common_seq_first_n_l543_543004


namespace range_of_function_l543_543792

noncomputable def function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, 0 ≤ 5 + 4 * x - x^2 ∧ y = sqrt (5 + 4 * x - x^2) }

theorem range_of_function :
  function_range = Set.Icc 0 3 :=
sorry

end range_of_function_l543_543792


namespace cos_double_angle_l543_543405

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l543_543405


namespace extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l543_543458

-- Given function
def f (x : ℝ) : ℝ := x^3 - 12 * x + 12

-- Definition of derivative
def f' (x : ℝ) : ℝ := (3 : ℝ) * x^2 - (12 : ℝ)

-- Part 1: Extreme values
theorem extreme_values : 
  (f (-2) = 28) ∧ (f 2 = -4) :=
by
  sorry

-- Part 2: Maximum and minimum values on the interval [-3, 4]
theorem max_min_on_interval :
  (∀ x, -3 ≤ x ∧ x ≤ 4 → f x ≤ 28) ∧ (∀ x, -3 ≤ x ∧ x ≤ 4 → f x ≥ -4) :=
by
  sorry

-- Part 3: Coordinates of midpoint A and B with parallel tangents
theorem coordinates_midpoint_parallel_tangents :
  (f' x1 = f' x2 ∧ x1 + x2 = 0) → ((x1 + x2) / 2 = 0 ∧ (f x1 + f x2) / 2 = 12) :=
by
  sorry

end extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l543_543458


namespace cube_plane_probability_l543_543056

theorem cube_plane_probability : 
  let n := 8
  let k := 4
  let total_ways := Nat.choose n k
  let favorable_ways := 12
  ∃ p : ℚ, p = favorable_ways / total_ways ∧ p = 6 / 35 :=
sorry

end cube_plane_probability_l543_543056


namespace garden_borders_length_l543_543303

theorem garden_borders_length 
  (a b c d e : ℕ)
  (h1 : 6 * 7 = a^2 + b^2 + c^2 + d^2 + e^2)
  (h2 : a * a + b * b + c * c + d * d + e * e = 42) -- This is analogous to the condition
    
: 15 = (4*a + 4*b + 4*c + 4*d + 4*e - 2*(6 + 7)) / 2 :=
by sorry

end garden_borders_length_l543_543303


namespace probability_even_sum_l543_543069

def cards : Finset ℕ := {1, 2, 3, 4, 5}
def all_pairs : Finset (ℕ × ℕ) := cards.product cards
def even_sum_pairs : Finset (ℕ × ℕ) := all_pairs.filter (λ pair, (pair.1 + pair.2) % 2 = 0)

theorem probability_even_sum : (even_sum_pairs.card : ℚ) / all_pairs.card = 2 / 5 :=
by 
  sorry

end probability_even_sum_l543_543069


namespace find_150th_letter_in_pattern_l543_543661

theorem find_150th_letter_in_pattern : 
  (let sequence := "ABCD";
   sequence.length = 4 → 
   sequence[(150 % 4)] = 'B') :=
by
  sorry

end find_150th_letter_in_pattern_l543_543661


namespace intercepts_l543_543460

def line_equation (x y : ℝ) : Prop :=
  5 * x + 3 * y - 15 = 0

theorem intercepts (a b : ℝ) : line_equation a 0 ∧ line_equation 0 b → (a = 3 ∧ b = 5) :=
  sorry

end intercepts_l543_543460


namespace triplet_solution_l543_543789

theorem triplet_solution (a b c : ℝ)
  (h1 : a^2 + b = c^2)
  (h2 : b^2 + c = a^2)
  (h3 : c^2 + a = b^2) :
  (a = 0 ∧ b = 0 ∧ c = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = -1) ∨
  (a = -1 ∧ b = 0 ∧ c = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 0) :=
sorry

end triplet_solution_l543_543789


namespace tangent_line_at_0_max_value_in_interval_l543_543456

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x - x

theorem tangent_line_at_0 :
  let p := (0, f 0)
  in ∃ (m b : ℝ), m = 0 ∧ b = 0 ∧ ∀ x, (m * x + b) = f x → x = 0 := sorry

theorem max_value_in_interval : 
  ∃ (x : ℝ), x ∈ set.Icc 0 (Real.pi / 2) ∧ f x = Real.exp (Real.pi / 2) - Real.pi / 2 := sorry

end tangent_line_at_0_max_value_in_interval_l543_543456


namespace sum_of_digits_of_d_Isabella_problem_l543_543932

-- Define the exchange rate and the amount spent
def exchange_rate_Usd_to_Cad := 3 / 2
def amount_spent_Cad := 45

-- Define the main hypothesis that relates the exchanged money and the amount left
theorem sum_of_digits_of_d (d : ℕ) (h : 1.5 * d - 45 = d) : d = 90 :=
by sorry

-- Sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Main theorem: sum of digits of d is 9
theorem Isabella_problem (d : ℕ) (h : 1.5 * d - 45 = d) : sum_of_digits d = 9 := 
by
  have hd : d = 90 := sum_of_digits_of_d d h
  rw hd
  show sum_of_digits 90 = 9
  sorry

end sum_of_digits_of_d_Isabella_problem_l543_543932


namespace parabola_vertex_x_coord_l543_543602

theorem parabola_vertex_x_coord (a b c : ℝ)
  (h1 : 5 = a * 2^2 + b * 2 + c)
  (h2 : 5 = a * 8^2 + b * 8 + c)
  (h3 : 11 = a * 9^2 + b * 9 + c) :
  5 = (2 + 8) / 2 := 
sorry

end parabola_vertex_x_coord_l543_543602


namespace cyclic_quadrilateral_inscribed_radius_l543_543022

theorem cyclic_quadrilateral_inscribed_radius :
  ∀ (AB BC CD DA : ℝ), 
  AB = 13 → BC = 10 → CD = 8 → DA = 11 →
  let s := (AB + BC + CD + DA) / 2 in
  let A := Real.sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA)) in
  let r := A / s in
  r = 3 * Real.sqrt 5 :=
begin
  intros,
  sorry,
end

end cyclic_quadrilateral_inscribed_radius_l543_543022


namespace op_ast_n1_op_ast_21_l543_543340

-- Define a new operation ∗ for natural numbers
def op_ast : ℕ → ℕ → ℕ
| 1, 1 := 1
| (n + 1), 1 := 3 * (op_ast n 1)
| _, _ := 0 -- other cases (though not strictly needed)

-- Theorem to prove the desired results
theorem op_ast_n1 (n : ℕ) : op_ast n 1 = 3^(n - 1) :=
by sorry

-- Specific case for n = 2
theorem op_ast_21 : op_ast 2 1 = 3 :=
by sorry

end op_ast_n1_op_ast_21_l543_543340


namespace find_angle_l543_543759

theorem find_angle (x : Real) : 
  (x - (1 / 2) * (180 - x) = -18 - 24/60 - 36/3600) -> 
  x = 47 + 43/60 + 36/3600 :=
by
  sorry

end find_angle_l543_543759


namespace distribute_weights_l543_543550

theorem distribute_weights (max_weight : ℕ) (w_gbeans w_milk w_carrots w_apples w_bread w_rice w_oranges w_pasta : ℕ)
  (h_max_weight : max_weight = 20)
  (h_w_gbeans : w_gbeans = 4)
  (h_w_milk : w_milk = 6)
  (h_w_carrots : w_carrots = 2 * w_gbeans)
  (h_w_apples : w_apples = 3)
  (h_w_bread : w_bread = 1)
  (h_w_rice : w_rice = 5)
  (h_w_oranges : w_oranges = 2)
  (h_w_pasta : w_pasta = 3)
  : (w_gbeans + w_milk + w_carrots + w_apples + w_bread - 2 = max_weight) ∧ 
    (w_rice + w_oranges + w_pasta + 2 ≤ max_weight) :=
by
  sorry

end distribute_weights_l543_543550


namespace geometric_sequence_a7_l543_543152

variable {a : Nat → Nat}
variable (h₁ : a 3 = 3)
variable (h₂ : a 6 = 24)
variable (h₃ : ∃ q, a 3 * q ^ 3 = a 6 ∧ a 7 = a 3 * q ^ 4)

theorem geometric_sequence_a7 : a 7 = 48 :=
by
  obtain ⟨q, h₄, h₅⟩ := h₃
  rw [h₁, h₂] at h₄
  have hq : q = 2 := by sorry
  rw [h₁, hq, pow_succ, pow_succ, pow_zero] at h₅
  simp at h₅
  exact h₅

end geometric_sequence_a7_l543_543152


namespace cheerleaders_uniforms_l543_543219

theorem cheerleaders_uniforms (total_cheerleaders : ℕ) (size_6_cheerleaders : ℕ) (half_size_6_cheerleaders : ℕ) (size_2_cheerleaders : ℕ) : 
  total_cheerleaders = 19 →
  size_6_cheerleaders = 10 →
  half_size_6_cheerleaders = size_6_cheerleaders / 2 →
  size_2_cheerleaders = total_cheerleaders - (size_6_cheerleaders + half_size_6_cheerleaders) →
  size_2_cheerleaders = 4 :=
by
  intros
  sorry

end cheerleaders_uniforms_l543_543219


namespace inequality_solution_l543_543486

theorem inequality_solution (m : ℝ) : 
  (∀ x : ℝ, 2 * x + 7 > 3 * x + 2 ∧ 2 * x - 2 < 2 * m → x < 5) → m ≥ 4 :=
by
  sorry

end inequality_solution_l543_543486


namespace lines_intersect_l543_543610

open Real

theorem lines_intersect (k : ℝ) :
  (∃ x y : ℝ, y = 6*x + 5 ∧ y = -3*x - 30 ∧ y = 4*x + k) → k = -25/9 :=
by
  intro h
  obtain ⟨x, y, h1, h2, h3⟩ := h
  let xVal := -35/9
  have h_x_eq := eq_of_sub_eq_zero (calc
    (6 * xVal + 5) - (-3 * xVal - 30)
      = 9 * xVal + 35 : by ring
  h_x_eq.symm)
  have : x = xVal := by simp [xVal, h_x_eq]
  rw [this] at h1 h3
  have : y = 6 * (-35/9) + 5 := h1
  rw [this] at h3
  have h_y := (eq_of_sub_eq_zero (by norm_num [h3])).symm
  rw [h_y]
  exact sorry

end lines_intersect_l543_543610


namespace trig_identity_simplification_l543_543578

variable {x y φ : ℝ}

theorem trig_identity_simplification (x y φ : ℝ) :
  sin x ^ 2 + sin (x + y + φ) ^ 2 - 2 * sin x * sin (y + φ) * sin (x + y + φ) = 1 - sin (x + y + φ) ^ 2 :=
by sorry

end trig_identity_simplification_l543_543578


namespace number_of_squares_in_P_l543_543343

def ℤ² := { p : ℤ × ℤ // (∃ k : ℕ, p.1^2 + p.2^2 = 2^k) ∨ p = (0,0) }
def P (n : ℕ) : Set ℤ² := { p | (∃ k, k ≤ n ∧ p.1^2 + p.2^2 = 2^k) ∨ p = (0, 0) }

def number_of_squares (n : ℕ) : ℕ := 5 * n + 1

theorem number_of_squares_in_P (n : ℕ) :
  (number_of_squares n) = 5 * n + 1 := sorry

end number_of_squares_in_P_l543_543343


namespace triangle_area_l543_543300

/-- A right triangle ABC with vertices at (0, 0), (0, 5), and (3, 0) has an area of 7.5 square units. -/
theorem triangle_area :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 5 : ℝ)
  let C := (3 : ℝ, 0 : ℝ)
  ∃ (area : ℝ), area = 7.5 ∧ 
    (1 / 2) * (abs (B.2 - A.2)) * (abs (C.1 - A.1)) = area := 
by {
  sorry
}

end triangle_area_l543_543300


namespace compute_complex_power_l543_543019

theorem compute_complex_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 :=
by
  sorry

end compute_complex_power_l543_543019


namespace appropriate_words_in_range_l543_543551

-- Definitions for conditions
def min_duration := 40
def max_duration := 50
def speaking_rate := 160

def min_words := min_duration * speaking_rate
def max_words := max_duration * speaking_rate

def appropriate_length (words : Nat) : Prop :=
  words >= min_words ∧ words <= max_words

-- Statement needing proof
theorem appropriate_words_in_range :
  appropriate_length 6800 ∧ appropriate_length 7600 :=
by 
  unfold appropriate_length min_words max_words min_duration max_duration speaking_rate;
  simp;
  exact sorry

end appropriate_words_in_range_l543_543551


namespace total_gum_l543_543649

-- Define the conditions
def original_gum : ℕ := 38
def additional_gum : ℕ := 16

-- Define the statement to be proved
theorem total_gum : original_gum + additional_gum = 54 :=
by
  -- Proof omitted
  sorry

end total_gum_l543_543649


namespace valid_3_word_sentences_count_l543_543606

-- Define the words in the language of Trolldom
inductive TrolldomWords
  | thwap
  | brog
  | naffle
  | gorp

open TrolldomWords

-- Define what makes a sequence of words valid
def valid_sentence (s : List TrolldomWords) : Prop :=
  s.length = 3 ∧
  ¬ (s.head = some brog ∧ s.tail.head = some thwap) ∧
  ¬ (s.tail.head = some gorp ∧ s.tail.tail.head = some naffle)

-- The number of valid 3-word sentences
theorem valid_3_word_sentences_count : 
  ∃ (n : ℕ), n = 56 ∧ 
  n = List.length (List.filter valid_sentence ([thwap, brog, naffle, gorp].product ([thwap, brog, naffle, gorp].product [thwap, brog, naffle, gorp])).map (fun p => [p.1, p.2.1, p.2.2])) :=
by
  sorry

end valid_3_word_sentences_count_l543_543606


namespace max_m_n_value_l543_543634

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end max_m_n_value_l543_543634


namespace pyramid_cube_volume_l543_543298

/-- A pyramid with an equilateral triangular base with side length 1 has lateral faces that are
isosceles triangles with equal sides equal to the height of the pyramid. A cube is placed within
this pyramid such that one vertex of the cube touches the center of the triangular base and
the opposite vertex touches the pyramid's apex. What is the volume of this cube? -/
theorem pyramid_cube_volume :
  let s := 1 / 4 in
  let V := s ^ 3 in
  V = 1 / 64 := 
by
  sorry

end pyramid_cube_volume_l543_543298


namespace min_value_is_3_cbrt_4_l543_543089

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y^2 = 4) : ℝ :=
  x + 2 * y

theorem min_value_is_3_cbrt_4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y^2 = 4) : 
  minimum_value x y hx hy h = 3 * real.cbrt 4 :=
sorry

end min_value_is_3_cbrt_4_l543_543089


namespace total_weight_collected_l543_543058

def GinaCollectedBags : ℕ := 8
def NeighborhoodFactor : ℕ := 120
def WeightPerBag : ℕ := 6

theorem total_weight_collected :
  (GinaCollectedBags * NeighborhoodFactor + GinaCollectedBags) * WeightPerBag = 5808 :=
by
  sorry

end total_weight_collected_l543_543058


namespace area_of_quadrilateral_l543_543839
noncomputable def c := sqrt (16 - 4) -- √12 = 2√3

theorem area_of_quadrilateral (a b : ℝ) (P Q F1 F2 : ℝ×ℝ) :
  let e : set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} in
  let ellipse_symmetric := (P ∈ e) ∧ (Q ∈ e) ∧ (P.1 = -Q.1) ∧ (P.2 = -Q.2) ∧ dist (P, Q) = 2 * c in
  let F1F2 := 2 * c in
  let mn := 8 in
  (dist (P, F1) + dist (P, F2) = 2 * a) →
  (dist (P, F1)^2 + dist (P, F2)^2 = F1F2^2 / 4 * (a^2 - b^2)) →
  (mn = 8) →
  area_of_quadrilateral P F1 Q F2 = 8 :=
by
  sorry

end area_of_quadrilateral_l543_543839


namespace projectile_max_height_l543_543297

def h (t : ℝ) : ℝ := -9 * t^2 + 36 * t + 24

theorem projectile_max_height : ∃ t : ℝ, h t = 60 := 
sorry

end projectile_max_height_l543_543297


namespace circle_sum_l543_543698

theorem circle_sum (n : ℕ) (x : ℕ → ℤ) (h_sum : ∑ i in finset.range n, x i = n - 1) :
  ∃ (x1 : ℤ) (xo : fin n → ℤ), (xo 0 = x1) ∧ ∀ k : fin n, ∑ i in finset.range k, xo i ≤ k - 1 := 
begin
  sorry
end

end circle_sum_l543_543698


namespace main_theorem_l543_543118

-- Define the sets M and N
def M : Set ℝ := { x | 0 < x ∧ x < 10 }
def N : Set ℝ := { x | x < -4/3 ∨ x > 3 }

-- Define the complement of N in ℝ
def comp_N : Set ℝ := { x | ¬ (x < -4/3 ∨ x > 3) }

-- The main theorem to be proved
theorem main_theorem : M ∩ comp_N = { x | 0 < x ∧ x ≤ 3 } := 
by
  sorry

end main_theorem_l543_543118


namespace max_xy_min_function_l543_543714

-- Problem 1: Prove that the maximum value of xy is 8 given the conditions
theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 8) : xy ≤ 8 :=
sorry

-- Problem 2: Prove that the minimum value of the function is 9 given the conditions
theorem min_function (x : ℝ) (hx : -1 < x) : (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

end max_xy_min_function_l543_543714


namespace min_ab_diff_value_l543_543539

noncomputable def min_ab_diff (x y z : ℝ) : ℝ :=
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2

theorem min_ab_diff_value : ∀ (x y z : ℝ),
  0 ≤ x → 0 ≤ y → 0 ≤ z → min_ab_diff x y z = 36 :=
by
  intros x y z hx hy hz
  sorry

end min_ab_diff_value_l543_543539


namespace find_wickets_in_last_match_l543_543722

-- Define the conditions
def bowling_average_before : ℝ := 12.4
def wickets_before : ℕ := 85
def runs_in_last_match : ℝ := 26
def average_decrease : ℝ := 0.4

-- Define the function to calculate the new wickets
noncomputable def wickets_in_last_match : ℕ :=
  let total_runs_before := bowling_average_before * (wickets_before : ℝ)
  let new_average := bowling_average_before - average_decrease
  let total_runs_after := total_runs_before + runs_in_last_match
  let total_wickets_after := wickets_before + (5 : ℕ) -- we are proving this equals 5
  5

-- Theorem to prove
theorem find_wickets_in_last_match : wickets_in_last_match = 5 := by
  sorry -- proof goes here

#print Theorem find_wickets_in_last_match

end find_wickets_in_last_match_l543_543722


namespace polynomial_coefficients_l543_543842

theorem polynomial_coefficients (a : ℕ → ℤ) :
  (∀ x : ℤ, (2 * x - 1) * ((x + 1) ^ 7) = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + 
  (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 + (a 7) * x^7 + (a 8) * x^8) →
  (a 0 = -1) ∧
  (a 0 + a 2 + a 4 + a 6 + a 8 = 64) ∧
  (a 1 + 2 * (a 2) + 3 * (a 3) + 4 * (a 4) + 5 * (a 5) + 6 * (a 6) + 7 * (a 7) + 8 * (a 8) = 704) := by
  sorry

end polynomial_coefficients_l543_543842


namespace systematic_sampling_problem_l543_543370

theorem systematic_sampling_problem :
  ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 50) ∧ 
  S.card = 5 ∧ 
  ∃ d : ℕ, d = 10 ∧ ∀ x y ∈ S, x ≠ y → (y = x + d ∨ x = y + d) :=
  ∃ S, S = {3, 13, 23, 33, 43} :=
sorry

end systematic_sampling_problem_l543_543370


namespace exceptional_face_edges_multiple_of_3_l543_543599

theorem exceptional_face_edges_multiple_of_3
  (polyhedron : Polyhedron)
  (coloring : polyhedron.faces → ℕ)
  (H_color : ∀ f1 f2 : polyhedron.faces, polyhedron.is_adjacent f1 f2 → coloring f1 ≠ coloring f2)
  (H_mult_of_3 : ∀ f : polyhedron.faces, f ≠ exceptional_face → (polyhedron.edges_on_face f) % 3 = 0)
  : (polyhedron.edges_on_face exceptional_face) % 3 = 0 := 
sorry

end exceptional_face_edges_multiple_of_3_l543_543599


namespace number_of_symmetric_hexominoes_l543_543783

def hexominoes : Set Shape := {s | is_hexomino s}

def has_rotational_symmetry (s : Shape) : Prop :=
  s.symmetry ∈ Set.of_list [90, 180, 270]

theorem number_of_symmetric_hexominoes :
  (hexominoes.filter has_rotational_symmetry).card = 7 := sorry

end number_of_symmetric_hexominoes_l543_543783


namespace Minjeong_family_juice_consumption_l543_543968

theorem Minjeong_family_juice_consumption :
  (∀ (amount_per_time : ℝ) (times_per_day : ℕ) (days_per_week : ℕ),
  amount_per_time = 0.2 → times_per_day = 3 → days_per_week = 7 → 
  amount_per_time * times_per_day * days_per_week = 4.2) :=
by
  intros amount_per_time times_per_day days_per_week h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end Minjeong_family_juice_consumption_l543_543968


namespace union_of_A_and_B_l543_543524

def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} :=
by
  sorry

end union_of_A_and_B_l543_543524


namespace area_of_quadrilateral_l543_543840
noncomputable def c := sqrt (16 - 4) -- √12 = 2√3

theorem area_of_quadrilateral (a b : ℝ) (P Q F1 F2 : ℝ×ℝ) :
  let e : set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} in
  let ellipse_symmetric := (P ∈ e) ∧ (Q ∈ e) ∧ (P.1 = -Q.1) ∧ (P.2 = -Q.2) ∧ dist (P, Q) = 2 * c in
  let F1F2 := 2 * c in
  let mn := 8 in
  (dist (P, F1) + dist (P, F2) = 2 * a) →
  (dist (P, F1)^2 + dist (P, F2)^2 = F1F2^2 / 4 * (a^2 - b^2)) →
  (mn = 8) →
  area_of_quadrilateral P F1 Q F2 = 8 :=
by
  sorry

end area_of_quadrilateral_l543_543840


namespace negation_of_exisential_inequality_l543_543999

open Classical

theorem negation_of_exisential_inequality :
  ¬ (∃ x : ℝ, x^2 - x + 1/4 ≤ 0) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 := 
by 
sorry

end negation_of_exisential_inequality_l543_543999


namespace correct_statements_l543_543911

variable (a_1 a_2 b_1 b_2 : ℝ)

def ellipse1 := ∀ x y : ℝ, x^2 / a_1^2 + y^2 / b_1^2 = 1
def ellipse2 := ∀ x y : ℝ, x^2 / a_2^2 + y^2 / b_2^2 = 1

axiom a1_pos : a_1 > 0
axiom b1_pos : b_1 > 0
axiom a2_gt_b2_pos : a_2 > b_2 ∧ b_2 > 0
axiom same_foci : a_1^2 - b_1^2 = a_2^2 - b_2^2
axiom a1_gt_a2 : a_1 > a_2

theorem correct_statements : 
  (¬(∃ x y, (x^2 / a_1^2 + y^2 / b_1^2 = 1) ∧ (x^2 / a_2^2 + y^2 / b_2^2 = 1))) ∧ 
  (a_1^2 - a_2^2 = b_1^2 - b_2^2) :=
by 
  sorry

end correct_statements_l543_543911


namespace f_max_value_f_zeros_l543_543103

-- Define the function f(x)
def f (x : ℝ) : ℝ := (sqrt 3) * sin (2 * x) - 2 * (sin x)^2

-- Step 1: Prove that the maximum value of f(x) is 1
theorem f_max_value : ∃ x : ℝ, f x = 1 := sorry

-- Step 2: Prove that the set of zeros of f(x) is { x | x = k * π ∨ x = k * π + π / 3, k ∈ ℤ }
theorem f_zeros : { x : ℝ | f x = 0 } = { x : ℝ | ∃ k : ℤ, x = k * π ∨ x = k * π + π / 3 } := sorry

end f_max_value_f_zeros_l543_543103


namespace diagonal_of_rectangular_solid_l543_543092

theorem diagonal_of_rectangular_solid 
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 24)
  (h2 : a + b + c = 6) : 
  real.sqrt (a^2 + b^2 + c^2) = 2 * real.sqrt 3 := 
  sorry

end diagonal_of_rectangular_solid_l543_543092


namespace one_third_of_1206_is_100_5_percent_of_400_l543_543270

theorem one_third_of_1206_is_100_5_percent_of_400 : (1206 / 3) / 400 * 100 = 100.5 := by
  sorry

end one_third_of_1206_is_100_5_percent_of_400_l543_543270


namespace domain_length_eq_1_over_81_p_plus_q_eq_82_l543_543537

def g (x : ℝ) : ℝ := log (1 / 3) (log 9 (log (1 / 9) (log 81 (log (1 / 81) x))))

theorem domain_length_eq_1_over_81 (x : ℝ) :
  1 / 81^81 < x ∧ x < 1 / 81 → (1 / 81 - 1 / 81^81 = (81^80 - 1) / 81^81) :=
by sorry

theorem p_plus_q_eq_82 :
  let p := 1
  let q := 81
  p + q = 82 :=
by sorry

end domain_length_eq_1_over_81_p_plus_q_eq_82_l543_543537


namespace vertex_parabola_shape_l543_543528

theorem vertex_parabola_shape
  (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (P : ℝ → ℝ → Prop), 
  (∀ t : ℝ, ∃ (x y : ℝ), P x y ∧ (x = (-t / (2 * a))) ∧ (y = -a * (x^2) + d)) ∧
  (∀ x y : ℝ, P x y ↔ (y = -a * (x^2) + d)) :=
by
  sorry

end vertex_parabola_shape_l543_543528


namespace sum_of_three_numbers_l543_543231

theorem sum_of_three_numbers (a b c : ℕ) (h1 : b = 10)
                            (h2 : (a + b + c) / 3 = a + 15)
                            (h3 : (a + b + c) / 3 = c - 25) :
                            a + b + c = 60 :=
sorry

end sum_of_three_numbers_l543_543231


namespace cosine_identity_l543_543388

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l543_543388


namespace examination_students_total_l543_543707

/-
  Problem Statement:
  Given:
  - 35% of the students passed the examination.
  - 546 students failed the examination.

  Prove:
  - The total number of students who appeared for the examination is 840.
-/

theorem examination_students_total (T : ℝ) (h1 : 0.35 * T + 0.65 * T = T) (h2 : 0.65 * T = 546) : T = 840 :=
by
  -- skipped proof part
  sorry

end examination_students_total_l543_543707


namespace rhombus_area_l543_543705

theorem rhombus_area (side d1 d2 : ℝ) (h_side : side = 25) (h_d1 : d1 = 30) (h_diag : d2 = 40) :
  (d1 * d2) / 2 = 600 :=
by
  rw [h_d1, h_diag]
  norm_num

end rhombus_area_l543_543705


namespace bridge_length_l543_543608

theorem bridge_length (train_length : ℕ) (train_speed_kmph : ℕ) (cross_time_sec : ℕ) (train_length = 160) (train_speed_kmph = 45) (cross_time_sec = 30) :
  let speed_mps := (train_speed_kmph * 1000) / 3600 in
  let total_distance := speed_mps * cross_time_sec in
  let bridge_length := total_distance - train_length in
  bridge_length = 215 :=
by
  sorry

end bridge_length_l543_543608


namespace chemist_mixture_l543_543784

-- Define the conditions as Lean 4 statements.
variables {a b : ℝ}

-- Conditions
def mix_condition (a b : ℝ) : Prop :=
  a + b = 30 ∧ 4 * a + 6 * b = 170

-- Proof statement
theorem chemist_mixture : ∃ a : ℝ, mix_condition a (30 - a) ∧ a = 5 :=
begin
  use 5,
  split,
  { exact ⟨by norm_num, by norm_num⟩ },
  { refl }
end

end chemist_mixture_l543_543784


namespace samples_from_workshop_l543_543292

theorem samples_from_workshop (T S P : ℕ) (hT : T = 2048) (hS : S = 128) (hP : P = 256) : 
  (s : ℕ) → (s : ℕ) = (256 * 128 / 2048) → s = 16 :=
by
  intros s hs
  rw [Nat.div_eq (256 * 128) 2048] at hs
  sorry

end samples_from_workshop_l543_543292


namespace translate_right_one_unit_l543_543752

theorem translate_right_one_unit (x y : ℤ) (hx : x = 4) (hy : y = -3) : (x + 1, y) = (5, -3) :=
by
  -- The proof would go here
  sorry

end translate_right_one_unit_l543_543752


namespace cos_double_angle_sum_l543_543378

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l543_543378


namespace insurance_ratio_discrepancy_future_cost_prediction_l543_543693

-- Define the conditions given in the problem.
def insurance_coverage_russia (coverage_r : ℝ) (premium_r : ℝ) : ℝ :=
  coverage_r / premium_r

def insurance_coverage_germany (coverage_g : ℝ) (premium_g : ℝ) : ℝ :=
  coverage_g / premium_g

-- Conditions based on the given problem.
axiom russia_ratio_conditions :
  (insurance_coverage_russia 100000 2000 = 50) ∧ 
  (insurance_coverage_russia 1500000 23000 ≈ 65.22)

axiom germany_ratio_conditions :
  insurance_coverage_germany 3000000 80 = 37500

-- Define the problem statements based on the correct answers.
theorem insurance_ratio_discrepancy :
  ∃ r_f : ℝ, r_f = 37500 ∧ ∃ r_r : ℝ, r_r ~ 65 ∧ r_f > r_r :=
by sorry

theorem future_cost_prediction :
  ∃ f : ℝ → ℝ → Prop, 
    (∀ d s, f d s = d ∨ f d s = s) → 
    (∀ d s, f d s ∈ {inc, dec}) :=
by sorry

end insurance_ratio_discrepancy_future_cost_prediction_l543_543693


namespace smallest_three_digit_number_l543_543821

open Nat

theorem smallest_three_digit_number
  (x : ℕ)
  (h1 : 5 * x % 10 = 15 % 10)
  (h2 : (3 * x + 4) % 8 = 7 % 8)
  (h3 : (-3 * x + 2) % 17 = x % 17)
  (h4 : 100 ≤ x)  -- smallest three-digit number constraint
  (h5 : x < 1000) -- without this condition, it could not be considered a three-digit number
  : x = 230 := 
sorry

end smallest_three_digit_number_l543_543821


namespace Cary_walked_miles_round_trip_l543_543780

theorem Cary_walked_miles_round_trip : ∀ (m : ℕ), 
  150 * m - 200 = 250 → m = 3 := 
by
  intros m h
  sorry

end Cary_walked_miles_round_trip_l543_543780


namespace min_value_inequality_l543_543076

noncomputable def minValue : ℝ := 17 / 2

theorem min_value_inequality (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_cond : a + 2 * b = 1) :
  a^2 + 4 * b^2 + 1 / (a * b) = minValue := 
by
  sorry

end min_value_inequality_l543_543076


namespace count_squares_and_rectangles_l543_543336

theorem count_squares_and_rectangles (m n : ℕ) (m_eq : m = 3) (n_eq : n = 5) :
  let total_squares := 
    ∑ i in (finset.range (m + 1)), ∑ j in (finset.range (n + 1)), (m - i) * (n - j)
  let total_rectangles := 
    ∑ i in (finset.range (m + 1)), ∑ j in (finset.range (n + 1)), if i ≠ j then (m - i) * (n - j) else 0
  total_squares + total_rectangles = 72 := 
by 
  sorry

end count_squares_and_rectangles_l543_543336


namespace fewest_students_possible_l543_543293

theorem fewest_students_possible (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) ↔ N = 59 :=
by
  sorry

end fewest_students_possible_l543_543293


namespace wendy_second_level_treasures_l543_543653

theorem wendy_second_level_treasures
  (points_per_treasure : ℕ)
  (first_level_treasures : ℕ)
  (total_score : ℕ) :
  points_per_treasure = 5 →
  first_level_treasures = 4 →
  total_score = 35 →
  (total_score - (first_level_treasures * points_per_treasure)) / points_per_treasure = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp [h1, h2, h3]
  sorry

end wendy_second_level_treasures_l543_543653


namespace quadrilateral_area_is_8_l543_543835

noncomputable section
open Real

def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def origin_symmetric (P Q : ℝ × ℝ) : Prop := P.1 = -Q.1 ∧ P.2 = -Q.2

def distance (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_quadrilateral (P Q F1 F2 : ℝ × ℝ) : Prop :=
  ∃ a b c d, a = P ∧ b = F1 ∧ c = Q ∧ d = F2

def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1*B.2 + B.1*C.2 + C.1*D.2 + D.1*A.2 - (B.1*A.2 + C.1*B.2 + D.1*C.2 + A.1*D.2))

theorem quadrilateral_area_is_8 (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  origin_symmetric P Q →
  distance P Q = distance f1 f2 →
  is_quadrilateral P Q f1 f2 →
  area_of_quadrilateral P f1 Q f2 = 8 := 
by
  sorry

end quadrilateral_area_is_8_l543_543835


namespace shorter_than_average_height_l543_543591

theorem shorter_than_average_height : 
  ∀ (avg_height taller shorter : ℤ), 
    avg_height = 175 → 
    taller = 2 → 
    shorter = -2 → 
    (taller = +2) → 
    (avg_height - 2) = (avg_height + shorter) :=
by
  intros avg_height taller shorter h_avg h_taller h_shorter h_taller_notation
  cases h_avg
  cases h_taller
  cases h_shorter
  cases h_taller_notation
  sorry

end shorter_than_average_height_l543_543591


namespace find_xy_l543_543449

theorem find_xy (x y : ℝ) (h : (x^2 + 6 * x + 12) * (5 * y^2 + 2 * y + 1) = 12 / 5) : 
    x * y = 3 / 5 :=
sorry

end find_xy_l543_543449


namespace usual_time_cover_journey_l543_543255

theorem usual_time_cover_journey (S T : ℝ) (H : S / T = (5/6 * S) / (T + 8)) : T = 48 :=
by
  sorry

end usual_time_cover_journey_l543_543255


namespace number_of_solutions_l543_543031

open Real

theorem number_of_solutions :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 * π ∧ 3 * (sin x)^3 - 7 * (sin x)^2 + 4 * sin x = 0) ↔
  (finset.univ.filter (λ x, 0 ≤ x ∧ x ≤ 2 * π ∧ 3 * (sin x)^3 - 7 * (sin x)^2 + 4 * sin x = 0)).card = 4 :=
sorry

end number_of_solutions_l543_543031


namespace race_results_l543_543723

/-- First squirrel's statements -/
def P1 : Prop := hare = 1  -- "The hare took first place."
def P2 : Prop := fox = 2   -- "The fox was second."

/-- Second squirrel's statements -/
def Q1 : Prop := hare = 2  -- "The hare took second place."
def Q2 : Prop := moose = 1 -- "The moose was first."

/-- Owl's observation: one part of each statement is true and the other is false -/
axiom owl_observation: (P1 ∧ ¬P2 ∨ ¬P1 ∧ P2) ∧ (Q1 ∧ ¬Q2 ∨ ¬Q1 ∧ Q2)

/-- Prove who took first and second place -/
theorem race_results (hare fox moose : ℕ) : moose = 1 ∧ fox = 2 :=
by
  sorry

end race_results_l543_543723


namespace opposite_face_of_7_l543_543563

theorem opposite_face_of_7
  (faces : Fin 6 → ℕ)
  (h_faces : (finset.univ : Finset (Fin 6)).val.map faces = [6, 7, 8, 9, 10, 11])
  (sum_faces : ∀ (a b : Fin 6), a ≠ b → faces a ∈ [33, 35] → faces b ∈ [33, 35])
  : (faces (Fin.ofNat 1) = 7 → (faces (Fin.ofNat 2) = 9 ∨ faces (Fin.ofNat 2) = 11)) :=
by
  sorry

end opposite_face_of_7_l543_543563


namespace th150th_letter_is_B_l543_543658

def pattern := "ABCD".data

def nth_letter_in_pattern (n : ℕ) : Char :=
  let len := pattern.length
  pattern.get n % len

theorem th150th_letter_is_B :
  nth_letter_in_pattern 150 = 'B' :=
by {
  -- This proof is placed here as a placeholder
  sorry
}

end th150th_letter_is_B_l543_543658


namespace cos_double_angle_l543_543401

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l543_543401


namespace simplify_sqrt_l543_543990

theorem simplify_sqrt : sqrt ((-7) ^ 2) = 7 :=
by
  sorry

end simplify_sqrt_l543_543990


namespace daily_greening_areas_days_team_B_minimum_working_days_team_A_l543_543744

-- Definition of daily greening amounts.
def daily_greening_area_team_B (x : ℕ) : Prop :=
  (400 / x) - (400 / (2 * x)) = 4

def daily_greening_area_team_A (x : ℕ) : Prop :=
  2 * x

-- Prove that the daily greening area for team A and B.
theorem daily_greening_areas : ∃ x : ℕ, daily_greening_area_team_B x ∧ daily_greening_area_team_A x = 100 :=
by
  use 50
  sorry

-- Definition of working days based on team A's days.
def working_days_team_B (a : ℕ) : ℕ :=
  36 - 2 * a

-- Prove the number of days for team B given days for team A.
theorem days_team_B (a : ℕ) : 36 - 2 * a =
  (1800 - 100 * a) / 50 :=
by
  sorry

-- Cost constraints for the teams working.
def cost_constraint (a : ℕ) : Prop :=
  (0.4 * a + 0.25 * (working_days_team_B a) ≤ 8)

-- Prove the minimum number of days team A should work.
theorem minimum_working_days_team_A : ∃ a : ℕ, a ≥ 10 ∧ cost_constraint a :=
by
  use 10
  sorry

end daily_greening_areas_days_team_B_minimum_working_days_team_A_l543_543744


namespace cos_double_angle_l543_543384

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l543_543384


namespace cosine_identity_l543_543387

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l543_543387


namespace feet_of_perpendiculars_on_circle_l543_543225

structure Point (α : Type) := (x y : α)

structure Triangle (α : Type) :=
(A B C : Point α)

structure Circle (α : Type) :=
(center : Point α) (radius : α)

structure Line (α : Type) :=
(p1 p2 : Point α)

def Orthocenter (α : Type) (T : Triangle α) (O : Point α) : Prop := sorry

def Midpoint (α : Type) (P1 P2 MP : Point α) : Prop := sorry

def OnCircle (α : Type) (P : Point α) (C : Circle α) : Prop := sorry

def OnLine (α : Type) (P : Point α) (L : Line α) : Prop := sorry

def EulerLine (α : Type) (T : Triangle α) (O C G E : Point α) : Prop := sorry

def Ratio (α : Type) [Field α] (a b c : α) (br : α → α → α) : Prop := sorry

theorem feet_of_perpendiculars_on_circle
  {α : Type} [Field α] (T : Triangle α) (N P Q K M L W X Y Z : Point α) (C : Circle α) :
  Orthocenter α T X →
  Midpoint α X T.A K →
  Midpoint α X T.B M →
  Midpoint α X T.C L →
  OnCircle α K C →
  OnCircle α M C →
  OnCircle α L C →
  C.center = W →
  EulerLine α T X Z Y W →
  Ratio α (dist α X W) (dist α W Y) (dist α Y Z) (3 : α) (1 : α) (2 : α) →
  OnCircle α N C →
  OnCircle α P C →
  OnCircle α Q C := 
sorry

end feet_of_perpendiculars_on_circle_l543_543225


namespace number_of_solutions_l543_543892

noncomputable def satisfy_condition (θ : ℝ) : Prop :=
  θ > 0 ∧ θ <= 4 * Real.pi ∧ (2 - 4 * Real.cos θ + 3 * Real.sin (2 * θ) - 2 * Real.cos (4 * θ) = 0)

theorem number_of_solutions : (Finset.filter satisfy_condition (Finset.range (4 * 3142))).card = 8 := 
sorry

end number_of_solutions_l543_543892


namespace prop_C_correct_prop_D_correct_l543_543099

-- Define vectors as points in space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Vector subtraction
def vec_sub (p q : Point) : Point :=
  ⟨p.x - q.x, p.y - q.y, p.z - q.z⟩

-- Scalar multiplication
def scalar_mul (c : ℝ) (p : Point) : Point :=
  ⟨c * p.x, c * p.y, c * p.z⟩

-- Vector addition
def vec_add (p q : Point) : Point :=
  ⟨p.x + q.x, p.y + q.y, p.z + q.z⟩

-- Proposition C setup
def prop_C (P A B C : Point) : Prop :=
  vec_sub P C = vec_add (scalar_mul (1/4) (vec_sub P A)) (scalar_mul (3/4) (vec_sub P B))

def are_collinear (A B C : Point) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ C = vec_add B (scalar_mul k (vec_sub B A))

-- Proposition D setup
def parallel_planes (a b : Point) : Prop :=
  ∃ λ : ℝ, b = scalar_mul λ a

-- Conditions
def normal_vector_alpha : Point := ⟨1, 3, -4⟩
def normal_vector_beta (k : ℝ) : Point := ⟨-2, -6, k⟩
def k_value := (8 : ℝ)

-- Final theorem statements
theorem prop_C_correct (P A B C : Point) (h : prop_C P A B C) : are_collinear A B C := sorry

theorem prop_D_correct (k : ℝ) (h : parallel_planes normal_vector_alpha (normal_vector_beta k)) : k = k_value := sorry

end prop_C_correct_prop_D_correct_l543_543099


namespace chord_square_sum_l543_543164

theorem chord_square_sum (O A B C D E : Point) (r : ℝ) :
  AB.is_diameter O r ∧ CD.is_chord_in_circle O r ∧ (B.dist E = 5) ∧ ∠ (A E C) = 60 :=
  CE^2 + DE^2 = 100 :=
by
  sorry

end chord_square_sum_l543_543164


namespace probability_multiple_of_3_or_4_l543_543620

theorem probability_multiple_of_3_or_4 :
  let numbers := {n | 1 ≤ n ∧ n ≤ 30},
      multiples_of_3 := {n | n ∈ numbers ∧ n % 3 = 0},
      multiples_of_4 := {n | n ∈ numbers ∧ n % 4 = 0},
      multiples_of_12 := {n | n ∈ numbers ∧ n % 12 = 0},
      favorable_outcomes := multiples_of_3 ∪ multiples_of_4,
      double_counted_outcomes := multiples_of_12,
      total_favorable_outcomes := set.card favorable_outcomes - set.card double_counted_outcomes,
      total_outcomes := set.card numbers in
  total_favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_multiple_of_3_or_4_l543_543620


namespace sum_of_fraction_values_l543_543043

theorem sum_of_fraction_values : 
  ∃ n1 n2 : ℕ, 
  (8 * n1 + 157) / (4 * n1 + 7) = 15 ∧ 
  (8 * n2 + 157) / (4 * n2 + 7) = 3 ∧ 
  n1 ≠ n2 ∧ 
  n1 > 0 ∧ n2 > 0 ∧ 
  (15 + 3 = 18) :=
by { use [1, 34], 
     split,
     { norm_num },
     split,
     { norm_num },
     split,
     { exact dec_trivial },
     split,
     { norm_num },
     split,
     { norm_num },
     exact dec_trivial }

end sum_of_fraction_values_l543_543043


namespace max_experiments_needed_l543_543719

-- Definitions of conditions
def experimental_range : Set ℝ := {x : ℝ | 60 ≤ x ∧ x ≤ 81}
def interval_length : ℝ := 81 - 60
def F (n : ℕ) : ℕ

-- Hypotheses from the problem
axiom fractional_method_F7_property : F 7 = 20

-- The theorem to be proved
theorem max_experiments_needed : F 7 = 20 → 6 = 6 :=
by
  intro h
  sorry

end max_experiments_needed_l543_543719


namespace intersection_A_B_l543_543463

-- Definitions of the sets A and B
def A := { x : ℤ | abs x < 2 }
def B := { -1, 0, 1, 2, 3 }

-- Statement to prove
theorem intersection_A_B : A ∩ B = {-1, 0, 1} := 
by sorry

end intersection_A_B_l543_543463


namespace division_addition_correct_l543_543258

theorem division_addition_correct : 0.2 / 0.005 + 0.1 = 40.1 :=
by
  sorry

end division_addition_correct_l543_543258


namespace eq_AM_eq_AF_iff_angle_BAC_sixty_l543_543172

variable {A B C M F : Type}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space F]

def is_acute_triangle (A B C : Type) : Prop := sorry
def is_midpoint (M : Type) (A C : Type) : Prop := sorry
def is_foot_of_altitude (F : Type) (C : Type) (A B : Type) : Prop := sorry

theorem eq_AM_eq_AF_iff_angle_BAC_sixty
  (ABC_acute : is_acute_triangle A B C)
  (M_mid : is_midpoint M A C)
  (F_foot : is_foot_of_altitude F C A B) :
  dist A M = dist A F ↔ angle B A C = 60 :=
sorry

end eq_AM_eq_AF_iff_angle_BAC_sixty_l543_543172


namespace triangle_lengths_and_angles_l543_543640

noncomputable theory

def arithmeticTriangleSides (b d : ℝ) : ℝ × ℝ × ℝ :=
  let a := b - d
  let c := b + d
  (a, b, c)

def heronArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_lengths_and_angles (S d b : ℝ) (hS : S = 6) (hd : d = 1) (hb : b = 4) :
  let (a, _, c) := arithmeticTriangleSides b d
  a = 3 ∧ c = 5 ∧
  let sin_alpha := 3 / 5
  let alpha := real.arcsin sin_alpha
  let beta := real.arccos sin_alpha
  let gamma := real.arcsin (sqrt(1 - sin_alpha^2))
  -- Expected angles as radians:
  alpha = 36 * (π / 180) + 52 * ((π / 180) / 60) ∧
  beta = 53 * (π / 180) + 8 * ((π / 180) / 60) ∧
  gamma = π / 2 := by
  sorry

end triangle_lengths_and_angles_l543_543640


namespace x_squared_minus_4_l543_543073

theorem x_squared_minus_4 (x : ℝ) (h : 2^x + 2^x + 2^x = 64) : 
  x^2 - 4 = 32 - 12 * real.log 3 / real.log 2 + (real.log 3 / real.log 2)^2 :=
sorry

end x_squared_minus_4_l543_543073


namespace area_of_triangle_CDM_l543_543351

noncomputable theory -- Required for real number operations

open Real

theorem area_of_triangle_CDM :
  let A B C D M : Point
  in EquilateralTriangle ABC 2 → Midpoint C B D → Midpoint M A C → 
     triangle_area C D M = sqrt 3 / 2 :=
by
  intro A B C D M
  intro h_eq ABC_2
  intro h_mid_C_B_D
  intro h_mid_M_A_C
  sorry

end area_of_triangle_CDM_l543_543351


namespace maxAbsZ3_is_sqrt5_l543_543871

noncomputable def maxAbsZ3 (z1 z2 z3 : ℂ) : ℝ :=
  if h : abs z1 ≤ 1 ∧ abs z2 ≤ 2 ∧ abs (2 * z3 - z1 - z2) ≤ abs (z1 - z2) then abs z3 else 0

theorem maxAbsZ3_is_sqrt5 (z1 z2 z3 : ℂ) (h1 : abs z1 ≤ 1) (h2 : abs z2 ≤ 2) (h3 : abs (2 * z3 - z1 - z2) ≤ abs (z1 - z2)) :
  maxAbsZ3 z1 z2 z3 = sqrt 5 := sorry

end maxAbsZ3_is_sqrt5_l543_543871


namespace line_intersects_ellipse_l543_543863

def f (a x: ℝ) : ℝ := (1/3) * a * x^3 - (1/2) * a * x^2 - x
def f' (a x: ℝ) : ℝ := a * x^2 - a * x - 1
def ellipse (x y: ℝ) : Prop := x^2 / 2 + y^2 = 1
def line_eq (a x y x1: ℝ) : Prop := y = a * x - a

theorem line_intersects_ellipse (a x1 x2: ℝ) (h1: f' a x1 = 0) (h2: f' a x2 = 0) (h3: x1 + x2 = 1) (h4: x1 * x2 = -1 / a) :
  ∃ x y, ellipse x y ∧ line_eq a x y x1 :=
sorry

end line_intersects_ellipse_l543_543863


namespace find_T_l543_543643

variable (a b c T : ℕ)

theorem find_T (h1 : a + b + c = 84) (h2 : a - 5 = T) (h3 : b + 9 = T) (h4 : 5 * c = T) : T = 40 :=
sorry

end find_T_l543_543643


namespace arrangement_count_l543_543644

def no_adjacent_students_arrangements (teachers students : ℕ) : ℕ :=
  if teachers = 3 ∧ students = 3 then 144 else 0

theorem arrangement_count :
  no_adjacent_students_arrangements 3 3 = 144 :=
by
  sorry

end arrangement_count_l543_543644


namespace weight_of_one_baseball_l543_543807

structure Context :=
  (numberBaseballs : ℕ)
  (numberBicycles : ℕ)
  (weightBicycles : ℕ)
  (weightTotalBicycles : ℕ)

def problem (ctx : Context) :=
  ctx.weightTotalBicycles = ctx.numberBicycles * ctx.weightBicycles ∧
  ctx.numberBaseballs * ctx.weightBicycles = ctx.weightTotalBicycles →
  (ctx.weightTotalBicycles / ctx.numberBaseballs) = 8

theorem weight_of_one_baseball (ctx : Context) : problem ctx :=
sorry

end weight_of_one_baseball_l543_543807


namespace triangle_trig_identity_l543_543491

theorem triangle_trig_identity (A B C : ℝ) (a b c h : ℝ)
  (h1 : c = 2 * a) : 
  sin ((C - A) / 2) + cos ((C + A) / 2) = 1 :=
by
  sorry

end triangle_trig_identity_l543_543491


namespace ellipse_constant_k_product_l543_543860

theorem ellipse_constant_k_product
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (1 / 2) = sqrt (1 - (b/a)^2)) 
  (circle_radius : ℝ) (hx : circle_radius = b) (line : ℝ → ℝ → ℝ) 
  (h4 : ∀ x y : ℝ, line x y = sqrt 7 * x - sqrt 5 * y + 12) 
  (A R P Q M N : ℝ × ℝ)
  (hA : A = (-4, 0))
  (hR : R = (3, 0))
  (hC : ∀ x y : ℝ, (x, y) ∈ P → (x, y) ∈ Q → (x^2 / 16 + y^2 / 12 = 1))
  (hAP : collinear A P M)
  (hAQ : collinear A Q N)
  (hMR : collinear M R)
  (hNR : collinear N R)
  (slopes : ℝ) (k1 k2 : ℝ)
  (hM : M.1 = 16 / 3) (hN : N.1 = 16 / 3)
  (h_k1 : k1 = (M.2 - R.2) / (M.1 - R.1))
  (h_k2 : k2 = (N.2 - R.2) / (N.1 - R.1)) :
  k1 * k2 = -12 / 7 :=
sorry

end ellipse_constant_k_product_l543_543860


namespace total_number_of_cantelopes_l543_543828

def number_of_cantelopes_fred : ℕ := 38
def number_of_cantelopes_tim : ℕ := 44

theorem total_number_of_cantelopes : number_of_cantelopes_fred + number_of_cantelopes_tim = 82 := by
  sorry

end total_number_of_cantelopes_l543_543828


namespace th150th_letter_is_B_l543_543657

def pattern := "ABCD".data

def nth_letter_in_pattern (n : ℕ) : Char :=
  let len := pattern.length
  pattern.get n % len

theorem th150th_letter_is_B :
  nth_letter_in_pattern 150 = 'B' :=
by {
  -- This proof is placed here as a placeholder
  sorry
}

end th150th_letter_is_B_l543_543657


namespace card_green_given_green_l543_543280

theorem card_green_given_green :
  let num_green_sides := 2 * 2 + 2 * 1 in
  let num_green_on_both_sides := 2 * 2 in
  num_green_on_both_sides / num_green_sides = 2 / 3 :=
by
  let num_green_sides := 6
  let num_green_on_both_sides := 4
  calc
    num_green_on_both_sides / num_green_sides = 4 / 6 : by rfl
    ... = 2 / 3 : by norm_num

end card_green_given_green_l543_543280


namespace cos_double_angle_sum_l543_543373

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l543_543373


namespace sum_cubes_eq_sum_squares_l543_543040

theorem sum_cubes_eq_sum_squares (n : ℕ) : 
  ∑ k in finset.range (n+1), k^3 = (∑ k in finset.range (n+1), k)^2 := 
sorry

end sum_cubes_eq_sum_squares_l543_543040


namespace cos_double_angle_sum_l543_543372

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l543_543372


namespace revenue_increase_l543_543561

open Real

theorem revenue_increase
  (P Q : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q) :
  let R := P * Q
  let P_new := P * 1.60
  let Q_new := Q * 0.65
  let R_new := P_new * Q_new
  (R_new - R) / R * 100 = 4 := by
sorry

end revenue_increase_l543_543561


namespace conical_pile_wheat_l543_543721

def base_circumference : ℝ := 31.4
def height : ℝ := 1.5
def weight_per_cubic_meter_kg : ℝ := 750
def pi : ℝ := 3.14

noncomputable def base_radius (circumference : ℝ) (pi : ℝ) : ℝ :=
  circumference / (2 * pi)

noncomputable def base_area (radius : ℝ) (pi : ℝ) : ℝ :=
  pi * radius^2

noncomputable def cone_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

noncomputable def weight_in_tons (volume : ℝ) (weight_per_cubic_meter_kg : ℝ) : ℝ :=
  volume * (weight_per_cubic_meter_kg / 1000)

theorem conical_pile_wheat :
  let r := base_radius base_circumference pi,
      area := base_area r pi,
      volume := cone_volume area height,
      weight := weight_in_tons volume weight_per_cubic_meter_kg in
  area = 78.5 ∧ weight ≈ 29.4375 :=
by
  let r := base_radius base_circumference pi
  let area := base_area r pi
  let volume := cone_volume area height
  let weight := weight_in_tons volume weight_per_cubic_meter_kg
  have h_area: area = 78.5 := sorry
  have h_weight: weight ≈ 29.4375 := sorry
  exact ⟨h_area, h_weight⟩

end conical_pile_wheat_l543_543721


namespace terminating_fraction_count_l543_543824

theorem terminating_fraction_count : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 299 ∧ (∃ k, n = 3 * k)) ∧ 
  (∃ (count : ℕ), count = 99) :=
by
  sorry

end terminating_fraction_count_l543_543824


namespace sum_of_cubes_mod_11_l543_543681

theorem sum_of_cubes_mod_11 :
  (Finset.range 10).sum (λ k, (k + 1)^3) % 11 = 0 :=
sorry

end sum_of_cubes_mod_11_l543_543681


namespace temperature_43_l543_543218

theorem temperature_43 (T W Th F : ℝ)
  (h1 : (T + W + Th) / 3 = 42)
  (h2 : (W + Th + F) / 3 = 44)
  (h3 : T = 37) : F = 43 :=
by
  sorry

end temperature_43_l543_543218


namespace cos_double_angle_l543_543402

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l543_543402


namespace rotate_90_degrees_l543_543716

theorem rotate_90_degrees (z : ℂ) (h : z = -8 + 6*Complex.i) :
  Complex.i * z = -6 - 8*Complex.i :=
by 
  rw [h]
  simp [Complex.mul, Complex.i_mul_i]
  sorry

end rotate_90_degrees_l543_543716


namespace gcd_fx_x_l543_543079

noncomputable def f (x : ℕ) : ℕ := (5 * x + 3) * (8 * x + 2) * (12 * x + 7) * (3 * x + 11)

theorem gcd_fx_x (x : ℕ) (h : ∃ k : ℕ, x = 18720 * k) : Nat.gcd (f x) x = 462 :=
sorry

end gcd_fx_x_l543_543079


namespace solve_equation1_solve_equation2_l543_543996

-- Define the first equation and state the theorem that proves its roots
def equation1 (x : ℝ) : Prop := 2 * x^2 + 1 = 3 * x

theorem solve_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = 1/2) :=
by sorry

-- Define the second equation and state the theorem that proves its roots
def equation2 (x : ℝ) : Prop := (2 * x - 1)^2 = (3 - x)^2

theorem solve_equation2 (x : ℝ) : equation2 x ↔ (x = -2 ∨ x = 4 / 3) :=
by sorry

end solve_equation1_solve_equation2_l543_543996


namespace sum_m_n_eq_zero_l543_543846

theorem sum_m_n_eq_zero (m n p : ℝ) (h1 : m * n + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 := 
  sorry

end sum_m_n_eq_zero_l543_543846


namespace poly_integer_conditions_l543_543344

theorem poly_integer_conditions (P : ℝ → ℝ) (h_int_coeffs : ∀ n : ℕ, P n ∈ ℤ) :
  (∀ s t : ℝ, P s ∈ ℤ → P t ∈ ℤ → P (s * t) ∈ ℤ) →
  ∃ (n : ℕ) (k : ℤ), P = (λ x, ↑(k : ℤ)) + (λ x, (n : ℕ) • x^n) ∨ P = (λ x, ↑(k : ℤ)) + (λ x, -(n : ℕ) • x^n) :=
begin
  sorry
end

end poly_integer_conditions_l543_543344


namespace circle_center_radius_sum_l543_543938

-- We define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 14 * x + y^2 + 16 * y + 100 = 0

-- We need to find that the center and radius satisfy a specific relationship
theorem circle_center_radius_sum :
  let a' := 7
  let b' := -8
  let r' := Real.sqrt 13
  a' + b' + r' = -1 + Real.sqrt 13 :=
by
  sorry

end circle_center_radius_sum_l543_543938


namespace proof_problem_l543_543455

noncomputable def periodic_fun (x : ℝ) : ℝ := (√3) * Real.sin (2 * x) - Real.cos (2 * x)

-- Definitions for parts (Ⅰ) and (Ⅱ)
def part1_condition (x : ℝ) : Prop := periodic_fun x = 1
def part2_condition (m : ℝ) : Prop := m > 0 ∧ (∀ x : ℝ, periodic_fun (x + m) = 2 * Real.cos (2 * x))

-- The resulting theorem to prove
theorem proof_problem (k : ℤ) :
  (∃ x : ℝ, part1_condition x ∧ (x = k * Real.pi + Real.pi / 6 ∨ x = k * Real.pi + Real.pi / 2)) ∧
  (∃ m : ℝ, part2_condition m ∧ (m = Real.pi / 3)) :=
by
  sorry

end proof_problem_l543_543455


namespace letter_150_in_pattern_l543_543676

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l543_543676


namespace garden_borders_length_l543_543304

theorem garden_borders_length 
  (a b c d e : ℕ)
  (h1 : 6 * 7 = a^2 + b^2 + c^2 + d^2 + e^2)
  (h2 : a * a + b * b + c * c + d * d + e * e = 42) -- This is analogous to the condition
    
: 15 = (4*a + 4*b + 4*c + 4*d + 4*e - 2*(6 + 7)) / 2 :=
by sorry

end garden_borders_length_l543_543304


namespace find_150th_letter_in_pattern_l543_543666

theorem find_150th_letter_in_pattern : 
  (let sequence := "ABCD";
   sequence.length = 4 → 
   sequence[(150 % 4)] = 'B') :=
by
  sorry

end find_150th_letter_in_pattern_l543_543666


namespace main_theorem_l543_543457

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  x^2 - Real.log2 (2 * x + 2)

-- State the main theorem
theorem main_theorem {b : ℝ} (hb1 : 0 < b) (hb2 : b < 1) : 
  f b < f 2 := 
by
  -- Given conditions
  have domain_cond : -1 < b := by linarith [hb1]
  have domain_cond_2 : 2 * b + 2 > 0 := by linarith [hb1, hb2]
  -- Please fill in the proof here
  sorry

end main_theorem_l543_543457


namespace smallest_k_for_perfect_square_l543_543368

theorem smallest_k_for_perfect_square:
  ∃ (k : ℕ), (2018 * 2019 * 2020 * 2021 + k) = n^2 ∧ (∀ m, (2018 * 2019 * 2020 * 2021 + m = n^2) → k ≤ m) :=
begin
  sorry
end

end smallest_k_for_perfect_square_l543_543368


namespace decimal_725th_digit_l543_543257

def fraction_7_div_29 := 7 / 29

noncomputable def decimal_representation : ℚ := 0.2413793103448275862068965517 -- the repeating block of 29 digits representation

theorem decimal_725th_digit:
  let sequence := "2413793103448275862068965517".to_list in
  let repeating_block := sequence in
  let position := 725 in
  position % repeating_block.length = 12 →
  sequence.nth (position % repeating_block.length) = some '0' :=
by
  let sequence := "2413793103448275862068965517".to_list
  let repeating_block := sequence
  let position := 725
  have h1 : position % repeating_block.length = 12 := by sorry
  show sequence.nth 12 = some '0' from sorry

end decimal_725th_digit_l543_543257


namespace f_deriv_at_0_l543_543477

noncomputable def f (x : ℝ) : ℝ := 2 * x * (deriv f 1) + x^2

theorem f_deriv_at_0 : deriv f 0 = -4 :=
by
  sorry

end f_deriv_at_0_l543_543477


namespace sumOfSquares_l543_543211

open Nat 

-- Define condition #1: Positive integers a, b, and c such that a + b + c = 30
def isValidSum (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 30

-- Define condition #2: Sum of GCDs
def isValidGcdSum (a b c : ℕ) : Prop :=
  gcd a b + gcd b c + gcd c a = 11

-- Define condition #3: Exactly two of the numbers are prime
def isExactlyTwoPrimes (a b c : ℕ) : Prop :=
  (Nat.Prime a ∧ Nat.Prime b ∧ ¬Nat.Prime c) ∨ 
  (Nat.Prime a ∧ ¬Nat.Prime b ∧ Nat.Prime c) ∨ 
  (¬Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c)

-- Main theorem to prove
theorem sumOfSquares :
  ∃ (a b c : ℕ), isValidSum a b c ∧ isValidGcdSum a b c ∧ isExactlyTwoPrimes a b c ∧ 
  a^2 + b^2 + c^2 = 398 :=
sorry

end sumOfSquares_l543_543211


namespace evaluate_expression_l543_543038

section ComplexNumbers

noncomputable def i : ℂ := Complex.I

lemma cycle_of_i (n : ℕ) : (i ^ (n % 4)) = (if n % 4 = 0 then 1 else if n % 4 = 1 then i else if n % 4 = 2 then -1 else -i) :=
by sorry

theorem evaluate_expression : i ^ 45 - i ^ 305 = 0 :=
by
  have h45 : i ^ 45 = i ^ (45 % 4) := (congr_arg (λ x, i ^ x) (nat.mod_eq_of_lt (nat.lt_of_lt_of_le (show 45 % 4 < 4, by norm_num) (le_refl 4)))).symm
  have h305 : i ^ 305 = i ^ (305 % 4) := (congr_arg (λ x, i ^ x) (nat.mod_eq_of_lt (show 305 % 4 < 4, by norm_num) (le_refl 4))).symm
  rw [h45, h305]
  have h45' : 45 % 4 = 1 := by norm_num
  have h305' : 305 % 4 = 1 := by norm_num
  rw [h45', h305']
  norm_num
  sorry
end ComplexNumbers

end evaluate_expression_l543_543038


namespace find_angle_l543_543757

def degree := ℝ

def complement (x : degree) : degree := 180 - x

def angle_condition (x : degree) : Prop :=
  x - (complement x / 2) = -18 - 24/60 - 36/3600

theorem find_angle : ∃ x : degree, angle_condition x ∧ x = 47 + 43/60 + 36/3600 :=
by
  sorry

end find_angle_l543_543757


namespace max_queens_20x20_l543_543203

-- Definitions based on conditions
def queen_attacks (q₁ q₂ : ℕ × ℕ) : Prop :=
  let (r1, c1) := q₁
  let (r2, c2) := q₂
  r1 = r2 ∨ c1 = c2 ∨ abs (r1 - r2) = abs (c1 - c2)

-- Goal: Place 23 queens on a 20x20 grid such that each attacks at most one other
noncomputable def maxQueens (n : ℕ) : ℕ := sorry

theorem max_queens_20x20 : maxQueens 20 = 23 :=
sorry

end max_queens_20x20_l543_543203


namespace find_k_l543_543181

theorem find_k 
  (k : ℝ)
  (h1 : ∀ x y : ℝ, y = 2 * x + 3 ↔ (x, y) ∈ {(1, 5)}) 
  (h2 : ∀ x y : ℝ, y = k * x + 4 ↔ (x, y) ∈ {(1, 5)}) 
  (h_intersect : (1, 5) ∈ {(1, 5)}) :
  k = 1 := 
by
  sorry

end find_k_l543_543181


namespace length_of_midsegment_l543_543147

/-- Given a quadrilateral ABCD where sides AB and CD are parallel with lengths 7 and 3 
    respectively, and the other sides BC and DA are of lengths 5 and 4 respectively, 
    prove that the length of the segment joining the midpoints of sides BC and DA is 5. -/
theorem length_of_midsegment (A B C D : ℝ × ℝ)
  (HAB : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 7 ∧ B.2 = 0)
  (HBC : dist B C = 5)
  (HCD : dist C D = 3)
  (HDA : dist D A = 4)
  (Hparallel : B.2 = 0 ∧ D.2 ≠ 0 → C.2 = D.2) :
  dist ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ((A.1 + D.1) / 2, (A.2 + D.2) / 2) = 5 :=
sorry

end length_of_midsegment_l543_543147


namespace cannot_finish_fourth_l543_543581

variables P Q R S T U : ℕ

-- Conditions
def valid_race_order (P Q R S T U : ℕ -> ℕ) : Prop :=
  (P > R) ∧ (P > S) ∧
  (Q > S) ∧ (Q < U) ∧
  (U < P) ∧ (U > T) ∧
  (T < Q)

theorem cannot_finish_fourth (P Q R S T U : ℕ -> ℕ) (h : valid_race_order P Q R S T U) :
  ∀ (x : ℕ), 4.x ∉ {P, S} :=
sorry

end cannot_finish_fourth_l543_543581


namespace triangle_function_linear_l543_543041

theorem triangle_function_linear (f: ℝ → ℝ) (h1: ∀ {a b c: ℝ}, 0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
  ((a + b > c ∧ a + c > b ∧ b + c > a) ↔ (f(a) + f(b) > f(c) ∧ f(a) + f(c) > f(b) ∧ f(b) + f(c) > f(a)))):
  ∃ c > 0, ∀ x > 0, f(x) = c * x := by
  sorry

end triangle_function_linear_l543_543041


namespace general_formula_for_a_sum_of_series_T_l543_543882

noncomputable def S (n : ℕ) : ℝ := (3^(n+1) - 3) / 2
noncomputable def a (n : ℕ) : ℝ := 3^n
noncomputable def b (n : ℕ) : ℝ := 2 * Real.log 3 (a n)

theorem general_formula_for_a (n : ℕ) : a n = 3^n := by
  simp [a]

theorem sum_of_series_T (n : ℕ) : 
  (∑ i in Finset.range n, (-1) ^ i * a (i+1) + b (i+1)) = 
  ((-(-3)^(n+1) - 3) / 4 + n^2) := by
  sorry

end general_formula_for_a_sum_of_series_T_l543_543882


namespace celina_paid_multiple_of_diego_l543_543611

theorem celina_paid_multiple_of_diego
  (D : ℕ) (x : ℕ)
  (h_total : (x + 1) * D + 1000 = 50000)
  (h_positive : D > 0) :
  x = 48 :=
sorry

end celina_paid_multiple_of_diego_l543_543611


namespace sum_of_first_six_terms_l543_543163

theorem sum_of_first_six_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (hS : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 2 = 2 → S 4 = 10 → S 6 = 24 := 
by
  intros h1 h2
  sorry

end sum_of_first_six_terms_l543_543163


namespace directed_sequence_exists_l543_543427

-- Definitions of the conditions in the problem
variable (n : Nat)
variable (P : Fin n → Type) 
variable (dir : ∀ i j : Fin n, i ≠ j → Prop)

-- Theorem statement translating the problem's requirements
theorem directed_sequence_exists (h : ∀ i j : Fin n, i ≠ j → (dir i j ∨ dir j i)) :
  ∃ (σ : Fin n → Fin n), ∀ i j : Fin n, i < j → dir (σ i) (σ j) :=
by
  sorry -- The proof, which is basically the induction process, is omitted

end directed_sequence_exists_l543_543427


namespace cos_double_angle_l543_543400

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l543_543400


namespace hyperbola_eccentricity_proof_l543_543853

def hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (area_cond : ∀ (c: ℝ), (area OFQ = 4 * (area OPQ) → ∃ e, (e = c / a))) : ℝ :=
sqrt 3

theorem hyperbola_eccentricity_proof :
  ∀ (a b : ℝ), 
  (a > 0) → 
  (b > 0) → 
  (∀ (c : ℝ), 
    (area (OFQ := [origin, focus, Q]) = 4 * area (OPQ := [origin, P, Q])) → 
    (c^2 = a^2 + b^2) → 
    ∃ e, e = sqrt 3) : 
  ∀ a b, hyperbola_eccentricity a b = sqrt 3 :=
begin
  intros a b ha hb area_cond,
  sorry
end

end hyperbola_eccentricity_proof_l543_543853


namespace initial_gain_percentage_l543_543730

-- Definitions
def CP : ℝ := 1000
def NCP : ℝ := CP * 0.95
def NSP : ℝ := 1.1 * NCP
def SP : ℝ := NSP + 5
def P : ℝ := SP - CP
def PP : ℝ := (P / CP) * 100

-- Statement of the theorem
theorem initial_gain_percentage : PP = 5 := 
by
  -- Proof goes here (Skipping as only the statement is required)
  sorry

end initial_gain_percentage_l543_543730


namespace bacon_strips_needed_l543_543327

theorem bacon_strips_needed (plates : ℕ) (eggs_per_plate : ℕ) (bacon_per_plate : ℕ) (customers : ℕ) :
  eggs_per_plate = 2 →
  bacon_per_plate = 2 * eggs_per_plate →
  customers = 14 →
  plates = customers →
  plates * bacon_per_plate = 56 := by
  sorry

end bacon_strips_needed_l543_543327


namespace radius_of_inscribed_circle_l543_543146

theorem radius_of_inscribed_circle (AC BC : ℝ) (CM : ℝ) (x : ℝ)
  (isosceles : AC = BC)
  (height_CM : CM = 20)
  (ratio : AB = 4 * x ∧ AC = 3 * x) :
  let AB := 4 * x in
  let AM := 2 * x in
  let OM := (2 / 5) * CM in
  OM = 8 :=
by
  sorry

end radius_of_inscribed_circle_l543_543146


namespace box_weight_difference_l543_543970

theorem box_weight_difference:
  let w1 := 2
  let w2 := 3
  let w3 := 13
  let w4 := 7
  let w5 := 10
  (max (max (max (max w1 w2) w3) w4) w5) - (min (min (min (min w1 w2) w3) w4) w5) = 11 :=
by
  sorry

end box_weight_difference_l543_543970


namespace sequence_a_2017_eq_2_l543_543462

noncomputable def a : ℕ → ℝ
| 0       := 2
| (n + 1) := 1 - 1 / a n

theorem sequence_a_2017_eq_2 : a 2017 = 2 := 
by sorry

end sequence_a_2017_eq_2_l543_543462


namespace interior_diagonals_count_l543_543791

-- We will define the polyhedron's properties first.
def polyhedron (V : Type) := ∃ (vertices : Finset V), vertices.card = 15 ∧
  ∀ (v : V), (∃ (adj : Finset V), adj.card = 6 ∧ ∀ u ∈ adj, u ≠ v)

-- Then we will prove the statement about the number of interior diagonals.
theorem interior_diagonals_count (P : Type) [polyhedron P] : 
      (UniqueInteriorDiagonals P) = 60 :=
by
  sorry

end interior_diagonals_count_l543_543791


namespace angles_OAP_OAQ_l543_543508

noncomputable def ABC : Type := triangle -- Assume there is a triangle type
variables (A B C P Q : point) (O : circle.center)

-- Conditions
def angle_ACB := 65
def angle_ABC := 70

def BP_eq_AB := BP = AB
def CQ_eq_AC := CQ = AC

-- Definitions related to the extensions and circle center
def B_between_P_and_C := between B P C
def C_between_B_and_Q := between C B Q
def circle_center := center_circle_through A P Q O -- Assume a circle center definition

-- Theorem to prove the angles
theorem angles_OAP_OAQ : 
  angle P O A = 57 + 30 / 60 ∧ angle Q O A = 55 :=
sorry

end angles_OAP_OAQ_l543_543508


namespace meet_times_l543_543553

noncomputable theory
open_locale classical

variables (S x y z : ℝ)
variables (meet_time_mh meet_time_mn : ℝ) -- times when Mikhail meets Hariton and Nikolai

-- Conditions from the problem
def condition1 := true -- Mikhail departs from Berdsk to Cherepanovo at 8:00 AM (implicitly considered in variables)
def condition2 := true -- Hariton and Nikolai depart from Cherepanovo towards Berdsk at 8:00 AM (implicitly considered in variables)

def condition3 := 1.5 * S - 1.5 * y - 1.5 * x = (1.5 * y - 1.5 * z)
def condition4 := 2 * S - 2 * y - 2 * x = (2 * S - 2 * z - 2 * y)

-- Derived system of equations
def equation1 := x + 2 * y - z = (2/3) * S
def equation2 := 2 * x + y + z = S

-- Proof statement for Mikhail meeting Hariton at 9:48 AM
def proof_mh := meet_time_mh = 9 + 48/60
def proof_mn := meet_time_mn = 10 + 15/60

-- Main theorem to be proved
theorem meet_times (hc3 : condition3) (hc4 : condition4) (heq1 : equation1) (heq2 : equation2) :
  proof_mh ∧ proof_mn :=
begin
  sorry
end

end meet_times_l543_543553


namespace f_inv_undefined_at_1_l543_543481

def f (x : ℝ) : ℝ := (x - 2) / (x - 5)

noncomputable def f_inv (x : ℝ) : ℝ := (2 - 5 * x) / (1 - x)

theorem f_inv_undefined_at_1 : ∀ x, (1 - x = 0) → (f_inv x).denom = 0 := 
begin
  intros x hx,
  have : (1 - x) = 0, from hx,
  rw this,
  sorry, 
end

end f_inv_undefined_at_1_l543_543481


namespace parallelepiped_volume_l543_543897

open Real
open EuclideanSpace

variables {a b : ℝ^3}
variables (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : angle a b = π / 4)

theorem parallelepiped_volume :
  abs (a • ((b + 2 * b × a) × b)) = 1 :=
sorry

end parallelepiped_volume_l543_543897


namespace distance_AC_is_8_or_2_l543_543865

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.dist p q

theorem distance_AC_is_8_or_2 :
  ∀ (A B C : ℝ × ℝ) (on_line : ∀ {X Y Z : ℝ × ℝ}, X ≠ Y → Y ≠ Z →
        Y ≠ X → (distance X Y + distance Y Z = distance X Z ∨ distance Y X + distance X Z = distance Y Z)),
  distance A B = 5 ∧ distance B C = 3 →
  (distance A C = 8 ∨ distance A C = 2) :=
by intro A B C h_dist h_AB_BC; sorry

end distance_AC_is_8_or_2_l543_543865


namespace lillian_candies_l543_543548

theorem lillian_candies (initial_candies : ℕ) (additional_candies : ℕ) (total_candies : ℕ) :
  initial_candies = 88 → additional_candies = 5 → total_candies = initial_candies + additional_candies → total_candies = 93 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end lillian_candies_l543_543548


namespace part1a_part1b_part2_l543_543342

def min (a b : ℝ) := if a < b then a else b
def max (a b : ℝ) := if a < b then b else a

theorem part1a : min (-2) (-3/2) = -2 := sorry
theorem part1b (a : ℝ) : max (a^2 - 1) (a^2) = a^2 := sorry
theorem part2 (m : ℝ) : min m (m - 3) + (3/2) * max (-1/2 * m) (-1/2 * (m - 1)) = -4 → m = -13 := sorry

end part1a_part1b_part2_l543_543342


namespace girl_buys_roses_l543_543727

theorem girl_buys_roses 
  (x y : ℤ)
  (h1 : y = 1)
  (h2 : x > 0)
  (h3 : (200 : ℤ) / (x + 10) < (100 : ℤ) / x)
  (h4 : (80 : ℤ) / 12 = ((100 : ℤ) / x) - ((200 : ℤ) / (x + 10))) :
  x = 5 ∧ y = 1 :=
by
  sorry

end girl_buys_roses_l543_543727


namespace distance_from_T_to_face_ABC_eq_2_sqrt_6_l543_543973

theorem distance_from_T_to_face_ABC_eq_2_sqrt_6 (A B C T : ℝ × ℝ × ℝ)
  (h_perp : ∀ (X Y Z : ℝ × ℝ × ℝ), X ≠ Y → Y ≠ Z → Z ≠ X →
    (X - Y) ⬝ (X - Z) = 0)
  (h_TA : dist T A = 12)
  (h_TB : dist T B = 12)
  (h_TC : dist T C = 6) :
  distance_to_plane T A B C = 2 * real.sqrt 6 := 
sorry

end distance_from_T_to_face_ABC_eq_2_sqrt_6_l543_543973


namespace no_real_intersection_l543_543952

theorem no_real_intersection (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x * f y = y * f x) 
  (h2 : f 1 = -1) : ¬∃ x : ℝ, f x = x^2 + 1 :=
by
  sorry

end no_real_intersection_l543_543952


namespace locus_of_centers_l543_543338

-- The Lean 4 statement
theorem locus_of_centers (a b : ℝ) 
  (C1 : (x y : ℝ) → x^2 + y^2 = 1)
  (C2 : (x y : ℝ) → (x - 3)^2 + y^2 = 25) :
  4 * a^2 + 4 * b^2 - 52 * a - 169 = 0 :=
sorry

end locus_of_centers_l543_543338


namespace probability_multiple_of_3_or_4_l543_543619

theorem probability_multiple_of_3_or_4 :
  let numbers := {n | 1 ≤ n ∧ n ≤ 30},
      multiples_of_3 := {n | n ∈ numbers ∧ n % 3 = 0},
      multiples_of_4 := {n | n ∈ numbers ∧ n % 4 = 0},
      multiples_of_12 := {n | n ∈ numbers ∧ n % 12 = 0},
      favorable_outcomes := multiples_of_3 ∪ multiples_of_4,
      double_counted_outcomes := multiples_of_12,
      total_favorable_outcomes := set.card favorable_outcomes - set.card double_counted_outcomes,
      total_outcomes := set.card numbers in
  total_favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_multiple_of_3_or_4_l543_543619


namespace distance_calculation_l543_543117

noncomputable def distance_point_to_line : ℝ :=
  let ρ := 2 * Real.sqrt 2
  let θ := 7 * Real.pi / 4
  let r := 2  -- Coefficient of the angle in polar to Cartesian conversion
  let A_x := ρ * Real.cos θ
  let A_y := ρ * Real.sin θ
  let l_A := -1
  let l_B := 1
  let l_C := -1
  (Real.abs ((l_A * A_x) + (l_B * A_y) + l_C) / (Real.sqrt (l_A^2 + l_B^2)))

theorem distance_calculation:
  distance_point_to_line = 5 * Real.sqrt 2 / 2 :=
by
  sorry

end distance_calculation_l543_543117


namespace first_class_circular_permutations_second_class_circular_permutations_l543_543815

section CircularPermutations

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def perm_count (a b c : ℕ) : ℕ :=
  factorial (a + b + c) / (factorial a * factorial b * factorial c)

theorem first_class_circular_permutations : perm_count 2 2 4 / 8 = 52 := by
  sorry

theorem second_class_circular_permutations : perm_count 2 2 4 / 2 / 4 = 33 := by
  sorry

end CircularPermutations

end first_class_circular_permutations_second_class_circular_permutations_l543_543815


namespace cannot_determine_orange_groups_l543_543645

-- Definitions of the conditions
def oranges := 87
def bananas := 290
def bananaGroups := 2
def bananasPerGroup := 145

-- Lean statement asserting that the number of groups of oranges 
-- cannot be determined from the given conditions
theorem cannot_determine_orange_groups:
  ∀ (number_of_oranges_per_group : ℕ), 
  (bananasPerGroup * bananaGroups = bananas) ∧ (oranges = 87) → 
  ¬(∃ (number_of_orange_groups : ℕ), oranges = number_of_oranges_per_group * number_of_orange_groups) :=
by
  sorry -- Since we are not required to provide the proof here

end cannot_determine_orange_groups_l543_543645


namespace log_sqrt3_of_27sqrt3_eq_7_l543_543803

theorem log_sqrt3_of_27sqrt3_eq_7 : 
  log ( (sqrt 3) : ℝ ) ( 27 * sqrt 3 ) = 7 :=
by
  sorry

end log_sqrt3_of_27sqrt3_eq_7_l543_543803


namespace percentage_difference_is_50_percent_l543_543552

-- Definitions of hourly wages
def Mike_hourly_wage : ℕ := 14
def Phil_hourly_wage : ℕ := 7

-- Calculating the percentage difference
theorem percentage_difference_is_50_percent :
  (Mike_hourly_wage - Phil_hourly_wage) * 100 / Mike_hourly_wage = 50 :=
by
  sorry

end percentage_difference_is_50_percent_l543_543552


namespace tangent_circle_l543_543502

variables {A B C P X E F K L O : Type}

def is_acute (Δ : Triangle A B C) : Prop :=
  ∀ a b c : ℝ, ∠a < π / 2 ∧ ∠b < π / 2 ∧ ∠c < π / 2

def circumcircle (T : Triangle A B C) : Circle O := sorry

def tangent_at (C : Circle O) (A : Point) (CB : Line) : Point :=
  intersection_tangent_extend CB C A

def on_segment (X P : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = t • P

def right_angle (AXP : Point) : Prop :=
  angle A X P = π / 2

def on_side (P : Point) (AB AC : Line) : Prop :=
  line_contains AB P ∨ line_contains AC P

def equal_angles (EXP ACX FXO ABX : Angle) : Prop :=
  EXP = ACX ∧ FXO = ABX

def intersect_circ (EF Circ : Line) : Point :=
  intersection_line_circle EF Circ

theorem tangent_circle (A B C K L X O P E F : Point) (AB AC : Line) (Δ : Triangle A B C) :
  is_acute Δ →
  circumcircle Δ = ⊙O →
  tangent_at ⊙O A (CB : Line) = P →
  on_segment X P →
  right_angle (angle A X P) →
  on_side E (AB : Line) →
  on_side F (AC : Line) →
  equal_angles (angle E X P) (angle A C X) (angle F X O) (angle A B X) →
  (EF : Line) intersects ⊙O at K L →
  tangent ⊙O K L X (OP : Line) := 
    sorry

end tangent_circle_l543_543502


namespace num_real_roots_f_l543_543346

noncomputable def f (x : ℝ) := log x - 2 * x + 6

theorem num_real_roots_f :
  ∀ f : ℝ → ℝ,
  (∀ x, f x = log x - 2 * x + 6) →
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ (∀ c, f c = 0 → (c = a ∨ c = b))) :=
by
  sorry

end num_real_roots_f_l543_543346


namespace find_150th_letter_l543_543667

def repeating_sequence : String := "ABCD"

def position := 150

theorem find_150th_letter :
  repeating_sequence[(position % 4) - 1] = 'B' := 
sorry

end find_150th_letter_l543_543667


namespace alternate_interior_angles_converse_l543_543755

theorem alternate_interior_angles_converse:
  (∀ (l1 l2 : Line) (α β : Angle), Parallel l1 l2 → AlternateInteriorAngle l1 l2 α β → α = β) →
  (∀ (l1 l2 : Line) (α β : Angle), α = β ∧ AlternateInteriorAngle l1 l2 α β → Parallel l1 l2) :=
by
  intros h1 l1 l2 α β h2
  sorry

end alternate_interior_angles_converse_l543_543755


namespace ball_hits_ground_at_l543_543597

variable (t : ℚ) 

def height_eqn (t : ℚ) : ℚ :=
  -16 * t^2 + 30 * t + 50

theorem ball_hits_ground_at :
  (height_eqn t = 0) -> t = 47 / 16 :=
by
  sorry

end ball_hits_ground_at_l543_543597


namespace initial_principal_is_979_02_l543_543817

-- Define the conditions
def A : ℝ := 1120
def r : ℝ := 0.06
def t : ℝ := 2.4

-- Define the formula for P given the conditions
def P : ℝ := A / (1 + r * t)

-- State the theorem to be proved
theorem initial_principal_is_979_02 : P ≈ 979.02 := by
  sorry

end initial_principal_is_979_02_l543_543817


namespace sum_Q_alpha_squared_l543_543160

noncomputable def P (x : ℂ) : ℂ := x^2020 + x + 2
def Q (x : ℂ) : ℂ := -- definition of Q can be complex
  sorry

theorem sum_Q_alpha_squared (α : ℂ) :
  P(α) = 4 →
  (∑ (i : Fin 2020), (Q(α^2)^2)) = 2020 * 2^2019 :=
by
  sorry

end sum_Q_alpha_squared_l543_543160


namespace intersection_A_B_l543_543439

-- Definitions based on the conditions
def A : Set ℝ := {x | x ≥ 0}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

-- Proof statement
theorem intersection_A_B :
  (A ∩ (B : Set ℝ)) = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_A_B_l543_543439


namespace johns_weekly_earnings_increase_l543_543519

noncomputable def percentageIncrease (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem johns_weekly_earnings_increase :
  percentageIncrease 30 40 = 33.33 :=
by
  sorry

end johns_weekly_earnings_increase_l543_543519


namespace log_base3_0_216_l543_543802

noncomputable def log10 : ℝ → ℝ := sorry -- Let’s assume we have a log10 function available

-- Main theorem to prove
theorem log_base3_0_216 : Real.log 3 0.216 = -1.395 := by
  -- Define intermediate logarithms with base 10 approximations for 5 and 3
  let log10_5 : ℝ := 0.6990
  let log10_3 : ℝ := 0.4771
  -- Using the change of base formula
  let log3_5 := log10_5 / log10_3
  -- Calculate the desired log value
  have : Real.log 3 5 = log3_5, by sorry
  -- Final computation
  show Real.log 3 0.216 = 3 - 3 * log3_5, by sorry
  show 3 - 3 * log3_5 = -1.395, by sorry

end log_base3_0_216_l543_543802


namespace magnitude_of_projection_l543_543120

variables {V : Type*} [inner_product_space ℝ V]

def vec_projection (a b : V) : V := ((inner_product a b) / (inner_product b b)) • b

theorem magnitude_of_projection (a b : V) (h : 0 < inner_product b b) 
  (angle_eq : real.angle_of_vectors a b = real.pi / 3) :
  ∥vec_projection a b∥ = (1/2) * ∥a∥ :=
by
  sorry

end magnitude_of_projection_l543_543120


namespace calculate_product_l543_543767

theorem calculate_product :
  6^5 * 3^5 = 1889568 := by
  sorry

end calculate_product_l543_543767


namespace university_minimum_spend_l543_543262

-- Define the conditions
def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def box_cost : ℝ := 1.30
def total_volume : ℕ := 3060000

-- Calculate the minimum amount the university must spend on boxes
theorem university_minimum_spend : 
  let volume_per_box := box_length * box_width * box_height,
      num_boxes := (total_volume + volume_per_box - 1) / volume_per_box in
  num_boxes * box_cost = 663 := by
  sorry

end university_minimum_spend_l543_543262


namespace simplify_expression_l543_543579

theorem simplify_expression (n : ℤ) : 
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+4)) = 29 / 48 := by
  sorry

end simplify_expression_l543_543579


namespace converse_equiv_l543_543598

/-
Original proposition: If ¬p then ¬q
Converse of the original proposition: If ¬q then ¬p
Equivalent proposition to the converse proposition: If p then q

We need to prove that (If ¬p then ¬q) implies (If q then p)
-/
theorem converse_equiv (p q : Prop) : (¬p → ¬q) → (p → q) :=
sorry

end converse_equiv_l543_543598


namespace green_other_side_l543_543281
open Classical

def numBlueBlueCards : Nat := 4
def numBlueGreenCards : Nat := 2
def numGreenGreenCards : Nat := 2

noncomputable def totalGreenSides : Nat := (2 * numBlueGreenCards) + (2 * numGreenGreenCards)
noncomputable def greenGreenSides : Nat := 2 * numGreenGreenCards

theorem green_other_side (h : totalGreenSides = 6 ∧ greenGreenSides = 4) :
  probability_that_other_side_is_green : ℚ :=
  by
    unfold totalGreenSides at h
    unfold greenGreenSides at h
    have h1 : totalGreenSides = 6 := h.1
    have h2 : greenGreenSides = 4 := h.2
    
    sorry
  
#check green_other_side

end green_other_side_l543_543281


namespace scientific_notation_of_29000_l543_543589

theorem scientific_notation_of_29000 : 
  (∃ a n : ℝ, 29000 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n ∈ ℤ) := 
by
  use [2.9, 4]
  split
  { 
    norm_num,
  }
  split
  { 
    norm_num,
  }
  {
    split
    {
      norm_num
    }
    exact int.cast (-4)
  }

end scientific_notation_of_29000_l543_543589


namespace min_value_abs_sum_pqr_inequality_l543_543082

theorem min_value_abs_sum (x : ℝ) : |x + 1| + |x - 2| ≥ 3 :=
by
  sorry

theorem pqr_inequality (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := 
by
  have f_min : ∀ x, |x + 1| + |x - 2| ≥ 3 := min_value_abs_sum
  sorry

end min_value_abs_sum_pqr_inequality_l543_543082


namespace donna_pizza_slices_left_l543_543034

def total_slices_initial : ℕ := 12
def slices_eaten_lunch (slices : ℕ) : ℕ := slices / 2
def slices_remaining_after_lunch (slices : ℕ) : ℕ := slices - slices_eaten_lunch slices
def slices_eaten_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices / 3
def slices_remaining_after_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices - slices_eaten_dinner slices
def slices_shared_friend (slices : ℕ) : ℕ := slices_remaining_after_dinner slices / 4
def slices_remaining_final (slices : ℕ) : ℕ := slices_remaining_after_dinner slices - slices_shared_friend slices

theorem donna_pizza_slices_left : slices_remaining_final total_slices_initial = 3 :=
sorry

end donna_pizza_slices_left_l543_543034


namespace pen_defect_probability_l543_543495

theorem pen_defect_probability :
  ∀ (n m : ℕ) (k : ℚ), n = 12 → m = 4 → k = 2 → 
  (8 / 12) * (7 / 11) = 141 / 330 := 
by
  intros n m k h1 h2 h3
  sorry

end pen_defect_probability_l543_543495


namespace find_n_l543_543741

theorem find_n 
  (N : ℕ) 
  (hn : ¬ (N = 0))
  (parts_inv_prop : ∀ k, 1 ≤ k → k ≤ n → N / (k * (k + 1)) = x / (n * (n + 1))) 
  (smallest_part : (N : ℝ) / 400 = N / (n * (n + 1))) : 
  n = 20 :=
sorry

end find_n_l543_543741


namespace natural_number_reverse_difference_divisible_by_9_l543_543711

theorem natural_number_reverse_difference_divisible_by_9 (N : ℕ) (N' : ℕ) (h : ∀ (d : string), d ∈ repr N ↔ d ∈ repr N') : 
  (N - N') % 9 = 0 := 
sorry

end natural_number_reverse_difference_divisible_by_9_l543_543711


namespace total_borders_length_is_15_l543_543305

def garden : ℕ × ℕ := (6, 7)
def num_beds : ℕ := 5
def total_length_of_borders (length width : ℕ) : ℕ := 15

theorem total_borders_length_is_15 :
  ∃ a b : ℕ, 
  garden = (a, b) ∧ 
  num_beds = 5 ∧ 
  total_length_of_borders a b = 15 :=
by
  use (6, 7)
  rw [garden]
  rw [num_beds]
  exact ⟨rfl, rfl, sorry⟩

end total_borders_length_is_15_l543_543305


namespace tank_overflow_time_l543_543972

theorem tank_overflow_time :
  (let rate_X := 1 in
   let rate_Y := 2 in
   let total_volume := 1 in
   let time_Y_open := λ t : ℝ, t - 0.25 in
   let volume_filled := λ t : ℝ, rate_X * t + rate_Y * (time_Y_open t) in
   ∀ t : ℝ, volume_filled t = total_volume → t = 0.5) :=
begin
  sorry
end

end tank_overflow_time_l543_543972


namespace john_height_proof_l543_543936

theorem john_height_proof :
  ∀ (building_height shadow_building shadow_john : ℝ), 
  building_height = 60 ∧ 
  shadow_building = 20 ∧ 
  shadow_john = 18 / 12 →
  (shadow_john * (building_height / shadow_building)) = 4.5 :=
by
  intros building_height shadow_building shadow_john h
  cases h with hb hs
  cases hs with hs1 hs2
  rw [hs2]
  rw [hb, hs1]
  norm_num
  rw [← div_eq_inv_mul 18 12] 
  norm_num
  rfl

end john_height_proof_l543_543936


namespace simplify_expression_l543_543224

theorem simplify_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) =
  (3 * (x - (7 + Real.sqrt 37) / 6) * (x - (7 - Real.sqrt 37) / 6)) / ((x - 3) * (x + 2)) :=
by
  sorry

end simplify_expression_l543_543224


namespace min_value_of_z_l543_543171

noncomputable def z_min_value := (60 : ℝ) / 13

theorem min_value_of_z (z : ℂ) (h : complex.abs (z - 12) + complex.abs (z - complex.I * 5) = 13) : 
  complex.abs z = z_min_value :=
sorry

end min_value_of_z_l543_543171


namespace true_propositions_l543_543121

def z : ℂ := 2 - complex.I

theorem true_propositions :
  (conj z = 2 + complex.I) ∧ (z^2 = 3 - 4 * complex.I) :=
by
  -- proofs go here
  sorry

end true_propositions_l543_543121


namespace th150th_letter_is_B_l543_543660

def pattern := "ABCD".data

def nth_letter_in_pattern (n : ℕ) : Char :=
  let len := pattern.length
  pattern.get n % len

theorem th150th_letter_is_B :
  nth_letter_in_pattern 150 = 'B' :=
by {
  -- This proof is placed here as a placeholder
  sorry
}

end th150th_letter_is_B_l543_543660


namespace root_power_inequality_l543_543192

variables {a : ℝ} {n m : ℕ}

theorem root_power_inequality (h_pos_n : 0 < n) (h_pos_m : 0 < m) :
  (a > 1 → real.root n (a ^ m) > 1) ∧ (a < 1 → real.root n (a ^ m) < 1) :=
by sorry

end root_power_inequality_l543_543192


namespace equation_parallel_equation_perpendicular_l543_543119

variables {x y : ℝ}

def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x - 5 * y + 14 = 0
def l3 (x y : ℝ) := 2 * x - y + 7 = 0

theorem equation_parallel {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : 2 * x - y + 6 = 0 :=
sorry

theorem equation_perpendicular {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : x + 2 * y - 2 = 0 :=
sorry

end equation_parallel_equation_perpendicular_l543_543119


namespace binomial_prime_div_l543_543568

theorem binomial_prime_div {p : ℕ} {m : ℕ} (hp : Nat.Prime p) (hm : 0 < m) : (Nat.choose (p ^ m) p - p ^ (m - 1)) % p ^ m = 0 := 
  sorry

end binomial_prime_div_l543_543568


namespace dheo_grocery_bill_l543_543795

variable (n_bills n_coins : ℕ) (bill_value coin_value : ℕ)

theorem dheo_grocery_bill : 
  n_bills = 11 → n_coins = 11 → bill_value = 20 → coin_value = 5 → 
  n_bills + n_coins = 24 → 
  n_bills * bill_value + n_coins * coin_value = 275 :=
by 
  intros hn_bills hn_coins hbill_value hcoin_value htotal
  have hab := htotal
  rw [hn_bills, hn_coins, hbill_value, hcoin_value] at hab
  exact hab

end dheo_grocery_bill_l543_543795


namespace max_integral_solution_of_f_eq_3_l543_543080

def f (x : ℝ) : ℝ :=
  if x < 1 then -2 * x + 1 else x^2 - 2 * x

theorem max_integral_solution_of_f_eq_3 : 
  (∃ d : ℤ, d ∈ set_of (λ x, f x = 3) ∧ ∀ y : ℤ, y ∈ set_of (λ x, f x = 3) → y ≤ d) ∧
  d = 3 :=
sorry

end max_integral_solution_of_f_eq_3_l543_543080


namespace turtles_remaining_l543_543728

-- Define the initial number of turtles
def initial_turtles : ℕ := 9

-- Define the number of turtles that climbed onto the log
def climbed_turtles : ℕ := 3 * initial_turtles - 2

-- Define the total number of turtles on the log before any jump off
def total_turtles_before_jumping : ℕ := initial_turtles + climbed_turtles

-- Define the number of turtles remaining after half jump off
def remaining_turtles : ℕ := total_turtles_before_jumping / 2

theorem turtles_remaining : remaining_turtles = 17 :=
  by
  -- Placeholder for the proof
  sorry

end turtles_remaining_l543_543728


namespace cos_of_double_angles_l543_543411

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l543_543411


namespace local_value_of_7_in_diff_l543_543679

-- Definitions based on conditions
def local_value (n : ℕ) (d : ℕ) : ℕ :=
  if h : d < 10 ∧ (n / Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)) % 10 = d then
    d * Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)
  else
    0

def diff (a b : ℕ) : ℕ := a - b

-- Question translated to Lean 4 statement
theorem local_value_of_7_in_diff :
  local_value (diff 100889 (local_value 28943712 3)) 7 = 70000 :=
by sorry

end local_value_of_7_in_diff_l543_543679


namespace a_general_formula_b_geometric_sequence_sum_first_n_terms_l543_543503

open Nat

noncomputable def a (n : ℕ) : ℕ := 2 * n
noncomputable def b (n : ℕ) : ℕ := 2 ^ a n
noncomputable def T (n : ℕ) : ℕ := n^2 + n + (4 * (geom_sum (fun i => 4) n)) / 3

theorem a_general_formula (n : ℕ) : a n = 2 * n := by sorry

theorem b_geometric_sequence (n : ℕ) : 
  (∀ n, (n ≥ 1) → b (n + 1) / b n = 4) := by sorry

theorem sum_first_n_terms (n : ℕ) : 
  (∑ i in range (n + 1), a i + b i) = T n := by sorry

end a_general_formula_b_geometric_sequence_sum_first_n_terms_l543_543503


namespace angle_B48_B49_B47_zero_l543_543527

theorem angle_B48_B49_B47_zero (s : ℝ) (B : ℕ → ℝ × ℝ)
  (h_square : ∀ i, B (i+4) = ((B i.1 + B (i+2).1) / 2, (B i.2 + B (i+2).2) / 2))
  (h_midpoints : B 5 = B 7 ∧ B 6 = B 8)
  (h_centered : ∀ n, B (n+8) = B n) :
  ∠ B 48 B 49 B 47 = 0 := sorry

end angle_B48_B49_B47_zero_l543_543527


namespace michael_thrifty_savings_investment_l543_543967

noncomputable def initial_investment_at_thrifty_savings : ℝ :=
  let x := 720.84 in
  let total_investment := 1500 in
  let thrifty_growth_rate := 1.04 in
  let rich_growth_rate := 1.06 in
  let years := 3 in
  let final_amount := 1738.84 in
  if x * thrifty_growth_rate ^ years + (total_investment - x) * rich_growth_rate ^ years = final_amount then x else 0

theorem michael_thrifty_savings_investment :
  initial_investment_at_thrifty_savings = 720.84 :=
by
  -- detailed proof would go here
  sorry

end michael_thrifty_savings_investment_l543_543967


namespace water_level_decrease_l543_543137

theorem water_level_decrease (h : ∀ (m : ℝ), (m > 0 → +m = m)): ∀ (d : ℝ), (d > 0 → -d = -d) :=
by
  intro d
  intro hd
  rw [←neg_inj] at h
  exact neg_inj.mpr (h d hd)

end water_level_decrease_l543_543137


namespace perfect_square_divisor_probability_l543_543026

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def prime_factors_15_factorial : Prop :=
  factorial 15 = 2 ^ 11 * 3 ^ 6 * 5 ^ 3 * 7 ^ 2 * 11 ^ 1 * 13 ^ 1

theorem perfect_square_divisor_probability : prime_factors_15_factorial →
  let m := 1
  let n := 42
  m + n = 43 :=
by
  intros
  let m := 1
  let n := 42
  exact 43

end perfect_square_divisor_probability_l543_543026


namespace child_l543_543321

-- Definitions of the given conditions
def total_money : ℕ := 35
def adult_ticket_cost : ℕ := 8
def number_of_children : ℕ := 9

-- Statement of the math proof problem
theorem child's_ticket_cost : ∃ C : ℕ, total_money - adult_ticket_cost = C * number_of_children ∧ C = 3 :=
by
  sorry

end child_l543_543321


namespace possible_values_of_a_for_function_decreasing_l543_543851

theorem possible_values_of_a_for_function_decreasing:
  (∀ (a : ℝ), ∃ (f : ℝ → ℝ),
  (∀ (x : ℝ), f(2 - x) = x + 1 / (a - x)) ∧
  (∀ (x1 x2 : ℝ), 1 < x1 → x1 < x2 → x2 < 4 →
    (f x1 - f x2) / (x1 - x2) < -1)) →
  (a = -2 ∨ a = 1) :=
by
admit -- Placeholder for the actual proof

end possible_values_of_a_for_function_decreasing_l543_543851


namespace min_triangle_area_line_MN_fixed_point_l543_543437

open Real

-- Given point P and the parabola
def P := (2 : ℝ, 0 : ℝ)
def parabola (x y : ℝ) := y^2 = 4 * x

-- Define the midpoints M and N
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Area of triangle PMN
def triangle_area (P M N : ℝ × ℝ) : ℝ :=
  abs ((P.1 * (M.2 - N.2) + M.1 * (N.2 - P.2) + N.1 * (P.2 - M.2)) / 2)

-- Problem (I): Minimum area when k1 * k2 = -1
theorem min_triangle_area (k1 k2 : ℝ) (h1 : k1 * k2 = -1)
  (A B C D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hA : parabola A.1 A.2) (hB : parabola B.1 B.2)
  (hC : parabola C.1 C.2) (hD : parabola D.1 D.2)
  (hM : M = midpoint A B) (hN : N = midpoint C D)
  (hA_P : P = (2, 0)) (hB_P : P = (2, 0))
  (hC_P : P = (2, 0)) (hD_P : P = (2, 0)) :
  triangle_area P M N = 4 := 
sorry

-- Problem (II): MN passes through (2, 2) when k1 + k2 = 1
theorem line_MN_fixed_point (k1 k2 : ℝ) (h2 : k1 + k2 = 1)
  (A B C D : ℝ × ℝ) (M N : ℝ × ℝ)
  (hA : parabola A.1 A.2) (hB : parabola B.1 B.2)
  (hC : parabola C.1 C.2) (hD : parabola D.1 D.2)
  (hM : M = midpoint A B) (hN : N = midpoint C D)
  (hM_expr : M.2 = (k1 * (M.1 - 2)))
  (hN_expr : N.2 = ((1 - k2) * (N.1 - 2)))
  (fixed_point : (2, 2)) :
  ∃ y: ℝ, (y = (k1 - k1^2) * (fixed_point.1 - 2) + 2) := 
sorry

end min_triangle_area_line_MN_fixed_point_l543_543437


namespace complement_of_A_l543_543093

theorem complement_of_A (U : Set ℝ) (A : Set ℝ)
  (hU : U = Set.univ) (hA : A = {x | 2 < x ∧ x ≤ 5}) :
  ∀ x, (x ∉ A) = (x ∈ ((Set.Iic 2) ∪ (Set.Ioi 5))) :=
by
  intros
  rw [Set.mem_compl_iff, Set.mem_union, Set.mem_Iic, Set.mem_Ioi]
  sorry

end complement_of_A_l543_543093


namespace find_marked_angle_l543_543751

theorem find_marked_angle 
  (A B C D M N : Type)
  (angle_BAD : ∀ α : ℝ, α = 90)
  (angle_MAN : ∀ β : ℝ, β = 45)
  (angle_AMN : ∀ γ : ℝ, γ = 60) :
  angle A N M = 75 := 
sorry

end find_marked_angle_l543_543751


namespace partI_partII_l543_543869

-- Definitions based on conditions
def a_seq (n : ℕ) : ℕ := 2 * n - 1
def S_seq (n : ℕ) : ℕ := n^2
def b_seq (n : ℕ) : ℚ := (2 * n + 1) / (n^2 * (n + 1)^2)

-- Theorem part (I): Prove that the general term formula for the sequence {a_n} is 2n - 1
theorem partI (n : ℕ) (h_an_pos : ∀ k, a_seq k > 0) (h_Sn_sqrt_arith : ∀ m n, (S_seq m)^(1/2) + (S_seq n)^(1/2) = (S_seq (m+n))^(1/2)) :
  a_seq n = 2 * n - 1 :=
sorry

-- Theorem part (II): Prove that the sum of the first n terms of the sequence {b_n} is (n^2 + 2n) / (n+1)^2
theorem partII (n : ℕ) :
  (finset.sum (finset.range n) (λ k, b_seq k)) = (n^2 + 2*n)/(n+1)^2 :=
sorry

end partI_partII_l543_543869


namespace cosine_identity_l543_543389

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l543_543389


namespace limit_of_function_l543_543782

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2 * Real.pi) ^ 2 / Real.tan (Real.cos x - 1)

-- Define the limit point "2 * Real.pi"
def limit_point : ℝ := 2 * Real.pi

-- Statement of the limit problem
theorem limit_of_function : Filter.Tendsto f (nhds limit_point) (nhds (-2)) := by
  sorry

end limit_of_function_l543_543782


namespace abs_h_eq_2_l543_543810

-- Definitions based on the given conditions
def sum_of_squares_of_roots (h : ℝ) : Prop :=
  let a := 1
  let b := -4 * h
  let c := -8
  let sum_of_roots := -b / a
  let prod_of_roots := c / a
  let sum_of_squares := sum_of_roots^2 - 2 * prod_of_roots
  sum_of_squares = 80

-- Theorem to prove the absolute value of h is 2
theorem abs_h_eq_2 (h : ℝ) (h_condition : sum_of_squares_of_roots h) : |h| = 2 :=
by
  sorry

end abs_h_eq_2_l543_543810


namespace picnic_students_count_l543_543288

theorem picnic_students_count (x : ℕ) (h1 : (x / 2) + (x / 3) + (x / 4) = 65) : x = 60 :=
by
  -- Proof goes here
  sorry

end picnic_students_count_l543_543288


namespace inequality_solution_l543_543209

theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a+3)*x + 2*(a+1) ≥ 0 ↔ 
    (if a ≥ 1 then (x ≥ a+1 ∨ x ≤ 2) else (x ≥ 2 ∨ x ≤ a+1))) :=
begin
  sorry
end

end inequality_solution_l543_543209


namespace xiao_ming_should_use_median_xiao_ming_qualification_l543_543141

/-- Define the condition that there are 13 unique race results -/
def race_results (results : Fin 13 → ℕ) : Prop :=
  Function.Injective results

/-- Define the condition that to qualify, one must be in the top 6 -/
def qualify_top_6 (results : Fin 13 → ℕ) (xiao_ming_result : ℕ) : Prop :=
  (∃ i : Fin 6, xiao_ming_result = results i)

/-- State that the median result is the benchmark Xiao Ming should use -/
theorem xiao_ming_should_use_median (results : Fin 13 → ℕ) (xiao_ming_result : ℕ) 
  (h_results : race_results results) : Prop :=
  let sorted_results := Finset.sort (≤) (Finset.univ.image results)
  xiao_ming_result < sorted_results 6 ∨ xiao_ming_result = sorted_results 6

/-- Main statement combining the conditions -/
theorem xiao_ming_qualification (results : Fin 13 → ℕ) (xiao_ming_result : ℕ) 
  (h_results : race_results results) : Prop :=
  xiao_ming_should_use_median results xiao_ming_result h_results ∧ qualify_top_6 results xiao_ming_result

end xiao_ming_should_use_median_xiao_ming_qualification_l543_543141


namespace mutual_connection_exists_l543_543796

universe u

noncomputable def group (α : Type u) := set α

variables {α : Type u} [fintype α]
variables (A B C : group α)
variables (knows : α → α → Prop)
variable [decidable_rel knows]

/--
Each student knows at least n+1 students from the other groups.
-/
def knows_at_least_n_add_one (n : ℕ) (grp1 grp2 grp3 : group α) :=
  ∀ a ∈ grp1, ∃ (sb ∈ grp2), ∃ (sc ∈ grp3), 
  (card {b | b ∈ grp2 ∧ knows a b} + card {c | c ∈ grp3 ∧ knows a c} ≥ n + 1)

theorem mutual_connection_exists (n : ℕ) (hA : fintype.card A = n) (hB : fintype.card B = n) (hC : wintype.card C = n)
  (h_knows : knows_at_least_n_add_one n A B C):
  ∃ (a ∈ A) (b ∈ B) (c ∈ C), knows a b ∧ knows b c ∧ knows c a :=
sorry

end mutual_connection_exists_l543_543796


namespace sum_of_prime_h_values_is_zero_l543_543365

def h (n : ℕ) : ℕ := n^4 - 500 * n^2 + 625

theorem sum_of_prime_h_values_is_zero : 
  (∑ n in (Finset.filter (λ n, nat.prime (h n)) (Finset.range 1)), h n) = 0 :=
by
  -- Proof goes here.
  sorry

end sum_of_prime_h_values_is_zero_l543_543365


namespace exists_complex_on_line_y_eq_neg_x_l543_543908

open Complex

theorem exists_complex_on_line_y_eq_neg_x :
  ∃ (z : ℂ), ∃ (a b : ℝ), z = a + b * I ∧ b = -a :=
by
  use 1 - I
  use 1, -1
  sorry

end exists_complex_on_line_y_eq_neg_x_l543_543908


namespace swan_count_l543_543053

theorem swan_count (total_birds : ℕ) (fraction_ducks : ℚ):
  fraction_ducks = 5 / 6 →
  total_birds = 108 →
  ∃ (num_swans : ℕ), num_swans = 18 :=
by
  intro h_fraction_ducks h_total_birds
  sorry

end swan_count_l543_543053


namespace find_150th_letter_l543_543671

def repeating_sequence : String := "ABCD"

def position := 150

theorem find_150th_letter :
  repeating_sequence[(position % 4) - 1] = 'B' := 
sorry

end find_150th_letter_l543_543671


namespace permits_cost_l543_543935

-- Definitions based on conditions
def total_cost : ℕ := 2950
def contractor_hourly_rate : ℕ := 150
def contractor_hours_per_day : ℕ := 5
def contractor_days : ℕ := 3
def inspector_discount_rate : ℕ := 80

-- Proving the cost of permits
theorem permits_cost : ∃ (permits_cost : ℕ), permits_cost = 250 :=
by
  let contractor_hours := contractor_days * contractor_hours_per_day
  let contractor_cost := contractor_hours * contractor_hourly_rate
  let inspector_hourly_rate := contractor_hourly_rate - (contractor_hourly_rate * inspector_discount_rate / 100)
  let inspector_cost := contractor_hours * inspector_hourly_rate
  let total_cost_without_permits := contractor_cost + inspector_cost
  let permits_cost := total_cost - total_cost_without_permits
  use permits_cost
  sorry

end permits_cost_l543_543935


namespace minimize_quadratic_expression_l543_543055

noncomputable def quadratic_expression (b : ℝ) : ℝ :=
  (1 / 3) * b^2 + 7 * b - 6

theorem minimize_quadratic_expression : ∃ b : ℝ, quadratic_expression b = -10.5 :=
  sorry

end minimize_quadratic_expression_l543_543055


namespace second_reduction_is_18_point_1_percent_l543_543748

noncomputable def second_reduction_percentage (P : ℝ) : ℝ :=
  let first_price := 0.91 * P
  let second_price := 0.819 * P
  let R := (first_price - second_price) / first_price
  R * 100

theorem second_reduction_is_18_point_1_percent (P : ℝ) : second_reduction_percentage P = 18.1 :=
by
  -- Proof omitted
  sorry

end second_reduction_is_18_point_1_percent_l543_543748


namespace conditionForEuler_EulerLineLoci_l543_543654

section GeometryTriangle

variables {A B C : Type} [Triangle A B C] [Real a α β γ : ℝ]
variables {cos : ℝ → ℝ} {sin : ℝ → ℝ} {cot : ℝ → ℝ}
variables (AS AK : Line α β) (ASM AKA_1 : Triangle)
noncomputable def calculusOfTriangle {α β γ : ℝ} (a : ℝ) : Prop :=
  (cos α) / (sin β * sin γ) = 2 / 3

noncomputable def EulerLineParallelBC (AS AK : Line) (triangle : Triangle) : Prop :=
  AS ∥ AK

theorem conditionForEuler (a : ℝ) (h_cos_sin_condition : calculusOfTriangle α β γ a) : 
  EulerLineParallelBC AS AK triangle :=
sorry

variables {major_axis_A : ℝ} {minor_axis_A : ℝ}
variables {major_axis_S : ℝ} {minor_axis_S : ℝ}
variables {major_axis_M : ℝ}

noncomputable def locusA (a : ℝ) : Prop :=
  major_axis_A = a * sqrt 3 ∧ minor_axis_A = a

noncomputable def locusS (a : ℝ) : Prop :=
  major_axis_S = a * sqrt 3 / 3 ∧ minor_axis_S = a / 3

noncomputable def locusM (a : ℝ) : Prop :=
  major_axis_M = a

theorem EulerLineLoci (a : ℝ) (h_cos_sin_condition : calculusOfTriangle α β γ a) :
  EulerLineParallelBC AS AK triangle → locusA a ∧ locusS a ∧ locusM a :=
sorry

end GeometryTriangle

end conditionForEuler_EulerLineLoci_l543_543654


namespace unit_square_transformation_l543_543940

/-- Definitions of the unit square and the transformation in the xy-plane and uv-plane -/
def unit_square_vertices : set (ℝ × ℝ) := {(0,0), (1,0), (1,1), (0,1)}

def transformation (p : ℝ × ℝ) : ℝ × ℝ := (p.1^2 - p.2^2, 2 * p.1 * p.2)

/-- The set of transformed vertices in the uv-plane -/
def transformed_vertices : set (ℝ × ℝ) := {(0,0), (1,0), (0,2), (-1,0)}

/-- The transformation applied to each side of the square results in parabolic segments -/
theorem unit_square_transformation :
  ∃ curves, 
    curves = {((x, y), transformation (x, y)) | ∃ z ∈ unit_square_vertices, curve_formula z (x, y)} ∧ 
    (image_of_unit_square curves = { (0,0), (1,0), (0,2), (-1,0) } ∪ parabolic_segments) :=
sorry

end unit_square_transformation_l543_543940


namespace min_k_proof_l543_543947

noncomputable def f (k: ℕ) (x: ℝ) : ℝ :=
  let theta := (k*x) / 4
  in (Real.sin theta) ^ 6 + (Real.cos theta) ^ 6

def min_k_satisfies_condition : ℕ :=
  7

theorem min_k_proof (k : ℕ) (h_pos : 0 < k) :
  (∀ (a : ℝ), ∃ (x : ℝ), a < x ∧ x < a + 1 ∧ f k x = min_k_satisfies_condition) →
  k = 7 :=
sorry

end min_k_proof_l543_543947


namespace Tammy_earnings_3_weeks_l543_543585

theorem Tammy_earnings_3_weeks
  (trees : ℕ)
  (oranges_per_tree_per_day : ℕ)
  (oranges_per_pack : ℕ)
  (price_per_pack : ℕ)
  (weeks : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  oranges_per_pack = 6 →
  price_per_pack = 2 →
  weeks = 3 →
  (trees * oranges_per_tree_per_day * weeks * 7) / oranges_per_pack * price_per_pack = 840 :=
by
  intro ht ht12 h6 h2 h3
  -- proof to be filled in here
  sorry

end Tammy_earnings_3_weeks_l543_543585


namespace snow_volume_l543_543966

-- Definitions based on conditions
variables (length : ℝ) (width : ℝ) (depth : ℝ)
variables (V : ℝ)

-- Given conditions
def given_conditions : Prop :=
  length = 30 ∧ width = 3 ∧ depth = 3 / 4

-- The statement we need to prove
theorem snow_volume (h : given_conditions) : V = 30 * 3 * (3 / 4) :=
by
  cases h -- unpack the given conditions
  simp [h] -- simplify the statement using the given conditions
  sorry

end snow_volume_l543_543966


namespace count_4_letter_words_with_E_l543_543471

theorem count_4_letter_words_with_E : 
  let total_words := (5 : ℕ) ^ 4
  let words_without_E := (4 : ℕ) ^ 4
  total_words - words_without_E = 369 := 
by
  let total_words := (5 : ℕ) ^ 4
  let words_without_E := (4 : ℕ) ^ 4
  have h1 : total_words = 625 := by norm_num
  have h2 : words_without_E = 256 := by norm_num
  calc
    total_words - words_without_E
      = 625 - 256 : by rw [h1, h2]
  ... = 369 : by norm_num

end count_4_letter_words_with_E_l543_543471


namespace find_150th_letter_in_pattern_l543_543663

theorem find_150th_letter_in_pattern : 
  (let sequence := "ABCD";
   sequence.length = 4 → 
   sequence[(150 % 4)] = 'B') :=
by
  sorry

end find_150th_letter_in_pattern_l543_543663


namespace expansion_coefficient_l543_543533

theorem expansion_coefficient :
  ∀ (x : ℝ), (∃ (a₀ a₁ a₂ b : ℝ), x^6 + x^4 = a₀ + a₁ * (x + 2) + a₂ * (x + 2)^2 + b * (x + 2)^3) →
  (a₀ = 0 ∧ a₁ = 0 ∧ a₂ = 0 ∧ b = -168) :=
by
  sorry

end expansion_coefficient_l543_543533


namespace smallest_number_l543_543793

def sum_of_decimal_digits (n : ℕ) : ℕ := 
  (Nat.digits 10 n).sum

theorem smallest_number 
  (N : ℕ)
  (h1 : ∃ (seq1 : Fin 2002 → ℕ), sum seq1 = N ∧ ∀ i, sum_of_decimal_digits (seq1 i) = sum_of_decimal_digits (seq1 0))
  (h2 : ∃ (seq2 : Fin 2003 → ℕ), sum seq2 = N ∧ ∀ i, sum_of_decimal_digits (seq2 i) = sum_of_decimal_digits (seq2 0)) :
  N = 10010 :=
sorry

end smallest_number_l543_543793


namespace lucy_initial_money_l543_543182

theorem lucy_initial_money : 
  ∃ (x : ℝ),  (x = 15) ∧ 
       (let doubled := 2 * x in 
       let lost_one_third := doubled - (1 / 3) * doubled in 
       let spent_one_fourth := lost_one_third - (1 / 4) * lost_one_fourth in 
       spent_one_fourth = 15) :=
  sorry

end lucy_initial_money_l543_543182


namespace subtract_base_conversion_l543_543766

theorem subtract_base_conversion :
  let a := 354
      b := 261
      base_a := 9
      base_b := 6
  in (a / base_a^2) * base_a^2 + ((a % base_a^2) / base_a) * base_a + (a % base_a) 
   - ((b / base_b^2) * base_b^2 + ((b % base_b^2) / base_b) * base_b + (b % base_b)) = 183 :=
by
  sorry

end subtract_base_conversion_l543_543766


namespace length_of_AD_l543_543148

-- Define the problem conditions as variables
variable (AB BC CD AD : ℝ)
variable (angle_B : ℝ)
variable (sin_C cos_B : ℝ)

-- Conditions provided in the problem
axiom AB_eq_6 : AB = 6
axiom BC_eq_7 : BC = 7
axiom CD_eq_25 : CD = 25
axiom B_obtuse : angle_B > π / 2 ∧ angle_B < π
axiom sinC_eq_cosB : sin_C = cos_B
axiom cosB_eq_4_div_5 : cos_B = 4 / 5
axiom sinC_eq_4_div_5 : sin_C = 4 / 5

-- Formulate the problem
theorem length_of_AD : AD = 38 :=
by
  sorry

end length_of_AD_l543_543148


namespace hyperbola_asymptotes_l543_543428

theorem hyperbola_asymptotes : 
  ∀ (m n : ℝ), 
    m * n ≠ 0 ∧ 
    (∃e : ℝ, e = 2) ∧ 
    (1, 0) ∈ { p : ℝ × ℝ | p.1^2 = 4 * p.2 } ∧ -- Focus condition
    (∃ f : ℝ × ℝ, f.1^2 / m - f.2^2 / n = 1 ∧ sqrt (m + n) = 1) 
  → (∀ (x y : ℝ), 3*x^2 = y^2 → sqrt 3 * x - y = 0 ∨ sqrt 3 * x + y = 0) :=
by 
  sorry

end hyperbola_asymptotes_l543_543428


namespace cos_double_angle_proof_l543_543415

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l543_543415


namespace sector_area_l543_543443

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = 2 * Real.pi / 5) (hr : r = 20) :
  1 / 2 * r^2 * θ = 80 * Real.pi := by
  sorry

end sector_area_l543_543443


namespace sum_of_cubes_mod_11_l543_543680

theorem sum_of_cubes_mod_11 :
  (Finset.range 10).sum (λ k, (k + 1)^3) % 11 = 0 :=
sorry

end sum_of_cubes_mod_11_l543_543680


namespace card_green_given_green_l543_543279

theorem card_green_given_green :
  let num_green_sides := 2 * 2 + 2 * 1 in
  let num_green_on_both_sides := 2 * 2 in
  num_green_on_both_sides / num_green_sides = 2 / 3 :=
by
  let num_green_sides := 6
  let num_green_on_both_sides := 4
  calc
    num_green_on_both_sides / num_green_sides = 4 / 6 : by rfl
    ... = 2 / 3 : by norm_num

end card_green_given_green_l543_543279


namespace no_visits_l543_543555

theorem no_visits (total_days : ℕ) (f1 f2 f3 : ℕ) :
  total_days = 366 → f1 = 6 → f2 = 8 → f3 = 10 →
  let days_with_visits := (total_days / f1) + (total_days / f2) + (total_days / f3) -
                          (total_days / Nat.lcm f1 f2) - 
                          (total_days / Nat.lcm f1 f3) - 
                          (total_days / Nat.lcm f2 f3) +
                          (total_days / Nat.lcm (Nat.lcm f1 f2) f3) in
  (total_days - days_with_visits) = 257 :=
by
  intros h1 h2 h3 h4
  let days_with_visits := (total_days / f1) + (total_days / f2) + (total_days / f3) -
                         (total_days / Nat.lcm f1 f2) - 
                         (total_days / Nat.lcm f1 f3) - 
                         (total_days / Nat.lcm f2 f3) +
                         (total_days / Nat.lcm (Nat.lcm f1 f2) f3)
  show (total_days - days_with_visits) = 257
  sorry

end no_visits_l543_543555


namespace angle_between_a_b_l543_543861

variables {a b : ℝ^3} (h1 : ‖b‖ = 2 * ‖a‖) (h2 : ‖a‖ = 1) (h3 : a + b ⬝ a = 0)

theorem angle_between_a_b (h1 : ‖b‖ = 2 * ‖a‖) (h2 : ‖a‖ = 1) (h3 : a + b ⬝ a = 0) :
  real.arccos (a ⬝ b / (‖a‖ * ‖b‖)) = 2 * real.pi / 3 :=
sorry

end angle_between_a_b_l543_543861


namespace common_ratio_and_sum_of_geometric_sequence_l543_543912

theorem common_ratio_and_sum_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (a 2 + a 4 = 20) →
  (a 3 + a 5 = 40) →
  (q = 2) ∧ (∀ n : ℕ, ∑ i in finset.range (n + 1), a i = 2^(n + 1) - 2) :=
by {
  sorry
}

end common_ratio_and_sum_of_geometric_sequence_l543_543912


namespace sum_row_numbers_not_distinct_or_diff_2010_l543_543177

theorem sum_row_numbers_not_distinct_or_diff_2010 :
  ∀ (a : Fin 2010 → ℕ), (∀ n, a (Fin.mk n (by linarith)) ∈ Finset.range 2010.succ) → 
  ∃ i j, i ≠ j ∧ ((a i + i) = (a j + j) ∨ |(a i + i) - (a j + j)| = 2010) :=
by
  sorry

end sum_row_numbers_not_distinct_or_diff_2010_l543_543177


namespace line_intersects_circle_at_two_points_midpoint_trajectory_four_points_distance_l543_543426

def circle (C : Type) : Prop :=
  ∃ x y : ℝ, (x + 2)^2 + y^2 = 5

def line (l : Type) (m : ℝ) : Prop :=
  ∃ x y : ℝ, mx - y + 1 + 2m = 0

theorem line_intersects_circle_at_two_points (C : Type) (l : Type) (m : ℝ) (hC : circle C) (hl : line l m) :
  True :=
sorry

theorem midpoint_trajectory (C : Type) (l : Type) (m : ℝ) (hC : circle C) (hl : line l m) :
  ∃ x y : ℝ, (x + 2)^2 + (y - 1 / 2)^2 = 1 / 4 :=
sorry

theorem four_points_distance (C : Type) (l : Type) (hC : circle C) :
  ∀ (m : ℝ), (m > 2 ∨ m < -2) :=
sorry

end line_intersects_circle_at_two_points_midpoint_trajectory_four_points_distance_l543_543426


namespace principal_sum_l543_543311

theorem principal_sum (R P : ℝ) (h : (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81) : P = 900 :=
by
  sorry

end principal_sum_l543_543311


namespace functions_strictly_increasing_l543_543319

def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := (x - 1)^2
def f3 (x : ℝ) : ℝ := Real.exp x
def f4 (x : ℝ) : ℝ := Real.log (x + 1)

theorem functions_strictly_increasing :
  (∀ x1 x2 ∈ Set.Ioi 0, x1 < x2 → f3 x1 < f3 x2) ∧
  (∀ x1 x2 ∈ Set.Ioi 0, x1 < x2 → f4 x1 < f4 x2) :=
by
  sorry

end functions_strictly_increasing_l543_543319


namespace zoo_animals_total_l543_543307

-- Conditions as definitions
def initial_animals : ℕ := 68
def gorillas_sent_away : ℕ := 6
def hippopotamus_adopted : ℕ := 1
def rhinos_taken_in : ℕ := 3
def lion_cubs_born : ℕ := 8
def meerkats_per_cub : ℕ := 2

-- Theorem to prove the resulting number of animals
theorem zoo_animals_total :
  (initial_animals - gorillas_sent_away + hippopotamus_adopted + rhinos_taken_in + lion_cubs_born + meerkats_per_cub * lion_cubs_born) = 90 :=
by 
  sorry

end zoo_animals_total_l543_543307


namespace simplify_and_evaluate_l543_543206

-- Definitions and conditions 
def x := ℝ
def given_condition (x: ℝ) : Prop := x + 2 = Real.sqrt 2

-- The problem statement translated into Lean 4
theorem simplify_and_evaluate (x: ℝ) (h: given_condition x) :
  ((x^2 + 1) / x + 2) / ((x - 3) * (x + 1) / (x^2 - 3 * x)) = Real.sqrt 2 - 1 :=
sorry

end simplify_and_evaluate_l543_543206


namespace multiply_and_simplify_fractions_l543_543771

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end multiply_and_simplify_fractions_l543_543771


namespace four_digit_sandwich_odd_even_l543_543358

theorem four_digit_sandwich_odd_even :
  ∃ (n : ℕ), n = 8 ∧
    ∀ (digits : list ℕ), 
    digits = [1, 2, 3, 4] →
    let four_digit_numbers := {num : list ℕ // num.length = 4 ∧ list.nodup num} in
    let valid_numbers := {num ∈ four_digit_numbers | 
      ∃ i, num.nth i ∈ [2, 4] ∧ 
      num.nth (i-1) ∈ [1, 3] ∧ 
      num.nth (i+1) ∈ [1, 3]} in
    valid_numbers.card = n := 
begin
  use 8,
  split,
  exact rfl,
  sorry,
end

end four_digit_sandwich_odd_even_l543_543358


namespace max_mn_on_parabola_l543_543631

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end max_mn_on_parabola_l543_543631


namespace cos_double_angle_proof_l543_543419

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l543_543419


namespace find_principal_l543_543272

-- Definitions and conditions
variable (P : ℝ)
constant (R : ℝ := 20)
constant (T : ℝ := 2)
constant (difference : ℝ := 360)

-- Simple Interest (SI) calculation
def simple_interest : ℝ := P * (R * T) / 100

-- Compound Interest (CI) calculation
def compound_interest : ℝ := P * ((1 + R / 100)^T - 1)

-- The mathematically equivalent proof problem statement
theorem find_principal :
  compound_interest P 20 2 - simple_interest P 20 2 = 360 -> P = 9000 :=
  sorry

end find_principal_l543_543272


namespace polynomial_inequality_l543_543997

theorem polynomial_inequality (x : ℝ) : -6 * x^2 + 2 * x - 8 < 0 :=
sorry

end polynomial_inequality_l543_543997


namespace cyclist_journey_l543_543291

noncomputable def total_distance := 48 -- in meters
noncomputable def distance_flat := 16 -- in meters
noncomputable def speed_flat := 8 -- in meters per second
noncomputable def distance_ascend := 12 -- in meters
noncomputable def speed_ascend := 6 -- in meters per second
noncomputable def distance_descend := 20 -- in meters
noncomputable def speed_descend := 12 -- in meters per second

noncomputable def time_flat := distance_flat / speed_flat -- in seconds
noncomputable def time_ascend := distance_ascend / speed_ascend -- in seconds
noncomputable def time_descend := distance_descend / speed_descend -- in seconds
noncomputable def total_time := time_flat + time_ascend + time_descend -- in seconds
noncomputable def average_speed := total_distance / total_time -- in meters per second

theorem cyclist_journey :
  total_time = 5.67 ∧ average_speed ≈ 8.47 := 
by 
  -- proof steps would go here
  sorry

end cyclist_journey_l543_543291


namespace volume_of_pyramid_MABCD_l543_543937

variables (d a : ℕ) (A B C D M : Type)

-- Conditions:
-- ABCD is a square
def is_square (A B C D : Type) : Prop :=
  ∀ (a b c d : ℝ), a = b ∧ b = c ∧ c = d

-- DM is perpendicular to the plane of ABCD
def is_perpendicular (DM : Type) (ABCD : Type) : Prop :=
  ∃ (d : ℝ), d ≠ 0

-- DM is an integer
def integer_length_dm (d : ℕ) : Prop := 
  true

-- Lengths of MA, MC, and MB are consecutive even positive integers
def consecutive_even_integers (MA MC MB : ℕ) : Prop :=
  MA + 2 = MC ∧ MC + 2 = MB

-- Volume of the pyramid MABCD
def volume_of_pyramid (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- Prove the volume of pyramid MABCD given the conditions
theorem volume_of_pyramid_MABCD (d : ℕ) (ι : is_square A B C D)
  (h_perpendicular : is_perpendicular DM (A × B × C × D))
  (h_dm : integer_length_dm d)
  (h_consecutive : consecutive_even_integers (classical.some h_dm) (classical.some h_dm + 2) (classical.some h_dm + 4)) :
  volume_of_pyramid (32 + 16 * real.sqrt 2) d = (1/3) * (32 + 16 * real.sqrt 2) * d :=
by
sorry

end volume_of_pyramid_MABCD_l543_543937


namespace count_ways_l543_543057

-- Mathematically equivalent statement
theorem count_ways (n : ℕ) (h_n : n ≥ 10) : (Σ' (i j k : ℕ), (1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n) ∧ (j ≥ i + 4) ∧ (k ≥ j + 5) = (n - 7).choose 3 :=
begin
  sorry

end count_ways_l543_543057


namespace find_b_eq_1001_l543_543441

theorem find_b_eq_1001 :
  (∃ b : ℝ, b * ∑ r in Finset.range 1000, 1 / ((2 * r + 1) * (2 * r + 3)) =
  2 * ∑ r in Finset.range 1000, (r + 1)^2 / ((2 * (r + 1) - 1) * (2 * (r + 1) + 1))) →
  b = 1001 :=
by
  sorry

end find_b_eq_1001_l543_543441


namespace sally_bought_cards_l543_543982

theorem sally_bought_cards (initial_cards : ℕ) (dan_gave : ℕ) (total_cards : ℕ) (cards_bought : ℕ) :
  initial_cards = 27 ∧ dan_gave = 41 ∧ total_cards = 88 → cards_bought = 20 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end sally_bought_cards_l543_543982


namespace new_person_weight_l543_543271

theorem new_person_weight (weights : List ℝ) (len_weights : weights.length = 8) (replace_weight : ℝ) (new_weight : ℝ)
  (weight_diff :  (weights.sum - replace_weight + new_weight) / 8 = (weights.sum / 8) + 3) 
  (replace_weight_eq : replace_weight = 70):
  new_weight = 94 :=
sorry

end new_person_weight_l543_543271


namespace initial_completion_days_l543_543322

theorem initial_completion_days
  (road_length : ℝ)
  (initial_men : ℕ)
  (initial_days_worked : ℕ)
  (completed_road : ℝ)
  (extra_men : ℕ)
  (remaining_work_days : ℕ)
  (total_initial_days : ℕ)
  (total_days : total_initial_days - initial_days_worked = remaining_work_days)
  (total_men : initial_men + extra_men)
  (man_day_per_km : ℝ)
  (total_work : road_length / man_day_per_km)
  (initial_man_days : ℕ)
  (remaining_man_days : ℕ)
  (work_rate_per_man_per_day : man_day_per_km = completed_road / (initial_men * initial_days_worked))
  (initial_man_days_worked : initial_man_days = initial_men * initial_days_worked)
  (remaining_man_days_value : remaining_man_days = remaining_work_days * total_men)
  (remaining_work_eq : remaining_man_days = total_work - initial_man_days) :
  total_initial_days = 30 := 
sorry

end initial_completion_days_l543_543322


namespace part1_ABC_inquality_part2_ABCD_inquality_l543_543195

theorem part1_ABC_inquality (a b c ABC : ℝ) : 
  (ABC <= (a^2 + b^2) / 4) -> 
  (ABC <= (b^2 + c^2) / 4) -> 
  (ABC <= (a^2 + c^2) / 4) -> 
    (ABC < (a^2 + b^2 + c^2) / 6) :=
sorry

theorem part2_ABCD_inquality (a b c d ABC BCD CDA DAB ABCD : ℝ) :
  (ABCD = 1/2 * ((ABC) + (BCD) + (CDA) + (DAB))) -> 
  (ABC < (a^2 + b^2 + c^2) / 6) -> 
  (BCD < (b^2 + c^2 + d^2) / 6) -> 
  (CDA < (c^2 + d^2 + a^2) / 6) -> 
  (DAB < (d^2 + a^2 + b^2) / 6) -> 
    (ABCD < (a^2 + b^2 + c^2 + d^2) / 6) :=
sorry

end part1_ABC_inquality_part2_ABCD_inquality_l543_543195


namespace eduardo_ate_fraction_of_remaining_l543_543248

theorem eduardo_ate_fraction_of_remaining (init_cookies : ℕ) (nicole_fraction : ℚ) (remaining_percent : ℚ) :
  init_cookies = 600 →
  nicole_fraction = 2 / 5 →
  remaining_percent = 24 / 100 →
  (360 - (600 * 24 / 100)) / 360 = 3 / 5 := by
  sorry

end eduardo_ate_fraction_of_remaining_l543_543248


namespace exists_triangle_with_angle_ge_120_l543_543847

variable (P : Fin 6 → ℝ × ℝ)

def no_three_points_collinear : Prop :=
  ∀ (i j k : Fin 6), i ≠ j → j ≠ k → k ≠ i → ¬ collinear (P i) (P j) (P k)

theorem exists_triangle_with_angle_ge_120 (h : no_three_points_collinear P) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ angle (P i) (P j) (P k) ≥ 120 :=
sorry

end exists_triangle_with_angle_ge_120_l543_543847


namespace solve_inequality_l543_543866
open Real

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ < x₁ * f x₂ + x₂ * f x₁

theorem solve_inequality (f : ℝ → ℝ) (h_even : even_function f) (h_decreasing : decreasing_on_nonneg f) :
  ∀ x, 1 + (Exp (1 / 2)) / Exp 1 < x ∧ x < 1 + Exp (1 / 2) → 
  f (log (x - 1)) > f ((1 / Exp 1) ^ (log 2)) := by sorry

end solve_inequality_l543_543866


namespace sum_of_cubes_mod_11_l543_543682

theorem sum_of_cubes_mod_11 : 
  (∑ i in Finset.range 10, (i + 1)^3) % 11 = 0 := 
by
  sorry

end sum_of_cubes_mod_11_l543_543682


namespace total_number_of_cantelopes_l543_543829

def number_of_cantelopes_fred : ℕ := 38
def number_of_cantelopes_tim : ℕ := 44

theorem total_number_of_cantelopes : number_of_cantelopes_fred + number_of_cantelopes_tim = 82 := by
  sorry

end total_number_of_cantelopes_l543_543829


namespace smallest_initial_number_sum_of_digits_l543_543763

theorem smallest_initial_number_sum_of_digits : ∃ (N : ℕ), 
  (0 ≤ N ∧ N < 1000) ∧ 
  ∃ (k : ℕ), 16 * N + 700 + 50 * k < 1000 ∧ 
  (N = 16) ∧ 
  (Nat.digits 10 N).sum = 7 := 
by
  sorry

end smallest_initial_number_sum_of_digits_l543_543763


namespace sum_f_to_2017_l543_543877

noncomputable def f (x : ℕ) : ℝ := Real.cos (x * Real.pi / 3)

theorem sum_f_to_2017 : (Finset.range 2017).sum f = 1 / 2 :=
by
  sorry

end sum_f_to_2017_l543_543877


namespace correct_equations_l543_543987

variable (x y : ℝ)

theorem correct_equations :
  (18 * x = y + 3) ∧ (17 * x = y - 4) ↔ (18 * x = y + 3) ∧ (17 * x = y - 4) :=
by
  sorry

end correct_equations_l543_543987


namespace max_pieces_in_a_box_l543_543190

-- Define the number of pieces and boxes
def pieces : Nat := 48
def boxes : Nat := 9

-- Define the constraints
def box_constraint (f : Fin boxes → Nat) : Prop :=
  (∀ i j : Fin boxes, i ≠ j → f i ≠ f j) ∧
  (∀ i : Fin boxes, f i ≥ 1) ∧
  (∑ i in Fin (boxes), f i = pieces)

-- Define our function f
def f (i : Fin boxes) : Nat := 
  if i.val < 8 then i.val + 1 else pieces - (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8)

-- The theorem we want to prove
theorem max_pieces_in_a_box : ∃ f : Fin boxes → Nat, box_constraint f ∧ (∀ (i j : Fin boxes), f j = 12 → 
  (f i ≤ f j ∨ i = j)) :=
by
  use f
  apply and.intro sorry sorry

end max_pieces_in_a_box_l543_543190


namespace money_left_after_shopping_l543_543984

-- Define the initial amount of money Sandy took for shopping
def initial_amount : ℝ := 310

-- Define the percentage of money spent in decimal form
def percentage_spent : ℝ := 0.30

-- Define the remaining money as per the given conditions
def remaining_money : ℝ := initial_amount * (1 - percentage_spent)

-- The statement we need to prove
theorem money_left_after_shopping :
  remaining_money = 217 :=
by
  sorry

end money_left_after_shopping_l543_543984


namespace rectangle_ratio_theorem_l543_543493

noncomputable def find_W : ℝ :=
  let W := 1 + Real.sqrt 5 in (W / 2)

theorem rectangle_ratio_theorem (W : ℝ) 
  (unit_square : true) 
  (ratio_condition : W / 1 = 1 / (W - 1)) :
  W = (1 + Real.sqrt 5) / 2 := 
sorry

end rectangle_ratio_theorem_l543_543493


namespace flat_fee_shipping_l543_543013

theorem flat_fee_shipping (w : ℝ) (c : ℝ) (C : ℝ) (F : ℝ) 
  (h_w : w = 5) 
  (h_c : c = 0.80) 
  (h_C : C = 9)
  (h_shipping : C = F + (c * w)) :
  F = 5 :=
by
  -- proof skipped
  sorry

end flat_fee_shipping_l543_543013


namespace evaluate_expression_l543_543686

theorem evaluate_expression :
  let a := 24
  let b := 7
  3 * (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2258 :=
by
  let a := 24
  let b := 7
  sorry

end evaluate_expression_l543_543686


namespace tangent_line_to_ellipse_m_squared_l543_543484

theorem tangent_line_to_ellipse_m_squared :
  ∀ (m : ℝ), (∀ (x y : ℝ), y = m * x + 2 ∧ x^2 + 9 * y^2 = 9 → 0) ↔ (m^2 = 1/3) :=
by
  sorry

end tangent_line_to_ellipse_m_squared_l543_543484


namespace area_PQRS_l543_543191

open Real

def is_square (A B C D : Point) : Prop :=
  ∃ s : ℝ, s > 0 ∧ (dist A B = s) ∧ (dist B C = s) ∧ (dist C D = s) ∧ (dist D A = s)

def is_equilateral_triangle (A B C : Point) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A) ∧ 
  (dist A B = dist C A)

structure Point :=
  (x : ℝ) (y : ℝ)

noncomputable def area_square (s : ℝ) : ℝ :=
  s * s

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3) * (s * s) / 4

theorem area_PQRS {A B C D P Q R S : Point} 
  (h1 : is_square A B C D)
  (h2 : is_equilateral_triangle A P B)
  (h3 : is_equilateral_triangle B Q C)
  (h4 : is_equilateral_triangle C R D)
  (h5 : is_equilateral_triangle D S A)
  (h6 : ∃ s, s > 0 ∧ area_square s = 36)
  : area_PQRS = 36 + 36 * sqrt 6 := 
sorry

end area_PQRS_l543_543191


namespace line_intersects_circle_at_two_points_midpoint_trajectory_four_points_distance_l543_543425

def circle (C : Type) : Prop :=
  ∃ x y : ℝ, (x + 2)^2 + y^2 = 5

def line (l : Type) (m : ℝ) : Prop :=
  ∃ x y : ℝ, mx - y + 1 + 2m = 0

theorem line_intersects_circle_at_two_points (C : Type) (l : Type) (m : ℝ) (hC : circle C) (hl : line l m) :
  True :=
sorry

theorem midpoint_trajectory (C : Type) (l : Type) (m : ℝ) (hC : circle C) (hl : line l m) :
  ∃ x y : ℝ, (x + 2)^2 + (y - 1 / 2)^2 = 1 / 4 :=
sorry

theorem four_points_distance (C : Type) (l : Type) (hC : circle C) :
  ∀ (m : ℝ), (m > 2 ∨ m < -2) :=
sorry

end line_intersects_circle_at_two_points_midpoint_trajectory_four_points_distance_l543_543425


namespace cos_double_angle_sum_l543_543376

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l543_543376


namespace river_depth_l543_543301

theorem river_depth (W R V: ℝ) (hW: W = 55) (hR: R = 16.67) (hV: V = 2750) : 
  ∃ D: ℝ, D ≈ 3 :=
by
  sorry

end river_depth_l543_543301


namespace cauchy_problem_solution_l543_543583

noncomputable def x (t : ℝ) : ℝ := -4 * Real.exp (-2 * t) - 2 * Real.sin (4 * t)
noncomputable def y (t : ℝ) : ℝ := Real.exp (-2 * t) - Real.cos (4 * t)
noncomputable def z (t : ℝ) : ℝ := Real.exp (-2 * t) - 2 * Real.sin (4 * t)

theorem cauchy_problem_solution :
  ∀ t : ℝ,
    (∂ x t = 8 * y t) ∧
    (∂ y t = -2 * z t) ∧
    (∂ z t = 2 * x t + 8 * y t - 2 * z t) ∧
    (x 0 = -4) ∧ (y 0 = 0) ∧ (z 0 = 1) :=
by
  sorry

end cauchy_problem_solution_l543_543583


namespace find_f_2_l543_543879

theorem find_f_2 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 2 = 5 :=
sorry

end find_f_2_l543_543879


namespace periodic_function_of_2011_l543_543169

noncomputable def f : ℕ → (ℝ → ℝ)
| 0     := λ x, Real.cos x
| (n+1) := λ x, f n x.deriv

theorem periodic_function_of_2011 :
  ∀ x, f 2011 x = Real.sin x := by
sorry

end periodic_function_of_2011_l543_543169


namespace probability_multiple_of_3_or_4_l543_543621

-- Given the numbers 1 through 30 are written on 30 cards one number per card,
-- and Sara picks one of the 30 cards at random,
-- the probability that the number on her card is a multiple of 3 or 4 is 1/2.

-- Define the set of numbers from 1 to 30
def numbers := finset.range 30 \ {0}

-- Define what it means to be a multiple of 3 or 4 within the given range
def is_multiple_of_3_or_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 4 = 0

-- Define the set of multiples of 3 or 4 within the given range
def multiples_of_3_or_4 := numbers.filter is_multiple_of_3_or_4

-- The probability calculation
theorem probability_multiple_of_3_or_4 : 
  (multiples_of_3_or_4.card : ℚ) / numbers.card = 1 / 2 :=
begin
  -- The set multiples_of_3_or_4 contains 15 elements
  have h_multiples_card : multiples_of_3_or_4.card = 15, sorry,
  -- The set numbers contains 30 elements
  have h_numbers_card : numbers.card = 30, sorry,
  -- Therefore, the probability is 15/30 = 1/2
  rw [h_multiples_card, h_numbers_card],
  norm_num,
end

end probability_multiple_of_3_or_4_l543_543621


namespace probability_multiple_of_3_or_4_l543_543618

theorem probability_multiple_of_3_or_4 :
  let numbers := {n | 1 ≤ n ∧ n ≤ 30},
      multiples_of_3 := {n | n ∈ numbers ∧ n % 3 = 0},
      multiples_of_4 := {n | n ∈ numbers ∧ n % 4 = 0},
      multiples_of_12 := {n | n ∈ numbers ∧ n % 12 = 0},
      favorable_outcomes := multiples_of_3 ∪ multiples_of_4,
      double_counted_outcomes := multiples_of_12,
      total_favorable_outcomes := set.card favorable_outcomes - set.card double_counted_outcomes,
      total_outcomes := set.card numbers in
  total_favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_multiple_of_3_or_4_l543_543618


namespace cost_of_five_slices_l543_543520

def pizzas : ℕ := 3
def slices_per_pizza : ℕ := 12
def total_cost : ℝ := 72
def slices : ℕ := pizzas * slices_per_pizza
def cost_per_slice : ℝ := total_cost / slices
def slices_to_find_cost_for : ℕ := 5

theorem cost_of_five_slices : (cost_per_slice * slices_to_find_cost_for) = 10 := by sorry

end cost_of_five_slices_l543_543520


namespace definite_integral_example_l543_543354

theorem definite_integral_example :
  ∫ x in 0..(Real.pi / 2), (3 * x + Real.sin x) = (3 * Real.pi^2 / 8) + 1 :=
by
  sorry

end definite_integral_example_l543_543354


namespace min_acquaintance_pairs_l543_543504

namespace Sosnovka

open Function

variables {inhabitants : Type} [Fintype inhabitants] [Nonempty inhabitants] (acquainted : inhabitants → inhabitants → Prop)

/-- Any five inhabitants can be seated at a round table such that each of them is acquainted with both their neighbors -/
axiom acquaintance_condition 
  (h : inhabited (Fin 5 → inhabitants)) :
  ∃ p : Fin 5 → inhabitants, 
    ∀ i : Fin 5, acquainted (p i) (p ((i + 1) % 5)) ∧ acquainted (p i) (p ((i + 4) % 5))

/-- The minimum number of pairs of acquainted inhabitants in Sosnovka is 28440 -/
theorem min_acquaintance_pairs 
  (h_card : Fintype.card inhabitants = 240) :
  ∃ n : ℕ, n = 28440 ∧ (∀ x : inhabitants, ∃ S ⊆ univ inhabitants, card S = 238 ∧ ∀ y ∈ S, acquainted x y) :=
sorry

end Sosnovka

end min_acquaintance_pairs_l543_543504


namespace center_of_circle_is_2_1_l543_543715

-- Definition of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y - 5 = 0

-- Theorem stating the center of the circle
theorem center_of_circle_is_2_1 (x y : ℝ) (h : circle_eq x y) : (x, y) = (2, 1) := sorry

end center_of_circle_is_2_1_l543_543715


namespace monotonicity_of_f_zero_of_f_l543_543878

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp (2 * x) + (a + 2) * exp x + a * x

theorem monotonicity_of_f {a : ℝ} : 
  (∀ x : ℝ, 0 ≤ a → 0 < deriv (f x a)) ∧
  (∀ x : ℝ, a < 0 → if x < real.log (-a / 2) then deriv (f x a) < 0 else deriv (f x a) > 0) :=
sorry

theorem zero_of_f {a : ℝ} (ha : 0 < a) (x₀ : ℝ) (hx₀ : f x₀ a = 0) : 
  real.log (2 * a / (4 * a + 5)) < x₀ ∧ x₀ < -1 / real.exp 1 :=
sorry

end monotonicity_of_f_zero_of_f_l543_543878


namespace cosine_identity_l543_543391

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l543_543391


namespace value_of_k_l543_543482

theorem value_of_k (k : ℝ) (h1 : k ≠ 0) (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k * x₁ - 100) < (k * x₂ - 100)) : k = 1 :=
by
  have h3 : k > 0 :=
    sorry -- We know that if y increases as x increases, then k > 0
  have h4 : k = 1 :=
    sorry -- For this specific problem, we can take k = 1 which satisfies the conditions
  exact h4

end value_of_k_l543_543482


namespace fn_2011_equals_sin_l543_543167

noncomputable def fn : ℕ → (ℝ → ℝ)
| 0       := λ x, Real.cos x
| (n + 1) := λ x, (fn n x).deriv

theorem fn_2011_equals_sin (x : ℝ) : fn 2011 x = Real.sin x :=
by sorry

end fn_2011_equals_sin_l543_543167


namespace num_integers_divisible_by_6_10_15_in_range_500_1000_l543_543474

-- Define a function to compute the LCM of three numbers
def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Define the problem statement
theorem num_integers_divisible_by_6_10_15_in_range_500_1000 : 
  let lcm_6_10_15 := lcm_three 6 10 15 in
  let start := 500 in
  let end := 1000 in
  let smallest_multiple := Nat.ceil (start / lcm_6_10_15) * lcm_6_10_15 in
  let largest_multiple := Nat.floor (end / lcm_6_10_15) * lcm_6_10_15 in
  let num_multiples := (largest_multiple - smallest_multiple) / lcm_6_10_15 + 1 in
  num_multiples = 17 := sorry

end num_integers_divisible_by_6_10_15_in_range_500_1000_l543_543474


namespace angle_of_inclination_l543_543215

theorem angle_of_inclination (θ : ℝ) (h1 : θ ∈ set.Ico 0 (2 * Real.pi))
  (h2 : Real.tan θ = Real.sqrt 3) :
  θ = (Real.pi / 3) :=
sorry

end angle_of_inclination_l543_543215


namespace find_angle_l543_543756

def degree := ℝ

def complement (x : degree) : degree := 180 - x

def angle_condition (x : degree) : Prop :=
  x - (complement x / 2) = -18 - 24/60 - 36/3600

theorem find_angle : ∃ x : degree, angle_condition x ∧ x = 47 + 43/60 + 36/3600 :=
by
  sorry

end find_angle_l543_543756


namespace pipe_C_emptying_time_l543_543249

theorem pipe_C_emptying_time:
  (∀ (tank : ℕ), 
   let rate_A := 1 / 20,
       rate_B := 1 / 30,
       t_open := 2 in
   ( ∃ x : ℝ, 
     let rate_C := 1 / x in
     3 * t_open * (rate_A + rate_B - rate_C) = 1 )
   → x = 3) := sorry

end pipe_C_emptying_time_l543_543249


namespace g_range_l543_543951

theorem g_range (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < g a b c ∧ g a b c < 1.5 :=
by
  sorry

def g (a b c : ℝ) : ℝ :=
  a / (a + b + 1) + b / (b + c + 1) + c / (c + a + 1)

end g_range_l543_543951


namespace part_a_part_b_l543_543630

-- Definitions and conditions
def is_convex_hexagon (A B C D E F : Point) : Prop := 
  -- The convexity condition of the hexagon

def opposite_sides_parallel (A B C D E F : Point) : Prop := 
  -- The condition that opposite sides are pairwise parallel

-- The theorem statements
theorem part_a (A B C D E F : Point) 
  (Hconvex : is_convex_hexagon A B C D E F) 
  (Hparallel : opposite_sides_parallel A B C D E F) : 
  area_ACE A C E ≥ area_hexagon A B C D E F / 2 :=
sorry

theorem part_b (A B C D E F : Point) 
  (Hconvex : is_convex_hexagon A B C D E F) 
  (Hparallel : opposite_sides_parallel A B C D E F) : 
  area_ACE A C E = area_BDF B D F :=
sorry

end part_a_part_b_l543_543630


namespace area_of_triangle_BCF_l543_543926

open Real Geometry

/-- Given a square ABCD with side length of 1 cm, E as the midpoint of the diagonal BD, and F as the midpoint of the segment BE, the area of the triangle BCF is 0.125 cm² --/
theorem area_of_triangle_BCF :
  ∃ A B C D E F : Point, 
  is_square A B C D 1 ∧
  midpoint B D E ∧
  midpoint B E F →
  area (triangle B C F) = 0.125 := 
sorry

end area_of_triangle_BCF_l543_543926


namespace find_y_l543_543216

-- Define the sequence from 1 to 50
def seq_sum : ℕ := (50 * 51) / 2

-- Define y and the average condition
def average_condition (y : ℚ) : Prop :=
  (seq_sum + y) / 51 = 51 * y

-- Theorem statement
theorem find_y (y : ℚ) (h : average_condition y) : y = 51 / 104 :=
by
  sorry

end find_y_l543_543216


namespace james_divisions_to_odd_less_than_5_l543_543210

theorem james_divisions_to_odd_less_than_5 : 
  let f := Nat.floor
  ∃ n : ℕ, n = 4 ∧ (let x := Nat.iterate (λ x => f (x / 3)) n 144 in x < 5 ∧ x % 2 = 1) :=
sorry

end james_divisions_to_odd_less_than_5_l543_543210


namespace probability_two_red_two_green_l543_543276

theorem probability_two_red_two_green (total_red total_blue total_green : ℕ)
  (total_marbles total_selected : ℕ) (probability : ℚ)
  (h_total_marbles: total_marbles = total_red + total_blue + total_green)
  (h_total_selected: total_selected = 4)
  (h_red_selected: 2 ≤ total_red)
  (h_green_selected: 2 ≤ total_green)
  (h_total_selected_le: total_selected ≤ total_marbles)
  (h_probability: probability = (Nat.choose total_red 2 * Nat.choose total_green 2) / (Nat.choose total_marbles total_selected))
  (h_total_red: total_red = 12)
  (h_total_blue: total_blue = 8)
  (h_total_green: total_green = 5):
  probability = 2 / 39 :=
by
  sorry

end probability_two_red_two_green_l543_543276


namespace fraction_product_simplified_l543_543768

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end fraction_product_simplified_l543_543768


namespace total_participants_and_books_l543_543350

variables (R F : ℕ)

theorem total_participants_and_books (R F : ℕ) 
  (E := 150) 
  (S := E + (R / 5) + (F / 7)) 
  (total_books := 2 * S) 
  (participants := 150 + (F / 3) + (R / 2)) :
  R = 420 → F = 168 →
  participants = 416 ∧ total_books = 738 := 
by
  intros hR hF
  have hE : E = 150 := rfl
  have hS : S = 150 + (R / 5) + (F / 7) := rfl
  have hTotalBooks : total_books = 2 * (150 + (R / 5) + (F / 7)) := rfl
  have hParticipants : participants = 150 + (F / 3) + (R / 2) := rfl
  rw [hR, hF, hE, hS, hTotalBooks, hParticipants]
  split
  · simp
    norm_num
  · norm_num
    simp
    norm_num
    sorry

end total_participants_and_books_l543_543350


namespace students_only_solving_B_l543_543919

variables
  (x_A x_B x_C x_AB x_BC x_CA x_ABC : ℕ)
  (total_students : ℕ)

axioms
  (H1 : x_A + x_B + x_C + x_AB + x_BC + x_CA + x_ABC = 25)
  (H2 : x_B + x_BC = 2 * (x_C + x_BC))
  (H3 : x_A = x_AB + x_CA + x_ABC + 1)
  (H4 : x_A = x_B + x_C)

theorem students_only_solving_B : x_B = 6 :=
by
  sorry

end students_only_solving_B_l543_543919


namespace find_150th_letter_l543_543672

def repeating_sequence : String := "ABCD"

def position := 150

theorem find_150th_letter :
  repeating_sequence[(position % 4) - 1] = 'B' := 
sorry

end find_150th_letter_l543_543672


namespace quadratic_two_distinct_real_roots_l543_543641

theorem quadratic_two_distinct_real_roots :
  ∃ a b c : ℝ, a = 1 ∧ b = -7 ∧ c = 5 ∧ (b^2 - 4 * a * c) > 0 :=
by
  use 1, -7, 5
  split
  . refl
  split
  . refl
  split
  . refl
  calc
    (-7)^2 - 4 * 1 * 5 = 49 - 20 := by norm_num
    ... = 29                   := by norm_num
    ... > 0                    := by norm_num
  sorry

end quadratic_two_distinct_real_roots_l543_543641


namespace sin_alpha_at_point_l543_543487

open Real

theorem sin_alpha_at_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (1, -2)) :
  sin α = -2 * sqrt 5 / 5 :=
sorry

end sin_alpha_at_point_l543_543487


namespace ellipse_properties_l543_543433

-- Definitions for the conditions
def ellipse_contains_point (a b : ℝ) (E : ℝ × ℝ → Prop) :=
  E (-√3, 1/2)

def eccentricity_relation (a b : ℝ) :=
  (√3 / 2) * a = √(a^2 - b^2)

def midpoint_condition (H : ℝ × ℝ) : Prop :=
  ∃ n m, let t := 4 + m^2 in 
  H = (4 * n / t, -m * n / t) ∧ n^2 = (4 + m^2)^2 / (16 + m^2)

def points_max_area (a b : ℝ) (E : ℝ × ℝ → Prop) :=
  ∀ P Q H O : ℝ × ℝ,
  E P ∧ E Q → 
  (H = (P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  O = (0, 0) →
  dist O H = 1 →
  ∃ n m, let T := (12 * 16 * (4 + m^2)) / (16 + m^2)^2 in
  T / (t + 144 / t + 24) ≤ 1 / 48 → area (triangle O P Q) = 1

-- Statement to prove
theorem ellipse_properties (a b : ℝ) (E : ℝ × ℝ → Prop) :
  (∃ a b, a > b > 0 ∧ a^2 = 4 ∧ b^2 = 1 ∧ 
  ellipse_contains_point a b E ∧
  eccentricity_relation a b ∧
  points_max_area a b E)
  → 
  ∀ P Q H O, 
  midpoint_condition H →
  ∃ n m, 
  (points_max_area a b E) ↔ area (triangle O P Q) = 1 :=
begin
  sorry
end

end ellipse_properties_l543_543433


namespace letter_150_in_pattern_l543_543673

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l543_543673


namespace initial_tickets_count_l543_543008

def spent_tickets : ℕ := 5
def additional_tickets : ℕ := 10
def current_tickets : ℕ := 16

theorem initial_tickets_count (initial_tickets : ℕ) :
  initial_tickets - spent_tickets + additional_tickets = current_tickets ↔ initial_tickets = 11 :=
by
  sorry

end initial_tickets_count_l543_543008


namespace inequality_proofs_l543_543844

theorem inequality_proofs
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hab : a + b = 1) :
  (ab : ℝ) ≤ 1 / 4 ∧
  (1 / a + 1 / b > 4) ∧
  (sqrt a + sqrt b ≤ sqrt 2) ∧
  (a^2 + b^2 ≥ 1 / 2) :=
by
  sorry

end inequality_proofs_l543_543844


namespace radii_computation_correct_l543_543316

noncomputable def compute_radii (m F V : ℝ) : ℝ × ℝ :=
  let A := 3 * V / (Real.pi * m)
  let B := F / Real.pi
  let discriminant_part1 := m^2 + 2 * B
  let discriminant_part2 := m^2 + 16 * A - 6 * B
  let discriminant_sqrt := Real.sqrt(discriminant_part1 * discriminant_part2)
  let v := (m^2 - 2 * B + discriminant_sqrt) / 8 
  let u := A - v
  let R_sum_r_sq := u + 2 * v
  let R_sub_r_sq := u - 2 * v
  let R_plus_r := Real.sqrt(R_sum_r_sq)
  let R_minus_r := Real.sqrt(R_sub_r_sq)
  let R := (R_plus_r + R_minus_r) / 2
  let r := (R_plus_r - R_minus_r) / 2
  (R, r)

theorem radii_computation_correct (m F V : ℝ) :
  let (R, r) := compute_radii m F V
  R = 
    (1 / 2) * (Real.sqrt((3 * V / (Real.pi * m) + (m^2 - 2 * (F / Real.pi) +
    Real.sqrt((m^2 + 2 * (F / Real.pi)) * (m^2 + 16 * (3 * V / (Real.pi * m)) -
    6 * (F / Real.pi)))) / 8)) +
    Real.sqrt((3 * V / (Real.pi * m) - 3 * (m^2 - 2 * (F / Real.pi) +
    Real.sqrt((m^2 + 2 * (F / Real.pi)) * (m^2 + 16 * (3 * V / (Real.pi * m)) -
    6 * (F / Real.pi)))) / 8))) ∧
  r = 
    (1 / 2) * (Real.sqrt((3 * V / (Real.pi * m) + (m^2 - 2 * (F / Real.pi) +
    Real.sqrt((m^2 + 2 * (F / Real.pi)) * (m^2 + 16 * (3 * V / (Real.pi * m)) -
    6 * (F / Real.pi)))) / 8)) -
    Real.sqrt((3 * V / (Real.pi * m) - 3 * (m^2 - 2 * (F / Real.pi) +
    Real.sqrt((m^2 + 2 * (F / Real.pi)) * (m^2 + 16 * (3 * V / (Real.pi * m)) -
    6 * (F / Real.pi)))) / 8))) := 
by sorry

end radii_computation_correct_l543_543316


namespace algebraic_expressions_difference_l543_543808

noncomputable def algebraic_sum_condition (M N : ℤ → ℤ) (x : ℤ) : Prop :=
  let P := -x^2 + 2*x + N x in
  let middle_first_col := 2*x^2 - 3*x - 1 in
  M x - N x = -2*x^2 + 4*x

theorem algebraic_expressions_difference (M N : ℤ → ℤ) (x : ℤ) :
  algebraic_sum_condition M N x → M x - N x = -2*x^2 + 4*x := by
  sorry

end algebraic_expressions_difference_l543_543808


namespace strawberry_ratio_l543_543015

theorem strawberry_ratio (Christine_picked : ℕ) (pies_needed : ℕ) (straw_rate_per_pie : ℕ) (Rachel_picked : ℕ)
  (h1 : Christine_picked = 10)
  (h2 : pies_needed = 10)
  (h3 : straw_rate_per_pie = 3)
  (h4 : Christine_picked + Rachel_picked = pies_needed * straw_rate_per_pie) :
  Rachel_picked / Christine_picked = 2 :=
by
  rw [h1, h2, h3] at h4
  exact Nat.eq_of_add_eq_add_left h4
  sorry

end strawberry_ratio_l543_543015


namespace line_through_P_with_min_intercepts_l543_543294

theorem line_through_P_with_min_intercepts : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧ ((1 / a + 4 / b = 1) ∧ a + b = min_intercept_sum) ∧ 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1 / a' + 4 / b' = 1 → a' + b' ≥ a + b) → 
  ∃ (c d e : ℝ), c = 1 ∧ d = 2 ∧ e = -6 ∧ ∀ x y : ℝ, c * x + d * y + e = 0 :=
begin
  sorry
end

end line_through_P_with_min_intercepts_l543_543294


namespace noConsecutiveNumbersOnCube_l543_543969

noncomputable def cubeProbability : ℚ := 1 / 672

theorem noConsecutiveNumbersOnCube {l : List ℕ} (h1 : l = [1, 2, 3, 4, 5, 6, 7, 8, 9])
    (h2 : l.length = 9)
    (h3 : ∃ x ∈ l, l.erase x).length = 8) : 
    ∃ m n : ℕ, Nat.Coprime m n ∧ cubeProbability = m / n ∧ m + n = 673 :=
by
  sorry

end noConsecutiveNumbersOnCube_l543_543969


namespace max_valid_numbers_l543_543564

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def first_two_digits_differ_by_2 (a b : ℕ) : Prop :=
  abs (a - b) = 2

def last_digit_is_6_or_7 (n : ℕ) : Prop :=
  n % 10 = 6 ∨ n % 10 = 7

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  is_divisible_by_3 n ∧
  first_two_digits_differ_by_2 (n / 100) ((n / 10) % 10) ∧
  last_digit_is_6_or_7 n

theorem max_valid_numbers :
  let valid_numbers := {n : ℕ | is_valid_number n} in
  finset.card valid_numbers = 9 := by
  sorry

end max_valid_numbers_l543_543564


namespace arithmetic_sequence_ratio_a10_b10_l543_543886

variable {a : ℕ → ℕ} {b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- We assume S_n and T_n are the sums of the first n terms of sequences a and b respectively.
-- We also assume the provided ratio condition between S_n and T_n.
axiom sum_of_first_n_terms_a (n : ℕ) : S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2
axiom sum_of_first_n_terms_b (n : ℕ) : T n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1))) / 2
axiom ratio_condition (n : ℕ) : (S n) / (T n) = (3 * n - 1) / (2 * n + 3)

theorem arithmetic_sequence_ratio_a10_b10 : (a 10) / (b 10) = 56 / 41 :=
by sorry

end arithmetic_sequence_ratio_a10_b10_l543_543886


namespace bingo_prize_money_is_2400_l543_543275

def totalPrize (P : ℝ) : Prop :=
  let first_winner := (1 / 3) * P
  let remaining := (2 / 3) * P
  let next_ten_winners := (1 / 10) * remaining
  (next_ten_winners = 160) → P = 2400

theorem bingo_prize_money_is_2400 : ∃ P : ℝ, totalPrize P :=
begin
  use 2400,
  unfold totalPrize,
  intro h,
  sorry
end

end bingo_prize_money_is_2400_l543_543275


namespace proof_aim_l543_543881

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + (2 - a) = 0

theorem proof_aim (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
sorry

end proof_aim_l543_543881


namespace combined_area_of_runners_l543_543647

-- Define the given conditions
variables (table_area runners_area total_covered_area two_layers three_layers : ℕ)
variables (runners_cover_pct : ℚ)

-- Given conditions
axiom table_area_def : table_area = 175
axiom runners_cover_pct_def : runners_cover_pct = 0.8
axiom total_covered_area_def : total_covered_area = (runners_cover_pct * table_area : ℕ)
axiom two_layers_def : two_layers = 24
axiom three_layers_def : three_layers = 22

-- Given result based on the conditions
theorem combined_area_of_runners : (total_covered_area - two_layers - three_layers) + 2 * two_layers + 3 * three_layers = 208 := by
  have total_covered : total_covered_area = 140 := by
    rw [table_area_def, runners_cover_pct_def]
    norm_num
  have one_layer : total_covered_area - two_layers - three_layers = 94 := by
    rw [total_covered, two_layers_def, three_layers_def]
    norm_num
  rw [one_layer, two_layers_def, three_layers_def]
  norm_num
  sorry

end combined_area_of_runners_l543_543647


namespace exists_subset_A_l543_543202

noncomputable def c := 0.0001
noncomputable def α := Real.log 5 / Real.log 16

theorem exists_subset_A (n : ℕ) (h_pos : 0 < n) :
  ∃ (A : Set ℕ), A ⊆ {x | 1 ≤ x ∧ x ≤ n} ∧
  A.card ≥ (c * n ^ α).to_nat ∧
  ∀ (x y ∈ A), x ≠ y → ¬∃ k : ℕ, (x - y) = k ^ 2 :=
sorry

end exists_subset_A_l543_543202


namespace probability_multiple_of_3_or_4_l543_543628

theorem probability_multiple_of_3_or_4 :
  let numbers := Finset.range 30
  let multiples_of_3 := {n ∈ numbers | n % 3 = 0}
  let multiples_of_4 := {n ∈ numbers | n % 4 = 0}
  let multiples_of_12 := {n ∈ numbers | n % 12 = 0}
  let favorable_count := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card
  let probability := (favorable_count : ℚ) / numbers.card
  probability = (1 / 2 : ℚ) :=
by
  sorry

end probability_multiple_of_3_or_4_l543_543628


namespace intervals_of_monotonicity_min_value_condition_range_of_a_l543_543104

noncomputable def f (a x : ℝ) : ℝ := exp x + 2 * a * x

theorem intervals_of_monotonicity (a : ℝ):
  (∀ x, 0 ≤ a → (∀ y z, (y < z) → (f a y < f a z))) ∧ 
  (∀ x, a < 0 → (∀ y z, (y < z ∧ y < log (-2 * a)) → (f a y > f a z)) ∧ (∀ y z, (y < z ∧ log (-2 * a) < y) → (f a y < f a z))) :=
sorry

theorem min_value_condition (a : ℝ) :
  ((∃ x, (1 ≤ x) ∧ (f a x = 0)) → a = - (exp 1) / 2) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, (0 ≤ x) → (f a x ≥ exp (-x))) ↔ (a ∈ set.Ici (-1)) :=
sorry

end intervals_of_monotonicity_min_value_condition_range_of_a_l543_543104


namespace area_of_bounded_region_l543_543046

theorem area_of_bounded_region (a : ℝ) (ha : 0 < a) :
  let f1 (x y : ℝ) := (x + 2 * a * y)^2 = 16 * a^2
      f2 (x y : ℝ) := (2 * a * x - y)^2 = 4 * a^2
   in 
    ∃ A, 
      (A = 32 * a^2 / (1 + 4 * a^2)) ∧
      (∀ x y : ℝ, f1 x y → f2 x y → A) :=
sorry

end area_of_bounded_region_l543_543046


namespace car_q_time_l543_543779

-- Definitions
variables (v t : ℝ) (a : ℝ)
def final_velocity : ℝ := 4 * v
def initial_velocity : ℝ := 2 * v
def distance : ℝ := v * t

-- Condition on acceleration
def acceleration_condition : Prop := 
  final_velocity ^ 2 = initial_velocity ^ 2 + 2 * a * distance

-- Proof statement
theorem car_q_time (h_accel : acceleration_condition) : 
  ∃ t_Q : ℝ, t_Q = t / 3 :=
by
  -- Skip the proof
  sorry

end car_q_time_l543_543779


namespace find_150th_letter_l543_543670

def repeating_sequence : String := "ABCD"

def position := 150

theorem find_150th_letter :
  repeating_sequence[(position % 4) - 1] = 'B' := 
sorry

end find_150th_letter_l543_543670


namespace rhombus_area_correct_l543_543064

noncomputable def rhombus_area (s : ℝ) (theta : ℝ) : ℝ :=
  let d1 := 2 * s * Real.cos (theta / 2)
  let d2 := 2 * s * Real.sin (theta / 2)
  (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 8 (55 * Real.pi / 180) ≈ 53.288 :=
by
  sorry

end rhombus_area_correct_l543_543064


namespace x_plus_p_eq_2p_plus_2_l543_543479

-- Define the conditions and the statement to be proved
theorem x_plus_p_eq_2p_plus_2 (x p : ℝ) (h1 : x > 2) (h2 : |x - 2| = p) : x + p = 2 * p + 2 :=
by
  -- Proof goes here
  sorry

end x_plus_p_eq_2p_plus_2_l543_543479


namespace spinner_product_probability_l543_543335

theorem spinner_product_probability :
  let P := finset.card ((finset.filter (λ (x : ℕ × ℕ), x.1 * x.2 < 42) (finset.product (finset.range 8) (finset.range 13))) :
    ℝ) / (finset.card ((finset.product (finset.range 8) (finset.range 13))) : ℝ) in
  P = 31 / 42 :=
by
  sorry

end spinner_product_probability_l543_543335


namespace number_of_true_propositions_is_one_l543_543100

theorem number_of_true_propositions_is_one : 
  let p1 := ¬(∀ a b m : ℝ, a < b → am^2 < bm^2)      -- Propositional inverse of 'am^2 < bm^2' -> 'a < b'
  let p2 := ¬(∀ p q : Prop, p ∨ q → p ∧ q)         -- If 'p or q' is true, then both 'p' and 'q' are true
  let p3 := ¬(∀ x : ℝ, x > 1 → x > 2)             -- 'x > 1' is a sufficient condition for 'x > 2'
  let p4 := (∀ x : ℝ, x^2 - x ≤ 0)                -- Negation of '∃ x ℝ. x^2 - x > 0' is '∀ x ℝ. x^2 - x ≤ 0'
  in (if p4, 1, 0) = 1 :=
sorry

end number_of_true_propositions_is_one_l543_543100


namespace money_left_l543_543733

variable (S : ℚ)

-- Given conditions
def house_rent := (2/5) * S
def food := (3/10) * S
def conveyance := (1/8) * S
def total_food_conveyance := 3400

-- Given that total expenditure on food and conveyance is $3400
axiom h_food_conveyance : food + conveyance = total_food_conveyance

-- Prove the total money left after all expenditures
theorem money_left : S = 8000 → S - (house_rent + food + conveyance) = 1400 :=
by
  intros hS
  have h_total_expenditure : house_rent + food + conveyance = 3200 + 3400 :=
  sorry  -- This will be proven using further steps, skipping here for brevity
  show S - (house_rent + food + conveyance) = 1400 from
  sorry  -- Similar skipping of proof steps to comply with instructions

end money_left_l543_543733


namespace cartesian_C2_eq_min_distance_l543_543112

def parametric_curve_C1 (t : ℝ) : ℝ × ℝ :=
  (2 * t - 1, -4 * t - 2)

def polar_curve_C2 (θ : ℝ) : ℝ :=
  2 / (1 - Real.cos θ)

theorem cartesian_C2_eq (x y : ℝ) : (y^2 = 4 * (x - 1)) ↔ 
  ∃ (θ : ℝ), (x = 2 / (1 - Real.cos θ) + 2 ∧ y = 2 * Real.sin θ) := sorry

theorem min_distance {t : ℝ} {r : ℝ} : 
  (∃ (t : ℝ), parametric_curve_C1 t = (x, y)) ∧ 
  (∃ (θ : ℝ), (r = polar_curve_C2 θ) ∧ (x, y) = (r^2 - 1, 2 * r)) → 
  dist (2 * x + y + 4 = 0, (r^2 - 1, 2 * r)) = 
  (3 * Real.sqrt 5 / 10) := sorry

end cartesian_C2_eq_min_distance_l543_543112


namespace sequence_formula_sum_sequence_l543_543025

noncomputable def sequence (n : ℕ) : ℕ → ℚ
| 1 => 2
| n + 1 => 2 / (2 * n + 1)

lemma sequence_condition (n : ℕ) (h : n ≥ 1) :
  (finset.range n).sum (λ k, (2 * k + 1) * sequence k) = 2 * n := sorry

theorem sequence_formula (n : ℕ) (h : n ≥ 1) : sequence n = 2 / (2 * n - 1) := sorry

theorem sum_sequence (n : ℕ) (h : n ≥ 1) :
  (finset.range n).sum (λ k, sequence k / (2 * k + 1)) = 2 * n / (2 * n + 1) := sorry

end sequence_formula_sum_sequence_l543_543025


namespace solution_set_of_inequality_l543_543361

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1/3 ≤ x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l543_543361


namespace cos_double_angle_proof_l543_543414

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l543_543414


namespace czakler_inequality_czakler_equality_pairs_l543_543538

theorem czakler_inequality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (xy - 10)^2 ≥ 64 :=
sorry

theorem czakler_equality_pairs (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
(xy - 10)^2 = 64 ↔ (x, y) = (1,2) ∨ (x, y) = (-3, -6) :=
sorry

end czakler_inequality_czakler_equality_pairs_l543_543538


namespace sum_of_b_for_one_solution_l543_543362

theorem sum_of_b_for_one_solution (b : ℝ) (has_single_solution : ∃ x, 3 * x^2 + (b + 12) * x + 11 = 0) :
  ∃ b₁ b₂ : ℝ, (3 * x^2 + (b + 12) * x + 11) = 0 ∧ b₁ + b₂ = -24 := by
  sorry

end sum_of_b_for_one_solution_l543_543362


namespace books_at_beginning_l543_543740

-- Defining constants and initial conditions
def books_loaned_out : ℝ := 45.00000000000001
def return_rate : ℝ := 0.8
def end_month_books : ℕ := 66

-- Defining the hypothesis to prove
theorem books_at_beginning (B : ℝ) : 
  B - books_loaned_out + return_rate * books_loaned_out = ↑end_month_books → B = 75 :=
by
  intro h
  have h2 : B = 75.00000000000001 := sorry
  exact (eq_of_le_of_not_lt (by linarith) (by linarith)).symm

end books_at_beginning_l543_543740


namespace sum_of_radii_gtr_inradius_l543_543593

noncomputable def circle_tangent_to_sides (abc : Triangle) (r_AC_AB r_BC_AB r_incircle : ℝ) (s₁ s₂ s : Circle) : Prop :=
  s₁.TangentToSideAC abc ∧ s₁.TangentToSideAB abc ∧
  s₂.TangentToSideBC abc ∧ s₂.TangentToSideAB abc ∧
  s₁.TangentToEachOther s₂ ∧
  s.IncircleOfTriangle abc

theorem sum_of_radii_gtr_inradius {abc : Triangle} {r_AC_AB r_BC_AB r_incircle : ℝ} (s₁ s₂ s : Circle)
  (h : circle_tangent_to_sides abc r_AC_AB r_BC_AB r_incircle s₁ s₂ s) :
  r_AC_AB + r_BC_AB > r_incircle := sorry

end sum_of_radii_gtr_inradius_l543_543593


namespace equation_of_ellipse_l543_543068

noncomputable def
  a > b > 0 : Prop := sorry

noncomputable def
  ellipse_eq (a b : ℝ) : Prop := 
  ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def
  eccentricity (a : ℝ) : Prop := 
  (√3) / 3

noncomputable def
  perimeter_triangle (a : ℝ) : Prop :=
  4 * a = 4 * √3

theorem equation_of_ellipse :
  ∃ a b : ℝ,
    a > b > 0 ∧ 
    ellipse_eq a b ∧ 
    eccentricity a ∧ 
    perimeter_triangle a →
    (a = √3 ∧ b = √2) →
    ∀ x y : ℝ, (x^2) / 3 + (y^2) / 2 = 1 := 
by {
  sorry
}

end equation_of_ellipse_l543_543068


namespace cosine_identity_l543_543390

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l543_543390


namespace division_of_powers_l543_543021

theorem division_of_powers :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 :=
by sorry

end division_of_powers_l543_543021


namespace find_a_for_parallel_lines_l543_543131

-- Define the problem as a Lean statement
theorem find_a_for_parallel_lines : 
  (∀ a : ℝ, (∃ (x y : ℝ), 2*x - y - 1 = 0) ∧ (∃ (x y : ℝ), 2*x + (a+1)*y + 2 = 0) → 
  (2 = -2/(a+1))) → a = -2 :=
begin
  sorry
end

end find_a_for_parallel_lines_l543_543131


namespace find_a_l543_543339

def mul_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : mul_op a 3 = 7) : a = 8 :=
sorry

end find_a_l543_543339


namespace smallest_possible_value_of_N_l543_543746

noncomputable def smallest_N (N : ℕ) : Prop :=
  ∃ l m n : ℕ, l * m * n = N ∧ (l - 1) * (m - 1) * (n - 1) = 378

theorem smallest_possible_value_of_N : smallest_N 560 :=
  by {
    sorry
  }

end smallest_possible_value_of_N_l543_543746


namespace insurance_ratio_discrepancy_future_cost_prediction_l543_543694

-- Define the conditions given in the problem.
def insurance_coverage_russia (coverage_r : ℝ) (premium_r : ℝ) : ℝ :=
  coverage_r / premium_r

def insurance_coverage_germany (coverage_g : ℝ) (premium_g : ℝ) : ℝ :=
  coverage_g / premium_g

-- Conditions based on the given problem.
axiom russia_ratio_conditions :
  (insurance_coverage_russia 100000 2000 = 50) ∧ 
  (insurance_coverage_russia 1500000 23000 ≈ 65.22)

axiom germany_ratio_conditions :
  insurance_coverage_germany 3000000 80 = 37500

-- Define the problem statements based on the correct answers.
theorem insurance_ratio_discrepancy :
  ∃ r_f : ℝ, r_f = 37500 ∧ ∃ r_r : ℝ, r_r ~ 65 ∧ r_f > r_r :=
by sorry

theorem future_cost_prediction :
  ∃ f : ℝ → ℝ → Prop, 
    (∀ d s, f d s = d ∨ f d s = s) → 
    (∀ d s, f d s ∈ {inc, dec}) :=
by sorry

end insurance_ratio_discrepancy_future_cost_prediction_l543_543694


namespace f_2015_value_l543_543062

noncomputable def f : ℝ → ℝ := sorry

axiom symmetry : ∀ x : ℝ, f(-x) = -f(x)
axiom period_4 : ∀ x : ℝ, f(x + 4) = f(x)
axiom f_log : ∀ x : ℝ, (0 < x ∧ x < 2) → f(x) = Real.log (1 + 3 * x) / Real.log 2

theorem f_2015_value : f 2015 = -2 := by
  sorry

end f_2015_value_l543_543062


namespace min_exposed_surface_area_l543_543732

-- Definitions for the conditions
def volume (x y z : ℝ) := x * y * z = 128
def face_areas1 (x y z : ℝ) :=
  (x * y = 4) ∧ (y * z = 32)
def face_areas2 (x y z : ℝ) :=
  (x * y = 64) ∧ (y * z = 16)
def face_areas3 (x y z : ℝ) :=
  (x * y = 8) ∧ (y * z = 32)

-- Statement to prove
theorem min_exposed_surface_area :
  ∃ (x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ),
    volume x1 y1 z1 ∧ volume x2 y2 z2 ∧ volume x3 y3 z3 ∧
    face_areas1 x1 y1 z1 ∧ face_areas2 x2 y2 z2 ∧ face_areas3 x3 y3 z3 ∧
    (let possible_areas := [x1*y1, x1*z1, y1*z1, x2*y2, x2*z2, y2*z2, x3*y3, x3*z3, y3*z3] in
     let heights := [z1, z2, z3] in
     let max_height_idx := heights.indexOf (heights.maximum) in
     let min_exposed_area := 832 in
     ∃ height (ii : nat), minimum_total_surface possible_areas height ii = min_exposed_area) :=
sorry

end min_exposed_surface_area_l543_543732


namespace daves_earnings_l543_543787

theorem daves_earnings
  (hourly_wage : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (monday_earning : monday_hours * hourly_wage = 36)
  (tuesday_earning : tuesday_hours * hourly_wage = 12) :
  monday_hours * hourly_wage + tuesday_hours * hourly_wage = 48 :=
by
  sorry

end daves_earnings_l543_543787


namespace manny_problem_l543_543549

noncomputable def num_slices_left (num_pies : Nat) (slices_per_pie : Nat) (num_classmates : Nat) (num_teachers : Nat) (num_slices_per_person : Nat) : Nat :=
  let total_slices := num_pies * slices_per_pie
  let total_people := 1 + num_classmates + num_teachers
  let slices_taken := total_people * num_slices_per_person
  total_slices - slices_taken

theorem manny_problem : num_slices_left 3 10 24 1 1 = 4 := by
  sorry

end manny_problem_l543_543549


namespace point_B_value_l543_543186

theorem point_B_value (A : ℝ) (B : ℝ) (hA : A = -5) (hB : B = -1 ∨ B = -9) :
  ∃ B : ℝ, (B = A + 4 ∨ B = A - 4) :=
by sorry

end point_B_value_l543_543186


namespace multiply_and_simplify_fractions_l543_543773

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end multiply_and_simplify_fractions_l543_543773


namespace quadrilateral_is_rectangle_l543_543496

noncomputable def is_perpendicular_bisector (p1 p2 : Point) (p : Point) : Prop :=
  dist p p1 = dist p p2

theorem quadrilateral_is_rectangle
  (A B C D E F : Point)
  (H1 : convex_quad A B C D)
  (H2 : E ∈ segment A C)
  (H3 : is_perpendicular_bisector A B E)
  (H4 : is_perpendicular_bisector C D E)
  (H5 : F ∈ segment B D)
  (H6 : is_perpendicular_bisector A D F)
  (H7 : is_perpendicular_bisector B C F) :
  is_rectangle A B C D :=
sorry

-- Definitions used for clarity and completeness
structure Point := (x y : ℝ)
def dist (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)
def segment (p1 p2 : Point) : set Point :=
  { p | ∃ t ∈ Icc (0 : ℝ) 1, p = ⟨p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y)⟩ }
def convex_quad (A B C D : Point) : Prop :=
  true  -- Placeholder for the actual convex quadrilateral definition
def is_rectangle (A B C D : Point) : Prop :=
  true  -- Placeholder for the actual rectangle definition

end quadrilateral_is_rectangle_l543_543496


namespace sum_integers_between_5_and_16_l543_543685

open Nat

theorem sum_integers_between_5_and_16 : (∑ i in (Ico 6 16), i) = 105 := 
by
  sorry

end sum_integers_between_5_and_16_l543_543685


namespace challenging_math_problem_l543_543994

theorem challenging_math_problem :
  ((9^2 + (3^3 - 1) * 4^2) % 6) * Real.sqrt 49 + (15 - 3 * 5) = 35 :=
by
  sorry

end challenging_math_problem_l543_543994


namespace polar_coordinates_of_point_l543_543084

noncomputable def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
let ρ := Real.sqrt (x^2 + y^2) in
let θ := if y < 0 then -Real.arcsin (y / ρ) else Real.arccos (x / ρ) in
(ρ, θ)

theorem polar_coordinates_of_point : cartesian_to_polar 1 (-Real.sqrt 3) = (2, -Real.arccos (1/2)) := by
  sorry

end polar_coordinates_of_point_l543_543084


namespace sum_common_seq_first_n_l543_543005

def seq1 (n : ℕ) := 2 * n - 1
def seq2 (n : ℕ) := 3 * n - 2

def common_seq (n : ℕ) := 6 * n - 5

def sum_first_n_terms (a : ℕ) (d : ℕ) (n : ℕ) := 
  n * (2 * a + (n - 1) * d) / 2

theorem sum_common_seq_first_n (n : ℕ) : 
  sum_first_n_terms 1 6 n = 3 * n^2 - 2 * n := 
by sorry

end sum_common_seq_first_n_l543_543005


namespace find_m_l543_543180

theorem find_m (m : ℝ) (A B C D : ℝ × ℝ)
  (h1 : A = (m, 1)) (h2 : B = (-3, 4))
  (h3 : C = (0, 2)) (h4 : D = (1, 1))
  (h_parallel : (4 - 1) / (-3 - m) = (1 - 2) / (1 - 0)) :
  m = 0 :=
  by
  sorry

end find_m_l543_543180


namespace inequality_holds_iff_m_eq_n_l543_543355

theorem inequality_holds_iff_m_eq_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∀ (α β : ℝ), 
    ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ 
    ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) ↔ m = n :=
by
  sorry

end inequality_holds_iff_m_eq_n_l543_543355


namespace trajectory_of_G_l543_543435

def circle (x y : ℝ) := (x + real.sqrt 7)^2 + y^2 = 64
def N : ℝ × ℝ := (real.sqrt 7, 0)
def on_circle (x y : ℝ) := circle x y
def midpoint (A B Q : ℝ × ℝ) := 2 * Q = A + B
def perp_vector (A B Q G : ℝ × ℝ) := (G - Q).1 * (B - A).1 + (G - Q).2 * (B - A).2 = 0
def major_axis : ℝ := 4
def semi_focal_distance : ℝ := real.sqrt 7

theorem trajectory_of_G :
  ∀ (P Q G : ℝ × ℝ), on_circle P.1 P.2 →
  midpoint N P Q →
  perp_vector P N Q G →
  (G.1^2 / 16) + (G.2^2 / 9) = 1 :=
by
  assume P Q G h_circle h_midpoint h_perpendicular
  -- Proof omitted
  sorry

end trajectory_of_G_l543_543435


namespace cosine_identity_l543_543386

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l543_543386


namespace soccer_goal_difference_l543_543920

theorem soccer_goal_difference (n : ℕ) (h : n = 2020) :
  ¬ ∃ g : Fin n → ℤ,
    (∀ i j : Fin n, i < j → (g i < g j)) ∧ 
    (∀ i : Fin n, ∃ x y : ℕ, x + y = n - 1 ∧ 3 * x = (n - 1 - x) ∧ g i = x - y) :=
by
  sorry

end soccer_goal_difference_l543_543920


namespace total_people_equal_306_l543_543515

def vans : ℕ := 6
def minibusses : ℕ := 4
def coach_buses : ℕ := 2

def students_per_van : ℕ := 10
def teachers_per_van : ℕ := 2
def parent_chaperones_per_van : ℕ := 1

def students_per_minibus : ℕ := 24
def teachers_per_minibus : ℕ := 3
def parent_chaperones_per_minibus : ℕ := 2

def students_per_coach_bus : ℕ := 48
def teachers_per_coach_bus : ℕ := 4
def parent_chaperones_per_coach_bus : ℕ := 4

theorem total_people_equal_306 : 
  let students_in_vans := vans * students_per_van,
      students_in_minibusses := minibusses * students_per_minibus,
      students_in_coach_buses := coach_buses * students_per_coach_bus,
      total_students := students_in_vans + students_in_minibusses + students_in_coach_buses,
      
      teachers_in_vans := vans * teachers_per_van,
      teachers_in_minibusses := minibusses * teachers_per_minibus,
      teachers_in_coach_buses := coach_buses * teachers_per_coach_bus,
      total_teachers := teachers_in_vans + teachers_in_minibusses + teachers_in_coach_buses,
      
      parent_chaperones_in_vans := vans * parent_chaperones_per_van,
      parent_chaperones_in_minibusses := minibusses * parent_chaperones_per_minibus,
      parent_chaperones_in_coach_buses := coach_buses * parent_chaperones_per_coach_bus,
      total_parent_chaperones := parent_chaperones_in_vans + parent_chaperones_in_minibusses + parent_chaperones_in_coach_buses,
      
      total_people := total_students + total_teachers + total_parent_chaperones
  in total_people = 306 :=
by
  let students_in_vans := vans * students_per_van
  let students_in_minibusses := minibusses * students_per_minibus
  let students_in_coach_buses := coach_buses * students_per_coach_bus
  let total_students := students_in_vans + students_in_minibusses + students_in_coach_buses

  let teachers_in_vans := vans * teachers_per_van
  let teachers_in_minibusses := minibusses * teachers_per_minibus
  let teachers_in_coach_buses := coach_buses * teachers_per_coach_bus
  let total_teachers := teachers_in_vans + teachers_in_minibusses + teachers_in_coach_buses

  let parent_chaperones_in_vans := vans * parent_chaperones_per_van
  let parent_chaperones_in_minibusses := minibusses * parent_chaperones_per_minibus
  let parent_chaperones_in_coach_buses := coach_buses * parent_chaperones_per_coach_bus
  let total_parent_chaperones := parent_chaperones_in_vans + parent_chaperones_in_minibusses + parent_chaperones_in_coach_buses

  let total_people := total_students + total_teachers + total_parent_chaperones

  show total_people = 306
  from sorry

end total_people_equal_306_l543_543515


namespace rohan_savings_l543_543573

/-- The monthly savings of Rohan should be Rs. 1500 given his spending pattern and salary. --/
theorem rohan_savings :
  ∀ (salary : ℝ) (food_percentage house_rent_percentage entertainment_percentage conveyance_percentage : ℝ),
    salary = 7500 →
    food_percentage = 0.4 →
    house_rent_percentage = 0.2 →
    entertainment_percentage = 0.1 →
    conveyance_percentage = 0.1 →
    let total_spent := (food_percentage + house_rent_percentage + entertainment_percentage + conveyance_percentage) * salary in
    salary - total_spent = 1500 :=
by
  intros salary food_percentage house_rent_percentage entertainment_percentage conveyance_percentage
  assume h_salary h_food h_house h_entertainment h_conveyance
  let total_spent := (food_percentage + house_rent_percentage + entertainment_percentage + conveyance_percentage) * salary
  show salary - total_spent = 1500
  sorry

end rohan_savings_l543_543573


namespace bumper_cars_line_l543_543009

theorem bumper_cars_line (initial in_line_leaving newcomers : ℕ) 
  (h_initial : initial = 9)
  (h_leaving : in_line_leaving = 6)
  (h_newcomers : newcomers = 3) :
  initial - in_line_leaving + newcomers = 6 :=
by
  sorry

end bumper_cars_line_l543_543009


namespace leopard_arrangement_l543_543961

theorem leopard_arrangement : ∃ (f : Fin 9 → ℕ), 
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  let shortest_three := {f 0, f 1, f 2} in
  let middle_six := {f 3, f 4, f 5, f 6, f 7, f 8} in
  shortest_three = {0, 1, 2} ∧
  ∃ (perm : Finset.perm (Fin 9)),
    Finset.card shortest_three * Finset.card middle_six = 3! * 6! ∧ 
    3! * 6! = 4320 :=
by sorry

end leopard_arrangement_l543_543961


namespace minimum_value_expression_l543_543944

variable (a b c k : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a = k ∧ b = k ∧ c = k)

theorem minimum_value_expression : 
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) = 9 / 2 :=
by
  sorry

end minimum_value_expression_l543_543944


namespace ratio_germany_higher_future_cost_policy_uncertain_l543_543695

-- Define the ratios in Russia and Germany
def insurance_ratio_russia : ℝ := 1500000 / 23000
def insurance_ratio_germany : ℝ := 3000000 / 80

-- Define the statement that the ratio in Germany is significantly higher than in Russia
theorem ratio_germany_higher (threshold : ℝ) : insurance_ratio_germany > insurance_ratio_russia + threshold := sorry

-- Define future cost of insurance policy influenced by demand and supply increase
theorem future_cost_policy_uncertain (demand_increase supply_increase : ℝ) : 
  ∃ (future_cost : ℝ), (future_cost = demand_increase + supply_increase) ∨ (future_cost = demand_increase - supply_increase) := sorry

end ratio_germany_higher_future_cost_policy_uncertain_l543_543695


namespace complex_number_solution_l543_543371

theorem complex_number_solution (z : ℂ) (h : conj z * (1 - complex.i) = (1 + complex.i)) : z = -complex.i := 
sorry

end complex_number_solution_l543_543371


namespace find_value_of_a_l543_543955

def A := {-1, 1, 3}
def B (a : ℝ) := {a + 2, a^2 + 4}

theorem find_value_of_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
by
  sorry

end find_value_of_a_l543_543955


namespace chocolate_chip_cookies_l543_543183

theorem chocolate_chip_cookies (cookies_per_bag : ℕ) (bags : ℕ) (oatmeal_cookies : ℕ) (total_cookies : ℕ) :
  cookies_per_bag = 8 → bags = 3 → oatmeal_cookies = 19 → total_cookies = cookies_per_bag * bags →
  total_cookies - oatmeal_cookies = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  simp only [Mul.mul] at h4
  rw [h4, h3]
  norm_num
  sorry

end chocolate_chip_cookies_l543_543183


namespace cos_of_double_angles_l543_543412

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l543_543412


namespace cos_of_double_angles_l543_543410

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l543_543410


namespace curvilinear_triangle_area_l543_543287

theorem curvilinear_triangle_area (r : ℝ) : 
  let
    f_θ_x := λ θ : ℝ, r * (3 - 2 * Real.cos θ - Real.cos (2 * θ))
    f_θ_y := λ θ : ℝ, r * (2 * Real.sin θ - Real.sin (2 * θ))
    area_integral := 2 * r^2 * ∫ (θ : ℝ) in 0..(2 * Real.pi), (2 * Real.sin θ - Real.sin (2 * θ)) * (1 + 2 * Real.cos θ)
  in
    area_integral = 2 * Real.pi * r^2 :=
by sorry

end curvilinear_triangle_area_l543_543287


namespace problem_I_problem_II_problem_III_l543_543523

-- Problem I
def table1 : List (List ℝ) := [[1, 2, 3, -7], [-2, 1, 0, 1]]

-- Problem I Statement
theorem problem_I : 
  ∃ (A' : List (List ℝ)),
  is_transformed_by_two_operations table1 A' ∧ 
  (row_sums_nonnegative A' ∧ column_sums_nonnegative A') := by
  sorry

-- Problem II
def table2 (a : ℤ) : List (List ℤ) := [[a, a^2 - 1, -a, -a^2], [2-a, 1-a^2, a-2, a^2]]

-- Problem II Statement
theorem problem_II : 
  ∀ (a : ℤ), 
  (row_sums_nonnegative (table2 a) ∧ column_sums_nonnegative (table2 a)) →
  a = 0 ∨ a = -1 := by
  sorry

-- Problem III
def general_table (m n : ℕ) := matrix m n ℝ

-- Problem III Statement
theorem problem_III :
  ∀ (A : general_table m n), ∃ (operations_count : ℕ), 
  (row_sums_nonnegative A ∧ column_sums_nonnegative A) := by
  sorry

end problem_I_problem_II_problem_III_l543_543523


namespace length_of_one_side_of_regular_pentagon_l543_543130

-- Define the conditions
def is_regular_pentagon (P : ℝ) (n : ℕ) : Prop := n = 5 ∧ P = 23.4

-- State the theorem
theorem length_of_one_side_of_regular_pentagon (P : ℝ) (n : ℕ) 
  (h : is_regular_pentagon P n) : P / n = 4.68 :=
by
  sorry

end length_of_one_side_of_regular_pentagon_l543_543130


namespace locus_of_C_l543_543827

namespace AcuteTriangleWithMedianAngle

def isAcuteTriangle (A B C : Point) : Prop :=
  ∠BAC < 90 ∧ ∠ABC < 90 ∧ ∠ACB < 90

def isMedianAngle (A B C : Point) : Prop :=
  (∠B ≤ ∠A ∧ ∠A ≤ ∠C) ∨ (∠C ≤ ∠A ∧ ∠A ≤ ∠B)

theorem locus_of_C (A B : Point) (hAB : A ≠ B) :
  {C : Point | isAcuteTriangle A B C ∧ isMedianAngle A B C} = 
  {C : Point | sorry} :=
sorry

end AcuteTriangleWithMedianAngle

end locus_of_C_l543_543827


namespace problem_part1_problem_part2_l543_543101

-- Define the function f
def f (x : ℝ) : ℝ := cos x ^ 2 + (1 / 2) * sin (2 * x + π / 2) - (1 / 2)

-- Conditions and theorems
theorem problem_part1 (x : ℝ) :
    x ∈ Set.Ioo (π / 6) (2 * π / 3) →
    f x ∈ Set.Ico (-1) (1 / 2) :=
sorry

theorem problem_part2 (A B C a : ℝ) (hABC : A + B + C = π) (hC : f (C / 2) = sqrt 2 / 2) (h_c_eq_sqrt2_a : c = sqrt 2 * a) :
    A = π / 6 :=
sorry

end problem_part1_problem_part2_l543_543101


namespace find_sum_of_cubes_l543_543165

theorem find_sum_of_cubes (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : (a^3 + 9) / a = (b^3 + 9) / b)
  (h₅ : (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
by
  sorry

end find_sum_of_cubes_l543_543165


namespace semicircle_ratio_l543_543326

-- Define the problem
variables (r : ℝ) (AB BC CD DE EF FG : ℝ)

-- Assumptions
axiom distances_equal : AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FG
axiom semicircles_green : ∀ (i : ℕ), i < 3 → ∃ (r : ℝ), reducible
axiom semicircles_red : ∀ (j : ℕ), j < 3 → ∃ (r : ℝ), reducible

-- Ratio theorem statement
theorem semicircle_ratio : AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FG →
    let L_green := 3 * π * r in
    let L_red := 3 * π * r in
    L_green = L_red :=
sorry

end semicircle_ratio_l543_543326


namespace flight_duration_l543_543574

theorem flight_duration (h m : ℕ) (H1 : 11 * 60 + 7 < 14 * 60 + 45) (H2 : 0 < m) (H3 : m < 60) :
  h + m = 41 := 
sorry

end flight_duration_l543_543574


namespace relationship_among_abc_l543_543522

noncomputable def a : ℝ := (3:ℝ)^(1/3)
noncomputable def b : ℝ := ((1/4:ℝ)^(3.1))
noncomputable def c : ℝ := Real.logBase 0.4 3

theorem relationship_among_abc : a > b ∧ b > c := 
by 
  have h1 : a = 3^(1/3) := rfl
  have h2 : b = (1/4)^(3.1) := rfl
  have h3 : c = Real.logBase 0.4 3 := rfl
  sorry

end relationship_among_abc_l543_543522


namespace simplify_fraction_l543_543205

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = (65 : ℚ) / 12 := 
by
  sorry

end simplify_fraction_l543_543205


namespace letter_150_in_pattern_l543_543678

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l543_543678


namespace fraction_product_simplified_l543_543770

theorem fraction_product_simplified:
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := by
  sorry

end fraction_product_simplified_l543_543770


namespace certain_event_l543_543690

/--
Given the following events:
1. Event A: Buying a lottery ticket and winning.
2. Event B: A bag containing only 5 black balls, drawing a black ball from it.
3. Event C: Tossing a coin and getting heads.
4. Event D: Turning on the TV and an advertisement playing.

Prove that Event B is the only certain event among the given options.
-/
theorem certain_event :
  ∀ (A B C D : Prop), 
  (A → ¬ B) ∧  -- Buying a lottery ticket and winning is not certain (probability less than 1)
  (B → true) ∧ -- Drawing a black ball from a bag with only black balls is certain (probability 1)
  (C → ¬ B) ∧ -- Tossing a coin and getting heads is not certain (probability 1/2)
  (D → ¬ B) → -- Turning on the TV and an advertisement playing is not certain (probability less than 1)
  B := 
by 
  intros A B C D conditions, 
  have h_B : B := sorry, 
  exact h_B

end certain_event_l543_543690


namespace polygon_sides_l543_543296

-- Definition of the conditions
def sum_of_exterior_angles : ℝ := 360
def exterior_angle : ℝ := 45
def number_of_sides (sum_angles exterior_angle : ℝ) : ℝ := sum_angles / exterior_angle

-- The goal to prove
theorem polygon_sides 
  (sum_angles ext_angle : ℝ)
  (h₁ : sum_angles = 360)
  (h₂ : ext_angle = 45) :
  number_of_sides sum_angles ext_angle = 8 :=
by
  sorry

end polygon_sides_l543_543296


namespace triangle_angle_C_and_equilateral_l543_543431

variables (a b c A B C : ℝ)
variables (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
variables (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1)

theorem triangle_angle_C_and_equilateral (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
                                         (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
                                         (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1) :
  C = π / 3 ∧ A = π / 3 ∧ B = π / 3 :=
sorry

end triangle_angle_C_and_equilateral_l543_543431


namespace minimum_value_of_M_l543_543874

noncomputable def f (x b c : ℝ) : ℝ := (1 - x^2) * (x^2 + b * x + c)

noncomputable def M (b c : ℝ) : ℝ :=
  Real.sup (Set.image (λ x => abs (f x b c)) (Set.Icc (-1) 1))

theorem minimum_value_of_M : (min (λ bc : ℝ × ℝ => M bc.fst bc.snd)) = 3 - 2 * Real.sqrt 2 := 
begin
  sorry
end

end minimum_value_of_M_l543_543874


namespace cos_double_angle_l543_543395

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l543_543395


namespace problem_inequality_l543_543567

theorem problem_inequality (n : ℕ) (a : ℕ → ℝ) (h₁ : 0 < n) (h₂ : ∀ i, 1 ≤ a i) :
  (∏ i in finset.range n, (a i + 1)) ≥ 2 ^ (n - 1) * (finset.sum (finset.range n) a - ↑n + 2) :=
sorry

end problem_inequality_l543_543567


namespace estimate_white_chess_pieces_l543_543914

theorem estimate_white_chess_pieces 
  (total_trials : ℕ := 300) 
  (black_trials : ℕ := 100)
  (black_pieces : ℕ := 10) : 
  ∃ (white_pieces : ℕ), 
    (let probability_black := black_trials / total_trials;
         probability_white := 1 - probability_black;
         x := (2 * black_pieces) / probability_white in 
     x = 20) :=
by sorry

end estimate_white_chess_pieces_l543_543914


namespace imaginary_unit_fraction_l543_543532

theorem imaginary_unit_fraction :
  (1 - Complex.i) / (1 + Complex.i) = -Complex.i :=
by
  sorry

end imaginary_unit_fraction_l543_543532


namespace diagonal_length_of_square_l543_543814

theorem diagonal_length_of_square (A : ℝ) (hA : A = 7.22) : 
  (∃ d : ℝ, d^2 = 2 * A ∧ d ≈ 3.8) :=
by 
  sorry

end diagonal_length_of_square_l543_543814


namespace compare_trig_values_l543_543061

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 7)
noncomputable def b : ℝ := Real.tan (5 * Real.pi / 7)
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 7)

theorem compare_trig_values :
  (0 < 2 * Real.pi / 7 ∧ 2 * Real.pi / 7 < Real.pi / 2) →
  (Real.pi / 2 < 5 * Real.pi / 7 ∧ 5 * Real.pi / 7 < 3 * Real.pi / 4) →
  b < c ∧ c < a :=
by
  intro h1 h2
  sorry

end compare_trig_values_l543_543061


namespace mary_needs_canvas_l543_543964

noncomputable def canvas_needed : ℝ :=
  let rectangular_area := 5 * 8
  let right_triangle1_area := (3 * 4) / 2
  let right_triangle2_area := (4 * 6) / 2
  let equilateral_triangle_area := (4^2 * Real.sqrt 3) / 4
  let trapezoid_area := (5 + 7) * 3 / 2
  rectangular_area + right_triangle1_area + right_triangle2_area +
  equilateral_triangle_area + trapezoid_area

theorem mary_needs_canvas : canvas_needed ≈ 82.928 := 
by 
  let rectangular_area := 40
  let right_triangle1_area := 6
  let right_triangle2_area := 12
  let equilateral_triangle_area := 4 * 1.732
  let trapezoid_area := 18
  let total_area := rectangular_area + right_triangle1_area + right_triangle2_area + 
  equilateral_triangle_area + trapezoid_area
  have h_total_area : total_area ≈ 82.928 := by 
    calc
      total_area = 82.928 : sorry
  exact h_total_area

#eval canvas_needed -- for verification

end mary_needs_canvas_l543_543964


namespace centroid_of_triangle_l543_543507

-- Define the points and triangles
variables {A B C P : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P]

-- Definition of area function for a triangle
def area (A B C : Type) : ℝ := sorry -- Assuming here a placeholder for area

-- Given conditions in the problem
variables (A B C P : Type)
  (h : area A B P = area B C P ∧ area B C P = area C A P)

-- Theorem statement
theorem centroid_of_triangle (h : area A B P = area B C P ∧ area B C P = area C A P) : 
  is_centroid A B C P :=
sorry

end centroid_of_triangle_l543_543507


namespace complex_fourth_power_l543_543017

theorem complex_fourth_power (i : ℂ) (hi : i^2 = -1) : (1 - i)^4 = -4 := 
sorry

end complex_fourth_power_l543_543017


namespace terminal_side_of_alpha_second_quadrant_l543_543115

theorem terminal_side_of_alpha_second_quadrant 
  (α : ℝ) 
  (h1 : tan α < 0) 
  (h2 : cos α < 0) : 
  quadrant_of_angle α = 2 := 
sorry

end terminal_side_of_alpha_second_quadrant_l543_543115


namespace number_of_leap_years_l543_543295

noncomputable def is_leap_year (year : ℕ) : Prop :=
  (year % 1300 = 300 ∨ year % 1300 = 700) ∧ 2000 ≤ year ∧ year ≤ 5000

noncomputable def leap_years : List ℕ :=
  [2900, 4200, 3300, 4600]

theorem number_of_leap_years : leap_years.length = 4 ∧ ∀ y ∈ leap_years, is_leap_year y := by
  sorry

end number_of_leap_years_l543_543295


namespace letter_150_in_pattern_l543_543677

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l543_543677


namespace probability_multiple_of_3_or_4_l543_543625

theorem probability_multiple_of_3_or_4 :
  let numbers := Finset.range 30
  let multiples_of_3 := {n ∈ numbers | n % 3 = 0}
  let multiples_of_4 := {n ∈ numbers | n % 4 = 0}
  let multiples_of_12 := {n ∈ numbers | n % 12 = 0}
  let favorable_count := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card
  let probability := (favorable_count : ℚ) / numbers.card
  probability = (1 / 2 : ℚ) :=
by
  sorry

end probability_multiple_of_3_or_4_l543_543625


namespace triangle_BO_length_l543_543562

theorem triangle_BO_length
  (A B C D E O : Type)
  (AC_diameter : ∀ P Q : Type, AC P Q -> is_diameter_of_circle P Q)
  (circle : Circle)
  (intersects : intersects circle (connect A B) (connect B C))
  (angle_EDC : angle D E C = 30)
  (length_EC : length E C = 1)
  (area_DBE_half_ABC : area (triangle D B E) = 0.5 * area (triangle A B C))
  (O_intersection : intersection (segment A E) (segment C D) = O):
  length (segment B O) = 2 :=
by
  sorry

end triangle_BO_length_l543_543562


namespace conjugate_complex_sum_l543_543422

theorem conjugate_complex_sum (x y : ℝ) (i : ℂ) (h_i : i = complex.I)
  (h_conj : x + y * i = (complex.conj ((2 : ℂ) + i) / (complex.conj ((1 : ℂ) + i)))) : 
  x + y = 2 := 
by 
  sorry

end conjugate_complex_sum_l543_543422


namespace part_I_part_II_l543_543037

noncomputable
def x₀ : ℝ := 2

noncomputable
def f (x m : ℝ) : ℝ := |x - m| + |x + 1/m| - x₀

theorem part_I (x : ℝ) : |x + 3| - 2 * x - 1 < 0 ↔ x > 2 :=
by sorry

theorem part_II (m : ℝ) (h : m > 0) :
  (∃ x : ℝ, f x m = 0) → m = 1 :=
by sorry

end part_I_part_II_l543_543037


namespace value_after_jumps_is_correct_l543_543260

-- Define the conditions as variables
def initial_value : ℕ := 320
def number_of_jumps : ℕ := 4
def reduction_per_jump : ℕ := 10

-- Lean 4 proof statement
theorem value_after_jumps_is_correct (initial_value : ℕ) (number_of_jumps : ℕ) (reduction_per_jump : ℕ) (final_value : ℕ) :
  (initial_value - number_of_jumps * reduction_per_jump) = final_value := by
  sorry

-- Assign the conditions and expected final value
#eval value_after_jumps_is_correct 320 4 10 280

end value_after_jumps_is_correct_l543_543260


namespace yogurt_combinations_l543_543314

theorem yogurt_combinations (f : ℕ) (t : ℕ) (h_f : f = 4) (h_t : t = 6) :
  (f * (t.choose 2) = 60) :=
by
  rw [h_f, h_t]
  sorry

end yogurt_combinations_l543_543314


namespace simplify_sqrt_l543_543687

variable (m n : ℝ)
variable (h : m < 0)

theorem simplify_sqrt (h1 : m < 0) : sqrt (m^2 * n) = -m * sqrt n :=
  sorry

end simplify_sqrt_l543_543687


namespace arrange_leopards_correct_l543_543959

-- Definitions for conditions
def num_shortest : ℕ := 3
def total_leopards : ℕ := 9
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Calculation of total ways to arrange given conditions
def arrange_leopards (num_shortest : ℕ) (total_leopards : ℕ) : ℕ :=
  let choose2short := (num_shortest * (num_shortest - 1)) / 2
  let arrange2short := 2 * factorial (total_leopards - num_shortest)
  choose2short * arrange2short * factorial (total_leopards - num_shortest)

theorem arrange_leopards_correct :
  arrange_leopards num_shortest total_leopards = 30240 := by
  sorry

end arrange_leopards_correct_l543_543959


namespace solve_for_x_l543_543127

theorem solve_for_x {x : ℝ} (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (15 * x) * Real.sqrt (4 * x) * Real.sqrt (10 * x) = 20) :
  x = 2^(1/4) / Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end solve_for_x_l543_543127


namespace cos_double_angle_sum_l543_543374

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l543_543374


namespace parabola_equation_is_correct_distance_MN_is_correct_l543_543094

-- Definitions and conditions
def hyperbola (a b : ℝ) (x y : ℝ) := (x^2) / a^2 - (y^2) / b^2 = 1
def parabola (p x y : ℝ) := y^2 = 2 * p * x
def vertex := (0, 0) -- The vertex of the parabola at the origin
def axisOfSymmetryIsCoordinateAxis := ∀ (x y : ℝ), (x = 0 → y = y)
def directrix (x c : ℝ) := x = -c

noncomputable def equation_of_parabola := parabola 4

theorem parabola_equation_is_correct : 
  ∃ p, p = 4 ∧ parabola p := 
  by
    sorry -- Proof required here

def point_in_line_with_slope (x₀ y₀ m x y : ℝ) := y - y₀ = m * (x - x₀)
def distance (M N : ℝ × ℝ) := real.sqrt (((N.1 - M.1)^2) + ((N.2 - M.2)^2))

-- Given a point P and a line equation passing through P with slope 1
def P := (3, 1)
def line_through_P := point_in_line_with_slope P.1 P.2 1

-- Given that the line intersects the parabola
def intersect_points (M N : ℝ × ℝ) := 
  (∃ x₁ y₁, M = (x₁, y₁) ∧ line_through_P x₁ y₁) ∧ 
  (∃ x₂ y₂, N = (x₂, y₂) ∧ line_through_P x₂ y₂)

noncomputable
def MN_distance (M N : ℝ × ℝ) := 
  |M.1 + N.1 - 2 * (M.1 * N.1)^0.5| -- Simplified distance formula in terms of x₁ and x₂

theorem distance_MN_is_correct : 
  ∀ M N, intersect_points M N → MN_distance M N = 16 :=
  by
    sorry -- Proof required here

end parabola_equation_is_correct_distance_MN_is_correct_l543_543094


namespace distance_between_foci_of_ellipse_l543_543811

theorem distance_between_foci_of_ellipse (x y : ℝ) :
  x^2 + 9 * y^2 = 324 →
  ∃ d : ℝ, d = 24 * real.sqrt 2 :=
by
  intros h
  exist 24 * real.sqrt 2
  sorry

end distance_between_foci_of_ellipse_l543_543811


namespace greatest_closed_broken_line_l543_543850

-- Define the structure of a convex (2n+1)-gon labeled with the vertices A_i
variables (n : ℕ)
variables (A : fin (2*n + 1) → Type) -- A type over the vertices

-- Assume a convex polygon with vertices A_1, A_3, A_5, ..., A_{2n+1}, A_2, ..., A_{2n}
axiom convex_polygon : ∀ (i j k : fin (2*n + 1)), convex (A i) (A j) (A k)

-- Define the closed broken line with the given vertices in the sequence
def closed_broken_line_longest : Prop :=
  ∀ (B : list (fin (2*n + 1))), 
  (∃ (perm : list (fin (2*n + 1)) → Prop), perm B) →
  length (A 0) (A 1) + length (A 1) (A 2) + ... + length (A (2*n)) (A 0) ≥ length_of_any_permutation B

-- The main proof statement
theorem greatest_closed_broken_line :
  closed_broken_line_longest n A :=
sorry

end greatest_closed_broken_line_l543_543850


namespace cos_double_angle_sum_l543_543375

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l543_543375


namespace line_circle_intersection_l543_543891

theorem line_circle_intersection :
  let line_eq : ℝ → ℝ → Prop := λ x y, 3 * x + 4 * y = 12
  let circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 16
  ∃ (x₁ y₁ x₂ y₂ : ℝ), line_eq x₁ y₁ ∧ circle_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧ circle_eq x₂ y₂ ∧ x₁ ≠ x₂ :=
sorry

end line_circle_intersection_l543_543891


namespace inductive_reasoning_characterization_l543_543349

def is_inductive_reasoning (i : ℕ) : Prop :=
  i = 2 ∨ i = 4

theorem inductive_reasoning_characterization :
  ∀ i : ℕ, (i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4) → is_inductive_reasoning i ↔ i = 2 ∨ i = 4 := by
  intros i hi
  cases hi with
  | inl h => left; exact sorry
  | inr hi =>
    cases hi with
    | inl h => left; exact sorry
    | inr hi =>
      cases hi with
      | inl h => right; left; exact sorry
      | inr h => right; right; exact sorry

end inductive_reasoning_characterization_l543_543349


namespace inscribed_circle_radius_l543_543500

theorem inscribed_circle_radius
  (A p s : ℝ) (h1 : A = p) (h2 : s = p / 2) (r : ℝ) (h3 : A = r * s) :
  r = 2 :=
sorry

end inscribed_circle_radius_l543_543500


namespace number_of_strictly_increasing_sequences_l543_543816

def strictly_increasing_sequences (n : ℕ) : ℕ :=
if n = 0 then 1 else if n = 1 then 1 else strictly_increasing_sequences (n - 1) + strictly_increasing_sequences (n - 2)

theorem number_of_strictly_increasing_sequences :
  strictly_increasing_sequences 12 = 144 :=
by
  sorry

end number_of_strictly_increasing_sequences_l543_543816


namespace max_flow_increase_proof_l543_543325

noncomputable def max_flow_increase : ℕ :=
  sorry

theorem max_flow_increase_proof
  (initial_pipes_AB: ℕ) (initial_pipes_BC: ℕ) (flow_increase_per_pipes_swap: ℕ)
  (swap_increase: initial_pipes_AB = 10)
  (swap_increase_2: initial_pipes_BC = 10)
  (flow_increment: flow_increase_per_pipes_swap = 30) : 
  max_flow_increase = 150 :=
  sorry

end max_flow_increase_proof_l543_543325


namespace problem_l543_543065

-- Define the sequence {a_n}
noncomputable def a : ℕ → ℝ
| 0       := 2
| (n + 1) := Real.sqrt (4 + (a n) * (a n))

-- Define the sum sequence
noncomputable def sum_seq (n : ℕ) : ℝ :=
(∑ i in Finset.range n, 1 / (a (i + 1) + a i))

-- Main theorem to prove
theorem problem (n : ℕ) (h1 : 0 < n) (h2 : a 0 = 2) 
    (h3 : ∀ n, a (n + 1) - a n = (4 / (a (n + 1) + a n))) 
    (h4 : sum_seq n = 5) : 
  n = 120 :=
s

end problem_l543_543065


namespace evaluate_expression_l543_543805

variable (x y : ℝ)

theorem evaluate_expression
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hsum_sq : x^2 + y^2 ≠ 0)
  (hsum : x + y ≠ 0) :
    (x^2 + y^2)⁻¹ * ((x + y)⁻¹ + (x / y)⁻¹) = (1 + y) / ((x^2 + y^2) * (x + y)) :=
sorry

end evaluate_expression_l543_543805


namespace letter_150_in_pattern_l543_543675

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l543_543675


namespace least_value_a2000_l543_543302

theorem least_value_a2000 (a : ℕ → ℕ)
  (h1 : ∀ m n, (m ∣ n) → (m < n) → (a m ∣ a n))
  (h2 : ∀ m n, (m ∣ n) → (m < n) → (a m < a n)) :
  a 2000 >= 128 :=
sorry

end least_value_a2000_l543_543302


namespace area_of_tangency_triangle_l543_543016

noncomputable def area_of_triangle : ℝ :=
  let r1 := 2
  let r2 := 3
  let r3 := 4
  let s := (r1 + r2 + r3) / 2
  let A := Real.sqrt (s * (s - (r1 + r2)) * (s - (r2 + r3)) * (s - (r1 + r3)))
  let inradius := A / s
  let area_points_of_tangency := A * (inradius / r1) * (inradius / r2) * (inradius / r3)
  area_points_of_tangency

theorem area_of_tangency_triangle :
  area_of_triangle = (16 * Real.sqrt 6) / 3 :=
sorry

end area_of_tangency_triangle_l543_543016


namespace cos_double_angle_l543_543381

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l543_543381


namespace triangle_area_correct_l543_543566

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_correct : 
  area_of_triangle (0, 0) (2, 0) (2, 3) = 3 :=
by
  sorry

end triangle_area_correct_l543_543566


namespace inequality_condition_l543_543749

-- Define the inequality (x - 2) * (x + 2) > 0
def inequality_holds (x : ℝ) : Prop := (x - 2) * (x + 2) > 0

-- The sufficient and necessary condition for the inequality to hold is x > 2 or x < -2
theorem inequality_condition (x : ℝ) : inequality_holds x ↔ (x > 2 ∨ x < -2) :=
  sorry

end inequality_condition_l543_543749


namespace railway_max_passengers_l543_543035

-- Step 1: Define that y is a linear function of x
def is_linear_function (y : ℕ → ℤ) : Prop :=
∃ (a b : ℤ), ∀ x : ℕ, y x = a * x + b

-- Step 2: Given conditions
def satisfies_conditions (y : ℕ → ℤ) : Prop :=
( y 4 = 16 ∧ y 7 = 10 )

-- Step 3: Define functions
def round_trip_function (x : ℕ) : ℤ :=
-2 * x + 24

def carriages_function (x : ℕ) : ℤ :=
-2 * x^2 + 24 * x

def passengers_function (x : ℕ) : ℤ :=
-220 * x^2 + 2640 * x

-- Step 4: Prove equivalent proof problem
theorem railway_max_passengers : 
  is_linear_function round_trip_function ∧
  satisfies_conditions round_trip_function ∧
  (∀ x, carriages_function x = (-2) * x^2 + 24 * x) ∧
  (argmax_passengers : ∀ x : ℕ, 0 ≤ x ∧ x ≤ 12 → x = 6 ∧ passengers_function 6 = 7920) :=
by
  sorry

end railway_max_passengers_l543_543035


namespace max_license_plates_l543_543317

theorem max_license_plates :
  let α := {A, B, C, D, E}
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let choose2letters := (Finset.powersetLen 2 α).card
  let choose3consecutiveNumbers := 8  -- since there are 8 sets of 3 consecutive numbers
  let permute5elements := 5!
  let permute3Numbers := 3!
  choose2letters * choose3consecutiveNumbers * permute5elements * permute3Numbers = 43200 
:= sorry

end max_license_plates_l543_543317


namespace product_of_c_l543_543051

-- Definition of the cubic polynomial
def P (x c : ℝ) : ℝ := 3 * x^3 + 9 * x^2 + 17 * x + c

-- Statement to prove the desired product
theorem product_of_c (h : ∃ x : ℝ, ∀ c : ℝ, P x c = 0) :
  (∏ c in finset.range (11 + 1).filter (λ c, 0 < c), c) = 39916800 := by
  sorry

end product_of_c_l543_543051


namespace sum_of_legs_eq_40_l543_543604

theorem sum_of_legs_eq_40
  (x : ℝ)
  (h1 : x > 0)
  (h2 : x^2 + (x + 2)^2 = 29^2) :
  x + (x + 2) = 40 :=
by
  sorry

end sum_of_legs_eq_40_l543_543604


namespace quadrilateral_area_is_8_l543_543834

noncomputable section
open Real

def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def origin_symmetric (P Q : ℝ × ℝ) : Prop := P.1 = -Q.1 ∧ P.2 = -Q.2

def distance (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_quadrilateral (P Q F1 F2 : ℝ × ℝ) : Prop :=
  ∃ a b c d, a = P ∧ b = F1 ∧ c = Q ∧ d = F2

def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1*B.2 + B.1*C.2 + C.1*D.2 + D.1*A.2 - (B.1*A.2 + C.1*B.2 + D.1*C.2 + A.1*D.2))

theorem quadrilateral_area_is_8 (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  origin_symmetric P Q →
  distance P Q = distance f1 f2 →
  is_quadrilateral P Q f1 f2 →
  area_of_quadrilateral P f1 Q f2 = 8 := 
by
  sorry

end quadrilateral_area_is_8_l543_543834


namespace residential_ratio_l543_543717

theorem residential_ratio (B R O E : ℕ) (h1 : B = 300) (h2 : E = 75) (h3 : E = O ∧ R + 2 * E = B) : R / B = 1 / 2 :=
by
  sorry

end residential_ratio_l543_543717


namespace rate_per_square_meter_l543_543607

-- Define the conditions
def length (L : ℝ) := L = 8
def width (W : ℝ) := W = 4.75
def total_cost (C : ℝ) := C = 34200
def area (A : ℝ) (L W : ℝ) := A = L * W
def rate (R C A : ℝ) := R = C / A

-- The theorem to prove
theorem rate_per_square_meter (L W C A R : ℝ) 
  (hL : length L) (hW : width W) (hC : total_cost C) (hA : area A L W) : 
  rate R C A :=
by
  -- By the conditions, length is 8, width is 4.75, and total cost is 34200.
  simp [length, width, total_cost, area, rate] at hL hW hC hA ⊢
  -- It remains to calculate the rate and use conditions
  have hA : A = L * W := hA
  rw [hL, hW] at hA
  have hA' : A = 8 * 4.75 := by simp [hA]
  rw [hA']
  simp [rate]
  sorry -- The detailed proof is omitted.

end rate_per_square_meter_l543_543607


namespace sin_half_gamma_le_c_div_a_add_b_l543_543193

theorem sin_half_gamma_le_c_div_a_add_b
  {a b c : ℝ}  -- side lengths are real numbers
  {γ : ℝ}      -- angle γ at vertex C is a real number
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angle : 0 < γ ∧ γ < π)  -- γ is between 0 and π
  (h_sin_half_gamma : sin (γ / 2) = c / (a + b))
  : sin (γ / 2) ≤ c / (a + b) :=
begin
  sorry
end

end sin_half_gamma_le_c_div_a_add_b_l543_543193


namespace max_discount_l543_543718

theorem max_discount (cost_price selling_price : ℝ) (min_profit_margin : ℝ) (x : ℝ) : 
  cost_price = 400 → selling_price = 500 → min_profit_margin = 0.0625 → 
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin * cost_price) → x ≤ 15 :=
by
  intros h1 h2 h3 h4
  sorry

end max_discount_l543_543718


namespace min_transaction_amount_l543_543226

theorem min_transaction_amount (y : ℝ) (h : 2000 = 2 * 10^10 * y^(-2)) : y = 10^(3.5) := 
by
  sorry

end min_transaction_amount_l543_543226


namespace volume_of_sector_l543_543243

-- Define the conditions
variables (V : ℝ) (α : ℝ)

-- Define the statement to prove
theorem volume_of_sector (V : ℝ) (α : ℝ) : 
  V * (sin (α / 4))^2 = V * (sin (α / 4))^2 :=
sorry

end volume_of_sector_l543_543243


namespace divide_figure_l543_543364

-- Definitions for the centers and disks
structure Center (α : Type) :=
  (x : α)
  (y : α)

variable {α : Type} [LinearOrder α]

-- Points representing the centers of the disks
def A : Center α := sorry 
def B : Center α := sorry 
def C : Center α := sorry 
def D : Center α := sorry 
def F : Center α := sorry 

-- Center O of the square ABCD
def O : Center α :=
  { x := (A.x + B.x + C.x + D.x) / 4,
    y := (A.y + B.y + C.y + D.y) / 4 }

-- Line FO
def line_FO (F O : Center α) : Prop :=
  sorry  -- Definition of a line through two points

theorem divide_figure (F O : Center α) :
  line_FO F O → divides_five_disks_equally F O :=
begin
  sorry  -- Proof not required, only the statement
end

end divide_figure_l543_543364


namespace exists_clique_of_4_l543_543917

/-- 
In a graph G with 20 vertices where each vertex has a degree of at least 14, 
prove that there exists a clique of size 4. 
--/
theorem exists_clique_of_4 (G : SimpleGraph (Fin 20))
  (h : ∀ v : Fin 20, 14 ≤ (G.degree v)) : 
  ∃ (K : Finset (Fin 20)), K.card = 4 ∧ ∀ (u v : (Fin 20)), u ∈ K → v ∈ K → u ≠ v → G.adj u v :=
sorry

end exists_clique_of_4_l543_543917


namespace distance_between_foci_of_ellipse_l543_543760

theorem distance_between_foci_of_ellipse :
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  2 * c = 4 * Real.sqrt 15 :=
by
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  show 2 * c = 4 * Real.sqrt 15
  sorry

end distance_between_foci_of_ellipse_l543_543760


namespace determine_k_l543_543445

noncomputable def given_values {x : ℝ} (x₁ x₂ k : ℝ) : Prop :=
  x₁ + x₂ = -5 ∧ x₁ * x₂ = k ∧ (x₁ - x₂)^2 = 9

theorem determine_k (x₁ x₂ k : ℝ) (h : given_values x₁ x₂ k) : k = 4 :=
by {
  have h₁ : (x₁ + x₂) = -5 := h.1,
  have h₂ : (x₁ * x₂) = k := h.2.1,
  have h₃ : (x₁ - x₂)^2 = 9 := h.2.2,
  suffices : 25 - 4 * k = 9,
  -- k should be 4 from this equation
  sorry
}

end determine_k_l543_543445


namespace right_triangle_area_l543_543603

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 10) (h_angle : angle = 30) :
  let x := hypotenuse / 2 in
  let base := x in
  let height := x * Real.sqrt 3 in
  (1 / 2) * base * height = 25 * Real.sqrt 3 / 2 :=
by
  intros
  sorry

end right_triangle_area_l543_543603


namespace isosceles_triangle_perimeter_l543_543501

theorem isosceles_triangle_perimeter 
  (a b : ℕ) 
  (h_iso : a = b ∨ a = 3 ∨ b = 3) 
  (h_sides : a = 6 ∨ b = 6) 
  : a + b + 3 = 15 := by
  sorry

end isosceles_triangle_perimeter_l543_543501


namespace probability_multiple_of_3_or_4_l543_543623

-- Given the numbers 1 through 30 are written on 30 cards one number per card,
-- and Sara picks one of the 30 cards at random,
-- the probability that the number on her card is a multiple of 3 or 4 is 1/2.

-- Define the set of numbers from 1 to 30
def numbers := finset.range 30 \ {0}

-- Define what it means to be a multiple of 3 or 4 within the given range
def is_multiple_of_3_or_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 4 = 0

-- Define the set of multiples of 3 or 4 within the given range
def multiples_of_3_or_4 := numbers.filter is_multiple_of_3_or_4

-- The probability calculation
theorem probability_multiple_of_3_or_4 : 
  (multiples_of_3_or_4.card : ℚ) / numbers.card = 1 / 2 :=
begin
  -- The set multiples_of_3_or_4 contains 15 elements
  have h_multiples_card : multiples_of_3_or_4.card = 15, sorry,
  -- The set numbers contains 30 elements
  have h_numbers_card : numbers.card = 30, sorry,
  -- Therefore, the probability is 15/30 = 1/2
  rw [h_multiples_card, h_numbers_card],
  norm_num,
end

end probability_multiple_of_3_or_4_l543_543623


namespace find_last_date_l543_543273

def sum_digits (date : Nat) : Nat :=
  date.digits.sum

def valid_date (d m y : Nat) : Prop :=
  d <= 31 ∧ m <= 12 ∧ y = 2008

def date_problem (d m y : Nat) : Prop :=
  let d1 := d / 10
  let d2 := d % 10
  let m1 := m / 10
  let m2 := m % 10
  let y1 := (y / 1000) % 10
  let y2 := (y / 100) % 10
  let y3 := (y / 10) % 10
  let y4 := y % 10
  valid_date d m y ∧ (d1 + d2 + m1 + m2 = y1 + y2 + y3 + y4)

theorem find_last_date : ∃ (d m : Nat), date_problem d m 2008 ∧
                         ∀ (d' m' : Nat), date_problem d' m' 2008 → d' * 100 + m' <= d * 100 + m :=
  ⟨25, 12, 
  by {
    -- Proof that 25.12.2008 meets the conditions would go here, using sorry for now.
    sorry
  }⟩

end find_last_date_l543_543273


namespace parallelepiped_pentagon_angles_l543_543745

theorem parallelepiped_pentagon_angles :
  ∀ (pentagon : Type) [pentagon.is_section_of_parallelepiped]
  (side_ratio : ℝ → ℝ → Prop),
  (∀ (a b : ℝ), side_ratio a b → (a = b ∨ a = 2 * b ∨ a = b / 2)) →
  ∃ (angles : list ℝ),
  length angles = 5 ∧
  (count angles 120 = 4) ∧
  (count angles 60 = 1) :=
begin
  sorry
end

end parallelepiped_pentagon_angles_l543_543745


namespace find_lambda_l543_543888

def vector_a (λ : ℝ) := (λ + 1, 2 : ℝ)
def vector_b := (1, -2 : ℝ)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem find_lambda (λ : ℝ) :
  collinear (vector_a λ) vector_b ↔ λ = -2 :=
by sorry

end find_lambda_l543_543888


namespace intersection_M_N_l543_543464

-- Define the set M based on the given condition
def M : Set ℝ := { x | x^2 > 1 }

-- Define the set N based on the given elements
def N : Set ℝ := { x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 }

-- Prove that the intersection of M and N is {-2, 2}
theorem intersection_M_N : M ∩ N = { -2, 2 } := by
  sorry

end intersection_M_N_l543_543464


namespace student_percentage_l543_543310

theorem student_percentage (s1 s3 overall : ℕ) (percentage_second_subject : ℕ) :
  s1 = 60 →
  s3 = 85 →
  overall = 75 →
  (s1 + percentage_second_subject + s3) / 3 = overall →
  percentage_second_subject = 80 := by
  intros h1 h2 h3 h4
  sorry

end student_percentage_l543_543310


namespace right_triangle_labeling_inequality_l543_543145

-- Definition of right triangle and the problem statement
noncomputable def right_triangle_problem (A B C : Point) (P : Fin n -> Point) (c : ℝ) : Prop :=
  let hypotenuse := dist A B in
  (angle C = 90) ∧
  (hypotenuse = c) →
  (∃ P_1, ∃ P_2, ..., ∃ P_n, Σ t : Fin n, nat (dist P_1 P_2 ^ 2 + dist P_2 P_3 ^ 2 + ... + dist P_{n-1} P_n ^ 2 ≤ c^2))

-- Declaration of the main theorem
theorem right_triangle_labeling_inequality (n : ℕ) :
  ∀ (A B C : Point) (P : Fin n -> Point) (c : ℝ),
    right_triangle_problem A B C P c :=
by
  sorry

end right_triangle_labeling_inequality_l543_543145


namespace triangle_larger_segment_cutoff_l543_543240

open Real

theorem triangle_larger_segment_cutoff (a b c h s₁ s₂ : ℝ) (habc : a = 35) (hbc : b = 85) (hca : c = 90)
  (hh : h = 90)
  (eq₁ : a^2 = s₁^2 + h^2)
  (eq₂ : b^2 = s₂^2 + h^2)
  (h_sum : s₁ + s₂ = c) :
  max s₁ s₂ = 78.33 :=
by
  sorry

end triangle_larger_segment_cutoff_l543_543240


namespace find_a_values_l543_543490

noncomputable def find_a (b c : ℝ) (B : ℝ) : Set ℝ :=
  {a : ℝ | b^2 = a^2 + c^2 - 2 * a * c * real.cos B}

theorem find_a_values :
  find_a (real.sqrt 3) 3 (real.pi / 6) = {real.sqrt 3, 2 * real.sqrt 3} :=
by
  -- Proof can be constructed here
  sorry

end find_a_values_l543_543490


namespace trays_needed_l543_543184

theorem trays_needed (cookies_classmates cookies_teachers cookies_per_tray : ℕ) 
  (hc1 : cookies_classmates = 276) 
  (hc2 : cookies_teachers = 92) 
  (hc3 : cookies_per_tray = 12) : 
  (cookies_classmates + cookies_teachers + cookies_per_tray - 1) / cookies_per_tray = 31 :=
by
  sorry

end trays_needed_l543_543184


namespace cos_double_angle_proof_l543_543418

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l543_543418


namespace max_min_difference_l543_543247

def avg_three (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem max_min_difference (x y : ℝ) (h : avg_three 23 x y = 31) : (max (max 23 x) y) - (min (min 23 x) y) = 17 := 
by
  sorry

end max_min_difference_l543_543247


namespace solve_for_x_l543_543895

theorem solve_for_x (x : ℤ) (h : 3 * x - 5 = 4 * x + 10) : x = -15 :=
sorry

end solve_for_x_l543_543895


namespace find_a_for_tangency_l543_543442

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1

def circle2 (x y a : ℝ) : Prop := (x + 4)^2 + (y - a)^2 = 25

theorem find_a_for_tangency (a : ℝ) :
  (∀ (x y : ℝ), circle1 x y ↔ (x = 0 ∧ y = 0)) ∧
  (∀ (x y : ℝ), circle2 x y a ↔ (x = -4 ∧ y = a)) ∧
  ((4^2 + a^2 = 36) ∨ (4^2 + a^2 = 16)) →
  a = 0 ∨ a = 2 * real.sqrt 5 ∨ a = -2 * real.sqrt 5 := by
  sorry

end find_a_for_tangency_l543_543442


namespace solve_trig_eq_l543_543045

open Real

theorem solve_trig_eq (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 :=
by
  sorry

end solve_trig_eq_l543_543045


namespace problem_eval_at_x_eq_3_l543_543352

theorem problem_eval_at_x_eq_3 : ∀ x : ℕ, x = 3 → (x^x)^(x^x) = 27^27 :=
by
  intros x hx
  rw [hx]
  sorry

end problem_eval_at_x_eq_3_l543_543352


namespace abs_eq_sol_l543_543472

theorem abs_eq_sol {x : ℝ} : (|x - 9| = |x + 3|) → ∃! x, |x - 9| = |x + 3| := by
  sorry

end abs_eq_sol_l543_543472


namespace max_mn_on_parabola_l543_543632

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end max_mn_on_parabola_l543_543632


namespace convert_neg_angle_l543_543028

theorem convert_neg_angle (k : ℤ) (α : ℝ) (h₀ : 0 ≤ α) (h₁ : α < 360) :
  -1125 = k * 360 + α ↔ k = -4 ∧ α = 315 := 
by
  constructor
  { sorry }

end convert_neg_angle_l543_543028


namespace olivia_had_initially_l543_543559

noncomputable def olivia_initial_money (nigel_money : ℕ) (cost_per_ticket : ℕ) (num_tickets : ℕ) (total_remaining : ℕ) : ℕ :=
let total_initial_money := num_tickets * cost_per_ticket + total_remaining in
total_initial_money - nigel_money

theorem olivia_had_initially (nigel_money : ℕ) (cost_per_ticket : ℕ) (num_tickets : ℕ) (total_remaining : ℕ) :
  nigel_money = 139 → cost_per_ticket = 28 → num_tickets = 6 → total_remaining = 83 → olivia_initial_money nigel_money cost_per_ticket num_tickets total_remaining = 112 :=
by
  intros h1 h2 h3 h4
  simp [olivia_initial_money, h1, h2, h3, h4]
  sorry

end olivia_had_initially_l543_543559


namespace det_of_2C_mul_D_l543_543896

variable {C D : Matrix (Fin 3) (Fin 3) ℝ}

theorem det_of_2C_mul_D (hC : det C = 3) (hD : det D = 8) : det (2 • (C ⬝ D)) = 192 := by sorry

end det_of_2C_mul_D_l543_543896


namespace angle_ratio_in_triangle_l543_543153

theorem angle_ratio_in_triangle
  (triangle : Type)
  (A B C P Q M : triangle)
  (angle : triangle → triangle → triangle → ℝ)
  (ABC_half : angle A B Q = angle Q B C)
  (BP_BQ_bisect_ABC : angle A B P = angle P B Q)
  (BM_bisects_PBQ : angle M B Q = angle M B P)
  : angle M B Q / angle A B Q = 1 / 4 :=
by 
  sorry

end angle_ratio_in_triangle_l543_543153


namespace longer_diagonal_of_rhombus_l543_543299

theorem longer_diagonal_of_rhombus {a b d1 : ℕ} (h1 : a = b) (h2 : a = 65) (h3 : d1 = 60) : 
  ∃ d2, (d2^2) = (2 * (a^2) - (d1^2)) ∧ d2 = 110 :=
by
  sorry

end longer_diagonal_of_rhombus_l543_543299


namespace sum_of_fraction_values_l543_543044

theorem sum_of_fraction_values : 
  ∃ n1 n2 : ℕ, 
  (8 * n1 + 157) / (4 * n1 + 7) = 15 ∧ 
  (8 * n2 + 157) / (4 * n2 + 7) = 3 ∧ 
  n1 ≠ n2 ∧ 
  n1 > 0 ∧ n2 > 0 ∧ 
  (15 + 3 = 18) :=
by { use [1, 34], 
     split,
     { norm_num },
     split,
     { norm_num },
     split,
     { exact dec_trivial },
     split,
     { norm_num },
     split,
     { norm_num },
     exact dec_trivial }

end sum_of_fraction_values_l543_543044


namespace maximize_sum_pairs_l543_543256

theorem maximize_sum_pairs (l : List ℤ) (target_sum : ℤ) :
  l = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] →
  target_sum = 12 →
  (∃ x ∈ l, ∀ y ≠ x, y ∈ l → x = 6) :=
by
  intros h_list h_target
  use 6
  split
  { rw h_list, simp }
  { intros y hy h_mem
    rw h_list at h_mem, simp at h_mem,
    -- here we need to show that removing any number other than 6 reduces the count of valid pairs
    sorry
  }

end maximize_sum_pairs_l543_543256


namespace faster_train_speed_l543_543651

theorem faster_train_speed:
  ∃ (v: ℝ), 
    (let slower_train_speed : ℝ := v in
    let faster_train_speed : ℝ := 2 * v in
    let total_distance : ℝ := 100 + 100 in
    let relative_speed : ℝ := slower_train_speed + faster_train_speed in
    let time : ℝ := 8 in
    relative_speed * time = total_distance → faster_train_speed = 16.66) := 
sorry

end faster_train_speed_l543_543651


namespace binom_2p_p_mod_p_l543_543950

theorem binom_2p_p_mod_p (p : ℕ) (hp : p.Prime) : Nat.choose (2 * p) p ≡ 2 [MOD p] := 
by
  sorry

end binom_2p_p_mod_p_l543_543950


namespace cos_double_angle_l543_543397

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l543_543397


namespace solution_set_inequality_l543_543478

def f (x : ℝ) : ℝ := x ^ (1 / 4 : ℝ)

theorem solution_set_inequality (x : ℝ) (h : 0 ≤ x) (h_increasing : ∀ a b : ℝ, 0 ≤ a → 0 ≤ b → a ≤ b → f(a) ≤ f(b)) :
  f(x) > f(8 * x - 16) ↔ 2 ≤ x ∧ x < 16 / 7 := 
sorry

end solution_set_inequality_l543_543478


namespace jelly_bean_probability_l543_543729

theorem jelly_bean_probability :
  ∀ (P_red P_orange P_green P_yellow : ℝ),
  P_red = 0.1 →
  P_orange = 0.4 →
  P_green = 0.2 →
  P_red + P_orange + P_green + P_yellow = 1 →
  P_yellow = 0.3 :=
by
  intros P_red P_orange P_green P_yellow h_red h_orange h_green h_sum
  sorry

end jelly_bean_probability_l543_543729


namespace bernardo_wins_l543_543764

/-- 
Bernardo and Silvia play the following game. An integer between 0 and 999 inclusive is selected
and given to Bernardo. Whenever Bernardo receives a number, he doubles it and passes the result 
to Silvia. Whenever Silvia receives a number, she adds 50 to it and passes the result back. 
The winner is the last person who produces a number less than 1000. The smallest initial number 
that results in a win for Bernardo is 16, and the sum of the digits of 16 is 7.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem bernardo_wins (N : ℕ) (h : 16 ≤ N ∧ N ≤ 18) : sum_of_digits 16 = 7 :=
by
  sorry

end bernardo_wins_l543_543764


namespace length_of_square_side_in_right_triangle_l543_543572

noncomputable def side_length_of_square (a b : ℕ) : ℚ := 
if h : a ^ 2 + b ^ 2 = (real.sqrt (a ^ 2 + b ^ 2)) ^ 2 
then (real.sqrt ((a * b) ^ 2 / (a ^ 2 + b ^ 2))) 
else 0

theorem length_of_square_side_in_right_triangle : 
  let a := 5
  let b := 12
  let c := real.sqrt (a ^ 2 + b ^ 2)
  c = 13 ∧ a ^ 2 + b ^ 2 = c ^ 2 → 
  side_length_of_square a b = 156 / 25 :=
by
  intros
  sorry

end length_of_square_side_in_right_triangle_l543_543572


namespace no_integer_solutions_l543_543995

theorem no_integer_solutions (x y z : ℤ) : 28 ^ x ≠ 19 ^ y + 87 ^ z :=
by sorry

end no_integer_solutions_l543_543995


namespace Iso_E_is_group_l543_543525

-- Define the Euclidean distance function on real coordinate space
def euclidean_distance {n : ℕ} (x y : ℝ^n) : ℝ :=
  real.sqrt (finset.sum (finset.range n) (λ i, (x i - y i)^2))

-- Define an isometry on a non-empty subset of ℝ^n
def isometry {n : ℕ} (E : set (ℝ^n)) (f : ℝ^n → ℝ^n) : Prop :=
  ∀ x y ∈ E, euclidean_distance (f x) (f y) = euclidean_distance x y

-- Define the set of isometries on E
def Iso {n : ℕ} (E : set (ℝ^n)) : set (ℝ^n → ℝ^n) :=
  { f | isometry E f }

-- Prove that (Iso(E), ∘) is a group
theorem Iso_E_is_group {n : ℕ} (E : set (ℝ^n)) (hE : E.nonempty) : group (Iso E) :=
begin
  sorry
end

end Iso_E_is_group_l543_543525


namespace cubic_function_increasing_l543_543872

noncomputable def f (a x : ℝ) := x ^ 3 + a * x ^ 2 + 7 * a * x

theorem cubic_function_increasing (a : ℝ) (h : 0 ≤ a ∧ a ≤ 21) :
    ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
sorry

end cubic_function_increasing_l543_543872


namespace no_real_solution_range_of_a_l543_543713

theorem no_real_solution_range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| + |x - 2| < a)) → a ≤ 3 :=
by
  sorry  -- Proof skipped

end no_real_solution_range_of_a_l543_543713


namespace smallest_integer_x_l543_543684

theorem smallest_integer_x (x : ℤ) (h : x < 3 * x - 15) : x ≥ 8 :=
begin
  sorry
end

example : (∃ (x : ℤ), x < 3 * x - 15 ∧ x = 8) :=
begin
  use 8,
  split,
  { norm_num, },
  { refl, }
end

end smallest_integer_x_l543_543684


namespace cos_double_angle_l543_543385

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l543_543385


namespace paint_needed_for_similar_statues_l543_543738

theorem paint_needed_for_similar_statues :
  (∀ (n m : ℕ) (h₁ : n > 0) (h₂ : m > 0), let scale := (n / m : ℚ) 
   in 2 * (scale ^ 2) * m = paint_needed (h₁, h₂))
  → let paint_4ft := 120 * (1 / 4) * 2
    let paint_2ft := 80 * (1 / 16) * 2
    in paint_4ft + paint_2ft = 70 := sorry

end paint_needed_for_similar_statues_l543_543738


namespace tray_height_l543_543309

theorem tray_height (a b m n : ℕ) (m_lt_1000 : m < 1000) 
  (m_not_div_nth_prime : ∀ k : ℕ, Prime k → ¬ k^(n : ℕ) ∣ m) 
  (sqrt_expr : {a} = 4 ∧ b = 1) 
  (height_expr : \sqrt[b][n]{m}) 
  (tray_conditions : 
    length_100 : side_length = 100 ∧
    dist_17 : from_corner_dist = \sqrt{17} ∧
    angle_60 : angle_cut = 60 (degrees)) :
  m + n = 871 :=
by
  sorry

end tray_height_l543_543309


namespace part1_1_part1_2_part1_3_general_final_value_l543_543105

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem part1_1 : f 2 + f (1/2) = 1 := sorry
theorem part1_2 : f 3 + f (1/3) = 1 := sorry
theorem part1_3 : f 4 + f (1/4) = 1 := sorry

theorem general : ∀ n : ℕ, f n + f (1 / n) = 1 := sorry

theorem final_value : (finset.range 2017).sum (λ n, f (n + 1)) - (finset.range 2017).sum (λ n, f (1 / (n + 2))) = 4033 / 2 := sorry

end part1_1_part1_2_part1_3_general_final_value_l543_543105


namespace arithmetic_sequence_a12_l543_543451

theorem arithmetic_sequence_a12 :
  ∃ (a : ℕ → ℚ) (d a₁: ℚ),
  (a 7 + a 9 = 16) ∧
  (a 4 = 1) ∧
  (a n = a₁ + (n - 1) * d) →
  a 12 = 15 := 
begin
  sorry
end

end arithmetic_sequence_a12_l543_543451


namespace simplify_and_sum_exponents_l543_543577

theorem simplify_and_sum_exponents 
    (a b c : ℝ) : 
    let simplified_expr := 2 * a * b^2 * c^4 * (48 * a^2 * b^2 * c^2)^(1/3)
    in 1 + 2 + 4 = 7 :=
by
  sorry

end simplify_and_sum_exponents_l543_543577


namespace min_value_a_plus_2b_minus_3c_l543_543529

theorem min_value_a_plus_2b_minus_3c
  (a b c : ℝ)
  (h : ∀ (x y : ℝ), x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  ∃ m : ℝ, m = a + 2 * b - 3 * c ∧ m = -4 :=
by
  sorry

end min_value_a_plus_2b_minus_3c_l543_543529


namespace clock_angle_at_4_20_l543_543697

/--
At 4:20 PM, the angle between the hour and minute hands on a clock is 10°.
-/
theorem clock_angle_at_4_20
  (hour_hand_rate : ℕ → ℕ → ℝ) -- rate of hour hand movement (degrees per hour)
  (minute_hand_rate : ℕ → ℕ → ℝ) -- rate of minute hand movement (degrees per minute)
  (time : ℕ × ℕ) -- time in hours and minutes
  (h_hour_rate : ∀ h, hour_hand_rate h 60 = 30 * h) -- hour hand rotates 30° per hour
  (h_minute_rate : ∀ m, minute_hand_rate m 60 = 6 * m) -- minute hand rotates 6° per minute
  (h_time : time = (4, 20)) :
  let hour_hand_pos := hour_hand_rate 4 60 + hour_hand_rate 1 60 * (20 / 60),
      minute_hand_pos := minute_hand_rate 20 60
  in abs (hour_hand_pos - minute_hand_pos) = 10 := by 
  -- The proof is omitted as requested.
  sorry

end clock_angle_at_4_20_l543_543697


namespace fractions_product_simplified_l543_543774

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end fractions_product_simplified_l543_543774


namespace original_average_l543_543592

theorem original_average (A : ℝ)
  (h : 2 * A = 160) : A = 80 :=
by sorry

end original_average_l543_543592


namespace minimum_working_buttons_l543_543986

/--
Given the set of working buttons on a calculator {0, 1, 3, 4, 5}, 
prove that any natural number from 1 to 99,999,999 can either 
be entered using only these buttons or as the sum of two natural 
numbers, each of which can be entered using only these buttons.
The smallest number of such working buttons is 5.
-/
theorem minimum_working_buttons (n : ℕ) (working_buttons : Finset ℕ) (h : working_buttons = \{0, 1, 3, 4, 5\}) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 99999999 →
    (∃ a b : ℕ, a ∈ working_buttons ∧ b ∈ working_buttons ∧ k = a + b) ∨
    (∃ d ∈ working_buttons, k = d) :=
by
  sorry

end minimum_working_buttons_l543_543986


namespace number_of_candy_bars_per_box_l543_543516

noncomputable def candy_bars_per_box (x : ℕ) : Prop :=
  let profit_per_bar := 1.50 - 1.00
  let total_bars := 5 * x
  let total_profit := profit_per_bar * total_bars
  total_profit = 25.00

theorem number_of_candy_bars_per_box (x : ℕ) : candy_bars_per_box x → x = 10 :=
by
  sorry

end number_of_candy_bars_per_box_l543_543516


namespace point_in_third_quadrant_l543_543070

noncomputable def is_second_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b > 0

noncomputable def is_third_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b < 0

theorem point_in_third_quadrant (a b : ℝ) (h : is_second_quadrant a b) : is_third_quadrant a (-b) :=
by
  sorry

end point_in_third_quadrant_l543_543070


namespace statement_c_correct_l543_543366

theorem statement_c_correct (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
by sorry

end statement_c_correct_l543_543366


namespace largest_number_among_options_l543_543333

theorem largest_number_among_options :
  let A := 0.983
  let B := 0.9829
  let C := 0.9831
  let D := 0.972
  let E := 0.9819
  C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end largest_number_among_options_l543_543333


namespace find_pq_l543_543942

theorem find_pq (p q : ℚ) :
  (p - q = -2) ∧ (p^2 - q^2 = -15) → (p = 11/4 ∧ q = 19/4) :=
by
  intro h
  cases h with h1 h2
  sorry

end find_pq_l543_543942


namespace cos_double_angle_l543_543396

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l543_543396
