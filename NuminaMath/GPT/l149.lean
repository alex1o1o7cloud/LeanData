import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.ContinuedFractions.Basic
import Mathlib.Algebra.Cubic
import Mathlib.Algebra.CubicDiscriminant
import Mathlib.Algebra.Function
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.ordered_ring
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Fintype.Probability
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.PrimeFactors
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Rel
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.NumberTheory.Powers
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Trigonometric
import real

namespace parabola_line_equation_l149_149854

noncomputable def ParabolaEquation : Prop :=
  ∃ k : ℝ, ( ∀ x y : ℝ, y^2 = 2*x ↔ y = k * (x - 1/2)) ∧ (k = sqrt 3 ∨ k = -sqrt 3)

theorem parabola_line_equation :
  ∀ (F : ℝ × ℝ) (A B : ℝ × ℝ), 
    F = (1/2, 0) →
    (A.2 = sqrt 3 * (A.1 - 1/2) ∨ A.2 = -sqrt 3 * (A.1 - 1/2)) ∧ 
    (B.2 = sqrt 3 * (B.1 - 1/2) ∨ B.2 = -sqrt 3 * (B.1 - 1/2)) →
    abs (fst A - fst F) = 3 * abs (fst B - fst F) →
    ParabolaEquation :=
begin
  intros,
  sorry
end

end parabola_line_equation_l149_149854


namespace average_sleep_hours_l149_149110

theorem average_sleep_hours (h_monday: ℕ) (h_tuesday: ℕ) (h_wednesday: ℕ) (h_thursday: ℕ) (h_friday: ℕ)
  (h_monday_eq: h_monday = 8) (h_tuesday_eq: h_tuesday = 7) (h_wednesday_eq: h_wednesday = 8)
  (h_thursday_eq: h_thursday = 10) (h_friday_eq: h_friday = 7) :
  (h_monday + h_tuesday + h_wednesday + h_thursday + h_friday) / 5 = 8 :=
by
  sorry

end average_sleep_hours_l149_149110


namespace cindy_total_travel_time_l149_149382

def speed_run := 3 -- Cindy's running speed in miles per hour
def speed_walk := 1 -- Cindy's walking speed in miles per hour
def distance_run := 0.5 -- Distance Cindy runs in miles
def distance_walk := 0.5 -- Distance Cindy walks in miles
def time_run := distance_run / speed_run * 60 -- Time to run half a mile in minutes
def time_walk := distance_walk / speed_walk * 60 -- Time to walk half a mile in minutes

theorem cindy_total_travel_time : time_run + time_walk = 40 := by
  -- skipping proof
  sorry

end cindy_total_travel_time_l149_149382


namespace p_sufficient_not_necessary_for_q_l149_149042

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l149_149042


namespace odd_function_zero_at_origin_zero_at_origin_not_implying_odd_l149_149772

def is_odd_function_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f (-x) = -f (x)

theorem odd_function_zero_at_origin {f : ℝ → ℝ} :
  (is_odd_function_on f (-1) 1) → f 0 = 0 :=
by
  intros h
  have : f 0 = -f 0 := h 0 (by linarith)
  sorry

theorem zero_at_origin_not_implying_odd {f : ℝ → ℝ} :
  (f 0 = 0) → ¬ (∀ x, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f (x)) :=
by
  intros h
  use (λ x, abs x)
  sorry

end odd_function_zero_at_origin_zero_at_origin_not_implying_odd_l149_149772


namespace f_at_two_f_decreasing_solve_inequality_l149_149431

noncomputable def f : ℝ → ℝ := sorry -- placeholder for f

def odd_function (f : ℝ → ℝ) (a b : ℝ) := ∀ x ∈ Icc a b, f (-x) = -f x

axiom f_odd : odd_function f (-3) 3
axiom f_add : ∀ x y, f (x + y) = f x + f y
axiom f_neg : ∀ x > 0, f x < 0
axiom f_at_one : f 1 = -2

theorem f_at_two : f 2 = -4 :=
by sorry

theorem f_decreasing : ∀ x1 x2, -3 ≤ x1 → x1 < x2 → x2 ≤ 3 → f x2 < f x1 :=
by sorry

theorem solve_inequality : {x : ℝ | f (x - 1) > 4} = Ico (-2 : ℝ) (-1) :=
by sorry

end f_at_two_f_decreasing_solve_inequality_l149_149431


namespace sin_double_angle_l149_149807

theorem sin_double_angle (α : ℝ) (h : sin α - cos α = real.sqrt 2) : sin (2 * α) = -1 :=
by 
  sorry

end sin_double_angle_l149_149807


namespace steve_take_home_pay_l149_149615

def annual_salary : ℕ := 40000
def tax_percentage : ℝ := 0.20
def healthcare_percentage : ℝ := 0.10
def union_dues : ℕ := 800

theorem steve_take_home_pay :
  annual_salary - (annual_salary * tax_percentage).to_nat - (annual_salary * healthcare_percentage).to_nat - union_dues = 27200 :=
by
  sorry

end steve_take_home_pay_l149_149615


namespace sin_of_cos_theta_is_negative_l149_149120

theorem sin_of_cos_theta_is_negative {θ : ℝ} (hθ : θ > π / 2 ∧ θ < π) : sin (cos θ) < 0 :=
by {
  have h1 : cos θ < 0,
  { exact Real.cos_neg_of_neg θ (by norm_num : 0 < π),
    norm_num at hθ,
    linarith, },
  exact sorry
}

end sin_of_cos_theta_is_negative_l149_149120


namespace solve_fun_problem_l149_149453

variable (f : ℝ → ℝ)

-- Definitions of the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_monotonic_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- The main theorem
theorem solve_fun_problem (h_even : is_even f) (h_monotonic : is_monotonic_on_pos f) :
  {x : ℝ | f (x + 1) = f (2 * x)} = {1, -1 / 3} := 
sorry

end solve_fun_problem_l149_149453


namespace rectangle_area_l149_149511

   variable (ABCD : Type) [Rectangle ABCD]
   variable (s : ℝ) -- side length of the shaded square
   variable (larger_side_length : ℝ) -- side length of the larger square
   variable (smaller_area : ℝ = 4) -- area of each of the two smaller squares
   variable (larger_area : ℝ = 16) -- area of the larger square

   -- Define the conditions
   def side_length_squared_equals_area : Prop := s^2 = 4
   def larger_square_side_twice_smaller : Prop := larger_side_length = 2 * s
   def all_squares_fit_exactly :
     Prop := larger_side_length^2 + 2 * smaller_area = 24
  
   -- Proof statement
   theorem rectangle_area : side_length_squared_equals_area ∧ 
                            larger_square_side_twice_smaller ∧ 
                            all_squares_fit_exactly →
                            (24 = smaller_area + smaller_area + larger_area) :=
   sorry
   
end rectangle_area_l149_149511


namespace two_diamond_five_eq_l149_149438

variable {α : Type*} [Nontrivial α] [LinearOrderedField α]

def diamond (a b : α) : α := a / b^2

theorem two_diamond_five_eq :
  (∀ a b : α, a > 0 ∧ b > 0 → diamond a b = diamond b a) ∧ 
  (∀ a : α, a > 0 → diamond (diamond a 1) a = diamond a 1) ∧ 
  (diamond 1 1 = 1) →
  diamond 2 5 = 2 / 25 :=
by
  sorry

end two_diamond_five_eq_l149_149438


namespace minimum_value_of_a_plus_b_l149_149494
noncomputable def minimum_value (a b: ℝ) : ℝ := a + b

theorem minimum_value_of_a_plus_b (a b: ℝ) (h : log 4 (3 * a + 4 * b) = log 2 (sqrt (a * b))) :
  minimum_value a b = 7 + 4 * sqrt 3 :=
sorry

end minimum_value_of_a_plus_b_l149_149494


namespace tom_steps_l149_149199

theorem tom_steps (matt_rate : ℕ) (tom_extra_rate : ℕ) (matt_steps : ℕ) (tom_rate : ℕ := matt_rate + tom_extra_rate) (time : ℕ := matt_steps / matt_rate)
(H_matt_rate : matt_rate = 20)
(H_tom_extra_rate : tom_extra_rate = 5)
(H_matt_steps : matt_steps = 220) :
  tom_rate * time = 275 :=
by
  -- We start the proof here, but leave it as sorry.
  sorry

end tom_steps_l149_149199


namespace reflection_matrix_over_vector_l149_149788

/-- 
The matrix A corresponds to the reflection over the vector (4, 1). 
The problem is to prove that reflecting any vector over (4, 1) results in matrix A.
-/
theorem reflection_matrix_over_vector : 
  let u := (⟨4, 1⟩ : ℝ × ℝ)
  let A := (λ (v : ℝ × ℝ), (⟨ (15/17) * v.1 + (8/17) * v.2, 
                              (8/17) * v.1 - (15/17) * v.2 ⟩ : ℝ × ℝ)) in 
  ∀ (v : ℝ × ℝ), 
  (let p := ((v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)) in
   let r := (2 * p * u.1 - v.1, 2 * p * u.2 - v.2) in
   r = A v) := sorry

end reflection_matrix_over_vector_l149_149788


namespace log2_integer_probability_l149_149355

theorem log2_integer_probability :
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999},
      powers_of_two := {n : ℕ | ∃ k : ℕ, n = 2^k},
      valid_numbers := three_digit_numbers ∩ powers_of_two
  in
  (∀ n, n ∈ three_digit_numbers → 0 < n)
  ∧ (∃ p, card valid_numbers = p)
  → (finset.card valid_numbers.to_finset * 300 = finset.card three_digit_numbers.to_finset) := 
sorry

end log2_integer_probability_l149_149355


namespace sequence_contains_at_most_one_square_l149_149601

theorem sequence_contains_at_most_one_square 
  (a : ℕ → ℕ) 
  (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) : 
  ∀ m n, (m ≠ n) → ¬ (∃ k, a m = k^2 ∧ a n = k^2) :=
sorry

end sequence_contains_at_most_one_square_l149_149601


namespace right_triangle_AB_and_perimeter_l149_149885

theorem right_triangle_AB_and_perimeter (A B C : Type*) [metric_space A] [metric_space B] [metric_space C]
  (h1 : ∀ (a b c : A), ∃! (A B C : Type*), ∠A = 90 ∧ tan (A, C) = 3 ∧ B C = 90) :
  AB = 27*sqrt 10 ∧ perimeter ABC = 36*sqrt 10 + 90 :=
by
  sorry

end right_triangle_AB_and_perimeter_l149_149885


namespace betty_started_with_36_brownies_l149_149748

def betty_sweet_tooth : Prop :=
  ∃ (cookies_start brownies_start B : ℕ),
    (B = 3) ∧
    (brownies_start = cookies_start) ∧
    (brownies_start -= 7 * B) ∧
    (brownies_start = 36)

theorem betty_started_with_36_brownies : betty_sweet_tooth → brownies_start = 36 :=
by {
  sorry
}

end betty_started_with_36_brownies_l149_149748


namespace num_pairs_mod_eq_l149_149919

theorem num_pairs_mod_eq (k : ℕ) (h : k ≥ 7) :
  ∃ n : ℕ, n = 2^(k+5) ∧
  (∀ x y : ℕ, 0 ≤ x ∧ x < 2^k ∧ 0 ≤ y ∧ y < 2^k → (73^(73^x) ≡ 9^(9^y) [MOD 2^k]) → true) :=
sorry

end num_pairs_mod_eq_l149_149919


namespace find_natural_numbers_satisfying_prime_square_l149_149776

-- Define conditions as a Lean statement
theorem find_natural_numbers_satisfying_prime_square (n : ℕ) (h : ∃ p : ℕ, Prime p ∧ (2 * n^2 + 3 * n - 35 = p^2)) :
  n = 4 ∨ n = 12 :=
sorry

end find_natural_numbers_satisfying_prime_square_l149_149776


namespace rotated_angle_540_deg_l149_149266

theorem rotated_angle_540_deg (θ : ℝ) (h : θ = 60) : 
  (θ - 540) % 360 % 180 = 60 :=
by
  sorry

end rotated_angle_540_deg_l149_149266


namespace area_square_II_l149_149979

theorem area_square_II (a b : ℝ) :
  let diag_I := 2 * (a + b)
  let area_I := (a + b) * (a + b) * 2
  let area_II := area_I * 3
  area_II = 6 * (a + b) ^ 2 :=
by
  sorry

end area_square_II_l149_149979


namespace logarithm_function_decreasing_l149_149988

theorem logarithm_function_decreasing (a : ℝ) : 
  (∀ x ∈ Set.Ici (-1), (3 * x^2 - a * x + 5) ≤ (3 * x^2 - a * (x + 1) + 5)) ↔ (-8 < a ∧ a ≤ -6) :=
by
  sorry

end logarithm_function_decreasing_l149_149988


namespace remaining_homes_proof_l149_149545

-- Define the total number of homes
def total_homes : ℕ := 200

-- Distributed homes after the first hour
def homes_distributed_first_hour : ℕ := (2 * total_homes) / 5

-- Remaining homes after the first hour
def remaining_homes_first_hour : ℕ := total_homes - homes_distributed_first_hour

-- Distributed homes in the next 2 hours
def homes_distributed_next_two_hours : ℕ := (60 * remaining_homes_first_hour) / 100

-- Remaining homes after the next 2 hours
def homes_remaining : ℕ := remaining_homes_first_hour - homes_distributed_next_two_hours

theorem remaining_homes_proof : homes_remaining = 48 := by
  sorry

end remaining_homes_proof_l149_149545


namespace apples_final_count_l149_149596

theorem apples_final_count :
  let initial_apples := 200
  let shared_apples := 5
  let remaining_after_share := initial_apples - shared_apples
  let sister_takes := remaining_after_share / 2
  let half_rounded_down := 97 -- explicitly rounding down since 195 cannot be split exactly
  let remaining_after_sister := remaining_after_share - half_rounded_down
  let received_gift := 7
  let final_count := remaining_after_sister + received_gift
  final_count = 105 :=
by
  sorry

end apples_final_count_l149_149596


namespace example1_example2_example3_example4_l149_149267

section PerfectNumber

-- 1. Prove that 29 == 5^2 + 2^2
theorem example1 : ∃ (a b : ℤ), 29 = a^2 + b^2 :=
by { use [5, 2], sorry }

-- 2. Prove that x^2 - 6x + 5 can be rewritten as (x - 3)^2 - 4 and mn = -12 for m = 3 and n = -4
theorem example2 : ∃ m n : ℤ, (∃ x : ℝ, x^2 - 6 * x + 5 = (x - m)^2 + n) ∧ m * n = -12 :=
by { use [3, -4], sorry }

-- 3. Prove that x + y = -1 given x^2 + y^2 - 2x + 4y + 5 = 0
theorem example3 (x y : ℝ) (h : x^2 + y^2 - 2 * x + 4 * y + 5 = 0) : x + y = -1 :=
by { sorry }

-- 4. Prove that S = x^2 + 4y^2 + 4x - 12y + 13 is a "perfect number"
theorem example4 (x y : ℤ) : ∃ k, k = 13 ∧ (x + 2)^2 + (2 * y - 3)^2 = x^2 + 4 * y^2 + 4 * x - 12 * y + k :=
by { use 13, sorry }

end PerfectNumber

end example1_example2_example3_example4_l149_149267


namespace p_implies_q_and_not_q_implies_p_l149_149055

variable {x : ℝ}

def p : Prop := 0 < x ∧ x < 2
def q : Prop := -1 < x ∧ x < 3

theorem p_implies_q_and_not_q_implies_p : (p → q) ∧ ¬ (q → p) := 
by
  sorry

end p_implies_q_and_not_q_implies_p_l149_149055


namespace fraction_halfway_between_one_fourth_and_one_sixth_l149_149781

theorem fraction_halfway_between_one_fourth_and_one_sixth :
  (1/4 + 1/6) / 2 = 5 / 24 :=
by
  sorry

end fraction_halfway_between_one_fourth_and_one_sixth_l149_149781


namespace p_implies_q_and_not_q_implies_p_l149_149058

variable {x : ℝ}

def p : Prop := 0 < x ∧ x < 2
def q : Prop := -1 < x ∧ x < 3

theorem p_implies_q_and_not_q_implies_p : (p → q) ∧ ¬ (q → p) := 
by
  sorry

end p_implies_q_and_not_q_implies_p_l149_149058


namespace find_a1_plus_a10_l149_149079

-- Define the geometric sequence and initial conditions
def geometric_sequence (a : ℕ → ℤ) (a1 : ℤ) (r : ℤ) := ∀ n, a n = a1 * r ^ (n - 1)

axiom condition1 (a : ℕ → ℤ) (a1 : ℤ) (r : ℤ) [geometric_sequence a a1 r] :
  a 4 + a 7 = 2

axiom condition2 (a : ℕ → ℤ) (a1 : ℤ) (r : ℤ) [geometric_sequence a a1 r] :
  a 2 * a 9 = -8

theorem find_a1_plus_a10 (a : ℕ → ℤ) (a1 : ℤ) (r : ℤ) [geometric_sequence a a1 r] 
  (h1 : condition1 a a1 r) (h2 : condition2 a a1 r) : 
  a1 + a 10 = -7 :=
sorry

end find_a1_plus_a10_l149_149079


namespace projection_calculation_l149_149794

def vector_proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let v_norm_sq := v.1 * v.1 + v.2 * v.2
  let scalar := dot_product / v_norm_sq
  (scalar * v.1, scalar * v.2)

theorem projection_calculation : 
  vector_proj (5, 3) (3, 1) = (5.4, 1.8) :=
by
  sorry

end projection_calculation_l149_149794


namespace cadence_total_earnings_l149_149374

noncomputable def total_earnings (years_old : ℝ) (monthly_salary_old : ℝ) (bonus_fraction_old : ℝ)
                                 (years_new : ℝ) (raise_fraction_new : ℝ) (bonus_fraction_new : ℝ)
                                 (deduction_year : ℝ) (deduction_fraction : ℝ) : ℝ :=
  let months_old := years_old * 12
  let total_salary_old := months_old * monthly_salary_old
  let yearly_bonus_old := bonus_fraction_old * monthly_salary_old
  let total_bonus_old := (years_old.floor * yearly_bonus_old) + ((years_old - years_old.floor) * yearly_bonus_old)
  let total_earnings_old := total_salary_old + total_bonus_old

  let new_monthly_salary := monthly_salary_old * (1 + raise_fraction_new)
  let months_new := years_new * 12
  let total_salary_new := months_new * new_monthly_salary
  let yearly_bonus_new := bonus_fraction_new * new_monthly_salary
  let total_bonus_new := years_new * yearly_bonus_new
  let annual_salary_new := new_monthly_salary * 12
  let deduction := if deduction_year ≤ years_new then deduction_fraction * annual_salary_new else 0
  let total_earnings_new := total_salary_new + total_bonus_new - deduction

  total_earnings_old + total_earnings_new

theorem cadence_total_earnings :
  total_earnings 3.5 5000 0.5 4 0.2 1 3 0.02 = 529310 := by
  sorry


end cadence_total_earnings_l149_149374


namespace scientific_notation_of_87000000_l149_149521

theorem scientific_notation_of_87000000 :
  87000000 = 8.7 * 10^7 := 
sorry

end scientific_notation_of_87000000_l149_149521


namespace quadrilateral_ratio_l149_149917

/-- Given a convex quadrilateral ABCD with an area of 2014, points P on AB and Q on AD such that the
    areas of triangles ABQ and ADP are both 1, and R being the intersection of AC and PQ,
    the ratio RC/RA is 2013. -/
theorem quadrilateral_ratio (ABCD : Quadrilateral) (P Q R : Point) (AC PQ : Line)
  (h_convex : is_convex ABCD)
  (h_area_ABCD : area ABCD = 2014)
  (h_point_P : on_segment P (side1 ABCD))
  (h_point_Q : on_segment Q (side4 ABCD))
  (h_area_ABQ : area (triangle A B Q) = 1)
  (h_area_ADP : area (triangle A D P) = 1)
  (h_intersection_R : intersect (line_through A C) (line_through P Q) = R) :
  (distance R C) / (distance R A) = 2013 :=
sorry

end quadrilateral_ratio_l149_149917


namespace den_passes_through_midpoint_BC_l149_149754

open Real EuclideanGeometry

variables (ω : Circle) (A B C D E F M N T : Point)
variables (hBC : Chord ω B C) (hDE : Chord ω D E) (hMeet : intersectionPoint hBC hDE = A)
variables (hParallel : Parallel (LineThrough D) (LineThrough B C) F)
variables (hFA : LineThrough F A) (hMeetFA : CircleIntersectionPoints ω (LineThrough F A) T)
variables (hM : IntersectionPoint (LineThrough E T) (LineThrough B C) = M)
variables (hN : Reflection A M = N)
variables (midBC : Midpoint B C)

theorem den_passes_through_midpoint_BC : PassesThrough (CircleThrough D E N) midBC :=
sorry

end den_passes_through_midpoint_BC_l149_149754


namespace concyclic_points_l149_149916

-- Definitions based on the given conditions
variables {A B C A' H O P Q K L : Type*}

-- Assumptions (conditions)
variables (triangle_inscribed : ∀ {X Y Z}, O X Y Z)
variables (orthocenter_H : is_orthocenter H A B C O)
variables (line_d : ∀ {R}, passes_through R H P Q)
variables (diameter_AA' : is_diameter A A' O)
variables (intersection_K : ∀ {M}, intersects M A' P K BC)
variables (intersection_L : ∀ {N}, intersects N A' Q L BC)

-- Prove that O, K, L, and A' are concyclic
theorem concyclic_points : concyclic O K L A' :=
sorry

end concyclic_points_l149_149916


namespace p_sufficient_but_not_necessary_q_l149_149062

theorem p_sufficient_but_not_necessary_q :
  ∀ x : ℝ, (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros x hx
  cases hx with h1 h2
  apply And.intro
  apply lt_of_lt_of_le h1
  linarith
  apply h2

end p_sufficient_but_not_necessary_q_l149_149062


namespace options_AD_correct_l149_149065

noncomputable def f : ℝ → ℝ := sorry

/-- Continuity of f on ℝ -/
axiom f_continuous : continuous f

/-- f is an even function -/
axiom f_even : ∀ x : ℝ, f (-x) = f x

/-- f is monotonically increasing on (0, +∞) -/
axiom f_increasing : ∀ (x₁ x₂ : ℝ), 0 < x₁ → 0 < x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0

/-- f(-1) = 0 -/
axiom f_at_neg_one : f (-1) = 0

theorem options_AD_correct :
  (∀ x : ℝ, x < 0 → f (x) > f (-x)) ∧ 
  (∀ x : ℝ, f x / x > 0 → x ∈ (Set.Ioo (-1) 0) ∪ (Set.Ioi 1)) :=
sorry

end options_AD_correct_l149_149065


namespace minimum_n_is_10_l149_149509

theorem minimum_n_is_10 (n : ℕ) (h₁ : n ≥ 4)
  (h₂ : ∀ (S : Finset ℕ), S.card = n - 1 → 
    ∃ (f : ℕ → ℕ), (∀ x ∈ S, f x ∈ S) ∧ 
                     (∀ x ∈ S, (∃ y ∈ S, f x = y) ∧ (∃ z ∈ S, f z = x) ∧ f (f x) ≠ x))
  (h₃ : ¬∃ (f : ℕ → ℕ), (∀ x ∈ Finset.range n, f x ∈ Finset.range n) ∧ 
                         (∀ x ∈ Finset.range n, (∃ y ∈ Finset.range n, f x = y) ∧ 
                                                (∃ z ∈ Finset.range n, f z = x) ∧
                                                f (f x) ≠ x)) : 
  n = 10 :=
sorry

end minimum_n_is_10_l149_149509


namespace complex_conjugate_square_l149_149707

theorem complex_conjugate_square (a b : ℝ) (i : ℂ) (h : a - i = conj (2 + b * i)) : (a + b * i)^2 = 3 + 4 * i :=
by
  -- Given conditions from the problem.
  have ha : a = 2, from sorry,
  have hb : b = 1, from sorry,
  -- The final goal to prove.
  calc
    (a + b * i)^2 = (2 + 1 * i)^2 : by rw [ha, hb]
             ... = (2 + i)^2 : by simp
             ... = 3 + 4 * i : by norm_num

end complex_conjugate_square_l149_149707


namespace fn_difference_l149_149486

noncomputable def f(n : ℕ) : ℚ :=
  (Finset.range (n + 1)).sum (λ k => 1 / (n + 1 + k : ℚ))

theorem fn_difference (n : ℕ) :
  f(n + 1) - f(n) = (1 / (2 * n + 1 : ℚ)) + (1 / (2 * n + 2 : ℚ)) - (1 / (n + 1 : ℚ)) :=
by
  sorry

end fn_difference_l149_149486


namespace needle_cannot_swap_endpoints_l149_149351

open Complex

noncomputable def omega : ℂ := exp (π * I / 4)

def needle_endpoints_swapped (n : ℕ) (endpoints : ℕ → ℂ) : Prop :=
 endpoints n = (λ z : ℕ → ℂ, z (n - 1)) ∧ 
 endpoints (n - 1) = (λ z : ℕ → ℂ, z n)

theorem needle_cannot_swap_endpoints :
  ∀ (n : ℕ) (endpoints : ℕ → ℂ), (odd n) → ¬ needle_endpoints_swapped n endpoints :=
by
intro n endpoints hn odd
sorry

end needle_cannot_swap_endpoints_l149_149351


namespace remy_gallons_used_l149_149223

def roman_usage (R : ℕ) : Prop := R + (3 * R + 1) = 33

def remy_usage (R : ℕ) (Remy : ℕ) : Prop := Remy = 3 * R + 1

theorem remy_gallons_used :
  ∃ R Remy : ℕ, roman_usage R ∧ remy_usage R Remy ∧ Remy = 25 :=
  by
    sorry

end remy_gallons_used_l149_149223


namespace constant_term_expansion_eq_40_l149_149251

theorem constant_term_expansion_eq_40 : 
  constant_term_of_bin_expansion (x^{2} - 2 / x^{3})^{5} = 40 := 
by sorry

end constant_term_expansion_eq_40_l149_149251


namespace strands_of_duct_tape_used_l149_149487

-- Define the conditions
def hannah_cut_rate : ℕ := 8  -- Hannah's cutting rate
def son_cut_rate : ℕ := 3     -- Son's cutting rate
def minutes : ℕ := 2          -- Time taken to free the younger son

-- Define the total cutting rate
def total_cut_rate : ℕ := hannah_cut_rate + son_cut_rate

-- Define the total number of strands
def total_strands : ℕ := total_cut_rate * minutes

-- State the theorem to prove
theorem strands_of_duct_tape_used : total_strands = 22 :=
by
  sorry

end strands_of_duct_tape_used_l149_149487


namespace proof_problem_l149_149134

-- Define the basic entities and their relationships
variables {a b c A B C : ℝ}

-- Define the given conditions
def condition1 : Prop := 2 * cos (A / 2) ^ 2 = (√3 / 3) * sin A
def condition2 : Prop := sin (B - C) = 4 * cos B * sin C

-- Define our main theorem statement
theorem proof_problem (h1 : condition1) (h2 : condition2) : b / c = 1 + √6 :=
sorry

end proof_problem_l149_149134


namespace infinite_solutions_of_linear_eq_l149_149105

theorem infinite_solutions_of_linear_eq (a b : ℝ) : 
  (∃ b : ℝ, ∃ a : ℝ, 5 * a - 11 * b = 21) := sorry

end infinite_solutions_of_linear_eq_l149_149105


namespace g_of_5_l149_149987

theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 5 = 402 / 70 := 
sorry

end g_of_5_l149_149987


namespace bmc_angle_l149_149164

noncomputable def angle_BMC (a : ℝ) : ℝ :=
  90

theorem bmc_angle (a : ℝ) (A B C D : ℝ × ℝ × ℝ)
  (h_tetrahedron : ∀ u v, u ∈ {A, B, C, D} → v ∈ {A, B, C, D} → u ≠ v → dist u v = a)
  (M H : ℝ × ℝ × ℝ)
  (h_AH : dist A H = (sqrt 6 / 3) * a)
  (h_midpoint : dist A M = dist M H) :
  ∠BMC = 90 :=
by
  sorry

end bmc_angle_l149_149164


namespace reflection_matrix_over_vector_l149_149785

theorem reflection_matrix_over_vector : 
  ∃ M : Matrix (Fin 2) (Fin 2) ℚ, 
    M.mul_vec ⟨[4, 1], by decide⟩ = ⟨[4, 1], by decide⟩ ∧
    M = !![ 15/17, 8/17; 8/17, -15/17 ]
    sorry

end reflection_matrix_over_vector_l149_149785


namespace total_pokemon_cards_l149_149964

variable (n_people : ℕ) (cards_per_person : ℕ)  -- Define the variables for the number of people and cards per person.
variable h1 : n_people = 6  -- Condition: There are 6 people.
variable h2 : cards_per_person = 100  -- Condition: Each person has 100 Pokemon cards.

theorem total_pokemon_cards (n_people cards_per_person : ℕ) (h1 : n_people = 6) (h2 : cards_per_person = 100) : n_people * cards_per_person = 600 := by
  rw [h1, h2]
  exact Nat.mul_eq_mul_right (Nat.succ_ne_zero 5)

-- Add sorry to represent skipping the proof steps.

end total_pokemon_cards_l149_149964


namespace sum_of_arithmetic_progression_l149_149366

theorem sum_of_arithmetic_progression :
  let a := 30
  let d := -3
  let n := 20
  let S_n := n / 2 * (2 * a + (n - 1) * d)
  S_n = 30 :=
by
  sorry

end sum_of_arithmetic_progression_l149_149366


namespace eval_expression_at_1999_l149_149813

theorem eval_expression_at_1999 : 
  let x := 1999 in
  (|4 * x ^ 2 - 5 * x + 1| - 4 * |x ^ 2 + 2 * x + 2| + 3 * x + 7) = -19990 := by
  sorry

end eval_expression_at_1999_l149_149813


namespace correct_statements_l149_149898

-- Define hypotheses in Lean to represent the given conditions and statements.
variable (polygon : Type) (pyramid : Type) (triangle : Type) (trapezoid : Type)
variable (translate : polygon → Type) (cut : pyramid → (pyramid × frustum))
variable (rotate_triangle : triangle → cone) (rotate_trapezoid : trapezoid → truncated_cone)

-- Definitions for geometric transformations
def is_prism (body : Type) : Prop := sorry
def is_frustum (body : Type) : Prop := sorry
def is_cone (body : Type) : Prop := sorry
def is_truncated_cone (body : Type) : Prop := sorry

-- Statement 1: Translating a polygon forms a prism
axiom stmt1 : is_prism (translate polygon)

-- Statement 2: Cutting a pyramid results in a frustum and another pyramid
axiom stmt2 : ∀ (p: pyramid), ¬ is_frustum (cut p).2

-- Statement 3: Rotating a triangle forms a cone
axiom stmt3 : ∀ (t: triangle), ¬ is_cone (rotate_triangle t)

-- Statement 4: Rotating a trapezoid forms a truncated cone
axiom stmt4 : is_truncated_cone (rotate_trapezoid trapezoid)

-- Proof problem: Prove that statements 1 and 4 are correct (1 ∧ 4), and statements 2 and 3 are incorrect (¬2 ∧ ¬3).
theorem correct_statements : stmt1 ∧ stmt4 ∧ ¬stmt2 ∧ ¬stmt3 := sorry

end correct_statements_l149_149898


namespace cylindrical_to_cartesian_l149_149452

theorem cylindrical_to_cartesian :
  ∀ (r θ z : ℝ), r = 2 → θ = π / 3 → z = 2 → 
  (r * Real.cos θ, r * Real.sin θ, z) = (1, Real.sqrt 3, 2) :=
by
  intros r θ z hr hθ hz
  sorry

end cylindrical_to_cartesian_l149_149452


namespace sum_closest_integer_series_l149_149798

noncomputable def closest_integer (n : ℕ) : ℤ := 
if n < 1 then 0 else 
  let k := Int.ofNat (Int.natAbs (Int.ofNat n).sqrt) in 
  if (Int.ofNat n - k * k) < (k + 1) * (k + 1) - Int.ofNat n 
  then k else k + 1

theorem sum_closest_integer_series : 
  ∑' n : ℕ, (if n = 0 then 0 else (2 ^ closest_integer n + 2 ^ -closest_integer n) / 2 ^ n) = 3 :=
sorry

end sum_closest_integer_series_l149_149798


namespace ninety_nine_fives_not_perfect_square_l149_149507

-- Define the property of the given number "n"
def ninety_nine_fives_one_different_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), 0 ≤ d < 10 ∧
  ((d ≠ 5) ∧ (99.fives.push(d)) = n)

-- Theorem
theorem ninety_nine_fives_not_perfect_square (n : ℕ) :
  ninety_nine_fives_one_different_digit(n) → ¬ (∃ k : ℕ, k^2 = n) :=
by
  sorry

end ninety_nine_fives_not_perfect_square_l149_149507


namespace gdp_doubles_in_9_years_l149_149132

noncomputable def gdp_double_in_years (a : ℝ) (growth_rate : ℝ) (log2 : ℝ) (log1_08 : ℝ) : ℝ :=
  log2 / log1_08

theorem gdp_doubles_in_9_years (a : ℝ) (growth_rate : ℝ) (log2 log1_08 : ℝ)
  (h1 : growth_rate = 0.08)
  (h2 : log2 ≈ 0.3010)
  (h3 : log1_08 ≈ 0.0334) :
  8.5 < gdp_double_in_years a growth_rate log2 log1_08 ∧ gdp_double_in_years a growth_rate log2 log1_08 < 9.5 :=
by
  sorry

end gdp_doubles_in_9_years_l149_149132


namespace positive_t_value_l149_149027

theorem positive_t_value (t : ℝ) (ht : t > 0) : abs (complex.mk 8 (3 * t)) = 13 → t = real.sqrt (105) / 3 :=
by 
  sorry

end positive_t_value_l149_149027


namespace sequence_inequality_l149_149474

theorem sequence_inequality (a : ℕ → ℤ) (h₀ : a 1 > a 0) 
  (h₁ : ∀ n : ℕ, n ≥ 1 → a (n+1) = 3 * a n - 2 * a (n-1)) : 
  a 100 > 2^99 := 
sorry

end sequence_inequality_l149_149474


namespace bird_families_left_l149_149306

theorem bird_families_left (B_initial B_flew_away : ℕ) (h_initial : B_initial = 41) (h_flew_away : B_flew_away = 27) :
  B_initial - B_flew_away = 14 :=
by
  sorry

end bird_families_left_l149_149306


namespace remainder_of_sum_mod_eight_l149_149868

theorem remainder_of_sum_mod_eight (m : ℤ) : 
  ((10 - 3 * m) + (5 * m + 6)) % 8 = (2 * m) % 8 :=
by
  sorry

end remainder_of_sum_mod_eight_l149_149868


namespace term_37_l149_149617

section GeometricSequence

variable {a b : ℕ → ℝ}
variable (q p : ℝ)

-- Definition of geometric sequences
def is_geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n ≥ 1, a (n + 1) = r * a n

-- Given conditions
axiom a1_25 : a 1 = 25
axiom b1_4 : b 1 = 4
axiom a2b2_100 : a 2 * b 2 = 100

-- Assume a and b are geometric sequences
axiom a_geom_seq : is_geometric_seq a q
axiom b_geom_seq : is_geometric_seq b p

-- Main theorem to prove
theorem term_37 (n : ℕ) (hn : n = 37) : (a n * b n) = 100 :=
sorry

end GeometricSequence

end term_37_l149_149617


namespace p_sufficient_not_necessary_q_l149_149046

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l149_149046


namespace probability_no_adjacent_standing_l149_149975

-- Define the problem conditions in Lean 4.
def total_outcomes := 2^10
def favorable_outcomes := 123

-- The probability is given by favorable outcomes over total outcomes.
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now state the theorem regarding the probability.
theorem probability_no_adjacent_standing : 
  probability = 123 / 1024 :=
by {
  sorry
}

end probability_no_adjacent_standing_l149_149975


namespace illuminated_area_ratio_l149_149626

theorem illuminated_area_ratio (r x y l : ℝ)
  (h1 : y = r / 4)
  (h2 : x = (r / 4) * Real.sqrt 15)
  (h3 : l = r * Real.sqrt 15)
  (distance : dist := 3 * r) :
  let F := 2 * Real.pi * r * (r - y),
      P := Real.pi * x * l in
  F / P = 2 / 5 := by
  sorry

end illuminated_area_ratio_l149_149626


namespace pyramid_surface_area_l149_149734

-- Define the conditions given in the problem
def side_length_hexagon : ℝ := 8
def height_pyramid : ℝ := 15

-- We need to express that the total surface area of the pyramid is approximately 543.2 square centimeters
theorem pyramid_surface_area :
  ∀ (a h : ℝ), a = side_length_hexagon → h = height_pyramid → 
  ( -- Total surface area formula for a pyramid with a hexagonal base
    let slant_height := real.sqrt (h^2 + (a^2 * 3 / 4)) in
    let area_tri_face := 1/2 * a * slant_height in
    let total_tri_faces_area := 6 * area_tri_face in
    let hex_base_area := (3 * real.sqrt 3 / 2) * a^2 in
    total_tri_faces_area + hex_base_area ≈ 543.2) :=
begin
  intros a h ha hh,
  -- Definitions of intermediate terms
  let slant_height := real.sqrt (h^2 + (a^2 * 3 / 4)),
  let area_tri_face := 1/2 * a * slant_height,
  let total_tri_faces_area := 6 * area_tri_face,
  let hex_base_area := (3 * real.sqrt 3 / 2) * a^2,
  -- State the equation for total surface area
  have surface_area_eq : total_tri_faces_area + hex_base_area = 543.2,
  { sorry }, -- Proof is omitted
end

end pyramid_surface_area_l149_149734


namespace find_common_difference_l149_149927

variable {a : ℕ → ℤ} 
variable {S : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def problem_conditions (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 4 = 8 ∧ S 8 = 48

theorem find_common_difference :
  ∃ d, problem_conditions a S d ∧ is_arithmetic_sequence a d ∧ sum_of_first_n_terms a S ∧ d = 2 :=
by
  sorry

end find_common_difference_l149_149927


namespace inequality_preserves_neg_half_l149_149495

variable (a b : ℝ)

theorem inequality_preserves_neg_half (h : a ≤ b) : -a / 2 ≥ -b / 2 := by
  sorry

end inequality_preserves_neg_half_l149_149495


namespace weekly_expenses_l149_149913

-- Definitions
def weekly_rent : ℝ := 1200
def utilities_percentage : ℝ := 0.20
def cashier_hourly_wage : ℝ := 12.50
def shift_manager_hourly_wage : ℝ := 15.00
def hours_per_shift : ℝ := 8
def shifts_per_day : ℝ := 3
def days_per_week : ℝ := 5
def loan_principal : ℝ := 20000
def annual_interest_rate : ℝ := 0.04
def weeks_in_year : ℝ := 52
def total_revenue : ℝ := 8000
def sales_tax_rate : ℝ := 0.05

-- Calculations
def weekly_utilities := utilities_percentage * weekly_rent
def total_rent_and_utilities := weekly_rent + weekly_utilities

def hours_per_week := hours_per_shift * shifts_per_day * days_per_week
def weekly_cashier_wages := cashier_hourly_wage * hours_per_week
def weekly_shift_manager_wages := shift_manager_hourly_wage * hours_per_week
def total_weekly_wages := weekly_cashier_wages + weekly_shift_manager_wages

def weekly_interest := annual_interest_rate * loan_principal / weeks_in_year
def total_loan_repayment := loan_principal + weekly_interest * weeks_in_year
def weekly_loan_installment := total_loan_repayment / weeks_in_year

def weekly_sales_tax := sales_tax_rate * total_revenue

def total_weekly_expenses := total_rent_and_utilities + total_weekly_wages + weekly_loan_installment + weekly_sales_tax

-- Theorem
theorem weekly_expenses : total_weekly_expenses = 5540 := by
  sorry

end weekly_expenses_l149_149913


namespace notebooks_distribution_l149_149021

variables (k v y s se : ℕ)

theorem notebooks_distribution :
  k + v + y + s + se = 100 ∧
  k + v = 52 ∧
  v + y = 43 ∧
  y + s = 34 ∧
  s + se = 30 →
  k = 27 ∧ v = 25 ∧ y = 18 ∧ s = 16 ∧ se = 14 :=
by
  intro h
  cases h with _ h1
  cases h1 with _ h2
  cases h2 with _ h3
  cases h3 with _ h4
  cases h4 with _ h5
  exact sorry

end notebooks_distribution_l149_149021


namespace red_blue_boxes_equal_subsum_l149_149425

-- Define the conditions
def boxes_constraints (n_red n_blue : ℕ) (max_red max_blue : ℕ) := 
  {x : Fin n_red → ℕ // (∀ i, x i ≥ 1 ∧ x i ≤ max_red)} ∧
  {y : Fin n_blue → ℕ // (∀ i, y i ≥ 1 ∧ y i ≤ max_blue)} ∧
  (∑ i, (x.1 i) < ∑ i, (y.1 i))

-- Prove the final statement
theorem red_blue_boxes_equal_subsum (x y : Fin 19 → ℕ) (hx : ∀ i, 1 ≤ x i ∧ x i ≤ 200)
  (hy : ∀ j, 1 ≤ y j ∧ y j ≤ 19) (h_sum : ∑ i, x i < ∑ j, y j) :
  ∃ (sx : Finset (Fin 19)) (sy : Finset (Fin 200)), ∑ i in sx, x i = ∑ j in sy, y j :=
sorry

end red_blue_boxes_equal_subsum_l149_149425


namespace mixture_price_l149_149690

-- Define constants
noncomputable def V1 (X : ℝ) : ℝ := 3.50 * X
noncomputable def V2 : ℝ := 4.30 * 6.25
noncomputable def W2 : ℝ := 6.25
noncomputable def W1 (X : ℝ) : ℝ := X

-- Define the total mixture weight condition
theorem mixture_price (X : ℝ) (P : ℝ) (h1 : W1 X + W2 = 10) (h2 : 10 * P = V1 X + V2) :
  P = 4 := by
  sorry

end mixture_price_l149_149690


namespace time_per_sweater_l149_149570

variable (hours_hat : ℝ) (hours_scarf : ℝ) (hours_mittens : ℝ) (hours_socks : ℝ) (total_hours : ℝ) (sets : ℕ) (hours_sweater : ℝ)

-- Define the knitting times
def knitting_times := {
  hat := hours_hat,
  scarf := hours_scarf,
  mittens := hours_mittens,
  socks := hours_socks,
  total := total_hours,
  sets := sets,
  sweater := hours_sweater
}

-- Prove that it takes 6 hours to knit each sweater
theorem time_per_sweater (h_hat : knitting_times.hat = 2) 
  (h_scarf : knitting_times.scarf = 3) 
  (h_mittens : knitting_times.mittens = 1) 
  (h_socks : knitting_times.socks = 1.5) 
  (h_total : knitting_times.total = 48) 
  (h_sets : knitting_times.sets = 3) 
  (h_sweater : knitting_times.sweater = 6) 
  (h_other_items : 3 * (2 + 3 + 2 * 1 + 2 * 1.5) = 30) 
  (proof_eq : 48 - 30 = 18) :
  knitting_times.sweater = 6 :=
by 
  sorry

end time_per_sweater_l149_149570


namespace find_c_d_of_cubic_common_roots_l149_149792

theorem find_c_d_of_cubic_common_roots 
  (c d : ℝ)
  (h1 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + c * r ^ 2 + 12 * r + 7 = 0) ∧ (s ^ 3 + c * s ^ 2 + 12 * s + 7 = 0))
  (h2 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + d * r ^ 2 + 15 * r + 9 = 0) ∧ (s ^ 3 + d * s ^ 2 + 15 * s + 9 = 0)) :
  c = 5 ∧ d = 4 :=
sorry

end find_c_d_of_cubic_common_roots_l149_149792


namespace planes_perpendicular_l149_149969

-- Define the normal vectors to the planes
def n1 : ℝ × ℝ × ℝ := (2, 3, -4)
def n2 : ℝ × ℝ × ℝ := (5, -2, 1)

-- Define the dot product function for three-dimensional vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Theorem stating that the planes are perpendicular
theorem planes_perpendicular : dot_product n1 n2 = 0 := 
begin
  -- Simplify to show that the dot product is zero
  sorry
end

end planes_perpendicular_l149_149969


namespace square_root_of_2_pow_12_l149_149504

theorem square_root_of_2_pow_12 : √(2^12) = 64 :=
by
  sorry

end square_root_of_2_pow_12_l149_149504


namespace proof_problem_l149_149858

theorem proof_problem (x : ℝ) (h : 2^x + 2^(-x) = 5) : 4^x + 4^(-x) = 23 := 
sorry

end proof_problem_l149_149858


namespace unique_representation_of_natural_numbers_l149_149428

theorem unique_representation_of_natural_numbers
  (n k : ℕ) (hkn: k < n) : 
  ∀ x : ℕ, x < Nat.choose n k →
    ∃! (a : Fin k → ℕ), 
    (∀ i : Fin k, a i < n) ∧ (StrictMono a) ∧ 
    (x = ∑ i in Fin.range k, Nat.choose (a i) (i + 1)) := 
by 
  sorry

end unique_representation_of_natural_numbers_l149_149428


namespace f_properties_l149_149262

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

lemma f_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

lemma f_increasing : ∀ x : ℝ, f' x ≥ 0 := by
  sorry

theorem f_properties : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f' x ≥ 0) := by
  sorry

end f_properties_l149_149262


namespace solution_set_correct_l149_149019

noncomputable def solution_set (x : ℝ) : Prop :=
  5 / (x + 2) ≥ 1

theorem solution_set_correct :
  {x : ℝ | solution_set x} = set.Ioc (-2 : ℝ) 3 ∪ {3} :=
by
  sorry

end solution_set_correct_l149_149019


namespace discount_percentage_l149_149914

theorem discount_percentage (P : ℝ) (D : ℝ) (John_paid_more_than_Jane : ℝ) 
    (original_price : P = 84.00000000000009)
    (John_tip : P * 1.15)
    (Jane_tip : P * (1 - D / 100) * 1.15)
    (difference : John_tip - Jane_tip = 1.26) :
  D ≈ 1.33 :=
sorry

end discount_percentage_l149_149914


namespace james_birthday_stickers_l149_149170

theorem james_birthday_stickers :
  ∀ (initial_stickers birthday_stickers : ℕ), initial_stickers = 39 → birthday_stickers = 61 → (birthday_stickers - initial_stickers) = 22 :=
by
  intros initial_stickers birthday_stickers h_initial h_birthday,
  rw [h_initial, h_birthday],
  exact rfl

end james_birthday_stickers_l149_149170


namespace find_solutions_l149_149005

noncomputable def solutions := { x : ℝ | (∛(18 * x - 2) + ∛(16 * x + 2)) = 5 * ∛x }

theorem find_solutions :
  solutions = {0, (-2 + Real.sqrt 1048) / 261, (-2 - Real.sqrt 1048) / 261} :=
sorry

end find_solutions_l149_149005


namespace find_positive_real_satisfying_eq_l149_149415

theorem find_positive_real_satisfying_eq :
  ∃ x : ℝ, 0 < x ∧ 3 * real.sqrt (x^2 + x) + 3 * real.sqrt (x^2 - x) = 6 * real.sqrt 2 ∧
  x = 4 * real.sqrt 7 / 7 :=
sorry

end find_positive_real_satisfying_eq_l149_149415


namespace reciprocal_neg_one_div_2022_l149_149276

theorem reciprocal_neg_one_div_2022 : (1 / (-1 / 2022)) = -2022 :=
by sorry

end reciprocal_neg_one_div_2022_l149_149276


namespace constant_term_expansion_l149_149252

noncomputable def binomial_expansion (x : ℝ) : ℝ :=
  (x - (1 / x)) * (2 * x + (1 / x))^5

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  ∃ c : ℝ, binomial_expansion x = c ∧ c = -40 :=
by
  sorry

end constant_term_expansion_l149_149252


namespace num_pos_int_lt_zero_prod_l149_149013

theorem num_pos_int_lt_zero_prod :
  let P := ∏ i in finset.range 50, (2 * i + 1 - n)
  ∃ (N : ℕ), N = 25 ∧ ∀ (n : ℕ), 0 < n → n * P < 0 → n < 100 :=
  sorry

end num_pos_int_lt_zero_prod_l149_149013


namespace problem_1_problem_2_l149_149598

open_locale classical
noncomputable theory

-- Definitions for the first probability problem
def total_people : ℕ := 6
def total_boys : ℕ := 4
def total_girls : ℕ := 2
def select_people : ℕ := 3

def total_combinations : ℕ := nat.choose total_people select_people
def boy_combinations : ℕ := nat.choose total_boys select_people

def probability_all_boys : ℚ := boy_combinations / total_combinations

-- Definitions for the second probability problem
def girl_combinations : ℕ := nat.choose total_girls 1
def boy_combinations_2 : ℕ := nat.choose total_boys 2

def probability_one_girl : ℚ := girl_combinations * boy_combinations_2 / total_combinations

-- Statements to prove the equivalence
theorem problem_1 : probability_all_boys = 1/5 := by sorry

theorem problem_2 : probability_one_girl = 3/5 := by sorry

end problem_1_problem_2_l149_149598


namespace arithmetic_seq_sum_l149_149902

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d) (h_a5 : a 5 = 15) :
  a 3 + a 4 + a 6 + a 7 = 60 :=
sorry

end arithmetic_seq_sum_l149_149902


namespace average_salary_excluding_manager_l149_149153

variable {A : ℝ} -- Average salary of 20 employees

-- Conditions
def employees_num : ℝ := 20
def manager_salary : ℝ := 3700
def average_increase : ℝ := 100
def total_with_manager := 21 * (A + average_increase)
def total_without_manager := employees_num * A + manager_salary

-- Statement
theorem average_salary_excluding_manager :
  total_with_manager = total_without_manager → A = 1600 := by
  sorry

end average_salary_excluding_manager_l149_149153


namespace tan_simplification_l149_149119

theorem tan_simplification (x : ℝ) (h : tan x = 1 / 2) : 
  (3 * sin x ^ 2 - 2) / (sin x * cos x) = -7 / 2 := 
sorry

end tan_simplification_l149_149119


namespace smallest_prime_factor_2457_l149_149674

theorem smallest_prime_factor_2457 : ∃ p : ℕ, nat.prime p ∧ p ∣ 2457 ∧ ∀ q : ℕ, nat.prime q → q ∣ 2457 → q ≥ p :=
by {
  use 3,
  sorry
}

end smallest_prime_factor_2457_l149_149674


namespace least_positive_y_tan_l149_149934

theorem least_positive_y_tan (p q y m : ℝ) (h1 : Real.tan y = p / q) (h2 : Real.tan (3 * y) = q / (p + q))
    (h3 : y = Real.atan m) : m = 1 :=
sorry

end least_positive_y_tan_l149_149934


namespace steps_Tom_by_time_Matt_reaches_220_l149_149198

theorem steps_Tom_by_time_Matt_reaches_220 (rate_Matt rate_Tom : ℕ) (time_Matt_time_Tom : ℕ) (steps_Matt steps_Tom : ℕ) :
  rate_Matt = 20 →
  rate_Tom = rate_Matt + 5 →
  steps_Matt = 220 →
  time_Matt_time_Tom = steps_Matt / rate_Matt →
  steps_Tom = steps_Matt + time_Matt_time_Tom * 5 →
  steps_Tom = 275 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4, h2, h1] at h5
  norm_num at h5
  exact h5

end steps_Tom_by_time_Matt_reaches_220_l149_149198


namespace sufficient_condition_not_necessary_condition_l149_149825

variables {α : Type} [inner_product_space ℝ α]
variables (a l : α) (α_plane : set α)
variables (h_subset : a ∈ α_plane) (h_perp_plane : ∀ v ∈ α_plane, inner_product_space.real_inner l v = 0)

-- Sufficient condition: l ⊥ α_plane implies l ⊥ a
theorem sufficient_condition (h1 : ∀ v ∈ α_plane, inner_product_space.real_inner l v = 0) : inner_product_space.real_inner l a = 0 := 
by {
  sorry
}

-- Necessary condition: l ⊥ a does not necessarily imply l ⊥ α_plane
theorem not_necessary_condition (h_sub_a : inner_product_space.real_inner l a = 0) : ¬ (∀ v ∈ α_plane, inner_product_space.real_inner l v = 0) :=
by {
  sorry
}

end sufficient_condition_not_necessary_condition_l149_149825


namespace range_f_l149_149810

def f (x : ℝ) : ℝ := -x^2 + 4*x

theorem range_f (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : b ≤ 2) :
  set.range (λ x, f x) = set.Icc 0 4 :=
sorry

end range_f_l149_149810


namespace limit_seq_tends_to_neg_infinity_l149_149702

noncomputable def sequence (n : ℕ) : ℝ := 
  (Real.sqrt ((n^5 + 1) * (n^2 - 1)) - n * Real.sqrt (n * (n^4 + 1))) / n

theorem limit_seq_tends_to_neg_infinity :
  filter.tendsto sequence filter.at_top filter.at_bot :=
sorry

end limit_seq_tends_to_neg_infinity_l149_149702


namespace remainder_of_6_pow_2045_mod_11_l149_149673

theorem remainder_of_6_pow_2045_mod_11 : (6 ^ 2045) % 11 = 10 :=
by
  -- conditions from the problem
  have h1 : 6 ^ 10 % 11 = 1 := by sorry
  -- proof using the conditions and simplifying as per the solution steps
  have h2 : (6 ^ 2045) % 11 = ((6 ^ 10) ^ 204 * 6 ^ 5) % 11 := by sorry
  rw h1 at h2
  have h3 : ((1 ^ 204) * 6 ^ 5) % 11 = 6 ^ 5 % 11 := by sorry
  have h4 : 6 ^ 5 % 11 = 10 := by sorry
  rw h4 at h3
  exact h3

end remainder_of_6_pow_2045_mod_11_l149_149673


namespace percentage_decrease_of_b_l149_149639

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b)
  (h1 : a / b = 4 / 5)
  (h2 : x = a + 0.25 * a)
  (h3 : m = b * (1 - p / 100))
  (h4 : m / x = 0.4) :
  p = 60 :=
by
  sorry

end percentage_decrease_of_b_l149_149639


namespace center_of_circle_l149_149007

theorem center_of_circle (x y : ℝ) : x^2 - 8 * x + y^2 - 4 * y = 4 → (x, y) = (4, 2) :=
by
  sorry

end center_of_circle_l149_149007


namespace arrangement_count_l149_149745

/-- April has five different basil plants and five different tomato plants. --/
def basil_plants : ℕ := 5
def tomato_plants : ℕ := 5

/-- All tomato plants must be placed next to each other. --/
def tomatoes_next_to_each_other := true

/-- The row must start with a basil plant. --/
def starts_with_basil := true

/-- The number of ways to arrange the plants in a row under the given conditions is 11520. --/
theorem arrangement_count :
  basil_plants = 5 ∧ tomato_plants = 5 ∧ tomatoes_next_to_each_other ∧ starts_with_basil → 
  ∃ arrangements : ℕ, arrangements = 11520 :=
by 
  sorry

end arrangement_count_l149_149745


namespace geometric_progression_exists_l149_149006

theorem geometric_progression_exists :
  ∃ (b1 b2 b3 b4: ℤ) (q: ℤ), 
    b2 = b1 * q ∧ 
    b3 = b1 * q^2 ∧ 
    b4 = b1 * q^3 ∧  
    b3 - b1 = 9 ∧ 
    b2 - b4 = 18 ∧ 
    b1 = 3 ∧ b2 = -6 ∧ b3 = 12 ∧ b4 = -24 :=
sorry

end geometric_progression_exists_l149_149006


namespace value_range_of_a_l149_149806

variable (A B : Set ℝ)

noncomputable def A_def : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
noncomputable def B_def (a : ℝ) : Set ℝ := { x | x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0 }

theorem value_range_of_a (a : ℝ) (hA : A = A_def) (hB : B = B_def a) :
    (Bᶜ ∩ A = ∅) → (0 ≤ a ∧ a ≤ 0.5) := 
sorry

end value_range_of_a_l149_149806


namespace triangle_abc_is_right_triangle_l149_149530

theorem triangle_abc_is_right_triangle (BC AC AB : ℕ) 
  (hBC : BC = 6) 
  (hAC : AC = 8) 
  (hAB : AB = 10) : BC^2 + AC^2 = AB^2 :=
by
  rw [hBC, hAC, hAB]
  norm_num
  sorry

end triangle_abc_is_right_triangle_l149_149530


namespace sufficient_but_not_necessary_l149_149322

noncomputable def sufficient_condition (x : ℝ) : Prop := x > 1
noncomputable def necessary_condition (x : ℝ) : Prop := log (1 / 2) (x + 2) < 0

theorem sufficient_but_not_necessary (x : ℝ) :
  sufficient_condition x → necessary_condition x ∧ ¬(∀ y, necessary_condition y → sufficient_condition y) :=
begin
  sorry
end

end sufficient_but_not_necessary_l149_149322


namespace contrapositive_of_parallel_implies_equal_alternate_interior_angles_l149_149303

theorem contrapositive_of_parallel_implies_equal_alternate_interior_angles :
  (∀ (l₁ l₂ : Line), parallel l₁ l₂ → ∀ (α β : Angle), alternate_interior_angles α β l₁ l₂ → α = β) ↔
  (∀ (l₁ l₂ : Line), ∀ (α β : Angle), α ≠ β → ¬alternate_interior_angles α β l₁ l₂ → ¬parallel l₁ l₂) :=
sorry

end contrapositive_of_parallel_implies_equal_alternate_interior_angles_l149_149303


namespace reflect_x_axis_coordinates_l149_149518

structure Point where
  x : ℝ
  y : ℝ

def reflect_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

theorem reflect_x_axis_coordinates (p : Point) (hp : p = ⟨3, 1⟩) :
  reflect_x_axis p = ⟨3, -1⟩ := by
  simp [reflect_x_axis, hp]
  sorry

end reflect_x_axis_coordinates_l149_149518


namespace num_of_digits_M_divisible_by_4_l149_149342

theorem num_of_digits_M_divisible_by_4 : setOf (λ M : Nat, (20 + M) % 4 = 0).card = 3 :=
by {
  -- The proof would be required here, but for now, we acknowledge the conditions and the target theorem.
  sorry
}

end num_of_digits_M_divisible_by_4_l149_149342


namespace last_digit_of_seven_power_1000_is_7_last_two_digits_of_seven_power_1000_is_07_last_digit_of_seven_exponential_tower_1001_is_3_l149_149313

-- Prove that the last digit of 7^7^7... (1000 times) is 7
theorem last_digit_of_seven_power_1000_is_7 :
  let x := (7 : ℕ)
  let n := 1000 in
  x^n % 10 = 7 := 
sorry

-- Prove that the last two digits of 7^7^7... (1000 times) is 07
theorem last_two_digits_of_seven_power_1000_is_07 :
  let x := (7 : ℕ)
  let n := 1000 in
  x^n % 100 = 7 := 
sorry

-- Prove that the last digit of 7^(7^(7^(... using 1001 sevens))) is 3
theorem last_digit_of_seven_exponential_tower_1001_is_3 :
  let x := (7 : ℕ)
  let n := 1001 in
  sorry

end last_digit_of_seven_power_1000_is_7_last_two_digits_of_seven_power_1000_is_07_last_digit_of_seven_exponential_tower_1001_is_3_l149_149313


namespace tan_arithmetic_seq_l149_149130

theorem tan_arithmetic_seq (x y z : ℝ) (h₁ : x = y - π / 2) (h₂ : z = y + π / 2) :
    tan x * tan y + tan y * tan z + tan z * tan x = -3 := 
sorry

end tan_arithmetic_seq_l149_149130


namespace determine_a_l149_149820

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then x^3 + 1 else x^2 - a * x

theorem determine_a (a : ℝ) : 
  f (f 0 a) a = -2 → a = 3 :=
by
  sorry

end determine_a_l149_149820


namespace triangle_extension_equilateral_necessitates_k_l149_149552

open Complex

noncomputable def k : ℝ := 1 / Real.sqrt 3

theorem triangle_extension_equilateral_necessitates_k 
  (A B C D E F : ℂ) -- points in the complex plane representing vertices and intersections
  (k_pos : k > 0)
  (AA'_length : |A - A'| = k * |B - C|)
  (BB'_length : |B - B'| = k * |A - C|)
  (CC'_length : |C - C'| = k * |A - B|)
  (A''_def : ∃θ : ℝ, θ = π / 3 ∧ A'' = rotate (A' - A) θ + A)
  (tri_eq : is_equilateral A'' B' C') : 
  k = 1 / Real.sqrt 3 :=
begin
  sorry
end

end triangle_extension_equilateral_necessitates_k_l149_149552


namespace ellipse_dot_product_range_l149_149847

-- Define the ellipse and its conditions
def ellipse_eqn (x y : ℝ) (a : ℝ) : Prop :=
  a > 1 ∧ (x^2 / a^2 + y^2 = 1)

-- Define the geometric sequence condition for P inside the ellipse
def geom_seq_condition (x y : ℝ) (a : ℝ) (c : ℝ) (F1 F2 O : ℝ × ℝ) : Prop :=
  let PF1 := (x + c)^2 + y^2
  let PF2 := (x - c)^2 + y^2
  let PO := x^2 + y^2
  (PO = sqrt PF1 * sqrt PF2)

-- Define the dot product condition
def dot_product_condition (x y : ℝ) (a : ℝ) (c : ℝ) : Prop :=
  x^2 - y^2 = 1 ∧ (1 ≤ x^2 ∧ x^2 < 3 / 2) ∧ (-1 ≤ x^2 - 2 + y^2 ∧ x^2 - 2 + y^2 < 0)

-- Prove the equivalent problem
theorem ellipse_dot_product_range (a c x y : ℝ) (F1 F2 O : ℝ × ℝ) :
  ellipse_eqn x y a →
  geom_seq_condition x y a c F1 F2 O →
  dot_product_condition x y a c →
  -1 ≤ x^2 - 2 + y^2 ∧ x^2 - 2 + y^2 < 0 :=
by
  -- Proof steps would go here
  sorry

end ellipse_dot_product_range_l149_149847


namespace area_of_triangle_from_intercepts_l149_149489

theorem area_of_triangle_from_intercepts :
  let f := (λ x : ℝ, (x-3)^2 * (x+5))
  let x_intercepts := {x : ℝ | f x = 0}
  let y_intercept := f 0
  let base := abs (3 - (-5))
  let height := y_intercept
  in base = 8 ∧ height = 45 ∧ 
     (1 / 2) * base * height = 180 :=
by
  -- Definitions based on the intercepts
  let f := (λ x : ℝ, (x-3)^2 * (x+5))
  let x_intercepts := {x : ℝ | f x = 0}
  let y_intercept := f 0
  let base := abs (3 - (-5))
  let height := y_intercept
  -- Assert conditions
  have hx_intercepts : x_intercepts = {-5, 3},
  -- Calculate x-intercepts
  sorry,
  have hy_intercept : y_intercept = 45,
  -- Calculate y-intercept
  sorry,
  have hbase : base = 8,
  -- Calculate base
  sorry,
  have hheight : height = 45,
  -- Calculate height
  sorry,
  show base = 8 ∧ height = 45 ∧ (1 / 2) * base * height = 180,
  from And.intro hbase (And.intro hheight (by
    -- Calculate area
    sorry
  ))


end area_of_triangle_from_intercepts_l149_149489


namespace michael_needs_four_laps_to_pass_donovan_l149_149399

axiom donovan_normal_time : ℕ := 45
axiom michael_normal_time : ℕ := 40
axiom obstacles_per_lap : ℕ := 3
axiom donovan_obstacle_time : ℕ := 10
axiom michael_obstacle_time : ℕ := 5

def donovan_lap_time_with_obstacles : ℕ :=
  donovan_normal_time + obstacles_per_lap * donovan_obstacle_time

def michael_lap_time_with_obstacles : ℕ :=
  michael_normal_time + obstacles_per_lap * michael_obstacle_time

def time_gained_per_lap : ℕ :=
  donovan_lap_time_with_obstacles - michael_lap_time_with_obstacles

theorem michael_needs_four_laps_to_pass_donovan :
  (75 / time_gained_per_lap).ceil = 4 :=
by
  sorry

end michael_needs_four_laps_to_pass_donovan_l149_149399


namespace truck_wheels_l149_149652

theorem truck_wheels (t x : ℝ) (h₁ : t = 3.50 + 0.50 * (x - 2)) (h₂ : t = 5) : 
  let axles := 5 in
  let front_axle_wheels := 2 in
  let other_axle_wheels := 4 * (axles - 1) in
  front_axle_wheels + other_axle_wheels = 18 := by
sorry

end truck_wheels_l149_149652


namespace distance_between_A_and_B_l149_149245

-- Definition of points A and B on the number line
def A : ℤ := 3
def B : ℤ := -2

-- Definition of distance between points on a number line
def distance (x y : ℤ) : ℤ := abs (x - y)

-- The theorem to prove the distance between points A and B is 5
theorem distance_between_A_and_B : distance A B = 5 := by
  -- Utilize the defined points and distance function
  show distance 3 (-2) = 5,
  -- Calculate the distance
  calc
    distance 3 (-2) = abs (3 - (-2))  : rfl
                ... = abs (3 + 2)     : by rw sub_neg_eq_add
                ... = abs 5           : rfl
                ... = 5               : abs_of_nonneg (by norm_num)

end distance_between_A_and_B_l149_149245


namespace geometric_sequence_sum_l149_149067

theorem geometric_sequence_sum
  (q : ℤ) (a₁ : ℤ)
  (h₁ : a₁ + a₁ * q^3 = 18)
  (h₂ : a₁ * q + a₁ * q^2 = 12)
  (h₃ : q ≠ 1) :
  let S₈ := (a₁ * (1 - q^8)) / (1 - q)
  in S₈ = 510 :=
by
  sorry

end geometric_sequence_sum_l149_149067


namespace area_of_triangle_l149_149740

def vertex1 : ℝ × ℝ := (3, 1)
def vertex2 : ℝ × ℝ := (3, 6)
def vertex3 : ℝ × ℝ := (8, 6)

noncomputable def calculateArea (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  let base := Real.abs (v3.1 - v1.1)
  let height := Real.abs (v2.2 - v1.2)
  (1/2) * base * height

theorem area_of_triangle : calculateArea vertex1 vertex2 vertex3 = 12.5 := by
  sorry

end area_of_triangle_l149_149740


namespace apple_eating_contest_l149_149402

theorem apple_eating_contest (a z : ℕ) (h_most : a = 8) (h_fewest : z = 1) : a - z = 7 :=
by
  sorry

end apple_eating_contest_l149_149402


namespace bottom_rightmost_rectangle_is_E_l149_149895

-- Definitions of the given conditions
structure Rectangle where
  w : ℕ
  y : ℕ

def A : Rectangle := { w := 5, y := 8 }
def B : Rectangle := { w := 2, y := 4 }
def C : Rectangle := { w := 4, y := 6 }
def D : Rectangle := { w := 8, y := 5 }
def E : Rectangle := { w := 10, y := 9 }

-- The theorem we need to prove
theorem bottom_rightmost_rectangle_is_E :
    (E.w = 10) ∧ (E.y = 9) :=
by
  -- Proof would go here
  sorry

end bottom_rightmost_rectangle_is_E_l149_149895


namespace johns_outfit_cost_l149_149540

theorem johns_outfit_cost (pants_cost shirt_cost outfit_cost : ℝ)
    (h_pants : pants_cost = 50)
    (h_shirt : shirt_cost = pants_cost + 0.6 * pants_cost)
    (h_outfit : outfit_cost = pants_cost + shirt_cost) :
    outfit_cost = 130 :=
by
  sorry

end johns_outfit_cost_l149_149540


namespace tangent_lines_to_curve_at_l149_149838

noncomputable
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x

noncomputable
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 2)

theorem tangent_lines_to_curve_at (a : ℝ) :
  is_even_function (f' a) →
  (∀ x, f a x = - 2 → (2*x + (- f a x) = 0 ∨ 19*x - 4*(- f a x) - 27 = 0)) :=
by
  sorry

end tangent_lines_to_curve_at_l149_149838


namespace constant_term_in_expansion_l149_149978

-- Problem: Prove that the constant term in the expansion of (x * sqrt(x) - 1/x)^5 is -10
theorem constant_term_in_expansion :
  ∀ x : ℝ, (∃ k : ℝ, (x * k = 1) → 
  let term := (x * real.sqrt x - 1 / x) ^ 5 in 
    (∃ c : ℝ, c = -10)) :=
by sorry

end constant_term_in_expansion_l149_149978


namespace books_per_shelf_l149_149655

theorem books_per_shelf (total_distance : ℕ) (total_shelves : ℕ) (one_way_distance : ℕ) 
  (h1 : total_distance = 3200) (h2 : total_shelves = 4) (h3 : one_way_distance = total_distance / 2) 
  (h4 : one_way_distance = 1600) :
  ∀ books_per_shelf : ℕ, books_per_shelf = one_way_distance / total_shelves := 
by
  sorry

end books_per_shelf_l149_149655


namespace x_squared_minus_y_squared_l149_149869

theorem x_squared_minus_y_squared {x y : ℚ} 
    (h1 : x + y = 3/8) 
    (h2 : x - y = 5/24) 
    : x^2 - y^2 = 5/64 := 
by 
    -- The proof would go here
    sorry

end x_squared_minus_y_squared_l149_149869


namespace good_time_eq_bad_time_l149_149254

-- Definition of the behavior of the clock hands
def minute_hand_position (t : ℝ) : ℝ := 6 * (t / 60)
def hour_hand_position (t : ℝ) : ℝ := 0.5 * (t / 60)
def second_hand_position (t : ℝ) : ℝ := 6 * t

-- Definition of "good" time
def is_good_time (t : ℝ) : Prop :=
  minute_hand_position t > hour_hand_position t ∧
  second_hand_position t > minute_hand_position t

-- Total duration of "good" time over a 24-hour period equals the duration of "bad" time
theorem good_time_eq_bad_time : 
  ∃ (T : ℝ), ∀ t ∈ (0..86400), ∫ t in 0..86400, indicator (is_good_time t) = ∫ t in 0..86400, indicator (¬ is_good_time t) := 
sorry

end good_time_eq_bad_time_l149_149254


namespace floor_equation_has_finite_solutions_l149_149630

noncomputable def floor_equation_solutions : Prop := 
  ∃ x > 0, ([⌊x⌋] + [⌊1/x⌋]) = 3

theorem floor_equation_has_finite_solutions :
  floor_equation_solutions → ∃ n, (0 < n ∧ ∀ m, m > n → ¬floor_equation_solutions) :=
sorry

end floor_equation_has_finite_solutions_l149_149630


namespace income_from_investment_l149_149156

-- Definitions of the given conditions
def market_value : ℝ := 96
def rate : ℝ := 0.10
def investment : ℝ := 6240
def face_value_per_unit : ℝ := 100

-- Proof problem: Prove the income from the investment
theorem income_from_investment :
  let number_of_units := investment / market_value in
  let total_face_value := number_of_units * face_value_per_unit in
  let income := total_face_value * rate in
  income = 650 := 
by
  sorry

end income_from_investment_l149_149156


namespace area_of_gray_region_is_27pi_l149_149165

-- Define the conditions
def concentric_circles (inner_radius outer_radius : ℝ) :=
  2 * inner_radius = outer_radius

def width_of_gray_region (inner_radius outer_radius width : ℝ) :=
  width = outer_radius - inner_radius

-- Define the proof problem
theorem area_of_gray_region_is_27pi
(inner_radius outer_radius : ℝ) 
(h1 : concentric_circles inner_radius outer_radius)
(h2 : width_of_gray_region inner_radius outer_radius 3) :
π * outer_radius^2 - π * inner_radius^2 = 27 * π :=
by
  -- Proof goes here, but it is not required as per instructions
  sorry

end area_of_gray_region_is_27pi_l149_149165


namespace solve_equation_l149_149232

theorem solve_equation : ∃ x : ℝ, (2 * x - 1) / 3 - (x - 2) / 6 = 2 ∧ x = 4 :=
by
  sorry

end solve_equation_l149_149232


namespace convex_quad_angle_conditions_l149_149723

theorem convex_quad_angle_conditions (ABCD : ∀ (A B C D : ℝ), Prop)
  (convex : ∀ (A B C D : ℝ), ABCD A B C D → convex_quad A B C D)
  (no_triangle : ∀ (A B C D : ℝ), ABCD A B C D → ¬triangle_formed_by_three_sides A B C D):
  (∃ (A B C D : ℝ), ABCD A B C D ∧ (convex_quad A B C D → angle_condition_60 A B C D)) ∧ 
  (∃ (A B C D : ℝ), ABCD A B C D ∧ (convex_quad A B C D → angle_condition_120 A B C D)) :=
sorry

-- Definitions related to given conditions
def convex_quad (A B C D : ℝ) : Prop := sorry -- Placeholder for convex quadrilateral definition
def triangle_formed_by_three_sides (A B C D : ℝ) : Prop := sorry -- Placeholder for triangle formation rule 
def angle_condition_60 (A B C D : ℝ) : Prop := ∃ α, α ≤ 60 ∧ (one_angle_of_quad_is α A B C D)
def angle_condition_120 (A B C D : ℝ) : Prop := ∃ β, β ≥ 120 ∧ (one_angle_of_quad_is β A B C D)
def one_angle_of_quad_is (angle : ℝ) (A B C D : ℝ) : Prop := sorry -- Placeholder for specific angle condition

end convex_quad_angle_conditions_l149_149723


namespace p_sufficient_not_necessary_q_l149_149050

theorem p_sufficient_not_necessary_q (x : ℝ) :
  (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros h
  cases h with h1 h2
  split
  case left => linarith
  case right => linarith

end p_sufficient_not_necessary_q_l149_149050


namespace tom_steps_l149_149200

theorem tom_steps (matt_rate : ℕ) (tom_extra_rate : ℕ) (matt_steps : ℕ) (tom_rate : ℕ := matt_rate + tom_extra_rate) (time : ℕ := matt_steps / matt_rate)
(H_matt_rate : matt_rate = 20)
(H_tom_extra_rate : tom_extra_rate = 5)
(H_matt_steps : matt_steps = 220) :
  tom_rate * time = 275 :=
by
  -- We start the proof here, but leave it as sorry.
  sorry

end tom_steps_l149_149200


namespace triangle_perimeter_l149_149835

noncomputable def hyperbola_params : ℝ × ℝ × ℝ :=
  let a := 4
  let b := 3
  let c := Real.sqrt (a^2 + b^2)
  (a, b, c)

noncomputable def perimeter_triangle {F₁ F₂ : ℝ × ℝ} (AB : ℝ) : ℝ :=
  let (a, b, c) := hyperbola_params
  2 * a + AB + 2 * c

theorem triangle_perimeter {F₁ F₂ : ℝ × ℝ} (h : ∀ {x y : ℝ}, (x / 16)^2 - (y / 9)^2 = 1) :
  let F₁ := (-hyperbola_params.snd.snd, 0) in
  let F₂ := (hyperbola_params.snd.snd, 0) in
  perimeter_triangle 6 = 28 :=
by sorry

end triangle_perimeter_l149_149835


namespace min_value_of_a_l149_149879

theorem min_value_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_value_of_a_l149_149879


namespace wise_men_saved_l149_149636
-- Required import for basic mathematics operations

-- Defining the problem setup and the proof target
theorem wise_men_saved : ∃ strat : (fin 100 → bool) → (fin 100 → bool), 
  (∀ (caps : fin 100 → bool),
    let guesses := λ caps i => strat caps i in
    ∑ i in finset.range 100, ite (guesses i = caps i) 1 0 ≥ 99) := sorry

end wise_men_saved_l149_149636


namespace arctan_neg_sqrt_3_l149_149390

theorem arctan_neg_sqrt_3 :
  arctan (-sqrt 3) = -π / 3 := by
sorry

end arctan_neg_sqrt_3_l149_149390


namespace a_general_formula_T_general_formula_l149_149829

noncomputable def S (n : ℕ) : ℕ := n^2 + 7 * n

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_general_formula (n : ℕ) (hn : n ≥ 1) : a n = 2 * n + 6 := sorry

noncomputable def c (n : ℕ) : ℝ :=
  1 / ((a n - 7) * (a n - 5))

noncomputable def T (n : ℕ) : ℝ :=
  (∑ i in finset.range n, c (i + 1))

theorem T_general_formula (n : ℕ) (hn : n ≥ 1) : T n = n / (2 * n + 1) := sorry

end a_general_formula_T_general_formula_l149_149829


namespace limit_C_of_f_is_2_l149_149087

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}
variable {f' : ℝ}

noncomputable def differentiable_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ f' : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs (f (x + h) - f x - f' * h) / abs (h) < ε

axiom hf_differentiable : differentiable_at f x₀
axiom f'_at_x₀ : f' = 1

theorem limit_C_of_f_is_2 
  (hf_differentiable : differentiable_at f x₀) 
  (h_f'_at_x₀ : f' = 1) : 
  (∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ + 2 * Δx) - f x₀) / Δx - 2) < ε) :=
sorry

end limit_C_of_f_is_2_l149_149087


namespace probability_exactly_3_tails_l149_149805

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_exactly_3_tails :
  binomial_probability 8 3 (2 / 3) = 448 / 6561 :=
by
  sorry

end probability_exactly_3_tails_l149_149805


namespace original_algae_number_l149_149950

-- Define given conditions
def current_algae := 3263
def increase_algae := 2454

-- Define the original algae based on the conditions
def original_algae := current_algae - increase_algae

-- State the theorem to prove the original number of algae given the conditions
theorem original_algae_number :
  original_algae = 809 :=
by
  dsimp [original_algae, current_algae, increase_algae]
  rfl

end original_algae_number_l149_149950


namespace base_7_digits_count_l149_149112

theorem base_7_digits_count (n : ℕ) (h : n = 2048) :
  nat.log 7 n + 1 = 4 := 
by {
  sorry
}

end base_7_digits_count_l149_149112


namespace domain_of_f_l149_149779

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 6*x + 9)

theorem domain_of_f :
  {x : ℝ | f x ≠ f (-3)} = Iio (-3) ∪ Ioi (-3) :=
by
  sorry

end domain_of_f_l149_149779


namespace memorial_visits_l149_149599

theorem memorial_visits (x : ℕ) (total_visits : ℕ) (difference : ℕ) 
  (h1 : total_visits = 589) 
  (h2 : difference = 56) 
  (h3 : 2 * x + difference = total_visits - x) : 
  2 * x + 56 = 589 - x :=
by
  -- proof steps would go here
  sorry

end memorial_visits_l149_149599


namespace probability_passing_exam_l149_149146

-- Define probabilities for sets A, B, and C, and passing conditions
def P_A := 0.3
def P_B := 0.3
def P_C := 1 - P_A - P_B
def P_D_given_A := 0.8
def P_D_given_B := 0.6
def P_D_given_C := 0.8

-- Total probability of passing
def P_D := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

-- Proof that the total probability of passing is 0.74
theorem probability_passing_exam : P_D = 0.74 :=
by
  -- (skip the proof steps)
  sorry

end probability_passing_exam_l149_149146


namespace average_sleep_is_8_l149_149109

-- Define the hours of sleep for each day
def mondaySleep : ℕ := 8
def tuesdaySleep : ℕ := 7
def wednesdaySleep : ℕ := 8
def thursdaySleep : ℕ := 10
def fridaySleep : ℕ := 7

-- Calculate the total hours of sleep over the week
def totalSleep : ℕ := mondaySleep + tuesdaySleep + wednesdaySleep + thursdaySleep + fridaySleep
-- Define the total number of days
def totalDays : ℕ := 5

-- Calculate the average sleep per night
def averageSleepPerNight : ℕ := totalSleep / totalDays

-- Prove the statement
theorem average_sleep_is_8 : averageSleepPerNight = 8 := 
by
  -- All conditions are automatically taken into account as definitions
  -- Add a placeholder to skip the actual proof
  sorry

end average_sleep_is_8_l149_149109


namespace equal_sets_count_l149_149684

theorem equal_sets_count (x y a : ℝ) : 
  (3 * x^2 = x^2 + x^2 + x^2) ∧ ((x-y)^2 = (y-x)^2) ∧ ((a^2)^3 = (a^3)^2) → 3 =
  begin
    sorry
  end :=
begin
  sorry
end

end equal_sets_count_l149_149684


namespace decorations_cost_correct_l149_149573

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end decorations_cost_correct_l149_149573


namespace volume_ratio_lateral_surface_ratio_l149_149733

variables (R p H l : ℝ)
 
-- Conditions
axiom cone_volume : ∀ H R, (1/3) * Real.pi * R^2 * H
axiom pyramid_volume : ∀ H R p, (1/3) * p * R * H
axiom cone_lateral_surface : ∀ R l, Real.pi * R * l
axiom pyramid_lateral_surface : ∀ p l, p * l

theorem volume_ratio (R p H : ℝ) (hR : R > 0) (hp : p > 0) (hH : H > 0) :
    let V1 := cone_volume H R
    let V2 := pyramid_volume H R p
    V1 / V2 = Real.pi * R / p := 
by 
    intro V1 V2
    rw [cone_volume, pyramid_volume]
    field_simp [hp, hH, hR]
    ring
    sorry

theorem lateral_surface_ratio (R p l : ℝ) (hR : R > 0) (hp : p > 0) (hl : l > 0) :
    let S1 := cone_lateral_surface R l
    let S2 := pyramid_lateral_surface p l
    S1 / S2 = Real.pi * R / p := 
by 
    intro S1 S2
    rw [cone_lateral_surface, pyramid_lateral_surface]
    field_simp [hp, hl, hR]
    ring
    sorry

end volume_ratio_lateral_surface_ratio_l149_149733


namespace square_area_correct_l149_149312

-- Define the length of the side of the square
def side_length : ℕ := 15

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- Define the area calculation for a triangle using the square area division
def triangle_area (square_area : ℕ) : ℕ := square_area / 2

-- Theorem stating that the area of a square with given side length is 225 square units
theorem square_area_correct : square_area side_length = 225 := by
  sorry

end square_area_correct_l149_149312


namespace remaining_homes_l149_149542

theorem remaining_homes (total_homes : ℕ) (first_hour_fraction : ℚ) (second_hour_fraction : ℚ) : 
  total_homes = 200 →
  first_hour_fraction = 2/5 →
  second_hour_fraction = 60/100 →
  let
    first_distributed := first_hour_fraction * total_homes,
    remaining_after_first := total_homes - first_distributed,
    second_distributed := second_hour_fraction * remaining_after_first,
    remaining_after_second := remaining_after_first - second_distributed
  in
  remaining_after_second = 48 := 
by
  intros h_total h_first_fraction h_second_fraction,
  let first_distributed := first_hour_fraction * total_homes,
  let remaining_after_first := total_homes - first_distributed,
  let second_distributed := second_hour_fraction * remaining_after_first,
  let remaining_after_second := remaining_after_first - second_distributed,
  sorry -- proof goes here

end remaining_homes_l149_149542


namespace final_digit_after_removal_l149_149307

theorem final_digit_after_removal (n : ℕ) (h : n = 1995) : 
  (let digits := (List.range' 1 (n + 1)).bind (λ m => m.digits 10);
   let rec remove_digits (ds : List ℕ) (even : Bool) : List ℕ :=
     match ds with
     | [] => []
     | x::xs => if even then remove_digits xs false else x::(remove_digits xs true);
   (digit := (List.repeat (λ _ => true) (digits.length)).reverse;
   List.length digits % 3 = 0 → remove_digits (remove_digits digits true) false ! list := [(9: nat)])) := sorry

end final_digit_after_removal_l149_149307


namespace problem_statement_l149_149679

theorem problem_statement (x y z : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : z = 3) :
  x^2 + y^2 + z^2 + 2*x*z = 26 :=
by
  rw [h1, h2, h3]
  norm_num

end problem_statement_l149_149679


namespace sum_of_reciprocals_leq_series_l149_149178

theorem sum_of_reciprocals_leq_series (n_k : ℕ → ℕ)
  (h : ∀ i j : ℕ, i < j → ¬(n_k i).to_string.is_prefix_of (n_k j).to_string) :
  ∑' k, (1 : ℝ) / n_k k ≤ ∑ k in finset.range 9, (1 : ℝ) / (k + 1) := 
sorry

end sum_of_reciprocals_leq_series_l149_149178


namespace ratio_arithmetic_geometric_seq_l149_149082

theorem ratio_arithmetic_geometric_seq (x y a1 a2 b1 b2 : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h_arith_seq : a1 = (x + y) / 2 ∧ a2 = (x + y) / 2)
  (h_geom_seq : b1 * b2 = x * y) :
  (0 < x ∧ 0 < y) → (a1 + a2)^2 / (b1 * b2) ≥ 4 :=
begin
  sorry
end

end ratio_arithmetic_geometric_seq_l149_149082


namespace product_leq_neg_inv_l149_149647

theorem product_leq_neg_inv {n : ℕ} (h_n : 0 < n) (x : Fin n → ℝ) 
  (h_sum : ∑ i, x i = 0) 
  (h_sum_sq : ∑ i, (x i)^2 = 1) : 
  ∃ i j, i ≠ j ∧ x i * x j ≤ -1 / n :=
by sorry

end product_leq_neg_inv_l149_149647


namespace proof_problem1_proof_problem2_l149_149325

noncomputable def problem1 : ℝ :=
  3 * (-2 : ℝ)^3 - (1 / 2 : ℝ)^0 + (0.25 : ℝ)^(1 / 2) * (1 / Real.sqrt 2)^(-4)

theorem proof_problem1 : problem1 = -23 :=
by
  calc problem1 = 3 * (-2)^3 - (1 / 2)^0 + 0.25^(1 / 2) * (1 / Real.sqrt 2)^(-4) : rfl
            ... = -23 : sorry

noncomputable def problem2 : ℝ :=
  (Real.log 2)^2 + Real.log 5 * Real.log 20 + Real.log 100 + Real.log (1 / 6) + Real.log 0.006

theorem proof_problem2 : problem2 = 2 :=
by
  calc problem2 = (Real.log 2)^2 + Real.log 5 * Real.log 20 + Real.log 100 + Real.log (1 / 6) + Real.log 0.006 : rfl
            ... = 2 : sorry

end proof_problem1_proof_problem2_l149_149325


namespace tangent_line_eq_max_min_values_l149_149467

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x + 1

theorem tangent_line_eq : (Real.exp 2 - 1) * (x - 2) + (Real.exp 2 - 1) = (Real.exp 2 - 1) * x - y - Real.exp 2 + 1 :=
by
  sorry

theorem max_min_values :
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, f(-2) ≤ f x ∧ f x ≤ f(1)) ∧
    f(0) = 2 :=
by
  sorry

end tangent_line_eq_max_min_values_l149_149467


namespace find_valid_integers_count_l149_149414

theorem find_valid_integers_count :
  let count := (Finset.filter (λ n, Nat.gcd n 17 ≠ 17) (Finset.range (2018))).card
  in count = 1899 := by
let count := (Finset.filter (λ n, Nat.gcd n 17 ≠ 17) (Finset.range (2018))).card
show count = 1899
sorry

end find_valid_integers_count_l149_149414


namespace circumference_of_circle_x_l149_149755

theorem circumference_of_circle_x (A_x A_y : ℝ) (r_x r_y C_x : ℝ)
  (h_area: A_x = A_y) (h_half_radius_y: r_y = 2 * 5)
  (h_area_y: A_y = Real.pi * r_y^2)
  (h_area_x: A_x = Real.pi * r_x^2)
  (h_circumference_x: C_x = 2 * Real.pi * r_x) :
  C_x = 20 * Real.pi :=
by
  sorry

end circumference_of_circle_x_l149_149755


namespace find_coordinates_of_M_l149_149440

theorem find_coordinates_of_M :
  ∃ y : ℝ, (∀ M : ℝ × ℝ × ℝ, M = (0, y, 0) → 
    ((real.sqrt ((3 - 0)^2 + (2 - y)^2 + (0 - 0)^2) = real.sqrt ((2 - 0)^2 + (-1 - y)^2 + (2 - 0)^2)) → 
    M = (0, 2/3, 0))) :=
sorry

end find_coordinates_of_M_l149_149440


namespace sum_of_digits_of_2001st_remarkable_l149_149945

-- Definition: A natural number is "remarkable" if it is the smallest among natural numbers with the same sum of digits as it.
def is_remarkable (n : ℕ) : Prop :=
  ∀ m : ℕ, (∑ d in n.digits 10, d) = (∑ d in m.digits 10, d) → n ≤ m

-- The main theorem to prove
theorem sum_of_digits_of_2001st_remarkable :
  ∃ n : ℕ, is_remarkable n ∧ n = 2001 ∧ (∑ d in n.digits 10, d) = 2001 :=
by
  sorry

end sum_of_digits_of_2001st_remarkable_l149_149945


namespace find_k_l149_149857

-- Define the vectors and the condition that k · a + b is perpendicular to a
theorem find_k 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (k : ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (-2, 0))
  (h_perpendicular : ∀ (k : ℝ), (k * a.1 + b.1, k * a.2 + b.2) • a = 0 ) : k = 2 / 5 :=
sorry

end find_k_l149_149857


namespace largest_is_314_l149_149658

constant y : ℕ
constant x : ℕ
constant a : ℕ
constant b : ℕ

axiom h1 : 4 ∈ {4, 314, y, x, a, b}
axiom h2 : 314 ∈ {4, 314, y, x, a, b}
axiom h3 : (∀ n ∈ {4, 314, y, x, a, b}, 0 ≤ n) -- assuming all numbers are natural
axiom range_12 : ∃ min max, min ∈ {4, 314, y, x, a, b} ∧ max ∈ {4, 314, y, x, a, b} ∧ max - min = 12
axiom x_diff : (max_x = max { x | x ∈ {4, 314, y, x, a, b} } ∧ min_x = min { x | x ∈ {4, 314, y, x, a, b} }) ∧ max_x - min_x = 13

theorem largest_is_314 :
  ∃ largest ∈ {4, 314, y, x, a, b}, largest = 314 :=
sorry

end largest_is_314_l149_149658


namespace systematic_sampling_mean_A_eq_98_mean_B_eq_98_variance_A_eq_22_div_5_variance_B_eq_64_div_5_more_stable_A_l149_149344

noncomputable def mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) : ℝ :=
  let x̄ = mean data
  ((data.map (λ x, (x - x̄)^2)).sum) / (data.length)

def workshop_weights_A : List ℝ := [100, 96, 101, 96, 97]
def workshop_weights_B : List ℝ := [103, 93, 100, 95, 99]

theorem systematic_sampling : true := sorry

theorem mean_A_eq_98 : mean workshop_weights_A = 98 := sorry

theorem mean_B_eq_98 : mean workshop_weights_B = 98 := sorry

theorem variance_A_eq_22_div_5 : variance workshop_weights_A = 22 / 5 := sorry

theorem variance_B_eq_64_div_5 : variance workshop_weights_B = 64 / 5 := sorry

theorem more_stable_A : variance workshop_weights_A < variance workshop_weights_B := sorry

end systematic_sampling_mean_A_eq_98_mean_B_eq_98_variance_A_eq_22_div_5_variance_B_eq_64_div_5_more_stable_A_l149_149344


namespace max_k_exists_l149_149033

theorem max_k_exists :
  ∃ (k : ℕ), 
    (∀ (a b : ℕ) (i j : ℕ), i ≠ j → a_i < b_i ∧ a_i + b_i < 4000 ∧ a_i + b_i ≠ a_j + b_j) →
    (∀ (x : ℕ), x ∈ (set.range $ λ i, a_i) ∪ (set.range $ λ i, b_i) → x ≤ 3000) → 
    k = 1599 :=
by
  sorry

end max_k_exists_l149_149033


namespace sum_of_two_primes_is_odd_then_one_is_two_l149_149651

theorem sum_of_two_primes_is_odd_then_one_is_two (p q : ℕ) (hp : p.prime) (hq : q.prime) (h_sum_odd : (p + q) % 2 = 1) : p = 2 ∨ q = 2 :=
sorry

end sum_of_two_primes_is_odd_then_one_is_two_l149_149651


namespace steve_take_home_pay_l149_149616

def annual_salary : ℕ := 40000
def tax_percentage : ℝ := 0.20
def healthcare_percentage : ℝ := 0.10
def union_dues : ℕ := 800

theorem steve_take_home_pay :
  annual_salary - (annual_salary * tax_percentage).to_nat - (annual_salary * healthcare_percentage).to_nat - union_dues = 27200 :=
by
  sorry

end steve_take_home_pay_l149_149616


namespace area_triangle_ABP_l149_149840

noncomputable theory

-- Definitions based on conditions:
def is_focus (C : parabola) (F : Point) : Prop := C.focus = F
def is_perpendicular (l : Line) (axis : Line) : Prop := l ⊥ axis
def chord_intersects_parabola (l : Line) (C : parabola) (A B : Point) : Prop := 
  intersects l C A ∧ intersects l C B ∧ distance A B = 12
def point_on_directrix (P : Point) (C : parabola) : Prop := P ∈ C.directrix

-- The goal is to prove:
theorem area_triangle_ABP (C : parabola) (l : Line) (F P A B : Point) 
  (hF : is_focus C F) 
  (h_perpendicular : is_perpendicular l C.axis_of_symmetry) 
  (h_chord : chord_intersects_parabola l C A B) 
  (hP : point_on_directrix P C) : 
  area (triangle A B P) = 36 :=
sorry

end area_triangle_ABP_l149_149840


namespace tangent_circles_through_P_l149_149479

variable {C_1 C_2 : Circle}
variable {P : Point}
variable (B_1 B_2 : Circle)
variable (radical_axis : Line)

-- Conditions
axiom tangent_C1_C2 : Tangent C_1 C_2
axiom P_on_radical_axis : On P radical_axis
axiom radical_axis_perpendicular : Perpendicular radical_axis (line_through_centers C_1 C_2)

-- Conclusion to prove
theorem tangent_circles_through_P :
  (∀ C, Tangent C C_1 ∧ Tangent C C_2 ∧ On P C.center -> C = B_1 ∨ C = B_2) := 
by
  sorry

end tangent_circles_through_P_l149_149479


namespace num_valid_choices_l149_149541

theorem num_valid_choices : 
  {n : Int | 1 ≤ n ∧ n ≤ 500 ∧ 360 % n = 0}.card = 24 :=
sorry

end num_valid_choices_l149_149541


namespace hybrids_with_full_headlights_l149_149141

theorem hybrids_with_full_headlights
  (total_cars : ℕ) (hybrid_percentage : ℕ) (one_headlight_percentage : ℕ) :
  total_cars = 600 → hybrid_percentage = 60 → one_headlight_percentage = 40 →
  let total_hybrids := (hybrid_percentage * total_cars) / 100 in
  let one_headlight_hybrids := (one_headlight_percentage * total_hybrids) / 100 in
  let full_headlight_hybrids := total_hybrids - one_headlight_hybrids in
  full_headlight_hybrids = 216 :=
by
  intros h1 h2 h3
  sorry

end hybrids_with_full_headlights_l149_149141


namespace james_brothers_count_l149_149537

theorem james_brothers_count (market_value : ℝ) (house_price_multiplier : ℝ) (revenue_after_taxes_percentage : ℝ) (amount_per_person : ℝ) : 
  (market_value = 500000) ∧ (house_price_multiplier = 1.2) ∧ (revenue_after_taxes_percentage = 0.9) ∧ (amount_per_person = 135000) →
  let selling_price := market_value * house_price_multiplier in
  let revenue_after_taxes := selling_price * revenue_after_taxes_percentage in
  let number_of_people := revenue_after_taxes / amount_per_person in
  let number_of_brothers := number_of_people - 1 in
  number_of_brothers = 3 :=
by
  intros h
  have := h.1
  sorry

end james_brothers_count_l149_149537


namespace greatest_value_q_minus_r_l149_149268

theorem greatest_value_q_minus_r : ∃ q r : ℕ, 1043 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 37) :=
by {
  sorry
}

end greatest_value_q_minus_r_l149_149268


namespace steve_take_home_pay_l149_149611

-- Defining the conditions
def annual_salary : ℕ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℕ := 800

-- Defining the taxes function
def taxes (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the healthcare function
def healthcare (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the total deductions function
def total_deductions (salary : ℕ) (tax_rate : ℝ) (healthcare_rate : ℝ) (dues : ℕ) : ℝ :=
  (taxes salary tax_rate) + (healthcare salary healthcare_rate) + dues

-- Defining the take-home pay function
def take_home_pay (salary : ℕ) (deductions : ℝ) : ℝ :=
  salary - deductions

-- Using definitions to assert the take-home pay
theorem steve_take_home_pay : take_home_pay annual_salary (total_deductions annual_salary tax_rate healthcare_rate union_dues) = 27200 :=
by
  -- proof has been skipped
  sorry

end steve_take_home_pay_l149_149611


namespace exists_infinitely_many_good_pairs_l149_149499

def is_prime_power {n : ℕ} (a : ℕ) : Prop :=
  ∃ p k : ℕ, Nat.Prime p ∧ a = p ^ k ∧ k > 1

def is_good_number (n : ℕ) : Prop :=
  ∀ p : ℕ, p ∣ n → is_prime_power p

theorem exists_infinitely_many_good_pairs :
  ∃ f : ℕ → ℕ, (∀ n, is_good_number (f n) ∧ is_good_number (f n + 1) ∧ Nat.coprime (f n) (f n + 1)) ∧
  f 0 = 8 ∧ ∀ n, f (n + 1) = 4 * (f n) * (f n + 1) :=
by
  sorry

end exists_infinitely_many_good_pairs_l149_149499


namespace polynomial_no_rational_roots_l149_149314

noncomputable def has_no_rational_root (p : ℤ[X]) : Prop :=
  ∀ (r : ℚ), ¬ (p.eval r = 0)

theorem polynomial_no_rational_roots
  (p : ℤ[X])
  (h1 : p.leadingCoeff % 2 = 1)
  (h2 : p.coeff 0 % 2 = 1)
  (h3 : (∑ i in p.support, p.coeff i) % 2 = 1)
  : has_no_rational_root p := 
sorry

end polynomial_no_rational_roots_l149_149314


namespace Mika_water_left_l149_149202

theorem Mika_water_left :
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  initial_amount - used_amount = 5 / 4 :=
by
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  show initial_amount - used_amount = 5 / 4
  sorry

end Mika_water_left_l149_149202


namespace chord_intersects_inner_circle_probability_l149_149667

theorem chord_intersects_inner_circle_probability :
  let outer_circle_radius := 4
  let inner_circle_radius := 2
  (∀ (A B : ℝ) (ha : 0 ≤ A < 2 * π) (hb : 0 ≤ B < 2 * π), 
      ∃ (P Q : ℝ), 
      (chord_intersects_inner_circle outer_circle_radius inner_circle_radius ha hb)) →
  probability (chord_intersects inner_circle_radius outer_circle_radius) = 5 / 6 := 
sorry

end chord_intersects_inner_circle_probability_l149_149667


namespace rooks_same_distance_l149_149660

theorem rooks_same_distance (rooks : Fin 8 → (ℕ × ℕ)) 
    (h_non_attacking : ∀ i j, i ≠ j → Prod.fst (rooks i) ≠ Prod.fst (rooks j) ∧ Prod.snd (rooks i) ≠ Prod.snd (rooks j)) 
    : ∃ i j k l, i ≠ j ∧ k ≠ l ∧ (Prod.fst (rooks i) - Prod.fst (rooks k))^2 + (Prod.snd (rooks i) - Prod.snd (rooks k))^2 = (Prod.fst (rooks j) - Prod.fst (rooks l))^2 + (Prod.snd (rooks j) - Prod.snd (rooks l))^2 :=
by 
  -- Proof goes here
  sorry

end rooks_same_distance_l149_149660


namespace triangle_AO_parallel_HK_l149_149436

section TriangleOrthocenterIncenterCircumcenter

variables {A B C H I O K : Point}
variables (triangle : Triangle A B C)
variables [triangle.orthocenter H]
variables [triangle.incenter I]
variables [triangle.circumcenter O]
variables [triangle.incircle_point K (LineSegment B C)]
variables [IO_parallel_BC : IO ∥ BC]

open Triangle

theorem triangle_AO_parallel_HK
  (h1 : H = orthocenter A B C)
  (h2 : I = incenter A B C)
  (h3 : O = circumcenter A B C)
  (h4 : incircle_point K (LineSegment B C))
  (h5 : IO ∥ LineSegment B C) :
  AO ∥ HK := 
sorry

end TriangleOrthocenterIncenterCircumcenter

end triangle_AO_parallel_HK_l149_149436


namespace rect_RS_over_HJ_zero_l149_149594

theorem rect_RS_over_HJ_zero :
  ∃ (A B C D H I J R S: ℝ × ℝ),
    (A = (0, 6)) ∧
    (B = (8, 6)) ∧
    (C = (8, 0)) ∧
    (D = (0, 0)) ∧
    (H = (5, 6)) ∧
    (I = (8, 4)) ∧
    (J = (3, 0)) ∧
    (R = (15 / 13, -12 / 13)) ∧
    (S = (15 / 13, -12 / 13)) ∧
    (RS = dist R S) ∧
    (HJ = dist H J) ∧
    (HJ ≠ 0) ∧
    (RS / HJ = 0) :=
sorry

end rect_RS_over_HJ_zero_l149_149594


namespace yoongi_age_l149_149691

theorem yoongi_age (H Yoongi : ℕ) : H = Yoongi + 2 ∧ H + Yoongi = 18 → Yoongi = 8 :=
by
  sorry

end yoongi_age_l149_149691


namespace products_of_two_distinct_divisors_of_1000_l149_149928

-- Define the set of positive integer divisors of 1000
def T : Set ℕ := {d | d ∣ 1000 ∧ d > 0}

-- Prove the number of numbers which are products of two distinct elements of T is 46
theorem products_of_two_distinct_divisors_of_1000 : 
  (set.count {x | ∃ a b ∈ T, a ≠ b ∧ x = a * b}) = 46 := 
sorry

end products_of_two_distinct_divisors_of_1000_l149_149928


namespace p_implies_q_and_not_q_implies_p_l149_149057

variable {x : ℝ}

def p : Prop := 0 < x ∧ x < 2
def q : Prop := -1 < x ∧ x < 3

theorem p_implies_q_and_not_q_implies_p : (p → q) ∧ ¬ (q → p) := 
by
  sorry

end p_implies_q_and_not_q_implies_p_l149_149057


namespace tan_405_eq_1_l149_149391

theorem tan_405_eq_1 : Real.tan (405 * Real.pi / 180) = 1 := 
by 
  sorry

end tan_405_eq_1_l149_149391


namespace a_divides_next_l149_149764

def a : ℕ → ℕ
| 0       := 2
| (n + 1) := 2^(a n) + 2

theorem a_divides_next (n : ℕ) (h : n ≥ 1) : a n ∣ a (n + 1) :=
sorry

end a_divides_next_l149_149764


namespace inequality_holds_l149_149812

theorem inequality_holds (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by 
  sorry

end inequality_holds_l149_149812


namespace mary_initial_blue_crayons_l149_149196

/-- **Mathematically equivalent proof problem**:
  Given that Mary has 5 green crayons and gives away 3 green crayons and 1 blue crayon,
  and she has 9 crayons left, prove that she initially had 8 blue crayons. 
  -/
theorem mary_initial_blue_crayons (initial_green_crayons : ℕ) (green_given_away : ℕ) (blue_given_away : ℕ)
  (crayons_left : ℕ) (initial_crayons : ℕ) :
  initial_green_crayons = 5 →
  green_given_away = 3 →
  blue_given_away = 1 →
  crayons_left = 9 →
  initial_crayons = crayons_left + (green_given_away + blue_given_away) →
  initial_crayons - initial_green_crayons = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end mary_initial_blue_crayons_l149_149196


namespace p_sufficient_not_necessary_q_l149_149052

theorem p_sufficient_not_necessary_q (x : ℝ) :
  (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros h
  cases h with h1 h2
  split
  case left => linarith
  case right => linarith

end p_sufficient_not_necessary_q_l149_149052


namespace probability_of_two_eights_l149_149401

-- Define a function that calculates the factorial of a number
noncomputable def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Definition of binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  fact n / (fact k * fact (n - k))

-- Probability of exactly two dice showing 8 out of eight 8-sided dice
noncomputable def prob_exactly_two_eights : ℚ :=
  binom 8 2 * ((1 / 8 : ℚ) ^ 2) * ((7 / 8 : ℚ) ^ 6)

-- Main theorem statement
theorem probability_of_two_eights :
  prob_exactly_two_eights = 0.196 := by
  sorry

end probability_of_two_eights_l149_149401


namespace students_participated_in_function_l149_149712

theorem students_participated_in_function :
  ∀ (B G : ℕ),
  B + G = 800 →
  (3 / 4 : ℚ) * G = 150 →
  (2 / 3 : ℚ) * B + 150 = 550 :=
by
  intros B G h1 h2
  sorry

end students_participated_in_function_l149_149712


namespace steve_take_home_pay_l149_149613

-- Defining the conditions
def annual_salary : ℕ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℕ := 800

-- Defining the taxes function
def taxes (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the healthcare function
def healthcare (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the total deductions function
def total_deductions (salary : ℕ) (tax_rate : ℝ) (healthcare_rate : ℝ) (dues : ℕ) : ℝ :=
  (taxes salary tax_rate) + (healthcare salary healthcare_rate) + dues

-- Defining the take-home pay function
def take_home_pay (salary : ℕ) (deductions : ℝ) : ℝ :=
  salary - deductions

-- Using definitions to assert the take-home pay
theorem steve_take_home_pay : take_home_pay annual_salary (total_deductions annual_salary tax_rate healthcare_rate union_dues) = 27200 :=
by
  -- proof has been skipped
  sorry

end steve_take_home_pay_l149_149613


namespace complement_union_l149_149450

open Set

universe u

variable {U : Type u} [Fintype U] [DecidableEq U]
variable {A B : Set U}

def complement (s : Set U) : Set U := {x | x ∉ s}

theorem complement_union {U : Set ℕ} (A B : Set ℕ) 
  (h1 : complement A ∩ B = {1})
  (h2 : A ∩ B = {3})
  (h3 : complement A ∩ complement B = {2}) :
  complement (A ∪ B) = {2} :=
by sorry

end complement_union_l149_149450


namespace sqrt_2023_bounds_l149_149656

theorem sqrt_2023_bounds :
  40 < Real.sqrt 2023 ∧ Real.sqrt 2023 < 45 :=
by
  -- Condition: 1600 < 2023 < 2025
  have h₁ : 1600 < 2023 := by norm_num
  have h₂ : 2023 < 2025 := by norm_num
  -- Taking square roots and using known values
  have h₃ : Real.sqrt 1600 = 40 := Real.sqrt_eq_iff_sqr_eq.2 ⟨40, by norm_num⟩
  have h₄ : Real.sqrt 2025 = 45 := Real.sqrt_eq_iff_sqr_eq.2 ⟨45, by norm_num⟩
  -- Therefore
  have h₅ : Real.sqrt 1600 < Real.sqrt 2023 := Real.sqrt_lt.2 ⟨h₁, (by norm_num : 0 ≤ 1600)⟩
  have h₆ : Real.sqrt 2023 < Real.sqrt 2025 := Real.sqrt_lt.2 ⟨h₂, (by norm_num : 0 ≤ 2023)⟩
  -- Combining bounds
  exact ⟨by simp [h₃, h₅], by simp [h₄, h₆]⟩

end sqrt_2023_bounds_l149_149656


namespace shortest_side_of_triangle_l149_149513

theorem shortest_side_of_triangle
  (a b : ℝ)
  (h_base : a + b = 90)
  (h_angle : ∃ C : Type, ∃ B : Type, ∃ A : Type, C - (A + B) = 60)
  (base_length: ∃ AB : Type, AB = 80)
  : min a b = 40 :=
by
  sorry

end shortest_side_of_triangle_l149_149513


namespace distance_between_table_1_and_table_3_l149_149034

theorem distance_between_table_1_and_table_3
  (last_year_race_distance : ℕ)
  (this_year_multiplier : ℕ)
  (num_tables : ℕ)
  (last_table_at_finish : Bool)
  (race_distance : ℕ := this_year_multiplier * last_year_race_distance)
  (intervals : ℕ := num_tables - 1)
  (distance_between_tables : ℕ := race_distance / intervals)
  (target_distance : ℕ := 2 * distance_between_tables)
  (last_year_race_distance = 300 : last_year_race_distance = 300)
  (this_year_multiplier = 4 : this_year_multiplier = 4)
  (num_tables = 6 : num_tables = 6)
  (last_table_at_finish = true : last_table_at_finish = true)
: target_distance = 480 := 
sorry

end distance_between_table_1_and_table_3_l149_149034


namespace min_sum_distances_to_corners_of_rectangle_center_l149_149718

theorem min_sum_distances_to_corners_of_rectangle_center (P A B C D : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (1, 0))
  (hC : C = (1, 1))
  (hD : D = (0, 1))
  (hP_center : P = (0.5, 0.5)) :
  ∀ Q, (dist Q A + dist Q B + dist Q C + dist Q D) ≥ (dist P A + dist P B + dist P C + dist P D) := 
sorry

end min_sum_distances_to_corners_of_rectangle_center_l149_149718


namespace tangent_lines_through_P_l149_149460

def curve (x : ℝ) : ℝ := (1 / 3) * x^3 + (4 / 3)

theorem tangent_lines_through_P :
  let P := (2 : ℝ, 4 : ℝ) in
  ∃ m b : ℝ, (∀ x : ℝ, curve x = m * x + b → (4 : ℝ) = m * (2 : ℝ) + b)
  ∧ ((∀ x : ℝ, P.2 = m * P.1 + b → (b = 2 ∧ m = 1) ∨ (b = -4 ∧ m = 4))) :=
sorry

end tangent_lines_through_P_l149_149460


namespace dividend_value_l149_149417

def dividend (divisor quotient remainder : ℝ) := (divisor * quotient) + remainder

theorem dividend_value :
  dividend 35.8 21.65 11.3 = 786.47 :=
by
  sorry

end dividend_value_l149_149417


namespace cindy_total_time_to_travel_one_mile_l149_149384

-- Definitions for the conditions
def run_speed : ℝ := 3 -- Cindy's running speed in miles per hour.
def walk_speed : ℝ := 1 -- Cindy's walking speed in miles per hour.
def run_distance : ℝ := 0.5 -- Distance run by Cindy in miles.
def walk_distance : ℝ := 0.5 -- Distance walked by Cindy in miles.

-- Theorem statement
theorem cindy_total_time_to_travel_one_mile : 
  ((run_distance / run_speed) + (walk_distance / walk_speed)) * 60 = 40 := 
by
  sorry

end cindy_total_time_to_travel_one_mile_l149_149384


namespace find_c_range_l149_149445

noncomputable def c_range (c : ℝ) : Prop :=
c > 0 ∧ c ≠ 1 ∧ (¬(0 < c ∧ c < 1 ∧ 0 < c ∧ c ≤ 1/2)) ∧ (0 < c ∧ c < 1 ∨ 0 < c ∧ c ≤ 1/2)

theorem find_c_range (c : ℝ) : c_range c ↔ (c ∈ set.Ioo (1/2 : ℝ) 1) :=
by
  sorry

end find_c_range_l149_149445


namespace marys_remaining_money_l149_149949

def drinks_cost (p : ℝ) := 4 * p
def medium_pizzas_cost (p : ℝ) := 3 * (3 * p)
def large_pizzas_cost (p : ℝ) := 2 * (5 * p)
def total_initial_money := 50

theorem marys_remaining_money (p : ℝ) : 
  total_initial_money - (drinks_cost p + medium_pizzas_cost p + large_pizzas_cost p) = 50 - 23 * p :=
by
  sorry

end marys_remaining_money_l149_149949


namespace yogurt_price_is_5_l149_149966

theorem yogurt_price_is_5
  (yogurt_pints : ℕ)
  (gum_packs : ℕ)
  (shrimp_trays : ℕ)
  (total_cost : ℝ)
  (shrimp_cost : ℝ)
  (gum_fraction : ℝ)
  (price_frozen_yogurt : ℝ) :
  yogurt_pints = 5 →
  gum_packs = 2 →
  shrimp_trays = 5 →
  total_cost = 55 →
  shrimp_cost = 5 →
  gum_fraction = 0.5 →
  5 * price_frozen_yogurt + 2 * (gum_fraction * price_frozen_yogurt) + 5 * shrimp_cost = total_cost →
  price_frozen_yogurt = 5 :=
by
  intro hp hg hs ht hc hf h_formula
  sorry

end yogurt_price_is_5_l149_149966


namespace div_condition_l149_149028

theorem div_condition (N : ℤ) : (∃ k : ℤ, N^2 - 71 = k * (7 * N + 55)) ↔ (N = 57 ∨ N = -8) := 
by
  sorry

end div_condition_l149_149028


namespace max_value_k_l149_149413

theorem max_value_k : ∃ k, (∀ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 →
  (a / (1 + 9 * b * c + k * (b - c) ^ 2) +
   b / (1 + 9 * c * a + k * (c - a) ^ 2) +
   c / (1 + 9 * a * b + k * (a - b) ^ 2)) ≥ 1 / 2) ∧ 
  (∀ k', k' > k → ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ 
    (a / (1 + 9 * b * c + k' * (b - c) ^ 2) +
     b / (1 + 9 * c * a + k' * (c - a) ^ 2) +
     c / (1 + 9 * a * b + k' * (a - b) ^ 2)) < 1 / 2)) :=
sorry

end max_value_k_l149_149413


namespace find_w_l149_149238

theorem find_w (p q r u v w : ℝ)
  (h₁ : (x : ℝ) → x^3 - 6 * x^2 + 11 * x + 10 = (x - p) * (x - q) * (x - r))
  (h₂ : (x : ℝ) → x^3 + u * x^2 + v * x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p)))
  (h₃ : p + q + r = 6) :
  w = 80 := sorry

end find_w_l149_149238


namespace ellipse_eccentricity_l149_149093

noncomputable def eccentricity_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → let M := (x, y) in
   let MP := (0, -y) and A1P := (x + a, 0) and A2P := (x - a, 0) in
   (y ^ 2) / ((x + a) * (a - x)) = 1 / 3) :
  eccentricity_of_ellipse a b = Real.sqrt (6) / 3 := sorry

end ellipse_eccentricity_l149_149093


namespace problem_solution_l149_149880

theorem problem_solution (m : ℝ) (h : (m - 2023)^2 + (2024 - m)^2 = 2025) :
  (m - 2023) * (2024 - m) = -1012 :=
sorry

end problem_solution_l149_149880


namespace cindy_total_travel_time_l149_149381

def speed_run := 3 -- Cindy's running speed in miles per hour
def speed_walk := 1 -- Cindy's walking speed in miles per hour
def distance_run := 0.5 -- Distance Cindy runs in miles
def distance_walk := 0.5 -- Distance Cindy walks in miles
def time_run := distance_run / speed_run * 60 -- Time to run half a mile in minutes
def time_walk := distance_walk / speed_walk * 60 -- Time to walk half a mile in minutes

theorem cindy_total_travel_time : time_run + time_walk = 40 := by
  -- skipping proof
  sorry

end cindy_total_travel_time_l149_149381


namespace non_intersecting_segments_pairing_l149_149816

theorem non_intersecting_segments_pairing (S : Finset (ℝ × ℝ)) (h_even : S.card % 2 = 0) 
  (h_no_collinear : ∀ (a b c : ℝ × ℝ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → ¬ collinear {a, b, c}) :
  ∃ (P : Finset (ℝ × ℝ) × Finset (ℝ × ℝ)), 
    (∀ p ∈ S, ∃ q ∈ S, (p, q) ∈ P) ∧ 
    (∀ q ∈ S, ∃ p ∈ S, (p, q) ∈ P) ∧ 
    (∀ (p1 p2 q1 q2 : ℝ × ℝ), (p1, q1) ∈ P → (p2, q2) ∈ P → (p1, q1) ≠ (p2, q2) → 
     ¬ intersects_segment ((p1, q1)) ((p2, q2))) :=
sorry

end non_intersecting_segments_pairing_l149_149816


namespace solve_equation_l149_149972

theorem solve_equation (x : ℝ) (h : x ≠ 0 ∧ x ≠ -1) : (x / (x + 1) = 1 + (1 / x)) ↔ (x = -1 / 2) :=
by
  sorry

end solve_equation_l149_149972


namespace general_term_formula_l149_149648

theorem general_term_formula (n : ℕ) :
  ∀ (S : ℕ → ℝ), (∀ k : ℕ, S k = 1 - 2^k) → 
  (∀ a : ℕ → ℝ, a 1 = (S 1) ∧ (∀ m : ℕ, m > 1 → a m = S m - S (m - 1)) → 
  a n = -2 ^ (n - 1)) :=
by
  intro S hS a ha
  sorry

end general_term_formula_l149_149648


namespace p_sufficient_but_not_necessary_q_l149_149060

theorem p_sufficient_but_not_necessary_q :
  ∀ x : ℝ, (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros x hx
  cases hx with h1 h2
  apply And.intro
  apply lt_of_lt_of_le h1
  linarith
  apply h2

end p_sufficient_but_not_necessary_q_l149_149060


namespace simplify_fraction_l149_149605

theorem simplify_fraction :
  (3 / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 72)) = (3 * Real.sqrt 2 / 44) := 
by
  have h1 : 2 * Real.sqrt 50 = 10 * Real.sqrt 2, sorry,
  have h2 : 3 * Real.sqrt 8 = 6 * Real.sqrt 2, sorry,
  have h3 : Real.sqrt 72 = 6 * Real.sqrt 2, sorry,
  have h4 : 2 * Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 72 = 22 * Real.sqrt 2,
  { rw [h1, h2, h3], sorry },
  have h5 : 3 / (22 * Real.sqrt 2) = 3 * Real.sqrt 2 / 44, sorry,
  rw [h4] at h5,
  exact h5

end simplify_fraction_l149_149605


namespace find_abc_l149_149839

open Real

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc (a b c : ℝ)
  (h₁ : a - b = 3)
  (h₂ : a^2 + b^2 = 39)
  (h₃ : a + b + c = 10) :
  abc_value a b c = -150 + 15 * Real.sqrt 69 :=
by
  sorry

end find_abc_l149_149839


namespace inequality_proof_l149_149429

variable (n : ℕ) (a : Fin n → ℝ) (b : ℝ)

theorem inequality_proof
  (h : b < (∑ i, a i)^2 / (n - 1) - ∑ i, (a i)^2) :
  ∀ i j : Fin n, i ≠ j → b < 2 * (a i) * (a j) :=
by
  sorry

end inequality_proof_l149_149429


namespace michael_truck_meet_times_l149_149201

-- Definitions for the conditions
def michael_speed : ℝ := 6
def truck_speed : ℝ := 10
def truck_stop_time : ℝ := 40
def pail_distance : ℝ := 200
def initial_distance : ℝ := 250

-- Theorem statement that Michael and the truck meet 7 times
theorem michael_truck_meet_times : 
  (number_of_meetings michael_speed truck_speed truck_stop_time pail_distance initial_distance) = 7 :=
sorry

end michael_truck_meet_times_l149_149201


namespace horiz_asymptote_of_ratio_l149_149870

noncomputable def z (x : ℝ) : ℝ :=
15 * x ^ 4 + 3 * x ^ 3 + 7 * x ^ 2 + 6 * x + 2 / (5 * x ^ 4 + x ^ 3 + 4 * x ^ 2 + 2 * x + 1)

theorem horiz_asymptote_of_ratio : (∃ L : ℝ, isHorizontalAsymptote L (z) := 3 :=
by {
  sorry
}

end horiz_asymptote_of_ratio_l149_149870


namespace find_range_of_m_l149_149833

theorem find_range_of_m (m : ℝ) : 
  (m < 1 ∨ m > -2) ∧ ¬(m < 1 ∧ m > -2) → (m ≤ -2 ∨ m ≥ 1) := 
by 
  intros h
  rcases h with ⟨hor, hand⟩
  by_cases p : m < 1
  case pos {
    have q : ¬ (m > -2), from hand (and.intro p q)
    linarith
  }
  case neg { 
    have q : m > -2, from hor.resolve_left p
    linarith
  }
  sorry

end find_range_of_m_l149_149833


namespace find_acute_angle_l149_149551

variables (x : ℝ)
noncomputable def a := (Real.sin x, 3 / 4 : ℝ)
noncomputable def b := (1 / 3 : ℝ, 1 / 2 * Real.cos x)

theorem find_acute_angle (h : ∃ k : ℝ, a = k • b) : x = π / 4 :=
sorry

end find_acute_angle_l149_149551


namespace probability_multiple_of_4_or_6_l149_149359

-- Define the range of numbers from 1 to 100
def range_1_to_100 := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

-- Define the condition for being a multiple of 4
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

-- Define the condition for being a multiple of 6
def is_multiple_of_6 (n : ℕ) : Prop := n % 6 = 0

-- Define the probability function
noncomputable def probability (event : ℕ → Prop) : ℚ :=
  ((finset.card (finset.filter event (finset.filter (λ n, n ∈ range_1_to_100) (finset.range 101)))) : ℚ)
  / (finset.card (finset.filter (λ n, n ∈ range_1_to_100) (finset.range 101))) 

-- Prove that the probability of a number being a multiple of 4, 6, or both is 33/100
theorem probability_multiple_of_4_or_6 :
  probability (λ n, is_multiple_of_4 n ∨ is_multiple_of_6 n) = 33 / 100 :=
by
  sorry

end probability_multiple_of_4_or_6_l149_149359


namespace range_of_a_l149_149426

def f (x : ℝ) : ℝ := 3 * x * |x|

theorem range_of_a : {a : ℝ | f (1 - a) + f (2 * a) < 0 } = {a : ℝ | a < -1} :=
by
  sorry

end range_of_a_l149_149426


namespace area_of_triangle_APQ_l149_149241

noncomputable def point := ℝ × ℝ
noncomputable def line := (ℝ, ℝ) -- slope and y-intercept

def intersection (l1 l2 : line) : point :=
  let (m1, b1) := l1
  let (m2, b2) := l2
  ( (b2 - b1) / (m1 - m2), ((m1 * b2) - (m2 * b1)) / (m1 - m2) )

def area_of_triangle (A P Q : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := P
  let (x3, y3) := Q
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_of_triangle_APQ : 
  ∃ (P Q : point), 
    let A := (8, 6)
    in let Px := P.fst
    in let Qx := Q.fst
    in let Py := 0 
    in forall (m₁ m₂ : ℝ), m₁ * m₂ = -1 -> 
      Px + Qx = 0 -> 
      intersection (m₁, A.snd - m₁ * A.fst) (m₂, A.snd - m₂ * A.fst) = A ->
      area_of_triangle A P Q = 48 :=
sorry

end area_of_triangle_APQ_l149_149241


namespace angle_BMC_is_114_degrees_l149_149896

theorem angle_BMC_is_114_degrees 
  (A B C M : Type) 
  [IsoscelesTriangle ABC AB BC] 
  (h1 : ∠ ABC = 108) 
  (h2 : ∠ BAM = 18) 
  (h3 : ∠ BMA = 30) : 
  ∠ BMC = 114 :=
sorry

end angle_BMC_is_114_degrees_l149_149896


namespace number_of_positive_integers_l149_149801

theorem number_of_positive_integers (count_x : ℕ) :
  (∀ x : ℕ, (50 ≤ (x + 3) ^ 2 ∧ (x + 3) ^ 2 ≤ 100) ↔ (x = 5 ∨ x = 6 ∨ x = 7)) →
  count_x = 3 :=
by
  intro h
  have : count_x = (finset.filter (λ x, 50 ≤ (x + 3) ^ 2 ∧ (x + 3) ^ 2 ≤ 100) (finset.range 8)).card,
  {
    sorry
  }
  rw [finset.filter_congr h] at this,
  exact this

end number_of_positive_integers_l149_149801


namespace problem_l149_149762

def otimes (x y : ℝ) : ℝ := x^3 + y - 2 * x

theorem problem (k : ℝ) : otimes k (otimes k k) = 2 * k^3 - 3 * k :=
by
  sorry

end problem_l149_149762


namespace bus_problem_l149_149329

theorem bus_problem
  (initial_children : ℕ := 18)
  (final_total_children : ℕ := 25) :
  final_total_children - initial_children = 7 :=
by
  sorry

end bus_problem_l149_149329


namespace measure_diagonal_without_pythagorean_theorem_l149_149589

variables (a b c : ℝ)

-- Definition of the function to measure the diagonal distance
def diagonal_method (a b c : ℝ) : ℝ :=
  -- by calculating the hypotenuse scaled by sqrt(3), we ignore using the Pythagorean theorem directly
  sorry

-- Calculate distance by arranging bricks
theorem measure_diagonal_without_pythagorean_theorem (distance_extreme_corners : ℝ) :
  distance_extreme_corners = (diagonal_method a b c) :=
  sorry

end measure_diagonal_without_pythagorean_theorem_l149_149589


namespace graph_transformation_correct_l149_149125

def initial_function (x : ℝ) : ℝ := (1 / 2) * Real.cos x

def transformed_function (x : ℝ) : ℝ := initial_function (2 * x)

def final_function (x : ℝ) : ℝ := transformed_function (x - (π / 4))

theorem graph_transformation_correct :
  final_function x = (1 / 2) * Real.cos (2 * x - π / 2) := sorry

end graph_transformation_correct_l149_149125


namespace p_sufficient_not_necessary_q_l149_149049

theorem p_sufficient_not_necessary_q (x : ℝ) :
  (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros h
  cases h with h1 h2
  split
  case left => linarith
  case right => linarith

end p_sufficient_not_necessary_q_l149_149049


namespace quadratic_function_expression_a_value_range_l149_149827

theorem quadratic_function_expression
  (a b c : ℝ)
  (h1 : a + b + c = -2) 
  (h2 : 9 * a + 3 * b + c = -6) 
  (h3 : b^2 - 4 * a * (c + 6 * a) = 0) : 
  (a = -1 / 5 ∧ b = -6 / 5 ∧ c = -3 / 5) → 
  ∃ y : ℝ, y = -1/5 * (x^2) - 6/5 * x - 3/5 :=
by
  sorry

theorem a_value_range (a b c : ℝ) 
  (h1 : a + b + c = -2) 
  (h2 : 9 * a + 3 * b + c = -6) 
  (h4 : b = -2 - 4 * a) 
  (h5 : c = 3 * a)
  (h6 : b^2 - 4 * a * c > 0) : 
  ( a > -2 + real.sqrt 3 ∨ a < -2 - real.sqrt 3) :=
by
  sorry

end quadratic_function_expression_a_value_range_l149_149827


namespace find_e_m_l149_149101

variable {R : Type} [Field R]

def matrix_B (e : R) : Matrix (Fin 2) (Fin 2) R :=
  !![3, 4; 6, e]

theorem find_e_m (e m : R) (hB_inv : (matrix_B e)⁻¹ = m • (matrix_B e)) :
  e = -3 ∧ m = (1 / 11) := by
  sorry

end find_e_m_l149_149101


namespace seventy_fifth_elem_in_s_l149_149942

-- Define the set s
def s : Set ℕ := {x | ∃ n : ℕ, x = 8 * n + 5}

-- State the main theorem
theorem seventy_fifth_elem_in_s : (∃ n : ℕ, n = 74 ∧ (8 * n + 5) = 597) :=
by
  -- The proof is skipped using sorry
  sorry

end seventy_fifth_elem_in_s_l149_149942


namespace find_a_l149_149992

noncomputable def a := 1/2

theorem find_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 1 - a^2 = 3/4) : a = 1/2 :=
sorry

end find_a_l149_149992


namespace triangle_is_right_triangle_l149_149527

theorem triangle_is_right_triangle (BC AC AB : ℝ) (h1 : BC = 6) (h2 : AC = 8) (h3 : AB = 10) :
  (BC^2 + AC^2 = AB^2) → right_triangle ABC := by
  sorry

end triangle_is_right_triangle_l149_149527


namespace num_distinct_acute_triangles_with_perimeter_24_l149_149113

def is_valid_acute_triangle (a b c : ℕ) : Prop :=
  a + b + c = 24 ∧
  a ≥ b ∧ b ≥ c ∧
  a^2 < b^2 + c^2 ∧
  a < b + c

theorem num_distinct_acute_triangles_with_perimeter_24 :
  { (a, b, c) | is_valid_acute_triangle a b c } = { (8, 8, 8), (9, 8, 7), (9, 9, 6), (10, 9, 5), (10, 10, 4), (11, 11, 2) } ∧
  (|{ (a, b, c) | is_valid_acute_triangle a b c}| = 6) :=
sorry

end num_distinct_acute_triangles_with_perimeter_24_l149_149113


namespace original_number_solution_l149_149644

theorem original_number_solution (x : ℝ) (h : x^2 + 45 = 100) : x = Real.sqrt 55 ∨ x = -Real.sqrt 55 :=
by
  sorry

end original_number_solution_l149_149644


namespace p_implies_q_and_not_q_implies_p_l149_149054

variable {x : ℝ}

def p : Prop := 0 < x ∧ x < 2
def q : Prop := -1 < x ∧ x < 3

theorem p_implies_q_and_not_q_implies_p : (p → q) ∧ ¬ (q → p) := 
by
  sorry

end p_implies_q_and_not_q_implies_p_l149_149054


namespace limit_sequence_l149_149700

noncomputable def sequence (n : ℕ) : ℝ :=
  (sqrt ((n^5 + 1) * (n^2 - 1)) - n * sqrt (n * (n^4 + 1))) / n

theorem limit_sequence :
  tendsto sequence at_top (nhds (-∞)) :=
begin
  sorry
end

end limit_sequence_l149_149700


namespace parametric_vector_equation_l149_149280

open Real

noncomputable def vec (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

noncomputable def proj (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let norm_sq : ℝ := u.1 * u.1 + u.2 * u.2 + u.3 * u.3
  let dot := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  ((dot / norm_sq * u.1), (dot / norm_sq * u.2), (dot / norm_sq * u.3))

theorem parametric_vector_equation (x y z : ℝ) (s t : ℝ) :
  proj (3, -2, 6) (x, y, z) = (3, -2, 6) →
  ∃ s t, vec x y z = (s, t, 43 / 6 * s - 1 / 3 * t + 49 / 6) :=
by
  intro h
  sorry

end parametric_vector_equation_l149_149280


namespace abigail_writing_time_l149_149364

def total_additional_time (words_needed : ℕ) (words_per_half_hour : ℕ) (words_already_written : ℕ) (proofreading_time : ℕ) : ℕ :=
  let remaining_words := words_needed - words_already_written
  let half_hour_blocks := (remaining_words + words_per_half_hour - 1) / words_per_half_hour -- ceil(remaining_words / words_per_half_hour)
  let writing_time := half_hour_blocks * 30
  writing_time + proofreading_time

theorem abigail_writing_time :
  total_additional_time 1500 250 200 45 = 225 :=
by {
  -- Adding the proof in Lean:
  -- fail to show you the detailed steps, hence added sorry
  sorry
}

end abigail_writing_time_l149_149364


namespace simplify_and_evaluate_division_l149_149604

theorem simplify_and_evaluate_division (m : ℕ) (h : m = 10) : 
  (1 - (m / (m + 2))) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 1 / 4 :=
by sorry

end simplify_and_evaluate_division_l149_149604


namespace number_of_solutions_at_most_2_l149_149177

-- Define the data for the problem
structure Circle :=
(center : Point)
(radius : ℝ)

structure Point :=
(x : ℝ)
(y : ℝ)

def dist (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

def PowerOfPoint (P : Point) (C : Circle) : ℝ :=
  dist P C.center - C.radius^2

theorem number_of_solutions_at_most_2
  (P : Point) 
  (C : Circle)
  (h1 : dist P C.center > C.radius^2) : 
  ∃ Q R : Point, Q ≠ R ∧ OnCircle Q C ∧ OnCircle R C ∧ Collinear P Q R ∧ Midpoint Q P R :=
  sorry

end number_of_solutions_at_most_2_l149_149177


namespace ratio_of_b_to_a_l149_149029

noncomputable def ratio_b_to_a (a b : ℝ) : ℝ :=
  b / a

theorem ratio_of_b_to_a {a b : ℝ} (h1: a > 0) (h2: b > 0) 
  (h3: a + sqrt 3 * b = 2) : ratio_b_to_a a b = 2 + sqrt 3 :=
by
  sorry

end ratio_of_b_to_a_l149_149029


namespace determine_a_l149_149821

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then x^3 + 1 else x^2 - a * x

theorem determine_a (a : ℝ) : 
  f (f 0 a) a = -2 → a = 3 :=
by
  sorry

end determine_a_l149_149821


namespace number_is_93_75_l149_149696

theorem number_is_93_75 (x : ℝ) (h : 0.16 * (0.40 * x) = 6) : x = 93.75 :=
by
  -- The proof is omitted.
  sorry

end number_is_93_75_l149_149696


namespace difference_of_values_l149_149163

theorem difference_of_values (num : Nat) : 
  (num = 96348621) →
  let face_value := 8
  let local_value := 8 * 10000
  local_value - face_value = 79992 := 
by
  intros h_eq
  have face_value := 8
  have local_value := 8 * 10000
  sorry

end difference_of_values_l149_149163


namespace squirrel_rise_per_circuit_l149_149739

theorem squirrel_rise_per_circuit
  (h_post_height : ℕ := 12)
  (h_circumference : ℕ := 3)
  (h_travel_distance : ℕ := 9) :
  (h_post_height / (h_travel_distance / h_circumference) = 4) :=
  sorry

end squirrel_rise_per_circuit_l149_149739


namespace product_formula_l149_149375

theorem product_formula :
  (∏ k in Finset.range' 2 100, (1 - (1 / k))) = 1 / 100 :=
by
  sorry

end product_formula_l149_149375


namespace distance_between_vertices_l149_149924

theorem distance_between_vertices :
  let C := (2, 1)
      D := (-3, 4)
  in real.sqrt ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2) = real.sqrt 34 :=
by
  let C := (2, 1)
  let D := (-3, 4)
  sorry

end distance_between_vertices_l149_149924


namespace prime_pairs_l149_149767

open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, 2 ≤ m → m ≤ n / 2 → n % m ≠ 0

theorem prime_pairs :
  ∀ (p q : ℕ), is_prime p → is_prime q →
  1 < p → p < 100 →
  1 < q → q < 100 →
  is_prime (p + 6) →
  is_prime (p + 10) →
  is_prime (q + 4) →
  is_prime (q + 10) →
  is_prime (p + q + 1) →
  (p, q) = (7, 3) ∨ (p, q) = (13, 3) ∨ (p, q) = (37, 3) ∨ (p, q) = (97, 3) :=
by
  sorry

end prime_pairs_l149_149767


namespace sequence_ratio_l149_149085

theorem sequence_ratio :
  ∀ {a : ℕ → ℝ} (h₁ : a 1 = 1/2) (h₂ : ∀ n, a n = (a (n + 1)) * (a (n + 1))),
  (a 200 / a 300) = (301 / 201) :=
by
  sorry

end sequence_ratio_l149_149085


namespace reflection_matrix_over_vector_l149_149784

theorem reflection_matrix_over_vector : 
  ∃ M : Matrix (Fin 2) (Fin 2) ℚ, 
    M.mul_vec ⟨[4, 1], by decide⟩ = ⟨[4, 1], by decide⟩ ∧
    M = !![ 15/17, 8/17; 8/17, -15/17 ]
    sorry

end reflection_matrix_over_vector_l149_149784


namespace find_prime_solution_l149_149407

theorem find_prime_solution :
  ∀ p x y : ℕ, Prime p → x > 0 → y > 0 →
    (p ^ x = y ^ 3 + 1) ↔ 
    ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) := 
by
  sorry

end find_prime_solution_l149_149407


namespace range_of_a_l149_149502

theorem range_of_a (a : ℝ) (h : ∅ ⊂ {x : ℝ | x^2 ≤ a }) : a ∈ set.Ici 0 :=
by
  sorry

end range_of_a_l149_149502


namespace smallest_other_number_l149_149634

theorem smallest_other_number (x : ℕ)  (h_pos : 0 < x) (n : ℕ)
  (h_gcd : Nat.gcd 60 n = x + 3)
  (h_lcm : Nat.lcm 60 n = x * (x + 3)) :
  n = 45 :=
sorry

end smallest_other_number_l149_149634


namespace find_Allyson_age_l149_149299

variable (Hiram Allyson : ℕ)

theorem find_Allyson_age (h : Hiram = 40)
  (condition : Hiram + 12 = 2 * Allyson - 4) : Allyson = 28 := by
  sorry

end find_Allyson_age_l149_149299


namespace sum_log2_a_n_first_10_l149_149843

theorem sum_log2_a_n_first_10 (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = 2^n - 1) → 
  (∀ n, n ≥ 1 → a n = S n - S (n-1)) →
  (∑ i in finset.range 10, (Int.log2 (a (i+1)))) = 45 :=
by
  sorry

end sum_log2_a_n_first_10_l149_149843


namespace product_fib_l149_149935

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n + 1) + fib n

lemma fib_start : fib 1 = 1 := rfl
lemma fib_start2 : fib 2 = 1 := rfl

theorem product_fib (a b : ℕ) (h : (a, b) = (101, 102)) :
  (∏ k in finset.range (99 + 1) \ finset.range 2, (fib (k + 2) / fib (k + 1) - fib (k + 2) / fib (k + 3))) = (fib 101 / fib 102) := 
sorry

end product_fib_l149_149935


namespace exist_n_consecutive_not_perfect_power_l149_149557

theorem exist_n_consecutive_not_perfect_power (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, ∀ k : ℕ, k < n → ¬ (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (m + k) = a ^ b) :=
sorry

end exist_n_consecutive_not_perfect_power_l149_149557


namespace derivative_at_neg_one_l149_149624

def f (x : ℝ) : ℝ := List.prod (List.map (λ k => (x^3 + k)) (List.range' 1 100))

theorem derivative_at_neg_one : deriv f (-1) = 3 * Nat.factorial 99 := by
  sorry

end derivative_at_neg_one_l149_149624


namespace evaluate_series_l149_149023

noncomputable def closestInt (n : ℕ) : ℕ := ⌊Real.sqrt n + 0.5⌋₊

theorem evaluate_series :
  (∑' n : ℕ, (3^(closestInt n) + 3^(-(closestInt n))) / 3^n) = 3 :=
by
  sorry

end evaluate_series_l149_149023


namespace decorations_cost_correct_l149_149579

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end decorations_cost_correct_l149_149579


namespace min_value_reciprocal_sum_l149_149853

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) : 
  (1/m + 1/n) = 4 :=
by
  sorry

end min_value_reciprocal_sum_l149_149853


namespace simplify_fraction_l149_149970

theorem simplify_fraction :
  (30 / 35) * (21 / 45) * (70 / 63) - (2 / 3) = - (8 / 15) :=
by
  sorry

end simplify_fraction_l149_149970


namespace count_positive_integers_proof_l149_149799

noncomputable def countPositiveIntegersInRange : ℕ :=
  let boundedIntegers := {x : ℕ | 2 ≤ x ∧ x ≤ 6}
  let validIntegers := {x : boundedIntegers | 50 < (x + 3) * (x + 3) ∧ (x + 3) * (x + 3) < 100}
  validIntegers.card

theorem count_positive_integers_proof : countPositiveIntegersInRange = 3 :=
sorry

end count_positive_integers_proof_l149_149799


namespace zero_point_theorem_l149_149565

theorem zero_point_theorem (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) (h_sign : f a * f b < 0) :
  ∃ c, a < c ∧ c < b ∧ f c = 0 :=
begin
  sorry
end

end zero_point_theorem_l149_149565


namespace impossible_to_determine_total_games_l149_149283

-- Define the conditions as hypotheses
def total_games : ℕ -- Assume the total number of football games
def night_games : ℕ := 4 -- 4 games are played at night
def missed_games : ℕ := 4 -- Keith missed 4 games
def attended_games : ℕ := 4 -- Keith attended 4 games

-- Prove that with given conditions, determining total_games is impossible
theorem impossible_to_determine_total_games (total_games night_games missed_games attended_games : ℕ) :
  night_games = 4 ∧ missed_games = 4 ∧ attended_games = 4 → (∃ total_games, true) :=
by {
  intros,
  -- Since total_games is assumed but cannot be determined uniquely,
  -- we conclude with a statement that we cannot determine the exact number.
  contradiction, -- We express that it's impossible to determine with the given information.
}

end impossible_to_determine_total_games_l149_149283


namespace find_smallest_number_l149_149294

noncomputable def smallest_number := 3075837206
def numbers := [11, 17, 19, 23, 29, 37, 41]

theorem find_smallest_number :
  let lcm := numbers.foldl Nat.lcm 1 in 
  (smallest_number - 27) % lcm = 0 := 
by
  sorry

end find_smallest_number_l149_149294


namespace range_of_m_l149_149446

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 then 0 else if x > 0 then 2^x - 1 else - (2^(-x) - 1)

def g (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem range_of_m :
  (∀ x1 ∈ Icc (-2 : ℝ) 2, ∃ x2 ∈ Icc (-2 : ℝ) 2, g x2 m = f x1) ↔
  -5 ≤ m ∧ m ≤ -2 :=
by
  sorry

end range_of_m_l149_149446


namespace p_implies_q_and_not_q_implies_p_l149_149056

variable {x : ℝ}

def p : Prop := 0 < x ∧ x < 2
def q : Prop := -1 < x ∧ x < 3

theorem p_implies_q_and_not_q_implies_p : (p → q) ∧ ¬ (q → p) := 
by
  sorry

end p_implies_q_and_not_q_implies_p_l149_149056


namespace inequality_proof_l149_149588

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
by 
  sorry

end inequality_proof_l149_149588


namespace number_of_footballs_l149_149278

theorem number_of_footballs (x y : ℕ) (h1 : x + y = 20) (h2 : 6 * x + 3 * y = 96) : x = 12 :=
by {
  sorry
}

end number_of_footballs_l149_149278


namespace parabola_directrix_tangent_to_circle_l149_149473

theorem parabola_directrix_tangent_to_circle (p : ℝ) (h : p > 0) : 
  let center := (-3, 0),
      radius := 1,
      directrix := -p / 2 
  in
  (dist center (directrix, 0) = radius) ↔ (p = 4 ∨ p = 8) := by
  sorry

end parabola_directrix_tangent_to_circle_l149_149473


namespace mod_exp_l149_149017

theorem mod_exp (a : ℕ) (h : a = 3^1999) : (a % 13) = 3 := by
  have h1 : 3^3 % 13 = 1 := by norm_num
  have h2 : 3^1999 = 3^(3 * 666 + 1) := by norm_num
  have h3 : 3^(3 * 666 + 1) % 13 = ((3^3)^666 * 3) % 13 := by rw ←pow_add
  rw [pow_mul, h1, one_pow] at h3
  exact h3

end mod_exp_l149_149017


namespace wall_length_l149_149311

theorem wall_length (mirror_side length width : ℝ) (h1 : mirror_side = 21) (h2 : width = 28) 
  (h3 : 2 * mirror_side^2 = width * length) : length = 31.5 := by
  sorry

end wall_length_l149_149311


namespace find_m_value_l149_149849

theorem find_m_value (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f(x) = m - |x - 3|) :
  (∀ x, (2 < x ∧ x < 4) ↔ f(x) > 2) → m = 3 :=
sorry

end find_m_value_l149_149849


namespace NumberOfValidTenDigitNumbers_l149_149769

theorem NumberOfValidTenDigitNumbers : 
  let digits : Finset ℕ := Finset.range 10       -- The set of digits {0, 1, 2, ..., 9}
  let valid_numbers := {p : Finset (Fin 10) // 
    ∀ x ∈ p.1.filter (λ d, d ≠ 9), ∃ y ∈ p.1, x < y} -- Sets of digits fulfilling the conditions.
  in valid_numbers.card = 256 := 
by sorry

end NumberOfValidTenDigitNumbers_l149_149769


namespace steve_take_home_pay_l149_149614

def annual_salary : ℕ := 40000
def tax_percentage : ℝ := 0.20
def healthcare_percentage : ℝ := 0.10
def union_dues : ℕ := 800

theorem steve_take_home_pay :
  annual_salary - (annual_salary * tax_percentage).to_nat - (annual_salary * healthcare_percentage).to_nat - union_dues = 27200 :=
by
  sorry

end steve_take_home_pay_l149_149614


namespace vitamin_supplement_problem_l149_149571

theorem vitamin_supplement_problem :
  let packA := 7
  let packD := 17
  (∀ n : ℕ, n ≠ 0 → (packA * n = packD * n)) → n = 119 :=
by
  sorry

end vitamin_supplement_problem_l149_149571


namespace integer_solutions_exist_l149_149590

theorem integer_solutions_exist (R₀ : ℝ) : 
  ∃ (x₁ x₂ x₃ : ℤ), (x₁^2 + x₂^2 + x₃^2 = x₁ * x₂ * x₃) ∧ (R₀ < x₁) ∧ (R₀ < x₂) ∧ (R₀ < x₃) := 
sorry

end integer_solutions_exist_l149_149590


namespace acute_angle_ratio_l149_149244

theorem acute_angle_ratio (α : ℝ) (r R : ℝ) (h1 : 0 < α) (h2 : α < π / 2) 
    (h3 : R = r / (2 * sqrt 2 * sin (π / 4 - α / 2) * sin (α / 2))) :
    (1 / h3) = (2 * sqrt 2 * sin (π / 4 - α / 2) * sin (α / 2)) ∧ 
    (argmax (λ α:ℝ, 2 * sqrt 2 * sin (π / 4 - α / 2) * sin (α / 2)) (Ioo 0 (π / 2)) = π / 4) := 
sorry

end acute_angle_ratio_l149_149244


namespace unique_solution_of_system_l149_149973

theorem unique_solution_of_system (n k m : ℕ) (hnk : n + k = Nat.gcd n k ^ 2) (hkm : k + m = Nat.gcd k m ^ 2) (hmn : m + n = Nat.gcd m n ^ 2) : 
  n = 2 ∧ k = 2 ∧ m = 2 :=
by
  sorry

end unique_solution_of_system_l149_149973


namespace cola_cost_l149_149195

theorem cola_cost (h c : ℝ) (h1 : 3 * h + 2 * c = 360) (h2 : 2 * h + 3 * c = 390) : c = 90 :=
by
  sorry

end cola_cost_l149_149195


namespace factorize_expression_l149_149775

theorem factorize_expression (a b : ℝ) : a^2 - b^2 + 4a + 2b + 3 = (a + b + 1) * (a - b + 3) :=
by
  sorry

end factorize_expression_l149_149775


namespace waiter_total_customers_l149_149362

theorem waiter_total_customers (n_tables : ℕ) (n_women : ℕ) (n_men : ℕ) (total_customers : ℕ) 
  (h1 : n_tables = 9) (h2 : n_women = 7) (h3 : n_men = 3) (h4 : total_customers = n_tables * (n_women + n_men)) : 
  total_customers = 90 :=
by 
  rw [h1, h2, h3] at h4
  exact h4
  sorry

end waiter_total_customers_l149_149362


namespace smallest_number_of_students_l149_149346

-- Define the structure of the problem
def unique_row_configurations (n : ℕ) : Prop :=
  (∀ k : ℕ, k ∣ n → k < 10) → ∃ divs : Finset ℕ, divs.card = 9 ∧ ∀ d ∈ divs, d ∣ n ∧ (∀ d' ∈ divs, d ≠ d') 

-- The main statement to be proven in Lean 4
theorem smallest_number_of_students : ∃ n : ℕ, unique_row_configurations n ∧ n = 36 :=
by
  sorry

end smallest_number_of_students_l149_149346


namespace equidistant_points_count_l149_149393

theorem equidistant_points_count (r : ℝ) (O : Point) (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = r^2)
  (tangent1_eq : ∀ (x : ℝ), y = r) (tangent2_eq : ∀ (x : ℝ), y = -2r) : 
  ∃ p1 p2 : Point, (dist p1 circle_layer = dist p1 tangent_lines) ∧ 
                 (dist p2 circle_layer = dist p2 tangent_lines) ∧ 
                 p1 ≠ p2 := 
sorry

end equidistant_points_count_l149_149393


namespace non_neg_int_solutions_eq_10_l149_149861

theorem non_neg_int_solutions_eq_10 :
  ∃ n : ℕ, n = 55 ∧
  (∃ (x y z : ℕ), x + y + z = 10) :=
by
  sorry

end non_neg_int_solutions_eq_10_l149_149861


namespace square_99_is_white_l149_149645

def grid_properties (grid : Fin 9 × Fin 9 → Prop) : Prop :=
  grid (4, 4) = true ∧  -- Square 44 is black
  grid (4, 9) = true ∧  -- Square 49 is black
  (∀ (i j : Fin 9), grid (i, j) → (∃! k, (k ≠ i ∨ k ≠ j) ∧ (grid (i.succ, j) = true ∨ grid (i.pred, j) = true ∨ grid (i, j.succ) = true ∨ grid (i, j.pred) = true) → Prop)) ∧  -- Each black square shares edge with at most one other black square
  (∀ (i j : Fin 9), ¬ grid (i, j) → (∃! k, (k ≠ i ∨ k ≠ j) ∧ (grid (i.succ, j) = false ∨ grid (i.pred, j) = false ∨ grid (i, j.succ) = false ∨ grid (i, j.pred) = false) → Prop))  -- Each white square shares edge with at most one other white square

theorem square_99_is_white (grid : Fin 9 × Fin 9 → Prop) (h : grid_properties grid) :
  ¬ grid (9, 9) :=
sorry

end square_99_is_white_l149_149645


namespace slope_symmetric_line_l149_149831

  theorem slope_symmetric_line {l1 l2 : ℝ → ℝ} 
     (hl1 : ∀ x, l1 x = 2 * x + 3)
     (hl2_sym : ∀ x, l2 x = 2 * x + 3 -> l2 (-x) = -2 * x - 3) :
     ∀ x, l2 x = -2 * x + 3 :=
  sorry
  
end slope_symmetric_line_l149_149831


namespace hybrids_with_full_headlights_l149_149142

theorem hybrids_with_full_headlights
  (total_cars : ℕ) (hybrid_percentage : ℕ) (one_headlight_percentage : ℕ) :
  total_cars = 600 → hybrid_percentage = 60 → one_headlight_percentage = 40 →
  let total_hybrids := (hybrid_percentage * total_cars) / 100 in
  let one_headlight_hybrids := (one_headlight_percentage * total_hybrids) / 100 in
  let full_headlight_hybrids := total_hybrids - one_headlight_hybrids in
  full_headlight_hybrids = 216 :=
by
  intros h1 h2 h3
  sorry

end hybrids_with_full_headlights_l149_149142


namespace find_Allyson_age_l149_149300

variable (Hiram Allyson : ℕ)

theorem find_Allyson_age (h : Hiram = 40)
  (condition : Hiram + 12 = 2 * Allyson - 4) : Allyson = 28 := by
  sorry

end find_Allyson_age_l149_149300


namespace correct_statement_estimation_l149_149682

theorem correct_statement_estimation 
  (A : Prop := "The result of the sample is the result of the population")
  (B : Prop := "The larger the sample size, the more accurate the estimate")
  (C : Prop := "The standard deviation of the sample can approximately reflect the average state of the population")
  (D : Prop := "The larger the variance of the data, the more stable the data") :
  B :=
begin
  sorry -- proof placeholder
end

end correct_statement_estimation_l149_149682


namespace statement_A_statement_B_statement_C_statement_D_l149_149685

-- Statement A
def z1 := complex.I ^ 3
theorem statement_A : complex.conj z1 = complex.I := sorry

-- Statement B
def z2 := 1 + (1 / complex.I)
theorem statement_B : complex.imaginaryPart z2 ≠ -complex.I := sorry

-- Statement C
variables (a : ℝ)
def z3 := a + a * complex.I
theorem statement_C : ¬(z3.im = 0) ∨ ¬(z3.re = 0) := sorry

-- Statement D
variable (z : ℂ)
theorem statement_D (hz : (1 / z).im = 0) : z.im = 0 := sorry

end statement_A_statement_B_statement_C_statement_D_l149_149685


namespace right_triangle_of_trig_identity_l149_149506

theorem right_triangle_of_trig_identity
  {A B C : ℝ}
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hABC : A + B + C = π)
  (h : sin (A - B) = 1 + 2 * cos (B + C) * sin (A + C)) :
  C = π / 2 :=
by
  sorry

end right_triangle_of_trig_identity_l149_149506


namespace function_domain_l149_149628

noncomputable def myFunctionDomain : Set ℝ := { x | x ≥ 3 / 2 ∧ x ≠ 3 }

theorem function_domain :
  (∀ x : ℝ, (∃ y : ℝ, y = sqrt (2 * x - 3) + 1 / (x - 3)) ↔ (x ∈ ([3 / 2, 3) ∪ (3, +∞)))) :=
by
  sorry

end function_domain_l149_149628


namespace valid_outfits_count_l149_149308

theorem valid_outfits_count
    (red_shirts : ℕ) (green_shirts : ℕ) (blue_shirts : ℕ)
    (red_pants : ℕ) (green_pants : ℕ) (blue_pants : ℕ)
    (red_hats : ℕ) (green_hats : ℕ) (blue_hats : ℕ)
    (distinct_items : ∀ items (x y : items), x ≠ y) :
    red_shirts = 6 →
    green_shirts = 6 →
    blue_shirts = 6 →
    red_pants = 7 →
    green_pants = 7 →
    blue_pants = 7 →
    red_hats = 9 →
    green_hats = 9 →
    blue_hats = 9 →
    (Σ x in [1, 2, 3].sum, 
      (if x = 1 then (red_shirts * (green_pants + blue_pants) * (green_hats + blue_hats)) else if x = 2 then (green_shirts * (red_pants + blue_pants) * (red_hats + blue_hats)) else (blue_shirts * (red_pants + green_pants) * (red_hats + green_hats))) = 4536 :=
by
  intros
  sorry

end valid_outfits_count_l149_149308


namespace min_dist_l149_149558

noncomputable def z_min_distance (z : ℂ) : ℝ :=
  if |z - 3 * complex.I| + |z - (4 : ℂ)| = 5 then
    real.min (|z|)
  else
    0

theorem min_dist (z : ℂ) (hz : |z - 3 * complex.I| + |z - (4 : ℂ)| = 5) : z_min_distance z = 12 / 5 := by
  sorry

end min_dist_l149_149558


namespace product_of_digits_of_non_divisible_by_4_l149_149954

-- Definitions based on given conditions
def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def not_divisible_by_4 (n : ℕ) : Prop :=
  ¬ is_divisible_by_4 n

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

-- Main statement
theorem product_of_digits_of_non_divisible_by_4 :
  ∃ n ∈ ({4628, 4638, 4648, 4658, 4662} : set ℕ), not_divisible_by_4 (last_two_digits n) ∧
  let tens_digit := (n % 100) / 10
      units_digit := n % 10
  in tens_digit * units_digit = 24 :=
by
  sorry

end product_of_digits_of_non_divisible_by_4_l149_149954


namespace quadratic_no_real_roots_l149_149075

open Real

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (hpq : p ≠ q)
  (hpositive_p : 0 < p)
  (hpositive_q : 0 < q)
  (hpositive_a : 0 < a)
  (hpositive_b : 0 < b)
  (hpositive_c : 0 < c)
  (h_geo_sequence : a^2 = p * q)
  (h_ari_sequence : b + c = p + q) :
  (a^2 - b * c) < 0 :=
by
  sorry

end quadratic_no_real_roots_l149_149075


namespace expected_value_of_win_l149_149343

noncomputable def expected_value_win : ℝ :=
  let probabilities := (List.range' 1 8).map (λ n => (1/8 : ℝ))
  let winnings := (List.range' 1 (8+1)).map (λ n => (8 - 2 * n : ℝ))
  (List.zip probabilities winnings).sum (λ x => x.1 * x.2)

theorem expected_value_of_win : expected_value_win = -1 := sorry

end expected_value_of_win_l149_149343


namespace find_a_given_coefficient_l149_149815

-- Define the conditions and problem statement
theorem find_a_given_coefficient 
  (a : ℝ) -- \( a \) is a real number
  (h1 : a > 0)  -- \( a > 0 \)
  (h2 : (∑ k in Finset.range 10, Nat.choose 9 k * (x ^ (9 - 2 * k)) * (a ^ k) = 144)) -- Coefficient of \( x^5 \) is 144
  : a = 2 :=
sorry  -- Proof to be filled in later

end find_a_given_coefficient_l149_149815


namespace marks_difference_l149_149654

variable (P C M : ℕ)

-- Conditions
def total_marks_more_than_physics := P + C + M > P
def average_chemistry_mathematics := (C + M) / 2 = 65

-- Proof Statement
theorem marks_difference (h1 : total_marks_more_than_physics P C M) (h2 : average_chemistry_mathematics C M) : 
  P + C + M = P + 130 := by
  sorry

end marks_difference_l149_149654


namespace cube_bug_probability_l149_149758

theorem cube_bug_probability :
  ∃ n : ℕ, (∃ p : ℚ, p = 547/2187) ∧ (p = n/6561) ∧ n = 1641 :=
by
  sorry

end cube_bug_probability_l149_149758


namespace bumper_car_rides_correct_l149_149373

def tickets_per_ride : ℕ := 7
def total_tickets : ℕ := 63
def ferris_wheel_rides : ℕ := 5

def tickets_for_bumper_cars : ℕ :=
  total_tickets - ferris_wheel_rides * tickets_per_ride

def bumper_car_rides : ℕ :=
  tickets_for_bumper_cars / tickets_per_ride

theorem bumper_car_rides_correct : bumper_car_rides = 4 :=
by
  sorry

end bumper_car_rides_correct_l149_149373


namespace area_of_curvilinear_trapezoid_steps_l149_149410

theorem area_of_curvilinear_trapezoid_steps (steps : List String) :
  (steps = ["division", "approximation", "summation", "taking the limit"]) :=
sorry

end area_of_curvilinear_trapezoid_steps_l149_149410


namespace excess_calories_l149_149536

-- Conditions
def calories_from_cheezits (bags: ℕ) (ounces_per_bag: ℕ) (calories_per_ounce: ℕ) : ℕ :=
  bags * ounces_per_bag * calories_per_ounce

def calories_from_chocolate_bars (bars: ℕ) (calories_per_bar: ℕ) : ℕ :=
  bars * calories_per_bar

def calories_from_popcorn (calories: ℕ) : ℕ :=
  calories

def calories_burned_running (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_swimming (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_cycling (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

-- Hypothesis
def total_calories_consumed : ℕ :=
  calories_from_cheezits 3 2 150 + calories_from_chocolate_bars 2 250 + calories_from_popcorn 500

def total_calories_burned : ℕ :=
  calories_burned_running 40 12 + calories_burned_swimming 30 15 + calories_burned_cycling 20 10

-- Theorem
theorem excess_calories : total_calories_consumed - total_calories_burned = 770 := by
  sorry

end excess_calories_l149_149536


namespace sum_of_two_numbers_l149_149650

theorem sum_of_two_numbers (x : ℤ) (sum certain value : ℤ) (h₁ : 25 - x = 5) : 25 + x = 45 := by
  sorry

end sum_of_two_numbers_l149_149650


namespace find_t_l149_149646

theorem find_t :
  ∃ (B : ℝ × ℝ) (t : ℝ), 
  B.1^2 + B.2^2 = 100 ∧ 
  B.1 - 2 * B.2 + 10 = 0 ∧ 
  B.1 > 0 ∧ B.2 > 0 ∧ 
  t = 20 ∧ 
  (∃ m : ℝ, 
    m = -2 ∧ 
    B.2 = m * B.1 + (8 + 2 * B.1 - m * B.1)) := 
by
  sorry

end find_t_l149_149646


namespace true_proposition_l149_149832

def parallel (x y : Type) : Prop := sorry -- Placeholder definition for parallelism

variables {α β a b : Type}

-- Conditions for Proposition P
def prop_P : Prop := (parallel α β) ∧ (parallel a α) → parallel a β

-- Conditions for Proposition Q
def prop_Q : Prop := (parallel a α) ∧ (parallel a β) ∧ (α ∩ β = b) → parallel a b

-- The assertion that (\neg P) \wedge Q is true
theorem true_proposition : ¬prop_P ∧ prop_Q :=
by
  sorry

end true_proposition_l149_149832


namespace how_many_statements_are_correct_l149_149182

variable {n : ℕ}
variable {S : ℕ → ℕ}
variable {a : ℕ → ℕ}

-- Given conditions
def problem_conditions :=
  (S 5 > S 6) ∧
  (S 6 = S 7) ∧
  (S 7 < S 8)

-- Statement evaluations
def statement_1 := ∃ d, d < 0 ∧ d = a (n + 1) - a n
def statement_2 := a 7 = 0
def statement_3 := S 9 > S 4
def statement_4 := ∃ k, (S k = S (k+1)) ∧ (∀ m, (S m ≠ S (m+1)) → m = k)

-- Main theorem to prove
theorem how_many_statements_are_correct :
  problem_conditions →
  (statement_1 ↔ false) + (statement_2 ↔ true) + (statement_3 ↔ false) + (statement_4 ↔ true) = 2 :=
sorry

end how_many_statements_are_correct_l149_149182


namespace absolute_value_of_k_l149_149408

theorem absolute_value_of_k (k : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + 4*k*x + 1 = 0 → x^2 + 2*x*k + 1/(4*k^2) = 0)
  (h2 : ∀ r s : ℝ, (r + s) = -4*k ∧ (r * s) = 1)
  (h3 : ∀ r s : ℝ, (r^2 + s^2) = 17) :
  abs(k) = sqrt(19) / 4 :=
sorry

end absolute_value_of_k_l149_149408


namespace negation_of_p_l149_149855

theorem negation_of_p :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * exp x > 1)) ↔ (∀ x : ℝ, x ≤ 0 → (x + 1) * exp x ≤ 1) :=
sorry

end negation_of_p_l149_149855


namespace hyperbola_eccentricity_is_2_l149_149822

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := 4 * a
  let e := c / a
  e

theorem hyperbola_eccentricity_is_2
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity a b ha hb = 2 := 
sorry

end hyperbola_eccentricity_is_2_l149_149822


namespace positive_real_solution_count_l149_149488

noncomputable def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 1728*x^5 - 1380*x^4

theorem positive_real_solution_count : ∃! x > 0, f x = 0 := 
by 
  sorry

end positive_real_solution_count_l149_149488


namespace hybrids_with_full_headlights_l149_149140

theorem hybrids_with_full_headlights
  (total_cars : ℕ) (hybrid_percentage : ℕ) (one_headlight_percentage : ℕ) :
  total_cars = 600 → hybrid_percentage = 60 → one_headlight_percentage = 40 →
  let total_hybrids := (hybrid_percentage * total_cars) / 100 in
  let one_headlight_hybrids := (one_headlight_percentage * total_hybrids) / 100 in
  let full_headlight_hybrids := total_hybrids - one_headlight_hybrids in
  full_headlight_hybrids = 216 :=
by
  intros h1 h2 h3
  sorry

end hybrids_with_full_headlights_l149_149140


namespace probability_ratio_l149_149284

theorem probability_ratio (p q : ℚ) :
  (∃ (slips : Finset ℕ), slips.card = 30 ∧ (∀ n ∈ slips, n ∈ {1, 2, 3, 4, 5, 6}) ∧
  (∀ n ∈ {1, 2, 3, 4, 5, 6}, (slips.filter(λ x, x = n)).card = 5) ∧
  (p = 30 / (Nat.choose 30 4)) ∧
  (q = 1500 / (Nat.choose 30 4))) → q / p = 50 := 
by
  sorry

end probability_ratio_l149_149284


namespace scientific_notation_periodicals_l149_149365

theorem scientific_notation_periodicals :
  (56000000 : ℝ) = 5.6 * 10^7 := by
sorry

end scientific_notation_periodicals_l149_149365


namespace find_y10_l149_149442

-- Definitions based on the conditions
def parabola_focus : (ℝ × ℝ) := (0, 1 / 4)
def point_on_parabola (n : ℕ) (x_n : ℝ) : Prop := (x_n^2 : ℝ) = (y_n : ℝ)
def distance_condition (y_n y_n1 : ℝ) : Prop := y_n1 - y_n = 2

-- The main theorem to prove the given question with the given conditions
theorem find_y10 (x_3 : ℝ) (hx3 : x_3 = 2) (y_3 : ℝ) (hy3 : y_3 = x_3^2)
  (h_dist : ∀ n : ℕ, distance_condition (2 * n - 2) (2 * (n + 1) - 2)) :
  (y_10 : ℝ) = 18 :=
begin
  -- Given x3 = 2, we know y3 = 4
  have hy3 : y_3 = 4 := by { rw [hx3, ← hy3], norm_num },
  
  -- y_n is an arithmetic sequence starting from y_3 with common difference 2
  have h_yn : ∀ n : ℕ, (y_n : ℝ) = 2 * n - 2,
  { intro n,
    induction n with n ih,
    { simp, },
    { simp, linarith [ih], } },
  
  -- Using the general form of y_n, we find y_10
  calc
    (y_10 : ℝ) = 2 * 10 - 2 : h_yn 10 ... = 18 : by norm_num,
end

end find_y10_l149_149442


namespace h_decreasing_on_positive_reals_range_of_m_l149_149563

open Real

namespace ProofProblem

-- Given functions f and g
def f (x : ℝ) : ℝ := x * log x
def g (x : ℝ) : ℝ := x^2 - 1

-- Define h(x) as f(x) - g(x)
def h (x : ℝ) : ℝ := f x - g x

-- Question 1: Prove that h(x) is decreasing on (0, +∞)
theorem h_decreasing_on_positive_reals : ∀x > 0, ∀y > 0, x < y → h x > h y := 
by
  sorry

-- Define F(x) for Question 2
def F (x : ℝ) (m : ℝ) : ℝ := f x - m * g x

-- Question 2: Prove the range of m such that f(x) - m * g(x) ≤ 0 for x ≥ 1
theorem range_of_m : {m : ℝ | ∀ x, x ≥ 1 → F x m ≤ 0} = {m : ℝ | m ≥ (1 / 2)} := 
by
  sorry

end ProofProblem

end h_decreasing_on_positive_reals_range_of_m_l149_149563


namespace jackie_walks_daily_l149_149167

theorem jackie_walks_daily (x : ℝ) :
  (∀ t : ℕ, t = 6 →
    6 * x = 6 * 1.5 + 3) →
  x = 2 :=
by
  sorry

end jackie_walks_daily_l149_149167


namespace smallest_prime_factor_2457_l149_149675

theorem smallest_prime_factor_2457 : ∃ p : ℕ, nat.prime p ∧ p ∣ 2457 ∧ ∀ q : ℕ, nat.prime q → q ∣ 2457 → q ≥ p :=
by {
  use 3,
  sorry
}

end smallest_prime_factor_2457_l149_149675


namespace angle_between_vectors_acute_l149_149443

def isAcuteAngle (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 > 0

def notCollinear (a b : ℝ × ℝ) : Prop :=
  ¬ ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem angle_between_vectors_acute (m : ℝ) :
  let a := (-1, 1)
  let b := (2 * m, m + 3)
  isAcuteAngle a b ∧ notCollinear a b ↔ m < 3 ∧ m ≠ -1 :=
by
  sorry

end angle_between_vectors_acute_l149_149443


namespace number_of_correct_statements_l149_149817

def is_k_type_function (f : ℝ → ℝ) (D : Set ℝ) (k : ℝ) : Prop :=
  ∃ m n : ℝ, [m, n] ⊆ D ∧ (∀ x ∈ [m, n], f x ∈ Set.Icc (k * m) (k * n))

def statement1 : Prop := ¬∃ k : ℝ, is_k_type_function (λ x, 3 - 4 / x) (Set.univ \ {0}) k

def statement2 : Prop := 
  ∀ (a : ℝ), a ≠ 0 → (is_k_type_function (λ x, (a^2 + a) / (a * x) - 1 / (a * x)) (Set.univ \ {0}) 1 →
  ∃ m n : ℝ, n - m = (2 * Real.sqrt 3) / 3)

def statement3 : Prop := 
  is_k_type_function (λ x, - (1 / 2) * x^2 + x) Set.univ 3 → ∃ m n : ℝ, m = -4 ∧ n = 0

theorem number_of_correct_statements : 
  [statement1, statement2, statement3].count true = 2 :=
sorry 

end number_of_correct_statements_l149_149817


namespace minimum_dot_product_l149_149074

-- Define the points A and B
def A : ℝ × ℝ × ℝ := (1, 2, 0)
def B : ℝ × ℝ × ℝ := (0, 1, -1)

-- Define the vector AP
def vector_AP (x : ℝ) := (x - 1, -2, 0)

-- Define the vector BP
def vector_BP (x : ℝ) := (x, -1, 1)

-- Define the dot product of vector AP and vector BP
def dot_product (x : ℝ) : ℝ := (x - 1) * x + (-2) * (-1) + 0 * 1

-- State the theorem
theorem minimum_dot_product : ∃ x : ℝ, dot_product x = (x - 1) * x + 2 ∧ 
  (∀ y : ℝ, dot_product y ≥ dot_product (1/2)) := 
sorry

end minimum_dot_product_l149_149074


namespace arithmetic_sequence_properties_l149_149072

-- Definitions of the conditions
def a_3 : ℤ := 7
def a_5_plus_a_7 : ℤ := 26
def b_n (n : ℕ) (C : ℤ) [hC : C ≠ 0] : ℤ := C^(2 * n + 1)

theorem arithmetic_sequence_properties
  (C : ℤ) [hC : C ≠ 0] (a_3 : ℤ = 7) (a_5_plus_a_7 : ℤ = 26)
  (a_n : ℕ → ℤ := λ n, 2 * n + 1)
  (S_n : ℕ → ℤ := λ n, n^2 + 2 * n) :

  -- Prove that given conditions, a_n = 2n + 1 and S_n = n^2 + 2n
  (∀ n, 2 * n + 1 = a_n n) ∧
  (∀ n, (∑ i in range (n+1), a_n i) = S_n n) ∧

  -- Prove that b_n = C^(a_n) is a geometric sequence
  (∀ n ≥ 1, b_n n C = (b_n (n-1) C) * (C^2)) :=
sorry

end arithmetic_sequence_properties_l149_149072


namespace charles_ate_no_bananas_l149_149689

theorem charles_ate_no_bananas (W C B : ℝ) (h1 : W = 48) (h2 : C = 35) (h3 : W + C = 83) : B = 0 :=
by
  -- Proof goes here
  sorry

end charles_ate_no_bananas_l149_149689


namespace poll_participation_l149_149151

-- A problem setup with conditions and the requirement to prove the total participants are 260.
variables (x : ℕ) (initial_votes_Oct22 : ℕ) (initial_votes_Oct29 : ℕ)
variables (additional_votes_Oct22 : ℕ) (final_percentage_Oct29 : ℚ)

noncomputable def total_participants (x : ℕ) : ℕ :=
  let initial_votes_Oct22 := 0.35 * x
  let initial_votes_Oct29 := 0.65 * x
  let additional_votes_Oct22 := 80
  let total_votes_Oct22 := initial_votes_Oct22 + additional_votes_Oct22
  let total_votes := x + additional_votes_Oct22
  total_votes

theorem poll_participation :
  let x := 180
  final_percentage_Oct29 = 0.45
  additional_votes_Oct22 = 80 →
  initial_votes_Oct22 = 0.35 * x →
  initial_votes_Oct29 = 0.65 * x →
  total_participants x = 260 :=
by
  sorry

end poll_participation_l149_149151


namespace video_duration_correct_l149_149957

/-
Define the conditions as given:
1. Vasya's time from home to school
2. Petya's time from school to home
3. Meeting conditions
-/

-- Define the times for Vasya and Petya
def vasya_time : ℕ := 8
def petya_time : ℕ := 5

-- Define the total video duration when correctly merged
def video_duration : ℕ := 5

-- State the theorem to be proved in Lean:
theorem video_duration_correct : vasya_time = 8 → petya_time = 5 → video_duration = 5 :=
by
  intros h1 h2
  exact sorry

end video_duration_correct_l149_149957


namespace residual_at_sample_point_l149_149103

/-- Given the regression equation y_hat = 0.85x - 85.7 and sample point (165, 57), 
the residual at this sample point is 2.45. -/
theorem residual_at_sample_point :
  ∀ (x y : ℝ), (x = 165) → (y = 57) →
  let y_hat := 0.85 * x - 85.7 in
  (y - y_hat) = 2.45 :=
by
  intros x y hx hy
  rw [hx, hy]
  let y_hat := 0.85 * 165 - 85.7
  rw (show 0.85 * 165 - 85.7 = 54.55 by norm_num) at y_hat
  rw (show 57 - 54.55 = 2.45 by norm_num)
  sorry

end residual_at_sample_point_l149_149103


namespace sum_of_sequence_S_2019_l149_149984

noncomputable theory
open_locale big_operators

def sequence_a (n : ℕ) : ℕ :=
nat.rec_on n 2 (λ n a_n, 3 * a_n + 2)

def sequence_b (n : ℕ) : ℕ :=
nat.log 3 (sequence_a n + 1)

def term (n : ℕ) : ℝ :=
1 / (sequence_b (2 * n - 1) * sequence_b (2 * n + 1))

def sum_S (n : ℕ) : ℝ :=
∑ i in finset.range n, term i

theorem sum_of_sequence_S_2019 : sum_S 2019 = 2019 / 4039 := by
  sorry

end sum_of_sequence_S_2019_l149_149984


namespace shift_upwards_l149_149631

theorem shift_upwards (a : ℝ) :
  (∀ x : ℝ, y = -2 * x + a) -> (a = 1) :=
by
  sorry

end shift_upwards_l149_149631


namespace inradius_of_triangle_l149_149100

noncomputable def hyperbola := {x : ℝ × ℝ // x.1^2 - (x.2 / 3) = 1}

def focus_1 : ℝ × ℝ := (-2, 0)
def focus_2 : ℝ × ℝ := (2, 0)

def line_intersection (t : ℝ) : ℝ × ℝ := (2 + t, t * m)

axiom angle_condition (P Q : ℝ × ℝ) : 
  ∃ (m : ℝ), m ≠ 0 ∧ 
  ∃ (P Q : ℝ × ℝ), P \in hyperbola ∧ Q \in hyperbola ∧ 
  ∃ γ, angle focus_1 P Q = γ ∧ γ = 90

def radius (F1 P Q : ℝ × ℝ) : ℝ :=
  let area := 1/2 * ((fst F1 * (snd P - snd Q)) + (fst P * (snd Q - snd F1)) + (fst Q * (snd F1 - snd P))) in
  let a := dist F1 P in
  let b := dist P Q in
  let c := dist Q F1 in
  let s := (a + b + c) / 2 in
  area / s

theorem inradius_of_triangle : 
  ∀ (P Q : ℝ × ℝ), 
    angle_condition P Q → radius focus_1 P Q = sqrt 7 - 1 :=
by sorry

end inradius_of_triangle_l149_149100


namespace range_of_a_l149_149066

noncomputable def has_two_distinct_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

theorem range_of_a (a : ℝ) (b : ℝ) (h₀ : a ∈ Ioo 0 1) (h₁ : b > 3 * Real.exp 2) :
  has_two_distinct_zeros (λ x, a^x + 2 * b * x + Real.exp 2) → a ∈ Set.Ico (Real.exp (-6)) 1 := 
sorry

end range_of_a_l149_149066


namespace proposition_2_l149_149963

variables {m n : Line} {α β : Plane}

theorem proposition_2 (h1 : m ∥ n) 
                     (h2 : m ⊆ α) 
                     (h3 : n ⟂ β)
                     : α ⟂ β := 
sorry

end proposition_2_l149_149963


namespace find_a_l149_149819

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then
    x^3 + 1
  else
    x^2 - a * x

theorem find_a : ∃ a : ℝ, f (f 0 a) a = -2 :=
by
  have h₁ : f 0 a = 1 := by
    simp [f]
    sorry
  have h₂ : f 1 a = 1 - a := by
    simp [f]
    sorry
  rw [h₁, h₂]
  existsi 3
  sorry

end find_a_l149_149819


namespace right_triangle_legs_l149_149990

theorem right_triangle_legs (a : ℝ) (α : ℝ) :
  ∀ (A B C H M : Type) 
  (hACB : ∠ACB = 90) (hCH : CH = a) (hα : ∠MCH = α),
  (BC = a / real.sin (real.pi / 4 - α / 2)) ∧ 
  (AC = a / real.sin (real.pi / 4 + α / 2)) :=
by sorry

end right_triangle_legs_l149_149990


namespace compare_neg_fractions_l149_149389

theorem compare_neg_fractions : 
  (- (8:ℚ) / 21) > - (3 / 7) :=
by sorry

end compare_neg_fractions_l149_149389


namespace smallest_resolvable_debt_l149_149286

theorem smallest_resolvable_debt (p g : ℤ) : 
  ∃ p g : ℤ, (500 * p + 350 * g = 50) ∧ ∀ D > 0, (∃ p g : ℤ, 500 * p + 350 * g = D) → 50 ≤ D :=
by {
  sorry
}

end smallest_resolvable_debt_l149_149286


namespace max_chess_pieces_l149_149671

theorem max_chess_pieces :
  ∃ W B : ℕ, W = 32 ∧ B = 16 ∧
  ∀ r c : fin 8, (∃ Wr Br : ℕ, Wr + Br = 8 ∧ Wr = 2 * Br ∧ Wr = W / 8 ∧ Br = B / 8) ∧
  (Wr = W / 8 ∧ Br = B / 8) :=
sorry

end max_chess_pieces_l149_149671


namespace find_divisor_l149_149295

def div_remainder (a b r : ℕ) : Prop :=
  ∃ k : ℕ, a = k * b + r

theorem find_divisor :
  ∃ D : ℕ, (div_remainder 242 D 15) ∧ (div_remainder 698 D 27) ∧ (div_remainder (242 + 698) D 5) ∧ D = 37 := 
by
  sorry

end find_divisor_l149_149295


namespace minimum_area_l149_149465

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

def S (a : ℝ) : ℝ :=
  let I := set.Icc (a - 1) (a + 1)
  2 * (max' (fun n => (f(n))) I - min' (fun n => (f(n))) I)

theorem minimum_area (a : ℝ) : S(a) ≥ 2 :=
sorry

end minimum_area_l149_149465


namespace circle_transformation_l149_149385

theorem circle_transformation (x y : ℝ) (h : x = 3 ∧ y = -4) : 
  let (x', y') := (x, -y) in
  let (x'', y'') := (x' + 5, y') in
  (x'', y'') = (8, 4) :=
by 
  obtain ⟨hx, hy⟩ := h
  have : x' = x ∧ y' = -y := by simp [hx, hy]
  have : x'' = x' + 5 ∧ y'' = y' := by simp [this]
  simp [hx, hy] at this
  exact this

end circle_transformation_l149_149385


namespace log_subtraction_l149_149405

theorem log_subtraction (a b : Real) (hab : a = 2 * Real.sqrt 2) (hb : b = 2) :
  Real.log 4 a - Real.log 4 b = 1 / 4 := by
  sorry

end log_subtraction_l149_149405


namespace inequality_transform_l149_149497

theorem inequality_transform (x y : ℝ) (h : y > x) : 2 * y > 2 * x := 
  sorry

end inequality_transform_l149_149497


namespace more_than_20_components_exists_l149_149516

-- Definitions for grid, diagonal, and connected components
def grid (m n : ℕ) := fin m → fin n → bool  -- Simplistic representation for grid

def diagonal (g : grid 8 8) :=  -- Function to define if a cell contains a diagonal
  λ x y : fin 8, g x y

def connected_components (g : grid 8 8) : ℕ :=  -- Simplistic definition of connected components
  sorry -- Placeholder for the actual function that calculates connected components

-- The theorem stating that there exists a grid configuration where the number of 
-- connected components formed by the diagonals is greater than 20
theorem more_than_20_components_exists : ∃ (g : grid 8 8), connected_components g > 20 :=
  sorry

end more_than_20_components_exists_l149_149516


namespace circle_graph_representation_correct_l149_149301

theorem circle_graph_representation_correct :
  ∀ (yellow_shirts red_shirts blue_shirts green_shirts total_shirts : ℕ),
  yellow_shirts = 8 → red_shirts = 4 → blue_shirts = 2 → green_shirts = 2 →
  total_shirts = yellow_shirts + red_shirts + blue_shirts + green_shirts →
  (yellow_shirts / total_shirts = 1 / 2) ∧
  (red_shirts / total_shirts = 1 / 4) ∧
  (blue_shirts / total_shirts = 1 / 8) ∧
  (green_shirts / total_shirts = 1 / 8) →
  (50 : ℕ) = 50 ∧
  (25 : ℕ) = 25 ∧
  (12.5 : ℕ) = 13 ∧
  (12.5 : ℕ) = 13 :=
by {
  intros,
  sorry
}

end circle_graph_representation_correct_l149_149301


namespace sum_g_equals_2015_l149_149081

-- Define the cubic function g(x)
def g (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * x^2 + 2 * x + (1 / 12)

-- Define the symmetric center condition for g(x)
def symmetric_center (f : ℝ → ℝ) (c : ℝ) (y : ℝ) : Prop :=
  ∀ x : ℝ, f (c + x) + f (c - x) = 2 * y

-- Given data for the problem
theorem sum_g_equals_2015 : 
  symmetric_center g (1 / 2) 1 →
  (∑ k in finset.range 2015, g ((k + 1) / 2016)) = 2015 :=
by
  intros h
  sorry

end sum_g_equals_2015_l149_149081


namespace problem_1_problem_2_l149_149036

open BigOperators

-- Question 1
theorem problem_1 (a : Fin 2021 → ℝ) :
  (1 + 2 * x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  (∑ i in Finset.range 2021, (i * a i)) = 4040 * 3 ^ 2019 :=
sorry

-- Question 2
theorem problem_2 (a : Fin 2021 → ℝ) :
  (1 - x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  ((∑ i in Finset.range 2021, 1 / a i)) = 2021 / 1011 :=
sorry

end problem_1_problem_2_l149_149036


namespace positive_value_of_a_l149_149852

noncomputable def distance_point_to_line (a b c x₀ y₀ : ℝ) : ℝ :=
  (abs (a * x₀ + b * y₀ + c)) / (real.sqrt (a^2 + b^2))

theorem positive_value_of_a 
  (a : ℝ)
  (h1 : ∀ x y : ℝ, x^2 + (y - a)^2 = 16 → x = -1 ∧ y = a ∨ x ≠ -1 ∧ y ≠ a)
  (h2 : ∀ x y: ℝ, ax - y + 6 = 0 → False)
  (h3 : distance_point_to_line a (-1 : ℝ) 6 (-1 : ℝ) a = 2 * real.sqrt (2))
  (h4 : a > 0) : a = 1 := 
  sorry

end positive_value_of_a_l149_149852


namespace number_of_lines_at_given_distances_l149_149215

-- Define the points C and D and the distance between them
variables {C D : Point}
axiom distance_CD_eq_7 : dist C D = 7

-- Define the concept of lines at fixed distances from points
def line_at_fixed_distances (p1 p2 : Point) (d1 d2 : ℝ) : ℕ := 
  sorry -- representation of the number of lines

-- State the theorem that we want to prove
theorem number_of_lines_at_given_distances : 
  line_at_fixed_distances C D 3 4 = 3 := 
sorry

end number_of_lines_at_given_distances_l149_149215


namespace nth_equation_l149_149582

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - 1 = 4 * n * (n + 1) := 
by
  sorry

end nth_equation_l149_149582


namespace trevor_eggs_left_l149_149535

theorem trevor_eggs_left :
  let eggs_from_gertrude := 4,
      eggs_from_blanche := 3,
      eggs_from_nancy := 2,
      eggs_from_martha := 2,
      eggs_dropped := 2,
      total_eggs_collected := eggs_from_gertrude + eggs_from_blanche + eggs_from_nancy + eggs_from_martha,
      eggs_left := total_eggs_collected - eggs_dropped
  in eggs_left = 9 :=
by
  let eggs_from_gertrude := 4
  let eggs_from_blanche := 3
  let eggs_from_nancy := 2
  let eggs_from_martha := 2
  let eggs_dropped := 2
  let total_eggs_collected := eggs_from_gertrude + eggs_from_blanche + eggs_from_nancy + eggs_from_martha
  let eggs_left := total_eggs_collected - eggs_dropped
  have h : eggs_left = 9 := by simp
  exact h

end trevor_eggs_left_l149_149535


namespace radius_of_circle_through_A_B_tangent_to_CD_l149_149070

-- Definitions based on conditions
structure Rectangle where
  A B C D : ℝ × ℝ
  length : ℝ
  width : ℝ
  wlength : length = dist A B ∧ length = dist C D
  wwidth : width = dist B C ∧ width = dist D A

structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : (ℝ × ℝ) → Prop
  passes_through A := dist center A = radius

structure Tangent where
  circle : Circle
  line_segment : (ℝ × ℝ) → (ℝ × ℝ) → Prop
  tangent_at : (ℝ × ℝ) → Prop
  tangent_at E := ∃ P Q, line_segment P Q ∧ dist P E + dist Q E = dist P Q

-- Given rectangle ABCD with specific dimensions
def rectangleABCD : Rectangle :=
  { A := (0, 0),
    B := (10, 0),
    C := (10, 6),
    D := (0, 6),
    length := 10,
    width := 6,
    wlength := by simp [dist],
    wwidth := by simp [dist] }

-- Circle through A, B, tangent to CD
def problem_statement : Prop :=
  ∃ (circle : Circle),
  circle.passes_through (rectangleABCD.A) ∧
  circle.passes_through (rectangleABCD.B) ∧
  (∃ (E : ℝ × ℝ), (Tangent circle).tangent_at E ∧
  ∃ (line_segment : (ℝ × ℝ) → (ℝ × ℝ) → Prop),
  (E = (line_segment.1).2 ∨ E = (line_segment.2).1) ∧
  line_segment (rectangleABCD.C) (rectangleABCD.D)) ∧
  circle.radius = 5

-- Proof problem (statement only)
theorem radius_of_circle_through_A_B_tangent_to_CD :
  problem_statement :=
  sorry

end radius_of_circle_through_A_B_tangent_to_CD_l149_149070


namespace construct_symmetric_trees_l149_149704

-- Definitions based on problem conditions.
structure Path where
  name : String

structure Tree where
  position : (ℝ × ℝ)

-- Existing condition
def O : Tree := { position := (0, 0) }

-- Paths definition
def paths : List Path :=
  [ { name := "Path 1" },
    { name := "Path 2" },
    { name := "Path 3" },
    { name := "Path 4" } ]

-- Statement: Constructing three additional trees to maintain symmetry
theorem construct_symmetric_trees :
  ∃ A B C : Tree,
    (A.position.1 ≠ 0 ∨ A.position.2 ≠ 0) ∧ -- A is not at the origin
    (B.position.1 ≠ 0 ∨ B.position.2 ≠ 0) ∧ -- B is not at the origin
    (C.position.1 ≠ 0 ∨ C.position.2 ≠ 0) ∧ -- C is not at the origin
    (A.position = (-O.position.1, -O.position.2) ∨ B.position = (-O.position.1, -O.position.2) ∨ C.position = (-O.position.1, -O.position.2)) ∧ 
    -- ensuring positioned symmetrically based on one tree at a distance
    -- and satisfy symmetry for equal number on both sides:
    ((A.position.1 = -B.position.1 ∧ A.position.2 = B.position.2) ∨
     (B.position.1 = -C.position.1 ∧ B.position.2 = C.position.2) ∨
     (C.position.1 = -A.position.1 ∧ C.position.2 = A.position.2)) := sorry

end construct_symmetric_trees_l149_149704


namespace angle_APB_is_135_degrees_l149_149309

-- Defining the problem statement in Lean
theorem angle_APB_is_135_degrees
  (A B C D P : Point)
  (square : isSquare A B C D)
  (P_inside : isInsideSquare P A B C D)
  (PA : dist P A = 1)
  (PB : dist P B = 2)
  (PC : dist P C = 3) :
  angle A P B = 135 :=
begin
  -- Proof will be provided here
  sorry
end

end angle_APB_is_135_degrees_l149_149309


namespace evaluate_product_series_l149_149695

def product_series (a b : ℕ) : ℚ :=
  ∏ n in Finset.range (b - a + 1), (1 - (1 / (a + n : ℚ)))

theorem evaluate_product_series :
  product_series 10 100 = 9 / 100 := by
  sorry

end evaluate_product_series_l149_149695


namespace geometric_series_solution_l149_149756

theorem geometric_series_solution : 
  (∀ (x : ℝ), (1 - 1/3 + 1/9 - 1/27 + ...) * (1 + 1/3 + 1/9 + 1/27 + ...) = 1/(1 - 1/x) → x = 9) := 
sorry

end geometric_series_solution_l149_149756


namespace gcd_153_119_eq_17_l149_149782

theorem gcd_153_119_eq_17 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_eq_17_l149_149782


namespace p_sufficient_not_necessary_for_q_l149_149039

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l149_149039


namespace ellipse_to_circle_passes_point_l149_149941

-- Definitions for the problem
variables {a b p q : ℝ}
variables (x_A y_A x_B y_B : ℝ)

-- Conditions
def on_ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = a^2
def passes_through_M (x y : ℝ) : Prop := x = p ∧ y = q

-- Theorem statement
theorem ellipse_to_circle_passes_point
  (ellipse_eq : a > b > 0)
  (A_on_ellipse : on_ellipse a b x_A y_A)
  (B_on_ellipse : on_ellipse a b x_B y_B)
  (AB_through_M : passes_through_M a b p q)
  : 
  ∀ x y, ∃ q' : ℝ, on_circle a a b (x, q') → passes_through_M a b p (a / b * q) :=
sorry

end ellipse_to_circle_passes_point_l149_149941


namespace geometric_sum_is_1024_l149_149470

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 3 = 3

def binom_sum (a : ℕ → ℝ) : ℝ :=
  (∑ k in Finset.range 11, (Nat.choose 10 k) * (a (k+1))) *
  (∑ k in Finset.range 11, (Nat.choose 10 k) * ((-1 : ℝ)^k) * (a (k+1)))

theorem geometric_sum_is_1024 (a : ℕ → ℝ) (h : geometric_sequence a) :
  binom_sum a = 1024 :=
sorry

end geometric_sum_is_1024_l149_149470


namespace total_votes_l149_149512

theorem total_votes (A B C V : ℝ)
  (h1 : A = B + 0.10 * V)
  (h2 : A = C + 0.15 * V)
  (h3 : A - 3000 = B + 3000)
  (h4 : B + 3000 = A - 0.10 * V)
  (h5 : B + 3000 = C + 0.05 * V)
  : V = 60000 := 
sorry

end total_votes_l149_149512


namespace cylinder_volume_l149_149664

noncomputable def volume_cylinder (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) : ℝ :=
  let ratio_r := r_cylinder / r_cone
  let ratio_h := h_cylinder / h_cone
  (3 : ℝ) * ratio_r^2 * ratio_h * V_cone

theorem cylinder_volume (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) :
    r_cylinder / r_cone = 2 / 3 →
    h_cylinder / h_cone = 4 / 3 →
    V_cone = 5.4 →
    volume_cylinder V_cone r_cylinder r_cone h_cylinder h_cone = 3.2 :=
by
  intros h1 h2 h3
  rw [volume_cylinder, h1, h2, h3]
  sorry

end cylinder_volume_l149_149664


namespace find_x_solution_l149_149421

theorem find_x_solution (x : ℚ) : (∀ y : ℚ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by
  sorry

end find_x_solution_l149_149421


namespace hexagon_area_l149_149258

theorem hexagon_area (s : ℝ) (hex_area : ℝ) (p q : ℤ) :
  s = 3 ∧ hex_area = (3 * Real.sqrt 3 / 2) * s^2 ∧ hex_area = Real.sqrt p + Real.sqrt q → p + q = 545 :=
by
  sorry

end hexagon_area_l149_149258


namespace positional_relationship_uncertain_l149_149587

variables {α β : Type*}
variables [plane α] [plane β]
variables {b m : line}

-- Conditions
axiom plane_perpendicular (h₁ : α ⊥ β)
axiom line_in_plane_b (h₂ : b ∈ α)
axiom line_in_plane_m (h₃ : m ∈ β)
axiom line_perpendicular (h₄ : b ⊥ m)

-- The statement to be proven
theorem positional_relationship_uncertain (h₁ : α ⊥ β) (h₂ : b ∈ α) (h₃ : m ∈ β) (h₄ : b ⊥ m) : 
  relationship b β = "uncertain" :=
sorry

end positional_relationship_uncertain_l149_149587


namespace binomial_third_term_coefficient_l149_149977

theorem binomial_third_term_coefficient (n : ℕ) (C : ℕ → ℕ → ℕ) :
  (C n 2 = C n 3) → (C 5 2 * 2^3 * (-1)^2 = 80) :=
by
  intro h
  sorry

end binomial_third_term_coefficient_l149_149977


namespace num_pos_int_lt_zero_prod_l149_149014

theorem num_pos_int_lt_zero_prod :
  let P := ∏ i in finset.range 50, (2 * i + 1 - n)
  ∃ (N : ℕ), N = 25 ∧ ∀ (n : ℕ), 0 < n → n * P < 0 → n < 100 :=
  sorry

end num_pos_int_lt_zero_prod_l149_149014


namespace p_sufficient_not_necessary_for_q_l149_149040

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l149_149040


namespace remy_gallons_used_l149_149224

def roman_usage (R : ℕ) : Prop := R + (3 * R + 1) = 33

def remy_usage (R : ℕ) (Remy : ℕ) : Prop := Remy = 3 * R + 1

theorem remy_gallons_used :
  ∃ R Remy : ℕ, roman_usage R ∧ remy_usage R Remy ∧ Remy = 25 :=
  by
    sorry

end remy_gallons_used_l149_149224


namespace gather_checkers_in_n_minus_1_moves_l149_149531

theorem gather_checkers_in_n_minus_1_moves (n : ℕ) (h : n ≥ 1) :
  ∃ moves : ℕ, moves = n - 1 ∧
  (∀ board : (fin n) → ℕ, (∀ i, board i = 1) → ∃ square : fin n, board square = n) :=
sorry

end gather_checkers_in_n_minus_1_moves_l149_149531


namespace sqrt_pos_int_for_n_greater_than_1_l149_149279

def a (n : ℕ) : ℚ
| 0     := 1 -- Lean sequence indexing starts from 0
| (n+1) := a n / 2 + 1 / (4 * a n)

theorem sqrt_pos_int_for_n_greater_than_1 (n : ℕ) (h : n > 1) :
  ∃ k : ℕ, k > 0 ∧ (k : ℚ)^2 = 2 / (2 * (a n)^2 - 1) :=
sorry

end sqrt_pos_int_for_n_greater_than_1_l149_149279


namespace min_ab_value_l149_149708

theorem min_ab_value 
  (a b : ℝ) 
  (hab_pos : a * b > 0)
  (collinear_condition : 2 * a + 2 * b + a * b = 0) :
  a * b ≥ 16 := 
sorry

end min_ab_value_l149_149708


namespace lines_intersections_l149_149281

theorem lines_intersections (n : ℕ) (h1 : ∀ i j, i ≠ j → ¬parallel (lines i) (lines j)) 
  (h2 : ∀ i j k, distinct [i, j, k] → ¬concurrent (lines i) (lines j) (lines k)) : 
  (∑ k in finset.range (n), k) = (n * (n - 1)) / 2 :=
by
  sorry

where
  lines (i : ℕ) : line -- Definition to be provided as required
  parallel (l1 l2 : line) : Prop -- Definition to be provided as required
  concurrent (l1 l2 l3 : line) : Prop -- Definition to be provided as required
  distinct (l : list ℕ) : Prop -- Definition to be provided as required

end lines_intersections_l149_149281


namespace square_field_area_l149_149738

noncomputable def square_field_dimensions (height_board : ℝ) (board_length : ℝ) (x : ℝ) : ℝ :=
  if (height_board * board_length = 40) ∧ 
     (1.6 * x = x^2 / 10000) then 
    x 
  else 
    sorry

theorem square_field_area (height_board : ℝ) (board_length : ℝ) (x : ℝ) :
  height_board = 4 → board_length = 10 → x = 16000 :=
by
  intros
  have h1 : height_board * board_length = 40,
  { calc height_board * board_length = 4 * 10 : by sorry },
  have h2 : 1.6 * x = x^2 / 10000,
  { calc 1.6 * x = x^2 / 10000 : by sorry },
  exact sorry

end square_field_area_l149_149738


namespace activity_participants_l149_149032

variable (A B C D : Prop)

theorem activity_participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) : B ∧ C ∧ ¬A ∧ ¬D :=
by
  sorry

end activity_participants_l149_149032


namespace price_of_other_pieces_l149_149174

theorem price_of_other_pieces (total_spent : ℕ) (total_pieces : ℕ) (price_piece1 : ℕ) (price_piece2 : ℕ) 
  (remaining_pieces : ℕ) (price_remaining_piece : ℕ) (h1 : total_spent = 610) (h2 : total_pieces = 7)
  (h3 : price_piece1 = 49) (h4 : price_piece2 = 81) (h5 : remaining_pieces = (total_pieces - 2))
  (h6 : total_spent - price_piece1 - price_piece2 = remaining_pieces * price_remaining_piece) :
  price_remaining_piece = 96 := 
by
  sorry

end price_of_other_pieces_l149_149174


namespace find_x_l149_149037

variables {x : ℝ}
def vector_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x
  (h1 : (6, 1) = (6, 1))
  (h2 : (x, -3) = (x, -3))
  (h3 : vector_parallel (6, 1) (x, -3)) :
  x = -18 := by
  sorry

end find_x_l149_149037


namespace not_possible_to_arrange_segments_l149_149533

theorem not_possible_to_arrange_segments : ¬ ∃ (S : Fin 1968 → Set (ℝ × ℝ)), 
  (∀ i j, i ≠ j → 
    (S i ≠ S j ∧ 
     (∃ a b c d, S i = {p : ℝ × ℝ | a < p.1 ∧ p.1 < b ∧ c < p.2 ∧ p.2 < d}) ∧
     ∀ (x y : ℝ × ℝ), (S j x y → a < x.1 ∧ x.1 < b ∧ c < y.2 ∧ y.2 < d))) := 
begin
  sorry
end

end not_possible_to_arrange_segments_l149_149533


namespace distinct_paintings_count_l149_149905

def numberOfDifferentPaintings : ℕ := 21

theorem distinct_paintings_count
    (disks : Finset ℕ)
    (n : ℕ)
    (number_blue : ℕ)
    (number_red : ℕ)
    (number_green : ℕ)
    (set_eq : disks.card = n)
    (blue_disks : disks.filter (λ x, x < number_blue) = range number_blue)
    (red_disks : disks.filter (λ x, x ≥ number_blue ∧ x < number_blue + number_red) 
        = range number_blue (number_blue + number_red))
    (green_disks : disks.filter (λ x, x = number_blue + number_red + number_green) 
        = range (number_blue + number_red + number_green) n)
    (is_circle : True)
    :
    (∃ unique_colorings : ℕ, iso_invariant_color_count disks unique_colorings ∧ unique_colorings = numberOfDifferentPaintings) :=
sorry

end distinct_paintings_count_l149_149905


namespace sample_size_proof_l149_149339

-- Definitions
def total_staff := 160
def middle_managers := 30
def senior_managers := 10
def total_population := total_staff + middle_managers + senior_managers

-- Proportion of senior managers in the population
def proportion_senior_managers : ℚ := senior_managers / total_population

-- Stratified sampling condition: 1 senior manager in the sample
def sample_size := 20
def sample_proportion_senior_managers : ℚ := 1 / sample_size

-- Lean 4 statement to prove
theorem sample_size_proof (h : proportion_senior_managers = sample_proportion_senior_managers) : sample_size = 20 :=
by {
  -- Import proportion_senior_managers
  have p := proportion_senior_managers,
  -- Import sample_proportion_senior_managers
  have q := sample_proportion_senior_managers,
  -- Use h directly
  exact sorry
}

end sample_size_proof_l149_149339


namespace arrange_animals_in_cages_l149_149243

theorem arrange_animals_in_cages :
  let chickens : ℕ := 3
  let dogs : ℕ := 3
  let cats : ℕ := 4
  finset.card {arrangement | 
    arrangement.count chickens = chickens ∧ 
    arrangement.count dogs = dogs ∧ 
    arrangement.count cats = cats ∧
    arrangement.chunk 3 = [chickens, dogs, cats] ∨ 
    arrangement.chunk 3 = [chickens, cats, dogs] ∨
    arrangement.chunk 3 = [dogs, chickens, cats] ∨ 
    arrangement.chunk 3 = [dogs, cats, chickens] ∨ 
    arrangement.chunk 3 = [cats, chickens, dogs] ∨ 
    arrangement.chunk 3 = [cats, dogs, chickens]} = 5184 := 
sorry

end arrange_animals_in_cages_l149_149243


namespace suitable_for_comprehensive_survey_l149_149305

def optionA : Prop := "Survey on the number of waste batteries discarded in the city every day is not suitable for comprehensive survey"
def optionB : Prop := "Survey on the quality of ice cream in the cold drink market is not suitable for comprehensive survey"
def optionC : Prop := "Survey on the current status of students' mental health in a school in Huicheng District is suitable for comprehensive survey"
def optionD : Prop := "Survey on the duration of air conditioner usage by residents in our city is not suitable for comprehensive survey"

theorem suitable_for_comprehensive_survey : optionC = "Survey on the current status of students' mental health in a school in Huicheng District is suitable for comprehensive survey" :=
by 
  sorry

end suitable_for_comprehensive_survey_l149_149305


namespace select_student_based_on_variance_l149_149235

-- Define the scores for students A and B
def scoresA : List ℚ := [12.1, 12.1, 12.0, 11.9, 11.8, 12.1]
def scoresB : List ℚ := [12.2, 12.0, 11.8, 12.0, 12.3, 11.7]

-- Define the function to calculate the mean of a list of rational numbers
def mean (scores : List ℚ) : ℚ := (scores.foldr (· + ·) 0) / scores.length

-- Define the function to calculate the variance of a list of rational numbers
def variance (scores : List ℚ) : ℚ :=
  let m := mean scores
  (scores.foldr (λ x acc => acc + (x - m) ^ 2) 0) / scores.length

-- Prove that the variance of student A's scores is less than the variance of student B's scores
theorem select_student_based_on_variance :
  variance scoresA < variance scoresB := by
  sorry

end select_student_based_on_variance_l149_149235


namespace stable_performance_l149_149289

-- Define the variances of the students' long jump performance
def S_A^2 : ℝ := 0.04
def S_B^2 : ℝ := 0.13

-- The theorem to prove that Student A's performance is more stable than Student B's
theorem stable_performance :
  S_A^2 < S_B^2 :=
  by
    sorry

end stable_performance_l149_149289


namespace log4_eval_l149_149406

theorem log4_eval :
  (log 4 (1 / 4 + 1 / sqrt 4) = log 4 3 - 1) :=
by 
  sorry

end log4_eval_l149_149406


namespace parts_placement_sequence_l149_149586

theorem parts_placement_sequence :
  ∃ seq : List ℕ, seq = [2, 7, 5, 6, 4, 1, 3] ∧ 
  (∀ x ∈ seq, x ∈ [1, 2, 3, 4, 5, 6, 7]) :=
by
  use [2, 7, 5, 6, 4, 1, 3]
  split
  · rfl
  · intro x hx
    simp at hx
    tauto

end parts_placement_sequence_l149_149586


namespace p_sufficient_not_necessary_q_l149_149044

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l149_149044


namespace area_of_overlap_l149_149288

noncomputable theory

variables {α : ℝ} (sin_α_ne_zero : sin α ≠ 0)

def first_strip_width := 1
def second_strip_width := 2

theorem area_of_overlap (overlap_area : ℝ) :
  overlap_area = (1 / sin α) :=
by sorry

end area_of_overlap_l149_149288


namespace quadratic_discriminant_l149_149009

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/5) (1/5) = 576 / 25 := by
  sorry

end quadratic_discriminant_l149_149009


namespace slope_is_45_degrees_l149_149454

variable (A B : ℝ × ℝ)
variable (x1 x2 y1 y2 : ℝ)

-- Define points A and B
def A := (1, 3)
def B := (5, 7)

-- Define coordinates of points A and B
def x1 := 1
def y1 := 3
def x2 := 5
def y2 := 7

-- Define the slope formula
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- The theorem stating the problem and verifying the solution
theorem slope_is_45_degrees : slope x1 y1 x2 y2 = 1 ∧ angle_of_slope (slope x1 y1 x2 y2) = 45 := 
by sorry

end slope_is_45_degrees_l149_149454


namespace p_sufficient_but_not_necessary_q_l149_149061

theorem p_sufficient_but_not_necessary_q :
  ∀ x : ℝ, (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros x hx
  cases hx with h1 h2
  apply And.intro
  apply lt_of_lt_of_le h1
  linarith
  apply h2

end p_sufficient_but_not_necessary_q_l149_149061


namespace prime_remainders_l149_149550

theorem prime_remainders (N : ℕ) 
  (hN1 : odd N) 
  (f : Fin 999 → Fin) 
  (hf : ∀ i j, i ≠ j → f i ≠ f j) 
  (hN2 : ∃ k, f k = 0) :
  ∃ k, (500 ≤ k ∧ k ≤ 1000) ∧ Prime k ∧ f k = 0 := sorry

end prime_remainders_l149_149550


namespace slope_of_line_at_120_deg_l149_149875

theorem slope_of_line_at_120_deg : 
  let θ := 120
  ∃ (k : ℝ), k = - Real.sqrt 3 ∧ θ = 120 := 
by 
  let θ := 120
  use -Real.sqrt 3
  split
  { refl }
  sorry

end slope_of_line_at_120_deg_l149_149875


namespace p_sufficient_but_not_necessary_q_l149_149063

theorem p_sufficient_but_not_necessary_q :
  ∀ x : ℝ, (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros x hx
  cases hx with h1 h2
  apply And.intro
  apply lt_of_lt_of_le h1
  linarith
  apply h2

end p_sufficient_but_not_necessary_q_l149_149063


namespace max_value_trig_expr_l149_149412

theorem max_value_trig_expr (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π) : 
  ∃ x : ℝ, (x = cos (θ / 2) * (2 - sin θ)) ∧ (x <= 3 / 2) :=
by
  sorry

end max_value_trig_expr_l149_149412


namespace z_in_second_quadrant_l149_149496

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := -1 + 3 * i

def is_in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem z_in_second_quadrant : is_in_second_quadrant (-1, 3) :=
by
  -- This is the point corresponding to z.
  sorry

end z_in_second_quadrant_l149_149496


namespace chess_match_duration_l149_149216

def time_per_move_polly := 28
def time_per_move_peter := 40
def total_moves := 30
def moves_per_player := total_moves / 2

def Polly_time := moves_per_player * time_per_move_polly
def Peter_time := moves_per_player * time_per_move_peter
def total_time_seconds := Polly_time + Peter_time
def total_time_minutes := total_time_seconds / 60

theorem chess_match_duration : total_time_minutes = 17 := by
  sorry

end chess_match_duration_l149_149216


namespace unique_real_function_l149_149777

theorem unique_real_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, (f (x * y) / 2 + f (x * z) / 2 - f x * f (y * z)) ≥ 1 / 4) →
  (∀ x : ℝ, f x = 1 / 2) :=
by
  intro h
  -- proof steps go here
  sorry

end unique_real_function_l149_149777


namespace correct_proposition_l149_149188

-- Definitions of lines and planes
variables {Line Plane : Type}
variable  perpendicular : Line → Plane → Prop
variable  parallel : Line → Line → Prop
variable  inc : Line → Plane → Prop

-- Different lines and planes
variables (l m n : Line)
variables (alpha beta : Plane)

-- Problem conditions
axiom diff_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n
axiom diff_planes : alpha ≠ beta
axiom m_perpendicular_alpha : perpendicular m alpha
axiom n_parallel_beta : parallel n beta
axiom alpha_parallel_beta : parallel alpha beta

-- Problem statement to prove
theorem correct_proposition : perpendicular m n :=
sorry

end correct_proposition_l149_149188


namespace pechkin_made_mistake_l149_149959

-- Definitions based on the given conditions
def total_distance (S : ℝ) : ℝ := 2 * S
def walked_distance (S : ℝ) : ℝ := S
def walked_time (S : ℝ) : ℝ := S / 5
def total_time (S : ℝ) : ℝ := 3 * walked_time S / 2
def biking_time (S : ℝ) : ℝ := total_time S - walked_time S
def biking_distance (S : ℝ) : ℝ := 12 * biking_time S

-- Main theorem stating the contradiction
theorem pechkin_made_mistake (S : ℝ) : biking_distance S ≠ walked_distance S := 
sorry

end pechkin_made_mistake_l149_149959


namespace smartphone_customers_l149_149126

theorem smartphone_customers (k : ℝ) (p1 p2 c1 c2 : ℝ)
  (h₁ : p1 * c1 = k)
  (h₂ : 20 = p1)
  (h₃ : 200 = c1)
  (h₄ : 400 = c2) :
  p2 * c2 = k  → p2 = 10 :=
by
  sorry

end smartphone_customers_l149_149126


namespace beta_fish_in_tank_1_l149_149619

variable (x : ℕ)

def tank_1_total_fish : ℕ := 7 + x
def tank_2_total_fish : ℕ := 2 * (7 + x)
def tank_3_total_fish : ℕ := (2 * (7 + x)) / 3

theorem beta_fish_in_tank_1 : tank_3_total_fish x = 10 → x = 8 := by
  intro h
  have h1 : tank_2_total_fish x = 30 := by
    rw [tank_3_total_fish, h]
    norm_num
  have h2 : tank_1_total_fish x = 15 := by
    rwa [tank_2_total_fish, h1]
  have h3 : 7 + x = 15 := by
    exact h2
  linarith

end beta_fish_in_tank_1_l149_149619


namespace total_number_of_workers_l149_149316

theorem total_number_of_workers (W N : ℕ) 
    (avg_all : ℝ) 
    (avg_technicians : ℝ) 
    (avg_non_technicians : ℝ)
    (h1 : avg_all = 8000)
    (h2 : avg_technicians = 20000)
    (h3 : avg_non_technicians = 6000)
    (h4 : 7 * avg_technicians + N * avg_non_technicians = (7 + N) * avg_all) :
  W = 49 := by
  sorry

end total_number_of_workers_l149_149316


namespace roof_length_width_difference_l149_149277

variable (w l : ℕ)

theorem roof_length_width_difference (h1 : l = 7 * w) (h2 : l * w = 847) : l - w = 66 :=
by 
  sorry

end roof_length_width_difference_l149_149277


namespace product_of_chord_lengths_eq_l149_149920

def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 18)

def A : ℂ := 4
def B : ℂ := -4
def C (k : ℕ) : ℂ := 4 * ω^k

noncomputable def chord_length_AC (k : ℕ) : ℝ := Complex.abs (A - C k)
noncomputable def chord_length_BC (k : ℕ) : ℝ := Complex.abs (B - C k)

theorem product_of_chord_lengths_eq : 
  (∏ k in Finset.range 8, chord_length_AC (k + 1)) *
  (∏ k in Finset.range 8, chord_length_BC (k + 1)) = 4718592 := by
  sorry

end product_of_chord_lengths_eq_l149_149920


namespace rahim_books_second_shop_l149_149961

variable (x : ℕ)

-- Definitions of the problem's conditions
def total_cost : ℕ := 520 + 248
def total_books (x : ℕ) : ℕ := 42 + x
def average_price : ℕ := 12

-- The problem statement in Lean 4
theorem rahim_books_second_shop : x = 22 → total_cost / total_books x = average_price :=
  sorry

end rahim_books_second_shop_l149_149961


namespace correct_transformation_l149_149302

theorem correct_transformation (a b c : ℝ) (h : (b / (a^2 + 1)) > (c / (a^2 + 1))) : b > c :=
by {
  -- Placeholder proof
  sorry
}

end correct_transformation_l149_149302


namespace gcd_of_repeated_three_digit_number_is_constant_l149_149728

theorem gcd_of_repeated_three_digit_number_is_constant (m : ℕ) (h1 : 100 ≤ m) (h2 : m < 1000) : 
  ∃ d, d = 1001001 ∧ ∀ n, n = 10010013 * m → (gcd 1001001 n) = 1001001 :=
by
  -- The proof would go here
  sorry

end gcd_of_repeated_three_digit_number_is_constant_l149_149728


namespace k_eq_4_cos_angle_correct_l149_149485

noncomputable def k_value (a b : ℝ × ℝ) (h : b.1 * (a.1 - 3 * b.1) + b.2 * (a.2 - 3 * b.2) = 0) : ℝ :=
  let k := a.2 in k

theorem k_eq_4 (a b : ℝ × ℝ) (h : b.1 * (a.1 - 3 * b.1) + b.2 * (a.2 - 3 * b.2) = 0) : 
  k_value a b h = 4 :=
begin
  -- proof steps go here
  sorry
end

noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

theorem cos_angle_correct (a b : ℝ × ℝ) (h : b.1 * (a.1 - 3 * b.1) + b.2 * (a.2 - 3 * b.2) = 0) :
  cos_angle (2, 4) (1, 1) = 3 * Real.sqrt 10 / 10 :=
begin
  -- proof steps go here
  sorry
end

end k_eq_4_cos_angle_correct_l149_149485


namespace subgraph_with_max_degree_is_topological_l149_149710

-- Definitions corresponding to conditions in the problem.
variable (X : Type) [graph X]
variable (TX IX : set X)
variable (Δ : X → ℕ)

-- Problem statement using the conditions
theorem subgraph_with_max_degree_is_topological (h1 : ∀ x, TX x → IX x)
  (h2 : ∀ x, Δ x ≤ 3) : ∀ x, IX x → ∃ y, TX y :=
by {
  -- Express the constraints and goal of the theorem in Lean.
  sorry 
}

end subgraph_with_max_degree_is_topological_l149_149710


namespace distance_to_directrix_l149_149088

theorem distance_to_directrix (p : ℝ) (h1 : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2 * Real.sqrt 2)) :
  abs (2 - (-1)) = 3 :=
by
  sorry

end distance_to_directrix_l149_149088


namespace distance_product_constant_minimum_distance_to_fixed_point_l149_149851

-- Definition of the hyperbola and its properties
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- Definition of the asymptotes
def asymptote1 (x y : ℝ) : Prop :=
  x - 2 * y = 0

def asymptote2 (x y : ℝ) : Prop :=
  x + 2 * y = 0

-- Distance between a point and a line in ℝ²
noncomputable def distance (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

-- First statement: Prove that the product of the distances from any point P(x₀, y₀)
-- to the two asymptotes is a constant
theorem distance_product_constant (x₀ y₀ : ℝ) (h₀ : hyperbola x₀ y₀) :
  distance x₀ y₀ 1 (-2) 0 * distance x₀ y₀ 1 2 0 = 4 / 5 :=
sorry

-- Second statement: Find the minimum value of |PA| where A = (5, 0)
theorem minimum_distance_to_fixed_point (x₀ y₀ : ℝ) (A : ℝ × ℝ := (5, 0)) (h₀ : hyperbola x₀ y₀) :
  let d := sqrt ((x₀ - A.fst)^2 + (y₀ - A.snd)^2) in
  min d = 2 :=
sorry

end distance_product_constant_minimum_distance_to_fixed_point_l149_149851


namespace sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l149_149635

theorem sum_of_digits_of_greatest_prime_divisor_of_16385_is_13 : 
  ∃ p : ℕ, (p ∣ 16385 ∧ Nat.Prime p ∧ (∀ q : ℕ, q ∣ 16385 → Nat.Prime q → q ≤ p)) ∧ (Nat.digits 10 p).sum = 13 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l149_149635


namespace steve_take_home_pay_l149_149608

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem steve_take_home_pay : 
  (annual_salary - (annual_salary * tax_rate + annual_salary * healthcare_rate + union_dues)) = 27200 := 
by 
  sorry

end steve_take_home_pay_l149_149608


namespace z_one_add_i_eq_neg_one_add_3i_l149_149459

noncomputable def a : ℝ := 2
def z : ℂ := 1 + a * complex.i

theorem z_one_add_i_eq_neg_one_add_3i 
  (h1 : z = 1 + (2 : ℝ) * complex.i)
  (h2 : complex.abs z = real.sqrt 5) : 
  z * (1 + complex.i) = -1 + 3 * complex.i := 
by
  sorry

end z_one_add_i_eq_neg_one_add_3i_l149_149459


namespace area_ratio_of_squares_l149_149271

theorem area_ratio_of_squares (s t : ℝ) (h : 4 * s = 4 * (4 * t)) : (s ^ 2) / (t ^ 2) = 16 :=
by
  sorry

end area_ratio_of_squares_l149_149271


namespace areas_of_triangles_are_equal_l149_149160

variables {A B C D E F : Type*}

-- Assume hexagon exists with some points
variables [convex_hexagon A B C D E F]

-- Given parallel conditions
variables (h1: parallel (line_through A B) (line_through C F))
variables (h2: parallel (line_through C D) (line_through B E))
variables (h3: parallel (line_through E F) (line_through A D))

-- To prove the areas of triangles ACE and BFD are equal
theorem areas_of_triangles_are_equal :
  area (triangle A C E) = area (triangle B F D) :=
by
  sorry

end areas_of_triangles_are_equal_l149_149160


namespace imaginary_part_of_conjugate_of_z_l149_149458

-- Define the condition in the problem
def z (z : ℂ) := i * z = -(1/2) * (1 + i)

-- State the theorem to be proved
theorem imaginary_part_of_conjugate_of_z (z : ℂ) (hz : z z) : z.conj.im = -1/2 := by
  sorry

end imaginary_part_of_conjugate_of_z_l149_149458


namespace number_of_girls_in_class_l149_149887

section
variables (g b : ℕ)

/-- Given the total number of students and the ratio of girls to boys, this theorem states the number of girls in Ben's class. -/
theorem number_of_girls_in_class (h1 : 3 * b = 4 * g) (h2 : g + b = 35) : g = 15 :=
sorry
end

end number_of_girls_in_class_l149_149887


namespace find_number_l149_149713

theorem find_number (x : ℝ) (h : 0.30 * x = 108.0) : x = 360 := 
sorry

end find_number_l149_149713


namespace angle_equality_perpendiculars_l149_149213

theorem angle_equality_perpendiculars (A M P Q K : Point) (hA_ne_M: A ≠ M)
  (angle_acute: ∀ B C, ∠ A B P < π / 2 ∧ ∠ A C Q < π / 2)
  (perp_MP_PA : perpendicular_line MP (line_through A P))
  (perp_MQ_QA : perpendicular_line MQ (line_through A Q))
  (perp_AK_PQ : perpendicular_segment AK (line_segment P Q)) :
  ∠ PAK = ∠ MAQ :=
sorry

end angle_equality_perpendiculars_l149_149213


namespace target_water_percentage_is_two_percent_l149_149203

variable (initial_milk_volume pure_milk_volume : ℕ)
variable (initial_water_percentage target_water_percentage : ℚ)

-- Conditions: Initial milk contains 5% water and we add 15 liters of pure milk
axiom initial_milk_condition : initial_milk_volume = 10
axiom pure_milk_condition : pure_milk_volume = 15
axiom initial_water_condition : initial_water_percentage = 5 / 100

-- Prove that target percentage of water in the milk is 2%
theorem target_water_percentage_is_two_percent :
  target_water_percentage = 2 / 100 := by
  sorry

end target_water_percentage_is_two_percent_l149_149203


namespace slopes_product_constant_l149_149449

noncomputable def pointP_on_circle (x y : ℝ) : Prop :=
  (x + sqrt 6)^2 + y^2 = 32

noncomputable def pointA : Prop :=
  ∃ (x y: ℝ), x = sqrt 6 ∧ y = 0

noncomputable def perpendicular_bisector_condition (P Q A : ℝ × ℝ) : Prop := 
  dist Q P = dist Q A

noncomputable def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 2 = 1

noncomputable def area_condition (E F M : ℝ × ℝ) (area : ℝ) : Prop :=
  let a := dist E F
  let d := dist_from_point_to_line_segment M E F
  (0.5 * a * d) = area

theorem slopes_product_constant 
  (P Q A E F M : ℝ × ℝ)
  (k1 k2 : ℝ) 
  (hp : pointP_on_circle P.1 P.2) 
  (ha : pointA)
  (hap : perpendicular_bisector_condition P Q A)
  (hefc : on_ellipse E.1 E.2 ∧ on_ellipse F.1 F.2)
  (hm : on_ellipse M.1 M.2 ∧ E ≠ M ∧ E ≠ F ∧ F ≠ M)
  (area_eq_four : area_condition E F M 4) :
  k1 * k2 = -1 / 4 :=
sorry

end slopes_product_constant_l149_149449


namespace production_profit_in_10th_year_l149_149404

noncomputable def predicted_production_profit (profits : List ℝ) (year : ℕ) : ℝ :=
  let n := profits.length
  let mean_years := (1 + n : ℝ) / 2
  let mean_profits := profits.sum / n
  let numerator := List.sum (List.map (λ (p : ℝ × ℕ), (p.2 : ℝ - mean_years) * (p.1 - mean_profits)) (List.zip profits (List.range 1 (n+1))))
  let denominator := List.sum (List.map (λ (y : ℕ), (y : ℝ - mean_years)^2) (List.range 1 (n+1)))
  let b := numerator / denominator
  let a := mean_profits - b * mean_years
  a + b * year

theorem production_profit_in_10th_year :
  predicted_production_profit [0.7, 0.8, 1.0, 1.1, 1.4] 10 = 2.19 := 
  sorry

end production_profit_in_10th_year_l149_149404


namespace split_cube_333_l149_149026

noncomputable theory

def split_first (m : ℕ) : ℕ :=
  m * (m - 1) + 1

def is_split_number (m a : ℕ) : Prop :=
  ∃ k : ℕ, a = split_first m + 2 * k

theorem split_cube_333:
  ∃ (m : ℕ), m > 1 ∧ is_split_number m 333 ∧ m = 18 :=
by
  sorry

end split_cube_333_l149_149026


namespace prime_exponent_in_factorial_l149_149591

theorem prime_exponent_in_factorial (p : ℕ) (n : ℕ) (m : ℕ) 
  (hp : p.prime) (hn : 0 < n) (hm : p^m ≤ n ∧ n < p^(m+1)) :
  nat.factors_exp n! p = ∑ i in range (m+1), ⌊n / p^i⌋ :=
by sorry

end prime_exponent_in_factorial_l149_149591


namespace count_three_digit_integers_l149_149862

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

theorem count_three_digit_integers : 
  let num_hundreds := 4 in  -- choices for hundreds place
  let num_tens := 4 in      -- choices for tens place
  let num_units := 5 in     -- choices for units place (including 0)
  num_hundreds * num_tens * num_units = 80 := 
by
  -- Proof is omitted for brevity
  sorry

end count_three_digit_integers_l149_149862


namespace red_blue_distances_equal_l149_149944

theorem red_blue_distances_equal (n : ℕ) (hn : n > 0) :
  ∀ (R B : Finset ℤ),
  R.card = n →
  B.card = n →
  R ∩ B = ∅ →
  R ∪ B = Finset.range (2 * n) →
  (let distances (S : Finset ℤ) := (Finset.image (λ p : ℤ × ℤ, abs (p.fst - p.snd)) (S.product S)).filter (λ x, x ≠ 0) in
  distances R = distances B) :=
by
  intro R B hR hB hRB hUnion
  let f := λ (x : ℤ), ∑ r in R, x ^ r
  let g := λ (x : ℤ), ∑ b in B, x ^ b
  have h1 : f 1 + g 1 = 2 * n := by
    sorry
  have h2 : ∀ z : ℂ, z ^ (2 * n) = 1 → abs (f z) = abs (g z) := by
    sorry
  let red_distances := (Finset.image (λ p : ℤ × ℤ, abs (p.fst - p.snd)) (R.product R)).filter (λ x, x ≠ 0)
  let blue_distances := (Finset.image (λ p : ℤ × ℤ, abs (p.fst - p.snd)) (B.product B)).filter (λ x, x ≠ 0)
  have h3 : red_distances = blue_distances := by
    sorry
  exact h3

end red_blue_distances_equal_l149_149944


namespace work_days_l149_149317

theorem work_days (W : ℕ) (d : ℕ) :
  (d * (W / 40) + 20 * (W / 25) = W) → d = 8 :=
by 
  intro h,
  sorry

end work_days_l149_149317


namespace B_equals_1_2_3_l149_149104

def A : Set ℝ := { x | x^2 ≤ 4 }
def B : Set ℕ := { x | x > 0 ∧ (x - 1:ℝ) ∈ A }

theorem B_equals_1_2_3 : B = {1, 2, 3} :=
by
  sorry

end B_equals_1_2_3_l149_149104


namespace central_angle_of_sector_l149_149358

theorem central_angle_of_sector (r : ℝ) (θ : ℝ) (h_perimeter: 2 * r + θ * r = π * r / 2) : θ = π - 2 :=
sorry

end central_angle_of_sector_l149_149358


namespace prob_no_english_teams_prob_all_english_teams_prob_some_english_teams_l149_149321

-- Define the total number of possible pairings
def total_pairings: ℕ := 105

-- Define the probability calculations for each required condition
def prob_no_english_pairs: ℚ := (4! : ℚ) / total_pairings
def prob_all_english_pairs: ℚ := 18 / total_pairings
def prob_some_english_pairs: ℚ := 72 / total_pairings

-- Theorems to prove the above probabilities
theorem prob_no_english_teams : prob_no_english_pairs = 24 / 105 :=
by sorry

theorem prob_all_english_teams : prob_all_english_pairs = 18 / 105 :=
by sorry

theorem prob_some_english_teams : prob_some_english_pairs = 72 / 105 :=
by sorry

end prob_no_english_teams_prob_all_english_teams_prob_some_english_teams_l149_149321


namespace reciprocal_neg_one_div_2022_l149_149275

theorem reciprocal_neg_one_div_2022 : (1 / (-1 / 2022)) = -2022 :=
by sorry

end reciprocal_neg_one_div_2022_l149_149275


namespace altitude_eqn_equidistant_eqn_l149_149089

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definition of a line in the form Ax + By + C = 0
structure Line :=
  (A B C : ℝ)
  (non_zero : A ≠ 0 ∨ B ≠ 0)

-- Equation of line l1 (altitude to side BC)
def l1 : Line := { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) }

-- Equation of line l2 (passing through C, equidistant from A and B), two possible values
def l2a : Line := { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) }
def l2b : Line := { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) }

-- Prove the equations for l1 and l2 are correct given the points A, B, and C
theorem altitude_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l1 = { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) } := sorry

theorem equidistant_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l2a = { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) } ∨
  l2b = { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) } := sorry

end altitude_eqn_equidistant_eqn_l149_149089


namespace heart_digit_proof_l149_149118

noncomputable def heart_digit : ℕ := 3

theorem heart_digit_proof (heartsuit : ℕ) (h : heartsuit * 9 + 6 = heartsuit * 10 + 3) : 
  heartsuit = heart_digit := 
by
  sorry

end heart_digit_proof_l149_149118


namespace sum_of_digits_eq_24_l149_149333

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.to_string.to_list.foldl (λ acc c, acc + c.to_nat - '0'.to_nat) 0

theorem sum_of_digits_eq_24 (x : ℕ) (h₁ : 100 ≤ x ∧ x < 1000) (h₂ : 1000 ≤ x + 32 ∧ x + 32 < 10000) (hx : is_palindrome x) (hx32 : is_palindrome (x + 32)) :
  sum_of_digits x = 24 :=
  sorry

end sum_of_digits_eq_24_l149_149333


namespace cubic_third_root_l149_149327

theorem cubic_third_root (f : ℝ → ℝ) (line_eq : ℝ → ℝ) (x1 x2 x3 : ℝ) :
  f = λ x, x^3 + x + 2014 →
  line_eq = λ x, 877 * x - 7506 →
  f 20 = 10034 →
  f 14 = 4772 →
  x1 = 20 →
  x2 = 14 →
  x3 = -34 →
  (x1 - 20) * (x1 - 14) * (x1 - (-34)) = 0 := sorry

end cubic_third_root_l149_149327


namespace students_in_photo_l149_149638

theorem students_in_photo (m n : ℕ) (h1 : n = m + 5) (h2 : n = m + 5 ∧ m = 3) : 
  m * n = 24 :=
by
  -- h1: n = m + 5    (new row is 4 students fewer)
  -- h2: m = 3        (all rows have the same number of students after rearrangement)
  -- Prove m * n = 24
  sorry

end students_in_photo_l149_149638


namespace Frank_is_14_l149_149247

variable {d e f : ℕ}

theorem Frank_is_14
  (h1 : d + e + f = 30)
  (h2 : f - 5 = d)
  (h3 : e + 2 = 3 * (d + 2) / 4) :
  f = 14 :=
sorry

end Frank_is_14_l149_149247


namespace count_integers_satisfying_ineqs_l149_149025

theorem count_integers_satisfying_ineqs :
  {n : ℤ | (n + 1: ℝ)^(1/2) ≤ (5 * n - 7: ℝ)^(1/2) ∧ (5 * n - 7: ℝ)^(1/2) < (3 * n + 4: ℝ)^(1/2)}.to_finset.card = 4 :=
by
  sorry

end count_integers_satisfying_ineqs_l149_149025


namespace displacement_of_point_mass_l149_149354

open Real

theorem displacement_of_point_mass : 
  (∫ t in 1..2, (t^2 - t + 2)) = 17 / 6 :=
by 
  sorry

end displacement_of_point_mass_l149_149354


namespace platform_length_l149_149334

theorem platform_length (train_length : ℕ) (tree_cross_time : ℕ) (platform_cross_time : ℕ) (platform_length : ℕ)
  (h_train_length : train_length = 1200)
  (h_tree_cross_time : tree_cross_time = 120)
  (h_platform_cross_time : platform_cross_time = 160)
  (h_speed_calculation : (train_length / tree_cross_time = 10))
  : (train_length + platform_length) / 10 = platform_cross_time → platform_length = 400 :=
sorry

end platform_length_l149_149334


namespace triangle_is_right_triangle_l149_149528

theorem triangle_is_right_triangle (BC AC AB : ℝ) (h1 : BC = 6) (h2 : AC = 8) (h3 : AB = 10) :
  (BC^2 + AC^2 = AB^2) → right_triangle ABC := by
  sorry

end triangle_is_right_triangle_l149_149528


namespace mixture_percentage_is_36_l149_149330

-- Define the volumes and percentages
def volume1 : ℝ := 6
def percentage1 : ℝ := 0.20
def volume2 : ℝ := 4
def percentage2 : ℝ := 0.60

-- Define the total volume
def total_volume : ℝ := volume1 + volume2

-- Calculate the amounts taken from each solution
def amount1 : ℝ := percentage1 * volume1
def amount2 : ℝ := percentage2 * volume2

-- Total mixture volume
def mixture_volume : ℝ := amount1 + amount2

-- Calculate the percentage of the mixture relative to the original total volume
def mixture_percentage : ℝ := (mixture_volume / total_volume) * 100

-- Theorem stating the mixture percentage is 36%
theorem mixture_percentage_is_36 : mixture_percentage = 36 := by
  sorry

end mixture_percentage_is_36_l149_149330


namespace sales_of_stationery_accessories_l149_149620

def percentage_of_sales_notebooks : ℝ := 25
def percentage_of_sales_markers : ℝ := 40
def total_sales_percentage : ℝ := 100

theorem sales_of_stationery_accessories : 
  percentage_of_sales_notebooks + percentage_of_sales_markers = 65 → 
  total_sales_percentage - (percentage_of_sales_notebooks + percentage_of_sales_markers) = 35 :=
by
  sorry

end sales_of_stationery_accessories_l149_149620


namespace rotated_triangle_volume_l149_149584

noncomputable def volume_of_rotated_triangle 
  (R : ℝ) (α : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  (2 / 3) * π * R^3 * sin (4 * α) * sin (2 * α)

theorem rotated_triangle_volume
  (R : ℝ) (α : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < π / 2) 
  (AB : ℝ) (hAB : AB = 2 * R)
  (semicircle : ∀ (x y : ℝ), x^2 + y^2 = R^2 → y ≥ 0) 
  (CD_parallel : ∀ (C D : ℝ × ℝ), C.2 = D.2 → C.1 ≠ D.1)
  (inscribed_angle : ∀ (A C D : ℝ × ℝ), A.1 = -R ∧ A.2 = 0 ∧ C.1 = R * cos α ∧ C.2 = R * sin α ∧ D.1 = R * cos α ∧ D.2 = -R * sin α) :
  volume_of_rotated_triangle R α hR hα = (2 / 3) * π * R^3 * sin (4 * α) * sin (2 * α) :=
by 
  -- Proof omitted
  sorry

end rotated_triangle_volume_l149_149584


namespace variance_transformation_l149_149622

theorem variance_transformation (a1 a2 a3 : ℝ) 
  (h1 : (a1 + a2 + a3) / 3 = 4) 
  (h2 : ((a1 - 4)^2 + (a2 - 4)^2 + (a3 - 4)^2) / 3 = 3) : 
  ((3 * a1 - 2 - (3 * 4 - 2))^2 + (3 * a2 - 2 - (3 * 4 - 2))^2 + (3 * a3 - 2 - (3 * 4 - 2))^2) / 3 = 27 := 
sorry

end variance_transformation_l149_149622


namespace slower_train_cross_time_l149_149669

-- Definitions and assumptions
def speed_train_1 := 315 -- km/hr
def speed_train_2 := 135 -- km/hr
def length_train_1 := 1.65 -- km
def length_train_2 := 1.35 -- km

-- Theorem statement
theorem slower_train_cross_time : 
  let v1 := speed_train_1 in
  let v2 := speed_train_2 in
  let l1 := length_train_1 in
  let l2 := length_train_2 in
  let relative_speed := (v1 + v2) * 1000 / 3600 in -- in m/s
  let combined_length := (l1 + l2) * 1000 in -- in m
  (combined_length / relative_speed) = 24 := 
by
  sorry

end slower_train_cross_time_l149_149669


namespace tangent_line_eq_l149_149878

def f (x : ℝ) (f'1 f'2 : ℝ) : ℝ :=
  - (1 / 3) * x^3 + (1 / 2) * f'1 * x^2 - f'2 * x + 5

theorem tangent_line_eq (f'1 f'2 : ℝ)
  (h1 : -1 + f'1 - f'2 = 1)
  (h2 : -4 + 2 * f'1 - f'2 = -1) :
  ∀ (x y : ℝ), x - y + 5 = 0 ↔ f 0 f'1 f'2 = 5 ∧ (∇(f x f'1 f'2)).at (0, 5) = (1, 0) := sorry

end tangent_line_eq_l149_149878


namespace problem1_problem2_problem3_l149_149836

noncomputable def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * tan (-α + 3 * π)) / 
  (cos (-α - π) * sin (-α - π))

-- Problem 1: Simplification of f(α)
theorem problem1 (α : ℝ) (h : π < α ∧ α < 3 * π) : f α = tan α :=
by
  sorry

-- Problem 2: Finding f(α) given condition
theorem problem2 (α : ℝ) (hα : π < α ∧ α < 3 * π) (h : cos (α - 3 * π / 2) = 1 / 5) :
  f α = (-sqrt 6) / 12 :=
by
  sorry

-- Problem 3: Finding f(α) given specific α
theorem problem3 : f (-1860 * π / 180) = -sqrt 3 :=
by
  sorry

end problem1_problem2_problem3_l149_149836


namespace max_min_m_l149_149106

theorem max_min_m (a b c m : ℝ) (h1 : 3 * a + 2 * b + c = 5) (h2 : 2 * a + b - 3 * c = 1)
(h3 : a ≥ 0) (h4 : b ≥ 0) (h5 : c ≥ 0) : 
    (∀ c, 3/7 ≤ c ∧ c ≤ 7/11 → -5/7 ≤ 3*c - 2 ∧ 3*c - 2 ≤ -1/11) → 
    let m := 3 * a + b - 7 * c in 
    True :=
by sorry

end max_min_m_l149_149106


namespace city_A_fare_higher_than_city_B_l149_149897

def fare_in_city_A (x : ℝ) : ℝ :=
  10 + 2 * (x - 3)

def fare_in_city_B (x : ℝ) : ℝ :=
  8 + 2.5 * (x - 3)

theorem city_A_fare_higher_than_city_B (x : ℝ) (h : x > 3) :
  fare_in_city_A x > fare_in_city_B x → 3 < x ∧ x < 7 :=
by
  sorry

end city_A_fare_higher_than_city_B_l149_149897


namespace runners_meet_after_800_seconds_l149_149424

theorem runners_meet_after_800_seconds :
  ∀ (t : ℕ),
    (3 * t) % 400 = (3.5 * t) % 400 ∧
    (3.5 * t) % 400 = (4 * t) % 400 ∧
    (4 * t) % 400 = (4.5 * t) % 400 →
    t = 800 :=
by
  -- conditions stated as congruences
  assume t,
  assume h : (3 * t) % 400 = (3.5 * t) % 400 ∧
              (3.5 * t) % 400 = (4 * t) % 400 ∧
              (4 * t) % 400 = (4.5 * t) % 400,
  -- proof steps skipped
  sorry

end runners_meet_after_800_seconds_l149_149424


namespace probability_sum_odd_l149_149269

open_locale big_operators
open finset

-- Define the finset of numbers
def numbers : finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define what it means for a sequence to be an odd position product
def odd_product (x y z : ℕ) : Prop := odd (x * y * z)

-- Define what it means for a sequence to be an even position product
def even_product (x y z : ℕ) : Prop := ¬odd (x * y * z)

-- Define what it means for a product of two sequences to sum to an odd number
def sum_is_odd (a b c d e f : ℕ) : Prop := odd ((a * b * c) + (d * e * f))

-- Probability calculation setup
def favorable_outcomes : finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  {t ∈ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers |
    sum_is_odd t.1.1 t.1.2 t.1.3 t.2.1 t.2.2 t.2.3}

def total_outcomes : finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers

theorem probability_sum_odd :
  probability (favorable_outcomes : finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)) = 1 / 10 :=
begin
  sorry
end

end probability_sum_odd_l149_149269


namespace hyperbola_asymptotes_l149_149471

-- conditions
def hyperbola_asymptotes_eq (a b : ℝ): Prop :=
  ∀ x y, (x^2) / (a^2) - (y^2) / (b^2) = 1 → (y = sqrt 3 * x ∨ y = - sqrt 3 * x)

-- Assume conditions
theorem hyperbola_asymptotes
  (a b c : ℝ)
  (h1 : 2 * a = 2)
  (h2 : 2 * c = 4)
  (h3 : b = sqrt (c^2 - a^2)):
  hyperbola_asymptotes_eq a b :=
by
  sorry

end hyperbola_asymptotes_l149_149471


namespace ramesh_transport_cost_l149_149221

theorem ramesh_transport_cost :
  ∃ (P T : ℝ), 
    0.80 * P = 12500 ∧
    12500 + T + 250 = 18560 ∧
    T = 5810 :=
by {
  let P := 15625,
  let T := 5810,
  have h_labelled_price : 0.80 * P = 12500 := by norm_num,
  have h_total_cost : 12500 + T + 250 = 18560 := by norm_num,
  exact ⟨P, T, ⟨h_labelled_price, h_total_cost, rfl⟩⟩,
}

end ramesh_transport_cost_l149_149221


namespace balls_in_boxes_count_l149_149116

theorem balls_in_boxes_count : 
  (number_of_ways_to_distribute_balls : ℕ) := 92 :=
sorry

end balls_in_boxes_count_l149_149116


namespace area_of_new_shape_l149_149361

noncomputable def unit_equilateral_triangle_area : ℝ :=
  (1 : ℝ)^2 * Real.sqrt 3 / 4

noncomputable def area_removed_each_step (k : ℕ) : ℝ :=
  3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def total_removed_area : ℝ :=
  ∑' k, 3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def final_area := unit_equilateral_triangle_area - total_removed_area

theorem area_of_new_shape :
  final_area = Real.sqrt 3 / 10 := sorry

end area_of_new_shape_l149_149361


namespace intersecting_lines_l149_149911

theorem intersecting_lines :
  ∃ (s : ℕ) (k : Fin s → ℕ),
    (s = 6) ∧
    (k 0 + k 1 + k 2 + k 3 + k 4 + k 5 = 17) ∧
    (k 0 ^ 2 + k 1 ^ 2 + k 2 ^ 2 + k 3 ^ 2 + k 4 ^ 2 + k 5 ^ 2 = 87) :=
begin
  let k : Fin 6 → ℕ := λ i, match i.1 with
    | 0 => 8
    | 1 => 4
    | 2 => 2
    | 3 => 1
    | 4 => 1
    | _ => 1 -- For i.1 >= 5
  end,
  use [6, k],
  dsimp [k],
  split,
  { 
    refl
  },
  split,
  { 
    norm_num
  },
  { 
    norm_num
  }
end

end intersecting_lines_l149_149911


namespace p_sufficient_but_not_necessary_q_l149_149059

theorem p_sufficient_but_not_necessary_q :
  ∀ x : ℝ, (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros x hx
  cases hx with h1 h2
  apply And.intro
  apply lt_of_lt_of_le h1
  linarith
  apply h2

end p_sufficient_but_not_necessary_q_l149_149059


namespace correct_propositions_l149_149107

-- Definitions based directly on conditions
variables {α β : Type} [plane α] [plane β]
variables {m n : Type} [line m] [line n]

-- Propositions
def prop1 : Prop :=
  (parallel m n) ∧ (perpendicular m α) → perpendicular n α

def prop2 : Prop :=
  (perpendicular m α) ∧ (perpendicular m β) → parallel α β

def prop3 : Prop :=
  (perpendicular m α) ∧ (parallel m n) ∧ (subset n β) → perpendicular α β

-- Problem statement
theorem correct_propositions : (prop1 ∧ prop2 ∧ prop3) ↔ (3 = 3) :=
by
  split
  { intro h, -- introduction of the hypothesis
    exact eq.refl 3, -- simpler proof showing equivalence
  }
  { intro h, -- no need to prove as the other way
    sorry -- allows to skip the proof
  }

end correct_propositions_l149_149107


namespace sum_binom_eq_l149_149593

theorem sum_binom_eq: ∀ (n : ℕ), 
  (∑ k in Finset.range (n + 1), Nat.choose (2 * n) k) = 2^(2 * n - 1) + ((2 * n).factorial / (2 * n.factorial * n.factorial)) := 
by 
  sorry

end sum_binom_eq_l149_149593


namespace fraction_of_journey_asleep_l149_149737

theorem fraction_of_journey_asleep (x y : ℝ) (hx : x > 0) (hy : y = x / 3) :
  y / x = 1 / 3 :=
by
  sorry

end fraction_of_journey_asleep_l149_149737


namespace find_c_and_M_l149_149187

noncomputable def g₁ (x : ℝ) : ℝ := real.sqrt (x^2 + 3)

noncomputable def g (n : ℕ) (x : ℝ) : ℝ := 
  nat.rec_on n 
    (g₁ x)
    (λ k f, f (real.sqrt (k^2 + x)))

theorem find_c_and_M : 
  ∃ (c : ℝ) (M : ℕ), M = 5 ∧ c = -25 ∧ ∀ x : ℝ, g M x = g M c :=
by
  sorry

end find_c_and_M_l149_149187


namespace find_angle_B_l149_149135

theorem find_angle_B (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = 45) 
  (h2 : a = 6) 
  (h3 : b = 3 * Real.sqrt 2)
  (h4 : ∀ A' B' C' : ℝ, 
        ∃ a' b' c' : ℝ, 
        (a' = a) ∧ (b' = b) ∧ (A' = A) ∧ 
        (b' < a') → (B' < A') ∧ (A' = 45)) :
  B = 30 :=
by
  sorry

end find_angle_B_l149_149135


namespace robot_swap_eventually_non_swappable_l149_149149

theorem robot_swap_eventually_non_swappable (n : ℕ) (a : Fin n → ℕ) :
  ∃ t : ℕ, ∀ i : Fin (n - 1), ¬ (a (⟨i, sorry⟩ : Fin n) > a (⟨i + 1, sorry⟩ : Fin n)) ↔ n > 1 :=
sorry

end robot_swap_eventually_non_swappable_l149_149149


namespace probability_integer_division_l149_149698

def set_x := {78, 910}
def set_y := {23, 45}

theorem probability_integer_division : 
  (∀ x ∈ set_x, ∀ y ∈ set_y, ¬∃ k: ℤ, x = k * y) → (0 : ℝ) = 0 := 
by sorry

end probability_integer_division_l149_149698


namespace no_labeling_possible_45_gon_l149_149379

def vertex_labeling_possible (n : ℕ) (d : ℕ) : Prop :=
  ∃ (labeling : fin n → fin d), ∀ i j : fin d, i ≠ j → ∃ (k : fin n), labeling k = i ∧ labeling ((k + 1) % n) = j

theorem no_labeling_possible_45_gon : ¬ vertex_labeling_possible 45 10 :=
sorry

end no_labeling_possible_45_gon_l149_149379


namespace hybrids_with_full_headlights_l149_149143

theorem hybrids_with_full_headlights (total_cars hybrids_percentage one_headlight_percentage : ℝ) 
  (hc : total_cars = 600) (hp : hybrids_percentage = 0.60) (ho : one_headlight_percentage = 0.40) : 
  total_cars * hybrids_percentage - total_cars * hybrids_percentage * one_headlight_percentage = 216 := by
  sorry

end hybrids_with_full_headlights_l149_149143


namespace collinear_distance_sum_l149_149721

-- Definitions of the conditions
def diameter_smaller : ℕ := 500
def diameter_larger : ℕ := 1700
def starting_point_same : Bool := true
def speed_same : Bool := true
def track_1_direction : String := "clockwise"
def track_2_direction : String := "clockwise"
def externally_tangent_at_A : Bool := true

-- Theorem statement
theorem collinear_distance_sum (m n : ℕ) 
  (h1 : m + n = 501)
  (h2 : Nat.gcd m n = 1)
  (h3 : externally_tangent_at_A)
  (h4 : starting_point_same)
  (h5 : speed_same)
  (h6 : track_1_direction = "clockwise")
  (h7 : track_2_direction = "clockwise") : 
  ((diameter_smaller + diameter_larger) * Nat.pi / Nat.lcm 500 1700 = m * Nat.pi / n) :=
  sorry

end collinear_distance_sum_l149_149721


namespace largest_constant_ineq_l149_149012

theorem largest_constant_ineq (a b c d e : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) :
  (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e)) + sqrt (e / (a + b + c + d))) > 2 :=
sorry

end largest_constant_ineq_l149_149012


namespace polynomial_root_a_b_sum_l149_149562

noncomputable def a_b_sum (a b : ℝ) : ℝ :=
  a + b

theorem polynomial_root_a_b_sum : 
  ∀ (a b : ℝ), 
    (∃ r : ℂ, r = 2 + complex.I * real.sqrt 2 ∧ polynomial a b r * r^2 = 0) →
    a_b_sum a b = 14 :=
by
  sorry

end polynomial_root_a_b_sum_l149_149562


namespace probability_A_plus_B_l149_149515

def event_A := {x : ℕ | x < 5 ∧ x % 2 = 0}
def event_B := {x : ℕ | x < 5}

theorem probability_A_plus_B :
  let P : set ℕ → ℚ := λ s, s.card / 6 in
  disjoint event_A event_B ∧ P event_A = 1 / 3 ∧ P event_B = 2 / 3 →
  P (event_A ∪ event_B) = 2 / 3 :=
by
  sorry

end probability_A_plus_B_l149_149515


namespace quotient_of_cars_per_hour_l149_149952

theorem quotient_of_cars_per_hour :
  let v_R : ℕ := 50 -- Speed of the right lane in km/h
  let v_L : ℕ := v_R + 10 -- Speed of the left lane in km/h
  let car_length : ℕ := 5 -- Car length in meters
  let total_distance_right_lane := v_R * 1000 -- Total distance right lane cars travel per hour in meters
  let total_distance_left_lane := v_L * 1000 -- Total distance left lane cars travel per hour in meters
  let car_lengths_right := 5 -- Number of car lengths for right lane safety rule
  let car_lengths_left := 6 -- Number of car lengths for left lane safety rule
  let unit_distance_right := car_lengths_right * car_length + car_length -- Total meters per unit in right lane
  let unit_distance_left := car_lengths_left * car_length + car_length -- Total meters per unit in left lane
  let cars_right_lane := total_distance_right_lane / unit_distance_right -- Cars passing in right lane
  let cars_left_lane := total_distance_left_lane / unit_distance_left -- Cars passing in left lane
  let M := cars_right_lane + cars_left_lane -- Total cars passing
  M / 10 = 338 :=
by {
  sorry,
}

end quotient_of_cars_per_hour_l149_149952


namespace sum_elements_A_l149_149797

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def f (x : ℝ) : ℤ := floor (2 * x) + floor (4 * x) + floor (8 * x)

def A : Set ℤ := {y | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1/2 ∧ y = f x}

theorem sum_elements_A : ∑ y in A.to_finset, y = 15 := by
  sorry

end sum_elements_A_l149_149797


namespace jeans_price_increase_l149_149693

theorem jeans_price_increase (M R C : ℝ) (hM : M = 100) 
  (hR : R = M * 1.4)
  (hC : C = R * 1.1) : 
  (C - M) / M * 100 = 54 :=
by
  sorry

end jeans_price_increase_l149_149693


namespace general_term_sequence_a_l149_149881

def sequence_a (n : ℕ) : ℕ → ℝ
| 0       := 5
| (n + 1) := (3 * sequence_a n - 1) / (-sequence_a n + 3)

theorem general_term_sequence_a (n : ℕ) :
  sequence_a n = (2 * (2^n + 1)) / (1 - 2^n) :=
sorry

end general_term_sequence_a_l149_149881


namespace weight_of_b_l149_149623

-- Definitions based on conditions
variables (A B C : ℝ)

def avg_abc := (A + B + C) / 3 = 45
def avg_ab := (A + B) / 2 = 40
def avg_bc := (B + C) / 2 = 44

-- The theorem to prove
theorem weight_of_b (h1 : avg_abc A B C) (h2 : avg_ab A B) (h3 : avg_bc B C) :
  B = 33 :=
sorry

end weight_of_b_l149_149623


namespace polynomial_divisibility_l149_149771

theorem polynomial_divisibility (n : ℕ) (a : ℝ) :
  (∀ x, x^n - a * x^(n-1) + a * x - 1 = (x-1)^2 * (some_poly x)) →
  a = n / (n-1) :=
by
  sorry

end polynomial_divisibility_l149_149771


namespace remainder_of_k_l149_149683

theorem remainder_of_k {k : ℕ} (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 8 = 7) (h4 : k % 11 = 3) (h5 : k < 168) :
  k % 13 = 8 := 
sorry

end remainder_of_k_l149_149683


namespace hyperbola_perimeter_proof_l149_149210

noncomputable def hyperbola_perimeter : ℝ :=
  let a := 8
  let c := sqrt (8^2 + 6^2)
  let PF := 3
  let PF' := c + a - PF
  let FF' := 2 * c
  PF + PF' + FF'

theorem hyperbola_perimeter_proof :
  ∀ x y : ℝ, (y^2 / 64 - x^2 / 36 = 1) → PF = 3 → hyperbola_perimeter = 42 :=
by
  intros x y h_eq h_PF
  let a := 8
  let c := sqrt (8^2 + 6^2)
  have PF' := c + a - 3
  have FF' := 2 * c
  calc
    PF + PF' + FF' = 3 + (19 : ℝ) + 20 : by sorry
               ... = 42 : by norm_num

end hyperbola_perimeter_proof_l149_149210


namespace problem_F_decreasing_l149_149564

theorem problem_F_decreasing {f : ℝ → ℝ} (h_deriv : ∀ x, deriv f x < f x) :
  f 2 < exp 2 * f 0 ∧ f 2012 < exp 2012 * f 0 :=
by
  let F : ℝ → ℝ := λ x, f x / exp x
  have h_F_decreasing : ∀ x, deriv F x < 0 := sorry
  have h_F0_gt_F2 : F 0 > F 2 := sorry
  have h_F0_gt_F2012 : F 0 > F 2012 := sorry
  have f_2_lt_e2_f0 : f 2 < exp 2 * f 0 := sorry
  have f_2012_lt_e2012_f0 : f 2012 < exp 2012 * f 0 := sorry
  exact ⟨f_2_lt_e2_f0, f_2012_lt_e2012_f0⟩

end problem_F_decreasing_l149_149564


namespace line_equation_through_point_perpendicular_l149_149726

theorem line_equation_through_point_perpendicular (P : ℝ × ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  P = (1, 1) ∧ a * 1 + b * 2 + c = 0 → (∃ d : ℝ, 2 * 1 - 1 + d = 0 ∧ l = {p : ℝ × ℝ | 2 * p.1 - p.2 = d}) → l = {p : ℝ × ℝ | 2 * p.1 - p.2 = -1} :=
by
  intro h1
  cases h1 with hP hperpen
  cases hP
  cases hperpen with c1 hline
  simp at c1
  use (-1)
  ext p
  rw [hline, c1]
  refl


end line_equation_through_point_perpendicular_l149_149726


namespace median_of_data_set_l149_149433

open List

def data_set : List ℕ := [96, 89, 92, 95, 98]

def ordered_data_set : List ℕ := sort (· ≤ ·) data_set

theorem median_of_data_set :
  (ordered_data_set.length > 0) →
  ordered_data_set = [89, 92, 95, 96, 98] →
  ordered_data_set.nth ((ordered_data_set.length - 1) / 2) = some 95 :=
by
  intros h_length h_ordered
  rw [h_ordered]
  -- by calculating (ordered_data_set.length - 1) / 2 = 2 since length is 5.
  -- the nth index (counting from 0) is the 3rd element.
  simp
  sorry

end median_of_data_set_l149_149433


namespace trajectory_of_vertex_A_l149_149899

theorem trajectory_of_vertex_A (B C : ℝ × ℝ) (P Q : ℝ × ℝ) (A : ℝ × ℝ) :
  B = (0, -1) → C = (0, 1) → 
  (↑∑ i in [A, B, C], if i = A then 1 else (if i = B then 1 else 1)) = (0, 0) → 
  (dist Q A = dist Q B ∧ dist Q B = dist Q C ∧ dist Q A = dist Q C) → 
  (P.1 - Q.1) * (B.2 - C.2) = (P.2 - Q.2) * (B.1 - C.1) → 
  ∃ x y, A = (x, y) ∧ (x ≠ 0) ∧ (x^2 / 3 + y^2 = 1) :=
begin
  intros hB hC hP hQ hPQ,
  use (A.1, A.2),
  split,
  { exact rfl },
  split,
  { intro hx, apply hQ.1, rw [hx], exact hQ.2.1 },
  { sorry }
end

end trajectory_of_vertex_A_l149_149899


namespace cylinder_volume_pairs_l149_149953

theorem cylinder_volume_pairs (t : ℕ) : 
    t = 30 → 
    ((∃ (a b : ℕ), 2 * (sqrt 2)^2 * a + 5 * (sqrt 5)^2 * b = 10 * π * t ∧ a > 0 ∧ b > 0) ↔ 
     (count_pairs (λ a b, 2 * a + 5 * b = 10 * t ∧ a > 0 ∧ b > 0) = 29)) :=
    begin
    sorry
end

end cylinder_volume_pairs_l149_149953


namespace line_passes_through_point_l149_149220

theorem line_passes_through_point (k : ℝ) :
  ∀ k : ℝ, (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 :=
by
  intro k
  sorry

end line_passes_through_point_l149_149220


namespace bridge_crossing_possible_l149_149661

/-- 
  There are four people A, B, C, and D. 
  The time it takes for each of them to cross the bridge is 2, 4, 6, and 8 minutes respectively.
  No more than two people can be on the bridge at the same time.
  Prove that it is possible for all four people to cross the bridge in 10 minutes.
--/
theorem bridge_crossing_possible : 
  ∃ (cross : ℕ → ℕ), 
  cross 1 = 2 ∧ cross 2 = 4 ∧ cross 3 = 6 ∧ cross 4 = 8 ∧
  (∀ (t : ℕ), t ≤ 2 → cross 1 + cross 2 + cross 3 + cross 4 = 10) :=
by
  sorry

end bridge_crossing_possible_l149_149661


namespace sum_ABC_eq_7_base_8_l149_149117

/-- Lean 4 statement for the problem.

A, B, C: are distinct non-zero digits less than 8 in base 8, and
A B C_8 + B C_8 = A C A_8 holds true.
-/
theorem sum_ABC_eq_7_base_8 :
  ∃ (A B C : ℕ), A < 8 ∧ B < 8 ∧ C < 8 ∧ 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (A * 64 + B * 8 + C) + (B * 8 + C) = A * 64 + C * 8 + A ∧
  A + B + C = 7 :=
by { sorry }

end sum_ABC_eq_7_base_8_l149_149117


namespace distance_between_vertices_l149_149922

theorem distance_between_vertices : 
  let C := (2, 1)
  let D := (-3, 4)
  let distance := Real.sqrt ((2 - (-3))^2 + (1 - 4)^2)
  (C.vertex == (2, 1)) ∧ (D.vertex == (-3, 4)) ∧ (distance == Real.sqrt 34) := sorry

end distance_between_vertices_l149_149922


namespace reach_destination_in_time_l149_149363

theorem reach_destination_in_time : 
  (∀ A B C : Type, 
  (∀ a b c : A → ℕ, (walk_speed : ℕ = 5) → 
  (car_speed : ℕ = 50) → 
  (destination : ℕ = 62) →
  (max_time : ℕ = 3)) → 
  ∃ A_ B_ C_ : A, ∃ t : ℝ, t < 3 := sorry

end reach_destination_in_time_l149_149363


namespace statement_a_statement_c_statement_d_l149_149687

-- Statement A: If z = i^3, then \overline{z} = i
theorem statement_a (z : ℂ) (h : z = complex.I ^ 3) : complex.conj z = complex.I := 
sorry

-- Statement C: If z = a + ai (a ∈ ℝ), then z cannot be a purely imaginary number
theorem statement_c (a : ℝ) (h : a ≠ 0) (z : ℂ) (h_eq : z = a + a * complex.I) : ¬ (z.im = 0) := 
sorry

-- Statement D: If the complex number z satisfies 1/z ∈ ℝ, then z ∈ ℝ
theorem statement_d (z : ℂ) (h : (1/z).im = 0) : z.im = 0 :=
sorry

end statement_a_statement_c_statement_d_l149_149687


namespace element_in_set_l149_149863

variable (A : Set ℕ) (a b : ℕ)
def condition : Prop := A = {a, b, 1}

theorem element_in_set (h : condition A a b) : 1 ∈ A :=
by sorry

end element_in_set_l149_149863


namespace hospital_staff_total_l149_149510

def initial_doctors := 11
def initial_nurses := 18
def initial_medical_assistants := 9
def initial_interns := 6

def doctors_quit := 5
def nurses_quit := 2
def medical_assistants_quit := 3
def nurses_transferred := 2
def interns_transferred := 4
def doctors_vacation := 4
def nurses_vacation := 3

def new_doctors := 3
def new_nurses := 5

def remaining_doctors := initial_doctors - doctors_quit - doctors_vacation
def remaining_nurses := initial_nurses - nurses_quit - nurses_transferred - nurses_vacation
def remaining_medical_assistants := initial_medical_assistants - medical_assistants_quit
def remaining_interns := initial_interns - interns_transferred

def final_doctors := remaining_doctors + new_doctors
def final_nurses := remaining_nurses + new_nurses
def final_medical_assistants := remaining_medical_assistants
def final_interns := remaining_interns

def total_staff := final_doctors + final_nurses + final_medical_assistants + final_interns

theorem hospital_staff_total : total_staff = 29 := by
  unfold total_staff
  unfold final_doctors
  unfold final_nurses
  unfold final_medical_assistants
  unfold final_interns
  unfold remaining_doctors
  unfold remaining_nurses
  unfold remaining_medical_assistants
  unfold remaining_interns
  unfold initial_doctors initial_nurses initial_medical_assistants initial_interns
  unfold doctors_quit nurses_quit medical_assistants_quit nurses_transferred interns_transferred
  unfold doctors_vacation nurses_vacation
  unfold new_doctors new_nurses
  sorry

end hospital_staff_total_l149_149510


namespace line_intersection_range_b_l149_149129

theorem line_intersection_range_b 
  (b : ℝ) 
  (line : ℝ → ℝ := λ x, x + b)
  (curve : ℝ × ℝ → Prop := λ p, (p.1 - 2)^2 + (p.2 - 3)^2 = 4)
  (cond1 : ∃ x, 0 ≤ x ∧ x ≤ 4 ∧ ∃ y, 1 ≤ y ∧ y ≤ 3 ∧ line x = y)
  (cond2 : ∃ x y, curve (x, y) ∧ line x = y) : 
  b ∈ (Icc (1 - 2 * real.sqrt 2) 3) :=
sorry

end line_intersection_range_b_l149_149129


namespace compare_fractions_l149_149386

theorem compare_fractions : (-8 / 21: ℝ) > (-3 / 7: ℝ) :=
sorry

end compare_fractions_l149_149386


namespace clothing_price_l149_149173

theorem clothing_price
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (price_piece_1 : ℕ)
  (price_piece_2 : ℕ)
  (num_remaining_pieces : ℕ)
  (total_remaining_pieces_price : ℕ)
  (price_remaining_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  price_piece_1 = 49 →
  price_piece_2 = 81 →
  num_remaining_pieces = 5 →
  total_spent = price_piece_1 + price_piece_2 + total_remaining_pieces_price →
  total_remaining_pieces_price = price_remaining_piece * num_remaining_pieces →
  price_remaining_piece = 96 :=
by
  intros h_total_spent h_num_pieces h_price_piece_1 h_price_piece_2 h_num_remaining_pieces h_total_remaining_price h_price_remaining_piece
  sorry

end clothing_price_l149_149173


namespace new_triangle_exists_l149_149968

-- Defining the conditions: Existence of triangles with given sides
def is_triangle (a b c : ℝ) := a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def exists_new_triangle (a b c A B C : ℝ) : Prop :=
  is_triangle (sqrt (a^2 + A^2)) (sqrt (b^2 + B^2)) (sqrt (c^2 + C^2))

theorem new_triangle_exists
  (a b c A B C : ℝ)
  (h1 : is_triangle a b c)
  (h2 : is_triangle A B C) :
  exists_new_triangle a b c A B C :=
sorry

end new_triangle_exists_l149_149968


namespace find_5b_l149_149492

-- Define variables and conditions
variables (a b : ℝ)
axiom h1 : 6 * a + 3 * b = 0
axiom h2 : a = b - 3

-- State the theorem to prove
theorem find_5b : 5 * b = 10 :=
sorry

end find_5b_l149_149492


namespace tony_average_speed_l149_149285

theorem tony_average_speed :
  let speeds := [50, 62, 73, 70, 40] in
  (List.sum speeds : ℕ) / speeds.length = 59 := by
  sorry

end tony_average_speed_l149_149285


namespace average_sleep_hours_l149_149111

theorem average_sleep_hours (h_monday: ℕ) (h_tuesday: ℕ) (h_wednesday: ℕ) (h_thursday: ℕ) (h_friday: ℕ)
  (h_monday_eq: h_monday = 8) (h_tuesday_eq: h_tuesday = 7) (h_wednesday_eq: h_wednesday = 8)
  (h_thursday_eq: h_thursday = 10) (h_friday_eq: h_friday = 7) :
  (h_monday + h_tuesday + h_wednesday + h_thursday + h_friday) / 5 = 8 :=
by
  sorry

end average_sleep_hours_l149_149111


namespace ellipse_equation_and_chords_min_length_l149_149629

variable {a b : ℝ}
variable (x y : ℝ)
variable (e : ℝ) (P F_1 F_2 : ℝ)
variable (r I : ℝ)

-- Given conditions
def is_ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (e = 1 / 2) ∧ (r = 1) ∧ (S₁ + S₂ = 2)

-- Correct answer statements
theorem ellipse_equation_and_chords_min_length
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (h : is_ellipse a b)
  (eq_ellipse : a = 2 ∧ b = real.sqrt 3)
  (min_length : (|AB| + |CD|) = 48/7) : 
  ( x^2 / 4 + y^2 / 3 = 1 ) ∧ (|\overrightarrow{AB}| + |\overrightarrow{CD}| = 48/7) :=
sorry

end ellipse_equation_and_chords_min_length_l149_149629


namespace vessel_base_length_is_correct_l149_149341

-- Define the volume of a cube
def volume_of_cube (edge : ℝ) : ℝ :=
  edge^3

-- Define the volume of a rectangular prism given its base dimensions and height
def volume_of_prism (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

-- Constants given in the problem
def cube_edge : ℝ := 17
def vessel_width : ℝ := 15
def water_rise : ℝ := 16.376666666666665

-- The volume of the cube
def V_cube := volume_of_cube cube_edge

-- The length of one side of the base of the vessel (to be proved)
def length_of_vessel_base : ℝ := V_cube / (vessel_width * water_rise)

-- The theorem to prove
theorem vessel_base_length_is_correct : length_of_vessel_base ≈ 19.99 := 
  sorry

end vessel_base_length_is_correct_l149_149341


namespace hybrids_with_full_headlights_l149_149145

theorem hybrids_with_full_headlights (total_cars hybrids_percentage one_headlight_percentage : ℝ) 
  (hc : total_cars = 600) (hp : hybrids_percentage = 0.60) (ho : one_headlight_percentage = 0.40) : 
  total_cars * hybrids_percentage - total_cars * hybrids_percentage * one_headlight_percentage = 216 := by
  sorry

end hybrids_with_full_headlights_l149_149145


namespace vector_identity_proof_l149_149444

variables (a b m n p : Type)
variable [AddCommGroup a] [AddCommGroup b] [AddCommGroup m] [AddCommGroup n] [AddCommGroup p]
variable [HasSmul ℚ a] [HasSmul ℚ b] [HasSmul ℚ m] [HasSmul ℚ n] [HasSmul ℚ p]

variables (v_a v_b : ℚ →₀ a) (v_b' : ℚ →₀ b) (v_m : ℚ →₀ m) (v_n : ℚ →₀ n) (v_p : ℚ →₀ p)

-- Conditions
def non_collinear (v_a : ℚ →₀ a) (v_b : ℚ →₀ b) : Prop :=
  ∀ (c : ℚ), c • v_a ≠ v_b

def m_def (v_m v_a v_b : ℚ → a) : Prop :=
  v_m = (2 : ℚ) • v_a - (3 : ℚ) • v_b

def n_def (v_n v_a v_b : ℚ → n) : Prop :=
  v_n = (4 : ℚ) • v_a - (2 : ℚ) • v_b

def p_def (v_p v_a v_b : ℚ → p) : Prop :=
  v_p = (3 : ℚ) • v_a + v_b

-- Statement to prove
theorem vector_identity_proof :
  non_collinear v_a v_b →
  m_def v_m v_a v_b →
  n_def v_n v_a v_b →
  p_def v_p v_a v_b →
  v_p = (11 : ℚ) • v_n - (10 : ℚ) • v_m / 8 :=
sorry

end vector_identity_proof_l149_149444


namespace statement_a_statement_c_statement_d_l149_149688

-- Statement A: If z = i^3, then \overline{z} = i
theorem statement_a (z : ℂ) (h : z = complex.I ^ 3) : complex.conj z = complex.I := 
sorry

-- Statement C: If z = a + ai (a ∈ ℝ), then z cannot be a purely imaginary number
theorem statement_c (a : ℝ) (h : a ≠ 0) (z : ℂ) (h_eq : z = a + a * complex.I) : ¬ (z.im = 0) := 
sorry

-- Statement D: If the complex number z satisfies 1/z ∈ ℝ, then z ∈ ℝ
theorem statement_d (z : ℂ) (h : (1/z).im = 0) : z.im = 0 :=
sorry

end statement_a_statement_c_statement_d_l149_149688


namespace price_per_rose_is_correct_l149_149746

-- Declare conditions as assumptions
variables {total_roses : ℕ} {roses_left : ℕ} {total_earned : ℕ}

-- Provide specific values based on the problem description
def total_roses := 9
def roses_left := 4
def total_earned := 35

-- Calculate number of roses sold
def roses_sold := (total_roses - roses_left)

-- Calculate price per rose
def price_per_rose := (total_earned / roses_sold)

-- Lean statement of the proof problem
theorem price_per_rose_is_correct : price_per_rose = 7 := by
  sorry

end price_per_rose_is_correct_l149_149746


namespace ba_atoms_in_compound_l149_149340

theorem ba_atoms_in_compound (fw : ℝ) (bw : ℝ) (mw : ℝ) (nF : ℕ) (nBa : ℝ)
  (hf : fw = 19.00)
  (hb : bw = 137.33)
  (hm : mw = 175)
  (hnf : nF = 2)
  (hcomp : nF * fw + nBa * bw = mw)
  : nBa ≈ 1 :=
sorry

end ba_atoms_in_compound_l149_149340


namespace interest_difference_20_years_l149_149752

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

theorem interest_difference_20_years :
  compound_interest 15000 0.06 20 - simple_interest 15000 0.08 20 = 9107 :=
by
  sorry

end interest_difference_20_years_l149_149752


namespace frac_subtraction_l149_149750

theorem frac_subtraction : (18 / 42) - (3 / 8) = (3 / 56) := by
  -- Conditions
  have h1 : 18 / 42 = 3 / 7 := by sorry
  have h2 : 3 / 7 = 24 / 56 := by sorry
  have h3 : 3 / 8 = 21 / 56 := by sorry
  -- Proof using the conditions
  sorry

end frac_subtraction_l149_149750


namespace volleyball_serve_probability_range_l149_149641

theorem volleyball_serve_probability_range (p : ℝ) (h1: p > 0) (h2: p < 1) :
  (1 * p + 2 * p * (1 - p) + 3 * (1 - p)^2 > 1.75) ↔ (p ∈ set.Ioo 0 (1/2)) :=
by
  sorry

end volleyball_serve_probability_range_l149_149641


namespace inner_quadrilateral_diagonal_sum_l149_149910

theorem inner_quadrilateral_diagonal_sum {d d' : ℝ}
  (h₁ : ∃ (P : ℝ) (ABCD : Type) [convex_quadrilateral ABCD], sum_of_diagonals ABCD = d)
  (h₂ : ∃ (P' : ℝ) (A'B'C'D' : Type) [convex_quadrilateral A'B'C'D'] 
        [inside_quadrilateral A'B'C'D' ABCD], sum_of_diagonals A'B'C'D' = d') :
  d' < 2 * d := 
sorry

end inner_quadrilateral_diagonal_sum_l149_149910


namespace least_5_digit_number_divisible_by_15_25_40_75_125_140_l149_149411

theorem least_5_digit_number_divisible_by_15_25_40_75_125_140 : 
  ∃ n : ℕ, (10000 ≤ n) ∧ (n < 100000) ∧ 
  (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧ (125 ∣ n) ∧ (140 ∣ n) ∧ (n = 21000) :=
by
  sorry

end least_5_digit_number_divisible_by_15_25_40_75_125_140_l149_149411


namespace find_x_l149_149418

theorem find_x (x : ℝ) : 
  (∀ (y : ℝ), 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := sorry

end find_x_l149_149418


namespace percentage_flowering_plants_l149_149947

variable (P : ℝ)

theorem percentage_flowering_plants (h : 5 * (1 / 4) * (P / 100) * 80 = 40) : P = 40 :=
by
  -- This is where the proof would go, but we will use sorry to skip it for now
  sorry

end percentage_flowering_plants_l149_149947


namespace min_dist_intersection_points_eq_l149_149102

theorem min_dist_intersection_points_eq :
  ∃ m : ℝ, (0 < m) →
  |m^2 - log m| = (1/2) + (1/2) * log 2 :=
by
  sorry

end min_dist_intersection_points_eq_l149_149102


namespace _l149_149217

noncomputable def steiner_theorem :
  ∀ (A B C P H : Point)
    (circumcircle : Circumcircle A B C)
    (is_orthocenter : Orthocenter H A B C)
    (is_Simson_line : SimsonLine P A B C),
    midpoint P H ∈ is_Simson_line := 
sorry

end _l149_149217


namespace solution_mixture_l149_149315

theorem solution_mixture :
  ∀ (x y : ℕ), 
    (solution_X_alcohol_percentage : ℕ) (solution_Y_alcohol_percentage : ℕ),
    solution_X_alcohol_percentage = 10 →
    solution_Y_alcohol_percentage = 30 →
    100 = x →
    300 = y →
    (10 + 0.30 * y) / (100 + y) = 0.25 :=
by
  intros x y solution_X_alcohol_percentage solution_Y_alcohol_percentage h1 h2 hx hy
  rw [hx, hy, h1, h2]
  sorry

end solution_mixture_l149_149315


namespace range_of_a_squared_minus_2b_l149_149427

variable (a b : ℝ)

def quadratic_has_two_real_roots_in_01 (a b : ℝ) : Prop :=
  b ≥ 0 ∧ 1 + a + b ≥ 0 ∧ -2 ≤ a ∧ a ≤ 0 ∧ a^2 - 4 * b ≥ 0

theorem range_of_a_squared_minus_2b (a b : ℝ)
  (h : quadratic_has_two_real_roots_in_01 a b) : 0 ≤ a^2 - 2 * b ∧ a^2 - 2 * b ≤ 2 :=
sorry

end range_of_a_squared_minus_2b_l149_149427


namespace g_at_3_l149_149633

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 :
  (∀ (x : ℝ), x ≠ 0 → g x - 3 * g (1 / x) = 3 ^ x + 1) →
  g 3 = -17 / 4 :=
by
  -- Assume the condition for all x ≠ 0
  intros h
  -- We would include the proof steps here, but it is omitted
  sorry

end g_at_3_l149_149633


namespace general_formula_no_arithmetic_sequence_l149_149193

-- Given condition
def Sn (a : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a n - 3 * n

-- Theorem 1: General formula for the sequence a_n
theorem general_formula (a : ℕ → ℤ) (n : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) : 
  a n = 3 * 2^n - 3 :=
sorry

-- Theorem 2: No three terms of the sequence form an arithmetic sequence
theorem no_arithmetic_sequence (a : ℕ → ℤ) (x y z : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) (hx : x < y) (hy : y < z) :
  ¬ (a x + a z = 2 * a y) :=
sorry

end general_formula_no_arithmetic_sequence_l149_149193


namespace triangle_sin_cos_and_sides_l149_149886

theorem triangle_sin_cos_and_sides (a b c : ℝ) (A : ℝ) 
  (h1 : 3 * (b^2 + c^2) = 3 * a^2 + 2 * b * c)
  (h2 : a = 3/2)
  (h3 : b > c)
  (hS : S = (sqrt 2) / 2) :
  sin A = (2 * sqrt 2) / 3 ∧ (b = 3/2 ∧ c = 1) := by
  sorry

end triangle_sin_cos_and_sides_l149_149886


namespace two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l149_149871

def is_singular_number (n : ℕ) (num : ℕ) : Prop :=
  let first_n_digits := num / 10^n;
  let last_n_digits := num % 10^n;
  (num > 0) ∧
  (first_n_digits > 0) ∧
  (last_n_digits > 0) ∧
  (first_n_digits < 10^n) ∧
  (last_n_digits < 10^n) ∧
  (num = first_n_digits * 10^n + last_n_digits) ∧
  (∃ k, num = k^2) ∧
  (∃ k, first_n_digits = k^2) ∧
  (∃ k, last_n_digits = k^2)

-- (1) Prove that 49 is a two-digit singular number and 1681 is a four-digit singular number
theorem two_and_four_digit_singular_numbers :
  is_singular_number 1 49 ∧ is_singular_number 2 1681 :=
sorry

-- (2) Prove that 256036 is a six-digit singular number
theorem six_digit_singular_number :
  is_singular_number 3 256036 :=
sorry

-- (3) Prove the existence of a 20-digit singular number
theorem exists_twenty_digit_singular_number :
  ∃ num, is_singular_number 10 num :=
sorry

-- (4) Prove that there are at most 10 singular numbers with 100 digits
theorem at_most_ten_singular_numbers_with_100_digits :
  ∃! n, n <= 10 ∧ ∀ num, num < 10^100 → is_singular_number 50 num → num < 10 ∧ num > 0 :=
sorry

-- (5) Prove the existence of a 30-digit singular number
theorem exists_thirty_digit_singular_number :
  ∃ num, is_singular_number 15 num :=
sorry

end two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l149_149871


namespace income_from_investment_l149_149157

-- Definitions of the given conditions
def market_value : ℝ := 96
def rate : ℝ := 0.10
def investment : ℝ := 6240
def face_value_per_unit : ℝ := 100

-- Proof problem: Prove the income from the investment
theorem income_from_investment :
  let number_of_units := investment / market_value in
  let total_face_value := number_of_units * face_value_per_unit in
  let income := total_face_value * rate in
  income = 650 := 
by
  sorry

end income_from_investment_l149_149157


namespace role_assignment_l149_149731

theorem role_assignment (m w : ℕ) (m_roles w_roles e_roles : ℕ) 
  (hm : m = 5) (hw : w = 6) (hm_roles : m_roles = 2) (hw_roles : w_roles = 2) (he_roles : e_roles = 2) :
  ∃ (total_assignments : ℕ), total_assignments = 25200 :=
by
  sorry

end role_assignment_l149_149731


namespace minimum_value_of_z_l149_149292

theorem minimum_value_of_z :
  ∀ (x : ℝ), ∃ (z : ℝ), z = 3 * x^2 + 18 * x + 7 ∧ z ≥ -20 := 
by {
    intro x,
    use (3 * x^2 + 18 * x + 7),
    split,
    { refl, },
    { sorry, }
  }

end minimum_value_of_z_l149_149292


namespace range_of_a_l149_149883

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l149_149883


namespace waiter_earnings_l149_149371

def num_customers : ℕ := 9
def num_no_tip : ℕ := 5
def tip_per_customer : ℕ := 8
def num_tipping_customers := num_customers - num_no_tip

theorem waiter_earnings : num_tipping_customers * tip_per_customer = 32 := by
  sorry

end waiter_earnings_l149_149371


namespace curve_in_polar_coords_segment_AB_length_l149_149091

theorem curve_in_polar_coords :
  (∀ θ: ℝ, 
    let x := 1 + sqrt 3 * cos θ,
        y := sqrt 3 * sin θ in 
    (x - 1)^2 + y^2 = 3) →
  ∀ ρ θ: ℝ, 
    (ρ^2 - 2 * ρ * cos θ = 2) :=
by
  sorry

theorem segment_AB_length :
  (∀ ρ: ℝ, 
    (ρ * cos (π / 3 - π / 6) = 3 * sqrt 3) →
    ρ = 6) →
  (∀ θ ρ: ℝ,
    θ = π / 3 → 
    sqrt 3 * sin θ = ρ) →
  (6 - 2 = 4) :=
by
  sorry

end curve_in_polar_coords_segment_AB_length_l149_149091


namespace smallest_prime_factor_2457_l149_149676

theorem smallest_prime_factor_2457 : ∃ p: ℕ, prime p ∧ p ∣ 2457 ∧ ∀ q: ℕ, q ∣ 2457 ∧ prime q → p ≤ q :=
by
  sorry

end smallest_prime_factor_2457_l149_149676


namespace find_a_l149_149475

open Set

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) (h : (A ∪ B a) ⊆ (A ∩ B a)) : a = 1 :=
sorry

end find_a_l149_149475


namespace intersection_S_T_l149_149435

open Set

def S : Set ℝ := { x | x ≥ 1 }
def T : Set ℝ := { -2, -1, 0, 1, 2 }

theorem intersection_S_T : S ∩ T = { 1, 2 } := by
  sorry

end intersection_S_T_l149_149435


namespace no_nat_x_y_square_l149_149378

theorem no_nat_x_y_square (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ y^2 + x = b^2) := 
by 
  sorry

end no_nat_x_y_square_l149_149378


namespace minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l149_149811

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 3 / 2

theorem minimum_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x := by sorry

theorem decreasing_intervals_of_f : ∀ k : ℤ, ∀ x : ℝ,
  (Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ (2 * Real.pi / 3 + k * Real.pi) → ∀ y : ℝ, 
  (Real.pi / 6 + k * Real.pi) ≤ y ∧ y ≤ (2 * Real.pi / 3 + k * Real.pi) → x ≤ y → f y ≤ f x := by sorry

theorem maximum_value_of_f : ∃ k : ℤ, ∃ x : ℝ, x = (Real.pi / 6 + k * Real.pi) ∧ f x = 5 / 2 := by sorry

end minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l149_149811


namespace additional_cost_per_kg_l149_149369

theorem additional_cost_per_kg (l a : ℝ) 
  (h1 : 30 * l + 3 * a = 333) 
  (h2 : 30 * l + 6 * a = 366) 
  (h3 : 15 * l = 150) 
  : a = 11 := 
by
  sorry

end additional_cost_per_kg_l149_149369


namespace Frank_has_four_one_dollar_bills_l149_149030

noncomputable def Frank_one_dollar_bills : ℕ :=
  let total_money := 4 * 5 + 2 * 10 + 20 -- Money from five, ten, and twenty dollar bills
  let peanuts_cost := 10 - 4 -- Cost of peanuts (given $10 and received $4 in change)
  let one_dollar_bills_value := 54 - total_money -- Total money Frank has - money from large bills
  (one_dollar_bills_value : ℕ)

theorem Frank_has_four_one_dollar_bills 
   (five_dollar_bills : ℕ := 4) 
   (ten_dollar_bills : ℕ := 2)
   (twenty_dollar_bills : ℕ := 1)
   (peanut_price : ℚ := 3)
   (change : ℕ := 4)
   (total_money : ℕ := 50)
   (total_money_incl_change : ℚ := 54):
   Frank_one_dollar_bills = 4 := by
  sorry

end Frank_has_four_one_dollar_bills_l149_149030


namespace root_equation_value_l149_149501

theorem root_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2026 - m^2 + 2 * m = 2023 :=
sorry

end root_equation_value_l149_149501


namespace determine_sequence_p_gt_5_l149_149180

theorem determine_sequence_p_gt_5 (p : ℕ) [prime p] (h1 : 3 < p) :
  ∀ (a : ℕ → ℕ), (∀ (i j : ℕ), i ≠ j → i < (p-1)/2 + 1 → j < (p-1)/2 + 1 → 
  ∃ k : ℕ, k = (a i * a j) % p) → 
  (∀ (b : ℕ), b < (p-1)/2 + 1 → ∃ (c : ℕ), a c = b) → 
  p > 5 :=
by
  sorry

end determine_sequence_p_gt_5_l149_149180


namespace total_money_correct_l149_149227

def shelly_has_total_money : Prop :=
  ∃ (ten_dollar_bills five_dollar_bills : ℕ), 
    ten_dollar_bills = 10 ∧
    five_dollar_bills = ten_dollar_bills - 4 ∧
    (10 * ten_dollar_bills + 5 * five_dollar_bills = 130)

theorem total_money_correct : shelly_has_total_money :=
by
  sorry

end total_money_correct_l149_149227


namespace product_possible_values_l149_149372

theorem product_possible_values (N L M M_5: ℤ) :
  M = L + N → 
  M_5 = M - 8 → 
  ∃ L_5, L_5 = L + 5 ∧ |M_5 - L_5| = 6 →
  N = 19 ∨ N = 7 → 19 * 7 = 133 :=
by {
  sorry
}

end product_possible_values_l149_149372


namespace intersection_probability_l149_149225

open Real  -- Assuming Real is used for real number operations/snippets.

theorem intersection_probability:
  let k : ℝ := by sorry,
  have h1 : k ∈ Icc (-sqrt 2) (sqrt 2) := by sorry,
  let distance (x y : ℝ) : ℝ := abs (3 * k) / sqrt (k^2 + 1),
  have h2 : ∀ (x y : ℝ), (distance x y < 1) := by sorry,
  have h3 : ∀ (k : ℝ), abs(k) < sqrt(1/8) := by sorry,
  have interval_len : (sqrt 2 / 4 - (- sqrt 2 / 4)) / (2 * sqrt 2) = 1 / 4 := by sorry,
  show (interval_len : ℝ) = 1 / 4 := by sorry

end intersection_probability_l149_149225


namespace find_xyz_l149_149814

theorem find_xyz (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x^2 + y^2 + z^2 = 14) (h3 : x^3 + y^3 + z^3 = 17) : x * y * z = -7 :=
by
  sorry

end find_xyz_l149_149814


namespace smallest_AB_value_l149_149517

noncomputable def is_int (a : ℝ) : Prop := ∃ k : ℤ, a = k

-- Definition of the rhombus area and summation of areas as described in the conditions
def area_rhombus_ABCD (x : ℝ) := (x ^ 2 * Real.sqrt 3) / 2
def sum_areas_rhombi (x : ℝ) := (4 * x ^ 2 * Real.sqrt 3) / 7

theorem smallest_AB_value (x : ℝ) :
  (is_int (area_rhombus_ABCD x) ∧ is_int (sum_areas_rhombi x)) →
  x = Real.sqrt 42 / Real.sqrt 3 ^ (1/4) :=
by
  sorry

end smallest_AB_value_l149_149517


namespace range_of_function_l149_149795

theorem range_of_function : 
  ∀ x : ℝ, x ≠ -2 -> ∃ y : ℝ, y = (x^2 + 5 * x + 6) / (x + 2) ∧ (y ∈ set.Ioo (-∞ : ℝ) 1 ∪ set.Ioo 1 (∞ : ℝ)) :=
sorry

end range_of_function_l149_149795


namespace sum_of_atomic_and_nonatomic_is_atomic_l149_149940

-- Define the Lebesgue measure and the counting measure
noncomputable def lebesgue_measure : measure ℝ := by sorry
noncomputable def counting_measure : measure ℝ := {
  to_outer_measure := sorry,
  m_Union := sorry,
  trimmed := sorry
}

-- Define the sum of the Lebesgue measure and the counting measure
noncomputable def eta : measure ℝ := lebesgue_measure + counting_measure

-- Define the atomicity for measures
def is_atomic (μ : measure ℝ) : Prop := ∀ B, μ B > 0 → ∃ A, (A ⊆ B) ∧ (μ A > 0) ∧ (∀ B, (B ⊂ A) → μ B = 0)

-- Define the theorem to prove the sum of atomic and non-atomic measures is atomic
theorem sum_of_atomic_and_nonatomic_is_atomic :
  is_atomic counting_measure → is_atomic lebesgue_measure → is_atomic eta := sorry

end sum_of_atomic_and_nonatomic_is_atomic_l149_149940


namespace num_of_start_days_with_equal_tuesdays_and_fridays_l149_149349

def month_has_same_number_of_tuesdays_and_fridays (days_in_month : ℕ) : Prop := 
  ∃ first_day : ℕ, (days_in_month = 30) ∧ 
                   ((first_day mod 7 = 0 ∧ (days_in_month / 7 * 4 + if days_in_month % 7 > first_day then 1 else 0 = 4) ∧
                     (days_in_month / 7 * 4 + if days_in_month % 7 > (first_day + 4) % 7 then 1 else 0 = 4)) ∨
                    (first_day mod 7 = 3 ∧ (days_in_month / 7 * 4 + if days_in_month % 7 > first_day then 1 else 0 = 4) ∧
                     (days_in_month / 7 * 4 + if days_in_month % 7 > (first_day + 4) % 7 then 1 else 0 = 4)) ∨
                    (first_day mod 7 = 6 ∧ (days_in_month / 7 * 4 + if days_in_month % 7 > first_day then 1 else 0 = 4) ∧
                     (days_in_month / 7 * 4 + if days_in_month % 7 > (first_day + 4) % 7 then 1 else 0 = 4)))

theorem num_of_start_days_with_equal_tuesdays_and_fridays : 
  ∃ n : ℕ, n = 3 ∧ month_has_same_number_of_tuesdays_and_fridays 30 :=
sorry

end num_of_start_days_with_equal_tuesdays_and_fridays_l149_149349


namespace minimum_value_sum_l149_149190

open Real

theorem minimum_value_sum (y : Fin 50 → ℝ) (h_y_pos : ∀ i, 0 < y i)
  (h_sum_sq : ∑ i, y i ^ 2 = 1) :
  (∑ i, y i / (2 - y i ^ 2)) ≥ 9 / 4 := by
  sorry

end minimum_value_sum_l149_149190


namespace find_derivative_value_l149_149468

-- Define the function f, based on the initial condition
def f (df1 : ℝ) (x : ℝ) : ℝ := df1 * x^3 - 2 * x^2 + 3

-- Define the derivative of f, based on the computed derivative
def f' (df1 : ℝ) (x : ℝ) : ℝ := 3 * df1 * x^2 - 4 * x

theorem find_derivative_value (df1 : ℝ) (h : df1 = 2) : f' df1 2 = 16 :=
by
  rw h
  simp [f']
  norm_num

end find_derivative_value_l149_149468


namespace problem_I_problem_II_l149_149098

-- Definitions based on problem conditions
def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * x - a
def g (a : ℝ) (x : ℝ) : ℝ := (x - 2) * exp x + f a x

-- Assertions to prove
theorem problem_I (a : ℝ) (h1 : ∃ x, f a x = 0) : a = 1 :=
sorry

theorem problem_II (M : ℝ) (a : ℝ) (h2 : a = 1) (h3 : M = (let x := (sorry : ℝ) in (x - 1) * exp x - x - 1) ∧ M = f a (sorry : ℝ)) : -2.5 < M ∧ M < -2 :=
sorry

end problem_I_problem_II_l149_149098


namespace hephaestus_victory_iff_alpha_gt_2_l149_149918

/-
  Let α be a real number such that α ≥ 1. Poseidon initially floods some finite number of cells on an infinite grid.
  Hephaestus can build a levee by adding walls each turn, ensuring that the total length of walls does not exceed αn
  after n turns. Hephaestus wins if he encloses all flooded cells within a loop of walls.
  We need to prove that Hephaestus can guarantee victory if and only if α > 2.
-/

theorem hephaestus_victory_iff_alpha_gt_2 (α : ℝ) (α_ge_1 : α ≥ 1) :
  (∀ flooded_cells : set (ℤ × ℤ), finite flooded_cells → 
      ∃ turns : ℕ, ∃ walls : set (ℤ × ℤ × ℤ × ℤ), 
      (length walls ≤ α * turns) ∧ encloses_levee walls flooded_cells)
  ↔ α > 2 :=
sorry

end hephaestus_victory_iff_alpha_gt_2_l149_149918


namespace find_a_l149_149818

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then
    x^3 + 1
  else
    x^2 - a * x

theorem find_a : ∃ a : ℝ, f (f 0 a) a = -2 :=
by
  have h₁ : f 0 a = 1 := by
    simp [f]
    sorry
  have h₂ : f 1 a = 1 - a := by
    simp [f]
    sorry
  rw [h₁, h₂]
  existsi 3
  sorry

end find_a_l149_149818


namespace miles_driven_on_friday_is_16_l149_149546

-- Define constants and given conditions
constant reimbursement_rate : ℝ := 0.36
constant miles_monday : ℝ := 18
constant miles_tuesday : ℝ := 26
constant miles_wednesday : ℝ := 20
constant miles_thursday : ℝ := 20
constant total_reimbursement : ℝ := 36

-- Define the core proof problem: Prove that the miles driven on Friday is 16
theorem miles_driven_on_friday_is_16 
: ∀ (miles_friday : ℝ),
  (total_reimbursement = reimbursement_rate * (miles_monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday)) →
  miles_friday = 16 :=
by sorry

end miles_driven_on_friday_is_16_l149_149546


namespace ball_bounce_height_l149_149335

theorem ball_bounce_height :
  ∃ (k : ℕ), k = 6 ∧ 800 * (1 / 3 : ℝ) ^ k < 2 :=
by
  use 6
  simp
  linarith
  sorry

end ball_bounce_height_l149_149335


namespace unique_non_congruent_triangles_l149_149860

-- Define the points in the triangular array
structure Point :=
  (x : ℕ)
  (y : ℕ)

def A := Point.mk 0 0
def B := Point.mk 1 0
def C := Point.mk 2 0
def D := Point.mk 3 0
def E := Point.mk 5 1
def F := Point.mk 15 1
def G := Point.mk 25 1
def H := Point.mk 1 2
def I := Point.mk 2 2
def J := Point.mk 15 3

-- List of points
def points : List Point := [A, B, C, D, E, F, G, H, I, J]

-- Statement of the theorem
theorem unique_non_congruent_triangles : 
  list.countp (λ t, is_triangle_non_congruent t points) (triangle_combinations points) = 11 := 
sorry

end unique_non_congruent_triangles_l149_149860


namespace length_of_ZQ_l149_149759

theorem length_of_ZQ (XY XZ YZ ZQ : ℝ) 
  (hXYZ_right : XY^2 + YZ^2 = XZ^2)
  (hXZ : XZ = real.sqrt 85)
  (hXY : XY = 7)
  (hYZ_ZQ : YZ = ZQ) : 
  ZQ = 6 := 
by 
  sorry

end length_of_ZQ_l149_149759


namespace cosine_identity_l149_149808

theorem cosine_identity (α : ℝ) (h : sin (π / 6 - α) = 1 / 3) : cos (2 * π / 3 + 2 * α) = -7 / 9 :=
  sorry

end cosine_identity_l149_149808


namespace apples_prepared_l149_149240

variables (n_x n_l : ℕ)

theorem apples_prepared (hx : 3 * n_x = 5 * n_l - 12) (hs : 6 * n_l = 72) : n_x = 12 := 
by sorry

end apples_prepared_l149_149240


namespace sequence_a1_value_sequence_sum_first_20_terms_l149_149194

theorem sequence_a1_value (a : ℕ → ℕ) (h1 : ∀ n : ℕ, ∃ k, n = 2 * k → a (n + 1) = 2 * a n)
(h2 : ∀ n : ℕ, ∃ k, n = 2 * k - 1 → a (n + 1) = a n + 1)
(h3 : a 2^2 = a 1 * a 3) : a 1 = 1 := by
  sorry

theorem sequence_sum_first_20_terms (a : ℕ → ℕ) (h1 : ∀ n : ℕ, ∃ k, n = 2 * k → a (n + 1) = 2 * a n)
(h2 : ∀ n : ℕ, ∃ k, n = 2 * k - 1 → a (n + 1) = a n + 1)
(h3 : a 2^2 = a 1 * a 3) : ∑ i in Finset.range 20, a i = 6108 := by
  sorry

end sequence_a1_value_sequence_sum_first_20_terms_l149_149194


namespace proof_problem_l149_149766

theorem proof_problem (n p : ℕ) (Hp : nat.prime p) (Hn : n ≤ 2 * p) 
(Hdiv : n ^ (p - 1) ∣ (p - 1) ^ n + 1) : 
(n, p) = (1, p) ∨ (n, p) = (2, 2) ∨ (n, p) = (3, 3) :=
by
  sorry -- the proof is to be provided.

end proof_problem_l149_149766


namespace quilt_width_l149_149171

-- Definitions according to the conditions
def quilt_length : ℕ := 16
def patch_area : ℕ := 4
def first_10_patches_cost : ℕ := 100
def total_cost : ℕ := 450
def remaining_budget : ℕ := total_cost - first_10_patches_cost
def cost_per_additional_patch : ℕ := 5
def num_additional_patches : ℕ := remaining_budget / cost_per_additional_patch
def total_patches : ℕ := 10 + num_additional_patches
def total_area : ℕ := total_patches * patch_area

-- Theorem statement
theorem quilt_width :
  (total_area / quilt_length) = 20 :=
by
  sorry

end quilt_width_l149_149171


namespace problem_statement_l149_149547

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem problem_statement :
  let a := sqrt 3 + sqrt 5 + sqrt 15,
      b := -sqrt 3 + sqrt 5 + sqrt 15,
      c := sqrt 3 - sqrt 5 + sqrt 15,
      d := -sqrt 3 - sqrt 5 + sqrt 15 in
  ( (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 ) = 240 / 841 := 
by
  sorry

end problem_statement_l149_149547


namespace sequence_2007th_term_l149_149632

noncomputable def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c, (c.toNat - '0'.toNat) ^ 2)).sum

noncomputable def seq (n : ℕ) : ℕ → ℕ
| 0     := 2007
| (k+1) := sum_of_squares_of_digits (seq k)

theorem sequence_2007th_term : seq 2006 = 89 :=
by
    sorry

end sequence_2007th_term_l149_149632


namespace contrapositive_example_l149_149253

theorem contrapositive_example (x : ℝ) (h : -2 < x ∧ x < 2) : x^2 < 4 :=
sorry

end contrapositive_example_l149_149253


namespace cars_given_to_vinnie_l149_149912

variable (initialCars : Nat)
variable (boughtCars : Nat)
variable (birthdayCars : Nat)
variable (givenToSister : Nat)
variable (carsLeft : Nat)
variable (carsGivenToVinnie : Nat)

-- Define the conditions
def condition1 : initialCars = 14 := by
  sorry

def condition2 : boughtCars = 28 := by
  sorry

def condition3 : birthdayCars = 12 := by
  sorry

def condition4 : givenToSister = 8 := by
  sorry

def condition5 : carsLeft = 43 := by
  sorry

-- The main theorem
theorem cars_given_to_vinnie :
  initialCars + boughtCars + birthdayCars - givenToSister - carsGivenToVinnie = carsLeft 
  → carsGivenToVinnie = 3 := by
  intro h
  rw [condition1, condition2, condition3, condition4, condition5] at h
  sorry

end cars_given_to_vinnie_l149_149912


namespace steve_take_home_pay_l149_149609

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem steve_take_home_pay : 
  (annual_salary - (annual_salary * tax_rate + annual_salary * healthcare_rate + union_dues)) = 27200 := 
by 
  sorry

end steve_take_home_pay_l149_149609


namespace decorations_cost_l149_149577

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end decorations_cost_l149_149577


namespace tail_length_l149_149538

theorem tail_length {length body tail : ℝ} (h1 : length = 30) (h2 : tail = body / 2) (h3 : length = body) : tail = 15 := by
  sorry

end tail_length_l149_149538


namespace reflection_matrix_over_vector_l149_149786

theorem reflection_matrix_over_vector : 
  ∃ M : Matrix (Fin 2) (Fin 2) ℚ, 
    M.mul_vec ⟨[4, 1], by decide⟩ = ⟨[4, 1], by decide⟩ ∧
    M = !![ 15/17, 8/17; 8/17, -15/17 ]
    sorry

end reflection_matrix_over_vector_l149_149786


namespace area_of_circumcircle_of_triangle_ABC_l149_149526

-- Define the given conditions
variables {A B C : ℝ} -- A, B, C are the angles of the triangle
variables {a b c : ℝ} -- a, b, c are the sides opposite to angles A, B, C respectively

-- Given conditions as premises
def given_conditions : Prop :=
  (b^2 + c^2 - a^2 = real.sqrt 3 * b * c) ∧ (a = 1)

-- The statement to prove the question equals the answer
theorem area_of_circumcircle_of_triangle_ABC (h : given_conditions) : 
  let R := 1 in 
  π * R^2 = π :=
by 
  sorry

end area_of_circumcircle_of_triangle_ABC_l149_149526


namespace find_c_l149_149976

-- Definitions of variables and assumptions
variable (a b c y : ℝ)
variable (h₁ : b = -8)
variable (h₂ : c = 5)
variable (h₃ : x y (x = a * y ^ 2 + b * y + c))
variable (h₄ : parabola_vertex : (5, 4))

-- Statement to prove
theorem find_c : ∃ c, c = -27 :=
by {
  -- conditions from problem
  have vertex_form : x = a * (y - 4) ^ 2 + 5 := by sorry,
  have pass_point : x = a * (6 - 4) ^ 2 + 5 := by sorry,
  -- solve for a and find c
  have solve_a : a = -2 := by sorry,
  have c_value : c = -27 := by sorry,
  -- conclusion
  exact ⟨c, c_value⟩,
}

end find_c_l149_149976


namespace last_card_position_l149_149692

/-- Definition of the card elimination process. -/
def card_elimination (n : ℕ) : ℕ :=
  if h : n ≠ 1 then card_elimination (fin.pred (bit1 (to_fin h))) else 44

/-- Prove the position of the last remaining card after following the described elimination process. -/
theorem last_card_position : card_elimination 54 = 44 :=
  sorry

end last_card_position_l149_149692


namespace complex_roots_l149_149932

theorem complex_roots (p q r : ℂ) (h1 : p + q + r = -1) (h2 : p * q + p * r + q * r = -1) (h3 : p * q * r = -1) :
  multiset {p, q, r} = multiset {-1, 1, 1} :=
by
  sorry

end complex_roots_l149_149932


namespace vector_dot_product_circumcenter_l149_149505

theorem vector_dot_product_circumcenter
  (A B C O : EuclideanSpace ℝ (Fin 3))
  (hAB : dist A B = 3)
  (hAC : dist A C = 5)
  (hO : ∃ R, ∀ P, P ∈ [A, B, C] → dist P O = R) :
  (A -ᵥ O) • (C -ᵥ B) = 8 :=
sorry

end vector_dot_product_circumcenter_l149_149505


namespace sum_f_1_to_100_l149_149432

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation_1 (x : ℝ) : f(1 + x) = f(3 - x)
axiom functional_equation_2 (x : ℝ) : f(2 + x) = -f(1 - x)

theorem sum_f_1_to_100 : (finset.range 100).sum (λ n, f (n + 1)) = 0 := by {
  sorry
}

end sum_f_1_to_100_l149_149432


namespace BCEF_is_parallelogram_l149_149705

variable {A B C D E F M : Type}
variable [has_midpoint A C M]

axiom cyclic_quadrilateral (A B C D : Type) : Prop
axiom is_midpoint (M : Type) (A C : Type) : Prop
axiom parallel (D : Type) (BC : Type) : Type → Prop
axiom lies_on (E : Type) (BM : Type) : Prop
axiom circumcircle (A B C D : Type) (arc : Type) : Prop
axiom apart (F D : Type) : Prop

theorem BCEF_is_parallelogram (h1: cyclic_quadrilateral A B C D) 
  (h2 : lies_on_internal_bisectors_intersection_on_diagonal A B C D)
  (h3 : is_midpoint M A C)
  (h4 : parallel D BC E)
  (h5 : lies_on E BM)
  (h6 : circumcircle A B C D F)
  (h7 : apart F D) :
  parallelogram B C E F := 
sorry

end BCEF_is_parallelogram_l149_149705


namespace incorrect_statement_C_l149_149595

theorem incorrect_statement_C (x : ℝ) (h : x > -2) : (6 / x) > -3 :=
sorry

end incorrect_statement_C_l149_149595


namespace slopes_product_constant_l149_149830

def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_tangent_to_line (x y : ℝ) (line_eq : ℝ → ℝ → ℝ := λ x y, sqrt 7 * x - sqrt 5 * y + 12) : Prop :=
  line_eq x y = 0

def circle_eq (x y : ℝ) (radius : ℝ) : Prop :=
  x^2 + y^2 = radius^2

def points_collinear (A P M : ℝ × ℝ) : Prop :=
  -- Checks if points A, P, M are collinear.
  (P.1 - A.1) * (M.2 - A.2) = (M.1 - A.1) * (P.2 - A.2)

def find_ellipse_eq (a b : ℝ) (short_axis_eq : ℝ) : Prop :=
  -- Define the equation of ellipse
  a = 4 ∧ b = 2 * sqrt 3 ∧ short_axis_eq = b

theorem slopes_product_constant (a b : ℝ) (x1 y1 x2 y2 : ℝ) (m : ℝ) :
  ∃ k1 k2 : ℝ, find_ellipse_eq a b (2 * sqrt 3) ∧ 
  ellipse_eq x1 y1 a b ∧ 
  ellipse_eq x2 y2 a b ∧
  (x1 + 4) ≠ 0 ∧ (x2 + 4) ≠ 0 ∧ 
  points_collinear (-4, 0) (x1, y1) (16 / 3, m * (16 / 3) + 3) ∧ 
  points_collinear (-4, 0) (x2, y2) (16 / 3, m * (16 / 3) + 3) ∧ 
  k1 * k2 = -12 / 7 :=
sorry

end slopes_product_constant_l149_149830


namespace sum_of_digits_eq_24_l149_149332

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.to_string.to_list.foldl (λ acc c, acc + c.to_nat - '0'.to_nat) 0

theorem sum_of_digits_eq_24 (x : ℕ) (h₁ : 100 ≤ x ∧ x < 1000) (h₂ : 1000 ≤ x + 32 ∧ x + 32 < 10000) (hx : is_palindrome x) (hx32 : is_palindrome (x + 32)) :
  sum_of_digits x = 24 :=
  sorry

end sum_of_digits_eq_24_l149_149332


namespace variance_2ξ_plus_3_l149_149846

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Conditions: The random variable ξ has a discrete probability distribution.
noncomputable def ξ (ω : Ω) : ℝ := 1 * indicator {ω | ξ = 1} ω 
                                   + 2 * indicator {ω | ξ = 2} ω 
                                   + 3 * indicator {ω | ξ = 3} ω

axiom ξ_dist : ∀ k ∈ {1, 2, 3}, P {ω | ξ ω = k} = 1 / 3

-- Prove that the variance of 2ξ + 3 is 8/3
theorem variance_2ξ_plus_3 : variance (λ ω, 2 * ξ ω + 3) = 8 / 3 := by
  sorry

end variance_2ξ_plus_3_l149_149846


namespace decorations_cost_correct_l149_149578

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end decorations_cost_correct_l149_149578


namespace p_sufficient_not_necessary_q_l149_149053

theorem p_sufficient_not_necessary_q (x : ℝ) :
  (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros h
  cases h with h1 h2
  split
  case left => linarith
  case right => linarith

end p_sufficient_not_necessary_q_l149_149053


namespace point_in_first_quadrant_l149_149038

-- Definition for problem conditions
def problem_conditions (a b : ℝ) : Prop :=
  (1 : ℂ) + (1 : ℂ) * complex.I - (1 : ℂ) * (b : ℂ) * complex.I = (a : ℂ)

-- Definition to determine the quadrant
def first_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

-- The theorem statement
theorem point_in_first_quadrant (a b : ℝ) (h : problem_conditions a b) : first_quadrant a b :=
sorry

end point_in_first_quadrant_l149_149038


namespace triangle_area_from_squares_l149_149456

noncomputable def area_of_triangle (S1 S2 : ℝ) : ℝ :=
  let side1 := Real.sqrt S1
  let side2 := Real.sqrt S2
  0.5 * side1 * side2

theorem triangle_area_from_squares
  (A1 A2 : ℝ)
  (h1 : A1 = 196)
  (h2 : A2 = 100) :
  area_of_triangle A1 A2 = 70 :=
by
  rw [h1, h2]
  unfold area_of_triangle
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
  norm_num
  sorry

end triangle_area_from_squares_l149_149456


namespace equivalent_single_discount_l149_149744

theorem equivalent_single_discount (p : ℝ) :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  let final_price := (1 - discount1) * (1 - discount2) * (1 - discount3) * p
  (1 - final_price / p) = 0.27325 :=
by
  sorry

end equivalent_single_discount_l149_149744


namespace problem_one_l149_149328
   
   theorem problem_one (n : ℤ) (a : Fin (n+1) → ℤ) :
     ∃ i j : Fin (n+1), i ≠ j ∧ n ∣ (a i - a j) := 
   sorry
   
end problem_one_l149_149328


namespace eccentricity_value_angle_range_of_N_on_ellipse_l149_149991

variables {a b : ℝ} (h_ab : a > b ∧ b > 0)
variables (F1 F2 : ℝ × ℝ) (h_foci : F1 = (-√(a^2 - b^2), 0) ∧ F2 = (√(a^2 - b^2), 0))
variables (B : ℝ × ℝ) (h_B : B = (0, -b))
variables (A : ℝ × ℝ) (h_A : A = (a, 0))
variables (M : ℝ × ℝ) (h_M : (M.1 = √(a^2 - b^2) ∨ M.1 = -√(a^2 - b^2)) ∧ M.2 = (b^2) / a)
variables (O : ℝ × ℝ) (h_O : O = (0, 0))
variables (h_OM_parallel_AB : ∃ k : ℝ, O.2 + k * O.1 = M.2 ∧ B.2 + k * B.1 = A.2)

noncomputable def ellipse_eccentricity : ℝ :=
  √(1 - (b^2 / a^2))

theorem eccentricity_value : ellipse_eccentricity a b = √2 / 2 :=
by sorry

theorem angle_range_of_N_on_ellipse (N : ℝ × ℝ) (h_N_on_ellipse : N ≠ (a, 0) ∧ N ≠ (-a, 0) ∧ (N.1 / a)^2 + (N.2 / b)^2 = 1) :
  ∃ θ : ℝ, 0 < θ ∧ θ ≤ π / 2 ∧ θ = λ θ, angle F1 (N : ℝ × ℝ) F2 := 
by sorry

end eccentricity_value_angle_range_of_N_on_ellipse_l149_149991


namespace distance_between_table_1_and_table_3_l149_149035

theorem distance_between_table_1_and_table_3
  (last_year_race_distance : ℕ)
  (this_year_multiplier : ℕ)
  (num_tables : ℕ)
  (last_table_at_finish : Bool)
  (race_distance : ℕ := this_year_multiplier * last_year_race_distance)
  (intervals : ℕ := num_tables - 1)
  (distance_between_tables : ℕ := race_distance / intervals)
  (target_distance : ℕ := 2 * distance_between_tables)
  (last_year_race_distance = 300 : last_year_race_distance = 300)
  (this_year_multiplier = 4 : this_year_multiplier = 4)
  (num_tables = 6 : num_tables = 6)
  (last_table_at_finish = true : last_table_at_finish = true)
: target_distance = 480 := 
sorry

end distance_between_table_1_and_table_3_l149_149035


namespace length_of_arc_correct_l149_149621

open Real

noncomputable def length_of_arc (r θ : ℝ) := θ * r

theorem length_of_arc_correct (A r θ : ℝ) (hA : A = (θ / (2 * π)) * (π * r^2)) (hr : r = 5) (hA_val : A = 13.75) :
  length_of_arc r θ = 5.5 :=
by
  -- Proof steps are omitted
  sorry

end length_of_arc_correct_l149_149621


namespace transform_inequality_l149_149809

theorem transform_inequality (a b : ℝ) (h : a < b) : -3 * a > -3 * b :=
begin
  sorry
end

end transform_inequality_l149_149809


namespace reflection_matrix_over_vector_l149_149789

/-- 
The matrix A corresponds to the reflection over the vector (4, 1). 
The problem is to prove that reflecting any vector over (4, 1) results in matrix A.
-/
theorem reflection_matrix_over_vector : 
  let u := (⟨4, 1⟩ : ℝ × ℝ)
  let A := (λ (v : ℝ × ℝ), (⟨ (15/17) * v.1 + (8/17) * v.2, 
                              (8/17) * v.1 - (15/17) * v.2 ⟩ : ℝ × ℝ)) in 
  ∀ (v : ℝ × ℝ), 
  (let p := ((v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)) in
   let r := (2 * p * u.1 - v.1, 2 * p * u.2 - v.2) in
   r = A v) := sorry

end reflection_matrix_over_vector_l149_149789


namespace find_k_l149_149484

-- Define the vectors a and b.
def vector_a (k : ℝ) : ℝ × ℝ := (1, k)
def vector_b : ℝ × ℝ := (2, 1)

-- Define the condition that vector a is parallel to vector b.
def is_parallel (a b : ℝ × ℝ) : Prop := ∃ (λ : ℝ), a = (λ * b.1, λ * b.2)

-- Prove that if vector a is parallel to vector b, then k = 1/2.
theorem find_k (k : ℝ) (h : is_parallel (vector_a k) vector_b) : k = 1/2 :=
  sorry

end find_k_l149_149484


namespace measure_of_AB_l149_149161

theorem measure_of_AB (a b : ℝ) (AB CD DE BC : ℝ) (θ : ℝ) (AD_measure CD_measure : ℝ) 
(h1 : AB ∥ CD) (h2 : DE ∥ BC) (h3 : ∠D = 3 * ∠ B) (h4 : AD_measure = 2 * a) (h5 : CD_measure = 3 * b) :
  AB = 2 * a + 3 * b := 
  by 
  sorry

end measure_of_AB_l149_149161


namespace gain_percentage_for_40_clocks_is_10_l149_149353

-- Condition: Cost price per clock
def cost_price := 79.99999999999773

-- Condition: Selling price of 50 clocks at a gain of 20%
def selling_price_50 := 50 * cost_price * 1.20

-- Uniform profit condition
def uniform_profit_total := 90 * cost_price * 1.15

-- Given total revenue difference Rs. 40
def total_revenue := uniform_profit_total + 40

-- Question: Prove that selling price of 40 clocks leads to 10% gain
theorem gain_percentage_for_40_clocks_is_10 :
    40 * cost_price * 1.10 = total_revenue - selling_price_50 :=
by
  sorry

end gain_percentage_for_40_clocks_is_10_l149_149353


namespace john_work_days_l149_149394

theorem john_work_days (J : ℝ) : 
    (1 / 5 + 1 / J = 1 / 3.2142857142857144) → J ≈ 9 :=
by
    sorry

end john_work_days_l149_149394


namespace sum_of_cubes_roots_l149_149555

theorem sum_of_cubes_roots (a b c : ℝ) 
  (h1 : IsRoot (fun x => 5 * x^3 + 2003 * x + 3005) a)
  (h2 : IsRoot (fun x => 5 * x^3 + 2003 * x + 3005) b)
  (h3 : IsRoot (fun x => 5 * x^3 + 2003 * x + 3005) c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 :=
by
  sorry

end sum_of_cubes_roots_l149_149555


namespace simplify_and_rationalize_l149_149231

theorem simplify_and_rationalize :
  ( ( ( √6 / √7 ) * ( √6 / √14 ) * ( √9 / √21 ) ) = ( √2058 / 114 ) ) :=
by sorry

end simplify_and_rationalize_l149_149231


namespace least_value_m_plus_n_l149_149556

theorem least_value_m_plus_n :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
  nat.gcd (m + n) 330 = 1 ∧ 
  (n^n ∣ m^m) ∧ 
  ¬ (m ∣ n) ∧ 
  m + n = 507 :=
sorry

end least_value_m_plus_n_l149_149556


namespace distance_between_vertices_l149_149923

theorem distance_between_vertices :
  let C := (2, 1)
      D := (-3, 4)
  in real.sqrt ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2) = real.sqrt 34 :=
by
  let C := (2, 1)
  let D := (-3, 4)
  sorry

end distance_between_vertices_l149_149923


namespace derivative_y_l149_149409

noncomputable def y (x : ℝ) : ℝ :=
  (3 * x + 1)^4 * Real.arcsin (1 / (3 * x + 1))
  + (3 * x^2 + 2 * x + 1) * Real.sqrt (9 * x^2 + 6 * x)

theorem derivative_y (x : ℝ) (h : 3 * x + 1 > 0) :
  deriv y x = 12 * (3 * x + 1)^3 * Real.arcsin (1 / (3 * x + 1))
  + (3 * x + 1) * (18 * x^2) / Real.sqrt (9 * x^2 + 6 * x) := by
  sorry

end derivative_y_l149_149409


namespace find_number_l149_149892

def number_of_faces : ℕ := 6

noncomputable def probability (n : ℕ) : ℚ :=
  (number_of_faces - n : ℕ) / number_of_faces

theorem find_number (n : ℕ) (h: n < number_of_faces) :
  probability n = 1 / 3 → n = 4 :=
by
  -- proof goes here
  sorry

end find_number_l149_149892


namespace kittens_remaining_l149_149741

theorem kittens_remaining (original_kittens : ℕ) (kittens_given_away : ℕ) 
  (h1 : original_kittens = 8) (h2 : kittens_given_away = 4) : 
  original_kittens - kittens_given_away = 4 := by
  sorry

end kittens_remaining_l149_149741


namespace compare_fractions_l149_149387

theorem compare_fractions : (-8 / 21: ℝ) > (-3 / 7: ℝ) :=
sorry

end compare_fractions_l149_149387


namespace line_symmetric_points_l149_149873

noncomputable def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

noncomputable def slope (a b : ℝ × ℝ) : ℝ :=
  (b.2 - a.2) / (b.1 - a.1)

noncomputable def perpendicular_slope (m : ℝ) : ℝ :=
  -1 / m

theorem line_symmetric_points (m n : ℝ) :
  let P := (m - 2, n + 1)
  let Q := (n, m - 1)
  let M := midpoint P Q
  let slope_pq := slope P Q
  let slope_l := perpendicular_slope slope_pq
  (M = (m + n - 2) / 2, (m + n) / 2) →
  slope_pq = (m - n - 2) / (n - m + 2) →
  slope_l = 1 →
  ∃ (l : ℝ → ℝ), l = λ x, x + 1 ∧
  ∀ p, p ∈ { P, Q } → (p.2 = l p.1) :=
by
  sorry

end line_symmetric_points_l149_149873


namespace Allyson_age_is_28_l149_149298

-- Define the conditions of the problem
def Hiram_age : ℕ := 40
def add_12_to_Hiram_age (h_age : ℕ) : ℕ := h_age + 12
def twice_Allyson_age (a_age : ℕ) : ℕ := 2 * a_age
def condition (h_age : ℕ) (a_age : ℕ) : Prop := add_12_to_Hiram_age h_age = twice_Allyson_age a_age - 4

-- Define the theorem to be proven
theorem Allyson_age_is_28 (a_age : ℕ) (h_age : ℕ) (h_condition : condition h_age a_age) (h_hiram : h_age = Hiram_age) : a_age = 28 :=
by sorry

end Allyson_age_is_28_l149_149298


namespace min_a_condition_l149_149024

-- Definitions of the conditions
def real_numbers (x : ℝ) := true

def in_interval (a m n : ℝ) : Prop := 0 < n ∧ n < m ∧ m < 1 / a

def inequality (a m n : ℝ) : Prop :=
  (n^(1/m) / m^(1/n) > (n^a) / (m^a))

-- Lean statement
theorem min_a_condition (a m n : ℝ) (h1 : real_numbers m) (h2 : real_numbers n)
    (h3 : in_interval a m n) : inequality a m n ↔ 1 ≤ a :=
sorry

end min_a_condition_l149_149024


namespace ellipse_properties_slope_range_point_P_on_fixed_line_l149_149064

-- Conditions from the problem statement
variables {a b c : ℝ}
variables {λ : ℝ}

-- Ellipse equation and conditions
def ellipse_eq (x y : ℝ) := (x^2) / a^2 + (y^2) / b^2 = 1
axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom a2_eq_2 : a^2 = 2
axiom b2_eq_1 : b^2 = 1

-- Semi-focal length
def semi_focal_length := c = Real.sqrt (a^2 - b^2)

-- Line equation and point conditions
def line_eq (x : ℝ) := x = -a^2 / c
def N := (- (a^2 / c), 0)
def F1 := (-c, 0)
def F2 := (c, 0)
axiom F1F2_length : (2 * c = 2)
axiom F1F2_eq_2NF1 : (2 * N.1 = 2 * F1.1)

-- Collinearity condition
axiom collinear_NAB : ∀ {A B : ℝ × ℝ}, A.2 > 0 ∧ B.2 > 0 → λ ∈ [1/5, 1/3] → (N.1 = λ * N.1)

-- Proof goals
theorem ellipse_properties :
  ellipse_eq = λ (x y : ℝ), (x^2) / 2 + (y^2) / 1 = 1 :=
by sorry

theorem slope_range :
  ∀ (k : ℝ), (0 < k ∧ k < Real.sqrt(2) / 2) → (Real.sqrt(2) / 6 ≤ k ∧ k ≤ 1/2) :=
by sorry

theorem point_P_on_fixed_line :
  ∀ (P : ℝ × ℝ), P.2 = 0 → P.1 = -1 :=
by sorry

end ellipse_properties_slope_range_point_P_on_fixed_line_l149_149064


namespace club_president_vice_president_combinations_144_l149_149722

variables (boys_total girls_total : Nat)
variables (senior_boys junior_boys senior_girls junior_girls : Nat)
variables (choose_president_vice_president : Nat)

-- Define the conditions
def club_conditions : Prop :=
  boys_total = 12 ∧
  girls_total = 12 ∧
  senior_boys = 6 ∧
  junior_boys = 6 ∧
  senior_girls = 6 ∧
  junior_girls = 6

-- Define the proof problem
def president_vice_president_combinations (boys_total girls_total senior_boys junior_boys senior_girls junior_girls : Nat) : Nat :=
  2 * senior_boys * junior_boys + 2 * senior_girls * junior_girls

-- The main theorem to prove
theorem club_president_vice_president_combinations_144 :
  club_conditions boys_total girls_total senior_boys junior_boys senior_girls junior_girls →
  president_vice_president_combinations boys_total girls_total senior_boys junior_boys senior_girls junior_girls = 144 :=
sorry

end club_president_vice_president_combinations_144_l149_149722


namespace absent_children_l149_149951

-- Definitions
def total_children := 840
def bananas_per_child_present := 4
def bananas_per_child_if_all_present := 2
def total_bananas_if_all_present := total_children * bananas_per_child_if_all_present

-- The theorem to prove
theorem absent_children (A : ℕ) (P : ℕ) :
  P = total_children - A →
  total_bananas_if_all_present = P * bananas_per_child_present →
  A = 420 :=
by
  sorry

end absent_children_l149_149951


namespace median_of_data_set_l149_149434

open List

def data_set : List ℕ := [96, 89, 92, 95, 98]

def ordered_data_set : List ℕ := sort (· ≤ ·) data_set

theorem median_of_data_set :
  (ordered_data_set.length > 0) →
  ordered_data_set = [89, 92, 95, 96, 98] →
  ordered_data_set.nth ((ordered_data_set.length - 1) / 2) = some 95 :=
by
  intros h_length h_ordered
  rw [h_ordered]
  -- by calculating (ordered_data_set.length - 1) / 2 = 2 since length is 5.
  -- the nth index (counting from 0) is the 3rd element.
  simp
  sorry

end median_of_data_set_l149_149434


namespace degree_of_product_l149_149237

-- Definitions for the conditions
def isDegree (p : Polynomial ℝ) (n : ℕ) : Prop :=
  p.degree = n

variable {h j : Polynomial ℝ}

-- Given conditions
axiom h_deg : isDegree h 3
axiom j_deg : isDegree j 6

-- The theorem to prove
theorem degree_of_product : h.degree = 3 → j.degree = 6 → (Polynomial.degree (Polynomial.comp h (Polynomial.X ^ 4) * Polynomial.comp j (Polynomial.X ^ 3)) = 30) :=
by
  intros h3 j6
  sorry

end degree_of_product_l149_149237


namespace domain_of_function_l149_149627

theorem domain_of_function (x : ℝ) :
  (2 * x + 1 ≥ 0) ∧ (x ≠ 2) ↔ (x ≥ -1 / 2) ∧ (x ≠ 2) :=
by
  split
  intro h
  exact ⟨h.1, h.2⟩
  intro h
  exact ⟨h.1, h.2⟩

end domain_of_function_l149_149627


namespace union_A_B_l149_149856

-- Define the universal set U
def U := {1, 2, 3, 4, 5, 6}

-- Define complements of sets A and B with respect to U
def complement_A := {1, 2, 4}
def complement_B := {3, 4, 5}

-- Define sets A and B
def A := U \ complement_A
def B := U \ complement_B

-- State the theorem that A ∪ B = {1, 2, 3, 5, 6}
theorem union_A_B : A ∪ B = {1, 2, 3, 5, 6} :=
by
  -- proof goes here
  sorry

end union_A_B_l149_149856


namespace solution_set_for_x_l149_149649

theorem solution_set_for_x (x : ℝ) (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 :=
sorry

end solution_set_for_x_l149_149649


namespace smallest_whole_number_gt_total_sum_l149_149416

-- Declarations of the fractions involved
def term1 : ℚ := 3 + 1/3
def term2 : ℚ := 4 + 1/6
def term3 : ℚ := 5 + 1/12
def term4 : ℚ := 6 + 1/8

-- Definition of the entire sum
def total_sum : ℚ := term1 + term2 + term3 + term4

-- Statement of the theorem
theorem smallest_whole_number_gt_total_sum : 
  ∀ n : ℕ, (n > total_sum) → (∀ m : ℕ, (m >= 0) → (m > total_sum) → (n ≤ m)) → n = 19 := by
  sorry -- the proof is omitted

end smallest_whole_number_gt_total_sum_l149_149416


namespace children_multiple_of_four_l149_149368

theorem children_multiple_of_four (C : ℕ) 
  (h_event : ∃ (A : ℕ) (T : ℕ), A = 12 ∧ T = 4 ∧ 12 % T = 0 ∧ C % T = 0) : ∃ k : ℕ, C = 4 * k :=
by
  obtain ⟨A, T, hA, hT, hA_div, hC_div⟩ := h_event
  rw [hA, hT] at *
  sorry

end children_multiple_of_four_l149_149368


namespace imaginary_part_of_fraction_l149_149783

def complex_imaginary_part (z : ℂ) : ℝ :=
  z.im

theorem imaginary_part_of_fraction :
  complex_imaginary_part (2 / (2 + complex.I)) = -2 / 5 :=
by
  sorry

end imaginary_part_of_fraction_l149_149783


namespace percentage_decrease_l149_149955

-- Define the initial conditions
def total_cans : ℕ := 600
def initial_people : ℕ := 40
def new_total_cans : ℕ := 420

-- Use the conditions to define the resulting quantities
def cans_per_person : ℕ := total_cans / initial_people
def new_people : ℕ := new_total_cans / cans_per_person

-- Prove the percentage decrease in the number of people
theorem percentage_decrease :
  let original_people := initial_people
  let new_people := new_people
  let decrease := original_people - new_people
  let percentage_decrease := (decrease * 100) / original_people
  percentage_decrease = 30 :=
by
  sorry

end percentage_decrease_l149_149955


namespace area_of_ABCD_correct_l149_149357

noncomputable def distance (p1 p2 : (ℝ × ℝ × ℝ)) : ℝ :=
  Math.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

noncomputable def area_ABCD : ℝ :=
  let a := distance (0, 0, 0) (1, 0, 0) -- AB
  let b := distance (1, 0, 0) (2, 0.5, 0) -- BC
  let c := distance (2, 0.5, 0) (0, 1, 0.5) -- CD
  let d := distance (0, 1, 0.5) (0, 0, 0) -- DA
  let s := (a + b + c + d) / 2
  let diagonal_AC := distance (0, 0, 0) (2, 0.5, 0)
  let diagonal_BD := distance (1, 0, 0) (0, 1, 0.5)
  let angle_θ_radians : ℝ := -- example value as angle measure assumption
    Math.atan ((diagonal_BD/diagonal_AC) * Math.tan (Math.rad 45) )
  Math.sqrt((s - a) * (s - b) * (s - c) * (s - d) - (a * c * b * d) * Math.cos angle_θ_radians ^ 2)

/-- Prove the area of quadrilateral ABCD formed by the slicing plane through the given points. -/
theorem area_of_ABCD_correct :
  area_ABCD = /- Correct area value simplified from Bretschneider’s formula assuming proper calculation -/ :=
  sorry

end area_of_ABCD_correct_l149_149357


namespace hilt_has_2_pennies_l149_149205

-- Define the total value of coins each person has without considering Mrs. Hilt's pennies
def dimes : ℕ := 2
def nickels : ℕ := 2
def hilt_base_amount : ℕ := dimes * 10 + nickels * 5 -- 30 cents

def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1
def jacob_amount : ℕ := jacob_pennies * 1 + jacob_nickels * 5 + jacob_dimes * 10 -- 19 cents

def difference : ℕ := 13
def hilt_pennies : ℕ := 2 -- The solution's correct answer

theorem hilt_has_2_pennies : hilt_base_amount - jacob_amount + hilt_pennies = difference := by sorry

end hilt_has_2_pennies_l149_149205


namespace p_sufficient_not_necessary_for_q_l149_149043

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l149_149043


namespace calculate_expression_l149_149751

variable (a : ℝ)

theorem calculate_expression (h : a ≠ 0) : (6 * a^2) / (a / 2) = 12 * a := by
  sorry

end calculate_expression_l149_149751


namespace decorations_cost_l149_149576

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end decorations_cost_l149_149576


namespace right_tetrahedron_pythagorean_l149_149219

theorem right_tetrahedron_pythagorean (A B C D : Type) 
  (S1 S2 S3 S4 : ℝ) 
  (h1 : ∠ A D B = π / 2) 
  (h2 : ∠ A D C = π / 2) 
  (h3 : ∠ B D C = π / 2) 
  (h4 : S1 = S4 * real.cos α) 
  (h5 : S2 = S4 * real.cos β) 
  (h6 : S3 = S4 * real.cos γ)
  (h7 : real.cos (α)^2 + real.cos β^2 + real.cos γ^2 = 1) :
  S1^2 + S2^2 + S3^2 = S4^2 :=
begin
  sorry
end

end right_tetrahedron_pythagorean_l149_149219


namespace max_value_of_f_l149_149768

-- Define the quadratic function
def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

-- Define a proof problem to show that the maximum value of f(x) is 81/16
theorem max_value_of_f : ∃ x : ℝ, f x = 81 / 16 :=
by
  -- The vertex of the quadratic function gives the maximum value since the parabola opens downward
  let x := 9 / (2 * 4)
  use x
  -- sorry to skip the proof steps
  sorry

end max_value_of_f_l149_149768


namespace sum_of_4_corners_is_200_l149_149720

-- Define the conditions: 9x9 grid, numbers start from 10, and filled sequentially from left to right and top to bottom.
def topLeftCorner : ℕ := 10
def topRightCorner : ℕ := 18
def bottomLeftCorner : ℕ := 82
def bottomRightCorner : ℕ := 90

-- The main theorem stating that the sum of the numbers in the four corners is 200.
theorem sum_of_4_corners_is_200 :
  topLeftCorner + topRightCorner + bottomLeftCorner + bottomRightCorner = 200 :=
by
  -- Placeholder for proof
  sorry

end sum_of_4_corners_is_200_l149_149720


namespace pascal_triangle_ratio_l149_149395

theorem pascal_triangle_ratio (n r : ℕ) :
  (r + 1 = (4 * (n - r)) / 5) ∧ (r + 2 = (5 * (n - r - 1)) / 6) → n = 53 :=
by sorry

end pascal_triangle_ratio_l149_149395


namespace hyperbola_eccentricity_l149_149982

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : b = 2 * sqrt 2 * a) :
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = 3 :=
by
  sorry

end hyperbola_eccentricity_l149_149982


namespace solution_exists_l149_149472

-- Define the necessary conditions for the problem
def divisible_by (n k : ℕ) : Prop := ∃ m, n = k * m

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

def alternating_sum_of_digits (n : ℕ) : ℕ :=
  let rec alt_sum (n : ℕ) (is_even : bool) : ℕ :=
    if n = 0 then 0
    else if is_even then (n % 10) + alt_sum (n / 10) (not is_even)
    else -(n % 10) + alt_sum (n / 10) (not is_even)
  abs (alt_sum n true)

noncomputable def num_ways_to_replace_zeros :=
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) →
  let num := 8 * 10 ^ 81 + a * 10 ^ X + b * 10 ^ Y + 8 in
    divisible_by num 2 ∧
    divisible_by (sum_of_digits num) 9 ∧
    divisible_by (alternating_sum_of_digits num) 11

theorem solution_exists : num_ways_to_replace_zeros = 14080 := by
  sorry

end solution_exists_l149_149472


namespace find_point_A_l149_149008

noncomputable def point := ℝ × ℝ × ℝ

def equidistant (A B C : point) : Prop :=
  let d_AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2 + (B.3 - A.3)^2)
  let d_AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2 + (C.3 - A.3)^2)
  d_AB = d_AC

theorem find_point_A (x : ℝ) : 
  equidistant (x, 0, 0) (8, 1, -7) (10, -2, 1) → x = -9 / 4 := sorry

end find_point_A_l149_149008


namespace num_seating_arrangements_l149_149400

open Finset

-- Definitions using the conditions
def students := {1, 2, 3, 4, 5} -- Representing students as 1, 2, 3, 4, 5 or equivalently A, B, C, D, E

-- Function to check if exactly one student is in their assigned seat
def exactly_one_fixed (perm : Perm (Fin 5)) : Prop :=
  (count (λ i, perm i = i) (Finset.univ : Finset (Fin 5))) = 1

-- The proof statement: number of seating arrangements with exactly one fixed point equals 45
theorem num_seating_arrangements : 
  (Finset.filter exactly_one_fixed (Finset.univ : Finset (Perm (Fin 5)))).card = 45 :=
by
  sorry

end num_seating_arrangements_l149_149400


namespace jade_initial_pieces_l149_149168

theorem jade_initial_pieces (n w l p : ℕ) (hn : n = 11) (hw : w = 7) (hl : l = 23) (hp : p = n * w + l) : p = 100 :=
by
  sorry

end jade_initial_pieces_l149_149168


namespace valid_starting_days_for_30_day_month_l149_149348

theorem valid_starting_days_for_30_day_month : ∀ (days_in_month : ℕ), 
  days_in_month = 30 → 
  ∃ valid_days : ℕ, valid_days = 4 :=
by
sintro days_in_month h_days_in_month
have h : ∀ (d : ℕ), d < 7 → 
  ∃ tuesdays fridays : ℕ, tuesdays = fridays ∧ valid_days = 4 :=
sorry

end valid_starting_days_for_30_day_month_l149_149348


namespace arithmetic_mean_difference_l149_149123

-- Definitions and conditions
variable (p q r : ℝ)
variable (h1 : (p + q) / 2 = 10)
variable (h2 : (q + r) / 2 = 26)

-- Theorem statement
theorem arithmetic_mean_difference : r - p = 32 := by
  -- Proof goes here
  sorry

end arithmetic_mean_difference_l149_149123


namespace x_gt_y_l149_149181

variables (a b s t u v x y : ℝ)

-- Conditions
def condition_1 : Prop := 0 < a ∧ a < b
def condition_2 : Prop := s = (2 * a + b) / 3 ∧ t = (a + 2 * b) / 3
def condition_3 : Prop := u = (a^2 * b)^(1/3) ∧ v = (a * b^2)^(1/3)
def x_def : Prop := x = s * t * (s + t)
def y_def : Prop := y = u * v * (u + v)

theorem x_gt_y (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (hx : x_def) (hy : y_def) : x > y := 
by 
  sorry

end x_gt_y_l149_149181


namespace complex_number_solution_l149_149457

theorem complex_number_solution (z : ℂ) (hz : (z-1) * complex.I = 1 + complex.I) : z = 2 - complex.I := by
  sorry

end complex_number_solution_l149_149457


namespace unique_function_f_l149_149001

noncomputable def f : ℕ → ℕ := fun n => n

theorem unique_function_f (f : ℕ → ℕ) :
  (∀ n : ℕ, 3 * f(f(f(n))) + 2 * f(f(n)) + f(n) = 6 * n) ↔ (∀ n : ℕ, f(n) = n) :=
by
  sorry

end unique_function_f_l149_149001


namespace circuit_boards_failed_inspection_l149_149889

constant TotalBoards : Nat
constant FaultyBoards : Nat
constant PassVerificationProcessRate : Rat
constant PassVerificationProcessFaultRate : Rat

theorem circuit_boards_failed_inspection :
  TotalBoards = 3200 →
  FaultyBoards = 456 →
  PassVerificationProcessFaultRate = 1 / 8 →
  ∃ (P F : Nat), P + F = TotalBoards ∧
                  1 / PassVerificationProcessRate * P + F = FaultyBoards ∧
                  F = 64 :=
by
  -- Proof goes here
  sorry

end circuit_boards_failed_inspection_l149_149889


namespace smallest_number_remainder_problem_l149_149018

theorem smallest_number_remainder_problem :
  ∃ N : ℕ, (N % 13 = 2) ∧ (N % 15 = 4) ∧ (∀ n : ℕ, (n % 13 = 2 ∧ n % 15 = 4) → n ≥ N) :=
sorry

end smallest_number_remainder_problem_l149_149018


namespace find_x_l149_149236

theorem find_x (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^4 / y = 2) 
  (h2 : y^3 / z = 6)
  (h3 : z^2 / x = 8) : 
  x = (18432:ℝ)^(1/23) :=
by
  sorry

end find_x_l149_149236


namespace sara_remaining_red_balloons_l149_149597

-- Given conditions
def initial_red_balloons := 31
def red_balloons_given := 24

-- Statement to prove
theorem sara_remaining_red_balloons : (initial_red_balloons - red_balloons_given = 7) :=
by
  -- Proof can be skipped
  sorry

end sara_remaining_red_balloons_l149_149597


namespace OX_times_OY_eq_r_squared_l149_149709

-- Lean statement for part (a).
theorem OX_times_OY_eq_r_squared 
  (O X P Q : Point)
  (r : ℝ)
  (C_Circle : ∀ (O : Point) (r : ℝ), Circle)
  (X_outside : Point)
  (hC : Circle O r)
  (hX : X_outside ≠ O)
  (circle_XO : Circle X_outside (dist X_outside O))
  (hPQ : ∀ (P Q: Point), (hC P) ∧ (hC Q) ∧ (circle_XO P) ∧ (circle_XO Q) → are_points_on_same_circle P Q (c O r))
  (circle_P_center : ∀ P : Point, Circle P (dist P O))
  (circle_Q_center : ∀ Q : Point, Circle Q (dist Q O))
  (hYPQ: ∀ (Y : Point), (circle_P_center P) Y ∧ (circle_Q_center Q) Y → Y ≠ O) :
  dist O X_outside * dist O (hYPQ Y) = r^2 :=
by
  sorry

end OX_times_OY_eq_r_squared_l149_149709


namespace smallest_lcm_of_4digit_multiples_of_5_l149_149867

theorem smallest_lcm_of_4digit_multiples_of_5 :
  ∃ m n : ℕ, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (1000 ≤ n) ∧ (n ≤ 9999) ∧ (Nat.gcd m n = 5) ∧ (Nat.lcm m n = 201000) := 
sorry

end smallest_lcm_of_4digit_multiples_of_5_l149_149867


namespace decorations_cost_correct_l149_149572

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end decorations_cost_correct_l149_149572


namespace decorations_cost_correct_l149_149574

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end decorations_cost_correct_l149_149574


namespace no_uphill_divisible_by_45_l149_149763

def digits := {d : ℕ // d > 0 ∧ d < 10}

def isUphill (n : ℕ) : Prop :=
  ∃ ds : List ℕ, (∀ i j, i < j → ds.nth i < ds.nth j) ∧ (ds.foldr (λ d, 10 * d + · ) 0 = n)

theorem no_uphill_divisible_by_45 : 
  ¬ ∃ n : ℕ, isUphill n ∧ (n % 45 = 0) :=
begin
  sorry
end

end no_uphill_divisible_by_45_l149_149763


namespace evaluate_f_at_t_plus_one_l149_149185

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- Define the proposition to be proved
theorem evaluate_f_at_t_plus_one (t : ℝ) : f (t + 1) = 3 * t + 2 := by
  sorry

end evaluate_f_at_t_plus_one_l149_149185


namespace number_of_positive_integers_l149_149802

theorem number_of_positive_integers (count_x : ℕ) :
  (∀ x : ℕ, (50 ≤ (x + 3) ^ 2 ∧ (x + 3) ^ 2 ≤ 100) ↔ (x = 5 ∨ x = 6 ∨ x = 7)) →
  count_x = 3 :=
by
  intro h
  have : count_x = (finset.filter (λ x, 50 ≤ (x + 3) ^ 2 ∧ (x + 3) ^ 2 ≤ 100) (finset.range 8)).card,
  {
    sorry
  }
  rw [finset.filter_congr h] at this,
  exact this

end number_of_positive_integers_l149_149802


namespace city_population_l149_149331

theorem city_population (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 :=
by
  sorry

end city_population_l149_149331


namespace raisin_weight_l149_149872

theorem raisin_weight (Wg : ℝ) (dry_grapes_fraction : ℝ) (dry_raisins_fraction : ℝ) :
  Wg = 101.99999999999999 → dry_grapes_fraction = 0.10 → dry_raisins_fraction = 0.85 → 
  Wg * dry_grapes_fraction / dry_raisins_fraction = 12 := 
by
  intros h1 h2 h3
  sorry

end raisin_weight_l149_149872


namespace average_sleep_is_8_l149_149108

-- Define the hours of sleep for each day
def mondaySleep : ℕ := 8
def tuesdaySleep : ℕ := 7
def wednesdaySleep : ℕ := 8
def thursdaySleep : ℕ := 10
def fridaySleep : ℕ := 7

-- Calculate the total hours of sleep over the week
def totalSleep : ℕ := mondaySleep + tuesdaySleep + wednesdaySleep + thursdaySleep + fridaySleep
-- Define the total number of days
def totalDays : ℕ := 5

-- Calculate the average sleep per night
def averageSleepPerNight : ℕ := totalSleep / totalDays

-- Prove the statement
theorem average_sleep_is_8 : averageSleepPerNight = 8 := 
by
  -- All conditions are automatically taken into account as definitions
  -- Add a placeholder to skip the actual proof
  sorry

end average_sleep_is_8_l149_149108


namespace mapDistanceInCentimeters_l149_149209

-- Define the given conditions
def inchesToMiles (i : ℝ) : ℝ := i * (24 / 1.5) -- 1.5 inches = 24 miles
def milesToInches (m : ℝ) : ℝ := m * (1.5 / 24) -- reverse of above

def inchesToCentimeters (i : ℝ) : ℝ := i * 2.54 -- 1 inch = 2.54 cm

-- Given miles
def givenMiles : ℝ := 277.1653543307087

-- Prove the distance in cm
theorem mapDistanceInCentimeters :
  inchesToCentimeters (milesToInches givenMiles) ≈ 44.09 := sorry

end mapDistanceInCentimeters_l149_149209


namespace length_of_AB_l149_149439

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def reflection_across_xoy (p : Point) : Point :=
  { p with z := -p.z }

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem length_of_AB :
  let A := Point.mk 2 (-3) 5
  let B := reflection_across_xoy A
  distance A B = 10 :=
by 
  sorry

end length_of_AB_l149_149439


namespace trailing_zeros_100_factorial_l149_149490

theorem trailing_zeros_100_factorial : 
  (nat.num_trailing_zeros (nat.factorial 100) = 24) := sorry

end trailing_zeros_100_factorial_l149_149490


namespace k_at_27_l149_149938

noncomputable def h (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem k_at_27 (k : ℝ → ℝ)
    (hk_cubic : ∀ x, ∃ a b c, k x = a * x^3 + b * x^2 + c * x)
    (hk_at_0 : k 0 = 1)
    (hk_roots : ∀ a b c, (h a = 0) → (h b = 0) → (h c = 0) → 
                 ∃ (p q r: ℝ), k (p^3) = 0 ∧ k (q^3) = 0 ∧ k (r^3) = 0) :
    k 27 = -704 :=
sorry

end k_at_27_l149_149938


namespace children_on_bus_l149_149714

/-- Prove the number of children on the bus after the bus stop equals 14 given the initial conditions -/
theorem children_on_bus (initial_children : ℕ) (children_got_off : ℕ) (extra_children_got_on : ℕ) (final_children : ℤ) :
  initial_children = 5 →
  children_got_off = 63 →
  extra_children_got_on = 9 →
  final_children = (initial_children - children_got_off) + (children_got_off + extra_children_got_on) →
  final_children = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end children_on_bus_l149_149714


namespace num_of_start_days_with_equal_tuesdays_and_fridays_l149_149350

def month_has_same_number_of_tuesdays_and_fridays (days_in_month : ℕ) : Prop := 
  ∃ first_day : ℕ, (days_in_month = 30) ∧ 
                   ((first_day mod 7 = 0 ∧ (days_in_month / 7 * 4 + if days_in_month % 7 > first_day then 1 else 0 = 4) ∧
                     (days_in_month / 7 * 4 + if days_in_month % 7 > (first_day + 4) % 7 then 1 else 0 = 4)) ∨
                    (first_day mod 7 = 3 ∧ (days_in_month / 7 * 4 + if days_in_month % 7 > first_day then 1 else 0 = 4) ∧
                     (days_in_month / 7 * 4 + if days_in_month % 7 > (first_day + 4) % 7 then 1 else 0 = 4)) ∨
                    (first_day mod 7 = 6 ∧ (days_in_month / 7 * 4 + if days_in_month % 7 > first_day then 1 else 0 = 4) ∧
                     (days_in_month / 7 * 4 + if days_in_month % 7 > (first_day + 4) % 7 then 1 else 0 = 4)))

theorem num_of_start_days_with_equal_tuesdays_and_fridays : 
  ∃ n : ℕ, n = 3 ∧ month_has_same_number_of_tuesdays_and_fridays 30 :=
sorry

end num_of_start_days_with_equal_tuesdays_and_fridays_l149_149350


namespace least_pawns_required_l149_149176

theorem least_pawns_required (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (h3 : 2 * k > n) (h4 : 3 * k ≤ 2 * n) : 
  ∃ (m : ℕ), m = 4 * (n - k) :=
sorry

end least_pawns_required_l149_149176


namespace ratio_max_min_distance_l149_149823

theorem ratio_max_min_distance (n : ℕ) (h : n ≥ 2) (a : Fin n → ℝ) 
  (distinct_increasing : ∀ i j : Fin n, i < j → a i < a j) :
  let λ_n := (a ⟨n - 1, Nat.sub_lt h (nat.lt_succ_self 1)⟩ - a 0) / 
                      (Finset.min' (Finset.filter (λ ij, ij.1 < ij.2) (Fin n ×ˢ Fin n)) 
                                   (by sorry) 
                                   (λ p, a p.2 - a p.1)) in
  λ_n ≥ Real.sqrt (n * (n + 1) / 6) :=
by sorry

end ratio_max_min_distance_l149_149823


namespace intersecting_diagonals_nine_point_circle_l149_149642

theorem intersecting_diagonals_nine_point_circle 
  (triangle : EuclideanGeometry.Triangle)
  (l p : EuclideanGeometry.Line)
  (h_perpendicular : EuclideanGeometry.Perpendicular l p) :
  ∃ M : EuclideanGeometry.Point,
  (∀ P Q R : Parallelogram,
    (triangle.side1 = P.diagonal ∧ triangle.side2 = Q.diagonal ∧ triangle.side3 = R.diagonal) →
    (P.side1 ∥ l ∧ P.side2 ∥ p) ∧ (Q.side1 ∥ l ∧ Q.side2 ∥ p) ∧ (R.side1 ∥ l ∧ R.side2 ∥ p) →
    ∃ D1 D2 D3 : EuclideanGeometry.Line,
    (D1 = P.diagonal ∧ D2 = Q.diagonal ∧ D3 = R.diagonal) ∧
    D1 ≠ triangle.side1 ∧ D2 ≠ triangle.side2 ∧ D3 ≠ triangle.side3 →
    D1 ∩ D2 = M ∧ D2 ∩ D3 = M ∧ D1 ∩ D3 = M) ∧
  M ∈ EuclideanGeometry.ninePointCircle triangle :=
sorry

end intersecting_diagonals_nine_point_circle_l149_149642


namespace differential_eq_solution_l149_149166

theorem differential_eq_solution :
  ∀ (C₁ C₂ : ℝ), ∃ (y : ℝ → ℝ),
  (∀ (x : ℝ), y x = (C₁ + C₂ * x) * Real.exp (-2 * x) - (1 / 4) * Real.cos (2 * x) + (3 / 8) * Real.sin (2 * x)) ∧
  (∀ (x : ℝ), ((derivative (derivative y)) x + 4 * (derivative y x) + 4 * y x = 2 * Real.sin (2 * x) + 3 * Real.cos (2 * x))) :=
by
  sorry

end differential_eq_solution_l149_149166


namespace total_money_correct_l149_149226

def shelly_has_total_money : Prop :=
  ∃ (ten_dollar_bills five_dollar_bills : ℕ), 
    ten_dollar_bills = 10 ∧
    five_dollar_bills = ten_dollar_bills - 4 ∧
    (10 * ten_dollar_bills + 5 * five_dollar_bills = 130)

theorem total_money_correct : shelly_has_total_money :=
by
  sorry

end total_money_correct_l149_149226


namespace area_of_circle_l149_149260

theorem area_of_circle (r : ℝ) : 
  (S = π * r^2) :=
sorry

end area_of_circle_l149_149260


namespace inequality_proof_l149_149561

theorem inequality_proof
  (a b c d : ℝ)
  (hpos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (hcond: (a + b) * (b + c) * (c + d) * (d + a) = 1) :
  (2 * a + b + c) * (2 * b + c + d) * (2 * c + d + a) * (2 * d + a + b) * (a * b * c * d) ^ 2 ≤ 1 / 16 := 
by
  sorry

end inequality_proof_l149_149561


namespace A_P_not_76_l149_149356

theorem A_P_not_76 :
    ∀ (w : ℕ), w > 0 → (2 * w^2 + 6 * w) ≠ 76 :=
by
  intro w hw
  sorry

end A_P_not_76_l149_149356


namespace consecutive_product_even_product_divisible_by_6_l149_149930

theorem consecutive_product_even (n : ℕ) : ∃ k, n * (n + 1) = 2 * k := 
sorry

theorem product_divisible_by_6 (n : ℕ) : 6 ∣ (n * (n + 1) * (2 * n + 1)) :=
sorry

end consecutive_product_even_product_divisible_by_6_l149_149930


namespace peter_spent_on_repairs_l149_149956

variable (C : ℝ)

def repairs_cost (C : ℝ) := 0.10 * C

def profit (C : ℝ) := 1.20 * C - C

theorem peter_spent_on_repairs :
  ∀ C, profit C = 1100 → repairs_cost C = 550 :=
by
  intro C
  sorry

end peter_spent_on_repairs_l149_149956


namespace dot_product_sum_l149_149477

variables (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [NormedAddTorsor A B] [NormedAddTorsor B C]

def vector_length (u v : A) : ℝ := dist u v

theorem dot_product_sum 
  (A B C : A) 
  (h1 : vector_length A B = 3) 
  (h2 : vector_length B C = 4) 
  (h3 : vector_length C A = 5) : 
  (dist A B) * (dist B C) * real.cos(angle A B C) + 
  (dist B C) * (dist C A) * real.cos(angle B C A) + 
  (dist C A) * (dist A B) * real.cos(angle C A B) = -25 := 
sorry

end dot_product_sum_l149_149477


namespace triangle_return_to_original_position_l149_149183

/-- Let T be the triangle in the coordinate plane with vertices (0,0), (6,0), and (0,4).
Consider the following five isometries of the plane:
  1. Rotation by 90 degrees counterclockwise around the origin.
  2. Rotation by 180 degrees counterclockwise around the origin.
  3. Rotation by 270 degrees counterclockwise around the origin.
  4. Reflection across the x-axis.
  5. Reflection across the line y = x.
Prove that there exist exactly 18 sequences of three transformations (sequences may not be distinct) 
that will return this triangle T to its original position out of the 125 possible sequences. -/
theorem triangle_return_to_original_position : ∃ seqs : Finset (List (isometry EuclideanPlane)) // seqs.card = 18 :=
sorry

end triangle_return_to_original_position_l149_149183


namespace point_in_at_least_15_circles_l149_149282

theorem point_in_at_least_15_circles
  (C : Fin 100 → Set (ℝ × ℝ))
  (h1 : ∀ i j, ∃ p, p ∈ C i ∧ p ∈ C j)
  : ∃ p, ∃ S : Finset (Fin 100), S.card ≥ 15 ∧ ∀ i ∈ S, p ∈ C i :=
sorry

end point_in_at_least_15_circles_l149_149282


namespace increasing_sum_sequence_l149_149926

theorem increasing_sum_sequence (a : ℕ → ℝ) (Sn : ℕ → ℝ)
  (ha : ∀ n : ℕ, 0 < a (n + 1))
  (hSn : ∀ n : ℕ, Sn (n + 1) = Sn n + a (n + 1)) :
  (∀ n : ℕ, Sn (n + 1) > Sn n)
  ∧ ¬ (∀ n : ℕ, Sn (n + 1) > Sn n → 0 < a (n + 1)) :=
sorry

end increasing_sum_sequence_l149_149926


namespace total_money_received_l149_149204

-- Define the hours worked each day
def hours_day1 : ℕ := 12
def hours_day2 : ℕ := 10
def hours_day3 : ℕ := 8
def hours_day4 : ℕ := 6
def hours_day5 : ℕ := 14

-- Define the number of workers
def workers : ℚ := 2.5

-- Define the hourly rate in dollars
def hourly_rate : ℕ := 15

-- Calculate the total amount of money received
theorem total_money_received : workers * (hours_day1 + hours_day2 + hours_day3 + hours_day4 + hours_day5) * hourly_rate = 1875 := by
  -- Convert workers to an integer representation for calculation (multiply hours by 2.5)
  have man_hours : ℕ := (hours_day1 + hours_day2 + hours_day3 + hours_day4 + hours_day5) * 5 / 2
  -- Calculate the total amount received
  have total_amount : ℤ := man_hours * hourly_rate
  -- Check the calculated total amount
  have : total_amount = 1875 := by norm_num
  sorry

end total_money_received_l149_149204


namespace alex_mel_chelsea_probability_l149_149514

theorem alex_mel_chelsea_probability
  (h_alex : ∀ round, alex_wins round → 1/2)
  (h_mel : ∀ round, mel_wins round → 3 * chelsea_wins round)
  (h_total : ∀ round, alex_wins round + mel_wins round + chelsea_wins round = 1) :
  alex_wins 4 ∧ mel_wins 3 ∧ chelsea_wins 1 → 
    probability (alex_wins 4 ∧ mel_wins 3 ∧ chelsea_wins 1) = 945 / 8192 :=
begin
  obtain ⟨c, hc⟩ := h_mel round,
  have : probability (alex_wins 4 ∧ mel_wins 3 ∧ chelsea_wins 1) = sorry,
  exact this,
end

end alex_mel_chelsea_probability_l149_149514


namespace valid_starting_days_for_30_day_month_l149_149347

theorem valid_starting_days_for_30_day_month : ∀ (days_in_month : ℕ), 
  days_in_month = 30 → 
  ∃ valid_days : ℕ, valid_days = 4 :=
by
sintro days_in_month h_days_in_month
have h : ∀ (d : ℕ), d < 7 → 
  ∃ tuesdays fridays : ℕ, tuesdays = fridays ∧ valid_days = 4 :=
sorry

end valid_starting_days_for_30_day_month_l149_149347


namespace symmetry_center_l149_149090

noncomputable def f (x : Real) : Real := 2 * Real.sin (3 * x + Real.pi / 6) + 1

theorem symmetry_center :
  let (a, b) := (3, 1)
  (1 + Complex.i) * (a + b * Complex.i) = (2 + 4 * Complex.i) →
  ∃ x₀ y₀ : Real, 
  x₀ = 5 * Real.pi / 18 ∧ y₀ = f x₀ := 
by
  intros
  sorry

end symmetry_center_l149_149090


namespace quadratic_disc_gt_zero_range_of_m_l149_149069

variable (m : ℝ)

def quadratic (x : ℝ) : ℝ := x^2 - (m - 2) * x - m

-- Part (1): The quadratic function intersects the x-axis at two points
theorem quadratic_disc_gt_zero (m : ℝ) : 
  let Δ := (m - 2)^2 + 4 * m in
  Δ > 0 :=
by
  have Δ_eq : (m - 2)^2 + 4 * m = m^2 + 4 := by {
    calc (m - 2)^2 + 4 * m 
        = (m^2 - 4 * m + 4) + 4 * m : by ring
    ... = m^2 + 4 : by ring,
    }
  rw Δ_eq
  exact (lt_add_one 0).trans (by norm_num)
  sorry

-- Part (2): If y increases as x increases for x ≥ 3, find the range of values for m
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ici (3 : ℝ), ∀ y : ℝ, y = quadratic m x → ∀ x' ∈ Set.Ici x, quadratic m x' ≥ y) → m ≤ 8 :=
by
  intro h
  have h₀ : ∀ x, quadratic m (x + 1) ≥ quadratic m x, from λ x, by {
    rename h _ _ nt_zero
    calc quadratic m (x + 1) = quadratic m x + 1 * (2 * x + 1 - (m - 2)) - m + 1
                      : by norm_num
                       ... ≥ quadratic m x : by linarith,
  }
  sorry

end quadratic_disc_gt_zero_range_of_m_l149_149069


namespace midpoint_locus_l149_149716

noncomputable def midpoint {A B : Type} [NormedAddCommGroup A] [NormedAddCommonGroup B] (p1 : A) (p2 : B) : (A × B) :=
  (p1 + p2) / 2

theorem midpoint_locus {A B : Type} [NormedAddCommGroup A] [NormedAddCommonGroup B] :
  let A : ℝ × ℝ := (-c, 0)
  let B : ℝ × ℝ := (0, b)
  let C : ℝ × ℝ := (c, 0)
  let D : ℝ × ℝ := (λ, (b * (λ + c) / c))
  let E : ℝ × ℝ := (c + λ, b * (-λ) / c)
  let M := midpoint D E
  y (M : ℝ × ℝ) = b / 2
:= by
  sorry

end midpoint_locus_l149_149716


namespace john_february_phone_bill_l149_149719

-- Define given conditions
def base_cost : ℕ := 30
def included_hours : ℕ := 50
def overage_cost_per_minute : ℕ := 15 -- costs per minute in cents
def hours_talked_in_February : ℕ := 52

-- Define conversion from dollars to cents
def cents_per_dollar : ℕ := 100

-- Define total cost calculation
def total_cost (base_cost : ℕ) (included_hours : ℕ) (overage_cost_per_minute : ℕ) (hours_talked : ℕ) : ℕ :=
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_cost := extra_minutes * overage_cost_per_minute
  base_cost * cents_per_dollar + extra_cost

-- State the theorem
theorem john_february_phone_bill : total_cost base_cost included_hours overage_cost_per_minute hours_talked_in_February = 4800 := by
  sorry

end john_february_phone_bill_l149_149719


namespace reflection_matrix_over_vector_l149_149787

/-- 
The matrix A corresponds to the reflection over the vector (4, 1). 
The problem is to prove that reflecting any vector over (4, 1) results in matrix A.
-/
theorem reflection_matrix_over_vector : 
  let u := (⟨4, 1⟩ : ℝ × ℝ)
  let A := (λ (v : ℝ × ℝ), (⟨ (15/17) * v.1 + (8/17) * v.2, 
                              (8/17) * v.1 - (15/17) * v.2 ⟩ : ℝ × ℝ)) in 
  ∀ (v : ℝ × ℝ), 
  (let p := ((v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)) in
   let r := (2 * p * u.1 - v.1, 2 * p * u.2 - v.2) in
   r = A v) := sorry

end reflection_matrix_over_vector_l149_149787


namespace cos_identity_l149_149083

variable {α β : ℝ}

-- Defining the conditions
def α_in_interval : Prop := α ∈ Ioo (3*π/4) π
def β_in_interval : Prop := β ∈ Ioo (3*π/4) π

def sin_sum_condition : Prop := sin (α + β) = -4/5
def sin_diff_condition : Prop := sin (β - π/4) = 12/13

-- Stating the main theorem to prove
theorem cos_identity (hα : α_in_interval) (hβ : β_in_interval)
    (h_sin_sum : sin_sum_condition) (h_sin_diff : sin_diff_condition) :
    cos (α + π/4) = -63/65 :=
sorry

end cos_identity_l149_149083


namespace hockeyPlayers_count_l149_149148

def numPlayers := 50
def cricketPlayers := 12
def footballPlayers := 11
def softballPlayers := 10

theorem hockeyPlayers_count : 
  let hockeyPlayers := numPlayers - (cricketPlayers + footballPlayers + softballPlayers)
  hockeyPlayers = 17 :=
by
  sorry

end hockeyPlayers_count_l149_149148


namespace bartender_cheating_l149_149519

theorem bartender_cheating (cost_whiskey cost_pipe : ℝ)
  (total_cost_bartender : ℝ)
  (cost_tobacco cost_matches : ℝ) :
  cost_whiskey = 3 → cost_pipe = 6 →
  total_cost_bartender = 11.80 →
  ¬(∃ (T M : ℝ),
    T = cost_tobacco * 3 ∧ 
    M = cost_matches * 9 ∧
    (cost_whiskey + cost_pipe + T + M) = total_cost_bartender ∧
    (total_cost_bartender / 3) ∈ ℤ) :=
by
  intros h1 h2 h3
  sorry

end bartender_cheating_l149_149519


namespace summer_spending_l149_149989

theorem summer_spending :
  let end_may := 2.2
  let end_august := 4.8
  (end_august - end_may ≈ 2.6) →
  (abs ((end_august - end_may) - 2.5) < 0.1) :=
by
  let end_may := 2.2
  let end_august := 4.8
  have := (end_august - end_may ≈ 2.6) sorry
  have := (abs ((end_august - end_may) - 2.5) < 0.1) sorry

end summer_spending_l149_149989


namespace David_shots_in_fourth_game_l149_149761

theorem David_shots_in_fourth_game (shots_first_3_games : ℕ) (shots_taken_first_3_games : ℕ)
  (initial_average : ℝ) (additional_shots_taken : ℕ) (new_average : ℝ) :
  shots_first_3_games = 18 ->
  shots_taken_first_3_games = 45 ->
  initial_average = (18:ℝ) / 45 ->
  additional_shots_taken = 15 ->
  new_average = 0.55 ->
  let total_shots := shots_taken_first_3_games + additional_shots_taken in
  let total_made_shots := total_shots * new_average in
  let shots_made_in_fourth_game := total_made_shots - shots_first_3_games in
  shots_made_in_fourth_game = 15 :=
by
  intro h1 h2 h3 h4 h5
  let total_shots := shots_taken_first_3_games + additional_shots_taken
  let total_made_shots := total_shots * new_average
  let shots_made_in_fourth_game := total_made_shots - shots_first_3_games
  show shots_made_in_fourth_game = 15 from sorry

end David_shots_in_fourth_game_l149_149761


namespace determine_mine_positions_l149_149893

theorem determine_mine_positions (n : ℕ) (h₁ : n = 2009 ∨ n = 2007) : 
  ∀ grid : Array (Array ℕ), (∀ i j, grid[i][j] ≤ 9) → 
  (∀ i j, ∃ mines, grid[i][j] = count_mines i j mines) → 
  (∃ mines, ∀ i j, locate_mine grid i j = mines[i][j]) :=
sorry

end determine_mine_positions_l149_149893


namespace gcd_of_polynomial_coefficients_roots_of_poly_Q_l149_149549

-- Part (a)
theorem gcd_of_polynomial_coefficients (a b c : ℤ) (p q r : ℤ) :
  (P : ℤ[X]) = X^3 - p * X^2 + q * X - r →
  P = (X - a) * (X - b) * (X - c) →
  Int.gcd (Int.gcd a b) c = 1 →
  Int.gcd (Int.gcd p q) r = 1 :=
by
  sorry

-- Part (b)
theorem roots_of_poly_Q (s t : ℤ) :
  Q = X^3 - 98 * X^2 + 98 * s * X - 98 * t →
  s > 0 → t > 0 →
  {14, 28, 56} ⊆ multiset.to_finset (Q.roots.map (λ x, x * 14)) :=
by
  sorry

end gcd_of_polynomial_coefficients_roots_of_poly_Q_l149_149549


namespace fraction_of_shaded_area_l149_149725

theorem fraction_of_shaded_area (length width : ℕ) (h1 : length = 15) (h2 : width = 20) :
  let total_area := length * width,
      shaded_area := (total_area / 4) / 2
  in shaded_area / total_area = (1 : ℝ) / 8 := 
by
  sorry

end fraction_of_shaded_area_l149_149725


namespace hybrids_with_full_headlights_l149_149144

theorem hybrids_with_full_headlights (total_cars hybrids_percentage one_headlight_percentage : ℝ) 
  (hc : total_cars = 600) (hp : hybrids_percentage = 0.60) (ho : one_headlight_percentage = 0.40) : 
  total_cars * hybrids_percentage - total_cars * hybrids_percentage * one_headlight_percentage = 216 := by
  sorry

end hybrids_with_full_headlights_l149_149144


namespace limit_sequence_l149_149699

noncomputable def sequence (n : ℕ) : ℝ :=
  (sqrt ((n^5 + 1) * (n^2 - 1)) - n * sqrt (n * (n^4 + 1))) / n

theorem limit_sequence :
  tendsto sequence at_top (nhds (-∞)) :=
begin
  sorry
end

end limit_sequence_l149_149699


namespace min_square_side_length_l149_149360

theorem min_square_side_length 
  (table_length : ℕ) (table_breadth : ℕ) (cube_side : ℕ) (num_tables : ℕ)
  (cond1 : table_length = 12)
  (cond2 : table_breadth = 16)
  (cond3 : cube_side = 4)
  (cond4 : num_tables = 4) :
  (2 * table_length + 2 * table_breadth) = 56 := 
by
  sorry

end min_square_side_length_l149_149360


namespace garden_perimeter_is_44_l149_149724

-- Define the original garden's side length given the area
noncomputable def original_side_length (A : ℕ) := Nat.sqrt A

-- Given condition: Area of the original garden is 49 square meters
def original_area := 49

-- Define the new side length after expanding each side by 4 meters
def new_side_length (original_side : ℕ) := original_side + 4

-- Define the perimeter of the new garden given the new side length
def new_perimeter (new_side : ℕ) := 4 * new_side

-- Proof statement: The perimeter of the new garden given the original area is 44 meters
theorem garden_perimeter_is_44 : new_perimeter (new_side_length (original_side_length original_area)) = 44 := by
  -- This is where the proof would go
  sorry

end garden_perimeter_is_44_l149_149724


namespace determine_faces_l149_149659

namespace DiceProof

-- Define a structure to represent a cube with the dot distributions
structure Cube :=
(faces : Fin 6 → ℕ)
-- Each cube has one face with 3 points, two faces with 2 points, three faces with 1 point
(def_cube : ∀ c : Cube, ∃ i j k l m n, 
  [c.faces i, c.faces j, c.faces k, c.faces l, c.faces m, c.faces n] = [3, 2, 2, 1, 1, 1])

-- Condition: there are 7 such cubes
def seven_cubes := Vector Cube 7

-- Define the structure for the specific configuration
def shapeП (cubes : seven_cubes) : Prop :=
-- All touching faces between cubes must have the same number of dots
∀ (i j : Fin 7) (f : Fin 6), touching_faces i j f → 
  (cubes.nth i).faces f = (cubes.nth j).faces f

-- Main theorem to prove the dots on faces A, B, and C
theorem determine_faces (cubes : seven_cubes) (h : shapeП cubes)
  (A B C : Fin 6) : 
  A = 2 ∧ B = 2 ∧ C = 3 :=
sorry

end DiceProof

end determine_faces_l149_149659


namespace remy_gallons_used_l149_149222

def roman_usage (R : ℕ) : Prop := R + (3 * R + 1) = 33

def remy_usage (R : ℕ) (Remy : ℕ) : Prop := Remy = 3 * R + 1

theorem remy_gallons_used :
  ∃ R Remy : ℕ, roman_usage R ∧ remy_usage R Remy ∧ Remy = 25 :=
  by
    sorry

end remy_gallons_used_l149_149222


namespace new_median_with_ten_l149_149735

theorem new_median_with_ten (s : List ℕ) (h₁ : s.length = 7) 
                             (h₂ : (s.sum : ℝ) / s.length = 5.7) 
                             (h₃ : s.mode = [5]) 
                             (h₄ : s.nth (s.length / 2) = some 6) : 
                             (let s' := s ++ [10] in (s'.nth (s'.length / 2) = some 6)) :=
by
  sorry

end new_median_with_ten_l149_149735


namespace eccentricity_of_ellipse_l149_149534

def is_ellipse (a b : ℝ) : Prop :=
  ∃ c : ℝ, 0 < c ∧ c < a ∧ (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 → (c^2 / a^2) + (y^2 / b^2) = 1)

theorem eccentricity_of_ellipse {a b e : ℝ} 
  (h1 : is_ellipse a b)
  (h2 : ∃ F1 F2 : ℝ × ℝ, F1.2 = 0 ∧ F2.2 = 0 ∧ (F1.1^2 / a^2) + (F2.1^2 / b^2) = 1)
  (h3 : ∃ A B : ℝ × ℝ, A.1 = F1.1 ∧ B.1 = F1.1 ∧ A.2 ≠ B.2 ∧ ∃ F2, angle A F2 B = 90)
  (h4 : 2 * e = 1 - e^2) :
  e = sqrt 2 - 1 :=
by
  sorry

end eccentricity_of_ellipse_l149_149534


namespace find_x_value_l149_149020

theorem find_x_value :
  ∃ x : Real, ( ( x * 0.48 * 2.50 ) / ( 0.12 * 0.09 * 0.5 ) = 2400.0000000000005 ) ∧ ( x = 10.8 ) :=
by
  existsi (10.8 : Real)
  split
  · -- the expression equals 2400.0000000000005
    sorry
  · -- x equals 10.8
    rfl

end find_x_value_l149_149020


namespace part1_part2_l149_149828

-- Definitions of sequences {a_n} and {S_n} with the given condition.
def seq_a : ℕ+ → ℝ := 
  λ n, 2 / 3^n

def seq_S : ℕ+ → ℝ :=
  λ n, (1 - (1/3)^n)

-- Part 1: Proving the general formula for the sequence {a_n}
theorem part1 (n : ℕ+) : 
  (1 : ℝ) = seq_S n + (1/2) * seq_a n := 
by sorry

-- Definition of sequence {b_n} based on {S_n}
def seq_b : ℕ+ → ℝ :=
  λ n, real.log_base (1/3) (1 - seq_S n)

-- Part 2: Proving the value of the sum of reciprocals products of b_n and b_{n+1}
theorem part2 (n : ℕ+) : 
  (∑ k in finset.range n, (1 / (seq_b k.succ * seq_b (k.succ+1)))) = n / (n + 1) := 
by sorry

end part1_part2_l149_149828


namespace num_perfect_cube_factors_of_3920_l149_149997

theorem num_perfect_cube_factors_of_3920 : 
  let a_bound := 3
      b_bound := 1
      c_bound := 2
      is_multiple_of_3 (n : Nat) := n % 3 = 0
  in 
  let choices_for_cubes :=
    ((List.filter is_multiple_of_3 [0, a_bound]).length) *
    ((List.filter is_multiple_of_3 [0, b_bound]).length) *
    ((List.filter is_multiple_of_3 [0, c_bound]).length)
  in choices_for_cubes = 2 :=
by
  let a_bound := 3
  let b_bound := 1
  let c_bound := 2
  let is_multiple_of_3 := λ n : Nat => n % 3 = 0
  let choices_for_cubes :=
    (List.filter is_multiple_of_3 [0, a_bound]).length *
    (List.filter is_multiple_of_3 [0, b_bound]).length *
    (List.filter is_multiple_of_3 [0, c_bound]).length
  have h1 : choices_for_cubes = 2 := sorry
  exact h1

end num_perfect_cube_factors_of_3920_l149_149997


namespace find_omega_l149_149463

noncomputable def sin_cos_sum (ω x : ℝ) : ℝ :=
  sin (ω * x) + cos (ω * x)

theorem find_omega {ω : ℝ} (h1 : 0 < ω)
    (h2 : ∀ x1 x2, -ω < x1 ∧ x1 < x2 ∧ x2 < ω → sin_cos_sum ω x1 ≤ sin_cos_sum ω x2)
    (h3 : sin_cos_sum ω ω = sin_cos_sum ω (-ω)) :
  ω = sqrt pi / 2 :=
sorry

end find_omega_l149_149463


namespace complete_sets_l149_149567

-- Define a natural number set to be complete
def is_complete (A : Set ℕ) : Prop :=
  A.nonempty ∧ ∀ a b : ℕ, a + b ∈ A → a * b ∈ A

-- Prove that the only complete sets are {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, and ℕ
theorem complete_sets :
  ∀ A : Set ℕ, is_complete A ↔ 
    A = {1} ∨
    A = {1, 2} ∨
    A = {1, 2, 3} ∨
    A = {1, 2, 3, 4} ∨
    A = Set.univ :=
by
  sorry

end complete_sets_l149_149567


namespace blue_ball_higher_than_yellow_l149_149337

-- Define the probability function for landing in bin k
def bin_prob (k : ℕ) : ℝ := 3^(-k)

-- Define the event for both balls landing in the same bin
def same_bin_prob : ℝ := ∑' k, bin_prob k * bin_prob k

-- Define the event for the balls landing in different bins
def different_bin_prob : ℝ := 1 - same_bin_prob

-- Calculate the probability of the blue ball being in a higher-numbered bin than the yellow ball
def blue_higher_prob : ℝ := (1 / 2) * different_bin_prob

theorem blue_ball_higher_than_yellow : blue_higher_prob = 7 / 16 := by
  sorry

end blue_ball_higher_than_yellow_l149_149337


namespace sixth_expression_nth_expression_product_calculation_l149_149581

-- Definition of the observed pattern
def pattern (k : ℕ) : Prop :=
  k * (k + 2) + 1 = (k + 1)^2

-- Proof that the 6th expression is correct 
theorem sixth_expression : pattern 6 :=
  by sorry

-- Proof that the n-th expression is correct
theorem nth_expression (n : ℕ) : pattern n :=
  by sorry

-- Definition of the product of sequences
def product (k : ℕ) := ((1 + 1 / (k * (k + 2))) : ℚ)

-- Product calculation from 1 to 98
theorem product_calculation : 
  Π (n : ℕ), n = 98 → 
  ∏ i in (finset.range n.succ).erase 0, product i = 99 / 50 :=
  by sorry

end sixth_expression_nth_expression_product_calculation_l149_149581


namespace range_of_m_l149_149841

theorem range_of_m 
  (m : ℝ) 
  (line_intersects_ellipse : ∃ A B: ℝ × ℝ, 
                              A ≠ B ∧ 
                              (∃ x y: ℝ, line_eq: x + 3*y = m ∧ ellipse_eq: (x^2)/9 + y^2 = 1 ∧
                                          fst A = x ∧ snd A = y ∧ fst B ≠ x)) :
  ( -3 * Real.sqrt 2 < m ∧ m < -3) ∨ (3 < m ∧ m < 3 * Real.sqrt 2) :=
by 
  sorry

end range_of_m_l149_149841


namespace total_cups_l149_149274

theorem total_cups (butter flour sugar : ℕ) 
                   (h_ratio : butter:flour:sugar = 2:5:3) 
                   (h_flour : flour = 10) : 
                   butter + flour + sugar = 20 :=
sorry

end total_cups_l149_149274


namespace centroid_of_diameter_l149_149995

theorem centroid_of_diameter (x1 y1 x2 y2 : ℝ) (h1 : x1 = 5) (h2 : y1 = 10) (h3 : x2 = -4) (h4 : y2 = -2) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 = 4.5 :=
by
  rw [h1, h2, h3, h4]
  dsimp
  norm_num

end centroid_of_diameter_l149_149995


namespace train_capacity_l149_149585

theorem train_capacity (T : ℝ) (h : 2 * (T / 6) = 40) : T = 120 :=
sorry

end train_capacity_l149_149585


namespace second_concert_attendance_l149_149206

theorem second_concert_attendance (n1 : ℕ) (h1 : n1 = 65899) (h2 : n2 = n1 + 119) : n2 = 66018 :=
by
  -- proof goes here
  sorry

end second_concert_attendance_l149_149206


namespace line_divides_parallelogram_into_given_ratio_l149_149729

variable {A B C D X : Type} [AffineSpace A B C D] [LocallyCompactSpace A] [ConcreteMulAction B]

def parallelogram (A B C D : Type) [AffineSpace A B C D] : Prop := 
Line.parallel A B C D

theorem line_divides_parallelogram_into_given_ratio :
  parallelogram A B C D → 
  ∃ (X : Type), parallel X D ∧ (area A X Y C / area A B D C = 2 / 3) :=
by
  sorry

end line_divides_parallelogram_into_given_ratio_l149_149729


namespace domain_of_y_l149_149256

noncomputable def domain_of_function (x : ℝ) : Bool :=
  x < 0 ∧ x ≠ -1

theorem domain_of_y :
  {x : ℝ | (∃ y, y = (x + 1) ^ 0 / Real.sqrt (|x| - x)) } =
  {x : ℝ | domain_of_function x} :=
by
  sorry

end domain_of_y_l149_149256


namespace units_digit_quotient_l149_149397

theorem units_digit_quotient (n : ℕ) (h1 : n % 2 = 1): 
  (4^n + 6^n) / 10 % 10 = 1 :=
by 
  -- Given the cyclical behavior of 4^n % 10 and 6^n % 10
  -- 4^n % 10 cycles between 4 and 6, 6^n % 10 is always 6
  -- Since n is odd, 4^n % 10 = 4 and 6^n % 10 = 6
  -- Adding them gives us 4 + 6 = 10, and thus a quotient of 1
  sorry

end units_digit_quotient_l149_149397


namespace g_243_l149_149986

noncomputable def g (n : ℕ) : ℕ := sorry

axiom g_property1 : ∀ n : ℕ, 0 < n → g(g(n)) = 3 * n
axiom g_property2 : ∀ n : ℕ, 0 < n → g(3 * n + 2) = 3 * n + 1

theorem g_243 : g 243 = 243 :=
by
  sorry

end g_243_l149_149986


namespace S_neg1_1_neg2div3_S_1_1_neg1_neg1_min_S3_S3_neg1_attainable_min_S_odd_S_odd_minimum_attainable_l149_149077

noncomputable def S (l : List ℝ) : ℝ :=
  l.sublists.filter (λ sl, sl.length = 2).sum (λ sl, sl.head * sl.tail.head)

-- Prove S(-1, 1, -2/3) = -1
theorem S_neg1_1_neg2div3 : S [-1, 1, -2/3] = -1 := 
  by sorry

-- Prove S(1, 1, -1, -1) = -2
theorem S_1_1_neg1_neg1 : S [1, 1, -1, -1] = -2 := 
  by sorry

-- For n = 3, prove minimum S(x₁, x₂, x₃) is -1
theorem min_S3 : ∀ (x₁ x₂ x₃ : ℝ), (|x₁| ≤ 1) → (|x₂| ≤ 1) → (|x₃| ≤ 1) → 
  S [x₁, x₂, x₃] ≥ -1 := 
  by sorry
theorem S3_neg1_attainable : (|1| ≤ 1) → (|1| ≤ 1) → (|-1| ≤ 1) → 
  S [1, 1, -1] = -1 := 
  by sorry

-- For odd n, proving minimum S(x₁, x₂, ..., x_n) is -1/2*(n-1)
theorem min_S_odd (n : ℕ) (hn : Odd n) : ∀ (x : Fin n → ℝ), (∀ i, |x i| ≤ 1) → 
  S (List.of_fn x) ≥ -((n - 1) / 2) := 
  by sorry
theorem S_odd_minimum_attainable (n : ℕ) (hn : Odd n) : 
  S (List.of_fn (λ i, if i.val < (n - 1) / 2 then 1 else -1)) = -((n - 1) / 2) := 
  by sorry

end S_neg1_1_neg2div3_S_1_1_neg1_neg1_min_S3_S3_neg1_attainable_min_S_odd_S_odd_minimum_attainable_l149_149077


namespace digit_pairs_satisfying_eq_l149_149778

theorem digit_pairs_satisfying_eq (a b : ℕ) (h : a < 10 ∧ b < 10) :
  (√(a / 9: ℝ) = b / 9) → ((a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 9 ∧ b = 9)) :=
begin
  sorry
end

end digit_pairs_satisfying_eq_l149_149778


namespace max_cube_edge_length_l149_149532

-- Define the given conditions
def sphere_radius : ℝ := 1
def spheres_touch_in_pairs :=
  ∀ (x y z : ℝ × ℝ × ℝ), x ≠ y → y ≠ z → z ≠ x → ∥x - y∥ = sphere_radius * 2 ∧ ∥y - z∥ = sphere_radius * 2 ∧ ∥z - x∥ = sphere_radius * 2

-- State the theorem that asserts the maximum edge length of the cube in this configuration
theorem max_cube_edge_length
  (a : ℝ)
  (h1 : sphere_radius = 1)
  (h2 : spheres_touch_in_pairs) :
  a = (3 * real.sqrt 2 - 2 * real.sqrt 3) / 3 :=
sorry

end max_cube_edge_length_l149_149532


namespace find_x_l149_149711

variable (x : ℝ)

theorem find_x (h : (15 - 2 + 4 / 1 / 2) * x = 77) : x = 77 / (15 - 2 + 4 / 1 / 2) :=
by sorry

end find_x_l149_149711


namespace sum_of_rational_roots_l149_149796

def h (x : ℚ) : ℚ := 2*x^3 - 8*x^2 + 11*x - 3

theorem sum_of_rational_roots : 
  (∑ x in {y | h y = 0 ∧ y.isRational}.toFinset, x) = 3 / 2 := 
by
  sorry

end sum_of_rational_roots_l149_149796


namespace range_of_x_l149_149491

def interval1 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def interval2 : Set ℝ := {x | x < 1 ∨ x > 4}
def false_statement (x : ℝ) : Prop := x ∈ interval1 ∨ x ∈ interval2

theorem range_of_x (x : ℝ) (h : ¬ false_statement x) : x ∈ Set.Ico 1 2 :=
by
  sorry

end range_of_x_l149_149491


namespace cubic_polynomials_common_roots_c_d_l149_149791

theorem cubic_polynomials_common_roots_c_d (c d : ℝ) :
  (∀ (r s : ℝ), r ≠ s ∧
     (r^3 + c*r^2 + 12*r + 7 = 0) ∧ (s^3 + c*s^2 + 12*s + 7 = 0) ∧
     (r^3 + d*r^2 + 15*r + 9 = 0) ∧ (s^3 + d*s^2 + 15*s + 9 = 0)) →
  (c = -5 ∧ d = -6) := 
by
  sorry

end cubic_polynomials_common_roots_c_d_l149_149791


namespace carpenter_problem_l149_149338

-- Declaring noncomputable theory to handle division
noncomputable theory

variables
  (c1 : ℝ) -- Work rate of the first carpenter on the first job
  (c2 : ℝ) -- Work rate of the second carpenter on the first job (unknown initially)
  (c3 : ℝ) -- Work rate of the third carpenter on the last job (what we need to find)
  (t1 : ℝ) (t2 : ℝ) (t3 : ℝ) -- Days taken by each carpenter to finish their respective jobs
  (joint_work_rate : ℝ) -- Combined work rate of two carpenters
  (joint_days : ℝ) -- Days taken by two carpenters to finish a job together
  (correct_days : ℝ) -- Correct number of days for the third carpenter to finish alone

-- Assume the work rates and times for the jobs according to the conditions
axiom condition_1 : c1 = 1 / 7
axiom condition_2 : ∀ x, joint_work_rate = (1 / 7) + x → joint_days = 3 → x = 1 / (21 / 4)
axiom condition_3 : ∀ x, t1 = 8 → t2 = 10 → joint_work_rate = (1 / 10) + x → joint_days = 5 → x = 1 / 10

-- The final proof problem to prove the correct number of days for the third carpenter working alone
theorem carpenter_problem : correct_days = 10 :=
by
  apply condition_3
  sorry

end carpenter_problem_l149_149338


namespace number_of_subsets_of_complement_l149_149476

def U := {x : ℕ | -1 < x ∧ x < 5}
def A := {1, 3}
def complement_U_A := U \ A

theorem number_of_subsets_of_complement :
  ∃ n : ℕ, n = 2 ^ (complement_U_A.to_finset.card) ∧ n = 8 := by
  sorry

end number_of_subsets_of_complement_l149_149476


namespace cos_690_eq_sqrt3_div2_l149_149377

theorem cos_690_eq_sqrt3_div2 :
  cos (690 : ℝ) = (√3 / 2 : ℝ) := by 
  -- given conditions as hypotheses
  have h1 : ∀ α, cos (α + 360) = cos α := sorry,
  have h2 : ∀ θ, cos (-θ) = cos θ := sorry,
  have h3 : cos 30 = √3 / 2 := sorry,
  sorry

end cos_690_eq_sqrt3_div2_l149_149377


namespace sum_of_solutions_abs_eq_six_l149_149678

theorem sum_of_solutions_abs_eq_six :
  let S := { x : ℝ | |3 * x - 9| = 6 } in 
  ∑ x in S, x = 6 :=
by
  sorry

end sum_of_solutions_abs_eq_six_l149_149678


namespace problem_statement_l149_149122

noncomputable def a : ℝ := Real.log 25 / Real.log 10
noncomputable def b : ℝ := Real.log 49 / Real.log 10

theorem problem_statement : 5^(a / b) + 7^(b / a) = 74 := by
  sorry

end problem_statement_l149_149122


namespace population_growth_pattern_l149_149272

variable (birth_rate low : Prop)
variable (death_rate low : Prop)
variable (growth_rate low : Prop)
variable (population_pattern_modern : Prop)

theorem population_growth_pattern (h1 : birth_rate low) (h2 : death_rate low) (h3 : growth_rate low) : population_pattern_modern := 
sorry

end population_growth_pattern_l149_149272


namespace triangle_abc_is_right_triangle_l149_149529

theorem triangle_abc_is_right_triangle (BC AC AB : ℕ) 
  (hBC : BC = 6) 
  (hAC : AC = 8) 
  (hAB : AB = 10) : BC^2 + AC^2 = AB^2 :=
by
  rw [hBC, hAC, hAB]
  norm_num
  sorry

end triangle_abc_is_right_triangle_l149_149529


namespace expression_values_l149_149554

theorem expression_values :
  let a := 1
  let b := -1
  let c := 0
  ∃ d : ℚ, (d = 1 ∨ d = -1) ∧ (a^2 - b^2 + 2 * d - c = 2 ∨ a^2 - b^2 + 2 * d - c = -2) :=
by {
  let a := 1,
  let b := -1,
  let c := 0,
  use 1,
  split,
  exact eq.refl 1,
  simp,
  use -1,
  split,
  exact eq.refl (-1),
  simp,
  sorry
}

end expression_values_l149_149554


namespace area_outside_circle_l149_149980

theorem area_outside_circle (r : ℝ) : 
  let S_ABC := (√3 / 4) * (2 * r) ^ 2,
      S_sector := (real.pi * r^2) / 6 in
  (S_ABC - S_sector) = (r^2 * (3 * √3 - real.pi)) / 6 :=
by
  sorry

end area_outside_circle_l149_149980


namespace max_earning_per_day_l149_149320

def hourly_rate_father := 1500
def hourly_rate_mother := 1200 / 1.5
def hourly_rate_son := 450 / 0.75

def time_cooking_father := 1
def time_cooking_mother := 2
def time_cooking_son := 4
def time_shopping_father := 1
def time_shopping_mother := 2
def time_shopping_son := 3
def time_walking_dog := 1
def time_sleeping := 8
def time_rest_personal := 8

def available_hours_per_day := 24 - time_sleeping - time_rest_personal

theorem max_earning_per_day :
  let total_hours_available := available_hours_per_day in
  let father_work_hours := total_hours_available - time_cooking_father - time_shopping_father in
  let mother_work_hours := total_hours_available in
  let son_work_hours := total_hours_available - time_walking_dog in
  let earning_father := father_work_hours * hourly_rate_father in
  let earning_mother := mother_work_hours * hourly_rate_mother in
  let earning_son := son_work_hours * hourly_rate_son in
  earning_father + earning_mother + earning_son = 19600 :=
begin
  sorry
end

end max_earning_per_day_l149_149320


namespace grasshopper_jump_distance_l149_149264

variable (F G M : ℕ) -- F for frog's jump, G for grasshopper's jump, M for mouse's jump

theorem grasshopper_jump_distance (h1 : F = G + 39) (h2 : M = F - 94) (h3 : F = 58) : G = 19 := 
by
  sorry

end grasshopper_jump_distance_l149_149264


namespace conjugate_plus_modulus_l149_149845

noncomputable def z : Complex := -1 / 2 + Complex.I * (Real.sqrt 3 / 2)

theorem conjugate_plus_modulus (z : Complex) (h : z = -1 / 2 + Complex.I * (Real.sqrt 3 / 2)) : 
    Complex.conj z + Complex.abs z = 1 / 2 - Complex.I * (Real.sqrt 3 / 2) :=
by
  rw [h]
  -- Proof goes here
  sorry

end conjugate_plus_modulus_l149_149845


namespace max_distance_between_lines_l149_149094

theorem max_distance_between_lines (a b c d : ℝ) :
  (∀ (a b : ℝ), a + b = -1 ∧ ab = c) →
  (0 ≤ c ∧ c ≤ 1/8) →
  d = |a - b| / √2 →
  d ≤ √(3/4) :=
by
  intro h1 h2 h3
  sorry

end max_distance_between_lines_l149_149094


namespace sum_thirteen_and_double_a_seven_l149_149901

variable (a : ℕ → ℝ)

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + m) = a n + m * (a 1)

-- Define the sequence sum S_n
def sum_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

-- Main statement
theorem sum_thirteen_and_double_a_seven (h₁ : is_arithmetic_sequence a) 
  (h₂ : 2 * (a 1 + a 4 + a 7) + 3 * (a 9 + a 11) = 24) :
  sum_sequence a 13 + 2 * a 7 = 30 :=
sorry

end sum_thirteen_and_double_a_seven_l149_149901


namespace emilias_tree_eight_weeks_l149_149403

-- Define the sequence based on the given conditions
def branches (n : ℕ) : ℕ
| 0 => 0 -- We start with week 0 having 0 branches (for completeness)
| 1 => 0 -- Week 1 also 0 branches since new branches start at week 2
| 2 => 1 -- Initial condition
| n => branches (n-1) + branches (n-2) -- Recurrence relation

-- The theorem to prove the number of branches at week 8
theorem emilias_tree_eight_weeks : branches 8 = 21 :=
by
  -- Proof steps will go here
  sorry

end emilias_tree_eight_weeks_l149_149403


namespace probability_even_product_and_sum_gt_7_l149_149668

open Set

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_pair_sum_greater_than_seven (p : ℕ × ℕ) : Prop := p.1 + p.2 > 7

def selected_pairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 ≠ p.2}

def valid_pairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  {p | p ∈ selected_pairs s ∧ is_even (p.1 * p.2) ∧ is_pair_sum_greater_than_seven p}

def probability_of_valid_pairs : ℚ :=
  let total_pairs := (Set.card (selected_pairs {1, 2, 3, 4, 5, 6})).nat_cast
  let valid_pairs_count := (Set.card (valid_pairs {1, 2, 3, 4, 5, 6})).nat_cast
  valid_pairs_count / total_pairs

theorem probability_even_product_and_sum_gt_7 :
  probability_of_valid_pairs = 4 / 15 := 
sorry

end probability_even_product_and_sum_gt_7_l149_149668


namespace triangle_inequality_l149_149548

variable (a b c : ℝ)
variable (h1 : a * b + b * c + c * a = 18)
variable (h2 : 1 < a)
variable (h3 : 1 < b)
variable (h4 : 1 < c)

theorem triangle_inequality :
  (1 / (a - 1)^3 + 1 / (b - 1)^3 + 1 / (c - 1)^3) > (1 / (a + b + c - 3)) :=
by
  sorry

end triangle_inequality_l149_149548


namespace problem1a_problem1b_problem2_problem3_l149_149469

open Real

def f (x : ℝ) : ℝ := sin x ^ 2 + sin (2 * x) + 3 * cos x ^ 2

noncomputable def minimum_value (y : ℝ) : Prop :=
  y = 2 - sqrt 2

noncomputable def minimum_value_set (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k * π - (3 * π / 8)

noncomputable def decreasing_intervals (x : ℝ) : Prop :=
  ∃ (k : ℤ), k * π + (π / 8) ≤ x ∧ x ≤ k * π + (5 * π / 8)

noncomputable def range_of_f (y : ℝ) : Prop :=
  1 ≤ y ∧ y ≤ 2 + sqrt 2

theorem problem1a :
  ∃ x, minimum_value (f x) :=
sorry

theorem problem1b :
  ∀ x, minimum_value_set x → f x = 2 - sqrt 2 :=
sorry

theorem problem2 :
  ∀ x, decreasing_intervals x → ∀ y, y = f x → decreasing_intervals y :=
sorry

theorem problem3 :
  ∀ x ∈ Icc (-π/4) (π/4), range_of_f (f x) :=
sorry

end problem1a_problem1b_problem2_problem3_l149_149469


namespace not_unique_triangle_shape_l149_149207

theorem not_unique_triangle_shape :
  ¬(∀ (A B C D E : triangle_data), determines_unique_shape D) :=
sorry

/- Definitions for better clarity -/
structure triangle_data :=
(side1 : ℝ)
(side2 : ℝ)
(angle_incl : ℝ)
(ratio_sides : ℝ)
(median1 : ℝ)
(median2 : ℝ)
(angle_between_medians : ℝ)
(altitude_base1 : ℝ)
(altitude_base2 : ℝ)
(angle_A : ℝ)
(angle_B : ℝ)
(angle_C : ℝ)

def determines_unique_shape (data : triangle_data) : Prop :=
  sorry -- Placeholder to define the conditions which uniquely determine a triangle.

end not_unique_triangle_shape_l149_149207


namespace probability_back_row_taller_l149_149971

theorem probability_back_row_taller :
  let n := 6 in
  let k := 3 in
  let total_students := n! in
  let ways_to_choose_front_row := Nat.choose n k in
  let valid_arrangements := 1 in
  let valid_cases := ways_to_choose_front_row * valid_arrangements in
  let probability := valid_cases.toRat / total_students.toRat in
  probability = (1 : ℚ) / 8 := 
  sorry

end probability_back_row_taller_l149_149971


namespace verify_original_prices_l149_149653

noncomputable def original_price_of_sweater : ℝ := 43.11
noncomputable def original_price_of_shirt : ℝ := 35.68
noncomputable def original_price_of_pants : ℝ := 71.36

def price_of_shirt (sweater_price : ℝ) : ℝ := sweater_price - 7.43
def price_of_pants (shirt_price : ℝ) : ℝ := 2 * shirt_price
def discounted_sweater_price (sweater_price : ℝ) : ℝ := 0.85 * sweater_price
def total_cost (shirt_price pants_price discounted_sweater_price : ℝ) : ℝ := shirt_price + pants_price + discounted_sweater_price

theorem verify_original_prices 
  (total_cost_value : ℝ)
  (price_of_shirt_value : ℝ)
  (price_of_pants_value : ℝ)
  (discounted_sweater_price_value : ℝ) :
  total_cost_value = 143.67 ∧ 
  price_of_shirt_value = original_price_of_shirt ∧ 
  price_of_pants_value = original_price_of_pants ∧
  discounted_sweater_price_value = discounted_sweater_price original_price_of_sweater →
  total_cost (price_of_shirt original_price_of_sweater) 
             (price_of_pants (price_of_shirt original_price_of_sweater)) 
             (discounted_sweater_price original_price_of_sweater) = 143.67 :=
by
  intros
  sorry

end verify_original_prices_l149_149653


namespace distinct_sum_of_five_integers_l149_149929

theorem distinct_sum_of_five_integers 
  (a b c d e : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
  (h_condition : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = -120) : 
  a + b + c + d + e = 25 :=
sorry

end distinct_sum_of_five_integers_l149_149929


namespace tiles_difference_l149_149212

-- Definitions based on given conditions
def initial_blue_tiles : Nat := 20
def initial_green_tiles : Nat := 10
def first_border_green_tiles : Nat := 24
def second_border_green_tiles : Nat := 36

-- Problem statement
theorem tiles_difference :
  initial_green_tiles + first_border_green_tiles + second_border_green_tiles - initial_blue_tiles = 50 :=
by
  sorry

end tiles_difference_l149_149212


namespace min_side_value_l149_149092

-- Definitions based on the conditions provided
variables (a b c : ℕ) (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0)

theorem min_side_value (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0) : c ≥ 7 :=
sorry

end min_side_value_l149_149092


namespace bert_phone_price_l149_149747

theorem bert_phone_price :
  ∃ x : ℕ, x * 8 = 144 := sorry

end bert_phone_price_l149_149747


namespace b_remainder_l149_149931

theorem b_remainder (n : ℕ) (hn : n > 0) : ∃ b : ℕ, b % 11 = 5 :=
by
  sorry

end b_remainder_l149_149931


namespace income_calculation_l149_149155

def income_from_investment (R : ℝ) (MV : ℝ) (I : ℝ) : ℝ := 
  let N := I / MV
  let FV := N * 100
  (R / 100) * FV

theorem income_calculation
  (R : ℝ) (MV : ℝ) (I : ℝ) (Inc : ℝ)
  (hR : R = 10)
  (hMV : MV = 96)
  (hI : I = 6240)
  : income_from_investment R MV I = 650 :=
by
  sorry

end income_calculation_l149_149155


namespace snake_price_correct_l149_149169

-- Define the conditions
def num_snakes : ℕ := 3
def eggs_per_snake : ℕ := 2
def total_eggs : ℕ := num_snakes * eggs_per_snake
def super_rare_multiple : ℕ := 4
def total_revenue : ℕ := 2250

-- The question: How much does each regular baby snake sell for?
def price_of_regular_baby_snake := 250

-- The proof statement
theorem snake_price_correct
  (x : ℕ)
  (h1 : total_eggs = 6)
  (h2 : 5 * x + super_rare_multiple * x = total_revenue)
  :
  x = price_of_regular_baby_snake := 
sorry

end snake_price_correct_l149_149169


namespace relationship_among_abc_l149_149447

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = f (-x + 1)

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, 1 < x → 1 < y → x < y → f x > f y

-- Definition of a, b, c
def a : ℝ := f 2
def b : ℝ := f (Real.log 3 / Real.log 4)
def c : ℝ := f (1 / 2)

-- Statement proving the relationship
theorem relationship_among_abc (h1 : is_even_function f) (h2 : is_monotonically_decreasing f) : a f < c f ∧ c f < b f :=
sorry

end relationship_among_abc_l149_149447


namespace election_result_l149_149665

def votes_A : ℕ := 12
def votes_B : ℕ := 3
def votes_C : ℕ := 15

def is_class_president (candidate_votes : ℕ) : Prop :=
  candidate_votes = max (max votes_A votes_B) votes_C

theorem election_result : is_class_president votes_C :=
by
  unfold is_class_president
  rw [votes_A, votes_B, votes_C]
  sorry

end election_result_l149_149665


namespace determine_velocities_l149_149287

theorem determine_velocities (V1 V2 : ℝ) (h1 : 60 / V2 = 60 / V1 + 5) (h2 : |V1 - V2| = 1)
  (h3 : 0 < V1) (h4 : 0 < V2) : V1 = 4 ∧ V2 = 3 :=
by
  sorry

end determine_velocities_l149_149287


namespace correct_number_of_judgments_l149_149162

variables {a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℕ}

/-- Conditions of the problem -/
def distinct_positive_numbers (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℕ) : Prop :=
  a11 ≠ a12 ∧ a12 ≠ a13 ∧ a11 ≠ a13 ∧ 
  a21 ≠ a22 ∧ a22 ≠ a23 ∧ a21 ≠ a23 ∧ 
  a31 ≠ a32 ∧ a32 ≠ a33 ∧ a31 ≠ a33 ∧ 
  a11 > 0 ∧ a12 > 0 ∧ a13 > 0 ∧
  a21 > 0 ∧ a22 > 0 ∧ a23 > 0 ∧
  a31 > 0 ∧ a32 > 0 ∧ a33 > 0

def rows_are_arithmetic (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℕ) : Prop :=
  a12 = a11 + (a13 - a11) / 2 ∧
  a22 = a21 + (a23 - a21) / 2 ∧
  a32 = a31 + (a33 - a31) / 2

def sums_are_geometric (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℕ) : Prop :=
  ∃ r, a21 + a22 + a23 = r * (a11 + a12 + a13) ∧ 
       a31 + a32 + a33 = r^2 * (a11 + a12 + a13)

/-- Proof that the correct number of judgments is 4 --/
theorem correct_number_of_judgments 
  (h1 : distinct_positive_numbers a11 a12 a13 a21 a22 a23 a31 a32 a33)
  (h2 : rows_are_arithmetic a11 a12 a13 a21 a22 a23 a31 a32 a33)
  (h3 : sums_are_geometric a11 a12 a13 a21 a22 a23 a31 a32 a33) : 
  4 :=
sorry

end correct_number_of_judgments_l149_149162


namespace problem_correct_answer_l149_149503

def second_order_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℕ → ℝ, ∀ n, a (n + 1) - a n = d n ∧ d (n + 1) - d n = d 0 - d 1

def not_geometric (a : ℕ → ℝ) : Prop :=
  ¬∃ r, ∀ n, a (n + 1) = r * a n

def local_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ ∃ r, a j * a j = a i * a k

theorem problem_correct_answer :
  (second_order_arithmetic_sequence (λ n, n) ∧ not_geometric (λ n, n) ∧ local_geometric_sequence (λ n, n))
  ∧ (second_order_arithmetic_sequence (λ n, n ^ 2) ∧ not_geometric (λ n, n ^ 2) ∧ local_geometric_sequence (λ n, n ^ 2)) :=
by
  sorry

end problem_correct_answer_l149_149503


namespace co_complementary_angles_equal_l149_149265

def co_complementary (A : ℝ) : ℝ := 90 - A

theorem co_complementary_angles_equal (A B : ℝ) (h : co_complementary A = co_complementary B) : A = B :=
sorry

end co_complementary_angles_equal_l149_149265


namespace compare_neg_fractions_l149_149388

theorem compare_neg_fractions : 
  (- (8:ℚ) / 21) > - (3 / 7) :=
by sorry

end compare_neg_fractions_l149_149388


namespace num_ordered_pairs_squares_diff_30_l149_149115

theorem num_ordered_pairs_squares_diff_30 :
  ∃ (n : ℕ), n = 0 ∧
  ∀ (m n: ℕ), 0 < m ∧ 0 < n ∧ m ≥ n ∧ m^2 - n^2 = 30 → false :=
by
  sorry

end num_ordered_pairs_squares_diff_30_l149_149115


namespace smallest_powerful_integer_l149_149732

-- Definition of a powerful integer.
def is_powerful (k : ℕ) : Prop :=
  ∃ p q r s t : ℕ, p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t ∧
  p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0 ∧
  k % p^2 = 0 ∧ k % q^3 = 0 ∧ k % r^5 = 0 ∧ k % s^7 = 0 ∧ k % t^{11} = 0

-- Prove that the smallest powerful integer is 2^34.
theorem smallest_powerful_integer : ∃ k, is_powerful k ∧ ∀ n, is_powerful n → k ≤ n :=
  ⟨2^34, 
   by {
     use [16, 8, 4, 2, 1],
     repeat { split }; try { norm_num }, sorry,
   },
   by {
     intros n hn,
     sorry 
   }⟩

end smallest_powerful_integer_l149_149732


namespace find_k_of_parallel_l149_149481

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, a = (λ * b.1, λ * b.2)

theorem find_k_of_parallel (k : ℝ) 
  (h : vector_parallel (1, k) (2, 1)) : k = 1 / 2 :=
sorry

end find_k_of_parallel_l149_149481


namespace ellipse_equation_and_AB_distance_l149_149522

open Real

theorem ellipse_equation_and_AB_distance :
  ∃ (a b : ℝ), (a > b ∧ b > 0) ∧ 
  (2^2 / a^2 + 1^2 / b^2 = 1) ∧ 
  (sqrt (1 - b^2 / a^2) = sqrt 3 / 2) ∧ 
  (a^2 = 8 ∧ b^2 = 2) ∧ 
  (let C := ((x : ℝ) (y : ℝ), x^2 / 8 + y^2 / 2 = 1) in
  let l := (x : ℝ, -1 / 2 * x + 1) in
  let f := λ x, (x^2 + 2 * x - 2) = 0 in
  (|AB|, points_line_ellipse_intersect := solve_roots f,
  |AB| = sqrt 15)) :=
sorry

end ellipse_equation_and_AB_distance_l149_149522


namespace max_product_arithmetic_sequence_l149_149894

theorem max_product_arithmetic_sequence (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_pos : ∀ n, a n > 0) (h_sum : ∑ k in finset.range 10, a k = 30) : 
  ∃ a5 a6, a5 = a 4 ∧ a6 = a 5 ∧ a5 * a6 = 9 :=
by
  sorry

end max_product_arithmetic_sequence_l149_149894


namespace equal_lengths_l149_149568

noncomputable def triangle (A K L : Type) : Prop :=
  ∃ (B C P : Type), 
    (P ∈ (BC : Set Point)) ∧
    (lines_through B C = lines_through A P)

theorem equal_lengths {A K L : Type} 
  (h : triangle A K L) :
  ∀ (PA PJ : Length), PA = PJ := 
by 
  intro PA PJ
  sorry

end equal_lengths_l149_149568


namespace part1_part2_l149_149848

noncomputable def f (x : Real) : Real :=
  2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x) - 1

noncomputable def h (x t : Real) : Real :=
  f (x + t)

theorem part1 (t : Real) (ht : 0 < t ∧ t < Real.pi / 2) :
  (h (-Real.pi / 6) t = 0) → t = Real.pi / 3 :=
sorry

theorem part2 (A B C : Real) (hA : 0 < A ∧ A < Real.pi / 2) (hA1 : h A (Real.pi / 3) = 1) :
  1 < ((Real.sqrt 3 - 1) * Real.sin B + Real.sqrt 2 * Real.sin (Real.pi / 2 - B)) ∧
  ((Real.sqrt 3 - 1) * Real.sin B + Real.sqrt 2 * Real.sin (Real.pi / 2 - B)) ≤ 2 :=
sorry

end part1_part2_l149_149848


namespace Catherine_overall_score_combined_test_l149_149380

theorem Catherine_overall_score_combined_test :
  (let correct_in_40 := 0.65 * 40 in
   let correct_in_30 := 0.75 * 30 in
   let correct_in_20 := 0.85 * 20 in
   let total_correct := correct_in_40 + correct_in_30.round + correct_in_20 in
   let overall_score := total_correct / 90 in
   overall_score.round) = 0.73 :=
by
  sorry

end Catherine_overall_score_combined_test_l149_149380


namespace range_tan_l149_149273

theorem range_tan (x : ℝ) (h : x ∈ Icc (-π / 6) (5 * π / 12)) :
  -2 * Real.sqrt 3 ≤ 2 * Real.tan(x - π / 6) ∧ 2 * Real.tan(x - π / 6) ≤ 2 :=
sorry

end range_tan_l149_149273


namespace find_cos_beta_l149_149451

noncomputable theory

open Real

variables (e1 e2 : EuclideanSpace ℝ (Fin 2)) (α β : ℝ)
variables (a b : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def angle_between_e1_e2 := (e1.dot e2) = (1/3) * (∥e1∥ * ∥e2∥)
def vector_a := a = 3 • e1 - 2 • e2
def vector_b := b = 3 • e1 - e2

-- The proof problem
theorem find_cos_beta : 
  angle_between_e1_e2 e1 e2 α → 
  vector_a e1 e2 a → 
  vector_b e1 e2 b → 
  cos β = (2 * sqrt 2) / 3 := sorry

end find_cos_beta_l149_149451


namespace percentage_deducted_from_list_price_l149_149742

theorem percentage_deducted_from_list_price (CP SP profit LP DA : ℝ) (h₁ : CP = 51.50) (h₂ : SP = 67.76) (h₃ : profit = 0.25 * CP) (h₄ : LP = SP + (0.25 * CP)) (h₅ : DA = LP - SP) :
  (DA / LP) * 100 ≈ 4.995 :=
by
  -- We intend to prove that:
  -- (DA / LP) * 100
  -- where DA = LP - SP and LP is derived from the selling price SP adjusted by the profit derived from CP.
  sorry

end percentage_deducted_from_list_price_l149_149742


namespace sum_of_a_b_l149_149127

theorem sum_of_a_b (a b : ℕ) : 
  (∏ i in (finset.range (a-3+1)).map (λ n, if n+3 = a then (a / b:ℚ) else (n+4 / n+3:ℚ)) = 16) → 
  a + b = 95 :=
by
  sorry

end sum_of_a_b_l149_149127


namespace conjugate_in_fourth_quadrant_l149_149844

def is_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem conjugate_in_fourth_quadrant :
  let z := (3 + complex.i) / (1 - complex.i) in
  is_fourth_quadrant z.conj :=
by
  sorry

end conjugate_in_fourth_quadrant_l149_149844


namespace cindy_total_time_to_travel_one_mile_l149_149383

-- Definitions for the conditions
def run_speed : ℝ := 3 -- Cindy's running speed in miles per hour.
def walk_speed : ℝ := 1 -- Cindy's walking speed in miles per hour.
def run_distance : ℝ := 0.5 -- Distance run by Cindy in miles.
def walk_distance : ℝ := 0.5 -- Distance walked by Cindy in miles.

-- Theorem statement
theorem cindy_total_time_to_travel_one_mile : 
  ((run_distance / run_speed) + (walk_distance / walk_speed)) * 60 = 40 := 
by
  sorry

end cindy_total_time_to_travel_one_mile_l149_149383


namespace not_constant_fn_constant_value_1_constant_value_2_constant_fn_range_l149_149765

theorem not_constant_fn (x : ℝ) : (x^2 + 2 * x + 2) ≠ 1 ∧ ((x^2 + 2 * x + 2) ≥ 1) := sorry

theorem constant_value_1 (a b c : ℝ) (h₁ : 3 * abs (a * x^2 + b * x + c) + 2 = 2) (h₂ : a > 0) (h₃ : c < 0) :
  3 * abs (a * x^2 + b * x + c) + 2 ≥ 2 := sorry

theorem constant_value_2 (a b c : ℕ) (h₁ : a + b + c = 12) (h₂ : b ≠ 0) (h₃ : ∃ r : ℚ, b = a * r ∧ c = a * r^2) :
  4 ≤ a ∧ a ≤ 16 ∧ 
  (a = 4 → (∀ (b = 4) (c = 4) ∨ (b = -8 ∧ c = 16))) ∧
  (a = 16 → (b = -8 ∧ c = 4)) := sorry

theorem constant_fn_range (a b c : ℝ) (h₁ : a > 0) (h₂ : b > a) (h₃ : b^2 - 4 * a * c = 0) (h₄ : ax^2 + bx + c = 0) :
  (a + b + c) / (a + b) > m → m ≤ 9 / 8 := sorry

end not_constant_fn_constant_value_1_constant_value_2_constant_fn_range_l149_149765


namespace find_other_number_l149_149242

theorem find_other_number (hcf lcm a b: ℕ) (hcf_value: hcf = 12) (lcm_value: lcm = 396) (a_value: a = 36) (gcd_ab: Nat.gcd a b = hcf) (lcm_ab: Nat.lcm a b = lcm) : b = 132 :=
by
  sorry

end find_other_number_l149_149242


namespace total_amount_paid_l149_149736

-- Definitions of the conditions
def P_orig := 150 -- Original price of the jacket in dollars
def D_init_percent := 0.30 -- Initial discount percentage as a decimal
def D_coupon := 10 -- Additional discount amount in dollars
def T := 0.10 -- Sales tax rate as a decimal

-- Problem statement: Prove the final amount paid is 104.50 dollars
theorem total_amount_paid : 
  let P_after_init_discount := P_orig * (1 - D_init_percent) in
  let P_after_coupon := P_after_init_discount - D_coupon in
  let A := P_after_coupon * (1 + T) in
  A = 104.50 := by
  sorry

end total_amount_paid_l149_149736


namespace values_of_a_exist_l149_149319

theorem values_of_a_exist (a x y : ℝ) :
  (x - a)^2 = 8 * (2 * y - x + a - 2) ∧ 
  (1 - sqrt y) / (1 - sqrt (x / 2)) = 1 →
  a ≥ 2 →
  (a ≠ 2 ∧ a ≠ 10 → (∃ x y, (x - a)^2 = 8 * (2 * y - x + a - 2) ∧ (1 - sqrt y) / (1 - sqrt (x / 2)) = 1)) ∧
  (a = 10 → x = 18 ∧ y = 9) :=
by
  sorry

end values_of_a_exist_l149_149319


namespace n_is_2_mod_4_l149_149179

theorem n_is_2_mod_4 (n : ℕ) (x : Fin n^2 → Fin n) (σ : Fin n → Fin n):
  (∀ i : Fin n, σ i ≠ i) ∧ 
  (∀ i : Fin n^2, x i ∈ Fin n) ∧ 
  (∀ i : Fin (n^2 - 1), (x i ≠ x (i + 1))) ∧ 
  (∃ k : ℤ, ∀ i : Fin n^2, x ((i : ℤ + k) % (n^2) : Fin n^2) = σ (x i))
  → n % 4 = 2 := 
by 
  sorry

end n_is_2_mod_4_l149_149179


namespace smallest_prime_factor_2457_l149_149677

theorem smallest_prime_factor_2457 : ∃ p: ℕ, prime p ∧ p ∣ 2457 ∧ ∀ q: ℕ, q ∣ 2457 ∧ prime q → p ≤ q :=
by
  sorry

end smallest_prime_factor_2457_l149_149677


namespace arithmetic_sequence_common_difference_l149_149900

theorem arithmetic_sequence_common_difference (a_1 a_4 a_5 d : ℤ) 
  (h1 : a_1 + a_5 = 10) 
  (h2 : a_4 = 7) 
  (h3 : a_4 = a_1 + 3 * d) 
  (h4 : a_5 = a_1 + 4 * d) : 
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l149_149900


namespace sum_coefficients_l149_149773

noncomputable def S (x : ℕ) := (Finset.range n).sum (λ k => (k + 1) * x^(k + 1))

theorem sum_coefficients (n : ℕ) (x : ℕ):
  let expanded_sum := (S(x) * S(x))
  (Finset.Icc (n + 1) (2 * n)).sum (λ k => coefficient expanded_sum k) = (n * (n + 1) * (5 * n^2 + 5 * n + 2)) / 24 :=
sorry

end sum_coefficients_l149_149773


namespace students_drawn_from_class_A_l149_149508

-- Given conditions
def classA_students : Nat := 40
def classB_students : Nat := 50
def total_sample : Nat := 18

-- Predicate that checks if the number of students drawn from Class A is correct
theorem students_drawn_from_class_A (students_from_A : Nat) : students_from_A = 9 :=
by
  sorry

end students_drawn_from_class_A_l149_149508


namespace original_number_is_5_div_4_l149_149211

-- Define the condition in Lean.
def condition (y : ℚ) : Prop :=
  1 - 1 / y = 1 / 5

-- Define the theorem to prove that y = 5 / 4 given the condition.
theorem original_number_is_5_div_4 (y : ℚ) (h : condition y) : y = 5 / 4 :=
by
  sorry

end original_number_is_5_div_4_l149_149211


namespace coefficient_binomial_expansion_l149_149249

theorem coefficient_binomial_expansion (a : ℝ) :
  let term := (binomial_coef 3 1) * (a ^ 2) * ((- (sqrt 3) / 6) ^ 1) in
  term = - (sqrt 3) / 2 → (a = 1 ∨ a = -1) :=
begin
  sorry
end

end coefficient_binomial_expansion_l149_149249


namespace find_x_l149_149419

theorem find_x (x : ℝ) : 
  (∀ (y : ℝ), 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := sorry

end find_x_l149_149419


namespace find_length_of_a_in_triangle_l149_149907

theorem find_length_of_a_in_triangle (b c : ℝ) (cos_B_minus_C : ℝ) :
  b = 7 ∧ c = 6 ∧ cos_B_minus_C = 37/40 → (∃ a, a = real.sqrt (66.1)) := 
by
  intros h
  rcases h with ⟨hb, hc, hcos⟩
  use real.sqrt 66.1
  sorry

end find_length_of_a_in_triangle_l149_149907


namespace solve_equation_l149_149003

noncomputable def equation (x : ℝ) : Prop :=
    real.cbrt (18 * x - 2) + real.cbrt (16 * x + 2) = 5 * real.cbrt x

theorem solve_equation (x : ℝ) :
    equation x ↔ x = 0 ∨ x = 31 / 261 ∨ x = -35 / 261 :=
by
  intros
  sorry

end solve_equation_l149_149003


namespace remainder_sum_div_2019_eq_2_l149_149068

-- Definitions for problem conditions
def has_2017_divisors (n : ℕ) : Prop := 
  ∃ p : ℕ, prime p ∧ (n = p ^ 2016)

def not_coprime_with_2018 (n : ℕ) : Prop := 
  ¬ coprime n 2018

-- The specific problem statement
theorem remainder_sum_div_2019_eq_2 : 
  (∑ n in (finset.filter (λ n, has_2017_divisors n ∧ not_coprime_with_2018 n) (finset.range 2018)), n) % 2019 = 2 :=
sorry

end remainder_sum_div_2019_eq_2_l149_149068


namespace steps_Tom_by_time_Matt_reaches_220_l149_149197

theorem steps_Tom_by_time_Matt_reaches_220 (rate_Matt rate_Tom : ℕ) (time_Matt_time_Tom : ℕ) (steps_Matt steps_Tom : ℕ) :
  rate_Matt = 20 →
  rate_Tom = rate_Matt + 5 →
  steps_Matt = 220 →
  time_Matt_time_Tom = steps_Matt / rate_Matt →
  steps_Tom = steps_Matt + time_Matt_time_Tom * 5 →
  steps_Tom = 275 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4, h2, h1] at h5
  norm_num at h5
  exact h5

end steps_Tom_by_time_Matt_reaches_220_l149_149197


namespace lattice_points_count_l149_149906

theorem lattice_points_count : 
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 16}.to_finset.card = 16 := 
sorry

end lattice_points_count_l149_149906


namespace fraction_reach_impossible_l149_149462

theorem fraction_reach_impossible :
  ¬ ∃ (a b : ℕ), (2 + 2013 * a) / (3 + 2014 * b) = 3 / 5 := by
  sorry

end fraction_reach_impossible_l149_149462


namespace polynomial_irreducible_l149_149939

open Polynomial

variables (n : ℕ) (hn : n ≥ 3) (p : ℕ) [fact (Nat.Prime p)]

theorem polynomial_irreducible :
  irreducible (X^n + C (p^2) * X^(n-1) + C (p^2) * X^(n-2) + ... + C (p^2) * X + C (p^2)) :=
sorry

end polynomial_irreducible_l149_149939


namespace fraction_to_terminating_decimal_l149_149000

theorem fraction_to_terminating_decimal (a b : ℕ) (h : a = 49 ∧ b = 2^3 * 5^4) : ↑(a : ℝ) / ↑(b : ℝ) = 6.125 :=
by
  cases h with ha hb
  rw [ha, hb]
  norm_num
  sorry

end fraction_to_terminating_decimal_l149_149000


namespace basketball_player_ft_rate_l149_149336

theorem basketball_player_ft_rate :
  ∃ P : ℝ, 1 - P^2 = 16 / 25 ∧ P = 3 / 5 := sorry

end basketball_player_ft_rate_l149_149336


namespace P_plus_Q_divisible_by_1987_l149_149218

def P : ℕ := (List.range' 1 993).map (λ n, 2 * n - 1).prod
def Q : ℕ := (List.range' 1 993).map (λ n, 2 * n).prod

theorem P_plus_Q_divisible_by_1987 : (P + Q) % 1987 = 0 := by
  sorry

end P_plus_Q_divisible_by_1987_l149_149218


namespace p_sufficient_not_necessary_for_q_l149_149041

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l149_149041


namespace shelly_total_money_l149_149228

-- Define the conditions
def num_of_ten_dollar_bills : ℕ := 10
def num_of_five_dollar_bills : ℕ := num_of_ten_dollar_bills - 4

-- Problem statement: How much money does Shelly have in all?
theorem shelly_total_money :
  (num_of_ten_dollar_bills * 10) + (num_of_five_dollar_bills * 5) = 130 :=
by
  sorry

end shelly_total_money_l149_149228


namespace calculate_expression_l149_149376

theorem calculate_expression :
  ( (3 / 5) ^ (-4) * (2 / 3) ^ 5 * 6 = 8 / 13 ) :=
by
  have h1: (3 / 5)^(-4) = (5 / 3)^4 := by
    simp [pow_neg, inv_eq_div, div_pow]
  have h2: (5 / 3)^4 = 625 / 81 := by
    norm_num
  have h3: (2 / 3)^5 = 32 / 243 := by
    norm_num
  have h4: ( (5 / 3) ^ 4 * (32 / 243) * 6 = 8 / 13 ) := by
    field_simp
    norm_num
    linarith
  rw [h1, h2, h3, h4]

end calculate_expression_l149_149376


namespace coefficient_x9_sum_l149_149500

theorem coefficient_x9_sum (x : ℕ) :
  let poly_sum := (1 + x)^9 + (1 + x)^10 + (1 + x)^11 + (1 + x)^12 + (1 + x)^13 + (1 + x)^14 in
  (poly_sum.coeff 9 = 3003) :=
by
  sorry

end coefficient_x9_sum_l149_149500


namespace find_k_of_parallel_l149_149482

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, a = (λ * b.1, λ * b.2)

theorem find_k_of_parallel (k : ℝ) 
  (h : vector_parallel (1, k) (2, 1)) : k = 1 / 2 :=
sorry

end find_k_of_parallel_l149_149482


namespace jogged_distance_is_13_point_5_l149_149730

noncomputable def jogger_distance (x t d : ℝ) : Prop :=
  d = x * t ∧
  d = (x + 3/4) * (3 * t / 4) ∧
  d = (x - 3/4) * (t + 3)

theorem jogged_distance_is_13_point_5:
  ∃ (x t d : ℝ), jogger_distance x t d ∧ d = 13.5 :=
by
  sorry

end jogged_distance_is_13_point_5_l149_149730


namespace p_sufficient_not_necessary_q_l149_149051

theorem p_sufficient_not_necessary_q (x : ℝ) :
  (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros h
  cases h with h1 h2
  split
  case left => linarith
  case right => linarith

end p_sufficient_not_necessary_q_l149_149051


namespace cake_pieces_kept_l149_149569

theorem cake_pieces_kept (total_pieces : ℕ) (two_fifths_eaten : ℕ) (extra_pieces_eaten : ℕ)
  (h1 : total_pieces = 35)
  (h2 : two_fifths_eaten = 2 * total_pieces / 5)
  (h3 : extra_pieces_eaten = 3)
  (correct_answer : ℕ)
  (h4 : correct_answer = total_pieces - (two_fifths_eaten + extra_pieces_eaten)) :
  correct_answer = 18 := by
  sorry

end cake_pieces_kept_l149_149569


namespace solve_for_y_l149_149680

-- Define the given condition as a Lean definition
def equation (y : ℝ) : Prop :=
  (2 / y) + ((3 / y) / (6 / y)) = 1.2

-- Theorem statement proving the solution given the condition
theorem solve_for_y (y : ℝ) (h : equation y) : y = 20 / 7 := by
  sorry

-- Example usage to instantiate and make use of the definition
example : equation (20 / 7) := by
  unfold equation
  sorry

end solve_for_y_l149_149680


namespace max_value_f_in_interval_l149_149884

noncomputable def f (x p q : ℝ) : ℝ := x^2 + p * x + q

noncomputable def g (x : ℝ) : ℝ := x + 1 / (x^2)

theorem max_value_f_in_interval :
  ∃ (p q : ℝ), (∀ (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2),
    f x p q = g x → (∃ (c : ℝ), 1 ≤ c ∧ c ≤ 2 ∧ ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 →
    f x p q ≥ f c p q)) → 
    ∃ (c : ℝ), 1 ≤ c ∧ c ≤ 2 ∧ f c p q = 4 - 5 / 2 * real.cbrt 2 + real.cbrt 4 :=
sorry

end max_value_f_in_interval_l149_149884


namespace half_plus_five_l149_149859

theorem half_plus_five (n : ℕ) (h : n = 16) : n / 2 + 5 = 13 := by
  sorry

end half_plus_five_l149_149859


namespace hyperbola_vertex_distance_l149_149010

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  4 * x^2 + 24 * x - 4 * y^2 + 16 * y + 44 = 0 →
  2 = 2 :=
by
  intros x y h
  sorry

end hyperbola_vertex_distance_l149_149010


namespace find_lines_through_p_and_intersecting_circle_l149_149824

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

noncomputable def passes_through (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = l P.1

noncomputable def chord_length (c p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

theorem find_lines_through_p_and_intersecting_circle :
  ∃ l : ℝ → ℝ, (passes_through l (-2, 3)) ∧
  (∃ p1 p2 : ℝ × ℝ, trajectory_equation p1.1 p1.2 ∧ trajectory_equation p2.1 p2.2 ∧
  chord_length (1, 2) p1 p2 = 8^2) :=
by
  sorry

end find_lines_through_p_and_intersecting_circle_l149_149824


namespace function_range_g_increasing_f_eq_g_no_solution_a_range_l149_149099

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1/2 then -1/2 * x + 1/4
  else if 1/2 < x ∧ x ≤ 1 then 2*x^2 / (x + 2)
  else 0 -- out of bound handling, can be ignored

def g (a x : ℝ) : ℝ :=
  a * Real.sin (π/3 * x + 3/2 * π) - 2*a + 2

theorem function_range (x : ℝ) :
  (0 ≤ x ∧ x ≤ 1/2 ∧ 0 ≤ -1/2 * x + 1/4 ∧ -1/2 * x + 1/4 ≤ 1/4)
  ∨ (1/2 < x ∧ x ≤ 1 ∧ 1/5 < 2 * (x + 2) + 8/(x + 2) - 8 ∧ 2 * (x + 2) + 8/(x + 2) - 8 ≤ 2/3) :=
sorry

theorem g_increasing (a : ℝ) (h₀ : 0 < a) (x : ℝ) :
  (0 ≤ x ∧ x ≤ 1) → ( π/3 * x + 3/2 * π ∈ (Icc (3/2 * π) (3/2 * π + π/3)) → MonotoneOn (g a) (Icc 0 1)) :=
sorry

theorem f_eq_g_no_solution (a : ℝ) (h₀ : 0 < a) (x : ℝ) :
  ¬(0 ≤ x ∧ x ≤ 1 ∧ f x = g a x) :=
sorry

theorem a_range (x₁ x₂ : ℝ) (h₀ : 0 ≤ x₁ ∧ x₁ ≤ 1 ∧ 0 ≤ x₂ ∧ x₂ ≤ 1) (h₁ : f x₁ = g x₂) :
  4/9 ≤ a ∧ a ≤ 4/5 :=
sorry

end

end function_range_g_increasing_f_eq_g_no_solution_a_range_l149_149099


namespace joy_quadrilateral_rod_selection_l149_149915

/--
Joy has 40 thin rods, each with integer lengths from 1 cm to 40 cm.
She places rods of lengths 4 cm, 8 cm, and 17 cm on a table.
She wants to select one more rod to form a quadrilateral with a positive area.
Prove that there are 21 possible rods.
-/
theorem joy_quadrilateral_rod_selection :
  let rods := { n | 1 ≤ n ∧ n ≤ 40 }
  let selected := {4, 8, 17}
  let valid_rods := {d | 6 ≤ d ∧ d ≤ 28 ∧ d ≠ 8 ∧ d ≠ 17}
  card valid_rods = 21 := sorry

end joy_quadrilateral_rod_selection_l149_149915


namespace region_perimeter_l149_149904

theorem region_perimeter
  (ten_sides_length : ∀ (i : ℕ), i < 10 → length i = 2)
  (total_area : area = 104) :
  perimeter = 52.4 := 
sorry

end region_perimeter_l149_149904


namespace probability_Laurent_greater_Chloe_l149_149753

noncomputable def even_numbers_in_interval (a b : ℝ) : set ℝ :=
  {x | a ≤ x ∧ x ≤ b ∧ (∃ n : ℕ, x = 2 * n)}

noncomputable def probability_greater (a b c d : ℝ) : ℝ :=
  let S1 := even_numbers_in_interval a b
  let S2 := even_numbers_in_interval c d
  let favorable_area := 2040100 + 8160800
  let total_area := (finset.card (S1.to_finset)) * (finset.card (S2.to_finset))
  favorable_area / total_area

theorem probability_Laurent_greater_Chloe :
  probability_greater 0 2020 0 4040 = 3 / 4 :=
sorry

end probability_Laurent_greater_Chloe_l149_149753


namespace num_assignments_l149_149663

/-- 
Mr. Wang originally planned to grade at a rate of 6 assignments per hour.
After grading for 2 hours, he increased his rate to 8 assignments per hour,
finishing 3 hours earlier than initially planned. 
Prove that the total number of assignments is 84. 
-/
theorem num_assignments (x : ℕ) (h : ℕ) (H1 : 6 * h = x) (H2 : 8 * (h - 5) = x - 12) : x = 84 :=
by
  sorry

end num_assignments_l149_149663


namespace problem_l149_149850
noncomputable def f (x : ℝ) : ℝ := 1 + 1/x + Real.log x + Real.log x / x
def g (x : ℝ) : ℝ := 2 * Real.exp (x - 1) / (x * Real.exp x + 1)

theorem problem 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ)
  (h₁ : ∀ x : ℝ, f x = 1 + 1/x + Real.log x + Real.log x / x)
  (h₂ : ∀ x : ℝ, g x = 2 * Real.exp (x - 1) / (x * Real.exp x + 1)) :
  (a = 1 → (∀ x > 0, 1/x + x ≥ 1)) 
  ∧ (∀ x > 0, f x = 1 + 1/x + Real.log x + Real.log x / x 
  → ∀ x₁ x₂ ≥ 0, f'(x₁) ≥ f'(x₂)  → x₁ ≥ x₂) 
  ∧ (∀ x > 1, f(x) > (Real.exp 1 + 1) * g(x)) :=
sorry

end problem_l149_149850


namespace symmetric_point_R_l149_149826

variable (a b : ℝ) 

def symmetry_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def symmetry_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_R :
  let M := (a, b)
  let N := symmetry_x M
  let P := symmetry_y N
  let Q := symmetry_x P
  let R := symmetry_y Q
  R = (a, b) := by
  unfold symmetry_x symmetry_y
  sorry

end symmetric_point_R_l149_149826


namespace product_of_marbles_l149_149981

theorem product_of_marbles (R B : ℕ) (h1 : R - B = 12) (h2 : R + B = 52) : R * B = 640 := by
  sorry

end product_of_marbles_l149_149981


namespace Fonzie_paid_l149_149022

-- Given conditions
def AuntBee_payment : ℝ := 8000
def Lapis_payment : ℝ := 9000
def Treasure_value : ℝ := 900000
def Lapis_share : ℝ := 337500

-- Theorem to prove
theorem Fonzie_paid :
  ∃ F : ℝ, (9000 / (F + 8000 + 9000)) = (337500 / 900000) ∧
           3 * (F + 17000) = 8 * 9000 ∧
           3 * F + 51000 = 72000 ∧
           F = 7000 :=
by {
  -- Proof steps go here
  sorry,
}

end Fonzie_paid_l149_149022


namespace rope_cut_into_segments_l149_149214

theorem rope_cut_into_segments:
  ∀ (n : ℕ), n = 5 → let segments := 2^n in segments + 1 = 33 :=
begin
  intro n,
  intros hn,
  simp only [hn],
  let segments := 2 ^ 5,
  exact calc
    segments + 1 = 32 + 1 : by rw pow_succ,
                     -- 2^5 = 2^4 * 2 = 16 * 2 = 32
                ... = 33 : by norm_num,
end

end rope_cut_into_segments_l149_149214


namespace count_positive_integers_proof_l149_149800

noncomputable def countPositiveIntegersInRange : ℕ :=
  let boundedIntegers := {x : ℕ | 2 ≤ x ∧ x ≤ 6}
  let validIntegers := {x : boundedIntegers | 50 < (x + 3) * (x + 3) ∧ (x + 3) * (x + 3) < 100}
  validIntegers.card

theorem count_positive_integers_proof : countPositiveIntegersInRange = 3 :=
sorry

end count_positive_integers_proof_l149_149800


namespace distance_focus_directrix_parabola_l149_149011

def parabola_focus (a : ℝ) (b : ℝ) (k : ℝ) : Prop :=
  a = 0 ∧ b = -1/16 ∧ k = 1/16

theorem distance_focus_directrix_parabola : 
  parabola_focus 0 (-1/16) (1/16) → 
  (abs((1/16) - (-1/16))) / sqrt(1^2) = 1/8 :=
by 
  intro h
  sorry

end distance_focus_directrix_parabola_l149_149011


namespace number_of_baggies_l149_149208

/-- Conditions -/
def cookies_per_bag : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41

/-- Question: Prove the total number of baggies Olivia can make is 6 --/
theorem number_of_baggies : (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := sorry

end number_of_baggies_l149_149208


namespace score_in_first_round_l149_149323

theorem score_in_first_round (cards : List ℕ) (scores : List ℕ) 
  (total_rounds : ℕ) (last_round_score : ℕ) (total_score : ℕ) : 
  cards = [2, 4, 7, 13] ∧ scores = [16, 17, 21, 24] ∧ total_rounds = 3 ∧ last_round_score = 2 ∧ total_score = 16 →
  ∃ first_round_score, first_round_score = 7 := by
  sorry

end score_in_first_round_l149_149323


namespace barefoot_kids_l149_149657

theorem barefoot_kids (total_kids kids_socks kids_shoes kids_both : ℕ) 
  (h1 : total_kids = 22) 
  (h2 : kids_socks = 12) 
  (h3 : kids_shoes = 8) 
  (h4 : kids_both = 6) : 
  (total_kids - (kids_socks - kids_both + kids_shoes - kids_both + kids_both) = 8) :=
by
  -- following sorry to skip proof.
  sorry

end barefoot_kids_l149_149657


namespace probability_of_selecting_prime_is_two_thirds_l149_149606

open Finset

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def probability_of_prime (balls : Finset ℕ) : ℚ :=
  let primes := balls.filter is_prime
  primes.card / balls.card

theorem probability_of_selecting_prime_is_two_thirds :
  probability_of_prime (finset.range 8 \ {0, 1}) = 2 / 3 :=
by sorry

end probability_of_selecting_prime_is_two_thirds_l149_149606


namespace marathon_yards_l149_149727

theorem marathon_yards (marathons : ℕ) (yards_per_marathon : ℕ) (miles_per_marathon : ℕ) (yards_in_mile : ℕ) (y : ℕ)
  (h1 : marathons = 10)
  (h2 : yards_per_marathon = 385)
  (h3 : miles_per_marathon = 26)
  (h4 : yards_in_mile = 1760)
  (h5 : 0 ≤ y)
  (h6 : y < 1760) :
  let total_yards := marathons * yards_per_marathon in
  y = total_yards % yards_in_mile :=
by
  sorry

end marathon_yards_l149_149727


namespace area_of_region_l149_149670

theorem area_of_region :
  let eqn := λ (x y : ℝ), x^2 + y^2 + 5 = 6 * y - 10 * x + 7 in
  ∃ (r : ℝ), (∀ x y, eqn x y ↔ (x + 5)^2 + (y - 3)^2 = r^2) ∧ r^2 = 36 ∧ π * 36 = 36 * π :=
by
  let eqn := λ (x y : ℝ), x^2 + y^2 + 5 = 6 * y - 10 * x + 7
  use 6
  refine ⟨_, _, _⟩
  sorry
  sorry
  sorry

end area_of_region_l149_149670


namespace f_3_add_f_neg_1_f_explicit_f_is_odd_l149_149985

-- Condition: f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

-- f(x) = 2^x - 1 if x >= 0
def f (x : ℝ) : ℝ :=
  if x >= 0 then 2^x - 1 else 1 - 2^(-x)

-- Proof that f(3) + f(-1) = 6
theorem f_3_add_f_neg_1 : f 3 + f (-1) = 6 := by
  sorry

-- Proof of the explicit formula
theorem f_explicit (x : ℝ) : f x = 
  if x >= 0 then 2^x - 1 else 1 - 2^(-x) := by
  sorry

-- Proving the function is odd
theorem f_is_odd : is_odd_function f := by
  sorry

end f_3_add_f_neg_1_f_explicit_f_is_odd_l149_149985


namespace magnitude_difference_l149_149480

variables {α : Type*} [inner_product_space ℝ α]
variables (a b : α)
variables (ha : ∥a∥ = 2) (hb : ∥b∥ = 3) (dot_cond : ⟪a, b - a⟫ = 1)

theorem magnitude_difference : ∥a - b∥ = 2 * real.sqrt 2 :=
by sorry

end magnitude_difference_l149_149480


namespace area_circle_through_D_P_F_l149_149152

theorem area_circle_through_D_P_F :
  ∃ (r : ℝ), ∀ (D F P : ℝ^2), 
    let r := 8 * Real.sqrt 3 / 3 in
    EquilateralTriangle D F 8 →
    IncenterOfTriangle P D F →
    AreaOfCircle r = 64 * Real.pi / 3 :=
sorry

end area_circle_through_D_P_F_l149_149152


namespace angle_bisectors_pass_through_midpoint_l149_149071

-- Definitions and conditions
variable (A B C D E : Point)
variable [Trapezoid ABCD]
variable (h_parallel : AD ∥ BC)
variable (h_sum : AB = AD + BC)
variable (h_midpoint : Midpoint E CD)

-- The theorem statement 
theorem angle_bisectors_pass_through_midpoint :
  (Bisects (Angle BAD) (Line AE)) ∧ (Bisects (Angle ABC) (Line BE)) := 
sorry

end angle_bisectors_pass_through_midpoint_l149_149071


namespace tan_alpha_in_third_quadrant_l149_149078

theorem tan_alpha_in_third_quadrant (α : Real) (h1 : Real.sin α = -5/13) (h2 : ∃ k : ℕ, π < α + k * 2 * π ∧ α + k * 2 * π < 3 * π) : 
  Real.tan α = 5/12 :=
sorry

end tan_alpha_in_third_quadrant_l149_149078


namespace sort_first_list_correct_sort_second_list_correct_l149_149370

def firstList := [1090, 1009, 1100, 1909]
def secondList := [9999, 8900, 9990, 8909, 10000]
def sortedFirstList := [1909, 1100, 1090, 1009]
def sortedSecondList := [10000, 9999, 9990, 8909, 8900]

theorem sort_first_list_correct :
  List.sort (>) firstList = sortedFirstList := by
  sorry

theorem sort_second_list_correct :
  List.sort (>) secondList = sortedSecondList := by
  sorry

end sort_first_list_correct_sort_second_list_correct_l149_149370


namespace radius_of_circle_with_given_spherical_coordinates_l149_149999

theorem radius_of_circle_with_given_spherical_coordinates:
  (radius_of_circle (ρ φ : ℝ) (h1 : ρ = 2) (h2 : φ = π / 4) : ℝ) 
  : radius_of_circle 2 (π / 4) = sqrt 2 := 
sorry

end radius_of_circle_with_given_spherical_coordinates_l149_149999


namespace equation_of_line_l149_149423

def n1 : ℝ × ℝ × ℝ := (2, -1, -1)
def n2 : ℝ × ℝ × ℝ := (3, 2, 1)
def p0 : ℝ × ℝ × ℝ := (4, -3, 2)

theorem equation_of_line :
  let a := (n1.2.2 * n2.2 - n1.2 * n2.2.2, n1.2.2 * n2.1 - n1.1 * n2.2.2, n1.1 * n2.2 - n1.2 * n2.1) in
  ∃ t : ℝ,
  ∀ x y z : ℝ,
  x = p0.1 + t * a.1 ∧
  y = p0.2 + t * a.2 ∧
  z = p0.2.2 + t * a.2.2 →
  (x - 4) / 1 = (y + 3) / -5 ∧ (y + 3) / -5 = (z - 2) / 7 :=
by
  sorry

end equation_of_line_l149_149423


namespace solve_for_x_l149_149607

noncomputable def equation (x : ℝ) : Prop :=
  4^(Real.sqrt (16^x)) = 16^(Real.sqrt (2^x))

theorem solve_for_x :
  ∃ x : ℝ, equation x ∧ x = 2 / 3 :=
by
  let x := 2 / 3
  use x
  sorry

end solve_for_x_l149_149607


namespace find_k_of_quadratic_eq_ratio_3_to_1_l149_149770

theorem find_k_of_quadratic_eq_ratio_3_to_1 (k : ℝ) :
  (∃ (x : ℝ), x ≠ 0 ∧ (x^2 + 8 * x + k = 0) ∧
              (∃ (r : ℝ), x = 3 * r ∧ 3 * r + r = -8)) → k = 12 :=
by {
  sorry
}

end find_k_of_quadratic_eq_ratio_3_to_1_l149_149770


namespace price_of_other_pieces_l149_149175

theorem price_of_other_pieces (total_spent : ℕ) (total_pieces : ℕ) (price_piece1 : ℕ) (price_piece2 : ℕ) 
  (remaining_pieces : ℕ) (price_remaining_piece : ℕ) (h1 : total_spent = 610) (h2 : total_pieces = 7)
  (h3 : price_piece1 = 49) (h4 : price_piece2 = 81) (h5 : remaining_pieces = (total_pieces - 2))
  (h6 : total_spent - price_piece1 - price_piece2 = remaining_pieces * price_remaining_piece) :
  price_remaining_piece = 96 := 
by
  sorry

end price_of_other_pieces_l149_149175


namespace range_of_k_l149_149877

noncomputable def equation_has_one_real_root (k : ℝ) : Prop :=
  ∃ x : ℝ, real.log (k * x) = 2 * real.log (x + 1) ∧ 
    (∀ y : ℝ, real.log (k * y) = 2 * real.log (y + 1) → x = y)

theorem range_of_k :
  (k = 4 ∨ k < 0) ↔ ∀ k : ℝ, equation_has_one_real_root k → (k = 4 ∨ k < 0) :=
by
  sorry

end range_of_k_l149_149877


namespace gcd_probability_is_22_over_35_l149_149666

-- Define the original set and the condition of choosing three numbers
def original_set : set ℕ := {1, 2, 3, 4, 5, 6, 7}
def choose_three (s : set ℕ) : set (set ℕ) :=
  {t | t ⊆ s ∧ t.card = 3}

-- Define a function to check if all pairs in a subset are relatively prime
def all_pairs_rel_prime (subset : set ℕ) : Prop :=
  ∀ a b ∈ subset, a ≠ b → Nat.gcd a b = 1

-- Define the probability calculation
def gcd_probability (s : set ℕ) : ℚ :=
  let total_subsets := (choose_three s).card
  let valid_subsets := (choose_three s).filter all_pairs_rel_prime
  (valid_subsets.card : ℚ) / (total_subsets : ℚ)

-- The main statement to be proved
theorem gcd_probability_is_22_over_35 :
  gcd_probability original_set = 22 / 35 :=
by
  sorry

end gcd_probability_is_22_over_35_l149_149666


namespace negation_of_p_l149_149189

def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A (x : ℤ) : Prop := is_odd x
def B (x : ℤ) : Prop := is_even x
def p : Prop := ∀ x, A x → B (2 * x)

theorem negation_of_p : ¬ p ↔ ∃ x, A x ∧ ¬ B (2 * x) :=
by
  -- problem statement equivalent in Lean 4
  sorry

end negation_of_p_l149_149189


namespace find_expression_and_range_l149_149466

noncomputable theory

def A := (1 / 2 : ℝ)
def ω := 2
def φ := π / 6
def f (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Lean statement equivalent to the mathematical proof problem:
theorem find_expression_and_range (x : ℝ) (h1 : 0 < A) (h2 : 0 < ω)
  (h3 : 0 < φ) (h4 : φ < π) (h5 : ∀ x, f (x) = A * Real.sin (ω * x + φ))
  (h6 : (f (2 * π / 3) = - (1 / 2) )) :
  (f x = (1 / 2) * Real.sin (2 * x + (π / 6))) ∧ 
  ∀ x ∈ interval (- π / 6) (π / 3), - 1 / 4 ≤ f x ∧ f x ≤ 1 / 2 :=
begin
  sorry,
end

end find_expression_and_range_l149_149466


namespace relationship_between_abc_l149_149367

variables (f : ℝ → ℝ)
variables (a b c : ℝ)

axiom even_function : ∀ x, f x = f (-x)
axiom monotonic_on_negative : ∀ x y : ℝ, x < y → y ≤ 0 → f x > f y
axiom a_def : a = f (Real.logb 2 6)
axiom b_def : b = f (Real.logb 2 4.9)
axiom c_def : c = f (Real.exp (0.8 * Real.log 2))

theorem relationship_between_abc : a > b ∧ b > c :=
begin
  sorry
end

end relationship_between_abc_l149_149367


namespace initial_population_correct_l149_149996

noncomputable def initial_population (P_final : ℕ) (r : ℝ) (t : ℕ) : ℝ :=
  P_final / (1 + r) ^ t

theorem initial_population_correct :
  ∀ {P_final : ℕ} {r : ℝ} {t : ℕ},
    P_final = 297500 ∧ r = 0.07 ∧ t = 10 →
    initial_population P_final r t ≈ 151195 :=
by
  intros P_final r t h
  -- The proof will go here
  sorry

end initial_population_correct_l149_149996


namespace p_sufficient_not_necessary_q_l149_149048

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l149_149048


namespace clothing_price_l149_149172

theorem clothing_price
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (price_piece_1 : ℕ)
  (price_piece_2 : ℕ)
  (num_remaining_pieces : ℕ)
  (total_remaining_pieces_price : ℕ)
  (price_remaining_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  price_piece_1 = 49 →
  price_piece_2 = 81 →
  num_remaining_pieces = 5 →
  total_spent = price_piece_1 + price_piece_2 + total_remaining_pieces_price →
  total_remaining_pieces_price = price_remaining_piece * num_remaining_pieces →
  price_remaining_piece = 96 :=
by
  intros h_total_spent h_num_pieces h_price_piece_1 h_price_piece_2 h_num_remaining_pieces h_total_remaining_price h_price_remaining_piece
  sorry

end clothing_price_l149_149172


namespace sample_candy_solution_l149_149888

variable (C T : ℝ)
variable (U : ℝ)

def sample_candy_problem : Prop :=
  C = 22 ∧ T = 23.913043478260867 ∧ U = 23.913043478260867 - 22

theorem sample_candy_solution : U = 1.913043478260867 :=
by
  unfold sample_candy_problem
  intro h
  cases h
  rw h_left
  rw h_right.left
  rw h_right.right
  norm_num
  sorry

end sample_candy_solution_l149_149888


namespace no_full_conspiracies_in_same_lab_l149_149255

theorem no_full_conspiracies_in_same_lab
(six_conspiracies : Finset (Finset (Fin 10)))
(h_conspiracies : ∀ c ∈ six_conspiracies, c.card = 3)
(h_total : six_conspiracies.card = 6) :
  ∃ (lab1 lab2 : Finset (Fin 10)), lab1 ∩ lab2 = ∅ ∧ lab1 ∪ lab2 = Finset.univ ∧ ∀ c ∈ six_conspiracies, ¬(c ⊆ lab1 ∨ c ⊆ lab2) :=
by
  sorry

end no_full_conspiracies_in_same_lab_l149_149255


namespace find_k_l149_149483

-- Define the vectors a and b.
def vector_a (k : ℝ) : ℝ × ℝ := (1, k)
def vector_b : ℝ × ℝ := (2, 1)

-- Define the condition that vector a is parallel to vector b.
def is_parallel (a b : ℝ × ℝ) : Prop := ∃ (λ : ℝ), a = (λ * b.1, λ * b.2)

-- Prove that if vector a is parallel to vector b, then k = 1/2.
theorem find_k (k : ℝ) (h : is_parallel (vector_a k) vector_b) : k = 1/2 :=
  sorry

end find_k_l149_149483


namespace simplify_sqrt_fraction_l149_149603

theorem simplify_sqrt_fraction : (sqrt 6 / sqrt 10) = (sqrt 15 / 5) := 
by
  sorry

end simplify_sqrt_fraction_l149_149603


namespace multiple_of_x_l149_149874

theorem multiple_of_x (k x y : ℤ) (hk : k * x + y = 34) (hx : 2 * x - y = 20) (hy : y^2 = 4) : k = 4 :=
sorry

end multiple_of_x_l149_149874


namespace ratio_of_rats_l149_149965

theorem ratio_of_rats (x y : ℝ) (h : (0.56 * x) / (0.84 * y) = 1 / 2) : x / y = 3 / 4 :=
sorry

end ratio_of_rats_l149_149965


namespace symmetry_axis_eq_pi_l149_149983

noncomputable def function_y (x : ℝ) : ℝ :=
  sin (2 * x + (Real.pi / 3)) * cos (x - (Real.pi / 6)) +
  cos (2 * x + (Real.pi / 3)) * sin (x - (Real.pi / 6))

theorem symmetry_axis_eq_pi : (∃ x : ℝ, x = Real.pi) ↔
  (function_y (x : ℝ) = function_y (Real.pi - x)) :=
by {
  sorry
}

end symmetry_axis_eq_pi_l149_149983


namespace pow_sum_nineteen_eq_zero_l149_149834

variable {a b c : ℝ}

theorem pow_sum_nineteen_eq_zero (h₁ : a + b + c = 0) (h₂ : a^3 + b^3 + c^3 = 0) : a^19 + b^19 + c^19 = 0 :=
sorry

end pow_sum_nineteen_eq_zero_l149_149834


namespace decorations_cost_correct_l149_149580

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end decorations_cost_correct_l149_149580


namespace collinear_P_Q_A_l149_149560

-- Given an acute-angled triangle ABC
variables {A B C M N P Q : Type}
variables [acute_angled_triangle : triangle A B C]
variables [on_side_M : point_on_side M A B]
variables [on_side_N : point_on_side N A C]

-- Define circles with diameters BN and CM
noncomputable def circle_BN : circle BN := sorry
noncomputable def circle_CM : circle CM := sorry

-- Assume P and Q are the intersection points of the above circles
variables (P Q : Type)
variables [intersection_points : intersect circle_BN circle_CM P Q]

-- Prove P, Q and the vertex A are collinear
theorem collinear_P_Q_A : collinear P Q A :=
sorry

end collinear_P_Q_A_l149_149560


namespace perfect_square_factors_of_7200_l149_149114

noncomputable def nat_factors : ℕ → list ℕ
| n := n.factors

def is_perfect_square (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

def is_factor (n d : ℕ) : Prop :=
d ∣ n

theorem perfect_square_factors_of_7200 :
  have hr7200 : nat_factors 7200 = [2, 2, 2, 2, 3, 3, 5, 5],
  have ha : list.count 2 (nat_factors 7200) = 4,
  have hb : list.count 3 (nat_factors 7200) = 2,
  have hc : list.count 5 (nat_factors 7200) = 2,
  ∑ d in (list.sublists' [2, 2, 2, 2, 3, 3, 5, 5]).map (list.prod), is_perfect_square d ∧ is_factor 7200 d = 12 := sorry

end perfect_square_factors_of_7200_l149_149114


namespace triangle_CD_length_l149_149136

theorem triangle_CD_length :
  ∀ (A B C D : Type) [Triangle ABC]
    (h_angle_ABC : angle ABC = 150)
    (h_AB : length AB = 5)
    (h_BC : length BC = 6)
    (h_perpendicular_AB : Perpendicular (line_through A B) (line_through B D))
    (h_perpendicular_BC : Perpendicular (line_through B C) (line_through C D)),
    length CD = 6 * Real.tan (15 * Real.pi / 180) :=
by
  sorry

end triangle_CD_length_l149_149136


namespace hybrids_with_full_headlights_l149_149137

-- Definitions for the conditions
def total_cars : ℕ := 600
def percentage_hybrids : ℕ := 60
def percentage_one_headlight : ℕ := 40

-- The proof statement
theorem hybrids_with_full_headlights :
  (percentage_hybrids * total_cars) / 100 - (percentage_one_headlight * (percentage_hybrids * total_cars) / 100) / 100 = 216 :=
by
  sorry

end hybrids_with_full_headlights_l149_149137


namespace arithmetic_sequence_remainder_l149_149293

theorem arithmetic_sequence_remainder :
  let a : ℕ → ℕ := λ n, 8 * n - 5,
      sum := ∑ i in finset.range 44, a (i + 1)
  in sum % 8 = 4 :=
by
  let a : ℕ → ℕ := λ n, 8 * n - 5
  let sum := ∑ i in finset.range 44, a (i + 1)
  have h1 : a 1 = 3 := by rfl
  have h44 : a 44 = 347 := by norm_num
  have ha : (∀ n, a n % 8 = 3) := by
    intro n
    exact mod_eq_of_lt (lt_add_of_pos_right (8 * n) (by norm_num))

  calc
    sum % 8 = (∑ i in finset.range 44, (a (i + 1) % 8)) % 8 : by
        rw finset.sum_mod
    ... = (∑ _ in finset.range 44, 3) % 8  : by
        simp [ha]
    ... = (44 * 3) % 8 : by
        exact finset.sum_const_mod 44 3 8
    ... = 4 : by
        norm_num

end arithmetic_sequence_remainder_l149_149293


namespace time_to_cross_l149_149717

-- Define the necessary parameters and conversions
def km_per_hr_to_m_per_s (speed_km_per_hr : ℕ) : ℝ :=
  speed_km_per_hr * (1000 / 3600)

-- The given conditions
def train_length : ℝ := 250
def train_speed_km_per_hr : ℕ := 75

-- Convert the speed
def train_speed_m_per_s : ℝ := km_per_hr_to_m_per_s train_speed_km_per_hr

-- The expected answer
def expected_time_seconds : ℝ := 12

-- The proof statement
theorem time_to_cross (length : ℝ) (speed_km_per_hr : ℕ) : 
  length / (km_per_hr_to_m_per_s speed_km_per_hr) = expected_time_seconds := 
by
  -- The proof would go here
  sorry

end time_to_cross_l149_149717


namespace find_M_l149_149864

theorem find_M : ∀ M : ℕ, (10 + 11 + 12 : ℕ) / 3 = (2024 + 2025 + 2026 : ℕ) / M → M = 552 :=
by
  intro M
  sorry

end find_M_l149_149864


namespace probability_of_divisibility_l149_149498

noncomputable def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def is_prime_digit_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, is_prime_digit d

noncomputable def is_divisible_by_3_and_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0

theorem probability_of_divisibility (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999 ∨ 10 ≤ n ∧ n ≤ 99) →
  is_prime_digit_number n →
  ¬ is_divisible_by_3_and_4 n :=
by
  intros h1 h2
  sorry

end probability_of_divisibility_l149_149498


namespace houses_in_block_l149_149600

theorem houses_in_block (candies_per_house : ℕ) (total_candies_per_block : ℕ) 
  (h1 : candies_per_house = 7) (h2 : total_candies_per_block = 35) :
  (total_candies_per_block / candies_per_house) = 5 :=
by
  rw [h1, h2]
  norm_num
  sorry

end houses_in_block_l149_149600


namespace solve_inequality_l149_149324

variable (x : ℝ)

theorem solve_inequality : 3 * (x + 2) - 1 ≥ 5 - 2 * (x - 2) → x ≥ 4 / 5 :=
by
  sorry

end solve_inequality_l149_149324


namespace true_propositions_l149_149095

-- Proposition 1: Hyperbola and Ellipse foci condition
def hyperbola (x y : ℝ) := (x^2 / 25) - (y^2 / 9) = 1
def ellipse (x y : ℝ) := (x^2 / 35) + y^2 = 1

-- Proposition 2: Incorrect derivative of ln x
def deriv_ln (x : ℝ) := (ln x)' = 1 / (x * ln e)

-- Proposition 3: Correct derivative of tan x
def deriv_tan (x : ℝ) := (tan x)' = 1 / (cos x)^2

-- Proposition 4: Incorrect derivative of u/v
def deriv_quot (u v : ℝ → ℝ) (x : ℝ) :=
  (u x / v x)' = (u x * (v x)' - v x * (u x)') / (v x)^2

-- Proposition 5: Quadratic condition
def quadratic_condition (x : ℝ) := x^2 - 3 * x + 3 ≠ 0

theorem true_propositions : 
  (hyperbola x y ∧ ellipse x y → ∃ c : ℝ, c^2 = 34) ∧
  ¬ deriv_ln x ∧
  deriv_tan x ∧
  ¬ deriv_quot u v x ∧
  quadratic_condition x :=
sorry

end true_propositions_l149_149095


namespace sum_of_roots_l149_149184

def greatest_integer_not_exceeding (x : ℝ) : ℤ := ⌊x⌋

theorem sum_of_roots (x : ℝ) (hx: x^2 - 4 * (greatest_integer_not_exceeding x) + 3 = 0) : 
  ∃ y z w : ℝ, (x = y ∨ x = z ∨ x = w) ∧ y + z + w = 4 + real.sqrt 5 :=
sorry

end sum_of_roots_l149_149184


namespace max_distance_A_l149_149974

-- Conditions from part a)
section

variable (A B C D E F A' : Point)
variable (ABCD : Square A B C D)
variable (side_length_ABCD : A.distance_to(B) = 4)
variable (E_midpoint_AB : E = midpoint A B)
variable (F_on_BC : is_on_line F B C)
variable (folding_ADF_coincide : fold ADE DE = A')
variable (folding_DCF_coincide : fold DCF DF = A')

-- Theorem to be proved
theorem max_distance_A'_to_DEF : 
  ∀ E_midpoint_AB F_on_BC folding_ADF_coincide folding_DCF_coincide,
  distance_to_plane A' DEF = 4 * sqrt 5 / 5 :=
sorry

end

end max_distance_A_l149_149974


namespace problem1_problem2_problem3_problem4_l149_149706

-- (1) Prove that \(\sqrt{18} - \sqrt{8} + \sqrt{\frac{1}{8}} = \frac{5 \sqrt{2}}{4}\)
theorem problem1 : sqrt 18 - sqrt 8 + sqrt (1 / 8) = (5 * sqrt 2) / 4 := 
sorry

-- (2) Prove that \((1 + \sqrt{3})(2 - \sqrt{3}) = \sqrt{3} - 1\)
theorem problem2 : (1 + sqrt 3) * (2 - sqrt 3) = sqrt 3 - 1 := 
sorry

-- (3) Prove that \(\frac{\sqrt{15} + \sqrt{60}}{\sqrt{3}} - 3 \sqrt{5} = -5 \sqrt{5}\)
theorem problem3 : (sqrt 15 + sqrt 60) / sqrt 3 - 3 * sqrt 5 = -5 * sqrt 5 := 
sorry

-- (4) Prove that \((\sqrt{7} + \sqrt{3})(\sqrt{7} - \sqrt{3}) - \sqrt{36} = -2\)
theorem problem4 : (sqrt 7 + sqrt 3) * (sqrt 7 - sqrt 3) - sqrt 36 = -2 := 
sorry

end problem1_problem2_problem3_problem4_l149_149706


namespace angle_BFD_108_l149_149903

theorem angle_BFD_108 (D E : Point) (F : Point) (A B C : Point) 
  (hD_on_AB : lies_on AB D) (hE_on_AB : lies_on AB E) (hF_on_AC : lies_on AC F)
  (hDA_DF_DE : DA = DF ∧ DA = DE)
  (hBE_EF : BE = EF)
  (hBF_BC : BF = BC)
  (hAngle_ABC_2ACB : ∠ABC = 2 * ∠ACB) : 
  ∠BFD = 108 :=
sorry

end angle_BFD_108_l149_149903


namespace limit_seq_tends_to_neg_infinity_l149_149701

noncomputable def sequence (n : ℕ) : ℝ := 
  (Real.sqrt ((n^5 + 1) * (n^2 - 1)) - n * Real.sqrt (n * (n^4 + 1))) / n

theorem limit_seq_tends_to_neg_infinity :
  filter.tendsto sequence filter.at_top filter.at_bot :=
sorry

end limit_seq_tends_to_neg_infinity_l149_149701


namespace power_of_two_last_digit_product_divisible_by_6_l149_149967

theorem power_of_two_last_digit_product_divisible_by_6 (n : Nat) (h : 3 < n) :
  ∃ d m : Nat, (2^n = 10 * m + d) ∧ (m * d) % 6 = 0 :=
by
  sorry

end power_of_two_last_digit_product_divisible_by_6_l149_149967


namespace part1_part2_l149_149523

-- Define angle alpha and conditions in problem
variable (α : ℝ)

/-- Conditions given in the problem -/
def terminal_side_condition : Prop := ∃ (k : ℝ), y = 2*x ∧ x ≤ 0

-- First part: Prove that tan α = 2
theorem part1 (h1 : terminal_side_condition α) : Real.tan α = 2 := sorry

-- Second part: Prove the complex trigonometric identities given that tan α = 2
theorem part2 (h2 : Real.tan α = 2) : 
  ∀ α : ℝ, 
  (Real.cos (α - Real.pi) - 2 * Real.cos (Real.pi / 2 + α)) / 
  (Real.sin (α - 3 * Real.pi / 2) - Real.sin α) = -3 := sorry

end part1_part2_l149_149523


namespace steve_take_home_pay_l149_149612

-- Defining the conditions
def annual_salary : ℕ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℕ := 800

-- Defining the taxes function
def taxes (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the healthcare function
def healthcare (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the total deductions function
def total_deductions (salary : ℕ) (tax_rate : ℝ) (healthcare_rate : ℝ) (dues : ℕ) : ℝ :=
  (taxes salary tax_rate) + (healthcare salary healthcare_rate) + dues

-- Defining the take-home pay function
def take_home_pay (salary : ℕ) (deductions : ℝ) : ℝ :=
  salary - deductions

-- Using definitions to assert the take-home pay
theorem steve_take_home_pay : take_home_pay annual_salary (total_deductions annual_salary tax_rate healthcare_rate union_dues) = 27200 :=
by
  -- proof has been skipped
  sorry

end steve_take_home_pay_l149_149612


namespace trig_identity_simplification_l149_149230

theorem trig_identity_simplification :
  (sin (Real.pi / 12) + sin (Real.pi / 6) + sin (Real.pi / 4) + sin (Real.pi / 3) + sin (5 * Real.pi / 12)) /
  (cos (Real.pi / 18) * cos (Real.pi / 9) * cos (Real.pi / 6)) = 4 * Real.sqrt 6 :=
by
  sorry

end trig_identity_simplification_l149_149230


namespace constants_are_minus_6_and_minus_1_l149_149842

noncomputable def find_constants (a c : ℝ) : Prop :=
(a * (1/2)^2 + 5 * (1/2) + c = 0) ∧
(a * (1/3)^2 + 5 * (1/3) + c = 0) ∧
(a < 0)

theorem constants_are_minus_6_and_minus_1 : ∃ (a c : ℝ), find_constants a c ∧ a = -6 ∧ c = -1 :=
by
  exists -6, -1
  split 
  · simp
    sorry
  · simp
    sorry

end constants_are_minus_6_and_minus_1_l149_149842


namespace specific_toss_probability_l149_149296

open ProbabilityTheory

-- Define the probability of getting a specific sequence of outcomes from tossing 5 fair coins
def coin_toss_probability : ℕ := 5 
def fair_coin_outcome_probability : ℝ := 1 / 2

-- Define the specific sequence we're analyzing (TTHHT)
def specific_sequence (sequence : list (ℕ → bool)) : Prop :=
  sequence = [λ i, i = 1, λ i, i = 2, λ i, i = 6, λ i, i = 6, λ i, i = 1]

theorem specific_toss_probability :
  probability (sequence [λ i, i = 1, λ i, i = 2, λ i, i = 6, λ i, i = 6, λ i, i = 1]) = (1/2)^5 :=
sorry

end specific_toss_probability_l149_149296


namespace p_sufficient_not_necessary_q_l149_149045

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l149_149045


namespace roots_of_polynomial_and_sqrt2_lt_real_l149_149602

theorem roots_of_polynomial_and_sqrt2_lt_real (h₁ : ∀ x, (x * x - 198 * x + 1 = 0) → (x > 1/198 ∧ x < 197.9949494949)) :
  (∀ x, (x * x - 198 * x + 1 = 0) → 1/198 < x ∧ x < 197.9949494949) →
  (sqrt 2 < (1 + 41 / 100 + 421356 / (10^8 * (1 - 10^(-6))))) ∧ (sqrt 2 < 1.41421356) :=
by
  intros h_poly h_interval
  have h_root_interval : ∀ x, (x * x - 198 * x + 1 = 0) → (x > 1/198 ∧ x < 197.9949494949) := h_interval
  have h_real_sqrt_approx : sqrt 2 < (1 + 41 / 100 + 421356 / (10^8 * (1 - 10^(-6)))) := sorry
  have h_real_sqrt : sqrt 2 < 1.41421356 := sorry
  exact ⟨h_real_sqrt_approx, h_real_sqrt⟩

end roots_of_polynomial_and_sqrt2_lt_real_l149_149602


namespace number_of_drunk_drivers_l149_149890

theorem number_of_drunk_drivers (D : ℕ) :
  let speeders := 7 * D - 3
  let seatbelt_violators := 2 * D
  (D + speeders + seatbelt_violators = 105) → D = 11 :=
by {
  intros h,
  sorry
}

end number_of_drunk_drivers_l149_149890


namespace find_x0_l149_149464

def f (x : ℝ) : ℝ :=
if x ≤ 0 then (1 / 2) ^ x else real.log (x + 2) / real.log 2

theorem find_x0 : ∃ x0 : ℝ, (f x0 = 2) ∧ (x0 = 2 ∨ x0 = -1) :=
begin
  use 2,  -- providing the first possible solution
  split,
  { simp [f], norm_num },
  { left, refl },
  use -1,  -- providing the second possible solution
  split,
  { simp [f], norm_num },
  { right, refl },
end 

end find_x0_l149_149464


namespace number_and_sum_of_f_3_l149_149936

namespace ProofProblem

def S := { x : ℝ // x ≠ 0 }

def f (x : S) : S

axiom f_property : ∀ (x y : S), (x.val + y.val ≠ 0) → f x + f y = f ⟨(x.val + y.val) / f (⟨x.val * y.val, _⟩ : S).val, _⟩

theorem number_and_sum_of_f_3 : (∃ n s : ℝ, n = 1 ∧ s = 1 / 9 ∧ n * s = 1 / 9) :=
sorry

end ProofProblem

end number_and_sum_of_f_3_l149_149936


namespace Allyson_age_is_28_l149_149297

-- Define the conditions of the problem
def Hiram_age : ℕ := 40
def add_12_to_Hiram_age (h_age : ℕ) : ℕ := h_age + 12
def twice_Allyson_age (a_age : ℕ) : ℕ := 2 * a_age
def condition (h_age : ℕ) (a_age : ℕ) : Prop := add_12_to_Hiram_age h_age = twice_Allyson_age a_age - 4

-- Define the theorem to be proven
theorem Allyson_age_is_28 (a_age : ℕ) (h_age : ℕ) (h_condition : condition h_age a_age) (h_hiram : h_age = Hiram_age) : a_age = 28 :=
by sorry

end Allyson_age_is_28_l149_149297


namespace modulus_of_complex_l149_149876

theorem modulus_of_complex (z : ℂ) (hz : z = 2 / (1 + I)) : complex.abs z = real.sqrt 2 := 
by
  rw [hz],
  sorry

end modulus_of_complex_l149_149876


namespace find_solutions_l149_149004

noncomputable def solutions := { x : ℝ | (∛(18 * x - 2) + ∛(16 * x + 2)) = 5 * ∛x }

theorem find_solutions :
  solutions = {0, (-2 + Real.sqrt 1048) / 261, (-2 - Real.sqrt 1048) / 261} :=
sorry

end find_solutions_l149_149004


namespace tan_half_alpha_pos_l149_149837

theorem tan_half_alpha_pos (α : ℝ) (k : ℤ) (h : sin α > 0) : tan (α / 2) > 0 := 
sorry

end tan_half_alpha_pos_l149_149837


namespace sin_phi_eq_sqrt3_div_2_l149_149553

/-- Let a, b, and c be nonzero vectors, no two of which are parallel, 
such that (b × c) × a = (1/2) * ‖c‖ * ‖a‖ * b. Let φ be the angle 
between c and a. Prove that sin φ = √3 / 2. -/
theorem sin_phi_eq_sqrt3_div_2
  (a b c : ℝ^3) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (hbc : ¬ (∃ k : ℝ, b = k • c))
  (hab : ¬ (∃ k : ℝ, a = k • b))
  (hac : ¬ (∃ k : ℝ, a = k • c))
  (h : (b × c) × a = (1/2) * ‖c‖ * ‖a‖ * b) 
  : real.sin (real.angle_between a c) = √3 / 2 := 
by 
  sorry

end sin_phi_eq_sqrt3_div_2_l149_149553


namespace math_problem_solution_l149_149441

open Real

noncomputable def math_problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a + b + c + d = 4) : Prop :=
  (b / sqrt (a + 2 * c) + c / sqrt (b + 2 * d) + d / sqrt (c + 2 * a) + a / sqrt (d + 2 * b)) ≥ (4 * sqrt 3) / 3

theorem math_problem_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) :
  math_problem a b c d ha hb hc hd h := by sorry

end math_problem_solution_l149_149441


namespace factor_expression_l149_149774

theorem factor_expression (c : ℝ) : 270 * c^2 + 45 * c - 15 = 15 * c * (18 * c + 2) :=
by
  sorry

end factor_expression_l149_149774


namespace length_MN_right_triangle_l149_149158

-- Define the conditions of the right triangle, points M and N, and their distances.
theorem length_MN_right_triangle (A B C M N : Point)
  (hABC : ∠ABC = 90°)
  (hAC : dist A C = 5)
  (hBC : dist B C = 12)
  (hAM : dist A M = 12)
  (hBN : dist B N = 5)
  (h : is_on_line_segment A B M)
  (h' : is_on_line_segment A B N) :
  dist M N = 4 := 
sorry

end length_MN_right_triangle_l149_149158


namespace accurate_value_D_l149_149147

theorem accurate_value_D (D : ℝ) (ε : ℝ) (H_D : D = 9.71439) (H_ε : ε = 0.00289) : 
  let D_upper := D + ε,
      D_lower := D - ε in
  (Real.floor (D_upper * 10) / 10 = Real.floor (D_lower * 10) / 10) →
  Real.floor (D_upper * 10) / 10 = 9.7 :=
by
  have H1 : D_upper = 9.71728 := sorry
  have H2 : D_lower = 9.71150 := sorry
  exact sorry

end accurate_value_D_l149_149147


namespace segment_CM_length_l149_149234

theorem segment_CM_length (A B C D M N : ℝ) (side : ℝ) (h1 : side = 4)
  (h2 : ∀ P : ℝ × ℝ, P ∈ [⟨C, M⟩, ⟨C, N⟩] → quadrilateral ABCD.area / 4 = triangle C P M.area):
  C M = 2 * sqrt 5 :=
by
  sorry

end segment_CM_length_l149_149234


namespace prove_circumcenter_l149_149958

variables {A B C P : Type} [AddGroup P] [Module ℝ P]

-- Let α, β, γ be the angles of the triangle at vertices A, B, and C.
variables (α β γ : ℝ)
variables (PA PB PC : P)

-- Definitions of the vector conditions
def vector_cond1 := PA + PB + PC = 0
def vector_cond2 := sin α • PA + sin β • PB + sin γ • PC = 0
def vector_cond3 := sin (2 * α) • PA + sin (2 * β) • PB + sin (2 * γ) • PC = 0

-- Defining the circumcenter using the conditions
def is_circumcenter (α β γ : ℝ) (PA PB PC : P) : Prop :=
  sin (2 * α) • PA + sin (2 * β) • PB + sin (2 * γ) • PC = 0 → 
  unique (λ P : P, sin (2 * α) • PA + sin (2 * β) • PB + sin (2 * γ) • PC = 0)

theorem prove_circumcenter (α β γ : ℝ) (PA PB PC : P) :
  is_circumcenter α β γ PA PB PC :=
by
  sorry -- Proof omitted

end prove_circumcenter_l149_149958


namespace number_of_bipartite_graphs_l149_149757

variables (X Y : Type)

-- Conditions: |X| = p and |Y| = q
variables (p q : Nat) (hX : Fintype.card X = p) (hY : Fintype.card Y = q)

-- Theorem: Number of bipartite graphs between these sets
theorem number_of_bipartite_graphs : 
  (∃ (E : (X → Y → Prop)), True) = 2 ^ (p * q) :=
sorry

end number_of_bipartite_graphs_l149_149757


namespace remaining_homes_l149_149543

theorem remaining_homes (total_homes : ℕ) (first_hour_fraction : ℚ) (second_hour_fraction : ℚ) : 
  total_homes = 200 →
  first_hour_fraction = 2/5 →
  second_hour_fraction = 60/100 →
  let
    first_distributed := first_hour_fraction * total_homes,
    remaining_after_first := total_homes - first_distributed,
    second_distributed := second_hour_fraction * remaining_after_first,
    remaining_after_second := remaining_after_first - second_distributed
  in
  remaining_after_second = 48 := 
by
  intros h_total h_first_fraction h_second_fraction,
  let first_distributed := first_hour_fraction * total_homes,
  let remaining_after_first := total_homes - first_distributed,
  let second_distributed := second_hour_fraction * remaining_after_first,
  let remaining_after_second := remaining_after_first - second_distributed,
  sorry -- proof goes here

end remaining_homes_l149_149543


namespace shelly_total_money_l149_149229

-- Define the conditions
def num_of_ten_dollar_bills : ℕ := 10
def num_of_five_dollar_bills : ℕ := num_of_ten_dollar_bills - 4

-- Problem statement: How much money does Shelly have in all?
theorem shelly_total_money :
  (num_of_ten_dollar_bills * 10) + (num_of_five_dollar_bills * 5) = 130 :=
by
  sorry

end shelly_total_money_l149_149229


namespace evaluate_f_neg_a_l149_149097

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem evaluate_f_neg_a (a : ℝ) (h : f a = 1 / 3) : f (-a) = 5 / 3 :=
by sorry

end evaluate_f_neg_a_l149_149097


namespace other_candidate_votes_l149_149697

-- Define the conditions as Lean terms
def total_votes := 5500
def invalid_percentage := 0.20
def one_candidate_percentage := 0.55

-- Step 1: Compute invalid votes
def invalid_votes := invalid_percentage * total_votes

-- Step 2: Compute valid votes
def valid_votes := total_votes - invalid_votes

-- Step 3: Compute votes for one candidate
def votes_one_candidate := one_candidate_percentage * valid_votes

-- Step 4: Compute votes for the other candidate
def votes_other_candidate := valid_votes - votes_one_candidate

-- The theorem to prove
theorem other_candidate_votes : votes_other_candidate = 1980 := by
  sorry

end other_candidate_votes_l149_149697


namespace parabola_conclusions_l149_149270

theorem parabola_conclusions (a c : ℝ) (h1 : a ≠ 0) (h2 : c > 0)
    (hA : 0 = a * (3 : ℝ)^2 - 2 * a * (3 : ℝ) + c) :
    let y := λ x : ℝ, a * x^2 - 2 * a * x + c,
        B := (-1 : ℝ),
        p := c - 3 * a
    in 
    (y B = 0) ∧ 
    (2 * a + c > 0) ∧
    (∀ t y1 y2, y (t + 2022 : ℝ) = y1 ∧ y (t + 2023 : ℝ) = y2 ∧ y1 > y2 → t > -2021) ∧
    (∀ m n p, (a * m^2 + 2 * a * m + c = p) ∧ (a * n^2 + 2 * a * n + c = p) ∧ p > 0 ∧ m < n → -3 < m ∧ n < 1) →
    3 conclusions are correct :=
by
  sorry

end parabola_conclusions_l149_149270


namespace books_in_either_collection_not_both_l149_149539

theorem books_in_either_collection_not_both (jessica_books : ℕ) (shared_books : ℕ) (tina_books_not_jessica : ℕ)
  (h_jessica : jessica_books = 22)
  (h_shared : shared_books = 12)
  (h_tina_not_jessica : tina_books_not_jessica = 10) :
  (jessica_books - shared_books) + tina_books_not_jessica = 20 :=
by {
  have jessica_unique : ℕ := jessica_books - shared_books,
  have h_jessica_unique : jessica_unique = 10, by { rw [h_jessica, h_shared], norm_num, },
  rw h_jessica_unique,
  exact h_tina_not_jessica,
  sorry
}

end books_in_either_collection_not_both_l149_149539


namespace intersection_set_l149_149943

open Set

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_set: M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_set_l149_149943


namespace necessary_but_not_sufficient_condition_l149_149326
-- Import the required Mathlib library in Lean 4

-- State the equivalent proof problem
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (|a| ≤ 1 → a ≤ 1) ∧ ¬ (a ≤ 1 → |a| ≤ 1) :=
by
  sorry

end necessary_but_not_sufficient_condition_l149_149326


namespace decorations_cost_l149_149575

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end decorations_cost_l149_149575


namespace bouquet_combinations_l149_149345

theorem bouquet_combinations : 
  ∃ n, n = 6 ∧ ∃ (r c : ℕ), 4 * r + 3 * c = 60 :=
begin
  -- Definitions and conditions
  sorry
end

end bouquet_combinations_l149_149345


namespace max_sum_base_seven_digits_l149_149291

theorem max_sum_base_seven_digits (n : ℕ) (h : n < 3000) : 
  ∃ d : ℕ, (digits 7 n).sum = d ∧ d = 24 :=
by {
  sorry
}

end max_sum_base_seven_digits_l149_149291


namespace average_of_possible_values_l149_149866

theorem average_of_possible_values (x : ℝ) (h : sqrt (3 * x^2 + 4) = sqrt 28) : 
  (x = 2 * sqrt 2 ∨ x = -2 * sqrt 2) → 
  ( ∃ (x1 x2 : ℝ), x1 = 2 * sqrt 2 ∧ x2 = -2 * sqrt 2 ∧ (x1 + x2) / 2 = 0 ) :=
by sorry

end average_of_possible_values_l149_149866


namespace conditional_probability_of_light_bulb_lasting_l149_149998

open ProbabilityTheory

theorem conditional_probability_of_light_bulb_lasting
  (P_3000 : ℝ) (P_4500 : ℝ) (h1 : P_3000 = 0.8) (h2 : P_4500 = 0.2) :
  (P_4500 / P_3000) = 0.25 :=
by
  simp [h1, h2]
  norm_num
  sorry

end conditional_probability_of_light_bulb_lasting_l149_149998


namespace intersection_A_B_l149_149186

-- Define sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | ∃ y ∈ A, |y| = x}

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_A_B :
  A ∩ B = {0, 2} :=
by
  sorry

end intersection_A_B_l149_149186


namespace zero_function_l149_149703

open Complex

noncomputable def is_regular_ngon (n : ℕ) (z : Fin n → ℂ) : Prop :=
  ∀ (k : Fin n), z k = z 0 * exp (2 * Real.pi * I * k / n)

noncomputable def satisfies_condition (n : ℕ) (f : ℂ → ℂ) :=
  ∀ (z : Fin n → ℂ), is_regular_ngon n z → (Finset.univ.sum (λ k, f (z k)) = 0)

theorem zero_function (n : ℕ) (h_n : n ≥ 3) (f : ℂ → ℂ)
  (h_f : satisfies_condition n f) :
  ∀ z : ℂ, f z = 0 :=
sorry

end zero_function_l149_149703


namespace tank_A_is_60_percent_of_tank_B_capacity_l149_149239

-- Conditions
def height_A : ℝ := 10
def circumference_A : ℝ := 6
def height_B : ℝ := 6
def circumference_B : ℝ := 10

-- Statement
theorem tank_A_is_60_percent_of_tank_B_capacity (V_A V_B : ℝ) (radius_A radius_B : ℝ)
  (hA : radius_A = circumference_A / (2 * Real.pi))
  (hB : radius_B = circumference_B / (2 * Real.pi))
  (vol_A : V_A = Real.pi * radius_A^2 * height_A)
  (vol_B : V_B = Real.pi * radius_B^2 * height_B) :
  (V_A / V_B) * 100 = 60 :=
by
  sorry

end tank_A_is_60_percent_of_tank_B_capacity_l149_149239


namespace intervals_monotonically_increasing_range_on_interval_l149_149096

def f (x: Real) := 2 * (sin x)^2 + sqrt 3 * sin (2 * x) + 1

theorem intervals_monotonically_increasing (k: Int) :
    ∀ x, x ∈ set.Icc (-π/6 + k * π) (π/3 + k * π) → 
        ∀ y, y ∈ set.Icc x (π/3 + k * π) → f(y) ≥ f(x) :=
sorry

theorem range_on_interval :
    ∀ x, x ∈ set.Icc 0 (π/2) → f(x) ∈ set.Icc 1 4 :=
sorry

end intervals_monotonically_increasing_range_on_interval_l149_149096


namespace length_real_axis_hyperbola_l149_149084

theorem length_real_axis_hyperbola :
  (∃ (C : ℝ → ℝ → Prop) (a b : ℝ), (a > 0) ∧ (b > 0) ∧ 
    (∀ x y : ℝ, C x y = ((x ^ 2) / a ^ 2 - (y ^ 2) / b ^ 2 = 1)) ∧
      (∀ x y : ℝ, ((x ^ 2) / 9 - (y ^ 2) / 16 = 1) → ((x ^ 2) / a ^ 2 - (y ^ 2) / b ^ 2 = 1)) ∧
      C (-3) (2 * Real.sqrt 3)) →
  2 * (3 / 2) = 3 :=
by {
  sorry
}

end length_real_axis_hyperbola_l149_149084


namespace symmetric_graphs_a_eq_pi_24_l149_149263

theorem symmetric_graphs_a_eq_pi_24 :
    ∃ a : ℝ, (∀ x : ℝ, sin (2*x - π/3) = cos (2*x + 2*π/3)) ↔ a = π/24 := 
sorry

end symmetric_graphs_a_eq_pi_24_l149_149263


namespace min_attendees_conference_l149_149662

-- Definitions based on the problem conditions
def writers : Nat := 45
def min_editors : Nat := 39
def max_x : Nat := 6
def both_writers_and_editors (x : Nat) : Nat := x
def neither_writers_nor_editors (x : Nat) : Nat := 2 * x

-- Proof statement that the minimum number of people attending the conference is 90
theorem min_attendees_conference (x : Nat) (h1 : x ≤ max_x) (h2 : min_editors > 38) :
  let total_attendees := writers + min_editors - both_writers_and_editors x + neither_writers_nor_editors x in
  total_attendees ≥ 90 :=
by
  sorry

end min_attendees_conference_l149_149662


namespace odd_coeffs_sum_geq_first_l149_149865

-- Definitions based strictly on conditions:
def w (P : Polynomial ℤ) : ℕ := P.coeffs.count (λ c, Int.Mod2 c = 1)

def Q (i : ℕ) : Polynomial ℤ := Polynomial.CoeffBinomial 1 i

-- The theorem statement based on equivalent proof problem:
theorem odd_coeffs_sum_geq_first (seq : List ℕ) (h_seq : seq.Sorted (· ≤ ·)) :
    w (seq.map Q.sum) ≥ w (Q seq.head) :=
sorry

end odd_coeffs_sum_geq_first_l149_149865


namespace roots_sixth_power_sum_l149_149933

variable {r s : ℝ}

theorem roots_sixth_power_sum :
  (r ^ 6 + s ^ 6) = 970 :=
by
  assume (h1 : r ^ 2 - 2 * r * sqrt 3 + 1 = 0)
  assume (h2 : s ^ 2 - 2 * s * sqrt 3 + 1 = 0)
  have h3 : r + s = 2 * sqrt 3 := sorry
  have h4 : r * s = 1 := sorry
  have h5 : r ^ 2 + s ^ 2 = 10 := sorry
  have h6 : r ^ 4 + s ^ 4 = 80 := sorry
  calc
    r ^ 6 + s ^ 6 = (r ^ 2 + s ^ 2) ^ 3 - 3 * r^2 * s^2 * (r^2 + s^2) : by sorry
               ... = 10 ^ 3 - 3 * 1 * 10 : by sorry
               ... = 970 : by sorry

end roots_sixth_power_sum_l149_149933


namespace number_of_solutions_l149_149016

theorem number_of_solutions : 
  ∃! (ns : Finset ℕ), (ns = { n | (1 ≤ n) ∧ (n ≤ 100) ∧ (n % 2 = 0) }) ∧  ns.card = 25 := by
sorry

end number_of_solutions_l149_149016


namespace distance_between_vertices_l149_149921

theorem distance_between_vertices : 
  let C := (2, 1)
  let D := (-3, 4)
  let distance := Real.sqrt ((2 - (-3))^2 + (1 - 4)^2)
  (C.vertex == (2, 1)) ∧ (D.vertex == (-3, 4)) ∧ (distance == Real.sqrt 34) := sorry

end distance_between_vertices_l149_149921


namespace find_line_eq_intersecting_l149_149780

theorem find_line_eq_intersecting (a b : ℝ) :
  (∀ x : ℝ, x^4 + 4 * x^3 - 26 * x^2 = a * x + b → 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x^4 + 4 * x^3 - 26 * x^2 = a * x + b) ∧ (x1 + x2 = -2) ∧ (x1 * x2 = -15))
  → (a = 60 ∧ b = -225) :=
begin
  sorry
end

end find_line_eq_intersecting_l149_149780


namespace coefficient_of_middle_term_l149_149248

theorem coefficient_of_middle_term : 
  (∀ x : ℝ, let expansion := (2*x + 1)^8 in
            (let terms := 9 in
            let middle_term := 5 in
            middle_term > 0 ∧ middle_term ≤ terms →
            let coeff := Nat.choose 8 4 * (2:ℝ)^4 in
            coeff = 1120)) :=
by sorry

end coefficient_of_middle_term_l149_149248


namespace binomial_coefficient_sum_l149_149882

noncomputable def term_expansion_x (n : ℕ) (r : ℕ) : ℤ := 
  nat.choose n r * (-1)^r * x^(2 * n - 3 * r)

theorem binomial_coefficient_sum :
  let n := 8 in
  (∀ x ∈ {r : ℕ | r + 1 = 6}, term_expansion_x n r = x) →
  let expansion := (1 - 3 * x)^n in
  let coeffs := tuple.map coeff expansion in
  let sum_terms := list.map abs (coeffs.tail.tail) in
  list.sum sum_terms = 2^16 - 1 :=
begin
  sorry
end

end binomial_coefficient_sum_l149_149882


namespace find_c_d_of_cubic_common_roots_l149_149793

theorem find_c_d_of_cubic_common_roots 
  (c d : ℝ)
  (h1 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + c * r ^ 2 + 12 * r + 7 = 0) ∧ (s ^ 3 + c * s ^ 2 + 12 * s + 7 = 0))
  (h2 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + d * r ^ 2 + 15 * r + 9 = 0) ∧ (s ^ 3 + d * s ^ 2 + 15 * s + 9 = 0)) :
  c = 5 ∧ d = 4 :=
sorry

end find_c_d_of_cubic_common_roots_l149_149793


namespace total_distance_of_the_race_l149_149150

-- Define the given conditions
def A_beats_B_by_56_meters_or_7_seconds : Prop :=
  ∃ D : ℕ, ∀ S_B S_A : ℕ, S_B = 8 ∧ S_A = D / 8 ∧ D = S_B * (8 + 7)

-- Define the question and correct answer
theorem total_distance_of_the_race : A_beats_B_by_56_meters_or_7_seconds → ∃ D : ℕ, D = 120 :=
by
  sorry

end total_distance_of_the_race_l149_149150


namespace fractional_exponent_representation_of_sqrt_l149_149261

theorem fractional_exponent_representation_of_sqrt (a : ℝ) : 
  Real.sqrt (a * 3 * a * Real.sqrt a) = a ^ (3 / 4) := 
sorry

end fractional_exponent_representation_of_sqrt_l149_149261


namespace no_x0_leq_zero_implies_m_gt_1_l149_149128

theorem no_x0_leq_zero_implies_m_gt_1 (m : ℝ) :
  (¬ ∃ x0 : ℝ, x0^2 + 2 * x0 + m ≤ 0) ↔ m > 1 :=
sorry

end no_x0_leq_zero_implies_m_gt_1_l149_149128


namespace min_keystrokes_1_to_250_l149_149290

theorem min_keystrokes_1_to_250 : 
  (∃ (f : ℕ → ℕ), f 1 = 250 ∧ (∀ n, f (n + 1) = f n + 1 ∨ f (n + 1) = f n * 2) ∧ (∀ g : ℕ → ℕ, g 1 = 250 ∧ (∀ n, g (n + 1) = g n + 1 ∨ g (n + 1) = g n * 2) → (∃ m ≤ 12, g 1 = g (12) + .... + g (m)))):
  ∃ (f : ℕ → ℕ), f 1 = 250 ∧ (∀ n, f (n + 1) = f n + 1 ∨ f (n + 1) = f n * 2) :=
by
  sorry

end min_keystrokes_1_to_250_l149_149290


namespace solve_for_s_l149_149398

theorem solve_for_s (s : ℤ) : 9 = 3^(2*s + 4) → s = -1 :=
by
  intro h
  -- our proof will go here: use the fact that both sides are powers of 3

  sorry -- proof omitted

end solve_for_s_l149_149398


namespace value_of_f_neg_2009_l149_149430

def f (a b x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem value_of_f_neg_2009 (a b : ℝ) (h : f a b 2009 = 10) :
  f a b (-2009) = -14 :=
by 
  sorry

end value_of_f_neg_2009_l149_149430


namespace trapezium_polygon_side_counts_l149_149625

theorem trapezium_polygon_side_counts :
  ∃ N, N ∈ {3, 4, 5, 6, 7, 8} ∧
      (∃ T1 T2 T3: Trapezium, 
         T1.edges.card = 6 ∧ T2.edges.card = 6 ∧ T3.edges.card = 6 ∧
         (combined_edges T1 T2 T3).poly_with_N_sides N) :=
sorry

-- Definitions (to be defined appropriately)
structure Trapezium := (edges : finset (fin 6))

-- Helper functions (to be defined appropriately)
def combined_edges (T1 T2 T3 : Trapezium) : finset (fin 18) :=
sorry

def finset.poly_with_N_sides (edges: finset (fin 18)) (N: ℕ) : Prop :=
sorry

end trapezium_polygon_side_counts_l149_149625


namespace right_triangle_exists_l149_149031

-- Define the setup: equilateral triangle ABC, point P, and angle condition
def Point (α : Type*) := α 
def inside {α : Type*} (p : Point α) (A B C : Point α) : Prop := sorry
def angle_at {α : Type*} (p q r : Point α) (θ : ℝ) : Prop := sorry
noncomputable def PA {α : Type*} (P A : Point α) : ℝ := sorry
noncomputable def PB {α : Type*} (P B : Point α) : ℝ := sorry
noncomputable def PC {α : Type*} (P C : Point α) : ℝ := sorry

-- Theorem we need to prove
theorem right_triangle_exists {α : Type*} 
  (A B C P : Point α)
  (h1 : inside P A B C)
  (h2 : angle_at P A B 150) :
  ∃ (Q : Point α), angle_at P Q B 90 :=
sorry

end right_triangle_exists_l149_149031


namespace cubic_polynomials_common_roots_c_d_l149_149790

theorem cubic_polynomials_common_roots_c_d (c d : ℝ) :
  (∀ (r s : ℝ), r ≠ s ∧
     (r^3 + c*r^2 + 12*r + 7 = 0) ∧ (s^3 + c*s^2 + 12*s + 7 = 0) ∧
     (r^3 + d*r^2 + 15*r + 9 = 0) ∧ (s^3 + d*s^2 + 15*s + 9 = 0)) →
  (c = -5 ∧ d = -6) := 
by
  sorry

end cubic_polynomials_common_roots_c_d_l149_149790


namespace pi_is_only_irrational_l149_149304

-- Definitions of the options
def pi_number : Real := Real.pi
def num_b : Real := 1.010010001
def num_c : Rational := ⟨3, 5⟩
def num_d : Real := Real.sqrt 9

-- Conditions of the problem
axiom pi_irrational : ¬ Rational pi_number
axiom b_rational : Rational num_b
axiom c_rational : Rational num_c
axiom d_rational : Rational num_d

-- The statement to prove
theorem pi_is_only_irrational :
  (¬ Rational pi_number) ∧ Rational num_b ∧ Rational num_c ∧ Rational num_d := by
  exact ⟨pi_irrational, b_rational, c_rational, d_rational⟩

end pi_is_only_irrational_l149_149304


namespace arithmetic_sequence_a7_value_l149_149520

variable (a : ℕ → ℝ) (a1 a13 a7 : ℝ)

theorem arithmetic_sequence_a7_value
  (h1 : a 1 = a1)
  (h13 : a 13 = a13)
  (h_sum : a1 + a13 = 12)
  (h_arith : 2 * a7 = a1 + a13) :
  a7 = 6 :=
by
  sorry

end arithmetic_sequence_a7_value_l149_149520


namespace total_birds_on_fence_l149_149715

theorem total_birds_on_fence (initial_birds additional_birds storks : ℕ) 
  (h1 : initial_birds = 6) 
  (h2 : additional_birds = 4) 
  (h3 : storks = 8) :
  initial_birds + additional_birds + storks = 18 :=
by
  sorry

end total_birds_on_fence_l149_149715


namespace conjugate_of_given_complex_number_l149_149250

noncomputable def given_complex_number : ℂ := 5 / (2 - I)

theorem conjugate_of_given_complex_number : complex.conj given_complex_number = 2 - I := by
  sorry

end conjugate_of_given_complex_number_l149_149250


namespace triangle_area_in_grid_l149_149804

theorem triangle_area_in_grid (P : Finset (Fin 5 × Fin 5)) 
  (hP_card : P.card = 6) 
  (hP_no_collinear : ∀ (x y z : (Fin 5 × Fin 5)), x ∈ P → y ∈ P → z ∈ P → 
    ¬(x.1 = y.1 = z.1 ∨ x.2 = y.2 = z.2 ∨ (x.1 - y.1) * (y.2 - z.2) = (x.2 - y.2) * (y.1 - z.1))) :
  ∃ (a b c : (Fin 5 × Fin 5)), a ∈ P ∧ b ∈ P ∧ c ∈ P ∧ 
    (abs ((a.1 * b.2 + b.1 * c.2 + c.1 * a.2 - a.2 * b.1 - b.2 * c.1 - c.2 * a.1) / 2) ≤ 2) :=
sorry

end triangle_area_in_grid_l149_149804


namespace new_block_weight_correct_l149_149743

-- Definitions: 
-- original weight, original dimensions, dimension increase factor.
def original_weight : ℝ := 4
def original_length : ℝ := 2
def original_width : ℝ := 3
def original_height : ℝ := 4
def increase_factor : ℝ := 1.5

-- Theorem statement: prove the weight of the new block
theorem new_block_weight_correct :
  let original_volume := original_length * original_width * original_height in
  let new_length := increase_factor * original_length in
  let new_width := increase_factor * original_width in
  let new_height := increase_factor * original_height in
  let new_volume := new_length * new_width * new_height in
  let volume_ratio := new_volume / original_volume in
  let new_weight := volume_ratio * original_weight in
  new_weight = 13.5 :=
by
  sorry

end new_block_weight_correct_l149_149743


namespace geometric_series_sum_is_correct_l149_149681

theorem geometric_series_sum_is_correct :
  ∑ i in finset.range 15, (4:ℚ)^i / (3:ℚ)^i = 4233171668 / 14348907 := by
  sorry

end geometric_series_sum_is_correct_l149_149681


namespace p_sufficient_not_necessary_q_l149_149047

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l149_149047


namespace sixty_th_term_arithmetic_seq_l149_149259

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 : ℝ)
variable (a13 : ℝ)
variable (a60 : ℝ)

def arithmetic_seq (a1 a13 a60 : ℝ) (d : ℝ) : Prop :=
  a 1 = a1 ∧ a 13 = a13 ∧ a60 = a1 + 59 * d

theorem sixty_th_term_arithmetic_seq (h1 : arithmetic_seq 6 32 (803 / 6) (13 / 6)) :
  a 60 = 803 / 6 := by
  sorry

end sixty_th_term_arithmetic_seq_l149_149259


namespace hybrids_with_full_headlights_l149_149139

-- Definitions for the conditions
def total_cars : ℕ := 600
def percentage_hybrids : ℕ := 60
def percentage_one_headlight : ℕ := 40

-- The proof statement
theorem hybrids_with_full_headlights :
  (percentage_hybrids * total_cars) / 100 - (percentage_one_headlight * (percentage_hybrids * total_cars) / 100) / 100 = 216 :=
by
  sorry

end hybrids_with_full_headlights_l149_149139


namespace matrix_det_eq_132_l149_149392

theorem matrix_det_eq_132 : 
  let M := !![ [ 3, 0, 2 ], [ 8, 5, -1 ], [ 3, 3, 7 ] ] 
  Matrix.det M = 132 := 
by
  sorry

end matrix_det_eq_132_l149_149392


namespace income_calculation_l149_149154

def income_from_investment (R : ℝ) (MV : ℝ) (I : ℝ) : ℝ := 
  let N := I / MV
  let FV := N * 100
  (R / 100) * FV

theorem income_calculation
  (R : ℝ) (MV : ℝ) (I : ℝ) (Inc : ℝ)
  (hR : R = 10)
  (hMV : MV = 96)
  (hI : I = 6240)
  : income_from_investment R MV I = 650 :=
by
  sorry

end income_calculation_l149_149154


namespace length_CD_eq_16_1333_l149_149640

-- Define the conditions of the problem
def radius : ℝ := 5
def volume : ℝ := 570 * Real.pi

-- Define the volume calculation for the given region
def total_volume (L : ℝ) : ℝ :=
  let V_cylinder := 25 * Real.pi * L
  let V_hemisphere := 2 * (Real.pi * radius^3 * 2 / 3) / 2
  V_cylinder + V_hemisphere

-- Statement of the problem to prove the correct length of CD
theorem length_CD_eq_16_1333 :
  ∃ (L : ℝ), total_volume(L) = volume ∧ Real.toRational L = 16.1333 :=
begin
  sorry
end

end length_CD_eq_16_1333_l149_149640


namespace steve_take_home_pay_l149_149610

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem steve_take_home_pay : 
  (annual_salary - (annual_salary * tax_rate + annual_salary * healthcare_rate + union_dues)) = 27200 := 
by 
  sorry

end steve_take_home_pay_l149_149610


namespace area_of_square_l149_149159

theorem area_of_square (A B C D P Q R : Point) 
(h1 : square A B C D) 
(h2 : OnLine P (line A D)) 
(h3 : OnLine Q (line A B)) 
(h4 : R = intersection (line B P) (line C Q)) 
(h5 : distance B R = 8) 
(h6 : distance P R = 9) 
(h7 : right_triangle B R P (angle BRP)) :
area_of_square A B C D = 225 := 
sorry

end area_of_square_l149_149159


namespace cucumber_weight_after_evaporation_l149_149618

theorem cucumber_weight_after_evaporation 
    (initial_weight : ℝ) 
    (initial_average_water_content : ℝ) 
    (evaporation_water_content : ℝ)
    (initial_weight = 100)
    (initial_average_water_content = 0.95)
    (evaporation_water_content = 0.75) :
    let dry_matter := initial_weight * (1 - initial_average_water_content)
    let new_weight := dry_matter / (1 - evaporation_water_content)
    new_weight = 500 :=
by
  sorry

end cucumber_weight_after_evaporation_l149_149618


namespace fraction_simplest_form_l149_149637

theorem fraction_simplest_form (a b : ℕ) (h : 0.63575 = a / b) (h_gcd : Nat.gcd a b = 1) : a + b = 6543 :=
sorry

end fraction_simplest_form_l149_149637


namespace combined_mpg_and_total_cost_l149_149962

theorem combined_mpg_and_total_cost:
  ∀ (ray_mpg tom_mpg miles driven : ℕ) (gas_cost : ℕ),
  ray_mpg = 30 → tom_mpg = 10 → miles = 120 → gas_cost = 4 →
  (∃ combined_mpg total_cost, combined_mpg = 15 ∧ total_cost = 64) :=
begin
  sorry
end

end combined_mpg_and_total_cost_l149_149962


namespace additional_water_added_l149_149891

variable (M W : ℕ)

theorem additional_water_added (M W : ℕ) (initial_mix : ℕ) (initial_ratio : ℕ × ℕ) (new_ratio : ℚ) :
  initial_mix = 45 →
  initial_ratio = (4, 1) →
  new_ratio = 4 / 3 →
  (4 / 5) * initial_mix = M →
  (1 / 5) * initial_mix + W = 3 / 4 * M →
  W = 18 :=
by
  sorry

end additional_water_added_l149_149891


namespace repeating_decimal_fractions_l149_149937

theorem repeating_decimal_fractions {a b c : ℕ} (habc : 1 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 998)
  (not_all_zero : ¬(a = 0 ∧ b = 0 ∧ c = 0))
  (not_all_nine : ¬(a = 9 ∧ b = 9 ∧ c = 9)) :
  ∃ n, (0 < n ∧ n < 999) ∧ (∑ i in finset.range 999, if n.gcd 999 = 1 then 1 else 0) = 660 :=
sorry

end repeating_decimal_fractions_l149_149937


namespace range_of_a_l149_149076

theorem range_of_a (x a : ℝ) (hp : x^2 + 2 * x - 3 > 0) (hq : x > a)
  (h_suff : x^2 + 2 * x - 3 > 0 → ¬ (x > a)):
  a ≥ 1 := 
by
  sorry

end range_of_a_l149_149076


namespace total_pairs_is_11_l149_149694

-- Definitions for the conditions
def soft_lens_price : ℕ := 150
def hard_lens_price : ℕ := 85
def total_sales_last_week : ℕ := 1455

-- Variables
variables (H S : ℕ)

-- Condition that she sold 5 more pairs of soft lenses than hard lenses
def sold_more_soft : Prop := S = H + 5

-- Equation for total sales
def total_sales_eq : Prop := (hard_lens_price * H) + (soft_lens_price * S) = total_sales_last_week

-- Total number of pairs of contact lenses sold
def total_pairs_sold : ℕ := H + S

-- The theorem to prove
theorem total_pairs_is_11 (H S : ℕ) (h1 : sold_more_soft H S) (h2 : total_sales_eq H S) : total_pairs_sold H S = 11 :=
sorry

end total_pairs_is_11_l149_149694


namespace principal_age_correct_l149_149643

def principal_age (anya borya vova galya dima sasha: Prop) : ℕ :=
  if anya ∧ ¬borya ∧ vova ∧ ¬galya ∧ ¬dima ∧ ¬sasha then 39 else 0

theorem principal_age_correct (anya borya vova galya dima sasha : Prop)
  (Hanya : ∀ age, (anya ↔ age > 38))
  (Hborya : ∀ age, (borya ↔ age < 35))
  (Hvova : ∀ age, (vova ↔ age < 40))
  (Hgalya : ∀ age, (galya ↔ age > 40))
  (Hdima : ∀ borya vova, dima ↔ borya ∧ vova)
  (Hsasha : sasha ↔ ¬anya ∧ ¬borya ∧ ¬vova ∧ ¬galya ∧ ¬dima)
  (Hparity : true): 
  principal_age anya borya vova galya dima sasha = 39 :=
  sorry

end principal_age_correct_l149_149643


namespace triangle_AEC_area_l149_149246

theorem triangle_AEC_area (ABC_area : ℝ) (AD DB : ℝ) (triangle_ABC_area : 40 = ABC_area)
  (AD_length : 3 = AD) (DB_length : 5 = DB)
  (equal_areas : ∀ E F, 2 * (∃ BE BF : ℝ, True ∧ 
    ∃ A B E F, True ∧ 
    ∀ DE_par_AF, True ∧ 
    ((BE * A) = DB * E) ∧ (BE * F = ABC_area)) = 3 * A) :
  ∃ AEC_area : ℝ, AEC_area = 15 :=
by sorry

end triangle_AEC_area_l149_149246


namespace first_plot_germination_rate_l149_149422

-- Define the known quantities and conditions
def plot1_seeds : ℕ := 300
def plot2_seeds : ℕ := 200
def plot2_germination_rate : ℚ := 35 / 100
def total_germination_percentage : ℚ := 26 / 100

-- Define a statement to prove the percentage of seeds that germinated in the first plot
theorem first_plot_germination_rate : 
  ∃ (x : ℚ), (x / 100) * plot1_seeds + (plot2_germination_rate * plot2_seeds) = total_germination_percentage * (plot1_seeds + plot2_seeds) ∧ x = 20 :=
by
  sorry

end first_plot_germination_rate_l149_149422


namespace area_of_section_l149_149257

variable (a : ℝ) -- Edge length of the regular tetrahedron

-- Definitions:
def regular_tetrahedron (a : ℝ) := True -- Placeholder for the actual structure of a regular tetrahedron

def plane_through_A_parallel_BC (a : ℝ) := True -- Placeholder for the plane condition

def angle_between_AB_and_plane_30 (a : ℝ) := True -- Placeholder for the angle condition

theorem area_of_section (a : ℝ) 
  (h1 : regular_tetrahedron a) 
  (h2 : plane_through_A_parallel_BC a) 
  (h3 : angle_between_AB_and_plane_30 a) :
  let area := (3 * a^2 * Real.sqrt 2) / 25 in
  area = (3 * a^2 * Real.sqrt 2) / 25 := 
by 
  sorry

end area_of_section_l149_149257


namespace proof_C_I_M_cap_N_l149_149566

open Set

variable {𝕜 : Type _} [LinearOrderedField 𝕜]

def I : Set 𝕜 := Set.univ
def M : Set 𝕜 := {x : 𝕜 | -2 ≤ x ∧ x ≤ 2}
def N : Set 𝕜 := {x : 𝕜 | x < 1}
def C_I (A : Set 𝕜) : Set 𝕜 := I \ A

theorem proof_C_I_M_cap_N :
  C_I M ∩ N = {x : 𝕜 | x < -2} := by
  sorry

end proof_C_I_M_cap_N_l149_149566


namespace max_sum_of_colored_board_l149_149946

/-
 A 9x9 board where two squares are friends if they share a side or if they are at opposite ends of the same row or column.
 Each square is painted one of three colors: green, blue or red.
 Numbers written based on the color of friends:
 - Green square: number of red friends + 2 * number of blue friends
 - Red square: number of blue friends + 2 * number of green friends
 - Blue square: number of green friends + 2 * number of red friends

 Prove that the maximum possible sum of written numbers in all squares 
 given Leticia can choose the coloring is 486.
-/

noncomputable def max_sum_colored_board (n : ℕ := 9) : ℕ :=
let num_squares := n * n,
    num_borders := (num_squares * 4) / 2
in num_borders * 3

theorem max_sum_of_colored_board : max_sum_colored_board = 486 := by
  sorry

end max_sum_of_colored_board_l149_149946


namespace number_of_brave_children_l149_149583

-- Define the problem conditions and the goal
def brave_children_count : ℕ := 10

theorem number_of_brave_children :
  ∀ (initial_pairing : ℕ) (total_children : ℕ) (final_pairing : ℕ),
    initial_pairing = 1 →
    total_children = 20 →
    final_pairing = 21 →
    (final_pairing - initial_pairing) / 2 = brave_children_count :=
begin
  intros initial_pairing total_children final_pairing h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end number_of_brave_children_l149_149583


namespace probability_neither_perfect_square_cube_sixth_power_l149_149993

theorem probability_neither_perfect_square_cube_sixth_power : 
  (∃ (x : ℚ), x = 183 / 200) := 
by
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 2
  let overlap := 2 -- adjustments for overlaps which were counted twice
  let total_special_numbers := perfect_squares + perfect_cubes - overlap
  let non_special_numbers := total_numbers - total_special_numbers
  let probability := non_special_numbers / total_numbers
  have h : probability = 183 / 200 := by
    sorry
  existsi (183 / 200)
  exact h

end probability_neither_perfect_square_cube_sixth_power_l149_149993


namespace prove_main_proposition_true_l149_149960

-- Define propositions p and q according to the problem description
def proposition_p : Prop := ∀ x : ℝ, ∃ (y : ℝ), y = tan x ∧ monotone_increasing (λ x, tan x)

def proposition_q : Prop := ∀ (A B C : ∇), 
  (angle A > angle B ↔ sin A > sin B)

-- Define the overall proposition to determine truth value of p ∨ q
def main_proposition : Prop := proposition_p ∨ proposition_q

-- Given conditions
axiom proposition_p_false : ¬ proposition_p
axiom proposition_q_true : proposition_q

-- The ultimate proposition to prove
theorem prove_main_proposition_true : main_proposition := 
by {
  left, -- sorry is added to skip the proof as instructed!
  exact proposition_q_true,
}

end prove_main_proposition_true_l149_149960


namespace solve_equation_l149_149002

noncomputable def equation (x : ℝ) : Prop :=
    real.cbrt (18 * x - 2) + real.cbrt (16 * x + 2) = 5 * real.cbrt x

theorem solve_equation (x : ℝ) :
    equation x ↔ x = 0 ∨ x = 31 / 261 ∨ x = -35 / 261 :=
by
  intros
  sorry

end solve_equation_l149_149002


namespace Sum2016Equals1008_l149_149437

-- Defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Defining the sum of the first n terms of an arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

-- Given conditions
variables {a : ℕ → ℚ} {S : ℕ → ℚ}
hypothesis arithmetic_seq : arithmetic_sequence a
hypothesis sum_def : sum_of_first_n_terms a S
hypothesis collinearity_condition : a 4 + a 2013 = 1

-- The main theorem to prove
theorem Sum2016Equals1008 : S 2016 = 1008 :=
by sorry

end Sum2016Equals1008_l149_149437


namespace area_of_triangle_l149_149455

theorem area_of_triangle (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 144) : 
  1/2 * a * b = 30 :=
by sorry

end area_of_triangle_l149_149455


namespace fibonacci_sequence_x_l149_149525

noncomputable def x := 13

theorem fibonacci_sequence_x :
  ∀ a b n, (a = 8) ∧ (b = 21) → (b = a + x) → x = 13 :=
by
  intros a b n h1 h2
  cases h1
  rw h1_left at h1_right
  rw h1_right at h2
  exact h2.symm
  sorry -- Replace this with the actual proof when required

end fibonacci_sequence_x_l149_149525


namespace part_1_part_2_part_3_l149_149191

variable {R : Type*} [LinearOrderedField R]

noncomputable def f (x : R) : R := sorry

axiom f_eqn : ∀ x y : R, f(x + y) = f(x) * f(y)
axiom f_neg : ∀ x : R, x < 0 → f(x) > 1

theorem part_1 (x : R) (h1 : x > 0) : 0 < f(x) ∧ f(x) < 1 :=
by sorry

theorem part_2 : ∀ x1 x2 : R, x1 < x2 → f(x2) < f(x1) :=
by sorry

theorem part_3 (a : R) : (∀ x y : R, f(x^2) * f(y^2) ≤ f(a * x * y)) ↔ -2 ≤ a ∧ a ≤ 2 :=
by sorry

end part_1_part_2_part_3_l149_149191


namespace find_x_solution_l149_149420

theorem find_x_solution (x : ℚ) : (∀ y : ℚ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by
  sorry

end find_x_solution_l149_149420


namespace araceli_goal_with_4_crosses_araceli_goal_not_achievable_with_3_crosses_l149_149318

-- Define a board and cross structure
structure Board :=
  (size : ℕ)
  (matrix : fin size → fin size → bool)

def valid_cross_position (b : Board) (pos : fin b.size × fin b.size) : bool :=
  let (x, y) := pos in
  b.matrix x y    &&
  b.matrix (x - 1) y &&
  b.matrix x (y - 1) &&
  (x + 1 < b.size && b.matrix (x + 1) y) &&
  (y + 1 < b.size && b.matrix x (y + 1))

-- We specify the 8x8 board and how a cross covers it
noncomputable def initial_board : Board :=
{ size := 8,
  matrix := λ x y, true } -- Initially, all cells are empty and available

-- Define statements
theorem araceli_goal_with_4_crosses : 
  ∃ (board_4_crosses : Board), 
  (board_4_crosses.size = 8) ∧
  -- Placing 4 crosses is sufficient to block Florinda
  (valid_cross_position board_4_crosses (⟨2, sorry⟩, sorry) = false) ∧
  (valid_cross_position board_4_crosses (⟨4, sorry⟩, sorry) = false) ∧
  (valid_cross_position board_4_crosses (⟨6, sorry⟩, sorry) = false) ∧
  (valid_cross_position board_4_crosses (⟨8, sorry⟩, sorry) = false) := 
sorry

theorem araceli_goal_not_achievable_with_3_crosses : 
  ∀ (board_3_crosses : Board),
  (board_3_crosses.size = 8) →
  ¬(valid_cross_position board_3_crosses (⟨2, sorry⟩, sorry) = false ∨
   valid_cross_position board_3_crosses (⟨4, sorry⟩, sorry) = false ∨
   valid_cross_position board_3_crosses (⟨6, sorry⟩, sorry) = false ∨
   valid_cross_position board_3_crosses (⟨8, sorry⟩, sorry) = false):
sorry

end araceli_goal_with_4_crosses_araceli_goal_not_achievable_with_3_crosses_l149_149318


namespace minimized_sum_distance_coordinates_l149_149073

noncomputable def point : Type := ℝ × ℝ

def is_on_parabola (M : point) : Prop :=
  M.2 ^ 2 = 5 * M.1

def focus_of_parabola : point := (5 / 4, 0)

def distance (P Q : point) : ℝ :=
  Math.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def minimizes_sum_of_distances (A F M : point) : Prop :=
  ∀ M' : point, is_on_parabola M' → 
    distance M A + distance M F ≤ distance M' A + distance M' F

theorem minimized_sum_distance_coordinates :
  ∃ (M : point), is_on_parabola M ∧ minimizes_sum_of_distances (3, 1) focus_of_parabola M ∧ M = (1 / 5, 1) :=
sorry

end minimized_sum_distance_coordinates_l149_149073


namespace minimum_c_range_l149_149080

noncomputable def a_b_c_condition (a b c : ℝ) : Prop :=
  (∀ a b : ℝ, (0 < a) → (0 < b) → (1 / a + 2 / b = 2) → (a + b ≥ c))

theorem minimum_c_range :
  a_b_c_condition a b (3/2 + Real.sqrt 2) :=
begin
  sorry
end

end minimum_c_range_l149_149080


namespace triangle_area_l149_149749

noncomputable def s (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c : ℝ) : ℝ := Real.sqrt (s a b c * (s a b c - a) * (s a b c - b) * (s a b c - c))

theorem triangle_area (a b c : ℝ) (ha : a = 13) (hb : b = 12) (hc : c = 5) : area a b c = 30 := by
  rw [ha, hb, hc]
  show area 13 12 5 = 30
  -- manually calculate and reduce the expression to verify the theorem
  sorry

end triangle_area_l149_149749


namespace min_value_of_function_l149_149448

theorem min_value_of_function (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, y = x + 1 / (x - 4) ∧ (∀ z : ℝ, z = x + 1 / (x - 4) → z ≥ 6) :=
sorry

end min_value_of_function_l149_149448


namespace ratio_sum_odd_to_even_divisors_l149_149559

def M : ℕ := 35 * 36 * 65 * 280

def sum_of_odd_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ d, d % 2 = 1) (n.divisors)), d

def sum_of_all_divisors (n : ℕ) : ℕ :=
  ∑ d in n.divisors, d

def sum_of_even_divisors (n : ℕ) : ℕ :=
  sum_of_all_divisors n - sum_of_odd_divisors n

theorem ratio_sum_odd_to_even_divisors :
  sum_of_odd_divisors M = 1 / 62 * sum_of_even_divisors M :=
by sorry

end ratio_sum_odd_to_even_divisors_l149_149559


namespace number_of_solutions_l149_149015

theorem number_of_solutions : 
  ∃! (ns : Finset ℕ), (ns = { n | (1 ≤ n) ∧ (n ≤ 100) ∧ (n % 2 = 0) }) ∧  ns.card = 25 := by
sorry

end number_of_solutions_l149_149015


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l149_149233

theorem solve_eq1 :
  ∀ x : ℝ, 6 * x - 7 = 4 * x - 5 ↔ x = 1 := by
  intro x
  sorry

theorem solve_eq2 :
  ∀ x : ℝ, 5 * (x + 8) - 5 = 6 * (2 * x - 7) ↔ x = 11 := by
  intro x
  sorry

theorem solve_eq3 :
  ∀ x : ℝ, x - (x - 1) / 2 = 2 - (x + 2) / 5 ↔ x = 11 / 7 := by
  intro x
  sorry

theorem solve_eq4 :
  ∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8 := by
  intro x
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l149_149233


namespace hypotenuse_of_right_triangle_l149_149672

theorem hypotenuse_of_right_triangle (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end hypotenuse_of_right_triangle_l149_149672


namespace hybrids_with_full_headlights_l149_149138

-- Definitions for the conditions
def total_cars : ℕ := 600
def percentage_hybrids : ℕ := 60
def percentage_one_headlight : ℕ := 40

-- The proof statement
theorem hybrids_with_full_headlights :
  (percentage_hybrids * total_cars) / 100 - (percentage_one_headlight * (percentage_hybrids * total_cars) / 100) / 100 = 216 :=
by
  sorry

end hybrids_with_full_headlights_l149_149138


namespace continuity_at_9_l149_149592

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 4

theorem continuity_at_9 : ∀ (ε > 0), ∃ δ > 0, (∀ x, abs (x - 9) < δ → abs (f x - f 9) < ε) :=
by
  intros ε hε
  use ε / 76
  use div_pos hε (by norm_num)
  intros x hx
  dsimp [f]
  calc
    abs ((4 * x^2 + 4) - (4 * 9^2 + 4)) = abs (4 * x^2 - 324) : by ring
    ... = 4 * abs (x^2 - 81) : by rw [abs_mul, abs_of_nonneg (show (4:ℝ) ≥ 0, by norm_num)]
    ... < 4 * (ε / (4 * 19)) : by simp [abs_sub_lt_iff.mp hx]
    ... = 4 * ε / 76 : by field_simp [(by norm_num : (4:ℝ) * 19 ≠ 0)]
    ... = ε : by ring

end continuity_at_9_l149_149592


namespace triangle_PQS_PRS_area_ratio_l149_149908

noncomputable def triangle_area_ratio (P Q R S: Type)
  [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
  (PQ PR QR PQ_value PR_value QR_value : ℝ)
  (PS_angle_bisector : Prop) : Prop :=
  (PQ = 20) ∧ (PR = 30) ∧ (QR = 34) ∧ PS_angle_bisector →
  (area_ratio : ℝ) := sorry

theorem triangle_PQS_PRS_area_ratio (P Q R S: Type)
  [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
  (PQ PR QR PQ_value PR_value QR_value : ℝ)
  (PS_angle_bisector : Prop) (h : triangle_area_ratio P Q R S PQ PR QR PQ_value PR_value QR_value PS_angle_bisector) :
  (area_ratio = 2/3) := sorry

end triangle_PQS_PRS_area_ratio_l149_149908


namespace perpendicular_condition_l149_149478

theorem perpendicular_condition (a : ℝ) :
  let l1 (x y : ℝ) := x + a * y - 2
  let l2 (x y : ℝ) := x - a * y - 1
  (∀ x y, (l1 x y = 0 ↔ l2 x y ≠ 0) ↔ 1 - a * a = 0) →
  (a = -1) ∨ (a = 1) :=
by
  intro
  sorry

end perpendicular_condition_l149_149478


namespace k_range_for_ellipse_l149_149124

theorem k_range_for_ellipse (k : ℝ) :
  (∃ (x y : ℝ), x^2 + k * y^2 = 2) → (k ∈ (0, 1) ∪ (1, +∞)) :=
by
  sorry

end k_range_for_ellipse_l149_149124


namespace valid_years_count_l149_149803

def is_valid_year (yy : Nat) : Prop :=
  Nat.gcd 20 yy = 1

def count_valid_years (start end_ : Nat) : Nat :=
  (List.range (end_ - start + 1)).countp (λ n => is_valid_year (start + n))

theorem valid_years_count : count_valid_years 2026 2099 = 30 := by
  sorry

end valid_years_count_l149_149803


namespace remaining_homes_proof_l149_149544

-- Define the total number of homes
def total_homes : ℕ := 200

-- Distributed homes after the first hour
def homes_distributed_first_hour : ℕ := (2 * total_homes) / 5

-- Remaining homes after the first hour
def remaining_homes_first_hour : ℕ := total_homes - homes_distributed_first_hour

-- Distributed homes in the next 2 hours
def homes_distributed_next_two_hours : ℕ := (60 * remaining_homes_first_hour) / 100

-- Remaining homes after the next 2 hours
def homes_remaining : ℕ := remaining_homes_first_hour - homes_distributed_next_two_hours

theorem remaining_homes_proof : homes_remaining = 48 := by
  sorry

end remaining_homes_proof_l149_149544


namespace closest_point_on_line_l149_149396

theorem closest_point_on_line 
  (line_eq : ∀ x, y = (x + 7) / 3)
  (target_point : (ℝ × ℝ))
  (target_point = (8, 0))
  : ∃ p : ℝ × ℝ, (p.1 = 49 / 10 ∧ p.2 = 51 / 10) ∧ (p ∈ set_of (λ (x, y), y = (x + 7) / 3)) ∧ 
  ∀ q ∈ set_of (λ (x, y), y = (x + 7) / 3), dist p target_point ≤ dist q target_point := 
sorry

end closest_point_on_line_l149_149396


namespace total_length_l149_149352

def rubber_len : ℝ := sorry
def pen_len : ℝ := rubber_len + 3
def pencil_len : ℝ := 12
def ruler_len : ℝ := 3 * rubber_len
def marker_len : ℝ := (rubber_len + pen_len + pencil_len) / 3

theorem total_length : pen_len = pencil_len - 2 →
  marker_len = ruler_len / 2 →
  rubber_len + pen_len + pencil_len + marker_len + ruler_len = 60.5 :=
by
  intros h_pen h_marker
  sorry  -- Proof will be provided later in a complete proof.

end total_length_l149_149352


namespace adjacent_product_negative_l149_149524

-- Define the sequence
def a (n : ℕ) : ℤ := 2*n - 17

-- Define the claim about the product of adjacent terms being negative
theorem adjacent_product_negative : a 8 * a 9 < 0 :=
by sorry

end adjacent_product_negative_l149_149524


namespace incorrect_statement_c_l149_149131

-- Define the vectors
def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (0, 2)

-- Define the conditions as Lean definitions
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)
def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- The statement to prove
theorem incorrect_statement_c : 
  ¬ is_parallel a b ∧ 
  add_vectors a b = (2, 2) ∧ 
  magnitude a = magnitude b ∧ 
  is_perpendicular a b :=
by
  sorry

end incorrect_statement_c_l149_149131


namespace find_a_l149_149192

theorem find_a (a : ℝ) : (∀ x : ℝ, (0 <= x → (x, ax - log(x + 1)) = (0, 0) → (deriv (λ x, a * x - log(x + 1)) 0) = 2)) → a = 3 :=
by 
  sorry

end find_a_l149_149192


namespace ellipse_line_intersection_distance_l149_149461

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

theorem ellipse_line_intersection_distance :
  let ellipse := λ x y : ℝ, x ^ 2 + 4 * y ^ 2 = 16
  let lineAB := λ x y : ℝ, y + 1 = (1 / 2) * (x - 2)
  let A := (4, 0)
  let B := (0, -2)
  distance A B = 2 * real.sqrt 5 :=
by
  sorry

end ellipse_line_intersection_distance_l149_149461


namespace remainder_div_13_l149_149310

theorem remainder_div_13 {k : ℤ} (N : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 :=
by
  sorry

end remainder_div_13_l149_149310


namespace sum_eq_formula_l149_149760

theorem sum_eq_formula {a1 : ℝ} (n : ℕ) (hn : n > 0):
  let q := 2,
  let a := λ n, a1 * q ^ (n - 1),
  let b := λ n, a n,
  let S := λ n, (Finset.range n).sum (λ k, b k.succ)
  in S n = 2^n - 1 - 1 / ((n : ℝ) + 1) :=
begin
  intros,
  sorry,
end

end sum_eq_formula_l149_149760


namespace triangle_side_length_c_l149_149133

open Real

variables (A B C a b c : ℝ)
variables (angle_A angle_B angle_C : ℝ)

-- Given conditions
def triangle_area_eq : ℝ := 2 * sqrt 3
def sum_of_sides_eq : ℝ := 6
def trigonometric_condition : ℝ := (a * cos angle_B + b * cos angle_A) / c
def double_cosine_C : ℝ := 2 * cos angle_C

-- Proof statement
theorem triangle_side_length_c (h1 : 1 / 2 * a * b * sin angle_C = triangle_area_eq)
                                (h2 : a + b = sum_of_sides_eq)
                                (h3 : trigonometric_condition = double_cosine_C)
                                (h4 : angle_C = π / 3) : c = 2 * sqrt 3 :=
sorry

end triangle_side_length_c_l149_149133


namespace fraction_numerator_greater_than_denominator_l149_149994

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  (4 * x + 2 > 8 - 3 * x) ↔ (6 / 7 < x ∧ x ≤ 3) :=
by
  sorry

end fraction_numerator_greater_than_denominator_l149_149994


namespace sign_of_a_l149_149493

theorem sign_of_a (a b c d : ℝ) (h : b * (3 * d + 2) ≠ 0) (ineq : a / b < -c / (3 * d + 2)) : 
  (a = 0 ∨ a > 0 ∨ a < 0) :=
sorry

end sign_of_a_l149_149493


namespace parabola_line_no_intersection_l149_149925

def parabola (x : ℝ) : ℝ := x^2

def line (m x : ℝ) : ℝ := m * (x - 10) + 25

def has_no_real_solutions (m : ℝ) : Prop :=
  let a := 1
  let b := -m
  let c := 10 * m - 25
  quadratic_discriminant a b c < 0

theorem parabola_line_no_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, has_no_real_solutions m ↔ r < m ∧ m < s) ∧ r + s = 40 :=
sorry

end parabola_line_no_intersection_l149_149925


namespace range_of_y_l149_149121

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 132) : y ∈ Ioo (-12 : ℝ) (-11 : ℝ) :=
sorry

end range_of_y_l149_149121


namespace statement_A_statement_B_statement_C_statement_D_l149_149686

-- Statement A
def z1 := complex.I ^ 3
theorem statement_A : complex.conj z1 = complex.I := sorry

-- Statement B
def z2 := 1 + (1 / complex.I)
theorem statement_B : complex.imaginaryPart z2 ≠ -complex.I := sorry

-- Statement C
variables (a : ℝ)
def z3 := a + a * complex.I
theorem statement_C : ¬(z3.im = 0) ∨ ¬(z3.re = 0) := sorry

-- Statement D
variable (z : ℂ)
theorem statement_D (hz : (1 / z).im = 0) : z.im = 0 := sorry

end statement_A_statement_B_statement_C_statement_D_l149_149686


namespace find_f_inv_l149_149086

variables (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Defining the conditions in Lean 4
def has_inverse : Prop := function.bijective f ∧ (∀ x, f (g x) = x ∧ g (f x) = x)
def passes_through (P : ℝ × ℝ) : Prop := f P.1 = P.2

-- The main theorem statement
theorem find_f_inv :
  has_inverse f g ∧ passes_through f (2, -1) → g (-1) = 2 :=
by
  intro h,
  sorry

end find_f_inv_l149_149086


namespace sum_of_remaining_numbers_last_number_written_l149_149909

open Nat

-- Define the initial sequence as a list from 1 to 100
def initial_sequence := List.range' 1 100

-- Function to perform one step of the given operation: remove the first 6 numbers and append their sum
def step (seq: List ℕ) : List ℕ :=
  let (first_six, rest) := List.splitAt 6 seq
  List.append rest [first_six.sum]

-- Function to repeatedly apply the step until fewer than 6 numbers remain
def repeat_step (seq: List ℕ) : List ℕ :=
  if seq.length < 6 then seq
  else repeat_step (step seq)

-- Prove the sum of the remaining numbers is 5050
theorem sum_of_remaining_numbers :
  repeat_step initial_sequence.sum = 5050 :=
by
  sorry

-- Prove the last number written on the blackboard
theorem last_number_written :
  let final_sequence := repeat_step initial_sequence
  final_sequence.lastD 0 = 2394 :=
by
  sorry

end sum_of_remaining_numbers_last_number_written_l149_149909


namespace maria_total_earnings_l149_149948

noncomputable def total_earnings : ℕ := 
  let tulips_day1 := 30
  let roses_day1 := 20
  let lilies_day1 := 15
  let sunflowers_day1 := 10
  let tulips_day2 := tulips_day1 * 2
  let roses_day2 := roses_day1 * 2
  let lilies_day2 := lilies_day1
  let sunflowers_day2 := sunflowers_day1 * 3
  let tulips_day3 := tulips_day2 / 10
  let roses_day3 := 16
  let lilies_day3 := lilies_day1 / 2
  let sunflowers_day3 := sunflowers_day2
  let price_tulip := 2
  let price_rose := 3
  let price_lily := 4
  let price_sunflower := 5
  let day1_earnings := tulips_day1 * price_tulip + roses_day1 * price_rose + lilies_day1 * price_lily + sunflowers_day1 * price_sunflower
  let day2_earnings := tulips_day2 * price_tulip + roses_day2 * price_rose + lilies_day2 * price_lily + sunflowers_day2 * price_sunflower
  let day3_earnings := tulips_day3 * price_tulip + roses_day3 * price_rose + lilies_day3 * price_lily + sunflowers_day3 * price_sunflower
  day1_earnings + day2_earnings + day3_earnings

theorem maria_total_earnings : total_earnings = 920 := 
by 
  unfold total_earnings
  sorry

end maria_total_earnings_l149_149948
