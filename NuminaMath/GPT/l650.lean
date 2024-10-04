import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.EuclideanDomain
import Mathlib.Algebra.Field
import Mathlib.Algebra.GeomSeries
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.CriticalPoints
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Limit
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Graph
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Cast
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Numbers.Basic
import Mathlib.Order.Floor
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace volume_of_first_bottle_l650_650710

theorem volume_of_first_bottle (V_2 V_3 : ℕ) (V_total : ℕ):
  V_2 = 750 ∧ V_3 = 250 ∧ V_total = 3 * 1000 →
  (V_total - V_2 - V_3) / 1000 = 2 :=
by
  sorry

end volume_of_first_bottle_l650_650710


namespace range_of_a_l650_650275

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then log a (x + a - 1) else (2 * a - 1) * x - a

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l650_650275


namespace x2008_is_sum_of_two_squares_l650_650433

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else (sequence (n-1) * sequence (n-1) + 3) / sequence (n-2)

theorem x2008_is_sum_of_two_squares : ∃ a b : ℕ, a^2 + b^2 = sequence 2008 :=
sorry

end x2008_is_sum_of_two_squares_l650_650433


namespace triangle_inequality_l650_650460

variables {R : Type*} [linear_ordered_ring R] 

noncomputable def dist (P Q : R × R) : R := real.sqrt (((P.1 - Q.1) ^ 2) + ((P.2 - Q.2) ^ 2))

theorem triangle_inequality (A B C P : R × R)
  (a b c : R)
  (hA : dist B C = a)
  (hB : dist A C = b)
  (hC : dist A B = c)
  (G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  a * (dist P A) ^ 3 + b * (dist P B) ^ 3 + c * (dist P C) ^ 3 ≥ 3 * a * b * c * (dist P G) :=
sorry

end triangle_inequality_l650_650460


namespace divisors_squared_prime_l650_650124

theorem divisors_squared_prime (p : ℕ) [hp : Fact (Nat.Prime p)] (m : ℕ) (h : m = p^3) (hm_div : Nat.divisors m = 4) :
  Nat.divisors (m^2) = 7 :=
sorry

end divisors_squared_prime_l650_650124


namespace MaryNeedingToGrow_l650_650888

/-- Mary's height is 2/3 of her brother's height. --/
def MarysHeight (brothersHeight : ℕ) : ℕ := (2 * brothersHeight) / 3

/-- Mary needs to grow a certain number of centimeters to meet the minimum height
    requirement for riding Kingda Ka. --/
def RequiredGrowth (minimumHeight maryHeight : ℕ) : ℕ := minimumHeight - maryHeight

theorem MaryNeedingToGrow 
  (minimumHeight : ℕ := 140)
  (brothersHeight : ℕ := 180)
  (brothersHeightIs180 : brothersHeight = 180 := rfl)
  (heightRatio : ℕ → ℕ := MarysHeight)
  (maryHeight : ℕ := heightRatio brothersHeight)
  (maryHeightProof : maryHeight = 120 := by simp [MarysHeight, brothersHeightIs180])
  (requiredGrowth : ℕ := RequiredGrowth minimumHeight maryHeight) :
  requiredGrowth = 20 :=
by
  unfold RequiredGrowth MarysHeight
  rw [maryHeightProof]
  exact rfl

end MaryNeedingToGrow_l650_650888


namespace rectangle_diagonal_length_l650_650411

theorem rectangle_diagonal_length
  {A B C D O : Type} [rectangle A B C D O]
  (hOA : OA = 5) :
  length BD = 10 :=
by
  sorry

end rectangle_diagonal_length_l650_650411


namespace no_solution_to_system_l650_650005

theorem no_solution_to_system :
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 12 ∧ 9 * x - 12 * y = 15) :=
by
  sorry

end no_solution_to_system_l650_650005


namespace lines_are_concurrent_l650_650774

-- Given definitions
variables (A B C P Q T M : Point)
variable (Ω : Circle)
variables (ω : Circle)

-- Given conditions
axiom passes_through_BC (h1 : Ω.pass_through B) (h2 : Ω.pass_through C)
axiom tangent_to_Ω_at_T (h3 : ω.tangent_at Ω T)
axiom tangent_to_AB_AT_P (h4 : ω.tangent_at_side AB P)
axiom tangent_to_AC_AT_Q (h5 : ω.tangent_at_side AC Q)
axiom midpoint_of_arc_BC (h6 : M = midpoint_of_arc Ω B C T)

-- The theorem to prove
theorem lines_are_concurrent 
  (h1 : Ω.pass_through B)
  (h2 : Ω.pass_through C)
  (h3 : ω.tangent_at Ω T)
  (h4 : ω.tangent_at_side AB P)
  (h5 : ω.tangent_at_side AC Q)
  (h6 : M = midpoint_of_arc Ω B C T) :
  concurrent (line_through P Q) (line_through B C) (line_through M T) :=
sorry

end lines_are_concurrent_l650_650774


namespace cost_per_gallon_l650_650466

theorem cost_per_gallon (weekly_spend : ℝ) (two_week_usage : ℝ) (weekly_spend_eq : weekly_spend = 36) (two_week_usage_eq : two_week_usage = 24) : 
  (2 * weekly_spend / two_week_usage) = 3 :=
by sorry

end cost_per_gallon_l650_650466


namespace find_D_l650_650908

theorem find_D'' :
  let D := (4, 1)
  let D' := (4, -1)
  let D'' := (-2, 5)
  D'' = reflect_point_across_line (reflect_point_across_x_axis D) (y = x - 1) :=
sorry

-- Reflect a point across the x-axis
def reflect_point_across_x_axis (p : ℕ × ℕ) : ℕ × ℕ :=
  (p.1, -p.2)

-- Reflect point across the line y = x - 1
def reflect_point_across_line (p : ℕ × ℕ) (L : ℕ) : ℕ × ℕ :=
  let translated := (p.1, p.2 - 1) in
  let reflected := (translated.2, translated.1) in
  (reflected.1, reflected.2 + 1)

end find_D_l650_650908


namespace equivalent_angle_l650_650940

variable (k : ℤ)

def angle_set : Set ℝ := { β | ∃ k ∈ ℤ, β = k * 360 - 415 }

theorem equivalent_angle (h : ∃ β, β ∈ angle_set k ∧ 0 ≤ β ∧ β < 360) : ∃ β, β = 305 :=
begin
  sorry
end

end equivalent_angle_l650_650940


namespace expected_number_of_first_sequence_l650_650150

-- Define the concept of harmonic number
def harmonic (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), (1 : ℚ) / (i + 1)

-- Define the problem statement
theorem expected_number_of_first_sequence (n : ℕ) (h : n = 100) : harmonic n = 5.187 := by
  sorry

end expected_number_of_first_sequence_l650_650150


namespace original_cost_of_tomatoes_correct_l650_650374

noncomputable def original_cost_of_tomatoes := 
  let original_order := 25
  let new_tomatoes := 2.20
  let new_lettuce := 1.75
  let old_lettuce := 1.00
  let new_celery := 2.00
  let old_celery := 1.96
  let delivery_tip := 8
  let new_total_bill := 35
  let new_groceries := new_total_bill - delivery_tip
  let increase_in_cost := (new_lettuce - old_lettuce) + (new_celery - old_celery)
  let difference_due_to_substitutions := new_groceries - original_order
  let x := new_tomatoes + (difference_due_to_substitutions - increase_in_cost)
  x

theorem original_cost_of_tomatoes_correct :
  original_cost_of_tomatoes = 3.41 := by
  sorry

end original_cost_of_tomatoes_correct_l650_650374


namespace express_g_in_terms_of_f_l650_650297

variable {R : Type*} [CommRing R]

def isOdd (g : R → R) : Prop := ∀ x, g (-x) = -g x
def isEven (h : R → R) : Prop := ∀ x, h (-x) = h x
def symmetricDomain (f : R → R) : Prop := ∀ x, f (-x) = f x

theorem express_g_in_terms_of_f
  (f g h : R → R)
  (sym_f : symmetricDomain f)
  (hyp1 : ∀ x, f x = g x + h x)
  (hyp2 : isOdd g)
  (hyp3 : isEven h) :
  ∀ x, g x = (f x - f (-x)) / 2 :=
by
  sorry

end express_g_in_terms_of_f_l650_650297


namespace max_product_of_slopes_l650_650067

theorem max_product_of_slopes 
  (m₁ m₂ : ℝ)
  (h₁ : m₂ = 3 * m₁)
  (h₂ : abs ((m₂ - m₁) / (1 + m₁ * m₂)) = Real.sqrt 3) :
  m₁ * m₂ ≤ 2 :=
sorry

end max_product_of_slopes_l650_650067


namespace problem_l650_650458

theorem problem (a b : ℝ)
  (h1 : a^2 * (b^2 + 1) + b * (b + 2 * a) = 40)
  (h2 : a * (b + 1) + b = 8) :
  1 / a^2 + 1 / b^2 = 8 :=
begin
  sorry
end

end problem_l650_650458


namespace number_of_honest_dwarfs_l650_650906

-- Given conditions
variable (q1 q2 q3 q4 total_yes total_dwarfs honest_dwarfs : ℕ)
variable (h_q1 : q1 = 40)
variable (h_q2 : q2 = 50)
variable (h_q3 : q3 = 70)
variable (h_q4 : q4 = 100)
variable (h_total_dwarfs : total_dwarfs = 100)
variable (total_yes : total_yes = q1 + q2 + q3 = 160)

-- Target statement
theorem number_of_honest_dwarfs :
  honest_dwarfs = 40 :=
sorry

end number_of_honest_dwarfs_l650_650906


namespace speed_of_sound_l650_650115

theorem speed_of_sound (time_heard : ℕ) (time_occured : ℕ) (distance : ℝ) : 
  time_heard = 30 * 60 + 20 → 
  time_occured = 30 * 60 → 
  distance = 6600 → 
  (distance / ((time_heard - time_occured) / 3600)) / 3600 = 330 :=
by 
  intros h1 h2 h3
  sorry

end speed_of_sound_l650_650115


namespace b10_value_l650_650454

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 4 else
  if n = 2 then 5 else
  sequence (n - 1) * sequence (n - 2)

theorem b10_value : sequence 10 = 2560000000000000000000000000000000 := by
  sorry

end b10_value_l650_650454


namespace quadratic_function_symmetry_l650_650855

theorem quadratic_function_symmetry (a b x_1 x_2: ℝ) (h_roots: x_1^2 + a * x_1 + b = 0 ∧ x_2^2 + a * x_2 + b = 0)
(h_symmetry: ∀ x, (x - 2015)^2 + a * (x - 2015) + b = (x + 2015 - 2016)^2 + a * (x + 2015 - 2016) + b):
  (x_1 + x_2) / 2 = 2015 :=
sorry

end quadratic_function_symmetry_l650_650855


namespace pos_real_satisfies_eq_l650_650692

theorem pos_real_satisfies_eq (x : ℝ) (hx : x > 0) (h : real.cbrt (1 - x^4) + real.cbrt (1 + x^4) = 1) : x^8 = 28 / 27 := 
sorry

end pos_real_satisfies_eq_l650_650692


namespace number_of_squares_in_figure_150_l650_650841

theorem number_of_squares_in_figure_150 :
  let f (n : ℕ) := 2 * n^2 + 4 * n + 4 in
  f 150 = 45604 :=
by
  let f : ℕ → ℕ := λ n, 2 * n^2 + 4 * n + 4
  have h1 : f 0 = 4 := by rfl
  have h2 : f 1 = 10 := by rfl
  have h3 : f 2 = 20 := by rfl
  have h4 : f 3 = 34 := by rfl
  show f 150 = 45604 from by sorry

end number_of_squares_in_figure_150_l650_650841


namespace max_lg_value_l650_650280

noncomputable def max_lg_product (x y : ℝ) (hx: x > 1) (hy: y > 1) (hxy: Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) : ℝ :=
  4

theorem max_lg_value (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  max_lg_product x y hx hy hxy = 4 := 
by
  unfold max_lg_product
  sorry

end max_lg_value_l650_650280


namespace sum_of_diagonals_l650_650437

def FG : ℝ := 4
def HI : ℝ := 4
def GH : ℝ := 11
def IJ : ℝ := 11
def FJ : ℝ := 15

theorem sum_of_diagonals (x y z : ℝ) (h1 : z^2 = 4 * x + 121) (h2 : z^2 = 11 * y + 16)
  (h3 : x * y = 44 + 15 * z) (h4 : x * z = 4 * z + 225) (h5 : y * z = 11 * z + 60) :
  3 * z + x + y = 90 :=
sorry

end sum_of_diagonals_l650_650437


namespace no_pythagorean_triple_15_8_19_l650_650163

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a*a + b*b = c*c

theorem no_pythagorean_triple_15_8_19 :
  ¬is_pythagorean_triple 15 8 19 :=
by {
  simp [is_pythagorean_triple],
  norm_num,
  exact dec_trivial
}

end no_pythagorean_triple_15_8_19_l650_650163


namespace parallelogram_area_l650_650711

noncomputable def area_of_parallelogram
  (p q a b : ℝ^3)
  (hp : ‖p‖ = 1)
  (hq : ‖q‖ = 2)
  (θ : ℝ)
  (hθ : θ = Real.pi / 3)
  (ha : a = 5 * p + q)
  (hb : b = p - 3 * q) : ℝ :=
  ‖a × b‖

theorem parallelogram_area
  (p q a b : ℝ^3)
  (hp : ‖p‖ = 1)
  (hq : ‖q‖ = 2)
  (θ : ℝ)
  (hθ : θ = Real.pi / 3)
  (ha : a = 5 * p + q)
  (hb : b = p - 3 * q) :
  area_of_parallelogram p q a b hp hq θ hθ ha hb = 16 * Real.sqrt 3 :=
sorry

end parallelogram_area_l650_650711


namespace total_paved_1120_l650_650693

-- Definitions based on given problem conditions
def workers_paved_april : ℕ := 480
def less_than_march : ℕ := 160
def workers_paved_march : ℕ := workers_paved_april + less_than_march
def total_paved : ℕ := workers_paved_april + workers_paved_march

-- The statement to prove
theorem total_paved_1120 : total_paved = 1120 := by
  sorry

end total_paved_1120_l650_650693


namespace fewest_tiles_needed_to_cover_rectangle_l650_650159

noncomputable def height_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 2) * side_length

noncomputable def area_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (1 / 2) * side_length * height_of_equilateral_triangle side_length

noncomputable def area_of_floor_in_square_inches (length_in_feet : ℝ) (width_in_feet : ℝ) : ℝ :=
  length_in_feet * width_in_feet * (12 * 12)

noncomputable def number_of_tiles_required (floor_area : ℝ) (tile_area : ℝ) : ℝ :=
  floor_area / tile_area

theorem fewest_tiles_needed_to_cover_rectangle :
  number_of_tiles_required (area_of_floor_in_square_inches 3 4) (area_of_equilateral_triangle 2) = 997 := 
by
  sorry

end fewest_tiles_needed_to_cover_rectangle_l650_650159


namespace percentage_of_whole_l650_650092

theorem percentage_of_whole (part whole percent : ℕ) (h1 : part = 120) (h2 : whole = 80) (h3 : percent = 150) : 
  part = (percent / 100) * whole :=
by
  sorry

end percentage_of_whole_l650_650092


namespace simplify_division_l650_650928

theorem simplify_division :
  (9 * 10^10) / (3 * 10^3 - 2 * 10^3) = 90000000 :=
by
  have h : 3 * 10^3 - 2 * 10^3 = 10^3 :=
    by
    calc
      3 * 10^3 - 2 * 10^3 = (3 - 2) * 10^3 : by sorry
                        ... = 1 * 10^3 : by sorry
                        ... = 10^3 : by sorry
  calc
    (9 * 10^10) / (3 * 10^3 - 2 * 10^3) = (9 * 10^10) / 10^3 : by rw h
                                  ... = 9 * 10^10 * 10^(-3) : by sorry
                                  ... = 9 * 10^7 : by sorry
                                  ... = 90000000 : by norm_num

end simplify_division_l650_650928


namespace three_consecutive_cards_sum_l650_650500

theorem three_consecutive_cards_sum (x : ℤ) (h: (x - 6) + x + (x + 6) = 342) :
  {a b c : ℤ // a = x - 6 ∧ b = x ∧ c = x + 6} = {108, 114, 120} :=
sorry

end three_consecutive_cards_sum_l650_650500


namespace six_letter_words_count_l650_650730

def first_letter_possibilities := 26
def second_letter_possibilities := 26
def third_letter_possibilities := 26
def fourth_letter_possibilities := 26

def number_of_six_letter_words : Nat := 
  first_letter_possibilities * 
  second_letter_possibilities * 
  third_letter_possibilities * 
  fourth_letter_possibilities

theorem six_letter_words_count : number_of_six_letter_words = 456976 := by
  sorry

end six_letter_words_count_l650_650730


namespace largest_prime_factor_of_1729_l650_650626

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650626


namespace no_prime_in_sequence_l650_650657

theorem no_prime_in_sequence :
  let Q := ∏ p in Finset.filter Nat.Prime (Finset.range 32), p
  ∀ m : ℕ, 2 ≤ m ∧ m ≤ 32 → ¬ Nat.Prime (Q + m) :=
by
  sorry

end no_prime_in_sequence_l650_650657


namespace chris_and_dana_rest_days_l650_650180

def chris_schedule : ℕ → bool
| n := (n % 6 = 4) ∨ (n % 6 = 5)

def dana_schedule : ℕ → bool
| n := (n % 7 = 5) ∨ (n % 7 = 6)

def coinciding_rest_days_up_to (days : ℕ) : ℕ :=
(finset.range days).filter (λ n, chris_schedule n ∧ dana_schedule n).card

theorem chris_and_dana_rest_days (days : ℕ) : coinciding_rest_days_up_to 1000 = 23 := by
  sorry

end chris_and_dana_rest_days_l650_650180


namespace function_has_neither_min_nor_max_l650_650317

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

def domain (x : ℝ) : Prop := x ∈ Set.Icc (-2 : ℝ) (1 : ℝ) ∨ x ∈ Set.Ioc (1 : ℝ) (6 : ℝ)

theorem function_has_neither_min_nor_max :
  ¬ (∃ x_min ∈ Set.Icc (-2 : ℝ) (1 : ℝ) ∨ x_min ∈ Set.Ioc (1 : ℝ) (6 : ℝ), 
      ∀ x ∈ Set.Icc (-2 : ℝ) (1 : ℝ) ∨ x ∈ Set.Ioc (1 : ℝ) (6 : ℝ), f(x_min) ≤ f(x)) ∧
  ¬ (∃ x_max ∈ Set.Icc (-2 : ℝ) (1 : ℝ) ∨ x_max ∈ Set.Ioc (1 : ℝ) (6 : ℝ), 
      ∀ x ∈ Set.Icc (-2 : ℝ) (1 : ℝ) ∨ x ∈ Set.Ioc (1 : ℝ) (6 : ℝ), f(x_max) ≥ f(x)) :=
sorry

end function_has_neither_min_nor_max_l650_650317


namespace largest_prime_factor_1729_l650_650571

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650571


namespace multiplication_of_variables_l650_650658

theorem multiplication_of_variables 
  (a b c d : ℚ)
  (h1 : 3 * a + 2 * b + 4 * c + 6 * d = 48)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : 2 * c - 2 = d) :
  a * b * c * d = -58735360 / 81450625 := 
sorry

end multiplication_of_variables_l650_650658


namespace infinite_n_gcd_floor_sqrt_D_eq_m_l650_650776

theorem infinite_n_gcd_floor_sqrt_D_eq_m (D m : ℕ) (hD : ∀ k : ℕ, k * k ≠ D) (hm : 0 < m) :
  ∃ᶠ n in filter.at_top, Int.gcd n (Nat.floor (Real.sqrt D * n)) = m :=
sorry

end infinite_n_gcd_floor_sqrt_D_eq_m_l650_650776


namespace local_max_neg_f_neg_x_is_local_min_l650_650459

variable (f : ℝ → ℝ)
variable (x0 : ℝ)
variable (h : x0 ≠ 0)
variable (h_local_max : ∃ ε > 0, ∀ x ∈ set.Ioo (x0 - ε) (x0 + ε), f x ≤ f x0)

theorem local_max_neg_f_neg_x_is_local_min :
  ∃ ε > 0, ∀ x ∈ set.Ioo (-x0 - ε) (-x0 + ε), -f (-x) ≥ -f (-x0) :=
sorry

end local_max_neg_f_neg_x_is_local_min_l650_650459


namespace smallest_positive_four_digit_number_l650_650996

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def has_two_even_two_odd_digits (n : ℕ) : Prop := 
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  (digits.count (λ d, d % 2 = 0) = 2) ∧ (digits.count (λ d, d % 2 = 1) = 2)
def thousands_digit_between_2_and_3 (n : ℕ) : Prop := 
  let d := n / 1000 % 10 in d = 2 ∨ d = 3

theorem smallest_positive_four_digit_number :
  ∃ n : ℕ, is_four_digit n ∧ is_divisible_by_3 n ∧ has_two_even_two_odd_digits n ∧ thousands_digit_between_2_and_3 n ∧
  (∀ m : ℕ, is_four_digit m ∧ is_divisible_by_3 m ∧ has_two_even_two_odd_digits m ∧ thousands_digit_between_2_and_3 m → n ≤ m) ∧ n = 3009 :=
by 
  sorry

end smallest_positive_four_digit_number_l650_650996


namespace num_zeros_of_f_l650_650031

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x - 2

theorem num_zeros_of_f : ∃! x₁ x₂ ∈ ℝ, x₁ ≠ x₂ ∧ f(x₁) = 0 ∧ f(x₂) = 0 :=
by
  sorry

end num_zeros_of_f_l650_650031


namespace problem_value_l650_650075

theorem problem_value :
  1 - (-2) - 3 - (-4) - 5 - (-6) = 5 :=
by sorry

end problem_value_l650_650075


namespace express_g_in_terms_of_f_l650_650295

variables {X Y : Type} [NormedAddCommGroup X] [NormedSpace ℝ X] [NormedAddCommGroup Y] [NormedSpace ℝ Y]

-- Define odd and even functions
def is_odd (g : X → Y) := ∀ x, g (- x) = - g x
def is_even (h : X → Y) := ∀ x, h (- x) = h x

-- Define f, g, h and their relationships
variables (f g h : X → Y)
variable [h1: ∀ x, f x = g x + h x]
variable [h2: is_odd g]
variable [h3: is_even h]

-- Proof statement
theorem express_g_in_terms_of_f:
  g = λ x, (f x - f (- x)) / 2 :=
by sorry

end express_g_in_terms_of_f_l650_650295


namespace count100DigitEvenNumbers_is_correct_l650_650337

noncomputable def count100DigitEvenNumbers : ℕ :=
  let validDigits : Finset ℕ := {0, 1, 3}
  let firstDigitChoices : ℕ := 2  -- Only 1 or 3
  let middleDigitsChoices : ℕ := 3 ^ 98  -- 3 choices for each of the 98 middle positions
  let lastDigitChoices : ℕ := 1  -- Only 0 (even number requirement)
  firstDigitChoices * middleDigitsChoices * lastDigitChoices

theorem count100DigitEvenNumbers_is_correct :
  count100DigitEvenNumbers = 2 * 3 ^ 98 := by
  sorry

end count100DigitEvenNumbers_is_correct_l650_650337


namespace range_of_x_l650_650376

variable {x θ : ℝ}

theorem range_of_x (h : sin θ = 1 - log x / log 2) : 1 ≤ x ∧ x ≤ 4 := by
  sorry

end range_of_x_l650_650376


namespace proof_l650_650014

noncomputable theory

def arithmetic_sequence (n : ℕ) (d : ℝ) : ℝ := 1 + (n-1) * d
def sum_arithmetic_sequence (n : ℕ) (d : ℝ) : ℝ :=
  ∑ i in Finset.range (n + 1), arithmetic_sequence i d

def geometric_sequence (n : ℕ) (q : ℝ) : ℝ := 2 * q^(n-1)

def c_sequence (n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) : ℝ := 
  b n + 1 / (∑ i in Finset.range (n + 1), a i)

def t_sum (n : ℕ) (c : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range (n + 1), c i

theorem proof (d q : ℝ) (n : ℕ) (hn : 0 < d) (a : ℕ → ℝ := arithmetic_sequence d)
  (b : ℕ → ℝ := geometric_sequence q) :

  b 2 * sum_arithmetic_sequence 2 d = 12 → 
  b 2 + sum_arithmetic_sequence 3 d = 10 →

  (∀ n, a n = n) ∧ 
  (∀ n, b n = 2^n) ∧ 
  t_sum n (c_sequence (a, b)) = 2^(n+1) - 2 / (n+1) :=
sorry

end proof_l650_650014


namespace max_a_value_l650_650301

noncomputable def piecewise_function (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + 1 / x + a

theorem max_a_value : ∀ (a : ℝ),
  (∀ x, piecewise_function a 0 ≤ piecewise_function a x) ↔ a = 2 :=
begin
  sorry
end

end max_a_value_l650_650301


namespace rename_not_always_possible_l650_650087

variable (G : SimpleGraph ℕ)
variable (A B : ℕ)
variable (adj : G.Adj)

theorem rename_not_always_possible : 
  ¬(∀ (A B : ℕ) (G' : SimpleGraph ℕ), (∀ W, (G.Adj W A ↔ G'.Adj W B) ∧ (G.Adj W B ↔ G'.Adj W A)) → (G = G')) :=
sorry

end rename_not_always_possible_l650_650087


namespace number_of_factors_180_l650_650357

theorem number_of_factors_180 : 
  let factors180 (n : ℕ) := 180 
  in factors180 180 = (2 + 1) * (2 + 1) * (1 + 1) => factors180 180 = 18 := 
by
  sorry

end number_of_factors_180_l650_650357


namespace problem_statement_l650_650848

structure Point where
  x : ℝ
  y : ℝ

def lineEquation (A B : Point) : Real × Real :=
  let k := (B.y - A.y) / (B.x - A.x)
  let b := A.y - k * A.x
  (k, b)

def collinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - B.x) = (C.y - B.y) * (B.x - A.x)

noncomputable def pointA : Point := ⟨-1, 4⟩
noncomputable def pointB : Point := ⟨-3, 2⟩
noncomputable def pointC : Point := ⟨0, 6⟩

theorem problem_statement :
  let line_ab := lineEquation pointA pointB
in line_ab = (1, 5) ∧ ¬collinear pointA pointB pointC :=
by
  let line_ab := lineEquation pointA pointB
  show line_ab = (1, 5) ∧ ¬collinear pointA pointB pointC
  sorry

end problem_statement_l650_650848


namespace greatest_value_x_l650_650956

theorem greatest_value_x (x : ℕ) (h : lcm (lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_value_x_l650_650956


namespace largest_prime_factor_of_1729_l650_650554

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650554


namespace ratio_of_squares_l650_650007

theorem ratio_of_squares (y : ℝ) : 
  let area_small := (3 * y) ^ 2,
      area_large := (9 * y) ^ 2
  in area_small / area_large = 1 / 9 :=
by
  sorry

end ratio_of_squares_l650_650007


namespace total_water_intake_l650_650044

def theo_weekday := 8
def mason_weekday := 7
def roxy_weekday := 9
def zara_weekday := 10
def lily_weekday := 6

def theo_weekend := 10
def mason_weekend := 8
def roxy_weekend := 11
def zara_weekend := 12
def lily_weekend := 7

def total_cups_in_week (weekday_cups weekend_cups : ℕ) : ℕ :=
  5 * weekday_cups + 2 * weekend_cups

theorem total_water_intake :
  total_cups_in_week theo_weekday theo_weekend +
  total_cups_in_week mason_weekday mason_weekend +
  total_cups_in_week roxy_weekday roxy_weekend +
  total_cups_in_week zara_weekday zara_weekend +
  total_cups_in_week lily_weekday lily_weekend = 296 :=
by sorry

end total_water_intake_l650_650044


namespace length_of_DB_l650_650849

theorem length_of_DB 
  (ABC_right_angle : ∠ABC = 90)
  (ADB_right_angle : ∠ADB = 90)
  (AC : ℝ := 24.7) 
  (AD : ℝ := 7) : 
  ∃ DB, DB = Real.sqrt 123.9 :=
by
  sorry

end length_of_DB_l650_650849


namespace GIS_leads_to_overfishing_l650_650006

noncomputable def GIS_effect_on_fishery_production
  (locate_schools: Prop)  -- Fishermen have used GIS technology to locate schools of fish
  (increase_catch: Prop)  -- Using GIS technology can increase the fish catch in a short time
  : Prop :=
  locate_schools ∧ increase_catch → 
  ∃ (overfishing: Prop), overfishing ∧ fishery_resources_exhausted

-- Suppose the following assumptions hold
axiom locate_schools : Prop
axiom increase_catch : Prop
axiom fishery_resources_exhausted : Prop

-- We need to prove that if GIS technology is widely introduced, it can lead to overfishing
theorem GIS_leads_to_overfishing : GIS_effect_on_fishery_production locate_schools increase_catch :=
sorry

end GIS_leads_to_overfishing_l650_650006


namespace largest_prime_factor_1729_l650_650574

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650574


namespace sodium_chloride_solution_l650_650426

theorem sodium_chloride_solution (n y : ℝ) (h1 : n > 30) 
  (h2 : 0.01 * n * n = 0.01 * (n - 8) * (n + y)) : 
  y = 8 * n / (n + 8) :=
sorry

end sodium_chloride_solution_l650_650426


namespace largest_prime_factor_of_1729_l650_650590

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650590


namespace problem_proof_l650_650432

open EuclideanGeometry

-- Declare the variables, types, and the problem statement
variables {P : Type*} [metric_space P] [normed_add_torsor ℝ P] -- Metric space to deal with points in Euclidean geometry

-- Given conditions and definitions in the problem
variables (A B C D E F : P)
variable (quadrilateral_ABC: P → P → P → P → Prop)
variable (angle_bisector: P → P → P → P → Prop)
variable (intersects: P → P → P → P → Prop)
variable (cyclic: set P → Prop)

-- Definitions related to the given problem statement
def convex_quadrilateral (A B C D : P) := quadrilateral_ABC A B C D ∧ 
  ¬ (collinear A B D ∨ collinear B C D ∨ collinear C D A ∨ collinear D A B)

def angle_bisectors_intersect (A B E : P) := angle_bisector A B E ∧ angle_bisector B C E

def lines_intersect (A B C D F : P) := intersects A B C D ∧ F ∈ line_through A B ∩ line_through C D

-- Given the provided conditions in the problem statement
axiom angle_bisectors_at_E {A B C D E : P} (h₁ : convex_quadrilateral A B C D) (h₂ : angle_bisectors_intersect B C E) : true

-- Provided and to be proved conclusion in the problem statement
theorem problem_proof {A B C D E F : P} (h₁ : convex_quadrilateral A B C D) (h₂ : angle_bisectors_intersect B C E)
   (h₃ : lines_intersect A B C D F) (h₄ : dist A B + dist C D = dist B C) : cyclic {A, D, E, F} :=
sorry

end problem_proof_l650_650432


namespace ratio_of_squares_l650_650008

theorem ratio_of_squares (y : ℝ) : 
  let area_small := (3 * y) ^ 2,
      area_large := (9 * y) ^ 2
  in area_small / area_large = 1 / 9 :=
by
  sorry

end ratio_of_squares_l650_650008


namespace mia_one_die_reroll_probability_l650_650891

open BigOperators

-- Define the probability function for Mia, given her strategy
def mia_reroll_one_probability : ℚ :=
  let successful_outcomes := 11
  let total_outcomes := 6^3
  successful_outcomes / total_outcomes

-- Prove that Mia's probability of rerolling exactly one die to get a sum of 9 is 11/216
theorem mia_one_die_reroll_probability : mia_reroll_one_probability = 11 / 216 := by
  unfold mia_reroll_one_probability
  norm_num
  done sorry

end mia_one_die_reroll_probability_l650_650891


namespace find_eqidistant_point_l650_650754

theorem find_eqidistant_point :
  ∃ (y z : ℝ), 
    (y = 1 ∧ z = -1) ∧ 
    (0 = ((y-1)^2 + (z-1)^2 - (y-2)^2 - (z-1)^2 - 4)) ∧ 
    (0 = ((y-1)^2 + (z-1)^2 - (y-3)^2 - (z+2)^2 - 9)) :=
by {
  use [1, -1],
  sorry
}

end find_eqidistant_point_l650_650754


namespace terminal_side_half_angle_l650_650799

theorem terminal_side_half_angle {k : ℤ} {α : ℝ} 
  (h : 2 * k * π < α ∧ α < 2 * k * π + π / 2) : 
  (k * π < α / 2 ∧ α / 2 < k * π + π / 4) ∨ (k * π + π <= α / 2 ∧ α / 2 < (k + 1) * π + π / 4) :=
sorry

end terminal_side_half_angle_l650_650799


namespace digitalEarth_incorrect_statement_l650_650915

-- Define each condition as a separate proposition
def canSimulateEnvironmentalImpact (digitalEarth : Type) : Prop :=
  ∀ species, digitalEarth.simulateImpactOnSpecies species

def canMonitorCropPests (digitalEarth : Type) : Prop :=
  digitalEarth.monitorCropPestsAndGrowth

def canPredictSubmersion (digitalEarth : Type) : Prop :=
  digitalEarth.predictSubmergedAreasAfterGreenhouseEffect

def canSimulatePastButNotPredictFuture (digitalEarth : Type) : Prop :=
  digitalEarth.simulatePastEnvironments ∧ ¬digitalEarth.predictFuture

-- The proposed incorrect statement
def predictFutureAndNotSimulatePast (digitalEarth : Type) : Prop :=
  digitalEarth.predictFuture ∧ ¬digitalEarth.simulatePastEnvironments

-- The proof problem
theorem digitalEarth_incorrect_statement (digitalEarth : Type)
  (H1 : canSimulateEnvironmentalImpact digitalEarth)
  (H2 : canMonitorCropPests digitalEarth)
  (H3 : canPredictSubmersion digitalEarth)
  (H4 : canSimulatePastButNotPredictFuture digitalEarth) :
  ¬predictFutureAndNotSimulatePast digitalEarth :=
begin
  sorry -- proof to be completed
end

end digitalEarth_incorrect_statement_l650_650915


namespace geometric_sequence_problem_l650_650417

variable {a : ℕ → ℝ}

-- Define the conditions of the problem
def condition1 : Prop := a 5 * a 11 = 3
def condition2 : Prop := a 3 + a 13 = 4

theorem geometric_sequence_problem (h1 : condition1) (h2 : condition2) : 
  ∃ q : ℝ, q = a 15 / a 5 ∧ (q = 1 / 3 ∨ q = 3) :=
by
  sorry

end geometric_sequence_problem_l650_650417


namespace diff_count_of_set_l650_650816

theorem diff_count_of_set : 
  let S := {1, 2, 3, 4, 5, 7}
  in (card {d | ∃ x y ∈ S, x ≠ y ∧ d = |x - y|}) = 6 :=
by {
  sorry
}

end diff_count_of_set_l650_650816


namespace number_of_moles_of_HCl_l650_650213

-- Defining the chemical equation relationship
def reaction_relation (HCl NaHCO3 NaCl H2O CO2 : ℕ) : Prop :=
  H2O = HCl ∧ H2O = NaHCO3

-- Conditions
def conditions (HCl NaHCO3 H2O : ℕ) : Prop :=
  NaHCO3 = 3 ∧ H2O = 3

-- Theorem statement proving the number of moles of HCl given the conditions
theorem number_of_moles_of_HCl (HCl NaHCO3 NaCl H2O CO2 : ℕ) 
  (h1 : reaction_relation HCl NaHCO3 NaCl H2O CO2) 
  (h2 : conditions HCl NaHCO3 H2O) :
  HCl = 3 :=
sorry

end number_of_moles_of_HCl_l650_650213


namespace minimum_value_of_product_l650_650453

theorem minimum_value_of_product (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 30 := 
sorry

end minimum_value_of_product_l650_650453


namespace correlation_coefficient_interpretation_l650_650653

-- Definitions and problem statement

/-- 
Theorem: Correct interpretation of the correlation coefficient r.
Given r in (-1, 1):
The closer |r| is to zero, the weaker the correlation between the two variables.
-/
theorem correlation_coefficient_interpretation (r : ℝ) (h : -1 < r ∧ r < 1) :
  (r > 0 -> false) ∧ (r > 1 -> false) ∧ (0 < r -> false) ∧ (|r| -> Prop) :=
sorry

end correlation_coefficient_interpretation_l650_650653


namespace basketball_team_win_requirement_l650_650677

theorem basketball_team_win_requirement :
  ∀ (initial_wins : ℕ) (initial_games : ℕ) (total_games : ℕ) (target_win_rate : ℚ) (total_wins : ℕ),
    initial_wins = 30 →
    initial_games = 60 →
    total_games = 100 →
    target_win_rate = 65 / 100 →
    total_wins = total_games * target_win_rate →
    total_wins - initial_wins = 35 :=
by
  -- variables and hypotheses declaration are omitted
  sorry

end basketball_team_win_requirement_l650_650677


namespace same_result_probability_l650_650373
open Nat Real 
open ProbabilityTheory

def side_distribution (n : ℕ) : Probability :=
  match n with
  | 3 => 1/4   -- Purple sides
  | 4 => 1/3   -- Green sides
  | 1 => 1/12  -- Glittery side
  | _ => 1/3   -- Orange sides (default to 1/3 for remaining sides)

theorem same_result_probability :
  let p_purple : Probability := (side_distribution 3) ^ 2,
      p_green : Probability := (side_distribution 4) ^ 2,
      p_orange : Probability := (side_distribution 4) ^ 2,
      p_glittery : Probability := (side_distribution 1) ^ 2 in
  (p_purple + p_green + p_orange + p_glittery) = 7 / 24 := sorry

end same_result_probability_l650_650373


namespace value_of_abc_div_def_l650_650825

noncomputable def ratio1 (a b : ℝ) := a / b = 1 / 3
noncomputable def ratio2 (b c : ℝ) := b / c = 2
noncomputable def ratio3 (c d : ℝ) := c / d = 1 / 2
noncomputable def ratio4 (d e : ℝ) := d / e = 3
noncomputable def ratio5 (e f : ℝ) := e / f = 1 / 2

theorem value_of_abc_div_def (a b c d e f : ℝ) 
  (h1 : ratio1 a b) 
  (h2 : ratio2 b c) 
  (h3 : ratio3 c d) 
  (h4 : ratio4 d e) 
  (h5 : ratio5 e f) : 
  abc / def = 1 / 12 :=
sorry

end value_of_abc_div_def_l650_650825


namespace count_ways_to_exhaust_black_matches_l650_650127

theorem count_ways_to_exhaust_black_matches 
  (n r g : ℕ) 
  (h_r_le_n : r ≤ n) 
  (h_g_le_n : g ≤ n) 
  (h_r_ge_0 : 0 ≤ r) 
  (h_g_ge_0 : 0 ≤ g) 
  (h_n_ge_0 : 0 < n) :
  ∃ ways : ℕ, ways = (Nat.factorial (3 * n - r - g - 1)) / (Nat.factorial (n - 1) * Nat.factorial (n - r) * Nat.factorial (n - g)) :=
by
  sorry

end count_ways_to_exhaust_black_matches_l650_650127


namespace symmetric_point_z_axis_l650_650507

def reflection_z_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-P.1, -P.2, P.3)

theorem symmetric_point_z_axis :
  reflection_z_axis (1, 1, 1) = (-1, -1, 1) :=
by
  -- proof goes here
  sorry

end symmetric_point_z_axis_l650_650507


namespace sequence_count_l650_650189

theorem sequence_count : 
  let valid_sequences := { s : list ℕ // s.length = 15 ∧ ((∀ i, s.nth i ≠ some 0) ∨ (∀ i, s.nth i ≠ some 1) ∨
    (∀ i j, i < j → s.nth i = some 0 → s.nth j ≠ some 1) ∨
    (∀ i j, i < j → s.nth i = some 1 → s.nth j ≠ some 0)) ∧
    ¬ (∀ i < 6, s.nth i = some 0) ∧ ¬ (∀ i < 9, s.nth i = some 1) } 
in
  valid_sequences.card = 269 := 
by
  sorry

end sequence_count_l650_650189


namespace trapezoid_inradii_relation_l650_650670

open Real

theorem trapezoid_inradii_relation 
  (ABCD : Trapezoid) 
  (isInscribed : Inscribed ABCD) 
  (AC BD : Line) 
  (E : Point) 
  (intersection : E = intersection_point AC BD) 
  (r1 r2 r3 r4 : ℝ) 
  (r1_is_inradius_ABE : is_inradius E (triangle E ABCD.a ABCD.b)) 
  (r2_is_inradius_BCE : is_inradius E (triangle E ABCD.b ABCD.c)) 
  (r3_is_inradius_CDE : is_inradius E (triangle E ABCD.c ABCD.d)) 
  (r4_is_inradius_DAE : is_inradius E (triangle E ABCD.d ABCD.a)) 
  : (1 / r1 + 1 / r3 = 1 / r2 + 1 / r4) :=
sorry

end trapezoid_inradii_relation_l650_650670


namespace f1_not_in_A_f2_in_A_l650_650041

def f1 (x : ℝ) : ℝ := Real.log2 x
def f2 (x : ℝ) : ℝ := (x + 1) ^ 2

def condition (f : ℝ → ℝ) : Prop := 
  ∀ (x y : ℝ), x > 0 → y > 0 → x ≠ y → f x + 2 * f y > 3 * f ((x + 2 * y) / 3)

theorem f1_not_in_A : ¬ condition f1 := 
sorry
  
theorem f2_in_A : condition f2 := 
sorry

end f1_not_in_A_f2_in_A_l650_650041


namespace smallest_value_abs_diff_is_11_l650_650732

noncomputable def absolute_difference (m n : ℕ) : ℕ := 
  abs (36 ^ m - 5 ^ n)

theorem smallest_value_abs_diff_is_11 : absolute_difference 1 2 = 11 := by
  sorry

end smallest_value_abs_diff_is_11_l650_650732


namespace misha_darts_score_l650_650464

theorem misha_darts_score (x : ℕ) 
  (h1 : x >= 24)
  (h2 : x * 3 <= 72) : 
  2 * x = 48 :=
by
  sorry

end misha_darts_score_l650_650464


namespace sally_money_l650_650920

def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20

theorem sally_money : total_seashells * price_per_seashell = 54 :=
by
  sorry

end sally_money_l650_650920


namespace number_of_factors_180_l650_650353

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.eraseDuplicates.map (λ p, n.factors.count p + 1)).foldr (· * ·) 1

theorem number_of_factors_180 : number_of_factors 180 = 18 := by
  sorry

end number_of_factors_180_l650_650353


namespace sum_prime_factors_of_expression_l650_650037

theorem sum_prime_factors_of_expression : ∃ p1 p2 p3 p4 p5 p6 : ℕ,
  (p1, p2, p3, p4, p5, p6).pairwise (≠) ∧
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ p6.prime ∧
  (∃ n : ℕ, 
    n = ∑ i in finset.range (10), (i + (-9)^i) * 8^(9-i) * (nat.choose 9 i) ∧ 
    n % p1 = 0 ∧ n % p2 = 0 ∧ n % p3 = 0 ∧ n % p4 = 0 ∧ n % p5 = 0 ∧ n % p6 = 0 ∧
    p1 + p2 + p3 + p4 + p5 + p6 = 835) :=
begin
  sorry
end

end sum_prime_factors_of_expression_l650_650037


namespace problem_solution_interval_l650_650726

theorem problem_solution_interval (a : ℝ)
  (h1 : ∃ x y z : ℝ, 3 * x^2 + 2 * y^2 + 2 * z^2 = a)
  (h2 : ∃ x y z : ℝ, 4 * x^2 + 4 * y^2 + 5 * z^2 = 1 - a) :
  a ∈ set.Icc (2 / 7 : ℝ) (3 / 7 : ℝ) :=
sorry

end problem_solution_interval_l650_650726


namespace count_100_digit_even_numbers_l650_650328

theorem count_100_digit_even_numbers : 
  let valid_digits := {0, 1, 3}
  let num_digits := 100
  let num_even_digits := 2 * 3^98
  ∀ n : ℕ, n = num_digits → (∃ (digits : Fin n → ℕ), 
    (∀ i, digits i ∈ valid_digits) ∧ 
    digits 0 ≠ 0 ∧ 
    digits (n-1) = 0) → 
    (num_even_digits = 2 * 3^98) :=
by
  sorry

end count_100_digit_even_numbers_l650_650328


namespace proof_problem_statement_l650_650694

noncomputable def proof_problem : Prop :=
  ∀ (A B C C1 B1 H : EuclideanSpace ℝ (Fin 2)), 
  -- A semicircle is inscribed in triangle ABC
  let semicircle_center := midpoint B C,
  let semicircle_radius := dist semicircle_center B / 2,
  let s : Set (EuclideanSpace ℝ (Fin 2)) := (Metric.closedBall semicircle_center semicircle_radius) ∩ { p | ∃ x, dist p (midpoint B C) = max (dist p B) (dist p C) },
  -- The diameter of the semicircle lies on side BC
  semicircle_center = midpoint B C →
  -- The arc touches sides AB and AC at points C1 and B1 respectively
  dist A C1 = dist B C1 → -- C1 on the semicircle touches AB
  dist A B1 = dist C B1 → -- B1 on the semicircle touches AC
  -- H is the foot of the altitude drawn from point A to side BC
  dist (line B C).proj A H = 0 →
  -- Prove the given equality
  (dist A C1 / dist C1 B) * (dist B H / dist H C) * (dist C B1 / dist B1 A) = 1

theorem proof_problem_statement : proof_problem := sorry

end proof_problem_statement_l650_650694


namespace unique_solution_inequality_l650_650832

theorem unique_solution_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) → a = 2 :=
by
  sorry

end unique_solution_inequality_l650_650832


namespace parabola_relationship_l650_650032

theorem parabola_relationship 
  (c : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : y1 = 2*(-2 - 1)^2 + c) 
  (h2 : y2 = 2*(0 - 1)^2 + c) 
  (h3 : y3 = 2*((5:ℝ)/3 - 1)^2 + c):
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end parabola_relationship_l650_650032


namespace range_of_a_decreasing_function_l650_650399

theorem range_of_a_decreasing_function :
  (∀ x y : ℝ, x < y → (2 * a - 5)^x > (2 * a - 5)^y) ↔ (5 / 2 < a ∧ a < 3) :=
by
  sorry

end range_of_a_decreasing_function_l650_650399


namespace scientific_notation_35000_l650_650907

theorem scientific_notation_35000 : ∃ (a : ℝ) (n : ℤ), 3.5 = a ∧ 4 = n ∧ 35000 = a * 10^n :=
by
    use 3.5, 4
    split
    exact rfl
    split
    exact rfl
    sorry

end scientific_notation_35000_l650_650907


namespace value_of_b_minus_a_l650_650084

variable (a b : ℕ)

theorem value_of_b_minus_a 
  (h1 : b = 10)
  (h2 : a * b = 2 * (a + b) + 12) : b - a = 6 :=
by sorry

end value_of_b_minus_a_l650_650084


namespace simplify_expression_l650_650004

theorem simplify_expression (x y : ℝ) :
  ((x + y)^2 - y * (2 * x + y) - 6 * x) / (2 * x) = (1 / 2) * x - 3 :=
by
  sorry

end simplify_expression_l650_650004


namespace heaviest_object_weight_distinct_weights_count_l650_650779

theorem heaviest_object_weight {W : Type} [has_add W] [has_le W] (w1 w4 w9 : W) (h1: w1 = 1) (h4: w4 = 4) (h9: w9 = 9) :
  w1 + w4 + w9 = 14 := sorry

theorem distinct_weights_count {W : Type} [has_add W] [has_le W] (w1 w4 w9 : W) (h1: w1 = 1) (h4: w4 = 4) (h9: w9 = 9) :
  ∃ (weights : Finset W), weights.card = 14 ∧ (∀ w ∈ weights, ∃ (a b c : ℕ), w = a * w1 + b * w4 + c * w9) := sorry

end heaviest_object_weight_distinct_weights_count_l650_650779


namespace largest_prime_factor_of_1729_l650_650547

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650547


namespace range_of_m_l650_650320

open Set Real

noncomputable def e := exp 1

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x1 ∈ Icc (-1 : ℝ) 2, ∃ x0 ∈ Icc (-1 : ℝ) 1, g x0 = f x1) →
  (∀ x, f x = x - exp x) →
  (∀ x, g x = m * x + 1) →
  m ∈ Icc (e + 1) +∞ := 
by
  intro h1 h2 h3
  sorry

end range_of_m_l650_650320


namespace general_formula_for_a_n_l650_650444

-- Given conditions
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
variable (h1 : ∀ n : ℕ, a n > 0)
variable (h2 : ∀ n : ℕ, 4 * S n = (a n - 1) * (a n + 3))

theorem general_formula_for_a_n :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end general_formula_for_a_n_l650_650444


namespace largest_prime_factor_of_1729_is_19_l650_650605

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650605


namespace friends_professions_l650_650235

theorem friends_professions
  (architect barista veterinarian guitarist : ℕ)
  (P : ℕ → Sort)
  (Andrey Boris Vyacheslav Gennady : P)
  (seats : P → ℕ)
  (h1 : ∀ {x y}, seats x = veterinarian → seats y = architect → (seats x = seats y + 1 ∨ seats x = seats y - 1))
  (h2 : ∀ {x y}, seats x = barista → seats y = Boris → seats y = seats x + 1)
  (h3 : seats Vyacheslav > seats Andrey ∧ seats Vyacheslav > seats Boris)
  (h4 : ∀ {x}, x = Andrey → ∃ y z, (seats y = seats x + 1 ∨ seats y = seats x - 1) ∧ (seats z = seats x + 1 ∨ seats z = seats x - 1))
  (h5 : ∀ {x y}, seats x = guitarist → seats y = barista → seats x ≠ seats y + 1 ∧ seats x ≠ seats y - 1)
  : (∀ x, (x = Gennady → seats x = barista)
    ∧ (x = Boris → seats x = architect)
    ∧ (x = Andrey → seats x = veterinarian)
    ∧ (x = Vyacheslav → seats x = guitarist)) :=
by
  sorry

end friends_professions_l650_650235


namespace complement_A_in_U_l650_650812

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 + x - 2 < 0}

theorem complement_A_in_U :
  (U \ A) = {-2, 1, 2} :=
by 
  -- proof will be done here
  sorry

end complement_A_in_U_l650_650812


namespace proposition_p_proposition_q_l650_650773

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 1

-- Prove the propositions p and q
theorem proposition_p : ∀ x : ℝ, f x > 0 :=
sorry

theorem proposition_q : ∃ x : ℝ, 0 < x ∧ g x = 0 :=
sorry

end proposition_p_proposition_q_l650_650773


namespace greatest_segment_length_l650_650103

-- Define the given conditions
variables (hexagon : Type) [convex_hexagon hexagon]
variables (radius : ℝ)
variables (circumcircle : circle)

-- Assume the given conditions
axiom hexagon_circumscribed : circumscribes hexagon circumcircle
axiom radii_eq_one : circumcircle.radius = 1

-- Define the problem to be proved
theorem greatest_segment_length (r : ℝ) : r ≤ 4 * real.sqrt 3 / 3 :=
sorry

end greatest_segment_length_l650_650103


namespace expected_number_of_first_sequence_l650_650151

-- Define the concept of harmonic number
def harmonic (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), (1 : ℚ) / (i + 1)

-- Define the problem statement
theorem expected_number_of_first_sequence (n : ℕ) (h : n = 100) : harmonic n = 5.187 := by
  sorry

end expected_number_of_first_sequence_l650_650151


namespace vivian_mail_in_august_l650_650068

-- Conditions
def april_mail : ℕ := 5
def may_mail : ℕ := 2 * april_mail
def june_mail : ℕ := 2 * may_mail
def july_mail : ℕ := 2 * june_mail

-- Question: Prove that Vivian will send 80 pieces of mail in August.
theorem vivian_mail_in_august : 2 * july_mail = 80 :=
by
  -- Sorry to skip the proof
  sorry

end vivian_mail_in_august_l650_650068


namespace part1_part2_l650_650033

/-
Part 1: Given the conditions of parabola and line intersection, prove the range of slope k of the line.
-/
theorem part1 (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  k > -2 + 2 * Real.sqrt 2 ∨ k < -2 - 2 * Real.sqrt 2 :=
  sorry

/-
Part 2: Given the conditions of locus of point Q on the line segment P1P2, prove the equation of the locus.
-/
theorem part2 (x y : ℝ) (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  2 * x - y + 1 = 0 ∧ (-Real.sqrt 2 - 1 < x ∧ x < Real.sqrt 2 - 1 ∧ x ≠ -1) :=
  sorry

end part1_part2_l650_650033


namespace seating_arrangement_l650_650240

-- Definitions of friends and their professions
inductive Friend
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq

-- Seating condition definitions
def seated_right (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = a ∧ seating (i + 1) = b)

def seated_between (a b c : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = b ∧ seating (i - 1) = a ∧ seating (i + 1) = c)

def neighbors (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, (seating i = a ∧ seating (i - 1) = b) ∨ (seating i = a ∧ seating (i + 1) = b))

-- The proposition to be proved
theorem seating_arrangement
  (seating : ℕ → Friend)
  (profession : Friend → Profession)
  (h1 : ∃ i, seating_between Profession.Veterinarian Profession.Architect Profession.Guitarist seating)
  (h2 : seated_right Profession.Barista Profession.Boris seating)
  (h3 : ∃ i, seating i = Friend.Vyacheslav ∧ (∃ j, seating j = Friend.Andrey ∧ j < i) ∧ (∃ k, seating k = Friend.Boris ∧ k < i))
  (h4 : ∀ a, seating a = Friend.Andrey → neighbors Friend.Andrey seating)
  (h5 : ∀ a, seating a = Profession.Guitarist → ∀ b, seating (b - 1) = Profession.Barista ∨ seating (b + 1) = Profession.Barista → false) :
  (profession Friend.Gennady = Profession.Barista ∧
   profession Friend.Boris = Profession.Architect ∧
   profession Friend.Andrey = Profession.Veterinarian ∧
   profession Friend.Vyacheslav = Profession.Guitarist) :=
begin
  sorry
end

end seating_arrangement_l650_650240


namespace derivative_f_at_zero_l650_650090

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 4 * x * (1 - |x|) else 0

theorem derivative_f_at_zero : HasDerivAt f 4 0 :=
by
  -- Proof omitted
  sorry

end derivative_f_at_zero_l650_650090


namespace inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l650_650398

noncomputable def inverse_of_half_pow (x : ℝ) : ℝ := Real.log x / Real.log (1 / 2)

theorem inverse_function_of_1_div_2_pow_eq_log_base_1_div_2 (x : ℝ) (hx : 0 < x) :
  inverse_of_half_pow x = Real.log x / Real.log (1 / 2) :=
by
  sorry

end inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l650_650398


namespace total_flowers_in_vases_l650_650975

theorem total_flowers_in_vases :
  let vase_count := 5
  let flowers_per_vase_4 := 5
  let flowers_per_vase_1 := 6
  let vases_with_5_flowers := 4
  let vases_with_6_flowers := 1
  (4 * 5 + 1 * 6 = 26) := by
  let total_flowers := 4 * 5 + 1 * 6
  show total_flowers = 26
  sorry

end total_flowers_in_vases_l650_650975


namespace complex_num_inequality_l650_650875

noncomputable def f (a : ℝ) (z : ℂ) : ℂ := z^2 - z + a

theorem complex_num_inequality (a : ℝ) (h : 0 < a ∧ a < 1) (z : ℂ) (hz : |z| ≥ 1) :
  ∃ z₀ : ℂ, |z₀| = 1 ∧ |f a z₀| ≤ |f a z| :=
sorry

end complex_num_inequality_l650_650875


namespace difference_sum_first_100_odds_evens_l650_650022

def sum_first_n_odds (n : ℕ) : ℕ :=
  n^2

def sum_first_n_evens (n : ℕ) : ℕ :=
  n * (n-1)

theorem difference_sum_first_100_odds_evens :
  sum_first_n_odds 100 - sum_first_n_evens 100 = 100 := by
  sorry

end difference_sum_first_100_odds_evens_l650_650022


namespace divide_5440_K_l650_650196

theorem divide_5440_K (a b c d : ℕ) 
  (h1 : 5440 = a + b + c + d)
  (h2 : 2 * b = 3 * a)
  (h3 : 3 * c = 5 * b)
  (h4 : 5 * d = 6 * c) : 
  a = 680 ∧ b = 1020 ∧ c = 1700 ∧ d = 2040 :=
by 
  sorry

end divide_5440_K_l650_650196


namespace non_touching_cells_l650_650470

theorem non_touching_cells (cells : Finset (ℕ × ℕ)) (h_cells : cells.card = 2000) :
  ∃ (S : Finset (ℕ × ℕ)), S.card ≥ 500 ∧ (∀ (c1 c2 : ℕ × ℕ), c1 ∈ S → c2 ∈ S → c1 ≠ c2 → ¬(touches c1 c2)) :=
by
  -- We need to define the touching relationship
  def touches (c1 c2 : ℕ × ℕ) : Prop :=
    (abs (c1.1 - c2.1) ≤ 1 ∧ abs (c1.2 - c2.2) ≤ 1)
  sorry

end non_touching_cells_l650_650470


namespace angle_EHG_of_parallelogram_l650_650909

theorem angle_EHG_of_parallelogram (EFGH : Parallelogram) (EFG_45 : ∠EFG = 45) (diagonal_EH : Diagonal E H) :
  ∠EHG = 67.5 := 
sorry

end angle_EHG_of_parallelogram_l650_650909


namespace count100DigitEvenNumbers_is_correct_l650_650336

noncomputable def count100DigitEvenNumbers : ℕ :=
  let validDigits : Finset ℕ := {0, 1, 3}
  let firstDigitChoices : ℕ := 2  -- Only 1 or 3
  let middleDigitsChoices : ℕ := 3 ^ 98  -- 3 choices for each of the 98 middle positions
  let lastDigitChoices : ℕ := 1  -- Only 0 (even number requirement)
  firstDigitChoices * middleDigitsChoices * lastDigitChoices

theorem count100DigitEvenNumbers_is_correct :
  count100DigitEvenNumbers = 2 * 3 ^ 98 := by
  sorry

end count100DigitEvenNumbers_is_correct_l650_650336


namespace divisors_of_m_squared_l650_650119

theorem divisors_of_m_squared {m : ℕ} (h₁ : ∀ d, d ∣ m → d = 1 ∨ d = m ∨ prime d) (h₂ : nat.divisors m = 4) :
  (nat.divisors (m ^ 2) = 7 ∨ nat.divisors (m ^ 2) = 9) :=
sorry

end divisors_of_m_squared_l650_650119


namespace pen_and_notebook_cost_l650_650884

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 17 * p + 5 * n = 200 ∧ p > n ∧ p + n = 16 := 
by
  sorry

end pen_and_notebook_cost_l650_650884


namespace largest_prime_factor_1729_l650_650578

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650578


namespace largest_prime_factor_1729_l650_650613

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650613


namespace profit_share_of_B_l650_650701

-- Defining the initial investments
def a : ℕ := 8000
def b : ℕ := 10000
def c : ℕ := 12000

-- Given difference between profit shares of A and C
def diff_AC : ℕ := 680

-- Define total profit P
noncomputable def P : ℕ := (diff_AC * 15) / 2

-- Calculate B's profit share
noncomputable def B_share : ℕ := (5 * P) / 15

-- The theorem stating B's profit share
theorem profit_share_of_B : B_share = 1700 :=
by sorry

end profit_share_of_B_l650_650701


namespace not_finite_many_symmetry_centers_l650_650178

variable (α : Type) [AddCommGroup α] [Module ℝ α]

def SymmetryCenter (O : α) (S : α → α) := ∀ x : α, S (S x) = x

theorem not_finite_many_symmetry_centers :
  (∀ (O1 O2 : α) (S : α → α), 
    SymmetryCenter O1 S → SymmetryCenter O2 S → 
    ∃ O3 : α, O3 ≠ O1 ∧ O3 ≠ O2 ∧ SymmetryCenter O3 S) 
  → ¬ (∃ (centers : Finset α) (S : α → α), Finset.card centers > 1 ∧ ∀ O ∈ centers, SymmetryCenter O S) :=
by
  intro h
  sorry

end not_finite_many_symmetry_centers_l650_650178


namespace largest_integer_base8_square_digits_l650_650440

theorem largest_integer_base8_square_digits (M : ℕ) (h : 8^3 ≤ M^2 ∧ M^2 < 8^4) : M = 31 ∧ nat.digits 8 M = [7, 3] :=
by
  sorry

end largest_integer_base8_square_digits_l650_650440


namespace AD_DB_rational_l650_650958

theorem AD_DB_rational {a b c : ℚ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  ∃ (AD DB : ℚ), AD + DB = c ∧ 
                  AD ^ 2 + (2 * sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) / c) / 2) ^ 2 = b ^ 2 ∧
                  DB ^ 2 + (2 * sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) / c) / 2) ^ 2 = a ^ 2 :=
  sorry

end AD_DB_rational_l650_650958


namespace expected_first_sequence_length_is_harmonic_100_l650_650129

open Real

-- Define the harmonic number
def harmonic_number (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (i + 1 : ℝ)

-- Define the expected length of the first sequence
def expected_first_sequence_length (n : ℕ) : ℝ :=
  harmonic_number n

-- Prove that the expected number of suitors in the first sequence is the 100th harmonic number
theorem expected_first_sequence_length_is_harmonic_100 :
  expected_first_sequence_length 100 = harmonic_number 100 :=
by
  sorry

end expected_first_sequence_length_is_harmonic_100_l650_650129


namespace shaded_area_proof_l650_650491

def equilateral_area (s : ℝ) : ℝ :=
  (Math.sqrt 3 / 4) * s^2

def area_side_1 := equilateral_area 1
def area_side_2 := equilateral_area 2
def area_side_3 := equilateral_area 3
def area_side_4 := equilateral_area 4

def shaded_area := area_side_4 - area_side_3 + area_side_2

theorem shaded_area_proof : shaded_area = 26 * area_side_1 :=
by
  sorry

end shaded_area_proof_l650_650491


namespace expected_first_sequence_length_is_harmonic_100_l650_650130

open Real

-- Define the harmonic number
def harmonic_number (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (i + 1 : ℝ)

-- Define the expected length of the first sequence
def expected_first_sequence_length (n : ℕ) : ℝ :=
  harmonic_number n

-- Prove that the expected number of suitors in the first sequence is the 100th harmonic number
theorem expected_first_sequence_length_is_harmonic_100 :
  expected_first_sequence_length 100 = harmonic_number 100 :=
by
  sorry

end expected_first_sequence_length_is_harmonic_100_l650_650130


namespace isosceles_not_necessarily_similar_with_45_angle_l650_650076

theorem isosceles_not_necessarily_similar_with_45_angle
  (T1 T2 : Triangle)
  (is_isosceles_T1 : T1.is_isosceles)
  (is_isosceles_T2 : T2.is_isosceles)
  (has_45_angle_T1 : ∃ A ∈ T1.angles, A = 45)
  (has_45_angle_T2 : ∃ A ∈ T2.angles, A = 45) :
  ¬ (T1 ≅ T2) := 
sorry

end isosceles_not_necessarily_similar_with_45_angle_l650_650076


namespace largest_distinct_digit_multiple_of_99_l650_650953

theorem largest_distinct_digit_multiple_of_99 : ∃ n : ℕ, (n % 99 = 0) ∧ (∀ i j, i ≠ j → ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0 → d = n.digits 10.get i) ∧ ∀ m : ℕ, (m % 99 = 0 ∧ (∀ i j, i ≠ j → ∀ d : ℕ, d ∈ m.digits 10 → d ≠ 0 → d = m.digits 10.get i)) → m ≤ 9876524130 := 
sorry

end largest_distinct_digit_multiple_of_99_l650_650953


namespace expected_first_sequence_100_l650_650143

noncomputable def expected_first_sequence (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1).filter (λ k, k > 0), (1 : ℝ) / k

theorem expected_first_sequence_100 : expected_first_sequence 100 = 
    ∑ k in (finset.range 101).filter (λ k, k > 0), (1 : ℝ) / k :=
by
  -- The proof would involve showing this sum represents the harmonic number H_100
  sorry

end expected_first_sequence_100_l650_650143


namespace Toms_swimming_speed_is_2_l650_650517

theorem Toms_swimming_speed_is_2
  (S : ℝ)
  (h1 : 2 * S + 4 * S = 12) :
  S = 2 :=
by
  sorry

end Toms_swimming_speed_is_2_l650_650517


namespace hexagon_area_l650_650739

noncomputable def area_hexagon_ABCFEH : ℝ := 4 * real.sqrt 3

def equilateral_triangle_PQR (P Q R : Type) (d : ℝ) : Prop :=
  d = 2 ∧ ∀ (A : P Q R), side_length A = d

def squares_on_triangle (P Q R : Type) (A B C F E H : Type) (d : ℝ) : Prop :=
  d = 2 ∧ ∀ (S : A B C F E H), side_length S = d

theorem hexagon_area (P Q R A B C F E H : Type) (d : ℝ) :
  equilateral_triangle_PQR P Q R d →
  squares_on_triangle P Q R A B C F E H d →
  area_hexagon_ABCFEH = 4 * real.sqrt 3 :=
by
  intros h1 h2 
  sorry

end hexagon_area_l650_650739


namespace cash_realized_on_selling_stock_l650_650943

theorem cash_realized_on_selling_stock
  (total_amount_before_brokerage : ℝ)
  (brokerage_rate : ℝ)
  (h_total_amount : total_amount_before_brokerage = 107)
  (h_brokerage_rate : brokerage_rate = 0.25)
  : total_amount_before_brokerage - (brokerage_rate / 100 * total_amount_before_brokerage).round(2) = 106.73 :=
by
  sorry

end cash_realized_on_selling_stock_l650_650943


namespace largest_prime_factor_of_1729_l650_650595

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650595


namespace number_of_trains_l650_650976

/-- Given the conditions on a train station with trains, carriages, and wheels,
    we prove that the number of trains can be determined -/
theorem number_of_trains (c r w T_w : ℕ) (hc : c = 4) (hr : r = 3) (hw : w = 5) (hT_w : T_w = 240) :
    T_w / (c * r * w) = 4 :=
by
  -- Utilize hypotheses to simplify the number of trains calculation
  rw [hc, hr, hw, hT_w]
  sorry

end number_of_trains_l650_650976


namespace second_number_is_72_l650_650085

theorem second_number_is_72 
  (sum_eq_264 : ∀ (x : ℝ), 2 * x + x + (2 / 3) * x = 264) 
  (first_eq_2_second : ∀ (x : ℝ), first = 2 * x)
  (third_eq_1_3_first : ∀ (first : ℝ), third = 1 / 3 * first) :
  second = 72 :=
by
  sorry

end second_number_is_72_l650_650085


namespace standard_deviation_of_entire_set_l650_650735

def group1_mean : ℝ := 50
def group1_variance : ℝ := 33
def group2_mean : ℝ := 40
def group2_variance : ℝ := 45
def n : ℕ := 10

theorem standard_deviation_of_entire_set : 
  let group1_sd := (real.sqrt group1_variance),
      group2_sd := (real.sqrt group2_variance) in
  let total_variance := 
    (n * group1_variance + n * group2_variance) / (2 * n) +
    (n * n) / ((2 * n) * (2 * n)) * (group1_mean - group2_mean) ^ 2 in
  real.sqrt total_variance = 8 :=
by {
  let group1_sd := (real.sqrt group1_variance),
  let group2_sd := (real.sqrt group2_variance),
  let total_variance := 
    (n * group1_variance + n * group2_variance) / (2 * n) +
    (n * n) / ((2 * n) * (2 * n)) * (group1_mean - group2_mean) ^ 2,
  sorry
}

end standard_deviation_of_entire_set_l650_650735


namespace range_of_a_max_value_in_interval_inequality_natural_numbers_l650_650314

open Real

def f (x : ℝ) (a : ℝ) : ℝ :=
  ln x + (1 - x) / (a * x)

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 1 ≤ x → 0 ≤ (f' x)) ↔ 1 ≤ a :=
sorry

theorem max_value_in_interval (a : ℝ) (h : 0 < a) :
  max (f 1 a) (f 2 a) = 
  if 0 < a ∧ a ≤ 1 / (2 * ln 2) then f 1 a 
  else f 2 a :=
sorry

theorem inequality_natural_numbers (n : ℕ) (hn : 1 < n) :
  ∑ i in Finset.range n, 1 / (i + 1) > ln n :=
sorry

end range_of_a_max_value_in_interval_inequality_natural_numbers_l650_650314


namespace triangle_condition_l650_650423

theorem triangle_condition (A B C D C0 : Type) [triangle A B C] 
  (mid_AB : midpoint C0 A B) (on_BC : D ∈ line B C) 
  (angle_eq : ∠ CAD = ∠ CBA) (AC_lt_BC : AC < BC) :
  AC = 1 / 2 * BC → parallel (line D C0) (angle_bisector C) := 
sorry

end triangle_condition_l650_650423


namespace perimeter_triangle_PQR_is_24_l650_650414

noncomputable def perimeter_triangle_PQR (QR PR : ℝ) : ℝ :=
  let PQ := Real.sqrt (QR^2 + PR^2)
  PQ + QR + PR

theorem perimeter_triangle_PQR_is_24 :
  perimeter_triangle_PQR 8 6 = 24 := by
  sorry

end perimeter_triangle_PQR_is_24_l650_650414


namespace domain_of_fx_l650_650494

noncomputable def f : ℝ → ℝ :=
  λ x, if 0 < x ∧ x ≤ 3 then 2 * x - x^2 else if -2 ≤ x ∧ x ≤ 0 then x^2 + 6 * x else 0

theorem domain_of_fx : 
  {x : ℝ | (0 < x ∧ x ≤ 3) ∨ (-2 ≤ x ∧ x ≤ 0)} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := sorry

end domain_of_fx_l650_650494


namespace largest_prime_factor_1729_l650_650583

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650583


namespace part_a_part_b_l650_650435

variables {α : Type*} [linear_ordered_field α] [has_sqrt α]

-- Define the points and relevant geometric relations
structure Point := (x : α) (y : α)
def is_parallel (A B C D : Point) : Prop := (B.y - A.y) * (D.x - C.x) = (D.y - C.y) * (B.x - A.x)
def is_perpendicular (A B C D : Point) : Prop := (B.y - A.y) * (D.y - C.y) + (B.x - A.x) * (D.x - C.x) = 0
def midpoint (A B : Point) : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

variables (A B C D M N O E : Point)
variable (h_parallel : is_parallel A B C D)
variable (h_perpendicular : is_perpendicular A C B D)
variable (hO : O.x = (A.x + C.x) / 2 ∧ O.y = (A.y + C.y) / 2)
variable (hM : M.x = 2 * O.x - A.x ∧ M.y = 2 * O.y - A.y)
variable (hN : N.x = 2 * O.x - B.x ∧ N.y = 2 * O.y - B.y)
variable (h_mid : E = midpoint M N)
variable (h_angle_ANC_90 : (N.y - C.y) * (N.x - C.x) + (A.y - N.y) * (A.x - N.x) = 0)
variable (h_angle_BMD_90 : (M.y - D.y) * (M.x - D.x) + (B.y - M.y) * (B.x - M.x) = 0)

-- Part (a): Prove that ΔOMN ~ ΔOBA
theorem part_a : ∀ h_parallel h_perpendicular hO hM hN h_mid h_angle_ANC_90 h_angle_BMD_90, 
  ΔOMN ∼ ΔOBA :=
sorry

-- Part (b): Prove that OE ⊥ AB
theorem part_b : ∀ h_parallel h_perpendicular hO hM hN h_mid h_angle_ANC_90 h_angle_BMD_90, 
  is_perpendicular O E A B :=
sorry

end part_a_part_b_l650_650435


namespace symmetry_condition_l650_650227

variable {α β : Type} [Preorder α] [LinearOrder β]

noncomputable def horizontalShift (f : α → β) (a : α) : α → β :=
  fun x => f (x - a)

noncomputable def reflectYAxis (g : α → β) : α → β :=
  fun x => g (-x)

theorem symmetry_condition (f : ℝ → ℝ) :
  (∀ x, horizontalShift f 1 x = f (x - 1)) →
  (∀ x, reflectYAxis (horizontalShift f 1) x = f (-(x - 1))) →
  ∀ x, horizontalShift f 1 x = reflectYAxis (horizontalShift f 1) (1 - x) :=
by
  intros h_shift h_reflect x
  sorry

end symmetry_condition_l650_650227


namespace largest_prime_factor_of_1729_l650_650556

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650556


namespace artemCanArrangeCoins_l650_650168

structure Coin :=
  (diameter : ℕ)
  (mass : ℕ)
  (year : ℕ)

def coins : List Coin := sorry -- This represents the list of 27 unique coins.

def canArrange (coins : List Coin) : Prop :=
  ∃ arrangement : Array (Array (Array Coin)), -- A 3x3x3 array.
    -- Condition 1: Each coin is lighter than the coin directly below it.
    (∀ i j k, i < 2 → j < 3 → k < 3 → arrangement[i][j][k].mass < arrangement[i + 1][j][k].mass) ∧
    -- Condition 2: Each coin is smaller than the coin to its right.
    (∀ i j k, i < 3 → j < 2 → k < 3 → arrangement[i][j][k].diameter < arrangement[i][j + 1][k].diameter) ∧
    -- Condition 3: Each coin is older than the coin directly in front of it.
    (∀ i j k, i < 3 → j < 3 → k < 2 → arrangement[i][j][k].year > arrangement[i][j][k + 1].year)

theorem artemCanArrangeCoins : canArrange coins :=
  sorry

end artemCanArrangeCoins_l650_650168


namespace largest_prime_factor_of_1729_l650_650598

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650598


namespace circle_equation_a_value_l650_650397

theorem circle_equation_a_value (a : ℝ) : (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
sorry

end circle_equation_a_value_l650_650397


namespace find_x_2187_l650_650794

theorem find_x_2187 (x : ℂ) (h : x - 1/x = complex.I * real.sqrt 3) : x^2187 - 1/(x^2187) = 0 :=
sorry

end find_x_2187_l650_650794


namespace largest_prime_factor_1729_l650_650535

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650535


namespace ratio_qp_l650_650028

theorem ratio_qp (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 6 → 
    P / (x + 3) + Q / (x * (x - 6)) = (x^2 - 4 * x + 15) / (x * (x + 3) * (x - 6))) : 
  Q / P = 5 := 
sorry

end ratio_qp_l650_650028


namespace solve_profession_arrangement_l650_650247

inductive Profession
| architect
| barista
| veterinarian
| guitarist

inductive Person
| Andrey
| Boris
| Vyacheslav
| Gennady

open Profession
open Person

structure SeatingArrangement :=
(seat1 : Person)
(seat2 : Person)
(seat3 : Person)
(seat4 : Person)

structure Assignment :=
(profession : Person → Profession)

def correct_arrangement_and_profession (sa : SeatingArrangement) (asgmt : Assignment) : Prop :=
(sa.seat1 ≠ sa.seat2 ∧ sa.seat1 ≠ sa.seat3 ∧ sa.seat1 ≠ sa.seat4 ∧
 sa.seat2 ≠ sa.seat3 ∧ sa.seat2 ≠ sa.seat4 ∧ sa.seat3 ≠ sa.seat4) ∧
((asgmt profession sa.seat1 = barista) ∧ (asgmt profession sa.seat2 = architect) ∧ 
 (asgmt profession sa.seat3 = veterinarian) ∧ (asgmt profession sa.seat4 = guitarist)) ∧
(sa.seat3 = Andrey) ∧ 
(sa.seat2 = Boris) ∧ 
(sa.seat4 = Vyacheslav)

theorem solve_profession_arrangement : 
  ∃ (sa : SeatingArrangement) (asgmt : Assignment), correct_arrangement_and_profession sa asgmt :=
sorry

end solve_profession_arrangement_l650_650247


namespace units_digit_of_M_M8_l650_650012

-- Definitions for the Lucas-like sequence M_n
def M : ℕ → ℕ
| 0       := 3
| 1       := 2
| (n + 2) := 2 * (M (n + 1)) + M n

-- A function to extract the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- The main theorem statement
theorem units_digit_of_M_M8 : units_digit (M (M 8)) = 6 := 
sorry

end units_digit_of_M_M8_l650_650012


namespace largest_prime_factor_of_1729_l650_650642

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650642


namespace sqrt_factorial_div_l650_650221

theorem sqrt_factorial_div {n : ℕ} (hn : n = 10) (d : ℕ) (hd : d = 210) :
  sqrt (↑((nat.factorial n) / d)) = 24 * sqrt 30 :=
by
  sorry

end sqrt_factorial_div_l650_650221


namespace M_subset_P_l650_650461

universe u

-- Definitions of the sets
def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

-- Proof statement
theorem M_subset_P : M ⊆ P := by
  sorry

end M_subset_P_l650_650461


namespace solve_inequality_l650_650042

theorem solve_inequality : { x : ℝ | x^2 - x - 6 < 0 } = set.Ioo (-2 : ℝ) 3 := 
by {
  sorry
}

end solve_inequality_l650_650042


namespace find_positive_integer_n_l650_650214

theorem find_positive_integer_n (n : ℕ) : 
  let D (k : ℕ) := (k * (k - 3)) / 2 in
  ∀ (k1 k2 : ℕ), k1 = 3 * n + 2 →
                    k2 = 5 * n - 2 →
                      D k1 = 385 * D k2 / 1000 →
                        n = 26 :=
by
  intros
  -- Definitions
  let D := λ k, (k * (k - 3)) / 2
  -- Given conditions
  have h1 : k1 = 3 * n + 2 := by assumption
  have h2 : k2 = 5 * n - 2 := by assumption
  have h3 : D k1 = 385 * D k2 / 1000 := by assumption
  -- Conclusion
  sorry

end find_positive_integer_n_l650_650214


namespace solve_x_sq_plus_y_sq_l650_650820

theorem solve_x_sq_plus_y_sq (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = 2) : x^2 + y^2 = 5 :=
by
  sorry

end solve_x_sq_plus_y_sq_l650_650820


namespace f_monotonic_intervals_l650_650312

noncomputable def f (x a b : ℝ) : ℝ := (x - 1) ^ 3 - a * x - b

theorem f_monotonic_intervals (a b : ℝ) :
  if a ≤ 0 then (∀ x y : ℝ, x < y → f x a b < f y a b) else
  (∀ x : ℝ, x < (1 - real.sqrt (a / 3)) ∨ (1 + real.sqrt (a / 3)) < x → f x a b > f (x + 1) a b) ∧
  (∀ x y : ℝ, (1 - real.sqrt (a / 3)) < x ∧ x < (1 + real.sqrt (a / 3)) ∧ x < y → f x a b > f y a b) :=
sorry

end f_monotonic_intervals_l650_650312


namespace find_x_l650_650224

theorem find_x : 
  (∃ x : ℝ, 
    2.5 * ((3.6 * 0.48 * 2.5) / (0.12 * x * 0.5)) = 2000.0000000000002) → 
  x = 0.225 :=
by
  sorry

end find_x_l650_650224


namespace road_connections_possible_l650_650877

-- Definitions based on the conditions
def positive_integer (n : ℕ) : Prop :=
  n > 0

def num_cities (n : ℕ) : ℕ :=
  2018 * n + 1

def city_has_n_neighbors_at_distance_i (n : ℕ) (i : ℕ) (C : ℕ) : Prop :=
  ∃ city_set : finset ℕ, city_set.card = n ∧ ∀ c ∈ city_set, (distance C c = i)

-- Main theorem statement
theorem road_connections_possible (n : ℕ)
    (h_pos : positive_integer n)
    (h_cities : ∀ C, ∀ i, 1 ≤ i → i ≤ 2018 → city_has_n_neighbors_at_distance_i n i C) :
    ∃ k : ℕ, n = 2 * k :=
sorry

end road_connections_possible_l650_650877


namespace find_value_of_some_number_l650_650503

theorem find_value_of_some_number :
  ∃ (some_number : ℝ), (3.242 * 12) / some_number = 0.038903999999999994 ∧ some_number ≈ 1000 :=
begin
  sorry
end

end find_value_of_some_number_l650_650503


namespace angle_between_vectors_l650_650822

open Real EuclideanSpace

variables {n : Type*} [EuclideanAffineSpace ℝ n] {a b : n}

-- Conditions
def non_zero (x : n) : Prop := x ≠ 0

def perpendicular (x y : n) : Prop := inner x y = 0

-- Problem
theorem angle_between_vectors (h1 : non_zero a) (h2 : non_zero b)
  (h3 : perpendicular (a - 2 • b) a)
  (h4 : perpendicular (b - 2 • a) b) :
  ∃ θ, θ = real.pi / 3 :=
sorry

end angle_between_vectors_l650_650822


namespace solve_profession_arrangement_l650_650249

inductive Profession
| architect
| barista
| veterinarian
| guitarist

inductive Person
| Andrey
| Boris
| Vyacheslav
| Gennady

open Profession
open Person

structure SeatingArrangement :=
(seat1 : Person)
(seat2 : Person)
(seat3 : Person)
(seat4 : Person)

structure Assignment :=
(profession : Person → Profession)

def correct_arrangement_and_profession (sa : SeatingArrangement) (asgmt : Assignment) : Prop :=
(sa.seat1 ≠ sa.seat2 ∧ sa.seat1 ≠ sa.seat3 ∧ sa.seat1 ≠ sa.seat4 ∧
 sa.seat2 ≠ sa.seat3 ∧ sa.seat2 ≠ sa.seat4 ∧ sa.seat3 ≠ sa.seat4) ∧
((asgmt profession sa.seat1 = barista) ∧ (asgmt profession sa.seat2 = architect) ∧ 
 (asgmt profession sa.seat3 = veterinarian) ∧ (asgmt profession sa.seat4 = guitarist)) ∧
(sa.seat3 = Andrey) ∧ 
(sa.seat2 = Boris) ∧ 
(sa.seat4 = Vyacheslav)

theorem solve_profession_arrangement : 
  ∃ (sa : SeatingArrangement) (asgmt : Assignment), correct_arrangement_and_profession sa asgmt :=
sorry

end solve_profession_arrangement_l650_650249


namespace area_at_stage_8_l650_650386

theorem area_at_stage_8 
  (side_length : ℕ)
  (stage : ℕ)
  (num_squares : ℕ)
  (square_area : ℕ) 
  (total_area : ℕ) 
  (h1 : side_length = 4) 
  (h2 : stage = 8) 
  (h3 : num_squares = stage) 
  (h4 : square_area = side_length * side_length) 
  (h5 : total_area = num_squares * square_area) :
  total_area = 128 :=
sorry

end area_at_stage_8_l650_650386


namespace option_b_correct_l650_650702

theorem option_b_correct (a b c : ℝ) (hc : c ≠ 0) (h : a * c^2 > b * c^2) : a > b :=
sorry

end option_b_correct_l650_650702


namespace num_100_digit_even_numbers_l650_650333

theorem num_100_digit_even_numbers : 
  let digit_set := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let valid_number (digits : list ℕ) := 
    digits.length = 100 ∧ digits.head ∈ {1, 3} ∧ 
    digits.last ∈ {0} ∧ 
    ∀ d ∈ digits.tail.init, d ∈ digit_set
  (∃ (m : ℕ), valid_number (m.digits 10)) = 2 * 3^98 := 
sorry

end num_100_digit_even_numbers_l650_650333


namespace line_equation_l650_650081

theorem line_equation (m n : ℝ) (p : ℝ) (h : p = 3) :
  ∃ b : ℝ, ∀ x y : ℝ, (y = n + 21) → (x = m + 3) → y = 7 * x + b ∧ b = n - 7 * m :=
by sorry

end line_equation_l650_650081


namespace length_of_EG_ratio_of_AB_BC_l650_650169

theorem length_of_EG (EH EF EG : ℝ) (hEH : EH = 3) (hEF : EF = 4) : EG = 5 := 
by sorry

theorem ratio_of_AB_BC (EH EF : ℝ) (h_ratio : EH / EF = 1 / 2) : (AB BC : ℝ) := 
by 
  have h_AB_BC := (5 / 4) -- the conclusion result for the ratio of AB to BC.
  sorry


end length_of_EG_ratio_of_AB_BC_l650_650169


namespace largest_prime_factor_1729_l650_650579

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650579


namespace clear_correlation_l650_650077

namespace CorrelationProof

variables (a V : ℝ) (θ : ℝ) (sinθ : ℝ) (land_area yield total_yield constant_productivity sunlight rice_yield : ℝ)

-- Conditions
def cube_volume : Prop := V = a ^ 3
def sine_function : Prop := sinθ = sin θ
def yield_linear_relation : Prop := total_yield = land_area * constant_productivity
def rice_yield_dependence : Prop := rice_yield = function_of_sunlight sunlight

-- Proof Problem Statement
theorem clear_correlation : 
  (cube_volume ∧ sine_function ∧ yield_linear_relation ∧ rice_yield_dependence) → 
  (correlated_a C := yield_linear_relation) := sorry

end CorrelationProof

end clear_correlation_l650_650077


namespace vegetarian_count_l650_650837

variables (v_only v_nboth vegan pesc nvboth : ℕ)
variables (hv_only : v_only = 13) (hv_nboth : v_nboth = 8)
          (hvegan_tot : vegan = 5) (hvegan_v : vveg1 = 3)
          (hpesc_tot : pesc = 4) (hpesc_vnboth : nvboth = 2)

theorem vegetarian_count (total_veg : ℕ) 
  (H_total : total_veg = v_only + v_nboth + (vegan - vveg1)) :
  total_veg = 23 :=
sorry

end vegetarian_count_l650_650837


namespace remainder_of_444_power_444_mod_13_l650_650994

theorem remainder_of_444_power_444_mod_13 :
  (444^444) % 13 = 1 :=
by
  have h1: 444 % 13 = 3 := by norm_num
  have h2: ∀ n, 3^3^n % 13 = 1 := by
    intros n; induction n with d hd
    · rw [pow_zero, pow_zero]; norm_num
    exact mod_eq_zero_of_dvd (dvd_trans (mod_dvd (pow_succ_pow 3 d (by norm_num))) hd)
  rw [← pow_mul, h1, h2 148]
  sorry

end remainder_of_444_power_444_mod_13_l650_650994


namespace value_divided_by_sqrt_1936_l650_650971

theorem value_divided_by_sqrt_1936 (x : ℝ) (h1 : real.sqrt 1936 / x = 4) : x = 11 :=
by
  sorry

end value_divided_by_sqrt_1936_l650_650971


namespace partOneCorrectProbability_partTwoCorrectProbability_l650_650926

noncomputable def teachers_same_gender_probability (mA fA mB fB : ℕ) : ℚ :=
  let total_outcomes := mA * mB + mA * fB + fA * mB + fA * fB
  let same_gender := mA * mB + fA * fB
  same_gender / total_outcomes

noncomputable def teachers_same_school_probability (SA SB : ℕ) : ℚ :=
  let total_teachers := SA + SB
  let total_outcomes := (total_teachers * (total_teachers - 1)) / 2
  let same_school := (SA * (SA - 1)) / 2 + (SB * (SB - 1)) / 2
  same_school / total_outcomes

theorem partOneCorrectProbability : teachers_same_gender_probability 2 1 1 2 = 4 / 9 := by
  sorry

theorem partTwoCorrectProbability : teachers_same_school_probability 3 3 = 2 / 5 := by
  sorry

end partOneCorrectProbability_partTwoCorrectProbability_l650_650926


namespace problem_statement_l650_650488

noncomputable def common_sum (arrangement : matrix (fin 4) (fin 4) ℤ) : ℤ :=
  let rows_sum := ∀ i : fin 4, finset.univ.sum (λ j, arrangement i j)
  let columns_sum := ∀ j : fin 4, finset.univ.sum (λ i, arrangement i j)
  let diag1_sum := (finset.univ.sum (λ k : fin 4, arrangement k k))
  let diag2_sum := (finset.univ.sum (λ k : fin 4, arrangement k (3 - k)))
  in if (rows_sum = columns_sum ∧ rows_sum = diag1_sum ∧ rows_sum = diag2_sum) then rows_sum 0 else 0

theorem problem_statement : ∀ (arrangement : matrix (fin 4) (fin 4) ℤ), 
  (∀ i j, arrangement i j ∈ finset.range 16 ∧ arrangement i j + -8 ≤ 7) →
  (∀ i, finset.univ.sum (λ j, arrangement i j) = finset.univ.sum (λ k, arrangement k (i : fin 4))) →
  (∀ i, finset.univ.sum (λ j, arrangement j i) = finset.univ.sum (λ k, arrangement k (i : fin 4))) →
  (finset.univ.sum (λ k, arrangement k k) = finset.univ.sum (λ k, arrangement k (3 - k))) →
  common_sum arrangement = -2 :=
begin
  sorry
end

end problem_statement_l650_650488


namespace largest_prime_factor_of_1729_l650_650565

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650565


namespace find_smallest_x_l650_650995

def smallest_x_divisible (y : ℕ) : ℕ :=
  if y = 11 then 257 else 0

theorem find_smallest_x : 
  smallest_x_divisible 11 = 257 ∧ 
  ∃ k : ℕ, 264 * k - 7 = 257 :=
by
  sorry

end find_smallest_x_l650_650995


namespace bases_with_final_digit_one_l650_650766

theorem bases_with_final_digit_one :
  { b : ℕ | 3 ≤ b ∧ b ≤ 10 ∧ 624 % b = 0 }.card = 4 :=
by
  sorry

end bases_with_final_digit_one_l650_650766


namespace sequence_periodic_100th_term_sequence_sum_100th_term_l650_650778

variable (a b : ℝ)

def x : ℕ → ℝ
| 1        := a
| 2        := b
| (n + 1)  := if n < 2 then 0 else x (n - 1) - x (n - 2)

def S (n : ℕ) : ℝ := (Finset.range n).sum (fun m => x a b (m + 1))

theorem sequence_periodic_100th_term :
  x a b 100 = -a :=
sorry

theorem sequence_sum_100th_term :
  S a b 100 = 2 * b - a :=
sorry

end sequence_periodic_100th_term_sequence_sum_100th_term_l650_650778


namespace locus_of_points_nonexistent_l650_650839

theorem locus_of_points_nonexistent :
  ¬ ∃ (P : ℝ × ℝ), let x := P.1, y := P.2 in
  real.sqrt ((x + 1)^2 + y^2) + real.sqrt ((x - 1)^2 + y^2) = 1 :=
by
  sorry

end locus_of_points_nonexistent_l650_650839


namespace seating_profession_solution_l650_650258

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l650_650258


namespace find_f_and_g_minimum_l650_650806

-- Given conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b / x

axiom f_at_2 : f 2 a b = 5 / 2
axiom f_at_neg1 : f (-1) a b = -2

def g (x m : ℝ) (a b : ℝ) : ℝ := x^2 + (1 / x^2) + 2 * m * (f x a b)

-- Proof of the requirements
theorem find_f_and_g_minimum (a b m : ℝ) :
  (∃ a b, (∀ x, f x a b = x + 1 / x)) ∧
  (∀ x ∈ set.Icc 1 (2 : ℝ), g x m a b = 
    if m ≥ -2 then 4 * m + 2 
    else if -5 / 2 < m ∧ m < -2 then -m^2 - 2 
    else 5 * m + 17 / 4) :=
sorry

end find_f_and_g_minimum_l650_650806


namespace process_continues_indefinitely_l650_650745

theorem process_continues_indefinitely {t : ℝ} (ht : 1 < t ∧ t < 2) (h_poly : t^3 = t^2 + t + 1) :
  ∀ (sticks : ℝ) (a b c : ℝ),
  (sticks = { t^3, t^2, t }) →
  (¬ (a + b > c ∧ a + c > b ∧ b + c > a)) →
  (∃ new_a new_b new_c : ℝ,
    new_a + new_b + new_c = a + b + c ∧
    (new_c = a + b ∨ new_a = b + c ∨ new_b = a + c) ∧ 
    (¬ (new_a + new_b > new_c ∧ new_a + new_c > new_b ∧ new_b + new_c > new_a)) 
    ∧ ... -- and this can continue indefinitely)
  sorry

end process_continues_indefinitely_l650_650745


namespace impossible_intersection_l650_650810

def setA : set ℝ := {x | x ≥ 1}
def setB : set ℝ := {x | 1 - real.sqrt 2 < x ∧ x < 1 + real.sqrt 2}

theorem impossible_intersection : setA ∩ setB = ∅ :=
by
  sorry

end impossible_intersection_l650_650810


namespace largest_prime_factor_of_1729_l650_650639

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650639


namespace binom_fraction_computation_l650_650378

noncomputable def binom (x : ℝ) (k : ℕ) : ℝ := 
  if k = 0 then 1 else (List.range k).foldl (λ acc i, acc * (x - i) / (i + 1)) 1

theorem binom_fraction_computation : 
  (binom (1 / 2) 2014 * 4 ^ 2014) / (binom 4028 2014) = - (1 : ℝ) / 4027 := 
by 
  sorry

end binom_fraction_computation_l650_650378


namespace largest_prime_factor_of_1729_l650_650546

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650546


namespace algebra_expression_correct_l650_650306

theorem algebra_expression_correct {x y : ℤ} (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
  sorry

end algebra_expression_correct_l650_650306


namespace MaryNeedingToGrow_l650_650889

/-- Mary's height is 2/3 of her brother's height. --/
def MarysHeight (brothersHeight : ℕ) : ℕ := (2 * brothersHeight) / 3

/-- Mary needs to grow a certain number of centimeters to meet the minimum height
    requirement for riding Kingda Ka. --/
def RequiredGrowth (minimumHeight maryHeight : ℕ) : ℕ := minimumHeight - maryHeight

theorem MaryNeedingToGrow 
  (minimumHeight : ℕ := 140)
  (brothersHeight : ℕ := 180)
  (brothersHeightIs180 : brothersHeight = 180 := rfl)
  (heightRatio : ℕ → ℕ := MarysHeight)
  (maryHeight : ℕ := heightRatio brothersHeight)
  (maryHeightProof : maryHeight = 120 := by simp [MarysHeight, brothersHeightIs180])
  (requiredGrowth : ℕ := RequiredGrowth minimumHeight maryHeight) :
  requiredGrowth = 20 :=
by
  unfold RequiredGrowth MarysHeight
  rw [maryHeightProof]
  exact rfl

end MaryNeedingToGrow_l650_650889


namespace limit_ln_cos_ratio_l650_650183

open Real

theorem limit_ln_cos_ratio :
  tendsto (λ x, (ln (cos (2 * x)) / ln (cos (4 * x)))) (nhds π) (nhds (1 / 4)) := 
sorry

end limit_ln_cos_ratio_l650_650183


namespace find_fourth_number_l650_650942

-- Define the average and the known numbers in the list
def average : ℝ := 223
def num_values : ℕ := 6
def known_numbers : List ℝ := [55, 48, 507, 684, 42]

-- Define the fourth number in the list
def fourth_number (x : ℝ) : ℝ :=
  let total_sum := average * (num_values : ℝ)
  let sum_known := known_numbers.sum
  total_sum - sum_known

-- The statement to prove
theorem find_fourth_number (x : ℝ) (h : (55 + 48 + 507 + x + 684 + 42) / num_values = average) :
  x = 2 :=
by
  have total_sum : ℝ := average * (num_values : ℝ)
  have sum_known : ℝ := known_numbers.sum
  have eq_sum : 55 + 48 + 507 + 684 + 42 = sum_known := by rfl
  calc
    x = total_sum - sum_known : sorry
    ... = 2 : by norm_num

end find_fourth_number_l650_650942


namespace largest_prime_factor_of_1729_l650_650567

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650567


namespace value_of_expression_l650_650401

variable {α : ℝ}

theorem value_of_expression (h : cos α = tan α) : (1 / sin α) + (cos α)^4 = 2 := 
sorry

end value_of_expression_l650_650401


namespace problem_statement_l650_650302

def f : ℝ → ℝ :=
  sorry

lemma even_function (x : ℝ) : f (-x) = f x :=
  sorry

lemma periodicity (x : ℝ) (hx : 0 ≤ x) : f (x + 2) = -f x :=
  sorry

lemma value_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 2) : f x = Real.log (x + 1) :=
  sorry

theorem problem_statement : f (-2001) + f 2012 = 1 :=
  sorry

end problem_statement_l650_650302


namespace playground_area_l650_650034

theorem playground_area (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 100)
  (h_length : l = 3 * w) : l * w = 468.75 :=
by
  sorry

end playground_area_l650_650034


namespace mr_brown_yield_l650_650892

def length_steps := 18
def width_steps := 25
def step_length_feet := 3
def yield_per_sqft := 0.75
def length_feet := length_steps * step_length_feet
def width_feet := width_steps * step_length_feet
def area_sqft := length_feet * width_feet
def total_yield := area_sqft * yield_per_sqft

theorem mr_brown_yield : total_yield = 3037.5 := by
  sorry

end mr_brown_yield_l650_650892


namespace num_factors_180_l650_650348

theorem num_factors_180 : 
  ∃ (n : ℕ), n = 180 ∧ prime_factorization n = [(2, 2), (3, 2), (5, 1)] ∧ positive_factors n = 18 :=
by
  sorry

end num_factors_180_l650_650348


namespace determine_lambdas_l650_650867

namespace MathProofProblem

def vec (R : Type) := R × R × R

variables {R : Type} [Field R] 
noncomputable def vect_m : vec R := (1, 0, 0)
noncomputable def vect_j : vec R := (0, 1, 0)
noncomputable def vect_k : vec R := (0, 0, 1)

def a1 (m j k : vec R) := (2 * m.1 - j.1 + k.1, 2 * m.2 - j.2 + k.2, 2 * m.3 - j.3 + k.3)
def a2 (m j k : vec R) := (m.1 + 3 * j.1 - 2 * k.1, m.2 + 3 * j.2 - 2 * k.2, m.3 + 3 * j.3 - 2 * k.3)
def a3 (m j k : vec R) := (-2 * m.1 + j.1 - 3 * k.1, -2 * m.2 + j.2 - 3 * k.2, -2 * m.3 + j.3 - 3 * k.3)
def a4 (m j k : vec R) := (3 * m.1 + 2 * j.1 + 5 * k.1, 3 * m.2 + 2 * j.2 + 5 * k.2, 3 * m.3 + 2 * j.3 + 5 * k.3)

theorem determine_lambdas : 
  ∃ (λ μ ν : R), 
    a4 vect_m vect_j vect_k = (λ * a1 vect_m vect_j vect_k).1 + (μ * a2 vect_m vect_j vect_k).1 + (ν * a3 vect_m vect_j vect_k).1 ∧
    a4 vect_m vect_j vect_k = (λ * a1 vect_m vect_j vect_k).2 + (μ * a2 vect_m vect_j vect_k).2 + (ν * a3 vect_m vect_j vect_k).2 ∧
    a4 vect_m vect_j vect_k = (λ * a1 vect_m vect_j vect_k).3 + (μ * a2 vect_m vect_j vect_k).3 + (ν * a3 vect_m vect_j vect_k).3 ∧
    λ = -2 ∧ μ = 1 ∧ ν = -3 :=
sorry

end MathProofProblem

end determine_lambdas_l650_650867


namespace largest_prime_factor_of_1729_is_19_l650_650601

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650601


namespace max_arithmetic_sequence_of_primes_less_than_150_l650_650691

theorem max_arithmetic_sequence_of_primes_less_than_150 : 
  ∀ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x) ∧ (∀ x ∈ S, x < 150) ∧ (∃ d, ∀ x ∈ S, ∃ n : ℕ, x = S.min' (by sorry) + n * d) → S.card ≤ 5 := 
by
  sorry

end max_arithmetic_sequence_of_primes_less_than_150_l650_650691


namespace expected_first_sequence_grooms_l650_650134

-- Define the harmonic number function
def harmonic (n : ℕ) : ℝ :=
  ∑ k in finset.range n, 1 / (k + 1 : ℝ)

-- Define the expected number of grooms in the first sequence
theorem expected_first_sequence_grooms :
  harmonic 100 = 5.187 :=
by
  sorry

end expected_first_sequence_grooms_l650_650134


namespace sin_cosine_value_l650_650422

-- Define the conditions and questions in Lean 4 syntax
def Triangle (A B C: ℝ) (a b c: ℝ) : Prop :=
  angleOpposite A B C a b c

def sin_C_eq (C: ℝ) : Prop :=
  sin C = 2 * sin (C/2)^2 - sin (C/2)

def sin_C_value : Prop :=
  sin C = 3/4

def cosine_law (a: ℝ) (b: ℝ) (c: ℝ) (C: ℝ) : Prop :=
  c = sqrt (a^2 + b^2 - 2 * a * b * cos C)

-- Create the main theorem to prove sin_C and c values
theorem sin_cosine_value (A B C a b c : ℝ) 
  (h1: Triangle A B C a b c)
  (h2: sin_C_eq C)
  (h3: a = 2)
  (h4: (a^2 + b^2) * sin (A - B) = (a^2 - b^2) * sin (A + B)) :
  sin_C_value ∧ cosine_law a b c C :=
by
  sorry

end sin_cosine_value_l650_650422


namespace legos_given_away_l650_650467

theorem legos_given_away :
  ∀ (initial_legos lost_legos remaining_legos given_legos : ℕ),
    initial_legos = 380 →
    lost_legos = 57 →
    remaining_legos = 299 →
    initial_legos - lost_legos - remaining_legos = given_legos →
    given_legos = 24 :=
by
  intros initial_legos lost_legos remaining_legos given_legos
  intros h_initial h_lost h_remaining h_equation
  rw [h_initial, h_lost, h_remaining] at h_equation
  exact h_equation
  sorry

end legos_given_away_l650_650467


namespace compute_fraction_l650_650648

theorem compute_fraction : ((5 * 7) - 3) / 9 = 32 / 9 := by
  sorry

end compute_fraction_l650_650648


namespace divisors_squared_prime_l650_650121

theorem divisors_squared_prime (p : ℕ) [hp : Fact (Nat.Prime p)] (m : ℕ) (h : m = p^3) (hm_div : Nat.divisors m = 4) :
  Nat.divisors (m^2) = 7 :=
sorry

end divisors_squared_prime_l650_650121


namespace speed_of_B_l650_650675

theorem speed_of_B 
    (initial_distance : ℕ)
    (speed_of_A : ℕ)
    (time : ℕ)
    (distance_covered_by_A : ℕ)
    (distance_covered_by_B : ℕ)
    : initial_distance = 24 → speed_of_A = 5 → time = 2 → distance_covered_by_A = speed_of_A * time → distance_covered_by_B = initial_distance - distance_covered_by_A → distance_covered_by_B / time = 7 :=
by
  sorry

end speed_of_B_l650_650675


namespace find_value_l650_650871

variable (x y : ℝ)

def conditions (x y : ℝ) :=
  y > 2 * x ∧ 2 * x > 0 ∧ (x / y + y / x = 8)

theorem find_value (h : conditions x y) : (x + y) / (x - y) = -Real.sqrt (5 / 3) :=
sorry

end find_value_l650_650871


namespace probability_same_gate_l650_650993

open Finset

-- Definitions based on the conditions
def num_gates : ℕ := 3
def total_combinations : ℕ := num_gates * num_gates -- total number of combinations for both persons
def favorable_combinations : ℕ := num_gates         -- favorable combinations (both choose same gate)

-- Problem statement
theorem probability_same_gate : 
  ∃ (p : ℚ), p = (favorable_combinations : ℚ) / (total_combinations : ℚ) ∧ p = (1 / 3 : ℚ) := 
by
  sorry

end probability_same_gate_l650_650993


namespace dreamCarCost_l650_650690

-- Definitions based on given conditions
def monthlyEarnings : ℕ := 4000
def monthlySavings : ℕ := 500
def totalEarnings : ℕ := 360000

-- Theorem stating the desired result
theorem dreamCarCost :
  (totalEarnings / monthlyEarnings) * monthlySavings = 45000 :=
by
  sorry

end dreamCarCost_l650_650690


namespace rectangle_area_stage_8_l650_650383

def square_side_length : ℕ := 4
def stage_count : ℕ := 8

-- The function to compute the area of one square
def square_area (side_length: ℕ) : ℕ :=
  side_length * side_length

-- The function to compute the total area at a given stage
def total_area_at_stage (side_length: ℕ) (stages: ℕ) : ℕ :=
  stages * (square_area side_length)

theorem rectangle_area_stage_8 :
  total_area_at_stage square_side_length stage_count = 128 :=
  by
    sorry

end rectangle_area_stage_8_l650_650383


namespace candies_converge_to_uniform_l650_650109

/-- A function that describes one round of candy redistribution -/
def redistribute (candies : List ℕ) : List ℕ :=
  let n := candies.length
  candies.mapIdx (λ i candy =>
    let next_candy := (candy + candies[(i + n - 1) % n]) / 2
    if next_candy % 2 = 1 then next_candy + 1 else next_candy
  )

/-- The main theorem stating that after a finite number of redistributions, all children will have the same number of candies -/
theorem candies_converge_to_uniform (n : ℕ) (candies : List ℕ) (h_even : ∀ c ∈ candies, c % 2 = 0) :
  ∃ k : ℕ, ∃ final_candies : List ℕ,
    candies.length = n ∧
    (∀ i, final_candies.nth i = some (final_candies.head.getD 0)) ∧
    (iterate redistribute k candies = final_candies) := sorry

end candies_converge_to_uniform_l650_650109


namespace largest_prime_factor_1729_l650_650542

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650542


namespace num_factors_180_l650_650349

theorem num_factors_180 : 
  ∃ (n : ℕ), n = 180 ∧ prime_factorization n = [(2, 2), (3, 2), (5, 1)] ∧ positive_factors n = 18 :=
by
  sorry

end num_factors_180_l650_650349


namespace inequality_positives_l650_650476

theorem inequality_positives (x1 x2 x3 x4 x5 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (hx5 : 0 < x5) : 
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x3 * x4 + x5 * x1 + x2 * x3 + x4 * x5) :=
sorry

end inequality_positives_l650_650476


namespace find_principal_amount_l650_650824

-- Defining the conditions
def gain_of_B (P : ℝ) : ℝ := P * ((17 / 100) * 4 - (15 / 100) * 4)

-- Given conditions for the proof
theorem find_principal_amount
  (gain : ℝ)
  (h_gain_eq : gain = 160) :
  ∃ P : ℝ, gain_of_B P = gain ∧ P = 2000 :=
by
  -- We set the gain of B (Rs. 160) as given
  use 2000
  split
  rfl
  sorry

end find_principal_amount_l650_650824


namespace number_of_factors_180_l650_650356

theorem number_of_factors_180 : 
  let factors180 (n : ℕ) := 180 
  in factors180 180 = (2 + 1) * (2 + 1) * (1 + 1) => factors180 180 = 18 := 
by
  sorry

end number_of_factors_180_l650_650356


namespace rectangle_dimension_solution_l650_650720

theorem rectangle_dimension_solution (x : ℝ) :
  (x + 3) * (3x - 4) = 5x + 14 → x = (Real.sqrt 78) / 3 :=
by
  intro h,
  sorry

end rectangle_dimension_solution_l650_650720


namespace largest_prime_factor_of_1729_l650_650558

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650558


namespace count_three_digit_integers_ending_in_zero_divisible_by_20_l650_650370

-- Define the predicates to express the conditions
def is_three_digit (n : ℕ) := 100 ≤ n ∧ n ≤ 999
def ends_in_zero (n : ℕ) := n % 10 = 0
def is_divisible_by_20 (n : ℕ) := n % 20 = 0

-- State the theorem based on the problem translation
theorem count_three_digit_integers_ending_in_zero_divisible_by_20 :
  (finset.filter (λ n, is_three_digit n ∧ ends_in_zero n ∧ is_divisible_by_20 n) (finset.range 1000)).card = 45 :=
by
  sorry

end count_three_digit_integers_ending_in_zero_divisible_by_20_l650_650370


namespace largest_prime_factor_of_1729_is_19_l650_650602

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650602


namespace six_digit_divisibility_l650_650842

theorem six_digit_divisibility (a1 a2 a3 : ℕ) (h1 : a1 < 10) (h2 : a2 < 10) (h3 : a3 < 10) :
  let N := 100100 * a1 + 1010 * a2 + 1001 * a3
  in 7 ∣ N ∧ 11 ∣ N ∧ 13 ∣ N :=
by
  let N := 100100 * a1 + 1010 * a2 + 1001 * a3
  sorry

end six_digit_divisibility_l650_650842


namespace beckys_age_ratio_l650_650737

theorem beckys_age_ratio (Eddie_age : ℕ) (Irene_age : ℕ)
  (becky_age: ℕ)
  (H1 : Eddie_age = 92)
  (H2 : Irene_age = 46)
  (H3 : Irene_age = 2 * becky_age) :
  becky_age / Eddie_age = 1 / 4 :=
by
  sorry

end beckys_age_ratio_l650_650737


namespace total_cost_of_shirt_and_sweater_l650_650511

-- Define the given conditions
def price_of_shirt := 36.46
def diff_price_shirt_sweater := 7.43
def price_of_sweater := price_of_shirt + diff_price_shirt_sweater

-- Statement to prove
theorem total_cost_of_shirt_and_sweater :
  price_of_shirt + price_of_sweater = 80.35 :=
by
  -- Proof goes here
  sorry

end total_cost_of_shirt_and_sweater_l650_650511


namespace surface_area_proof_l650_650698

structure SolidRightPrism :=
  (height : ℝ)
  (side_length : ℝ)

def midpoint (a b : ℝ) := (a + b) / 2

axiom slicing_midpoints 
  (PQ QR RP : ℝ) (M N O: ℝ) : 
  M = midpoint PQ RP ∧ 
  N = midpoint QR PQ ∧ 
  O = midpoint RP QR

noncomputable def surface_area_sliced_off 
  (PQRSTUV : SolidRightPrism) 
  (M N O : ℝ) : ℝ := 
  45 + 6.25 * real.sqrt 3 + 2.5 * real.sqrt 104

theorem surface_area_proof: 
  (P : SolidRightPrism)
  (M N O : ℝ)
  (H : slicing_midpoints P.side_length P.side_length P.side_length M N O):
  surface_area_sliced_off P M N O = 45 + 6.25 * real.sqrt 3 + 2.5 * real.sqrt 104 := 
  sorry

end surface_area_proof_l650_650698


namespace inequality_am_gm_l650_650876

theorem inequality_am_gm (a b c d : ℝ) (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by
  sorry

end inequality_am_gm_l650_650876


namespace x_squared_inverse_y_fourth_l650_650010

theorem x_squared_inverse_y_fourth (x y : ℝ) (k : ℝ) (h₁ : x = 8) (h₂ : y = 2) (h₃ : (x^2) * (y^4) = k) : x^2 = 4 :=
by
  sorry

end x_squared_inverse_y_fourth_l650_650010


namespace expected_value_first_outstanding_sequence_l650_650142

-- Define indicator variables and the harmonic number
noncomputable def I (k : ℕ) := 1 / k
noncomputable def H (n : ℕ) := ∑ i in Finset.range (n + 1), I (i + 1)

-- Theorem: Expected value of the first sequence of outstanding grooms
theorem expected_value_first_outstanding_sequence : 
  (H 100) = 5.187 := 
sorry

end expected_value_first_outstanding_sequence_l650_650142


namespace largest_prime_factor_1729_l650_650531

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650531


namespace Elizabeth_lost_bottles_l650_650738

theorem Elizabeth_lost_bottles :
  ∃ (L : ℕ), (10 - L - 1) * 3 = 21 ∧ L = 2 := by
  sorry

end Elizabeth_lost_bottles_l650_650738


namespace explicit_formula_for_f_minimum_value_of_sq_plus_range_of_a_l650_650309

noncomputable def f (x : ℝ) : ℝ := log x / log 2 + 1

theorem explicit_formula_for_f :
  ∀ x : ℝ, x > 0 → f (2 ^ x) = x + 1 :=
begin
  intro x,
  intro hx,
  unfold f,
  have h2x : log (2 ^ x) = x * log 2,
  from real.logb_mul_of_pos real.two_pos hx,
  rw [h2x, ← mul_div_assoc, mul_one_div_cancel (ne_of_gt (log2_pos)), one_mul],
end

theorem minimum_value_of_sq_plus :
  ∃ x : ℝ, x > 0 ∧ y = (log 2 / log x + 1)^2 + log 2 * x + 1 + 1 :=
begin
  sorry
end

theorem range_of_a :
  (∀ a : ℝ, h(x) - a = 0 → x ∈ set.Ioo 1 2) → a ∈ set.Ioo 3 6 :=
begin
  sorry
end

end explicit_formula_for_f_minimum_value_of_sq_plus_range_of_a_l650_650309


namespace expected_number_of_first_sequence_l650_650149

-- Define the concept of harmonic number
def harmonic (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), (1 : ℚ) / (i + 1)

-- Define the problem statement
theorem expected_number_of_first_sequence (n : ℕ) (h : n = 100) : harmonic n = 5.187 := by
  sorry

end expected_number_of_first_sequence_l650_650149


namespace find_a_plus_b_plus_c_l650_650486

noncomputable def side_length_square (a b c : ℕ) : ℝ :=
  (a - real.sqrt b) / c

lemma smaller_square_side_length (a b c : ℕ) (h₀ : a = 3)
  (h₁ : b = 2) (h₂ : c = 5)
  (h₃ : square_ABCD (1 : ℝ))
  (h₄ : points_EF_on_BC_CD h₀ h₁)
  (h₅ : right_triangle_AEF h₀ h₁ h₂)
  (h₆ : square_with_vertex_B_parallel_sides h₃ h₄ h₅) :
  side_length_square a b c = (3 - real.sqrt 2) / 5 :=
sorry

theorem find_a_plus_b_plus_c : ∃ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 5 ∧ a + b + c = 10 :=
begin
  use [3, 2, 5],
  split,
  refl,
  split,
  refl,
  split,
  refl,
  norm_num,
end

end find_a_plus_b_plus_c_l650_650486


namespace vivian_mail_june_l650_650988

theorem vivian_mail_june :
  ∀ (m_apr m_may m_jul m_aug : ℕ),
  m_apr = 5 →
  m_may = 10 →
  m_jul = 40 →
  ∃ m_jun : ℕ,
  ∃ pattern : ℕ → ℕ,
  (pattern m_apr = m_may) →
  (pattern m_may = m_jun) →
  (pattern m_jun = m_jul) →
  (pattern m_jul = m_aug) →
  (m_aug = 80) →
  pattern m_may = m_may * 2 →
  pattern m_jun = m_jun * 2 →
  pattern m_jun = 20 :=
by
  sorry

end vivian_mail_june_l650_650988


namespace hexagon_area_is_five_l650_650989

theorem hexagon_area_is_five :
  ∃ (hexagon : set (ℝ × ℝ)), is_regular_right_triangle_hexagonal_constructed hexagon ∧ 
  is_central_square_side_len_one hexagon ∧ 
  distances_correct hexagon ∧ 
  area hexagon = 5 :=
sorry

end hexagon_area_is_five_l650_650989


namespace prime_solutions_l650_650750

theorem prime_solutions : 
  ∃ (ps : List (ℤ × ℤ × ℤ)), 
    (∀ pqr ∈ ps, 
      (let ⟨p, q, r⟩ := pqr in 
        prime p ∧ prime q ∧ prime r ∧ p ≠ q + r ∧ q ≠ 0 ∧ r ≠ 0 ∧ 
        1 / (p - q - r : ℚ) = 1 / (q : ℚ) + 1 / (r : ℚ))) ∧ 
    ps = [(5, 2, 2), (-5, -2, -2), (-5, 3, -2), (-5, -2, 3), (5, 2, -3), (5, -3, 2)] :=
sorry

end prime_solutions_l650_650750


namespace count_100_digit_even_numbers_l650_650343

theorem count_100_digit_even_numbers : 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  in
  count_valid_numbers = 2 * 3 ^ 98 :=
by 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  have : count_valid_numbers = 2 * 3 ^ 98 := by sorry
  exact this

end count_100_digit_even_numbers_l650_650343


namespace max_squares_covered_l650_650100

theorem max_squares_covered 
    (board_square_side : ℝ) 
    (card_side : ℝ) 
    (n : ℕ) 
    (h1 : board_square_side = 1) 
    (h2 : card_side = 2) 
    (h3 : ∀ x y : ℝ, (x*x + y*y ≤ card_side*card_side) → card_side*card_side ≤ 4) :
    n ≤ 9 := sorry

end max_squares_covered_l650_650100


namespace mandy_accepted_schools_l650_650886

variable (total_schools : Nat) (fraction_applied fraction_accepted : Rat)

-- Conditions
def mandy_researched : Prop := total_schools = 96
def applied_fraction : Prop := fraction_applied = 5 / 8
def accepted_fraction : Prop := fraction_accepted = 3 / 5

-- Calculate number of schools applied to
def num_applied : Nat :=
  (fraction_applied * total_schools).toInt

-- Calculate number of schools accepted to
def num_accepted : Nat :=
  (fraction_accepted * num_applied).toInt

-- Theorem: Number of schools Mandy was accepted to
theorem mandy_accepted_schools (h1 : mandy_researched)
                                (h2 : applied_fraction)
                                (h3 : accepted_fraction) :
  num_accepted = 36 := by
  -- Conditions guarantee valid calculation in Natural numbers
  have h_num_applied : num_applied = 60 := by
    simp [num_applied, fraction_applied, total_schools, h1, h2]
    sorry -- Fill in details of computation

  have h_num_accepted : num_accepted = 36 := by
    simp [num_accepted, fraction_accepted, num_applied, h_num_applied, h3]
    sorry -- Fill in details of computation

  exact h_num_accepted

end mandy_accepted_schools_l650_650886


namespace largest_prime_factor_of_1729_is_19_l650_650603

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650603


namespace total_cost_of_shirt_and_sweater_l650_650510

-- Define the given conditions
def price_of_shirt := 36.46
def diff_price_shirt_sweater := 7.43
def price_of_sweater := price_of_shirt + diff_price_shirt_sweater

-- Statement to prove
theorem total_cost_of_shirt_and_sweater :
  price_of_shirt + price_of_sweater = 80.35 :=
by
  -- Proof goes here
  sorry

end total_cost_of_shirt_and_sweater_l650_650510


namespace centroid_incenter_parallel_l650_650777

theorem centroid_incenter_parallel {A B C : Type} (a b c : ℝ) (h : a + b = 2 * c) :
  parallel (lineThrough centroid(A, B, C)) (lineThrough incenter(A, B, C)) (lineThrough C (midpoint A B)) :=
sorry

end centroid_incenter_parallel_l650_650777


namespace S3_inter_S4_empty_S3_inter_S5_empty_l650_650862

noncomputable def S (n : ℕ) : Set ℕ := 
  {k | ∃ g : ℕ, g ≥ 2 ∧ k = (1 + ∑ i in Finset.range n, g ^ i)}

theorem S3_inter_S4_empty : 
  S 3 ∩ S 4 = ∅ := 
by sorry

theorem S3_inter_S5_empty : 
  S 3 ∩ S 5 = ∅ := 
by sorry

end S3_inter_S4_empty_S3_inter_S5_empty_l650_650862


namespace negation_implication_l650_650499

-- Define the propositions p and q as Boolean variables
def p : Prop := sorry
def q : Prop := sorry

-- Define implies (=>) in Lean
def impl (a b : Prop) : Prop := a → b

-- Prove that the negation of the implication "If p, then q" is equivalent to "If p, then ¬q"
theorem negation_implication (p q : Prop) : ¬(impl p q) = impl p (¬q) :=
sorry

end negation_implication_l650_650499


namespace train_speeds_l650_650521

theorem train_speeds (d t1 v1 v2 : ℝ) (h1 : d = 100) (h2 : t1 = 4 / 3) (h3 : v1 = 30) :
  v2 = 45 :=
by
  -- Defining the equality based on given conditions and expected result
  have h : 30 * (4 / 3) + v2 * (4 / 3) = 100, from sorry,
  have simplify_h : 40 + (4 / 3) * v2 = 100, from sorry,

  -- Proving the final step
  sorry  -- Explanation and final steps for proving this will go here

end train_speeds_l650_650521


namespace minimum_value_of_f_l650_650211

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ 5/2) ∧ (f 1 = 5/2) := by
  sorry

end minimum_value_of_f_l650_650211


namespace expected_first_sequence_grooms_l650_650135

-- Define the harmonic number function
def harmonic (n : ℕ) : ℝ :=
  ∑ k in finset.range n, 1 / (k + 1 : ℝ)

-- Define the expected number of grooms in the first sequence
theorem expected_first_sequence_grooms :
  harmonic 100 = 5.187 :=
by
  sorry

end expected_first_sequence_grooms_l650_650135


namespace total_cost_of_fencing_l650_650029

theorem total_cost_of_fencing (length breadth : ℕ) (cost_per_metre : ℕ) 
  (h1 : length = breadth + 20) 
  (h2 : length = 200) 
  (h3 : cost_per_metre = 26): 
  2 * (length + breadth) * cost_per_metre = 20140 := 
by sorry

end total_cost_of_fencing_l650_650029


namespace sin_pi_eighth_lt_pi_eighth_l650_650791

theorem sin_pi_eighth_lt_pi_eighth : 
  (a : ℝ) (b : ℝ), 
  a = Real.sin (Real.pi / 8) → 
  b = Real.pi / 8 → 
  a < b := 
by
  sorry

end sin_pi_eighth_lt_pi_eighth_l650_650791


namespace factorize_1_factorize_2_l650_650744

theorem factorize_1 {x : ℝ} : 2*x^2 - 4*x = 2*x*(x - 2) := 
by sorry

theorem factorize_2 {a b x y : ℝ} : a^2*(x - y) + b^2*(y - x) = (x - y) * (a + b) * (a - b) := 
by sorry

end factorize_1_factorize_2_l650_650744


namespace taylor_coefficient_a3_l650_650380

theorem taylor_coefficient_a3 : 
  let f := λ x : ℝ, x^5 + 3 * x^3 + 1
  let taylor_series := (0:ℕ) → ℝ
  (∃ a : fin 6 → ℝ, (∀ x, f x = a 0 + a 1 * (x-1) + a 2 * (x-1)^2 + a 3 * (x-1)^3 + a 4 * (x-1)^4 + a 5 * (x-1)^5) 
  ∧ taylor_series 3 = 13) :=
begin
  sorry
end

end taylor_coefficient_a3_l650_650380


namespace damien_jog_days_determinate_damien_jog_days_indeterminate_l650_650723

theorem damien_jog_days_determinate
  (total_miles : ℕ)
  (miles_per_day : ℕ)
  (total_days : ℕ)
  (h_total_miles : total_miles = 75)
  (h_miles_per_day : miles_per_day = 5)
  (h_total_days : total_days * miles_per_day = total_miles) :
  total_days = 15 :=
by {
  rw [h_total_miles, h_miles_per_day] at h_total_days,
  linarith,
}

theorem damien_jog_days_indeterminate
  (days_of_week : list string)
  (days_jog_schedule : ℕ → list string)
  (total_days : ℕ)
  (h_total_days : total_days = 15) :
  ∀ n, days_jog_schedule n = days_of_week :=
sorry

end damien_jog_days_determinate_damien_jog_days_indeterminate_l650_650723


namespace largest_prime_factor_of_1729_is_19_l650_650611

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650611


namespace largest_prime_factor_1729_l650_650582

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650582


namespace largest_prime_factor_of_1729_is_19_l650_650607

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650607


namespace product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l650_650966

theorem product_of_three_divisors_of_5_pow_4_eq_5_pow_4 (a b c : ℕ) (h1 : a * b * c = 5^4) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : a ≠ c) : a + b + c = 131 :=
sorry

end product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l650_650966


namespace smallest_bdf_l650_650901

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end smallest_bdf_l650_650901


namespace largest_prime_factor_of_1729_l650_650549

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650549


namespace exist_irrational_gt_one_diff_floor_l650_650199

theorem exist_irrational_gt_one_diff_floor:
  ∃ a b : ℝ, irrational a ∧ irrational b ∧ 1 < a ∧ 1 < b ∧ (∀ m n : ℕ, ⌊a^m⌋ ≠ ⌊b^n⌋) := 
sorry

end exist_irrational_gt_one_diff_floor_l650_650199


namespace solve_dogsled_distance_l650_650984

noncomputable def dogsled_distance : ℕ :=
  let T_W := 15
  let V_W := 20
  let T_A := T_W - 3
  let V_A := V_W + 5
  20 * T_W

theorem solve_dogsled_distance : dogsled_distance = 300 := by
  let T_W := 15
  let V_W := 20
  let T_A := T_W - 3
  let V_A := V_W + 5
  have h1 : 20 * T_W = 300 := by
    calc
      20 * 15 = 300 : by norm_num
  exact h1

end solve_dogsled_distance_l650_650984


namespace max_distance_centroid_l650_650184

theorem max_distance_centroid :
  ∀ (a1 a2 b1 b2 c1 c2 : ℝ), (a1 ∈ {-1, 0, 1}) → (a2 ∈ {-1, 0, 1}) →
  (b1 ∈ {-1, 0, 1}) → (b2 ∈ {-1, 0, 1}) →
  (c1 ∈ {-1, 0, 1}) → (c2 ∈ {-1, 0, 1}) →
  let s1 := (a1 + b1 + c1) / 3
      s2 := (a2 + b2 + c2) / 3 in
  (√(s1^2 + s2^2) ≤ (2 * √2) / 3) :=
by {
  sorry
}

end max_distance_centroid_l650_650184


namespace solve_profession_arrangement_l650_650245

inductive Profession
| architect
| barista
| veterinarian
| guitarist

inductive Person
| Andrey
| Boris
| Vyacheslav
| Gennady

open Profession
open Person

structure SeatingArrangement :=
(seat1 : Person)
(seat2 : Person)
(seat3 : Person)
(seat4 : Person)

structure Assignment :=
(profession : Person → Profession)

def correct_arrangement_and_profession (sa : SeatingArrangement) (asgmt : Assignment) : Prop :=
(sa.seat1 ≠ sa.seat2 ∧ sa.seat1 ≠ sa.seat3 ∧ sa.seat1 ≠ sa.seat4 ∧
 sa.seat2 ≠ sa.seat3 ∧ sa.seat2 ≠ sa.seat4 ∧ sa.seat3 ≠ sa.seat4) ∧
((asgmt profession sa.seat1 = barista) ∧ (asgmt profession sa.seat2 = architect) ∧ 
 (asgmt profession sa.seat3 = veterinarian) ∧ (asgmt profession sa.seat4 = guitarist)) ∧
(sa.seat3 = Andrey) ∧ 
(sa.seat2 = Boris) ∧ 
(sa.seat4 = Vyacheslav)

theorem solve_profession_arrangement : 
  ∃ (sa : SeatingArrangement) (asgmt : Assignment), correct_arrangement_and_profession sa asgmt :=
sorry

end solve_profession_arrangement_l650_650245


namespace initial_garrison_men_l650_650106

theorem initial_garrison_men (M : ℕ) (h1 : 62 * M = 62 * M) 
  (h2 : M * 47 = (M + 2700) * 20) : M = 2000 := by
  sorry

end initial_garrison_men_l650_650106


namespace perfect_square_or_cube_less_than_900_l650_650818

theorem perfect_square_or_cube_less_than_900 : 
  ∃ n : ℕ, n = 35 ∧ 
  n = 
    (Finset.card (Finset.filter (λ x, x * x < 900) (Finset.range 30))) + 
    (Finset.card (Finset.filter (λ x, x * x * x < 900) (Finset.range 10))) - 
    (Finset.card (Finset.filter (λ x, x ^ 6 < 900) (Finset.range 4))) :=
by
  sorry

end perfect_square_or_cube_less_than_900_l650_650818


namespace regular_hexagon_area_l650_650990

theorem regular_hexagon_area (r : ℝ) (h1 : π * r^2 = 256 * π) :
  6 * (r^2 * sqrt 3 / 4) = 384 * sqrt 3 :=
by
  sorry

end regular_hexagon_area_l650_650990


namespace matt_revenue_l650_650890

def area (length : ℕ) (width : ℕ) : ℕ := length * width

def grams_peanuts (area : ℕ) (g_per_sqft : ℕ) : ℕ := area * g_per_sqft

def kg_peanuts (grams : ℕ) : ℕ := grams / 1000

def grams_peanut_butter (grams_peanuts : ℕ) : ℕ := (grams_peanuts / 20) * 5

def kg_peanut_butter (grams_pb : ℕ) : ℕ := grams_pb / 1000

def revenue (kg_pb : ℕ) (price_per_kg : ℕ) : ℕ := kg_pb * price_per_kg

theorem matt_revenue : 
  let length := 500
  let width := 500
  let g_per_sqft := 50
  let conversion_ratio := 20 / 5
  let price_per_kg := 10
  revenue (kg_peanut_butter (grams_peanut_butter (grams_peanuts (area length width) g_per_sqft))) price_per_kg = 31250 := by
  sorry

end matt_revenue_l650_650890


namespace playground_area_l650_650035

theorem playground_area (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 100)
  (h_length : l = 3 * w) : l * w = 468.75 :=
by
  sorry

end playground_area_l650_650035


namespace part_1_part_2_l650_650313

noncomputable def f (x : ℝ) : ℝ := ln ((x + 1) / (x - 1))
noncomputable def g (x : ℝ) : ℝ := ln x - (x - 1)

theorem part_1 (x : ℝ) (h₁ : x ∈ (-∞ : ℝ, -1) ∪ (1, ∞ : ℝ)) : f (-x) = -f (x) := 
by sorry

theorem part_2 (n : ℕ) (h₂ : ∀ x > 1, g x < 0) (h₃ : 0 < n) : f 2 + f 4 + ... + f (2 * n) < 2 * n := 
by sorry

end part_1_part_2_l650_650313


namespace num_bases_for_625_ending_in_1_l650_650764

theorem num_bases_for_625_ending_in_1 :
  (Finset.card (Finset.filter (λ b : ℕ, 624 % b = 0) (Finset.Icc 3 10))) = 4 :=
by
  sorry

end num_bases_for_625_ending_in_1_l650_650764


namespace largest_whole_number_l650_650209

theorem largest_whole_number (x : ℕ) (h1 : 11 * x < 150) (h2 : x % 3 = 0) : x ≤ 13 ∧ (∀ y : ℕ, 11 * y < 150 → y % 3 = 0 → y ≤ x) :=
begin
  use 12,
  split,
  { exact by norm_num },
  { intros y hy1 hy2,
    sorry
  }
end

end largest_whole_number_l650_650209


namespace largest_prime_factor_of_1729_is_19_l650_650606

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650606


namespace smallest_bdf_l650_650900

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end smallest_bdf_l650_650900


namespace largest_prime_factor_1729_l650_650527

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650527


namespace expected_first_sequence_length_is_harmonic_100_l650_650132

open Real

-- Define the harmonic number
def harmonic_number (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (i + 1 : ℝ)

-- Define the expected length of the first sequence
def expected_first_sequence_length (n : ℕ) : ℝ :=
  harmonic_number n

-- Prove that the expected number of suitors in the first sequence is the 100th harmonic number
theorem expected_first_sequence_length_is_harmonic_100 :
  expected_first_sequence_length 100 = harmonic_number 100 :=
by
  sorry

end expected_first_sequence_length_is_harmonic_100_l650_650132


namespace num_bases_ending_in_1_l650_650762

theorem num_bases_ending_in_1 : 
  (∃ bases : Finset ℕ, 
  ∀ b ∈ bases, 3 ≤ b ∧ b ≤ 10 ∧ (625 % b = 1) ∧ bases.card = 4) :=
sorry

end num_bases_ending_in_1_l650_650762


namespace minimum_points_in_M_l650_650980

-- Define the set of points M and the seven circles with the given conditions.
def set_of_points : Type := sorry
def C (n : ℕ) : set_of_points -> Prop := sorry

axiom points_on_circles :
  ∀ n : ℕ, (n > 0 ∧ n ≤ 7) → ∃ (M : set set_of_points), 
  M.card = n ∧ ∀ (x ∈ M), C n x

-- Formulate the problem to validate that the minimum number of points in M is 12
theorem minimum_points_in_M : 
  (∃ (M : set set_of_points) (C1 C2 C3 C4 C5 C6 C7 : set set_of_points), 
  ∀ (x : set_of_points), 
  (C7 x ↔ x ∈ M ∧ M.card = 7) ∧
  (C6 x ↔ x ∈ M ∧ M.card = 6) ∧
  (C5 x ↔ x ∈ M ∧ M.card = 5) ∧
  (C4 x ↔ x ∈ M ∧ M.card = 4) ∧
  (C3 x ↔ x ∈ M ∧ M.card = 3) ∧
  (C2 x ↔ x ∈ M ∧ M.card = 2) ∧
  (C1 x ↔ x ∈ M ∧ M.card = 1)) →
  M.card = 12 :=
sorry

end minimum_points_in_M_l650_650980


namespace largest_prime_factor_1729_l650_650568

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650568


namespace steve_answered_38_questions_l650_650974

theorem steve_answered_38_questions (total_questions : ℕ) (blank_questions : ℕ) (h1 : total_questions = 45) (h2 : blank_questions = 7) : (total_questions - blank_questions = 38) :=
by
  rw [h1, h2]
  sorry

end steve_answered_38_questions_l650_650974


namespace repeating_decimals_count_l650_650769

theorem repeating_decimals_count :
  {n : ℤ | 1 ≤ n ∧ n ≤ 15 ∧ ¬ is_terminating_decimal (n / 15)}.to_finset.card = 14 :=
by
  sorry

def is_terminating_decimal (q : ℚ) : Prop :=
  let p := (q.denom : ℕ)
  ∀ d, d ∣ p → d = 1 ∨ d = 2 ∨ d = 5

end repeating_decimals_count_l650_650769


namespace factors_of_180_l650_650365

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l650_650365


namespace max_expression_value_unit_vectors_l650_650292

variables {F : Type*} [inner_product_space ℝ F]
variables (u v w : F)
variable [is_unit_vector_u : ∥u∥ = 1]
variable [is_unit_vector_v : ∥v∥ = 1]
variable [is_unit_vector_w : ∥w∥ = 1]

theorem max_expression_value_unit_vectors
  (hu : ∥u∥ = 1) (hv : ∥v∥ = 1) (hw : ∥w∥ = 1) :
  ∥u - v∥^2 + ∥u - w∥^2 + ∥v - w∥^2 + ∥u + v + w∥^2 ≤ 12 :=
sorry

end max_expression_value_unit_vectors_l650_650292


namespace melinda_doughnuts_picked_l650_650469

theorem melinda_doughnuts_picked :
  (∀ d h_coffee m_coffee : ℕ, d = 3 → h_coffee = 4 → m_coffee = 6 →
    ∀ cost_d cost_h cost_m : ℝ, cost_d = 0.45 → 
    cost_h = 4.91 → cost_m = 7.59 → 
    ∃ m_doughnuts : ℕ, cost_m - m_coffee * ((cost_h - d * cost_d) / h_coffee) = m_doughnuts * cost_d) → 
  ∃ n : ℕ, n = 5 := 
by sorry

end melinda_doughnuts_picked_l650_650469


namespace find_m_l650_650299

theorem find_m (m x_1 x_2 : ℝ) 
  (h1 : x_1^2 + m * x_1 - 3 = 0) 
  (h2 : x_2^2 + m * x_2 - 3 = 0) 
  (h3 : x_1 + x_2 - x_1 * x_2 = 5) : 
  m = -2 :=
sorry

end find_m_l650_650299


namespace range_a_l650_650727

theorem range_a (A B : Set ℝ) (a : ℝ) :
  (A = {x : ℝ | x < -1 ∨ x > 3}) ∧
  (B = {y : ℝ | -a < y ∧ y ≤ 4 - a}) ∧
  (∀ y, y ∈ B → y ∈ A) →
  a ∈ Set.Icc (-∞) (-3) ∪ Set.Ioc (5) (+∞) :=
  sorry

end range_a_l650_650727


namespace average_daily_net_income_correct_l650_650096

-- Define the income, tips, and expenses for each day.
def day1_income := 300
def day1_tips := 50
def day1_expenses := 80

def day2_income := 150
def day2_tips := 20
def day2_expenses := 40

def day3_income := 750
def day3_tips := 100
def day3_expenses := 150

def day4_income := 200
def day4_tips := 30
def day4_expenses := 50

def day5_income := 600
def day5_tips := 70
def day5_expenses := 120

-- Define the net income for each day as income + tips - expenses.
def day1_net_income := day1_income + day1_tips - day1_expenses
def day2_net_income := day2_income + day2_tips - day2_expenses
def day3_net_income := day3_income + day3_tips - day3_expenses
def day4_net_income := day4_income + day4_tips - day4_expenses
def day5_net_income := day5_income + day5_tips - day5_expenses

-- Calculate the total net income over the 5 days.
def total_net_income := 
  day1_net_income + day2_net_income + day3_net_income + day4_net_income + day5_net_income

-- Define the number of days.
def number_of_days := 5

-- Calculate the average daily net income.
def average_daily_net_income := total_net_income / number_of_days

-- Statement to prove that the average daily net income is $366.
theorem average_daily_net_income_correct :
  average_daily_net_income = 366 := by
  sorry

end average_daily_net_income_correct_l650_650096


namespace ellipse_and_line_conditions_l650_650795

noncomputable def standard_ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
x^2 / a^2 + y^2 / b^2 = 1

noncomputable def line_intersection_condition (m n x₁ x₂ x₀ k : ℝ) : Prop :=
m = x₁ / (x₁ - 2) ∧ n = x₂ / (x₂ - 2) ∧
(m + n = 10)

theorem ellipse_and_line_conditions :
  let a : ℝ := sqrt 5,
      b : ℝ := 1,
      e : ℝ := 2 * sqrt 5 / 5,
      F := (2 : ℝ, 0),
      k : ℝ := 1 in
  ∀ x y : ℝ,
  (standard_ellipse_equation a b x y) ∧ 
  ∀ x₁ x₂ y₀ : ℝ,
  (line_intersection_condition (x₁ / (x₁ - 2)) (x₂ / (x₂ - 2)) x₁ x₂ y₀ e) :=
by 
  intros x y x₁ x₂ y₀;
  split;
  sorry

end ellipse_and_line_conditions_l650_650795


namespace expected_first_sequence_100_l650_650146

noncomputable def expected_first_sequence (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1).filter (λ k, k > 0), (1 : ℝ) / k

theorem expected_first_sequence_100 : expected_first_sequence 100 = 
    ∑ k in (finset.range 101).filter (λ k, k > 0), (1 : ℝ) / k :=
by
  -- The proof would involve showing this sum represents the harmonic number H_100
  sorry

end expected_first_sequence_100_l650_650146


namespace tug_of_war_matches_l650_650927

-- Define the number of classes
def num_classes : ℕ := 7

-- Define the number of matches Grade 3 Class 6 competes in
def matches_class6 : ℕ := num_classes - 1

-- Define the total number of matches
def total_matches : ℕ := (num_classes - 1) * num_classes / 2

-- Main theorem stating the problem
theorem tug_of_war_matches :
  matches_class6 = 6 ∧ total_matches = 21 := by
  sorry

end tug_of_war_matches_l650_650927


namespace inequality_proof_l650_650802

variable {R : Type*} [LinearOrderedField R]

theorem inequality_proof (a b c x y z : R) (h1 : x^2 < a^2) (h2 : y^2 < b^2) (h3 : z^2 < c^2) :
  x^2 + y^2 + z^2 < a^2 + b^2 + c^2 ∧ x^3 + y^3 + z^3 < a^3 + b^3 + c^3 :=
by
  sorry

end inequality_proof_l650_650802


namespace not_in_second_quadrant_l650_650027

-- Define the function and prove that it does not intersect the second quadrant.
def function_def (x : ℝ) : ℝ := 3^x - 2

theorem not_in_second_quadrant :
  ¬ ∃ x : ℝ, function_def x > 0 ∧ x < 0 :=
sorry

end not_in_second_quadrant_l650_650027


namespace solveProfessions_l650_650252

-- Define our types for Friends and Professions
inductive Friend | Andrey | Boris | Vyacheslav | Gennady
inductive Profession | Architect | Barista | Veterinarian | Guitarist

-- Define the seating arrangement and conditions
def seatingArrangement : Friend → Position := sorry
def profession : Friend → Profession := sorry

-- Conditions translated to Lean
axiom condition1 : ∀ (f : Friend), 
  profession f = Profession.Veterinarian → 
  ∃ (neighbor1 neighbor2 : Friend), 
    seatingArrangement neighbor1 = seatingArrangement f + 1 ∨ seatingArrangement neighbor1 = seatingArrangement f - 1 ∧ 
    seatingArrangement neighbor2 = seatingArrangement f + 1 ∨ seatingArrangement neighbor2 = seatingArrangement f - 1 ∧
    (profession neighbor1 = Profession.Architect ∧ profession neighbor2 = Profession.Guitarist) 

axiom condition2 : ∀ (f : Friend), 
  profession f = Profession.Barista → 
  ∃ (rightNeighbor : Friend), 
    seatingArrangement rightNeighbor = seatingArrangement f + 1 ∧ 
    rightNeighbor = Friend.Boris

axiom condition3 : seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Andrey ∧ seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Boris

axiom condition4 : ∀ (f : Friend), 
  f = Friend.Andrey → 
  ∃ (leftNeighbor rightNeighbor : Friend), 
    seatingArrangement leftNeighbor = seatingArrangement f - 1 ∧ 
    seatingArrangement rightNeighbor = seatingArrangement f + 1

axiom condition5 : ∀ (f1 f2 : Friend), 
  (profession f1 = Profession.Guitarist ∧ profession f2 = Profession.Barista) → 
  seatingArrangement f1 ≠ seatingArrangement f2 + 1 ∧ seatingArrangement f1 ≠ seatingArrangement f2 - 1

-- Define the positions
def Position : Type := Fin 4

-- The main theorem to prove the assignments
theorem solveProfessions :
  profession Friend.Gennady = Profession.Barista ∧ 
  profession Friend.Boris = Profession.Architect ∧ 
  profession Friend.Andrey = Profession.Veterinarian ∧ 
  profession Friend.Vyacheslav = Profession.Guitarist :=
sorry

end solveProfessions_l650_650252


namespace shirts_returned_l650_650327

theorem shirts_returned (bought ended_with : ℕ) (h : bought = 11) (e : ended_with = 5) : (bought - ended_with) = 6 :=
by
  simp [h, e]
  sorry

end shirts_returned_l650_650327


namespace digits_of_8_pow_20_mul_3_pow_30_l650_650645

open Real

theorem digits_of_8_pow_20_mul_3_pow_30 :
  let n := 8 ^ 20 * 3 ^ 30 in
  Nat.floor (log10 n) + 1 = 24 :=
by
  sorry

end digits_of_8_pow_20_mul_3_pow_30_l650_650645


namespace exists_N_monotonic_sequence_l650_650288

universe u

def is_monotonic {α : Type u} [linear_order α] (a : ℕ → α) (i m : ℕ) : Prop :=
  (∀ j < m - 1, a i < a (i + 1 + j)) ∨ (∀ j < m - 1, a i > a (i + 1 + j))

def in_monotonic_segment {α : Type u} [linear_order α] (a : ℕ → α) (k : ℕ) : Prop :=
  ∃ i, is_monotonic a i (k + 1) ∧ i ≤ k ∧ k < i + (k + 1)

theorem exists_N_monotonic_sequence {α : Type u} [linear_order α] (a : ℕ → α)
  (h1 : ∀ i j, i ≠ j → a i ≠ a j)
  (h2 : ∀ k, in_monotonic_segment a k) :
  ∃ N, ∀ m, is_monotonic a N (m + 1) :=
sorry

end exists_N_monotonic_sequence_l650_650288


namespace rectangle_area_l650_650498

-- Definitions based on conditions
def square_area : ℝ := 900
def square_side := real.sqrt square_area
def circle_radius := square_side
def rectangle_length := (2 / 5) * circle_radius

-- Theorem statement
theorem rectangle_area (B : ℝ) : (rectangle_length * B) = 12 * B := by
  -- proof placeholder
  sorry

end rectangle_area_l650_650498


namespace cos_eq_4_solutions_l650_650753

theorem cos_eq_4_solutions : 
  ∃ x1 x2 x3 x4 ∈ Icc (-π) π, 
  (cos (3 * x1) + cos (2 * x1)^2 + cos (x1)^3 = 0) ∧
  (cos (3 * x2) + cos (2 * x2)^2 + cos (x2)^3 = 0) ∧
  (cos (3 * x3) + cos (2 * x3)^2 + cos (x3)^3 = 0) ∧
  (cos (3 * x4) + cos (2 * x4)^2 + cos (x4)^3 = 0) ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 
  :=
sorry

end cos_eq_4_solutions_l650_650753


namespace final_professions_correct_l650_650266

-- Define the people and their positions
inductive Person
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq, Inhabited

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq, Inhabited

open Person Profession

-- Given positions in seats
def seatingArrangement : Person → Nat := 
λ p, 
  match p with
  | Andrey    => 3
  | Boris     => 2
  | Vyacheslav => 4
  | Gennady   => 1

-- Given professions
def profession : Person → Profession := 
λ p, 
  match p with
  | Andrey    => Veterinarian
  | Boris     => Architect
  | Vyacheslav => Guitarist
  | Gennady   => Barista

-- Define the conditions
def condition1 : Prop :=
  (seatingArrangement Andrey = 3) ∧ 
  (seatingArrangement Boris = 2) ∧
  (seatingArrangement Vyacheslav = 4) ∧ 
  (seatingArrangement Gennady = 1)

def condition2 : Prop := 
  seatingArrangement (Person.Barista) + 1 = seatingArrangement Boris

def condition3 : Prop := 
  seatingArrangement Vyacheslav > seatingArrangement Andrey ∧ 
  seatingArrangement Vyacheslav > seatingArrangement Boris

def condition4 : Prop := 
  seatingArrangement (Person.Veterinarian) ≠ 1 ∧ seatingArrangement (Person.Veterinarian) ≠ 4

def condition5 : Prop := 
  (seatingArrangement (Person.Barista) ≠ seatingArrangement (Person.Guitarist) + 1 ∧
  seatingArrangement (Person.Guitarist) ≠ seatingArrangement (Person.Barista) + 1)

-- Prove the final professions
theorem final_professions_correct :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  profession Gennady = Barista ∧
  profession Boris = Architect ∧
  profession Andrey = Veterinarian ∧
  profession Vyacheslav = Guitarist :=
by sorry

end final_professions_correct_l650_650266


namespace find_p0_over_q0_l650_650949

-- Definitions

def p (x : ℝ) := 3 * (x - 4) * (x - 2)
def q (x : ℝ) := (x - 4) * (x + 3)

theorem find_p0_over_q0 : (p 0) / (q 0) = -2 :=
by
  -- Prove the equality given the conditions
  sorry

end find_p0_over_q0_l650_650949


namespace jennifer_spent_124_dollars_l650_650430

theorem jennifer_spent_124_dollars 
  (initial_cans : ℕ := 40)
  (cans_per_set : ℕ := 5)
  (additional_cans_per_set : ℕ := 6)
  (total_cans_mark : ℕ := 30)
  (price_per_can_whole : ℕ := 2)
  (discount_threshold_whole : ℕ := 10)
  (discount_amount_whole : ℕ := 4) : 
  (initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) * price_per_can_whole - 
  (discount_amount_whole * ((initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) / discount_threshold_whole)) = 124 := by
  sorry

end jennifer_spent_124_dollars_l650_650430


namespace speed_ratio_is_2_l650_650493

def distance_to_work : ℝ := 20
def total_hours_on_road : ℝ := 6
def speed_back_home : ℝ := 10

theorem speed_ratio_is_2 :
  (∃ v : ℝ, (20 / v) + (20 / 10) = 6) → (10 = 2 * v) :=
by sorry

end speed_ratio_is_2_l650_650493


namespace num_bases_ending_in_1_l650_650761

theorem num_bases_ending_in_1 : 
  (∃ bases : Finset ℕ, 
  ∀ b ∈ bases, 3 ≤ b ∧ b ≤ 10 ∧ (625 % b = 1) ∧ bases.card = 4) :=
sorry

end num_bases_ending_in_1_l650_650761


namespace smallest_bdf_l650_650902

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end smallest_bdf_l650_650902


namespace largest_prime_factor_1729_l650_650539

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650539


namespace linear_function_quadrant_l650_650497

theorem linear_function_quadrant :
  ∀ (x y : ℝ), y = -2 * x + 1 → 
    (∃ q : ℝ, ¬ (q = 3) → 
      (∃ x, ∀ y, y ≠ -2 * x + 1)) :=
by
  intros x y h_linear
  use 3
  intros hneq_q
  use x
  intros hy
  contradiction
  sorry

end linear_function_quadrant_l650_650497


namespace remainder_when_divided_by_product_l650_650451

noncomputable def Q : Polynomial ℝ := sorry

theorem remainder_when_divided_by_product (Q : Polynomial ℝ)
    (h1 : Q.eval 20 = 100)
    (h2 : Q.eval 100 = 20) :
    ∃ R : Polynomial ℝ, ∃ a b : ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 100) * R + Polynomial.C a * Polynomial.X + Polynomial.C b ∧
    a = -1 ∧ b = 120 :=
by
  sorry

end remainder_when_divided_by_product_l650_650451


namespace sally_seashells_l650_650917

theorem sally_seashells 
  (seashells_monday : ℕ)
  (seashells_tuesday : ℕ)
  (price_per_seashell : ℝ)
  (h_monday : seashells_monday = 30)
  (h_tuesday : seashells_tuesday = seashells_monday / 2)
  (h_price : price_per_seashell = 1.2) :
  let total_seashells := seashells_monday + seashells_tuesday in
  let total_money := total_seashells * price_per_seashell in
  total_money = 54 := 
by
  sorry

end sally_seashells_l650_650917


namespace part1_part2_l650_650088

-- Part (1) prove maximum value of 4 - 2x - 1/x when x > 0 is 0.
theorem part1 (x : ℝ) (h : 0 < x) : 
  4 - 2 * x - (2 / x) ≤ 0 :=
sorry

-- Part (2) prove minimum value of 1/a + 1/b when a + 2b = 1 and a > 0, b > 0 is 3 + 2 * sqrt 2.
theorem part2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 1) :
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end part1_part2_l650_650088


namespace simplify_cuberoot_exponents_l650_650003

theorem simplify_cuberoot_exponents {a b c : ℝ} :
  let expr := (72 * (a ^ 5) * (b ^ 7) * (c ^ 14))
  let simplified_expr := (2 * 3^(2/3)) * a * b * (c ^ 4) * (expr^(1/3))
  (∑ v in [1, 1, 4], v) = 6 := by
  sorry

end simplify_cuberoot_exponents_l650_650003


namespace possible_integer_roots_l650_650490

theorem possible_integer_roots (b c d e f: ℤ) : 
  let p := polynomial.C f + polynomial.C e * polynomial.X + polynomial.C d * polynomial.X^2 +
            polynomial.C c * polynomial.X^3 + polynomial.C b * polynomial.X^4 +
            polynomial.X^5 in
  let n := (polynomial.roots p).count (λ r, r.is_integer) in
  n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 5 := sorry

end possible_integer_roots_l650_650490


namespace sum_paths_length_2x2_grid_l650_650945

theorem sum_paths_length_2x2_grid : 
  let grid := (2, 2)
  let lower_left := (0, 0)
  let upper_right := (2, 2)
  let valid_paths_length_sum := 52
  in sum_of_all_paths_lengths grid lower_left upper_right = valid_paths_length_sum :=
sorry

end sum_paths_length_2x2_grid_l650_650945


namespace sum_of_three_integers_l650_650964

theorem sum_of_three_integers (a b c : ℕ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diff: a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_prod: a * b * c = 5^4) : a + b + c = 131 :=
sorry

end sum_of_three_integers_l650_650964


namespace number_of_digits_l650_650191

-- Define the expression
def expr : ℕ := 3^10 * 5^6

-- Define the function to count the number of digits
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log10 (n) + 1

-- Statement to prove the number of digits of the expression
theorem number_of_digits : num_digits expr = 9 := by
  sorry

end number_of_digits_l650_650191


namespace final_professions_correct_l650_650269

-- Define the people and their positions
inductive Person
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq, Inhabited

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq, Inhabited

open Person Profession

-- Given positions in seats
def seatingArrangement : Person → Nat := 
λ p, 
  match p with
  | Andrey    => 3
  | Boris     => 2
  | Vyacheslav => 4
  | Gennady   => 1

-- Given professions
def profession : Person → Profession := 
λ p, 
  match p with
  | Andrey    => Veterinarian
  | Boris     => Architect
  | Vyacheslav => Guitarist
  | Gennady   => Barista

-- Define the conditions
def condition1 : Prop :=
  (seatingArrangement Andrey = 3) ∧ 
  (seatingArrangement Boris = 2) ∧
  (seatingArrangement Vyacheslav = 4) ∧ 
  (seatingArrangement Gennady = 1)

def condition2 : Prop := 
  seatingArrangement (Person.Barista) + 1 = seatingArrangement Boris

def condition3 : Prop := 
  seatingArrangement Vyacheslav > seatingArrangement Andrey ∧ 
  seatingArrangement Vyacheslav > seatingArrangement Boris

def condition4 : Prop := 
  seatingArrangement (Person.Veterinarian) ≠ 1 ∧ seatingArrangement (Person.Veterinarian) ≠ 4

def condition5 : Prop := 
  (seatingArrangement (Person.Barista) ≠ seatingArrangement (Person.Guitarist) + 1 ∧
  seatingArrangement (Person.Guitarist) ≠ seatingArrangement (Person.Barista) + 1)

-- Prove the final professions
theorem final_professions_correct :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  profession Gennady = Barista ∧
  profession Boris = Architect ∧
  profession Andrey = Veterinarian ∧
  profession Vyacheslav = Guitarist :=
by sorry

end final_professions_correct_l650_650269


namespace three_distinct_real_roots_of_g_l650_650870

noncomputable def g (d : ℝ) : ℝ → ℝ := λ x, x^2 - 4 * x + d

theorem three_distinct_real_roots_of_g (d : ℝ) :
  (∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ g d (g d x1) = 0 ∧ g d (g d x2) = 0 ∧ g d (g d x3) = 0) ↔ d = 3 := sorry

end three_distinct_real_roots_of_g_l650_650870


namespace large_beds_l650_650326

theorem large_beds {L : ℕ} {M : ℕ} 
    (h1 : M = 2) 
    (h2 : ∀ (x : ℕ), 100 <= x → L = (320 - 60 * M) / 100) : 
  L = 2 :=
by
  sorry

end large_beds_l650_650326


namespace largest_prime_factor_of_1729_l650_650563

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650563


namespace number_of_teams_l650_650843

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end number_of_teams_l650_650843


namespace average_weight_of_boys_l650_650083

theorem average_weight_of_boys 
  (n1 n2 : ℕ) 
  (w1 w2 : ℝ) 
  (h1 : n1 = 22) 
  (h2 : n2 = 8) 
  (h3 : w1 = 50.25) 
  (h4 : w2 = 45.15) : 
  (n1 * w1 + n2 * w2) / (n1 + n2) = 48.89 :=
by
  sorry

end average_weight_of_boys_l650_650083


namespace largest_prime_factor_1729_l650_650622

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650622


namespace odd_function_fixed_points_odd_even_function_fixed_points_not_necessarily_even_l650_650023

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f (x)
def is_fixed_point (f : ℝ → ℝ) (c : ℝ) : Prop := f (c) = c

-- Proposition 1: Odd function
theorem odd_function_fixed_points_odd (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) (h_finite : ∃ n : ℕ, ∀ c : ℝ, is_fixed_point f c → c = (list.nth_le (list.range n) sorry sorry)) :
  ∃ k : ℕ, 2 * k + 1 = list.length (list.filter (λ x, is_fixed_point f x) (list.range sorry)) :=
sorry

-- Proposition 2: Even function
theorem even_function_fixed_points_not_necessarily_even (f : ℝ → ℝ) 
  (h_even : is_even_function f) (h_finite : ∃ n : ℕ, ∀ c : ℝ, is_fixed_point f c → c = (list.nth_le (list.range n) sorry sorry)) :
  ¬∀ k : ℕ, 2 * k = list.length (list.filter (λ x, is_fixed_point f x) (list.range sorry)) :=
sorry

end odd_function_fixed_points_odd_even_function_fixed_points_not_necessarily_even_l650_650023


namespace second_tap_fills_in_15_hours_l650_650057

theorem second_tap_fills_in_15_hours 
  (r1 r3 : ℝ) 
  (x : ℝ) 
  (H1 : r1 = 1 / 10) 
  (H2 : r3 = 1 / 6) 
  (H3 : r1 + 1 / x + r3 = 1 / 3) : 
  x = 15 :=
sorry

end second_tap_fills_in_15_hours_l650_650057


namespace problem_statement_l650_650746

theorem problem_statement (h : ∀ x y : ℝ, x < y → (0.3 : ℝ)^x > (0.3 : ℝ)^y) : (0.3 : ℝ)^(-0.4) < (0.3 : ℝ)^(-0.5) :=
by
  have h_monotone : ∀ x y : ℝ, x < y → (0.3 : ℝ)^x > (0.3 : ℝ)^y := by exact h
  sorry

end problem_statement_l650_650746


namespace ratio_copper_to_zinc_l650_650165

theorem ratio_copper_to_zinc (copper zinc : ℝ) (hc : copper = 24) (hz : zinc = 10.67) : (copper / zinc) = 2.25 :=
by
  rw [hc, hz]
  -- Add the arithmetic operation
  sorry

end ratio_copper_to_zinc_l650_650165


namespace repeating_decimal_sum_l650_650650

theorem repeating_decimal_sum : 
  let x := 0.123123 in
  let frac := 41 / 333 in
  sum_n_d (frac) = 374 :=
begin
  sorry
end

end repeating_decimal_sum_l650_650650


namespace half_circle_locus_l650_650019

noncomputable theory

open Complex Real

def locus_of_omega (θ : ℝ) : Set ℂ :=
  let z : ℂ := Complex.ofReal (cos θ) - Complex.I * (Complex.ofReal (sin θ) - 1)
  let ω : ℂ := z ^ 2 - 2 * Complex.I * z
  {ω | ∃ (θ : ℝ), θ ∈ set.Ioc (π / 2) π ∧ ω = z ^ 2 - 2 * Complex.I * z}

theorem half_circle_locus :
  ∀ (θ : ℝ), θ ∈ set.Ioc (π / 2) π →
  let z : ℂ := Complex.ofReal (cos θ) - Complex.I * (Complex.ofReal (sin θ) - 1)
  let ω : ℂ := z ^ 2 - 2 * Complex.I * z
  ∃ x y : ℝ, x = cos (2 * θ) + 1 ∧ y = -sin (2 * θ) ∧ x > 0 ∧ y > 0 ∧ (x - 1) ^ 2 + y ^ 2 = 1 :=
sorry

end half_circle_locus_l650_650019


namespace length_segment_in_cube_4_l650_650229

noncomputable def X : ℝ × ℝ × ℝ := (0, 0, 0)
noncomputable def Y : ℝ × ℝ × ℝ := (5, 5, 14)

-- Cube 1 extends from (0,0,0) to (2,2,2)
-- Cube 2 extends from (0,0,2) to (3,3,5)
-- Cube 3 extends from (0,0,5) to (4,4,9)
-- Cube 4 extends from (0,0,9) to (5,5,14)

-- Prove that the length of the segment of XY inside the cube with edge length 4 (extending from (0,0,5) to (4,4,9)) is 4 * sqrt 3

theorem length_segment_in_cube_4 : 
  let X := (0 : ℝ, 0 : ℝ, 0 : ℝ) in
  let Y := (5 : ℝ, 5 : ℝ, 14 : ℝ) in
  let cube_3_start := (0 : ℝ, 0 : ℝ, 5 : ℝ) in
  let cube_3_end := (4 : ℝ, 4 : ℝ, 9 : ℝ) in
  let distance := (p1 p2 : ℝ × ℝ × ℝ) -> (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2 in
  distance X cube_3_start = 4^2 * 3 :=
sorry

end length_segment_in_cube_4_l650_650229


namespace largest_prime_factor_1729_l650_650618

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650618


namespace min_value_expression_l650_650879

-- Let x and y be positive integers such that x^2 + y^2 - 2017 * x * y > 0 and it is not a perfect square.
theorem min_value_expression (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_not_square : ¬ ∃ z : ℕ, (x^2 + y^2 - 2017 * x * y) = z^2) :
  x^2 + y^2 - 2017 * x * y > 0 → ∃ k : ℕ, k = 2019 ∧ ∀ m : ℕ, (m > 0 → ¬ ∃ z : ℤ, (x^2 + y^2 - 2017 * x * y) = z^2 ∧ x^2 + y^2 - 2017 * x * y < k) :=
sorry

end min_value_expression_l650_650879


namespace smallest_consecutive_natural_number_sum_l650_650053

theorem smallest_consecutive_natural_number_sum (a n : ℕ) (hn : n > 1) (h : n * a + (n * (n - 1)) / 2 = 2016) :
  ∃ a, a = 1 :=
by
  sorry

end smallest_consecutive_natural_number_sum_l650_650053


namespace fermat_point_minimizes_sum_distances_l650_650780

theorem fermat_point_minimizes_sum_distances 
  (A B C M : Point)
  (h_triangle : ∀ α β γ, α + β + γ = 180 ∧ α < 120 ∧ β < 120 ∧ γ < 120)
  (h_inside : inside_triangle M A B C)
  (h_fermat : ∀ α β γ, α = 120 ∧ β = 120 ∧ γ = 120) :
  ∃ M, (dist M A + dist M B + dist M C) = minimized_distance A B C := 
sorry

end fermat_point_minimizes_sum_distances_l650_650780


namespace number_of_children_bikes_l650_650171

theorem number_of_children_bikes (c : ℕ) 
  (regular_bikes : ℕ) (wheels_per_regular_bike : ℕ) 
  (wheels_per_children_bike : ℕ) (total_wheels : ℕ)
  (h1 : regular_bikes = 7) 
  (h2 : wheels_per_regular_bike = 2) 
  (h3 : wheels_per_children_bike = 4) 
  (h4 : total_wheels = 58) 
  (h5 : total_wheels = (regular_bikes * wheels_per_regular_bike) + (c * wheels_per_children_bike)) 
  : c = 11 :=
by
  sorry

end number_of_children_bikes_l650_650171


namespace books_on_shelf_A_eq_160_l650_650051

-- Define the original problem as a theorem
theorem books_on_shelf_A_eq_160 (x y : ℕ) (h1 : x + y = 280) (h2 : y + 0.125 * x = 140) : x = 160 :=
sorry

end books_on_shelf_A_eq_160_l650_650051


namespace expected_first_sequence_100_l650_650145

noncomputable def expected_first_sequence (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1).filter (λ k, k > 0), (1 : ℝ) / k

theorem expected_first_sequence_100 : expected_first_sequence 100 = 
    ∑ k in (finset.range 101).filter (λ k, k > 0), (1 : ℝ) / k :=
by
  -- The proof would involve showing this sum represents the harmonic number H_100
  sorry

end expected_first_sequence_100_l650_650145


namespace line_bisects_segment_l650_650064

open_locale classical

variables {K : Type*} [euclidean_domain K]

structure Circle (K : Type*) :=
(center : K × K)
(radius : K)

def Point_on_circle (c : Circle K) (P : K × K) : Prop :=
let (x, y) := c.center in
(P.fst - x)^2 + (P.snd - y)^2 = c.radius^2

variables (A B M N : K × K)
variables (c₁ c₂ : Circle K)

-- Two circles intersect at points A and B
axiom circles_intersect : Point_on_circle c₁ A ∧ Point_on_circle c₂ A ∧ Point_on_circle c₁ B ∧ Point_on_circle c₂ B

-- MN is a common tangent to both circles
axiom common_tangent : ∃ M N : K × K, (M.fst - N.fst)^2 + (M.snd - N.snd)^2 ≠ 0 

-- Prove that the line AB bisects segment MN
theorem line_bisects_segment :
  let O := (M.fst + N.fst) / 2 in
  let OM := (O - M).fst^2 + (O - M).snd^2 in
  let ON := (O - N).fst^2 + (O - N).snd^2 in
  OM = ON :=
begin
  sorry
end

end line_bisects_segment_l650_650064


namespace player_A_wins_if_and_only_if_odd_l650_650912

theorem player_A_wins_if_and_only_if_odd (n : ℕ) (hn : n > 0) : 
  (∃ m : ℕ, n = 2 * m + 1) ↔ player_A_has_winning_strategy n :=
sorry

def player_A_has_winning_strategy (n : ℕ) : Prop :=
  -- Definition based on the given conditions and problem setup
  ∃ strategy : (ℕ → option ℕ), ∀ k : ℕ, 0 < k → k ≤ n → 
    ∃ l : ℕ, k = l ∨ (l < k ∧ gcd l k = 1 ∧ option.is_some (strategy (k - l))) ∧ 
    -- more strategy conditions satisfying the winning strategy for player A
    true

end player_A_wins_if_and_only_if_odd_l650_650912


namespace number_of_factors_180_l650_650359

theorem number_of_factors_180 : 
  let factors180 (n : ℕ) := 180 
  in factors180 180 = (2 + 1) * (2 + 1) * (1 + 1) => factors180 180 = 18 := 
by
  sorry

end number_of_factors_180_l650_650359


namespace divisors_of_m_squared_l650_650118

theorem divisors_of_m_squared {m : ℕ} (h₁ : ∀ d, d ∣ m → d = 1 ∨ d = m ∨ prime d) (h₂ : nat.divisors m = 4) :
  (nat.divisors (m ^ 2) = 7 ∨ nat.divisors (m ^ 2) = 9) :=
sorry

end divisors_of_m_squared_l650_650118


namespace g_200_eq_60_l650_650724

def g : ℕ → ℕ
| x := if ∃ (n : ℕ), 2^n = x then log2 x
       else 2 + g (x + 2)

-- Statement to prove
theorem g_200_eq_60 : g 200 = 60 := 
by 
  sorry

end g_200_eq_60_l650_650724


namespace largest_prime_factor_of_1729_l650_650644

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650644


namespace percentage_fruits_in_good_condition_l650_650661

theorem percentage_fruits_in_good_condition (total_oranges total_bananas : ℕ)
    (rotten_oranges_percentage rotten_bananas_percentage : ℝ)
    (h_oranges : total_oranges = 600)
    (h_bananas : total_bananas = 400)
    (h_rotten_oranges_percentage : rotten_oranges_percentage = 0.15)
    (h_rotten_bananas_percentage : rotten_bananas_percentage = 0.04) :
    let total_fruits := total_oranges + total_bananas
    let rotten_oranges := (rotten_oranges_percentage * total_oranges : ℝ)
    let rotten_bananas := (rotten_bananas_percentage * total_bananas : ℝ)
    let rotten_fruits := rotten_oranges + rotten_bananas
    let good_fruits := total_fruits - rotten_fruits.to_nat
    let percentage_good_fruits := (good_fruits : ℝ) / total_fruits * 100
  in percentage_good_fruits = 89.4 :=
by
  sorry

end percentage_fruits_in_good_condition_l650_650661


namespace smallest_angle_l650_650755

theorem smallest_angle (x : ℝ) (hx : x > 0) (h : sin (4 * x) * sin (5 * x) = cos (4 * x) * cos (5 * x)) : 
  x = 10 := 
by 
  sorry

end smallest_angle_l650_650755


namespace range_of_a_l650_650828

theorem range_of_a (a : ℝ) :
  let Z := (a - 1 + 2 * a * complex.i) / (1 - complex.i) in
  (Z.re < 0 ∧ Z.im > 0) → a > 1 / 3 := 
by 
  sorry

end range_of_a_l650_650828


namespace largest_prime_factor_of_1729_l650_650628

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650628


namespace find_103rd_digit_l650_650830

-- Definition of the sequence
def sequence : List Char :=
  "908988878685848382818079787776757473727170696867666564636261605958575655545352515049484746454443424140393837363534333231302928272625242322212019181716151413121110987654321".toList

-- Define a function to find the n-th digit in the sequence
def nth_digit (n : ℕ) : Char :=
  sequence.get ⟨n - 1, by simp [sequence]⟩

-- The condition states we need to find the 103rd digit
theorem find_103rd_digit : nth_digit 103 = '3' := by
  sorry

end find_103rd_digit_l650_650830


namespace can_adjust_to_357_l650_650465

structure Ratio (L O V : ℕ) :=
(lemon : ℕ)
(oil : ℕ)
(vinegar : ℕ)

def MixA : Ratio 1 2 3 := ⟨1, 2, 3⟩
def MixB : Ratio 3 4 5 := ⟨3, 4, 5⟩
def TargetC : Ratio 3 5 7 := ⟨3, 5, 7⟩

theorem can_adjust_to_357 (x y : ℕ) (hA : x * MixA.lemon + y * MixB.lemon = 3 * (x + y))
    (hO : x * MixA.oil + y * MixB.oil = 5 * (x + y))
    (hV : x * MixA.vinegar + y * MixB.vinegar = 7 * (x + y)) :
    (∃ a b : ℕ, x = 3 * a ∧ y = 2 * b) :=
sorry

end can_adjust_to_357_l650_650465


namespace find_constants_of_sine_function_l650_650172

theorem find_constants_of_sine_function
  (a b c : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_max : ∃ x : ℝ, x = π/3 ∧ a * sin(b * x + c) = 3) :
  a = 3 ∧ c = π / 6 := by
  sorry

end find_constants_of_sine_function_l650_650172


namespace lambda_tangency_l650_650062

def is_tangent (line curve : ℝ → ℝ → ℝ) (λ : ℝ) : Prop :=
  ∀ x y : ℝ, (line x y = λ) → ∃ a, b : ℝ, (curve x y = 0) ∧ (a * x + b * y = 0)

noncomputable def translated_line (x y λ : ℝ) : ℝ :=
  (x + 1) - 2 * (y + 2) + λ - 3

def curve (x y : ℝ) : ℝ :=
  x^2 + y^2 + 2 * x - 4 * y

-- The theorem we aim to prove
theorem lambda_tangency :
  ∃ (λ : ℝ), (λ = 13 ∨ λ = 3) ∧ is_tangent translated_line curve λ :=
sorry

end lambda_tangency_l650_650062


namespace count100DigitEvenNumbers_is_correct_l650_650338

noncomputable def count100DigitEvenNumbers : ℕ :=
  let validDigits : Finset ℕ := {0, 1, 3}
  let firstDigitChoices : ℕ := 2  -- Only 1 or 3
  let middleDigitsChoices : ℕ := 3 ^ 98  -- 3 choices for each of the 98 middle positions
  let lastDigitChoices : ℕ := 1  -- Only 0 (even number requirement)
  firstDigitChoices * middleDigitsChoices * lastDigitChoices

theorem count100DigitEvenNumbers_is_correct :
  count100DigitEvenNumbers = 2 * 3 ^ 98 := by
  sorry

end count100DigitEvenNumbers_is_correct_l650_650338


namespace csc_inequality_l650_650844

variables {A B C : ℝ} -- angles of the triangle
variables {s r : ℝ} -- semi-perimeter and incircle radius
variables (h_triangle : A + B + C = π) (h_s : s = (A + B + C) / 2) (h_r : r > 0)

theorem csc_inequality (h_triangle : A + B + C = π) (h_s : s = (A + B + C) / 2) (h_r : r > 0) :
  (real.csc (A / 4) + real.csc (B / 4) + real.csc (C / 4)) > (2 * s) / r + (real.pi) / 3 :=
sorry

end csc_inequality_l650_650844


namespace total_cost_of_shirt_and_sweater_l650_650509

theorem total_cost_of_shirt_and_sweater (S : ℝ) : 
  (S - 7.43 = 36.46) → (36.46 + S = 80.35) :=
by
  assume h1 : S - 7.43 = 36.46
  sorry

end total_cost_of_shirt_and_sweater_l650_650509


namespace a5_value_l650_650322

def sequence (n : ℕ) : ℚ :=
  if n = 0 then -2
  else 2 + (2 * sequence (n - 1)) / (1 - sequence (n - 1))

theorem a5_value : sequence 5 = 10 / 7 := by
  sorry

end a5_value_l650_650322


namespace sally_seashells_l650_650918

theorem sally_seashells 
  (seashells_monday : ℕ)
  (seashells_tuesday : ℕ)
  (price_per_seashell : ℝ)
  (h_monday : seashells_monday = 30)
  (h_tuesday : seashells_tuesday = seashells_monday / 2)
  (h_price : price_per_seashell = 1.2) :
  let total_seashells := seashells_monday + seashells_tuesday in
  let total_money := total_seashells * price_per_seashell in
  total_money = 54 := 
by
  sorry

end sally_seashells_l650_650918


namespace selection_probability_equal_l650_650272

theorem selection_probability_equal :
  let n := 2012
  let eliminated := 12
  let remaining := n - eliminated
  let selected := 50
  let probability := (remaining / n) * (selected / remaining)
  probability = 25 / 1006 :=
by
  sorry

end selection_probability_equal_l650_650272


namespace new_ratio_second_term_l650_650125

theorem new_ratio_second_term (x : ℕ) (h : x = 29) : (15 + x) = 44 :=
by {
  rw h,
  norm_num,
}

end new_ratio_second_term_l650_650125


namespace nth_derivative_ln_correct_l650_650396

noncomputable def nth_derivative_ln (n : ℕ) : ℝ → ℝ
| x => (-1)^(n-1) * (Nat.factorial (n-1)) / (1 + x) ^ n

theorem nth_derivative_ln_correct (n : ℕ) (x : ℝ) :
  deriv^[n] (λ x => Real.log (1 + x)) x = nth_derivative_ln n x := 
by
  sorry

end nth_derivative_ln_correct_l650_650396


namespace greatest_possible_x_l650_650955

-- Define the numbers and the lcm condition
def num1 := 12
def num2 := 18
def lcm_val := 108

-- Function to calculate the lcm of three numbers
def lcm3 (a b c : ℕ) := Nat.lcm (Nat.lcm a b) c

-- Proposition stating the problem condition
theorem greatest_possible_x (x : ℕ) (h : lcm3 x num1 num2 = lcm_val) : x ≤ lcm_val := sorry

end greatest_possible_x_l650_650955


namespace trail_length_l650_650066

variable (v : ℝ) (t : ℝ) (L : ℝ)

-- Friend P's speed is 20% faster than Friend Q's speed
def friend_p_speed := 1.2 * v

-- Friend P will have walked 12 kilometers when they pass each other
def friend_p_distance := friend_p_speed * t = 12

-- Friend Q's distance covered when they meet
def friend_q_distance := v * t = L - 12

theorem trail_length :
  (friend_p_distance v t) → (friend_q_distance v t L) → L = 22 := by
  sorry

end trail_length_l650_650066


namespace students_in_activities_l650_650154

theorem students_in_activities (n : ℕ) (a : set (finset ℕ)) :
  a.card = 20 →
  ∀ x ∈ a, x.card = 5 →
  ∀ (x1 x2 ∈ a), x1 ∩ x2 ≠ ∅ → x1 = x2 →
  n ≥ 21 :=
by sorry 

end students_in_activities_l650_650154


namespace volume_to_surface_area_ratio_l650_650697

-- Define the shape as described in the problem
structure Shape :=
(center_cube : ℕ)  -- Center cube
(surrounding_cubes : ℕ)  -- Surrounding cubes
(unit_volume : ℕ)  -- Volume of each unit cube
(unit_face_area : ℕ)  -- Surface area of each face of the unit cube

-- Conditions and definitions
def is_special_shape (s : Shape) : Prop :=
  s.center_cube = 1 ∧ s.surrounding_cubes = 7 ∧ s.unit_volume = 1 ∧ s.unit_face_area = 1

-- Theorem statement
theorem volume_to_surface_area_ratio (s : Shape) (h : is_special_shape s) : (s.center_cube + s.surrounding_cubes) * s.unit_volume / (s.surrounding_cubes * 5 * s.unit_face_area) = 8 / 35 :=
by
  sorry

end volume_to_surface_area_ratio_l650_650697


namespace similar_triangles_proportion_l650_650063

theorem similar_triangles_proportion :
  ∀ (P Q R X Y Z : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace X] [MetricSpace Y] [MetricSpace Z],
  -- Conditions: Given the side lengths
  (PQ QR YZ XY : ℝ)
  -- Similarity condition
  (h_sim: triangle_similar P Q R X Y Z)
  -- Given lengths
  (h_PQ : PQ = 9)
  (h_QR : QR = 18)
  (h_YZ : YZ = 27),
  XY = 13.5 :=
by
  sorry

end similar_triangles_proportion_l650_650063


namespace root_interval_l650_650952

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - 1 / x

theorem root_interval (h₁ : f 1 = exp 1 - 1 > 0)
                      (h₂ : f (1 / 2) = sqrt (exp 1) - 2 < 0)
                      (h₃ : ∀ x > 0, deriv f x = exp x + 1 / (x ^ 2) > 0)
                      : ∃ c ∈ Ioo (1 / 2) 1, f c = 0 :=
by {
  sorry
}

end root_interval_l650_650952


namespace verify_mass_percentage_l650_650210

-- Define the elements in HBrO3
def hydrogen : String := "H"
def bromine : String := "Br"
def oxygen : String := "O"

-- Define the given molar masses
def molar_masses (e : String) : Float :=
  if e = hydrogen then 1.01
  else if e = bromine then 79.90
  else if e = oxygen then 16.00
  else 0.0

-- Define the molar mass of HBrO3
def molar_mass_HBrO3 : Float := 128.91

-- Function to calculate mass percentage of a given element in HBrO3
def mass_percentage (e : String) : Float :=
  if e = bromine then 79.90 / molar_mass_HBrO3 * 100
  else if e = hydrogen then 1.01 / molar_mass_HBrO3 * 100
  else if e = oxygen then 48.00 / molar_mass_HBrO3 * 100
  else 0.0

-- The proof problem statement
theorem verify_mass_percentage (e : String) (h : e ∈ [hydrogen, bromine, oxygen]) : mass_percentage e = 0.78 :=
sorry

end verify_mass_percentage_l650_650210


namespace num_factors_180_l650_650360

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l650_650360


namespace ellipse_foci_coordinates_l650_650020

-- Define the equation of the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  y^2 / 9 + x^2 / 4 = 1

-- Define the semi-major axis a and semi-minor axis b of the ellipse
def semi_major_axis : ℝ := 3
def semi_minor_axis : ℝ := 2

-- Define the value of c based on the ellipse properties.
noncomputable def foci_distance : ℝ := Real.sqrt (semi_major_axis^2 - semi_minor_axis^2)

-- Theorem stating the coordinates of the foci of the given ellipse
theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, ellipse_eq x y) → 
  (foci_distance = Real.sqrt 5) ∧ 
  (∀ c, c = foci_distance → 
       (∀ (x y : ℝ), ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) :=
begin
  sorry
end

end ellipse_foci_coordinates_l650_650020


namespace largest_prime_factor_of_1729_l650_650634

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650634


namespace cylindrical_to_rectangular_l650_650186

-- Define the cylindrical coordinates
def r : ℝ := 4
def θ : ℝ := Real.pi / 2
def z_cyl : ℝ := 3

-- Define the conversion formulas
def x : ℝ := r * Real.cos θ
def y : ℝ := r * Real.sin θ
def z_rect : ℝ := z_cyl

-- Prove the conversion to rectangular coordinates
theorem cylindrical_to_rectangular : (x, y, z_rect) = (0, 4, 3) := by
  sorry

end cylindrical_to_rectangular_l650_650186


namespace negation_of_proposition_p_l650_650913

def has_real_root (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

def negation_of_p : Prop := ∀ m : ℝ, ¬ has_real_root m

theorem negation_of_proposition_p : negation_of_p :=
by sorry

end negation_of_proposition_p_l650_650913


namespace minimum_surface_area_of_circumscribed_sphere_of_prism_l650_650419

theorem minimum_surface_area_of_circumscribed_sphere_of_prism :
  ∃ S : ℝ, 
    (∀ h r, r^2 * h = 4 → r^2 + (h^2 / 4) = R → 4 * π * R^2 = S) ∧ 
    (∀ S', S' ≤ S) ∧ 
    S = 12 * π :=
sorry

end minimum_surface_area_of_circumscribed_sphere_of_prism_l650_650419


namespace largest_prime_factor_1729_l650_650525

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650525


namespace quadratic_equation_with_roots_sum_and_difference_l650_650216

theorem quadratic_equation_with_roots_sum_and_difference (p q : ℚ)
  (h1 : p + q = 10)
  (h2 : abs (p - q) = 2) :
  (Polynomial.eval₂ (RingHom.id ℚ) p (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) ∧
  (Polynomial.eval₂ (RingHom.id ℚ) q (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) :=
by sorry

end quadratic_equation_with_roots_sum_and_difference_l650_650216


namespace largest_prime_factor_1729_l650_650529

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650529


namespace find_function_expression_l650_650833

noncomputable def f (a b x : ℝ) : ℝ := 2 ^ (a * x + b)

theorem find_function_expression
  (a b : ℝ)
  (h1 : f a b 1 = 2)
  (h2 : ∃ g : ℝ → ℝ, (∀ x y : ℝ, f (-a) (-b) x = y ↔ f a b y = x) ∧ g (f a b 1) = 1) :
  ∃ (a b : ℝ), f a b x = 2 ^ (-x + 2) :=
by
  sorry

end find_function_expression_l650_650833


namespace sum_of_reciprocals_of_lcms_lt_four_l650_650863

theorem sum_of_reciprocals_of_lcms_lt_four (n : ℕ) (a : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) (h_positive : ∀ i : Fin n, a i > 0) :
  (∑ k in Finset.range n, 1 / Real.ofNat (Nat.lcm_list (List.ofFn (a ∘ Fin.val.index k)))) < 4 :=
sorry

end sum_of_reciprocals_of_lcms_lt_four_l650_650863


namespace focus_of_parabola_l650_650281

noncomputable def parabola_focus_coordinates (y : ℝ → ℝ) : ℝ × ℝ :=
  let a := 1 / 8 in (0, 1 / (4 * (1 / (4 * a))))

theorem focus_of_parabola : parabola_focus_coordinates (λ x, 8 * x ^ 2) = (0, 1 / 32) := by
  sorry

end focus_of_parabola_l650_650281


namespace necessary_condition_for_inequality_l650_650412

-- Definitions based on the conditions in a)
variables (A B C D : ℝ)

-- Main statement translating c) into Lean
theorem necessary_condition_for_inequality (h : C < D) : A > B :=
by sorry

end necessary_condition_for_inequality_l650_650412


namespace divisors_of_m_squared_l650_650120

theorem divisors_of_m_squared {m : ℕ} (h₁ : ∀ d, d ∣ m → d = 1 ∨ d = m ∨ prime d) (h₂ : nat.divisors m = 4) :
  (nat.divisors (m ^ 2) = 7 ∨ nat.divisors (m ^ 2) = 9) :=
sorry

end divisors_of_m_squared_l650_650120


namespace second_fish_length_l650_650688

-- Defining the conditions
def first_fish_length : ℝ := 0.3
def length_difference : ℝ := 0.1

-- Proof statement
theorem second_fish_length : ∀ (second_fish : ℝ), first_fish_length = second_fish + length_difference → second_fish = 0.2 :=
by 
  intro second_fish
  intro h
  sorry

end second_fish_length_l650_650688


namespace remainder_97_25_mod_50_l650_650646

theorem remainder_97_25_mod_50 :
  (97 ^ 25) % 50 = 7 :=
by
  -- assuming 97 ≡ -3 (mod 50)
  have h1 : 97 % 50 = -3 % 50, by norm_num
  -- we need to prove (97 ^ 25) % 50 = 7
  have h2: (97^25) % 50 = ((-3)^25) % 50, sorry
  -- simplifying (-3)^25 modulo 50
  have h3: ((-3)^25) % 50 = 7, sorry
  show (97 ^ 25) % 50 = 7, from
    by 
      rw [h2, h3]
      exact h3

end remainder_97_25_mod_50_l650_650646


namespace smallest_product_bdf_l650_650904

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end smallest_product_bdf_l650_650904


namespace factors_of_180_l650_650368

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l650_650368


namespace find_n_l650_650689

theorem find_n : ∃ n : ℕ, 50^4 + 43^4 + 36^4 + 6^4 = n^4 := by
  sorry

end find_n_l650_650689


namespace count_valid_integers_l650_650817

def is_valid_integer (n : ℕ) : Prop :=
  3000 < n ∧ n < 4010 ∧ 
  let d := n % 10 in
  let abc := n / 10 in
  d = (abc % 10) + (abc / 10 % 10) + (abc / 100) ∧
  n % 3 = 0

theorem count_valid_integers : 
  (Finset.card (Finset.filter is_valid_integer (Finset.Ico 3001 4010))) = 12 :=
begin
  sorry
end

end count_valid_integers_l650_650817


namespace largest_prime_factor_of_1729_l650_650636

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650636


namespace at_least_one_did_not_land_stably_l650_650736

-- Define the propositions p and q
variables (p q : Prop)

-- Define the theorem to prove
theorem at_least_one_did_not_land_stably :
  (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by
  sorry

end at_least_one_did_not_land_stably_l650_650736


namespace ilya_running_speed_is_15_l650_650925

-- Definitions based on given conditions
def total_distance : ℝ := 600 -- in meters
def sasha_running_speed : ℝ := 10 * 1000 / 3600 -- converting km/h to m/s
def walking_speed : ℝ := 5 * 1000 / 3600 -- converting km/h to m/s

-- Sasha runs half the time and walks the other half
def sasha_run_time : ℝ := (total_distance / 2) / sasha_running_speed
def sasha_walk_time : ℝ := (total_distance / 2) / walking_speed
def sasha_total_time : ℝ := sasha_run_time + sasha_walk_time

-- Ilya runs half the distance and walks the other half
def ilya_walk_distance : ℝ := total_distance / 2 -- meters
def ilya_run_distance : ℝ := total_distance / 2 -- meters

-- Proving Ilya's running speed is 15 km/h
def ilya_running_speed : Prop :=
  sasha_total_time = (ilya_walk_distance / walking_speed) + (ilya_run_distance / ?)

theorem ilya_running_speed_is_15: ilya_running_speed := 
by 
  -- Sasha's total time is known
  have run_time := sasha_total_time 
  -- Plugging in to solve for Ilya's running speed in km/h
  have ilya_run_time := run_time - (ilya_walk_distance / walking_speed)
  have speed := ilya_run_distance / ilya_run_time
  show speed = 15 * 1000 / 3600 -- converting 15 km/h to m/s
  sorry

end ilya_running_speed_is_15_l650_650925


namespace total_amount_shared_l650_650678

-- Define the initial conditions
def ratioJohn : ℕ := 2
def ratioJose : ℕ := 4
def ratioBinoy : ℕ := 6
def JohnShare : ℕ := 2000
def partValue : ℕ := JohnShare / ratioJohn

-- Define the shares based on the ratio and part value
def JoseShare := ratioJose * partValue
def BinoyShare := ratioBinoy * partValue

-- Prove the total amount shared is Rs. 12000
theorem total_amount_shared : (JohnShare + JoseShare + BinoyShare) = 12000 :=
  by
  sorry

end total_amount_shared_l650_650678


namespace area_of_bounded_region_eq_l650_650751

theorem area_of_bounded_region_eq :
  ∀ x y : ℝ, (y = 2 * |x|) ∧ (x^2 + y^2 = 9) ∧ (y = 3) → 
  (∫ (y in 0..3), ∫ (x in -sqrt(9 - y^2) .. sqrt(9 - y^2)), 1 dx dy) = 9 * real.arctan(2) :=
by
  intro x y h
  sorry

end area_of_bounded_region_eq_l650_650751


namespace expected_first_sequence_length_is_harmonic_100_l650_650128

open Real

-- Define the harmonic number
def harmonic_number (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (i + 1 : ℝ)

-- Define the expected length of the first sequence
def expected_first_sequence_length (n : ℕ) : ℝ :=
  harmonic_number n

-- Prove that the expected number of suitors in the first sequence is the 100th harmonic number
theorem expected_first_sequence_length_is_harmonic_100 :
  expected_first_sequence_length 100 = harmonic_number 100 :=
by
  sorry

end expected_first_sequence_length_is_harmonic_100_l650_650128


namespace solve_for_x_l650_650930

theorem solve_for_x (x : ℝ) : 
  5 * x + 9 * x = 420 - 12 * (x - 4) -> 
  x = 18 :=
by
  intro h
  -- derivation will follow here
  sorry

end solve_for_x_l650_650930


namespace max_PA_on_ellipse_l650_650787

def ellipse_max_PA (a b : ℝ) (h1 : a > b) (h2 : b > 0) :=
  let PA_max :=
    if h3 : a ≥ Real.sqrt 2 * b then
      (a^2 / Real.sqrt (a^2 - b^2))
    else
      (2 * b)
  PA_max

theorem max_PA_on_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ PA_max, PA_max = 
    if h3 : a ≥ Real.sqrt 2 * b then
      (a^2 / Real.sqrt (a^2 - b^2))
    else
      (2 * b) :=
begin
  use ellipse_max_PA a b h1 h2,
  split_ifs,
  { refl, },
  { refl, },
end

end max_PA_on_ellipse_l650_650787


namespace Martha_height_in_meters_l650_650887

theorem Martha_height_in_meters :
  ∀ (height_in_inches : ℝ) (inches_to_cm : ℝ) (cm_to_m : ℝ),
  height_in_inches = 72 ∧ inches_to_cm = 2.54 ∧ cm_to_m = 1 / 100 →
  (Real.ceil ((height_in_inches * inches_to_cm * cm_to_m) * 100) / 100) = 1.83 :=
by
  intros height_in_inches inches_to_cm cm_to_m
  rintro ⟨h₁, h₂, h₃⟩
  have h₄ : height_in_inches * inches_to_cm * cm_to_m = 72 * 2.54 * (1 / 100) := by
    rw [h₁, h₂, h₃]
  /- We skip the proof details. -/
  sorry

end Martha_height_in_meters_l650_650887


namespace num_factors_180_l650_650363

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l650_650363


namespace min_pieces_chessboard_l650_650473

namespace Chessboard

def is_even (n : ℕ) : Prop := n % 2 = 0

def min_pieces_required (n : ℕ) : ℕ :=
if is_even n then 2 * n else 2 * n + 1

theorem min_pieces_chessboard (n : ℕ) : 
  (∀ n : ℕ, 
    (∀ piece_positions : Finset (Fin n × Fin n), 
      (∀ i : Fin n, ∃ (j : Fin n), (i, j) ∈ piece_positions) ∧ 
      (∀ j : Fin n, ∃ (i : Fin n), (i, j) ∈ piece_positions) ∧ 
      (∀ diag : ℤ, ∃ (i j : Fin n), diag = i.1 - j.1 ∧ (i, j) ∈ piece_positions) ∧ 
      (∀ anti_diag : ℤ, ∃ (i j : Fin n), anti_diag = i.1 + j.1 ∧ (i, j) ∈ piece_positions)) → 
    piece_positions.card ≥ min_pieces_required n) ∧ 
  ∃ piece_positions : Finset (Fin n × Fin n), 
    piece_positions.card = min_pieces_required n ∧ 
    (∀ i : Fin n, ∃ (j : Fin n), (i, j) ∈ piece_positions) ∧ 
    (∀ j : Fin n, ∃ (i : Fin n), (i, j) ∈ piece_positions) ∧ 
    (∀ diag : ℤ, ∃ (i j : Fin n), diag = i.1 - j.1 ∧ (i, j) ∈ piece_positions) ∧ 
    (∀ anti_diag : ℤ, ∃ (i j : Fin n), anti_diag = i.1 + j.1 ∧ (i, j) ∈ piece_positions)
  := sorry

end Chessboard

end min_pieces_chessboard_l650_650473


namespace sum_of_differences_eq_68900_l650_650864

def diff_sum (T : List ℕ) : ℕ :=
  let diffs := T.product T |>.filter (λ p, p.1 > p.2) |>.map (λ p, p.1 - p.2)
  diffs.sum

theorem sum_of_differences_eq_68900 : 
  diff_sum [3^0, 3^1, 3^2, 3^3, 3^4, 3^5, 3^6, 3^7, 3^8] = 68900 :=
sorry

end sum_of_differences_eq_68900_l650_650864


namespace find_B_l650_650070

theorem find_B (A B : ℝ) : (1 / 4 * 1 / 8 = 1 / (4 * A) ∧ 1 / 32 = 1 / B) → B = 32 := by
  intros h
  sorry

end find_B_l650_650070


namespace largest_prime_factor_of_1729_l650_650564

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650564


namespace polynomial_root_product_l650_650215

theorem polynomial_root_product :
  let P : Polynomial ℝ := Polynomial.ofCoeffs [2018, -4038, -2015, 4, 1]
  (∃ r1 r2 r3 r4 : ℝ, (P.eval r1 = 0) ∧ (P.eval r2 = 0) ∧ (P.eval r3 = 0) ∧ (P.eval r4 = 0)
  ∧ (-1)^4 * (r1 * r2 * r3 * r4) = 2018) :=
by
  sorry

end polynomial_root_product_l650_650215


namespace alicia_points_score_l650_650407

theorem alicia_points_score 
    (total_points : ℕ) (num_other_players : ℕ) (average_points_other_players : ℕ) 
    (team_score : total_points = 75) (num_other_players_condition : num_other_players = 8)
    (average_points_condition : average_points_other_players = 6) :
    total_points - (num_other_players * average_points_other_players) = 27 :=
by 
    rw [team_score, num_other_players_condition, average_points_condition]
    sorry

end alicia_points_score_l650_650407


namespace negation_of_proposition_l650_650809

theorem negation_of_proposition :
  (∀ x : ℝ, x > 0 → exp x ≥ x + 1) → (∃ x : ℝ, x > 0 ∧ exp x < x + 1) :=
begin
  intro h,
  by_contradiction h1,
  push_neg at h1,
  apply h1,
  exact h,
end

end negation_of_proposition_l650_650809


namespace students_play_neither_l650_650835

-- Define the given conditions
def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_sports_players : ℕ := 17

-- The goal is to prove the number of students playing neither sport is 7
theorem students_play_neither :
  total_students - (football_players + long_tennis_players - both_sports_players) = 7 :=
by
  sorry

end students_play_neither_l650_650835


namespace video_cassettes_in_second_set_l650_650093

-- Define the variables and conditions
variable (A V x1 x2 : ℝ)

-- Given conditions
def condition1 : Prop := 5 * A + V * x1 = 1350
def condition2 : Prop := 7 * A + 3 * V = 1110
def condition3 : Prop := V = 300

-- Theorem to prove the number of video cassettes in the second set
theorem video_cassettes_in_second_set (h1 : condition1) (h2 : condition2) (h3 : condition3) : x2 = 3 :=
by
  sorry

end video_cassettes_in_second_set_l650_650093


namespace solve_profession_arrangement_l650_650244

inductive Profession
| architect
| barista
| veterinarian
| guitarist

inductive Person
| Andrey
| Boris
| Vyacheslav
| Gennady

open Profession
open Person

structure SeatingArrangement :=
(seat1 : Person)
(seat2 : Person)
(seat3 : Person)
(seat4 : Person)

structure Assignment :=
(profession : Person → Profession)

def correct_arrangement_and_profession (sa : SeatingArrangement) (asgmt : Assignment) : Prop :=
(sa.seat1 ≠ sa.seat2 ∧ sa.seat1 ≠ sa.seat3 ∧ sa.seat1 ≠ sa.seat4 ∧
 sa.seat2 ≠ sa.seat3 ∧ sa.seat2 ≠ sa.seat4 ∧ sa.seat3 ≠ sa.seat4) ∧
((asgmt profession sa.seat1 = barista) ∧ (asgmt profession sa.seat2 = architect) ∧ 
 (asgmt profession sa.seat3 = veterinarian) ∧ (asgmt profession sa.seat4 = guitarist)) ∧
(sa.seat3 = Andrey) ∧ 
(sa.seat2 = Boris) ∧ 
(sa.seat4 = Vyacheslav)

theorem solve_profession_arrangement : 
  ∃ (sa : SeatingArrangement) (asgmt : Assignment), correct_arrangement_and_profession sa asgmt :=
sorry

end solve_profession_arrangement_l650_650244


namespace correlation_coefficient_correct_option_l650_650656

variable (r : ℝ)

-- Definitions of Conditions
def positive_correlation : Prop := r > 0 → ∀ x y : ℝ, x * y > 0
def range_r : Prop := -1 < r ∧ r < 1
def correlation_strength : Prop := |r| < 1 → (∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ δ < ε ∧ |r| < δ)

-- Theorem statement
theorem correlation_coefficient_correct_option :
  (positive_correlation r) ∧
  (range_r r) ∧
  (correlation_strength r) →
  (r ≠ 0 → |r| < 1) :=
by
  sorry

end correlation_coefficient_correct_option_l650_650656


namespace find_a_l650_650276

variable {a : ℝ} (f : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f = λ x, a^(x - 1/2)) (h4 : f (log 10 a) = real.sqrt 10)

theorem find_a : a = 10 ∨ a = 10^(-1/2) :=
by
  sorry

end find_a_l650_650276


namespace friends_professions_l650_650233

theorem friends_professions
  (architect barista veterinarian guitarist : ℕ)
  (P : ℕ → Sort)
  (Andrey Boris Vyacheslav Gennady : P)
  (seats : P → ℕ)
  (h1 : ∀ {x y}, seats x = veterinarian → seats y = architect → (seats x = seats y + 1 ∨ seats x = seats y - 1))
  (h2 : ∀ {x y}, seats x = barista → seats y = Boris → seats y = seats x + 1)
  (h3 : seats Vyacheslav > seats Andrey ∧ seats Vyacheslav > seats Boris)
  (h4 : ∀ {x}, x = Andrey → ∃ y z, (seats y = seats x + 1 ∨ seats y = seats x - 1) ∧ (seats z = seats x + 1 ∨ seats z = seats x - 1))
  (h5 : ∀ {x y}, seats x = guitarist → seats y = barista → seats x ≠ seats y + 1 ∧ seats x ≠ seats y - 1)
  : (∀ x, (x = Gennady → seats x = barista)
    ∧ (x = Boris → seats x = architect)
    ∧ (x = Andrey → seats x = veterinarian)
    ∧ (x = Vyacheslav → seats x = guitarist)) :=
by
  sorry

end friends_professions_l650_650233


namespace seating_arrangement_l650_650238

-- Definitions of friends and their professions
inductive Friend
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq

-- Seating condition definitions
def seated_right (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = a ∧ seating (i + 1) = b)

def seated_between (a b c : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = b ∧ seating (i - 1) = a ∧ seating (i + 1) = c)

def neighbors (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, (seating i = a ∧ seating (i - 1) = b) ∨ (seating i = a ∧ seating (i + 1) = b))

-- The proposition to be proved
theorem seating_arrangement
  (seating : ℕ → Friend)
  (profession : Friend → Profession)
  (h1 : ∃ i, seating_between Profession.Veterinarian Profession.Architect Profession.Guitarist seating)
  (h2 : seated_right Profession.Barista Profession.Boris seating)
  (h3 : ∃ i, seating i = Friend.Vyacheslav ∧ (∃ j, seating j = Friend.Andrey ∧ j < i) ∧ (∃ k, seating k = Friend.Boris ∧ k < i))
  (h4 : ∀ a, seating a = Friend.Andrey → neighbors Friend.Andrey seating)
  (h5 : ∀ a, seating a = Profession.Guitarist → ∀ b, seating (b - 1) = Profession.Barista ∨ seating (b + 1) = Profession.Barista → false) :
  (profession Friend.Gennady = Profession.Barista ∧
   profession Friend.Boris = Profession.Architect ∧
   profession Friend.Andrey = Profession.Veterinarian ∧
   profession Friend.Vyacheslav = Profession.Guitarist) :=
begin
  sorry
end

end seating_arrangement_l650_650238


namespace ratio_d_over_e_l650_650948

-- Define the given conditions from the problem
variables {a b c d e : ℝ}

-- Assume the roots of the polynomial are 1, 2, 3, and 4.
axiom roots_eq : ∀ x : ℝ, x * (x - 1) * (x - 2) * (x - 3) * (x - 4) = a * x^4 + b * x^3 + c * x^2 + d * x + e

-- Define using Vieta's formulas for specific conditions given
def Vieta_conditions := 
  (-d) / a = 50 ∧ 
  e / a = 24

-- The proof to show the ratio d/e given the conditions
theorem ratio_d_over_e : Vieta_conditions → d / e = -25 / 12 :=
by
  sorry

end ratio_d_over_e_l650_650948


namespace part1_geometric_sequence_part2_sum_first_n_terms_part3_max_integer_m_l650_650798

noncomputable def b : ℕ → ℝ
| 1 := 1
| n + 1 := b n + 3

noncomputable def a : ℕ → ℝ
| n := (1/2)^n

noncomputable def c (n : ℕ) : ℝ :=
1 / (b n * b (n + 1))

noncomputable def S (n : ℕ) : ℝ :=
n / (3 * n + 1)

noncomputable def d (n : ℕ) : ℝ :=
(3 * n + 1) * S n

theorem part1_geometric_sequence : ∀ n, a (n + 1) = (1 / 2) * a n := sorry

theorem part2_sum_first_n_terms (n : ℕ) : ∑ i in finset.range n, c i = S n := sorry

theorem part3_max_integer_m (n : ℕ) : (∀ n, ∑ i in finset.range n, (1 / (n + d i)) > 11 / 24) := sorry

end part1_geometric_sequence_part2_sum_first_n_terms_part3_max_integer_m_l650_650798


namespace cube_division_l650_650104

theorem cube_division :
  ∃ N, 
    let edge_large := 6 
    let volume_large := edge_large^3;
    volumes = [1, 2, 3, 6]
    N = 164 ∧ 
    volume_large = 2 * 3^3 + 162 * 1^3 :=
begin
  sorry
end

end cube_division_l650_650104


namespace sally_earnings_l650_650922

-- Definitions based on the conditions
def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20
def total_money : ℝ := total_seashells * price_per_seashell

-- Lean 4 statement to prove the total amount of money is $54
theorem sally_earnings : total_money = 54 := by
  -- Proof will go here
  sorry

end sally_earnings_l650_650922


namespace base_n_representation_of_b_l650_650393

theorem base_n_representation_of_b (n : ℕ) (b : ℕ)
  (h₁ : n > 12)
  (h₂ : (n - 7)^2 % n^2 = 49) -- equivalent to $(n - m)^2 = 49_n$
  (h₃ : 2 * n + 1 = a) -- conversion: $21_n$ to decimal
  : b = n * (n - 7) ∧ b.base_repr n = "60" :=
by
  sorry

end base_n_representation_of_b_l650_650393


namespace series_sum_l650_650718

noncomputable theory

def series_term (n : ℕ) : ℝ := (6 * n + 1) / ((6 * n - 1) ^ 2 * (6 * n + 5) ^ 2)

theorem series_sum : ∑' n : ℕ, series_term n = 1 / 300 :=
by sorry

end series_sum_l650_650718


namespace parallel_HM_OI_l650_650859

theorem parallel_HM_OI 
  (ABC : Triangle)
  (O : Point)
  (I : Point)
  (A_excircle : Excircle ABC)
  (B_excircle : Excircle ABC)
  (C_excircle : Excircle ABC)
  (A1 : Point)
  (B1 : Point)
  (C1 : Point)
  (H : Point)
  (P : Point)
  (M : Point)
  (mid_PA1 : M = midpoint P A1) 
  (orthocenter_ABC : H = orthocenter ABC)
  (orthocenter_AB1C1 : P = orthocenter (Triangle.mk A B1 C1))
  (touch_A : A1 ∈ (line BC))
  (touch_B : B1 ∈ (line CA))
  (touch_C : C1 ∈ (line AB)) : 
  parallel (line HM) (line OI) :=
sorry

end parallel_HM_OI_l650_650859


namespace digit_for_divisibility_by_45_l650_650190

theorem digit_for_divisibility_by_45 (n : ℕ) (h₀ : n < 10)
  (h₁ : 5 ∣ (5 + 10 * (7 + 4 * (1 + 5 * (8 + n))))) 
  (h₂ : 9 ∣ (5 + 7 + 4 + n + 5 + 8)) : 
  n = 7 :=
by { sorry }

end digit_for_divisibility_by_45_l650_650190


namespace prime_list_count_l650_650204

theorem prime_list_count {L : ℕ → ℕ} 
  (hL₀ : L 0 = 29)
  (hL : ∀ (n : ℕ), L (n + 1) = L n * 101 + L 0) :
  (∃! n, n = 0 ∧ Prime (L n)) ∧ ∀ m > 0, ¬ Prime (L m) := 
by
  sorry

end prime_list_count_l650_650204


namespace badges_total_l650_650814

theorem badges_total :
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  hermione_badges + luna_badges + celestia_badges = 83 :=
by
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  sorry

end badges_total_l650_650814


namespace Toby_daily_step_goal_l650_650516

theorem Toby_daily_step_goal 
  (S T : ℕ) 
  (h_SSunday : S = 9400)
  (h_SMonday : T = S + 9100)
  (h_STuesday : T = S + 8_300 + 9100)
  (h_SWed : T = S + 9_200 + 8_300 + 9_100)
  (h_SThursday : T = 44_900)
  (h_FriSat : T = 44_900 + 18_100)
  : (S + T) / 7 = 9000 := 
  by
  sorry

end Toby_daily_step_goal_l650_650516


namespace smallest_product_bdf_l650_650903

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end smallest_product_bdf_l650_650903


namespace area_of_rectangle_stage_8_l650_650387

theorem area_of_rectangle_stage_8 : 
  (∀ n, 4 * 4 = 16) →
  (∀ k, k ≤ 8 → k = k) →
  (8 * 16 = 128) :=
by
  intros h_sq_area h_sequence
  sorry

end area_of_rectangle_stage_8_l650_650387


namespace rectangle_area_stage_8_l650_650381

def square_side_length : ℕ := 4
def stage_count : ℕ := 8

-- The function to compute the area of one square
def square_area (side_length: ℕ) : ℕ :=
  side_length * side_length

-- The function to compute the total area at a given stage
def total_area_at_stage (side_length: ℕ) (stages: ℕ) : ℕ :=
  stages * (square_area side_length)

theorem rectangle_area_stage_8 :
  total_area_at_stage square_side_length stage_count = 128 :=
  by
    sorry

end rectangle_area_stage_8_l650_650381


namespace vanessa_quarters_needed_l650_650522

theorem vanessa_quarters_needed 
    (total_quarters : ℕ) 
    (quarters_per_soda : ℕ)
    (remaining_quarters : ℕ) 
    (quotient : ℕ)
    (total_quarters = 855) 
    (quarters_per_soda = 7) 
    (quotient = total_quarters / quarters_per_soda) 
    (remaining_quarters = total_quarters % quarters_per_soda): 
    quotient + 1 = 123 → remaining_quarters = 1 → 
    quarters_per_soda - remaining_quarters = 6 :=
by
  intro h1 h2
  sorry

end vanessa_quarters_needed_l650_650522


namespace PH_parallel_BC_l650_650860

variables {A B C O H K M P : Type} 
variable [ABC : Triangle A B C]

def is_circumcenter (O : Type) : Prop := sorry
def is_orthocenter (H : Type) : Prop := sorry
def is_midpoint (M : Type) (p1 p2 : Type) : Prop := sorry
def on_circumcircle (t : Triangle A B C) (p : Type) : Prop := sorry
def projection (p1 p2 : Type) (line : Line) : Type := sorry 
def parallel (line1 line2 : Line) : Prop := sorry
def line_through_points (point1 point2 : Type) : Line := sorry 
def line_parallel_to_seg (seg : Segment) (p : Type) : Line := sorry
def meets (line1 line2 : Line) (p : Type) : Prop := sorry

axiom circumcenter_exists : ∃ O, is_circumcenter O
axiom orthocenter_exists : ∃ H, is_orthocenter H
axiom midpoint_exists : ∃ M, is_midpoint M A B
axiom point_K : ∃ K, on_circumcircle (Triangle.mk A B C) K ∧ 
                     meets (line_through_points M H) (line_parallel_to_seg AB O) K
axiom point_P : ∃ P, projection K AC = P

theorem PH_parallel_BC : ∀ (A B C O H K M P : Type), 
  is_circumcenter O → 
  is_orthocenter H → 
  is_midpoint M A B → 
  (on_circumcircle (Triangle.mk A B C) K ∧ meets (line_through_points M H) (line_parallel_to_seg (Segment.mk A B) O) K) → 
  projection (line_k AC) P → 
  parallel (line_through_points P H) (line_segment.mk B C) := 
sorry

end PH_parallel_BC_l650_650860


namespace solveProfessions_l650_650255

-- Define our types for Friends and Professions
inductive Friend | Andrey | Boris | Vyacheslav | Gennady
inductive Profession | Architect | Barista | Veterinarian | Guitarist

-- Define the seating arrangement and conditions
def seatingArrangement : Friend → Position := sorry
def profession : Friend → Profession := sorry

-- Conditions translated to Lean
axiom condition1 : ∀ (f : Friend), 
  profession f = Profession.Veterinarian → 
  ∃ (neighbor1 neighbor2 : Friend), 
    seatingArrangement neighbor1 = seatingArrangement f + 1 ∨ seatingArrangement neighbor1 = seatingArrangement f - 1 ∧ 
    seatingArrangement neighbor2 = seatingArrangement f + 1 ∨ seatingArrangement neighbor2 = seatingArrangement f - 1 ∧
    (profession neighbor1 = Profession.Architect ∧ profession neighbor2 = Profession.Guitarist) 

axiom condition2 : ∀ (f : Friend), 
  profession f = Profession.Barista → 
  ∃ (rightNeighbor : Friend), 
    seatingArrangement rightNeighbor = seatingArrangement f + 1 ∧ 
    rightNeighbor = Friend.Boris

axiom condition3 : seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Andrey ∧ seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Boris

axiom condition4 : ∀ (f : Friend), 
  f = Friend.Andrey → 
  ∃ (leftNeighbor rightNeighbor : Friend), 
    seatingArrangement leftNeighbor = seatingArrangement f - 1 ∧ 
    seatingArrangement rightNeighbor = seatingArrangement f + 1

axiom condition5 : ∀ (f1 f2 : Friend), 
  (profession f1 = Profession.Guitarist ∧ profession f2 = Profession.Barista) → 
  seatingArrangement f1 ≠ seatingArrangement f2 + 1 ∧ seatingArrangement f1 ≠ seatingArrangement f2 - 1

-- Define the positions
def Position : Type := Fin 4

-- The main theorem to prove the assignments
theorem solveProfessions :
  profession Friend.Gennady = Profession.Barista ∧ 
  profession Friend.Boris = Profession.Architect ∧ 
  profession Friend.Andrey = Profession.Veterinarian ∧ 
  profession Friend.Vyacheslav = Profession.Guitarist :=
sorry

end solveProfessions_l650_650255


namespace largest_prime_factor_1729_l650_650587

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650587


namespace num_of_complex_solutions_l650_650752

theorem num_of_complex_solutions (z : ℂ) (h1 : |z| = 1) (h2 : |(z / conj(z) + conj(z) / z)| = 2 / 3) : 
  ∃ (s : Finset ℂ), s.card = 8 ∧ ∀ w ∈ s, |w| = 1 ∧ |(w / conj(w) + conj(w) / w)| = 2 / 3 := 
sorry

end num_of_complex_solutions_l650_650752


namespace concentric_circles_three_equal_segments_l650_650323

-- Define the problem in a Lean theorem statement
theorem concentric_circles_three_equal_segments
  (S₁ S₂ : Circle) 
  (h_concentric : S₁.center = S₂.center) 
  (r₁ r₂ : ℝ) 
  (h_radius : S₁.radius = r₁ ∧ S₂.radius = r₂)
  (h_radii_ineq : r₁ < r₂) : 
  ∃ (X Y : Point), (X ∈ S₁ ∧ Y ∈ S₂ ∧ SegmentsAreEqual (X, Y, S₁, S₂)) := 
sorry

end concentric_circles_three_equal_segments_l650_650323


namespace gcd_of_78_and_36_l650_650986

theorem gcd_of_78_and_36 :
  Nat.gcd 78 36 = 6 :=
by
  sorry

end gcd_of_78_and_36_l650_650986


namespace find_2023rd_letter_l650_650071

def sequence : List Char := ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'O', 'N', 'M', 'L', 'K', 'J', 'I', 'H', 'G', 'F']

def nth_letter_in_repeating_sequence (n : ℕ) : Char :=
  sequence[(n % sequence.length)]

theorem find_2023rd_letter : nth_letter_in_repeating_sequence 2023 = 'L' := by
  sorry

end find_2023rd_letter_l650_650071


namespace intersection_A_B_l650_650290

def A := {-1, 0, 1, 2, 3}
def B := {x | 0 < x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l650_650290


namespace acres_used_for_corn_and_potatoes_l650_650683

noncomputable def total_land : ℝ := 2586
def ratio_beans_wheat_corn_potatoes_barley : List ℝ := [7, 3, 6, 2, 5]

theorem acres_used_for_corn_and_potatoes : 
  (total_land / ratio_beans_wheat_corn_potatoes_barley.sum) * ((ratio_beans_wheat_corn_potatoes_barley.nthLe 2 _ + ratio_beans_wheat_corn_potatoes_barley.nthLe 3 _) : ℝ) = 899 :=
by
  sorry

end acres_used_for_corn_and_potatoes_l650_650683


namespace area_triangle_MPN_l650_650065

-- Defining the variables and conditions
variables (O1 O2 M N P D C : Point)
variables (MO1 O1D NO2 CO2 : ℝ)
variables (angle1 angle2 : ℝ)

-- Defining the conditions
axiom circles_on_MN : (O1 lies_on MN) ∧ (O2 lies_on MN)
axiom circles_touch : circles_touch_at O1 O2
axiom circles_intersect : (M, D) ∈ (MP ∩ O1.circle()) ∧ (N, C) ∈ (PN ∩ O2.circle())
axiom MO1_eq_3 : MO1 = 3
axiom O1D_eq_3 : O1D = 3
axiom NO2_eq_6 : NO2 = 6
axiom CO2_eq_6 : CO2 = 6
axiom area_ratio : area(M C O2) / area(O1 D N) = 8 * sqrt(2 - sqrt(3))

-- Prove the area of the triangle MPN
theorem area_triangle_MPN : area(M N P) = (81 * (sqrt(3) - 1)) / 2 :=
sorry

end area_triangle_MPN_l650_650065


namespace green_eyes_count_l650_650047

theorem green_eyes_count (total_people : ℕ) (blue_eyes : ℕ) (brown_eyes : ℕ) (black_eyes : ℕ) (green_eyes : ℕ) :
  total_people = 100 → 
  blue_eyes = 19 → 
  brown_eyes = total_people / 2 → 
  black_eyes = total_people / 4 → 
  green_eyes = total_people - (blue_eyes + brown_eyes + black_eyes) → 
  green_eyes = 6 := 
by 
  intros h_total h_blue h_brown h_black h_green 
  rw [h_total, h_blue, h_brown, h_black] at h_green 
  exact h_green.symm

end green_eyes_count_l650_650047


namespace final_professions_correct_l650_650267

-- Define the people and their positions
inductive Person
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq, Inhabited

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq, Inhabited

open Person Profession

-- Given positions in seats
def seatingArrangement : Person → Nat := 
λ p, 
  match p with
  | Andrey    => 3
  | Boris     => 2
  | Vyacheslav => 4
  | Gennady   => 1

-- Given professions
def profession : Person → Profession := 
λ p, 
  match p with
  | Andrey    => Veterinarian
  | Boris     => Architect
  | Vyacheslav => Guitarist
  | Gennady   => Barista

-- Define the conditions
def condition1 : Prop :=
  (seatingArrangement Andrey = 3) ∧ 
  (seatingArrangement Boris = 2) ∧
  (seatingArrangement Vyacheslav = 4) ∧ 
  (seatingArrangement Gennady = 1)

def condition2 : Prop := 
  seatingArrangement (Person.Barista) + 1 = seatingArrangement Boris

def condition3 : Prop := 
  seatingArrangement Vyacheslav > seatingArrangement Andrey ∧ 
  seatingArrangement Vyacheslav > seatingArrangement Boris

def condition4 : Prop := 
  seatingArrangement (Person.Veterinarian) ≠ 1 ∧ seatingArrangement (Person.Veterinarian) ≠ 4

def condition5 : Prop := 
  (seatingArrangement (Person.Barista) ≠ seatingArrangement (Person.Guitarist) + 1 ∧
  seatingArrangement (Person.Guitarist) ≠ seatingArrangement (Person.Barista) + 1)

-- Prove the final professions
theorem final_professions_correct :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  profession Gennady = Barista ∧
  profession Boris = Architect ∧
  profession Andrey = Veterinarian ∧
  profession Vyacheslav = Guitarist :=
by sorry

end final_professions_correct_l650_650267


namespace surface_area_of_brick_l650_650756

-- Define the three dimensions of the brick
def length := 8
def width := 6
def height := 2

-- Calculate surface area for a brick with given dimensions
def surface_area : ℕ := 
  (2 * length * width) + (2 * length * height) + (2 * width * height)

-- Prove that the surface area is 152 cm²
theorem surface_area_of_brick : surface_area = 152 := 
by
  sorry

end surface_area_of_brick_l650_650756


namespace solveProfessions_l650_650256

-- Define our types for Friends and Professions
inductive Friend | Andrey | Boris | Vyacheslav | Gennady
inductive Profession | Architect | Barista | Veterinarian | Guitarist

-- Define the seating arrangement and conditions
def seatingArrangement : Friend → Position := sorry
def profession : Friend → Profession := sorry

-- Conditions translated to Lean
axiom condition1 : ∀ (f : Friend), 
  profession f = Profession.Veterinarian → 
  ∃ (neighbor1 neighbor2 : Friend), 
    seatingArrangement neighbor1 = seatingArrangement f + 1 ∨ seatingArrangement neighbor1 = seatingArrangement f - 1 ∧ 
    seatingArrangement neighbor2 = seatingArrangement f + 1 ∨ seatingArrangement neighbor2 = seatingArrangement f - 1 ∧
    (profession neighbor1 = Profession.Architect ∧ profession neighbor2 = Profession.Guitarist) 

axiom condition2 : ∀ (f : Friend), 
  profession f = Profession.Barista → 
  ∃ (rightNeighbor : Friend), 
    seatingArrangement rightNeighbor = seatingArrangement f + 1 ∧ 
    rightNeighbor = Friend.Boris

axiom condition3 : seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Andrey ∧ seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Boris

axiom condition4 : ∀ (f : Friend), 
  f = Friend.Andrey → 
  ∃ (leftNeighbor rightNeighbor : Friend), 
    seatingArrangement leftNeighbor = seatingArrangement f - 1 ∧ 
    seatingArrangement rightNeighbor = seatingArrangement f + 1

axiom condition5 : ∀ (f1 f2 : Friend), 
  (profession f1 = Profession.Guitarist ∧ profession f2 = Profession.Barista) → 
  seatingArrangement f1 ≠ seatingArrangement f2 + 1 ∧ seatingArrangement f1 ≠ seatingArrangement f2 - 1

-- Define the positions
def Position : Type := Fin 4

-- The main theorem to prove the assignments
theorem solveProfessions :
  profession Friend.Gennady = Profession.Barista ∧ 
  profession Friend.Boris = Profession.Architect ∧ 
  profession Friend.Andrey = Profession.Veterinarian ∧ 
  profession Friend.Vyacheslav = Profession.Guitarist :=
sorry

end solveProfessions_l650_650256


namespace monotonic_intervals_tangent_line_exists_l650_650311

noncomputable def f (k x : ℝ) : ℝ :=
  (1 - k) * x + 1 / Real.exp x

def f_derivative (k x : ℝ) : ℝ :=
  ((1 - k) * Real.exp x - 1) / Real.exp x

theorem monotonic_intervals (k : ℝ) :
  (k ≥ 1 → ∀ x: ℝ, f_derivative k x < 0) ∧
  (k < 1 →
    ∀ x: ℝ, x < -Real.log (1 - k) → f_derivative k x < 0 ∧
    ∀ x: ℝ, x > -Real.log (1 - k) → f_derivative k x > 0) :=
  by sorry

noncomputable def M (x : ℝ) : ℝ :=
  (x + 1) / Real.exp x

def M_derivative (x : ℝ) : ℝ :=
  (-x) / Real.exp x

theorem tangent_line_exists (t : ℝ) :
  ∀ (x₀ : ℝ), x₀ = 0 → M x₀ = 1 → t ≤ 1 :=
  by sorry

end monotonic_intervals_tangent_line_exists_l650_650311


namespace verify_statements_l650_650293

theorem verify_statements (a b : ℝ) :
  ( (ab < 0 ∧ (a < 0 ∧ b > 0 ∨ a > 0 ∧ b < 0)) → (a / b = -1)) ∧
  ( (a + b < 0 ∧ ab > 0) → (|2 * a + 3 * b| = -(2 * a + 3 * b)) ) ∧
  ( (|a - b| + a - b = 0) → (b > a) = False ) ∧
  ( (|a| > |b|) → ((a + b) * (a - b) < 0) = False ) :=
by
  sorry

end verify_statements_l650_650293


namespace largest_prime_factor_of_1729_l650_650591

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650591


namespace neg_x₀_is_root_of_exp_f_sub_one_l650_650294

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}

-- Assume f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- x₀ is a root of y = f(x) + e^x
def is_root_of_sum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ + real.exp x₀ = 0

theorem neg_x₀_is_root_of_exp_f_sub_one (hf : is_odd_function f) (hx₀ : is_root_of_sum f x₀) :
  e^(-x₀) * f(-x₀) - 1 = 0 := 
sorry

end neg_x₀_is_root_of_exp_f_sub_one_l650_650294


namespace prod_exp_diff_eq_one_l650_650719

noncomputable def Q (x : ℂ) : ℂ := ∏ k in finset.range 7, (x - complex.exp (2 * real.pi * complex.I * k / 9))

noncomputable def R (x : ℂ) : ℂ := ∏ j in finset.range 6, (x - complex.exp (2 * real.pi * complex.I * j / 8))

theorem prod_exp_diff_eq_one :
  ∏ k in finset.range 7, ∏ j in finset.range 6, (complex.exp (2 * real.pi * complex.I * j / 8) - complex.exp (2 * real.pi * complex.I * k / 9)) = 1 :=
sorry

end prod_exp_diff_eq_one_l650_650719


namespace inverse_function_point_l650_650831

variables {α β : Type} [partial_order α] [partial_order β]

def f (x : α) : β := sorry

theorem inverse_function_point (h₁ : f 1 = 2) : function.inverse (λ x, f (x + 2)) 2 = -1 :=
by sorry

end inverse_function_point_l650_650831


namespace avg_age_of_5_is_14_l650_650015

open Nat Real

def average_age_of_5_persons (total_age_17 : ℝ) (avg_age_9 : ℝ) (age_15th : ℝ) := 
  (total_age_17 - (9 * avg_age_9) - age_15th) / 5

theorem avg_age_of_5_is_14 
  (avg_age_17 : ℝ) (total_age_17 : ℝ)
  (avg_age_9 : ℝ) (age_15th : ℝ) :
  (17 * avg_age_17 = total_age_17) → 
  (avg_age_9 = 16) → 
  (age_15th = 41) →
  average_age_of_5_persons total_age_17 avg_age_9 age_15th = 14 :=
by
  sorry

end avg_age_of_5_is_14_l650_650015


namespace complex_div_eq_neg_im_l650_650801

noncomputable def z : ℂ := 1 - Complex.i

theorem complex_div_eq_neg_im :
  (1 - Complex.i) / z = -Complex.i :=
begin
  simp [z],
  field_simp,
  sorry
end

end complex_div_eq_neg_im_l650_650801


namespace num_factors_180_l650_650361

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l650_650361


namespace triangle_side_length_l650_650054

theorem triangle_side_length {x : ℝ} (h1 : 6 + x + x = 20) : x = 7 :=
by 
  sorry

end triangle_side_length_l650_650054


namespace correct_conclusions_l650_650157

theorem correct_conclusions :
  let f1 (x : ℝ) := x + (1 / x)
  ∀ x < 0, f1(x) ≤ -2 ∧ (∃ x, (x = -1) ∧ f1(x) = -2) →
  let f2 (x : ℕ) := 1 / (Real.log (x + 2))
  ¬ ∀ x > -2, f2(x) is_decreasing →
  ∃ a : ℝ, ∀ x : ℝ, 
    let f3 (x : ℝ) := x / ((2*x + 1) * (x - a))
    f3 (-x) = -f3 (x) →
  ¬ ∀ x y : ℝ, 
    x > 0 ∧ y > 0 →
    let log (x : ℝ) := Real.log x
    log (x * y) = log(x) + log(y)
  → 
  ({1, 3} : Set ℕ) :=
by
  sorry

end correct_conclusions_l650_650157


namespace largest_prime_factor_1729_l650_650534

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650534


namespace friends_professions_l650_650234

theorem friends_professions
  (architect barista veterinarian guitarist : ℕ)
  (P : ℕ → Sort)
  (Andrey Boris Vyacheslav Gennady : P)
  (seats : P → ℕ)
  (h1 : ∀ {x y}, seats x = veterinarian → seats y = architect → (seats x = seats y + 1 ∨ seats x = seats y - 1))
  (h2 : ∀ {x y}, seats x = barista → seats y = Boris → seats y = seats x + 1)
  (h3 : seats Vyacheslav > seats Andrey ∧ seats Vyacheslav > seats Boris)
  (h4 : ∀ {x}, x = Andrey → ∃ y z, (seats y = seats x + 1 ∨ seats y = seats x - 1) ∧ (seats z = seats x + 1 ∨ seats z = seats x - 1))
  (h5 : ∀ {x y}, seats x = guitarist → seats y = barista → seats x ≠ seats y + 1 ∧ seats x ≠ seats y - 1)
  : (∀ x, (x = Gennady → seats x = barista)
    ∧ (x = Boris → seats x = architect)
    ∧ (x = Andrey → seats x = veterinarian)
    ∧ (x = Vyacheslav → seats x = guitarist)) :=
by
  sorry

end friends_professions_l650_650234


namespace friends_professions_l650_650232

theorem friends_professions
  (architect barista veterinarian guitarist : ℕ)
  (P : ℕ → Sort)
  (Andrey Boris Vyacheslav Gennady : P)
  (seats : P → ℕ)
  (h1 : ∀ {x y}, seats x = veterinarian → seats y = architect → (seats x = seats y + 1 ∨ seats x = seats y - 1))
  (h2 : ∀ {x y}, seats x = barista → seats y = Boris → seats y = seats x + 1)
  (h3 : seats Vyacheslav > seats Andrey ∧ seats Vyacheslav > seats Boris)
  (h4 : ∀ {x}, x = Andrey → ∃ y z, (seats y = seats x + 1 ∨ seats y = seats x - 1) ∧ (seats z = seats x + 1 ∨ seats z = seats x - 1))
  (h5 : ∀ {x y}, seats x = guitarist → seats y = barista → seats x ≠ seats y + 1 ∧ seats x ≠ seats y - 1)
  : (∀ x, (x = Gennady → seats x = barista)
    ∧ (x = Boris → seats x = architect)
    ∧ (x = Andrey → seats x = veterinarian)
    ∧ (x = Vyacheslav → seats x = guitarist)) :=
by
  sorry

end friends_professions_l650_650232


namespace range_of_m_l650_650803

def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + b

def has_local_extremum_at (a b x : ℝ) : Prop :=
  f_prime a b x = 0 ∧ f a b x = 0

def h (a b m x : ℝ) : ℝ := f a b x - m + 1

theorem range_of_m (a b m : ℝ) :
  (has_local_extremum_at 2 9 (-1) ∧
   ∀ x, f 2 9 x = x^3 + 6 * x^2 + 9 * x + 4) →
  (∀ x, (x^3 + 6 * x^2 + 9 * x + 4 - m + 1 = 0) → 
  1 < m ∧ m < 5) := 
sorry

end range_of_m_l650_650803


namespace polynomial_reducible_over_F2_polynomial_not_always_irreducible_over_F2_l650_650914

theorem polynomial_reducible_over_F2 (n : ℕ) (hn_comp : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n + 1 = a * b) : 
  ¬ irreducible (1 + Finset.sum (Finset.range (n + 1)) (λ k, monomial k 1 : Polynomial (ZMod 2))) :=
sorry

theorem polynomial_not_always_irreducible_over_F2 (n : ℕ) (hn_prime : Prime (n + 1)) : 
  ¬ irreducible (1 + Finset.sum (Finset.range (n + 1)) (λ k, monomial k 1 : Polynomial (ZMod 2))) :=
sorry

end polynomial_reducible_over_F2_polynomial_not_always_irreducible_over_F2_l650_650914


namespace find_variable_value_l650_650823

axiom variable_property (x : ℝ) (h : 4 + 1 / x ≠ 0) : 5 / (4 + 1 / x) = 1 → x = 1

-- Given condition: 5 / (4 + 1 / x) = 1
-- Prove: x = 1
theorem find_variable_value (x : ℝ) (h : 4 + 1 / x ≠ 0) (h1 : 5 / (4 + 1 / x) = 1) : x = 1 :=
variable_property x h h1

end find_variable_value_l650_650823


namespace bases_with_final_digit_one_l650_650767

theorem bases_with_final_digit_one :
  { b : ℕ | 3 ≤ b ∧ b ≤ 10 ∧ 624 % b = 0 }.card = 4 :=
by
  sorry

end bases_with_final_digit_one_l650_650767


namespace solveProfessions_l650_650253

-- Define our types for Friends and Professions
inductive Friend | Andrey | Boris | Vyacheslav | Gennady
inductive Profession | Architect | Barista | Veterinarian | Guitarist

-- Define the seating arrangement and conditions
def seatingArrangement : Friend → Position := sorry
def profession : Friend → Profession := sorry

-- Conditions translated to Lean
axiom condition1 : ∀ (f : Friend), 
  profession f = Profession.Veterinarian → 
  ∃ (neighbor1 neighbor2 : Friend), 
    seatingArrangement neighbor1 = seatingArrangement f + 1 ∨ seatingArrangement neighbor1 = seatingArrangement f - 1 ∧ 
    seatingArrangement neighbor2 = seatingArrangement f + 1 ∨ seatingArrangement neighbor2 = seatingArrangement f - 1 ∧
    (profession neighbor1 = Profession.Architect ∧ profession neighbor2 = Profession.Guitarist) 

axiom condition2 : ∀ (f : Friend), 
  profession f = Profession.Barista → 
  ∃ (rightNeighbor : Friend), 
    seatingArrangement rightNeighbor = seatingArrangement f + 1 ∧ 
    rightNeighbor = Friend.Boris

axiom condition3 : seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Andrey ∧ seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Boris

axiom condition4 : ∀ (f : Friend), 
  f = Friend.Andrey → 
  ∃ (leftNeighbor rightNeighbor : Friend), 
    seatingArrangement leftNeighbor = seatingArrangement f - 1 ∧ 
    seatingArrangement rightNeighbor = seatingArrangement f + 1

axiom condition5 : ∀ (f1 f2 : Friend), 
  (profession f1 = Profession.Guitarist ∧ profession f2 = Profession.Barista) → 
  seatingArrangement f1 ≠ seatingArrangement f2 + 1 ∧ seatingArrangement f1 ≠ seatingArrangement f2 - 1

-- Define the positions
def Position : Type := Fin 4

-- The main theorem to prove the assignments
theorem solveProfessions :
  profession Friend.Gennady = Profession.Barista ∧ 
  profession Friend.Boris = Profession.Architect ∧ 
  profession Friend.Andrey = Profession.Veterinarian ∧ 
  profession Friend.Vyacheslav = Profession.Guitarist :=
sorry

end solveProfessions_l650_650253


namespace largest_prime_factor_of_1729_is_19_l650_650608

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650608


namespace cyclic_sum_inequality_l650_650455

theorem cyclic_sum_inequality {a : ℕ → ℝ} (n : ℕ) (hpos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i)
  (hsum : (∑ i in finset.range n, a i.succ) = 1) :
  (∑ i in finset.range n, (a i.succ)^2 / (a i.succ + a (i+1) % n + 1)) ≥ 1 / 2 :=
by
  sorry

end cyclic_sum_inequality_l650_650455


namespace divisible_by_6_l650_650208

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n^3 - n + 6) :=
by
  sorry

end divisible_by_6_l650_650208


namespace sum_of_eight_smallest_multiples_of_10_l650_650647

theorem sum_of_eight_smallest_multiples_of_10 :
  (∑ i in finset.range 8, 10 * (i + 1)) = 360 :=
by
  sorry

end sum_of_eight_smallest_multiples_of_10_l650_650647


namespace number_of_digits_in_4_16_5_25_l650_650961

theorem number_of_digits_in_4_16_5_25 : 
  (nat.log10 (4^16 * 5^25)).nat_abs + 1 = 28 :=
sorry

end number_of_digits_in_4_16_5_25_l650_650961


namespace largest_prime_factor_1729_l650_650576

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650576


namespace find_a_l650_650025

theorem find_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-3 / 2) 2, f x ≤ 1) ∧ (∃ x ∈ Set.Icc (-3 / 2) 2, f x = 1) ↔
  (a = 3 / 4 ∨ a = (-3 + 2 * Real.sqrt 2) / 2 ∨ a = (-3 - 2 * Real.sqrt 2) / 2) :=
by
  -- Define the function f(x)
  let f (x : ℝ) := a * x^2 + (2 * a - 1) * x - 3
  sorry

end find_a_l650_650025


namespace largest_prime_factor_of_1729_l650_650640

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650640


namespace largest_prime_factor_1729_l650_650572

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650572


namespace avg_first_10_primes_gt_100_l650_650205

def first_10_primes_gt_100 := [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]

theorem avg_first_10_primes_gt_100 : 
    (list.sum first_10_primes_gt_100 / 10 : ℝ) = 121.6 :=
by
  sorry

end avg_first_10_primes_gt_100_l650_650205


namespace preimage_of_point_l650_650285

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

-- Define the statement of the problem
theorem preimage_of_point {x y : ℝ} (h1 : f x y = (3, 1)) : (x = 2 ∧ y = 1) :=
by
  sorry

end preimage_of_point_l650_650285


namespace arithmetic_sequence_and_general_formula_l650_650790

-- Define the arithmetic sequence and its properties
def a_n (n : ℕ) : ℕ := 2 * n

-- Define the geometric mean sequence {b_n}
def b_n (n : ℕ) : ℝ := real.sqrt ((a_n n) * (a_n (n + 1)))

-- Define the sequence {c_n}
def c_n (n : ℕ) : ℝ := (b_n (n + 1))^2 - (b_n n)^2

-- Prove that {c_n} is an arithmetic sequence and find the general formula for {a_n}
theorem arithmetic_sequence_and_general_formula (n : ℕ) :
  (∀ n, c_n n - c_n (n - 1) = 8) ∧ (c_1 = 16 → ∀ n : ℕ, a_n n = 2 * n) := 
by 
  -- Proof steps would go here.
  sorry

end arithmetic_sequence_and_general_formula_l650_650790


namespace coefficient_of_x_l650_650018

open BigOperators

-- Definition for binomial coefficients
noncomputable def binomial (n k : ℕ) : ℕ := if k > n then 0 else Nat.choose n k

-- The theorem to prove
theorem coefficient_of_x {R : Type*} [CommRing R] (x : R) :
  let f := (1 - 2 * x)^5 * (1 + 3 * x)^4
  (binomial 5 1 * (-2) + binomial 4 1 * 3) = 2 := by
  sorry

end coefficient_of_x_l650_650018


namespace net_displacement_total_fuel_consumed_l650_650515

def distances : List ℤ := [+5, -4, +3, -10, +3, -9]
def fuel_consumption_per_km : ℝ := 0.4

theorem net_displacement :
  List.sum distances = -12 :=
by
  sorry

theorem total_fuel_consumed :
  List.sum (distances.map Int.natAbs) * fuel_consumption_per_km = 13.6 :=
by
  sorry

end net_displacement_total_fuel_consumed_l650_650515


namespace polynomial_evaluation_l650_650758

theorem polynomial_evaluation (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 15) - 2 = -x^4 + 3*x^3 - 5*x^2 + 15*x - 2 :=
by
  sorry

end polynomial_evaluation_l650_650758


namespace count_100_digit_even_numbers_l650_650342

theorem count_100_digit_even_numbers : 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  in
  count_valid_numbers = 2 * 3 ^ 98 :=
by 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  have : count_valid_numbers = 2 * 3 ^ 98 := by sorry
  exact this

end count_100_digit_even_numbers_l650_650342


namespace cos_alpha_value_cos_2alpha_value_l650_650800

noncomputable def x : ℤ := -3
noncomputable def y : ℤ := 4
noncomputable def r : ℝ := Real.sqrt (x^2 + y^2)
noncomputable def cos_alpha : ℝ := x / r
noncomputable def cos_2alpha : ℝ := 2 * cos_alpha^2 - 1

theorem cos_alpha_value : cos_alpha = -3 / 5 := by
  sorry

theorem cos_2alpha_value : cos_2alpha = -7 / 25 := by
  sorry

end cos_alpha_value_cos_2alpha_value_l650_650800


namespace combined_salaries_of_ABCD_l650_650040

theorem combined_salaries_of_ABCD 
  (A B C D E : ℝ)
  (h1 : E = 9000)
  (h2 : (A + B + C + D + E) / 5 = 8600) :
  A + B + C + D = 34000 := 
sorry

end combined_salaries_of_ABCD_l650_650040


namespace rectangle_dimensions_l650_650962

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : 2 * l + 2 * w = 150) 
  (h2 : l = w + 15) : 
  w = 30 ∧ l = 45 := 
  by 
  sorry

end rectangle_dimensions_l650_650962


namespace PQ_length_l650_650421

theorem PQ_length (BC AD : ℝ) (angle_A angle_D : ℝ) (P Q : ℝ) 
  (H1 : BC = 700) (H2 : AD = 1400) (H3 : angle_A = 45) (H4 : angle_D = 45) 
  (mid_BC : P = BC / 2) (mid_AD : Q = AD / 2) :
  abs (Q - P) = 350 :=
by
  sorry

end PQ_length_l650_650421


namespace selling_price_30_percent_profit_l650_650699

noncomputable def store_cost : ℝ := 2412.31 / 1.40

theorem selling_price_30_percent_profit : 
  let selling_price_30 := 1.30 * store_cost in
  selling_price_30 = 2240.00 :=
by
  have h : store_cost = 1723.08 := by sorry
  have : 1.30 * store_cost = 2240.00 :=
    by
      calc
        1.30 * store_cost = 1.30 * 1723.08 : by rw [h]
        ... = 2240.00 : by norm_num
  exact this

end selling_price_30_percent_profit_l650_650699


namespace sequence_sum_l650_650304

theorem sequence_sum (a : ℕ → ℝ) (n : ℕ) (h_pos : ∀ i, 1 ≤ i → i ≤ n → a i > 0)
  (h_seq : ∑ i in Finset.range (n+1), sqrt (a i) = n^2 + n) :
  (∑ i in Finset.range n, 1 / (a (i+1) - 1)) = n / (2*n + 1) :=
sorry

end sequence_sum_l650_650304


namespace divide_cakes_l650_650197

/-- Statement: Eleven cakes can be divided equally among six girls without cutting any cake into 
exactly six equal parts such that each girl receives 1 + 1/2 + 1/4 + 1/12 cakes -/
theorem divide_cakes (cakes girls : ℕ) (h_cakes : cakes = 11) (h_girls : girls = 6) :
  ∃ (parts : ℕ → ℝ), (∀ i, parts i = 1 + 1 / 2 + 1 / 4 + 1 / 12) ∧ (cakes = girls * (1 + 1 / 2 + 1 / 4 + 1 / 12)) :=
by
  sorry

end divide_cakes_l650_650197


namespace valid_integers_count_l650_650344

def count_valid_integers : ℕ :=
  let digits : List ℕ := [0, 1, 2, 3, 4, 6, 7, 8, 9]
  let first_digit_count := 7  -- from 2 to 9 excluding 5
  let second_digit_count := 8
  let third_digit_count := 7
  let fourth_digit_count := 6
  first_digit_count * second_digit_count * third_digit_count * fourth_digit_count

theorem valid_integers_count : count_valid_integers = 2352 := by
  -- intermediate step might include nice counting macros
  sorry

end valid_integers_count_l650_650344


namespace modulus_of_2_over_1_plus_i_l650_650089

theorem modulus_of_2_over_1_plus_i : (Complex.abs (2 / (1 + Complex.i)) = Real.sqrt 2) :=
by
  sorry

end modulus_of_2_over_1_plus_i_l650_650089


namespace largest_prime_factor_of_1729_l650_650555

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650555


namespace absolute_value_of_slope_l650_650055

noncomputable def circle_center1 : ℝ × ℝ := (14, 92)
noncomputable def circle_center2 : ℝ × ℝ := (17, 76)
noncomputable def circle_center3 : ℝ × ℝ := (19, 84)
noncomputable def radius : ℝ := 3
noncomputable def point_on_line : ℝ × ℝ := (17, 76)

theorem absolute_value_of_slope :
  ∃ m : ℝ, ∀ line : ℝ × ℝ → Prop,
    (line point_on_line) ∧ 
    (∀ p, (line p) → true) → 
    abs m = 24 := 
  sorry

end absolute_value_of_slope_l650_650055


namespace largest_prime_factor_1729_l650_650573

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650573


namespace largest_prime_factor_1729_l650_650617

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650617


namespace range_of_a_l650_650796

variables {f : ℝ → ℝ}

-- Define the conditions as hypotheses
hypothesis h_dom : ∀ x, x ∈ Icc (-4 : ℝ) 4 → f x ∈ ℝ
hypothesis h_even : ∀ x, f (-x) = f x
hypothesis h_decrease : ∀ x1 x2 ∈ Icc (0 : ℝ) 4, x1 < x2 → x1 * f x1 - x2 * f x2 > 0

-- Define the main statement regarding the range for a
theorem range_of_a :
  ∀ a : ℝ, (a + 2) * f (a + 2) < (1 - a) * f (1 - a) → (-1/2 : ℝ) < a ∧ a ≤ 2 :=
sorry

end range_of_a_l650_650796


namespace coefficient_of_x_squared_l650_650206

noncomputable def f_k : ℕ → (ℝ → ℝ)
| 0 := λ x, x
| (n + 1) := λ x, (f_k n x - 2)^2

def a_k (k : ℕ) : ℝ :=
(4^k - 1) / 3 * 4^(k - 1)

theorem coefficient_of_x_squared (k : ℕ) :
  (∂^2 / ∂x^2 (f_k k x) | x = 0) = 2 * a_k k :=
sorry

end coefficient_of_x_squared_l650_650206


namespace line_passing_through_quadrants_l650_650808

theorem line_passing_through_quadrants (a : ℝ) :
  (∀ x : ℝ, (3 * a - 1) * x - 1 ≠ 0) →
  (3 * a - 1 > 0) →
  a > 1 / 3 :=
by
  intro h1 h2
  -- proof to be filled
  sorry

end line_passing_through_quadrants_l650_650808


namespace largest_prime_factor_1729_l650_650569

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650569


namespace num_ways_to_arrange_8_letters_l650_650819

theorem num_ways_to_arrange_8_letters : (Finset.univ.card : ℕ) = 40320 := by
  sorry

end num_ways_to_arrange_8_letters_l650_650819


namespace smallest_c_is_52_l650_650979

def seq (n : ℕ) : ℤ := -103 + (n:ℤ) * 2

theorem smallest_c_is_52 :
  ∃ c : ℕ, 
  (∀ n : ℕ, n < c → (∀ m : ℕ, m < n → seq m < 0) ∧ seq n = 0) ∧
  seq c > 0 ∧
  c = 52 :=
by
  sorry

end smallest_c_is_52_l650_650979


namespace parabola_equation_chord_length_parabola_l650_650321

-- Definitions for the problem
def parabola (p : ℝ) (x y : ℝ) := x^2 = 2 * p * y
def line (k : ℝ) (x y : ℝ) := y = k * x + 2
def dot_product (x1 y1 x2 y2 : ℝ) := x1 * x2 + y1 * y2

-- Given conditions
variable (p k : ℝ)
variable (x1 y1 x2 y2 : ℝ)
variable (O : (ℝ × ℝ)) -- Origin

-- Prove the equation of the parabola E
theorem parabola_equation (h1 : parabola p x1 y1) (h2 : parabola p x2 y2) 
                          (h3 : line k x1 y1) (h4 : line k x2 y2) 
                          (h5 : dot_product x1 y1 x2 y2 = 2) : 
                          x1^2 + x2^2 = 1 ∧ y1 = x1^2 / 2 ∧ y2 = x2^2 / 2 := 
sorry

-- Prove the length of the chord AB when k = 1
theorem chord_length_parabola (h1 : x1 + x2 = 1) (h2 : x1 * x2 = -2) 
                              (h3 : k = 1) : 
                              real.abs ((x1 - x2) + ((x1 - x2)^2 + 4 * (x1 * x2)) / 2) = 3 * real.sqrt 2 := 
sorry

end parabola_equation_chord_length_parabola_l650_650321


namespace largest_prime_factor_of_1729_l650_650638

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650638


namespace problem_statement_l650_650274

noncomputable def a : ℝ := Real.tan (1 / 2)
noncomputable def b : ℝ := Real.tan (2 / Real.pi)
noncomputable def c : ℝ := Real.sqrt 3 / Real.pi

theorem problem_statement : a < c ∧ c < b := by
  sorry

end problem_statement_l650_650274


namespace jillian_max_apartment_size_l650_650707

theorem jillian_max_apartment_size :
  ∀ s : ℝ, (1.10 * s = 880) → s = 800 :=
by
  intros s h
  sorry

end jillian_max_apartment_size_l650_650707


namespace sum_coefficients_l650_650379

theorem sum_coefficients :
  (∀ x : ℝ, (2*x + 1)^10 = a_0 + a_1*(x + 1) + a_2*(x + 1)^2 + a_3*(x + 1)^3 + a_4*(x + 1)^4 +
            a_5*(x + 1)^5 + a_6*(x + 1)^6 + a_7*(x + 1)^7 + a_8*(x + 1)^8 + a_9*(x + 1)^9 + a_{10}*(x + 1)^10) →
  (1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10}) →
  (1 = a_0) →
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} = 0) :=
by
  intros h1 h2 h3
  sorry

end sum_coefficients_l650_650379


namespace side_XY_length_l650_650519

-- Define the conditions
def is_isosceles_right_triangle (X Y Z : Type) [MetricSpace X] (x y z : X) : Prop :=
  (dist x y = dist y z ∧ dist x z ^ 2 = dist x y ^ 2 + dist y z ^ 2)

-- Define the theorem that we need to prove
theorem side_XY_length {X Y Z : Type} [MetricSpace X] (x y z : X)
  (h1 : is_isosceles_right_triangle x y z)
  (h2 : "Isosceles right triangle with XYZ, XY > YZ") -- Reformulate accordingly
  (h3 : area_of_triangle x y z = 36) :
  dist x y = 12 :=
begin
  sorry
end

end side_XY_length_l650_650519


namespace friends_professions_l650_650236

theorem friends_professions
  (architect barista veterinarian guitarist : ℕ)
  (P : ℕ → Sort)
  (Andrey Boris Vyacheslav Gennady : P)
  (seats : P → ℕ)
  (h1 : ∀ {x y}, seats x = veterinarian → seats y = architect → (seats x = seats y + 1 ∨ seats x = seats y - 1))
  (h2 : ∀ {x y}, seats x = barista → seats y = Boris → seats y = seats x + 1)
  (h3 : seats Vyacheslav > seats Andrey ∧ seats Vyacheslav > seats Boris)
  (h4 : ∀ {x}, x = Andrey → ∃ y z, (seats y = seats x + 1 ∨ seats y = seats x - 1) ∧ (seats z = seats x + 1 ∨ seats z = seats x - 1))
  (h5 : ∀ {x y}, seats x = guitarist → seats y = barista → seats x ≠ seats y + 1 ∧ seats x ≠ seats y - 1)
  : (∀ x, (x = Gennady → seats x = barista)
    ∧ (x = Boris → seats x = architect)
    ∧ (x = Andrey → seats x = veterinarian)
    ∧ (x = Vyacheslav → seats x = guitarist)) :=
by
  sorry

end friends_professions_l650_650236


namespace math_proof_problem_l650_650847

-- Definitions as conditions
def parametric_equation := 
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (∀ φ : ℝ, ((M : ℝ × ℝ) = (2, Real.sqrt 3)) →
    (M.fst = a * Real.cos (π / 3)) ∧ (M.snd = b * Real.sin (π / 3)))

def polar_equation_C2 (R : ℝ) : Prop := ∀ θ : ℝ, D = (Real.sqrt 2, π / 4) → 
  ρ = 2 * R * Real.cos θ ∧ R = 1

def standard_equation_C1 : Prop := ∀ x y : ℝ, 
  (x = a * Real.cos φ ∧ y = b * Real.sin φ) → x ^ 2 / 16 + y ^ 2 / 4 = 1

def sum_inverse_squared_radii (ρ1 ρ2 θ : ℝ) : Prop := 
  1 / ρ1 ^ 2 + 1 / ρ2 ^ 2 = 5 / 16

-- Lean statement
theorem math_proof_problem (a b : ℝ)
  (M : ℝ × ℝ = (2, Real.sqrt 3))
  (param_eq : parametric_equation a b M)
  (polar_eq_C2 : polar_equation_C2 1)
  (std_eq_C1 : standard_equation_C1 a b)
  (sum_inv_sq_radii : sum_inverse_squared_radii ρ1 ρ2 θ)
  : 
  (∀ x y : ℝ, std_eq_C1 x y) ∧ 
  (ρ = 2 * Real.cos θ → polar_eq_C2 1) ∧ 
  (1 / ρ1 ^ 2 + 1 / ρ2 ^ 2 = 5 / 16) := by
  exact sorry

end math_proof_problem_l650_650847


namespace correlation_coefficient_interpretation_l650_650654

-- Definitions and problem statement

/-- 
Theorem: Correct interpretation of the correlation coefficient r.
Given r in (-1, 1):
The closer |r| is to zero, the weaker the correlation between the two variables.
-/
theorem correlation_coefficient_interpretation (r : ℝ) (h : -1 < r ∧ r < 1) :
  (r > 0 -> false) ∧ (r > 1 -> false) ∧ (0 < r -> false) ∧ (|r| -> Prop) :=
sorry

end correlation_coefficient_interpretation_l650_650654


namespace number_of_factors_180_l650_650355

theorem number_of_factors_180 : 
  let factors180 (n : ℕ) := 180 
  in factors180 180 = (2 + 1) * (2 + 1) * (1 + 1) => factors180 180 = 18 := 
by
  sorry

end number_of_factors_180_l650_650355


namespace min_m_n_sum_l650_650935

theorem min_m_n_sum (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_eq : 45 * m = n^3) : m + n = 90 :=
sorry

end min_m_n_sum_l650_650935


namespace number_of_valid_subsets_l650_650721

def isPowerOf2 (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2^k

def validSubset (A B : set ℕ) : Prop :=
  ∀ x y ∈ A, (isPowerOf2 (x + y) → (x ∈ B ↔ y ∉ B))

theorem number_of_valid_subsets (n : ℕ) (hn : 2 ≤ n) :
    (λ A : set ℕ, ∀ A = { k | k <= 2^n ∧ 0 < k < 2^n + 1}, 
    ∃ (B : set ℕ), validSubset A B){
     2 ^ (n+1)
    } := 
     
sorry

end number_of_valid_subsets_l650_650721


namespace count_100_digit_even_numbers_l650_650330

theorem count_100_digit_even_numbers : 
  let valid_digits := {0, 1, 3}
  let num_digits := 100
  let num_even_digits := 2 * 3^98
  ∀ n : ℕ, n = num_digits → (∃ (digits : Fin n → ℕ), 
    (∀ i, digits i ∈ valid_digits) ∧ 
    digits 0 ≠ 0 ∧ 
    digits (n-1) = 0) → 
    (num_even_digits = 2 * 3^98) :=
by
  sorry

end count_100_digit_even_numbers_l650_650330


namespace nancy_metal_beads_l650_650893

variable (M P : ℕ)
variable (crystalBeads stoneBeads bracelets beadsPerBracelet totalBeads : ℕ)

theorem nancy_metal_beads :
  -- Conditions
  (P = M + 20) →
  (crystalBeads = 20) →
  (stoneBeads = 2 * crystalBeads) →
  (bracelets = 20) →
  (beadsPerBracelet = 8) →
  (totalBeads = 160) →
  -- Proof
  (M + P + crystalBeads + stoneBeads = totalBeads) →
  M = 40 :=
begin
  intros hp hcrystal hstone hbracelets hbeads hbtotal htotal,
  have hrosesum := hstone.symm.trans hcrystal,
  have hsum := htotal,
  obtain rfl : P = M + 20 := hp,
  rw [hrosesum] at hsum,
  sorry, -- Further proof steps to show M = 40
end

end nancy_metal_beads_l650_650893


namespace count_100_digit_even_numbers_l650_650331

theorem count_100_digit_even_numbers : 
  let valid_digits := {0, 1, 3}
  let num_digits := 100
  let num_even_digits := 2 * 3^98
  ∀ n : ℕ, n = num_digits → (∃ (digits : Fin n → ℕ), 
    (∀ i, digits i ∈ valid_digits) ∧ 
    digits 0 ≠ 0 ∧ 
    digits (n-1) = 0) → 
    (num_even_digits = 2 * 3^98) :=
by
  sorry

end count_100_digit_even_numbers_l650_650331


namespace prob_a_prob_b_prob_c_prob_d_l650_650857

-- Definitions corresponding to the conditions
def isLatticePoint(x : ℤ, y : ℤ) : Prop := x ∈ ℤ ∧ y ∈ ℤ

-- Proof Problem for a)
theorem prob_a : (P : ℤ × ℤ), (P = (0, 0)) → (n = 1) → 
  (reachable_points n).card = 4 :=
sorry

-- Proof Problem for b)
theorem prob_b : (P : ℤ × ℤ), (P = (0, 0)) → (n ≤ 2) → 
  (reachable_points n).card = 13 :=
sorry

-- Proof Problem for c)
theorem prob_c : (P : ℤ × ℤ), (P = (0, 0)) → (n = 3) → 
  (reachable_points n).card = 16 :=
sorry

-- Proof Problem for d)
theorem prob_d : (P : ℤ × ℤ), (P = (0, 0)) → (line_eq : P.1 + P.2 = 9) → (n = 9) → 
  (favorable_outcomes / total_outcomes) = 1/10 :=
sorry

end prob_a_prob_b_prob_c_prob_d_l650_650857


namespace symmetric_axis_of_parabola_l650_650506

theorem symmetric_axis_of_parabola :
  (∃ x : ℝ, x = 6 ∧ (∀ y : ℝ, y = 1/2 * x^2 - 6 * x + 21)) :=
sorry

end symmetric_axis_of_parabola_l650_650506


namespace price_per_glass_correct_l650_650896

def price_per_glass_on_day_two (O : ℝ) : ℝ :=
  let P := (2 * O * 0.50) / (3 * O)
  P

theorem price_per_glass_correct (O : ℝ) (h : O ≠ 0) : price_per_glass_on_day_two O = 0.33 := 
by
  unfold price_per_glass_on_day_two
  field_simp [h]
  norm_num
  done

-- Usage example
#eval price_per_glass_on_day_two 1 -- This should output 0.33

end price_per_glass_correct_l650_650896


namespace average_income_proof_l650_650941

theorem average_income_proof:
  ∀ (A B C : ℝ),
    (A + B) / 2 = 5050 →
    (B + C) / 2 = 6250 →
    A = 4000 →
    (A + C) / 2 = 5200 := by
  sorry

end average_income_proof_l650_650941


namespace range_of_a_sq_l650_650226

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ m n : ℕ, a (m + n) = a m + a n

theorem range_of_a_sq {n : ℕ}
  (h_arith : arithmetic_sequence a)
  (h_cond : a 1 ^ 2 + a (2 * n + 1) ^ 2 = 1) :
  ∃ (L R : ℝ), (L = 2) ∧ (∀ k : ℕ, a (n+1) ^ 2 + a (3*n+1) ^ 2 ≥ L) := sorry

end range_of_a_sq_l650_650226


namespace largest_prime_factor_1729_l650_650619

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650619


namespace divisors_squared_prime_l650_650123

theorem divisors_squared_prime (p : ℕ) [hp : Fact (Nat.Prime p)] (m : ℕ) (h : m = p^3) (hm_div : Nat.divisors m = 4) :
  Nat.divisors (m^2) = 7 :=
sorry

end divisors_squared_prime_l650_650123


namespace base_8_not_divisible_by_five_l650_650228

def base_b_subtraction_not_divisible_by_five (b : ℕ) : Prop :=
  let num1 := 3 * b^3 + 1 * b^2 + 0 * b + 2
  let num2 := 3 * b^2 + 0 * b + 2
  let diff := num1 - num2
  ¬ (diff % 5 = 0)

theorem base_8_not_divisible_by_five : base_b_subtraction_not_divisible_by_five 8 := 
by
  sorry

end base_8_not_divisible_by_five_l650_650228


namespace pipes_fill_time_l650_650079

def time_to_fill_completely (rate_pipe_a rate_pipe_b : ℝ) : ℝ := 
  let combined_rate := rate_pipe_a + rate_pipe_b
  1 / combined_rate

theorem pipes_fill_time 
  (h1 : 1 / 36 = (1/3) * (1/12))
  (h2 : 1 / 24 = (1/3) * (1/8)) :
  time_to_fill_completely (1/36) (1/24) = 14.4 :=
by {
  sorry
}

end pipes_fill_time_l650_650079


namespace proof_ratios_form_arithmetic_sequence_l650_650873

open Real

variables (A B C M P Q N : Point)
variable (AM : Line)
variables (AB AC AM_inter : Line)
variable (k d : ℝ)

-- Conditions
axiom median_AM : is_median A B C AM
axiom intersection_P : P ∈ AB ∧ P ∈ line_intersection AM AB
axiom intersection_Q : Q ∈ AC ∧ Q ∈ line_intersection AM AC
axiom intersection_N : N ∈ AM ∧ N ∈ line_intersection AM (line_intersection AB AC)

-- Proposition (to be proved)
noncomputable def ratios_form_arithmetic_sequence : Prop :=
  let AB_AP := (dist A B) / (dist A P)
  let AM_AN := (dist A M) / (dist A N)
  let AC_AQ := (dist A C) / (dist A Q) in
  (2 * AM_AN = AB_AP + AC_AQ)

theorem proof_ratios_form_arithmetic_sequence :
  ratios_form_arithmetic_sequence A B C M P Q N AM AB AC AM_inter :=
sorry

end proof_ratios_form_arithmetic_sequence_l650_650873


namespace makes_at_least_one_shot_l650_650078
noncomputable section

/-- The probability of making the free throw. -/
def free_throw_make_prob : ℚ := 4/5

/-- The probability of making the high school 3-pointer. -/
def high_school_make_prob : ℚ := 1/2

/-- The probability of making the professional 3-pointer. -/
def pro_make_prob : ℚ := 1/3

/-- The probability of making at least one of the three shots. -/
theorem makes_at_least_one_shot :
  (1 - ((1 - free_throw_make_prob) * (1 - high_school_make_prob) * (1 - pro_make_prob))) = 14 / 15 :=
by
  sorry

end makes_at_least_one_shot_l650_650078


namespace card_A_is_1_and_3_l650_650977

-- Definition of the cards
def card_set : set (set ℕ) := {{1, 2}, {1, 3}, {2, 3}}

-- Definitions for peoples' cards
variables {A B C : set ℕ}

-- Function to find the common number between two cards
def common_num (x y : set ℕ) : set ℕ := x ∩ y

-- Conditions of the problem
axiom A_statement : ∀ (B : set ℕ), A ∈ card_set → B ∈ card_set → (common_num A B) ≠ {2}
axiom B_statement : ∀ (C : set ℕ), B ∈ card_set → C ∈ card_set → (common_num B C) ≠ {1}
axiom C_statement : C ∈ card_set → 1 + 2 ≠ 5 → 1 + 3 ≠ 5

-- The question and the desired proof goal
theorem card_A_is_1_and_3 : A = {1, 3} :=
sorry

end card_A_is_1_and_3_l650_650977


namespace largest_prime_factor_1729_l650_650584

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650584


namespace data_set_variance_l650_650696

theorem data_set_variance :
  let data := [6, 8, 10, 12, 14]
  let mean := (6 + 8 + 10 + 12 + 14 : ℝ) / 5
  let variance := ((6 - mean)^2 + (8 - mean)^2 + (10 - mean)^2 + (12 - mean)^2 + (14 - mean)^2) / 5
  variance = 8 :=
by
  let data := [6, 8, 10, 12, 14]
  let mean := (6 + 8 + 10 + 12 + 14 : ℝ) / 5
  let variance := ((6 - mean)^2 + (8 - mean)^2 + (10 - mean)^2 + (12 - mean)^2 + (14 - mean)^2) / 5
  show variance = 8 from sorry

end data_set_variance_l650_650696


namespace probability_of_inequality_l650_650390

noncomputable def probability_event (x : ℝ) (A : set ℝ) (interval : set ℝ) : ℝ :=
  if x ∈ A then (card (A ∩ interval) : ℝ) / (card interval : ℝ) else 0

theorem probability_of_inequality (x : ℝ) (h : x ∈ Ioc (-1 : ℝ) 4) :
  probability_event x (set_of (λ x, 2 * x - 2 * x^2 ≥ -4)) (Ioc (-1 : ℝ) 4) = 3 / 5 :=
by sorry

end probability_of_inequality_l650_650390


namespace slope_of_intersecting_chord_l650_650983

theorem slope_of_intersecting_chord : 
  ∀ (C D : ℝ × ℝ),
  let circle1_eq := λ x y : ℝ, x^2 + y^2 + 6 * x - 8 * y - 40 = 0,
      circle2_eq := λ x y : ℝ, x^2 + y^2 + 22 * x - 2 * y + 20 = 0,
      line_slope := (16 / 6 : ℝ) in
  (circle1_eq C.fst C.snd) → (circle2_eq C.fst C.snd) →
  (circle1_eq D.fst D.snd) → (circle2_eq D.fst D.snd) →
  let slope_cd := (D.snd - C.snd) / (D.fst - C.fst) in
  slope_cd = line_slope
:= sorry

end slope_of_intersecting_chord_l650_650983


namespace inequality_proof_l650_650759

theorem inequality_proof (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
    (a * b + b * c + c * a) * (1 / (a + b)^2 + 1 / (b + c)^2 + 1 / (c + a)^2) ≥ 9 / 4 := 
by
  sorry

end inequality_proof_l650_650759


namespace combined_work_time_l650_650937

theorem combined_work_time (s c r : ℕ) (h_s : s = 45) (h_c : c = 30) (h_r : r = 60) :
  (1 / ((1 / s) + (1 / c) + (1 / r))) ≈ 13.85 :=
by
  have h_lcm : Nat.lcm (Nat.lcm s c) r = 180,
    from calc
      Nat.lcm (Nat.lcm s c) r = Nat.lcm (Nat.lcm 45 30) 60 : by rw [h_s, h_c, h_r]
      ... = 180 : by norm_num,
  have h_rate_s : (1: ℝ) / 45 = 4 / 180, by norm_num,
  have h_rate_c : (1: ℝ) / 30 = 6 / 180, by norm_num,
  have h_rate_r : (1: ℝ) / 60 = 3 / 180, by norm_num,
  have combined_rate : ((4 + 6 + 3): ℝ) / 180 = 13 / 180,
    from by norm_num,
  have time_to_complete : (1: ℝ) / (13 / 180) = 180 / 13,
    from by rw [one_div, div_div_eq_mul_div, mul_one],
  have final_calc : (180 / 13): ℝ ≈ 13.85,
    from by norm_num,
  exact final_calc

end combined_work_time_l650_650937


namespace find_n_l650_650749

theorem find_n (n : ℕ) (h₁ : n > 0) (h₂ : ∏ (d : ℕ) in (finset.range (n + 1)).filter (λ d, n % d = 0), d = 24^240) : n = 24^5 := 
sorry

end find_n_l650_650749


namespace largest_prime_factor_of_1729_l650_650566

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650566


namespace find_x_2187_l650_650793

theorem find_x_2187 (x : ℂ) (h : x - 1/x = complex.I * real.sqrt 3) : x^2187 - 1/(x^2187) = 0 :=
sorry

end find_x_2187_l650_650793


namespace trajectory_and_exists_E_l650_650303

-- Define the fixed point F and the line of distance ratio
def F : ℝ × ℝ := (-Real.sqrt 3, 0)
def line_x := -4 * Real.sqrt 3 / 3

-- Condition: distance ratio
def distance_ratio (M : ℝ × ℝ) : ℝ :=
  let d1 := Real.sqrt ((M.1 + Real.sqrt 3)^2 + M.2^2)
  let d2 := Real.abs (M.1 + 4 * Real.sqrt 3 / 3)
  d1 / d2

-- The trajectory equation C as an ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4 + y^2 = 1)

-- Determine if E exists and the condition for constancy
def exists_E (E : ℝ × ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (E : ℝ × ℝ), ∀ (M N : ℝ × ℝ), (on_line E M N) ∧ C M.1 M.2 ∧ C N.1 N.2 →
    (1 / (distance E M)^2 + 1 / (distance E N)^2 = 5)

-- Main theorem
theorem trajectory_and_exists_E :
  (∀ M, distance_ratio M = k → ellipse_eq M.1 M.2) ∧
  ∃ E, exists_E E ellipse_eq :=
sorry

end trajectory_and_exists_E_l650_650303


namespace find_total_amount_l650_650099

noncomputable def total_amount (A T yearly_income : ℝ) : Prop :=
  0.05 * A + 0.06 * (T - A) = yearly_income

theorem find_total_amount :
  ∃ T : ℝ, total_amount 1600 T 140 ∧ T = 2600 :=
sorry

end find_total_amount_l650_650099


namespace probability_A_or_B_selected_probability_B_probability_B_given_A_l650_650102

noncomputable def total_ways : ℕ := (Nat.choose 6 3)

noncomputable def ways_A_or_B_selected : ℕ := (Nat.choose 2 1) * (Nat.choose 4 2) + (Nat.choose 2 2) * (Nat.choose 4 1)

noncomputable def P_A_or_B : ℚ := ways_A_or_B_selected / total_ways

def event_A : Event := { ω | ω.contains "A" }

def event_B : Event := { ω | ω.contains "B" }

noncomputable def P_B : ℚ := (Nat.choose 5 2) / total_ways

noncomputable def P_AB : ℚ := (Nat.choose 4 1) / total_ways

noncomputable def P_B_given_A : ℚ := P_AB / P_B

theorem probability_A_or_B_selected :
  P_A_or_B = 4 / 5 := 
by
  sorry

theorem probability_B :
  P_B = 1 / 2 :=
by
  sorry

theorem probability_B_given_A :
  P_B_given_A = 2 / 5 :=
by
  sorry

end probability_A_or_B_selected_probability_B_probability_B_given_A_l650_650102


namespace seating_arrangement_l650_650242

-- Definitions of friends and their professions
inductive Friend
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq

-- Seating condition definitions
def seated_right (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = a ∧ seating (i + 1) = b)

def seated_between (a b c : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = b ∧ seating (i - 1) = a ∧ seating (i + 1) = c)

def neighbors (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, (seating i = a ∧ seating (i - 1) = b) ∨ (seating i = a ∧ seating (i + 1) = b))

-- The proposition to be proved
theorem seating_arrangement
  (seating : ℕ → Friend)
  (profession : Friend → Profession)
  (h1 : ∃ i, seating_between Profession.Veterinarian Profession.Architect Profession.Guitarist seating)
  (h2 : seated_right Profession.Barista Profession.Boris seating)
  (h3 : ∃ i, seating i = Friend.Vyacheslav ∧ (∃ j, seating j = Friend.Andrey ∧ j < i) ∧ (∃ k, seating k = Friend.Boris ∧ k < i))
  (h4 : ∀ a, seating a = Friend.Andrey → neighbors Friend.Andrey seating)
  (h5 : ∀ a, seating a = Profession.Guitarist → ∀ b, seating (b - 1) = Profession.Barista ∨ seating (b + 1) = Profession.Barista → false) :
  (profession Friend.Gennady = Profession.Barista ∧
   profession Friend.Boris = Profession.Architect ∧
   profession Friend.Andrey = Profession.Veterinarian ∧
   profession Friend.Vyacheslav = Profession.Guitarist) :=
begin
  sorry
end

end seating_arrangement_l650_650242


namespace find_smallest_integer_l650_650056

noncomputable def smallest_integer (x y z : ℝ) : ℝ :=
if x ≤ y ∧ x ≤ z then x else
if y ≤ x ∧ y ≤ z then y else z

theorem find_smallest_integer :
  ∃ (x y z : ℝ), x + y = 23 ∧ x + z = 31 ∧ y + z = 11 ∧ smallest_integer x y z = 21.5 :=
by
  use 21.5, 1.5, 9.5
  split
  -- x + y = 23
  { norm_num }
  split
  -- x + z = 31
  { norm_num }
  split
  -- y + z = 11
  { norm_num }
  -- smallest_integer 21.5 1.5 9.5 = 21.5
  { rw smallest_integer
    split_ifs
    all_goals { norm_num } }

end find_smallest_integer_l650_650056


namespace lucas_cleaning_days_l650_650885

-- Definitions of conditions
def windows_per_floor := 3
def floors := 3
def pay_per_window := 2
def deduction_per_period := 1
def days_per_period := 3
def total_payment := 16

-- Proof problem statement
theorem lucas_cleaning_days : 
  let total_windows := windows_per_floor * floors in
  let potential_earnings := total_windows * pay_per_window in
  let total_deductions := potential_earnings - total_payment in
  let periods := total_deductions / deduction_per_period in
  let total_days := periods * days_per_period in
  total_days = 6 :=
by 
  sorry

end lucas_cleaning_days_l650_650885


namespace new_average_is_10_5_l650_650082

-- define the conditions
def average_of_eight_numbers (numbers : List ℝ) : Prop :=
  numbers.length = 8 ∧ (numbers.sum / 8) = 8

def add_four_to_five_numbers (numbers : List ℝ) (new_numbers : List ℝ) : Prop :=
  new_numbers = (numbers.take 5).map (λ x => x + 4) ++ numbers.drop 5

-- state the theorem
theorem new_average_is_10_5 (numbers new_numbers : List ℝ) 
  (h1 : average_of_eight_numbers numbers)
  (h2 : add_four_to_five_numbers numbers new_numbers) :
  (new_numbers.sum / 8) = 10.5 := 
by 
  sorry

end new_average_is_10_5_l650_650082


namespace seating_profession_solution_l650_650262

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l650_650262


namespace sum_first_four_terms_geometric_sequence_l650_650418

theorem sum_first_four_terms_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (a₁ : ℝ)
    (h₁ : a 2 = 9)
    (h₂ : a 5 = 243)
    (h₃ : ∀ n, a (n + 1) = a n * r) :
    a₁ + a₁ * r + a₁ * r^2 + a₁ * r^3 = 120 := 
by 
  sorry

end sum_first_four_terms_geometric_sequence_l650_650418


namespace sequence_geometric_chain_sum_of_sequence_l650_650287

-- Problem (1): Prove that sequence is geometric and find general formula
theorem sequence_geometric_chain (a_n S_n : ℕ → ℝ) (h : ∀ n, S_n + ↑n = 2 * a_n): 
  (∀ n, a_n + 1 = 2 * 2^(n - 1) ) ∧ (∀ n, a_n = 2^n - 1) := 
by 
  sorry

-- Problem (2): Find sum of sequence b_n
theorem sum_of_sequence (a_n b_n S_n : ℕ → ℝ) (h₁ : ∀ n, S_n + ↑n = 2 * a_n) (h₂ : ∀ n, b_n = a_n + 2*↑n + 1) : 
  ∀ n, (∑ k in finset.range n, b_n k) = 2^(n+1) + n^2 + n - 2 := 
by 
  sorry

end sequence_geometric_chain_sum_of_sequence_l650_650287


namespace geometric_sequence_a4_l650_650416

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ {m n p q}, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h : geometric_sequence a) (h2 : a 2 = 4) (h6 : a 6 = 16) :
  a 4 = 8 :=
by {
  -- Here you can provide the proof steps if needed
  sorry
}

end geometric_sequence_a4_l650_650416


namespace solve_equation_l650_650485

theorem solve_equation : ∀ x : ℝ,
  (3 ^ (2 * x ^ 2 + 6 * x - 9) + 4 * 15 ^ (x ^ 2 + 3 * x - 5) = 3 * 5 ^ (2 * x ^ 2 + 6 * x - 9))
  ↔ (x = -4 ∨ x = 1) :=
by
  intro x
  -- Placeholder for proof
  sorry

end solve_equation_l650_650485


namespace solveProfessions_l650_650251

-- Define our types for Friends and Professions
inductive Friend | Andrey | Boris | Vyacheslav | Gennady
inductive Profession | Architect | Barista | Veterinarian | Guitarist

-- Define the seating arrangement and conditions
def seatingArrangement : Friend → Position := sorry
def profession : Friend → Profession := sorry

-- Conditions translated to Lean
axiom condition1 : ∀ (f : Friend), 
  profession f = Profession.Veterinarian → 
  ∃ (neighbor1 neighbor2 : Friend), 
    seatingArrangement neighbor1 = seatingArrangement f + 1 ∨ seatingArrangement neighbor1 = seatingArrangement f - 1 ∧ 
    seatingArrangement neighbor2 = seatingArrangement f + 1 ∨ seatingArrangement neighbor2 = seatingArrangement f - 1 ∧
    (profession neighbor1 = Profession.Architect ∧ profession neighbor2 = Profession.Guitarist) 

axiom condition2 : ∀ (f : Friend), 
  profession f = Profession.Barista → 
  ∃ (rightNeighbor : Friend), 
    seatingArrangement rightNeighbor = seatingArrangement f + 1 ∧ 
    rightNeighbor = Friend.Boris

axiom condition3 : seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Andrey ∧ seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Boris

axiom condition4 : ∀ (f : Friend), 
  f = Friend.Andrey → 
  ∃ (leftNeighbor rightNeighbor : Friend), 
    seatingArrangement leftNeighbor = seatingArrangement f - 1 ∧ 
    seatingArrangement rightNeighbor = seatingArrangement f + 1

axiom condition5 : ∀ (f1 f2 : Friend), 
  (profession f1 = Profession.Guitarist ∧ profession f2 = Profession.Barista) → 
  seatingArrangement f1 ≠ seatingArrangement f2 + 1 ∧ seatingArrangement f1 ≠ seatingArrangement f2 - 1

-- Define the positions
def Position : Type := Fin 4

-- The main theorem to prove the assignments
theorem solveProfessions :
  profession Friend.Gennady = Profession.Barista ∧ 
  profession Friend.Boris = Profession.Architect ∧ 
  profession Friend.Andrey = Profession.Veterinarian ∧ 
  profession Friend.Vyacheslav = Profession.Guitarist :=
sorry

end solveProfessions_l650_650251


namespace alex_mean_score_l650_650770

theorem alex_mean_score (scores : list ℝ) (jane_scores : list ℝ) (alex_scores : list ℝ)
  (h : scores = [86, 88, 90, 91, 95, 99])
  (h_length_jane : jane_scores.length = 2)
  (h_length_alex : alex_scores.length = 4)
  (h_partition : jane_scores ++ alex_scores = scores)
  (h_jane_mean : (jane_scores.sum) / (jane_scores.length) = 93) :
  (alex_scores.sum) / (alex_scores.length) = 90.75 := by
  sorry

end alex_mean_score_l650_650770


namespace largest_prime_factor_1729_l650_650570

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650570


namespace find_a_b_find_k_l650_650319

/-- The mathematical problem given the conditions and required proofs -/
noncomputable def g (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1 + b

noncomputable def f (a b x : ℝ) : ℝ := g a b (abs x)

theorem find_a_b (h₁ : ∀ x ∈ Icc 2 4, g a b x ≤ 9 ∧ g a b x ≥ 1) (pos_a : 0 < a) :
  a = 1 ∧ b = 0 :=
sorry

theorem find_k (pos_a : 0 < a) (k : ℝ) :
  f 1 0 (Real.log2 k) > f 1 0 2 ↔ k > 4 ∨ (0 < k ∧ k < 1/4) :=
sorry

end find_a_b_find_k_l650_650319


namespace quotient_is_61_l650_650968

theorem quotient_is_61 (quotient : ℕ) (divisor : ℕ) (remainder : ℕ) (dividend : ℕ)
  (h1 : remainder = 19) (h2 : dividend = 507) (h3 : divisor = 8) :
  dividend = (divisor * quotient) + remainder → quotient = 61 :=
by
  intro h,
  sorry

end quotient_is_61_l650_650968


namespace largest_prime_factor_of_1729_l650_650597

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650597


namespace festival_allowance_daily_rate_l650_650405

def total_festival_allowance (amount_given accountant_amount petty_cash_amount days staff_members : ℕ) : ℕ :=
amount_given + petty_cash_amount

def daily_rate (total_festival_allowance days staff_members : ℕ) : ℕ :=
total_festival_allowance / (days * staff_members)

theorem festival_allowance_daily_rate :
  let amount_given := 65000
  let petty_cash_amount := 1000
  let days := 30
  let staff_members := 20
  let total := total_festival_allowance amount_given accountant_amount petty_cash_amount days staff_members
  daily_rate total amount_given days staff_members = 110 := 
by
  sorry

end festival_allowance_daily_rate_l650_650405


namespace expected_number_of_first_sequence_l650_650148

-- Define the concept of harmonic number
def harmonic (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), (1 : ℚ) / (i + 1)

-- Define the problem statement
theorem expected_number_of_first_sequence (n : ℕ) (h : n = 100) : harmonic n = 5.187 := by
  sorry

end expected_number_of_first_sequence_l650_650148


namespace seating_arrangement_l650_650241

-- Definitions of friends and their professions
inductive Friend
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq

-- Seating condition definitions
def seated_right (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = a ∧ seating (i + 1) = b)

def seated_between (a b c : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = b ∧ seating (i - 1) = a ∧ seating (i + 1) = c)

def neighbors (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, (seating i = a ∧ seating (i - 1) = b) ∨ (seating i = a ∧ seating (i + 1) = b))

-- The proposition to be proved
theorem seating_arrangement
  (seating : ℕ → Friend)
  (profession : Friend → Profession)
  (h1 : ∃ i, seating_between Profession.Veterinarian Profession.Architect Profession.Guitarist seating)
  (h2 : seated_right Profession.Barista Profession.Boris seating)
  (h3 : ∃ i, seating i = Friend.Vyacheslav ∧ (∃ j, seating j = Friend.Andrey ∧ j < i) ∧ (∃ k, seating k = Friend.Boris ∧ k < i))
  (h4 : ∀ a, seating a = Friend.Andrey → neighbors Friend.Andrey seating)
  (h5 : ∀ a, seating a = Profession.Guitarist → ∀ b, seating (b - 1) = Profession.Barista ∨ seating (b + 1) = Profession.Barista → false) :
  (profession Friend.Gennady = Profession.Barista ∧
   profession Friend.Boris = Profession.Architect ∧
   profession Friend.Andrey = Profession.Veterinarian ∧
   profession Friend.Vyacheslav = Profession.Guitarist) :=
begin
  sorry
end

end seating_arrangement_l650_650241


namespace triangles_congruence_l650_650520

-- Definitions and conditions
def intersecting_circles (P Q A A' B B' : Point) (circle1 circle2 : Circle) : Prop :=
  (P ∈ circle1) ∧ (Q ∈ circle1) ∧ 
  (P ∈ circle2) ∧ (Q ∈ circle2) ∧
  (A ∈ circle1) ∧ (A' ∈ circle2) ∧ 
  (B ∈ circle1) ∧ (B' ∈ circle2) ∧
  lies_on_line P A A' ∧ 
  lies_on_line Q B B' ∧
  is_parallel (line_through P A) (line_through Q B)

-- Theorem to prove
theorem triangles_congruence (P Q A A' B B' : Point) (circle1 circle2 : Circle) :
  intersecting_circles P Q A A' B B' circle1 circle2 →
  congruent (triangle P B B') (triangle Q A A') :=
begin
  intros hc,
  sorry -- Proof to be completed
end

end triangles_congruence_l650_650520


namespace tetrahedrons_in_sphere_l650_650938

-- Define the conditions: 
-- Ten points on the surface of a sphere.
-- Lines connecting every pair of points.
-- No three lines intersect in a single point inside the sphere.

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: 
-- Prove that the number of tetrahedrons formed by the lines intersecting 
-- inside a sphere given 10 points is 210.
theorem tetrahedrons_in_sphere (n : ℕ) (h : n = 10) : binom n 4 = 210 :=
by
  rw h
  have : binom 10 4 = 210 := by sorry
  exact this

end tetrahedrons_in_sphere_l650_650938


namespace find_n_l650_650951

theorem find_n (n : ℕ) (h1 : Nat.gcd n 180 = 12) (h2 : Nat.lcm n 180 = 720) : n = 48 := 
by
  sorry

end find_n_l650_650951


namespace seating_profession_solution_l650_650263

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l650_650263


namespace number_of_cats_adopted_l650_650427

theorem number_of_cats_adopted (c : ℕ) 
  (h1 : 50 * c + 3 * 100 + 2 * 150 = 700) :
  c = 2 :=
by
  sorry

end number_of_cats_adopted_l650_650427


namespace inequality_sqrt_sum_ge_one_l650_650861

variable (a b c : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variable (prod_abc : a * b * c = 1)

theorem inequality_sqrt_sum_ge_one :
  (Real.sqrt (a / (8 + a)) + Real.sqrt (b / (8 + b)) + Real.sqrt (c / (8 + c)) ≥ 1) :=
by
  sorry

end inequality_sqrt_sum_ge_one_l650_650861


namespace senya_triangle_area_l650_650000

theorem senya_triangle_area :
  ∃ a b c : ℝ, a = 18 ∧ b = 24 ∧ c = 30 ∧ a^2 + b^2 = c^2 ∧ (1/2) * a * b = 216 := 
by 
  use [18, 24, 30]
  split
  repeat {split}
  all_goals { ring_nf, norm_num, linarith }

end senya_triangle_area_l650_650000


namespace pasture_never_exceeds_K_l650_650069

noncomputable def smallest_K : Real :=
  let initial_area := (sqrt 3) / 2
  2 * initial_area

theorem pasture_never_exceeds_K :
  (∀ n : ℕ, let area := (2^n - 1) / (2^n - 2) * (sqrt 3) / 2 in area < smallest_K) ∧ 
  (√3 = smallest_K) :=
by
  sorry

end pasture_never_exceeds_K_l650_650069


namespace friends_professions_l650_650230

theorem friends_professions
  (architect barista veterinarian guitarist : ℕ)
  (P : ℕ → Sort)
  (Andrey Boris Vyacheslav Gennady : P)
  (seats : P → ℕ)
  (h1 : ∀ {x y}, seats x = veterinarian → seats y = architect → (seats x = seats y + 1 ∨ seats x = seats y - 1))
  (h2 : ∀ {x y}, seats x = barista → seats y = Boris → seats y = seats x + 1)
  (h3 : seats Vyacheslav > seats Andrey ∧ seats Vyacheslav > seats Boris)
  (h4 : ∀ {x}, x = Andrey → ∃ y z, (seats y = seats x + 1 ∨ seats y = seats x - 1) ∧ (seats z = seats x + 1 ∨ seats z = seats x - 1))
  (h5 : ∀ {x y}, seats x = guitarist → seats y = barista → seats x ≠ seats y + 1 ∧ seats x ≠ seats y - 1)
  : (∀ x, (x = Gennady → seats x = barista)
    ∧ (x = Boris → seats x = architect)
    ∧ (x = Andrey → seats x = veterinarian)
    ∧ (x = Vyacheslav → seats x = guitarist)) :=
by
  sorry

end friends_professions_l650_650230


namespace expected_first_sequence_100_l650_650147

noncomputable def expected_first_sequence (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1).filter (λ k, k > 0), (1 : ℝ) / k

theorem expected_first_sequence_100 : expected_first_sequence 100 = 
    ∑ k in (finset.range 101).filter (λ k, k > 0), (1 : ℝ) / k :=
by
  -- The proof would involve showing this sum represents the harmonic number H_100
  sorry

end expected_first_sequence_100_l650_650147


namespace quadratic_function_solution_l650_650217

theorem quadratic_function_solution (c d : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f(x) = x^2 + c * x + d)
  (h2 : ∀ x, (f(f(x) + x)) / (f(x)) = x^2 + 1779 * x + 2013) :
  c = 1777 ∧ d = 235 :=
by
  sorry

end quadratic_function_solution_l650_650217


namespace bowling_ball_weight_l650_650225

theorem bowling_ball_weight (b c : ℕ) 
  (h1 : 5 * b = 3 * c) 
  (h2 : 3 * c = 105) : 
  b = 21 := 
  sorry

end bowling_ball_weight_l650_650225


namespace collinear_P_U_V_l650_650669

-- Definitions of the points and circles as given in conditions

noncomputable theory

variables {k₁ k₂ k₃ : Type} [Circle k₁] [Circle k₂] [Circle k₃]
variables {P Q R S T U V : Point}
variables {line₁ : Line} (touch₁₂ : ∃ P, tangent k₂ k₃ P) 
variables {line₂ : Line} (touch₂₃ : ∃ Q, tangent k₃ k₁ Q)
variables {line₃ : Line} (touch₁₃ : ∃ R, tangent k₁ k₂ R) 
variables {line₄: Line} (meet_PQ: intersect point_line P Q S k₁)
variables {line₅: Line} (meet_PR: intersect point_line P R T k₁)
variables {line₆: Line} (meet_RS: intersect point_line R S U k₂)
variables {line₇: Line} (meet_QT: intersect point_line Q T V k₃)

-- Statement to prove collinearity
theorem collinear_P_U_V (h₁: ∃ P, touch₁₂) (h₂: ∃ Q, touch₂₃) (h₃: ∃ R, touch₁₃) 
(h₄: ∃ S, meet_PQ P Q S) (h₅: ∃ T, meet_PR P R T) 
(h₆: ∃ U, meet_RS R S U) (h₇: ∃ V, meet_QT Q T V) : collinear {P, U, V} :=
by {
  sorry
}

end collinear_P_U_V_l650_650669


namespace cross_product_uv_l650_650712

def u : ℝ × ℝ × ℝ := (3, -4, 1)
def v : ℝ × ℝ × ℝ := (2, 0, 5)

theorem cross_product_uv :
  (u.2.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2.1 * v.1) = (-20, -13, 8) := 
by
  unfold u v
  sorry

end cross_product_uv_l650_650712


namespace solve_quadratic_equation_l650_650970

theorem solve_quadratic_equation (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
sorry

end solve_quadratic_equation_l650_650970


namespace median_of_list_is_correct_l650_650072

-- Definitions coming from the conditions and final conclusions
def total_terms : ℕ := 9060
def median_pos1 : ℕ := total_terms / 2
def median_pos2 : ℕ := median_pos1 + 1
def median_value : ℚ := 1482.5

theorem median_of_list_is_correct :
  let list := (List.range 3020).map (λ n, n + 1) ++ -- 1, 2, …, 3020
               (List.range 3020).map (λ n, (n + 1)^2) ++ -- 1^2, 2^2, …, 3020^2
               (List.range 3020).map (λ n, (n + 1)^3) -- 1^3, 2^3, …, 3020^3
  in List.length list = total_terms ∧
     let sorted_list := list.qsort (≤)
     in ((sorted_list.get? (median_pos1 - 1)).iget.cast ℚ + 
         (sorted_list.get? (median_pos2 - 1)).iget.cast ℚ) / 2 = median_value :=
by
  sorry

end median_of_list_is_correct_l650_650072


namespace quadratic_eq_solutions_l650_650934

theorem quadratic_eq_solutions : ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ x = 3 / 2 ∨ x = 1 :=
by
  sorry

end quadratic_eq_solutions_l650_650934


namespace largest_prime_factor_of_1729_l650_650625

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650625


namespace smallest_n_for_isosceles_trapezoid_coloring_l650_650218

def isIsoscelesTrapezoid (a b c d : ℕ) : Prop :=
  -- conditions to check if vertices a, b, c, d form an isosceles trapezoid in a regular n-gon
  sorry  -- definition of an isosceles trapezoid

def vertexColors (n : ℕ) : Fin n → Fin 3 :=
  sorry  -- vertex coloring function

theorem smallest_n_for_isosceles_trapezoid_coloring :
  ∃ n : ℕ, (∀ (vertices : Fin n → Fin 3), ∃ (a b c d : Fin n),
    vertexColors n a = vertexColors n b ∧
    vertexColors n b = vertexColors n c ∧
    vertexColors n c = vertexColors n d ∧
    isIsoscelesTrapezoid a b c d) ∧ n = 17 :=
by
  sorry

end smallest_n_for_isosceles_trapezoid_coloring_l650_650218


namespace divisors_squared_prime_l650_650122

theorem divisors_squared_prime (p : ℕ) [hp : Fact (Nat.Prime p)] (m : ℕ) (h : m = p^3) (hm_div : Nat.divisors m = 4) :
  Nat.divisors (m^2) = 7 :=
sorry

end divisors_squared_prime_l650_650122


namespace quadratic_solution_l650_650931

theorem quadratic_solution : 
  ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ (x = 1 ∨ x = 3 / 2) :=
by
  intro x
  constructor
  sorry

end quadratic_solution_l650_650931


namespace quadratic_roots_value_of_k_l650_650195

theorem quadratic_roots_value_of_k (k : ℝ) : 
  (∃ x : ℂ, x^2 + (5 * (1/9) : ℝ) * x + k * (1/81 : ℝ) = 0 ∧ 
    (x = (-5 + Complex.i * Real.sqrt 371) / 18 ∨ x = (-5 - Complex.i * Real.sqrt 371) / 18)) 
  → k = 11 := 
by 
  sorry

end quadratic_roots_value_of_k_l650_650195


namespace smaller_of_two_digit_numbers_with_product_2210_l650_650038

theorem smaller_of_two_digit_numbers_with_product_2210 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2210 ∧ a ≤ b ∧ a = 26 :=
by
  sorry

end smaller_of_two_digit_numbers_with_product_2210_l650_650038


namespace birches_per_three_boys_l650_650049

theorem birches_per_three_boys (total_students total_plants total_birches : ℕ) 
    (roses_per_girl : ℕ) (plants : nat) (students : fin total_students)
    (girls_per_group planted_roses planted_birches total_girls total_boys : ℕ) :
    total_students = 24 → 
    total_plants = 24 → 
    total_birches = 6 →
    total_boys = 18 →
    total_girls = 6 →
    (girls_per_group * roses_per_girl) = 18 →
    (total_boys / 3) = 6 → 
    (total_boys / 3) * girls_per_group = total_birches →
    total_plants = total_birches + planted_roses  →
    students = total_students → sorry

end birches_per_three_boys_l650_650049


namespace largest_prime_factor_of_1729_l650_650548

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650548


namespace quadratic_eq_solutions_l650_650933

theorem quadratic_eq_solutions : ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ x = 3 / 2 ∨ x = 1 :=
by
  sorry

end quadratic_eq_solutions_l650_650933


namespace num_friends_solved_problems_l650_650685

theorem num_friends_solved_problems (x y n : ℕ) (h1 : 24 * x + 28 * y = 256) (h2 : n = x + y) : n = 10 :=
by
  -- Begin the placeholder proof
  sorry

end num_friends_solved_problems_l650_650685


namespace num_factors_180_l650_650345

theorem num_factors_180 : 
  ∃ (n : ℕ), n = 180 ∧ prime_factorization n = [(2, 2), (3, 2), (5, 1)] ∧ positive_factors n = 18 :=
by
  sorry

end num_factors_180_l650_650345


namespace sum_x_y_l650_650878

variable {x y : ℝ}

theorem sum_x_y (h₁ : x ≠ y)
    (h₂ : Matrix.det ![\[1, 6, 8\], \[4, x, y\], \[4, y, x\]] = 0) :
    x + y = 56 :=
  sorry

end sum_x_y_l650_650878


namespace probability_of_gon_l650_650502

noncomputable def probability_vectors (n : ℕ) : ℚ :=
  671 / 1007

theorem probability_of_gon (n : ℕ) (hn : n = 2015) (hf : ∀ i, ∥i∥ = n):
  let A := fin n → unit_circle_complex in
  let P := (λ (i j : A), ∥(i + j : ℂ)∥ ≥ 1) in
  probability_vectors n = 671 / 1007 :=
sorry

end probability_of_gon_l650_650502


namespace hexagon_area_l650_650722

-- Definitions for the given conditions
variables {A B C D E F G : Type}
variables (ACDF ABDE : parallelogram A C D F) (AGB CGB : set)
variables (Area_ACDF : Area ACDF = 168) (Area_ABDE: Area ABDE = 168)
variables (Intersection_AC_BD : ∃ G. (Line AC).intersects (Line BD))
variables (AreaAGBCGBCondition : Area AGB = Area CGB + 10)

-- Statement to be proven
theorem hexagon_area (h1 : Area ACDF = 168)
(h2: Area ABDE = 168)
(h3 : ∀ G, (Line AC).intersects (Line BD))
(h4 : ∀ G, Area AGB = Area CGB + 10)
(AreaAGB := let x := Area AGB,
zonesplit x Area CGB := x - 10,
total_parallelogram := Area AGB + Area CGB = 2 * 84)
: ∀ A B C D E F, hexagon_area A B C D E F = Area ACDF + Area ABDE - _) -- Here _ denotes the areas outside 
:= 168 + 168 - Assumption.

s Theorem smallest_possible_area : hexagon_area = 196 :=
begin
   sorry
end

end hexagon_area_l650_650722


namespace sally_seashells_l650_650916

theorem sally_seashells 
  (seashells_monday : ℕ)
  (seashells_tuesday : ℕ)
  (price_per_seashell : ℝ)
  (h_monday : seashells_monday = 30)
  (h_tuesday : seashells_tuesday = seashells_monday / 2)
  (h_price : price_per_seashell = 1.2) :
  let total_seashells := seashells_monday + seashells_tuesday in
  let total_money := total_seashells * price_per_seashell in
  total_money = 54 := 
by
  sorry

end sally_seashells_l650_650916


namespace area_of_rectangle_stage_8_l650_650389

theorem area_of_rectangle_stage_8 : 
  (∀ n, 4 * 4 = 16) →
  (∀ k, k ≤ 8 → k = k) →
  (8 * 16 = 128) :=
by
  intros h_sq_area h_sequence
  sorry

end area_of_rectangle_stage_8_l650_650389


namespace seating_profession_solution_l650_650259

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l650_650259


namespace num_factors_180_l650_650347

theorem num_factors_180 : 
  ∃ (n : ℕ), n = 180 ∧ prime_factorization n = [(2, 2), (3, 2), (5, 1)] ∧ positive_factors n = 18 :=
by
  sorry

end num_factors_180_l650_650347


namespace intersection_line1_perpendicular_l650_650991

variables (x y : ℝ)

def line1 (x : ℝ) : ℝ := 3 * x - 4

def perpendicular_line (x : ℝ) : ℝ := - (1 / 3) * x - 1

def intersection_point : ℝ × ℝ :=
  (9 / 10, -13 / 10)

theorem intersection_line1_perpendicular :
  ∃ x y : ℝ, (line1 x = y) ∧ (perpendicular_line x = y) ∧
    (x, y) = intersection_point :=
  sorry

end intersection_line1_perpendicular_l650_650991


namespace number_of_green_eyes_l650_650046

-- Definitions based on conditions
def total_people : Nat := 100
def blue_eyes : Nat := 19
def brown_eyes : Nat := total_people / 2
def black_eyes : Nat := total_people / 4

-- Theorem stating the main question and its answer
theorem number_of_green_eyes : 
  (total_people - (blue_eyes + brown_eyes + black_eyes)) = 6 := by
  sorry

end number_of_green_eyes_l650_650046


namespace largest_prime_factor_of_1729_l650_650552

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650552


namespace ellipse_major_axis_length_l650_650705

/-- An ellipse is tangent to both the x-axis and the y-axis, with its foci located 
at (4, 1 + 2√2) and (4, 1 - 2√2). Prove that the length of the major axis is 2. -/
theorem ellipse_major_axis_length :
  ∀ (h k : ℝ), (f1x = 4 ∧ f1y = 1 + 2 * real.sqrt 2 ∧ f2x = 4 ∧ f2y = 1 - 2 * real.sqrt 2) →
  (is_tangent_to_x_axis ∧ is_tangent_to_y_axis) →
  (axis_length = 2) :=
by
  intros h k 
  have f1x := 4
  have f1y := 1 + 2 * real.sqrt 2
  have f2x := 4
  have f2y := 1 - 2 * real.sqrt 2
  have is_tangent_to_x_axis := true
  have is_tangent_to_y_axis := true
  have axis_length := 2
  sorry

end ellipse_major_axis_length_l650_650705


namespace areas_difference_l650_650680

theorem areas_difference (r : ℝ) (s : ℝ) (h_r : r = 3) (h_s : s = 9) :
  let circle_area := π * r^2,
      triangle_area := (sqrt 3 / 4) * s^2,
      intersecting_area := (1 / 2) * (s / 2) * ((sqrt 3 / 2) * s),
      area_circle_outside_triangle := circle_area - intersecting_area,
      area_triangle_outside_circle := triangle_area - intersecting_area
  in area_circle_outside_triangle - area_triangle_outside_circle = 9 * π - (81 * sqrt 3) / 4 :=
by
  sorry

end areas_difference_l650_650680


namespace largest_prime_factor_of_1729_l650_650592

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650592


namespace largest_prime_factor_1729_l650_650621

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650621


namespace locate_quadrant_l650_650192

def z : ℂ := (5 * complex.I) / (1 + 2 * complex.I)

def conjugate_z : ℂ := complex.conj z

def point := (conjugate_z.re, conjugate_z.im)

def quadrant (x y : ℝ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On the axis"

theorem locate_quadrant : quadrant (point.1) (point.2) = "Fourth quadrant" := by
  sorry

end locate_quadrant_l650_650192


namespace car_R_average_speed_l650_650972

theorem car_R_average_speed :
  ∃ (v : ℕ), (600 / v) - 2 = 600 / (v + 10) ∧ v = 50 :=
by sorry

end car_R_average_speed_l650_650972


namespace sum_of_coeffs_g_eq_l650_650963

noncomputable theory 

def f : ℤ → ℤ := λ x, ∑ i in Finset.range 21, (-x)^i

def g (y : ℤ) : ℤ := f(y + 4)

theorem sum_of_coeffs_g_eq :
  ∑ i in Finset.range 21, (g i) = (1 + 5^21) / 6 := 
sorry

end sum_of_coeffs_g_eq_l650_650963


namespace largest_prime_factor_1729_l650_650537

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650537


namespace plane_sphere_intersection_plane_sphere_tangent_plane_not_intersect_sphere_l650_650703

variables {A B C D x₀ y₀ z₀ R : ℝ}

def plane_eq (x y z : ℝ) : Prop := A * x + B * y + C * z + D = 0

def distance_to_plane (A B C D x₀ y₀ z₀ : ℝ) : ℝ :=
  (abs (A * x₀ + B * y₀ + C * z₀ + D)) / (sqrt (A * A + B * B + C * C))

theorem plane_sphere_intersection :
  let d := distance_to_plane A B C D x₀ y₀ z₀ in
  (d < R) → (∃ x y z, plane_eq x y z ∧ (x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2 = R^2) :=
begin
  intros d H,
  sorry
end

theorem plane_sphere_tangent :
  let d := distance_to_plane A B C D x₀ y₀ z₀ in
  (d = R) → (∀ x y z, plane_eq x y z → (x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2 = R^2) :=
begin
  intros d H,
  sorry
end

theorem plane_not_intersect_sphere :
  let d := distance_to_plane A B C D x₀ y₀ z₀ in
  (d > R) → (∀ x y z, ¬ (plane_eq x y z ∧ (x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2 = R^2)) :=
begin
  intros d H,
  sorry
end

end plane_sphere_intersection_plane_sphere_tangent_plane_not_intersect_sphere_l650_650703


namespace expected_value_first_outstanding_sequence_l650_650141

-- Define indicator variables and the harmonic number
noncomputable def I (k : ℕ) := 1 / k
noncomputable def H (n : ℕ) := ∑ i in Finset.range (n + 1), I (i + 1)

-- Theorem: Expected value of the first sequence of outstanding grooms
theorem expected_value_first_outstanding_sequence : 
  (H 100) = 5.187 := 
sorry

end expected_value_first_outstanding_sequence_l650_650141


namespace total_interest_received_l650_650110

-- Conditions
def principal_B : ℝ := 5000
def time_B : ℕ := 2
def principal_C : ℝ := 3000
def time_C : ℕ := 4
def rate_of_interest : ℝ := 7.000000000000001

-- Simple Interest Formula
def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := (P * R * T) / 100

-- Interests from B and C
def interest_B := simple_interest principal_B rate_of_interest time_B
def interest_C := simple_interest principal_C rate_of_interest time_C

-- Total Interest
def total_interest : ℝ := interest_B + interest_C

-- Theorem to be proved
theorem total_interest_received :
  total_interest = 1540.0000000000015 :=
by
  sorry

end total_interest_received_l650_650110


namespace intersection_lies_on_line_l650_650854

open EuclideanGeometry

variables (A B C P : Point) 

-- Definitions for the given conditions
def is_angle_bisector (l1 l2 : Line) (A : Point) : Prop := sorry  -- Proper predicate definition is needed
def intersection (l1 l2 : Line) : Point := sorry  -- Proper predicate definition is needed
def line_containing (A B : Point) : Line := sorry  -- Proper predicate definition is needed

variable (K : Point)  -- Intersection of CP and AB
variable (M : Point)  -- Intersection of angle bisectors of BAC and ACP
variable (N : Point)  -- Intersection of angle bisector of PBA and the line of angle bisector of BPC

axiom inside_triangle : inside_triangle ABC P
axiom M_def : is_angle_bisector (line_containing A B) (line_containing A C) M
axiom N_def : intersect (line_containing P B) (line_containing P C) N

theorem intersection_lies_on_line :
    K = intersection (line_containing C P) (line_containing A B) → 
    K ∈ line_containing M N :=
sorry

end intersection_lies_on_line_l650_650854


namespace greatest_value_x_l650_650957

theorem greatest_value_x (x : ℕ) (h : lcm (lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_value_x_l650_650957


namespace smallest_bdf_value_l650_650899

open Nat

theorem smallest_bdf_value
  (a b c d e f : ℕ)
  (h1 : (a + 1) * c * e = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) :
  (b * d * f = 60) :=
sorry

end smallest_bdf_value_l650_650899


namespace divisors_of_m_squared_l650_650117

theorem divisors_of_m_squared {m : ℕ} (h₁ : ∀ d, d ∣ m → d = 1 ∨ d = m ∨ prime d) (h₂ : nat.divisors m = 4) :
  (nat.divisors (m ^ 2) = 7 ∨ nat.divisors (m ^ 2) = 9) :=
sorry

end divisors_of_m_squared_l650_650117


namespace power_of_7_mod_10_l650_650073

theorem power_of_7_mod_10 (k : ℕ) (h : 7^4 ≡ 1 [MOD 10]) : 7^150 ≡ 9 [MOD 10] :=
sorry

end power_of_7_mod_10_l650_650073


namespace largest_prime_factor_of_1729_l650_650561

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650561


namespace MKNL_is_rectangle_l650_650882

variables {A B C D M N K L : Type*} [TopologicalSpace A] [TopologicalSpace B] [TopologicalSpace C] [TopologicalSpace D]
  [TopologicalSpace M] [TopologicalSpace N] [TopologicalSpace K] [TopologicalSpace L]

-- Given points and their midpoints
variables (hM : is_midpoint M A B) (hN : is_midpoint N C D)
  (hK : is_midpoint K A C) (hL : is_midpoint L B D)

-- Given parallelism and equality of segments
variables (hML_kn_parallel : parallel ML KN) (hML_kn_equal : length ML = length KN)
  (hMK_ln_parallel : parallel MK LN) (hMK_ln_equal : length MK = length LN)

-- Definition of a rectangle in terms of parallel and equal opposite sides
def is_rectangle (M N K L : Type*) :=
  parallel ML KN ∧ length ML = length KN ∧
  parallel MK LN ∧ length MK = length LN

theorem MKNL_is_rectangle :
  is_rectangle M N K L :=
by {
  sorry
}

end MKNL_is_rectangle_l650_650882


namespace expected_value_first_outstanding_sequence_l650_650139

-- Define indicator variables and the harmonic number
noncomputable def I (k : ℕ) := 1 / k
noncomputable def H (n : ℕ) := ∑ i in Finset.range (n + 1), I (i + 1)

-- Theorem: Expected value of the first sequence of outstanding grooms
theorem expected_value_first_outstanding_sequence : 
  (H 100) = 5.187 := 
sorry

end expected_value_first_outstanding_sequence_l650_650139


namespace rectangle_area_stage_8_l650_650382

def square_side_length : ℕ := 4
def stage_count : ℕ := 8

-- The function to compute the area of one square
def square_area (side_length: ℕ) : ℕ :=
  side_length * side_length

-- The function to compute the total area at a given stage
def total_area_at_stage (side_length: ℕ) (stages: ℕ) : ℕ :=
  stages * (square_area side_length)

theorem rectangle_area_stage_8 :
  total_area_at_stage square_side_length stage_count = 128 :=
  by
    sorry

end rectangle_area_stage_8_l650_650382


namespace football_field_width_l650_650017

theorem football_field_width (length : ℕ) (total_distance : ℕ) (laps : ℕ) (width : ℕ) 
  (h1 : length = 100) (h2 : total_distance = 1800) (h3 : laps = 6) :
  width = 50 :=
by 
  -- Proof omitted
  sorry

end football_field_width_l650_650017


namespace ratio_city_XY_l650_650717

variable (popZ popY popX : ℕ)

-- Definition of the conditions
def condition1 := popY = 2 * popZ
def condition2 := popX = 16 * popZ

-- The goal to prove
theorem ratio_city_XY 
  (h1 : condition1 popY popZ)
  (h2 : condition2 popX popZ) :
  popX / popY = 8 := 
  by sorry

end ratio_city_XY_l650_650717


namespace calculate_expression_l650_650714

theorem calculate_expression :
  (1 / 2)^(-1) + (Real.pi + 2023)^0 - 2 * Real.cos (Real.pi / 3) + Real.sqrt 9 = 5 :=
by
  sorry

end calculate_expression_l650_650714


namespace ratio_of_areas_l650_650846

-- Defining the necessary points and triangle

def P := (0, 0)
def Q := (0, 8)
def R := (15, 0)

-- Midpoints S and T
def S := midpoint P Q
def T := midpoint Q R

-- Statements
theorem ratio_of_areas :
  let Y := intersection (line RS) (line PT) in
  area (quadrilateral P S Y T) = area (triangle Q Y R) := 
  sorry

end ratio_of_areas_l650_650846


namespace largest_prime_factor_1729_l650_650538

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650538


namespace DE_FG_sum_eq_one_l650_650409

variables {A B C D E F G : Type} 

/-- In an equilateral triangle ABC with AB = 2, 
points E and G lie on AC, and points D and F lie on AB, 
with both DE and FG parallel to BC. Given that DE = FG = x 
and DF = GE = 1 - x, and that the trapezoids DFGE and FBCG 
have the same perimeter, prove that DE + FG = 1. -/
theorem DE_FG_sum_eq_one (ABC : Type) [triangle ABC]
  (AB : ℝ) (AB_eq_two : AB = 2)
  {DE FG x : ℝ} (E G : ABC)
  (D F : ABC)
  (parallel_DE_BC : parallel DE BC)
  (parallel_FG_BC : parallel FG BC)
  (DE_eq_FG : DE = FG)
  (DF_eq_GE : DF = GE)
  (DF_eq_one_minus_x : DF = 1 - x)
  (perimeters_equal : 2 = 6 - x) :
  DE + FG = 1 :=
sorry

end DE_FG_sum_eq_one_l650_650409


namespace point_on_graph_l650_650026

theorem point_on_graph (x y : ℝ) :
    (x = -2 ∧ y = 2) →
    (∃ k, x = k ∧ y = 2^(k + 2) + 1) :=
by
  intro h
  cases h with hx hy
  use -2
  assumption

end point_on_graph_l650_650026


namespace ancient_tree_age_l650_650039

noncomputable def carbon14_age (k0 : ℝ) (k : ℝ) : ℝ :=
  (5730 : ℝ) * (Real.log (k / k0) / Real.log (1/2))

theorem ancient_tree_age :
  carbon14_age k0 (0.6 * k0) ≈ 4202.00 :=
by
  -- Proof steps go here. This is a placeholder.
  sorry

end ancient_tree_age_l650_650039


namespace ab_plus_a_plus_b_l650_650445

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x^2 - x + 2
-- Define the conditions on a and b
def is_root (x : ℝ) : Prop := poly x = 0

-- State the theorem
theorem ab_plus_a_plus_b (a b : ℝ) (ha : is_root a) (hb : is_root b) : a * b + a + b = 1 :=
sorry

end ab_plus_a_plus_b_l650_650445


namespace largest_prime_factor_of_1729_l650_650550

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650550


namespace problem_1_problem_2_l650_650715

-- Problem 1: Prove that sqrt(6) * sqrt(1/3) - sqrt(16) * sqrt(18) = -11 * sqrt(2)
theorem problem_1 : Real.sqrt 6 * Real.sqrt (1 / 3) - Real.sqrt 16 * Real.sqrt 18 = -11 * Real.sqrt 2 := 
by
  sorry

-- Problem 2: Prove that (2 - sqrt(5)) * (2 + sqrt(5)) + (2 - sqrt(2))^2 = 5 - 4 * sqrt(2)
theorem problem_2 : (2 - Real.sqrt 5) * (2 + Real.sqrt 5) + (2 - Real.sqrt 2) ^ 2 = 5 - 4 * Real.sqrt 2 := 
by
  sorry

end problem_1_problem_2_l650_650715


namespace regression_analysis_notes_l650_650997

-- Define the conditions
def applicable_population (reg_eq: Type) (sample: Type) : Prop := sorry
def temporality (reg_eq: Type) : Prop := sorry
def sample_value_range_influence (reg_eq: Type) (sample: Type) : Prop := sorry
def prediction_precision (reg_eq: Type) : Prop := sorry

-- Define the key points to note
def key_points_to_note (reg_eq: Type) (sample: Type) : Prop :=
  applicable_population reg_eq sample ∧
  temporality reg_eq ∧
  sample_value_range_influence reg_eq sample ∧
  prediction_precision reg_eq

-- The main statement
theorem regression_analysis_notes (reg_eq: Type) (sample: Type) :
  key_points_to_note reg_eq sample := sorry

end regression_analysis_notes_l650_650997


namespace minimum_integral_l650_650061

noncomputable def int_condition (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x + f y ≥ |x - y|

noncomputable def preferred_f : ℝ → ℝ := λ x, |x - 0.5|

theorem minimum_integral : (∫ x in 0..1, preferred_f x) = 1 / 4 :=
by
  have hf : int_condition preferred_f,
  sorry -- Proof that preferred_f satisfies the condition int_condition
  sorry -- Proof that the integral equals 1/4.

end minimum_integral_l650_650061


namespace larger_root_of_degree_11_l650_650998

theorem larger_root_of_degree_11 {x : ℝ} :
  (∃ x₁, x₁ > 0 ∧ (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9)) ∧
  (∃ x₂, x₂ > 0 ∧ (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11)) →
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧
    (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9) ∧
    (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11) ∧
    x₁ < x₂) :=
by
  sorry

end larger_root_of_degree_11_l650_650998


namespace team_X_games_l650_650487

/-- Let x be the number of games played by team X.
    Team X wins 3/4 of its games and team Y wins 2/3 of its games.
    Team Y has won 5 more games and lost 5 more games than team X.
    Prove that the number of games played by team X is 20. -/
theorem team_X_games (x y : ℕ) (hx1 : x * 3 / 4) (hy1 : y * 2 / 3)
  (won_diff : y * 2 / 3 - x * 3 / 4 = 5) 
  (lost_diff : y * 1 / 3 - x * 1 / 4 = 5) : 
  x = 20 := 
  sorry

end team_X_games_l650_650487


namespace no_500_good_trinomials_l650_650725

def is_good_quadratic_trinomial (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b^2 - 4 * a * c) > 0

theorem no_500_good_trinomials (S : Finset ℤ) (hS: S.card = 10)
  (hs_pos: ∀ x ∈ S, x > 0) : ¬(∃ T : Finset (ℤ × ℤ × ℤ), 
  T.card = 500 ∧ (∀ (a b c : ℤ), (a, b, c) ∈ T → is_good_quadratic_trinomial a b c)) :=
by
  sorry

end no_500_good_trinomials_l650_650725


namespace abc_order_l650_650164

noncomputable def a : ℝ := Real.log (3 / 2) - 3 / 2
noncomputable def b : ℝ := Real.log Real.pi - Real.pi
noncomputable def c : ℝ := Real.log 3 - 3

theorem abc_order : a > c ∧ c > b := by
  have h₁: a = Real.log (3 / 2) - 3 / 2 := rfl
  have h₂: b = Real.log Real.pi - Real.pi := rfl
  have h₃: c = Real.log 3 - 3 := rfl
  sorry

end abc_order_l650_650164


namespace largest_prime_factor_of_1729_l650_650624

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650624


namespace initial_people_in_line_l650_650981

theorem initial_people_in_line (X : ℕ) 
  (h1 : X - 6 + 3 = 18) : X = 21 :=
  sorry

end initial_people_in_line_l650_650981


namespace transportation_trucks_l650_650001

theorem transportation_trucks (boxes : ℕ) (total_weight : ℕ) (box_weight : ℕ) (truck_capacity : ℕ) :
  (total_weight = 10) → (∀ (b : ℕ), b ≤ boxes → box_weight ≤ 1) → (truck_capacity = 3) → 
  ∃ (trucks : ℕ), trucks = 5 :=
by
  sorry

end transportation_trucks_l650_650001


namespace find_AB_square_l650_650881

noncomputable theory

-- Defining the given lengths
def BC : ℝ := 10
def CA : ℝ := 15

-- Defining the concurrency condition statement
def cevians_concurrent (A B C : Type) [EuclideanGeometry A B C] : Prop :=
∃ G, (G ∈ line_through A (foot A B C)) ∧ (G ∈ line_through B (midpoint B A C)) ∧ (G ∈ line_through C (bisector C A B))

-- Problem statement
theorem find_AB_square (A B C : Type) [EuclideanGeometry A B C] 
    (h1 : cevians_concurrent A B C)
    (BC_len : distance B C = BC)
    (CA_len : distance C A = CA) : distance A B ^ 2 = 205 :=
sorry

end find_AB_square_l650_650881


namespace smallest_product_bdf_l650_650905

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end smallest_product_bdf_l650_650905


namespace real_solution_set_l650_650202

theorem real_solution_set : {x : ℝ | (x^2 - x - 6) / (x - 4) ≥ 3} = set.Ioo (-(∞ : ℝ)) 4 ∪ set.Ioo 4 (∞ : ℝ) := 
sorry

end real_solution_set_l650_650202


namespace no_solutions_for_prime_greater_than_5_l650_650477

theorem no_solutions_for_prime_greater_than_5 (p : ℕ) (x : ℤ) (hp : Nat.Prime p) (h5 : p > 5) : x^4 + 4^x ≠ p :=
sorry

end no_solutions_for_prime_greater_than_5_l650_650477


namespace smallest_positive_period_f_monotonically_decreasing_interval_l650_650325

-- Given vectors a and b
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

-- Definition of f(x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Statement for the smallest positive period
theorem smallest_positive_period : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

-- Statement for the interval where f(x) is monotonically decreasing in [0, π/2]
theorem f_monotonically_decreasing_interval (x : ℝ) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  f' x < 0 ↔ x ∈ Set.Icc (Real.pi * 3 / 8) (Real.pi / 2) :=
sorry

end smallest_positive_period_f_monotonically_decreasing_interval_l650_650325


namespace polynomial_perfect_square_l650_650400

theorem polynomial_perfect_square (k : ℤ) : (∃ b : ℤ, (x + b)^2 = x^2 + 8 * x + k) -> k = 16 := by
  sorry

end polynomial_perfect_square_l650_650400


namespace largest_prime_factor_1729_l650_650545

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650545


namespace li_fang_outfits_l650_650462

theorem li_fang_outfits (shirts skirts dresses outfits: ℕ) (h_shirts: shirts = 4) (h_skirts: skirts = 3) (h_dresses: dresses = 2) :
  outfits = shirts * skirts + dresses → outfits = 14 := 
by
  intros h1,
  rw [h_shirts, h_skirts, h_dresses] at h1,
  norm_num at h1,
  exact h1,

end li_fang_outfits_l650_650462


namespace inequality_holds_l650_650024

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_holds (h_cont : Continuous f) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, 2 * f x - (deriv f x) > 0) : 
  f 1 > (f 2) / (Real.exp 2) :=
sorry

end inequality_holds_l650_650024


namespace find_S21_l650_650504

noncomputable def a : ℕ → ℚ
| 0     := 0             -- this is simply extended to ℕ including 0 for indexing convenience
| 1     := 0             -- irrelevant initial value
| (n+2) := 2             -- given condition a₂ = 2
| (n+3) := -3/2          -- arbitrary placeholder to enable the recursion pattern

-- Helper for the relation a_n + a_n+1 = 1/2
def sequence_relation (a : ℕ → ℚ) : Prop :=
  ∀ n, n ∈ {n : ℕ | 1 < n} → a n + a (n + 1) = 1/2

def element_of_sequence (a : ℕ → ℚ) : Prop :=
  a 2 = 2

noncomputable def sum_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = ∑ i in Finset.range n, a (i + 1)

theorem find_S21 :
  sequence_relation a →
  element_of_sequence a →
  sum_sequence a S →
  S 21 = 7/2 :=
by
  intros _ _ _
  sorry

end find_S21_l650_650504


namespace find_n_l650_650788

theorem find_n (x : ℝ) (n : ℝ)
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -2)
  (h2 : Real.log (Real.sin x + Real.cos x) = 1 / 2 * (Real.log n - 2)) :
  n = Real.exp 2 + 2 :=
by
  sorry

end find_n_l650_650788


namespace angle_CMD_24_degrees_l650_650947

-- Definitions based on problem conditions
def is_regular_pentagon (ABCDE : Point → Prop) : Prop :=
  ∀ A B C D E, ∃ r, ∃ θ, 
  A = ⟨r * cos(θ), r * sin(θ)⟩ ∧
  B = ⟨r * cos(θ + 2 * π / 5), r * sin(θ + 2 * π / 5)⟩ ∧
  C = ⟨r * cos(θ + 4 * π / 5), r * sin(θ + 4 * π / 5)⟩ ∧
  D = ⟨r * cos(θ + 6 * π / 5), r * sin(θ + 6 * π / 5)⟩ ∧
  E = ⟨r * cos(θ + 8 * π / 5), r * sin(θ + 8 * π / 5)⟩

def is_equilateral_triangle (MNP : Point → Prop) : Prop :=
  ∀ M N P, ∃ r, ∃ θ, 
  M = ⟨r * cos(θ), r * sin(θ)⟩ ∧
  N = ⟨r * cos(θ + 2 * π / 3), r * sin(θ + 2 * π / 3)⟩ ∧
  P = ⟨r * cos(θ + 4 * π / 3), r * sin(θ + 4 * π / 3)⟩

-- Main statement we want to prove
theorem angle_CMD_24_degrees (ABCDE MNP : Point → Prop) (A B C D E M N P : Point) 
  (h1 : is_regular_pentagon ABCDE) 
  (h2 : is_equilateral_triangle MNP) : 
  (angle C M D = 24) :=
by
  sorry

end angle_CMD_24_degrees_l650_650947


namespace original_number_of_men_l650_650662

theorem original_number_of_men (W : ℝ) (M : ℝ) (total_work : ℝ) :
  (M * W * 11 = (M + 10) * W * 8) → M = 27 :=
by
  sorry

end original_number_of_men_l650_650662


namespace hendecagon_diagonals_l650_650203

-- Define the number of sides n of the hendecagon
def n : ℕ := 11

-- Define the formula for calculating the number of diagonals in an n-sided polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that there are 44 diagonals in a hendecagon
theorem hendecagon_diagonals : diagonals n = 44 :=
by
  -- Proof is skipped using sorry
  sorry

end hendecagon_diagonals_l650_650203


namespace price_increase_X_is_30_cents_l650_650501

-- Define the conditions of the problem
def price_X_in_2001 : ℝ := 4.20
def price_Y_in_2001 : ℝ := 4.40
def price_increase_X (x : ℝ) : ℕ → ℝ := λ n, price_X_in_2001 + n * x
def price_increase_Y : ℕ → ℝ := λ n, price_Y_in_2001 + n * 0.20

-- Given the data in 2007 (6 years later)
def year_difference : ℕ := 6
def extra_cost_of_X_in_2007 : ℝ := 0.40

-- Theorem statement
theorem price_increase_X_is_30_cents :
  (price_increase_X (30 / 100) 6) = (price_increase_Y 6) + extra_cost_of_X_in_2007 :=
by sorry

end price_increase_X_is_30_cents_l650_650501


namespace find_geometric_sequence_term_l650_650850

noncomputable def geometric_sequence_term (a q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

theorem find_geometric_sequence_term (a : ℝ) (q : ℝ)
  (h1 : a * (1 - q ^ 3) / (1 - q) = 7)
  (h2 : a * (1 - q ^ 6) / (1 - q) = 63) :
  ∀ n : ℕ, geometric_sequence_term a q n = 2^(n-1) :=
by
  sorry

end find_geometric_sequence_term_l650_650850


namespace smallest_bdf_value_l650_650897

open Nat

theorem smallest_bdf_value
  (a b c d e f : ℕ)
  (h1 : (a + 1) * c * e = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) :
  (b * d * f = 60) :=
sorry

end smallest_bdf_value_l650_650897


namespace largest_prime_factor_1729_l650_650532

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650532


namespace percentage_of_loss_is_25_l650_650686

-- Definitions from conditions
def CP : ℝ := 2800
def SP : ℝ := 2100

-- Proof statement
theorem percentage_of_loss_is_25 : ((CP - SP) / CP) * 100 = 25 := by
  sorry

end percentage_of_loss_is_25_l650_650686


namespace solve_for_y_l650_650666

theorem solve_for_y (y : ℝ) (h : 9 ^ y = 3 ^ 12) : y = 6 :=
sorry

end solve_for_y_l650_650666


namespace value_of_a_plus_d_l650_650665

variable {R : Type} [LinearOrderedField R]
variables {a b c d : R}

theorem value_of_a_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 13 := by
  sorry

end value_of_a_plus_d_l650_650665


namespace max_levels_passable_prob_pass_three_levels_l650_650674

-- Define the condition for passing a level
def passes_level (n : ℕ) (sum : ℕ) : Prop :=
  sum > 2^n

-- Define the maximum sum possible for n dice rolls
def max_sum (n : ℕ) : ℕ :=
  6 * n

-- Define the probability of passing the n-th level
def prob_passing_level (n : ℕ) : ℚ :=
  if n = 1 then 2/3
  else if n = 2 then 5/6
  else if n = 3 then 20/27
  else 0 

-- Combine probabilities for passing the first three levels
def prob_passing_three_levels : ℚ :=
  (2/3) * (5/6) * (20/27)

-- Theorem statement for the maximum number of levels passable
theorem max_levels_passable : 4 = 4 :=
sorry

-- Theorem statement for the probability of passing the first three levels
theorem prob_pass_three_levels : prob_passing_three_levels = 100 / 243 :=
sorry

end max_levels_passable_prob_pass_three_levels_l650_650674


namespace equilateral_triangle_l650_650471

namespace TriangleEquilateral

-- Define the structure of a triangle and given conditions
structure Triangle :=
  (A B C : ℝ)  -- vertices
  (angleA : ℝ) -- angle at vertex A
  (sideBC : ℝ) -- length of side BC
  (perimeter : ℝ)  -- perimeter of the triangle

-- Define the proof problem
theorem equilateral_triangle (T : Triangle) (h1 : T.angleA = 60)
  (h2 : T.sideBC = T.perimeter / 3) : 
  T.A = T.B ∧ T.B = T.C ∧ T.A = T.C ∧ T.A = T.B ∧ T.B = T.C ∧ T.A = T.C :=
  sorry

end TriangleEquilateral

end equilateral_triangle_l650_650471


namespace largest_prime_factor_1729_l650_650575

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650575


namespace sum_of_solutions_in_range_l650_650443

theorem sum_of_solutions_in_range :
  let S := ∑ x in {x : ℝ | x > 0 ∧ x ^ (2 ^ (Real.sqrt 2)) = (Real.sqrt 2) ^ (2 ^ x)}, x
  ∃ S : ℝ, 2 ≤ S ∧ S < 6 :=
by
  sorry

end sum_of_solutions_in_range_l650_650443


namespace largest_prime_factor_1729_l650_650577

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l650_650577


namespace periodic_function_solution_l650_650805

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def f₁ (x : ℝ) : ℝ := (deriv f) x 

noncomputable def f₂ (x : ℝ) : ℝ := (deriv f₁) x

noncomputable def f₃ (x : ℝ) : ℝ := (deriv f₂) x

noncomputable def f₄ (x : ℝ) : ℝ := (deriv f₃) x

def is_periodic (f : ℝ → ℝ) (n : ℕ) : Prop := ∀ x, f (n * 4 + x) = f x

theorem periodic_function (x : ℝ) : 
  is_periodic f 2016 ∧ (f 2016 x = Real.sin x + Real.cos x) := by
  sorry

theorem solution (x := Real.pi / 3) : f₄ x = (Real.sqrt 3 + 1) / 2 := by 
  rw [is_periodic, f, Real.sin_pi_div_three, Real.cos_pi_div_three]
  norm_num
  sorry

end periodic_function_solution_l650_650805


namespace BXYM_cyclic_l650_650474

open EuclideanGeometry

theorem BXYM_cyclic {A B C M P Q X Y : Point} (h1: IsMidpoint M A C) 
(h2: P ∈ LineSegment A M) (h3: Q ∈ LineSegment C M) 
(h4: distance P Q = distance A C / 2) 
(h5: Circle (circumcircle A B Q) B ≠ X ∧ X ∈ LineSegment B C ∧ X ≠ B)
(h6: Circle (circumcircle B C P) B ≠ Y ∧ Y ∈ LineSegment A B ∧ Y ≠ B) :
CyclicQuadrilateral B X Y M :=
by sorry

end BXYM_cyclic_l650_650474


namespace find_number_l650_650406

theorem find_number (x : ℝ) (h : (5 / 6) * x = (5 / 16) * x + 300) : x = 576 :=
sorry

end find_number_l650_650406


namespace oranges_weight_is_10_l650_650111

def applesWeight (A : ℕ) : ℕ := A
def orangesWeight (A : ℕ) : ℕ := 5 * A
def totalWeight (A : ℕ) (O : ℕ) : ℕ := A + O
def totalCost (A : ℕ) (x : ℕ) (O : ℕ) (y : ℕ) : ℕ := A * x + O * y

theorem oranges_weight_is_10 (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := by
  sorry

end oranges_weight_is_10_l650_650111


namespace find_y_l650_650651

theorem find_y (y : ℝ) (h : 2 * y / 3 = 12) : y = 18 :=
by
  sorry

end find_y_l650_650651


namespace largest_prime_factor_of_1729_l650_650632

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650632


namespace leon_required_score_l650_650858

noncomputable def leon_scores : List ℕ := [72, 68, 75, 81, 79]

theorem leon_required_score (n : ℕ) :
  (List.sum leon_scores + n) / (List.length leon_scores + 1) ≥ 80 ↔ n ≥ 105 :=
by sorry

end leon_required_score_l650_650858


namespace intersection_of_M_and_N_l650_650883

open Set

def M : Set ℤ := {-1, 0, 1}

def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : (M ∩ N : Set ℝ) = {0, 1} :=
by
  sorry

end intersection_of_M_and_N_l650_650883


namespace part_I_part_II_l650_650324

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem part_I :
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6) + 1) →
  ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = Real.pi :=
sorry

theorem part_II :
  ∀ x ∈ Icc (-Real.pi/6) (Real.pi/3),
    1 ≤ f x ∧ f x ≤ 3 :=
sorry

end part_I_part_II_l650_650324


namespace perp_sum_constant_for_base_points_l650_650059

variable (n : ℕ) (Base : Set ℝ³) (M : ℝ³ → Prop) (N : ℕ → ℝ³ → ℝ) (α : ℝ)

def regular_pyramid (Base : Set ℝ³) :=
  ∃ Apex : ℝ³, ∀ x ∈ Base, (Apex - x).norm = (Apex - M x).norm

def perpendiculars_sum_constant (Base : Set ℝ³) (M : ℝ³) (N : ℕ → ℝ³ → ℝ) : Prop :=
  ∀ M₁ M₂ ∈ Base, (N 1 M₁ + N 2 M₁ + ... + N n M₁ = N 1 M₂ + N 2 M₂ + ... + N n M₂)

theorem perp_sum_constant_for_base_points :
  regular_pyramid Base →
  perpendiculars_sum_constant Base M N :=
sorry

end perp_sum_constant_for_base_points_l650_650059


namespace rhombus_area_l650_650840

theorem rhombus_area :
  let P1 := (0, 3.5)
  let P2 := (11, 0)
  let P3 := (0, -3.5)
  let P4 := (-11, 0)
  let d1 := dist P1 P3
  let d2 := dist P2 P4
  d1 = 7 ∧ d2 = 22 → 
  (d1 * d2) / 2 = 77 :=
by
  let P1 := (0, 3.5)
  let P2 := (11, 0)
  let P3 := (0, -3.5)
  let P4 := (-11, 0)
  let d1 := dist P1 P3
  let d2 := dist P2 P4
  rw [dist_eq, dist_eq]
  sorry

end rhombus_area_l650_650840


namespace max_candies_takeable_l650_650845

theorem max_candies_takeable : 
  ∃ (max_take : ℕ), max_take = 159 ∧
  ∀ (boxes: Fin 5 → ℕ), 
    boxes 0 = 11 → 
    boxes 1 = 22 → 
    boxes 2 = 33 → 
    boxes 3 = 44 → 
    boxes 4 = 55 →
    (∀ (i : Fin 5), 
      ∀ (new_boxes : Fin 5 → ℕ),
      (new_boxes i = boxes i - 4) ∧ 
      (∀ (j : Fin 5), j ≠ i → new_boxes j = boxes j + 1) →
      boxes i = 0 → max_take = new_boxes i) :=
sorry

end max_candies_takeable_l650_650845


namespace friends_professions_l650_650231

theorem friends_professions
  (architect barista veterinarian guitarist : ℕ)
  (P : ℕ → Sort)
  (Andrey Boris Vyacheslav Gennady : P)
  (seats : P → ℕ)
  (h1 : ∀ {x y}, seats x = veterinarian → seats y = architect → (seats x = seats y + 1 ∨ seats x = seats y - 1))
  (h2 : ∀ {x y}, seats x = barista → seats y = Boris → seats y = seats x + 1)
  (h3 : seats Vyacheslav > seats Andrey ∧ seats Vyacheslav > seats Boris)
  (h4 : ∀ {x}, x = Andrey → ∃ y z, (seats y = seats x + 1 ∨ seats y = seats x - 1) ∧ (seats z = seats x + 1 ∨ seats z = seats x - 1))
  (h5 : ∀ {x y}, seats x = guitarist → seats y = barista → seats x ≠ seats y + 1 ∧ seats x ≠ seats y - 1)
  : (∀ x, (x = Gennady → seats x = barista)
    ∧ (x = Boris → seats x = architect)
    ∧ (x = Andrey → seats x = veterinarian)
    ∧ (x = Vyacheslav → seats x = guitarist)) :=
by
  sorry

end friends_professions_l650_650231


namespace fraction_of_women_married_l650_650080

theorem fraction_of_women_married (total : ℕ) (women men married: ℕ) (h1 : total = women + men)
(h2 : women = 76 * total / 100) (h3 : married = 60 * total / 100) (h4 : 2 * (men - married) = 3 * men):
 (married - (total - women - married) * 1 / 3) = 13 * women / 19 :=
sorry

end fraction_of_women_married_l650_650080


namespace correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l650_650999

theorem correct_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

-- Incorrect options for clarity
theorem incorrect_division (a : ℝ) : a^6 / a^2 ≠ a^3 :=
by sorry

theorem incorrect_multiplication (a : ℝ) : a^2 * a^3 ≠ a^6 :=
by sorry

theorem incorrect_addition (a : ℝ) : (a^2 + a^3) ≠ a^5 :=
by sorry

end correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l650_650999


namespace modulus_of_squared_complex_l650_650740

def complex_square (z : Complex) : Complex :=
  z * z

def modulus (z : Complex) : Real :=
  Complex.abs z

theorem modulus_of_squared_complex :
  let z := Complex.mk (-3) (5 / 4)
  modulus (complex_square z) = 169 / 16 :=
by
  sorry

end modulus_of_squared_complex_l650_650740


namespace mass_of_CCl4_produced_l650_650728

def molar_mass (element : String) : Float :=
  match element with
  | "C"  => 12.01
  | "Cl" => 35.45
  | _    => 0

def mass_of_ccl4 (moles : Float) : Float :=
  let mass_c = molar_mass "C"
  let mass_cl = molar_mass "Cl"
  let molar_mass_ccl4 = mass_c + 4 * mass_cl
  moles * molar_mass_ccl4

theorem mass_of_CCl4_produced :
  mass_of_ccl4 8 = 1230.48 :=
by
  sorry

end mass_of_CCl4_produced_l650_650728


namespace chickens_on_farm_l650_650682

theorem chickens_on_farm (C : ℕ)
  (h1 : 0.20 * C = bcm)
  (h2 : 0.80 * bcm = bcm_hens)
  (h3 : bcm_hens = 16) :
  C = 100 :=
by
  sorry

end chickens_on_farm_l650_650682


namespace find_a_13_l650_650789

variable {a_n : ℕ → ℝ}
variable (d : ℝ)

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n m, a_n (n + 1) = a_n n + d

def forms_geometric_sequence (a_1 a_5 a_9 : ℝ) : Prop :=
  a_1^2 = a_9 * a_5

def sum_condition (a_1 a_5 a_9 : ℝ) : Prop :=
  a_1 + 3*a_5 + a_9 = 20

theorem find_a_13 (h1 : is_arithmetic_sequence a_n d) (h2 : d ≠ 0)
  (h3 : forms_geometric_sequence (a_n 1) (a_n 5) (a_n 9))
  (h4 : sum_condition (a_n 1) (a_n 5) (a_n 9)) : 
  a_n 13 = 28 :=
sorry

end find_a_13_l650_650789


namespace maximum_k_value_l650_650797

theorem maximum_k_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : abs ((3 * m + sqrt 6 * n) / sqrt ((m+1)^2 + (n+1/2)^2)) = sqrt 5) (h4 : 2 * m + n ≥ k) : k = 3 :=
sorry

end maximum_k_value_l650_650797


namespace largest_prime_factor_of_1729_l650_650627

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650627


namespace diagonals_sum_pentagon_inscribed_in_circle_l650_650439

theorem diagonals_sum_pentagon_inscribed_in_circle
  (FG HI GH IJ FJ : ℝ)
  (h1 : FG = 4)
  (h2 : HI = 4)
  (h3 : GH = 11)
  (h4 : IJ = 11)
  (h5 : FJ = 15) :
  3 * FJ + (FJ^2 - 121) / 4 + (FJ^2 - 16) / 11 = 80 := by {
  sorry
}

end diagonals_sum_pentagon_inscribed_in_circle_l650_650439


namespace largest_prime_factor_1729_l650_650540

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650540


namespace num_bases_for_625_ending_in_1_l650_650765

theorem num_bases_for_625_ending_in_1 :
  (Finset.card (Finset.filter (λ b : ℕ, 624 % b = 0) (Finset.Icc 3 10))) = 4 :=
by
  sorry

end num_bases_for_625_ending_in_1_l650_650765


namespace karlson_can_repair_propeller_l650_650060

theorem karlson_can_repair_propeller :
  ∃ initial_blades initial_screws subsequent_blades subsequent_expense:
    ℕ, ℕ, ℕ, ℕ,
  initial_blades = 2 ∧ initial_screws = 2 ∧ subsequent_blades = 1 ∧
  let cost_blade := 120 in
  let cost_screw := 9 in
  let discount := 0.2 in
  let initial_purchase := (initial_blades * cost_blade) + (initial_screws * cost_screw) in
  let subsequent_purchase := (subsequent_blades * cost_blade * (1 - discount)) in
  let total_expense := initial_purchase + subsequent_purchase in
  initial_purchase >= 250 ∧
  subsequent_expense = cost_blade * (1 - discount) ∧
  total_expense ≤ 360 := 
by
  have initial_blades := 2 
  have initial_screws := 2 
  have subsequent_blades := 1 
  have cost_blade := 120 
  have cost_screw := 9 
  have discount := 0.2 
  let initial_purchase := (initial_blades * cost_blade) + (initial_screws * cost_screw)
  let subsequent_purchase := subsequent_blades * cost_blade * (1 - discount)
  let total_expense := initial_purchase + subsequent_purchase 
  have h1 : initial_purchase = 258 := sorry -- calculated as 258
  have h2 : subsequent_purchase = 96 := sorry -- calculated as 96
  have h3 : total_expense = 354 := sorry -- 354 is less than or equal to 360
  exact ⟨initial_blades, initial_screws, subsequent_blades, subsequent_expense, rfl, rfl, rfl,rfl,by 
    dsimp [initial_purchase, subsequent_purchase, total_expense] 
    exact h3⟩ -- Write the proof separately. I'm focusing on the statement.

end karlson_can_repair_propeller_l650_650060


namespace largest_prime_factor_1729_l650_650524

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650524


namespace ratio_of_sums_eq_neg_sqrt_2_l650_650872

open Real

theorem ratio_of_sums_eq_neg_sqrt_2
    (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 6) :
    (x + y) / (x - y) = -Real.sqrt 2 :=
by sorry

end ratio_of_sums_eq_neg_sqrt_2_l650_650872


namespace zeros_of_f_l650_650973

def f (x : ℝ) : ℝ := x^2 - x - 2

theorem zeros_of_f :
  {x : ℝ | f x = 0} = ({-1, 2} : set ℝ) :=
sorry

end zeros_of_f_l650_650973


namespace area_of_rectangle_stage_8_l650_650388

theorem area_of_rectangle_stage_8 : 
  (∀ n, 4 * 4 = 16) →
  (∀ k, k ≤ 8 → k = k) →
  (8 * 16 = 128) :=
by
  intros h_sq_area h_sequence
  sorry

end area_of_rectangle_stage_8_l650_650388


namespace part1_monotonic_when_a_is_0_part2_extremum_point_on_interval_l650_650804

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x * Real.log x - x + a) / (x ^ 2)

theorem part1_monotonic_when_a_is_0 :
  (∀ x, 0 < x ∧ x < Real.exp 2 → DifferentiableAt ℝ (λ x, f x 0) x ∧ Deriv (λ x, f x 0).MonotoneOn (Set.Ioo 0 (Real.exp 2)))
  ∧ (∀ x, x > Real.exp 2 → DifferentiableAt ℝ (λ x, f x 0) x ∧ Deriv (λ x, f x 0)).AntitoneOn (Set.Ioi (Real.exp 2))
:=
by
  sorry

theorem part2_extremum_point_on_interval :
  ∀ a, f (1 : ℝ) a =?= f (Real.exp (2 : ℝ)) a → (0 < a ∧ a < Real.exp 1 / 2)
:=
by
  sorry

end part1_monotonic_when_a_is_0_part2_extremum_point_on_interval_l650_650804


namespace goods_train_passes_in_9_seconds_l650_650687

-- Definitions based on conditions
def speedTrain_A_kmph : ℝ := 60
def speedTrain_B_kmph : ℝ := 52
def lengthTrain_B_m : ℝ := 280
def kmph_to_mps (v : ℝ) : ℝ := v * (5 / 18)

-- Combined conditions into relative speed and time calculation
def relative_speed_mps : ℝ := kmph_to_mps (speedTrain_A_kmph + speedTrain_B_kmph)
def time_to_pass (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Main statement proving that the goods train takes 9 seconds to pass the man
theorem goods_train_passes_in_9_seconds :
  time_to_pass lengthTrain_B_m relative_speed_mps = 9 :=
sorry

end goods_train_passes_in_9_seconds_l650_650687


namespace num_factors_180_l650_650362

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l650_650362


namespace phil_won_more_games_than_charlie_l650_650910

theorem phil_won_more_games_than_charlie :
  ∀ (P D C Ph : ℕ),
  (P = D + 5) → (C = D - 2) → (Ph = 12) → (P = Ph + 4) →
  Ph - C = 3 :=
by
  intros P D C Ph hP hC hPh hPPh
  sorry

end phil_won_more_games_than_charlie_l650_650910


namespace unique_number_satisfying_conditions_l650_650747

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

-- Product of digits function
def product_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).prod

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem unique_number_satisfying_conditions (x : ℕ) :
  (product_of_digits x = 44 * x - 86868) →
  (is_perfect_square (sum_of_digits x)) →
  x = 1989 :=
by
  sorry

end unique_number_satisfying_conditions_l650_650747


namespace average_of_first_5_subjects_l650_650708

theorem average_of_first_5_subjects (avg_6_subjects : ℝ) (marks_6th_subject : ℝ) (total_subjects : ℕ) (first_5_subjects : ℕ) :
  avg_6_subjects = 80 →
  marks_6th_subject = 110 →
  total_subjects = 6 →
  first_5_subjects = 5 →
  (total_subjects * avg_6_subjects - marks_6th_subject) / first_5_subjects = 74 :=
by
  intros avg_6_trivial marks_6_trivial total_subjects_trivial first_5_trivial
  rw [avg_6_trivial, marks_6_trivial, total_subjects_trivial, first_5_trivial]
  have : 6 * 80 - 110 = 370 := by norm_num
  rw [this]
  norm_num
  sorry

end average_of_first_5_subjects_l650_650708


namespace num_factors_180_l650_650364

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l650_650364


namespace problem_statement_l650_650316

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x b : ℝ) : ℝ := -x^2 + 2*x + b
noncomputable def h (x : ℝ) : ℝ := f x - (1 / f x)

theorem problem_statement :
  ∀ b : ℝ, (∀ x : ℝ, h (x) = f x - 1 / f x ∧ h (-x) = -h (x) ∧
  (∀ y : ℝ, ∀ z : ℝ, y < z → h (y) < h (z))) ∧
  (∀ x ∈ Icc 1 2, ∃ (x1 x2 : ℝ), x1 ∈ Icc 1 2 ∧ x2 ∈ Icc 1 2 ∧ f x ≤ f x1 ∧ g x x2 = f x1 → b = Real.exp 2 - 1) :=
by 
  sorry

end problem_statement_l650_650316


namespace quadratic_solution_l650_650932

theorem quadratic_solution : 
  ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ (x = 1 ∨ x = 3 / 2) :=
by
  intro x
  constructor
  sorry

end quadratic_solution_l650_650932


namespace cosine_sum_inequality_in_triangle_l650_650425

theorem cosine_sum_inequality_in_triangle {α β γ : ℝ}
  (hα : α + β + γ = π):
  cos α + cos β + cos γ ≤ 3 / 2 :=
sorry

end cosine_sum_inequality_in_triangle_l650_650425


namespace factors_of_180_l650_650367

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l650_650367


namespace sum_of_remainders_l650_650652

theorem sum_of_remainders (a b c : ℕ) 
  (ha : a % 15 = 12) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 9 := 
by 
  sorry

end sum_of_remainders_l650_650652


namespace calculate_amount_l650_650173

theorem calculate_amount (p1 p2 p3: ℝ) : 
  p1 = 0.15 * 4000 ∧ 
  p2 = p1 - 0.25 * p1 ∧ 
  p3 = 0.07 * p2 -> 
  (p3 + 0.10 * p3) = 34.65 := 
by 
  sorry

end calculate_amount_l650_650173


namespace find_x_l650_650813

theorem find_x
  (x : ℝ)
  (a : ℝ × ℝ := (1, 2))
  (b : ℝ × ℝ := (x, 1))
  (h : (a.1 + 2 * b.1, a.2 + 2 * b.2) ∥ (2 * a.1 - b.1, 2 * a.2 - b.2)) :
  x = 1 / 2 :=
by sorry

end find_x_l650_650813


namespace seating_arrangement_l650_650239

-- Definitions of friends and their professions
inductive Friend
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq

-- Seating condition definitions
def seated_right (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = a ∧ seating (i + 1) = b)

def seated_between (a b c : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = b ∧ seating (i - 1) = a ∧ seating (i + 1) = c)

def neighbors (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, (seating i = a ∧ seating (i - 1) = b) ∨ (seating i = a ∧ seating (i + 1) = b))

-- The proposition to be proved
theorem seating_arrangement
  (seating : ℕ → Friend)
  (profession : Friend → Profession)
  (h1 : ∃ i, seating_between Profession.Veterinarian Profession.Architect Profession.Guitarist seating)
  (h2 : seated_right Profession.Barista Profession.Boris seating)
  (h3 : ∃ i, seating i = Friend.Vyacheslav ∧ (∃ j, seating j = Friend.Andrey ∧ j < i) ∧ (∃ k, seating k = Friend.Boris ∧ k < i))
  (h4 : ∀ a, seating a = Friend.Andrey → neighbors Friend.Andrey seating)
  (h5 : ∀ a, seating a = Profession.Guitarist → ∀ b, seating (b - 1) = Profession.Barista ∨ seating (b + 1) = Profession.Barista → false) :
  (profession Friend.Gennady = Profession.Barista ∧
   profession Friend.Boris = Profession.Architect ∧
   profession Friend.Andrey = Profession.Veterinarian ∧
   profession Friend.Vyacheslav = Profession.Guitarist) :=
begin
  sorry
end

end seating_arrangement_l650_650239


namespace reflection_line_is_x_eq_0_l650_650518

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  X : Point
  Y : Point
  Z : Point

def reflect_point (p : Point) (m : ℝ) : Point :=
  ⟨2*m - p.x, p.y⟩

theorem reflection_line_is_x_eq_0 :
  ∀ (X Y Z : Point),
  X.x = 1 → X.y = 4 →
  Y.x = 6 → Y.y = 5 →
  Z.x = -3 → Z.y = 2 →
  let M := 0 in
  reflect_point ⟨X.x, X.y⟩ M = ⟨-1, 4⟩ →
  reflect_point ⟨Y.x, Y.y⟩ M = ⟨-6, 5⟩ →
  reflect_point ⟨Z.x, Z.y⟩ M = ⟨3, 2⟩ →
  M = 0 :=
  by
    intros X Y Z
    intros hXx hXy hYx hYy hZx hZy
    intros M hX' hY' hZ'
    sorry

end reflection_line_is_x_eq_0_l650_650518


namespace final_professions_correct_l650_650265

-- Define the people and their positions
inductive Person
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq, Inhabited

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq, Inhabited

open Person Profession

-- Given positions in seats
def seatingArrangement : Person → Nat := 
λ p, 
  match p with
  | Andrey    => 3
  | Boris     => 2
  | Vyacheslav => 4
  | Gennady   => 1

-- Given professions
def profession : Person → Profession := 
λ p, 
  match p with
  | Andrey    => Veterinarian
  | Boris     => Architect
  | Vyacheslav => Guitarist
  | Gennady   => Barista

-- Define the conditions
def condition1 : Prop :=
  (seatingArrangement Andrey = 3) ∧ 
  (seatingArrangement Boris = 2) ∧
  (seatingArrangement Vyacheslav = 4) ∧ 
  (seatingArrangement Gennady = 1)

def condition2 : Prop := 
  seatingArrangement (Person.Barista) + 1 = seatingArrangement Boris

def condition3 : Prop := 
  seatingArrangement Vyacheslav > seatingArrangement Andrey ∧ 
  seatingArrangement Vyacheslav > seatingArrangement Boris

def condition4 : Prop := 
  seatingArrangement (Person.Veterinarian) ≠ 1 ∧ seatingArrangement (Person.Veterinarian) ≠ 4

def condition5 : Prop := 
  (seatingArrangement (Person.Barista) ≠ seatingArrangement (Person.Guitarist) + 1 ∧
  seatingArrangement (Person.Guitarist) ≠ seatingArrangement (Person.Barista) + 1)

-- Prove the final professions
theorem final_professions_correct :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  profession Gennady = Barista ∧
  profession Boris = Architect ∧
  profession Andrey = Veterinarian ∧
  profession Vyacheslav = Guitarist :=
by sorry

end final_professions_correct_l650_650265


namespace final_professions_correct_l650_650270

-- Define the people and their positions
inductive Person
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq, Inhabited

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq, Inhabited

open Person Profession

-- Given positions in seats
def seatingArrangement : Person → Nat := 
λ p, 
  match p with
  | Andrey    => 3
  | Boris     => 2
  | Vyacheslav => 4
  | Gennady   => 1

-- Given professions
def profession : Person → Profession := 
λ p, 
  match p with
  | Andrey    => Veterinarian
  | Boris     => Architect
  | Vyacheslav => Guitarist
  | Gennady   => Barista

-- Define the conditions
def condition1 : Prop :=
  (seatingArrangement Andrey = 3) ∧ 
  (seatingArrangement Boris = 2) ∧
  (seatingArrangement Vyacheslav = 4) ∧ 
  (seatingArrangement Gennady = 1)

def condition2 : Prop := 
  seatingArrangement (Person.Barista) + 1 = seatingArrangement Boris

def condition3 : Prop := 
  seatingArrangement Vyacheslav > seatingArrangement Andrey ∧ 
  seatingArrangement Vyacheslav > seatingArrangement Boris

def condition4 : Prop := 
  seatingArrangement (Person.Veterinarian) ≠ 1 ∧ seatingArrangement (Person.Veterinarian) ≠ 4

def condition5 : Prop := 
  (seatingArrangement (Person.Barista) ≠ seatingArrangement (Person.Guitarist) + 1 ∧
  seatingArrangement (Person.Guitarist) ≠ seatingArrangement (Person.Barista) + 1)

-- Prove the final professions
theorem final_professions_correct :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  profession Gennady = Barista ∧
  profession Boris = Architect ∧
  profession Andrey = Veterinarian ∧
  profession Vyacheslav = Guitarist :=
by sorry

end final_professions_correct_l650_650270


namespace largest_prime_factor_of_1729_l650_650557

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650557


namespace adam_tattoos_l650_650161

theorem adam_tattoos (j_arm_tattoos j_leg_tattoos j_arms j_legs : ℕ)
  (h1 : j_arm_tattoos = 2) (h2 : j_leg_tattoos = 3)
  (h3 : j_arms = 2) (h4 : j_legs = 2) :
  let j_total_tattoos := (j_arm_tattoos * j_arms) + (j_leg_tattoos * j_legs)
  let a_tattoos := 2 * j_total_tattoos + 3
  in a_tattoos = 23 := by
  sorry

end adam_tattoos_l650_650161


namespace volume_intersection_sphere_tetrahedron_l650_650286

theorem volume_intersection_sphere_tetrahedron :
  let O := (0, 0, 0)
  let A := (4, 0, 0)
  let B := (2, 2 * Real.sqrt 3, 0)
  let C := (2, Real.sqrt 3, 4/3 * Real.sqrt 6)
  let D := (0, Real.sqrt 3, 4/3 * Real.sqrt 6)
  let tetrahedron := {O, A, B, C, D}
  let sphere_radius := 2
  let intersection_volume := (1/6) * (4/3 * Real.pi * sphere_radius^3)
  intersection_volume = 16 * Real.pi / 9 :=
begin
  sorry
end

end volume_intersection_sphere_tetrahedron_l650_650286


namespace solveProfessions_l650_650257

-- Define our types for Friends and Professions
inductive Friend | Andrey | Boris | Vyacheslav | Gennady
inductive Profession | Architect | Barista | Veterinarian | Guitarist

-- Define the seating arrangement and conditions
def seatingArrangement : Friend → Position := sorry
def profession : Friend → Profession := sorry

-- Conditions translated to Lean
axiom condition1 : ∀ (f : Friend), 
  profession f = Profession.Veterinarian → 
  ∃ (neighbor1 neighbor2 : Friend), 
    seatingArrangement neighbor1 = seatingArrangement f + 1 ∨ seatingArrangement neighbor1 = seatingArrangement f - 1 ∧ 
    seatingArrangement neighbor2 = seatingArrangement f + 1 ∨ seatingArrangement neighbor2 = seatingArrangement f - 1 ∧
    (profession neighbor1 = Profession.Architect ∧ profession neighbor2 = Profession.Guitarist) 

axiom condition2 : ∀ (f : Friend), 
  profession f = Profession.Barista → 
  ∃ (rightNeighbor : Friend), 
    seatingArrangement rightNeighbor = seatingArrangement f + 1 ∧ 
    rightNeighbor = Friend.Boris

axiom condition3 : seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Andrey ∧ seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Boris

axiom condition4 : ∀ (f : Friend), 
  f = Friend.Andrey → 
  ∃ (leftNeighbor rightNeighbor : Friend), 
    seatingArrangement leftNeighbor = seatingArrangement f - 1 ∧ 
    seatingArrangement rightNeighbor = seatingArrangement f + 1

axiom condition5 : ∀ (f1 f2 : Friend), 
  (profession f1 = Profession.Guitarist ∧ profession f2 = Profession.Barista) → 
  seatingArrangement f1 ≠ seatingArrangement f2 + 1 ∧ seatingArrangement f1 ≠ seatingArrangement f2 - 1

-- Define the positions
def Position : Type := Fin 4

-- The main theorem to prove the assignments
theorem solveProfessions :
  profession Friend.Gennady = Profession.Barista ∧ 
  profession Friend.Boris = Profession.Architect ∧ 
  profession Friend.Andrey = Profession.Veterinarian ∧ 
  profession Friend.Vyacheslav = Profession.Guitarist :=
sorry

end solveProfessions_l650_650257


namespace lottery_probability_l650_650112

def binom (n k : ℕ) : ℚ := nat.choose n k

theorem lottery_probability : 
  2 * (1 / 30 * 1 / (binom 50 6)) = 2 / 477621000 :=
by 
  sorry

end lottery_probability_l650_650112


namespace total_cost_of_shirt_and_sweater_l650_650508

theorem total_cost_of_shirt_and_sweater (S : ℝ) : 
  (S - 7.43 = 36.46) → (36.46 + S = 80.35) :=
by
  assume h1 : S - 7.43 = 36.46
  sorry

end total_cost_of_shirt_and_sweater_l650_650508


namespace inverse_function_property_l650_650950

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_property (a : ℝ) (h : g a 2 = 4) : f a 2 = 1 := by
  have g_inverse_f : g a (f a 2) = 2 := by sorry
  have a_value : a = 2 := by sorry
  rw [a_value]
  sorry

end inverse_function_property_l650_650950


namespace exists_positive_real_u_l650_650475

theorem exists_positive_real_u (n : ℕ) (h_pos : n > 0) : 
  ∃ u : ℝ, u > 0 ∧ ∀ n : ℕ, n > 0 → (⌊u^n⌋ - n) % 2 = 0 :=
sorry

end exists_positive_real_u_l650_650475


namespace sheet_length_l650_650160

theorem sheet_length (L : ℝ) : 
  (20 * L > 0) → 
  ((16 * (L - 6)) / (20 * L) = 0.64) → 
  L = 30 :=
by
  intro h1 h2
  sorry

end sheet_length_l650_650160


namespace largest_prime_factor_1729_l650_650581

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650581


namespace roots_of_abs_exp_eq_b_l650_650731

theorem roots_of_abs_exp_eq_b (b : ℝ) (h : 0 < b ∧ b < 1) : 
  ∃! (x1 x2 : ℝ), x1 ≠ x2 ∧ abs (2^x1 - 1) = b ∧ abs (2^x2 - 1) = b :=
sorry

end roots_of_abs_exp_eq_b_l650_650731


namespace count_100_digit_even_numbers_l650_650341

theorem count_100_digit_even_numbers : 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  in
  count_valid_numbers = 2 * 3 ^ 98 :=
by 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  have : count_valid_numbers = 2 * 3 ^ 98 := by sorry
  exact this

end count_100_digit_even_numbers_l650_650341


namespace find_sum_pqr_l650_650936

theorem find_sum_pqr (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h : (p + q + r)^3 - p^3 - q^3 - r^3 = 200) : 
  p + q + r = 7 :=
by 
  sorry

end find_sum_pqr_l650_650936


namespace equilateral_triangle_area_eq_circumscribed_polygon_area_l650_650775

theorem equilateral_triangle_area_eq_circumscribed_polygon_area
  (r s : ℝ) 
  (h_r_pos : 0 < r) 
  (h_s_pos : 0 < s) 
  (polygon : Type) 
  (perimeter : polygon → ℝ) 
  (circumscribed : polygon → Prop) 
  (h_polygon_circumscribed : circumscribed polygon) 
  (h_perimeter : perimeter polygon = 2 * s) : 
  ∃ x : ℝ, x = sqrt ((2 * s) / 3 * 2 * r * sqrt 3) ∧ (∃ (triangle : Type) 
    (equilateral : triangle → Prop) 
    (h_equilateral : equilateral triangle) 
    (area : triangle → ℝ),
    area triangle = r * s ∧ r * s = perimeter polygon) :=
begin
  sorry
end

end equilateral_triangle_area_eq_circumscribed_polygon_area_l650_650775


namespace total_distance_covered_l650_650514

def teams_data : List (String × Nat × Nat) :=
  [("Green Bay High", 5, 150), 
   ("Blue Ridge Middle", 7, 200),
   ("Sunset Valley Elementary", 4, 100),
   ("Riverbend Prep", 6, 250)]

theorem total_distance_covered (team : String) (members relays : Nat) :
  (team, members, relays) ∈ teams_data →
    (team = "Green Bay High" → members * relays = 750) ∧
    (team = "Blue Ridge Middle" → members * relays = 1400) ∧
    (team = "Sunset Valley Elementary" → members * relays = 400) ∧
    (team = "Riverbend Prep" → members * relays = 1500) :=
  by
    intros; sorry -- Proof omitted

end total_distance_covered_l650_650514


namespace number_of_factors_180_l650_650350

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.eraseDuplicates.map (λ p, n.factors.count p + 1)).foldr (· * ·) 1

theorem number_of_factors_180 : number_of_factors 180 = 18 := by
  sorry

end number_of_factors_180_l650_650350


namespace unique_triangled_pair_l650_650187

theorem unique_triangled_pair (a b x y : ℝ) (h : ∀ a b : ℝ, (a, b) = (a * x + b * y, a * y + b * x)) : (x, y) = (1, 0) :=
by sorry

end unique_triangled_pair_l650_650187


namespace new_stats_corrected_scores_l650_650101

/-- Let a class have 50 students. In one exam, the average score was 70 and the variance was 102. Later, it was found that the scores of two students were recorded incorrectly: one actually scored 80 but was recorded as 50, and the other actually scored 60 but was recorded as 90. After corrections, the new average score is 70 and the new variance is 90. -/
theorem new_stats_corrected_scores :
  ∃ (scores : Fin 50 → ℝ), (avg scores = 70) ∧ (variance scores = 102) ∧
  (let scores' := (λ i, if i = 0 then 80 else if i = 1 then 60 else scores i) in
  avg scores' = 70 ∧ variance scores' = 90) :=
by
  sorry

-- Definitions for average and variance
def avg (scores : Fin 50 → ℝ) : ℝ :=
  (∑ i, scores i) / 50

def variance (scores : Fin 50 → ℝ) : ℝ :=
  let μ := avg scores in
  (∑ i, (scores i - μ) ^ 2) / 50

end new_stats_corrected_scores_l650_650101


namespace find_rstu_l650_650495

theorem find_rstu (a x y c : ℝ) (r s t u : ℤ) (hc : a^10 * x * y - a^8 * y - a^7 * x = a^6 * (c^3 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^u) = a^6 * c^3 ∧ r * s * t * u = 0 :=
by
  sorry

end find_rstu_l650_650495


namespace find_angle_C_area_range_l650_650783

-- Definitions and conditions
variables {A B C : Real}
variable {a b c p : Real}
axiom triangle_ABC : c = 4
axiom tan_roots : ∀ x, x^2 + (1 + p) * x + p + 2 = 0 → x = tan A ∨ x = tan B

-- Finding the magnitude of angle C
theorem find_angle_C : C = 3 * Real.pi / 4 :=
sorry

-- Determining the range of possible areas for the triangle
theorem area_range (ab : Real) (S : Real) 
    (h_cos: c^2 = a^2 + b^2 - 2 * a * b * (-(Real.sqrt 2 / 2))) 
    (h_area: S = 1/2 * a * b * (Real.sqrt 2 / 2)) : 
    0 < S ∧ S ≤ 4 * Real.sqrt 2 - 4 :=
sorry

end find_angle_C_area_range_l650_650783


namespace scientific_notation_example_l650_650741

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (3650000 : ℝ) = a * 10 ^ n :=
sorry

end scientific_notation_example_l650_650741


namespace f_difference_l650_650456

def f (n : ℕ) : ℝ :=
  (5 + 3 * Real.sqrt 5) / 10 * ((1 + Real.sqrt 5) / 2) ^ n +
  (5 - 3 * Real.sqrt 5) / 10 * ((1 - Real.sqrt 5) / 2) ^ n

theorem f_difference (n : ℕ) : f (n + 1) - f (n - 1) = f n :=
  sorry

end f_difference_l650_650456


namespace sum_of_diagonals_l650_650436

def FG : ℝ := 4
def HI : ℝ := 4
def GH : ℝ := 11
def IJ : ℝ := 11
def FJ : ℝ := 15

theorem sum_of_diagonals (x y z : ℝ) (h1 : z^2 = 4 * x + 121) (h2 : z^2 = 11 * y + 16)
  (h3 : x * y = 44 + 15 * z) (h4 : x * z = 4 * z + 225) (h5 : y * z = 11 * z + 60) :
  3 * z + x + y = 90 :=
sorry

end sum_of_diagonals_l650_650436


namespace problem_statement_l650_650279

variable (x : ℝ)
def A := ({-3, x^2, x + 1} : Set ℝ)
def B := ({x - 3, 2 * x - 1, x^2 + 1} : Set ℝ)

theorem problem_statement (hx : A x ∩ B x = {-3}) : 
  x = -1 ∧ A x ∪ B x = ({-4, -3, 0, 1, 2} : Set ℝ) :=
by
  sorry

end problem_statement_l650_650279


namespace smallest_bdf_value_l650_650898

open Nat

theorem smallest_bdf_value
  (a b c d e f : ℕ)
  (h1 : (a + 1) * c * e = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) :
  (b * d * f = 60) :=
sorry

end smallest_bdf_value_l650_650898


namespace largest_prime_factor_of_1729_is_19_l650_650604

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650604


namespace sqrt_factorial_div_l650_650222

theorem sqrt_factorial_div {n : ℕ} (hn : n = 10) (d : ℕ) (hd : d = 210) :
  sqrt (↑((nat.factorial n) / d)) = 24 * sqrt 30 :=
by
  sorry

end sqrt_factorial_div_l650_650222


namespace negation_of_universal_statement_l650_650960

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end negation_of_universal_statement_l650_650960


namespace max_coach_handshakes_l650_650170

-- Define the problem variables and conditions
noncomputable def coach_max_handshakes (total_handshakes : ℕ) := 
  ∃ (n k : ℕ), nat.choose n 2 + k = total_handshakes ∧ k = 0

-- Statement with total handshakes set to 465
theorem max_coach_handshakes : coach_max_handshakes 465 :=
sorry

end max_coach_handshakes_l650_650170


namespace check_correct_digit_increase_l650_650013

-- Definition of the numbers involved
def number1 : ℕ := 732
def number2 : ℕ := 648
def number3 : ℕ := 985
def given_sum : ℕ := 2455
def calc_sum : ℕ := number1 + number2 + number3
def difference : ℕ := given_sum - calc_sum

-- Specify the smallest digit that needs to be increased by 1
def smallest_digit_to_increase : ℕ := 8

-- Theorem to check the validity of the problem's claim
theorem check_correct_digit_increase :
  (smallest_digit_to_increase = 8) →
  (calc_sum + 10 = given_sum - 80) :=
by
  intro h
  sorry

end check_correct_digit_increase_l650_650013


namespace log_inequality_l650_650375

theorem log_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : 
  Real.log b / Real.log a + Real.log a / Real.log b ≤ -2 := sorry

end log_inequality_l650_650375


namespace still_water_time_l650_650155

-- Definitions based on conditions
variables (d : ℝ) (v u : ℝ)

-- Conditions translated into Lean statements
def downstream_condition : Prop := d = 6 * (v + u)
def upstream_condition : Prop := d = 8 * (v - u)

-- Goal statement for the proof
theorem still_water_time (h1 : downstream_condition d v u) (h2 : upstream_condition d v u) :
  ∃ t : ℝ, t = 48 / 7 ∧ d = t * v :=
begin
  sorry
end

end still_water_time_l650_650155


namespace num_100_digit_even_numbers_l650_650332

theorem num_100_digit_even_numbers : 
  let digit_set := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let valid_number (digits : list ℕ) := 
    digits.length = 100 ∧ digits.head ∈ {1, 3} ∧ 
    digits.last ∈ {0} ∧ 
    ∀ d ∈ digits.tail.init, d ∈ digit_set
  (∃ (m : ℕ), valid_number (m.digits 10)) = 2 * 3^98 := 
sorry

end num_100_digit_even_numbers_l650_650332


namespace pure_imaginary_conjugate_l650_650829

def i : ℂ := complex.I

noncomputable def z (b : ℂ) : ℂ :=
  (1 + b * i) / (2 + i)

theorem pure_imaginary_conjugate (b : ℝ) (h : z b).re = 0 : 
  conj (z b) = i :=
sorry

end pure_imaginary_conjugate_l650_650829


namespace largest_prime_factor_of_1729_l650_650553

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650553


namespace no_finite_centers_of_symmetry_l650_650177

-- We define what it means for a figure to have a center of symmetry.
-- Assume that having more than one implies an infinite number.
def has_center_of_symmetry {F : Type} (figure : F) (center : F → F) : Prop :=
  ∀ x, (center (center x)) = x  -- A point symmetric with respect to the center remains the same after double symmetry.

-- The main theorem statement.
theorem no_finite_centers_of_symmetry {F : Type} (figure : F) (center : F → F) :
  (∃ O1 O2 : F, O1 ≠ O2 ∧ has_center_of_symmetry figure center O1 ∧ has_center_of_symmetry figure center O2) →
  ∀ n : ℕ, ¬ (∃ L : list F, L.length = n ∧ ∀ O ∈ L, has_center_of_symmetry figure center O) :=
sorry

end no_finite_centers_of_symmetry_l650_650177


namespace bases_with_final_digit_one_l650_650768

theorem bases_with_final_digit_one :
  { b : ℕ | 3 ≤ b ∧ b ≤ 10 ∧ 624 % b = 0 }.card = 4 :=
by
  sorry

end bases_with_final_digit_one_l650_650768


namespace length_of_other_train_is_correct_l650_650094

-- Define the conditions
def first_train_length : ℝ := 240
def first_train_speed_kmph : ℝ := 120
def second_train_speed_kmph : ℝ := 80
def crossing_time : ℝ := 9

-- Convert speeds from km/h to m/s
def kmph_to_mps (s: ℝ) : ℝ := s * 1000 / 3600

def first_train_speed_mps := kmph_to_mps first_train_speed_kmph
def second_train_speed_mps := kmph_to_mps second_train_speed_kmph

-- Define the relative speed and distance covered
def relative_speed := first_train_speed_mps + second_train_speed_mps
def total_distance := relative_speed * crossing_time

-- Define the length of the second train
def second_train_length : ℝ := total_distance - first_train_length

theorem length_of_other_train_is_correct :
  second_train_length = 259.95 :=
by 
  -- Calculation done in the proof step
  -- Showing that the calculated length is equal to 259.95
  sorry

end length_of_other_train_is_correct_l650_650094


namespace time_to_fill_cistern_l650_650659

def rateA (C : ℝ) : ℝ := C / 10
def rateB (C : ℝ) : ℝ := C / 12
def netRate (C : ℝ) : ℝ := rateA C - rateB C

theorem time_to_fill_cistern (C : ℝ) (hC : C > 0) : C / netRate C = 60 := by
  have h1 : netRate C = C / 60 := by
    calc
      netRate C = rateA C - rateB C        : rfl
              _ = C / 10 - C / 12           : rfl
              _ = (6 * C) / 60 - (5 * C) / 60 : by rw [div_eq_mul_one_div, div_eq_mul_one_div]
              _ = C / 60                    : by ring
  rw [h1, div_div_eq_div_mul, mul_inv_cancel hC]
  norm_num
  exact hC

#check time_to_fill_cistern

end time_to_fill_cistern_l650_650659


namespace min_value_of_f_on_interval_l650_650869

noncomputable def f (x : ℝ) : ℝ := (1/2)*x^2 - x - 2*Real.log x

theorem min_value_of_f_on_interval : 
  ∃ c ∈ set.Icc (1:ℝ) Real.exp 1, ∀ x ∈ set.Icc (1:ℝ) Real.exp 1, f x ≥ f c ∧ f c = -2 * Real.log 2 := 
by
  sorry

end min_value_of_f_on_interval_l650_650869


namespace parallel_lines_condition_l650_650671

theorem parallel_lines_condition (a : ℝ) : 
  (∃ l1 l2 : ℝ → ℝ, 
    (∀ x y : ℝ, l1 x + a * y + 6 = 0) ∧ 
    (∀ x y : ℝ, (a - 2) * x + 3 * y + 2 * a = 0) ∧
    l1 = l2 ↔ a = 3) :=
sorry

end parallel_lines_condition_l650_650671


namespace correlation_coefficient_correct_option_l650_650655

variable (r : ℝ)

-- Definitions of Conditions
def positive_correlation : Prop := r > 0 → ∀ x y : ℝ, x * y > 0
def range_r : Prop := -1 < r ∧ r < 1
def correlation_strength : Prop := |r| < 1 → (∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ δ < ε ∧ |r| < δ)

-- Theorem statement
theorem correlation_coefficient_correct_option :
  (positive_correlation r) ∧
  (range_r r) ∧
  (correlation_strength r) →
  (r ≠ 0 → |r| < 1) :=
by
  sorry

end correlation_coefficient_correct_option_l650_650655


namespace fraction_value_unchanged_l650_650415

theorem fraction_value_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / (x + y) = (2 * x) / (2 * (x + y))) :=
by sorry

end fraction_value_unchanged_l650_650415


namespace largest_prime_factor_1729_l650_650585

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650585


namespace last_number_in_michael_game_l650_650838

noncomputable def michael_game_last_number(n : ℕ) : ℕ :=
if h : n > 0 then
  let rec eliminate (l : List ℕ) (step : ℕ) :=
    match l with
    | [] => []
    | (x :: xs) => x :: eliminate (xs.drop 1) (step + 1)
  let numbers := List.range n |>.map (λ i => i + 1)
  in (eliminate numbers 1).getLast h
else 0

theorem last_number_in_michael_game : michael_game_last_number 150 = 128 :=
by
  sorry

end last_number_in_michael_game_l650_650838


namespace share_of_B_in_profit_l650_650700

variable {D : ℝ} (hD_pos : 0 < D)

def investment (D : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let C := 2.5 * D
  let B := 1.25 * D
  let A := 5 * B
  (A, B, C, D)

def totalInvestment (A B C D : ℝ) : ℝ :=
  A + B + C + D

theorem share_of_B_in_profit (D : ℝ) (profit : ℝ) (hD : 0 < D)
  (h_profit : profit = 8000) :
  let ⟨A, B, C, D⟩ := investment D
  B / totalInvestment A B C D * profit = 1025.64 :=
by
  sorry

end share_of_B_in_profit_l650_650700


namespace iterated_six_times_l650_650449

noncomputable def s (θ : ℝ) : ℝ := 1 / (1 + θ)

theorem iterated_six_times (θ : ℝ) :
  s (s (s (s (s (s θ))))) = (8 + 5 * θ) / (13 + 8 * θ) :=
sorry

example : s (s (s (s (s (s 50))))) = 258 / 413 :=
by rw [iterated_six_times 50]

end iterated_six_times_l650_650449


namespace count_duty_arrangements_l650_650681

open Finset

-- Define the students and the days.
inductive Student
| A | B | C | D | E

inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday

-- Define the duty roster condition. 
def valid_arrangement (roster : Day → Student) : Prop :=
  (roster Day.Monday = Student.A ∨ roster Day.Tuesday = Student.A) ∧
  roster Day.Friday ≠ Student.B

-- Define the number of unique valid arrangements.
def count_valid_arrangements : Nat :=
  (2:Nat) * (3:Nat) * (3.factorial : Nat) -- 2 choices for A, 3 for B, 6 for the rest.

-- The theorem we need to prove.
theorem count_duty_arrangements : count_valid_arrangements = 36 := 
  sorry

end count_duty_arrangements_l650_650681


namespace find_an_from_sums_l650_650853

noncomputable def geometric_sequence (a : ℕ → ℝ) (q r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_an_from_sums (a : ℕ → ℝ) (q r : ℝ) (S3 S6 : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 :  a 1 * (1 - q^3) / (1 - q) = S3) 
  (h3 : a 1 * (1 - q^6) / (1 - q) = S6) : 
  ∃ a1 q, a n = a1 * q^(n-1) := 
by
  sorry

end find_an_from_sums_l650_650853


namespace largest_prime_factor_of_1729_l650_650599

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650599


namespace largest_prime_factor_of_1729_l650_650641

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650641


namespace proof_sin_times_sin_sub_l650_650273

noncomputable def sin_times_sin_sub := 
  λ (α : ℝ) (h : Real.tan α = 3), Real.sin α * Real.sin (3 * Real.pi / 2 - α)

theorem proof_sin_times_sin_sub :
  ∀ α : ℝ, Real.tan α = 3 → sin_times_sin_sub α (Real.tan α) = -3/10 :=
by
  intros α h
  sorry

end proof_sin_times_sin_sub_l650_650273


namespace exponential_function_property_l650_650672

theorem exponential_function_property (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = 5^x) (h2 : f(a + b) = 3) :
  f(a) * f(b) = 3 :=
by
  sorry

end exponential_function_property_l650_650672


namespace irrigation_ditches_length_l650_650706

theorem irrigation_ditches_length (L : ℝ) : 
  (∀ p : ℝ × ℝ, p.1 ≥ 0 ∧ p.1 ≤ 12 ∧ p.2 ≥ 0 ∧ p.2 ≤ 12 → ∃ d : ℝ × ℝ → ℝ, (d(p) ≤ 1 ∧ ∀ q : ℝ × ℝ, ∃ r : ℝ, r ∈ [0, L]) ∧ d(q) ≤ 1)) 
  → (L > 70) := 
by
  sorry

end irrigation_ditches_length_l650_650706


namespace not_finite_many_symmetry_centers_l650_650179

variable (α : Type) [AddCommGroup α] [Module ℝ α]

def SymmetryCenter (O : α) (S : α → α) := ∀ x : α, S (S x) = x

theorem not_finite_many_symmetry_centers :
  (∀ (O1 O2 : α) (S : α → α), 
    SymmetryCenter O1 S → SymmetryCenter O2 S → 
    ∃ O3 : α, O3 ≠ O1 ∧ O3 ≠ O2 ∧ SymmetryCenter O3 S) 
  → ¬ (∃ (centers : Finset α) (S : α → α), Finset.card centers > 1 ∧ ∀ O ∈ centers, SymmetryCenter O S) :=
by
  intro h
  sorry

end not_finite_many_symmetry_centers_l650_650179


namespace largest_prime_factor_of_1729_l650_650560

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650560


namespace largest_prime_factor_of_1729_l650_650633

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650633


namespace jigsaw_puzzle_completion_l650_650463

theorem jigsaw_puzzle_completion (p : ℝ) :
  let total_pieces := 1000
  let pieces_first_day := total_pieces * 0.10
  let remaining_after_first_day := total_pieces - pieces_first_day

  let pieces_second_day := remaining_after_first_day * (p / 100)
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day

  let pieces_third_day := remaining_after_second_day * 0.30
  let remaining_after_third_day := remaining_after_second_day - pieces_third_day

  remaining_after_third_day = 504 ↔ p = 20 := 
by {
    sorry
}

end jigsaw_puzzle_completion_l650_650463


namespace solve_profession_arrangement_l650_650248

inductive Profession
| architect
| barista
| veterinarian
| guitarist

inductive Person
| Andrey
| Boris
| Vyacheslav
| Gennady

open Profession
open Person

structure SeatingArrangement :=
(seat1 : Person)
(seat2 : Person)
(seat3 : Person)
(seat4 : Person)

structure Assignment :=
(profession : Person → Profession)

def correct_arrangement_and_profession (sa : SeatingArrangement) (asgmt : Assignment) : Prop :=
(sa.seat1 ≠ sa.seat2 ∧ sa.seat1 ≠ sa.seat3 ∧ sa.seat1 ≠ sa.seat4 ∧
 sa.seat2 ≠ sa.seat3 ∧ sa.seat2 ≠ sa.seat4 ∧ sa.seat3 ≠ sa.seat4) ∧
((asgmt profession sa.seat1 = barista) ∧ (asgmt profession sa.seat2 = architect) ∧ 
 (asgmt profession sa.seat3 = veterinarian) ∧ (asgmt profession sa.seat4 = guitarist)) ∧
(sa.seat3 = Andrey) ∧ 
(sa.seat2 = Boris) ∧ 
(sa.seat4 = Vyacheslav)

theorem solve_profession_arrangement : 
  ∃ (sa : SeatingArrangement) (asgmt : Assignment), correct_arrangement_and_profession sa asgmt :=
sorry

end solve_profession_arrangement_l650_650248


namespace perimeter_triang_OMF_l650_650786

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem perimeter_triang_OMF (x y : ℝ)
  (hM_on_parabola : x^2 = 4 * y)
  (hF_focus : F = (0, 1))
  (hM_dist_F : distance (x, y) (0, 1) = 5)
  (hO_orig : O = (0, 0)) :
  let M := (x, y)
      O := (0, 0)
      F := (0, 1)
      OM := distance O M
      MF := distance M F
      OF := distance O F in
  OM + MF + OF = 6 + 4 * real.sqrt 2 :=
begin
  sorry
end

end perimeter_triang_OMF_l650_650786


namespace largest_prime_factor_of_1729_l650_650594

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650594


namespace count_100_digit_even_numbers_l650_650329

theorem count_100_digit_even_numbers : 
  let valid_digits := {0, 1, 3}
  let num_digits := 100
  let num_even_digits := 2 * 3^98
  ∀ n : ℕ, n = num_digits → (∃ (digits : Fin n → ℕ), 
    (∀ i, digits i ∈ valid_digits) ∧ 
    digits 0 ≠ 0 ∧ 
    digits (n-1) = 0) → 
    (num_even_digits = 2 * 3^98) :=
by
  sorry

end count_100_digit_even_numbers_l650_650329


namespace largest_prime_factor_1729_l650_650536

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650536


namespace seating_arrangement_l650_650237

-- Definitions of friends and their professions
inductive Friend
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq

-- Seating condition definitions
def seated_right (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = a ∧ seating (i + 1) = b)

def seated_between (a b c : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = b ∧ seating (i - 1) = a ∧ seating (i + 1) = c)

def neighbors (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, (seating i = a ∧ seating (i - 1) = b) ∨ (seating i = a ∧ seating (i + 1) = b))

-- The proposition to be proved
theorem seating_arrangement
  (seating : ℕ → Friend)
  (profession : Friend → Profession)
  (h1 : ∃ i, seating_between Profession.Veterinarian Profession.Architect Profession.Guitarist seating)
  (h2 : seated_right Profession.Barista Profession.Boris seating)
  (h3 : ∃ i, seating i = Friend.Vyacheslav ∧ (∃ j, seating j = Friend.Andrey ∧ j < i) ∧ (∃ k, seating k = Friend.Boris ∧ k < i))
  (h4 : ∀ a, seating a = Friend.Andrey → neighbors Friend.Andrey seating)
  (h5 : ∀ a, seating a = Profession.Guitarist → ∀ b, seating (b - 1) = Profession.Barista ∨ seating (b + 1) = Profession.Barista → false) :
  (profession Friend.Gennady = Profession.Barista ∧
   profession Friend.Boris = Profession.Architect ∧
   profession Friend.Andrey = Profession.Veterinarian ∧
   profession Friend.Vyacheslav = Profession.Guitarist) :=
begin
  sorry
end

end seating_arrangement_l650_650237


namespace equal_areas_of_hexagons_l650_650478

-- Definition of a Hexagon with its vertices
structure Hexagon :=
(vertices : Fin 6 → Point)

-- Definition of Points
structure Point :=
(x : ℝ)
(y : ℝ)

-- Function to calculate the midpoint of a line segment
def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2 }

-- Condition that hexagons have the same midpoints of their sides
def same_midpoints (P Q : Hexagon) :=
  ∀ i : Fin 6, 
  midpoint (P.vertices i) (P.vertices (i + 1) % 6) = midpoint (Q.vertices i) (Q.vertices (i + 1) % 6)

-- Definition of the area of a hexagon (Placeholder)
noncomputable def area (H : Hexagon) : ℝ := sorry

-- Theorem stating that areas are equal given the same midpoints
theorem equal_areas_of_hexagons (H₁ H₂ : Hexagon) (h : same_midpoints H₁ H₂) : area H₁ = area H₂ :=
  sorry

end equal_areas_of_hexagons_l650_650478


namespace greatest_possible_x_l650_650954

-- Define the numbers and the lcm condition
def num1 := 12
def num2 := 18
def lcm_val := 108

-- Function to calculate the lcm of three numbers
def lcm3 (a b c : ℕ) := Nat.lcm (Nat.lcm a b) c

-- Proposition stating the problem condition
theorem greatest_possible_x (x : ℕ) (h : lcm3 x num1 num2 = lcm_val) : x ≤ lcm_val := sorry

end greatest_possible_x_l650_650954


namespace num_odd_sum_subsets_l650_650372

theorem num_odd_sum_subsets : 
  let s := [85, 91, 99, 132, 166, 170, 175] in
  (s.toFinset.subsetsOfCard 3).filter (λ t, t.sum % 2 = 1).card = 16 :=
by sorry

end num_odd_sum_subsets_l650_650372


namespace largest_prime_factor_of_1729_l650_650596

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650596


namespace cost_of_graveling_roads_l650_650660

def lawn := { length : ℝ, width : ℝ }
def road := { width : ℝ }
def cost_per_sq_m : ℝ := 4

noncomputable def area_of_road (lawn_dim : lawn) (road_dim : road) : ℝ :=
  let road_length := lawn_dim.length
  let road_breadth := lawn_dim.width
  let intersection_area := road_dim.width * road_dim.width
  let area_road_lengthwise := road_length * road_dim.width
  let area_road_breadthwise := road_breadth * road_dim.width
  area_road_lengthwise + area_road_breadthwise - intersection_area

noncomputable def total_cost (area : ℝ) : ℝ := area * cost_per_sq_m

theorem cost_of_graveling_roads :
  ∀ (lawn_dim : lawn) (road_dim : road),
  lawn_dim.length = 80 →
  lawn_dim.width = 60 →
  road_dim.width = 10 →
  total_cost (area_of_road lawn_dim road_dim) = 5200 := by
  sorry

end cost_of_graveling_roads_l650_650660


namespace find_a_plus_b_l650_650181

noncomputable def area_of_triangle (P1 P2 P3 : Point) : ℝ :=
  let s := (dist P1 P2 + dist P2 P3 + dist P3 P1) / 2
  real.sqrt (s * (s - dist P1 P2) * (s - dist P2 P3) * (s - dist P3 P1))

theorem find_a_plus_b : ∃ a b : ℕ, 
  let ω1 := circle (center₁ : Point) 4
      ω2 := circle (center₂ : Point) 4
      ω3 := circle (center₃ : Point) 4 in
  ∀ P1 P2 P3 : Point, 
  P1 ∈ ω1 → P2 ∈ ω2 → P3 ∈ ω3 →
  dist P1 P2 = dist P2 P3 ∧ dist P2 P3 = dist P3 P1 ∧
  tangent P1 ω1 ∧ tangent P2 ω2 ∧ tangent P3 ω3 →
  area_of_triangle P1 P2 P3 = real.sqrt a + real.sqrt b →
  a + b = 552 :=
begin
  sorry
end

end find_a_plus_b_l650_650181


namespace value_of_a2012_l650_650420

def sequence (a : ℤ) (b : ℤ) : ℕ → ℤ
| 1     := a
| 2     := b
| (n+3) := sequence (n+2) + sequence (n+4)

theorem value_of_a2012 (a b : ℤ) : sequence a b 2012 = b :=
sorry

end value_of_a2012_l650_650420


namespace ball_first_less_than_25_cm_l650_650676

theorem ball_first_less_than_25_cm (n : ℕ) :
  ∀ n, (200 : ℝ) * (3 / 4) ^ n < 25 ↔ n ≥ 6 := by sorry

end ball_first_less_than_25_cm_l650_650676


namespace unit_prices_minimum_B_seedlings_l650_650200

-- Definition of the problem conditions and the results of Part 1
theorem unit_prices (x : ℝ) : 
  (1200 / (1.5 * x) + 10 = 900 / x) ↔ x = 10 :=
by
  sorry

-- Definition of the problem conditions and the result of Part 2
theorem minimum_B_seedlings (m : ℕ) : 
  (10 * m + 15 * (100 - m) ≤ 1314) ↔ m ≥ 38 :=
by
  sorry

end unit_prices_minimum_B_seedlings_l650_650200


namespace sqrt_ratio_eq_correct_answer_l650_650220

open Real 

def ten_factorial : ℝ := 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def two_hundred_ten : ℝ := 2 * 3 * 5 * 7
def ratio : ℝ := ten_factorial / two_hundred_ten
def correct_answer : ℝ := 24 * Real.sqrt 30

theorem sqrt_ratio_eq_correct_answer :
  Real.sqrt ratio = correct_answer :=
by
  sorry

end sqrt_ratio_eq_correct_answer_l650_650220


namespace find_N_l650_650684

theorem find_N :
  ∀ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ d ≠ 0
    → let X := 1000 * a + 100 * b + 10 * c + d in
      let Y := 1000 * d + 100 * c + 10 * b + a in
      let N := X + Y in
      N % 100 = 0
      → a + d = 10
      → b + c = 9
      → N = 11000 := 
sorry

end find_N_l650_650684


namespace average_problem_l650_650827

theorem average_problem
  (a b c d : ℚ)
  (h1 : (a + d) / 2 = 40)
  (h2 : (b + d) / 2 = 60)
  (h3 : (a + b) / 2 = 50)
  (h4 : (b + c) / 2 = 70) :
  c - a = 40 :=
begin
  sorry,
end

end average_problem_l650_650827


namespace tan_a1_a13_eq_sqrt3_l650_650305

-- Definition of required constants and properties of the geometric sequence
noncomputable def a (n : Nat) : ℝ := sorry -- Geometric sequence definition (abstract)

-- Given condition: a_3 * a_11 + 2 * a_7^2 = 4π
axiom geom_seq_cond : a 3 * a 11 + 2 * (a 7)^2 = 4 * Real.pi

-- Property of geometric sequence: a_3 * a_11 = a_7^2
axiom geom_seq_property : a 3 * a 11 = (a 7)^2

-- To prove: tan(a_1 * a_13) = √3
theorem tan_a1_a13_eq_sqrt3 : Real.tan (a 1 * a 13) = Real.sqrt 3 := by
  sorry

end tan_a1_a13_eq_sqrt3_l650_650305


namespace part1_part2_l650_650315

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 + x ^ 2

noncomputable def g (f : ℝ -> ℝ) (x : ℝ) : ℝ := f x * Real.exp x

theorem part1 (a : ℝ) (h : deriv (f a) (-4 / 3) = 0) : a = 1 / 2 :=
by
  sorry

theorem part2 : 
  ∃ (I_decreasing I_increasing : Set ℝ), 
    I_decreasing = {x | x < -4} ∪ {x | -1 < x ∧ x ≤ 0} ∧
    I_increasing = {x | -4 < x ∧ x ≤ -1} ∪ {x | 0 < x} ∧
    ∀ x, (deriv (g (f (1 / 2)) x) < 0 ↔ x ∈ I_decreasing) ∧ 
         (deriv (g (f (1 / 2)) x) > 0 ↔ x ∈ I_increasing) :=
by
  sorry

end part1_part2_l650_650315


namespace train_crossing_time_l650_650664

-- Define the length of the train and bridge
def train_length : ℝ := 110
def bridge_length : ℝ := 112

-- Define the speed of the train in km/hr and convert it to m/s
def speed_km_hr : ℝ := 72
def speed_m_s : ℝ := speed_km_hr * 1000 / 3600

-- Calculate total distance
def total_distance : ℝ := train_length + bridge_length

-- Theorem statement
theorem train_crossing_time : total_distance / speed_m_s = 11.1 :=
by
  have h1 : total_distance = 222 := by sorry
  have h2 : speed_m_s = 20 := by sorry
  rw [h1, h2]
  norm_num

end train_crossing_time_l650_650664


namespace largest_prime_factor_of_1729_l650_650623

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650623


namespace trajectory_center_M_l650_650116

noncomputable theory

-- Definition of the hyperbola with the given equation
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 15 = 1

-- Conditions: M passes through the left focus of the hyperbola and is tangent to the line x = 4
def left_focus (point : ℝ × ℝ) : Prop := point = (-4, 0)
def tangent_line (M : ℝ × ℝ) : Prop := abs (M.fst - 4) = abs (M.fst + 4)

-- Given the conditions to prove the equation of the trajectory
theorem trajectory_center_M (M : ℝ × ℝ) (p : Prop) :
  left_focus M → tangent_line M → (p ↔ (M.snd)^2 = -16 * (M.fst)) :=
by
  sorry

end trajectory_center_M_l650_650116


namespace break_even_solution_l650_650894

noncomputable def handle_break_even (X : ℕ) : Prop :=
  let cost_per_handle := 0.60
  let fixed_cost_per_week := 7640
  let selling_price_per_handle := 4.60
  (fixed_cost_per_week + cost_per_handle * X) = (selling_price_per_handle * X)

theorem break_even_solution : ∃ X, handle_break_even X ∧ X = 1910 :=
by
  use 1910
  unfold handle_break_even
  simp
  sorry

end break_even_solution_l650_650894


namespace expected_value_first_outstanding_sequence_l650_650140

-- Define indicator variables and the harmonic number
noncomputable def I (k : ℕ) := 1 / k
noncomputable def H (n : ℕ) := ∑ i in Finset.range (n + 1), I (i + 1)

-- Theorem: Expected value of the first sequence of outstanding grooms
theorem expected_value_first_outstanding_sequence : 
  (H 100) = 5.187 := 
sorry

end expected_value_first_outstanding_sequence_l650_650140


namespace speed_increase_l650_650126

theorem speed_increase :
  ∀ (normal_speed increased_speed journey_distance : ℕ),
  normal_speed = 25 →
  journey_distance = 300 →
  (journey_distance / normal_speed - journey_distance / increased_speed) = 2 →
  (increased_speed - normal_speed) = 5 :=
by {
  intros,
  sorry,
}

end speed_increase_l650_650126


namespace ownership_of_all_three_pets_l650_650413

theorem ownership_of_all_three_pets :
  (total_pets : ℕ) (cat_owners : ℕ) (dog_owners : ℕ) (rabbit_owners : ℕ) (owners_two_types : ℕ)
  (h1 : total_pets = 60) (h2 : cat_owners = 30) (h3 : dog_owners = 40) (h4 : rabbit_owners = 16)
  (h5 : owners_two_types = 12) :
  ∃ (x : ℕ), x = 7 :=  
by
  sorry

end ownership_of_all_three_pets_l650_650413


namespace correct_calculation_l650_650673

theorem correct_calculation (x : ℤ) (h : x - 32 = 33) : x + 32 = 97 := 
by 
  sorry

end correct_calculation_l650_650673


namespace girls_with_brown_eyes_and_light_brown_skin_l650_650050

theorem girls_with_brown_eyes_and_light_brown_skin 
  (total_girls : ℕ)
  (light_brown_skin_girls : ℕ)
  (blue_eyes_fair_skin_girls : ℕ)
  (brown_eyes_total : ℕ)
  (total_girls_50 : total_girls = 50)
  (light_brown_skin_31 : light_brown_skin_girls = 31)
  (blue_eyes_fair_skin_14 : blue_eyes_fair_skin_girls = 14)
  (brown_eyes_18 : brown_eyes_total = 18) :
  ∃ (brown_eyes_light_brown_skin_girls : ℕ), brown_eyes_light_brown_skin_girls = 13 :=
by sorry

end girls_with_brown_eyes_and_light_brown_skin_l650_650050


namespace sum_of_three_integers_l650_650965

theorem sum_of_three_integers (a b c : ℕ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diff: a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_prod: a * b * c = 5^4) : a + b + c = 131 :=
sorry

end sum_of_three_integers_l650_650965


namespace largest_prime_factor_of_1729_l650_650637

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650637


namespace number_of_sister_pairs_is_two_l650_650395

noncomputable def f : ℝ → ℝ :=
λ x, if x < 0 then x^2 + 2 * x else 2 / (Real.exp x)

def symmetric_about_origin (A B : ℝ × ℝ) : Prop :=
B.1 = -A.1 ∧ B.2 = -A.2

def sister_pairs (f : ℝ → ℝ) : ℕ :=
Set.card {pair : ℝ × ℝ | (symmetric_about_origin pair.fst pair.snd) ∧ pair.fst.2 = f pair.fst.1 ∧ pair.snd.2 = f pair.snd.1}

theorem number_of_sister_pairs_is_two : sister_pairs f = 2 := by
  sorry

end number_of_sister_pairs_is_two_l650_650395


namespace angle_acb_75_degrees_l650_650403

theorem angle_acb_75_degrees
  (A B C D : Type)
  [real_points A B C D]
  (angle_ABC_eq_60 : ∠ABC = 60)
  (D_on_BC : D ∈ segment B C)
  (BD_two_cd : 2 * dist B D = dist D C)
  (angle_DAB_30 : ∠DAB = 30) :
  ∠ACB = 75 :=
by {
  sorry
}

end angle_acb_75_degrees_l650_650403


namespace sphere_ratio_l650_650985
-- Importing the necessary libraries

-- Formulating the Lean 4 statement
theorem sphere_ratio (R r : ℝ) 
  (h1 : (∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D ∧ D = 2 * sqrt (R * r))) 
  (h2 : (∀ (a b : ℝ), (a / 2) ^ 2 + (b / 2) ^ 2 = (2 * sqrt (R * r)) ^ 2))
  : R / r = 2 + sqrt 3 := 
sorry

end sphere_ratio_l650_650985


namespace largest_prime_factor_1729_l650_650612

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650612


namespace final_professions_correct_l650_650271

-- Define the people and their positions
inductive Person
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq, Inhabited

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq, Inhabited

open Person Profession

-- Given positions in seats
def seatingArrangement : Person → Nat := 
λ p, 
  match p with
  | Andrey    => 3
  | Boris     => 2
  | Vyacheslav => 4
  | Gennady   => 1

-- Given professions
def profession : Person → Profession := 
λ p, 
  match p with
  | Andrey    => Veterinarian
  | Boris     => Architect
  | Vyacheslav => Guitarist
  | Gennady   => Barista

-- Define the conditions
def condition1 : Prop :=
  (seatingArrangement Andrey = 3) ∧ 
  (seatingArrangement Boris = 2) ∧
  (seatingArrangement Vyacheslav = 4) ∧ 
  (seatingArrangement Gennady = 1)

def condition2 : Prop := 
  seatingArrangement (Person.Barista) + 1 = seatingArrangement Boris

def condition3 : Prop := 
  seatingArrangement Vyacheslav > seatingArrangement Andrey ∧ 
  seatingArrangement Vyacheslav > seatingArrangement Boris

def condition4 : Prop := 
  seatingArrangement (Person.Veterinarian) ≠ 1 ∧ seatingArrangement (Person.Veterinarian) ≠ 4

def condition5 : Prop := 
  (seatingArrangement (Person.Barista) ≠ seatingArrangement (Person.Guitarist) + 1 ∧
  seatingArrangement (Person.Guitarist) ≠ seatingArrangement (Person.Barista) + 1)

-- Prove the final professions
theorem final_professions_correct :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  profession Gennady = Barista ∧
  profession Boris = Architect ∧
  profession Andrey = Veterinarian ∧
  profession Vyacheslav = Guitarist :=
by sorry

end final_professions_correct_l650_650271


namespace probability_green_slope_l650_650158

-- problem setup
variables {α β γ : ℝ}
hypothesis h : cos(α)^2 + cos(β)^2 + cos(γ)^2 = 1

-- statement to prove
theorem probability_green_slope (h : cos(α)^2 + cos(β)^2 + cos(γ)^2 = 1) : 
  1 - cos(α)^2 - cos(β)^2 = cos(γ)^2 :=
by 
  sorry

end probability_green_slope_l650_650158


namespace expected_value_product_l650_650492

/-- 
  The expected value of the product of two three-digit numbers M and N, 
  where the digits 1, 2, 3, 4, 5, and 6 are chosen randomly without replacement 
  to form M = ABC and N = DEF, is 143745.
-/
theorem expected_value_product : 
  (∃ (M N : ℕ), M ∈ {100 * A + 10 * B + C | A B C ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C} ∧ 
                      N ∈ {100 * D + 10 * E + F | D E F ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧ D ≠ E ∧ E ≠ F ∧ D ≠ F}) → 
    ∑ (M, N : ℕ), (M * N) / (6.choose 3 * (6-3).choose 3) = 143745 :=
by 
  sorry

end expected_value_product_l650_650492


namespace number_of_divisors_of_prime_factorization_l650_650394

theorem number_of_divisors_of_prime_factorization {p q : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) (n : ℕ) :
  Nat.divisors_count (p^n * q^7) = 56 → n = 6 := 
by 
  sorry

end number_of_divisors_of_prime_factorization_l650_650394


namespace largest_prime_factor_of_1729_l650_650562

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650562


namespace largest_prime_factor_1729_l650_650614

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650614


namespace concurrency_or_parallel_of_concyclic_l650_650880

variables {A B C D E F G H : Type}
variables [cyclic_quadrilateral ABCD]
variables [circumcenter_triangles G H BCE ADF]
variables {points_concyclic : (Point A) ∧ (Point B) ∧ (Point E) ∧ (Point F) }

theorem concurrency_or_parallel_of_concyclic:
  (¬parallel AD BC) → (lie_on E CD) → (lie_on F CD) → 
  (concurrent AB CD GH ∨ parallel AB CD ∨ parallel AB GH ∨ parallel CD GH) ↔ points_concyclic :=
sorry

end concurrency_or_parallel_of_concyclic_l650_650880


namespace hard_candy_food_coloring_l650_650097

theorem hard_candy_food_coloring
  (lollipop_coloring : ℕ) (hard_candy_coloring : ℕ)
  (num_lollipops : ℕ) (num_hardcandies : ℕ)
  (total_coloring : ℕ)
  (H1 : lollipop_coloring = 8)
  (H2 : num_lollipops = 150)
  (H3 : num_hardcandies = 20)
  (H4 : total_coloring = 1800) :
  (20 * hard_candy_coloring + 150 * lollipop_coloring = total_coloring) → 
  hard_candy_coloring = 30 :=
by
  sorry

end hard_candy_food_coloring_l650_650097


namespace rebecca_haircut_charge_l650_650480

-- Define the conditions
variable (H : ℕ) -- Charge for a haircut
def perm_charge : ℕ := 40
def dye_charge : ℕ := 60
def dye_cost : ℕ := 10
def haircuts_today : ℕ := 4
def perms_today : ℕ := 1
def dye_jobs_today : ℕ := 2
def tips_today : ℕ := 50
def total_amount_end_day : ℕ := 310

-- State the proof problem
theorem rebecca_haircut_charge :
  4 * H + perms_today * perm_charge + dye_jobs_today * dye_charge + tips_today - dye_jobs_today * dye_cost = total_amount_end_day →
  H = 30 :=
by
  sorry

end rebecca_haircut_charge_l650_650480


namespace probability_of_stopping_after_700_l650_650513

/-- The bag contains 800 marbles colored in 100 colors, with 8 marbles per color. -/
def marbles_in_bag := 800
def colors_in_bag := 100
def marbles_per_color := 8

/-- Anna draws marbles without replacement until she has 8 marbles of the same color and stops. -/
def draws_until_stop (n : Nat) := n + 1 = 8

/-- Anna has not stopped after drawing 699 marbles. -/
def not_stopped_after_699 := ∀ (c : Fin.colors_in_bag), (number_of_color_drawn c 699) < 8

/-- The probability that Anna stops immediately after drawing the 700th marble. -/
theorem probability_of_stopping_after_700 :
  ∀ (marbles_drawn : Nat), (marbles_drawn = 700) → (calculate_probability marbles_drawn = 99 / 101) :=
  by
  sorry

end probability_of_stopping_after_700_l650_650513


namespace min_days_to_owe_triple_l650_650162

theorem min_days_to_owe_triple (principal : ℝ) (rate : ℝ) (triple : ℝ) : ℕ :=
  let daily_interest := principal * rate
  let target_amount := triple * principal
  let x := ⌈(target_amount - principal) / daily_interest⌉
  x

noncomputable def proof_min_days_to_owe_triple : ℕ :=
  have h1 : min_days_to_owe_triple 20 0.10 3 ≥ 20 := sorry
  20

end min_days_to_owe_triple_l650_650162


namespace sqrt_ratio_eq_correct_answer_l650_650219

open Real 

def ten_factorial : ℝ := 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def two_hundred_ten : ℝ := 2 * 3 * 5 * 7
def ratio : ℝ := ten_factorial / two_hundred_ten
def correct_answer : ℝ := 24 * Real.sqrt 30

theorem sqrt_ratio_eq_correct_answer :
  Real.sqrt ratio = correct_answer :=
by
  sorry

end sqrt_ratio_eq_correct_answer_l650_650219


namespace squares_of_sides_eq_squares_of_medians_l650_650959

-- Definition of the medians intersecting at point O of the triangle ABC.
variables (A B C O : Type) [inner_product_space ℝ A]
variables {a b c : A}
variables (h1 : median A B C O) (h2 : median B C A O) (h3 : median C A B O)

-- The theorem to be proven
theorem squares_of_sides_eq_squares_of_medians :
  ∥A - B∥^2 + ∥B - C∥^2 + ∥C - A∥^2 = 3 * (∥O - A∥^2 + ∥O - B∥^2 + ∥O - C∥^2) :=
sorry

end squares_of_sides_eq_squares_of_medians_l650_650959


namespace jamie_nickels_l650_650429

theorem jamie_nickels (x : ℕ) (hx : 5 * x + 10 * x + 25 * x = 1320) : x = 33 :=
sorry

end jamie_nickels_l650_650429


namespace largest_prime_factor_of_1729_l650_650629

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650629


namespace pq_length_sum_l650_650442

theorem pq_length_sum (P Q : ℝ × ℝ) (m n : ℕ) (h_rel_prime : Int.gcd m n = 1)
  (hR : (10, 7) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))
  (hP_line : 9 * P.2 = 18 * P.1)
  (hQ_line : 12 * Q.2 = 5 * Q.1)
  (hPQ_len : real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = m / n) :
  m + n = 263 := sorry

end pq_length_sum_l650_650442


namespace largest_prime_factor_of_1729_l650_650600

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650600


namespace find_k_l650_650505

theorem find_k (σ μ : ℝ) (hσ : σ = 2) (hμ : μ = 55) :
  ∃ k : ℝ, μ - k * σ > 48 ∧ k = 3 :=
by
  sorry

end find_k_l650_650505


namespace distance_QR_l650_650481

noncomputable def point :=
  ℝ × ℝ

variables (D E F Q R : point)
variables (DE EF DF : ℝ)
variables (right_triangle : Prop)
variables (tangent_Q : Prop)
variables (tangent_R : Prop)

def right_triangle_DEF : Prop :=
  right_triangle →
  dist D E = DE ∧
  dist E F = EF ∧
  dist D F = DF ∧
  angle D E F = π / 2

def circle_Q_tangent_to_DE_at_D_passes_through_F : Prop :=
  tangent_Q →
  dist (D) D = 0 ∧
  dist (F) D = dist (Q) F

def circle_R_tangent_to_EF_at_E_passes_through_F : Prop :=
  tangent_R →
  dist (E) E = 0 ∧
  dist (F) E = dist (R) F

theorem distance_QR :
  right_triangle_DEF D E F 5 12 13 →
  circle_Q_tangent_to_DE_at_D_passes_through_F D Q F →
  circle_R_tangent_to_EF_at_E_passes_through_F E R F →
  dist Q R = 13.54 :=
  sorry

end distance_QR_l650_650481


namespace main_theorem_l650_650283

variable {a b c : ℝ}

noncomputable def inequality_1 (a b c : ℝ) : Prop :=
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b)

noncomputable def inequality_2 (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ a^2 / (b * c) + b^2 / (c * a) + c^2 / (a * b)

theorem main_theorem (h : a < 0 ∧ b < 0 ∧ c < 0) :
  inequality_1 a b c ∧ inequality_2 a b c := by sorry

end main_theorem_l650_650283


namespace express_g_in_terms_of_f_l650_650296

variables {X Y : Type} [NormedAddCommGroup X] [NormedSpace ℝ X] [NormedAddCommGroup Y] [NormedSpace ℝ Y]

-- Define odd and even functions
def is_odd (g : X → Y) := ∀ x, g (- x) = - g x
def is_even (h : X → Y) := ∀ x, h (- x) = h x

-- Define f, g, h and their relationships
variables (f g h : X → Y)
variable [h1: ∀ x, f x = g x + h x]
variable [h2: is_odd g]
variable [h3: is_even h]

-- Proof statement
theorem express_g_in_terms_of_f:
  g = λ x, (f x - f (- x)) / 2 :=
by sorry

end express_g_in_terms_of_f_l650_650296


namespace expected_first_sequence_length_is_harmonic_100_l650_650131

open Real

-- Define the harmonic number
def harmonic_number (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (i + 1 : ℝ)

-- Define the expected length of the first sequence
def expected_first_sequence_length (n : ℕ) : ℝ :=
  harmonic_number n

-- Prove that the expected number of suitors in the first sequence is the 100th harmonic number
theorem expected_first_sequence_length_is_harmonic_100 :
  expected_first_sequence_length 100 = harmonic_number 100 :=
by
  sorry

end expected_first_sequence_length_is_harmonic_100_l650_650131


namespace solve_profession_arrangement_l650_650246

inductive Profession
| architect
| barista
| veterinarian
| guitarist

inductive Person
| Andrey
| Boris
| Vyacheslav
| Gennady

open Profession
open Person

structure SeatingArrangement :=
(seat1 : Person)
(seat2 : Person)
(seat3 : Person)
(seat4 : Person)

structure Assignment :=
(profession : Person → Profession)

def correct_arrangement_and_profession (sa : SeatingArrangement) (asgmt : Assignment) : Prop :=
(sa.seat1 ≠ sa.seat2 ∧ sa.seat1 ≠ sa.seat3 ∧ sa.seat1 ≠ sa.seat4 ∧
 sa.seat2 ≠ sa.seat3 ∧ sa.seat2 ≠ sa.seat4 ∧ sa.seat3 ≠ sa.seat4) ∧
((asgmt profession sa.seat1 = barista) ∧ (asgmt profession sa.seat2 = architect) ∧ 
 (asgmt profession sa.seat3 = veterinarian) ∧ (asgmt profession sa.seat4 = guitarist)) ∧
(sa.seat3 = Andrey) ∧ 
(sa.seat2 = Boris) ∧ 
(sa.seat4 = Vyacheslav)

theorem solve_profession_arrangement : 
  ∃ (sa : SeatingArrangement) (asgmt : Assignment), correct_arrangement_and_profession sa asgmt :=
sorry

end solve_profession_arrangement_l650_650246


namespace express_g_in_terms_of_f_l650_650298

variable {R : Type*} [CommRing R]

def isOdd (g : R → R) : Prop := ∀ x, g (-x) = -g x
def isEven (h : R → R) : Prop := ∀ x, h (-x) = h x
def symmetricDomain (f : R → R) : Prop := ∀ x, f (-x) = f x

theorem express_g_in_terms_of_f
  (f g h : R → R)
  (sym_f : symmetricDomain f)
  (hyp1 : ∀ x, f x = g x + h x)
  (hyp2 : isOdd g)
  (hyp3 : isEven h) :
  ∀ x, g x = (f x - f (-x)) / 2 :=
by
  sorry

end express_g_in_terms_of_f_l650_650298


namespace triangle_similarity_and_median_l650_650742

theorem triangle_similarity_and_median
  (E M D P Q B L A C: Type)
  (hEMD : ∃ (D: E) (circle: L), E extends to intersect at D on the circle)
  (hP : P lies_on (extension_of AB))
  (hParallel: ∀ (BL ED:Type) (equal_arc : ∃ arc, arc_of_equal_chords (BL ED)) (bisector: divides (arc_AC) into_two_equal_parts BL))
  (angleP: ∀ angle, ∠angle P = (arc_AL - arc_BE) / 2 = (arc_CL - arc_DL) / 2 = (arc_CD) / 2 = A)
  (angleQ: ∀ angle, ∠angle Q = (arc_CL + arc_BE) / 2 = (arc_AL + arc_LD) / 2 = (arc_AD) / 2 = C) :
  (is_similar_triangle BPQ DAC)
  (corresponding_median BE DM) :=
  sorry

end triangle_similarity_and_median_l650_650742


namespace find_angle_C_l650_650716

variable {A B C : Real} 
variable {a b c : Real}

-- Given Condition ①
def cond1 := 2 * Real.sin A - Real.sin B = 2 * Real.sin C * Real.cos B

-- Problem 1: Proving angle C
theorem find_angle_C (h : cond1) : C = Real.pi / 3 := by
  sorry

-- Problem 2: Proving the range of 2a - b given c = 2
noncomputable def range_2a_b (h : cond1) (hc : c = 2) : Real := by
  sorry

end find_angle_C_l650_650716


namespace simplify_and_evaluate_l650_650929

variable (a : ℝ)
variable (ha : a = Real.sqrt 3 - 1)

theorem simplify_and_evaluate : 
  (1 + 3 / (a - 2)) / ((a^2 + 2 * a + 1) / (a - 2)) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_l650_650929


namespace net_displacement_east_of_A_total_fuel_consumed_l650_650113

def distances : List Int := [22, -3, 4, -2, -8, -17, -2, 12, 7, -5]
def fuel_consumption_per_km : ℝ := 0.07

theorem net_displacement_east_of_A :
  List.sum distances = 8 := by
  sorry

theorem total_fuel_consumed :
  List.sum (distances.map Int.natAbs) * fuel_consumption_per_km = 5.74 := by
  sorry

end net_displacement_east_of_A_total_fuel_consumed_l650_650113


namespace expected_number_of_first_sequence_l650_650152

-- Define the concept of harmonic number
def harmonic (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), (1 : ℚ) / (i + 1)

-- Define the problem statement
theorem expected_number_of_first_sequence (n : ℕ) (h : n = 100) : harmonic n = 5.187 := by
  sorry

end expected_number_of_first_sequence_l650_650152


namespace sally_money_l650_650919

def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20

theorem sally_money : total_seashells * price_per_seashell = 54 :=
by
  sorry

end sally_money_l650_650919


namespace plot_size_is_138240_acres_l650_650036

noncomputable def area_of_plot 
  (leg1_cm : ℝ) (leg2_cm : ℝ) (scale : ℝ) (acre_per_milesq : ℝ) : ℝ :=
  let area_cm_sq := 0.5 * leg1_cm * leg2_cm
  let area_miles_sq := area_cm_sq * (scale * scale)
  in area_miles_sq * acre_per_milesq

theorem plot_size_is_138240_acres : 
  area_of_plot 8 6 3 640 = 138240 :=
by
  simp [area_of_plot]
  sorry

end plot_size_is_138240_acres_l650_650036


namespace max_distance_circle_ellipse_l650_650457

theorem max_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 10 + p.2^2 = 1}
  ∀ (P Q : ℝ × ℝ), P ∈ circle → Q ∈ ellipse → 
  dist P Q ≤ 6 * Real.sqrt 2 :=
by
  intro circle ellipse P Q hP hQ
  sorry

end max_distance_circle_ellipse_l650_650457


namespace largest_prime_factor_1729_l650_650528

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650528


namespace find_an_from_sums_l650_650852

noncomputable def geometric_sequence (a : ℕ → ℝ) (q r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_an_from_sums (a : ℕ → ℝ) (q r : ℝ) (S3 S6 : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 :  a 1 * (1 - q^3) / (1 - q) = S3) 
  (h3 : a 1 * (1 - q^6) / (1 - q) = S6) : 
  ∃ a1 q, a n = a1 * q^(n-1) := 
by
  sorry

end find_an_from_sums_l650_650852


namespace meaningful_sqrt_l650_650834

theorem meaningful_sqrt {x : ℝ} : (∃ y : ℝ, y = sqrt (x + 1)) ↔ x ≥ -1 := 
by sorry

end meaningful_sqrt_l650_650834


namespace mod_inverse_35_36_l650_650212

theorem mod_inverse_35_36 : ∃ a : ℤ, 0 ≤ a ∧ a < 36 ∧ (35 * a) % 36 = 1 :=
  ⟨35, by sorry⟩

end mod_inverse_35_36_l650_650212


namespace seating_profession_solution_l650_650261

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l650_650261


namespace compute_expression_l650_650182

theorem compute_expression : 2 * (Real.sqrt 144)^2 = 288 := by
  sorry

end compute_expression_l650_650182


namespace largest_prime_factor_of_1729_l650_650551

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l650_650551


namespace intersection_is_correct_l650_650785

-- Define sets A and B
def A := {x : ℝ | x ≤ 3}
def B := {1, 2, 3, 4}

-- Prove the intersection of A and B is {1, 2, 3}
theorem intersection_is_correct : (A ∩ B : Set ℝ) = {1, 2, 3} := by
  sorry

end intersection_is_correct_l650_650785


namespace calc_df_length_l650_650404

theorem calc_df_length
  (A B C D E F : Type)
  (angle_BAC angle_EDF : ℝ)
  (AB AC DE : ℝ)
  (area_ABC : ℝ)
  (h₁ : angle_BAC = 60)
  (h₂ : angle_EDF = 60)
  (h₃ : AB = 4)
  (h₄ : AC = 5)
  (h₅ : DE = 2)
  (h₆ : 2 * area_ABC = 10 * real.sqrt 3) :
  ∃ DF, DF = 10 := by
  sorry

end calc_df_length_l650_650404


namespace cos_2alpha_sin_2alpha_pi_over_3_tan_2alpha_l650_650771

variable (α : ℝ)

def sin_alpha (h : 0 < α ∧ α < Real.pi / 2) : Real :=
  1 / 3

theorem cos_2alpha (h : 0 < α ∧ α < Real.pi / 2) : 
  cos (2 * α) = 7 / 9 := sorry

theorem sin_2alpha_pi_over_3 (h : 0 < α ∧ α < Real.pi / 2) :
  sin (2 * α + Real.pi / 3) = (4 * Real.sqrt 2 + 7 * Real.sqrt 3) / 18 := sorry

theorem tan_2alpha (h : 0 < α ∧ α < Real.pi / 2) :
  tan (2 * α) = 4 * Real.sqrt 2 / 7 := sorry

end cos_2alpha_sin_2alpha_pi_over_3_tan_2alpha_l650_650771


namespace trig_identity_simplify_l650_650484

-- Define the problem in Lean 4
theorem trig_identity_simplify (α : Real) : (Real.sin (α - Real.pi / 2) * Real.tan (Real.pi - α)) = Real.sin α :=
by
  sorry

end trig_identity_simplify_l650_650484


namespace length_PC_in_rectangle_l650_650441

theorem length_PC_in_rectangle (PA PB PD: ℝ) (P_inside: True) 
(h1: PA = 5) (h2: PB = 7) (h3: PD = 3) : PC = Real.sqrt 65 := 
sorry

end length_PC_in_rectangle_l650_650441


namespace number_of_green_eyes_l650_650045

-- Definitions based on conditions
def total_people : Nat := 100
def blue_eyes : Nat := 19
def brown_eyes : Nat := total_people / 2
def black_eyes : Nat := total_people / 4

-- Theorem stating the main question and its answer
theorem number_of_green_eyes : 
  (total_people - (blue_eyes + brown_eyes + black_eyes)) = 6 := by
  sorry

end number_of_green_eyes_l650_650045


namespace max_excellent_boys_l650_650826

structure Person :=
  (height : ℕ)
  (weight : ℕ)

def not_worse_than (A B : Person) : Prop :=
  A.height ≥ B.height ∨ A.weight ≥ B.weight

def excellent_boy (A : Person) (others : List Person) : Prop :=
  ∀ B ∈ others, not_worse_than A B

theorem max_excellent_boys {persons : List Person} (h : persons.length = 100) :
  ∃ excellent_boys : List Person, excellent_boys.length = 100 ∧
  ∀ A ∈ excellent_boys, excellent_boy A (persons.erase A) :=
sorry

end max_excellent_boys_l650_650826


namespace inequality_holds_l650_650188

variable (x a : ℝ)

def tensor (x y : ℝ) : ℝ :=
  (1 - x) * (1 + y)

theorem inequality_holds (h : ∀ x : ℝ, tensor (x - a) (x + a) < 1) : -2 < a ∧ a < 0 := by
  sorry

end inequality_holds_l650_650188


namespace remainder_gx12_div_gx_l650_650447

-- Definition of the polynomial g(x)
def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Theorem stating the problem
theorem remainder_gx12_div_gx : ∀ x : ℂ, (g (x^12)) % (g x) = 6 := by
  sorry

end remainder_gx12_div_gx_l650_650447


namespace all_rational_l650_650734

noncomputable def expr1 : ℚ := real.sqrt (2 ^ 2)
noncomputable def expr2 : ℚ := real.cbrt (27 / 64)
noncomputable def expr3 : ℚ := real.root 4 (1 / 625)
noncomputable def expr4 : ℚ := real.cbrt 1 * real.sqrt (1 / 16)

theorem all_rational :
  (∃ (q1 q2 q3 q4 : ℚ), expr1 = q1 ∧ expr2 = q2 ∧ expr3 = q3 ∧ expr4 = q4) :=
begin
  sorry
end

end all_rational_l650_650734


namespace num_factors_180_l650_650346

theorem num_factors_180 : 
  ∃ (n : ℕ), n = 180 ∧ prime_factorization n = [(2, 2), (3, 2), (5, 1)] ∧ positive_factors n = 18 :=
by
  sorry

end num_factors_180_l650_650346


namespace expected_first_sequence_grooms_l650_650133

-- Define the harmonic number function
def harmonic (n : ℕ) : ℝ :=
  ∑ k in finset.range n, 1 / (k + 1 : ℝ)

-- Define the expected number of grooms in the first sequence
theorem expected_first_sequence_grooms :
  harmonic 100 = 5.187 :=
by
  sorry

end expected_first_sequence_grooms_l650_650133


namespace natural_number_bounds_l650_650193

theorem natural_number_bounds (n : ℕ) (h: ∃ k : ℕ, 0.5 * n ≤ k ∧ k + 99 ≤ 0.6 * n) : 
  997 ≤ n ∧ n ≤ 1010 :=
by sorry

end natural_number_bounds_l650_650193


namespace tangent_line_at_P_l650_650278

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + 3

theorem tangent_line_at_P :
  let P : ℝ × ℝ := (1, 2)
  in P = (1, f 1) ∧ (∀ x y : ℝ, y = x - 1 + 2 → x - y + 1 = 0) :=
by
  sorry

end tangent_line_at_P_l650_650278


namespace assign_teachers_to_classes_l650_650091

-- Define the given conditions as variables and constants
theorem assign_teachers_to_classes :
  (∃ ways : ℕ, ways = 36) :=
by
  sorry

end assign_teachers_to_classes_l650_650091


namespace description_of_T_l650_650865

def T : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | 
    ∃ (c : ℝ), 
      (c = 5 ∧ (p.fst + 3 = c ∧ p.snd - 2 ≤ c) ∨ 
       c = 5 ∧ (p.snd - 2 = c ∧ p.fst + 3 ≤ c) ∨ 
       c = p.fst + 3 ∧ p.snd - 2 = c ∧ p.fst ≤ 2)}

theorem description_of_T :
  T = {p : ℝ × ℝ | p = (2, 7) ∨ 
                      (p.fst = 2 ∧ p.snd ≤ 7) ∨
                      (p.snd = 7 ∧ p.fst ≤ 2) ∨
                      (p.snd = p.fst + 5 ∧ p.fst ≤ 2) } :=
by 
  sorry

end description_of_T_l650_650865


namespace area_at_stage_8_l650_650385

theorem area_at_stage_8 
  (side_length : ℕ)
  (stage : ℕ)
  (num_squares : ℕ)
  (square_area : ℕ) 
  (total_area : ℕ) 
  (h1 : side_length = 4) 
  (h2 : stage = 8) 
  (h3 : num_squares = stage) 
  (h4 : square_area = side_length * side_length) 
  (h5 : total_area = num_squares * square_area) :
  total_area = 128 :=
sorry

end area_at_stage_8_l650_650385


namespace Kyler_wins_l650_650472

variable (K : ℕ) -- Kyler's wins

/- Constants based on the problem statement -/
def Peter_wins := 5
def Peter_losses := 3
def Emma_wins := 2
def Emma_losses := 4
def Total_games := 15
def Kyler_losses := 4

/- Definition that calculates total games played -/
def total_games_played := 2 * Total_games

/- Game equation based on the total count of played games -/
def game_equation := Peter_wins + Peter_losses + Emma_wins + Emma_losses + K + Kyler_losses = total_games_played

/- Question: Calculate Kyler's wins assuming the given conditions -/
theorem Kyler_wins : K = 1 :=
by
  sorry

end Kyler_wins_l650_650472


namespace number_of_factors_180_l650_650358

theorem number_of_factors_180 : 
  let factors180 (n : ℕ) := 180 
  in factors180 180 = (2 + 1) * (2 + 1) * (1 + 1) => factors180 180 = 18 := 
by
  sorry

end number_of_factors_180_l650_650358


namespace _l650_650663

variables (circle : Type) [metric_space circle] (A B C D : circle)
variables (ray_AB : line through (A, B)) (ray_AC : line through (A, C))
variables (arc_BC arc_BDC : circle → ℝ)

def angle_BAC (A B C : circle) : ℝ := sorry -- the specific definition of the angle at A

lemma angle_measure_external_theorem 
  (h1 : point_outside_circle A) 
  (h2 : ray_intersect_circle ray_AB ray_AC B C)
  : angle_BAC A B C = 1/2 * (arc_measure arc_BDC - arc_measure arc_BC) :=
sorry

end _l650_650663


namespace largest_prime_factor_1729_l650_650615

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650615


namespace sum_100th_bracket_is_1891_l650_650198

def sequence (n : ℕ) : ℕ := 2 * n + 1

-- Representation of cyclical grouping as described in the problem
def group_sequence (n : ℕ) : list (list ℕ) :=
  let seq := list.map sequence (list.range n)
  let group_1 := list.take 1 seq
  let group_2 := list.take 2 (list.drop 1 seq)
  let group_3 := list.take 3 (list.drop 3 seq)
  let group_4 := list.take 4 (list.drop 6 seq)
  group_1 :: group_2 :: group_3 :: group_4 :: group_sequence (n - 10)

def sum_of_100th_bracket : ℕ :=
  let nth_group := 24 * 4 -- 100th bracket corresponds to the 4th bracket of 25th cycle
  list.sum (group_sequence nth_group 3)

theorem sum_100th_bracket_is_1891 : sum_of_100th_bracket = 1891 :=
  by sorry

end sum_100th_bracket_is_1891_l650_650198


namespace avg_salary_supervisors_l650_650836

-- Definitions based on the conditions of the problem
def total_workers : Nat := 48
def supervisors : Nat := 6
def laborers : Nat := 42
def avg_salary_total : Real := 1250
def avg_salary_laborers : Real := 950

-- Given the above conditions, we need to prove the average salary of the supervisors.
theorem avg_salary_supervisors :
  (supervisors * (supervisors * total_workers * avg_salary_total - laborers * avg_salary_laborers) / supervisors) = 3350 :=
by
  sorry

end avg_salary_supervisors_l650_650836


namespace largest_digit_divisible_by_6_l650_650992

theorem largest_digit_divisible_by_6 :
  ∃ M, M ∈ {0, 2, 4, 6, 8} ∧ (54320 + M) % 2 = 0 ∧ (5 + 4 + 3 + 2 + M) % 3 = 0 ∧
    (∀ m ∈ {0, 2, 4, 6, 8}, (54320 + m) % 2 = 0 ∧ (5 + 4 + 3 + 2 + m) % 3 = 0 → m ≤ M) :=
by
  sorry

end largest_digit_divisible_by_6_l650_650992


namespace range_of_a_l650_650377

noncomputable def f (a : ℝ) (x : ℝ) := log a (2 - a * x)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 0 1, f a x ∈ set.univ) ∧ 
  (∀ x y ∈ set.Icc 0 1, x < y → f a x > f a y) →
  1 < a ∧ a < 2 :=
by
  assume h,
  sorry

end range_of_a_l650_650377


namespace stream_speed_l650_650095

theorem stream_speed (v : ℝ) : (24 + v) = 168 / 6 → v = 4 :=
by
  intro h
  sorry

end stream_speed_l650_650095


namespace num_bases_ending_in_1_l650_650760

theorem num_bases_ending_in_1 : 
  (∃ bases : Finset ℕ, 
  ∀ b ∈ bases, 3 ≤ b ∧ b ≤ 10 ∧ (625 % b = 1) ∧ bases.card = 4) :=
sorry

end num_bases_ending_in_1_l650_650760


namespace sally_money_l650_650921

def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20

theorem sally_money : total_seashells * price_per_seashell = 54 :=
by
  sorry

end sally_money_l650_650921


namespace largest_prime_factor_1729_l650_650589

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650589


namespace proposition_not_true_at_9_l650_650153

variable {P : ℕ → Prop}

theorem proposition_not_true_at_9 (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1)) (h10 : ¬P 10) : ¬P 9 :=
by
  sorry

end proposition_not_true_at_9_l650_650153


namespace largest_prime_factor_of_1729_l650_650630

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650630


namespace largest_prime_factor_1729_l650_650616

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650616


namespace largest_prime_factor_of_1729_l650_650635

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650635


namespace man_son_work_together_l650_650114

theorem man_son_work_together (man_days : ℝ) (son_days : ℝ) (combined_days : ℝ) :
  man_days = 4 → son_days = 12 → (1 / man_days + 1 / son_days) = 1 / combined_days → combined_days = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end man_son_work_together_l650_650114


namespace log_of_fraction_l650_650757

theorem log_of_fraction :
  let a := (0.54 : ℝ)
  let approximations := 
    (log_3_2 : ℝ) ≈ 0.6309 ∧ (log_3_5 : ℝ) ≈ 1.4647

  (log_3 (a)) ≈ -0.5603 :=
by
  let a: ℝ := 0.54
  have fraction_def : a = (3^3 : ℝ) / (2 * 5^2 : ℝ) := by ring
  have log_3_3 : log_3 (3^3) = 3 := by simp [log, log_base]
  have log_fraction : log_3 a = log_3 (3^3) - (log_3 2 + 2 * log_3 5) := by sorry
  have h1 : log_3 2 ≈ 0.6309 := by sorry
  have h2 : log_3 5 ≈ 1.4647 := by sorry
  have calc : log_3 a ≈ log_3 (3^3) - (log_3 2 + 2 * log_3 5) := by sorry
  calc
    log_3 a
        ≈ 3 - (0.6309 + 2 * 1.4647) := by sorry
    ... ≈ -0.5603 := by ring


end log_of_fraction_l650_650757


namespace angle_y_in_triangle_l650_650729

theorem angle_y_in_triangle (y : ℝ) (h1 : ∀ a b c : ℝ, a + b + c = 180) (h2 : 3 * y + y + 40 = 180) : y = 35 :=
sorry

end angle_y_in_triangle_l650_650729


namespace repeating_sequence_length_1_over_221_l650_650496

theorem repeating_sequence_length_1_over_221 : ∃ n : ℕ, (10 ^ n ≡ 1 [MOD 221]) ∧ (∀ m : ℕ, (10 ^ m ≡ 1 [MOD 221]) → (n ≤ m)) ∧ n = 48 :=
by
  sorry

end repeating_sequence_length_1_over_221_l650_650496


namespace greatest_possible_average_speed_l650_650709

-- Palindrome checking function
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

-- Variables and constants
constant initial_odometer : ℕ := 13831
constant max_speed : ℕ := 80
constant trip_duration : ℕ := 5

-- Definitions based on the conditions
def final_odometer (d : ℕ) : ℕ := initial_odometer + d

-- Problem statement
theorem greatest_possible_average_speed :
  ∃ (final_dist : ℕ),
    is_palindrome (final_odometer final_dist) ∧
    final_dist ≤ max_speed * trip_duration ∧
    final_dist / trip_duration = 62 := sorry

end greatest_possible_average_speed_l650_650709


namespace expected_first_sequence_grooms_l650_650137

-- Define the harmonic number function
def harmonic (n : ℕ) : ℝ :=
  ∑ k in finset.range n, 1 / (k + 1 : ℝ)

-- Define the expected number of grooms in the first sequence
theorem expected_first_sequence_grooms :
  harmonic 100 = 5.187 :=
by
  sorry

end expected_first_sequence_grooms_l650_650137


namespace num_100_digit_even_numbers_l650_650335

theorem num_100_digit_even_numbers : 
  let digit_set := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let valid_number (digits : list ℕ) := 
    digits.length = 100 ∧ digits.head ∈ {1, 3} ∧ 
    digits.last ∈ {0} ∧ 
    ∀ d ∈ digits.tail.init, d ∈ digit_set
  (∃ (m : ℕ), valid_number (m.digits 10)) = 2 * 3^98 := 
sorry

end num_100_digit_even_numbers_l650_650335


namespace diagonals_sum_pentagon_inscribed_in_circle_l650_650438

theorem diagonals_sum_pentagon_inscribed_in_circle
  (FG HI GH IJ FJ : ℝ)
  (h1 : FG = 4)
  (h2 : HI = 4)
  (h3 : GH = 11)
  (h4 : IJ = 11)
  (h5 : FJ = 15) :
  3 * FJ + (FJ^2 - 121) / 4 + (FJ^2 - 16) / 11 = 80 := by {
  sorry
}

end diagonals_sum_pentagon_inscribed_in_circle_l650_650438


namespace AIAS_seating_arrangements_l650_650489

theorem AIAS_seating_arrangements : 
  let Martians := 4
  let Venusians := 6
  let Earthlings := 5
  ∃ N : ℕ,
    N * (factorial Martians * factorial Venusians * factorial Earthlings) = 1 * (factorial 4 * factorial 6 * factorial 5) :=
by { sorry }

end AIAS_seating_arrangements_l650_650489


namespace program_size_l650_650987

noncomputable def download_speed_MB_per_sec : ℝ := 50
noncomputable def download_time_hours : ℝ := 2
noncomputable def MB_per_GB : ℝ := 1024

theorem program_size : 
  let download_speed_GB_per_hour := (download_speed_MB_per_sec * 60 * 60) / MB_per_GB in
  let total_download_GB := download_speed_GB_per_hour * download_time_hours in
  total_download_GB = 351.5625 :=
by
  -- This definition is derived from the conditions and not solution steps.
  sorry

end program_size_l650_650987


namespace average_of_consecutive_numbers_l650_650016

-- Define the 7 consecutive numbers and their properties
variables (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ) (f : ℝ) (g : ℝ)

-- Conditions given in the problem
def consecutive_numbers (a b c d e f g : ℝ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 ∧ g = a + 6

def percent_relationship (a g : ℝ) : Prop :=
  g = 1.5 * a

-- The proof problem
theorem average_of_consecutive_numbers (a b c d e f g : ℝ)
  (h1 : consecutive_numbers a b c d e f g)
  (h2 : percent_relationship a g) :
  (a + b + c + d + e + f + g) / 7 = 15 :=
by {
  sorry -- Proof goes here
}

-- To ensure it passes the type checker but without providing the actual proof, we use sorry.

end average_of_consecutive_numbers_l650_650016


namespace coinPaymentDifference_l650_650856

noncomputable def minCoins (amount : ℕ) : ℕ :=
if h : amount = 50 then 1 else 0

noncomputable def maxCoins (amount : ℕ) : ℕ :=
if h : amount = 50 then amount / 5 else 0

theorem coinPaymentDifference : 
  let amount := 50
  ∀ (denom1 denom2 denom3 : ℕ), denom1 = 5 → denom2 = 20 → denom3 = 50 →
  (maxCoins amount - minCoins amount) = 9 :=
by
  intros _ _ _ h1 h2 h3
  have h4 : amount = 50 := rfl
  rw [maxCoins, minCoins, h4, h4]
  simp [h1, h2, h3]
  norm_num
  sorry

end coinPaymentDifference_l650_650856


namespace sin_eq_solutions_l650_650371

theorem sin_eq_solutions :
  (∃ count : ℕ, 
    count = 4007 ∧ 
    (∀ (x : ℝ), 
      0 ≤ x ∧ x ≤ 2 * Real.pi → 
      (∃ (k1 k2 : ℤ), 
        x = -2 * k1 * Real.pi ∨ 
        x = 2 * Real.pi ∨ 
        x = (2 * k2 + 1) * Real.pi / 4005)
    )) :=
sorry

end sin_eq_solutions_l650_650371


namespace angle_between_skew_lines_l650_650307

variables (α β l m n : Type) [linear_ordered_field α] [linear_ordered_field β] [linear_ordered_field l] [linear_ordered_field m] [linear_ordered_field n]

-- Lean 4 statement
theorem angle_between_skew_lines (dihedral_angle : ∀ (α l β: Type), α) 
  (skew_lines : ∀ (m n : Type), float) 
  (m_perpendicular : ∀ (m α : Type), Prop) 
  (n_perpendicular : ∀ (n β : Type), Prop) 
  (h1 : dihedral_angle α l β = 60) 
  (h2 : skew_lines m n = 60) 
  (h3 : m_perpendicular m α) 
  (h4 : n_perpendicular n β) :
  skew_lines m n = 60 :=
sorry

end angle_between_skew_lines_l650_650307


namespace expected_value_first_outstanding_sequence_l650_650138

-- Define indicator variables and the harmonic number
noncomputable def I (k : ℕ) := 1 / k
noncomputable def H (n : ℕ) := ∑ i in Finset.range (n + 1), I (i + 1)

-- Theorem: Expected value of the first sequence of outstanding grooms
theorem expected_value_first_outstanding_sequence : 
  (H 100) = 5.187 := 
sorry

end expected_value_first_outstanding_sequence_l650_650138


namespace proof_problem_l650_650784

noncomputable def lines_planes_conditions (l m : Line) (alpha beta : Plane) : Prop :=
  (l ⊥ alpha) ∧ (m ∥ beta) ∧ (alpha ∥ beta → l ⊥ m)

theorem proof_problem 
  (l m : Line) 
  (alpha beta : Plane) 
  (h1 : l ⊥ alpha) 
  (h2 : m ∥ beta) 
  (h3 : alpha ∥ beta) : l ⊥ m := 
by 
  sorry

end proof_problem_l650_650784


namespace cot_diff_l650_650424

variable {α β γ : Type}
variables {A B C D P : α}
variables {x y : ℝ}

-- Basic setup for the triangle and median condition
def triangle (A B C : Type) := true
def median (A D : Type) := true
def angle (D P : Type) := true
def makes_angle (AD BC : Type) (θ : ℝ) := true

-- Definitions using the conditions from the problem
axiom tr : triangle A B C
axiom med : median A D
axiom ang_cond : makes_angle (angle A D P) (angle B C) (30 * π / 180)

-- Definitions of x and y as given in the solution steps
noncomputable def BD := x
noncomputable def CD := x
noncomputable def BP := y

-- Definition of cot B and cot C as functions of x and y
noncomputable def cotB : ℝ := -y / (x + y)
noncomputable def cotC : ℝ := (2 * x + y) / (x + y)

-- Final theorem statement
theorem cot_diff : |cotB - cotC| = 3 := by
  sorry

end cot_diff_l650_650424


namespace probability_of_red_ball_probability_of_non_red_ball_l650_650410

def total_balls : ℕ := 3 + 5 + 7
def red_balls : ℕ := 3
def non_red_balls : ℕ := 5 + 7

theorem probability_of_red_ball (h : total_balls = 15) : (red_balls : ℚ) / total_balls = 1 / 5 :=
by
  rw [red_balls, total_balls, h]
  norm_num

theorem probability_of_non_red_ball (h : total_balls = 15) : (non_red_balls : ℚ) / total_balls = 4 / 5 :=
by
  rw [non_red_balls, total_balls, h]
  norm_num

end probability_of_red_ball_probability_of_non_red_ball_l650_650410


namespace max_rectangle_area_l650_650391

theorem max_rectangle_area (L : ℝ) (hL : L = 32) : 
  ∃ S : ℝ, S = 64 ∧ ∀ (x : ℝ), x * (L / 2 - x) ≤ S :=
by
  let K := L / 2
  let S := K ^ 2
  use S
  split
  norm_num; assumption
  intro x
  have A : x * (K - x) ≤ (K/2) * (K/2) := sorry
  rw [hL] at ⊢ A
  norm_num at A
  exact A

end max_rectangle_area_l650_650391


namespace largest_prime_factor_1729_l650_650588

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650588


namespace general_term_sum_formula_l650_650781

-- Conditions for the sequence
variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a2_eq_5 : a 2 = 5
axiom S4_eq_28 : S 4 = 28

-- The sequence is an arithmetic sequence
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- Statement 1: Proof that a_n = 4n - 3
theorem general_term (n : ℕ) : a n = 4 * n - 3 :=
by
  sorry

-- Statement 2: Proof that S_n = 2n^2 - n
theorem sum_formula (n : ℕ) : S n = 2 * n^2 - n :=
by
  sorry

end general_term_sum_formula_l650_650781


namespace true_proposition_is_C_l650_650289

open Real

def p : Prop := ∀ x ∈ Ioo 0 (π / 2), sin x - x < 0
def q : Prop := ∃ x₀ ∈ Ioi 0, 2 ^ x₀ = 1 / 2

theorem true_proposition_is_C : p ∧ ¬q := by
  sorry

end true_proposition_is_C_l650_650289


namespace final_result_is_110_l650_650156

theorem final_result_is_110 (x : ℕ) (h1 : x = 155) : (x * 2 - 200) = 110 :=
by
  -- placeholder for the solution proof
  sorry

end final_result_is_110_l650_650156


namespace sudoku_fourth_column_number_l650_650021

theorem sudoku_fourth_column_number :
  let grid : Matrix (Fin 9) (Fin 9) ℕ := 
      ![[_,_,_,3,_,_,_,_,_],  -- row 1
        [_,_,_,2,_,_,_,_,_],  -- row 2
        [_,_,_,_,_,_,_,_,_],  -- row 3
        [_,_,_,4,_,_,_,_,_],  -- row 4
        [_,_,_,_,_,_,_,_,_],  -- row 5
        [_,_,_,_,_,_,_,_,_],  -- row 6
        [_,_,_,_,_,_,_,_,_],  -- row 7
        [_,_,_,5,_,_,_,_,_],  -- row 8
        [_,_,_,1,_,_,_,_,_]]  -- row 9
  in
  ∃ (new_grid : Matrix (Fin 9) (Fin 9) ℕ), 
    (∀ i j : Fin 9, new_grid i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (∀ i, ∃! j, new_grid i j = (grid i j)) ∧
    (∀ i, ∃! j, new_grid j i = (grid i j)) ∧
    (∀ subgrid_ij : Fin 3 × Fin 3, 
       ∀ i j : Fin 3, 
         ∃! k l : Fin 3, 
           new_grid (subgrid_ij.1 * 3 + i) (subgrid_ij.2 * 3 + j) = 
             new_grid (subgrid_ij.1 * 3 + k) (subgrid_ij.2 * 3 + l)) ∧
    (new_grid (Fin.ofNat 0) (Fin.ofNat 3 ) *
     100_000_000 +
     new_grid (Fin.ofNat 1) (Fin.ofNat 3 ) *
     10_000_000 +
     new_grid (Fin.ofNat 2) (Fin.ofNat 3 ) *
     1_000_000 +
     new_grid (Fin.ofNat 3) (Fin.ofNat 3 ) *
     100_000 +
     new_grid (Fin.ofNat 4) (Fin.ofNat 3 ) *
     10_000 +
     new_grid (Fin.ofNat 5) (Fin.ofNat 3 ) *
     1_000 +
     new_grid (Fin.ofNat 6) (Fin.ofNat 3 ) *
     100 +
     new_grid (Fin.ofNat 7) (Fin.ofNat 3 ) *
     10 +
     new_grid (Fin.ofNat 8) (Fin.ofNat 3 )) =
    327468951 :=
sorry

end sudoku_fourth_column_number_l650_650021


namespace sum_of_distinct_real_values_l650_650733

theorem sum_of_distinct_real_values :
  ∀ (x : ℝ),
    ((nat.iterate (λ y, abs y + x) 2016 (abs x + x) = 1) ∧ (nat.iterate (λ y, - abs y + x) 2016 (- abs x + x) = 1)) →
    x = 1 / 2017 ∨ x = -1 →
    (1/2017 + (-1) = -2016/2017) :=
by
  sorry

end sum_of_distinct_real_values_l650_650733


namespace seating_arrangement_l650_650243

-- Definitions of friends and their professions
inductive Friend
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq

-- Seating condition definitions
def seated_right (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = a ∧ seating (i + 1) = b)

def seated_between (a b c : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = b ∧ seating (i - 1) = a ∧ seating (i + 1) = c)

def neighbors (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, (seating i = a ∧ seating (i - 1) = b) ∨ (seating i = a ∧ seating (i + 1) = b))

-- The proposition to be proved
theorem seating_arrangement
  (seating : ℕ → Friend)
  (profession : Friend → Profession)
  (h1 : ∃ i, seating_between Profession.Veterinarian Profession.Architect Profession.Guitarist seating)
  (h2 : seated_right Profession.Barista Profession.Boris seating)
  (h3 : ∃ i, seating i = Friend.Vyacheslav ∧ (∃ j, seating j = Friend.Andrey ∧ j < i) ∧ (∃ k, seating k = Friend.Boris ∧ k < i))
  (h4 : ∀ a, seating a = Friend.Andrey → neighbors Friend.Andrey seating)
  (h5 : ∀ a, seating a = Profession.Guitarist → ∀ b, seating (b - 1) = Profession.Barista ∨ seating (b + 1) = Profession.Barista → false) :
  (profession Friend.Gennady = Profession.Barista ∧
   profession Friend.Boris = Profession.Architect ∧
   profession Friend.Andrey = Profession.Veterinarian ∧
   profession Friend.Vyacheslav = Profession.Guitarist) :=
begin
  sorry
end

end seating_arrangement_l650_650243


namespace not_product_of_two_primes_l650_650452

theorem not_product_of_two_primes (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : ∃ n : ℕ, a^3 + b^3 = n^2) :
  ¬ (∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ a + b = p * q) :=
by
  sorry

end not_product_of_two_primes_l650_650452


namespace final_professions_correct_l650_650268

-- Define the people and their positions
inductive Person
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq, Inhabited

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq, Inhabited

open Person Profession

-- Given positions in seats
def seatingArrangement : Person → Nat := 
λ p, 
  match p with
  | Andrey    => 3
  | Boris     => 2
  | Vyacheslav => 4
  | Gennady   => 1

-- Given professions
def profession : Person → Profession := 
λ p, 
  match p with
  | Andrey    => Veterinarian
  | Boris     => Architect
  | Vyacheslav => Guitarist
  | Gennady   => Barista

-- Define the conditions
def condition1 : Prop :=
  (seatingArrangement Andrey = 3) ∧ 
  (seatingArrangement Boris = 2) ∧
  (seatingArrangement Vyacheslav = 4) ∧ 
  (seatingArrangement Gennady = 1)

def condition2 : Prop := 
  seatingArrangement (Person.Barista) + 1 = seatingArrangement Boris

def condition3 : Prop := 
  seatingArrangement Vyacheslav > seatingArrangement Andrey ∧ 
  seatingArrangement Vyacheslav > seatingArrangement Boris

def condition4 : Prop := 
  seatingArrangement (Person.Veterinarian) ≠ 1 ∧ seatingArrangement (Person.Veterinarian) ≠ 4

def condition5 : Prop := 
  (seatingArrangement (Person.Barista) ≠ seatingArrangement (Person.Guitarist) + 1 ∧
  seatingArrangement (Person.Guitarist) ≠ seatingArrangement (Person.Barista) + 1)

-- Prove the final professions
theorem final_professions_correct :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  profession Gennady = Barista ∧
  profession Boris = Architect ∧
  profession Andrey = Veterinarian ∧
  profession Vyacheslav = Guitarist :=
by sorry

end final_professions_correct_l650_650268


namespace jacob_younger_than_michael_l650_650428

theorem jacob_younger_than_michael :
  ∃ (J M : ℕ), (J + 4 = 5) ∧ (M + 11 = 2 * (J + 11)) ∧ (M - J = 12) :=
by
  existsi 1
  existsi 13
  split
  { sorry }
  split
  { sorry }
  { sorry }

end jacob_younger_than_michael_l650_650428


namespace count100DigitEvenNumbers_is_correct_l650_650339

noncomputable def count100DigitEvenNumbers : ℕ :=
  let validDigits : Finset ℕ := {0, 1, 3}
  let firstDigitChoices : ℕ := 2  -- Only 1 or 3
  let middleDigitsChoices : ℕ := 3 ^ 98  -- 3 choices for each of the 98 middle positions
  let lastDigitChoices : ℕ := 1  -- Only 0 (even number requirement)
  firstDigitChoices * middleDigitsChoices * lastDigitChoices

theorem count100DigitEvenNumbers_is_correct :
  count100DigitEvenNumbers = 2 * 3 ^ 98 := by
  sorry

end count100DigitEvenNumbers_is_correct_l650_650339


namespace return_speed_is_48_l650_650098

variable (d r : ℕ)
variable (t_1 t_2 : ℚ)

-- Given conditions
def distance_each_way : Prop := d = 120
def time_to_travel_A_to_B : Prop := t_1 = d / 80
def time_to_travel_B_to_A : Prop := t_2 = d / r
def average_speed_round_trip : Prop := 60 * (t_1 + t_2) = 2 * d

-- Statement to prove
theorem return_speed_is_48 :
  distance_each_way d ∧
  time_to_travel_A_to_B d t_1 ∧
  time_to_travel_B_to_A d r t_2 ∧
  average_speed_round_trip d t_1 t_2 →
  r = 48 :=
by
  intros
  sorry

end return_speed_is_48_l650_650098


namespace largest_prime_factor_1729_l650_650586

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650586


namespace largest_prime_factor_1729_l650_650580

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l650_650580


namespace largest_prime_factor_1729_l650_650544

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650544


namespace exists_z_with_mod_one_and_fz_mod_sqrt_n_minus_two_l650_650434

open Complex BigOperators

noncomputable def A : Set ℕ := sorry

def f (x : ℂ) : ℂ := ∑ a in A, x ^ a

theorem exists_z_with_mod_one_and_fz_mod_sqrt_n_minus_two
  (n : ℕ) (hn : 2 ≤ n) (hA : ∀ a ∈ A, 0 < a)
  (h_card : Set.card A = n) :
    ∃ z : ℂ, abs z = 1 ∧ abs (f z) = Real.sqrt (n - 2) :=
sorry

end exists_z_with_mod_one_and_fz_mod_sqrt_n_minus_two_l650_650434


namespace sum_of_coefficients_l650_650175

theorem sum_of_coefficients :
  (4 * (2 * x^8 + 3 * x^5 - 5) + 6 * (x^6 - 5 * x^3 + 4)).coefficients.sum = 0 :=
sorry

end sum_of_coefficients_l650_650175


namespace factors_of_180_l650_650366

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l650_650366


namespace eigenvectors_not_orthogonal_l650_650207

variables {m : ℝ}

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]

def λ1 : ℝ := -1
def λ2 : ℝ := 4

def x1 : Vector (Fin 2) ℝ := ![1, -1]
def x2 : Vector (Fin 2) ℝ := ![1, (3:ℝ) / 2]

theorem eigenvectors_not_orthogonal : 
  dot_product x1 x2 ≠ 0 := 
sorry

end eigenvectors_not_orthogonal_l650_650207


namespace sin_product_identity_l650_650223

noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)
noncomputable def sin_30_deg := Real.sin (30 * Real.pi / 180)
noncomputable def sin_75_deg := Real.sin (75 * Real.pi / 180)

theorem sin_product_identity :
  sin_15_deg * sin_30_deg * sin_75_deg = 1 / 8 :=
by
  sorry

end sin_product_identity_l650_650223


namespace annual_average_growth_rate_estimated_output_value_2006_l650_650911

-- First problem: Prove the annual average growth rate from 2003 to 2005
theorem annual_average_growth_rate (x : ℝ) (h : 6.4 * (1 + x)^2 = 10) : 
  x = 1/4 :=
by
  sorry

-- Second problem: Prove the estimated output value for 2006 given the annual growth rate
theorem estimated_output_value_2006 (x : ℝ) (output_2005 : ℝ) (h_growth : x = 1/4) (h_2005 : output_2005 = 10) : 
  output_2005 * (1 + x) = 12.5 :=
by 
  sorry

end annual_average_growth_rate_estimated_output_value_2006_l650_650911


namespace sum_largest_smallest_prime_factors_of_546_l650_650074

theorem sum_largest_smallest_prime_factors_of_546 :
  let primes := [2, 3, 7, 13]
  (2 + 13 = 15) ∧ (4 * 6 * 14 * 26 = 8736) :=
by
  let primes := [2, 3, 7, 13]
  have : 2 + 13 = 15 := rfl
  have : 4 * 6 * 14 * 26 = 8736 := rfl
  exact ⟨this, this⟩

end sum_largest_smallest_prime_factors_of_546_l650_650074


namespace largest_prime_factor_of_1729_l650_650559

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l650_650559


namespace largest_prime_factor_of_1729_l650_650631

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l650_650631


namespace exists_divisor_in_interval_l650_650448

theorem exists_divisor_in_interval
  (n : ℕ) (k : ℕ)
  (h1 : n % 2 = 1) 
  (h2 : 0 < k) 
  (h3 : finset.card {d : ℕ | d ∣ (2 * n) ∧ d ≤ k} % 2 = 1) :
  ∃ d : ℕ, d ∣ (2 * n) ∧ k < d ∧ d ≤ 2 * k := 
sorry

end exists_divisor_in_interval_l650_650448


namespace expression_evaluation_l650_650512

theorem expression_evaluation :
  \(\frac {\sin 20° \sqrt {1+\cos 40° }}{\cos 50° } = \frac {\sqrt{2}}{2}\) :=
sorry

end expression_evaluation_l650_650512


namespace symmetry_of_circle_l650_650946

noncomputable def circle_symmetry : Prop :=
  let circle_equation := ∀ x y, (x - 1)^2 + (y - 1)^2 = 1
  let symmetry_line := ∀ x y, y = 5 * x - 4
  ∃ (x y : ℝ), circle_equation x y → symmetry_line x y → (x - 1)^2 + (y - 1)^2 = 1

-- Now state the theorem
theorem symmetry_of_circle (x y : ℝ) : circle_symmetry := sorry

end symmetry_of_circle_l650_650946


namespace pool_length_l650_650939

theorem pool_length (r : ℕ) (t : ℕ) (w : ℕ) (d : ℕ) (L : ℕ) 
  (H1 : r = 60)
  (H2 : t = 2000)
  (H3 : w = 80)
  (H4 : d = 10)
  (H5 : L = (r * t) / (w * d)) : L = 150 :=
by
  rw [H1, H2, H3, H4] at H5
  exact H5


end pool_length_l650_650939


namespace num_bases_for_625_ending_in_1_l650_650763

theorem num_bases_for_625_ending_in_1 :
  (Finset.card (Finset.filter (λ b : ℕ, 624 % b = 0) (Finset.Icc 3 10))) = 4 :=
by
  sorry

end num_bases_for_625_ending_in_1_l650_650763


namespace domain_of_function_l650_650523

theorem domain_of_function :
  (∀ x, x ≠ 7 ∧ x^2 - 49 ≥ 0 → x ∈ (-∞, -7] ∨ x ∈ (7, ∞)) :=
by
  intros x hx
  have h1 : x ≠ 7 := hx.1
  have h2 : x^2 - 49 ≥ 0 := hx.2
  -- Proof body not required
  sorry

end domain_of_function_l650_650523


namespace solve_profession_arrangement_l650_650250

inductive Profession
| architect
| barista
| veterinarian
| guitarist

inductive Person
| Andrey
| Boris
| Vyacheslav
| Gennady

open Profession
open Person

structure SeatingArrangement :=
(seat1 : Person)
(seat2 : Person)
(seat3 : Person)
(seat4 : Person)

structure Assignment :=
(profession : Person → Profession)

def correct_arrangement_and_profession (sa : SeatingArrangement) (asgmt : Assignment) : Prop :=
(sa.seat1 ≠ sa.seat2 ∧ sa.seat1 ≠ sa.seat3 ∧ sa.seat1 ≠ sa.seat4 ∧
 sa.seat2 ≠ sa.seat3 ∧ sa.seat2 ≠ sa.seat4 ∧ sa.seat3 ≠ sa.seat4) ∧
((asgmt profession sa.seat1 = barista) ∧ (asgmt profession sa.seat2 = architect) ∧ 
 (asgmt profession sa.seat3 = veterinarian) ∧ (asgmt profession sa.seat4 = guitarist)) ∧
(sa.seat3 = Andrey) ∧ 
(sa.seat2 = Boris) ∧ 
(sa.seat4 = Vyacheslav)

theorem solve_profession_arrangement : 
  ∃ (sa : SeatingArrangement) (asgmt : Assignment), correct_arrangement_and_profession sa asgmt :=
sorry

end solve_profession_arrangement_l650_650250


namespace green_eyes_count_l650_650048

theorem green_eyes_count (total_people : ℕ) (blue_eyes : ℕ) (brown_eyes : ℕ) (black_eyes : ℕ) (green_eyes : ℕ) :
  total_people = 100 → 
  blue_eyes = 19 → 
  brown_eyes = total_people / 2 → 
  black_eyes = total_people / 4 → 
  green_eyes = total_people - (blue_eyes + brown_eyes + black_eyes) → 
  green_eyes = 6 := 
by 
  intros h_total h_blue h_brown h_black h_green 
  rw [h_total, h_blue, h_brown, h_black] at h_green 
  exact h_green.symm

end green_eyes_count_l650_650048


namespace expression_for_f_when_x_lt_0_l650_650782

variable {ℝ : Type} [LinearOrderedField ℝ]

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 4 * x else if x = 0 then 0 else -x^2 + 4 * x

theorem expression_for_f_when_x_lt_0 (x : ℝ) (h : x < 0) (odd_f : ∀ x, f(-x) = - f(x)) :
  f(x) = -x^2 + 4 * x :=
by
  sorry

end expression_for_f_when_x_lt_0_l650_650782


namespace strictly_increasing_interval_l650_650308

def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem strictly_increasing_interval (φ : ℝ) (k : ℤ)
  (h1 : ∀ x : ℝ, f x φ ≤ |f (π / 4) φ|)
  (h2 : f (π / 2) φ > f π φ) :
  ∃ I : set ℝ, ∀ x : ℝ, x ∈ I ↔ (k * π ≤ x ∧ x < k * π + π / 4) :=
sorry

end strictly_increasing_interval_l650_650308


namespace cone_height_l650_650402

theorem cone_height (r h : ℝ) (π : ℝ) (Hπ : Real.pi = π) (slant_height : ℝ) (lateral_area : ℝ) (base_area : ℝ) 
  (H1 : slant_height = 2) 
  (H2 : lateral_area = 2 * π * r) 
  (H3 : base_area = π * r^2) 
  (H4 : lateral_area = 4 * base_area) 
  (H5 : r^2 + h^2 = slant_height^2) 
  : h = π / 2 := by 
sorry

end cone_height_l650_650402


namespace probability_of_disease_given_positive_test_l650_650011

-- Define the probabilities given in the problem
noncomputable def pr_D : ℝ := 1 / 1000
noncomputable def pr_Dc : ℝ := 1 - pr_D
noncomputable def pr_T_given_D : ℝ := 1
noncomputable def pr_T_given_Dc : ℝ := 0.05

-- Define the total probability of a positive test using the law of total probability
noncomputable def pr_T := 
  pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Using Bayes' theorem
noncomputable def pr_D_given_T := 
  pr_T_given_D * pr_D / pr_T

-- Theorem to prove the desired probability
theorem probability_of_disease_given_positive_test : 
  pr_D_given_T = 1 / 10 :=
by
  sorry

end probability_of_disease_given_positive_test_l650_650011


namespace largest_prime_factor_of_1729_l650_650593

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l650_650593


namespace f_inverse_of_4_f_of_x_equals_3_l650_650310

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2 * x

theorem f_inverse_of_4 :
  f (1 / f 2) = 1 / 16 := by
  sorry

theorem f_of_x_equals_3 (x : ℝ) :
  f x = 3 → x = Real.sqrt 3 := by
  sorry

end f_inverse_of_4_f_of_x_equals_3_l650_650310


namespace factors_of_180_l650_650369

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l650_650369


namespace det_A_pow_4_l650_650009

variable {A : Matrix (Fin n) (Fin n) ℝ} (h : det A = 7)

theorem det_A_pow_4 : det (A^4) = 2401 :=
by
  sorry

end det_A_pow_4_l650_650009


namespace find_first_term_and_common_diff_l650_650166

def is_arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def cos_sequence_is_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, (a n * Real.cos (a n)) * ((n:ℝ)) + d = (a (n+1) * Real.cos (a(n+1)))

noncomputable def possible_values (a : ℕ → ℝ) (d : ℝ) (a₁ : ℝ → Prop) (d₁ : ℝ → Prop) : Prop :=
  (a₁ = (λ m : ℤ, - Real.pi / 6 + 2 * Real.pi * m)) ∧ 
  (d₁ = (λ k : ℤ, 2 * Real.pi * k)) ∧
  k ≠ 0 ∧ m ∈ ℤ

theorem find_first_term_and_common_diff :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  is_arithmetic_progression a d →
  cos_sequence_is_arithmetic a d →
  (∀ n : ℕ, Real.sin (2 * a n) + Real.cos (a (n + 1)) = 0) →
  (∃ a₁ d₁ (m k : ℤ), 
    possible_values (a) (d) (a₁ m) (d₁ k)) :=
by
  intros a d h1 h2 h3
  sorry

end find_first_term_and_common_diff_l650_650166


namespace area_of_overlap_is_one_l650_650108

-- Definitions
def point := (ℝ × ℝ)
def grid := {p : point | (∃ i j : ℕ, i ≤ 2 ∧ j ≤ 2 ∧ p = (i, j))}
def square := {p : point | p = (0,0) ∨ p = (0,2) ∨ p = (2,2) ∨ p = (2,0)}
def triangle := {p : point | p = (2,2) ∨ p = (0,1) ∨ p = (1,0)}

-- Theorem to prove that the area of the overlap is 1 square unit
theorem area_of_overlap_is_one : 
  let overlap := {p ∈ grid | p ∈ square ∧ p ∈ triangle} in
  let base := 1 in
  let height := 1 in
  area_overlap overlap = 1 := 
sorry

end area_of_overlap_is_one_l650_650108


namespace find_weights_l650_650978

theorem find_weights (x y z : ℕ) (h1 : x + y + z = 11) (h2 : 3 * x + 7 * y + 14 * z = 108) :
  x = 1 ∧ y = 5 ∧ z = 5 :=
by
  sorry

end find_weights_l650_650978


namespace evaluate_expression_l650_650201

theorem evaluate_expression : (10 ^ (-3) * 5 ^ (-2)) / (10 ^ (-5)) = 4 := by
  sorry

end evaluate_expression_l650_650201


namespace area_at_stage_8_l650_650384

theorem area_at_stage_8 
  (side_length : ℕ)
  (stage : ℕ)
  (num_squares : ℕ)
  (square_area : ℕ) 
  (total_area : ℕ) 
  (h1 : side_length = 4) 
  (h2 : stage = 8) 
  (h3 : num_squares = stage) 
  (h4 : square_area = side_length * side_length) 
  (h5 : total_area = num_squares * square_area) :
  total_area = 128 :=
sorry

end area_at_stage_8_l650_650384


namespace num_100_digit_even_numbers_l650_650334

theorem num_100_digit_even_numbers : 
  let digit_set := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let valid_number (digits : list ℕ) := 
    digits.length = 100 ∧ digits.head ∈ {1, 3} ∧ 
    digits.last ∈ {0} ∧ 
    ∀ d ∈ digits.tail.init, d ∈ digit_set
  (∃ (m : ℕ), valid_number (m.digits 10)) = 2 * 3^98 := 
sorry

end num_100_digit_even_numbers_l650_650334


namespace days_before_reinforcement_l650_650107

theorem days_before_reinforcement
    (garrison_1 : ℕ)
    (initial_days : ℕ)
    (reinforcement : ℕ)
    (additional_days : ℕ)
    (total_men_after_reinforcement : ℕ)
    (man_days_initial : ℕ)
    (man_days_after : ℕ)
    (x : ℕ) :
    garrison_1 * (initial_days - x) = total_men_after_reinforcement * additional_days →
    garrison_1 = 2000 →
    initial_days = 54 →
    reinforcement = 1600 →
    additional_days = 20 →
    total_men_after_reinforcement = garrison_1 + reinforcement →
    man_days_initial = garrison_1 * initial_days →
    man_days_after = total_men_after_reinforcement * additional_days →
    x = 18 :=
by
  intros h_eq g_1 i_days r_f a_days total_men m_days_i m_days_a
  sorry

end days_before_reinforcement_l650_650107


namespace largest_prime_factor_1729_l650_650533

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650533


namespace largest_prime_factor_of_1729_is_19_l650_650610

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650610


namespace factorization_l650_650743

theorem factorization (x y : ℝ) : (x + y)^2 - 14 * (x + y) + 49 = (x + y - 7)^2 :=
by
  sorry

end factorization_l650_650743


namespace complement_of_60_is_30_l650_650392

noncomputable def complement (angle : ℝ) : ℝ := 90 - angle

theorem complement_of_60_is_30 : complement 60 = 30 :=
by 
  sorry

end complement_of_60_is_30_l650_650392


namespace pyramid_volume_l650_650982

-- Lean 4 statement corresponding to the mathematically equivalent proof problem
theorem pyramid_volume (AB BC PA : ℝ) (h_AB : AB = 10) (h_BC : BC = 6) (h_PA : PA = 8) 
  (perp_PA_AB : ∀ P A B : ℝ, ∃ PA, PA = 8 ∧ PA ⊥ AB)
  (perp_PA_AC : ∀ P A C : ℝ, ∃ PA, PA = 8 ∧ PA ⊥ AC) :
  let base_area := (1 / 2) * AB * BC in
  let volume := (1 / 3) * base_area * PA in
  volume = 80 :=
by 
  sorry

end pyramid_volume_l650_650982


namespace geometric_sequence_properties_l650_650300

variable {S : ℕ → ℝ}
variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ (n : ℕ), a (n+1) = a n * r

noncomputable def is_arithmetic_sequence (b : ℕ → ℝ) :=
  ∀ (n : ℕ), b (n+1) - b n = b (n+2) - b (n+1)

noncomputable def Sn (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), S n = ∑ i in finset.range(n+1), a i

noncomputable def maximum_minimum_Sn (S : ℕ → ℝ) (max_val min_val : ℝ) : Prop :=
  (∀ (n : ℕ), S n ≤ max_val) ∧
  (∀ (n : ℕ), S n ≥ min_val)

theorem geometric_sequence_properties :
  (a 1 = 3/2) →
  (∀ (n : ℕ) , S n = ∑ i in finset.range(n+1), a i) →
  (is_arithmetic_sequence (λ n, if n = 0 then -2 * S 2 else if n = 1 then S 3 else 4 * S 4)) →
  (∀ n, a n = (-1)^(n-1) * (3 / 2^n)) ∧
  (maximum_minimum_Sn S (3/2) (3/4)) :=
by
  intros h1 h2 h3
  sorry

end geometric_sequence_properties_l650_650300


namespace largest_prime_factor_1729_l650_650541

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650541


namespace M_inter_N_eq_l650_650811

open Set

def M : Set ℝ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 < n ∧ n ≤ 3 }

theorem M_inter_N_eq : M ∩ (coe '' N) = {0, 1} :=
by sorry

end M_inter_N_eq_l650_650811


namespace largest_prime_factor_1729_l650_650543

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l650_650543


namespace minimal_clicks_to_uniform_color_l650_650895

-- Define the problem conditions
def toggle_rectangle (board : list (list bool)) (rect : ℕ × ℕ × ℕ × ℕ) : list (list bool) :=
  sorry  -- Function to toggle colors inside a given rectangle (left undefined for now)

-- Main proof statement
theorem minimal_clicks_to_uniform_color (n : ℕ) (hn : even n): 
  ∃ (k : ℕ), (∀ board : list (list bool), list (list bool) → bool) 
                (toggle_rectangle : list (list bool) → ℕ × ℕ × ℕ × ℕ → list (list bool)),
                k = n ∧ 
                (∀ final_board : list (list bool), 
                  (∀ i j, (i < n ∧ j < n → final_board[i][j] = final_board[0][0])) 
                  → (∀ board, ∃ toggles, 
                      (∀ t ∈ toggles, 
                        toggle_rectangle board t = final_board) 
                      ∧ length toggles = k)
                 )

end minimal_clicks_to_uniform_color_l650_650895


namespace angle_between_vectors_l650_650282

variables (a b e : V)
variables (V : Type) [inner_product_space ℝ V]

-- Given conditions as premises
variables (ha : ∥a∥ = 3)
variables (hb : ∥b∥ = 3)
variables (he : ∥e∥ = 1)
variables (hbe : b = 3 • e)
variables (proj_ab : (inner_product a b / ∥b∥^2) • b = (3/2) • e)

-- Goal: Prove that the angle between a and b is π/3
theorem angle_between_vectors (ha : ∥a∥ = 3) (hb : ∥b∥ = 3) (he : ∥e∥ = 1)
  (hbe : b = 3 • e) (proj_ab : (inner_product a b / ∥b∥^2) • b = (3/2) • e) :
  real.angle_of a b = real.angle_of_with_values 3 3 (1/2) :=
sorry

end angle_between_vectors_l650_650282


namespace frog_arrangement_count_l650_650052

theorem frog_arrangement_count :
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let frogs := green_frogs + red_frogs + blue_frogs
  -- Descriptions:
  -- 1. green_frogs refuse to sit next to red_frogs
  -- 2. green_frogs and red_frogs are fine sitting next to blue_frogs
  -- 3. blue_frogs can sit next to each other
  frogs = 7 → 
  ∃ arrangements : ℕ, arrangements = 72 :=
by 
  sorry

end frog_arrangement_count_l650_650052


namespace locus_equation_exists_lambda_l650_650668

theorem locus_equation (x y : ℝ) 
  (h : (Real.norm (x + 2, y) - Real.norm (x - 2, y) = 2)) : 
  x^2 - y^2 / 3 = 1 := sorry

theorem exists_lambda (P : ℝ × ℝ) 
  (hP : P.1 ^ 2 - P.2 ^ 2 / 3 = 1)
  (hP_x_pos : P.1 > 0) :
  ∃ λ > 0, λ = 2 ∧ 
    ∀ A F : ℝ × ℝ, A = (-1, 0) ∧ F = (2, 0) →
    ∃ α β : ℝ, λ = α / β ∧ α ≠ 0 ∧ β ≠ 0 ∧
    ∠ P F A = α ∠ P A F :=
sorry

end locus_equation_exists_lambda_l650_650668


namespace number_of_factors_180_l650_650352

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.eraseDuplicates.map (λ p, n.factors.count p + 1)).foldr (· * ·) 1

theorem number_of_factors_180 : number_of_factors 180 = 18 := by
  sorry

end number_of_factors_180_l650_650352


namespace parallel_lines_condition_l650_650944

-- Definitions of the lines
def line1 (a : ℝ) : ℝ × ℝ → Prop := λ p, p.1 + a * p.2 = 1
def line2 (a : ℝ) : ℝ × ℝ → Prop := λ p, a * p.1 + p.2 = 5

-- Proof that a = -1 is sufficient but not necessary for the lines to be parallel
theorem parallel_lines_condition (a : ℝ) :
  (∀ p q : ℝ × ℝ, line1 a p → line1 a q → line2 a p → line2 a q → a = -1) ∧
  (∃ b : ℝ, b ≠ -1 ∧ ∀ p q : ℝ × ℝ, line1 b p → line1 b q → line2 b p → line2 b q → ¬(a = -1)) :=
by
  sorry

end parallel_lines_condition_l650_650944


namespace sum_of_integers_from_neg15_to_5_l650_650713

-- Define the problem with conditions
theorem sum_of_integers_from_neg15_to_5 :
  (∑ i in Finset.Icc (-15) 5, i) = -105 :=
by
  sorry

end sum_of_integers_from_neg15_to_5_l650_650713


namespace hyperbolas_same_asymptotes_l650_650185

theorem hyperbolas_same_asymptotes :
  ∃ (M : ℝ), M = 4.5 ∧
    (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = (4 / 3) * x ∨ y = -(4 / 3) * x) ∧
               (y^2 / 8 - x^2 / M = 1 → y = sqrt (8 / M) * x ∨ y = -sqrt (8 / M) * x) :=
begin
  existsi 4.5,
  split,
  { refl },
  { intros x y,
    split,
    { intro h,
      sorry },
    { intro h,
      sorry }
  }
end

end hyperbolas_same_asymptotes_l650_650185


namespace sum_digits_18_to_21_l650_650043

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_18_to_21 :
  sum_of_digits 18 + sum_of_digits 19 + sum_of_digits 20 + sum_of_digits 21 = 24 :=
by
  sorry

end sum_digits_18_to_21_l650_650043


namespace volume_frustum_pyramid_l650_650105

-- Define the given conditions and problem statement
variables (R β : ℝ)
-- Angle between lateral edge and the plane of the base
variable (β_angle : 0 < β ∧ β < π / 2)

-- Define the volume function based on the provided formula
def volume_frustum (R β : ℝ) : ℝ :=
  (2 / 3) * R^3 * (Real.sin (2 * β)) * (1 + (Real.cos (2 * β)) ^ 2 - Real.cos (2 * β))

-- The theorem stating the volume of the frustum based on given conditions
theorem volume_frustum_pyramid :
  volume_frustum R β = (2 / 3) * R^3 * (Real.sin (2 * β)) * (1 + (Real.cos (2 * β)) ^ 2 - Real.cos (2 * β)) :=
  sorry

end volume_frustum_pyramid_l650_650105


namespace solution_expression_for_f_l650_650792

-- Given an odd function f, and its definition for x > 0, prove its expression for x < 0.
variable (f : ℝ → ℝ)
variable odd_f : ∀ x, f (-x) = -f x
variable pos_def : ∀ x : ℝ, 0 < x → f x = 3 * x + 5

theorem solution_expression_for_f (x : ℝ) (hx : x < 0) : f x = 3 * x - 5 := by
  sorry

end solution_expression_for_f_l650_650792


namespace solve_abs_eq_l650_650194

theorem solve_abs_eq (x : ℝ) : 
    (3 * x + 9 = abs (-20 + 4 * x)) ↔ 
    (x = 29) ∨ (x = 11 / 7) := 
by sorry

end solve_abs_eq_l650_650194


namespace find_value_of_a_l650_650807

noncomputable def f (x a : ℝ) : ℝ := (3 * Real.log x - x^2 - a - 2)^2 + (x - a)^2

theorem find_value_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ 8) ↔ a = -1 :=
by
  sorry

end find_value_of_a_l650_650807


namespace odd_function_domain_real_l650_650868

theorem odd_function_domain_real
  (a : ℤ)
  (h_condition : a = -1 ∨ a = 1 ∨ a = 3) :
  (∀ x : ℝ, ∃ y : ℝ, x ≠ 0 → y = x^a) →
  (∀ x : ℝ, x ≠ 0 → (x^a = (-x)^a)) →
  (a = 1 ∨ a = 3) :=
sorry

end odd_function_domain_real_l650_650868


namespace count_100_digit_even_numbers_l650_650340

theorem count_100_digit_even_numbers : 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  in
  count_valid_numbers = 2 * 3 ^ 98 :=
by 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  have : count_valid_numbers = 2 * 3 ^ 98 := by sorry
  exact this

end count_100_digit_even_numbers_l650_650340


namespace negation_of_proposition_l650_650030

noncomputable def irrational_has_irrational_square : Prop :=
  ∀ (x : ℝ), ¬(x ∈ ℚ) → ¬(x^2 ∈ ℚ)

theorem negation_of_proposition :
  (¬ ∃ (x : ℝ), ¬(x ∈ ℚ) ∧ (x^2 ∈ ℚ)) ↔ irrational_has_irrational_square :=
by
  sorry

end negation_of_proposition_l650_650030


namespace average_time_per_leg_l650_650086

-- Conditions
def time_y : ℕ := 58
def time_z : ℕ := 26
def total_time : ℕ := time_y + time_z
def number_of_legs : ℕ := 2

-- Theorem stating the average time per leg
theorem average_time_per_leg : total_time / number_of_legs = 42 := by
  sorry

end average_time_per_leg_l650_650086


namespace no_finite_centers_of_symmetry_l650_650176

-- We define what it means for a figure to have a center of symmetry.
-- Assume that having more than one implies an infinite number.
def has_center_of_symmetry {F : Type} (figure : F) (center : F → F) : Prop :=
  ∀ x, (center (center x)) = x  -- A point symmetric with respect to the center remains the same after double symmetry.

-- The main theorem statement.
theorem no_finite_centers_of_symmetry {F : Type} (figure : F) (center : F → F) :
  (∃ O1 O2 : F, O1 ≠ O2 ∧ has_center_of_symmetry figure center O1 ∧ has_center_of_symmetry figure center O2) →
  ∀ n : ℕ, ¬ (∃ L : list F, L.length = n ∧ ∀ O ∈ L, has_center_of_symmetry figure center O) :=
sorry

end no_finite_centers_of_symmetry_l650_650176


namespace group_of_2008_is_37_l650_650468

theorem group_of_2008_is_37 :
  let group_start := λ (n : ℕ), 1 + 3 * (n * (n - 1) / 2)
  let group_num_terms := λ (n : ℕ), n
  ∃ n : ℕ, 
    (group_start (n +1) - 3 ≤ 2008) ∧ (2008 < group_start (n +2) - 3) → n = 37 :=
sorry

end group_of_2008_is_37_l650_650468


namespace exists_bijective_function_with_property_l650_650483

theorem exists_bijective_function_with_property :
  ∃ f : ℕ → ℕ, function.bijective f ∧ (∀ m n : ℕ, f (3 * m * n + m + n) = 4 * f m * f n + f m + f n) :=
sorry

end exists_bijective_function_with_property_l650_650483


namespace largest_prime_factor_of_1729_l650_650643

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l650_650643


namespace math_problem_l650_650704

variable (n : ℕ) (p : ℝ)

-- Conditions
def condition_1 : Prop := (0 : ℝ) < 0.65 ∧ 0.65 < 1 ∧ n > 0
def condition_2 : Prop := (0 : ℝ) < p ∧ p < 1
def condition_3 : Prop := (0 : ℝ) < p ∧ p < 1 ∧ n > 0
def condition_4 : Prop := (0 : ℝ) < 0.6 ∧ 0.6 < 1 ∧ 50 > 0

-- Question
def question : Prop := 
  let X_1 : Type := ℕ in
  let X_2 : Type := ℕ in
  let X_3 : Type := ℕ in
  let X_4 : Type := ℕ in
  -- Only X_2 is NOT binomial
  (¬ binomial_distribution X_2) ∧ 
  binomial_distribution X_1 ∧ binomial_distribution X_3 ∧ binomial_distribution X_4

-- Correct answer
theorem math_problem (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) : question :=
sorry

end math_problem_l650_650704


namespace closest_integer_to_a2011_l650_650695

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 3 → 6 * a n + 5 * a (n - 2) = 20 + 11 * a (n - 1)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 0 ∧ a 2 = 1

theorem closest_integer_to_a2011 (a : ℕ → ℝ) (h_seq : sequence a) (h_init : initial_conditions a) :
  abs (a 2011 - 40086) < 1 :=
sorry

end closest_integer_to_a2011_l650_650695


namespace expected_first_sequence_grooms_l650_650136

-- Define the harmonic number function
def harmonic (n : ℕ) : ℝ :=
  ∑ k in finset.range n, 1 / (k + 1 : ℝ)

-- Define the expected number of grooms in the first sequence
theorem expected_first_sequence_grooms :
  harmonic 100 = 5.187 :=
by
  sorry

end expected_first_sequence_grooms_l650_650136


namespace log_proof_l650_650174

-- Define the logarithm function
def lg : ℝ → ℝ := sorry

-- Define the condition (logarithmic operation rule)
axiom log_rule (a b : ℝ) : lg a + lg b = lg (a * b)

-- Prove that lg 2 + lg 5 = lg 10
theorem log_proof : lg 2 + lg 5 = lg 10 :=
by
  apply log_rule

end log_proof_l650_650174


namespace sally_earnings_l650_650924

-- Definitions based on the conditions
def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20
def total_money : ℝ := total_seashells * price_per_seashell

-- Lean 4 statement to prove the total amount of money is $54
theorem sally_earnings : total_money = 54 := by
  -- Proof will go here
  sorry

end sally_earnings_l650_650924


namespace largest_prime_factor_1729_l650_650530

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650530


namespace not_enough_shots_to_guarantee_damage_l650_650167

theorem not_enough_shots_to_guarantee_damage : 
  let points := 29
  let total_triangles := Nat.choose points 3
  let triangles_per_shot := points - 2
  let max_shots := 134
  let max_damaged_triangles := max_shots * triangles_per_shot
  total_triangles > max_damaged_triangles := 
by
  let points := 29
  let total_triangles := Nat.choose points 3
  let triangles_per_shot := points - 2
  let max_shots := 134
  let max_damaged_triangles := max_shots * triangles_per_shot
  have h : total_triangles = Nat.choose points 3 := rfl
  have h1 : triangles_per_shot = points - 2 := rfl
  have h2 : max_damaged_triangles = max_shots * triangles_per_shot := rfl
  show total_triangles > max_damaged_triangles
  rw h
  rw h1
  rw h2
  simp
  sorry

end not_enough_shots_to_guarantee_damage_l650_650167


namespace total_distance_walked_l650_650482

-- Define the conditions
def walking_rate : ℝ := 4
def time_before_break : ℝ := 2
def time_after_break : ℝ := 0.5

-- Define the required theorem
theorem total_distance_walked : 
  walking_rate * time_before_break + walking_rate * time_after_break = 10 := 
sorry

end total_distance_walked_l650_650482


namespace expected_first_sequence_100_l650_650144

noncomputable def expected_first_sequence (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1).filter (λ k, k > 0), (1 : ℝ) / k

theorem expected_first_sequence_100 : expected_first_sequence 100 = 
    ∑ k in (finset.range 101).filter (λ k, k > 0), (1 : ℝ) / k :=
by
  -- The proof would involve showing this sum represents the harmonic number H_100
  sorry

end expected_first_sequence_100_l650_650144


namespace pentagonal_faces_count_l650_650002

theorem pentagonal_faces_count (x y : ℕ) (h : (5 * x + 6 * y) % 6 = 0) (h1 : ∃ v e f, v - e + f = 2 ∧ f = x + y ∧ e = (5 * x + 6 * y) / 2 ∧ v = (5 * x + 6 * y) / 3 ∧ (5 * x + 6 * y) / 3 * 3 = 5 * x + 6 * y) : 
  x = 12 :=
sorry

end pentagonal_faces_count_l650_650002


namespace range_of_m_l650_650277

theorem range_of_m (f g : ℝ → ℝ) (h1 : ∃ m : ℝ, ∀ x : ℝ, f x = m * (x - m) * (x + m + 3))
  (h2 : ∀ x : ℝ, g x = 2 ^ x - 4)
  (h3 : ∀ x : ℝ, f x < 0 ∨ g x < 0) :
  ∃ m : ℝ, -5 < m ∧ m < 0 :=
sorry

end range_of_m_l650_650277


namespace b_is_int_iff_a_is_half_n_n_sq_plus_3_l650_650874

theorem b_is_int_iff_a_is_half_n_n_sq_plus_3 (a : ℝ) (n : ℕ) (hp_a : 0 < a) (h_b_def : 
  let b := (a + real.sqrt(a^2 + 1))^(1/3) + (a - real.sqrt(a^2 + 1))^(1/3) in b ∈ ℤ) :
  ∃ (n : ℕ+), a = (1/2) * n * (n^2 + 3) := sorry

end b_is_int_iff_a_is_half_n_n_sq_plus_3_l650_650874


namespace product_inequality_l650_650479

theorem product_inequality (n : ℕ) (x : Fin n → ℝ) (h_pos : ∀ i, 0 < x i) (h_prod : (∏ i, x i) = 1) :
  (∏ i, 1 + x i) ≥ 2 ^ n :=
sorry

end product_inequality_l650_650479


namespace number_of_factors_180_l650_650351

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.eraseDuplicates.map (λ p, n.factors.count p + 1)).foldr (· * ·) 1

theorem number_of_factors_180 : number_of_factors 180 = 18 := by
  sorry

end number_of_factors_180_l650_650351


namespace elongation_improvement_l650_650679

variables (x y : Fin 10 → ℝ)

-- Define the differences
def z (i : Fin 10) : ℝ := x i - y i

-- Calculate the mean of z
def mean_z : ℝ := (∑ i, z x y i) / 10

-- Calculate the sample variance of z
def sample_variance_z : ℝ :=
(∑ i, (z x y i - mean_z x y)^2) / 10

theorem elongation_improvement
  (hx : x = ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548])
  (hy : y = ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]) :
  mean_z x y = 11 ∧ sample_variance_z x y = 61 ∧ (mean_z x y ≥ 2 * (sqrt (sample_variance_z x y / 10))) := by
  -- Placeholder for the proof.
  sorry

end elongation_improvement_l650_650679


namespace tan_zero_l650_650284

theorem tan_zero (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ real.pi)
    (h2 : 3 * real.sin (x / 2) = real.sqrt (1 + real.sin x) - real.sqrt (1 - real.sin x)) : 
    real.tan x = 0 := 
by
  sorry

end tan_zero_l650_650284


namespace inequality_of_fractions_l650_650772

theorem inequality_of_fractions
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (c d : ℝ) (h3 : c < d) (h4 : d < 0)
  (e : ℝ) (h5 : e < 0) :
  (e / ((a - c)^2)) > (e / ((b - d)^2)) :=
by
  sorry

end inequality_of_fractions_l650_650772


namespace product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l650_650967

theorem product_of_three_divisors_of_5_pow_4_eq_5_pow_4 (a b c : ℕ) (h1 : a * b * c = 5^4) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : a ≠ c) : a + b + c = 131 :=
sorry

end product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l650_650967


namespace two_bisecting_lines_through_A_l650_650058

-- Definitions for point A, line l, and circle S1
variable (A : Point)
variable (l : Line)
variable (S1 : Circle)

-- Assuming bisect_segment_line_condition holds for bisecting segment condition
noncomputable def bisect_segment_line_condition (line : Line) : Prop :=
  ∃ P1 P2 : Point, (P1 ∈ (line ∩ S1)) ∧ (P2 ∈ (line ∩ l)) ∧ 
  (A ∈ segment P1 P2) ∧ (dist P1 A = dist A P2)

-- Theorem stating there are exactly two such lines
theorem two_bisecting_lines_through_A :
  ∃! line1 : Line, bisect_segment_line_condition A l S1 line1 ∧
  ∃! line2 : Line, bisect_segment_line_condition A l S1 line2 :=
sorry

end two_bisecting_lines_through_A_l650_650058


namespace seating_profession_solution_l650_650260

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l650_650260


namespace eight_digit_numbers_count_l650_650815

theorem eight_digit_numbers_count :
  let first_digit_choices := 9
  let remaining_digits_choices := 10 ^ 7
  9 * 10^7 = 90000000 :=
by
  sorry

end eight_digit_numbers_count_l650_650815


namespace max_value_l650_650866

variables {a b c : ℝ^3}

-- Conditions
def norm_a : ∥a∥ = 2 := sorry
def norm_b : ∥b∥ = 3 := sorry
def norm_c : ∥c∥ = 4 := sorry
def a_dot_b : a ⬝ b = 6 := sorry

-- Problem Statement
theorem max_value :
  (∥a - 3 • b∥^2 + ∥b - 3 • c∥^2 + ∥c - 3 • a∥^2) ≤ 134 :=
sorry

end max_value_l650_650866


namespace sin_minus_cos_eq_l650_650291

variable {α : ℝ} (h₁ : 0 < α ∧ α < π) (h₂ : Real.sin α + Real.cos α = 1/3)

theorem sin_minus_cos_eq : Real.sin α - Real.cos α = Real.sqrt 17 / 3 :=
by 
  -- Proof goes here
  sorry

end sin_minus_cos_eq_l650_650291


namespace largest_prime_factor_1729_l650_650620

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l650_650620


namespace max_statements_true_l650_650450

variable (x : ℝ)

def condition1 : Prop := 0 < x^2 ∧ x^2 < 2
def condition2 : Prop := x^2 > 2
def condition3 : Prop := -1 < x ∧ x < 0
def condition4 : Prop := 0 < x ∧ x < 2
def condition5 : Prop := 0 < x^3 - x^2 ∧ x^3 - x^2 <  2

noncomputable def maxTrueStatements : ℝ → ℕ
| x := 
  if condition1 x 
  then if condition4 x
    then if 0 < x ∧ (x^3 - x^2) < 2 
      then 3  -- Statements 1, 4, 5
      else 2
  else if condition2 x
    then if condition3 x
      then if -1 < x ∧ 0 < x ∧ x < 2
        then 3
        else 2
else if condition3 x
    then if 0 < x ∧ x < 2 
      then 3 
      else 2
    else if condition4 x ∧ x >2 
        then maxTrueStatements x
        else nat.succ 2

theorem max_statements_true : ∃ x, maxTrueStatements x = 3 :=
  sorry

end max_statements_true_l650_650450


namespace largest_prime_factor_1729_l650_650526

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l650_650526


namespace simplify_expression_l650_650446

theorem simplify_expression (a b c d x k : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (p x = ((x + a)^4 / (a - b) / (a - c) / (a - d)) + ((x + b)^4 / (b - a) / (b - c) / (b - d)) + ((x + c)^4 / (c - a) / (c - b) / (c - d)) + ((x + d)^4 / (d - a) / (d - b) / (d - c))) →
  ∃ k, p x = k * (x + a) * (x + b) * (x + c) * (x + d) := by
  sorry

end simplify_expression_l650_650446


namespace find_geometric_sequence_term_l650_650851

noncomputable def geometric_sequence_term (a q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

theorem find_geometric_sequence_term (a : ℝ) (q : ℝ)
  (h1 : a * (1 - q ^ 3) / (1 - q) = 7)
  (h2 : a * (1 - q ^ 6) / (1 - q) = 63) :
  ∀ n : ℕ, geometric_sequence_term a q n = 2^(n-1) :=
by
  sorry

end find_geometric_sequence_term_l650_650851


namespace part1_part2_l650_650318

-- Definition of the function
def f (x : ℝ) (m : ℝ) := Real.exp x - m * x - 1

-- Theorem statement for part 1
theorem part1 (m : ℝ) (h : m > 0) (tangent_zero : ∀ x, (x = 0) → f x m = 0) : m = 1 :=
by sorry

-- Theorem statement for part 2
theorem part2 (m : ℝ) (h : m > 0) :
  (∀ x, x < Real.log m → deriv (λ x, f x m) x < 0) ∧
  (∀ x, x > Real.log m → deriv (λ x, f x m) x > 0) ∧
  (deriv (λ x, f x m) (Real.log m) = 0) ∧
  (∀ x, x = Real.log m → f x m = m - m * Real.log m - 1) :=
by sorry

end part1_part2_l650_650318


namespace seating_profession_solution_l650_650264

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l650_650264


namespace total_art_cost_l650_650431

noncomputable def art_total_cost : ℝ := 45000

def cost_per_piece_country_a : ℝ := art_total_cost / 3

def cost_per_piece_country_b (x : ℝ) : ℝ :=
  let multiplier := 1 + 0.25 in x * multiplier

def total_cost_country_b (x : ℝ) : ℝ :=
  let cost := cost_per_piece_country_b x in 2 * cost

def cost_per_piece_country_c (x : ℝ) : ℝ :=
  let multiplier := 1 + 0.50 in x * multiplier

def total_cost_country_c (x : ℝ) : ℝ := 
  let cost := cost_per_piece_country_c x in 3 * cost

def cost_piece_country_d (total_cost_c : ℝ) : ℝ := 2 * total_cost_c

theorem total_art_cost : (art_total_cost + total_cost_country_b cost_per_piece_country_a 
  + total_cost_country_c cost_per_piece_country_a + cost_piece_country_d (total_cost_country_c cost_per_piece_country_a)) = 285000 :=
by
  sorry

end total_art_cost_l650_650431


namespace solveProfessions_l650_650254

-- Define our types for Friends and Professions
inductive Friend | Andrey | Boris | Vyacheslav | Gennady
inductive Profession | Architect | Barista | Veterinarian | Guitarist

-- Define the seating arrangement and conditions
def seatingArrangement : Friend → Position := sorry
def profession : Friend → Profession := sorry

-- Conditions translated to Lean
axiom condition1 : ∀ (f : Friend), 
  profession f = Profession.Veterinarian → 
  ∃ (neighbor1 neighbor2 : Friend), 
    seatingArrangement neighbor1 = seatingArrangement f + 1 ∨ seatingArrangement neighbor1 = seatingArrangement f - 1 ∧ 
    seatingArrangement neighbor2 = seatingArrangement f + 1 ∨ seatingArrangement neighbor2 = seatingArrangement f - 1 ∧
    (profession neighbor1 = Profession.Architect ∧ profession neighbor2 = Profession.Guitarist) 

axiom condition2 : ∀ (f : Friend), 
  profession f = Profession.Barista → 
  ∃ (rightNeighbor : Friend), 
    seatingArrangement rightNeighbor = seatingArrangement f + 1 ∧ 
    rightNeighbor = Friend.Boris

axiom condition3 : seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Andrey ∧ seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Boris

axiom condition4 : ∀ (f : Friend), 
  f = Friend.Andrey → 
  ∃ (leftNeighbor rightNeighbor : Friend), 
    seatingArrangement leftNeighbor = seatingArrangement f - 1 ∧ 
    seatingArrangement rightNeighbor = seatingArrangement f + 1

axiom condition5 : ∀ (f1 f2 : Friend), 
  (profession f1 = Profession.Guitarist ∧ profession f2 = Profession.Barista) → 
  seatingArrangement f1 ≠ seatingArrangement f2 + 1 ∧ seatingArrangement f1 ≠ seatingArrangement f2 - 1

-- Define the positions
def Position : Type := Fin 4

-- The main theorem to prove the assignments
theorem solveProfessions :
  profession Friend.Gennady = Profession.Barista ∧ 
  profession Friend.Boris = Profession.Architect ∧ 
  profession Friend.Andrey = Profession.Veterinarian ∧ 
  profession Friend.Vyacheslav = Profession.Guitarist :=
sorry

end solveProfessions_l650_650254


namespace three_digit_number_cubed_sum_l650_650748

theorem three_digit_number_cubed_sum {n : ℕ} (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 100 * a + 10 * b + c ∧ n = a^3 + b^3 + c^3) ↔
  n = 153 ∨ n = 370 ∨ n = 371 ∨ n = 407 :=
by
  sorry

end three_digit_number_cubed_sum_l650_650748


namespace roots_of_equation_l650_650969

theorem roots_of_equation :
  ∀ x : ℝ, -x * (x + 3) = x * (x + 3) ↔ (x = 0 ∨ x = -3) :=
by
  intro x
  split
  . intro h
    have h1 : -x * (x + 3) + x * (x + 3) = 0 := by sorry
    have h2 : 2 * x * (x + 3) = 0 := by sorry
    cases' eq_zero_or_eq_zero_of_mul_eq_zero h2 with h3 h4
    . left
      exact eq_zero_of_two_mul_eq_zero h3
    . right
      exact eq_neg_of_add_eq_zero_left h4
  . intro h
    cases' h with h5 h6
    . rw [h5, zero_mul, mul_zero, neg_zero]
    . rw [h6, ← neg_add_cancel_left, neg_mul_eq_mul_neg]
      rw [mul_zero, neg_zero]
      rfl

end roots_of_equation_l650_650969


namespace sally_earnings_l650_650923

-- Definitions based on the conditions
def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20
def total_money : ℝ := total_seashells * price_per_seashell

-- Lean 4 statement to prove the total amount of money is $54
theorem sally_earnings : total_money = 54 := by
  -- Proof will go here
  sorry

end sally_earnings_l650_650923


namespace young_member_age_diff_l650_650667

-- Definitions
def A : ℝ := sorry    -- Average age of committee members 4 years ago
def O : ℝ := sorry    -- Age of the old member
def N : ℝ := sorry    -- Age of the new member

-- Hypotheses
axiom avg_same : ∀ (t : ℝ), t = t
axiom replacement : 10 * A + 4 * 10 - 40 = 10 * A

-- Theorem
theorem young_member_age_diff : O - N = 40 := by
  -- proof goes here
  sorry

end young_member_age_diff_l650_650667


namespace number_of_factors_180_l650_650354

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.eraseDuplicates.map (λ p, n.factors.count p + 1)).foldr (· * ·) 1

theorem number_of_factors_180 : number_of_factors 180 = 18 := by
  sorry

end number_of_factors_180_l650_650354


namespace largest_prime_factor_of_1729_is_19_l650_650609

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l650_650609


namespace complex_magnitude_sqrt5_l650_650821

theorem complex_magnitude_sqrt5 (a b : ℝ) (h : a/(1 - complex.I) = 1 - b * complex.I) : (complex.abs (a + b * complex.I)) = real.sqrt 5 := 
sorry

end complex_magnitude_sqrt5_l650_650821


namespace repeating_decimal_sum_l650_650649

theorem repeating_decimal_sum : 
  let x := 0.123123 in
  let frac := 41 / 333 in
  sum_n_d (frac) = 374 :=
begin
  sorry
end

end repeating_decimal_sum_l650_650649


namespace odd_black_cells_cross_exists_l650_650408

-- Define the grid and related properties
variable (m n : ℕ) (H_even_m : m % 2 = 0) (H_even_n : n % 2 = 0)
variable (grid : Fin m → Fin n → Prop) -- grid(i, j) is true if cell (i, j) is black
variable (H_black_exists : ∃ (i : Fin m) (j : Fin n), grid i j)

-- Define a function to count number of black cells in a row or column
def count_black_cells_row (r : Fin m) : ℕ := (Finset.univ.filter (λ j : Fin n, grid r j)).card
def count_black_cells_col (c : Fin n) : ℕ := (Finset.univ.filter (λ i : Fin m, grid i c)).card
def count_black_cells_cross (r : Fin m) (c : Fin n) : ℕ :=
  count_black_cells_row grid r + count_black_cells_col grid c 
  - if grid r c then 1 else 0

-- Prove that there exists a cross with an odd number of black cells
theorem odd_black_cells_cross_exists : 
  ∃ r : Fin m, ∃ c : Fin n, count_black_cells_cross grid r c % 2 = 1 := 
sorry

end odd_black_cells_cross_exists_l650_650408
